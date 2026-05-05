"""
DETR Training with Adaptive Sampler
Automatically adjusts class distribution based on per-class validation performance
"""

import torch
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datetime import datetime  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modeling.detr.detr_with_existing_pipeline import DETRWithExistingDataPipeline


from modeling.detr.detr_evaluation import DETREvaluator


class DETRTrainerWithAdaptiveSampling(DETRWithExistingDataPipeline):
    """DETR trainer with adaptive class sampling"""

    def __init__(self, config):
        super().__init__(config)

        if config['data'].get('use_adaptive_sampler', False):
            self._setup_adaptive_sampler(config)

        self.eval_every_n_epochs = config['training'].get('eval_every_n_epochs', 5)
        self.save_best_model = config['training'].get('save_best_model', True)
        self.save_ckpt_every = config['training'].get('save_checkpoint_every', 5)
        self.patience = config['training'].get('patience', 10)

        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.evaluator = DETREvaluator(
            model=self.model, data_loader=self.val_loader, processor=self.processor, device=self.device, config=config,
        )
        self.test_evaluator = DETREvaluator(
            model=self.model, data_loader=self.test_loader, processor=self.processor, device=self.device, config=config,
        )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        data_cfg = config.get('data', {})
        
        if data_cfg.get('use_adaptive_sampler', False):
            sampler_str = f"adapt_{data_cfg.get('initial_mode', 'equal')}"
        elif data_cfg.get('use_balanced_sampler', False):
            sampler_str = f"bal_{data_cfg.get('balance_mode', 'sqrt')}"
        else:
            sampler_str = "no_sampler"
            
        bg_str = "dynBG" if data_cfg.get('dynamic_background', False) else f"fixBG_{data_cfg.get('background_ratio', 0.5)}"
        log_dir = f"runs/detr_{sampler_str}_{bg_str}_{timestamp}"
        
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")

    def _setup_adaptive_sampler(self, config):
        from turbine_processing.sampler_adaptive import AdaptiveDETRSampler
        from torch.utils.data import DataLoader

        self.adaptive_sampler = AdaptiveDETRSampler(
            dataset=self.train_dataset,
            epoch_size=len(self.train_dataset),
            initial_mode=config['data'].get('initial_mode', 'equal'),
            adaptation_rate=config['data'].get('adaptation_rate', 0.3),
            min_weight=config['data'].get('min_weight', 0.1),
            max_weight=config['data'].get('max_weight', 5.0),
            background_ratio=config['data'].get('background_ratio', 0.5),
            dynamic_background=config['data'].get('dynamic_background', True),
            min_bg_ratio=config['data'].get('min_bg_ratio', 0.15),
            max_bg_ratio=config['data'].get('max_bg_ratio', 0.50),
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,  
            sampler=self.adaptive_sampler,
            num_workers=config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=config['training']['num_workers'] > 0,
            prefetch_factor=2 if config['training']['num_workers'] > 0 else None,
        )

    def _update_sampler_weights(self, metrics):
        if not hasattr(self, 'adaptive_sampler') or 'pr_data' not in metrics:
            return

        class_ap = {class_id: data['ap'] for class_id, data in metrics['pr_data'].items() if class_id > 0}
        self.adaptive_sampler.update_class_weights(class_ap, metrics.get('bg_accuracy', None))

        for class_id, weight in self.adaptive_sampler.get_current_weights().items():
            class_name = self.config['model']['class_names'][class_id] if class_id < len(self.config['model']['class_names']) else f"class_{class_id}"
            self.writer.add_scalar(f'Sampler_Weights/{class_name}', weight, self.current_epoch)
            
        if hasattr(self.adaptive_sampler, 'bg_ratio'):
            self.writer.add_scalar('Sampler_Weights/Background_Ratio', self.adaptive_sampler.bg_ratio, self.current_epoch)

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        valid_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']:3d} Train",
            leave=True, dynamic_ncols=True, unit='batch',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for batch_idx, (pixel_values, pixel_mask, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            pixel_mask = pixel_mask.to(self.device)
            
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            # =========================================================
            # INVINCIBLE FORWARD PASS
            # =========================================================
            try:
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
                loss = outputs.loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n⚠ Warning: NaN/Inf loss detected at batch {batch_idx}. Skipping update.")
                    self.optimizer.zero_grad() 
                    continue 

            except Exception as e:
                print(f"\n⚠ Forward pass crashed at batch {batch_idx} with error: {e}. Skipping.")
                self.optimizer.zero_grad()
                continue
            # =========================================================

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_loss += loss.item()
            valid_batches += 1
            avg_loss = total_loss / valid_batches
            
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch_loss': f'{loss.item():.4f}'}, refresh=False)

        avg_loss = total_loss / max(valid_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        valid_batches = 0

        pbar = tqdm(
            self.val_loader, desc=f"Epoch {self.current_epoch+1:3d}/{self.config['training']['num_epochs']:3d} Valid",
            leave=True, dynamic_ncols=True, unit='batch',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for batch_idx, (pixel_values, pixel_mask, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            pixel_mask = pixel_mask.to(self.device)
            
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            try:
                outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
                loss = outputs.loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
            except:
                continue

            total_loss += loss.item()
            valid_batches += 1
            avg_loss = total_loss / valid_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'batch_loss': f'{loss.item():.4f}'}, refresh=False)

        avg_loss = total_loss / max(valid_batches, 1)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        print("="*70)
        print("DETR Training with Adaptive Sampling")
        print("="*70)
        print(f"  Training samples   : {len(self.train_dataset)}")
        print(f"  Learning rate      : {self.config['training']['learning_rate']}")

        num_epochs = self.config['training']['num_epochs']
        start_epoch = 0

        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch  
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} / {num_epochs}")
            print(f"{'='*60}")
            
            if hasattr(self, 'adaptive_sampler') and epoch > 0:
                self.adaptive_sampler.print_current_distribution()

            train_loss = self.train_one_epoch(epoch)
            print(f"  Train loss : {train_loss:.4f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)

            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss = self.validate()
                print(f"  Val loss   : {val_loss:.4f}")
                self.writer.add_scalar('Loss/validation', val_loss, epoch)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"  ⚠ No improvement for {self.patience_counter}/{self.patience} evaluations")

                print(f"\n  Running detailed evaluation at epoch {epoch + 1}…")
                metrics, _, _ = self.evaluator.evaluate(epoch=epoch + 1)

                if 'mAP' in metrics:
                    self.writer.add_scalar('Metrics/mAP', metrics['mAP'], epoch)
                    if self.save_best_model and metrics['mAP'] > getattr(self, 'best_map', 0.0):
                        self.best_map = metrics['mAP']
                        best_path = self.config['training']['output_model_path'].replace('.pth', '_best.pth')
                        torch.save(self.model.state_dict(), best_path)
                        print(f"  🎉 New best mAP={self.best_map:.4f} → {best_path}")

                    self._update_sampler_weights(metrics)

                if self.patience_counter >= self.patience:
                    print(f"\n⏹  Early stopping at epoch {epoch + 1}.")
                    break

            if hasattr(self, 'scheduler'):
                self.scheduler.step()

        print("\n" + "="*60)
        print("Training Summary Complete")
        self.writer.close()
        return self.model


def main():
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # REMINDER: Check your learning rate in config.yaml! 
    # It MUST be 0.0001 (1e-4) or 0.0002 (2e-4) for Deformable DETR.

    trainer = DETRTrainerWithAdaptiveSampling(config)
    trainer.train()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
