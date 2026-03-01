"""
DETR Training with Adaptive Sampler
Automatically adjusts class distribution based on per-class validation performance
"""

import torch
import yaml
import os
from datetime import datetime  
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from detr_with_existing_pipeline import DETRWithExistingDataPipeline
from detr_evaluation import DETREvaluator


class DETRTrainerWithAdaptiveSampling(DETRWithExistingDataPipeline):
    """DETR trainer with adaptive class sampling"""

    def __init__(self, config):
        super().__init__(config)

        # Override the sampler if adaptive mode is enabled
        if config['data'].get('use_adaptive_sampler', False):
            self._setup_adaptive_sampler(config)

        # Evaluation settings
        self.eval_every_n_epochs = config['training'].get('eval_every_n_epochs', 5)
        self.save_best_model = config['training'].get('save_best_model', True)
        self.save_ckpt_every = config['training'].get('save_checkpoint_every', 5)
        self.patience = config['training'].get('patience', 10)

        # Early stopping state
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Evaluators
        self.evaluator = DETREvaluator(
            model=self.model,
            data_loader=self.val_loader,
            processor=self.processor,
            device=self.device,
            config=config,
        )
        
        self.test_evaluator = DETREvaluator(
            model=self.model,
            data_loader=self.test_loader,
            processor=self.processor,
            device=self.device,
            config=config,
        )

        # --- TIMESTAMPED TENSORBOARD LOGGING ---
        # Get current time: YYYY-MM-DD_HH-MM-SS
        # --- TIMESTAMPED TENSORBOARD LOGGING ---
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # 1. CREATE A DESCRIPTIVE FOLDER NAME
        data_cfg = config.get('data', {})
        
        # Figure out what sampler we are using
        if data_cfg.get('use_adaptive_sampler', False):
            sampler_str = f"adapt_{data_cfg.get('initial_mode', 'equal')}"
        elif data_cfg.get('use_balanced_sampler', False):
            sampler_str = f"bal_{data_cfg.get('balance_mode', 'sqrt')}"
        else:
            sampler_str = "no_sampler"
            
        # Figure out background mode
        bg_str = "dynBG" if data_cfg.get('dynamic_background', False) else f"fixBG_{data_cfg.get('background_ratio', 0.5)}"
        
        # Combine them into the folder name
        # Example: runs/detr_adapt_equal_dynBG_2024-02-18_14-30-00
        log_dir = f"runs/detr_{sampler_str}_{bg_str}_{timestamp}"
        
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 TensorBoard logging to: {log_dir}")
        print(f"   (Run 'tensorboard --logdir=runs' to view)")

        # 2. LOG EXACT SETTINGS TO TENSORBOARD'S "TEXT" TAB
        # We format it as Markdown so it looks clean in the dashboard
        config_md = f"""
        ### Adaptive Sampler Settings
        * **Mode:** `{data_cfg.get('initial_mode')}`
        * **Adaptation Rate:** `{data_cfg.get('adaptation_rate')}`
        * **Weight Bounds:** `[{data_cfg.get('min_weight')}, {data_cfg.get('max_weight')}]`

        ### Background Settings
        * **Dynamic Background:** `{data_cfg.get('dynamic_background')}`
        * **Initial Ratio:** `{data_cfg.get('background_ratio')}`
        * **Ratio Bounds:** `[{data_cfg.get('min_bg_ratio')}, {data_cfg.get('max_bg_ratio')}]`

        ### Training Settings
        * **Batch Size:** `{config['training'].get('batch_size')}`
        * **Learning Rate:** `{config['training'].get('learning_rate')}`
        * **Epochs:** `{config['training'].get('num_epochs')}`
                """
        
        # Write to global step 0
        self.writer.add_text("Experiment_Configuration", config_md, 0)

    def _setup_adaptive_sampler(self, config):
        """Setup adaptive sampler and recreate dataloader"""
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

        # Recreate train loader with adaptive sampler
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,  # Sampler handles shuffling
            sampler=self.adaptive_sampler,
            num_workers=config['training']['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=config['training']['num_workers'] > 0,
            prefetch_factor=2 if config['training']['num_workers'] > 0 else None,
        )

    def _update_sampler_weights(self, metrics):
        """Update adaptive sampler based on validation metrics"""
        if not hasattr(self, 'adaptive_sampler'):
            return

        if 'pr_data' not in metrics:
            return

        # Extract per-class AP
        class_ap = {}
        for class_id, data in metrics['pr_data'].items():
            if class_id > 0:  # Skip background
                class_ap[class_id] = data['ap']

        # Extract Background Accuracy (if available)
        bg_accuracy = metrics.get('bg_accuracy', None)

        # --- UPDATE THIS LINE TO PASS bg_accuracy ---
        self.adaptive_sampler.update_class_weights(class_ap, bg_accuracy)

        # Log weights to tensorboard
        for class_id, weight in self.adaptive_sampler.get_current_weights().items():
            if class_id < len(self.config['model']['class_names']):
                class_name = self.config['model']['class_names'][class_id]
            else:
                class_name = f"class_{class_id}"
                
            self.writer.add_scalar(f'Sampler_Weights/{class_name}', weight, self.current_epoch)
            
        # Log the dynamic background ratio so you can watch it change
        if hasattr(self.adaptive_sampler, 'bg_ratio'):
            self.writer.add_scalar('Sampler_Weights/Background_Ratio', self.adaptive_sampler.bg_ratio, self.current_epoch)

    def train_one_epoch(self, epoch: int):
        """Train for one epoch with tqdm progress bar"""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:3d}/{self.config['training']['num_epochs']:3d} Train",
            leave=True,
            dynamic_ncols=True,
            unit='batch',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for batch_idx, (pixel_values, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]

            outputs = self.model(pixel_values=pixel_values, labels=targets)
            loss = outputs.loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'batch_loss': f'{loss.item():.4f}'
            })

        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate the model with tqdm progress bar"""
        self.model.eval()
        total_loss = 0.0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.current_epoch+1:3d}/{self.config['training']['num_epochs']:3d} Valid",
            leave=True,
            dynamic_ncols=True,
            unit='batch',
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )

        for batch_idx, (pixel_values, targets) in enumerate(pbar):
            pixel_values = pixel_values.to(self.device)
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                for t in targets
            ]
            outputs = self.model(pixel_values=pixel_values, labels=targets)
            total_loss += outputs.loss.item()
            
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'batch_loss': f'{outputs.loss.item():.4f}'
            })

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss

    def train(self):
        """Main training loop with adaptive sampling"""
        print("="*70)
        print("DETR Training with Adaptive Sampling")
        print("="*70)
        print(f"  Training samples   : {len(self.train_dataset)}")
        print(f"  Validation samples : {len(self.val_dataset)}")
        print(f"  Test samples       : {len(self.test_dataset)}")
        print(f"  Epochs             : {self.config['training']['num_epochs']}")
        print(f"  Batch size         : {self.config['training']['batch_size']}")
        print(f"  Learning rate      : {self.config['training']['learning_rate']}")
        print(f"  Eval every         : {self.eval_every_n_epochs} epoch(s)")
        print(f"  Adaptive sampling  : {self.config['data'].get('use_adaptive_sampler', False)}")
        print("="*70 + "\n")

        num_epochs = self.config['training']['num_epochs']

        for epoch in range(num_epochs):
            self.current_epoch = epoch  # Track for tensorboard logging
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1} / {num_epochs}")
            print(f"{'='*60}")
            
            # Print expected class distribution for this epoch
            if hasattr(self, 'adaptive_sampler') and epoch > 0:
                self.adaptive_sampler.print_current_distribution()

            # Train
            train_loss = self.train_one_epoch(epoch)
            print(f"  Train loss : {train_loss:.4f}")
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Validate + evaluate every N epochs
            if (epoch + 1) % self.eval_every_n_epochs == 0:

                val_loss = self.validate()
                print(f"  Val loss   : {val_loss:.4f}")
                self.writer.add_scalar('Loss/validation', val_loss, epoch)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    print(f"  ⚠ No improvement for {self.patience_counter}/{self.patience} evaluations")

                # Full evaluation (Unpacking fixed here)
                print(f"\n  Running detailed evaluation at epoch {epoch + 1}…")
                metrics, _, _ = self.evaluator.evaluate(epoch=epoch + 1)

                if 'mAP' in metrics:
                    self.writer.add_scalar('Metrics/mAP', metrics['mAP'], epoch)
                    
                    for class_id, data in metrics['pr_data'].items():
                        # Safe class naming for tensorboard
                        c_name = data["class_name"].replace(" ", "_")
                        self.writer.add_scalar(f'AP/{c_name}', data['ap'], epoch)

                    # Save best model
                    if self.save_best_model and metrics['mAP'] > self.best_map:
                        self.best_map = metrics['mAP']
                        best_path = self.config['training']['output_model_path'].replace('.pth', '_best.pth')
                        torch.save(self.model.state_dict(), best_path)
                        print(f"  🎉 New best mAP={self.best_map:.4f} → {best_path}")

                    # UPDATE ADAPTIVE SAMPLER
                    self._update_sampler_weights(metrics)

                # Early stop?
                if self.patience_counter >= self.patience:
                    print(f"\n⏹  Early stopping at epoch {epoch + 1}.")
                    break

            else:
                next_eval = ((epoch // self.eval_every_n_epochs) + 1) * self.eval_every_n_epochs
                print(f"  (Skipping validation — next eval at epoch {next_eval})")

            # Scheduler step
            self.scheduler.step()

            # Checkpoint
            if self.save_ckpt_every > 0 and (epoch + 1) % self.save_ckpt_every == 0:
                ckpt_dir = self.config['training'].get('checkpoint_dir', 'checkpoints/')
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
                
                # Save sampler state too
                sampler_state = self.adaptive_sampler.get_current_weights() if hasattr(self, 'adaptive_sampler') else None
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_loss': train_loss,
                    'best_val_loss': self.best_val_loss,
                    'best_map': self.best_map,
                    'sampler_weights': sampler_state,
                    'config': self.config,
                }, ckpt_path)
                print(f"  Checkpoint saved → {ckpt_path}")

        # Final test evaluation
        print("\n" + "="*60)
        print("Final evaluation on test set…")
        print("="*60)
        test_metrics, _, _ = self.test_evaluator.evaluate(epoch='final_test')

        out_path = self.config['training']['output_model_path']
        torch.save(self.model.state_dict(), out_path)

        # Print summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"  Best validation mAP : {self.best_map:.4f}")
        print(f"  Best val loss       : {self.best_val_loss:.4f}")
        print(f"  Final test mAP      : {test_metrics.get('mAP', 0.0):.4f}")
        print(f"  Epochs completed    : {epoch + 1}")
        print(f"  Model saved         : {out_path}")
        print(f"  TensorBoard Log     : {self.writer.log_dir}")
        
        # Print final sampler weights if adaptive
        if hasattr(self, 'adaptive_sampler'):
            print("\n  Final class weights:")
            weights = self.adaptive_sampler.get_current_weights()
            for cls_id in sorted(weights.keys())[:10]:  # Show top 10
                cls_name = self.config['model']['class_names'][cls_id] if cls_id < len(self.config['model']['class_names']) else f"class_{cls_id}"
                print(f"    {cls_name:30s}: {weights[cls_id]:.4f}")
        
        print("="*60)

        self.writer.close()
        return self.model


def main():
    config_path = 'config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    trainer = DETRTrainerWithAdaptiveSampling(config)
    trainer.train()
    print("\n✅ Done!")


if __name__ == "__main__":
    main()