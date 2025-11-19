from __future__ import annotations

from typing import Dict, Any
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import for detection metrics (mAP, precision, recall)
from torchmetrics.detection import MeanAveragePrecision


class Trainer:
    """
    Trainer for Faster R-CNN with:
      - Training loss
      - Validation loss
      - Validation metrics (mAP, precision, recall)
      - TensorBoard logging
      - ETA via tqdm
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str = "runs/faster_rcnn_experiment",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir)
        
        # Initialize the mAP metric calculator
        # box_format='xyxy' matches the dataset output format
        self.metric_calculator = MeanAveragePrecision(box_format="xyxy").to(device)


    # -----------------------------------------------------------
    #   TRAIN ONE EPOCH  (TRAIN LOSS)
    # -----------------------------------------------------------
    def train_one_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:

        self.model.train()
        running_loss = 0.0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch} [Training]",
            dynamic_ncols=True,
            unit="batch",
            smoothing=0.1,
            leave=True,
        )

        for batch_idx, (images, targets) in enumerate(progress):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_item = loss.item()
            running_loss += loss_item
            avg_loss = running_loss / (batch_idx + 1)
            
            # --- TensorBoard Logging (Per Batch) ---
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar("Loss/Batch/Total", loss_item, global_step)
            for k, v in loss_dict.items():
                self.writer.add_scalar(f"Loss/Batch/{k}", v.item(), global_step)

            progress.set_postfix({
                "loss": f"{loss_item:.4f}",
                "avg": f"{avg_loss:.4f}",
            })

        train_loss = running_loss / max(1, len(train_loader))
        self.writer.add_scalar("Loss/Epoch/Train", train_loss, epoch)

        # ---- VALIDATION LOSS ----
        val_loss = self.validate_loss(val_loader)
        self.writer.add_scalar("Loss/Epoch/Validation", val_loss, epoch)

        print(
            f"[INFO] Epoch {epoch} Completed | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        return {"train_loss": train_loss, "val_loss": val_loss}

    # -----------------------------------------------------------
    #   VALIDATION LOSS (UNMODIFIED)
    # -----------------------------------------------------------
    @torch.no_grad()
    def validate_loss(self, val_loader: DataLoader) -> float:
        """
        Faster R-CNN returns loss ONLY in training mode,
        so we must keep model.train() but skip optimizer updates.
        """
        self.model.train()
        running_loss = 0.0

        progress = tqdm(
            val_loader,
            desc="  [Validating Loss]",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        )

        for images, targets in progress:
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            loss = sum(v for v in loss_dict.values())

            running_loss += loss.item()

        return running_loss / max(1, len(val_loader))

    # -----------------------------------------------------------
    #   VALIDATION METRICS (mAP, Precision, Recall)
    # -----------------------------------------------------------
    @torch.no_grad()
    def validate_metrics(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Run model in evaluation mode to get predictions and compute mAP metrics.
        """
        self.model.eval()
        self.metric_calculator.reset()
        
        progress = tqdm(
            val_loader,
            desc="  [Validating Metrics]",
            dynamic_ncols=True,
            unit="batch",
            leave=False,
        )
        
        for images, targets in progress:
            images = [img.to(self.device) for img in images]
            
            # targets must be a list of dicts (Pytorch detection format)
            targets_gpu = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Run inference
            predictions = self.model(images)
            
            # Update metric calculator with predictions and ground truth
            self.metric_calculator.update(predictions, targets_gpu)

        # Compute metrics
        metrics = self.metric_calculator.compute()
        
        # Log metrics to TensorBoard
        self.writer.add_scalar("Metrics/mAP_50", metrics["map_50"].item(), epoch)
        self.writer.add_scalar("Metrics/mAP_75", metrics["map_75"].item(), epoch)
        self.writer.add_scalar("Metrics/mAP_Total", metrics["map"].item(), epoch)
        
        # Optionally log precision/recall specific metrics if available in the full output
        # (Torchmetrics computes many, but map is the primary summary)
        print(f"[INFO] Epoch {epoch} Metrics: mAP@0.5: {metrics['map_50'].item():.4f}, mAP: {metrics['map'].item():.4f}")
        
        return metrics

    # -----------------------------------------------------------
    #   INFERENCE (PREDICT) (UNMODIFIED)
    # -----------------------------------------------------------
    @torch.no_grad()
    def inference(self, image_tensor):
        """
        Run inference on a single image or a batch.
        Input should be Tensor[C,H,W].
        """
        self.model.eval()

        if isinstance(image_tensor, torch.Tensor) and image_tensor.dim() == 3:
            images = [image_tensor.to(self.device)]
        else:
            images = [img.to(self.device) for img in image_tensor]

        return self.model(images)

    # -----------------------------------------------------------
    #   WRITER CLOSE
    # -----------------------------------------------------------
    def close(self):
        """Closes the TensorBoard writer."""
        self.writer.close()
