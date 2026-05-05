# modeling/trainer.py

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, List

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# from modeling.coco_eval import evaluate_map50, predictions_to_coco_results


def _move_to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def _build_coco_label_maps_from_dataset(train_dataset) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Build category_id <-> contiguous maps from dataset.coco.
    Returns:
      cat2contig: {category_id -> 0..K-1}
      contig2cat: {0..K-1 -> category_id}
    """
    cat_ids = sorted(train_dataset.coco.getCatIds())
    cat2contig = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    contig2cat = {i: cat_id for cat_id, i in cat2contig.items()}
    return cat2contig, contig2cat


def _map_targets_labels(targets, cat2contig: Dict[int, int]) -> List[dict]:
    """
    Convert COCO category_id labels to contiguous [0..K-1].
    For DETR training.
    """
    mapped = []
    for t in targets:
        t2 = dict(t)
        if t2["labels"].numel() > 0:
            labels = t2["labels"].tolist()
            labels = [cat2contig[int(x)] for x in labels]
            t2["labels"] = torch.tensor(labels, dtype=torch.int64, device=t2["labels"].device)
        mapped.append(t2)
    return mapped


@torch.no_grad()
def _postprocess_detr(outputs, images, score_thresh: float):
    """
    Make DETR outputs compatible with COCO eval:
      list of dicts: boxes xyxy abs, scores, labels
    Supports both:
      - torchvision postprocessed list outputs
      - raw dict outputs with pred_logits/pred_boxes
    """
    # Case A: already postprocessed by torchvision
    if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and "boxes" in outputs[0]:
        # Apply score threshold ourselves if needed
        cleaned = []
        for out in outputs:
            keep = out["scores"] >= score_thresh
            cleaned.append(
                {
                    "boxes": out["boxes"][keep].detach().cpu(),
                    "scores": out["scores"][keep].detach().cpu(),
                    "labels": out["labels"][keep].detach().cpu(),
                }
            )
        return cleaned

    # Case B: raw DETR outputs
    if not (isinstance(outputs, dict) and "pred_logits" in outputs and "pred_boxes" in outputs):
        raise ValueError(f"Unexpected DETR output format: {type(outputs)}")

    pred_logits = outputs["pred_logits"]  # [B, Q, C]
    pred_boxes = outputs["pred_boxes"]    # [B, Q, 4] cxcywh (relative)

    prob = pred_logits.softmax(-1)
    # ignore last class (no-object)
    scores, labels = prob[..., :-1].max(-1)  # [B, Q], [B, Q]

    batch_preds = []
    for i in range(pred_boxes.shape[0]):
        img_h, img_w = images[i].shape[-2], images[i].shape[-1]

        s = scores[i]
        l = labels[i]
        b = pred_boxes[i]

        keep = s >= score_thresh
        s = s[keep]
        l = l[keep]
        b = b[keep]

        # cxcywh -> xyxy (relative)
        cx, cy, w, h = b.unbind(-1)
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h
        b_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)

        # scale to absolute pixels
        b_xyxy[:, 0::2] *= img_w
        b_xyxy[:, 1::2] *= img_h

        batch_preds.append(
            {
                "boxes": b_xyxy.detach().cpu(),
                "scores": s.detach().cpu(),
                "labels": l.detach().cpu(),
            }
        )

    return batch_preds


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        log_dir: str,
        model_name: str,
        train_dataset=None,
        val_dataset=None,
        score_thresh: float = 0.05,
        grad_clip_norm: Optional[float] = 0.1,
        use_amp: bool = True,
        grad_accum_steps: int = 1 
    ) -> None:
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.scaler = None
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.writer = SummaryWriter(log_dir=log_dir)
        self.model_name = model_name.lower().strip()
        self.score_thresh = float(score_thresh)
        self.grad_clip_norm = grad_clip_norm

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.cat2contig = None
        self.contig2cat = None

        if self.model_name == "detr":
            if train_dataset is None or val_dataset is None:
                raise ValueError("For DETR, pass train_dataset and val_dataset to Trainer.")
            self.cat2contig, self.contig2cat = _build_coco_label_maps_from_dataset(train_dataset)

    def close(self):
        self.writer.close()

    def train_one_epoch(self, train_loader, val_loader, epoch: int):
        self.model.to(self.device)
        self.model.train()

        use_amp = bool(getattr(self, "use_amp", False))
        grad_accum_steps = int(getattr(self, "grad_accum_steps", 1))
        grad_accum_steps = max(1, grad_accum_steps)

        scaler = getattr(self, "scaler", None)
        if use_amp and scaler is None:
            self.scaler = torch.cuda.amp.GradScaler()
            scaler = self.scaler

        running_loss = 0.0
        loss_parts_sum: Dict[str, float] = {}

        self.optimizer.zero_grad(set_to_none=True)

        # -----------------------------
        # TQDM progress bar
        # -----------------------------
        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=len(train_loader),
            desc=f"Epoch {epoch} [{self.model_name}]",
            leave=True,
            dynamic_ncols=True,
        )

        for step, (images, targets) in pbar:
            images, targets = _move_to_device(list(images), list(targets), self.device)

            if self.model_name == "detr":
                targets = _map_targets_labels(targets, self.cat2contig)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss_dict = self.model(images, targets)
                loss = sum(loss_dict.values()) / grad_accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                if self.grad_clip_norm is not None:
                    if use_amp:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                if use_amp:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

            running_loss += float(loss.item()) * grad_accum_steps
            for k, v in loss_dict.items():
                loss_parts_sum[k] = loss_parts_sum.get(k, 0.0) + float(v.item())

            # -----------------------------
            # Update TQDM display
            # -----------------------------
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / step:.4f}",
                    "amp": use_amp,
                    "accum": grad_accum_steps,
                }
            )

        avg_loss = running_loss / max(1, len(train_loader))
        self.writer.add_scalar("Loss/train_total", avg_loss, epoch)

        for k, v in loss_parts_sum.items():
            self.writer.add_scalar(f"Loss/{k}", v / max(1, len(train_loader)), epoch)

        print(f"[INFO] Epoch {epoch} train loss: {avg_loss:.4f}")
    @torch.no_grad()
    def validate_metrics(self, val_loader, epoch: int):
        self.model.eval()

        # COCO GT object for eval
        if self.val_dataset is None or not hasattr(self.val_dataset, "coco"):
            raise ValueError("val_dataset with .coco is required for COCO metrics.")
        coco_gt = self.val_dataset.coco

        all_preds = []
        all_img_ids = []

        for (images, targets) in val_loader:
            images = list(images)
            targets = list(targets)

            img_ids = [int(t["image_id"].item()) for t in targets]
            all_img_ids.extend(img_ids)

            images_t, _ = _move_to_device(images, targets, self.device)

            outputs = self.model(images_t)

            if self.model_name == "detr":
                # our DETRBaseline already returns list of dicts
                preds = []
                for out in outputs:
                    keep = out["scores"] >= self.score_thresh
                    preds.append(
                        {
                            "boxes": out["boxes"][keep].detach().cpu(),
                            "scores": out["scores"][keep].detach().cpu(),
                            "labels": out["labels"][keep].detach().cpu(),
                        }
                    )
                label_map_back = self.contig2cat

            all_preds.extend(preds)

        coco_results = predictions_to_coco_results(all_preds, all_img_ids, label_map_back=label_map_back)
        map50 = evaluate_map50(coco_gt, coco_results)

        self.writer.add_scalar("mAP/val_map50", map50, epoch)
        print(f"[INFO] Epoch {epoch} val mAP@0.5: {map50:.4f}")
        return map50