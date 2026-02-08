"""
mAP Evaluation Script for Object Detection Models (Faster R-CNN + Custom DETRBaseline)

- Evaluates a trained .pth model on test/val dataset
- Computes COCO-style mAP using TorchMetrics MeanAveragePrecision
- Maps DETR contiguous labels (0..K-1) -> COCO category IDs using contig2cat
- Supports Faster R-CNN outputs (already COCO category IDs)
- Saves PR curve to pr_curve.png

Usage:
  python inference/mAPGenerator.py --model_path PATH_TO_PTH --config config/config.yaml --batch_size 1
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

# -----------------------------
# Add repo root to sys.path
# -----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from turbine_processing.dataset import TurbineCocoDataset
from turbine_processing.dataloader import TurbineDataLoader
from turbine_processing.transforms import get_val_transform
from modeling.model import get_model


# -----------------------------
# Config utils
# -----------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_label_maps(dataset: TurbineCocoDataset):
    """
    Build mapping between COCO category_id space and DETR contiguous space.

    dataset targets use COCO category ids (e.g., 1..5).
    DETRBaseline uses contiguous ids (0..K-1).
    """
    cat_ids = sorted(dataset.coco.getCatIds())
    cat2contig = {cid: i for i, cid in enumerate(cat_ids)}
    contig2cat = {i: cid for cid, i in cat2contig.items()}
    return cat2contig, contig2cat


# -----------------------------
# Label mapping helpers
# -----------------------------
def map_predictions(
    preds: list[dict],
    model_name: str,
    contig2cat: dict[int, int],
    device: torch.device,
    score_thresh: float | None = None,
):
    """
    Convert model predictions into TorchMetrics format (labels in COCO category id space).

    - FasterRCNN: already COCO category ids -> pass through
    - DETRBaseline: labels are contiguous 0..K-1 -> map to COCO ids via contig2cat

    score_thresh: if provided, filter out predictions below this score.
    """
    model_name = model_name.lower().strip()
    mapped = []

    for p in preds:
        boxes = p["boxes"]
        scores = p["scores"]
        labels = p["labels"]

        # Ensure tensor types
        if not torch.is_tensor(boxes):
            boxes = torch.as_tensor(boxes)
        if not torch.is_tensor(scores):
            scores = torch.as_tensor(scores)
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)

        if model_name == "detr":
            # IMPORTANT: Custom DETRBaseline outputs labels in contiguous space 0..K-1.
            # Do NOT do labels>0 filtering and do NOT subtract 1.
            if labels.numel() > 0:
                labels_cpu = labels.detach().cpu().tolist()
                labels = torch.tensor(
                    [contig2cat[int(l)] for l in labels_cpu],
                    dtype=torch.int64,
                    device=device,
                )
            else:
                labels = labels.to(device)

        else:
            # Faster R-CNN: labels already in COCO space
            labels = labels.to(device).to(torch.int64)

        boxes = boxes.to(device)
        scores = scores.to(device)

        # Optional score threshold filtering
        if score_thresh is not None:
            keep = scores >= float(score_thresh)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        mapped.append({"boxes": boxes, "scores": scores, "labels": labels})

    return mapped


def map_targets(targets: list[dict], device: torch.device):
    """
    Convert dataset targets to TorchMetrics expected format.
    Targets are already in COCO category space.
    """
    mapped = []
    for t in targets:
        mapped.append(
            {
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device).to(torch.int64),
            }
        )
    return mapped


# -----------------------------
# Precision–Recall curves
# -----------------------------
def compute_pr_curves(preds: list[dict], targets: list[dict], iou_thr: float = 0.5):
    """
    Compute PR curves per class (COCO category id space).
    Simple greedy matching at a fixed IoU threshold.
    """
    from torchvision.ops import box_iou

    class_ids = sorted({int(l) for t in targets for l in t["labels"].tolist()})
    curves = {}

    for cid in class_ids:
        thresholds = np.linspace(0, 1, 101)
        tp = np.zeros_like(thresholds)
        fp = np.zeros_like(thresholds)

        gt_total = sum((t["labels"] == cid).sum().item() for t in targets)
        if gt_total == 0:
            continue

        for i, thr in enumerate(thresholds):
            for p, t in zip(preds, targets):
                p_mask = (p["labels"] == cid) & (p["scores"] >= thr)
                pb = p["boxes"][p_mask]
                tb = t["boxes"][t["labels"] == cid]

                if len(pb) == 0:
                    continue
                if len(tb) == 0:
                    fp[i] += len(pb)
                    continue

                ious = box_iou(pb, tb)
                matched = set()

                for r in range(len(pb)):
                    m, idx = ious[r].max(0)
                    if float(m) >= iou_thr and int(idx) not in matched:
                        tp[i] += 1
                        matched.add(int(idx))
                    else:
                        fp[i] += 1

        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / gt_total

        curves[f"class_{cid}"] = {"precision": precision, "recall": recall}

    return curves


def plot_pr(curves: dict) -> PILImage.Image:
    fig, ax = plt.subplots(figsize=(8, 6))
    for k, v in curves.items():
        ax.plot(v["recall"], v["precision"], label=k)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (IoU=0.5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf)
    plt.close(fig)
    return img


# -----------------------------
# Evaluation loop
# -----------------------------
@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    contig2cat: dict[int, int],
    score_thresh: float | None,
    debug_limit: int | None = None,
):
    model.eval()
    model.to(device)

    metric = MeanAveragePrecision(box_format="xyxy").to(device)
    all_preds, all_targs = [], []

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Evaluating", dynamic_ncols=True)):
        if debug_limit is not None and batch_idx >= debug_limit:
            print(f"[DEBUG] Early stop at batch {batch_idx}")
            break

        images = [img.to(device) for img in images]
        targets_dev = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)  # list[dict(boxes,scores,labels)]

        # Debug first batch
        if batch_idx == 0:
            print("---- RAW MODEL OUTPUT (first batch) ----")
            print("Number of predictions:", len(outputs))
            if len(outputs) > 0:
                print("First pred keys:", outputs[0].keys())
                print("First pred boxes shape:", outputs[0]["boxes"].shape)
                print("First pred scores:", outputs[0]["scores"][:5] if outputs[0]["scores"].numel() > 0 else "EMPTY")
                print("First pred labels:", outputs[0]["labels"][:5] if outputs[0]["labels"].numel() > 0 else "EMPTY")

        preds = map_predictions(outputs, model_name, contig2cat, device, score_thresh=score_thresh)
        targs = map_targets(targets_dev, device)

        if batch_idx == 0:
            print("---- AFTER MAPPING ----")
            print("Pred boxes:", preds[0]["boxes"].shape)
            print("Pred scores:", preds[0]["scores"][:5] if preds[0]["scores"].numel() > 0 else "EMPTY")
            print("Pred labels:", preds[0]["labels"][:5] if preds[0]["labels"].numel() > 0 else "EMPTY")
            print("GT boxes:", targs[0]["boxes"].shape)
            print("GT labels:", targs[0]["labels"][:10] if targs[0]["labels"].numel() > 0 else "EMPTY")
            print("contig2cat mapping:", contig2cat)

        # IMPORTANT: do NOT skip metric update for empty preds; torchmetrics handles it.
        metric.update(preds, targs)

        # Store CPU copies for PR plot
        for p in preds:
            all_preds.append(
                {
                    "boxes": p["boxes"].detach().cpu(),
                    "scores": p["scores"].detach().cpu(),
                    "labels": p["labels"].detach().cpu(),
                }
            )
        for t in targs:
            all_targs.append(
                {
                    "boxes": t["boxes"].detach().cpu(),
                    "labels": t["labels"].detach().cpu(),
                }
            )

    return metric.compute(), all_preds, all_targs


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_val", action="store_true")
    parser.add_argument("--debug_limit", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"].lower().strip()

    # Pick split
    split = "val" if args.use_val else "test"
    data_cfg = cfg["data"]

    root = Path(data_cfg[f"{split}_root_dir"])
    ann = root / data_cfg[f"{split}_annotation_file"]
    img_root = root / data_cfg.get(f"{split}_images_root", ".")

    print(f"[INFO] Split: {split}")
    print(f"[INFO] Images root: {img_root}")
    print(f"[INFO] Annotation: {ann}")

    dataset = TurbineCocoDataset(
        images_dir=str(img_root),
        ann_file=str(ann),
        transforms=get_val_transform(),
    )
    print(f"[INFO] Dataset size: {len(dataset)}")

    # Sanity: how many images have empty GT boxes?
    empty = 0
    for i in range(min(200, len(dataset))):
        _, t = dataset[i]
        if t["boxes"].shape[0] == 0:
            empty += 1
    print(f"[SANITY] Empty-GT images in first 200: {empty}/200")

    _, contig2cat = build_label_maps(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TurbineDataLoader.collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    model = get_model(
        model_name=cfg["model"]["name"],
        num_object_classes=int(cfg["model"]["num_object_classes"]),
        pretrained=False,
        cfg=cfg.get("model", {}),
    ).to(device)

    # Load checkpoint safely
    ckpt = torch.load(args.model_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[INFO] Example missing key:", missing[0])
    if len(unexpected) > 0:
        print("[INFO] Example unexpected key:", unexpected[0])

    # Score threshold
    # For DETR, it's often helpful to filter very low confidence predictions.
    score_thresh = None
    if model_name == "detr":
        score_thresh = float(cfg.get("training", {}).get("detr_score_thresh", 0.05))
        print(f"[INFO] Using DETR score threshold: {score_thresh}")
    else:
        print("[INFO] No score threshold filtering for Faster R-CNN (using raw outputs).")

    metrics, preds, targs = evaluate_model(
        model=model,
        loader=loader,
        device=device,
        model_name=model_name,
        contig2cat=contig2cat,
        score_thresh=score_thresh,
        debug_limit=args.debug_limit,
    )

    print("\n===== mAP RESULTS =====")
    for k, v in metrics.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                print(f"{k:18s}: {v.item():.4f}")
            else:
                # tensors like per-class arrays
                print(f"{k:18s}: tensor(shape={tuple(v.shape)})")
        else:
            print(f"{k:18s}: {v}")

    # PR curve plot
    pr = compute_pr_curves(preds, targs, iou_thr=0.5)
    if len(pr) > 0:
        plot_pr(pr).save("pr_curve.png")
        print("[INFO] Saved PR curve → pr_curve.png")
    else:
        print("[INFO] No PR curve generated (no valid classes/GT).")


if __name__ == "__main__":
    main()

