# modeling/model.py

from __future__ import annotations
from typing import Optional, Dict, Any

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from modeling.detr_baseline import DETRBaseline, DETRConfig


def _build_fasterrcnn(num_classes: int, pretrained: bool = True):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT" if pretrained else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _build_detr(num_object_classes: int, pretrained_backbone: bool = True, cfg: Optional[Dict[str, Any]] = None):
    cfg = cfg or {}

    detr_cfg = DETRConfig(
        num_classes=num_object_classes,
        num_queries=int(cfg.get("detr_num_queries", 100)),
        hidden_dim=int(cfg.get("detr_hidden_dim", 256)),
        enc_layers=int(cfg.get("detr_enc_layers", 6)),
        dec_layers=int(cfg.get("detr_dec_layers", 6)),
        nheads=int(cfg.get("detr_nheads", 8)),
    )
    return DETRBaseline(cfg=detr_cfg, pretrained_backbone=pretrained_backbone)


def get_model(
    num_classes: int = None,
    model_name: str = "fasterrcnn",
    num_object_classes: int = None,
    pretrained: bool = True,
    cfg: Optional[Dict[str, Any]] = None,
):
    """
    Backward-compatible get_model.

    Notes:
    - FasterRCNN expects num_classes INCLUDING background.
    - DETRBaseline expects num_object_classes (foreground only).
    """
    model_name = (model_name or "fasterrcnn").lower().strip()

    # ----- Old API path -----
    if num_classes is not None and num_object_classes is None:
        # assume FasterRCNN old usage
        return _build_fasterrcnn(num_classes=num_classes, pretrained=pretrained)

    # ----- New API path -----
    if num_object_classes is None:
        raise ValueError("num_object_classes must be provided when using model_name-based API.")

    if model_name == "fasterrcnn":
        return _build_fasterrcnn(num_classes=num_object_classes + 1, pretrained=pretrained)

    if model_name == "detr":
        return _build_detr(num_object_classes=num_object_classes, pretrained_backbone=pretrained, cfg=cfg)

    raise ValueError(f"Unknown model_name: {model_name}. Use 'fasterrcnn' or 'detr'.")