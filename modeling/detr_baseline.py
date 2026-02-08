# modeling/detr_baseline.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from modeling.detr_utils import nested_tensor_from_tensor_list, MLP, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, generalized_box_iou
from modeling.detr_transformer import PositionEmbeddingSine, DETRTransformer
from modeling.detr_matcher import HungarianMatcher


@dataclass
class DETRConfig:
    num_classes: int          # foreground classes K (NOT including no-object)
    num_queries: int = 100
    hidden_dim: int = 256
    nheads: int = 8
    enc_layers: int = 6
    dec_layers: int = 6
    dropout: float = 0.1

    # loss weights
    cls_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0

    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0
    eos_coef: float = 0.1  # weight for no-object class


class DETRBaseline(nn.Module):
    """
    Minimal DETR baseline:
      - ResNet-50 backbone (conv4 feature map)
      - sine positional encoding
      - transformer encoder-decoder
      - Hungarian matching loss (class, L1, GIoU)
    """

    def __init__(self, cfg: DETRConfig, pretrained_backbone: bool = True):
        super().__init__()
        self.cfg = cfg
        self.num_classes_with_noobj = cfg.num_classes + 1  # last is no-object

        # Backbone: ResNet50 up to layer4; use C5 output
        backbone = resnet50(weights="DEFAULT" if pretrained_backbone else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # [B,2048,H/32,W/32]

        self.input_proj = nn.Conv2d(2048, cfg.hidden_dim, kernel_size=1)

        self.pos_embed = PositionEmbeddingSine(num_pos_feats=cfg.hidden_dim // 2, normalize=True)
        self.transformer = DETRTransformer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nheads,
            num_encoder_layers=cfg.enc_layers,
            num_decoder_layers=cfg.dec_layers,
            dropout=cfg.dropout,
        )

        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        self.class_embed = nn.Linear(cfg.hidden_dim, self.num_classes_with_noobj)
        self.bbox_embed = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, 3)

        self.matcher = HungarianMatcher(cost_class=cfg.cls_cost, cost_bbox=cfg.bbox_cost, cost_giou=cfg.giou_cost)

        # classification loss weight for no-object
        empty_weight = torch.ones(self.num_classes_with_noobj)
        empty_weight[-1] = cfg.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None):
        """
        Train:  returns dict of losses
        Eval:   returns list[{boxes,scores,labels}]
        """
        device = images[0].device
        batch, mask = nested_tensor_from_tensor_list(images)  # [B,3,H,W], [B,H,W]
        features = self.backbone(batch)                        # [B,2048,h,w]
        src = self.input_proj(features)                        # [B,256,h,w]

        # downsample mask to feature size
        mask_small = F.interpolate(mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]  # [B,h,w]
        pos = self.pos_embed(mask_small)  # [B,256,h,w]

        # transformer
        B, C, h, w = src.shape
        src_key_padding_mask = mask_small.flatten(1)  # [B,hw]
        hs = self.transformer(
            src=src,
            src_key_padding_mask=src_key_padding_mask,
            query_embed=self.query_embed.weight,
            pos_embed=pos,
        )  # [B,Q,C]

        pred_logits = self.class_embed(hs)                 # [B,Q,C+1]
        pred_boxes = self.bbox_embed(hs).sigmoid()         # [B,Q,4] normalized cxcywh

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

        if targets is None:
            return self._inference(out, images)

        return self._losses(out, targets, images)

    def _inference(self, outputs, images, score_thresh: float = 0.01):
        prob = outputs["pred_logits"].softmax(-1)
        scores, labels = prob[..., :-1].max(-1)  # ignore no-object
        boxes = outputs["pred_boxes"]

        preds = []
        for i in range(boxes.shape[0]):
            img_h, img_w = images[i].shape[-2], images[i].shape[-1]
            keep = scores[i] >= score_thresh

            b = boxes[i][keep]
            s = scores[i][keep]
            l = labels[i][keep]

            b = box_cxcywh_to_xyxy(b)
            b[:, 0::2] *= img_w
            b[:, 1::2] *= img_h

            preds.append({"boxes": b.detach(), "scores": s.detach(), "labels": l.detach()})
        return preds

    def _normalize_targets(self, targets, images):
        """
        Convert target boxes (xyxy absolute) -> normalized cxcywh in [0,1]
        """
        norm_targets = []
        for t, img in zip(targets, images):
            img_h, img_w = img.shape[-2], img.shape[-1]
            t2 = dict(t)
            if t2["boxes"].numel() > 0:
                b = t2["boxes"].clone()
                b[:, 0::2] /= img_w
                b[:, 1::2] /= img_h
                b = box_xyxy_to_cxcywh(b)
                t2["boxes"] = b
            norm_targets.append(t2)
        return norm_targets

    def _losses(self, outputs, targets, images):
        # normalize gt boxes to cxcywh
        targets = self._normalize_targets(targets, images)

        # match
        indices = self.matcher(outputs, targets)

        # classification loss
        bs, num_queries = outputs["pred_logits"].shape[:2]
        tgt_classes = torch.full((bs, num_queries), self.num_classes_with_noobj - 1,
                                 dtype=torch.int64, device=outputs["pred_logits"].device)

        for b, (idx_pred, idx_tgt) in enumerate(indices):
            if idx_pred.numel() == 0:
                continue
            tgt_classes[b, idx_pred] = targets[b]["labels"][idx_tgt]

        loss_ce = F.cross_entropy(outputs["pred_logits"].transpose(1, 2), tgt_classes, weight=self.empty_weight)

        # bbox + giou losses for matched pairs
        loss_bbox = torch.tensor(0.0, device=loss_ce.device)
        loss_giou = torch.tensor(0.0, device=loss_ce.device)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = max(num_boxes, 1)

        for b, (idx_pred, idx_tgt) in enumerate(indices):
            if idx_pred.numel() == 0:
                continue
            src_boxes = outputs["pred_boxes"][b, idx_pred]  # cxcywh norm
            tgt_boxes = targets[b]["boxes"][idx_tgt]        # cxcywh norm

            loss_bbox = loss_bbox + F.l1_loss(src_boxes, tgt_boxes, reduction="sum")

            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
            loss_giou = loss_giou + (1 - giou).sum()

        loss_bbox = loss_bbox / num_boxes
        loss_giou = loss_giou / num_boxes

        losses = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox * self.cfg.bbox_loss_coef,
            "loss_giou": loss_giou * self.cfg.giou_loss_coef,
        }
        return losses