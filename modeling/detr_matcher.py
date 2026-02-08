# modeling/detr_matcher.py

from __future__ import annotations
import torch
from scipy.optimize import linear_sum_assignment

from modeling.detr_utils import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(torch.nn.Module):
    """
    Computes 1-to-1 matching between predictions and targets for each batch element.
    Cost = w_cls * cost_class + w_bbox * L1 + w_giou * (1 - GIoU)
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = float(cost_class)
        self.cost_bbox = float(cost_bbox)
        self.cost_giou = float(cost_giou)

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs:
          pred_logits: [B,Q,C]
          pred_boxes:  [B,Q,4] in cxcywh normalized
        targets: list of dicts with:
          labels: [Ni] contiguous 0..K-1
          boxes:  [Ni,4] in xyxy absolute pixels (we convert to normalized cxcywh in loss)
        returns:
          list of (idx_pred, idx_tgt) for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        out_prob = outputs["pred_logits"].softmax(-1)  # [B,Q,C]
        out_bbox = outputs["pred_boxes"]               # [B,Q,4] normalized cxcywh

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]  # [Ni]
            tgt_bbox = targets[b]["boxes"]  # [Ni,4] normalized cxcywh already (we will enforce that in wrapper)

            if tgt_bbox.numel() == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64),
                                torch.as_tensor([], dtype=torch.int64)))
                continue

            # classification cost: negative prob of tgt class
            cost_class = -out_prob[b][:, tgt_ids]  # [Q,Ni]

            # L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)  # [Q,Ni]

            # GIoU cost
            out_xyxy = box_cxcywh_to_xyxy(out_bbox[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = 1 - generalized_box_iou(out_xyxy, tgt_xyxy)  # [Q,Ni]

            C = (self.cost_bbox * cost_bbox
                 + self.cost_class * cost_class
                 + self.cost_giou * cost_giou)

            C = C.cpu()
            i, j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(i, dtype=torch.int64),
                            torch.as_tensor(j, dtype=torch.int64)))

        return indices