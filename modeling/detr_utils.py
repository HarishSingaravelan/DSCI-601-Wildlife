# modeling/detr_utils.py

from __future__ import annotations
import torch
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = x.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h),
         (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes: xyxy
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


def nested_tensor_from_tensor_list(tensor_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads images to same size:
      returns:
        batch_images: [B,3,Hmax,Wmax]
        mask:         [B,Hmax,Wmax] where True is padding
    """
    max_h = max(img.shape[-2] for img in tensor_list)
    max_w = max(img.shape[-1] for img in tensor_list)
    b = len(tensor_list)

    batch = tensor_list[0].new_zeros((b, 3, max_h, max_w))
    mask = torch.ones((b, max_h, max_w), dtype=torch.bool, device=tensor_list[0].device)

    for i, img in enumerate(tensor_list):
        c, h, w = img.shape
        batch[i, :c, :h, :w] = img
        mask[i, :h, :w] = False

    return batch, mask


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(torch.nn.Linear(in_d, out_d))
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x