# modeling/coco_eval.py

from __future__ import annotations
import numpy as np
from pycocotools.cocoeval import COCOeval


def _xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def predictions_to_coco_results(predictions, image_ids, label_map_back=None):
    """
    predictions: list of dicts with 'boxes' (Nx4 xyxy), 'scores', 'labels'
    image_ids: list of int aligned with predictions
    label_map_back: optional dict {contig_label -> category_id}
    """
    coco_results = []
    for pred, img_id in zip(predictions, image_ids):
        boxes = pred["boxes"]
        scores = pred["scores"]
        labels = pred["labels"]

        for b, s, l in zip(boxes, scores, labels):
            cat_id = int(l)
            if label_map_back is not None:
                cat_id = int(label_map_back[cat_id])

            coco_results.append(
                {
                    "image_id": int(img_id),
                    "category_id": cat_id,
                    "bbox": _xyxy_to_xywh(b.tolist()),
                    "score": float(s),
                }
            )
    return coco_results


def evaluate_map50(coco_gt, coco_results, iou_type="bbox") -> float:
    """
    coco_gt: pycocotools COCO object (ground truth)
    coco_results: list in COCO det format
    Returns mAP@0.5
    """
    if len(coco_results) == 0:
        return 0.0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType=iou_type)

    coco_eval.params.iouThrs = np.array([0.5], dtype=np.float32)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return float(coco_eval.stats[0])