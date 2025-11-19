from __future__ import annotations

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes: int, load_weights: bool = True):
    """
    Create a Faster R-CNN model with a ResNet-50 FPN backbone.

    Args:
        num_classes: number of classes including background
        load_weights: If True (default), load weights pre-trained on COCO. 
                      Set to False to skip loading weights for fast initialization.

    Returns:
        model: torch.nn.Module
    """
    # Load a model pre-trained on COCO (or without weights)
    weights_arg = "DEFAULT" if load_weights else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights_arg)

    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model