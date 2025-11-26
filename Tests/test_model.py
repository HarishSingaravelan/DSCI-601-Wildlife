import torch

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modeling.model import get_model


def test_get_model_forward_pass():
    # 5 foreground classes + 1 background
    num_classes = 6
    
    # --- OPTIMIZATION: Skip loading pre-trained weights for speed ---
    model = get_model(num_classes=num_classes, load_weights=False)
    
    model.eval()

    # Single dummy image
    dummy_images = [torch.rand(3, 128, 128)]

    with torch.no_grad():
        outputs = model(dummy_images)

    assert isinstance(outputs, list)
    assert len(outputs) == 1
    assert "boxes" in outputs[0]
    assert "labels" in outputs[0]