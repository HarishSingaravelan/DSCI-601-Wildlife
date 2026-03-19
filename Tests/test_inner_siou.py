import torch
import pytest
import transformers.models.detr.modeling_detr as modeling_detr

# ==========================================
# 1. THE PATCH FUNCTION
# ==========================================
def patch_inner_siou_loss():
    def inner_siou_matrix(boxes1, boxes2, ratio=1.25, eps=1e-7):
        x1_1, y1_1, x2_1, y2_1 = boxes1.unsqueeze(1).unbind(dim=2) 
        x1_2, y1_2, x2_2, y2_2 = boxes2.unsqueeze(0).unbind(dim=2) 
        
        w1, h1 = x2_1 - x1_1, y2_1 - y1_1
        w2, h2 = x2_2 - x1_2, y2_2 - y1_2
        cx1, cy1 = (x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2
        cx2, cy2 = (x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2
        
        in_w1, in_h1 = w1 * ratio, h1 * ratio
        in_w2, in_h2 = w2 * ratio, h2 * ratio
        
        in_x1_1, in_y1_1 = cx1 - in_w1 / 2, cy1 - in_h1 / 2
        in_x2_1, in_y2_1 = cx1 + in_w1 / 2, cy1 + in_h1 / 2
        in_x1_2, in_y1_2 = cx2 - in_w2 / 2, cy2 - in_h2 / 2
        in_x2_2, in_y2_2 = cx2 + in_w2 / 2, cy2 + in_h2 / 2
        
        inter_x1 = torch.max(in_x1_1, in_x1_2)
        inter_y1 = torch.max(in_y1_1, in_y1_2)
        inter_x2 = torch.min(in_x2_1, in_x2_2)
        inter_y2 = torch.min(in_y2_1, in_y2_2)
        
        inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_w * inter_h
        
        area1, area2 = in_w1 * in_h1, in_w2 * in_h2
        union = area1 + area2 - inter_area
        inner_iou = inter_area / (union + eps)
        
        s_cw = torch.abs(cx2 - cx1)
        s_ch = torch.abs(cy2 - cy1)
        
        Lambda = 2 * s_cw * s_ch / (s_cw**2 + s_ch**2 + eps)
        gamma = 2 - Lambda
        
        enc_x1 = torch.min(x1_1, x1_2)
        enc_y1 = torch.min(y1_1, y1_2)
        enc_x2 = torch.max(x2_1, x2_2)
        enc_y2 = torch.max(y2_1, y2_2)
        
        cw = torch.clamp(enc_x2 - enc_x1, min=eps)
        ch = torch.clamp(enc_y2 - enc_y1, min=eps)
        
        rho_x = (s_cw / cw) ** 2
        rho_y = (s_ch / ch) ** 2
        Delta = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))
        
        omega_w = torch.abs(w1 - w2) / torch.clamp(torch.max(w1, w2), min=eps)
        omega_h = torch.abs(h1 - h2) / torch.clamp(torch.max(h1, h2), min=eps)
        Omega = (1 - torch.exp(-omega_w)) ** 4 + (1 - torch.exp(-omega_h)) ** 4
        
        siou_penalty = (Delta + Omega) / 2
        return inner_iou - siou_penalty

    modeling_detr.generalized_box_iou = inner_siou_matrix

# ==========================================
# 2. PYTEST SUITE
# ==========================================

def test_patch_applied():
    """Verify that the monkey patch actually overwrites the HuggingFace function."""
    patch_inner_siou_loss()
    assert "inner_siou_matrix" in modeling_detr.generalized_box_iou.__name__, "Patch failed to apply!"

def test_inner_siou_basic_properties():
    """Test standard bounding box comparisons for shape, NaNs, and logical scoring."""
    patch_inner_siou_loss()
    
    # Format: [x1, y1, x2, y2]
    preds = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],  # Normal box
        [15.0, 15.0, 25.0, 25.0]   # Offset box
    ])
    
    targets = torch.tensor([
        [10.0, 10.0, 20.0, 20.0],     # Identical to preds[0]
        [100.0, 100.0, 110.0, 110.0], # Completely disjoint from everything
        [12.0, 12.0, 18.0, 18.0]      # Smaller box fully inside preds[0]
    ])
    
    # Execute the loss calculation
    result = modeling_detr.generalized_box_iou(preds, targets)
    
    # 1. Output Shape Test [N, M]
    assert result.shape == (2, 3), f"Expected shape (2, 3), got {result.shape}"
    
    # 2. Numerical Stability Test
    assert not torch.isnan(result).any(), "CRITICAL: Loss function generated NaN values!"
    assert not torch.isinf(result).any(), "CRITICAL: Loss function generated Infinite values!"
    
    # 3. Identical Box Test (Should be ~1.0)
    # If prediction perfectly matches ground truth, Inner-IoU=1 and Penalty=0
    assert torch.isclose(result[0, 0], torch.tensor(1.0), atol=1e-4), f"Identical boxes should yield ~1.0, got {result[0, 0]}"
    
    # 4. Disjoint Box Test (Should be negative)
    # If boxes are far apart, IoU=0 and Penalty pushes score negative
    assert result[0, 1] < 0, f"Disjoint boxes should yield a negative score, got {result[0, 1]}"

def test_zero_area_edge_case():
    """Test division-by-zero safeguards by passing impossible geometry."""
    patch_inner_siou_loss()
    
    # A box with 0 width and 0 height (a single point)
    bad_preds = torch.tensor([[10.0, 10.0, 10.0, 10.0]]) 
    targets = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    
    result = modeling_detr.generalized_box_iou(bad_preds, targets)
    
    # If the `eps=1e-7` safety nets fail, this will trigger a NaN error
    assert not torch.isnan(result).any(), "CRITICAL: Zero-area box caused division by zero (NaN)!"
    assert not torch.isinf(result).any(), "CRITICAL: Zero-area box caused Infinity!"

if __name__ == "__main__":
    # Allows you to run this easily on the login node without needing the pytest CLI
    print("Running Inner-SIoU diagnostics...")
    test_patch_applied()
    print("Patch successfully binds to HuggingFace.")
    
    test_inner_siou_basic_properties()
    print("Matrix shapes, mathematical bounds, and logic pass.")
    
    test_zero_area_edge_case()
    print("Division-by-zero safeguards pass.")
    
    print("\nInner-SIoU Loss is numerically stable and safe to queue on the cluster!")