import pytest
import math
from typing import List

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import your sampler (adjust the import path if necessary)
from turbine_processing.sampler_adaptive import AdaptiveDETRSampler

# ==========================================
# 1. MOCK DATASET SETUP
# ==========================================
class MockCOCO:
    """Fakes the COCO API so the sampler can initialize"""
    def getAnnIds(self, imgIds: List[int]):
        img_id = imgIds[0]
        # Images 0-49 are background (no annotations)
        if img_id < 50:
            return []
        # Images 50-99 are birds
        return [img_id]

    def loadAnns(self, ann_ids: List[int]):
        # Bird images are split into Class 1 and Class 2
        cat_id = 1 if ann_ids[0] % 2 == 0 else 2
        return [{"category_id": cat_id}]

    def loadCats(self, cat_ids: List[int]):
        return [{"name": f"MockBird_{cat_ids[0]}"}]

class MockDataset:
    def __init__(self):
        self.ids = list(range(100))
        self.coco = MockCOCO()
        
    def __len__(self):
        return len(self.ids)


# ==========================================
# 2. PYTEST FIXTURES
# ==========================================
@pytest.fixture
def sampler():
    """Creates a fresh sampler before every test"""
    dataset = MockDataset()
    return AdaptiveDETRSampler(
        dataset=dataset,
        epoch_size=100,
        initial_mode='equal',
        adaptation_rate=0.3,       # 30% adaptation speed
        background_ratio=0.5,      # Start at 50%
        dynamic_background=True,
        min_bg_ratio=0.1,          # Hard floor at 10%
        max_bg_ratio=0.5           # Hard ceiling at 50%
    )


# ==========================================
# 3. THE TESTS
# ==========================================

def test_dynamic_background_decrease(sampler):
    """Test: High accuracy should lower the background ratio"""
    old_bg = sampler.bg_ratio # Should be 0.5
    
    # Simulate a great epoch: 95% background accuracy, decent bird APs
    mock_metrics = {1: 0.5, 2: 0.5}
    sampler.update_class_weights(mock_metrics, bg_accuracy=0.95)
    
    # Target should be: 1.0 - 0.95 + 0.1 = 0.15
    # Blended should be: (0.7 * 0.5) + (0.3 * 0.15) = 0.35 + 0.045 = 0.395
    assert math.isclose(sampler.bg_ratio, 0.395, rel_tol=1e-4), \
        f"Expected 0.395, but got {sampler.bg_ratio}"
    assert sampler.bg_ratio < old_bg, "Ratio failed to decrease!"

def test_dynamic_background_increase(sampler):
    """Test: Low accuracy should bounce the background ratio back up"""
    # Force ratio down to 0.2 first
    sampler.bg_ratio = 0.2
    
    # Simulate a terrible epoch: 40% background accuracy (hallucinations)
    mock_metrics = {1: 0.5, 2: 0.5}
    sampler.update_class_weights(mock_metrics, bg_accuracy=0.40)
    
    # Target should be: 1.0 - 0.40 + 0.1 = 0.70
    # Blended should be: (0.7 * 0.2) + (0.3 * 0.70) = 0.14 + 0.21 = 0.35
    assert math.isclose(sampler.bg_ratio, 0.35, rel_tol=1e-4), \
        f"Expected 0.35, but got {sampler.bg_ratio}"

def test_background_ratio_hits_floor(sampler):
    """Test: Ratio should never drop below min_bg_ratio (0.1)"""
    # Give it 40 epochs to smoothly decay down to the floor
    for _ in range(40):
        sampler.update_class_weights({1: 0.5, 2: 0.5}, bg_accuracy=1.0)
    
    assert sampler.bg_ratio >= 0.1, "Ratio dropped below the hard minimum!"
    assert math.isclose(sampler.bg_ratio, 0.1, rel_tol=1e-2), \
        f"Expected to hit floor of 0.1, got {sampler.bg_ratio}"

def test_iterator_respects_dynamic_ratio(sampler):
    """Test: __iter__ physically pulls the exact right amount of images"""
    # Force the ratio to exactly 30%
    sampler.bg_ratio = 0.30
    
    # Get one epoch of indices
    indices = list(iter(sampler))
    
    # Total epoch size is 100
    assert len(indices) == 100, "Epoch size is wrong"
    
    # Count how many are background (ID < 50) vs birds (ID >= 50)
    bg_pulled = sum(1 for idx in indices if idx < 50)
    birds_pulled = sum(1 for idx in indices if idx >= 50)
    
    # It should be exactly 30 background and 70 birds
    assert bg_pulled == 30, f"Expected 30 background images, got {bg_pulled}"
    assert birds_pulled == 70, f"Expected 70 bird images, got {birds_pulled}"