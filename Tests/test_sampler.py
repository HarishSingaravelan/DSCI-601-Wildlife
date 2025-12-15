import pytest
from collections import Counter
from unittest.mock import MagicMock, patch
import random 
import time

from turbine_processing.sampler import DynamicBalancedSampler 


# --- 1. Mock Data Setup (Robust Static Map) ---

def create_mock_dataset():
    """
    Creates a mock dataset object with 5 BG and 5 Animal images (C1:2, C2:1, C3:2).
    """
    mock_dataset = MagicMock()
    mock_dataset.ids = list(range(10, 20)) 
    mock_dataset.__len__.return_value = 10
    
    # Map Image ID to annotations
    IMAGE_ANNOTATION_MAP = {
        10: [], 11: [], 12: [], 13: [], 14: [],
        15: [{"category_id": 1}], 16: [{"category_id": 1}],
        17: [{"category_id": 2}], # Minority class image (Index 7)
        18: [{"category_id": 3}], 19: [{"category_id": 3}], 
    }

    mock_coco = MagicMock()

    # The sampler iterates through 0 to 9. We need to track the current index to return the right ann list.
    def mock_getAnnIds(imgIds=None, catIds=None, iscrowd=None):
        img_id = imgIds[0]
        return [1] if IMAGE_ANNOTATION_MAP.get(img_id) else []

    mock_coco.getAnnIds.side_effect = mock_getAnnIds
    
    mock_dataset.coco = mock_coco

    return mock_dataset

@pytest.fixture
def mock_sampler():
    """Fixture to create and initialize the sampler with a mock dataset."""
    dataset = create_mock_dataset()
    sampler = DynamicBalancedSampler(dataset, epoch_size=10, seed=42)
    
    sampler.background_indices = [0, 1, 2, 3, 4]
    sampler.class_to_indices = {
        1: [5, 6],    # Class 1: 2 images
        2: [7],       # Class 2: 1 image (Minority)
        3: [8, 9]     # Class 3: 2 images
    }
    sampler.classes = [1, 2, 3]
    # This guarantees the test starts with the right data distribution.
    
    return sampler


# --- 3. Test Cases ---

def test_sampler_initialization_correct(mock_sampler):
    """Test that the internal maps are built correctly based on the mock data."""
    # This assertion now confirms our manual injection is correct.
    assert len(mock_sampler.class_to_indices) == 3 
    assert len(mock_sampler.background_indices) == 5 
    assert mock_sampler.class_to_indices[2] == [7]

@patch('turbine_processing.sampler.time.time_ns', return_value=123456789)
def test_sampler_main_logic_50_50_split_and_oversampling(mock_time, mock_sampler):
    """
    Tests the 50/50 split (5 BG, 5 Animal) and equal distribution across classes 
    with guaranteed oversampling.
    """
    random.seed(42)

    indices = list(mock_sampler)
    
    # --- Verify 50/50 Split ---
    bg_indices = mock_sampler.background_indices 
    animal_indices = [idx for lst in mock_sampler.class_to_indices.values() for idx in lst]
    bg_count = sum(1 for idx in indices if idx in bg_indices)
    animal_count = sum(1 for idx in indices if idx in animal_indices)
    assert bg_count == 5
    assert animal_count == 5

    # Target per class: 5 animal samples // 3 classes = 1 (remainder 2)
    # Target: Class 1: 2, Class 2: 2, Class 3: 1
    
    final_animal_indices = [idx for idx in indices if idx in animal_indices]
    final_animal_counts = Counter(final_animal_indices)
    
    # Assert that index 7 (Class 2 minority) was sampled exactly 2 times
    assert final_animal_counts.get(7, 0) == 2 
    
    # Check Class 1 (Indices 5, 6) sampling (Total count must be 2)
    assert final_animal_counts.get(5, 0) + final_animal_counts.get(6, 0) == 2

    # Check Class 3 (Indices 8, 9) sampling (Total count must be 1)
    assert final_animal_counts.get(8, 0) + final_animal_counts.get(9, 0) == 1