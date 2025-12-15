import os
from pathlib import Path

# Find the project root (assuming your test file is inside a 'tests' directory)
PROJECT_ROOT = Path(__file__).parent.parent 
SAMPLE_IMAGES_DIR = PROJECT_ROOT / "Sample_images"

def test_sample_images_directory_exists_and_is_not_empty():
    """
    Checks that the Sample_images directory exists and contains at least one entry.
    """
    # 1. Check if the directory exists and is a directory
    assert SAMPLE_IMAGES_DIR.is_dir(), \
        f"The directory does not exist at: {SAMPLE_IMAGES_DIR.resolve()}"
    
    # 2. Check if the directory is not empty
    has_files = any(SAMPLE_IMAGES_DIR.iterdir())
    
    assert has_files, \
        f"The directory is empty: {SAMPLE_IMAGES_DIR.resolve()}. Please add sample images."