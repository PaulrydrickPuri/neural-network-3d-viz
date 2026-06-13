import pytest
import numpy as np
from pathlib import Path


def test_load_image_returns_correct_shape(tmp_path):
    """Test load_image returns (H, W, 3) array normalized to [0, 1]."""
    # Create dummy image
    import cv2
    img = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    img_path = tmp_path / "test.jpg"
    cv2.imwrite(str(img_path), img)

    from viz_tool.utils import load_image
    result = load_image(str(img_path), target_size=640)

    assert result.shape == (640, 640, 3)
    assert result.dtype == np.float32
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_get_image_paths_returns_list(tmp_path):
    """Test get_image_paths returns list of image paths."""
    # Create dummy images
    import cv2
    for i in range(5):
        img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"img{i}.jpg"), img)

    from viz_tool.utils import get_image_paths
    paths = get_image_paths(str(tmp_path), max_images=3)

    assert isinstance(paths, list)
    assert len(paths) == 3
    assert all(p.endswith('.jpg') for p in paths)
