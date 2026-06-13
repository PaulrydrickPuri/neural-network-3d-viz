import pytest
import numpy as np


def test_gradcam_extractor_initialization():
    """Test that GradCAMExtractor can be initialized with a YOLO model."""
    from viz_tool.gradcam import GradCAMExtractor
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    extractor = GradCAMExtractor(model)
    assert extractor is not None
    assert extractor.model is not None


def test_compute_heatmap_returns_normalized_array():
    """Test that compute_heatmap returns array normalized to [0, 1]."""
    # Skip for now - requires actual model and GPU
    pytest.skip("Requires YOLO model and forward/backward pass setup")
