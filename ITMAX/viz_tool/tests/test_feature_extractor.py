import pytest
import numpy as np


def test_project_to_3d_returns_correct_shape():
    """Test that project_to_3d returns (N, 3) array."""
    from viz_tool.feature_extractor import project_to_3d

    # Create dummy embeddings
    embeddings = np.random.randn(100, 512)

    # Project to 3D
    coords_3d = project_to_3d(embeddings)

    assert coords_3d.shape == (100, 3)
    assert coords_3d.dtype == np.float32 or coords_3d.dtype == np.float64


def test_feature_extractor_initialization():
    """Test that FeatureExtractor can be initialized with a YOLO model."""
    from viz_tool.feature_extractor import FeatureExtractor
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    extractor = FeatureExtractor(model)
    assert extractor is not None
    assert extractor.model is not None
