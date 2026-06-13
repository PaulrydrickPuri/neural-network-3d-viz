import pytest
from pathlib import Path


def test_load_yolo_model_returns_none_for_missing_file():
    """Test that load_yolo_model returns None when file doesn't exist."""
    from viz_tool.model_loader import load_yolo_model

    result = load_yolo_model("/nonexistent/path/model.pt")
    assert result is None


def test_load_yolo_model_returns_ultralytics_instance(tmp_path):
    """Test that load_yolo_model returns YOLO instance for valid .pt file."""
    from viz_tool.model_loader import load_yolo_model
    from ultralytics import YOLO

    # Create a minimal YOLO model for testing
    model = YOLO("yolo11n.pt")  # Load small pretrained model
    model_path = tmp_path / "test_model.pt"
    model.save(str(model_path))

    # Test loading
    loaded_model = load_yolo_model(str(model_path))
    assert loaded_model is not None
    assert isinstance(loaded_model, YOLO)
