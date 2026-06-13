import pytest
import plotly.graph_objects as go
import numpy as np


def test_create_3d_scatter_returns_figure():
    """Test 3D scatter plot creation."""
    from viz_tool.visualization import create_3d_feature_space

    coords = np.random.randn(100, 3)
    labels = np.random.randint(0, 5, 100)
    class_names = ["class_a", "class_b", "class_c", "class_d", "class_e"]

    fig = create_3d_feature_space(coords, labels, class_names)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 5  # One trace per class


def test_create_gradcam_overlay_returns_figure():
    """Test Grad-CAM overlay visualization."""
    from viz_tool.visualization import create_gradcam_overlay

    image = np.random.rand(640, 640, 3)
    heatmap = np.random.rand(20, 20)

    fig = create_gradcam_overlay(image, heatmap)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Image + heatmap overlay


def test_create_training_metrics_returns_figure():
    """Test training metrics chart creation."""
    from viz_tool.visualization import create_training_metrics

    epochs = list(range(1, 11))
    loss = [1.0 - i * 0.05 for i in range(10)]
    map50 = [0.5 + i * 0.03 for i in range(10)]

    fig = create_training_metrics(epochs, loss, map50)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Loss + mAP traces
