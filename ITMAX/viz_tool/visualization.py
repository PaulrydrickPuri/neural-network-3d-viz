import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import List


def create_3d_feature_space(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> go.Figure:
    """Create interactive 3D scatter plot of feature space.

    Args:
        coords: (N, 3) 3D coordinates
        labels: (N,) class labels
        class_names: List of class names

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        if mask.sum() > 0:  # Only add trace if class has samples
            fig.add_trace(go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode='markers',
                marker=dict(size=4, opacity=0.7),
                name=class_names[class_idx]
            ))

    fig.update_layout(
        title="3D Feature Space",
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3"
        )
    )
    return fig


def create_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> go.Figure:
    """Create Grad-CAM heatmap overlay.

    Args:
        image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Overlay opacity

    Returns:
        Plotly Figure
    """
    # Resize heatmap to image size
    from PIL import Image
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(
        (image.shape[1], image.shape[0]), Image.BILINEAR
    ))

    fig = go.Figure()
    fig.add_trace(go.Image(z=image))
    fig.add_trace(go.Heatmap(
        z=heatmap_resized,
        colorscale='jet',
        opacity=alpha,
        showscale=False
    ))
    fig.update_layout(title="Grad-CAM Overlay")
    return fig


def create_training_metrics(
    epochs: List[int],
    loss: List[float],
    map50: List[float]
) -> go.Figure:
    """Create training metrics dashboard.

    Args:
        epochs: List of epoch numbers
        loss: List of loss values
        map50: List of mAP@50 values

    Returns:
        Plotly Figure with subplots
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "mAP@50"))

    fig.add_trace(go.Scatter(x=epochs, y=loss, name="Loss"), row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=map50, name="mAP@50"), row=1, col=2)

    fig.update_layout(title="Training Metrics", height=400)
    return fig
