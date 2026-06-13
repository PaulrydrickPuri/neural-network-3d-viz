import torch
import numpy as np
from typing import Tuple
from ultralytics import YOLO


class GradCAMExtractor:
    """Extract Grad-CAM heatmaps from YOLO model."""

    def __init__(self, model: YOLO, target_layer: str = "model.22"):
        self.model = model.model
        self.target_layer = self._get_layer(target_layer)
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _get_layer(self, layer_name: str):
        """Get layer by name."""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        # If layer not found, use last layer as fallback
        layers = list(self.model.named_modules())
        return layers[-1][1]

    def _register_hooks(self):
        """Register forward and backward hooks."""
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def compute_heatmap(self, image: np.ndarray, class_idx: int) -> np.ndarray:
        """Compute Grad-CAM heatmap for given image and class.

        Args:
            image: Input image (H, W, 3) normalized to [0, 1]
            class_idx: Target class index

        Returns:
            Heatmap (H, W) normalized to [0, 1]
        """
        # Forward pass
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        output = self.model(input_tensor)

        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        heatmap = heatmap.squeeze().cpu().numpy()

        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap


def compute_gradcam(model: YOLO, image: np.ndarray, class_idx: int) -> np.ndarray:
    """Convenience function to compute Grad-CAM."""
    extractor = GradCAMExtractor(model)
    return extractor.compute_heatmap(image, class_idx)
