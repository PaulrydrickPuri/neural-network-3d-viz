import torch
import numpy as np
from typing import List, Tuple
from ultralytics import YOLO
from umap import UMAP


class FeatureExtractor:
    """Extract embeddings from YOLO model."""

    def __init__(self, model: YOLO, layer_name: str = "model.21"):
        self.model = model.model
        self.layer = self._get_layer(layer_name)
        self.features = None
        self._register_hook()

    def _get_layer(self, layer_name: str):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        # Fallback to second-to-last layer
        layers = list(self.model.named_modules())
        return layers[-2][1] if len(layers) > 1 else layers[-1][1]

    def _register_hook(self):
        self.layer.register_forward_hook(self._save_features)

    def _save_features(self, module, input, output):
        self.features = output.detach()

    def extract(self, images: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings for batch of images.

        Args:
            images: List of images (H, W, 3)

        Returns:
            Embeddings (N, D) where D is embedding dimension
        """
        embeddings = []
        for img in images:
            input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            _ = self.model(input_tensor)
            # Global average pooling
            feat = self.features.mean(dim=(2, 3)).squeeze().cpu().numpy()
            embeddings.append(feat)
        return np.stack(embeddings)


def project_to_3d(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 3D using UMAP.

    Args:
        embeddings: (N, D) embeddings

    Returns:
        (N, 3) 3D coordinates
    """
    umap = UMAP(n_components=3, random_state=42)
    return umap.fit_transform(embeddings)
