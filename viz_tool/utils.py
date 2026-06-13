import cv2
import numpy as np
from pathlib import Path
from typing import List


def load_image(image_path: str, target_size: int = 640) -> np.ndarray:
    """Load and preprocess image for YOLO.

    Args:
        image_path: Path to image
        target_size: Target size (will be resized and padded)

    Returns:
        Preprocessed image (H, W, 3) normalized to [0, 1]
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize with padding
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))

    # Pad
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # Normalize
    img = img.astype(np.float32) / 255.0
    return img


def get_image_paths(dataset_path: str, max_images: int = 100) -> List[str]:
    """Get list of image paths from dataset.

    Args:
        dataset_path: Path to dataset directory
        max_images: Maximum number of images to return

    Returns:
        List of image paths
    """
    path = Path(dataset_path)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [str(p) for p in path.rglob('*') if p.suffix.lower() in extensions]
    return images[:max_images]


def load_training_logs(log_path: str) -> dict:
    """Load training logs from CSV file.

    Args:
        log_path: Path to results.csv

    Returns:
        Dictionary with epoch, loss, map50 lists
    """
    import pandas as pd
    df = pd.read_csv(log_path)
    return {
        'epochs': df['epoch'].tolist(),
        'loss': df['train/box_loss'].tolist(),
        'map50': df['metrics/mAP50(B)'].tolist()
    }
