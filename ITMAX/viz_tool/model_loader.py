from pathlib import Path
from ultralytics import YOLO
from typing import Optional


def load_yolo_model(model_path: str) -> Optional[YOLO]:
    """Load YOLO model from .pt file.

    Args:
        model_path: Path to .pt file

    Returns:
        YOLO model instance or None if file not found
    """
    path = Path(model_path)
    if not path.exists():
        return None
    return YOLO(str(path))
