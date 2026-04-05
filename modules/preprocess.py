from typing import Optional

import numpy as np


try:
    import cv2
except Exception:
    cv2 = None


def pil_to_bgr(image):
    """Convert PIL image to OpenCV BGR ndarray."""
    arr = np.array(image)
    if arr.ndim == 2:
        return arr
    return arr[:, :, ::-1]


def preprocess_roi(roi: np.ndarray, denoise: bool = True) -> np.ndarray:
    """Basic ROI preprocessing for OCR and binary analysis."""
    if cv2 is None:
        return roi

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

    if denoise:
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # OTSU threshold helps both text visibility and checkbox density logic.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def clamp_bbox(bbox, width: int, height: int):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(1, min(int(x2), width))
    y2 = max(1, min(int(y2), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def crop_bbox(image_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, w, h)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi
