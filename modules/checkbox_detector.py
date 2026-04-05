import numpy as np


try:
    import cv2
except Exception:
    cv2 = None


def detect_checkbox_state(roi, checked_threshold: float = 0.15) -> str:
    """Return 'checked' or 'unchecked' based on filled pixel density."""
    if roi is None or roi.size == 0:
        return "unknown"

    if cv2 is None:
        # Fallback to a simple grayscale threshold without OpenCV.
        gray = roi.mean(axis=2) if roi.ndim == 3 else roi
        binary = (gray < 180).astype(np.uint8)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        binary = (binary > 0).astype(np.uint8)

    fill_ratio = float(binary.sum()) / float(binary.size)
    return "checked" if fill_ratio >= checked_threshold else "unchecked"
