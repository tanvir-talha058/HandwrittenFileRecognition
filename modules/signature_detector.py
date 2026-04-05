import numpy as np


try:
    import cv2
except Exception:
    cv2 = None


def detect_signature_present(roi, ink_threshold: float = 0.03) -> bool:
    """Heuristic signature detection using ink coverage ratio."""
    if roi is None or roi.size == 0:
        return False

    if cv2 is None:
        gray = roi.mean(axis=2) if roi.ndim == 3 else roi
        ink = (gray < 160).astype(np.uint8)
    else:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        _, ink_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = (ink_mask > 0).astype(np.uint8)

    ratio = float(ink.sum()) / float(ink.size)
    return ratio >= ink_threshold


def signature_placeholder(present: bool) -> str:
    return "[SIGNED]" if present else "[NOT_SIGNED]"
