import numpy as np


try:
    import cv2
except Exception:
    cv2 = None


def detect_stamp_present(roi, min_ratio: float = 0.01) -> bool:
    """Detect probable blue/red official stamp regions by color mask ratio."""
    if roi is None or roi.size == 0:
        return False

    if roi.ndim != 3:
        return False

    if cv2 is None:
        # Basic RGB logic when OpenCV is unavailable.
        b = roi[:, :, 0]
        g = roi[:, :, 1]
        r = roi[:, :, 2]
        blue_mask = (b > 120) & (b > g + 20) & (b > r + 20)
        red_mask = (r > 120) & (r > g + 20) & (r > b + 20)
    else:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        blue_mask = cv2.inRange(hsv, (90, 60, 40), (140, 255, 255)) > 0
        red_mask1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255)) > 0
        red_mask2 = cv2.inRange(hsv, (160, 60, 40), (179, 255, 255)) > 0
        red_mask = red_mask1 | red_mask2

    mask = blue_mask | red_mask
    ratio = float(mask.sum()) / float(mask.size)
    return ratio >= min_ratio
