from typing import Optional

import numpy as np


class HybridOCREngine:
    def __init__(self, language: str = "en", use_paddle: bool = True, use_easy: bool = True, use_tesseract: bool = True):
        self.language = language
        self.use_paddle = use_paddle
        self.use_easy = use_easy
        self.use_tesseract = use_tesseract

        self._paddle = None
        self._easy = None
        self._tesseract_available = False

        self._init_engines()

    def _init_engines(self):
        if self.use_paddle:
            try:
                from paddleocr import PaddleOCR

                self._paddle = PaddleOCR(use_angle_cls=False, lang=self.language)
            except Exception:
                self._paddle = None

        if self.use_easy:
            try:
                import easyocr

                self._easy = easyocr.Reader([self.language])
            except Exception:
                self._easy = None

        if self.use_tesseract:
            try:
                import pytesseract  # noqa: F401

                self._tesseract_available = True
            except Exception:
                self._tesseract_available = False

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\n", " ").split())

    def read_text(self, roi: np.ndarray) -> str:
        if roi is None or roi.size == 0:
            return ""

        text = self._read_with_paddle(roi)
        if text:
            return self._normalize(text)

        text = self._read_with_easy(roi)
        if text:
            return self._normalize(text)

        text = self._read_with_tesseract(roi)
        if text:
            return self._normalize(text)

        return ""

    def _read_with_paddle(self, roi: np.ndarray) -> Optional[str]:
        if self._paddle is None:
            return None
        try:
            result = self._paddle.ocr(roi, cls=False)
            lines = []
            for block in result or []:
                for item in block or []:
                    if len(item) >= 2 and item[1]:
                        lines.append(item[1][0])
            return " ".join(lines).strip()
        except Exception:
            return None

    def _read_with_easy(self, roi: np.ndarray) -> Optional[str]:
        if self._easy is None:
            return None
        try:
            result = self._easy.readtext(roi, detail=0)
            return " ".join(result).strip() if result else None
        except Exception:
            return None

    def _read_with_tesseract(self, roi: np.ndarray) -> Optional[str]:
        if not self._tesseract_available:
            return None
        try:
            import pytesseract

            text = pytesseract.image_to_string(roi, config="--psm 6")
            return text.strip() if text else None
        except Exception:
            return None
