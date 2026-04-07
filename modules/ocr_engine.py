import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .preprocess import preprocess_roi


def _ensure_langchain_docstore_compat() -> None:
    """Provide a minimal langchain.docstore alias for paddlex on langchain 1.x."""
    if (
        "langchain.docstore.document" in sys.modules
        and "langchain.text_splitter" in sys.modules
    ):
        return

    try:
        from langchain_core.documents import Document
    except Exception:
        return

    docstore_module = sys.modules.get("langchain.docstore")
    if docstore_module is None:
        docstore_module = types.ModuleType("langchain.docstore")
        sys.modules["langchain.docstore"] = docstore_module

    if "langchain.docstore.document" not in sys.modules:
        document_module = types.ModuleType("langchain.docstore.document")
        document_module.Document = Document
        sys.modules["langchain.docstore.document"] = document_module
        docstore_module.document = document_module

    if "langchain.text_splitter" not in sys.modules:
        try:
            from langchain_text_splitters import (
                CharacterTextSplitter,
                RecursiveCharacterTextSplitter,
            )

            text_splitter_module = types.ModuleType("langchain.text_splitter")
            text_splitter_module.CharacterTextSplitter = CharacterTextSplitter
            text_splitter_module.RecursiveCharacterTextSplitter = (
                RecursiveCharacterTextSplitter
            )
            sys.modules["langchain.text_splitter"] = text_splitter_module
        except Exception:
            pass


class HybridOCREngine:
    def __init__(
        self,
        language: str = "en",
        use_paddle: bool = True,
        use_easy: bool = True,
        use_tesseract: bool = True,
        strict_paddle: bool = False,
        paddle_cache_dir: Optional[Path] = None,
        paddle_device: str = "cpu",
        paddle_enable_mkldnn: bool = False,
        paddle_ocr_version: str = "PP-OCRv5",
        paddle_use_doc_orientation_classify: bool = False,
        paddle_use_doc_unwarping: bool = False,
        paddle_use_textline_orientation: bool = False,
        paddle_disable_model_source_check: bool = True,
        paddle_doc_orientation_model_dir: Optional[Path] = None,
        paddle_doc_unwarping_model_dir: Optional[Path] = None,
        paddle_text_detection_model_dir: Optional[Path] = None,
        paddle_textline_orientation_model_dir: Optional[Path] = None,
        paddle_text_recognition_model_dir: Optional[Path] = None,
    ):
        self.language = language
        self.use_paddle = use_paddle
        self.use_easy = use_easy
        self.use_tesseract = use_tesseract
        self.strict_paddle = strict_paddle
        self.paddle_cache_dir = paddle_cache_dir
        self.paddle_device = paddle_device
        self.paddle_enable_mkldnn = paddle_enable_mkldnn
        self.paddle_ocr_version = paddle_ocr_version
        self.paddle_use_doc_orientation_classify = paddle_use_doc_orientation_classify
        self.paddle_use_doc_unwarping = paddle_use_doc_unwarping
        self.paddle_use_textline_orientation = paddle_use_textline_orientation
        self.paddle_disable_model_source_check = paddle_disable_model_source_check
        self.paddle_doc_orientation_model_dir = paddle_doc_orientation_model_dir
        self.paddle_doc_unwarping_model_dir = paddle_doc_unwarping_model_dir
        self.paddle_text_detection_model_dir = paddle_text_detection_model_dir
        self.paddle_textline_orientation_model_dir = paddle_textline_orientation_model_dir
        self.paddle_text_recognition_model_dir = paddle_text_recognition_model_dir

        self._paddle = None
        self._easy = None
        self._tesseract_available = False
        self._paddle_init_error = None

        self._init_engines()

    def _init_engines(self):
        if self.use_paddle:
            try:
                if self.paddle_disable_model_source_check:
                    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                if self.paddle_cache_dir:
                    Path(self.paddle_cache_dir).mkdir(parents=True, exist_ok=True)
                    os.environ["PADDLE_PDX_CACHE_HOME"] = str(self.paddle_cache_dir)
                local_font_path = self._default_paddle_font_path()
                if local_font_path:
                    os.environ.setdefault("PADDLE_PDX_LOCAL_FONT_FILE_PATH", local_font_path)

                _ensure_langchain_docstore_compat()

                from paddleocr import PaddleOCR

                paddle_kwargs = {
                    "lang": self.language,
                    "device": self.paddle_device,
                    "enable_mkldnn": self.paddle_enable_mkldnn,
                    "ocr_version": self.paddle_ocr_version,
                    "use_doc_orientation_classify": self.paddle_use_doc_orientation_classify,
                    "use_doc_unwarping": self.paddle_use_doc_unwarping,
                    "use_textline_orientation": self.paddle_use_textline_orientation,
                }

                optional_model_dirs = {
                    "doc_orientation_classify": self.paddle_doc_orientation_model_dir,
                    "doc_unwarping": self.paddle_doc_unwarping_model_dir,
                    "text_detection": self.paddle_text_detection_model_dir,
                    "textline_orientation": self.paddle_textline_orientation_model_dir,
                    "text_recognition": self.paddle_text_recognition_model_dir,
                }
                for key, value in optional_model_dirs.items():
                    if value:
                        value_path = Path(value)
                        paddle_kwargs[f"{key}_model_dir"] = str(value_path)
                        paddle_kwargs[f"{key}_model_name"] = value_path.name

                self._paddle = PaddleOCR(**paddle_kwargs)
            except Exception as exc:
                self._paddle = None
                self._paddle_init_error = f"{type(exc).__name__}: {exc}"
                if self.strict_paddle:
                    raise RuntimeError(self._paddle_unavailable_message()) from exc

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

    def _paddle_unavailable_message(self) -> str:
        details = self._paddle_init_error or "unknown initialization error"
        return (
            "PaddleOCR 3.x could not initialize. Install `paddlepaddle>=3.0.0` in the "
            "active environment and make sure the PP-OCR models are available locally or "
            f"downloadable. Details: {details}"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.replace("\n", " ").split())

    @staticmethod
    def _polygon_from_rect(x: int, y: int, w: int, h: int) -> List[List[int]]:
        return [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ]

    @staticmethod
    def _detection_sort_key(item: Dict):
        bbox = item.get("bbox") or [[0, 0]]
        y = min(point[1] for point in bbox)
        x = min(point[0] for point in bbox)
        return (round(y / 10), x)

    @staticmethod
    def _default_paddle_font_path() -> Optional[str]:
        candidates = [
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return None

    @staticmethod
    def _to_sequence(value) -> List:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if hasattr(value, "tolist"):
            converted = value.tolist()
            return converted if isinstance(converted, list) else [converted]
        return [value]

    @staticmethod
    def _prepare_image_for_paddle(image: np.ndarray) -> np.ndarray:
        prepared = np.asarray(image)
        if prepared.dtype != np.uint8:
            prepared = np.clip(prepared, 0, 255).astype(np.uint8)
        if prepared.ndim == 2:
            prepared = np.stack([prepared] * 3, axis=-1)
        elif prepared.ndim == 3 and prepared.shape[2] == 4:
            prepared = prepared[:, :, :3]
        return prepared

    def _coerce_polygon(self, polygon) -> Optional[List[List[int]]]:
        if polygon is None:
            return None

        arr = np.asarray(polygon)
        if arr.size == 0:
            return None

        if arr.ndim == 1:
            if arr.size >= 8 and arr.size % 2 == 0:
                arr = arr.reshape(-1, 2)
            elif arr.size >= 4:
                x1, y1, x2, y2 = [int(round(float(value))) for value in arr[:4]]
                return self._polygon_from_rect(x1, y1, max(1, x2 - x1), max(1, y2 - y1))
            else:
                return None

        if arr.ndim >= 2:
            points = arr.reshape(-1, arr.shape[-1])
            if points.shape[1] >= 2 and len(points) >= 4:
                return [
                    [int(round(float(x))), int(round(float(y)))]
                    for x, y in points[:4, :2]
                ]

        return None

    @staticmethod
    def _extract_paddle_payload(item) -> Dict:
        if item is None:
            return {}
        if hasattr(item, "json"):
            try:
                payload = item.json
                if isinstance(payload, dict):
                    return payload.get("res", payload)
            except Exception:
                pass
        if isinstance(item, dict):
            return item.get("res", item)
        return {}

    @staticmethod
    def _unique_variants(*variants):
        seen = set()
        unique = []
        for variant in variants:
            if variant is None:
                continue
            marker = (variant.shape, variant.dtype.str, variant.tobytes()[:128])
            if marker in seen:
                continue
            seen.add(marker)
            unique.append(variant)
        return unique

    def _roi_variants(self, roi: np.ndarray):
        return self._unique_variants(
            roi,
            preprocess_roi(roi, denoise=True, binarize=False),
            preprocess_roi(roi, denoise=True, binarize=True),
        )

    def _page_variants(self, image: np.ndarray):
        return self._unique_variants(
            image,
            preprocess_roi(image, denoise=True, binarize=False),
        )

    def read_page(self, image: np.ndarray) -> List[Dict]:
        if image is None or image.size == 0:
            return []

        for variant in self._page_variants(image):
            detections = self._read_page_with_paddle(variant)
            if detections:
                return detections

        for variant in self._page_variants(image):
            detections = self._read_page_with_easy(variant)
            if detections:
                return detections

        for variant in self._page_variants(image):
            detections = self._read_page_with_tesseract(variant)
            if detections:
                return detections

        return []

    def read_region(self, roi: np.ndarray) -> Dict:
        if roi is None or roi.size == 0:
            return {
                "text": "",
                "confidence": 0.0,
                "detections": [],
                "engine": None,
            }

        for variant in self._roi_variants(roi):
            detections = self._read_page_with_paddle(variant)
            if detections:
                return self._summarize_region(detections, "paddle")

        for variant in self._roi_variants(roi):
            detections = self._read_page_with_easy(variant)
            if detections:
                return self._summarize_region(detections, "easyocr")

        for variant in self._roi_variants(roi):
            detections = self._read_page_with_tesseract(variant)
            if detections:
                return self._summarize_region(detections, "tesseract")

        return {
            "text": "",
            "confidence": 0.0,
            "detections": [],
            "engine": None,
        }

    def read_text(self, roi: np.ndarray) -> str:
        return self.read_region(roi).get("text", "")

    def _summarize_region(self, detections: List[Dict], engine: str) -> Dict:
        ordered = sorted(detections, key=self._detection_sort_key)
        text = self._normalize(" ".join(item["text"] for item in ordered if item.get("text")))
        confidences = [float(item.get("confidence", 0.0)) for item in ordered if item.get("text")]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": text,
            "confidence": round(confidence, 4),
            "detections": ordered,
            "engine": engine,
        }

    def _parse_paddle_predict_results(self, results) -> List[Dict]:
        detections = []

        for page in results or []:
            payload = self._extract_paddle_payload(page)
            texts = self._to_sequence(payload.get("rec_texts"))
            scores = self._to_sequence(payload.get("rec_scores"))
            polygons = self._to_sequence(payload.get("rec_polys")) or self._to_sequence(payload.get("dt_polys"))
            boxes = self._to_sequence(payload.get("rec_boxes"))

            for idx, raw_text in enumerate(texts):
                text = self._normalize(str(raw_text or ""))
                if not text:
                    continue

                polygon = None
                if idx < len(polygons):
                    polygon = self._coerce_polygon(polygons[idx])
                if polygon is None and idx < len(boxes):
                    polygon = self._coerce_polygon(boxes[idx])
                if polygon is None:
                    continue

                try:
                    confidence = float(scores[idx]) if idx < len(scores) else 0.0
                except Exception:
                    confidence = 0.0

                detections.append(
                    {
                        "text": text,
                        "confidence": confidence,
                        "bbox": polygon,
                        "engine": "paddle",
                    }
                )

        return detections

    def _parse_paddle_legacy_results(self, results) -> List[Dict]:
        detections = []

        for block in results or []:
            for item in block or []:
                if len(item) < 2 or not item[1]:
                    continue
                polygon = [[int(round(x)), int(round(y))] for x, y in item[0]]
                text, confidence = item[1]
                text = self._normalize(text or "")
                if not text:
                    continue
                detections.append(
                    {
                        "text": text,
                        "confidence": float(confidence or 0.0),
                        "bbox": polygon,
                        "engine": "paddle",
                    }
                )

        return detections

    def _read_page_with_paddle(self, image: np.ndarray) -> List[Dict]:
        if self._paddle is None:
            return []

        prepared = self._prepare_image_for_paddle(image)

        try:
            return self._parse_paddle_predict_results(self._paddle.predict(prepared))
        except Exception as exc:
            try:
                legacy_result = self._paddle.ocr(prepared, cls=False)
            except Exception:
                if self.strict_paddle:
                    raise RuntimeError(f"PaddleOCR 3.x inference failed: {exc}") from exc
                return []

        detections = self._parse_paddle_predict_results(legacy_result)
        if detections:
            return detections
        return self._parse_paddle_legacy_results(legacy_result)

    def _read_page_with_easy(self, image: np.ndarray) -> List[Dict]:
        if self._easy is None:
            return []
        try:
            result = self._easy.readtext(image, detail=1, paragraph=False)
        except Exception:
            return []

        detections = []
        for item in result or []:
            if len(item) < 3:
                continue
            bbox, text, confidence = item[:3]
            text = self._normalize(text or "")
            if not text:
                continue
            polygon = [[int(round(x)), int(round(y))] for x, y in bbox]
            detections.append(
                {
                    "text": text,
                    "confidence": float(confidence or 0.0),
                    "bbox": polygon,
                    "engine": "easyocr",
                }
            )
        return detections

    def _read_page_with_tesseract(self, image: np.ndarray) -> List[Dict]:
        if not self._tesseract_available:
            return []
        try:
            import pytesseract

            result = pytesseract.image_to_data(
                image,
                config="--psm 6",
                output_type=pytesseract.Output.DICT,
            )
        except Exception:
            return []

        grouped: Dict[tuple, Dict] = {}
        count = len(result.get("text", []))
        for idx in range(count):
            text = self._normalize(result["text"][idx] or "")
            try:
                confidence = float(result["conf"][idx])
            except Exception:
                confidence = -1.0

            if not text or confidence < 0:
                continue

            x = int(result["left"][idx])
            y = int(result["top"][idx])
            w = int(result["width"][idx])
            h = int(result["height"][idx])
            key = (
                result.get("block_num", [0])[idx],
                result.get("par_num", [0])[idx],
                result.get("line_num", [0])[idx],
            )
            line = grouped.setdefault(
                key,
                {
                    "texts": [],
                    "confidences": [],
                    "x1": x,
                    "y1": y,
                    "x2": x + w,
                    "y2": y + h,
                },
            )
            line["texts"].append((x, text))
            line["confidences"].append(confidence)
            line["x1"] = min(line["x1"], x)
            line["y1"] = min(line["y1"], y)
            line["x2"] = max(line["x2"], x + w)
            line["y2"] = max(line["y2"], y + h)

        detections = []
        for line in grouped.values():
            ordered_words = [word for _, word in sorted(line["texts"], key=lambda item: item[0])]
            text = self._normalize(" ".join(ordered_words))
            if not text:
                continue
            bbox = self._polygon_from_rect(
                line["x1"],
                line["y1"],
                line["x2"] - line["x1"],
                line["y2"] - line["y1"],
            )
            detections.append(
                {
                    "text": text,
                    "confidence": sum(line["confidences"]) / len(line["confidences"]),
                    "bbox": bbox,
                    "engine": "tesseract",
                }
            )
        return detections
