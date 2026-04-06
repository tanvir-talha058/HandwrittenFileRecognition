from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRConfig:
    use_paddle: bool = True
    use_easyocr: bool = False
    use_tesseract: bool = False
    strict_paddle: bool = True
    language: str = "en"
    paddle_device: str = "cpu"
    paddle_enable_mkldnn: bool = False
    paddle_ocr_version: str = "PP-OCRv5"
    paddle_use_doc_orientation_classify: bool = False
    paddle_use_doc_unwarping: bool = False
    paddle_use_textline_orientation: bool = False
    paddle_disable_model_source_check: bool = True
    paddle_doc_orientation_model_dir: Path | None = None
    paddle_doc_unwarping_model_dir: Path | None = None
    paddle_text_detection_model_dir: Path | None = None
    paddle_textline_orientation_model_dir: Path | None = None
    paddle_text_recognition_model_dir: Path | None = None


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent
    uploads_dir: Path = project_root / "uploads"
    outputs_dir: Path = project_root / "outputs"
    paddle_cache_dir: Path = project_root / ".paddlex_cache"
    default_field_map: Path = project_root / "field_maps" / "ucb_template.json"
    ocr: OCRConfig = field(default_factory=OCRConfig)

    def __post_init__(self):
        model_root = self.paddle_cache_dir / "official_models"
        detection_dir = model_root / "PP-OCRv5_server_det"
        recognition_dir = model_root / "PP-OCRv5_server_rec"
        if self.ocr.paddle_text_detection_model_dir is None and detection_dir.exists():
            self.ocr.paddle_text_detection_model_dir = detection_dir
        if self.ocr.paddle_text_recognition_model_dir is None and recognition_dir.exists():
            self.ocr.paddle_text_recognition_model_dir = recognition_dir
