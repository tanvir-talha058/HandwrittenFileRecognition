from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRConfig:
    use_paddle: bool = True
    use_easyocr: bool = True
    use_tesseract: bool = True
    language: str = "en"


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parent
    uploads_dir: Path = project_root / "uploads"
    outputs_dir: Path = project_root / "outputs"
    default_field_map: Path = project_root / "field_maps" / "ucb_template.json"
    ocr: OCRConfig = field(default_factory=OCRConfig)
