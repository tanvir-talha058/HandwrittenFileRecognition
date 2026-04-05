import json
from pathlib import Path
from typing import Dict, List

from .preprocess import crop_bbox


class TemplateMapper:
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.template = self._load_template(template_path)

    @staticmethod
    def _load_template(template_path: Path) -> Dict:
        if not template_path.exists():
            raise FileNotFoundError(f"Template map not found: {template_path}")
        with template_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def get_fields(self) -> List[Dict]:
        return self.template.get("fields", [])

    def get_table_regions(self) -> List[Dict]:
        return self.template.get("tables", [])

    def get_signature_regions(self) -> List[Dict]:
        return self.template.get("signatures", [])

    def get_stamp_regions(self) -> List[Dict]:
        return self.template.get("stamps", [])

    def extract_region(self, page_bgr, bbox):
        return crop_bbox(page_bgr, bbox)
