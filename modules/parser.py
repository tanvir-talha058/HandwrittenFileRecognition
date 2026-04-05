from typing import Dict, List

from .checkbox_detector import detect_checkbox_state
from .ocr_engine import HybridOCREngine
from .preprocess import preprocess_roi
from .signature_detector import detect_signature_present, signature_placeholder
from .stamp_detector import detect_stamp_present
from .table_parser import parse_table_text
from .template_mapper import TemplateMapper


class LoanFormParser:
    def __init__(self, mapper: TemplateMapper, ocr_engine: HybridOCREngine):
        self.mapper = mapper
        self.ocr = ocr_engine

    def parse(self, pages_bgr: List) -> Dict:
        output = {
            "fields": {},
            "tables": {},
            "signatures": {},
            "stamps": {},
            "meta": {
                "page_count": len(pages_bgr),
            },
        }

        self._parse_fields(pages_bgr, output)
        self._parse_tables(pages_bgr, output)
        self._parse_signatures(pages_bgr, output)
        self._parse_stamps(pages_bgr, output)

        return output

    def _get_page(self, pages_bgr, page_number: int):
        idx = max(0, page_number - 1)
        if idx >= len(pages_bgr):
            return None
        return pages_bgr[idx]

    def _parse_fields(self, pages_bgr, output: Dict):
        for field in self.mapper.get_fields():
            name = field["name"]
            page = int(field["page"])
            bbox = field["bbox"]
            field_type = field.get("type", "text")

            page_img = self._get_page(pages_bgr, page)
            if page_img is None:
                output["fields"][name] = None
                continue

            roi = self.mapper.extract_region(page_img, bbox)
            if roi is None:
                output["fields"][name] = None
                continue

            if field_type == "checkbox":
                state = detect_checkbox_state(roi)
                mapping = field.get("value_map", {})
                output["fields"][name] = mapping.get(state, state)
            else:
                proc = preprocess_roi(roi)
                output["fields"][name] = self.ocr.read_text(proc)

    def _parse_tables(self, pages_bgr, output: Dict):
        for table in self.mapper.get_table_regions():
            name = table["name"]
            page = int(table["page"])
            bbox = table["bbox"]
            schema = table.get("schema", {})

            page_img = self._get_page(pages_bgr, page)
            if page_img is None:
                output["tables"][name] = {}
                continue

            roi = self.mapper.extract_region(page_img, bbox)
            if roi is None:
                output["tables"][name] = {}
                continue

            text = self.ocr.read_text(roi)
            output["tables"][name] = parse_table_text(text, schema)

    def _parse_signatures(self, pages_bgr, output: Dict):
        for item in self.mapper.get_signature_regions():
            name = item["name"]
            page = int(item["page"])
            bbox = item["bbox"]

            page_img = self._get_page(pages_bgr, page)
            if page_img is None:
                output["signatures"][name] = signature_placeholder(False)
                continue

            roi = self.mapper.extract_region(page_img, bbox)
            present = detect_signature_present(roi)
            output["signatures"][name] = signature_placeholder(present)

    def _parse_stamps(self, pages_bgr, output: Dict):
        for item in self.mapper.get_stamp_regions():
            name = item["name"]
            page = int(item["page"])
            bbox = item["bbox"]

            page_img = self._get_page(pages_bgr, page)
            if page_img is None:
                output["stamps"][name] = "[NO_STAMP]"
                continue

            roi = self.mapper.extract_region(page_img, bbox)
            present = detect_stamp_present(roi)
            output["stamps"][name] = "[STAMP_DETECTED]" if present else "[NO_STAMP]"
