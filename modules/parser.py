import re
from typing import Dict, List

from .checkbox_detector import detect_checkbox_state
from .ocr_engine import HybridOCREngine
from .preprocess import clamp_bbox, preprocess_roi
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
        self._enrich_key_fields_from_page_text(pages_bgr, output)
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
                output["fields"][name] = self._read_text_with_bbox_expansion(page_img, bbox)

    def _read_text_with_bbox_expansion(self, page_img, bbox) -> str:
        h, w = page_img.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(1, int(x2 - x1))
        bh = max(1, int(y2 - y1))

        best_text = ""
        best_score = (-1, -1.0, -1)
        for pad_x_ratio, pad_y_ratio in [(0.0, 0.0), (0.12, 0.2), (0.22, 0.35)]:
            expanded = clamp_bbox(
                [
                    int(x1 - (bw * pad_x_ratio)),
                    int(y1 - (bh * pad_y_ratio)),
                    int(x2 + (bw * pad_x_ratio)),
                    int(y2 + (bh * pad_y_ratio)),
                ],
                w,
                h,
            )
            roi = self.mapper.extract_region(page_img, expanded)
            if roi is None:
                continue

            proc = preprocess_roi(roi)
            region = self.ocr.read_region(proc)
            text = str(region.get("text", "")).strip()
            confidence = float(region.get("confidence", 0.0) or 0.0)
            score = (1 if text else 0, confidence, len(text))
            if score > best_score:
                best_score = score
                best_text = text

        return best_text

    @staticmethod
    def _normalize_mobile(candidate: str) -> str:
        raw = (
            candidate.replace("O", "0")
            .replace("o", "0")
            .replace("I", "1")
            .replace("i", "1")
            .replace("l", "1")
        )
        digits = "".join(ch for ch in raw if ch.isdigit())
        if len(digits) < 9:
            return ""
        if len(digits) > 13:
            digits = digits[:13]
        return digits

    @staticmethod
    def _clean_text_value(value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value or "").strip(" .:-\n\t")
        return cleaned

    @staticmethod
    def _is_noisy_name(value: str) -> bool:
        text = (value or "").lower()
        if not text.strip():
            return True
        banned = ["business", "profession", "female", "male", "gender", "fomolo", "mot"]
        return any(token in text for token in banned)

    def _extract_page_text(self, page_img) -> str:
        detections = self.ocr.read_page(page_img)
        if not detections:
            return ""
        ordered = sorted(detections, key=self.ocr._detection_sort_key)
        return "\n".join(item.get("text", "") for item in ordered if item.get("text"))

    def _extract_full_name(self, text: str) -> str:
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        start = -1
        for idx, line in enumerate(lines):
            if re.search(r"full\s*n[ao]me", line, flags=re.IGNORECASE):
                start = idx
                break
        if start < 0:
            return ""

        tokens = []
        stop_words = {
            "profession",
            "gender",
            "gonoor",
            "date",
            "dote",
            "education",
            "marital",
            "status",
        }
        for line in lines[start + 1 : start + 6]:
            normalized = re.sub(r"[^a-z]", "", line.lower())
            if any(word in normalized for word in stop_words):
                break
            cleaned = re.sub(r"[^A-Za-z.\s]", " ", line)
            cleaned = self._clean_text_value(cleaned)
            if cleaned:
                tokens.append(cleaned)

        candidate = self._clean_text_value(" ".join(tokens))
        if len(candidate) < 3:
            return ""
        return candidate

    def _extract_dob(self, text: str) -> str:
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        area = ""
        for idx, line in enumerate(lines):
            if re.search(r"d[ao]te\s*of\s*birth", line, flags=re.IGNORECASE):
                area = " ".join(lines[idx : idx + 4])
                break
        if not area:
            area = text.replace("\n", " ")

        date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", area)
        if date_match:
            return date_match.group(0)

        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", area)
        if year_match:
            return year_match.group(1)

        compact = text.replace("\n", " ")
        around_birth = re.search(r"birth.{0,40}?(\d{3,4})", compact, flags=re.IGNORECASE)
        if around_birth:
            year_text = around_birth.group(1)
            if len(year_text) == 4:
                return year_text
            if len(year_text) == 3:
                return f"1{year_text}" if year_text.startswith("9") else f"2{year_text}"

        short_year = re.search(r"\b(\d{3})\b", area)
        if short_year:
            yr = short_year.group(1)
            if yr.startswith("9"):
                return f"1{yr}"
            return f"2{yr}"
        return ""

    def _extract_mobile(self, text: str) -> str:
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for idx, line in enumerate(lines):
            if re.search(r"m[oa]b\w*\s*no", line, flags=re.IGNORECASE):
                best = ""
                for probe in lines[idx : idx + 5]:
                    candidate = self._normalize_mobile(probe)
                    if len(candidate) in {10, 11} and candidate.startswith("01"):
                        return candidate
                    if len(candidate) > len(best):
                        best = candidate
                if best:
                    return best

        compact = text.replace("\n", " ")
        match = re.search(r"(01[0-9OIil\s\-]{8,14})", compact)
        if match:
            return self._normalize_mobile(match.group(1))
        return ""

    def _extract_business_name(self, text: str) -> str:
        if not text:
            return ""
        compact = text.replace("\n", " ")
        match = re.search(
            r"n[ao]me\s*of\s*(?:compa?n[yi]|business)\s*[:;]?\s*(.{2,80}?)(?:address|phone|mobile|profession|$)",
            compact,
            flags=re.IGNORECASE,
        )
        if not match:
            return ""
        value = self._clean_text_value(match.group(1))
        if len(value) < 3:
            return ""
        return value

    def _enrich_key_fields_from_page_text(self, pages_bgr, output: Dict):
        fields = output.get("fields", {})
        if not fields:
            return

        page1 = self._get_page(pages_bgr, 1)
        page2 = self._get_page(pages_bgr, 2)

        page1_text = self._extract_page_text(page1) if page1 is not None else ""
        page2_text = self._extract_page_text(page2) if page2 is not None else ""

        if self._is_noisy_name(str(fields.get("full_name") or "")):
            fields["full_name"] = self._extract_full_name(page1_text)

        if not fields.get("dob"):
            fields["dob"] = self._extract_dob(page1_text)

        mobile_value = self._normalize_mobile(str(fields.get("mobile") or ""))
        if not mobile_value or not mobile_value.startswith("01"):
            fields["mobile"] = self._extract_mobile(page1_text)

        business_name = str(fields.get("business_name") or "").strip().lower()
        if not business_name or "name of" in business_name or "compony" in business_name:
            extracted_business_name = self._extract_business_name(page2_text)
            if extracted_business_name:
                fields["business_name"] = extracted_business_name

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
