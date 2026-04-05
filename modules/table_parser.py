import re
from typing import Dict


def _to_number(token: str):
    cleaned = token.replace(",", "").strip()
    if not cleaned:
        return None
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    if re.fullmatch(r"-?\d+\.\d+", cleaned):
        return float(cleaned)
    return None


def parse_table_text(raw_text: str, schema: Dict[str, str]) -> Dict:
    """Parse row values using schema aliases from OCR text lines."""
    result = {}
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    for field_name, alias in schema.items():
        best_value = None
        for line in lines:
            if alias.lower() in line.lower():
                number_tokens = re.findall(r"-?[\d,]+(?:\.\d+)?", line)
                if number_tokens:
                    best_value = _to_number(number_tokens[-1])
                    break
        result[field_name] = best_value

    return result
