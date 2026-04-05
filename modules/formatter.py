import json
from pathlib import Path
from typing import Dict


try:
    import pandas as pd
except Exception:
    pd = None


def flatten_output(parsed: Dict) -> Dict:
    flat = {}

    for key, value in parsed.get("fields", {}).items():
        flat[key] = value

    for table_name, fields in parsed.get("tables", {}).items():
        for key, value in fields.items():
            flat[f"{table_name}.{key}"] = value

    for key, value in parsed.get("signatures", {}).items():
        flat[f"signature.{key}"] = value

    for key, value in parsed.get("stamps", {}).items():
        flat[f"stamp.{key}"] = value

    return flat


def save_json(parsed: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)


def save_excel(parsed: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if pd is None:
        # Write a companion JSON if pandas/openpyxl is unavailable.
        fallback = out_path.with_suffix(".json")
        save_json({"note": "Excel export fallback", "data": flatten_output(parsed)}, fallback)
        return

    flat = flatten_output(parsed)
    df = pd.DataFrame([flat])
    df.to_excel(out_path, index=False)


def save_filled_pdf_placeholder(parsed: Dict, out_path: Path):
    """Placeholder for template re-rendering; writes a simple text artifact for now."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_output(parsed)
    text_path = out_path.with_suffix(".txt")
    lines = ["Auto-filled PDF placeholder", "", "Extracted values:"]
    lines.extend([f"- {k}: {v}" for k, v in flat.items()])
    text_path.write_text("\n".join(lines), encoding="utf-8")
