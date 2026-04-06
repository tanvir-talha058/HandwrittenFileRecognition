import json
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional


try:
    import pandas as pd
except Exception:
    pd = None

try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    PdfReader = None
    PdfWriter = None

try:
    from reportlab.pdfgen import canvas
except Exception:
    canvas = None


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


def build_form_field_aliases(field_map: Dict) -> Dict[str, str]:
    """Map PDF field names or aliases to extracted field keys."""
    aliases: Dict[str, str] = {}

    for field in field_map.get("fields", []):
        source_name = field.get("name")
        if not source_name:
            continue

        for alias_key in ("pdf_field_name", "form_field_name", "target_name"):
            alias = field.get(alias_key)
            if alias:
                aliases[str(alias)] = source_name

        for alias in field.get("form_field_aliases", []) or []:
            aliases[str(alias)] = source_name

    return aliases


def save_json(parsed: Dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=False, indent=2)


def save_text(text: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


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


def _normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _non_empty_text(value) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def _get_pdf_form_field_names(template_pdf_path: Path) -> list[str]:
    if PdfReader is None:
        raise RuntimeError("PDF form dependencies are missing. Install pypdf.")

    reader = PdfReader(str(template_pdf_path))
    form_fields = reader.get_fields() or {}
    return list(form_fields.keys())


def _match_pdf_form_values(
    parsed: Dict,
    pdf_field_names: Iterable[str],
    field_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    flat_output = flatten_output(parsed)
    direct_values = {
        key: text
        for key, value in flat_output.items()
        if (text := _non_empty_text(value)) is not None
    }

    normalized_values = {}
    for key, value in direct_values.items():
        normalized_values.setdefault(_normalize_field_name(key), value)

    matched: Dict[str, str] = {}
    alias_map = field_aliases or {}
    for pdf_field_name in pdf_field_names:
        value = direct_values.get(pdf_field_name)

        if value is None:
            source_key = alias_map.get(pdf_field_name)
            if source_key:
                value = direct_values.get(source_key)

        if value is None:
            value = normalized_values.get(_normalize_field_name(pdf_field_name))

        if value is not None:
            matched[pdf_field_name] = value

    return matched


def _save_fillable_pdf_form(
    parsed: Dict,
    template_pdf_path: Path,
    out_pdf_path: Path,
    field_aliases: Optional[Mapping[str, str]] = None,
) -> Dict[str, int | str]:
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("PDF fill dependencies are missing. Install pypdf.")

    if not template_pdf_path.exists():
        raise FileNotFoundError(f"Template PDF not found: {template_pdf_path}")

    form_field_names = _get_pdf_form_field_names(template_pdf_path)
    if not form_field_names:
        raise ValueError(
            "The provided PDF has no fillable form fields. Upload a fillable PDF or use the built-in mapped template."
        )

    matched_values = _match_pdf_form_values(parsed, form_field_names, field_aliases)
    if not matched_values:
        raise ValueError(
            "The provided PDF is fillable, but none of its field names matched the extracted keys. "
            "Use field names like full_name, dob, or mobile, or add aliases in the field map."
        )

    reader = PdfReader(str(template_pdf_path))
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    writer.set_need_appearances_writer()
    writer.update_page_form_field_values(
        list(writer.pages),
        matched_values,
        auto_regenerate=False,
    )

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf_path.open("wb") as out_handle:
        writer.write(out_handle)

    return {
        "fill_mode": "fillable_pdf_form",
        "filled_field_count": len(matched_values),
        "available_pdf_field_count": len(form_field_names),
    }


def _save_overlay_pdf(
    parsed: Dict,
    template_pdf_path: Path,
    out_pdf_path: Path,
    field_map: Dict,
    page_image_sizes: Dict[int, tuple],
) -> Dict[str, int | str]:
    """Render extracted text values onto a template PDF using field bbox coordinates."""
    if PdfReader is None or PdfWriter is None or canvas is None:
        raise RuntimeError("PDF fill dependencies are missing. Install pypdf and reportlab.")

    if not template_pdf_path.exists():
        raise FileNotFoundError(f"Template PDF not found: {template_pdf_path}")

    reader = PdfReader(str(template_pdf_path))
    writer = PdfWriter()

    fields = field_map.get("fields", [])
    field_values = parsed.get("fields", {})

    fields_by_page = {}
    for field in fields:
        page_no = int(field.get("page", 1))
        fields_by_page.setdefault(page_no, []).append(field)

    rendered_count = 0

    for idx, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)

        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=(width, height))
        c.setFont("Helvetica", 9)

        img_w, img_h = page_image_sizes.get(idx, (1024, 1448))
        sx = width / float(img_w)
        sy = height / float(img_h)

        for field in fields_by_page.get(idx, []):
            name = field.get("name")
            value = field_values.get(name)
            if value is None:
                continue

            text = str(value).strip()
            if not text:
                continue

            x1, y1, x2, y2 = field.get("bbox", [0, 0, 0, 0])
            x_pdf = float(x1) * sx
            # Convert top-left image coordinates to PDF bottom-left coordinates.
            y_pdf = height - (float(y2) * sy) + 2.0

            max_width = max(40.0, float(x2 - x1) * sx)
            clipped = text
            if c.stringWidth(clipped, "Helvetica", 9) > max_width:
                while clipped and c.stringWidth(clipped + "...", "Helvetica", 9) > max_width:
                    clipped = clipped[:-1]
                clipped = (clipped + "...") if clipped else ""

            c.drawString(x_pdf, y_pdf, clipped)
            rendered_count += 1

        c.save()
        packet.seek(0)

        overlay_reader = PdfReader(packet)
        overlay_page = overlay_reader.pages[0]
        page.merge_page(overlay_page)
        writer.add_page(page)

    out_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf_path.open("wb") as out_handle:
        writer.write(out_handle)

    return {
        "fill_mode": "mapped_overlay",
        "filled_field_count": rendered_count,
    }


def save_filled_pdf(
    parsed: Dict,
    template_pdf_path: Path,
    out_pdf_path: Path,
    field_map: Optional[Dict] = None,
    page_image_sizes: Optional[Dict[int, tuple]] = None,
    mode: str = "auto",
    field_aliases: Optional[Mapping[str, str]] = None,
):
    if mode not in {"auto", "form", "overlay"}:
        raise ValueError(f"Unsupported PDF fill mode: {mode}")

    if mode == "form":
        return _save_fillable_pdf_form(
            parsed,
            template_pdf_path=template_pdf_path,
            out_pdf_path=out_pdf_path,
            field_aliases=field_aliases,
        )

    if mode == "auto":
        form_field_names = _get_pdf_form_field_names(template_pdf_path)
        if form_field_names:
            return _save_fillable_pdf_form(
                parsed,
                template_pdf_path=template_pdf_path,
                out_pdf_path=out_pdf_path,
                field_aliases=field_aliases,
            )

    if field_map is None or page_image_sizes is None:
        raise ValueError(
            "Field map data and page image sizes are required for overlay PDF filling."
        )

    return _save_overlay_pdf(
        parsed,
        template_pdf_path=template_pdf_path,
        out_pdf_path=out_pdf_path,
        field_map=field_map,
        page_image_sizes=page_image_sizes,
    )
