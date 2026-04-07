import json
import re
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

from config import AppConfig
from modules.document_loader import SUPPORTED_INPUT_SUFFIXES, is_supported_input_file
from modules.form_pipeline import PREVIEW_PREFIX, align_page_to_template, run_form_ocr_pipeline
from modules.formatter import save_excel, save_filled_pdf, save_json, save_text
from modules.parser import LoanFormParser
from modules.preprocess import pil_to_bgr
from modules.ocr_engine import HybridOCREngine
from modules.template_mapper import TemplateMapper
from modules.document_loader import load_document_images


app = Flask(__name__)
app.secret_key = "loan-form-secret-key"

cfg = AppConfig()
UPLOAD_DIR = cfg.uploads_dir
FORM_UPLOAD_DIR = UPLOAD_DIR / "forms"
OUTPUT_DIR = cfg.outputs_dir
SUPPORTED_FORM_SUFFIXES = {".pdf"}
OUTPUT_FILES = [
    "extracted_raw.json",
    "loan_form_output.json",
    "loan_form_output.xlsx",
    "ocr_full_text.txt",
]
DOWNLOADABLE_OUTPUTS = set(OUTPUT_FILES + ["loan_form_output_filled.pdf"])

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FORM_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_ocr_engine() -> HybridOCREngine:
    return HybridOCREngine(
        language=cfg.ocr.language,
        use_paddle=cfg.ocr.use_paddle,
        use_easy=cfg.ocr.use_easyocr,
        use_tesseract=cfg.ocr.use_tesseract,
        strict_paddle=cfg.ocr.strict_paddle,
        paddle_cache_dir=cfg.paddle_cache_dir,
        paddle_device=cfg.ocr.paddle_device,
        paddle_enable_mkldnn=cfg.ocr.paddle_enable_mkldnn,
        paddle_ocr_version=cfg.ocr.paddle_ocr_version,
        paddle_use_doc_orientation_classify=cfg.ocr.paddle_use_doc_orientation_classify,
        paddle_use_doc_unwarping=cfg.ocr.paddle_use_doc_unwarping,
        paddle_use_textline_orientation=cfg.ocr.paddle_use_textline_orientation,
        paddle_disable_model_source_check=cfg.ocr.paddle_disable_model_source_check,
        paddle_doc_orientation_model_dir=cfg.ocr.paddle_doc_orientation_model_dir,
        paddle_doc_unwarping_model_dir=cfg.ocr.paddle_doc_unwarping_model_dir,
        paddle_text_detection_model_dir=cfg.ocr.paddle_text_detection_model_dir,
        paddle_textline_orientation_model_dir=cfg.ocr.paddle_textline_orientation_model_dir,
        paddle_text_recognition_model_dir=cfg.ocr.paddle_text_recognition_model_dir,
    )


def get_latest_uploaded_input():
    uploads = sorted(
        [path for path in UPLOAD_DIR.iterdir() if path.is_file() and is_supported_input_file(path)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return uploads[0] if uploads else None


def is_supported_form_file(path: Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_FORM_SUFFIXES


def get_latest_uploaded_form():
    forms = sorted(
        [path for path in FORM_UPLOAD_DIR.iterdir() if path.is_file() and is_supported_form_file(path)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return forms[0] if forms else None


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_output_files():
    return [name for name in OUTPUT_FILES if (OUTPUT_DIR / name).exists()]


def available_preview_images():
    preview_files = [path.name for path in OUTPUT_DIR.glob(f"{PREVIEW_PREFIX}*.png") if path.is_file()]

    def sort_key(name: str):
        stem = Path(name).stem
        page_no = stem.replace(PREVIEW_PREFIX.rstrip("_"), "").split("_")[-1]
        try:
            return int(page_no)
        except Exception:
            return 0

    return sorted(preview_files, key=sort_key)


def build_upload_path(
    filename: str,
    *,
    destination_dir: Path = UPLOAD_DIR,
    allowed_suffixes=None,
    default_stem: str = "handwritten_upload",
) -> Path:
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Uploaded filename is invalid.")

    allowed_suffixes = allowed_suffixes or SUPPORTED_INPUT_SUFFIXES
    suffix = Path(safe_name).suffix.lower()
    if suffix not in allowed_suffixes:
        allowed = ", ".join(sorted(allowed_suffixes))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    destination_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(safe_name).stem or default_stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = destination_dir / f"{stem}_{timestamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = destination_dir / f"{stem}_{timestamp}_{counter}{suffix}"
        counter += 1
    return candidate


def remove_filled_output():
    filled_output = OUTPUT_DIR / "loan_form_output_filled.pdf"
    if filled_output.exists():
        filled_output.unlink()


def build_full_text(parsed: dict) -> str:
    pages = parsed.get("pages", [])
    chunks = []
    for page in pages:
        page_number = page.get("page_number")
        text = (page.get("text") or "").strip()
        if not text:
            continue
        chunks.append(f"Page {page_number}\n{text}")
    return "\n\n".join(chunks)


def _normalize_mobile(candidate: str) -> str:
    raw = (
        (candidate or "")
        .replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("i", "1")
        .replace("l", "1")
    )
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) in {10, 11, 12} and digits.startswith("01"):
        return digits
    return ""


def _enrich_fields_from_page_text(parsed: dict):
    fields = parsed.get("fields", {})
    pages = parsed.get("pages", [])
    page1_text = (pages[0].get("text") if pages else "") or ""
    page2_text = (pages[1].get("text") if len(pages) > 1 else "") or ""

    compact1 = page1_text.replace("\n", " ")
    compact2 = page2_text.replace("\n", " ")

    current_name = str(fields.get("full_name") or "").strip()
    if not current_name or any(word in current_name.lower() for word in ["business", "busines", "profession", "female", "male", "fomolo"]):
        match = re.search(
            r"full\s*n[ao]me\s*[:;]?\s*(.{2,80}?)(?:profession|gender|gonoor|d[ao]te\s*of\s*birth)",
            compact1,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = re.sub(r"[^A-Za-z.\s]", " ", match.group(1))
            candidate = " ".join(candidate.split())
            if len(candidate) >= 3 and not any(w in candidate.lower() for w in ["business", "profession"]):
                fields["full_name"] = candidate

    if not str(fields.get("dob") or "").strip():
        area = ""
        area_match = re.search(r"d[ao]te\s*of\s*birth.{0,40}", compact1, flags=re.IGNORECASE)
        if area_match:
            area = area_match.group(0)
        date_match = re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", area or compact1)
        if date_match:
            fields["dob"] = date_match.group(0)
        else:
            yr4 = re.search(r"\b(19\d{2}|20\d{2})\b", area or compact1)
            if yr4:
                fields["dob"] = yr4.group(1)
            else:
                yr3 = re.search(r"\b(\d{3})\b", area or "")
                if yr3:
                    y = yr3.group(1)
                    fields["dob"] = f"1{y}" if y.startswith("9") else f"2{y}"

    if not _normalize_mobile(str(fields.get("mobile") or "")):
        mobile_block = re.search(r"m[oa]b\w*\s*no.{0,40}", compact1, flags=re.IGNORECASE)
        if mobile_block:
            mobile = _normalize_mobile(mobile_block.group(0))
            if mobile:
                fields["mobile"] = mobile
        if not str(fields.get("mobile") or "").strip():
            for token in re.findall(r"[0-9OIilo\-]{10,20}", compact1):
                mobile = _normalize_mobile(token)
                if mobile:
                    fields["mobile"] = mobile
                    break

    current_business = str(fields.get("business_name") or "").strip().lower()
    if not current_business or "name of" in current_business or "compony" in current_business:
        match = re.search(
            r"n[ao]me\s*of\s*(?:compa?n[yi]|business)\s*[:;]?\s*(.{2,80}?)(?:address|mobile|phone|$)",
            compact2,
            flags=re.IGNORECASE,
        )
        if match:
            candidate = " ".join(re.sub(r"[^A-Za-z0-9&.,'\-\s]", " ", match.group(1)).split())
            if len(candidate) >= 3:
                fields["business_name"] = candidate


def process_input_file(input_path: Path, target_form_path: Path | None = None):
    ocr_engine = build_ocr_engine()
    parsed = run_form_ocr_pipeline(
        input_path=input_path,
        output_dir=OUTPUT_DIR,
        ocr_engine=ocr_engine,
        target_form_path=target_form_path,
    )
    parsed.setdefault("meta", {})
    parsed["meta"]["input_type"] = input_path.suffix.lower().lstrip(".")
    parsed["meta"]["parsed_at"] = datetime.now().isoformat(timespec="seconds")
    parsed["meta"]["workflow_mode"] = "ocr_first_form_fill"

    form_status = parsed["meta"].get("form_fill_status")
    if target_form_path is not None and parsed["meta"].get("available_pdf_field_count", 0) > 0:
        try:
            fill_meta = save_filled_pdf(
                parsed,
                template_pdf_path=Path(target_form_path),
                out_pdf_path=OUTPUT_DIR / "loan_form_output_filled.pdf",
                mode="form",
            )
            parsed["meta"].update(fill_meta)
            parsed["meta"]["form_fill_status"] = "completed"
            parsed["meta"]["form_fill_message"] = f"Filled {Path(target_form_path).name} using OCR-extracted field values."
        except Exception as exc:
            remove_filled_output()
            parsed["meta"]["form_fill_status"] = "failed"
            parsed["meta"]["form_fill_message"] = str(exc)
    elif target_form_path is not None and parsed["meta"].get("available_pdf_field_count", 0) == 0:
        try:
            mapper = TemplateMapper(cfg.default_field_map)
            source_pages = [pil_to_bgr(page) for page in load_document_images(input_path)]
            target_pages = [pil_to_bgr(page) for page in load_document_images(Path(target_form_path))]

            aligned_pages = []
            for idx, source_page in enumerate(source_pages):
                if idx < len(target_pages):
                    aligned_page, _ = align_page_to_template(source_page, target_pages[idx])
                    aligned_pages.append(aligned_page)
                else:
                    aligned_pages.append(source_page)

            mapped = LoanFormParser(mapper, ocr_engine).parse(aligned_pages)
            parsed["fields"] = mapped.get("fields", {})
            parsed["tables"] = mapped.get("tables", {})
            parsed["signatures"] = mapped.get("signatures", {})
            parsed["stamps"] = mapped.get("stamps", {})
            _enrich_fields_from_page_text(parsed)
            parsed["field_matches"] = [
                {
                    "name": name,
                    "page_number": next(
                        (
                            item.get("page")
                            for item in mapper.get_fields()
                            if item.get("name") == name
                        ),
                        None,
                    ),
                    "value": value,
                    "confidence": None,
                    "field_type": "mapped_text",
                    "status": "mapped_extracted" if str(value).strip() else "empty",
                }
                for name, value in parsed["fields"].items()
            ]

            page_image_sizes = {
                idx: (int(page.shape[1]), int(page.shape[0]))
                for idx, page in enumerate(target_pages, start=1)
            }

            fill_meta = save_filled_pdf(
                parsed,
                template_pdf_path=Path(target_form_path),
                out_pdf_path=OUTPUT_DIR / "loan_form_output_filled.pdf",
                field_map=mapper.template,
                page_image_sizes=page_image_sizes,
                mode="overlay",
            )
            parsed["meta"].update(fill_meta)
            parsed["meta"]["form_fill_status"] = "completed"
            parsed["meta"]["form_fill_message"] = (
                "Filled the provided template using mapped OCR regions because the PDF has no fillable form fields."
            )
        except Exception as exc:
            remove_filled_output()
            parsed["meta"]["form_fill_status"] = "failed"
            parsed["meta"]["form_fill_message"] = str(exc)
    elif form_status != "ready":
        remove_filled_output()

    full_text = build_full_text(parsed)
    save_json(parsed, OUTPUT_DIR / "extracted_raw.json")
    save_json(parsed, OUTPUT_DIR / "loan_form_output.json")
    save_excel(parsed, OUTPUT_DIR / "loan_form_output.xlsx")
    save_text(full_text, OUTPUT_DIR / "ocr_full_text.txt")
    return parsed


def build_flash_message(parsed: dict, input_name: str) -> tuple[str, str]:
    status = parsed.get("meta", {}).get("form_fill_status")
    message = parsed.get("meta", {}).get("form_fill_message", "")
    if status == "completed":
        target_name = parsed.get("meta", {}).get("target_form_file", "the provided form")
        return "success", f"Processed {input_name}, extracted the handwriting, and filled {target_name}."
    if status == "failed":
        return "error", f"Processed {input_name}, but form filling failed: {message}"
    return "success", f"Processed {input_name}. OCR results are ready. {message}"


@app.get("/")
def index():
    parsed = read_json_if_exists(OUTPUT_DIR / "loan_form_output.json")
    filled_pdf = OUTPUT_DIR / "loan_form_output_filled.pdf"
    uploaded_input = get_latest_uploaded_input()
    uploaded_form = get_latest_uploaded_form()
    return render_template(
        "index.html",
        parsed=parsed,
        uploaded_file_exists=uploaded_input is not None,
        uploaded_file_path=str(uploaded_input) if uploaded_input else "",
        uploaded_file_name=uploaded_input.name if uploaded_input else "",
        provided_form_exists=uploaded_form is not None,
        provided_form_path=str(uploaded_form) if uploaded_form else "",
        provided_form_name=uploaded_form.name if uploaded_form else "",
        accepted_formats=", ".join(sorted(SUPPORTED_INPUT_SUFFIXES)),
        filled_pdf_exists=filled_pdf.exists(),
        filled_pdf_name=filled_pdf.name,
        preview_images=available_preview_images(),
        outputs=available_output_files(),
    )


@app.post("/upload")
def upload_file():
    uploaded_file = request.files.get("handwritten_file")
    target_form = request.files.get("target_form")
    existing_form = get_latest_uploaded_form()

    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose a handwritten PDF or image file to upload.", "error")
        return redirect(url_for("index"))

    try:
        upload_path = build_upload_path(uploaded_file.filename)
        target_form_path = existing_form
        if target_form is not None and target_form.filename:
            target_form_path = build_upload_path(
                target_form.filename,
                destination_dir=FORM_UPLOAD_DIR,
                allowed_suffixes=SUPPORTED_FORM_SUFFIXES,
                default_stem="provided_form",
            )
            target_form.save(str(target_form_path))

        uploaded_file.save(str(upload_path))
        parsed = process_input_file(upload_path, target_form_path=target_form_path)
        category, message = build_flash_message(parsed, upload_path.name)
        flash(message, category)
    except Exception as exc:
        flash(f"Processing failed: {exc}", "error")

    return redirect(url_for("index"))


@app.post("/parse")
def parse_pdf():
    input_file = get_latest_uploaded_input()
    target_template = get_latest_uploaded_form()

    if input_file is None:
        flash(
            f"No uploaded handwritten file found in {UPLOAD_DIR}. Upload a PDF or image first.",
            "error",
        )
        return redirect(url_for("index"))

    try:
        parsed = process_input_file(input_file, target_form_path=target_template)
        category, message = build_flash_message(parsed, input_file.name)
        flash(message, category)
    except Exception as exc:
        flash(f"Processing failed: {exc}", "error")

    return redirect(url_for("index"))


@app.get("/download/<path:filename>")
def download(filename):
    if filename not in DOWNLOADABLE_OUTPUTS:
        flash("Invalid file request.", "error")
        return redirect(url_for("index"))

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        flash(f"File not found: {filename}", "error")
        return redirect(url_for("index"))

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


@app.get("/uploaded/<path:filename>")
def uploaded_file(filename):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists() or not is_supported_input_file(file_path):
        flash("Uploaded file not found.", "error")
        return redirect(url_for("index"))

    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)


@app.get("/provided-form/<path:filename>")
def provided_form(filename):
    file_path = FORM_UPLOAD_DIR / filename
    if not file_path.exists() or not is_supported_form_file(file_path):
        flash("Provided form not found.", "error")
        return redirect(url_for("index"))

    return send_from_directory(FORM_UPLOAD_DIR, filename, as_attachment=False)


@app.get("/artifact/<path:filename>")
def artifact(filename):
    if not filename.startswith(PREVIEW_PREFIX) or not filename.endswith(".png"):
        flash("Invalid preview artifact request.", "error")
        return redirect(url_for("index"))

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        flash(f"Artifact not found: {filename}", "error")
        return redirect(url_for("index"))

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/preview/<path:filename>")
def preview(filename):
    if filename != "loan_form_output_filled.pdf":
        flash("Invalid preview request.", "error")
        return redirect(url_for("index"))

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        flash(f"File not found: {filename}", "error")
        return redirect(url_for("index"))

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
