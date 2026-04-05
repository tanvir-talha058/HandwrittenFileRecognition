import json
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
from modules.document_loader import (
    SUPPORTED_INPUT_SUFFIXES,
    is_supported_input_file,
    load_document_images,
)
from modules.formatter import save_excel, save_filled_pdf, save_json
from modules.ocr_engine import HybridOCREngine
from modules.parser import LoanFormParser
from modules.pdf_processor import pdf_to_images
from modules.preprocess import pil_to_bgr
from modules.template_mapper import TemplateMapper


app = Flask(__name__)
app.secret_key = "loan-form-secret-key"

cfg = AppConfig()
UPLOAD_DIR = cfg.uploads_dir
OUTPUT_DIR = cfg.outputs_dir
FIELD_MAP_DEFAULT = cfg.default_field_map
TEMPLATE_PDF = cfg.project_root / "templates" / "Home_Loan_Booklet.pdf"
OUTPUT_FILES = [
    "extracted_raw.json",
    "loan_form_output.json",
    "loan_form_output.xlsx",
]
DOWNLOADABLE_OUTPUTS = set(OUTPUT_FILES + ["loan_form_output_filled.pdf"])

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_parser(field_map_path: Path) -> LoanFormParser:
    mapper = TemplateMapper(field_map_path)
    ocr = HybridOCREngine(
        language=cfg.ocr.language,
        use_paddle=cfg.ocr.use_paddle,
        use_easy=cfg.ocr.use_easyocr,
        use_tesseract=cfg.ocr.use_tesseract,
    )
    return LoanFormParser(mapper, ocr)


def get_latest_uploaded_input():
    uploads = sorted(
        [path for path in UPLOAD_DIR.iterdir() if path.is_file() and is_supported_input_file(path)],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return uploads[0] if uploads else None


def read_json_if_exists(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def available_output_files():
    return [name for name in OUTPUT_FILES if (OUTPUT_DIR / name).exists()]


def build_upload_path(filename: str) -> Path:
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Uploaded filename is invalid.")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_INPUT_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_INPUT_SUFFIXES))
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    stem = Path(safe_name).stem or "handwritten_upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = UPLOAD_DIR / f"{stem}_{timestamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = UPLOAD_DIR / f"{stem}_{timestamp}_{counter}{suffix}"
        counter += 1
    return candidate


def process_input_file(input_path: Path):
    parser = build_parser(Path(FIELD_MAP_DEFAULT))
    pil_pages = load_document_images(input_path)
    pages_bgr = [pil_to_bgr(page) for page in pil_pages]
    parsed = parser.parse(pages_bgr)
    parsed.setdefault("meta", {})
    parsed["meta"]["source_file"] = input_path.name
    parsed["meta"]["template_file"] = TEMPLATE_PDF.name
    parsed["meta"]["input_type"] = input_path.suffix.lower().lstrip(".")
    parsed["meta"]["parsed_at"] = datetime.now().isoformat(timespec="seconds")

    save_json(parsed, OUTPUT_DIR / "extracted_raw.json")
    save_json(parsed, OUTPUT_DIR / "loan_form_output.json")
    save_excel(parsed, OUTPUT_DIR / "loan_form_output.xlsx")
    template_pages = pdf_to_images(TEMPLATE_PDF)
    page_image_sizes = {
        idx + 1: (page.size[0], page.size[1])
        for idx, page in enumerate(template_pages)
    }
    with Path(FIELD_MAP_DEFAULT).open("r", encoding="utf-8") as handle:
        field_map_data = json.load(handle)
    save_filled_pdf(
        parsed,
        template_pdf_path=TEMPLATE_PDF,
        out_pdf_path=OUTPUT_DIR / "loan_form_output_filled.pdf",
        field_map=field_map_data,
        page_image_sizes=page_image_sizes,
    )
    return parsed


@app.get("/")
def index():
    parsed = read_json_if_exists(OUTPUT_DIR / "loan_form_output.json")
    filled_pdf = OUTPUT_DIR / "loan_form_output_filled.pdf"
    uploaded_input = get_latest_uploaded_input()
    return render_template(
        "index.html",
        parsed=parsed,
        uploaded_file_exists=uploaded_input is not None,
        uploaded_file_path=str(uploaded_input) if uploaded_input else "",
        uploaded_file_name=uploaded_input.name if uploaded_input else "",
        accepted_formats=", ".join(sorted(SUPPORTED_INPUT_SUFFIXES)),
        template_pdf_exists=TEMPLATE_PDF.exists(),
        template_pdf_path=str(TEMPLATE_PDF),
        filled_pdf_exists=filled_pdf.exists(),
        filled_pdf_name=filled_pdf.name,
        outputs=available_output_files(),
    )


@app.post("/upload")
def upload_file():
    uploaded_file = request.files.get("handwritten_file")

    if uploaded_file is None or not uploaded_file.filename:
        flash("Choose a handwritten PDF or image file to upload.", "error")
        return redirect(url_for("index"))

    if not TEMPLATE_PDF.exists():
        flash(f"Template PDF not found: {TEMPLATE_PDF}", "error")
        return redirect(url_for("index"))

    try:
        upload_path = build_upload_path(uploaded_file.filename)
        uploaded_file.save(str(upload_path))
        process_input_file(upload_path)
        flash(
            f"Uploaded {upload_path.name}, parsed it, and filled {TEMPLATE_PDF.name}.",
            "success",
        )
    except Exception as exc:
        flash(f"Upload failed: {exc}", "error")

    return redirect(url_for("index"))


@app.post("/parse")
def parse_pdf():
    input_file = get_latest_uploaded_input()

    if not TEMPLATE_PDF.exists():
        flash(f"Template PDF not found: {TEMPLATE_PDF}", "error")
        return redirect(url_for("index"))

    if input_file is None:
        flash(
            f"No uploaded handwritten file found in {UPLOAD_DIR}. Upload a PDF or image first.",
            "error",
        )
        return redirect(url_for("index"))

    try:
        process_input_file(input_file)
        flash(
            f"Reprocessed {input_file.name} and regenerated the filled PDF.",
            "success",
        )
    except Exception as exc:
        flash(f"Parse failed: {exc}", "error")

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


@app.get("/preview/<path:filename>")
def preview(filename):
    allowed = {
        "loan_form_output_filled.pdf",
    }
    if filename not in allowed:
        flash("Invalid preview request.", "error")
        return redirect(url_for("index"))

    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        flash(f"File not found: {filename}", "error")
        return redirect(url_for("index"))

    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
