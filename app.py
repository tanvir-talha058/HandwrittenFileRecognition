import argparse
import json
from pathlib import Path

from config import AppConfig
from modules.formatter import save_excel, save_filled_pdf_placeholder, save_json
from modules.ocr_engine import HybridOCREngine
from modules.parser import LoanFormParser
from modules.pdf_processor import pdf_to_images
from modules.preprocess import pil_to_bgr
from modules.template_mapper import TemplateMapper


def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten Loan Form Digitization")
    parser.add_argument("--input-pdf", required=True, help="Path to handwritten input PDF")
    parser.add_argument("--field-map", default=None, help="Path to field map JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--review-json",
        default=None,
        help="Optional corrected JSON to merge after manual validation",
    )
    return parser.parse_args()


def deep_merge(base: dict, override: dict):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value


def main():
    args = parse_args()
    cfg = AppConfig()

    input_pdf = Path(args.input_pdf).expanduser().resolve()
    field_map = (
        Path(args.field_map).expanduser().resolve()
        if args.field_map
        else cfg.default_field_map
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else cfg.outputs_dir
    )

    mapper = TemplateMapper(field_map)
    ocr = HybridOCREngine(
        language=cfg.ocr.language,
        use_paddle=cfg.ocr.use_paddle,
        use_easy=cfg.ocr.use_easyocr,
        use_tesseract=cfg.ocr.use_tesseract,
    )

    pil_pages = pdf_to_images(input_pdf)
    pages_bgr = [pil_to_bgr(page) for page in pil_pages]

    parser = LoanFormParser(mapper, ocr)
    parsed = parser.parse(pages_bgr)

    if args.review_json:
        review_path = Path(args.review_json).expanduser().resolve()
        if review_path.exists():
            corrected = json.loads(review_path.read_text(encoding="utf-8"))
            deep_merge(parsed, corrected)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = output_dir / "extracted_raw.json"
    final_json_path = output_dir / "loan_form_output.json"
    excel_path = output_dir / "loan_form_output.xlsx"
    pdf_placeholder = output_dir / "loan_form_output_filled.pdf"

    save_json(parsed, raw_json_path)
    save_json(parsed, final_json_path)
    save_excel(parsed, excel_path)
    save_filled_pdf_placeholder(parsed, pdf_placeholder)

    print("Processing complete")
    print(f"- Raw JSON: {raw_json_path}")
    print(f"- Final JSON: {final_json_path}")
    print(f"- Excel: {excel_path}")
    print(f"- Filled PDF placeholder: {pdf_placeholder.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
