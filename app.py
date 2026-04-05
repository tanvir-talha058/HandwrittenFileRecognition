import argparse
import json
from pathlib import Path

from config import AppConfig
from modules.document_loader import load_document_images
from modules.formatter import save_excel, save_filled_pdf, save_json
from modules.ocr_engine import HybridOCREngine
from modules.parser import LoanFormParser
from modules.pdf_processor import pdf_to_images
from modules.preprocess import pil_to_bgr
from modules.template_mapper import TemplateMapper


def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten Loan Form Digitization")
    parser.add_argument(
        "--input-file",
        "--input-pdf",
        dest="input_file",
        required=True,
        help="Path to the handwritten input PDF or image",
    )
    parser.add_argument("--field-map", default=None, help="Path to field map JSON")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument(
        "--template-pdf",
        default=None,
        help="Path to the blank template PDF used for the filled output",
    )
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

    input_file = Path(args.input_file).expanduser().resolve()
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
    template_pdf = (
        Path(args.template_pdf).expanduser().resolve()
        if args.template_pdf
        else cfg.project_root / "templates" / "Home_Loan_Booklet.pdf"
    )

    mapper = TemplateMapper(field_map)
    ocr = HybridOCREngine(
        language=cfg.ocr.language,
        use_paddle=cfg.ocr.use_paddle,
        use_easy=cfg.ocr.use_easyocr,
        use_tesseract=cfg.ocr.use_tesseract,
    )

    pil_pages = load_document_images(input_file)
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
    filled_pdf_path = output_dir / "loan_form_output_filled.pdf"

    save_json(parsed, raw_json_path)
    save_json(parsed, final_json_path)
    save_excel(parsed, excel_path)
    template_pages = pdf_to_images(template_pdf)
    page_image_sizes = {
        idx + 1: (page.size[0], page.size[1])
        for idx, page in enumerate(template_pages)
    }
    with field_map.open("r", encoding="utf-8") as handle:
        field_map_data = json.load(handle)
    save_filled_pdf(
        parsed,
        template_pdf_path=template_pdf,
        out_pdf_path=filled_pdf_path,
        field_map=field_map_data,
        page_image_sizes=page_image_sizes,
    )

    print("Processing complete")
    print(f"- Raw JSON: {raw_json_path}")
    print(f"- Final JSON: {final_json_path}")
    print(f"- Excel: {excel_path}")
    print(f"- Filled PDF: {filled_pdf_path}")


if __name__ == "__main__":
    main()
