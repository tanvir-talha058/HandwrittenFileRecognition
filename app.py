import argparse
from pathlib import Path

from config import AppConfig
from modules.form_pipeline import run_form_ocr_pipeline
from modules.formatter import save_excel, save_filled_pdf, save_json, save_text
from modules.ocr_engine import HybridOCREngine


def parse_args():
    parser = argparse.ArgumentParser(description="Handwritten form OCR and PDF filling")
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to the handwritten source PDF or image",
    )
    parser.add_argument(
        "--target-form",
        default=None,
        help="Optional fillable PDF form to align against and fill",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory")
    return parser.parse_args()


def build_full_text(parsed: dict) -> str:
    chunks = []
    for page in parsed.get("pages", []):
        page_number = page.get("page_number")
        text = (page.get("text") or "").strip()
        if not text:
            continue
        chunks.append(f"Page {page_number}\n{text}")
    return "\n\n".join(chunks)


def main():
    args = parse_args()
    cfg = AppConfig()

    input_file = Path(args.input_file).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else cfg.outputs_dir
    )
    target_form = (
        Path(args.target_form).expanduser().resolve()
        if args.target_form
        else None
    )

    ocr = HybridOCREngine(
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

    parsed = run_form_ocr_pipeline(
        input_path=input_file,
        output_dir=output_dir,
        ocr_engine=ocr,
        target_form_path=target_form,
    )
    parsed.setdefault("meta", {})
    parsed["meta"]["input_type"] = input_file.suffix.lower().lstrip(".")

    if target_form is not None and parsed["meta"].get("available_pdf_field_count", 0) > 0:
        try:
            fill_meta = save_filled_pdf(
                parsed,
                template_pdf_path=target_form,
                out_pdf_path=output_dir / "loan_form_output_filled.pdf",
                mode="form",
            )
            parsed["meta"].update(fill_meta)
            parsed["meta"]["form_fill_status"] = "completed"
        except Exception as exc:
            parsed["meta"]["form_fill_status"] = "failed"
            parsed["meta"]["form_fill_message"] = str(exc)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json_path = output_dir / "extracted_raw.json"
    final_json_path = output_dir / "loan_form_output.json"
    excel_path = output_dir / "loan_form_output.xlsx"
    text_path = output_dir / "ocr_full_text.txt"

    save_json(parsed, raw_json_path)
    save_json(parsed, final_json_path)
    save_excel(parsed, excel_path)
    save_text(build_full_text(parsed), text_path)

    print("Processing complete")
    print(f"- Raw JSON: {raw_json_path}")
    print(f"- Final JSON: {final_json_path}")
    print(f"- Excel: {excel_path}")
    print(f"- OCR text: {text_path}")
    if target_form is not None:
        print(f"- Filled PDF: {output_dir / 'loan_form_output_filled.pdf'}")


if __name__ == "__main__":
    main()
