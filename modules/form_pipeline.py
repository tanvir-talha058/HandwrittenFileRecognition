from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageDraw
except Exception:
    Image = None
    ImageDraw = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from .document_loader import load_document_images
from .ocr_engine import HybridOCREngine
from .preprocess import clamp_bbox, crop_bbox, pil_to_bgr


PREVIEW_PREFIX = "ocr_preview_page_"


def run_form_ocr_pipeline(
    input_path: Path,
    output_dir: Path,
    ocr_engine: HybridOCREngine,
    target_form_path: Optional[Path] = None,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_preview_images(output_dir)

    source_pages = load_document_images(input_path)
    source_pages_bgr = [pil_to_bgr(page) for page in source_pages]

    # Full-page OCR previews are expensive on CPU; limit them when Paddle is unavailable.
    max_preview_ocr_pages = len(source_pages_bgr)
    if getattr(ocr_engine, "_paddle", None) is None:
        max_preview_ocr_pages = min(2, len(source_pages_bgr))

    pages = build_page_previews(
        source_pages_bgr,
        output_dir,
        ocr_engine,
        max_ocr_pages=max_preview_ocr_pages,
        preview_max_side=1300 if getattr(ocr_engine, "_paddle", None) is None else 1700,
    )
    result = {
        "fields": {},
        "field_matches": [],
        "pages": pages,
        "meta": {
            "page_count": len(source_pages_bgr),
            "source_file": input_path.name,
        },
    }

    if target_form_path is None:
        result["meta"]["form_fill_status"] = "skipped"
        result["meta"]["form_fill_message"] = "No target form was provided, so only OCR results were generated."
        return result

    result["meta"]["target_form_file"] = Path(target_form_path).name

    form_fields = load_pdf_form_fields(target_form_path)
    result["meta"]["available_pdf_field_count"] = len(form_fields)
    if not form_fields:
        result["meta"]["form_fill_status"] = "failed"
        result["meta"]["form_fill_message"] = "The provided PDF has no fillable form fields."
        return result

    target_pages = load_document_images(Path(target_form_path))
    target_pages_bgr = [pil_to_bgr(page) for page in target_pages]

    extracted = extract_fields_from_form(
        source_pages_bgr=source_pages_bgr,
        target_pages_bgr=target_pages_bgr,
        form_fields=form_fields,
        ocr_engine=ocr_engine,
    )
    result["fields"] = extracted["fields"]
    result["field_matches"] = extracted["field_matches"]
    result["meta"]["alignment"] = extracted["alignment"]
    result["meta"]["filled_field_count"] = len(
        [value for value in extracted["fields"].values() if str(value).strip()]
    )
    result["meta"]["form_fill_status"] = "ready"
    result["meta"]["form_fill_message"] = "OCR extraction is ready for PDF form filling."
    return result


def clear_preview_images(output_dir: Path):
    for path in output_dir.glob(f"{PREVIEW_PREFIX}*.png"):
        try:
            path.unlink()
        except Exception:
            continue


def build_page_previews(
    pages_bgr: List[np.ndarray],
    output_dir: Path,
    ocr_engine: HybridOCREngine,
    max_ocr_pages: int = 0,
    preview_max_side: int = 1700,
) -> List[Dict]:
    pages = []
    for idx, page in enumerate(pages_bgr, start=1):
        preview_page = resize_for_preview(page, max_side=preview_max_side)
        detections = ocr_engine.read_page(preview_page) if idx <= max_ocr_pages else []
        preview_name = f"{PREVIEW_PREFIX}{idx}.png"
        save_ocr_preview(preview_page, detections, output_dir / preview_name)
        pages.append(
            {
                "page_number": idx,
                "size": [int(preview_page.shape[1]), int(preview_page.shape[0])],
                "text": "\n".join(item["text"] for item in detections if item.get("text")),
                "preview_file": preview_name,
                "detections": detections,
            }
        )
    return pages


def resize_for_preview(image_bgr: np.ndarray, max_side: int = 1700) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(height, width)))
    if scale >= 1.0 or cv2 is None:
        return image_bgr
    return cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def save_ocr_preview(image_bgr: np.ndarray, detections: List[Dict], out_path: Path):
    if Image is None or ImageDraw is None:
        return

    rgb = image_bgr[:, :, ::-1] if image_bgr.ndim == 3 else image_bgr
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for item in detections:
        polygon = [tuple(point) for point in item.get("bbox", [])]
        if len(polygon) < 4:
            continue
        draw.line(polygon + [polygon[0]], fill=(236, 99, 59), width=3)
        label = item.get("text", "")[:60]
        if label:
            x = min(point[0] for point in polygon)
            y = max(0, min(point[1] for point in polygon) - 22)
            text_box = draw.textbbox((x + 8, y + 5), label)
            draw.rounded_rectangle(
                [x, y, text_box[2] + 8, text_box[3] + 5],
                radius=8,
                fill=(29, 36, 48),
            )
            draw.text((x + 8, y + 5), label, fill=(255, 255, 255))

    image.save(out_path)


def load_pdf_form_fields(pdf_path: Path) -> List[Dict]:
    if PdfReader is None:
        raise RuntimeError("pypdf is required to inspect fillable PDF fields.")

    reader = PdfReader(str(pdf_path))
    fields = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        annotations = page.get("/Annots") or []
        for annot_ref in annotations:
            annot = annot_ref.get_object()
            if str(annot.get("/Subtype")) != "/Widget":
                continue

            name = annot.get("/T")
            rect = annot.get("/Rect")
            if not name or not rect or len(rect) != 4:
                continue

            x1, y1, x2, y2 = [float(value) for value in rect]
            field_type = str(annot.get("/FT") or "/Tx")
            fields.append(
                {
                    "name": str(name),
                    "page_number": page_number,
                    "pdf_rect": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                    "pdf_page_size": [page_width, page_height],
                    "field_type": field_type,
                    "label": str(annot.get("/TU") or name),
                }
            )
    return fields


def extract_fields_from_form(
    source_pages_bgr: List[np.ndarray],
    target_pages_bgr: List[np.ndarray],
    form_fields: List[Dict],
    ocr_engine: HybridOCREngine,
) -> Dict:
    fields = {item["name"]: "" for item in form_fields}
    field_matches = []
    alignment = []

    fields_by_page: Dict[int, List[Dict]] = {}
    for item in form_fields:
        fields_by_page.setdefault(int(item["page_number"]), []).append(item)

    page_count = max(len(source_pages_bgr), len(target_pages_bgr))
    for page_number in range(1, page_count + 1):
        source_page = source_pages_bgr[page_number - 1] if page_number <= len(source_pages_bgr) else None
        target_page = target_pages_bgr[page_number - 1] if page_number <= len(target_pages_bgr) else None

        if source_page is None or target_page is None:
            alignment.append(
                {
                    "page_number": page_number,
                    "status": "missing_page",
                }
            )
            continue

        aligned_page, alignment_meta = align_page_to_template(source_page, target_page)
        alignment.append(
            {
                "page_number": page_number,
                **alignment_meta,
            }
        )

        page_fields = fields_by_page.get(page_number, [])
        target_height, target_width = target_page.shape[:2]
        for field in page_fields:
            if field.get("field_type") != "/Tx":
                field_matches.append(
                    {
                        "name": field["name"],
                        "page_number": page_number,
                        "value": "",
                        "confidence": 0.0,
                        "field_type": field.get("field_type"),
                        "status": "unsupported_field_type",
                    }
                )
                continue

            bbox = pdf_rect_to_image_bbox(
                field["pdf_rect"],
                field["pdf_page_size"],
                [target_width, target_height],
            )
            roi = crop_bbox(aligned_page, bbox)
            region = ocr_engine.read_region(roi)
            text = region.get("text", "")
            fields[field["name"]] = text
            field_matches.append(
                {
                    "name": field["name"],
                    "page_number": page_number,
                    "value": text,
                    "confidence": region.get("confidence", 0.0),
                    "field_type": field.get("field_type"),
                    "ocr_engine": region.get("engine"),
                    "bbox": bbox,
                    "status": "extracted" if text else "empty",
                }
            )

    return {
        "fields": fields,
        "field_matches": field_matches,
        "alignment": alignment,
    }


def pdf_rect_to_image_bbox(
    pdf_rect: List[float],
    pdf_page_size: List[float],
    image_size: List[int],
    padding_ratio: float = 0.12,
) -> List[int]:
    pdf_width, pdf_height = pdf_page_size
    image_width, image_height = image_size
    x1, y1, x2, y2 = pdf_rect

    left = (x1 / pdf_width) * image_width
    right = (x2 / pdf_width) * image_width
    top = image_height - ((y2 / pdf_height) * image_height)
    bottom = image_height - ((y1 / pdf_height) * image_height)

    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    pad_x = width * padding_ratio
    pad_y = height * max(0.18, padding_ratio)

    return clamp_bbox(
        [
            int(round(left - pad_x)),
            int(round(top - pad_y)),
            int(round(right + pad_x)),
            int(round(bottom + pad_y)),
        ],
        image_width,
        image_height,
    )


def align_page_to_template(source_bgr: np.ndarray, template_bgr: np.ndarray):
    target_height, target_width = template_bgr.shape[:2]
    if cv2 is None:
        resized = _resize_to_target(source_bgr, (target_width, target_height))
        return resized, {"status": "resized_only", "match_count": 0}

    source_gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY) if source_bgr.ndim == 3 else source_bgr
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY) if template_bgr.ndim == 3 else template_bgr

    orb = cv2.ORB_create(3500)
    keypoints_src, descriptors_src = orb.detectAndCompute(source_gray, None)
    keypoints_tgt, descriptors_tgt = orb.detectAndCompute(template_gray, None)

    if descriptors_src is None or descriptors_tgt is None or len(keypoints_src) < 8 or len(keypoints_tgt) < 8:
        resized = _resize_to_target(source_bgr, (target_width, target_height))
        return resized, {"status": "resized_only", "match_count": 0}

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(descriptors_src, descriptors_tgt, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        first, second = match_pair
        if first.distance < 0.75 * second.distance:
            good_matches.append(first)

    if len(good_matches) < 12:
        resized = _resize_to_target(source_bgr, (target_width, target_height))
        return resized, {"status": "resized_only", "match_count": len(good_matches)}

    source_points = np.float32([keypoints_src[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_points = np.float32([keypoints_tgt[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
    if homography is None:
        resized = _resize_to_target(source_bgr, (target_width, target_height))
        return resized, {"status": "resized_only", "match_count": len(good_matches)}

    aligned = cv2.warpPerspective(
        source_bgr,
        homography,
        (target_width, target_height),
        borderValue=(255, 255, 255),
    )
    inliers = int(mask.sum()) if mask is not None else len(good_matches)
    return aligned, {"status": "aligned", "match_count": len(good_matches), "inliers": inliers}


def _resize_to_target(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_width, target_height = size
    if cv2 is None:
        return image_bgr
    return cv2.resize(image_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
