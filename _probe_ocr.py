import traceback
import numpy as np

from flask_app import build_ocr_engine, get_latest_uploaded_input
from modules.document_loader import load_document_images

print("probe:start", flush=True)
path = get_latest_uploaded_input()
print(f"probe:input={path}", flush=True)
images = load_document_images(path)
print(f"probe:pages={len(images)}", flush=True)
engine = build_ocr_engine()
print(f"probe:engine_ok={bool(engine._paddle)} init_error={engine._paddle_init_error}", flush=True)
arr = np.array(images[0])
print(f"probe:arr={arr.shape} {arr.dtype}", flush=True)
try:
    detections = engine.read_page(arr)
    print(f"probe:detections={len(detections)}", flush=True)
except Exception as exc:
    print(f"probe:error={type(exc).__name__}: {exc}", flush=True)
    traceback.print_exc()
    raise
print("probe:done", flush=True)
