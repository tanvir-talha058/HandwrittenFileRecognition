import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Loan Form Manual Review", layout="wide")
st.title("Manual Review - Handwritten Loan Form")

output_dir = Path("outputs")
raw_json_path = output_dir / "extracted_raw.json"

if not raw_json_path.exists():
    st.warning("No extracted output found at outputs/extracted_raw.json")
    st.info("Run app.py first to generate OCR output.")
    st.stop()

parsed = json.loads(raw_json_path.read_text(encoding="utf-8"))

st.subheader("Editable Extracted Fields")
fields = parsed.get("fields", {})
for key, value in list(fields.items()):
    fields[key] = st.text_input(key, value="" if value is None else str(value))

st.subheader("Tables")
for table_name, table_data in parsed.get("tables", {}).items():
    st.markdown(f"### {table_name}")
    for key, value in list(table_data.items()):
        raw_value = st.text_input(f"{table_name}.{key}", value="" if value is None else str(value))
        try:
            if raw_value.strip() == "":
                table_data[key] = None
            elif "." in raw_value:
                table_data[key] = float(raw_value)
            else:
                table_data[key] = int(raw_value)
        except ValueError:
            table_data[key] = raw_value

st.subheader("Signature and Stamp Decisions")
for section in ["signatures", "stamps"]:
    data = parsed.get(section, {})
    st.markdown(f"### {section.capitalize()}")
    for key, value in list(data.items()):
        data[key] = st.text_input(f"{section}.{key}", value=str(value))

if st.button("Save Corrected JSON", type="primary"):
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / "corrected_review.json"
    corrected_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    st.success(f"Saved: {corrected_path}")
