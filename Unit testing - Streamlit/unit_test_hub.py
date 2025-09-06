"""
Unit Testing Hub (Streamlit)
----------------------------
Place in: Unit testing - Streamlit/unit_test_hub.py

Coordinates:
  - Tools (reads PDFs from a configured local folder)
  - Relationships & Enrichments (reads JSONs from a configured local folder or upload)

Contracts:
  * Each tool module:     run_tool(pdf_path: str) -> dict
  * Each relationship:    run_job(json_path: str) -> dict
"""

import os, glob, json
from pathlib import Path
import importlib.util
import streamlit as st

DEFAULT_PDF_DIR = r"C:\Users\piete\OneDrive\Documents\Python Projects\Document Info Retrieval Project\Text Examples"
DEFAULT_JSON_DIR = r"C:\Users\piete\OneDrive\Documents\Python Projects\Document Info Retrieval Project\JSON files for Unit Testing"

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent
TOOLS_DIR = REPO_ROOT / "Unit testing - Tools"
REL_DIR   = REPO_ROOT / "Unit testing - Text Relationships"

TOOL_FILE_MAP = {
    "Azure StyleFont (FR v3 & DI v4)":               "di_stylefont.py",
    "Tesseract (WordFontAttributes)":                 "tesseract.py",
    "EasyOCR (GPU optional)":                         "easyocr.py",
    "TrOCR (HF, GPU)":                                "trocr.py",
    "PaddleOCR (GPU)":                                "paddleocr.py",
    "docTR (GPU)":                                    "doctr.py",
    "Kraken (HTR, GPU)":                              "kraken.py",
    "Calamari (HTR, GPU)":                            "calamari.py",
    "Azure Document Intelligence (Layout/Styles)":    "azure_di.py",
    "Azure Vision (Read/Brands)":                     "azure_vision.py",
    "Google Document AI":                             "google_document_ai.py",
    "Google Vision (OCR/Logos)":                      "google_vision.py",
    "AWS Textract":                                   "aws_textract.py",
    "ABBYY FineReader SDK":                           "abbyy_fine_reader.py",
    "LayoutParser (blocks/tables)":                   "layoutparser_blocks.py",
    "PyMuPDF (styles from true PDF)":                 "pymupdf_styles.py",
    "Camelot (tables from vector PDF)":               "camelot_tables.py",
    "pdfplumber (tables/lines)":                      "pdfplumber_tables.py",
}

REL_FILE_MAP = {
    "Interrelationships of text":   "interrelationships_text.py",
    "Internal features of text":    "internal_text_features.py",
    "Natural Language Processing":  "nlp.py",
    "Fraud Detection":              "fraud_detection.py",
    "Address normalization (ZA bias)": "address_normalization_za.py",
    "Logo registry check":             "logo_registry_check.py",
    "Table validation & totals":       "table_validation.py",
}

def list_files(dir_path, patterns):
    try:
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(dir_path, pat)))
        return sorted(files)
    except Exception:
        return []

def import_by_path(py_path: Path):
    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
        if not spec or not spec.loader:
            return None, f"Cannot create import spec for: {py_path}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        return mod, None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"

st.set_page_config(page_title="Unit Test Hub", layout="wide")
st.title("Unit Testing Hub")

mode = st.selectbox("Category", ["Tools", "Relationships & Enrichments"], index=0)

if mode == "Tools":
    st.subheader("Tools")
    existing = {Path(p).name for p in list_files(str(TOOLS_DIR), ("*.py",))}
    labels = []
    missing = []
    for k, v in TOOL_FILE_MAP.items():
        if v in existing:
            labels.append(k)
        else:
            labels.append(f"{k}  (missing file: {v})")
            missing.append(v)

    tool_choice = st.selectbox("Select tool", labels)

    pdf_dir = st.text_input("PDF folder", value=DEFAULT_PDF_DIR)
    pdfs = list_files(pdf_dir, ("*.pdf",))
    if not pdfs:
        st.warning("No PDFs found in the selected folder.")
    pdf_choice = st.selectbox("PDF file", pdfs, index=0 if pdfs else None)

    if st.button("Run tool"):
        if not pdf_choice:
            st.error("Please select a PDF.")
        else:
            base = tool_choice.split("  (missing")[0].strip()
            fname = TOOL_FILE_MAP.get(base)
            if not fname:
                st.error("Unrecognized tool.")
            else:
                mpath = TOOLS_DIR / fname
                if not mpath.exists():
                    st.error(f"Missing tool file: {mpath}")
                else:
                    mod, err = import_by_path(mpath)
                    if err or not mod:
                        st.error(f"Import error: {err}")
                    elif not hasattr(mod, "run_tool"):
                        st.error("Module is missing: run_tool(pdf_path: str) -> dict")
                    else:
                        try:
                            out = mod.run_tool(pdf_choice)  # type: ignore
                            st.success("Tool executed.")
                            st.json(out)
                        except Exception as ex:
                            st.exception(ex)

    if missing:
        with st.expander("Missing tool files"):
            st.code("\n".join(sorted(set(missing))), language="text")

else:
    st.subheader("Relationships & Enrichments")
    existing = {Path(p).name for p in list_files(str(REL_DIR), ("*.py",))}
    labels = []
    missing = []
    for k, v in REL_FILE_MAP.items():
        if v in existing:
            labels.append(k)
        else:
            labels.append(f"{k}  (missing file: {v})")
            missing.append(v)

    rel_choice = st.selectbox("Select analysis", labels)

    up = st.file_uploader("Upload Step1/Step2 JSON", type=["json"])
    json_dir = st.text_input("JSON folder", value=DEFAULT_JSON_DIR)
    jsons = list_files(json_dir, ("*.json",))
    json_choice = st.selectbox("Or pick JSON file", jsons, index=0 if jsons else None)

    if st.button("Run analysis"):
        chosen = None
        if up is not None:
            tmp = Path("tmp_upload.json")
            tmp.write_bytes(up.read())
            chosen = str(tmp.resolve())
        elif json_choice:
            chosen = json_choice

        if not chosen:
            st.error("Provide a JSON (upload or choose from folder).")
        else:
            base = rel_choice.split("  (missing")[0].strip()
            fname = REL_FILE_MAP.get(base)
            if not fname:
                st.error("Unrecognized analysis.")
            else:
                mpath = REL_DIR / fname
                if not mpath.exists():
                    st.error(f"Missing analysis file: {mpath}")
                else:
                    mod, err = import_by_path(mpath)
                    if err or not mod:
                        st.error(f"Import error: {err}")
                    elif not hasattr(mod, "run_job"):
                        st.error("Module is missing: run_job(json_path: str) -> dict")
                    else:
                        try:
                            out = mod.run_job(chosen)  # type: ignore
                            st.success("Analysis executed.")
                            st.json(out)
                        except Exception as ex:
                            st.exception(ex)

    if missing:
        with st.expander("Missing analysis files"):
            st.code("\n".join(sorted(set(missing))), language="text")
