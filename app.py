"""
app.py
======
Streamlit UI for Roadmap Step 1 with a forward-compatible shell for the whole
pipeline. You can run individual modules or (later) the full pipeline.

Key UI features for Step 1:
- Upload a PDF or image file.
- Toggle DI and Vision collection; choose whether to run Vision on PDF pages.
- Show service health pre-checks.
- Run Step 1 and download raw artifacts (.json).

Future modules already appear in the sidebar but are stubs until implemented.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from config import new_run_id
from core.models import ModuleOutput
from core.service_health import get_health_report
from core.io import load_json
from modules import mod1_acquire_raw as mod1

st.set_page_config(page_title="SA OCR/Fraud POC", layout="wide")

# ---------- Sidebar: Module Runner & Health ----------
st.sidebar.title("Module Runner")
module_choice = st.sidebar.selectbox(
    "Choose a module",
    [
        "Step 1: Acquire Raw (DI & Vision)",
        "Step 2: Preprocess (stub)",
        "Step 3: Structure JSON (stub)",
        "Step 4: Prompt Wrap (stub)",
        "Step 5: LLM Evaluate (stub)",
        "Step 6: Self-Validate (stub)",
        "Step 7: Merge (stub)",
    ],
    index=0,
)

st.sidebar.markdown("---")
if st.sidebar.button("Check Service Health"):
    health = get_health_report()
    st.sidebar.success("Health checked")
    st.sidebar.json(health.model_dump())
else:
    st.sidebar.info("Click 'Check Service Health' to pre-flight APIs")

st.sidebar.markdown("---")
st.sidebar.caption("Context bias: South Africa (ZA). OpenAI default, Grok fallback.")

# ---------- Main Area ----------
st.title("South Africa OCR / Fraud POC – Step 1")
st.write(
    "Upload a PDF or image. This step collects raw signals from Azure Document Intelligence "
    "and Azure Vision. Artifacts are saved to `outputs/<run_id>/` for later modules."
)

if module_choice != "Step 1: Acquire Raw (DI & Vision)":
    st.warning("Selected module is a placeholder for now. Implementations will follow. Please use Step 1 for raw capture.")
else:
    with st.expander("Options", expanded=True):
        store_source = st.checkbox("Store source file in outputs/<run_id>/ (for Step 2 PDF style extraction)", value=False)
        call_di = st.checkbox("Call Azure Document Intelligence (DI)", value=True)
        call_vision = st.checkbox("Call Azure Vision (Image Analysis v4)", value=True)
        vision_on_pdf_pages = st.checkbox("If PDF, also run Vision on first N pages", value=True)
        vision_pages_limit = st.number_input("Max PDF pages for Vision", min_value=1, max_value=10, value=3, step=1)

    uploaded = st.file_uploader("Upload PDF or image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])

    col_run, col_sp, col_id = st.columns([1, 3, 2], vertical_alignment="center")
    with col_id:
        custom_run_id = st.text_input("Run ID (optional)")
    with col_run:
        run_now = st.button("Run Step 1", type="primary")

    # --- run button handler ---
if run_now:
    if not uploaded:
        st.error("Please upload a file first.")
    else:
        run_id = custom_run_id or new_run_id("run")
        file_bytes = uploaded.read()

        out: ModuleOutput = mod1.run(
            (uploaded.name, file_bytes),
            store_source_file=store_source,
            run_id=run_id,
            call_di=call_di,
            call_vision=call_vision,
            vision_on_pdf_pages=vision_on_pdf_pages,
            vision_pages_limit=int(vision_pages_limit),
        )

        # Persist for subsequent reruns (e.g., when clicking a download button)
        st.session_state["step1_out_ok"] = out.ok
        st.session_state["step1_out_msg"] = out.message
        st.session_state["step1_run_id"] = out.run_id
        st.session_state["step1_artifacts"] = out.artifact_paths

# --- helper to display results if we have them in session_state ---
def _render_step1_results():
    if not st.session_state.get("step1_artifacts"):
        return
    ok = st.session_state.get("step1_out_ok", False)
    msg = st.session_state.get("step1_out_msg", "")
    run_id = st.session_state.get("step1_run_id", "run-unknown")
    artifacts = st.session_state["step1_artifacts"]

    if ok:
        st.success(msg)
    else:
        st.error(msg)
    st.write(f"**Run ID:** {run_id}")
    st.write("Artifacts written:")
    st.json(artifacts)

    # Download buttons with stable keys so they survive reruns
    import os
    from core.io import load_json

    for label, path in artifacts.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            st.download_button(
                label=f"Download {label}.json",
                data=data,
                file_name=os.path.basename(path),
                mime="application/json",
                key=f"dl-{label}-{run_id}",
            )
        except Exception as ex:
            st.warning(f"Could not read {path}: {ex}")

    st.subheader("Preview: Combined raw_signals.step1.json")
    try:
        st.json(load_json(artifacts.get("combined", "")))
    except Exception as ex:
        st.warning(f"Preview failed: {ex}")

# Always render last results if present (so they persist after downloads)
_render_step1_results()


# ---------------------- Step 2 UI ----------------------
if module_choice == "Step 2: Preprocess (stub)":
    st.header("Step 2: Preprocess & Enrich")

    step2_col1, step2_col2 = st.columns(2)
    with step2_col1:
        use_last = st.checkbox("Use last Step 1 results in this session", value=True)
    with step2_col2:
        step2_run_id = st.text_input("Run ID for Step 2 (optional; defaults to last or new)", value=st.session_state.get("step1_run_id", ""))

    step1_json = None
    source_hint = None

    if use_last and st.session_state.get("step1_artifacts"):
        artifacts = st.session_state["step1_artifacts"]
        step1_json = artifacts.get("combined")
        source_hint = artifacts.get("source_file")

    st.write("— or —")
    up2 = st.file_uploader("Upload raw_signals.step1.json", type=["json"], key="step2_uploader")
    if up2:
        import json, tempfile, os
        buf = up2.read().decode("utf-8")
        # stash to temp file so mod2 can read a path
        tmpdir = os.path.join("tmp")
        os.makedirs(tmpdir, exist_ok=True)
        tmp_step1 = os.path.join(tmpdir, "uploaded_step1.json")
        with open(tmp_step1, "w", encoding="utf-8") as f:
            f.write(buf)
        step1_json = tmp_step1

    source_path = st.text_input("Optional source file path (outputs/<run_id>/source.pdf if you stored it)")

    run2 = st.button("Run Step 2")
    if run2:
        if not step1_json:
            st.error("Provide Step 1 combined JSON (use last results or upload the file).")
        else:
            from modules import mod2_preprocess as mod2
            rid = step2_run_id or (st.session_state.get("step1_run_id") or new_run_id("run"))
            out2 = mod2.run(step1_json, run_id=rid, source_file_path=source_path or source_hint)
            if out2.ok:
                st.success(out2.message)
                st.json(out2.artifact_paths)
                # Download button
                try:
                    with open(out2.artifact_paths["preprocess"], "r", encoding="utf-8") as f:
                        data = f.read()
                    st.download_button(
                        label="Download preprocess.step2.json",
                        data=data,
                        file_name=os.path.basename(out2.artifact_paths["preprocess"]),
                        mime="application/json",
                        key=f"dl-step2-{rid}"
                    )
                except Exception as ex:
                    st.warning(f"Could not read preprocess file: {ex}")
            else:
                st.error(out2.message)
