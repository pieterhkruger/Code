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
from typing import Optional

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

            # Cache combined Step-1 JSON in session so Step-2 can run and re-render after reruns
            if not st.session_state.get("combined_step1"):
                # Try several common keys your Step-1 module may use
                combined_path = (
                    out.artifact_paths.get("combined_step1")
                    or out.artifact_paths.get("raw_signals")
                    or out.artifact_paths.get("raw_signals_step1")
                    or out.artifact_paths.get("raw_signals_step1_json")
                )
                if combined_path and os.path.exists(combined_path):
                    try:
                        with open(combined_path, "r", encoding="utf-8") as f:
                            st.session_state["combined_step1"] = json.load(f)
                    except Exception:
                        pass

            # Flag that Step-1 is complete so Step-2 panel can render automatically
            st.session_state["step1_done"] = True


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
    from core.io import load_json
    import os, json, mimetypes

    for label, path in artifacts.items():
        try:
            if not path or not os.path.exists(path):
                st.warning(f"{label}: not found at {path}")
                continue

            mime, _ = mimetypes.guess_type(path)
            ext = os.path.splitext(path)[1].lower()

            # JSON preview (text mode)
            if ext == ".json":
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        st.json(json.load(f))
                    except Exception:
                        f.seek(0)
                        st.code(f.read())
                st.download_button(
                    label=f"Download {os.path.basename(path)}",
                    data=open(path, "rb").read(),          # send bytes even for JSON (safer)
                    file_name=os.path.basename(path),
                    mime=mime or "application/json",
                    key=f"dl-{label}-{run_id}",
                )

            # Everything else (PDF, images, etc.) — binary
            else:
                st.download_button(
                    label=f"Download {os.path.basename(path)}",
                    data=open(path, "rb").read(),
                    file_name=os.path.basename(path),
                    mime=mime or "application/octet-stream",
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

def _render_step2_seg_debug_from_session():
    """Render the segmentation summary/table from session artifacts (survives reruns)."""
    artifacts = st.session_state.get("step2_artifacts") or {}
    if not artifacts:
        return
    seg_pages = []
    # Prefer the separate artifact
    seg_debug_path = artifacts.get("segmentation_debug")
    if seg_debug_path and os.path.exists(seg_debug_path):
        try:
            with open(seg_debug_path, "r", encoding="utf-8") as f:
                seg_json = json.load(f)
            seg_pages = seg_json.get("pages") or []
        except Exception as e:
            st.warning(f"Could not read segmentation debug file: {e}")

    # Fallback to embedded debug inside preprocess.step2.json
    if not seg_pages and artifacts.get("preprocess") and os.path.exists(artifacts["preprocess"]):
        try:
            with open(artifacts["preprocess"], "r", encoding="utf-8") as f:
                pre_json = json.load(f)
            seg_pages = (((pre_json.get("debug") or {}).get("segmentation") or {}).get("pages")) or []
        except Exception as e:
            st.warning(f"Could not read embedded segmentation debug: {e}")

    if not seg_pages:
        return

    # Summarize
    rows = []
    for pg in seg_pages:
        d = pg.get("debug") or {}
        method = d.get("method") or d.get("reason") or ""
        best_cp = d.get("best_score")
        if best_cp is None:
            best_cp = (d.get("best_change_point") or {}).get("score")
        rows.append({
            "page": pg.get("page_index"),
            "accepted": pg.get("accepted"),
            "boundary_index": pg.get("boundary_index"),
            "method": method,
            "best_score": best_cp,
        })

    st.subheader("Segmentation debug summary")
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows))
    except Exception:
        st.table(rows)

    with st.expander("Per-page debug details"):
        for pg in seg_pages:
            st.markdown(
                f"**Page {pg.get('page_index')}** — "
                f"accepted: {pg.get('accepted')}, "
                f"boundary_index: {pg.get('boundary_index')}"
            )
            st.json(pg.get("debug") or {})


# ---------------------- Step 2 UI ----------------------
if module_choice == "Step 2: Preprocess (stub)" or st.session_state.get("step1_done"):
    # In app.py, right after you finish Step 1 and have `combined_step1` (dict)
    import modules.mod2_preprocess as mod2

    st.markdown("### Next step")

    if st.session_state.get("combined_step1"):
        if st.button("Run Step 2 on these results"):
            out2 = mod2.run(
                st.session_state["combined_step1"],
                run_id=current_run_id,
                source_file_path=source_path
            )
            # Persist Step-2 results so they survive reruns (e.g., after any download click)
            st.session_state["step2_out_ok"] = out2.ok
            st.session_state["step2_out_msg"] = out2.message
            st.session_state["step2_run_id"] = out2.run_id
            st.session_state["step2_artifacts"] = out2.artifact_paths
        else:
            st.info("Step 1 must complete before running Step 2.")

    # if st.button("Run Step 2 on these results"):
    #     out2 = mod2.run(combined_step1, run_id=current_run_id, source_file_path=source_path)
    #     if out2.ok:
    #         st.success("Step 2 complete.")
    #         st.write(out2.message)
    #         with open(out2.artifact_paths["preprocess"], "rb") as fh:
    #             st.download_button("Download Step 2 JSON", data=fh.read(),
    #                             file_name="preprocess.step2.json", mime="application/json")

    # If Step-2 artifacts exist in session, render all download buttons every rerun
    if st.session_state.get("step2_artifacts"):
        if st.session_state.get("step2_out_ok"):
            st.success(st.session_state.get("step2_out_msg", "Step 2 complete."))
        else:
            st.warning(st.session_state.get("step2_out_msg", "Step 2 finished with warnings."))

        for label, path in st.session_state["step2_artifacts"].items():
            try:
                with open(path, "rb") as fh:
                    data = fh.read()
                fname = os.path.basename(path)
                # Uniform download buttons, including segmentation.debug.step2.json and eval template
                st.download_button(
                    label=f"Download: {fname}",
                    data=data,
                    file_name=fname,
                    mime="application/json",
                    key=f"dl_step2_{label}_{st.session_state.get('step2_run_id')}"
                )
            except Exception as e:
                st.warning(f"{label}: {e}")
    #----------------------------------------------------------------------------------    

    st.header("Step 2: Preprocess & Enrich — Text Styles")

    # Defaults / context from last run
    use_last = st.checkbox("Use last Step 1 results in this session", value=True)
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
        tmpdir = os.path.join("tmp")
        os.makedirs(tmpdir, exist_ok=True)
        tmp_step1 = os.path.join(tmpdir, "uploaded_step1.json")
        with open(tmp_step1, "w", encoding="utf-8") as f:
            f.write(buf)
        step1_json = tmp_step1

    source_path = st.text_input("Optional source file path (PDF/image). If you ticked 'Store source file' in Step 1, it's prefilled here.", value=source_hint or "")

    st.markdown("### Backends and weights")
    from services import text_style_panel as tsp
    colA, colB = st.columns(2)
    with colA:
        en_pm = st.checkbox("Enable PyMuPDF (true PDF spans)", value=tsp.ENABLE_PYMUPDF)
        en_az = st.checkbox("Enable Azure (DI v4 → FR v3)", value=tsp.ENABLE_AZURE)
        en_te = st.checkbox("Enable Tesseract + WFA", value=tsp.ENABLE_TESSERACT)
        en_vi = st.checkbox("Enable Vision pixel ROI metrics", value=tsp.ENABLE_VISION_PIXEL)
    with colB:
        wt_pm = st.slider("Weight: PyMuPDF", 0.0, 1.0, float(tsp.WEIGHT_PYMUPDF), 0.05)
        wt_az = st.slider("Weight: Azure", 0.0, 1.0, float(tsp.WEIGHT_AZURE), 0.05)
        wt_te = st.slider("Weight: Tesseract", 0.0, 1.0, float(tsp.WEIGHT_TESSERACT), 0.05)
        wt_vi = st.slider("Weight: Vision pixel", 0.0, 1.0, float(tsp.WEIGHT_VISION_PIXEL), 0.05)

    include_ops = st.checkbox("Include per-service opinions in JSON (for audit)", value=tsp.INCLUDE_BACKEND_OPINIONS)

    run2 = st.button("Run Step 2 (Build Text Style Panel)", type="primary")
    if run2:
        if not step1_json:
            st.error("Provide Step 1 combined JSON (use last results or upload the file).")
        else:
            from modules import mod2_preprocess as mod2
            rid = step2_run_id or (st.session_state.get("step1_run_id") or new_run_id("run"))
            out2 = mod2.run(
                step1_json, run_id=rid, source_file_path=source_path or source_hint,
                service_toggles={"pymupdf": en_pm, "azure": en_az, "tesseract": en_te, "vision_pixel": en_vi},
                weights={"pymupdf": wt_pm, "azure": wt_az, "tesseract": wt_te, "vision_pixel": wt_vi},
                include_backend_opinions=include_ops,
            )
            st.session_state["step2_out_ok"] = out2.ok
            st.session_state["step2_out_msg"] = out2.message
            st.session_state["step2_run_id"] = out2.run_id
            st.session_state["step2_artifacts"] = out2.artifact_paths
            st.session_state["step2_payload"] = out2.payload

            if out2.ok:
                st.success(out2.message)

                # show a JSON preview of the main file (optional)
                if "preprocess" in out2.artifact_paths:
                    try:
                        with open(out2.artifact_paths["preprocess"], "r", encoding="utf-8") as f:
                            st.expander("Preview: preprocess.step2.json (first 4000 chars)").write(f.read()[:4000])
                    except Exception as e:
                        st.warning(f"Preview failed: {e}")

                # render a download button for every artifact the module returned
                for label, path in out2.artifact_paths.items():
                    try:
                        with open(path, "rb") as fh:
                            data = fh.read()
                        fname = os.path.basename(path)
                        st.download_button(
                            label=f"Download: {fname}" if label == "preprocess" else f"Download: {label}.json" if not fname.endswith(".json") else f"Download: {fname}",
                            data=data,
                            file_name=fname,
                            mime="application/json",
                            key=f"dl_step2_{label}",
                        )
                    except Exception as e:
                        st.warning(f"{label}: {e}")

            _render_step2_seg_debug_from_session()

    # --- Render Step 2 results if available ---
    if st.session_state.get("step2_artifacts"):
        ok = st.session_state.get("step2_out_ok", False)
        msg = st.session_state.get("step2_out_msg", "")
        run_id = st.session_state.get("step2_run_id", "run-unknown")
        artifacts2 = st.session_state["step2_artifacts"]
        payload2 = st.session_state.get("step2_payload")

        if ok:
            st.success(msg)
        else:
            st.error(msg)
        st.write(f"**Run ID:** {run_id}")
        st.write("Artifacts written:")
        st.json(artifacts2)

        # Show preview + allow downloads
        import json, os, mimetypes
        for label, path in artifacts2.items():
            try:
                if not path or not os.path.exists(path):
                    st.warning(f"{label}: not found at {path}")
                    continue
                mime, _ = mimetypes.guess_type(path)
                ext = os.path.splitext(path)[1].lower()
                if ext == ".json":
                    with open(path, "r", encoding="utf-8") as f:
                        try:
                            st.json(json.load(f))
                        except Exception:
                            f.seek(0)
                            st.code(f.read())
                    st.download_button(
                        label=f"Download {os.path.basename(path)}",
                        data=open(path, "rb").read(),
                        file_name=os.path.basename(path),
                        mime=mime or "application/json",
                        key=f"dl-step2-{label}-{run_id}",
                    )
                else:
                    st.download_button(
                        label=f"Download {os.path.basename(path)}",
                        data=open(path, "rb").read(),
                        file_name=os.path.basename(path),
                        mime=mime or "application/octet-stream",
                        key=f"dl-step2-{label}-{run_id}",
                    )
            except Exception as ex:
                st.warning(f"Could not read {path}: {ex}")

        _render_step2_seg_debug_from_session()
        
    st.markdown("---")
    st.markdown("### Evaluate services against your truth labels")
    st.write("Download `textstyles.eval_template.step2.json`, fill `truth.bold` / `truth.italic`, then upload it here.")
    up_eval = st.file_uploader("Upload adjudicated JSON", type=["json"], key="step2_eval_uploader")
    if up_eval:
        try:
            import json
            obj = json.loads(up_eval.read().decode("utf-8"))
            from modules import mod2_preprocess as mod2
            metrics = mod2.evaluate_adjudicated(obj)
            if "error" in metrics:
                st.error(metrics["error"])
            else:
                st.json(metrics)
        except Exception as ex:
            st.error(f"Could not evaluate: {ex}")
