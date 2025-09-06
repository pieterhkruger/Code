"""
app.py
======
Streamlit UI for Step 1 (Acquire Raw) and Step 2 (Preprocess: Text Styles).

This file:
- Runs Step 1: DI + Vision capture with optional source file persistence.
- Runs Step 2: Preprocess + Text-Style Panel (with duplicate-merging toggle).
- Always shows latest Step-2 artifacts, segmentation table, and backend summary.
"""

from __future__ import annotations

import os
import json
import mimetypes
from typing import Optional

import streamlit as st

from config import new_run_id
from core.models import ModuleOutput
from core.service_health import get_health_report
from core.io import load_json
from modules import mod1_acquire_raw as mod1


# --- helpers for stable JSON preview / downloads ---
def safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as ex:
        st.warning(f"Could not read JSON at {path}: {ex}")
        return None

def render_download_button(label: str, path: str, key: str):
    try:
        with open(path, "rb") as f:
            st.download_button(label, f, file_name=os.path.basename(path), key=key)
    except Exception as ex:
        st.warning(f"Download unavailable ({label}): {ex}")

# --- Service health ribbon ---
def show_health_ribbon():
    try:
        hr = get_health_report()  # pydantic model
        d = hr.model_dump()
    except Exception as ex:
        st.warning(f"Service health check unavailable: {ex}")
        return

    cols = st.columns(4)
    cells = [
        ("Azure DI", d.get("azure_di", {})),
        ("Azure Vision", d.get("azure_vision", {})),
        ("OpenAI", d.get("openai", {})),
        ("Grok", d.get("grok", {})),
    ]

    for (label, s), col in zip(cells, cols):
        ok = bool(s.get("ok"))
        configured = bool(s.get("is_configured"))
        rt = s.get("roundtrip_ms")
        symbol = "✅" if ok else ("⚠️" if configured else "⛔")
        msg = f"{symbol} {label}"
        with col:
            st.markdown(msg)
            if rt is not None:
                st.caption(f"{rt:.0f} ms")
            elif s.get("message"):
                st.caption(str(s.get("message")))


st.set_page_config(page_title="SA OCR/Fraud POC", layout="wide")

# ---------- Sidebar: Module Runner & Health ----------
st.sidebar.title("Module Runner")
module_choice = st.sidebar.selectbox(
    "Choose a module",
    [
        "Step 1: Acquire Raw (DI & Vision)",
        "Step 2: Preprocess (Text Styles)",
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
        store_source = st.checkbox("Store source file in outputs/<run_id>/ (for Step 2 PDF style extraction)", value=True)
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

            st.session_state["step1_out_ok"] = out.ok
            st.session_state["step1_out_msg"] = out.message
            st.session_state["step1_run_id"] = out.run_id
            st.session_state["step1_artifacts"] = out.artifact_paths

            if not st.session_state.get("combined_step1"):
                combined_path = (
                    out.artifact_paths.get("combined")
                    or out.artifact_paths.get("combined_step1")
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

            st.session_state["step1_done"] = True


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

    for label, path in artifacts.items():
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
                    key=f"dl-{label}-{run_id}",
                )
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

# Persist renders after downloads
_render_step1_results()


def _render_step2_seg_debug_from_session():
    """Render the segmentation summary/table from session artifacts (survives reruns)."""
    artifacts = st.session_state.get("step2_artifacts") or {}
    if not artifacts:
        return
    seg_pages = []
    # Prefer separate artifact
    seg_debug_path = artifacts.get("segmentation_debug")
    if seg_debug_path and os.path.exists(seg_debug_path):
        try:
            with open(seg_debug_path, "r", encoding="utf-8") as f:
                seg_json = json.load(f)
            seg_pages = seg_json.get("pages") or []
        except Exception as e:
            st.warning(f"Could not read segmentation debug file: {e}")

    # Fallback to embedded debug
    if not seg_pages and artifacts.get("preprocess") and os.path.exists(artifacts["preprocess"]):
        try:
            with open(artifacts["preprocess"], "r", encoding="utf-8") as f:
                pre_json = json.load(f)
            seg_pages = (((pre_json.get("debug") or {}).get("segmentation") or {}).get("pages")) or []
        except Exception as e:
            st.warning(f"Could not read embedded segmentation debug: {e}")

    if not seg_pages:
        return

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
if module_choice == "Step 2: Preprocess (Text Styles)" or st.session_state.get("step1_done"):
    import modules.mod2_preprocess as mod2
    from services import text_style_panel as tsp

    st.header("Step 2: Preprocess & Enrich — Text Styles")
    show_health_ribbon()

    # Defaults / context from last run
    use_last = st.checkbox("Use last Step 1 results in this session", value=True)
    step2_run_id = st.text_input("Run ID for Step 2 (optional; defaults to last or new)", value=st.session_state.get("step1_run_id", ""))

    step1_json = None
    source_hint = None

    if use_last and st.session_state.get("step1_artifacts"):
        artifacts = st.session_state["step1_artifacts"]
        step1_json = artifacts.get("combined")

    st.write("— or —")
    up2 = st.file_uploader("Upload raw_signals.step1.json", type=["json"], key="step2_uploader")
    if up2:
        buf = up2.read().decode("utf-8")
        tmpdir = os.path.join("tmp")
        os.makedirs(tmpdir, exist_ok=True)
        tmp_step1 = os.path.join(tmpdir, "uploaded_step1.json")
        with open(tmp_step1, "w", encoding="utf-8") as f:
            f.write(buf)
        step1_json = tmp_step1

    combined_step1 = st.session_state.get("combined_step1") or {}
    artifacts = st.session_state.get("step1_artifacts") or {}
    source_hint = (
        artifacts.get("source_file") or
        combined_step1.get("source_file_path")
    )

    source_path = st.text_input("Optional source file path (PDF/image). If you ticked 'Store source file' in Step 1, it's prefilled here.", value=source_hint or "")
    # Status
    if source_path:
        if os.path.exists(source_path):
            st.success(f"Source file found: {source_path}")
        else:
            st.warning(f"Source file not found at path: {source_path} — Tesseract/Vision pixel will be skipped.")
    else:
        st.info("No source path provided. If Step 1 stored the file, it will still work for Tesseract/Vision if UI supplied it earlier.")

    st.markdown("### Backends and weights")
    colA, colB = st.columns(2)
    with colA:
        en_pm = st.checkbox("Enable PyMuPDF (true PDF spans)", value=tsp.ENABLE_PYMUPDF)
        en_az = st.checkbox("Enable Azure (DI v4 → FR v3)", value=tsp.ENABLE_AZURE)
        en_te = st.checkbox("Enable Tesseract + WFA", value=tsp.ENABLE_TESSERACT)
        en_vi = st.checkbox("Enable Vision pixel ROI metrics)", value=tsp.ENABLE_VISION_PIXEL)
    with colB:
        wt_pm = st.slider("Weight: PyMuPDF", 0.0, 1.0, float(tsp.WEIGHT_PYMUPDF), 0.05)
        wt_az = st.slider("Weight: Azure", 0.0, 1.0, float(tsp.WEIGHT_AZURE), 0.05)
        wt_te = st.slider("Weight: Tesseract", 0.0, 1.0, float(tsp.WEIGHT_TESSERACT), 0.05)
        wt_vi = st.slider("Weight: Vision pixel", 0.0, 1.0, float(tsp.WEIGHT_VISION_PIXEL), 0.05)

    include_ops = st.checkbox("Include per-service opinions in JSON (for audit)", value=tsp.INCLUDE_BACKEND_OPINIONS)
    dont_merge_dups = st.checkbox("Style panel audit: don't merge duplicate tokens on the same page", value=False)

    run2 = st.button("Run Step 2 (Build Text Style Panel)", type="primary")
    if run2:
        if not step1_json:
            st.error("Provide Step 1 combined JSON (use last results or upload the file).")
        else:
            rid = step2_run_id or (st.session_state.get("step1_run_id") or new_run_id("run"))
            out2 = mod2.run(
                step1_json, run_id=rid, source_file_path=source_path or source_hint,
                service_toggles={"pymupdf": en_pm, "azure": en_az, "tesseract": en_te, "vision_pixel": en_vi},
                weights={"pymupdf": wt_pm, "azure": wt_az, "tesseract": wt_te, "vision_pixel": wt_vi},
                include_backend_opinions=include_ops,
                merge_duplicates=not dont_merge_dups,
            )
            st.session_state["step2_out_ok"] = out2.ok
            st.session_state["step2_out_msg"] = out2.message
            st.session_state["step2_run_id"] = out2.run_id
            st.session_state["step2_artifacts"] = out2.artifact_paths
            st.session_state["step2_payload"] = out2.payload

    # Always render Step-2 artifacts
    art2 = st.session_state.get("step2_artifacts") or {}
    if art2:
        st.subheader("Step 2 artifacts")
        for k, label in [
            ("preprocess", "Download preprocess.step2.json"),
            ("segmentation_debug", "Download segmentation.debug.step2.json"),
            ("textstyles_consensus", "Download textstyles.consensus.step2.json"),
            ("textstyles_opinions", "Download textstyles.opinions.step2.json"),
            ("textstyles_eval_template", "Download textstyles.eval_template.step2.json"),
        ]:
            p = art2.get(k)
            if p:
                render_download_button(label, p, key=f"dl_{k}")

        # Backend run summary
        ops_path = art2.get("textstyles_opinions")
        if ops_path and os.path.exists(ops_path):
            panel = safe_read_json(ops_path) or {}
            summary = panel.get("summary") or {}
            counts = (summary.get("counts") or {})
            errors = (summary.get("errors") or {})

            st.subheader("Text-styles backend summary")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("PyMuPDF spans", counts.get("pymupdf_spans", 0))
            with c2:
                st.metric("Azure style spans", counts.get("azure_style_spans", 0))
            with c3:
                st.metric("Tesseract words", counts.get("tesseract_words", 0))
            with c4:
                st.metric("Vision word ROIs", counts.get("vision_word_rois", 0))

            if errors:
                for k, v in errors.items():
                    st.warning(f"{k}: {v}")
            else:
                st.success("All requested backends ran without reported errors.")
        else:
            st.info("Run Step 2 to see per-backend counts and any error reasons.")

        # Segmentation summary/table
        pre_path = art2.get("preprocess")
        pre_obj = safe_read_json(pre_path) if pre_path else None

        st.subheader("Segmentation summary")
        if not pre_obj:
            st.info("No preprocess.step2.json found yet.")
        else:
            segs = pre_obj.get("segments") or []
            if not segs:
                st.write("0 segments produced on this run. See debug for candidate scores and reasons.")
            else:
                rows = []
                for s in segs:
                    rows.append({
                        "Segment ID": s.get("id"),
                        "Page": s.get("page"),
                        "Lines": len(s.get("line_indices") or []),
                        "BBox": s.get("bbox"),
                    })
                st.dataframe(rows, use_container_width=True)

            seg_dbg_path = art2.get("segmentation_debug")
            if seg_dbg_path:
                dbg = safe_read_json(seg_dbg_path)
                with st.expander("Segmentation debug (per-page reasons, scores)"):
                    st.json(dbg or {})

# ---- Evaluation helper section ----
st.markdown("---")
st.markdown("### Evaluate services against your truth labels")
st.write("Download `textstyles.eval_template.step2.json`, fill `truth.bold` / `truth.italic`, then upload it here.")
up_eval = st.file_uploader("Upload adjudicated JSON", type=["json"], key="step2_eval_uploader")
if up_eval:
    try:
        obj = json.loads(up_eval.read().decode("utf-8"))
        import modules.mod2_preprocess as mod2
        metrics = mod2.evaluate_adjudicated(obj)
        if "error" in metrics:
            st.error(metrics["error"])
        else:
            st.json(metrics)
    except Exception as ex:
        st.error(f"Could not evaluate: {ex}")
