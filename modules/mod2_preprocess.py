"""
modules/mod2_preprocess.py
==========================
Roadmap Step 2: Preprocess/enrich the raw signals for downstream modules.

This file is a *drop-in replacement* that preserves Step-2 behavior (producing
`preprocess.step2.json`) and **adds** optional text-style fusion artifacts
(`textstyles.*.step2.json`) without breaking anything else.

What this module does
---------------------
1) Load Step-1 combined signals (the JSON produced by Step 1).
2) Normalize OCR geometry into a simple list of lines with pixel bboxes.
3) Compute basic height statistics per page + overall (handy for later heuristics).
4) (If a PDF source is available) extract *true* font/style spans via PyMuPDF.
5) Write `outputs/<run_id>/preprocess.step2.json` for downstream modules.
6) **Optionally**, build a centralized text-style panel that fuses:
   - PyMuPDF spans
   - Azure DI v4 (with FR v3 fallback)
   - Tesseract + WordFontAttributes
   - Azure Vision pixel ROI metrics
   It writes:
     - `textstyles.opinions.step2.json`
     - `textstyles.consensus.step2.json`
     - `textstyles.eval_template.step2.json` (for you to fill `truth.bold/italic`)
   These are produced on a best-effort basis and will NOT cause Step-2 to fail.

Inputs
------
- step1_combined_json_path: path to `raw_signals.step1.json` (or a dict already loaded)
- run_id: ID used for the outputs/ subfolder
- source_file_path: optional path to the original PDF/image to enable PDF spans
  and pixel-based metrics; Step 1 can store this for you.

Outputs
-------
- outputs/<run_id>/preprocess.step2.json
- (optional) textstyles.*.step2.json

Notes
-----
- All external service calls in the text-style panel are exception-shielded.
- If you don’t want the extra artifacts, simply ignore them. Your previous
  downstream steps can keep using only `preprocess.step2.json`.
"""

from __future__ import annotations

import os
import io
import json
from typing import Any, Dict, List, Optional, Tuple, Union

# --- Internal project helpers (existing in your repo) ---
from core.io import load_json, save_json
from core.models import ModuleOutput
from config import ensure_output_dir

# --- Optional style panel (new) ---
try:
    from services.text_style_panel import build_text_style_panel
    _HAS_TSTYLE = True
except Exception:
    _HAS_TSTYLE = False

# --- Optional deps for PDF parsing ---
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_step1(step1_path_or_payload: Any) -> Dict[str, Any]:
    """Accept a path or a dict that already contains Step-1 combined signals."""
    if isinstance(step1_path_or_payload, str):
        return load_json(step1_path_or_payload)
    return dict(step1_path_or_payload or {})

def _extract_vision_lines(step1: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten Azure Vision READ 'lines' -> [{'page', 'text', 'bbox':[x,y,w,h]}].
    Expects Step 1 to have the structure:
      step1['vision']['pages'][i]['result']['read']['lines'] with 'bbox' in pixels.
    If not present, returns an empty list.
    """
    out: List[Dict[str, Any]] = []
    pages = (step1.get("vision") or {}).get("pages") or []
    for p in pages:
        pno = int(p.get("page_index", 0))
        read = (p.get("result") or {}).get("read") or {}
        lines = read.get("lines") or []
        for ln in lines:
            text = (ln.get("text") or "").strip()
            bbox = ln.get("bbox")
            if not text or not bbox or len(bbox) != 4:
                continue
            out.append({"page": pno, "text": text, "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]})
    return out

def _height_stats(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple height statistics overall and by page."""
    by_page: Dict[int, List[float]] = {}
    all_h: List[float] = []
    for ln in lines:
        bb = ln.get("bbox")
        if not bb or len(bb) != 4:
            continue
        h = float(bb[3])
        all_h.append(h)
        by_page.setdefault(int(ln.get("page", 0)), []).append(h)
    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr_sorted = sorted(arr)
        n = len(arr_sorted)
        mean = sum(arr_sorted)/n
        med = arr_sorted[n//2] if n % 2 else 0.5*(arr_sorted[n//2-1] + arr_sorted[n//2])
        return {"count": n, "mean": mean, "median": med, "min": arr_sorted[0], "max": arr_sorted[-1]}
    per_page = {p: _stats(v) for p, v in by_page.items()}
    overall = _stats(all_h)
    return {"overall": overall, "per_page": per_page}

def _extract_pdf_style_spans(source_file_path: Optional[str]) -> List[Dict[str, Any]]:
    """
    Extract font/style spans from a *true* PDF using PyMuPDF.
    Returns a simplified list of spans with text, font, size, bold/italic hints.
    If the file is not a PDF or PyMuPDF is unavailable, returns [].
    """
    if not source_file_path or not os.path.exists(source_file_path):
        return []
    if not source_file_path.lower().endswith(".pdf"):
        return []
    if fitz is None:
        return []
    spans: List[Dict[str, Any]] = []
    try:
        with open(source_file_path, "rb") as f:
            data = f.read()
        doc = fitz.open(stream=data, filetype="pdf")
        for pno in range(len(doc)):
            page = doc[pno]
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    heights = [float(s.get("size", 0.0)) for s in line.get("spans", [])]
                    h_med = sorted(heights)[len(heights)//2] if heights else 0.0
                    for s in line.get("spans", []):
                        text = (s.get("text") or "").strip()
                        if not text:
                            continue
                        fontname = s.get("font") or ""
                        size = float(s.get("size") or 0.0)
                        name_low = fontname.lower()
                        is_bold = any(k in name_low for k in ("bold", "black", "semibold", "demibold", "heavy")) or (h_med and size > 1.10*h_med)
                        is_italic = any(k in name_low for k in ("italic", "oblique"))
                        spans.append({
                            "page": pno,
                            "text": text,
                            "font": fontname,
                            "size": size,
                            "is_bold": bool(is_bold),
                            "is_italic": bool(is_italic),
                        })
        doc.close()
    except Exception:
        return []
    return spans


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    step1_combined_json_path: Union[str, Dict[str, Any]],
    run_id: str,
    *,
    source_file_path: Optional[str] = None,
    # The following kwargs are passed through to the (optional) text-style panel
    service_toggles: Optional[Dict[str, bool]] = None,
    weights: Optional[Dict[str, float]] = None,
    include_backend_opinions: Optional[bool] = None,
) -> ModuleOutput:
    """
    Perform Step-2 preprocessing. This preserves existing behavior of emitting
    `preprocess.step2.json` and *adds* (optionally) text-style fusion artifacts.

    Parameters
    ----------
    step1_combined_json_path : str|dict
        Path to `raw_signals.step1.json` (or the dict already loaded).
    run_id : str
        Output folder tag inside outputs/.
    source_file_path : str|None
        Optional path to the original PDF/image. Improves PDF spans and pixel metrics.
    service_toggles / weights / include_backend_opinions : dict/bool|None
        Optional overrides forwarded to the text-style panel. Ignore if not used.

    Returns
    -------
    ModuleOutput
        - ok/message indicate success/failure of Step-2.
        - artifact_paths includes *preprocess* (always) plus optional *textstyles_* files.
    """
    out_dir = ensure_output_dir(run_id)

    # 1) Load Step-1 combined signals
    try:
        step1 = _load_step1(step1_combined_json_path)
    except Exception as ex:
        return ModuleOutput(
            run_id=run_id,
            module_name="mod2_preprocess",
            ok=False,
            message=f"Could not read Step 1 combined JSON: {ex}",
            payload=None,
            artifact_paths={},
        )

    # 2) Normalize Vision READ lines (if any)
    lines = _extract_vision_lines(step1)
    stats = _height_stats(lines)

    # 3) Extract PDF style spans (true PDF only; best-effort)
    pdf_spans = _extract_pdf_style_spans(source_file_path)

    # 4) Build the enriched Step-2 JSON (backward-compatible)
    enriched: Dict[str, Any] = {
        "summary": {
            "pages": len((step1.get("vision") or {}).get("pages") or []),
            "line_count": len(lines),
            "height_stats": stats,
            "pdf_span_count": len(pdf_spans),
        },
        "lines": lines,        # flattened OCR lines with pixel bboxes
        "pdf_styles": pdf_spans,  # true PDF spans (if available)
    }

    # 5) Save the original Step-2 artifact
    preprocess_path = os.path.join(out_dir, "preprocess.step2.json")
    try:
        save_json(preprocess_path, enriched)
    except Exception as ex:
        return ModuleOutput(
            run_id=run_id,
            module_name="mod2_preprocess",
            ok=False,
            message=f"Failed writing preprocess.step2.json: {ex}",
            payload=None,
            artifact_paths={},
        )

    # 6) (Optional) Build the centralized text-style panel and write its artifacts.
    tstyle_artifacts: Dict[str, str] = {}
    if _HAS_TSTYLE:
        try:
            # Read source bytes for panel if present
            src_bytes, src_ext = None, None
            if source_file_path and os.path.exists(source_file_path):
                with open(source_file_path, "rb") as f:
                    src_bytes = f.read()
                _, ext = os.path.splitext(source_file_path)
                src_ext = ext.lower()

            panel = build_text_style_panel(
                step1_raw=step1,
                source_file_bytes=src_bytes,
                source_file_ext=src_ext,
                service_toggles=service_toggles,
                weights=weights,
                include_backend_opinions=include_backend_opinions,
            )

            opinions_path = os.path.join(out_dir, "textstyles.opinions.step2.json")
            consensus_path = os.path.join(out_dir, "textstyles.consensus.step2.json")
            eval_template_path = os.path.join(out_dir, "textstyles.eval_template.step2.json")

            # Save opinions
            save_json(opinions_path, panel)

            # Save consolidated-only view
            consolidated_only = {
                "items": [
                    {
                        "id": it["id"],
                        "page": it["page"],
                        "text": it["text"],
                        "bbox": it.get("bbox"),
                        "consolidated": it.get("consolidated"),
                    } for it in (panel.get("items") or [])
                ],
                "summary": panel.get("summary"),
            }
            save_json(consensus_path, consolidated_only)

            # Save eval template (opinions + instructions)
            eval_template = dict(panel)
            eval_template["instructions"] = (
                "Fill in truth.bold and truth.italic for as many items as you can. "
                "Set each to true/false; leave as null if unsure. Save and upload this JSON "
                "in Step 2 to compute service accuracy."
            )
            save_json(eval_template_path, eval_template)

            tstyle_artifacts = {
                "textstyles_opinions": opinions_path,
                "textstyles_consensus": consensus_path,
                "textstyles_eval_template": eval_template_path,
            }
        except Exception as ex:
            # Safe: do not fail Step 2 if the panel fails
            tstyle_artifacts = {"textstyles_error": f"{ex}"}

    # 7) Return consistent ModuleOutput with added artifact paths
    return ModuleOutput(
        run_id=run_id,
        module_name="mod2_preprocess",
        ok=True,
        message="Preprocess/enrich complete",
        payload=None,
        artifact_paths={"preprocess": preprocess_path, **tstyle_artifacts},
    )
