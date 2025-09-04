"""
modules/mod2_preprocess.py
==========================
Roadmap Step 2: Preprocess/enrich the raw signals for downstream modules.

Goals:
- Normalize OCR geometry and compute basic text-line metrics (height stats, simple clustering).
- For PDFs, extract *true* font/style spans using PyMuPDF (OCRFont substitute).
- Produce a compact enriched JSON for later steps (structure/prompt/LLM).

Inputs:
- Step 1 combined artifact (raw_signals.step1.json) and (optionally) the stored source file.

Outputs:
- outputs/<run_id>/preprocess.step2.json
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from config import ensure_output_dir
from core.models import ModuleOutput
from core.io import load_json, save_json
from services.pdf_style_probe import extract_pdf_styles

def _load_step1(step1_path_or_payload: Any) -> Dict[str, Any]:
    if isinstance(step1_path_or_payload, str):
        return load_json(step1_path_or_payload)
    # assume dict-like
    return step1_path_or_payload

def _line_height_stats(lines: List[Dict[str, Any]]) -> Dict[str, float]:
    hs = [float(l["bbox"][3]) for l in lines if l.get("bbox") and isinstance(l["bbox"][3], (int, float))]
    if not hs:
        return {"count": 0, "median": 0.0, "mean": 0.0}
    hs_sorted = sorted(hs)
    n = len(hs_sorted)
    median = hs_sorted[n // 2] if n % 2 else 0.5 * (hs_sorted[n//2 - 1] + hs_sorted[n//2])
    mean = sum(hs_sorted) / n
    return {"count": n, "median": float(median), "mean": float(mean)}

def _cluster_lines_vertically(lines: List[Dict[str, Any]], gap_multiplier: float = 1.6) -> List[List[Dict[str, Any]]]:
    """
    Very simple vertical clustering: sort by top y; break when vertical gap > (gap_multiplier * median line height).
    Returns list of clusters (each cluster is a list of line dicts).
    """
    L = [l for l in lines if l.get("bbox")]
    if not L:
        return []
    L.sort(key=lambda l: float(l["bbox"][1]))  # sort by y
    stats = _line_height_stats(L)
    thresh = stats["median"] * gap_multiplier if stats["median"] else 30.0
    clusters: List[List[Dict[str, Any]]] = []
    curr: List[Dict[str, Any]] = []
    last_y = None
    for ln in L:
        y = float(ln["bbox"][1])
        if last_y is None:
            curr = [ln]
        else:
            if (y - last_y) > thresh:
                clusters.append(curr)
                curr = [ln]
            else:
                curr.append(ln)
        last_y = y
    if curr:
        clusters.append(curr)
    return clusters

def run(step1_combined_or_path: Any, run_id: str, source_file_path: Optional[str] = None) -> ModuleOutput:
    """
    step1_combined_or_path:
        - path to outputs/<run_id>/raw_signals.step1.json, or
        - dict already loaded from that JSON.
    source_file_path:
        - optional; if provided and is a .pdf, we will extract true PDF styles via PyMuPDF.
          If omitted but the Step 1 artifacts included 'source_file', use that.
    """
    out_dir = ensure_output_dir(run_id)
    step1 = _load_step1(step1_combined_or_path)

    file_meta = step1.get("file_meta") or {}
    content_type = (file_meta.get("content_type") or "").lower()

    # Gather Vision OCR lines (Step 1 normalized these as pages[].result.read.lines)
    vision_lines: List[Dict[str, Any]] = []
    pages = (step1.get("vision") or {}).get("pages") or []
    # 'pages' may be a list of dicts or list of Pydantic-dumped dicts; handle both
    for p in pages:
        res = (p.get("result") or {})
        read = res.get("read")
        if read and isinstance(read.get("lines"), list):
            for ln in read["lines"]:
                # keep minimal normalized fields
                if ln.get("text"):
                    vision_lines.append({
                        "text": ln["text"],
                        "bbox": ln.get("bbox"),   # [x,y,w,h] from our Step 1 wrapper
                        "page_index": p.get("page_index", 0),
                    })

    # Compute basic line metrics & clusters
    height_stats = _line_height_stats(vision_lines)
    clusters = _cluster_lines_vertically(vision_lines)

    # PDF true styles via PyMuPDF (if possible)
    pdf_spans: List[Dict[str, Any]] = []
    if content_type == "application/pdf":
        # If no explicit source path provided, try to infer from Step 1 artifacts in the typical location
        inferred_source = None
        # users often keep artifact_paths in UI session state; try to reconstruct standard path
        # We can't assume a specific name here reliably; prefer explicit source_file_path if you passed it
        if source_file_path and os.path.exists(source_file_path):
            inferred_source = source_file_path
        # Extract styles if we have a PDF
        try:
            if inferred_source and inferred_source.lower().endswith(".pdf"):
                pdf_spans = extract_pdf_styles(pdf_path=inferred_source)
        except Exception:
            pdf_spans = []

    enriched = {
        "file_meta": file_meta,
        "derived": {
            "vision_line_count": len(vision_lines),
            "vision_height_stats": height_stats,
            "vision_vertical_clusters_count": len(clusters),
        },
        "ocr": {
            "vision_lines": vision_lines,  # kept small; later steps can build graphs from this
        },
        "styles": {
            "pdf_spans": pdf_spans,   # empty if not PDF or source not provided
            "notes": [
                "pdf_spans populated only when source PDF is available (PyMuPDF).",
                "vision_lines contain bbox=[x,y,w,h] normalized in Step 1.",
            ],
        },
        "notes": [
            "Step 2 preprocessing complete",
            "Basic geometry stats and optional PDF styles extracted",
        ],
    }

    out_path = os.path.join(out_dir, "preprocess.step2.json")
    save_json(out_path, enriched)

    return ModuleOutput(
        run_id=run_id,
        module_name="mod2_preprocess",
        ok=True,
        message="Preprocess/enrich complete",
        payload=None,
        artifact_paths={"preprocess": out_path},
    )
