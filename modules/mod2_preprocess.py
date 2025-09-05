# --- START OF FILE: modules/mod2_preprocess.py ---
"""
modules/mod2_preprocess.py
==========================

Step 2: Preprocess / enrich raw signals for downstream modules.

What this module does:
- Accepts either an in-memory Step-1 dict or a path to raw_signals.step1.json
- Writes preprocess.step2.json into the run's output dir
- Normalizes DI styles to suppress obvious handwriting false positives
- Computes basic Vision line stats
- NEW: Type-agnostic page segmentation (generic change-point detection).
  For each Vision page we:
    • sort lines by top-y
    • evaluate every potential split between consecutive lines
      using a score that blends:
         - normalized vertical gap between blocks
         - change in left margin (layout)
         - change in average line height (layout/style proxy)
         - change in digit share (content style)
         - change in “FIELD: value” line ratio (content style)
    • choose the split with the highest score, if it clears a threshold

  We write:
    read.documents = [
        {"id":"doc_A","lines":[...],"bbox":[x,y,w,h],"signals":{...}},
        {"id":"doc_B","lines":[...],"bbox":[x,y,w,h],"signals":{...}}
    ]
  For backward compatibility only, we mirror:
    read.read_sections = {"invoice": doc_A.lines, "receipt": doc_B.lines}
  (No semantics are implied by those names; they are just the first and second segments.)

Signals captured per document segment (for downstream LLMs):
- heading_candidates (tallest lines near the segment top)
- digit_share
- field_value_lines (count of "FIELD: value" or "FIELD = value")
- line_count
"""

from __future__ import annotations

import os
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from core.io import load_json, save_json
from core.models import ModuleOutput
from config import ensure_output_dir

# Optional centralized text-style panel (kept from your project; safe no-op if absent)
try:
    from services.text_style_panel import build_text_style_panel, evaluate_against_truth  # type: ignore
    _HAS_TSTYLE = True
except Exception:
    _HAS_TSTYLE = False

# ---------------------------- Tunables ---------------------------- #

# Change-point scoring weights (blend geometry + style)
W_GAP = 1.0          # vertical gap between blocks (normalized by median line height)
W_LEFT_SHIFT = 0.8   # change in mean left margin (normalized by page width)
W_H_SHIFT = 0.8      # change in mean line height (normalized by median height)
W_DIGIT = 0.6        # change in mean digit share
W_FIELD = 0.6        # change in ratio of FIELD:VALUE lines

# Decision threshold to accept a split (higher = stricter)
SPLIT_SCORE_THRESHOLD = 1.20

# Heading extraction (for LLM context)
TOP_N_TALLEST_FOR_HEADING = 3
TOP_STRIP_FRACTION = 0.25     # consider top 25% of segment height for heading pick

# Regex for POS-like “FIELD: value” / “FIELD = value”
RE_FIELD_VALUE = re.compile(r"\b[A-Z]{2,8}\s*[:=]\s*\S")

# ------------------------------------------------------------------ #

# ---------------------------- Helpers ----------------------------- #

def _load_step1(step1: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    return load_json(step1) if isinstance(step1, str) else dict(step1 or {})

def _extract_vision_lines(step1: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            out.append({
                "page": pno,
                "text": text,
                "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            })
    return out

def _height_stats(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_page: Dict[int, List[float]] = {}
    all_h: List[float] = []
    for ln in lines:
        bb = ln.get("bbox")
        if not bb:
            continue
        h = float(bb[3])
        all_h.append(h)
        by_page.setdefault(int(ln.get("page", 0)), []).append(h)

    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        arr = sorted(arr); n = len(arr)
        mean = sum(arr) / n
        med = arr[n//2] if n % 2 else 0.5 * (arr[n//2 - 1] + arr[n//2])
        return {"count": n, "mean": mean, "median": med, "min": arr[0], "max": arr[-1]}

    return {"overall": _stats(all_h), "per_page": {p: _stats(v) for p, v in by_page.items()}}

def _digit_share(s: str) -> float:
    if not s:
        return 0.0
    digits = sum(ch.isdigit() for ch in s)
    return digits / float(len(s))

def _bbox_union(b1: Optional[List[float]], b2: Optional[List[float]]) -> Optional[List[float]]:
    if not b1: return b2
    if not b2: return b1
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    X0 = min(x1, x2); Y0 = min(y1, y2)
    X1 = max(x1 + w1, x2 + w2); Y1 = max(y1 + h1, y2 + h2)
    return [X0, Y0, X1 - X0, Y1 - Y0]

# -------------------- Generic segmentation logic ------------------- #

def _sorted_lines(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(lines, key=lambda ln: (ln["bbox"][1], ln["bbox"][0]))

def _page_width(lines: List[Dict[str, Any]]) -> float:
    lefts, rights = [], []
    for ln in lines:
        x, y, w, h = ln["bbox"]
        lefts.append(x)
        rights.append(x + w)
    if not lefts or not rights:
        return 1.0
    return max(rights) - min(lefts) or 1.0

def _median_line_height(lines: List[Dict[str, Any]]) -> float:
    hs = [ln["bbox"][3] for ln in lines if ln.get("bbox")]
    if not hs:
        return 1.0
    hs.sort()
    n = len(hs)
    return hs[n//2] if n % 2 else 0.5 * (hs[n//2 - 1] + hs[n//2])

def _block_stats(lines: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute stats used in score for a block of lines."""
    if not lines:
        return {"mean_left": 0.0, "mean_height": 0.0, "mean_digit": 0.0, "field_ratio": 0.0}
    lefts = [ln["bbox"][0] for ln in lines]
    heights = [ln["bbox"][3] for ln in lines]
    digits = [_digit_share((ln.get("text") or "").lower()) for ln in lines]
    field_count = sum(1 for ln in lines if RE_FIELD_VALUE.search(ln.get("text") or ""))
    return {
        "mean_left": sum(lefts) / len(lefts),
        "mean_height": sum(heights) / len(heights),
        "mean_digit": sum(digits) / len(digits),
        "field_ratio": field_count / float(len(lines)),
    }

def _score_split(
    lines_sorted: List[Dict[str, Any]],
    i: int,
    page_w: float,
    med_h: float
) -> Tuple[float, Dict[str, float]]:
    """
    Score a candidate split between lines i-1 and i (0<i<len).
    Returns (score, components).
    """
    top_block = lines_sorted[:i]
    bot_block = lines_sorted[i:]
    if not top_block or not bot_block:
        return 0.0, {"gap": 0.0, "left": 0.0, "h": 0.0, "digit": 0.0, "field": 0.0}

    # Vertical gap between blocks
    gap = (bot_block[0]["bbox"][1] - (top_block[-1]["bbox"][1] + top_block[-1]["bbox"][3])) / (med_h or 1.0)

    s_top = _block_stats(top_block)
    s_bot = _block_stats(bot_block)

    left_shift = abs(s_top["mean_left"] - s_bot["mean_left"]) / (page_w or 1.0)
    h_shift = abs(s_top["mean_height"] - s_bot["mean_height"]) / (med_h or 1.0)
    digit_delta = abs(s_top["mean_digit"] - s_bot["mean_digit"])
    field_delta = abs(s_top["field_ratio"] - s_bot["field_ratio"])

    score = (W_GAP * max(0.0, gap)) + \
            (W_LEFT_SHIFT * left_shift) + \
            (W_H_SHIFT * h_shift) + \
            (W_DIGIT * digit_delta) + \
            (W_FIELD * field_delta)

    return score, {
        "gap": max(0.0, gap),
        "left": left_shift,
        "h": h_shift,
        "digit": digit_delta,
        "field": field_delta,
    }

def _choose_boundary_generic(lines: List[Dict[str, Any]]) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Choose an index 'i' such that lines[:i] and lines[i:] are two coherent blocks.
    Returns (index_or_None, debug_dict). No bias to bottom/top; purely change-point.
    """
    if not lines or len(lines) < 2:
        return None, {"reason": "insufficient_lines"}

    lines_sorted = _sorted_lines(lines)
    page_w = _page_width(lines_sorted)
    med_h = _median_line_height(lines_sorted)

    best_i = None
    best_score = 0.0
    comps_at_best: Dict[str, float] = {}
    cand_scores = []

    for i in range(1, len(lines_sorted)):
        score, comps = _score_split(lines_sorted, i, page_w, med_h)
        cand_scores.append({"i": i, "score": score, **comps})
        if score > best_score:
            best_score = score
            best_i = i
            comps_at_best = comps

    if best_i is not None and best_score >= SPLIT_SCORE_THRESHOLD:
        return best_i, {
            "method": "change_point",
            "best_index": best_i,
            "best_score": best_score,
            "components": comps_at_best,
            "threshold": SPLIT_SCORE_THRESHOLD,
            "median_line_height": med_h,
            "page_width": page_w,
            "candidates": cand_scores[:20],  # shorten debug
        }

    return None, {
        "reason": "no_split_above_threshold",
        "best_index": best_i,
        "best_score": best_score,
        "threshold": SPLIT_SCORE_THRESHOLD,
        "candidates": cand_scores[:20],
    }

def _segment_to_features(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute lightweight features for LLM consumption."""
    if not lines:
        return {
            "line_count": 0, "digit_share": 0.0,
            "field_value_lines": 0, "heading_candidates": []
        }

    # digit share over the whole segment
    all_text = "".join((ln.get("text") or "") for ln in lines).lower()
    ds = _digit_share(all_text)

    # count of FIELD:VALUE lines
    fvl = sum(1 for ln in lines if RE_FIELD_VALUE.search(ln.get("text") or ""))

    # heading candidates: pick tallest N among the top strip of the segment
    ys = [ln["bbox"][1] for ln in lines if ln.get("bbox")]
    ye = [ln["bbox"][1] + ln["bbox"][3] for ln in lines if ln.get("bbox")]
    ymin, ymax = (min(ys) if ys else 0.0), (max(ye) if ye else 0.0)
    top_edge = ymin + (ymax - ymin) * TOP_STRIP_FRACTION
    top_lines = [ln for ln in lines if ln.get("bbox") and ln["bbox"][1] <= top_edge]
    top_lines.sort(key=lambda l: (-float(l["bbox"][3]), l["bbox"][1]))  # tallest first
    heading = [ln.get("text") or "" for ln in top_lines[:TOP_N_TALLEST_FOR_HEADING]]

    return {
        "line_count": len(lines),
        "digit_share": ds,
        "field_value_lines": fvl,
        "heading_candidates": heading
    }

def _assign_documents_inplace(step1: Dict[str, Any]) -> None:
    """
    For each Vision page, split lines into two document-like segments using
    generic change-point detection (no class names, no bottom bias).
    Writes:
        read.documents = [doc_A, doc_B]  (either or both may be present)
        read.read_sections = {"invoice": doc_A.lines, "receipt": doc_B.lines}  # compatibility only
        read._sectioning_debug = {...}
    """
    pages = (step1.get("vision") or {}).get("pages") or []
    for pg in pages:
        res = (pg.get("result") or {})
        read = res.get("read") or {}
        lines: List[Dict[str, Any]] = read.get("lines") or []
        if not lines:
            continue

        # Choose index boundary by generic change-point
        idx, dbg = _choose_boundary_generic(lines)

        if idx is None:
            # No split accepted → single document spanning all lines
            all_bbox: Optional[List[float]] = None
            for ln in lines:
                all_bbox = _bbox_union(all_bbox, ln.get("bbox"))
            doc_A = {
                "id": "doc_A",
                "lines": _sorted_lines(lines),
                "bbox": all_bbox,
                "signals": _segment_to_features(lines),
            }
            read["documents"] = [doc_A]
            read["read_sections"] = {"invoice": doc_A["lines"], "receipt": []}
            read["_sectioning_debug"] = {**dbg, "boundary_index": None}
            continue

        lines_sorted = _sorted_lines(lines)
        top_lines = lines_sorted[:idx]
        bot_lines = lines_sorted[idx:]

        # bboxes
        top_box: Optional[List[float]] = None
        for ln in top_lines:
            top_box = _bbox_union(top_box, ln.get("bbox"))
        bot_box: Optional[List[float]] = None
        for ln in bot_lines:
            bot_box = _bbox_union(bot_box, ln.get("bbox"))

        doc_A = {"id": "doc_A", "lines": top_lines, "bbox": top_box, "signals": _segment_to_features(top_lines)}
        doc_B = {"id": "doc_B", "lines": bot_lines, "bbox": bot_box, "signals": _segment_to_features(bot_lines)}

        read["documents"] = [doc_A, doc_B]
        # Compatibility mirror (legacy field names). No semantics implied.
        read["read_sections"] = {"invoice": top_lines, "receipt": bot_lines}
        read["_sectioning_debug"] = dbg

# -------------------- DI style normalization -------------------- #

def _normalize_handwriting_inplace(step1: Dict[str, Any]) -> None:
    """
    Write di.result.layout.style_normalized: copy of styles with obvious
    false-positives for handwriting set to False.
    Conservative rule: if any span has very short length (<3), set isHandwritten=False.
    """
    di = step1.get("di") or {}
    res = di.get("result") or {}
    layout = res.get("layout") or {}
    styles: List[Dict[str, Any]] = layout.get("styles") or []
    if not styles:
        return
    norm = []
    for st in styles:
        is_hw = bool(st.get("isHandwritten", st.get("is_handwritten", False)))
        spans = st.get("spans") or []
        if spans and any((s.get("length") or 0) < 3 for s in spans):
            is_hw = False
        norm.append({**st, "isHandwritten": is_hw})
    layout["style_normalized"] = norm
    step1["di"]["result"]["layout"] = layout  # reattach

# -------------------- public API -------------------- #

def run(
    step1_combined_json_path: Union[str, Dict[str, Any]],
    run_id: str,
    *,
    source_file_path: Optional[str] = None,
    service_toggles: Optional[Dict[str, bool]] = None,
    weights: Optional[Dict[str, float]] = None,
    include_backend_opinions: Optional[bool] = None,
) -> ModuleOutput:
    out_dir = ensure_output_dir(run_id)

    # Load
    try:
        step1 = _load_step1(step1_combined_json_path)
    except Exception as ex:
        return ModuleOutput(
            run_id=run_id, module_name="mod2_preprocess",
            ok=False, message=f"Could not read Step-1 JSON: {ex}",
            payload=None, artifact_paths={}
        )

    # Enrich (non-destructive; we alter Step-1 copy)
    try:
        _assign_documents_inplace(step1)
    except Exception as ex:
        step1.setdefault("_warnings", []).append(f"Vision segmentation failed: {ex}")

    _normalize_handwriting_inplace(step1)

    # Flatten Vision lines + basic stats (unchanged)
    lines = _extract_vision_lines(step1)
    stats = _height_stats(lines)

    enriched = {
        "summary": {
            "pages": len((step1.get("vision") or {}).get("pages") or []),
            "line_count": len(lines),
            "height_stats": stats,
            "has_di_styles": bool(((step1.get("di") or {}).get("result") or {}).get("layout", {}).get("styles")),
            "has_di_style_normalized": bool(((step1.get("di") or {}).get("result") or {}).get("layout", {}).get("style_normalized")),
        },
        "lines": lines,
        "step1_plus": step1,  # keep the whole enriched Step-1 for downstream use
    }

    # Save Step-2 main artifact
    preprocess_path = os.path.join(out_dir, "preprocess.step2.json")
    try:
        save_json(preprocess_path, enriched)
    except Exception as ex:
        return ModuleOutput(
            run_id=run_id, module_name="mod2_preprocess",
            ok=False, message=f"Failed writing preprocess.step2.json: {ex}",
            payload=None, artifact_paths={}
        )

    # Optional centralized text-style panel (as in your project)
    tstyle_artifacts: Dict[str, str] = {}
    if _HAS_TSTYLE:
        try:
            src_bytes = None; src_ext = None
            if source_file_path and os.path.exists(source_file_path):
                with open(source_file_path, "rb") as f:
                    src_bytes = f.read()
                src_ext = os.path.splitext(source_file_path)[1].lower()
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

            save_json(opinions_path, panel)
            consolidated_only = {
                "items": [
                    {"id": it["id"], "page": it["page"], "text": it["text"], "bbox": it.get("bbox"),
                     "consolidated": it.get("consolidated")}
                    for it in (panel.get("items") or [])
                ],
                "summary": panel.get("summary"),
            }
            save_json(consensus_path, consolidated_only)
            eval_template = dict(panel)
            eval_template["instructions"] = (
                "Fill in truth.bold and truth.italic for as many items as you can. "
                "Use true/false; leave null if unsure. Re-upload in Step 2 to score accuracy."
            )
            save_json(eval_template_path, eval_template)

            tstyle_artifacts = {
                "textstyles_opinions": opinions_path,
                "textstyles_consensus": consensus_path,
                "textstyles_eval_template": eval_template_path,
            }
        except Exception as ex:
            tstyle_artifacts = {"textstyles_error": f"{ex}"}

    return ModuleOutput(
        run_id=run_id, module_name="mod2_preprocess",
        ok=True, message="Preprocess/enrich complete (generic segmentation; styles normalized).",
        payload=None, artifact_paths={"preprocess": preprocess_path, **tstyle_artifacts},
    )

def evaluate_adjudicated(obj: dict) -> dict:
    if not _HAS_TSTYLE:
        return {"error": "Evaluator not available"}
    items = obj.get("items") or []
    return evaluate_against_truth(items)

# --- END OF FILE: modules/mod2_preprocess.py ---
