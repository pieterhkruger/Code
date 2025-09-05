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
- Type-agnostic, location-agnostic page segmentation:
    • Primary: generic change-point detection over sorted lines using a blended score
      (vertical gap, left-margin shift, height shift, digit-share shift, field:value ratio shift).
    • Adaptive acceptance: absolute threshold OR statistical outlier (z-score / percentile).
    • Fallback: 1-D k-means on line y-centers (k=2) with separation heuristics.
- Writes for each Vision page:
    read.documents = [doc_A, doc_B]   # zero/one/two segments; no semantics
    read.read_sections = {"invoice": doc_A.lines, "receipt": doc_B.lines}  # compatibility only
    read._sectioning_debug = {...}
- Additionally persists a run-level debug artifact:
    segmentation.debug.step2.json

NEW in this drop-in:
- We also embed the same debug under:
    preprocess.step2.json → debug.segmentation.pages
  so you can share it even if the UI doesn’t render a separate download button.
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

# Optional centralized text-style panel (safe no-op if absent)
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

# Absolute decision threshold; adaptive acceptance may override when strong outlier
SPLIT_SCORE_THRESHOLD = 1.20

# Adaptive acceptance guards (no NumPy)
MIN_SCORE_FOR_ADAPTIVE = 0.70       # don’t accept vanishing splits even if they’re top
ZSCORE_MIN_IMPROVEMENT = 2.0        # best >= mean + 2*std
PERCENTILE_ACCEPT = 0.90            # best >= 90th percentile
MIN_CANDIDATES_FOR_ADAPTIVE = 5

# K-means fallback (1-D on y-centers)
USE_KMEANS_FALLBACK = True
KMEANS_MAX_ITERS = 20
KMEANS_MIN_SIZE_FRACTION = 0.2      # each cluster must have at least 20% of lines
KMEANS_MIN_SEPARATION = 0.25        # |c2-c1| / page_span must be >= 0.25 to accept

# Heading extraction (for LLM context)
TOP_N_TALLEST_FOR_HEADING = 3
TOP_STRIP_FRACTION = 0.25           # consider top 25% of segment height for heading pick

# Regex for generic “FIELD: value” / “FIELD = value”
RE_FIELD_VALUE = re.compile(r"\b[A-Z]{2,8}\s*[:=]\s*\S")


MIN_BLOCK_FRACTION = 0.15   # each side must have ≥15% of lines for change-point acceptance

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

def _page_span(lines: List[Dict[str, Any]]) -> float:
    ys = [ln["bbox"][1] for ln in lines]
    ye = [ln["bbox"][1] + ln["bbox"][3] for ln in lines]
    if not ys or not ye:
        return 1.0
    return max(ye) - min(ys) or 1.0

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

def _score_split(lines_sorted: List[Dict[str, Any]], i: int, page_w: float, med_h: float
) -> Tuple[float, Dict[str, float]]:
    """
    Score a candidate split between lines i-1 and i (0<i<len).
    Returns (score, components).
    """
    top_block = lines_sorted[:i]
    bot_block = lines_sorted[i:]
    if not top_block or not bot_block:
        return 0.0, {"gap": 0.0, "left": 0.0, "h": 0.0, "digit": 0.0, "field": 0.0}

    # Vertical gap between blocks (normalized)
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

    return score, {"gap": max(0.0, gap), "left": left_shift, "h": h_shift, "digit": digit_delta, "field": field_delta}

def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = float(len(values))
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(var)

def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    k = max(0, min(len(arr) - 1, int(round((len(arr) - 1) * p))))
    return arr[k]

def _accept_split_abs_or_adaptive(best: float, all_scores: List[float]) -> Tuple[bool, str, Dict[str, float]]:
    # Absolute threshold
    if best >= SPLIT_SCORE_THRESHOLD:
        return True, "abs_threshold", {"best": best, "threshold": SPLIT_SCORE_THRESHOLD}

    # Adaptive rules when we have enough candidates
    if len(all_scores) >= MIN_CANDIDATES_FOR_ADAPTIVE and best >= MIN_SCORE_FOR_ADAPTIVE:
        mean, std = _mean_std(all_scores)
        p90 = _percentile(all_scores, PERCENTILE_ACCEPT)
        if std > 1e-6 and best >= mean + ZSCORE_MIN_IMPROVEMENT * std:
            return True, "zscore", {"best": best, "mean": mean, "std": std}
        if best >= p90:
            return True, "percentile", {"best": best, "p90": p90, "mean": mean}

    return False, "below_threshold", {"best": best, "threshold": SPLIT_SCORE_THRESHOLD}

def _kmeans_1d(yvals: List[float], max_iters: int = KMEANS_MAX_ITERS) -> Tuple[Tuple[float, float], List[int]]:
    """
    Simple k=2 k-means on 1-D values. Returns (centroids c1,c2 in sorted order), labels.
    """
    if not yvals or len(yvals) < 2:
        return (0.0, 0.0), [0] * len(yvals)
    y_sorted = sorted(yvals)
    c1, c2 = y_sorted[len(y_sorted) // 3], y_sorted[2 * len(y_sorted) // 3]  # spaced initialization
    for _ in range(max_iters):
        lab = [0 if abs(v - c1) <= abs(v - c2) else 1 for v in yvals]
        if not any(lab) or all(lab):
            break
        n1 = sum(1 for l in lab if l == 0); n2 = len(lab) - n1
        new_c1 = sum(v for v, l in zip(yvals, lab) if l == 0) / float(n1) if n1 else c1
        new_c2 = sum(v for v, l in zip(yvals, lab) if l == 1) / float(n2) if n2 else c2
        if abs(new_c1 - c1) < 1e-6 and abs(new_c2 - c2) < 1e-6:
            c1, c2 = new_c1, new_c2
            break
        c1, c2 = new_c1, new_c2
    if c1 > c2:
        c1, c2 = c2, c1
    labels = [0 if abs(v - c1) <= abs(v - c2) else 1 for v in yvals]
    return (c1, c2), labels

def _choose_boundary_generic(lines: List[Dict[str, Any]]) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Choose index boundary i so that lines[:i], lines[i:] form two blocks.
    Primary: generic change-point with adaptive acceptance.
    Fallback (optional): 1-D k-means on y-centers with separation heuristics.
    Returns (index or None, debug_dict).
    """
    dbg: Dict[str, Any] = {}
    if not lines or len(lines) < 2:
        return None, {"reason": "insufficient_lines"}

    lines_sorted = _sorted_lines(lines)
    page_w = _page_width(lines_sorted)
    med_h = _median_line_height(lines_sorted)
    span = _page_span(lines_sorted)

    # Evaluate all splits
    best_i = None
    best_score = 0.0
    comps_at_best: Dict[str, float] = {}
    cand_scores: List[float] = []
    cand_dbg: List[Dict[str, float]] = []

    for i in range(1, len(lines_sorted)):
        score, comps = _score_split(lines_sorted, i, page_w, med_h)
        cand_scores.append(score)
        cand_dbg.append({"i": i, "score": score, **comps})
        if score > best_score:
            best_score = score
            best_i = i
            comps_at_best = comps

    # Reject trivial top/bottom splits for change-point unless both blocks are large enough
    # Size guard: both blocks must have enough lines for change-point to be accepted
    min_count = max(1, int(MIN_BLOCK_FRACTION * len(lines_sorted)))

    # Decide by absolute/adaptive acceptance
    accepted, method, meta = _accept_split_abs_or_adaptive(best_score, cand_scores)

    # Apply size guard BEFORE returning
    if accepted and best_i is not None:
        if best_i < min_count or (len(lines_sorted) - best_i) < min_count:
            accepted = False
            method = "below_min_block_fraction"
            meta = {"best": best_score, "min_block_fraction": MIN_BLOCK_FRACTION}

    # Only return if still accepted; otherwise continue to k-means fallback
    if accepted and best_i is not None:
        dbg.update({
            "method": f"change_point::{method}",
            "best_index": best_i,
            "best_score": best_score,
            "components": comps_at_best,
            "threshold": SPLIT_SCORE_THRESHOLD,
            "candidates_sample": cand_dbg[:30],
            "page_width": page_w,
            "median_line_height": med_h,
            "page_span": span,
        })
        return best_i, dbg


    # Fallback: k-means on y-centers
    if USE_KMEANS_FALLBACK:
        ymid = [ln["bbox"][1] + ln["bbox"][3] / 2.0 for ln in lines_sorted]
        (c1, c2), labels = _kmeans_1d(ymid, KMEANS_MAX_ITERS)
        n1 = sum(1 for l in labels if l == 0)
        n2 = len(labels) - n1
        sep = abs(c2 - c1) / (span or 1.0)
        if sep >= KMEANS_MIN_SEPARATION and \
           n1 >= KMEANS_MIN_SIZE_FRACTION * len(labels) and \
           n2 >= KMEANS_MIN_SIZE_FRACTION * len(labels):
            switch_idx = None
            for i in range(1, len(labels)):
                if labels[i - 1] != labels[i]:
                    switch_idx = i
                    break
            if switch_idx is not None:
                dbg.update({
                    "method": "kmeans_fallback",
                    "centroids": [c1, c2],
                    "cluster_sizes": [n1, n2],
                    "separation_norm_span": sep,
                    "kmeans_boundary_index": switch_idx,
                    "page_span": span,
                    "note": "Accepted by k-means fallback",
                })
                return switch_idx, dbg

        dbg.update({
            "method": "none",
            "reason": "no_change_point_and_kmeans_rejected",
            "best_change_point": {"index": best_i, "score": best_score, "accepted": False, "accept_method": method, **meta},
            "kmeans": {
                "centroids": [c1, c2], "sizes": [n1, n2],
                "sep_norm_span": sep, "min_sep_norm_span": KMEANS_MIN_SEPARATION,
                "min_size_fraction": KMEANS_MIN_SIZE_FRACTION
            },
            "page_span": span,
            "candidates_sample": cand_dbg[:30],
        })
        return None, dbg

    dbg.update({
        "method": "none",
        "reason": "no_split_above_threshold_no_fallback",
        "best_change_point": {"index": best_i, "score": best_score, "accepted": False, "accept_method": method, **meta},
        "page_span": span,
        "candidates_sample": cand_dbg[:30],
    })
    return None, dbg

def _segment_to_features(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute lightweight features for LLM consumption."""
    if not lines:
        return {"line_count": 0, "digit_share": 0.0, "field_value_lines": 0, "heading_candidates": []}

    all_text = "".join((ln.get("text") or "") for ln in lines).lower()
    ds = _digit_share(all_text)
    fvl = sum(1 for ln in lines if RE_FIELD_VALUE.search(ln.get("text") or ""))

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

def _assign_documents_inplace(step1: Dict[str, Any], debug_pages: List[Dict[str, Any]]) -> None:
    """
    For each Vision page, split lines into two document-like segments using
    generic change-point detection with k-means fallback. Writes:
        read.documents = [doc_A, doc_B]  (either or both may be present)
        read.read_sections = {"invoice": doc_A.lines, "receipt": doc_B.lines}  # compatibility only
        read._sectioning_debug = {...}
    Collects a compact debug summary per page into debug_pages.
    """
    pages = (step1.get("vision") or {}).get("pages") or []
    for pg in pages:
        res = (pg.get("result") or {})
        read = res.get("read") or {}
        lines: List[Dict[str, Any]] = read.get("lines") or []
        if not lines:
            continue

        idx, dbg = _choose_boundary_generic(lines)
        lines_sorted = _sorted_lines(lines)

        if idx is None:
            all_bbox: Optional[List[float]] = None
            for ln in lines_sorted:
                all_bbox = _bbox_union(all_bbox, ln.get("bbox"))
            doc_A = {
                "id": "doc_A",
                "lines": lines_sorted,
                "bbox": all_bbox,
                "signals": _segment_to_features(lines_sorted),
            }
            read["documents"] = [doc_A]
            read["read_sections"] = {"invoice": doc_A["lines"], "receipt": []}
            read["_sectioning_debug"] = {**dbg, "boundary_index": None}
        else:
            top_lines = lines_sorted[:idx]
            bot_lines = lines_sorted[idx:]
            top_box: Optional[List[float]] = None
            for ln in top_lines:
                top_box = _bbox_union(top_box, ln.get("bbox"))
            bot_box: Optional[List[float]] = None
            for ln in bot_lines:
                bot_box = _bbox_union(bot_box, ln.get("bbox"))
            doc_A = {"id": "doc_A", "lines": top_lines, "bbox": top_box, "signals": _segment_to_features(top_lines)}
            doc_B = {"id": "doc_B", "lines": bot_lines, "bbox": bot_box, "signals": _segment_to_features(bot_lines)}
            read["documents"] = [doc_A, doc_B]
            read["read_sections"] = {"invoice": top_lines, "receipt": bot_lines}
            read["_sectioning_debug"] = dbg

        debug_pages.append({
            "page_index": int(pg.get("page_index", 0)),
            "line_count": len(lines_sorted),
            "accepted": idx is not None,
            "boundary_index": idx,
            "debug": read.get("_sectioning_debug"),
        })

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
    segmentation_debug_pages: List[Dict[str, Any]] = []
    try:
        _assign_documents_inplace(step1, segmentation_debug_pages)
    except Exception as ex:
        step1.setdefault("_warnings", []).append(f"Vision segmentation failed: {ex}")

    _normalize_handwriting_inplace(step1)

    # Flatten Vision lines + basic stats (unchanged)
    lines = _extract_vision_lines(step1)
    stats = _height_stats(lines)

    # Save segmentation debug artifact (separate file)
    seg_debug_path = os.path.join(out_dir, "segmentation.debug.step2.json")
    try:
        save_json(seg_debug_path, {"pages": segmentation_debug_pages})
    except Exception:
        # Non-fatal
        pass

    # Main enriched payload (now also embeds debug)
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
        "debug": {            # <— embedded debug for convenience
            "segmentation": {"pages": segmentation_debug_pages}
        },
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
        ok=True,
        message="Preprocess complete (generic segmentation + k-means fallback; styles normalized).",
        payload=None,
        artifact_paths={
            "preprocess": preprocess_path,
            "segmentation_debug": seg_debug_path,   # <— UI can render this
            **tstyle_artifacts,
        },
    )

def evaluate_adjudicated(obj: dict) -> dict:
    if not _HAS_TSTYLE:
        return {"error": "Evaluator not available"}
    items = obj.get("items") or []
    return evaluate_against_truth(items)

# --- END OF FILE: modules/mod2_preprocess.py ---
