# --- START OF FILE: services/di_client.py ---
"""
services/di_client.py
=====================

Azure Document Intelligence (DI) wrapper with:
- SDK-first (v4) and REST fallback (fixes 'body' kwarg situations)
- Stable envelope for downstream modules
- Hierarchical layout: content → paragraphs → lines → words (+ flat lists)
- Robust paragraph bboxes and nesting:
    * Use boundingRegions when present
    * Else recover via text-matching to page lines and union their bboxes
- Local fallbacks when DI is unavailable:
    • PDFs → build the same hierarchy via PyMuPDF
    • Images → emit OCR-free pixel proxies (ink/weight metrics)

Filtering & thinning (tunable via globals below):
- Remove single-character paragraphs and lines (single-character words are fine).
- Rule 1: if a line has > MAX_WORDS_PER_LINE_FOR_WORDS, omit its "words".
- Rule 2: if a paragraph has > MAX_LINES_PER_PARAGRAPH_FOR_LINES AND
          avg(words_per_line) > AVG_WORDS_PER_LINE_THRESHOLD, omit its "lines".
"""

from __future__ import annotations

import io
import json
import time
import re
from typing import Any, Dict, List, Optional, Tuple

# -------------------- Global knobs (you can tweak these) -------------------- #
MAX_WORDS_PER_LINE_FOR_WORDS = 20
MAX_LINES_PER_PARAGRAPH_FOR_LINES = 5
AVG_WORDS_PER_LINE_THRESHOLD = 15

DROP_SINGLE_CHAR_LINES = True
DROP_SINGLE_CHAR_PARAGRAPHS = True
# --------------------------------------------------------------------------- #

# Optional deps (import-safe)
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

# Azure SDK (v4). We try SDK first, then REST.
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
    from azure.core.credentials import AzureKeyCredential  # type: ignore
except Exception:
    DocumentIntelligenceClient = None  # type: ignore
    AzureKeyCredential = None  # type: ignore

# PyMuPDF for PDF fallback hierarchy
try:
    import fitz  # type: ignore
except Exception:
    fitz = None  # type: ignore

# Requests for REST fallback
try:
    import requests  # type: ignore
except Exception:
    requests = None  # type: ignore


# ------------------------- Generic helpers ------------------------- #

def _result_to_dict(di_result: Any) -> Dict[str, Any]:
    """Coerce Azure SDK result objects into a plain dict."""
    for attr in ("to_dict", "as_dict", "model_dump"):
        if hasattr(di_result, attr):
            try:
                return getattr(di_result, attr)()
            except Exception:
                pass
    if hasattr(di_result, "to_json"):
        try:
            return json.loads(di_result.to_json())
        except Exception:
            pass
    return {"raw_repr": repr(di_result)}

def _infer_is_pdf(content_type: str) -> bool:
    return (content_type or "").lower().strip() == "application/pdf"

def _poly_to_bbox(poly: Any) -> Optional[List[float]]:
    """
    Convert DI polygon formats to bbox [x, y, w, h].
    Accepts:
      - [x1,y1,x2,y2,x3,y3,x4,y4]
      - [{"x":..,"y":..}, ...]
    """
    if not poly:
        return None
    xs, ys = [], []
    if isinstance(poly, list):
        if poly and isinstance(poly[0], dict):
            for pt in poly:
                xs.append(float(pt.get("x", 0.0)))
                ys.append(float(pt.get("y", 0.0)))
        else:
            it = list(poly)
            if len(it) % 2 != 0:
                it = it[:-1]
            for i in range(0, len(it), 2):
                xs.append(float(it[i])); ys.append(float(it[i + 1]))
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]

def _bbox_union(b1: Optional[List[float]], b2: Optional[List[float]]) -> Optional[List[float]]:
    if not b1: return b2
    if not b2: return b1
    x1, y1, w1, h1 = b1; x2, y2, w2, h2 = b2
    X0 = min(x1, x2); Y0 = min(y1, y2)
    X1 = max(x1 + w1, x2 + w2); Y1 = max(y1 + h1, y2 + h2)
    return [X0, Y0, X1 - X0, Y1 - Y0]

_WORD_SPLIT = re.compile(r"[^\w]+", re.UNICODE)

def _tokens(s: str) -> List[str]:
    return [t for t in _WORD_SPLIT.split((s or "").lower()) if t]

def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b: return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return float(inter) / float(union) if union else 0.0

def _single_char(s: str) -> bool:
    return len((s or "").strip()) <= 1


# ------------------------- PDF fallback hierarchy ------------------------- #

def _pdf_hierarchy_from_words(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Use PyMuPDF 'words' to build:
      paragraphs (≈ blocks) → lines → words with [x,y,w,h]
    Obeys: DROP_SINGLE_CHAR_* and thinning rules.
    """
    if fitz is None:
        return {"content": "", "paragraphs": [], "lines": [], "words": [], "notes": ["PyMuPDF not available"]}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paragraphs: List[Dict[str, Any]] = []
    flat_lines: List[Dict[str, Any]] = []
    flat_words: List[Dict[str, Any]] = []
    content_parts: List[str] = []
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            words = page.get_text("words") or []  # [x0,y0,x1,y1,text, block_no, line_no, word_no]
            by_line: Dict[Tuple[int, int], List[Tuple[float,float,float,float,str,int,int,int]]] = {}
            for w in words:
                x0, y0, x1, y1, text, bno, lno, wno = w
                by_line.setdefault((int(bno), int(lno)), []).append(w)
                flat_words.append({
                    "page": pno,
                    "text": text,
                    "bbox": [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                })

            by_block: Dict[int, List[Tuple[int, List[Tuple[float,float,float,float,str,int,int,int]]]]] = {}
            for (bno, lno), arr in by_line.items():
                by_block.setdefault(int(bno), []).append((int(lno), arr))

            for _, lines in sorted(by_block.items(), key=lambda t: t[0]):
                para_lines = []
                px0 = py0 = 1e9; px1 = py1 = -1e9
                para_text_parts = []
                for _, arr in sorted(lines, key=lambda t: t[0]):
                    lx0=min(a[0] for a in arr); ly0=min(a[1] for a in arr)
                    lx1=max(a[2] for a in arr); ly1=max(a[3] for a in arr)
                    lb=[float(lx0), float(ly0), float(lx1-lx0), float(ly1-ly0)]
                    line_text=" ".join(a[4] for a in sorted(arr, key=lambda x: x[7]))
                    if DROP_SINGLE_CHAR_LINES and _single_char(line_text):
                        # skip single-character line (do not add to flat_lines either)
                        continue
                    words_json=[{"text": a[4], "bbox":[float(a[0]), float(a[1]), float(a[2]-a[0]), float(a[3]-a[1])]}
                                for a in sorted(arr, key=lambda x: x[7])]

                    # --- Rule 1: if a line has too many words, omit its words array
                    shown_words = [] if len(words_json) > MAX_WORDS_PER_LINE_FOR_WORDS else words_json

                    para_lines.append({"page": pno, "text": line_text, "bbox": lb, "words": shown_words})
                    flat_lines.append({"page": pno, "text": line_text, "bbox": lb})
                    para_text_parts.append(line_text)
                    px0, py0 = min(px0,lx0), min(py0,ly0)
                    px1, py1 = max(px1,lx1), max(py1,ly1)

                if not para_lines:
                    continue  # nothing to add for this paragraph

                # --- Rule 2: if paragraph is very long and dense, drop lines
                counts = [len(ln.get("words") or []) for ln in para_lines]
                avg_words = (sum(counts)/len(counts)) if counts else 0.0
                if len(para_lines) > MAX_LINES_PER_PARAGRAPH_FOR_LINES and avg_words > AVG_WORDS_PER_LINE_THRESHOLD:
                    kept_lines = []  # drop lines per rule
                else:
                    kept_lines = para_lines

                para_text = " ".join(para_text_parts)
                if DROP_SINGLE_CHAR_PARAGRAPHS and _single_char(para_text):
                    continue  # drop single-char paragraph

                paragraphs.append({
                    "page": pno,
                    "text": para_text,
                    "bbox": [float(px0), float(py0), float(px1-px0), float(py1-py0)],
                    "lines": kept_lines,
                })
                content_parts.append(para_text)
        doc.close()
    except Exception:
        doc.close()
        return {"content": "", "paragraphs": [], "lines": [], "words": [], "notes": ["PyMuPDF words extraction failed"]}

    return {"content": " ".join(content_parts), "paragraphs": paragraphs, "lines": flat_lines, "words": flat_words}


# ------------------------- DI v4 (SDK/REST) normalizer ------------------------- #

def _normalize_di_v4_layout(di_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize DI v4 JSON to our hierarchical layout.
    Robust against missing paragraph boundingRegions by falling back
    to text-matching with page lines and unioning their bboxes.
    Applies: DROP_SINGLE_CHAR_* and thinning rules.
    """
    ar = di_json.get("analyzeResult") or di_json.get("result") or di_json
    content = ar.get("content", "")

    # Flat lines/words from pages
    flat_lines: List[Dict[str, Any]] = []
    flat_words: List[Dict[str, Any]] = []
    pages = ar.get("pages") or []
    for p in pages:
        pno = int(p.get("pageNumber", 1)) - 1
        for ln in p.get("lines") or []:
            text = ln.get("content") or ""
            if DROP_SINGLE_CHAR_LINES and _single_char(text):
                continue
            bbox = _poly_to_bbox(ln.get("polygon"))
            flat_lines.append({"page": pno, "text": text, "bbox": bbox})
        for wd in p.get("words") or []:
            text = wd.get("content") or ""
            bbox = _poly_to_bbox(wd.get("polygon"))
            flat_words.append({"page": pno, "text": text, "bbox": bbox})

    # Index lines by page for quick matching
    lines_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for ln in flat_lines:
        lines_by_page.setdefault(int(ln["page"]), []).append(ln)

    # Paragraphs; nest lines/words
    paragraphs: List[Dict[str, Any]] = []
    for para in ar.get("paragraphs") or []:
        p_text = para.get("content") or ""
        if DROP_SINGLE_CHAR_PARAGRAPHS and _single_char(p_text):
            continue  # drop single-character paragraph
        p_tokens = _tokens(p_text)

        # Page & bbox from boundingRegions if present
        page_index = None
        bbox = None
        breg_list = para.get("boundingRegions") or []
        if breg_list:
            breg = breg_list[0]
            page_index = int(breg.get("pageNumber", 1)) - 1
            bbox = _poly_to_bbox(breg.get("polygon"))

        # Candidate paragraph lines either by geometry or text match
        cand_lines: List[Dict[str, Any]] = []

        if page_index is not None and bbox:
            # geometry: select line centers inside paragraph bbox
            x, y, w, h = bbox
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            for ln in lines_by_page.get(page_index, []):
                lb = ln.get("bbox")
                if not lb:
                    continue
                cx = lb[0] + lb[2] / 2.0
                cy = lb[1] + lb[3] / 2.0
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    cand_lines.append(ln)

        # If bbox missing or gave no lines, fall back to text matching
        if not cand_lines:
            pages_to_search = [page_index] if page_index is not None else sorted(lines_by_page.keys())
            for pno in pages_to_search:
                for ln in lines_by_page.get(pno, []):
                    lt = ln.get("text") or ""
                    if not lt:
                        continue
                    lt_tokens = _tokens(lt)
                    if not lt_tokens:
                        continue
                    j = _jaccard(lt_tokens, p_tokens)
                    if j >= 0.7 or lt in p_text or p_text in lt:
                        cand_lines.append({**ln, "page": pno})
            if cand_lines and page_index is None:
                page_index = cand_lines[0]["page"]

            # synthesize paragraph bbox from candidate lines
            for ln in cand_lines:
                bbox = _bbox_union(bbox, ln.get("bbox"))

        # Build nested lines with words
        nested_lines: List[Dict[str, Any]] = []
        per_line_counts: List[int] = []
        for ln in cand_lines:
            lb = ln.get("bbox")
            if not lb:
                continue
            words_for_line: List[Dict[str, Any]] = []

            lx0, ly0, lx1, ly1 = lb[0], lb[1], lb[0] + lb[2], lb[1] + lb[3]
            for wd in flat_words:
                if wd["page"] != page_index or not wd.get("bbox"):
                    continue
                wb = wd["bbox"]; wx0, wy0, ww, wh = wb
                wcx, wcy = wx0 + ww / 2.0, wy0 + wh / 2.0
                if lx0 <= wcx <= lx1 and ly0 <= wcy <= ly1:
                    words_for_line.append({"text": wd["text"], "bbox": wd["bbox"]})

            per_line_counts.append(len(words_for_line))
            # --- Rule 1: if a line has too many words, strip its words array
            shown_words = [] if len(words_for_line) > MAX_WORDS_PER_LINE_FOR_WORDS else words_for_line

            nested_lines.append({
                "page": page_index if page_index is not None else 0,
                "text": ln.get("text") or "",
                "bbox": lb,
                "words": shown_words
            })

        # --- Rule 2: paragraph density pruning
        if nested_lines:
            avg_words = (sum(per_line_counts) / float(len(per_line_counts))) if per_line_counts else 0.0
            if len(nested_lines) > MAX_LINES_PER_PARAGRAPH_FOR_LINES and avg_words > AVG_WORDS_PER_LINE_THRESHOLD:
                nested_lines = []

        paragraphs.append({
            "page": page_index if page_index is not None else 0,
            "text": p_text,
            "bbox": bbox,
            "lines": nested_lines,
        })

    layout = {
        "content": content,
        "paragraphs": paragraphs,
        "lines": flat_lines,
        "words": flat_words,
        "styles": ar.get("styles") or [],  # raw styles preserved if present
    }
    return layout


# ------------------------- DI REST (v4) ------------------------- #

def _rest_analyze(*, endpoint: str, key: str, model_id: str, content: bytes,
                  content_type: str, features: List[str], timeout: int) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests not available for REST fallback")

    url = f"{endpoint}/documentintelligence/documentModels/{model_id}:analyze"
    params = {"api-version": "2024-02-29-preview"}
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Accept": "application/json",
        "Content-Type": content_type or "application/octet-stream",
    }
    if features:
        headers["x-di-features"] = ",".join(features)

    resp = requests.post(url, params=params, headers=headers, data=content, timeout=timeout)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"DI REST POST failed: {resp.status_code} {resp.text}")

    op_url = resp.headers.get("operation-location")
    if not op_url:
        try:
            return resp.json()
        except Exception:
            raise RuntimeError("No operation-location and no JSON body from DI")

    t0 = time.time()
    while True:
        r = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": key}, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"DI REST GET failed: {r.status_code} {r.text}")
        j = r.json()
        status = (j.get("status") or "").lower()
        if status in ("succeeded", "failed", "partiallysucceeded", "partiallysucceeded"):
            return j
        if time.time() - t0 > timeout:
            raise RuntimeError("DI REST polling timed out")
        time.sleep(0.7)


# ------------------------- Main service ------------------------- #

class DIService:
    """
    Thin wrapper around Azure DI with SDK-first and REST fallback, returning a
    stable envelope and hierarchical layout. Fallbacks ensure downstream modules
    always receive usable structure.
    """

    def __init__(self, endpoint: str, key: str, timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.timeout = timeout

        if DocumentIntelligenceClient is None or AzureKeyCredential is None:
            self.client = None
        else:
            try:
                self.client = DocumentIntelligenceClient(
                    endpoint=self.endpoint, credential=AzureKeyCredential(key)
                )
            except Exception:
                self.client = None

    def analyze_layout(
        self,
        *,
        content: bytes,
        content_type: str = "application/pdf",
        model_id: str = "prebuilt-layout",
        want_style_font: bool = True,
        request_kv: bool = False,
        features: Optional[List[str] | str] = None,
    ) -> Dict[str, Any]:
        # Normalize features
        req_features: List[str] = []
        if features is not None:
            if isinstance(features, str):
                req_features = [s.strip() for s in features.split(",") if s.strip()]
            else:
                req_features = [str(f).strip() for f in features if str(f).strip()]
        else:
            if want_style_font:
                req_features.append("styleFont")
            if request_kv:
                req_features.append("keyValuePairs")
        # Dedupe
        seen = set(); _tmp=[]
        for f in req_features:
            if f and f not in seen:
                _tmp.append(f); seen.add(f)
        req_features = _tmp

        nego: Dict[str, Any] = {
            "requested_features": req_features[:],
            "used_features": None,
            "content_type": content_type,
            "model_id": model_id,
        }

        # -------- 1) SDK attempt --------
        if self.client is not None:
            try:
                poller = self.client.begin_analyze_document(
                    model_id=model_id,
                    body=content,
                    content_type=content_type,
                    features=req_features or None,
                    timeout=self.timeout,
                )
                result = poller.result()
                layout_like = _normalize_di_v4_layout(_result_to_dict(result))
                nego["used_features"] = req_features[:]
                nego["via"] = "SDK"
                out = {
                    "layout": layout_like,
                    "result": layout_like,
                    "style_fallback": None,
                    "error": None,
                    "_featureNegotiation": nego,
                }

                # Attach KV if present in raw dict
                di_raw = _result_to_dict(result)
                kv_pairs = di_raw.get("analyzeResult", {}).get("keyValuePairs") \
                           or di_raw.get("keyValuePairs") or []
                if request_kv and kv_pairs:
                    out["result"]["keyValuePairs"] = [
                        {
                            "key": (kv.get("key") or {}).get("content") or "",
                            "value": (kv.get("value") or {}).get("content") or "",
                            "confidence": kv.get("confidence"),
                        }
                        for kv in kv_pairs
                    ]

                # If styles requested but absent (common), signal PDF style fallback availability
                if want_style_font and not layout_like.get("styles") and _infer_is_pdf(content_type):
                    out["style_fallback"] = {"source": "pymupdf"}

                return out
            except Exception as e_sdk:
                nego["sdk_error"] = str(e_sdk)

        # -------- 2) REST fallback (fixes 'body' error situations) --------
        try:
            di_obj = _rest_analyze(
                endpoint=self.endpoint, key=self.key, model_id=model_id,
                content=content, content_type=content_type or "application/octet-stream",
                features=req_features, timeout=self.timeout
            )
            layout_like = _normalize_di_v4_layout(di_obj.get("analyzeResult") or di_obj)
            nego["used_features"] = req_features[:]
            nego["via"] = "REST"
            return {
                "layout": layout_like,
                "result": layout_like,
                "style_fallback": None,
                "error": None,
                "_featureNegotiation": nego,
            }
        except Exception as e_rest_primary:
            nego["rest_error_primary"] = str(e_rest_primary)

        # -------- 3) REST conservative retries --------
        try:
            di_obj = _rest_analyze(
                endpoint=self.endpoint, key=self.key, model_id=model_id,
                content=content, content_type="application/octet-stream",
                features=[f for f in req_features if f not in ("styleFont", "ocrFont")],
                timeout=self.timeout
            )
            layout_like = _normalize_di_v4_layout(di_obj.get("analyzeResult") or di_obj)
            nego["used_features"] = [f for f in req_features if f not in ("styleFont", "ocrFont")]
            nego["content_type_downgraded"] = True
            nego["via"] = "REST"
            return {
                "layout": layout_like,
                "result": layout_like,
                "style_fallback": None,
                "error": None,
                "_featureNegotiation": nego,
            }
        except Exception as e_rest_drop:
            nego["rest_error_dropstyle"] = str(e_rest_drop)

        # Switch to prebuilt-document as a last resort
        try:
            di_obj = _rest_analyze(
                endpoint=self.endpoint, key=self.key, model_id="prebuilt-document",
                content=content, content_type="application/octet-stream",
                features=[], timeout=self.timeout
            )
            layout_like = _normalize_di_v4_layout(di_obj.get("analyzeResult") or di_obj)
            nego["used_features"] = None
            nego["model_switched_to"] = "prebuilt-document"
            nego["via"] = "REST"
            return {
                "layout": layout_like,
                "result": layout_like,
                "style_fallback": None,
                "error": None,
                "_featureNegotiation": nego,
            }
        except Exception as e_rest_doc:
            nego["rest_error_doc"] = str(e_rest_doc)

        # -------- 4) Local fallbacks (PDF / image) --------
        if _infer_is_pdf(content_type):
            layout_like = _pdf_hierarchy_from_words(content)
            style_fb = {"source": "pymupdf"}
        else:
            proxies = _image_style_proxies(content)
            layout_like = {
                "content": "",
                "paragraphs": [],
                "lines": [],
                "words": [],
                "style": {"proxies": proxies},
                "notes": ["No DI; emitted image proxies"],
            }
            style_fb = {"source": "image_proxies", **proxies}

        return {
            "layout": layout_like,
            "result": layout_like,
            "style_fallback": style_fb,
            "error": "Azure DI failed after SDK + REST attempts; using local fallback",
            "_featureNegotiation": {**nego, "fallback": "local"},
        }


# ------------------------- Image proxies (used in image fallback) ----------- #

def _image_style_proxies(image_bytes: bytes) -> Dict[str, Any]:
    """OCR-free proxies for ink/weight on images using OpenCV."""
    if Image is None or np is None or cv2 is None:
        return {"error": "PIL/NumPy/OpenCV unavailable for image proxies"}
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        gray = np.array(img)
        binv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(binv, cv2.MORPH_OPEN, k, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)

        h, w = clean.shape[:2]
        total = float(h * w)
        text_px = float((clean > 0).sum())
        ink_ratio = text_px / total if total else 0.0

        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        wh = [(int(s[2]), int(s[3])) for s in stats[1:] if s[4] >= 10 and s[2] < w*0.9 and s[3] < h*0.9]
        avg_w = float(np.mean([p[0] for p in wh])) if wh else 0.0
        avg_h = float(np.mean([p[1] for p in wh])) if wh else 0.0

        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
        est_sw = float(2.0 * np.mean(dist[clean > 0])) if text_px > 0 else 0.0

        mean_text = float(np.mean(gray[clean > 0])) if text_px > 0 else 0.0
        mean_bg = float(np.mean(gray[clean == 0])) if (total - text_px) > 0 else 0.0

        return {
            "image_size": [w, h],
            "mean_text_intensity": mean_text,
            "mean_bg_intensity": mean_bg,
            "ink_ratio": ink_ratio,
            "avg_component_width": avg_w,
            "avg_component_height": avg_h,
            "estimated_stroke_width": est_sw,
        }
    except Exception as ex:
        return {"error": repr(ex)}

# --- END OF FILE: services/di_client.py ---
