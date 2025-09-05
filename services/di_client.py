"""
services/di_client.py
=====================

Azure Document Intelligence (DI) wrapper with:
- Stable return envelope for downstream modules
- SDK-first, REST fallback (fixes 'body' kwarg error)
- Normalization to a *hierarchical* layout:
    content → paragraphs → lines → words  (+ flat lists)
- Robust local fallbacks when DI is unavailable:
    • PDFs → build the same hierarchy via PyMuPDF
    • Images → emit OCR-free pixel proxies (ink/weight metrics)

Returned payload (top-level from analyze_layout):
{
    "layout": {...},                 # preferred key
    "result": {...},                 # alias to the same object
    "style_fallback": {...}|None,    # PDF spans / proxies if DI failed
    "error": "..."|None,             # error message if DI failed
    "_featureNegotiation": {...}     # attempt trace
}
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple

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

# Azure SDK (we will *try* it first, then REST if it fails)
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
            # flat list of floats
            it = list(poly)
            if len(it) % 2 != 0:
                it = it[:-1]
            for i in range(0, len(it), 2):
                xs.append(float(it[i]))
                ys.append(float(it[i + 1]))
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]

def _image_style_proxies(image_bytes: bytes) -> Dict[str, Any]:
    """OCR-free proxies for ink/weight/slant on images using OpenCV."""
    if Image is None or np is None or cv2 is None:
        return {"error": "PIL/NumPy/OpenCV unavailable for image proxies"}

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        gray = np.array(img)

        binv = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(binv, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        h, w = clean.shape[:2]
        total = float(h * w)
        text_px = float((clean > 0).sum())
        ink_ratio = text_px / total if total else 0.0

        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        whs: List[Tuple[int, int]] = []
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < 10:
                continue
            if ww > w * 0.9 or hh > h * 0.9:
                continue
            whs.append((ww, hh))
        avg_w = float(np.mean([p[0] for p in whs])) if whs else 0.0
        avg_h = float(np.mean([p[1] for p in whs])) if whs else 0.0

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


# ------------------------- PDF fallback hierarchy ------------------------- #

def _pdf_hierarchy_from_words(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Use PyMuPDF 'words' to build:
      paragraphs (≈ blocks) → lines → words with [x,y,w,h]
    """
    if fitz is None:
        return {"content": "", "paragraphs": [], "lines": [], "words": [], "notes": ["PyMuPDF not available"]}

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_words = []
    paragraphs: List[Dict[str, Any]] = []
    flat_lines: List[Dict[str, Any]] = []
    flat_words: List[Dict[str, Any]] = []
    content_parts: List[str] = []

    try:
        for pno in range(len(doc)):
            page = doc[pno]
            # words: [x0,y0,x1,y1,"text", block_no, line_no, word_no]
            words = page.get_text("words") or []
            # Group by (block_no, line_no)
            by_line: Dict[Tuple[int, int], List[Tuple[float,float,float,float,str,int,int,int]]] = {}
            for w in words:
                x0, y0, x1, y1, text, bno, lno, wno = w
                by_line.setdefault((int(bno), int(lno)), []).append(w)
                flat_words.append({
                    "page": pno,
                    "text": text,
                    "bbox": [float(x0), float(y0), float(x1-x0), float(y1-y0)],
                })
                all_words.append(text)

            # Group lines by block_no → paragraphs
            by_block: Dict[int, List[Tuple[int, List[Tuple[float,float,float,float,str,int,int,int]]]]] = {}
            for (bno, lno), arr in by_line.items():
                by_block.setdefault(int(bno), []).append((int(lno), arr))

            for bno, lines in sorted(by_block.items(), key=lambda t: t[0]):
                para_lines = []
                px0=py0=1e9; px1=py1=-1e9
                para_text_parts = []
                for _, arr in sorted(lines, key=lambda t: t[0]):
                    # line bbox
                    lx0 = min(a[0] for a in arr); ly0 = min(a[1] for a in arr)
                    lx1 = max(a[2] for a in arr); ly1 = max(a[3] for a in arr)
                    lb = [float(lx0), float(ly0), float(lx1-lx0), float(ly1-ly0)]
                    line_text = " ".join(a[4] for a in sorted(arr, key=lambda x: x[7]))
                    words_json = [
                        {"text": a[4], "bbox": [float(a[0]), float(a[1]), float(a[2]-a[0]), float(a[3]-a[1])]}
                        for a in sorted(arr, key=lambda x: x[7])
                    ]
                    para_lines.append({"page": pno, "text": line_text, "bbox": lb, "words": words_json})
                    flat_lines.append({"page": pno, "text": line_text, "bbox": lb})
                    para_text_parts.append(line_text)

                    px0 = min(px0, lx0); py0 = min(py0, ly0); px1 = max(px1, lx1); py1 = max(py1, ly1)

                if para_lines:
                    paragraphs.append({
                        "page": pno,
                        "text": " ".join(para_text_parts),
                        "bbox": [float(px0), float(py0), float(px1-px0), float(py1-py0)],
                        "lines": para_lines,
                    })
                    content_parts.append(" ".join(para_text_parts))
        doc.close()
    except Exception:
        doc.close()
        return {"content": "", "paragraphs": [], "lines": [], "words": [], "notes": ["PyMuPDF words extraction failed"]}

    return {
        "content": " ".join(content_parts),
        "paragraphs": paragraphs,
        "lines": flat_lines,
        "words": flat_words,
    }


# ------------------------- DI v4 REST normalizer ------------------------- #

def _normalize_di_v4_layout(di_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize DI v4 REST (or SDK) JSON to our hierarchical layout.
    Works if di_json is either the raw 'analyzeResult' or a wrapper containing it.
    """
    # Locate analyzeResult
    ar = di_json.get("analyzeResult") or di_json.get("result") or di_json
    content = ar.get("content", "")

    # Flat lines/words from pages
    flat_lines: List[Dict[str, Any]] = []
    flat_words: List[Dict[str, Any]] = []
    pages = ar.get("pages") or []
    for p in pages:
        pno = int(p.get("pageNumber", 1)) - 1  # zero-based
        for ln in p.get("lines") or []:
            text = ln.get("content") or ""
            bbox = _poly_to_bbox(ln.get("polygon"))
            flat_lines.append({"page": pno, "text": text, "bbox": bbox})
        for wd in p.get("words") or []:
            text = wd.get("content") or ""
            bbox = _poly_to_bbox(wd.get("polygon"))
            flat_words.append({"page": pno, "text": text, "bbox": bbox})

    # Paragraphs; try to nest lines/words by bbox inclusion (center point)
    def _center(bb: Optional[List[float]]) -> Optional[Tuple[float, float]]:
        if not bb:
            return None
        x, y, w, h = bb
        return (x + w / 2.0, y + h / 2.0)

    paragraphs: List[Dict[str, Any]] = []
    for para in ar.get("paragraphs") or []:
        p_text = para.get("content") or ""
        breg = (para.get("boundingRegions") or [{}])[0]
        page_index = int(breg.get("pageNumber", 1)) - 1
        bbox = _poly_to_bbox(breg.get("polygon"))

        # lines within this paragraph (simple center-in-rect heuristic)
        p_lines = []
        if bbox:
            x, y, w, h = bbox
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            for ln in flat_lines:
                if ln["page"] != page_index or not ln["bbox"]:
                    continue
                cx, cy = _center(ln["bbox"])
                if cx is None:
                    continue
                if xmin <= cx <= xmax and ymin <= cy <= ymax:
                    # Attach words that lie within the line bbox
                    lw_x, lw_y, lw_w, lw_h = ln["bbox"]
                    l_xmin, l_ymin, l_xmax, l_ymax = lw_x, lw_y, lw_x + lw_w, lw_y + lw_h
                    words = []
                    for wd in flat_words:
                        if wd["page"] != page_index or not wd["bbox"]:
                            continue
                        wx, wy, ww, wh = wd["bbox"]
                        wcx, wcy = wx + ww / 2.0, wy + wh / 2.0
                        if l_xmin <= wcx <= l_xmax and l_ymin <= wcy <= l_ymax:
                            words.append({"text": wd["text"], "bbox": wd["bbox"]})
                    p_lines.append({"page": page_index, "text": ln["text"], "bbox": ln["bbox"], "words": words})

        paragraphs.append({
            "page": page_index,
            "text": p_text,
            "bbox": bbox,
            "lines": p_lines,
        })

    layout = {
        "content": content,
        "paragraphs": paragraphs,
        "lines": flat_lines,
        "words": flat_words,
        # expose whatever style info exists (SDK/REST variants differ; keep raw)
        "style": ar.get("styles") or {},
    }
    return layout


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

    # --- REST: POST bytes and poll operation ---
    def _rest_analyze(self, *, model_id: str, content: bytes, content_type: str, features: List[str]) -> Dict[str, Any]:
        if requests is None:
            raise RuntimeError("requests not available for REST fallback")

        # Use the 2024-02-29-preview (widely supported) analyze endpoint
        url = f"{self.endpoint}/documentintelligence/documentModels/{model_id}:analyze"
        params = {"api-version": "2024-02-29-preview"}
        # Some features are controlled via 'features' in body in newer APIs; in preview versions,
        # they are queried via 'features' too. To keep it robust, send them as a comma CSV header.
        headers = {
            "Ocp-Apim-Subscription-Key": self.key,
            "Accept": "application/json",
            "Content-Type": content_type or "application/octet-stream",
        }
        if "styleFont" in features or "ocrFont" in features:
            # server-side will treat as style extraction if supported
            headers["x-di-features"] = ",".join(features)

        resp = requests.post(url, params=params, headers=headers, data=content, timeout=self.timeout)
        if resp.status_code not in (200, 202):
            raise RuntimeError(f"DI REST POST failed: {resp.status_code} {resp.text}")

        # Operation-location header to poll
        op_url = resp.headers.get("operation-location")
        if not op_url:
            # Some regions return the result inline (200)
            try:
                obj = resp.json()
                return obj
            except Exception:
                raise RuntimeError("No operation-location and no JSON body from DI")

        # Poll until done (bounded by timeout)
        t0 = time.time()
        while True:
            r = requests.get(op_url, headers={"Ocp-Apim-Subscription-Key": self.key}, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"DI REST GET failed: {r.status_code} {r.text}")
            j = r.json()
            status = (j.get("status") or "").lower()
            if status in ("succeeded", "failed", "partiallysucceeded", "partiallysucceeded"):
                return j
            if time.time() - t0 > self.timeout:
                raise RuntimeError("DI REST polling timed out")
            time.sleep(0.7)

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
                return {
                    "layout": layout_like,
                    "result": layout_like,
                    "style_fallback": None,
                    "error": None,
                    "_featureNegotiation": nego,
                }
            except Exception as e_sdk:
                # If the environment triggers the 'body' kwarg error (requests), switch to REST.
                nego["sdk_error"] = str(e_sdk)

        # -------- 2) REST fallback (fixes 'body' error) --------
        try:
            di_obj = self._rest_analyze(
                model_id=model_id,
                content=content,
                content_type=content_type or "application/octet-stream",
                features=req_features,
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
        # Drop style features and use octet-stream as a safe default
        try:
            di_obj = self._rest_analyze(
                model_id=model_id,
                content=content,
                content_type="application/octet-stream",
                features=[f for f in req_features if f not in ("styleFont", "ocrFont")],
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
            di_obj = self._rest_analyze(
                model_id="prebuilt-document",
                content=content,
                content_type="application/octet-stream",
                features=[],
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
