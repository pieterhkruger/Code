# --- START OF FILE: services/text_style_panel.py ---
"""
services/text_style_panel.py
===========================
Centralized text-style extraction, fusion, and evaluation.

Purpose
-------
Provide a single place where *multiple services* emit their "opinions" about
text styles (bold/italic/size-ish strength) and a weighted fusion produces a
consolidated view. This module is intentionally defensive:
- Any individual backend can fail; failures are captured and surfaced to the UI.
- You can toggle services on/off and adjust their weights at runtime.
- A JSON artifact can include either just the consolidated view or (optionally)
  the per-service opinions for audit and weight-tuning.

Backends ("services")
---------------------
1) PyMuPDF true-PDF spans  (keeps vector fonts, fast)               -> ENABLE_PYMUPDF
2) Azure "Azure opinion":
   - Prefer Document Intelligence (DI) v4 styles (styleFont)
   - Fallback to Form Recognizer (FR) v3, using appearance/style spans          -> ENABLE_AZURE
3) Tesseract via tesserocr + WordFontAttributes() on rendered pages  -> ENABLE_TESSERACT
   * Optional heuristic CV checks (stroke width & slant) boost/penalize the opinion.
4) Azure Vision READ ROI pixel metrics (fill ratio / stroke width / slant)      -> ENABLE_VISION_PIXEL

Outputs
-------
- items: list of { id, page, text, bbox, services: {...}, consolidated: {...}, truth: {... or null} }
- summary: per-service counts, error notes, and the exact weights used.

Notes
-----
* "bbox" is [x, y, w, h] in pixels of the rasterized page used for Tesseract/Vision.
* When only vector spans are available (PDF) and no raster exists, bbox may be None.
* Consolidation uses a simple weighted vote. A service contributes a signed score
  for bold and italic:  +confidence if predicts True,  -confidence if predicts False.
  These are multiplied by the service weight then summed. A sigmoid squashes to [0,1].
"""

from __future__ import annotations

import io, os, math, json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# Optional deps. We guard imports to keep module import-safe even if extras not installed.
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    import tesserocr
    from PIL import Image
except Exception:  # pragma: no cover
    tesserocr = None
    from PIL import Image

# Project helpers (present in this repo)
try:
    from core.pdf_utils import pdf_to_images
    from config import PDF_TO_IMAGE_DPI
except Exception:
    PDF_TO_IMAGE_DPI = 300
    def pdf_to_images(data: bytes, dpi: int = 300, first_n_pages: Optional[int] = None):
        raise RuntimeError("pdf_to_images helper not available")

# ----------------- Global Toggles & Weights (can be changed centrally) -----------------
ENABLE_PYMUPDF: bool = True
ENABLE_AZURE: bool = True
ENABLE_TESSERACT: bool = True
ENABLE_VISION_PIXEL: bool = True

WEIGHT_PYMUPDF: float = 0.30
WEIGHT_AZURE: float  = 0.40
WEIGHT_TESSERACT: float = 0.20
WEIGHT_VISION_PIXEL: float = 0.10

INCLUDE_BACKEND_OPINIONS: bool = True   # if False, items[].services is omitted to keep JSON compact

# ----------------- Public API -----------------

def build_text_style_panel(
    *,
    step1_raw: Dict[str, Any],
    source_file_bytes: Optional[bytes] = None,
    source_file_ext: Optional[str] = None,
    service_toggles: Optional[Dict[str, bool]] = None,
    weights: Optional[Dict[str, float]] = None,
    include_backend_opinions: Optional[bool] = None,
    max_pages: int = 3,
) -> Dict[str, Any]:
    """
    Entry point called by Step 2.
    - step1_raw: contents of raw_signals.step1.json as dict.
    - source_file_bytes: original file bytes (to render PDF pages for tesseract/vision crops).
    - source_file_ext: file extension, e.g. ".pdf", ".png" (optional but helps)
    - service_toggles: overrides for ENABLE_* flags.
    - weights: overrides for WEIGHT_* dict.
    - include_backend_opinions: overrides INCLUDE_BACKEND_OPINIONS.

    Returns dict with keys: {"items": [...], "summary": {...}}
    """
    # Resolve config
    toggles = {
        "pymupdf": ENABLE_PYMUPDF,
        "azure": ENABLE_AZURE,
        "tesseract": ENABLE_TESSERACT,
        "vision_pixel": ENABLE_VISION_PIXEL,
    }
    if service_toggles:
        toggles.update(service_toggles)

    wts = {
        "pymupdf": WEIGHT_PYMUPDF,
        "azure": WEIGHT_AZURE,
        "tesseract": WEIGHT_TESSERACT,
        "vision_pixel": WEIGHT_VISION_PIXEL,
    }
    if weights:
        wts.update(weights)

    include_ops = INCLUDE_BACKEND_OPINIONS if include_backend_opinions is None else include_backend_opinions

    errors: Dict[str, str] = {}
    pages_images: List[Image.Image] = []
    image_mode = False

    # Prepare raster pages for downstream pixel ops (tesseract + ROI metrics)
    try:
        if source_file_bytes and (source_file_ext or "").lower() == ".pdf":
            pages_images = pdf_bytes_to_pil_pages(source_file_bytes, first_n_pages=max_pages)
        elif source_file_bytes:
            # treat as single image (ensure RGB)
            img = Image.open(io.BytesIO(source_file_bytes)).convert("RGB")
            pages_images = [img]
            image_mode = True
        else:
            # No source bytes: try to infer from Vision raw (not perfect).
            pages_images = []
    except Exception as ex:
        errors["rasterize"] = repr(ex)

    # ---- Collect raw opinions by backend ----
    # 1) PyMuPDF true PDF spans
    pm_spans = []
    if toggles["pymupdf"]:
        try:
            if fitz is None:
                raise RuntimeError("PyMuPDF not installed")
            if source_file_bytes and (source_file_ext or "").lower() == ".pdf":
                pm_spans = pymupdf_spans(source_file_bytes)
            else:
                pm_spans = []
        except Exception as ex:
            errors["pymupdf"] = repr(ex)

    # 2) Azure (DI v4 preferred; FR v3 fallback)
    az_styles = []
    if toggles["azure"]:
        try:
            az_styles = azure_styles_from_backends(step1_raw, source_file_bytes, source_file_ext)
        except Exception as ex:
            errors["azure"] = repr(ex)

    # 3) Tesseract + WFA on raster pages
    tess_words = []
    if toggles["tesseract"]:
        try:
            tess_words = tesseract_words_with_wfa(pages_images)
        except Exception as ex:
            errors["tesseract"] = repr(ex)

    # 4) Vision pixel ROI metrics (using Step 1 Vision READ + our pages_images)
    vision_metrics = []
    per_page_stats = {}
    if toggles["vision_pixel"]:
        try:
            vision_pages = (step1_raw or {}).get("vision", {}).get("pages", [])
            vision_metrics, per_page_stats = vision_roi_metrics(vision_pages, pages_images)
        except Exception as ex:
            errors["vision_pixel"] = repr(ex)

    # ---- Fuse opinions by a simple key: (page, normalized_text) ----
    items = fuse_by_text_key(
        pm_spans=pm_spans,
        az_styles=az_styles,
        tess_words=tess_words,
        vision_metrics=vision_metrics,
        weights=wts,
        include_ops=include_ops,
        per_page_stats=per_page_stats,
    )

    summary = {
        "toggles": toggles,
        "weights": wts,
        "errors": errors,
        "counts": {
            "pymupdf_spans": len(pm_spans),
            "azure_style_spans": len(az_styles),
            "tesseract_words": len(tess_words),
            "vision_word_rois": len(vision_metrics),
            "items": len(items),
        },
        "notes": [
            "Consolidation key is (page, normalized_text). Multiple occurrences of the same word on a page will merge.",
            "Tune weights in this module or via the Step 2 UI sliders.",
        ],
    }
    return {"items": items, "summary": summary}

# ----------------- Helpers: rasterization -----------------

def pdf_bytes_to_pil_pages(pdf_bytes: bytes, first_n_pages: int = 3) -> List[Image.Image]:
    """Render PDF to PIL images at configured DPI for tesseract/stroke metrics."""
    pages = pdf_to_images(pdf_bytes, dpi=PDF_TO_IMAGE_DPI, first_n_pages=first_n_pages)
    # Ensure RGB
    out = []
    for p in pages:
        if p.mode != "RGB":
            out.append(p.convert("RGB"))
        else:
            out.append(p)
    return out

# ----------------- Backend 1: PyMuPDF spans -----------------

def pymupdf_spans(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract spans with font/style info directly from the PDF (true vector text)."""
    if fitz is None:
        return []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = []
    for pno in range(len(doc)):
        page = doc[pno]
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                # compute line median height to assist bold heuristic
                heights = [span.get("size", 0.0) for span in line.get("spans", [])]
                h_med = float(np.median(heights)) if heights else 0.0
                for span in line.get("spans", []):
                    text = (span.get("text") or "").strip()
                    if not text:
                        continue
                    fontname = span.get("font") or ""
                    size = float(span.get("size") or 0.0)
                    flags = int(span.get("flags") or 0)
                    bbox = span.get("bbox")

                    # Heuristics: bold/italic via font name & relative size
                    name_low = fontname.lower()
                    is_bold = any(k in name_low for k in ("bold", "black", "semibold", "demibold", "heavy"))
                    is_italic = any(k in name_low for k in ("italic", "oblique"))
                    # size boost if clearly bigger than neighbors
                    if size > h_med * 1.1:
                        is_bold = True

                    conf = 0.75 if (is_bold or is_italic) else 0.60
                    out.append({
                        "service": "pymupdf",
                        "page": pno,
                        "text": text,
                        "bbox": None,  # vector; not mapped to pixels by default
                        "font": fontname,
                        "size": size,
                        "is_bold": bool(is_bold),
                        "is_italic": bool(is_italic),
                        "confidence": float(conf),
                    })
    doc.close()
    return out

# ----------------- Backend 2: Azure (DI v4 preferred; FR v3 fallback) -----------------
from config import get_config
from services.di_client import DIService

def azure_styles_from_backends(step1_raw: Dict[str, Any], source_file_bytes: Optional[bytes], content_type_hint: Optional[str]) -> List[Dict[str, Any]]:
    """
    Resolve Azure opinion:
    1) If Step 1 DI result already contains styles, parse them.
    2) Else, try DI v4 directly with styleFont.
    3) Else, try FR v3 (best-effort).
    """
    # 1) Parse existing DI raw if present
    di_result = (step1_raw or {}).get("di", {}).get("result") or {}
    styles = azure_styles_from_di_result(di_result)
    if styles:
        return styles

    # 2) Try DI v4 directly
    try:
        cfg = get_config()
        if cfg.azure.di_endpoint and cfg.azure.di_key and source_file_bytes:
            di = DIService(cfg.azure.di_endpoint, cfg.azure.di_key)
            # best content-type guess
            ctype = "application/pdf" if (content_type_hint or "").lower() == ".pdf" else "application/octet-stream"
            di_out = di.analyze_layout(content=source_file_bytes, content_type=ctype, model_id="prebuilt-layout", want_style_font=True, request_kv=False)
            styles = azure_styles_from_di_result(di_out.get("result") or {})
            if styles:
                return styles
    except Exception:
        pass

    # 3) Try FR v3 (very light-weight; styles are more limited)
    try:
        from azure.ai.formrecognizer import DocumentAnalysisClient
        from azure.core.credentials import AzureKeyCredential
        cfg = get_config()
        if cfg.azure.di_endpoint and cfg.azure.di_key and source_file_bytes:
            fr = DocumentAnalysisClient(endpoint=cfg.azure.di_endpoint, credential=AzureKeyCredential(cfg.azure.di_key))
            poller = fr.begin_analyze_document(model_id="prebuilt-layout", document=source_file_bytes)
            res = poller.result()
            # Normalize to dict
            try:
                resd = res.to_dict()
            except Exception:
                resd = json.loads(res.to_json()) if hasattr(res, "to_json") else {}
            # FR v3 exposes 'styles' (handwriting) and 'paragraphs'[...]['role'].
            # It may not expose font weight/style, so we convert what we can.
            out = []
            content = resd.get("content") or ""
            for st in (resd.get("styles") or []):
                spans = st.get("spans") or []
                conf = float(st.get("confidence", 0.7))
                snippet = ""
                if content and spans:
                    s0 = spans[0]; off = int(s0.get("offset", 0)); length = int(s0.get("length", 0))
                    try: snippet = content[off: off + length]
                    except Exception: snippet = ""
                # FR v3 styles don't have bold/italic; leave as None
                out.append({
                    "service": "azure",
                    "page": None,
                    "text": (snippet or "").strip(),
                    "bbox": None,
                    "fontStyle": None,
                    "fontWeight": None,
                    "isHandwritten": st.get("is_handwritten") or st.get("isHandwritten"),
                    "is_bold": None,
                    "is_italic": None,
                    "confidence": conf,
                })
            return out
    except Exception:
        pass

    return []

def azure_styles_from_di_result(di_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract style spans from DI/FR result dict (supports both shapes).
    Expected keys:
      di_result["styles"]             : list of styles with spans[]
      di_result["content"]            : full text
      style fields: fontStyle/font_style, fontWeight/font_weight, isHandwritten, confidence
    """
    if not di_result:
        return []
    content = di_result.get("content") or ""
    styles = list(di_result.get("styles") or [])
    out = []
    for idx, style in enumerate(styles):
        spans = style.get("spans") or []
        f_style = style.get("fontStyle") or style.get("font_style")
        f_weight = style.get("fontWeight") or style.get("font_weight")
        is_hw = style.get("isHandwritten", None)
        conf = float(style.get("confidence", 0.7))

        # Make one row per span slice (use first span as representative text)
        snippet = ""
        if content and spans:
            s0 = spans[0]
            off = int(s0.get("offset", 0))
            length = int(s0.get("length", 0))
            try:
                snippet = content[off: off + length]
            except Exception:
                snippet = ""

        out.append({
            "service": "azure",
            "page": None,                # DI spans are in content offsets; page mapping not trivial here
            "text": (snippet or "").strip(),
            "bbox": None,
            "fontStyle": f_style,
            "fontWeight": f_weight,
            "isHandwritten": is_hw,
            "is_bold": str(f_weight).lower() == "bold",
            "is_italic": str(f_style).lower() == "italic",
            "confidence": conf,
        })
    return out

# ----------------- Backend 3: Tesseract + WordFontAttributes -----------------

def tesseract_words_with_wfa(pages: List[Image.Image]) -> List[Dict[str, Any]]:
    """Run Tesseract on provided PIL pages and gather word-level font attrs."""
    if tesserocr is None:
        return []
    out: List[Dict[str, Any]] = []
    for pno, img in enumerate(pages):
        try:
            with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.AUTO) as api:
                api.SetImage(img)
                ri = api.GetIterator()
                level = tesserocr.RIL.WORD
                if ri:
                    while True:
                        text = (ri.GetUTF8Text(level) or "").strip()
                        conf = float(ri.Confidence(level) or 0.0) / 100.0  # 0..1
                        bbox = ri.BoundingBox(level)
                        try:
                            # returns tuple: (bold, italic, underlined, monospace, serif, smallcaps, pointsize, font_id, font_name)
                            attrs = ri.WordFontAttributes()
                            is_bold = bool(attrs[0])
                            is_italic = bool(attrs[1])
                            font_name = attrs[8] if len(attrs) >= 9 else None
                            pointsize = float(attrs[6]) if len(attrs) >= 7 and attrs[6] is not None else None
                        except Exception:
                            is_bold = is_italic = False
                            font_name = None
                            pointsize = None

                        if text:
                            out.append({
                                "service": "tesseract",
                                "page": pno,
                                "text": text,
                                "bbox": list(bbox) if bbox else None,  # [x0,y0,x1,y1] (image pixels)
                                "font": font_name,
                                "size": pointsize,
                                "is_bold": is_bold,
                                "is_italic": is_italic,
                                "confidence": conf,
                            })
                        if not ri.Next(level):
                            break
        except Exception as ex:
            # collect page-level error and continue
            out.append({"service": "tesseract", "page": pno, "error": repr(ex)})
    return out

# ----------------- Backend 4: Vision ROI pixel metrics -----------------

def vision_roi_metrics(vision_pages: List[Dict[str, Any]], pages: List[Image.Image]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, float]]]:
    """
    Use Vision READ line/word bboxes to crop ROIs and compute pixel metrics:
    - fill_ratio (foreground px / area)
    - stroke_width (2 * mean distance to background)
    - slant_deg (Hough-based average shear angle)
    Returns (metrics_list, per_page_stats)
    """
    metrics: List[Dict[str, Any]] = []
    per_page: Dict[int, Dict[str, float]] = {}
    for vp in vision_pages:
        pno = int(vp.get("page_index", 0))
        vdict = vp.get("result") or {}
        read = (vdict.get("read") or {}).get("lines") or []
        if pno >= len(pages):
            continue
        img = pages[pno]
        w, h = img.size
        # collect ratios to compute median per page
        ratios = []
        for line in read:
            text = (line.get("text") or "").strip()
            bb = line.get("bbox")  # [x,y,w,h] in image pixels (we normalized in Vision client)
            if not text or not bb:
                continue
            # crop and compute metrics
            x, y, ww, hh = map(int, bb)
            x0 = max(0, x); y0 = max(0, y); x1 = min(w, x + ww); y1 = min(h, y + hh)
            if x1 <= x0 or y1 <= y0: 
                continue
            crop = np.array(img.crop((x0, y0, x1, y1)).convert("L"))

            m = roi_metrics_from_gray(crop)
            ratios.append(m["fill_ratio"])

            metrics.append({
                "service": "vision_pixel",
                "page": pno,
                "text": text,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                **m,
            })
        # page-level stats
        if ratios:
            ratios_arr = np.array(ratios, dtype=np.float32)
            median = float(np.median(ratios_arr))
            mad = float(np.median(np.abs(ratios_arr - median)) + 1e-6)
            per_page[pno] = {"fill_median": median, "fill_mad": mad}
    # Normalize bold_hint by z-score > 2.0
    for m in metrics:
        pno = m["page"]
        pg = per_page.get(pno, {"fill_median": 0.0, "fill_mad": 1.0})
        z = (m["fill_ratio"] - pg["fill_median"]) / (pg["fill_mad"] * 1.4826 + 1e-6)
        m["fill_z"] = float(z)
        m["bold_hint"] = bool(z > 2.0)  # threshold can be tuned
        m["italic_hint"] = bool(abs(m.get("slant_deg", 0.0)) > 8.0)
        # confidence proxy from how extreme the z-score is (cap 1.0)
        m["confidence"] = float(max(0.0, min(abs(z) / 4.0, 1.0)))
    return metrics, per_page

def roi_metrics_from_gray(gray: np.ndarray) -> Dict[str, float]:
    """Compute fill ratio, stroke width, slant angle on a grayscale ROI."""
    # Binarize with Otsu + cleanup
    thr, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    h, w = clean.shape[:2]
    area = float(h * w)
    ink = float(np.count_nonzero(clean > 0))
    fill_ratio = (ink / area) if area else 0.0

    # Stroke width via distance transform
    dist = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
    stroke_width = float(2.0 * np.mean(dist[clean > 0])) if ink > 0 else 0.0

    # Slant via Hough on edges of skeleton-ish image
    edges = cv2.Canny(clean, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180.0, threshold=30, minLineLength=max(10, w//5), maxLineGap=5)
    angles = []
    if lines is not None:
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = map(int, l)
            dx = float(x2 - x1); dy = float(y2 - y1)
            if abs(dx) < 1e-3 or abs(dy) < 1e-3:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            # Only consider near-vertical-ish strokes for italic estimate
            if 50 <= abs(angle) <= 80:
                # Positive angle means slanting right; negative left
                angles.append(90 - abs(angle))
    slant_deg = float(np.median(angles)) if angles else 0.0

    return {"fill_ratio": float(fill_ratio), "stroke_width": float(stroke_width), "slant_deg": slant_deg}

# ----------------- Fusion -----------------

def fuse_by_text_key(
    *,
    pm_spans: List[Dict[str, Any]],
    az_styles: List[Dict[str, Any]],
    tess_words: List[Dict[str, Any]],
    vision_metrics: List[Dict[str, Any]],
    weights: Dict[str, float],
    include_ops: bool,
    per_page_stats: Dict[int, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """
    Merge backend opinions by (page, normalized_text).
    For backends that lack page info (Azure DI), we use page=None key and still fold them in by text only.
    """
    def key_for(page, text):
        t = (text or "").strip()
        t = " ".join(t.split())  # collapse whitespace
        t = t[:64].lower()
        return (page, t)

    buckets: Dict[Tuple[Optional[int], str], Dict[str, Any]] = {}

    def add(service_name: str, page: Optional[int], text: str, payload: Dict[str, Any]):
        k = key_for(page, text)
        b = buckets.setdefault(k, {"page": page, "text": text, "services": {}})
        b["services"][service_name] = payload

    for row in pm_spans:
        add("pymupdf", row.get("page"), row.get("text"), row)

    for row in az_styles:
        add("azure", row.get("page"), row.get("text"), row)

    for row in tess_words:
        add("tesseract", row.get("page"), row.get("text"), row)

    for row in vision_metrics:
        add("vision_pixel", row.get("page"), row.get("text"), row)

    # Consolidate
    items: List[Dict[str, Any]] = []
    for (page, key_text), b in buckets.items():
        services = b["services"]

        def pick(service: str, field: str, default=None):
            return (services.get(service) or {}).get(field, default)

        # For bbox we prefer tesseract (has pixel coords), else vision
        bbox = pick("tesseract", "bbox") or pick("vision_pixel", "bbox")

        # Turn each service into a signed vote for bold/italic
        def sv(service: str, pred: Optional[bool], conf: float, weight: float):
            if pred is None:
                return 0.0
            return (1.0 if pred else -1.0) * float(conf) * float(weight)

        # Service-level predictions
        bold_votes = []
        italic_votes = []

        # PyMuPDF
        if "pymupdf" in services:
            pb = bool(pick("pymupdf", "is_bold", False))
            pi = bool(pick("pymupdf", "is_italic", False))
            pc = float(pick("pymupdf", "confidence", 0.6))
            bold_votes.append(sv("pymupdf", pb, pc, weights["pymupdf"]))
            italic_votes.append(sv("pymupdf", pi, pc, weights["pymupdf"]))

        # Azure
        if "azure" in services:
            ab = (str(pick("azure", "is_bold", "false")).lower() == "true")
            ai = (str(pick("azure", "is_italic", "false")).lower() == "true")
            ac = float(pick("azure", "confidence", 0.7))
            bold_votes.append(sv("azure", ab, ac, weights["azure"]))
            italic_votes.append(sv("azure", ai, ac, weights["azure"]))

        # Tesseract
        if "tesseract" in services:
            tb = bool(pick("tesseract", "is_bold", False))
            ti = bool(pick("tesseract", "is_italic", False))
            tc = float(pick("tesseract", "confidence", 0.5))
            # Adjust with pixel hints if present
            if "vision_pixel" in services:
                # If both agree it's bold, boost; if they disagree, penalize
                vb = bool(pick("vision_pixel", "bold_hint", False))
                vi = bool(pick("vision_pixel", "italic_hint", False))
                if vb == tb:
                    tc = min(1.0, tc + 0.2)
                else:
                    tc = max(0.0, tc - 0.2)
                if vi != ti:
                    tc = max(0.0, tc - 0.1)
            bold_votes.append(sv("tesseract", tb, tc, weights["tesseract"]))
            italic_votes.append(sv("tesseract", ti, tc, weights["tesseract"]))

        # Vision pixel
        if "vision_pixel" in services:
            vb = bool(pick("vision_pixel", "bold_hint", False))
            vi = bool(pick("vision_pixel", "italic_hint", False))
            vc = float(pick("vision_pixel", "confidence", 0.4))
            bold_votes.append(sv("vision_pixel", vb, vc, weights["vision_pixel"]))
            italic_votes.append(sv("vision_pixel", vi, vc, weights["vision_pixel"]))

        # Aggregate with sigmoid to 0..1
        bold_score = float(sigmoid(sum(bold_votes)))
        italic_score = float(sigmoid(sum(italic_votes)))
        consolidated = {
            "bold": bool(bold_score >= 0.5),
            "italic": bool(italic_score >= 0.5),
            "bold_score": bold_score,
            "italic_score": italic_score,
        }

        item = {
            "id": f"pg{page if page is not None else 'NA'}-{abs(hash(key_text))%10_000_000}",
            "page": page,
            "text": key_text,
            "bbox": bbox,
            "consolidated": consolidated,
            "truth": {"bold": None, "italic": None},  # to be filled manually by adjudicator
        }
        if include_ops:
            item["services"] = services
        items.append(item)
    return items

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ----------------- Evaluation -----------------

def evaluate_against_truth(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given adjudicated items (with truth.bold/italic set to True/False), compute
    per-service and consolidated accuracy.
    Returns dict with per-service {bold_acc, italic_acc, overall_acc}.
    """
    services = ["pymupdf", "azure", "tesseract", "vision_pixel", "consolidated"]
    correct = {s: {"bold": 0, "italic": 0, "total": 0} for s in services}
    total_items = 0

    for it in items:
        tr = it.get("truth") or {}
        tb = tr.get("bold")
        ti = tr.get("italic")
        if tb is None and ti is None:
            continue  # skip not adjudicated
        total_items += 1

        # Helper to score one service
        def score_svc(sname: str, pb: Optional[bool], pi: Optional[bool]):
            if tb is not None and pb is not None:
                correct[sname]["bold"] += int(bool(tb) == bool(pb))
                correct[sname]["total"] += 1
            if ti is not None and pi is not None:
                correct[sname]["italic"] += int(bool(ti) == bool(pi))
                correct[sname]["total"] += 1

        svcs = (it.get("services") or {})
        score_svc("pymupdf", (svcs.get("pymupdf") or {}).get("is_bold"), (svcs.get("pymupdf") or {}).get("is_italic"))
        score_svc("azure", (svcs.get("azure") or {}).get("is_bold"), (svcs.get("azure") or {}).get("is_italic"))
        score_svc("tesseract", (svcs.get("tesseract") or {}).get("is_bold"), (svcs.get("tesseract") or {}).get("is_italic"))
        score_svc("vision_pixel", (svcs.get("vision_pixel") or {}).get("bold_hint"), (svcs.get("vision_pixel") or {}).get("italic_hint"))
        cons = it.get("consolidated") or {}
        score_svc("consolidated", cons.get("bold"), cons.get("italic"))

    def acc(s):
        d = correct[s]
        if d["total"] == 0:
            return {"bold_acc": None, "italic_acc": None, "overall_acc": None, "samples": 0}
        # overall = (bold + italic) / total comparisons
        return {
            "bold_acc": d["bold"] / max(1, (d["total"]//2)),
            "italic_acc": d["italic"] / max(1, (d["total"]//2)),
            "overall_acc": (d["bold"] + d["italic"]) / d["total"],
            "samples": d["total"],
        }

    return {s: acc(s) for s in services}
# --- END OF FILE ---
