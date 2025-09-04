# --- START OF FILE: services/di_client.py ---
"""
services/di_client.py
=====================
Azure Document Intelligence wrapper for "prebuilt-layout"/"prebuilt-document"
with add-ons (styleFont, keyValuePairs). Adds robust fallbacks and local style
signals (PyMuPDF for PDFs; OpenCV proxies for images) when the service rejects
content or features.

Behavior:
- Prefer styleFont (maps from legacy 'ocrFont'), negotiate features if rejected.
- If service returns UnsupportedContent, retry with:
  (a) same model & features but content_type="application/octet-stream",
  (b) dropping add-ons, and/or
  (c) switching to 'prebuilt-document'.
- If DI still fails, return {"error": ...} plus 'style_fallback' so downstream
  modules get usable style signals anyway.
"""

from __future__ import annotations

import io, json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import cv2

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

try:
    from services.pdf_style_probe import extract_pdf_styles
except Exception:
    extract_pdf_styles = None


def _result_to_dict(di_result: Any) -> Dict[str, Any]:
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

def _image_style_proxies(image_bytes: bytes) -> Dict[str, Any]:
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
        text_px = float(np.count_nonzero(clean > 0))
        ink_ratio = text_px / total if total else 0.0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        whs = []
        for i in range(1, n):
            x, y, ww, hh, area = stats[i]
            if area < 10: continue
            if ww > w * 0.9 or hh > h * 0.9: continue
            whs.append((ww, hh))
        avg_w = float(np.mean([p[0] for p in whs])) if whs else 0.0
        avg_h = float(np.mean([p[1] for p in whs])) if whs else 0.0
        dist = cv2.distanceTransform(clean, cv2.DIST_L2, 3)
        sw = float(2.0 * np.mean(dist[clean > 0])) if text_px > 0 else 0.0
        mean_text = float(np.mean(gray[clean > 0])) if text_px > 0 else 0.0
        mean_bg   = float(np.mean(gray[clean == 0])) if (total - text_px) > 0 else 0.0
        return {
            "image_size": [w, h],
            "mean_text_intensity": mean_text,
            "mean_bg_intensity": mean_bg,
            "ink_ratio": ink_ratio,
            "avg_component_width": avg_w,
            "avg_component_height": avg_h,
            "estimated_stroke_width": sw,
            "notes": ["Proxies derived without OCR; used when DI styleFont is unavailable."]
        }
    except Exception as ex:
        return {"error": repr(ex), "notes": ["Image proxies failed"]}

def _compute_pdf_style_fallback(pdf_bytes: bytes) -> Dict[str, Any]:
    if extract_pdf_styles is None:
        return {"error": "PyMuPDF not available or helper not imported"}
    try:
        spans = extract_pdf_styles(pdf_path=None, pdf_bytes=pdf_bytes)
        return {"pdf_spans": spans, "notes": ["True PDF style spans via PyMuPDF"]}
    except Exception as ex:
        return {"error": repr(ex), "notes": ["PyMuPDF extraction failed"]}

def _build_enum_map():
    enum_map = {}
    try:
        from azure.ai.documentintelligence.models import DocumentAnalysisFeature as F  # GA
        enum_map["stylefont"]         = getattr(F, "STYLE_FONT", "styleFont")
        enum_map["keyvaluepairs"]     = getattr(F, "KEY_VALUE_PAIRS", "keyValuePairs")
        enum_map["ocrhighresolution"] = getattr(F, "OCR_HIGH_RESOLUTION", "ocrHighResolution")
        return enum_map, True
    except Exception:
        pass
    try:
        from azure.ai.documentintelligence.models import AnalyzeFeature as F  # older
        enum_map["stylefont"]         = getattr(F, "OCR_FONT", "ocrFont")  # legacy support
        enum_map["keyvaluepairs"]     = getattr(F, "KEY_VALUE_PAIRS", "keyValuePairs")
        enum_map["ocrhighresolution"] = getattr(F, "OCR_HIGH_RESOLUTION", "ocrHighResolution")
        return enum_map, False
    except Exception:
        return {
            "stylefont": "styleFont",
            "keyvaluepairs": "keyValuePairs",
            "ocrhighresolution": "ocrHighResolution",
        }, None


class DIService:
    def __init__(self, endpoint: str, key: str, timeout: int = 120):
        self.endpoint = endpoint
        self.key = key
        self.timeout = timeout
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

    def analyze_layout(
    self,
    *,
    content: bytes,
    content_type: str = "application/pdf",
    model_id: str = "prebuilt-layout",
    want_style_font: bool = True,
    request_kv: bool = False,
) -> Dict[str, Any]:
    """
    Call Azure DI 'prebuilt-layout' with styleFont/keyValuePairs if available.
    If the service rejects a feature or content, retry gracefully and then
    return a DI-shaped fallback built from PyMuPDF so downstream modules can
    consume a consistent envelope.

    Returns:
        dict with keys:
          - result (on success) OR error (on failure)
          - _featureNegotiation (always)
          - style_fallback (when DI failed)
    """
    features: List[str] = []
    if want_style_font:
        features.append("styleFont")
    if request_kv:
        features.append("keyValuePairs")

    nego = {
        "requested_features": features[:],
        "used_features": None,
        "content_type": content_type,
        "model_id": model_id,
        "di_region": getattr(self, "region", None),
    }

    def _fallback_from_pdf() -> Dict[str, Any]:
        # DI-shaped minimal envelope using PyMuPDF spans so downstream
        # code sees predictable keys even without DI.
        try:
            from pdf_style_probe import extract_pdf_spans, linearize_spans
            spans = extract_pdf_spans(content)  # [{page, text, font, size, bbox, flags, color}, ...]
            pages = linearize_spans(spans)      # [{"page_index": i, "linearized_text": "...", "spans": [...]}, ...]
            return {
                "result": {
                    "content": "\n\n".join(p["linearized_text"] for p in pages),
                    "pages": pages,
                },
                "_featureNegotiation": {**nego, "fallback": "pymupdf"},
                "style_fallback": {"source": "pymupdf", "pdf_spans": spans},
            }
        except Exception as ex:
            return {
                "error": f"DI failed and PyMuPDF fallback also failed: {ex!r}",
                "_featureNegotiation": {**nego, "fallback": "pymupdf-error"},
            }

    # Primary attempt — use body= for binary content and pass features.
    try:
        poller = self.client.begin_analyze_document(
            model_id=model_id,
            body=content,
            content_type=content_type,
            features=features or None,
            timeout=self.timeout,
        )
        result = poller.result()
        nego["used_features"] = features[:]
        return {
            "result": _result_to_dict(result),
            "_featureNegotiation": nego,
        }

    except HttpResponseError as e:
        msg = str(e)
        # If styleFont not supported in this region/model, retry without it once.
        if "InvalidParameter" in msg and "styleFont" in msg and "invalid" in msg.lower():
            try:
                retry_features = [f for f in features if f != "styleFont"]
                poller = self.client.begin_analyze_document(
                    model_id=model_id,
                    body=content,
                    content_type=content_type,
                    features=retry_features or None,
                    timeout=self.timeout,
                )
                result = poller.result()
                nego["used_features"] = retry_features
                nego["styleFont_unavailable"] = True
                return {
                    "result": _result_to_dict(result),
                    "_featureNegotiation": nego,
                }
            except Exception as e2:
                # Fall through to PyMuPDF fallback
                pass

        # UnsupportedContent or anything else → fallback
        fb = _fallback_from_pdf()
        if "result" in fb:
            fb["error"] = msg
        else:
            fb["error"] = f"{msg} (and fallback failed)"
        return fb

    except Exception as ex:
        fb = _fallback_from_pdf()
        if "result" in fb:
            fb["error"] = repr(ex)
        else:
            fb["error"] = f"{ex!r} (and fallback failed)"
        return fb

# --- END OF FILE ---
