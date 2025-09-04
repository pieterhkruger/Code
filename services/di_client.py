"""
services/di_client.py
=====================
Azure Document Intelligence wrapper focused on the "prebuilt-layout" (or "prebuilt-document")
model with add-ons (styleFont, keyValuePairs). Returns the raw service response as a dict and
appends a local style fallback (PyMuPDF for PDFs; OpenCV proxies for images) when needed.

Notes:
- We keep vector text by sending original PDF bytes to DI.
- Compatible with azure-ai-documentintelligence 1.0.x+:
  * Tries newer enum names first (DocumentAnalysisFeature.STYLE_FONT), then gracefully
    falls back to older AnalyzeFeature enums if present.
- If 'styleFont' is rejected by the region/model, we retry without it and attach a
  'style_fallback' section with locally computed style signals.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
import numpy as np
import cv2  # OpenCV headless is in your requirements

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# Optional: use your existing PDF style helper if present in the repo
try:
    from services.pdf_style_probe import extract_pdf_styles
except Exception:
    extract_pdf_styles = None  # we will only call it if import worked


# ------------------------- helpers -------------------------

def _result_to_dict(di_result: Any) -> Dict[str, Any]:
    """Convert SDK result to a dict across minor SDK variations."""
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
    """
    Compute light-weight proxies for text style on scans/photos using OpenCV/Pillow:
      - mean_text_intensity / mean_bg_intensity (grayscale)
      - ink_ratio (text pixels / total)
      - avg_component_width/height (proxy for font size)
      - estimated_stroke_width (2 * mean distance inside components)
      - image_size
    This does not require OCR; it approximates text regions via adaptive thresholding and morphology.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
        gray = np.array(img)

        # Adaptive threshold to get text-like foreground (white on black mask after inversion)
        # We invert so that text pixels are 1 for distance transform convenience.
        binv = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31, 15
        )

        # Clean small speckles, join strokes a bit
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(binv, cv2.MORPH_OPEN, kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

        h, w = clean.shape[:2]
        total_pixels = float(h * w)
        text_pixels = float(np.count_nonzero(clean > 0))
        ink_ratio = text_pixels / total_pixels if total_pixels else 0.0

        # Connected components to estimate component size
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
        # Filter out background label 0 and too small/too large components
        whs = []
        for i in range(1, num_labels):
            x, y, ww, hh, area = stats[i]
            if area < 10:
                continue
            if ww > w * 0.9 or hh > h * 0.9:
                continue
            whs.append((ww, hh))
        avg_w = float(np.mean([p[0] for p in whs])) if whs else 0.0
        avg_h = float(np.mean([p[1] for p in whs])) if whs else 0.0

        # Stroke width proxy via distance transform on the text mask
        # (distance inside foreground to nearest background). Multiply by 2 for diameter approx.
        dist = cv2.distanceTransform(clean, distanceType=cv2.DIST_L2, maskSize=3)
        sw = float(2.0 * np.mean(dist[clean > 0])) if text_pixels > 0 else 0.0

        # Intensity means
        mean_text = float(np.mean(gray[clean > 0])) if text_pixels > 0 else 0.0
        mean_bg   = float(np.mean(gray[clean == 0])) if (total_pixels - text_pixels) > 0 else 0.0

        return {
            "image_size": [w, h],
            "mean_text_intensity": mean_text,
            "mean_bg_intensity": mean_bg,
            "ink_ratio": ink_ratio,
            "avg_component_width": avg_w,
            "avg_component_height": avg_h,
            "estimated_stroke_width": sw,
            "notes": [
                "Proxies derived without OCR; used when DI styleFont is unavailable."
            ]
        }
    except Exception as ex:
        return {"error": repr(ex), "notes": ["Image proxies failed"]}


def _compute_pdf_style_fallback(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Use PyMuPDF to read true font spans from born-digital PDFs.
    """
    if extract_pdf_styles is None:
        return {"error": "PyMuPDF not available or helper not imported"}
    try:
        spans = extract_pdf_styles(pdf_path=None, pdf_bytes=pdf_bytes)
        return {"pdf_spans": spans, "notes": ["True PDF style spans via PyMuPDF"]}
    except Exception as ex:
        return {"error": repr(ex), "notes": ["PyMuPDF extraction failed"]}


def _build_enum_map():
    """
    Try new GA enum names first; fall back to older names if needed.
    Returns (enum_map, use_new_flag) where enum_map maps normalized feature strings to SDK enums/strings.
    """
    enum_map = {}
    try:
        # Newer GA (names may be DocumentAnalysisFeature.STYLE_FONT, KEY_VALUE_PAIRS, etc.)
        from azure.ai.documentintelligence.models import DocumentAnalysisFeature as F  # type: ignore
        enum_map["stylefont"]       = getattr(F, "STYLE_FONT", "styleFont")
        enum_map["keyvaluepairs"]   = getattr(F, "KEY_VALUE_PAIRS", "keyValuePairs")
        enum_map["ocrhighresolution"] = getattr(F, "OCR_HIGH_RESOLUTION", "ocrHighResolution")
        return enum_map, True
    except Exception:
        pass
    try:
        # Older names (AnalyzeFeature)
        from azure.ai.documentintelligence.models import AnalyzeFeature as F  # type: ignore
        # No STYLE_FONT in older—fall back to OCR_FONT
        enum_map["stylefont"]       = getattr(F, "OCR_FONT", "ocrFont")
        enum_map["keyvaluepairs"]   = getattr(F, "KEY_VALUE_PAIRS", "keyValuePairs")
        enum_map["ocrhighresolution"] = getattr(F, "OCR_HIGH_RESOLUTION", "ocrHighResolution")
        return enum_map, False
    except Exception:
        # Last resort: return strings so service still sees feature tokens if accepted
        return {
            "stylefont": "styleFont",
            "keyvaluepairs": "keyValuePairs",
            "ocrhighresolution": "ocrHighResolution",
        }, None


# ------------------------- service wrapper -------------------------

class DIService:
    def __init__(self, endpoint: str, key: str, timeout: int = 120):
        self.endpoint = endpoint
        self.key = key
        self.timeout = timeout
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

    def analyze_layout(
        self,
        content: bytes,
        content_type: str,
        features: Optional[List[str]] = None,
        model_id: str = "prebuilt-layout",
    ) -> Dict[str, Any]:
        """
        Run DI layout (or document) with add-ons (styleFont/keyValuePairs).
        Gracefully handles regions that don't support styleFont and attaches a local style fallback.
        """
        # ---------------- feature negotiation setup ----------------
        # Normalize requested features (accept 'ocrFont' but map to 'styleFont')
        requested = [f.lower().replace("-", "").replace("_", "") for f in (features or ["styleFont", "keyValuePairs"])]
        requested = ["stylefont" if f == "ocrfont" else f for f in requested]  # alias

        enum_map, uses_new_enums = _build_enum_map()

        def _to_sdk_features(keys: List[str]) -> List[Any]:
            out = []
            for k in keys:
                out.append(enum_map.get(k, k))
            return out

        # Build request body (prefer 'body=' signature on GA)
        try:
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest  # type: ignore
            analyze_request = AnalyzeDocumentRequest(bytes_source=content)
        except Exception:
            analyze_request = content

        def _call(model: str, keys: List[str]):
            feats = _to_sdk_features(keys)
            try:
                poller = self.client.begin_analyze_document(
                    model_id=model,
                    body=analyze_request,
                    content_type=content_type,
                    features=feats,
                    timeout=self.timeout,
                )
            except TypeError:
                poller = self.client.begin_analyze_document(
                    model,
                    analyze_request=analyze_request,
                    content_type=content_type,
                    features=feats,
                    timeout=self.timeout,
                )
            return poller.result()

        # ---------------- primary call (as requested) ---------------
        negotiation: Dict[str, Any] = {
            "requested": requested[:],
            "used_model_id": model_id,
            "used_features": None,
            "fallbacks": [],
        }

        try:
            result = _call(model_id, requested)
            di_dict = _result_to_dict(result)
            negotiation["used_features"] = requested[:]
            di_dict["_featureNegotiation"] = negotiation
            return di_dict
        except HttpResponseError as e1:
            err1 = str(e1).lower()
            style_rejected = ("stylefont" in err1) or ("ocrfont" in err1)
            kv_rejected_on_layout = ("keyvaluepairs" in err1 and model_id == "prebuilt-layout")

            # Prepare a mutable copy of features
            current_feats = requested[:]

            # If styleFont rejected, drop it and note fallback
            if style_rejected and "stylefont" in current_feats:
                current_feats = [k for k in current_feats if k != "stylefont"]
                negotiation["fallbacks"].append("styleFont rejected by service; retrying without it")

            # If keyValuePairs is rejected on layout, try prebuilt-document
            use_model = model_id
            if kv_rejected_on_layout:
                use_model = "prebuilt-document"
                negotiation["fallbacks"].append("keyValuePairs rejected on prebuilt-layout; switching to prebuilt-document")

            # Retry with adjusted combo
            try:
                result = _call(use_model, current_feats)
                di_dict = _result_to_dict(result)
                negotiation["used_model_id"] = use_model
                negotiation["used_features"] = current_feats[:]

                # If styleFont was rejected, attach local fallback style info
                if style_rejected:
                    if _infer_is_pdf(content_type):
                        di_dict["style_fallback"] = {
                            "source": "pymupdf",
                            **_compute_pdf_style_fallback(content)
                        }
                    else:
                        di_dict["style_fallback"] = {
                            "source": "opencv_proxies",
                            "proxies": _image_style_proxies(content)
                        }

                di_dict["_featureNegotiation"] = negotiation
                return di_dict

            except HttpResponseError as e2:
                # If prebuilt-document still rejects keyValuePairs, drop it and retry once more
                if "keyvaluepairs" in str(e2).lower():
                    negotiation["fallbacks"].append("keyValuePairs rejected again; dropping keyValuePairs")
                    current_feats = [k for k in current_feats if k != "keyvaluepairs"]
                    try:
                        result = _call(use_model, current_feats)
                        di_dict = _result_to_dict(result)
                        negotiation["used_features"] = current_feats[:]
                        if style_rejected:
                            if _infer_is_pdf(content_type):
                                di_dict["style_fallback"] = {
                                    "source": "pymupdf",
                                    **_compute_pdf_style_fallback(content)
                                }
                            else:
                                di_dict["style_fallback"] = {
                                    "source": "opencv_proxies",
                                    "proxies": _image_style_proxies(content)
                                }
                        di_dict["_featureNegotiation"] = negotiation
                        return di_dict
                    except Exception as e3:
                        return {"error": f"After feature fallbacks: {e3!r}", "_featureNegotiation": negotiation}
                # No recognized recovery path
                return {"error": str(e2), "_featureNegotiation": negotiation}
        except Exception as ex:
            return {"error": repr(ex)}
