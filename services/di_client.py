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
        content: bytes,
        content_type: str,
        features: Optional[List[str]] = None,
        model_id: str = "prebuilt-layout",
    ) -> Dict[str, Any]:
        """
        Analyze with DI; negotiate features, and fall back to local style extraction
        if the service rejects the content/features.
        """
        requested = [f.lower().replace("-", "").replace("_", "") for f in (features or ["styleFont","keyValuePairs"])]
        requested = ["stylefont" if f == "ocrfont" else f for f in requested]  # accept legacy alias

        enum_map, _ = _build_enum_map()

        def _to_sdk_feats(keys: List[str]) -> List[Any]:
            return [enum_map.get(k, k) for k in keys]

        # Build request body
        try:
            from azure.ai.documentintelligence.models import AnalyzeDocumentRequest  # GA
            analyze_request = AnalyzeDocumentRequest(bytes_source=content)
        except Exception:
            analyze_request = content

        def _call(model: str, keys: List[str], ct_override: Optional[str] = None):
            feats = _to_sdk_feats(keys)
            ctype = ct_override or content_type
            try:
                poller = self.client.begin_analyze_document(
                    model_id=model,
                    body=analyze_request,
                    content_type=ctype,
                    features=feats,
                    timeout=self.timeout,
                )
            except TypeError:
                poller = self.client.begin_analyze_document(
                    model,
                    analyze_request=analyze_request,
                    content_type=ctype,
                    features=feats,
                    timeout=self.timeout,
                )
            return poller.result()

        negotiation: Dict[str, Any] = {
            "requested": requested[:],
            "used_model_id": model_id,
            "used_features": None,
            "fallbacks": [],
        }

        def _attach_style_fallback(di_dict: Dict[str, Any]):
            # attach proxies/spans so downstream still gets style signals
            if _infer_is_pdf(content_type):
                di_dict["style_fallback"] = {"source": "pymupdf", **_compute_pdf_style_fallback(content)}
            else:
                di_dict["style_fallback"] = {"source": "opencv_proxies", "proxies": _image_style_proxies(content)}

        # 1) primary attempt
        try:
            res = _call(model_id, requested)
            d = _result_to_dict(res)
            negotiation["used_features"] = requested[:]
            d["_featureNegotiation"] = negotiation
            return d

        except HttpResponseError as e1:
            err1 = str(e1).lower()
            unsupported = "unsupportedcontent" in err1 or "bad or unrecognizable" in err1
            style_rejected = ("stylefont" in err1) or ("ocrfont" in err1)
            kv_rejected_on_layout = ("keyvaluepairs" in err1 and model_id == "prebuilt-layout")

            current_feats = requested[:]
            use_model = model_id

            if style_rejected and "stylefont" in current_feats:
                current_feats = [k for k in current_feats if k != "stylefont"]
                negotiation["fallbacks"].append("styleFont rejected; retrying without it")

            if kv_rejected_on_layout:
                use_model = "prebuilt-document"
                negotiation["fallbacks"].append("keyValuePairs rejected on layout; switching to prebuilt-document")

            # 2) retry after feature/model tweaks
            try:
                res = _call(use_model, current_feats)
                d = _result_to_dict(res)
                negotiation["used_model_id"] = use_model
                negotiation["used_features"] = current_feats[:]
                if style_rejected:
                    _attach_style_fallback(d)
                d["_featureNegotiation"] = negotiation
                return d
            except HttpResponseError as e2:
                # 3) if UnsupportedContent, try application/octet-stream
                if unsupported:
                    negotiation["fallbacks"].append("UnsupportedContent; retry with application/octet-stream")
                    try:
                        res = _call(use_model, current_feats, ct_override="application/octet-stream")
                        d = _result_to_dict(res)
                        negotiation["used_model_id"] = use_model
                        negotiation["used_features"] = current_feats[:]
                        d["_featureNegotiation"] = negotiation
                        if style_rejected:
                            _attach_style_fallback(d)
                        return d
                    except Exception as e2b:
                        pass

                # 4) if still failing, drop all add-ons and try again once
                if current_feats:
                    negotiation["fallbacks"].append("Dropping all add-ons and retrying")
                    try:
                        res = _call(use_model, [])
                        d = _result_to_dict(res)
                        negotiation["used_model_id"] = use_model
                        negotiation["used_features"] = []
                        d["_featureNegotiation"] = negotiation
                        # No styleFont now; attach local style
                        _attach_style_fallback(d)
                        return d
                    except Exception as e3:
                        pass

                # 5) give up: return error + local style fallback so downstream isn’t blind
                out = {"error": str(e2)}
                _attach_style_fallback(out)
                out["_featureNegotiation"] = negotiation
                return out

        except Exception as ex:
            out = {"error": repr(ex)}
            _attach_style_fallback(out)
            out["_featureNegotiation"] = {
                "requested": requested[:],
                "used_model_id": model_id,
                "used_features": None,
                "fallbacks": ["Client-side exception; attached style fallback"],
            }
            return out
# --- END OF FILE ---
