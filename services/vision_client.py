"""
services/vision_client.py
=========================
Azure AI Vision Image Analysis v4 wrapper. Supports READ OCR, OBJECTS, BRANDS,
and CAPTION to capture non-document cues and brand/logo signals.

- For PDFs, convert first N pages to images (dpi from config) and analyze each page.
- For images, analyze directly.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional
from PIL import Image

from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

def _result_to_dict(vision_result: Any) -> Dict[str, Any]:
    """
    Normalize Azure AI Vision v4 analyze() result into a plain dict.

    - caption → {"text", "confidence"}
    - tags    → [{"name","confidence"}]
    - objects → [{"name","confidence","bbox"}]  (bbox = [x,y,w,h], from bounding_box or polygon)
    - read    → {"lines":[{"text","bbox","words":[...]}]}       (bbox normalized as above)

    Notes:
      * v4 may provide polygons instead of simple boxes. We convert polygons to a tight axis-aligned box.
      * The SDK shapes vary slightly across minor releases; all accesses are guarded.
    """
    out: Dict[str, Any] = {}

    def _poly_to_box(poly) -> Optional[list]:
        """Convert polygon points (with .x/.y) to [x, y, w, h]."""
        try:
            pts = []
            # poly can be an iterable of points or None
            for p in (poly or []):
                x = getattr(p, "x", None)
                y = getattr(p, "y", None)
                if x is not None and y is not None:
                    pts.append((float(x), float(y)))
            if not pts:
                return None
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
            return [xmin, ymin, xmax - xmin, ymax - ymin]
        except Exception:
            return None

    def _box_from_any(bb, poly) -> Optional[list]:
        """Prefer bounding_box; fall back to polygon."""
        # Try structured bounding_box first
        if bb is not None:
            try:
                # SDK commonly exposes .x/.y/.w/.h
                return [bb.x, bb.y, bb.w, bb.h]
            except Exception:
                # Sometimes it may already be a list/tuple
                try:
                    return [bb[0], bb[1], bb[2], bb[3]]
                except Exception:
                    pass
        # Fall back to polygon → box
        return _poly_to_box(poly)

    # ---- caption -------------------------------------------------------------
    try:
        cap = getattr(vision_result, "caption", None)
        if cap is not None:
            out["caption"] = {
                "text": getattr(cap, "text", None),
                "confidence": getattr(cap, "confidence", None),
            }
    except Exception:
        pass

    # ---- tags (v4) -----------------------------------------------------------
    try:
        tags = getattr(vision_result, "tags", None)
        if tags:
            out["tags"] = [
                {"name": getattr(t, "name", None), "confidence": getattr(t, "confidence", None)}
                for t in tags
                if getattr(t, "name", None)  # keep only tags with a name
            ]
    except Exception:
        pass

    # ---- objects -------------------------------------------------------------
    try:
        objs = getattr(vision_result, "objects", None)
        if objs:
            olist = []
            for o in objs:
                name = getattr(o, "name", None)
                if not name:
                    try:
                        t0 = (getattr(o, "tags", None) or [None])[0]
                        name = getattr(t0, "name", None) or "object"
                    except Exception:
                        name = "object"

                bb = getattr(o, "bounding_box", None)
                poly = getattr(o, "bounding_polygon", None) or getattr(o, "polygon", None)
                bbox = _box_from_any(bb, poly)  # your existing helper to normalize polygons → [x,y,w,h]

                olist.append({
                    "name": name,
                    "confidence": getattr(o, "confidence", None),
                    "bbox": bbox,
                })

            # Filter out placeholders/noise
            olist = [
                o for o in olist
                if o.get("name") and o["name"] != "object" and o.get("bbox") and (o.get("confidence") is not None)
            ]
            out["objects"] = olist
    except Exception:
        pass


    # ---- read (OCR) ----------------------------------------------------------
    try:
        read = getattr(vision_result, "read", None)
        if read is not None:
            lines_out = []
            for block in (getattr(read, "blocks", None) or []):
                for line in (getattr(block, "lines", None) or []):
                    # Prefer bounding_box; fall back to polygon fields
                    bb = getattr(line, "bounding_box", None)
                    poly = getattr(line, "bounding_polygon", None) or getattr(line, "polygon", None)
                    bbox = _box_from_any(bb, poly)

                    # Collect word texts (geometry for words not needed in Step 1)
                    words = []
                    for w in (getattr(line, "words", None) or []):
                        words.append(getattr(w, "text", None))

                    lines_out.append({
                        "text": getattr(line, "text", None),
                        "bbox": bbox,
                        "words": words,
                    })
            out["read"] = {"lines": lines_out}
    except Exception:
        pass

    return out


class VisionService:
    def __init__(self, endpoint: str, key: str, timeout: int = 120):
        self.endpoint = endpoint
        self.key = key
        self.client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        self.timeout = timeout

    def _clamp_image_dimensions(self, image_bytes: bytes, min_side: int = 50, max_side: int = 16000) -> bytes:
        """
        Ensure the image dimensions satisfy Vision v4 constraints (>=50 and <=16000).
        If resizing is needed, keep aspect ratio and re-encode (preserving format when possible).
        """
        try:
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            # Determine scale factor to bring both sides into [min_side, max_side]
            scale = 1.0
            if max(w, h) > max_side:
                scale = min(scale, max_side / float(max(w, h)))
            if min(w, h) < min_side:
                scale = max(scale, min_side / float(min(w, h)))
            if scale != 1.0:
                new_size = (max(int(w * scale), min_side), max(int(h * scale), min_side))
                fmt = (img.format or "JPEG").upper()
                if fmt not in ("JPEG", "PNG", "WEBP"):
                    fmt = "JPEG"
                out = io.BytesIO()
                img.resize(new_size, Image.LANCZOS).save(out, format=fmt, quality=90, optimize=True)
                return out.getvalue()
        except Exception:
            # If anything goes wrong, fall back to original bytes.
            pass
        return image_bytes

    def analyze_image_bytes(self, image_bytes: bytes, features: Optional[List[VisualFeatures]] = None) -> Dict[str, Any]:
        features = features or [VisualFeatures.READ, VisualFeatures.OBJECTS, VisualFeatures.TAGS, VisualFeatures.CAPTION]
        try:
            # Clamp image to Vision v4 size limits before calling the service
            image_bytes = self._clamp_image_dimensions(image_bytes)
            result = self.client.analyze(image_data=image_bytes, visual_features=features, timeout=self.timeout)
            return _result_to_dict(result)
        except HttpResponseError as e:
            return {"error": str(e)}
        except Exception as ex:
            return {"error": repr(ex)}
