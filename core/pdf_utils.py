"""
core/pdf_utils.py
=================
Utilities for determining file types, counting pages, and (when needed) converting
PDF pages to images for Vision analysis. DI itself should receive the original PDF
bytes to preserve vector text.
"""

from __future__ import annotations

import io
import os
from typing import List, Tuple, Optional

from pdf2image import convert_from_bytes
from PIL import Image

def infer_content_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "application/pdf"
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        return "image/tiff"
    # default try image
    return "application/octet-stream"

def pdf_to_images(pdf_bytes: bytes, dpi: int = 300, first_n_pages: Optional[int] = None) -> List[Image.Image]:
    """
    Convert PDF bytes to a list of PIL Images. Requires poppler installed on the system.
    """
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    if first_n_pages is not None:
        images = images[:max(0, first_n_pages)]
    return images

def pil_to_jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = io.BytesIO()
    rgb = img.convert("RGB")
    rgb.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
