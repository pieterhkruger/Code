"""
services/pdf_style_probe.py
===========================
Extract true PDF font/style spans using PyMuPDF (fitz).
Falls back gracefully if the file is not a PDF or if parsing fails.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import fitz  # PyMuPDF

def extract_pdf_styles(pdf_path: Optional[str] = None, pdf_bytes: Optional[bytes] = None) -> List[Dict[str, Any]]:
    """
    Return a list of spans with true style info:
      [
        {"page": 0, "text": "...", "font": "TimesNewRomanPSMT", "size": 11.0,
         "color": 0, "flags": 0, "bbox": [x0, y0, x1, y1]}
      ]
    Either `pdf_path` or `pdf_bytes` is required (prefer pdf_path if both provided).
    """
    if not pdf_path and not pdf_bytes:
        return []

    doc = None
    try:
        if pdf_path:
            doc = fitz.open(pdf_path)
        else:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        out: List[Dict[str, Any]] = []
        for pno in range(len(doc)):
            page = doc[pno]
            # blocks -> lines -> spans
            d = page.get_text("dict")
            for block in d.get("blocks", []):
                # type 0 = text
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        out.append({
                            "page": pno,
                            "text": span.get("text", ""),
                            "font": span.get("font", None),
                            "size": span.get("size", None),
                            "color": span.get("color", None),
                            "flags": span.get("flags", None),
                            "bbox": span.get("bbox", None),
                        })
        return out
    except Exception:
        return []
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
