"""
modules/mod1_acquire_raw.py
===========================
Roadmap Step 1: Acquire *raw signals* from Azure Document Intelligence (DI) and
Azure Vision Image Analysis v4, then persist them for downstream modules.

Design goals:
- Keep vector text by feeding original PDF bytes to DI (no rasterization here).
- Optionally run Vision (READ/OBJECTS/BRANDS/CAPTION) on images *and* on the first N
  PDF pages converted to images (for brand/logo verification and photo-like cues).
- Return a single RawSignalsPayload with file metadata and both raw responses.
- Write artifacts to outputs/<run_id>/ for debugging and chaining to later modules.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, List, Tuple

from pydantic import ValidationError
from PIL import Image

from config import get_config, ensure_output_dir, PDF_TO_IMAGE_DPI, MAX_PAGES_FOR_VISION_ON_PDF, DEFAULT_COUNTRY_ISO
from core.models import RawSignalsPayload, FileMeta, RawDIResult, RawVisionResult, RawVisionPageResult, ModuleOutput
from core.io import save_json, safe_name
from core.pdf_utils import infer_content_type, pdf_to_images, pil_to_jpeg_bytes
from services.di_client import DIService
from services.vision_client import VisionService

def _try_pdf_page_count(data: bytes) -> Optional[int]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        n = len(doc)
        doc.close()
        return int(n)
    except Exception:
        return None


def _read_file_bytes(path_or_bytes) -> Tuple[bytes, str, str, int]:
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raise ValueError("Binary input provided without filename; supply a tuple (filename, bytes).")
    if isinstance(path_or_bytes, tuple):
        filename, data = path_or_bytes
        content_type = infer_content_type(filename)
        return (bytes(data), filename, content_type, len(data))

    # Assume path
    filename = os.path.basename(str(path_or_bytes))
    with open(path_or_bytes, "rb") as f:
        data = f.read()
    content_type = infer_content_type(filename)
    return (data, filename, content_type, len(data))

def run(
    path_or_tuple,
    run_id: str,
    call_di: bool = True,
    call_vision: bool = True,
    vision_on_pdf_pages: bool = True,
    vision_pages_limit: int = MAX_PAGES_FOR_VISION_ON_PDF,
    store_source_file: bool = False,   # NEW
) -> ModuleOutput:

    """
    Execute Step 1 on a given file path or (filename, bytes) tuple.
    Returns ModuleOutput with RawSignalsPayload and artifact paths.
    """
    cfg = get_config()
    out_dir = ensure_output_dir(run_id)

    # ---- Read input
    data, filename, content_type, size = _read_file_bytes(path_or_tuple)

    # Optionally persist the original for downstream modules (dev/test convenience)
    source_path = None
    if store_source_file:
        ext = os.path.splitext(filename)[1] or ".bin"
        source_path = os.path.join(out_dir, f"source{ext}")
        with open(source_path, "wb") as _sf:
            _sf.write(data)

    # ---- DI (keep vector text when PDF)
    di_json: Dict = {}
    if call_di and cfg.azure.di_endpoint and cfg.azure.di_key:
        di = DIService(cfg.azure.di_endpoint, cfg.azure.di_key)
        di_json = di.analyze_layout(
            content=data,
            content_type=content_type,
            features=["styleFont", "keyValuePairs"],
            model_id="prebuilt-layout",  # can switch to "prebuilt-document" later if needed
        )
        # --- Compatibility: guarantee both keys for downstream readers ---
        # Some older runs only had {"result": {...}}; newer code prefers {"layout": {...}}.
        # Keep them as aliases to the *same* dict so both paths work.
        if isinstance(di_json, dict) and "result" in di_json and "layout" not in di_json:
            di_json["layout"] = di_json["result"]
        save_json(os.path.join(out_dir, "di_raw.json"), di_json)
    else:
        di_json = {"skipped": True, "reason": "DI not configured or disabled"}
        save_json(os.path.join(out_dir, "di_raw.json"), di_json)

    # ---- Vision
    vision_pages: List[RawVisionPageResult] = []
    if call_vision and cfg.azure.vision_endpoint and cfg.azure.vision_key:
        vs = VisionService(cfg.azure.vision_endpoint, cfg.azure.vision_key)

        if content_type == "application/pdf" and vision_on_pdf_pages:
            try:
                images = pdf_to_images(data, dpi=PDF_TO_IMAGE_DPI, first_n_pages=vision_pages_limit)
                for i, img in enumerate(images):
                    img_bytes = pil_to_jpeg_bytes(img)
                    vdict = vs.analyze_image_bytes(img_bytes)
                    vision_pages.append(RawVisionPageResult(page_index=i, result=vdict))
            except Exception as ex:
                vision_pages.append(RawVisionPageResult(page_index=0, result={"error": repr(ex)}))
        else:
            # Single image document
            try:
                vdict = vs.analyze_image_bytes(data)
                vision_pages.append(RawVisionPageResult(page_index=0, result=vdict))
            except Exception as ex:
                vision_pages.append(RawVisionPageResult(page_index=0, result={"error": repr(ex)}))

        # Persist vision pages
        vision_all = {"pages": [p.model_dump() for p in vision_pages]}
        save_json(os.path.join(out_dir, "vision_raw.json"), vision_all)
    else:
        vision_all = {"pages": [], "skipped": True, "reason": "Vision not configured or disabled"}
        save_json(os.path.join(out_dir, "vision_raw.json"), vision_all)

    # ---- Compose payload
    pc = _try_pdf_page_count(data) if content_type == "application/pdf" else None
    file_meta = FileMeta(filename=filename, content_type=content_type, size_bytes=size, page_count=pc)
    payload = RawSignalsPayload(
        file_meta=file_meta,
        di=RawDIResult(result=di_json),
        vision=RawVisionResult(pages=vision_pages),
        country_bias=DEFAULT_COUNTRY_ISO,
        notes=[
            "Step 1 raw acquisition complete",
            "DI preserves vector text; Vision used for brand/objects/READ on images",
        ],
    )

    # Persist combined
    combined_path = os.path.join(out_dir, "raw_signals.step1.json")
    save_json(combined_path, payload.model_dump())

    artifacts = {
        "combined": combined_path,
        "di_raw": os.path.join(out_dir, "di_raw.json"),
        "vision_raw": os.path.join(out_dir, "vision_raw.json"),
    }
    if source_path:
        artifacts["source_file"] = source_path

    return ModuleOutput(
        run_id=run_id,
        module_name="mod1_acquire_raw",
        ok=True,
        message="Acquired raw DI/Vision signals",
        payload=payload,
        artifact_paths=artifacts,
    )
