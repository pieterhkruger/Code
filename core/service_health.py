"""
core/service_health.py
======================
Best-effort service health checks to pre-flight API access before running modules.
We keep it lightweight: try client construction and a tiny call where feasible.

Checks:
- Azure Document Intelligence: client construction
- Azure Vision (Image Analysis v4): analyze a tiny synthetic image
- OpenAI: list models (lightweight)
- Grok: key presence (reachability not probed here)
"""

from __future__ import annotations

import io
import time
from typing import Optional

from PIL import Image

from core.models import ServiceHealthReport, ServiceStatus
from config import get_config
from services.di_client import DIService
from services.vision_client import VisionService


def _check_openai() -> ServiceStatus:
    cfg = get_config()
    if not cfg.llm.openai_api_key:
        return ServiceStatus(
            name="openai",
            is_configured=False,
            ok=False,
            message="Not configured",
        )

    try:
        # OpenAI SDK v1.x
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=cfg.llm.openai_api_key)
        t0 = time.perf_counter()
        _ = client.models.list()
        dt = (time.perf_counter() - t0) * 1000.0
        return ServiceStatus(
            name="openai",
            is_configured=True,
            ok=True,
            message="OpenAI reachable",
            roundtrip_ms=dt,
        )
    except Exception as ex:
        return ServiceStatus(
            name="openai",
            is_configured=True,
            ok=False,
            message=repr(ex),
        )


def _check_grok() -> ServiceStatus:
    cfg = get_config()
    if not cfg.llm.grok_api_key:
        return ServiceStatus(name="grok", is_configured=False, ok=False, message="Not configured")

    try:
        from services.grok_client import GrokClient
        gc = GrokClient(api_key=cfg.llm.grok_api_key)
        # Cost-free: list models
        t0 = time.perf_counter()
        _ = gc.list_models()
        dt = (time.perf_counter() - t0) * 1000.0
        return ServiceStatus(
            name="grok",
            is_configured=True,
            ok=True,
            message=f"Grok reachable (model={gc.model})",
            roundtrip_ms=dt,
        )
    except Exception as ex:
        return ServiceStatus(name="grok", is_configured=True, ok=False, message=repr(ex))



def _check_di() -> ServiceStatus:
    cfg = get_config()
    if not (cfg.azure.di_endpoint and cfg.azure.di_key):
        return ServiceStatus(
            name="azure_di",
            is_configured=False,
            ok=False,
            message="Not configured",
        )
    try:
        _ = DIService(cfg.azure.di_endpoint, cfg.azure.di_key)
        # Client construction is a good pre-flight check without incurring cost.
        return ServiceStatus(
            name="azure_di",
            is_configured=True,
            ok=True,
            message="Client constructed",
        )
    except Exception as ex:
        return ServiceStatus(
            name="azure_di",
            is_configured=True,
            ok=False,
            message=repr(ex),
        )


def _check_vision() -> ServiceStatus:
    cfg = get_config()
    if not (cfg.azure.vision_endpoint and cfg.azure.vision_key):
        return ServiceStatus(
            name="azure_vision",
            is_configured=False,
            ok=False,
            message="Not configured",
        )
    try:
        svc = VisionService(cfg.azure.vision_endpoint, cfg.azure.vision_key)
        # Use >= 50 px per side to satisfy Vision v4 limits
        img = Image.new("RGB", (128, 128), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)

        from azure.ai.vision.imageanalysis.models import VisualFeatures
        
        t0 = time.perf_counter()
        # Keep the probe light: ask only for CAPTION
        res = svc.analyze_image_bytes(buf.getvalue(), features=[VisualFeatures.CAPTION])
        dt = (time.perf_counter() - t0) * 1000.0
        ok = "error" not in res
        return ServiceStatus(
            name="azure_vision",
            is_configured=True,
            ok=ok,
            message="Analyze call ok" if ok else f"Analyze failed: {res.get('error')!s}",
            roundtrip_ms=dt,
        )
    except Exception as ex:
        return ServiceStatus(
            name="azure_vision",
            is_configured=True,
            ok=False,
            message=repr(ex),
        )


def get_health_report() -> ServiceHealthReport:
    return ServiceHealthReport(
        azure_di=_check_di(),
        azure_vision=_check_vision(),
        openai=_check_openai(),
        grok=_check_grok(),
    )
