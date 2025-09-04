"""
core/models.py
==============
Typed data contracts (Pydantic models) for module I/O, service health, and
raw signals captured from Azure Document Intelligence and Azure Vision.

These models are intentionally generous (dict[str, any]) to hold raw SDK
responses without lossy conversion while still providing structure for the app.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ServiceStatus(BaseModel):
    name: str
    is_configured: bool
    ok: bool
    message: str = ""
    roundtrip_ms: Optional[float] = None

class ServiceHealthReport(BaseModel):
    azure_di: ServiceStatus
    azure_vision: ServiceStatus
    openai: ServiceStatus
    grok: ServiceStatus

class FileMeta(BaseModel):
    filename: str
    content_type: str
    size_bytes: int
    page_count: Optional[int] = None

class RawDIResult(BaseModel):
    """
    Container for Document Intelligence raw JSON-like result.
    """
    result: Dict[str, Any] = Field(default_factory=dict)

class RawVisionPageResult(BaseModel):
    page_index: int
    result: Dict[str, Any] = Field(default_factory=dict)

class RawVisionResult(BaseModel):
    """
    Container for Vision results over zero or more pages (PDF pages or a single image).
    """
    pages: List[RawVisionPageResult] = Field(default_factory=list)

class RawSignalsPayload(BaseModel):
    """
    The core output of Roadmap Step 1: everything we need downstream.
    """
    file_meta: FileMeta
    di: RawDIResult
    vision: RawVisionResult
    # Add early hints that help downstream modules pre-empt work (not authoritative yet):
    country_bias: str = "ZA"
    notes: List[str] = Field(default_factory=list)

class ModuleOutput(BaseModel):
    """
    Standard wrapper returned by each module. `artifact_paths` contains any
    files written to disk (for download buttons / chaining modules).
    """
    run_id: str
    module_name: str
    ok: bool
    message: str = ""
    payload: Optional[RawSignalsPayload] = None
    artifact_paths: Dict[str, str] = Field(default_factory=dict)
