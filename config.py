"""
config.py
=========
Project-wide configuration and environment loading for the OCR/DI/Vision POC.

- Loads API keys and endpoints from a .env file using python-dotenv.
- Centralizes constants (default DPI, output folders, country bias, etc.).
- Provides small helpers for creating per-run output directories.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load .env from project root if present
load_dotenv(override=True)

# ---- Constants (tune as needed) ----
PROJECT_NAME = "SA_OCR_Fraud_POC"
DEFAULT_COUNTRY_ISO = "ZA"  # South Africa bias for downstream modules
OUTPUT_ROOT = "outputs"
TMP_ROOT = "tmp"
PDF_TO_IMAGE_DPI = 300  # Preserve quality when rasterizing for Vision (DI will keep vector text)
MAX_PAGES_FOR_VISION_ON_PDF = 3  # Safety valve for photos/brand checks on first pages

# Token & timeout leeway
HTTP_TIMEOUT_SECS = 120

@dataclass
class AzureConfig:
    di_endpoint: Optional[str] = os.getenv("AZURE_DI_ENDPOINT") or None
    di_key: Optional[str] = os.getenv("AZURE_DI_KEY") or None
    vision_endpoint: Optional[str] = os.getenv("AZURE_VISION_ENDPOINT") or None
    vision_key: Optional[str] = os.getenv("AZURE_VISION_KEY") or None

@dataclass
class LLMConfig:
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY") or None
    grok_api_key: Optional[str] = os.getenv("GROK_API_KEY") or None

@dataclass
class MapsConfig:
    google_maps_api_key: Optional[str] = os.getenv("GOOGLE_MAPS_API_KEY") or None

@dataclass
class AppConfig:
    azure: AzureConfig
    llm: LLMConfig
    maps: MapsConfig

def get_config() -> AppConfig:
    return AppConfig(azure=AzureConfig(), llm=LLMConfig(), maps=MapsConfig())

def ensure_output_dir(run_id: str) -> str:
    """
    Create and return a per-run output directory path under OUTPUT_ROOT.
    """
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def new_run_id(prefix: str = "run") -> str:
    """
    Generate a unique run id (e.g., run-2025-09-04T10-22-30-uuid).
    """
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
    return f"{prefix}-{ts}"
