"""
TrOCR (HF, GPU) — Tool unit test skeleton

Contract:
    run_tool(pdf_path: str) -> dict
"""

from pathlib import Path
from typing import Dict, Any

def run_tool(pdf_path: str) -> Dict[str, Any]:
    p = Path(pdf_path)
    return {
        "tool": "TrOCR (HF, GPU)",
        "pdf_path": str(p),
        "pdf_exists": p.exists(),
        "status": "not_implemented",
        "message": "Replace skeleton with real logic."
    }
