"""
modules/mod7_merge.py
=================
Roadmap Step 7: Merge results into final JSON export

This is a placeholder to make the Streamlit UI and module registry forward-compatible.
Implementations will be filled in subsequent steps while preserving the same interfaces.
"""

from __future__ import annotations

from core.models import ModuleOutput

def run(*args, **kwargs) -> ModuleOutput:
    return ModuleOutput(
        run_id=kwargs.get("run_id", "run-unknown"),
        module_name="mod7_merge",
        ok=False,
        message="Not implemented yet. Use Step 1 to generate inputs first.",
        payload=None,
        artifact_paths={},
    )
