"""
modules/mod5_llm_eval.py
=================
Roadmap Step 5: Send prompts to OpenAI/Grok; receive JSON output

This is a placeholder to make the Streamlit UI and module registry forward-compatible.
Implementations will be filled in subsequent steps while preserving the same interfaces.
"""

from __future__ import annotations

from core.models import ModuleOutput

def run(*args, **kwargs) -> ModuleOutput:
    return ModuleOutput(
        run_id=kwargs.get("run_id", "run-unknown"),
        module_name="mod5_llm_eval",
        ok=False,
        message="Not implemented yet. Use Step 1 to generate inputs first.",
        payload=None,
        artifact_paths={},
    )
