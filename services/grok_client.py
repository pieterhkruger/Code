"""
services/grok_client.py
=======================
Thin wrapper for xAI Grok via OpenAI-compatible client.

- Uses the existing `openai` Python SDK, but points `base_url` to xAI.
- Model name defaults to the environment variable GROK_MODEL (else "grok-4").

Usage (later, in Step 5):
    from services.grok_client import GrokClient
    gc = GrokClient(api_key=cfg.llm.grok_api_key)
    resp = gc.chat([{"role": "user", "content": "Say hi"}])
    print(resp["choices"][0]["message"]["content"])
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from openai import OpenAI  # uses your existing dependency

XAI_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")

class GrokClient:
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or os.getenv("GROK_MODEL", "grok-4")
        # OpenAI-compatible client but pointed at xAI
        self.client = OpenAI(api_key=self.api_key, base_url=XAI_BASE_URL)

    def list_models(self) -> Dict[str, Any]:
        """Cost-free reachability check."""
        return self.client.models.list().model_dump()

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform a chat completion call against Grok.
        Returns the raw dict so downstream can stay model-agnostic.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return resp.model_dump()
    