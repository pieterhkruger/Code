"""
core/logging_utils.py
=====================
Tiny logging helpers to standardize debug/info messages across modules.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Tuple, Optional

@contextmanager
def timed(section: str) -> Iterator[Tuple[str, float]]:
    start = time.perf_counter()
    try:
        yield (section, start)
    finally:
        end = time.perf_counter()
        elapsed = end - start
        print(f"[timed] {section}: {elapsed:.2f}s")
