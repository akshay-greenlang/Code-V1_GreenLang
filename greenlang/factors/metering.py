# -*- coding: utf-8 -*-
"""
Lightweight request metering for Factors API (in-process counters; export to metrics later).
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import DefaultDict, Dict


class FactorsRequestMeter:
    """Thread-safe counters keyed by route template or path."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: DefaultDict[str, int] = defaultdict(int)

    def record(self, key: str, increment: int = 1) -> None:
        with self._lock:
            self._counts[key] += increment

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)


GLOBAL_METER = FactorsRequestMeter()
