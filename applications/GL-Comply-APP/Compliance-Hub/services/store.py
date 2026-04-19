# -*- coding: utf-8 -*-
"""In-memory job + report store. Swap for PG-backed in production (COMPLY-APP 6)."""

from __future__ import annotations

import threading
from typing import Optional

from schemas.models import UnifiedComplianceReport


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reports: dict[str, UnifiedComplianceReport] = {}

    def save(self, report: UnifiedComplianceReport) -> str:
        with self._lock:
            self._reports[report.job_id] = report
        return report.job_id

    def get(self, job_id: str) -> Optional[UnifiedComplianceReport]:
        with self._lock:
            return self._reports.get(job_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._reports.keys())

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()


_default = JobStore()


def default_store() -> JobStore:
    return _default
