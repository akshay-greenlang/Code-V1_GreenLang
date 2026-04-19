# -*- coding: utf-8 -*-
"""In-memory computation store.

Stores ComputationResponse by computation_id for retrieval by API. Production
deployments swap this for a PG-backed implementation; migrations for
`scope_computations` and `scope_activity_results` land alongside V437+ when
INFRA-002 provisions the prod schema (future work, track A2 Factors Deploy).
"""

from __future__ import annotations

import threading
from typing import Optional

from greenlang.scope_engine.models import ComputationResponse


class InMemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, ComputationResponse] = {}

    def put(self, response: ComputationResponse) -> str:
        cid = response.computation.computation_id
        with self._lock:
            self._data[cid] = response
        return cid

    def get(self, computation_id: str) -> Optional[ComputationResponse]:
        with self._lock:
            return self._data.get(computation_id)

    def all_ids(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_default_store = InMemoryStore()


def get_default_store() -> InMemoryStore:
    return _default_store
