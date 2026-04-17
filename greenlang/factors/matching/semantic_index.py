# -*- coding: utf-8 -*-
"""
Semantic retrieval index (M3) — integration point.

Hosted deployments may back this with ``pgvector`` or a sidecar embedding service.
The Factors API does not require vectors for deterministic list/search/match v1.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol


class SemanticIndex(Protocol):
    def embed_text(self, text: str) -> List[float]:
        ...

    def search(self, edition_id: str, vector: List[float], k: int) -> List[Dict[str, Any]]:
        ...


class NoopSemanticIndex:
    """Placeholder until embeddings are configured."""

    def embed_text(self, text: str) -> List[float]:
        return []

    def search(self, edition_id: str, vector: List[float], k: int) -> List[Dict[str, Any]]:
        return []
