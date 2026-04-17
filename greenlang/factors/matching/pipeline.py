# -*- coding: utf-8 -*-
"""
Hybrid matching pipeline for emission factor search (F042).

Stages:
  1. Facet filter (geography, fuel_type, scope)
  2a. Lexical search via token overlap
  2b. Semantic search via vector similarity (pgvector, optional)
  3. Reciprocal Rank Fusion (RRF) combining lexical + semantic rankings
  4. DQS boost
  5. Optional LLM rerank (enterprise, via rerank_fn callback)

Graceful degradation: falls back to lexical-only when semantic index unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from greenlang.factors.catalog_repository import FactorCatalogRepository

logger = logging.getLogger(__name__)

# RRF constant (standard value from Cormack et al. 2009)
RRF_K = 60


@dataclass
class MatchRequest:
    activity_description: str
    geography: Optional[str] = None
    fuel_type: Optional[str] = None
    scope: Optional[str] = None
    limit: int = 10


@dataclass
class HybridConfig:
    """Configuration for hybrid matching weights."""

    lexical_weight: float = 0.4
    semantic_weight: float = 0.6
    dqs_boost_factor: float = 0.01
    semantic_k: int = 50
    rrf_k: int = RRF_K
    enable_semantic: bool = True


def _token_overlap(query: str, blob: str) -> float:
    q = set(query.lower().split())
    b = set(blob.lower().split())
    if not q:
        return 0.0
    return len(q & b) / max(1, len(q))


def _reciprocal_rank_fusion(
    rankings: List[List[str]],
    weights: List[float],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Combine multiple ranked lists using weighted Reciprocal Rank Fusion.

    Args:
        rankings: List of ranked lists (each is a list of factor_ids)
        weights: Weight for each ranking list
        k: RRF smoothing constant (default 60)

    Returns:
        List of (factor_id, rrf_score) sorted by score descending.
    """
    scores: Dict[str, float] = {}
    for ranked_list, weight in zip(rankings, weights):
        for rank, factor_id in enumerate(ranked_list):
            rrf_score = weight / (k + rank + 1)
            scores[factor_id] = scores.get(factor_id, 0.0) + rrf_score

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def run_match(
    repo: FactorCatalogRepository,
    edition_id: str,
    req: MatchRequest,
    *,
    include_preview: bool = False,
    include_connector: bool = False,
    semantic_search_fn: Optional[Callable[[str, str, int], List[Dict[str, Any]]]] = None,
    hybrid_config: Optional[HybridConfig] = None,
    rerank_fn: Optional[Callable[[str, List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Run hybrid matching pipeline.

    Args:
        repo: Factor catalog repository
        edition_id: Edition to search
        req: Match request with activity description and filters
        include_preview: Include preview-status factors
        include_connector: Include connector-only factors
        semantic_search_fn: Optional callback(query, edition_id, k) -> list of
            dicts with {factor_id, similarity}. If None, lexical-only mode.
        hybrid_config: Configuration for lexical/semantic weights
        rerank_fn: Optional LLM rerank callback(activity_description, candidates) -> reranked

    Returns:
        List of match result dicts with factor_id, score, explanation.
    """
    cfg = hybrid_config or HybridConfig()
    q = (req.activity_description or "").strip()[:120] or "energy"
    if len(q) < 2:
        q = "energy"

    # Stage 1: Lexical retrieval + facet filter
    hits = repo.search_factors(
        edition_id,
        query=q,
        geography=req.geography,
        limit=max(50, req.limit * 5),
        include_preview=include_preview,
        include_connector=include_connector,
    )

    # Build record lookup and apply facet filters
    record_map: Dict[str, Any] = {}
    qtext = req.activity_description
    lexical_scored: List[Tuple[float, str]] = []

    for rec in hits:
        if req.fuel_type and rec.fuel_type.lower() != req.fuel_type.lower():
            continue
        if req.scope and rec.scope.value != req.scope:
            continue
        fid = rec.factor_id
        record_map[fid] = rec
        blob = " ".join(
            [rec.factor_id, rec.fuel_type, rec.geography, rec.scope.value,
             " ".join(rec.tags), rec.notes or ""]
        )
        score = _token_overlap(qtext, blob)
        lexical_scored.append((score, fid))

    # Sort lexical by score descending
    lexical_scored.sort(key=lambda x: x[0], reverse=True)
    lexical_ranking = [fid for _, fid in lexical_scored]

    # Stage 2: Semantic retrieval (if available)
    semantic_ranking: List[str] = []
    search_mode = "lexical"

    if semantic_search_fn and cfg.enable_semantic:
        try:
            sem_results = semantic_search_fn(qtext, edition_id, cfg.semantic_k)
            if sem_results:
                semantic_ranking = [r["factor_id"] for r in sem_results]
                # Add any semantic-only factors to record_map if not already present
                # (they may not have been returned by lexical search)
                search_mode = "hybrid"
                logger.debug(
                    "Semantic search returned %d candidates for query=%r",
                    len(semantic_ranking), qtext,
                )
        except Exception as exc:
            logger.warning("Semantic search failed, falling back to lexical: %s", exc)

    # Stage 3: Combine via RRF or use lexical-only
    if search_mode == "hybrid" and semantic_ranking:
        fused = _reciprocal_rank_fusion(
            [lexical_ranking, semantic_ranking],
            [cfg.lexical_weight, cfg.semantic_weight],
            k=cfg.rrf_k,
        )
    else:
        # Lexical-only: convert to same format
        fused = [(fid, score) for score, fid in lexical_scored]

    # Stage 4: DQS boost
    boosted: List[Tuple[str, float]] = []
    for fid, score in fused:
        rec = record_map.get(fid)
        if rec:
            dqs_bonus = cfg.dqs_boost_factor * float(rec.dqs.overall_score)
            boosted.append((fid, score + dqs_bonus))
        else:
            # Factor from semantic search not in lexical results — include with base score
            boosted.append((fid, score))

    boosted.sort(key=lambda x: x[1], reverse=True)

    # Build output
    candidates = boosted[: max(req.limit, 20)]  # Keep at least 20 for reranking
    out: List[Dict[str, Any]] = []
    for fid, score in candidates:
        out.append({
            "factor_id": fid,
            "score": round(float(score), 4),
            "explanation": {
                "why": f"{search_mode}_match",
                "assumptions": [],
                "alternatives_note": "Expand gold set (M5) to tune ranking.",
            },
        })

    # Stage 5: Optional LLM reranking
    if rerank_fn and out:
        try:
            out = rerank_fn(qtext, out)
            for item in out:
                item["explanation"]["why"] = f"llm_reranked_{search_mode}"
            logger.debug("LLM reranking applied to %d candidates", len(out))
        except Exception as exc:
            logger.warning("LLM reranking failed, using base ranking: %s", exc)

    logger.debug(
        "Match query=%r edition=%s mode=%s hits=%d scored=%d",
        req.activity_description, edition_id, search_mode, len(hits), len(out),
    )
    return out[: req.limit]
