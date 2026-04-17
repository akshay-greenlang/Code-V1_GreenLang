# -*- coding: utf-8 -*-
"""
Comprehensive matching evaluation suite (F044).

Metrics:
- Precision@k (k=1, 3, 5, 10)
- Recall@k (k=5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Per-domain breakdown
- A/B comparison between search modes

Usage:
    evaluator = MatchEvaluator(repo, edition_id)
    report = evaluator.evaluate(gold_cases)
    print(report.summary())
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from greenlang.factors.matching.pipeline import MatchRequest, run_match
from greenlang.factors.catalog_repository import FactorCatalogRepository

logger = logging.getLogger(__name__)


@dataclass
class GoldCase:
    """A single gold evaluation case."""

    id: str
    activity: str
    expected_fuel_type: str
    geography: Optional[str] = None
    scope: Optional[str] = None
    domain: Optional[str] = None
    difficulty: str = "normal"
    expected_factor_ids: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> GoldCase:
        return cls(
            id=d["id"],
            activity=d["activity"],
            expected_fuel_type=d["expected_fuel_type"],
            geography=d.get("geography"),
            scope=d.get("scope"),
            domain=d.get("domain"),
            difficulty=d.get("difficulty", "normal"),
            expected_factor_ids=d.get("expected_factor_ids"),
        )


@dataclass
class CaseResult:
    """Result for a single evaluation case."""

    case_id: str
    expected_fuel_type: str
    matched_fuel_types: List[str]
    matched_factor_ids: List[str]
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    hit_at_10: bool
    reciprocal_rank: float
    domain: Optional[str] = None
    difficulty: str = "normal"
    latency_ms: float = 0.0


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""

    total_cases: int = 0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    avg_latency_ms: float = 0.0
    misses_at_1: int = 0
    misses_at_5: int = 0


@dataclass
class EvalReport:
    """Complete evaluation report with per-domain breakdowns."""

    mode: str  # "lexical", "hybrid", "llm_reranked"
    overall: EvalMetrics = field(default_factory=EvalMetrics)
    by_domain: Dict[str, EvalMetrics] = field(default_factory=dict)
    by_difficulty: Dict[str, EvalMetrics] = field(default_factory=dict)
    case_results: List[CaseResult] = field(default_factory=list)
    timestamp: str = ""

    def summary(self) -> str:
        """Human-readable summary of evaluation results."""
        lines = [
            f"--- Eval Report: {self.mode} ({self.overall.total_cases} cases) ---",
            f"  Precision@1:  {self.overall.precision_at_1:.2%}",
            f"  Precision@3:  {self.overall.precision_at_3:.2%}",
            f"  Precision@5:  {self.overall.precision_at_5:.2%}",
            f"  Precision@10: {self.overall.precision_at_10:.2%}",
            f"  Recall@5:     {self.overall.recall_at_5:.2%}",
            f"  Recall@10:    {self.overall.recall_at_10:.2%}",
            f"  MRR:          {self.overall.mrr:.4f}",
            f"  NDCG@5:       {self.overall.ndcg_at_5:.4f}",
            f"  NDCG@10:      {self.overall.ndcg_at_10:.4f}",
            f"  Avg latency:  {self.overall.avg_latency_ms:.1f}ms",
            f"  Misses@1:     {self.overall.misses_at_1}",
            f"  Misses@5:     {self.overall.misses_at_5}",
        ]
        if self.by_domain:
            lines.append("\n  Per-domain breakdown:")
            for domain, metrics in sorted(self.by_domain.items()):
                lines.append(
                    f"    {domain:20s}: P@1={metrics.precision_at_1:.2%} "
                    f"P@5={metrics.precision_at_5:.2%} MRR={metrics.mrr:.4f} "
                    f"({metrics.total_cases} cases)"
                )
        if self.by_difficulty:
            lines.append("\n  Per-difficulty breakdown:")
            for diff, metrics in sorted(self.by_difficulty.items()):
                lines.append(
                    f"    {diff:20s}: P@1={metrics.precision_at_1:.2%} "
                    f"P@5={metrics.precision_at_5:.2%} MRR={metrics.mrr:.4f} "
                    f"({metrics.total_cases} cases)"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dict (for JSON export)."""
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "overall": {
                "total_cases": self.overall.total_cases,
                "precision_at_1": self.overall.precision_at_1,
                "precision_at_3": self.overall.precision_at_3,
                "precision_at_5": self.overall.precision_at_5,
                "precision_at_10": self.overall.precision_at_10,
                "recall_at_5": self.overall.recall_at_5,
                "recall_at_10": self.overall.recall_at_10,
                "mrr": self.overall.mrr,
                "ndcg_at_5": self.overall.ndcg_at_5,
                "ndcg_at_10": self.overall.ndcg_at_10,
                "avg_latency_ms": self.overall.avg_latency_ms,
            },
            "by_domain": {
                d: {"precision_at_1": m.precision_at_1, "mrr": m.mrr, "total_cases": m.total_cases}
                for d, m in self.by_domain.items()
            },
        }


def _extract_fuel_type(factor_id: str) -> str:
    """Extract fuel_type token from factor_id like 'US:diesel:gallons:...'."""
    parts = factor_id.split(":")
    if len(parts) >= 2:
        return parts[1]
    return ""


def _reciprocal_rank(ranked_items: List[str], expected: str) -> float:
    """1/rank of first match, or 0 if not found."""
    for i, item in enumerate(ranked_items):
        if item == expected:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def _ndcg(relevances: List[float], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return _dcg(relevances, k) / idcg


def _compute_metrics(results: List[CaseResult]) -> EvalMetrics:
    """Compute aggregate metrics from case results."""
    if not results:
        return EvalMetrics()

    n = len(results)
    return EvalMetrics(
        total_cases=n,
        precision_at_1=sum(1 for r in results if r.hit_at_1) / n,
        precision_at_3=sum(1 for r in results if r.hit_at_3) / n,
        precision_at_5=sum(1 for r in results if r.hit_at_5) / n,
        precision_at_10=sum(1 for r in results if r.hit_at_10) / n,
        recall_at_5=sum(1 for r in results if r.hit_at_5) / n,
        recall_at_10=sum(1 for r in results if r.hit_at_10) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        ndcg_at_5=sum(
            _ndcg(
                [1.0 if ft == r.expected_fuel_type else 0.0 for ft in r.matched_fuel_types],
                5,
            )
            for r in results
        ) / n,
        ndcg_at_10=sum(
            _ndcg(
                [1.0 if ft == r.expected_fuel_type else 0.0 for ft in r.matched_fuel_types],
                10,
            )
            for r in results
        ) / n,
        avg_latency_ms=sum(r.latency_ms for r in results) / n,
        misses_at_1=sum(1 for r in results if not r.hit_at_1),
        misses_at_5=sum(1 for r in results if not r.hit_at_5),
    )


class MatchEvaluator:
    """
    Runs gold-set evaluation against the matching pipeline.

    Supports A/B comparison between search modes.
    """

    def __init__(
        self,
        repo: FactorCatalogRepository,
        edition_id: str,
    ):
        self._repo = repo
        self._edition_id = edition_id

    def evaluate(
        self,
        gold_cases: List[Dict[str, Any]],
        *,
        mode: str = "lexical",
        semantic_search_fn: Optional[Callable] = None,
        hybrid_config: Optional[Any] = None,
        rerank_fn: Optional[Callable] = None,
        match_limit: int = 10,
    ) -> EvalReport:
        """
        Run evaluation on a set of gold cases.

        Args:
            gold_cases: List of gold case dicts
            mode: Label for this evaluation run
            semantic_search_fn: Optional semantic search callback
            hybrid_config: Optional HybridConfig
            rerank_fn: Optional LLM rerank callback
            match_limit: Number of results to retrieve per query

        Returns:
            EvalReport with metrics and per-case results.
        """
        from datetime import datetime

        cases = [GoldCase.from_dict(c) for c in gold_cases]
        case_results: List[CaseResult] = []

        for case in cases:
            start = time.monotonic()
            req = MatchRequest(
                activity_description=case.activity,
                geography=case.geography,
                scope=case.scope,
                limit=match_limit,
            )

            matches = run_match(
                self._repo,
                self._edition_id,
                req,
                semantic_search_fn=semantic_search_fn,
                hybrid_config=hybrid_config,
                rerank_fn=rerank_fn,
            )
            elapsed = (time.monotonic() - start) * 1000

            matched_ids = [m["factor_id"] for m in matches]
            matched_fuels = [_extract_fuel_type(fid) for fid in matched_ids]
            expected = case.expected_fuel_type

            case_results.append(CaseResult(
                case_id=case.id,
                expected_fuel_type=expected,
                matched_fuel_types=matched_fuels,
                matched_factor_ids=matched_ids,
                hit_at_1=len(matched_fuels) > 0 and matched_fuels[0] == expected,
                hit_at_3=expected in matched_fuels[:3],
                hit_at_5=expected in matched_fuels[:5],
                hit_at_10=expected in matched_fuels[:10],
                reciprocal_rank=_reciprocal_rank(matched_fuels, expected),
                domain=case.domain,
                difficulty=case.difficulty,
                latency_ms=elapsed,
            ))

        # Compute overall metrics
        overall = _compute_metrics(case_results)

        # Per-domain breakdown
        by_domain: Dict[str, EvalMetrics] = {}
        domains = set(r.domain for r in case_results if r.domain)
        for domain in domains:
            domain_results = [r for r in case_results if r.domain == domain]
            by_domain[domain] = _compute_metrics(domain_results)

        # Per-difficulty breakdown
        by_difficulty: Dict[str, EvalMetrics] = {}
        difficulties = set(r.difficulty for r in case_results)
        for diff in difficulties:
            diff_results = [r for r in case_results if r.difficulty == diff]
            by_difficulty[diff] = _compute_metrics(diff_results)

        report = EvalReport(
            mode=mode,
            overall=overall,
            by_domain=by_domain,
            by_difficulty=by_difficulty,
            case_results=case_results,
            timestamp=datetime.utcnow().isoformat(),
        )

        logger.info(
            "Evaluation complete: mode=%s cases=%d P@1=%.2f%% MRR=%.4f",
            mode, overall.total_cases,
            overall.precision_at_1 * 100, overall.mrr,
        )
        return report

    def ab_compare(
        self,
        gold_cases: List[Dict[str, Any]],
        configs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, EvalReport]:
        """
        A/B comparison: run evaluation with multiple configurations.

        Args:
            gold_cases: Gold eval cases
            configs: Dict of mode_name -> kwargs for evaluate()

        Returns:
            Dict of mode_name -> EvalReport
        """
        reports = {}
        for name, kwargs in configs.items():
            reports[name] = self.evaluate(gold_cases, mode=name, **kwargs)
        return reports


def load_gold_cases(path: str | Path) -> List[Dict[str, Any]]:
    """Load gold evaluation cases from JSON file."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))
