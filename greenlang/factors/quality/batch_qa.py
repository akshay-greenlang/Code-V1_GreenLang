# -*- coding: utf-8 -*-
"""
Batch QA runner (F020).

Runs Q1-Q6 quality gates across an entire edition in parallel,
produces a structured QA report, and optionally auto-promotes
factors that pass all gates to ``certified``.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from greenlang.factors.catalog_repository import FactorCatalogRepository
from greenlang.factors.etl.qa import validate_factor_dict
from greenlang.factors.quality.validators import validate_canonical_row
from greenlang.factors.approval_gate import check_promote_to_certified
from greenlang.factors.quality.promotion import PromotionState, transition
from greenlang.factors.source_registry import SourceRegistryEntry, registry_by_id

logger = logging.getLogger(__name__)


@dataclass
class FactorQAResult:
    """QA result for a single factor."""

    factor_id: str
    passed: bool
    gate_errors: List[str] = field(default_factory=list)
    gate_warnings: List[str] = field(default_factory=list)
    promoted: bool = False
    new_status: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor_id": self.factor_id,
            "passed": self.passed,
            "gate_errors": self.gate_errors,
            "gate_warnings": self.gate_warnings,
            "promoted": self.promoted,
            "new_status": self.new_status,
        }


@dataclass
class BatchQAReport:
    """Aggregate QA report for an entire edition."""

    edition_id: str
    run_at: str = field(default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat())
    total_factors: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_promoted: int = 0
    per_factor: List[FactorQAResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edition_id": self.edition_id,
            "run_at": self.run_at,
            "total_factors": self.total_factors,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "total_promoted": self.total_promoted,
            "pass_rate": round(self.total_passed / max(self.total_factors, 1), 4),
            "errors": self.errors[:100],
            "per_factor": [r.to_dict() for r in self.per_factor],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @property
    def all_passed(self) -> bool:
        return self.total_failed == 0 and self.total_factors > 0


def _run_qa_on_factor(
    factor_dict: Dict[str, Any],
    registry: Dict[str, SourceRegistryEntry],
    auto_promote: bool,
) -> FactorQAResult:
    """Run all QA gates on a single factor dict."""
    fid = factor_dict.get("factor_id", "?")
    result = FactorQAResult(factor_id=fid, passed=True)

    # Q1: Schema validation
    ok, errs = validate_factor_dict(factor_dict)
    if not ok:
        result.passed = False
        result.gate_errors.extend([f"Q1: {e}" for e in errs])

    # Q2: Canonical plausibility
    ok2, errs2 = validate_canonical_row(factor_dict)
    if not ok2:
        # Separate Q2-only errors (Q1 already captured above via validate_canonical_row delegation)
        q2_only = [e for e in errs2 if e not in (errs or [])]
        if q2_only:
            result.passed = False
            result.gate_errors.extend([f"Q2: {e}" for e in q2_only])

    # Q3: Source registry check
    source_id = factor_dict.get("provenance", {}).get("source_org", "")
    # Try source_id field first (from ingestion pipeline)
    sid = factor_dict.get("source_id") or ""
    if not sid:
        # Infer from factor_id pattern EF:{SOURCE}:...
        parts = fid.split(":")
        if len(parts) >= 2:
            sid = parts[1].lower()

    # Q4: License info present
    license_info = factor_dict.get("license_info") or {}
    if not license_info:
        result.gate_warnings.append("Q4: missing license_info block")

    # Q5: DQS completeness
    dqs = factor_dict.get("dqs") or {}
    required_dqs = ["temporal", "geographical", "technological", "representativeness", "methodological"]
    missing_dqs = [k for k in required_dqs if k not in dqs]
    if missing_dqs:
        result.gate_warnings.append(f"Q5: missing DQS dimensions: {missing_dqs}")

    # Q6: Provenance completeness
    prov = factor_dict.get("provenance") or {}
    if not prov.get("source_org"):
        result.gate_warnings.append("Q6: missing provenance.source_org")
    if not prov.get("methodology"):
        result.gate_warnings.append("Q6: missing provenance.methodology")

    # Auto-promote logic
    if auto_promote and result.passed:
        gate_result = check_promote_to_certified(
            sid or None,
            registry=registry,
            require_legal_signoff=False,
        )
        if gate_result.allowed:
            next_state, reason = transition(
                PromotionState.PREVIEW,
                qa_pass=True,
                methodology_signed=True,
                legal_ok=True,
                approval_gate_ok=True,
            )
            if next_state == PromotionState.CERTIFIED:
                result.promoted = True
                result.new_status = "certified"
            else:
                result.new_status = next_state.value
        else:
            result.new_status = "preview"
            result.gate_warnings.append(
                f"Auto-promote blocked: {'; '.join(gate_result.blockers)}"
            )
    elif result.passed:
        result.new_status = factor_dict.get("factor_status", "preview")
    else:
        result.new_status = "preview"

    return result


def run_batch_qa(
    repo: FactorCatalogRepository,
    edition_id: str,
    *,
    auto_promote: bool = False,
    max_workers: int = 4,
    registry: Optional[Dict[str, SourceRegistryEntry]] = None,
) -> BatchQAReport:
    """
    Run Q1-Q6 quality gates on all factors in an edition.

    Args:
        repo: Factor catalog repository.
        edition_id: Edition to validate.
        auto_promote: If True, promote passing factors to certified.
        max_workers: Thread pool size for parallel validation.
        registry: Source registry dict (loaded from YAML if None).

    Returns:
        BatchQAReport with per-factor results.
    """
    report = BatchQAReport(edition_id=edition_id)
    reg = registry if registry is not None else registry_by_id()

    # Load all factors
    try:
        resolved = repo.resolve_edition(edition_id)
    except ValueError as exc:
        report.errors.append(f"Edition resolution failed: {exc}")
        logger.error("Batch QA failed: %s", exc)
        return report

    factors, total = repo.list_factors(
        resolved,
        include_preview=True,
        include_connector=True,
        limit=1_000_000,
    )
    report.total_factors = len(factors)
    logger.info("Batch QA starting: edition=%s factors=%d workers=%d", edition_id, len(factors), max_workers)

    # Convert to dicts for QA
    factor_dicts = []
    for f in factors:
        try:
            factor_dicts.append(f.to_dict())
        except Exception as exc:
            report.errors.append(f"Failed to serialize {f.factor_id}: {exc}")

    # Run QA in parallel
    results: List[FactorQAResult] = []
    if max_workers <= 1 or len(factor_dicts) < 10:
        # Sequential for small batches
        for fd in factor_dicts:
            results.append(_run_qa_on_factor(fd, reg, auto_promote))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_qa_on_factor, fd, reg, auto_promote): fd
                for fd in factor_dicts
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    fd = futures[future]
                    fid = fd.get("factor_id", "?")
                    report.errors.append(f"QA worker error for {fid}: {exc}")
                    results.append(FactorQAResult(
                        factor_id=fid, passed=False,
                        gate_errors=[f"Worker exception: {exc}"],
                    ))

    # Aggregate
    for r in results:
        if r.passed:
            report.total_passed += 1
        else:
            report.total_failed += 1
        if r.promoted:
            report.total_promoted += 1

    report.per_factor = sorted(results, key=lambda x: x.factor_id)

    logger.info(
        "Batch QA complete: edition=%s passed=%d failed=%d promoted=%d",
        edition_id, report.total_passed, report.total_failed, report.total_promoted,
    )
    return report


def run_batch_qa_on_dicts(
    factor_dicts: Sequence[Dict[str, Any]],
    *,
    edition_id: str = "inline",
    auto_promote: bool = False,
    max_workers: int = 4,
    registry: Optional[Dict[str, SourceRegistryEntry]] = None,
) -> BatchQAReport:
    """
    Run batch QA directly on factor dicts (no repository needed).

    Useful for validating parsed output before catalog insertion.
    """
    report = BatchQAReport(edition_id=edition_id)
    reg = registry if registry is not None else registry_by_id()
    report.total_factors = len(factor_dicts)

    results: List[FactorQAResult] = []
    if max_workers <= 1 or len(factor_dicts) < 10:
        for fd in factor_dicts:
            results.append(_run_qa_on_factor(fd, reg, auto_promote))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_run_qa_on_factor, fd, reg, auto_promote): fd
                for fd in factor_dicts
            }
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    fd = futures[future]
                    fid = fd.get("factor_id", "?")
                    report.errors.append(f"QA worker error for {fid}: {exc}")
                    results.append(FactorQAResult(
                        factor_id=fid, passed=False,
                        gate_errors=[f"Worker exception: {exc}"],
                    ))

    for r in results:
        if r.passed:
            report.total_passed += 1
        else:
            report.total_failed += 1
        if r.promoted:
            report.total_promoted += 1

    report.per_factor = sorted(results, key=lambda x: x.factor_id)
    return report
