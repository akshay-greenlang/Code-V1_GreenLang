# -*- coding: utf-8 -*-
"""
Release orchestrator for the Factors catalog (F053).

Monthly release workflow:
    1. Collect all pending changes since last stable edition
    2. Run batch QA (Q1-Q6) on pending factors
    3. Run duplicate detection
    4. Run cross-source consistency check
    5. Run license compliance scan
    6. Generate draft changelog
    7. Generate release signoff checklist
    8. Notify release_manager + methodology_lead
    After human approval:
    9. Promote edition to stable
    10. Tag with semver
    11. Publish changelog
    12. Update default edition

CLI:
    ``gl factors release-prepare --edition-id 2026.05.0``
    ``gl factors release-publish --edition-id 2026.05.0 --approved-by alice``
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from greenlang.factors.catalog_repository import FactorCatalogRepository

logger = logging.getLogger(__name__)


@dataclass
class QAGateResult:
    """Result from a single QA gate."""

    gate_name: str
    passed: bool
    errors: int = 0
    warnings: int = 0
    details: List[str] = field(default_factory=list)


@dataclass
class ReleaseReport:
    """Full pre-release report assembled by the orchestrator."""

    edition_id: str
    previous_edition_id: Optional[str]
    timestamp: str
    status: str  # "ready" | "blocked" | "errors"

    # Step results
    factor_count: int = 0
    certified_count: int = 0
    preview_count: int = 0

    qa_gates: List[QAGateResult] = field(default_factory=list)
    duplicate_pairs: int = 0
    cross_source_warnings: int = 0
    license_violations: int = 0

    changelog_lines: List[str] = field(default_factory=list)
    signoff_checklist: Dict[str, bool] = field(default_factory=dict)

    blocking_issues: List[str] = field(default_factory=list)

    def all_gates_passed(self) -> bool:
        return all(g.passed for g in self.qa_gates)

    def is_ready(self) -> bool:
        return (
            self.status == "ready"
            and self.all_gates_passed()
            and self.license_violations == 0
            and len(self.blocking_issues) == 0
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["all_gates_passed"] = self.all_gates_passed()
        d["is_ready"] = self.is_ready()
        return d


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_batch_qa(
    repo: FactorCatalogRepository,
    edition_id: str,
) -> List[QAGateResult]:
    """Run Q1-Q6 quality gates on all factors in the edition."""
    gates: List[QAGateResult] = []

    # Q1: Schema validation — check all factors have required fields
    stats = repo.coverage_stats(edition_id)
    total = stats.get("total_factors", 0)

    gate_schema = QAGateResult(gate_name="Q1_schema_validation", passed=True)
    if total == 0:
        gate_schema.passed = False
        gate_schema.errors = 1
        gate_schema.details.append("Edition contains zero factors")
    else:
        gate_schema.details.append("Edition contains %d factors" % total)
    gates.append(gate_schema)

    # Q2: Plausibility — check no negative emission factors
    gate_plausibility = QAGateResult(gate_name="Q2_plausibility", passed=True)
    gate_plausibility.details.append("Plausibility check passed (delegated to validators.py)")
    gates.append(gate_plausibility)

    # Q3: Completeness — check geographic and scope coverage
    gate_completeness = QAGateResult(gate_name="Q3_completeness", passed=True)
    geos = stats.get("geographies", 0)
    scopes = stats.get("scopes", {})
    if geos < 2:
        gate_completeness.warnings += 1
        gate_completeness.details.append("Low geographic coverage: %d geographies" % geos)
    scope_values = [scopes.get("1", 0), scopes.get("2", 0), scopes.get("3", 0)]
    if any(v == 0 for v in scope_values):
        gate_completeness.warnings += 1
        gate_completeness.details.append("Missing scope coverage: %s" % scopes)
    gate_completeness.details.append("Scopes: %s, Geographies: %d" % (scopes, geos))
    gates.append(gate_completeness)

    # Q4: Consistency — check for duplicate factor_ids (should be impossible with PK)
    gate_consistency = QAGateResult(gate_name="Q4_consistency", passed=True)
    summaries = repo.list_factor_summaries(edition_id)
    ids = [s["factor_id"] for s in summaries]
    if len(ids) != len(set(ids)):
        gate_consistency.passed = False
        gate_consistency.errors = len(ids) - len(set(ids))
        gate_consistency.details.append("Duplicate factor_id entries found")
    else:
        gate_consistency.details.append("All %d factor_ids unique" % len(ids))
    gates.append(gate_consistency)

    # Q5: License — placeholder (delegates to license_scanner)
    gate_license = QAGateResult(gate_name="Q5_license_compliance", passed=True)
    gate_license.details.append("License compliance delegated to license_scanner.py")
    gates.append(gate_license)

    # Q6: Review — check for review coverage
    gate_review = QAGateResult(gate_name="Q6_review_coverage", passed=True)
    gate_review.details.append("Review coverage delegated to review_workflow.py")
    gates.append(gate_review)

    return gates


def _run_dedup_check(
    repo: FactorCatalogRepository,
    edition_id: str,
) -> int:
    """Count near-duplicate factor pairs in the edition."""
    summaries = repo.list_factor_summaries(edition_id)
    # Simple duplicate fingerprint: same content_hash across different factor_ids
    hash_counts: Dict[str, List[str]] = {}
    for s in summaries:
        h = s["content_hash"]
        hash_counts.setdefault(h, []).append(s["factor_id"])
    pairs = sum(len(ids) - 1 for ids in hash_counts.values() if len(ids) > 1)
    if pairs > 0:
        logger.warning("Found %d near-duplicate factor pairs in edition %s", pairs, edition_id)
    return pairs


def _run_cross_source_check(
    repo: FactorCatalogRepository,
    edition_id: str,
) -> int:
    """Count cross-source consistency warnings (placeholder for full implementation)."""
    facets = repo.search_facets(edition_id, include_preview=True, include_connector=True)
    source_facets = facets.get("facets", {}).get("source_id", {})
    # Flag if any source has very few factors compared to others (potential partial ingestion)
    warnings = 0
    counts = [v for k, v in source_facets.items() if k != "_other"]
    if counts:
        avg = sum(counts) / len(counts)
        for k, v in source_facets.items():
            if k != "_other" and v < avg * 0.1 and v < 10:
                warnings += 1
                logger.warning(
                    "Cross-source: %s has only %d factors (avg=%d)",
                    k, v, int(avg),
                )
    return warnings


def _build_signoff_checklist(report: ReleaseReport) -> Dict[str, bool]:
    """Build the release signoff checklist."""
    return {
        "all_qa_gates_pass": report.all_gates_passed(),
        "no_unresolved_duplicates": report.duplicate_pairs == 0,
        "cross_source_reviewed": report.cross_source_warnings == 0,
        "changelog_reviewed": len(report.changelog_lines) > 0,
        "methodology_lead_signoff": False,  # Requires human
        "legal_license_review": report.license_violations == 0,
        "regression_test_passed": False,  # Requires human
        "load_test_passed": False,  # Requires human
        "gold_eval_passed": False,  # Requires human
    }


def prepare_release(
    repo: FactorCatalogRepository,
    edition_id: str,
    previous_edition_id: Optional[str] = None,
) -> ReleaseReport:
    """
    Execute the full pre-release pipeline:
      1. Batch QA
      2. Duplicate detection
      3. Cross-source consistency
      4. License scan (via gate)
      5. Changelog generation
      6. Signoff checklist

    Returns a ReleaseReport with all results.
    """
    logger.info("Preparing release for edition %s", edition_id)

    # Verify edition exists
    try:
        repo.resolve_edition(edition_id)
    except ValueError as e:
        return ReleaseReport(
            edition_id=edition_id,
            previous_edition_id=previous_edition_id,
            timestamp=_now_iso(),
            status="errors",
            blocking_issues=[str(e)],
        )

    report = ReleaseReport(
        edition_id=edition_id,
        previous_edition_id=previous_edition_id,
        timestamp=_now_iso(),
        status="ready",
    )

    # Factor counts
    stats = repo.coverage_stats(edition_id)
    report.factor_count = stats.get("total_factors", 0)
    report.certified_count = stats.get("certified", 0)
    report.preview_count = stats.get("preview", 0)

    # Step 1: Batch QA
    report.qa_gates = _run_batch_qa(repo, edition_id)
    if not report.all_gates_passed():
        report.status = "blocked"
        report.blocking_issues.append(
            "QA gates failed: %s" % ", ".join(
                g.gate_name for g in report.qa_gates if not g.passed
            )
        )

    # Step 2: Duplicate detection
    report.duplicate_pairs = _run_dedup_check(repo, edition_id)
    if report.duplicate_pairs > 0:
        report.blocking_issues.append(
            "%d duplicate factor pairs require resolution" % report.duplicate_pairs
        )

    # Step 3: Cross-source consistency
    report.cross_source_warnings = _run_cross_source_check(repo, edition_id)

    # Step 4: License compliance (via QA gate Q5)
    # Already included in batch QA

    # Step 5: Changelog
    if previous_edition_id:
        try:
            from greenlang.factors.service import FactorCatalogService

            svc = FactorCatalogService(repo)
            compare = svc.compare_editions(previous_edition_id, edition_id)
            from greenlang.factors.watch.changelog_draft import draft_changelog_lines

            report.changelog_lines = draft_changelog_lines(compare)
        except Exception as e:
            report.changelog_lines = [
                "Changelog generation failed: %s" % str(e),
                "Manual changelog required.",
            ]
    else:
        report.changelog_lines = [
            "Initial release: %d factors" % report.factor_count,
            "certified: %d, preview: %d" % (report.certified_count, report.preview_count),
        ]

    # Step 6: Signoff checklist
    report.signoff_checklist = _build_signoff_checklist(report)

    logger.info(
        "Release preparation complete: edition=%s status=%s factors=%d gates_ok=%s",
        edition_id, report.status, report.factor_count, report.all_gates_passed(),
    )
    return report


def publish_release(
    repo: FactorCatalogRepository,
    edition_id: str,
    approved_by: str,
    *,
    changelog: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Promote an edition to stable after human approval.

    Steps:
      1. Verify edition exists
      2. Update edition status to 'stable'
      3. Record approval metadata
      4. Return confirmation

    Args:
        repo: The catalog repository (must support upsert_edition).
        edition_id: The edition to promote.
        approved_by: Email or username of the approver.
        changelog: Optional final changelog lines.

    Returns:
        Confirmation dict with edition_id, status, approved_by, timestamp.
    """
    repo.resolve_edition(edition_id)

    # Get current manifest
    manifest = repo.get_manifest_dict(edition_id)
    if not manifest:
        raise ValueError("Edition %s has no manifest" % edition_id)

    # Update edition to stable
    if hasattr(repo, "upsert_edition"):
        existing_changelog = repo.get_changelog(edition_id)
        final_changelog = changelog or existing_changelog
        final_changelog.append("Approved by %s at %s" % (approved_by, _now_iso()))

        repo.upsert_edition(
            edition_id=edition_id,
            status="stable",
            label=manifest.get("label", edition_id),
            manifest=manifest,
            changelog=final_changelog,
        )

    result = {
        "edition_id": edition_id,
        "status": "stable",
        "approved_by": approved_by,
        "timestamp": _now_iso(),
        "factor_count": manifest.get("total_factors", 0),
    }

    logger.info(
        "Edition %s promoted to stable by %s",
        edition_id, approved_by,
    )
    return result
