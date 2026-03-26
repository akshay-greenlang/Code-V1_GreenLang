# -*- coding: utf-8 -*-
"""
Consolidation Workflow
====================================

5-phase workflow for multi-site GHG emissions consolidation covering
site data gathering, intra-group elimination checks, equity/control
adjustments, reconciliation, and consolidated total generation within
PACK-049 GHG Multi-Site Management Pack.

Phases:
    1. SiteDataGather        -- Collect approved site-level emission totals
                                by scope and category.
    2. EliminationCheck      -- Identify and remove intra-group transactions
                                (inter-site energy transfers, fleet sharing).
    3. EquityAdjust          -- Apply equity share or control-based adjustments
                                to each site's emissions based on ownership %.
    4. Reconcile             -- Reconcile bottom-up site totals against top-down
                                corporate estimates, flag discrepancies.
    5. ConsolidatedTotal     -- Generate final consolidated totals by scope with
                                SHA-256 provenance.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3, 6) -- Consolidation
    ISO 14064-1:2018 (Cl. 5.2) -- Quantification
    CSRD / ESRS E1 -- Climate change consolidation

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsolidationPhase(str, Enum):
    SITE_DATA_GATHER = "site_data_gather"
    ELIMINATION_CHECK = "elimination_check"
    EQUITY_ADJUST = "equity_adjust"
    RECONCILE = "reconcile"
    CONSOLIDATED_TOTAL = "consolidated_total"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class EliminationType(str, Enum):
    INTER_SITE_ENERGY = "inter_site_energy"
    INTER_SITE_TRANSPORT = "inter_site_transport"
    SHARED_FLEET = "shared_fleet"
    INTRA_GROUP_PURCHASE = "intra_group_purchase"
    OTHER = "other"


class ReconciliationStatus(str, Enum):
    RECONCILED = "reconciled"
    MINOR_VARIANCE = "minor_variance"
    MAJOR_VARIANCE = "major_variance"
    NOT_RECONCILED = "not_reconciled"


# =============================================================================
# REFERENCE DATA
# =============================================================================

RECONCILIATION_THRESHOLDS = {
    "minor_variance_pct": Decimal("5.0"),
    "major_variance_pct": Decimal("10.0"),
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SiteEmissionTotal(BaseModel):
    """Approved emission total for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_tco2e: Decimal = Field(Decimal("0"))
    ownership_pct: Decimal = Field(Decimal("100.00"))
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )
    is_approved: bool = Field(True)
    reporting_period: str = Field("")


class EliminationEntry(BaseModel):
    """An intra-group elimination entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    elimination_id: str = Field(default_factory=_new_uuid)
    from_site_id: str = Field(...)
    to_site_id: str = Field(...)
    elimination_type: EliminationType = Field(EliminationType.OTHER)
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    eliminated_tco2e: Decimal = Field(Decimal("0"))
    description: str = Field("")
    evidence_ref: str = Field("")


class EquityAdjustment(BaseModel):
    """Equity/control adjustment for a single site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    original_total_tco2e: Decimal = Field(Decimal("0"))
    ownership_pct: Decimal = Field(Decimal("100.00"))
    reporting_pct: Decimal = Field(Decimal("100.00"))
    adjusted_scope_1: Decimal = Field(Decimal("0"))
    adjusted_scope_2_location: Decimal = Field(Decimal("0"))
    adjusted_scope_2_market: Decimal = Field(Decimal("0"))
    adjusted_scope_3: Decimal = Field(Decimal("0"))
    adjusted_total_tco2e: Decimal = Field(Decimal("0"))
    adjustment_method: str = Field("")


class ReconciliationRecord(BaseModel):
    """Reconciliation between bottom-up and top-down totals."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    scope: str = Field(...)
    bottom_up_tco2e: Decimal = Field(Decimal("0"))
    top_down_tco2e: Decimal = Field(Decimal("0"))
    variance_tco2e: Decimal = Field(Decimal("0"))
    variance_pct: Decimal = Field(Decimal("0"))
    status: ReconciliationStatus = Field(ReconciliationStatus.NOT_RECONCILED)
    explanation: str = Field("")


class ConsolidatedTotals(BaseModel):
    """Final consolidated emission totals."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    total_market_tco2e: Decimal = Field(Decimal("0"))
    eliminations_tco2e: Decimal = Field(Decimal("0"))
    sites_count: int = Field(0)
    provenance_hash: str = Field("")


class ConsolidationInput(BaseModel):
    """Input for the consolidation workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )
    site_totals: List[Dict[str, Any]] = Field(default_factory=list)
    eliminations: List[Dict[str, Any]] = Field(default_factory=list)
    top_down_estimates: Optional[Dict[str, Any]] = Field(None)
    skip_phases: List[str] = Field(default_factory=list)


class ConsolidationResult(BaseModel):
    """Output from the consolidation workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    consolidated_totals: Optional[ConsolidatedTotals] = Field(None)
    equity_adjustments: List[EquityAdjustment] = Field(default_factory=list)
    eliminations_applied: List[EliminationEntry] = Field(default_factory=list)
    reconciliation: List[ReconciliationRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class ConsolidationWorkflow:
    """
    5-phase consolidation workflow for multi-site GHG inventories.

    Gathers approved site totals, removes intra-group double-counting,
    applies equity/control adjustments, reconciles with top-down estimates,
    and produces final consolidated totals with SHA-256 provenance.

    Example:
        >>> wf = ConsolidationWorkflow()
        >>> inp = ConsolidationInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     site_totals=[{"site_id": "S1", "scope_1_tco2e": "1000"}],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[ConsolidationPhase] = [
        ConsolidationPhase.SITE_DATA_GATHER,
        ConsolidationPhase.ELIMINATION_CHECK,
        ConsolidationPhase.EQUITY_ADJUST,
        ConsolidationPhase.RECONCILE,
        ConsolidationPhase.CONSOLIDATED_TOTAL,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._site_totals: List[SiteEmissionTotal] = []
        self._eliminations: List[EliminationEntry] = []
        self._adjustments: List[EquityAdjustment] = []

    def execute(self, input_data: ConsolidationInput) -> ConsolidationResult:
        """Execute the full consolidation workflow."""
        start = _utcnow()
        result = ConsolidationResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            ConsolidationPhase.SITE_DATA_GATHER: self._phase_site_data_gather,
            ConsolidationPhase.ELIMINATION_CHECK: self._phase_elimination_check,
            ConsolidationPhase.EQUITY_ADJUST: self._phase_equity_adjust,
            ConsolidationPhase.RECONCILE: self._phase_reconcile,
            ConsolidationPhase.CONSOLIDATED_TOTAL: self._phase_consolidated_total,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue
            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed, errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED

        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- SITE DATA GATHER
    # -----------------------------------------------------------------

    def _phase_site_data_gather(
        self, input_data: ConsolidationInput, result: ConsolidationResult,
    ) -> Dict[str, Any]:
        """Collect approved site-level totals."""
        logger.info("Phase 1 -- Site Data Gather: %d sites", len(input_data.site_totals))
        totals: List[SiteEmissionTotal] = []

        for raw in input_data.site_totals:
            s1 = self._dec(raw.get("scope_1_tco2e", "0"))
            s2l = self._dec(raw.get("scope_2_location_tco2e", "0"))
            s2m = self._dec(raw.get("scope_2_market_tco2e", "0"))
            s3 = self._dec(raw.get("scope_3_tco2e", "0"))
            total = s1 + s2l + s3  # location-based total

            try:
                approach = ConsolidationApproach(
                    raw.get("consolidation_approach", input_data.consolidation_approach.value)
                )
            except ValueError:
                approach = input_data.consolidation_approach

            st = SiteEmissionTotal(
                site_id=raw.get("site_id", _new_uuid()),
                site_name=raw.get("site_name", ""),
                scope_1_tco2e=s1, scope_2_location_tco2e=s2l,
                scope_2_market_tco2e=s2m, scope_3_tco2e=s3,
                total_tco2e=total,
                ownership_pct=self._dec(raw.get("ownership_pct", "100")),
                consolidation_approach=approach,
                is_approved=raw.get("is_approved", True),
                reporting_period=raw.get("reporting_period", ""),
            )
            totals.append(st)

        approved = [t for t in totals if t.is_approved]
        unapproved = [t for t in totals if not t.is_approved]
        if unapproved:
            result.warnings.append(
                f"{len(unapproved)} site(s) not yet approved -- excluded from consolidation"
            )

        self._site_totals = approved
        grand_total = sum(t.total_tco2e for t in approved)

        logger.info("Gathered %d approved sites, total %.2f tCO2e",
                     len(approved), float(grand_total))
        return {
            "sites_gathered": len(totals),
            "approved": len(approved),
            "unapproved": len(unapproved),
            "pre_adjustment_total_tco2e": float(grand_total),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- ELIMINATION CHECK
    # -----------------------------------------------------------------

    def _phase_elimination_check(
        self, input_data: ConsolidationInput, result: ConsolidationResult,
    ) -> Dict[str, Any]:
        """Identify and account for intra-group eliminations."""
        logger.info("Phase 2 -- Elimination Check: %d raw", len(input_data.eliminations))
        entries: List[EliminationEntry] = []
        total_eliminated = Decimal("0")

        site_ids = {t.site_id for t in self._site_totals}

        for raw in input_data.eliminations:
            from_id = raw.get("from_site_id", "")
            to_id = raw.get("to_site_id", "")

            if from_id not in site_ids or to_id not in site_ids:
                result.warnings.append(
                    f"Elimination references unknown site: {from_id} -> {to_id}"
                )
                continue

            try:
                etype = EliminationType(raw.get("elimination_type", "other"))
            except ValueError:
                etype = EliminationType.OTHER

            try:
                scope = EmissionScope(raw.get("scope", "scope_1"))
            except ValueError:
                scope = EmissionScope.SCOPE_1

            amount = self._dec(raw.get("eliminated_tco2e", "0"))
            if amount <= Decimal("0"):
                continue

            entry = EliminationEntry(
                from_site_id=from_id, to_site_id=to_id,
                elimination_type=etype, scope=scope,
                eliminated_tco2e=amount,
                description=raw.get("description", ""),
                evidence_ref=raw.get("evidence_ref", ""),
            )
            entries.append(entry)
            total_eliminated += amount

        self._eliminations = entries
        result.eliminations_applied = entries

        logger.info("Eliminations: %d entries, %.2f tCO2e eliminated",
                     len(entries), float(total_eliminated))
        return {
            "eliminations_count": len(entries),
            "total_eliminated_tco2e": float(total_eliminated),
            "by_type": self._count_elim_by_type(entries),
        }

    def _count_elim_by_type(self, entries: List[EliminationEntry]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for e in entries:
            k = e.elimination_type.value
            counts[k] = counts.get(k, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # PHASE 3 -- EQUITY ADJUST
    # -----------------------------------------------------------------

    def _phase_equity_adjust(
        self, input_data: ConsolidationInput, result: ConsolidationResult,
    ) -> Dict[str, Any]:
        """Apply equity/control adjustments to site totals."""
        approach = input_data.consolidation_approach
        logger.info("Phase 3 -- Equity Adjust (%s)", approach.value)
        adjustments: List[EquityAdjustment] = []

        for st in self._site_totals:
            if approach == ConsolidationApproach.EQUITY_SHARE:
                factor = (st.ownership_pct / Decimal("100")).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
                method = f"equity_share @ {st.ownership_pct}%"
            else:
                factor = Decimal("1.0000")
                method = f"{approach.value} @ 100%"

            adj_s1 = (st.scope_1_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s2l = (st.scope_2_location_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s2m = (st.scope_2_market_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s3 = (st.scope_3_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_total = adj_s1 + adj_s2l + adj_s3

            adj = EquityAdjustment(
                site_id=st.site_id, site_name=st.site_name,
                original_total_tco2e=st.total_tco2e,
                ownership_pct=st.ownership_pct,
                reporting_pct=(factor * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                adjusted_scope_1=adj_s1, adjusted_scope_2_location=adj_s2l,
                adjusted_scope_2_market=adj_s2m, adjusted_scope_3=adj_s3,
                adjusted_total_tco2e=adj_total,
                adjustment_method=method,
            )
            adjustments.append(adj)

        self._adjustments = adjustments
        result.equity_adjustments = adjustments

        total_before = sum(st.total_tco2e for st in self._site_totals)
        total_after = sum(a.adjusted_total_tco2e for a in adjustments)
        adjustment_delta = total_before - total_after

        logger.info("Adjustments: %.2f -> %.2f tCO2e (delta %.2f)",
                     float(total_before), float(total_after), float(adjustment_delta))
        return {
            "sites_adjusted": len(adjustments),
            "total_before_tco2e": float(total_before),
            "total_after_tco2e": float(total_after),
            "adjustment_delta_tco2e": float(adjustment_delta),
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- RECONCILE
    # -----------------------------------------------------------------

    def _phase_reconcile(
        self, input_data: ConsolidationInput, result: ConsolidationResult,
    ) -> Dict[str, Any]:
        """Reconcile bottom-up vs top-down estimates."""
        logger.info("Phase 4 -- Reconcile")
        records: List[ReconciliationRecord] = []

        # Bottom-up from adjustments
        bu_s1 = sum(a.adjusted_scope_1 for a in self._adjustments)
        bu_s2l = sum(a.adjusted_scope_2_location for a in self._adjustments)
        bu_s2m = sum(a.adjusted_scope_2_market for a in self._adjustments)
        bu_s3 = sum(a.adjusted_scope_3 for a in self._adjustments)

        # Remove eliminations from bottom-up
        elim_by_scope: Dict[str, Decimal] = {}
        for e in self._eliminations:
            elim_by_scope[e.scope.value] = elim_by_scope.get(e.scope.value, Decimal("0")) + e.eliminated_tco2e

        bu_s1 -= elim_by_scope.get("scope_1", Decimal("0"))
        bu_s2l -= elim_by_scope.get("scope_2_location", Decimal("0"))
        bu_s3 -= elim_by_scope.get("scope_3", Decimal("0"))

        top_down = input_data.top_down_estimates or {}
        td_s1 = self._dec(top_down.get("scope_1_tco2e", "0"))
        td_s2l = self._dec(top_down.get("scope_2_location_tco2e", "0"))
        td_s2m = self._dec(top_down.get("scope_2_market_tco2e", "0"))
        td_s3 = self._dec(top_down.get("scope_3_tco2e", "0"))

        scopes = [
            ("scope_1", bu_s1, td_s1),
            ("scope_2_location", bu_s2l, td_s2l),
            ("scope_2_market", bu_s2m, td_s2m),
            ("scope_3", bu_s3, td_s3),
        ]

        for scope_name, bu_val, td_val in scopes:
            variance = bu_val - td_val
            variance_pct = Decimal("0")
            if td_val > Decimal("0"):
                variance_pct = (abs(variance) / td_val * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            if td_val == Decimal("0") and bu_val == Decimal("0"):
                status = ReconciliationStatus.RECONCILED
                explanation = "Both zero"
            elif td_val == Decimal("0"):
                status = ReconciliationStatus.NOT_RECONCILED
                explanation = "No top-down estimate provided"
            elif variance_pct <= RECONCILIATION_THRESHOLDS["minor_variance_pct"]:
                status = ReconciliationStatus.RECONCILED
                explanation = f"Variance {variance_pct}% within tolerance"
            elif variance_pct <= RECONCILIATION_THRESHOLDS["major_variance_pct"]:
                status = ReconciliationStatus.MINOR_VARIANCE
                explanation = f"Minor variance {variance_pct}% -- review recommended"
            else:
                status = ReconciliationStatus.MAJOR_VARIANCE
                explanation = f"Major variance {variance_pct}% -- investigation required"
                result.warnings.append(f"Major reconciliation variance for {scope_name}: {variance_pct}%")

            records.append(ReconciliationRecord(
                scope=scope_name, bottom_up_tco2e=bu_val.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                top_down_tco2e=td_val, variance_tco2e=variance.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                variance_pct=variance_pct, status=status, explanation=explanation,
            ))

        result.reconciliation = records
        reconciled_count = sum(1 for r in records if r.status == ReconciliationStatus.RECONCILED)

        logger.info("Reconciliation: %d/%d scopes reconciled", reconciled_count, len(records))
        return {
            "scopes_checked": len(records),
            "reconciled": reconciled_count,
            "minor_variance": sum(1 for r in records if r.status == ReconciliationStatus.MINOR_VARIANCE),
            "major_variance": sum(1 for r in records if r.status == ReconciliationStatus.MAJOR_VARIANCE),
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- CONSOLIDATED TOTAL
    # -----------------------------------------------------------------

    def _phase_consolidated_total(
        self, input_data: ConsolidationInput, result: ConsolidationResult,
    ) -> Dict[str, Any]:
        """Generate final consolidated totals."""
        logger.info("Phase 5 -- Consolidated Total")

        s1 = sum(a.adjusted_scope_1 for a in self._adjustments)
        s2l = sum(a.adjusted_scope_2_location for a in self._adjustments)
        s2m = sum(a.adjusted_scope_2_market for a in self._adjustments)
        s3 = sum(a.adjusted_scope_3 for a in self._adjustments)
        elim_total = sum(e.eliminated_tco2e for e in self._eliminations)

        # Subtract eliminations proportionally from scope_1 by default
        s1 = max(s1 - elim_total, Decimal("0"))

        total_loc = (s1 + s2l + s3).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_mkt = (s1 + s2m + s3).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        now_iso = _utcnow().isoformat()
        prov = _compute_hash(
            f"{input_data.organisation_id}|{input_data.reporting_year}|"
            f"{float(s1)}|{float(s2l)}|{float(s2m)}|{float(s3)}|{now_iso}"
        )

        ct = ConsolidatedTotals(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            approach=input_data.consolidation_approach,
            scope_1_tco2e=s1.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            scope_2_location_tco2e=s2l.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            scope_2_market_tco2e=s2m.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            scope_3_tco2e=s3.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_location_tco2e=total_loc,
            total_market_tco2e=total_mkt,
            eliminations_tco2e=elim_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            sites_count=len(self._site_totals),
            provenance_hash=prov,
        )
        result.consolidated_totals = ct

        logger.info(
            "Consolidated: S1=%.2f S2L=%.2f S2M=%.2f S3=%.2f Total(L)=%.2f tCO2e",
            float(s1), float(s2l), float(s2m), float(s3), float(total_loc),
        )
        return {
            "scope_1_tco2e": float(ct.scope_1_tco2e),
            "scope_2_location_tco2e": float(ct.scope_2_location_tco2e),
            "scope_2_market_tco2e": float(ct.scope_2_market_tco2e),
            "scope_3_tco2e": float(ct.scope_3_tco2e),
            "total_location_tco2e": float(ct.total_location_tco2e),
            "total_market_tco2e": float(ct.total_market_tco2e),
            "eliminations_tco2e": float(ct.eliminations_tco2e),
            "provenance_hash": prov,
        }

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        """Safely parse to Decimal."""
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ConsolidationWorkflow",
    "ConsolidationInput",
    "ConsolidationResult",
    "ConsolidationPhase",
    "EmissionScope",
    "ConsolidationApproach",
    "EliminationType",
    "ReconciliationStatus",
    "SiteEmissionTotal",
    "EliminationEntry",
    "EquityAdjustment",
    "ReconciliationRecord",
    "ConsolidatedTotals",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
