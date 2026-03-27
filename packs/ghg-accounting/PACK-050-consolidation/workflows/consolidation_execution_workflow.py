# -*- coding: utf-8 -*-
"""
Consolidation Execution Workflow
====================================

6-phase workflow for executing multi-entity GHG emissions consolidation
per GHG Protocol Corporate Standard Chapter 3 and Chapter 6 within
PACK-050 GHG Consolidation Pack.

Phases:
    1. DataGathering              -- Gather validated entity-level GHG data
                                     from all entities in the boundary.
    2. EquityAdjustment           -- Apply equity share adjustments per
                                     entity ownership percentage.
    3. ControlAdjustment          -- Apply operational/financial control
                                     inclusion logic (100%/0%).
    4. IntercompanyElimination    -- Run intercompany elimination to remove
                                     double-counting of intra-group transfers.
    5. AdjustmentApplication      -- Apply manual adjustments including
                                     corrections and reclassifications.
    6. ConsolidatedTotal          -- Calculate final consolidated totals
                                     with full reconciliation trail.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3, 6) -- Consolidation
    ISO 14064-1:2018 (Cl. 5.2) -- Quantification at organisation level
    CSRD / ESRS E1 -- Climate change consolidation
    IFRS S2 -- Climate-related financial disclosures

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

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


class ConsolidationExecPhase(str, Enum):
    DATA_GATHERING = "data_gathering"
    EQUITY_ADJUSTMENT = "equity_adjustment"
    CONTROL_ADJUSTMENT = "control_adjustment"
    INTERCOMPANY_ELIMINATION = "intercompany_elimination"
    ADJUSTMENT_APPLICATION = "adjustment_application"
    CONSOLIDATED_TOTAL = "consolidated_total"


class ConsolidationApproach(str, Enum):
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class AdjustmentType(str, Enum):
    CORRECTION = "correction"
    RECLASSIFICATION = "reclassification"
    RESTATEMENT = "restatement"
    METHODOLOGY_CHANGE = "methodology_change"
    DATA_IMPROVEMENT = "data_improvement"


class EliminationType(str, Enum):
    INTER_ENTITY_ENERGY = "inter_entity_energy"
    INTER_ENTITY_TRANSPORT = "inter_entity_transport"
    INTER_ENTITY_PRODUCT = "inter_entity_product"
    INTER_ENTITY_WASTE = "inter_entity_waste"
    SHARED_SERVICE = "shared_service"
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


class EntityEmissionRecord(BaseModel):
    """Validated emission record for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(...)
    entity_name: str = Field("")
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_location_tco2e: Decimal = Field(Decimal("0"))
    scope_2_market_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    total_location_tco2e: Decimal = Field(Decimal("0"))
    ownership_pct: Decimal = Field(Decimal("100.00"))
    has_operational_control: bool = Field(True)
    has_financial_control: bool = Field(True)
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )


class EquityAdjustmentRecord(BaseModel):
    """Equity share adjustment for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(...)
    entity_name: str = Field("")
    original_total_tco2e: Decimal = Field(Decimal("0"))
    ownership_pct: Decimal = Field(Decimal("100.00"))
    equity_factor: Decimal = Field(Decimal("1.0000"))
    adjusted_scope_1: Decimal = Field(Decimal("0"))
    adjusted_scope_2_location: Decimal = Field(Decimal("0"))
    adjusted_scope_2_market: Decimal = Field(Decimal("0"))
    adjusted_scope_3: Decimal = Field(Decimal("0"))
    adjusted_total_tco2e: Decimal = Field(Decimal("0"))


class ControlAdjustmentRecord(BaseModel):
    """Control-based adjustment for a single entity."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    entity_id: str = Field(...)
    entity_name: str = Field("")
    approach: ConsolidationApproach = Field(ConsolidationApproach.OPERATIONAL_CONTROL)
    has_control: bool = Field(True)
    inclusion_factor: Decimal = Field(Decimal("1.0"))
    pre_adjustment_tco2e: Decimal = Field(Decimal("0"))
    post_adjustment_tco2e: Decimal = Field(Decimal("0"))


class EliminationEntry(BaseModel):
    """Intercompany elimination entry."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elimination_id: str = Field(default_factory=_new_uuid)
    seller_entity_id: str = Field(...)
    buyer_entity_id: str = Field(...)
    elimination_type: EliminationType = Field(EliminationType.OTHER)
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    eliminated_tco2e: Decimal = Field(Decimal("0"))
    description: str = Field("")
    evidence_ref: str = Field("")


class ManualAdjustment(BaseModel):
    """Manual adjustment record."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjustment_id: str = Field(default_factory=_new_uuid)
    entity_id: str = Field("")
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    adjustment_type: AdjustmentType = Field(AdjustmentType.CORRECTION)
    amount_tco2e: Decimal = Field(Decimal("0"))
    reason: str = Field("")
    approved_by: str = Field("")
    applied_at: str = Field("")


class ConsolidatedTotal(BaseModel):
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
    manual_adjustments_tco2e: Decimal = Field(Decimal("0"))
    entities_count: int = Field(0)
    reconciliation_status: ReconciliationStatus = Field(ReconciliationStatus.NOT_RECONCILED)
    provenance_hash: str = Field("")


class ConsolidationExecInput(BaseModel):
    """Input for the consolidation execution workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL
    )
    entity_emissions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Entity-level emission data"
    )
    intercompany_transfers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Intra-group transfer records"
    )
    manual_adjustments: List[Dict[str, Any]] = Field(
        default_factory=list, description="Manual adjustments to apply"
    )
    top_down_estimates: Optional[Dict[str, Any]] = Field(None)
    skip_phases: List[str] = Field(default_factory=list)


class ConsolidationExecResult(BaseModel):
    """Output from the consolidation execution workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    consolidated_total: Optional[ConsolidatedTotal] = Field(None)
    equity_adjustments: List[EquityAdjustmentRecord] = Field(default_factory=list)
    control_adjustments: List[ControlAdjustmentRecord] = Field(default_factory=list)
    eliminations: List[EliminationEntry] = Field(default_factory=list)
    manual_adjustments_applied: List[ManualAdjustment] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class ConsolidationExecutionWorkflow:
    """
    6-phase consolidation execution workflow for multi-entity GHG.

    Gathers entity data, applies equity and control adjustments,
    eliminates intercompany transfers, applies manual adjustments,
    and produces consolidated totals with SHA-256 provenance.

    Example:
        >>> wf = ConsolidationExecutionWorkflow()
        >>> inp = ConsolidationExecInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     entity_emissions=[{"entity_id": "E1", "scope_1_tco2e": "1000"}],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.consolidated_total is not None
    """

    PHASE_ORDER: List[ConsolidationExecPhase] = [
        ConsolidationExecPhase.DATA_GATHERING,
        ConsolidationExecPhase.EQUITY_ADJUSTMENT,
        ConsolidationExecPhase.CONTROL_ADJUSTMENT,
        ConsolidationExecPhase.INTERCOMPANY_ELIMINATION,
        ConsolidationExecPhase.ADJUSTMENT_APPLICATION,
        ConsolidationExecPhase.CONSOLIDATED_TOTAL,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._entities: List[EntityEmissionRecord] = []
        self._equity_adjusted: Dict[str, EquityAdjustmentRecord] = {}
        self._control_adjusted: Dict[str, ControlAdjustmentRecord] = {}
        self._eliminations: List[EliminationEntry] = []
        self._manual_adjs: List[ManualAdjustment] = []

    def execute(self, input_data: ConsolidationExecInput) -> ConsolidationExecResult:
        """Execute the full 6-phase consolidation execution workflow."""
        start = _utcnow()
        result = ConsolidationExecResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            ConsolidationExecPhase.DATA_GATHERING: self._phase_data_gathering,
            ConsolidationExecPhase.EQUITY_ADJUSTMENT: self._phase_equity_adjustment,
            ConsolidationExecPhase.CONTROL_ADJUSTMENT: self._phase_control_adjustment,
            ConsolidationExecPhase.INTERCOMPANY_ELIMINATION: self._phase_intercompany_elimination,
            ConsolidationExecPhase.ADJUSTMENT_APPLICATION: self._phase_adjustment_application,
            ConsolidationExecPhase.CONSOLIDATED_TOTAL: self._phase_consolidated_total,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.SKIPPED,
                ))
                continue

            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                ph_hash = _compute_hash(str(phase_out))
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=ph_hash,
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed,
                    errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value} failed: {exc}")
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
    # PHASE 1 -- DATA GATHERING
    # -----------------------------------------------------------------

    def _phase_data_gathering(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Gather validated entity-level GHG data."""
        logger.info("Phase 1 -- Data Gathering: %d entities", len(input_data.entity_emissions))
        entities: List[EntityEmissionRecord] = []

        for raw in input_data.entity_emissions:
            s1 = self._dec(raw.get("scope_1_tco2e", "0"))
            s2l = self._dec(raw.get("scope_2_location_tco2e", "0"))
            s2m = self._dec(raw.get("scope_2_market_tco2e", "0"))
            s3 = self._dec(raw.get("scope_3_tco2e", "0"))
            total_loc = s1 + s2l + s3

            try:
                approach = ConsolidationApproach(
                    raw.get("consolidation_approach", input_data.consolidation_approach.value)
                )
            except ValueError:
                approach = input_data.consolidation_approach

            record = EntityEmissionRecord(
                entity_id=raw.get("entity_id", _new_uuid()),
                entity_name=raw.get("entity_name", ""),
                scope_1_tco2e=s1,
                scope_2_location_tco2e=s2l,
                scope_2_market_tco2e=s2m,
                scope_3_tco2e=s3,
                total_location_tco2e=total_loc,
                ownership_pct=self._dec(raw.get("ownership_pct", "100")),
                has_operational_control=bool(raw.get("has_operational_control", True)),
                has_financial_control=bool(raw.get("has_financial_control", True)),
                consolidation_approach=approach,
            )
            entities.append(record)

        self._entities = entities
        grand_total = sum(e.total_location_tco2e for e in entities)

        logger.info("Gathered %d entities, pre-adjustment total %.2f tCO2e",
                     len(entities), float(grand_total))
        return {
            "entities_gathered": len(entities),
            "pre_adjustment_total_tco2e": float(grand_total),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- EQUITY ADJUSTMENT
    # -----------------------------------------------------------------

    def _phase_equity_adjustment(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Apply equity share adjustments per entity ownership percentage."""
        logger.info("Phase 2 -- Equity Adjustment (%s)", input_data.consolidation_approach.value)
        adjustments: Dict[str, EquityAdjustmentRecord] = {}

        for entity in self._entities:
            if input_data.consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
                factor = (entity.ownership_pct / Decimal("100")).quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                )
            else:
                factor = Decimal("1.0000")

            adj_s1 = (entity.scope_1_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s2l = (entity.scope_2_location_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s2m = (entity.scope_2_market_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_s3 = (entity.scope_3_tco2e * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            adj_total = adj_s1 + adj_s2l + adj_s3

            adj = EquityAdjustmentRecord(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                original_total_tco2e=entity.total_location_tco2e,
                ownership_pct=entity.ownership_pct,
                equity_factor=factor,
                adjusted_scope_1=adj_s1,
                adjusted_scope_2_location=adj_s2l,
                adjusted_scope_2_market=adj_s2m,
                adjusted_scope_3=adj_s3,
                adjusted_total_tco2e=adj_total,
            )
            adjustments[entity.entity_id] = adj

        self._equity_adjusted = adjustments
        result.equity_adjustments = list(adjustments.values())

        total_before = sum(e.total_location_tco2e for e in self._entities)
        total_after = sum(a.adjusted_total_tco2e for a in adjustments.values())

        logger.info("Equity adjustment: %.2f -> %.2f tCO2e", float(total_before), float(total_after))
        return {
            "entities_adjusted": len(adjustments),
            "total_before_tco2e": float(total_before),
            "total_after_tco2e": float(total_after),
            "adjustment_delta_tco2e": float(total_before - total_after),
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- CONTROL ADJUSTMENT
    # -----------------------------------------------------------------

    def _phase_control_adjustment(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Apply operational/financial control inclusion logic (100%/0%)."""
        approach = input_data.consolidation_approach
        logger.info("Phase 3 -- Control Adjustment (%s)", approach.value)
        adjustments: Dict[str, ControlAdjustmentRecord] = {}
        included = 0
        excluded = 0

        for entity in self._entities:
            equity_adj = self._equity_adjusted.get(entity.entity_id)
            pre_total = equity_adj.adjusted_total_tco2e if equity_adj else entity.total_location_tco2e

            has_control = True
            if approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                has_control = entity.has_operational_control
            elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
                has_control = entity.has_financial_control
            # Equity share: all entities included proportionally (already handled)

            inclusion_factor = Decimal("1.0") if has_control else Decimal("0")
            post_total = (pre_total * inclusion_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            if has_control:
                included += 1
            else:
                excluded += 1

            adj = ControlAdjustmentRecord(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                approach=approach,
                has_control=has_control,
                inclusion_factor=inclusion_factor,
                pre_adjustment_tco2e=pre_total,
                post_adjustment_tco2e=post_total,
            )
            adjustments[entity.entity_id] = adj

        self._control_adjusted = adjustments
        result.control_adjustments = list(adjustments.values())

        logger.info("Control adjustment: %d included, %d excluded", included, excluded)
        return {
            "entities_included": included,
            "entities_excluded": excluded,
            "approach": approach.value,
        }

    # -----------------------------------------------------------------
    # PHASE 4 -- INTERCOMPANY ELIMINATION
    # -----------------------------------------------------------------

    def _phase_intercompany_elimination(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Run intercompany elimination to remove double-counting."""
        logger.info("Phase 4 -- Intercompany Elimination: %d transfers",
                     len(input_data.intercompany_transfers))

        entity_ids = {e.entity_id for e in self._entities}
        entries: List[EliminationEntry] = []
        total_eliminated = Decimal("0")

        for raw in input_data.intercompany_transfers:
            seller_id = raw.get("seller_entity_id", "")
            buyer_id = raw.get("buyer_entity_id", "")

            if seller_id not in entity_ids or buyer_id not in entity_ids:
                result.warnings.append(
                    f"Elimination references unknown entity: {seller_id} -> {buyer_id}"
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
                seller_entity_id=seller_id,
                buyer_entity_id=buyer_id,
                elimination_type=etype,
                scope=scope,
                eliminated_tco2e=amount,
                description=raw.get("description", ""),
                evidence_ref=raw.get("evidence_ref", ""),
            )
            entries.append(entry)
            total_eliminated += amount

        self._eliminations = entries
        result.eliminations = entries

        logger.info("Eliminations: %d entries, %.2f tCO2e", len(entries), float(total_eliminated))
        return {
            "eliminations_count": len(entries),
            "total_eliminated_tco2e": float(total_eliminated),
            "by_type": self._count_by_key(entries, "elimination_type"),
        }

    def _count_by_key(self, entries: List[EliminationEntry], key: str) -> Dict[str, int]:
        """Count entries by a given attribute."""
        counts: Dict[str, int] = {}
        for e in entries:
            val = getattr(e, key, "other")
            k = val.value if hasattr(val, "value") else str(val)
            counts[k] = counts.get(k, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # PHASE 5 -- ADJUSTMENT APPLICATION
    # -----------------------------------------------------------------

    def _phase_adjustment_application(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Apply manual adjustments (corrections, reclassifications)."""
        logger.info("Phase 5 -- Adjustment Application: %d adjustments",
                     len(input_data.manual_adjustments))

        applied: List[ManualAdjustment] = []
        total_adjustment = Decimal("0")
        now_iso = _utcnow().isoformat()

        for raw in input_data.manual_adjustments:
            try:
                adj_type = AdjustmentType(raw.get("adjustment_type", "correction"))
            except ValueError:
                adj_type = AdjustmentType.CORRECTION

            try:
                scope = EmissionScope(raw.get("scope", "scope_1"))
            except ValueError:
                scope = EmissionScope.SCOPE_1

            amount = self._dec(raw.get("amount_tco2e", "0"))

            adj = ManualAdjustment(
                entity_id=raw.get("entity_id", ""),
                scope=scope,
                adjustment_type=adj_type,
                amount_tco2e=amount,
                reason=raw.get("reason", ""),
                approved_by=raw.get("approved_by", ""),
                applied_at=now_iso,
            )
            applied.append(adj)
            total_adjustment += amount

        self._manual_adjs = applied
        result.manual_adjustments_applied = applied

        logger.info("Manual adjustments: %d applied, net %.2f tCO2e",
                     len(applied), float(total_adjustment))
        return {
            "adjustments_applied": len(applied),
            "net_adjustment_tco2e": float(total_adjustment),
            "by_type": self._count_adj_types(applied),
        }

    def _count_adj_types(self, adjs: List[ManualAdjustment]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in adjs:
            k = a.adjustment_type.value
            counts[k] = counts.get(k, 0) + 1
        return counts

    # -----------------------------------------------------------------
    # PHASE 6 -- CONSOLIDATED TOTAL
    # -----------------------------------------------------------------

    def _phase_consolidated_total(
        self, input_data: ConsolidationExecInput, result: ConsolidationExecResult,
    ) -> Dict[str, Any]:
        """Calculate final consolidated totals with reconciliation."""
        logger.info("Phase 6 -- Consolidated Total")

        # Sum up control-adjusted values for included entities
        s1 = Decimal("0")
        s2l = Decimal("0")
        s2m = Decimal("0")
        s3 = Decimal("0")
        entity_count = 0

        for entity in self._entities:
            ctrl = self._control_adjusted.get(entity.entity_id)
            equity = self._equity_adjusted.get(entity.entity_id)

            if ctrl and not ctrl.has_control:
                continue

            entity_count += 1
            if equity:
                s1 += equity.adjusted_scope_1
                s2l += equity.adjusted_scope_2_location
                s2m += equity.adjusted_scope_2_market
                s3 += equity.adjusted_scope_3
            else:
                s1 += entity.scope_1_tco2e
                s2l += entity.scope_2_location_tco2e
                s2m += entity.scope_2_market_tco2e
                s3 += entity.scope_3_tco2e

        # Subtract eliminations by scope
        elim_by_scope: Dict[str, Decimal] = {}
        for e in self._eliminations:
            elim_by_scope[e.scope.value] = elim_by_scope.get(e.scope.value, Decimal("0")) + e.eliminated_tco2e

        s1 -= elim_by_scope.get("scope_1", Decimal("0"))
        s2l -= elim_by_scope.get("scope_2_location", Decimal("0"))
        s2m -= elim_by_scope.get("scope_2_market", Decimal("0"))
        s3 -= elim_by_scope.get("scope_3", Decimal("0"))

        # Apply manual adjustments by scope
        adj_by_scope: Dict[str, Decimal] = {}
        for a in self._manual_adjs:
            adj_by_scope[a.scope.value] = adj_by_scope.get(a.scope.value, Decimal("0")) + a.amount_tco2e

        s1 += adj_by_scope.get("scope_1", Decimal("0"))
        s2l += adj_by_scope.get("scope_2_location", Decimal("0"))
        s2m += adj_by_scope.get("scope_2_market", Decimal("0"))
        s3 += adj_by_scope.get("scope_3", Decimal("0"))

        # Ensure non-negative
        s1 = max(s1, Decimal("0"))
        s2l = max(s2l, Decimal("0"))
        s2m = max(s2m, Decimal("0"))
        s3 = max(s3, Decimal("0"))

        total_loc = (s1 + s2l + s3).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_mkt = (s1 + s2m + s3).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        elim_total = sum(e.eliminated_tco2e for e in self._eliminations)
        adj_total = sum(a.amount_tco2e for a in self._manual_adjs)

        # Reconciliation
        recon_status = self._reconcile_with_top_down(input_data, s1, s2l, s3, result)

        now_iso = _utcnow().isoformat()
        prov = _compute_hash(
            f"{input_data.organisation_id}|{input_data.reporting_year}|"
            f"{float(s1)}|{float(s2l)}|{float(s2m)}|{float(s3)}|{now_iso}"
        )

        ct = ConsolidatedTotal(
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
            manual_adjustments_tco2e=adj_total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            entities_count=entity_count,
            reconciliation_status=recon_status,
            provenance_hash=prov,
        )
        result.consolidated_total = ct

        logger.info(
            "Consolidated: S1=%.2f S2L=%.2f S2M=%.2f S3=%.2f Total(L)=%.2f",
            float(s1), float(s2l), float(s2m), float(s3), float(total_loc),
        )
        return {
            "scope_1_tco2e": float(ct.scope_1_tco2e),
            "scope_2_location_tco2e": float(ct.scope_2_location_tco2e),
            "scope_2_market_tco2e": float(ct.scope_2_market_tco2e),
            "scope_3_tco2e": float(ct.scope_3_tco2e),
            "total_location_tco2e": float(ct.total_location_tco2e),
            "total_market_tco2e": float(ct.total_market_tco2e),
            "entities_count": entity_count,
            "provenance_hash": prov,
        }

    def _reconcile_with_top_down(
        self, input_data: ConsolidationExecInput,
        s1: Decimal, s2l: Decimal, s3: Decimal,
        result: ConsolidationExecResult,
    ) -> ReconciliationStatus:
        """Reconcile consolidated totals with top-down estimates."""
        top_down = input_data.top_down_estimates
        if not top_down:
            return ReconciliationStatus.NOT_RECONCILED

        td_total = self._dec(top_down.get("total_tco2e", "0"))
        bu_total = s1 + s2l + s3

        if td_total == Decimal("0"):
            return ReconciliationStatus.NOT_RECONCILED

        variance_pct = (abs(bu_total - td_total) / td_total * Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if variance_pct <= RECONCILIATION_THRESHOLDS["minor_variance_pct"]:
            return ReconciliationStatus.RECONCILED
        elif variance_pct <= RECONCILIATION_THRESHOLDS["major_variance_pct"]:
            result.warnings.append(f"Minor reconciliation variance: {variance_pct}%")
            return ReconciliationStatus.MINOR_VARIANCE
        else:
            result.warnings.append(f"Major reconciliation variance: {variance_pct}%")
            return ReconciliationStatus.MAJOR_VARIANCE

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
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
    "ConsolidationExecutionWorkflow",
    "ConsolidationExecInput",
    "ConsolidationExecResult",
    "ConsolidationExecPhase",
    "ConsolidationApproach",
    "EmissionScope",
    "AdjustmentType",
    "EliminationType",
    "ReconciliationStatus",
    "EntityEmissionRecord",
    "EquityAdjustmentRecord",
    "ControlAdjustmentRecord",
    "EliminationEntry",
    "ManualAdjustment",
    "ConsolidatedTotal",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
