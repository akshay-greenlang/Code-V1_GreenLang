# -*- coding: utf-8 -*-
"""
Elimination Workflow
====================================

4-phase workflow for identifying and eliminating intra-group transfers
to prevent double-counting in GHG consolidation within PACK-050
GHG Consolidation Pack.

Phases:
    1. TransferIdentification    -- Identify all intra-group transfers
                                    (energy, waste, products, services)
                                    across entities in the consolidation
                                    boundary.
    2. MatchingVerification      -- Match seller records with buyer records
                                    to verify transfer pairs and amounts.
    3. EliminationCalculation    -- Calculate elimination amounts per
                                    transfer pair and emission scope.
    4. ReconciliationCheck       -- Verify eliminations balance, reconcile
                                    net impact, and generate audit trail.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 3, 6) -- Consolidation & double counting
    ISO 14064-1:2018 (Cl. 5.2) -- Avoidance of double counting
    CSRD / ESRS E1 -- Climate change (intra-group transactions)

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
from typing import Any, Dict, List, Optional, Tuple

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


class EliminationPhase(str, Enum):
    TRANSFER_IDENTIFICATION = "transfer_identification"
    MATCHING_VERIFICATION = "matching_verification"
    ELIMINATION_CALCULATION = "elimination_calculation"
    RECONCILIATION_CHECK = "reconciliation_check"


class TransferType(str, Enum):
    ENERGY_ELECTRICITY = "energy_electricity"
    ENERGY_STEAM = "energy_steam"
    ENERGY_HEAT = "energy_heat"
    ENERGY_COOLING = "energy_cooling"
    WASTE_TREATMENT = "waste_treatment"
    PRODUCT_INTERMEDIATE = "product_intermediate"
    PRODUCT_FINISHED = "product_finished"
    TRANSPORT_SERVICE = "transport_service"
    SHARED_FACILITY = "shared_facility"
    OTHER = "other"


class MatchStatus(str, Enum):
    MATCHED = "matched"
    PARTIAL_MATCH = "partial_match"
    UNMATCHED = "unmatched"
    DISPUTED = "disputed"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class ReconciliationVerdict(str, Enum):
    BALANCED = "balanced"
    MINOR_IMBALANCE = "minor_imbalance"
    MAJOR_IMBALANCE = "major_imbalance"
    FAILED = "failed"


# =============================================================================
# REFERENCE DATA
# =============================================================================

MATCHING_TOLERANCE_PCT = Decimal("5.0")
RECONCILIATION_TOLERANCE_PCT = Decimal("2.0")

TRANSFER_SCOPE_MAPPING: Dict[str, str] = {
    "energy_electricity": "scope_2_location",
    "energy_steam": "scope_2_location",
    "energy_heat": "scope_2_location",
    "energy_cooling": "scope_2_location",
    "waste_treatment": "scope_1",
    "product_intermediate": "scope_3",
    "product_finished": "scope_3",
    "transport_service": "scope_3",
    "shared_facility": "scope_1",
    "other": "scope_1",
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


class TransferRecord(BaseModel):
    """A single intra-group transfer record."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    transfer_id: str = Field(default_factory=_new_uuid)
    seller_entity_id: str = Field(...)
    seller_entity_name: str = Field("")
    buyer_entity_id: str = Field(...)
    buyer_entity_name: str = Field("")
    transfer_type: TransferType = Field(TransferType.OTHER)
    quantity: Decimal = Field(Decimal("0"), description="Quantity transferred")
    unit: str = Field("", description="Unit of quantity (MWh, tonnes, etc.)")
    seller_reported_tco2e: Decimal = Field(Decimal("0"))
    buyer_reported_tco2e: Decimal = Field(Decimal("0"))
    reporting_period: str = Field("")
    evidence_ref: str = Field("")
    source: str = Field("manual")


class MatchedPair(BaseModel):
    """A matched seller-buyer transfer pair."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    match_id: str = Field(default_factory=_new_uuid)
    seller_transfer_id: str = Field("")
    buyer_transfer_id: str = Field("")
    seller_entity_id: str = Field(...)
    buyer_entity_id: str = Field(...)
    transfer_type: TransferType = Field(TransferType.OTHER)
    seller_amount_tco2e: Decimal = Field(Decimal("0"))
    buyer_amount_tco2e: Decimal = Field(Decimal("0"))
    match_status: MatchStatus = Field(MatchStatus.UNMATCHED)
    variance_pct: Decimal = Field(Decimal("0"))
    agreed_amount_tco2e: Decimal = Field(Decimal("0"))


class EliminationRecord(BaseModel):
    """Calculated elimination amount for a transfer."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elimination_id: str = Field(default_factory=_new_uuid)
    match_id: str = Field("")
    seller_entity_id: str = Field(...)
    buyer_entity_id: str = Field(...)
    transfer_type: TransferType = Field(TransferType.OTHER)
    scope: EmissionScope = Field(EmissionScope.SCOPE_1)
    seller_elimination_tco2e: Decimal = Field(Decimal("0"))
    buyer_elimination_tco2e: Decimal = Field(Decimal("0"))
    net_elimination_tco2e: Decimal = Field(Decimal("0"))
    method: str = Field("", description="Elimination method applied")
    provenance_hash: str = Field("")


class ReconciliationSummary(BaseModel):
    """Reconciliation summary for all eliminations."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_seller_eliminations_tco2e: Decimal = Field(Decimal("0"))
    total_buyer_eliminations_tco2e: Decimal = Field(Decimal("0"))
    net_elimination_tco2e: Decimal = Field(Decimal("0"))
    imbalance_tco2e: Decimal = Field(Decimal("0"))
    imbalance_pct: Decimal = Field(Decimal("0"))
    verdict: ReconciliationVerdict = Field(ReconciliationVerdict.BALANCED)
    transfers_matched: int = Field(0)
    transfers_unmatched: int = Field(0)
    provenance_hash: str = Field("")


class EliminationInput(BaseModel):
    """Input for the elimination workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    transfers: List[Dict[str, Any]] = Field(
        default_factory=list, description="Intra-group transfer records"
    )
    entity_ids: List[str] = Field(
        default_factory=list, description="Entity IDs in consolidation boundary"
    )
    matching_tolerance_pct: Decimal = Field(MATCHING_TOLERANCE_PCT)
    skip_phases: List[str] = Field(default_factory=list)


class EliminationResult(BaseModel):
    """Output from the elimination workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    identified_transfers: List[TransferRecord] = Field(default_factory=list)
    matched_pairs: List[MatchedPair] = Field(default_factory=list)
    eliminations: List[EliminationRecord] = Field(default_factory=list)
    reconciliation: Optional[ReconciliationSummary] = Field(None)
    total_eliminated_tco2e: Decimal = Field(Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class EliminationWorkflow:
    """
    4-phase elimination workflow for intra-group GHG transfers.

    Identifies transfers, matches seller-buyer records, calculates
    elimination amounts, and reconciles with SHA-256 provenance.

    Example:
        >>> wf = EliminationWorkflow()
        >>> inp = EliminationInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     transfers=[{
        ...         "seller_entity_id": "E1", "buyer_entity_id": "E2",
        ...         "transfer_type": "energy_electricity",
        ...         "seller_reported_tco2e": "100", "buyer_reported_tco2e": "98",
        ...     }],
        ...     entity_ids=["E1", "E2"],
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.total_eliminated_tco2e > 0
    """

    PHASE_ORDER: List[EliminationPhase] = [
        EliminationPhase.TRANSFER_IDENTIFICATION,
        EliminationPhase.MATCHING_VERIFICATION,
        EliminationPhase.ELIMINATION_CALCULATION,
        EliminationPhase.RECONCILIATION_CHECK,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._transfers: List[TransferRecord] = []
        self._matched: List[MatchedPair] = []
        self._eliminations: List[EliminationRecord] = []

    def execute(self, input_data: EliminationInput) -> EliminationResult:
        """Execute the full 4-phase elimination workflow."""
        start = _utcnow()
        result = EliminationResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            EliminationPhase.TRANSFER_IDENTIFICATION: self._phase_transfer_identification,
            EliminationPhase.MATCHING_VERIFICATION: self._phase_matching_verification,
            EliminationPhase.ELIMINATION_CALCULATION: self._phase_elimination_calculation,
            EliminationPhase.RECONCILIATION_CHECK: self._phase_reconciliation_check,
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
            f"{result.workflow_id}|{result.organisation_id}|"
            f"{float(result.total_eliminated_tco2e)}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- TRANSFER IDENTIFICATION
    # -----------------------------------------------------------------

    def _phase_transfer_identification(
        self, input_data: EliminationInput, result: EliminationResult,
    ) -> Dict[str, Any]:
        """Identify all intra-group transfers across boundary entities."""
        logger.info("Phase 1 -- Transfer Identification: %d raw transfers",
                     len(input_data.transfers))

        boundary_ids = set(input_data.entity_ids)
        transfers: List[TransferRecord] = []
        external_skipped = 0
        invalid_skipped = 0

        for raw in input_data.transfers:
            seller_id = raw.get("seller_entity_id", "")
            buyer_id = raw.get("buyer_entity_id", "")

            # Both entities must be in the boundary for intra-group
            if seller_id not in boundary_ids or buyer_id not in boundary_ids:
                external_skipped += 1
                continue

            if seller_id == buyer_id:
                result.warnings.append(f"Self-transfer detected: {seller_id} -- skipped")
                invalid_skipped += 1
                continue

            try:
                ttype = TransferType(raw.get("transfer_type", "other"))
            except ValueError:
                ttype = TransferType.OTHER

            record = TransferRecord(
                seller_entity_id=seller_id,
                seller_entity_name=raw.get("seller_entity_name", ""),
                buyer_entity_id=buyer_id,
                buyer_entity_name=raw.get("buyer_entity_name", ""),
                transfer_type=ttype,
                quantity=self._dec(raw.get("quantity", "0")),
                unit=raw.get("unit", ""),
                seller_reported_tco2e=self._dec(raw.get("seller_reported_tco2e", "0")),
                buyer_reported_tco2e=self._dec(raw.get("buyer_reported_tco2e", "0")),
                reporting_period=raw.get("reporting_period", ""),
                evidence_ref=raw.get("evidence_ref", ""),
                source=raw.get("source", "manual"),
            )
            transfers.append(record)

        self._transfers = transfers
        result.identified_transfers = transfers

        type_dist: Dict[str, int] = {}
        for t in transfers:
            type_dist[t.transfer_type.value] = type_dist.get(t.transfer_type.value, 0) + 1

        logger.info("Identified %d intra-group transfers, %d external skipped",
                     len(transfers), external_skipped)
        return {
            "transfers_identified": len(transfers),
            "external_skipped": external_skipped,
            "invalid_skipped": invalid_skipped,
            "type_distribution": type_dist,
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- MATCHING VERIFICATION
    # -----------------------------------------------------------------

    def _phase_matching_verification(
        self, input_data: EliminationInput, result: EliminationResult,
    ) -> Dict[str, Any]:
        """Match seller records with buyer records and verify amounts."""
        logger.info("Phase 2 -- Matching Verification: %d transfers", len(self._transfers))
        tolerance = input_data.matching_tolerance_pct
        matched_pairs: List[MatchedPair] = []

        # Group transfers by entity pair and type for matching
        pair_groups: Dict[str, List[TransferRecord]] = {}
        for t in self._transfers:
            key = f"{t.seller_entity_id}|{t.buyer_entity_id}|{t.transfer_type.value}"
            if key not in pair_groups:
                pair_groups[key] = []
            pair_groups[key].append(t)

        matched_count = 0
        partial_count = 0
        unmatched_count = 0

        for key, group in pair_groups.items():
            # Aggregate seller and buyer amounts for the pair
            total_seller = sum(t.seller_reported_tco2e for t in group)
            total_buyer = sum(t.buyer_reported_tco2e for t in group)
            first = group[0]

            # Calculate variance
            variance_pct = Decimal("0")
            reference = max(total_seller, total_buyer)
            if reference > Decimal("0"):
                variance_pct = (abs(total_seller - total_buyer) / reference * Decimal("100")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

            # Determine match status
            if total_seller == Decimal("0") and total_buyer == Decimal("0"):
                match_status = MatchStatus.UNMATCHED
                unmatched_count += 1
            elif variance_pct <= tolerance:
                match_status = MatchStatus.MATCHED
                matched_count += 1
            elif variance_pct <= tolerance * Decimal("2"):
                match_status = MatchStatus.PARTIAL_MATCH
                partial_count += 1
                result.warnings.append(
                    f"Partial match for {first.seller_entity_id} -> {first.buyer_entity_id} "
                    f"({first.transfer_type.value}): variance {variance_pct}%"
                )
            else:
                match_status = MatchStatus.DISPUTED
                unmatched_count += 1
                result.warnings.append(
                    f"Disputed transfer {first.seller_entity_id} -> {first.buyer_entity_id}: "
                    f"seller={total_seller}, buyer={total_buyer}, variance={variance_pct}%"
                )

            # Agreed amount: use seller amount for matched, average for partial
            if match_status == MatchStatus.MATCHED:
                agreed = total_seller
            elif match_status == MatchStatus.PARTIAL_MATCH:
                agreed = ((total_seller + total_buyer) / Decimal("2")).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            else:
                agreed = Decimal("0")

            pair = MatchedPair(
                seller_entity_id=first.seller_entity_id,
                buyer_entity_id=first.buyer_entity_id,
                transfer_type=first.transfer_type,
                seller_amount_tco2e=total_seller.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                buyer_amount_tco2e=total_buyer.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                match_status=match_status,
                variance_pct=variance_pct,
                agreed_amount_tco2e=agreed,
            )
            matched_pairs.append(pair)

        self._matched = matched_pairs
        result.matched_pairs = matched_pairs

        logger.info("Matching: %d matched, %d partial, %d unmatched",
                     matched_count, partial_count, unmatched_count)
        return {
            "total_pairs": len(matched_pairs),
            "matched": matched_count,
            "partial_match": partial_count,
            "unmatched": unmatched_count,
            "tolerance_pct": float(tolerance),
        }

    # -----------------------------------------------------------------
    # PHASE 3 -- ELIMINATION CALCULATION
    # -----------------------------------------------------------------

    def _phase_elimination_calculation(
        self, input_data: EliminationInput, result: EliminationResult,
    ) -> Dict[str, Any]:
        """Calculate elimination amounts per matched pair."""
        logger.info("Phase 3 -- Elimination Calculation: %d pairs", len(self._matched))
        eliminations: List[EliminationRecord] = []
        total_eliminated = Decimal("0")

        for pair in self._matched:
            if pair.match_status in (MatchStatus.UNMATCHED, MatchStatus.DISPUTED):
                continue

            if pair.agreed_amount_tco2e <= Decimal("0"):
                continue

            # Determine scope for elimination
            scope_str = TRANSFER_SCOPE_MAPPING.get(pair.transfer_type.value, "scope_1")
            try:
                scope = EmissionScope(scope_str)
            except ValueError:
                scope = EmissionScope.SCOPE_1

            # Seller elimination: reduce seller's reported emissions
            # Buyer elimination: reduce buyer's scope 2/3 reported emissions
            seller_elim = pair.agreed_amount_tco2e
            buyer_elim = pair.agreed_amount_tco2e

            # Net elimination avoids double-counting
            net_elim = pair.agreed_amount_tco2e

            prov_input = (
                f"{pair.seller_entity_id}|{pair.buyer_entity_id}|"
                f"{pair.transfer_type.value}|{float(net_elim)}"
            )
            prov_hash = _compute_hash(prov_input)

            method = (
                f"Matched elimination ({pair.match_status.value}): "
                f"seller and buyer agree on {pair.agreed_amount_tco2e} tCO2e"
            )

            record = EliminationRecord(
                match_id=pair.match_id,
                seller_entity_id=pair.seller_entity_id,
                buyer_entity_id=pair.buyer_entity_id,
                transfer_type=pair.transfer_type,
                scope=scope,
                seller_elimination_tco2e=seller_elim,
                buyer_elimination_tco2e=buyer_elim,
                net_elimination_tco2e=net_elim,
                method=method,
                provenance_hash=prov_hash,
            )
            eliminations.append(record)
            total_eliminated += net_elim

        self._eliminations = eliminations
        result.eliminations = eliminations
        result.total_eliminated_tco2e = total_eliminated.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        logger.info("Eliminations calculated: %d records, %.2f tCO2e total",
                     len(eliminations), float(total_eliminated))
        return {
            "eliminations_calculated": len(eliminations),
            "total_eliminated_tco2e": float(total_eliminated),
            "by_scope": self._count_by_scope(eliminations),
        }

    def _count_by_scope(self, elims: List[EliminationRecord]) -> Dict[str, float]:
        """Sum eliminations by scope."""
        by_scope: Dict[str, Decimal] = {}
        for e in elims:
            by_scope[e.scope.value] = by_scope.get(e.scope.value, Decimal("0")) + e.net_elimination_tco2e
        return {k: float(v) for k, v in by_scope.items()}

    # -----------------------------------------------------------------
    # PHASE 4 -- RECONCILIATION CHECK
    # -----------------------------------------------------------------

    def _phase_reconciliation_check(
        self, input_data: EliminationInput, result: EliminationResult,
    ) -> Dict[str, Any]:
        """Verify eliminations balance and generate audit trail."""
        logger.info("Phase 4 -- Reconciliation Check")

        total_seller = sum(e.seller_elimination_tco2e for e in self._eliminations)
        total_buyer = sum(e.buyer_elimination_tco2e for e in self._eliminations)
        net_elim = sum(e.net_elimination_tco2e for e in self._eliminations)

        imbalance = abs(total_seller - total_buyer)
        imbalance_pct = Decimal("0")
        if max(total_seller, total_buyer) > Decimal("0"):
            imbalance_pct = (
                imbalance / max(total_seller, total_buyer) * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        if imbalance_pct <= RECONCILIATION_TOLERANCE_PCT:
            verdict = ReconciliationVerdict.BALANCED
        elif imbalance_pct <= RECONCILIATION_TOLERANCE_PCT * Decimal("3"):
            verdict = ReconciliationVerdict.MINOR_IMBALANCE
            result.warnings.append(f"Minor elimination imbalance: {imbalance_pct}%")
        else:
            verdict = ReconciliationVerdict.MAJOR_IMBALANCE
            result.warnings.append(f"Major elimination imbalance: {imbalance_pct}%")

        matched_count = sum(
            1 for p in self._matched
            if p.match_status in (MatchStatus.MATCHED, MatchStatus.PARTIAL_MATCH)
        )
        unmatched_count = sum(
            1 for p in self._matched
            if p.match_status in (MatchStatus.UNMATCHED, MatchStatus.DISPUTED)
        )

        now_iso = _utcnow().isoformat()
        recon_prov = _compute_hash(
            f"{input_data.organisation_id}|{float(net_elim)}|"
            f"{verdict.value}|{now_iso}"
        )

        summary = ReconciliationSummary(
            total_seller_eliminations_tco2e=total_seller.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            total_buyer_eliminations_tco2e=total_buyer.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            net_elimination_tco2e=net_elim.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            imbalance_tco2e=imbalance.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            imbalance_pct=imbalance_pct,
            verdict=verdict,
            transfers_matched=matched_count,
            transfers_unmatched=unmatched_count,
            provenance_hash=recon_prov,
        )
        result.reconciliation = summary

        logger.info("Reconciliation: %s, net %.2f tCO2e, imbalance %.2f%%",
                     verdict.value, float(net_elim), float(imbalance_pct))
        return {
            "verdict": verdict.value,
            "net_elimination_tco2e": float(net_elim),
            "imbalance_pct": float(imbalance_pct),
            "transfers_matched": matched_count,
            "transfers_unmatched": unmatched_count,
            "provenance_hash": recon_prov,
        }

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
    "EliminationWorkflow",
    "EliminationInput",
    "EliminationResult",
    "EliminationPhase",
    "TransferType",
    "MatchStatus",
    "EmissionScope",
    "ReconciliationVerdict",
    "TransferRecord",
    "MatchedPair",
    "EliminationRecord",
    "ReconciliationSummary",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
