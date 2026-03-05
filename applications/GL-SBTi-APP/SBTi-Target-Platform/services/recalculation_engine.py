"""
Recalculation Engine -- Base Year Emissions Recalculation Management

Implements base year recalculation triggers, threshold checks, M&A impact
modeling, organic/inorganic growth separation, target updates, and audit
trail generation per SBTi criterion C11 and the SBTi Corporate Net-Zero
Standard v1.2.

The SBTi requires base year emissions to be recalculated when significant
structural changes (mergers, acquisitions, divestitures, outsourcing,
insourcing, methodology changes, or data error corrections) result in a
change of 5% or more in base year emissions.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria and Recommendations v5.1 (2023), Criterion C11
    - SBTi Corporate Net-Zero Standard v1.2, Section 7
    - GHG Protocol Corporate Standard, Chapter 5 (Base Year)

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = RecalculationEngine(SBTiAppConfig())
    >>> check = engine.monitor_base_year_change("org-1", 100000.0, 106000.0)
    >>> print(check.exceeds_threshold)
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    RecalculationTrigger,
    SBTiAppConfig,
    SBTI_MINIMUM_AMBITION,
)
from .models import (
    Recalculation,
    FiveYearReview,
    Target,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ThresholdCheck(BaseModel):
    """Result of checking whether a base year change exceeds the 5% threshold."""

    org_id: str = Field(...)
    original_emissions: float = Field(..., ge=0.0)
    new_emissions: float = Field(..., ge=0.0)
    absolute_change: float = Field(default=0.0)
    percentage_change: float = Field(default=0.0)
    threshold_pct: float = Field(default=5.0)
    exceeds_threshold: bool = Field(default=False)
    trigger_recommended: bool = Field(default=False)
    checked_at: datetime = Field(default_factory=_now)


class RevalidationAssessment(BaseModel):
    """Assessment of whether a recalculation triggers target revalidation."""

    recalculation_id: str = Field(...)
    org_id: str = Field(...)
    requires_revalidation: bool = Field(default=False)
    reason: str = Field(default="")
    affected_targets: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)


class MAImpactResult(BaseModel):
    """M&A impact assessment on SBTi targets."""

    org_id: str = Field(...)
    acquisition_name: str = Field(default="")
    acquired_emissions: float = Field(default=0.0, ge=0.0)
    original_boundary_emissions: float = Field(default=0.0, ge=0.0)
    new_boundary_emissions: float = Field(default=0.0, ge=0.0)
    boundary_change_pct: float = Field(default=0.0)
    triggers_recalculation: bool = Field(default=False)
    organic_emissions: float = Field(default=0.0, ge=0.0)
    inorganic_emissions: float = Field(default=0.0, ge=0.0)
    recommended_base_year_adjustment: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class GrowthSeparation(BaseModel):
    """Separation of organic vs inorganic emission changes."""

    org_id: str = Field(...)
    total_emissions_change: float = Field(default=0.0)
    organic_change: float = Field(default=0.0)
    inorganic_change: float = Field(default=0.0)
    organic_pct: float = Field(default=0.0)
    inorganic_pct: float = Field(default=0.0)
    organic_sources: List[Dict[str, Any]] = Field(default_factory=list)
    inorganic_sources: List[Dict[str, Any]] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=_now)


class AuditTrail(BaseModel):
    """Recalculation audit trail for provenance tracking."""

    recalculation_id: str = Field(...)
    org_id: str = Field(...)
    events: List[Dict[str, Any]] = Field(default_factory=list)
    original_data_hash: str = Field(default="")
    recalculated_data_hash: str = Field(default="")
    approval_chain: List[Dict[str, str]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class RecalculationSummary(BaseModel):
    """Summary of all recalculations for an organization."""

    org_id: str = Field(...)
    total_recalculations: int = Field(default=0)
    applied_count: int = Field(default=0)
    pending_count: int = Field(default=0)
    triggers_breakdown: Dict[str, int] = Field(default_factory=dict)
    latest_recalculation_date: Optional[datetime] = Field(None)
    cumulative_change_pct: float = Field(default=0.0)
    generated_at: datetime = Field(default_factory=_now)


class StoredRecalculation(BaseModel):
    """In-memory representation of a recalculation record."""

    id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    trigger: RecalculationTrigger = Field(...)
    trigger_description: str = Field(default="")
    original_emissions: float = Field(default=0.0, ge=0.0)
    recalculated_emissions: float = Field(default=0.0, ge=0.0)
    scope_1_original: Optional[float] = Field(None)
    scope_1_recalculated: Optional[float] = Field(None)
    scope_2_original: Optional[float] = Field(None)
    scope_2_recalculated: Optional[float] = Field(None)
    scope_3_original: Optional[float] = Field(None)
    scope_3_recalculated: Optional[float] = Field(None)
    percentage_change: float = Field(default=0.0)
    affected_target_ids: List[str] = Field(default_factory=list)
    applied: bool = Field(default=False)
    applied_at: Optional[datetime] = Field(None)
    approved_by: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# RecalculationEngine
# ---------------------------------------------------------------------------

class RecalculationEngine:
    """
    Base year emissions recalculation engine per SBTi criterion C11.

    Monitors emission boundary changes, evaluates threshold exceedance,
    manages recalculation records, assesses revalidation requirements,
    models M&A impacts, separates organic/inorganic growth, applies
    recalculations to affected targets, and generates audit trails.

    Attributes:
        config: Application configuration.
        _recalculations: In-memory store keyed by org_id.
        _targets: In-memory target store keyed by org_id.

    Example:
        >>> engine = RecalculationEngine(SBTiAppConfig())
        >>> check = engine.monitor_base_year_change("org-1", 100000.0, 106000.0)
        >>> print(check.exceeds_threshold)
        True
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the RecalculationEngine."""
        self.config = config or SBTiAppConfig()
        self._recalculations: Dict[str, List[StoredRecalculation]] = {}
        self._targets: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("RecalculationEngine initialized (threshold=%.1f%%)",
                     float(self.config.recalculation_threshold_pct))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def monitor_base_year_change(
        self,
        org_id: str,
        original_emissions: float,
        new_emissions: float,
    ) -> ThresholdCheck:
        """
        Check if a base year emissions change exceeds the recalculation threshold.

        Per SBTi C11, a significant change is defined as >= 5% of base year
        emissions.  This method computes the absolute and percentage change and
        determines whether a formal recalculation is recommended.

        Args:
            org_id: Organization identifier.
            original_emissions: Original base year emissions (tCO2e).
            new_emissions: Revised base year emissions (tCO2e).

        Returns:
            ThresholdCheck with exceedance determination.
        """
        absolute_change = abs(new_emissions - original_emissions)
        pct_change = (
            (absolute_change / original_emissions * 100.0)
            if original_emissions > 0 else 0.0
        )
        threshold = float(self.config.recalculation_threshold_pct)
        exceeds = pct_change >= threshold

        result = ThresholdCheck(
            org_id=org_id,
            original_emissions=original_emissions,
            new_emissions=new_emissions,
            absolute_change=round(absolute_change, 2),
            percentage_change=round(pct_change, 2),
            threshold_pct=threshold,
            exceeds_threshold=exceeds,
            trigger_recommended=exceeds,
        )

        logger.info(
            "Threshold check for org %s: change=%.2f%% (threshold=%.1f%%), exceeds=%s",
            org_id, pct_change, threshold, exceeds,
        )
        return result

    def create_recalculation(
        self,
        org_id: str,
        trigger: RecalculationTrigger,
        original: float,
        recalculated: float,
        trigger_description: str = "",
        scope_1_original: Optional[float] = None,
        scope_1_recalculated: Optional[float] = None,
        scope_2_original: Optional[float] = None,
        scope_2_recalculated: Optional[float] = None,
        scope_3_original: Optional[float] = None,
        scope_3_recalculated: Optional[float] = None,
    ) -> StoredRecalculation:
        """
        Create a new base year recalculation record.

        Args:
            org_id: Organization identifier.
            trigger: The trigger event causing recalculation.
            original: Original base year total emissions (tCO2e).
            recalculated: Recalculated base year total emissions (tCO2e).
            trigger_description: Narrative description of the trigger event.
            scope_1_original: Optional original Scope 1 emissions.
            scope_1_recalculated: Optional recalculated Scope 1 emissions.
            scope_2_original: Optional original Scope 2 emissions.
            scope_2_recalculated: Optional recalculated Scope 2 emissions.
            scope_3_original: Optional original Scope 3 emissions.
            scope_3_recalculated: Optional recalculated Scope 3 emissions.

        Returns:
            StoredRecalculation record with computed percentage change.
        """
        pct_change = self.calculate_percentage_change(original, recalculated)

        # Identify affected targets for this org
        affected = [t["id"] for t in self._targets.get(org_id, [])]

        provenance = _sha256(
            f"{org_id}:{trigger.value}:{original}:{recalculated}"
        )

        record = StoredRecalculation(
            org_id=org_id,
            trigger=trigger,
            trigger_description=trigger_description,
            original_emissions=original,
            recalculated_emissions=recalculated,
            scope_1_original=scope_1_original,
            scope_1_recalculated=scope_1_recalculated,
            scope_2_original=scope_2_original,
            scope_2_recalculated=scope_2_recalculated,
            scope_3_original=scope_3_original,
            scope_3_recalculated=scope_3_recalculated,
            percentage_change=pct_change,
            affected_target_ids=affected,
            provenance_hash=provenance,
        )

        self._recalculations.setdefault(org_id, []).append(record)

        logger.info(
            "Created recalculation %s for org %s: trigger=%s, change=%.2f%%",
            record.id, org_id, trigger.value, pct_change,
        )
        return record

    def get_recalculation_history(
        self, org_id: str,
    ) -> List[StoredRecalculation]:
        """
        Get the full recalculation history for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            List of StoredRecalculation records sorted by creation date.
        """
        records = self._recalculations.get(org_id, [])
        sorted_records = sorted(records, key=lambda r: r.created_at, reverse=True)
        logger.info(
            "Retrieved %d recalculations for org %s", len(sorted_records), org_id,
        )
        return sorted_records

    def assess_revalidation_need(
        self, recalculation: StoredRecalculation,
    ) -> RevalidationAssessment:
        """
        Assess whether a recalculation triggers target revalidation.

        Per SBTi, revalidation is required when the recalculation materially
        changes the base year emissions and therefore the reduction pathway.
        Structural triggers (M&A, divestiture) with >= 5% change always
        require revalidation.

        Args:
            recalculation: A StoredRecalculation record.

        Returns:
            RevalidationAssessment with determination and recommended actions.
        """
        threshold = float(self.config.recalculation_threshold_pct)
        requires = abs(recalculation.percentage_change) >= threshold
        actions: List[str] = []
        reason = ""

        if requires:
            reason = (
                f"Base year emissions changed by {recalculation.percentage_change:.2f}% "
                f"(threshold: {threshold:.1f}%). Revalidation is required."
            )
            actions.append("Submit updated base year inventory to SBTi")
            actions.append("Recalculate all affected target pathways")
            actions.append("Update annual progress tracking against new baseline")

            if recalculation.trigger in (
                RecalculationTrigger.MERGERS_ACQUISITIONS,
                RecalculationTrigger.DIVESTITURE,
            ):
                actions.append("Provide M&A/divestiture documentation to SBTi")
                actions.append("Demonstrate organic vs inorganic separation")

            if recalculation.trigger == RecalculationTrigger.METHODOLOGY_CHANGE:
                actions.append("Document methodology change rationale")
                actions.append("Apply new methodology consistently across all years")
        else:
            reason = (
                f"Base year change of {recalculation.percentage_change:.2f}% "
                f"is below {threshold:.1f}% threshold. Revalidation not required."
            )

        result = RevalidationAssessment(
            recalculation_id=recalculation.id,
            org_id=recalculation.org_id,
            requires_revalidation=requires,
            reason=reason,
            affected_targets=recalculation.affected_target_ids,
            recommended_actions=actions,
        )

        logger.info(
            "Revalidation assessment for recalc %s: required=%s",
            recalculation.id, requires,
        )
        return result

    def calculate_percentage_change(
        self, original: float, recalculated: float,
    ) -> float:
        """
        Calculate the percentage change between original and recalculated values.

        Args:
            original: Original emissions value.
            recalculated: Recalculated emissions value.

        Returns:
            Percentage change (positive = increase, negative = decrease).
        """
        if original <= 0:
            return 0.0
        change = ((recalculated - original) / original) * 100.0
        return round(change, 2)

    def model_ma_impact(
        self,
        org_id: str,
        acquisition_data: Dict[str, Any],
    ) -> MAImpactResult:
        """
        Model the impact of a merger or acquisition on SBTi targets.

        Computes the boundary change, determines whether the threshold is
        exceeded, and calculates the recommended base year adjustment.

        Args:
            org_id: Organization identifier.
            acquisition_data: Dict with keys:
                - acquisition_name: str
                - acquired_emissions: float (tCO2e)
                - original_boundary_emissions: float (tCO2e)
                - acquired_scope_1: float (optional)
                - acquired_scope_2: float (optional)
                - acquired_scope_3: float (optional)

        Returns:
            MAImpactResult with boundary change analysis.
        """
        acquired = acquisition_data.get("acquired_emissions", 0.0)
        original = acquisition_data.get("original_boundary_emissions", 0.0)
        acq_name = acquisition_data.get("acquisition_name", "Unknown")

        new_boundary = original + acquired
        boundary_change_pct = (
            (acquired / original * 100.0) if original > 0 else 0.0
        )
        threshold = float(self.config.recalculation_threshold_pct)
        triggers = boundary_change_pct >= threshold

        # Recommended adjustment is the full acquired emissions added to base year
        recommended_adj = acquired if triggers else 0.0

        provenance = _sha256(
            f"{org_id}:ma:{acq_name}:{original}:{acquired}"
        )

        result = MAImpactResult(
            org_id=org_id,
            acquisition_name=acq_name,
            acquired_emissions=acquired,
            original_boundary_emissions=original,
            new_boundary_emissions=round(new_boundary, 2),
            boundary_change_pct=round(boundary_change_pct, 2),
            triggers_recalculation=triggers,
            organic_emissions=original,
            inorganic_emissions=acquired,
            recommended_base_year_adjustment=round(recommended_adj, 2),
            provenance_hash=provenance,
        )

        logger.info(
            "M&A impact for org %s (%s): change=%.2f%%, triggers=%s",
            org_id, acq_name, boundary_change_pct, triggers,
        )
        return result

    def separate_organic_inorganic(
        self,
        org_id: str,
        data: Dict[str, Any],
    ) -> GrowthSeparation:
        """
        Separate organic vs inorganic emission changes.

        Organic growth arises from operational changes within the existing
        boundary.  Inorganic growth arises from structural changes (M&A,
        divestiture, outsourcing/insourcing).

        Args:
            org_id: Organization identifier.
            data: Dict with keys:
                - total_change: float
                - organic_sources: List[Dict] (each has name, emissions)
                - inorganic_sources: List[Dict] (each has name, emissions)

        Returns:
            GrowthSeparation with breakdown by type.
        """
        total_change = data.get("total_change", 0.0)
        organic_sources = data.get("organic_sources", [])
        inorganic_sources = data.get("inorganic_sources", [])

        organic_total = sum(s.get("emissions", 0.0) for s in organic_sources)
        inorganic_total = sum(s.get("emissions", 0.0) for s in inorganic_sources)

        # Normalize if needed
        combined = organic_total + inorganic_total
        if combined > 0 and abs(combined - abs(total_change)) > 0.01:
            scale = abs(total_change) / combined
            organic_total *= scale
            inorganic_total *= scale

        organic_pct = (organic_total / abs(total_change) * 100.0) if total_change != 0 else 0.0
        inorganic_pct = (inorganic_total / abs(total_change) * 100.0) if total_change != 0 else 0.0

        result = GrowthSeparation(
            org_id=org_id,
            total_emissions_change=round(total_change, 2),
            organic_change=round(organic_total, 2),
            inorganic_change=round(inorganic_total, 2),
            organic_pct=round(organic_pct, 2),
            inorganic_pct=round(inorganic_pct, 2),
            organic_sources=organic_sources,
            inorganic_sources=inorganic_sources,
        )

        logger.info(
            "Growth separation for org %s: organic=%.1f%%, inorganic=%.1f%%",
            org_id, organic_pct, inorganic_pct,
        )
        return result

    def apply_recalculation_to_targets(
        self, recalculation_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Apply a recalculation to all affected targets.

        Updates base year emissions on each target and recalculates the
        annual reduction rate and target year emissions accordingly.

        Args:
            recalculation_id: ID of the recalculation to apply.

        Returns:
            List of updated target dicts with old/new values.

        Raises:
            ValueError: If recalculation_id is not found.
        """
        # Find the recalculation
        recalc = self._find_recalculation(recalculation_id)
        if recalc is None:
            raise ValueError(f"Recalculation {recalculation_id} not found")

        if recalc.applied:
            logger.warning("Recalculation %s already applied", recalculation_id)
            return []

        updated_targets: List[Dict[str, Any]] = []
        org_targets = self._targets.get(recalc.org_id, [])

        for target in org_targets:
            old_base = target.get("base_year_emissions", 0.0)
            if old_base <= 0:
                continue

            # Apply proportional adjustment
            ratio = recalc.recalculated_emissions / recalc.original_emissions
            new_base = old_base * ratio
            reduction_pct = target.get("target_reduction_pct", 0.0)
            years = target.get("target_year", 2030) - target.get("base_year", 2020)
            new_annual_rate = (reduction_pct / years) if years > 0 else 0.0

            target["base_year_emissions"] = round(new_base, 2)
            target["annual_reduction_rate"] = round(new_annual_rate, 2)
            target["recalculated"] = True
            target["recalculation_id"] = recalculation_id

            updated_targets.append({
                "target_id": target.get("id", ""),
                "old_base_year_emissions": round(old_base, 2),
                "new_base_year_emissions": round(new_base, 2),
                "change_pct": round((ratio - 1) * 100, 2),
                "new_annual_rate": round(new_annual_rate, 2),
            })

        # Mark recalculation as applied
        recalc.applied = True
        recalc.applied_at = _now()

        logger.info(
            "Applied recalculation %s to %d targets for org %s",
            recalculation_id, len(updated_targets), recalc.org_id,
        )
        return updated_targets

    def generate_recalculation_audit(
        self, recalculation_id: str,
    ) -> AuditTrail:
        """
        Generate a complete audit trail for a recalculation.

        Includes provenance hashes of original and recalculated data,
        the approval chain, and a chronological event log.

        Args:
            recalculation_id: ID of the recalculation.

        Returns:
            AuditTrail with complete provenance information.

        Raises:
            ValueError: If recalculation_id is not found.
        """
        recalc = self._find_recalculation(recalculation_id)
        if recalc is None:
            raise ValueError(f"Recalculation {recalculation_id} not found")

        original_hash = _sha256(
            f"{recalc.org_id}:original:{recalc.original_emissions}"
        )
        recalculated_hash = _sha256(
            f"{recalc.org_id}:recalculated:{recalc.recalculated_emissions}"
        )

        events: List[Dict[str, Any]] = [
            {
                "event": "recalculation_created",
                "timestamp": recalc.created_at.isoformat(),
                "trigger": recalc.trigger.value,
                "description": recalc.trigger_description,
            },
            {
                "event": "threshold_check",
                "timestamp": recalc.created_at.isoformat(),
                "percentage_change": recalc.percentage_change,
                "threshold": float(self.config.recalculation_threshold_pct),
                "exceeds": abs(recalc.percentage_change) >= float(
                    self.config.recalculation_threshold_pct
                ),
            },
        ]

        if recalc.applied:
            events.append({
                "event": "recalculation_applied",
                "timestamp": (
                    recalc.applied_at.isoformat() if recalc.applied_at else ""
                ),
                "affected_targets": recalc.affected_target_ids,
            })

        approval_chain: List[Dict[str, str]] = []
        if recalc.approved_by:
            approval_chain.append({
                "approver": recalc.approved_by,
                "action": "approved",
                "timestamp": (
                    recalc.applied_at.isoformat() if recalc.applied_at else ""
                ),
            })

        provenance = _sha256(
            f"{recalculation_id}:{original_hash}:{recalculated_hash}"
        )

        audit = AuditTrail(
            recalculation_id=recalculation_id,
            org_id=recalc.org_id,
            events=events,
            original_data_hash=original_hash,
            recalculated_data_hash=recalculated_hash,
            approval_chain=approval_chain,
            provenance_hash=provenance,
        )

        logger.info(
            "Generated audit trail for recalculation %s: %d events",
            recalculation_id, len(events),
        )
        return audit

    def get_recalculation_summary(
        self, org_id: str,
    ) -> RecalculationSummary:
        """
        Get a summary of all recalculations for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            RecalculationSummary with counts and cumulative change.
        """
        records = self._recalculations.get(org_id, [])
        applied = [r for r in records if r.applied]
        pending = [r for r in records if not r.applied]

        # Breakdown by trigger type
        trigger_counts: Dict[str, int] = {}
        for r in records:
            key = r.trigger.value
            trigger_counts[key] = trigger_counts.get(key, 0) + 1

        # Cumulative change from all applied recalculations
        cumulative = 0.0
        for r in applied:
            cumulative += r.percentage_change

        latest_date = None
        if records:
            latest_date = max(r.created_at for r in records)

        summary = RecalculationSummary(
            org_id=org_id,
            total_recalculations=len(records),
            applied_count=len(applied),
            pending_count=len(pending),
            triggers_breakdown=trigger_counts,
            latest_recalculation_date=latest_date,
            cumulative_change_pct=round(cumulative, 2),
        )

        logger.info(
            "Recalculation summary for org %s: total=%d, applied=%d, pending=%d",
            org_id, len(records), len(applied), len(pending),
        )
        return summary

    # ------------------------------------------------------------------
    # Target store helpers (for apply_recalculation_to_targets)
    # ------------------------------------------------------------------

    def register_target(self, org_id: str, target_data: Dict[str, Any]) -> None:
        """
        Register a target in the in-memory store for recalculation tracking.

        Args:
            org_id: Organization identifier.
            target_data: Dict with target fields (id, base_year_emissions, etc.).
        """
        self._targets.setdefault(org_id, []).append(target_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_recalculation(
        self, recalculation_id: str,
    ) -> Optional[StoredRecalculation]:
        """Find a recalculation by ID across all organizations."""
        for records in self._recalculations.values():
            for record in records:
                if record.id == recalculation_id:
                    return record
        return None
