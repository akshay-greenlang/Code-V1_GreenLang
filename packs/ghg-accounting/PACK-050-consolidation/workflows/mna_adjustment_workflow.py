# -*- coding: utf-8 -*-
"""
M&A Adjustment Workflow
====================================

5-phase workflow for handling mergers, acquisitions, and divestitures
within GHG consolidation per GHG Protocol Corporate Standard Chapter 5
within PACK-050 GHG Consolidation Pack.

Phases:
    1. EventCapture              -- Capture M&A event details (acquisition,
                                    divestiture, merger, restructuring) with
                                    effective dates and entity impacts.
    2. BoundaryImpactAssessment  -- Assess impact of the M&A event on the
                                    organisational boundary and consolidation
                                    scope.
    3. ProRataCalculation        -- Calculate pro-rata emissions for partial
                                    year periods (pre/post transaction).
    4. BaseYearRestatement       -- Restate base year emissions if structural
                                    change triggers the significance threshold.
    5. DisclosureGeneration      -- Generate M&A disclosure notes for
                                    reporting and assurance.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 5) -- Tracking emissions over time
    GHG Protocol Corporate Standard (Ch. 5.3) -- Acquisitions & divestitures
    ISO 14064-1:2018 (Cl. 5.5) -- Recalculation of base year
    CSRD / ESRS E1 -- Climate change (structural changes)

Author: GreenLang Team
Version: 50.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
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


class MnAPhase(str, Enum):
    EVENT_CAPTURE = "event_capture"
    BOUNDARY_IMPACT_ASSESSMENT = "boundary_impact_assessment"
    PRO_RATA_CALCULATION = "pro_rata_calculation"
    BASE_YEAR_RESTATEMENT = "base_year_restatement"
    DISCLOSURE_GENERATION = "disclosure_generation"


class MnAEventType(str, Enum):
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    SPIN_OFF = "spin_off"
    RESTRUCTURING = "restructuring"
    JOINT_VENTURE_ENTRY = "joint_venture_entry"
    JOINT_VENTURE_EXIT = "joint_venture_exit"


class BoundaryImpact(str, Enum):
    ENTITY_ADDED = "entity_added"
    ENTITY_REMOVED = "entity_removed"
    OWNERSHIP_CHANGED = "ownership_changed"
    NO_IMPACT = "no_impact"


class RestatementTrigger(str, Enum):
    STRUCTURAL_CHANGE = "structural_change"
    METHODOLOGY_CHANGE = "methodology_change"
    DATA_IMPROVEMENT = "data_improvement"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    NOT_TRIGGERED = "not_triggered"


class EmissionScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


# =============================================================================
# REFERENCE DATA
# =============================================================================

DEFAULT_RESTATEMENT_THRESHOLD_PCT = Decimal("5.0")
DAYS_IN_YEAR = Decimal("365")


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


class MnAEvent(BaseModel):
    """M&A event record."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str = Field(default_factory=_new_uuid)
    event_type: MnAEventType = Field(...)
    event_name: str = Field("", description="Descriptive name of the M&A event")
    effective_date: str = Field(..., description="Effective date ISO format")
    target_entity_id: str = Field("", description="Entity being acquired/divested")
    target_entity_name: str = Field("")
    counterparty_name: str = Field("", description="Acquiring/divesting party")
    ownership_before_pct: Decimal = Field(Decimal("0"))
    ownership_after_pct: Decimal = Field(Decimal("0"))
    transaction_value: Decimal = Field(Decimal("0"))
    currency: str = Field("USD")
    annual_emissions_tco2e: Decimal = Field(
        Decimal("0"), description="Full-year emissions of target entity"
    )
    scope_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    notes: str = Field("")


class BoundaryImpactAssessment(BaseModel):
    """Assessment of M&A event's impact on boundary."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str = Field(...)
    impact_type: BoundaryImpact = Field(BoundaryImpact.NO_IMPACT)
    entities_added: List[str] = Field(default_factory=list)
    entities_removed: List[str] = Field(default_factory=list)
    ownership_changes: List[Dict[str, Any]] = Field(default_factory=list)
    emissions_impact_tco2e: Decimal = Field(Decimal("0"))
    emissions_impact_pct: Decimal = Field(Decimal("0"))
    triggers_restatement: bool = Field(False)
    assessment_notes: str = Field("")


class ProRataResult(BaseModel):
    """Pro-rata emissions calculation for partial year."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str = Field(...)
    entity_id: str = Field("")
    entity_name: str = Field("")
    effective_date: str = Field("")
    days_in_period: int = Field(0)
    days_in_year: int = Field(365)
    pro_rata_factor: Decimal = Field(Decimal("0"))
    annual_emissions_tco2e: Decimal = Field(Decimal("0"))
    pro_rata_scope_1: Decimal = Field(Decimal("0"))
    pro_rata_scope_2_location: Decimal = Field(Decimal("0"))
    pro_rata_scope_2_market: Decimal = Field(Decimal("0"))
    pro_rata_scope_3: Decimal = Field(Decimal("0"))
    pro_rata_total_tco2e: Decimal = Field(Decimal("0"))
    method: str = Field("", description="Pro-rata calculation method")


class BaseYearRestatementResult(BaseModel):
    """Base year restatement calculation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    event_id: str = Field(...)
    trigger: RestatementTrigger = Field(RestatementTrigger.NOT_TRIGGERED)
    base_year: int = Field(0)
    original_base_year_tco2e: Decimal = Field(Decimal("0"))
    adjustment_tco2e: Decimal = Field(Decimal("0"))
    restated_base_year_tco2e: Decimal = Field(Decimal("0"))
    change_pct: Decimal = Field(Decimal("0"))
    threshold_pct: Decimal = Field(DEFAULT_RESTATEMENT_THRESHOLD_PCT)
    restatement_required: bool = Field(False)
    restatement_method: str = Field("")
    scope_adjustments: Dict[str, Decimal] = Field(default_factory=dict)


class DisclosureNote(BaseModel):
    """Generated M&A disclosure note."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    note_id: str = Field(default_factory=_new_uuid)
    event_id: str = Field(...)
    title: str = Field("")
    narrative: str = Field("")
    quantitative_impact: Dict[str, Any] = Field(default_factory=dict)
    base_year_restated: bool = Field(False)
    pro_rata_applied: bool = Field(False)
    frameworks_applicable: List[str] = Field(default_factory=list)
    provenance_hash: str = Field("")


class MnAInput(BaseModel):
    """Input for the M&A adjustment workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    base_year: int = Field(0, description="Base year for restatement checks")
    base_year_emissions_tco2e: Decimal = Field(Decimal("0"))
    base_year_scope_breakdown: Dict[str, Decimal] = Field(default_factory=dict)
    current_total_emissions_tco2e: Decimal = Field(Decimal("0"))
    events: List[Dict[str, Any]] = Field(
        default_factory=list, description="M&A event records"
    )
    restatement_threshold_pct: Decimal = Field(DEFAULT_RESTATEMENT_THRESHOLD_PCT)
    skip_phases: List[str] = Field(default_factory=list)


class MnAResult(BaseModel):
    """Output from the M&A adjustment workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    events: List[MnAEvent] = Field(default_factory=list)
    boundary_impacts: List[BoundaryImpactAssessment] = Field(default_factory=list)
    pro_rata_results: List[ProRataResult] = Field(default_factory=list)
    restatement_results: List[BaseYearRestatementResult] = Field(default_factory=list)
    disclosure_notes: List[DisclosureNote] = Field(default_factory=list)
    total_pro_rata_adjustment_tco2e: Decimal = Field(Decimal("0"))
    base_year_restated: bool = Field(False)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class MnAAdjustmentWorkflow:
    """
    5-phase M&A adjustment workflow for GHG consolidation.

    Captures M&A events, assesses boundary impact, calculates pro-rata
    emissions, restates base year if needed, and generates disclosure
    notes with SHA-256 provenance.

    Example:
        >>> wf = MnAAdjustmentWorkflow()
        >>> inp = MnAInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     base_year=2020, base_year_emissions_tco2e=Decimal("50000"),
        ...     events=[{
        ...         "event_type": "acquisition", "effective_date": "2025-07-01",
        ...         "target_entity_name": "Target Co",
        ...         "annual_emissions_tco2e": "5000",
        ...     }],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[MnAPhase] = [
        MnAPhase.EVENT_CAPTURE,
        MnAPhase.BOUNDARY_IMPACT_ASSESSMENT,
        MnAPhase.PRO_RATA_CALCULATION,
        MnAPhase.BASE_YEAR_RESTATEMENT,
        MnAPhase.DISCLOSURE_GENERATION,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._events: List[MnAEvent] = []
        self._impacts: List[BoundaryImpactAssessment] = []
        self._pro_rata: List[ProRataResult] = []
        self._restatements: List[BaseYearRestatementResult] = []

    def execute(self, input_data: MnAInput) -> MnAResult:
        """Execute the full 5-phase M&A adjustment workflow."""
        start = _utcnow()
        result = MnAResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING,
            started_at=start.isoformat(),
        )

        phase_methods = {
            MnAPhase.EVENT_CAPTURE: self._phase_event_capture,
            MnAPhase.BOUNDARY_IMPACT_ASSESSMENT: self._phase_boundary_impact,
            MnAPhase.PRO_RATA_CALCULATION: self._phase_pro_rata,
            MnAPhase.BASE_YEAR_RESTATEMENT: self._phase_base_year_restatement,
            MnAPhase.DISCLOSURE_GENERATION: self._phase_disclosure_generation,
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
            f"{len(self._events)}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- EVENT CAPTURE
    # -----------------------------------------------------------------

    def _phase_event_capture(
        self, input_data: MnAInput, result: MnAResult,
    ) -> Dict[str, Any]:
        """Capture M&A event details."""
        logger.info("Phase 1 -- Event Capture: %d events", len(input_data.events))
        events: List[MnAEvent] = []

        for raw in input_data.events:
            try:
                event_type = MnAEventType(raw.get("event_type", "acquisition"))
            except ValueError:
                event_type = MnAEventType.ACQUISITION
                result.warnings.append(f"Unknown event type: {raw.get('event_type')}")

            scope_breakdown: Dict[str, Decimal] = {}
            for scope in ["scope_1", "scope_2_location", "scope_2_market", "scope_3"]:
                if scope in raw:
                    scope_breakdown[scope] = self._dec(raw[scope])

            event = MnAEvent(
                event_type=event_type,
                event_name=raw.get("event_name", f"{event_type.value} event"),
                effective_date=raw.get("effective_date", ""),
                target_entity_id=raw.get("target_entity_id", ""),
                target_entity_name=raw.get("target_entity_name", ""),
                counterparty_name=raw.get("counterparty_name", ""),
                ownership_before_pct=self._dec(raw.get("ownership_before_pct", "0")),
                ownership_after_pct=self._dec(raw.get("ownership_after_pct", "100")),
                transaction_value=self._dec(raw.get("transaction_value", "0")),
                currency=raw.get("currency", "USD"),
                annual_emissions_tco2e=self._dec(raw.get("annual_emissions_tco2e", "0")),
                scope_breakdown=scope_breakdown,
                notes=raw.get("notes", ""),
            )
            events.append(event)

        self._events = events
        result.events = events

        type_dist: Dict[str, int] = {}
        for e in events:
            type_dist[e.event_type.value] = type_dist.get(e.event_type.value, 0) + 1

        logger.info("Captured %d M&A events: %s", len(events), type_dist)
        return {
            "events_captured": len(events),
            "event_type_distribution": type_dist,
            "total_annual_emissions_tco2e": float(
                sum(e.annual_emissions_tco2e for e in events)
            ),
        }

    # -----------------------------------------------------------------
    # PHASE 2 -- BOUNDARY IMPACT ASSESSMENT
    # -----------------------------------------------------------------

    def _phase_boundary_impact(
        self, input_data: MnAInput, result: MnAResult,
    ) -> Dict[str, Any]:
        """Assess impact of M&A events on the organisational boundary."""
        logger.info("Phase 2 -- Boundary Impact Assessment: %d events", len(self._events))
        impacts: List[BoundaryImpactAssessment] = []
        total_impact = Decimal("0")
        restatement_triggers = 0

        for event in self._events:
            impact_type, entities_added, entities_removed = self._determine_impact(event)
            emissions_impact = self._calculate_emissions_impact(event, impact_type)
            total_impact += emissions_impact

            # Calculate as percentage of current total
            impact_pct = Decimal("0")
            if input_data.current_total_emissions_tco2e > Decimal("0"):
                impact_pct = (
                    abs(emissions_impact) / input_data.current_total_emissions_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            triggers = impact_pct >= input_data.restatement_threshold_pct
            if triggers:
                restatement_triggers += 1

            notes = self._build_impact_notes(event, impact_type, emissions_impact, impact_pct)

            assessment = BoundaryImpactAssessment(
                event_id=event.event_id,
                impact_type=impact_type,
                entities_added=entities_added,
                entities_removed=entities_removed,
                emissions_impact_tco2e=emissions_impact.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                emissions_impact_pct=impact_pct,
                triggers_restatement=triggers,
                assessment_notes=notes,
            )
            impacts.append(assessment)

        self._impacts = impacts
        result.boundary_impacts = impacts

        logger.info("Boundary impact: %.2f tCO2e total, %d trigger restatement",
                     float(total_impact), restatement_triggers)
        return {
            "events_assessed": len(impacts),
            "total_emissions_impact_tco2e": float(total_impact),
            "restatement_triggers": restatement_triggers,
        }

    def _determine_impact(
        self, event: MnAEvent
    ) -> tuple:
        """Determine the boundary impact type for an event."""
        entities_added: List[str] = []
        entities_removed: List[str] = []

        if event.event_type in (MnAEventType.ACQUISITION, MnAEventType.MERGER,
                                MnAEventType.JOINT_VENTURE_ENTRY):
            impact_type = BoundaryImpact.ENTITY_ADDED
            if event.target_entity_id:
                entities_added.append(event.target_entity_id)
        elif event.event_type in (MnAEventType.DIVESTITURE, MnAEventType.SPIN_OFF,
                                  MnAEventType.JOINT_VENTURE_EXIT):
            impact_type = BoundaryImpact.ENTITY_REMOVED
            if event.target_entity_id:
                entities_removed.append(event.target_entity_id)
        elif event.event_type == MnAEventType.RESTRUCTURING:
            if event.ownership_after_pct != event.ownership_before_pct:
                impact_type = BoundaryImpact.OWNERSHIP_CHANGED
            else:
                impact_type = BoundaryImpact.NO_IMPACT
        else:
            impact_type = BoundaryImpact.NO_IMPACT

        return impact_type, entities_added, entities_removed

    def _calculate_emissions_impact(
        self, event: MnAEvent, impact_type: BoundaryImpact
    ) -> Decimal:
        """Calculate the emissions impact of an M&A event."""
        if impact_type == BoundaryImpact.ENTITY_ADDED:
            return event.annual_emissions_tco2e
        elif impact_type == BoundaryImpact.ENTITY_REMOVED:
            return -event.annual_emissions_tco2e
        elif impact_type == BoundaryImpact.OWNERSHIP_CHANGED:
            delta_pct = event.ownership_after_pct - event.ownership_before_pct
            return (event.annual_emissions_tco2e * delta_pct / Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        return Decimal("0")

    def _build_impact_notes(
        self, event: MnAEvent, impact: BoundaryImpact,
        emissions: Decimal, pct: Decimal,
    ) -> str:
        """Build deterministic impact assessment notes."""
        return (
            f"{event.event_type.value.replace('_', ' ').title()} of "
            f"{event.target_entity_name or 'entity'}: "
            f"{impact.value.replace('_', ' ')} with "
            f"{abs(emissions):.2f} tCO2e impact ({pct}% of total). "
            f"Effective date: {event.effective_date}."
        )

    # -----------------------------------------------------------------
    # PHASE 3 -- PRO-RATA CALCULATION
    # -----------------------------------------------------------------

    def _phase_pro_rata(
        self, input_data: MnAInput, result: MnAResult,
    ) -> Dict[str, Any]:
        """Calculate pro-rata emissions for partial year periods."""
        logger.info("Phase 3 -- Pro-Rata Calculation")
        pro_rata_results: List[ProRataResult] = []
        total_adjustment = Decimal("0")
        year_start = date(input_data.reporting_year, 1, 1)
        year_end = date(input_data.reporting_year, 12, 31)
        days_in_year = (year_end - year_start).days + 1

        for event in self._events:
            if not event.effective_date:
                result.warnings.append(
                    f"Event {event.event_name} missing effective date -- skipped pro-rata"
                )
                continue

            try:
                eff_date = date.fromisoformat(event.effective_date[:10])
            except ValueError:
                result.warnings.append(f"Invalid date format: {event.effective_date}")
                continue

            # Only process events within the reporting year
            if eff_date.year != input_data.reporting_year:
                continue

            days_in_period = self._calculate_days_in_period(
                event.event_type, eff_date, year_start, year_end
            )
            pro_rata_factor = (Decimal(str(days_in_period)) / Decimal(str(days_in_year))).quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            )

            # Calculate pro-rata by scope
            scope_bd = event.scope_breakdown
            s1 = scope_bd.get("scope_1", event.annual_emissions_tco2e * Decimal("0.4"))
            s2l = scope_bd.get("scope_2_location", event.annual_emissions_tco2e * Decimal("0.3"))
            s2m = scope_bd.get("scope_2_market", s2l)
            s3 = scope_bd.get("scope_3", event.annual_emissions_tco2e * Decimal("0.3"))

            pr_s1 = (s1 * pro_rata_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pr_s2l = (s2l * pro_rata_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pr_s2m = (s2m * pro_rata_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pr_s3 = (s3 * pro_rata_factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            pr_total = pr_s1 + pr_s2l + pr_s3

            method = (
                f"Time-based pro-rata: {days_in_period}/{days_in_year} days "
                f"(factor={float(pro_rata_factor):.4f})"
            )

            pr = ProRataResult(
                event_id=event.event_id,
                entity_id=event.target_entity_id,
                entity_name=event.target_entity_name,
                effective_date=event.effective_date,
                days_in_period=days_in_period,
                days_in_year=days_in_year,
                pro_rata_factor=pro_rata_factor,
                annual_emissions_tco2e=event.annual_emissions_tco2e,
                pro_rata_scope_1=pr_s1,
                pro_rata_scope_2_location=pr_s2l,
                pro_rata_scope_2_market=pr_s2m,
                pro_rata_scope_3=pr_s3,
                pro_rata_total_tco2e=pr_total,
                method=method,
            )
            pro_rata_results.append(pr)
            total_adjustment += pr_total

        self._pro_rata = pro_rata_results
        result.pro_rata_results = pro_rata_results
        result.total_pro_rata_adjustment_tco2e = total_adjustment.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        logger.info("Pro-rata: %d calculations, total %.2f tCO2e",
                     len(pro_rata_results), float(total_adjustment))
        return {
            "pro_rata_calculations": len(pro_rata_results),
            "total_pro_rata_tco2e": float(total_adjustment),
        }

    def _calculate_days_in_period(
        self, event_type: MnAEventType, eff_date: date,
        year_start: date, year_end: date,
    ) -> int:
        """Calculate days in period based on event type and effective date."""
        if event_type in (MnAEventType.ACQUISITION, MnAEventType.MERGER,
                          MnAEventType.JOINT_VENTURE_ENTRY):
            # Entity added: count days from effective date to year end
            return (year_end - eff_date).days + 1
        elif event_type in (MnAEventType.DIVESTITURE, MnAEventType.SPIN_OFF,
                            MnAEventType.JOINT_VENTURE_EXIT):
            # Entity removed: count days from year start to effective date
            return (eff_date - year_start).days
        else:
            # Restructuring: full year
            return (year_end - year_start).days + 1

    # -----------------------------------------------------------------
    # PHASE 4 -- BASE YEAR RESTATEMENT
    # -----------------------------------------------------------------

    def _phase_base_year_restatement(
        self, input_data: MnAInput, result: MnAResult,
    ) -> Dict[str, Any]:
        """Restate base year if structural change exceeds threshold."""
        logger.info("Phase 4 -- Base Year Restatement")

        if input_data.base_year == 0 or input_data.base_year_emissions_tco2e == Decimal("0"):
            result.warnings.append("No base year configured -- skipping restatement check")
            return {"restatement_required": False, "reason": "no_base_year_configured"}

        restatements: List[BaseYearRestatementResult] = []
        any_restatement = False

        for impact in self._impacts:
            if not impact.triggers_restatement:
                restatements.append(BaseYearRestatementResult(
                    event_id=impact.event_id,
                    trigger=RestatementTrigger.NOT_TRIGGERED,
                    base_year=input_data.base_year,
                    original_base_year_tco2e=input_data.base_year_emissions_tco2e,
                    threshold_pct=input_data.restatement_threshold_pct,
                    restatement_required=False,
                ))
                continue

            # Calculate adjustment to base year
            adjustment = impact.emissions_impact_tco2e
            restated = input_data.base_year_emissions_tco2e + adjustment
            change_pct = Decimal("0")
            if input_data.base_year_emissions_tco2e > Decimal("0"):
                change_pct = (
                    abs(adjustment) / input_data.base_year_emissions_tco2e * Decimal("100")
                ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            restatement_required = change_pct >= input_data.restatement_threshold_pct
            if restatement_required:
                any_restatement = True

            # Scope-level adjustments (proportional to base year breakdown)
            scope_adjs: Dict[str, Decimal] = {}
            base_breakdown = input_data.base_year_scope_breakdown
            if base_breakdown and input_data.base_year_emissions_tco2e > Decimal("0"):
                for scope, base_val in base_breakdown.items():
                    ratio = base_val / input_data.base_year_emissions_tco2e
                    scope_adjs[scope] = (adjustment * ratio).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )

            method = (
                f"Structural change restatement per GHG Protocol Ch. 5: "
                f"base year {input_data.base_year} adjusted by "
                f"{float(adjustment):.2f} tCO2e ({float(change_pct):.1f}%)"
            )

            restatements.append(BaseYearRestatementResult(
                event_id=impact.event_id,
                trigger=RestatementTrigger.STRUCTURAL_CHANGE,
                base_year=input_data.base_year,
                original_base_year_tco2e=input_data.base_year_emissions_tco2e,
                adjustment_tco2e=adjustment.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                restated_base_year_tco2e=restated.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                change_pct=change_pct,
                threshold_pct=input_data.restatement_threshold_pct,
                restatement_required=restatement_required,
                restatement_method=method if restatement_required else "",
                scope_adjustments=scope_adjs,
            ))

        self._restatements = restatements
        result.restatement_results = restatements
        result.base_year_restated = any_restatement

        logger.info("Restatement: %d checks, restatement required: %s",
                     len(restatements), any_restatement)
        return {
            "events_checked": len(restatements),
            "restatement_required": any_restatement,
            "base_year": input_data.base_year,
        }

    # -----------------------------------------------------------------
    # PHASE 5 -- DISCLOSURE GENERATION
    # -----------------------------------------------------------------

    def _phase_disclosure_generation(
        self, input_data: MnAInput, result: MnAResult,
    ) -> Dict[str, Any]:
        """Generate M&A disclosure notes for reporting."""
        logger.info("Phase 5 -- Disclosure Generation")
        notes: List[DisclosureNote] = []
        now_iso = _utcnow().isoformat()

        for event in self._events:
            impact = next(
                (i for i in self._impacts if i.event_id == event.event_id), None
            )
            pro_rata = next(
                (p for p in self._pro_rata if p.event_id == event.event_id), None
            )
            restatement = next(
                (r for r in self._restatements if r.event_id == event.event_id), None
            )

            title = self._build_disclosure_title(event)
            narrative = self._build_disclosure_narrative(event, impact, pro_rata, restatement)

            quant_impact: Dict[str, Any] = {
                "annual_emissions_tco2e": float(event.annual_emissions_tco2e),
            }
            if pro_rata:
                quant_impact["pro_rata_tco2e"] = float(pro_rata.pro_rata_total_tco2e)
                quant_impact["pro_rata_factor"] = float(pro_rata.pro_rata_factor)
            if restatement and restatement.restatement_required:
                quant_impact["base_year_adjustment_tco2e"] = float(restatement.adjustment_tco2e)
                quant_impact["restated_base_year_tco2e"] = float(restatement.restated_base_year_tco2e)

            prov_hash = _compute_hash(
                f"{event.event_id}|{title}|{float(event.annual_emissions_tco2e)}|{now_iso}"
            )

            note = DisclosureNote(
                event_id=event.event_id,
                title=title,
                narrative=narrative,
                quantitative_impact=quant_impact,
                base_year_restated=restatement.restatement_required if restatement else False,
                pro_rata_applied=pro_rata is not None,
                frameworks_applicable=["GHG Protocol", "ISO 14064-1", "CSRD/ESRS E1"],
                provenance_hash=prov_hash,
            )
            notes.append(note)

        result.disclosure_notes = notes

        logger.info("Generated %d disclosure notes", len(notes))
        return {
            "notes_generated": len(notes),
            "base_year_restated": result.base_year_restated,
        }

    def _build_disclosure_title(self, event: MnAEvent) -> str:
        """Build disclosure note title."""
        action = event.event_type.value.replace("_", " ").title()
        target = event.target_entity_name or "entity"
        return f"{action} of {target} -- GHG Inventory Impact"

    def _build_disclosure_narrative(
        self, event: MnAEvent,
        impact: Optional[BoundaryImpactAssessment],
        pro_rata: Optional[ProRataResult],
        restatement: Optional[BaseYearRestatementResult],
    ) -> str:
        """Build deterministic disclosure narrative."""
        parts: List[str] = []
        action = event.event_type.value.replace("_", " ")
        parts.append(
            f"On {event.effective_date}, the organisation completed the {action} "
            f"of {event.target_entity_name or 'an entity'}."
        )

        if impact:
            parts.append(
                f"This resulted in a {impact.impact_type.value.replace('_', ' ')} "
                f"with an emissions impact of {float(impact.emissions_impact_tco2e):.2f} tCO2e "
                f"({float(impact.emissions_impact_pct):.1f}% of total)."
            )

        if pro_rata:
            parts.append(
                f"Pro-rata emissions of {float(pro_rata.pro_rata_total_tco2e):.2f} tCO2e "
                f"were calculated for the {pro_rata.days_in_period}-day period "
                f"(factor: {float(pro_rata.pro_rata_factor):.4f})."
            )

        if restatement and restatement.restatement_required:
            parts.append(
                f"The base year ({restatement.base_year}) has been restated from "
                f"{float(restatement.original_base_year_tco2e):.2f} to "
                f"{float(restatement.restated_base_year_tco2e):.2f} tCO2e "
                f"per GHG Protocol Corporate Standard Chapter 5."
            )

        return " ".join(parts)

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
    "MnAAdjustmentWorkflow",
    "MnAInput",
    "MnAResult",
    "MnAPhase",
    "MnAEventType",
    "BoundaryImpact",
    "RestatementTrigger",
    "EmissionScope",
    "MnAEvent",
    "BoundaryImpactAssessment",
    "ProRataResult",
    "BaseYearRestatementResult",
    "DisclosureNote",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
