"""
PACK-050 GHG Consolidation Pack - Acquisition & Divestiture Engine
====================================================================

Handles M&A events (acquisitions, divestitures, mergers, demergers,
joint venture formation/dissolution) and their impact on GHG
inventories.  Implements pro-rata emission calculations, base year
restatement triggers per GHG Protocol Chapter 5, and organic vs.
structural growth separation.

Regulatory Basis:
    - GHG Protocol Corporate Standard (Chapter 5): Tracking emissions
      over time - recalculating base year for structural changes.
    - GHG Protocol Corporate Standard (Chapter 3): Organisational
      boundary changes from M&A activity.
    - ISO 14064-1:2018 (Clause 5.4.3): Recalculation policy for
      significant structural changes.
    - ESRS E1-6: Requires restatement of comparative figures when
      organisational boundaries change.
    - IFRS S2: Climate-related disclosures require consistent
      boundary treatment for M&A.

Calculation Methodology:
    Pro-Rata Factor:
        pro_rata_factor = days_included / total_days_in_period

    Prorated Emissions:
        prorated_emissions = entity_annual_emissions * pro_rata_factor

    Base Year Restatement:
        base_year_adjusted = base_year_original +/- structural_change

    Organic Growth Separation:
        organic_growth = current_year_adj - base_year_adj

Capabilities:
    - Acquisition handling with pro-rata from acquisition date
    - Divestiture handling with pro-rata to divestiture date
    - Base year recalculation triggers per GHG Protocol Chapter 5
    - Organic vs. structural growth separation
    - Merger integration (combine two entity inventories)
    - Demerger handling (split one entity's inventory)
    - Joint venture formation and dissolution
    - Complete M&A event timeline

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result object

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-050 GHG Consolidation
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash, excluding volatile fields."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("created_at", "updated_at", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert any value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Divide safely, returning *default* when denominator is zero."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> Decimal:
    """Round a value to two decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _round4(value: Any) -> Decimal:
    """Round a value to four decimal places using ROUND_HALF_UP."""
    return Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def _days_in_year(year: int) -> int:
    """Return the number of days in a given year."""
    start = date(year, 1, 1)
    end = date(year + 1, 1, 1)
    return (end - start).days


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MnAEventType(str, Enum):
    """Types of M&A events affecting GHG inventory boundaries."""
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    MERGER = "MERGER"
    DEMERGER = "DEMERGER"
    JV_FORMATION = "JV_FORMATION"
    JV_DISSOLUTION = "JV_DISSOLUTION"
    OUTSOURCING = "OUTSOURCING"
    INSOURCING = "INSOURCING"


class RestatementTrigger(str, Enum):
    """Triggers for base year restatement per GHG Protocol Chapter 5."""
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
    METHODOLOGY_CHANGE = "METHODOLOGY_CHANGE"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    SCOPE_BOUNDARY_CHANGE = "SCOPE_BOUNDARY_CHANGE"


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_SIGNIFICANCE_THRESHOLD_PCT = Decimal("5")


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class MnAEvent(BaseModel):
    """An M&A event that changes the organisational boundary.

    Records the event type, affected entity, effective date, and
    the emissions data associated with the structural change.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    event_id: str = Field(
        default_factory=_new_uuid,
        description="Unique event identifier.",
    )
    event_type: str = Field(
        ...,
        description="Type of M&A event.",
    )
    entity_id: str = Field(
        ...,
        description="Primary entity affected by this event.",
    )
    entity_name: Optional[str] = Field(
        None,
        description="Human-readable name of the entity.",
    )
    counterparty_entity_id: Optional[str] = Field(
        None,
        description="Counterparty entity (e.g. acquired company).",
    )
    effective_date: date = Field(
        ...,
        description="Date the event takes effect.",
    )
    reporting_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="Reporting year for this event.",
    )
    equity_pct: Decimal = Field(
        default=Decimal("100"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Equity percentage for JVs or partial acquisitions.",
    )
    annual_emissions_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Full-year emissions of the affected entity (tCO2e).",
    )
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Full-year Scope 1 emissions (tCO2e).",
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Full-year Scope 2 location-based emissions (tCO2e).",
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Full-year Scope 2 market-based emissions (tCO2e).",
    )
    scope3_tco2e: Decimal = Field(
        default=Decimal("0"),
        ge=Decimal("0"),
        description="Full-year Scope 3 emissions (tCO2e).",
    )
    description: Optional[str] = Field(
        None,
        description="Description of the M&A event.",
    )
    requires_base_year_restatement: bool = Field(
        default=False,
        description="Whether this event triggers base year restatement.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the event was registered.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "equity_pct", "annual_emissions_tco2e",
        "scope1_tco2e", "scope2_location_tco2e",
        "scope2_market_tco2e", "scope3_tco2e", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))

    @field_validator("event_type")
    @classmethod
    def _validate_event_type(cls, v: str) -> str:
        valid = {et.value for et in MnAEventType}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid event_type '{v}'. Must be one of {sorted(valid)}."
            )
        return v.upper()


class ProRataCalculation(BaseModel):
    """Pro-rata emission calculation for a partial-year event.

    Computes the proportion of annual emissions attributable to
    the period when the entity was within the reporting boundary.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    calculation_id: str = Field(
        default_factory=_new_uuid,
        description="Unique calculation identifier.",
    )
    event_id: str = Field(
        ...,
        description="M&A event this calculation relates to.",
    )
    entity_id: str = Field(
        ...,
        description="Entity whose emissions are prorated.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    period_start: date = Field(
        ...,
        description="Start of the inclusion period.",
    )
    period_end: date = Field(
        ...,
        description="End of the inclusion period.",
    )
    days_included: int = Field(
        ...,
        ge=0,
        description="Number of days within the boundary.",
    )
    total_days_in_period: int = Field(
        ...,
        ge=1,
        description="Total days in the reporting period.",
    )
    pro_rata_factor: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("1"),
        description="Pro-rata factor (days_included / total_days).",
    )
    annual_emissions_tco2e: Decimal = Field(
        ...,
        description="Full-year emissions before pro-rata.",
    )
    prorated_emissions_tco2e: Decimal = Field(
        ...,
        description="Emissions after pro-rata adjustment.",
    )
    scope1_prorated: Decimal = Field(
        default=Decimal("0"),
        description="Prorated Scope 1.",
    )
    scope2_location_prorated: Decimal = Field(
        default=Decimal("0"),
        description="Prorated Scope 2 (location).",
    )
    scope2_market_prorated: Decimal = Field(
        default=Decimal("0"),
        description="Prorated Scope 2 (market).",
    )
    scope3_prorated: Decimal = Field(
        default=Decimal("0"),
        description="Prorated Scope 3.",
    )
    equity_pct: Decimal = Field(
        default=Decimal("100"),
        description="Equity percentage applied.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the calculation was performed.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "pro_rata_factor", "annual_emissions_tco2e",
        "prorated_emissions_tco2e", "scope1_prorated",
        "scope2_location_prorated", "scope2_market_prorated",
        "scope3_prorated", "equity_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class BaseYearRestatement(BaseModel):
    """Base year restatement record per GHG Protocol Chapter 5.

    Documents the adjustment to the base year inventory when a
    structural change triggers a restatement.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    restatement_id: str = Field(
        default_factory=_new_uuid,
        description="Unique restatement identifier.",
    )
    event_id: str = Field(
        ...,
        description="M&A event triggering this restatement.",
    )
    base_year: int = Field(
        ...,
        ge=2000,
        le=2100,
        description="The base year being restated.",
    )
    trigger: str = Field(
        default=RestatementTrigger.STRUCTURAL_CHANGE.value,
        description="Reason for the restatement.",
    )
    original_total_tco2e: Decimal = Field(
        ...,
        description="Original base year total before restatement.",
    )
    adjustment_tco2e: Decimal = Field(
        ...,
        description="Adjustment amount (positive = increase, negative = decrease).",
    )
    restated_total_tco2e: Decimal = Field(
        ...,
        description="Restated base year total.",
    )
    original_scope1: Decimal = Field(default=Decimal("0"))
    restated_scope1: Decimal = Field(default=Decimal("0"))
    original_scope2_location: Decimal = Field(default=Decimal("0"))
    restated_scope2_location: Decimal = Field(default=Decimal("0"))
    original_scope2_market: Decimal = Field(default=Decimal("0"))
    restated_scope2_market: Decimal = Field(default=Decimal("0"))
    original_scope3: Decimal = Field(default=Decimal("0"))
    restated_scope3: Decimal = Field(default=Decimal("0"))
    significance_pct: Decimal = Field(
        default=Decimal("0"),
        description="Significance as % of base year total.",
    )
    is_significant: bool = Field(
        default=False,
        description="Whether the change meets the significance threshold.",
    )
    description: Optional[str] = Field(
        None,
        description="Description of the restatement.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When the restatement was recorded.",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash.",
    )

    @field_validator(
        "original_total_tco2e", "adjustment_tco2e",
        "restated_total_tco2e", "significance_pct",
        "original_scope1", "restated_scope1",
        "original_scope2_location", "restated_scope2_location",
        "original_scope2_market", "restated_scope2_market",
        "original_scope3", "restated_scope3", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class StructuralChangeRecord(BaseModel):
    """Summary of all structural changes in a reporting period.

    Tracks the cumulative impact of M&A events on the emissions
    inventory and the base year.
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    record_id: str = Field(
        default_factory=_new_uuid,
        description="Unique record identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Reporting year.",
    )
    total_events: int = Field(
        default=0,
        description="Total M&A events in the period.",
    )
    acquisitions_count: int = Field(default=0)
    divestitures_count: int = Field(default=0)
    mergers_count: int = Field(default=0)
    other_count: int = Field(default=0)
    net_emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Net impact on emissions from structural changes.",
    )
    events: List[MnAEvent] = Field(
        default_factory=list,
        description="All M&A events in the period.",
    )
    prorate_calculations: List[ProRataCalculation] = Field(
        default_factory=list,
        description="All pro-rata calculations.",
    )
    restatements: List[BaseYearRestatement] = Field(
        default_factory=list,
        description="Base year restatements triggered.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator("net_emission_impact_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


class OrganicGrowthAnalysis(BaseModel):
    """Separates organic growth from structural growth.

    Distinguishes emission changes attributable to operational
    performance (organic) from those caused by M&A activity
    (structural).
    """
    model_config = ConfigDict(
        json_encoders={Decimal: str},
        validate_default=True,
    )

    analysis_id: str = Field(
        default_factory=_new_uuid,
        description="Unique analysis identifier.",
    )
    reporting_year: int = Field(
        ...,
        description="Current reporting year.",
    )
    base_year: int = Field(
        ...,
        description="Base year for comparison.",
    )
    base_year_original_tco2e: Decimal = Field(
        ...,
        description="Original base year total.",
    )
    base_year_adjusted_tco2e: Decimal = Field(
        ...,
        description="Base year total after structural adjustments.",
    )
    current_year_tco2e: Decimal = Field(
        ...,
        description="Current year total emissions.",
    )
    structural_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Emissions change from structural changes.",
    )
    organic_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Emissions change from organic (operational) growth.",
    )
    total_change_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total change from base year.",
    )
    organic_change_pct: Decimal = Field(
        default=Decimal("0"),
        description="Organic change as % of adjusted base year.",
    )
    structural_change_pct: Decimal = Field(
        default=Decimal("0"),
        description="Structural change as % of original base year.",
    )
    total_change_pct: Decimal = Field(
        default=Decimal("0"),
        description="Total change as % of original base year.",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
    )
    provenance_hash: str = Field(default="")

    @field_validator(
        "base_year_original_tco2e", "base_year_adjusted_tco2e",
        "current_year_tco2e", "structural_change_tco2e",
        "organic_change_tco2e", "total_change_tco2e",
        "organic_change_pct", "structural_change_pct",
        "total_change_pct", mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Any:
        return Decimal(str(v))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class AcquisitionDivestitureEngine:
    """Handles M&A events and their impact on GHG inventories.

    Implements pro-rata calculations, base year restatement, and
    organic vs. structural growth separation per GHG Protocol.

    Attributes:
        _events: Dict mapping event_id to MnAEvent.
        _prorates: Dict mapping calculation_id to ProRataCalculation.
        _restatements: Dict mapping restatement_id to BaseYearRestatement.
        _significance_threshold_pct: Threshold for restatement trigger.

    Example:
        >>> engine = AcquisitionDivestitureEngine()
        >>> event = engine.register_event({
        ...     "event_type": "ACQUISITION",
        ...     "entity_id": "ENT-B",
        ...     "effective_date": "2025-07-01",
        ...     "reporting_year": 2025,
        ...     "annual_emissions_tco2e": "10000",
        ...     "scope1_tco2e": "4000",
        ...     "scope2_location_tco2e": "6000",
        ... })
        >>> prorate = engine.calculate_prorate(event.event_id)
        >>> assert prorate.pro_rata_factor < Decimal("1")
    """

    def __init__(
        self,
        significance_threshold_pct: Optional[
            Union[Decimal, str, int, float]
        ] = None,
    ) -> None:
        """Initialise the AcquisitionDivestitureEngine.

        Args:
            significance_threshold_pct: Threshold (%) for triggering
                base year restatement.  Defaults to 5%.
        """
        self._events: Dict[str, MnAEvent] = {}
        self._prorates: Dict[str, ProRataCalculation] = {}
        self._restatements: Dict[str, BaseYearRestatement] = {}
        self._significance_threshold_pct = _decimal(
            significance_threshold_pct
            if significance_threshold_pct is not None
            else DEFAULT_SIGNIFICANCE_THRESHOLD_PCT
        )
        logger.info(
            "AcquisitionDivestitureEngine v%s initialised "
            "(significance threshold=%s%%).",
            _MODULE_VERSION,
            self._significance_threshold_pct,
        )

    # ------------------------------------------------------------------
    # Event Registration
    # ------------------------------------------------------------------

    def register_event(
        self,
        event_data: Dict[str, Any],
    ) -> MnAEvent:
        """Register an M&A event.

        Args:
            event_data: Dictionary of event attributes.

        Returns:
            The created MnAEvent with provenance hash.

        Raises:
            ValueError: If required fields are invalid.
        """
        logger.info(
            "Registering M&A event: %s for entity '%s'.",
            event_data.get("event_type", "?"),
            event_data.get("entity_id", "?"),
        )

        if "event_id" not in event_data or not event_data["event_id"]:
            event_data["event_id"] = _new_uuid()

        # Convert string date if necessary
        if isinstance(event_data.get("effective_date"), str):
            event_data["effective_date"] = date.fromisoformat(
                event_data["effective_date"]
            )

        event = MnAEvent(**event_data)
        event.provenance_hash = _compute_hash(event)
        self._events[event.event_id] = event

        logger.info(
            "M&A event '%s' registered: %s, entity='%s', "
            "effective=%s, annual=%s tCO2e.",
            event.event_id,
            event.event_type,
            event.entity_id,
            event.effective_date,
            event.annual_emissions_tco2e,
        )
        return event

    # ------------------------------------------------------------------
    # Pro-Rata Calculation
    # ------------------------------------------------------------------

    def calculate_prorate(
        self,
        event_id: str,
        period_start: Optional[date] = None,
        period_end: Optional[date] = None,
    ) -> ProRataCalculation:
        """Calculate pro-rata emissions for an M&A event.

        For acquisitions: includes emissions from effective_date to
        period end.  For divestitures: includes emissions from
        period start to effective_date.

        Args:
            event_id: The M&A event to prorate.
            period_start: Start of reporting period (default: Jan 1).
            period_end: End of reporting period (default: Dec 31).

        Returns:
            ProRataCalculation with prorated emissions.

        Raises:
            KeyError: If event not found.
        """
        if event_id not in self._events:
            raise KeyError(f"M&A event '{event_id}' not found.")

        event = self._events[event_id]
        year = event.reporting_year

        if period_start is None:
            period_start = date(year, 1, 1)
        if period_end is None:
            period_end = date(year, 12, 31)

        total_days = (period_end - period_start).days + 1

        # Determine inclusion period based on event type
        eff = event.effective_date
        if event.event_type in (
            MnAEventType.ACQUISITION.value,
            MnAEventType.MERGER.value,
            MnAEventType.JV_FORMATION.value,
            MnAEventType.INSOURCING.value,
        ):
            # Include from effective date to period end
            incl_start = max(eff, period_start)
            incl_end = period_end
        elif event.event_type in (
            MnAEventType.DIVESTITURE.value,
            MnAEventType.DEMERGER.value,
            MnAEventType.JV_DISSOLUTION.value,
            MnAEventType.OUTSOURCING.value,
        ):
            # Include from period start to effective date
            incl_start = period_start
            incl_end = min(eff, period_end)
        else:
            incl_start = period_start
            incl_end = period_end

        days_included = max(0, (incl_end - incl_start).days + 1)

        # Handle case where effective date is outside reporting period
        if eff > period_end or eff < period_start:
            if event.event_type in (
                MnAEventType.ACQUISITION.value,
                MnAEventType.MERGER.value,
                MnAEventType.JV_FORMATION.value,
                MnAEventType.INSOURCING.value,
            ):
                days_included = total_days if eff <= period_start else 0
            else:
                days_included = 0 if eff <= period_start else total_days

        pro_rata_factor = _round4(
            _safe_divide(
                _decimal(days_included),
                _decimal(total_days),
            )
        )

        # Apply equity percentage
        equity_multiplier = _safe_divide(
            event.equity_pct, Decimal("100")
        )

        # Calculate prorated emissions
        annual = event.annual_emissions_tco2e
        prorated = _round2(annual * pro_rata_factor * equity_multiplier)
        s1_pr = _round2(
            event.scope1_tco2e * pro_rata_factor * equity_multiplier
        )
        s2_loc_pr = _round2(
            event.scope2_location_tco2e * pro_rata_factor * equity_multiplier
        )
        s2_mkt_pr = _round2(
            event.scope2_market_tco2e * pro_rata_factor * equity_multiplier
        )
        s3_pr = _round2(
            event.scope3_tco2e * pro_rata_factor * equity_multiplier
        )

        calc = ProRataCalculation(
            event_id=event_id,
            entity_id=event.entity_id,
            reporting_year=year,
            period_start=incl_start,
            period_end=incl_end,
            days_included=days_included,
            total_days_in_period=total_days,
            pro_rata_factor=pro_rata_factor,
            annual_emissions_tco2e=annual,
            prorated_emissions_tco2e=prorated,
            scope1_prorated=s1_pr,
            scope2_location_prorated=s2_loc_pr,
            scope2_market_prorated=s2_mkt_pr,
            scope3_prorated=s3_pr,
            equity_pct=event.equity_pct,
        )
        calc.provenance_hash = _compute_hash(calc)
        self._prorates[calc.calculation_id] = calc

        logger.info(
            "Pro-rata for event '%s': %d/%d days, factor=%s, "
            "prorated=%s tCO2e (equity=%s%%).",
            event_id, days_included, total_days,
            pro_rata_factor, prorated, event.equity_pct,
        )
        return calc

    # ------------------------------------------------------------------
    # Base Year Restatement
    # ------------------------------------------------------------------

    def trigger_base_year_restatement(
        self,
        event_id: str,
        base_year: int,
        base_year_total_tco2e: Union[Decimal, str, int, float],
        base_year_scope1: Union[Decimal, str, int, float] = "0",
        base_year_scope2_location: Union[Decimal, str, int, float] = "0",
        base_year_scope2_market: Union[Decimal, str, int, float] = "0",
        base_year_scope3: Union[Decimal, str, int, float] = "0",
    ) -> BaseYearRestatement:
        """Trigger base year restatement for a structural change.

        Per GHG Protocol Chapter 5, base year must be restated when
        structural changes (acquisitions, divestitures) exceed the
        significance threshold.

        For acquisitions: adds the entity's base year emissions.
        For divestitures: subtracts the entity's base year emissions.

        Args:
            event_id: The M&A event triggering restatement.
            base_year: The base year to restate.
            base_year_total_tco2e: Original base year total.
            base_year_scope1: Original base year Scope 1.
            base_year_scope2_location: Original base year Scope 2 (loc).
            base_year_scope2_market: Original base year Scope 2 (mkt).
            base_year_scope3: Original base year Scope 3.

        Returns:
            BaseYearRestatement with adjusted totals.

        Raises:
            KeyError: If event not found.
        """
        if event_id not in self._events:
            raise KeyError(f"M&A event '{event_id}' not found.")

        event = self._events[event_id]
        original_total = _decimal(base_year_total_tco2e)
        orig_s1 = _decimal(base_year_scope1)
        orig_s2_loc = _decimal(base_year_scope2_location)
        orig_s2_mkt = _decimal(base_year_scope2_market)
        orig_s3 = _decimal(base_year_scope3)

        # Determine adjustment direction
        if event.event_type in (
            MnAEventType.ACQUISITION.value,
            MnAEventType.MERGER.value,
            MnAEventType.JV_FORMATION.value,
            MnAEventType.INSOURCING.value,
        ):
            sign = Decimal("1")
        else:
            sign = Decimal("-1")

        # Apply equity percentage to the event's emissions
        eq_mult = _safe_divide(event.equity_pct, Decimal("100"))
        adj_total = _round2(event.annual_emissions_tco2e * eq_mult * sign)
        adj_s1 = _round2(event.scope1_tco2e * eq_mult * sign)
        adj_s2_loc = _round2(event.scope2_location_tco2e * eq_mult * sign)
        adj_s2_mkt = _round2(event.scope2_market_tco2e * eq_mult * sign)
        adj_s3 = _round2(event.scope3_tco2e * eq_mult * sign)

        restated_total = _round2(original_total + adj_total)
        restated_s1 = _round2(orig_s1 + adj_s1)
        restated_s2_loc = _round2(orig_s2_loc + adj_s2_loc)
        restated_s2_mkt = _round2(orig_s2_mkt + adj_s2_mkt)
        restated_s3 = _round2(orig_s3 + adj_s3)

        # Significance check
        significance_pct = _round2(
            _safe_divide(abs(adj_total), original_total) * Decimal("100")
        ) if original_total != Decimal("0") else Decimal("0")

        is_significant = significance_pct >= self._significance_threshold_pct

        restatement = BaseYearRestatement(
            event_id=event_id,
            base_year=base_year,
            trigger=RestatementTrigger.STRUCTURAL_CHANGE.value,
            original_total_tco2e=original_total,
            adjustment_tco2e=adj_total,
            restated_total_tco2e=restated_total,
            original_scope1=orig_s1,
            restated_scope1=restated_s1,
            original_scope2_location=orig_s2_loc,
            restated_scope2_location=restated_s2_loc,
            original_scope2_market=orig_s2_mkt,
            restated_scope2_market=restated_s2_mkt,
            original_scope3=orig_s3,
            restated_scope3=restated_s3,
            significance_pct=significance_pct,
            is_significant=is_significant,
            description=(
                f"Base year {base_year} restated for {event.event_type} "
                f"of entity '{event.entity_id}': {adj_total:+} tCO2e "
                f"({significance_pct}% of base year)."
            ),
        )
        restatement.provenance_hash = _compute_hash(restatement)
        self._restatements[restatement.restatement_id] = restatement

        # Mark the event
        event.requires_base_year_restatement = is_significant

        logger.info(
            "Base year restatement: year=%d, original=%s, adj=%s, "
            "restated=%s tCO2e, significant=%s (%s%%).",
            base_year, original_total, adj_total,
            restated_total, is_significant, significance_pct,
        )
        return restatement

    # ------------------------------------------------------------------
    # Organic vs. Structural Growth
    # ------------------------------------------------------------------

    def separate_organic_structural(
        self,
        reporting_year: int,
        base_year: int,
        base_year_original_tco2e: Union[Decimal, str, int, float],
        base_year_adjusted_tco2e: Union[Decimal, str, int, float],
        current_year_tco2e: Union[Decimal, str, int, float],
    ) -> OrganicGrowthAnalysis:
        """Separate organic from structural emission changes.

        Structural change = base_year_adjusted - base_year_original.
        Organic change = current_year - base_year_adjusted.
        Total change = current_year - base_year_original.

        Args:
            reporting_year: Current reporting year.
            base_year: Base year.
            base_year_original_tco2e: Original base year total.
            base_year_adjusted_tco2e: Adjusted base year total.
            current_year_tco2e: Current year total emissions.

        Returns:
            OrganicGrowthAnalysis with separated components.
        """
        by_orig = _decimal(base_year_original_tco2e)
        by_adj = _decimal(base_year_adjusted_tco2e)
        cy = _decimal(current_year_tco2e)

        structural = _round2(by_adj - by_orig)
        organic = _round2(cy - by_adj)
        total = _round2(cy - by_orig)

        organic_pct = _round2(
            _safe_divide(organic, by_adj) * Decimal("100")
        ) if by_adj != Decimal("0") else Decimal("0")

        structural_pct = _round2(
            _safe_divide(structural, by_orig) * Decimal("100")
        ) if by_orig != Decimal("0") else Decimal("0")

        total_pct = _round2(
            _safe_divide(total, by_orig) * Decimal("100")
        ) if by_orig != Decimal("0") else Decimal("0")

        analysis = OrganicGrowthAnalysis(
            reporting_year=reporting_year,
            base_year=base_year,
            base_year_original_tco2e=by_orig,
            base_year_adjusted_tco2e=by_adj,
            current_year_tco2e=cy,
            structural_change_tco2e=structural,
            organic_change_tco2e=organic,
            total_change_tco2e=total,
            organic_change_pct=organic_pct,
            structural_change_pct=structural_pct,
            total_change_pct=total_pct,
        )
        analysis.provenance_hash = _compute_hash(analysis)

        logger.info(
            "Growth separation: structural=%s tCO2e (%s%%), "
            "organic=%s tCO2e (%s%%), total=%s tCO2e (%s%%).",
            structural, structural_pct,
            organic, organic_pct,
            total, total_pct,
        )
        return analysis

    # ------------------------------------------------------------------
    # M&A Timeline
    # ------------------------------------------------------------------

    def get_mna_timeline(
        self,
        reporting_year: Optional[int] = None,
        entity_id: Optional[str] = None,
    ) -> List[MnAEvent]:
        """Get M&A events sorted by effective date.

        Args:
            reporting_year: Filter by year.
            entity_id: Filter by entity.

        Returns:
            List of MnAEvents sorted chronologically.
        """
        events = list(self._events.values())

        if reporting_year is not None:
            events = [
                e for e in events if e.reporting_year == reporting_year
            ]
        if entity_id is not None:
            events = [
                e for e in events
                if e.entity_id == entity_id
                or e.counterparty_entity_id == entity_id
            ]

        events.sort(key=lambda e: e.effective_date)
        logger.info("M&A timeline: %d event(s) returned.", len(events))
        return events

    def get_structural_change_record(
        self,
        reporting_year: int,
    ) -> StructuralChangeRecord:
        """Get a summary of all structural changes in a year.

        Args:
            reporting_year: The year to summarise.

        Returns:
            StructuralChangeRecord with all events and impacts.
        """
        events = [
            e for e in self._events.values()
            if e.reporting_year == reporting_year
        ]
        prorates = [
            p for p in self._prorates.values()
            if p.reporting_year == reporting_year
        ]
        restatements = [
            r for r in self._restatements.values()
            if any(
                self._events[r.event_id].reporting_year == reporting_year
                for eid in [r.event_id]
                if r.event_id in self._events
            )
        ]

        acq = sum(
            1 for e in events
            if e.event_type == MnAEventType.ACQUISITION.value
        )
        div = sum(
            1 for e in events
            if e.event_type == MnAEventType.DIVESTITURE.value
        )
        mer = sum(
            1 for e in events
            if e.event_type == MnAEventType.MERGER.value
        )
        other = len(events) - acq - div - mer

        net_impact = Decimal("0")
        for p in prorates:
            event = self._events.get(p.event_id)
            if event and event.event_type in (
                MnAEventType.ACQUISITION.value,
                MnAEventType.MERGER.value,
                MnAEventType.JV_FORMATION.value,
                MnAEventType.INSOURCING.value,
            ):
                net_impact += p.prorated_emissions_tco2e
            else:
                net_impact -= p.prorated_emissions_tco2e

        record = StructuralChangeRecord(
            reporting_year=reporting_year,
            total_events=len(events),
            acquisitions_count=acq,
            divestitures_count=div,
            mergers_count=mer,
            other_count=other,
            net_emission_impact_tco2e=_round2(net_impact),
            events=events,
            prorate_calculations=prorates,
            restatements=restatements,
        )
        record.provenance_hash = _compute_hash(record)

        logger.info(
            "Structural change record for %d: %d event(s), "
            "net impact=%s tCO2e.",
            reporting_year, len(events), record.net_emission_impact_tco2e,
        )
        return record

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_event(self, event_id: str) -> MnAEvent:
        """Retrieve an M&A event by ID.

        Args:
            event_id: The event ID.

        Returns:
            The MnAEvent.

        Raises:
            KeyError: If not found.
        """
        if event_id not in self._events:
            raise KeyError(f"M&A event '{event_id}' not found.")
        return self._events[event_id]

    def get_all_events(self) -> List[MnAEvent]:
        """Return all registered M&A events.

        Returns:
            List of all MnAEvents.
        """
        return list(self._events.values())

    def get_prorate(self, calculation_id: str) -> ProRataCalculation:
        """Retrieve a pro-rata calculation by ID.

        Args:
            calculation_id: The calculation ID.

        Returns:
            The ProRataCalculation.

        Raises:
            KeyError: If not found.
        """
        if calculation_id not in self._prorates:
            raise KeyError(
                f"Pro-rata calculation '{calculation_id}' not found."
            )
        return self._prorates[calculation_id]

    def get_restatement(self, restatement_id: str) -> BaseYearRestatement:
        """Retrieve a base year restatement by ID.

        Args:
            restatement_id: The restatement ID.

        Returns:
            The BaseYearRestatement.

        Raises:
            KeyError: If not found.
        """
        if restatement_id not in self._restatements:
            raise KeyError(
                f"Restatement '{restatement_id}' not found."
            )
        return self._restatements[restatement_id]

    def get_all_restatements(self) -> List[BaseYearRestatement]:
        """Return all base year restatements.

        Returns:
            List of all BaseYearRestatements.
        """
        return list(self._restatements.values())
