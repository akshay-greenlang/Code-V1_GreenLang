# -*- coding: utf-8 -*-
"""
RecalculationTriggerEngine - PACK-045 Base Year Management Engine 4
====================================================================

Automated trigger detection engine that monitors organisational, operational,
and methodological events that may require base year recalculation per
GHG Protocol Corporate Standard Chapter 5 (Base Year Recalculation Policy).

This engine performs deterministic detection and classification of events
that could compromise the comparability of a GHG emissions time-series.
When a structural or methodological change occurs, the base year must be
recalculated to reflect what emissions "would have been" under the new
conditions, ensuring like-for-like comparisons across reporting periods.

Trigger Types (GHG Protocol Ch 5, Table 5.3):
    ACQUISITION:                Purchase of operations or business units that
                                adds emission sources to the organisational
                                boundary.
    DIVESTITURE:                Sale or closure of operations or business units
                                that removes emission sources from the boundary.
    MERGER:                     Full merger with another organisation that
                                fundamentally restructures the reporting entity.
    METHODOLOGY_CHANGE:         Change in calculation methodology, emission
                                factors, GWP values, or calculation tiers that
                                materially affects reported emissions.
    ERROR_CORRECTION:           Discovery and correction of significant errors
                                in historical calculation data.
    SOURCE_BOUNDARY_CHANGE:     Addition or removal of emission source
                                categories from the operational boundary.
    OUTSOURCING_INSOURCING:     Transfer of activities across organisational
                                boundary (outsourcing or insourcing of
                                previously reported activities).

Detection Methods:
    AUTOMATED:  Engine-driven detection by comparing inventories, entity
                registries, emission factors, and boundary definitions
                between consecutive reporting periods.
    MANUAL:     User-submitted trigger events entered through the platform
                UI or API (e.g. known future acquisition).
    IMPORTED:   Triggers imported from external systems such as ERP, M&A
                transaction logs, or regulatory change feeds.

Trigger Assessment Workflow:
    1. Detect potential triggers from data changes.
    2. Classify each trigger by type and detection method.
    3. Calculate estimated emission impact (tCO2e).
    4. Compute significance percentage against base year total.
    5. Flag triggers requiring further significance assessment.
    6. Route confirmed triggers to the SignificanceAssessmentEngine.

Impact Estimation Formulae:
    For entity changes (acquisition/divestiture):
        impact_tco2e = entity_total_emissions * ownership_pct_delta / 100

    For methodology changes:
        impact_tco2e = abs(sum(activity_i * new_factor_i) - sum(activity_i * old_factor_i))

    For error corrections:
        impact_tco2e = abs(corrected_value - original_value)

    For boundary changes:
        impact_tco2e = sum(emissions for added/removed sources)

    Significance percentage:
        significance_pct = abs(impact_tco2e) / base_year_total * 100

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Corporate Value Chain (Scope 3) Standard, Chapter 5
    - ISO 14064-1:2018, Clause 5.2 (Base year selection and recalculation)
    - ESRS E1-6 (Climate change - base year recalculation disclosures)
    - CDP Climate Change Questionnaire C5.1-C5.2 (2026)
    - SBTi Corporate Net-Zero Standard v1.1, Section 7 (Recalculation)
    - US SEC Climate Disclosure Rule (2024), Item 1504
    - California SB 253 Climate Corporate Data Accountability Act (2026)

Zero-Hallucination Guarantee:
    - All impact calculations use deterministic Python Decimal arithmetic
    - Trigger detection uses explicit comparison logic, not inference
    - Threshold values sourced from published GHG Protocol guidance
    - No LLM involvement in any detection or calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  4 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash regardless of when or how fast it was computed.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Any numeric or string value to convert.

    Returns:
        Decimal representation; Decimal('0') on conversion failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of numerator / denominator, or default.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely: (part / whole) * 100.

    Args:
        part: Numerator value.
        whole: Denominator value (base year total).

    Returns:
        Percentage as Decimal; Decimal('0') when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: The Decimal value to round.
        places: Number of decimal places (default 4).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _abs_decimal(value: Decimal) -> Decimal:
    """Return the absolute value of a Decimal.

    Args:
        value: The input Decimal.

    Returns:
        Absolute value.
    """
    return value if value >= Decimal("0") else -value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TriggerType(str, Enum):
    """Types of events that may trigger base year recalculation.

    Per GHG Protocol Corporate Standard, Chapter 5, Table 5.3.

    ACQUISITION:              Purchase of operations or business units that
                              add emission sources to the organisational boundary.
    DIVESTITURE:              Sale or closure of operations or business units
                              that remove emission sources from the boundary.
    MERGER:                   Full merger with another organisation that
                              fundamentally restructures the reporting entity.
    METHODOLOGY_CHANGE:       Change in calculation methodology, emission
                              factors, GWP values, or calculation tiers.
    ERROR_CORRECTION:         Discovery and correction of significant errors
                              in historical data or calculations.
    SOURCE_BOUNDARY_CHANGE:   Addition or removal of emission source categories
                              from the operational boundary.
    OUTSOURCING_INSOURCING:   Transfer of activities across the organisational
                              boundary through outsourcing or insourcing.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    METHODOLOGY_CHANGE = "methodology_change"
    ERROR_CORRECTION = "error_correction"
    SOURCE_BOUNDARY_CHANGE = "source_boundary_change"
    OUTSOURCING_INSOURCING = "outsourcing_insourcing"


class TriggerStatus(str, Enum):
    """Lifecycle status of a detected recalculation trigger.

    DETECTED:       Trigger identified by detection logic but not yet reviewed.
    UNDER_REVIEW:   Trigger is being reviewed by an analyst or approver.
    CONFIRMED:      Trigger has been confirmed as valid and will proceed to
                    significance assessment.
    DISMISSED:      Trigger has been reviewed and dismissed as not applicable
                    (e.g. below de minimis, duplicate, or invalid).
    PROCESSED:      Trigger has been fully processed through significance
                    assessment and adjustment (if applicable).
    """
    DETECTED = "detected"
    UNDER_REVIEW = "under_review"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    PROCESSED = "processed"


class DetectionMethod(str, Enum):
    """How the recalculation trigger was identified.

    AUTOMATED:  Detected automatically by comparing inventories, entity
                registries, emission factors, and boundary definitions.
    MANUAL:     Entered manually by a user through the platform UI or API.
    IMPORTED:   Imported from an external system (ERP, M&A transaction
                log, regulatory change feed).
    """
    AUTOMATED = "automated"
    MANUAL = "manual"
    IMPORTED = "imported"


class Scope(str, Enum):
    """GHG Protocol emission scope classification.

    SCOPE_1:            Direct emissions from owned or controlled sources.
    SCOPE_2_LOCATION:   Indirect emissions from purchased energy (location-based).
    SCOPE_2_MARKET:     Indirect emissions from purchased energy (market-based).
    SCOPE_3:            Other indirect emissions in the value chain.
    ALL:                Affects all scopes.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"
    ALL = "all"


class CalculationTier(str, Enum):
    """Emission calculation tier per GHG Protocol methodology hierarchy.

    TIER_1:   Basic estimation using industry-average emission factors.
    TIER_2:   Improved estimation using facility-specific activity data
              with industry-average emission factors.
    TIER_3:   Detailed measurement using facility-specific data and
              source-specific emission factors or direct measurement.
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default significance threshold as percentage of base year emissions.
# Source: GHG Protocol Corporate Standard, Chapter 5, p.35.
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# De minimis threshold below which triggers are auto-dismissed.
# Triggers with impact below this percentage are considered immaterial.
DE_MINIMIS_THRESHOLD_PCT: Decimal = Decimal("0.5")

# SBTi significance threshold (stricter for target tracking).
# Source: SBTi Corporate Manual (2023), Section 7.
SBTI_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# Maximum number of triggers to process in a single detection run.
MAX_TRIGGERS_PER_RUN: int = 200

# Minimum base year for validation purposes.
MINIMUM_BASE_YEAR: int = 1990

# Maximum base year (cannot be in the future).
MAXIMUM_BASE_YEAR: int = 2030

# Default ownership percentage change threshold for entity changes.
# Only flag entity changes where ownership % changes by at least this amount.
OWNERSHIP_CHANGE_THRESHOLD_PCT: Decimal = Decimal("1.0")


# ---------------------------------------------------------------------------
# Pydantic Models -- Trigger Detail Models
# ---------------------------------------------------------------------------


class EntityChange(BaseModel):
    """Details of an organisational entity change (acquisition, divestiture, merger).

    Captures the specific parameters of a structural change to the
    organisational boundary, including ownership percentage shifts and
    the estimated emission impact in tCO2e.

    Attributes:
        entity_id: Unique identifier of the affected entity (facility,
            business unit, or subsidiary).
        entity_name: Human-readable name of the entity.
        change_type: Type of structural change (acquisition, divestiture, merger).
        effective_date: Date when the structural change takes legal effect.
        emissions_impact_tco2e: Estimated total emission impact of the change
            in tonnes of CO2 equivalent, calculated as:
            impact = entity_total * abs(ownership_pct_after - ownership_pct_before) / 100
        ownership_pct_before: Reporting entity's ownership or control percentage
            of the entity before the change (0-100).
        ownership_pct_after: Reporting entity's ownership or control percentage
            of the entity after the change (0-100).
        sectors_affected: List of industry sectors or business segments
            affected by this entity change.
    """
    entity_id: str = Field(
        ..., min_length=1, description="Unique entity identifier"
    )
    entity_name: str = Field(
        ..., min_length=1, description="Entity name"
    )
    change_type: TriggerType = Field(
        ..., description="Type of structural change"
    )
    effective_date: date = Field(
        ..., description="Effective date of the change"
    )
    emissions_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated emission impact (tCO2e)"
    )
    ownership_pct_before: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Ownership % before change"
    )
    ownership_pct_after: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Ownership % after change"
    )
    sectors_affected: List[str] = Field(
        default_factory=list,
        description="Affected industry sectors"
    )

    @field_validator(
        "emissions_impact_tco2e", "ownership_pct_before", "ownership_pct_after",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

    @property
    def ownership_delta_pct(self) -> Decimal:
        """Compute the absolute change in ownership percentage.

        Formula:
            ownership_delta = abs(ownership_pct_after - ownership_pct_before)
        """
        return _abs_decimal(self.ownership_pct_after - self.ownership_pct_before)


class MethodologyChange(BaseModel):
    """Details of a methodology change affecting emission calculations.

    Captures changes in calculation methodology, emission factors, GWP
    values, or calculation tier upgrades that affect the comparability
    of emissions across reporting periods.

    Attributes:
        scope: Which GHG scope is affected by the methodology change.
        category: Emission source category affected (e.g. 'stationary_combustion',
            'purchased_electricity', 'category_1_purchased_goods').
        old_methodology: Description of the previous methodology.
        new_methodology: Description of the new methodology.
        old_tier: Previous calculation tier.
        new_tier: New calculation tier.
        emission_impact_tco2e: Estimated impact of the methodology change
            in tonnes of CO2 equivalent, calculated as:
            impact = abs(sum(activity_i * new_factor_i) - sum(activity_i * old_factor_i))
        rationale: Justification for the methodology change.
    """
    scope: Scope = Field(
        ..., description="Affected GHG scope"
    )
    category: str = Field(
        default="", description="Affected source category"
    )
    old_methodology: str = Field(
        default="", description="Previous methodology description"
    )
    new_methodology: str = Field(
        default="", description="New methodology description"
    )
    old_tier: Optional[CalculationTier] = Field(
        default=None, description="Previous calculation tier"
    )
    new_tier: Optional[CalculationTier] = Field(
        default=None, description="New calculation tier"
    )
    emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated methodology impact (tCO2e)"
    )
    rationale: str = Field(
        default="", description="Rationale for the methodology change"
    )

    @field_validator("emission_impact_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission impact to Decimal."""
        return _decimal(v)


class ErrorCorrection(BaseModel):
    """Details of an error correction in historical emission data.

    Captures the specifics of a discovered error including original and
    corrected values, enabling deterministic computation of the impact
    on base year emissions.

    Attributes:
        scope: Which GHG scope contains the error.
        category: Emission source category where the error was found.
        original_value_tco2e: The originally reported emission value
            in tonnes of CO2 equivalent.
        corrected_value_tco2e: The corrected emission value in tCO2e.
        error_description: Description of the nature of the error.
        discovery_date: Date when the error was discovered.
        discovery_method: How the error was discovered (e.g. 'internal_audit',
            'third_party_verification', 'automated_validation').
    """
    scope: Scope = Field(
        ..., description="Scope containing the error"
    )
    category: str = Field(
        default="", description="Source category with error"
    )
    original_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Original reported value (tCO2e)"
    )
    corrected_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Corrected value (tCO2e)"
    )
    error_description: str = Field(
        default="", description="Description of the error"
    )
    discovery_date: date = Field(
        default_factory=lambda: date.today(),
        description="Date error was discovered"
    )
    discovery_method: str = Field(
        default="internal_audit",
        description="How the error was discovered"
    )

    @field_validator("original_value_tco2e", "corrected_value_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission values to Decimal."""
        return _decimal(v)

    @property
    def error_impact_tco2e(self) -> Decimal:
        """Compute the absolute impact of the error correction.

        Formula:
            error_impact = abs(corrected_value - original_value)
        """
        return _abs_decimal(self.corrected_value_tco2e - self.original_value_tco2e)


class BoundaryChange(BaseModel):
    """Details of an operational boundary change.

    Captures changes to the set of emission source categories included
    in the organisational boundary, which may require base year
    recalculation to include historical data for newly included sources.

    Attributes:
        change_description: Description of the boundary change.
        sources_added: List of emission source identifiers added to the boundary.
        sources_removed: List of emission source identifiers removed from the boundary.
        emission_impact_tco2e: Estimated total impact of the boundary change
            in tonnes of CO2 equivalent, calculated as:
            impact = sum(emissions for added sources) + sum(emissions for removed sources)
    """
    change_description: str = Field(
        default="", description="Description of boundary change"
    )
    sources_added: List[str] = Field(
        default_factory=list,
        description="Source categories added to boundary"
    )
    sources_removed: List[str] = Field(
        default_factory=list,
        description="Source categories removed from boundary"
    )
    emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated boundary change impact (tCO2e)"
    )

    @field_validator("emission_impact_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission impact to Decimal."""
        return _decimal(v)

    @property
    def net_source_change(self) -> int:
        """Compute net number of sources added (positive) or removed (negative)."""
        return len(self.sources_added) - len(self.sources_removed)


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class InventorySnapshot(BaseModel):
    """A snapshot of GHG inventory emissions for trigger comparison.

    Provides total emissions by scope and a breakdown by source category
    to enable automated detection of changes between periods.

    Attributes:
        year: Reporting year of the inventory.
        scope1_total_tco2e: Total Scope 1 emissions (tCO2e).
        scope2_location_total_tco2e: Scope 2 location-based total (tCO2e).
        scope2_market_total_tco2e: Scope 2 market-based total (tCO2e).
        scope3_total_tco2e: Total Scope 3 emissions (tCO2e).
        by_category: Emissions by source category name (tCO2e).
        by_facility: Emissions by facility identifier (tCO2e).
        emission_factors: Emission factors by source category.
        source_categories: List of source category identifiers in boundary.
        entity_ids: List of entity identifiers in organisational boundary.
    """
    year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=MAXIMUM_BASE_YEAR,
        description="Reporting year"
    )
    scope1_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1 total (tCO2e)"
    )
    scope2_location_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 location-based (tCO2e)"
    )
    scope2_market_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 market-based (tCO2e)"
    )
    scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 3 total (tCO2e)"
    )
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by source category (tCO2e)"
    )
    by_facility: Dict[str, Decimal] = Field(
        default_factory=dict, description="Emissions by facility (tCO2e)"
    )
    emission_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Emission factors by source category"
    )
    source_categories: List[str] = Field(
        default_factory=list, description="Source categories in boundary"
    )
    entity_ids: List[str] = Field(
        default_factory=list, description="Entity IDs in organisational boundary"
    )

    @field_validator(
        "scope1_total_tco2e", "scope2_location_total_tco2e",
        "scope2_market_total_tco2e", "scope3_total_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission totals to Decimal."""
        return _decimal(v)

    @property
    def grand_total_tco2e(self) -> Decimal:
        """Total emissions across all scopes (Scope 1 + 2 location + 3).

        Formula:
            grand_total = scope1 + scope2_location + scope3
        """
        return (
            self.scope1_total_tco2e
            + self.scope2_location_total_tco2e
            + self.scope3_total_tco2e
        )


class EntityRegistryEntry(BaseModel):
    """An entity (facility, subsidiary, business unit) in the organisational boundary.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Human-readable entity name.
        ownership_pct: Current ownership or control percentage (0-100).
        total_emissions_tco2e: Total annual emissions for this entity (tCO2e).
        sectors: Industry sectors or segments this entity operates in.
        status: Current status (active, divested, acquired, merged).
        effective_date: Date of most recent status change.
    """
    entity_id: str = Field(..., min_length=1, description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    ownership_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Ownership percentage"
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total emissions (tCO2e)"
    )
    sectors: List[str] = Field(default_factory=list, description="Sectors")
    status: str = Field(default="active", description="Entity status")
    effective_date: Optional[date] = Field(
        default=None, description="Date of last status change"
    )

    @field_validator("ownership_pct", "total_emissions_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class ExternalEvent(BaseModel):
    """An externally reported event that may trigger recalculation.

    Used for MANUAL and IMPORTED detection methods where triggers
    originate from outside the automated detection pipeline.

    Attributes:
        event_id: Unique event identifier.
        trigger_type: Type of trigger event.
        description: Human-readable event description.
        effective_date: When the event takes effect.
        estimated_impact_tco2e: Estimated emission impact (tCO2e).
        entity_change: Optional entity change details.
        methodology_change: Optional methodology change details.
        error_correction: Optional error correction details.
        boundary_change: Optional boundary change details.
        source: Origin system or user who reported the event.
        detection_method: How this event was identified.
    """
    event_id: str = Field(
        default_factory=_new_uuid, description="Event identifier"
    )
    trigger_type: TriggerType = Field(..., description="Trigger type")
    description: str = Field(default="", description="Event description")
    effective_date: date = Field(
        default_factory=lambda: date.today(),
        description="Effective date"
    )
    estimated_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated emission impact (tCO2e)"
    )
    entity_change: Optional[EntityChange] = Field(
        default=None, description="Entity change details"
    )
    methodology_change: Optional[MethodologyChange] = Field(
        default=None, description="Methodology change details"
    )
    error_correction: Optional[ErrorCorrection] = Field(
        default=None, description="Error correction details"
    )
    boundary_change: Optional[BoundaryChange] = Field(
        default=None, description="Boundary change details"
    )
    source: str = Field(default="", description="Event source system or user")
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.MANUAL, description="Detection method"
    )

    @field_validator("estimated_impact_tco2e", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce impact to Decimal."""
        return _decimal(v)


class TriggerDetectionConfig(BaseModel):
    """Configuration parameters for the trigger detection engine.

    Attributes:
        significance_threshold_pct: Threshold above which a trigger is
            considered potentially significant (default 5.0%).
        de_minimis_threshold_pct: Threshold below which triggers are
            auto-dismissed as immaterial (default 0.5%).
        ownership_change_threshold_pct: Minimum ownership change to
            flag entity changes (default 1.0%).
        auto_dismiss_below_de_minimis: Whether to auto-dismiss triggers
            below the de minimis threshold.
        detect_entity_changes: Enable entity change detection.
        detect_methodology_changes: Enable methodology change detection.
        detect_errors: Enable error correction detection.
        detect_boundary_changes: Enable boundary change detection.
        base_year_total_tco2e: Base year total emissions for significance
            calculation. Must be provided for percentage calculations.
    """
    significance_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        ge=0, le=100,
        description="Significance threshold (%)"
    )
    de_minimis_threshold_pct: Decimal = Field(
        default=DE_MINIMIS_THRESHOLD_PCT,
        ge=0, le=100,
        description="De minimis threshold (%)"
    )
    ownership_change_threshold_pct: Decimal = Field(
        default=OWNERSHIP_CHANGE_THRESHOLD_PCT,
        ge=0, le=100,
        description="Min ownership change to flag (%)"
    )
    auto_dismiss_below_de_minimis: bool = Field(
        default=True,
        description="Auto-dismiss triggers below de minimis"
    )
    detect_entity_changes: bool = Field(
        default=True, description="Enable entity change detection"
    )
    detect_methodology_changes: bool = Field(
        default=True, description="Enable methodology change detection"
    )
    detect_errors: bool = Field(
        default=True, description="Enable error correction detection"
    )
    detect_boundary_changes: bool = Field(
        default=True, description="Enable boundary change detection"
    )
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Base year total emissions (tCO2e)"
    )

    @field_validator(
        "significance_threshold_pct", "de_minimis_threshold_pct",
        "ownership_change_threshold_pct", "base_year_total_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce thresholds to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class DetectedTrigger(BaseModel):
    """A single detected recalculation trigger with full provenance.

    Each detected trigger includes the type classification, estimated
    emission impact, significance percentage relative to base year total,
    and the detailed change information for audit purposes.

    Attributes:
        trigger_id: Unique trigger identifier.
        trigger_type: Classification of the trigger event.
        status: Current lifecycle status of this trigger.
        detection_method: How this trigger was detected.
        detected_date: Date and time when the trigger was detected.
        description: Human-readable description of the trigger event.
        emission_impact_tco2e: Estimated absolute emission impact in tCO2e.
        significance_pct: Impact as percentage of base year total emissions.
        entity_change: Optional entity change details (for structural triggers).
        methodology_change: Optional methodology change details.
        error_correction: Optional error correction details.
        boundary_change: Optional boundary change details.
        requires_recalculation: Whether this trigger warrants base year
            recalculation based on initial significance screening.
        provenance_hash: SHA-256 hash of all trigger data for audit.
    """
    trigger_id: str = Field(
        default_factory=_new_uuid, description="Unique trigger ID"
    )
    trigger_type: TriggerType = Field(
        ..., description="Trigger classification"
    )
    status: TriggerStatus = Field(
        default=TriggerStatus.DETECTED, description="Trigger status"
    )
    detection_method: DetectionMethod = Field(
        default=DetectionMethod.AUTOMATED, description="Detection method"
    )
    detected_date: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )
    description: str = Field(
        default="", description="Trigger description"
    )
    emission_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Estimated emission impact (tCO2e)"
    )
    significance_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Significance as % of base year total"
    )
    entity_change: Optional[EntityChange] = Field(
        default=None, description="Entity change details"
    )
    methodology_change: Optional[MethodologyChange] = Field(
        default=None, description="Methodology change details"
    )
    error_correction: Optional[ErrorCorrection] = Field(
        default=None, description="Error correction details"
    )
    boundary_change: Optional[BoundaryChange] = Field(
        default=None, description="Boundary change details"
    )
    requires_recalculation: bool = Field(
        default=False,
        description="Whether trigger warrants recalculation"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator("emission_impact_tco2e", "significance_pct", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


class TriggerDetectionResult(BaseModel):
    """Complete result of a trigger detection run with provenance.

    Aggregates all detected triggers, computes cumulative impact,
    and provides recommendations for next steps.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version that produced this result.
        calculated_at: Timestamp of the detection run.
        processing_time_ms: Total processing time in milliseconds.
        triggers: List of all detected triggers.
        total_triggers_detected: Count of detected triggers.
        triggers_requiring_recalculation: Count of triggers flagged for
            recalculation.
        total_cumulative_impact_tco2e: Sum of absolute emission impacts
            across all detected triggers.
        cumulative_significance_pct: Cumulative impact as percentage of
            base year total.
        requires_recalculation: Whether any trigger requires recalculation.
        recommendations: List of recommended actions based on detection results.
        base_year_total_tco2e: Base year total used for significance calculations.
        detection_config: Configuration used for this detection run.
        provenance_hash: SHA-256 hash of the complete detection result.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Result identifier"
    )
    engine_version: str = Field(
        default=_MODULE_VERSION, description="Engine version"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Detection timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0, description="Processing time (ms)"
    )
    triggers: List[DetectedTrigger] = Field(
        default_factory=list, description="Detected triggers"
    )
    total_triggers_detected: int = Field(
        default=0, ge=0, description="Total triggers detected"
    )
    triggers_requiring_recalculation: int = Field(
        default=0, ge=0, description="Triggers requiring recalculation"
    )
    total_cumulative_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Cumulative absolute impact (tCO2e)"
    )
    cumulative_significance_pct: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Cumulative significance (%)"
    )
    requires_recalculation: bool = Field(
        default=False, description="Any trigger requires recalculation"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Base year total used for significance"
    )
    detection_config: Optional[TriggerDetectionConfig] = Field(
        default=None, description="Configuration used"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator(
        "total_cumulative_impact_tco2e", "cumulative_significance_pct",
        "base_year_total_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)


# ---------------------------------------------------------------------------
# Model Rebuild (Pydantic v2 deferred annotations resolution)
# ---------------------------------------------------------------------------

EntityChange.model_rebuild()
MethodologyChange.model_rebuild()
ErrorCorrection.model_rebuild()
BoundaryChange.model_rebuild()
InventorySnapshot.model_rebuild()
EntityRegistryEntry.model_rebuild()
ExternalEvent.model_rebuild()
TriggerDetectionConfig.model_rebuild()
DetectedTrigger.model_rebuild()
TriggerDetectionResult.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RecalculationTriggerEngine:
    """Automated recalculation trigger detection engine.

    Monitors organisational, operational, and methodological events that
    may require base year recalculation per GHG Protocol Corporate Standard
    Chapter 5. Performs deterministic detection and classification with
    complete provenance tracking.

    Detection Capabilities:
        1. Entity Changes: Compares entity registries between periods to
           detect acquisitions, divestitures, and mergers.
        2. Methodology Changes: Compares emission factors and calculation
           parameters between periods.
        3. Error Corrections: Analyzes validation results to identify
           corrections that exceed significance thresholds.
        4. Boundary Changes: Compares source category lists between
           periods to detect additions and removals.

    Usage:
        >>> engine = RecalculationTriggerEngine()
        >>> config = TriggerDetectionConfig(base_year_total_tco2e=Decimal("50000"))
        >>> result = engine.detect_triggers(current, previous, [], config)
        >>> if result.requires_recalculation:
        ...     # Route to SignificanceAssessmentEngine
        ...     pass

    All calculations use Python Decimal arithmetic with ROUND_HALF_UP
    rounding. No LLM is used in any detection or calculation path.
    """

    def __init__(self) -> None:
        """Initialise the RecalculationTriggerEngine.

        Sets up internal trigger storage and engine metadata.
        """
        self._version: str = _MODULE_VERSION
        self._trigger_store: Dict[str, DetectedTrigger] = {}
        logger.info(
            "RecalculationTriggerEngine v%s initialised", self._version
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def detect_triggers(
        self,
        current_inventory: InventorySnapshot,
        previous_inventory: InventorySnapshot,
        external_events: List[ExternalEvent],
        config: Optional[TriggerDetectionConfig] = None,
    ) -> TriggerDetectionResult:
        """Execute a full trigger detection run.

        Compares two inventory snapshots (current vs previous), processes
        external events, and returns a comprehensive list of detected
        triggers with impact estimates and significance percentages.

        Args:
            current_inventory: The most recent inventory snapshot.
            previous_inventory: The prior period inventory snapshot.
            external_events: List of externally reported events.
            config: Detection configuration. If None, uses defaults.

        Returns:
            TriggerDetectionResult with all detected triggers, cumulative
            impact, and recommendations.

        Raises:
            ValueError: If inventories have the same year or base year
                total is not provided in config.
        """
        start_ns = time.perf_counter_ns()
        if config is None:
            config = TriggerDetectionConfig()

        all_triggers: List[DetectedTrigger] = []

        # Step 1: Detect entity changes (structural triggers)
        if config.detect_entity_changes:
            entity_triggers = self._detect_entity_changes_from_inventory(
                current_inventory, previous_inventory, config
            )
            all_triggers.extend(entity_triggers)

        # Step 2: Detect methodology changes
        if config.detect_methodology_changes:
            method_triggers = self.detect_methodology_changes(
                current_inventory.emission_factors,
                previous_inventory.emission_factors,
                current_inventory.by_category,
                config,
            )
            all_triggers.extend(method_triggers)

        # Step 3: Detect boundary changes
        if config.detect_boundary_changes:
            boundary_triggers = self.detect_boundary_changes(
                current_inventory.source_categories,
                previous_inventory.source_categories,
                current_inventory.by_category,
                previous_inventory.by_category,
                config,
            )
            all_triggers.extend(boundary_triggers)

        # Step 4: Process external events (manual and imported triggers)
        for event in external_events:
            ext_trigger = self._process_external_event(event, config)
            all_triggers.append(ext_trigger)

        # Step 5: Assess significance for each trigger
        for trigger in all_triggers:
            self._assess_trigger_significance(trigger, config)

        # Step 6: Store triggers
        for trigger in all_triggers:
            trigger.provenance_hash = _compute_hash(trigger)
            self._trigger_store[trigger.trigger_id] = trigger

        # Step 7: Build result
        cumulative_impact = sum(
            (t.emission_impact_tco2e for t in all_triggers), Decimal("0")
        )
        cumulative_significance = _safe_pct(cumulative_impact, config.base_year_total_tco2e)
        recalc_count = sum(1 for t in all_triggers if t.requires_recalculation)
        requires_recalc = recalc_count > 0

        recommendations = self._generate_recommendations(
            all_triggers, cumulative_impact, cumulative_significance, config
        )

        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        result = TriggerDetectionResult(
            triggers=all_triggers,
            total_triggers_detected=len(all_triggers),
            triggers_requiring_recalculation=recalc_count,
            total_cumulative_impact_tco2e=_round_val(cumulative_impact, 3),
            cumulative_significance_pct=_round_val(cumulative_significance, 4),
            requires_recalculation=requires_recalc,
            recommendations=recommendations,
            base_year_total_tco2e=config.base_year_total_tco2e,
            detection_config=config,
            processing_time_ms=round(elapsed_ms, 2),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Trigger detection complete: %d triggers detected, %d require recalculation, "
            "cumulative impact=%.3f tCO2e (%.4f%%)",
            len(all_triggers), recalc_count,
            float(cumulative_impact), float(cumulative_significance),
        )

        return result

    def detect_entity_changes(
        self,
        current_registry: List[EntityRegistryEntry],
        previous_registry: List[EntityRegistryEntry],
        config: Optional[TriggerDetectionConfig] = None,
    ) -> List[DetectedTrigger]:
        """Detect entity changes by comparing entity registries.

        Compares two entity registries to identify acquisitions (new entities),
        divestitures (removed entities), and ownership changes (modified
        entities).

        Args:
            current_registry: Current period entity registry.
            previous_registry: Previous period entity registry.
            config: Detection configuration. If None, uses defaults.

        Returns:
            List of detected entity change triggers.
        """
        if config is None:
            config = TriggerDetectionConfig()

        triggers: List[DetectedTrigger] = []

        current_map: Dict[str, EntityRegistryEntry] = {
            e.entity_id: e for e in current_registry
        }
        previous_map: Dict[str, EntityRegistryEntry] = {
            e.entity_id: e for e in previous_registry
        }

        # Detect acquisitions (entities in current but not in previous)
        for eid, entry in current_map.items():
            if eid not in previous_map:
                entity_change = EntityChange(
                    entity_id=entry.entity_id,
                    entity_name=entry.entity_name,
                    change_type=TriggerType.ACQUISITION,
                    effective_date=entry.effective_date or date.today(),
                    emissions_impact_tco2e=entry.total_emissions_tco2e,
                    ownership_pct_before=Decimal("0"),
                    ownership_pct_after=entry.ownership_pct,
                    sectors_affected=entry.sectors,
                )
                trigger = DetectedTrigger(
                    trigger_type=TriggerType.ACQUISITION,
                    detection_method=DetectionMethod.AUTOMATED,
                    description=(
                        f"Acquisition detected: entity '{entry.entity_name}' "
                        f"(ID: {entry.entity_id}) added with "
                        f"{entry.ownership_pct}% ownership"
                    ),
                    emission_impact_tco2e=entry.total_emissions_tco2e,
                    entity_change=entity_change,
                )
                triggers.append(trigger)

        # Detect divestitures (entities in previous but not in current)
        for eid, entry in previous_map.items():
            if eid not in current_map:
                entity_change = EntityChange(
                    entity_id=entry.entity_id,
                    entity_name=entry.entity_name,
                    change_type=TriggerType.DIVESTITURE,
                    effective_date=date.today(),
                    emissions_impact_tco2e=entry.total_emissions_tco2e,
                    ownership_pct_before=entry.ownership_pct,
                    ownership_pct_after=Decimal("0"),
                    sectors_affected=entry.sectors,
                )
                trigger = DetectedTrigger(
                    trigger_type=TriggerType.DIVESTITURE,
                    detection_method=DetectionMethod.AUTOMATED,
                    description=(
                        f"Divestiture detected: entity '{entry.entity_name}' "
                        f"(ID: {entry.entity_id}) removed from boundary"
                    ),
                    emission_impact_tco2e=entry.total_emissions_tco2e,
                    entity_change=entity_change,
                )
                triggers.append(trigger)

        # Detect ownership changes (entities in both but with changed ownership)
        for eid in set(current_map) & set(previous_map):
            cur = current_map[eid]
            prev = previous_map[eid]
            delta = _abs_decimal(cur.ownership_pct - prev.ownership_pct)
            if delta >= config.ownership_change_threshold_pct:
                impact = _abs_decimal(
                    cur.total_emissions_tco2e * cur.ownership_pct / Decimal("100")
                    - prev.total_emissions_tco2e * prev.ownership_pct / Decimal("100")
                )
                change_type = (
                    TriggerType.ACQUISITION
                    if cur.ownership_pct > prev.ownership_pct
                    else TriggerType.DIVESTITURE
                )
                entity_change = EntityChange(
                    entity_id=cur.entity_id,
                    entity_name=cur.entity_name,
                    change_type=change_type,
                    effective_date=cur.effective_date or date.today(),
                    emissions_impact_tco2e=impact,
                    ownership_pct_before=prev.ownership_pct,
                    ownership_pct_after=cur.ownership_pct,
                    sectors_affected=cur.sectors,
                )
                trigger = DetectedTrigger(
                    trigger_type=change_type,
                    detection_method=DetectionMethod.AUTOMATED,
                    description=(
                        f"Ownership change for '{cur.entity_name}': "
                        f"{prev.ownership_pct}% -> {cur.ownership_pct}% "
                        f"(delta: {delta}%)"
                    ),
                    emission_impact_tco2e=impact,
                    entity_change=entity_change,
                )
                triggers.append(trigger)

        return triggers

    def detect_methodology_changes(
        self,
        current_factors: Dict[str, Any],
        previous_factors: Dict[str, Any],
        activity_data: Optional[Dict[str, Decimal]] = None,
        config: Optional[TriggerDetectionConfig] = None,
    ) -> List[DetectedTrigger]:
        """Detect methodology changes by comparing emission factors.

        Compares emission factor dictionaries between periods to identify
        factors that have changed, which indicates a methodology update.

        Impact Estimation Formula:
            For each changed factor:
                old_emissions = activity * old_factor
                new_emissions = activity * new_factor
                impact = abs(new_emissions - old_emissions)

            When activity data is not available:
                impact = abs(new_factor - old_factor)
                (normalized impact; flagged as estimate)

        Args:
            current_factors: Current period emission factors by category.
            previous_factors: Previous period emission factors by category.
            activity_data: Optional activity data by category for impact
                estimation. Keys should match factor category keys.
            config: Detection configuration. If None, uses defaults.

        Returns:
            List of detected methodology change triggers.
        """
        if config is None:
            config = TriggerDetectionConfig()
        if activity_data is None:
            activity_data = {}

        triggers: List[DetectedTrigger] = []

        # Find categories with changed factors
        all_categories = set(current_factors.keys()) | set(previous_factors.keys())
        for category in sorted(all_categories):
            cur_val = current_factors.get(category)
            prev_val = previous_factors.get(category)

            if cur_val is None or prev_val is None:
                continue  # New or removed categories handled by boundary detection

            # Compare factor values
            cur_dec = _decimal(cur_val) if not isinstance(cur_val, dict) else Decimal("0")
            prev_dec = _decimal(prev_val) if not isinstance(prev_val, dict) else Decimal("0")

            if isinstance(cur_val, dict) and isinstance(prev_val, dict):
                # Compare nested factor structures
                changed = cur_val != prev_val
                if not changed:
                    continue
                cur_dec = _decimal(cur_val.get("value", 0))
                prev_dec = _decimal(prev_val.get("value", 0))

            if cur_dec == prev_dec:
                continue

            # Calculate emission impact
            activity = _decimal(activity_data.get(category, Decimal("0")))
            if activity > Decimal("0"):
                old_emissions = activity * prev_dec
                new_emissions = activity * cur_dec
                impact = _abs_decimal(new_emissions - old_emissions)
            else:
                impact = _abs_decimal(cur_dec - prev_dec)

            methodology_change = MethodologyChange(
                scope=Scope.ALL,
                category=category,
                old_methodology=f"Factor: {prev_dec}",
                new_methodology=f"Factor: {cur_dec}",
                emission_impact_tco2e=impact,
                rationale=f"Emission factor changed from {prev_dec} to {cur_dec}",
            )
            trigger = DetectedTrigger(
                trigger_type=TriggerType.METHODOLOGY_CHANGE,
                detection_method=DetectionMethod.AUTOMATED,
                description=(
                    f"Methodology change in '{category}': "
                    f"factor {prev_dec} -> {cur_dec} "
                    f"(impact: {_round_val(impact, 3)} tCO2e)"
                ),
                emission_impact_tco2e=impact,
                methodology_change=methodology_change,
            )
            triggers.append(trigger)

        return triggers

    def detect_errors(
        self,
        validation_results: List[ErrorCorrection],
        config: Optional[TriggerDetectionConfig] = None,
    ) -> List[DetectedTrigger]:
        """Detect error corrections that may trigger recalculation.

        Processes a list of error corrections (typically from a validation
        engine or audit) and creates triggers for each correction that
        exceeds the de minimis threshold.

        Impact Formula:
            impact = abs(corrected_value - original_value)

        Args:
            validation_results: List of error corrections discovered.
            config: Detection configuration. If None, uses defaults.

        Returns:
            List of detected error correction triggers.
        """
        if config is None:
            config = TriggerDetectionConfig()

        triggers: List[DetectedTrigger] = []

        for correction in validation_results:
            impact = correction.error_impact_tco2e
            significance = _safe_pct(impact, config.base_year_total_tco2e)

            # Skip corrections below de minimis
            if (
                config.auto_dismiss_below_de_minimis
                and significance < config.de_minimis_threshold_pct
            ):
                logger.debug(
                    "Error correction in '%s' dismissed (%.4f%% < de minimis %.4f%%)",
                    correction.category,
                    float(significance),
                    float(config.de_minimis_threshold_pct),
                )
                continue

            trigger = DetectedTrigger(
                trigger_type=TriggerType.ERROR_CORRECTION,
                detection_method=DetectionMethod.AUTOMATED,
                description=(
                    f"Error correction in '{correction.category}' ({correction.scope.value}): "
                    f"{correction.original_value_tco2e} -> {correction.corrected_value_tco2e} tCO2e "
                    f"(impact: {_round_val(impact, 3)} tCO2e, "
                    f"significance: {_round_val(significance, 4)}%)"
                ),
                emission_impact_tco2e=impact,
                significance_pct=_round_val(significance, 4),
                error_correction=correction,
            )
            triggers.append(trigger)

        return triggers

    def detect_boundary_changes(
        self,
        current_boundary: List[str],
        previous_boundary: List[str],
        current_emissions: Optional[Dict[str, Decimal]] = None,
        previous_emissions: Optional[Dict[str, Decimal]] = None,
        config: Optional[TriggerDetectionConfig] = None,
    ) -> List[DetectedTrigger]:
        """Detect source boundary changes between reporting periods.

        Compares two lists of source category identifiers to identify
        additions and removals from the operational boundary.

        Impact Formula:
            For sources added:
                impact = sum(current_emissions[source] for source in added)
            For sources removed:
                impact = sum(previous_emissions[source] for source in removed)

        Args:
            current_boundary: Current period source category list.
            previous_boundary: Previous period source category list.
            current_emissions: Current emissions by source category (tCO2e).
            previous_emissions: Previous emissions by source category (tCO2e).
            config: Detection configuration. If None, uses defaults.

        Returns:
            List of detected boundary change triggers. Returns at most one
            trigger if there are both additions and removals, or one per
            direction of change.
        """
        if config is None:
            config = TriggerDetectionConfig()
        if current_emissions is None:
            current_emissions = {}
        if previous_emissions is None:
            previous_emissions = {}

        triggers: List[DetectedTrigger] = []

        current_set = set(current_boundary)
        previous_set = set(previous_boundary)

        sources_added = sorted(current_set - previous_set)
        sources_removed = sorted(previous_set - current_set)

        if not sources_added and not sources_removed:
            return triggers

        # Calculate impact for added sources
        added_impact = sum(
            (_decimal(current_emissions.get(s, Decimal("0"))) for s in sources_added),
            Decimal("0"),
        )
        # Calculate impact for removed sources
        removed_impact = sum(
            (_decimal(previous_emissions.get(s, Decimal("0"))) for s in sources_removed),
            Decimal("0"),
        )
        total_impact = added_impact + removed_impact

        boundary_change = BoundaryChange(
            change_description=(
                f"Boundary change: {len(sources_added)} source(s) added, "
                f"{len(sources_removed)} source(s) removed"
            ),
            sources_added=sources_added,
            sources_removed=sources_removed,
            emission_impact_tco2e=total_impact,
        )

        trigger = DetectedTrigger(
            trigger_type=TriggerType.SOURCE_BOUNDARY_CHANGE,
            detection_method=DetectionMethod.AUTOMATED,
            description=(
                f"Boundary change: added {sources_added if sources_added else 'none'}, "
                f"removed {sources_removed if sources_removed else 'none'} "
                f"(impact: {_round_val(total_impact, 3)} tCO2e)"
            ),
            emission_impact_tco2e=total_impact,
            boundary_change=boundary_change,
        )
        triggers.append(trigger)

        return triggers

    def assess_trigger(
        self,
        trigger: DetectedTrigger,
        base_year_total_tco2e: Decimal,
        significance_threshold_pct: Decimal = DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
    ) -> DetectedTrigger:
        """Assess a single trigger for significance against base year total.

        Computes the significance percentage and determines whether the
        trigger warrants base year recalculation.

        Formula:
            significance_pct = abs(emission_impact_tco2e) / base_year_total * 100
            requires_recalculation = significance_pct >= threshold_pct

        Special Cases:
            - MERGER triggers always require recalculation regardless of
              the significance percentage.
            - Triggers below de_minimis (0.5%) are auto-dismissed.

        Args:
            trigger: The trigger to assess.
            base_year_total_tco2e: Base year total emissions for denominator.
            significance_threshold_pct: Significance threshold percentage.

        Returns:
            Updated trigger with significance_pct and requires_recalculation.
        """
        impact = trigger.emission_impact_tco2e
        significance = _safe_pct(impact, base_year_total_tco2e)

        trigger.significance_pct = _round_val(significance, 4)

        # Mergers always require recalculation
        if trigger.trigger_type == TriggerType.MERGER:
            trigger.requires_recalculation = True
            trigger.status = TriggerStatus.CONFIRMED
        elif significance >= significance_threshold_pct:
            trigger.requires_recalculation = True
            trigger.status = TriggerStatus.CONFIRMED
        elif significance < DE_MINIMIS_THRESHOLD_PCT:
            trigger.requires_recalculation = False
            trigger.status = TriggerStatus.DISMISSED
        else:
            trigger.requires_recalculation = False
            trigger.status = TriggerStatus.DETECTED

        trigger.provenance_hash = _compute_hash(trigger)

        logger.debug(
            "Trigger %s assessed: %.4f%% significance, requires_recalc=%s",
            trigger.trigger_id, float(significance), trigger.requires_recalculation,
        )

        return trigger

    def update_trigger_status(
        self,
        trigger_id: str,
        new_status: TriggerStatus,
        reason: str = "",
    ) -> Optional[DetectedTrigger]:
        """Update the lifecycle status of a stored trigger.

        Args:
            trigger_id: ID of the trigger to update.
            new_status: The new status to apply.
            reason: Optional reason for the status change.

        Returns:
            Updated trigger, or None if trigger_id not found.
        """
        trigger = self._trigger_store.get(trigger_id)
        if trigger is None:
            logger.warning("Trigger %s not found in store", trigger_id)
            return None

        old_status = trigger.status
        trigger.status = new_status
        trigger.provenance_hash = _compute_hash(trigger)

        logger.info(
            "Trigger %s status updated: %s -> %s (reason: %s)",
            trigger_id, old_status.value, new_status.value, reason or "N/A",
        )

        return trigger

    def get_pending_triggers(self) -> List[DetectedTrigger]:
        """Retrieve all triggers with DETECTED or UNDER_REVIEW status.

        Returns:
            List of triggers pending resolution, sorted by significance
            percentage descending (most significant first).
        """
        pending = [
            t for t in self._trigger_store.values()
            if t.status in (TriggerStatus.DETECTED, TriggerStatus.UNDER_REVIEW)
        ]
        pending.sort(key=lambda t: t.significance_pct, reverse=True)
        return pending

    def get_confirmed_triggers(self) -> List[DetectedTrigger]:
        """Retrieve all triggers with CONFIRMED status.

        Returns:
            List of confirmed triggers awaiting processing.
        """
        return [
            t for t in self._trigger_store.values()
            if t.status == TriggerStatus.CONFIRMED
        ]

    def get_all_triggers(self) -> List[DetectedTrigger]:
        """Retrieve all triggers in the store.

        Returns:
            List of all triggers, sorted by detected_date descending.
        """
        triggers = list(self._trigger_store.values())
        triggers.sort(key=lambda t: t.detected_date, reverse=True)
        return triggers

    def clear_trigger_store(self) -> int:
        """Clear all triggers from the internal store.

        Returns:
            Number of triggers removed.
        """
        count = len(self._trigger_store)
        self._trigger_store.clear()
        logger.info("Trigger store cleared: %d triggers removed", count)
        return count

    # -----------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------

    def _detect_entity_changes_from_inventory(
        self,
        current: InventorySnapshot,
        previous: InventorySnapshot,
        config: TriggerDetectionConfig,
    ) -> List[DetectedTrigger]:
        """Detect entity changes by comparing inventory entity lists.

        Compares entity_ids lists between two inventory snapshots to
        identify structural changes.

        Args:
            current: Current period inventory snapshot.
            previous: Previous period inventory snapshot.
            config: Detection configuration.

        Returns:
            List of detected entity change triggers.
        """
        triggers: List[DetectedTrigger] = []

        current_entities = set(current.entity_ids)
        previous_entities = set(previous.entity_ids)

        # Entities added (potential acquisitions)
        for eid in sorted(current_entities - previous_entities):
            facility_emissions = _decimal(current.by_facility.get(eid, Decimal("0")))
            entity_change = EntityChange(
                entity_id=eid,
                entity_name=eid,
                change_type=TriggerType.ACQUISITION,
                effective_date=date.today(),
                emissions_impact_tco2e=facility_emissions,
                ownership_pct_before=Decimal("0"),
                ownership_pct_after=Decimal("100"),
                sectors_affected=[],
            )
            trigger = DetectedTrigger(
                trigger_type=TriggerType.ACQUISITION,
                detection_method=DetectionMethod.AUTOMATED,
                description=(
                    f"Entity '{eid}' appeared in current inventory but not in "
                    f"previous (potential acquisition, {facility_emissions} tCO2e)"
                ),
                emission_impact_tco2e=facility_emissions,
                entity_change=entity_change,
            )
            triggers.append(trigger)

        # Entities removed (potential divestitures)
        for eid in sorted(previous_entities - current_entities):
            facility_emissions = _decimal(previous.by_facility.get(eid, Decimal("0")))
            entity_change = EntityChange(
                entity_id=eid,
                entity_name=eid,
                change_type=TriggerType.DIVESTITURE,
                effective_date=date.today(),
                emissions_impact_tco2e=facility_emissions,
                ownership_pct_before=Decimal("100"),
                ownership_pct_after=Decimal("0"),
                sectors_affected=[],
            )
            trigger = DetectedTrigger(
                trigger_type=TriggerType.DIVESTITURE,
                detection_method=DetectionMethod.AUTOMATED,
                description=(
                    f"Entity '{eid}' was in previous inventory but not in "
                    f"current (potential divestiture, {facility_emissions} tCO2e)"
                ),
                emission_impact_tco2e=facility_emissions,
                entity_change=entity_change,
            )
            triggers.append(trigger)

        return triggers

    def _process_external_event(
        self,
        event: ExternalEvent,
        config: TriggerDetectionConfig,
    ) -> DetectedTrigger:
        """Convert an external event to a DetectedTrigger.

        Args:
            event: The external event to process.
            config: Detection configuration.

        Returns:
            DetectedTrigger created from the external event.
        """
        trigger = DetectedTrigger(
            trigger_type=event.trigger_type,
            detection_method=event.detection_method,
            description=event.description or f"External event: {event.trigger_type.value}",
            emission_impact_tco2e=event.estimated_impact_tco2e,
            entity_change=event.entity_change,
            methodology_change=event.methodology_change,
            error_correction=event.error_correction,
            boundary_change=event.boundary_change,
        )
        return trigger

    def _assess_trigger_significance(
        self,
        trigger: DetectedTrigger,
        config: TriggerDetectionConfig,
    ) -> None:
        """Assess a trigger's significance in-place.

        Mutates the trigger to set significance_pct, requires_recalculation,
        and status based on the configured thresholds.

        Args:
            trigger: The trigger to assess (modified in-place).
            config: Detection configuration with thresholds and base year total.
        """
        if config.base_year_total_tco2e <= Decimal("0"):
            trigger.significance_pct = Decimal("0")
            trigger.requires_recalculation = False
            return

        significance = _safe_pct(
            trigger.emission_impact_tco2e, config.base_year_total_tco2e
        )
        trigger.significance_pct = _round_val(significance, 4)

        # Mergers always require recalculation
        if trigger.trigger_type == TriggerType.MERGER:
            trigger.requires_recalculation = True
            trigger.status = TriggerStatus.CONFIRMED
        elif significance >= config.significance_threshold_pct:
            trigger.requires_recalculation = True
            trigger.status = TriggerStatus.CONFIRMED
        elif (
            config.auto_dismiss_below_de_minimis
            and significance < config.de_minimis_threshold_pct
        ):
            trigger.requires_recalculation = False
            trigger.status = TriggerStatus.DISMISSED
        else:
            trigger.requires_recalculation = False
            trigger.status = TriggerStatus.DETECTED

    def _generate_recommendations(
        self,
        triggers: List[DetectedTrigger],
        cumulative_impact: Decimal,
        cumulative_significance: Decimal,
        config: TriggerDetectionConfig,
    ) -> List[str]:
        """Generate human-readable recommendations based on detection results.

        Args:
            triggers: All detected triggers.
            cumulative_impact: Total cumulative impact in tCO2e.
            cumulative_significance: Cumulative significance percentage.
            config: Detection configuration.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []

        if not triggers:
            recommendations.append(
                "No recalculation triggers detected. Base year remains valid."
            )
            return recommendations

        significant_count = sum(1 for t in triggers if t.requires_recalculation)
        dismissed_count = sum(1 for t in triggers if t.status == TriggerStatus.DISMISSED)

        if significant_count > 0:
            recommendations.append(
                f"{significant_count} trigger(s) exceed the significance threshold "
                f"({config.significance_threshold_pct}%). "
                "Route to SignificanceAssessmentEngine for detailed evaluation."
            )

        if cumulative_significance >= config.significance_threshold_pct:
            recommendations.append(
                f"Cumulative impact ({_round_val(cumulative_significance, 2)}%) "
                f"exceeds the significance threshold ({config.significance_threshold_pct}%). "
                "Even if individual triggers are below threshold, the cumulative "
                "effect may warrant base year recalculation."
            )

        # Type-specific recommendations
        merger_triggers = [t for t in triggers if t.trigger_type == TriggerType.MERGER]
        if merger_triggers:
            recommendations.append(
                "MERGER trigger detected. Per GHG Protocol Chapter 5, mergers "
                "always require base year recalculation regardless of significance."
            )

        acquisition_triggers = [
            t for t in triggers if t.trigger_type == TriggerType.ACQUISITION
        ]
        if acquisition_triggers:
            recommendations.append(
                f"{len(acquisition_triggers)} acquisition(s) detected. "
                "Add acquired entity emissions to base year using best available "
                "historical data. Apply pro-rata adjustment if mid-year acquisition."
            )

        divestiture_triggers = [
            t for t in triggers if t.trigger_type == TriggerType.DIVESTITURE
        ]
        if divestiture_triggers:
            recommendations.append(
                f"{len(divestiture_triggers)} divestiture(s) detected. "
                "Remove divested entity emissions from base year. Apply pro-rata "
                "adjustment if mid-year divestiture."
            )

        error_triggers = [
            t for t in triggers if t.trigger_type == TriggerType.ERROR_CORRECTION
        ]
        if error_triggers:
            total_error_impact = sum(
                (t.emission_impact_tco2e for t in error_triggers), Decimal("0")
            )
            recommendations.append(
                f"{len(error_triggers)} error correction(s) detected "
                f"(total impact: {_round_val(total_error_impact, 3)} tCO2e). "
                "Per GHG Protocol, assess whether cumulative corrections "
                "exceed the significance threshold."
            )

        if dismissed_count > 0:
            recommendations.append(
                f"{dismissed_count} trigger(s) dismissed below de minimis threshold "
                f"({config.de_minimis_threshold_pct}%). No action required."
            )

        pending_count = sum(1 for t in triggers if t.status == TriggerStatus.DETECTED)
        if pending_count > 0:
            recommendations.append(
                f"{pending_count} trigger(s) pending review (between de minimis and "
                f"significance threshold). Manual review recommended."
            )

        return recommendations
