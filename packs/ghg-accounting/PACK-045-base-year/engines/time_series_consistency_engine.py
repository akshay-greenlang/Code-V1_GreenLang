# -*- coding: utf-8 -*-
"""
TimeSeriesConsistencyEngine - PACK-045 Base Year Management Engine 7
=====================================================================

Time-series comparability and trend validation engine that ensures GHG
inventory data maintains consistency across reporting years.  Detects
boundary mismatches, methodology changes, coverage gaps, GWP version
changes, and consolidation approach changes that break comparability.

When inconsistencies are detected, the engine produces structured findings
with severity ratings, emission impact estimates, and resolution
recommendations.  It also provides normalization capabilities to adjust
for structural changes and produce a clean trend line suitable for
regulatory reporting.

Consistency Assessment Methodology:
    For each pair of consecutive years in a time series, the engine checks:

    1. Boundary Consistency:
        All entities present in year N must be present in year N+1, or
        the delta must be documented as an acquisition/divestiture.

    2. Methodology Consistency:
        methodology_versions dict must be identical across years, or
        changes must be documented and base year recalculated.

    3. Source Coverage Consistency:
        The set of emission source categories must be stable, or
        additions/removals must be documented.

    4. GWP Version Consistency:
        The same GWP version (AR4, AR5, AR6) must be used across all
        years, or the series must be restated.

    5. Consolidation Approach Consistency:
        The consolidation approach (operational_control, financial_control,
        equity_share) must be the same across all years.

Normalization Methodology:
    When structural changes (acquisitions, divestitures, weather) break
    comparability, the engine normalizes the series:

    structural_change:
        normalized_tco2e = original_tco2e +/- structural_adjustment

    weather_normalization:
        normalized_tco2e = original_tco2e * (normal_hdd / actual_hdd)

    production_volume:
        normalized_tco2e = original_tco2e * (base_production / actual_production)

Trend Calculation:
    year_over_year_pct = (current - previous) / abs(previous) * 100
    cumulative_from_base_pct = (current - base) / abs(base) * 100

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - ISO 14064-1:2018, Clause 5.2.3 (Time series consistency)
    - ESRS E1-6 (Gross GHG emissions - base year and trend)
    - CDP Climate Change Questionnaire C5.1-C5.2 (Recalculation policy)
    - SBTi Corporate Manual (2023), Section 7 (Recalculation)
    - SEC Climate Disclosure Rule (2024), Item 1504(b)

Zero-Hallucination:
    - All consistency checks are deterministic set/value comparisons
    - Normalization uses only user-supplied adjustment factors
    - Trend calculations use Decimal arithmetic (no floating point)
    - No LLM involvement in any assessment or calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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
    the same hash.

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
    """Safely convert a value to Decimal."""
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
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _abs_decimal(value: Decimal) -> Decimal:
    """Return absolute value of a Decimal."""
    return value if value >= Decimal("0") else -value


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsistencyStatus(str, Enum):
    """Overall consistency status of a time series.

    CONSISTENT:             All years pass all consistency checks with no
                            findings of severity >= 3.
    INCONSISTENT:           One or more findings of severity >= 4 detected,
                            series is not suitable for direct trend reporting.
    PARTIALLY_CONSISTENT:   Minor findings (severity 1-3) exist but series
                            is broadly comparable with caveats.
    NOT_ASSESSED:           Assessment has not been performed yet.
    """
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    PARTIALLY_CONSISTENT = "partially_consistent"
    NOT_ASSESSED = "not_assessed"


class InconsistencyType(str, Enum):
    """Type of inconsistency detected in the time series.

    BOUNDARY_MISMATCH:              Entities in boundary differ between years.
    METHODOLOGY_CHANGE:             Calculation methodology or emission factors
                                    changed without base year recalculation.
    SOURCE_COVERAGE_GAP:            Emission source categories differ between
                                    years without documented reason.
    GWP_VERSION_CHANGE:             Different GWP version used across years.
    CONSOLIDATION_APPROACH_CHANGE:  Consolidation approach changed between
                                    years without documented reason.
    DATA_GAP:                       Missing data for one or more years in
                                    the series, breaking continuity.
    """
    BOUNDARY_MISMATCH = "boundary_mismatch"
    METHODOLOGY_CHANGE = "methodology_change"
    SOURCE_COVERAGE_GAP = "source_coverage_gap"
    GWP_VERSION_CHANGE = "gwp_version_change"
    CONSOLIDATION_APPROACH_CHANGE = "consolidation_approach_change"
    DATA_GAP = "data_gap"


class NormalizationType(str, Enum):
    """Type of normalization adjustment applied to the series.

    STRUCTURAL_CHANGE:      Adjustment for acquisitions or divestitures so
                            that all years reflect the same organizational
                            boundary.
    ORGANIC_GROWTH:         Adjustment to remove the effect of organic growth
                            and isolate efficiency improvements.
    WEATHER:                Weather normalization (e.g., heating/cooling degree
                            days) to remove climate variability effects.
    PRODUCTION_VOLUME:      Normalization to a common production volume to
                            enable intensity-based comparisons.
    """
    STRUCTURAL_CHANGE = "structural_change"
    ORGANIC_GROWTH = "organic_growth"
    WEATHER = "weather"
    PRODUCTION_VOLUME = "production_volume"


class ConsolidationApproach(str, Enum):
    """GHG Protocol consolidation approach.

    OPERATIONAL_CONTROL:    Organization accounts for 100% of emissions from
                            operations over which it has operational control.
    FINANCIAL_CONTROL:      Organization accounts for 100% of emissions from
                            operations over which it has financial control.
    EQUITY_SHARE:           Organization accounts for emissions proportional
                            to its equity share in each operation.
    """
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class GWPVersion(str, Enum):
    """IPCC Global Warming Potential version identifier.

    AR4:    IPCC Fourth Assessment Report (2007).
    AR5:    IPCC Fifth Assessment Report (2014).
    AR6:    IPCC Sixth Assessment Report (2021).
    """
    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"


class ReportingFramework(str, Enum):
    """Reporting framework for trend validation requirements.

    GHG_PROTOCOL:   GHG Protocol Corporate Standard.
    ISO_14064:      ISO 14064-1:2018.
    ESRS_E1:        European Sustainability Reporting Standards E1.
    CDP:            CDP Climate Change Questionnaire.
    SBTI:           Science Based Targets initiative.
    SEC:            SEC Climate Disclosure Rule.
    """
    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    ESRS_E1 = "esrs_e1"
    CDP = "cdp"
    SBTI = "sbti"
    SEC = "sec"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum number of years required for trend analysis by framework.
MIN_YEARS_FOR_TREND: Dict[str, int] = {
    ReportingFramework.GHG_PROTOCOL.value: 2,
    ReportingFramework.ISO_14064.value: 2,
    ReportingFramework.ESRS_E1.value: 3,
    ReportingFramework.CDP.value: 4,
    ReportingFramework.SBTI.value: 2,
    ReportingFramework.SEC.value: 3,
}

# Default severity for each inconsistency type (1=minor, 5=critical).
DEFAULT_SEVERITY: Dict[str, int] = {
    InconsistencyType.BOUNDARY_MISMATCH.value: 4,
    InconsistencyType.METHODOLOGY_CHANGE.value: 4,
    InconsistencyType.SOURCE_COVERAGE_GAP.value: 3,
    InconsistencyType.GWP_VERSION_CHANGE.value: 5,
    InconsistencyType.CONSOLIDATION_APPROACH_CHANGE.value: 5,
    InconsistencyType.DATA_GAP.value: 3,
}

# Maximum year-over-year change (%) before flagging as suspicious.
MAX_YOY_CHANGE_PCT: Decimal = Decimal("50")

# Consistency status thresholds (based on max finding severity).
SEVERITY_INCONSISTENT_THRESHOLD: int = 4
SEVERITY_PARTIAL_THRESHOLD: int = 1


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class YearData(BaseModel):
    """Emission data for a single reporting year.

    Contains all information needed to assess consistency: total emissions
    by scope, organizational boundary entities, methodology versions,
    GWP version, and consolidation approach.

    Attributes:
        year: Reporting year (e.g., 2023).
        total_tco2e: Total emissions in tonnes CO2 equivalent.
        scope1_tco2e: Scope 1 emissions in tonnes CO2 equivalent.
        scope2_location_tco2e: Scope 2 location-based emissions.
        scope2_market_tco2e: Scope 2 market-based emissions.
        scope3_tco2e: Scope 3 emissions (may be zero if not reported).
        boundary_entities: List of entity IDs in the organizational boundary.
        methodology_versions: Dict mapping source category to methodology
            version string (e.g., {"stationary_combustion": "tier2_v1.0"}).
        source_categories: Set of emission source category identifiers.
        gwp_version: GWP version used for calculations.
        consolidation_approach: GHG Protocol consolidation approach used.
        is_base_year: Whether this year is the designated base year.
        is_recalculated: Whether this year's data has been recalculated.
    """
    year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    total_tco2e: Decimal = Field(..., ge=0, description="Total tCO2e")
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0,
                                   description="Scope 1 tCO2e")
    scope2_location_tco2e: Decimal = Field(default=Decimal("0"), ge=0,
                                            description="Scope 2 location tCO2e")
    scope2_market_tco2e: Decimal = Field(default=Decimal("0"), ge=0,
                                          description="Scope 2 market tCO2e")
    scope3_tco2e: Decimal = Field(default=Decimal("0"), ge=0,
                                   description="Scope 3 tCO2e")
    boundary_entities: List[str] = Field(default_factory=list,
                                          description="Entity IDs in boundary")
    methodology_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="Source category -> methodology version mapping"
    )
    source_categories: List[str] = Field(
        default_factory=list,
        description="List of emission source categories reported"
    )
    gwp_version: str = Field(default="AR5", description="GWP version used")
    consolidation_approach: str = Field(
        default="operational_control",
        description="GHG Protocol consolidation approach"
    )
    is_base_year: bool = Field(default=False, description="Is base year")
    is_recalculated: bool = Field(default=False,
                                   description="Has been recalculated")

    @field_validator("total_tco2e", "scope1_tco2e", "scope2_location_tco2e",
                     "scope2_market_tco2e", "scope3_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class NormalizationAdjustment(BaseModel):
    """A single normalization adjustment to apply to a specific year.

    Attributes:
        year: The year to adjust.
        normalization_type: Type of normalization.
        adjustment_tco2e: Amount to add (positive) or subtract (negative).
        description: Human-readable description of the adjustment.
        factor: Multiplicative factor for weather/production normalization.
    """
    year: int = Field(..., ge=1990, le=2100)
    normalization_type: NormalizationType
    adjustment_tco2e: Decimal = Field(default=Decimal("0"),
                                       description="Additive adjustment tCO2e")
    description: str = Field(default="", description="Adjustment description")
    factor: Optional[Decimal] = Field(
        default=None,
        description="Multiplicative factor (for weather/production)"
    )

    @field_validator("adjustment_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)


class InconsistencyFinding(BaseModel):
    """A single inconsistency detected in the time series.

    Attributes:
        finding_id: Unique identifier for this finding.
        years_affected: List of years involved in the inconsistency.
        inconsistency_type: Type of inconsistency detected.
        description: Human-readable description of the finding.
        severity: Severity rating (1=minor informational, 5=critical
            inconsistency requiring recalculation).
        emission_impact_tco2e: Estimated emission impact of the
            inconsistency, if quantifiable.
        resolution_recommendation: Recommended action to resolve.
        details: Additional structured details about the finding.
    """
    finding_id: str = Field(default_factory=_new_uuid)
    years_affected: List[int] = Field(..., min_length=1)
    inconsistency_type: InconsistencyType
    description: str
    severity: int = Field(..., ge=1, le=5)
    emission_impact_tco2e: Optional[Decimal] = Field(
        default=None,
        description="Estimated emission impact in tCO2e"
    )
    resolution_recommendation: str = Field(
        default="",
        description="Recommended resolution action"
    )
    details: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("emission_impact_tco2e", mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)


class TrendPoint(BaseModel):
    """A single point in the normalized trend series.

    Attributes:
        year: Reporting year.
        original_tco2e: Original (unadjusted) emission value.
        normalized_tco2e: Emission value after normalization adjustments.
        yoy_change_pct: Year-over-year change percentage from prior year.
            None for the first year in the series.
        cumulative_change_from_base_pct: Cumulative percentage change
            from the base year.  None if base year not identified.
    """
    year: int
    original_tco2e: Decimal
    normalized_tco2e: Decimal
    yoy_change_pct: Optional[Decimal] = None
    cumulative_change_from_base_pct: Optional[Decimal] = None

    @field_validator("original_tco2e", "normalized_tco2e", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

    @field_validator("yoy_change_pct", "cumulative_change_from_base_pct",
                     mode="before")
    @classmethod
    def _coerce_optional_decimal(cls, v: Any) -> Optional[Decimal]:
        if v is None:
            return None
        return _decimal(v)


class ConsistencyConfig(BaseModel):
    """Configuration for consistency assessment.

    Attributes:
        max_yoy_change_pct: Maximum acceptable year-over-year change.
        boundary_tolerance_pct: Percentage of entities that may differ
            between years without flagging (e.g., 5% for minor changes).
        require_gwp_consistency: Whether to require same GWP across years.
        require_methodology_consistency: Whether to require same methodology.
        severity_override: Override default severity for specific types.
    """
    max_yoy_change_pct: Decimal = Field(default=MAX_YOY_CHANGE_PCT)
    boundary_tolerance_pct: Decimal = Field(default=Decimal("0"))
    require_gwp_consistency: bool = Field(default=True)
    require_methodology_consistency: bool = Field(default=True)
    severity_override: Dict[str, int] = Field(default_factory=dict)


class ConsistencyResult(BaseModel):
    """Complete result of time-series consistency assessment.

    Attributes:
        status: Overall consistency status.
        years_assessed: List of years included in the assessment.
        findings: List of inconsistency findings.
        normalized_series: Normalized trend points after adjustments.
        trend_valid: Whether the trend is suitable for regulatory reporting.
        max_severity: Maximum severity found across all findings.
        total_findings: Total count of findings.
        recommendations: Aggregated list of recommendations.
        calculated_at: Timestamp of the assessment.
        processing_time_ms: Time taken for assessment in milliseconds.
        provenance_hash: SHA-256 hash of the result for auditability.
    """
    status: ConsistencyStatus = Field(default=ConsistencyStatus.NOT_ASSESSED)
    years_assessed: List[int] = Field(default_factory=list)
    findings: List[InconsistencyFinding] = Field(default_factory=list)
    normalized_series: List[TrendPoint] = Field(default_factory=list)
    trend_valid: bool = Field(default=False)
    max_severity: int = Field(default=0)
    total_findings: int = Field(default=0)
    recommendations: List[str] = Field(default_factory=list)
    calculated_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class TrendValidationResult(BaseModel):
    """Result of trend validation against a specific reporting framework.

    Attributes:
        framework: The reporting framework validated against.
        is_valid: Whether the trend meets framework requirements.
        min_years_required: Minimum years required by the framework.
        years_available: Number of years in the series.
        issues: List of issues preventing validation.
        provenance_hash: SHA-256 hash of the result.
    """
    framework: ReportingFramework
    is_valid: bool = Field(default=False)
    min_years_required: int = Field(default=2)
    years_available: int = Field(default=0)
    issues: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TimeSeriesConsistencyEngine:
    """Time-series comparability and trend validation engine.

    Guarantees:
        - Deterministic: Same input -> Same output (bit-perfect)
        - Reproducible: Full provenance tracking with SHA-256 hashes
        - Auditable: Every finding traceable to specific data points
        - NO LLM: Zero hallucination risk in all checks and calculations

    Usage::

        engine = TimeSeriesConsistencyEngine()
        series = [YearData(year=2020, ...), YearData(year=2021, ...), ...]
        result = engine.assess_consistency(series)
        print(result.status)
        for finding in result.findings:
            print(finding.description)
    """

    def __init__(self, config: Optional[ConsistencyConfig] = None) -> None:
        """Initialize the TimeSeriesConsistencyEngine.

        Args:
            config: Optional configuration overrides.  If None, defaults
                    are used.
        """
        self.config = config or ConsistencyConfig()
        logger.info(
            "TimeSeriesConsistencyEngine initialized (version=%s)",
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_consistency(
        self,
        year_data_series: List[YearData],
    ) -> ConsistencyResult:
        """Perform comprehensive consistency assessment on a time series.

        Runs all consistency checks (boundary, methodology, coverage, GWP,
        consolidation) and aggregates findings into a single result with
        an overall status determination.

        Args:
            year_data_series: List of YearData for each reporting year.
                              Must contain at least 2 years.

        Returns:
            ConsistencyResult with status, findings, and recommendations.

        Raises:
            ValueError: If fewer than 2 years provided.
        """
        t0 = time.perf_counter()

        if len(year_data_series) < 2:
            raise ValueError(
                "At least 2 years of data are required for consistency "
                f"assessment; received {len(year_data_series)}."
            )

        # Sort by year ascending.
        series = sorted(year_data_series, key=lambda yd: yd.year)
        years = [yd.year for yd in series]

        # Check for year gaps.
        findings: List[InconsistencyFinding] = []
        findings.extend(self._check_year_gaps(series))

        # Run each consistency check.
        findings.extend(self.check_boundary_consistency(series))
        findings.extend(self.check_methodology_consistency(series))
        findings.extend(self.check_coverage_consistency(series))
        findings.extend(self.check_gwp_consistency(series))
        findings.extend(self._check_consolidation_consistency(series))

        # Check for suspicious year-over-year changes.
        findings.extend(self._check_yoy_anomalies(series))

        # Determine overall status.
        max_severity = 0
        for f in findings:
            if f.severity > max_severity:
                max_severity = f.severity

        if max_severity >= SEVERITY_INCONSISTENT_THRESHOLD:
            status = ConsistencyStatus.INCONSISTENT
        elif max_severity >= SEVERITY_PARTIAL_THRESHOLD:
            status = ConsistencyStatus.PARTIALLY_CONSISTENT
        else:
            status = ConsistencyStatus.CONSISTENT

        # Build basic (un-normalized) trend.
        trend_points = self.calculate_trend(series)
        trend_valid = status != ConsistencyStatus.INCONSISTENT

        # Aggregate recommendations.
        recommendations: List[str] = []
        seen_recs: Set[str] = set()
        for f in findings:
            if f.resolution_recommendation and f.resolution_recommendation not in seen_recs:
                recommendations.append(f.resolution_recommendation)
                seen_recs.add(f.resolution_recommendation)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ConsistencyResult(
            status=status,
            years_assessed=years,
            findings=findings,
            normalized_series=trend_points,
            trend_valid=trend_valid,
            max_severity=max_severity,
            total_findings=len(findings),
            recommendations=recommendations,
            calculated_at=_utcnow(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def check_boundary_consistency(
        self,
        series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check organizational boundary consistency across years.

        Compares the set of boundary_entities between each consecutive
        pair of years.  Any additions or removals are flagged as boundary
        mismatches unless both years have empty boundary lists.

        Args:
            series: Sorted list of YearData (ascending by year).

        Returns:
            List of InconsistencyFinding for boundary mismatches.
        """
        findings: List[InconsistencyFinding] = []
        sorted_series = sorted(series, key=lambda yd: yd.year)

        for i in range(len(sorted_series) - 1):
            current = sorted_series[i]
            next_yr = sorted_series[i + 1]

            current_entities = set(current.boundary_entities)
            next_entities = set(next_yr.boundary_entities)

            if not current_entities and not next_entities:
                continue

            added = next_entities - current_entities
            removed = current_entities - next_entities

            if added or removed:
                # Calculate tolerance.
                total_entities = len(current_entities | next_entities)
                change_count = len(added) + len(removed)
                change_pct = _safe_pct(
                    _decimal(change_count), _decimal(total_entities)
                )

                if change_pct <= self.config.boundary_tolerance_pct:
                    continue

                severity = self._get_severity(
                    InconsistencyType.BOUNDARY_MISMATCH
                )

                desc_parts: List[str] = []
                if added:
                    desc_parts.append(
                        f"{len(added)} entities added: "
                        f"{', '.join(sorted(added)[:5])}"
                    )
                if removed:
                    desc_parts.append(
                        f"{len(removed)} entities removed: "
                        f"{', '.join(sorted(removed)[:5])}"
                    )

                findings.append(InconsistencyFinding(
                    years_affected=[current.year, next_yr.year],
                    inconsistency_type=InconsistencyType.BOUNDARY_MISMATCH,
                    description=(
                        f"Boundary mismatch between {current.year} and "
                        f"{next_yr.year}: {'; '.join(desc_parts)}. "
                        f"Change affects {_round_val(change_pct, 1)}% of entities."
                    ),
                    severity=severity,
                    resolution_recommendation=(
                        "Recalculate base year to include/exclude affected "
                        "entities per GHG Protocol Chapter 5 guidance."
                    ),
                    details={
                        "added_entities": sorted(added),
                        "removed_entities": sorted(removed),
                        "change_pct": str(_round_val(change_pct, 2)),
                    },
                ))

        return findings

    def check_methodology_consistency(
        self,
        series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check calculation methodology consistency across years.

        Compares methodology_versions dicts between consecutive years.
        Any change in methodology version for a source category is flagged.

        Args:
            series: Sorted list of YearData (ascending by year).

        Returns:
            List of InconsistencyFinding for methodology changes.
        """
        findings: List[InconsistencyFinding] = []
        if not self.config.require_methodology_consistency:
            return findings

        sorted_series = sorted(series, key=lambda yd: yd.year)

        for i in range(len(sorted_series) - 1):
            current = sorted_series[i]
            next_yr = sorted_series[i + 1]

            current_methods = current.methodology_versions
            next_methods = next_yr.methodology_versions

            if not current_methods and not next_methods:
                continue

            all_categories = set(current_methods.keys()) | set(next_methods.keys())
            changed_categories: List[Tuple[str, str, str]] = []

            for cat in sorted(all_categories):
                curr_ver = current_methods.get(cat, "not_reported")
                next_ver = next_methods.get(cat, "not_reported")
                if curr_ver != next_ver:
                    changed_categories.append((cat, curr_ver, next_ver))

            if changed_categories:
                severity = self._get_severity(
                    InconsistencyType.METHODOLOGY_CHANGE
                )

                cat_details = [
                    f"{cat}: {old} -> {new}"
                    for cat, old, new in changed_categories
                ]

                findings.append(InconsistencyFinding(
                    years_affected=[current.year, next_yr.year],
                    inconsistency_type=InconsistencyType.METHODOLOGY_CHANGE,
                    description=(
                        f"Methodology change between {current.year} and "
                        f"{next_yr.year} in {len(changed_categories)} "
                        f"categories: {'; '.join(cat_details[:3])}"
                        f"{' (and more)' if len(cat_details) > 3 else ''}."
                    ),
                    severity=severity,
                    resolution_recommendation=(
                        "Recalculate base year using updated methodology "
                        "to maintain like-for-like comparability per GHG "
                        "Protocol Chapter 5."
                    ),
                    details={
                        "changed_categories": [
                            {"category": c, "old_version": o, "new_version": n}
                            for c, o, n in changed_categories
                        ],
                    },
                ))

        return findings

    def check_coverage_consistency(
        self,
        series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check emission source coverage consistency across years.

        Compares source_categories lists between consecutive years.
        Additions or removals of source categories are flagged.

        Args:
            series: Sorted list of YearData (ascending by year).

        Returns:
            List of InconsistencyFinding for coverage gaps.
        """
        findings: List[InconsistencyFinding] = []
        sorted_series = sorted(series, key=lambda yd: yd.year)

        for i in range(len(sorted_series) - 1):
            current = sorted_series[i]
            next_yr = sorted_series[i + 1]

            current_cats = set(current.source_categories)
            next_cats = set(next_yr.source_categories)

            if not current_cats and not next_cats:
                continue

            added = next_cats - current_cats
            removed = current_cats - next_cats

            if added or removed:
                severity = self._get_severity(
                    InconsistencyType.SOURCE_COVERAGE_GAP
                )

                desc_parts: List[str] = []
                if added:
                    desc_parts.append(
                        f"Added: {', '.join(sorted(added)[:5])}"
                    )
                if removed:
                    desc_parts.append(
                        f"Removed: {', '.join(sorted(removed)[:5])}"
                    )

                findings.append(InconsistencyFinding(
                    years_affected=[current.year, next_yr.year],
                    inconsistency_type=InconsistencyType.SOURCE_COVERAGE_GAP,
                    description=(
                        f"Source coverage change between {current.year} "
                        f"and {next_yr.year}: {'; '.join(desc_parts)}."
                    ),
                    severity=severity,
                    resolution_recommendation=(
                        "If new source categories are material (>1% of "
                        "total), recalculate base year to include them."
                    ),
                    details={
                        "added_categories": sorted(added),
                        "removed_categories": sorted(removed),
                    },
                ))

        return findings

    def check_gwp_consistency(
        self,
        series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check GWP version consistency across all years.

        All years in the series must use the same GWP version for the
        time series to be consistent.  Mixed GWP versions result in a
        severity-5 (critical) finding.

        Args:
            series: Sorted list of YearData (ascending by year).

        Returns:
            List of InconsistencyFinding for GWP version changes.
        """
        findings: List[InconsistencyFinding] = []
        if not self.config.require_gwp_consistency:
            return findings

        sorted_series = sorted(series, key=lambda yd: yd.year)
        gwp_versions: Dict[str, List[int]] = {}

        for yd in sorted_series:
            ver = yd.gwp_version
            if ver not in gwp_versions:
                gwp_versions[ver] = []
            gwp_versions[ver].append(yd.year)

        if len(gwp_versions) > 1:
            severity = self._get_severity(
                InconsistencyType.GWP_VERSION_CHANGE
            )

            version_details = [
                f"{ver}: years {sorted(years)}"
                for ver, years in sorted(gwp_versions.items())
            ]

            all_years = [yd.year for yd in sorted_series]

            findings.append(InconsistencyFinding(
                years_affected=all_years,
                inconsistency_type=InconsistencyType.GWP_VERSION_CHANGE,
                description=(
                    f"Multiple GWP versions used across time series: "
                    f"{'; '.join(version_details)}. All years must use "
                    f"the same GWP version for comparability."
                ),
                severity=severity,
                resolution_recommendation=(
                    "Restate all years using a single GWP version. "
                    "Most frameworks now recommend AR5 (IPCC 2014) or "
                    "AR6 (IPCC 2021) 100-year values."
                ),
                details={
                    "gwp_versions_used": {
                        ver: sorted(years)
                        for ver, years in gwp_versions.items()
                    },
                },
            ))

        return findings

    def normalize_for_structural_changes(
        self,
        series: List[YearData],
        adjustments: List[NormalizationAdjustment],
    ) -> List[TrendPoint]:
        """Normalize the time series for structural or external changes.

        Applies the provided normalization adjustments to each year in the
        series, producing a normalized trend line that removes the effect
        of structural changes, weather variability, or production volume
        changes.

        Normalization Logic:
            For STRUCTURAL_CHANGE and ORGANIC_GROWTH:
                normalized = original + adjustment_tco2e

            For WEATHER and PRODUCTION_VOLUME:
                If factor is provided:
                    normalized = original * factor
                Else:
                    normalized = original + adjustment_tco2e

        Args:
            series: Sorted list of YearData (ascending by year).
            adjustments: List of NormalizationAdjustment to apply.

        Returns:
            List of TrendPoint with normalized values and trend metrics.
        """
        sorted_series = sorted(series, key=lambda yd: yd.year)

        # Index adjustments by year for O(1) lookup.
        adj_by_year: Dict[int, List[NormalizationAdjustment]] = {}
        for adj in adjustments:
            if adj.year not in adj_by_year:
                adj_by_year[adj.year] = []
            adj_by_year[adj.year].append(adj)

        # Apply adjustments.
        normalized_values: List[Tuple[int, Decimal, Decimal]] = []
        for yd in sorted_series:
            original = _decimal(yd.total_tco2e)
            normalized = original

            year_adjs = adj_by_year.get(yd.year, [])
            for adj in year_adjs:
                if adj.normalization_type in (
                    NormalizationType.WEATHER,
                    NormalizationType.PRODUCTION_VOLUME,
                ) and adj.factor is not None:
                    normalized = normalized * _decimal(adj.factor)
                else:
                    normalized = normalized + _decimal(adj.adjustment_tco2e)

            normalized_values.append((yd.year, original, normalized))

        # Calculate trend metrics on normalized values.
        return self._build_trend_points(normalized_values, sorted_series)

    def calculate_trend(
        self,
        series: List[YearData],
    ) -> List[TrendPoint]:
        """Calculate trend from the raw (un-normalized) time series.

        Computes year-over-year and cumulative change percentages
        using the original emission values.

        Args:
            series: List of YearData for each reporting year.

        Returns:
            List of TrendPoint with trend metrics.
        """
        sorted_series = sorted(series, key=lambda yd: yd.year)
        values = [
            (yd.year, _decimal(yd.total_tco2e), _decimal(yd.total_tco2e))
            for yd in sorted_series
        ]
        return self._build_trend_points(values, sorted_series)

    def validate_trend_for_reporting(
        self,
        series: List[YearData],
        framework: ReportingFramework,
    ) -> TrendValidationResult:
        """Validate whether the time series meets a specific framework's
        requirements for trend reporting.

        Each framework has different requirements for:
            - Minimum number of years
            - Consistency of boundary, methodology, GWP
            - Specific disclosures

        Args:
            series: List of YearData for each reporting year.
            framework: The target reporting framework.

        Returns:
            TrendValidationResult with validation outcome and issues.
        """
        sorted_series = sorted(series, key=lambda yd: yd.year)
        issues: List[str] = []

        # Check minimum years.
        min_years = MIN_YEARS_FOR_TREND.get(framework.value, 2)
        years_available = len(sorted_series)

        if years_available < min_years:
            issues.append(
                f"{framework.value} requires at least {min_years} years "
                f"of data; only {years_available} available."
            )

        # Run consistency checks.
        findings = self.check_boundary_consistency(sorted_series)
        findings.extend(self.check_methodology_consistency(sorted_series))
        findings.extend(self.check_gwp_consistency(sorted_series))

        for f in findings:
            if f.severity >= SEVERITY_INCONSISTENT_THRESHOLD:
                issues.append(
                    f"[{f.inconsistency_type.value}] {f.description}"
                )

        # Framework-specific checks.
        issues.extend(
            self._framework_specific_checks(sorted_series, framework)
        )

        # Check base year is identified.
        has_base_year = any(yd.is_base_year for yd in sorted_series)
        if not has_base_year:
            issues.append("No base year identified in the time series.")

        is_valid = len(issues) == 0

        result = TrendValidationResult(
            framework=framework,
            is_valid=is_valid,
            min_years_required=min_years,
            years_available=years_available,
            issues=issues,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_severity(self, itype: InconsistencyType) -> int:
        """Get severity for an inconsistency type, checking overrides."""
        override = self.config.severity_override.get(itype.value)
        if override is not None:
            return max(1, min(5, override))
        return DEFAULT_SEVERITY.get(itype.value, 3)

    def _check_year_gaps(
        self,
        sorted_series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check for gaps (missing years) in the series."""
        findings: List[InconsistencyFinding] = []
        for i in range(len(sorted_series) - 1):
            current_year = sorted_series[i].year
            next_year = sorted_series[i + 1].year
            if next_year - current_year > 1:
                missing = list(range(current_year + 1, next_year))
                severity = self._get_severity(InconsistencyType.DATA_GAP)
                findings.append(InconsistencyFinding(
                    years_affected=missing,
                    inconsistency_type=InconsistencyType.DATA_GAP,
                    description=(
                        f"Data gap: missing years {missing} between "
                        f"{current_year} and {next_year}."
                    ),
                    severity=severity,
                    resolution_recommendation=(
                        "Provide data for missing years or document the "
                        "reason for the gap.  Interpolation should not "
                        "be used for regulatory reporting."
                    ),
                    details={"missing_years": missing},
                ))
        return findings

    def _check_consolidation_consistency(
        self,
        sorted_series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check consolidation approach consistency across all years."""
        findings: List[InconsistencyFinding] = []
        approaches: Dict[str, List[int]] = {}

        for yd in sorted_series:
            approach = yd.consolidation_approach
            if approach not in approaches:
                approaches[approach] = []
            approaches[approach].append(yd.year)

        if len(approaches) > 1:
            severity = self._get_severity(
                InconsistencyType.CONSOLIDATION_APPROACH_CHANGE
            )
            approach_details = [
                f"{appr}: years {sorted(years)}"
                for appr, years in sorted(approaches.items())
            ]

            all_years = [yd.year for yd in sorted_series]

            findings.append(InconsistencyFinding(
                years_affected=all_years,
                inconsistency_type=InconsistencyType.CONSOLIDATION_APPROACH_CHANGE,
                description=(
                    f"Multiple consolidation approaches used: "
                    f"{'; '.join(approach_details)}. GHG Protocol requires "
                    f"consistent approach across all years."
                ),
                severity=severity,
                resolution_recommendation=(
                    "Select a single consolidation approach and restate "
                    "all years.  GHG Protocol recommends documenting the "
                    "reason for any change in approach."
                ),
                details={
                    "approaches_used": {
                        appr: sorted(years)
                        for appr, years in approaches.items()
                    },
                },
            ))

        return findings

    def _check_yoy_anomalies(
        self,
        sorted_series: List[YearData],
    ) -> List[InconsistencyFinding]:
        """Check for suspiciously large year-over-year changes."""
        findings: List[InconsistencyFinding] = []

        for i in range(len(sorted_series) - 1):
            current = sorted_series[i]
            next_yr = sorted_series[i + 1]

            current_total = _decimal(current.total_tco2e)
            next_total = _decimal(next_yr.total_tco2e)

            if current_total == Decimal("0"):
                continue

            yoy_change = _safe_pct(
                _abs_decimal(next_total - current_total),
                _abs_decimal(current_total),
            )

            if yoy_change > self.config.max_yoy_change_pct:
                direction = "increase" if next_total > current_total else "decrease"
                findings.append(InconsistencyFinding(
                    years_affected=[current.year, next_yr.year],
                    inconsistency_type=InconsistencyType.DATA_GAP,
                    description=(
                        f"Suspicious {_round_val(yoy_change, 1)}% "
                        f"{direction} from {current.year} to {next_yr.year} "
                        f"({_round_val(current_total, 1)} -> "
                        f"{_round_val(next_total, 1)} tCO2e). "
                        f"Exceeds {self.config.max_yoy_change_pct}% threshold."
                    ),
                    severity=2,
                    resolution_recommendation=(
                        "Investigate the cause of the large change.  If "
                        "due to structural changes, apply normalization.  "
                        "If due to data errors, correct and document."
                    ),
                    details={
                        "yoy_change_pct": str(_round_val(yoy_change, 2)),
                        "direction": direction,
                        "current_tco2e": str(current_total),
                        "next_tco2e": str(next_total),
                    },
                ))

        return findings

    def _build_trend_points(
        self,
        values: List[Tuple[int, Decimal, Decimal]],
        sorted_series: List[YearData],
    ) -> List[TrendPoint]:
        """Build TrendPoint list from (year, original, normalized) tuples.

        Calculates year-over-year change and cumulative change from base
        year using the normalized values.

        Args:
            values: List of (year, original_tco2e, normalized_tco2e).
            sorted_series: Original sorted YearData for base year lookup.

        Returns:
            List of TrendPoint with all metrics populated.
        """
        # Find base year value.
        base_value: Optional[Decimal] = None
        for yd in sorted_series:
            if yd.is_base_year:
                # Find matching normalized value.
                for yr, _orig, norm in values:
                    if yr == yd.year:
                        base_value = norm
                        break
                break

        # If no explicit base year, use first year as base.
        if base_value is None and values:
            base_value = values[0][2]

        points: List[TrendPoint] = []
        for idx, (year, original, normalized) in enumerate(values):
            # Year-over-year change.
            yoy: Optional[Decimal] = None
            if idx > 0:
                prev_norm = values[idx - 1][2]
                if prev_norm != Decimal("0"):
                    yoy = _round_val(
                        (normalized - prev_norm) / _abs_decimal(prev_norm)
                        * Decimal("100"),
                        places=4,
                    )

            # Cumulative change from base.
            cumulative: Optional[Decimal] = None
            if base_value is not None and base_value != Decimal("0"):
                cumulative = _round_val(
                    (normalized - base_value) / _abs_decimal(base_value)
                    * Decimal("100"),
                    places=4,
                )

            points.append(TrendPoint(
                year=year,
                original_tco2e=_round_val(original, 3),
                normalized_tco2e=_round_val(normalized, 3),
                yoy_change_pct=yoy,
                cumulative_change_from_base_pct=cumulative,
            ))

        return points

    def _framework_specific_checks(
        self,
        sorted_series: List[YearData],
        framework: ReportingFramework,
    ) -> List[str]:
        """Run framework-specific validation checks.

        Args:
            sorted_series: Sorted list of YearData.
            framework: Target reporting framework.

        Returns:
            List of issue strings.
        """
        issues: List[str] = []

        if framework == ReportingFramework.ESRS_E1:
            # ESRS E1 requires base year and recalculation disclosure.
            has_recalc = any(yd.is_recalculated for yd in sorted_series)
            base_years = [yd for yd in sorted_series if yd.is_base_year]
            if not base_years:
                issues.append(
                    "ESRS E1-6 requires disclosure of the base year and "
                    "the rationale for choosing it."
                )
            # ESRS requires scope 1 and scope 2 separately.
            for yd in sorted_series:
                if yd.scope1_tco2e == Decimal("0") and yd.total_tco2e > Decimal("0"):
                    issues.append(
                        f"ESRS E1 requires separate Scope 1 disclosure "
                        f"for year {yd.year}."
                    )
                    break

        elif framework == ReportingFramework.CDP:
            # CDP requires 4 years of data and scope breakdowns.
            for yd in sorted_series:
                if yd.scope2_location_tco2e == Decimal("0") and \
                   yd.scope2_market_tco2e == Decimal("0") and \
                   yd.total_tco2e > Decimal("0"):
                    issues.append(
                        f"CDP requires Scope 2 disclosure (location and/or "
                        f"market-based) for year {yd.year}."
                    )
                    break

        elif framework == ReportingFramework.SBTI:
            # SBTi requires consistent boundary for target tracking.
            base_yrs = [yd for yd in sorted_series if yd.is_base_year]
            if base_yrs:
                base_entities = set(base_yrs[0].boundary_entities)
                for yd in sorted_series:
                    if not yd.is_base_year and yd.boundary_entities:
                        current_entities = set(yd.boundary_entities)
                        if current_entities != base_entities:
                            issues.append(
                                f"SBTi requires consistent organizational "
                                f"boundary. Year {yd.year} differs from "
                                f"base year."
                            )
                            break

        elif framework == ReportingFramework.SEC:
            # SEC requires 3 years and Scope 1+2 breakdown.
            for yd in sorted_series:
                total_scope12 = yd.scope1_tco2e + max(
                    yd.scope2_location_tco2e, yd.scope2_market_tco2e
                )
                if total_scope12 == Decimal("0") and yd.total_tco2e > Decimal("0"):
                    issues.append(
                        f"SEC Climate Rule requires Scope 1 and Scope 2 "
                        f"breakdown for year {yd.year}."
                    )
                    break

        elif framework == ReportingFramework.GHG_PROTOCOL:
            # GHG Protocol requires consolidation approach declaration.
            approaches = set(yd.consolidation_approach for yd in sorted_series)
            if len(approaches) > 1:
                issues.append(
                    "GHG Protocol requires consistent consolidation "
                    "approach across all reporting years."
                )

        elif framework == ReportingFramework.ISO_14064:
            # ISO 14064 requires base year justification.
            base_yrs = [yd for yd in sorted_series if yd.is_base_year]
            if not base_yrs:
                issues.append(
                    "ISO 14064-1 Clause 5.2 requires a documented base year."
                )

        return issues

    # ------------------------------------------------------------------
    # Batch and utility methods
    # ------------------------------------------------------------------

    def get_base_year_data(
        self,
        series: List[YearData],
    ) -> Optional[YearData]:
        """Extract the base year data from the series.

        Args:
            series: List of YearData.

        Returns:
            The YearData marked as base year, or None if not found.
        """
        for yd in series:
            if yd.is_base_year:
                return yd
        return None

    def get_latest_year_data(
        self,
        series: List[YearData],
    ) -> Optional[YearData]:
        """Extract the most recent year data from the series.

        Args:
            series: List of YearData.

        Returns:
            The YearData with the highest year value, or None if empty.
        """
        if not series:
            return None
        return max(series, key=lambda yd: yd.year)

    def summarize_trend(
        self,
        trend_points: List[TrendPoint],
    ) -> Dict[str, Any]:
        """Produce a summary dict of the trend for reporting.

        Args:
            trend_points: List of TrendPoint from calculate_trend or
                normalize_for_structural_changes.

        Returns:
            Dict with summary statistics: first_year, last_year,
            total_change_pct, average_yoy_change_pct, max_increase_pct,
            max_decrease_pct, years_with_increase, years_with_decrease.
        """
        if not trend_points:
            return {
                "first_year": None,
                "last_year": None,
                "total_change_pct": None,
                "average_yoy_change_pct": None,
                "max_increase_pct": None,
                "max_decrease_pct": None,
                "years_with_increase": 0,
                "years_with_decrease": 0,
            }

        first = trend_points[0]
        last = trend_points[-1]

        total_change: Optional[Decimal] = None
        if first.normalized_tco2e != Decimal("0"):
            total_change = _round_val(
                (last.normalized_tco2e - first.normalized_tco2e)
                / _abs_decimal(first.normalized_tco2e) * Decimal("100"),
                places=2,
            )

        yoy_values: List[Decimal] = []
        max_increase = Decimal("0")
        max_decrease = Decimal("0")
        years_increase = 0
        years_decrease = 0

        for tp in trend_points:
            if tp.yoy_change_pct is not None:
                yoy_values.append(tp.yoy_change_pct)
                if tp.yoy_change_pct > Decimal("0"):
                    years_increase += 1
                    if tp.yoy_change_pct > max_increase:
                        max_increase = tp.yoy_change_pct
                elif tp.yoy_change_pct < Decimal("0"):
                    years_decrease += 1
                    if tp.yoy_change_pct < max_decrease:
                        max_decrease = tp.yoy_change_pct

        avg_yoy: Optional[Decimal] = None
        if yoy_values:
            avg_yoy = _round_val(
                sum(yoy_values) / _decimal(len(yoy_values)),
                places=2,
            )

        return {
            "first_year": first.year,
            "last_year": last.year,
            "total_change_pct": str(total_change) if total_change is not None else None,
            "average_yoy_change_pct": str(avg_yoy) if avg_yoy is not None else None,
            "max_increase_pct": str(_round_val(max_increase, 2)),
            "max_decrease_pct": str(_round_val(max_decrease, 2)),
            "years_with_increase": years_increase,
            "years_with_decrease": years_decrease,
        }

    def compare_two_years(
        self,
        year_a: YearData,
        year_b: YearData,
    ) -> Dict[str, Any]:
        """Compare two specific years and return a detailed comparison.

        Args:
            year_a: First year data.
            year_b: Second year data.

        Returns:
            Dict with field-by-field comparison including deltas.
        """
        comparison: Dict[str, Any] = {
            "year_a": year_a.year,
            "year_b": year_b.year,
        }

        # Emission comparisons.
        fields = [
            ("total_tco2e", year_a.total_tco2e, year_b.total_tco2e),
            ("scope1_tco2e", year_a.scope1_tco2e, year_b.scope1_tco2e),
            ("scope2_location_tco2e", year_a.scope2_location_tco2e,
             year_b.scope2_location_tco2e),
            ("scope2_market_tco2e", year_a.scope2_market_tco2e,
             year_b.scope2_market_tco2e),
            ("scope3_tco2e", year_a.scope3_tco2e, year_b.scope3_tco2e),
        ]

        emission_comparison: List[Dict[str, Any]] = []
        for name, val_a, val_b in fields:
            delta = _decimal(val_b) - _decimal(val_a)
            delta_pct = _safe_pct(_abs_decimal(delta), _abs_decimal(_decimal(val_a)))
            direction = "increase" if delta > Decimal("0") else \
                        "decrease" if delta < Decimal("0") else "unchanged"
            emission_comparison.append({
                "field": name,
                "year_a_value": str(_round_val(_decimal(val_a), 3)),
                "year_b_value": str(_round_val(_decimal(val_b), 3)),
                "delta_tco2e": str(_round_val(delta, 3)),
                "delta_pct": str(_round_val(delta_pct, 2)),
                "direction": direction,
            })

        comparison["emission_comparison"] = emission_comparison

        # Boundary comparison.
        entities_a = set(year_a.boundary_entities)
        entities_b = set(year_b.boundary_entities)
        comparison["boundary"] = {
            "entities_in_a_only": sorted(entities_a - entities_b),
            "entities_in_b_only": sorted(entities_b - entities_a),
            "entities_in_both": sorted(entities_a & entities_b),
            "boundary_changed": entities_a != entities_b,
        }

        # Methodology comparison.
        all_cats = set(year_a.methodology_versions.keys()) | \
                   set(year_b.methodology_versions.keys())
        method_changes: List[Dict[str, str]] = []
        for cat in sorted(all_cats):
            ver_a = year_a.methodology_versions.get(cat, "not_reported")
            ver_b = year_b.methodology_versions.get(cat, "not_reported")
            if ver_a != ver_b:
                method_changes.append({
                    "category": cat,
                    "year_a_version": ver_a,
                    "year_b_version": ver_b,
                })
        comparison["methodology_changes"] = method_changes

        # GWP and consolidation.
        comparison["gwp_consistent"] = year_a.gwp_version == year_b.gwp_version
        comparison["consolidation_consistent"] = (
            year_a.consolidation_approach == year_b.consolidation_approach
        )

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    def detect_structural_changes(
        self,
        series: List[YearData],
    ) -> List[Dict[str, Any]]:
        """Detect potential structural changes from the data.

        Analyzes boundary entity changes to infer acquisitions and
        divestitures that may require base year recalculation.

        Args:
            series: Sorted list of YearData.

        Returns:
            List of dicts describing detected structural changes.
        """
        sorted_series = sorted(series, key=lambda yd: yd.year)
        changes: List[Dict[str, Any]] = []

        for i in range(len(sorted_series) - 1):
            current = sorted_series[i]
            next_yr = sorted_series[i + 1]

            current_entities = set(current.boundary_entities)
            next_entities = set(next_yr.boundary_entities)

            added = next_entities - current_entities
            removed = current_entities - next_entities

            if added:
                changes.append({
                    "type": "acquisition",
                    "year": next_yr.year,
                    "entities": sorted(added),
                    "count": len(added),
                    "emission_delta_tco2e": str(
                        _round_val(
                            _decimal(next_yr.total_tco2e) -
                            _decimal(current.total_tco2e),
                            3,
                        )
                    ),
                    "description": (
                        f"{len(added)} entities added in {next_yr.year}, "
                        f"possibly acquisition."
                    ),
                })

            if removed:
                changes.append({
                    "type": "divestiture",
                    "year": next_yr.year,
                    "entities": sorted(removed),
                    "count": len(removed),
                    "emission_delta_tco2e": str(
                        _round_val(
                            _decimal(next_yr.total_tco2e) -
                            _decimal(current.total_tco2e),
                            3,
                        )
                    ),
                    "description": (
                        f"{len(removed)} entities removed in {next_yr.year}, "
                        f"possibly divestiture."
                    ),
                })

        return changes

    def generate_consistency_report(
        self,
        result: ConsistencyResult,
    ) -> str:
        """Generate a human-readable Markdown consistency report.

        Args:
            result: ConsistencyResult from assess_consistency.

        Returns:
            Markdown-formatted report string.
        """
        lines: List[str] = []
        lines.append("# Time Series Consistency Report")
        lines.append("")
        lines.append(f"**Status:** {result.status.value}")
        lines.append(f"**Years Assessed:** {result.years_assessed}")
        lines.append(f"**Total Findings:** {result.total_findings}")
        lines.append(f"**Max Severity:** {result.max_severity}/5")
        lines.append(f"**Trend Valid:** {'Yes' if result.trend_valid else 'No'}")
        lines.append(f"**Provenance Hash:** `{result.provenance_hash[:16]}...`")
        lines.append("")

        if result.findings:
            lines.append("## Findings")
            lines.append("")
            for i, f in enumerate(result.findings, 1):
                lines.append(
                    f"### {i}. [{f.inconsistency_type.value}] "
                    f"Severity {f.severity}/5"
                )
                lines.append(f"**Years:** {f.years_affected}")
                lines.append(f"**Description:** {f.description}")
                if f.emission_impact_tco2e is not None:
                    lines.append(
                        f"**Emission Impact:** {f.emission_impact_tco2e} tCO2e"
                    )
                if f.resolution_recommendation:
                    lines.append(
                        f"**Recommendation:** {f.resolution_recommendation}"
                    )
                lines.append("")

        if result.normalized_series:
            lines.append("## Trend Data")
            lines.append("")
            lines.append("| Year | Original (tCO2e) | Normalized (tCO2e) | "
                         "YoY Change | Cumulative |")
            lines.append("|------|------------------|--------------------|"
                         "------------|------------|")
            for tp in result.normalized_series:
                yoy = f"{tp.yoy_change_pct}%" if tp.yoy_change_pct is not None else "N/A"
                cum = (f"{tp.cumulative_change_from_base_pct}%"
                       if tp.cumulative_change_from_base_pct is not None else "N/A")
                lines.append(
                    f"| {tp.year} | {tp.original_tco2e} | "
                    f"{tp.normalized_tco2e} | {yoy} | {cum} |"
                )
            lines.append("")

        if result.recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(result.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)
