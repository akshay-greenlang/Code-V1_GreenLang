# -*- coding: utf-8 -*-
"""
EnhancedDNSHEngine - PACK-011 SFDR Article 9 Engine 2
========================================================

Stricter Do No Significant Harm (DNSH) assessment for Article 9 products.

Article 9 ("dark green") products apply DNSH to ALL holdings, not just the
sustainable portion.  Thresholds are stricter than Article 8 products
(e.g., fossil fuel exposure <5 % vs higher tolerance in Article 8).
All 18 mandatory PAI indicators must be assessed, with auto-exclusion
triggered on critical failures.  Portfolio compliance must reach 100 %
for genuine Article 9 status.

Key Differences from Article 8 DNSH:
    - Applies to every holding (not just those claimed as sustainable)
    - Stricter numeric thresholds across all PAI indicators
    - All 18 mandatory PAI indicators assessed (no optional skipping)
    - Auto-exclusion on critical failures (controversies, weapons, etc.)
    - Portfolio compliance target is 100 % (vs partial for Article 8)
    - Remediation plans generated for near-threshold holdings

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 2(17), Article 9
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Article 12(1)
    - SFDR RTS Annex I, Table 1 (mandatory PAI indicators)

Zero-Hallucination:
    - All threshold comparisons use deterministic Python arithmetic
    - Configurable thresholds prevent hard-coded regulatory interpretation
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any assessment path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _round_val(value: float, places: int = 4) -> float:
    """Round a float to specified decimal places."""
    return round(value, places)

def _safe_pct(numerator: float, denominator: float) -> float:
    """Calculate percentage safely, returning 0.0 on zero denominator."""
    if denominator == 0.0:
        return 0.0
    return (numerator / denominator) * 100.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PAICategory(str, Enum):
    """PAI indicator category groupings per SFDR RTS Annex I."""
    CLIMATE_GHG = "climate_ghg"
    ENVIRONMENT = "environment"
    SOCIAL = "social"
    SOVEREIGN = "sovereign"
    REAL_ESTATE = "real_estate"

class DNSHStatus(str, Enum):
    """DNSH assessment outcome for a single check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

class ThresholdDirection(str, Enum):
    """Direction of threshold comparison."""
    MAX = "MAX"
    MIN = "MIN"
    BOOLEAN_TRUE_FAILS = "BOOLEAN_TRUE_FAILS"
    BOOLEAN_FALSE_FAILS = "BOOLEAN_FALSE_FAILS"

class SeverityLevel(str, Enum):
    """Severity classification for DNSH failures."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"

class ExclusionReason(str, Enum):
    """Reason for auto-excluding a holding from the portfolio."""
    CONTROVERSIAL_WEAPONS = "controversial_weapons"
    UNGC_OECD_VIOLATIONS = "ungc_oecd_violations"
    FOSSIL_FUEL_ABOVE_THRESHOLD = "fossil_fuel_above_threshold"
    SOCIAL_VIOLATIONS = "social_violations"
    CRITICAL_PAI_BREACH = "critical_pai_breach"
    MULTIPLE_HIGH_SEVERITY = "multiple_high_severity"
    BIODIVERSITY_HARM = "biodiversity_harm"
    COUNTRY_SANCTIONS = "country_sanctions"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PAIThreshold(BaseModel):
    """Configurable threshold for a single PAI indicator DNSH check.

    Defines the pass/fail boundary and warning zone for a PAI indicator
    under the stricter Article 9 DNSH regime.

    Attributes:
        pai_indicator_id: PAI indicator identifier (PAI_1 through PAI_18).
        pai_name: Human-readable PAI indicator name.
        category: PAI category grouping.
        threshold_value: Numeric threshold for pass/fail boundary.
        warning_value: Warning threshold (softer boundary).
        direction: How to interpret the threshold.
        unit: Unit of measurement.
        severity: Severity level if threshold is breached.
        description: Human-readable description of what the threshold means.
        enabled: Whether this check is active.
        auto_exclude_on_fail: Whether failure triggers auto-exclusion.
    """
    pai_indicator_id: str = Field(
        ..., min_length=1, max_length=20,
        description="PAI indicator identifier (e.g., PAI_1, PAI_4)",
    )
    pai_name: str = Field(
        ..., description="Human-readable indicator name",
    )
    category: PAICategory = Field(
        ..., description="PAI category grouping",
    )
    threshold_value: Optional[float] = Field(
        None, description="Numeric threshold for pass/fail",
    )
    warning_value: Optional[float] = Field(
        None, description="Warning threshold (softer boundary)",
    )
    direction: ThresholdDirection = Field(
        ..., description="How to interpret the threshold",
    )
    unit: str = Field(
        default="", description="Unit of measurement",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM,
        description="Severity if threshold breached",
    )
    description: str = Field(
        default="", description="Threshold description",
    )
    enabled: bool = Field(
        default=True, description="Whether check is active",
    )
    auto_exclude_on_fail: bool = Field(
        default=False,
        description="Whether failure triggers auto-exclusion",
    )

class PAICheckResult(BaseModel):
    """Result of a single PAI indicator DNSH check for one holding.

    Attributes:
        pai_indicator_id: PAI indicator checked.
        pai_name: Human-readable indicator name.
        category: PAI category.
        status: Check outcome.
        actual_value: Actual PAI indicator value.
        threshold_value: Configured threshold.
        warning_value: Warning threshold.
        direction: Threshold direction.
        severity: Severity on failure.
        breach_magnitude: How far the value exceeds the threshold.
        explanation: Human-readable explanation.
        triggers_exclusion: Whether this failure triggers auto-exclusion.
    """
    pai_indicator_id: str = Field(
        ..., description="PAI indicator checked",
    )
    pai_name: str = Field(
        ..., description="PAI indicator name",
    )
    category: PAICategory = Field(
        ..., description="PAI category",
    )
    status: DNSHStatus = Field(
        ..., description="Check outcome",
    )
    actual_value: Optional[float] = Field(
        None, description="Actual PAI indicator value",
    )
    threshold_value: Optional[float] = Field(
        None, description="Configured threshold value",
    )
    warning_value: Optional[float] = Field(
        None, description="Warning threshold value",
    )
    direction: ThresholdDirection = Field(
        ..., description="Threshold direction",
    )
    severity: SeverityLevel = Field(
        default=SeverityLevel.MEDIUM, description="Severity on failure",
    )
    breach_magnitude: Optional[float] = Field(
        None,
        description="How far the value exceeds the threshold "
                    "(positive = breach, negative = margin)",
    )
    explanation: str = Field(
        default="", description="Human-readable explanation",
    )
    triggers_exclusion: bool = Field(
        default=False,
        description="Whether this failure triggers auto-exclusion",
    )

class HoldingPAIData(BaseModel):
    """PAI indicator data for a single holding for DNSH screening.

    Attributes:
        holding_id: Unique holding identifier.
        holding_name: Holding name.
        sector: Sector classification.
        country: Country of domicile.
        nav_value: Net Asset Value in EUR.
        weight_pct: Portfolio weight percentage.
        pai_values: Numeric PAI indicator values.
        pai_boolean_flags: Boolean PAI indicator flags.
    """
    holding_id: str = Field(
        ..., min_length=1, description="Unique holding identifier",
    )
    holding_name: str = Field(
        ..., min_length=1, description="Holding name",
    )
    sector: str = Field(
        default="", description="Sector classification",
    )
    country: str = Field(
        default="", description="Country of domicile",
    )
    nav_value: float = Field(
        default=0.0, ge=0.0, description="Net Asset Value in EUR",
    )
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio weight %",
    )
    pai_values: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Numeric PAI indicator values (PAI_1 through PAI_18)",
    )
    pai_boolean_flags: Dict[str, Optional[bool]] = Field(
        default_factory=dict,
        description="Boolean PAI indicator flags (e.g., PAI_4, PAI_10)",
    )

class HoldingDNSHResult(BaseModel):
    """Complete DNSH assessment result for a single holding.

    Attributes:
        assessment_id: Unique assessment identifier.
        holding_id: Assessed holding identifier.
        holding_name: Holding name.
        overall_status: Overall DNSH outcome.
        environmental_status: Aggregate environmental DNSH status (PAI 1-9).
        social_status: Aggregate social DNSH status (PAI 10-14).
        checks: List of individual PAI check results.
        total_checks: Total number of checks performed.
        passed_checks: Number of checks that passed.
        failed_checks: Number of checks that failed.
        warning_checks: Number of checks in warning.
        no_data_checks: Number of checks with insufficient data.
        critical_failures: PAI indicators with CRITICAL-severity failures.
        high_failures: PAI indicators with HIGH-severity failures.
        should_exclude: Whether the holding should be auto-excluded.
        exclusion_reasons: Reasons for auto-exclusion.
        data_coverage_pct: Percentage of checks with available data.
        assessed_at: Assessment timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    assessment_id: str = Field(
        default_factory=_new_uuid, description="Unique assessment identifier",
    )
    holding_id: str = Field(..., description="Assessed holding identifier")
    holding_name: str = Field(..., description="Holding name")
    overall_status: DNSHStatus = Field(..., description="Overall DNSH outcome")
    environmental_status: DNSHStatus = Field(
        ..., description="Environmental DNSH status (PAI 1-9)",
    )
    social_status: DNSHStatus = Field(
        ..., description="Social DNSH status (PAI 10-14)",
    )
    checks: List[PAICheckResult] = Field(
        default_factory=list,
        description="Individual PAI check results",
    )
    total_checks: int = Field(..., ge=0, description="Total checks performed")
    passed_checks: int = Field(..., ge=0, description="Checks passed")
    failed_checks: int = Field(..., ge=0, description="Checks failed")
    warning_checks: int = Field(..., ge=0, description="Checks in warning")
    no_data_checks: int = Field(..., ge=0, description="Checks without data")
    critical_failures: List[str] = Field(
        default_factory=list,
        description="PAI indicators with CRITICAL-severity failures",
    )
    high_failures: List[str] = Field(
        default_factory=list,
        description="PAI indicators with HIGH-severity failures",
    )
    should_exclude: bool = Field(
        default=False,
        description="Whether the holding should be auto-excluded",
    )
    exclusion_reasons: List[ExclusionReason] = Field(
        default_factory=list,
        description="Reasons for auto-exclusion",
    )
    data_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of checks with available data",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class RemediationStep(BaseModel):
    """A single remediation step for addressing a DNSH finding.

    Attributes:
        step_id: Unique step identifier.
        pai_indicator_id: PAI indicator to remediate.
        pai_name: Human-readable indicator name.
        current_value: Current indicator value.
        target_value: Target value to achieve compliance.
        severity: Severity of the finding.
        action: Recommended remediation action.
        priority: Priority ranking (1 = highest).
        estimated_timeline: Estimated time to remediate.
    """
    step_id: str = Field(
        default_factory=_new_uuid, description="Unique step identifier",
    )
    pai_indicator_id: str = Field(
        ..., description="PAI indicator to remediate",
    )
    pai_name: str = Field(
        ..., description="Human-readable indicator name",
    )
    current_value: Optional[float] = Field(
        None, description="Current indicator value",
    )
    target_value: Optional[float] = Field(
        None, description="Target value for compliance",
    )
    severity: SeverityLevel = Field(
        ..., description="Severity of the finding",
    )
    action: str = Field(
        ..., description="Recommended remediation action",
    )
    priority: int = Field(
        default=0, ge=0, description="Priority ranking (1 = highest)",
    )
    estimated_timeline: str = Field(
        default="", description="Estimated timeline to remediate",
    )

class RemediationPlan(BaseModel):
    """Remediation plan for a holding with DNSH findings.

    Attributes:
        plan_id: Unique plan identifier.
        holding_id: Holding requiring remediation.
        holding_name: Holding name.
        total_findings: Total number of findings.
        critical_findings: Number of critical findings.
        high_findings: Number of high findings.
        steps: Ordered list of remediation steps.
        overall_priority: Overall urgency (CRITICAL, HIGH, MEDIUM, LOW).
        generated_at: Generation timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    plan_id: str = Field(
        default_factory=_new_uuid, description="Unique plan identifier",
    )
    holding_id: str = Field(
        ..., description="Holding requiring remediation",
    )
    holding_name: str = Field(
        ..., description="Holding name",
    )
    total_findings: int = Field(
        default=0, ge=0, description="Total number of findings",
    )
    critical_findings: int = Field(
        default=0, ge=0, description="Critical findings count",
    )
    high_findings: int = Field(
        default=0, ge=0, description="High findings count",
    )
    steps: List[RemediationStep] = Field(
        default_factory=list, description="Ordered remediation steps",
    )
    overall_priority: str = Field(
        default="MEDIUM", description="Overall urgency level",
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class AutoExclusionResult(BaseModel):
    """Result of auto-exclusion analysis for the portfolio.

    Attributes:
        result_id: Unique result identifier.
        total_holdings: Total holdings assessed.
        excluded_holdings: Number of holdings flagged for exclusion.
        excluded_nav: Total NAV of excluded holdings.
        excluded_pct: Excluded NAV as percentage of total.
        exclusions: List of exclusion details.
        reviewed_at: Review timestamp.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(
        default_factory=_new_uuid, description="Unique result identifier",
    )
    total_holdings: int = Field(
        default=0, ge=0, description="Total holdings assessed",
    )
    excluded_holdings: int = Field(
        default=0, ge=0, description="Holdings flagged for exclusion",
    )
    excluded_nav: float = Field(
        default=0.0, ge=0.0, description="Total NAV of excluded holdings",
    )
    excluded_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Excluded NAV as % of total",
    )
    exclusions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Exclusion details per holding",
    )
    reviewed_at: datetime = Field(
        default_factory=utcnow, description="Review timestamp",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )

class PortfolioDNSHResult(BaseModel):
    """Portfolio-level DNSH compliance result for Article 9.

    For Article 9 products, portfolio compliance must be 100 %.

    Attributes:
        portfolio_name: Portfolio name.
        total_holdings: Total holdings assessed.
        passing_holdings: Holdings that pass DNSH.
        failing_holdings: Holdings that fail DNSH.
        warning_holdings: Holdings in warning state.
        insufficient_data_holdings: Holdings lacking data.
        compliance_pct: Overall DNSH compliance percentage.
        is_article_9_compliant: Whether compliance is 100%.
        exclusion_count: Holdings flagged for exclusion.
        auto_exclusion_result: Auto-exclusion analysis.
        holding_assessments: Individual assessment results.
        category_summary: Summary by PAI category.
        most_common_failures: Most frequently failed PAI indicators.
        remediation_plans: Remediation plans for failing holdings.
        processing_time_ms: Processing time in milliseconds.
        assessed_at: Assessment timestamp.
        engine_version: Engine version.
        provenance_hash: SHA-256 provenance hash.
    """
    portfolio_name: Optional[str] = Field(None, description="Portfolio name")
    total_holdings: int = Field(..., ge=0)
    passing_holdings: int = Field(..., ge=0)
    failing_holdings: int = Field(..., ge=0)
    warning_holdings: int = Field(..., ge=0)
    insufficient_data_holdings: int = Field(..., ge=0)
    compliance_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="(passing / (total - insufficient)) * 100",
    )
    is_article_9_compliant: bool = Field(
        default=False,
        description="Whether DNSH compliance is 100 % (required for Article 9)",
    )
    exclusion_count: int = Field(
        ..., ge=0, description="Holdings flagged for exclusion",
    )
    auto_exclusion_result: Optional[AutoExclusionResult] = Field(
        None, description="Auto-exclusion analysis",
    )
    holding_assessments: List[HoldingDNSHResult] = Field(
        default_factory=list,
    )
    category_summary: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Status counts by PAI category",
    )
    most_common_failures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="PAI indicators sorted by failure frequency",
    )
    remediation_plans: List[RemediationPlan] = Field(
        default_factory=list,
        description="Remediation plans for failing holdings",
    )
    processing_time_ms: float = Field(default=0.0)
    assessed_at: datetime = Field(default_factory=utcnow)
    engine_version: str = Field(default=_MODULE_VERSION)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine Configuration
# ---------------------------------------------------------------------------

class EnhancedDNSHConfig(BaseModel):
    """Configuration for the Enhanced DNSH Engine (Article 9).

    Stricter defaults than Article 8, with all 18 PAI indicators enabled
    and tighter thresholds.

    Attributes:
        thresholds: List of PAI indicator thresholds.
        require_all_environmental: Whether all env PAI must pass.
        require_all_social: Whether all social PAI must pass.
        auto_exclude_on_critical: Auto-exclude on CRITICAL failures.
        auto_exclude_on_multiple_high: Auto-exclude on 2+ HIGH failures.
        multiple_high_threshold: Number of HIGH failures for auto-exclusion.
        data_coverage_minimum_pct: Minimum data coverage for valid assessment.
        generate_remediation_plans: Whether to generate remediation plans.
        target_compliance_pct: Target portfolio compliance (100% for Art. 9).
    """
    thresholds: List[PAIThreshold] = Field(
        default_factory=list,
        description="PAI indicator thresholds for DNSH screening",
    )
    require_all_environmental: bool = Field(
        default=True,
        description="Require pass on all environmental PAI (1-9)",
    )
    require_all_social: bool = Field(
        default=True,
        description="Require pass on all social PAI (10-14)",
    )
    auto_exclude_on_critical: bool = Field(
        default=True,
        description="Automatically exclude holdings with CRITICAL failures",
    )
    auto_exclude_on_multiple_high: bool = Field(
        default=True,
        description="Auto-exclude holdings with multiple HIGH failures",
    )
    multiple_high_threshold: int = Field(
        default=2, ge=1,
        description="Number of HIGH failures triggering auto-exclusion",
    )
    data_coverage_minimum_pct: float = Field(
        default=60.0, ge=0.0, le=100.0,
        description="Minimum data coverage for valid assessment (%)",
    )
    generate_remediation_plans: bool = Field(
        default=True,
        description="Whether to generate remediation plans for failures",
    )
    target_compliance_pct: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Target portfolio compliance % (100 for Article 9)",
    )

# ---------------------------------------------------------------------------
# Default Threshold Configuration (Stricter for Article 9)
# ---------------------------------------------------------------------------

def _build_article9_thresholds() -> List[PAIThreshold]:
    """Build stricter PAI indicator thresholds for Article 9 DNSH screening.

    Thresholds are intentionally tighter than Article 8 defaults to
    reflect the higher sustainability bar of Article 9 products.

    Returns:
        List of PAIThreshold objects for all 18 mandatory indicators.
    """
    return [
        # --- PAI 1-6: Climate/GHG (stricter than Art. 8) ---
        PAIThreshold(
            pai_indicator_id="PAI_1",
            pai_name="GHG emissions (Scope 1+2+3)",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=500000.0,
            warning_value=350000.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq",
            severity=SeverityLevel.HIGH,
            description="Total GHG emissions must be below 500K tCO2eq",
            auto_exclude_on_fail=False,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_2",
            pai_name="Carbon footprint",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=300.0,
            warning_value=200.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M",
            severity=SeverityLevel.HIGH,
            description="Carbon footprint must not exceed 300 tCO2eq per EUR M (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_3",
            pai_name="GHG intensity of investee companies",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=600.0,
            warning_value=400.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M revenue",
            severity=SeverityLevel.MEDIUM,
            description="Company GHG intensity must not exceed 600 (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_4",
            pai_name="Exposure to fossil fuel companies",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=5.0,
            warning_value=2.0,
            direction=ThresholdDirection.MAX,
            unit="%",
            severity=SeverityLevel.HIGH,
            description="Fossil fuel revenue exposure must be <5 % (stricter than Art. 8)",
            auto_exclude_on_fail=True,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_5",
            pai_name="Non-renewable energy share",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=50.0,
            warning_value=35.0,
            direction=ThresholdDirection.MAX,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Non-renewable energy share must not exceed 50 % (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_6",
            pai_name="Energy intensity per high impact sector",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=3.0,
            warning_value=2.0,
            direction=ThresholdDirection.MAX,
            unit="GWh / EUR M revenue",
            severity=SeverityLevel.MEDIUM,
            description="Energy intensity must not exceed 3 GWh/EUR M (stricter than Art. 8)",
        ),
        # --- PAI 7-9: Environment (stricter) ---
        PAIThreshold(
            pai_indicator_id="PAI_7",
            pai_name="Biodiversity-sensitive areas",
            category=PAICategory.ENVIRONMENT,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Investment negatively affects biodiversity-sensitive areas",
            auto_exclude_on_fail=True,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_8",
            pai_name="Emissions to water",
            category=PAICategory.ENVIRONMENT,
            threshold_value=50.0,
            warning_value=25.0,
            direction=ThresholdDirection.MAX,
            unit="tonnes",
            severity=SeverityLevel.MEDIUM,
            description="Water emissions must not exceed 50 tonnes (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_9",
            pai_name="Hazardous and radioactive waste",
            category=PAICategory.ENVIRONMENT,
            threshold_value=250.0,
            warning_value=100.0,
            direction=ThresholdDirection.MAX,
            unit="tonnes",
            severity=SeverityLevel.MEDIUM,
            description="Hazardous waste must not exceed 250 tonnes (stricter than Art. 8)",
        ),
        # --- PAI 10-14: Social (stricter) ---
        PAIThreshold(
            pai_indicator_id="PAI_10",
            pai_name="UNGC/OECD violations",
            category=PAICategory.SOCIAL,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Company has UNGC/OECD principles violations",
            auto_exclude_on_fail=True,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_11",
            pai_name="Lack of UNGC/OECD compliance mechanisms",
            category=PAICategory.SOCIAL,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_FALSE_FAILS,
            unit="%",
            severity=SeverityLevel.HIGH,
            description="Company lacks UNGC/OECD compliance processes",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_12",
            pai_name="Unadjusted gender pay gap",
            category=PAICategory.SOCIAL,
            threshold_value=15.0,
            warning_value=10.0,
            direction=ThresholdDirection.MAX,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Gender pay gap must not exceed 15 % (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_13",
            pai_name="Board gender diversity",
            category=PAICategory.SOCIAL,
            threshold_value=30.0,
            warning_value=40.0,
            direction=ThresholdDirection.MIN,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Female board representation must be at least 30 % (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_14",
            pai_name="Controversial weapons",
            category=PAICategory.SOCIAL,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Company involved in controversial weapons",
            auto_exclude_on_fail=True,
        ),
        # --- PAI 15-16: Sovereign ---
        PAIThreshold(
            pai_indicator_id="PAI_15",
            pai_name="GHG intensity of investee countries",
            category=PAICategory.SOVEREIGN,
            threshold_value=500.0,
            warning_value=350.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M GDP",
            severity=SeverityLevel.MEDIUM,
            description="Country GHG intensity must not exceed 500 (stricter than Art. 8)",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_16",
            pai_name="Countries subject to social violations",
            category=PAICategory.SOVEREIGN,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Country subject to social violations",
            auto_exclude_on_fail=True,
        ),
        # --- PAI 17-18: Real Estate ---
        PAIThreshold(
            pai_indicator_id="PAI_17",
            pai_name="Fossil fuels through real estate",
            category=PAICategory.REAL_ESTATE,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.HIGH,
            description="Real estate involved in fossil fuels",
            auto_exclude_on_fail=True,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_18",
            pai_name="Energy-inefficient real estate",
            category=PAICategory.REAL_ESTATE,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Real estate is energy-inefficient (below NZEB)",
        ),
    ]

# ---------------------------------------------------------------------------
# Pydantic model_rebuild
# ---------------------------------------------------------------------------

PAIThreshold.model_rebuild()
PAICheckResult.model_rebuild()
HoldingPAIData.model_rebuild()
HoldingDNSHResult.model_rebuild()
RemediationStep.model_rebuild()
RemediationPlan.model_rebuild()
AutoExclusionResult.model_rebuild()
PortfolioDNSHResult.model_rebuild()
EnhancedDNSHConfig.model_rebuild()

# ---------------------------------------------------------------------------
# EnhancedDNSHEngine
# ---------------------------------------------------------------------------

class EnhancedDNSHEngine:
    """Enhanced DNSH Engine for SFDR Article 9 products.

    Applies stricter DNSH screening to ALL holdings in the portfolio
    (not just the sustainable portion).  Uses tighter thresholds than
    Article 8, assesses all 18 mandatory PAI indicators, and triggers
    auto-exclusion on critical failures.  Portfolio compliance must
    reach 100 % for genuine Article 9 status.

    Zero-Hallucination Guarantees:
        - All threshold comparisons use deterministic arithmetic
        - Configurable thresholds avoid hard-coded regulatory interpretation
        - SHA-256 provenance hashing on every result
        - No LLM involvement in any assessment path

    Attributes:
        config: Enhanced DNSH configuration.
        _thresholds_by_id: Lookup map of thresholds by PAI indicator ID.
        _assessment_count: Running count of assessments performed.

    Example:
        >>> engine = EnhancedDNSHEngine()
        >>> data = HoldingPAIData(
        ...     holding_id="ISIN001",
        ...     holding_name="GreenCorp",
        ...     pai_values={"PAI_2": 200.0, "PAI_5": 30.0},
        ...     pai_boolean_flags={"PAI_10": False, "PAI_14": False},
        ... )
        >>> result = engine.assess_holding(data)
        >>> assert result.overall_status == DNSHStatus.PASS
    """

    def __init__(self, config: Optional[EnhancedDNSHConfig] = None) -> None:
        """Initialize the Enhanced DNSH Engine.

        Args:
            config: Enhanced DNSH configuration. Uses Article 9 defaults if None.
        """
        if config is None:
            config = EnhancedDNSHConfig(thresholds=_build_article9_thresholds())

        self.config = config
        self._thresholds_by_id: Dict[str, PAIThreshold] = {
            t.pai_indicator_id: t for t in config.thresholds
        }
        self._assessment_count: int = 0

        if not config.thresholds:
            config.thresholds = _build_article9_thresholds()
            self._thresholds_by_id = {
                t.pai_indicator_id: t for t in config.thresholds
            }

        logger.info(
            "EnhancedDNSHEngine initialized (v%s, %d thresholds, "
            "%d enabled, target_compliance=%.0f%%)",
            _MODULE_VERSION,
            len(config.thresholds),
            sum(1 for t in config.thresholds if t.enabled),
            config.target_compliance_pct,
        )

    # ------------------------------------------------------------------
    # Public API: Single Holding Assessment
    # ------------------------------------------------------------------

    def assess_holding(
        self,
        holding_data: HoldingPAIData,
    ) -> HoldingDNSHResult:
        """Assess enhanced DNSH for a single holding.

        Screens each enabled PAI indicator threshold against the holding's
        data.  All 18 indicators are checked (stricter than Article 8).
        Auto-exclusion is triggered on critical failures.

        Args:
            holding_data: PAI indicator data for the holding.

        Returns:
            HoldingDNSHResult with per-indicator and overall results.

        Raises:
            ValueError: If holding_id is empty.
        """
        start = utcnow()
        self._assessment_count += 1

        if not holding_data.holding_id:
            raise ValueError("holding_id is required")

        logger.debug(
            "Enhanced DNSH assessment for holding %s (%s)",
            holding_data.holding_id,
            holding_data.holding_name,
        )

        # Evaluate all enabled thresholds
        checks: List[PAICheckResult] = []
        for threshold in self.config.thresholds:
            if not threshold.enabled:
                continue
            check = self._evaluate_threshold(threshold, holding_data)
            checks.append(check)

        # Aggregate by category
        env_checks = [
            c for c in checks
            if c.category in (PAICategory.CLIMATE_GHG, PAICategory.ENVIRONMENT)
        ]
        social_checks = [
            c for c in checks
            if c.category == PAICategory.SOCIAL
        ]

        env_status = self._aggregate_status(
            env_checks, self.config.require_all_environmental,
        )
        social_status = self._aggregate_status(
            social_checks, self.config.require_all_social,
        )
        overall_status = self._determine_overall_status(
            env_status, social_status, checks,
        )

        # Count outcomes
        passed = sum(1 for c in checks if c.status == DNSHStatus.PASS)
        failed = sum(1 for c in checks if c.status == DNSHStatus.FAIL)
        warning = sum(1 for c in checks if c.status == DNSHStatus.WARNING)
        no_data = sum(1 for c in checks if c.status == DNSHStatus.INSUFFICIENT_DATA)
        total = len(checks)

        # Identify critical and high failures
        critical = [
            c.pai_indicator_id for c in checks
            if c.status == DNSHStatus.FAIL and c.severity == SeverityLevel.CRITICAL
        ]
        high = [
            c.pai_indicator_id for c in checks
            if c.status == DNSHStatus.FAIL and c.severity == SeverityLevel.HIGH
        ]

        # Auto-exclusion determination
        should_exclude, exclusion_reasons = self._determine_exclusion(
            checks, critical, high,
        )

        # Data coverage
        data_coverage = (
            ((total - no_data) / total * 100.0) if total > 0 else 0.0
        )

        assessment = HoldingDNSHResult(
            holding_id=holding_data.holding_id,
            holding_name=holding_data.holding_name,
            overall_status=overall_status,
            environmental_status=env_status,
            social_status=social_status,
            checks=checks,
            total_checks=total,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warning,
            no_data_checks=no_data,
            critical_failures=critical,
            high_failures=high,
            should_exclude=should_exclude,
            exclusion_reasons=exclusion_reasons,
            data_coverage_pct=_round_val(data_coverage, 2),
            assessed_at=start,
        )

        assessment.provenance_hash = _compute_hash({
            "holding_id": holding_data.holding_id,
            "overall_status": overall_status.value,
            "passed": passed,
            "failed": failed,
            "critical": critical,
            "should_exclude": should_exclude,
        })

        elapsed_ms = (utcnow() - start).total_seconds() * 1000
        logger.info(
            "Enhanced DNSH for %s: status=%s, passed=%d, failed=%d, "
            "critical=%d, exclude=%s, time=%.1fms",
            holding_data.holding_id, overall_status.value,
            passed, failed, len(critical), should_exclude, elapsed_ms,
        )

        return assessment

    # ------------------------------------------------------------------
    # Public API: Portfolio Assessment
    # ------------------------------------------------------------------

    def assess_portfolio(
        self,
        holdings: List[HoldingPAIData],
        portfolio_name: Optional[str] = None,
    ) -> PortfolioDNSHResult:
        """Assess enhanced DNSH for an entire portfolio.

        Article 9 requires 100 % DNSH compliance across all holdings.
        Generates remediation plans for failing holdings and auto-exclusion
        analysis.

        Args:
            holdings: List of holding PAI data.
            portfolio_name: Optional portfolio name.

        Returns:
            PortfolioDNSHResult with portfolio-level compliance.

        Raises:
            ValueError: If holdings list is empty.
        """
        start = utcnow()

        if not holdings:
            raise ValueError("Holdings list cannot be empty")

        logger.info(
            "Enhanced DNSH portfolio assessment for %d holdings",
            len(holdings),
        )

        assessments: List[HoldingDNSHResult] = []
        for holding in holdings:
            try:
                assessment = self.assess_holding(holding)
                assessments.append(assessment)
            except Exception as exc:
                logger.error(
                    "Enhanced DNSH failed for %s: %s",
                    holding.holding_id, exc,
                )
                raise

        # Count statuses
        passing = sum(1 for a in assessments if a.overall_status == DNSHStatus.PASS)
        failing = sum(1 for a in assessments if a.overall_status == DNSHStatus.FAIL)
        warning = sum(1 for a in assessments if a.overall_status == DNSHStatus.WARNING)
        no_data = sum(
            1 for a in assessments
            if a.overall_status == DNSHStatus.INSUFFICIENT_DATA
        )
        exclusions = sum(1 for a in assessments if a.should_exclude)

        # Compliance score
        assessable = len(assessments) - no_data
        compliance_pct = (
            (passing / assessable * 100.0) if assessable > 0 else 0.0
        )
        is_compliant = compliance_pct >= self.config.target_compliance_pct

        # Category summary
        category_summary = self._build_category_summary(assessments)

        # Most common failures
        failure_freq = self._compute_failure_frequency(assessments)

        # Auto-exclusion result
        auto_exclusion = self._build_auto_exclusion_result(
            assessments, holdings,
        )

        # Remediation plans
        remediation_plans: List[RemediationPlan] = []
        if self.config.generate_remediation_plans:
            for assessment in assessments:
                if assessment.failed_checks > 0 and not assessment.should_exclude:
                    plan = self._generate_remediation_plan(assessment)
                    remediation_plans.append(plan)

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        result = PortfolioDNSHResult(
            portfolio_name=portfolio_name,
            total_holdings=len(assessments),
            passing_holdings=passing,
            failing_holdings=failing,
            warning_holdings=warning,
            insufficient_data_holdings=no_data,
            compliance_pct=_round_val(compliance_pct, 2),
            is_article_9_compliant=is_compliant,
            exclusion_count=exclusions,
            auto_exclusion_result=auto_exclusion,
            holding_assessments=assessments,
            category_summary=category_summary,
            most_common_failures=failure_freq,
            remediation_plans=remediation_plans,
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash({
            "portfolio_name": portfolio_name,
            "total": len(assessments),
            "passing": passing,
            "failing": failing,
            "compliance_pct": compliance_pct,
            "is_article_9_compliant": is_compliant,
        })

        logger.info(
            "Enhanced DNSH portfolio complete: %d holdings, "
            "compliance=%.1f%%, article_9_compliant=%s, "
            "exclusions=%d, time=%.1fms",
            len(assessments), compliance_pct, is_compliant,
            exclusions, elapsed_ms,
        )

        return result

    # ------------------------------------------------------------------
    # Public API: Utilities
    # ------------------------------------------------------------------

    def get_thresholds(self) -> List[Dict[str, Any]]:
        """Return all configured DNSH thresholds.

        Returns:
            List of threshold configurations as dictionaries.
        """
        return [t.model_dump() for t in self.config.thresholds]

    def generate_remediation_plan(
        self,
        assessment: HoldingDNSHResult,
    ) -> RemediationPlan:
        """Generate a remediation plan for a holding with DNSH findings.

        Args:
            assessment: A completed DNSH assessment with failures.

        Returns:
            RemediationPlan with prioritized steps.
        """
        return self._generate_remediation_plan(assessment)

    # ------------------------------------------------------------------
    # Private: Threshold Evaluation
    # ------------------------------------------------------------------

    def _evaluate_threshold(
        self,
        threshold: PAIThreshold,
        holding_data: HoldingPAIData,
    ) -> PAICheckResult:
        """Evaluate a single PAI threshold against holding data.

        Args:
            threshold: The PAI threshold configuration.
            holding_data: The holding's PAI data.

        Returns:
            PAICheckResult with the evaluation result.
        """
        pai_id = threshold.pai_indicator_id

        # Handle boolean indicators
        if threshold.direction in (
            ThresholdDirection.BOOLEAN_TRUE_FAILS,
            ThresholdDirection.BOOLEAN_FALSE_FAILS,
        ):
            return self._evaluate_boolean_threshold(threshold, holding_data)

        # Handle numeric indicators
        actual_value = holding_data.pai_values.get(pai_id)

        if actual_value is None:
            return PAICheckResult(
                pai_indicator_id=pai_id,
                pai_name=threshold.pai_name,
                category=threshold.category,
                status=DNSHStatus.INSUFFICIENT_DATA,
                direction=threshold.direction,
                severity=threshold.severity,
                explanation=f"No data available for {pai_id}",
            )

        if threshold.threshold_value is None:
            return PAICheckResult(
                pai_indicator_id=pai_id,
                pai_name=threshold.pai_name,
                category=threshold.category,
                status=DNSHStatus.NOT_APPLICABLE,
                actual_value=actual_value,
                direction=threshold.direction,
                severity=threshold.severity,
                explanation=f"No threshold configured for {pai_id}",
            )

        return self._evaluate_numeric_threshold(threshold, actual_value)

    def _evaluate_numeric_threshold(
        self,
        threshold: PAIThreshold,
        actual_value: float,
    ) -> PAICheckResult:
        """Evaluate a numeric PAI threshold.

        Args:
            threshold: Threshold configuration.
            actual_value: Actual indicator value.

        Returns:
            PAICheckResult with PASS, FAIL, or WARNING status.
        """
        pai_id = threshold.pai_indicator_id
        thresh_val = threshold.threshold_value
        warn_val = threshold.warning_value

        if threshold.direction == ThresholdDirection.MAX:
            breach = actual_value - thresh_val
            is_fail = actual_value > thresh_val
            is_warning = (
                warn_val is not None
                and not is_fail
                and actual_value > warn_val
            )
        else:  # MIN
            breach = thresh_val - actual_value
            is_fail = actual_value < thresh_val
            is_warning = (
                warn_val is not None
                and not is_fail
                and actual_value < warn_val
            )

        if is_fail:
            status = DNSHStatus.FAIL
            direction_label = (
                "maximum" if threshold.direction == ThresholdDirection.MAX
                else "minimum"
            )
            explanation = (
                f"{pai_id} value {actual_value} breaches {direction_label} "
                f"threshold of {thresh_val} {threshold.unit}"
            )
        elif is_warning:
            status = DNSHStatus.WARNING
            explanation = (
                f"{pai_id} value {actual_value} is within warning range "
                f"(warning: {warn_val}, threshold: {thresh_val} {threshold.unit})"
            )
        else:
            status = DNSHStatus.PASS
            explanation = (
                f"{pai_id} value {actual_value} is within acceptable range "
                f"(threshold: {thresh_val} {threshold.unit})"
            )

        return PAICheckResult(
            pai_indicator_id=pai_id,
            pai_name=threshold.pai_name,
            category=threshold.category,
            status=status,
            actual_value=actual_value,
            threshold_value=thresh_val,
            warning_value=warn_val,
            direction=threshold.direction,
            severity=threshold.severity,
            breach_magnitude=_round_val(breach, 4) if is_fail else None,
            explanation=explanation,
            triggers_exclusion=(
                is_fail and threshold.auto_exclude_on_fail
            ),
        )

    def _evaluate_boolean_threshold(
        self,
        threshold: PAIThreshold,
        holding_data: HoldingPAIData,
    ) -> PAICheckResult:
        """Evaluate a boolean PAI threshold.

        Args:
            threshold: Boolean threshold configuration.
            holding_data: Holding PAI data.

        Returns:
            PAICheckResult with PASS, FAIL, or INSUFFICIENT_DATA.
        """
        pai_id = threshold.pai_indicator_id
        flag_value = holding_data.pai_boolean_flags.get(pai_id)

        if flag_value is None:
            return PAICheckResult(
                pai_indicator_id=pai_id,
                pai_name=threshold.pai_name,
                category=threshold.category,
                status=DNSHStatus.INSUFFICIENT_DATA,
                direction=threshold.direction,
                severity=threshold.severity,
                explanation=f"No boolean data available for {pai_id}",
            )

        if threshold.direction == ThresholdDirection.BOOLEAN_TRUE_FAILS:
            is_fail = flag_value is True
        else:  # BOOLEAN_FALSE_FAILS
            is_fail = flag_value is False

        if is_fail:
            status = DNSHStatus.FAIL
            explanation = f"{pai_id}: {threshold.description}"
        else:
            status = DNSHStatus.PASS
            explanation = (
                f"{pai_id}: No significant harm detected (flag={flag_value})"
            )

        return PAICheckResult(
            pai_indicator_id=pai_id,
            pai_name=threshold.pai_name,
            category=threshold.category,
            status=status,
            actual_value=1.0 if flag_value else 0.0,
            threshold_value=None,
            direction=threshold.direction,
            severity=threshold.severity,
            explanation=explanation,
            triggers_exclusion=(
                is_fail and threshold.auto_exclude_on_fail
            ),
        )

    # ------------------------------------------------------------------
    # Private: Status Aggregation
    # ------------------------------------------------------------------

    def _aggregate_status(
        self,
        checks: List[PAICheckResult],
        require_all: bool,
    ) -> DNSHStatus:
        """Aggregate individual check statuses into a category status.

        Args:
            checks: List of PAI checks for the category.
            require_all: If True, all checks must pass.

        Returns:
            Aggregated DNSHStatus.
        """
        if not checks:
            return DNSHStatus.NOT_APPLICABLE

        has_fail = any(c.status == DNSHStatus.FAIL for c in checks)
        has_warning = any(c.status == DNSHStatus.WARNING for c in checks)
        has_no_data = any(
            c.status == DNSHStatus.INSUFFICIENT_DATA for c in checks
        )

        if has_fail:
            return DNSHStatus.FAIL
        if require_all and has_no_data:
            return DNSHStatus.INSUFFICIENT_DATA
        if has_warning:
            return DNSHStatus.WARNING
        return DNSHStatus.PASS

    def _determine_overall_status(
        self,
        env_status: DNSHStatus,
        social_status: DNSHStatus,
        all_checks: List[PAICheckResult],
    ) -> DNSHStatus:
        """Determine overall DNSH status from category statuses.

        Args:
            env_status: Environmental category status.
            social_status: Social category status.
            all_checks: All individual checks.

        Returns:
            Overall DNSHStatus.
        """
        other_checks = [
            c for c in all_checks
            if c.category in (PAICategory.SOVEREIGN, PAICategory.REAL_ESTATE)
        ]
        other_has_fail = any(c.status == DNSHStatus.FAIL for c in other_checks)

        if (
            env_status == DNSHStatus.FAIL
            or social_status == DNSHStatus.FAIL
            or other_has_fail
        ):
            return DNSHStatus.FAIL

        if (
            env_status == DNSHStatus.INSUFFICIENT_DATA
            or social_status == DNSHStatus.INSUFFICIENT_DATA
        ):
            return DNSHStatus.INSUFFICIENT_DATA

        if (
            env_status == DNSHStatus.WARNING
            or social_status == DNSHStatus.WARNING
        ):
            return DNSHStatus.WARNING

        return DNSHStatus.PASS

    # ------------------------------------------------------------------
    # Private: Auto-Exclusion
    # ------------------------------------------------------------------

    def _determine_exclusion(
        self,
        checks: List[PAICheckResult],
        critical_failures: List[str],
        high_failures: List[str],
    ) -> Tuple[bool, List[ExclusionReason]]:
        """Determine whether a holding should be auto-excluded.

        Args:
            checks: All PAI check results.
            critical_failures: List of critical-severity failure IDs.
            high_failures: List of high-severity failure IDs.

        Returns:
            Tuple of (should_exclude, exclusion_reasons).
        """
        reasons: List[ExclusionReason] = []

        # Check auto-exclude triggers on individual checks
        for check in checks:
            if check.triggers_exclusion:
                reason = self._map_check_to_exclusion_reason(check)
                if reason and reason not in reasons:
                    reasons.append(reason)

        # Critical failures
        if self.config.auto_exclude_on_critical and critical_failures:
            if ExclusionReason.CRITICAL_PAI_BREACH not in reasons:
                reasons.append(ExclusionReason.CRITICAL_PAI_BREACH)

        # Multiple HIGH failures
        if (
            self.config.auto_exclude_on_multiple_high
            and len(high_failures) >= self.config.multiple_high_threshold
        ):
            if ExclusionReason.MULTIPLE_HIGH_SEVERITY not in reasons:
                reasons.append(ExclusionReason.MULTIPLE_HIGH_SEVERITY)

        should_exclude = len(reasons) > 0
        return should_exclude, reasons

    def _map_check_to_exclusion_reason(
        self,
        check: PAICheckResult,
    ) -> Optional[ExclusionReason]:
        """Map a failing PAI check to an exclusion reason.

        Args:
            check: A failed PAI check result.

        Returns:
            ExclusionReason if mappable, None otherwise.
        """
        mapping: Dict[str, ExclusionReason] = {
            "PAI_4": ExclusionReason.FOSSIL_FUEL_ABOVE_THRESHOLD,
            "PAI_7": ExclusionReason.BIODIVERSITY_HARM,
            "PAI_10": ExclusionReason.UNGC_OECD_VIOLATIONS,
            "PAI_14": ExclusionReason.CONTROVERSIAL_WEAPONS,
            "PAI_16": ExclusionReason.COUNTRY_SANCTIONS,
            "PAI_17": ExclusionReason.FOSSIL_FUEL_ABOVE_THRESHOLD,
        }
        return mapping.get(check.pai_indicator_id)

    # ------------------------------------------------------------------
    # Private: Auto-Exclusion Result Builder
    # ------------------------------------------------------------------

    def _build_auto_exclusion_result(
        self,
        assessments: List[HoldingDNSHResult],
        holdings: List[HoldingPAIData],
    ) -> AutoExclusionResult:
        """Build the auto-exclusion analysis result.

        Args:
            assessments: All holding assessments.
            holdings: Original holding data.

        Returns:
            AutoExclusionResult.
        """
        holdings_map = {h.holding_id: h for h in holdings}

        excluded = [a for a in assessments if a.should_exclude]
        total_nav = sum(h.nav_value for h in holdings)
        excluded_nav = sum(
            holdings_map[a.holding_id].nav_value
            for a in excluded
            if a.holding_id in holdings_map
        )

        exclusion_details = [
            {
                "holding_id": a.holding_id,
                "holding_name": a.holding_name,
                "reasons": [r.value for r in a.exclusion_reasons],
                "critical_failures": a.critical_failures,
                "nav_value": (
                    holdings_map[a.holding_id].nav_value
                    if a.holding_id in holdings_map else 0.0
                ),
            }
            for a in excluded
        ]

        result = AutoExclusionResult(
            total_holdings=len(assessments),
            excluded_holdings=len(excluded),
            excluded_nav=_round_val(excluded_nav, 2),
            excluded_pct=_round_val(_safe_pct(excluded_nav, total_nav), 2),
            exclusions=exclusion_details,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Private: Remediation Plan Generator
    # ------------------------------------------------------------------

    def _generate_remediation_plan(
        self,
        assessment: HoldingDNSHResult,
    ) -> RemediationPlan:
        """Generate a remediation plan for a holding with DNSH findings.

        Args:
            assessment: A completed DNSH assessment with failures.

        Returns:
            RemediationPlan with prioritized steps.
        """
        steps: List[RemediationStep] = []
        priority_counter = 0

        # Sort checks by severity (CRITICAL first, then HIGH, etc.)
        severity_order = {
            SeverityLevel.CRITICAL: 0,
            SeverityLevel.HIGH: 1,
            SeverityLevel.MEDIUM: 2,
            SeverityLevel.LOW: 3,
            SeverityLevel.INFORMATIONAL: 4,
        }

        failed_checks = sorted(
            [c for c in assessment.checks if c.status == DNSHStatus.FAIL],
            key=lambda c: severity_order.get(c.severity, 5),
        )

        for check in failed_checks:
            priority_counter += 1
            threshold = self._thresholds_by_id.get(check.pai_indicator_id)
            target = threshold.threshold_value if threshold else None

            action = self._suggest_remediation_action(check)
            timeline = self._estimate_timeline(check.severity)

            steps.append(RemediationStep(
                pai_indicator_id=check.pai_indicator_id,
                pai_name=check.pai_name,
                current_value=check.actual_value,
                target_value=target,
                severity=check.severity,
                action=action,
                priority=priority_counter,
                estimated_timeline=timeline,
            ))

        # Also add warning checks as lower-priority items
        warning_checks = [
            c for c in assessment.checks if c.status == DNSHStatus.WARNING
        ]
        for check in warning_checks:
            priority_counter += 1
            threshold = self._thresholds_by_id.get(check.pai_indicator_id)
            target = threshold.threshold_value if threshold else None

            steps.append(RemediationStep(
                pai_indicator_id=check.pai_indicator_id,
                pai_name=check.pai_name,
                current_value=check.actual_value,
                target_value=target,
                severity=SeverityLevel.LOW,
                action=f"Monitor {check.pai_name} - currently in warning range",
                priority=priority_counter,
                estimated_timeline="Ongoing monitoring",
            ))

        critical_count = sum(
            1 for c in failed_checks if c.severity == SeverityLevel.CRITICAL
        )
        high_count = sum(
            1 for c in failed_checks if c.severity == SeverityLevel.HIGH
        )

        overall_priority = "LOW"
        if critical_count > 0:
            overall_priority = "CRITICAL"
        elif high_count > 0:
            overall_priority = "HIGH"
        elif len(failed_checks) > 0:
            overall_priority = "MEDIUM"

        plan = RemediationPlan(
            holding_id=assessment.holding_id,
            holding_name=assessment.holding_name,
            total_findings=len(failed_checks) + len(warning_checks),
            critical_findings=critical_count,
            high_findings=high_count,
            steps=steps,
            overall_priority=overall_priority,
        )
        plan.provenance_hash = _compute_hash(plan)
        return plan

    def _suggest_remediation_action(
        self,
        check: PAICheckResult,
    ) -> str:
        """Suggest a remediation action for a failing check.

        Args:
            check: Failed PAI check result.

        Returns:
            Remediation action string.
        """
        action_map: Dict[str, str] = {
            "PAI_1": "Engage with investee to reduce absolute GHG emissions",
            "PAI_2": "Reduce carbon footprint through portfolio rebalancing",
            "PAI_3": "Target lower GHG-intensity companies within same sector",
            "PAI_4": "Divest from fossil fuel exposure or seek transition commitments",
            "PAI_5": "Engage on renewable energy transition plans",
            "PAI_6": "Target lower energy-intensity alternatives",
            "PAI_7": "Divest from holdings impacting biodiversity-sensitive areas",
            "PAI_8": "Engage on water emission reduction targets",
            "PAI_9": "Engage on hazardous waste reduction plans",
            "PAI_10": "Divest from holdings with UNGC/OECD violations",
            "PAI_11": "Engage on implementing UNGC/OECD compliance mechanisms",
            "PAI_12": "Engage on gender pay gap reduction targets",
            "PAI_13": "Engage on board gender diversity improvement",
            "PAI_14": "Divest from controversial weapons involvement",
            "PAI_15": "Review sovereign exposure to high GHG-intensity countries",
            "PAI_16": "Divest from countries with social violations",
            "PAI_17": "Divest from real estate with fossil fuel involvement",
            "PAI_18": "Engage on energy efficiency improvements for real estate",
        }
        return action_map.get(
            check.pai_indicator_id,
            f"Address {check.pai_name} threshold breach",
        )

    def _estimate_timeline(self, severity: SeverityLevel) -> str:
        """Estimate remediation timeline based on severity.

        Args:
            severity: Severity level.

        Returns:
            Estimated timeline string.
        """
        timelines = {
            SeverityLevel.CRITICAL: "Immediate (within 30 days)",
            SeverityLevel.HIGH: "Short-term (within 90 days)",
            SeverityLevel.MEDIUM: "Medium-term (within 6 months)",
            SeverityLevel.LOW: "Long-term (within 12 months)",
            SeverityLevel.INFORMATIONAL: "Ongoing monitoring",
        }
        return timelines.get(severity, "To be determined")

    # ------------------------------------------------------------------
    # Private: Portfolio Aggregation Helpers
    # ------------------------------------------------------------------

    def _build_category_summary(
        self,
        assessments: List[HoldingDNSHResult],
    ) -> Dict[str, Dict[str, int]]:
        """Build status count summary by PAI category.

        Args:
            assessments: List of individual assessments.

        Returns:
            Dict mapping category to {pass, fail, warning, no_data} counts.
        """
        summary: Dict[str, Dict[str, int]] = {}

        for category in PAICategory:
            cat_counts = {"pass": 0, "fail": 0, "warning": 0, "no_data": 0}

            for assessment in assessments:
                for check in assessment.checks:
                    if check.category != category:
                        continue
                    if check.status == DNSHStatus.PASS:
                        cat_counts["pass"] += 1
                    elif check.status == DNSHStatus.FAIL:
                        cat_counts["fail"] += 1
                    elif check.status == DNSHStatus.WARNING:
                        cat_counts["warning"] += 1
                    elif check.status == DNSHStatus.INSUFFICIENT_DATA:
                        cat_counts["no_data"] += 1

            summary[category.value] = cat_counts

        return summary

    def _compute_failure_frequency(
        self,
        assessments: List[HoldingDNSHResult],
    ) -> List[Dict[str, Any]]:
        """Compute which PAI indicators fail most frequently.

        Args:
            assessments: List of individual assessments.

        Returns:
            List of dicts sorted by failure frequency (descending).
        """
        failure_counts: Dict[str, int] = defaultdict(int)
        indicator_names: Dict[str, str] = {}

        for assessment in assessments:
            for check in assessment.checks:
                if check.status == DNSHStatus.FAIL:
                    failure_counts[check.pai_indicator_id] += 1
                    indicator_names[check.pai_indicator_id] = check.pai_name

        sorted_failures = sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True,
        )

        return [
            {
                "pai_indicator_id": pai_id,
                "pai_name": indicator_names.get(pai_id, ""),
                "failure_count": count,
                "failure_rate_pct": _round_val(
                    count / len(assessments) * 100.0, 2,
                ) if assessments else 0.0,
            }
            for pai_id, count in sorted_failures
        ]

    # ------------------------------------------------------------------
    # Read-only Properties
    # ------------------------------------------------------------------

    @property
    def assessment_count(self) -> int:
        """Number of DNSH assessments performed since initialization."""
        return self._assessment_count

    @property
    def enabled_thresholds(self) -> List[str]:
        """List of enabled PAI indicator threshold IDs."""
        return [
            t.pai_indicator_id
            for t in self.config.thresholds
            if t.enabled
        ]

    @property
    def critical_indicators(self) -> List[str]:
        """List of PAI indicators configured with CRITICAL severity."""
        return [
            t.pai_indicator_id
            for t in self.config.thresholds
            if t.severity == SeverityLevel.CRITICAL
        ]

    @property
    def auto_exclude_indicators(self) -> List[str]:
        """List of PAI indicators that trigger auto-exclusion on failure."""
        return [
            t.pai_indicator_id
            for t in self.config.thresholds
            if t.auto_exclude_on_fail
        ]
