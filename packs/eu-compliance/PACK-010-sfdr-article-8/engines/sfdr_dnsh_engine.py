# -*- coding: utf-8 -*-
"""
SFDRDNSHEngine - PACK-010 SFDR Article 8 Engine 3
====================================================

SFDR-specific Do No Significant Harm (DNSH) assessment engine.

Under SFDR, DNSH has a distinct meaning from the Taxonomy Regulation's DNSH.
In the SFDR context, DNSH means verifying that an investment does not cause
significant harm to any of the sustainability objectives by checking PAI
indicators against configurable severity thresholds.

SFDR DNSH is assessed at the investment level by screening each PAI indicator
against thresholds. If any indicator breaches its threshold, the investment
fails DNSH for that category. Portfolio-level DNSH compliance scores
aggregate the investment-level results.

Key Regulatory References:
    - Regulation (EU) 2019/2088 (SFDR) Article 2(17)
    - Delegated Regulation (EU) 2022/1288 (SFDR RTS) Article 12(1)
    - SFDR RTS Annex I, Table 1 (mandatory PAI indicators)

Zero-Hallucination:
    - All threshold comparisons use deterministic Python arithmetic
    - Configurable thresholds prevent hard-coded regulatory interpretation
    - SHA-256 provenance hashing on every result
    - No LLM involvement in any assessment path

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PAICategory(str, Enum):
    """PAI indicator groupings relevant to DNSH assessment."""

    CLIMATE_GHG = "climate_ghg"
    ENVIRONMENT = "environment"
    SOCIAL = "social"
    SOVEREIGN = "sovereign"
    REAL_ESTATE = "real_estate"

class DNSHStatus(str, Enum):
    """DNSH assessment outcome for a single check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"              # Below threshold but in warning range
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"

class ThresholdDirection(str, Enum):
    """Direction of threshold check."""

    MAX = "MAX"   # Fail if value > threshold (e.g., emissions)
    MIN = "MIN"   # Fail if value < threshold (e.g., board diversity)
    BOOLEAN_TRUE_FAILS = "BOOLEAN_TRUE_FAILS"    # Fail if True (e.g., violations)
    BOOLEAN_FALSE_FAILS = "BOOLEAN_FALSE_FAILS"  # Fail if False (e.g., lacks compliance)

class SeverityLevel(str, Enum):
    """Severity classification for DNSH failures."""

    CRITICAL = "CRITICAL"     # Absolute exclusion required
    HIGH = "HIGH"             # Strong adverse signal
    MEDIUM = "MEDIUM"         # Moderate concern
    LOW = "LOW"               # Minor concern
    INFORMATIONAL = "INFORMATIONAL"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PAIThreshold(BaseModel):
    """Configurable threshold for a single PAI indicator DNSH check.

    Defines the pass/fail boundary and warning zone for a PAI indicator
    when used as a DNSH criterion under SFDR.

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

class DNSHConfig(BaseModel):
    """Configuration for the SFDR DNSH Engine.

    Attributes:
        thresholds: List of PAI indicator thresholds for DNSH checks.
        require_all_environmental: Whether all env PAI must pass.
        require_all_social: Whether all social PAI must pass.
        exclusion_on_critical_fail: Auto-exclude on CRITICAL severity failures.
        data_coverage_minimum_pct: Minimum data coverage for valid assessment.
    """

    thresholds: List[PAIThreshold] = Field(
        default_factory=list,
        description="PAI indicator thresholds for DNSH screening",
    )
    require_all_environmental: bool = Field(
        default=True,
        description="Require pass on all environmental PAI (1-9) for DNSH pass",
    )
    require_all_social: bool = Field(
        default=True,
        description="Require pass on all social PAI (10-14) for DNSH pass",
    )
    exclusion_on_critical_fail: bool = Field(
        default=True,
        description="Automatically exclude investments with CRITICAL failures",
    )
    data_coverage_minimum_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum data coverage to consider DNSH result valid (%)",
    )

class PAIDNSHCheck(BaseModel):
    """Result of a single PAI indicator DNSH check for one investment."""

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
        None, description="Actual PAI indicator value for the investment",
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
        default="", description="Human-readable explanation of the result",
    )

class DNSHAssessment(BaseModel):
    """Complete SFDR DNSH assessment for a single investment.

    Attributes:
        investment_id: Unique investment identifier.
        investment_name: Investment name.
        overall_status: Overall DNSH outcome.
        environmental_status: Aggregate environmental DNSH status.
        social_status: Aggregate social DNSH status.
        checks: List of individual PAI checks performed.
        total_checks: Total number of checks.
        passed_checks: Number of passed checks.
        failed_checks: Number of failed checks.
        warning_checks: Number of warning checks.
        no_data_checks: Number of checks with insufficient data.
        critical_failures: List of CRITICAL severity failures.
        should_exclude: Whether the investment should be excluded.
        data_coverage_pct: Percentage of PAI checks with available data.
        provenance_hash: SHA-256 hash for audit trail.
        assessed_at: Assessment timestamp.
    """

    investment_id: str = Field(..., description="Investment identifier")
    investment_name: str = Field(..., description="Investment name")
    overall_status: DNSHStatus = Field(..., description="Overall DNSH outcome")
    environmental_status: DNSHStatus = Field(
        ..., description="Environmental DNSH status (PAI 1-9)",
    )
    social_status: DNSHStatus = Field(
        ..., description="Social DNSH status (PAI 10-14)",
    )
    checks: List[PAIDNSHCheck] = Field(
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
    should_exclude: bool = Field(
        default=False,
        description="Whether investment should be excluded from portfolio",
    )
    data_coverage_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Percentage of checks with available data",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash",
    )
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp",
    )

class InvestmentPAIData(BaseModel):
    """PAI indicator data for a single investment for DNSH screening.

    Maps PAI indicator IDs to their actual values. Values can be numeric
    (float) or boolean depending on the indicator.

    Attributes:
        investment_id: Unique investment identifier.
        investment_name: Investment name.
        pai_values: Dictionary mapping PAI indicator IDs to values.
    """

    investment_id: str = Field(
        ..., min_length=1, description="Unique investment identifier",
    )
    investment_name: str = Field(
        ..., min_length=1, description="Investment name",
    )
    pai_values: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="PAI indicator values (PAI_1 through PAI_18)",
    )
    pai_boolean_flags: Dict[str, Optional[bool]] = Field(
        default_factory=dict,
        description="Boolean PAI indicator flags (e.g., PAI_4, PAI_10)",
    )

class PortfolioDNSHResult(BaseModel):
    """Portfolio-level DNSH compliance aggregation.

    Attributes:
        portfolio_name: Name of the portfolio.
        total_investments: Total investments assessed.
        passing_investments: Investments that pass DNSH.
        failing_investments: Investments that fail DNSH.
        warning_investments: Investments in warning state.
        insufficient_data_investments: Investments lacking data.
        compliance_score_pct: Overall DNSH compliance percentage.
        exclusion_count: Number of investments flagged for exclusion.
        investment_assessments: Individual assessment results.
        category_summary: Summary by PAI category.
        most_common_failures: Most frequently failed PAI indicators.
        provenance_hash: SHA-256 hash.
        assessed_at: Assessment timestamp.
        processing_time_ms: Processing time in milliseconds.
    """

    portfolio_name: Optional[str] = Field(None, description="Portfolio name")
    total_investments: int = Field(..., ge=0)
    passing_investments: int = Field(..., ge=0)
    failing_investments: int = Field(..., ge=0)
    warning_investments: int = Field(..., ge=0)
    insufficient_data_investments: int = Field(..., ge=0)
    compliance_score_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="(passing / (total - insufficient)) * 100",
    )
    exclusion_count: int = Field(
        ..., ge=0, description="Investments flagged for exclusion",
    )
    investment_assessments: List[DNSHAssessment] = Field(
        default_factory=list,
    )
    category_summary: Dict[str, Dict[str, int]] = Field(
        default_factory=dict,
        description="Failure counts by PAI category",
    )
    most_common_failures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="PAI indicators sorted by failure frequency",
    )
    provenance_hash: str = Field(default="")
    assessed_at: datetime = Field(default_factory=utcnow)
    processing_time_ms: float = Field(default=0.0)
    engine_version: str = Field(default=_MODULE_VERSION)

class DNSHReportSection(BaseModel):
    """A section of the DNSH compliance report."""

    title: str = Field(..., description="Section title")
    category: PAICategory = Field(..., description="PAI category")
    indicators_checked: int = Field(..., ge=0)
    indicators_passed: int = Field(..., ge=0)
    indicators_failed: int = Field(..., ge=0)
    indicators_warning: int = Field(..., ge=0)
    indicators_no_data: int = Field(..., ge=0)
    details: List[Dict[str, Any]] = Field(default_factory=list)

# ---------------------------------------------------------------------------
# Default Threshold Configuration
# ---------------------------------------------------------------------------

def _build_default_thresholds() -> List[PAIThreshold]:
    """Build default PAI indicator thresholds for SFDR DNSH screening.

    These thresholds represent reasonable defaults based on market practice
    and regulatory guidance. Financial market participants should customize
    these based on their specific product characteristics and sustainability
    commitments.

    Returns:
        List of PAIThreshold objects for all 18 mandatory indicators.
    """
    return [
        # --- PAI 1-6: Climate/GHG ---
        PAIThreshold(
            pai_indicator_id="PAI_1",
            pai_name="GHG emissions",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=None,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq",
            severity=SeverityLevel.HIGH,
            description="No absolute threshold; assessed relative to sector peers",
            enabled=False,
        ),
        PAIThreshold(
            pai_indicator_id="PAI_2",
            pai_name="Carbon footprint",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=500.0,
            warning_value=350.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M",
            severity=SeverityLevel.HIGH,
            description="Carbon footprint should not exceed 500 tCO2eq per EUR M invested",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_3",
            pai_name="GHG intensity of investee companies",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=1000.0,
            warning_value=700.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M revenue",
            severity=SeverityLevel.MEDIUM,
            description="Company GHG intensity should not exceed 1000 tCO2eq per EUR M revenue",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_4",
            pai_name="Exposure to fossil fuel companies",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.HIGH,
            description="Investment is in a fossil fuel company",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_5",
            pai_name="Non-renewable energy share",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=80.0,
            warning_value=60.0,
            direction=ThresholdDirection.MAX,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Non-renewable energy share should not exceed 80%",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_6",
            pai_name="Energy intensity per high impact sector",
            category=PAICategory.CLIMATE_GHG,
            threshold_value=5.0,
            warning_value=3.0,
            direction=ThresholdDirection.MAX,
            unit="GWh / EUR M revenue",
            severity=SeverityLevel.MEDIUM,
            description="Energy intensity per sector should not exceed 5 GWh/EUR M",
        ),
        # --- PAI 7-9: Environment ---
        PAIThreshold(
            pai_indicator_id="PAI_7",
            pai_name="Biodiversity-sensitive areas",
            category=PAICategory.ENVIRONMENT,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.HIGH,
            description="Investment negatively affects biodiversity-sensitive areas",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_8",
            pai_name="Emissions to water",
            category=PAICategory.ENVIRONMENT,
            threshold_value=100.0,
            warning_value=50.0,
            direction=ThresholdDirection.MAX,
            unit="tonnes",
            severity=SeverityLevel.MEDIUM,
            description="Water pollutant emissions should not exceed 100 tonnes",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_9",
            pai_name="Hazardous and radioactive waste",
            category=PAICategory.ENVIRONMENT,
            threshold_value=500.0,
            warning_value=200.0,
            direction=ThresholdDirection.MAX,
            unit="tonnes",
            severity=SeverityLevel.MEDIUM,
            description="Hazardous/radioactive waste should not exceed 500 tonnes",
        ),
        # --- PAI 10-14: Social ---
        PAIThreshold(
            pai_indicator_id="PAI_10",
            pai_name="UNGC/OECD violations",
            category=PAICategory.SOCIAL,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Company has UNGC/OECD principles violations",
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
            threshold_value=25.0,
            warning_value=15.0,
            direction=ThresholdDirection.MAX,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Gender pay gap should not exceed 25%",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_13",
            pai_name="Board gender diversity",
            category=PAICategory.SOCIAL,
            threshold_value=20.0,
            warning_value=30.0,
            direction=ThresholdDirection.MIN,
            unit="%",
            severity=SeverityLevel.MEDIUM,
            description="Female board representation should be at least 20%",
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
        ),
        # --- PAI 15-16: Sovereign ---
        PAIThreshold(
            pai_indicator_id="PAI_15",
            pai_name="GHG intensity of investee countries",
            category=PAICategory.SOVEREIGN,
            threshold_value=800.0,
            warning_value=500.0,
            direction=ThresholdDirection.MAX,
            unit="tCO2eq / EUR M GDP",
            severity=SeverityLevel.MEDIUM,
            description="Country GHG intensity should not exceed 800 tCO2eq/EUR M GDP",
        ),
        PAIThreshold(
            pai_indicator_id="PAI_16",
            pai_name="Countries subject to social violations",
            category=PAICategory.SOVEREIGN,
            threshold_value=None,
            direction=ThresholdDirection.BOOLEAN_TRUE_FAILS,
            unit="%",
            severity=SeverityLevel.CRITICAL,
            description="Country is subject to social violations",
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
# Engine
# ---------------------------------------------------------------------------

class SFDRDNSHEngine:
    """SFDR-specific Do No Significant Harm Assessment Engine.

    Under SFDR, DNSH means checking whether an investment causes significant
    harm to any sustainability objective by screening PAI indicators against
    configurable severity thresholds. This is distinct from the Taxonomy
    Regulation's DNSH, which checks environmental objectives against
    Delegated Act criteria.

    This engine provides investment-level DNSH screening, portfolio-level
    DNSH compliance scoring, and structured DNSH reporting.

    Zero-Hallucination Guarantees:
        - All threshold comparisons use deterministic arithmetic
        - Configurable thresholds avoid hard-coded regulatory interpretation
        - SHA-256 provenance hashing on every result
        - No LLM involvement in any assessment path

    Attributes:
        config: DNSH configuration with PAI thresholds.
        _thresholds_by_id: Lookup map of thresholds by PAI indicator ID.
        _assessment_count: Running count of assessments performed.

    Example:
        >>> config = DNSHConfig(thresholds=_build_default_thresholds())
        >>> engine = SFDRDNSHEngine(config)
        >>> data = InvestmentPAIData(
        ...     investment_id="ISIN001",
        ...     investment_name="Example Corp",
        ...     pai_values={"PAI_2": 350.0, "PAI_5": 65.0},
        ...     pai_boolean_flags={"PAI_10": False, "PAI_14": False},
        ... )
        >>> result = engine.assess_dnsh(data)
        >>> assert result.overall_status == DNSHStatus.PASS
    """

    def __init__(self, config: Optional[DNSHConfig] = None) -> None:
        """Initialize the SFDR DNSH Engine.

        Args:
            config: DNSH configuration. If None, uses default thresholds.
        """
        if config is None:
            config = DNSHConfig(thresholds=_build_default_thresholds())

        self.config = config
        self._thresholds_by_id: Dict[str, PAIThreshold] = {
            t.pai_indicator_id: t for t in config.thresholds
        }
        self._assessment_count: int = 0

        if not config.thresholds:
            config.thresholds = _build_default_thresholds()
            self._thresholds_by_id = {
                t.pai_indicator_id: t for t in config.thresholds
            }

        logger.info(
            "SFDRDNSHEngine initialized (v%s, %d thresholds configured, "
            "%d enabled)",
            _MODULE_VERSION,
            len(config.thresholds),
            sum(1 for t in config.thresholds if t.enabled),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_dnsh(
        self,
        investment_data: InvestmentPAIData,
    ) -> DNSHAssessment:
        """Assess SFDR DNSH for a single investment.

        Screens each enabled PAI indicator threshold against the investment's
        data. Aggregates results into environmental and social DNSH statuses,
        and determines the overall DNSH outcome.

        Args:
            investment_data: PAI indicator data for the investment.

        Returns:
            DNSHAssessment with per-indicator and overall results.

        Raises:
            ValueError: If investment_id is empty.
        """
        start = utcnow()
        self._assessment_count += 1

        if not investment_data.investment_id:
            raise ValueError("investment_id is required")

        logger.debug(
            "Assessing DNSH for investment %s (%s)",
            investment_data.investment_id,
            investment_data.investment_name,
        )

        checks: List[PAIDNSHCheck] = []

        for threshold in self.config.thresholds:
            if not threshold.enabled:
                continue

            check = self._evaluate_threshold(
                threshold, investment_data
            )
            checks.append(check)

        # Aggregate results
        env_checks = [
            c for c in checks
            if c.category in (PAICategory.CLIMATE_GHG, PAICategory.ENVIRONMENT)
        ]
        social_checks = [
            c for c in checks
            if c.category == PAICategory.SOCIAL
        ]

        env_status = self._aggregate_status(
            env_checks, self.config.require_all_environmental
        )
        social_status = self._aggregate_status(
            social_checks, self.config.require_all_social
        )

        overall_status = self._determine_overall_status(env_status, social_status, checks)

        # Count outcomes
        passed = sum(1 for c in checks if c.status == DNSHStatus.PASS)
        failed = sum(1 for c in checks if c.status == DNSHStatus.FAIL)
        warning = sum(1 for c in checks if c.status == DNSHStatus.WARNING)
        no_data = sum(1 for c in checks if c.status == DNSHStatus.INSUFFICIENT_DATA)
        total = len(checks)

        # Critical failures
        critical = [
            c.pai_indicator_id for c in checks
            if c.status == DNSHStatus.FAIL and c.severity == SeverityLevel.CRITICAL
        ]

        # Exclusion decision
        should_exclude = (
            self.config.exclusion_on_critical_fail and len(critical) > 0
        )

        # Data coverage
        data_coverage = (
            ((total - no_data) / total * 100.0) if total > 0 else 0.0
        )

        assessment = DNSHAssessment(
            investment_id=investment_data.investment_id,
            investment_name=investment_data.investment_name,
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
            should_exclude=should_exclude,
            data_coverage_pct=_round_val(data_coverage, 2),
            assessed_at=start,
        )

        assessment.provenance_hash = _compute_hash({
            "investment_id": investment_data.investment_id,
            "overall_status": overall_status.value,
            "passed": passed,
            "failed": failed,
            "critical": critical,
        })

        elapsed_ms = (utcnow() - start).total_seconds() * 1000
        logger.info(
            "DNSH assessment for %s: overall=%s, passed=%d, failed=%d, "
            "warning=%d, critical=%d, time=%.1fms",
            investment_data.investment_id, overall_status.value,
            passed, failed, warning, len(critical), elapsed_ms,
        )

        return assessment

    def assess_portfolio_dnsh(
        self,
        investments: List[InvestmentPAIData],
        portfolio_name: Optional[str] = None,
    ) -> PortfolioDNSHResult:
        """Assess SFDR DNSH for an entire portfolio.

        Performs individual DNSH assessment for each investment and
        aggregates into a portfolio-level compliance score.

        Args:
            investments: List of investment PAI data.
            portfolio_name: Optional portfolio name.

        Returns:
            PortfolioDNSHResult with portfolio-level compliance metrics.

        Raises:
            ValueError: If investments list is empty.
        """
        start = utcnow()

        if not investments:
            raise ValueError("Investments list cannot be empty")

        logger.info(
            "Assessing portfolio DNSH for %d investments", len(investments)
        )

        assessments: List[DNSHAssessment] = []
        for inv in investments:
            try:
                assessment = self.assess_dnsh(inv)
                assessments.append(assessment)
            except Exception as exc:
                logger.error(
                    "DNSH assessment failed for %s: %s",
                    inv.investment_id, exc,
                )
                raise

        # Count statuses
        passing = sum(
            1 for a in assessments if a.overall_status == DNSHStatus.PASS
        )
        failing = sum(
            1 for a in assessments if a.overall_status == DNSHStatus.FAIL
        )
        warning = sum(
            1 for a in assessments if a.overall_status == DNSHStatus.WARNING
        )
        no_data = sum(
            1 for a in assessments
            if a.overall_status == DNSHStatus.INSUFFICIENT_DATA
        )
        exclusions = sum(1 for a in assessments if a.should_exclude)

        # Compliance score: passing / (total - no_data)
        assessable = len(assessments) - no_data
        compliance_score = (
            (passing / assessable * 100.0) if assessable > 0 else 0.0
        )

        # Category summary
        category_summary = self._build_category_summary(assessments)

        # Most common failures
        failure_freq = self._compute_failure_frequency(assessments)

        elapsed_ms = (utcnow() - start).total_seconds() * 1000

        result = PortfolioDNSHResult(
            portfolio_name=portfolio_name,
            total_investments=len(assessments),
            passing_investments=passing,
            failing_investments=failing,
            warning_investments=warning,
            insufficient_data_investments=no_data,
            compliance_score_pct=_round_val(compliance_score, 2),
            exclusion_count=exclusions,
            investment_assessments=assessments,
            category_summary=category_summary,
            most_common_failures=failure_freq,
            processing_time_ms=round(elapsed_ms, 2),
        )

        result.provenance_hash = _compute_hash({
            "portfolio_name": portfolio_name,
            "total": len(assessments),
            "passing": passing,
            "failing": failing,
            "compliance_score": compliance_score,
        })

        logger.info(
            "Portfolio DNSH complete: %d investments, compliance=%.1f%%, "
            "exclusions=%d, time=%.1fms",
            len(assessments), compliance_score, exclusions, elapsed_ms,
        )

        return result

    def get_dnsh_criteria(self) -> List[Dict[str, Any]]:
        """Return all configured DNSH criteria (PAI thresholds).

        Returns:
            List of threshold configurations as dictionaries.
        """
        return [
            t.model_dump() for t in self.config.thresholds
        ]

    def generate_dnsh_report(
        self,
        assessment: DNSHAssessment,
    ) -> List[DNSHReportSection]:
        """Generate a structured DNSH report from an assessment.

        Organizes the assessment results into report sections by PAI
        category, suitable for regulatory disclosure.

        Args:
            assessment: A completed DNSH assessment.

        Returns:
            List of DNSHReportSection objects.
        """
        sections: List[DNSHReportSection] = []

        category_config = [
            (PAICategory.CLIMATE_GHG, "Climate and Greenhouse Gas Indicators (PAI 1-6)"),
            (PAICategory.ENVIRONMENT, "Environmental Indicators (PAI 7-9)"),
            (PAICategory.SOCIAL, "Social and Governance Indicators (PAI 10-14)"),
            (PAICategory.SOVEREIGN, "Sovereign Indicators (PAI 15-16)"),
            (PAICategory.REAL_ESTATE, "Real Estate Indicators (PAI 17-18)"),
        ]

        for category, title in category_config:
            cat_checks = [
                c for c in assessment.checks if c.category == category
            ]

            if not cat_checks:
                continue

            passed = sum(1 for c in cat_checks if c.status == DNSHStatus.PASS)
            failed = sum(1 for c in cat_checks if c.status == DNSHStatus.FAIL)
            warning = sum(1 for c in cat_checks if c.status == DNSHStatus.WARNING)
            no_data = sum(
                1 for c in cat_checks
                if c.status == DNSHStatus.INSUFFICIENT_DATA
            )

            details = [
                {
                    "pai_indicator": c.pai_indicator_id,
                    "name": c.pai_name,
                    "status": c.status.value,
                    "actual_value": c.actual_value,
                    "threshold": c.threshold_value,
                    "severity": c.severity.value,
                    "explanation": c.explanation,
                }
                for c in cat_checks
            ]

            sections.append(DNSHReportSection(
                title=title,
                category=category,
                indicators_checked=len(cat_checks),
                indicators_passed=passed,
                indicators_failed=failed,
                indicators_warning=warning,
                indicators_no_data=no_data,
                details=details,
            ))

        return sections

    # ------------------------------------------------------------------
    # Private: Threshold Evaluation
    # ------------------------------------------------------------------

    def _evaluate_threshold(
        self,
        threshold: PAIThreshold,
        investment_data: InvestmentPAIData,
    ) -> PAIDNSHCheck:
        """Evaluate a single PAI threshold against investment data.

        Handles both numeric thresholds (MAX/MIN) and boolean flags
        (BOOLEAN_TRUE_FAILS / BOOLEAN_FALSE_FAILS).

        Args:
            threshold: The PAI threshold configuration.
            investment_data: The investment's PAI data.

        Returns:
            PAIDNSHCheck with the evaluation result.
        """
        pai_id = threshold.pai_indicator_id

        # Handle boolean indicators
        if threshold.direction in (
            ThresholdDirection.BOOLEAN_TRUE_FAILS,
            ThresholdDirection.BOOLEAN_FALSE_FAILS,
        ):
            return self._evaluate_boolean_threshold(threshold, investment_data)

        # Handle numeric indicators
        actual_value = investment_data.pai_values.get(pai_id)

        if actual_value is None:
            return PAIDNSHCheck(
                pai_indicator_id=pai_id,
                pai_name=threshold.pai_name,
                category=threshold.category,
                status=DNSHStatus.INSUFFICIENT_DATA,
                direction=threshold.direction,
                severity=threshold.severity,
                explanation=f"No data available for {pai_id}",
            )

        if threshold.threshold_value is None:
            # Threshold not defined; report as not applicable
            return PAIDNSHCheck(
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
    ) -> PAIDNSHCheck:
        """Evaluate a numeric PAI threshold.

        Args:
            threshold: Threshold configuration.
            actual_value: Actual indicator value.

        Returns:
            PAIDNSHCheck with PASS, FAIL, or WARNING status.
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
            explanation = (
                f"{pai_id} value {actual_value} breaches "
                f"{'maximum' if threshold.direction == ThresholdDirection.MAX else 'minimum'} "
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

        return PAIDNSHCheck(
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
        )

    def _evaluate_boolean_threshold(
        self,
        threshold: PAIThreshold,
        investment_data: InvestmentPAIData,
    ) -> PAIDNSHCheck:
        """Evaluate a boolean PAI threshold.

        Args:
            threshold: Boolean threshold configuration.
            investment_data: Investment PAI data.

        Returns:
            PAIDNSHCheck with PASS, FAIL, or INSUFFICIENT_DATA.
        """
        pai_id = threshold.pai_indicator_id
        flag_value = investment_data.pai_boolean_flags.get(pai_id)

        if flag_value is None:
            return PAIDNSHCheck(
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
            explanation = (
                f"{pai_id}: {threshold.description}"
            )
        else:
            status = DNSHStatus.PASS
            explanation = (
                f"{pai_id}: No significant harm detected "
                f"(flag={flag_value})"
            )

        return PAIDNSHCheck(
            pai_indicator_id=pai_id,
            pai_name=threshold.pai_name,
            category=threshold.category,
            status=status,
            actual_value=1.0 if flag_value else 0.0,
            threshold_value=None,
            direction=threshold.direction,
            severity=threshold.severity,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Private: Status Aggregation
    # ------------------------------------------------------------------

    def _aggregate_status(
        self,
        checks: List[PAIDNSHCheck],
        require_all: bool,
    ) -> DNSHStatus:
        """Aggregate individual check statuses into a category status.

        Args:
            checks: List of PAI checks for the category.
            require_all: If True, all checks must pass for category to pass.

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
        all_checks: List[PAIDNSHCheck],
    ) -> DNSHStatus:
        """Determine overall DNSH status from category statuses.

        Args:
            env_status: Environmental category status.
            social_status: Social category status.
            all_checks: All individual checks (including sovereign/RE).

        Returns:
            Overall DNSHStatus.
        """
        # Check sovereign and real estate too
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
    # Private: Portfolio Aggregation Helpers
    # ------------------------------------------------------------------

    def _build_category_summary(
        self,
        assessments: List[DNSHAssessment],
    ) -> Dict[str, Dict[str, int]]:
        """Build failure count summary by PAI category.

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
        assessments: List[DNSHAssessment],
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
            failure_counts.items(), key=lambda x: x[1], reverse=True
        )

        return [
            {
                "pai_indicator_id": pai_id,
                "pai_name": indicator_names.get(pai_id, ""),
                "failure_count": count,
                "failure_rate_pct": _round_val(
                    count / len(assessments) * 100.0, 2
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
