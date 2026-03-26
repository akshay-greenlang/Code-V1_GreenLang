# -*- coding: utf-8 -*-
"""
Intensity Calculation Workflow
====================================

4-phase workflow for multi-scope intensity metric computation within
PACK-046 Intensity Metrics Pack.

Phases:
    1. DataIngestion              -- Ingest emissions data from MRV agents
                                     (PACK-041/042/043 outputs) and denominator
                                     data from DenominatorSetupWorkflow or
                                     DATA agents, validate completeness.
    2. ScopeConfiguration         -- Configure scope inclusion rules per
                                     framework requirements (e.g. Scope 1+2
                                     for CDP, Scope 1+2+3 for ESRS E1, Scope
                                     1+2 location-based and market-based).
    3. IntensityCalculation       -- Run IntensityCalculationEngine for every
                                     scope/denominator/period combination,
                                     producing tCO2e per denominator unit with
                                     full provenance.
    4. QualityAssurance           -- Quality checks including data completeness
                                     validation, outlier detection (>3 sigma),
                                     consistency across scopes, year-over-year
                                     reasonableness, and cross-framework alignment.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    GRI 305-4 (2016) - GHG emissions intensity
    ESRS E1-6 - Climate change intensity metrics
    CDP C6.10 (2026) - Emissions intensities
    ISO 14064-1:2018 Clause 5 - Quantification per unit
    SEC Climate Disclosure Rules - Intensity metrics

Schedule: Annually after emissions calculation, or quarterly for interim
Estimated duration: 1-2 weeks

Author: GreenLang Team
Version: 46.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.utcnow().isoformat() + "Z"


def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class CalcPhase(str, Enum):
    """Intensity calculation workflow phases."""

    DATA_INGESTION = "data_ingestion"
    SCOPE_CONFIGURATION = "scope_configuration"
    INTENSITY_CALCULATION = "intensity_calculation"
    QUALITY_ASSURANCE = "quality_assurance"


class ScopeInclusion(str, Enum):
    """Scope inclusion rules for intensity calculation."""

    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_1_2_LOCATION = "scope_1_2_location"
    SCOPE_1_2_MARKET = "scope_1_2_market"
    SCOPE_1_2_3 = "scope_1_2_3"
    SCOPE_3_ONLY = "scope_3_only"
    ALL_SCOPES = "all_scopes"


class IntensityUnit(str, Enum):
    """Unit of intensity metric output."""

    TCO2E_PER_USD_MILLION = "tCO2e/USD_million"
    TCO2E_PER_EUR_MILLION = "tCO2e/EUR_million"
    TCO2E_PER_FTE = "tCO2e/FTE"
    TCO2E_PER_SQM = "tCO2e/sqm"
    TCO2E_PER_TONNE = "tCO2e/tonne"
    TCO2E_PER_MWH = "tCO2e/MWh"
    TCO2E_PER_GWH = "tCO2e/GWh"
    TCO2E_PER_PKM = "tCO2e/passenger_km"
    TCO2E_PER_TKM = "tCO2e/tonne_km"
    TCO2E_PER_UNIT = "tCO2e/unit"
    CUSTOM = "custom"


class QualityCheckType(str, Enum):
    """Type of quality assurance check."""

    COMPLETENESS = "completeness"
    OUTLIER_DETECTION = "outlier_detection"
    CONSISTENCY = "consistency"
    REASONABLENESS = "reasonableness"
    CROSS_FRAMEWORK = "cross_framework"


class QualityOutcome(str, Enum):
    """Outcome of a quality check."""

    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# FRAMEWORK SCOPE RULES (Zero-Hallucination Reference Data)
# =============================================================================

FRAMEWORK_SCOPE_RULES: Dict[str, List[str]] = {
    "esrs_e1": ["scope_1_2_location", "scope_1_2_market", "scope_1_2_3"],
    "cdp_c6": ["scope_1_2_location", "scope_1_2_market"],
    "sec_climate": ["scope_1_2_location", "scope_1_2_3"],
    "sbti_sda": ["scope_1_2_location", "scope_1_2_3"],
    "iso_14064": ["scope_1_2_location"],
    "tcfd": ["scope_1_2_location", "scope_1_2_3"],
    "gri_305_4": ["scope_1_2_location", "scope_1_2_3"],
    "ifrs_s2": ["scope_1_2_location", "scope_1_2_3"],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class EmissionsDataSet(BaseModel):
    """Emissions data for intensity calculation."""

    period: str = Field(..., description="Reporting period, e.g. 2024")
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_location_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_market_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_categories: Dict[str, float] = Field(default_factory=dict)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    source: str = Field(default="mrv_agents")


class DenominatorDataSet(BaseModel):
    """Denominator data for intensity calculation."""

    denominator_type: str = Field(..., description="e.g. revenue, fte, floor_area")
    unit: str = Field(default="")
    period: str = Field(..., description="Reporting period")
    value: float = Field(..., gt=0.0, description="Denominator value (must be positive)")
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    source: str = Field(default="denominator_setup")


class ScopeRule(BaseModel):
    """Configured scope inclusion rule for a framework."""

    framework: str = Field(...)
    scope_inclusion: ScopeInclusion = Field(...)
    description: str = Field(default="")


class IntensityMetric(BaseModel):
    """A computed intensity metric."""

    metric_id: str = Field(default_factory=lambda: f"im-{_new_uuid()[:8]}")
    period: str = Field(...)
    scope_inclusion: ScopeInclusion = Field(...)
    denominator_type: str = Field(...)
    numerator_tco2e: float = Field(default=0.0, ge=0.0)
    denominator_value: float = Field(default=0.0, gt=0.0)
    intensity_value: float = Field(default=0.0, ge=0.0)
    intensity_unit: IntensityUnit = Field(default=IntensityUnit.CUSTOM)
    frameworks: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class QualityCheckResult(BaseModel):
    """Result of a single quality assurance check."""

    check_type: QualityCheckType = Field(...)
    outcome: QualityOutcome = Field(...)
    metric_id: str = Field(default="")
    details: str = Field(default="")
    threshold: float = Field(default=0.0)
    actual_value: float = Field(default=0.0)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class IntensityCalcInput(BaseModel):
    """Input data model for IntensityCalculationWorkflow."""

    organization_id: str = Field(..., min_length=1)
    emissions_data: List[EmissionsDataSet] = Field(
        ..., min_length=1, description="Emissions data per period",
    )
    denominator_data: List[DenominatorDataSet] = Field(
        ..., min_length=1, description="Denominator data per type/period",
    )
    applicable_frameworks: List[str] = Field(
        default_factory=lambda: ["esrs_e1", "cdp_c6", "gri_305_4"],
    )
    custom_scope_rules: List[ScopeRule] = Field(
        default_factory=list,
        description="Override default framework scope rules",
    )
    outlier_sigma_threshold: float = Field(
        default=3.0, ge=1.0, le=5.0,
        description="Standard deviations for outlier detection",
    )
    yoy_change_threshold_pct: float = Field(
        default=30.0, ge=5.0, le=100.0,
        description="Year-over-year change threshold for reasonableness",
    )
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)


class IntensityCalcResult(BaseModel):
    """Complete result from intensity calculation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="intensity_calculation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    intensity_metrics: List[IntensityMetric] = Field(default_factory=list)
    scope_rules: List[ScopeRule] = Field(default_factory=list)
    quality_checks: List[QualityCheckResult] = Field(default_factory=list)
    quality_pass_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    metrics_count: int = Field(default=0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class IntensityCalculationWorkflow:
    """
    4-phase workflow for multi-scope intensity metric computation.

    Ingests emissions and denominator data, configures scope inclusion rules
    per framework, calculates intensity metrics for all combinations, and
    runs quality assurance checks.

    Zero-hallucination: intensity = emissions / denominator using deterministic
    division only; no LLM calls in calculation path; SHA-256 provenance on
    every metric.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _emissions_by_period: Indexed emissions data.
        _denominators_by_key: Indexed denominator data.
        _scope_rules: Configured scope rules.
        _metrics: Computed intensity metrics.
        _quality_checks: Quality assurance results.

    Example:
        >>> wf = IntensityCalculationWorkflow()
        >>> emissions = [EmissionsDataSet(period="2024", scope1_tco2e=5000)]
        >>> denominators = [DenominatorDataSet(
        ...     denominator_type="revenue", period="2024", value=100.0)]
        >>> inp = IntensityCalcInput(
        ...     organization_id="org-001",
        ...     emissions_data=emissions,
        ...     denominator_data=denominators,
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[CalcPhase] = [
        CalcPhase.DATA_INGESTION,
        CalcPhase.SCOPE_CONFIGURATION,
        CalcPhase.INTENSITY_CALCULATION,
        CalcPhase.QUALITY_ASSURANCE,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize IntensityCalculationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._emissions_by_period: Dict[str, EmissionsDataSet] = {}
        self._denominators_by_key: Dict[str, DenominatorDataSet] = {}
        self._scope_rules: List[ScopeRule] = []
        self._metrics: List[IntensityMetric] = []
        self._quality_checks: List[QualityCheckResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: IntensityCalcInput) -> IntensityCalcResult:
        """
        Execute the 4-phase intensity calculation workflow.

        Args:
            input_data: Emissions data, denominator data, and framework config.

        Returns:
            IntensityCalcResult with computed metrics and quality checks.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting intensity calculation %s org=%s periods=%d denominators=%d",
            self.workflow_id, input_data.organization_id,
            len(input_data.emissions_data), len(input_data.denominator_data),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_data_ingestion,
            self._phase_2_scope_configuration,
            self._phase_3_intensity_calculation,
            self._phase_4_quality_assurance,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Intensity calculation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Calculate quality pass rate
        total_checks = len(self._quality_checks)
        passed = sum(
            1 for qc in self._quality_checks
            if qc.outcome in (QualityOutcome.PASS, QualityOutcome.NOT_APPLICABLE)
        )
        pass_rate = (passed / max(total_checks, 1)) * 100.0

        result = IntensityCalcResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            intensity_metrics=self._metrics,
            scope_rules=self._scope_rules,
            quality_checks=self._quality_checks,
            quality_pass_rate=round(pass_rate, 2),
            metrics_count=len(self._metrics),
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Intensity calculation %s completed in %.2fs status=%s metrics=%d pass_rate=%.1f%%",
            self.workflow_id, elapsed, overall_status.value,
            len(self._metrics), pass_rate,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Ingestion
    # -------------------------------------------------------------------------

    async def _phase_1_data_ingestion(
        self, input_data: IntensityCalcInput,
    ) -> PhaseResult:
        """Ingest and index emissions and denominator data."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Index emissions by period
        self._emissions_by_period = {}
        for ed in input_data.emissions_data:
            if ed.period in self._emissions_by_period:
                warnings.append(f"Duplicate emissions data for period {ed.period}")
            self._emissions_by_period[ed.period] = ed

        # Index denominators by (type, period) key
        self._denominators_by_key = {}
        for dd in input_data.denominator_data:
            key = f"{dd.denominator_type}|{dd.period}"
            if key in self._denominators_by_key:
                warnings.append(f"Duplicate denominator data for {key}")
            self._denominators_by_key[key] = dd

        # Validate alignment
        emission_periods = set(self._emissions_by_period.keys())
        denom_periods = set(dd.period for dd in input_data.denominator_data)
        missing_in_denom = emission_periods - denom_periods
        missing_in_emissions = denom_periods - emission_periods

        if missing_in_denom:
            warnings.append(f"Periods with emissions but no denominators: {missing_in_denom}")
        if missing_in_emissions:
            warnings.append(f"Periods with denominators but no emissions: {missing_in_emissions}")

        # Determine unique denominator types
        denom_types = sorted(set(dd.denominator_type for dd in input_data.denominator_data))
        aligned_periods = sorted(emission_periods & denom_periods)

        outputs["emission_periods"] = sorted(emission_periods)
        outputs["denominator_types"] = denom_types
        outputs["aligned_periods"] = aligned_periods
        outputs["emissions_ingested"] = len(input_data.emissions_data)
        outputs["denominators_ingested"] = len(input_data.denominator_data)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 DataIngestion: %d emissions, %d denominators, %d aligned periods",
            len(input_data.emissions_data), len(input_data.denominator_data),
            len(aligned_periods),
        )
        return PhaseResult(
            phase_name="data_ingestion", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Scope Configuration
    # -------------------------------------------------------------------------

    async def _phase_2_scope_configuration(
        self, input_data: IntensityCalcInput,
    ) -> PhaseResult:
        """Configure scope inclusion rules per framework."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._scope_rules = []

        # Apply custom rules first
        custom_frameworks = set()
        for rule in input_data.custom_scope_rules:
            self._scope_rules.append(rule)
            custom_frameworks.add(rule.framework)

        # Apply default rules for remaining frameworks
        for fw in input_data.applicable_frameworks:
            if fw in custom_frameworks:
                continue
            default_scopes = FRAMEWORK_SCOPE_RULES.get(fw, ["scope_1_2_location"])
            for scope_str in default_scopes:
                try:
                    scope_incl = ScopeInclusion(scope_str)
                except ValueError:
                    warnings.append(f"Unknown scope inclusion {scope_str} for {fw}")
                    continue
                self._scope_rules.append(ScopeRule(
                    framework=fw,
                    scope_inclusion=scope_incl,
                    description=f"Default rule for {fw}: {scope_str}",
                ))

        outputs["rules_configured"] = len(self._scope_rules)
        outputs["frameworks_covered"] = sorted(set(r.framework for r in self._scope_rules))
        outputs["scope_inclusions"] = sorted(set(r.scope_inclusion.value for r in self._scope_rules))
        outputs["custom_overrides"] = len(input_data.custom_scope_rules)

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 ScopeConfiguration: %d rules for %d frameworks",
            len(self._scope_rules), len(outputs["frameworks_covered"]),
        )
        return PhaseResult(
            phase_name="scope_configuration", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Intensity Calculation
    # -------------------------------------------------------------------------

    async def _phase_3_intensity_calculation(
        self, input_data: IntensityCalcInput,
    ) -> PhaseResult:
        """Calculate intensity metrics for all scope/denominator/period combos."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._metrics = []
        denom_types = sorted(set(dd.denominator_type for dd in input_data.denominator_data))

        # Get unique scope inclusions from rules
        scope_inclusions = sorted(set(r.scope_inclusion for r in self._scope_rules), key=lambda s: s.value)

        for period, emissions in sorted(self._emissions_by_period.items()):
            for dtype in denom_types:
                key = f"{dtype}|{period}"
                denom_data = self._denominators_by_key.get(key)
                if not denom_data:
                    warnings.append(f"No denominator {dtype} for period {period}")
                    continue

                for scope_incl in scope_inclusions:
                    # Calculate numerator based on scope inclusion
                    numerator = self._calculate_numerator(emissions, scope_incl)

                    # Zero-hallucination: deterministic division
                    intensity = numerator / denom_data.value

                    # Determine applicable frameworks for this scope rule
                    frameworks = [
                        r.framework for r in self._scope_rules
                        if r.scope_inclusion == scope_incl
                    ]

                    # Determine intensity unit
                    intensity_unit = self._map_intensity_unit(denom_data.unit)

                    metric_data = {
                        "period": period,
                        "scope": scope_incl.value,
                        "denominator": dtype,
                        "numerator": round(numerator, 6),
                        "denominator_value": denom_data.value,
                        "intensity": round(intensity, 6),
                    }

                    self._metrics.append(IntensityMetric(
                        period=period,
                        scope_inclusion=scope_incl,
                        denominator_type=dtype,
                        numerator_tco2e=round(numerator, 6),
                        denominator_value=denom_data.value,
                        intensity_value=round(intensity, 6),
                        intensity_unit=intensity_unit,
                        frameworks=frameworks,
                        provenance_hash=_compute_hash(metric_data),
                    ))

        outputs["metrics_calculated"] = len(self._metrics)
        outputs["periods_covered"] = sorted(set(m.period for m in self._metrics))
        outputs["scope_combinations"] = sorted(set(m.scope_inclusion.value for m in self._metrics))
        outputs["denominator_types"] = sorted(set(m.denominator_type for m in self._metrics))

        if self._metrics:
            intensities = [m.intensity_value for m in self._metrics]
            outputs["intensity_range"] = {
                "min": round(min(intensities), 6),
                "max": round(max(intensities), 6),
                "mean": round(sum(intensities) / len(intensities), 6),
            }

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 IntensityCalculation: %d metrics computed",
            len(self._metrics),
        )
        return PhaseResult(
            phase_name="intensity_calculation", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quality Assurance
    # -------------------------------------------------------------------------

    async def _phase_4_quality_assurance(
        self, input_data: IntensityCalcInput,
    ) -> PhaseResult:
        """Run quality checks on computed intensity metrics."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._quality_checks = []

        # Check 1: Completeness - metrics exist for all expected combos
        self._check_completeness(input_data)

        # Check 2: Outlier detection using z-score
        self._check_outliers(input_data.outlier_sigma_threshold)

        # Check 3: Consistency across scope rules for same denominator/period
        self._check_consistency()

        # Check 4: Year-over-year reasonableness
        self._check_reasonableness(input_data.yoy_change_threshold_pct)

        # Check 5: Cross-framework alignment
        self._check_cross_framework()

        # Summarise
        pass_count = sum(1 for qc in self._quality_checks if qc.outcome == QualityOutcome.PASS)
        fail_count = sum(1 for qc in self._quality_checks if qc.outcome == QualityOutcome.FAIL)
        warn_count = sum(1 for qc in self._quality_checks if qc.outcome == QualityOutcome.WARNING)

        outputs["total_checks"] = len(self._quality_checks)
        outputs["passed"] = pass_count
        outputs["failed"] = fail_count
        outputs["warnings"] = warn_count
        outputs["pass_rate"] = round(
            (pass_count / max(len(self._quality_checks), 1)) * 100.0, 2,
        )

        if fail_count > 0:
            warnings.append(f"{fail_count} quality check(s) failed")

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 QualityAssurance: %d checks, %d pass, %d fail, %d warn",
            len(self._quality_checks), pass_count, fail_count, warn_count,
        )
        return PhaseResult(
            phase_name="quality_assurance", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Quality Check Helpers
    # -------------------------------------------------------------------------

    def _check_completeness(self, input_data: IntensityCalcInput) -> None:
        """Check that metrics exist for all expected combinations."""
        periods = sorted(set(e.period for e in input_data.emissions_data))
        denom_types = sorted(set(d.denominator_type for d in input_data.denominator_data))
        scope_inclusions = sorted(
            set(r.scope_inclusion for r in self._scope_rules), key=lambda s: s.value,
        )

        expected = len(periods) * len(denom_types) * len(scope_inclusions)
        actual = len(self._metrics)

        self._quality_checks.append(QualityCheckResult(
            check_type=QualityCheckType.COMPLETENESS,
            outcome=QualityOutcome.PASS if actual >= expected else QualityOutcome.WARNING,
            details=f"Expected {expected} metrics, got {actual}",
            threshold=float(expected),
            actual_value=float(actual),
        ))

    def _check_outliers(self, sigma_threshold: float) -> None:
        """Detect outlier intensity values using z-score."""
        if len(self._metrics) < 3:
            self._quality_checks.append(QualityCheckResult(
                check_type=QualityCheckType.OUTLIER_DETECTION,
                outcome=QualityOutcome.NOT_APPLICABLE,
                details="Insufficient data points for outlier detection",
            ))
            return

        # Group by (scope, denominator) for z-score calculation
        groups: Dict[str, List[IntensityMetric]] = {}
        for m in self._metrics:
            gkey = f"{m.scope_inclusion.value}|{m.denominator_type}"
            groups.setdefault(gkey, []).append(m)

        for gkey, group_metrics in groups.items():
            values = [m.intensity_value for m in group_metrics]
            if len(values) < 2:
                continue

            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance) if variance > 0 else 0.0

            if std_dev == 0:
                continue

            for m in group_metrics:
                z_score = abs(m.intensity_value - mean_val) / std_dev
                if z_score > sigma_threshold:
                    self._quality_checks.append(QualityCheckResult(
                        check_type=QualityCheckType.OUTLIER_DETECTION,
                        outcome=QualityOutcome.WARNING,
                        metric_id=m.metric_id,
                        details=(
                            f"Outlier detected: z-score={z_score:.2f} > {sigma_threshold} "
                            f"for {gkey} period {m.period}"
                        ),
                        threshold=sigma_threshold,
                        actual_value=z_score,
                    ))
                else:
                    self._quality_checks.append(QualityCheckResult(
                        check_type=QualityCheckType.OUTLIER_DETECTION,
                        outcome=QualityOutcome.PASS,
                        metric_id=m.metric_id,
                        details=f"Within bounds: z-score={z_score:.2f}",
                        threshold=sigma_threshold,
                        actual_value=z_score,
                    ))

    def _check_consistency(self) -> None:
        """Check scope hierarchy consistency (S1 < S1+S2 < S1+S2+S3)."""
        # Group by (period, denominator)
        groups: Dict[str, Dict[str, float]] = {}
        for m in self._metrics:
            gkey = f"{m.period}|{m.denominator_type}"
            groups.setdefault(gkey, {})[m.scope_inclusion.value] = m.intensity_value

        for gkey, scope_vals in groups.items():
            s1 = scope_vals.get("scope_1", 0)
            s12_loc = scope_vals.get("scope_1_2_location", 0)
            s123 = scope_vals.get("scope_1_2_3", 0)

            if s1 > 0 and s12_loc > 0 and s1 > s12_loc:
                self._quality_checks.append(QualityCheckResult(
                    check_type=QualityCheckType.CONSISTENCY,
                    outcome=QualityOutcome.FAIL,
                    details=f"S1 intensity > S1+S2 intensity for {gkey}",
                ))
            elif s12_loc > 0 and s123 > 0 and s12_loc > s123:
                self._quality_checks.append(QualityCheckResult(
                    check_type=QualityCheckType.CONSISTENCY,
                    outcome=QualityOutcome.FAIL,
                    details=f"S1+S2 intensity > S1+S2+S3 intensity for {gkey}",
                ))
            else:
                self._quality_checks.append(QualityCheckResult(
                    check_type=QualityCheckType.CONSISTENCY,
                    outcome=QualityOutcome.PASS,
                    details=f"Scope hierarchy consistent for {gkey}",
                ))

    def _check_reasonableness(self, yoy_threshold_pct: float) -> None:
        """Check year-over-year intensity changes for reasonableness."""
        # Group by (scope, denominator), sorted by period
        groups: Dict[str, List[IntensityMetric]] = {}
        for m in self._metrics:
            gkey = f"{m.scope_inclusion.value}|{m.denominator_type}"
            groups.setdefault(gkey, []).append(m)

        for gkey, group_metrics in groups.items():
            sorted_metrics = sorted(group_metrics, key=lambda m: m.period)
            for i in range(1, len(sorted_metrics)):
                prev = sorted_metrics[i - 1].intensity_value
                curr = sorted_metrics[i].intensity_value
                if prev > 0:
                    change_pct = ((curr - prev) / prev) * 100.0
                    outcome = (
                        QualityOutcome.PASS
                        if abs(change_pct) <= yoy_threshold_pct
                        else QualityOutcome.WARNING
                    )
                    self._quality_checks.append(QualityCheckResult(
                        check_type=QualityCheckType.REASONABLENESS,
                        outcome=outcome,
                        metric_id=sorted_metrics[i].metric_id,
                        details=(
                            f"YoY change {change_pct:+.2f}% for {gkey} "
                            f"{sorted_metrics[i - 1].period}->{sorted_metrics[i].period}"
                        ),
                        threshold=yoy_threshold_pct,
                        actual_value=abs(change_pct),
                    ))

    def _check_cross_framework(self) -> None:
        """Check that metrics are available for all required frameworks."""
        framework_coverage: Dict[str, int] = {}
        for m in self._metrics:
            for fw in m.frameworks:
                framework_coverage[fw] = framework_coverage.get(fw, 0) + 1

        for rule in self._scope_rules:
            count = framework_coverage.get(rule.framework, 0)
            self._quality_checks.append(QualityCheckResult(
                check_type=QualityCheckType.CROSS_FRAMEWORK,
                outcome=QualityOutcome.PASS if count > 0 else QualityOutcome.FAIL,
                details=f"Framework {rule.framework}: {count} metrics available",
                actual_value=float(count),
            ))

    # -------------------------------------------------------------------------
    # Calculation Helpers
    # -------------------------------------------------------------------------

    def _calculate_numerator(
        self, emissions: EmissionsDataSet, scope_incl: ScopeInclusion,
    ) -> float:
        """Calculate emissions numerator based on scope inclusion rule."""
        scope_map: Dict[ScopeInclusion, float] = {
            ScopeInclusion.SCOPE_1: emissions.scope1_tco2e,
            ScopeInclusion.SCOPE_2_LOCATION: emissions.scope2_location_tco2e,
            ScopeInclusion.SCOPE_2_MARKET: emissions.scope2_market_tco2e,
            ScopeInclusion.SCOPE_1_2_LOCATION: (
                emissions.scope1_tco2e + emissions.scope2_location_tco2e
            ),
            ScopeInclusion.SCOPE_1_2_MARKET: (
                emissions.scope1_tco2e + emissions.scope2_market_tco2e
            ),
            ScopeInclusion.SCOPE_1_2_3: (
                emissions.scope1_tco2e + emissions.scope2_location_tco2e
                + emissions.scope3_tco2e
            ),
            ScopeInclusion.SCOPE_3_ONLY: emissions.scope3_tco2e,
            ScopeInclusion.ALL_SCOPES: (
                emissions.scope1_tco2e + emissions.scope2_location_tco2e
                + emissions.scope3_tco2e
            ),
        }
        return scope_map.get(scope_incl, 0.0)

    def _map_intensity_unit(self, denom_unit: str) -> IntensityUnit:
        """Map denominator unit to intensity unit."""
        unit_map: Dict[str, IntensityUnit] = {
            "USD_million": IntensityUnit.TCO2E_PER_USD_MILLION,
            "EUR_million": IntensityUnit.TCO2E_PER_EUR_MILLION,
            "headcount": IntensityUnit.TCO2E_PER_FTE,
            "sqm": IntensityUnit.TCO2E_PER_SQM,
            "tonnes": IntensityUnit.TCO2E_PER_TONNE,
            "MWh": IntensityUnit.TCO2E_PER_MWH,
            "GWh": IntensityUnit.TCO2E_PER_GWH,
            "passenger_km": IntensityUnit.TCO2E_PER_PKM,
            "tonne_km": IntensityUnit.TCO2E_PER_TKM,
            "unit_count": IntensityUnit.TCO2E_PER_UNIT,
        }
        return unit_map.get(denom_unit, IntensityUnit.CUSTOM)

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: IntensityCalcInput, phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio
                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._emissions_by_period = {}
        self._denominators_by_key = {}
        self._scope_rules = []
        self._metrics = []
        self._quality_checks = []

    def _compute_provenance(self, result: IntensityCalcResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(p.provenance_hash for p in result.phases if p.provenance_hash)
        chain += f"|{result.workflow_id}|{result.organization_id}|{result.metrics_count}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
