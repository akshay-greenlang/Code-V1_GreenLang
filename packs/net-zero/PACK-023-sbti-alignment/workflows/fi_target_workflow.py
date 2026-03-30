# -*- coding: utf-8 -*-
"""
FI Target Workflow
======================

4-phase workflow for financial institution portfolio target setting
within PACK-023 SBTi Alignment Pack.  The workflow maps the
portfolio to asset classes with financed emissions attribution,
sets asset-class-specific targets using SBTi FINZ V1.0 methods,
calculates portfolio coverage and engagement metrics, and validates
the complete FI target package for submission readiness.

Phases:
    1. PortfolioMap       -- Map portfolio to asset classes with PCAF financed emissions
    2. AssetClassTarget   -- Set asset-class-specific targets (SDA/convergence/coverage)
    3. CoverageCalc       -- Calculate portfolio coverage and engagement metrics
    4. Validate           -- Validate against FINZ V1.0 requirements

Regulatory references:
    - SBTi Financial Institutions Net-Zero Standard V1.0 (2024)
    - SBTi FI Guidance V2.0 (2024)
    - PCAF Global GHG Accounting Standard V3.0 (2023)
    - PCAF Data Quality Framework (2023)
    - TCFD Recommendations (2017, updated 2022)
    - NZBA Guidelines V2.0 (2024)
    - SBTi Portfolio Coverage Approach (PCA) Guidance
    - SBTi Temperature Rating Methodology V3.0
    - Paris Agreement Art. 2.1(c)
    - ISO 14097:2021

Zero-hallucination: all thresholds from SBTi FI Standard V1.0.
PCAF scores from PCAF Global Standard V3.0.  No LLM calls in the
numeric computation path.

Author: GreenLang Team
Version: 23.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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

class AssetClass(str, Enum):
    """PCAF / SBTi FI asset classes."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BONDS = "corporate_bonds"
    BUSINESS_LOANS = "business_loans"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGES = "mortgages"
    MOTOR_VEHICLES = "motor_vehicles"
    SOVEREIGN_DEBT = "sovereign_debt"

class PCAFDataQuality(int, Enum):
    """PCAF data quality scores (1 = best, 5 = worst)."""

    SCORE_1 = 1  # Reported, verified emissions
    SCORE_2 = 2  # Reported, unverified emissions
    SCORE_3 = 3  # Physical activity-based estimates
    SCORE_4 = 4  # Economic activity-based estimates
    SCORE_5 = 5  # Estimated / proxy data

class FITargetMethod(str, Enum):
    """Target-setting methods for FI asset classes."""

    SDA = "sda"                       # Sectoral Decarbonization Approach
    CONVERGENCE = "convergence"       # Temperature convergence
    PCA = "portfolio_coverage"        # Portfolio Coverage Approach
    ENGAGEMENT = "engagement"         # Engagement-based
    ABSOLUTE = "absolute"             # Absolute emissions reduction
    TEMPERATURE = "temperature"       # Temperature rating

class TemperatureAlignment(str, Enum):
    """Temperature alignment classification."""

    ALIGNED_1_5C = "1.5C"
    ALIGNED_WB2C = "WB2C"
    ALIGNED_2C = "2C"
    MISALIGNED = "misaligned"
    NOT_ASSESSED = "not_assessed"

class ValidationSeverity(str, Enum):
    """Severity of a validation finding."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"

# =============================================================================
# REFERENCE DATA (Zero-Hallucination Lookups)
# =============================================================================

# Asset class display names
ASSET_CLASS_NAMES: Dict[str, str] = {
    "listed_equity": "Listed Equity & Corporate Bonds (Listed)",
    "corporate_bonds": "Corporate Bonds",
    "business_loans": "Business Loans & Unlisted Equity",
    "project_finance": "Project Finance",
    "commercial_real_estate": "Commercial Real Estate",
    "mortgages": "Residential Mortgages",
    "motor_vehicles": "Motor Vehicle Loans",
    "sovereign_debt": "Sovereign Debt",
}

# Required target methods by asset class (SBTi FINZ V1.0)
ASSET_CLASS_TARGET_METHODS: Dict[str, List[str]] = {
    "listed_equity": ["sda", "convergence", "portfolio_coverage", "temperature"],
    "corporate_bonds": ["sda", "convergence", "portfolio_coverage", "temperature"],
    "business_loans": ["sda", "convergence", "engagement"],
    "project_finance": ["sda", "absolute"],
    "commercial_real_estate": ["sda", "convergence"],
    "mortgages": ["sda", "convergence"],
    "motor_vehicles": ["sda", "convergence"],
    "sovereign_debt": ["engagement", "temperature"],
}

# PCAF recommended data quality by asset class
PCAF_RECOMMENDED_DQ: Dict[str, int] = {
    "listed_equity": 2,
    "corporate_bonds": 2,
    "business_loans": 3,
    "project_finance": 3,
    "commercial_real_estate": 3,
    "mortgages": 4,
    "motor_vehicles": 4,
    "sovereign_debt": 3,
}

# FINZ V1.0 minimum coverage requirements by asset class
FINZ_MIN_COVERAGE: Dict[str, float] = {
    "listed_equity": 67.0,       # 67% near-term, 90% long-term
    "corporate_bonds": 67.0,
    "business_loans": 67.0,
    "project_finance": 67.0,
    "commercial_real_estate": 67.0,
    "mortgages": 67.0,
    "motor_vehicles": 67.0,
    "sovereign_debt": 50.0,      # Lower threshold for sovereign
}

# Engagement minimum: SBTi requires 100% of high-emitting companies engaged
ENGAGEMENT_HIGH_EMITTER_PCT = 100.0
ENGAGEMENT_PORTFOLIO_MIN_PCT = 67.0

# Temperature score thresholds
TEMP_ALIGNED_1_5C = 1.5
TEMP_ALIGNED_WB2C = 1.75
TEMP_ALIGNED_2C = 2.0

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class PortfolioHolding(BaseModel):
    """A single holding in the FI portfolio."""

    holding_id: str = Field(default="")
    entity_name: str = Field(default="")
    asset_class: AssetClass = Field(...)
    outstanding_amount_usd: float = Field(default=0.0, ge=0.0)
    entity_value_usd: float = Field(default=0.0, ge=0.0,
                                     description="EVIC or total value for attribution")
    entity_emissions_tco2e: float = Field(default=0.0, ge=0.0,
                                           description="Total S1+S2 emissions of entity")
    data_quality: PCAFDataQuality = Field(default=PCAFDataQuality.SCORE_5)
    sector: str = Field(default="")
    has_sbti_target: bool = Field(default=False)
    temperature_score: float = Field(default=3.2,
                                      description="Temperature score (C)")
    country: str = Field(default="")
    is_high_emitter: bool = Field(default=False)
    is_engaged: bool = Field(default=False)

class AssetClassSummary(BaseModel):
    """Portfolio summary for a single asset class."""

    asset_class: str = Field(default="")
    asset_class_name: str = Field(default="")
    holding_count: int = Field(default=0)
    total_outstanding_usd: float = Field(default=0.0)
    pct_of_portfolio: float = Field(default=0.0)
    financed_emissions_tco2e: float = Field(default=0.0)
    pct_of_total_financed: float = Field(default=0.0)
    weighted_data_quality: float = Field(default=5.0)
    weighted_temperature_score: float = Field(default=3.2)
    recommended_target_method: str = Field(default="")

class AssetClassTarget(BaseModel):
    """Target definition for a single asset class."""

    asset_class: str = Field(default="")
    asset_class_name: str = Field(default="")
    target_method: FITargetMethod = Field(default=FITargetMethod.SDA)
    base_year: int = Field(default=2022)
    target_year: int = Field(default=2030)
    base_financed_emissions_tco2e: float = Field(default=0.0)
    target_financed_emissions_tco2e: float = Field(default=0.0)
    reduction_pct: float = Field(default=0.0)
    annual_reduction_rate_pct: float = Field(default=0.0)
    intensity_metric: str = Field(default="")
    base_intensity: float = Field(default=0.0)
    target_intensity: float = Field(default=0.0)
    engagement_target_pct: float = Field(default=0.0)
    portfolio_coverage_target_pct: float = Field(default=0.0)
    temperature_target: float = Field(default=0.0)
    temperature_alignment: TemperatureAlignment = Field(default=TemperatureAlignment.NOT_ASSESSED)
    notes: List[str] = Field(default_factory=list)

class CoverageMetrics(BaseModel):
    """Portfolio coverage and engagement metrics."""

    total_portfolio_usd: float = Field(default=0.0)
    total_financed_emissions_tco2e: float = Field(default=0.0)
    portfolio_coverage_pct: float = Field(default=0.0,
                                           description="% of FE with SBTi targets")
    engagement_coverage_pct: float = Field(default=0.0,
                                            description="% of FE with engagement")
    high_emitter_engagement_pct: float = Field(default=0.0,
                                                description="% of high-emitter FE engaged")
    weighted_portfolio_temperature: float = Field(default=0.0)
    weighted_data_quality_score: float = Field(default=0.0)
    asset_class_coverage: Dict[str, float] = Field(default_factory=dict)
    asset_class_engagement: Dict[str, float] = Field(default_factory=dict)
    meets_minimum_coverage: bool = Field(default=False)
    meets_engagement_requirement: bool = Field(default=False)

class ValidationFinding(BaseModel):
    """A single validation finding for FI targets."""

    finding_id: str = Field(default="")
    criterion: str = Field(default="")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    description: str = Field(default="")
    remediation: str = Field(default="")
    asset_class: str = Field(default="", description="Specific asset class or 'portfolio'")

class FIValidationResult(BaseModel):
    """Complete FI target validation result."""

    submission_ready: bool = Field(default=False)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    findings: List[ValidationFinding] = Field(default_factory=list)
    pass_count: int = Field(default=0)
    fail_count: int = Field(default=0)
    warning_count: int = Field(default=0)
    blocking_findings: List[str] = Field(default_factory=list)

class FITargetWorkflowConfig(BaseModel):
    """Configuration for the FI target workflow."""

    # Portfolio data
    holdings: List[PortfolioHolding] = Field(default_factory=list)
    base_year: int = Field(default=2022, ge=2015, le=2050)
    near_term_target_year: int = Field(default=2030, ge=2025, le=2040)
    long_term_target_year: int = Field(default=2050, ge=2035, le=2060)

    # Institution info
    institution_name: str = Field(default="")
    institution_type: str = Field(default="bank",
                                   description="bank, asset_manager, asset_owner, insurer")
    total_aum_usd: float = Field(default=0.0, ge=0.0)

    # Target preferences
    preferred_method: Optional[FITargetMethod] = Field(None)
    ambition_1_5c: bool = Field(default=True)

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class FITargetWorkflowResult(BaseModel):
    """Complete result from the FI target workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="fi_target")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    asset_class_summaries: List[AssetClassSummary] = Field(default_factory=list)
    asset_class_targets: List[AssetClassTarget] = Field(default_factory=list)
    coverage_metrics: Optional[CoverageMetrics] = Field(None)
    validation: Optional[FIValidationResult] = Field(None)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class FITargetWorkflow:
    """
    4-phase FI portfolio target setting workflow for SBTi FINZ V1.0.

    Maps the portfolio to asset classes with PCAF financed emissions
    attribution, sets asset-class-specific targets, calculates
    portfolio coverage and engagement metrics, and validates the
    complete target package against FINZ V1.0 requirements.

    Zero-hallucination: all thresholds from SBTi FI Standard V1.0.
    PCAF scores from PCAF Global Standard V3.0.  No LLM calls in
    the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = FITargetWorkflow()
        >>> config = FITargetWorkflowConfig(
        ...     holdings=[PortfolioHolding(
        ...         asset_class=AssetClass.LISTED_EQUITY,
        ...         outstanding_amount_usd=1e8,
        ...         entity_value_usd=1e9,
        ...         entity_emissions_tco2e=50000,
        ...     )],
        ... )
        >>> result = await wf.execute(config)
        >>> assert result.validation is not None
    """

    def __init__(self) -> None:
        """Initialise FITargetWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._ac_summaries: List[AssetClassSummary] = []
        self._ac_targets: List[AssetClassTarget] = []
        self._coverage: Optional[CoverageMetrics] = None
        self._validation: Optional[FIValidationResult] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: FITargetWorkflowConfig) -> FITargetWorkflowResult:
        """
        Execute the 4-phase FI target workflow.

        Args:
            config: FI target configuration with portfolio holdings,
                target years, and institution information.

        Returns:
            FITargetWorkflowResult with asset class summaries, targets,
            coverage metrics, and validation results.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting FI target workflow %s, holdings=%d, institution=%s",
            self.workflow_id, len(config.holdings), config.institution_name,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Portfolio Mapping
            phase1 = await self._phase_portfolio_map(config)
            self._phase_results.append(phase1)

            # Phase 2: Asset Class Target Setting
            phase2 = await self._phase_asset_class_target(config)
            self._phase_results.append(phase2)

            # Phase 3: Coverage Calculation
            phase3 = await self._phase_coverage_calc(config)
            self._phase_results.append(phase3)

            # Phase 4: Validation
            phase4 = await self._phase_validate(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("FI target workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = FITargetWorkflowResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            asset_class_summaries=self._ac_summaries,
            asset_class_targets=self._ac_targets,
            coverage_metrics=self._coverage,
            validation=self._validation,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "FI target workflow %s completed in %.2fs, targets=%d",
            self.workflow_id, elapsed, len(self._ac_targets),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Portfolio Mapping
    # -------------------------------------------------------------------------

    async def _phase_portfolio_map(self, config: FITargetWorkflowConfig) -> PhaseResult:
        """Map portfolio to asset classes with PCAF financed emissions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._ac_summaries = []

        # Group holdings by asset class
        ac_groups: Dict[str, List[PortfolioHolding]] = {}
        for holding in config.holdings:
            ac = holding.asset_class.value
            if ac not in ac_groups:
                ac_groups[ac] = []
            ac_groups[ac].append(holding)

        total_portfolio_usd = sum(h.outstanding_amount_usd for h in config.holdings)
        total_financed_emissions = 0.0

        for ac, holdings in ac_groups.items():
            ac_outstanding = sum(h.outstanding_amount_usd for h in holdings)
            pct_of_portfolio = (ac_outstanding / total_portfolio_usd * 100.0) if total_portfolio_usd > 0 else 0.0

            # Calculate financed emissions for each holding using PCAF attribution
            ac_financed = 0.0
            dq_weighted_sum = 0.0
            temp_weighted_sum = 0.0
            total_weight = 0.0

            for h in holdings:
                # Attribution factor = outstanding / entity value
                if h.entity_value_usd > 0:
                    attribution = h.outstanding_amount_usd / h.entity_value_usd
                else:
                    attribution = 1.0  # Conservative: full emissions if no entity value

                financed = attribution * h.entity_emissions_tco2e
                ac_financed += financed

                # Weighted DQ score
                weight = h.outstanding_amount_usd
                dq_weighted_sum += h.data_quality.value * weight
                temp_weighted_sum += h.temperature_score * weight
                total_weight += weight

            total_financed_emissions += ac_financed

            # Weighted averages
            weighted_dq = (dq_weighted_sum / total_weight) if total_weight > 0 else 5.0
            weighted_temp = (temp_weighted_sum / total_weight) if total_weight > 0 else 3.2

            # Recommended method
            methods = ASSET_CLASS_TARGET_METHODS.get(ac, ["sda"])
            recommended = methods[0] if methods else "sda"

            self._ac_summaries.append(AssetClassSummary(
                asset_class=ac,
                asset_class_name=ASSET_CLASS_NAMES.get(ac, ac),
                holding_count=len(holdings),
                total_outstanding_usd=round(ac_outstanding, 2),
                pct_of_portfolio=round(pct_of_portfolio, 2),
                financed_emissions_tco2e=round(ac_financed, 2),
                pct_of_total_financed=0.0,  # Updated after total
                weighted_data_quality=round(weighted_dq, 2),
                weighted_temperature_score=round(weighted_temp, 2),
                recommended_target_method=recommended,
            ))

        # Update pct_of_total_financed
        for acs in self._ac_summaries:
            if total_financed_emissions > 0:
                acs.pct_of_total_financed = round(
                    acs.financed_emissions_tco2e / total_financed_emissions * 100.0, 2
                )

        # Sort by financed emissions descending
        self._ac_summaries.sort(
            key=lambda s: s.financed_emissions_tco2e, reverse=True
        )

        # Data quality warnings
        for acs in self._ac_summaries:
            recommended_dq = PCAF_RECOMMENDED_DQ.get(acs.asset_class, 3)
            if acs.weighted_data_quality > recommended_dq + 1:
                warnings.append(
                    f"{acs.asset_class_name}: data quality {acs.weighted_data_quality:.1f} "
                    f"is below recommended {recommended_dq} for this asset class"
                )

        outputs["asset_classes"] = len(self._ac_summaries)
        outputs["total_holdings"] = len(config.holdings)
        outputs["total_portfolio_usd"] = round(total_portfolio_usd, 2)
        outputs["total_financed_emissions_tco2e"] = round(total_financed_emissions, 2)
        outputs["asset_class_breakdown"] = {
            s.asset_class: round(s.financed_emissions_tco2e, 2) for s in self._ac_summaries
        }

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Portfolio map: %d classes, %d holdings, FE=%.2f tCO2e",
            len(self._ac_summaries), len(config.holdings), total_financed_emissions,
        )
        return PhaseResult(
            phase_name="portfolio_map",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Asset Class Target Setting
    # -------------------------------------------------------------------------

    async def _phase_asset_class_target(self, config: FITargetWorkflowConfig) -> PhaseResult:
        """Set asset-class-specific targets using FINZ V1.0 methods."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._ac_targets = []

        # Group holdings for target calculations
        ac_holdings: Dict[str, List[PortfolioHolding]] = {}
        for h in config.holdings:
            ac = h.asset_class.value
            if ac not in ac_holdings:
                ac_holdings[ac] = []
            ac_holdings[ac].append(h)

        for acs in self._ac_summaries:
            ac = acs.asset_class
            methods = ASSET_CLASS_TARGET_METHODS.get(ac, ["sda"])
            holdings = ac_holdings.get(ac, [])

            # Select method
            if config.preferred_method and config.preferred_method.value in methods:
                method = config.preferred_method
            else:
                method = FITargetMethod(methods[0])

            # Calculate target based on method
            target = self._calculate_ac_target(
                acs, method, config, holdings,
            )
            self._ac_targets.append(target)

        # Sort targets by financed emissions
        self._ac_targets.sort(
            key=lambda t: t.base_financed_emissions_tco2e, reverse=True
        )

        outputs["targets_set"] = len(self._ac_targets)
        for t in self._ac_targets:
            outputs[f"{t.asset_class}_method"] = t.target_method.value
            outputs[f"{t.asset_class}_reduction_pct"] = round(t.reduction_pct, 2)
        outputs["methods_used"] = list(set(t.target_method.value for t in self._ac_targets))

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Asset class targets: %d set, methods=%s",
            len(self._ac_targets),
            list(set(t.target_method.value for t in self._ac_targets)),
        )
        return PhaseResult(
            phase_name="asset_class_target",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _calculate_ac_target(
        self, acs: AssetClassSummary, method: FITargetMethod,
        config: FITargetWorkflowConfig, holdings: List[PortfolioHolding],
    ) -> AssetClassTarget:
        """Calculate target for a specific asset class."""
        base_fe = acs.financed_emissions_tco2e
        total_years = config.near_term_target_year - config.base_year
        notes: List[str] = []

        # Default annual reduction rate based on ambition
        annual_rate = 0.042 if config.ambition_1_5c else 0.025  # 4.2% or 2.5%

        # Method-specific calculations
        if method == FITargetMethod.SDA:
            # SDA convergence for homogeneous sectors
            reduction_pct = min(
                (1.0 - (1.0 - annual_rate) ** total_years) * 100.0, 95.0
            )
            target_fe = base_fe * (1.0 - reduction_pct / 100.0)
            notes.append(
                f"SDA convergence: {annual_rate * 100:.1f}%/yr for {total_years} years"
            )

        elif method == FITargetMethod.CONVERGENCE:
            # Temperature convergence
            reduction_pct = min(
                (1.0 - (1.0 - annual_rate) ** total_years) * 100.0, 95.0
            )
            target_fe = base_fe * (1.0 - reduction_pct / 100.0)
            notes.append(
                f"Temperature convergence to {'1.5C' if config.ambition_1_5c else 'WB2C'}"
            )

        elif method == FITargetMethod.PCA:
            # Portfolio Coverage: target % of FE with SBTi targets
            current_coverage = sum(
                1 for h in holdings if h.has_sbti_target
            ) / max(len(holdings), 1) * 100.0
            target_coverage = min(
                current_coverage + (100.0 - current_coverage) * 0.5, 100.0
            )
            reduction_pct = 0.0
            target_fe = base_fe  # Coverage approach doesn't set FE target
            notes.append(
                f"Portfolio coverage: current {current_coverage:.1f}% -> "
                f"target {target_coverage:.1f}%"
            )

        elif method == FITargetMethod.ENGAGEMENT:
            # Engagement-based
            current_engaged = sum(
                1 for h in holdings if h.is_engaged
            ) / max(len(holdings), 1) * 100.0
            target_engagement = min(current_engaged + 20.0, 100.0)
            reduction_pct = 0.0
            target_fe = base_fe
            notes.append(
                f"Engagement: current {current_engaged:.1f}% -> "
                f"target {target_engagement:.1f}%"
            )

        elif method == FITargetMethod.TEMPERATURE:
            # Temperature rating
            target_temp = TEMP_ALIGNED_1_5C if config.ambition_1_5c else TEMP_ALIGNED_WB2C
            reduction_pct = 0.0
            target_fe = base_fe
            notes.append(
                f"Temperature alignment target: {target_temp}C"
            )

        else:
            # Absolute reduction
            reduction_pct = min(
                (1.0 - (1.0 - annual_rate) ** total_years) * 100.0, 95.0
            )
            target_fe = base_fe * (1.0 - reduction_pct / 100.0)

        # Temperature alignment classification
        temp = acs.weighted_temperature_score
        if temp <= TEMP_ALIGNED_1_5C:
            temp_alignment = TemperatureAlignment.ALIGNED_1_5C
        elif temp <= TEMP_ALIGNED_WB2C:
            temp_alignment = TemperatureAlignment.ALIGNED_WB2C
        elif temp <= TEMP_ALIGNED_2C:
            temp_alignment = TemperatureAlignment.ALIGNED_2C
        else:
            temp_alignment = TemperatureAlignment.MISALIGNED

        # Engagement and coverage targets
        engagement_pct = 0.0
        coverage_pct = 0.0
        if method == FITargetMethod.ENGAGEMENT:
            engagement_pct = min(
                sum(1 for h in holdings if h.is_engaged) / max(len(holdings), 1) * 100.0 + 20.0,
                100.0,
            )
        if method == FITargetMethod.PCA:
            coverage_pct = min(
                sum(1 for h in holdings if h.has_sbti_target) / max(len(holdings), 1) * 100.0 + 20.0,
                100.0,
            )

        return AssetClassTarget(
            asset_class=acs.asset_class,
            asset_class_name=acs.asset_class_name,
            target_method=method,
            base_year=config.base_year,
            target_year=config.near_term_target_year,
            base_financed_emissions_tco2e=round(base_fe, 2),
            target_financed_emissions_tco2e=round(target_fe, 2),
            reduction_pct=round(reduction_pct, 2),
            annual_reduction_rate_pct=round(annual_rate * 100.0, 2),
            engagement_target_pct=round(engagement_pct, 2),
            portfolio_coverage_target_pct=round(coverage_pct, 2),
            temperature_target=TEMP_ALIGNED_1_5C if config.ambition_1_5c else TEMP_ALIGNED_WB2C,
            temperature_alignment=temp_alignment,
            notes=notes,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Coverage Calculation
    # -------------------------------------------------------------------------

    async def _phase_coverage_calc(self, config: FITargetWorkflowConfig) -> PhaseResult:
        """Calculate portfolio coverage and engagement metrics."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_portfolio = sum(h.outstanding_amount_usd for h in config.holdings)
        total_fe = sum(s.financed_emissions_tco2e for s in self._ac_summaries)

        # Portfolio coverage: % of FE where counterparties have SBTi targets
        fe_with_targets = 0.0
        fe_engaged = 0.0
        fe_high_emitter = 0.0
        fe_high_emitter_engaged = 0.0
        dq_weighted_sum = 0.0
        temp_weighted_sum = 0.0
        total_weight = 0.0

        for h in config.holdings:
            # Calculate financed emissions for this holding
            if h.entity_value_usd > 0:
                attribution = h.outstanding_amount_usd / h.entity_value_usd
            else:
                attribution = 1.0
            holding_fe = attribution * h.entity_emissions_tco2e

            if h.has_sbti_target:
                fe_with_targets += holding_fe

            if h.is_engaged:
                fe_engaged += holding_fe

            if h.is_high_emitter:
                fe_high_emitter += holding_fe
                if h.is_engaged:
                    fe_high_emitter_engaged += holding_fe

            weight = h.outstanding_amount_usd
            dq_weighted_sum += h.data_quality.value * weight
            temp_weighted_sum += h.temperature_score * weight
            total_weight += weight

        portfolio_coverage = (fe_with_targets / total_fe * 100.0) if total_fe > 0 else 0.0
        engagement_coverage = (fe_engaged / total_fe * 100.0) if total_fe > 0 else 0.0
        high_emitter_engagement = (
            fe_high_emitter_engaged / fe_high_emitter * 100.0
        ) if fe_high_emitter > 0 else 0.0

        weighted_dq = (dq_weighted_sum / total_weight) if total_weight > 0 else 5.0
        weighted_temp = (temp_weighted_sum / total_weight) if total_weight > 0 else 3.2

        # Asset class coverage
        ac_coverage: Dict[str, float] = {}
        ac_engagement: Dict[str, float] = {}
        ac_holdings: Dict[str, List[PortfolioHolding]] = {}
        for h in config.holdings:
            ac = h.asset_class.value
            if ac not in ac_holdings:
                ac_holdings[ac] = []
            ac_holdings[ac].append(h)

        for ac, holdings in ac_holdings.items():
            ac_fe_total = 0.0
            ac_fe_targets = 0.0
            ac_fe_engaged = 0.0
            for h in holdings:
                attr = h.outstanding_amount_usd / h.entity_value_usd if h.entity_value_usd > 0 else 1.0
                hfe = attr * h.entity_emissions_tco2e
                ac_fe_total += hfe
                if h.has_sbti_target:
                    ac_fe_targets += hfe
                if h.is_engaged:
                    ac_fe_engaged += hfe

            ac_coverage[ac] = round(
                ac_fe_targets / ac_fe_total * 100.0 if ac_fe_total > 0 else 0.0, 2
            )
            ac_engagement[ac] = round(
                ac_fe_engaged / ac_fe_total * 100.0 if ac_fe_total > 0 else 0.0, 2
            )

        # Check minimum coverage requirements
        meets_coverage = True
        for ac, cov in ac_coverage.items():
            min_req = FINZ_MIN_COVERAGE.get(ac, 67.0)
            if cov < min_req:
                meets_coverage = False
                warnings.append(
                    f"{ASSET_CLASS_NAMES.get(ac, ac)}: coverage {cov:.1f}% "
                    f"< minimum {min_req:.0f}%"
                )

        meets_engagement = high_emitter_engagement >= ENGAGEMENT_HIGH_EMITTER_PCT
        if not meets_engagement and fe_high_emitter > 0:
            warnings.append(
                f"High-emitter engagement: {high_emitter_engagement:.1f}% "
                f"< required {ENGAGEMENT_HIGH_EMITTER_PCT:.0f}%"
            )

        self._coverage = CoverageMetrics(
            total_portfolio_usd=round(total_portfolio, 2),
            total_financed_emissions_tco2e=round(total_fe, 2),
            portfolio_coverage_pct=round(portfolio_coverage, 2),
            engagement_coverage_pct=round(engagement_coverage, 2),
            high_emitter_engagement_pct=round(high_emitter_engagement, 2),
            weighted_portfolio_temperature=round(weighted_temp, 2),
            weighted_data_quality_score=round(weighted_dq, 2),
            asset_class_coverage=ac_coverage,
            asset_class_engagement=ac_engagement,
            meets_minimum_coverage=meets_coverage,
            meets_engagement_requirement=meets_engagement,
        )

        outputs["portfolio_coverage_pct"] = round(portfolio_coverage, 2)
        outputs["engagement_coverage_pct"] = round(engagement_coverage, 2)
        outputs["high_emitter_engagement_pct"] = round(high_emitter_engagement, 2)
        outputs["weighted_temperature"] = round(weighted_temp, 2)
        outputs["weighted_data_quality"] = round(weighted_dq, 2)
        outputs["meets_coverage"] = meets_coverage
        outputs["meets_engagement"] = meets_engagement
        outputs["asset_class_coverage"] = ac_coverage

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Coverage calc: coverage=%.1f%%, engagement=%.1f%%, temp=%.2fC",
            portfolio_coverage, engagement_coverage, weighted_temp,
        )
        return PhaseResult(
            phase_name="coverage_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Validation
    # -------------------------------------------------------------------------

    async def _phase_validate(self, config: FITargetWorkflowConfig) -> PhaseResult:
        """Validate against FINZ V1.0 requirements."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        findings: List[ValidationFinding] = []
        finding_counter = 0

        # F1: Portfolio completeness
        finding_counter += 1
        has_holdings = len(config.holdings) > 0
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="Portfolio completeness",
            severity=ValidationSeverity.PASS if has_holdings else ValidationSeverity.FAIL,
            description=(
                f"Portfolio contains {len(config.holdings)} holdings across "
                f"{len(self._ac_summaries)} asset classes"
                if has_holdings else "No portfolio holdings provided"
            ),
            remediation="" if has_holdings else "Provide portfolio holdings data",
            asset_class="portfolio",
        ))

        # F2: PCAF data quality
        finding_counter += 1
        dq_ok = self._coverage and self._coverage.weighted_data_quality_score <= 3.5
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="PCAF data quality",
            severity=ValidationSeverity.PASS if dq_ok else ValidationSeverity.WARNING,
            description=(
                f"Weighted DQ score: {self._coverage.weighted_data_quality_score:.1f}"
                if self._coverage else "No coverage data"
            ),
            remediation="" if dq_ok else "Improve data quality to PCAF Score 3 or better",
            asset_class="portfolio",
        ))

        # F3: Coverage minimum per asset class
        if self._coverage:
            for ac, cov in self._coverage.asset_class_coverage.items():
                finding_counter += 1
                min_req = FINZ_MIN_COVERAGE.get(ac, 67.0)
                cov_ok = cov >= min_req
                findings.append(ValidationFinding(
                    finding_id=f"FI-{finding_counter:03d}",
                    criterion=f"Coverage minimum ({ac})",
                    severity=ValidationSeverity.PASS if cov_ok else ValidationSeverity.FAIL,
                    description=(
                        f"{ASSET_CLASS_NAMES.get(ac, ac)}: {cov:.1f}% coverage "
                        f"({'meets' if cov_ok else 'below'} {min_req:.0f}% minimum)"
                    ),
                    remediation="" if cov_ok else (
                        f"Engage counterparties in {ASSET_CLASS_NAMES.get(ac, ac)} "
                        f"to set SBTi targets; need {min_req:.0f}% coverage"
                    ),
                    asset_class=ac,
                ))

        # F4: Engagement requirement
        finding_counter += 1
        eng_ok = self._coverage and self._coverage.meets_engagement_requirement
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="High-emitter engagement",
            severity=ValidationSeverity.PASS if eng_ok else ValidationSeverity.FAIL,
            description=(
                f"High-emitter engagement: {self._coverage.high_emitter_engagement_pct:.1f}%"
                if self._coverage else "No engagement data"
            ),
            remediation="" if eng_ok else (
                "Engage 100% of high-emitting portfolio companies on SBTi target setting"
            ),
            asset_class="portfolio",
        ))

        # F5: Temperature alignment
        finding_counter += 1
        temp_ok = self._coverage and self._coverage.weighted_portfolio_temperature <= TEMP_ALIGNED_WB2C
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="Temperature alignment",
            severity=ValidationSeverity.PASS if temp_ok else ValidationSeverity.WARNING,
            description=(
                f"Portfolio temperature: {self._coverage.weighted_portfolio_temperature:.2f}C"
                if self._coverage else "No temperature data"
            ),
            remediation="" if temp_ok else "Reduce portfolio temperature to below 1.75C (WB2C)",
            asset_class="portfolio",
        ))

        # F6: All material asset classes have targets
        finding_counter += 1
        targeted_acs = {t.asset_class for t in self._ac_targets}
        material_acs = {s.asset_class for s in self._ac_summaries if s.pct_of_portfolio >= 5.0}
        missing_targets = material_acs - targeted_acs
        targets_ok = len(missing_targets) == 0
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="Material asset class coverage",
            severity=ValidationSeverity.PASS if targets_ok else ValidationSeverity.FAIL,
            description=(
                "All material asset classes (>5% of portfolio) have targets"
                if targets_ok else f"Missing targets for: {', '.join(missing_targets)}"
            ),
            remediation="" if targets_ok else (
                f"Set targets for material asset classes: {', '.join(missing_targets)}"
            ),
            asset_class="portfolio",
        ))

        # F7: Target ambition
        finding_counter += 1
        ambition_ok = all(
            t.annual_reduction_rate_pct >= 2.5 for t in self._ac_targets
            if t.target_method in (FITargetMethod.SDA, FITargetMethod.CONVERGENCE, FITargetMethod.ABSOLUTE)
        )
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="Target ambition",
            severity=ValidationSeverity.PASS if ambition_ok else ValidationSeverity.WARNING,
            description=(
                "All emission reduction targets meet WB2C minimum ambition"
                if ambition_ok else "Some targets below WB2C minimum ambition (2.5%/yr)"
            ),
            remediation="" if ambition_ok else "Increase annual reduction rate to at least 2.5%/yr",
            asset_class="portfolio",
        ))

        # F8: Base year
        finding_counter += 1
        by_ok = config.base_year >= 2015
        findings.append(ValidationFinding(
            finding_id=f"FI-{finding_counter:03d}",
            criterion="Base year validity",
            severity=ValidationSeverity.PASS if by_ok else ValidationSeverity.FAIL,
            description=f"Base year: {config.base_year} ({'valid' if by_ok else 'before 2015'})",
            remediation="" if by_ok else "Select base year 2015 or later",
            asset_class="portfolio",
        ))

        # Calculate readiness
        pass_count = sum(1 for f in findings if f.severity == ValidationSeverity.PASS)
        fail_count = sum(1 for f in findings if f.severity == ValidationSeverity.FAIL)
        warning_count = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
        total = len(findings)

        readiness = (pass_count / total * 100.0) if total > 0 else 0.0
        submission_ready = fail_count == 0

        blocking = [f.finding_id for f in findings if f.severity == ValidationSeverity.FAIL]

        self._validation = FIValidationResult(
            submission_ready=submission_ready,
            readiness_score=round(readiness, 2),
            findings=findings,
            pass_count=pass_count,
            fail_count=fail_count,
            warning_count=warning_count,
            blocking_findings=blocking,
        )

        outputs["submission_ready"] = submission_ready
        outputs["readiness_score"] = round(readiness, 2)
        outputs["pass_count"] = pass_count
        outputs["fail_count"] = fail_count
        outputs["warning_count"] = warning_count
        outputs["blocking_findings"] = len(blocking)

        if not submission_ready:
            warnings.append(
                f"Not submission-ready: {fail_count} criteria failed "
                f"({', '.join(blocking)})"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Validate: readiness=%.1f%%, submission_ready=%s, findings=%d",
            readiness, submission_ready, len(findings),
        )
        return PhaseResult(
            phase_name="validate",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
