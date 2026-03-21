# -*- coding: utf-8 -*-
"""
Full Acceleration Workflow
================================

8-phase master workflow that chains all net-zero acceleration sub-workflows
into a unified strategy within PACK-022 Net-Zero Acceleration Pack.
This orchestrator runs scenario analysis, SDA target setting, supplier
engagement, transition finance, advanced progress tracking, temperature
alignment, VCMI certification, and compiles an acceleration strategy.

Phases:
    1. Scenarios       -- Run scenario analysis workflow
    2. SDA             -- Run SDA target workflow (conditional: SDA-eligible sector)
    3. Suppliers       -- Run supplier engagement program workflow
    4. Finance         -- Run transition finance workflow
    5. Progress        -- Run advanced progress tracking workflow
    6. TempScore       -- Run temperature alignment workflow
    7. VCMI            -- Run VCMI certification workflow (conditional: has credits)
    8. Strategy        -- Compile unified acceleration strategy

Zero-hallucination: all numeric results are derived from deterministic
sub-workflow calculations.  SHA-256 provenance hashes across the full
chain guarantee end-to-end auditability.

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .scenario_analysis_workflow import (
    ScenarioAnalysisConfig,
    ScenarioAnalysisResult,
    ScenarioAnalysisWorkflow,
    ScenarioDefinition,
)
from .sda_target_workflow import (
    SDATargetConfig,
    SDATargetResult,
    SDATargetWorkflow,
    SDA_BENCHMARKS,
)
from .supplier_program_workflow import (
    SupplierProgramConfig,
    SupplierProgramResult,
    SupplierProgramWorkflow,
    SupplierRecord,
)
from .transition_finance_workflow import (
    CapExItem,
    TransitionFinanceConfig,
    TransitionFinanceResult,
    TransitionFinanceWorkflow,
)
from .advanced_progress_workflow import (
    AdvancedProgressConfig,
    AdvancedProgressResult,
    AdvancedProgressWorkflow,
    AnnualEmissionRecord,
    TargetPathwayPoint,
)
from .temperature_alignment_workflow import (
    EntityTarget,
    EntityWeight,
    TemperatureAlignmentConfig,
    TemperatureAlignmentResult,
    TemperatureAlignmentWorkflow,
)
from .vcmi_certification_workflow import (
    CreditPortfolioItem,
    VCMICertificationConfig,
    VCMICertificationResult,
    VCMICertificationWorkflow,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


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


class AccelerationMaturity(str, Enum):
    """Net-zero acceleration maturity level."""

    NASCENT = "nascent"
    EMERGING = "emerging"
    ACCELERATING = "accelerating"
    ADVANCED = "advanced"
    LEADING = "leading"


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


class AccelerationScorecard(BaseModel):
    """Net-zero acceleration scorecard."""

    scenario_score: float = Field(default=0.0, ge=0.0, le=100.0)
    target_score: float = Field(default=0.0, ge=0.0, le=100.0)
    supplier_score: float = Field(default=0.0, ge=0.0, le=100.0)
    finance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    progress_score: float = Field(default=0.0, ge=0.0, le=100.0)
    temperature_score: float = Field(default=0.0, ge=0.0, le=100.0)
    vcmi_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    maturity: AccelerationMaturity = Field(default=AccelerationMaturity.NASCENT)


class AccelerationStrategySummary(BaseModel):
    """Unified acceleration strategy summary."""

    assessment_date: str = Field(default="")
    recommended_scenario: str = Field(default="")
    recommended_scenario_score: float = Field(default=0.0)
    sda_sector: str = Field(default="")
    sda_eligible: bool = Field(default=False)
    sda_near_term_intensity: float = Field(default=0.0)
    supplier_coverage_pct: float = Field(default=0.0)
    supplier_reduction_tco2e: float = Field(default=0.0)
    climate_capex_pct: float = Field(default=0.0)
    taxonomy_aligned_pct: float = Field(default=0.0)
    total_npv_usd: float = Field(default=0.0)
    cumulative_reduction_pct: float = Field(default=0.0)
    forecast_trend: str = Field(default="")
    critical_alerts: int = Field(default=0)
    portfolio_temperature_c: float = Field(default=0.0)
    paris_aligned: bool = Field(default=False)
    vcmi_tier: str = Field(default="")
    vcmi_eligible: bool = Field(default=False)
    acceleration_score: float = Field(default=0.0)
    maturity_level: str = Field(default="")
    key_actions: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)


class FullAccelerationConfig(BaseModel):
    """Configuration combining all sub-workflow configs."""

    # Scenario Analysis
    scenarios: List[ScenarioDefinition] = Field(default_factory=list)
    baseline_emissions_tco2e: float = Field(default=100000.0, ge=0.0)
    monte_carlo_runs: int = Field(default=1000, ge=100, le=100000)
    seed: int = Field(default=42)
    discount_rate: float = Field(default=0.08, ge=0.0, le=0.30)
    carbon_price_scenario: str = Field(default="moderate")

    # SDA Target
    sector: str = Field(default="other")
    base_year: int = Field(default=2024, ge=2015, le=2050)
    base_year_intensity: float = Field(default=0.0, ge=0.0)
    base_year_activity: float = Field(default=0.0, ge=0.0)
    activity_growth_rate: float = Field(default=0.02, ge=-0.10, le=0.20)
    near_term_target_year: int = Field(default=2030)

    # Supplier Program
    suppliers: List[SupplierRecord] = Field(default_factory=list)
    scope3_baseline_tco2e: float = Field(default=50000.0, ge=0.0)
    coverage_target_pct: float = Field(default=67.0, ge=0.0, le=100.0)
    engagement_budget_usd: float = Field(default=500000.0, ge=0.0)

    # Transition Finance
    capex_items: List[CapExItem] = Field(default_factory=list)
    carbon_price_usd: float = Field(default=100.0, ge=0.0)

    # Advanced Progress
    annual_data: List[AnnualEmissionRecord] = Field(default_factory=list)
    target_pathway: List[TargetPathwayPoint] = Field(default_factory=list)

    # Temperature Alignment
    entity_targets: List[EntityTarget] = Field(default_factory=list)
    entity_weights: List[EntityWeight] = Field(default_factory=list)
    aggregation_method: str = Field(default="wats")

    # VCMI Certification
    has_sbti_target: bool = Field(default=False)
    sbti_validated: bool = Field(default=False)
    reduction_progress_pct: float = Field(default=0.0)
    credit_portfolio: List[CreditPortfolioItem] = Field(default_factory=list)
    residual_emissions_tco2e: float = Field(default=0.0)
    vcmi_target_tier: str = Field(default="silver")

    # Conditional flags
    skip_sda: bool = Field(default=False, description="Force-skip SDA phase")
    skip_vcmi: bool = Field(default=False, description="Force-skip VCMI phase")

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class FullAccelerationResult(BaseModel):
    """Complete result from the full acceleration workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_acceleration")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    scenario_result: Optional[ScenarioAnalysisResult] = Field(None)
    sda_result: Optional[SDATargetResult] = Field(None)
    supplier_result: Optional[SupplierProgramResult] = Field(None)
    finance_result: Optional[TransitionFinanceResult] = Field(None)
    progress_result: Optional[AdvancedProgressResult] = Field(None)
    temperature_result: Optional[TemperatureAlignmentResult] = Field(None)
    vcmi_result: Optional[VCMICertificationResult] = Field(None)
    scorecard: AccelerationScorecard = Field(default_factory=AccelerationScorecard)
    strategy: AccelerationStrategySummary = Field(default_factory=AccelerationStrategySummary)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullAccelerationWorkflow:
    """
    8-phase master workflow chaining all net-zero acceleration sub-workflows.

    Runs scenario analysis, SDA targets (conditional), supplier engagement,
    transition finance, advanced progress, temperature alignment, VCMI
    certification (conditional), and compiles an acceleration strategy.

    Conditional phases:
    - SDA is skipped if the sector is not SDA-eligible or skip_sda is True.
    - VCMI is skipped if no carbon credits exist or skip_vcmi is True.

    Zero-hallucination: all numeric values propagate from deterministic
    sub-workflow calculations.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = FullAccelerationWorkflow()
        >>> config = FullAccelerationConfig(sector="steel", ...)
        >>> result = await wf.execute(config)
        >>> assert result.status in (WorkflowStatus.COMPLETED, WorkflowStatus.PARTIAL)
    """

    def __init__(self) -> None:
        """Initialise FullAccelerationWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._scenario_result: Optional[ScenarioAnalysisResult] = None
        self._sda_result: Optional[SDATargetResult] = None
        self._supplier_result: Optional[SupplierProgramResult] = None
        self._finance_result: Optional[TransitionFinanceResult] = None
        self._progress_result: Optional[AdvancedProgressResult] = None
        self._temp_result: Optional[TemperatureAlignmentResult] = None
        self._vcmi_result: Optional[VCMICertificationResult] = None
        self._scorecard: AccelerationScorecard = AccelerationScorecard()
        self._strategy: AccelerationStrategySummary = AccelerationStrategySummary()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: FullAccelerationConfig) -> FullAccelerationResult:
        """
        Execute the 8-phase full acceleration workflow.

        Args:
            config: Full acceleration configuration combining all
                sub-workflow configurations.

        Returns:
            FullAccelerationResult with all sub-results and unified strategy.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting full acceleration workflow %s, sector=%s",
            self.workflow_id, config.sector,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Scenario Analysis
            phase1 = await self._phase_scenarios(config)
            self._phase_results.append(phase1)

            # Phase 2: SDA Targets (conditional)
            phase2 = await self._phase_sda(config)
            self._phase_results.append(phase2)

            # Phase 3: Supplier Program
            phase3 = await self._phase_suppliers(config)
            self._phase_results.append(phase3)

            # Phase 4: Transition Finance
            phase4 = await self._phase_finance(config)
            self._phase_results.append(phase4)

            # Phase 5: Advanced Progress
            phase5 = await self._phase_progress(config)
            self._phase_results.append(phase5)

            # Phase 6: Temperature Alignment
            phase6 = await self._phase_temperature(config)
            self._phase_results.append(phase6)

            # Phase 7: VCMI Certification (conditional)
            phase7 = await self._phase_vcmi(config)
            self._phase_results.append(phase7)

            # Phase 8: Strategy Compilation
            phase8 = await self._phase_strategy(config)
            self._phase_results.append(phase8)

            failed = [
                p for p in self._phase_results
                if p.status == PhaseStatus.FAILED
            ]
            skipped = [
                p for p in self._phase_results
                if p.status == PhaseStatus.SKIPPED
            ]
            if not failed:
                overall_status = WorkflowStatus.COMPLETED
            elif len(failed) < len(self._phase_results) - len(skipped):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Full acceleration workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = FullAccelerationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            scenario_result=self._scenario_result,
            sda_result=self._sda_result,
            supplier_result=self._supplier_result,
            finance_result=self._finance_result,
            progress_result=self._progress_result,
            temperature_result=self._temp_result,
            vcmi_result=self._vcmi_result,
            scorecard=self._scorecard,
            strategy=self._strategy,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Full acceleration workflow %s completed in %.2fs, maturity=%s",
            self.workflow_id, elapsed, self._scorecard.maturity.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Scenario Analysis
    # -------------------------------------------------------------------------

    async def _phase_scenarios(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run scenario analysis workflow."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            sa_config = ScenarioAnalysisConfig(
                scenarios=config.scenarios,
                baseline_emissions_tco2e=config.baseline_emissions_tco2e,
                monte_carlo_runs=config.monte_carlo_runs,
                seed=config.seed,
                discount_rate=config.discount_rate,
                carbon_price_scenario=config.carbon_price_scenario,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = ScenarioAnalysisWorkflow()
            self._scenario_result = await wf.execute(sa_config)

            outputs["status"] = self._scenario_result.status.value
            outputs["recommended_scenario"] = self._scenario_result.decision_matrix.recommended_scenario_id
            outputs["recommended_score"] = round(
                self._scenario_result.decision_matrix.comparisons[0].weighted_total
                if self._scenario_result.decision_matrix.comparisons else 0.0, 2
            )
            outputs["scenario_count"] = len(self._scenario_result.validated_scenarios)
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Scenarios phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="scenarios",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: SDA Targets (Conditional)
    # -------------------------------------------------------------------------

    async def _phase_sda(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run SDA target workflow if sector is SDA-eligible."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        sector = config.sector.lower().strip()
        is_sda_eligible = sector in SDA_BENCHMARKS and not config.skip_sda

        if not is_sda_eligible:
            reason = "skip_sda flag set" if config.skip_sda else f"Sector '{sector}' not SDA-eligible"
            outputs["skipped_reason"] = reason
            warnings.append(f"SDA phase skipped: {reason}")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="sda_targets",
                status=PhaseStatus.SKIPPED,
                duration_seconds=round(elapsed, 4),
                outputs=outputs,
                warnings=warnings,
                provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            )

        try:
            sda_config = SDATargetConfig(
                sector=sector,
                base_year=config.base_year,
                base_year_intensity=config.base_year_intensity,
                base_year_activity=config.base_year_activity,
                growth_rate=config.activity_growth_rate,
                near_term_target_year=config.near_term_target_year,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = SDATargetWorkflow()
            self._sda_result = await wf.execute(sda_config)

            outputs["status"] = self._sda_result.status.value
            outputs["sda_valid"] = self._sda_result.validation.overall_valid
            if self._sda_result.near_term_target:
                outputs["near_term_intensity"] = self._sda_result.near_term_target.target_intensity
                outputs["near_term_reduction_pct"] = self._sda_result.near_term_target.reduction_from_base_pct
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("SDA phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="sda_targets",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Supplier Program
    # -------------------------------------------------------------------------

    async def _phase_suppliers(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run supplier engagement program workflow."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            sp_config = SupplierProgramConfig(
                suppliers=config.suppliers,
                scope3_baseline_tco2e=config.scope3_baseline_tco2e,
                coverage_target_pct=config.coverage_target_pct,
                engagement_budget_usd=config.engagement_budget_usd,
                base_year=config.base_year,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = SupplierProgramWorkflow()
            self._supplier_result = await wf.execute(sp_config)

            outputs["status"] = self._supplier_result.status.value
            outputs["coverage_pct"] = self._supplier_result.impact_report.coverage_pct
            outputs["reduction_tco2e"] = self._supplier_result.impact_report.estimated_reduction_tco2e
            outputs["reduction_pct"] = self._supplier_result.impact_report.estimated_reduction_pct
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Suppliers phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="suppliers",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Transition Finance
    # -------------------------------------------------------------------------

    async def _phase_finance(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run transition finance workflow."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            tf_config = TransitionFinanceConfig(
                capex_items=config.capex_items,
                carbon_price_usd=config.carbon_price_usd,
                discount_rate=config.discount_rate,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = TransitionFinanceWorkflow()
            self._finance_result = await wf.execute(tf_config)

            outputs["status"] = self._finance_result.status.value
            outputs["climate_capex_pct"] = self._finance_result.climate_capex_pct
            outputs["taxonomy_aligned_pct"] = self._finance_result.taxonomy_aligned_pct
            outputs["total_npv_usd"] = self._finance_result.total_npv_usd
            outputs["bond_eligible_usd"] = self._finance_result.bond_eligible_usd
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Finance phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="finance",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Advanced Progress
    # -------------------------------------------------------------------------

    async def _phase_progress(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run advanced progress tracking workflow."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            ap_config = AdvancedProgressConfig(
                annual_data=config.annual_data,
                target_pathway=config.target_pathway,
                base_year=config.base_year,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = AdvancedProgressWorkflow()
            self._progress_result = await wf.execute(ap_config)

            outputs["status"] = self._progress_result.status.value
            outputs["cumulative_reduction_pct"] = self._progress_result.cumulative_reduction_pct
            outputs["trend"] = self._progress_result.overall_trend.value
            outputs["alert_count"] = len(self._progress_result.alerts)
            outputs["critical_alerts"] = sum(
                1 for a in self._progress_result.alerts if a.severity.value == "critical"
            )
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Progress phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="progress",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 6: Temperature Alignment
    # -------------------------------------------------------------------------

    async def _phase_temperature(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run temperature alignment workflow."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            ta_config = TemperatureAlignmentConfig(
                entities=config.entity_targets,
                entity_weights=config.entity_weights,
                aggregation_method=config.aggregation_method,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = TemperatureAlignmentWorkflow()
            self._temp_result = await wf.execute(ta_config)

            outputs["status"] = self._temp_result.status.value
            if self._temp_result.primary_portfolio_score:
                outputs["portfolio_temp_c"] = self._temp_result.primary_portfolio_score.temperature_score_c
                outputs["paris_aligned"] = self._temp_result.primary_portfolio_score.is_paris_aligned
            outputs["paris_aligned_pct"] = self._temp_result.paris_aligned_pct
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Temperature phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="temperature",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 7: VCMI Certification (Conditional)
    # -------------------------------------------------------------------------

    async def _phase_vcmi(self, config: FullAccelerationConfig) -> PhaseResult:
        """Run VCMI certification workflow if credits exist."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        has_credits = len(config.credit_portfolio) > 0
        if not has_credits or config.skip_vcmi:
            reason = "skip_vcmi flag set" if config.skip_vcmi else "No carbon credits in portfolio"
            outputs["skipped_reason"] = reason
            warnings.append(f"VCMI phase skipped: {reason}")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="vcmi",
                status=PhaseStatus.SKIPPED,
                duration_seconds=round(elapsed, 4),
                outputs=outputs,
                warnings=warnings,
                provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            )

        try:
            vcmi_config = VCMICertificationConfig(
                has_sbti_target=config.has_sbti_target,
                sbti_validated=config.sbti_validated,
                reduction_progress_pct=config.reduction_progress_pct,
                credit_portfolio=config.credit_portfolio,
                residual_emissions_tco2e=config.residual_emissions_tco2e,
                target_tier=config.vcmi_target_tier,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )
            wf = VCMICertificationWorkflow()
            self._vcmi_result = await wf.execute(vcmi_config)

            outputs["status"] = self._vcmi_result.status.value
            outputs["eligible_tier"] = self._vcmi_result.claim_determination.eligible_tier.value
            outputs["total_score"] = self._vcmi_result.claim_determination.total_score
            outputs["greenwashing_risk"] = self._vcmi_result.claim_determination.greenwashing_risk.value
            status = PhaseStatus.COMPLETED

        except Exception as exc:
            self.logger.error("VCMI phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="vcmi",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 8: Strategy Compilation
    # -------------------------------------------------------------------------

    async def _phase_strategy(self, config: FullAccelerationConfig) -> PhaseResult:
        """Compile all outputs into unified acceleration strategy."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Build scorecard
        self._scorecard = self._build_scorecard()

        # Build strategy summary
        self._strategy = self._build_strategy_summary(config)

        outputs["acceleration_score"] = round(self._scorecard.overall_score, 2)
        outputs["maturity"] = self._scorecard.maturity.value
        outputs["key_actions"] = len(self._strategy.key_actions)
        outputs["key_risks"] = len(self._strategy.key_risks)
        outputs["paris_aligned"] = self._strategy.paris_aligned

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Strategy compiled: score=%.1f, maturity=%s",
                         self._scorecard.overall_score, self._scorecard.maturity.value)
        return PhaseResult(
            phase_name="strategy",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _build_scorecard(self) -> AccelerationScorecard:
        """Build acceleration scorecard from sub-workflow results."""
        # Scenario score (0-100): based on recommendation quality
        scenario_score = 0.0
        if self._scenario_result and self._scenario_result.decision_matrix.comparisons:
            best = self._scenario_result.decision_matrix.comparisons[0]
            scenario_score = min(best.weighted_total, 100.0)

        # Target score: SDA validation
        target_score = 0.0
        if self._sda_result:
            if self._sda_result.validation.overall_valid:
                target_score = 100.0
            else:
                target_score = 50.0 + min(self._sda_result.validation.pass_count * 10, 40.0)
        else:
            target_score = 30.0  # No SDA = lower score

        # Supplier score: based on coverage and reduction
        supplier_score = 0.0
        if self._supplier_result:
            coverage = self._supplier_result.impact_report.coverage_pct
            green_ratio = (
                self._supplier_result.impact_report.green_count /
                max(self._supplier_result.impact_report.total_suppliers_engaged, 1)
            ) * 100.0
            supplier_score = min(coverage * 0.6 + green_ratio * 0.4, 100.0)

        # Finance score: taxonomy alignment + positive NPV
        finance_score = 0.0
        if self._finance_result:
            tax_pct = self._finance_result.taxonomy_aligned_pct
            positive_npv = self._finance_result.total_npv_usd > 0
            finance_score = min(tax_pct * 0.7 + (30.0 if positive_npv else 0.0), 100.0)

        # Progress score: cumulative reduction + trend
        progress_score = 0.0
        if self._progress_result:
            cum = min(self._progress_result.cumulative_reduction_pct, 100.0)
            trend_bonus = 20.0 if self._progress_result.overall_trend.value == "decreasing" else 0.0
            alert_penalty = sum(
                10.0 for a in self._progress_result.alerts if a.severity.value == "critical"
            )
            progress_score = max(min(cum + trend_bonus - alert_penalty, 100.0), 0.0)

        # Temperature score: based on Paris alignment
        temp_score = 0.0
        if self._temp_result and self._temp_result.primary_portfolio_score:
            temp_c = self._temp_result.primary_portfolio_score.temperature_score_c
            # 1.5C = 100, 2.0C = 50, 3.2C = 0
            if temp_c <= 1.5:
                temp_score = 100.0
            elif temp_c <= 2.0:
                temp_score = 100.0 - (temp_c - 1.5) * 100.0
            elif temp_c <= 3.2:
                temp_score = max(50.0 - (temp_c - 2.0) * 41.67, 0.0)
            else:
                temp_score = 0.0

        # VCMI score: based on tier
        vcmi_score = 0.0
        if self._vcmi_result:
            tier_scores = {
                "platinum": 100.0,
                "gold": 80.0,
                "silver": 60.0,
                "not_eligible": 20.0,
            }
            vcmi_score = tier_scores.get(
                self._vcmi_result.claim_determination.eligible_tier.value, 0.0
            )

        # Overall (weighted average)
        weights = {
            "scenario": 0.10,
            "target": 0.20,
            "supplier": 0.15,
            "finance": 0.15,
            "progress": 0.20,
            "temperature": 0.10,
            "vcmi": 0.10,
        }
        overall = (
            scenario_score * weights["scenario"]
            + target_score * weights["target"]
            + supplier_score * weights["supplier"]
            + finance_score * weights["finance"]
            + progress_score * weights["progress"]
            + temp_score * weights["temperature"]
            + vcmi_score * weights["vcmi"]
        )

        maturity = self._determine_maturity(overall)

        return AccelerationScorecard(
            scenario_score=round(scenario_score, 2),
            target_score=round(target_score, 2),
            supplier_score=round(supplier_score, 2),
            finance_score=round(finance_score, 2),
            progress_score=round(progress_score, 2),
            temperature_score=round(temp_score, 2),
            vcmi_score=round(vcmi_score, 2),
            overall_score=round(overall, 2),
            maturity=maturity,
        )

    def _determine_maturity(self, score: float) -> AccelerationMaturity:
        """Map overall score to acceleration maturity level."""
        if score >= 85:
            return AccelerationMaturity.LEADING
        elif score >= 65:
            return AccelerationMaturity.ADVANCED
        elif score >= 45:
            return AccelerationMaturity.ACCELERATING
        elif score >= 25:
            return AccelerationMaturity.EMERGING
        else:
            return AccelerationMaturity.NASCENT

    def _build_strategy_summary(self, config: FullAccelerationConfig) -> AccelerationStrategySummary:
        """Build unified acceleration strategy summary."""
        # Scenario
        rec_scenario = ""
        rec_score = 0.0
        if self._scenario_result and self._scenario_result.decision_matrix.comparisons:
            rec_scenario = self._scenario_result.decision_matrix.recommended_scenario_name
            rec_score = self._scenario_result.decision_matrix.comparisons[0].weighted_total

        # SDA
        sda_sector = config.sector
        sda_eligible = self._sda_result is not None
        sda_nt_intensity = 0.0
        if self._sda_result and self._sda_result.near_term_target:
            sda_nt_intensity = self._sda_result.near_term_target.target_intensity

        # Suppliers
        supplier_coverage = 0.0
        supplier_reduction = 0.0
        if self._supplier_result:
            supplier_coverage = self._supplier_result.impact_report.coverage_pct
            supplier_reduction = self._supplier_result.impact_report.estimated_reduction_tco2e

        # Finance
        climate_capex_pct = 0.0
        taxonomy_pct = 0.0
        total_npv = 0.0
        if self._finance_result:
            climate_capex_pct = self._finance_result.climate_capex_pct
            taxonomy_pct = self._finance_result.taxonomy_aligned_pct
            total_npv = self._finance_result.total_npv_usd

        # Progress
        cum_reduction = 0.0
        forecast_trend = ""
        critical_alerts = 0
        if self._progress_result:
            cum_reduction = self._progress_result.cumulative_reduction_pct
            forecast_trend = self._progress_result.overall_trend.value
            critical_alerts = sum(
                1 for a in self._progress_result.alerts if a.severity.value == "critical"
            )

        # Temperature
        portfolio_temp = 0.0
        paris_aligned = False
        if self._temp_result and self._temp_result.primary_portfolio_score:
            portfolio_temp = self._temp_result.primary_portfolio_score.temperature_score_c
            paris_aligned = self._temp_result.primary_portfolio_score.is_paris_aligned

        # VCMI
        vcmi_tier = ""
        vcmi_eligible = False
        if self._vcmi_result:
            vcmi_tier = self._vcmi_result.claim_determination.eligible_tier.value
            vcmi_eligible = vcmi_tier != "not_eligible"

        # Key actions
        key_actions = self._compile_key_actions()

        # Key risks
        key_risks = self._compile_key_risks(config)

        return AccelerationStrategySummary(
            assessment_date=_utcnow().strftime("%Y-%m-%d"),
            recommended_scenario=rec_scenario,
            recommended_scenario_score=round(rec_score, 2),
            sda_sector=sda_sector,
            sda_eligible=sda_eligible,
            sda_near_term_intensity=round(sda_nt_intensity, 6),
            supplier_coverage_pct=round(supplier_coverage, 2),
            supplier_reduction_tco2e=round(supplier_reduction, 2),
            climate_capex_pct=round(climate_capex_pct, 2),
            taxonomy_aligned_pct=round(taxonomy_pct, 2),
            total_npv_usd=round(total_npv, 2),
            cumulative_reduction_pct=round(cum_reduction, 2),
            forecast_trend=forecast_trend,
            critical_alerts=critical_alerts,
            portfolio_temperature_c=round(portfolio_temp, 2),
            paris_aligned=paris_aligned,
            vcmi_tier=vcmi_tier,
            vcmi_eligible=vcmi_eligible,
            acceleration_score=self._scorecard.overall_score,
            maturity_level=self._scorecard.maturity.value,
            key_actions=key_actions,
            key_risks=key_risks,
        )

    def _compile_key_actions(self) -> List[str]:
        """Compile top key actions from all sub-workflows."""
        actions: List[str] = []

        if self._scenario_result and self._scenario_result.decision_matrix.recommended_scenario_name:
            actions.append(
                f"Adopt '{self._scenario_result.decision_matrix.recommended_scenario_name}' scenario "
                "as the primary transition pathway."
            )

        if self._sda_result and self._sda_result.near_term_target:
            actions.append(
                f"Implement SDA intensity target: {self._sda_result.near_term_target.target_intensity:.4f} "
                f"by {self._sda_result.near_term_target.year}."
            )

        if self._supplier_result:
            t1_count = sum(
                1 for ts in self._supplier_result.tier_summaries
                if ts.tier.value == "tier_1"
            )
            if t1_count > 0:
                actions.append(
                    f"Engage Tier 1 suppliers with deep decarbonisation programs."
                )

        if self._finance_result:
            positive_items = sum(
                1 for ic in self._finance_result.investment_cases
                if ic.npv_with_carbon_usd > 0
            )
            if positive_items > 0:
                actions.append(
                    f"Prioritise {positive_items} positive-NPV climate investments "
                    f"(total NPV: ${self._finance_result.total_npv_usd:,.0f})."
                )

        if self._progress_result and self._progress_result.alerts:
            crit = [a for a in self._progress_result.alerts if a.severity.value == "critical"]
            if crit:
                actions.append(
                    f"Address {len(crit)} critical deviation alert(s) with emergency corrective actions."
                )

        if self._vcmi_result:
            recs = self._vcmi_result.recommendations[:2]
            actions.extend(recs)

        return actions[:10]

    def _compile_key_risks(self, config: FullAccelerationConfig) -> List[str]:
        """Compile key risks from all sub-workflows."""
        risks: List[str] = []

        if self._progress_result:
            if self._progress_result.overall_trend.value == "increasing":
                risks.append(
                    "Emission trend is INCREASING. Current trajectory diverges from net-zero pathway."
                )

        if self._temp_result and self._temp_result.primary_portfolio_score:
            if not self._temp_result.primary_portfolio_score.is_paris_aligned:
                risks.append(
                    f"Portfolio temperature ({self._temp_result.primary_portfolio_score.temperature_score_c:.2f} deg C) "
                    "is not Paris-aligned. Improve entity-level targets."
                )

        if self._vcmi_result:
            if self._vcmi_result.claim_determination.greenwashing_risk.value == "high":
                risks.append(
                    "HIGH greenwashing risk identified. Address credibility gaps before public claims."
                )

        if self._supplier_result:
            red_count = self._supplier_result.impact_report.red_count
            if red_count > 0:
                risks.append(
                    f"Scope 3 risk: {red_count} suppliers with RED engagement status."
                )

        if self._finance_result and self._finance_result.total_npv_usd < 0:
            risks.append(
                "Transition finance risk: aggregate NPV is negative. "
                "Review cost assumptions and carbon price projections."
            )

        if not risks:
            risks.append("No critical risks identified. Continue monitoring all dimensions quarterly.")

        return risks[:8]
