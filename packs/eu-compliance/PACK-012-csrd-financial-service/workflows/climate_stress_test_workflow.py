# -*- coding: utf-8 -*-
"""
Climate Stress Test Workflow
================================

Five-phase workflow for climate stress testing financial portfolios under
ECB/EBA climate stress test frameworks and TCFD recommendations.

Phases:
    1. ExposureMapping - Map portfolio exposures to climate-sensitive sectors
    2. ScenarioSelection - Select NGFS/ECB climate scenarios
    3. RiskCalculation - Calculate physical and transition risk impacts
    4. ImpactQuantification - Quantify P&L, capital, and credit impacts
    5. ReportGeneration - Generate stress test results report

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

class PhaseStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"

class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(..., description="Organization identifier")
    execution_timestamp: datetime = Field(default_factory=utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED

class PhaseResult(BaseModel):
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)

class WorkflowResult(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(..., description="Workflow type identifier")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")

class ClimateScenario(str, Enum):
    """NGFS climate scenarios."""
    NET_ZERO_2050 = "NET_ZERO_2050"
    BELOW_2C = "BELOW_2C"
    DELAYED_TRANSITION = "DELAYED_TRANSITION"
    CURRENT_POLICIES = "CURRENT_POLICIES"
    NDC = "NDC"
    DIVERGENT_NET_ZERO = "DIVERGENT_NET_ZERO"

class RiskType(str, Enum):
    """Climate risk types."""
    PHYSICAL_ACUTE = "PHYSICAL_ACUTE"
    PHYSICAL_CHRONIC = "PHYSICAL_CHRONIC"
    TRANSITION_POLICY = "TRANSITION_POLICY"
    TRANSITION_TECHNOLOGY = "TRANSITION_TECHNOLOGY"
    TRANSITION_MARKET = "TRANSITION_MARKET"
    TRANSITION_REPUTATION = "TRANSITION_REPUTATION"

# ---------------------------------------------------------------------------
#  Input / Result Models
# ---------------------------------------------------------------------------

class PortfolioExposure(BaseModel):
    """Portfolio exposure for stress testing."""
    exposure_id: str = Field(..., description="Exposure identifier")
    counterparty_name: str = Field(default="")
    sector: str = Field(default="", description="NACE sector")
    country: str = Field(default="")
    exposure_amount: float = Field(..., ge=0.0, description="Exposure in EUR")
    maturity_years: float = Field(default=5.0, ge=0.0)
    collateral_type: str = Field(default="unsecured")
    probability_of_default: float = Field(default=0.01, ge=0.0, le=1.0)
    loss_given_default: float = Field(default=0.45, ge=0.0, le=1.0)
    carbon_intensity: Optional[float] = Field(None, ge=0.0, description="tCO2e/EUR M revenue")
    physical_risk_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    transition_risk_score: Optional[float] = Field(None, ge=0.0, le=10.0)

class ClimateStressTestInput(BaseModel):
    """Input for the climate stress test workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_date: str = Field(..., description="Reporting date YYYY-MM-DD")
    exposures: List[PortfolioExposure] = Field(default_factory=list)
    scenarios: List[str] = Field(
        default_factory=lambda: ["NET_ZERO_2050", "DELAYED_TRANSITION", "CURRENT_POLICIES"]
    )
    time_horizons: List[int] = Field(default_factory=lambda: [2030, 2040, 2050])
    carbon_price_assumptions: Dict[str, float] = Field(default_factory=dict)
    skip_phases: List[str] = Field(default_factory=list)

    @field_validator("reporting_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        datetime.strptime(v, "%Y-%m-%d")
        return v

class ClimateStressTestResult(WorkflowResult):
    """Result from the climate stress test workflow."""
    total_exposure_eur: float = Field(default=0.0)
    scenarios_tested: int = Field(default=0)
    max_credit_loss_pct: float = Field(default=0.0)
    max_market_value_impact_pct: float = Field(default=0.0)
    high_risk_exposure_pct: float = Field(default=0.0)
    physical_risk_exposure_pct: float = Field(default=0.0)
    transition_risk_exposure_pct: float = Field(default=0.0)
    counterparties_assessed: int = Field(default=0)
    sectors_assessed: int = Field(default=0)

# ---------------------------------------------------------------------------
#  Phase 1: Exposure Mapping
# ---------------------------------------------------------------------------

class ExposureMappingPhase:
    """Map portfolio exposures to climate-sensitive sectors."""

    PHASE_NAME = "exposure_mapping"

    CLIMATE_SENSITIVE_SECTORS = [
        "B", "C19", "C20", "C23", "C24", "D35", "H49", "H50", "H51",
        "A01", "A02", "F41", "F42", "L68",
    ]

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            exposures = config.get("exposures", [])

            total_exposure = sum(e.get("exposure_amount", 0.0) for e in exposures)
            outputs["total_exposure_eur"] = total_exposure

            climate_sensitive = []
            non_sensitive = []

            for exp in exposures:
                sector = exp.get("sector", "")
                is_sensitive = any(sector.startswith(s) for s in self.CLIMATE_SENSITIVE_SECTORS)
                if is_sensitive:
                    climate_sensitive.append({**exp, "climate_sensitive": True})
                else:
                    non_sensitive.append({**exp, "climate_sensitive": False})

            outputs["climate_sensitive_exposures"] = climate_sensitive
            outputs["non_sensitive_exposures"] = non_sensitive
            outputs["climate_sensitive_count"] = len(climate_sensitive)
            outputs["climate_sensitive_amount"] = sum(
                e.get("exposure_amount", 0.0) for e in climate_sensitive
            )
            outputs["climate_sensitive_pct"] = round(
                outputs["climate_sensitive_amount"] / max(total_exposure, 1.0) * 100, 2
            )

            # By sector aggregation
            by_sector: Dict[str, float] = {}
            for e in exposures:
                sec = e.get("sector", "UNKNOWN") or "UNKNOWN"
                by_sector[sec] = by_sector.get(sec, 0.0) + e.get("exposure_amount", 0.0)
            outputs["by_sector"] = {k: round(v, 2) for k, v in by_sector.items()}
            outputs["sectors_count"] = len(by_sector)

            status = PhaseStatus.COMPLETED
            records = len(exposures)

        except Exception as exc:
            logger.error("ExposureMapping failed: %s", exc, exc_info=True)
            errors.append(f"Exposure mapping failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs), records_processed=records,
        )

# ---------------------------------------------------------------------------
#  Phase 2: Scenario Selection
# ---------------------------------------------------------------------------

class ScenarioSelectionPhase:
    """Select and configure NGFS/ECB climate scenarios."""

    PHASE_NAME = "scenario_selection"

    SCENARIO_PARAMS = {
        "NET_ZERO_2050": {
            "temperature_target": 1.5,
            "carbon_price_2030": 130, "carbon_price_2050": 250,
            "transition_speed": "orderly", "physical_risk_level": "low",
        },
        "BELOW_2C": {
            "temperature_target": 1.8,
            "carbon_price_2030": 80, "carbon_price_2050": 200,
            "transition_speed": "orderly", "physical_risk_level": "low",
        },
        "DELAYED_TRANSITION": {
            "temperature_target": 1.8,
            "carbon_price_2030": 30, "carbon_price_2050": 350,
            "transition_speed": "disorderly", "physical_risk_level": "medium",
        },
        "CURRENT_POLICIES": {
            "temperature_target": 3.0,
            "carbon_price_2030": 15, "carbon_price_2050": 25,
            "transition_speed": "none", "physical_risk_level": "high",
        },
        "NDC": {
            "temperature_target": 2.5,
            "carbon_price_2030": 40, "carbon_price_2050": 80,
            "transition_speed": "moderate", "physical_risk_level": "medium",
        },
        "DIVERGENT_NET_ZERO": {
            "temperature_target": 1.5,
            "carbon_price_2030": 100, "carbon_price_2050": 300,
            "transition_speed": "disorderly", "physical_risk_level": "low",
        },
    }

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            requested = config.get("scenarios", ["NET_ZERO_2050", "CURRENT_POLICIES"])
            horizons = config.get("time_horizons", [2030, 2050])
            custom_carbon = config.get("carbon_price_assumptions", {})

            selected = []
            for scenario_name in requested:
                params = self.SCENARIO_PARAMS.get(scenario_name)
                if params:
                    s = {**params, "scenario_name": scenario_name}
                    # Apply custom carbon prices
                    for key, val in custom_carbon.items():
                        if key in s:
                            s[key] = val
                    selected.append(s)
                else:
                    warnings.append(f"Unknown scenario: {scenario_name}")

            outputs["selected_scenarios"] = selected
            outputs["scenarios_count"] = len(selected)
            outputs["time_horizons"] = horizons

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ScenarioSelection failed: %s", exc, exc_info=True)
            errors.append(f"Scenario selection failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 3: Risk Calculation
# ---------------------------------------------------------------------------

class RiskCalculationPhase:
    """Calculate physical and transition risk impacts per scenario."""

    PHASE_NAME = "risk_calculation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            mapping = context.get_phase_output("exposure_mapping")
            scenarios = context.get_phase_output("scenario_selection")
            sensitive = mapping.get("climate_sensitive_exposures", [])
            all_scenarios = scenarios.get("selected_scenarios", [])

            risk_results = []

            for scenario in all_scenarios:
                scenario_name = scenario.get("scenario_name", "")
                phys_level = scenario.get("physical_risk_level", "medium")
                carbon_2030 = scenario.get("carbon_price_2030", 50)
                trans_speed = scenario.get("transition_speed", "moderate")

                physical_multiplier = {"low": 0.01, "medium": 0.03, "high": 0.06}.get(phys_level, 0.03)
                transition_multiplier = {"none": 0.0, "moderate": 0.02, "orderly": 0.03,
                                          "disorderly": 0.05}.get(trans_speed, 0.02)

                exposure_risks = []
                for exp in sensitive:
                    exposure_amt = exp.get("exposure_amount", 0.0)
                    ci = exp.get("carbon_intensity", 100.0) or 100.0
                    phys_score = exp.get("physical_risk_score", 5.0) or 5.0
                    trans_score = exp.get("transition_risk_score", 5.0) or 5.0

                    phys_impact = exposure_amt * physical_multiplier * (phys_score / 10.0)
                    trans_impact = exposure_amt * transition_multiplier * (ci / 100.0) * (trans_score / 10.0)
                    total_impact = phys_impact + trans_impact

                    exposure_risks.append({
                        "exposure_id": exp.get("exposure_id", ""),
                        "sector": exp.get("sector", ""),
                        "physical_impact_eur": round(phys_impact, 2),
                        "transition_impact_eur": round(trans_impact, 2),
                        "total_impact_eur": round(total_impact, 2),
                    })

                total_phys = sum(r["physical_impact_eur"] for r in exposure_risks)
                total_trans = sum(r["transition_impact_eur"] for r in exposure_risks)

                risk_results.append({
                    "scenario": scenario_name,
                    "total_physical_impact_eur": round(total_phys, 2),
                    "total_transition_impact_eur": round(total_trans, 2),
                    "total_impact_eur": round(total_phys + total_trans, 2),
                    "exposure_details": exposure_risks,
                })

            outputs["risk_results"] = risk_results
            outputs["scenarios_calculated"] = len(risk_results)

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("RiskCalculation failed: %s", exc, exc_info=True)
            errors.append(f"Risk calculation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 4: Impact Quantification
# ---------------------------------------------------------------------------

class ImpactQuantificationPhase:
    """Quantify P&L, capital, and credit impacts from climate risks."""

    PHASE_NAME = "impact_quantification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            mapping = context.get_phase_output("exposure_mapping")
            risk_calc = context.get_phase_output("risk_calculation")
            total_exposure = mapping.get("total_exposure_eur", 0.0)
            risk_results = risk_calc.get("risk_results", [])

            scenario_impacts = []
            max_credit_loss_pct = 0.0
            max_mv_impact_pct = 0.0

            for result in risk_results:
                total_impact = result.get("total_impact_eur", 0.0)
                credit_loss_pct = round(total_impact / max(total_exposure, 1.0) * 100, 4)
                mv_impact_pct = round(total_impact / max(total_exposure, 1.0) * 100 * 0.7, 4)

                max_credit_loss_pct = max(max_credit_loss_pct, credit_loss_pct)
                max_mv_impact_pct = max(max_mv_impact_pct, mv_impact_pct)

                scenario_impacts.append({
                    "scenario": result.get("scenario", ""),
                    "total_impact_eur": total_impact,
                    "credit_loss_pct": credit_loss_pct,
                    "market_value_impact_pct": mv_impact_pct,
                    "physical_impact_eur": result.get("total_physical_impact_eur", 0.0),
                    "transition_impact_eur": result.get("total_transition_impact_eur", 0.0),
                })

            outputs["scenario_impacts"] = scenario_impacts
            outputs["max_credit_loss_pct"] = round(max_credit_loss_pct, 4)
            outputs["max_market_value_impact_pct"] = round(max_mv_impact_pct, 4)
            outputs["total_exposure_eur"] = total_exposure

            # High risk exposure
            sensitive_amt = mapping.get("climate_sensitive_amount", 0.0)
            outputs["high_risk_exposure_pct"] = round(
                sensitive_amt / max(total_exposure, 1.0) * 100, 2
            )

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("ImpactQuantification failed: %s", exc, exc_info=True)
            errors.append(f"Impact quantification failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Phase 5: Report Generation
# ---------------------------------------------------------------------------

class StressTestReportPhase:
    """Generate climate stress test results report."""

    PHASE_NAME = "report_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        started_at = utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            mapping = context.get_phase_output("exposure_mapping")
            scenarios = context.get_phase_output("scenario_selection")
            impact = context.get_phase_output("impact_quantification")

            outputs["report"] = {
                "reporting_date": config.get("reporting_date", ""),
                "total_exposure_eur": mapping.get("total_exposure_eur", 0.0),
                "scenarios_tested": scenarios.get("scenarios_count", 0),
                "max_credit_loss_pct": impact.get("max_credit_loss_pct", 0.0),
                "max_market_value_impact_pct": impact.get("max_market_value_impact_pct", 0.0),
                "high_risk_exposure_pct": impact.get("high_risk_exposure_pct", 0.0),
                "scenario_impacts": impact.get("scenario_impacts", []),
                "sectors_assessed": mapping.get("sectors_count", 0),
                "counterparties_assessed": len(mapping.get("climate_sensitive_exposures", [])),
            }

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("StressTestReport failed: %s", exc, exc_info=True)
            errors.append(f"Report generation failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME, status=status,
            started_at=started_at, completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs, errors=errors, warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )

# ---------------------------------------------------------------------------
#  Workflow Orchestrator
# ---------------------------------------------------------------------------

class ClimateStressTestWorkflow:
    """Five-phase climate stress test workflow for ECB/EBA compliance."""

    WORKFLOW_NAME = "climate_stress_test"
    PHASE_ORDER = ["exposure_mapping", "scenario_selection",
                    "risk_calculation", "impact_quantification", "report_generation"]

    def __init__(self, progress_callback: Optional[Callable[[str, str, float], None]] = None) -> None:
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "exposure_mapping": ExposureMappingPhase(),
            "scenario_selection": ScenarioSelectionPhase(),
            "risk_calculation": RiskCalculationPhase(),
            "impact_quantification": ImpactQuantificationPhase(),
            "report_generation": StressTestReportPhase(),
        }

    async def run(self, input_data: ClimateStressTestInput) -> ClimateStressTestResult:
        """Execute the complete 5-phase climate stress test workflow."""
        started_at = utcnow()
        logger.info("Starting climate stress test workflow %s org=%s",
                     self.workflow_id, input_data.organization_id)
        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=input_data.model_dump(),
        )
        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                ))
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue
            if context.is_phase_completed(phase_name):
                continue
            self._notify_progress(phase_name, f"Starting: {phase_name}", idx / len(self.PHASE_ORDER))
            context.mark_phase(phase_name, PhaseStatus.RUNNING)
            try:
                result = await self._phases[phase_name].execute(context)
                completed_phases.append(result)
                if result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, result.status)
                    if phase_name == "exposure_mapping":
                        overall_status = WorkflowStatus.FAILED
                        break
                context.errors.extend(result.errors)
                context.warnings.extend(result.warnings)
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                completed_phases.append(PhaseResult(
                    phase_name=phase_name, status=PhaseStatus.FAILED,
                    started_at=utcnow(), errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                ))
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                         for p in completed_phases)
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = utcnow()
        duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })
        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return ClimateStressTestResult(
            workflow_id=self.workflow_id, workflow_name=self.WORKFLOW_NAME,
            status=overall_status, started_at=started_at,
            completed_at=completed_at, total_duration_seconds=duration,
            phases=completed_phases, summary=summary, provenance_hash=provenance,
            total_exposure_eur=summary.get("total_exposure_eur", 0.0),
            scenarios_tested=summary.get("scenarios_tested", 0),
            max_credit_loss_pct=summary.get("max_credit_loss_pct", 0.0),
            max_market_value_impact_pct=summary.get("max_market_value_impact_pct", 0.0),
            high_risk_exposure_pct=summary.get("high_risk_exposure_pct", 0.0),
            physical_risk_exposure_pct=summary.get("physical_risk_exposure_pct", 0.0),
            transition_risk_exposure_pct=summary.get("transition_risk_exposure_pct", 0.0),
            counterparties_assessed=summary.get("counterparties_assessed", 0),
            sectors_assessed=summary.get("sectors_assessed", 0),
        )

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        mapping = context.get_phase_output("exposure_mapping")
        impact = context.get_phase_output("impact_quantification")
        report = context.get_phase_output("report_generation")
        rep = report.get("report", {})
        sens_pct = mapping.get("climate_sensitive_pct", 0.0)
        return {
            "total_exposure_eur": rep.get("total_exposure_eur", 0.0),
            "scenarios_tested": rep.get("scenarios_tested", 0),
            "max_credit_loss_pct": impact.get("max_credit_loss_pct", 0.0),
            "max_market_value_impact_pct": impact.get("max_market_value_impact_pct", 0.0),
            "high_risk_exposure_pct": rep.get("high_risk_exposure_pct", 0.0),
            "physical_risk_exposure_pct": round(sens_pct * 0.6, 2),
            "transition_risk_exposure_pct": round(sens_pct * 0.8, 2),
            "counterparties_assessed": rep.get("counterparties_assessed", 0),
            "sectors_assessed": rep.get("sectors_assessed", 0),
        }

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)
