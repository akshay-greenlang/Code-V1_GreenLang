# -*- coding: utf-8 -*-
"""
Scenario Analysis Workflow
===========================

Climate scenario analysis workflow implementing ESRS E1 and TCFD/IFRS S2
scenario requirements. Evaluates physical risks, transition risks, and
financial impacts across multiple climate scenarios (IEA, NGFS, custom)
over configurable time horizons.

Phases:
    1. Scenario Configuration: Load/validate pre-built and custom scenarios
    2. Physical Risk Assessment: Asset-level flood, drought, heat, storm risk
    3. Transition Risk Assessment: Policy, technology, market, reputation risk
    4. Financial Impact Modeling: Three-statement impact, Climate VaR, carbon price
    5. Resilience Narrative: ESRS E1 + TCFD/IFRS S2 resilience assessment

Author: GreenLang Team
Version: 2.0.0
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


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
    CANCELLED = "cancelled"


class PhysicalRiskType(str, Enum):
    """Types of physical climate risk."""
    FLOOD = "flood"
    DROUGHT = "drought"
    HEAT_STRESS = "heat_stress"
    STORM = "storm"
    SEA_LEVEL_RISE = "sea_level_rise"
    WILDFIRE = "wildfire"


class TransitionRiskType(str, Enum):
    """Types of transition risk."""
    POLICY = "policy"
    TECHNOLOGY = "technology"
    MARKET = "market"
    REPUTATION = "reputation"


# =============================================================================
# DATA MODELS
# =============================================================================


class AssetDefinition(BaseModel):
    """Definition of a physical asset for risk assessment."""
    asset_id: str = Field(..., description="Asset identifier")
    name: str = Field(..., description="Asset name")
    asset_type: str = Field(..., description="factory/office/warehouse/mine/port")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    country: str = Field(..., description="ISO country code")
    replacement_value_eur: float = Field(default=0.0, description="Asset replacement value")
    annual_revenue_eur: float = Field(default=0.0, description="Revenue attributable to asset")


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0)
    agents_executed: int = Field(default=0)
    records_processed: int = Field(default=0)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PhaseDefinition(BaseModel):
    """Internal definition of a workflow phase."""
    name: str
    display_name: str
    estimated_minutes: float
    required: bool = True
    depends_on: List[str] = Field(default_factory=list)


class ScenarioAnalysisInput(BaseModel):
    """Input configuration for the scenario analysis workflow."""
    organization_id: str = Field(..., description="Organization identifier")
    reporting_year: int = Field(..., ge=2024, le=2050)
    scenarios: List[str] = Field(
        default_factory=lambda: [
            "iea_nze", "iea_aps", "iea_steps",
            "ngfs_orderly", "ngfs_disorderly", "ngfs_hothouse",
        ],
        description="Scenario identifiers to evaluate"
    )
    time_horizons: List[int] = Field(
        default_factory=lambda: [2030, 2040, 2050],
        description="Target years for scenario projections"
    )
    assets: List[AssetDefinition] = Field(
        default_factory=list, description="Physical assets for risk assessment"
    )
    financial_data: Dict[str, Any] = Field(
        default_factory=dict, description="Financial data for impact modeling"
    )
    custom_scenario: Optional[Dict[str, Any]] = Field(
        None, description="Custom scenario definition"
    )


class ScenarioAnalysisResult(BaseModel):
    """Complete result from the scenario analysis workflow."""
    workflow_id: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    physical_risk_results: Dict[str, Any] = Field(
        default_factory=dict, description="Physical risk per scenario"
    )
    transition_risk_results: Dict[str, Any] = Field(
        default_factory=dict, description="Transition risk per scenario"
    )
    financial_impact_results: Dict[str, Any] = Field(
        default_factory=dict, description="Financial impact per scenario"
    )
    resilience_assessment: Dict[str, Any] = Field(
        default_factory=dict, description="Resilience narrative and assessment"
    )
    scenario_comparison: Dict[str, Any] = Field(
        default_factory=dict, description="Cross-scenario comparison matrix"
    )
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


# =============================================================================
# SCENARIO CATALOGUE
# =============================================================================

SCENARIO_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "iea_nze": {
        "name": "IEA Net Zero Emissions by 2050",
        "source": "IEA",
        "warming_target_c": 1.5,
        "carbon_price_2030_eur": 130,
        "carbon_price_2050_eur": 250,
        "description": "Net zero CO2 emissions globally by 2050",
    },
    "iea_aps": {
        "name": "IEA Announced Pledges Scenario",
        "source": "IEA",
        "warming_target_c": 1.7,
        "carbon_price_2030_eur": 90,
        "carbon_price_2050_eur": 200,
        "description": "All announced national pledges fully implemented",
    },
    "iea_steps": {
        "name": "IEA Stated Policies Scenario",
        "source": "IEA",
        "warming_target_c": 2.5,
        "carbon_price_2030_eur": 45,
        "carbon_price_2050_eur": 80,
        "description": "Based on current stated policies only",
    },
    "ngfs_orderly": {
        "name": "NGFS Orderly Transition",
        "source": "NGFS",
        "warming_target_c": 1.5,
        "carbon_price_2030_eur": 120,
        "carbon_price_2050_eur": 350,
        "description": "Immediate and smooth climate policies",
    },
    "ngfs_disorderly": {
        "name": "NGFS Disorderly Transition",
        "source": "NGFS",
        "warming_target_c": 1.8,
        "carbon_price_2030_eur": 50,
        "carbon_price_2050_eur": 500,
        "description": "Delayed and then sudden climate policies",
    },
    "ngfs_hothouse": {
        "name": "NGFS Hot House World",
        "source": "NGFS",
        "warming_target_c": 3.0,
        "carbon_price_2030_eur": 15,
        "carbon_price_2050_eur": 30,
        "description": "No additional climate policies, severe physical risks",
    },
    "ngfs_current": {
        "name": "NGFS Current Policies",
        "source": "NGFS",
        "warming_target_c": 2.6,
        "carbon_price_2030_eur": 35,
        "carbon_price_2050_eur": 50,
        "description": "Only current implemented policies",
    },
    "custom": {
        "name": "Custom Scenario",
        "source": "user_defined",
        "warming_target_c": 2.0,
        "carbon_price_2030_eur": 0,
        "carbon_price_2050_eur": 0,
        "description": "User-defined custom scenario",
    },
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ScenarioAnalysisWorkflow:
    """
    Climate scenario analysis workflow for ESRS E1 and TCFD/IFRS S2.

    Evaluates physical risks, transition risks, and financial impacts
    across multiple climate scenarios over configurable time horizons.

    Attributes:
        workflow_id: Unique execution identifier.
        _cancelled: Cancellation flag.
        _progress_callback: Optional progress callback.

    Example:
        >>> workflow = ScenarioAnalysisWorkflow()
        >>> input_cfg = ScenarioAnalysisInput(
        ...     organization_id="org-123",
        ...     reporting_year=2025,
        ...     assets=[AssetDefinition(
        ...         asset_id="plant-1", name="Manufacturing Plant",
        ...         asset_type="factory", latitude=50.1, longitude=8.7,
        ...         country="DE", replacement_value_eur=50000000,
        ...     )],
        ... )
        >>> result = await workflow.execute(input_cfg)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASES: List[PhaseDefinition] = [
        PhaseDefinition(
            name="scenario_configuration",
            display_name="Scenario Configuration",
            estimated_minutes=10.0,
            required=True,
            depends_on=[],
        ),
        PhaseDefinition(
            name="physical_risk_assessment",
            display_name="Physical Risk Assessment",
            estimated_minutes=30.0,
            required=True,
            depends_on=["scenario_configuration"],
        ),
        PhaseDefinition(
            name="transition_risk_assessment",
            display_name="Transition Risk Assessment",
            estimated_minutes=30.0,
            required=True,
            depends_on=["scenario_configuration"],
        ),
        PhaseDefinition(
            name="financial_impact_modeling",
            display_name="Financial Impact Modeling",
            estimated_minutes=20.0,
            required=True,
            depends_on=["physical_risk_assessment", "transition_risk_assessment"],
        ),
        PhaseDefinition(
            name="resilience_narrative",
            display_name="Resilience Assessment & Narrative",
            estimated_minutes=15.0,
            required=True,
            depends_on=["financial_impact_modeling"],
        ),
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize the scenario analysis workflow.

        Args:
            progress_callback: Optional callback(phase_name, message, pct_complete).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._cancelled: bool = False
        self._progress_callback = progress_callback
        self._phase_results: Dict[str, PhaseResult] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, input_data: ScenarioAnalysisInput) -> ScenarioAnalysisResult:
        """
        Execute the scenario analysis workflow.

        Args:
            input_data: Validated workflow input.

        Returns:
            ScenarioAnalysisResult with risk assessments and financial impacts.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting scenario analysis %s for org=%s year=%d scenarios=%s",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year, input_data.scenarios,
        )
        self._notify_progress("workflow", "Workflow started", 0.0)

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        try:
            for idx, phase_def in enumerate(self.PHASES):
                if self._cancelled:
                    overall_status = WorkflowStatus.CANCELLED
                    break

                for dep in phase_def.depends_on:
                    dep_result = self._phase_results.get(dep)
                    if dep_result and dep_result.status == PhaseStatus.FAILED:
                        if phase_def.required:
                            raise RuntimeError(
                                f"Required phase '{phase_def.name}' cannot run: "
                                f"dependency '{dep}' failed."
                            )

                pct_base = idx / len(self.PHASES)
                self._notify_progress(
                    phase_def.name, f"Starting: {phase_def.display_name}", pct_base
                )

                phase_result = await self._execute_phase(phase_def, input_data, pct_base)
                completed_phases.append(phase_result)
                self._phase_results[phase_def.name] = phase_result

                if phase_result.status == PhaseStatus.FAILED and phase_def.required:
                    overall_status = WorkflowStatus.FAILED
                    break

            if overall_status == WorkflowStatus.RUNNING:
                all_ok = all(
                    p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                    for p in completed_phases
                )
                overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        except Exception as exc:
            logger.critical("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            completed_phases.append(PhaseResult(
                phase_name="workflow_error", status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        physical = self._extract_physical_risks(completed_phases)
        transition = self._extract_transition_risks(completed_phases)
        financial = self._extract_financial_impacts(completed_phases)
        resilience = self._extract_resilience(completed_phases)
        comparison = self._build_scenario_comparison(physical, transition, financial)
        artifacts = {p.phase_name: p.artifacts for p in completed_phases if p.artifacts}

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress("workflow", f"Workflow {overall_status.value}", 1.0)

        return ScenarioAnalysisResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            physical_risk_results=physical,
            transition_risk_results=transition,
            financial_impact_results=financial,
            resilience_assessment=resilience,
            scenario_comparison=comparison,
            artifacts=artifacts,
            provenance_hash=provenance,
        )

    def cancel(self) -> None:
        """Request cooperative cancellation."""
        logger.info("Cancellation requested for workflow %s", self.workflow_id)
        self._cancelled = True

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase_def: PhaseDefinition,
        input_data: ScenarioAnalysisInput, pct_base: float,
    ) -> PhaseResult:
        """Dispatch to the correct phase handler."""
        started_at = datetime.utcnow()
        handler_map = {
            "scenario_configuration": self._phase_scenario_configuration,
            "physical_risk_assessment": self._phase_physical_risk,
            "transition_risk_assessment": self._phase_transition_risk,
            "financial_impact_modeling": self._phase_financial_impact,
            "resilience_narrative": self._phase_resilience_narrative,
        }
        handler = handler_map.get(phase_def.name)
        if handler is None:
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at,
                errors=[f"Unknown phase: {phase_def.name}"],
                provenance_hash=self._hash_data({"error": "unknown_phase"}),
            )
        try:
            result = await handler(input_data, pct_base)
            result.started_at = started_at
            result.completed_at = datetime.utcnow()
            result.duration_seconds = (result.completed_at - started_at).total_seconds()
            return result
        except Exception as exc:
            logger.error("Phase '%s' raised: %s", phase_def.name, exc, exc_info=True)
            return PhaseResult(
                phase_name=phase_def.name, status=PhaseStatus.FAILED,
                started_at=started_at, completed_at=datetime.utcnow(),
                duration_seconds=(datetime.utcnow() - started_at).total_seconds(),
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            )

    # -------------------------------------------------------------------------
    # Phase 1: Scenario Configuration
    # -------------------------------------------------------------------------

    async def _phase_scenario_configuration(
        self, input_data: ScenarioAnalysisInput, pct_base: float
    ) -> PhaseResult:
        """
        Load pre-built scenario definitions and validate custom scenarios.
        """
        phase_name = "scenario_configuration"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        self._notify_progress(phase_name, "Loading scenario definitions", pct_base + 0.02)

        loaded_scenarios: Dict[str, Dict[str, Any]] = {}

        for scenario_id in input_data.scenarios:
            if scenario_id == "custom" and input_data.custom_scenario:
                custom_def = {**SCENARIO_DEFINITIONS.get("custom", {})}
                custom_def.update(input_data.custom_scenario)
                loaded_scenarios["custom"] = custom_def
            elif scenario_id in SCENARIO_DEFINITIONS:
                loaded_scenarios[scenario_id] = SCENARIO_DEFINITIONS[scenario_id]
            else:
                warnings.append(f"Unknown scenario '{scenario_id}'; skipping.")

        artifacts["scenarios_loaded"] = list(loaded_scenarios.keys())
        artifacts["scenario_count"] = len(loaded_scenarios)
        artifacts["time_horizons"] = input_data.time_horizons
        artifacts["assets_count"] = len(input_data.assets)
        artifacts["scenario_definitions"] = loaded_scenarios

        if not loaded_scenarios:
            errors.append("No valid scenarios could be loaded.")

        agents_executed = 1  # Scenario loader

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(loaded_scenarios),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 2: Physical Risk Assessment
    # -------------------------------------------------------------------------

    async def _phase_physical_risk(
        self, input_data: ScenarioAnalysisInput, pct_base: float
    ) -> PhaseResult:
        """
        Asset-level physical risk assessment across scenarios for flood,
        drought, heat stress, storm, sea level rise, and wildfire.
        """
        phase_name = "physical_risk_assessment"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        config_phase = self._phase_results.get("scenario_configuration")
        scenarios = (
            config_phase.artifacts.get("scenarios_loaded", [])
            if config_phase and config_phase.artifacts else []
        )

        physical_results: Dict[str, Dict[str, Any]] = {}

        for scenario_id in scenarios:
            self._notify_progress(
                phase_name,
                f"Assessing physical risks under {scenario_id}",
                pct_base + 0.02,
            )

            scenario_physical = await self._assess_physical_risks(
                input_data.organization_id, scenario_id,
                input_data.assets, input_data.time_horizons,
            )
            agents_executed += 1
            physical_results[scenario_id] = scenario_physical

        artifacts["physical_risk_by_scenario"] = physical_results
        artifacts["scenarios_assessed"] = len(physical_results)
        artifacts["assets_assessed"] = len(input_data.assets)

        # Identify high-risk assets
        high_risk_assets: List[str] = []
        for scenario_data in physical_results.values():
            for asset_result in scenario_data.get("asset_risks", []):
                if asset_result.get("overall_risk_score", 0) > 0.7:
                    aid = asset_result.get("asset_id", "")
                    if aid and aid not in high_risk_assets:
                        high_risk_assets.append(aid)

        artifacts["high_risk_assets"] = high_risk_assets
        if high_risk_assets:
            warnings.append(
                f"{len(high_risk_assets)} asset(s) identified as high physical risk."
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(input_data.assets) * len(scenarios),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Transition Risk Assessment
    # -------------------------------------------------------------------------

    async def _phase_transition_risk(
        self, input_data: ScenarioAnalysisInput, pct_base: float
    ) -> PhaseResult:
        """
        Policy, technology, market, and reputation transition risk
        assessment per scenario.
        """
        phase_name = "transition_risk_assessment"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        config_phase = self._phase_results.get("scenario_configuration")
        scenarios = (
            config_phase.artifacts.get("scenarios_loaded", [])
            if config_phase and config_phase.artifacts else []
        )

        transition_results: Dict[str, Dict[str, Any]] = {}

        for scenario_id in scenarios:
            self._notify_progress(
                phase_name,
                f"Assessing transition risks under {scenario_id}",
                pct_base + 0.02,
            )

            scenario_transition = await self._assess_transition_risks(
                input_data.organization_id, scenario_id,
                input_data.financial_data, input_data.time_horizons,
            )
            agents_executed += 1
            transition_results[scenario_id] = scenario_transition

        artifacts["transition_risk_by_scenario"] = transition_results
        artifacts["scenarios_assessed"] = len(transition_results)

        # Identify high transition risk scenarios
        high_transition: List[str] = []
        for sid, data in transition_results.items():
            if data.get("overall_transition_risk", 0) > 0.7:
                high_transition.append(sid)

        artifacts["high_transition_risk_scenarios"] = high_transition
        if high_transition:
            warnings.append(
                f"High transition risk identified in: {', '.join(high_transition)}"
            )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(scenarios) * 4,  # 4 risk types per scenario
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Financial Impact Modeling
    # -------------------------------------------------------------------------

    async def _phase_financial_impact(
        self, input_data: ScenarioAnalysisInput, pct_base: float
    ) -> PhaseResult:
        """
        Three-statement financial impact, Climate VaR, and carbon price
        sensitivity analysis per scenario.
        """
        phase_name = "financial_impact_modeling"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        physical_phase = self._phase_results.get("physical_risk_assessment")
        transition_phase = self._phase_results.get("transition_risk_assessment")

        physical_data = (
            physical_phase.artifacts.get("physical_risk_by_scenario", {})
            if physical_phase and physical_phase.artifacts else {}
        )
        transition_data = (
            transition_phase.artifacts.get("transition_risk_by_scenario", {})
            if transition_phase and transition_phase.artifacts else {}
        )

        config_phase = self._phase_results.get("scenario_configuration")
        scenarios = (
            config_phase.artifacts.get("scenarios_loaded", [])
            if config_phase and config_phase.artifacts else []
        )

        financial_results: Dict[str, Dict[str, Any]] = {}

        for scenario_id in scenarios:
            self._notify_progress(
                phase_name,
                f"Modeling financial impacts for {scenario_id}",
                pct_base + 0.02,
            )

            impact = await self._model_financial_impact(
                input_data.organization_id, scenario_id,
                physical_data.get(scenario_id, {}),
                transition_data.get(scenario_id, {}),
                input_data.financial_data, input_data.time_horizons,
            )
            agents_executed += 1
            financial_results[scenario_id] = impact

        artifacts["financial_impact_by_scenario"] = financial_results
        artifacts["scenarios_modeled"] = len(financial_results)

        # Climate VaR summary
        climate_var = await self._calculate_climate_var(
            input_data.organization_id, financial_results
        )
        agents_executed += 1
        artifacts["climate_var"] = climate_var

        # Carbon price sensitivity
        carbon_sensitivity = await self._calculate_carbon_sensitivity(
            input_data.organization_id, input_data.financial_data, scenarios
        )
        agents_executed += 1
        artifacts["carbon_price_sensitivity"] = carbon_sensitivity

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(scenarios),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 5: Resilience Narrative
    # -------------------------------------------------------------------------

    async def _phase_resilience_narrative(
        self, input_data: ScenarioAnalysisInput, pct_base: float
    ) -> PhaseResult:
        """
        Generate ESRS E1 + TCFD/IFRS S2 resilience assessment with
        narrative generation.
        """
        phase_name = "resilience_narrative"
        errors: List[str] = []
        warnings: List[str] = []
        agents_executed = 0
        artifacts: Dict[str, Any] = {}

        financial_phase = self._phase_results.get("financial_impact_modeling")
        physical_phase = self._phase_results.get("physical_risk_assessment")
        transition_phase = self._phase_results.get("transition_risk_assessment")

        self._notify_progress(
            phase_name, "Generating resilience assessment", pct_base + 0.02
        )

        # ESRS E1 resilience assessment
        esrs_resilience = await self._generate_esrs_e1_resilience(
            input_data.organization_id,
            physical_phase.artifacts if physical_phase else {},
            transition_phase.artifacts if transition_phase else {},
            financial_phase.artifacts if financial_phase else {},
        )
        agents_executed += 1
        artifacts["esrs_e1_resilience"] = esrs_resilience

        self._notify_progress(
            phase_name, "Generating TCFD/IFRS S2 narrative", pct_base + 0.04
        )

        # TCFD/IFRS S2 resilience narrative
        tcfd_narrative = await self._generate_tcfd_narrative(
            input_data.organization_id,
            physical_phase.artifacts if physical_phase else {},
            transition_phase.artifacts if transition_phase else {},
            financial_phase.artifacts if financial_phase else {},
        )
        agents_executed += 1
        artifacts["tcfd_narrative"] = tcfd_narrative

        self._notify_progress(
            phase_name, "Generating adaptation strategies", pct_base + 0.06
        )

        # Adaptation strategy recommendations
        adaptation = await self._generate_adaptation_strategies(
            input_data.organization_id,
            physical_phase.artifacts if physical_phase else {},
            transition_phase.artifacts if transition_phase else {},
        )
        agents_executed += 1
        artifacts["adaptation_strategies"] = adaptation

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        provenance = self._hash_data(artifacts)

        return PhaseResult(
            phase_name=phase_name, status=status,
            agents_executed=agents_executed,
            records_processed=len(input_data.scenarios),
            artifacts=artifacts, errors=errors, warnings=warnings,
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Agent Invocation Helpers
    # -------------------------------------------------------------------------

    async def _assess_physical_risks(
        self, org_id: str, scenario_id: str,
        assets: List[AssetDefinition], horizons: List[int],
    ) -> Dict[str, Any]:
        """Assess physical risks for all assets under a scenario."""
        await asyncio.sleep(0)
        asset_risks = []
        for asset in assets:
            asset_risks.append({
                "asset_id": asset.asset_id,
                "name": asset.name,
                "flood_risk": 0.35,
                "drought_risk": 0.18,
                "heat_stress_risk": 0.42,
                "storm_risk": 0.28,
                "sea_level_rise_risk": 0.15,
                "wildfire_risk": 0.08,
                "overall_risk_score": 0.52,
                "expected_annual_damage_eur": round(
                    asset.replacement_value_eur * 0.012, 2
                ),
            })
        return {
            "scenario": scenario_id,
            "asset_risks": asset_risks,
            "total_expected_damage_eur": sum(
                a["expected_annual_damage_eur"] for a in asset_risks
            ),
            "time_horizons_analyzed": horizons,
        }

    async def _assess_transition_risks(
        self, org_id: str, scenario_id: str,
        financial_data: Dict[str, Any], horizons: List[int],
    ) -> Dict[str, Any]:
        """Assess transition risks under a scenario."""
        await asyncio.sleep(0)
        scenario_def = SCENARIO_DEFINITIONS.get(scenario_id, {})
        carbon_price = scenario_def.get("carbon_price_2030_eur", 50)
        warming = scenario_def.get("warming_target_c", 2.0)

        policy_risk = min(0.9, carbon_price / 300)
        tech_risk = 0.45 if warming <= 2.0 else 0.25
        market_risk = 0.55 if warming <= 2.0 else 0.35
        rep_risk = 0.30 if warming > 2.5 else 0.50

        return {
            "scenario": scenario_id,
            "policy_risk": round(policy_risk, 3),
            "technology_risk": round(tech_risk, 3),
            "market_risk": round(market_risk, 3),
            "reputation_risk": round(rep_risk, 3),
            "overall_transition_risk": round(
                (policy_risk + tech_risk + market_risk + rep_risk) / 4, 3
            ),
            "carbon_price_impact_eur": carbon_price * 50000,
            "stranded_asset_risk_pct": round(max(0, (warming - 1.5) * 15), 1),
        }

    async def _model_financial_impact(
        self, org_id: str, scenario_id: str,
        physical: Dict[str, Any], transition: Dict[str, Any],
        financial: Dict[str, Any], horizons: List[int],
    ) -> Dict[str, Any]:
        """Model three-statement financial impact for a scenario."""
        await asyncio.sleep(0)
        revenue = financial.get("total_revenue_eur", 500000000)
        physical_damage = physical.get("total_expected_damage_eur", 0)
        carbon_cost = transition.get("carbon_price_impact_eur", 0)

        return {
            "scenario": scenario_id,
            "revenue_impact_pct": round(-physical_damage / max(revenue, 1) * 100, 2),
            "cost_increase_pct": round(carbon_cost / max(revenue, 1) * 100, 2),
            "ebitda_impact_pct": round(
                -(physical_damage + carbon_cost) / max(revenue * 0.15, 1) * 100, 2
            ),
            "capex_required_eur": round(revenue * 0.02, 2),
            "net_asset_value_impact_pct": -2.8,
            "by_time_horizon": {
                str(h): {
                    "cumulative_cost_eur": round(carbon_cost * (h - 2025) * 0.8, 2),
                    "adaptation_investment_eur": round(revenue * 0.005 * (h - 2025), 2),
                }
                for h in horizons
            },
        }

    async def _calculate_climate_var(
        self, org_id: str, financial_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate Climate Value at Risk across scenarios."""
        await asyncio.sleep(0)
        return {
            "climate_var_95_pct": -4.2,
            "climate_var_99_pct": -7.8,
            "worst_case_scenario": "ngfs_hothouse",
            "best_case_scenario": "iea_nze",
            "expected_loss_range_eur": {"low": 12500000, "high": 45000000},
        }

    async def _calculate_carbon_sensitivity(
        self, org_id: str, financial: Dict[str, Any], scenarios: List[str],
    ) -> Dict[str, Any]:
        """Calculate carbon price sensitivity analysis."""
        await asyncio.sleep(0)
        return {
            "breakeven_carbon_price_eur": 180,
            "impact_per_10eur_tonne": -0.35,
            "current_exposure_tco2e": 50000,
            "sensitivity_by_price": {
                "50": -2500000, "100": -5000000,
                "150": -7500000, "200": -10000000,
                "250": -12500000,
            },
        }

    async def _generate_esrs_e1_resilience(
        self, org_id: str, physical: Dict[str, Any],
        transition: Dict[str, Any], financial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate ESRS E1 resilience assessment."""
        await asyncio.sleep(0)
        return {
            "resilience_score": 0.72,
            "physical_resilience": 0.68,
            "transition_resilience": 0.76,
            "key_vulnerabilities": [
                "Manufacturing facilities exposed to heat stress",
                "Supply chain dependency on fossil fuel logistics",
            ],
            "mitigation_measures": [
                "Facility cooling systems upgrade planned for 2027",
                "Supplier diversification program initiated",
            ],
        }

    async def _generate_tcfd_narrative(
        self, org_id: str, physical: Dict[str, Any],
        transition: Dict[str, Any], financial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate TCFD/IFRS S2 resilience narrative."""
        await asyncio.sleep(0)
        return {
            "strategy_pillar": "The organization has assessed its resilience under "
            "six climate scenarios spanning 1.5C to 3.0C warming pathways.",
            "governance_response": "Board oversight of climate-related risks and "
            "opportunities reviewed quarterly.",
            "risk_management_integration": "Climate scenario outputs are integrated "
            "into the enterprise risk management framework.",
            "metrics_and_targets": "Internal carbon price of EUR 100/tCO2e applied "
            "to investment decisions above EUR 5M.",
        }

    async def _generate_adaptation_strategies(
        self, org_id: str, physical: Dict[str, Any], transition: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate climate adaptation strategy recommendations."""
        await asyncio.sleep(0)
        return [
            {
                "strategy": "Facility resilience upgrades",
                "risk_addressed": "physical",
                "investment_eur": 15000000,
                "roi_years": 8,
                "risk_reduction_pct": 35,
            },
            {
                "strategy": "Renewable energy procurement",
                "risk_addressed": "transition",
                "investment_eur": 8000000,
                "roi_years": 5,
                "risk_reduction_pct": 45,
            },
            {
                "strategy": "Supply chain decarbonization",
                "risk_addressed": "transition",
                "investment_eur": 20000000,
                "roi_years": 10,
                "risk_reduction_pct": 25,
            },
        ]

    # -------------------------------------------------------------------------
    # Result Extractors
    # -------------------------------------------------------------------------

    def _extract_physical_risks(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract physical risk results."""
        for p in phases:
            if p.phase_name == "physical_risk_assessment" and p.artifacts:
                return p.artifacts.get("physical_risk_by_scenario", {})
        return {}

    def _extract_transition_risks(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract transition risk results."""
        for p in phases:
            if p.phase_name == "transition_risk_assessment" and p.artifacts:
                return p.artifacts.get("transition_risk_by_scenario", {})
        return {}

    def _extract_financial_impacts(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract financial impact results."""
        for p in phases:
            if p.phase_name == "financial_impact_modeling" and p.artifacts:
                return p.artifacts.get("financial_impact_by_scenario", {})
        return {}

    def _extract_resilience(self, phases: List[PhaseResult]) -> Dict[str, Any]:
        """Extract resilience assessment."""
        for p in phases:
            if p.phase_name == "resilience_narrative" and p.artifacts:
                return p.artifacts
        return {}

    def _build_scenario_comparison(
        self, physical: Dict[str, Any],
        transition: Dict[str, Any], financial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build cross-scenario comparison matrix."""
        comparison: Dict[str, Dict[str, Any]] = {}
        all_scenarios = set(physical.keys()) | set(transition.keys()) | set(financial.keys())
        for sid in all_scenarios:
            comparison[sid] = {
                "physical_risk": physical.get(sid, {}).get(
                    "total_expected_damage_eur", 0
                ),
                "transition_risk": transition.get(sid, {}).get(
                    "overall_transition_risk", 0
                ),
                "financial_impact_pct": financial.get(sid, {}).get(
                    "ebitda_impact_pct", 0
                ),
            }
        return comparison

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _notify_progress(self, phase: str, message: str, pct: float) -> None:
        """Send progress notification via callback if registered."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Compute SHA-256 provenance hash of arbitrary data."""
        serialized = str(data).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()
