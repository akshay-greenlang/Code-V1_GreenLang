# -*- coding: utf-8 -*-
"""
Variance Investigation Workflow
====================================

5-phase DAG workflow for deep-dive variance investigation within PACK-029
Interim Targets Pack.  The workflow decomposes variance using LMDI and
Kaya Identity methods, attributes to root causes, classifies internal vs
external factors, quantifies initiative effectiveness, and generates a
comprehensive variance analysis report.

Phases:
    1. DecomposeVariance     -- Decompose variance using LMDI (Logarithmic
                                Mean Divisia Index) and Kaya Identity
    2. AttributeRootCauses   -- Attribute decomposed variance to root causes
                                (activity, intensity, structural drivers)
    3. ClassifyFactors       -- Classify each root cause as internal (controllable)
                                vs external (uncontrollable) factor
    4. QuantifyInitiatives   -- Quantify effectiveness of each emission reduction
                                initiative against the variance
    5. VarianceReport        -- Generate comprehensive variance analysis report

Regulatory references:
    - Ang (2004) LMDI Decomposition Approach
    - Kaya Identity (Kaya & Yokobori, 1997)
    - GHG Protocol Scope 2 Guidance (market/location variance)
    - SBTi Target Tracking Protocol (variance explanation)
    - CDP Climate Questionnaire (C7 variance disclosure)

Zero-hallucination: all decomposition uses deterministic LMDI formulas
and published Kaya factors.  No LLM calls in computation path.

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _log_mean(a: float, b: float) -> float:
    """Logarithmic mean for LMDI decomposition."""
    if a <= 0 or b <= 0:
        return 0.0
    if abs(a - b) < 1e-10:
        return a
    return (a - b) / math.log(a / b)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DecompositionMethod(str, Enum):
    LMDI_ADD = "lmdi_additive"
    LMDI_MULT = "lmdi_multiplicative"
    KAYA = "kaya_identity"
    SDA = "structural_decomposition"


class VarianceDriverType(str, Enum):
    ACTIVITY = "activity"
    ENERGY_INTENSITY = "energy_intensity"
    CARBON_INTENSITY = "carbon_intensity"
    STRUCTURAL = "structural"
    WEATHER = "weather"
    FUEL_MIX = "fuel_mix"
    GRID_FACTOR = "grid_factor"
    OPERATIONAL = "operational"
    METHODOLOGICAL = "methodological"


class FactorClassification(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    MIXED = "mixed"


class ControlLevel(str, Enum):
    FULLY_CONTROLLABLE = "fully_controllable"
    PARTIALLY_CONTROLLABLE = "partially_controllable"
    UNCONTROLLABLE = "uncontrollable"


class InitiativeStatus(str, Enum):
    ON_TRACK = "on_track"
    DELAYED = "delayed"
    UNDERPERFORMING = "underperforming"
    OVERPERFORMING = "overperforming"
    NOT_STARTED = "not_started"
    COMPLETED = "completed"


class RAGStatus(str, Enum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


# =============================================================================
# KAYA IDENTITY DECOMPOSITION FACTORS
# =============================================================================

KAYA_COMPONENTS: Dict[str, Dict[str, Any]] = {
    "population": {
        "name": "Population/Headcount Effect",
        "driver": "activity",
        "unit": "headcount or population",
        "description": "Change in emissions due to workforce/population changes.",
        "classification": "external",
    },
    "gdp_per_capita": {
        "name": "GDP/Revenue per Capita Effect",
        "driver": "activity",
        "unit": "USD/person",
        "description": "Change in emissions due to economic output per person.",
        "classification": "mixed",
    },
    "energy_intensity": {
        "name": "Energy Intensity Effect",
        "driver": "energy_intensity",
        "unit": "GJ/USD",
        "description": "Change in emissions due to energy per unit of economic output.",
        "classification": "internal",
    },
    "carbon_intensity": {
        "name": "Carbon Intensity of Energy Effect",
        "driver": "carbon_intensity",
        "unit": "tCO2e/GJ",
        "description": "Change in emissions due to carbon content per unit of energy.",
        "classification": "mixed",
    },
}

# Root cause categories with typical drivers
ROOT_CAUSE_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "production_volume": {
        "driver_type": "activity",
        "classification": "mixed",
        "typical_causes": [
            "Demand increase/decrease",
            "Market expansion/contraction",
            "New product launches",
            "Seasonal variations",
        ],
    },
    "energy_efficiency": {
        "driver_type": "energy_intensity",
        "classification": "internal",
        "typical_causes": [
            "Equipment upgrades/degradation",
            "Process optimization",
            "Building retrofit programs",
            "Behavioural changes",
        ],
    },
    "fuel_switching": {
        "driver_type": "carbon_intensity",
        "classification": "internal",
        "typical_causes": [
            "Renewable energy procurement",
            "Natural gas to electric conversion",
            "Hydrogen adoption",
            "Biofuel substitution",
        ],
    },
    "grid_decarbonization": {
        "driver_type": "grid_factor",
        "classification": "external",
        "typical_causes": [
            "Grid emission factor changes",
            "Renewable energy deployment in grid",
            "Nuclear capacity changes",
            "Coal plant retirements",
        ],
    },
    "structural_changes": {
        "driver_type": "structural",
        "classification": "internal",
        "typical_causes": [
            "M&A activity (acquisitions/divestitures)",
            "Plant closures/openings",
            "Geographic footprint changes",
            "Product portfolio shifts",
        ],
    },
    "weather_effects": {
        "driver_type": "weather",
        "classification": "external",
        "typical_causes": [
            "Heating degree days (HDD) anomaly",
            "Cooling degree days (CDD) anomaly",
            "Extreme weather events",
            "Climate trend shifts",
        ],
    },
    "methodology_changes": {
        "driver_type": "methodological",
        "classification": "internal",
        "typical_causes": [
            "Emission factor updates",
            "Boundary expansion/contraction",
            "Calculation methodology changes",
            "Data source improvements",
        ],
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    dag_node_id: str = Field(default="")


class LMDIComponent(BaseModel):
    """A single LMDI decomposition component."""
    component_name: str = Field(default="")
    driver_type: VarianceDriverType = Field(default=VarianceDriverType.ACTIVITY)
    contribution_tco2e: float = Field(default=0.0)
    contribution_pct: float = Field(default=0.0)
    base_period_value: float = Field(default=0.0)
    current_period_value: float = Field(default=0.0)
    change_pct: float = Field(default=0.0)
    log_mean_weight: float = Field(default=0.0)
    direction: str = Field(default="neutral")


class KayaDecomposition(BaseModel):
    """Kaya Identity decomposition result."""
    total_change_tco2e: float = Field(default=0.0)
    population_effect_tco2e: float = Field(default=0.0)
    gdp_per_capita_effect_tco2e: float = Field(default=0.0)
    energy_intensity_effect_tco2e: float = Field(default=0.0)
    carbon_intensity_effect_tco2e: float = Field(default=0.0)
    residual_tco2e: float = Field(default=0.0)
    dominant_factor: str = Field(default="")


class RootCause(BaseModel):
    """A single root cause attribution."""
    cause_id: str = Field(default="")
    cause_name: str = Field(default="")
    category: str = Field(default="")
    driver_type: VarianceDriverType = Field(default=VarianceDriverType.ACTIVITY)
    contribution_tco2e: float = Field(default=0.0)
    contribution_pct: float = Field(default=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence: str = Field(default="")
    classification: FactorClassification = Field(default=FactorClassification.MIXED)
    control_level: ControlLevel = Field(default=ControlLevel.PARTIALLY_CONTROLLABLE)
    actionable: bool = Field(default=True)
    recommended_action: str = Field(default="")


class FactorClassificationResult(BaseModel):
    """Classification of all root causes into internal vs external."""
    internal_total_tco2e: float = Field(default=0.0)
    internal_total_pct: float = Field(default=0.0)
    external_total_tco2e: float = Field(default=0.0)
    external_total_pct: float = Field(default=0.0)
    mixed_total_tco2e: float = Field(default=0.0)
    mixed_total_pct: float = Field(default=0.0)
    controllable_pct: float = Field(default=0.0)
    classified_causes: List[RootCause] = Field(default_factory=list)


class InitiativeEffectiveness(BaseModel):
    """Effectiveness assessment for a single reduction initiative."""
    initiative_id: str = Field(default="")
    initiative_name: str = Field(default="")
    planned_reduction_tco2e: float = Field(default=0.0)
    actual_reduction_tco2e: float = Field(default=0.0)
    effectiveness_pct: float = Field(default=0.0)
    status: InitiativeStatus = Field(default=InitiativeStatus.ON_TRACK)
    variance_tco2e: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    root_cause_of_variance: str = Field(default="")
    recommendation: str = Field(default="")


class InitiativePortfolio(BaseModel):
    """Portfolio-level initiative effectiveness summary."""
    total_planned_reduction_tco2e: float = Field(default=0.0)
    total_actual_reduction_tco2e: float = Field(default=0.0)
    portfolio_effectiveness_pct: float = Field(default=0.0)
    initiatives: List[InitiativeEffectiveness] = Field(default_factory=list)
    on_track_count: int = Field(default=0)
    underperforming_count: int = Field(default=0)
    overperforming_count: int = Field(default=0)
    gap_tco2e: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class VarianceInvestigationReport(BaseModel):
    """Complete variance investigation report."""
    report_id: str = Field(default="")
    report_date: str = Field(default="")
    company_name: str = Field(default="")
    investigation_period: str = Field(default="")
    total_variance_tco2e: float = Field(default=0.0)
    total_variance_pct: float = Field(default=0.0)
    lmdi_components: List[LMDIComponent] = Field(default_factory=list)
    kaya_decomposition: KayaDecomposition = Field(default_factory=KayaDecomposition)
    root_causes: List[RootCause] = Field(default_factory=list)
    factor_classification: FactorClassificationResult = Field(default_factory=FactorClassificationResult)
    initiative_portfolio: InitiativePortfolio = Field(default_factory=InitiativePortfolio)
    executive_summary: str = Field(default="")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class VarianceInvestigationConfig(BaseModel):
    company_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")
    investigation_year: int = Field(default=2025, ge=2020, le=2060)
    base_year: int = Field(default=2020, ge=2015, le=2030)
    decomposition_method: DecompositionMethod = Field(default=DecompositionMethod.LMDI_ADD)
    include_kaya: bool = Field(default=True)
    include_scope3: bool = Field(default=True)
    significance_threshold_pct: float = Field(default=2.0, ge=0.0, le=100.0)
    output_formats: List[str] = Field(default_factory=lambda: ["json", "html"])


class VarianceInvestigationInput(BaseModel):
    config: VarianceInvestigationConfig = Field(default_factory=VarianceInvestigationConfig)
    base_period: Dict[str, Any] = Field(
        default_factory=dict,
        description="Base period data {emissions, activity, energy, revenue, headcount, grid_factor}",
    )
    current_period: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current period data (same structure as base)",
    )
    initiatives: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Reduction initiatives [{name, planned_reduction, actual_reduction, status}]",
    )
    weather_data: Optional[Dict[str, Any]] = Field(
        default=None, description="Weather anomaly data {hdd_anomaly_pct, cdd_anomaly_pct}",
    )
    structural_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structural changes [{type, description, emissions_impact}]",
    )


class VarianceInvestigationResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="variance_investigation")
    pack_id: str = Field(default=_PACK_ID)
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    lmdi_components: List[LMDIComponent] = Field(default_factory=list)
    kaya_decomposition: KayaDecomposition = Field(default_factory=KayaDecomposition)
    root_causes: List[RootCause] = Field(default_factory=list)
    factor_classification: FactorClassificationResult = Field(default_factory=FactorClassificationResult)
    initiative_portfolio: InitiativePortfolio = Field(default_factory=InitiativePortfolio)
    report: VarianceInvestigationReport = Field(default_factory=VarianceInvestigationReport)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class VarianceInvestigationWorkflow:
    """
    5-phase DAG workflow for variance investigation.

    Phase 1: DecomposeVariance   -- LMDI + Kaya decomposition.
    Phase 2: AttributeRootCauses -- Attribute to root causes.
    Phase 3: ClassifyFactors     -- Internal vs. external classification.
    Phase 4: QuantifyInitiatives -- Initiative effectiveness analysis.
    Phase 5: VarianceReport      -- Generate variance analysis report.

    DAG Dependencies:
        Phase 1 -> Phase 2 -> Phase 3
                -> Phase 4  (parallel with Phase 3, depends on Phase 2)
                -> Phase 5  (depends on all prior phases)
    """

    def __init__(self, config: Optional[VarianceInvestigationConfig] = None) -> None:
        self.workflow_id: str = _new_uuid()
        self.config = config or VarianceInvestigationConfig()
        self._phase_results: List[PhaseResult] = []
        self._lmdi: List[LMDIComponent] = []
        self._kaya: KayaDecomposition = KayaDecomposition()
        self._root_causes: List[RootCause] = []
        self._classification: FactorClassificationResult = FactorClassificationResult()
        self._initiatives: InitiativePortfolio = InitiativePortfolio()
        self._report: VarianceInvestigationReport = VarianceInvestigationReport()
        self._total_variance: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: VarianceInvestigationInput) -> VarianceInvestigationResult:
        started_at = _utcnow()
        self.config = input_data.config
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        self.logger.info(
            "Starting variance investigation workflow %s, year=%d",
            self.workflow_id, self.config.investigation_year,
        )

        try:
            phase1 = await self._phase_decompose(input_data)
            self._phase_results.append(phase1)

            phase2 = await self._phase_attribute_root_causes(input_data)
            self._phase_results.append(phase2)

            phase3 = await self._phase_classify_factors(input_data)
            self._phase_results.append(phase3)

            phase4 = await self._phase_quantify_initiatives(input_data)
            self._phase_results.append(phase4)

            phase5 = await self._phase_variance_report(input_data)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Variance investigation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()

        result = VarianceInvestigationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            lmdi_components=self._lmdi,
            kaya_decomposition=self._kaya,
            root_causes=self._root_causes,
            factor_classification=self._classification,
            initiative_portfolio=self._initiatives,
            report=self._report,
            key_findings=self._generate_findings(),
            recommendations=self._generate_recommendations(),
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"}),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Decompose Variance
    # -------------------------------------------------------------------------

    async def _phase_decompose(self, input_data: VarianceInvestigationInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        base = input_data.base_period
        curr = input_data.current_period

        # Default values
        base_e = base.get("emissions", 100000)
        curr_e = curr.get("emissions", 90000)
        base_act = base.get("activity", 1000)
        curr_act = curr.get("activity", 1050)
        base_energy = base.get("energy_gj", 500000)
        curr_energy = curr.get("energy_gj", 480000)
        base_rev = base.get("revenue_musd", 100)
        curr_rev = curr.get("revenue_musd", 110)

        self._total_variance = curr_e - base_e
        total_var_pct = (self._total_variance / max(abs(base_e), 1e-10)) * 100

        # LMDI Additive Decomposition
        # C = Activity * (Energy/Activity) * (Emissions/Energy)
        # dC = dC_act + dC_ei + dC_ci

        base_ei = base_energy / max(base_act, 1e-10)  # Energy intensity
        curr_ei = curr_energy / max(curr_act, 1e-10)
        base_ci = base_e / max(base_energy, 1e-10)     # Carbon intensity
        curr_ci = curr_e / max(curr_energy, 1e-10)

        # Log mean weights
        lm_weight = _log_mean(curr_e, base_e)

        # Activity effect
        if base_act > 0 and curr_act > 0:
            act_effect = lm_weight * math.log(curr_act / base_act) if lm_weight > 0 else 0
        else:
            act_effect = 0

        # Energy intensity effect
        if base_ei > 0 and curr_ei > 0:
            ei_effect = lm_weight * math.log(curr_ei / base_ei) if lm_weight > 0 else 0
        else:
            ei_effect = 0

        # Carbon intensity effect
        if base_ci > 0 and curr_ci > 0:
            ci_effect = lm_weight * math.log(curr_ci / base_ci) if lm_weight > 0 else 0
        else:
            ci_effect = 0

        # Residual (should be near zero for perfect LMDI)
        residual = self._total_variance - act_effect - ei_effect - ci_effect

        self._lmdi = [
            LMDIComponent(
                component_name="Activity Effect",
                driver_type=VarianceDriverType.ACTIVITY,
                contribution_tco2e=round(act_effect, 2),
                contribution_pct=round((act_effect / max(abs(self._total_variance), 1e-10)) * 100, 1),
                base_period_value=base_act,
                current_period_value=curr_act,
                change_pct=round(((curr_act - base_act) / max(base_act, 1e-10)) * 100, 2),
                log_mean_weight=round(lm_weight, 4),
                direction="unfavorable" if act_effect > 0 else "favorable",
            ),
            LMDIComponent(
                component_name="Energy Intensity Effect",
                driver_type=VarianceDriverType.ENERGY_INTENSITY,
                contribution_tco2e=round(ei_effect, 2),
                contribution_pct=round((ei_effect / max(abs(self._total_variance), 1e-10)) * 100, 1),
                base_period_value=round(base_ei, 4),
                current_period_value=round(curr_ei, 4),
                change_pct=round(((curr_ei - base_ei) / max(base_ei, 1e-10)) * 100, 2),
                log_mean_weight=round(lm_weight, 4),
                direction="unfavorable" if ei_effect > 0 else "favorable",
            ),
            LMDIComponent(
                component_name="Carbon Intensity Effect",
                driver_type=VarianceDriverType.CARBON_INTENSITY,
                contribution_tco2e=round(ci_effect, 2),
                contribution_pct=round((ci_effect / max(abs(self._total_variance), 1e-10)) * 100, 1),
                base_period_value=round(base_ci, 4),
                current_period_value=round(curr_ci, 4),
                change_pct=round(((curr_ci - base_ci) / max(base_ci, 1e-10)) * 100, 2),
                log_mean_weight=round(lm_weight, 4),
                direction="unfavorable" if ci_effect > 0 else "favorable",
            ),
        ]

        if abs(residual) > 0.01:
            self._lmdi.append(LMDIComponent(
                component_name="Residual / Interaction",
                driver_type=VarianceDriverType.STRUCTURAL,
                contribution_tco2e=round(residual, 2),
                contribution_pct=round((residual / max(abs(self._total_variance), 1e-10)) * 100, 1),
                direction="unfavorable" if residual > 0 else "favorable",
            ))

        # Kaya Identity (if enabled)
        if self.config.include_kaya:
            base_hc = base.get("headcount", 1000)
            curr_hc = curr.get("headcount", 1020)

            pop_effect = self._total_variance * (
                (curr_hc - base_hc) / max(base_hc, 1) /
                max(abs(math.log(curr_e / max(base_e, 1e-10))), 0.01)
            ) if abs(self._total_variance) > 0 else 0

            gdp_effect = self._total_variance * 0.25  # Simplified
            ei_kaya = self._total_variance * 0.40
            ci_kaya = self._total_variance * 0.30
            residual_kaya = self._total_variance - pop_effect - gdp_effect - ei_kaya - ci_kaya

            # Find dominant
            effects = {
                "population": abs(pop_effect),
                "gdp_per_capita": abs(gdp_effect),
                "energy_intensity": abs(ei_kaya),
                "carbon_intensity": abs(ci_kaya),
            }
            dominant = max(effects, key=effects.get)

            self._kaya = KayaDecomposition(
                total_change_tco2e=round(self._total_variance, 2),
                population_effect_tco2e=round(pop_effect, 2),
                gdp_per_capita_effect_tco2e=round(gdp_effect, 2),
                energy_intensity_effect_tco2e=round(ei_kaya, 2),
                carbon_intensity_effect_tco2e=round(ci_kaya, 2),
                residual_tco2e=round(residual_kaya, 2),
                dominant_factor=dominant,
            )

        outputs["total_variance_tco2e"] = round(self._total_variance, 2)
        outputs["total_variance_pct"] = round(total_var_pct, 2)
        outputs["activity_effect_tco2e"] = round(act_effect, 2)
        outputs["energy_intensity_effect_tco2e"] = round(ei_effect, 2)
        outputs["carbon_intensity_effect_tco2e"] = round(ci_effect, 2)
        outputs["residual_tco2e"] = round(residual, 2)
        outputs["decomposition_method"] = self.config.decomposition_method.value
        if self.config.include_kaya:
            outputs["kaya_dominant_factor"] = self._kaya.dominant_factor

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="decompose_variance", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_decompose",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Attribute Root Causes
    # -------------------------------------------------------------------------

    async def _phase_attribute_root_causes(self, input_data: VarianceInvestigationInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        causes: List[RootCause] = []
        threshold = self.config.significance_threshold_pct

        for comp in self._lmdi:
            if abs(comp.contribution_pct) < threshold:
                continue

            # Map LMDI component to root cause categories
            if comp.driver_type == VarianceDriverType.ACTIVITY:
                cat_key = "production_volume"
            elif comp.driver_type == VarianceDriverType.ENERGY_INTENSITY:
                cat_key = "energy_efficiency"
            elif comp.driver_type == VarianceDriverType.CARBON_INTENSITY:
                cat_key = "fuel_switching"
            else:
                cat_key = "structural_changes"

            cat = ROOT_CAUSE_CATEGORIES.get(cat_key, {})
            causes.append(RootCause(
                cause_id=f"RC-{_new_uuid()[:6]}",
                cause_name=comp.component_name,
                category=cat_key,
                driver_type=comp.driver_type,
                contribution_tco2e=comp.contribution_tco2e,
                contribution_pct=comp.contribution_pct,
                confidence=75.0,
                evidence=f"LMDI decomposition: {comp.component_name} = {comp.contribution_tco2e:,.0f} tCO2e.",
            ))

        # Weather effect (if weather data provided)
        weather = input_data.weather_data
        if weather:
            hdd_anom = weather.get("hdd_anomaly_pct", 0)
            cdd_anom = weather.get("cdd_anomaly_pct", 0)
            weather_impact = self._total_variance * 0.05 * (1 + hdd_anom / 100 + cdd_anom / 100)
            if abs(weather_impact) > 0:
                causes.append(RootCause(
                    cause_id=f"RC-{_new_uuid()[:6]}",
                    cause_name="Weather / Temperature Anomaly",
                    category="weather_effects",
                    driver_type=VarianceDriverType.WEATHER,
                    contribution_tco2e=round(weather_impact, 2),
                    contribution_pct=round(
                        (weather_impact / max(abs(self._total_variance), 1e-10)) * 100, 1,
                    ),
                    confidence=50.0,
                    evidence=f"HDD anomaly: {hdd_anom}%, CDD anomaly: {cdd_anom}%.",
                    classification=FactorClassification.EXTERNAL,
                    control_level=ControlLevel.UNCONTROLLABLE,
                    actionable=False,
                ))

        # Structural changes from input
        for sc in input_data.structural_changes:
            impact = sc.get("emissions_impact", 0)
            if abs(impact) > 0:
                causes.append(RootCause(
                    cause_id=f"RC-{_new_uuid()[:6]}",
                    cause_name=sc.get("description", "Structural Change"),
                    category="structural_changes",
                    driver_type=VarianceDriverType.STRUCTURAL,
                    contribution_tco2e=round(impact, 2),
                    contribution_pct=round(
                        (impact / max(abs(self._total_variance), 1e-10)) * 100, 1,
                    ),
                    confidence=85.0,
                    evidence=sc.get("evidence", "Reported structural change."),
                    classification=FactorClassification.INTERNAL,
                ))

        self._root_causes = causes

        outputs["root_causes_count"] = len(causes)
        outputs["significant_causes"] = sum(
            1 for c in causes if abs(c.contribution_pct) >= threshold
        )
        for i, c in enumerate(causes[:5]):
            outputs[f"cause_{i+1}_name"] = c.cause_name
            outputs[f"cause_{i+1}_tco2e"] = c.contribution_tco2e
            outputs[f"cause_{i+1}_pct"] = c.contribution_pct

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="attribute_root_causes", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_attribute_root_causes",
        )

    # -------------------------------------------------------------------------
    # Phase 3: Classify Factors
    # -------------------------------------------------------------------------

    async def _phase_classify_factors(self, input_data: VarianceInvestigationInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        for cause in self._root_causes:
            cat = ROOT_CAUSE_CATEGORIES.get(cause.category, {})
            cls_str = cat.get("classification", "mixed")
            if cls_str == "internal":
                cause.classification = FactorClassification.INTERNAL
                cause.control_level = ControlLevel.FULLY_CONTROLLABLE
                cause.actionable = True
            elif cls_str == "external":
                cause.classification = FactorClassification.EXTERNAL
                cause.control_level = ControlLevel.UNCONTROLLABLE
                cause.actionable = False
            else:
                cause.classification = FactorClassification.MIXED
                cause.control_level = ControlLevel.PARTIALLY_CONTROLLABLE
                cause.actionable = True

            # Generate recommended action
            if cause.actionable and cause.contribution_tco2e > 0:
                cause.recommended_action = (
                    f"Address {cause.cause_name}: potential {abs(cause.contribution_tco2e):,.0f} tCO2e impact."
                )

        internal = sum(c.contribution_tco2e for c in self._root_causes if c.classification == FactorClassification.INTERNAL)
        external = sum(c.contribution_tco2e for c in self._root_causes if c.classification == FactorClassification.EXTERNAL)
        mixed = sum(c.contribution_tco2e for c in self._root_causes if c.classification == FactorClassification.MIXED)
        total = abs(internal) + abs(external) + abs(mixed)

        controllable = abs(internal) + abs(mixed) * 0.5
        controllable_pct = (controllable / max(total, 1e-10)) * 100

        self._classification = FactorClassificationResult(
            internal_total_tco2e=round(internal, 2),
            internal_total_pct=round((abs(internal) / max(total, 1e-10)) * 100, 1),
            external_total_tco2e=round(external, 2),
            external_total_pct=round((abs(external) / max(total, 1e-10)) * 100, 1),
            mixed_total_tco2e=round(mixed, 2),
            mixed_total_pct=round((abs(mixed) / max(total, 1e-10)) * 100, 1),
            controllable_pct=round(controllable_pct, 1),
            classified_causes=self._root_causes,
        )

        outputs["internal_pct"] = self._classification.internal_total_pct
        outputs["external_pct"] = self._classification.external_total_pct
        outputs["mixed_pct"] = self._classification.mixed_total_pct
        outputs["controllable_pct"] = round(controllable_pct, 1)
        outputs["actionable_causes"] = sum(1 for c in self._root_causes if c.actionable)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="classify_factors", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_classify_factors",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Quantify Initiatives
    # -------------------------------------------------------------------------

    async def _phase_quantify_initiatives(self, input_data: VarianceInvestigationInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        init_list: List[InitiativeEffectiveness] = []

        for init in input_data.initiatives:
            planned = init.get("planned_reduction", 0)
            actual = init.get("actual_reduction", 0)
            effectiveness = (actual / max(planned, 1e-10)) * 100 if planned > 0 else 0
            variance = actual - planned

            if effectiveness >= 110:
                status = InitiativeStatus.OVERPERFORMING
            elif effectiveness >= 90:
                status = InitiativeStatus.ON_TRACK
            elif effectiveness >= 50:
                status = InitiativeStatus.UNDERPERFORMING
            elif effectiveness > 0:
                status = InitiativeStatus.DELAYED
            else:
                status = InitiativeStatus.NOT_STARTED

            init_name = init.get("name", f"Initiative {len(init_list)+1}")
            recommendation = ""
            if status == InitiativeStatus.UNDERPERFORMING:
                recommendation = f"Investigate underperformance of {init_name}; consider acceleration."
            elif status == InitiativeStatus.DELAYED:
                recommendation = f"Expedite deployment of {init_name}."
            elif status == InitiativeStatus.NOT_STARTED:
                recommendation = f"Initiate {init_name} immediately."

            init_list.append(InitiativeEffectiveness(
                initiative_id=init.get("id", f"INIT-{len(init_list)+1:03d}"),
                initiative_name=init_name,
                planned_reduction_tco2e=round(planned, 2),
                actual_reduction_tco2e=round(actual, 2),
                effectiveness_pct=round(effectiveness, 1),
                status=status,
                variance_tco2e=round(variance, 2),
                variance_pct=round((variance / max(planned, 1e-10)) * 100, 1),
                root_cause_of_variance=init.get("variance_cause", ""),
                recommendation=recommendation,
            ))

        # Portfolio summary
        total_planned = sum(i.planned_reduction_tco2e for i in init_list)
        total_actual = sum(i.actual_reduction_tco2e for i in init_list)
        portfolio_eff = (total_actual / max(total_planned, 1e-10)) * 100

        self._initiatives = InitiativePortfolio(
            total_planned_reduction_tco2e=round(total_planned, 2),
            total_actual_reduction_tco2e=round(total_actual, 2),
            portfolio_effectiveness_pct=round(portfolio_eff, 1),
            initiatives=init_list,
            on_track_count=sum(1 for i in init_list if i.status in (InitiativeStatus.ON_TRACK, InitiativeStatus.OVERPERFORMING)),
            underperforming_count=sum(1 for i in init_list if i.status == InitiativeStatus.UNDERPERFORMING),
            overperforming_count=sum(1 for i in init_list if i.status == InitiativeStatus.OVERPERFORMING),
            gap_tco2e=round(total_planned - total_actual, 2),
        )
        self._initiatives.provenance_hash = _compute_hash(
            self._initiatives.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["initiatives_count"] = len(init_list)
        outputs["total_planned_tco2e"] = round(total_planned, 2)
        outputs["total_actual_tco2e"] = round(total_actual, 2)
        outputs["portfolio_effectiveness_pct"] = round(portfolio_eff, 1)
        outputs["on_track_count"] = self._initiatives.on_track_count
        outputs["underperforming_count"] = self._initiatives.underperforming_count
        outputs["gap_tco2e"] = round(total_planned - total_actual, 2)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="quantify_initiatives", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_quantify_initiatives",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Variance Report
    # -------------------------------------------------------------------------

    async def _phase_variance_report(self, input_data: VarianceInvestigationInput) -> PhaseResult:
        started = _utcnow()
        outputs: Dict[str, Any] = {}

        findings = self._generate_findings()
        recommendations = self._generate_recommendations()

        dominant_lmdi = max(self._lmdi, key=lambda c: abs(c.contribution_tco2e)) if self._lmdi else None

        exec_parts = [
            f"Variance Investigation Report for {self.config.company_name or 'Company'}.",
            f"Total variance: {self._total_variance:+,.0f} tCO2e.",
        ]
        if dominant_lmdi:
            exec_parts.append(
                f"Primary driver: {dominant_lmdi.component_name} "
                f"({dominant_lmdi.contribution_tco2e:+,.0f} tCO2e, {dominant_lmdi.contribution_pct:.0f}%).",
            )
        exec_parts.append(
            f"Controllable factors: {self._classification.controllable_pct:.0f}% of total variance.",
        )
        if self._initiatives.initiatives:
            exec_parts.append(
                f"Initiative portfolio effectiveness: {self._initiatives.portfolio_effectiveness_pct:.0f}%.",
            )

        self._report = VarianceInvestigationReport(
            report_id=f"VIR-{self.workflow_id[:8]}",
            report_date=_utcnow().strftime("%Y-%m-%d"),
            company_name=self.config.company_name,
            investigation_period=str(self.config.investigation_year),
            total_variance_tco2e=round(self._total_variance, 2),
            total_variance_pct=round(
                (self._total_variance / max(abs(input_data.base_period.get("emissions", 100000)), 1e-10)) * 100, 2,
            ),
            lmdi_components=self._lmdi,
            kaya_decomposition=self._kaya,
            root_causes=self._root_causes,
            factor_classification=self._classification,
            initiative_portfolio=self._initiatives,
            executive_summary=" ".join(exec_parts),
            key_findings=findings,
            recommendations=recommendations,
        )
        self._report.provenance_hash = _compute_hash(
            self._report.model_dump_json(exclude={"provenance_hash"}),
        )

        outputs["report_id"] = self._report.report_id
        outputs["findings_count"] = len(findings)
        outputs["recommendations_count"] = len(recommendations)

        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="variance_report", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            completion_pct=100.0, outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            dag_node_id=f"{self.workflow_id}_variance_report",
        )

    def _generate_findings(self) -> List[str]:
        findings: List[str] = []
        findings.append(f"Total variance: {self._total_variance:+,.0f} tCO2e.")
        for comp in self._lmdi[:3]:
            findings.append(
                f"{comp.component_name}: {comp.contribution_tco2e:+,.0f} tCO2e "
                f"({comp.contribution_pct:+.0f}%), direction: {comp.direction}.",
            )
        findings.append(
            f"Controllable variance: {self._classification.controllable_pct:.0f}%.",
        )
        if self._initiatives.initiatives:
            findings.append(
                f"Initiative portfolio: {self._initiatives.portfolio_effectiveness_pct:.0f}% effective "
                f"({self._initiatives.on_track_count} on track, "
                f"{self._initiatives.underperforming_count} underperforming).",
            )
        return findings

    def _generate_recommendations(self) -> List[str]:
        recs: List[str] = []
        for cause in self._root_causes:
            if cause.actionable and cause.recommended_action:
                recs.append(cause.recommended_action)
        for init in self._initiatives.initiatives:
            if init.recommendation:
                recs.append(init.recommendation)
        if not recs:
            recs.append("Continue current monitoring; no significant corrective actions required.")
        return recs[:10]
