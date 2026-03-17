# -*- coding: utf-8 -*-
"""
Lifecycle Verification Workflow - PACK-018 EU Green Claims Prep
================================================================

5-phase workflow that verifies lifecycle-based environmental claims by
walking through the Product Environmental Footprint (PEF) methodology.
Covers scope definition, inventory analysis, impact assessment,
interpretation of results, and final PEF scoring with provenance
tracking at every step.

Phases:
    1. ScopeDefinition    -- Define system boundaries and functional unit
    2. InventoryAnalysis  -- Analyse material and energy flows
    3. ImpactAssessment   -- Calculate environmental impact categories
    4. Interpretation     -- Interpret results and identify hotspots
    5. PEFScoring         -- Compute PEF score and performance class

Reference:
    EU Green Claims Directive (COM/2023/166)
    EU PEF Recommendation (2013/179/EU)
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ImpactCategory(str, Enum):
    """PEF impact categories (16 categories per EU PEF 3.0)."""
    CLIMATE_CHANGE = "climate_change"
    OZONE_DEPLETION = "ozone_depletion"
    HUMAN_TOXICITY_CANCER = "human_toxicity_cancer"
    HUMAN_TOXICITY_NON_CANCER = "human_toxicity_non_cancer"
    PARTICULATE_MATTER = "particulate_matter"
    IONISING_RADIATION = "ionising_radiation"
    PHOTOCHEMICAL_OZONE = "photochemical_ozone"
    ACIDIFICATION = "acidification"
    EUTROPHICATION_TERRESTRIAL = "eutrophication_terrestrial"
    EUTROPHICATION_FRESHWATER = "eutrophication_freshwater"
    EUTROPHICATION_MARINE = "eutrophication_marine"
    ECOTOXICITY_FRESHWATER = "ecotoxicity_freshwater"
    LAND_USE = "land_use"
    WATER_USE = "water_use"
    RESOURCE_USE_MINERALS = "resource_use_minerals"
    RESOURCE_USE_FOSSILS = "resource_use_fossils"


class PEFPerformanceClass(str, Enum):
    """PEF performance class (A-E)."""
    CLASS_A = "A"
    CLASS_B = "B"
    CLASS_C = "C"
    CLASS_D = "D"
    CLASS_E = "E"


# =============================================================================
# DATA MODELS
# =============================================================================


class WorkflowInput(BaseModel):
    """Input model for LifecycleVerificationWorkflow."""
    product_name: str = Field(default="", description="Name of the product being assessed")
    lifecycle_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Lifecycle inventory data: materials, energy, transport, etc.",
    )
    functional_unit: str = Field(default="1 unit of product", description="Functional unit")
    system_boundary: str = Field(default="cradle-to-grave", description="System boundary type")
    entity_name: str = Field(default="", description="Reporting entity")
    config: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class WorkflowResult(BaseModel):
    """Complete result from LifecycleVerificationWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="lifecycle_verification")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class LifecycleVerificationWorkflow:
    """
    5-phase lifecycle verification workflow aligned with PEF methodology.

    Walks through scope definition, inventory analysis, impact assessment,
    interpretation, and PEF scoring to verify lifecycle-based environmental
    claims for EU Green Claims Directive compliance.

    Zero-hallucination: all impact calculations use deterministic formulas
    and characterisation factors. No LLM calls in numeric paths.

    Example:
        >>> wf = LifecycleVerificationWorkflow()
        >>> result = wf.execute(
        ...     product_name="Widget-X",
        ...     lifecycle_data={"materials": [{"name": "steel", "mass_kg": 2.5}]},
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "lifecycle_verification"

    # Simplified characterisation factors (kg CO2-eq per kg material)
    # Production-grade implementation would load these from a database
    EMISSION_FACTORS: Dict[str, float] = {
        "steel": 1.80,
        "aluminium": 8.24,
        "plastic_pp": 1.98,
        "plastic_pe": 1.80,
        "plastic_pet": 2.15,
        "glass": 0.86,
        "paper": 1.10,
        "cardboard": 0.94,
        "wood": 0.45,
        "concrete": 0.13,
        "copper": 3.81,
        "cotton": 5.89,
        "polyester": 5.55,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LifecycleVerificationWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 5-phase lifecycle verification pipeline.

        Keyword Args:
            product_name: Name of the product.
            lifecycle_data: Dict of lifecycle inventory data.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            product_name=kwargs.get("product_name", ""),
            lifecycle_data=kwargs.get("lifecycle_data", {}),
            functional_unit=kwargs.get("functional_unit", "1 unit of product"),
            system_boundary=kwargs.get("system_boundary", "cradle-to-grave"),
            entity_name=kwargs.get("entity_name", ""),
            config=kwargs.get("config", {}),
        )

        started_at = _utcnow()
        self.logger.info("Starting %s workflow %s for product '%s'",
                         self.WORKFLOW_NAME, self.workflow_id, input_data.product_name)
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Scope Definition
            phase_results.append(self._phase_scope_definition(input_data))

            # Phase 2 -- Inventory Analysis
            phase_results.append(self._phase_inventory_analysis(input_data))

            # Phase 3 -- Impact Assessment
            inventory_data = phase_results[1].result_data
            phase_results.append(self._phase_impact_assessment(input_data, inventory_data))

            # Phase 4 -- Interpretation
            impact_data = phase_results[2].result_data
            phase_results.append(self._phase_interpretation(input_data, impact_data))

            # Phase 5 -- PEF Scoring
            interpretation_data = phase_results[3].result_data
            phase_results.append(
                self._phase_pef_scoring(input_data, impact_data, interpretation_data)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=_utcnow(),
                completed_at=_utcnow(),
                error_message=str(exc),
            ))

        completed_at = _utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "product_name": input_data.product_name,
            "phases_completed": len(completed_phases),
            "phases_total": 5,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info("Workflow %s %s for '%s'",
                         self.workflow_id, overall_status.value, input_data.product_name)
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # Phase 1: Scope Definition
    # ------------------------------------------------------------------

    def _phase_scope_definition(self, input_data: WorkflowInput) -> PhaseResult:
        """Define system boundaries, functional unit, and scope."""
        started = _utcnow()
        self.logger.info("Phase 1/5 ScopeDefinition")

        lifecycle_stages = input_data.lifecycle_data.get("stages", [
            "raw_material_acquisition",
            "manufacturing",
            "distribution",
            "use_phase",
            "end_of_life",
        ])

        result_data: Dict[str, Any] = {
            "product_name": input_data.product_name,
            "functional_unit": input_data.functional_unit,
            "system_boundary": input_data.system_boundary,
            "lifecycle_stages": lifecycle_stages,
            "stages_count": len(lifecycle_stages),
            "includes_use_phase": "use_phase" in lifecycle_stages,
            "includes_end_of_life": "end_of_life" in lifecycle_stages,
            "data_quality_requirements": {
                "minimum_dqr_score": 2.0,
                "primary_data_required": True,
            },
        }

        return PhaseResult(
            phase_name="ScopeDefinition",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Inventory Analysis
    # ------------------------------------------------------------------

    def _phase_inventory_analysis(self, input_data: WorkflowInput) -> PhaseResult:
        """Analyse material and energy inputs/outputs across lifecycle."""
        started = _utcnow()
        self.logger.info("Phase 2/5 InventoryAnalysis")

        materials = input_data.lifecycle_data.get("materials", [])
        energy = input_data.lifecycle_data.get("energy", [])
        transport = input_data.lifecycle_data.get("transport", [])

        total_mass_kg = sum(m.get("mass_kg", 0.0) for m in materials)
        total_energy_kwh = sum(e.get("kwh", 0.0) for e in energy)
        total_transport_tkm = sum(t.get("tonne_km", 0.0) for t in transport)

        material_breakdown: List[Dict[str, Any]] = []
        for mat in materials:
            name = mat.get("name", "unknown").lower()
            mass = mat.get("mass_kg", 0.0)
            ef = self.EMISSION_FACTORS.get(name, 1.0)
            material_breakdown.append({
                "name": name,
                "mass_kg": mass,
                "emission_factor": ef,
                "co2_eq_kg": round(mass * ef, 4),
                "mass_pct": round((mass / total_mass_kg * 100) if total_mass_kg else 0.0, 1),
            })

        result_data: Dict[str, Any] = {
            "materials_count": len(materials),
            "energy_inputs_count": len(energy),
            "transport_entries_count": len(transport),
            "total_mass_kg": round(total_mass_kg, 4),
            "total_energy_kwh": round(total_energy_kwh, 4),
            "total_transport_tkm": round(total_transport_tkm, 4),
            "material_breakdown": material_breakdown,
            "total_material_co2_eq_kg": round(
                sum(m["co2_eq_kg"] for m in material_breakdown), 4
            ),
        }

        return PhaseResult(
            phase_name="InventoryAnalysis",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: Impact Assessment
    # ------------------------------------------------------------------

    def _phase_impact_assessment(
        self, input_data: WorkflowInput, inventory_data: Dict[str, Any],
    ) -> PhaseResult:
        """Calculate environmental impact category results."""
        started = _utcnow()
        self.logger.info("Phase 3/5 ImpactAssessment")

        material_co2 = inventory_data.get("total_material_co2_eq_kg", 0.0)
        energy_kwh = inventory_data.get("total_energy_kwh", 0.0)
        transport_tkm = inventory_data.get("total_transport_tkm", 0.0)

        # Simplified impact calculations (deterministic)
        # Grid electricity emission factor: 0.4 kg CO2-eq/kWh (EU average)
        energy_co2 = round(energy_kwh * 0.4, 4)
        # Transport emission factor: 0.062 kg CO2-eq/tkm (road average)
        transport_co2 = round(transport_tkm * 0.062, 4)
        total_co2 = round(material_co2 + energy_co2 + transport_co2, 4)

        # Normalised impact scores (simplified, per PEF normalisation refs)
        impact_results: Dict[str, Any] = {
            ImpactCategory.CLIMATE_CHANGE.value: {
                "value": total_co2,
                "unit": "kg CO2-eq",
                "normalised_score": round(total_co2 / 8100.0, 6),
            },
            ImpactCategory.WATER_USE.value: {
                "value": round(inventory_data.get("total_mass_kg", 0.0) * 0.5, 4),
                "unit": "m3 water-eq",
                "normalised_score": round(
                    inventory_data.get("total_mass_kg", 0.0) * 0.5 / 11500.0, 6
                ),
            },
            ImpactCategory.RESOURCE_USE_FOSSILS.value: {
                "value": round(energy_kwh * 3.6, 4),
                "unit": "MJ",
                "normalised_score": round(energy_kwh * 3.6 / 65000.0, 6),
            },
        }

        result_data: Dict[str, Any] = {
            "impact_results": impact_results,
            "total_climate_change_kg_co2_eq": total_co2,
            "material_contribution_pct": round(
                (material_co2 / total_co2 * 100) if total_co2 else 0.0, 1
            ),
            "energy_contribution_pct": round(
                (energy_co2 / total_co2 * 100) if total_co2 else 0.0, 1
            ),
            "transport_contribution_pct": round(
                (transport_co2 / total_co2 * 100) if total_co2 else 0.0, 1
            ),
            "categories_assessed": len(impact_results),
        }

        return PhaseResult(
            phase_name="ImpactAssessment",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Interpretation
    # ------------------------------------------------------------------

    def _phase_interpretation(
        self, input_data: WorkflowInput, impact_data: Dict[str, Any],
    ) -> PhaseResult:
        """Interpret results, identify hotspots, and assess data quality."""
        started = _utcnow()
        self.logger.info("Phase 4/5 Interpretation")

        # Identify the dominant contributor
        contributions = {
            "materials": impact_data.get("material_contribution_pct", 0.0),
            "energy": impact_data.get("energy_contribution_pct", 0.0),
            "transport": impact_data.get("transport_contribution_pct", 0.0),
        }
        hotspot = max(contributions, key=contributions.get)  # type: ignore[arg-type]

        total_co2 = impact_data.get("total_climate_change_kg_co2_eq", 0.0)
        # Data quality rating: simplified DQR score
        dqr_score = self._compute_data_quality_score(input_data.lifecycle_data)

        result_data: Dict[str, Any] = {
            "hotspot_category": hotspot,
            "hotspot_contribution_pct": contributions[hotspot],
            "contributions": contributions,
            "total_climate_change_kg_co2_eq": total_co2,
            "data_quality_rating": dqr_score,
            "data_quality_sufficient": dqr_score <= 3.0,
            "sensitivity_flag": total_co2 > 100.0,
            "improvement_opportunities": self._identify_improvements(contributions, hotspot),
        }

        return PhaseResult(
            phase_name="Interpretation",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 5: PEF Scoring
    # ------------------------------------------------------------------

    def _phase_pef_scoring(
        self,
        input_data: WorkflowInput,
        impact_data: Dict[str, Any],
        interpretation_data: Dict[str, Any],
    ) -> PhaseResult:
        """Compute final PEF score and assign performance class."""
        started = _utcnow()
        self.logger.info("Phase 5/5 PEFScoring")

        total_co2 = impact_data.get("total_climate_change_kg_co2_eq", 0.0)
        dqr = interpretation_data.get("data_quality_rating", 3.0)

        # Weighted single score (simplified from PEF weighting set)
        # Climate change has ~21% weight in PEF
        weighted_score = round(total_co2 * 0.2106, 4)

        # Assign performance class based on benchmarks
        performance_class = self._assign_performance_class(weighted_score)

        result_data: Dict[str, Any] = {
            "pef_weighted_score": weighted_score,
            "performance_class": performance_class.value,
            "total_climate_change_kg_co2_eq": total_co2,
            "data_quality_rating": dqr,
            "methodology": "PEF 3.0 (simplified)",
            "system_boundary": input_data.system_boundary,
            "functional_unit": input_data.functional_unit,
            "claim_verifiable": performance_class in (
                PEFPerformanceClass.CLASS_A, PEFPerformanceClass.CLASS_B
            ),
            "verification_summary": (
                f"Product '{input_data.product_name}' achieved PEF class "
                f"{performance_class.value} with weighted score {weighted_score:.4f}. "
                f"Data quality rating: {dqr:.1f}/5.0."
            ),
        }

        return PhaseResult(
            phase_name="PEFScoring",
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_data_quality_score(self, lifecycle_data: Dict[str, Any]) -> float:
        """Compute data quality rating (1.0 = best, 5.0 = worst)."""
        materials = lifecycle_data.get("materials", [])
        if not materials:
            return 4.0  # Poor -- no data
        has_primary = any(m.get("data_source") == "primary" for m in materials)
        has_measured = any(m.get("measurement") == "measured" for m in materials)
        if has_primary and has_measured:
            return 1.5
        if has_primary:
            return 2.0
        if len(materials) > 3:
            return 2.5
        return 3.0

    def _assign_performance_class(self, weighted_score: float) -> PEFPerformanceClass:
        """Assign PEF performance class from weighted score."""
        if weighted_score < 1.0:
            return PEFPerformanceClass.CLASS_A
        if weighted_score < 5.0:
            return PEFPerformanceClass.CLASS_B
        if weighted_score < 15.0:
            return PEFPerformanceClass.CLASS_C
        if weighted_score < 30.0:
            return PEFPerformanceClass.CLASS_D
        return PEFPerformanceClass.CLASS_E

    def _identify_improvements(
        self, contributions: Dict[str, float], hotspot: str,
    ) -> List[str]:
        """Identify improvement opportunities based on hotspot analysis."""
        improvements: List[str] = []
        if hotspot == "materials":
            improvements.append("Consider material substitution with lower-impact alternatives")
            improvements.append("Increase recycled content where feasible")
        elif hotspot == "energy":
            improvements.append("Transition to renewable energy sources")
            improvements.append("Improve energy efficiency in manufacturing")
        elif hotspot == "transport":
            improvements.append("Optimise logistics and reduce transport distances")
            improvements.append("Shift to lower-emission transport modes")
        return improvements
