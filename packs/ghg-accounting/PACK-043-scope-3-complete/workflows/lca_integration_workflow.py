# -*- coding: utf-8 -*-
"""
LCA Integration Workflow
===================================

4-phase workflow for integrating Life Cycle Assessment (LCA) data into Scope 3
inventories within PACK-043 Scope 3 Complete Pack.

Phases:
    1. PRODUCT_SELECTION     -- Select products for LCA analysis based on
                                revenue/volume materiality thresholds.
    2. BOM_MAPPING           -- Map bill-of-materials (BOM) components to
                                emission factor database entries.
    3. LCA_FACTOR_ASSIGNMENT -- Assign lifecycle emission factors per material,
                                process, and lifecycle stage.
    4. LIFECYCLE_CALCULATION -- Calculate cradle-to-gate and cradle-to-grave
                                footprint with sensitivity analysis.

The workflow follows GreenLang zero-hallucination principles: every emission
factor lookup and footprint calculation uses deterministic reference data
and arithmetic. SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISO 14040:2006 / ISO 14044:2006 -- LCA principles and framework
    GHG Protocol Product Life Cycle Standard
    PEF (Product Environmental Footprint) methodology
    ISO 14067:2018 -- Carbon footprint of products

Schedule: per product line or upon BOM changes
Estimated duration: 4-8 hours per product

Author: GreenLang Platform Team
Version: 43.0.0
"""

_MODULE_VERSION: str = "43.0.0"

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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


class LifecycleStage(str, Enum):
    """Product lifecycle stages."""

    RAW_MATERIAL_EXTRACTION = "raw_material_extraction"
    MATERIAL_PROCESSING = "material_processing"
    MANUFACTURING = "manufacturing"
    DISTRIBUTION = "distribution"
    USE_PHASE = "use_phase"
    END_OF_LIFE = "end_of_life"
    RECYCLING = "recycling"


class LCABoundary(str, Enum):
    """LCA system boundary types."""

    CRADLE_TO_GATE = "cradle_to_gate"
    CRADLE_TO_GRAVE = "cradle_to_grave"
    GATE_TO_GATE = "gate_to_gate"
    CRADLE_TO_CRADLE = "cradle_to_cradle"


class MaterialCategory(str, Enum):
    """Material classification categories."""

    METALS_FERROUS = "metals_ferrous"
    METALS_NON_FERROUS = "metals_non_ferrous"
    PLASTICS = "plastics"
    CHEMICALS = "chemicals"
    ELECTRONICS = "electronics"
    TEXTILES = "textiles"
    PAPER_WOOD = "paper_wood"
    GLASS_CERAMICS = "glass_ceramics"
    CONCRETE_MINERALS = "concrete_minerals"
    FOOD_AGRICULTURE = "food_agriculture"
    ENERGY_FUELS = "energy_fuels"
    OTHER = "other"


class SensitivityParameter(str, Enum):
    """Parameters for sensitivity analysis."""

    EMISSION_FACTOR = "emission_factor"
    MATERIAL_WEIGHT = "material_weight"
    TRANSPORT_DISTANCE = "transport_distance"
    USE_PHASE_ENERGY = "use_phase_energy"
    END_OF_LIFE_RATE = "end_of_life_rate"
    RECYCLED_CONTENT = "recycled_content"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Lifecycle emission factors by material category (kgCO2e per kg)
# Source: ecoinvent 3.9, GaBi, ELCD
MATERIAL_EMISSION_FACTORS: Dict[str, Dict[str, float]] = {
    MaterialCategory.METALS_FERROUS.value: {
        "steel_primary": 2.10,
        "steel_recycled": 0.70,
        "cast_iron": 1.80,
        "stainless_steel": 6.15,
        "default": 2.10,
    },
    MaterialCategory.METALS_NON_FERROUS.value: {
        "aluminum_primary": 11.50,
        "aluminum_recycled": 0.85,
        "copper": 3.80,
        "zinc": 3.10,
        "titanium": 36.00,
        "default": 5.50,
    },
    MaterialCategory.PLASTICS.value: {
        "pe_hdpe": 1.90,
        "pe_ldpe": 2.10,
        "pp": 1.95,
        "pvc": 2.40,
        "pet": 2.70,
        "ps": 3.40,
        "abs": 3.60,
        "nylon": 8.10,
        "bioplastic_pla": 1.30,
        "default": 2.50,
    },
    MaterialCategory.CHEMICALS.value: {
        "solvents": 2.80,
        "adhesives": 3.20,
        "coatings": 4.50,
        "fertilizers": 5.30,
        "default": 3.50,
    },
    MaterialCategory.ELECTRONICS.value: {
        "pcb": 60.00,
        "semiconductors": 150.00,
        "batteries_lithium": 12.00,
        "displays_lcd": 45.00,
        "default": 50.00,
    },
    MaterialCategory.TEXTILES.value: {
        "cotton": 5.90,
        "polyester": 5.50,
        "nylon_fabric": 9.00,
        "wool": 17.00,
        "organic_cotton": 3.80,
        "default": 6.00,
    },
    MaterialCategory.PAPER_WOOD.value: {
        "kraft_paper": 1.10,
        "cardboard": 0.95,
        "softwood_lumber": 0.45,
        "hardwood_lumber": 0.55,
        "plywood": 0.70,
        "default": 0.80,
    },
    MaterialCategory.GLASS_CERAMICS.value: {
        "float_glass": 1.20,
        "container_glass": 0.85,
        "ceramics": 1.50,
        "default": 1.10,
    },
    MaterialCategory.CONCRETE_MINERALS.value: {
        "portland_cement": 0.90,
        "concrete": 0.15,
        "sand_gravel": 0.005,
        "limestone": 0.012,
        "default": 0.30,
    },
    MaterialCategory.FOOD_AGRICULTURE.value: {
        "beef": 27.00,
        "poultry": 6.90,
        "dairy_milk": 3.20,
        "grains_wheat": 0.80,
        "vegetables": 0.50,
        "fruits": 0.70,
        "palm_oil": 7.60,
        "soy": 2.00,
        "default": 3.00,
    },
    MaterialCategory.ENERGY_FUELS.value: {
        "natural_gas_mj": 0.056,
        "diesel_liter": 2.68,
        "electricity_kwh_global": 0.48,
        "default": 1.00,
    },
    MaterialCategory.OTHER.value: {
        "default": 2.00,
    },
}

# Process emission factors (kgCO2e per unit of process output)
PROCESS_EMISSION_FACTORS: Dict[str, float] = {
    "injection_molding_per_kg": 0.75,
    "cnc_machining_per_kg": 1.20,
    "welding_per_meter": 0.45,
    "painting_per_m2": 0.35,
    "assembly_per_unit": 0.10,
    "heat_treatment_per_kg": 0.90,
    "casting_per_kg": 1.50,
    "stamping_per_kg": 0.40,
    "extrusion_per_kg": 0.60,
    "printing_per_m2": 0.25,
}

# Transport emission factors (kgCO2e per tonne-km)
TRANSPORT_EMISSION_FACTORS: Dict[str, float] = {
    "road_truck": 0.062,
    "rail_freight": 0.022,
    "ocean_container": 0.008,
    "air_freight": 0.602,
    "inland_waterway": 0.031,
    "pipeline": 0.025,
}

# Revenue materiality threshold for product selection
DEFAULT_REVENUE_THRESHOLD_PCT: float = 5.0
DEFAULT_VOLUME_THRESHOLD_PCT: float = 5.0


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class WorkflowState(BaseModel):
    """Persistent state for checkpoint/resume."""

    workflow_id: str = Field(default="")
    current_phase: int = Field(default=0)
    phase_statuses: Dict[str, str] = Field(default_factory=dict)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")


class ProductRecord(BaseModel):
    """Product record for LCA selection."""

    product_id: str = Field(default_factory=lambda: f"prod-{uuid.uuid4().hex[:8]}")
    product_name: str = Field(default="")
    product_category: str = Field(default="")
    annual_revenue_usd: float = Field(default=0.0, ge=0.0)
    annual_units_sold: int = Field(default=0, ge=0)
    unit_weight_kg: float = Field(default=0.0, ge=0.0)
    functional_unit: str = Field(default="1 unit", description="Functional unit for LCA")
    has_existing_lca: bool = Field(default=False)


class BOMComponent(BaseModel):
    """Bill-of-materials component."""

    component_id: str = Field(default_factory=lambda: f"comp-{uuid.uuid4().hex[:8]}")
    component_name: str = Field(default="")
    material_category: MaterialCategory = Field(default=MaterialCategory.OTHER)
    material_subtype: str = Field(default="default")
    weight_kg: float = Field(default=0.0, ge=0.0, description="Weight per functional unit")
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    supplier_country: str = Field(default="", description="ISO 3166-1 alpha-2")
    transport_mode: str = Field(default="road_truck")
    transport_distance_km: float = Field(default=500.0, ge=0.0)


class ProductBOM(BaseModel):
    """Complete BOM for a product."""

    product_id: str = Field(default="")
    product_name: str = Field(default="")
    components: List[BOMComponent] = Field(default_factory=list)
    manufacturing_processes: List[str] = Field(default_factory=list)
    total_weight_kg: float = Field(default=0.0, ge=0.0)


class FactorAssignment(BaseModel):
    """Emission factor assignment for a component or process."""

    item_id: str = Field(default="")
    item_name: str = Field(default="")
    lifecycle_stage: LifecycleStage = Field(default=LifecycleStage.RAW_MATERIAL_EXTRACTION)
    emission_factor_kgco2e: float = Field(default=0.0, ge=0.0)
    factor_source: str = Field(default="ecoinvent_3.9")
    quantity: float = Field(default=0.0, ge=0.0)
    unit: str = Field(default="kg")
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    data_quality_score: float = Field(default=3.0, ge=1.0, le=5.0)


class StageFootprint(BaseModel):
    """Carbon footprint for a lifecycle stage."""

    stage: LifecycleStage = Field(...)
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    pct_of_total: float = Field(default=0.0, ge=0.0, le=100.0)
    component_count: int = Field(default=0, ge=0)
    data_quality_avg: float = Field(default=3.0, ge=1.0, le=5.0)


class SensitivityResult(BaseModel):
    """Result of a single sensitivity scenario."""

    parameter: SensitivityParameter = Field(...)
    variation_pct: float = Field(default=0.0, description="% change applied")
    baseline_kgco2e: float = Field(default=0.0, ge=0.0)
    adjusted_kgco2e: float = Field(default=0.0, ge=0.0)
    delta_kgco2e: float = Field(default=0.0)
    delta_pct: float = Field(default=0.0)
    sensitivity_index: float = Field(
        default=0.0, description="Elasticity: %change output / %change input"
    )


class ProductFootprint(BaseModel):
    """Complete LCA footprint for a product."""

    product_id: str = Field(default="")
    product_name: str = Field(default="")
    boundary: LCABoundary = Field(default=LCABoundary.CRADLE_TO_GRAVE)
    functional_unit: str = Field(default="1 unit")
    total_kgco2e: float = Field(default=0.0, ge=0.0)
    total_tco2e_annual: float = Field(default=0.0, ge=0.0)
    stage_breakdown: List[StageFootprint] = Field(default_factory=list)
    factor_assignments: List[FactorAssignment] = Field(default_factory=list)
    sensitivity_results: List[SensitivityResult] = Field(default_factory=list)
    hotspot_stage: str = Field(default="")
    hotspot_component: str = Field(default="")


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class LCAIntegrationInput(BaseModel):
    """Input data model for LCAIntegrationWorkflow."""

    organization_name: str = Field(default="", description="Organization name")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    total_revenue_usd: float = Field(default=0.0, ge=0.0)
    products: List[ProductRecord] = Field(default_factory=list)
    bom_data: List[ProductBOM] = Field(default_factory=list)
    boundary: LCABoundary = Field(default=LCABoundary.CRADLE_TO_GRAVE)
    revenue_threshold_pct: float = Field(
        default=DEFAULT_REVENUE_THRESHOLD_PCT, ge=0.0, le=100.0
    )
    volume_threshold_pct: float = Field(
        default=DEFAULT_VOLUME_THRESHOLD_PCT, ge=0.0, le=100.0
    )
    sensitivity_variation_pct: float = Field(
        default=10.0, ge=1.0, le=50.0,
        description="% variation for sensitivity analysis",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class LCAIntegrationOutput(BaseModel):
    """Complete output from LCAIntegrationWorkflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="lca_integration")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_name: str = Field(default="")
    reporting_year: int = Field(default=2025)
    selected_products: List[str] = Field(default_factory=list)
    product_footprints: List[ProductFootprint] = Field(default_factory=list)
    total_portfolio_kgco2e: float = Field(default=0.0, ge=0.0)
    total_portfolio_tco2e_annual: float = Field(default=0.0, ge=0.0)
    mapped_components: int = Field(default=0, ge=0)
    unmapped_components: int = Field(default=0, ge=0)
    boundary: LCABoundary = Field(default=LCABoundary.CRADLE_TO_GRAVE)
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class LCAIntegrationWorkflow:
    """
    4-phase LCA integration workflow for product-level carbon footprinting.

    Selects material products by revenue/volume materiality, maps BOM
    components to emission factor databases, assigns lifecycle factors per
    stage, and calculates cradle-to-gate/grave footprints with sensitivity.

    Zero-hallucination: all emission factors come from deterministic reference
    data. All arithmetic is pure Python. No LLM calls in any numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _selected_products: Products selected for LCA.
        _bom_map: Product ID -> BOM mapping.
        _factor_assignments: All factor assignments.
        _product_footprints: Calculated footprints.
        _phase_results: Ordered phase outputs.
        _state: Checkpoint/resume state.

    Example:
        >>> wf = LCAIntegrationWorkflow()
        >>> inp = LCAIntegrationInput(
        ...     products=[ProductRecord(product_name="Widget A", annual_revenue_usd=10_000_000)],
        ...     bom_data=[ProductBOM(product_name="Widget A", components=[...])],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_NAMES: List[str] = [
        "product_selection",
        "bom_mapping",
        "lca_factor_assignment",
        "lifecycle_calculation",
    ]

    PHASE_WEIGHTS: Dict[str, float] = {
        "product_selection": 10.0,
        "bom_mapping": 25.0,
        "lca_factor_assignment": 30.0,
        "lifecycle_calculation": 35.0,
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize LCAIntegrationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._selected_products: List[ProductRecord] = []
        self._bom_map: Dict[str, ProductBOM] = {}
        self._factor_assignments: Dict[str, List[FactorAssignment]] = {}
        self._product_footprints: List[ProductFootprint] = []
        self._phase_results: List[PhaseResult] = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[LCAIntegrationInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> LCAIntegrationOutput:
        """
        Execute the 4-phase LCA integration workflow.

        Args:
            input_data: Full input model.
            config: Optional configuration overrides.

        Returns:
            LCAIntegrationOutput with product footprints and sensitivity.
        """
        if input_data is None:
            input_data = LCAIntegrationInput()

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting LCA integration workflow %s org=%s products=%d",
            self.workflow_id,
            input_data.organization_name,
            len(input_data.products),
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING
        self._update_progress(0.0)

        try:
            phase1 = await self._execute_with_retry(
                self._phase_product_selection, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")
            self._update_progress(10.0)

            phase2 = await self._execute_with_retry(
                self._phase_bom_mapping, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")
            self._update_progress(35.0)

            phase3 = await self._execute_with_retry(
                self._phase_lca_factor_assignment, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")
            self._update_progress(65.0)

            phase4 = await self._execute_with_retry(
                self._phase_lifecycle_calculation, input_data, phase_number=4
            )
            self._phase_results.append(phase4)
            if phase4.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 4 failed: {phase4.errors}")
            self._update_progress(100.0)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error(
                "LCA integration workflow failed: %s", exc, exc_info=True
            )
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(
                PhaseResult(
                    phase_name="error",
                    phase_number=0,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                )
            )

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        total_mapped = sum(
            len(bom.components) for bom in self._bom_map.values()
        )

        result = LCAIntegrationOutput(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_name=input_data.organization_name,
            reporting_year=input_data.reporting_year,
            selected_products=[p.product_name for p in self._selected_products],
            product_footprints=self._product_footprints,
            total_portfolio_kgco2e=round(
                sum(pf.total_kgco2e for pf in self._product_footprints), 2
            ),
            total_portfolio_tco2e_annual=round(
                sum(pf.total_tco2e_annual for pf in self._product_footprints), 2
            ),
            mapped_components=total_mapped,
            unmapped_components=0,
            boundary=input_data.boundary,
            progress_pct=self._state.progress_pct,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "LCA integration workflow %s completed in %.2fs status=%s "
            "products=%d total=%.1f tCO2e/yr",
            self.workflow_id,
            elapsed,
            overall_status.value,
            len(self._product_footprints),
            result.total_portfolio_tco2e_annual,
        )
        return result

    def get_state(self) -> WorkflowState:
        """Return current workflow state for checkpoint/resume."""
        return self._state.model_copy()

    async def resume(
        self, state: WorkflowState, input_data: LCAIntegrationInput
    ) -> LCAIntegrationOutput:
        """Resume workflow from a saved checkpoint state."""
        self._state = state
        self.workflow_id = state.workflow_id
        return await self.execute(input_data)

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: LCAIntegrationInput, phase_number: int
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
                        "Phase %d attempt %d/%d failed: %s", phase_number,
                        attempt, self.MAX_RETRIES, exc,
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
    # Phase 1: Product Selection
    # -------------------------------------------------------------------------

    async def _phase_product_selection(
        self, input_data: LCAIntegrationInput
    ) -> PhaseResult:
        """Select products for LCA analysis based on revenue/volume materiality."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._selected_products = []
        total_revenue = input_data.total_revenue_usd
        if total_revenue <= 0:
            total_revenue = sum(p.annual_revenue_usd for p in input_data.products)

        total_volume = sum(p.annual_units_sold for p in input_data.products)

        for product in input_data.products:
            revenue_pct = (
                (product.annual_revenue_usd / total_revenue * 100.0)
                if total_revenue > 0 else 0.0
            )
            volume_pct = (
                (product.annual_units_sold / total_volume * 100.0)
                if total_volume > 0 else 0.0
            )

            meets_revenue = revenue_pct >= input_data.revenue_threshold_pct
            meets_volume = volume_pct >= input_data.volume_threshold_pct

            if meets_revenue or meets_volume:
                self._selected_products.append(product)

        if not self._selected_products and input_data.products:
            warnings.append(
                "No products met materiality thresholds; selecting top 3 by revenue"
            )
            sorted_prods = sorted(
                input_data.products,
                key=lambda p: p.annual_revenue_usd,
                reverse=True,
            )
            self._selected_products = sorted_prods[:3]

        outputs["total_products_evaluated"] = len(input_data.products)
        outputs["products_selected"] = len(self._selected_products)
        outputs["selected_names"] = [p.product_name for p in self._selected_products]
        outputs["revenue_threshold_pct"] = input_data.revenue_threshold_pct
        outputs["volume_threshold_pct"] = input_data.volume_threshold_pct
        outputs["total_revenue_usd"] = round(total_revenue, 2)

        self._state.phase_statuses["product_selection"] = "completed"
        self._state.current_phase = 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 ProductSelection: %d/%d products selected",
            len(self._selected_products),
            len(input_data.products),
        )
        return PhaseResult(
            phase_name="product_selection",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: BOM Mapping
    # -------------------------------------------------------------------------

    async def _phase_bom_mapping(
        self, input_data: LCAIntegrationInput
    ) -> PhaseResult:
        """Map BOM components to emission factor database entries."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._bom_map = {}
        mapped_count = 0
        unmapped_count = 0

        # Build lookup: product_name/id -> BOM
        bom_lookup: Dict[str, ProductBOM] = {}
        for bom in input_data.bom_data:
            bom_lookup[bom.product_id] = bom
            bom_lookup[bom.product_name] = bom

        for product in self._selected_products:
            bom = bom_lookup.get(product.product_id) or bom_lookup.get(product.product_name)
            if bom is None:
                warnings.append(
                    f"No BOM data found for product '{product.product_name}'; "
                    f"will use weight-based estimation"
                )
                # Create synthetic BOM from product weight
                bom = self._create_synthetic_bom(product)

            # Validate component material mappings
            for comp in bom.components:
                cat_factors = MATERIAL_EMISSION_FACTORS.get(
                    comp.material_category.value, {}
                )
                if comp.material_subtype not in cat_factors and "default" in cat_factors:
                    comp.material_subtype = "default"
                    warnings.append(
                        f"Material subtype not found for '{comp.component_name}'; "
                        f"using default factor for {comp.material_category.value}"
                    )

                if cat_factors:
                    mapped_count += 1
                else:
                    unmapped_count += 1

            # Calculate total weight
            bom.total_weight_kg = sum(c.weight_kg for c in bom.components)
            self._bom_map[product.product_id] = bom

        outputs["products_with_bom"] = len(self._bom_map)
        outputs["total_components"] = mapped_count + unmapped_count
        outputs["mapped_components"] = mapped_count
        outputs["unmapped_components"] = unmapped_count
        outputs["mapping_rate_pct"] = round(
            (mapped_count / max(mapped_count + unmapped_count, 1)) * 100.0, 1
        )
        outputs["material_categories_used"] = list(set(
            comp.material_category.value
            for bom in self._bom_map.values()
            for comp in bom.components
        ))

        self._state.phase_statuses["bom_mapping"] = "completed"
        self._state.current_phase = 2

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 BOMMappng: %d products, %d components mapped, %d unmapped",
            len(self._bom_map),
            mapped_count,
            unmapped_count,
        )
        return PhaseResult(
            phase_name="bom_mapping",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: LCA Factor Assignment
    # -------------------------------------------------------------------------

    async def _phase_lca_factor_assignment(
        self, input_data: LCAIntegrationInput
    ) -> PhaseResult:
        """Assign lifecycle emission factors per material/process/stage."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._factor_assignments = {}
        total_assignments = 0

        for product in self._selected_products:
            bom = self._bom_map.get(product.product_id)
            if not bom:
                continue

            assignments: List[FactorAssignment] = []

            # Material extraction + processing factors
            for comp in bom.components:
                ef = self._get_material_factor(comp)
                # Adjust for recycled content
                recycled_factor = ef * 0.3  # Recycled material ~30% of primary
                effective_ef = (
                    ef * (1 - comp.recycled_content_pct / 100.0)
                    + recycled_factor * (comp.recycled_content_pct / 100.0)
                )

                assignments.append(FactorAssignment(
                    item_id=comp.component_id,
                    item_name=comp.component_name,
                    lifecycle_stage=LifecycleStage.RAW_MATERIAL_EXTRACTION,
                    emission_factor_kgco2e=round(effective_ef, 4),
                    factor_source="ecoinvent_3.9",
                    quantity=comp.weight_kg,
                    unit="kg",
                    emissions_kgco2e=round(effective_ef * comp.weight_kg, 4),
                    data_quality_score=3.0,
                ))

                # Transport to factory
                transport_ef = TRANSPORT_EMISSION_FACTORS.get(
                    comp.transport_mode, 0.062
                )
                transport_emissions = (
                    comp.weight_kg / 1000.0
                    * comp.transport_distance_km
                    * transport_ef
                )
                assignments.append(FactorAssignment(
                    item_id=f"{comp.component_id}_transport",
                    item_name=f"{comp.component_name} - inbound transport",
                    lifecycle_stage=LifecycleStage.DISTRIBUTION,
                    emission_factor_kgco2e=round(transport_ef, 6),
                    factor_source="ghg_protocol_transport",
                    quantity=comp.weight_kg / 1000.0 * comp.transport_distance_km,
                    unit="tonne-km",
                    emissions_kgco2e=round(transport_emissions, 4),
                    data_quality_score=2.5,
                ))

            # Manufacturing process factors
            for process_name in bom.manufacturing_processes:
                process_ef = PROCESS_EMISSION_FACTORS.get(process_name, 0.5)
                process_qty = bom.total_weight_kg  # Simplified: apply to total weight
                assignments.append(FactorAssignment(
                    item_id=f"process_{process_name}",
                    item_name=f"Manufacturing: {process_name}",
                    lifecycle_stage=LifecycleStage.MANUFACTURING,
                    emission_factor_kgco2e=round(process_ef, 4),
                    factor_source="industry_average",
                    quantity=process_qty,
                    unit="kg_processed",
                    emissions_kgco2e=round(process_ef * process_qty, 4),
                    data_quality_score=2.5,
                ))

            # Use phase estimation (if cradle-to-grave)
            if input_data.boundary in (
                LCABoundary.CRADLE_TO_GRAVE, LCABoundary.CRADLE_TO_CRADLE
            ):
                use_phase = self._estimate_use_phase(product, bom)
                if use_phase:
                    assignments.append(use_phase)

                eol = self._estimate_end_of_life(product, bom)
                if eol:
                    assignments.append(eol)

            self._factor_assignments[product.product_id] = assignments
            total_assignments += len(assignments)

        outputs["total_factor_assignments"] = total_assignments
        outputs["products_with_factors"] = len(self._factor_assignments)
        outputs["stages_covered"] = list(set(
            fa.lifecycle_stage.value
            for fas in self._factor_assignments.values()
            for fa in fas
        ))
        outputs["avg_data_quality"] = round(
            sum(
                fa.data_quality_score
                for fas in self._factor_assignments.values()
                for fa in fas
            ) / max(total_assignments, 1),
            2,
        )

        self._state.phase_statuses["lca_factor_assignment"] = "completed"
        self._state.current_phase = 3

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 LCAFactorAssignment: %d assignments for %d products",
            total_assignments,
            len(self._factor_assignments),
        )
        return PhaseResult(
            phase_name="lca_factor_assignment",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Lifecycle Calculation
    # -------------------------------------------------------------------------

    async def _phase_lifecycle_calculation(
        self, input_data: LCAIntegrationInput
    ) -> PhaseResult:
        """Calculate cradle-to-gate/grave footprint with sensitivity analysis."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._product_footprints = []

        for product in self._selected_products:
            assignments = self._factor_assignments.get(product.product_id, [])
            if not assignments:
                warnings.append(
                    f"No factor assignments for '{product.product_name}'; skipping"
                )
                continue

            # Aggregate by lifecycle stage
            stage_totals: Dict[str, float] = {}
            stage_counts: Dict[str, int] = {}
            stage_dq: Dict[str, List[float]] = {}

            for fa in assignments:
                key = fa.lifecycle_stage.value
                stage_totals[key] = stage_totals.get(key, 0.0) + fa.emissions_kgco2e
                stage_counts[key] = stage_counts.get(key, 0) + 1
                stage_dq.setdefault(key, []).append(fa.data_quality_score)

            total_kgco2e = sum(stage_totals.values())

            # Build stage breakdown
            stage_breakdown: List[StageFootprint] = []
            for stage in LifecycleStage:
                emissions = stage_totals.get(stage.value, 0.0)
                if emissions > 0 or stage.value in stage_totals:
                    dq_scores = stage_dq.get(stage.value, [3.0])
                    stage_breakdown.append(StageFootprint(
                        stage=stage,
                        emissions_kgco2e=round(emissions, 4),
                        pct_of_total=round(
                            (emissions / total_kgco2e * 100.0)
                            if total_kgco2e > 0 else 0.0, 2
                        ),
                        component_count=stage_counts.get(stage.value, 0),
                        data_quality_avg=round(
                            sum(dq_scores) / len(dq_scores), 2
                        ),
                    ))

            # Annual footprint
            annual_tco2e = (
                total_kgco2e / 1000.0 * product.annual_units_sold
                if product.annual_units_sold > 0
                else total_kgco2e / 1000.0
            )

            # Identify hotspots
            hotspot_stage_obj = max(
                stage_breakdown,
                key=lambda s: s.emissions_kgco2e,
                default=None,
            )
            hotspot_stage = hotspot_stage_obj.stage.value if hotspot_stage_obj else ""

            hotspot_comp_obj = max(
                assignments,
                key=lambda a: a.emissions_kgco2e,
                default=None,
            )
            hotspot_component = hotspot_comp_obj.item_name if hotspot_comp_obj else ""

            # Sensitivity analysis
            sensitivity = self._run_sensitivity_analysis(
                assignments, total_kgco2e, input_data.sensitivity_variation_pct
            )

            footprint = ProductFootprint(
                product_id=product.product_id,
                product_name=product.product_name,
                boundary=input_data.boundary,
                functional_unit=product.functional_unit,
                total_kgco2e=round(total_kgco2e, 4),
                total_tco2e_annual=round(annual_tco2e, 4),
                stage_breakdown=stage_breakdown,
                factor_assignments=assignments,
                sensitivity_results=sensitivity,
                hotspot_stage=hotspot_stage,
                hotspot_component=hotspot_component,
            )
            self._product_footprints.append(footprint)

        outputs["products_calculated"] = len(self._product_footprints)
        outputs["total_portfolio_kgco2e"] = round(
            sum(pf.total_kgco2e for pf in self._product_footprints), 2
        )
        outputs["total_portfolio_tco2e_annual"] = round(
            sum(pf.total_tco2e_annual for pf in self._product_footprints), 2
        )
        outputs["product_summaries"] = [
            {
                "name": pf.product_name,
                "kgco2e_per_unit": pf.total_kgco2e,
                "tco2e_annual": pf.total_tco2e_annual,
                "hotspot_stage": pf.hotspot_stage,
            }
            for pf in self._product_footprints
        ]

        self._state.phase_statuses["lifecycle_calculation"] = "completed"
        self._state.current_phase = 4

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 LifecycleCalculation: %d products, total=%.1f tCO2e/yr",
            len(self._product_footprints),
            outputs["total_portfolio_tco2e_annual"],
        )
        return PhaseResult(
            phase_name="lifecycle_calculation",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _get_material_factor(self, comp: BOMComponent) -> float:
        """Get emission factor for a material component."""
        cat_factors = MATERIAL_EMISSION_FACTORS.get(
            comp.material_category.value, {}
        )
        return cat_factors.get(
            comp.material_subtype,
            cat_factors.get("default", 2.0),
        )

    def _create_synthetic_bom(self, product: ProductRecord) -> ProductBOM:
        """Create synthetic BOM when actual BOM is unavailable."""
        weight = product.unit_weight_kg if product.unit_weight_kg > 0 else 1.0
        return ProductBOM(
            product_id=product.product_id,
            product_name=product.product_name,
            components=[
                BOMComponent(
                    component_name=f"{product.product_name} - primary material",
                    material_category=MaterialCategory.OTHER,
                    material_subtype="default",
                    weight_kg=weight,
                )
            ],
            total_weight_kg=weight,
        )

    def _estimate_use_phase(
        self, product: ProductRecord, bom: ProductBOM
    ) -> Optional[FactorAssignment]:
        """Estimate use-phase emissions based on product weight as proxy."""
        # Simplified: assume 10 kWh electricity per kg of product over lifetime
        weight = bom.total_weight_kg if bom.total_weight_kg > 0 else 1.0
        kwh = weight * 10.0
        ef = MATERIAL_EMISSION_FACTORS.get(
            MaterialCategory.ENERGY_FUELS.value, {}
        ).get("electricity_kwh_global", 0.48)
        emissions = kwh * ef
        return FactorAssignment(
            item_id=f"use_phase_{product.product_id}",
            item_name=f"{product.product_name} - use phase electricity",
            lifecycle_stage=LifecycleStage.USE_PHASE,
            emission_factor_kgco2e=round(ef, 4),
            factor_source="iea_global_average",
            quantity=kwh,
            unit="kWh",
            emissions_kgco2e=round(emissions, 4),
            data_quality_score=2.0,
        )

    def _estimate_end_of_life(
        self, product: ProductRecord, bom: ProductBOM
    ) -> Optional[FactorAssignment]:
        """Estimate end-of-life emissions."""
        weight = bom.total_weight_kg if bom.total_weight_kg > 0 else 1.0
        # Simplified: 0.5 kgCO2e per kg for waste treatment
        ef = 0.50
        emissions = weight * ef
        return FactorAssignment(
            item_id=f"eol_{product.product_id}",
            item_name=f"{product.product_name} - end of life treatment",
            lifecycle_stage=LifecycleStage.END_OF_LIFE,
            emission_factor_kgco2e=ef,
            factor_source="waste_treatment_average",
            quantity=weight,
            unit="kg",
            emissions_kgco2e=round(emissions, 4),
            data_quality_score=2.0,
        )

    def _run_sensitivity_analysis(
        self,
        assignments: List[FactorAssignment],
        baseline: float,
        variation_pct: float,
    ) -> List[SensitivityResult]:
        """Run sensitivity analysis on key parameters."""
        results: List[SensitivityResult] = []

        # Sensitivity to emission factors (+/- variation%)
        for direction in [1.0, -1.0]:
            delta_factor = direction * variation_pct / 100.0
            adjusted = sum(
                fa.emissions_kgco2e * (1 + delta_factor) for fa in assignments
            )
            delta = adjusted - baseline
            results.append(SensitivityResult(
                parameter=SensitivityParameter.EMISSION_FACTOR,
                variation_pct=round(direction * variation_pct, 1),
                baseline_kgco2e=round(baseline, 4),
                adjusted_kgco2e=round(adjusted, 4),
                delta_kgco2e=round(delta, 4),
                delta_pct=round(
                    (delta / baseline * 100.0) if baseline > 0 else 0.0, 2
                ),
                sensitivity_index=round(
                    abs((delta / baseline) / delta_factor) if baseline > 0 and delta_factor != 0 else 0.0,
                    4,
                ),
            ))

        # Sensitivity to material weight
        material_assignments = [
            fa for fa in assignments
            if fa.lifecycle_stage == LifecycleStage.RAW_MATERIAL_EXTRACTION
        ]
        for direction in [1.0, -1.0]:
            delta_factor = direction * variation_pct / 100.0
            mat_adjusted = sum(
                fa.emissions_kgco2e * (1 + delta_factor) for fa in material_assignments
            )
            other = sum(
                fa.emissions_kgco2e for fa in assignments
                if fa.lifecycle_stage != LifecycleStage.RAW_MATERIAL_EXTRACTION
            )
            adjusted = mat_adjusted + other
            delta = adjusted - baseline
            results.append(SensitivityResult(
                parameter=SensitivityParameter.MATERIAL_WEIGHT,
                variation_pct=round(direction * variation_pct, 1),
                baseline_kgco2e=round(baseline, 4),
                adjusted_kgco2e=round(adjusted, 4),
                delta_kgco2e=round(delta, 4),
                delta_pct=round(
                    (delta / baseline * 100.0) if baseline > 0 else 0.0, 2
                ),
                sensitivity_index=round(
                    abs((delta / baseline) / delta_factor) if baseline > 0 and delta_factor != 0 else 0.0,
                    4,
                ),
            ))

        return results

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._selected_products = []
        self._bom_map = {}
        self._factor_assignments = {}
        self._product_footprints = []
        self._phase_results = []
        self._state = WorkflowState(
            workflow_id=self.workflow_id,
            created_at=datetime.utcnow().isoformat(),
        )

    def _update_progress(self, pct: float) -> None:
        """Update progress percentage in state."""
        self._state.progress_pct = min(pct, 100.0)
        self._state.updated_at = datetime.utcnow().isoformat()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: LCAIntegrationOutput) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += f"|{result.workflow_id}|{result.total_portfolio_kgco2e}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
