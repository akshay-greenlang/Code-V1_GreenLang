# -*- coding: utf-8 -*-
"""
Carbon Footprint Assessment Workflow
==========================================

4-phase workflow for battery carbon footprint assessment per EU Battery
Regulation 2023/1542, Articles 7 and Annex II. Implements data collection
from bill-of-materials and energy records, life-cycle assessment (LCA)
calculation, performance class assignment, and carbon footprint declaration
generation.

Phases:
    1. DataCollection        -- Gather BOM, energy, and transport data
    2. LCACalculation        -- Apply emission factors across lifecycle stages
    3. PerformanceClass      -- Assign A-E performance class per thresholds
    4. DeclarationGeneration -- Produce EU-compliant carbon footprint declaration

Regulatory references:
    - EU Regulation 2023/1542 Art. 7 (carbon footprint declaration)
    - EU Regulation 2023/1542 Annex II (carbon footprint calculation rules)
    - Commission Delegated Regulation on carbon footprint calculation methodology
    - ISO 14067:2018 (carbon footprint of products)
    - ISO 14040/14044 (LCA methodology)

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

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

class WorkflowPhase(str, Enum):
    """Phases of the carbon footprint assessment workflow."""
    DATA_COLLECTION = "data_collection"
    LCA_CALCULATION = "lca_calculation"
    PERFORMANCE_CLASS = "performance_class"
    DECLARATION_GENERATION = "declaration_generation"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class LifecycleStage(str, Enum):
    """Battery lifecycle stages per EU Battery Regulation Annex II."""
    RAW_MATERIAL_ACQUISITION = "raw_material_acquisition"
    PREPROCESSING = "preprocessing"
    ACTIVE_MATERIAL_PRODUCTION = "active_material_production"
    CELL_MANUFACTURING = "cell_manufacturing"
    BATTERY_ASSEMBLY = "battery_assembly"
    DISTRIBUTION = "distribution"
    END_OF_LIFE_RECYCLING = "end_of_life_recycling"

class PerformanceClassGrade(str, Enum):
    """Carbon footprint performance class grades per Art. 7(3)."""
    CLASS_A = "A"
    CLASS_B = "B"
    CLASS_C = "C"
    CLASS_D = "D"
    CLASS_E = "E"

class BatteryChemistry(str, Enum):
    """Battery chemistry types."""
    NMC_111 = "NMC111"
    NMC_532 = "NMC532"
    NMC_622 = "NMC622"
    NMC_811 = "NMC811"
    NCA = "NCA"
    LFP = "LFP"
    LMO = "LMO"
    NMC_LMO = "NMC_LMO"
    SOLID_STATE = "SOLID_STATE"
    SODIUM_ION = "SODIUM_ION"
    OTHER = "OTHER"

class BatteryCategory(str, Enum):
    """Battery categories per EU Battery Regulation Art. 2."""
    EV_BATTERY = "ev_battery"
    INDUSTRIAL_BATTERY = "industrial_battery"
    LMT_BATTERY = "lmt_battery"
    SLI_BATTERY = "sli_battery"
    PORTABLE_BATTERY = "portable_battery"

# =============================================================================
# DEFAULT EMISSION FACTORS (kgCO2e per unit)
# =============================================================================

# kgCO2e per kg of material produced
MATERIAL_EMISSION_FACTORS: Dict[str, float] = {
    "lithium_carbonate": 8.50,
    "lithium_hydroxide": 10.20,
    "cobalt_sulfate": 7.30,
    "nickel_sulfate": 6.80,
    "manganese_sulfate": 1.50,
    "graphite_natural": 3.20,
    "graphite_synthetic": 12.50,
    "copper_foil": 4.10,
    "aluminium_foil": 8.90,
    "electrolyte_lipf6": 5.60,
    "separator_pe_pp": 3.80,
    "steel_casing": 2.10,
    "bms_electronics": 25.00,
    "iron_phosphate": 1.80,
    "sodium_carbonate": 0.95,
}

# kgCO2e per kWh of energy
ENERGY_EMISSION_FACTORS: Dict[str, float] = {
    "grid_eu_avg": 0.256,
    "grid_cn": 0.581,
    "grid_us_avg": 0.386,
    "grid_kr": 0.415,
    "grid_jp": 0.457,
    "grid_se": 0.013,
    "grid_no": 0.009,
    "grid_fr": 0.052,
    "grid_de": 0.338,
    "grid_pl": 0.623,
    "renewable_solar": 0.041,
    "renewable_wind": 0.012,
    "natural_gas": 0.185,
}

# kgCO2e per tonne-km for transport modes
TRANSPORT_EMISSION_FACTORS: Dict[str, float] = {
    "road_truck": 0.062,
    "rail_freight": 0.022,
    "sea_container": 0.008,
    "air_freight": 0.602,
    "inland_waterway": 0.031,
}

# Performance class thresholds: kgCO2e/kWh for EV batteries
PERFORMANCE_CLASS_THRESHOLDS_EV: Dict[str, float] = {
    "A": 50.0,
    "B": 65.0,
    "C": 80.0,
    "D": 100.0,
}

# Performance class thresholds: kgCO2e/kWh for industrial batteries
PERFORMANCE_CLASS_THRESHOLDS_INDUSTRIAL: Dict[str, float] = {
    "A": 55.0,
    "B": 70.0,
    "C": 90.0,
    "D": 110.0,
}

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

class MaterialRecord(BaseModel):
    """A bill-of-materials material entry."""
    material_id: str = Field(default_factory=lambda: f"mat-{_new_uuid()[:8]}")
    material_name: str = Field(..., description="Material name")
    mass_kg: float = Field(default=0.0, ge=0.0, description="Mass in kilograms")
    emission_factor_kgco2e_per_kg: float = Field(
        default=0.0, ge=0.0, description="EF in kgCO2e/kg"
    )
    emission_factor_source: str = Field(default="", description="EF source")
    lifecycle_stage: LifecycleStage = Field(
        default=LifecycleStage.RAW_MATERIAL_ACQUISITION
    )
    origin_country: str = Field(default="", description="ISO 3166 country code")
    recycled_content_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)

class EnergyRecord(BaseModel):
    """Energy consumption record for manufacturing phase."""
    energy_id: str = Field(default_factory=lambda: f"ene-{_new_uuid()[:8]}")
    energy_source: str = Field(..., description="Energy source key")
    consumption_kwh: float = Field(default=0.0, ge=0.0, description="Energy in kWh")
    emission_factor_kgco2e_per_kwh: float = Field(default=0.0, ge=0.0)
    lifecycle_stage: LifecycleStage = Field(
        default=LifecycleStage.CELL_MANUFACTURING
    )
    facility_location: str = Field(default="", description="Country code")
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)

class TransportRecord(BaseModel):
    """Transport emission record."""
    transport_id: str = Field(default_factory=lambda: f"trn-{_new_uuid()[:8]}")
    mode: str = Field(..., description="Transport mode key")
    distance_km: float = Field(default=0.0, ge=0.0)
    cargo_mass_tonnes: float = Field(default=0.0, ge=0.0)
    emission_factor_kgco2e_per_tkm: float = Field(default=0.0, ge=0.0)
    lifecycle_stage: LifecycleStage = Field(default=LifecycleStage.DISTRIBUTION)
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)

class CarbonFootprintInput(BaseModel):
    """Input data model for CarbonFootprintWorkflow."""
    battery_id: str = Field(default_factory=lambda: f"bat-{_new_uuid()[:8]}")
    battery_model: str = Field(default="", description="Battery model identifier")
    battery_category: BatteryCategory = Field(default=BatteryCategory.EV_BATTERY)
    battery_chemistry: BatteryChemistry = Field(default=BatteryChemistry.NMC_811)
    battery_capacity_kwh: float = Field(
        default=0.0, ge=0.0, description="Rated energy capacity in kWh"
    )
    battery_mass_kg: float = Field(default=0.0, ge=0.0, description="Total mass in kg")
    material_records: List[MaterialRecord] = Field(default_factory=list)
    energy_records: List[EnergyRecord] = Field(default_factory=list)
    transport_records: List[TransportRecord] = Field(default_factory=list)
    recycling_credit_kgco2e: float = Field(
        default=0.0, ge=0.0,
        description="End-of-life recycling credits (kgCO2e avoided)"
    )
    functional_unit: str = Field(
        default="kWh_total_energy",
        description="Functional unit for normalization"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class LifecycleStageResult(BaseModel):
    """Emissions subtotal for a lifecycle stage."""
    stage: str = Field(..., description="Lifecycle stage name")
    emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    record_count: int = Field(default=0, ge=0)

class CarbonFootprintDeclaration(BaseModel):
    """EU Battery Regulation compliant carbon footprint declaration."""
    declaration_id: str = Field(default_factory=lambda: f"cfd-{_new_uuid()[:8]}")
    battery_id: str = Field(default="")
    battery_model: str = Field(default="")
    battery_category: str = Field(default="")
    battery_chemistry: str = Field(default="")
    capacity_kwh: float = Field(default=0.0)
    total_carbon_footprint_kgco2e: float = Field(default=0.0)
    carbon_footprint_per_kwh: float = Field(default=0.0)
    performance_class: str = Field(default="")
    lifecycle_stages: List[LifecycleStageResult] = Field(default_factory=list)
    recycling_credit_kgco2e: float = Field(default=0.0)
    functional_unit: str = Field(default="kWh")
    methodology: str = Field(default="EU Battery Regulation 2023/1542 Annex II")
    reporting_year: int = Field(default=2025)
    issued_at: str = Field(default="")
    regulation_reference: str = Field(
        default="EU Regulation 2023/1542 Art. 7"
    )

class CarbonFootprintResult(BaseModel):
    """Complete result from carbon footprint assessment workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="carbon_footprint_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    total_duration_seconds: float = Field(default=0.0)
    duration_ms: float = Field(default=0.0)
    battery_id: str = Field(default="")
    total_carbon_footprint_kgco2e: float = Field(default=0.0, ge=0.0)
    carbon_footprint_per_kwh: float = Field(default=0.0, ge=0.0)
    performance_class: str = Field(default="")
    lifecycle_breakdown: List[LifecycleStageResult] = Field(default_factory=list)
    declaration: Optional[CarbonFootprintDeclaration] = Field(default=None)
    material_emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    energy_emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    transport_emissions_kgco2e: float = Field(default=0.0, ge=0.0)
    recycling_credit_kgco2e: float = Field(default=0.0, ge=0.0)
    reporting_year: int = Field(default=2025)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CarbonFootprintWorkflow:
    """
    4-phase carbon footprint assessment workflow per EU Battery Regulation.

    Implements end-to-end battery carbon footprint calculation following
    EU Regulation 2023/1542 Art. 7 and Annex II. Collects bill-of-materials,
    energy, and transport data, then applies lifecycle emission factors,
    assigns a performance class, and generates a declaration.

    Zero-hallucination: all emissions use deterministic arithmetic with
    documented emission factors. No LLM in numeric calculation paths.

    Example:
        >>> wf = CarbonFootprintWorkflow()
        >>> inp = CarbonFootprintInput(
        ...     battery_capacity_kwh=75.0,
        ...     material_records=[...],
        ...     energy_records=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.performance_class in ("A", "B", "C", "D", "E")
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CarbonFootprintWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._materials: List[MaterialRecord] = []
        self._energy: List[EnergyRecord] = []
        self._transport: List[TransportRecord] = []
        self._lifecycle_breakdown: List[LifecycleStageResult] = []
        self._total_kgco2e: float = 0.0
        self._per_kwh: float = 0.0
        self._performance_class: str = ""
        self._declaration: Optional[CarbonFootprintDeclaration] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DATA_COLLECTION.value, "description": "Gather BOM, energy, and transport data"},
            {"name": WorkflowPhase.LCA_CALCULATION.value, "description": "Apply emission factors across lifecycle stages"},
            {"name": WorkflowPhase.PERFORMANCE_CLASS.value, "description": "Assign A-E performance class"},
            {"name": WorkflowPhase.DECLARATION_GENERATION.value, "description": "Produce carbon footprint declaration"},
        ]

    def validate_inputs(self, input_data: CarbonFootprintInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if input_data.battery_capacity_kwh <= 0:
            issues.append("Battery capacity must be greater than zero")
        if not input_data.material_records and not input_data.energy_records:
            issues.append("At least material or energy records are required")
        for mat in input_data.material_records:
            if mat.mass_kg < 0:
                issues.append(f"Material {mat.material_id}: negative mass")
        for ene in input_data.energy_records:
            if ene.consumption_kwh < 0:
                issues.append(f"Energy {ene.energy_id}: negative consumption")
        for trn in input_data.transport_records:
            if trn.distance_km < 0:
                issues.append(f"Transport {trn.transport_id}: negative distance")
        return issues

    async def execute(
        self,
        input_data: Optional[CarbonFootprintInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CarbonFootprintResult:
        """
        Execute the 4-phase carbon footprint assessment workflow.

        Args:
            input_data: Full input model (preferred).
            config: Configuration overrides.

        Returns:
            CarbonFootprintResult with lifecycle emissions, performance class,
            and declaration.
        """
        if input_data is None:
            input_data = CarbonFootprintInput(config=config or {})

        started_at = utcnow()
        self.logger.info("Starting carbon footprint workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS
        phases_done = 0

        try:
            phase_results.append(await self._phase_data_collection(input_data))
            phases_done += 1
            phase_results.append(await self._phase_lca_calculation(input_data))
            phases_done += 1
            phase_results.append(await self._phase_performance_class(input_data))
            phases_done += 1
            phase_results.append(await self._phase_declaration_generation(input_data))
            phases_done += 1
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Carbon footprint workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        mat_total = sum(m.emissions_kgco2e for m in self._materials)
        ene_total = sum(e.emissions_kgco2e for e in self._energy)
        trn_total = sum(t.emissions_kgco2e for t in self._transport)

        result = CarbonFootprintResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=phases_done,
            total_duration_seconds=elapsed,
            duration_ms=round(elapsed * 1000, 2),
            battery_id=input_data.battery_id,
            total_carbon_footprint_kgco2e=round(self._total_kgco2e, 2),
            carbon_footprint_per_kwh=round(self._per_kwh, 2),
            performance_class=self._performance_class,
            lifecycle_breakdown=self._lifecycle_breakdown,
            declaration=self._declaration,
            material_emissions_kgco2e=round(mat_total, 2),
            energy_emissions_kgco2e=round(ene_total, 2),
            transport_emissions_kgco2e=round(trn_total, 2),
            recycling_credit_kgco2e=round(input_data.recycling_credit_kgco2e, 2),
            reporting_year=input_data.reporting_year,
            executed_at=utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Carbon footprint %s completed in %.2fs: %.2f kgCO2e total, class %s",
            self.workflow_id, elapsed, self._total_kgco2e, self._performance_class,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Data Collection
    # -------------------------------------------------------------------------

    async def _phase_data_collection(
        self, input_data: CarbonFootprintInput,
    ) -> PhaseResult:
        """Gather and validate BOM, energy, and transport data."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._materials = list(input_data.material_records)
        self._energy = list(input_data.energy_records)
        self._transport = list(input_data.transport_records)

        # Apply default emission factors where missing
        for mat in self._materials:
            if mat.emission_factor_kgco2e_per_kg <= 0:
                default_ef = self._lookup_material_ef(mat.material_name)
                if default_ef > 0:
                    mat.emission_factor_kgco2e_per_kg = default_ef
                    mat.emission_factor_source = "pack_020_defaults"
                    warnings.append(
                        f"Material {mat.material_id}: applied default EF "
                        f"{default_ef} kgCO2e/kg for {mat.material_name}"
                    )

        for ene in self._energy:
            if ene.emission_factor_kgco2e_per_kwh <= 0:
                default_ef = ENERGY_EMISSION_FACTORS.get(ene.energy_source, 0.0)
                if default_ef > 0:
                    ene.emission_factor_kgco2e_per_kwh = default_ef
                    warnings.append(
                        f"Energy {ene.energy_id}: applied default EF "
                        f"{default_ef} kgCO2e/kWh for {ene.energy_source}"
                    )

        for trn in self._transport:
            if trn.emission_factor_kgco2e_per_tkm <= 0:
                default_ef = TRANSPORT_EMISSION_FACTORS.get(trn.mode, 0.0)
                if default_ef > 0:
                    trn.emission_factor_kgco2e_per_tkm = default_ef
                    warnings.append(
                        f"Transport {trn.transport_id}: applied default EF "
                        f"{default_ef} kgCO2e/tkm for {trn.mode}"
                    )

        # Collect stage distribution
        stage_counts: Dict[str, int] = {}
        for rec in [*self._materials, *self._energy, *self._transport]:
            stage_counts[rec.lifecycle_stage.value] = (
                stage_counts.get(rec.lifecycle_stage.value, 0) + 1
            )

        total_material_mass = sum(m.mass_kg for m in self._materials)

        outputs["materials_collected"] = len(self._materials)
        outputs["energy_records_collected"] = len(self._energy)
        outputs["transport_records_collected"] = len(self._transport)
        outputs["total_material_mass_kg"] = round(total_material_mass, 2)
        outputs["lifecycle_stage_distribution"] = stage_counts
        outputs["battery_capacity_kwh"] = input_data.battery_capacity_kwh
        outputs["battery_chemistry"] = input_data.battery_chemistry.value

        if not self._materials:
            warnings.append("No material records provided; BOM emissions will be zero")
        if not self._energy:
            warnings.append("No energy records provided; manufacturing emissions will be zero")

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DataCollection: %d materials, %d energy, %d transport",
            len(self._materials), len(self._energy), len(self._transport),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DATA_COLLECTION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _lookup_material_ef(self, material_name: str) -> float:
        """Look up default material emission factor by name."""
        key = material_name.lower().replace(" ", "_").replace("-", "_")
        if key in MATERIAL_EMISSION_FACTORS:
            return MATERIAL_EMISSION_FACTORS[key]
        for ef_key, ef_val in MATERIAL_EMISSION_FACTORS.items():
            if key in ef_key or ef_key in key:
                return ef_val
        return 0.0

    # -------------------------------------------------------------------------
    # Phase 2: LCA Calculation
    # -------------------------------------------------------------------------

    async def _phase_lca_calculation(
        self, input_data: CarbonFootprintInput,
    ) -> PhaseResult:
        """Apply emission factors and compute lifecycle emissions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Calculate material emissions
        for mat in self._materials:
            if mat.emissions_kgco2e <= 0 and mat.emission_factor_kgco2e_per_kg > 0:
                mat.emissions_kgco2e = round(
                    mat.mass_kg * mat.emission_factor_kgco2e_per_kg, 4
                )
                # Apply recycled content reduction
                if mat.recycled_content_pct > 0:
                    reduction_factor = 1.0 - (mat.recycled_content_pct / 100.0 * 0.5)
                    mat.emissions_kgco2e = round(
                        mat.emissions_kgco2e * reduction_factor, 4
                    )

        # Calculate energy emissions
        for ene in self._energy:
            if ene.emissions_kgco2e <= 0 and ene.emission_factor_kgco2e_per_kwh > 0:
                ene.emissions_kgco2e = round(
                    ene.consumption_kwh * ene.emission_factor_kgco2e_per_kwh, 4
                )

        # Calculate transport emissions
        for trn in self._transport:
            if trn.emissions_kgco2e <= 0 and trn.emission_factor_kgco2e_per_tkm > 0:
                trn.emissions_kgco2e = round(
                    trn.distance_km * trn.cargo_mass_tonnes
                    * trn.emission_factor_kgco2e_per_tkm, 4
                )

        # Aggregate by lifecycle stage
        stage_totals: Dict[str, float] = {}
        stage_counts: Dict[str, int] = {}
        all_records = [*self._materials, *self._energy, *self._transport]
        for rec in all_records:
            stage_key = rec.lifecycle_stage.value
            stage_totals[stage_key] = stage_totals.get(stage_key, 0.0) + rec.emissions_kgco2e
            stage_counts[stage_key] = stage_counts.get(stage_key, 0) + 1

        gross_total = sum(stage_totals.values())
        # Apply recycling credit (end-of-life avoided burden)
        net_total = max(0.0, gross_total - input_data.recycling_credit_kgco2e)

        self._total_kgco2e = net_total
        self._lifecycle_breakdown = []
        for stage_val in LifecycleStage:
            stage_em = stage_totals.get(stage_val.value, 0.0)
            share = (stage_em / gross_total * 100.0) if gross_total > 0 else 0.0
            self._lifecycle_breakdown.append(LifecycleStageResult(
                stage=stage_val.value,
                emissions_kgco2e=round(stage_em, 2),
                share_pct=round(share, 1),
                record_count=stage_counts.get(stage_val.value, 0),
            ))

        # Calculate per-kWh figure
        if input_data.battery_capacity_kwh > 0:
            self._per_kwh = round(
                net_total / input_data.battery_capacity_kwh, 2
            )
        else:
            self._per_kwh = 0.0
            warnings.append("Battery capacity is zero; cannot compute per-kWh footprint")

        outputs["gross_total_kgco2e"] = round(gross_total, 2)
        outputs["recycling_credit_kgco2e"] = round(input_data.recycling_credit_kgco2e, 2)
        outputs["net_total_kgco2e"] = round(net_total, 2)
        outputs["per_kwh_kgco2e"] = self._per_kwh
        outputs["material_emissions_kgco2e"] = round(
            sum(m.emissions_kgco2e for m in self._materials), 2
        )
        outputs["energy_emissions_kgco2e"] = round(
            sum(e.emissions_kgco2e for e in self._energy), 2
        )
        outputs["transport_emissions_kgco2e"] = round(
            sum(t.emissions_kgco2e for t in self._transport), 2
        )
        outputs["lifecycle_stages_populated"] = sum(
            1 for s in self._lifecycle_breakdown if s.emissions_kgco2e > 0
        )

        missing_ef_count = sum(
            1 for rec in all_records if rec.emissions_kgco2e <= 0
        )
        if missing_ef_count > 0:
            warnings.append(
                f"{missing_ef_count} records have zero emissions (missing EF or data)"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 LCACalculation: %.2f kgCO2e gross, %.2f net, %.2f per kWh",
            gross_total, net_total, self._per_kwh,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.LCA_CALCULATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Performance Class
    # -------------------------------------------------------------------------

    async def _phase_performance_class(
        self, input_data: CarbonFootprintInput,
    ) -> PhaseResult:
        """Assign performance class (A-E) based on kgCO2e/kWh thresholds."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # Select threshold table by battery category
        if input_data.battery_category in (
            BatteryCategory.EV_BATTERY, BatteryCategory.LMT_BATTERY
        ):
            thresholds = PERFORMANCE_CLASS_THRESHOLDS_EV
        else:
            thresholds = PERFORMANCE_CLASS_THRESHOLDS_INDUSTRIAL

        per_kwh = self._per_kwh
        if per_kwh <= thresholds["A"]:
            self._performance_class = PerformanceClassGrade.CLASS_A.value
        elif per_kwh <= thresholds["B"]:
            self._performance_class = PerformanceClassGrade.CLASS_B.value
        elif per_kwh <= thresholds["C"]:
            self._performance_class = PerformanceClassGrade.CLASS_C.value
        elif per_kwh <= thresholds["D"]:
            self._performance_class = PerformanceClassGrade.CLASS_D.value
        else:
            self._performance_class = PerformanceClassGrade.CLASS_E.value

        # Compute distance to next better class
        distance_to_better: Optional[float] = None
        if self._performance_class == "B":
            distance_to_better = round(per_kwh - thresholds["A"], 2)
        elif self._performance_class == "C":
            distance_to_better = round(per_kwh - thresholds["B"], 2)
        elif self._performance_class == "D":
            distance_to_better = round(per_kwh - thresholds["C"], 2)
        elif self._performance_class == "E":
            distance_to_better = round(per_kwh - thresholds["D"], 2)

        outputs["carbon_footprint_per_kwh"] = per_kwh
        outputs["performance_class"] = self._performance_class
        outputs["battery_category"] = input_data.battery_category.value
        outputs["thresholds_applied"] = thresholds
        outputs["distance_to_better_class_kgco2e"] = distance_to_better

        if self._performance_class in ("D", "E"):
            warnings.append(
                f"Performance class {self._performance_class}: carbon footprint "
                f"exceeds recommended thresholds for {input_data.battery_category.value}"
            )

        if per_kwh <= 0:
            warnings.append(
                "Carbon footprint per kWh is zero; verify input data completeness"
            )

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 PerformanceClass: %.2f kgCO2e/kWh -> class %s",
            per_kwh, self._performance_class,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.PERFORMANCE_CLASS.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Declaration Generation
    # -------------------------------------------------------------------------

    async def _phase_declaration_generation(
        self, input_data: CarbonFootprintInput,
    ) -> PhaseResult:
        """Generate EU-compliant carbon footprint declaration."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._declaration = CarbonFootprintDeclaration(
            battery_id=input_data.battery_id,
            battery_model=input_data.battery_model,
            battery_category=input_data.battery_category.value,
            battery_chemistry=input_data.battery_chemistry.value,
            capacity_kwh=input_data.battery_capacity_kwh,
            total_carbon_footprint_kgco2e=round(self._total_kgco2e, 2),
            carbon_footprint_per_kwh=round(self._per_kwh, 2),
            performance_class=self._performance_class,
            lifecycle_stages=self._lifecycle_breakdown,
            recycling_credit_kgco2e=round(input_data.recycling_credit_kgco2e, 2),
            functional_unit=input_data.functional_unit,
            reporting_year=input_data.reporting_year,
            issued_at=utcnow().isoformat(),
        )

        # Validate declaration completeness
        completeness_checks = {
            "battery_id": bool(self._declaration.battery_id),
            "capacity_kwh": self._declaration.capacity_kwh > 0,
            "total_footprint": self._declaration.total_carbon_footprint_kgco2e > 0,
            "performance_class": bool(self._declaration.performance_class),
            "lifecycle_stages": len(self._declaration.lifecycle_stages) > 0,
        }

        passed = sum(1 for v in completeness_checks.values() if v)
        total_checks = len(completeness_checks)
        completeness_pct = round(passed / total_checks * 100, 1)

        failed_checks = [k for k, v in completeness_checks.items() if not v]
        if failed_checks:
            warnings.append(
                f"Declaration incomplete: missing {', '.join(failed_checks)}"
            )

        outputs["declaration_id"] = self._declaration.declaration_id
        outputs["declaration_complete"] = completeness_pct == 100.0
        outputs["completeness_pct"] = completeness_pct
        outputs["completeness_checks"] = completeness_checks
        outputs["total_carbon_footprint_kgco2e"] = self._declaration.total_carbon_footprint_kgco2e
        outputs["performance_class"] = self._declaration.performance_class
        outputs["regulation_reference"] = self._declaration.regulation_reference

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 DeclarationGeneration: %s issued, completeness %.1f%%",
            self._declaration.declaration_id, completeness_pct,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DECLARATION_GENERATION.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CarbonFootprintResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
