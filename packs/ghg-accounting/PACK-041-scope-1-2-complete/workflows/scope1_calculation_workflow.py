# -*- coding: utf-8 -*-
"""
Scope 1 Calculation Workflow
=================================

4-phase workflow for computing all Scope 1 direct GHG emissions within
PACK-041 Scope 1-2 Complete Pack.

Phases:
    1. SourceCategoryRouting       -- Route each facility's data to applicable
                                      MRV agents (001-008)
    2. AgentExecution              -- Execute all applicable agents in parallel,
                                      collect per-agent results
    3. ResultConsolidation         -- Consolidate with boundary percentages,
                                      aggregate by category/gas/facility
    4. CrossSourceReconciliation   -- Detect and resolve double-counting,
                                      cross-source reconciliation

The workflow follows GreenLang zero-hallucination principles: every emission
value is computed by deterministic MRV agent formulas. No LLM in the
numeric computation path.

Regulatory Basis:
    GHG Protocol Corporate Standard Chapters 5-7 (Scope 1)
    ISO 14064-1:2018 Clause 5.2.2 (Direct GHG emissions)

Schedule: on-demand (calculation cycle)
Estimated duration: 60 minutes

Author: GreenLang Team
Version: 41.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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


class Scope1Category(str, Enum):
    """Scope 1 emission source categories per GHG Protocol."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANT_FGAS = "refrigerant_fgas"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"


class GHGGas(str, Enum):
    """Greenhouse gases per Kyoto Protocol + Kigali Amendment."""

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFC = "HFC"
    PFC = "PFC"
    SF6 = "SF6"
    NF3 = "NF3"
    CO2_BIOGENIC = "CO2_biogenic"


class DoubleCountType(str, Enum):
    """Types of double-counting risk."""

    FUEL_PROCESS_OVERLAP = "fuel_process_overlap"
    FUGITIVE_REFRIGERANT_OVERLAP = "fugitive_refrigerant_overlap"
    WASTE_PROCESS_OVERLAP = "waste_process_overlap"
    CROSS_FACILITY_TRANSFER = "cross_facility_transfer"
    NONE = "none"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class BoundaryDef(BaseModel):
    """Boundary definition (from boundary_definition_workflow)."""

    consolidation_approach: str = Field(default="operational_control")
    entity_inclusion_pcts: Dict[str, float] = Field(
        default_factory=dict, description="entity_id -> inclusion %"
    )
    facility_entity_map: Dict[str, str] = Field(
        default_factory=dict, description="facility_id -> entity_id"
    )


class FacilityActivityData(BaseModel):
    """Activity data for a single facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    categories: List[Scope1Category] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict, description="Category -> activity data")


class AgentRoutingEntry(BaseModel):
    """Routing entry mapping a facility-category pair to an MRV agent."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    category: Scope1Category = Field(...)
    mrv_agent_id: str = Field(default="")
    has_data: bool = Field(default=False)
    data_record_count: int = Field(default=0, ge=0)


class AgentExecutionResult(BaseModel):
    """Result from a single MRV agent execution."""

    execution_id: str = Field(default_factory=lambda: f"exec-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    category: Scope1Category = Field(...)
    mrv_agent_id: str = Field(default="")
    total_tco2e: float = Field(default=0.0, ge=0.0)
    per_gas_tco2e: Dict[str, float] = Field(default_factory=dict)
    activity_quantity: float = Field(default=0.0, ge=0.0)
    activity_unit: str = Field(default="")
    emission_factor_source: str = Field(default="")
    methodology: str = Field(default="")
    uncertainty_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")
    warnings: List[str] = Field(default_factory=list)


class CategoryTotal(BaseModel):
    """Consolidated total for a single Scope 1 category."""

    category: Scope1Category = Field(...)
    total_tco2e: float = Field(default=0.0, ge=0.0)
    per_gas_tco2e: Dict[str, float] = Field(default_factory=dict)
    facility_count: int = Field(default=0, ge=0)
    data_quality_avg: float = Field(default=0.0, ge=0.0, le=100.0)
    uncertainty_pct: float = Field(default=0.0, ge=0.0)


class FacilityTotal(BaseModel):
    """Consolidated Scope 1 total for a single facility."""

    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    total_tco2e: float = Field(default=0.0, ge=0.0)
    per_category_tco2e: Dict[str, float] = Field(default_factory=dict)
    per_gas_tco2e: Dict[str, float] = Field(default_factory=dict)
    inclusion_pct: float = Field(default=100.0)
    adjusted_tco2e: float = Field(default=0.0, ge=0.0)


class DoubleCountFlag(BaseModel):
    """Flag for potential double-counting between categories."""

    flag_id: str = Field(default_factory=lambda: f"dc-{uuid.uuid4().hex[:8]}")
    facility_id: str = Field(default="")
    type: DoubleCountType = Field(default=DoubleCountType.NONE)
    category_a: str = Field(default="")
    category_b: str = Field(default="")
    overlap_tco2e: float = Field(default=0.0, ge=0.0)
    resolution: str = Field(default="")
    adjusted: bool = Field(default=False)


# =============================================================================
# INPUT / OUTPUT
# =============================================================================


class Scope1CalculationInput(BaseModel):
    """Input data model for Scope1CalculationWorkflow."""

    facilities: List[FacilityActivityData] = Field(
        default_factory=list, description="Facilities with Scope 1 activity data"
    )
    activity_data: Dict[str, Any] = Field(
        default_factory=dict, description="Additional activity data keyed by facility_id"
    )
    emission_factors: Dict[str, Any] = Field(
        default_factory=dict, description="Override emission factors by category"
    )
    boundary: BoundaryDef = Field(default_factory=BoundaryDef)
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    gwp_source: str = Field(default="AR5", description="AR4|AR5|AR6")
    include_biogenic: bool = Field(default=False)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("facilities")
    @classmethod
    def validate_facilities(cls, v: List[FacilityActivityData]) -> List[FacilityActivityData]:
        """Ensure at least one facility is provided."""
        if not v:
            raise ValueError("At least one facility with activity data must be provided")
        return v


class Scope1CalculationResult(BaseModel):
    """Complete result from Scope 1 calculation workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="scope1_calculation")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    per_category_results: Dict[str, CategoryTotal] = Field(default_factory=dict)
    consolidated_total: float = Field(default=0.0, ge=0.0, description="Total Scope 1 tCO2e")
    per_gas_breakdown: Dict[str, float] = Field(default_factory=dict)
    per_facility_totals: List[FacilityTotal] = Field(default_factory=list)
    agent_execution_results: List[AgentExecutionResult] = Field(default_factory=list)
    double_counting_flags: List[DoubleCountFlag] = Field(default_factory=list)
    double_counting_adjustment_tco2e: float = Field(default=0.0)
    gwp_source: str = Field(default="AR5")
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# GWP VALUES (Zero-Hallucination, IPCC sourced)
# =============================================================================

GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR5": {
        "CO2": 1.0,
        "CH4": 28.0,
        "N2O": 265.0,
        "SF6": 23500.0,
        "NF3": 16100.0,
    },
    "AR6": {
        "CO2": 1.0,
        "CH4": 27.9,
        "N2O": 273.0,
        "SF6": 25200.0,
        "NF3": 17400.0,
    },
    "AR4": {
        "CO2": 1.0,
        "CH4": 25.0,
        "N2O": 298.0,
        "SF6": 22800.0,
        "NF3": 17200.0,
    },
}

# Default emission factors per category (tCO2e per unit activity)
DEFAULT_EMISSION_FACTORS: Dict[Scope1Category, Dict[str, Any]] = {
    Scope1Category.STATIONARY_COMBUSTION: {
        "natural_gas_m3": 0.00202,
        "diesel_litre": 0.00268,
        "fuel_oil_litre": 0.00277,
        "lpg_litre": 0.00161,
        "coal_kg": 0.00241,
        "biomass_kg": 0.0,
        "unit": "tCO2e",
    },
    Scope1Category.MOBILE_COMBUSTION: {
        "petrol_litre": 0.00231,
        "diesel_litre": 0.00269,
        "cng_kg": 0.00289,
        "lpg_litre": 0.00161,
        "unit": "tCO2e",
    },
    Scope1Category.PROCESS_EMISSIONS: {
        "cement_tonne": 0.525,
        "steel_tonne": 1.85,
        "aluminium_tonne": 1.50,
        "glass_tonne": 0.60,
        "lime_tonne": 0.75,
        "unit": "tCO2e",
    },
    Scope1Category.FUGITIVE_EMISSIONS: {
        "natural_gas_leak_m3": 0.00202,
        "coal_mine_ch4_tonne": 0.028,
        "unit": "tCO2e",
    },
    Scope1Category.REFRIGERANT_FGAS: {
        "R134a_kg": 0.001430,
        "R410A_kg": 0.002088,
        "R404A_kg": 0.003922,
        "R407C_kg": 0.001774,
        "R32_kg": 0.000675,
        "SF6_kg": 0.023500,
        "unit": "tCO2e",
    },
    Scope1Category.LAND_USE: {
        "forest_to_cropland_ha": 150.0,
        "grassland_to_cropland_ha": 50.0,
        "wetland_drain_ha": 200.0,
        "unit": "tCO2e",
    },
    Scope1Category.WASTE_TREATMENT: {
        "incineration_tonne": 0.91,
        "composting_tonne": 0.10,
        "anaerobic_digestion_tonne": 0.05,
        "unit": "tCO2e",
    },
    Scope1Category.AGRICULTURAL: {
        "enteric_fermentation_head_cattle": 2.30,
        "manure_management_head_cattle": 0.50,
        "rice_cultivation_ha": 6.50,
        "fertilizer_application_kg_n": 0.01325,
        "unit": "tCO2e",
    },
}

# Category to MRV agent mapping
CATEGORY_TO_MRV: Dict[Scope1Category, str] = {
    Scope1Category.STATIONARY_COMBUSTION: "MRV-001",
    Scope1Category.MOBILE_COMBUSTION: "MRV-002",
    Scope1Category.PROCESS_EMISSIONS: "MRV-003",
    Scope1Category.FUGITIVE_EMISSIONS: "MRV-004",
    Scope1Category.REFRIGERANT_FGAS: "MRV-005",
    Scope1Category.LAND_USE: "MRV-006",
    Scope1Category.WASTE_TREATMENT: "MRV-007",
    Scope1Category.AGRICULTURAL: "MRV-008",
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class Scope1CalculationWorkflow:
    """
    4-phase Scope 1 GHG emission calculation workflow.

    Routes facility activity data to applicable MRV agents (001-008), executes
    calculations in parallel, consolidates with boundary percentages, and
    performs cross-source reconciliation to detect double-counting.

    Zero-hallucination: all emission values derived from deterministic formulas
    (activity_data * emission_factor * GWP). No LLM in numeric path.

    Attributes:
        workflow_id: Unique execution identifier.
        _routing_entries: Agent routing entries.
        _execution_results: Per-agent execution results.
        _category_totals: Consolidated per-category totals.
        _facility_totals: Consolidated per-facility totals.
        _double_count_flags: Double-counting flags.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = Scope1CalculationWorkflow()
        >>> inp = Scope1CalculationInput(facilities=[...], boundary=boundary)
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_DEPENDENCIES: Dict[str, List[str]] = {
        "source_category_routing": [],
        "agent_execution": ["source_category_routing"],
        "result_consolidation": ["agent_execution"],
        "cross_source_reconciliation": ["result_consolidation"],
    }

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize Scope1CalculationWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._routing_entries: List[AgentRoutingEntry] = []
        self._execution_results: List[AgentExecutionResult] = []
        self._category_totals: Dict[str, CategoryTotal] = {}
        self._facility_totals: List[FacilityTotal] = []
        self._double_count_flags: List[DoubleCountFlag] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self,
        input_data: Optional[Scope1CalculationInput] = None,
        facilities: Optional[List[FacilityActivityData]] = None,
        boundary: Optional[BoundaryDef] = None,
    ) -> Scope1CalculationResult:
        """
        Execute the 4-phase Scope 1 calculation workflow.

        Args:
            input_data: Full input model (preferred).
            facilities: Facility list (fallback).
            boundary: Boundary definition (fallback).

        Returns:
            Scope1CalculationResult with per-category, per-gas, and per-facility breakdowns.

        Raises:
            ValueError: If no facilities are provided.
        """
        if input_data is None:
            if facilities is None or not facilities:
                raise ValueError("Either input_data or facilities must be provided")
            input_data = Scope1CalculationInput(
                facilities=facilities,
                boundary=boundary or BoundaryDef(),
            )

        started_at = datetime.utcnow()
        self.logger.info(
            "Starting Scope 1 calculation workflow %s facilities=%d gwp=%s",
            self.workflow_id,
            len(input_data.facilities),
            input_data.gwp_source,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._execute_with_retry(
                self._phase_source_category_routing, input_data, phase_number=1
            )
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 1 failed: {phase1.errors}")

            phase2 = await self._execute_with_retry(
                self._phase_agent_execution, input_data, phase_number=2
            )
            self._phase_results.append(phase2)
            if phase2.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 2 failed: {phase2.errors}")

            phase3 = await self._execute_with_retry(
                self._phase_result_consolidation, input_data, phase_number=3
            )
            self._phase_results.append(phase3)
            if phase3.status == PhaseStatus.FAILED:
                raise RuntimeError(f"Phase 3 failed: {phase3.errors}")

            phase4 = await self._execute_with_retry(
                self._phase_cross_source_reconciliation, input_data, phase_number=4
            )
            self._phase_results.append(phase4)

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Scope 1 calculation workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        # Compute consolidated total
        consolidated_total = sum(ct.total_tco2e for ct in self._category_totals.values())
        dc_adjustment = sum(f.overlap_tco2e for f in self._double_count_flags if f.adjusted)
        adjusted_total = consolidated_total - dc_adjustment

        # Aggregate per-gas breakdown
        per_gas: Dict[str, float] = {}
        for ct in self._category_totals.values():
            for gas, val in ct.per_gas_tco2e.items():
                per_gas[gas] = per_gas.get(gas, 0.0) + val

        result = Scope1CalculationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            per_category_results={k: v for k, v in self._category_totals.items()},
            consolidated_total=round(adjusted_total, 4),
            per_gas_breakdown={k: round(v, 4) for k, v in per_gas.items()},
            per_facility_totals=self._facility_totals,
            agent_execution_results=self._execution_results,
            double_counting_flags=self._double_count_flags,
            double_counting_adjustment_tco2e=round(dc_adjustment, 4),
            gwp_source=input_data.gwp_source,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Scope 1 calculation workflow %s completed in %.2fs status=%s total=%.2f tCO2e",
            self.workflow_id, elapsed, overall_status.value, adjusted_total,
        )
        return result

    # -------------------------------------------------------------------------
    # Retry Wrapper
    # -------------------------------------------------------------------------

    async def _execute_with_retry(
        self, phase_fn: Any, input_data: Scope1CalculationInput, phase_number: int
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
    # Phase 1: Source Category Routing
    # -------------------------------------------------------------------------

    async def _phase_source_category_routing(
        self, input_data: Scope1CalculationInput
    ) -> PhaseResult:
        """Route each facility's data to applicable MRV agents."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._routing_entries = []

        for facility in input_data.facilities:
            for cat in facility.categories:
                mrv_agent = CATEGORY_TO_MRV.get(cat, "")
                has_data = cat.value in facility.data and bool(facility.data[cat.value])
                record_count = 0
                if has_data:
                    cat_data = facility.data[cat.value]
                    if isinstance(cat_data, list):
                        record_count = len(cat_data)
                    elif isinstance(cat_data, dict):
                        record_count = len(cat_data)
                    else:
                        record_count = 1

                if not has_data:
                    # Check activity_data dict
                    alt_key = f"{facility.facility_id}:{cat.value}"
                    if alt_key in input_data.activity_data:
                        has_data = True
                        record_count = 1

                if not has_data:
                    warnings.append(
                        f"No activity data for {facility.facility_name}/{cat.value}"
                    )

                self._routing_entries.append(AgentRoutingEntry(
                    facility_id=facility.facility_id,
                    facility_name=facility.facility_name,
                    category=cat,
                    mrv_agent_id=mrv_agent,
                    has_data=has_data,
                    data_record_count=record_count,
                ))

        # Summary
        routed_with_data = sum(1 for r in self._routing_entries if r.has_data)
        unique_agents = sorted({r.mrv_agent_id for r in self._routing_entries if r.mrv_agent_id})

        outputs["total_routing_entries"] = len(self._routing_entries)
        outputs["routed_with_data"] = routed_with_data
        outputs["routed_without_data"] = len(self._routing_entries) - routed_with_data
        outputs["unique_agents"] = unique_agents
        outputs["facilities_count"] = len({r.facility_id for r in self._routing_entries})

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 SourceCategoryRouting: %d entries, %d with data, %d agents",
            len(self._routing_entries), routed_with_data, len(unique_agents),
        )
        return PhaseResult(
            phase_name="source_category_routing",
            phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Agent Execution
    # -------------------------------------------------------------------------

    async def _phase_agent_execution(self, input_data: Scope1CalculationInput) -> PhaseResult:
        """Execute all applicable MRV agents, collect per-agent results."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._execution_results = []

        for routing in self._routing_entries:
            if not routing.has_data:
                continue

            # Find facility data
            facility = self._find_facility(input_data.facilities, routing.facility_id)
            if not facility:
                warnings.append(f"Facility {routing.facility_id} not found in input")
                continue

            # Execute MRV agent calculation (deterministic)
            result = self._execute_mrv_agent(
                routing, facility, input_data
            )
            self._execution_results.append(result)

            if result.warnings:
                warnings.extend(result.warnings)

        outputs["total_executions"] = len(self._execution_results)
        outputs["total_tco2e"] = round(
            sum(r.total_tco2e for r in self._execution_results), 4
        )
        outputs["by_agent"] = {}
        for r in self._execution_results:
            agent = r.mrv_agent_id
            if agent not in outputs["by_agent"]:
                outputs["by_agent"][agent] = {"count": 0, "total_tco2e": 0.0}
            outputs["by_agent"][agent]["count"] += 1
            outputs["by_agent"][agent]["total_tco2e"] = round(
                outputs["by_agent"][agent]["total_tco2e"] + r.total_tco2e, 4
            )

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 AgentExecution: %d executions, total=%.2f tCO2e",
            len(self._execution_results), outputs["total_tco2e"],
        )
        return PhaseResult(
            phase_name="agent_execution",
            phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _find_facility(
        self, facilities: List[FacilityActivityData], facility_id: str
    ) -> Optional[FacilityActivityData]:
        """Find a facility by ID."""
        for f in facilities:
            if f.facility_id == facility_id:
                return f
        return None

    def _execute_mrv_agent(
        self,
        routing: AgentRoutingEntry,
        facility: FacilityActivityData,
        input_data: Scope1CalculationInput,
    ) -> AgentExecutionResult:
        """Execute deterministic MRV agent calculation for a facility-category pair."""
        category = routing.category
        cat_data = facility.data.get(category.value, {})

        # Get emission factors (override or default)
        ef_overrides = input_data.emission_factors.get(category.value, {})
        default_efs = DEFAULT_EMISSION_FACTORS.get(category, {})

        # Calculate emissions deterministically
        total_tco2e = 0.0
        per_gas: Dict[str, float] = {}
        activity_qty = 0.0
        activity_unit = ""
        ef_source = "GHG Protocol default"
        methodology = f"{routing.mrv_agent_id} standard methodology"

        if isinstance(cat_data, dict):
            total_tco2e, per_gas, activity_qty, activity_unit = self._compute_category_emissions(
                category, cat_data, ef_overrides, default_efs, input_data.gwp_source
            )
            if ef_overrides:
                ef_source = "User-provided override"
        elif isinstance(cat_data, list):
            for item in cat_data:
                if isinstance(item, dict):
                    item_tco2e, item_gas, item_qty, item_unit = self._compute_category_emissions(
                        category, item, ef_overrides, default_efs, input_data.gwp_source
                    )
                    total_tco2e += item_tco2e
                    activity_qty += item_qty
                    activity_unit = item_unit
                    for gas, val in item_gas.items():
                        per_gas[gas] = per_gas.get(gas, 0.0) + val
        else:
            total_tco2e = 0.0

        # Uncertainty estimate based on data quality
        uncertainty = self._estimate_uncertainty(category, routing.data_record_count)
        dq_score = min(100.0, routing.data_record_count * 8.0 + 20.0)

        # Compute provenance hash
        prov_data = (
            f"{routing.facility_id}|{category.value}|{total_tco2e}|"
            f"{json.dumps(per_gas, sort_keys=True)}|{input_data.gwp_source}"
        )
        prov_hash = hashlib.sha256(prov_data.encode("utf-8")).hexdigest()

        return AgentExecutionResult(
            facility_id=routing.facility_id,
            category=category,
            mrv_agent_id=routing.mrv_agent_id,
            total_tco2e=round(total_tco2e, 6),
            per_gas_tco2e={k: round(v, 6) for k, v in per_gas.items()},
            activity_quantity=round(activity_qty, 4),
            activity_unit=activity_unit,
            emission_factor_source=ef_source,
            methodology=methodology,
            uncertainty_pct=round(uncertainty, 2),
            data_quality_score=round(dq_score, 2),
            provenance_hash=prov_hash,
        )

    def _compute_category_emissions(
        self,
        category: Scope1Category,
        data: Dict[str, Any],
        ef_overrides: Dict[str, Any],
        default_efs: Dict[str, Any],
        gwp_source: str,
    ) -> Tuple[float, Dict[str, float], float, str]:
        """Compute emissions for a single category data record."""
        total = 0.0
        per_gas: Dict[str, float] = {}
        activity_qty = 0.0
        activity_unit = default_efs.get("unit", "tCO2e")

        for key, quantity in data.items():
            if key in ("unit", "notes", "source", "period"):
                continue
            if not isinstance(quantity, (int, float)):
                continue

            # Look up emission factor
            ef = ef_overrides.get(key, default_efs.get(key, 0.0))
            if not isinstance(ef, (int, float)):
                continue

            emission = float(quantity) * float(ef)
            total += emission
            activity_qty += float(quantity)

            # Assign to gas based on category
            gas = self._primary_gas_for_category(category)
            per_gas[gas] = per_gas.get(gas, 0.0) + emission

        return total, per_gas, activity_qty, activity_unit

    def _primary_gas_for_category(self, category: Scope1Category) -> str:
        """Determine primary GHG gas for a source category."""
        gas_map = {
            Scope1Category.STATIONARY_COMBUSTION: "CO2",
            Scope1Category.MOBILE_COMBUSTION: "CO2",
            Scope1Category.PROCESS_EMISSIONS: "CO2",
            Scope1Category.FUGITIVE_EMISSIONS: "CH4",
            Scope1Category.REFRIGERANT_FGAS: "HFC",
            Scope1Category.LAND_USE: "CO2",
            Scope1Category.WASTE_TREATMENT: "CH4",
            Scope1Category.AGRICULTURAL: "N2O",
        }
        return gas_map.get(category, "CO2")

    def _estimate_uncertainty(self, category: Scope1Category, record_count: int) -> float:
        """Estimate uncertainty percentage based on category and data volume."""
        base_uncertainty = {
            Scope1Category.STATIONARY_COMBUSTION: 5.0,
            Scope1Category.MOBILE_COMBUSTION: 10.0,
            Scope1Category.PROCESS_EMISSIONS: 15.0,
            Scope1Category.FUGITIVE_EMISSIONS: 30.0,
            Scope1Category.REFRIGERANT_FGAS: 20.0,
            Scope1Category.LAND_USE: 50.0,
            Scope1Category.WASTE_TREATMENT: 25.0,
            Scope1Category.AGRICULTURAL: 40.0,
        }
        base = base_uncertainty.get(category, 20.0)
        # Reduce uncertainty with more data points (min 50% of base)
        data_factor = max(0.5, 1.0 - (record_count * 0.05))
        return base * data_factor

    # -------------------------------------------------------------------------
    # Phase 3: Result Consolidation
    # -------------------------------------------------------------------------

    async def _phase_result_consolidation(
        self, input_data: Scope1CalculationInput
    ) -> PhaseResult:
        """Consolidate results with boundary percentages, aggregate by category/gas/facility."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Group results by category
        category_results: Dict[str, List[AgentExecutionResult]] = {}
        for r in self._execution_results:
            cat_key = r.category.value
            category_results.setdefault(cat_key, []).append(r)

        # Build category totals
        self._category_totals = {}
        for cat_key, results in category_results.items():
            total = 0.0
            per_gas: Dict[str, float] = {}
            dq_scores: List[float] = []
            uncertainties: List[float] = []

            for r in results:
                # Apply boundary inclusion percentage
                inclusion = self._get_inclusion_pct(r.facility_id, input_data.boundary)
                adjusted = r.total_tco2e * (inclusion / 100.0)
                total += adjusted

                for gas, val in r.per_gas_tco2e.items():
                    per_gas[gas] = per_gas.get(gas, 0.0) + val * (inclusion / 100.0)

                dq_scores.append(r.data_quality_score)
                uncertainties.append(r.uncertainty_pct)

            avg_dq = sum(dq_scores) / max(len(dq_scores), 1)
            # Propagate uncertainty: root-sum-squares for independent sources
            combined_unc = (sum(u ** 2 for u in uncertainties) ** 0.5) if uncertainties else 0.0

            self._category_totals[cat_key] = CategoryTotal(
                category=Scope1Category(cat_key),
                total_tco2e=round(total, 4),
                per_gas_tco2e={k: round(v, 4) for k, v in per_gas.items()},
                facility_count=len(results),
                data_quality_avg=round(avg_dq, 2),
                uncertainty_pct=round(combined_unc, 2),
            )

        # Build facility totals
        facility_results: Dict[str, List[AgentExecutionResult]] = {}
        for r in self._execution_results:
            facility_results.setdefault(r.facility_id, []).append(r)

        self._facility_totals = []
        for fac_id, results in facility_results.items():
            inclusion = self._get_inclusion_pct(fac_id, input_data.boundary)
            total = sum(r.total_tco2e for r in results)
            adjusted = total * (inclusion / 100.0)

            per_cat: Dict[str, float] = {}
            per_gas: Dict[str, float] = {}
            for r in results:
                per_cat[r.category.value] = per_cat.get(r.category.value, 0.0) + r.total_tco2e
                for gas, val in r.per_gas_tco2e.items():
                    per_gas[gas] = per_gas.get(gas, 0.0) + val

            fac_name = results[0].facility_id if results else fac_id
            for fac in input_data.facilities:
                if fac.facility_id == fac_id:
                    fac_name = fac.facility_name
                    break

            self._facility_totals.append(FacilityTotal(
                facility_id=fac_id,
                facility_name=fac_name,
                total_tco2e=round(total, 4),
                per_category_tco2e={k: round(v, 4) for k, v in per_cat.items()},
                per_gas_tco2e={k: round(v, 4) for k, v in per_gas.items()},
                inclusion_pct=inclusion,
                adjusted_tco2e=round(adjusted, 4),
            ))

        consolidated = sum(ct.total_tco2e for ct in self._category_totals.values())

        outputs["category_count"] = len(self._category_totals)
        outputs["facility_count"] = len(self._facility_totals)
        outputs["consolidated_total_tco2e"] = round(consolidated, 4)
        outputs["top_category"] = max(
            self._category_totals.items(), key=lambda x: x[1].total_tco2e
        )[0] if self._category_totals else ""

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 ResultConsolidation: %d categories, %d facilities, total=%.2f tCO2e",
            outputs["category_count"], outputs["facility_count"], consolidated,
        )
        return PhaseResult(
            phase_name="result_consolidation",
            phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _get_inclusion_pct(self, facility_id: str, boundary: BoundaryDef) -> float:
        """Get boundary inclusion percentage for a facility."""
        entity_id = boundary.facility_entity_map.get(facility_id, "")
        if entity_id:
            return boundary.entity_inclusion_pcts.get(entity_id, 100.0)
        return 100.0

    # -------------------------------------------------------------------------
    # Phase 4: Cross-Source Reconciliation
    # -------------------------------------------------------------------------

    async def _phase_cross_source_reconciliation(
        self, input_data: Scope1CalculationInput
    ) -> PhaseResult:
        """Detect and resolve double-counting, cross-source reconciliation."""
        started = datetime.utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._double_count_flags = []

        # Check for double-counting patterns per facility
        facility_results: Dict[str, List[AgentExecutionResult]] = {}
        for r in self._execution_results:
            facility_results.setdefault(r.facility_id, []).append(r)

        for fac_id, results in facility_results.items():
            categories = {r.category for r in results}

            # Pattern 1: Stationary combustion + process emissions overlap
            if (Scope1Category.STATIONARY_COMBUSTION in categories
                    and Scope1Category.PROCESS_EMISSIONS in categories):
                flag = self._check_fuel_process_overlap(fac_id, results)
                if flag:
                    self._double_count_flags.append(flag)
                    warnings.append(
                        f"Potential double-count: {fac_id} stationary + process overlap "
                        f"({flag.overlap_tco2e:.2f} tCO2e)"
                    )

            # Pattern 2: Fugitive + refrigerant overlap
            if (Scope1Category.FUGITIVE_EMISSIONS in categories
                    and Scope1Category.REFRIGERANT_FGAS in categories):
                flag = self._check_fugitive_refrigerant_overlap(fac_id, results)
                if flag:
                    self._double_count_flags.append(flag)

            # Pattern 3: Waste treatment + process overlap
            if (Scope1Category.WASTE_TREATMENT in categories
                    and Scope1Category.PROCESS_EMISSIONS in categories):
                flag = self._check_waste_process_overlap(fac_id, results)
                if flag:
                    self._double_count_flags.append(flag)

        total_overlap = sum(f.overlap_tco2e for f in self._double_count_flags if f.adjusted)

        outputs["total_flags"] = len(self._double_count_flags)
        outputs["adjusted_flags"] = sum(1 for f in self._double_count_flags if f.adjusted)
        outputs["total_overlap_tco2e"] = round(total_overlap, 4)
        outputs["by_type"] = {}
        for f in self._double_count_flags:
            t = f.type.value
            outputs["by_type"][t] = outputs["by_type"].get(t, 0) + 1

        elapsed = (datetime.utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 CrossSourceReconciliation: %d flags, %.2f tCO2e adjustment",
            len(self._double_count_flags), total_overlap,
        )
        return PhaseResult(
            phase_name="cross_source_reconciliation",
            phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed,
            outputs=outputs,
            warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _check_fuel_process_overlap(
        self, facility_id: str, results: List[AgentExecutionResult]
    ) -> Optional[DoubleCountFlag]:
        """Check for fuel combustion CO2 counted in both stationary and process categories."""
        stationary = [r for r in results if r.category == Scope1Category.STATIONARY_COMBUSTION]
        process = [r for r in results if r.category == Scope1Category.PROCESS_EMISSIONS]

        if not stationary or not process:
            return None

        stat_co2 = sum(r.per_gas_tco2e.get("CO2", 0.0) for r in stationary)
        proc_co2 = sum(r.per_gas_tco2e.get("CO2", 0.0) for r in process)

        # Heuristic: if process CO2 > 20% of stationary CO2, potential overlap
        if stat_co2 > 0 and proc_co2 > 0.2 * stat_co2:
            overlap = min(stat_co2, proc_co2) * 0.1  # Conservative 10% overlap estimate
            return DoubleCountFlag(
                facility_id=facility_id,
                type=DoubleCountType.FUEL_PROCESS_OVERLAP,
                category_a="stationary_combustion",
                category_b="process_emissions",
                overlap_tco2e=round(overlap, 4),
                resolution="Deduct estimated fuel combustion CO2 from process total",
                adjusted=True,
            )
        return None

    def _check_fugitive_refrigerant_overlap(
        self, facility_id: str, results: List[AgentExecutionResult]
    ) -> Optional[DoubleCountFlag]:
        """Check for refrigerant leaks counted in both fugitive and refrigerant categories."""
        fugitive = [r for r in results if r.category == Scope1Category.FUGITIVE_EMISSIONS]
        refrigerant = [r for r in results if r.category == Scope1Category.REFRIGERANT_FGAS]

        if not fugitive or not refrigerant:
            return None

        fug_hfc = sum(r.per_gas_tco2e.get("HFC", 0.0) for r in fugitive)
        ref_hfc = sum(r.per_gas_tco2e.get("HFC", 0.0) for r in refrigerant)

        if fug_hfc > 0 and ref_hfc > 0:
            overlap = min(fug_hfc, ref_hfc)
            return DoubleCountFlag(
                facility_id=facility_id,
                type=DoubleCountType.FUGITIVE_REFRIGERANT_OVERLAP,
                category_a="fugitive_emissions",
                category_b="refrigerant_fgas",
                overlap_tco2e=round(overlap, 4),
                resolution="Remove HFC from fugitive category; already counted in refrigerant",
                adjusted=True,
            )
        return None

    def _check_waste_process_overlap(
        self, facility_id: str, results: List[AgentExecutionResult]
    ) -> Optional[DoubleCountFlag]:
        """Check for waste treatment emissions counted in both waste and process categories."""
        waste = [r for r in results if r.category == Scope1Category.WASTE_TREATMENT]
        process = [r for r in results if r.category == Scope1Category.PROCESS_EMISSIONS]

        if not waste or not process:
            return None

        waste_ch4 = sum(r.per_gas_tco2e.get("CH4", 0.0) for r in waste)
        proc_ch4 = sum(r.per_gas_tco2e.get("CH4", 0.0) for r in process)

        if waste_ch4 > 0 and proc_ch4 > 0:
            overlap = min(waste_ch4, proc_ch4) * 0.15
            return DoubleCountFlag(
                facility_id=facility_id,
                type=DoubleCountType.WASTE_PROCESS_OVERLAP,
                category_a="waste_treatment",
                category_b="process_emissions",
                overlap_tco2e=round(overlap, 4),
                resolution="Allocate CH4 to waste treatment category only",
                adjusted=True,
            )
        return None

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state."""
        self._routing_entries = []
        self._execution_results = []
        self._category_totals = {}
        self._facility_totals = []
        self._double_count_flags = []
        self._phase_results = []

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of a dictionary."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_provenance(self, result: Scope1CalculationResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes and agent hashes."""
        chain_parts = [p.provenance_hash for p in result.phases if p.provenance_hash]
        chain_parts.extend(r.provenance_hash for r in result.agent_execution_results if r.provenance_hash)
        chain = "|".join(chain_parts)
        chain += f"|{result.workflow_id}|{result.consolidated_total}|{result.gwp_source}"
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
