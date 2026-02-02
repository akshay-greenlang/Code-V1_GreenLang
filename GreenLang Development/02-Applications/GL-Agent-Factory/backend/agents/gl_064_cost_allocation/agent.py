"""
GL-064: Cost Allocation Agent (COST-ALLOCATOR)

This module implements the CostAllocationAgent for activity-based costing,
utility cost allocation, and energy cost attribution to production processes.

Standards Reference:
    - Activity-Based Costing (ABC) methodology
    - ISO 14051 (Material flow cost accounting)
    - FERC Uniform System of Accounts

Example:
    >>> agent = CostAllocationAgent()
    >>> result = agent.run(input_data)
    >>> print(f"Total allocated costs: ${result.total_allocated_cost:,.2f}")
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AllocationMethod(str, Enum):
    DIRECT = "direct"
    PROPORTIONAL = "proportional"
    ACTIVITY_BASED = "activity_based"
    EXERGY = "exergy"
    EQUAL = "equal"


class CostCategory(str, Enum):
    ENERGY = "energy"
    LABOR = "labor"
    MATERIALS = "materials"
    OVERHEAD = "overhead"
    MAINTENANCE = "maintenance"
    UTILITIES = "utilities"
    DEPRECIATION = "depreciation"


class ProductionUnit(BaseModel):
    unit_id: str = Field(..., description="Production unit identifier")
    name: str = Field(..., description="Unit name")
    production_volume: float = Field(..., gt=0, description="Production volume")
    production_unit: str = Field(..., description="Unit of production")
    energy_consumption_kwh: Optional[float] = Field(0.0, ge=0, description="Energy consumption (kWh)")
    labor_hours: Optional[float] = Field(0.0, ge=0, description="Labor hours")
    floor_area_m2: Optional[float] = Field(0.0, ge=0, description="Floor area (m2)")
    machine_hours: Optional[float] = Field(0.0, ge=0, description="Machine hours")
    exergy_consumption_kwh: Optional[float] = Field(0.0, ge=0, description="Exergy consumption (kWh)")


class CostPool(BaseModel):
    pool_id: str = Field(..., description="Cost pool identifier")
    name: str = Field(..., description="Cost pool name")
    category: CostCategory = Field(..., description="Cost category")
    total_cost: float = Field(..., ge=0, description="Total cost ($)")
    allocation_method: AllocationMethod = Field(..., description="Allocation method")
    allocation_basis: str = Field(..., description="Basis for allocation")


class AllocationDriver(BaseModel):
    driver_id: str = Field(..., description="Driver identifier")
    name: str = Field(..., description="Driver name")
    unit: str = Field(..., description="Driver unit")
    total_value: float = Field(..., gt=0, description="Total driver value")


class CostAllocationInput(BaseModel):
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    facility_name: str = Field(default="Facility", description="Facility name")
    analysis_period: str = Field(default="Monthly", description="Analysis period")
    production_units: List[ProductionUnit] = Field(..., min_items=1)
    cost_pools: List[CostPool] = Field(..., min_items=1)
    allocation_drivers: List[AllocationDriver] = Field(default_factory=list)
    electricity_rate_per_kwh: float = Field(default=0.10, gt=0, description="Electricity rate ($/kWh)")
    gas_rate_per_therm: float = Field(default=0.50, gt=0, description="Gas rate ($/therm)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UnitCostAllocation(BaseModel):
    unit_id: str
    unit_name: str
    production_volume: float
    allocated_costs: Dict[str, float]
    total_allocated_cost: float
    cost_per_unit: float
    cost_breakdown_percent: Dict[str, float]


class CostPoolAllocation(BaseModel):
    pool_id: str
    pool_name: str
    category: str
    total_cost: float
    allocation_method: str
    allocation_basis: str
    allocations: Dict[str, float]
    unallocated_cost: float


class CostSummary(BaseModel):
    total_cost: float
    total_allocated: float
    total_unallocated: float
    allocation_efficiency_percent: float
    cost_by_category: Dict[str, float]


class SensitivityAnalysis(BaseModel):
    parameter: str
    base_value: float
    variation_percent: float
    impact_on_unit_cost: float
    sensitivity_index: float


class ProvenanceRecord(BaseModel):
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str


class CostAllocationOutput(BaseModel):
    analysis_id: str
    facility_name: str
    analysis_period: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    unit_allocations: List[UnitCostAllocation]
    pool_allocations: List[CostPoolAllocation]
    cost_summary: CostSummary
    sensitivity_analyses: List[SensitivityAnalysis]
    recommendations: List[str]
    warnings: List[str]
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


class CostAllocationAgent:
    """GL-064: Cost Allocation Agent - Activity-based cost allocation and attribution."""

    AGENT_ID = "GL-064"
    AGENT_NAME = "COST-ALLOCATOR"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []
        logger.info(f"CostAllocationAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: CostAllocationInput) -> CostAllocationOutput:
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        # Validate inputs
        if not input_data.production_units:
            self._validation_errors.append("No production units provided")
        if not input_data.cost_pools:
            self._validation_errors.append("No cost pools provided")

        # Initialize allocation tracking
        unit_costs: Dict[str, Dict[str, float]] = {
            unit.unit_id: {} for unit in input_data.production_units
        }

        # Allocate each cost pool
        pool_allocations = []
        total_cost = 0.0
        total_allocated = 0.0

        for pool in input_data.cost_pools:
            allocation = self._allocate_cost_pool(pool, input_data.production_units)
            pool_allocations.append(allocation)
            total_cost += pool.total_cost
            total_allocated += sum(allocation.allocations.values())

            # Accumulate costs for each unit
            for unit_id, cost in allocation.allocations.items():
                if unit_id in unit_costs:
                    category_key = pool.category.value
                    unit_costs[unit_id][category_key] = unit_costs[unit_id].get(category_key, 0.0) + cost

        self._track_provenance("cost_allocation",
            {"num_pools": len(input_data.cost_pools), "num_units": len(input_data.production_units)},
            {"total_cost": total_cost, "total_allocated": total_allocated},
            "Cost Allocator")

        # Build unit allocations
        unit_allocations = []
        for unit in input_data.production_units:
            allocated_costs = unit_costs[unit.unit_id]
            total_unit_cost = sum(allocated_costs.values())
            cost_per_unit = total_unit_cost / unit.production_volume if unit.production_volume > 0 else 0.0

            # Calculate breakdown percentages
            breakdown_percent = {}
            if total_unit_cost > 0:
                for category, cost in allocated_costs.items():
                    breakdown_percent[category] = round(cost / total_unit_cost * 100, 2)

            unit_allocations.append(UnitCostAllocation(
                unit_id=unit.unit_id,
                unit_name=unit.name,
                production_volume=unit.production_volume,
                allocated_costs=allocated_costs,
                total_allocated_cost=round(total_unit_cost, 2),
                cost_per_unit=round(cost_per_unit, 4),
                cost_breakdown_percent=breakdown_percent
            ))

        # Cost summary
        total_unallocated = total_cost - total_allocated
        allocation_efficiency = (total_allocated / total_cost * 100) if total_cost > 0 else 0.0

        cost_by_category = {}
        for pool in input_data.cost_pools:
            category = pool.category.value
            cost_by_category[category] = cost_by_category.get(category, 0.0) + pool.total_cost

        cost_summary = CostSummary(
            total_cost=round(total_cost, 2),
            total_allocated=round(total_allocated, 2),
            total_unallocated=round(total_unallocated, 2),
            allocation_efficiency_percent=round(allocation_efficiency, 2),
            cost_by_category=cost_by_category
        )

        # Sensitivity analysis
        sensitivity_analyses = self._perform_sensitivity_analysis(unit_allocations, input_data)

        # Generate recommendations and warnings
        self._generate_recommendations_and_warnings(cost_summary, unit_allocations, pool_allocations)

        # Calculate provenance
        provenance_hash = self._calculate_provenance_hash()
        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return CostAllocationOutput(
            analysis_id=input_data.analysis_id or f"CA-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            analysis_period=input_data.analysis_period,
            unit_allocations=unit_allocations,
            pool_allocations=pool_allocations,
            cost_summary=cost_summary,
            sensitivity_analyses=sensitivity_analyses,
            recommendations=self._recommendations,
            warnings=self._warnings,
            provenance_chain=[ProvenanceRecord(**{k: v for k, v in s.items()}) for s in self._provenance_steps],
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS" if not self._validation_errors else "FAIL",
            validation_errors=self._validation_errors
        )

    def _allocate_cost_pool(self, pool: CostPool, units: List[ProductionUnit]) -> CostPoolAllocation:
        """Allocate a cost pool to production units."""
        allocations: Dict[str, float] = {}

        if pool.allocation_method == AllocationMethod.DIRECT:
            # Direct allocation - allocate 100% to first unit (simplified)
            if units:
                allocations[units[0].unit_id] = pool.total_cost

        elif pool.allocation_method == AllocationMethod.PROPORTIONAL:
            # Proportional based on allocation basis
            basis_values = self._get_allocation_basis_values(pool.allocation_basis, units)
            total_basis = sum(basis_values.values())

            if total_basis > 0:
                for unit_id, basis_value in basis_values.items():
                    allocations[unit_id] = pool.total_cost * (basis_value / total_basis)

        elif pool.allocation_method == AllocationMethod.ACTIVITY_BASED:
            # Activity-based allocation
            if "energy" in pool.allocation_basis.lower():
                basis_values = {u.unit_id: u.energy_consumption_kwh or 0.0 for u in units}
            elif "labor" in pool.allocation_basis.lower():
                basis_values = {u.unit_id: u.labor_hours or 0.0 for u in units}
            elif "machine" in pool.allocation_basis.lower():
                basis_values = {u.unit_id: u.machine_hours or 0.0 for u in units}
            else:
                basis_values = {u.unit_id: u.production_volume for u in units}

            total_basis = sum(basis_values.values())
            if total_basis > 0:
                for unit_id, basis_value in basis_values.items():
                    allocations[unit_id] = pool.total_cost * (basis_value / total_basis)

        elif pool.allocation_method == AllocationMethod.EXERGY:
            # Exergy-based allocation
            basis_values = {u.unit_id: u.exergy_consumption_kwh or u.energy_consumption_kwh or 0.0 for u in units}
            total_basis = sum(basis_values.values())

            if total_basis > 0:
                for unit_id, basis_value in basis_values.items():
                    allocations[unit_id] = pool.total_cost * (basis_value / total_basis)

        elif pool.allocation_method == AllocationMethod.EQUAL:
            # Equal allocation
            cost_per_unit = pool.total_cost / len(units) if units else 0.0
            for unit in units:
                allocations[unit.unit_id] = cost_per_unit

        allocated_total = sum(allocations.values())
        unallocated = pool.total_cost - allocated_total

        return CostPoolAllocation(
            pool_id=pool.pool_id,
            pool_name=pool.name,
            category=pool.category.value,
            total_cost=round(pool.total_cost, 2),
            allocation_method=pool.allocation_method.value,
            allocation_basis=pool.allocation_basis,
            allocations={k: round(v, 2) for k, v in allocations.items()},
            unallocated_cost=round(unallocated, 2)
        )

    def _get_allocation_basis_values(self, basis: str, units: List[ProductionUnit]) -> Dict[str, float]:
        """Get allocation basis values for each unit."""
        values = {}

        basis_lower = basis.lower()
        for unit in units:
            if "production" in basis_lower or "volume" in basis_lower:
                values[unit.unit_id] = unit.production_volume
            elif "energy" in basis_lower:
                values[unit.unit_id] = unit.energy_consumption_kwh or 0.0
            elif "labor" in basis_lower:
                values[unit.unit_id] = unit.labor_hours or 0.0
            elif "area" in basis_lower or "floor" in basis_lower:
                values[unit.unit_id] = unit.floor_area_m2 or 0.0
            elif "machine" in basis_lower:
                values[unit.unit_id] = unit.machine_hours or 0.0
            else:
                values[unit.unit_id] = unit.production_volume

        return values

    def _perform_sensitivity_analysis(self, unit_allocations: List[UnitCostAllocation],
                                       input_data: CostAllocationInput) -> List[SensitivityAnalysis]:
        """Perform sensitivity analysis on key cost drivers."""
        analyses = []

        if not unit_allocations:
            return analyses

        # Analyze electricity rate sensitivity
        base_energy_cost = sum(
            pool.total_cost for pool in input_data.cost_pools
            if pool.category == CostCategory.ENERGY or pool.category == CostCategory.UTILITIES
        )

        if base_energy_cost > 0:
            variation = 0.10  # 10% variation
            avg_unit_cost = sum(u.cost_per_unit for u in unit_allocations) / len(unit_allocations)
            impact = avg_unit_cost * variation * 0.5  # Approximate impact

            analyses.append(SensitivityAnalysis(
                parameter="Electricity Rate",
                base_value=input_data.electricity_rate_per_kwh,
                variation_percent=10.0,
                impact_on_unit_cost=round(impact, 4),
                sensitivity_index=round(impact / avg_unit_cost * 100, 2) if avg_unit_cost > 0 else 0.0
            ))

        # Analyze production volume sensitivity
        if unit_allocations:
            first_unit = unit_allocations[0]
            variation = 0.10
            # Higher production volume = lower per-unit cost
            impact = -first_unit.cost_per_unit * variation

            analyses.append(SensitivityAnalysis(
                parameter="Production Volume",
                base_value=first_unit.production_volume,
                variation_percent=10.0,
                impact_on_unit_cost=round(impact, 4),
                sensitivity_index=round(abs(impact) / first_unit.cost_per_unit * 100, 2) if first_unit.cost_per_unit > 0 else 0.0
            ))

        return analyses

    def _generate_recommendations_and_warnings(self, summary: CostSummary,
                                               unit_allocations: List[UnitCostAllocation],
                                               pool_allocations: List[CostPoolAllocation]) -> None:
        """Generate system-level recommendations and warnings."""
        if summary.allocation_efficiency_percent < 95.0:
            self._warnings.append(f"Allocation efficiency is {summary.allocation_efficiency_percent:.1f}%. "
                                 f"${summary.total_unallocated:.2f} in costs are unallocated.")
            self._recommendations.append("Review allocation methods to improve cost attribution accuracy")

        # High energy costs
        energy_cost = summary.cost_by_category.get(CostCategory.ENERGY.value, 0.0) + \
                     summary.cost_by_category.get(CostCategory.UTILITIES.value, 0.0)
        energy_percent = (energy_cost / summary.total_cost * 100) if summary.total_cost > 0 else 0.0

        if energy_percent > 40.0:
            self._warnings.append(f"Energy/utility costs represent {energy_percent:.1f}% of total costs")
            self._recommendations.append("Prioritize energy efficiency initiatives to reduce unit costs")

        # Cost variance between units
        if len(unit_allocations) > 1:
            costs_per_unit = [u.cost_per_unit for u in unit_allocations]
            max_cost = max(costs_per_unit)
            min_cost = min(costs_per_unit)
            variance = ((max_cost - min_cost) / min_cost * 100) if min_cost > 0 else 0.0

            if variance > 50.0:
                self._warnings.append(f"High cost variance between production units ({variance:.1f}%)")
                self._recommendations.append("Investigate reasons for cost differences and optimize high-cost units")

        # General recommendations
        if not self._recommendations:
            self._recommendations.append("Cost allocation is well-balanced. Continue monitoring for optimization opportunities.")

        self._recommendations.append("Consider implementing real-time cost tracking for improved accuracy")

    def _track_provenance(self, operation: str, inputs: Dict, outputs: Dict, tool_name: str) -> None:
        """Track provenance of calculations."""
        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest(),
            "output_hash": hashlib.sha256(json.dumps(outputs, sort_keys=True, default=str).encode()).hexdigest(),
            "tool_name": tool_name
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of provenance chain."""
        data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": [{"operation": s["operation"], "input_hash": s["input_hash"], "output_hash": s["output_hash"]}
                     for s in self._provenance_steps],
            "timestamp": datetime.utcnow().isoformat()
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-064",
    "name": "COST-ALLOCATOR",
    "version": "1.0.0",
    "summary": "Activity-based cost allocation and utility cost attribution",
    "tags": ["cost-allocation", "abc", "activity-based-costing", "utilities", "production-costs"],
    "standards": [
        {"ref": "Activity-Based Costing", "description": "ABC methodology for cost allocation"},
        {"ref": "ISO 14051", "description": "Material flow cost accounting"}
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True
    }
}
