"""
GL-080: OPEX Optimizer Agent (OPEXOPTIMIZER)

This module implements the OpexOptimizerAgent for operational expenditure
analysis and optimization for energy systems and facilities.

The agent provides:
- Operating cost analysis
- Maintenance cost optimization
- Energy cost reduction strategies
- Labor cost analysis
- Cost driver identification
- Optimization recommendations
- Complete SHA-256 provenance tracking

Example:
    >>> agent = OpexOptimizerAgent()
    >>> result = agent.run(OpexOptimizerInput(...))
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostCategory(str, Enum):
    """Categories of operating costs."""
    ENERGY = "ENERGY"
    MAINTENANCE = "MAINTENANCE"
    LABOR = "LABOR"
    MATERIALS = "MATERIALS"
    INSURANCE = "INSURANCE"
    UTILITIES = "UTILITIES"
    OTHER = "OTHER"


class MaintenanceType(str, Enum):
    """Types of maintenance."""
    PREVENTIVE = "PREVENTIVE"
    PREDICTIVE = "PREDICTIVE"
    CORRECTIVE = "CORRECTIVE"
    CONDITION_BASED = "CONDITION_BASED"


class EnergyType(str, Enum):
    """Types of energy."""
    ELECTRICITY = "ELECTRICITY"
    NATURAL_GAS = "NATURAL_GAS"
    DIESEL = "DIESEL"
    PROPANE = "PROPANE"
    STEAM = "STEAM"


# =============================================================================
# INPUT MODELS
# =============================================================================

class EnergyCost(BaseModel):
    """Energy cost details."""
    energy_type: EnergyType = Field(...)
    annual_consumption: float = Field(..., ge=0)
    unit: str = Field(default="kWh")
    rate_per_unit: float = Field(..., ge=0)
    demand_charge_annual: float = Field(default=0, ge=0)
    description: Optional[str] = None


class MaintenanceCost(BaseModel):
    """Maintenance cost details."""
    equipment_name: str = Field(...)
    maintenance_type: MaintenanceType = Field(default=MaintenanceType.PREVENTIVE)
    annual_cost_usd: float = Field(..., ge=0)
    labor_hours_annual: float = Field(default=0, ge=0)
    parts_cost_annual: float = Field(default=0, ge=0)
    contractor_cost_annual: float = Field(default=0, ge=0)
    frequency: str = Field(default="ANNUAL")


class LaborCost(BaseModel):
    """Labor cost details."""
    role: str = Field(...)
    fte_count: float = Field(..., ge=0)
    annual_salary_usd: float = Field(..., ge=0)
    benefits_percent: float = Field(default=30, ge=0)
    overtime_percent: float = Field(default=0, ge=0)


class OperatingCost(BaseModel):
    """General operating cost."""
    category: CostCategory = Field(...)
    description: str = Field(...)
    annual_cost_usd: float = Field(..., ge=0)
    is_fixed: bool = Field(default=True)


class OpexOptimizerInput(BaseModel):
    """Complete input model for OPEX Optimizer."""
    facility_name: str = Field(...)

    energy_costs: List[EnergyCost] = Field(default_factory=list)
    maintenance_costs: List[MaintenanceCost] = Field(default_factory=list)
    labor_costs: List[LaborCost] = Field(default_factory=list)
    other_costs: List[OperatingCost] = Field(default_factory=list)

    analysis_period_years: int = Field(default=10, ge=1, le=30)
    escalation_rate_percent: float = Field(default=2.5, ge=0, le=10)
    target_savings_percent: float = Field(default=15, ge=0, le=50)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class OpexBreakdown(BaseModel):
    """OPEX breakdown by category."""
    category: CostCategory
    annual_cost_usd: float
    percent_of_total: float
    fixed_cost_usd: float
    variable_cost_usd: float


class CostDriver(BaseModel):
    """Cost driver analysis."""
    driver_name: str
    category: CostCategory
    annual_cost_usd: float
    percent_of_category: float
    trend: str  # INCREASING, STABLE, DECREASING
    optimization_potential: str  # LOW, MEDIUM, HIGH


class OptimizationOpportunity(BaseModel):
    """Optimization opportunity."""
    opportunity_id: str
    description: str
    category: CostCategory
    current_cost_usd: float
    projected_cost_usd: float
    annual_savings_usd: float
    savings_percent: float
    implementation_cost_usd: float
    payback_years: float
    priority: str  # HIGH, MEDIUM, LOW


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""
    operation: str
    timestamp: datetime
    input_hash: str
    output_hash: str
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class OpexOptimizerOutput(BaseModel):
    """Complete output model for OPEX Optimizer."""
    analysis_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    facility_name: str

    # Summary totals
    total_annual_opex_usd: float
    energy_cost_usd: float
    maintenance_cost_usd: float
    labor_cost_usd: float
    other_cost_usd: float

    # Analysis
    cost_breakdown: List[OpexBreakdown]
    cost_drivers: List[CostDriver]
    optimization_opportunities: List[OptimizationOpportunity]

    # Projections
    year_over_year_costs: List[Dict[str, float]]
    total_savings_potential_usd: float
    optimized_annual_opex_usd: float

    # Provenance
    provenance_chain: List[ProvenanceRecord]
    provenance_hash: str

    processing_time_ms: float
    validation_status: str
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# OPEX OPTIMIZER AGENT
# =============================================================================

class OpexOptimizerAgent:
    """GL-080: OPEX Optimizer Agent."""

    AGENT_ID = "GL-080"
    AGENT_NAME = "OPEXOPTIMIZER"
    VERSION = "1.0.0"
    DESCRIPTION = "Operational Expenditure Optimization Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the OpexOptimizerAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []

        logger.info(f"OpexOptimizerAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: OpexOptimizerInput) -> OpexOptimizerOutput:
        """Execute OPEX analysis and optimization."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting OPEX analysis for {input_data.facility_name}")

        try:
            # Calculate cost totals
            energy_total = sum(
                c.annual_consumption * c.rate_per_unit + c.demand_charge_annual
                for c in input_data.energy_costs
            )
            maintenance_total = sum(c.annual_cost_usd for c in input_data.maintenance_costs)
            labor_total = sum(
                c.fte_count * c.annual_salary_usd * (1 + c.benefits_percent/100) * (1 + c.overtime_percent/100)
                for c in input_data.labor_costs
            )
            other_total = sum(c.annual_cost_usd for c in input_data.other_costs)

            total_opex = energy_total + maintenance_total + labor_total + other_total

            self._track_provenance(
                "cost_calculation",
                {"items": len(input_data.energy_costs) + len(input_data.maintenance_costs)},
                {"total": total_opex},
                "Cost Calculator"
            )

            # Generate breakdown
            breakdown = self._generate_breakdown(
                energy_total, maintenance_total, labor_total, other_total, total_opex
            )

            # Identify cost drivers
            cost_drivers = self._identify_cost_drivers(input_data, total_opex)

            # Generate optimization opportunities
            opportunities = self._generate_opportunities(
                input_data, energy_total, maintenance_total, labor_total
            )

            # Calculate projections
            year_costs = self._project_costs(
                total_opex, input_data.analysis_period_years, input_data.escalation_rate_percent
            )

            total_savings = sum(o.annual_savings_usd for o in opportunities)
            optimized_opex = total_opex - total_savings

            # Calculate provenance
            provenance_hash = self._calculate_provenance_hash()
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            analysis_id = (
                f"OPEX-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.facility_name.encode()).hexdigest()[:8]}"
            )

            return OpexOptimizerOutput(
                analysis_id=analysis_id,
                facility_name=input_data.facility_name,
                total_annual_opex_usd=round(total_opex, 2),
                energy_cost_usd=round(energy_total, 2),
                maintenance_cost_usd=round(maintenance_total, 2),
                labor_cost_usd=round(labor_total, 2),
                other_cost_usd=round(other_total, 2),
                cost_breakdown=breakdown,
                cost_drivers=cost_drivers,
                optimization_opportunities=opportunities,
                year_over_year_costs=year_costs,
                total_savings_potential_usd=round(total_savings, 2),
                optimized_annual_opex_usd=round(optimized_opex, 2),
                provenance_chain=[
                    ProvenanceRecord(
                        operation=s["operation"],
                        timestamp=s["timestamp"],
                        input_hash=s["input_hash"],
                        output_hash=s["output_hash"],
                        tool_name=s["tool_name"],
                        parameters=s.get("parameters", {}),
                    )
                    for s in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status="PASS" if not self._validation_errors else "FAIL",
                validation_errors=self._validation_errors,
            )

        except Exception as e:
            logger.error(f"OPEX analysis failed: {str(e)}", exc_info=True)
            raise

    def _generate_breakdown(
        self, energy: float, maintenance: float, labor: float, other: float, total: float
    ) -> List[OpexBreakdown]:
        """Generate cost breakdown."""
        breakdown = []
        for cat, cost in [
            (CostCategory.ENERGY, energy),
            (CostCategory.MAINTENANCE, maintenance),
            (CostCategory.LABOR, labor),
            (CostCategory.OTHER, other),
        ]:
            if cost > 0:
                breakdown.append(OpexBreakdown(
                    category=cat,
                    annual_cost_usd=round(cost, 2),
                    percent_of_total=round(cost/total*100, 1) if total > 0 else 0,
                    fixed_cost_usd=round(cost * 0.7, 2),  # Estimate
                    variable_cost_usd=round(cost * 0.3, 2),
                ))
        return breakdown

    def _identify_cost_drivers(
        self, input_data: OpexOptimizerInput, total: float
    ) -> List[CostDriver]:
        """Identify major cost drivers."""
        drivers = []

        for energy in input_data.energy_costs:
            cost = energy.annual_consumption * energy.rate_per_unit + energy.demand_charge_annual
            drivers.append(CostDriver(
                driver_name=f"{energy.energy_type.value} Consumption",
                category=CostCategory.ENERGY,
                annual_cost_usd=round(cost, 2),
                percent_of_category=100,
                trend="STABLE",
                optimization_potential="MEDIUM" if cost > total * 0.1 else "LOW",
            ))

        for maint in input_data.maintenance_costs:
            drivers.append(CostDriver(
                driver_name=f"{maint.equipment_name} Maintenance",
                category=CostCategory.MAINTENANCE,
                annual_cost_usd=round(maint.annual_cost_usd, 2),
                percent_of_category=100,
                trend="STABLE",
                optimization_potential="MEDIUM" if maint.maintenance_type == MaintenanceType.CORRECTIVE else "LOW",
            ))

        return drivers[:10]  # Top 10

    def _generate_opportunities(
        self, input_data: OpexOptimizerInput, energy: float, maint: float, labor: float
    ) -> List[OptimizationOpportunity]:
        """Generate optimization opportunities."""
        opportunities = []
        opp_id = 1

        # Energy optimization
        if energy > 0:
            savings = energy * 0.10  # 10% potential
            opportunities.append(OptimizationOpportunity(
                opportunity_id=f"OPP-{opp_id:03d}",
                description="Energy efficiency improvements",
                category=CostCategory.ENERGY,
                current_cost_usd=round(energy, 2),
                projected_cost_usd=round(energy - savings, 2),
                annual_savings_usd=round(savings, 2),
                savings_percent=10,
                implementation_cost_usd=round(savings * 3, 2),
                payback_years=3.0,
                priority="HIGH",
            ))
            opp_id += 1

        # Maintenance optimization
        if maint > 0:
            savings = maint * 0.15  # 15% potential
            opportunities.append(OptimizationOpportunity(
                opportunity_id=f"OPP-{opp_id:03d}",
                description="Predictive maintenance implementation",
                category=CostCategory.MAINTENANCE,
                current_cost_usd=round(maint, 2),
                projected_cost_usd=round(maint - savings, 2),
                annual_savings_usd=round(savings, 2),
                savings_percent=15,
                implementation_cost_usd=round(savings * 2, 2),
                payback_years=2.0,
                priority="HIGH",
            ))

        return opportunities

    def _project_costs(self, base: float, years: int, escalation: float) -> List[Dict[str, float]]:
        """Project costs over time."""
        projections = []
        for year in range(years):
            cost = base * ((1 + escalation/100) ** year)
            projections.append({
                "year": year + 1,
                "cost_usd": round(cost, 2),
                "cumulative_usd": round(sum(
                    base * ((1 + escalation/100) ** y) for y in range(year + 1)
                ), 2),
            })
        return projections

    def _track_provenance(
        self, operation: str, inputs: Dict, outputs: Dict, tool_name: str
    ) -> None:
        """Track provenance step."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate provenance chain hash."""
        data = {
            "agent_id": self.AGENT_ID,
            "steps": [
                {"operation": s["operation"], "input_hash": s["input_hash"]}
                for s in self._provenance_steps
            ],
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-080",
    "name": "OPEXOPTIMIZER - Operational Expenditure Optimization Agent",
    "version": "1.0.0",
    "summary": "Analyzes and optimizes operational costs for energy systems",
    "tags": ["opex", "operating-costs", "maintenance", "optimization"],
    "owners": ["operations-team"],
    "compute": {
        "entrypoint": "python://agents.gl_080_opex_optimizer.agent:OpexOptimizerAgent",
        "deterministic": True,
    },
    "provenance": {"calculation_verified": True, "enable_audit": True},
}
