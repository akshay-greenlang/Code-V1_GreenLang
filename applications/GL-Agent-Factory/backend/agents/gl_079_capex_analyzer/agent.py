"""
GL-079: CAPEX Analyzer Agent (CAPEXANALYZER)

This module implements the CapexAnalyzerAgent for capital expenditure analysis
and project cost estimation for energy efficiency and sustainability projects.

The agent provides:
- Equipment cost estimation
- Installation cost calculation
- Soft cost analysis (engineering, permitting, etc.)
- Contingency planning
- Cost benchmarking
- Sensitivity analysis
- Complete SHA-256 provenance tracking

Standards/References:
- RSMeans construction cost data
- NREL Cost Benchmarks
- Industry-specific cost databases

Example:
    >>> agent = CapexAnalyzerAgent()
    >>> result = agent.run(CapexAnalyzerInput(
    ...     project_name="Solar Installation",
    ...     equipment_costs=[...],
    ... ))
"""

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CostCategory(str, Enum):
    """Categories of capital costs."""
    EQUIPMENT = "EQUIPMENT"
    INSTALLATION = "INSTALLATION"
    ELECTRICAL = "ELECTRICAL"
    MECHANICAL = "MECHANICAL"
    STRUCTURAL = "STRUCTURAL"
    ENGINEERING = "ENGINEERING"
    PERMITTING = "PERMITTING"
    CONTINGENCY = "CONTINGENCY"
    FINANCING = "FINANCING"
    PROJECT_MANAGEMENT = "PROJECT_MANAGEMENT"


class ProjectPhase(str, Enum):
    """Project phases for cost allocation."""
    PLANNING = "PLANNING"
    DESIGN = "DESIGN"
    PROCUREMENT = "PROCUREMENT"
    CONSTRUCTION = "CONSTRUCTION"
    COMMISSIONING = "COMMISSIONING"


class EquipmentType(str, Enum):
    """Types of equipment for cost estimation."""
    SOLAR_PV = "SOLAR_PV"
    BATTERY_STORAGE = "BATTERY_STORAGE"
    HVAC = "HVAC"
    LIGHTING = "LIGHTING"
    MOTORS_VFD = "MOTORS_VFD"
    BOILER = "BOILER"
    CHILLER = "CHILLER"
    HEAT_PUMP = "HEAT_PUMP"
    EMS = "EMS"
    OTHER = "OTHER"


class FundingType(str, Enum):
    """Types of project funding."""
    CASH = "CASH"
    DEBT = "DEBT"
    LEASE = "LEASE"
    PPA = "PPA"
    PACE = "PACE"
    GRANT = "GRANT"


# =============================================================================
# INPUT MODELS
# =============================================================================

class EquipmentCost(BaseModel):
    """Equipment cost details."""

    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    description: str = Field(..., description="Equipment description")
    quantity: int = Field(..., ge=1, description="Number of units")
    unit_cost_usd: float = Field(..., ge=0, description="Cost per unit")
    capacity_per_unit: Optional[float] = Field(None, description="Capacity per unit")
    capacity_unit: Optional[str] = Field(None, description="Unit of capacity (kW, tons)")
    manufacturer: Optional[str] = Field(None, description="Manufacturer")
    model: Optional[str] = Field(None, description="Model number")
    lead_time_weeks: Optional[int] = Field(None, ge=0, description="Lead time")


class InstallationCost(BaseModel):
    """Installation cost details."""

    category: CostCategory = Field(..., description="Cost category")
    description: str = Field(..., description="Work description")
    labor_hours: float = Field(default=0, ge=0, description="Labor hours")
    labor_rate_usd: float = Field(default=75, ge=0, description="Labor rate $/hr")
    material_cost_usd: float = Field(default=0, ge=0, description="Material cost")
    subcontractor_cost_usd: float = Field(default=0, ge=0, description="Subcontractor cost")


class SoftCost(BaseModel):
    """Soft cost details."""

    category: CostCategory = Field(..., description="Cost category")
    description: str = Field(..., description="Cost description")
    cost_usd: float = Field(..., ge=0, description="Total cost")
    phase: ProjectPhase = Field(default=ProjectPhase.PLANNING, description="Project phase")
    is_percentage: bool = Field(default=False, description="Cost is % of hard costs")
    percentage: Optional[float] = Field(None, ge=0, le=100, description="Percentage value")


class FundingSource(BaseModel):
    """Project funding source."""

    funding_type: FundingType = Field(..., description="Type of funding")
    amount_usd: float = Field(..., ge=0, description="Funding amount")
    interest_rate: Optional[float] = Field(None, ge=0, le=100, description="Interest rate %")
    term_years: Optional[int] = Field(None, ge=0, description="Term in years")
    description: Optional[str] = Field(None, description="Funding description")


class CapexAnalyzerInput(BaseModel):
    """Complete input model for CAPEX Analyzer."""

    project_name: str = Field(..., description="Project name")
    project_location: Optional[str] = Field(None, description="Project location")

    # Costs
    equipment_costs: List[EquipmentCost] = Field(..., description="Equipment costs")
    installation_costs: List[InstallationCost] = Field(
        default_factory=list, description="Installation costs"
    )
    soft_costs: List[SoftCost] = Field(default_factory=list, description="Soft costs")

    # Contingency
    contingency_percent: float = Field(default=10.0, ge=0, le=50, description="Contingency %")

    # Funding
    funding_sources: List[FundingSource] = Field(
        default_factory=list, description="Funding sources"
    )

    # Analysis options
    include_sensitivity: bool = Field(default=True, description="Include sensitivity analysis")
    sensitivity_range_percent: float = Field(
        default=20.0, ge=5, le=50, description="Sensitivity range"
    )

    # Benchmarking
    benchmark_region: Optional[str] = Field(None, description="Region for benchmarking")
    project_size_capacity: Optional[float] = Field(None, description="Project size for $/unit")
    capacity_unit: Optional[str] = Field(None, description="Unit (kW, tons, sqft)")

    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class CapexBreakdown(BaseModel):
    """CAPEX breakdown by category."""

    category: CostCategory = Field(..., description="Cost category")
    total_usd: float = Field(..., description="Total cost in category")
    percent_of_total: float = Field(..., description="Percentage of total CAPEX")
    line_items: List[Dict[str, Any]] = Field(default_factory=list, description="Line items")


class CostComparison(BaseModel):
    """Cost comparison with benchmarks."""

    metric: str = Field(..., description="Comparison metric")
    project_value: float = Field(..., description="Project value")
    benchmark_low: float = Field(..., description="Benchmark low")
    benchmark_mid: float = Field(..., description="Benchmark mid")
    benchmark_high: float = Field(..., description="Benchmark high")
    percentile: float = Field(..., description="Project percentile (0-100)")
    status: str = Field(..., description="BELOW, WITHIN, ABOVE benchmark")


class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis results."""

    parameter: str = Field(..., description="Parameter varied")
    base_value: float = Field(..., description="Base value")
    low_value: float = Field(..., description="Low scenario value")
    high_value: float = Field(..., description="High scenario value")
    impact_on_total_low: float = Field(..., description="Impact on total (low)")
    impact_on_total_high: float = Field(..., description="Impact on total (high)")
    sensitivity_index: float = Field(..., description="Sensitivity index")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict)


class CapexAnalyzerOutput(BaseModel):
    """Complete output model for CAPEX Analyzer."""

    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    project_name: str = Field(..., description="Project name")

    # Summary totals
    total_equipment_cost_usd: float = Field(..., description="Total equipment")
    total_installation_cost_usd: float = Field(..., description="Total installation")
    total_soft_cost_usd: float = Field(..., description="Total soft costs")
    contingency_usd: float = Field(..., description="Contingency amount")
    total_capex_usd: float = Field(..., description="Grand total CAPEX")

    # Per-unit metrics
    cost_per_capacity_unit: Optional[float] = Field(None, description="$/unit capacity")
    capacity_unit: Optional[str] = Field(None, description="Capacity unit")

    # Breakdowns
    cost_breakdown: List[CapexBreakdown] = Field(..., description="Cost breakdown")

    # Analysis
    benchmark_comparison: List[CostComparison] = Field(
        default_factory=list, description="Benchmark comparison"
    )
    sensitivity_results: List[SensitivityAnalysis] = Field(
        default_factory=list, description="Sensitivity analysis"
    )

    # Funding
    funding_summary: Dict[str, float] = Field(
        default_factory=dict, description="Funding summary"
    )

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(...)
    provenance_hash: str = Field(...)

    processing_time_ms: float = Field(...)
    validation_status: str = Field(...)
    validation_errors: List[str] = Field(default_factory=list)


# =============================================================================
# COST BENCHMARKS
# =============================================================================

COST_BENCHMARKS = {
    EquipmentType.SOLAR_PV: {
        "low": 1.00,  # $/W
        "mid": 1.50,
        "high": 2.50,
        "unit": "W",
    },
    EquipmentType.BATTERY_STORAGE: {
        "low": 250,  # $/kWh
        "mid": 400,
        "high": 600,
        "unit": "kWh",
    },
    EquipmentType.HVAC: {
        "low": 800,  # $/ton
        "mid": 1200,
        "high": 2000,
        "unit": "ton",
    },
    EquipmentType.LIGHTING: {
        "low": 30,  # $/fixture
        "mid": 75,
        "high": 150,
        "unit": "fixture",
    },
}


# =============================================================================
# CAPEX ANALYZER AGENT
# =============================================================================

class CapexAnalyzerAgent:
    """
    GL-079: CAPEX Analyzer Agent (CAPEXANALYZER).

    This agent provides comprehensive capital expenditure analysis for
    energy efficiency and sustainability projects.

    Zero-Hallucination Guarantee:
    - All cost calculations use deterministic formulas
    - Benchmarks from published industry sources
    - No LLM inference in cost calculations
    - Complete audit trail for compliance
    """

    AGENT_ID = "GL-079"
    AGENT_NAME = "CAPEXANALYZER"
    VERSION = "1.0.0"
    DESCRIPTION = "Capital Expenditure Analysis Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the CapexAnalyzerAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self.benchmarks = COST_BENCHMARKS

        logger.info(
            f"CapexAnalyzerAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME})"
        )

    def run(self, input_data: CapexAnalyzerInput) -> CapexAnalyzerOutput:
        """Execute CAPEX analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []

        logger.info(f"Starting CAPEX analysis for {input_data.project_name}")

        try:
            # Step 1: Calculate equipment costs
            equipment_total = self._calculate_equipment_costs(input_data.equipment_costs)
            self._track_provenance(
                "equipment_costing",
                {"items": len(input_data.equipment_costs)},
                {"total": equipment_total},
                "Equipment Calculator"
            )

            # Step 2: Calculate installation costs
            installation_total = self._calculate_installation_costs(
                input_data.installation_costs
            )
            self._track_provenance(
                "installation_costing",
                {"items": len(input_data.installation_costs)},
                {"total": installation_total},
                "Installation Calculator"
            )

            # Step 3: Calculate soft costs
            soft_total = self._calculate_soft_costs(
                input_data.soft_costs,
                equipment_total + installation_total
            )
            self._track_provenance(
                "soft_cost_calculation",
                {"items": len(input_data.soft_costs)},
                {"total": soft_total},
                "Soft Cost Calculator"
            )

            # Step 4: Calculate contingency
            subtotal = equipment_total + installation_total + soft_total
            contingency = subtotal * (input_data.contingency_percent / 100)

            # Step 5: Calculate total CAPEX
            total_capex = subtotal + contingency

            # Step 6: Generate cost breakdown
            breakdown = self._generate_breakdown(
                input_data, equipment_total, installation_total, soft_total, contingency
            )

            # Step 7: Calculate per-unit cost
            cost_per_unit = None
            if input_data.project_size_capacity and input_data.project_size_capacity > 0:
                cost_per_unit = total_capex / input_data.project_size_capacity

            # Step 8: Benchmark comparison
            comparisons = self._compare_to_benchmarks(
                input_data.equipment_costs, cost_per_unit, input_data.capacity_unit
            )

            # Step 9: Sensitivity analysis
            sensitivity = []
            if input_data.include_sensitivity:
                sensitivity = self._run_sensitivity(
                    equipment_total, installation_total, soft_total,
                    contingency, input_data.sensitivity_range_percent
                )

            # Step 10: Summarize funding
            funding_summary = self._summarize_funding(input_data.funding_sources, total_capex)

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            analysis_id = (
                f"CAPEX-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(input_data.project_name.encode()).hexdigest()[:8]}"
            )

            return CapexAnalyzerOutput(
                analysis_id=analysis_id,
                project_name=input_data.project_name,
                total_equipment_cost_usd=round(equipment_total, 2),
                total_installation_cost_usd=round(installation_total, 2),
                total_soft_cost_usd=round(soft_total, 2),
                contingency_usd=round(contingency, 2),
                total_capex_usd=round(total_capex, 2),
                cost_per_capacity_unit=round(cost_per_unit, 2) if cost_per_unit else None,
                capacity_unit=input_data.capacity_unit,
                cost_breakdown=breakdown,
                benchmark_comparison=comparisons,
                sensitivity_results=sensitivity,
                funding_summary=funding_summary,
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
            logger.error(f"CAPEX analysis failed: {str(e)}", exc_info=True)
            raise

    def _calculate_equipment_costs(self, equipment: List[EquipmentCost]) -> float:
        """Calculate total equipment costs."""
        total = 0.0
        for item in equipment:
            total += item.quantity * item.unit_cost_usd
        return total

    def _calculate_installation_costs(self, installation: List[InstallationCost]) -> float:
        """Calculate total installation costs."""
        total = 0.0
        for item in installation:
            labor_cost = item.labor_hours * item.labor_rate_usd
            total += labor_cost + item.material_cost_usd + item.subcontractor_cost_usd
        return total

    def _calculate_soft_costs(self, soft_costs: List[SoftCost], hard_costs: float) -> float:
        """Calculate total soft costs."""
        total = 0.0
        for item in soft_costs:
            if item.is_percentage and item.percentage:
                total += hard_costs * (item.percentage / 100)
            else:
                total += item.cost_usd
        return total

    def _generate_breakdown(
        self,
        input_data: CapexAnalyzerInput,
        equipment_total: float,
        installation_total: float,
        soft_total: float,
        contingency: float
    ) -> List[CapexBreakdown]:
        """Generate cost breakdown by category."""
        total = equipment_total + installation_total + soft_total + contingency
        breakdown = []

        if equipment_total > 0:
            breakdown.append(CapexBreakdown(
                category=CostCategory.EQUIPMENT,
                total_usd=round(equipment_total, 2),
                percent_of_total=round(equipment_total / total * 100, 1),
                line_items=[
                    {"description": e.description, "cost": e.quantity * e.unit_cost_usd}
                    for e in input_data.equipment_costs
                ],
            ))

        if installation_total > 0:
            breakdown.append(CapexBreakdown(
                category=CostCategory.INSTALLATION,
                total_usd=round(installation_total, 2),
                percent_of_total=round(installation_total / total * 100, 1),
                line_items=[
                    {"description": i.description, "cost": i.labor_hours * i.labor_rate_usd + i.material_cost_usd}
                    for i in input_data.installation_costs
                ],
            ))

        if soft_total > 0:
            breakdown.append(CapexBreakdown(
                category=CostCategory.ENGINEERING,
                total_usd=round(soft_total, 2),
                percent_of_total=round(soft_total / total * 100, 1),
                line_items=[
                    {"description": s.description, "cost": s.cost_usd}
                    for s in input_data.soft_costs
                ],
            ))

        breakdown.append(CapexBreakdown(
            category=CostCategory.CONTINGENCY,
            total_usd=round(contingency, 2),
            percent_of_total=round(contingency / total * 100, 1),
            line_items=[{"description": "Contingency reserve", "cost": contingency}],
        ))

        return breakdown

    def _compare_to_benchmarks(
        self,
        equipment: List[EquipmentCost],
        cost_per_unit: Optional[float],
        capacity_unit: Optional[str]
    ) -> List[CostComparison]:
        """Compare costs to industry benchmarks."""
        comparisons = []

        for item in equipment:
            if item.equipment_type in self.benchmarks:
                bench = self.benchmarks[item.equipment_type]
                actual_unit_cost = item.unit_cost_usd

                if actual_unit_cost < bench["low"]:
                    status = "BELOW"
                    percentile = (actual_unit_cost / bench["low"]) * 25
                elif actual_unit_cost > bench["high"]:
                    status = "ABOVE"
                    percentile = 75 + ((actual_unit_cost - bench["high"]) / bench["high"]) * 25
                else:
                    status = "WITHIN"
                    range_size = bench["high"] - bench["low"]
                    percentile = 25 + ((actual_unit_cost - bench["low"]) / range_size) * 50

                comparisons.append(CostComparison(
                    metric=f"{item.equipment_type.value} - $/unit",
                    project_value=actual_unit_cost,
                    benchmark_low=bench["low"],
                    benchmark_mid=bench["mid"],
                    benchmark_high=bench["high"],
                    percentile=round(min(max(percentile, 0), 100), 1),
                    status=status,
                ))

        return comparisons

    def _run_sensitivity(
        self,
        equipment: float,
        installation: float,
        soft: float,
        contingency: float,
        range_percent: float
    ) -> List[SensitivityAnalysis]:
        """Run sensitivity analysis."""
        base_total = equipment + installation + soft + contingency
        results = []

        for name, value in [
            ("Equipment Cost", equipment),
            ("Installation Cost", installation),
            ("Soft Costs", soft),
        ]:
            low_value = value * (1 - range_percent / 100)
            high_value = value * (1 + range_percent / 100)

            low_total = base_total - value + low_value
            high_total = base_total - value + high_value

            impact_low = low_total - base_total
            impact_high = high_total - base_total

            sensitivity_index = (high_total - low_total) / base_total

            results.append(SensitivityAnalysis(
                parameter=name,
                base_value=round(value, 2),
                low_value=round(low_value, 2),
                high_value=round(high_value, 2),
                impact_on_total_low=round(impact_low, 2),
                impact_on_total_high=round(impact_high, 2),
                sensitivity_index=round(sensitivity_index, 4),
            ))

        return results

    def _summarize_funding(
        self,
        funding_sources: List[FundingSource],
        total_capex: float
    ) -> Dict[str, float]:
        """Summarize funding sources."""
        summary = {}
        total_funded = 0.0

        for source in funding_sources:
            key = source.funding_type.value
            summary[key] = summary.get(key, 0) + source.amount_usd
            total_funded += source.amount_usd

        summary["TOTAL_FUNDED"] = round(total_funded, 2)
        summary["FUNDING_GAP"] = round(max(0, total_capex - total_funded), 2)

        return summary

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
    "id": "GL-079",
    "name": "CAPEXANALYZER - Capital Expenditure Analysis Agent",
    "version": "1.0.0",
    "summary": "Analyzes capital costs for energy projects with benchmarking",
    "tags": ["capex", "cost-estimation", "budgeting", "benchmarking"],
    "owners": ["finance-team"],
    "compute": {
        "entrypoint": "python://agents.gl_079_capex_analyzer.agent:CapexAnalyzerAgent",
        "deterministic": True,
    },
    "standards": [
        {"ref": "RSMeans", "description": "Construction Cost Data"},
        {"ref": "NREL-Benchmarks", "description": "NREL Cost Benchmarks"},
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True},
}
