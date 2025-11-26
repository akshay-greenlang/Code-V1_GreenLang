"""Improvement Opportunity Analyzer.

This module identifies and prioritizes thermal efficiency improvement
opportunities, including heat recovery options, ROI calculations,
and implementation recommendations.

Features:
    - Loss reduction potential analysis
    - Heat recovery opportunities identification
    - ROI and payback calculations
    - Prioritized recommendations
    - Implementation roadmap

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime


class PriorityLevel(Enum):
    """Priority level for improvement opportunities."""
    CRITICAL = "critical"       # Immediate action required
    HIGH = "high"               # Implement within 3 months
    MEDIUM = "medium"           # Implement within 1 year
    LOW = "low"                 # Consider for next budget cycle
    INFORMATIONAL = "informational"  # For awareness only


class ImprovementCategory(Enum):
    """Category of improvement opportunity."""
    COMBUSTION_OPTIMIZATION = "combustion_optimization"
    HEAT_RECOVERY = "heat_recovery"
    INSULATION = "insulation"
    STEAM_SYSTEM = "steam_system"
    CONTROLS = "controls"
    MAINTENANCE = "maintenance"
    OPERATIONAL = "operational"
    EQUIPMENT_UPGRADE = "equipment_upgrade"
    PROCESS_INTEGRATION = "process_integration"
    FUEL_SWITCHING = "fuel_switching"


class ImplementationComplexity(Enum):
    """Complexity of implementation."""
    SIMPLE = "simple"           # No capital, operational change
    MODERATE = "moderate"       # Minor capital, limited downtime
    COMPLEX = "complex"         # Significant capital/engineering
    MAJOR = "major"             # Major project, extended downtime


@dataclass
class ROICalculation:
    """Return on Investment calculation.

    Attributes:
        capital_cost: Initial investment cost
        annual_savings: Yearly energy savings
        annual_maintenance: Additional yearly maintenance cost
        simple_payback_years: Simple payback period
        roi_percent: Return on investment (%)
        npv_10_year: Net present value over 10 years
        irr_percent: Internal rate of return (%)
        currency: Currency code (default USD)
    """
    capital_cost: float
    annual_savings: float
    annual_maintenance: float
    simple_payback_years: float
    roi_percent: float
    npv_10_year: float
    irr_percent: Optional[float]
    currency: str = "USD"


@dataclass
class HeatRecoveryOption:
    """Heat recovery opportunity details.

    Attributes:
        source_name: Name of heat source
        source_temperature_c: Source temperature
        source_energy_kw: Available heat energy
        sink_name: Name of heat sink (use)
        sink_temperature_c: Required sink temperature
        recoverable_kw: Recoverable energy
        recovery_efficiency: Expected recovery efficiency
        equipment_type: Type of heat recovery equipment
        estimated_cost: Estimated installation cost
        annual_savings: Estimated annual savings
    """
    source_name: str
    source_temperature_c: float
    source_energy_kw: float
    sink_name: str
    sink_temperature_c: float
    recoverable_kw: float
    recovery_efficiency: float
    equipment_type: str
    estimated_cost: float
    annual_savings: float


@dataclass
class ImprovementOpportunity:
    """Individual improvement opportunity.

    Attributes:
        opportunity_id: Unique identifier
        title: Brief title
        description: Detailed description
        category: Improvement category
        priority: Priority level
        complexity: Implementation complexity
        current_value: Current performance value
        target_value: Target performance value
        energy_savings_kw: Potential energy savings
        energy_savings_percent: Savings as % of input
        roi: ROI calculation (if applicable)
        heat_recovery: Heat recovery details (if applicable)
        implementation_steps: Steps to implement
        risks: Implementation risks
        notes: Additional notes
    """
    opportunity_id: str
    title: str
    description: str
    category: ImprovementCategory
    priority: PriorityLevel
    complexity: ImplementationComplexity
    current_value: float
    target_value: float
    energy_savings_kw: float
    energy_savings_percent: float
    roi: Optional[ROICalculation] = None
    heat_recovery: Optional[HeatRecoveryOption] = None
    implementation_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "id": self.opportunity_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "complexity": self.complexity.value,
            "energy_savings_kw": self.energy_savings_kw,
            "energy_savings_percent": self.energy_savings_percent,
            "implementation_steps": self.implementation_steps
        }
        if self.roi:
            result["roi"] = {
                "capital_cost": self.roi.capital_cost,
                "annual_savings": self.roi.annual_savings,
                "payback_years": self.roi.simple_payback_years
            }
        return result


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class ImprovementAnalysis:
    """Complete improvement analysis result.

    Attributes:
        total_savings_potential_kw: Total potential savings
        total_savings_potential_percent: Total savings as %
        opportunities: List of improvement opportunities
        priority_summary: Count by priority level
        category_summary: Savings by category
        quick_wins: Opportunities with <1 year payback
        major_projects: Opportunities requiring significant capital
        implementation_roadmap: Phased implementation plan
        calculation_steps: Audit trail
        provenance_hash: SHA-256 hash
        analysis_timestamp: When analyzed
    """
    total_savings_potential_kw: float
    total_savings_potential_percent: float
    opportunities: List[ImprovementOpportunity]
    priority_summary: Dict[str, int]
    category_summary: Dict[str, float]
    quick_wins: List[ImprovementOpportunity]
    major_projects: List[ImprovementOpportunity]
    implementation_roadmap: Dict[str, List[str]]
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    analysis_timestamp: str
    analyzer_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "total_savings_potential_kw": self.total_savings_potential_kw,
            "total_savings_potential_percent": self.total_savings_potential_percent,
            "opportunity_count": len(self.opportunities),
            "priority_summary": self.priority_summary,
            "category_summary": self.category_summary,
            "quick_wins_count": len(self.quick_wins),
            "opportunities": [o.to_dict() for o in self.opportunities],
            "implementation_roadmap": self.implementation_roadmap,
            "provenance_hash": self.provenance_hash,
            "analysis_timestamp": self.analysis_timestamp
        }


class ImprovementAnalyzer:
    """Improvement Opportunity Analyzer.

    Analyzes thermal system performance to identify and prioritize
    improvement opportunities with ROI calculations.

    Example:
        >>> analyzer = ImprovementAnalyzer()
        >>> result = analyzer.analyze(
        ...     energy_input_kw=5000,
        ...     current_efficiency_percent=82,
        ...     loss_breakdown={"flue_gas": 10, "radiation": 3, "other": 5},
        ...     operating_hours_per_year=8000
        ... )
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 2

    # Default cost assumptions (can be overridden)
    DEFAULT_ENERGY_COST_KWH: float = 0.05  # $/kWh
    DEFAULT_DISCOUNT_RATE: float = 0.08    # 8%
    EQUIPMENT_LIFETIME_YEARS: int = 15

    # Improvement opportunity templates
    OPPORTUNITY_TEMPLATES: Dict[str, Dict[str, Any]] = {
        "combustion_optimization": {
            "title": "Combustion Optimization",
            "description": "Optimize excess air and combustion efficiency",
            "category": ImprovementCategory.COMBUSTION_OPTIMIZATION,
            "complexity": ImplementationComplexity.SIMPLE,
            "typical_savings_percent": 2.0,
            "typical_cost_per_kw_saved": 100,
            "implementation_steps": [
                "Perform combustion analysis",
                "Adjust dampers/burner settings",
                "Calibrate O2 analyzer",
                "Tune fuel/air ratio",
                "Document optimal settings"
            ]
        },
        "economizer": {
            "title": "Install/Upgrade Economizer",
            "description": "Recover heat from flue gas to preheat feedwater",
            "category": ImprovementCategory.HEAT_RECOVERY,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 4.0,
            "typical_cost_per_kw_saved": 300,
            "implementation_steps": [
                "Evaluate flue gas conditions",
                "Size economizer for application",
                "Design installation layout",
                "Procure and install equipment",
                "Commission and optimize"
            ]
        },
        "air_preheater": {
            "title": "Install Combustion Air Preheater",
            "description": "Preheat combustion air using flue gas",
            "category": ImprovementCategory.HEAT_RECOVERY,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 3.0,
            "typical_cost_per_kw_saved": 350,
            "implementation_steps": [
                "Assess flue gas temperature",
                "Size air preheater",
                "Modify ductwork",
                "Install and commission"
            ]
        },
        "insulation_upgrade": {
            "title": "Upgrade Insulation",
            "description": "Repair or upgrade thermal insulation",
            "category": ImprovementCategory.INSULATION,
            "complexity": ImplementationComplexity.SIMPLE,
            "typical_savings_percent": 1.5,
            "typical_cost_per_kw_saved": 150,
            "implementation_steps": [
                "Perform thermal survey",
                "Identify hot spots",
                "Specify insulation requirements",
                "Install/repair insulation"
            ]
        },
        "blowdown_heat_recovery": {
            "title": "Blowdown Heat Recovery",
            "description": "Recover heat from boiler blowdown",
            "category": ImprovementCategory.HEAT_RECOVERY,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 1.0,
            "typical_cost_per_kw_saved": 200,
            "implementation_steps": [
                "Measure blowdown rate",
                "Size heat exchanger",
                "Install flash tank/HX",
                "Pipe to makeup water"
            ]
        },
        "condensate_return": {
            "title": "Improve Condensate Return",
            "description": "Increase condensate return rate",
            "category": ImprovementCategory.STEAM_SYSTEM,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 2.0,
            "typical_cost_per_kw_saved": 180,
            "implementation_steps": [
                "Survey condensate system",
                "Repair/replace traps",
                "Add return piping",
                "Install pumps if needed"
            ]
        },
        "o2_trim_controls": {
            "title": "Install O2 Trim Controls",
            "description": "Automatic excess air control",
            "category": ImprovementCategory.CONTROLS,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 2.5,
            "typical_cost_per_kw_saved": 250,
            "implementation_steps": [
                "Install O2 analyzer",
                "Install control valve/VFD",
                "Program control logic",
                "Tune and optimize"
            ]
        },
        "vfd_fans": {
            "title": "Install VFDs on Combustion Fans",
            "description": "Variable speed drives for fan control",
            "category": ImprovementCategory.CONTROLS,
            "complexity": ImplementationComplexity.MODERATE,
            "typical_savings_percent": 1.5,
            "typical_cost_per_kw_saved": 200,
            "implementation_steps": [
                "Evaluate fan operation",
                "Size VFD",
                "Install and wire VFD",
                "Program and commission"
            ]
        },
        "condensing_economizer": {
            "title": "Condensing Economizer",
            "description": "Deep heat recovery with condensation",
            "category": ImprovementCategory.HEAT_RECOVERY,
            "complexity": ImplementationComplexity.COMPLEX,
            "typical_savings_percent": 6.0,
            "typical_cost_per_kw_saved": 500,
            "implementation_steps": [
                "Assess flue gas conditions",
                "Evaluate corrosion concerns",
                "Design condensing system",
                "Install with materials selection",
                "Commission with acid neutralization"
            ]
        }
    }

    def __init__(
        self,
        energy_cost_per_kwh: float = 0.05,
        discount_rate: float = 0.08,
        precision: int = 2
    ) -> None:
        """Initialize the Improvement Analyzer.

        Args:
            energy_cost_per_kwh: Energy cost in $/kWh
            discount_rate: Discount rate for NPV calculations
            precision: Decimal places for rounding
        """
        self.energy_cost = energy_cost_per_kwh
        self.discount_rate = discount_rate
        self.precision = precision
        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def analyze(
        self,
        energy_input_kw: float,
        current_efficiency_percent: float,
        loss_breakdown: Dict[str, float],
        operating_hours_per_year: int = 8000,
        target_efficiency_percent: Optional[float] = None
    ) -> ImprovementAnalysis:
        """Analyze improvement opportunities.

        Args:
            energy_input_kw: Total energy input (kW)
            current_efficiency_percent: Current efficiency (%)
            loss_breakdown: Dict of {loss_type: percent_of_input}
            operating_hours_per_year: Annual operating hours
            target_efficiency_percent: Target efficiency (optional)

        Returns:
            ImprovementAnalysis with prioritized opportunities
        """
        self._reset_calculation_state()

        opportunities: List[ImprovementOpportunity] = []
        total_savings_kw = 0.0

        # Analyze each loss category
        for loss_type, loss_percent in loss_breakdown.items():
            loss_kw = energy_input_kw * (loss_percent / 100)

            # Identify applicable improvements
            applicable = self._identify_improvements_for_loss(loss_type, loss_kw)

            for template_name, savings_fraction in applicable:
                opportunity = self._create_opportunity(
                    template_name=template_name,
                    loss_kw=loss_kw,
                    savings_fraction=savings_fraction,
                    energy_input_kw=energy_input_kw,
                    operating_hours=operating_hours_per_year
                )
                opportunities.append(opportunity)
                total_savings_kw += opportunity.energy_savings_kw

        # Sort by priority and savings
        opportunities.sort(
            key=lambda x: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}[x.priority.value],
                -x.energy_savings_kw
            )
        )

        # Identify quick wins (payback < 1 year)
        quick_wins = [
            o for o in opportunities
            if o.roi and o.roi.simple_payback_years < 1.0
        ]

        # Identify major projects
        major_projects = [
            o for o in opportunities
            if o.complexity in [ImplementationComplexity.COMPLEX, ImplementationComplexity.MAJOR]
        ]

        # Calculate summaries
        priority_summary = self._summarize_by_priority(opportunities)
        category_summary = self._summarize_by_category(opportunities)

        # Create implementation roadmap
        roadmap = self._create_roadmap(opportunities)

        # Calculate total savings percent
        total_savings_percent = (total_savings_kw / energy_input_kw * 100) if energy_input_kw > 0 else 0

        # Generate provenance
        provenance = self._generate_provenance_hash(
            energy_input_kw, current_efficiency_percent, loss_breakdown
        )
        timestamp = datetime.utcnow().isoformat() + "Z"

        return ImprovementAnalysis(
            total_savings_potential_kw=self._round_value(total_savings_kw),
            total_savings_potential_percent=self._round_value(total_savings_percent),
            opportunities=opportunities,
            priority_summary=priority_summary,
            category_summary=category_summary,
            quick_wins=quick_wins,
            major_projects=major_projects,
            implementation_roadmap=roadmap,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance,
            analysis_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def analyze_heat_recovery(
        self,
        heat_sources: List[Dict[str, float]],
        heat_sinks: List[Dict[str, float]],
        operating_hours_per_year: int = 8000
    ) -> List[HeatRecoveryOption]:
        """Analyze heat recovery opportunities.

        Args:
            heat_sources: List of {name, temperature_c, energy_kw}
            heat_sinks: List of {name, temperature_c, energy_kw}
            operating_hours_per_year: Annual operating hours

        Returns:
            List of heat recovery opportunities
        """
        options: List[HeatRecoveryOption] = []

        for source in heat_sources:
            for sink in heat_sinks:
                # Check if temperature match is feasible
                temp_diff = source["temperature_c"] - sink["temperature_c"]

                if temp_diff > 20:  # Minimum approach temperature
                    # Calculate recoverable energy
                    max_recoverable = min(source["energy_kw"], sink["energy_kw"])
                    recovery_efficiency = min(0.85, (temp_diff - 10) / temp_diff)
                    recoverable = max_recoverable * recovery_efficiency

                    # Determine equipment type
                    if source["temperature_c"] > 400:
                        equipment = "Radiation recuperator"
                        cost_factor = 500
                    elif source["temperature_c"] > 200:
                        equipment = "Convective heat exchanger"
                        cost_factor = 300
                    else:
                        equipment = "Plate heat exchanger"
                        cost_factor = 200

                    estimated_cost = recoverable * cost_factor
                    annual_savings = (recoverable * operating_hours_per_year *
                                     self.energy_cost)

                    options.append(HeatRecoveryOption(
                        source_name=source["name"],
                        source_temperature_c=source["temperature_c"],
                        source_energy_kw=source["energy_kw"],
                        sink_name=sink["name"],
                        sink_temperature_c=sink["temperature_c"],
                        recoverable_kw=self._round_value(recoverable),
                        recovery_efficiency=self._round_value(recovery_efficiency),
                        equipment_type=equipment,
                        estimated_cost=self._round_value(estimated_cost),
                        annual_savings=self._round_value(annual_savings)
                    ))

        # Sort by annual savings
        options.sort(key=lambda x: -x.annual_savings)

        return options

    def calculate_roi(
        self,
        capital_cost: float,
        annual_energy_savings_kwh: float,
        annual_maintenance_cost: float = 0,
        project_lifetime_years: int = 15
    ) -> ROICalculation:
        """Calculate return on investment for an improvement.

        Args:
            capital_cost: Initial investment cost
            annual_energy_savings_kwh: Annual energy savings (kWh)
            annual_maintenance_cost: Additional annual maintenance
            project_lifetime_years: Expected project lifetime

        Returns:
            ROICalculation with financial metrics
        """
        annual_savings = annual_energy_savings_kwh * self.energy_cost
        net_annual_benefit = annual_savings - annual_maintenance_cost

        # Simple payback
        if net_annual_benefit > 0:
            payback = capital_cost / net_annual_benefit
        else:
            payback = float('inf')

        # ROI
        if capital_cost > 0:
            roi = (net_annual_benefit / capital_cost) * 100
        else:
            roi = float('inf')

        # NPV over 10 years
        npv = -capital_cost
        for year in range(1, 11):
            pv = net_annual_benefit / pow(1 + self.discount_rate, year)
            npv += pv

        # IRR (simplified estimation)
        # For IRR, find rate where NPV = 0
        # Using approximation: IRR ~ annual_benefit / capital_cost for simple cases
        if capital_cost > 0 and net_annual_benefit > 0:
            irr = (net_annual_benefit / capital_cost) * 100
        else:
            irr = None

        self._add_calculation_step(
            description="Calculate ROI metrics",
            operation="roi_calculation",
            inputs={
                "capital_cost": capital_cost,
                "annual_savings": annual_savings,
                "annual_maintenance": annual_maintenance_cost
            },
            output_value=payback,
            output_name="simple_payback_years",
            formula="Payback = Capital / Net Annual Benefit"
        )

        return ROICalculation(
            capital_cost=self._round_value(capital_cost),
            annual_savings=self._round_value(annual_savings),
            annual_maintenance=self._round_value(annual_maintenance_cost),
            simple_payback_years=self._round_value(payback),
            roi_percent=self._round_value(roi),
            npv_10_year=self._round_value(npv),
            irr_percent=self._round_value(irr) if irr else None
        )

    def _identify_improvements_for_loss(
        self,
        loss_type: str,
        loss_kw: float
    ) -> List[Tuple[str, float]]:
        """Identify applicable improvements for a loss type.

        Returns list of (template_name, savings_fraction).
        """
        loss_lower = loss_type.lower()
        improvements = []

        if "flue" in loss_lower or "stack" in loss_lower:
            improvements.append(("economizer", 0.40))
            improvements.append(("air_preheater", 0.25))
            improvements.append(("condensing_economizer", 0.60))
            improvements.append(("combustion_optimization", 0.20))

        if "radiation" in loss_lower:
            improvements.append(("insulation_upgrade", 0.50))

        if "convection" in loss_lower:
            improvements.append(("insulation_upgrade", 0.40))

        if "blowdown" in loss_lower:
            improvements.append(("blowdown_heat_recovery", 0.50))

        if "combustion" in loss_lower or "unburned" in loss_lower:
            improvements.append(("combustion_optimization", 0.50))
            improvements.append(("o2_trim_controls", 0.30))

        if not improvements:
            # Default improvements for unclassified losses
            improvements.append(("combustion_optimization", 0.10))

        return improvements

    def _create_opportunity(
        self,
        template_name: str,
        loss_kw: float,
        savings_fraction: float,
        energy_input_kw: float,
        operating_hours: int
    ) -> ImprovementOpportunity:
        """Create an improvement opportunity from template."""
        template = self.OPPORTUNITY_TEMPLATES.get(template_name, {})

        # Calculate savings
        savings_kw = loss_kw * savings_fraction
        savings_percent = (savings_kw / energy_input_kw * 100) if energy_input_kw > 0 else 0

        # Calculate ROI
        cost_per_kw = template.get("typical_cost_per_kw_saved", 200)
        capital_cost = savings_kw * cost_per_kw
        annual_energy_savings = savings_kw * operating_hours

        roi = self.calculate_roi(
            capital_cost=capital_cost,
            annual_energy_savings_kwh=annual_energy_savings
        )

        # Determine priority based on ROI
        if roi.simple_payback_years < 0.5:
            priority = PriorityLevel.CRITICAL
        elif roi.simple_payback_years < 1.0:
            priority = PriorityLevel.HIGH
        elif roi.simple_payback_years < 2.0:
            priority = PriorityLevel.MEDIUM
        elif roi.simple_payback_years < 5.0:
            priority = PriorityLevel.LOW
        else:
            priority = PriorityLevel.INFORMATIONAL

        return ImprovementOpportunity(
            opportunity_id=f"{template_name}_{hash(template_name) % 10000:04d}",
            title=template.get("title", template_name),
            description=template.get("description", ""),
            category=template.get("category", ImprovementCategory.OTHER),
            priority=priority,
            complexity=template.get("complexity", ImplementationComplexity.MODERATE),
            current_value=0,
            target_value=savings_kw,
            energy_savings_kw=self._round_value(savings_kw),
            energy_savings_percent=self._round_value(savings_percent),
            roi=roi,
            implementation_steps=template.get("implementation_steps", []),
            risks=["Actual savings may vary from estimates"]
        )

    def _summarize_by_priority(
        self,
        opportunities: List[ImprovementOpportunity]
    ) -> Dict[str, int]:
        """Summarize opportunities by priority level."""
        summary: Dict[str, int] = {}
        for o in opportunities:
            key = o.priority.value
            summary[key] = summary.get(key, 0) + 1
        return summary

    def _summarize_by_category(
        self,
        opportunities: List[ImprovementOpportunity]
    ) -> Dict[str, float]:
        """Summarize savings by category."""
        summary: Dict[str, float] = {}
        for o in opportunities:
            key = o.category.value
            summary[key] = summary.get(key, 0.0) + o.energy_savings_kw
        return {k: self._round_value(v) for k, v in summary.items()}

    def _create_roadmap(
        self,
        opportunities: List[ImprovementOpportunity]
    ) -> Dict[str, List[str]]:
        """Create phased implementation roadmap."""
        roadmap = {
            "immediate_0_3_months": [],
            "short_term_3_12_months": [],
            "medium_term_1_2_years": [],
            "long_term_2_plus_years": []
        }

        for o in opportunities:
            if o.priority == PriorityLevel.CRITICAL:
                roadmap["immediate_0_3_months"].append(o.title)
            elif o.priority == PriorityLevel.HIGH:
                roadmap["short_term_3_12_months"].append(o.title)
            elif o.priority == PriorityLevel.MEDIUM:
                roadmap["medium_term_1_2_years"].append(o.title)
            else:
                roadmap["long_term_2_plus_years"].append(o.title)

        return roadmap

    def _reset_calculation_state(self) -> None:
        """Reset calculation state."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        energy_input: float,
        efficiency: float,
        losses: Dict[str, float]
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "analyzer": "ImprovementAnalyzer",
            "version": self.VERSION,
            "energy_input_kw": energy_input,
            "efficiency_percent": efficiency,
            "losses": losses
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float) -> float:
        """Round value to precision."""
        if value is None or value == float('inf'):
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * self.precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
