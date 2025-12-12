"""
ISO 50001 Energy Management System (EnMS) Module

This module implements the ISO 50001:2018 Energy Management System standard
for systematic energy performance improvement within the GL-Agent-Factory platform.

Key ISO 50001 Concepts:
- Energy Performance Indicators (EnPIs)
- Energy Baselines (EnBs)
- Significant Energy Uses (SEUs)
- Energy Performance Improvement Plans

References:
- ISO 50001:2018 Energy management systems
- ISO 50006:2014 Measuring energy performance using energy baselines
- ISO 50015:2014 Measurement and verification of energy performance
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Models and Enums
# =============================================================================


class EnergyType(Enum):
    """Types of energy sources."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    FUEL_OIL = "fuel_oil"
    DIESEL = "diesel"
    LPG = "lpg"
    COAL = "coal"
    BIOMASS = "biomass"
    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    CHILLED_WATER = "chilled_water"
    SOLAR = "solar"
    WIND = "wind"


class RelevantVariableType(Enum):
    """Types of relevant variables affecting energy consumption."""

    PRODUCTION = "production"
    WEATHER = "weather"
    OCCUPANCY = "occupancy"
    OPERATING_HOURS = "operating_hours"
    DEGREE_DAYS = "degree_days"
    THROUGHPUT = "throughput"


class EnPIType(Enum):
    """Types of Energy Performance Indicators."""

    ABSOLUTE = "absolute"  # Total energy consumption
    INTENSITY = "intensity"  # Energy per unit (e.g., kWh/unit)
    RATIO = "ratio"  # Energy type ratio
    REGRESSION = "regression"  # Statistical model-based
    TARGET = "target"  # Target-based performance


class ActionStatus(Enum):
    """Status of energy improvement actions."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    CANCELLED = "cancelled"


@dataclass
class EnergyData:
    """Energy consumption data point."""

    timestamp: datetime
    energy_type: EnergyType
    consumption: Decimal
    unit: str
    cost: Optional[Decimal] = None
    co2_emissions: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelevantVariable:
    """A variable that affects energy consumption."""

    name: str
    variable_type: RelevantVariableType
    value: Decimal
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnPIDefinition:
    """Definition of an Energy Performance Indicator."""

    id: str
    name: str
    enpi_type: EnPIType
    energy_types: List[EnergyType]
    numerator_unit: str
    denominator_unit: Optional[str] = None
    relevant_variables: List[str] = field(default_factory=list)
    target_value: Optional[Decimal] = None
    boundary: str = "facility"
    description: str = ""


@dataclass
class EnPIValue:
    """Calculated EnPI value."""

    enpi_id: str
    period_start: datetime
    period_end: datetime
    value: Decimal
    baseline_value: Optional[Decimal] = None
    improvement_pct: Optional[float] = None
    status: str = "normal"  # "normal", "alert", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyBaseline:
    """Energy baseline (EnB) for performance comparison."""

    id: str
    name: str
    period_start: datetime
    period_end: datetime
    energy_types: List[EnergyType]
    total_consumption: Dict[EnergyType, Decimal]
    relevant_variables: Dict[str, List[Decimal]]
    regression_coefficients: Optional[Dict[str, Decimal]] = None
    r_squared: Optional[float] = None
    notes: str = ""


@dataclass
class SignificantEnergyUse:
    """Significant Energy Use (SEU) as defined by ISO 50001."""

    id: str
    name: str
    energy_types: List[EnergyType]
    percentage_of_total: float
    location: str
    equipment: List[str]
    relevant_variables: List[str]
    improvement_potential: float
    priority: int  # 1=highest
    current_consumption: Decimal
    baseline_consumption: Decimal
    notes: str = ""


@dataclass
class EnergyAction:
    """Energy improvement action."""

    id: str
    title: str
    description: str
    seu_id: Optional[str] = None
    status: ActionStatus = ActionStatus.PLANNED
    estimated_savings_kwh: Decimal = Decimal("0")
    estimated_cost: Decimal = Decimal("0")
    actual_savings_kwh: Optional[Decimal] = None
    payback_years: Optional[Decimal] = None
    responsible_person: str = ""
    start_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None
    verification_date: Optional[datetime] = None


@dataclass
class EnMSReport:
    """Energy Management System report."""

    period_start: datetime
    period_end: datetime
    total_energy_consumption: Dict[EnergyType, Decimal]
    total_energy_cost: Decimal
    total_co2_emissions: Decimal
    enpi_values: List[EnPIValue]
    seus: List[SignificantEnergyUse]
    actions: List[EnergyAction]
    improvement_vs_baseline_pct: float
    targets_achieved: bool
    recommendations: List[str]


# =============================================================================
# EnMS Service
# =============================================================================


class ISO50001EnMSService:
    """
    ISO 50001 Energy Management System Service.

    Provides:
    - Energy data collection and management
    - EnPI calculation and monitoring
    - Baseline management
    - SEU identification and tracking
    - Action planning and verification
    - Reporting and analytics
    """

    # Energy conversion factors to kWh
    CONVERSION_FACTORS = {
        EnergyType.ELECTRICITY: Decimal("1"),  # Already kWh
        EnergyType.NATURAL_GAS: Decimal("10.55"),  # kWh per m³
        EnergyType.FUEL_OIL: Decimal("10.35"),  # kWh per liter
        EnergyType.DIESEL: Decimal("10.0"),  # kWh per liter
        EnergyType.LPG: Decimal("7.08"),  # kWh per liter
        EnergyType.COAL: Decimal("8.14"),  # kWh per kg
        EnergyType.STEAM: Decimal("0.7"),  # kWh per kg
    }

    # CO2 emission factors (kg CO2 per kWh)
    CO2_FACTORS = {
        EnergyType.ELECTRICITY: Decimal("0.4"),  # Grid average
        EnergyType.NATURAL_GAS: Decimal("0.2"),
        EnergyType.FUEL_OIL: Decimal("0.27"),
        EnergyType.DIESEL: Decimal("0.27"),
        EnergyType.LPG: Decimal("0.21"),
        EnergyType.COAL: Decimal("0.34"),
        EnergyType.BIOMASS: Decimal("0.0"),  # Carbon neutral
    }

    def __init__(self):
        """Initialize the EnMS service."""
        self._energy_data: List[EnergyData] = []
        self._variables: List[RelevantVariable] = []
        self._enpis: Dict[str, EnPIDefinition] = {}
        self._baselines: Dict[str, EnergyBaseline] = {}
        self._seus: Dict[str, SignificantEnergyUse] = {}
        self._actions: Dict[str, EnergyAction] = {}

        # Initialize default EnPIs
        self._init_default_enpis()

    def _init_default_enpis(self) -> None:
        """Initialize default EnPI definitions."""
        default_enpis = [
            EnPIDefinition(
                id="enpi_total_energy",
                name="Total Energy Consumption",
                enpi_type=EnPIType.ABSOLUTE,
                energy_types=list(EnergyType),
                numerator_unit="kWh",
                description="Total energy consumption across all sources",
            ),
            EnPIDefinition(
                id="enpi_energy_intensity",
                name="Energy Intensity",
                enpi_type=EnPIType.INTENSITY,
                energy_types=[EnergyType.ELECTRICITY, EnergyType.NATURAL_GAS],
                numerator_unit="kWh",
                denominator_unit="unit",
                relevant_variables=["production"],
                description="Energy consumption per production unit",
            ),
            EnPIDefinition(
                id="enpi_specific_consumption",
                name="Specific Energy Consumption",
                enpi_type=EnPIType.INTENSITY,
                energy_types=list(EnergyType),
                numerator_unit="kWh",
                denominator_unit="m²",
                description="Energy consumption per floor area",
            ),
        ]

        for enpi in default_enpis:
            self._enpis[enpi.id] = enpi

    # =========================================================================
    # Data Management
    # =========================================================================

    def add_energy_data(self, data: EnergyData) -> None:
        """Add energy consumption data."""
        # Calculate CO2 if not provided
        if data.co2_emissions is None:
            kwh = self._convert_to_kwh(data.consumption, data.energy_type, data.unit)
            co2_factor = self.CO2_FACTORS.get(data.energy_type, Decimal("0.3"))
            data.co2_emissions = kwh * co2_factor

        self._energy_data.append(data)
        logger.debug(f"Added energy data: {data.energy_type.value} = {data.consumption}")

    def add_relevant_variable(self, variable: RelevantVariable) -> None:
        """Add relevant variable data."""
        self._variables.append(variable)

    def _convert_to_kwh(
        self, value: Decimal, energy_type: EnergyType, unit: str
    ) -> Decimal:
        """Convert energy value to kWh."""
        if unit.lower() == "kwh":
            return value

        factor = self.CONVERSION_FACTORS.get(energy_type, Decimal("1"))
        return value * factor

    # =========================================================================
    # EnPI Management
    # =========================================================================

    def define_enpi(self, enpi: EnPIDefinition) -> None:
        """Define a new EnPI."""
        self._enpis[enpi.id] = enpi
        logger.info(f"Defined EnPI: {enpi.name}")

    def calculate_enpi(
        self,
        enpi_id: str,
        period_start: datetime,
        period_end: datetime,
    ) -> EnPIValue:
        """Calculate EnPI value for a period."""
        enpi = self._enpis.get(enpi_id)
        if not enpi:
            raise ValueError(f"Unknown EnPI: {enpi_id}")

        # Get energy data for period
        period_data = [
            d for d in self._energy_data
            if period_start <= d.timestamp <= period_end
            and d.energy_type in enpi.energy_types
        ]

        # Calculate based on type
        if enpi.enpi_type == EnPIType.ABSOLUTE:
            value = self._calculate_absolute_enpi(period_data, enpi)
        elif enpi.enpi_type == EnPIType.INTENSITY:
            value = self._calculate_intensity_enpi(
                period_data, enpi, period_start, period_end
            )
        else:
            value = self._calculate_absolute_enpi(period_data, enpi)

        # Compare to baseline if available
        baseline_value = None
        improvement_pct = None
        for baseline in self._baselines.values():
            if enpi_id in baseline.id:
                baseline_value = sum(baseline.total_consumption.values())
                if baseline_value > 0:
                    improvement_pct = float(
                        (baseline_value - value) / baseline_value * 100
                    )
                break

        # Determine status
        status = "normal"
        if enpi.target_value:
            if value > enpi.target_value * Decimal("1.1"):
                status = "critical"
            elif value > enpi.target_value:
                status = "alert"

        return EnPIValue(
            enpi_id=enpi_id,
            period_start=period_start,
            period_end=period_end,
            value=value,
            baseline_value=baseline_value,
            improvement_pct=improvement_pct,
            status=status,
        )

    def _calculate_absolute_enpi(
        self, data: List[EnergyData], enpi: EnPIDefinition
    ) -> Decimal:
        """Calculate absolute EnPI (total consumption)."""
        total = Decimal("0")
        for d in data:
            kwh = self._convert_to_kwh(d.consumption, d.energy_type, d.unit)
            total += kwh
        return total

    def _calculate_intensity_enpi(
        self,
        data: List[EnergyData],
        enpi: EnPIDefinition,
        period_start: datetime,
        period_end: datetime,
    ) -> Decimal:
        """Calculate intensity EnPI (energy per unit)."""
        total_energy = self._calculate_absolute_enpi(data, enpi)

        # Get relevant variable total
        denominator = Decimal("0")
        for var_name in enpi.relevant_variables:
            var_data = [
                v for v in self._variables
                if v.name == var_name
                and period_start <= v.timestamp <= period_end
            ]
            denominator += sum(v.value for v in var_data)

        if denominator == 0:
            return Decimal("0")

        return total_energy / denominator

    # =========================================================================
    # Baseline Management
    # =========================================================================

    def create_baseline(
        self,
        baseline_id: str,
        name: str,
        period_start: datetime,
        period_end: datetime,
        energy_types: Optional[List[EnergyType]] = None,
    ) -> EnergyBaseline:
        """Create an energy baseline from historical data."""
        if energy_types is None:
            energy_types = list(EnergyType)

        # Collect consumption by energy type
        period_data = [
            d for d in self._energy_data
            if period_start <= d.timestamp <= period_end
            and d.energy_type in energy_types
        ]

        total_consumption = {}
        for et in energy_types:
            type_data = [d for d in period_data if d.energy_type == et]
            total = sum(
                self._convert_to_kwh(d.consumption, d.energy_type, d.unit)
                for d in type_data
            )
            total_consumption[et] = total

        # Collect relevant variable values
        relevant_vars = {}
        period_vars = [
            v for v in self._variables
            if period_start <= v.timestamp <= period_end
        ]
        for var in period_vars:
            if var.name not in relevant_vars:
                relevant_vars[var.name] = []
            relevant_vars[var.name].append(var.value)

        baseline = EnergyBaseline(
            id=baseline_id,
            name=name,
            period_start=period_start,
            period_end=period_end,
            energy_types=energy_types,
            total_consumption=total_consumption,
            relevant_variables=relevant_vars,
        )

        self._baselines[baseline_id] = baseline
        logger.info(f"Created baseline: {name}")
        return baseline

    # =========================================================================
    # SEU Management
    # =========================================================================

    def identify_seus(self, threshold_pct: float = 10.0) -> List[SignificantEnergyUse]:
        """Identify Significant Energy Uses based on consumption share."""
        # Calculate total consumption by area/equipment
        # This is a simplified implementation
        total = sum(
            self._convert_to_kwh(d.consumption, d.energy_type, d.unit)
            for d in self._energy_data
        )

        if total == 0:
            return []

        # Group by energy type as proxy for SEUs
        seus = []
        by_type = {}
        for d in self._energy_data:
            if d.energy_type not in by_type:
                by_type[d.energy_type] = Decimal("0")
            by_type[d.energy_type] += self._convert_to_kwh(
                d.consumption, d.energy_type, d.unit
            )

        priority = 1
        for et, consumption in sorted(by_type.items(), key=lambda x: -x[1]):
            pct = float(consumption / total * 100)
            if pct >= threshold_pct:
                seu = SignificantEnergyUse(
                    id=f"seu_{et.value}",
                    name=f"{et.value.replace('_', ' ').title()} Systems",
                    energy_types=[et],
                    percentage_of_total=pct,
                    location="Facility-wide",
                    equipment=[],
                    relevant_variables=[],
                    improvement_potential=10.0,  # Estimated
                    priority=priority,
                    current_consumption=consumption,
                    baseline_consumption=consumption,
                )
                seus.append(seu)
                self._seus[seu.id] = seu
                priority += 1

        logger.info(f"Identified {len(seus)} SEUs")
        return seus

    # =========================================================================
    # Action Management
    # =========================================================================

    def create_action(self, action: EnergyAction) -> None:
        """Create an energy improvement action."""
        self._actions[action.id] = action
        logger.info(f"Created action: {action.title}")

    def update_action_status(
        self, action_id: str, status: ActionStatus, **kwargs
    ) -> None:
        """Update action status."""
        action = self._actions.get(action_id)
        if not action:
            raise ValueError(f"Unknown action: {action_id}")

        action.status = status
        for key, value in kwargs.items():
            if hasattr(action, key):
                setattr(action, key, value)

        logger.info(f"Updated action {action_id} to {status.value}")

    def verify_action_savings(
        self, action_id: str, actual_savings: Decimal
    ) -> None:
        """Verify actual savings from completed action."""
        action = self._actions.get(action_id)
        if not action:
            raise ValueError(f"Unknown action: {action_id}")

        action.actual_savings_kwh = actual_savings
        action.status = ActionStatus.VERIFIED
        action.verification_date = datetime.utcnow()

        logger.info(
            f"Verified action {action_id}: "
            f"{actual_savings} kWh saved "
            f"(estimated: {action.estimated_savings_kwh} kWh)"
        )

    # =========================================================================
    # Reporting
    # =========================================================================

    def generate_report(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> EnMSReport:
        """Generate ISO 50001 compliance report."""
        # Calculate total consumption
        period_data = [
            d for d in self._energy_data
            if period_start <= d.timestamp <= period_end
        ]

        total_consumption = {}
        total_cost = Decimal("0")
        total_co2 = Decimal("0")

        for d in period_data:
            et = d.energy_type
            kwh = self._convert_to_kwh(d.consumption, et, d.unit)
            if et not in total_consumption:
                total_consumption[et] = Decimal("0")
            total_consumption[et] += kwh
            total_cost += d.cost or Decimal("0")
            total_co2 += d.co2_emissions or Decimal("0")

        # Calculate EnPIs
        enpi_values = []
        for enpi_id in self._enpis:
            try:
                value = self.calculate_enpi(enpi_id, period_start, period_end)
                enpi_values.append(value)
            except Exception as e:
                logger.warning(f"Failed to calculate EnPI {enpi_id}: {e}")

        # Get SEUs
        seus = list(self._seus.values())

        # Get actions
        actions = list(self._actions.values())

        # Calculate improvement vs baseline
        improvement = 0.0
        for baseline in self._baselines.values():
            baseline_total = sum(baseline.total_consumption.values())
            current_total = sum(total_consumption.values())
            if baseline_total > 0:
                improvement = float(
                    (baseline_total - current_total) / baseline_total * 100
                )
                break

        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_consumption, seus, actions
        )

        # Check targets
        targets_achieved = all(
            v.status == "normal" for v in enpi_values
        )

        return EnMSReport(
            period_start=period_start,
            period_end=period_end,
            total_energy_consumption=total_consumption,
            total_energy_cost=total_cost,
            total_co2_emissions=total_co2,
            enpi_values=enpi_values,
            seus=seus,
            actions=actions,
            improvement_vs_baseline_pct=improvement,
            targets_achieved=targets_achieved,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        consumption: Dict[EnergyType, Decimal],
        seus: List[SignificantEnergyUse],
        actions: List[EnergyAction],
    ) -> List[str]:
        """Generate energy improvement recommendations."""
        recommendations = []

        # High consumption energy types
        total = sum(consumption.values())
        for et, value in consumption.items():
            if total > 0 and value / total > Decimal("0.3"):
                recommendations.append(
                    f"Consider efficiency improvements for {et.value} "
                    f"({float(value/total*100):.1f}% of total)"
                )

        # SEUs without actions
        seu_ids_with_actions = {a.seu_id for a in actions if a.seu_id}
        for seu in seus:
            if seu.id not in seu_ids_with_actions:
                recommendations.append(
                    f"Develop improvement plan for SEU: {seu.name}"
                )

        # Completed actions pending verification
        for action in actions:
            if action.status == ActionStatus.COMPLETED:
                recommendations.append(
                    f"Verify savings for completed action: {action.title}"
                )

        if not recommendations:
            recommendations.append(
                "Continue monitoring and maintain current performance"
            )

        return recommendations


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ISO50001EnMSService",
    "EnergyData",
    "RelevantVariable",
    "EnPIDefinition",
    "EnPIValue",
    "EnergyBaseline",
    "SignificantEnergyUse",
    "EnergyAction",
    "EnMSReport",
    "EnergyType",
    "RelevantVariableType",
    "EnPIType",
    "ActionStatus",
]
