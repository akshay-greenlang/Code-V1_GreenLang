# -*- coding: utf-8 -*-
"""
Measurement Ontology for Process Heat Systems
=============================================

Physical quantities, units of measure, and measurement scales
for industrial process heat applications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable


class MeasurementDomain(str, Enum):
    """Measurement domains in process heat."""
    THERMODYNAMIC = "thermodynamic"
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    CHEMICAL = "chemical"
    FLOW = "flow"
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"


class UnitSystem(str, Enum):
    """Unit systems."""
    SI = "SI"
    IMPERIAL = "Imperial"
    CGS = "CGS"
    METRIC = "Metric"


@dataclass
class UnitOfMeasure:
    """Unit of measure definition."""
    symbol: str
    name: str
    quantity_type: str
    unit_system: UnitSystem
    si_conversion_factor: float = 1.0
    si_conversion_offset: float = 0.0
    qudt_uri: Optional[str] = None

    def to_si(self, value: float) -> float:
        """Convert value to SI units."""
        return value * self.si_conversion_factor + self.si_conversion_offset

    def from_si(self, si_value: float) -> float:
        """Convert SI value to this unit."""
        return (si_value - self.si_conversion_offset) / self.si_conversion_factor


@dataclass
class PhysicalQuantity:
    """Physical quantity definition."""
    id: str
    name: str
    symbol: str
    domain: MeasurementDomain
    description: str
    si_unit: UnitOfMeasure
    dimension: str  # e.g., "L2 M T-2" for energy
    common_units: List[UnitOfMeasure] = field(default_factory=list)
    typical_range: Optional[tuple] = None
    qudt_uri: Optional[str] = None


@dataclass
class MeasurementScale:
    """Measurement scale definition."""
    id: str
    name: str
    scale_type: str  # "ratio", "interval", "ordinal", "nominal"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    reference_point: Optional[float] = None


# =============================================================================
# Standard Units of Measure
# =============================================================================

UNITS = {
    # Temperature
    "K": UnitOfMeasure("K", "kelvin", "temperature", UnitSystem.SI, 1.0, 0.0,
                       "http://qudt.org/vocab/unit/K"),
    "degC": UnitOfMeasure("°C", "degree Celsius", "temperature", UnitSystem.METRIC, 1.0, 273.15,
                          "http://qudt.org/vocab/unit/DEG_C"),
    "degF": UnitOfMeasure("°F", "degree Fahrenheit", "temperature", UnitSystem.IMPERIAL, 5/9, 459.67 * 5/9,
                          "http://qudt.org/vocab/unit/DEG_F"),
    "degR": UnitOfMeasure("°R", "degree Rankine", "temperature", UnitSystem.IMPERIAL, 5/9, 0.0,
                          "http://qudt.org/vocab/unit/DEG_R"),

    # Pressure
    "Pa": UnitOfMeasure("Pa", "pascal", "pressure", UnitSystem.SI, 1.0, 0.0,
                        "http://qudt.org/vocab/unit/PA"),
    "kPa": UnitOfMeasure("kPa", "kilopascal", "pressure", UnitSystem.SI, 1000.0, 0.0),
    "MPa": UnitOfMeasure("MPa", "megapascal", "pressure", UnitSystem.SI, 1e6, 0.0),
    "bar": UnitOfMeasure("bar", "bar", "pressure", UnitSystem.METRIC, 1e5, 0.0,
                         "http://qudt.org/vocab/unit/BAR"),
    "barg": UnitOfMeasure("barg", "bar gauge", "pressure", UnitSystem.METRIC, 1e5, 101325.0),
    "psi": UnitOfMeasure("psi", "pound per square inch", "pressure", UnitSystem.IMPERIAL, 6894.757, 0.0,
                         "http://qudt.org/vocab/unit/PSI"),
    "psig": UnitOfMeasure("psig", "psi gauge", "pressure", UnitSystem.IMPERIAL, 6894.757, 101325.0),
    "atm": UnitOfMeasure("atm", "atmosphere", "pressure", UnitSystem.METRIC, 101325.0, 0.0),
    "mmHg": UnitOfMeasure("mmHg", "millimeter of mercury", "pressure", UnitSystem.METRIC, 133.322, 0.0),

    # Energy/Power
    "J": UnitOfMeasure("J", "joule", "energy", UnitSystem.SI, 1.0, 0.0,
                       "http://qudt.org/vocab/unit/J"),
    "kJ": UnitOfMeasure("kJ", "kilojoule", "energy", UnitSystem.SI, 1000.0, 0.0),
    "MJ": UnitOfMeasure("MJ", "megajoule", "energy", UnitSystem.SI, 1e6, 0.0),
    "GJ": UnitOfMeasure("GJ", "gigajoule", "energy", UnitSystem.SI, 1e9, 0.0),
    "kWh": UnitOfMeasure("kWh", "kilowatt-hour", "energy", UnitSystem.SI, 3.6e6, 0.0),
    "MWh": UnitOfMeasure("MWh", "megawatt-hour", "energy", UnitSystem.SI, 3.6e9, 0.0),
    "Btu": UnitOfMeasure("Btu", "British thermal unit", "energy", UnitSystem.IMPERIAL, 1055.06, 0.0,
                         "http://qudt.org/vocab/unit/BTU_IT"),
    "MMBtu": UnitOfMeasure("MMBtu", "million Btu", "energy", UnitSystem.IMPERIAL, 1.05506e9, 0.0),
    "therm": UnitOfMeasure("therm", "therm", "energy", UnitSystem.IMPERIAL, 1.05506e8, 0.0),

    "W": UnitOfMeasure("W", "watt", "power", UnitSystem.SI, 1.0, 0.0,
                       "http://qudt.org/vocab/unit/W"),
    "kW": UnitOfMeasure("kW", "kilowatt", "power", UnitSystem.SI, 1000.0, 0.0),
    "MW": UnitOfMeasure("MW", "megawatt", "power", UnitSystem.SI, 1e6, 0.0),
    "hp": UnitOfMeasure("hp", "horsepower", "power", UnitSystem.IMPERIAL, 745.7, 0.0),
    "Btu_h": UnitOfMeasure("Btu/h", "Btu per hour", "power", UnitSystem.IMPERIAL, 0.293071, 0.0),
    "MMBtu_h": UnitOfMeasure("MMBtu/h", "million Btu per hour", "power", UnitSystem.IMPERIAL, 293071.0, 0.0),

    # Mass Flow
    "kg_s": UnitOfMeasure("kg/s", "kilogram per second", "mass_flow", UnitSystem.SI, 1.0, 0.0),
    "kg_h": UnitOfMeasure("kg/h", "kilogram per hour", "mass_flow", UnitSystem.SI, 1/3600, 0.0),
    "t_h": UnitOfMeasure("t/h", "tonne per hour", "mass_flow", UnitSystem.METRIC, 1000/3600, 0.0),
    "lb_h": UnitOfMeasure("lb/h", "pound per hour", "mass_flow", UnitSystem.IMPERIAL, 0.453592/3600, 0.0),
    "klb_h": UnitOfMeasure("klb/h", "thousand pounds per hour", "mass_flow", UnitSystem.IMPERIAL, 453.592/3600, 0.0),

    # Volumetric Flow
    "m3_s": UnitOfMeasure("m³/s", "cubic meter per second", "volume_flow", UnitSystem.SI, 1.0, 0.0),
    "m3_h": UnitOfMeasure("m³/h", "cubic meter per hour", "volume_flow", UnitSystem.SI, 1/3600, 0.0),
    "L_min": UnitOfMeasure("L/min", "liter per minute", "volume_flow", UnitSystem.METRIC, 1e-3/60, 0.0),
    "gpm": UnitOfMeasure("gpm", "gallon per minute (US)", "volume_flow", UnitSystem.IMPERIAL, 6.309e-5, 0.0),
    "scfm": UnitOfMeasure("scfm", "standard cubic feet per minute", "volume_flow", UnitSystem.IMPERIAL, 4.719e-4, 0.0),
    "Nm3_h": UnitOfMeasure("Nm³/h", "normal cubic meter per hour", "volume_flow", UnitSystem.METRIC, 1/3600, 0.0),

    # Specific Enthalpy
    "kJ_kg": UnitOfMeasure("kJ/kg", "kilojoule per kilogram", "specific_enthalpy", UnitSystem.SI, 1000.0, 0.0),
    "Btu_lb": UnitOfMeasure("Btu/lb", "Btu per pound", "specific_enthalpy", UnitSystem.IMPERIAL, 2326.0, 0.0),

    # Heat Transfer Coefficient
    "W_m2K": UnitOfMeasure("W/m²K", "watt per square meter kelvin", "htc", UnitSystem.SI, 1.0, 0.0),
    "Btu_hft2F": UnitOfMeasure("Btu/h·ft²·°F", "Btu per hour square foot Fahrenheit", "htc",
                                UnitSystem.IMPERIAL, 5.678, 0.0),

    # Thermal Conductivity
    "W_mK": UnitOfMeasure("W/mK", "watt per meter kelvin", "thermal_conductivity", UnitSystem.SI, 1.0, 0.0),
    "Btu_hftF": UnitOfMeasure("Btu/h·ft·°F", "Btu per hour foot Fahrenheit", "thermal_conductivity",
                              UnitSystem.IMPERIAL, 1.731, 0.0),

    # Concentration
    "pct": UnitOfMeasure("%", "percent", "concentration", UnitSystem.SI, 0.01, 0.0),
    "ppm": UnitOfMeasure("ppm", "parts per million", "concentration", UnitSystem.SI, 1e-6, 0.0),
    "ppb": UnitOfMeasure("ppb", "parts per billion", "concentration", UnitSystem.SI, 1e-9, 0.0),
    "mg_Nm3": UnitOfMeasure("mg/Nm³", "milligram per normal cubic meter", "concentration",
                            UnitSystem.METRIC, 1e-6, 0.0),

    # Dimensionless
    "dimensionless": UnitOfMeasure("-", "dimensionless", "dimensionless", UnitSystem.SI, 1.0, 0.0),
}


# =============================================================================
# Physical Quantity Definitions
# =============================================================================

PHYSICAL_QUANTITIES = {
    # Thermodynamic quantities
    "temperature": PhysicalQuantity(
        id="temperature",
        name="Temperature",
        symbol="T",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Measure of thermal energy in a system",
        si_unit=UNITS["K"],
        dimension="Θ",
        common_units=[UNITS["degC"], UNITS["degF"], UNITS["degR"]],
        typical_range=(200, 2000),  # Kelvin
        qudt_uri="http://qudt.org/vocab/quantitykind/Temperature",
    ),
    "pressure": PhysicalQuantity(
        id="pressure",
        name="Pressure",
        symbol="P",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Force per unit area",
        si_unit=UNITS["Pa"],
        dimension="M L-1 T-2",
        common_units=[UNITS["bar"], UNITS["barg"], UNITS["psi"], UNITS["psig"], UNITS["kPa"], UNITS["MPa"]],
        typical_range=(0, 5e7),  # Pascal
        qudt_uri="http://qudt.org/vocab/quantitykind/Pressure",
    ),
    "specific_enthalpy": PhysicalQuantity(
        id="specific_enthalpy",
        name="Specific Enthalpy",
        symbol="h",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Enthalpy per unit mass",
        si_unit=UNITS["kJ_kg"],
        dimension="L2 T-2",
        common_units=[UNITS["Btu_lb"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/SpecificEnthalpy",
    ),

    # Energy and Power
    "energy": PhysicalQuantity(
        id="energy",
        name="Energy",
        symbol="E",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Capacity to do work or transfer heat",
        si_unit=UNITS["J"],
        dimension="L2 M T-2",
        common_units=[UNITS["kJ"], UNITS["MJ"], UNITS["GJ"], UNITS["kWh"], UNITS["Btu"], UNITS["MMBtu"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/Energy",
    ),
    "power": PhysicalQuantity(
        id="power",
        name="Power",
        symbol="P",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Rate of energy transfer",
        si_unit=UNITS["W"],
        dimension="L2 M T-3",
        common_units=[UNITS["kW"], UNITS["MW"], UNITS["hp"], UNITS["Btu_h"], UNITS["MMBtu_h"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/Power",
    ),
    "heat_flow_rate": PhysicalQuantity(
        id="heat_flow_rate",
        name="Heat Flow Rate",
        symbol="Q̇",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Thermal energy transfer per unit time",
        si_unit=UNITS["W"],
        dimension="L2 M T-3",
        common_units=[UNITS["kW"], UNITS["MW"], UNITS["Btu_h"], UNITS["MMBtu_h"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/HeatFlowRate",
    ),

    # Flow quantities
    "mass_flow_rate": PhysicalQuantity(
        id="mass_flow_rate",
        name="Mass Flow Rate",
        symbol="ṁ",
        domain=MeasurementDomain.FLOW,
        description="Mass per unit time",
        si_unit=UNITS["kg_s"],
        dimension="M T-1",
        common_units=[UNITS["kg_h"], UNITS["t_h"], UNITS["lb_h"], UNITS["klb_h"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/MassFlowRate",
    ),
    "volumetric_flow_rate": PhysicalQuantity(
        id="volumetric_flow_rate",
        name="Volumetric Flow Rate",
        symbol="V̇",
        domain=MeasurementDomain.FLOW,
        description="Volume per unit time",
        si_unit=UNITS["m3_s"],
        dimension="L3 T-1",
        common_units=[UNITS["m3_h"], UNITS["L_min"], UNITS["gpm"], UNITS["scfm"], UNITS["Nm3_h"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/VolumeFlowRate",
    ),

    # Heat Transfer
    "heat_transfer_coefficient": PhysicalQuantity(
        id="heat_transfer_coefficient",
        name="Heat Transfer Coefficient",
        symbol="U",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Heat flux per unit temperature difference",
        si_unit=UNITS["W_m2K"],
        dimension="M T-3 Θ-1",
        common_units=[UNITS["Btu_hft2F"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/HeatTransferCoefficient",
    ),
    "thermal_conductivity": PhysicalQuantity(
        id="thermal_conductivity",
        name="Thermal Conductivity",
        symbol="k",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Ability to conduct heat",
        si_unit=UNITS["W_mK"],
        dimension="L M T-3 Θ-1",
        common_units=[UNITS["Btu_hftF"]],
        qudt_uri="http://qudt.org/vocab/quantitykind/ThermalConductivity",
    ),

    # Efficiency
    "thermal_efficiency": PhysicalQuantity(
        id="thermal_efficiency",
        name="Thermal Efficiency",
        symbol="η",
        domain=MeasurementDomain.THERMODYNAMIC,
        description="Ratio of useful heat output to heat input",
        si_unit=UNITS["dimensionless"],
        dimension="1",
        common_units=[UNITS["pct"]],
        typical_range=(0, 1),
        qudt_uri="http://qudt.org/vocab/quantitykind/Efficiency",
    ),

    # Emissions
    "co2_concentration": PhysicalQuantity(
        id="co2_concentration",
        name="CO2 Concentration",
        symbol="CO₂",
        domain=MeasurementDomain.ENVIRONMENTAL,
        description="Carbon dioxide concentration in flue gas",
        si_unit=UNITS["pct"],
        dimension="1",
        common_units=[UNITS["ppm"]],
        typical_range=(0, 20),  # percent
    ),
    "nox_concentration": PhysicalQuantity(
        id="nox_concentration",
        name="NOx Concentration",
        symbol="NOx",
        domain=MeasurementDomain.ENVIRONMENTAL,
        description="Nitrogen oxides concentration",
        si_unit=UNITS["ppm"],
        dimension="1",
        common_units=[UNITS["mg_Nm3"]],
        typical_range=(0, 500),  # ppm
    ),
    "o2_concentration": PhysicalQuantity(
        id="o2_concentration",
        name="O2 Concentration",
        symbol="O₂",
        domain=MeasurementDomain.ENVIRONMENTAL,
        description="Oxygen concentration in flue gas",
        si_unit=UNITS["pct"],
        dimension="1",
        typical_range=(0, 21),  # percent
    ),
}


# =============================================================================
# Measurement Ontology Manager
# =============================================================================

class MeasurementOntology:
    """
    Manager for measurement ontology including quantities, units, and conversions.
    """

    def __init__(self):
        self.units = UNITS.copy()
        self.quantities = PHYSICAL_QUANTITIES.copy()
        self.scales: Dict[str, MeasurementScale] = {}
        self._initialize_scales()

    def _initialize_scales(self):
        """Initialize measurement scales."""
        self.scales["ratio"] = MeasurementScale(
            id="ratio",
            name="Ratio Scale",
            scale_type="ratio",
            min_value=0,
        )
        self.scales["interval"] = MeasurementScale(
            id="interval",
            name="Interval Scale",
            scale_type="interval",
        )
        self.scales["ordinal"] = MeasurementScale(
            id="ordinal",
            name="Ordinal Scale",
            scale_type="ordinal",
        )

    def get_unit(self, symbol: str) -> Optional[UnitOfMeasure]:
        """Get unit by symbol."""
        return self.units.get(symbol)

    def get_quantity(self, quantity_id: str) -> Optional[PhysicalQuantity]:
        """Get physical quantity by ID."""
        return self.quantities.get(quantity_id)

    def convert(self, value: float, from_unit: str, to_unit: str) -> float:
        """Convert value between units."""
        from_u = self.units.get(from_unit)
        to_u = self.units.get(to_unit)

        if not from_u or not to_u:
            raise ValueError(f"Unknown unit: {from_unit if not from_u else to_unit}")

        if from_u.quantity_type != to_u.quantity_type:
            raise ValueError(f"Cannot convert between {from_u.quantity_type} and {to_u.quantity_type}")

        # Convert to SI, then to target unit
        si_value = from_u.to_si(value)
        return to_u.from_si(si_value)

    def get_units_for_quantity(self, quantity_id: str) -> List[UnitOfMeasure]:
        """Get all units for a physical quantity."""
        quantity = self.quantities.get(quantity_id)
        if not quantity:
            return []
        return [quantity.si_unit] + quantity.common_units

    def get_quantities_by_domain(self, domain: MeasurementDomain) -> List[PhysicalQuantity]:
        """Get quantities by measurement domain."""
        return [q for q in self.quantities.values() if q.domain == domain]

    def validate_value(self, value: float, quantity_id: str, unit: str) -> tuple:
        """Validate a measurement value against quantity constraints."""
        quantity = self.quantities.get(quantity_id)
        if not quantity:
            return False, f"Unknown quantity: {quantity_id}"

        # Convert to SI for validation
        si_value = self.convert(value, unit, quantity.si_unit.symbol)

        if quantity.typical_range:
            min_val, max_val = quantity.typical_range
            if si_value < min_val * 0.1 or si_value > max_val * 10:
                return False, f"Value {value} {unit} is outside typical range"

        return True, "Valid"

    def get_statistics(self) -> Dict[str, int]:
        """Get ontology statistics."""
        return {
            "total_units": len(self.units),
            "total_quantities": len(self.quantities),
            "domains": len(set(q.domain for q in self.quantities.values())),
        }


# Module-level singleton
_measurement_ontology: Optional[MeasurementOntology] = None

def get_measurement_ontology() -> MeasurementOntology:
    """Get or create the measurement ontology instance."""
    global _measurement_ontology
    if _measurement_ontology is None:
        _measurement_ontology = MeasurementOntology()
    return _measurement_ontology
