"""
GL-011 FuelCraft - Calculators Package

Zero-hallucination calculation engines for multi-fuel procurement,
storage, and blending optimization.

All calculators are DETERMINISTIC with SHA-256 provenance tracking.
Follows GreenLang Global AI Standards v2.0.

Calculators:
- UnitConverter: Energy, mass, volume conversions with provenance
- HeatingValueCalculator: LHV/HHV and energy content calculations
- BlendCalculator: Linear/non-linear blending with quality constraints
- CarbonCalculator: Carbon intensity and emissions calculations
- InventoryCalculator: Tank level and inventory balance calculations
"""

from calculators.unit_converter import (
    UnitConverter,
    EnergyConverter,
    MassConverter,
    VolumeConverter,
    ConversionResult,
    ConversionFactor,
)
from calculators.heating_value_calculator import (
    HeatingValueCalculator,
    HeatingValueInput,
    HeatingValueResult,
    FuelProperties,
)
from calculators.blend_calculator import (
    BlendCalculator,
    BlendInput,
    BlendResult,
    BlendComponent,
    QualityConstraint,
    SafetyConstraint,
)
from calculators.carbon_calculator import (
    CarbonCalculator,
    CarbonInput,
    CarbonResult,
    EmissionFactor,
    EmissionBoundary,
)
from calculators.inventory_calculator import (
    InventoryCalculator,
    InventoryInput,
    InventoryResult,
    TankState,
    InventoryTransaction,
)


__all__ = [
    # Unit Converter
    "UnitConverter",
    "EnergyConverter",
    "MassConverter",
    "VolumeConverter",
    "ConversionResult",
    "ConversionFactor",
    # Heating Value
    "HeatingValueCalculator",
    "HeatingValueInput",
    "HeatingValueResult",
    "FuelProperties",
    # Blend Calculator
    "BlendCalculator",
    "BlendInput",
    "BlendResult",
    "BlendComponent",
    "QualityConstraint",
    "SafetyConstraint",
    # Carbon Calculator
    "CarbonCalculator",
    "CarbonInput",
    "CarbonResult",
    "EmissionFactor",
    "EmissionBoundary",
    # Inventory Calculator
    "InventoryCalculator",
    "InventoryInput",
    "InventoryResult",
    "TankState",
    "InventoryTransaction",
]

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-011"
