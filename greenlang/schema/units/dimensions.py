"""
Physical Dimension Definitions for GL-FOUND-X-002.

This module defines physical dimensions used in the GreenLang framework.

Dimensions:
    - energy: kWh, Wh, J, BTU
    - mass: kg, g, lb, tonne
    - length: m, km, ft, mile
    - area: m2, km2, ha, acre
    - volume: m3, L, gallon
    - temperature: K, C, F
    - time: s, min, h, day
    - power: W, kW, MW
    - emissions: kgCO2e, tCO2e
    - currency: USD, EUR, GBP
"""

from typing import Dict, List


class Dimension:
    """
    Physical dimension definition.

    Attributes:
        name: Dimension name (e.g., "energy")
        si_unit: SI base unit for this dimension
        description: Human-readable description
    """

    def __init__(
        self,
        name: str,
        si_unit: str,
        description: str = "",
    ):
        self.name = name
        self.si_unit = si_unit
        self.description = description


# Standard dimensions
DIMENSIONS: Dict[str, Dimension] = {
    "energy": Dimension("energy", "J", "Energy or work"),
    "mass": Dimension("mass", "kg", "Mass"),
    "length": Dimension("length", "m", "Length or distance"),
    "area": Dimension("area", "m2", "Area"),
    "volume": Dimension("volume", "m3", "Volume"),
    "temperature": Dimension("temperature", "K", "Temperature"),
    "time": Dimension("time", "s", "Time"),
    "power": Dimension("power", "W", "Power"),
    "pressure": Dimension("pressure", "Pa", "Pressure"),
    "force": Dimension("force", "N", "Force"),
    "emissions": Dimension("emissions", "kgCO2e", "Carbon dioxide equivalent emissions"),
    "currency": Dimension("currency", "USD", "Monetary value"),
    "dimensionless": Dimension("dimensionless", "1", "Dimensionless quantity"),
}


def get_dimension(name: str) -> Dimension:
    """Get a dimension by name."""
    if name not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {name}")
    return DIMENSIONS[name]


def list_dimensions() -> List[str]:
    """List all known dimension names."""
    return list(DIMENSIONS.keys())


def is_known_dimension(name: str) -> bool:
    """Check if a dimension name is known."""
    return name in DIMENSIONS
