"""
Scope3CalculatorAgent
GL-VCCI Scope 3 Platform

Production-ready Scope 3 emissions calculator for Categories 1, 4, and 6.

Usage:
    from services.agents.calculator import Scope3CalculatorAgent
    from services.factor_broker import FactorBroker

    # Initialize
    factor_broker = FactorBroker()
    calculator = Scope3CalculatorAgent(factor_broker=factor_broker)

    # Calculate Category 1
    from services.agents.calculator.models import Category1Input
    result = await calculator.calculate_category_1(
        Category1Input(
            product_name="Steel",
            quantity=1000,
            quantity_unit="kg",
            region="US",
            supplier_pcf=1.85
        )
    )

Version: 1.0.0
"""

from .agent import Scope3CalculatorAgent
from .models import (
    Category1Input,
    Category4Input,
    Category6Input,
    Category6FlightInput,
    Category6HotelInput,
    Category6GroundTransportInput,
    CalculationResult,
    BatchResult,
)
from .config import CalculatorConfig, TierType, TransportMode, CabinClass
from .exceptions import *

__version__ = "1.0.0"

__all__ = [
    "Scope3CalculatorAgent",
    "Category1Input",
    "Category4Input",
    "Category6Input",
    "Category6FlightInput",
    "Category6HotelInput",
    "Category6GroundTransportInput",
    "CalculationResult",
    "BatchResult",
    "CalculatorConfig",
    "TierType",
    "TransportMode",
    "CabinClass",
]
