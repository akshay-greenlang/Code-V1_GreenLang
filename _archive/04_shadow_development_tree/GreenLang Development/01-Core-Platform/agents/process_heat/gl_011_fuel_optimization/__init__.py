"""
GL-011 FUELCRAFT - Fuel Optimization Agent

This module implements the GL-011 FUELCRAFT agent for comprehensive fuel
optimization in industrial process heat applications. The agent provides
real-time fuel price integration, heating value calculations, multi-fuel
blending optimization, fuel switching automation, and inventory management.

Key Features:
    - Real-time fuel price integration (Henry Hub, Brent, regional markets)
    - Fuel heating value calculations (HHV, LHV, Wobbe Index)
    - Multi-fuel blending optimization with emission constraints
    - Automated fuel switching with economic trigger points
    - Fuel inventory management with delivery scheduling
    - Total cost of ownership optimization
    - Zero-hallucination deterministic calculations

Example:
    >>> from greenlang.agents.process_heat.gl_011_fuel_optimization import (
    ...     FuelOptimizationAgent,
    ...     FuelOptimizationConfig,
    ...     FuelOptimizationInput,
    ... )
    >>>
    >>> config = FuelOptimizationConfig(
    ...     facility_id="PLANT-001",
    ...     primary_fuel="natural_gas",
    ... )
    >>> agent = FuelOptimizationAgent(config)
    >>> result = agent.process(input_data)

Author: GreenLang Team
Version: 1.0.0
License: Proprietary
"""

from greenlang.agents.process_heat.gl_011_fuel_optimization.config import (
    FuelOptimizationConfig,
    FuelPricingConfig,
    BlendingConfig,
    SwitchingConfig,
    InventoryConfig,
    CostOptimizationConfig,
    FuelType,
    PriceSource,
    OptimizationMode,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.schemas import (
    FuelOptimizationInput,
    FuelOptimizationOutput,
    FuelProperties,
    FuelPrice,
    BlendRecommendation,
    SwitchingRecommendation,
    InventoryStatus,
    CostAnalysis,
    OptimizationResult,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.heating_value import (
    HeatingValueCalculator,
    HeatingValueInput,
    HeatingValueResult,
    WobbeIndexResult,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_pricing import (
    FuelPricingService,
    PriceQuote,
    PriceHistory,
    PriceForecast,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_blending import (
    FuelBlendingOptimizer,
    BlendInput,
    BlendOutput,
    BlendConstraints,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.fuel_switching import (
    FuelSwitchingController,
    SwitchingInput,
    SwitchingOutput,
    SwitchingTrigger,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.inventory import (
    InventoryManager,
    TankStatus,
    DeliverySchedule,
    InventoryAlert,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.cost_optimization import (
    CostOptimizer,
    TotalCostInput,
    TotalCostOutput,
    CostBreakdown,
)

from greenlang.agents.process_heat.gl_011_fuel_optimization.optimizer import (
    FuelOptimizationAgent,
)

__all__ = [
    # Main Agent
    "FuelOptimizationAgent",
    # Configuration
    "FuelOptimizationConfig",
    "FuelPricingConfig",
    "BlendingConfig",
    "SwitchingConfig",
    "InventoryConfig",
    "CostOptimizationConfig",
    "FuelType",
    "PriceSource",
    "OptimizationMode",
    # Schemas
    "FuelOptimizationInput",
    "FuelOptimizationOutput",
    "FuelProperties",
    "FuelPrice",
    "BlendRecommendation",
    "SwitchingRecommendation",
    "InventoryStatus",
    "CostAnalysis",
    "OptimizationResult",
    # Heating Value
    "HeatingValueCalculator",
    "HeatingValueInput",
    "HeatingValueResult",
    "WobbeIndexResult",
    # Pricing
    "FuelPricingService",
    "PriceQuote",
    "PriceHistory",
    "PriceForecast",
    # Blending
    "FuelBlendingOptimizer",
    "BlendInput",
    "BlendOutput",
    "BlendConstraints",
    # Switching
    "FuelSwitchingController",
    "SwitchingInput",
    "SwitchingOutput",
    "SwitchingTrigger",
    # Inventory
    "InventoryManager",
    "TankStatus",
    "DeliverySchedule",
    "InventoryAlert",
    # Cost Optimization
    "CostOptimizer",
    "TotalCostInput",
    "TotalCostOutput",
    "CostBreakdown",
]

__version__ = "1.0.0"
__agent_id__ = "GL-011"
__agent_name__ = "FUELCRAFT"
