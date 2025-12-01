# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT - Multi-Fuel Selection Optimizer Agent.

This agent provides comprehensive fuel management optimization for industrial
facilities including power plants, manufacturing sites, and process industries.

Core Capabilities:
- Multi-fuel optimization (coal, natural gas, biomass, hydrogen, fuel oil)
- Cost optimization with delivery and inventory considerations
- Fuel blending optimization for solid fuels
- Carbon footprint minimization following GHG Protocol
- Economic order quantity (EOQ) procurement optimization
- Real-time market price integration

Standards Compliance:
- ISO 6976:2016 - Natural gas calorific value calculation
- ISO 17225 - Solid biofuels specifications
- ASTM D4809 - Bomb calorimetric heat determination
- GHG Protocol - Scope 1 emissions accounting
- IPCC Guidelines - Emission factor databases

Zero-Hallucination Guarantee:
- All calculations use deterministic formulas
- Emission factors from validated databases
- Provenance tracking with SHA-256 hashes
- temperature=0.0, seed=42 for reproducibility

Example:
    >>> from GL011 import FuelManagementOrchestrator
    >>> from GL011.config import FuelManagementConfig
    >>>
    >>> config = FuelManagementConfig(environment='production')
    >>> orchestrator = FuelManagementOrchestrator(config)
    >>>
    >>> request = FuelOptimizationRequest(
    ...     energy_demand_mw=100,
    ...     optimization_objective='balanced',
    ...     available_fuels=['natural_gas', 'coal', 'biomass']
    ... )
    >>>
    >>> response = orchestrator.execute(request)
    >>> print(f"Optimal mix: {response.optimal_fuel_mix}")
    >>> print(f"Total cost: ${response.total_cost_usd:,.2f}")
    >>> print(f"Carbon intensity: {response.carbon_intensity_kg_mwh:.1f} kg/MWh")

Author: GreenLang Team
Version: 1.0.0
License: Proprietary
"""

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__agent_id__ = "GL-011"
__agent_name__ = "FUELCRAFT"

from .fuel_management_orchestrator import (
    FuelManagementOrchestrator,
    FuelOptimizationRequest,
    FuelOptimizationResponse,
    FuelOperationalState,
    ThreadSafeCache
)

from .config import (
    FuelManagementConfig,
    FuelSpecification,
    FuelInventory,
    MarketPriceData,
    BlendingConstraints,
    EmissionLimits,
    OptimizationObjective
)

from .tools import FuelManagementTools

# Calculator exports
from .calculators import (
    MultiFuelOptimizer,
    CostOptimizationCalculator,
    FuelBlendingCalculator,
    CarbonFootprintCalculator,
    CalorificValueCalculator,
    EmissionsFactorCalculator,
    ProcurementOptimizer,
    ProvenanceTracker
)

# Integration exports
from .integrations import (
    FuelStorageConnector,
    ProcurementSystemConnector,
    MarketPriceConnector,
    EmissionsMonitoringConnector
)

# Monitoring exports
from .monitoring import (
    FuelManagementMetrics,
    HealthChecker,
    HealthStatus,
    SystemHealthReport
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__agent_id__",
    "__agent_name__",

    # Core orchestrator
    "FuelManagementOrchestrator",
    "FuelOptimizationRequest",
    "FuelOptimizationResponse",
    "FuelOperationalState",
    "ThreadSafeCache",

    # Configuration
    "FuelManagementConfig",
    "FuelSpecification",
    "FuelInventory",
    "MarketPriceData",
    "BlendingConstraints",
    "EmissionLimits",
    "OptimizationObjective",

    # Tools
    "FuelManagementTools",

    # Calculators
    "MultiFuelOptimizer",
    "CostOptimizationCalculator",
    "FuelBlendingCalculator",
    "CarbonFootprintCalculator",
    "CalorificValueCalculator",
    "EmissionsFactorCalculator",
    "ProcurementOptimizer",
    "ProvenanceTracker",

    # Integrations
    "FuelStorageConnector",
    "ProcurementSystemConnector",
    "MarketPriceConnector",
    "EmissionsMonitoringConnector",

    # Monitoring
    "FuelManagementMetrics",
    "HealthChecker",
    "HealthStatus",
    "SystemHealthReport"
]
