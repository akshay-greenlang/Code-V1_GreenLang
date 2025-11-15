"""
GL-002 BoilerEfficiencyOptimizer Agent Module.

This module implements the BoilerEfficiencyOptimizer agent for optimizing
boiler operations, maximizing fuel efficiency, and minimizing emissions
in industrial facilities.
"""

from .boiler_efficiency_orchestrator import (
    BoilerEfficiencyOptimizer,
    OperationMode,
    OptimizationStrategy,
    BoilerOperationalState
)

from .config import (
    BoilerEfficiencyConfig,
    BoilerConfiguration,
    BoilerSpecification,
    OperationalConstraints,
    EmissionLimits,
    OptimizationParameters,
    IntegrationSettings,
    create_default_config
)

from .tools import (
    BoilerEfficiencyTools,
    CombustionOptimizationResult,
    SteamGenerationStrategy,
    EmissionsOptimizationResult,
    EfficiencyCalculationResult
)

__all__ = [
    # Main orchestrator
    'BoilerEfficiencyOptimizer',
    'OperationMode',
    'OptimizationStrategy',
    'BoilerOperationalState',

    # Configuration
    'BoilerEfficiencyConfig',
    'BoilerConfiguration',
    'BoilerSpecification',
    'OperationalConstraints',
    'EmissionLimits',
    'OptimizationParameters',
    'IntegrationSettings',
    'create_default_config',

    # Tools and results
    'BoilerEfficiencyTools',
    'CombustionOptimizationResult',
    'SteamGenerationStrategy',
    'EmissionsOptimizationResult',
    'EfficiencyCalculationResult'
]

__version__ = '1.0.0'
__agent_id__ = 'GL-002'
__agent_name__ = 'BoilerEfficiencyOptimizer'