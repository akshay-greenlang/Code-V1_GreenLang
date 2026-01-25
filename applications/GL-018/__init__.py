# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW - Flue Gas Analyzer Agent Package.

This package provides comprehensive flue gas analysis, combustion efficiency
optimization, and emissions compliance monitoring for industrial burners
and boilers.

Main Components:
    - FlueGasAnalyzerAgent: Main orchestrator for flue gas analysis
    - Data Models: Flue gas composition, burner operation, emissions compliance
    - Configuration: Burner and agent configuration models

Example Usage:
    >>> from greenlang.GL_018 import FlueGasAnalyzerAgent, AgentConfiguration
    >>> from greenlang.GL_018.config import (
    ...     BurnerConfiguration,
    ...     FuelSpecification,
    ...     EmissionsLimits,
    ...     SCADAIntegration,
    ... )
    >>>
    >>> # Create configuration
    >>> fuel_spec = FuelSpecification(
    ...     fuel_type="natural_gas",
    ...     fuel_name="Pipeline Natural Gas",
    ...     higher_heating_value_btu_scf=1020,
    ... )
    >>>
    >>> emissions_limits = EmissionsLimits(
    ...     emissions_standard="epa_nsps",
    ...     nox_limit_ppm=30.0,
    ...     co_limit_ppm=400.0,
    ... )
    >>>
    >>> burner_config = BurnerConfiguration(
    ...     burner_id="BURNER-001",
    ...     burner_type="low_nox",
    ...     design_firing_rate_mmbtu_hr=60.0,
    ...     minimum_firing_rate_mmbtu_hr=15.0,
    ...     fuel_specification=fuel_spec,
    ...     emissions_standard="epa_nsps",
    ...     emissions_limits=emissions_limits,
    ... )
    >>>
    >>> scada_config = SCADAIntegration(
    ...     enabled=True,
    ...     scada_system="Wonderware",
    ...     polling_interval_seconds=60,
    ... )
    >>>
    >>> agent_config = AgentConfiguration(
    ...     burners=[burner_config],
    ...     scada_integration=scada_config,
    ...     analysis_interval_seconds=60,
    ...     auto_optimization_enabled=False,
    ... )
    >>>
    >>> # Initialize agent
    >>> agent = FlueGasAnalyzerAgent(agent_config)
    >>>
    >>> # Execute analysis
    >>> import asyncio
    >>> result = asyncio.run(agent.execute())
    >>>
    >>> # Check results
    >>> print(f"System Status: {result.system_status}")
    >>> print(f"Combustion Efficiency: {result.combustion_analysis.combustion_efficiency_pct:.1f}%")
    >>> print(f"Emissions Compliance: {result.emissions_compliance.overall_compliance_status}")

Author: GreenLang Team
Date: December 2025
Version: 1.0.0
Status: Production Ready
"""

from greenlang.GL_018.flue_gas_analyzer_agent import (
    FlueGasAnalyzerAgent,
    FlueGasCompositionData,
    BurnerOperatingData,
    CombustionAnalysisResult,
    EfficiencyAssessment,
    AirFuelRatioRecommendation,
    EmissionsComplianceReport,
    FlueGasAnalysisResult,
)

from greenlang.GL_018.config import (
    BurnerType,
    FuelType,
    EmissionsStandard,
    AnalyzerType,
    ControlStrategy,
    FuelSpecification,
    EmissionsLimits,
    BurnerConfiguration,
    SCADAIntegration,
    AgentConfiguration,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"
__status__ = "Production Ready"

__all__ = [
    # Main Agent
    "FlueGasAnalyzerAgent",
    # Data Models
    "FlueGasCompositionData",
    "BurnerOperatingData",
    "CombustionAnalysisResult",
    "EfficiencyAssessment",
    "AirFuelRatioRecommendation",
    "EmissionsComplianceReport",
    "FlueGasAnalysisResult",
    # Configuration Enums
    "BurnerType",
    "FuelType",
    "EmissionsStandard",
    "AnalyzerType",
    "ControlStrategy",
    # Configuration Models
    "FuelSpecification",
    "EmissionsLimits",
    "BurnerConfiguration",
    "SCADAIntegration",
    "AgentConfiguration",
]
