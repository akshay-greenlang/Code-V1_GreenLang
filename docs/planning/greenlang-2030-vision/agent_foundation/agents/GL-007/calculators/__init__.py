# -*- coding: utf-8 -*-
"""
GL-007 FURNACEPULSE Calculators Module

This module provides deterministic, zero-hallucination calculation engines for
the FurnacePerformanceMonitor agent. All calculators guarantee bit-perfect
reproducibility with complete provenance tracking via SHA-256 hashing.

Standards Compliance:
- ASME PTC 4.2: Performance Test Code on Industrial Furnaces
- ISO 50001: Energy Management Systems
- ISO 50006: Energy Baseline and Energy Performance Indicators
- ISO 17359: Condition Monitoring and Diagnostics of Machines
- ISO 13579-1: Industrial Furnaces and Associated Processing Equipment

Zero-Hallucination Guarantees:
- All calculations use Decimal arithmetic for precision
- No LLM inference in calculation paths
- Complete provenance tracking with SHA-256 hashes
- Deterministic: Same input always produces same output

Module Structure:
    provenance.py - SHA-256 provenance tracking utilities
    thermal_efficiency_calculator.py - ASME PTC 4.2 efficiency calculations
    performance_kpi_calculator.py - Production and energy KPIs
    maintenance_predictor.py - Predictive maintenance calculations

Example:
    >>> from calculators import ThermalEfficiencyCalculator, FurnaceInputData
    >>> calculator = ThermalEfficiencyCalculator()
    >>> input_data = FurnaceInputData(
    ...     furnace_type=FurnaceType.REHEAT_FURNACE,
    ...     fuel_type=FuelType.NATURAL_GAS,
    ...     fuel_consumption_kg_hr=500.0,
    ...     flue_gas_temp_c=185.0,
    ...     flue_gas_o2_percent=3.0
    ... )
    >>> result = calculator.calculate(input_data)
    >>> print(f"Efficiency: {result.thermal_efficiency_lhv_percent}%")
    >>> print(f"Provenance Hash: {result.provenance.provenance_hash}")

Author: GL-CalculatorEngineer
Agent: GL-007 FURNACEPULSE
Version: 1.0.0
"""

# Provenance Tracking
from .provenance import (
    # Core classes
    ProvenanceTracker,
    ProvenanceRecord,
    ProvenanceValidator,
    CalculationStep,
    CalculationCategory,
    # Utility functions
    create_calculation_hash,
    verify_calculation_integrity,
)

# Thermal Efficiency Calculator
from .thermal_efficiency_calculator import (
    # Main calculator
    ThermalEfficiencyCalculator,
    # Data classes
    FurnaceInputData,
    ThermalEfficiencyResult,
    FuelProperties,
    # Enums
    FurnaceType,
    FuelType,
    # Constants
    FUEL_PROPERTIES_DB,
)

# Performance KPI Calculator
from .performance_kpi_calculator import (
    # Main calculator
    PerformanceKPICalculator,
    # Data classes
    FurnaceOperatingData,
    PerformanceKPIResult,
    KPIResult,
    # Enums
    KPICategory,
    ProductUnit,
)

# Maintenance Predictor
from .maintenance_predictor import (
    # Main predictor
    MaintenancePredictor,
    # Data classes
    FurnaceConditionData,
    MaintenanceScheduleResult,
    MaintenancePrediction,
    RefractoryWearResult,
    BurnerDegradationResult,
    RefractoryProperties,
    # Enums
    MaintenanceType,
    ComponentType,
    SeverityLevel,
    RefractoryType,
    # Constants
    REFRACTORY_PROPERTIES_DB,
)

# Version information
__version__ = "1.0.0"
__author__ = "GL-CalculatorEngineer"
__agent__ = "GL-007 FURNACEPULSE"

# Module-level docstring for easy access
__all__ = [
    # Provenance
    "ProvenanceTracker",
    "ProvenanceRecord",
    "ProvenanceValidator",
    "CalculationStep",
    "CalculationCategory",
    "create_calculation_hash",
    "verify_calculation_integrity",
    # Thermal Efficiency
    "ThermalEfficiencyCalculator",
    "FurnaceInputData",
    "ThermalEfficiencyResult",
    "FuelProperties",
    "FurnaceType",
    "FuelType",
    "FUEL_PROPERTIES_DB",
    # Performance KPIs
    "PerformanceKPICalculator",
    "FurnaceOperatingData",
    "PerformanceKPIResult",
    "KPIResult",
    "KPICategory",
    "ProductUnit",
    # Maintenance Prediction
    "MaintenancePredictor",
    "FurnaceConditionData",
    "MaintenanceScheduleResult",
    "MaintenancePrediction",
    "RefractoryWearResult",
    "BurnerDegradationResult",
    "RefractoryProperties",
    "MaintenanceType",
    "ComponentType",
    "SeverityLevel",
    "RefractoryType",
    "REFRACTORY_PROPERTIES_DB",
    # Version info
    "__version__",
    "__author__",
    "__agent__",
]


def get_calculator_info() -> dict:
    """
    Get information about the GL-007 calculators module.

    Returns:
        Dictionary with module information including version,
        available calculators, and compliance standards.
    """
    return {
        "agent_id": "GL-007",
        "agent_name": "FURNACEPULSE FurnacePerformanceMonitor",
        "module_version": __version__,
        "calculators": [
            {
                "name": "ThermalEfficiencyCalculator",
                "description": "ASME PTC 4.2 compliant furnace thermal efficiency",
                "standards": ["ASME PTC 4.2", "ISO 13579-1", "EN 746-2"]
            },
            {
                "name": "PerformanceKPICalculator",
                "description": "Furnace performance KPIs (SFC, Heat Rate, OEE)",
                "standards": ["ISO 50006", "ISO 50001", "EN 16231", "ISO 22400"]
            },
            {
                "name": "MaintenancePredictor",
                "description": "Predictive maintenance for furnace components",
                "standards": ["ISO 17359", "ISO 13379-1", "ASTM C155"]
            }
        ],
        "zero_hallucination_guarantee": True,
        "provenance_tracking": "SHA-256",
        "arithmetic_precision": "Python Decimal"
    }
