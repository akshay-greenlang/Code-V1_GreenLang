# -*- coding: utf-8 -*-
"""
Configuration module for FurnacePerformanceMonitor agent (GL-007).

This module defines the configuration models and settings for the
FurnacePerformanceMonitor agent, including furnace specifications,
operational constraints, optimization parameters, and integration settings.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from pathlib import Path
from greenlang.determinism import DeterministicClock


class FurnaceSpecification(BaseModel):
    """Furnace technical specifications."""

    furnace_id: str = Field(..., min_length=1, max_length=50, description="Unique furnace identifier")
    plant_id: str = Field(..., min_length=1, max_length=50, description="Plant identifier")
    furnace_type: str = Field(..., description="Furnace type (process_heater, heat_treat, reformer, calciner, melter, kiln)")

    # Capacity specifications
    design_capacity_mw: float = Field(..., ge=0.1, le=500, description="Design thermal capacity in MW")
    design_temperature_c: float = Field(..., ge=100, le=2000, description="Design temperature in Celsius")
    max_temperature_c: float = Field(..., ge=100, le=2000, description="Maximum operating temperature")
    min_temperature_c: float = Field(..., ge=50, le=1000, description="Minimum operating temperature")

    # Fuel specifications
    primary_fuel_type: str = Field(..., description="Primary fuel (natural_gas, propane, fuel_oil, coal, biomass, hydrogen)")
    fuel_heating_value_mj_kg: float = Field(..., gt=0, le=100, description="Fuel HHV in MJ/kg")

    # Design efficiency
    design_efficiency_percent: float = Field(..., ge=50, le=100, description="Design thermal efficiency")

    # Refractory condition
    refractory_condition: str = Field("good", description="Refractory condition (excellent, good, fair, poor)")

    # Age and maintenance
    commissioning_date: datetime = Field(..., description="Commissioning date")
    operating_hours: int = Field(..., ge=0, description="Total operating hours")


class OperationalConstraints(BaseModel):
    """Operational constraints for furnace optimization."""

    # Temperature constraints
    max_operating_temp_c: float = Field(..., gt=100, description="Maximum operating temperature")
    min_operating_temp_c: float = Field(..., ge=0, description="Minimum operating temperature")

    # Load constraints
    min_load_percent: float = Field(30.0, ge=0, le=100, description="Minimum operating load")
    max_load_percent: float = Field(100.0, ge=0, le=150, description="Maximum operating load")

    # Combustion constraints
    min_excess_air_percent: float = Field(5.0, ge=0, le=100, description="Minimum excess air")
    max_excess_air_percent: float = Field(30.0, ge=0, le=100, description="Maximum excess air")
    max_co_ppm: float = Field(100.0, ge=0, le=1000, description="Maximum CO in flue gas")
    max_nox_ppm: float = Field(150.0, ge=0, le=500, description="Maximum NOx in flue gas")

    # Safety limits
    max_pressure_mbar: float = Field(10.0, description="Maximum furnace pressure")
    min_pressure_mbar: float = Field(-10.0, description="Minimum furnace pressure")


class MonitoringConfiguration(BaseModel):
    """Monitoring configuration settings."""

    monitoring_interval_seconds: int = Field(60, ge=1, le=3600, description="Monitoring interval")
    optimization_frequency_minutes: int = Field(15, ge=1, le=1440, description="Optimization frequency")

    # Data quality thresholds
    min_data_availability_percent: float = Field(90.0, ge=0, le=100, description="Minimum data availability")

    # Alert thresholds
    efficiency_degradation_threshold_percent: float = Field(3.0, description="Efficiency drop threshold for alert")
    temperature_deviation_threshold_c: float = Field(50.0, description="Temperature deviation threshold")


class OptimizationParameters(BaseModel):
    """Optimization parameter settings."""

    optimization_objective: str = Field(
        "maximize_efficiency",
        description="Optimization objective (maximize_efficiency, minimize_cost, minimize_emissions, balanced)"
    )

    time_horizon_minutes: int = Field(60, ge=15, le=1440, description="Optimization time horizon")
    solution_tolerance: float = Field(0.01, ge=0.001, le=0.1, description="Solution tolerance")
    max_iterations: int = Field(100, ge=10, le=1000, description="Maximum iterations")


class FurnaceMonitorConfig(BaseModel):
    """Main configuration for FurnacePerformanceMonitor agent."""

    agent_name: str = Field("FurnacePerformanceMonitor", description="Agent name")
    agent_id: str = Field("GL-007", description="Agent ID")
    version: str = Field("1.0.0", description="Agent version")

    # Core settings
    calculation_timeout_seconds: int = Field(120, ge=10, le=600, description="Calculation timeout")
    cache_ttl_seconds: float = Field(60.0, ge=10, le=3600, description="Cache TTL")
    enable_monitoring: bool = Field(True, description="Enable metrics monitoring")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retries on error")

    # Operational settings
    furnace_spec: Optional[FurnaceSpecification] = None
    operational_constraints: OperationalConstraints = Field(default_factory=OperationalConstraints)
    monitoring_config: MonitoringConfiguration = Field(default_factory=MonitoringConfiguration)
    optimization_params: OptimizationParameters = Field(default_factory=OptimizationParameters)


class FurnaceConfiguration(BaseModel):
    """Furnace configuration wrapper."""

    furnace_id: str
    plant_id: str
    furnace_type: str
    design_capacity_mw: float
    design_temperature_c: float
