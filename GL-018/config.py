# -*- coding: utf-8 -*-
"""
GL-018 FLUEFLOW Configuration Models.

This module provides comprehensive Pydantic configuration models for the
Flue Gas Analyzer Agent, including burner configurations, fuel specifications,
emissions limits, SCADA integration, and agent settings.

All models include validators to ensure compliance with EPA/EU emissions
regulations and industry best practices.

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class BurnerType(str, Enum):
    """Burner type classification."""

    NATURAL_DRAFT = "natural_draft"
    FORCED_DRAFT = "forced_draft"
    INDUCED_DRAFT = "induced_draft"
    BALANCED_DRAFT = "balanced_draft"
    PREMIX = "premix"
    LOW_NOX = "low_nox"
    ULTRA_LOW_NOX = "ultra_low_nox"


class FuelType(str, Enum):
    """Fuel type classification."""

    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    DIESEL = "diesel"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_4 = "fuel_oil_4"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL = "coal"
    BIOMASS = "biomass"
    DUAL_FUEL = "dual_fuel"


class EmissionsStandard(str, Enum):
    """Emissions standard classification."""

    EPA_NSPS = "epa_nsps"  # EPA New Source Performance Standards
    EPA_MACT = "epa_mact"  # EPA Maximum Achievable Control Technology
    EU_IED = "eu_ied"  # EU Industrial Emissions Directive
    CARB = "carb"  # California Air Resources Board
    LOCAL = "local"  # Local regulations


class AnalyzerType(str, Enum):
    """Flue gas analyzer type."""

    EXTRACTIVE = "extractive"
    IN_SITU = "in_situ"
    PORTABLE = "portable"
    CEMS = "cems"  # Continuous Emissions Monitoring System
    PEMS = "pems"  # Predictive Emissions Monitoring System


class ControlStrategy(str, Enum):
    """Combustion control strategy."""

    O2_TRIM = "o2_trim"
    PARALLEL_POSITIONING = "parallel_positioning"
    CROSS_LIMITING = "cross_limiting"
    FEEDFORWARD = "feedforward"
    ADVANCED_CONTROL = "advanced_control"


# ============================================================================
# FUEL SPECIFICATION
# ============================================================================


class FuelSpecification(BaseModel):
    """
    Fuel specification and properties.

    Defines fuel characteristics that affect combustion calculations.
    """

    fuel_type: FuelType = Field(..., description="Fuel type classification")
    fuel_name: str = Field(..., description="Fuel name or identifier")

    # Heating value
    higher_heating_value_btu_scf: Optional[float] = Field(
        None, gt=0, description="HHV in Btu/SCF (for gaseous fuels)"
    )
    higher_heating_value_btu_gal: Optional[float] = Field(
        None, gt=0, description="HHV in Btu/gallon (for liquid fuels)"
    )
    higher_heating_value_btu_lb: Optional[float] = Field(
        None, gt=0, description="HHV in Btu/lb (for solid fuels)"
    )
    lower_heating_value_btu_scf: Optional[float] = Field(
        None, gt=0, description="LHV in Btu/SCF (for gaseous fuels)"
    )

    # Fuel composition (for combustion calculations)
    carbon_pct: float = Field(
        default=0.0, ge=0, le=100, description="Carbon content by mass %"
    )
    hydrogen_pct: float = Field(
        default=0.0, ge=0, le=100, description="Hydrogen content by mass %"
    )
    sulfur_pct: float = Field(
        default=0.0, ge=0, le=100, description="Sulfur content by mass %"
    )
    nitrogen_pct: float = Field(
        default=0.0, ge=0, le=100, description="Nitrogen content by mass %"
    )
    oxygen_pct: float = Field(
        default=0.0, ge=0, le=100, description="Oxygen content by mass %"
    )
    moisture_pct: float = Field(
        default=0.0, ge=0, le=100, description="Moisture content by mass %"
    )
    ash_pct: float = Field(
        default=0.0, ge=0, le=100, description="Ash content by mass %"
    )

    # Physical properties
    specific_gravity: Optional[float] = Field(
        None, gt=0, description="Specific gravity (relative to air or water)"
    )
    viscosity_ssu: Optional[float] = Field(
        None, gt=0, description="Viscosity in SSU (for liquid fuels)"
    )
    pour_point_f: Optional[float] = Field(
        None, description="Pour point in °F (for liquid fuels)"
    )

    # Stoichiometric properties
    stoichiometric_air_fuel_ratio: float = Field(
        default=15.0, gt=0, description="Stoichiometric A/F ratio (mass basis)"
    )
    theoretical_co2_max_pct: float = Field(
        default=12.0, gt=0, le=20, description="Maximum theoretical CO2 % (dry basis)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "fuel_type": "natural_gas",
                "fuel_name": "Pipeline Natural Gas",
                "higher_heating_value_btu_scf": 1020,
                "lower_heating_value_btu_scf": 920,
                "carbon_pct": 74.0,
                "hydrogen_pct": 24.0,
                "sulfur_pct": 0.01,
                "nitrogen_pct": 1.5,
                "oxygen_pct": 0.0,
                "specific_gravity": 0.60,
                "stoichiometric_air_fuel_ratio": 17.2,
                "theoretical_co2_max_pct": 11.7,
            }
        }


# ============================================================================
# EMISSIONS LIMITS
# ============================================================================


class EmissionsLimits(BaseModel):
    """
    Emissions limits and compliance thresholds.

    Defines regulatory limits for air pollutant emissions.
    """

    emissions_standard: EmissionsStandard = Field(
        ..., description="Applicable emissions standard"
    )

    # NOx limits
    nox_limit_ppm: float = Field(
        default=30.0, gt=0, description="NOx limit in ppm (corrected to reference O2)"
    )
    nox_limit_lb_mmbtu: Optional[float] = Field(
        None, gt=0, description="NOx limit in lb/MMBtu"
    )

    # CO limits
    co_limit_ppm: float = Field(
        default=400.0, gt=0, description="CO limit in ppm (corrected to reference O2)"
    )
    co_limit_lb_mmbtu: Optional[float] = Field(
        None, gt=0, description="CO limit in lb/MMBtu"
    )

    # SO2 limits
    so2_limit_ppm: float = Field(
        default=50.0, gt=0, description="SO2 limit in ppm (corrected to reference O2)"
    )
    so2_limit_lb_mmbtu: Optional[float] = Field(
        None, gt=0, description="SO2 limit in lb/MMBtu"
    )

    # Particulate matter limits
    pm_limit_mg_m3: float = Field(
        default=20.0, gt=0, description="PM limit in mg/m³ (corrected to reference O2)"
    )
    pm_limit_lb_mmbtu: Optional[float] = Field(
        None, gt=0, description="PM limit in lb/MMBtu"
    )

    # VOC limits (if applicable)
    voc_limit_ppm: Optional[float] = Field(
        None, gt=0, description="VOC limit in ppm"
    )

    # Reference O2 for corrections
    reference_o2_pct: float = Field(
        default=3.0, ge=0, le=20.95, description="Reference O2 for emissions corrections"
    )

    # Opacity limit
    opacity_limit_pct: float = Field(
        default=20.0, ge=0, le=100, description="Opacity limit in %"
    )

    # Compliance averaging period
    averaging_period_hours: float = Field(
        default=1.0, gt=0, description="Emissions averaging period in hours"
    )

    # Exceedance thresholds
    max_exceedances_per_24h: int = Field(
        default=0, ge=0, description="Maximum allowed exceedances per 24 hours"
    )
    max_exceedance_duration_minutes: float = Field(
        default=0.0, ge=0, description="Maximum allowed exceedance duration in minutes"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "emissions_standard": "epa_nsps",
                "nox_limit_ppm": 30.0,
                "nox_limit_lb_mmbtu": 0.036,
                "co_limit_ppm": 400.0,
                "so2_limit_ppm": 50.0,
                "pm_limit_mg_m3": 20.0,
                "reference_o2_pct": 3.0,
                "opacity_limit_pct": 20.0,
                "averaging_period_hours": 1.0,
            }
        }


# ============================================================================
# BURNER CONFIGURATION
# ============================================================================


class BurnerConfiguration(BaseModel):
    """
    Burner configuration and operating parameters.

    Defines the burner characteristics that influence combustion
    analysis and optimization strategies.
    """

    burner_id: str = Field(..., description="Unique burner identifier")
    burner_type: BurnerType = Field(..., description="Burner type classification")
    burner_manufacturer: Optional[str] = Field(
        None, description="Burner manufacturer"
    )
    burner_model: Optional[str] = Field(None, description="Burner model number")

    # Capacity
    design_firing_rate_mmbtu_hr: float = Field(
        ..., gt=0, description="Design firing rate in MMBtu/hr"
    )
    minimum_firing_rate_mmbtu_hr: float = Field(
        ..., gt=0, description="Minimum firing rate in MMBtu/hr"
    )
    turndown_ratio: float = Field(
        default=4.0, gt=1, le=20, description="Burner turndown ratio"
    )

    # Fuel specification
    fuel_specification: FuelSpecification = Field(
        ..., description="Fuel specification"
    )

    # Emissions limits
    emissions_standard: EmissionsStandard = Field(
        ..., description="Applicable emissions standard"
    )
    emissions_limits: EmissionsLimits = Field(
        ..., description="Emissions limits"
    )

    # Control system
    control_strategy: ControlStrategy = Field(
        default=ControlStrategy.O2_TRIM, description="Combustion control strategy"
    )
    has_vfd_fan: bool = Field(
        default=True, description="Variable frequency drive on fan"
    )
    has_modulating_fuel_valve: bool = Field(
        default=True, description="Modulating fuel valve installed"
    )

    # Monitoring equipment
    analyzer_type: AnalyzerType = Field(
        default=AnalyzerType.CEMS, description="Flue gas analyzer type"
    )
    has_o2_analyzer: bool = Field(default=True, description="O2 analyzer installed")
    has_co_analyzer: bool = Field(default=True, description="CO analyzer installed")
    has_nox_analyzer: bool = Field(default=True, description="NOx analyzer installed")
    has_co2_analyzer: bool = Field(default=True, description="CO2 analyzer installed")
    has_opacity_monitor: bool = Field(
        default=False, description="Opacity monitor installed"
    )

    # Stack parameters
    stack_height_ft: Optional[float] = Field(
        None, gt=0, description="Stack height in feet"
    )
    stack_diameter_inches: Optional[float] = Field(
        None, gt=0, description="Stack diameter in inches"
    )

    # Operating limits
    max_stack_temperature_f: float = Field(
        default=650, gt=0, description="Maximum allowable stack temperature in °F"
    )
    min_o2_pct: float = Field(
        default=2.0, ge=0, le=21, description="Minimum O2 setpoint %"
    )
    max_o2_pct: float = Field(
        default=8.0, ge=0, le=21, description="Maximum O2 setpoint %"
    )
    target_o2_pct: float = Field(
        default=3.0, ge=0, le=21, description="Target O2 setpoint %"
    )

    # Efficiency targets
    design_combustion_efficiency_pct: float = Field(
        default=83.0, ge=70, le=95, description="Design combustion efficiency %"
    )
    minimum_acceptable_efficiency_pct: float = Field(
        default=78.0, ge=60, le=90, description="Minimum acceptable efficiency %"
    )

    # Metadata
    location: Optional[str] = Field(None, description="Burner physical location")
    commissioning_date: Optional[datetime] = Field(
        None, description="Burner commissioning date"
    )
    last_tuning_date: Optional[datetime] = Field(
        None, description="Last combustion tuning date"
    )
    next_tuning_due_date: Optional[datetime] = Field(
        None, description="Next tuning due date"
    )

    @field_validator("turndown_ratio")
    @classmethod
    def validate_turndown_ratio(cls, v: float) -> float:
        """Validate turndown ratio is reasonable."""
        if v > 10:
            logger.warning(
                f"High turndown ratio ({v}:1) - ensure burner can operate stably"
            )
        return v

    @model_validator(mode="after")
    def validate_firing_rates(self) -> "BurnerConfiguration":
        """Validate firing rate consistency."""
        if self.minimum_firing_rate_mmbtu_hr > self.design_firing_rate_mmbtu_hr:
            raise ValueError(
                "Minimum firing rate cannot exceed design firing rate"
            )

        calculated_turndown = (
            self.design_firing_rate_mmbtu_hr / self.minimum_firing_rate_mmbtu_hr
        )
        if abs(calculated_turndown - self.turndown_ratio) > 0.5:
            logger.warning(
                f"Turndown ratio mismatch: specified {self.turndown_ratio}, "
                f"calculated {calculated_turndown:.1f}"
            )

        return self

    @model_validator(mode="after")
    def validate_o2_limits(self) -> "BurnerConfiguration":
        """Validate O2 limit consistency."""
        if self.min_o2_pct > self.target_o2_pct:
            raise ValueError("Minimum O2 cannot exceed target O2")

        if self.target_o2_pct > self.max_o2_pct:
            raise ValueError("Target O2 cannot exceed maximum O2")

        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "burner_id": "BURNER-001",
                "burner_type": "low_nox",
                "burner_manufacturer": "Cleaver-Brooks",
                "burner_model": "ClearFire-H",
                "design_firing_rate_mmbtu_hr": 60.0,
                "minimum_firing_rate_mmbtu_hr": 15.0,
                "turndown_ratio": 4.0,
                "emissions_standard": "epa_nsps",
                "control_strategy": "o2_trim",
                "analyzer_type": "cems",
                "target_o2_pct": 3.0,
                "design_combustion_efficiency_pct": 83.0,
            }
        }


# ============================================================================
# SCADA INTEGRATION
# ============================================================================


class SCADAIntegration(BaseModel):
    """
    SCADA system integration configuration.

    Defines connection parameters and tag mappings for SCADA integration.
    """

    enabled: bool = Field(default=True, description="SCADA integration enabled")
    scada_system: str = Field(default="", description="SCADA system type/vendor")
    connection_string: str = Field(default="", description="SCADA connection string")
    polling_interval_seconds: int = Field(
        default=60, ge=1, le=3600, description="Data polling interval in seconds"
    )

    # Tag mappings
    o2_tag: str = Field(default="", description="O2 analyzer tag")
    co2_tag: str = Field(default="", description="CO2 analyzer tag")
    co_tag: str = Field(default="", description="CO analyzer tag")
    nox_tag: str = Field(default="", description="NOx analyzer tag")
    so2_tag: str = Field(default="", description="SO2 analyzer tag")
    stack_temp_tag: str = Field(default="", description="Stack temperature tag")
    fuel_flow_tag: str = Field(default="", description="Fuel flow tag")
    air_flow_tag: str = Field(default="", description="Air flow tag")
    fd_fan_speed_tag: str = Field(default="", description="FD fan speed tag")
    id_fan_speed_tag: str = Field(default="", description="ID fan speed tag")
    firing_rate_tag: str = Field(default="", description="Firing rate tag")
    steam_flow_tag: str = Field(default="", description="Steam flow tag")

    # Control output tags (for automatic optimization)
    fd_fan_setpoint_tag: str = Field(
        default="", description="FD fan speed setpoint tag"
    )
    fuel_valve_setpoint_tag: str = Field(
        default="", description="Fuel valve position setpoint tag"
    )
    damper_position_setpoint_tag: str = Field(
        default="", description="Damper position setpoint tag"
    )

    # Data quality
    enable_data_validation: bool = Field(
        default=True, description="Enable SCADA data quality validation"
    )
    max_data_age_seconds: int = Field(
        default=300, ge=10, description="Maximum acceptable data age in seconds"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "enabled": True,
                "scada_system": "Wonderware",
                "polling_interval_seconds": 60,
                "o2_tag": "FG.O2_PCT",
                "co_tag": "FG.CO_PPM",
                "nox_tag": "FG.NOX_PPM",
                "stack_temp_tag": "FG.STACK_TEMP",
                "fuel_flow_tag": "BURNER.FUEL_FLOW",
                "air_flow_tag": "BURNER.AIR_FLOW",
            }
        }


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


class AgentConfiguration(BaseModel):
    """
    Main agent configuration.

    Top-level configuration for the Flue Gas Analyzer Agent.
    """

    agent_name: str = Field(
        default="GL-018 FLUEFLOW", description="Agent name"
    )
    version: str = Field(default="1.0.0", description="Agent version")
    environment: str = Field(
        default="production", description="Deployment environment"
    )

    # Burner configurations
    burners: List[BurnerConfiguration] = Field(
        ..., min_length=1, description="List of burner configurations"
    )

    # SCADA integration
    scada_integration: SCADAIntegration = Field(
        ..., description="SCADA integration configuration"
    )

    # Operational settings
    analysis_interval_seconds: int = Field(
        default=60, ge=10, le=3600, description="Analysis execution interval in seconds"
    )
    auto_optimization_enabled: bool = Field(
        default=False, description="Enable automatic optimization adjustments"
    )
    optimization_deadband_pct: float = Field(
        default=2.0, ge=0, le=10, description="Optimization deadband %"
    )

    # Alerting
    enable_email_alerts: bool = Field(
        default=True, description="Enable email alerts"
    )
    enable_sms_alerts: bool = Field(
        default=False, description="Enable SMS alerts"
    )
    alert_recipients: List[str] = Field(
        default_factory=list, description="Alert recipient email addresses"
    )

    # Reporting
    enable_hourly_reports: bool = Field(
        default=True, description="Generate hourly reports"
    )
    enable_daily_reports: bool = Field(
        default=True, description="Generate daily reports"
    )
    enable_monthly_reports: bool = Field(
        default=True, description="Generate monthly reports"
    )
    report_recipients: List[str] = Field(
        default_factory=list, description="Report recipient email addresses"
    )

    # Data retention
    historical_data_retention_days: int = Field(
        default=90, ge=7, le=730, description="Historical data retention in days"
    )

    # Advanced features
    enable_predictive_maintenance: bool = Field(
        default=True, description="Enable predictive maintenance alerts"
    )
    enable_efficiency_trending: bool = Field(
        default=True, description="Enable efficiency trending analysis"
    )
    enable_emissions_forecasting: bool = Field(
        default=False, description="Enable emissions forecasting"
    )

    def get_burner(self, burner_id: str) -> Optional[BurnerConfiguration]:
        """
        Get burner configuration by ID.

        Args:
            burner_id: Burner identifier

        Returns:
            BurnerConfiguration or None if not found
        """
        for burner in self.burners:
            if burner.burner_id == burner_id:
                return burner
        return None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "agent_name": "GL-018 FLUEFLOW",
                "version": "1.0.0",
                "environment": "production",
                "analysis_interval_seconds": 60,
                "auto_optimization_enabled": False,
                "enable_email_alerts": True,
                "enable_hourly_reports": True,
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "BurnerType",
    "FuelType",
    "EmissionsStandard",
    "AnalyzerType",
    "ControlStrategy",
    "FuelSpecification",
    "EmissionsLimits",
    "BurnerConfiguration",
    "SCADAIntegration",
    "AgentConfiguration",
]
