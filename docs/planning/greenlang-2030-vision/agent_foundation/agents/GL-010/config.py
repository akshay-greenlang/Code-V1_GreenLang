# -*- coding: utf-8 -*-
"""
Configuration module for GL-010 EMISSIONWATCH EmissionsComplianceAgent.

This module defines Pydantic V2 configuration classes for the EMISSIONWATCH agent,
including emissions limits, regulatory parameters, CEMS settings, alerting
configurations, and integration options.

SECURITY:
- Zero hardcoded credentials policy
- All secrets loaded from environment variables
- Validation enforced at startup

Standards Compliance:
- EPA 40 CFR Parts 60, 75 - Continuous Emissions Monitoring
- EU Industrial Emissions Directive 2010/75/EU
- China MEE Emission Standards (GB 13223-2011)
- ASME PTC 19.10 - Flue Gas Analysis

Author: GreenLang Foundation
Version: 1.0.0
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseModel, Field, validator as field_validator
    model_validator = None


# ============================================================================
# ENUMERATIONS
# ============================================================================

class Jurisdiction(Enum):
    """Regulatory jurisdictions supported."""
    EPA = "EPA"                    # US EPA regulations
    EU_IED = "EU_IED"              # EU Industrial Emissions Directive
    CHINA_MEE = "CHINA_MEE"        # China Ministry of Ecology and Environment
    CALIFORNIA_CARB = "CARB"       # California Air Resources Board
    TEXAS_TCEQ = "TCEQ"            # Texas Commission on Environmental Quality


class PollutantType(Enum):
    """Pollutant types monitored."""
    NOX = "nox"
    SOX = "sox"
    CO2 = "co2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    MERCURY = "mercury"
    OPACITY = "opacity"


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReportFormat(Enum):
    """Regulatory report formats."""
    EPA_ECMPS = "EPA_ECMPS"        # EPA Electronic Data Reporting
    EU_ELED = "EU_ELED"            # EU E-PRTR reporting
    CHINA_MEE = "CHINA_MEE"        # China MEE format
    CSV = "CSV"                     # Generic CSV
    JSON = "JSON"                   # JSON format
    PDF = "PDF"                     # PDF report


class FuelType(Enum):
    """Fuel types for emission calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLETS = "biomass_pellets"
    HYDROGEN = "hydrogen"
    PROPANE = "propane"


class ControlDeviceType(Enum):
    """Emission control device types."""
    SCR = "scr"                     # Selective Catalytic Reduction
    SNCR = "sncr"                   # Selective Non-Catalytic Reduction
    LOW_NOX_BURNER = "low_nox_burner"
    FGD_WET = "fgd_wet"            # Wet Flue Gas Desulfurization
    FGD_DRY = "fgd_dry"            # Dry FGD
    ESP = "esp"                     # Electrostatic Precipitator
    BAGHOUSE = "baghouse"           # Fabric Filter
    WET_SCRUBBER = "wet_scrubber"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# NOX CONFIGURATION
# ============================================================================

class NOxConfig(BaseModel):
    """Configuration for NOx emissions monitoring."""

    # Emission limits
    limit_ppm: float = Field(
        default=50.0,
        ge=0.0,
        le=5000.0,
        description="NOx concentration limit in ppm"
    )

    limit_lb_mmbtu: float = Field(
        default=0.10,
        ge=0.0,
        le=5.0,
        description="NOx emission rate limit in lb/MMBtu"
    )

    limit_mg_nm3: float = Field(
        default=100.0,
        ge=0.0,
        le=1000.0,
        description="NOx concentration limit in mg/Nm3 (EU/China)"
    )

    # Reference O2 correction
    reference_o2_percent: float = Field(
        default=3.0,
        ge=0.0,
        le=21.0,
        description="Reference O2 percentage for correction"
    )

    # Averaging periods
    averaging_period_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Averaging period for compliance in minutes"
    )

    # Control device settings
    scr_efficiency_percent: float = Field(
        default=90.0,
        ge=0.0,
        le=99.9,
        description="SCR removal efficiency percentage"
    )

    sncr_efficiency_percent: float = Field(
        default=50.0,
        ge=0.0,
        le=80.0,
        description="SNCR removal efficiency percentage"
    )

    # Warning thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Percentage of limit for warning alerts"
    )


# ============================================================================
# SOX CONFIGURATION
# ============================================================================

class SOxConfig(BaseModel):
    """Configuration for SOx emissions monitoring."""

    # Emission limits
    limit_ppm: float = Field(
        default=100.0,
        ge=0.0,
        le=5000.0,
        description="SOx concentration limit in ppm"
    )

    limit_lb_mmbtu: float = Field(
        default=0.15,
        ge=0.0,
        le=5.0,
        description="SOx emission rate limit in lb/MMBtu"
    )

    limit_mg_nm3: float = Field(
        default=150.0,
        ge=0.0,
        le=2000.0,
        description="SOx concentration limit in mg/Nm3"
    )

    # Fuel sulfur limits
    max_fuel_sulfur_percent: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Maximum allowable fuel sulfur content"
    )

    # Control device settings
    fgd_efficiency_percent: float = Field(
        default=95.0,
        ge=0.0,
        le=99.9,
        description="FGD removal efficiency percentage"
    )

    # Averaging periods
    averaging_period_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Averaging period for compliance in minutes"
    )

    # Warning thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Percentage of limit for warning alerts"
    )


# ============================================================================
# CO2 CONFIGURATION
# ============================================================================

class CO2Config(BaseModel):
    """Configuration for CO2 emissions monitoring."""

    # Emission limits
    limit_tons_hr: float = Field(
        default=50.0,
        ge=0.0,
        le=1000.0,
        description="CO2 emission limit in tons per hour"
    )

    limit_tons_mwh: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="CO2 emission rate limit in tons/MWh"
    )

    limit_g_kwh: float = Field(
        default=350.0,
        ge=0.0,
        le=1000.0,
        description="CO2 emission rate limit in g/kWh"
    )

    # Carbon intensity targets
    carbon_intensity_target_kg_mwh: float = Field(
        default=400.0,
        ge=0.0,
        le=2000.0,
        description="Target carbon intensity in kg/MWh"
    )

    # Biogenic carbon
    include_biogenic: bool = Field(
        default=False,
        description="Include biogenic CO2 in totals"
    )

    # Carbon capture settings
    ccs_efficiency_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=99.9,
        description="Carbon capture efficiency percentage"
    )

    # Warning thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Percentage of limit for warning alerts"
    )


# ============================================================================
# PM CONFIGURATION
# ============================================================================

class PMConfig(BaseModel):
    """Configuration for particulate matter monitoring."""

    # Emission limits
    limit_mg_m3: float = Field(
        default=30.0,
        ge=0.0,
        le=500.0,
        description="PM concentration limit in mg/m3"
    )

    limit_lb_mmbtu: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="PM emission rate limit in lb/MMBtu"
    )

    # PM size fractions
    pm10_fraction: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Fraction of total PM that is PM10"
    )

    pm25_fraction: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Fraction of total PM that is PM2.5"
    )

    # Opacity limits
    opacity_limit_percent: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Stack opacity limit percentage"
    )

    # Control device settings
    baghouse_efficiency_percent: float = Field(
        default=99.5,
        ge=0.0,
        le=99.99,
        description="Baghouse removal efficiency percentage"
    )

    esp_efficiency_percent: float = Field(
        default=99.0,
        ge=0.0,
        le=99.99,
        description="ESP removal efficiency percentage"
    )

    # Warning thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Percentage of limit for warning alerts"
    )


# ============================================================================
# REGULATORY LIMITS CONFIGURATION
# ============================================================================

class RegulatoryLimitsConfig(BaseModel):
    """Configuration for regulatory emission limits by jurisdiction."""

    jurisdiction: str = Field(
        default="EPA",
        description="Regulatory jurisdiction"
    )

    # EPA limits (40 CFR Part 60 Subpart Da)
    epa_nox_lb_mmbtu: float = Field(
        default=0.10,
        description="EPA NSPS NOx limit"
    )

    epa_sox_lb_mmbtu: float = Field(
        default=0.15,
        description="EPA NSPS SOx limit"
    )

    epa_pm_lb_mmbtu: float = Field(
        default=0.03,
        description="EPA NSPS PM limit"
    )

    epa_opacity_percent: float = Field(
        default=20.0,
        description="EPA opacity limit"
    )

    # EU IED limits (mg/Nm3 at 3% O2)
    eu_nox_mg_nm3: float = Field(
        default=100.0,
        description="EU IED BAT-AEL NOx limit"
    )

    eu_sox_mg_nm3: float = Field(
        default=150.0,
        description="EU IED BAT-AEL SOx limit"
    )

    eu_pm_mg_nm3: float = Field(
        default=5.0,
        description="EU IED BAT-AEL PM limit"
    )

    eu_co_mg_nm3: float = Field(
        default=100.0,
        description="EU IED CO limit"
    )

    # China MEE limits (GB 13223-2011 ultra-low)
    china_nox_mg_nm3: float = Field(
        default=50.0,
        description="China ultra-low NOx limit"
    )

    china_sox_mg_nm3: float = Field(
        default=35.0,
        description="China ultra-low SOx limit"
    )

    china_pm_mg_nm3: float = Field(
        default=10.0,
        description="China ultra-low PM limit"
    )

    china_mercury_ug_nm3: float = Field(
        default=30.0,
        description="China mercury limit"
    )


# ============================================================================
# CEMS CONFIGURATION
# ============================================================================

class CEMSConfig(BaseModel):
    """Configuration for Continuous Emissions Monitoring Systems."""

    # Data collection
    sampling_frequency_seconds: int = Field(
        default=15,
        ge=1,
        le=60,
        description="CEMS data sampling frequency in seconds"
    )

    averaging_period_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Data averaging period in minutes"
    )

    # Data quality
    min_data_availability_percent: float = Field(
        default=90.0,
        ge=75.0,
        le=100.0,
        description="Minimum required data availability"
    )

    substitute_data_enabled: bool = Field(
        default=True,
        description="Enable substitute data procedures"
    )

    # Calibration requirements
    daily_calibration_required: bool = Field(
        default=True,
        description="Require daily calibration checks"
    )

    calibration_drift_limit_percent: float = Field(
        default=2.5,
        ge=0.5,
        le=10.0,
        description="Calibration drift limit percentage"
    )

    rata_frequency_days: int = Field(
        default=365,
        ge=90,
        le=730,
        description="RATA (Relative Accuracy Test Audit) frequency"
    )

    # Analyzer ranges
    nox_analyzer_range_ppm: float = Field(
        default=500.0,
        ge=100.0,
        le=5000.0,
        description="NOx analyzer full scale range"
    )

    sox_analyzer_range_ppm: float = Field(
        default=1000.0,
        ge=100.0,
        le=5000.0,
        description="SOx analyzer full scale range"
    )

    co2_analyzer_range_percent: float = Field(
        default=20.0,
        ge=10.0,
        le=30.0,
        description="CO2 analyzer full scale range"
    )

    o2_analyzer_range_percent: float = Field(
        default=25.0,
        ge=21.0,
        le=30.0,
        description="O2 analyzer full scale range"
    )

    # Flow monitoring
    flow_rate_range_scfm: float = Field(
        default=100000.0,
        ge=1000.0,
        le=10000000.0,
        description="Stack flow rate full scale range"
    )


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================

class AlertConfig(BaseModel):
    """Configuration for emissions alerts and notifications."""

    # Alert enabling
    enable_email_alerts: bool = Field(
        default=True,
        description="Enable email notifications"
    )

    enable_sms_alerts: bool = Field(
        default=False,
        description="Enable SMS notifications"
    )

    enable_webhook_alerts: bool = Field(
        default=True,
        description="Enable webhook notifications"
    )

    enable_dashboard_alerts: bool = Field(
        default=True,
        description="Enable dashboard notifications"
    )

    # Alert thresholds
    warning_threshold_percent: float = Field(
        default=80.0,
        ge=50.0,
        le=99.0,
        description="Percentage of limit for warning"
    )

    critical_threshold_percent: float = Field(
        default=100.0,
        ge=90.0,
        le=150.0,
        description="Percentage of limit for critical alert"
    )

    # Alert frequency
    alert_cooldown_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Minimum time between repeated alerts"
    )

    escalation_delay_minutes: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Time before alert escalation"
    )

    # Alert recipients (loaded from environment)
    email_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for alerts"
    )

    sms_recipients: List[str] = Field(
        default_factory=list,
        description="Phone numbers for SMS alerts"
    )

    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for notifications"
    )

    @field_validator('webhook_url')
    @classmethod
    def validate_no_credentials_in_url(cls, v):
        """Validate URL does not contain embedded credentials."""
        if v is not None:
            parsed = urlparse(v)
            if parsed.username or parsed.password:
                raise ValueError(
                    "SECURITY VIOLATION: URL contains embedded credentials. "
                    "Use environment variables instead."
                )
        return v


# ============================================================================
# REPORTING CONFIGURATION
# ============================================================================

class ReportingConfig(BaseModel):
    """Configuration for regulatory reporting."""

    # Report formats
    default_format: ReportFormat = Field(
        default=ReportFormat.EPA_ECMPS,
        description="Default report format"
    )

    enabled_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.EPA_ECMPS, ReportFormat.CSV],
        description="Enabled report formats"
    )

    # Reporting periods
    quarterly_reports_enabled: bool = Field(
        default=True,
        description="Enable quarterly compliance reports"
    )

    annual_reports_enabled: bool = Field(
        default=True,
        description="Enable annual emissions reports"
    )

    # Report content
    include_hourly_data: bool = Field(
        default=True,
        description="Include hourly emissions data"
    )

    include_summary_statistics: bool = Field(
        default=True,
        description="Include summary statistics"
    )

    include_excess_emissions: bool = Field(
        default=True,
        description="Include excess emissions summary"
    )

    include_monitoring_downtime: bool = Field(
        default=True,
        description="Include monitoring downtime summary"
    )

    # Certification
    require_certification: bool = Field(
        default=True,
        description="Require designated representative certification"
    )

    designated_representative: Optional[str] = Field(
        default=None,
        description="Name of designated representative"
    )

    # Submission settings
    auto_submit_enabled: bool = Field(
        default=False,
        description="Enable automatic report submission"
    )

    submission_deadline_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Days after period end for submission"
    )


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

class IntegrationConfig(BaseModel):
    """Configuration for external system integrations."""

    # OPC UA integration
    opcua_enabled: bool = Field(
        default=False,
        description="Enable OPC UA integration"
    )

    opcua_endpoint: Optional[str] = Field(
        default=None,
        description="OPC UA server endpoint URL"
    )

    opcua_namespace: str = Field(
        default="ns=2",
        description="OPC UA namespace"
    )

    # Historian integration
    historian_enabled: bool = Field(
        default=False,
        description="Enable process historian integration"
    )

    historian_type: str = Field(
        default="osisoft_pi",
        description="Historian type (osisoft_pi, wonderware, honeywell)"
    )

    historian_endpoint: Optional[str] = Field(
        default=None,
        description="Historian API endpoint"
    )

    # DCS integration
    dcs_enabled: bool = Field(
        default=False,
        description="Enable DCS integration"
    )

    dcs_protocol: str = Field(
        default="modbus_tcp",
        description="DCS protocol (modbus_tcp, modbus_rtu, profinet)"
    )

    # EPA ECMPS integration
    ecmps_enabled: bool = Field(
        default=False,
        description="Enable EPA ECMPS integration"
    )

    ecmps_api_endpoint: str = Field(
        default="https://ecmps.epa.gov/api/v1",
        description="EPA ECMPS API endpoint"
    )

    # Webhook notifications
    webhook_enabled: bool = Field(
        default=False,
        description="Enable webhook notifications"
    )

    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for notifications"
    )

    @field_validator('opcua_endpoint', 'historian_endpoint', 'webhook_url')
    @classmethod
    def validate_no_credentials(cls, v):
        """Validate URLs do not contain embedded credentials."""
        if v is not None:
            parsed = urlparse(v)
            if parsed.username or parsed.password:
                raise ValueError(
                    "SECURITY VIOLATION: URL contains embedded credentials."
                )
        return v


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

class CacheConfig(BaseModel):
    """Configuration for caching settings."""

    # Cache size
    max_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum cache entries"
    )

    # TTL settings
    ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cache entry time-to-live in seconds"
    )

    # Emission factor cache
    emission_factor_cache_size: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Emission factor cache size"
    )

    emission_factor_ttl_seconds: int = Field(
        default=3600,
        ge=300,
        le=86400,
        description="Emission factor cache TTL"
    )


# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

class MonitoringConfig(BaseModel):
    """Configuration for real-time monitoring."""

    # Monitoring interval
    monitoring_interval_seconds: int = Field(
        default=15,
        ge=1,
        le=300,
        description="Data collection interval in seconds"
    )

    # Trend analysis
    trend_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours of data for trend analysis"
    )

    # Data retention
    realtime_buffer_size: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of real-time data points to buffer"
    )

    historical_retention_days: int = Field(
        default=1095,
        ge=365,
        le=3650,
        description="Historical data retention period (3+ years for Part 75)"
    )

    # Prometheus metrics
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint"
    )

    prometheus_port: int = Field(
        default=9010,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class EmissionsComplianceConfig(BaseModel):
    """
    Main configuration for GL-010 EMISSIONWATCH EmissionsComplianceAgent.

    This class contains all configuration settings for the agent, including
    emissions limits, regulatory parameters, CEMS settings, alerting options,
    and integration configurations.

    SECURITY:
    - Zero hardcoded credentials policy enforced
    - All secrets must be in environment variables
    - URL validation prevents credential leakage

    Example:
        >>> config = EmissionsComplianceConfig(
        ...     agent_id="GL-010",
        ...     jurisdiction="EPA",
        ...     nox_limit_ppm=50.0
        ... )
        >>> orchestrator = EmissionsComplianceOrchestrator(config)
    """

    # ========================================================================
    # AGENT IDENTIFICATION
    # ========================================================================

    agent_id: str = Field(
        default="GL-010",
        description="Unique agent identifier"
    )

    codename: str = Field(
        default="EMISSIONWATCH",
        description="Agent codename"
    )

    full_name: str = Field(
        default="EmissionsComplianceAgent",
        description="Full agent name"
    )

    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # ========================================================================
    # DETERMINISTIC SETTINGS
    # ========================================================================

    deterministic: bool = Field(
        default=True,
        description="Enable deterministic mode (required for compliance)"
    )

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=0.0,
        description="LLM temperature (must be 0.0 for determinism)"
    )

    llm_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    llm_model: str = Field(
        default="claude-3-haiku",
        description="LLM model for classification (not calculations)"
    )

    llm_max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum tokens for LLM responses"
    )

    # ========================================================================
    # REGULATORY SETTINGS
    # ========================================================================

    jurisdiction: str = Field(
        default="EPA",
        description="Primary regulatory jurisdiction (EPA, EU_IED, CHINA_MEE)"
    )

    facility_id: Optional[str] = Field(
        default=None,
        description="EPA ORIS code or facility identifier"
    )

    unit_id: Optional[str] = Field(
        default=None,
        description="Unit identifier"
    )

    # ========================================================================
    # EMISSION LIMITS
    # ========================================================================

    nox_limit_ppm: float = Field(
        default=50.0,
        ge=0.0,
        le=5000.0,
        description="NOx concentration limit in ppm"
    )

    sox_limit_ppm: float = Field(
        default=100.0,
        ge=0.0,
        le=5000.0,
        description="SOx concentration limit in ppm"
    )

    co2_limit_tons_hr: float = Field(
        default=50.0,
        ge=0.0,
        le=1000.0,
        description="CO2 emission limit in tons per hour"
    )

    pm_limit_mg_m3: float = Field(
        default=30.0,
        ge=0.0,
        le=500.0,
        description="PM concentration limit in mg/m3"
    )

    opacity_limit_percent: float = Field(
        default=20.0,
        ge=0.0,
        le=100.0,
        description="Stack opacity limit percentage"
    )

    # ========================================================================
    # CACHE SETTINGS
    # ========================================================================

    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cache time-to-live in seconds"
    )

    cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Maximum cache entries"
    )

    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================

    calculation_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for calculations in seconds"
    )

    max_concurrent_calculations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent calculations"
    )

    # ========================================================================
    # RETRY SETTINGS
    # ========================================================================

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    retry_initial_delay_ms: float = Field(
        default=100.0,
        ge=10.0,
        le=1000.0,
        description="Initial retry delay in milliseconds"
    )

    retry_max_delay_ms: float = Field(
        default=5000.0,
        ge=100.0,
        le=30000.0,
        description="Maximum retry delay in milliseconds"
    )

    enable_error_recovery: bool = Field(
        default=True,
        description="Enable automatic error recovery"
    )

    # ========================================================================
    # ALERT SETTINGS
    # ========================================================================

    enable_email_alerts: bool = Field(
        default=False,
        description="Enable email notifications"
    )

    enable_sms_alerts: bool = Field(
        default=False,
        description="Enable SMS notifications"
    )

    enable_webhook_alerts: bool = Field(
        default=False,
        description="Enable webhook notifications"
    )

    # ========================================================================
    # REPORTING SETTINGS
    # ========================================================================

    default_report_format: str = Field(
        default="EPA_ECMPS",
        description="Default regulatory report format"
    )

    # ========================================================================
    # MONITORING SETTINGS
    # ========================================================================

    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )

    monitoring_interval_seconds: int = Field(
        default=15,
        ge=1,
        le=300,
        description="Monitoring interval in seconds"
    )

    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================

    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashing"
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    zero_secrets: bool = Field(
        default=True,
        description="Enforce zero hardcoded secrets policy"
    )

    # ========================================================================
    # STORAGE PATHS
    # ========================================================================

    data_directory: Optional[Path] = Field(
        default=None,
        description="Data storage directory"
    )

    log_directory: Optional[Path] = Field(
        default=None,
        description="Log file directory"
    )

    cache_directory: Optional[Path] = Field(
        default=None,
        description="Cache storage directory"
    )

    # ========================================================================
    # SUB-CONFIGURATIONS
    # ========================================================================

    nox_config: NOxConfig = Field(
        default_factory=NOxConfig,
        description="NOx emissions configuration"
    )

    sox_config: SOxConfig = Field(
        default_factory=SOxConfig,
        description="SOx emissions configuration"
    )

    co2_config: CO2Config = Field(
        default_factory=CO2Config,
        description="CO2 emissions configuration"
    )

    pm_config: PMConfig = Field(
        default_factory=PMConfig,
        description="PM emissions configuration"
    )

    regulatory_limits: RegulatoryLimitsConfig = Field(
        default_factory=RegulatoryLimitsConfig,
        description="Regulatory limits configuration"
    )

    cems_config: CEMSConfig = Field(
        default_factory=CEMSConfig,
        description="CEMS configuration"
    )

    alert_config: AlertConfig = Field(
        default_factory=AlertConfig,
        description="Alert configuration"
    )

    reporting_config: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Reporting configuration"
    )

    integration_config: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )

    cache_config: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Cache configuration"
    )

    monitoring_config: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator('llm_temperature')
    @classmethod
    def validate_deterministic_temperature(cls, v):
        """Ensure temperature is 0.0 for deterministic operation."""
        if v != 0.0:
            raise ValueError(
                "COMPLIANCE VIOLATION: llm_temperature must be 0.0 for "
                "deterministic, zero-hallucination calculations"
            )
        return v

    @field_validator('zero_secrets')
    @classmethod
    def validate_zero_secrets(cls, v):
        """Ensure zero_secrets policy is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. "
                "No credentials allowed in configuration."
            )
        return v

    @field_validator('jurisdiction')
    @classmethod
    def validate_jurisdiction(cls, v):
        """Validate jurisdiction is supported."""
        valid_jurisdictions = ['EPA', 'EU_IED', 'CHINA_MEE', 'CARB', 'TCEQ']
        if v not in valid_jurisdictions:
            raise ValueError(f"Invalid jurisdiction: {v}. Must be one of {valid_jurisdictions}")
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Set default directories if not provided
        if self.data_directory is None:
            self.data_directory = Path("./gl010_data")
        if self.log_directory is None:
            self.log_directory = Path("./gl010_logs")
        if self.cache_directory is None:
            self.cache_directory = Path("./gl010_cache")

        # Create directories if they don't exist
        for directory in [self.data_directory, self.log_directory, self.cache_directory]:
            if directory:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass  # May fail in read-only environments

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def get_api_key(provider: str = "anthropic") -> Optional[str]:
        """
        Get API key from environment variable.

        SECURITY: API keys must be stored in environment variables.

        Args:
            provider: API provider name

        Returns:
            API key from environment, or None if not set
        """
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "epa_ecmps": "EPA_ECMPS_API_KEY",
        }

        env_var = env_var_map.get(provider.lower())
        if not env_var:
            raise ValueError(f"Unknown API provider: {provider}")

        return os.environ.get(env_var)

    @staticmethod
    def is_production() -> bool:
        """
        Check if running in production environment.

        Returns:
            True if GREENLANG_ENV is 'production' or 'prod'
        """
        env = os.environ.get("GREENLANG_ENV", "development").lower()
        return env in ["production", "prod"]

    @staticmethod
    def get_environment() -> str:
        """
        Get current environment name.

        Returns:
            Environment name (development, staging, production)
        """
        return os.environ.get("GREENLANG_ENV", "development").lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        str_strip_whitespace = True


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_config(
    env: Optional[str] = None,
    **overrides
) -> EmissionsComplianceConfig:
    """
    Create configuration based on environment.

    Args:
        env: Environment name (development, staging, production)
        **overrides: Configuration overrides

    Returns:
        Configured EmissionsComplianceConfig instance

    Example:
        >>> config = create_config("production", nox_limit_ppm=40.0)
    """
    if env is None:
        env = EmissionsComplianceConfig.get_environment()

    # Environment-specific defaults
    env_defaults = {
        "development": {
            "cache_ttl_seconds": 60,
            "max_retries": 1,
            "enable_monitoring": True,
            "enable_email_alerts": False,
            "enable_sms_alerts": False
        },
        "staging": {
            "cache_ttl_seconds": 300,
            "max_retries": 2,
            "enable_monitoring": True,
            "enable_email_alerts": True,
            "enable_sms_alerts": False
        },
        "production": {
            "cache_ttl_seconds": 600,
            "max_retries": 3,
            "enable_monitoring": True,
            "enable_email_alerts": True,
            "enable_sms_alerts": True
        }
    }

    # Apply environment defaults
    defaults = env_defaults.get(env, env_defaults["development"])
    defaults.update(overrides)

    return EmissionsComplianceConfig(**defaults)


def load_config_from_file(config_path: Union[str, Path]) -> EmissionsComplianceConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured EmissionsComplianceConfig instance
    """
    import json
    config_path = Path(config_path)

    if config_path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    elif config_path.suffix == '.json':
        with open(config_path) as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return EmissionsComplianceConfig(**config_data)
