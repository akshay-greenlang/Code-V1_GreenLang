# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Configuration Module

This module defines all configuration schemas for the EmissionsGuardian Agent,
including emissions monitoring parameters, CEMS integration settings, RATA QA/QC
configuration, carbon trading settings, LDAR program configuration, regulatory
reporting options, SHAP/LIME explainability settings, and provenance tracking.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults for industrial emissions monitoring applications.

Standards Compliance:
    - EPA 40 CFR Part 60/61/63 (NSPS/NESHAP)
    - EPA 40 CFR Part 75 (Acid Rain/CEMS)
    - EPA 40 CFR Part 98 (GHG Reporting)
    - EU ETS Monitoring and Reporting Regulation
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - ISO 14064: GHG Quantification and Reporting

Example:
    >>> from greenlang.agents.process_heat.gl_010_emissions_guardian.config import (
    ...     GL010Config,
    ...     EmissionsMonitoringConfig,
    ...     CEMSIntegrationConfig,
    ... )
    >>> config = GL010Config(
    ...     monitoring=EmissionsMonitoringConfig(
    ...         source_id="STACK-001",
    ...         permit_limits={"nox_lb_hr": 25.0, "co2_lb_hr": 50000.0}
    ...     ),
    ... )

Author: GreenLang Process Heat Team
Version: 2.0.0
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, validator, root_validator


# =============================================================================
# ENUMS - EMISSIONS AND COMPLIANCE CLASSIFICATIONS
# =============================================================================


class PollutantType(str, Enum):
    """
    Types of monitored pollutants.

    Classification based on regulatory category:
    - GHG: Greenhouse gases (CO2, CH4, N2O, etc.)
    - Criteria: EPA criteria pollutants (NOx, SO2, CO, PM, etc.)
    - HAP: Hazardous air pollutants
    """
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    NOX = "nox"
    SO2 = "so2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    HCL = "hcl"
    HG = "hg"
    O2 = "o2"
    OPACITY = "opacity"


class FuelType(str, Enum):
    """Fuel types per EPA Part 98 classifications."""
    NATURAL_GAS = "natural_gas"
    DISTILLATE_OIL = "distillate_oil"
    RESIDUAL_OIL = "residual_oil"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    COAL_LIGNITE = "coal_lignite"
    PETROLEUM_COKE = "petroleum_coke"
    WOOD_RESIDUALS = "wood_residuals"
    LANDFILL_GAS = "landfill_gas"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    OTHER = "other"


class MonitoringMethod(str, Enum):
    """Emissions monitoring methodologies."""
    CEMS = "cems"  # Continuous Emissions Monitoring System
    FUEL_ANALYSIS = "fuel_analysis"  # Tier 4 fuel sampling
    EMISSION_FACTOR = "emission_factor"  # EPA AP-42 factors
    MASS_BALANCE = "mass_balance"  # Mass balance calculation
    PREDICTIVE = "predictive"  # PEMS/CPMS
    ENGINEERING_ESTIMATE = "engineering_estimate"


class RegulatoryProgram(str, Enum):
    """Applicable regulatory programs."""
    EPA_PART_98 = "epa_part_98"  # GHG Mandatory Reporting
    EPA_PART_75 = "epa_part_75"  # Acid Rain Program
    EPA_PART_60 = "epa_part_60"  # NSPS
    EPA_PART_63 = "epa_part_63"  # NESHAP/MACT
    TITLE_V = "title_v"  # Operating Permits
    EU_ETS = "eu_ets"  # EU Emissions Trading
    CA_MRR = "ca_mrr"  # California MRR
    RGGI = "rggi"  # Regional GHG Initiative
    STATE_PERMIT = "state_permit"  # State air permit


class AlertSeverity(str, Enum):
    """Alert severity levels for emissions monitoring."""
    INFO = "info"
    ADVISORY = "advisory"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EXCEEDANCE = "exceedance"


class ComplianceStatus(str, Enum):
    """Emissions compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    EXCEEDANCE = "exceedance"
    DEVIATION = "deviation"
    UNKNOWN = "unknown"


class ExplainabilityMethod(str, Enum):
    """Explainability methods for ML models."""
    SHAP = "shap"
    LIME = "lime"
    SHAP_KERNEL = "shap_kernel"
    SHAP_TREE = "shap_tree"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    COUNTERFACTUAL = "counterfactual"


class OPCSecurityPolicy(str, Enum):
    """OPC-UA security policies."""
    NONE = "None"
    BASIC128RSA15 = "Basic128Rsa15"
    BASIC256 = "Basic256"
    BASIC256SHA256 = "Basic256Sha256"
    AES128_SHA256_RSAOAEP = "Aes128_Sha256_RsaOaep"
    AES256_SHA256_RSAPSS = "Aes256_Sha256_RsaPss"


class OPCSecurityMode(str, Enum):
    """OPC-UA security modes."""
    NONE = "None"
    SIGN = "Sign"
    SIGN_AND_ENCRYPT = "SignAndEncrypt"


# =============================================================================
# EMISSIONS MONITORING CONFIGURATION
# =============================================================================


class PermitLimitsConfig(BaseModel):
    """Permit emission limits configuration."""

    # GHG limits (lb/hr or tons/yr)
    co2_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 emission limit (lb/hr)"
    )
    co2_tons_yr: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 annual limit (tons/yr)"
    )

    # Criteria pollutant limits
    nox_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx emission limit (lb/hr)"
    )
    nox_lb_mmbtu: Optional[float] = Field(
        default=None,
        ge=0,
        description="NOx emission rate limit (lb/MMBtu)"
    )
    so2_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="SO2 emission limit (lb/hr)"
    )
    co_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO emission limit (lb/hr)"
    )
    pm_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="PM emission limit (lb/hr)"
    )
    voc_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="VOC emission limit (lb/hr)"
    )

    # Opacity limit
    opacity_pct: Optional[float] = Field(
        default=20.0,
        ge=0,
        le=100,
        description="Opacity limit (%)"
    )

    # Stack parameters
    stack_temperature_max_f: Optional[float] = Field(
        default=None,
        description="Maximum stack temperature (F)"
    )


class AlertThresholdsConfig(BaseModel):
    """Alert thresholds for emissions monitoring."""

    # Warning thresholds (% of limit)
    warning_threshold_pct: float = Field(
        default=80.0,
        ge=50,
        le=100,
        description="Warning threshold (% of limit)"
    )
    alarm_threshold_pct: float = Field(
        default=90.0,
        ge=70,
        le=100,
        description="Alarm threshold (% of limit)"
    )
    critical_threshold_pct: float = Field(
        default=95.0,
        ge=80,
        le=100,
        description="Critical threshold (% of limit)"
    )

    # Rate of change alerts
    rate_of_change_warning_pct_min: float = Field(
        default=5.0,
        ge=1,
        le=20,
        description="Rate of change warning (%/min)"
    )

    # Prediction thresholds
    exceedance_prediction_window_hr: float = Field(
        default=4.0,
        ge=1,
        le=24,
        description="Exceedance prediction window (hours)"
    )
    exceedance_probability_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Exceedance probability threshold for alert"
    )


class EmissionsMonitoringConfig(BaseModel):
    """
    Configuration for real-time emissions monitoring.

    Defines monitoring parameters, permit limits, alert thresholds,
    and calculation methods for emissions monitoring systems.

    Attributes:
        source_id: Emission source identifier
        monitoring_method: Primary monitoring methodology
        permit_limits: Emission permit limits
        alert_thresholds: Alert threshold configuration

    Example:
        >>> config = EmissionsMonitoringConfig(
        ...     source_id="STACK-001",
        ...     monitoring_method=MonitoringMethod.CEMS,
        ...     permit_limits=PermitLimitsConfig(nox_lb_hr=25.0),
        ... )
    """

    source_id: str = Field(
        ...,
        description="Emission source identifier"
    )
    source_name: str = Field(
        default="",
        description="Source description"
    )

    # Monitoring configuration
    monitoring_method: MonitoringMethod = Field(
        default=MonitoringMethod.CEMS,
        description="Primary monitoring method"
    )
    backup_monitoring_method: Optional[MonitoringMethod] = Field(
        default=MonitoringMethod.EMISSION_FACTOR,
        description="Backup monitoring method"
    )

    # Fuel configuration
    primary_fuel: FuelType = Field(
        default=FuelType.NATURAL_GAS,
        description="Primary fuel type"
    )
    secondary_fuels: List[FuelType] = Field(
        default_factory=list,
        description="Secondary fuel types"
    )

    # Monitored pollutants
    monitored_pollutants: List[PollutantType] = Field(
        default_factory=lambda: [
            PollutantType.CO2,
            PollutantType.NOX,
            PollutantType.CO,
            PollutantType.O2,
        ],
        description="Pollutants being monitored"
    )

    # Permit limits
    permit_limits: PermitLimitsConfig = Field(
        default_factory=PermitLimitsConfig,
        description="Permit emission limits"
    )

    # Alert thresholds
    alert_thresholds: AlertThresholdsConfig = Field(
        default_factory=AlertThresholdsConfig,
        description="Alert threshold configuration"
    )

    # Sampling configuration
    sampling_interval_s: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Data sampling interval (seconds)"
    )
    averaging_period_min: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Data averaging period (minutes)"
    )

    # Data quality
    min_data_availability_pct: float = Field(
        default=90.0,
        ge=75,
        le=100,
        description="Minimum data availability required (%)"
    )
    max_substitute_data_pct: float = Field(
        default=10.0,
        ge=0,
        le=25,
        description="Maximum substitute data allowed (%)"
    )

    # Regulatory programs
    applicable_programs: List[RegulatoryProgram] = Field(
        default_factory=lambda: [RegulatoryProgram.TITLE_V],
        description="Applicable regulatory programs"
    )

    # Enable predictive monitoring
    predictive_monitoring_enabled: bool = Field(
        default=True,
        description="Enable predictive exceedance monitoring"
    )
    anomaly_detection_enabled: bool = Field(
        default=True,
        description="Enable anomaly detection"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# CEMS INTEGRATION CONFIGURATION
# =============================================================================


class CEMSAnalyzerConfig(BaseModel):
    """CEMS analyzer configuration."""

    analyzer_id: str = Field(
        ...,
        description="Analyzer identifier"
    )
    pollutant: PollutantType = Field(
        ...,
        description="Monitored pollutant"
    )
    analyzer_type: str = Field(
        default="NDIR",
        description="Analyzer type (NDIR, UV, chemiluminescence, etc.)"
    )

    # Span and range
    span_value: float = Field(
        ...,
        gt=0,
        description="Analyzer span value"
    )
    measurement_range_low: float = Field(
        default=0.0,
        ge=0,
        description="Low measurement range"
    )
    measurement_range_high: float = Field(
        ...,
        gt=0,
        description="High measurement range"
    )
    unit: str = Field(
        default="ppm",
        description="Measurement unit (ppm, %, lb/hr)"
    )

    # Response characteristics
    response_time_s: float = Field(
        default=30.0,
        ge=1,
        le=300,
        description="T90 response time (seconds)"
    )

    # Calibration
    cal_gas_low_ppm: float = Field(
        default=0.0,
        ge=0,
        description="Low calibration gas concentration"
    )
    cal_gas_mid_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Mid calibration gas concentration"
    )
    cal_gas_high_ppm: float = Field(
        ...,
        gt=0,
        description="High calibration gas concentration"
    )


class CEMSIntegrationConfig(BaseModel):
    """
    CEMS integration configuration for EPA Part 75 compliance.

    Configures CEMS data acquisition, QA/QC requirements, and
    data validation for continuous emissions monitoring systems.

    Attributes:
        enabled: Enable CEMS integration
        unit_id: CEMS unit identifier
        analyzers: List of analyzer configurations

    Example:
        >>> config = CEMSIntegrationConfig(
        ...     enabled=True,
        ...     unit_id="CEMS-001",
        ...     analyzers=[
        ...         CEMSAnalyzerConfig(
        ...             analyzer_id="NOX-01",
        ...             pollutant=PollutantType.NOX,
        ...             span_value=500,
        ...             measurement_range_high=500,
        ...             cal_gas_high_ppm=400,
        ...         ),
        ...     ],
        ... )
    """

    enabled: bool = Field(
        default=False,
        description="Enable CEMS integration"
    )

    unit_id: str = Field(
        default="CEMS-001",
        description="CEMS unit identifier"
    )
    unit_name: str = Field(
        default="",
        description="CEMS unit name"
    )

    # Analyzer configurations
    analyzers: List[CEMSAnalyzerConfig] = Field(
        default_factory=list,
        description="CEMS analyzer configurations"
    )

    # QA/QC requirements
    daily_calibration_enabled: bool = Field(
        default=True,
        description="Enable daily calibration drift checks"
    )
    daily_calibration_time: str = Field(
        default="00:00",
        description="Daily calibration time (HH:MM)"
    )
    calibration_drift_limit_pct: float = Field(
        default=2.5,
        ge=0,
        le=10,
        description="Daily calibration drift limit (% of span)"
    )

    # RATA requirements
    rata_frequency_quarters: int = Field(
        default=4,
        ge=1,
        le=4,
        description="RATA frequency (1=annual, 4=quarterly)"
    )
    rata_relative_accuracy_limit: float = Field(
        default=10.0,
        ge=5,
        le=20,
        description="RATA relative accuracy limit (%)"
    )

    # Cylinder gas audit
    cga_frequency_quarters: int = Field(
        default=4,
        ge=1,
        le=4,
        description="CGA frequency (quarters)"
    )
    cga_accuracy_limit_pct: float = Field(
        default=5.0,
        ge=2,
        le=10,
        description="CGA accuracy limit (% of span)"
    )

    # Linearity
    linearity_check_frequency_quarters: int = Field(
        default=4,
        ge=1,
        le=4,
        description="Linearity check frequency"
    )
    linearity_error_limit_pct: float = Field(
        default=5.0,
        ge=2,
        le=10,
        description="Linearity error limit (%)"
    )

    # Data substitution
    missing_data_procedure: str = Field(
        default="standard_75",
        description="Missing data procedure (standard_75, maximum, lookback)"
    )
    lookback_hours: int = Field(
        default=720,
        ge=168,
        le=2160,
        description="Lookback hours for substitute data"
    )

    # Data validation
    out_of_range_handling: str = Field(
        default="flag",
        description="Out of range handling (flag, cap, reject)"
    )
    spike_detection_enabled: bool = Field(
        default=True,
        description="Enable spike detection"
    )
    spike_threshold_pct: float = Field(
        default=50.0,
        ge=20,
        le=100,
        description="Spike detection threshold (%)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# RATA QA/QC CONFIGURATION
# =============================================================================


class RATAConfig(BaseModel):
    """
    RATA (Relative Accuracy Test Audit) configuration per 40 CFR 75.

    Configures RATA scheduling, execution parameters, and pass/fail
    criteria for CEMS quality assurance testing.
    """

    enabled: bool = Field(
        default=True,
        description="Enable RATA tracking"
    )

    # Scheduling
    base_frequency: str = Field(
        default="quarterly",
        description="Base RATA frequency (quarterly, semiannual, annual)"
    )
    auto_schedule_enabled: bool = Field(
        default=True,
        description="Enable automatic scheduling"
    )
    advance_notice_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Advance notice for RATA scheduling (days)"
    )

    # Test requirements
    min_test_runs: int = Field(
        default=9,
        ge=9,
        le=12,
        description="Minimum valid test runs"
    )
    min_run_duration_min: int = Field(
        default=21,
        ge=15,
        le=60,
        description="Minimum run duration (minutes)"
    )

    # Pass/fail criteria
    relative_accuracy_limit_pct: float = Field(
        default=10.0,
        ge=5,
        le=20,
        description="Relative accuracy limit (%)"
    )
    alternate_ra_limit_pct: float = Field(
        default=20.0,
        ge=10,
        le=30,
        description="Alternate RA limit for low emissions (%)"
    )
    diluent_absolute_limit_pct: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Diluent (O2/CO2) absolute limit (%)"
    )

    # Bias adjustment
    bias_adjustment_threshold_pct: float = Field(
        default=5.0,
        ge=2,
        le=10,
        description="Bias adjustment factor threshold (%)"
    )
    auto_apply_bias_adjustment: bool = Field(
        default=True,
        description="Automatically apply bias adjustment"
    )

    # Frequency adjustment
    frequency_upgrade_threshold_pct: float = Field(
        default=7.5,
        ge=5,
        le=10,
        description="RA threshold for frequency upgrade (%)"
    )
    consecutive_good_ratas_for_upgrade: int = Field(
        default=2,
        ge=2,
        le=4,
        description="Consecutive good RATAs for frequency upgrade"
    )


# =============================================================================
# CARBON TRADING CONFIGURATION
# =============================================================================


class CarbonTradingConfig(BaseModel):
    """
    Carbon emission trading configuration.

    Configures carbon market integration, portfolio management,
    and compliance position tracking for emission trading schemes.
    """

    enabled: bool = Field(
        default=False,
        description="Enable carbon trading features"
    )

    # Market configuration
    primary_market: str = Field(
        default="voluntary",
        description="Primary trading market (eu_ets, ca_cap_trade, rggi, voluntary)"
    )
    secondary_markets: List[str] = Field(
        default_factory=list,
        description="Secondary trading markets"
    )

    # Entity information
    entity_id: str = Field(
        default="",
        description="Trading entity identifier"
    )
    registry_account: Optional[str] = Field(
        default=None,
        description="Registry account number"
    )

    # Compliance settings
    compliance_year_start_month: int = Field(
        default=1,
        ge=1,
        le=12,
        description="Compliance year start month"
    )
    offset_usage_limit_pct: float = Field(
        default=8.0,
        ge=0,
        le=20,
        description="Maximum offset usage for compliance (%)"
    )

    # Quality requirements
    minimum_offset_quality_score: float = Field(
        default=60.0,
        ge=0,
        le=100,
        description="Minimum acceptable offset quality score"
    )
    require_third_party_verification: bool = Field(
        default=True,
        description="Require third-party verification"
    )

    # Price risk management
    price_alert_enabled: bool = Field(
        default=True,
        description="Enable price movement alerts"
    )
    price_alert_threshold_pct: float = Field(
        default=10.0,
        ge=5,
        le=50,
        description="Price change alert threshold (%)"
    )

    # Position management
    auto_rebalance_enabled: bool = Field(
        default=False,
        description="Enable automatic portfolio rebalancing"
    )
    target_surplus_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Target surplus position (%)"
    )


# =============================================================================
# LDAR CONFIGURATION
# =============================================================================


class LDARConfig(BaseModel):
    """
    Leak Detection and Repair (LDAR) configuration per EPA Method 21.

    Configures fugitive emissions monitoring, inspection scheduling,
    and repair tracking for LDAR compliance programs.
    """

    enabled: bool = Field(
        default=False,
        description="Enable LDAR tracking"
    )

    # Regulatory program
    regulation_program: str = Field(
        default="epa_subpart_vva",
        description="LDAR regulation program"
    )

    # Leak thresholds (ppm)
    valve_leak_threshold_ppm: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Valve leak threshold (ppm)"
    )
    pump_leak_threshold_ppm: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Pump leak threshold (ppm)"
    )
    connector_leak_threshold_ppm: int = Field(
        default=500,
        ge=100,
        le=10000,
        description="Connector leak threshold (ppm)"
    )

    # Monitoring frequency
    valve_monitoring_frequency_days: int = Field(
        default=91,
        ge=7,
        le=365,
        description="Valve monitoring frequency (days)"
    )
    pump_monitoring_frequency_days: int = Field(
        default=7,
        ge=1,
        le=91,
        description="Pump monitoring frequency (days)"
    )

    # Repair timelines
    first_attempt_repair_days: int = Field(
        default=5,
        ge=1,
        le=15,
        description="First attempt repair deadline (days)"
    )
    final_repair_days: int = Field(
        default=15,
        ge=5,
        le=45,
        description="Final repair deadline (days)"
    )

    # OGI integration
    ogi_monitoring_enabled: bool = Field(
        default=False,
        description="Enable OGI monitoring"
    )
    ogi_survey_frequency_days: int = Field(
        default=60,
        ge=30,
        le=180,
        description="OGI survey frequency (days)"
    )


# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================


class ReportingConfig(BaseModel):
    """
    Regulatory reporting configuration.

    Configures automated report generation for EPA Part 98,
    Title V, state emission inventories, and other programs.
    """

    enabled: bool = Field(
        default=True,
        description="Enable automated reporting"
    )

    # Facility information
    facility_name: str = Field(
        default="",
        description="Facility name"
    )
    epa_facility_id: Optional[str] = Field(
        default=None,
        description="EPA facility ID (GHGRP)"
    )
    state_facility_id: Optional[str] = Field(
        default=None,
        description="State facility ID"
    )
    naics_code: str = Field(
        default="",
        description="Primary NAICS code"
    )

    # Report types
    part98_reporting_enabled: bool = Field(
        default=False,
        description="Enable EPA Part 98 reporting"
    )
    part98_subparts: List[str] = Field(
        default_factory=lambda: ["C"],
        description="Part 98 subparts to report"
    )

    title_v_reporting_enabled: bool = Field(
        default=True,
        description="Enable Title V reporting"
    )
    title_v_permit_number: Optional[str] = Field(
        default=None,
        description="Title V permit number"
    )

    emission_inventory_enabled: bool = Field(
        default=True,
        description="Enable emission inventory reporting"
    )

    # Deadline management
    deadline_reminder_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Deadline reminder advance (days)"
    )
    auto_generate_draft: bool = Field(
        default=True,
        description="Automatically generate draft reports"
    )

    # Export formats
    export_formats: List[str] = Field(
        default_factory=lambda: ["xml", "csv", "json"],
        description="Supported export formats"
    )

    # Data retention
    report_retention_years: int = Field(
        default=7,
        ge=3,
        le=20,
        description="Report retention period (years)"
    )


# =============================================================================
# SHAP/LIME EXPLAINABILITY CONFIGURATION
# =============================================================================


class SHAPConfig(BaseModel):
    """SHAP explainability configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable SHAP explainability"
    )
    method: str = Field(
        default="kernel",
        description="SHAP method (kernel, tree, deep, linear)"
    )
    n_samples: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of samples for SHAP calculation"
    )
    background_samples: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Background samples for SHAP kernel"
    )
    interaction_analysis: bool = Field(
        default=False,
        description="Enable SHAP interaction analysis"
    )
    cache_explanations: bool = Field(
        default=True,
        description="Cache SHAP explanations"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="SHAP cache TTL (hours)"
    )


class LIMEConfig(BaseModel):
    """LIME explainability configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable LIME explainability"
    )
    num_features: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Number of features in explanation"
    )
    num_samples: int = Field(
        default=5000,
        ge=100,
        le=20000,
        description="Number of samples for LIME"
    )
    discretize_continuous: bool = Field(
        default=True,
        description="Discretize continuous features"
    )
    feature_selection: str = Field(
        default="lasso_path",
        description="Feature selection method"
    )


class ExplainabilityConfig(BaseModel):
    """
    Complete explainability configuration for emissions models.

    Integrates SHAP and LIME configurations for comprehensive
    model explanation capabilities.
    """

    enabled: bool = Field(
        default=True,
        description="Enable explainability features"
    )
    primary_method: ExplainabilityMethod = Field(
        default=ExplainabilityMethod.SHAP,
        description="Primary explainability method"
    )

    # Method configurations
    shap: SHAPConfig = Field(
        default_factory=SHAPConfig,
        description="SHAP configuration"
    )
    lime: LIMEConfig = Field(
        default_factory=LIMEConfig,
        description="LIME configuration"
    )

    # Explanation generation
    auto_explain_exceedances: bool = Field(
        default=True,
        description="Auto-generate explanations for exceedances"
    )
    auto_explain_anomalies: bool = Field(
        default=True,
        description="Auto-generate explanations for anomalies"
    )

    # Feature importance
    feature_importance_enabled: bool = Field(
        default=True,
        description="Calculate global feature importance"
    )
    top_features_to_report: int = Field(
        default=10,
        ge=5,
        le=30,
        description="Number of top features to report"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# PROVENANCE TRACKING CONFIGURATION
# =============================================================================


class ProvenanceConfig(BaseModel):
    """
    Provenance tracking configuration for audit trail compliance.

    Implements SHA-256 hashing for complete data lineage and
    reproducibility in regulatory environments.
    """

    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    # Hashing configuration
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm (sha256, sha384, sha512)"
    )
    hash_inputs: bool = Field(
        default=True,
        description="Hash all input data"
    )
    hash_outputs: bool = Field(
        default=True,
        description="Hash all output data"
    )
    hash_intermediate: bool = Field(
        default=False,
        description="Hash intermediate calculation results"
    )

    # Data lineage
    track_data_sources: bool = Field(
        default=True,
        description="Track all data sources"
    )
    track_calculation_steps: bool = Field(
        default=True,
        description="Track calculation steps"
    )
    track_config_versions: bool = Field(
        default=True,
        description="Track configuration versions"
    )

    # Timestamps
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in provenance"
    )
    timestamp_format: str = Field(
        default="iso8601",
        description="Timestamp format (iso8601, unix)"
    )

    # Storage
    retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Provenance data retention (days)"
    )

    # Verification
    verify_on_read: bool = Field(
        default=False,
        description="Verify provenance on data read"
    )
    alert_on_mismatch: bool = Field(
        default=True,
        description="Alert on provenance mismatch"
    )


# =============================================================================
# OPC-UA INTEGRATION CONFIGURATION
# =============================================================================


class OPCUANodeConfig(BaseModel):
    """Configuration for an OPC-UA node mapping."""

    node_id: str = Field(
        ...,
        description="OPC-UA node ID"
    )
    tag_name: str = Field(
        ...,
        description="Local tag name"
    )
    data_type: str = Field(
        default="Double",
        description="Data type"
    )
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Sampling interval (milliseconds)"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )


class OPCUAConfig(BaseModel):
    """
    OPC-UA integration configuration for CEMS/DCS connectivity.

    Configures OPC-UA client connection for real-time data
    acquisition from industrial control systems.
    """

    enabled: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )

    # Connection settings
    endpoint: str = Field(
        default="opc.tcp://localhost:4840/greenlang/",
        description="OPC-UA server endpoint URL"
    )
    application_name: str = Field(
        default="GL010-EmissionsGuardian",
        description="OPC-UA application name"
    )

    # Security
    security_policy: OPCSecurityPolicy = Field(
        default=OPCSecurityPolicy.BASIC256SHA256,
        description="OPC-UA security policy"
    )
    security_mode: OPCSecurityMode = Field(
        default=OPCSecurityMode.SIGN_AND_ENCRYPT,
        description="OPC-UA security mode"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Client certificate path"
    )
    username: Optional[str] = Field(
        default=None,
        description="Username for authentication"
    )

    # Connection parameters
    timeout_ms: int = Field(
        default=30000,
        ge=1000,
        le=120000,
        description="Connection timeout (milliseconds)"
    )
    reconnect_interval_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Reconnect interval (milliseconds)"
    )

    # Node configurations
    nodes: List[OPCUANodeConfig] = Field(
        default_factory=list,
        description="OPC-UA node configurations"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# EPA/NFPA COMPLIANCE CONFIGURATION
# =============================================================================


class EPAComplianceConfig(BaseModel):
    """
    EPA regulatory compliance configuration.

    Configures compliance monitoring for various EPA programs
    including Part 60, 63, 75, and 98.
    """

    # Part 60 - NSPS
    part_60_applicable: bool = Field(
        default=False,
        description="Part 60 NSPS applicable"
    )
    part_60_subparts: List[str] = Field(
        default_factory=list,
        description="Applicable Part 60 subparts"
    )

    # Part 63 - NESHAP/MACT
    part_63_applicable: bool = Field(
        default=False,
        description="Part 63 NESHAP applicable"
    )
    part_63_subparts: List[str] = Field(
        default_factory=list,
        description="Applicable Part 63 subparts"
    )

    # Part 75 - Acid Rain
    part_75_applicable: bool = Field(
        default=False,
        description="Part 75 Acid Rain applicable"
    )
    acid_rain_permit_number: Optional[str] = Field(
        default=None,
        description="Acid Rain permit number"
    )

    # Part 98 - GHG Reporting
    part_98_applicable: bool = Field(
        default=False,
        description="Part 98 GHG Reporting applicable"
    )
    part_98_threshold_mtco2e: float = Field(
        default=25000,
        ge=0,
        description="Part 98 reporting threshold (mtCO2e)"
    )

    # Title V
    title_v_applicable: bool = Field(
        default=True,
        description="Title V applicable"
    )
    title_v_permit_number: Optional[str] = Field(
        default=None,
        description="Title V permit number"
    )

    # State requirements
    state_program: Optional[str] = Field(
        default=None,
        description="State air program"
    )


class NFPA85Config(BaseModel):
    """
    NFPA 85 compliance configuration for combustion safety.

    Configures boiler combustion safety interlocks and
    emissions-related safety monitoring.
    """

    enabled: bool = Field(
        default=False,
        description="Enable NFPA 85 compliance monitoring"
    )

    # High emissions interlocks
    high_co_interlock_enabled: bool = Field(
        default=True,
        description="Enable high CO interlock"
    )
    high_co_alarm_ppm: float = Field(
        default=400,
        ge=100,
        le=1000,
        description="High CO alarm setpoint (ppm)"
    )
    high_co_trip_ppm: float = Field(
        default=800,
        ge=200,
        le=2000,
        description="High CO trip setpoint (ppm)"
    )

    # Low O2 interlocks
    low_o2_interlock_enabled: bool = Field(
        default=True,
        description="Enable low O2 interlock"
    )
    low_o2_alarm_pct: float = Field(
        default=3.0,
        ge=1,
        le=10,
        description="Low O2 alarm setpoint (%)"
    )
    low_o2_trip_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=5,
        description="Low O2 trip setpoint (%)"
    )

    # Response times
    alarm_response_time_s: float = Field(
        default=5.0,
        ge=1,
        le=30,
        description="Alarm response time (seconds)"
    )
    trip_delay_s: float = Field(
        default=10.0,
        ge=1,
        le=60,
        description="Trip delay time (seconds)"
    )


class ComplianceConfig(BaseModel):
    """
    Complete regulatory compliance configuration.

    Integrates EPA and NFPA compliance requirements.
    """

    epa: EPAComplianceConfig = Field(
        default_factory=EPAComplianceConfig,
        description="EPA compliance configuration"
    )
    nfpa_85: NFPA85Config = Field(
        default_factory=NFPA85Config,
        description="NFPA 85 compliance configuration"
    )

    # Audit trail
    compliance_audit_enabled: bool = Field(
        default=True,
        description="Enable compliance audit trail"
    )
    audit_retention_years: int = Field(
        default=7,
        ge=3,
        le=20,
        description="Audit record retention (years)"
    )


# =============================================================================
# MASTER CONFIGURATION
# =============================================================================


class GL010Config(BaseModel):
    """
    Master configuration for GL-010 EmissionsGuardian Agent.

    This configuration combines all component configurations for
    comprehensive emissions monitoring, compliance tracking,
    and regulatory reporting.

    Attributes:
        monitoring: Emissions monitoring configuration
        cems: CEMS integration configuration
        rata: RATA QA/QC configuration
        trading: Carbon trading configuration
        ldar: LDAR configuration
        reporting: Regulatory reporting configuration
        explainability: SHAP/LIME explainability settings
        provenance: Provenance tracking configuration
        opcua: OPC-UA integration settings
        compliance: EPA/NFPA compliance configuration

    Example:
        >>> config = GL010Config(
        ...     monitoring=EmissionsMonitoringConfig(
        ...         source_id="STACK-001",
        ...         permit_limits=PermitLimitsConfig(nox_lb_hr=25.0),
        ...     ),
        ...     cems=CEMSIntegrationConfig(enabled=True),
        ... )

    Standards Compliance:
        - EPA 40 CFR Part 60/61/63
        - EPA 40 CFR Part 75
        - EPA 40 CFR Part 98
        - EU ETS MRR
        - NFPA 85
        - ISO 14064
    """

    # Component configurations
    monitoring: EmissionsMonitoringConfig = Field(
        ...,
        description="Emissions monitoring configuration"
    )
    cems: CEMSIntegrationConfig = Field(
        default_factory=CEMSIntegrationConfig,
        description="CEMS integration configuration"
    )
    rata: RATAConfig = Field(
        default_factory=RATAConfig,
        description="RATA QA/QC configuration"
    )
    trading: CarbonTradingConfig = Field(
        default_factory=CarbonTradingConfig,
        description="Carbon trading configuration"
    )
    ldar: LDARConfig = Field(
        default_factory=LDARConfig,
        description="LDAR configuration"
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Regulatory reporting configuration"
    )
    explainability: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="Explainability configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance tracking configuration"
    )
    opcua: OPCUAConfig = Field(
        default_factory=OPCUAConfig,
        description="OPC-UA integration configuration"
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Regulatory compliance configuration"
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-010",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="EmissionsGuardian",
        description="Agent name"
    )
    version: str = Field(
        default="2.0.0",
        description="Configuration version"
    )

    # Data management
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Operational data retention (days)"
    )
    trend_history_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Trend analysis history (days)"
    )

    # Performance settings
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )

    class Config:
        use_enum_values = True
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_default_config(
    source_id: str,
    source_name: str = "",
    primary_fuel: FuelType = FuelType.NATURAL_GAS,
    nox_limit_lb_hr: Optional[float] = None,
    co2_limit_lb_hr: Optional[float] = None,
) -> GL010Config:
    """
    Create a default GL-010 configuration with typical values.

    Args:
        source_id: Emission source identifier
        source_name: Source description
        primary_fuel: Primary fuel type
        nox_limit_lb_hr: NOx permit limit (lb/hr)
        co2_limit_lb_hr: CO2 permit limit (lb/hr)

    Returns:
        GL010Config with typical settings

    Example:
        >>> config = create_default_config(
        ...     source_id="BOILER-001",
        ...     nox_limit_lb_hr=25.0,
        ... )
    """
    permit_limits = PermitLimitsConfig(
        nox_lb_hr=nox_limit_lb_hr,
        co2_lb_hr=co2_limit_lb_hr,
    )

    return GL010Config(
        monitoring=EmissionsMonitoringConfig(
            source_id=source_id,
            source_name=source_name,
            primary_fuel=primary_fuel,
            permit_limits=permit_limits,
        ),
    )


def create_cems_config(
    source_id: str,
    unit_id: str,
    nox_span: float = 500.0,
    co_span: float = 500.0,
    o2_span: float = 25.0,
) -> GL010Config:
    """
    Create configuration for CEMS-equipped source.

    Args:
        source_id: Emission source identifier
        unit_id: CEMS unit identifier
        nox_span: NOx analyzer span (ppm)
        co_span: CO analyzer span (ppm)
        o2_span: O2 analyzer span (%)

    Returns:
        GL010Config optimized for CEMS monitoring
    """
    analyzers = [
        CEMSAnalyzerConfig(
            analyzer_id=f"{unit_id}-NOX",
            pollutant=PollutantType.NOX,
            span_value=nox_span,
            measurement_range_high=nox_span,
            cal_gas_high_ppm=nox_span * 0.8,
        ),
        CEMSAnalyzerConfig(
            analyzer_id=f"{unit_id}-CO",
            pollutant=PollutantType.CO,
            span_value=co_span,
            measurement_range_high=co_span,
            cal_gas_high_ppm=co_span * 0.8,
        ),
        CEMSAnalyzerConfig(
            analyzer_id=f"{unit_id}-O2",
            pollutant=PollutantType.O2,
            analyzer_type="paramagnetic",
            span_value=o2_span,
            measurement_range_high=o2_span,
            cal_gas_high_ppm=o2_span * 0.8,
            unit="%",
        ),
    ]

    return GL010Config(
        monitoring=EmissionsMonitoringConfig(
            source_id=source_id,
            monitoring_method=MonitoringMethod.CEMS,
        ),
        cems=CEMSIntegrationConfig(
            enabled=True,
            unit_id=unit_id,
            analyzers=analyzers,
        ),
        rata=RATAConfig(enabled=True),
    )
