# -*- coding: utf-8 -*-
"""
GL-005 COMBUSENSE Configuration Module
======================================

This module defines all configuration schemas for the GL-005 Combustion
Diagnostics Agent. Configuration is validated using Pydantic models with
strict typing and constraint validation.

Key Configuration Areas:
    - CQI (Combustion Quality Index) thresholds and weights
    - Anomaly detection parameters (SPC limits, ML sensitivity)
    - Fuel characterization settings
    - Maintenance advisory thresholds
    - Trending analysis windows
    - Compliance reporting configuration (EPA, EU IED)

IMPORTANT: GL-005 is a DIAGNOSTICS-ONLY agent. It does NOT execute control
actions. All recommendations are advisory outputs for GL-018 or CMMS systems.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class DiagnosticMode(str, Enum):
    """Operating mode for diagnostics."""
    REAL_TIME = "real_time"           # Continuous monitoring
    BATCH = "batch"                   # Periodic batch analysis
    ON_DEMAND = "on_demand"           # Manual trigger
    SCHEDULED = "scheduled"           # Scheduled analysis


class ComplianceFramework(str, Enum):
    """Supported compliance frameworks."""
    EPA_40CFR60 = "epa_40cfr60"       # US EPA Clean Air Act
    EPA_40CFR63 = "epa_40cfr63"       # US EPA NESHAP
    EU_IED = "eu_ied"                 # EU Industrial Emissions Directive
    EU_MCP = "eu_mcp"                 # EU Medium Combustion Plants
    ISO_50001 = "iso_50001"           # Energy Management
    CUSTOM = "custom"                 # Custom compliance rules


class FuelCategory(str, Enum):
    """Fuel categories for characterization."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    PROPANE = "propane"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_ANTHRACITE = "coal_anthracite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLET = "biomass_pellet"
    BIOGAS = "biogas"
    HYDROGEN = "hydrogen"
    MIXED = "mixed"


class MaintenancePriority(str, Enum):
    """Maintenance work order priority levels."""
    CRITICAL = "critical"             # Immediate action required
    HIGH = "high"                     # Within 24 hours
    MEDIUM = "medium"                 # Within 7 days
    LOW = "low"                       # Within 30 days
    ROUTINE = "routine"               # Next scheduled outage


class AnomalyType(str, Enum):
    """Types of combustion anomalies."""
    EXCESS_OXYGEN = "excess_oxygen"
    LOW_OXYGEN = "low_oxygen"
    HIGH_CO = "high_co"
    HIGH_NOX = "high_nox"
    HIGH_COMBUSTIBLES = "high_combustibles"
    FLAME_INSTABILITY = "flame_instability"
    FOULING_DETECTED = "fouling_detected"
    BURNER_WEAR = "burner_wear"
    AIR_FUEL_IMBALANCE = "air_fuel_imbalance"
    HEAT_TRANSFER_DEGRADATION = "heat_transfer_degradation"


# =============================================================================
# CQI CONFIGURATION
# =============================================================================

class CQIWeights(BaseModel):
    """Weights for Combustion Quality Index components."""

    oxygen: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Weight for O2 component (0-1)"
    )
    carbon_monoxide: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for CO component (0-1)"
    )
    carbon_dioxide: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for CO2 component (0-1)"
    )
    nox: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for NOx component (0-1)"
    )
    combustibles: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for unburned combustibles (0-1)"
    )

    @validator("combustibles")
    def validate_total_weight(cls, v, values):
        """Ensure weights sum to 1.0."""
        total = (
            values.get("oxygen", 0.25) +
            values.get("carbon_monoxide", 0.30) +
            values.get("carbon_dioxide", 0.15) +
            values.get("nox", 0.15) +
            v
        )
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"CQI weights must sum to 1.0, got {total}")
        return v


class CQIThresholds(BaseModel):
    """Threshold definitions for CQI scoring."""

    # Oxygen thresholds (% dry basis)
    o2_optimal_min: float = Field(default=2.0, ge=0.0, le=21.0)
    o2_optimal_max: float = Field(default=4.0, ge=0.0, le=21.0)
    o2_acceptable_max: float = Field(default=6.0, ge=0.0, le=21.0)
    o2_warning_max: float = Field(default=8.0, ge=0.0, le=21.0)

    # CO thresholds (ppm corrected to 3% O2)
    co_excellent: float = Field(default=50.0, ge=0.0)
    co_good: float = Field(default=100.0, ge=0.0)
    co_acceptable: float = Field(default=200.0, ge=0.0)
    co_warning: float = Field(default=400.0, ge=0.0)

    # NOx thresholds (ppm corrected to 3% O2)
    nox_excellent: float = Field(default=50.0, ge=0.0)
    nox_good: float = Field(default=100.0, ge=0.0)
    nox_acceptable: float = Field(default=150.0, ge=0.0)
    nox_warning: float = Field(default=250.0, ge=0.0)

    # Combustibles thresholds (% by volume)
    combustibles_excellent: float = Field(default=0.1, ge=0.0)
    combustibles_good: float = Field(default=0.3, ge=0.0)
    combustibles_acceptable: float = Field(default=0.5, ge=0.0)
    combustibles_warning: float = Field(default=1.0, ge=0.0)

    # CQI score interpretation
    cqi_excellent: float = Field(default=90.0, ge=0.0, le=100.0)
    cqi_good: float = Field(default=75.0, ge=0.0, le=100.0)
    cqi_acceptable: float = Field(default=60.0, ge=0.0, le=100.0)
    cqi_poor: float = Field(default=40.0, ge=0.0, le=100.0)


class CQIConfig(BaseModel):
    """Complete CQI configuration."""

    weights: CQIWeights = Field(default_factory=CQIWeights)
    thresholds: CQIThresholds = Field(default_factory=CQIThresholds)

    # Scoring method
    scoring_method: str = Field(
        default="weighted_linear",
        description="Scoring method: weighted_linear, weighted_sigmoid, fuzzy"
    )

    # Correction reference
    o2_reference_pct: float = Field(
        default=3.0,
        ge=0.0,
        le=21.0,
        description="Reference O2% for emission corrections"
    )

    # Update frequency
    calculation_interval_s: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="CQI calculation interval in seconds"
    )


# =============================================================================
# ANOMALY DETECTION CONFIGURATION
# =============================================================================

class SPCConfig(BaseModel):
    """Statistical Process Control configuration."""

    # Control limits
    sigma_warning: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        description="Sigma multiplier for warning limits"
    )
    sigma_control: float = Field(
        default=3.0,
        ge=2.0,
        le=5.0,
        description="Sigma multiplier for control limits"
    )

    # Run rules (Western Electric rules)
    enable_run_rules: bool = Field(
        default=True,
        description="Enable Western Electric run rules"
    )
    consecutive_one_side: int = Field(
        default=7,
        ge=5,
        le=10,
        description="Consecutive points on one side of centerline"
    )
    consecutive_trending: int = Field(
        default=6,
        ge=5,
        le=10,
        description="Consecutive points trending in same direction"
    )

    # Sample window
    baseline_window_size: int = Field(
        default=100,
        ge=20,
        le=1000,
        description="Number of samples for baseline calculation"
    )
    moving_window_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Moving window size for current statistics"
    )


class MLAnomalyConfig(BaseModel):
    """Machine learning anomaly detection configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable ML-based anomaly detection"
    )

    # Isolation Forest parameters
    contamination: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Expected proportion of anomalies"
    )
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of trees in Isolation Forest"
    )

    # Confidence thresholds
    anomaly_threshold: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Threshold for anomaly classification"
    )

    # Feature importance
    track_feature_importance: bool = Field(
        default=True,
        description="Track feature importance for anomalies"
    )

    # Model retraining
    retrain_interval_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Hours between model retraining"
    )
    min_samples_for_retrain: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Minimum samples required for retraining"
    )


class AnomalyDetectionConfig(BaseModel):
    """Complete anomaly detection configuration."""

    spc: SPCConfig = Field(default_factory=SPCConfig)
    ml: MLAnomalyConfig = Field(default_factory=MLAnomalyConfig)

    # Detection modes
    detection_modes: List[str] = Field(
        default=["spc", "ml", "rule_based"],
        description="Active detection modes"
    )

    # Alert aggregation
    alert_cooldown_s: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cooldown period between same-type alerts"
    )

    # Severity escalation
    escalation_window_s: int = Field(
        default=900,
        ge=300,
        le=7200,
        description="Window for escalation assessment"
    )
    escalation_threshold_count: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Anomaly count to trigger escalation"
    )


# =============================================================================
# FUEL CHARACTERIZATION CONFIGURATION
# =============================================================================

class FuelCharacterizationConfig(BaseModel):
    """Fuel characterization from flue gas analysis configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable fuel characterization"
    )

    # Reference fuel library
    fuel_library_path: Optional[str] = Field(
        default=None,
        description="Path to custom fuel property database"
    )

    # Analysis settings
    stoichiometric_tolerance: float = Field(
        default=0.02,
        ge=0.001,
        le=0.1,
        description="Tolerance for stoichiometric calculations"
    )

    # Carbon balance method
    carbon_balance_method: str = Field(
        default="flue_gas",
        description="Method: flue_gas, fuel_analysis, hybrid"
    )

    # Fuel blend detection
    detect_fuel_blends: bool = Field(
        default=True,
        description="Attempt to detect fuel blends"
    )
    blend_detection_confidence: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum confidence for blend detection"
    )

    # Update frequency
    characterization_interval_s: float = Field(
        default=300.0,
        ge=60.0,
        le=3600.0,
        description="Fuel characterization interval in seconds"
    )


# =============================================================================
# MAINTENANCE ADVISORY CONFIGURATION
# =============================================================================

class FoulingPredictionConfig(BaseModel):
    """Fouling prediction configuration."""

    enabled: bool = Field(default=True)

    # Prediction horizon
    prediction_horizon_days: int = Field(
        default=30,
        ge=7,
        le=180,
        description="Days to predict fouling ahead"
    )

    # Thresholds
    fouling_warning_pct: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Efficiency loss % for warning"
    )
    fouling_critical_pct: float = Field(
        default=10.0,
        ge=5.0,
        le=30.0,
        description="Efficiency loss % for critical"
    )

    # Features for prediction
    use_stack_temp: bool = Field(default=True)
    use_delta_t: bool = Field(default=True)
    use_excess_air: bool = Field(default=True)


class BurnerWearConfig(BaseModel):
    """Burner wear prediction configuration."""

    enabled: bool = Field(default=True)

    # Prediction horizon
    prediction_horizon_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Days to predict burner wear ahead"
    )

    # Wear indicators
    co_trend_threshold: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="CO increase rate (%/day) indicating wear"
    )
    flame_stability_threshold: float = Field(
        default=0.95,
        ge=0.8,
        le=1.0,
        description="Flame stability index threshold"
    )

    # Operating hours tracking
    track_operating_hours: bool = Field(default=True)
    expected_burner_life_hours: int = Field(
        default=20000,
        ge=5000,
        le=100000,
        description="Expected burner life in operating hours"
    )


class MaintenanceAdvisoryConfig(BaseModel):
    """Complete maintenance advisory configuration."""

    fouling: FoulingPredictionConfig = Field(default_factory=FoulingPredictionConfig)
    burner_wear: BurnerWearConfig = Field(default_factory=BurnerWearConfig)

    # CMMS integration
    cmms_enabled: bool = Field(
        default=False,
        description="Enable CMMS work order generation"
    )
    cmms_api_url: Optional[str] = Field(
        default=None,
        description="CMMS API endpoint"
    )
    cmms_system: str = Field(
        default="generic",
        description="CMMS system: sap_pm, maximo, fiix, generic"
    )

    # Work order defaults
    default_priority: MaintenancePriority = Field(
        default=MaintenancePriority.MEDIUM
    )
    auto_create_work_orders: bool = Field(
        default=False,
        description="Automatically create work orders"
    )
    work_order_approval_required: bool = Field(
        default=True,
        description="Require approval before creating work orders"
    )


# =============================================================================
# TRENDING CONFIGURATION
# =============================================================================

class TrendingConfig(BaseModel):
    """Long-term trending and analysis configuration."""

    enabled: bool = Field(default=True)

    # Data retention
    raw_data_retention_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Days to retain raw data"
    )
    aggregated_data_retention_days: int = Field(
        default=730,
        ge=365,
        le=3650,
        description="Days to retain aggregated data"
    )

    # Aggregation intervals
    aggregation_intervals: List[str] = Field(
        default=["1h", "1d", "1w", "1M"],
        description="Time aggregation intervals"
    )

    # Trend detection
    trend_detection_window_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Window for trend detection"
    )
    trend_significance_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.1,
        description="p-value threshold for trend significance"
    )

    # Seasonality detection
    detect_seasonality: bool = Field(default=True)
    seasonal_periods: List[int] = Field(
        default=[24, 168, 8760],
        description="Seasonal periods in hours (daily, weekly, yearly)"
    )

    # Baseline comparison
    baseline_period_days: int = Field(
        default=90,
        ge=30,
        le=365,
        description="Days for baseline comparison"
    )


# =============================================================================
# COMPLIANCE CONFIGURATION
# =============================================================================

class ComplianceConfig(BaseModel):
    """Compliance reporting configuration."""

    frameworks: List[ComplianceFramework] = Field(
        default=[ComplianceFramework.EPA_40CFR60],
        description="Active compliance frameworks"
    )

    # EPA specific
    epa_source_category: Optional[str] = Field(
        default=None,
        description="EPA source category code"
    )
    epa_facility_id: Optional[str] = Field(
        default=None,
        description="EPA facility identification"
    )

    # EU specific
    eu_installation_id: Optional[str] = Field(
        default=None,
        description="EU ETS installation ID"
    )
    eu_activity_code: Optional[str] = Field(
        default=None,
        description="EU IED activity code"
    )

    # Reporting
    reporting_interval: str = Field(
        default="monthly",
        description="Reporting interval: hourly, daily, monthly, quarterly, annual"
    )

    # Exceedance tracking
    track_exceedances: bool = Field(default=True)
    exceedance_notification_enabled: bool = Field(default=True)
    exceedance_notification_recipients: List[str] = Field(
        default_factory=list,
        description="Email recipients for exceedance notifications"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class GL005Config(BaseModel):
    """
    Complete GL-005 COMBUSENSE Agent Configuration.

    This is the master configuration class that combines all sub-configurations
    for the Combustion Diagnostics Agent. It defines operating parameters,
    thresholds, and integration settings.

    IMPORTANT: GL-005 is DIAGNOSTICS ONLY. It reads data and provides
    recommendations but does NOT execute any control actions. Control
    remains with GL-018 Unified Combustion Control agent.

    Example:
        >>> config = GL005Config(
        ...     agent_id="GL005-BOILER-01",
        ...     equipment_id="BLR-001",
        ...     mode=DiagnosticMode.REAL_TIME,
        ... )
        >>> agent = CombustionDiagnosticsAgent(config)
    """

    # Agent identification
    agent_id: str = Field(
        ...,
        description="Unique agent identifier (e.g., GL005-BOILER-01)"
    )
    agent_name: str = Field(
        default="GL-005 COMBUSENSE",
        description="Human-readable agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Equipment identification
    equipment_id: str = Field(
        ...,
        description="Target equipment identifier"
    )
    equipment_type: str = Field(
        default="boiler",
        description="Equipment type: boiler, furnace, heater, kiln"
    )
    equipment_name: Optional[str] = Field(
        default=None,
        description="Human-readable equipment name"
    )

    # Operating mode
    mode: DiagnosticMode = Field(
        default=DiagnosticMode.REAL_TIME,
        description="Diagnostic operating mode"
    )

    # Primary fuel type
    primary_fuel: FuelCategory = Field(
        default=FuelCategory.NATURAL_GAS,
        description="Primary fuel type"
    )

    # Sub-configurations
    cqi: CQIConfig = Field(
        default_factory=CQIConfig,
        description="Combustion Quality Index configuration"
    )
    anomaly_detection: AnomalyDetectionConfig = Field(
        default_factory=AnomalyDetectionConfig,
        description="Anomaly detection configuration"
    )
    fuel_characterization: FuelCharacterizationConfig = Field(
        default_factory=FuelCharacterizationConfig,
        description="Fuel characterization configuration"
    )
    maintenance: MaintenanceAdvisoryConfig = Field(
        default_factory=MaintenanceAdvisoryConfig,
        description="Maintenance advisory configuration"
    )
    trending: TrendingConfig = Field(
        default_factory=TrendingConfig,
        description="Trending analysis configuration"
    )
    compliance: ComplianceConfig = Field(
        default_factory=ComplianceConfig,
        description="Compliance reporting configuration"
    )

    # Data source (from GL-018)
    data_source_agent: str = Field(
        default="GL-018",
        description="Source agent for combustion data"
    )
    data_poll_interval_s: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Data polling interval in seconds"
    )

    # Audit and provenance
    enable_audit: bool = Field(
        default=True,
        description="Enable comprehensive audit trail"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # Performance settings
    max_concurrent_analyses: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum concurrent analysis tasks"
    )
    analysis_timeout_s: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Analysis timeout in seconds"
    )

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_config(
    agent_id: str,
    equipment_id: str,
    fuel_type: FuelCategory = FuelCategory.NATURAL_GAS,
) -> GL005Config:
    """
    Create a default GL-005 configuration.

    Args:
        agent_id: Unique agent identifier
        equipment_id: Target equipment identifier
        fuel_type: Primary fuel type

    Returns:
        GL005Config with sensible defaults
    """
    return GL005Config(
        agent_id=agent_id,
        equipment_id=equipment_id,
        primary_fuel=fuel_type,
    )


def create_high_precision_config(
    agent_id: str,
    equipment_id: str,
    fuel_type: FuelCategory = FuelCategory.NATURAL_GAS,
) -> GL005Config:
    """
    Create a high-precision configuration for critical applications.

    Uses tighter thresholds and more frequent calculations.

    Args:
        agent_id: Unique agent identifier
        equipment_id: Target equipment identifier
        fuel_type: Primary fuel type

    Returns:
        GL005Config optimized for precision
    """
    config = GL005Config(
        agent_id=agent_id,
        equipment_id=equipment_id,
        primary_fuel=fuel_type,
    )

    # Tighter CQI thresholds
    config.cqi.thresholds.co_excellent = 25.0
    config.cqi.thresholds.co_good = 50.0
    config.cqi.calculation_interval_s = 30.0

    # More sensitive anomaly detection
    config.anomaly_detection.spc.sigma_warning = 1.5
    config.anomaly_detection.spc.sigma_control = 2.5
    config.anomaly_detection.ml.contamination = 0.02

    # Faster data polling
    config.data_poll_interval_s = 2.0

    return config


def create_compliance_focused_config(
    agent_id: str,
    equipment_id: str,
    frameworks: List[ComplianceFramework],
    fuel_type: FuelCategory = FuelCategory.NATURAL_GAS,
) -> GL005Config:
    """
    Create a compliance-focused configuration.

    Args:
        agent_id: Unique agent identifier
        equipment_id: Target equipment identifier
        frameworks: Active compliance frameworks
        fuel_type: Primary fuel type

    Returns:
        GL005Config optimized for compliance reporting
    """
    config = GL005Config(
        agent_id=agent_id,
        equipment_id=equipment_id,
        primary_fuel=fuel_type,
    )

    # Enable all compliance features
    config.compliance.frameworks = frameworks
    config.compliance.track_exceedances = True
    config.compliance.exceedance_notification_enabled = True

    # Enhanced trending for compliance
    config.trending.aggregated_data_retention_days = 3650  # 10 years
    config.trending.aggregation_intervals = ["1h", "1d", "1w", "1M", "1Y"]

    # Enhanced audit
    config.enable_audit = True
    config.enable_provenance = True

    return config
