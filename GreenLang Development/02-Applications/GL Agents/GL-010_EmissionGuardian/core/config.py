"""
GL-010 EmissionsGuardian - Configuration Module

This module defines all configuration schemas for the EmissionsGuardian
emissions compliance monitoring agent including CEMS parameters, permit
compliance, RATA automation, fugitive detection, carbon trading, and
offset certificate tracking.

Standards Compliance:
    - EPA 40 CFR Part 75 (Continuous Emissions Monitoring)
    - EPA 40 CFR Part 60 (Standards of Performance for New Stationary Sources)
    - EPA 40 CFR Part 63 (National Emission Standards for Hazardous Air Pollutants)

Zero-Hallucination Principle:
    - All emissions calculations use deterministic EPA methods
    - LLM used only for: classification, entity resolution, materiality
      assessment, narrative generation, and natural language summaries
    - Complete provenance tracking with SHA-256 hashes
"""

from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import os

from pydantic import BaseModel, Field, validator, root_validator


class Pollutant(Enum):
    """Monitored pollutants per EPA 40 CFR Part 75."""
    NOX = "nox"
    SO2 = "so2"
    CO2 = "co2"
    CO = "co"
    PM = "pm"
    PM10 = "pm10"
    PM25 = "pm25"
    VOC = "voc"
    NH3 = "nh3"
    HCL = "hcl"
    HG = "hg"
    O2 = "o2"
    CO2_DILUENT = "co2_diluent"
    FLOW = "flow"
    OPACITY = "opacity"
    MOISTURE = "moisture"


class MeasurementBasis(Enum):
    """Measurement basis for gas concentrations."""
    WET = "wet"
    DRY = "dry"
    CORRECTED = "corrected"


class CEMSDataQuality(Enum):
    """CEMS data quality indicators per 40 CFR Part 75."""
    VALID = "valid"
    SUBSTITUTED = "substituted"
    MISSING = "missing"
    MAINTENANCE = "maintenance"
    CALIBRATION = "calibration"
    QA_FAILED = "qa_failed"
    OUT_OF_RANGE = "out_of_range"
    SUSPECT = "suspect"


class AveragingPeriod(Enum):
    """Compliance averaging periods."""
    HOURLY = "hourly"
    ROLLING_3HR = "rolling_3hr"
    ROLLING_24HR = "rolling_24hr"
    DAILY = "daily"
    BLOCK_30DAY = "block_30day"
    ROLLING_30DAY = "rolling_30day"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    OZONE_SEASON = "ozone_season"


class PermitType(Enum):
    """Air quality permit types."""
    TITLE_V = "title_v"
    SYNTHETIC_MINOR = "synthetic_minor"
    PSD = "psd"
    NNSR = "nnsr"
    STATE_OPERATING = "state_operating"
    FEDERAL_OPERATING = "federal_operating"


class RATATestType(Enum):
    """RATA test types per 40 CFR Part 75."""
    STANDARD = "standard"
    ABBREVIATED = "abbreviated"
    SINGLE_LOAD = "single_load"
    THREE_LOAD = "three_load"
    CYLINDER_GAS = "cylinder_gas"


class CalibrationGasLevel(Enum):
    """Calibration gas levels."""
    ZERO = "zero"
    LOW = "low"
    MID = "mid"
    HIGH = "high"


class FugitiveSourceType(Enum):
    """Fugitive emission source types."""
    VALVE = "valve"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    PRESSURE_RELIEF = "pressure_relief"
    CONNECTOR = "connector"
    FLANGE = "flange"
    SAMPLING_CONNECTION = "sampling_connection"
    OPEN_ENDED_LINE = "open_ended_line"
    AGITATOR = "agitator"
    TANK = "tank"
    COOLING_TOWER = "cooling_tower"
    WASTEWATER = "wastewater"
    EQUIPMENT_LEAK = "equipment_leak"
    UNKNOWN = "unknown"


class CarbonMarket(Enum):
    """Carbon market/registry types."""
    EU_ETS = "eu_ets"
    CA_CaT = "ca_cat"
    RGGI = "rggi"
    WCI = "wci"
    CORSIA = "corsia"
    VOLUNTARY = "voluntary"
    INTERNAL = "internal"


class OffsetStandard(Enum):
    """Carbon offset verification standards."""
    VERRA_VCS = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    CDM = "cdm"
    CORSIA_ELIGIBLE = "corsia_eligible"
    INTERNAL = "internal"


class OffsetProjectType(Enum):
    """Carbon offset project types."""
    FORESTRY = "forestry"
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    METHANE_CAPTURE = "methane_capture"
    INDUSTRIAL_GAS = "industrial_gas"
    COOKSTOVES = "cookstoves"
    AGRICULTURE = "agriculture"
    BLUE_CARBON = "blue_carbon"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    CARBON_CAPTURE_STORAGE = "carbon_capture_storage"


class SecurityAccessLevel(Enum):
    """OT security access levels."""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    OPERATOR = "operator"
    ENGINEER = "engineer"
    ADMIN = "admin"


class OperatingState(Enum):
    """Unit operating states for emissions monitoring."""
    OPERATING = "operating"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    STANDBY = "standby"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class AgentConfig(BaseModel):
    """GL-010 EmissionsGuardian Agent Configuration."""
    agent_id: str = Field(default_factory=lambda: f"GL-010-{os.getpid()}")
    agent_code: str = Field(default="GL-010")
    agent_name: str = Field(default="EmissionsGuardian")
    agent_version: str = Field(default="1.0.0")
    description: str = Field(default="Emissions compliance monitoring agent")
    environment: str = Field(default="production")
    log_level: str = Field(default="INFO")
    advisory_mode: bool = Field(default=True)
    enable_ml_detection: bool = Field(default=True)
    enable_trading_recommendations: bool = Field(default=True)
    enable_offset_tracking: bool = Field(default=True)
    enable_explainability: bool = Field(default=True)
    enable_nl_summaries: bool = Field(default=True)
    processing_interval_seconds: int = Field(default=60, ge=1, le=3600)
    batch_size: int = Field(default=1000, ge=1, le=100000)
    max_concurrent_jobs: int = Field(default=4, ge=1, le=32)
    audit_enabled: bool = Field(default=True)
    provenance_tracking: bool = Field(default=True)
    deterministic_mode: bool = Field(default=True)
    random_seed: int = Field(default=42, ge=0)
    data_retention_days: int = Field(default=2557, ge=365, le=10000)
    audit_retention_days: int = Field(default=3650, ge=365, le=10000)

    class Config:
        use_enum_values = True


class MonitorSpan(BaseModel):
    """Analyzer span configuration."""
    pollutant: Pollutant = Field(...)
    low_span: float = Field(..., ge=0.0)
    high_span: float = Field(...)
    units: str = Field(default="ppm")
    precision: int = Field(default=2, ge=0, le=6)


class AnalyzerConfig(BaseModel):
    """Individual CEMS analyzer configuration."""
    analyzer_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    manufacturer: str = Field(default="")
    model: str = Field(default="")
    serial_number: str = Field(default="")
    span: MonitorSpan = Field(...)
    dual_span: bool = Field(default=False)
    auto_range: bool = Field(default=False)
    calibration_drift_limit_percent: float = Field(default=2.5, ge=0.1, le=10.0)
    daily_calibration_required: bool = Field(default=True)
    calibration_gases: Dict[str, float] = Field(default_factory=lambda: {"zero": 0.0, "mid": 50.0, "high": 100.0})
    linearity_limit_percent: float = Field(default=2.0, ge=0.1, le=5.0)
    rata_frequency_quarters: int = Field(default=4, ge=1, le=8)
    bias_adjustment_factor: float = Field(default=1.0, ge=0.8, le=1.2)


class CEMSConfig(BaseModel):
    """Continuous Emissions Monitoring System Configuration per EPA 40 CFR Part 75."""
    cems_id: str = Field(default="CEMS-001")
    facility_id: str = Field(default="FACILITY-001")
    unit_id: str = Field(default="UNIT-001")
    stack_id: str = Field(default="STACK-001")
    monitored_pollutants: List[Pollutant] = Field(default_factory=lambda: [Pollutant.NOX, Pollutant.SO2, Pollutant.CO2, Pollutant.CO, Pollutant.PM])
    diluent_monitor: Pollutant = Field(default=Pollutant.O2)
    flow_monitoring_enabled: bool = Field(default=True)
    moisture_monitoring_enabled: bool = Field(default=True)
    analyzers: List[AnalyzerConfig] = Field(default_factory=list)
    data_collection_frequency_seconds: int = Field(default=15, ge=1, le=60)
    averaging_period_minutes: int = Field(default=60, ge=1, le=60)
    minimum_data_availability_percent: float = Field(default=75.0, ge=50.0, le=100.0)
    daily_calibration_enabled: bool = Field(default=True)
    calibration_window_start_hour: int = Field(default=0, ge=0, le=23)
    calibration_window_end_hour: int = Field(default=6, ge=0, le=23)
    cylinder_gas_audit_frequency_quarters: int = Field(default=1, ge=1, le=4)
    substitute_data_enabled: bool = Field(default=True)
    maximum_missing_hours_before_substitution: int = Field(default=24, ge=1, le=720)
    monitor_data_availability_threshold: float = Field(default=95.0, ge=80.0, le=100.0)
    reference_method_correlation_required: bool = Field(default=True)
    correlation_coefficient_minimum: float = Field(default=0.90, ge=0.80, le=1.0)
    dahs_integration_enabled: bool = Field(default=True)
    dahs_protocol: str = Field(default="MODBUS_TCP")
    dahs_polling_interval_ms: int = Field(default=1000, ge=100, le=60000)

    class Config:
        use_enum_values = True


class PermitLimit(BaseModel):
    """Individual permit emission limit."""
    limit_id: str = Field(...)
    pollutant: Pollutant = Field(...)
    limit_value: float = Field(..., ge=0.0)
    units: str = Field(default="lb/MMBtu")
    averaging_period: AveragingPeriod = Field(default=AveragingPeriod.HOURLY)
    compliance_margin_percent: float = Field(default=10.0, ge=0.0, le=50.0)
    regulatory_citation: str = Field(default="")
    effective_date: Optional[datetime] = Field(default=None)
    expiration_date: Optional[datetime] = Field(default=None)

    class Config:
        use_enum_values = True


class ComplianceConfig(BaseModel):
    """Emissions Compliance Configuration per EPA regulations."""
    facility_name: str = Field(default="")
    permit_number: str = Field(default="")
    permit_type: PermitType = Field(default=PermitType.TITLE_V)
    permit_effective_date: Optional[datetime] = Field(default=None)
    permit_expiration_date: Optional[datetime] = Field(default=None)
    emission_limits: List[PermitLimit] = Field(default_factory=list)
    default_nox_limit_lb_mmbtu: float = Field(default=0.10, ge=0.01, le=1.0)
    default_so2_limit_lb_mmbtu: float = Field(default=0.50, ge=0.01, le=5.0)
    default_pm_limit_lb_mmbtu: float = Field(default=0.03, ge=0.001, le=0.5)
    default_co_limit_lb_mmbtu: float = Field(default=0.08, ge=0.01, le=0.5)
    opacity_limit_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    opacity_averaging_minutes: int = Field(default=6, ge=1, le=60)
    startup_exemption_enabled: bool = Field(default=True)
    startup_exemption_hours: float = Field(default=4.0, ge=0.5, le=24.0)
    shutdown_exemption_enabled: bool = Field(default=True)
    shutdown_exemption_hours: float = Field(default=2.0, ge=0.5, le=12.0)
    evaluation_frequency_minutes: int = Field(default=15, ge=1, le=60)
    exceedance_notification_threshold_percent: float = Field(default=90.0, ge=50.0, le=99.0)
    auto_deviation_report: bool = Field(default=True)
    deviation_report_delay_hours: float = Field(default=24.0, ge=1.0, le=168.0)
    quarterly_reporting_enabled: bool = Field(default=True)
    annual_emissions_inventory: bool = Field(default=True)
    ghg_reporting_threshold_mtco2e: float = Field(default=25000.0, ge=0.0, le=1000000.0)
    epa_electronic_reporting: bool = Field(default=True)
    applicable_standards: List[str] = Field(default_factory=lambda: ["40 CFR 60 Subpart Db", "40 CFR 63 Subpart DDDDD", "40 CFR 75"])

    class Config:
        use_enum_values = True


class RATAConfig(BaseModel):
    """RATA Configuration per EPA 40 CFR Part 75 Appendix A."""
    rata_frequency_quarters: int = Field(default=4, ge=1, le=8)
    abbreviated_rata_allowed: bool = Field(default=True)
    abbreviated_rata_threshold_percent: float = Field(default=7.5, ge=5.0, le=10.0)
    ra_limit_nox_percent: float = Field(default=10.0, ge=5.0, le=20.0)
    ra_limit_so2_percent: float = Field(default=10.0, ge=5.0, le=20.0)
    ra_limit_co2_percent: float = Field(default=10.0, ge=5.0, le=20.0)
    ra_limit_flow_percent: float = Field(default=10.0, ge=5.0, le=20.0)
    ra_limit_diluent_absolute: float = Field(default=1.0, ge=0.5, le=2.0)
    bias_test_required: bool = Field(default=True)
    bias_significance_level: float = Field(default=0.05, ge=0.01, le=0.10)
    bias_adjustment_threshold_percent: float = Field(default=5.0, ge=1.0, le=10.0)
    minimum_run_duration_minutes: int = Field(default=21, ge=15, le=60)
    standard_runs_required: int = Field(default=9, ge=9, le=12)
    abbreviated_runs_required: int = Field(default=3, ge=3, le=6)
    operating_load_stability_percent: float = Field(default=10.0, ge=5.0, le=20.0)
    multi_load_rata_required: bool = Field(default=False)
    load_levels: List[float] = Field(default_factory=lambda: [0.5, 0.75, 1.0])
    runs_per_load_level: int = Field(default=3, ge=3, le=6)
    reference_method_nox: str = Field(default="Method 7E")
    reference_method_so2: str = Field(default="Method 6C")
    reference_method_co2: str = Field(default="Method 3A")
    reference_method_flow: str = Field(default="Method 2")
    advance_notice_days: int = Field(default=21, ge=7, le=90)
    grace_period_days: int = Field(default=30, ge=0, le=90)
    auto_schedule_enabled: bool = Field(default=True)
    preferred_test_window_hours: Tuple[int, int] = Field(default=(6, 18))
    approved_test_contractors: List[str] = Field(default_factory=list)
    contractor_certification_required: bool = Field(default=True)
    test_protocol_required: bool = Field(default=True)
    real_time_data_sharing: bool = Field(default=True)
    auto_report_generation: bool = Field(default=True)


class FugitiveConfig(BaseModel):
    """Fugitive Emissions Detection Configuration."""
    detection_enabled: bool = Field(default=True)
    ml_model_type: str = Field(default="isolation_forest")
    anomaly_threshold_sigma: float = Field(default=3.0, ge=2.0, le=5.0)
    confidence_threshold_percent: float = Field(default=85.0, ge=50.0, le=99.0)
    leak_definition_ppm: float = Field(default=500.0, ge=100.0, le=10000.0)
    action_level_ppm: float = Field(default=10000.0, ge=1000.0, le=50000.0)
    background_level_ppm: float = Field(default=50.0, ge=0.0, le=500.0)
    source_thresholds: Dict[str, float] = Field(default_factory=lambda: {"valve": 500.0, "pump": 1000.0, "compressor": 500.0, "pressure_relief": 500.0, "connector": 500.0, "flange": 500.0, "open_ended_line": 500.0})
    ldar_program_enabled: bool = Field(default=True)
    ldar_monitoring_frequency_days: int = Field(default=90, ge=7, le=365)
    first_attempt_repair_days: int = Field(default=5, ge=1, le=15)
    final_repair_days: int = Field(default=15, ge=5, le=45)
    delay_of_repair_allowed: bool = Field(default=True)
    max_delay_of_repair_days: int = Field(default=120, ge=30, le=365)
    ogi_enabled: bool = Field(default=False)
    ogi_survey_frequency_days: int = Field(default=60, ge=7, le=365)
    ogi_camera_sensitivity_ppm_m: float = Field(default=10.0, ge=1.0, le=100.0)
    satellite_monitoring_enabled: bool = Field(default=False)
    aerial_survey_frequency_months: int = Field(default=6, ge=1, le=24)
    alert_escalation_enabled: bool = Field(default=True)
    immediate_alert_ppm: float = Field(default=10000.0, ge=1000.0, le=100000.0)
    escalation_delay_hours: int = Field(default=4, ge=1, le=24)
    shap_enabled: bool = Field(default=True)
    lime_enabled: bool = Field(default=True)
    feature_importance_threshold: float = Field(default=0.05, ge=0.01, le=0.3)


class TradingConfig(BaseModel):
    """Carbon Market Trading Configuration."""
    trading_enabled: bool = Field(default=True)
    auto_trade_enabled: bool = Field(default=False)
    advisory_mode: bool = Field(default=True)
    primary_market: CarbonMarket = Field(default=CarbonMarket.VOLUNTARY)
    secondary_markets: List[CarbonMarket] = Field(default_factory=list)
    market_data_providers: List[str] = Field(default_factory=lambda: ["ICE", "CME", "CBL"])
    price_feed_update_interval_seconds: int = Field(default=60, ge=10, le=3600)
    max_position_mtco2e: float = Field(default=100000.0, ge=0.0, le=10000000.0)
    max_daily_trade_mtco2e: float = Field(default=10000.0, ge=0.0, le=1000000.0)
    max_single_trade_mtco2e: float = Field(default=5000.0, ge=0.0, le=100000.0)
    position_limit_utilization_warning: float = Field(default=80.0, ge=50.0, le=99.0)
    max_price_exposure_usd: float = Field(default=1000000.0, ge=0.0)
    stop_loss_percent: float = Field(default=10.0, ge=1.0, le=30.0)
    var_limit_usd: float = Field(default=100000.0, ge=0.0)
    var_confidence_level: float = Field(default=0.95, ge=0.90, le=0.99)
    approval_required_above_usd: float = Field(default=10000.0, ge=0.0)
    approval_required_above_mtco2e: float = Field(default=1000.0, ge=0.0)
    approval_timeout_hours: int = Field(default=24, ge=1, le=168)
    approver_roles: List[str] = Field(default_factory=lambda: ["sustainability_manager", "cfo", "ceo"])
    internal_carbon_price_usd_mtco2e: float = Field(default=50.0, ge=0.0, le=1000.0)
    shadow_price_enabled: bool = Field(default=True)
    price_escalation_rate_percent: float = Field(default=5.0, ge=0.0, le=20.0)
    hedging_enabled: bool = Field(default=True)
    hedge_ratio: float = Field(default=0.8, ge=0.0, le=1.0)
    hedge_horizon_months: int = Field(default=12, ge=1, le=60)
    trade_reporting_enabled: bool = Field(default=True)
    daily_position_report: bool = Field(default=True)
    mtm_frequency: str = Field(default="daily")

    class Config:
        use_enum_values = True


class OffsetsConfig(BaseModel):
    """Carbon Offset Certificate Tracking Configuration."""
    tracking_enabled: bool = Field(default=True)
    auto_retirement_enabled: bool = Field(default=False)
    provenance_verification: bool = Field(default=True)
    accepted_standards: List[OffsetStandard] = Field(default_factory=lambda: [OffsetStandard.VERRA_VCS, OffsetStandard.GOLD_STANDARD, OffsetStandard.ACR, OffsetStandard.CAR])
    minimum_vintage_years: int = Field(default=5, ge=1, le=20)
    preferred_project_types: List[OffsetProjectType] = Field(default_factory=lambda: [OffsetProjectType.FORESTRY, OffsetProjectType.RENEWABLE_ENERGY, OffsetProjectType.METHANE_CAPTURE])
    additionality_verification: bool = Field(default=True)
    permanence_verification: bool = Field(default=True)
    minimum_buffer_pool_percent: float = Field(default=10.0, ge=0.0, le=30.0)
    third_party_verification_required: bool = Field(default=True)
    co_benefits_required: bool = Field(default=False)
    certificate_expiry_warning_days: int = Field(default=90, ge=30, le=365)
    auto_renewal_enabled: bool = Field(default=False)
    retirement_documentation: bool = Field(default=True)
    blockchain_verification_enabled: bool = Field(default=False)
    dlt_network: str = Field(default="")
    inventory_buffer_percent: float = Field(default=10.0, ge=0.0, le=50.0)
    reorder_point_mtco2e: float = Field(default=1000.0, ge=0.0)
    reorder_quantity_mtco2e: float = Field(default=5000.0, ge=0.0)
    quarterly_inventory_report: bool = Field(default=True)
    retirement_notifications: bool = Field(default=True)
    audit_trail_enabled: bool = Field(default=True)

    class Config:
        use_enum_values = True


class SecurityConfig(BaseModel):
    """OT-Safe Security Configuration."""
    default_access_level: SecurityAccessLevel = Field(default=SecurityAccessLevel.READ_ONLY)
    enforce_read_only: bool = Field(default=True)
    write_requires_approval: bool = Field(default=True)
    auth_enabled: bool = Field(default=True)
    auth_method: str = Field(default="jwt")
    jwt_secret_env_var: str = Field(default="GL010_JWT_SECRET")
    token_expiry_hours: int = Field(default=8, ge=1, le=24)
    refresh_token_enabled: bool = Field(default=True)
    mtls_enabled: bool = Field(default=True)
    client_cert_required: bool = Field(default=True)
    cert_path_env_var: str = Field(default="GL010_CERT_PATH")
    key_path_env_var: str = Field(default="GL010_KEY_PATH")
    ca_path_env_var: str = Field(default="GL010_CA_PATH")
    allowed_ip_ranges: List[str] = Field(default_factory=lambda: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"])
    blocked_ip_ranges: List[str] = Field(default_factory=list)
    rate_limit_rpm: int = Field(default=1000, ge=10, le=100000)
    rate_limit_burst: int = Field(default=100, ge=10, le=1000)
    encryption_at_rest: bool = Field(default=True)
    encryption_algorithm: str = Field(default="AES-256-GCM")
    encryption_key_env_var: str = Field(default="GL010_ENCRYPTION_KEY")
    data_masking_enabled: bool = Field(default=True)
    secrets_backend: str = Field(default="vault")
    vault_address_env_var: str = Field(default="VAULT_ADDR")
    vault_token_env_var: str = Field(default="VAULT_TOKEN")
    security_audit_enabled: bool = Field(default=True)
    failed_auth_lockout_attempts: int = Field(default=5, ge=3, le=10)
    lockout_duration_minutes: int = Field(default=30, ge=5, le=1440)
    scada_read_only: bool = Field(default=True)
    dahs_read_only: bool = Field(default=True)
    control_system_isolation: bool = Field(default=True)
    safety_system_access_prohibited: bool = Field(default=True)
    soc2_compliance: bool = Field(default=True)
    gdpr_compliance: bool = Field(default=False)
    hipaa_compliance: bool = Field(default=False)

    class Config:
        use_enum_values = True


class APIConfig(BaseModel):
    """API server configuration."""
    rest_enabled: bool = Field(default=True)
    rest_port: int = Field(default=8010, ge=1024, le=65535)
    graphql_enabled: bool = Field(default=True)
    graphql_port: int = Field(default=8011, ge=1024, le=65535)
    grpc_enabled: bool = Field(default=True)
    grpc_port: int = Field(default=50010, ge=1024, le=65535)
    cors_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_docs_enabled: bool = Field(default=True)
    openapi_version: str = Field(default="3.1.0")


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""
    enabled: bool = Field(default=True)
    prefix: str = Field(default="greenlang_emissionsguardian")
    port: int = Field(default=9010, ge=1024, le=65535)
    push_gateway_url: Optional[str] = Field(default=None)
    collection_interval_seconds: float = Field(default=15.0, ge=1.0, le=300.0)
    histogram_buckets: List[float] = Field(default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])


class EmissionsGuardianConfig(BaseModel):
    """Complete GL-010 EmissionsGuardian Configuration."""
    agent: AgentConfig = Field(default_factory=AgentConfig)
    cems: CEMSConfig = Field(default_factory=CEMSConfig)
    compliance: ComplianceConfig = Field(default_factory=ComplianceConfig)
    rata: RATAConfig = Field(default_factory=RATAConfig)
    fugitive: FugitiveConfig = Field(default_factory=FugitiveConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    offsets: OffsetsConfig = Field(default_factory=OffsetsConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    class Config:
        use_enum_values = True

    @classmethod
    def from_environment(cls) -> "EmissionsGuardianConfig":
        """Create configuration from environment variables."""
        config_dict = {}
        prefix = "GL_EMISSIONSGUARDIAN_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("_")
                current = config_dict
                for part in config_path[:-1]:
                    current = current.setdefault(part, {})
                current[config_path[-1]] = value
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EmissionsGuardianConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def validate_zero_hallucination(self) -> List[str]:
        """Validate configuration supports zero-hallucination principles."""
        warnings = []
        if not self.agent.deterministic_mode:
            warnings.append("deterministic_mode disabled - calculations may not be reproducible")
        if not self.agent.provenance_tracking:
            warnings.append("provenance_tracking disabled - audit trail incomplete")
        if self.trading.auto_trade_enabled and not self.trading.advisory_mode:
            warnings.append("auto_trade_enabled without advisory_mode - requires additional controls")
        if not self.security.enforce_read_only:
            warnings.append("enforce_read_only disabled - OT write access possible")
        return warnings
