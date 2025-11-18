"""
GL-005 CombustionControlAgent - Configuration Management

Configuration settings for combustion control agent using Pydantic BaseSettings.
Supports environment variables and .env file loading.
"""

from typing import Dict, List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ============================================================================
    # Application Configuration
    # ============================================================================
    GREENLANG_ENV: str = Field("development", description="Environment: development, staging, production")
    APP_NAME: str = Field("GL-005-CombustionControlAgent", description="Application name")
    APP_VERSION: str = Field("1.0.0", description="Application version")
    LOG_LEVEL: str = Field("INFO", description="Logging level")
    DEBUG: bool = Field(False, description="Debug mode")

    # ============================================================================
    # Database Configuration
    # ============================================================================
    DATABASE_URL: str = Field(
        "postgresql+asyncpg://user:password@localhost:5432/greenlang",
        description="PostgreSQL connection string"
    )
    DB_POOL_SIZE: int = Field(10, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(20, description="Max overflow connections")
    DB_POOL_TIMEOUT: int = Field(30, description="Pool timeout (seconds)")

    # ============================================================================
    # Cache Configuration
    # ============================================================================
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis connection string")
    REDIS_POOL_SIZE: int = Field(10, description="Redis connection pool size")
    REDIS_TIMEOUT: int = Field(5, description="Redis timeout (seconds)")
    CACHE_TTL: int = Field(3600, description="Cache TTL (seconds)")

    # ============================================================================
    # Security Configuration
    # ============================================================================
    JWT_SECRET: str = Field("change-this-secret-key", description="JWT secret key")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRATION_HOURS: int = Field(24, description="JWT expiration hours")
    API_KEY: Optional[str] = Field(None, description="API key for authentication")

    # ============================================================================
    # Monitoring & Observability
    # ============================================================================
    METRICS_ENABLED: bool = Field(True, description="Enable Prometheus metrics")
    PROMETHEUS_PORT: int = Field(8001, description="Prometheus metrics port")
    TRACING_ENABLED: bool = Field(True, description="Enable distributed tracing")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field("http://localhost:4317", description="OTLP endpoint")
    LOG_FORMAT: str = Field("json", description="Log format: json or text")

    # ============================================================================
    # DCS (Distributed Control System) Configuration
    # ============================================================================
    DCS_HOST: str = Field("localhost", description="DCS host address")
    DCS_PORT: int = Field(502, description="DCS port")
    DCS_PROTOCOL: str = Field("modbus_tcp", description="DCS protocol: modbus_tcp, opcua, profinet")
    DCS_TIMEOUT_MS: int = Field(100, description="DCS communication timeout (ms)")
    DCS_RETRY_ATTEMPTS: int = Field(3, description="Number of retry attempts")
    DCS_RETRY_DELAY_MS: int = Field(50, description="Delay between retries (ms)")

    # ============================================================================
    # PLC (Programmable Logic Controller) Configuration
    # ============================================================================
    PLC_HOST: str = Field("localhost", description="PLC host address")
    PLC_PORT: int = Field(502, description="PLC port")
    PLC_MODBUS_ID: int = Field(1, description="Modbus slave ID")
    PLC_PROTOCOL: str = Field("modbus_tcp", description="PLC protocol")
    PLC_TIMEOUT_MS: int = Field(100, description="PLC timeout (ms)")
    PLC_BACKUP_ENABLED: bool = Field(True, description="Use PLC as backup control")

    # ============================================================================
    # Combustion Analyzer Configuration
    # ============================================================================
    COMBUSTION_ANALYZER_ENDPOINTS: List[str] = Field(
        default_factory=lambda: ["http://localhost:8080/api/v1/measurements"],
        description="Combustion analyzer API endpoints"
    )
    ANALYZER_TIMEOUT_MS: int = Field(200, description="Analyzer timeout (ms)")
    ANALYZER_POLL_RATE_HZ: int = Field(10, description="Analyzer polling rate (Hz)")

    # ============================================================================
    # Pressure Sensor Configuration
    # ============================================================================
    PRESSURE_SENSORS: List[Dict[str, str]] = Field(
        default_factory=lambda: [
            {"name": "fuel", "type": "differential", "range_kpa": "0-1000", "address": "40001"},
            {"name": "air", "type": "differential", "range_kpa": "0-100", "address": "40002"},
            {"name": "furnace", "type": "draft", "range_pa": "-500-500", "address": "40003"}
        ],
        description="Pressure sensor configuration"
    )

    # ============================================================================
    # Temperature Sensor Configuration
    # ============================================================================
    TEMPERATURE_SENSORS: List[Dict[str, str]] = Field(
        default_factory=lambda: [
            {"name": "flame", "type": "thermocouple_r", "range_c": "0-1800", "address": "40010"},
            {"name": "furnace", "type": "thermocouple_k", "range_c": "0-1400", "address": "40011"},
            {"name": "flue_gas", "type": "thermocouple_k", "range_c": "0-600", "address": "40012"},
            {"name": "ambient", "type": "rtd_pt100", "range_c": "-20-60", "address": "40013"}
        ],
        description="Temperature sensor configuration"
    )

    # ============================================================================
    # Flow Meter Configuration
    # ============================================================================
    FUEL_FLOW_METER: Dict[str, str] = Field(
        default_factory=lambda: {
            "type": "coriolis",
            "range": "0-2000",
            "units": "kg/hr",
            "address": "40020"
        },
        description="Fuel flow meter config"
    )

    AIR_FLOW_METER: Dict[str, str] = Field(
        default_factory=lambda: {
            "type": "vortex",
            "range": "0-20000",
            "units": "m3/hr",
            "address": "40021"
        },
        description="Air flow meter config"
    )

    # ============================================================================
    # SCADA Integration
    # ============================================================================
    SCADA_OPC_UA_ENDPOINT: str = Field("opc.tcp://localhost:4840", description="OPC UA endpoint")
    MQTT_BROKER_URL: str = Field("mqtt://localhost:1883", description="MQTT broker URL")
    MQTT_TOPIC_PREFIX: str = Field("greenlang/gl-005", description="MQTT topic prefix")
    MQTT_QOS: int = Field(1, description="MQTT QoS level")

    # ============================================================================
    # Fuel Configuration
    # ============================================================================
    FUEL_TYPE: str = Field("natural_gas", description="Fuel type: natural_gas, fuel_oil, coal, biomass, lng")
    FUEL_COMPOSITION: Dict[str, float] = Field(
        default_factory=lambda: {
            "C": 75.0,   # Carbon %
            "H": 25.0,   # Hydrogen %
            "O": 0.0,    # Oxygen %
            "N": 0.0,    # Nitrogen %
            "S": 0.0,    # Sulfur %
            "ash": 0.0,  # Ash %
            "moisture": 0.0  # Moisture %
        },
        description="Fuel composition by weight %"
    )
    FUEL_LHV_MJ_PER_KG: float = Field(50.0, description="Lower heating value (MJ/kg)")
    FUEL_HHV_MJ_PER_KG: float = Field(55.5, description="Higher heating value (MJ/kg)")
    FUEL_DENSITY_KG_PER_M3: float = Field(0.717, description="Fuel density (kg/m³ at STP for natural gas)")
    FUEL_COST_PER_KG: float = Field(0.30, description="Fuel cost ($/kg)")

    # ============================================================================
    # Operating Limits - Fuel Flow
    # ============================================================================
    MIN_FUEL_FLOW: float = Field(100.0, description="Minimum fuel flow (kg/hr or m3/hr)")
    MAX_FUEL_FLOW: float = Field(2000.0, description="Maximum fuel flow (kg/hr or m3/hr)")
    FUEL_FLOW_NORMAL_OPERATING: float = Field(1000.0, description="Normal operating fuel flow")

    # ============================================================================
    # Operating Limits - Air Flow
    # ============================================================================
    MIN_AIR_FLOW: float = Field(1000.0, description="Minimum air flow (m3/hr)")
    MAX_AIR_FLOW: float = Field(25000.0, description="Maximum air flow (m3/hr)")
    AIR_FLOW_NORMAL_OPERATING: float = Field(12500.0, description="Normal operating air flow")

    # ============================================================================
    # Temperature Limits
    # ============================================================================
    MAX_TEMPERATURE: float = Field(1600.0, description="Maximum safe temperature (°C)")
    MAX_FLAME_TEMPERATURE_C: float = Field(1800.0, description="Maximum flame temperature (°C)")
    MAX_FURNACE_TEMPERATURE_C: float = Field(1400.0, description="Maximum furnace temperature (°C)")
    MAX_FLUE_GAS_TEMPERATURE_C: float = Field(500.0, description="Maximum flue gas temperature (°C)")
    MIN_FURNACE_TEMPERATURE_C: float = Field(200.0, description="Minimum furnace temperature for stable combustion (°C)")

    # ============================================================================
    # Pressure Limits
    # ============================================================================
    MAX_PRESSURE: float = Field(900.0, description="Maximum safe pressure (kPa)")
    MIN_FUEL_PRESSURE_KPA: float = Field(80.0, description="Minimum fuel pressure (kPa)")
    MAX_FUEL_PRESSURE_KPA: float = Field(900.0, description="Maximum fuel pressure (kPa)")
    MIN_AIR_PRESSURE_KPA: float = Field(8.0, description="Minimum air pressure (kPa)")
    MAX_AIR_PRESSURE_KPA: float = Field(80.0, description="Maximum air pressure (kPa)")
    MAX_FURNACE_DRAFT_PRESSURE_PA: float = Field(50.0, description="Maximum furnace draft (Pa)")
    MIN_FURNACE_DRAFT_PRESSURE_PA: float = Field(-150.0, description="Minimum furnace draft (Pa)")

    # ============================================================================
    # Heat Output Configuration
    # ============================================================================
    HEAT_OUTPUT_TARGET_KW: float = Field(10000.0, description="Target heat output (kW)")
    HEAT_OUTPUT_MIN_KW: float = Field(2000.0, description="Minimum heat output (kW)")
    HEAT_OUTPUT_MAX_KW: float = Field(20000.0, description="Maximum heat output (kW)")
    HEAT_OUTPUT_TOLERANCE_PERCENT: float = Field(2.0, description="Acceptable heat output tolerance (%)")

    # ============================================================================
    # Combustion Control Parameters
    # ============================================================================
    # Excess air targets
    OPTIMAL_EXCESS_AIR_PERCENT: float = Field(15.0, description="Target excess air (%)")
    MIN_EXCESS_AIR_PERCENT: float = Field(5.0, description="Minimum excess air (safety margin) (%)")
    MAX_EXCESS_AIR_PERCENT: float = Field(35.0, description="Maximum excess air (efficiency limit) (%)")

    # O2 targets
    TARGET_O2_PERCENT: float = Field(3.0, description="Target O2 in flue gas (%)")
    MIN_O2_PERCENT: float = Field(1.5, description="Minimum O2 (safety limit) (%)")
    MAX_O2_PERCENT: float = Field(7.0, description="Maximum O2 (efficiency limit) (%)")

    # Efficiency targets
    TARGET_EFFICIENCY_PERCENT: float = Field(88.0, description="Target combustion efficiency (%)")
    MIN_EFFICIENCY_PERCENT: float = Field(75.0, description="Minimum acceptable efficiency (%)")

    # ============================================================================
    # Emissions Limits
    # ============================================================================
    MAX_NOX_PPM: float = Field(80.0, description="Maximum NOx (ppm @ 3% O2)")
    MAX_CO_PPM: float = Field(150.0, description="Maximum CO (ppm @ 3% O2)")
    MAX_SO2_PPM: float = Field(30.0, description="Maximum SO2 (ppm @ 3% O2)")

    # ============================================================================
    # Control Loop Timing
    # ============================================================================
    CONTROL_LOOP_INTERVAL_MS: int = Field(100, description="Control loop cycle time (ms) - Target <100ms")
    DATA_ACQUISITION_INTERVAL_MS: int = Field(50, description="Data read interval (ms)")
    SAFETY_CHECK_INTERVAL_MS: int = Field(50, description="Safety interlock check interval (ms)")
    O2_TRIM_INTERVAL_MS: int = Field(1000, description="O2 trim update interval (ms)")
    ERROR_RETRY_DELAY_MS: int = Field(1000, description="Error retry delay (ms)")

    # ============================================================================
    # PID Control Parameters - Fuel Flow
    # ============================================================================
    FUEL_CONTROL_KP: float = Field(2.0, description="Fuel PID proportional gain")
    FUEL_CONTROL_KI: float = Field(0.5, description="Fuel PID integral gain")
    FUEL_CONTROL_KD: float = Field(0.1, description="Fuel PID derivative gain")
    FUEL_CONTROL_AUTO: bool = Field(True, description="Fuel control in auto mode")

    # ============================================================================
    # PID Control Parameters - Air Flow
    # ============================================================================
    AIR_CONTROL_KP: float = Field(1.5, description="Air PID proportional gain")
    AIR_CONTROL_KI: float = Field(0.3, description="Air PID integral gain")
    AIR_CONTROL_KD: float = Field(0.08, description="Air PID derivative gain")
    AIR_CONTROL_AUTO: bool = Field(True, description="Air control in auto mode")

    # ============================================================================
    # PID Control Parameters - O2 Trim
    # ============================================================================
    O2_TRIM_KP: float = Field(100.0, description="O2 trim proportional gain (m3/hr per % O2)")
    O2_TRIM_KI: float = Field(20.0, description="O2 trim integral gain")
    O2_TRIM_KD: float = Field(5.0, description="O2 trim derivative gain")
    O2_TRIM_ENABLED: bool = Field(True, description="Enable O2 trim control")
    O2_TRIM_MAX_ADJUSTMENT: float = Field(500.0, description="Maximum O2 trim correction (m3/hr)")

    # ============================================================================
    # Feedforward Control
    # ============================================================================
    FEEDFORWARD_ENABLED: bool = Field(True, description="Enable feedforward control")
    FEEDFORWARD_GAIN: float = Field(0.8, description="Feedforward gain (0-1)")

    # ============================================================================
    # Stability Analysis Parameters
    # ============================================================================
    STABILITY_WINDOW_SIZE: int = Field(60, description="Stability analysis window (samples)")
    STABILITY_MIN_SAMPLES: int = Field(20, description="Minimum samples for stability analysis")
    TEMPERATURE_STABILITY_TOLERANCE_C: float = Field(10.0, description="Temperature stability tolerance (°C)")
    O2_STABILITY_TOLERANCE_PERCENT: float = Field(0.5, description="O2 stability tolerance (%)")

    # ============================================================================
    # Control Performance Monitoring
    # ============================================================================
    CONTROL_HISTORY_SIZE: int = Field(1000, description="Number of control actions to keep in history")
    STATE_HISTORY_SIZE: int = Field(1000, description="Number of states to keep in history")
    PERFORMANCE_MONITORING_ENABLED: bool = Field(True, description="Enable performance monitoring")

    # ============================================================================
    # Safety Configuration
    # ============================================================================
    SAFETY_INTERLOCKS_ENABLED: bool = Field(True, description="Enable safety interlocks")
    FLAME_DETECTION_REQUIRED: bool = Field(True, description="Require flame detection")
    PURGE_TIME_SECONDS: int = Field(60, description="Pre-purge time (seconds)")
    EMERGENCY_SHUTDOWN_ENABLED: bool = Field(True, description="Enable emergency shutdown")

    # ============================================================================
    # Ramp Rate Limits (for smooth control changes)
    # ============================================================================
    FUEL_FLOW_MAX_RAMP_RATE_PERCENT_PER_SEC: float = Field(
        10.0, description="Max fuel flow ramp rate (%/s)"
    )
    AIR_FLOW_MAX_RAMP_RATE_PERCENT_PER_SEC: float = Field(
        15.0, description="Max air flow ramp rate (%/s)"
    )

    # ============================================================================
    # Control Auto-Start
    # ============================================================================
    CONTROL_AUTO_START: bool = Field(False, description="Automatically start control on agent startup")

    # ============================================================================
    # Environmental Parameters
    # ============================================================================
    AMBIENT_TEMPERATURE_DEFAULT: float = Field(25.0, description="Default ambient temperature (°C)")
    AMBIENT_PRESSURE_KPA: float = Field(101.325, description="Ambient pressure (kPa)")
    AMBIENT_HUMIDITY_PERCENT: float = Field(60.0, description="Relative humidity (%)")

    # ============================================================================
    # Performance & Scaling
    # ============================================================================
    WORKER_COUNT: int = Field(4, description="Number of worker processes")
    MAX_CONNECTIONS: int = Field(1000, description="Maximum concurrent connections")
    TIMEOUT_SECONDS: int = Field(30, description="Request timeout")
    RATE_LIMIT_PER_MINUTE: int = Field(200, description="API rate limit")

    # ============================================================================
    # Feature Flags
    # ============================================================================
    ENABLE_PROFILING: bool = Field(False, description="Enable performance profiling")
    ENABLE_ADVANCED_ANALYTICS: bool = Field(False, description="Enable ML-based analytics")
    ENABLE_PREDICTIVE_MAINTENANCE: bool = Field(False, description="Enable predictive maintenance")
    ENABLE_ANOMALY_DETECTION: bool = Field(True, description="Enable combustion anomaly detection")

    # ============================================================================
    # Deployment Information
    # ============================================================================
    POD_NAME: Optional[str] = Field(None, description="Kubernetes pod name")
    POD_NAMESPACE: Optional[str] = Field(None, description="Kubernetes namespace")
    NODE_NAME: Optional[str] = Field(None, description="Kubernetes node name")
    DEPLOYMENT_TIMESTAMP: Optional[str] = Field(None, description="Deployment timestamp")

    @validator('FUEL_COMPOSITION')
    def validate_fuel_composition(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate fuel composition sums to ~100%"""
        total = sum(v.values())
        if not 99.0 <= total <= 101.0:
            raise ValueError(f"Fuel composition must sum to ~100%, got {total:.1f}%")
        return v

    @validator('GREENLANG_ENV')
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting"""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

    @validator('CONTROL_LOOP_INTERVAL_MS')
    def validate_control_loop_interval(cls, v: int) -> int:
        """Validate control loop interval is reasonable"""
        if v < 10:
            raise ValueError("Control loop interval must be >= 10ms")
        if v > 10000:
            raise ValueError("Control loop interval must be <= 10000ms")
        return v

    @validator('FUEL_TYPE')
    def validate_fuel_type(cls, v: str) -> str:
        """Validate fuel type"""
        valid_types = ['natural_gas', 'fuel_oil', 'coal', 'biomass', 'propane', 'lng']
        if v not in valid_types:
            raise ValueError(f"Fuel type must be one of {valid_types}")
        return v

    @validator('MIN_FUEL_FLOW', 'MIN_AIR_FLOW')
    def validate_min_flows(cls, v: float) -> float:
        """Validate minimum flows are positive"""
        if v <= 0:
            raise ValueError("Minimum flow must be positive")
        return v

    @validator('MAX_FUEL_FLOW')
    def validate_max_fuel_flow(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate max fuel flow > min fuel flow"""
        if 'MIN_FUEL_FLOW' in values and v <= values['MIN_FUEL_FLOW']:
            raise ValueError("MAX_FUEL_FLOW must be greater than MIN_FUEL_FLOW")
        return v

    @validator('MAX_AIR_FLOW')
    def validate_max_air_flow(cls, v: float, values: Dict[str, Any]) -> float:
        """Validate max air flow > min air flow"""
        if 'MIN_AIR_FLOW' in values and v <= values['MIN_AIR_FLOW']:
            raise ValueError("MAX_AIR_FLOW must be greater than MIN_AIR_FLOW")
        return v

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.GREENLANG_ENV == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.GREENLANG_ENV == "development"

    def get_control_loop_frequency_hz(self) -> float:
        """Get control loop frequency in Hz"""
        return 1000.0 / self.CONTROL_LOOP_INTERVAL_MS

    def get_fuel_flow_range(self) -> Tuple[float, float]:
        """Get fuel flow operating range"""
        return (self.MIN_FUEL_FLOW, self.MAX_FUEL_FLOW)

    def get_air_flow_range(self) -> Tuple[float, float]:
        """Get air flow operating range"""
        return (self.MIN_AIR_FLOW, self.MAX_AIR_FLOW)


# Add Tuple to imports for the type hint
from typing import Tuple

# Global settings instance
settings = Settings()
