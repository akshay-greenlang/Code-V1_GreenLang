# -*- coding: utf-8 -*-
"""
GL-004 BurnerOptimizationAgent - Configuration Management

Configuration settings for burner optimization agent using Pydantic BaseSettings.
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
    APP_NAME: str = Field("GL-004-BurnerOptimizationAgent", description="Application name")
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
    # AI Model Configuration (if needed for advanced analytics)
    # ============================================================================
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    ANTHROPIC_API_KEY: Optional[str] = Field(None, description="Anthropic API key")

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
    # Burner Controller Configuration
    # ============================================================================
    BURNER_CONTROLLER_HOST: str = Field("localhost", description="Burner controller host")
    BURNER_CONTROLLER_PORT: int = Field(502, description="Burner controller port")
    BURNER_CONTROLLER_PROTOCOL: str = Field("modbus_tcp", description="Protocol: modbus_tcp, opcua, profibus")
    BURNER_CONTROLLER_TYPE: str = Field("honeywell_burnerlogix", description="Controller type")

    # ============================================================================
    # Sensor Configuration
    # ============================================================================
    # O2 Analyzer
    O2_ANALYZER_HOST: str = Field("localhost", description="O2 analyzer host")
    O2_ANALYZER_PORT: int = Field(502, description="O2 analyzer port")
    O2_ANALYZER_TYPE: str = Field("zirconia", description="Analyzer type: zirconia, paramagnetic")

    # Emissions Monitor
    EMISSIONS_MONITOR_HOST: str = Field("localhost", description="CEMS host")
    EMISSIONS_MONITOR_PORT: int = Field(502, description="CEMS port")
    EMISSIONS_MONITOR_PROTOCOL: str = Field("modbus_tcp", description="CEMS protocol")

    # Flame Scanner
    FLAME_SCANNER_HOST: str = Field("localhost", description="Flame scanner host")
    FLAME_SCANNER_PORT: int = Field(502, description="Flame scanner port")
    FLAME_SCANNER_TYPE: str = Field("uv", description="Scanner type: uv, ir, flame_rod")

    # Temperature Sensors
    TEMPERATURE_SENSORS: List[Dict[str, str]] = Field(
        default_factory=lambda: [
            {"name": "flame_temp", "type": "thermocouple_r", "location": "flame_zone"},
            {"name": "furnace_temp", "type": "thermocouple_k", "location": "furnace"},
            {"name": "flue_gas_temp", "type": "thermocouple_k", "location": "flue_gas_outlet"}
        ],
        description="Temperature sensor configuration"
    )

    # ============================================================================
    # SCADA/DCS Integration
    # ============================================================================
    SCADA_OPC_UA_ENDPOINT: str = Field("opc.tcp://localhost:4840", description="OPC UA endpoint")
    MQTT_BROKER_URL: str = Field("mqtt://localhost:1883", description="MQTT broker URL")
    MQTT_TOPIC_PREFIX: str = Field("greenlang/gl-004", description="MQTT topic prefix")

    # ============================================================================
    # Fuel Configuration
    # ============================================================================
    FUEL_TYPE: str = Field("natural_gas", description="Fuel type: natural_gas, fuel_oil, coal, biomass")
    FUEL_COMPOSITION: Dict[str, float] = Field(
        default_factory=lambda: {
            "C": 75.0,  # Carbon %
            "H": 25.0,  # Hydrogen %
            "O": 0.0,   # Oxygen %
            "N": 0.0,   # Nitrogen %
            "S": 0.0,   # Sulfur %
            "ash": 0.0, # Ash %
            "moisture": 0.0  # Moisture %
        },
        description="Fuel composition by weight %"
    )
    FUEL_LHV_MJ_PER_KG: float = Field(50.0, description="Lower heating value (MJ/kg)")
    FUEL_HHV_MJ_PER_KG: float = Field(55.5, description="Higher heating value (MJ/kg)")
    FUEL_COST_PER_KG: float = Field(0.30, description="Fuel cost ($/kg)")

    # ============================================================================
    # Burner Operating Parameters
    # ============================================================================
    BURNER_MAX_CAPACITY: float = Field(10000.0, description="Maximum burner capacity (kW or BTU/hr)")
    MIN_FUEL_FLOW: float = Field(50.0, description="Minimum fuel flow (kg/hr or m3/hr)")
    MAX_FUEL_FLOW: float = Field(1000.0, description="Maximum fuel flow (kg/hr or m3/hr)")
    MIN_AIR_FLOW: float = Field(500.0, description="Minimum air flow (m3/hr)")
    MAX_AIR_FLOW: float = Field(15000.0, description="Maximum air flow (m3/hr)")
    BURNER_TURNDOWN_RATIO: float = Field(10.0, description="Burner turndown ratio")

    # ============================================================================
    # Combustion Optimization Parameters
    # ============================================================================
    # Efficiency targets
    TARGET_EFFICIENCY_PERCENT: float = Field(90.0, description="Target combustion efficiency (%)")
    MIN_ACCEPTABLE_EFFICIENCY: float = Field(80.0, description="Minimum acceptable efficiency (%)")

    # Excess air limits
    MIN_EXCESS_AIR_PERCENT: float = Field(5.0, description="Minimum excess air (% - safety margin)")
    MAX_EXCESS_AIR_PERCENT: float = Field(30.0, description="Maximum excess air (% - efficiency limit)")
    OPTIMAL_EXCESS_AIR_PERCENT: float = Field(15.0, description="Target excess air (%)")

    # O2 limits
    MIN_O2_PERCENT: float = Field(2.0, description="Minimum O2 (% - safety limit)")
    MAX_O2_PERCENT: float = Field(6.0, description="Maximum O2 (% - efficiency limit)")
    TARGET_O2_PERCENT: float = Field(3.0, description="Target O2 (%)")

    # ============================================================================
    # Emissions Limits
    # ============================================================================
    # EPA limits (example values - adjust based on regulations)
    MAX_NOX_PPM: float = Field(50.0, description="Maximum NOx (ppm @ 3% O2)")
    MAX_CO_PPM: float = Field(100.0, description="Maximum CO (ppm @ 3% O2)")
    MAX_SO2_PPM: float = Field(20.0, description="Maximum SO2 (ppm @ 3% O2)")
    MAX_PARTICULATES_MG_PER_NM3: float = Field(20.0, description="Maximum particulates (mg/Nm³)")

    # EU IED limits (example values)
    EU_IED_MAX_NOX_MG_PER_NM3: float = Field(100.0, description="EU IED NOx limit (mg/Nm³)")
    EU_IED_MAX_CO_MG_PER_NM3: float = Field(100.0, description="EU IED CO limit (mg/Nm³)")

    # ============================================================================
    # Safety Limits
    # ============================================================================
    MAX_FLAME_TEMPERATURE_C: float = Field(1800.0, description="Maximum flame temperature (°C)")
    MAX_FURNACE_TEMPERATURE_C: float = Field(1400.0, description="Maximum furnace temperature (°C)")
    MAX_FLUE_GAS_TEMPERATURE_C: float = Field(400.0, description="Maximum flue gas temperature (°C)")
    MIN_FUEL_PRESSURE_KPA: float = Field(50.0, description="Minimum fuel pressure (kPa)")
    MAX_FUEL_PRESSURE_KPA: float = Field(500.0, description="Maximum fuel pressure (kPa)")
    MIN_AIR_PRESSURE_KPA: float = Field(5.0, description="Minimum air pressure (kPa)")
    MAX_AIR_PRESSURE_KPA: float = Field(50.0, description="Maximum air pressure (kPa)")

    # ============================================================================
    # Optimization Algorithm Parameters
    # ============================================================================
    OPTIMIZATION_METHOD: str = Field("particle_swarm", description="Method: gradient_descent, particle_swarm, genetic")
    MAX_OPTIMIZATION_ITERATIONS: int = Field(100, description="Maximum iterations")
    CONVERGENCE_TOLERANCE: float = Field(0.001, description="Convergence tolerance")
    OPTIMIZATION_OBJECTIVES: List[str] = Field(
        default_factory=lambda: ["maximize_efficiency", "minimize_nox", "minimize_co"],
        description="Optimization objectives"
    )
    OBJECTIVE_WEIGHTS: Dict[str, float] = Field(
        default_factory=lambda: {
            "efficiency": 0.5,
            "nox": 0.3,
            "co": 0.2
        },
        description="Objective weight distribution"
    )

    # ============================================================================
    # Control Loop Parameters
    # ============================================================================
    # PID tuning for fuel control
    FUEL_CONTROL_KP: float = Field(1.0, description="Fuel control proportional gain")
    FUEL_CONTROL_KI: float = Field(0.1, description="Fuel control integral gain")
    FUEL_CONTROL_KD: float = Field(0.05, description="Fuel control derivative gain")

    # PID tuning for air control
    AIR_CONTROL_KP: float = Field(1.0, description="Air control proportional gain")
    AIR_CONTROL_KI: float = Field(0.1, description="Air control integral gain")
    AIR_CONTROL_KD: float = Field(0.05, description="Air control derivative gain")

    # O2 trim control
    O2_TRIM_KP: float = Field(0.5, description="O2 trim proportional gain")
    O2_TRIM_KI: float = Field(0.05, description="O2 trim integral gain")
    O2_TRIM_KD: float = Field(0.02, description="O2 trim derivative gain")

    # ============================================================================
    # Timing Parameters
    # ============================================================================
    OPTIMIZATION_INTERVAL_SECONDS: int = Field(300, description="Time between optimization cycles (5 min)")
    DATA_COLLECTION_INTERVAL_SECONDS: int = Field(10, description="Sensor data collection interval")
    SETPOINT_CHANGE_DELAY_SECONDS: float = Field(2.0, description="Delay between setpoint changes")
    VALIDATION_DURATION_SECONDS: int = Field(300, description="Optimization validation period (5 min)")
    ERROR_RETRY_DELAY_SECONDS: int = Field(60, description="Delay before retrying after error")

    # ============================================================================
    # Validation Tolerances
    # ============================================================================
    VALIDATION_EFFICIENCY_TOLERANCE: float = Field(0.5, description="Efficiency validation tolerance (%)")
    VALIDATION_NOX_TOLERANCE: float = Field(5.0, description="NOx validation tolerance (ppm)")
    VALIDATION_CO_TOLERANCE: float = Field(10.0, description="CO validation tolerance (ppm)")

    # ============================================================================
    # Environmental Parameters
    # ============================================================================
    AMBIENT_TEMPERATURE: float = Field(25.0, description="Ambient air temperature (°C)")
    AMBIENT_PRESSURE: float = Field(101.325, description="Ambient pressure (kPa)")
    AMBIENT_HUMIDITY: float = Field(60.0, description="Relative humidity (%)")

    # ============================================================================
    # Performance & Scaling
    # ============================================================================
    WORKER_COUNT: int = Field(4, description="Number of worker processes")
    MAX_CONNECTIONS: int = Field(1000, description="Maximum concurrent connections")
    TIMEOUT_SECONDS: int = Field(30, description="Request timeout")
    RATE_LIMIT_PER_MINUTE: int = Field(100, description="API rate limit")

    # ============================================================================
    # Feature Flags
    # ============================================================================
    ENABLE_PROFILING: bool = Field(False, description="Enable performance profiling")
    ENABLE_ADVANCED_ANALYTICS: bool = Field(False, description="Enable ML-based analytics")
    ENABLE_PREDICTIVE_MAINTENANCE: bool = Field(False, description="Enable predictive maintenance")

    # ============================================================================
    # Deployment Information
    # ============================================================================
    POD_NAME: Optional[str] = Field(None, description="Kubernetes pod name")
    POD_NAMESPACE: Optional[str] = Field(None, description="Kubernetes namespace")
    NODE_NAME: Optional[str] = Field(None, description="Kubernetes node name")

    @validator('FUEL_COMPOSITION')
    def validate_fuel_composition(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate fuel composition sums to ~100%"""
        total = sum(v.values())
        if not 99.0 <= total <= 101.0:
            raise ValueError(f"Fuel composition must sum to ~100%, got {total:.1f}%")
        return v

    @validator('OBJECTIVE_WEIGHTS')
    def validate_objective_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate objective weights sum to 1.0"""
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Objective weights must sum to 1.0, got {total:.3f}")
        return v

    @validator('GREENLANG_ENV')
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting"""
        valid_envs = ['development', 'staging', 'production']
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.GREENLANG_ENV == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.GREENLANG_ENV == "development"


# Global settings instance
settings = Settings()
