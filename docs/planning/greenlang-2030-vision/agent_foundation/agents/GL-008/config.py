# -*- coding: utf-8 -*-
"""
Configuration module for GL-008 SteamTrapInspector.

This module defines configuration classes and settings for the TRAPCATCHER agent,
including operational parameters, monitoring thresholds, ML model configurations,
and integration settings.

SECURITY:
- Zero hardcoded credentials policy
- All secrets loaded from environment variables
- Validation enforced at startup via security_validator.py
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class TrapType(Enum):
    """Steam trap types supported by the inspector."""
    THERMOSTATIC = "thermostatic"
    MECHANICAL = "mechanical"
    THERMODYNAMIC = "thermodynamic"
    INVERTED_BUCKET = "inverted_bucket"
    FLOAT_AND_THERMOSTATIC = "float_thermostatic"
    DISC = "disc"
    BIMETALLIC = "bimetallic"
    BALANCED_PRESSURE = "balanced_pressure"


class FailureMode(Enum):
    """Steam trap failure modes."""
    FAILED_OPEN = "failed_open"          # Passing live steam
    FAILED_CLOSED = "failed_closed"      # Condensate backup
    LEAKING = "leaking"                  # Partial failure
    PLUGGED = "plugged"                  # Blocked orifice
    WATERLOGGED = "waterlogged"          # Condensate accumulation
    CAVITATION = "cavitation"            # Pressure differential issues
    WORN_SEAT = "worn_seat"              # Mechanical wear
    NORMAL = "normal"                    # Functioning properly


class InspectionMethod(Enum):
    """Inspection methods available."""
    ACOUSTIC = "acoustic"                # Ultrasonic analysis
    THERMAL = "thermal"                  # IR thermography
    TEMPERATURE_DIFFERENTIAL = "temp_diff"  # Temperature sensors
    VISUAL = "visual"                    # Manual inspection
    MULTI_MODAL = "multi_modal"          # Combined methods


@dataclass
class AcousticConfig:
    """Configuration for acoustic analysis."""
    frequency_range_hz: tuple = (20000, 100000)  # 20-100 kHz ultrasonic
    sampling_rate_hz: int = 250000
    fft_window_size: int = 2048
    overlap_ratio: float = 0.5
    noise_floor_db: float = 30.0
    detection_threshold_db: float = 45.0
    recording_duration_sec: float = 5.0


@dataclass
class ThermalConfig:
    """Configuration for thermal imaging analysis."""
    image_resolution: tuple = (640, 480)
    temperature_range_c: tuple = (0, 250)
    emissivity: float = 0.95  # Carbon steel typical
    ambient_temp_c: float = 20.0
    delta_t_threshold_c: float = 10.0  # Anomaly threshold
    frame_rate_fps: int = 30
    integration_time_ms: int = 10


@dataclass
class MLModelConfig:
    """Configuration for ML models."""
    acoustic_model_path: Path = field(default_factory=lambda: Path("ml_models/acoustic_anomaly_detector.pkl"))
    thermal_model_path: Path = field(default_factory=lambda: Path("ml_models/thermal_cnn.h5"))
    rul_model_path: Path = field(default_factory=lambda: Path("ml_models/rul_predictor.pkl"))
    feature_extractor_path: Path = field(default_factory=lambda: Path("ml_models/feature_extractors.py"))
    model_version: str = "1.0.0"
    confidence_threshold: float = 0.85
    retrain_interval_days: int = 90


@dataclass
class EnergyLossConfig:
    """Configuration for energy loss calculations."""
    steam_pressure_psig: float = 100.0
    steam_temperature_f: float = 338.0  # Saturated at 100 psig
    condensate_temperature_f: float = 180.0
    steam_cost_usd_per_1000lb: float = 8.50  # Typical industrial cost
    operating_hours_per_year: int = 8760  # 24/7 operation
    co2_factor_kg_per_mmbtu: float = 53.06  # Natural gas combustion


@dataclass
class MaintenanceConfig:
    """Configuration for maintenance scheduling."""
    inspection_interval_days: int = 90  # Quarterly inspections
    preventive_maintenance_days: int = 365  # Annual PM
    emergency_response_hours: int = 4  # For critical failures
    maintenance_cost_per_trap_usd: float = 150.0
    replacement_cost_per_trap_usd: float = 500.0
    labor_cost_per_hour_usd: float = 75.0


@dataclass
class SteamTrapConfig:
    """Configuration for individual steam trap."""
    trap_id: str
    trap_type: TrapType
    location: str
    process_criticality: int = 5  # 1-10 scale
    installation_date: Optional[str] = None
    last_inspection_date: Optional[str] = None
    last_maintenance_date: Optional[str] = None
    steam_pressure_psig: float = 100.0
    expected_condensate_load_lb_hr: float = 1000.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrapInspectorConfig:
    """Main configuration for GL-008 SteamTrapInspector agent."""

    # Agent identification
    agent_id: str = "GL-008-TRAPCATCHER"
    agent_name: str = "SteamTrapInspector"
    version: str = "1.0.0"

    # Operational settings
    default_inspection_method: InspectionMethod = InspectionMethod.MULTI_MODAL
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 300  # 5 minutes

    # Component configurations
    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    ml_models: MLModelConfig = field(default_factory=MLModelConfig)
    energy_loss: EnergyLossConfig = field(default_factory=EnergyLossConfig)
    maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)

    # Performance settings
    calculation_timeout_seconds: float = 30.0
    cache_ttl_seconds: float = 300.0  # 5-minute cache
    max_concurrent_inspections: int = 10
    enable_monitoring: bool = True

    # AI/LLM settings (classification only)
    enable_llm_classification: bool = True
    llm_provider: str = "anthropic"
    llm_model: str = "claude-3-haiku"
    llm_temperature: float = 0.0  # Deterministic
    llm_seed: int = 42
    llm_max_tokens: int = 500
    llm_budget_usd: float = 0.50

    # Security and compliance
    enable_provenance_tracking: bool = True
    enable_audit_logging: bool = True
    zero_secrets: bool = True

    # Storage paths
    data_directory: Optional[Path] = None
    model_directory: Optional[Path] = None
    log_directory: Optional[Path] = None

    # Integration settings
    cmms_integration_enabled: bool = False
    cmms_api_endpoint: Optional[str] = None
    bms_integration_enabled: bool = False
    bms_api_endpoint: Optional[str] = None

    # Alert thresholds
    critical_energy_loss_threshold_usd_yr: float = 10000.0
    high_priority_threshold_usd_yr: float = 5000.0
    medium_priority_threshold_usd_yr: float = 2000.0

    # Retry and error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_error_recovery: bool = True

    def __post_init__(self):
        """
        Validate configuration after initialization.

        SECURITY: This method validates critical security settings.
        Full security validation performed by agents.security_validator.validate_startup_security()
        """
        # Set default paths if not provided
        if self.data_directory is None:
            self.data_directory = Path("./gl008_data")
        if self.model_directory is None:
            self.model_directory = Path("./gl008_models")
        if self.log_directory is None:
            self.log_directory = Path("./gl008_logs")

        # Create directories if they don't exist
        for directory in [self.data_directory, self.model_directory, self.log_directory]:
            directory.mkdir(parents=True, exist_ok=True)

        # Validate numeric ranges
        assert 0.0 <= self.llm_temperature <= 1.0, "LLM temperature must be between 0 and 1"
        assert self.llm_temperature == 0.0, "LLM temperature must be 0.0 for deterministic operation"
        assert self.llm_seed == 42, "LLM seed must be 42 for reproducibility"
        assert self.monitoring_interval_seconds > 0, "Monitoring interval must be positive"
        assert self.cache_ttl_seconds > 0, "Cache TTL must be positive"
        assert 1 <= self.max_concurrent_inspections <= 100, "Max concurrent inspections must be 1-100"

        # SECURITY: Validate zero-secrets policy
        self._validate_security_policy()

    def _validate_security_policy(self) -> None:
        """
        Validate security policy settings.

        SECURITY REQUIREMENTS per IEC 62443-4-2:
        - Zero secrets in configuration
        - Audit logging enabled
        - Provenance tracking enabled
        """
        # Ensure zero_secrets policy is enabled
        if not self.zero_secrets:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. "
                "No credentials allowed in configuration."
            )

        # Check CMMS/BMS endpoints don't contain credentials
        for endpoint_name, endpoint_value in [
            ("cmms_api_endpoint", self.cmms_api_endpoint),
            ("bms_api_endpoint", self.bms_api_endpoint),
        ]:
            if endpoint_value:
                self._validate_no_credentials_in_url(endpoint_name, endpoint_value)

    @staticmethod
    def _validate_no_credentials_in_url(name: str, url: str) -> None:
        """
        Validate URL does not contain embedded credentials.

        Args:
            name: Name of the configuration field
            url: URL to validate

        Raises:
            ValueError: If URL contains embedded credentials
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.username or parsed.password:
            raise ValueError(
                f"SECURITY VIOLATION: {name} contains embedded credentials. "
                f"Use environment variables or secret management instead."
            )

    @staticmethod
    def get_api_key(provider: str = "anthropic") -> Optional[str]:
        """
        Get API key from environment variable.

        SECURITY: API keys must be stored in environment variables, never in code.

        Args:
            provider: API provider name (anthropic, openai, etc.)

        Returns:
            API key from environment, or None if not set
        """
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
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


@dataclass
class FleetConfig:
    """Configuration for steam trap fleet management."""
    fleet_id: str
    fleet_name: str
    traps: List[SteamTrapConfig]
    total_steam_pressure_psig: float = 100.0
    total_annual_steam_cost_usd: float = 1000000.0
    facility_name: str = ""
    facility_location: str = ""
    operator_contact: str = ""

    def get_trap_count(self) -> int:
        """Get total number of traps in fleet."""
        return len(self.traps)

    def get_trap_by_id(self, trap_id: str) -> Optional[SteamTrapConfig]:
        """Retrieve specific trap configuration."""
        for trap in self.traps:
            if trap.trap_id == trap_id:
                return trap
        return None

    def get_traps_by_criticality(self, min_criticality: int = 7) -> List[SteamTrapConfig]:
        """Get high-criticality traps."""
        return [trap for trap in self.traps if trap.process_criticality >= min_criticality]
