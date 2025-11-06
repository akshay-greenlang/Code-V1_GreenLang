"""
Scope3CalculatorAgent Configuration
GL-VCCI Scope 3 Platform

Configuration management for the Scope3CalculatorAgent with environment
variable support and validation.

Version: 1.0.0
Date: 2025-10-30
"""

import os
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class TierType(str, Enum):
    """Data tier types for calculation quality."""
    TIER_1 = "tier_1"  # Primary data (supplier-specific)
    TIER_2 = "tier_2"  # Secondary data (database averages)
    TIER_3 = "tier_3"  # Tertiary data (spend-based, proxies)


class TransportMode(str, Enum):
    """Transport modes for Category 4 (ISO 14083)."""
    # Road
    ROAD_TRUCK_LIGHT = "road_truck_light"
    ROAD_TRUCK_MEDIUM = "road_truck_medium"
    ROAD_TRUCK_HEAVY = "road_truck_heavy"
    ROAD_VAN = "road_van"

    # Rail
    RAIL_FREIGHT = "rail_freight"
    RAIL_FREIGHT_ELECTRIC = "rail_freight_electric"
    RAIL_FREIGHT_DIESEL = "rail_freight_diesel"

    # Sea
    SEA_CONTAINER = "sea_container"
    SEA_BULK = "sea_bulk"
    SEA_TANKER = "sea_tanker"
    SEA_RO_RO = "sea_ro_ro"

    # Air
    AIR_CARGO = "air_cargo"
    AIR_FREIGHT = "air_freight"

    # Inland Waterway
    INLAND_WATERWAY = "inland_waterway"


class CabinClass(str, Enum):
    """Flight cabin classes for Category 6."""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class CalculatorConfig(BaseModel):
    """Main configuration for Scope3CalculatorAgent."""

    # General settings
    enable_monte_carlo: bool = Field(
        default=True,
        description="Enable Monte Carlo uncertainty propagation"
    )

    monte_carlo_iterations: int = Field(
        default=10000,
        ge=1000,
        le=100000,
        description="Number of Monte Carlo iterations"
    )

    enable_provenance: bool = Field(
        default=True,
        description="Enable provenance chain tracking"
    )

    enable_opa_policies: bool = Field(
        default=False,
        description="Enable OPA policy-based calculations"
    )

    # Category 1 settings
    category_1_enable_tier_fallback: bool = Field(
        default=True,
        description="Enable automatic tier fallback for Category 1"
    )

    category_1_prefer_supplier_pcf: bool = Field(
        default=True,
        description="Prefer supplier-specific PCF over averages"
    )

    category_1_min_dqi_score: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Minimum acceptable DQI score for Category 1"
    )

    # Category 4 settings (ISO 14083)
    category_4_enforce_iso14083: bool = Field(
        default=True,
        description="Enforce strict ISO 14083 compliance"
    )

    category_4_default_distance_unit: str = Field(
        default="km",
        description="Default distance unit (km or mi)"
    )

    category_4_default_weight_unit: str = Field(
        default="tonne",
        description="Default weight unit (tonne, kg, lb)"
    )

    category_4_apply_radiative_forcing: bool = Field(
        default=False,
        description="Apply radiative forcing for air transport (for comparison with Cat 6)"
    )

    # Category 6 settings
    category_6_radiative_forcing_factor: float = Field(
        default=1.9,
        ge=1.0,
        le=3.0,
        description="Radiative forcing factor for flights (DEFRA: 1.9)"
    )

    category_6_include_hotel_emissions: bool = Field(
        default=True,
        description="Include hotel stay emissions"
    )

    category_6_include_ground_transport: bool = Field(
        default=True,
        description="Include ground transport emissions"
    )

    # Performance settings
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Batch size for parallel processing"
    )

    enable_parallel_processing: bool = Field(
        default=True,
        description="Enable parallel processing for batch calculations"
    )

    max_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum parallel workers"
    )

    # OPA settings
    opa_server_url: str = Field(
        default="http://localhost:8181",
        description="OPA server URL"
    )

    opa_timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="OPA policy evaluation timeout"
    )

    # Data quality settings
    warn_on_low_dqi: bool = Field(
        default=True,
        description="Warn when DQI score is below threshold"
    )

    low_dqi_threshold: float = Field(
        default=60.0,
        ge=0.0,
        le=100.0,
        description="DQI threshold for warnings"
    )

    # Tier-specific DQI scores
    tier_1_dqi_score: float = Field(default=90.0, ge=0.0, le=100.0)
    tier_2_dqi_score: float = Field(default=70.0, ge=0.0, le=100.0)
    tier_3_dqi_score: float = Field(default=40.0, ge=0.0, le=100.0)

    @classmethod
    def from_env(cls) -> "CalculatorConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_monte_carlo=os.getenv("CALC_ENABLE_MONTE_CARLO", "true").lower() == "true",
            monte_carlo_iterations=int(os.getenv("CALC_MONTE_CARLO_ITERATIONS", "10000")),
            enable_provenance=os.getenv("CALC_ENABLE_PROVENANCE", "true").lower() == "true",
            enable_opa_policies=os.getenv("CALC_ENABLE_OPA", "false").lower() == "true",

            category_1_enable_tier_fallback=os.getenv("CALC_CAT1_TIER_FALLBACK", "true").lower() == "true",
            category_1_prefer_supplier_pcf=os.getenv("CALC_CAT1_PREFER_PCF", "true").lower() == "true",
            category_1_min_dqi_score=float(os.getenv("CALC_CAT1_MIN_DQI", "50.0")),

            category_4_enforce_iso14083=os.getenv("CALC_CAT4_ENFORCE_ISO14083", "true").lower() == "true",
            category_4_default_distance_unit=os.getenv("CALC_CAT4_DISTANCE_UNIT", "km"),
            category_4_default_weight_unit=os.getenv("CALC_CAT4_WEIGHT_UNIT", "tonne"),

            category_6_radiative_forcing_factor=float(os.getenv("CALC_CAT6_RF_FACTOR", "1.9")),
            category_6_include_hotel_emissions=os.getenv("CALC_CAT6_HOTELS", "true").lower() == "true",
            category_6_include_ground_transport=os.getenv("CALC_CAT6_GROUND", "true").lower() == "true",

            batch_size=int(os.getenv("CALC_BATCH_SIZE", "1000")),
            enable_parallel_processing=os.getenv("CALC_PARALLEL", "true").lower() == "true",
            max_workers=int(os.getenv("CALC_MAX_WORKERS", "4")),

            opa_server_url=os.getenv("OPA_SERVER_URL", "http://localhost:8181"),
            opa_timeout_seconds=int(os.getenv("OPA_TIMEOUT", "5")),

            warn_on_low_dqi=os.getenv("CALC_WARN_LOW_DQI", "true").lower() == "true",
            low_dqi_threshold=float(os.getenv("CALC_LOW_DQI_THRESHOLD", "60.0")),
        )

    def get_tier_dqi_score(self, tier: TierType) -> float:
        """Get DQI score for a tier."""
        tier_scores = {
            TierType.TIER_1: self.tier_1_dqi_score,
            TierType.TIER_2: self.tier_2_dqi_score,
            TierType.TIER_3: self.tier_3_dqi_score,
        }
        return tier_scores.get(tier, self.tier_3_dqi_score)

    class Config:
        json_schema_extra = {
            "example": {
                "enable_monte_carlo": True,
                "monte_carlo_iterations": 10000,
                "enable_provenance": True,
                "category_4_enforce_iso14083": True,
                "category_6_radiative_forcing_factor": 1.9,
            }
        }


# Default configuration instance
_default_config: Optional[CalculatorConfig] = None


def get_config() -> CalculatorConfig:
    """Get default configuration instance."""
    global _default_config
    if _default_config is None:
        _default_config = CalculatorConfig.from_env()
    return _default_config


def set_config(config: CalculatorConfig):
    """Set default configuration instance."""
    global _default_config
    _default_config = config


# Transport mode emission factor mappings (defaults, can be overridden by Factor Broker)
TRANSPORT_MODE_DEFAULTS = {
    # Road (kgCO2e per tonne-km)
    TransportMode.ROAD_TRUCK_LIGHT: 0.180,
    TransportMode.ROAD_TRUCK_MEDIUM: 0.110,
    TransportMode.ROAD_TRUCK_HEAVY: 0.062,
    TransportMode.ROAD_VAN: 0.250,

    # Rail (kgCO2e per tonne-km)
    TransportMode.RAIL_FREIGHT: 0.022,
    TransportMode.RAIL_FREIGHT_ELECTRIC: 0.012,
    TransportMode.RAIL_FREIGHT_DIESEL: 0.028,

    # Sea (kgCO2e per tonne-km)
    TransportMode.SEA_CONTAINER: 0.012,
    TransportMode.SEA_BULK: 0.008,
    TransportMode.SEA_TANKER: 0.005,
    TransportMode.SEA_RO_RO: 0.020,

    # Air (kgCO2e per tonne-km)
    TransportMode.AIR_CARGO: 0.680,
    TransportMode.AIR_FREIGHT: 0.602,

    # Inland Waterway (kgCO2e per tonne-km)
    TransportMode.INLAND_WATERWAY: 0.031,
}

# Flight emission factors by cabin class (kgCO2e per passenger-km)
FLIGHT_EMISSION_FACTORS = {
    CabinClass.ECONOMY: 0.115,
    CabinClass.PREMIUM_ECONOMY: 0.175,
    CabinClass.BUSINESS: 0.230,
    CabinClass.FIRST: 0.345,
}

# Hotel emission factors by region (kgCO2e per night)
HOTEL_EMISSION_FACTORS = {
    "US": 28.5,
    "GB": 24.2,
    "EU": 22.8,
    "CN": 35.6,
    "JP": 26.4,
    "Global": 26.0,
}

__all__ = [
    "CalculatorConfig",
    "TierType",
    "TransportMode",
    "CabinClass",
    "get_config",
    "set_config",
    "TRANSPORT_MODE_DEFAULTS",
    "FLIGHT_EMISSION_FACTORS",
    "HOTEL_EMISSION_FACTORS",
]
