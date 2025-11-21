# -*- coding: utf-8 -*-
"""
Methodologies Configuration Module

Configuration settings for uncertainty quantification, Monte Carlo simulation,
and data quality assessment.

Key Features:
- Default uncertainty values by category
- Simulation parameters (iterations, seed)
- GWP values (AR5, AR6) configuration
- Quality thresholds and scoring parameters
- Calculation precision settings

Version: 1.0.0
Date: 2025-10-30
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from .constants import (
    GWPVersion,
    DistributionType,
    MC_DEFAULT_ITERATIONS,
    MC_MIN_ITERATIONS,
    MC_MAX_ITERATIONS,
    DEFAULT_GWP_VERSION,
    BASIC_UNCERTAINTY,
    MIN_UNCERTAINTY,
    MAX_UNCERTAINTY,
)


# ============================================================================
# MONTE CARLO CONFIGURATION
# ============================================================================

class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation."""

    default_iterations: int = Field(
        default=MC_DEFAULT_ITERATIONS,
        ge=MC_MIN_ITERATIONS,
        le=MC_MAX_ITERATIONS,
        description="Default number of Monte Carlo iterations",
    )
    min_iterations: int = Field(
        default=MC_MIN_ITERATIONS,
        description="Minimum allowed iterations",
    )
    max_iterations: int = Field(
        default=MC_MAX_ITERATIONS,
        description="Maximum allowed iterations",
    )
    default_seed: Optional[int] = Field(
        default=None,
        description="Default random seed (None for random)",
    )
    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel computation for large simulations",
    )
    parallel_threshold: int = Field(
        default=50_000,
        description="Iteration threshold for enabling parallelization",
    )
    percentiles: list[float] = Field(
        default=[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        description="Percentiles to calculate in results",
    )
    confidence_levels: list[float] = Field(
        default=[0.90, 0.95],
        description="Confidence levels for interval calculation",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "default_iterations": 10000,
                    "min_iterations": 1000,
                    "max_iterations": 1000000,
                    "default_seed": None,
                    "enable_parallel": True,
                    "parallel_threshold": 50000,
                }
            ]
        }
    }


# ============================================================================
# UNCERTAINTY CONFIGURATION
# ============================================================================

class UncertaintyConfig(BaseModel):
    """Configuration for uncertainty quantification."""

    default_uncertainty: float = Field(
        default=BASIC_UNCERTAINTY,
        ge=MIN_UNCERTAINTY,
        le=MAX_UNCERTAINTY,
        description="Default uncertainty when no better data available",
    )
    min_uncertainty: float = Field(
        default=MIN_UNCERTAINTY,
        description="Minimum uncertainty floor (1%)",
    )
    max_uncertainty: float = Field(
        default=MAX_UNCERTAINTY,
        description="Maximum uncertainty ceiling (500%)",
    )
    default_distribution: DistributionType = Field(
        default=DistributionType.LOGNORMAL,
        description="Default probability distribution",
    )
    apply_floor: bool = Field(
        default=True,
        description="Apply minimum uncertainty floor",
    )
    apply_ceiling: bool = Field(
        default=True,
        description="Apply maximum uncertainty ceiling",
    )
    propagation_method: str = Field(
        default="monte_carlo",
        description="Uncertainty propagation method (monte_carlo, analytical)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "default_uncertainty": 0.5,
                    "min_uncertainty": 0.01,
                    "max_uncertainty": 5.0,
                    "default_distribution": "lognormal",
                    "apply_floor": True,
                    "apply_ceiling": True,
                }
            ]
        }
    }


# ============================================================================
# DATA QUALITY INDEX (DQI) CONFIGURATION
# ============================================================================

class DQIConfig(BaseModel):
    """Configuration for Data Quality Index calculation."""

    # Component weights (must sum to 1.0)
    pedigree_weight: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description="Weight for pedigree matrix score",
    )
    source_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for factor source quality",
    )
    tier_weight: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for data tier",
    )

    # Quality thresholds
    excellent_threshold: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Threshold for 'Excellent' quality label",
    )
    good_threshold: float = Field(
        default=70.0,
        ge=0.0,
        le=100.0,
        description="Threshold for 'Good' quality label",
    )
    fair_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Threshold for 'Fair' quality label",
    )

    # Scoring options
    use_pedigree_interpolation: bool = Field(
        default=True,
        description="Use interpolation for pedigree-to-DQI conversion",
    )
    apply_tier_penalty: bool = Field(
        default=True,
        description="Apply penalty for lower tier data",
    )

    def validate_weights(self) -> None:
        """Validate that weights sum to 1.0."""
        total = self.pedigree_weight + self.source_weight + self.tier_weight
        if not (0.99 <= total <= 1.01):  # Allow small floating-point errors
            raise ValueError(f"DQI weights must sum to 1.0, got {total}")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pedigree_weight": 0.5,
                    "source_weight": 0.3,
                    "tier_weight": 0.2,
                    "excellent_threshold": 90.0,
                    "good_threshold": 70.0,
                    "fair_threshold": 50.0,
                }
            ]
        }
    }


# ============================================================================
# GWP CONFIGURATION
# ============================================================================

class GWPConfig(BaseModel):
    """Configuration for Global Warming Potential calculations."""

    default_version: GWPVersion = Field(
        default=DEFAULT_GWP_VERSION,
        description="Default IPCC AR version for GWP values",
    )
    time_horizon: int = Field(
        default=100,
        description="Time horizon in years (typically 100)",
    )
    include_climate_feedbacks: bool = Field(
        default=True,
        description="Include climate-carbon feedbacks in GWP values",
    )
    biogenic_carbon: bool = Field(
        default=True,
        description="Account for biogenic carbon separately",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "default_version": "AR5",
                    "time_horizon": 100,
                    "include_climate_feedbacks": True,
                    "biogenic_carbon": True,
                }
            ]
        }
    }


# ============================================================================
# PEDIGREE MATRIX CONFIGURATION
# ============================================================================

class PedigreeMatrixConfig(BaseModel):
    """Configuration for Pedigree Matrix assessment."""

    use_ilcd_standard: bool = Field(
        default=True,
        description="Use ILCD standard pedigree matrix",
    )
    enable_temporal_adjustment: bool = Field(
        default=True,
        description="Automatically adjust temporal scores based on data age",
    )
    reference_year: int = Field(
        default=2024,
        description="Reference year for temporal assessment",
    )
    strict_validation: bool = Field(
        default=True,
        description="Apply strict validation to pedigree scores",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "use_ilcd_standard": True,
                    "enable_temporal_adjustment": True,
                    "reference_year": 2024,
                    "strict_validation": True,
                }
            ]
        }
    }


# ============================================================================
# CALCULATION CONFIGURATION
# ============================================================================

class CalculationConfig(BaseModel):
    """Configuration for calculation precision and rounding."""

    decimal_places: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Decimal places for rounding results",
    )
    scientific_notation_threshold: float = Field(
        default=1e-6,
        description="Threshold for using scientific notation",
    )
    zero_threshold: float = Field(
        default=1e-10,
        description="Threshold for treating values as zero",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of calculation results",
    )
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decimal_places": 4,
                    "scientific_notation_threshold": 1e-6,
                    "zero_threshold": 1e-10,
                    "enable_caching": True,
                    "cache_ttl": 3600,
                }
            ]
        }
    }


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

class MethodologiesConfig(BaseSettings):
    """
    Master configuration for Methodologies module.

    This configuration can be loaded from environment variables with prefix
    'METHODOLOGIES_'.

    Example:
        METHODOLOGIES_MONTE_CARLO__DEFAULT_ITERATIONS=10000
        METHODOLOGIES_UNCERTAINTY__DEFAULT_UNCERTAINTY=0.5
    """

    monte_carlo: MonteCarloConfig = Field(
        default_factory=MonteCarloConfig,
        description="Monte Carlo simulation configuration",
    )
    uncertainty: UncertaintyConfig = Field(
        default_factory=UncertaintyConfig,
        description="Uncertainty quantification configuration",
    )
    dqi: DQIConfig = Field(
        default_factory=DQIConfig,
        description="Data Quality Index configuration",
    )
    gwp: GWPConfig = Field(
        default_factory=GWPConfig,
        description="Global Warming Potential configuration",
    )
    pedigree: PedigreeMatrixConfig = Field(
        default_factory=PedigreeMatrixConfig,
        description="Pedigree Matrix configuration",
    )
    calculation: CalculationConfig = Field(
        default_factory=CalculationConfig,
        description="Calculation precision configuration",
    )

    model_config = {
        "env_prefix": "METHODOLOGIES_",
        "env_nested_delimiter": "__",
        "case_sensitive": False,
    }

    def validate_all(self) -> None:
        """Validate all configuration sections."""
        self.dqi.validate_weights()


# ============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================================

# Global configuration instance (can be overridden)
config = MethodologiesConfig()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config() -> MethodologiesConfig:
    """Get the global configuration instance."""
    return config


def reload_config() -> MethodologiesConfig:
    """Reload configuration from environment variables."""
    global config
    config = MethodologiesConfig()
    return config


def update_config(**kwargs) -> MethodologiesConfig:
    """
    Update configuration programmatically.

    Example:
        >>> update_config(
        ...     monte_carlo={"default_iterations": 20000},
        ...     uncertainty={"default_uncertainty": 0.3}
        ... )
    """
    global config
    config_dict = config.model_dump()

    for key, value in kwargs.items():
        if key in config_dict:
            if isinstance(value, dict):
                config_dict[key].update(value)
            else:
                config_dict[key] = value

    config = MethodologiesConfig(**config_dict)
    return config


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "MonteCarloConfig",
    "UncertaintyConfig",
    "DQIConfig",
    "GWPConfig",
    "PedigreeMatrixConfig",
    "CalculationConfig",
    "MethodologiesConfig",
    "config",
    "get_config",
    "reload_config",
    "update_config",
]
