# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Configuration Module

Pydantic configuration models for solver, forecasting, carbon accounting,
and safety parameters per Global AI Standards v2.0.

Reference Standards:
    - IEC 61511 (Functional Safety)
    - ISO 14064 (GHG Quantification)
    - TCFD (Climate Disclosure)
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class SolverType(str, Enum):
    """Supported optimization solvers."""
    HIGHS = "highs"
    CBC = "cbc"
    CPLEX = "cplex"
    GUROBI = "gurobi"


class CarbonBoundary(str, Enum):
    """Emission boundary definitions."""
    TTW = "tank_to_wheel"
    WTT = "well_to_tank"
    WTW = "well_to_wheel"


class SolverConfig(BaseModel):
    """LP/MILP solver configuration for deterministic optimization."""

    solver_type: SolverType = Field(default=SolverType.HIGHS)
    timeout_seconds: float = Field(default=300.0, ge=10, le=3600)
    mip_gap_tolerance: float = Field(default=0.01, ge=0, le=0.1)
    threads: int = Field(default=1, ge=1, le=32)
    random_seed: int = Field(default=42, ge=0)
    deterministic: bool = Field(default=True)
    log_level: int = Field(default=1, ge=0, le=3)


class ForecastConfig(BaseModel):
    """Price forecasting configuration."""

    horizons_hours: List[int] = Field(default=[24, 168, 720])
    quantiles: List[float] = Field(default=[0.1, 0.5, 0.9])
    min_history_days: int = Field(default=365, ge=30)
    feature_version: str = Field(default="1.0.0")
    model_registry_url: Optional[str] = Field(default=None)


class CarbonConfig(BaseModel):
    """Carbon accounting configuration."""

    boundary: CarbonBoundary = Field(default=CarbonBoundary.WTW)
    gwp_version: str = Field(default="AR6")
    intensity_unit: str = Field(default="kgCO2e/MJ")
    emission_factor_source: str = Field(default="governed")
    enable_pathway_specific: bool = Field(default=True)


class SafetyConfig(BaseModel):
    """IEC 61511 safety configuration."""

    sil_level: int = Field(default=2, ge=1, le=4)
    fail_closed_enabled: bool = Field(default=True)
    circuit_breaker_enabled: bool = Field(default=True)
    failure_threshold: int = Field(default=5, ge=1, le=20)
    recovery_timeout_seconds: float = Field(default=30.0, ge=5, le=600)


class FuelCraftConfig(BaseModel):
    """Master configuration for GL-011 FuelCraft."""

    site_id: str = Field(..., min_length=1)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    carbon: CarbonConfig = Field(default_factory=CarbonConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)

    # Operational parameters
    energy_unit: str = Field(default="MJ")
    mass_unit: str = Field(default="kg")
    volume_reference_temp_c: float = Field(default=15.0)
    audit_retention_days: int = Field(default=2555)  # 7 years

    class Config:
        frozen = True
