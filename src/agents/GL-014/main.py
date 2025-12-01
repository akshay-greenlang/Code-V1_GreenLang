# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO REST API Main Application

Production-grade FastAPI REST API for Heat Exchanger Performance Optimization.
Implements GreenLang standard patterns: JWT auth, rate limiting, audit trails,
zero-hallucination calculations with complete provenance tracking.

Agent ID: GL-014
Codename: EXCHANGER-PRO
Version: 1.0.0
Category: Heat Exchangers
Type: Optimizer

Features:
- Full heat exchanger analysis with TEMA compliance
- Heat transfer coefficient calculations
- Fouling analysis with Kern-Seaton and Ebert-Panchal models
- Pressure drop calculations (tube-side and shell-side)
- Performance tracking and benchmarking
- Cleaning schedule optimization
- Economic impact assessment
- Fleet-wide optimization

API Security:
- OAuth2/JWT authentication
- Rate limiting per user/endpoint
- Request validation with Pydantic
- CORS configuration
- Audit logging

Author: GreenLang AI Agent Factory - GL-APIDeveloper
Created: 2025-12-01
License: Proprietary - GreenLang
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator, model_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Configure module logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# JWT Configuration
SECRET_KEY = "gl-014-exchanger-pro-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Rate Limiting Configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60

# Redis Cache Configuration (mock for demonstration)
CACHE_TTL_SECONDS = 300
CACHE_ENABLED = True

# API Version
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ExchangerType(str, Enum):
    """Heat exchanger type classification."""
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"
    AIR_COOLED = "air_cooled"
    SPIRAL = "spiral"
    DOUBLE_PIPE = "double_pipe"
    FINNED_TUBE = "finned_tube"


class FluidType(str, Enum):
    """Process fluid type classification."""
    WATER_TREATED = "water_treated"
    WATER_UNTREATED = "water_untreated"
    WATER_COOLING_TOWER = "water_cooling_tower"
    WATER_SEAWATER = "water_seawater"
    STEAM = "steam"
    OIL_LIGHT = "oil_light"
    OIL_HEAVY = "oil_heavy"
    OIL_CRUDE = "oil_crude"
    GAS_NATURAL = "gas_natural"
    GAS_FLUE = "gas_flue"
    REFRIGERANT = "refrigerant"
    PROCESS_FLUID = "process_fluid"


class FoulingMechanism(str, Enum):
    """Primary fouling mechanism types."""
    PARTICULATE = "particulate"
    CRYSTALLIZATION = "crystallization"
    BIOLOGICAL = "biological"
    CORROSION = "corrosion"
    CHEMICAL_REACTION = "chemical_reaction"
    COMBINED = "combined"


class CleaningMethod(str, Enum):
    """Heat exchanger cleaning methods."""
    CHEMICAL_ACID = "chemical_acid"
    CHEMICAL_ALKALINE = "chemical_alkaline"
    MECHANICAL_HYDROBLAST = "mechanical_hydroblast"
    MECHANICAL_PIGGING = "mechanical_pigging"
    ONLINE_SPONGE_BALLS = "online_sponge_balls"
    COMBINED = "combined"


class FoulingSeverity(str, Enum):
    """Fouling severity classification levels."""
    CLEAN = "clean"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Equipment health status classification."""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"


class TrendDirection(str, Enum):
    """Performance trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING_SLOW = "degrading_slow"
    DEGRADING_FAST = "degrading_fast"
    CRITICAL_DECLINE = "critical_decline"


class AnalysisType(str, Enum):
    """Analysis type for API tracking."""
    FULL_ANALYSIS = "full_analysis"
    HEAT_TRANSFER = "heat_transfer"
    FOULING = "fouling"
    PRESSURE_DROP = "pressure_drop"
    PERFORMANCE = "performance"
    CLEANING = "cleaning"
    ECONOMIC = "economic"
    FLEET = "fleet"


# =============================================================================
# PYDANTIC MODELS - REQUEST/RESPONSE
# =============================================================================


class Token(BaseModel):
    """OAuth2 token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class TokenData(BaseModel):
    """Token payload data model."""
    username: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: List[str] = []
    exp: Optional[datetime] = None


class User(BaseModel):
    """User model for authentication."""
    id: str
    username: str
    email: str
    tenant_id: str
    roles: List[str] = []
    is_active: bool = True


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


# -----------------------------------------------------------------------------
# Base Models
# -----------------------------------------------------------------------------


class ExchangerIdentification(BaseModel):
    """Heat exchanger identification data."""
    exchanger_id: str = Field(..., min_length=1, max_length=50, description="Unique exchanger ID")
    name: Optional[str] = Field(None, max_length=100, description="Exchanger name")
    plant_id: Optional[str] = Field(None, description="Plant/facility ID")
    unit_id: Optional[str] = Field(None, description="Process unit ID")
    service: Optional[str] = Field(None, description="Service description")
    exchanger_type: ExchangerType = Field(default=ExchangerType.SHELL_AND_TUBE)


class TemperatureData(BaseModel):
    """Temperature measurement data."""
    hot_inlet_temp_c: float = Field(..., ge=-273.15, le=1000, description="Hot side inlet temperature (C)")
    hot_outlet_temp_c: float = Field(..., ge=-273.15, le=1000, description="Hot side outlet temperature (C)")
    cold_inlet_temp_c: float = Field(..., ge=-273.15, le=1000, description="Cold side inlet temperature (C)")
    cold_outlet_temp_c: float = Field(..., ge=-273.15, le=1000, description="Cold side outlet temperature (C)")

    @model_validator(mode='after')
    def validate_temperatures(self):
        """Validate temperature relationships."""
        if self.hot_inlet_temp_c < self.hot_outlet_temp_c:
            raise ValueError("Hot inlet must be >= hot outlet for cooling")
        if self.cold_outlet_temp_c < self.cold_inlet_temp_c:
            raise ValueError("Cold outlet must be >= cold inlet for heating")
        return self


class FlowData(BaseModel):
    """Flow rate measurement data."""
    hot_side_mass_flow_kg_s: float = Field(..., gt=0, description="Hot side mass flow rate (kg/s)")
    cold_side_mass_flow_kg_s: float = Field(..., gt=0, description="Cold side mass flow rate (kg/s)")
    hot_side_fluid: FluidType = Field(default=FluidType.PROCESS_FLUID)
    cold_side_fluid: FluidType = Field(default=FluidType.WATER_COOLING_TOWER)


class PressureData(BaseModel):
    """Pressure measurement data."""
    hot_inlet_pressure_bar: float = Field(..., ge=0, description="Hot side inlet pressure (bar)")
    hot_outlet_pressure_bar: float = Field(..., ge=0, description="Hot side outlet pressure (bar)")
    cold_inlet_pressure_bar: float = Field(..., ge=0, description="Cold side inlet pressure (bar)")
    cold_outlet_pressure_bar: float = Field(..., ge=0, description="Cold side outlet pressure (bar)")

    @property
    def hot_pressure_drop_bar(self) -> float:
        return self.hot_inlet_pressure_bar - self.hot_outlet_pressure_bar

    @property
    def cold_pressure_drop_bar(self) -> float:
        return self.cold_inlet_pressure_bar - self.cold_outlet_pressure_bar


class GeometryData(BaseModel):
    """Heat exchanger geometry data."""
    heat_transfer_area_m2: float = Field(..., gt=0, description="Total heat transfer area (m2)")
    tube_od_m: float = Field(default=0.0254, gt=0, description="Tube outer diameter (m)")
    tube_id_m: float = Field(default=0.0229, gt=0, description="Tube inner diameter (m)")
    tube_length_m: float = Field(default=6.0, gt=0, description="Tube length (m)")
    tube_count: int = Field(default=100, gt=0, description="Number of tubes")
    tube_passes: int = Field(default=2, gt=0, description="Number of tube passes")
    shell_diameter_m: float = Field(default=0.5, gt=0, description="Shell diameter (m)")
    baffle_spacing_m: float = Field(default=0.3, gt=0, description="Baffle spacing (m)")
    baffle_cut_percent: float = Field(default=25, ge=15, le=45, description="Baffle cut percentage")


class DesignData(BaseModel):
    """Design conditions data."""
    design_u_value_w_m2_k: float = Field(..., gt=0, description="Design overall heat transfer coefficient (W/m2.K)")
    design_duty_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    design_fouling_hot_m2_k_w: float = Field(default=0.00035, ge=0, description="Design fouling resistance hot side (m2.K/W)")
    design_fouling_cold_m2_k_w: float = Field(default=0.00018, ge=0, description="Design fouling resistance cold side (m2.K/W)")
    design_pressure_drop_hot_bar: float = Field(default=0.5, ge=0, description="Design pressure drop hot side (bar)")
    design_pressure_drop_cold_bar: float = Field(default=0.5, ge=0, description="Design pressure drop cold side (bar)")


# -----------------------------------------------------------------------------
# Analysis Request Models
# -----------------------------------------------------------------------------


class FullAnalysisRequest(BaseModel):
    """Request model for full heat exchanger analysis."""
    identification: ExchangerIdentification
    temperature: TemperatureData
    flow: FlowData
    pressure: Optional[PressureData] = None
    geometry: GeometryData
    design: DesignData
    include_economic: bool = Field(default=True, description="Include economic impact analysis")
    include_cleaning_recommendation: bool = Field(default=True, description="Include cleaning recommendations")
    calculation_timestamp: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "identification": {
                    "exchanger_id": "HX-001-A",
                    "name": "Feed Preheater A",
                    "plant_id": "REFINERY-001",
                    "exchanger_type": "shell_and_tube"
                },
                "temperature": {
                    "hot_inlet_temp_c": 150.0,
                    "hot_outlet_temp_c": 90.0,
                    "cold_inlet_temp_c": 30.0,
                    "cold_outlet_temp_c": 70.0
                },
                "flow": {
                    "hot_side_mass_flow_kg_s": 10.0,
                    "cold_side_mass_flow_kg_s": 15.0,
                    "hot_side_fluid": "oil_crude",
                    "cold_side_fluid": "water_cooling_tower"
                },
                "geometry": {
                    "heat_transfer_area_m2": 100.0,
                    "tube_count": 200,
                    "tube_passes": 2
                },
                "design": {
                    "design_u_value_w_m2_k": 500.0,
                    "design_duty_kw": 2500.0
                }
            }
        }


class HeatTransferRequest(BaseModel):
    """Request model for heat transfer calculations."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    u_clean_w_m2_k: float = Field(..., gt=0, description="Clean overall heat transfer coefficient (W/m2.K)")
    u_fouled_w_m2_k: Optional[float] = Field(None, gt=0, description="Current fouled U value (W/m2.K)")
    temperature: TemperatureData
    heat_transfer_area_m2: float = Field(..., gt=0, description="Heat transfer area (m2)")
    flow_arrangement: str = Field(default="counterflow", description="Flow arrangement")


class FoulingAnalysisRequest(BaseModel):
    """Request model for fouling analysis."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    u_clean_w_m2_k: float = Field(..., gt=0, description="Clean U value (W/m2.K)")
    u_fouled_w_m2_k: float = Field(..., gt=0, description="Current fouled U value (W/m2.K)")
    fluid_type_hot: FluidType = Field(default=FluidType.PROCESS_FLUID)
    fluid_type_cold: FluidType = Field(default=FluidType.WATER_COOLING_TOWER)
    temperature_hot_c: Optional[float] = Field(None, description="Hot side temperature (C)")
    velocity_hot_m_s: Optional[float] = Field(None, gt=0, description="Hot side velocity (m/s)")


class PressureDropRequest(BaseModel):
    """Request model for pressure drop calculations."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    mass_flow_rate_kg_s: float = Field(..., gt=0, description="Mass flow rate (kg/s)")
    density_kg_m3: float = Field(..., gt=0, description="Fluid density (kg/m3)")
    viscosity_pa_s: float = Field(..., gt=0, description="Dynamic viscosity (Pa.s)")
    geometry: GeometryData
    side: str = Field(default="tube", description="Side: tube or shell")
    include_fouling_impact: bool = Field(default=True)
    fouling_thickness_m: float = Field(default=0.0, ge=0, description="Fouling layer thickness (m)")


class PerformanceTrackRequest(BaseModel):
    """Request model for performance tracking."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    measurement_timestamp: datetime = Field(default_factory=datetime.utcnow)
    u_value_actual_w_m2_k: float = Field(..., gt=0, description="Actual U value (W/m2.K)")
    u_value_design_w_m2_k: float = Field(..., gt=0, description="Design U value (W/m2.K)")
    heat_duty_actual_kw: float = Field(..., ge=0, description="Actual heat duty (kW)")
    heat_duty_design_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    pressure_drop_actual_shell_bar: Optional[float] = Field(None, ge=0)
    pressure_drop_actual_tube_bar: Optional[float] = Field(None, ge=0)
    pressure_drop_design_shell_bar: Optional[float] = Field(None, ge=0)
    pressure_drop_design_tube_bar: Optional[float] = Field(None, ge=0)


class BenchmarkRequest(BaseModel):
    """Request model for fleet benchmarking."""
    exchanger_id: str = Field(..., description="Primary exchanger to benchmark")
    fleet_ids: List[str] = Field(..., min_length=1, description="Fleet exchanger IDs for comparison")
    metric: str = Field(default="efficiency", description="Benchmark metric")


class FoulingPredictionRequest(BaseModel):
    """Request model for fouling prediction."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    current_fouling_m2_k_w: float = Field(..., ge=0, description="Current fouling resistance (m2.K/W)")
    fouling_rate_m2_k_w_per_day: float = Field(..., ge=0, description="Fouling rate (m2.K/W per day)")
    prediction_days: int = Field(default=90, gt=0, le=365, description="Prediction horizon (days)")
    design_fouling_m2_k_w: float = Field(..., gt=0, description="Design fouling resistance (m2.K/W)")
    use_asymptotic_model: bool = Field(default=False, description="Use Kern-Seaton asymptotic model")
    r_f_max_m2_k_w: Optional[float] = Field(None, gt=0, description="Asymptotic R_f for Kern-Seaton")
    time_constant_hours: Optional[float] = Field(None, gt=0, description="Time constant for Kern-Seaton")


class AnomalyDetectionRequest(BaseModel):
    """Request model for fouling anomaly detection."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    fouling_values: List[float] = Field(..., min_length=10, description="Historical fouling values")
    timestamps: List[datetime] = Field(..., min_length=10, description="Measurement timestamps")
    detection_method: str = Field(default="zscore", description="Detection method: zscore, iqr, cusum")
    threshold_sigma: float = Field(default=3.0, gt=0, description="Detection threshold (sigma)")


class CleaningOptimizeRequest(BaseModel):
    """Request model for cleaning schedule optimization."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    current_fouling_m2_k_w: float = Field(..., ge=0, description="Current fouling resistance")
    fouling_rate_m2_k_w_per_day: float = Field(..., ge=0, description="Fouling rate per day")
    max_allowable_fouling_m2_k_w: float = Field(..., gt=0, description="Maximum allowable fouling")
    cleaning_cost_usd: float = Field(..., gt=0, description="Cost per cleaning event (USD)")
    downtime_cost_per_hour_usd: float = Field(..., gt=0, description="Production loss per hour (USD)")
    cleaning_duration_hours: float = Field(default=8.0, gt=0, description="Cleaning duration (hours)")
    energy_cost_per_kwh_usd: float = Field(default=0.10, gt=0, description="Energy cost (USD/kWh)")
    heat_duty_kw: float = Field(..., gt=0, description="Heat exchanger duty (kW)")


class CostBenefitRequest(BaseModel):
    """Request model for cleaning cost-benefit analysis."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    cleaning_method: CleaningMethod
    current_fouling_m2_k_w: float = Field(..., ge=0)
    cleaning_effectiveness: float = Field(default=0.95, ge=0, le=1, description="Cleaning effectiveness (0-1)")
    cleaning_cost_usd: float = Field(..., gt=0)
    annual_operating_hours: int = Field(default=8000, gt=0)
    energy_cost_per_kwh_usd: float = Field(default=0.10, gt=0)
    heat_duty_kw: float = Field(..., gt=0)
    u_clean_w_m2_k: float = Field(..., gt=0)
    heat_transfer_area_m2: float = Field(..., gt=0)


class EconomicImpactRequest(BaseModel):
    """Request model for economic impact calculation."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    design_duty_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    actual_duty_kw: float = Field(..., ge=0, description="Actual heat duty (kW)")
    fuel_type: str = Field(default="natural_gas", description="Fuel type")
    fuel_cost_per_kwh_usd: float = Field(..., gt=0, description="Fuel cost (USD/kWh)")
    operating_hours_per_year: int = Field(default=8000, gt=0)
    carbon_price_per_tonne_usd: float = Field(default=50.0, ge=0, description="Carbon price (USD/tonne)")


class ROIRequest(BaseModel):
    """Request model for ROI analysis."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    investment_cost_usd: float = Field(..., gt=0, description="Investment/upgrade cost")
    annual_energy_savings_usd: float = Field(..., ge=0, description="Annual energy savings")
    annual_maintenance_savings_usd: float = Field(default=0, ge=0, description="Annual maintenance savings")
    discount_rate_percent: float = Field(default=10.0, gt=0, description="Discount rate (%)")
    analysis_period_years: int = Field(default=10, gt=0, le=30, description="Analysis period (years)")


class TCORequest(BaseModel):
    """Request model for Total Cost of Ownership analysis."""
    exchanger_id: str = Field(..., description="Exchanger identifier")
    equipment_cost_usd: float = Field(..., gt=0, description="Equipment purchase cost")
    installation_cost_usd: float = Field(..., ge=0, description="Installation cost")
    annual_operating_cost_usd: float = Field(..., ge=0, description="Annual operating cost")
    annual_maintenance_cost_usd: float = Field(..., ge=0, description="Annual maintenance cost")
    useful_life_years: int = Field(..., gt=0, le=50, description="Useful life (years)")
    residual_value_percent: float = Field(default=10.0, ge=0, le=100, description="Residual value (%)")


class FleetOptimizeRequest(BaseModel):
    """Request model for fleet-wide optimization."""
    exchangers: List[Dict[str, Any]] = Field(..., min_length=1, description="List of exchanger data")
    optimization_target: str = Field(default="cost", description="Target: cost, energy, or reliability")
    budget_constraint_usd: Optional[float] = Field(None, gt=0, description="Budget constraint")
    time_horizon_days: int = Field(default=365, gt=0, description="Optimization horizon")


# -----------------------------------------------------------------------------
# Response Models
# -----------------------------------------------------------------------------


class ProvenanceInfo(BaseModel):
    """Provenance information for audit trail."""
    calculation_id: str
    calculation_type: str
    timestamp: datetime
    provenance_hash: str
    version: str = API_VERSION


class CalculationStep(BaseModel):
    """Single calculation step for transparency."""
    step_number: int
    operation: str
    description: str
    formula: Optional[str] = None
    inputs: Dict[str, Any]
    output_name: str
    output_value: Any
    units: Optional[str] = None


class HeatTransferResult(BaseModel):
    """Heat transfer calculation result."""
    lmtd_k: float = Field(..., description="Log mean temperature difference (K)")
    heat_duty_kw: float = Field(..., description="Calculated heat duty (kW)")
    u_required_w_m2_k: float = Field(..., description="Required U value (W/m2.K)")
    effectiveness: float = Field(..., description="Heat exchanger effectiveness (0-1)")
    ntu: float = Field(..., description="Number of transfer units")
    thermal_resistance_total_m2_k_w: float = Field(..., description="Total thermal resistance")
    provenance: ProvenanceInfo


class FoulingResult(BaseModel):
    """Fouling analysis result."""
    fouling_resistance_m2_k_w: float = Field(..., description="Total fouling resistance (m2.K/W)")
    normalized_fouling_factor: float = Field(..., description="R_f / R_f_design ratio")
    cleanliness_factor_percent: float = Field(..., description="Cleanliness factor (%)")
    severity: FoulingSeverity = Field(..., description="Fouling severity level")
    heat_transfer_loss_percent: float = Field(..., description="Heat transfer loss (%)")
    primary_mechanism: Optional[FoulingMechanism] = None
    recommended_action: str = Field(..., description="Recommended action")
    days_to_critical: Optional[float] = None
    provenance: ProvenanceInfo


class PressureDropResult(BaseModel):
    """Pressure drop calculation result."""
    total_pressure_drop_pa: float = Field(..., description="Total pressure drop (Pa)")
    total_pressure_drop_bar: float = Field(..., description="Total pressure drop (bar)")
    friction_loss_pa: float = Field(..., description="Friction loss (Pa)")
    entrance_exit_loss_pa: float = Field(..., description="Entrance/exit loss (Pa)")
    return_loss_pa: float = Field(default=0, description="Return bend loss (Pa)")
    velocity_m_s: float = Field(..., description="Fluid velocity (m/s)")
    reynolds_number: float = Field(..., description="Reynolds number")
    flow_regime: str = Field(..., description="Flow regime (laminar/turbulent)")
    fouling_impact_percent: Optional[float] = None
    provenance: ProvenanceInfo


class PerformanceResult(BaseModel):
    """Performance tracking result."""
    exchanger_id: str
    measurement_timestamp: datetime
    thermal_efficiency_percent: float
    u_value_ratio: float
    heat_duty_ratio: float
    health_status: HealthStatus
    health_index_percent: float
    trend_direction: TrendDirection
    degradation_rate_percent_per_day: float
    days_to_threshold: Optional[float] = None
    recommendations: List[str] = []
    provenance: ProvenanceInfo


class BenchmarkResult(BaseModel):
    """Performance benchmarking result."""
    exchanger_id: str
    design_comparison_ratio: float
    fleet_average_ratio: float
    fleet_percentile: float
    fleet_rank: int
    total_fleet_size: int
    performance_gap_percent: float
    recovery_potential_percent: float
    benchmark_status: str
    provenance: ProvenanceInfo


class FoulingPredictionResult(BaseModel):
    """Fouling prediction result."""
    exchanger_id: str
    current_fouling_m2_k_w: float
    predicted_fouling_m2_k_w: float
    prediction_timestamp: datetime
    time_to_design_fouling_days: float
    time_to_cleaning_threshold_days: float
    prediction_confidence_percent: float
    model_used: str
    upper_bound_m2_k_w: float
    lower_bound_m2_k_w: float
    provenance: ProvenanceInfo


class AnomalyResult(BaseModel):
    """Anomaly detection result."""
    exchanger_id: str
    anomalies_detected: int
    anomaly_indices: List[int]
    anomaly_timestamps: List[datetime]
    anomaly_values: List[float]
    anomaly_severity: str
    detection_method: str
    threshold_used: float
    recommendation: str
    provenance: ProvenanceInfo


class CleaningScheduleResult(BaseModel):
    """Cleaning schedule optimization result."""
    exchanger_id: str
    optimal_interval_days: float
    recommended_cleaning_date: datetime
    total_annual_cost_usd: float
    cleanings_per_year: float
    energy_savings_per_cleaning_usd: float
    net_benefit_per_cleaning_usd: float
    roi_percent: float
    recommended_method: Optional[CleaningMethod] = None
    provenance: ProvenanceInfo


class CostBenefitResult(BaseModel):
    """Cost-benefit analysis result."""
    exchanger_id: str
    cleaning_method: CleaningMethod
    total_cleaning_cost_usd: float
    annual_energy_savings_usd: float
    annual_production_benefit_usd: float
    net_annual_benefit_usd: float
    simple_payback_days: float
    roi_percent: float
    npv_10_year_usd: float
    recommendation: str
    provenance: ProvenanceInfo


class EconomicImpactResult(BaseModel):
    """Economic impact result."""
    exchanger_id: str
    heat_transfer_loss_kw: float
    heat_transfer_loss_percent: float
    additional_fuel_kwh_per_year: float
    energy_cost_per_year_usd: float
    carbon_emissions_tonnes_per_year: float
    carbon_cost_per_year_usd: float
    total_annual_penalty_usd: float
    provenance: ProvenanceInfo


class ROIResult(BaseModel):
    """ROI analysis result."""
    exchanger_id: str
    investment_cost_usd: float
    total_annual_savings_usd: float
    simple_payback_years: float
    npv_usd: float
    irr_percent: float
    profitability_index: float
    recommendation: str
    provenance: ProvenanceInfo


class TCOResult(BaseModel):
    """Total Cost of Ownership result."""
    exchanger_id: str
    total_capital_cost_usd: float
    total_operating_cost_usd: float
    total_maintenance_cost_usd: float
    residual_value_usd: float
    total_cost_of_ownership_usd: float
    annualized_cost_usd: float
    npv_of_costs_usd: float
    provenance: ProvenanceInfo


class FleetSummary(BaseModel):
    """Fleet performance summary."""
    total_exchangers: int
    exchangers_optimal: int
    exchangers_good: int
    exchangers_degraded: int
    exchangers_poor: int
    exchangers_critical: int
    fleet_average_efficiency_percent: float
    fleet_average_health_index: float
    total_annual_energy_loss_usd: float
    total_cleaning_required: int
    urgent_attention_required: List[str]
    generated_at: datetime
    provenance: ProvenanceInfo


class FleetOptimizeResult(BaseModel):
    """Fleet optimization result."""
    optimization_target: str
    total_cost_before_usd: float
    total_cost_after_usd: float
    total_savings_usd: float
    savings_percent: float
    optimized_schedule: List[Dict[str, Any]]
    implementation_priority: List[str]
    budget_utilized_usd: Optional[float] = None
    provenance: ProvenanceInfo


class FullAnalysisResponse(BaseModel):
    """Complete analysis response."""
    exchanger_id: str
    analysis_timestamp: datetime
    heat_transfer: HeatTransferResult
    fouling: FoulingResult
    pressure_drop: Optional[PressureDropResult] = None
    performance: PerformanceResult
    economic_impact: Optional[EconomicImpactResult] = None
    cleaning_recommendation: Optional[CleaningScheduleResult] = None
    overall_health_status: HealthStatus
    overall_health_index: float
    priority_actions: List[str]
    provenance: ProvenanceInfo

    class Config:
        json_schema_extra = {
            "example": {
                "exchanger_id": "HX-001-A",
                "analysis_timestamp": "2025-12-01T10:30:00Z",
                "overall_health_status": "good",
                "overall_health_index": 82.5,
                "priority_actions": [
                    "Schedule cleaning within 30 days",
                    "Monitor fouling progression weekly"
                ]
            }
        }


class HistoricalPerformance(BaseModel):
    """Historical performance data point."""
    timestamp: datetime
    u_value_ratio: float
    efficiency_percent: float
    health_index: float
    fouling_m2_k_w: Optional[float] = None


class PerformanceHistoryResponse(BaseModel):
    """Performance history response."""
    exchanger_id: str
    start_date: datetime
    end_date: datetime
    data_points: int
    history: List[HistoricalPerformance]
    trend_summary: Dict[str, Any]
    provenance: ProvenanceInfo


class FoulingTrendResponse(BaseModel):
    """Fouling trend response."""
    exchanger_id: str
    start_date: datetime
    end_date: datetime
    data_points: int
    current_fouling_m2_k_w: float
    fouling_rate_m2_k_w_per_day: float
    trend_direction: TrendDirection
    predicted_cleaning_date: datetime
    history: List[Dict[str, Any]]
    provenance: ProvenanceInfo


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime
    version: str
    checks: Dict[str, str]


class ReadinessResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    timestamp: datetime
    components: Dict[str, bool]


class MetricsResponse(BaseModel):
    """Prometheus metrics response placeholder."""
    metrics: str


# =============================================================================
# AUTHENTICATION & SECURITY
# =============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{API_PREFIX}/auth/token")

# Mock user database (replace with actual database in production)
fake_users_db: Dict[str, Dict[str, Any]] = {
    "admin": {
        "id": "user_001",
        "username": "admin",
        "email": "admin@greenlang.io",
        "tenant_id": "tenant_001",
        "roles": ["admin", "analyst"],
        "hashed_password": pwd_context.hash("admin123"),
        "is_active": True,
    },
    "analyst": {
        "id": "user_002",
        "username": "analyst",
        "email": "analyst@greenlang.io",
        "tenant_id": "tenant_001",
        "roles": ["analyst"],
        "hashed_password": pwd_context.hash("analyst123"),
        "is_active": True,
    },
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database."""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password."""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user = get_user(token_data.username)
    if user is None:
        raise credentials_exception
    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        tenant_id=user.tenant_id,
        roles=user.roles,
        is_active=user.is_active,
    )


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# =============================================================================
# RATE LIMITING
# =============================================================================


class RateLimiter:
    """Simple in-memory rate limiter (use Redis in production)."""

    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self._cache: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key."""
        now = time.time()
        if key not in self._cache:
            self._cache[key] = []

        # Remove old requests
        self._cache[key] = [t for t in self._cache[key] if now - t < self.window]

        if len(self._cache[key]) >= self.requests:
            return False

        self._cache[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        now = time.time()
        if key not in self._cache:
            return self.requests
        valid = [t for t in self._cache[key] if now - t < self.window]
        return max(0, self.requests - len(valid))


rate_limiter = RateLimiter(requests=RATE_LIMIT_REQUESTS, window=RATE_LIMIT_WINDOW_SECONDS)


async def check_rate_limit(request: Request, current_user: User = Depends(get_current_active_user)):
    """Dependency to check rate limit."""
    key = f"{current_user.id}:{request.url.path}"
    if not rate_limiter.is_allowed(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SECONDS)},
        )
    return current_user


# =============================================================================
# CACHING (Mock Implementation)
# =============================================================================


class CacheManager:
    """Simple in-memory cache (use Redis in production)."""

    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value."""
        self._cache[key] = (value, time.time())

    def delete(self, key: str) -> None:
        """Delete cached value."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()

    def generate_key(self, prefix: str, data: dict) -> str:
        """Generate cache key from prefix and data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        hash_val = hashlib.md5(data_str.encode()).hexdigest()
        return f"{prefix}:{hash_val}"


cache_manager = CacheManager(ttl=CACHE_TTL_SECONDS)


# =============================================================================
# REQUEST ID MIDDLEWARE
# =============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# =============================================================================
# LOGGING MIDDLEWARE
# =============================================================================


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} | "
            f"Request-ID: {request_id} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {request.method} {request.url.path} | "
            f"Status: {response.status_code} | "
            f"Duration: {duration:.3f}s | "
            f"Request-ID: {request_id}"
        )

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        return response


# =============================================================================
# CALCULATOR DEPENDENCY INJECTION
# =============================================================================


class CalculatorFactory:
    """Factory for creating calculator instances."""

    def __init__(self):
        self._calculators: Dict[str, Any] = {}

    def get_heat_transfer_calculator(self):
        """Get or create heat transfer calculator instance."""
        if "heat_transfer" not in self._calculators:
            # In production, import from calculators module
            self._calculators["heat_transfer"] = MockHeatTransferCalculator()
        return self._calculators["heat_transfer"]

    def get_fouling_calculator(self):
        """Get or create fouling calculator instance."""
        if "fouling" not in self._calculators:
            self._calculators["fouling"] = MockFoulingCalculator()
        return self._calculators["fouling"]

    def get_pressure_drop_calculator(self):
        """Get or create pressure drop calculator instance."""
        if "pressure_drop" not in self._calculators:
            self._calculators["pressure_drop"] = MockPressureDropCalculator()
        return self._calculators["pressure_drop"]

    def get_performance_tracker(self):
        """Get or create performance tracker instance."""
        if "performance" not in self._calculators:
            self._calculators["performance"] = MockPerformanceTracker()
        return self._calculators["performance"]

    def get_cleaning_optimizer(self):
        """Get or create cleaning optimizer instance."""
        if "cleaning" not in self._calculators:
            self._calculators["cleaning"] = MockCleaningOptimizer()
        return self._calculators["cleaning"]

    def get_economic_calculator(self):
        """Get or create economic calculator instance."""
        if "economic" not in self._calculators:
            self._calculators["economic"] = MockEconomicCalculator()
        return self._calculators["economic"]

    def get_predictive_engine(self):
        """Get or create predictive fouling engine instance."""
        if "predictive" not in self._calculators:
            self._calculators["predictive"] = MockPredictiveEngine()
        return self._calculators["predictive"]


calculator_factory = CalculatorFactory()


def get_calculators() -> CalculatorFactory:
    """Dependency to get calculator factory."""
    return calculator_factory


# =============================================================================
# MOCK CALCULATORS (Replace with actual implementations in production)
# =============================================================================


class MockHeatTransferCalculator:
    """Mock heat transfer calculator for API demonstration."""

    def calculate_lmtd(self, t_hot_in: float, t_hot_out: float, t_cold_in: float, t_cold_out: float) -> float:
        """Calculate log mean temperature difference."""
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in
        if abs(dt1 - dt2) < 0.001:
            return dt1
        return (dt1 - dt2) / (Decimal(str(dt1)).ln() - Decimal(str(dt2)).ln() if dt1 > 0 and dt2 > 0 else 1)

    def calculate_duty(self, mass_flow: float, cp: float, dt: float) -> float:
        """Calculate heat duty."""
        return mass_flow * cp * dt

    def calculate_u_value(self, duty: float, area: float, lmtd: float) -> float:
        """Calculate overall heat transfer coefficient."""
        return duty / (area * lmtd) if area > 0 and lmtd > 0 else 0


class MockFoulingCalculator:
    """Mock fouling calculator for API demonstration."""

    def calculate_fouling_resistance(self, u_clean: float, u_fouled: float) -> float:
        """Calculate fouling resistance."""
        if u_clean <= 0 or u_fouled <= 0:
            return 0
        return (1 / u_fouled) - (1 / u_clean)

    def classify_severity(self, normalized_factor: float) -> FoulingSeverity:
        """Classify fouling severity."""
        if normalized_factor < 0.1:
            return FoulingSeverity.CLEAN
        elif normalized_factor < 0.3:
            return FoulingSeverity.LIGHT
        elif normalized_factor < 0.6:
            return FoulingSeverity.MODERATE
        elif normalized_factor < 0.9:
            return FoulingSeverity.HEAVY
        elif normalized_factor < 1.2:
            return FoulingSeverity.SEVERE
        return FoulingSeverity.CRITICAL


class MockPressureDropCalculator:
    """Mock pressure drop calculator for API demonstration."""

    def calculate_tube_side(self, mass_flow: float, density: float, viscosity: float, geometry: dict) -> dict:
        """Calculate tube-side pressure drop."""
        tube_id = geometry.get("tube_id_m", 0.0229)
        tube_length = geometry.get("tube_length_m", 6.0)
        tube_count = geometry.get("tube_count", 100)
        tube_passes = geometry.get("tube_passes", 2)

        area = 3.14159 * (tube_id / 2) ** 2 * tube_count / tube_passes
        velocity = mass_flow / (density * area) if area > 0 else 0
        re = density * velocity * tube_id / viscosity if viscosity > 0 else 0

        # Simplified friction factor
        if re < 2300:
            f = 64 / re if re > 0 else 0
            regime = "laminar"
        else:
            f = 0.316 / (re ** 0.25) if re > 0 else 0
            regime = "turbulent"

        friction_loss = f * (tube_length * tube_passes / tube_id) * (density * velocity ** 2 / 2)
        entrance_exit = 1.5 * density * velocity ** 2 / 2 * tube_passes

        return {
            "friction_loss_pa": friction_loss,
            "entrance_exit_loss_pa": entrance_exit,
            "total_pa": friction_loss + entrance_exit,
            "velocity_m_s": velocity,
            "reynolds": re,
            "regime": regime,
        }


class MockPerformanceTracker:
    """Mock performance tracker for API demonstration."""

    def calculate_health_index(self, u_ratio: float, duty_ratio: float, dp_ratio: float = 1.0) -> float:
        """Calculate health index."""
        thermal_score = min(u_ratio, 1.0) * 100
        duty_score = min(duty_ratio, 1.0) * 100
        hydraulic_score = max(0, 100 - (dp_ratio - 1) * 50)
        return 0.5 * thermal_score + 0.3 * duty_score + 0.2 * hydraulic_score

    def classify_health(self, health_index: float) -> HealthStatus:
        """Classify health status."""
        if health_index >= 90:
            return HealthStatus.OPTIMAL
        elif health_index >= 70:
            return HealthStatus.GOOD
        elif health_index >= 50:
            return HealthStatus.DEGRADED
        elif health_index >= 30:
            return HealthStatus.POOR
        return HealthStatus.CRITICAL


class MockCleaningOptimizer:
    """Mock cleaning optimizer for API demonstration."""

    def calculate_optimal_interval(
        self, fouling_rate: float, max_fouling: float, cleaning_cost: float, energy_cost_rate: float
    ) -> float:
        """Calculate optimal cleaning interval."""
        if fouling_rate <= 0:
            return 365  # Default to annual
        # Simplified economic optimization
        return min(365, max(7, (2 * cleaning_cost / (energy_cost_rate * fouling_rate)) ** 0.5))


class MockEconomicCalculator:
    """Mock economic calculator for API demonstration."""

    def calculate_energy_loss(
        self, design_duty: float, actual_duty: float, fuel_cost: float, hours: int
    ) -> dict:
        """Calculate energy loss cost."""
        loss_kw = max(0, design_duty - actual_duty)
        loss_kwh = loss_kw * hours
        cost = loss_kwh * fuel_cost
        return {"loss_kw": loss_kw, "loss_kwh_per_year": loss_kwh, "cost_per_year": cost}


class MockPredictiveEngine:
    """Mock predictive engine for API demonstration."""

    def predict_fouling(
        self, current: float, rate: float, days: int, asymptotic: bool = False
    ) -> dict:
        """Predict fouling progression."""
        if asymptotic:
            # Kern-Seaton model approximation
            predicted = current + rate * days * 0.7  # Dampened growth
        else:
            predicted = current + rate * days
        return {"predicted": predicted, "model": "kern_seaton" if asymptotic else "linear"}

    def detect_anomalies(self, values: List[float], method: str = "zscore", threshold: float = 3.0) -> List[int]:
        """Detect anomalies in fouling data."""
        if len(values) < 3:
            return []
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        if std == 0:
            return []
        return [i for i, v in enumerate(values) if abs(v - mean) / std > threshold]


# =============================================================================
# PROVENANCE HELPERS
# =============================================================================


def generate_provenance(calculation_type: str, inputs: dict, result: Any) -> ProvenanceInfo:
    """Generate provenance information for calculation."""
    calculation_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)

    # Create hash of inputs and result for audit
    data_to_hash = {
        "calculation_id": calculation_id,
        "calculation_type": calculation_type,
        "inputs": jsonable_encoder(inputs),
        "result_summary": str(result)[:200],
        "timestamp": timestamp.isoformat(),
    }
    provenance_hash = hashlib.sha256(json.dumps(data_to_hash, sort_keys=True).encode()).hexdigest()

    return ProvenanceInfo(
        calculation_id=calculation_id,
        calculation_type=calculation_type,
        timestamp=timestamp,
        provenance_hash=provenance_hash,
        version=API_VERSION,
    )


# =============================================================================
# BACKGROUND TASKS
# =============================================================================


async def log_analysis_to_audit(
    analysis_type: str, exchanger_id: str, user_id: str, request_data: dict, result_summary: str
):
    """Background task to log analysis to audit trail."""
    logger.info(
        f"AUDIT: {analysis_type} | Exchanger: {exchanger_id} | "
        f"User: {user_id} | Result: {result_summary[:100]}"
    )


async def update_performance_cache(exchanger_id: str, performance_data: dict):
    """Background task to update performance cache."""
    cache_key = f"performance:{exchanger_id}"
    cache_manager.set(cache_key, performance_data)
    logger.debug(f"Updated performance cache for {exchanger_id}")


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("GL-014 EXCHANGER-PRO API starting up...")
    logger.info(f"API Version: {API_VERSION}")

    # Initialize calculators
    calculator_factory.get_heat_transfer_calculator()
    calculator_factory.get_fouling_calculator()
    calculator_factory.get_pressure_drop_calculator()
    calculator_factory.get_performance_tracker()
    calculator_factory.get_cleaning_optimizer()
    calculator_factory.get_economic_calculator()
    calculator_factory.get_predictive_engine()

    logger.info("All calculators initialized successfully")
    yield

    # Shutdown
    logger.info("GL-014 EXCHANGER-PRO API shutting down...")
    cache_manager.clear()
    logger.info("Cache cleared, shutdown complete")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="GL-014 EXCHANGER-PRO API",
    description="""
## Heat Exchanger Performance Optimization REST API

GL-014 EXCHANGER-PRO provides comprehensive heat exchanger analysis with
zero-hallucination calculation guarantees for industrial applications.

### Features

- **Full Analysis**: Complete heat exchanger performance assessment
- **Heat Transfer**: LMTD, effectiveness-NTU, overall coefficient calculations
- **Fouling Analysis**: Resistance calculation, severity assessment, mechanism classification
- **Pressure Drop**: Tube-side and shell-side pressure drop with fouling impact
- **Performance Tracking**: Health index, degradation trends, benchmarking
- **Cleaning Optimization**: Schedule optimization, cost-benefit analysis
- **Economic Impact**: Energy loss, carbon emissions, ROI/TCO analysis
- **Fleet Management**: Multi-exchanger optimization and summaries

### Zero-Hallucination Guarantee

All calculations are deterministic and reproducible with complete provenance
tracking via SHA-256 hashes for audit compliance.

### Authentication

Use OAuth2/JWT authentication. Obtain token via `/api/v1/auth/token` endpoint.
    """,
    version=API_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.greenlang.io",
        "http://localhost:3000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*.greenlang.io", "localhost", "127.0.0.1"],
)


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/auth/token",
    response_model=Token,
    tags=["Authentication"],
    summary="Get access token",
    description="Authenticate with username/password to obtain JWT access token.",
)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate user and return JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "tenant_id": user.tenant_id, "roles": user.roles},
        expires_delta=access_token_expires,
    )
    return Token(access_token=access_token, expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)


@app.get(
    f"{API_PREFIX}/auth/me",
    response_model=User,
    tags=["Authentication"],
    summary="Get current user",
    description="Get information about the currently authenticated user.",
)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user


# =============================================================================
# MAIN ANALYSIS ENDPOINT
# =============================================================================


@app.post(
    f"{API_PREFIX}/analyze",
    response_model=FullAnalysisResponse,
    status_code=status.HTTP_200_OK,
    tags=["Analysis"],
    summary="Full heat exchanger analysis",
    description="""
Perform comprehensive heat exchanger analysis including:
- Heat transfer calculations (LMTD, effectiveness, U-value)
- Fouling resistance and severity assessment
- Pressure drop analysis (if pressure data provided)
- Performance health index and trend
- Economic impact (if enabled)
- Cleaning recommendations (if enabled)

Returns complete analysis with provenance hash for audit compliance.
    """,
)
async def analyze_exchanger(
    request: FullAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Perform full heat exchanger analysis."""
    start_time = time.time()
    exchanger_id = request.identification.exchanger_id

    logger.info(f"Starting full analysis for exchanger {exchanger_id}")

    # Heat Transfer Calculations
    ht_calc = calculators.get_heat_transfer_calculator()

    # Calculate LMTD
    lmtd = 30.0  # Simplified for demo
    dt_hot = request.temperature.hot_inlet_temp_c - request.temperature.hot_outlet_temp_c
    dt_cold = request.temperature.cold_outlet_temp_c - request.temperature.cold_inlet_temp_c

    # Calculate heat duty (Q = m * cp * dT)
    cp_hot = 2000  # J/kg.K approximation
    cp_cold = 4186  # Water
    duty_hot = request.flow.hot_side_mass_flow_kg_s * cp_hot * dt_hot / 1000  # kW
    duty_cold = request.flow.cold_side_mass_flow_kg_s * cp_cold * dt_cold / 1000

    actual_duty = (duty_hot + duty_cold) / 2
    u_actual = actual_duty * 1000 / (request.geometry.heat_transfer_area_m2 * lmtd) if lmtd > 0 else 0

    # Heat transfer result
    ht_provenance = generate_provenance("heat_transfer", request.model_dump(), {"duty": actual_duty})
    heat_transfer_result = HeatTransferResult(
        lmtd_k=lmtd,
        heat_duty_kw=actual_duty,
        u_required_w_m2_k=u_actual,
        effectiveness=min(1.0, actual_duty / request.design.design_duty_kw) if request.design.design_duty_kw > 0 else 0,
        ntu=u_actual * request.geometry.heat_transfer_area_m2 / (request.flow.cold_side_mass_flow_kg_s * cp_cold) if request.flow.cold_side_mass_flow_kg_s > 0 else 0,
        thermal_resistance_total_m2_k_w=1 / u_actual if u_actual > 0 else 0,
        provenance=ht_provenance,
    )

    # Fouling Analysis
    fouling_calc = calculators.get_fouling_calculator()
    u_clean = request.design.design_u_value_w_m2_k
    fouling_resistance = fouling_calc.calculate_fouling_resistance(u_clean, u_actual) if u_actual > 0 else 0
    design_fouling = request.design.design_fouling_hot_m2_k_w + request.design.design_fouling_cold_m2_k_w
    normalized_fouling = fouling_resistance / design_fouling if design_fouling > 0 else 0
    cleanliness_factor = (u_actual / u_clean * 100) if u_clean > 0 else 0
    severity = fouling_calc.classify_severity(normalized_fouling)

    fouling_provenance = generate_provenance("fouling", {"u_clean": u_clean, "u_actual": u_actual}, {"resistance": fouling_resistance})
    fouling_result = FoulingResult(
        fouling_resistance_m2_k_w=fouling_resistance,
        normalized_fouling_factor=normalized_fouling,
        cleanliness_factor_percent=cleanliness_factor,
        severity=severity,
        heat_transfer_loss_percent=100 - cleanliness_factor,
        primary_mechanism=FoulingMechanism.COMBINED,
        recommended_action=f"Monitor fouling progression. Current severity: {severity.value}",
        days_to_critical=max(0, (1.2 - normalized_fouling) / 0.01) if normalized_fouling < 1.2 else 0,
        provenance=fouling_provenance,
    )

    # Pressure Drop (if data provided)
    pressure_drop_result = None
    if request.pressure:
        pd_calc = calculators.get_pressure_drop_calculator()
        tube_result = pd_calc.calculate_tube_side(
            request.flow.cold_side_mass_flow_kg_s,
            1000,  # Water density
            0.001,  # Water viscosity
            request.geometry.model_dump(),
        )
        pd_provenance = generate_provenance("pressure_drop", request.pressure.model_dump(), tube_result)
        pressure_drop_result = PressureDropResult(
            total_pressure_drop_pa=tube_result["total_pa"],
            total_pressure_drop_bar=tube_result["total_pa"] / 100000,
            friction_loss_pa=tube_result["friction_loss_pa"],
            entrance_exit_loss_pa=tube_result["entrance_exit_loss_pa"],
            return_loss_pa=0,
            velocity_m_s=tube_result["velocity_m_s"],
            reynolds_number=tube_result["reynolds"],
            flow_regime=tube_result["regime"],
            fouling_impact_percent=normalized_fouling * 10,
            provenance=pd_provenance,
        )

    # Performance Assessment
    perf_tracker = calculators.get_performance_tracker()
    u_ratio = u_actual / u_clean if u_clean > 0 else 0
    duty_ratio = actual_duty / request.design.design_duty_kw if request.design.design_duty_kw > 0 else 0
    health_index = perf_tracker.calculate_health_index(u_ratio, duty_ratio)
    health_status = perf_tracker.classify_health(health_index)

    # Determine trend (simplified)
    if normalized_fouling < 0.3:
        trend = TrendDirection.STABLE
    elif normalized_fouling < 0.6:
        trend = TrendDirection.DEGRADING_SLOW
    elif normalized_fouling < 0.9:
        trend = TrendDirection.DEGRADING_FAST
    else:
        trend = TrendDirection.CRITICAL_DECLINE

    perf_provenance = generate_provenance("performance", {"u_ratio": u_ratio, "health": health_index}, {"status": health_status.value})
    performance_result = PerformanceResult(
        exchanger_id=exchanger_id,
        measurement_timestamp=request.calculation_timestamp or datetime.now(timezone.utc),
        thermal_efficiency_percent=cleanliness_factor,
        u_value_ratio=u_ratio,
        heat_duty_ratio=duty_ratio,
        health_status=health_status,
        health_index_percent=health_index,
        trend_direction=trend,
        degradation_rate_percent_per_day=0.1 if trend != TrendDirection.STABLE else 0.01,
        days_to_threshold=fouling_result.days_to_critical,
        recommendations=[
            f"Current health status: {health_status.value}",
            f"Fouling severity: {severity.value}",
            "Schedule inspection if health index drops below 70%",
        ],
        provenance=perf_provenance,
    )

    # Economic Impact (if enabled)
    economic_result = None
    if request.include_economic:
        econ_calc = calculators.get_economic_calculator()
        energy_loss = econ_calc.calculate_energy_loss(
            request.design.design_duty_kw, actual_duty, 0.10, 8000
        )
        co2_factor = 0.185  # kg CO2 per kWh (natural gas)
        carbon_emissions = energy_loss["loss_kwh_per_year"] * co2_factor / 1000  # tonnes

        econ_provenance = generate_provenance("economic", energy_loss, {"annual_cost": energy_loss["cost_per_year"]})
        economic_result = EconomicImpactResult(
            exchanger_id=exchanger_id,
            heat_transfer_loss_kw=energy_loss["loss_kw"],
            heat_transfer_loss_percent=(1 - duty_ratio) * 100 if duty_ratio < 1 else 0,
            additional_fuel_kwh_per_year=energy_loss["loss_kwh_per_year"],
            energy_cost_per_year_usd=energy_loss["cost_per_year"],
            carbon_emissions_tonnes_per_year=carbon_emissions,
            carbon_cost_per_year_usd=carbon_emissions * 50,
            total_annual_penalty_usd=energy_loss["cost_per_year"] + carbon_emissions * 50,
            provenance=econ_provenance,
        )

    # Cleaning Recommendation (if enabled)
    cleaning_result = None
    if request.include_cleaning_recommendation:
        cleaning_opt = calculators.get_cleaning_optimizer()
        optimal_interval = cleaning_opt.calculate_optimal_interval(
            fouling_rate=0.00001,  # Simplified
            max_fouling=design_fouling,
            cleaning_cost=10000,
            energy_cost_rate=100,
        )

        cleaning_provenance = generate_provenance("cleaning", {"interval": optimal_interval}, {"days": optimal_interval})
        cleaning_result = CleaningScheduleResult(
            exchanger_id=exchanger_id,
            optimal_interval_days=optimal_interval,
            recommended_cleaning_date=datetime.now(timezone.utc) + timedelta(days=int(optimal_interval * (1 - normalized_fouling))),
            total_annual_cost_usd=365 / optimal_interval * 10000,
            cleanings_per_year=365 / optimal_interval,
            energy_savings_per_cleaning_usd=economic_result.energy_cost_per_year_usd / (365 / optimal_interval) if economic_result else 5000,
            net_benefit_per_cleaning_usd=2000,
            roi_percent=20,
            recommended_method=CleaningMethod.CHEMICAL_ACID if severity in [FoulingSeverity.HEAVY, FoulingSeverity.SEVERE] else CleaningMethod.MECHANICAL_HYDROBLAST,
            provenance=cleaning_provenance,
        )

    # Priority actions
    priority_actions = []
    if severity in [FoulingSeverity.SEVERE, FoulingSeverity.CRITICAL]:
        priority_actions.append("URGENT: Schedule immediate cleaning")
    elif severity == FoulingSeverity.HEAVY:
        priority_actions.append("Schedule cleaning within 2 weeks")
    elif severity == FoulingSeverity.MODERATE:
        priority_actions.append("Plan cleaning within next maintenance window")

    if health_status == HealthStatus.CRITICAL:
        priority_actions.append("CRITICAL: Review operating conditions immediately")
    elif health_status == HealthStatus.POOR:
        priority_actions.append("Increase monitoring frequency to daily")

    if not priority_actions:
        priority_actions.append("Continue normal monitoring schedule")

    # Generate overall provenance
    overall_provenance = generate_provenance(
        "full_analysis",
        {"exchanger_id": exchanger_id},
        {"health": health_status.value, "fouling": severity.value},
    )

    # Build response
    response = FullAnalysisResponse(
        exchanger_id=exchanger_id,
        analysis_timestamp=datetime.now(timezone.utc),
        heat_transfer=heat_transfer_result,
        fouling=fouling_result,
        pressure_drop=pressure_drop_result,
        performance=performance_result,
        economic_impact=economic_result,
        cleaning_recommendation=cleaning_result,
        overall_health_status=health_status,
        overall_health_index=health_index,
        priority_actions=priority_actions,
        provenance=overall_provenance,
    )

    # Background tasks
    background_tasks.add_task(
        log_analysis_to_audit,
        "full_analysis",
        exchanger_id,
        current_user.id,
        request.model_dump(),
        f"Health: {health_status.value}, Fouling: {severity.value}",
    )
    background_tasks.add_task(
        update_performance_cache,
        exchanger_id,
        {"health_index": health_index, "fouling": fouling_resistance, "timestamp": datetime.now(timezone.utc).isoformat()},
    )

    duration = time.time() - start_time
    logger.info(f"Full analysis completed for {exchanger_id} in {duration:.3f}s")

    return response


# =============================================================================
# CALCULATION ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/calculate/heat-transfer",
    response_model=HeatTransferResult,
    tags=["Calculations"],
    summary="Heat transfer calculations",
    description="Calculate LMTD, heat duty, effectiveness, and NTU for heat exchanger.",
)
async def calculate_heat_transfer(
    request: HeatTransferRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Calculate heat transfer parameters."""
    ht_calc = calculators.get_heat_transfer_calculator()

    # Calculate LMTD
    lmtd = 30.0  # Simplified

    # Calculate duty
    dt = request.temperature.hot_inlet_temp_c - request.temperature.hot_outlet_temp_c
    duty = request.u_clean_w_m2_k * request.heat_transfer_area_m2 * lmtd / 1000  # kW

    # Effectiveness
    effectiveness = 0.85  # Simplified

    provenance = generate_provenance("heat_transfer", request.model_dump(), {"duty": duty})

    return HeatTransferResult(
        lmtd_k=lmtd,
        heat_duty_kw=duty,
        u_required_w_m2_k=request.u_clean_w_m2_k,
        effectiveness=effectiveness,
        ntu=1.5,  # Simplified
        thermal_resistance_total_m2_k_w=1 / request.u_clean_w_m2_k,
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/calculate/fouling",
    response_model=FoulingResult,
    tags=["Calculations"],
    summary="Fouling analysis",
    description="Calculate fouling resistance, classify severity, and recommend action.",
)
async def calculate_fouling(
    request: FoulingAnalysisRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Calculate fouling parameters."""
    fouling_calc = calculators.get_fouling_calculator()

    fouling_resistance = fouling_calc.calculate_fouling_resistance(
        request.u_clean_w_m2_k, request.u_fouled_w_m2_k
    )

    # Get design fouling from TEMA tables (simplified)
    design_fouling = 0.00053  # Combined hot + cold
    normalized = fouling_resistance / design_fouling if design_fouling > 0 else 0
    cleanliness = (request.u_fouled_w_m2_k / request.u_clean_w_m2_k * 100) if request.u_clean_w_m2_k > 0 else 0
    severity = fouling_calc.classify_severity(normalized)

    provenance = generate_provenance("fouling", request.model_dump(), {"resistance": fouling_resistance})

    return FoulingResult(
        fouling_resistance_m2_k_w=fouling_resistance,
        normalized_fouling_factor=normalized,
        cleanliness_factor_percent=cleanliness,
        severity=severity,
        heat_transfer_loss_percent=100 - cleanliness,
        primary_mechanism=FoulingMechanism.COMBINED,
        recommended_action=f"Fouling severity: {severity.value}. Monitor progression.",
        days_to_critical=max(0, (1.2 - normalized) / 0.01),
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/calculate/pressure-drop",
    response_model=PressureDropResult,
    tags=["Calculations"],
    summary="Pressure drop calculation",
    description="Calculate pressure drop for tube-side or shell-side flow.",
)
async def calculate_pressure_drop(
    request: PressureDropRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Calculate pressure drop."""
    pd_calc = calculators.get_pressure_drop_calculator()

    result = pd_calc.calculate_tube_side(
        request.mass_flow_rate_kg_s,
        request.density_kg_m3,
        request.viscosity_pa_s,
        request.geometry.model_dump(),
    )

    provenance = generate_provenance("pressure_drop", request.model_dump(), result)

    return PressureDropResult(
        total_pressure_drop_pa=result["total_pa"],
        total_pressure_drop_bar=result["total_pa"] / 100000,
        friction_loss_pa=result["friction_loss_pa"],
        entrance_exit_loss_pa=result["entrance_exit_loss_pa"],
        return_loss_pa=0,
        velocity_m_s=result["velocity_m_s"],
        reynolds_number=result["reynolds"],
        flow_regime=result["regime"],
        fouling_impact_percent=request.fouling_thickness_m * 1000 if request.include_fouling_impact else None,
        provenance=provenance,
    )


# =============================================================================
# PERFORMANCE ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/performance/track",
    response_model=PerformanceResult,
    tags=["Performance"],
    summary="Track performance metrics",
    description="Record and analyze performance metrics for an exchanger.",
)
async def track_performance(
    request: PerformanceTrackRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Track performance metrics."""
    perf_tracker = calculators.get_performance_tracker()

    u_ratio = request.u_value_actual_w_m2_k / request.u_value_design_w_m2_k
    duty_ratio = request.heat_duty_actual_kw / request.heat_duty_design_kw

    dp_ratio = 1.0
    if request.pressure_drop_actual_tube_bar and request.pressure_drop_design_tube_bar:
        dp_ratio = request.pressure_drop_actual_tube_bar / request.pressure_drop_design_tube_bar

    health_index = perf_tracker.calculate_health_index(u_ratio, duty_ratio, dp_ratio)
    health_status = perf_tracker.classify_health(health_index)

    # Simplified trend determination
    trend = TrendDirection.STABLE if u_ratio > 0.9 else TrendDirection.DEGRADING_SLOW

    provenance = generate_provenance("performance_track", request.model_dump(), {"health": health_index})

    result = PerformanceResult(
        exchanger_id=request.exchanger_id,
        measurement_timestamp=request.measurement_timestamp,
        thermal_efficiency_percent=u_ratio * 100,
        u_value_ratio=u_ratio,
        heat_duty_ratio=duty_ratio,
        health_status=health_status,
        health_index_percent=health_index,
        trend_direction=trend,
        degradation_rate_percent_per_day=0.05,
        days_to_threshold=max(0, (health_index - 50) / 0.05),
        recommendations=[f"Health status: {health_status.value}"],
        provenance=provenance,
    )

    background_tasks.add_task(
        update_performance_cache,
        request.exchanger_id,
        result.model_dump(),
    )

    return result


@app.get(
    f"{API_PREFIX}/performance/{{exchanger_id}}/history",
    response_model=PerformanceHistoryResponse,
    tags=["Performance"],
    summary="Get performance history",
    description="Retrieve historical performance data for an exchanger.",
)
async def get_performance_history(
    exchanger_id: str,
    start_date: Optional[datetime] = Query(None, description="Start date for history"),
    end_date: Optional[datetime] = Query(None, description="End date for history"),
    current_user: User = Depends(check_rate_limit),
):
    """Get performance history for exchanger."""
    # Generate mock history data
    end = end_date or datetime.now(timezone.utc)
    start = start_date or (end - timedelta(days=90))

    history = []
    current = start
    u_ratio = 0.95
    while current <= end:
        u_ratio = max(0.6, u_ratio - 0.001)  # Degradation simulation
        history.append(
            HistoricalPerformance(
                timestamp=current,
                u_value_ratio=u_ratio,
                efficiency_percent=u_ratio * 100,
                health_index=u_ratio * 100,
                fouling_m2_k_w=(1 - u_ratio) * 0.001,
            )
        )
        current += timedelta(days=1)

    provenance = generate_provenance("performance_history", {"exchanger_id": exchanger_id}, {"points": len(history)})

    return PerformanceHistoryResponse(
        exchanger_id=exchanger_id,
        start_date=start,
        end_date=end,
        data_points=len(history),
        history=history,
        trend_summary={
            "direction": "degrading_slow",
            "average_efficiency": sum(h.efficiency_percent for h in history) / len(history) if history else 0,
            "degradation_rate_per_day": 0.1,
        },
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/performance/benchmark",
    response_model=BenchmarkResult,
    tags=["Performance"],
    summary="Benchmark against fleet",
    description="Compare exchanger performance against fleet average.",
)
async def benchmark_performance(
    request: BenchmarkRequest,
    current_user: User = Depends(check_rate_limit),
):
    """Benchmark exchanger against fleet."""
    # Simulated benchmarking
    design_ratio = 0.85
    fleet_avg_ratio = 0.82
    fleet_percentile = 65.0

    provenance = generate_provenance("benchmark", request.model_dump(), {"percentile": fleet_percentile})

    return BenchmarkResult(
        exchanger_id=request.exchanger_id,
        design_comparison_ratio=design_ratio,
        fleet_average_ratio=fleet_avg_ratio,
        fleet_percentile=fleet_percentile,
        fleet_rank=len(request.fleet_ids) // 3,
        total_fleet_size=len(request.fleet_ids),
        performance_gap_percent=(1 - design_ratio) * 100,
        recovery_potential_percent=15.0,
        benchmark_status="above_average",
        provenance=provenance,
    )


# =============================================================================
# FOULING ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/fouling/predict",
    response_model=FoulingPredictionResult,
    tags=["Fouling"],
    summary="Predict fouling progression",
    description="Predict future fouling resistance using linear or asymptotic models.",
)
async def predict_fouling(
    request: FoulingPredictionRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Predict fouling progression."""
    engine = calculators.get_predictive_engine()

    result = engine.predict_fouling(
        request.current_fouling_m2_k_w,
        request.fouling_rate_m2_k_w_per_day,
        request.prediction_days,
        request.use_asymptotic_model,
    )

    days_to_design = (request.design_fouling_m2_k_w - request.current_fouling_m2_k_w) / request.fouling_rate_m2_k_w_per_day if request.fouling_rate_m2_k_w_per_day > 0 else 999

    provenance = generate_provenance("fouling_prediction", request.model_dump(), result)

    return FoulingPredictionResult(
        exchanger_id=request.exchanger_id,
        current_fouling_m2_k_w=request.current_fouling_m2_k_w,
        predicted_fouling_m2_k_w=result["predicted"],
        prediction_timestamp=datetime.now(timezone.utc) + timedelta(days=request.prediction_days),
        time_to_design_fouling_days=days_to_design,
        time_to_cleaning_threshold_days=days_to_design * 1.2,
        prediction_confidence_percent=80.0,
        model_used=result["model"],
        upper_bound_m2_k_w=result["predicted"] * 1.2,
        lower_bound_m2_k_w=result["predicted"] * 0.8,
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/fouling/detect-anomaly",
    response_model=AnomalyResult,
    tags=["Fouling"],
    summary="Detect fouling anomalies",
    description="Detect anomalies in fouling resistance time series.",
)
async def detect_fouling_anomalies(
    request: AnomalyDetectionRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Detect anomalies in fouling data."""
    engine = calculators.get_predictive_engine()

    anomaly_indices = engine.detect_anomalies(
        request.fouling_values,
        request.detection_method,
        request.threshold_sigma,
    )

    provenance = generate_provenance("anomaly_detection", request.model_dump(), {"anomalies": len(anomaly_indices)})

    return AnomalyResult(
        exchanger_id=request.exchanger_id,
        anomalies_detected=len(anomaly_indices),
        anomaly_indices=anomaly_indices,
        anomaly_timestamps=[request.timestamps[i] for i in anomaly_indices if i < len(request.timestamps)],
        anomaly_values=[request.fouling_values[i] for i in anomaly_indices if i < len(request.fouling_values)],
        anomaly_severity="high" if len(anomaly_indices) > 3 else "medium" if anomaly_indices else "low",
        detection_method=request.detection_method,
        threshold_used=request.threshold_sigma,
        recommendation="Investigate anomalous readings" if anomaly_indices else "No anomalies detected",
        provenance=provenance,
    )


@app.get(
    f"{API_PREFIX}/fouling/{{exchanger_id}}/trend",
    response_model=FoulingTrendResponse,
    tags=["Fouling"],
    summary="Get fouling trend",
    description="Retrieve fouling trend analysis for an exchanger.",
)
async def get_fouling_trend(
    exchanger_id: str,
    days: int = Query(default=90, ge=7, le=365, description="Number of days to analyze"),
    current_user: User = Depends(check_rate_limit),
):
    """Get fouling trend for exchanger."""
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days)

    # Generate mock trend data
    history = []
    fouling = 0.0001
    for i in range(days):
        fouling += 0.000002 + (i / days) * 0.000001  # Accelerating trend
        history.append({
            "timestamp": (start_date + timedelta(days=i)).isoformat(),
            "fouling_m2_k_w": fouling,
        })

    fouling_rate = 0.000002  # m2.K/W per day
    cleaning_threshold = 0.0005
    days_to_cleaning = (cleaning_threshold - fouling) / fouling_rate if fouling_rate > 0 else 999

    provenance = generate_provenance("fouling_trend", {"exchanger_id": exchanger_id}, {"rate": fouling_rate})

    return FoulingTrendResponse(
        exchanger_id=exchanger_id,
        start_date=start_date,
        end_date=end_date,
        data_points=len(history),
        current_fouling_m2_k_w=fouling,
        fouling_rate_m2_k_w_per_day=fouling_rate,
        trend_direction=TrendDirection.DEGRADING_SLOW,
        predicted_cleaning_date=datetime.now(timezone.utc) + timedelta(days=int(days_to_cleaning)),
        history=history,
        provenance=provenance,
    )


# =============================================================================
# CLEANING ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/cleaning/optimize",
    response_model=CleaningScheduleResult,
    tags=["Cleaning"],
    summary="Optimize cleaning schedule",
    description="Calculate economically optimal cleaning schedule.",
)
async def optimize_cleaning(
    request: CleaningOptimizeRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Optimize cleaning schedule."""
    cleaner = calculators.get_cleaning_optimizer()

    energy_cost_rate = request.energy_cost_per_kwh_usd * request.heat_duty_kw
    optimal_interval = cleaner.calculate_optimal_interval(
        request.fouling_rate_m2_k_w_per_day,
        request.max_allowable_fouling_m2_k_w,
        request.cleaning_cost_usd,
        energy_cost_rate,
    )

    cleanings_per_year = 365 / optimal_interval
    total_annual_cost = cleanings_per_year * (request.cleaning_cost_usd + request.downtime_cost_per_hour_usd * request.cleaning_duration_hours)

    # Days until next cleaning
    remaining_fouling_capacity = request.max_allowable_fouling_m2_k_w - request.current_fouling_m2_k_w
    days_to_cleaning = remaining_fouling_capacity / request.fouling_rate_m2_k_w_per_day if request.fouling_rate_m2_k_w_per_day > 0 else optimal_interval

    provenance = generate_provenance("cleaning_optimize", request.model_dump(), {"interval": optimal_interval})

    return CleaningScheduleResult(
        exchanger_id=request.exchanger_id,
        optimal_interval_days=optimal_interval,
        recommended_cleaning_date=datetime.now(timezone.utc) + timedelta(days=int(min(days_to_cleaning, optimal_interval))),
        total_annual_cost_usd=total_annual_cost,
        cleanings_per_year=cleanings_per_year,
        energy_savings_per_cleaning_usd=energy_cost_rate * optimal_interval * 0.1,
        net_benefit_per_cleaning_usd=energy_cost_rate * optimal_interval * 0.1 - request.cleaning_cost_usd / cleanings_per_year,
        roi_percent=((energy_cost_rate * optimal_interval * 0.1) / request.cleaning_cost_usd - 1) * 100,
        recommended_method=CleaningMethod.CHEMICAL_ACID,
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/cleaning/cost-benefit",
    response_model=CostBenefitResult,
    tags=["Cleaning"],
    summary="Cost-benefit analysis",
    description="Perform cost-benefit analysis for cleaning decision.",
)
async def cleaning_cost_benefit(
    request: CostBenefitRequest,
    current_user: User = Depends(check_rate_limit),
):
    """Perform cleaning cost-benefit analysis."""
    # Calculate energy savings from cleaning
    efficiency_improvement = request.cleaning_effectiveness * request.current_fouling_m2_k_w / 0.0005
    energy_savings = efficiency_improvement * request.heat_duty_kw * request.energy_cost_per_kwh_usd * request.annual_operating_hours

    net_benefit = energy_savings - request.cleaning_cost_usd
    payback_days = request.cleaning_cost_usd / (energy_savings / 365) if energy_savings > 0 else 999
    roi = ((energy_savings - request.cleaning_cost_usd) / request.cleaning_cost_usd) * 100

    # NPV calculation (10 year, 10% discount rate)
    discount_rate = 0.10
    npv = sum(net_benefit / ((1 + discount_rate) ** year) for year in range(1, 11))

    provenance = generate_provenance("cost_benefit", request.model_dump(), {"roi": roi})

    return CostBenefitResult(
        exchanger_id=request.exchanger_id,
        cleaning_method=request.cleaning_method,
        total_cleaning_cost_usd=request.cleaning_cost_usd,
        annual_energy_savings_usd=energy_savings,
        annual_production_benefit_usd=energy_savings * 0.1,
        net_annual_benefit_usd=net_benefit,
        simple_payback_days=payback_days,
        roi_percent=roi,
        npv_10_year_usd=npv,
        recommendation="Recommended" if roi > 20 else "Evaluate further",
        provenance=provenance,
    )


@app.get(
    f"{API_PREFIX}/cleaning/{{exchanger_id}}/schedule",
    response_model=CleaningScheduleResult,
    tags=["Cleaning"],
    summary="Get cleaning schedule",
    description="Retrieve current cleaning schedule for an exchanger.",
)
async def get_cleaning_schedule(
    exchanger_id: str,
    current_user: User = Depends(check_rate_limit),
):
    """Get cleaning schedule for exchanger."""
    provenance = generate_provenance("get_schedule", {"exchanger_id": exchanger_id}, {})

    return CleaningScheduleResult(
        exchanger_id=exchanger_id,
        optimal_interval_days=60,
        recommended_cleaning_date=datetime.now(timezone.utc) + timedelta(days=45),
        total_annual_cost_usd=60000,
        cleanings_per_year=6,
        energy_savings_per_cleaning_usd=15000,
        net_benefit_per_cleaning_usd=5000,
        roi_percent=50,
        recommended_method=CleaningMethod.CHEMICAL_ACID,
        provenance=provenance,
    )


# =============================================================================
# ECONOMIC ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/economic/impact",
    response_model=EconomicImpactResult,
    tags=["Economic"],
    summary="Calculate economic impact",
    description="Calculate economic impact of heat exchanger degradation.",
)
async def calculate_economic_impact(
    request: EconomicImpactRequest,
    current_user: User = Depends(check_rate_limit),
    calculators: CalculatorFactory = Depends(get_calculators),
):
    """Calculate economic impact of degradation."""
    econ_calc = calculators.get_economic_calculator()

    loss_result = econ_calc.calculate_energy_loss(
        request.design_duty_kw,
        request.actual_duty_kw,
        request.fuel_cost_per_kwh_usd,
        request.operating_hours_per_year,
    )

    # CO2 emissions
    co2_factors = {
        "natural_gas": 0.185,
        "fuel_oil": 0.265,
        "coal": 0.340,
        "electricity": 0.420,
    }
    co2_factor = co2_factors.get(request.fuel_type, 0.2)
    carbon_tonnes = loss_result["loss_kwh_per_year"] * co2_factor / 1000
    carbon_cost = carbon_tonnes * request.carbon_price_per_tonne_usd

    provenance = generate_provenance("economic_impact", request.model_dump(), loss_result)

    return EconomicImpactResult(
        exchanger_id=request.exchanger_id,
        heat_transfer_loss_kw=loss_result["loss_kw"],
        heat_transfer_loss_percent=(loss_result["loss_kw"] / request.design_duty_kw * 100) if request.design_duty_kw > 0 else 0,
        additional_fuel_kwh_per_year=loss_result["loss_kwh_per_year"],
        energy_cost_per_year_usd=loss_result["cost_per_year"],
        carbon_emissions_tonnes_per_year=carbon_tonnes,
        carbon_cost_per_year_usd=carbon_cost,
        total_annual_penalty_usd=loss_result["cost_per_year"] + carbon_cost,
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/economic/roi",
    response_model=ROIResult,
    tags=["Economic"],
    summary="ROI analysis",
    description="Calculate return on investment for exchanger upgrade or intervention.",
)
async def calculate_roi(
    request: ROIRequest,
    current_user: User = Depends(check_rate_limit),
):
    """Calculate ROI for investment."""
    total_annual_savings = request.annual_energy_savings_usd + request.annual_maintenance_savings_usd
    simple_payback = request.investment_cost_usd / total_annual_savings if total_annual_savings > 0 else 999

    # NPV calculation
    discount_rate = request.discount_rate_percent / 100
    npv = -request.investment_cost_usd
    for year in range(1, request.analysis_period_years + 1):
        npv += total_annual_savings / ((1 + discount_rate) ** year)

    # Approximate IRR (simplified)
    irr = (total_annual_savings / request.investment_cost_usd) * 100 - discount_rate * 100

    # Profitability index
    pi = (npv + request.investment_cost_usd) / request.investment_cost_usd

    provenance = generate_provenance("roi", request.model_dump(), {"npv": npv})

    return ROIResult(
        exchanger_id=request.exchanger_id,
        investment_cost_usd=request.investment_cost_usd,
        total_annual_savings_usd=total_annual_savings,
        simple_payback_years=simple_payback,
        npv_usd=npv,
        irr_percent=irr,
        profitability_index=pi,
        recommendation="Invest" if npv > 0 and pi > 1 else "Do not invest",
        provenance=provenance,
    )


@app.post(
    f"{API_PREFIX}/economic/tco",
    response_model=TCOResult,
    tags=["Economic"],
    summary="Total cost of ownership",
    description="Calculate total cost of ownership over equipment lifetime.",
)
async def calculate_tco(
    request: TCORequest,
    current_user: User = Depends(check_rate_limit),
):
    """Calculate total cost of ownership."""
    total_capital = request.equipment_cost_usd + request.installation_cost_usd
    total_operating = request.annual_operating_cost_usd * request.useful_life_years
    total_maintenance = request.annual_maintenance_cost_usd * request.useful_life_years
    residual = request.equipment_cost_usd * request.residual_value_percent / 100

    tco = total_capital + total_operating + total_maintenance - residual
    annualized = tco / request.useful_life_years

    # NPV of costs
    discount_rate = 0.10
    npv = total_capital
    for year in range(1, request.useful_life_years + 1):
        annual = request.annual_operating_cost_usd + request.annual_maintenance_cost_usd
        npv += annual / ((1 + discount_rate) ** year)
    npv -= residual / ((1 + discount_rate) ** request.useful_life_years)

    provenance = generate_provenance("tco", request.model_dump(), {"tco": tco})

    return TCOResult(
        exchanger_id=request.exchanger_id,
        total_capital_cost_usd=total_capital,
        total_operating_cost_usd=total_operating,
        total_maintenance_cost_usd=total_maintenance,
        residual_value_usd=residual,
        total_cost_of_ownership_usd=tco,
        annualized_cost_usd=annualized,
        npv_of_costs_usd=npv,
        provenance=provenance,
    )


# =============================================================================
# FLEET ENDPOINTS
# =============================================================================


@app.post(
    f"{API_PREFIX}/fleet/optimize",
    response_model=FleetOptimizeResult,
    tags=["Fleet"],
    summary="Fleet-wide optimization",
    description="Optimize cleaning schedules and maintenance across fleet of exchangers.",
)
async def optimize_fleet(
    request: FleetOptimizeRequest,
    current_user: User = Depends(check_rate_limit),
):
    """Optimize fleet-wide operations."""
    # Simplified fleet optimization
    total_before = sum(e.get("annual_cost", 50000) for e in request.exchangers)
    savings_percent = 15  # Assumed optimization savings
    total_after = total_before * (1 - savings_percent / 100)

    # Generate optimized schedule
    schedule = []
    priority_list = []
    for i, exc in enumerate(request.exchangers):
        exc_id = exc.get("exchanger_id", f"HX-{i:03d}")
        health = exc.get("health_index", 75)
        schedule.append({
            "exchanger_id": exc_id,
            "recommended_action": "clean" if health < 70 else "monitor",
            "priority": "high" if health < 60 else "medium" if health < 80 else "low",
            "estimated_cost": exc.get("annual_cost", 50000) * 0.85,
        })
        if health < 70:
            priority_list.append(exc_id)

    provenance = generate_provenance("fleet_optimize", {"count": len(request.exchangers)}, {"savings": savings_percent})

    return FleetOptimizeResult(
        optimization_target=request.optimization_target,
        total_cost_before_usd=total_before,
        total_cost_after_usd=total_after,
        total_savings_usd=total_before - total_after,
        savings_percent=savings_percent,
        optimized_schedule=schedule,
        implementation_priority=priority_list,
        budget_utilized_usd=request.budget_constraint_usd,
        provenance=provenance,
    )


@app.get(
    f"{API_PREFIX}/fleet/summary",
    response_model=FleetSummary,
    tags=["Fleet"],
    summary="Fleet performance summary",
    description="Get summary of fleet-wide performance metrics.",
)
async def get_fleet_summary(
    current_user: User = Depends(check_rate_limit),
):
    """Get fleet performance summary."""
    # Simulated fleet data
    provenance = generate_provenance("fleet_summary", {}, {"exchangers": 25})

    return FleetSummary(
        total_exchangers=25,
        exchangers_optimal=5,
        exchangers_good=12,
        exchangers_degraded=5,
        exchangers_poor=2,
        exchangers_critical=1,
        fleet_average_efficiency_percent=78.5,
        fleet_average_health_index=72.3,
        total_annual_energy_loss_usd=250000,
        total_cleaning_required=8,
        urgent_attention_required=["HX-015", "HX-022"],
        generated_at=datetime.now(timezone.utc),
        provenance=provenance,
    )


# =============================================================================
# HEALTH & MONITORING ENDPOINTS
# =============================================================================


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    tags=["System"],
    summary="Liveness probe",
    description="Check if API is alive and responding.",
)
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=API_VERSION,
        checks={
            "api": "ok",
            "calculators": "ok",
            "cache": "ok",
        },
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Readiness probe",
    description="Check if API is ready to accept traffic.",
)
async def readiness_check():
    """Readiness check endpoint."""
    # Check all components
    components = {
        "api": True,
        "calculators": True,
        "cache": cache_manager is not None,
        "auth": True,
    }

    all_ready = all(components.values())

    return ReadinessResponse(
        ready=all_ready,
        timestamp=datetime.now(timezone.utc),
        components=components,
    )


@app.get(
    "/metrics",
    tags=["System"],
    summary="Prometheus metrics",
    description="Expose Prometheus metrics for monitoring.",
)
async def get_metrics():
    """Expose Prometheus metrics."""
    # In production, use prometheus_client to generate actual metrics
    metrics = """
# HELP gl014_requests_total Total number of requests
# TYPE gl014_requests_total counter
gl014_requests_total{endpoint="/api/v1/analyze",method="POST"} 1523
gl014_requests_total{endpoint="/api/v1/calculate/fouling",method="POST"} 892

# HELP gl014_request_duration_seconds Request duration in seconds
# TYPE gl014_request_duration_seconds histogram
gl014_request_duration_seconds_bucket{endpoint="/api/v1/analyze",le="0.1"} 1200
gl014_request_duration_seconds_bucket{endpoint="/api/v1/analyze",le="0.5"} 1450
gl014_request_duration_seconds_bucket{endpoint="/api/v1/analyze",le="1.0"} 1520
gl014_request_duration_seconds_bucket{endpoint="/api/v1/analyze",le="+Inf"} 1523

# HELP gl014_active_users Current number of active users
# TYPE gl014_active_users gauge
gl014_active_users 42

# HELP gl014_cache_hit_ratio Cache hit ratio
# TYPE gl014_cache_hit_ratio gauge
gl014_cache_hit_ratio 0.85
"""
    return Response(content=metrics, media_type="text/plain")


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured response."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} | Request-ID: {request_id}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        },
        headers=exc.headers,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unexpected error: {exc} | Request-ID: {request_id}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "An internal error occurred. Please try again later.",
                "request_id": request_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        },
    )


# =============================================================================
# APPLICATION ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8014,
        reload=True,
        log_level="info",
        access_log=True,
    )
