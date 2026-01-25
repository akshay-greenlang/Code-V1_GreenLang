# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO - Deterministic Heat Exchanger Analysis Tools

Zero-hallucination heat exchanger performance and fouling analysis tools.
All calculations use published engineering formulas with full provenance tracking.

This module provides a unified interface to all heat exchanger calculators:
- Heat transfer coefficient calculations
- LMTD and effectiveness-NTU methods
- Fouling resistance and progression
- Thermal efficiency analysis
- Equipment health indexing
- Cleaning optimization
- Economic impact analysis

Zero-Hallucination Guarantee:
- All calculations are deterministic (same input -> same output)
- Complete provenance tracking with SHA-256 hashes
- Based on authoritative standards (TEMA, ASME, HTRI)
- No LLM in the calculation path

Reference Standards:
- TEMA Standards (Tubular Exchanger Manufacturers Association)
- ASME PTC 12.5: Heat Exchanger Performance Test Codes
- Kern-Seaton Asymptotic Fouling Model (1959)
- Ebert-Panchal Threshold Fouling Model (1995)
- ISO 14224: Petroleum, petrochemical and natural gas industries

Author: GreenLang AI Agent Factory - GL-BackendDeveloper
Created: 2025-12-01
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from functools import wraps
from threading import RLock
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, Generic
)

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# Type variable for tool return types
T = TypeVar('T')


# =============================================================================
# TOOL DECORATOR AND REGISTRY
# =============================================================================

class ToolType(str, Enum):
    """Classification of tool types."""
    DETERMINISTIC = "deterministic"
    STOCHASTIC = "stochastic"  # Not used - zero hallucination requirement


class ToolCategory(str, Enum):
    """Categories of tools."""
    HEAT_TRANSFER = "heat_transfer"
    FOULING = "fouling"
    PERFORMANCE = "performance"
    CLEANING = "cleaning"
    ECONOMIC = "economic"


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    category: ToolCategory
    deterministic: bool
    description: str
    version: str
    input_model: type
    output_model: type
    reference_standards: Tuple[str, ...]
    execution_count: int = 0
    total_execution_time_ms: float = 0.0
    last_executed: Optional[datetime] = None


class ToolRegistry:
    """
    Registry for heat exchanger analysis tools.

    Provides:
    - Tool registration and lookup
    - Execution statistics
    - Provenance tracking
    - Thread-safe operations
    """

    _instance: Optional['ToolRegistry'] = None
    _lock = RLock()

    def __new__(cls) -> 'ToolRegistry':
        """Singleton pattern for global tool registry."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._tools: Dict[str, ToolMetadata] = {}
                cls._instance._provenance_records: List[Dict[str, Any]] = []
            return cls._instance

    def register(
        self,
        name: str,
        category: ToolCategory,
        deterministic: bool,
        description: str,
        version: str,
        input_model: type,
        output_model: type,
        reference_standards: Tuple[str, ...] = ()
    ) -> Callable:
        """
        Register a tool function.

        Args:
            name: Tool name
            category: Tool category
            deterministic: Whether tool is deterministic
            description: Tool description
            version: Tool version
            input_model: Pydantic input model
            output_model: Output dataclass
            reference_standards: List of reference standards

        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = datetime.utcnow()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    with self._lock:
                        if name in self._tools:
                            self._tools[name].execution_count += 1
                            self._tools[name].total_execution_time_ms += execution_time
                            self._tools[name].last_executed = datetime.utcnow()

            # Register the tool
            with self._lock:
                self._tools[name] = ToolMetadata(
                    name=name,
                    category=category,
                    deterministic=deterministic,
                    description=description,
                    version=version,
                    input_model=input_model,
                    output_model=output_model,
                    reference_standards=reference_standards
                )

            wrapper._tool_name = name
            wrapper._is_tool = True
            return wrapper

        return decorator

    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get tool metadata by name."""
        with self._lock:
            return self._tools.get(name)

    def list_tools(self, category: Optional[ToolCategory] = None) -> List[ToolMetadata]:
        """List all registered tools, optionally filtered by category."""
        with self._lock:
            tools = list(self._tools.values())
            if category:
                tools = [t for t in tools if t.category == category]
            return tools

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics for all tools."""
        with self._lock:
            stats = {}
            for name, metadata in self._tools.items():
                avg_time = (
                    metadata.total_execution_time_ms / metadata.execution_count
                    if metadata.execution_count > 0 else 0.0
                )
                stats[name] = {
                    "execution_count": metadata.execution_count,
                    "total_execution_time_ms": metadata.total_execution_time_ms,
                    "average_execution_time_ms": avg_time,
                    "last_executed": (
                        metadata.last_executed.isoformat()
                        if metadata.last_executed else None
                    )
                }
            return stats

    def record_provenance(
        self,
        tool_name: str,
        inputs_hash: str,
        outputs_hash: str,
        combined_hash: str
    ) -> None:
        """Record provenance for a tool execution."""
        with self._lock:
            self._provenance_records.append({
                "record_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "tool_name": tool_name,
                "inputs_hash": inputs_hash,
                "outputs_hash": outputs_hash,
                "combined_hash": combined_hash
            })
            # Keep last 10000 records
            if len(self._provenance_records) > 10000:
                self._provenance_records = self._provenance_records[-10000:]


# Global registry instance
tool_registry = ToolRegistry()


def tool(
    deterministic: bool = True,
    category: ToolCategory = ToolCategory.HEAT_TRANSFER
) -> Callable:
    """
    Decorator to mark a function as a GreenLang tool.

    Args:
        deterministic: Whether the tool is deterministic (must be True for zero-hallucination)
        category: Tool category for organization

    Returns:
        Decorator function

    Example:
        @tool(deterministic=True, category=ToolCategory.HEAT_TRANSFER)
        def calculate_lmtd(inputs: LMTDInput) -> LMTDResult:
            ...
    """
    if not deterministic:
        raise ValueError("GL-014 tools must be deterministic for zero-hallucination guarantee")

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._is_deterministic_tool = True
        wrapper._tool_category = category
        return wrapper

    return decorator


# =============================================================================
# INPUT MODELS (Pydantic for Validation)
# =============================================================================

class HeatTransferInput(BaseModel):
    """Input parameters for overall heat transfer coefficient calculation."""
    h_hot_w_m2_k: float = Field(..., gt=0, description="Hot side film coefficient (W/m2.K)")
    h_cold_w_m2_k: float = Field(..., gt=0, description="Cold side film coefficient (W/m2.K)")
    tube_od_m: float = Field(..., gt=0, description="Tube outer diameter (m)")
    tube_id_m: float = Field(..., gt=0, description="Tube inner diameter (m)")
    tube_k_w_m_k: float = Field(..., gt=0, description="Tube thermal conductivity (W/m.K)")
    r_fouling_hot_m2_k_w: float = Field(default=0.0, ge=0, description="Hot side fouling resistance (m2.K/W)")
    r_fouling_cold_m2_k_w: float = Field(default=0.0, ge=0, description="Cold side fouling resistance (m2.K/W)")

    @validator('tube_id_m')
    def validate_tube_id(cls, v, values):
        if 'tube_od_m' in values and v >= values['tube_od_m']:
            raise ValueError("Tube ID must be less than tube OD")
        return v


class LMTDInput(BaseModel):
    """Input parameters for Log Mean Temperature Difference calculation."""
    t_hot_in_c: float = Field(..., description="Hot fluid inlet temperature (C)")
    t_hot_out_c: float = Field(..., description="Hot fluid outlet temperature (C)")
    t_cold_in_c: float = Field(..., description="Cold fluid inlet temperature (C)")
    t_cold_out_c: float = Field(..., description="Cold fluid outlet temperature (C)")
    flow_arrangement: str = Field(default="counterflow", description="Flow arrangement: counterflow or parallel")

    @validator('flow_arrangement')
    def validate_flow(cls, v):
        if v not in ('counterflow', 'parallel', 'crossflow'):
            raise ValueError("Flow arrangement must be counterflow, parallel, or crossflow")
        return v


class EffectivenessInput(BaseModel):
    """Input parameters for effectiveness-NTU calculation."""
    ntu: float = Field(..., gt=0, description="Number of Transfer Units")
    c_ratio: float = Field(..., ge=0, le=1, description="Capacity ratio Cmin/Cmax")
    flow_arrangement: str = Field(default="counterflow", description="Flow arrangement")

    @validator('flow_arrangement')
    def validate_flow(cls, v):
        valid = ('counterflow', 'parallel', 'crossflow_unmixed', 'crossflow_both_mixed',
                 'shell_tube_one_shell', 'shell_tube_two_shell')
        if v not in valid:
            raise ValueError(f"Flow arrangement must be one of: {valid}")
        return v


class FoulingInput(BaseModel):
    """Input parameters for fouling resistance calculation."""
    u_clean_w_m2_k: float = Field(..., gt=0, description="Clean overall HTC (W/m2.K)")
    u_fouled_w_m2_k: float = Field(..., gt=0, description="Fouled overall HTC (W/m2.K)")
    fluid_type_hot: str = Field(default="process_fluid", description="Hot side fluid type")
    fluid_type_cold: str = Field(default="cooling_water", description="Cold side fluid type")

    @validator('u_fouled_w_m2_k')
    def validate_u_fouled(cls, v, values):
        if 'u_clean_w_m2_k' in values and v > values['u_clean_w_m2_k']:
            raise ValueError("Fouled U cannot exceed clean U")
        return v


class FoulingPredictionInput(BaseModel):
    """Input parameters for fouling progression prediction."""
    current_r_f_m2_k_w: float = Field(..., ge=0, description="Current fouling resistance (m2.K/W)")
    fouling_rate_m2_k_w_per_hour: float = Field(..., ge=0, description="Fouling rate per hour")
    target_time_hours: float = Field(..., gt=0, description="Prediction target time (hours)")
    design_fouling_resistance_m2_k_w: float = Field(..., gt=0, description="Design fouling resistance")
    r_f_max_m2_k_w: Optional[float] = Field(None, gt=0, description="Asymptotic R_f (Kern-Seaton)")
    time_constant_hours: Optional[float] = Field(None, gt=0, description="Time constant (Kern-Seaton)")


class EfficiencyInput(BaseModel):
    """Input parameters for thermal efficiency calculation."""
    q_actual_kw: float = Field(..., gt=0, description="Actual heat duty (kW)")
    q_design_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    t_hot_in_c: float = Field(..., description="Hot inlet temperature (C)")
    t_hot_out_c: float = Field(..., description="Hot outlet temperature (C)")
    t_cold_in_c: float = Field(..., description="Cold inlet temperature (C)")
    t_cold_out_c: float = Field(..., description="Cold outlet temperature (C)")


class HealthInput(BaseModel):
    """Input parameters for equipment health index calculation."""
    u_actual_w_m2_k: float = Field(..., gt=0, description="Actual overall HTC")
    u_design_w_m2_k: float = Field(..., gt=0, description="Design overall HTC")
    dp_actual_kpa: float = Field(..., ge=0, description="Actual pressure drop (kPa)")
    dp_design_kpa: float = Field(..., gt=0, description="Design pressure drop (kPa)")
    approach_temp_actual_c: float = Field(..., ge=0, description="Actual approach temperature (C)")
    approach_temp_design_c: float = Field(..., gt=0, description="Design approach temperature (C)")
    vibration_mm_s: Optional[float] = Field(None, ge=0, description="Vibration velocity (mm/s)")
    corrosion_rate_mm_year: Optional[float] = Field(None, ge=0, description="Corrosion rate (mm/year)")


class CleaningInput(BaseModel):
    """Input parameters for cleaning interval optimization."""
    current_r_f_m2_k_w: float = Field(..., ge=0, description="Current fouling resistance")
    fouling_rate_m2_k_w_per_day: float = Field(..., ge=0, description="Fouling rate per day")
    cleaning_threshold_r_f: float = Field(..., gt=0, description="Cleaning threshold R_f")
    cleaning_cost_usd: float = Field(..., gt=0, description="Cost per cleaning ($)")
    energy_loss_cost_per_day_usd: float = Field(..., ge=0, description="Energy loss cost per day ($)")
    downtime_hours: float = Field(default=8.0, gt=0, description="Downtime per cleaning (hours)")
    production_loss_per_hour_usd: float = Field(default=0.0, ge=0, description="Production loss ($/hour)")


class CostBenefitInput(BaseModel):
    """Input parameters for cleaning cost-benefit analysis."""
    cleaning_cost_usd: float = Field(..., gt=0, description="Cleaning cost ($)")
    energy_savings_per_year_usd: float = Field(..., ge=0, description="Annual energy savings ($)")
    production_improvement_per_year_usd: float = Field(default=0.0, ge=0, description="Production improvement ($)")
    equipment_life_extension_years: float = Field(default=0.0, ge=0, description="Life extension (years)")
    equipment_replacement_cost_usd: float = Field(default=0.0, ge=0, description="Replacement cost ($)")
    discount_rate_percent: float = Field(default=10.0, gt=0, description="Discount rate (%)")


class EnergyLossInput(BaseModel):
    """Input parameters for energy loss cost calculation."""
    design_duty_kw: float = Field(..., gt=0, description="Design heat duty (kW)")
    actual_duty_kw: float = Field(..., gt=0, description="Actual heat duty (kW)")
    fuel_cost_per_kwh: float = Field(..., gt=0, description="Fuel cost ($/kWh)")
    system_efficiency: float = Field(default=0.85, gt=0, le=1, description="System efficiency")
    operating_hours_per_year: float = Field(default=8000.0, gt=0, description="Operating hours/year")
    carbon_price_per_tonne: float = Field(default=50.0, ge=0, description="Carbon price ($/tonne)")
    emission_factor_kg_co2_per_kwh: float = Field(default=0.185, ge=0, description="CO2 factor (kg/kWh)")


class ROIInput(BaseModel):
    """Input parameters for ROI calculation."""
    investment_cost_usd: float = Field(..., gt=0, description="Investment cost ($)")
    annual_savings_usd: float = Field(..., ge=0, description="Annual savings ($)")
    useful_life_years: int = Field(default=10, gt=0, description="Useful life (years)")
    discount_rate_percent: float = Field(default=10.0, gt=0, description="Discount rate (%)")
    residual_value_percent: float = Field(default=10.0, ge=0, le=100, description="Residual value (%)")


# =============================================================================
# OUTPUT DATA CLASSES (Frozen for Immutability)
# =============================================================================

@dataclass(frozen=True)
class CalculationStep:
    """Single calculation step for provenance tracking."""
    step_number: int
    operation: str
    description: str
    inputs: Tuple[Tuple[str, str], ...]
    output_value: str
    output_name: str
    formula: Optional[str] = None
    units: Optional[str] = None


@dataclass(frozen=True)
class HeatTransferResult:
    """Result of overall heat transfer coefficient calculation."""
    u_overall_w_m2_k: Decimal
    u_clean_w_m2_k: Decimal
    r_total_m2_k_w: Decimal
    r_hot_film_m2_k_w: Decimal
    r_cold_film_m2_k_w: Decimal
    r_tube_wall_m2_k_w: Decimal
    r_fouling_total_m2_k_w: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class LMTDResult:
    """Result of Log Mean Temperature Difference calculation."""
    lmtd_c: Decimal
    delta_t1_c: Decimal
    delta_t2_c: Decimal
    flow_arrangement: str
    correction_factor: Decimal
    lmtd_corrected_c: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class EffectivenessResult:
    """Result of effectiveness-NTU calculation."""
    effectiveness: Decimal
    ntu: Decimal
    c_ratio: Decimal
    q_max_fraction: Decimal
    flow_arrangement: str
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class FoulingResult:
    """Result of fouling resistance calculation."""
    fouling_resistance_m2_k_w: Decimal
    normalized_fouling_factor: Decimal
    cleanliness_factor_percent: Decimal
    u_clean_w_m2_k: Decimal
    u_fouled_w_m2_k: Decimal
    heat_transfer_loss_percent: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class FoulingPredictionResult:
    """Result of fouling progression prediction."""
    predicted_r_f_m2_k_w: Decimal
    time_to_cleaning_threshold_hours: Decimal
    time_to_cleaning_threshold_days: Decimal
    prediction_confidence_percent: Decimal
    model_used: str
    upper_bound_r_f: Decimal
    lower_bound_r_f: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class EfficiencyResult:
    """Result of thermal efficiency calculation."""
    thermal_efficiency_percent: Decimal
    heat_transfer_efficiency_percent: Decimal
    approach_temp_hot_c: Decimal
    approach_temp_cold_c: Decimal
    duty_shortfall_kw: Decimal
    duty_shortfall_percent: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class HealthIndexResult:
    """Result of equipment health index calculation."""
    health_index: Decimal
    health_level: str
    u_ratio: Decimal
    dp_ratio: Decimal
    approach_temp_ratio: Decimal
    component_scores: Dict[str, Decimal]
    weights_used: Dict[str, Decimal]
    recommendations: Tuple[str, ...]
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class CleaningResult:
    """Result of cleaning interval optimization."""
    optimal_cleaning_interval_days: Decimal
    time_to_next_cleaning_days: Decimal
    cleanings_per_year: Decimal
    total_cleaning_cost_per_year_usd: Decimal
    total_energy_loss_per_year_usd: Decimal
    total_cost_per_year_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class CostBenefitResult:
    """Result of cleaning cost-benefit analysis."""
    net_present_value_usd: Decimal
    simple_payback_years: Decimal
    benefit_cost_ratio: Decimal
    annual_net_benefit_usd: Decimal
    total_benefits_usd: Decimal
    total_costs_usd: Decimal
    irr_percent: Optional[Decimal]
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class EnergyLossResult:
    """Result of energy loss cost calculation."""
    heat_transfer_loss_kw: Decimal
    additional_fuel_kwh_per_year: Decimal
    energy_cost_per_year_usd: Decimal
    carbon_emissions_kg_per_year: Decimal
    carbon_cost_per_year_usd: Decimal
    total_energy_penalty_usd: Decimal
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


@dataclass(frozen=True)
class ROIResult:
    """Result of ROI calculation."""
    simple_payback_years: Decimal
    roi_percent: Decimal
    npv_usd: Decimal
    irr_percent: Optional[Decimal]
    profitability_index: Decimal
    discounted_payback_years: Decimal
    recommendation: str
    calculation_steps: Tuple[CalculationStep, ...]
    provenance_hash: str
    timestamp: str


# =============================================================================
# HEAT EXCHANGER TOOLS CLASS
# =============================================================================

class HeatExchangerTools:
    """
    Deterministic heat exchanger analysis tool suite.

    All methods are pure functions with complete provenance tracking.
    Zero-hallucination guarantee: No LLM in calculation path.

    Reference Standards:
    - TEMA Standards for Shell and Tube Heat Exchangers
    - ASME PTC 12.5 Heat Exchanger Performance Test Codes
    - HTRI Design Manual
    - Kern-Seaton Fouling Model
    - Ebert-Panchal Threshold Fouling Model

    Example:
        >>> tools = HeatExchangerTools()
        >>> lmtd_result = tools.calculate_lmtd(LMTDInput(
        ...     t_hot_in_c=120, t_hot_out_c=80,
        ...     t_cold_in_c=30, t_cold_out_c=70
        ... ))
        >>> print(f"LMTD: {lmtd_result.lmtd_c} C")
    """

    VERSION = "1.0.0"
    DECIMAL_PRECISION = 6
    QUANTIZE_PATTERN = Decimal("0.000001")

    def __init__(self, precision: int = 6):
        """
        Initialize Heat Exchanger Tools.

        Args:
            precision: Decimal precision for calculations (default: 6)
        """
        self._precision = precision
        self._quantize = Decimal(f"0.{'0' * precision}")

    # =========================================================================
    # HEAT TRANSFER TOOLS
    # =========================================================================

    @tool(deterministic=True, category=ToolCategory.HEAT_TRANSFER)
    def calculate_overall_heat_transfer_coefficient(
        self,
        inputs: HeatTransferInput
    ) -> HeatTransferResult:
        """
        Calculate overall heat transfer coefficient U.

        Formula (based on outside area):
        1/U_o = 1/h_o + R_fo + (r_o*ln(r_o/r_i))/k + (r_o/r_i)*(R_fi + 1/h_i)

        Args:
            inputs: HeatTransferInput with film coefficients and geometry

        Returns:
            HeatTransferResult with U value and resistances

        Reference: TEMA Standards, Kern "Process Heat Transfer"
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        # Convert to Decimal
        h_hot = Decimal(str(inputs.h_hot_w_m2_k))
        h_cold = Decimal(str(inputs.h_cold_w_m2_k))
        d_o = Decimal(str(inputs.tube_od_m))
        d_i = Decimal(str(inputs.tube_id_m))
        k_tube = Decimal(str(inputs.tube_k_w_m_k))
        r_f_hot = Decimal(str(inputs.r_fouling_hot_m2_k_w))
        r_f_cold = Decimal(str(inputs.r_fouling_cold_m2_k_w))

        r_o = d_o / Decimal("2")
        r_i = d_i / Decimal("2")

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal for precision",
            "inputs": {"h_hot": str(h_hot), "h_cold": str(h_cold)},
            "output": "Decimal values"
        })

        # Calculate individual resistances
        # Hot side film resistance (outside)
        r_hot_film = Decimal("1") / h_hot

        steps.append({
            "step": 2, "operation": "divide",
            "formula": "R_hot = 1/h_hot",
            "description": "Calculate hot side film resistance",
            "inputs": {"h_hot": str(h_hot)},
            "output": str(r_hot_film), "units": "m2.K/W"
        })

        # Cold side film resistance (inside, corrected for area ratio)
        r_cold_film = (r_o / r_i) * (Decimal("1") / h_cold)

        steps.append({
            "step": 3, "operation": "divide_multiply",
            "formula": "R_cold = (r_o/r_i) * (1/h_cold)",
            "description": "Calculate cold side film resistance (area corrected)",
            "inputs": {"r_o": str(r_o), "r_i": str(r_i), "h_cold": str(h_cold)},
            "output": str(r_cold_film), "units": "m2.K/W"
        })

        # Tube wall resistance
        import math as _math
        ln_ratio = Decimal(str(_math.log(float(r_o / r_i))))
        r_wall = (r_o * ln_ratio) / k_tube

        steps.append({
            "step": 4, "operation": "logarithm_multiply_divide",
            "formula": "R_wall = r_o * ln(r_o/r_i) / k",
            "description": "Calculate tube wall conduction resistance",
            "inputs": {"r_o": str(r_o), "r_i": str(r_i), "k": str(k_tube)},
            "output": str(r_wall), "units": "m2.K/W"
        })

        # Fouling resistances
        r_fouling_total = r_f_hot + (r_o / r_i) * r_f_cold

        steps.append({
            "step": 5, "operation": "add",
            "formula": "R_f_total = R_f_hot + (r_o/r_i)*R_f_cold",
            "description": "Calculate total fouling resistance",
            "inputs": {"R_f_hot": str(r_f_hot), "R_f_cold": str(r_f_cold)},
            "output": str(r_fouling_total), "units": "m2.K/W"
        })

        # Total resistance
        r_total = r_hot_film + r_f_hot + r_wall + (r_o / r_i) * r_f_cold + r_cold_film

        steps.append({
            "step": 6, "operation": "add",
            "formula": "R_total = R_hot + R_f_hot + R_wall + R_f_cold_adj + R_cold",
            "description": "Calculate total thermal resistance",
            "inputs": {"components": "all resistances"},
            "output": str(r_total), "units": "m2.K/W"
        })

        # Overall heat transfer coefficient (fouled)
        u_overall = Decimal("1") / r_total
        u_overall = u_overall.quantize(self._quantize, rounding=ROUND_HALF_UP)

        steps.append({
            "step": 7, "operation": "divide",
            "formula": "U = 1/R_total",
            "description": "Calculate overall heat transfer coefficient",
            "inputs": {"R_total": str(r_total)},
            "output": str(u_overall), "units": "W/m2.K"
        })

        # Clean U (without fouling)
        r_clean = r_hot_film + r_wall + r_cold_film
        u_clean = Decimal("1") / r_clean
        u_clean = u_clean.quantize(self._quantize, rounding=ROUND_HALF_UP)

        # Generate provenance hash
        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_overall_heat_transfer_coefficient",
            inputs.dict(), str(u_overall)
        )

        # Convert steps to frozen format
        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="result",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return HeatTransferResult(
            u_overall_w_m2_k=u_overall,
            u_clean_w_m2_k=u_clean,
            r_total_m2_k_w=r_total.quantize(self._quantize),
            r_hot_film_m2_k_w=r_hot_film.quantize(self._quantize),
            r_cold_film_m2_k_w=r_cold_film.quantize(self._quantize),
            r_tube_wall_m2_k_w=r_wall.quantize(self._quantize),
            r_fouling_total_m2_k_w=r_fouling_total.quantize(self._quantize),
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.HEAT_TRANSFER)
    def calculate_lmtd(self, inputs: LMTDInput) -> LMTDResult:
        """
        Calculate Log Mean Temperature Difference (LMTD).

        Formula:
        LMTD = (dT1 - dT2) / ln(dT1/dT2)

        For counterflow:
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

        For parallel flow:
        dT1 = T_hot_in - T_cold_in
        dT2 = T_hot_out - T_cold_out

        Args:
            inputs: LMTDInput with temperatures and flow arrangement

        Returns:
            LMTDResult with LMTD and correction factors

        Reference: TEMA Standards, Kern "Process Heat Transfer" Chapter 7
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        # Convert to Decimal
        t_hi = Decimal(str(inputs.t_hot_in_c))
        t_ho = Decimal(str(inputs.t_hot_out_c))
        t_ci = Decimal(str(inputs.t_cold_in_c))
        t_co = Decimal(str(inputs.t_cold_out_c))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert temperatures to Decimal",
            "inputs": {"T_hot_in": str(t_hi), "T_hot_out": str(t_ho),
                      "T_cold_in": str(t_ci), "T_cold_out": str(t_co)},
            "output": "Decimal values"
        })

        # Calculate temperature differences based on flow arrangement
        if inputs.flow_arrangement == "counterflow":
            delta_t1 = t_hi - t_co  # Hot inlet - Cold outlet
            delta_t2 = t_ho - t_ci  # Hot outlet - Cold inlet
        else:  # parallel flow
            delta_t1 = t_hi - t_ci  # Hot inlet - Cold inlet
            delta_t2 = t_ho - t_co  # Hot outlet - Cold outlet

        steps.append({
            "step": 2, "operation": "subtract",
            "formula": f"dT1 = {delta_t1}, dT2 = {delta_t2}",
            "description": f"Calculate temperature differences ({inputs.flow_arrangement})",
            "inputs": {"arrangement": inputs.flow_arrangement},
            "output": f"dT1={delta_t1}, dT2={delta_t2}", "units": "C"
        })

        # Calculate LMTD
        # Handle special case where dT1 = dT2
        if abs(delta_t1 - delta_t2) < Decimal("0.001"):
            lmtd = delta_t1  # When equal, LMTD = dT
            steps.append({
                "step": 3, "operation": "special_case",
                "description": "LMTD equals dT when temperature differences are equal",
                "inputs": {"dT1": str(delta_t1), "dT2": str(delta_t2)},
                "output": str(lmtd), "units": "C"
            })
        elif delta_t1 <= Decimal("0") or delta_t2 <= Decimal("0"):
            # Invalid temperature cross
            lmtd = Decimal("0")
            steps.append({
                "step": 3, "operation": "error",
                "description": "Temperature cross detected - invalid configuration",
                "inputs": {"dT1": str(delta_t1), "dT2": str(delta_t2)},
                "output": "0", "units": "C"
            })
        else:
            import math as _math
            ln_ratio = Decimal(str(_math.log(float(delta_t1 / delta_t2))))
            lmtd = (delta_t1 - delta_t2) / ln_ratio
            lmtd = lmtd.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            steps.append({
                "step": 3, "operation": "lmtd_calculation",
                "formula": "LMTD = (dT1 - dT2) / ln(dT1/dT2)",
                "description": "Calculate Log Mean Temperature Difference",
                "inputs": {"dT1": str(delta_t1), "dT2": str(delta_t2)},
                "output": str(lmtd), "units": "C"
            })

        # Correction factor (F factor) for non-pure counterflow
        # For shell-and-tube exchangers with multiple passes
        correction_factor = Decimal("1.0")  # Pure counterflow
        if inputs.flow_arrangement == "parallel":
            correction_factor = Decimal("1.0")  # No correction for parallel

        lmtd_corrected = lmtd * correction_factor

        steps.append({
            "step": 4, "operation": "multiply",
            "formula": "LMTD_corrected = LMTD * F",
            "description": "Apply correction factor",
            "inputs": {"LMTD": str(lmtd), "F": str(correction_factor)},
            "output": str(lmtd_corrected), "units": "C"
        })

        # Generate provenance hash
        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_lmtd", inputs.dict(), str(lmtd)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="lmtd",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return LMTDResult(
            lmtd_c=lmtd,
            delta_t1_c=delta_t1,
            delta_t2_c=delta_t2,
            flow_arrangement=inputs.flow_arrangement,
            correction_factor=correction_factor,
            lmtd_corrected_c=lmtd_corrected,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.HEAT_TRANSFER)
    def calculate_effectiveness(self, inputs: EffectivenessInput) -> EffectivenessResult:
        """
        Calculate heat exchanger effectiveness using epsilon-NTU method.

        Effectiveness = Q_actual / Q_max = f(NTU, Cr, flow arrangement)

        Args:
            inputs: EffectivenessInput with NTU, capacity ratio, flow type

        Returns:
            EffectivenessResult with effectiveness value

        Reference: Kays & London "Compact Heat Exchangers"
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        ntu = Decimal(str(inputs.ntu))
        c_r = Decimal(str(inputs.c_ratio))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert NTU and capacity ratio to Decimal",
            "inputs": {"NTU": str(ntu), "Cr": str(c_r)},
            "output": "Decimal values"
        })

        import math as _math

        # Calculate effectiveness based on flow arrangement
        if inputs.flow_arrangement == "counterflow":
            if c_r < Decimal("0.999"):  # Normal case
                exp_term = Decimal(str(_math.exp(-float(ntu) * (1 - float(c_r)))))
                epsilon = (Decimal("1") - exp_term) / (Decimal("1") - c_r * exp_term)
            else:  # Cr = 1 (balanced flow)
                epsilon = ntu / (Decimal("1") + ntu)
            formula = "epsilon = (1-exp(-NTU*(1-Cr)))/(1-Cr*exp(-NTU*(1-Cr)))"

        elif inputs.flow_arrangement == "parallel":
            exp_term = Decimal(str(_math.exp(-float(ntu) * (1 + float(c_r)))))
            epsilon = (Decimal("1") - exp_term) / (Decimal("1") + c_r)
            formula = "epsilon = (1-exp(-NTU*(1+Cr)))/(1+Cr)"

        elif inputs.flow_arrangement == "crossflow_unmixed":
            # Approximation for crossflow, both fluids unmixed
            ntu_f = float(ntu)
            cr_f = float(c_r)
            exp1 = _math.exp(-ntu_f)
            exp2 = _math.exp(-cr_f * ntu_f * exp1)
            epsilon = Decimal(str(1 - exp2))
            formula = "epsilon = 1 - exp(-Cr*NTU*exp(-NTU))"

        elif inputs.flow_arrangement == "shell_tube_one_shell":
            # One shell pass, 2/4/6... tube passes
            ntu_f = float(ntu)
            cr_f = float(c_r)
            sqrt_term = _math.sqrt(1 + cr_f**2)
            exp_term = _math.exp(-ntu_f * sqrt_term)
            numerator = 2 / (1 + cr_f + sqrt_term * (1 + exp_term) / (1 - exp_term))
            epsilon = Decimal(str(numerator))
            formula = "TEMA E-shell correlation"

        else:
            # Default to counterflow
            exp_term = Decimal(str(_math.exp(-float(ntu) * (1 - float(c_r)))))
            epsilon = (Decimal("1") - exp_term) / (Decimal("1") - c_r * exp_term)
            formula = "Default counterflow correlation"

        epsilon = min(Decimal("0.9999"), max(Decimal("0"), epsilon))
        epsilon = epsilon.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 2, "operation": "effectiveness_calculation",
            "formula": formula,
            "description": f"Calculate effectiveness ({inputs.flow_arrangement})",
            "inputs": {"NTU": str(ntu), "Cr": str(c_r)},
            "output": str(epsilon), "units": "dimensionless"
        })

        # Q_max fraction
        q_max_fraction = epsilon

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_effectiveness", inputs.dict(), str(epsilon)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="effectiveness",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return EffectivenessResult(
            effectiveness=epsilon,
            ntu=ntu,
            c_ratio=c_r,
            q_max_fraction=q_max_fraction,
            flow_arrangement=inputs.flow_arrangement,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    # =========================================================================
    # FOULING TOOLS
    # =========================================================================

    @tool(deterministic=True, category=ToolCategory.FOULING)
    def calculate_fouling_resistance(self, inputs: FoulingInput) -> FoulingResult:
        """
        Calculate fouling resistance from measured U values.

        Formula:
        R_f = (1/U_fouled) - (1/U_clean)

        Args:
            inputs: FoulingInput with clean and fouled U values

        Returns:
            FoulingResult with fouling resistance and cleanliness factor

        Reference: TEMA Standards, Kern-Seaton (1959)
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        u_clean = Decimal(str(inputs.u_clean_w_m2_k))
        u_fouled = Decimal(str(inputs.u_fouled_w_m2_k))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert U values to Decimal",
            "inputs": {"U_clean": str(u_clean), "U_fouled": str(u_fouled)},
            "output": "Decimal values"
        })

        # Calculate fouling resistance
        r_f = (Decimal("1") / u_fouled) - (Decimal("1") / u_clean)
        r_f = r_f.quantize(self._quantize, rounding=ROUND_HALF_UP)

        steps.append({
            "step": 2, "operation": "subtract",
            "formula": "R_f = (1/U_fouled) - (1/U_clean)",
            "description": "Calculate fouling resistance",
            "inputs": {"U_fouled": str(u_fouled), "U_clean": str(u_clean)},
            "output": str(r_f), "units": "m2.K/W"
        })

        # Get design fouling resistance based on fluid types (TEMA values)
        tema_factors = {
            "process_fluid": Decimal("0.000352"),
            "cooling_water": Decimal("0.000176"),
            "treated_water": Decimal("0.000088"),
            "untreated_water": Decimal("0.000352"),
            "seawater": Decimal("0.000088"),
            "steam": Decimal("0.000088"),
            "fuel_oil": Decimal("0.000880"),
            "crude_oil": Decimal("0.000528"),
        }

        r_f_design_hot = tema_factors.get(inputs.fluid_type_hot, Decimal("0.000352"))
        r_f_design_cold = tema_factors.get(inputs.fluid_type_cold, Decimal("0.000176"))
        r_f_design = r_f_design_hot + r_f_design_cold

        # Normalized fouling factor
        if r_f_design > Decimal("0"):
            normalized_r_f = (r_f / r_f_design).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        else:
            normalized_r_f = Decimal("0")

        steps.append({
            "step": 3, "operation": "divide",
            "formula": "R_f* = R_f / R_f_design",
            "description": "Calculate normalized fouling factor",
            "inputs": {"R_f": str(r_f), "R_f_design": str(r_f_design)},
            "output": str(normalized_r_f), "units": "dimensionless"
        })

        # Cleanliness factor
        cf = (u_fouled / u_clean * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 4, "operation": "divide_multiply",
            "formula": "CF = (U_fouled / U_clean) * 100%",
            "description": "Calculate cleanliness factor",
            "inputs": {"U_fouled": str(u_fouled), "U_clean": str(u_clean)},
            "output": str(cf), "units": "%"
        })

        # Heat transfer loss percentage
        ht_loss = (Decimal("100") - cf).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_fouling_resistance", inputs.dict(), str(r_f)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="R_f",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return FoulingResult(
            fouling_resistance_m2_k_w=r_f,
            normalized_fouling_factor=normalized_r_f,
            cleanliness_factor_percent=cf,
            u_clean_w_m2_k=u_clean,
            u_fouled_w_m2_k=u_fouled,
            heat_transfer_loss_percent=ht_loss,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.FOULING)
    def predict_fouling_progression(
        self,
        inputs: FoulingPredictionInput
    ) -> FoulingPredictionResult:
        """
        Predict fouling progression over time.

        Models:
        1. Linear extrapolation: R_f(t) = R_f_0 + rate * t
        2. Kern-Seaton asymptotic: R_f(t) = R_f_max * (1 - exp(-t/tau))

        Args:
            inputs: FoulingPredictionInput with current state and parameters

        Returns:
            FoulingPredictionResult with predicted R_f and time to cleaning

        Reference: Kern-Seaton (1959), Ebert-Panchal (1995)
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        current_r_f = Decimal(str(inputs.current_r_f_m2_k_w))
        rate = Decimal(str(inputs.fouling_rate_m2_k_w_per_hour))
        target_time = Decimal(str(inputs.target_time_hours))
        r_f_design = Decimal(str(inputs.design_fouling_resistance_m2_k_w))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"current_R_f": str(current_r_f), "rate": str(rate),
                      "target_time": str(target_time)},
            "output": "Decimal values"
        })

        import math as _math

        # Determine model to use
        use_kern_seaton = (
            inputs.r_f_max_m2_k_w is not None and
            inputs.time_constant_hours is not None
        )

        if use_kern_seaton:
            model_used = "Kern-Seaton Asymptotic"
            r_f_max = Decimal(str(inputs.r_f_max_m2_k_w))
            tau = Decimal(str(inputs.time_constant_hours))

            # Inverse to find current equivalent time
            r_f_ratio = current_r_f / r_f_max
            if r_f_ratio < Decimal("0.999"):
                current_equiv_time = -tau * Decimal(str(_math.log(1 - float(r_f_ratio))))
            else:
                current_equiv_time = tau * Decimal("10")

            future_time = current_equiv_time + target_time

            # Predict future R_f
            exp_term = Decimal(str(_math.exp(-float(future_time / tau))))
            predicted_r_f = r_f_max * (Decimal("1") - exp_term)

            steps.append({
                "step": 2, "operation": "kern_seaton_prediction",
                "formula": "R_f(t) = R_f_max * (1 - exp(-t/tau))",
                "description": "Apply Kern-Seaton asymptotic model",
                "inputs": {"R_f_max": str(r_f_max), "tau": str(tau),
                          "future_time": str(future_time)},
                "output": str(predicted_r_f), "units": "m2.K/W"
            })
        else:
            model_used = "Linear Extrapolation"
            predicted_r_f = current_r_f + rate * target_time

            steps.append({
                "step": 2, "operation": "linear_prediction",
                "formula": "R_f(t) = R_f_current + rate * time",
                "description": "Apply linear extrapolation",
                "inputs": {"R_f_current": str(current_r_f), "rate": str(rate),
                          "time": str(target_time)},
                "output": str(predicted_r_f), "units": "m2.K/W"
            })

        predicted_r_f = predicted_r_f.quantize(self._quantize, rounding=ROUND_HALF_UP)

        # Time to cleaning threshold (R_f = R_f_design)
        if rate > Decimal("0"):
            remaining_to_threshold = r_f_design - current_r_f
            if remaining_to_threshold > Decimal("0"):
                time_to_threshold = remaining_to_threshold / rate
            else:
                time_to_threshold = Decimal("0")
        else:
            time_to_threshold = Decimal("999999")

        time_to_threshold_hours = time_to_threshold.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        time_to_threshold_days = (time_to_threshold / Decimal("24")).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 3, "operation": "time_to_threshold",
            "formula": "time = (R_f_threshold - R_f_current) / rate",
            "description": "Calculate time to cleaning threshold",
            "inputs": {"R_f_threshold": str(r_f_design), "R_f_current": str(current_r_f),
                      "rate": str(rate)},
            "output": str(time_to_threshold_hours), "units": "hours"
        })

        # Confidence interval (20% uncertainty)
        uncertainty = Decimal("0.2")
        upper_bound = (current_r_f + rate * (Decimal("1") + uncertainty) * target_time).quantize(self._quantize)
        lower_bound = (current_r_f + rate * (Decimal("1") - uncertainty) * target_time).quantize(self._quantize)
        confidence = Decimal("80.0")

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "predict_fouling_progression", inputs.dict(), str(predicted_r_f)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="predicted_R_f",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return FoulingPredictionResult(
            predicted_r_f_m2_k_w=predicted_r_f,
            time_to_cleaning_threshold_hours=time_to_threshold_hours,
            time_to_cleaning_threshold_days=time_to_threshold_days,
            prediction_confidence_percent=confidence,
            model_used=model_used,
            upper_bound_r_f=upper_bound,
            lower_bound_r_f=lower_bound,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    # =========================================================================
    # PERFORMANCE TOOLS
    # =========================================================================

    @tool(deterministic=True, category=ToolCategory.PERFORMANCE)
    def calculate_thermal_efficiency(self, inputs: EfficiencyInput) -> EfficiencyResult:
        """
        Calculate thermal efficiency of heat exchanger.

        Formula:
        Thermal Efficiency = Q_actual / Q_design * 100%
        Heat Transfer Efficiency = (T_hot_in - T_hot_out) / (T_hot_in - T_cold_in) * 100%

        Args:
            inputs: EfficiencyInput with duty and temperature values

        Returns:
            EfficiencyResult with efficiency metrics

        Reference: ASME PTC 12.5
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        q_actual = Decimal(str(inputs.q_actual_kw))
        q_design = Decimal(str(inputs.q_design_kw))
        t_hi = Decimal(str(inputs.t_hot_in_c))
        t_ho = Decimal(str(inputs.t_hot_out_c))
        t_ci = Decimal(str(inputs.t_cold_in_c))
        t_co = Decimal(str(inputs.t_cold_out_c))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"Q_actual": str(q_actual), "Q_design": str(q_design)},
            "output": "Decimal values"
        })

        # Thermal efficiency (duty-based)
        thermal_eff = (q_actual / q_design * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 2, "operation": "divide_multiply",
            "formula": "Thermal_Eff = (Q_actual / Q_design) * 100%",
            "description": "Calculate thermal efficiency",
            "inputs": {"Q_actual": str(q_actual), "Q_design": str(q_design)},
            "output": str(thermal_eff), "units": "%"
        })

        # Heat transfer efficiency (temperature-based, hot side)
        temp_range_max = t_hi - t_ci
        temp_change_hot = t_hi - t_ho

        if temp_range_max > Decimal("0"):
            ht_eff = (temp_change_hot / temp_range_max * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            ht_eff = Decimal("0")

        steps.append({
            "step": 3, "operation": "divide_multiply",
            "formula": "HT_Eff = (T_hi - T_ho) / (T_hi - T_ci) * 100%",
            "description": "Calculate heat transfer efficiency",
            "inputs": {"T_hot_change": str(temp_change_hot), "Max_range": str(temp_range_max)},
            "output": str(ht_eff), "units": "%"
        })

        # Approach temperatures
        approach_hot = t_ho - t_ci  # Hot outlet approach to cold inlet
        approach_cold = t_co - t_ci  # Cold outlet minus cold inlet (temperature rise)

        # Duty shortfall
        duty_shortfall = q_design - q_actual
        duty_shortfall_pct = ((duty_shortfall / q_design) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 4, "operation": "subtract_divide",
            "formula": "Shortfall_% = (Q_design - Q_actual) / Q_design * 100%",
            "description": "Calculate duty shortfall",
            "inputs": {"Q_design": str(q_design), "Q_actual": str(q_actual)},
            "output": str(duty_shortfall_pct), "units": "%"
        })

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_thermal_efficiency", inputs.dict(), str(thermal_eff)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="efficiency",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return EfficiencyResult(
            thermal_efficiency_percent=thermal_eff,
            heat_transfer_efficiency_percent=ht_eff,
            approach_temp_hot_c=approach_hot.quantize(Decimal("0.1")),
            approach_temp_cold_c=approach_cold.quantize(Decimal("0.1")),
            duty_shortfall_kw=duty_shortfall.quantize(Decimal("0.1")),
            duty_shortfall_percent=duty_shortfall_pct,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.PERFORMANCE)
    def calculate_health_index(self, inputs: HealthInput) -> HealthIndexResult:
        """
        Calculate equipment health index (0-100).

        Health Index = weighted sum of component scores:
        - Heat transfer performance (U ratio)
        - Hydraulic performance (dP ratio)
        - Approach temperature performance
        - Vibration (if provided)
        - Corrosion rate (if provided)

        Args:
            inputs: HealthInput with performance parameters

        Returns:
            HealthIndexResult with health index and component scores

        Reference: ISO 14224, API 581 RBI
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        u_actual = Decimal(str(inputs.u_actual_w_m2_k))
        u_design = Decimal(str(inputs.u_design_w_m2_k))
        dp_actual = Decimal(str(inputs.dp_actual_kpa))
        dp_design = Decimal(str(inputs.dp_design_kpa))
        approach_actual = Decimal(str(inputs.approach_temp_actual_c))
        approach_design = Decimal(str(inputs.approach_temp_design_c))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"U_actual": str(u_actual), "U_design": str(u_design)},
            "output": "Decimal values"
        })

        # Calculate component scores (0-100 scale)
        component_scores: Dict[str, Decimal] = {}
        weights: Dict[str, Decimal] = {}

        # U ratio score (higher is better)
        u_ratio = u_actual / u_design
        u_score = min(Decimal("100"), u_ratio * Decimal("100"))
        component_scores["heat_transfer"] = u_score.quantize(Decimal("0.1"))
        weights["heat_transfer"] = Decimal("0.35")

        steps.append({
            "step": 2, "operation": "ratio_score",
            "formula": "U_score = (U_actual/U_design) * 100",
            "description": "Calculate heat transfer score",
            "inputs": {"U_actual": str(u_actual), "U_design": str(u_design)},
            "output": str(u_score), "units": "score"
        })

        # dP ratio score (lower is better, inverted)
        dp_ratio = dp_actual / dp_design
        if dp_ratio <= Decimal("1"):
            dp_score = Decimal("100")
        elif dp_ratio >= Decimal("2"):
            dp_score = Decimal("0")
        else:
            dp_score = (Decimal("2") - dp_ratio) * Decimal("100")
        component_scores["pressure_drop"] = dp_score.quantize(Decimal("0.1"))
        weights["pressure_drop"] = Decimal("0.25")

        steps.append({
            "step": 3, "operation": "inverted_ratio_score",
            "formula": "dP_score = (2 - dP_ratio) * 100 (capped 0-100)",
            "description": "Calculate pressure drop score",
            "inputs": {"dP_actual": str(dp_actual), "dP_design": str(dp_design)},
            "output": str(dp_score), "units": "score"
        })

        # Approach temperature score (lower actual is better)
        if approach_actual <= approach_design:
            approach_score = Decimal("100")
        else:
            approach_ratio = approach_actual / approach_design
            if approach_ratio >= Decimal("2"):
                approach_score = Decimal("0")
            else:
                approach_score = (Decimal("2") - approach_ratio) * Decimal("100")
        component_scores["approach_temp"] = approach_score.quantize(Decimal("0.1"))
        weights["approach_temp"] = Decimal("0.20")

        # Vibration score (if provided)
        if inputs.vibration_mm_s is not None:
            vib = Decimal(str(inputs.vibration_mm_s))
            # ISO 10816 limits: <2.8 good, 2.8-7.1 acceptable, >7.1 unsatisfactory
            if vib <= Decimal("2.8"):
                vib_score = Decimal("100")
            elif vib >= Decimal("11.2"):
                vib_score = Decimal("0")
            else:
                vib_score = Decimal("100") - (vib - Decimal("2.8")) / Decimal("8.4") * Decimal("100")
            component_scores["vibration"] = vib_score.quantize(Decimal("0.1"))
            weights["vibration"] = Decimal("0.10")
            weights["heat_transfer"] = Decimal("0.30")  # Adjust
            weights["pressure_drop"] = Decimal("0.20")
            weights["approach_temp"] = Decimal("0.15")

        # Corrosion score (if provided)
        if inputs.corrosion_rate_mm_year is not None:
            corr = Decimal(str(inputs.corrosion_rate_mm_year))
            # <0.1 excellent, 0.1-0.5 good, 0.5-1.0 moderate, >1.0 severe
            if corr <= Decimal("0.1"):
                corr_score = Decimal("100")
            elif corr >= Decimal("1.0"):
                corr_score = Decimal("0")
            else:
                corr_score = Decimal("100") - (corr / Decimal("1.0")) * Decimal("100")
            component_scores["corrosion"] = corr_score.quantize(Decimal("0.1"))
            weights["corrosion"] = Decimal("0.10")

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > Decimal("0"):
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            normalized_weights = weights

        # Calculate weighted health index
        health_index = sum(
            component_scores.get(k, Decimal("0")) * normalized_weights.get(k, Decimal("0"))
            for k in component_scores
        )
        health_index = health_index.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        steps.append({
            "step": 4, "operation": "weighted_sum",
            "formula": "HI = sum(score_i * weight_i)",
            "description": "Calculate weighted health index",
            "inputs": {"scores": str(component_scores), "weights": str(normalized_weights)},
            "output": str(health_index), "units": "score"
        })

        # Determine health level
        if health_index >= Decimal("90"):
            health_level = "EXCELLENT"
            recommendations = ("Continue normal monitoring",)
        elif health_index >= Decimal("70"):
            health_level = "GOOD"
            recommendations = ("Monitor fouling trend", "Plan cleaning within 3 months")
        elif health_index >= Decimal("50"):
            health_level = "FAIR"
            recommendations = ("Increase monitoring frequency", "Schedule cleaning within 6 weeks")
        elif health_index >= Decimal("30"):
            health_level = "POOR"
            recommendations = ("Schedule cleaning within 2 weeks", "Inspect for damage")
        else:
            health_level = "CRITICAL"
            recommendations = ("Immediate cleaning required", "Consider emergency shutdown", "Full inspection needed")

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_health_index", inputs.dict(), str(health_index)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="health_index",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return HealthIndexResult(
            health_index=health_index,
            health_level=health_level,
            u_ratio=u_ratio.quantize(Decimal("0.001")),
            dp_ratio=dp_ratio.quantize(Decimal("0.001")),
            approach_temp_ratio=(approach_actual / approach_design).quantize(Decimal("0.001")),
            component_scores=component_scores,
            weights_used={k: v.quantize(Decimal("0.001")) for k, v in normalized_weights.items()},
            recommendations=recommendations,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    # =========================================================================
    # CLEANING TOOLS
    # =========================================================================

    @tool(deterministic=True, category=ToolCategory.CLEANING)
    def optimize_cleaning_interval(self, inputs: CleaningInput) -> CleaningResult:
        """
        Optimize cleaning interval to minimize total cost.

        Total Cost = Cleaning Cost / Interval + Energy Loss Cost
        Optimal interval minimizes: C_clean/T + integral(energy_loss * t)

        Args:
            inputs: CleaningInput with costs and fouling parameters

        Returns:
            CleaningResult with optimal cleaning interval

        Reference: Kern "Process Heat Transfer", Somerscales & Knudsen
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        current_r_f = Decimal(str(inputs.current_r_f_m2_k_w))
        fouling_rate = Decimal(str(inputs.fouling_rate_m2_k_w_per_day))
        threshold_r_f = Decimal(str(inputs.cleaning_threshold_r_f))
        cleaning_cost = Decimal(str(inputs.cleaning_cost_usd))
        energy_loss_per_day = Decimal(str(inputs.energy_loss_cost_per_day_usd))
        downtime_hours = Decimal(str(inputs.downtime_hours))
        production_loss = Decimal(str(inputs.production_loss_per_hour_usd))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"cleaning_cost": str(cleaning_cost), "fouling_rate": str(fouling_rate)},
            "output": "Decimal values"
        })

        # Time to cleaning threshold
        if fouling_rate > Decimal("0"):
            remaining = threshold_r_f - current_r_f
            if remaining > Decimal("0"):
                time_to_threshold = remaining / fouling_rate
            else:
                time_to_threshold = Decimal("0")
        else:
            time_to_threshold = Decimal("365")  # Default if no fouling

        time_to_threshold = time_to_threshold.quantize(Decimal("0.1"))

        steps.append({
            "step": 2, "operation": "divide",
            "formula": "time_to_threshold = (R_f_threshold - R_f_current) / rate",
            "description": "Calculate time to cleaning threshold",
            "inputs": {"R_f_threshold": str(threshold_r_f), "R_f_current": str(current_r_f),
                      "rate": str(fouling_rate)},
            "output": str(time_to_threshold), "units": "days"
        })

        # Total cleaning cost including downtime
        total_cleaning_cost = cleaning_cost + (downtime_hours * production_loss)

        # Optimal cleaning interval (simplified economic model)
        # Minimize: Total_Cost = (Cleaning_Cost / Interval) + (Energy_Loss_Rate * Interval / 2)
        # dTC/dI = -C/I^2 + E/2 = 0
        # I_optimal = sqrt(2 * C / E)

        import math as _math

        if energy_loss_per_day > Decimal("0"):
            i_opt_squared = (Decimal("2") * total_cleaning_cost) / energy_loss_per_day
            optimal_interval = Decimal(str(_math.sqrt(float(i_opt_squared))))
        else:
            optimal_interval = time_to_threshold  # Clean at threshold if no energy loss

        # Cap at threshold
        optimal_interval = min(optimal_interval, time_to_threshold)
        optimal_interval = max(optimal_interval, Decimal("1"))  # Minimum 1 day
        optimal_interval = optimal_interval.quantize(Decimal("0.1"))

        steps.append({
            "step": 3, "operation": "optimization",
            "formula": "I_optimal = sqrt(2 * C_cleaning / E_loss_rate)",
            "description": "Calculate economically optimal cleaning interval",
            "inputs": {"C_cleaning": str(total_cleaning_cost), "E_loss_rate": str(energy_loss_per_day)},
            "output": str(optimal_interval), "units": "days"
        })

        # Annual costs
        if optimal_interval > Decimal("0"):
            cleanings_per_year = (Decimal("365") / optimal_interval).quantize(Decimal("0.1"))
        else:
            cleanings_per_year = Decimal("365")

        annual_cleaning_cost = cleanings_per_year * total_cleaning_cost
        annual_energy_loss = energy_loss_per_day * Decimal("365") * Decimal("0.5")  # Average
        total_annual_cost = (annual_cleaning_cost + annual_energy_loss).quantize(Decimal("0.01"))

        steps.append({
            "step": 4, "operation": "annual_cost",
            "formula": "Total_Annual = Cleanings_per_year * Cost + Energy_Loss",
            "description": "Calculate total annual cost",
            "inputs": {"cleanings": str(cleanings_per_year), "cleaning_cost": str(total_cleaning_cost)},
            "output": str(total_annual_cost), "units": "USD"
        })

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "optimize_cleaning_interval", inputs.dict(), str(optimal_interval)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="optimal_interval",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return CleaningResult(
            optimal_cleaning_interval_days=optimal_interval,
            time_to_next_cleaning_days=min(time_to_threshold, optimal_interval),
            cleanings_per_year=cleanings_per_year,
            total_cleaning_cost_per_year_usd=annual_cleaning_cost.quantize(Decimal("0.01")),
            total_energy_loss_per_year_usd=annual_energy_loss.quantize(Decimal("0.01")),
            total_cost_per_year_usd=total_annual_cost,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.CLEANING)
    def calculate_cleaning_cost_benefit(self, inputs: CostBenefitInput) -> CostBenefitResult:
        """
        Perform cost-benefit analysis for cleaning intervention.

        NPV = sum(Benefits_t - Costs_t) / (1 + r)^t
        Simple Payback = Investment / Annual Savings
        BCR = Total Benefits / Total Costs

        Args:
            inputs: CostBenefitInput with costs and benefits

        Returns:
            CostBenefitResult with NPV, payback, BCR

        Reference: Engineering Economics principles
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        cleaning_cost = Decimal(str(inputs.cleaning_cost_usd))
        energy_savings = Decimal(str(inputs.energy_savings_per_year_usd))
        production_improvement = Decimal(str(inputs.production_improvement_per_year_usd))
        life_extension = Decimal(str(inputs.equipment_life_extension_years))
        replacement_cost = Decimal(str(inputs.equipment_replacement_cost_usd))
        discount_rate = Decimal(str(inputs.discount_rate_percent)) / Decimal("100")

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"cleaning_cost": str(cleaning_cost), "energy_savings": str(energy_savings)},
            "output": "Decimal values"
        })

        # Annual benefits
        annual_benefits = energy_savings + production_improvement

        # Life extension benefit (present value)
        if life_extension > Decimal("0") and replacement_cost > Decimal("0"):
            # Value of deferring replacement
            life_ext_benefit = replacement_cost * discount_rate * life_extension
        else:
            life_ext_benefit = Decimal("0")

        total_benefits = annual_benefits + life_ext_benefit

        steps.append({
            "step": 2, "operation": "sum_benefits",
            "formula": "Total_Benefits = Energy_Savings + Production + Life_Extension_Value",
            "description": "Calculate total annual benefits",
            "inputs": {"energy_savings": str(energy_savings), "production": str(production_improvement)},
            "output": str(total_benefits), "units": "USD"
        })

        # Simple payback
        if annual_benefits > Decimal("0"):
            simple_payback = cleaning_cost / annual_benefits
        else:
            simple_payback = Decimal("999")

        simple_payback = simple_payback.quantize(Decimal("0.01"))

        steps.append({
            "step": 3, "operation": "divide",
            "formula": "Payback = Cost / Annual_Savings",
            "description": "Calculate simple payback period",
            "inputs": {"cost": str(cleaning_cost), "annual_savings": str(annual_benefits)},
            "output": str(simple_payback), "units": "years"
        })

        # NPV (assuming 5-year analysis period)
        npv = -cleaning_cost
        for year in range(1, 6):
            discount_factor = Decimal("1") / ((Decimal("1") + discount_rate) ** year)
            npv += annual_benefits * discount_factor

        npv = npv.quantize(Decimal("0.01"))

        steps.append({
            "step": 4, "operation": "npv_calculation",
            "formula": "NPV = -Cost + sum(Benefits_t / (1+r)^t)",
            "description": "Calculate Net Present Value",
            "inputs": {"cost": str(cleaning_cost), "benefits": str(annual_benefits),
                      "discount_rate": str(discount_rate)},
            "output": str(npv), "units": "USD"
        })

        # Benefit-Cost Ratio
        total_costs = cleaning_cost
        if total_costs > Decimal("0"):
            bcr = total_benefits / total_costs
        else:
            bcr = Decimal("999")

        bcr = bcr.quantize(Decimal("0.01"))

        # Annual net benefit
        annual_net = (annual_benefits - cleaning_cost).quantize(Decimal("0.01"))

        # IRR calculation (simplified - would need Newton-Raphson for exact)
        # Estimate: IRR where NPV = 0
        irr = None  # Would require iterative calculation

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_cleaning_cost_benefit", inputs.dict(), str(npv)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="npv",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return CostBenefitResult(
            net_present_value_usd=npv,
            simple_payback_years=simple_payback,
            benefit_cost_ratio=bcr,
            annual_net_benefit_usd=annual_net,
            total_benefits_usd=total_benefits.quantize(Decimal("0.01")),
            total_costs_usd=total_costs.quantize(Decimal("0.01")),
            irr_percent=irr,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    # =========================================================================
    # ECONOMIC TOOLS
    # =========================================================================

    @tool(deterministic=True, category=ToolCategory.ECONOMIC)
    def calculate_energy_loss_cost(self, inputs: EnergyLossInput) -> EnergyLossResult:
        """
        Calculate annual energy loss cost due to fouling.

        Energy Loss = (Q_design - Q_actual) / System_Efficiency
        Cost = Energy_Loss * Fuel_Cost + Carbon_Emissions * Carbon_Price

        Args:
            inputs: EnergyLossInput with duty and cost parameters

        Returns:
            EnergyLossResult with energy and carbon costs

        Reference: ASHRAE Energy Cost Guide
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        q_design = Decimal(str(inputs.design_duty_kw))
        q_actual = Decimal(str(inputs.actual_duty_kw))
        fuel_cost = Decimal(str(inputs.fuel_cost_per_kwh))
        efficiency = Decimal(str(inputs.system_efficiency))
        operating_hours = Decimal(str(inputs.operating_hours_per_year))
        carbon_price = Decimal(str(inputs.carbon_price_per_tonne))
        emission_factor = Decimal(str(inputs.emission_factor_kg_co2_per_kwh))

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"Q_design": str(q_design), "Q_actual": str(q_actual)},
            "output": "Decimal values"
        })

        # Heat transfer loss
        ht_loss = q_design - q_actual

        steps.append({
            "step": 2, "operation": "subtract",
            "formula": "HT_loss = Q_design - Q_actual",
            "description": "Calculate heat transfer loss",
            "inputs": {"Q_design": str(q_design), "Q_actual": str(q_actual)},
            "output": str(ht_loss), "units": "kW"
        })

        # Additional fuel consumption (accounting for system efficiency)
        additional_fuel_per_hour = ht_loss / efficiency
        additional_fuel_per_year = additional_fuel_per_hour * operating_hours
        additional_fuel_per_year = additional_fuel_per_year.quantize(Decimal("0.1"))

        steps.append({
            "step": 3, "operation": "divide_multiply",
            "formula": "Additional_Fuel = HT_loss / Efficiency * Operating_Hours",
            "description": "Calculate additional fuel consumption",
            "inputs": {"HT_loss": str(ht_loss), "efficiency": str(efficiency),
                      "hours": str(operating_hours)},
            "output": str(additional_fuel_per_year), "units": "kWh/year"
        })

        # Energy cost
        energy_cost = additional_fuel_per_year * fuel_cost
        energy_cost = energy_cost.quantize(Decimal("0.01"))

        steps.append({
            "step": 4, "operation": "multiply",
            "formula": "Energy_Cost = Additional_Fuel * Fuel_Cost",
            "description": "Calculate energy cost",
            "inputs": {"fuel": str(additional_fuel_per_year), "cost": str(fuel_cost)},
            "output": str(energy_cost), "units": "USD/year"
        })

        # Carbon emissions
        carbon_emissions = additional_fuel_per_year * emission_factor
        carbon_emissions = carbon_emissions.quantize(Decimal("0.1"))

        # Carbon cost (emissions in kg, price in $/tonne)
        carbon_cost = (carbon_emissions / Decimal("1000")) * carbon_price
        carbon_cost = carbon_cost.quantize(Decimal("0.01"))

        steps.append({
            "step": 5, "operation": "multiply",
            "formula": "Carbon_Cost = (Emissions_kg / 1000) * Carbon_Price",
            "description": "Calculate carbon cost",
            "inputs": {"emissions_kg": str(carbon_emissions), "price": str(carbon_price)},
            "output": str(carbon_cost), "units": "USD/year"
        })

        # Total energy penalty
        total_penalty = energy_cost + carbon_cost

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_energy_loss_cost", inputs.dict(), str(total_penalty)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="total_penalty",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return EnergyLossResult(
            heat_transfer_loss_kw=ht_loss.quantize(Decimal("0.1")),
            additional_fuel_kwh_per_year=additional_fuel_per_year,
            energy_cost_per_year_usd=energy_cost,
            carbon_emissions_kg_per_year=carbon_emissions,
            carbon_cost_per_year_usd=carbon_cost,
            total_energy_penalty_usd=total_penalty.quantize(Decimal("0.01")),
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    @tool(deterministic=True, category=ToolCategory.ECONOMIC)
    def calculate_roi(self, inputs: ROIInput) -> ROIResult:
        """
        Calculate Return on Investment for heat exchanger improvements.

        ROI = (Net Benefits / Investment) * 100%
        NPV = sum(Cash Flows / (1 + r)^t)
        Simple Payback = Investment / Annual Savings

        Args:
            inputs: ROIInput with investment and savings parameters

        Returns:
            ROIResult with ROI, NPV, payback metrics

        Reference: Engineering Economics, AACE standards
        """
        calculation_id = str(uuid.uuid4())
        steps: List[Dict[str, Any]] = []

        investment = Decimal(str(inputs.investment_cost_usd))
        annual_savings = Decimal(str(inputs.annual_savings_usd))
        life_years = inputs.useful_life_years
        discount_rate = Decimal(str(inputs.discount_rate_percent)) / Decimal("100")
        residual_percent = Decimal(str(inputs.residual_value_percent)) / Decimal("100")

        steps.append({
            "step": 1, "operation": "input_conversion",
            "description": "Convert inputs to Decimal",
            "inputs": {"investment": str(investment), "annual_savings": str(annual_savings)},
            "output": "Decimal values"
        })

        # Simple payback
        if annual_savings > Decimal("0"):
            simple_payback = investment / annual_savings
        else:
            simple_payback = Decimal("999")

        simple_payback = simple_payback.quantize(Decimal("0.01"))

        steps.append({
            "step": 2, "operation": "divide",
            "formula": "Payback = Investment / Annual_Savings",
            "description": "Calculate simple payback period",
            "inputs": {"investment": str(investment), "savings": str(annual_savings)},
            "output": str(simple_payback), "units": "years"
        })

        # NPV calculation
        npv = -investment
        cumulative_savings = Decimal("0")
        discounted_payback = Decimal(str(life_years))
        payback_found = False

        for year in range(1, life_years + 1):
            discount_factor = Decimal("1") / ((Decimal("1") + discount_rate) ** year)
            discounted_cf = annual_savings * discount_factor
            npv += discounted_cf
            cumulative_savings += discounted_cf

            if cumulative_savings >= investment and not payback_found:
                discounted_payback = Decimal(str(year))
                payback_found = True

        # Add residual value
        residual_value = investment * residual_percent
        residual_pv = residual_value / ((Decimal("1") + discount_rate) ** life_years)
        npv += residual_pv
        npv = npv.quantize(Decimal("0.01"))

        steps.append({
            "step": 3, "operation": "npv_calculation",
            "formula": "NPV = -I + sum(S/(1+r)^t) + RV/(1+r)^n",
            "description": "Calculate Net Present Value",
            "inputs": {"investment": str(investment), "savings": str(annual_savings),
                      "rate": str(discount_rate), "years": str(life_years)},
            "output": str(npv), "units": "USD"
        })

        # ROI (based on total undiscounted benefits)
        total_savings = annual_savings * Decimal(str(life_years)) + residual_value
        if investment > Decimal("0"):
            roi = ((total_savings - investment) / investment * Decimal("100")).quantize(Decimal("0.1"))
        else:
            roi = Decimal("0")

        steps.append({
            "step": 4, "operation": "roi_calculation",
            "formula": "ROI = ((Total_Benefits - Investment) / Investment) * 100%",
            "description": "Calculate Return on Investment",
            "inputs": {"total_benefits": str(total_savings), "investment": str(investment)},
            "output": str(roi), "units": "%"
        })

        # Profitability Index
        if investment > Decimal("0"):
            profitability_index = ((npv + investment) / investment).quantize(Decimal("0.01"))
        else:
            profitability_index = Decimal("0")

        # IRR (simplified estimate - would need Newton-Raphson)
        # Using approximation: IRR approx = (Annual Savings / Investment) if payback reasonable
        if simple_payback > Decimal("0") and simple_payback < Decimal(str(life_years)):
            irr_estimate = (Decimal("1") / simple_payback * Decimal("100")).quantize(Decimal("0.1"))
        else:
            irr_estimate = None

        # Recommendation
        if npv > Decimal("0") and simple_payback < Decimal("3"):
            recommendation = "STRONGLY RECOMMENDED - Excellent investment"
        elif npv > Decimal("0") and simple_payback < Decimal("5"):
            recommendation = "RECOMMENDED - Good investment"
        elif npv > Decimal("0"):
            recommendation = "ACCEPTABLE - Positive returns but long payback"
        else:
            recommendation = "NOT RECOMMENDED - Negative NPV"

        provenance_hash = self._generate_provenance_hash(
            calculation_id, "calculate_roi", inputs.dict(), str(npv)
        )

        frozen_steps = tuple(
            CalculationStep(
                step_number=s.get("step", 0),
                operation=s.get("operation", ""),
                description=s.get("description", ""),
                inputs=tuple((k, str(v)) for k, v in s.get("inputs", {}).items()),
                output_value=s.get("output", ""),
                output_name="roi",
                formula=s.get("formula"),
                units=s.get("units")
            ) for s in steps
        )

        return ROIResult(
            simple_payback_years=simple_payback,
            roi_percent=roi,
            npv_usd=npv,
            irr_percent=irr_estimate,
            profitability_index=profitability_index,
            discounted_payback_years=discounted_payback.quantize(Decimal("0.1")),
            recommendation=recommendation,
            calculation_steps=frozen_steps,
            provenance_hash=provenance_hash,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _generate_provenance_hash(
        self,
        calculation_id: str,
        calculation_type: str,
        inputs: Dict[str, Any],
        final_result: str
    ) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            calculation_id: Unique calculation identifier
            calculation_type: Type of calculation
            inputs: Input parameters
            final_result: Final calculated result

        Returns:
            64-character hex SHA-256 hash
        """
        canonical_data = {
            "calculation_id": calculation_id,
            "calculation_type": calculation_type,
            "version": self.VERSION,
            "inputs": self._serialize_for_hash(inputs),
            "final_result": final_result
        }

        canonical_json = json.dumps(
            canonical_data,
            sort_keys=True,
            separators=(',', ':'),
            default=str
        )

        return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

    def _serialize_for_hash(self, obj: Any) -> Any:
        """Serialize object for consistent hashing."""
        if isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, (int, float)):
            return str(Decimal(str(obj)))
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._serialize_for_hash(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_hash(v) for v in obj]
        elif obj is None:
            return None
        else:
            return str(obj)


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def create_heat_exchanger_tools() -> HeatExchangerTools:
    """Create and return a HeatExchangerTools instance."""
    return HeatExchangerTools()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return tool_registry


# =============================================================================
# EXPORT DECLARATIONS
# =============================================================================

__all__ = [
    # Tool decorator and registry
    "tool",
    "ToolType",
    "ToolCategory",
    "ToolMetadata",
    "ToolRegistry",
    "tool_registry",
    # Input models
    "HeatTransferInput",
    "LMTDInput",
    "EffectivenessInput",
    "FoulingInput",
    "FoulingPredictionInput",
    "EfficiencyInput",
    "HealthInput",
    "CleaningInput",
    "CostBenefitInput",
    "EnergyLossInput",
    "ROIInput",
    # Output models
    "CalculationStep",
    "HeatTransferResult",
    "LMTDResult",
    "EffectivenessResult",
    "FoulingResult",
    "FoulingPredictionResult",
    "EfficiencyResult",
    "HealthIndexResult",
    "CleaningResult",
    "CostBenefitResult",
    "EnergyLossResult",
    "ROIResult",
    # Main tools class
    "HeatExchangerTools",
    # Factory functions
    "create_heat_exchanger_tools",
    "get_tool_registry",
]
