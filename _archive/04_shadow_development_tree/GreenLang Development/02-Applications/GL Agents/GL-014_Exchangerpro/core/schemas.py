# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Schema Definitions

Pydantic v2 models for all inputs, outputs, process data, thermal KPIs,
fouling states, and cleaning recommendations for the EXCHANGERPRO agent.

All schemas support zero-hallucination principles with deterministic
calculations, SHA-256 provenance tracking, and regulatory compliance.

Models:
    - ExchangerConfig: Heat exchanger configuration (TEMA type, geometry, materials)
    - OperatingState: Current operating conditions (temperatures, flows, pressures)
    - ThermalKPIs: Thermal performance indicators (Q, UA, LMTD, effectiveness, NTU)
    - FoulingState: Fouling assessment (UA degradation, Rf proxy, fouling rate)
    - CleaningRecommendation: Cleaning schedule optimization results

Author: GreenLang GL-014 EXCHANGERPRO
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator, computed_field

from .config import (
    TEMAType,
    FlowArrangement,
    ShellType,
    TubePattern,
    MaterialGrade,
)


# =============================================================================
# ENUMS
# =============================================================================


class AnalysisStatus(str, Enum):
    """Analysis execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


class FoulingSeverity(str, Enum):
    """Fouling severity classification."""
    CLEAN = "clean"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    CRITICAL = "critical"


class CleaningUrgency(str, Enum):
    """Cleaning recommendation urgency."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CleaningMethod(str, Enum):
    """Cleaning methods for heat exchangers."""
    CHEMICAL_ONLINE = "chemical_online"
    CHEMICAL_OFFLINE = "chemical_offline"
    MECHANICAL_HYDROBLAST = "mechanical_hydroblast"
    MECHANICAL_RODDING = "mechanical_rodding"
    MECHANICAL_BRUSHING = "mechanical_brushing"
    THERMAL_BAKEOUT = "thermal_bakeout"
    COMBINATION = "combination"


class DataQuality(str, Enum):
    """Data quality classification."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    INTERPOLATED = "interpolated"
    MANUAL = "manual"


# =============================================================================
# GEOMETRY SCHEMAS
# =============================================================================


class TubeGeometry(BaseModel):
    """
    Tube bundle geometry specification.

    Defines the physical dimensions and arrangement of tubes
    in a shell and tube heat exchanger.

    Example:
        >>> tubes = TubeGeometry(
        ...     outer_diameter_mm=25.4,
        ...     wall_thickness_mm=2.11,
        ...     length_m=6.0,
        ...     tube_count=200,
        ...     tube_passes=2,
        ...     pattern=TubePattern.TRIANGULAR_30,
        ...     pitch_mm=31.75
        ... )
    """

    outer_diameter_mm: float = Field(
        ...,
        ge=6.0,
        le=100.0,
        description="Tube outer diameter (mm)"
    )
    wall_thickness_mm: float = Field(
        ...,
        ge=0.5,
        le=10.0,
        description="Tube wall thickness (mm)"
    )
    length_m: float = Field(
        ...,
        ge=0.5,
        le=30.0,
        description="Tube length (m)"
    )
    tube_count: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Number of tubes"
    )
    tube_passes: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of tube passes"
    )
    pattern: TubePattern = Field(
        default=TubePattern.TRIANGULAR_30,
        description="Tube layout pattern"
    )
    pitch_mm: float = Field(
        ...,
        ge=8.0,
        le=150.0,
        description="Tube pitch (center to center) (mm)"
    )

    @computed_field
    @property
    def inner_diameter_mm(self) -> float:
        """Calculate tube inner diameter (mm)."""
        return self.outer_diameter_mm - 2 * self.wall_thickness_mm

    @computed_field
    @property
    def tube_side_area_m2(self) -> float:
        """Calculate total tube-side heat transfer area (m2)."""
        inner_d_m = self.inner_diameter_mm / 1000.0
        return math.pi * inner_d_m * self.length_m * self.tube_count

    @computed_field
    @property
    def shell_side_area_m2(self) -> float:
        """Calculate total shell-side heat transfer area (m2)."""
        outer_d_m = self.outer_diameter_mm / 1000.0
        return math.pi * outer_d_m * self.length_m * self.tube_count

    @computed_field
    @property
    def pitch_ratio(self) -> float:
        """Calculate pitch to diameter ratio."""
        return self.pitch_mm / self.outer_diameter_mm

    @field_validator("pitch_mm")
    @classmethod
    def validate_pitch(cls, v: float, info) -> float:
        """Validate tube pitch is greater than outer diameter."""
        # Note: Can't access other fields in field_validator in Pydantic v2
        # Use model_validator for cross-field validation
        return v

    @model_validator(mode="after")
    def validate_geometry(self) -> "TubeGeometry":
        """Validate geometric constraints."""
        if self.pitch_mm <= self.outer_diameter_mm:
            raise ValueError(
                f"Tube pitch ({self.pitch_mm}mm) must be greater than "
                f"outer diameter ({self.outer_diameter_mm}mm)"
            )
        if self.wall_thickness_mm >= self.outer_diameter_mm / 2:
            raise ValueError(
                f"Wall thickness ({self.wall_thickness_mm}mm) must be less than "
                f"half the outer diameter ({self.outer_diameter_mm/2}mm)"
            )
        return self


class BaffleConfig(BaseModel):
    """
    Baffle configuration for shell-side flow.

    Defines baffle geometry and spacing for shell-side
    heat transfer enhancement.
    """

    baffle_type: str = Field(
        default="segmental",
        description="Baffle type (segmental, disc_and_doughnut, orifice)"
    )
    baffle_cut_percent: float = Field(
        default=25.0,
        ge=15.0,
        le=45.0,
        description="Baffle cut as percentage of shell diameter"
    )
    baffle_spacing_mm: float = Field(
        ...,
        ge=50.0,
        le=5000.0,
        description="Baffle spacing (mm)"
    )
    baffle_count: int = Field(
        ...,
        ge=1,
        le=100,
        description="Number of baffles"
    )
    baffle_thickness_mm: float = Field(
        default=6.0,
        ge=3.0,
        le=25.0,
        description="Baffle thickness (mm)"
    )
    inlet_spacing_mm: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=5000.0,
        description="Inlet baffle spacing (if different)"
    )
    outlet_spacing_mm: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=5000.0,
        description="Outlet baffle spacing (if different)"
    )


class ShellGeometry(BaseModel):
    """
    Shell geometry specification.

    Defines the shell dimensions and configuration
    for a shell and tube heat exchanger.
    """

    shell_type: ShellType = Field(
        default=ShellType.E,
        description="Shell type per TEMA classification"
    )
    inner_diameter_mm: float = Field(
        ...,
        ge=100.0,
        le=5000.0,
        description="Shell inner diameter (mm)"
    )
    wall_thickness_mm: float = Field(
        default=10.0,
        ge=3.0,
        le=100.0,
        description="Shell wall thickness (mm)"
    )
    shell_passes: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of shell passes"
    )
    baffles: Optional[BaffleConfig] = Field(
        default=None,
        description="Baffle configuration"
    )

    @computed_field
    @property
    def outer_diameter_mm(self) -> float:
        """Calculate shell outer diameter (mm)."""
        return self.inner_diameter_mm + 2 * self.wall_thickness_mm

    @computed_field
    @property
    def cross_sectional_area_m2(self) -> float:
        """Calculate shell cross-sectional area (m2)."""
        inner_d_m = self.inner_diameter_mm / 1000.0
        return math.pi * (inner_d_m / 2) ** 2


class MaterialProperties(BaseModel):
    """
    Material thermal and mechanical properties.

    Defines properties for heat exchanger components.
    """

    material_grade: MaterialGrade = Field(
        ...,
        description="Material grade designation"
    )
    thermal_conductivity_w_mk: float = Field(
        ...,
        ge=1.0,
        le=500.0,
        description="Thermal conductivity (W/m-K)"
    )
    density_kg_m3: float = Field(
        default=7850.0,
        ge=1000.0,
        le=25000.0,
        description="Material density (kg/m3)"
    )
    specific_heat_j_kgk: float = Field(
        default=500.0,
        ge=100.0,
        le=5000.0,
        description="Specific heat capacity (J/kg-K)"
    )
    max_temperature_c: float = Field(
        default=400.0,
        ge=50.0,
        le=1200.0,
        description="Maximum operating temperature (C)"
    )
    corrosion_allowance_mm: float = Field(
        default=1.5,
        ge=0.0,
        le=10.0,
        description="Corrosion allowance (mm)"
    )


# =============================================================================
# EXCHANGER CONFIGURATION
# =============================================================================


class ExchangerConfig(BaseModel):
    """
    Complete heat exchanger configuration.

    Defines all physical and design parameters for a shell and tube
    heat exchanger per TEMA standards.

    Example:
        >>> config = ExchangerConfig(
        ...     exchanger_id="HX-101",
        ...     tema_type=TEMAType.BEM,
        ...     shell=ShellGeometry(inner_diameter_mm=500.0),
        ...     tubes=TubeGeometry(
        ...         outer_diameter_mm=25.4,
        ...         wall_thickness_mm=2.11,
        ...         length_m=6.0,
        ...         tube_count=200,
        ...         pitch_mm=31.75
        ...     ),
        ...     flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        ...     tube_material=MaterialProperties(
        ...         material_grade=MaterialGrade.STAINLESS_316L,
        ...         thermal_conductivity_w_mk=16.3
        ...     ),
        ...     shell_material=MaterialProperties(
        ...         material_grade=MaterialGrade.CARBON_STEEL,
        ...         thermal_conductivity_w_mk=50.0
        ...     )
        ... )
    """

    exchanger_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Unique exchanger identifier"
    )
    exchanger_name: str = Field(
        default="",
        max_length=200,
        description="Human-readable name"
    )
    tema_type: TEMAType = Field(
        default=TEMAType.BEM,
        description="TEMA type designation"
    )

    # Geometry
    shell: ShellGeometry = Field(
        ...,
        description="Shell geometry specification"
    )
    tubes: TubeGeometry = Field(
        ...,
        description="Tube bundle geometry specification"
    )
    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTER_CURRENT,
        description="Flow arrangement"
    )

    # Materials
    tube_material: MaterialProperties = Field(
        ...,
        description="Tube material properties"
    )
    shell_material: MaterialProperties = Field(
        ...,
        description="Shell material properties"
    )

    # Design parameters
    design_pressure_tube_kpa: float = Field(
        default=1000.0,
        ge=0.0,
        le=50000.0,
        description="Tube-side design pressure (kPa)"
    )
    design_pressure_shell_kpa: float = Field(
        default=1000.0,
        ge=0.0,
        le=50000.0,
        description="Shell-side design pressure (kPa)"
    )
    design_temperature_tube_c: float = Field(
        default=200.0,
        ge=-200.0,
        le=800.0,
        description="Tube-side design temperature (C)"
    )
    design_temperature_shell_c: float = Field(
        default=200.0,
        ge=-200.0,
        le=800.0,
        description="Shell-side design temperature (C)"
    )

    # Design fouling resistances (m2-K/W)
    design_fouling_tube_m2kw: float = Field(
        default=0.00018,
        ge=0.0,
        le=0.01,
        description="Design fouling resistance, tube side (m2-K/W)"
    )
    design_fouling_shell_m2kw: float = Field(
        default=0.00018,
        ge=0.0,
        le=0.01,
        description="Design fouling resistance, shell side (m2-K/W)"
    )

    # Clean overall heat transfer coefficient
    clean_u_w_m2k: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=10000.0,
        description="Clean overall U value (W/m2-K)"
    )

    # Metadata
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Installation date"
    )
    last_cleaning_date: Optional[datetime] = Field(
        default=None,
        description="Last cleaning date"
    )
    service_description: str = Field(
        default="",
        description="Service description (e.g., 'Crude oil / cooling water')"
    )

    @computed_field
    @property
    def heat_transfer_area_m2(self) -> float:
        """Calculate total heat transfer area based on tube OD (m2)."""
        return self.tubes.shell_side_area_m2

    model_config = {"use_enum_values": True}


# =============================================================================
# OPERATING STATE SCHEMAS
# =============================================================================


class TemperatureProfile(BaseModel):
    """
    Temperature measurements for a heat exchanger.

    Contains inlet and outlet temperatures for both
    hot and cold sides.
    """

    hot_inlet_c: float = Field(
        ...,
        ge=-200.0,
        le=1000.0,
        description="Hot side inlet temperature (C)"
    )
    hot_outlet_c: float = Field(
        ...,
        ge=-200.0,
        le=1000.0,
        description="Hot side outlet temperature (C)"
    )
    cold_inlet_c: float = Field(
        ...,
        ge=-200.0,
        le=1000.0,
        description="Cold side inlet temperature (C)"
    )
    cold_outlet_c: float = Field(
        ...,
        ge=-200.0,
        le=1000.0,
        description="Cold side outlet temperature (C)"
    )

    @computed_field
    @property
    def hot_delta_t_c(self) -> float:
        """Hot side temperature change (C)."""
        return self.hot_inlet_c - self.hot_outlet_c

    @computed_field
    @property
    def cold_delta_t_c(self) -> float:
        """Cold side temperature change (C)."""
        return self.cold_outlet_c - self.cold_inlet_c

    @model_validator(mode="after")
    def validate_temperatures(self) -> "TemperatureProfile":
        """Validate temperature physics."""
        # Hot side should cool down (or at minimum stay same for no heat transfer)
        if self.hot_outlet_c > self.hot_inlet_c + 0.1:
            raise ValueError(
                f"Hot outlet ({self.hot_outlet_c}C) cannot be significantly higher "
                f"than hot inlet ({self.hot_inlet_c}C)"
            )
        # Cold side should heat up
        if self.cold_outlet_c < self.cold_inlet_c - 0.1:
            raise ValueError(
                f"Cold outlet ({self.cold_outlet_c}C) cannot be significantly lower "
                f"than cold inlet ({self.cold_inlet_c}C)"
            )
        return self


class FlowProfile(BaseModel):
    """
    Flow rate measurements for a heat exchanger.

    Contains mass flow rates and fluid properties for
    both hot and cold sides.
    """

    hot_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Hot side mass flow rate (kg/s)"
    )
    cold_flow_kg_s: float = Field(
        ...,
        ge=0.0,
        le=10000.0,
        description="Cold side mass flow rate (kg/s)"
    )
    hot_cp_j_kgk: float = Field(
        default=4186.0,
        ge=100.0,
        le=20000.0,
        description="Hot side specific heat (J/kg-K)"
    )
    cold_cp_j_kgk: float = Field(
        default=4186.0,
        ge=100.0,
        le=20000.0,
        description="Cold side specific heat (J/kg-K)"
    )
    hot_density_kg_m3: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=25000.0,
        description="Hot side density (kg/m3)"
    )
    cold_density_kg_m3: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=25000.0,
        description="Cold side density (kg/m3)"
    )

    @computed_field
    @property
    def hot_heat_capacity_rate_w_k(self) -> float:
        """Hot side heat capacity rate C_hot = m_dot * Cp (W/K)."""
        return self.hot_flow_kg_s * self.hot_cp_j_kgk

    @computed_field
    @property
    def cold_heat_capacity_rate_w_k(self) -> float:
        """Cold side heat capacity rate C_cold = m_dot * Cp (W/K)."""
        return self.cold_flow_kg_s * self.cold_cp_j_kgk


class PressureProfile(BaseModel):
    """
    Pressure measurements for a heat exchanger.

    Contains inlet pressures and pressure drops for
    both hot and cold sides.
    """

    hot_inlet_kpa: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Hot side inlet pressure (kPa)"
    )
    cold_inlet_kpa: float = Field(
        ...,
        ge=0.0,
        le=50000.0,
        description="Cold side inlet pressure (kPa)"
    )
    hot_delta_p_kpa: float = Field(
        default=0.0,
        ge=0.0,
        le=5000.0,
        description="Hot side pressure drop (kPa)"
    )
    cold_delta_p_kpa: float = Field(
        default=0.0,
        ge=0.0,
        le=5000.0,
        description="Cold side pressure drop (kPa)"
    )

    @computed_field
    @property
    def hot_outlet_kpa(self) -> float:
        """Hot side outlet pressure (kPa)."""
        return self.hot_inlet_kpa - self.hot_delta_p_kpa

    @computed_field
    @property
    def cold_outlet_kpa(self) -> float:
        """Cold side outlet pressure (kPa)."""
        return self.cold_inlet_kpa - self.cold_delta_p_kpa


class OperatingState(BaseModel):
    """
    Current operating state of a heat exchanger.

    Captures a snapshot of all operating conditions including
    temperatures, flows, and pressures at a specific point in time.

    Example:
        >>> state = OperatingState(
        ...     exchanger_id="HX-101",
        ...     timestamp=datetime.now(timezone.utc),
        ...     temperatures=TemperatureProfile(
        ...         hot_inlet_c=120.0,
        ...         hot_outlet_c=80.0,
        ...         cold_inlet_c=30.0,
        ...         cold_outlet_c=65.0
        ...     ),
        ...     flows=FlowProfile(
        ...         hot_flow_kg_s=10.0,
        ...         cold_flow_kg_s=15.0
        ...     ),
        ...     pressures=PressureProfile(
        ...         hot_inlet_kpa=500.0,
        ...         cold_inlet_kpa=300.0,
        ...         hot_delta_p_kpa=20.0,
        ...         cold_delta_p_kpa=15.0
        ...     )
        ... )
    """

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Measurement timestamp (UTC)"
    )

    # Measurements
    temperatures: TemperatureProfile = Field(
        ...,
        description="Temperature measurements"
    )
    flows: FlowProfile = Field(
        ...,
        description="Flow rate measurements"
    )
    pressures: PressureProfile = Field(
        ...,
        description="Pressure measurements"
    )

    # Data quality
    data_quality: DataQuality = Field(
        default=DataQuality.GOOD,
        description="Data quality flag"
    )
    source_system: str = Field(
        default="dcs",
        description="Data source system"
    )

    # Optional measurements
    ambient_temperature_c: Optional[float] = Field(
        default=None,
        ge=-50.0,
        le=60.0,
        description="Ambient temperature (C)"
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# THERMAL KPI SCHEMAS
# =============================================================================


class HeatBalance(BaseModel):
    """
    Heat balance calculation results.

    Contains heat duty calculations for both sides and
    reconciliation metrics.
    """

    q_hot_w: float = Field(
        ...,
        description="Heat duty from hot side (W)"
    )
    q_cold_w: float = Field(
        ...,
        description="Heat duty from cold side (W)"
    )
    q_reconciled_w: float = Field(
        ...,
        description="Reconciled heat duty (W)"
    )
    heat_balance_error_percent: float = Field(
        ...,
        description="Heat balance error percentage"
    )
    is_balanced: bool = Field(
        ...,
        description="Whether heat balance is acceptable"
    )

    @classmethod
    def calculate(
        cls,
        temperatures: TemperatureProfile,
        flows: FlowProfile,
        tolerance_percent: float = 5.0,
    ) -> "HeatBalance":
        """
        Calculate heat balance from operating data.

        ZERO-HALLUCINATION: Uses deterministic formulas only.

        Args:
            temperatures: Temperature measurements
            flows: Flow measurements
            tolerance_percent: Acceptable heat balance error

        Returns:
            HeatBalance with calculated values
        """
        # Q = m_dot * Cp * delta_T
        q_hot = flows.hot_flow_kg_s * flows.hot_cp_j_kgk * temperatures.hot_delta_t_c
        q_cold = flows.cold_flow_kg_s * flows.cold_cp_j_kgk * temperatures.cold_delta_t_c

        # Reconciled duty (average)
        q_reconciled = (q_hot + q_cold) / 2.0

        # Error calculation (avoid division by zero)
        avg_q = (abs(q_hot) + abs(q_cold)) / 2.0
        if avg_q > 0:
            error_percent = abs(q_hot - q_cold) / avg_q * 100.0
        else:
            error_percent = 0.0

        return cls(
            q_hot_w=q_hot,
            q_cold_w=q_cold,
            q_reconciled_w=q_reconciled,
            heat_balance_error_percent=error_percent,
            is_balanced=error_percent <= tolerance_percent,
        )


class EffectivenessMetrics(BaseModel):
    """
    Heat exchanger effectiveness and NTU metrics.

    Contains epsilon-NTU analysis results.
    """

    epsilon: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Heat exchanger effectiveness"
    )
    ntu: float = Field(
        ...,
        ge=0.0,
        description="Number of Transfer Units"
    )
    c_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Capacity ratio (C_min/C_max)"
    )
    c_min_w_k: float = Field(
        ...,
        ge=0.0,
        description="Minimum heat capacity rate (W/K)"
    )
    c_max_w_k: float = Field(
        ...,
        ge=0.0,
        description="Maximum heat capacity rate (W/K)"
    )
    q_max_w: float = Field(
        ...,
        ge=0.0,
        description="Maximum possible heat transfer (W)"
    )


class ThermalKPIs(BaseModel):
    """
    Comprehensive thermal performance indicators.

    Contains all key thermal KPIs for heat exchanger monitoring
    including heat balance, UA, LMTD, effectiveness, and NTU.

    Example:
        >>> kpis = ThermalKPIs(
        ...     heat_balance=HeatBalance(...),
        ...     ua_w_k=5000.0,
        ...     lmtd_c=25.0,
        ...     lmtd_correction_factor=0.95,
        ...     effectiveness=EffectivenessMetrics(...)
        ... )
    """

    # Heat balance
    heat_balance: HeatBalance = Field(
        ...,
        description="Heat balance calculation"
    )

    # Overall heat transfer
    ua_w_k: float = Field(
        ...,
        ge=0.0,
        description="Overall UA value (W/K)"
    )
    u_w_m2k: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Overall U coefficient (W/m2-K)"
    )

    # LMTD
    lmtd_c: float = Field(
        ...,
        ge=0.0,
        description="Log Mean Temperature Difference (C)"
    )
    lmtd_correction_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="LMTD correction factor (F)"
    )

    # Effectiveness-NTU
    effectiveness: EffectivenessMetrics = Field(
        ...,
        description="Effectiveness-NTU metrics"
    )

    # Approach temperatures
    hot_end_approach_c: float = Field(
        ...,
        description="Hot end approach temperature (C)"
    )
    cold_end_approach_c: float = Field(
        ...,
        description="Cold end approach temperature (C)"
    )
    min_approach_c: float = Field(
        ...,
        description="Minimum approach temperature (C)"
    )

    # Calculation metadata
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    formula_version: str = Field(
        default="THERMAL_KPI_v1.0",
        description="Calculation formula version"
    )

    @classmethod
    def calculate(
        cls,
        operating_state: OperatingState,
        exchanger_config: ExchangerConfig,
        flow_arrangement: FlowArrangement = FlowArrangement.COUNTER_CURRENT,
    ) -> "ThermalKPIs":
        """
        Calculate thermal KPIs from operating state.

        ZERO-HALLUCINATION: Uses only deterministic thermodynamic formulas.

        Args:
            operating_state: Current operating conditions
            exchanger_config: Exchanger configuration
            flow_arrangement: Flow arrangement type

        Returns:
            ThermalKPIs with all calculated metrics
        """
        temps = operating_state.temperatures
        flows = operating_state.flows

        # Calculate heat balance
        heat_balance = HeatBalance.calculate(temps, flows)

        # Calculate LMTD for counter-current flow
        delta_t1 = temps.hot_inlet_c - temps.cold_outlet_c  # Hot end
        delta_t2 = temps.hot_outlet_c - temps.cold_inlet_c  # Cold end

        # Handle special cases for LMTD
        if abs(delta_t1 - delta_t2) < 0.001:
            lmtd = delta_t1
        elif delta_t1 <= 0 or delta_t2 <= 0:
            # Temperature cross - use arithmetic mean
            lmtd = (abs(delta_t1) + abs(delta_t2)) / 2.0
        else:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # LMTD correction factor (1.0 for pure counter-current)
        f_correction = 1.0
        if flow_arrangement in [FlowArrangement.SHELL_AND_TUBE_1_2,
                                  FlowArrangement.SHELL_AND_TUBE_2_4]:
            # Simplified correction factor calculation
            # In production, use proper F-factor charts
            f_correction = 0.9  # Conservative estimate

        # Calculate UA
        q_reconciled = heat_balance.q_reconciled_w
        corrected_lmtd = lmtd * f_correction
        if corrected_lmtd > 0:
            ua = q_reconciled / corrected_lmtd
        else:
            ua = 0.0

        # Calculate U if area is known
        area = exchanger_config.heat_transfer_area_m2
        u_coeff = ua / area if area > 0 else None

        # Effectiveness-NTU calculations
        c_hot = flows.hot_heat_capacity_rate_w_k
        c_cold = flows.cold_heat_capacity_rate_w_k
        c_min = min(c_hot, c_cold)
        c_max = max(c_hot, c_cold)

        # Avoid division by zero
        c_ratio = c_min / c_max if c_max > 0 else 0.0

        # Maximum possible heat transfer
        q_max = c_min * (temps.hot_inlet_c - temps.cold_inlet_c)

        # Effectiveness
        epsilon = abs(q_reconciled) / q_max if q_max > 0 else 0.0
        epsilon = min(epsilon, 1.0)  # Cap at 1.0

        # NTU from effectiveness for counter-current
        if c_ratio < 1.0 and epsilon < 1.0:
            if c_ratio > 0:
                ntu = math.log((1 - epsilon * c_ratio) / (1 - epsilon)) / (1 - c_ratio)
            else:
                ntu = epsilon / (1 - epsilon) if epsilon < 1.0 else 10.0
        else:
            ntu = ua / c_min if c_min > 0 else 0.0

        effectiveness_metrics = EffectivenessMetrics(
            epsilon=epsilon,
            ntu=ntu,
            c_ratio=c_ratio,
            c_min_w_k=c_min,
            c_max_w_k=c_max,
            q_max_w=q_max,
        )

        # Approach temperatures
        hot_end_approach = temps.hot_inlet_c - temps.cold_outlet_c
        cold_end_approach = temps.hot_outlet_c - temps.cold_inlet_c
        min_approach = min(hot_end_approach, cold_end_approach)

        # Create result and compute provenance hash
        result = cls(
            heat_balance=heat_balance,
            ua_w_k=ua,
            u_w_m2k=u_coeff,
            lmtd_c=lmtd,
            lmtd_correction_factor=f_correction,
            effectiveness=effectiveness_metrics,
            hot_end_approach_c=hot_end_approach,
            cold_end_approach_c=cold_end_approach,
            min_approach_c=min_approach,
        )

        # Compute provenance hash
        hash_input = {
            "q_reconciled": round(q_reconciled, 2),
            "ua": round(ua, 2),
            "lmtd": round(lmtd, 4),
            "epsilon": round(epsilon, 6),
            "ntu": round(ntu, 4),
        }
        result.provenance_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]

        return result


# =============================================================================
# FOULING STATE SCHEMAS
# =============================================================================


class FoulingTrend(BaseModel):
    """
    Fouling trend analysis results.

    Contains fouling rate estimation and trend metrics.
    """

    fouling_rate_m2kw_day: float = Field(
        ...,
        description="Fouling rate (m2-K/W per day)"
    )
    trend_direction: str = Field(
        ...,
        description="Trend direction (increasing, stable, decreasing)"
    )
    days_to_critical: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Days until critical fouling threshold"
    )
    confidence_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Confidence in trend estimate"
    )
    data_points_used: int = Field(
        default=0,
        ge=0,
        description="Number of data points used in trend calculation"
    )


class FoulingState(BaseModel):
    """
    Current fouling state assessment.

    Contains all fouling metrics including UA degradation,
    fouling resistance proxy, and severity classification.

    Example:
        >>> fouling = FoulingState(
        ...     exchanger_id="HX-101",
        ...     ua_current_w_k=4500.0,
        ...     ua_clean_w_k=5000.0,
        ...     rf_proxy_m2kw=0.00022,
        ...     severity=FoulingSeverity.MODERATE
        ... )
    """

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Assessment timestamp"
    )

    # UA degradation
    ua_current_w_k: float = Field(
        ...,
        ge=0.0,
        description="Current UA value (W/K)"
    )
    ua_clean_w_k: float = Field(
        ...,
        ge=0.0,
        description="Clean (reference) UA value (W/K)"
    )
    ua_degradation_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="UA degradation percentage"
    )

    # Fouling resistance
    rf_proxy_m2kw: float = Field(
        ...,
        ge=0.0,
        description="Fouling resistance proxy (m2-K/W)"
    )
    rf_design_m2kw: float = Field(
        default=0.00035,
        ge=0.0,
        description="Design fouling resistance (m2-K/W)"
    )
    rf_utilization_percent: float = Field(
        ...,
        ge=0.0,
        description="Fouling resistance utilization (%)"
    )

    # Severity classification
    severity: FoulingSeverity = Field(
        ...,
        description="Fouling severity classification"
    )

    # Trend analysis
    trend: Optional[FoulingTrend] = Field(
        default=None,
        description="Fouling trend analysis"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    calculation_method: str = Field(
        default="ua_degradation",
        description="Fouling calculation method used"
    )

    @classmethod
    def calculate(
        cls,
        exchanger_id: str,
        ua_current: float,
        ua_clean: float,
        area_m2: float,
        rf_design: float = 0.00035,
    ) -> "FoulingState":
        """
        Calculate fouling state from UA values.

        ZERO-HALLUCINATION: Uses only deterministic formulas.

        Args:
            exchanger_id: Exchanger identifier
            ua_current: Current UA value (W/K)
            ua_clean: Clean reference UA value (W/K)
            area_m2: Heat transfer area (m2)
            rf_design: Design fouling resistance (m2-K/W)

        Returns:
            FoulingState with calculated metrics
        """
        # UA degradation
        if ua_clean > 0:
            ua_degradation = (1.0 - ua_current / ua_clean) * 100.0
        else:
            ua_degradation = 0.0

        # Fouling resistance proxy: Rf = 1/U_fouled - 1/U_clean
        u_current = ua_current / area_m2 if area_m2 > 0 else 0
        u_clean = ua_clean / area_m2 if area_m2 > 0 else 0

        if u_current > 0 and u_clean > 0:
            rf_proxy = 1.0 / u_current - 1.0 / u_clean
            rf_proxy = max(rf_proxy, 0.0)  # Cannot be negative
        else:
            rf_proxy = 0.0

        # Fouling resistance utilization
        if rf_design > 0:
            rf_utilization = (rf_proxy / rf_design) * 100.0
        else:
            rf_utilization = 0.0

        # Severity classification
        if ua_degradation < 5.0:
            severity = FoulingSeverity.CLEAN
        elif ua_degradation < 15.0:
            severity = FoulingSeverity.LIGHT
        elif ua_degradation < 30.0:
            severity = FoulingSeverity.MODERATE
        elif ua_degradation < 50.0:
            severity = FoulingSeverity.HEAVY
        elif ua_degradation < 70.0:
            severity = FoulingSeverity.SEVERE
        else:
            severity = FoulingSeverity.CRITICAL

        # Compute provenance hash
        hash_input = {
            "exchanger_id": exchanger_id,
            "ua_current": round(ua_current, 2),
            "ua_clean": round(ua_clean, 2),
            "rf_proxy": round(rf_proxy, 8),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()[:16]

        return cls(
            exchanger_id=exchanger_id,
            ua_current_w_k=ua_current,
            ua_clean_w_k=ua_clean,
            ua_degradation_percent=ua_degradation,
            rf_proxy_m2kw=rf_proxy,
            rf_design_m2kw=rf_design,
            rf_utilization_percent=rf_utilization,
            severity=severity,
            provenance_hash=provenance_hash,
        )

    model_config = {"use_enum_values": True}


# =============================================================================
# CLEANING RECOMMENDATION SCHEMAS
# =============================================================================


class CleaningRecommendation(BaseModel):
    """
    Cleaning recommendation for a heat exchanger.

    Contains recommended cleaning date, expected recovery,
    and cost-benefit analysis.

    Example:
        >>> recommendation = CleaningRecommendation(
        ...     exchanger_id="HX-101",
        ...     recommended_date=datetime(2024, 3, 15),
        ...     urgency=CleaningUrgency.MEDIUM,
        ...     expected_ua_recovery_percent=95.0,
        ...     cleaning_method=CleaningMethod.CHEMICAL_OFFLINE,
        ...     estimated_cost_usd=8000.0
        ... )
    """

    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    recommendation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique recommendation identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Recommendation generation timestamp"
    )

    # Recommendation
    recommended_date: datetime = Field(
        ...,
        description="Recommended cleaning date"
    )
    urgency: CleaningUrgency = Field(
        ...,
        description="Cleaning urgency level"
    )
    cleaning_method: CleaningMethod = Field(
        ...,
        description="Recommended cleaning method"
    )

    # Expected recovery
    expected_ua_recovery_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Expected UA recovery after cleaning (%)"
    )
    expected_downtime_hours: float = Field(
        default=8.0,
        ge=0.0,
        le=720.0,
        description="Expected cleaning downtime (hours)"
    )

    # Cost-benefit
    estimated_cost_usd: float = Field(
        ...,
        ge=0.0,
        description="Estimated cleaning cost (USD)"
    )
    expected_energy_savings_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected annual energy savings (USD)"
    )
    payback_days: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Simple payback period (days)"
    )
    net_benefit_usd_year: float = Field(
        default=0.0,
        description="Net annual benefit (USD)"
    )

    # Current state reference
    current_fouling_state: Optional[FoulingState] = Field(
        default=None,
        description="Current fouling state"
    )

    # Reasoning
    reasoning: List[str] = Field(
        default_factory=list,
        description="Reasoning for recommendation"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Risk factors if cleaning delayed"
    )

    # Confidence and provenance
    confidence_percent: float = Field(
        default=80.0,
        ge=0.0,
        le=100.0,
        description="Confidence in recommendation"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )

    model_config = {"use_enum_values": True}


class CleaningSchedule(BaseModel):
    """
    Optimized cleaning schedule for multiple exchangers.

    Contains a coordinated cleaning plan with timing,
    resource allocation, and cost optimization.
    """

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique schedule identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Schedule generation timestamp"
    )

    # Planning horizon
    planning_start: datetime = Field(
        ...,
        description="Planning period start"
    )
    planning_end: datetime = Field(
        ...,
        description="Planning period end"
    )

    # Recommendations
    recommendations: List[CleaningRecommendation] = Field(
        default_factory=list,
        description="Ordered list of cleaning recommendations"
    )

    # Schedule metrics
    total_cleanings: int = Field(
        default=0,
        ge=0,
        description="Total number of cleanings in schedule"
    )
    total_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Total estimated cost (USD)"
    )
    total_downtime_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Total downtime (hours)"
    )
    total_energy_savings_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Total annual energy savings (USD)"
    )

    # Optimization metadata
    optimization_objective: str = Field(
        default="minimize_total_cost",
        description="Optimization objective used"
    )
    optimization_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Optimization computation time"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )


# =============================================================================
# RESULT SCHEMAS
# =============================================================================


class AnalysisResult(BaseModel):
    """
    Complete analysis result for a heat exchanger.

    Contains thermal KPIs, fouling assessment, and recommendations.
    """

    analysis_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique analysis identifier"
    )
    exchanger_id: str = Field(
        ...,
        description="Exchanger identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETED,
        description="Analysis status"
    )

    # Results
    thermal_kpis: Optional[ThermalKPIs] = Field(
        default=None,
        description="Thermal performance indicators"
    )
    fouling_state: Optional[FoulingState] = Field(
        default=None,
        description="Fouling assessment"
    )
    cleaning_recommendation: Optional[CleaningRecommendation] = Field(
        default=None,
        description="Cleaning recommendation"
    )

    # Execution metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Processing time (milliseconds)"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Analysis warnings"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Analysis errors"
    )

    # Provenance
    input_hash: str = Field(
        default="",
        description="SHA-256 hash of inputs"
    )
    output_hash: str = Field(
        default="",
        description="SHA-256 hash of outputs"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    model_config = {"use_enum_values": True}


class OptimizationResult(BaseModel):
    """
    Cleaning schedule optimization result.

    Contains optimized schedule and performance metrics.
    """

    optimization_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique optimization identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization timestamp"
    )
    status: AnalysisStatus = Field(
        default=AnalysisStatus.COMPLETED,
        description="Optimization status"
    )

    # Results
    schedule: Optional[CleaningSchedule] = Field(
        default=None,
        description="Optimized cleaning schedule"
    )

    # Comparison metrics
    baseline_cost_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Baseline annual cost without optimization"
    )
    optimized_cost_usd_year: float = Field(
        default=0.0,
        ge=0.0,
        description="Optimized annual cost"
    )
    savings_usd_year: float = Field(
        default=0.0,
        description="Annual savings from optimization"
    )
    savings_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Savings percentage"
    )

    # Solver metrics
    solver_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Solver computation time"
    )
    solver_status: str = Field(
        default="optimal",
        description="Solver termination status"
    )
    gap_percent: float = Field(
        default=0.0,
        ge=0.0,
        description="Optimality gap percentage"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance hash"
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Random seed used for reproducibility"
    )

    model_config = {"use_enum_values": True}


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "AnalysisStatus",
    "FoulingSeverity",
    "CleaningUrgency",
    "CleaningMethod",
    "DataQuality",
    # Geometry schemas
    "TubeGeometry",
    "BaffleConfig",
    "ShellGeometry",
    "MaterialProperties",
    # Configuration schemas
    "ExchangerConfig",
    # Operating state schemas
    "TemperatureProfile",
    "FlowProfile",
    "PressureProfile",
    "OperatingState",
    # Thermal KPI schemas
    "HeatBalance",
    "EffectivenessMetrics",
    "ThermalKPIs",
    # Fouling schemas
    "FoulingTrend",
    "FoulingState",
    # Cleaning schemas
    "CleaningRecommendation",
    "CleaningSchedule",
    # Result schemas
    "AnalysisResult",
    "OptimizationResult",
]
