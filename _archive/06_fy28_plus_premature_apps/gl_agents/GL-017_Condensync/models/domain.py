# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Domain Models

Domain-specific enumerations, value objects, and data models for condenser
performance optimization. These models represent the core domain concepts
for steam surface condenser analysis, vacuum optimization, and fouling
prediction.

Standards Reference:
- HEI Standards for Steam Surface Condensers (12th Edition)
- ASME PTC 12.2: Steam Surface Condensers
- EPRI Condenser Performance Guidelines

Zero-Hallucination Guarantee:
All domain models use deterministic types and validated enumerations.
No probabilistic or AI-inferred values in domain model definitions.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union


# ============================================================================
# MATERIAL ENUMERATIONS
# ============================================================================

class TubeMaterial(str, Enum):
    """
    Condenser tube material types.

    Each material has specific thermal conductivity, corrosion resistance,
    and fouling characteristics that affect cleanliness factor calculations.

    Thermal conductivities (W/m-K) at 20C:
    - Admiralty Brass: 111
    - Copper-Nickel 90/10: 45
    - Copper-Nickel 70/30: 29
    - Titanium: 21.9
    - Stainless Steel 304: 16.2
    - Stainless Steel 316: 16.2
    - AL-6XN: 11.5
    - Duplex 2205: 19
    """
    ADMIRALTY_BRASS = "admiralty_brass"
    COPPER_NICKEL_90_10 = "cu_ni_90_10"
    COPPER_NICKEL_70_30 = "cu_ni_70_30"
    TITANIUM_GRADE_2 = "titanium_grade_2"
    TITANIUM_GRADE_7 = "titanium_grade_7"
    STAINLESS_304 = "ss_304"
    STAINLESS_316 = "ss_316"
    STAINLESS_317 = "ss_317"
    AL_6XN = "al_6xn"
    DUPLEX_2205 = "duplex_2205"
    SUPER_DUPLEX_2507 = "super_duplex_2507"
    SEA_CURE = "sea_cure"
    MONEL_400 = "monel_400"
    INCONEL_625 = "inconel_625"

    @property
    def thermal_conductivity_w_m_k(self) -> float:
        """Get thermal conductivity in W/m-K at 20C."""
        conductivities = {
            TubeMaterial.ADMIRALTY_BRASS: 111.0,
            TubeMaterial.COPPER_NICKEL_90_10: 45.0,
            TubeMaterial.COPPER_NICKEL_70_30: 29.0,
            TubeMaterial.TITANIUM_GRADE_2: 21.9,
            TubeMaterial.TITANIUM_GRADE_7: 21.9,
            TubeMaterial.STAINLESS_304: 16.2,
            TubeMaterial.STAINLESS_316: 16.2,
            TubeMaterial.STAINLESS_317: 14.0,
            TubeMaterial.AL_6XN: 11.5,
            TubeMaterial.DUPLEX_2205: 19.0,
            TubeMaterial.SUPER_DUPLEX_2507: 17.0,
            TubeMaterial.SEA_CURE: 26.0,
            TubeMaterial.MONEL_400: 21.8,
            TubeMaterial.INCONEL_625: 9.8,
        }
        return conductivities.get(self, 16.0)

    @property
    def fouling_factor_typical_m2_k_w(self) -> float:
        """Get typical fouling factor in m2-K/W for seawater."""
        # Based on TEMA R-43 and HEI standards
        fouling_factors = {
            TubeMaterial.ADMIRALTY_BRASS: 0.000088,
            TubeMaterial.COPPER_NICKEL_90_10: 0.000088,
            TubeMaterial.COPPER_NICKEL_70_30: 0.000088,
            TubeMaterial.TITANIUM_GRADE_2: 0.000044,
            TubeMaterial.TITANIUM_GRADE_7: 0.000044,
            TubeMaterial.STAINLESS_304: 0.000088,
            TubeMaterial.STAINLESS_316: 0.000088,
            TubeMaterial.STAINLESS_317: 0.000088,
        }
        return fouling_factors.get(self, 0.000088)


class TubeSupport(str, Enum):
    """Tube support plate material and configuration."""
    CARBON_STEEL = "carbon_steel"
    STAINLESS_STEEL = "stainless_steel"
    MUNTZ_METAL = "muntz_metal"
    NAVAL_BRASS = "naval_brass"
    TITANIUM = "titanium"


class TubeEndConnection(str, Enum):
    """Tube-to-tubesheet connection type."""
    ROLLED = "rolled"
    ROLLED_AND_FLARED = "rolled_flared"
    WELDED = "welded"
    ROLLED_AND_WELDED = "rolled_welded"
    EXPLOSIVE_BONDED = "explosive_bonded"


# ============================================================================
# FAILURE MODE ENUMERATIONS
# ============================================================================

class FailureMode(str, Enum):
    """
    Condenser failure/degradation modes.

    Each mode has distinct signatures in sensor data and requires
    different remediation strategies.
    """
    NORMAL = "normal"                    # Normal operation
    FOULING_BIOLOGICAL = "fouling_bio"   # Macro/micro biological fouling
    FOULING_SCALE = "fouling_scale"      # Mineral scale deposits (CaCO3, etc.)
    FOULING_DEBRIS = "fouling_debris"    # Debris accumulation
    FOULING_CORROSION = "fouling_corr"   # Corrosion product deposits
    AIR_LEAK_MINOR = "air_leak_minor"    # <5 SCFM air in-leakage
    AIR_LEAK_MAJOR = "air_leak_major"    # >5 SCFM air in-leakage
    TUBE_LEAK = "tube_leak"              # CW-to-steam side leak
    TUBE_PLUGGED = "tube_plugged"        # Plugged/blocked tubes
    TUBE_VIBRATION = "tube_vibration"    # Flow-induced tube vibration
    WATERBOX_LEAK = "waterbox_leak"      # Waterbox gasket leak
    CW_PUMP_DEGRADED = "cw_pump"         # CW pump degradation
    VALVE_MALFUNCTION = "valve"          # CW valve malfunction
    HOTWELL_LEVEL = "hotwell"            # Hotwell level control issue
    AIR_EJECTOR_DEGRADED = "air_ejector" # Air removal system degraded
    UNKNOWN = "unknown"                  # Undiagnosed condition

    @property
    def typical_impact_mw(self) -> Tuple[float, float]:
        """Get typical MW impact range (min, max) for 500 MW unit."""
        impacts = {
            FailureMode.NORMAL: (0.0, 0.0),
            FailureMode.FOULING_BIOLOGICAL: (0.5, 5.0),
            FailureMode.FOULING_SCALE: (0.5, 3.0),
            FailureMode.FOULING_DEBRIS: (0.2, 2.0),
            FailureMode.AIR_LEAK_MINOR: (0.1, 1.0),
            FailureMode.AIR_LEAK_MAJOR: (1.0, 5.0),
            FailureMode.TUBE_LEAK: (0.0, 0.5),
            FailureMode.TUBE_PLUGGED: (0.1, 2.0),
            FailureMode.CW_PUMP_DEGRADED: (0.5, 3.0),
        }
        return impacts.get(self, (0.0, 1.0))


class FailureSeverity(str, Enum):
    """Severity classification for detected issues."""
    NONE = "none"            # No issue detected
    LOW = "low"              # Minor degradation, monitor
    MEDIUM = "medium"        # Moderate degradation, plan action
    HIGH = "high"            # Significant impact, schedule soon
    CRITICAL = "critical"    # Severe impact, immediate action


# ============================================================================
# CLEANING METHOD ENUMERATIONS
# ============================================================================

class CleaningMethod(str, Enum):
    """
    Condenser tube cleaning methods.

    Each method has specific effectiveness, cost, and applicability
    depending on fouling type and tube material.
    """
    ONLINE_BALL = "online_ball"           # Sponge ball system (Taprogge, etc.)
    ONLINE_BRUSH = "online_brush"         # Brush system (Conco, etc.)
    OFFLINE_HYDROLANCE = "offline_hydro"  # High-pressure water lance
    OFFLINE_BRUSH = "offline_brush"       # Mechanical brush cleaning
    OFFLINE_CHEMICAL = "offline_chem"     # Chemical cleaning
    OFFLINE_ACID = "offline_acid"         # Acid cleaning
    ULTRAVIOLET = "uv_treatment"          # UV treatment for biofouling
    CHLORINATION = "chlorination"         # Chlorine dosing
    ELECTROLYTIC = "electrolytic"         # Electrolytic anti-fouling
    NONE = "none"                         # No cleaning applied

    @property
    def typical_cost_usd_per_pass(self) -> float:
        """Get typical cost per cleaning pass (500 MW unit)."""
        costs = {
            CleaningMethod.ONLINE_BALL: 50.0,      # Per pass
            CleaningMethod.ONLINE_BRUSH: 75.0,     # Per pass
            CleaningMethod.OFFLINE_HYDROLANCE: 15000.0,
            CleaningMethod.OFFLINE_BRUSH: 20000.0,
            CleaningMethod.OFFLINE_CHEMICAL: 50000.0,
            CleaningMethod.OFFLINE_ACID: 75000.0,
        }
        return costs.get(self, 0.0)

    @property
    def typical_cf_recovery_percent(self) -> float:
        """Get typical cleanliness factor recovery (percent of lost CF)."""
        recovery = {
            CleaningMethod.ONLINE_BALL: 60.0,
            CleaningMethod.ONLINE_BRUSH: 70.0,
            CleaningMethod.OFFLINE_HYDROLANCE: 85.0,
            CleaningMethod.OFFLINE_BRUSH: 90.0,
            CleaningMethod.OFFLINE_CHEMICAL: 95.0,
            CleaningMethod.OFFLINE_ACID: 98.0,
        }
        return recovery.get(self, 0.0)

    @property
    def requires_outage(self) -> bool:
        """Check if cleaning method requires unit outage."""
        online_methods = {
            CleaningMethod.ONLINE_BALL,
            CleaningMethod.ONLINE_BRUSH,
            CleaningMethod.CHLORINATION,
            CleaningMethod.ULTRAVIOLET,
            CleaningMethod.ELECTROLYTIC,
        }
        return self not in online_methods


# ============================================================================
# ALERT AND STATUS ENUMERATIONS
# ============================================================================

class AlertLevel(str, Enum):
    """Alert severity levels for operator notifications."""
    INFO = "info"           # Informational, no action required
    WARNING = "warning"     # Attention needed, monitor closely
    CRITICAL = "critical"   # Immediate attention required
    EMERGENCY = "emergency" # Safety/equipment protection event


class OperatingMode(str, Enum):
    """Condenser operating mode."""
    NORMAL = "normal"             # Normal full-load operation
    STARTUP = "startup"           # Unit startup
    SHUTDOWN = "shutdown"         # Unit shutdown
    LOAD_FOLLOW = "load_follow"   # Load following mode
    BYPASS = "bypass"             # Turbine bypass operation
    MAINTENANCE = "maintenance"   # Under maintenance
    OFFLINE = "offline"           # Not in service


class WaterSource(str, Enum):
    """Cooling water source type."""
    ONCE_THROUGH_OCEAN = "ocean"
    ONCE_THROUGH_RIVER = "river"
    ONCE_THROUGH_LAKE = "lake"
    COOLING_TOWER_NATURAL = "ct_natural"
    COOLING_TOWER_MECHANICAL = "ct_mechanical"
    COOLING_TOWER_HYBRID = "ct_hybrid"
    COOLING_POND = "pond"
    DRY_COOLING = "dry"
    HYBRID_WET_DRY = "hybrid"


class VacuumControlMode(str, Enum):
    """Vacuum control system operating mode."""
    MANUAL = "manual"              # Manual setpoint
    AUTOMATIC_FIXED = "auto_fixed" # Fixed automatic setpoint
    AUTOMATIC_OPTIMIZED = "auto_opt"  # Economically optimized
    LOAD_BASED = "load_based"      # Load-dependent setpoint
    AMBIENT_BASED = "ambient"      # Ambient-condition based


# ============================================================================
# PHYSICAL PROPERTY DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class SteamProperties:
    """
    Thermodynamic properties of steam at condenser conditions.

    All properties are at saturation conditions for the given pressure.
    Uses IAPWS-IF97 standard formulations.

    Attributes:
        pressure_kpa_abs: Absolute pressure (kPa)
        saturation_temp_c: Saturation temperature (Celsius)
        enthalpy_steam_kj_kg: Specific enthalpy of saturated vapor (kJ/kg)
        enthalpy_liquid_kj_kg: Specific enthalpy of saturated liquid (kJ/kg)
        latent_heat_kj_kg: Latent heat of vaporization (kJ/kg)
        specific_volume_m3_kg: Specific volume of saturated vapor (m3/kg)
        density_kg_m3: Density of saturated vapor (kg/m3)
        entropy_steam_kj_kg_k: Specific entropy of saturated vapor (kJ/kg-K)
    """
    pressure_kpa_abs: Decimal
    saturation_temp_c: Decimal
    enthalpy_steam_kj_kg: Decimal
    enthalpy_liquid_kj_kg: Decimal
    latent_heat_kj_kg: Decimal
    specific_volume_m3_kg: Decimal
    density_kg_m3: Decimal
    entropy_steam_kj_kg_k: Decimal

    def __post_init__(self) -> None:
        """Validate steam properties are physically consistent."""
        if self.pressure_kpa_abs <= Decimal("0"):
            raise ValueError("Pressure must be positive")
        if self.latent_heat_kj_kg <= Decimal("0"):
            raise ValueError("Latent heat must be positive")

    @property
    def vacuum_mm_hg_abs(self) -> Decimal:
        """Convert pressure to mm Hg absolute."""
        return self.pressure_kpa_abs * Decimal("7.50062")

    @property
    def vacuum_in_hg_abs(self) -> Decimal:
        """Convert pressure to inches Hg absolute."""
        return self.pressure_kpa_abs * Decimal("0.295300")


@dataclass(frozen=True)
class CoolingWaterProperties:
    """
    Cooling water stream properties.

    Attributes:
        flow_rate_m3_s: Volumetric flow rate (m3/s)
        flow_rate_kg_s: Mass flow rate (kg/s)
        inlet_temp_c: CW inlet temperature (Celsius)
        outlet_temp_c: CW outlet temperature (Celsius)
        velocity_m_s: Tube-side velocity (m/s)
        density_kg_m3: Water density (kg/m3)
        specific_heat_kj_kg_k: Specific heat capacity (kJ/kg-K)
        viscosity_pa_s: Dynamic viscosity (Pa-s)
        thermal_conductivity_w_m_k: Thermal conductivity (W/m-K)
        salinity_ppt: Salinity (parts per thousand)
    """
    flow_rate_m3_s: Decimal
    flow_rate_kg_s: Decimal
    inlet_temp_c: Decimal
    outlet_temp_c: Decimal
    velocity_m_s: Decimal
    density_kg_m3: Decimal = Decimal("1000.0")
    specific_heat_kj_kg_k: Decimal = Decimal("4.186")
    viscosity_pa_s: Decimal = Decimal("0.001")
    thermal_conductivity_w_m_k: Decimal = Decimal("0.598")
    salinity_ppt: Decimal = Decimal("0.0")

    @property
    def temperature_rise_c(self) -> Decimal:
        """Calculate CW temperature rise."""
        return self.outlet_temp_c - self.inlet_temp_c

    @property
    def heat_capacity_rate_kw_k(self) -> Decimal:
        """Calculate heat capacity rate (m_dot * Cp) in kW/K."""
        return self.flow_rate_kg_s * self.specific_heat_kj_kg_k

    @property
    def reynolds_number(self) -> Decimal:
        """Calculate Reynolds number (requires tube diameter context)."""
        # Placeholder - actual calculation requires tube OD
        return Decimal("0")


@dataclass(frozen=True)
class AirInLeakage:
    """
    Air in-leakage measurement and characteristics.

    Attributes:
        total_scfm: Total air in-leakage (SCFM)
        dissolved_oxygen_ppb: Dissolved oxygen in condensate (ppb)
        air_ejector_capacity_scfm: Air removal system capacity (SCFM)
        subcooling_c: Condensate subcooling (Celsius)
        non_condensable_fraction: Non-condensable gas fraction
    """
    total_scfm: Decimal
    dissolved_oxygen_ppb: Decimal = Decimal("0")
    air_ejector_capacity_scfm: Decimal = Decimal("50.0")
    subcooling_c: Decimal = Decimal("0")
    non_condensable_fraction: Decimal = Decimal("0")

    @property
    def utilization_percent(self) -> Decimal:
        """Calculate air removal system utilization."""
        if self.air_ejector_capacity_scfm <= Decimal("0"):
            return Decimal("100.0")
        return (self.total_scfm / self.air_ejector_capacity_scfm) * Decimal("100")

    @property
    def is_excessive(self) -> bool:
        """Check if air in-leakage exceeds typical limit (5 SCFM/100MW)."""
        return self.total_scfm > Decimal("5.0")


# ============================================================================
# PERFORMANCE METRIC DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class CleanlinessFactorReading:
    """
    Single cleanliness factor measurement with metadata.

    Attributes:
        timestamp: Measurement timestamp
        cf_value: Cleanliness factor (0.0-1.0)
        ua_actual_kw_k: Actual overall heat transfer coefficient-area
        ua_design_kw_k: Design overall heat transfer coefficient-area
        confidence: Measurement confidence (0.0-1.0)
        measurement_source: Source of measurement (calculated, sensor, etc.)
    """
    timestamp: datetime
    cf_value: Decimal
    ua_actual_kw_k: Decimal
    ua_design_kw_k: Decimal
    confidence: Decimal = Decimal("0.95")
    measurement_source: str = "calculated"

    def __post_init__(self) -> None:
        """Validate cleanliness factor is in valid range."""
        if not (Decimal("0") <= self.cf_value <= Decimal("1.5")):
            raise ValueError(f"CF value {self.cf_value} outside valid range [0, 1.5]")


@dataclass(frozen=True)
class VacuumReading:
    """
    Vacuum pressure measurement.

    Attributes:
        timestamp: Measurement timestamp
        pressure_kpa_abs: Absolute condenser pressure (kPa)
        pressure_mm_hg_abs: Absolute pressure in mm Hg
        backpressure_design_kpa: Design backpressure (kPa)
        deviation_kpa: Deviation from expected pressure
    """
    timestamp: datetime
    pressure_kpa_abs: Decimal
    pressure_mm_hg_abs: Optional[Decimal] = None
    backpressure_design_kpa: Optional[Decimal] = None
    deviation_kpa: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Validate vacuum pressure is physically reasonable."""
        if self.pressure_kpa_abs <= Decimal("0"):
            raise ValueError("Absolute pressure must be positive")
        if self.pressure_kpa_abs > Decimal("101.325"):
            raise ValueError("Condenser pressure cannot exceed atmospheric")


@dataclass(frozen=True)
class TemperatureDifferential:
    """
    Key temperature differentials for condenser analysis.

    Attributes:
        ttd_c: Terminal Temperature Difference (Celsius)
        approach_c: CW approach temperature (Celsius)
        cw_rise_c: CW temperature rise (Celsius)
        subcooling_c: Condensate subcooling (Celsius)
    """
    ttd_c: Decimal
    approach_c: Decimal
    cw_rise_c: Decimal
    subcooling_c: Decimal = Decimal("0")

    @property
    def lmtd_c(self) -> Decimal:
        """Calculate Log Mean Temperature Difference."""
        if self.ttd_c == self.approach_c:
            return self.ttd_c
        import math
        ttd = float(self.ttd_c)
        approach = float(self.approach_c)
        if ttd <= 0 or approach <= 0:
            return Decimal("0")
        lmtd = (ttd - approach) / math.log(ttd / approach)
        return Decimal(str(round(lmtd, 4)))


# ============================================================================
# CMMS INTEGRATION DATA CLASSES
# ============================================================================

@dataclass(frozen=True)
class CMMSWorkOrder:
    """
    CMMS work order data for maintenance integration.

    Attributes:
        work_order_id: Unique work order identifier
        equipment_id: Equipment/asset identifier
        description: Work order description
        priority: Priority level (1=highest)
        work_type: Type of work (PM, CM, etc.)
        status: Current status
        estimated_hours: Estimated labor hours
        estimated_cost_usd: Estimated total cost
        created_date: Work order creation date
        target_completion: Target completion date
    """
    work_order_id: str
    equipment_id: str
    description: str
    priority: int = 3
    work_type: str = "CM"
    status: str = "CREATED"
    estimated_hours: Decimal = Decimal("0")
    estimated_cost_usd: Decimal = Decimal("0")
    created_date: Optional[datetime] = None
    target_completion: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "work_order_id": self.work_order_id,
            "equipment_id": self.equipment_id,
            "description": self.description,
            "priority": self.priority,
            "work_type": self.work_type,
            "status": self.status,
            "estimated_hours": float(self.estimated_hours),
            "estimated_cost_usd": float(self.estimated_cost_usd),
            "created_date": self.created_date.isoformat() if self.created_date else None,
            "target_completion": self.target_completion.isoformat() if self.target_completion else None,
        }


# ============================================================================
# REFERENCE DATA
# ============================================================================

@dataclass(frozen=True)
class HEIStandardConditions:
    """
    HEI Standard reference conditions for condenser performance.

    Based on HEI Standards for Steam Surface Condensers, 12th Edition.
    """
    cw_inlet_temp_c: Decimal = Decimal("21.1")  # 70F
    cw_velocity_fps: Decimal = Decimal("7.0")   # ft/s
    tube_cleanliness: Decimal = Decimal("0.85") # 85% clean
    tube_material: TubeMaterial = TubeMaterial.ADMIRALTY_BRASS
    tube_gauge: int = 18  # BWG
    fouling_factor_m2_k_w: Decimal = Decimal("0.000088")

    @property
    def cw_velocity_m_s(self) -> Decimal:
        """Convert CW velocity to m/s."""
        return self.cw_velocity_fps * Decimal("0.3048")


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Material enums
    "TubeMaterial",
    "TubeSupport",
    "TubeEndConnection",
    # Failure enums
    "FailureMode",
    "FailureSeverity",
    # Cleaning enums
    "CleaningMethod",
    # Status enums
    "AlertLevel",
    "OperatingMode",
    "WaterSource",
    "VacuumControlMode",
    # Property dataclasses
    "SteamProperties",
    "CoolingWaterProperties",
    "AirInLeakage",
    # Metric dataclasses
    "CleanlinessFactorReading",
    "VacuumReading",
    "TemperatureDifferential",
    # CMMS integration
    "CMMSWorkOrder",
    # Reference data
    "HEIStandardConditions",
]
