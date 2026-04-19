# -*- coding: utf-8 -*-
"""
GL-024 AIRPREHEATER Agent - Configuration Module
=================================================

This module defines all configuration schemas for the GL-024 Air Preheater
Optimization Agent. Configuration is validated using Pydantic models with
strict typing and constraint validation.

Key Configuration Areas:
    - Leakage detection and monitoring (air-to-gas, gas-to-air)
    - Cold end corrosion prevention (acid dew point calculations)
    - Pressure drop monitoring and baseline comparison
    - Effectiveness tracking and degradation detection
    - Operating mode-specific thresholds (NORMAL, STARTUP, SHUTDOWN, SOOT_BLOWING)

Air Preheater Types Supported:
    - Regenerative (Ljungstrom, Rothemuhle)
    - Recuperative (Tubular, Plate)
    - Heat Pipe

Standards Reference:
    - ASME PTC 4.3: Air Heater Performance Test Code
    - API 560: Fired Heaters for General Refinery Service
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces

Engineering References:
    - Verhoff-Banchero correlation for acid dew point
    - Okkes correlation for acid dew point (European standard)
    - Pierce correlation for acid dew point (sulfuric acid)

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class PreheaterType(str, Enum):
    """Types of air preheaters supported by GL-024.

    Regenerative types use rotating heat storage elements.
    Recuperative types use fixed heat transfer surfaces.
    """
    # Regenerative (rotating) preheaters
    LJUNGSTROM = "ljungstrom"           # Vertical axis regenerative
    ROTHEMUHLE = "rothemuhle"           # Horizontal axis regenerative
    REGENERATIVE = "regenerative"       # Generic regenerative type

    # Recuperative (static) preheaters
    TUBULAR = "tubular"                 # Tube-based recuperator
    PLATE = "plate"                     # Plate-type recuperator
    CAST_IRON = "cast_iron"             # Cast iron sectional
    RECUPERATIVE = "recuperative"       # Generic recuperative type
    RECUPERATIVE_TUBULAR = "recuperative_tubular"  # Alias for compatibility
    RECUPERATIVE_PLATE = "recuperative_plate"      # Alias for compatibility

    # Special types
    HEAT_PIPE = "heat_pipe"             # Heat pipe air preheater
    GLASS_TUBE = "glass_tube"           # Corrosion-resistant glass tubes


# Backward compatibility alias
AirPreheaterType = PreheaterType


class FlowArrangement(str, Enum):
    """Flow arrangement for recuperative preheaters."""
    COUNTERFLOW = "counterflow"         # Most efficient, highest NTU
    PARALLEL_FLOW = "parallel_flow"     # Lower effectiveness
    CROSSFLOW = "crossflow"             # Common for tubular
    CROSSFLOW_MIXED = "crossflow_mixed" # Mixed on one side
    CROSSFLOW_UNMIXED = "crossflow_unmixed"  # Unmixed both sides


class OperatingMode(str, Enum):
    """Operating modes with different threshold requirements.

    Each mode has specific operational characteristics that affect
    leakage, pressure drop, and effectiveness monitoring.
    """
    NORMAL = "normal"                   # Steady-state operation
    STARTUP = "startup"                 # Cold start, relaxed limits
    SHUTDOWN = "shutdown"               # Controlled shutdown
    SOOT_BLOWING = "soot_blowing"       # During soot blower operation
    LOW_LOAD = "low_load"               # Below 50% load
    HOT_STANDBY = "hot_standby"         # Unit in hot standby


class FuelType(str, Enum):
    """Fuel types for acid dew point and fouling calculations."""
    NATURAL_GAS = "natural_gas"
    NO2_FUEL_OIL = "no2_fuel_oil"
    NO6_FUEL_OIL = "no6_fuel_oil"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    COAL_LIGNITE = "coal_lignite"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLET = "biomass_pellet"
    PETROLEUM_COKE = "petroleum_coke"
    REFINERY_GAS = "refinery_gas"


class SectorMaterial(str, Enum):
    """Air preheater sector/basket materials."""
    CARBON_STEEL = "carbon_steel"           # Standard, up to 700F
    CORTEN = "corten"                       # Weathering steel, corrosion resistant
    ENAMELED = "enameled"                   # Glass-enamel coated
    STAINLESS_409 = "stainless_409"         # Cold end protection
    STAINLESS_316 = "stainless_316"         # High corrosion resistance
    ALLOY_22 = "alloy_22"                   # Severe service


class LeakageType(str, Enum):
    """Types of air preheater leakage."""
    AIR_TO_GAS = "air_to_gas"               # Air leaking into gas side (most common)
    GAS_TO_AIR = "gas_to_air"               # Gas leaking into air side (rare, dangerous)
    INTERNAL = "internal"                    # Between sectors
    EXTERNAL = "external"                    # To atmosphere


class AcidDewPointMethod(str, Enum):
    """Acid dew point calculation methods.

    Different methods have varying accuracy depending on fuel type
    and operating conditions.
    """
    VERHOFF_BANCHERO = "verhoff_banchero"   # Most widely used
    OKKES = "okkes"                          # European standard
    PIERCE = "pierce"                        # Pierce method
    ZNO_CORRELATION = "zno_correlation"      # ZnO correlation
    MEASURED = "measured"                    # From in-situ probe


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


# =============================================================================
# ENGINEERING CONSTANTS
# =============================================================================

class AcidDewPointCoefficients(BaseModel):
    """
    Acid dew point correlation coefficients for different calculation methods.

    Verhoff-Banchero (1974):
        T_dew = 1000 / (2.276 - 0.0294*ln(pH2O) - 0.0858*ln(pSO3) + 0.0062*ln(pH2O)*ln(pSO3))
        where T_dew is in Kelvin, pH2O and pSO3 are partial pressures in mmHg

    Okkes (1987):
        T_dew = 203.25 + 27.6*log10(pH2O) + 10.83*log10(pSO3) + 1.06*(log10(pSO3)+8)^2.19
        where T_dew is in Celsius

    Pierce:
        T_dew = f(SO3_ppm, H2O_pct) - empirical lookup

    Engineering Rationale:
        - Verhoff-Banchero is industry standard for coal and oil firing
        - Okkes provides better accuracy for European low-sulfur fuels
        - Pierce is preferred for very high sulfur applications
        - ZnO correlation used when acid probe data available
    """

    # Verhoff-Banchero coefficients
    vb_a: float = Field(
        default=2.276,
        description="Verhoff-Banchero constant A (dimensionless)"
    )
    vb_b: float = Field(
        default=0.0294,
        description="Verhoff-Banchero coefficient B for ln(pH2O)"
    )
    vb_c: float = Field(
        default=0.0858,
        description="Verhoff-Banchero coefficient C for ln(pSO3)"
    )
    vb_d: float = Field(
        default=0.0062,
        description="Verhoff-Banchero coefficient D for cross-term"
    )

    # Okkes coefficients (metric)
    okkes_a: float = Field(
        default=203.25,
        description="Okkes constant A (Celsius)"
    )
    okkes_b: float = Field(
        default=27.6,
        description="Okkes coefficient B for log10(pH2O)"
    )
    okkes_c: float = Field(
        default=10.83,
        description="Okkes coefficient C for log10(pSO3)"
    )
    okkes_d: float = Field(
        default=1.06,
        description="Okkes coefficient D for (log10(pSO3)+8)^n term"
    )
    okkes_n: float = Field(
        default=2.19,
        description="Okkes exponent n"
    )

    # Pierce correlation adjustment factor
    pierce_adjustment: float = Field(
        default=1.0,
        ge=0.9,
        le=1.1,
        description="Pierce method adjustment factor for site-specific conditions"
    )


class NTURanges(BaseModel):
    """
    Typical NTU (Number of Transfer Units) ranges by preheater type.

    NTU = UA / (m_dot * Cp)_min

    Higher NTU means more heat transfer capability and higher effectiveness.
    These ranges are based on industry experience and design standards.

    Engineering Rationale:
        - Regenerative preheaters achieve highest NTU due to large heat storage mass
        - Ljungstrom typically 2.5-4.0 NTU for utility boilers
        - Tubular recuperators limited by tube-side pressure drop
        - Plate types offer compact high-NTU designs
        - Heat pipes limited by wick capacity and orientation
    """

    ljungstrom_min: float = Field(default=2.0, description="Ljungstrom minimum NTU")
    ljungstrom_max: float = Field(default=4.5, description="Ljungstrom maximum NTU")
    ljungstrom_typical: float = Field(default=3.2, description="Ljungstrom typical NTU")

    rothemuhle_min: float = Field(default=1.8, description="Rothemuhle minimum NTU")
    rothemuhle_max: float = Field(default=4.0, description="Rothemuhle maximum NTU")
    rothemuhle_typical: float = Field(default=2.8, description="Rothemuhle typical NTU")

    tubular_min: float = Field(default=1.0, description="Tubular minimum NTU")
    tubular_max: float = Field(default=3.0, description="Tubular maximum NTU")
    tubular_typical: float = Field(default=2.0, description="Tubular typical NTU")

    plate_min: float = Field(default=1.5, description="Plate minimum NTU")
    plate_max: float = Field(default=4.0, description="Plate maximum NTU")
    plate_typical: float = Field(default=2.5, description="Plate typical NTU")

    heat_pipe_min: float = Field(default=1.0, description="Heat pipe minimum NTU")
    heat_pipe_max: float = Field(default=2.5, description="Heat pipe maximum NTU")
    heat_pipe_typical: float = Field(default=1.8, description="Heat pipe typical NTU")

    cast_iron_min: float = Field(default=0.8, description="Cast iron minimum NTU")
    cast_iron_max: float = Field(default=2.0, description="Cast iron maximum NTU")
    cast_iron_typical: float = Field(default=1.4, description="Cast iron typical NTU")

    def get_range(self, preheater_type: PreheaterType) -> Tuple[float, float, float]:
        """Get NTU range for given preheater type."""
        ranges = {
            PreheaterType.LJUNGSTROM: (self.ljungstrom_min, self.ljungstrom_max, self.ljungstrom_typical),
            PreheaterType.ROTHEMUHLE: (self.rothemuhle_min, self.rothemuhle_max, self.rothemuhle_typical),
            PreheaterType.TUBULAR: (self.tubular_min, self.tubular_max, self.tubular_typical),
            PreheaterType.PLATE: (self.plate_min, self.plate_max, self.plate_typical),
            PreheaterType.HEAT_PIPE: (self.heat_pipe_min, self.heat_pipe_max, self.heat_pipe_typical),
            PreheaterType.CAST_IRON: (self.cast_iron_min, self.cast_iron_max, self.cast_iron_typical),
        }
        return ranges.get(preheater_type, (1.0, 3.0, 2.0))


class HeatCapacityCorrelations(BaseModel):
    """
    Heat capacity correlations for air and flue gas.

    Cp(T) = a + b*T + c*T^2 + d*T^3 (kJ/kg-K or BTU/lb-F)
    where T is temperature in Kelvin (metric) or Rankine (imperial)

    Engineering Rationale:
        - Air Cp increases ~0.5% per 100F above ambient
        - Flue gas Cp depends on composition (CO2, H2O, N2, SO2)
        - Coal flue gas has higher Cp than natural gas due to CO2/H2O content
        - Accuracy important for effectiveness calculations (ASME PTC 4.3)
    """

    # Air heat capacity polynomial coefficients (SI: kJ/kg-K, T in K)
    air_a: float = Field(default=1.0036, description="Air Cp constant term")
    air_b: float = Field(default=2.014e-5, description="Air Cp linear coefficient")
    air_c: float = Field(default=4.564e-8, description="Air Cp quadratic coefficient")
    air_d: float = Field(default=-1.449e-11, description="Air Cp cubic coefficient")

    # Flue gas Cp coefficients for natural gas combustion (SI)
    flue_gas_ng_a: float = Field(default=1.040, description="NG flue gas Cp constant")
    flue_gas_ng_b: float = Field(default=3.2e-5, description="NG flue gas Cp linear")
    flue_gas_ng_c: float = Field(default=5.1e-8, description="NG flue gas Cp quadratic")

    # Flue gas Cp coefficients for coal combustion (SI)
    flue_gas_coal_a: float = Field(default=1.055, description="Coal flue gas Cp constant")
    flue_gas_coal_b: float = Field(default=3.8e-5, description="Coal flue gas Cp linear")
    flue_gas_coal_c: float = Field(default=4.8e-8, description="Coal flue gas Cp quadratic")

    # Flue gas Cp coefficients for oil combustion (SI)
    flue_gas_oil_a: float = Field(default=1.048, description="Oil flue gas Cp constant")
    flue_gas_oil_b: float = Field(default=3.5e-5, description="Oil flue gas Cp linear")
    flue_gas_oil_c: float = Field(default=5.0e-8, description="Oil flue gas Cp quadratic")

    # Reference temperature
    reference_temp_k: float = Field(default=298.15, description="Reference temperature (K)")

    # Correction factors for excess air
    excess_air_correction_per_pct: float = Field(
        default=-0.001,
        description="Cp correction per 1% excess air (fraction)"
    )


class FoulingFactors(BaseModel):
    """
    Fouling factors by fuel type for air preheater surfaces.

    Fouling reduces heat transfer effectiveness and increases pressure drop.
    Values are dimensionless multipliers applied to clean heat transfer.

    Fouling Factor = 1 / (1 + Rf * U_clean * A)

    where Rf is fouling resistance (m2-K/W or hr-ft2-F/BTU)

    Engineering Rationale:
        - Coal firing produces highest fouling due to ash
        - Biomass can cause severe fouling and corrosion
        - Natural gas produces minimal fouling
        - Oil fouling depends heavily on sulfur and vanadium content
        - Soot blowing effectiveness varies by fuel type
    """

    # Gas-side fouling resistance (m2-K/W)
    natural_gas: float = Field(
        default=0.00009,
        ge=0,
        description="Natural gas flue gas fouling resistance (m2-K/W)"
    )
    no2_fuel_oil: float = Field(
        default=0.00018,
        ge=0,
        description="No. 2 fuel oil flue gas fouling resistance"
    )
    no6_fuel_oil: float = Field(
        default=0.00053,
        ge=0,
        description="No. 6 fuel oil flue gas fouling resistance"
    )
    coal_bituminous: float = Field(
        default=0.00088,
        ge=0,
        description="Bituminous coal flue gas fouling resistance"
    )
    coal_sub_bituminous: float = Field(
        default=0.00070,
        ge=0,
        description="Sub-bituminous coal flue gas fouling resistance"
    )
    coal_lignite: float = Field(
        default=0.00106,
        ge=0,
        description="Lignite coal flue gas fouling resistance"
    )
    biomass_wood: float = Field(
        default=0.00088,
        ge=0,
        description="Wood biomass flue gas fouling resistance"
    )
    biomass_pellet: float = Field(
        default=0.00070,
        ge=0,
        description="Wood pellet flue gas fouling resistance"
    )
    petroleum_coke: float = Field(
        default=0.00106,
        ge=0,
        description="Petroleum coke flue gas fouling resistance"
    )

    # Air-side fouling (typically minimal)
    air_side_clean: float = Field(
        default=0.00005,
        ge=0,
        description="Clean air fouling resistance"
    )
    air_side_dusty: float = Field(
        default=0.00018,
        ge=0,
        description="Dusty environment air fouling resistance"
    )

    # Fouling rate (m2-K/W per day of operation)
    fouling_rate_coal: float = Field(
        default=0.000002,
        ge=0,
        description="Coal firing fouling accumulation rate per day"
    )
    fouling_rate_oil: float = Field(
        default=0.0000015,
        ge=0,
        description="Oil firing fouling accumulation rate per day"
    )
    fouling_rate_gas: float = Field(
        default=0.0000005,
        ge=0,
        description="Gas firing fouling accumulation rate per day"
    )

    def get_fouling_factor(self, fuel_type: FuelType) -> float:
        """Get fouling factor for given fuel type."""
        factors = {
            FuelType.NATURAL_GAS: self.natural_gas,
            FuelType.NO2_FUEL_OIL: self.no2_fuel_oil,
            FuelType.NO6_FUEL_OIL: self.no6_fuel_oil,
            FuelType.COAL_BITUMINOUS: self.coal_bituminous,
            FuelType.COAL_SUB_BITUMINOUS: self.coal_sub_bituminous,
            FuelType.COAL_LIGNITE: self.coal_lignite,
            FuelType.BIOMASS_WOOD: self.biomass_wood,
            FuelType.BIOMASS_PELLET: self.biomass_pellet,
            FuelType.PETROLEUM_COKE: self.petroleum_coke,
            FuelType.REFINERY_GAS: self.natural_gas,  # Similar to NG
        }
        return factors.get(fuel_type, self.coal_bituminous)


# =============================================================================
# REGULATORY REFERENCES
# =============================================================================

class ASMEPTC43Tolerances(BaseModel):
    """
    ASME PTC 4.3 Air Heater Test Code tolerances and requirements.

    ASME PTC 4.3 defines the procedures for testing air heaters to determine
    their thermal performance. These tolerances apply to acceptance testing.

    Engineering Rationale:
        - Temperature measurement tolerance critical for leakage calculations
        - Flow measurement accuracy affects effectiveness determination
        - Test duration ensures steady-state conditions achieved
        - Multiple test runs required for statistical significance
    """

    # Temperature measurement tolerances
    temp_measurement_tolerance_f: float = Field(
        default=2.0,
        ge=0,
        le=5.0,
        description="ASME PTC 4.3 temperature measurement tolerance (F)"
    )
    temp_spatial_variation_max_f: float = Field(
        default=10.0,
        ge=0,
        description="Maximum spatial temperature variation across duct (F)"
    )

    # Flow measurement tolerances
    air_flow_tolerance_pct: float = Field(
        default=2.0,
        ge=0,
        le=5.0,
        description="Air flow measurement tolerance (%)"
    )
    gas_flow_tolerance_pct: float = Field(
        default=2.5,
        ge=0,
        le=5.0,
        description="Gas flow measurement tolerance (%)"
    )

    # Leakage test requirements
    leakage_test_o2_precision_pct: float = Field(
        default=0.1,
        ge=0,
        le=0.5,
        description="O2 analyzer precision for leakage test (%)"
    )
    min_o2_difference_pct: float = Field(
        default=0.5,
        ge=0,
        description="Minimum O2 difference for valid leakage calculation (%)"
    )

    # Pressure measurement
    dp_measurement_tolerance_in_wc: float = Field(
        default=0.05,
        ge=0,
        description="Pressure drop measurement tolerance (in. WC)"
    )

    # Test duration requirements
    min_test_duration_minutes: int = Field(
        default=30,
        ge=15,
        le=120,
        description="Minimum test duration for steady state (minutes)"
    )
    steady_state_variation_pct: float = Field(
        default=2.0,
        ge=0,
        le=5.0,
        description="Maximum variation for steady state determination (%)"
    )
    num_test_runs: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of test runs required"
    )

    # Load requirements
    min_test_load_pct: float = Field(
        default=60.0,
        ge=50.0,
        le=80.0,
        description="Minimum load for valid test (%)"
    )
    load_variation_during_test_pct: float = Field(
        default=3.0,
        ge=0,
        le=5.0,
        description="Maximum load variation during test (%)"
    )


class API560Requirements(BaseModel):
    """
    API 560 Fired Heaters for General Refinery Service requirements.

    API 560 provides design requirements for fired heaters including
    air preheater systems. Key requirements for GL-024 include thermal
    design, materials, and operating limits.

    Engineering Rationale:
        - API 560 ensures adequate design margins for refinery service
        - Minimum approach temperatures prevent thermal stress
        - Material selection based on sulfur content and dew point
        - Tube velocity limits prevent erosion and vibration
    """

    # Thermal design requirements
    min_cold_end_approach_f: float = Field(
        default=50.0,
        ge=25,
        le=100,
        description="Minimum cold end approach temperature (F)"
    )
    max_gas_inlet_temp_f: float = Field(
        default=850.0,
        ge=500,
        le=1200,
        description="Maximum gas inlet temperature for carbon steel (F)"
    )
    max_tube_wall_temp_f: float = Field(
        default=750.0,
        ge=400,
        le=1000,
        description="Maximum tube wall temperature (F)"
    )

    # Design margins
    design_margin_pct: float = Field(
        default=10.0,
        ge=5,
        le=25,
        description="Design margin on heat transfer area (%)"
    )
    fouling_margin_pct: float = Field(
        default=20.0,
        ge=10,
        le=40,
        description="Fouling margin on heat transfer (%)"
    )

    # Velocity limits
    max_gas_velocity_fps: float = Field(
        default=80.0,
        ge=40,
        le=120,
        description="Maximum gas velocity (ft/s)"
    )
    max_air_velocity_fps: float = Field(
        default=60.0,
        ge=30,
        le=100,
        description="Maximum air velocity (ft/s)"
    )
    min_velocity_fps: float = Field(
        default=15.0,
        ge=5,
        le=30,
        description="Minimum velocity to prevent settling (ft/s)"
    )

    # Pressure drop limits
    max_gas_dp_in_wc: float = Field(
        default=6.0,
        ge=2,
        le=12,
        description="Maximum gas-side pressure drop (in. WC)"
    )
    max_air_dp_in_wc: float = Field(
        default=8.0,
        ge=2,
        le=15,
        description="Maximum air-side pressure drop (in. WC)"
    )

    # Structural requirements
    min_tube_wall_thickness_in: float = Field(
        default=0.083,
        ge=0.049,
        le=0.25,
        description="Minimum tube wall thickness (inches)"
    )
    corrosion_allowance_in: float = Field(
        default=0.0625,
        ge=0.03125,
        le=0.125,
        description="Corrosion allowance (inches)"
    )


class NFPA85Requirements(BaseModel):
    """
    NFPA 85 Boiler and Combustion Systems Hazards Code requirements.

    NFPA 85 addresses safety requirements for combustion systems including
    air preheaters. Key concerns are fire prevention, explosion protection,
    and proper interlock sequences.

    Engineering Rationale:
        - Air preheater fires can occur from combustibles in deposits
        - Proper purge before startup removes explosive mixtures
        - Temperature monitoring detects incipient fires
        - Interlock to boiler trip on high temperature or fire detection
    """

    # Fire prevention
    max_sector_temp_rise_rate_f_per_min: float = Field(
        default=50.0,
        ge=20,
        le=100,
        description="Max temperature rise rate indicating fire (F/min)"
    )
    fire_detection_temp_f: float = Field(
        default=500.0,
        ge=350,
        le=700,
        description="Temperature threshold for fire detection (F above normal)"
    )

    # Purge requirements
    min_purge_time_s: int = Field(
        default=300,
        ge=60,
        le=600,
        description="Minimum purge time before ignition (seconds)"
    )
    min_purge_airflow_pct: float = Field(
        default=25.0,
        ge=15,
        le=40,
        description="Minimum airflow during purge (% of rated)"
    )

    # Interlock requirements
    low_air_flow_trip_pct: float = Field(
        default=25.0,
        ge=15,
        le=40,
        description="Low airflow trip setpoint (% of rated)"
    )
    high_gas_temp_alarm_f: float = Field(
        default=750.0,
        ge=500,
        le=1000,
        description="High gas inlet temperature alarm (F)"
    )
    high_gas_temp_trip_f: float = Field(
        default=850.0,
        ge=600,
        le=1100,
        description="High gas inlet temperature trip (F)"
    )

    # Water wash requirements
    max_water_wash_temp_f: float = Field(
        default=300.0,
        ge=200,
        le=400,
        description="Maximum preheater temp to begin water wash (F)"
    )
    min_water_wash_duration_min: int = Field(
        default=30,
        ge=15,
        le=60,
        description="Minimum water wash duration (minutes)"
    )


class NFPA86Requirements(BaseModel):
    """
    NFPA 86 Standard for Ovens and Furnaces requirements.

    NFPA 86 applies to industrial furnaces and ovens, including
    heat recovery systems. Relevant to process heater air preheaters.

    Engineering Rationale:
        - Class A ovens (flammable volatiles) have stringent requirements
        - Class B ovens (non-flammable) have relaxed requirements
        - Air preheaters handling combustible dusts need special attention
        - Temperature classification affects material selection
    """

    # Classification
    furnace_class: str = Field(
        default="B",
        description="Furnace classification per NFPA 86 (A, B, C, or D)"
    )

    # Safety requirements
    max_operating_temp_f: float = Field(
        default=1000.0,
        ge=500,
        le=2000,
        description="Maximum operating temperature (F)"
    )
    min_exhaust_temp_f: float = Field(
        default=250.0,
        ge=150,
        le=400,
        description="Minimum exhaust temperature to prevent condensation (F)"
    )

    # Ventilation requirements
    min_air_changes_per_hour: float = Field(
        default=4.0,
        ge=2,
        le=10,
        description="Minimum air changes per hour"
    )

    # Interlock verification
    interlock_test_interval_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Interlock test interval (days)"
    )


# =============================================================================
# THRESHOLD CONFIGURATIONS
# =============================================================================

class LeakageThresholds(BaseModel):
    """
    Air preheater leakage detection thresholds.

    Leakage is the most critical operating parameter for regenerative
    air preheaters. Air-to-gas leakage reduces boiler efficiency;
    gas-to-air leakage is a safety concern (CO, combustibles in air).

    Leakage Calculation (ASME PTC 4.3 Method):
        Leakage % = [(O2_out - O2_in) / (21 - O2_out)] * 100
        where O2 is measured on the gas side

    Engineering Rationale:
        - 5-8% leakage is typical for well-maintained Ljungstrom
        - >12% leakage indicates seal wear requiring attention
        - >15% leakage significantly impacts efficiency
        - Gas-to-air leakage is rare but dangerous if >1%
        - Startup/shutdown allow higher leakage (thermal expansion)
        - Soot blowing temporarily increases apparent leakage
    """

    # Air-to-gas leakage thresholds (percentage of flue gas flow)
    air_to_gas_warning_pct: float = Field(
        default=8.0,
        ge=3.0,
        le=15.0,
        description=(
            "Air-to-gas leakage warning threshold (%). "
            "Typical new preheater: 5-6%. Trigger maintenance review."
        )
    )
    air_to_gas_alarm_pct: float = Field(
        default=12.0,
        ge=5.0,
        le=20.0,
        description=(
            "Air-to-gas leakage alarm threshold (%). "
            "Indicates significant seal wear. Schedule seal inspection."
        )
    )
    air_to_gas_critical_pct: float = Field(
        default=15.0,
        ge=8.0,
        le=25.0,
        description=(
            "Air-to-gas leakage critical threshold (%). "
            "Significant efficiency impact. Consider outage for repair."
        )
    )

    # Gas-to-air leakage thresholds (safety critical)
    gas_to_air_warning_pct: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description=(
            "Gas-to-air leakage warning threshold (%). "
            "Safety concern - CO entering air duct. Immediate investigation."
        )
    )
    gas_to_air_alarm_pct: float = Field(
        default=1.0,
        ge=0.3,
        le=3.0,
        description=(
            "Gas-to-air leakage alarm threshold (%). "
            "Potential safety hazard. Operator notification required."
        )
    )
    gas_to_air_trip_pct: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description=(
            "Gas-to-air leakage trip threshold (%). "
            "Unsafe condition. Consider unit trip per NFPA 85."
        )
    )

    # Mode-specific adjustments (multipliers)
    startup_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=2.0,
        description="Threshold multiplier during startup (thermal expansion)"
    )
    shutdown_multiplier: float = Field(
        default=1.3,
        ge=1.0,
        le=1.8,
        description="Threshold multiplier during shutdown"
    )
    soot_blowing_multiplier: float = Field(
        default=1.2,
        ge=1.0,
        le=1.5,
        description="Threshold multiplier during soot blowing"
    )

    # Trend analysis settings
    trend_window_hours: int = Field(
        default=168,
        ge=24,
        le=720,
        description="Hours of data for leakage trend analysis"
    )
    trend_threshold_pct_per_day: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="Leakage increase rate warning (% per day)"
    )

    @validator("air_to_gas_alarm_pct")
    def alarm_greater_than_warning(cls, v, values):
        """Ensure alarm > warning."""
        if "air_to_gas_warning_pct" in values and v <= values["air_to_gas_warning_pct"]:
            raise ValueError("Alarm threshold must be greater than warning")
        return v


class ColdEndMarginThresholds(BaseModel):
    """
    Cold end corrosion prevention thresholds.

    The cold end of air preheaters operates near the acid dew point of
    flue gas. Operating below the dew point causes severe corrosion.

    Margin = Metal Temperature - Acid Dew Point

    Engineering Rationale:
        - Acid dew point depends on SO3 and H2O in flue gas
        - Typical coal firing: 250-300F acid dew point
        - Natural gas: 150-200F acid dew point
        - Maintain 20-30F margin above dew point
        - <10F margin causes rapid corrosion (mils/year)
        - Corrosion rate increases exponentially below dew point
        - Low load operation is highest risk (low metal temps)
    """

    # Cold end margin thresholds (degrees F above acid dew point)
    margin_warning_f: float = Field(
        default=20.0,
        ge=10.0,
        le=40.0,
        description=(
            "Cold end margin warning threshold (F above acid dew point). "
            "Acceptable but monitor closely. Consider raising air preheat."
        )
    )
    margin_alarm_f: float = Field(
        default=10.0,
        ge=5.0,
        le=25.0,
        description=(
            "Cold end margin alarm threshold (F above acid dew point). "
            "Corrosion risk high. Take corrective action (raise inlet air temp)."
        )
    )
    margin_critical_f: float = Field(
        default=5.0,
        ge=0.0,
        le=15.0,
        description=(
            "Cold end margin critical threshold (F above acid dew point). "
            "Severe corrosion imminent. Emergency action required."
        )
    )

    # Absolute minimum temperatures
    min_cold_end_temp_coal_f: float = Field(
        default=280.0,
        ge=250,
        le=350,
        description="Minimum cold end temperature for coal firing (F)"
    )
    min_cold_end_temp_oil_f: float = Field(
        default=270.0,
        ge=240,
        le=320,
        description="Minimum cold end temperature for oil firing (F)"
    )
    min_cold_end_temp_gas_f: float = Field(
        default=200.0,
        ge=150,
        le=250,
        description="Minimum cold end temperature for gas firing (F)"
    )

    # Corrosion rate thresholds (mils per year)
    corrosion_rate_warning_mpy: float = Field(
        default=5.0,
        ge=1.0,
        le=15.0,
        description="Corrosion rate warning (mils per year)"
    )
    corrosion_rate_alarm_mpy: float = Field(
        default=10.0,
        ge=3.0,
        le=30.0,
        description="Corrosion rate alarm (mils per year)"
    )

    # Temperature monitoring
    cold_end_temp_scan_interval_s: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Cold end temperature scan interval (seconds)"
    )

    @validator("margin_alarm_f")
    def alarm_less_than_warning(cls, v, values):
        """Ensure alarm < warning for margin thresholds."""
        if "margin_warning_f" in values and v >= values["margin_warning_f"]:
            raise ValueError("Alarm margin must be less than warning margin")
        return v


class PressureDropThresholds(BaseModel):
    """
    Pressure drop monitoring thresholds.

    Pressure drop across the air preheater increases with fouling and
    structural damage. Monitoring vs. baseline indicates condition.

    Pressure Drop Ratio = Current DP / Baseline DP (corrected for flow)

    Engineering Rationale:
        - DP increases with flow^2 (correct to design flow)
        - 30% increase indicates significant fouling
        - 50% increase requires cleaning/maintenance
        - High DP reduces fan capacity and load capability
        - Sudden DP changes indicate structural issues
        - Gas-side DP more sensitive to fouling
        - Air-side DP indicates filter/inlet blockage
    """

    # Pressure drop ratio thresholds (actual/baseline at same flow)
    dp_ratio_warning: float = Field(
        default=1.3,
        ge=1.1,
        le=1.6,
        description=(
            "DP ratio warning threshold. "
            "30% above baseline indicates fouling accumulation. "
            "Schedule cleaning or soot blowing."
        )
    )
    dp_ratio_alarm: float = Field(
        default=1.5,
        ge=1.2,
        le=2.0,
        description=(
            "DP ratio alarm threshold. "
            "50% above baseline indicates heavy fouling. "
            "Clean at next opportunity."
        )
    )
    dp_ratio_critical: float = Field(
        default=1.8,
        ge=1.4,
        le=2.5,
        description=(
            "DP ratio critical threshold. "
            "Near operational limit. Urgent cleaning required."
        )
    )

    # Absolute pressure drop limits (in. WC)
    max_gas_dp_in_wc: float = Field(
        default=6.0,
        ge=2.0,
        le=12.0,
        description="Maximum allowable gas-side DP (in. WC)"
    )
    max_air_dp_in_wc: float = Field(
        default=8.0,
        ge=2.0,
        le=15.0,
        description="Maximum allowable air-side DP (in. WC)"
    )

    # Rate of change detection
    dp_rate_warning_in_wc_per_hour: float = Field(
        default=0.1,
        ge=0.01,
        le=0.5,
        description="DP increase rate warning (in. WC per hour)"
    )
    dp_sudden_change_pct: float = Field(
        default=20.0,
        ge=10.0,
        le=50.0,
        description="Sudden DP change threshold (%) - indicates structural issue"
    )

    # Mode-specific adjustments
    soot_blowing_dp_spike_allowed_pct: float = Field(
        default=50.0,
        ge=20.0,
        le=100.0,
        description="Allowed DP spike during soot blowing (%)"
    )

    @validator("dp_ratio_alarm")
    def alarm_greater_than_warning(cls, v, values):
        """Ensure alarm > warning."""
        if "dp_ratio_warning" in values and v <= values["dp_ratio_warning"]:
            raise ValueError("Alarm threshold must be greater than warning")
        return v


class EffectivenessThresholds(BaseModel):
    """
    Heat transfer effectiveness monitoring thresholds.

    Effectiveness = Actual Heat Transfer / Maximum Possible Heat Transfer
                  = (T_air_out - T_air_in) / (T_gas_in - T_air_in)

    Engineering Rationale:
        - Design effectiveness typically 70-85% for regenerative
        - 5% degradation indicates fouling or leakage issues
        - 10% degradation requires maintenance intervention
        - Effectiveness affected by leakage, fouling, seal wear
        - Load correction required for accurate comparison
        - Seasonal ambient temperature affects baseline
    """

    # Effectiveness degradation thresholds (% below design)
    effectiveness_warning_pct: float = Field(
        default=5.0,
        ge=2.0,
        le=10.0,
        description=(
            "Effectiveness degradation warning (% below design). "
            "5% degradation indicates fouling or increased leakage. "
            "Review operating parameters and schedule inspection."
        )
    )
    effectiveness_alarm_pct: float = Field(
        default=10.0,
        ge=5.0,
        le=20.0,
        description=(
            "Effectiveness degradation alarm (% below design). "
            "10% degradation significantly impacts efficiency. "
            "Maintenance required."
        )
    )
    effectiveness_critical_pct: float = Field(
        default=15.0,
        ge=8.0,
        le=25.0,
        description=(
            "Effectiveness degradation critical (% below design). "
            "Major performance issue. Urgent maintenance needed."
        )
    )

    # Absolute effectiveness limits
    min_effectiveness_pct: float = Field(
        default=50.0,
        ge=30.0,
        le=70.0,
        description="Minimum acceptable effectiveness (%)"
    )
    design_effectiveness_pct: float = Field(
        default=80.0,
        ge=60.0,
        le=95.0,
        description="Design effectiveness (%)"
    )

    # Load correction
    min_load_for_effectiveness_calc_pct: float = Field(
        default=40.0,
        ge=25.0,
        le=60.0,
        description="Minimum load for valid effectiveness calculation (%)"
    )
    effectiveness_load_correction_enabled: bool = Field(
        default=True,
        description="Apply load correction to effectiveness calculation"
    )

    # Trend analysis
    effectiveness_trend_window_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Days of data for effectiveness trend analysis"
    )
    effectiveness_degradation_rate_pct_per_day: float = Field(
        default=0.05,
        ge=0.01,
        le=0.2,
        description="Effectiveness degradation rate warning (% per day)"
    )

    @validator("effectiveness_alarm_pct")
    def alarm_greater_than_warning(cls, v, values):
        """Ensure alarm > warning."""
        if "effectiveness_warning_pct" in values and v <= values["effectiveness_warning_pct"]:
            raise ValueError("Alarm threshold must be greater than warning")
        return v


class AirPreheaterThresholds(BaseModel):
    """
    Complete threshold configuration for GL-024 Air Preheater Agent.

    Combines all threshold categories into a single configuration class.
    Each threshold category is documented with engineering rationale.
    """

    leakage: LeakageThresholds = Field(
        default_factory=LeakageThresholds,
        description="Air preheater leakage detection thresholds"
    )
    cold_end_margin: ColdEndMarginThresholds = Field(
        default_factory=ColdEndMarginThresholds,
        description="Cold end corrosion prevention thresholds"
    )
    pressure_drop: PressureDropThresholds = Field(
        default_factory=PressureDropThresholds,
        description="Pressure drop monitoring thresholds"
    )
    effectiveness: EffectivenessThresholds = Field(
        default_factory=EffectivenessThresholds,
        description="Heat transfer effectiveness thresholds"
    )


# =============================================================================
# OPERATING MODE CONFIGURATION
# =============================================================================

class ModeSpecificLimits(BaseModel):
    """
    Operating mode-specific limit adjustments.

    Different operating modes require different threshold settings
    to avoid nuisance alarms while maintaining safety.
    """

    # Temperature limits by mode
    max_gas_inlet_temp_f: float = Field(
        default=850.0,
        description="Maximum gas inlet temperature (F)"
    )
    min_air_inlet_temp_f: float = Field(
        default=40.0,
        description="Minimum air inlet temperature (F)"
    )
    max_metal_temp_f: float = Field(
        default=700.0,
        description="Maximum metal temperature (F)"
    )

    # Alarm delay adjustments
    alarm_delay_multiplier: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Alarm delay multiplier for this mode"
    )

    # Effectiveness requirements
    min_effectiveness_pct: float = Field(
        default=50.0,
        ge=20.0,
        le=80.0,
        description="Minimum effectiveness for this mode (%)"
    )

    # Leakage allowance
    leakage_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        le=2.0,
        description="Leakage threshold multiplier for this mode"
    )


class OperatingModeConfig(BaseModel):
    """
    Operating mode configurations with mode-specific limits.

    Each operating mode has different requirements based on
    thermal conditions, transient behavior, and operational needs.
    """

    normal: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=850.0,
            min_air_inlet_temp_f=40.0,
            max_metal_temp_f=700.0,
            alarm_delay_multiplier=1.0,
            min_effectiveness_pct=65.0,
            leakage_multiplier=1.0,
        ),
        description="Normal steady-state operation limits"
    )

    startup: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=700.0,
            min_air_inlet_temp_f=32.0,
            max_metal_temp_f=600.0,
            alarm_delay_multiplier=2.0,
            min_effectiveness_pct=40.0,
            leakage_multiplier=1.5,
        ),
        description="Cold startup limits (relaxed thresholds)"
    )

    shutdown: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=600.0,
            min_air_inlet_temp_f=40.0,
            max_metal_temp_f=500.0,
            alarm_delay_multiplier=1.5,
            min_effectiveness_pct=30.0,
            leakage_multiplier=1.3,
        ),
        description="Controlled shutdown limits"
    )

    soot_blowing: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=850.0,
            min_air_inlet_temp_f=40.0,
            max_metal_temp_f=700.0,
            alarm_delay_multiplier=1.5,
            min_effectiveness_pct=55.0,
            leakage_multiplier=1.2,
        ),
        description="Soot blowing operation limits"
    )

    low_load: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=750.0,
            min_air_inlet_temp_f=50.0,
            max_metal_temp_f=650.0,
            alarm_delay_multiplier=1.2,
            min_effectiveness_pct=50.0,
            leakage_multiplier=1.1,
        ),
        description="Low load operation limits (<50% load)"
    )

    hot_standby: ModeSpecificLimits = Field(
        default_factory=lambda: ModeSpecificLimits(
            max_gas_inlet_temp_f=500.0,
            min_air_inlet_temp_f=40.0,
            max_metal_temp_f=400.0,
            alarm_delay_multiplier=3.0,
            min_effectiveness_pct=20.0,
            leakage_multiplier=1.5,
        ),
        description="Hot standby limits"
    )

    def get_limits(self, mode: OperatingMode) -> ModeSpecificLimits:
        """Get limits for specified operating mode."""
        mode_map = {
            OperatingMode.NORMAL: self.normal,
            OperatingMode.STARTUP: self.startup,
            OperatingMode.SHUTDOWN: self.shutdown,
            OperatingMode.SOOT_BLOWING: self.soot_blowing,
            OperatingMode.LOW_LOAD: self.low_load,
            OperatingMode.HOT_STANDBY: self.hot_standby,
        }
        return mode_map.get(mode, self.normal)


# =============================================================================
# DESIGN CONFIGURATION
# =============================================================================

class PreheaterDesignConfig(BaseModel):
    """
    Air preheater design specifications.

    Captures the physical design parameters needed for performance
    calculations and monitoring.
    """

    # Physical design
    preheater_type: PreheaterType = Field(
        default=PreheaterType.LJUNGSTROM,
        description="Type of air preheater"
    )
    flow_arrangement: FlowArrangement = Field(
        default=FlowArrangement.COUNTERFLOW,
        description="Flow arrangement (recuperative types)"
    )

    # Dimensions
    rotor_diameter_ft: float = Field(
        default=40.0,
        gt=0,
        description="Rotor diameter for regenerative type (ft)"
    )
    rotor_depth_ft: float = Field(
        default=4.0,
        gt=0,
        description="Rotor depth for regenerative type (ft)"
    )
    rotation_speed_rpm: float = Field(
        default=1.0,
        ge=0.5,
        le=3.0,
        description="Rotor rotation speed (rpm)"
    )

    # Heat transfer surface
    total_surface_area_ft2: float = Field(
        default=100000.0,
        gt=0,
        description="Total heat transfer surface area (ft2)"
    )
    surface_type: str = Field(
        default="notched_flat",
        description="Surface type (notched_flat, corrugated, dnh, etc.)"
    )

    # Sector configuration
    num_sectors: int = Field(
        default=24,
        ge=12,
        le=48,
        description="Number of sectors in rotor"
    )
    sector_material: SectorMaterial = Field(
        default=SectorMaterial.CORTEN,
        description="Sector/basket material"
    )
    cold_end_material: SectorMaterial = Field(
        default=SectorMaterial.ENAMELED,
        description="Cold end sector material (corrosion resistant)"
    )

    # Seals
    radial_seal_type: str = Field(
        default="adjustable",
        description="Radial seal type"
    )
    axial_seal_type: str = Field(
        default="sector_plate",
        description="Axial seal type"
    )
    circumferential_seal_type: str = Field(
        default="leaf",
        description="Circumferential seal type"
    )

    # Design temperatures
    design_gas_inlet_temp_f: float = Field(
        default=750.0,
        ge=400,
        le=1200,
        description="Design gas inlet temperature (F)"
    )
    design_gas_outlet_temp_f: float = Field(
        default=300.0,
        ge=150,
        le=500,
        description="Design gas outlet temperature (F)"
    )
    design_air_inlet_temp_f: float = Field(
        default=80.0,
        ge=0,
        le=150,
        description="Design air inlet temperature (F)"
    )
    design_air_outlet_temp_f: float = Field(
        default=600.0,
        ge=300,
        le=800,
        description="Design air outlet temperature (F)"
    )

    # Design flows
    design_gas_flow_lb_hr: float = Field(
        default=2000000.0,
        gt=0,
        description="Design gas mass flow rate (lb/hr)"
    )
    design_air_flow_lb_hr: float = Field(
        default=1800000.0,
        gt=0,
        description="Design air mass flow rate (lb/hr)"
    )

    # Design pressure drops
    design_gas_dp_in_wc: float = Field(
        default=3.0,
        gt=0,
        le=10.0,
        description="Design gas-side pressure drop (in. WC)"
    )
    design_air_dp_in_wc: float = Field(
        default=4.0,
        gt=0,
        le=12.0,
        description="Design air-side pressure drop (in. WC)"
    )

    # Performance
    design_effectiveness: float = Field(
        default=0.80,
        ge=0.5,
        le=0.95,
        description="Design heat transfer effectiveness"
    )
    design_ntu: float = Field(
        default=3.2,
        gt=0,
        le=6.0,
        description="Design NTU value"
    )
    design_leakage_pct: float = Field(
        default=6.0,
        ge=2.0,
        le=12.0,
        description="Design air-to-gas leakage (%)"
    )

    class Config:
        use_enum_values = True


class SootBlowerConfig(BaseModel):
    """Soot blower configuration for air preheater cleaning."""

    num_soot_blowers: int = Field(
        default=4,
        ge=0,
        le=12,
        description="Number of soot blowers"
    )
    blower_type: str = Field(
        default="rotary",
        description="Soot blower type (rotary, stationary, acoustic)"
    )

    # Steam consumption
    steam_pressure_psig: float = Field(
        default=200.0,
        ge=50,
        le=600,
        description="Soot blowing steam pressure (psig)"
    )
    steam_flow_per_blower_lb: float = Field(
        default=1000.0,
        gt=0,
        description="Steam consumption per blower cycle (lb)"
    )

    # Scheduling
    fixed_schedule_enabled: bool = Field(
        default=False,
        description="Use fixed schedule vs. intelligent"
    )
    fixed_interval_hours: float = Field(
        default=8.0,
        ge=2,
        le=24,
        description="Fixed schedule interval (hours)"
    )
    min_interval_hours: float = Field(
        default=2.0,
        ge=1,
        le=8,
        description="Minimum interval between blowing (hours)"
    )
    max_interval_hours: float = Field(
        default=12.0,
        ge=4,
        le=48,
        description="Maximum interval between blowing (hours)"
    )

    # Intelligent triggers
    dp_trigger_ratio: float = Field(
        default=1.2,
        ge=1.05,
        le=1.5,
        description="DP ratio to trigger blowing"
    )
    effectiveness_trigger_pct: float = Field(
        default=3.0,
        ge=1,
        le=10,
        description="Effectiveness drop to trigger blowing (%)"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class AirPreheaterConfig(BaseModel):
    """
    Complete GL-024 Air Preheater Agent Configuration.

    This is the master configuration class combining all sub-configurations
    for the Air Preheater Optimization Agent. It defines operating parameters,
    thresholds, engineering constants, and regulatory requirements.

    Key Features:
        - Leakage detection and monitoring (ASME PTC 4.3 compliant)
        - Cold end corrosion prevention (acid dew point calculations)
        - Pressure drop trending and baseline comparison
        - Effectiveness tracking with load correction
        - Operating mode-specific thresholds
        - Provenance tracking for audit trail

    Standards Compliance:
        - ASME PTC 4.3: Air Heater Performance Test Code
        - API 560: Fired Heaters for General Refinery Service
        - NFPA 85: Boiler and Combustion Systems Hazards Code
        - NFPA 86: Standard for Ovens and Furnaces

    Example:
        >>> config = AirPreheaterConfig(
        ...     preheater_id="APH-001",
        ...     boiler_id="BLR-001",
        ...     fuel_type=FuelType.COAL_BITUMINOUS,
        ... )
        >>> agent = AirPreheaterAgent(config)
    """

    # =========================================================================
    # IDENTIFICATION
    # =========================================================================

    preheater_id: str = Field(
        ...,
        description="Unique air preheater identifier (e.g., APH-001)"
    )
    preheater_name: str = Field(
        default="",
        description="Human-readable preheater name"
    )
    boiler_id: str = Field(
        default="",
        description="Associated boiler identifier"
    )
    plant_id: str = Field(
        default="",
        description="Plant identifier"
    )

    # =========================================================================
    # OPERATING PARAMETERS
    # =========================================================================

    fuel_type: FuelType = Field(
        default=FuelType.COAL_BITUMINOUS,
        description="Primary fuel type for dew point and fouling calculations"
    )
    current_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )

    # =========================================================================
    # DESIGN CONFIGURATION
    # =========================================================================

    design: PreheaterDesignConfig = Field(
        default_factory=PreheaterDesignConfig,
        description="Air preheater design specifications"
    )
    soot_blower: SootBlowerConfig = Field(
        default_factory=SootBlowerConfig,
        description="Soot blower configuration"
    )

    # =========================================================================
    # THRESHOLDS
    # =========================================================================

    thresholds: AirPreheaterThresholds = Field(
        default_factory=AirPreheaterThresholds,
        description="Complete threshold configuration"
    )
    operating_modes: OperatingModeConfig = Field(
        default_factory=OperatingModeConfig,
        description="Operating mode-specific configurations"
    )

    # =========================================================================
    # ENGINEERING CONSTANTS
    # =========================================================================

    acid_dew_point_coefficients: AcidDewPointCoefficients = Field(
        default_factory=AcidDewPointCoefficients,
        description="Acid dew point calculation coefficients"
    )
    acid_dew_point_method: AcidDewPointMethod = Field(
        default=AcidDewPointMethod.VERHOFF_BANCHERO,
        description="Primary acid dew point calculation method"
    )
    ntu_ranges: NTURanges = Field(
        default_factory=NTURanges,
        description="Typical NTU ranges by preheater type"
    )
    heat_capacity_correlations: HeatCapacityCorrelations = Field(
        default_factory=HeatCapacityCorrelations,
        description="Heat capacity correlations for air and flue gas"
    )
    fouling_factors: FoulingFactors = Field(
        default_factory=FoulingFactors,
        description="Fouling factors by fuel type"
    )

    # =========================================================================
    # REGULATORY REFERENCES
    # =========================================================================

    asme_ptc_43: ASMEPTC43Tolerances = Field(
        default_factory=ASMEPTC43Tolerances,
        description="ASME PTC 4.3 test code tolerances"
    )
    api_560: API560Requirements = Field(
        default_factory=API560Requirements,
        description="API 560 design requirements"
    )
    nfpa_85: NFPA85Requirements = Field(
        default_factory=NFPA85Requirements,
        description="NFPA 85 safety requirements"
    )
    nfpa_86: NFPA86Requirements = Field(
        default_factory=NFPA86Requirements,
        description="NFPA 86 furnace safety requirements"
    )

    # =========================================================================
    # MONITORING SETTINGS
    # =========================================================================

    monitoring_interval_s: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Monitoring interval (seconds)"
    )
    data_collection_interval_s: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Data collection interval (seconds)"
    )
    historian_tag_prefix: str = Field(
        default="",
        description="Historian tag prefix for data storage"
    )

    # =========================================================================
    # SAFETY SETTINGS
    # =========================================================================

    high_gas_temp_alarm_f: float = Field(
        default=900.0,
        ge=700,
        le=1100,
        description="High gas inlet temperature alarm (F)"
    )
    high_gas_temp_trip_f: float = Field(
        default=1000.0,
        ge=800,
        le=1200,
        description="High gas inlet temperature trip (F)"
    )
    fire_detection_enabled: bool = Field(
        default=True,
        description="Enable fire detection monitoring"
    )
    low_rotation_alarm_enabled: bool = Field(
        default=True,
        description="Enable low rotation speed alarm (regenerative)"
    )

    # =========================================================================
    # AUDIT AND PROVENANCE
    # =========================================================================

    enable_audit: bool = Field(
        default=True,
        description="Enable comprehensive audit trail"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    # =========================================================================
    # AGENT SETTINGS
    # =========================================================================

    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    optimization_enabled: bool = Field(
        default=True,
        description="Enable automatic optimization recommendations"
    )
    ml_prediction_enabled: bool = Field(
        default=True,
        description="Enable ML-based predictions"
    )
    explainability_enabled: bool = Field(
        default=True,
        description="Enable decision explainability"
    )

    class Config:
        use_enum_values = True
        validate_assignment = True

    @validator("preheater_name", always=True)
    def set_default_name(cls, v, values):
        """Set default name from preheater_id."""
        if not v and "preheater_id" in values:
            return f"Air Preheater {values['preheater_id']}"
        return v


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_default_config(
    preheater_id: str,
    boiler_id: str = "",
    preheater_type: PreheaterType = PreheaterType.LJUNGSTROM,
    fuel_type: FuelType = FuelType.COAL_BITUMINOUS,
) -> AirPreheaterConfig:
    """
    Create a default GL-024 configuration.

    Args:
        preheater_id: Unique preheater identifier
        boiler_id: Associated boiler identifier
        preheater_type: Type of air preheater
        fuel_type: Primary fuel type

    Returns:
        AirPreheaterConfig with sensible defaults
    """
    config = AirPreheaterConfig(
        preheater_id=preheater_id,
        boiler_id=boiler_id,
        fuel_type=fuel_type,
    )
    config.design.preheater_type = preheater_type
    return config


def create_coal_fired_config(
    preheater_id: str,
    boiler_id: str = "",
    coal_type: FuelType = FuelType.COAL_BITUMINOUS,
) -> AirPreheaterConfig:
    """
    Create a configuration optimized for coal-fired applications.

    Coal firing has higher fouling, higher acid dew point, and
    requires more aggressive cold end protection.

    Args:
        preheater_id: Unique preheater identifier
        boiler_id: Associated boiler identifier
        coal_type: Type of coal

    Returns:
        AirPreheaterConfig optimized for coal firing
    """
    config = AirPreheaterConfig(
        preheater_id=preheater_id,
        boiler_id=boiler_id,
        fuel_type=coal_type,
    )

    # Tighter cold end margins for coal
    config.thresholds.cold_end_margin.margin_warning_f = 25.0
    config.thresholds.cold_end_margin.margin_alarm_f = 15.0
    config.thresholds.cold_end_margin.min_cold_end_temp_coal_f = 290.0

    # More aggressive soot blowing
    config.soot_blower.max_interval_hours = 8.0
    config.soot_blower.dp_trigger_ratio = 1.15

    # Enhanced monitoring
    config.monitoring_interval_s = 30

    return config


def create_gas_fired_config(
    preheater_id: str,
    boiler_id: str = "",
) -> AirPreheaterConfig:
    """
    Create a configuration optimized for natural gas firing.

    Gas firing has minimal fouling and lower acid dew point,
    allowing relaxed thresholds and longer cleaning intervals.

    Args:
        preheater_id: Unique preheater identifier
        boiler_id: Associated boiler identifier

    Returns:
        AirPreheaterConfig optimized for gas firing
    """
    config = AirPreheaterConfig(
        preheater_id=preheater_id,
        boiler_id=boiler_id,
        fuel_type=FuelType.NATURAL_GAS,
    )

    # Relaxed cold end margins for gas
    config.thresholds.cold_end_margin.margin_warning_f = 15.0
    config.thresholds.cold_end_margin.margin_alarm_f = 8.0
    config.thresholds.cold_end_margin.min_cold_end_temp_gas_f = 180.0

    # Longer soot blowing intervals (minimal fouling)
    config.soot_blower.min_interval_hours = 4.0
    config.soot_blower.max_interval_hours = 24.0

    # Standard monitoring (gas is cleaner)
    config.monitoring_interval_s = 60

    return config


def create_high_sulfur_config(
    preheater_id: str,
    boiler_id: str = "",
    fuel_type: FuelType = FuelType.NO6_FUEL_OIL,
) -> AirPreheaterConfig:
    """
    Create a configuration for high-sulfur fuel applications.

    High sulfur content raises acid dew point significantly,
    requiring enhanced cold end protection and monitoring.

    Args:
        preheater_id: Unique preheater identifier
        boiler_id: Associated boiler identifier
        fuel_type: High-sulfur fuel type

    Returns:
        AirPreheaterConfig optimized for high-sulfur fuels
    """
    config = AirPreheaterConfig(
        preheater_id=preheater_id,
        boiler_id=boiler_id,
        fuel_type=fuel_type,
    )

    # Much tighter cold end margins
    config.thresholds.cold_end_margin.margin_warning_f = 30.0
    config.thresholds.cold_end_margin.margin_alarm_f = 20.0
    config.thresholds.cold_end_margin.margin_critical_f = 10.0
    config.thresholds.cold_end_margin.min_cold_end_temp_oil_f = 300.0

    # Use Okkes method for European high-sulfur applications
    config.acid_dew_point_method = AcidDewPointMethod.OKKES

    # Enhanced corrosion monitoring
    config.thresholds.cold_end_margin.corrosion_rate_warning_mpy = 3.0
    config.thresholds.cold_end_margin.cold_end_temp_scan_interval_s = 30

    # More frequent monitoring
    config.monitoring_interval_s = 30
    config.data_collection_interval_s = 2

    return config


def create_test_config(preheater_id: str = "APH-TEST-001") -> AirPreheaterConfig:
    """
    Create a test configuration for unit testing.

    Args:
        preheater_id: Test preheater identifier

    Returns:
        AirPreheaterConfig for testing
    """
    return AirPreheaterConfig(
        preheater_id=preheater_id,
        boiler_id="BLR-TEST-001",
        fuel_type=FuelType.NATURAL_GAS,
        enable_audit=False,
        enable_provenance=False,
        ml_prediction_enabled=False,
    )
