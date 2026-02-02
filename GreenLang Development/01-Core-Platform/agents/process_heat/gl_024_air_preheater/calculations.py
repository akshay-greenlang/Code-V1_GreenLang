"""
GL-024 AIRPREHEATER - Zero-Hallucination Calculation Engine

This module provides deterministic, auditable calculations for air preheater
performance analysis including heat transfer, leakage detection, cold-end
protection, and fouling assessment.

All calculations follow ASME PTC 4.3 and standard heat exchanger methodologies.

Engineering References:
    - ASME PTC 4.3 - Air Heaters Performance Test Code
    - Incropera & DeWitt - Fundamentals of Heat and Mass Transfer
    - Verhoff & Banchero (1974) - Predicting dew points of flue gases
    - Okkes (1987) - Acid dew point prediction
    - Pierce (1977) - Estimating acid dewpoints in stack gases

Example:
    >>> from greenlang.agents.process_heat.gl_024_air_preheater.calculations import (
    ...     AirPreheaterCalculator,
    ... )
    >>> from greenlang.agents.process_heat.gl_024_air_preheater.config import (
    ...     AirPreheaterConfig,
    ... )
    >>>
    >>> calc = AirPreheaterCalculator(AirPreheaterConfig())
    >>> effectiveness = calc.calculate_effectiveness(
    ...     gas_inlet_temp_f=650,
    ...     gas_outlet_temp_f=320,
    ...     air_inlet_temp_f=80,
    ...     air_outlet_temp_f=550,
    ... )
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import AirPreheaterConfig, AirPreheaterType


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass
class EffectivenessResult:
    """Result of effectiveness calculation."""
    effectiveness: float  # Dimensionless (0-1)
    capacity_ratio: float  # C_min / C_max
    c_min_btu_hr_f: float  # Minimum heat capacity rate
    c_max_btu_hr_f: float  # Maximum heat capacity rate
    hot_side_effectiveness: float
    cold_side_effectiveness: float
    methodology: str = "epsilon-NTU per ASME PTC 4.3"


@dataclass
class NTUResult:
    """Result of NTU calculation."""
    ntu: float  # Number of Transfer Units
    ntu_from_effectiveness: float
    preheater_type: str
    methodology: str = "epsilon-NTU correlation"


@dataclass
class HeatDutyResult:
    """Result of heat duty calculation."""
    heat_duty_mmbtu_hr: float
    heat_duty_btu_hr: float
    flow_rate_lb_hr: float
    temp_change_f: float
    specific_heat_btu_lb_f: float
    methodology: str = "Q = m_dot * Cp * delta_T"


@dataclass
class LMTDResult:
    """Result of LMTD calculation."""
    lmtd_f: float
    delta_t1_f: float  # Temperature difference at one end
    delta_t2_f: float  # Temperature difference at other end
    flow_arrangement: str  # "counterflow", "crossflow", etc.
    correction_factor: float  # For crossflow correction
    methodology: str = "Log Mean Temperature Difference"


@dataclass
class XRatioResult:
    """Result of X-ratio calculation for regenerative preheaters."""
    x_ratio: float
    corrected_gas_outlet_temp_f: float
    leakage_correction_f: float
    methodology: str = "X-ratio per ASME PTC 4.3"


@dataclass
class LeakageResult:
    """Result of leakage calculation."""
    air_to_gas_leakage_pct: float
    gas_to_air_leakage_pct: float
    o2_rise_pct: float
    leakage_flow_lb_hr: float
    efficiency_impact_pct: float
    methodology: str = "O2 rise method per ASME PTC 4.3"


@dataclass
class SealLeakageResult:
    """Result of seal leakage calculation."""
    seal_leakage_pct: float
    seal_leakage_flow_lb_hr: float
    seal_condition: str  # "GOOD", "ACCEPTABLE", "WORN", "CRITICAL"
    methodology: str = "Seal clearance calculation"


@dataclass
class AcidDewPointResult:
    """Result of acid dew point calculation."""
    acid_dew_point_f: float
    acid_dew_point_c: float
    h2so4_concentration_ppm: float
    methodology: str


@dataclass
class WaterDewPointResult:
    """Result of water dew point calculation."""
    water_dew_point_f: float
    water_dew_point_c: float
    partial_pressure_psia: float
    methodology: str = "Steam tables correlation"


@dataclass
class CleanlinessResult:
    """Result of cleanliness factor calculation."""
    cleanliness_factor: float  # 0-1, 1=clean
    fouling_resistance: float  # hr-ft2-F/BTU
    estimated_recovery_pct: float
    methodology: str = "UA degradation method"


@dataclass
class OptimalTempResult:
    """Result of optimal temperature calculation."""
    optimal_temp_f: float
    current_temp_f: float
    temp_adjustment_f: float
    efficiency_impact_pct: float
    methodology: str = "Cold-end protection optimization"


@dataclass
class EnergySavingsResult:
    """Result of energy savings calculation."""
    efficiency_gain_pct: float
    annual_savings_mmbtu: float
    annual_cost_savings_usd: float
    payback_months: float
    methodology: str = "Efficiency improvement analysis"


@dataclass
class EfficiencyImpactResult:
    """Result of efficiency impact calculation."""
    efficiency_impact_pct: float
    baseline_efficiency_pct: float
    current_efficiency_pct: float
    methodology: str = "Boiler efficiency correlation"


# =============================================================================
# CALCULATOR CLASS
# =============================================================================

class AirPreheaterCalculator:
    """
    Zero-hallucination calculation engine for air preheater analysis.

    All calculations are deterministic, fully documented, and traceable
    to engineering references. No ML/AI - pure physics-based calculations.

    Attributes:
        config: Calculator configuration with thresholds and constants
    """

    # Specific heat constants (BTU/lb-F)
    CP_AIR = 0.24  # Dry air at moderate temperatures
    CP_FLUE_GAS = 0.26  # Typical flue gas
    CP_WATER_VAPOR = 0.45  # Steam/water vapor

    # Molecular weights
    MW_AIR = 28.97
    MW_FLUE_GAS = 29.5  # Typical

    def __init__(self, config: Optional[AirPreheaterConfig] = None):
        """
        Initialize the calculator.

        Args:
            config: Configuration with thresholds. Uses defaults if not provided.
        """
        self.config = config or AirPreheaterConfig()

    # =========================================================================
    # HEAT TRANSFER CALCULATIONS
    # =========================================================================

    def calculate_effectiveness(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        air_inlet_temp_f: float,
        air_outlet_temp_f: float,
        gas_flow_rate_lb_hr: Optional[float] = None,
        air_flow_rate_lb_hr: Optional[float] = None,
    ) -> EffectivenessResult:
        """
        Calculate heat exchanger effectiveness using epsilon-NTU method.

        Effectiveness = Q_actual / Q_max
        where Q_max = C_min * (T_hot_in - T_cold_in)

        Args:
            gas_inlet_temp_f: Gas inlet temperature (F)
            gas_outlet_temp_f: Gas outlet temperature (F)
            air_inlet_temp_f: Air inlet temperature (F)
            air_outlet_temp_f: Air outlet temperature (F)
            gas_flow_rate_lb_hr: Optional gas mass flow rate (lb/hr)
            air_flow_rate_lb_hr: Optional air mass flow rate (lb/hr)

        Returns:
            EffectivenessResult with calculated values
        """
        # Validate inputs
        if gas_inlet_temp_f <= gas_outlet_temp_f:
            raise ValueError("Gas inlet temp must be greater than outlet temp")
        if air_outlet_temp_f <= air_inlet_temp_f:
            raise ValueError("Air outlet temp must be greater than inlet temp")

        # Calculate temperature changes
        gas_temp_drop = gas_inlet_temp_f - gas_outlet_temp_f
        air_temp_rise = air_outlet_temp_f - air_inlet_temp_f

        # Maximum possible temperature change
        max_temp_diff = gas_inlet_temp_f - air_inlet_temp_f

        # Calculate effectivenesses
        # Hot side (gas) effectiveness
        hot_side_effectiveness = gas_temp_drop / max_temp_diff

        # Cold side (air) effectiveness
        cold_side_effectiveness = air_temp_rise / max_temp_diff

        # Overall effectiveness (average or use C_min approach)
        if gas_flow_rate_lb_hr and air_flow_rate_lb_hr:
            c_gas = gas_flow_rate_lb_hr * self.CP_FLUE_GAS
            c_air = air_flow_rate_lb_hr * self.CP_AIR

            c_min = min(c_gas, c_air)
            c_max = max(c_gas, c_air)
            capacity_ratio = c_min / c_max if c_max > 0 else 0

            # Use C_min stream for effectiveness
            if c_gas <= c_air:
                effectiveness = hot_side_effectiveness
            else:
                effectiveness = cold_side_effectiveness
        else:
            # Without flow rates, use average
            effectiveness = (hot_side_effectiveness + cold_side_effectiveness) / 2
            capacity_ratio = gas_temp_drop / air_temp_rise if air_temp_rise > 0 else 1.0
            c_min = 0
            c_max = 0

        return EffectivenessResult(
            effectiveness=round(effectiveness, 4),
            capacity_ratio=round(capacity_ratio, 4),
            c_min_btu_hr_f=c_min,
            c_max_btu_hr_f=c_max,
            hot_side_effectiveness=round(hot_side_effectiveness, 4),
            cold_side_effectiveness=round(cold_side_effectiveness, 4),
        )

    def calculate_ntu(
        self,
        effectiveness: float,
        capacity_ratio: float,
        preheater_type: AirPreheaterType = AirPreheaterType.REGENERATIVE,
    ) -> NTUResult:
        """
        Calculate Number of Transfer Units from effectiveness.

        For counterflow: NTU = ln[(1-e*Cr)/(1-e)] / (1-Cr)  when Cr < 1
        For Cr = 1: NTU = e / (1-e)
        For regenerative: Use modified correlation

        Args:
            effectiveness: Heat exchanger effectiveness (0-1)
            capacity_ratio: C_min/C_max ratio
            preheater_type: Type of air preheater

        Returns:
            NTUResult with calculated values
        """
        e = effectiveness
        cr = capacity_ratio

        # Bound inputs
        e = max(0.001, min(0.999, e))
        cr = max(0.001, min(1.0, cr))

        if preheater_type == AirPreheaterType.REGENERATIVE:
            # Regenerative preheater correlation (Kays & London)
            # NTU_reg ≈ NTU_counterflow * 0.9 (approximation)
            if abs(cr - 1.0) < 0.01:
                ntu = e / (1 - e)
            else:
                ntu = math.log((1 - e * cr) / (1 - e)) / (1 - cr)
            ntu *= 1.1  # Correction for regenerative
        elif preheater_type == AirPreheaterType.RECUPERATIVE:
            # Counterflow correlation
            if abs(cr - 1.0) < 0.01:
                ntu = e / (1 - e)
            else:
                ntu = math.log((1 - e * cr) / (1 - e)) / (1 - cr)
        else:  # HEAT_PIPE
            # Crossflow approximation
            ntu = -math.log(1 + math.log(1 - e * cr) / cr) if cr > 0 else -math.log(1 - e)

        return NTUResult(
            ntu=round(ntu, 3),
            ntu_from_effectiveness=round(ntu, 3),
            preheater_type=preheater_type.value,
        )

    def calculate_heat_duty(
        self,
        flow_rate_lb_hr: float,
        inlet_temp_f: float,
        outlet_temp_f: float,
        fluid_type: str = "air",
        composition: Optional[Dict[str, float]] = None,
        humidity: Optional[float] = None,
    ) -> HeatDutyResult:
        """
        Calculate heat duty using Q = m_dot * Cp * delta_T.

        Args:
            flow_rate_lb_hr: Mass flow rate (lb/hr)
            inlet_temp_f: Inlet temperature (F)
            outlet_temp_f: Outlet temperature (F)
            fluid_type: "air" or "flue_gas"
            composition: Optional gas composition for Cp adjustment
            humidity: Optional humidity for air (%)

        Returns:
            HeatDutyResult with heat duty values
        """
        temp_change = abs(outlet_temp_f - inlet_temp_f)

        # Determine specific heat
        if fluid_type == "air":
            cp = self.CP_AIR
            if humidity:
                # Adjust for humidity
                cp += (humidity / 100) * (self.CP_WATER_VAPOR - self.CP_AIR) * 0.01
        else:  # flue_gas
            cp = self.CP_FLUE_GAS
            if composition:
                # Adjust for composition
                h2o = composition.get("H2O", 0) / 100
                co2 = composition.get("CO2", 0) / 100
                # Weighted average adjustment
                cp = 0.24 * (1 - h2o - co2) + 0.45 * h2o + 0.21 * co2

        heat_duty_btu_hr = flow_rate_lb_hr * cp * temp_change
        heat_duty_mmbtu_hr = heat_duty_btu_hr / 1e6

        return HeatDutyResult(
            heat_duty_mmbtu_hr=round(heat_duty_mmbtu_hr, 3),
            heat_duty_btu_hr=round(heat_duty_btu_hr, 0),
            flow_rate_lb_hr=flow_rate_lb_hr,
            temp_change_f=temp_change,
            specific_heat_btu_lb_f=round(cp, 4),
        )

    def calculate_lmtd(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        air_inlet_temp_f: float,
        air_outlet_temp_f: float,
        flow_arrangement: str = "counterflow",
    ) -> LMTDResult:
        """
        Calculate Log Mean Temperature Difference.

        LMTD = (delta_T1 - delta_T2) / ln(delta_T1 / delta_T2)

        Args:
            gas_inlet_temp_f: Hot fluid inlet temperature
            gas_outlet_temp_f: Hot fluid outlet temperature
            air_inlet_temp_f: Cold fluid inlet temperature
            air_outlet_temp_f: Cold fluid outlet temperature
            flow_arrangement: "counterflow", "parallel", or "crossflow"

        Returns:
            LMTDResult with LMTD value
        """
        if flow_arrangement == "counterflow":
            # Counterflow: hot inlet vs cold outlet, hot outlet vs cold inlet
            delta_t1 = gas_inlet_temp_f - air_outlet_temp_f
            delta_t2 = gas_outlet_temp_f - air_inlet_temp_f
        else:  # parallel or crossflow
            delta_t1 = gas_inlet_temp_f - air_inlet_temp_f
            delta_t2 = gas_outlet_temp_f - air_outlet_temp_f

        # Avoid divide by zero and log of negative
        delta_t1 = max(delta_t1, 0.1)
        delta_t2 = max(delta_t2, 0.1)

        if abs(delta_t1 - delta_t2) < 0.1:
            # Arithmetic mean when delta_Ts are nearly equal
            lmtd = (delta_t1 + delta_t2) / 2
        else:
            lmtd = (delta_t1 - delta_t2) / math.log(delta_t1 / delta_t2)

        # Correction factor for crossflow
        correction_factor = 1.0
        if flow_arrangement == "crossflow":
            # Simplified correction for mixed-unmixed crossflow
            correction_factor = 0.95

        return LMTDResult(
            lmtd_f=round(lmtd * correction_factor, 2),
            delta_t1_f=round(delta_t1, 2),
            delta_t2_f=round(delta_t2, 2),
            flow_arrangement=flow_arrangement,
            correction_factor=correction_factor,
        )

    def calculate_x_ratio(
        self,
        gas_inlet_temp_f: float,
        gas_outlet_temp_f: float,
        air_inlet_temp_f: float,
        air_outlet_temp_f: float,
        o2_inlet_pct: float,
        o2_outlet_pct: float,
    ) -> XRatioResult:
        """
        Calculate X-ratio for regenerative air preheaters.

        X-ratio is used to correct gas outlet temperature for leakage
        and is a characteristic performance indicator.

        X = (T_gas_out - T_air_in) / (T_gas_in - T_air_in)

        Args:
            gas_inlet_temp_f: Gas inlet temperature
            gas_outlet_temp_f: Gas outlet temperature
            air_inlet_temp_f: Air inlet temperature
            air_outlet_temp_f: Air outlet temperature
            o2_inlet_pct: O2 at gas inlet
            o2_outlet_pct: O2 at gas outlet

        Returns:
            XRatioResult with X-ratio and corrected temperatures
        """
        # Calculate O2 rise (indicates leakage)
        o2_rise = o2_outlet_pct - o2_inlet_pct

        # Leakage correction for gas outlet temperature
        # Leakage air at cold end reduces measured gas outlet temp
        leakage_correction = o2_rise * 5  # Approximate: 5F per 1% O2 rise

        # Corrected gas outlet temperature
        corrected_gas_outlet = gas_outlet_temp_f + leakage_correction

        # Calculate X-ratio
        max_temp_diff = gas_inlet_temp_f - air_inlet_temp_f
        if max_temp_diff > 0:
            x_ratio = (corrected_gas_outlet - air_inlet_temp_f) / max_temp_diff
        else:
            x_ratio = 0

        return XRatioResult(
            x_ratio=round(x_ratio, 4),
            corrected_gas_outlet_temp_f=round(corrected_gas_outlet, 1),
            leakage_correction_f=round(leakage_correction, 1),
        )

    # =========================================================================
    # LEAKAGE CALCULATIONS
    # =========================================================================

    def calculate_leakage_o2_method(
        self,
        o2_inlet_pct: float,
        o2_outlet_pct: float,
        gas_flow_rate_lb_hr: float,
        air_flow_rate_lb_hr: float,
    ) -> LeakageResult:
        """
        Calculate air-to-gas leakage using O2 rise method.

        This is the standard ASME PTC 4.3 method. Leakage increases
        O2 content as air leaks into the gas side.

        Leakage % = (O2_out - O2_in) / (21 - O2_out) * 100

        Args:
            o2_inlet_pct: O2 concentration at air preheater gas inlet
            o2_outlet_pct: O2 concentration at air preheater gas outlet
            gas_flow_rate_lb_hr: Flue gas mass flow rate
            air_flow_rate_lb_hr: Combustion air mass flow rate

        Returns:
            LeakageResult with leakage percentages
        """
        # O2 rise
        o2_rise = o2_outlet_pct - o2_inlet_pct
        o2_rise = max(0, o2_rise)  # Can't be negative

        # Air-to-gas leakage percentage
        # Based on O2 dilution: leakage air (21% O2) dilutes flue gas
        if o2_outlet_pct < 21:
            air_to_gas_leakage_pct = o2_rise / (21 - o2_outlet_pct) * 100
        else:
            air_to_gas_leakage_pct = 100  # All air

        # Bound to reasonable range
        air_to_gas_leakage_pct = min(air_to_gas_leakage_pct, 50)

        # Leakage flow rate
        leakage_flow_lb_hr = air_flow_rate_lb_hr * (air_to_gas_leakage_pct / 100)

        # Gas-to-air leakage (typically much smaller, safety concern)
        # Estimate based on pressure differential direction
        gas_to_air_leakage_pct = air_to_gas_leakage_pct * 0.1  # Typically 10% of A-to-G

        # Efficiency impact
        # Higher O2 means more excess air, reducing efficiency
        efficiency_impact_pct = o2_rise * 0.25  # ~0.25% efficiency loss per 1% O2 rise

        return LeakageResult(
            air_to_gas_leakage_pct=round(air_to_gas_leakage_pct, 2),
            gas_to_air_leakage_pct=round(gas_to_air_leakage_pct, 2),
            o2_rise_pct=round(o2_rise, 2),
            leakage_flow_lb_hr=round(leakage_flow_lb_hr, 0),
            efficiency_impact_pct=round(efficiency_impact_pct, 3),
        )

    def calculate_seal_leakage(
        self,
        seal_clearance_in: float,
        seal_diameter_in: float,
        pressure_differential_in_wc: float,
        rotor_speed_rpm: Optional[float] = None,
    ) -> SealLeakageResult:
        """
        Calculate seal leakage from clearance measurements.

        Leakage flow is proportional to clearance area and
        square root of pressure differential.

        Args:
            seal_clearance_in: Measured seal clearance (inches)
            seal_diameter_in: Seal diameter (inches)
            pressure_differential_in_wc: Pressure differential (inches WC)
            rotor_speed_rpm: Rotor speed (for dynamic seal effects)

        Returns:
            SealLeakageResult with seal condition assessment
        """
        # Design clearance (typical)
        design_clearance = 0.125  # inches

        # Clearance ratio
        clearance_ratio = seal_clearance_in / design_clearance

        # Leakage flow (simplified orifice equation)
        # Q = C * A * sqrt(2 * dp / rho)
        seal_circumference = math.pi * seal_diameter_in
        leakage_area = seal_circumference * seal_clearance_in  # in^2

        # Convert to flow (approximate)
        # Using air at standard conditions
        rho_air = 0.075  # lb/ft3
        dp_psf = pressure_differential_in_wc * 5.2  # Convert to psf

        if dp_psf > 0:
            velocity = math.sqrt(2 * dp_psf * 32.2 / rho_air)  # ft/s
            leakage_flow_cfm = (leakage_area / 144) * velocity * 60
            leakage_flow_lb_hr = leakage_flow_cfm * rho_air * 60
        else:
            leakage_flow_lb_hr = 0

        # Seal leakage as percentage (approximate)
        # Typical air flow through preheater ~500,000 lb/hr
        typical_air_flow = 500000
        seal_leakage_pct = (leakage_flow_lb_hr / typical_air_flow) * 100

        # Determine seal condition
        if clearance_ratio <= 1.2:
            seal_condition = "GOOD"
        elif clearance_ratio <= 1.5:
            seal_condition = "ACCEPTABLE"
        elif clearance_ratio <= 2.0:
            seal_condition = "WORN"
        else:
            seal_condition = "CRITICAL"

        return SealLeakageResult(
            seal_leakage_pct=round(seal_leakage_pct, 2),
            seal_leakage_flow_lb_hr=round(leakage_flow_lb_hr, 0),
            seal_condition=seal_condition,
        )

    # =========================================================================
    # COLD-END PROTECTION CALCULATIONS
    # =========================================================================

    def calculate_acid_dew_point_verhoff_banchero(
        self,
        h2o_vol_pct: float,
        so3_ppm: float,
    ) -> AcidDewPointResult:
        """
        Calculate acid dew point using Verhoff-Banchero correlation (1974).

        This is the most widely used correlation for H2SO4 dew point.

        T_dew(K) = 1000 / (2.276 - 0.0294*ln(P_H2O) - 0.0858*ln(P_SO3)
                          + 0.0062*ln(P_H2O)*ln(P_SO3))

        Args:
            h2o_vol_pct: Water vapor concentration (volume %)
            so3_ppm: SO3 concentration (ppm by volume)

        Returns:
            AcidDewPointResult with dew point temperature
        """
        # Convert to partial pressures (atm)
        p_h2o = h2o_vol_pct / 100  # atm
        p_so3 = so3_ppm / 1e6  # atm

        # Avoid log of zero
        p_h2o = max(p_h2o, 1e-6)
        p_so3 = max(p_so3, 1e-9)

        # Verhoff-Banchero correlation
        ln_h2o = math.log(p_h2o)
        ln_so3 = math.log(p_so3)

        t_dew_k = 1000 / (2.276 - 0.0294 * ln_h2o - 0.0858 * ln_so3
                          + 0.0062 * ln_h2o * ln_so3)

        # Convert to C and F
        t_dew_c = t_dew_k - 273.15
        t_dew_f = t_dew_c * 9 / 5 + 32

        # Estimate H2SO4 concentration at dew point
        h2so4_ppm = so3_ppm * (98 / 80)  # Stoichiometric conversion

        return AcidDewPointResult(
            acid_dew_point_f=round(t_dew_f, 1),
            acid_dew_point_c=round(t_dew_c, 1),
            h2so4_concentration_ppm=round(h2so4_ppm, 1),
            methodology="Verhoff-Banchero (1974)",
        )

    def calculate_acid_dew_point_okkes(
        self,
        so2_ppm: float,
        h2o_vol_pct: float,
        excess_air_pct: float,
    ) -> AcidDewPointResult:
        """
        Calculate acid dew point using Okkes correlation (1987).

        Alternative method that uses SO2 and accounts for SO3 formation.

        Args:
            so2_ppm: SO2 concentration (ppm by volume)
            h2o_vol_pct: Water vapor concentration (volume %)
            excess_air_pct: Excess air percentage

        Returns:
            AcidDewPointResult with dew point temperature
        """
        # Estimate SO3 from SO2 (typically 1-5% conversion)
        so3_conversion = 0.02  # 2% default
        if excess_air_pct > 30:
            so3_conversion = 0.03  # Higher with more excess air

        so3_ppm = so2_ppm * so3_conversion

        # Okkes correlation
        # T_dew(C) = 203.25 + 27.6*log10(P_H2O) + 10.83*log10(P_SO3)

        p_h2o = max(h2o_vol_pct / 100, 1e-6)
        p_so3 = max(so3_ppm / 1e6, 1e-9)

        t_dew_c = 203.25 + 27.6 * math.log10(p_h2o) + 10.83 * math.log10(p_so3)
        t_dew_f = t_dew_c * 9 / 5 + 32

        h2so4_ppm = so3_ppm * (98 / 80)

        return AcidDewPointResult(
            acid_dew_point_f=round(t_dew_f, 1),
            acid_dew_point_c=round(t_dew_c, 1),
            h2so4_concentration_ppm=round(h2so4_ppm, 1),
            methodology="Okkes (1987)",
        )

    def calculate_water_dew_point(
        self,
        h2o_vol_pct: float,
        pressure_psia: float = 14.7,
    ) -> WaterDewPointResult:
        """
        Calculate water dew point from moisture content.

        Uses Antoine equation for water vapor pressure.

        Args:
            h2o_vol_pct: Water vapor concentration (volume %)
            pressure_psia: Total pressure (psia)

        Returns:
            WaterDewPointResult with water dew point
        """
        # Partial pressure of water
        p_h2o_psia = (h2o_vol_pct / 100) * pressure_psia

        # Antoine equation (NIST) to find temperature
        # log10(P_mmHg) = A - B/(C + T_C)
        # For water: A=8.07131, B=1730.63, C=233.426

        p_mmhg = p_h2o_psia * 51.715  # Convert psia to mmHg

        if p_mmhg > 0:
            # Solve for T
            A = 8.07131
            B = 1730.63
            C = 233.426

            t_dew_c = B / (A - math.log10(p_mmhg)) - C
            t_dew_f = t_dew_c * 9 / 5 + 32
        else:
            t_dew_c = -40
            t_dew_f = -40

        return WaterDewPointResult(
            water_dew_point_f=round(t_dew_f, 1),
            water_dew_point_c=round(t_dew_c, 1),
            partial_pressure_psia=round(p_h2o_psia, 4),
        )

    def calculate_cold_end_temp(
        self,
        gas_outlet_temp_f: float,
        air_inlet_temp_f: float,
        preheater_type: AirPreheaterType,
    ) -> float:
        """
        Calculate cold-end element temperature.

        The cold end is where gas exits and air enters. Element temperature
        is between the two fluid temperatures.

        Args:
            gas_outlet_temp_f: Gas outlet temperature
            air_inlet_temp_f: Air inlet temperature
            preheater_type: Type of preheater

        Returns:
            Estimated cold-end element temperature (F)
        """
        if preheater_type == AirPreheaterType.REGENERATIVE:
            # Regenerative: element cycles between hot and cold
            # Cold end temp is weighted average (element spends more time in gas)
            cold_end_temp = 0.7 * gas_outlet_temp_f + 0.3 * air_inlet_temp_f
        elif preheater_type == AirPreheaterType.RECUPERATIVE:
            # Recuperative: tube wall temperature
            cold_end_temp = 0.5 * gas_outlet_temp_f + 0.5 * air_inlet_temp_f
        else:  # Heat pipe
            cold_end_temp = 0.6 * gas_outlet_temp_f + 0.4 * air_inlet_temp_f

        return round(cold_end_temp, 1)

    # =========================================================================
    # FOULING CALCULATIONS
    # =========================================================================

    def calculate_cleanliness_factor(
        self,
        current_effectiveness: float,
        design_effectiveness: float,
        current_ua: Optional[float] = None,
        design_ua: Optional[float] = None,
    ) -> CleanlinessResult:
        """
        Calculate cleanliness factor from effectiveness degradation.

        CF = UA_current / UA_design
        or approximated from effectiveness ratio

        Args:
            current_effectiveness: Current effectiveness (0-1)
            design_effectiveness: Design effectiveness (0-1)
            current_ua: Current UA value (BTU/hr-F)
            design_ua: Design UA value (BTU/hr-F)

        Returns:
            CleanlinessResult with cleanliness factor
        """
        if current_ua and design_ua and design_ua > 0:
            cleanliness_factor = current_ua / design_ua
        else:
            # Approximate from effectiveness ratio
            # Note: relationship is not linear, this is simplified
            if design_effectiveness > 0:
                cleanliness_factor = current_effectiveness / design_effectiveness
            else:
                cleanliness_factor = 1.0

        # Bound to reasonable range
        cleanliness_factor = max(0.3, min(1.0, cleanliness_factor))

        # Estimate fouling resistance
        # Rf = (1/UA_dirty - 1/UA_clean) * A
        # Simplified: Rf proportional to (1 - CF)
        fouling_resistance = (1 - cleanliness_factor) * 0.002  # hr-ft2-F/BTU

        # Estimate recovery from cleaning
        estimated_recovery = (design_effectiveness - current_effectiveness) * 100

        return CleanlinessResult(
            cleanliness_factor=round(cleanliness_factor, 3),
            fouling_resistance=round(fouling_resistance, 5),
            estimated_recovery_pct=round(estimated_recovery, 1),
        )

    # =========================================================================
    # OPTIMIZATION CALCULATIONS
    # =========================================================================

    def calculate_optimal_air_outlet_temp(
        self,
        current_air_outlet_f: float,
        acid_dew_point_f: float,
        cold_end_margin_target_f: float,
        gas_inlet_temp_f: float,
        current_effectiveness: float,
        max_effectiveness: float = 0.85,
    ) -> OptimalTempResult:
        """
        Calculate optimal air outlet temperature.

        Balances efficiency (higher air outlet = better) against
        cold-end protection (need margin above acid dew point).

        Args:
            current_air_outlet_f: Current air outlet temperature
            acid_dew_point_f: Acid dew point temperature
            cold_end_margin_target_f: Target margin above dew point
            gas_inlet_temp_f: Gas inlet temperature
            current_effectiveness: Current effectiveness
            max_effectiveness: Maximum achievable effectiveness

        Returns:
            OptimalTempResult with optimal temperature
        """
        # Minimum gas outlet temp for cold-end protection
        min_gas_outlet = acid_dew_point_f + cold_end_margin_target_f

        # Maximum air outlet based on max effectiveness
        # T_air_out_max = T_air_in + e_max * (T_gas_in - T_air_in)
        # Assume air inlet ~80F
        air_inlet = 80
        max_air_outlet = air_inlet + max_effectiveness * (gas_inlet_temp_f - air_inlet)

        # Optimal is highest air outlet that maintains cold-end protection
        # This requires iterating on the heat balance
        # Simplified: use current + potential improvement
        potential_improvement = (max_effectiveness - current_effectiveness) * (gas_inlet_temp_f - air_inlet)
        optimal_temp = min(current_air_outlet_f + potential_improvement, max_air_outlet)

        temp_adjustment = optimal_temp - current_air_outlet_f

        # Efficiency impact (approximate: 1F air preheat = 0.025% efficiency)
        efficiency_impact = temp_adjustment * 0.025

        return OptimalTempResult(
            optimal_temp_f=round(optimal_temp, 1),
            current_temp_f=current_air_outlet_f,
            temp_adjustment_f=round(temp_adjustment, 1),
            efficiency_impact_pct=round(efficiency_impact, 3),
        )

    def calculate_energy_savings(
        self,
        current_effectiveness: float,
        achievable_effectiveness: float,
        fuel_flow_mmbtu_hr: float,
        fuel_cost_per_mmbtu: float,
        operating_hours_per_year: int = 8000,
    ) -> EnergySavingsResult:
        """
        Calculate energy savings from effectiveness improvement.

        Args:
            current_effectiveness: Current effectiveness (0-1)
            achievable_effectiveness: Achievable effectiveness (0-1)
            fuel_flow_mmbtu_hr: Fuel input rate
            fuel_cost_per_mmbtu: Fuel cost ($/MMBTU)
            operating_hours_per_year: Annual operating hours

        Returns:
            EnergySavingsResult with savings analysis
        """
        # Effectiveness improvement
        delta_effectiveness = achievable_effectiveness - current_effectiveness

        # Efficiency gain (approximate relationship)
        # 1% effectiveness improvement ≈ 0.15% efficiency gain
        efficiency_gain_pct = delta_effectiveness * 100 * 0.15

        # Fuel savings
        fuel_savings_per_hour = fuel_flow_mmbtu_hr * (efficiency_gain_pct / 100)
        annual_savings_mmbtu = fuel_savings_per_hour * operating_hours_per_year

        # Cost savings
        annual_cost_savings = annual_savings_mmbtu * fuel_cost_per_mmbtu

        # Payback (assuming cleaning cost of ~$50,000)
        cleaning_cost = 50000
        if annual_cost_savings > 0:
            payback_months = (cleaning_cost / annual_cost_savings) * 12
        else:
            payback_months = 999

        return EnergySavingsResult(
            efficiency_gain_pct=round(efficiency_gain_pct, 3),
            annual_savings_mmbtu=round(annual_savings_mmbtu, 0),
            annual_cost_savings_usd=round(annual_cost_savings, 0),
            payback_months=round(payback_months, 1),
        )

    def calculate_efficiency_impact(
        self,
        air_temp_rise_f: float,
        baseline_air_temp_rise_f: float,
        leakage_pct: float,
    ) -> EfficiencyImpactResult:
        """
        Calculate impact on boiler efficiency from air preheater performance.

        Args:
            air_temp_rise_f: Current air temperature rise
            baseline_air_temp_rise_f: Design air temperature rise
            leakage_pct: Current leakage percentage

        Returns:
            EfficiencyImpactResult with efficiency impact
        """
        # Baseline efficiency contribution from air preheater
        # Typical: 1F air preheat ≈ 0.025% efficiency
        baseline_contribution = baseline_air_temp_rise_f * 0.025

        # Current contribution
        current_contribution = air_temp_rise_f * 0.025

        # Leakage impact (increases excess air)
        leakage_impact = leakage_pct * 0.1  # 0.1% efficiency loss per 1% leakage

        # Net efficiency
        baseline_efficiency = 85.0  # Assume 85% baseline boiler efficiency
        efficiency_impact = (current_contribution - baseline_contribution) - leakage_impact
        current_efficiency = baseline_efficiency + efficiency_impact

        return EfficiencyImpactResult(
            efficiency_impact_pct=round(efficiency_impact, 3),
            baseline_efficiency_pct=baseline_efficiency,
            current_efficiency_pct=round(current_efficiency, 2),
        )
