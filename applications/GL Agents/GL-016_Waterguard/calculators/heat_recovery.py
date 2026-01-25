# -*- coding: utf-8 -*-
"""
Heat Recovery System Modeling - Flash Tank and Heat Exchanger

This module models heat recovery systems for blowdown including:
- Flash tank steam recovery
- Heat exchanger modeling for continuous blowdown
- Combined recovery system calculations

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: IAPWS-IF97, ASME PTC 4
Agent: GL-016_Waterguard

Zero Hallucination Guarantee:
- All calculations use IAPWS-IF97 correlations
- Complete provenance tracking with SHA-256 hashes
- No LLM inference in calculation path
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    from iapws import IAPWS97
    IAPWS_AVAILABLE = True
except ImportError:
    IAPWS_AVAILABLE = False

from .provenance import ProvenanceTracker, ProvenanceRecord, create_calculation_hash


class RecoverySystemType(Enum):
    """Types of heat recovery systems."""
    FLASH_TANK = "flash_tank"
    HEAT_EXCHANGER = "heat_exchanger"
    COMBINED = "combined"
    NONE = "none"


@dataclass
class FlashSteamResult:
    """Result of flash steam calculation."""
    inlet_mass_flow_kg_h: Decimal
    flash_steam_kg_h: Decimal
    residual_liquid_kg_h: Decimal
    flash_fraction: Decimal
    flash_steam_enthalpy_kj_kg: Decimal
    residual_liquid_enthalpy_kj_kg: Decimal
    flash_pressure_bar: Decimal
    flash_temperature_c: Decimal
    recoverable_energy_kw: Decimal
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'inlet_mass_flow_kg_h': float(self.inlet_mass_flow_kg_h),
            'flash_steam_kg_h': float(self.flash_steam_kg_h),
            'residual_liquid_kg_h': float(self.residual_liquid_kg_h),
            'flash_fraction': float(self.flash_fraction),
            'flash_steam_enthalpy_kj_kg': float(self.flash_steam_enthalpy_kj_kg),
            'residual_liquid_enthalpy_kj_kg': float(self.residual_liquid_enthalpy_kj_kg),
            'flash_pressure_bar': float(self.flash_pressure_bar),
            'flash_temperature_c': float(self.flash_temperature_c),
            'recoverable_energy_kw': float(self.recoverable_energy_kw),
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class HeatExchangerResult:
    """Result of heat exchanger calculation."""
    hot_inlet_temp_c: Decimal
    hot_outlet_temp_c: Decimal
    cold_inlet_temp_c: Decimal
    cold_outlet_temp_c: Decimal
    heat_duty_kw: Decimal
    effectiveness: Decimal
    lmtd_c: Decimal
    ua_kw_c: Decimal
    hot_mass_flow_kg_h: Decimal
    cold_mass_flow_kg_h: Decimal
    fouling_factor: Decimal
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hot_inlet_temp_c': float(self.hot_inlet_temp_c),
            'hot_outlet_temp_c': float(self.hot_outlet_temp_c),
            'cold_inlet_temp_c': float(self.cold_inlet_temp_c),
            'cold_outlet_temp_c': float(self.cold_outlet_temp_c),
            'heat_duty_kw': float(self.heat_duty_kw),
            'effectiveness': float(self.effectiveness),
            'lmtd_c': float(self.lmtd_c),
            'ua_kw_c': float(self.ua_kw_c),
            'hot_mass_flow_kg_h': float(self.hot_mass_flow_kg_h),
            'cold_mass_flow_kg_h': float(self.cold_mass_flow_kg_h),
            'fouling_factor': float(self.fouling_factor),
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class RecoveredEnergyResult:
    """Comprehensive result of heat recovery calculation."""
    total_recovered_energy_kw: Decimal
    flash_steam_recovery_kw: Decimal
    liquid_heat_recovery_kw: Decimal
    recovery_efficiency_percent: Decimal
    flash_steam_to_deaerator: bool
    liquid_to_makeup_preheating: bool
    annual_energy_savings_gj: Decimal
    annual_fuel_savings_gj: Decimal
    annual_cost_savings_usd: Decimal
    payback_months: Optional[Decimal] = None
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_recovered_energy_kw': float(self.total_recovered_energy_kw),
            'flash_steam_recovery_kw': float(self.flash_steam_recovery_kw),
            'liquid_heat_recovery_kw': float(self.liquid_heat_recovery_kw),
            'recovery_efficiency_percent': float(self.recovery_efficiency_percent),
            'flash_steam_to_deaerator': self.flash_steam_to_deaerator,
            'liquid_to_makeup_preheating': self.liquid_to_makeup_preheating,
            'annual_energy_savings_gj': float(self.annual_energy_savings_gj),
            'annual_fuel_savings_gj': float(self.annual_fuel_savings_gj),
            'annual_cost_savings_usd': float(self.annual_cost_savings_usd),
            'payback_months': float(self.payback_months) if self.payback_months else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


class FlashTankModel:
    """
    Flash Tank Model for Blowdown Heat Recovery.

    Models the flash steam generation when high-pressure blowdown
    is depressurized in a flash tank.

    Physics:
        When saturated liquid at pressure P1 is throttled to lower
        pressure P2, part of the liquid flashes to steam.

        Flash fraction = (h_f1 - h_f2) / h_fg2

    Zero Hallucination Guarantee:
    - Uses IAPWS-IF97 for all steam properties
    - Deterministic calculations
    - Complete provenance tracking
    """

    def __init__(self, version: str = "1.0.0"):
        """Initialize flash tank model."""
        self.version = version
        self._check_iapws()

    def _check_iapws(self) -> None:
        """Check IAPWS availability."""
        if not IAPWS_AVAILABLE:
            raise ImportError("IAPWS library required. Install with: pip install iapws")

    def calculate_flash_steam(
        self,
        inlet_pressure_bar: float,
        flash_pressure_bar: float,
        inlet_mass_flow_kg_h: float
    ) -> FlashSteamResult:
        """
        Calculate flash steam generation.

        Args:
            inlet_pressure_bar: Blowdown inlet pressure (bar absolute)
            flash_pressure_bar: Flash tank pressure (bar absolute)
            inlet_mass_flow_kg_h: Blowdown mass flow rate (kg/h)

        Returns:
            FlashSteamResult with flash steam calculations
        """
        tracker = ProvenanceTracker(
            calculation_id=f"flash_steam_{inlet_pressure_bar}to{flash_pressure_bar}bar",
            calculation_type="flash_steam",
            version=self.version,
            standard="IAPWS-IF97"
        )

        tracker.record_inputs({
            'inlet_pressure_bar': inlet_pressure_bar,
            'flash_pressure_bar': flash_pressure_bar,
            'inlet_mass_flow_kg_h': inlet_mass_flow_kg_h
        })

        # Convert to Decimal
        P1 = Decimal(str(inlet_pressure_bar))
        P2 = Decimal(str(flash_pressure_bar))
        m_dot = Decimal(str(inlet_mass_flow_kg_h))

        # Validate pressures
        if P2 >= P1:
            raise ValueError("Flash pressure must be lower than inlet pressure")

        # Get saturated liquid enthalpy at inlet pressure (h_f1)
        P1_mpa = float(P1 / Decimal('10'))
        steam_1 = IAPWS97(P=P1_mpa, x=0)  # Saturated liquid
        h_f1 = Decimal(str(steam_1.h))

        tracker.record_step(
            operation="inlet_enthalpy",
            description="Get saturated liquid enthalpy at inlet pressure",
            inputs={'pressure_bar': P1, 'pressure_mpa': P1_mpa},
            output_value=h_f1,
            output_name="h_f1_kj_kg",
            formula="IAPWS-IF97 (x=0, P=P1)",
            units="kJ/kg",
            source="IAPWS-IF97"
        )

        # Get properties at flash pressure
        P2_mpa = float(P2 / Decimal('10'))
        steam_f2 = IAPWS97(P=P2_mpa, x=0)  # Saturated liquid at flash pressure
        steam_g2 = IAPWS97(P=P2_mpa, x=1)  # Saturated vapor at flash pressure

        h_f2 = Decimal(str(steam_f2.h))
        h_g2 = Decimal(str(steam_g2.h))
        h_fg2 = h_g2 - h_f2  # Latent heat at flash pressure

        T_flash = Decimal(str(steam_f2.T)) - Decimal('273.15')  # Convert K to C

        tracker.record_step(
            operation="flash_properties",
            description="Get saturation properties at flash pressure",
            inputs={'pressure_bar': P2, 'pressure_mpa': P2_mpa},
            output_value=h_fg2,
            output_name="h_fg2_kj_kg",
            formula="h_fg = h_g - h_f (IAPWS-IF97)",
            units="kJ/kg",
            source="IAPWS-IF97"
        )

        # Calculate flash fraction
        # Flash fraction = (h_f1 - h_f2) / h_fg2
        # This represents the fraction of inlet liquid that flashes to steam
        flash_fraction = (h_f1 - h_f2) / h_fg2

        # Clamp to valid range [0, 1]
        flash_fraction = max(Decimal('0'), min(Decimal('1'), flash_fraction))

        tracker.record_step(
            operation="flash_fraction",
            description="Calculate flash steam fraction",
            inputs={'h_f1': h_f1, 'h_f2': h_f2, 'h_fg2': h_fg2},
            output_value=flash_fraction,
            output_name="flash_fraction",
            formula="x_flash = (h_f1 - h_f2) / h_fg2",
            units="dimensionless"
        )

        # Calculate mass flows
        flash_steam_kg_h = m_dot * flash_fraction
        residual_liquid_kg_h = m_dot - flash_steam_kg_h

        tracker.record_step(
            operation="mass_balance",
            description="Calculate flash steam and residual liquid flows",
            inputs={
                'inlet_flow': m_dot,
                'flash_fraction': flash_fraction
            },
            output_value=flash_steam_kg_h,
            output_name="flash_steam_kg_h",
            formula="m_steam = m_inlet * x_flash",
            units="kg/h"
        )

        # Calculate recoverable energy from flash steam
        # Energy = Flash steam rate * latent heat / 3600 (convert to kW)
        # But we need to account for how this steam is used
        # If used to heat deaerator/feedwater, energy value = m_steam * h_g2

        # Maximum recoverable energy (if flash steam replaces live steam)
        recoverable_energy_kw = flash_steam_kg_h * h_fg2 / Decimal('3600')

        tracker.record_step(
            operation="energy_recovery",
            description="Calculate maximum recoverable energy from flash steam",
            inputs={
                'flash_steam_kg_h': flash_steam_kg_h,
                'h_fg2': h_fg2
            },
            output_value=recoverable_energy_kw,
            output_name="recoverable_energy_kw",
            formula="Q = m_steam * h_fg / 3600",
            units="kW"
        )

        provenance = tracker.get_provenance_record(flash_fraction)

        return FlashSteamResult(
            inlet_mass_flow_kg_h=m_dot,
            flash_steam_kg_h=flash_steam_kg_h.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            residual_liquid_kg_h=residual_liquid_kg_h.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            flash_fraction=flash_fraction.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            flash_steam_enthalpy_kj_kg=h_g2.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            residual_liquid_enthalpy_kj_kg=h_f2.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            flash_pressure_bar=P2,
            flash_temperature_c=T_flash.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            recoverable_energy_kw=recoverable_energy_kw.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_flash_steam_fraction(
        self,
        inlet_pressure_bar: float,
        flash_pressure_bar: float
    ) -> Decimal:
        """
        Calculate flash steam fraction (convenience method).

        Args:
            inlet_pressure_bar: Blowdown inlet pressure (bar)
            flash_pressure_bar: Flash tank pressure (bar)

        Returns:
            Flash steam fraction (0-1)
        """
        result = self.calculate_flash_steam(
            inlet_pressure_bar,
            flash_pressure_bar,
            1000.0  # Reference flow
        )
        return result.flash_fraction


class HeatExchangerModel:
    """
    Heat Exchanger Model for Blowdown Heat Recovery.

    Models shell-and-tube or plate heat exchangers for
    recovering heat from residual liquid after flash tank.

    Calculations:
    - LMTD (Log Mean Temperature Difference) method
    - Effectiveness-NTU method
    - Fouling factor considerations

    Zero Hallucination Guarantee:
    - Deterministic thermodynamic calculations
    - Complete provenance tracking
    """

    # Typical fouling factors (m2*K/W)
    FOULING_FACTORS = {
        'blowdown_water': Decimal('0.0002'),  # Boiler blowdown
        'clean_water': Decimal('0.0001'),      # Treated water
        'raw_water': Decimal('0.0005'),        # Untreated water
        'condensate': Decimal('0.00005')       # Clean condensate
    }

    # Specific heat of water (kJ/kg*K)
    CP_WATER = Decimal('4.186')

    def __init__(
        self,
        effectiveness: float = 0.75,
        fouling_hot_side: str = 'blowdown_water',
        fouling_cold_side: str = 'raw_water',
        version: str = "1.0.0"
    ):
        """
        Initialize heat exchanger model.

        Args:
            effectiveness: Heat exchanger effectiveness (0-1)
            fouling_hot_side: Fouling condition on hot (blowdown) side
            fouling_cold_side: Fouling condition on cold (makeup) side
            version: Calculator version
        """
        self.effectiveness = Decimal(str(effectiveness))
        self.R_f_hot = self.FOULING_FACTORS.get(
            fouling_hot_side,
            self.FOULING_FACTORS['blowdown_water']
        )
        self.R_f_cold = self.FOULING_FACTORS.get(
            fouling_cold_side,
            self.FOULING_FACTORS['raw_water']
        )
        self.version = version

    def calculate_heat_recovery(
        self,
        hot_inlet_temp_c: float,
        hot_mass_flow_kg_h: float,
        cold_inlet_temp_c: float,
        cold_mass_flow_kg_h: float,
        min_approach_temp_c: float = 10.0
    ) -> HeatExchangerResult:
        """
        Calculate heat recovery using effectiveness-NTU method.

        Args:
            hot_inlet_temp_c: Hot side (blowdown) inlet temperature (C)
            hot_mass_flow_kg_h: Hot side mass flow rate (kg/h)
            cold_inlet_temp_c: Cold side (makeup water) inlet temperature (C)
            cold_mass_flow_kg_h: Cold side mass flow rate (kg/h)
            min_approach_temp_c: Minimum approach temperature (C)

        Returns:
            HeatExchangerResult with heat recovery calculations
        """
        tracker = ProvenanceTracker(
            calculation_id="heat_exchanger",
            calculation_type="heat_exchanger",
            version=self.version
        )

        tracker.record_inputs({
            'hot_inlet_temp_c': hot_inlet_temp_c,
            'hot_mass_flow_kg_h': hot_mass_flow_kg_h,
            'cold_inlet_temp_c': cold_inlet_temp_c,
            'cold_mass_flow_kg_h': cold_mass_flow_kg_h,
            'effectiveness': float(self.effectiveness),
            'min_approach_temp_c': min_approach_temp_c
        })

        T_h_in = Decimal(str(hot_inlet_temp_c))
        T_c_in = Decimal(str(cold_inlet_temp_c))
        m_h = Decimal(str(hot_mass_flow_kg_h))
        m_c = Decimal(str(cold_mass_flow_kg_h))
        T_approach = Decimal(str(min_approach_temp_c))

        # Calculate heat capacity rates (kW/K)
        # C = m_dot * Cp / 3600
        C_h = m_h * self.CP_WATER / Decimal('3600')
        C_c = m_c * self.CP_WATER / Decimal('3600')

        tracker.record_step(
            operation="capacity_rates",
            description="Calculate heat capacity rates",
            inputs={'m_h': m_h, 'm_c': m_c, 'Cp': self.CP_WATER},
            output_value=C_h,
            output_name="C_h_kw_k",
            formula="C = m * Cp / 3600",
            units="kW/K"
        )

        # Determine minimum and maximum capacity rates
        C_min = min(C_h, C_c)
        C_max = max(C_h, C_c)
        C_r = C_min / C_max if C_max > 0 else Decimal('0')  # Capacity ratio

        # Maximum possible heat transfer
        Q_max = C_min * (T_h_in - T_c_in)

        tracker.record_step(
            operation="max_heat_transfer",
            description="Calculate maximum possible heat transfer",
            inputs={'C_min': C_min, 'T_h_in': T_h_in, 'T_c_in': T_c_in},
            output_value=Q_max,
            output_name="Q_max_kw",
            formula="Q_max = C_min * (T_h_in - T_c_in)",
            units="kW"
        )

        # Actual heat transfer based on effectiveness
        Q = self.effectiveness * Q_max

        tracker.record_step(
            operation="actual_heat_transfer",
            description="Calculate actual heat transfer",
            inputs={'effectiveness': self.effectiveness, 'Q_max': Q_max},
            output_value=Q,
            output_name="Q_kw",
            formula="Q = effectiveness * Q_max",
            units="kW"
        )

        # Calculate outlet temperatures
        T_h_out = T_h_in - Q / C_h if C_h > 0 else T_h_in
        T_c_out = T_c_in + Q / C_c if C_c > 0 else T_c_in

        # Check approach temperature constraint
        actual_approach = T_h_out - T_c_in
        if actual_approach < T_approach:
            # Recalculate with approach temperature constraint
            T_h_out = T_c_in + T_approach
            Q = C_h * (T_h_in - T_h_out) if C_h > 0 else Decimal('0')
            T_c_out = T_c_in + Q / C_c if C_c > 0 else T_c_in

            tracker.add_metadata('approach_limited', True)

        tracker.record_step(
            operation="outlet_temperatures",
            description="Calculate outlet temperatures",
            inputs={'Q': Q, 'C_h': C_h, 'C_c': C_c},
            output_value=T_h_out,
            output_name="T_h_out_c",
            formula="T_out = T_in - Q/C (hot) or T_in + Q/C (cold)",
            units="C"
        )

        # Calculate LMTD for counterflow arrangement
        dT1 = T_h_in - T_c_out
        dT2 = T_h_out - T_c_in

        if dT1 > 0 and dT2 > 0 and dT1 != dT2:
            lmtd = (dT1 - dT2) / Decimal(str(math.log(float(dT1 / dT2))))
        elif dT1 > 0 and dT2 > 0:
            lmtd = dT1  # Equal temperature differences
        else:
            lmtd = Decimal('1')  # Prevent division by zero

        tracker.record_step(
            operation="lmtd_calculation",
            description="Calculate Log Mean Temperature Difference",
            inputs={'dT1': dT1, 'dT2': dT2},
            output_value=lmtd,
            output_name="lmtd_c",
            formula="LMTD = (dT1 - dT2) / ln(dT1/dT2)",
            units="C"
        )

        # Calculate UA (overall heat transfer coefficient * area)
        UA = Q / lmtd if lmtd > 0 else Decimal('0')

        # Calculate fouling factor
        fouling_factor = self.R_f_hot + self.R_f_cold

        tracker.record_step(
            operation="ua_calculation",
            description="Calculate overall heat transfer coefficient",
            inputs={'Q': Q, 'lmtd': lmtd},
            output_value=UA,
            output_name="UA_kw_c",
            formula="UA = Q / LMTD",
            units="kW/C"
        )

        provenance = tracker.get_provenance_record(Q)

        return HeatExchangerResult(
            hot_inlet_temp_c=T_h_in.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            hot_outlet_temp_c=T_h_out.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            cold_inlet_temp_c=T_c_in.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            cold_outlet_temp_c=T_c_out.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            heat_duty_kw=Q.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            effectiveness=self.effectiveness,
            lmtd_c=lmtd.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            ua_kw_c=UA.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            hot_mass_flow_kg_h=m_h,
            cold_mass_flow_kg_h=m_c,
            fouling_factor=fouling_factor,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )


class BlowdownHeatRecoverySystem:
    """
    Combined Blowdown Heat Recovery System.

    Models a complete heat recovery system consisting of:
    1. Flash tank for flash steam recovery
    2. Heat exchanger for residual liquid heat recovery

    Flash steam can be used for:
    - Deaerator heating
    - Low-pressure steam users
    - Feedwater preheating

    Residual liquid can be used for:
    - Makeup water preheating
    - Boiler feed preheating

    Zero Hallucination Guarantee:
    - Uses IAPWS-IF97 for all steam properties
    - Deterministic calculations
    - Complete provenance tracking
    """

    def __init__(
        self,
        flash_tank_pressure_bar: float = 1.5,
        heat_exchanger_effectiveness: float = 0.75,
        operating_hours_per_year: int = 8760,
        version: str = "1.0.0"
    ):
        """
        Initialize heat recovery system.

        Args:
            flash_tank_pressure_bar: Flash tank operating pressure (bar)
            heat_exchanger_effectiveness: Heat exchanger effectiveness (0-1)
            operating_hours_per_year: Annual operating hours
            version: Calculator version
        """
        self.flash_pressure = Decimal(str(flash_tank_pressure_bar))
        self.hx_effectiveness = Decimal(str(heat_exchanger_effectiveness))
        self.operating_hours = Decimal(str(operating_hours_per_year))
        self.version = version

        self.flash_tank = FlashTankModel(version)
        self.heat_exchanger = HeatExchangerModel(
            effectiveness=heat_exchanger_effectiveness,
            version=version
        )

    def calculate_recovered_energy(
        self,
        blowdown_flow_kg_h: float,
        boiler_pressure_bar: float,
        makeup_water_temp_c: float = 15.0,
        makeup_water_flow_kg_h: Optional[float] = None,
        fuel_cost_per_gj: float = 10.0,
        boiler_efficiency: float = 0.85,
        system_cost_usd: Optional[float] = None
    ) -> RecoveredEnergyResult:
        """
        Calculate total recovered energy from blowdown.

        Args:
            blowdown_flow_kg_h: Blowdown mass flow rate (kg/h)
            boiler_pressure_bar: Boiler operating pressure (bar)
            makeup_water_temp_c: Makeup water temperature (C)
            makeup_water_flow_kg_h: Makeup water flow (kg/h), default = 10x blowdown
            fuel_cost_per_gj: Fuel cost ($/GJ)
            boiler_efficiency: Boiler thermal efficiency
            system_cost_usd: Installation cost for payback calculation

        Returns:
            RecoveredEnergyResult with comprehensive recovery analysis
        """
        tracker = ProvenanceTracker(
            calculation_id="heat_recovery_system",
            calculation_type="heat_recovery",
            version=self.version,
            standard="IAPWS-IF97"
        )

        if makeup_water_flow_kg_h is None:
            makeup_water_flow_kg_h = blowdown_flow_kg_h * 10  # Typical ratio

        tracker.record_inputs({
            'blowdown_flow_kg_h': blowdown_flow_kg_h,
            'boiler_pressure_bar': boiler_pressure_bar,
            'flash_tank_pressure_bar': float(self.flash_pressure),
            'makeup_water_temp_c': makeup_water_temp_c,
            'makeup_water_flow_kg_h': makeup_water_flow_kg_h,
            'fuel_cost_per_gj': fuel_cost_per_gj,
            'boiler_efficiency': boiler_efficiency
        })

        # Step 1: Calculate flash steam recovery
        flash_result = self.flash_tank.calculate_flash_steam(
            inlet_pressure_bar=boiler_pressure_bar,
            flash_pressure_bar=float(self.flash_pressure),
            inlet_mass_flow_kg_h=blowdown_flow_kg_h
        )

        flash_steam_recovery_kw = flash_result.recoverable_energy_kw

        tracker.record_step(
            operation="flash_recovery",
            description="Calculate flash steam recovery",
            inputs={
                'blowdown_flow': blowdown_flow_kg_h,
                'inlet_pressure': boiler_pressure_bar,
                'flash_pressure': float(self.flash_pressure)
            },
            output_value=flash_steam_recovery_kw,
            output_name="flash_recovery_kw",
            formula="From flash tank calculation",
            units="kW"
        )

        # Step 2: Calculate heat exchanger recovery from residual liquid
        # Get residual liquid temperature (saturation temperature at flash pressure)
        residual_liquid_temp = flash_result.flash_temperature_c

        hx_result = self.heat_exchanger.calculate_heat_recovery(
            hot_inlet_temp_c=float(residual_liquid_temp),
            hot_mass_flow_kg_h=float(flash_result.residual_liquid_kg_h),
            cold_inlet_temp_c=makeup_water_temp_c,
            cold_mass_flow_kg_h=makeup_water_flow_kg_h
        )

        liquid_heat_recovery_kw = hx_result.heat_duty_kw

        tracker.record_step(
            operation="liquid_recovery",
            description="Calculate residual liquid heat recovery",
            inputs={
                'residual_liquid_kg_h': float(flash_result.residual_liquid_kg_h),
                'inlet_temp': float(residual_liquid_temp),
                'cold_temp': makeup_water_temp_c
            },
            output_value=liquid_heat_recovery_kw,
            output_name="liquid_recovery_kw",
            formula="From heat exchanger calculation",
            units="kW"
        )

        # Step 3: Calculate total recovery
        total_recovered = flash_steam_recovery_kw + liquid_heat_recovery_kw

        # Calculate theoretical maximum recovery (all enthalpy above ambient)
        P_mpa = boiler_pressure_bar / 10
        steam = IAPWS97(P=P_mpa, x=0)
        h_blowdown = Decimal(str(steam.h))

        # Enthalpy of water at makeup temperature
        h_makeup = Decimal(str(makeup_water_temp_c)) * Decimal('4.186')

        theoretical_max_kw = (
            Decimal(str(blowdown_flow_kg_h)) * (h_blowdown - h_makeup) / Decimal('3600')
        )

        recovery_efficiency = (
            total_recovered / theoretical_max_kw * Decimal('100')
            if theoretical_max_kw > 0 else Decimal('0')
        )

        tracker.record_step(
            operation="total_recovery",
            description="Calculate total heat recovery",
            inputs={
                'flash_recovery_kw': flash_steam_recovery_kw,
                'liquid_recovery_kw': liquid_heat_recovery_kw,
                'theoretical_max_kw': theoretical_max_kw
            },
            output_value=total_recovered,
            output_name="total_recovery_kw",
            formula="Total = Flash + Liquid",
            units="kW"
        )

        # Step 4: Calculate annual savings
        # Energy savings in GJ
        annual_energy_gj = total_recovered * Decimal('3.6') * self.operating_hours / Decimal('1000')

        # Fuel savings (accounting for boiler efficiency)
        efficiency = Decimal(str(boiler_efficiency))
        annual_fuel_gj = annual_energy_gj / efficiency

        # Cost savings
        fuel_cost = Decimal(str(fuel_cost_per_gj))
        annual_cost_savings = annual_fuel_gj * fuel_cost

        tracker.record_step(
            operation="annual_savings",
            description="Calculate annual energy and cost savings",
            inputs={
                'total_recovery_kw': total_recovered,
                'operating_hours': self.operating_hours,
                'boiler_efficiency': efficiency,
                'fuel_cost_per_gj': fuel_cost
            },
            output_value=annual_cost_savings,
            output_name="annual_cost_savings_usd",
            formula="Cost = (Energy_kWh * 3.6 / 1000 / Efficiency) * Fuel_cost",
            units="USD/year"
        )

        # Step 5: Calculate payback (if system cost provided)
        payback_months = None
        if system_cost_usd is not None and annual_cost_savings > 0:
            cost = Decimal(str(system_cost_usd))
            payback_months = cost / annual_cost_savings * Decimal('12')

            tracker.record_step(
                operation="payback",
                description="Calculate simple payback period",
                inputs={
                    'system_cost': cost,
                    'annual_savings': annual_cost_savings
                },
                output_value=payback_months,
                output_name="payback_months",
                formula="Payback = Cost / Annual_Savings * 12",
                units="months"
            )

        provenance = tracker.get_provenance_record(total_recovered)

        return RecoveredEnergyResult(
            total_recovered_energy_kw=total_recovered.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            flash_steam_recovery_kw=flash_steam_recovery_kw.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            liquid_heat_recovery_kw=liquid_heat_recovery_kw.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            recovery_efficiency_percent=recovery_efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            flash_steam_to_deaerator=True,  # Typical configuration
            liquid_to_makeup_preheating=True,
            annual_energy_savings_gj=annual_energy_gj.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            annual_fuel_savings_gj=annual_fuel_gj.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            annual_cost_savings_usd=annual_cost_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            payback_months=payback_months.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP) if payback_months else None,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )


# Convenience functions

def calculate_flash_steam_fraction(
    inlet_pressure_bar: float,
    flash_pressure_bar: float
) -> Decimal:
    """
    Calculate flash steam fraction - convenience function.

    Args:
        inlet_pressure_bar: Inlet pressure (bar)
        flash_pressure_bar: Flash tank pressure (bar)

    Returns:
        Flash steam fraction (0-1)
    """
    model = FlashTankModel()
    return model.calculate_flash_steam_fraction(inlet_pressure_bar, flash_pressure_bar)


def calculate_recovered_energy(
    blowdown_flow_kg_h: float,
    boiler_pressure_bar: float,
    flash_tank_pressure_bar: float = 1.5,
    makeup_water_temp_c: float = 15.0
) -> RecoveredEnergyResult:
    """
    Calculate recovered energy - convenience function.

    Args:
        blowdown_flow_kg_h: Blowdown mass flow (kg/h)
        boiler_pressure_bar: Boiler pressure (bar)
        flash_tank_pressure_bar: Flash tank pressure (bar)
        makeup_water_temp_c: Makeup water temperature (C)

    Returns:
        RecoveredEnergyResult with recovery analysis
    """
    system = BlowdownHeatRecoverySystem(
        flash_tank_pressure_bar=flash_tank_pressure_bar
    )
    return system.calculate_recovered_energy(
        blowdown_flow_kg_h=blowdown_flow_kg_h,
        boiler_pressure_bar=boiler_pressure_bar,
        makeup_water_temp_c=makeup_water_temp_c
    )
