# -*- coding: utf-8 -*-
"""
Thermal Calculator - IAPWS-IF97 Blowdown Energy Loss Calculations

This module implements blowdown energy loss calculations using IAPWS-IF97
water/steam property correlations with zero hallucination guarantee.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: IAPWS-IF97, ASME Steam Tables
Agent: GL-016_Waterguard

Dependencies:
    - iapws: IAPWS-IF97 implementation for water/steam properties
    - pint: Physical units handling
    - decimal: Arbitrary precision arithmetic
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    from iapws import IAPWS97
    IAPWS_AVAILABLE = True
except ImportError:
    IAPWS_AVAILABLE = False

try:
    import pint
    ureg = pint.UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

from .provenance import ProvenanceTracker, ProvenanceRecord, create_calculation_hash


class EnergyUnit(Enum):
    """Supported energy units."""
    KW = "kW"
    BTU_H = "Btu/h"
    KJ_H = "kJ/h"
    MW = "MW"
    GJ_H = "GJ/h"


class MassFlowUnit(Enum):
    """Supported mass flow units."""
    KG_H = "kg/h"
    KG_S = "kg/s"
    LB_H = "lb/h"
    T_H = "t/h"


class PressureUnit(Enum):
    """Supported pressure units."""
    BAR = "bar"
    MPA = "MPa"
    PSI = "psi"
    KPA = "kPa"
    ATM = "atm"


class TemperatureUnit(Enum):
    """Supported temperature units."""
    CELSIUS = "C"
    KELVIN = "K"
    FAHRENHEIT = "F"


@dataclass
class ThermalResult:
    """Result of thermal calculation with complete provenance."""
    value: Decimal
    unit: str
    uncertainty: Optional[Decimal] = None
    uncertainty_percent: Optional[Decimal] = None
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'value': float(self.value),
            'unit': self.unit,
            'uncertainty': float(self.uncertainty) if self.uncertainty else None,
            'uncertainty_percent': float(self.uncertainty_percent) if self.uncertainty_percent else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class BlowdownEnergyResult:
    """Comprehensive blowdown energy loss calculation result."""
    energy_loss_rate_kw: Decimal
    energy_loss_rate_btu_h: Decimal
    blowdown_enthalpy_kj_kg: Decimal
    feedwater_enthalpy_kj_kg: Decimal
    enthalpy_difference_kj_kg: Decimal
    blowdown_mass_flow_kg_h: Decimal
    saturation_temperature_c: Decimal
    uncertainty_kw: Optional[Decimal] = None
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'energy_loss_rate_kw': float(self.energy_loss_rate_kw),
            'energy_loss_rate_btu_h': float(self.energy_loss_rate_btu_h),
            'blowdown_enthalpy_kj_kg': float(self.blowdown_enthalpy_kj_kg),
            'feedwater_enthalpy_kj_kg': float(self.feedwater_enthalpy_kj_kg),
            'enthalpy_difference_kj_kg': float(self.enthalpy_difference_kj_kg),
            'blowdown_mass_flow_kg_h': float(self.blowdown_mass_flow_kg_h),
            'saturation_temperature_c': float(self.saturation_temperature_c),
            'uncertainty_kw': float(self.uncertainty_kw) if self.uncertainty_kw else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


class UnitConverter:
    """
    Unit conversion utilities for thermal calculations.

    All conversions are deterministic and traceable.
    """

    # Conversion factors (exact or NIST reference values)
    CONVERSION_FACTORS = {
        # Energy conversions
        ('kW', 'Btu/h'): Decimal('3412.142'),
        ('kW', 'kJ/h'): Decimal('3600'),
        ('kW', 'MW'): Decimal('0.001'),
        ('kW', 'GJ/h'): Decimal('0.0036'),

        # Mass flow conversions
        ('kg/h', 'kg/s'): Decimal('1') / Decimal('3600'),
        ('kg/h', 'lb/h'): Decimal('2.20462'),
        ('kg/h', 't/h'): Decimal('0.001'),

        # Pressure conversions
        ('bar', 'MPa'): Decimal('0.1'),
        ('bar', 'psi'): Decimal('14.5038'),
        ('bar', 'kPa'): Decimal('100'),
        ('bar', 'atm'): Decimal('0.986923'),

        # Temperature conversions (offsets, not factors)
        ('C', 'K'): Decimal('273.15'),  # T_K = T_C + 273.15
        ('C', 'F'): (Decimal('1.8'), Decimal('32')),  # T_F = T_C * 1.8 + 32
    }

    @classmethod
    def convert_energy(
        cls,
        value: Decimal,
        from_unit: str,
        to_unit: str
    ) -> Decimal:
        """Convert energy units - DETERMINISTIC."""
        if from_unit == to_unit:
            return value

        # Convert to kW first, then to target
        if from_unit != 'kW':
            factor = cls.CONVERSION_FACTORS.get((from_unit, 'kW'))
            if factor is None:
                # Try inverse
                factor = cls.CONVERSION_FACTORS.get(('kW', from_unit))
                if factor:
                    factor = Decimal('1') / factor
            if factor:
                value = value * factor

        # Now convert from kW to target
        if to_unit != 'kW':
            factor = cls.CONVERSION_FACTORS.get(('kW', to_unit))
            if factor:
                value = value * factor

        return value.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_pressure(
        cls,
        value: Decimal,
        from_unit: str,
        to_unit: str
    ) -> Decimal:
        """Convert pressure units - DETERMINISTIC."""
        if from_unit == to_unit:
            return value

        # Convert to bar first
        if from_unit != 'bar':
            factor = cls.CONVERSION_FACTORS.get((from_unit, 'bar'))
            if factor is None:
                factor = cls.CONVERSION_FACTORS.get(('bar', from_unit))
                if factor:
                    factor = Decimal('1') / factor
            if factor:
                value = value * factor

        # Convert from bar to target
        if to_unit != 'bar':
            factor = cls.CONVERSION_FACTORS.get(('bar', to_unit))
            if factor:
                value = value * factor

        return value.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_temperature(
        cls,
        value: Decimal,
        from_unit: str,
        to_unit: str
    ) -> Decimal:
        """Convert temperature units - DETERMINISTIC."""
        if from_unit == to_unit:
            return value

        # Convert to Celsius first
        if from_unit == 'K':
            value = value - Decimal('273.15')
        elif from_unit == 'F':
            value = (value - Decimal('32')) / Decimal('1.8')

        # Convert from Celsius to target
        if to_unit == 'K':
            value = value + Decimal('273.15')
        elif to_unit == 'F':
            value = value * Decimal('1.8') + Decimal('32')

        return value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    @classmethod
    def convert_mass_flow(
        cls,
        value: Decimal,
        from_unit: str,
        to_unit: str
    ) -> Decimal:
        """Convert mass flow units - DETERMINISTIC."""
        if from_unit == to_unit:
            return value

        # Convert to kg/h first
        if from_unit != 'kg/h':
            factor = cls.CONVERSION_FACTORS.get((from_unit, 'kg/h'))
            if factor is None:
                factor = cls.CONVERSION_FACTORS.get(('kg/h', from_unit))
                if factor:
                    factor = Decimal('1') / factor
            if factor:
                value = value * factor

        # Convert from kg/h to target
        if to_unit != 'kg/h':
            factor = cls.CONVERSION_FACTORS.get(('kg/h', to_unit))
            if factor:
                value = value * factor

        return value.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


class IAPWS97Calculator:
    """
    IAPWS-IF97 Steam Property Calculator.

    Provides deterministic thermodynamic property calculations
    based on the IAPWS-IF97 international standard.

    Zero Hallucination Guarantee:
    - All calculations use IAPWS-IF97 correlations
    - No LLM inference
    - Complete provenance tracking
    - Bit-perfect reproducibility
    """

    # Critical point constants (IAPWS-IF97)
    T_CRIT_K = Decimal('647.096')  # Critical temperature (K)
    P_CRIT_MPA = Decimal('22.064')  # Critical pressure (MPa)
    RHO_CRIT = Decimal('322')  # Critical density (kg/m3)

    # Specific gas constant for water
    R = Decimal('0.461526')  # kJ/(kg·K)

    def __init__(self, version: str = "1.0.0"):
        """Initialize IAPWS-IF97 calculator."""
        self.version = version
        self._check_iapws_availability()

    def _check_iapws_availability(self) -> None:
        """Check if IAPWS library is available."""
        if not IAPWS_AVAILABLE:
            raise ImportError(
                "IAPWS library not available. Install with: pip install iapws"
            )

    def get_saturated_liquid_enthalpy(
        self,
        pressure_bar: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get saturated liquid enthalpy at given pressure using IAPWS-IF97.

        This is h_f (liquid enthalpy at saturation).

        Args:
            pressure_bar: Absolute pressure in bar
            tracker: Optional provenance tracker

        Returns:
            Specific enthalpy of saturated liquid (kJ/kg)
        """
        P = Decimal(str(pressure_bar))
        P_mpa = float(P / Decimal('10'))  # Convert bar to MPa

        # Use IAPWS-IF97 library
        steam = IAPWS97(P=P_mpa, x=0)  # x=0 for saturated liquid
        h_f = Decimal(str(steam.h))  # kJ/kg

        if tracker:
            tracker.record_step(
                operation="iapws97_saturated_liquid_enthalpy",
                description="Calculate saturated liquid enthalpy using IAPWS-IF97",
                inputs={'pressure_bar': pressure_bar, 'pressure_mpa': P_mpa},
                output_value=h_f,
                output_name="h_f_kj_kg",
                formula="IAPWS-IF97 Region 4 (saturation)",
                units="kJ/kg",
                source="IAPWS-IF97"
            )

        return h_f.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get_saturated_vapor_enthalpy(
        self,
        pressure_bar: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get saturated vapor enthalpy at given pressure using IAPWS-IF97.

        This is h_g (vapor enthalpy at saturation).

        Args:
            pressure_bar: Absolute pressure in bar
            tracker: Optional provenance tracker

        Returns:
            Specific enthalpy of saturated vapor (kJ/kg)
        """
        P = Decimal(str(pressure_bar))
        P_mpa = float(P / Decimal('10'))

        steam = IAPWS97(P=P_mpa, x=1)  # x=1 for saturated vapor
        h_g = Decimal(str(steam.h))

        if tracker:
            tracker.record_step(
                operation="iapws97_saturated_vapor_enthalpy",
                description="Calculate saturated vapor enthalpy using IAPWS-IF97",
                inputs={'pressure_bar': pressure_bar, 'pressure_mpa': P_mpa},
                output_value=h_g,
                output_name="h_g_kj_kg",
                formula="IAPWS-IF97 Region 4 (saturation)",
                units="kJ/kg",
                source="IAPWS-IF97"
            )

        return h_g.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get_subcooled_enthalpy(
        self,
        temperature_c: float,
        pressure_bar: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get subcooled (compressed) liquid enthalpy using IAPWS-IF97.

        This is used for feedwater enthalpy calculation.

        Args:
            temperature_c: Temperature in Celsius
            pressure_bar: Absolute pressure in bar
            tracker: Optional provenance tracker

        Returns:
            Specific enthalpy of subcooled liquid (kJ/kg)
        """
        T = Decimal(str(temperature_c))
        P = Decimal(str(pressure_bar))

        T_k = float(T + Decimal('273.15'))  # Convert to Kelvin
        P_mpa = float(P / Decimal('10'))

        # Use IAPWS-IF97 for subcooled liquid (Region 1)
        water = IAPWS97(T=T_k, P=P_mpa)
        h = Decimal(str(water.h))

        if tracker:
            tracker.record_step(
                operation="iapws97_subcooled_enthalpy",
                description="Calculate subcooled liquid enthalpy using IAPWS-IF97",
                inputs={
                    'temperature_c': temperature_c,
                    'temperature_k': T_k,
                    'pressure_bar': pressure_bar,
                    'pressure_mpa': P_mpa
                },
                output_value=h,
                output_name="h_subcooled_kj_kg",
                formula="IAPWS-IF97 Region 1 (compressed liquid)",
                units="kJ/kg",
                source="IAPWS-IF97"
            )

        return h.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get_saturation_temperature(
        self,
        pressure_bar: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get saturation temperature at given pressure using IAPWS-IF97.

        Args:
            pressure_bar: Absolute pressure in bar
            tracker: Optional provenance tracker

        Returns:
            Saturation temperature in Celsius
        """
        P = Decimal(str(pressure_bar))
        P_mpa = float(P / Decimal('10'))

        steam = IAPWS97(P=P_mpa, x=0)
        T_sat_k = Decimal(str(steam.T))
        T_sat_c = T_sat_k - Decimal('273.15')

        if tracker:
            tracker.record_step(
                operation="iapws97_saturation_temperature",
                description="Calculate saturation temperature using IAPWS-IF97",
                inputs={'pressure_bar': pressure_bar, 'pressure_mpa': P_mpa},
                output_value=T_sat_c,
                output_name="T_sat_c",
                formula="IAPWS-IF97 Region 4 (saturation)",
                units="C",
                source="IAPWS-IF97"
            )

        return T_sat_c.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def get_latent_heat(
        self,
        pressure_bar: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get latent heat of vaporization at given pressure.

        h_fg = h_g - h_f

        Args:
            pressure_bar: Absolute pressure in bar
            tracker: Optional provenance tracker

        Returns:
            Latent heat of vaporization (kJ/kg)
        """
        h_f = self.get_saturated_liquid_enthalpy(pressure_bar, tracker)
        h_g = self.get_saturated_vapor_enthalpy(pressure_bar, tracker)

        h_fg = h_g - h_f

        if tracker:
            tracker.record_step(
                operation="latent_heat_calculation",
                description="Calculate latent heat of vaporization",
                inputs={'h_f': h_f, 'h_g': h_g},
                output_value=h_fg,
                output_name="h_fg_kj_kg",
                formula="h_fg = h_g - h_f",
                units="kJ/kg",
                source="IAPWS-IF97"
            )

        return h_fg.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


class BlowdownThermalCalculator:
    """
    Blowdown Energy Loss Calculator.

    Calculates energy losses from boiler blowdown using IAPWS-IF97
    thermodynamic properties.

    Zero Hallucination Guarantee:
    - All calculations are deterministic
    - Uses IAPWS-IF97 for steam/water properties
    - Complete provenance tracking with SHA-256 hashes
    - No LLM inference in calculation path

    Formula:
        Energy_Loss_Rate = Blowdown_Mass_Flow * (h_sat_liquid(Boiler_P) - h_feedwater(FW_T, FW_P))

    Reference:
        - ASME PTC 4 - Steam Generators
        - ABMA Boiler Water Limits
    """

    def __init__(self, version: str = "1.0.0"):
        """Initialize blowdown thermal calculator."""
        self.version = version
        self.iapws = IAPWS97Calculator(version)

        # Default uncertainty factors (based on typical sensor accuracy)
        self.uncertainty_factors = {
            'mass_flow': Decimal('0.02'),  # 2% flow meter accuracy
            'pressure': Decimal('0.005'),   # 0.5% pressure transducer
            'temperature': Decimal('0.01'), # 1% temperature sensor
        }

    def calculate_blowdown_energy_loss(
        self,
        blowdown_mass_flow_kg_h: float,
        boiler_pressure_bar: float,
        feedwater_temp_c: float,
        feedwater_pressure_bar: float,
        include_uncertainty: bool = True
    ) -> BlowdownEnergyResult:
        """
        Calculate blowdown energy loss rate.

        Formula:
            Energy_Loss_Rate = Blowdown_Mass_Flow * (h_sat_liquid(P_boiler) - h_feedwater(T_fw, P_fw))

        Args:
            blowdown_mass_flow_kg_h: Blowdown mass flow rate (kg/h)
            boiler_pressure_bar: Boiler operating pressure (bar absolute)
            feedwater_temp_c: Feedwater temperature (Celsius)
            feedwater_pressure_bar: Feedwater pressure (bar absolute)
            include_uncertainty: Calculate uncertainty bounds

        Returns:
            BlowdownEnergyResult with complete provenance
        """
        # Initialize provenance tracker
        calc_id = f"blowdown_energy_{boiler_pressure_bar}bar_{feedwater_temp_c}C"
        tracker = ProvenanceTracker(
            calculation_id=calc_id,
            calculation_type="blowdown_energy_loss",
            version=self.version,
            standard="IAPWS-IF97"
        )

        # Record inputs
        tracker.record_inputs({
            'blowdown_mass_flow_kg_h': blowdown_mass_flow_kg_h,
            'boiler_pressure_bar': boiler_pressure_bar,
            'feedwater_temp_c': feedwater_temp_c,
            'feedwater_pressure_bar': feedwater_pressure_bar
        })

        # Convert to Decimal for precision
        m_dot = Decimal(str(blowdown_mass_flow_kg_h))
        P_boiler = Decimal(str(boiler_pressure_bar))
        T_fw = Decimal(str(feedwater_temp_c))
        P_fw = Decimal(str(feedwater_pressure_bar))

        # Step 1: Get saturation temperature at boiler pressure
        T_sat = self.iapws.get_saturation_temperature(boiler_pressure_bar, tracker)

        # Step 2: Get saturated liquid enthalpy at boiler pressure
        h_blowdown = self.iapws.get_saturated_liquid_enthalpy(boiler_pressure_bar, tracker)

        tracker.record_step(
            operation="blowdown_enthalpy",
            description="Blowdown water is saturated liquid at boiler pressure",
            inputs={'boiler_pressure_bar': boiler_pressure_bar},
            output_value=h_blowdown,
            output_name="h_blowdown_kj_kg",
            formula="h_blowdown = h_f(P_boiler)",
            units="kJ/kg",
            source="IAPWS-IF97"
        )

        # Step 3: Get subcooled feedwater enthalpy
        h_feedwater = self.iapws.get_subcooled_enthalpy(
            feedwater_temp_c, feedwater_pressure_bar, tracker
        )

        tracker.record_step(
            operation="feedwater_enthalpy",
            description="Feedwater is subcooled liquid",
            inputs={
                'feedwater_temp_c': feedwater_temp_c,
                'feedwater_pressure_bar': feedwater_pressure_bar
            },
            output_value=h_feedwater,
            output_name="h_feedwater_kj_kg",
            formula="h_feedwater = h(T_fw, P_fw)",
            units="kJ/kg",
            source="IAPWS-IF97"
        )

        # Step 4: Calculate enthalpy difference
        delta_h = h_blowdown - h_feedwater

        tracker.record_step(
            operation="enthalpy_difference",
            description="Calculate enthalpy difference between blowdown and feedwater",
            inputs={'h_blowdown': h_blowdown, 'h_feedwater': h_feedwater},
            output_value=delta_h,
            output_name="delta_h_kj_kg",
            formula="delta_h = h_blowdown - h_feedwater",
            units="kJ/kg"
        )

        # Step 5: Calculate energy loss rate
        # Q = m_dot * delta_h
        # Units: (kg/h) * (kJ/kg) = kJ/h
        # Convert to kW: divide by 3600
        energy_loss_kj_h = m_dot * delta_h
        energy_loss_kw = energy_loss_kj_h / Decimal('3600')

        tracker.record_step(
            operation="energy_loss_calculation",
            description="Calculate blowdown energy loss rate",
            inputs={
                'mass_flow_kg_h': m_dot,
                'delta_h_kj_kg': delta_h
            },
            output_value=energy_loss_kw,
            output_name="energy_loss_kw",
            formula="Q = m_dot * delta_h / 3600",
            units="kW"
        )

        # Step 6: Convert to Btu/h
        energy_loss_btu_h = energy_loss_kw * Decimal('3412.142')

        tracker.record_step(
            operation="unit_conversion",
            description="Convert energy loss to Btu/h",
            inputs={'energy_loss_kw': energy_loss_kw},
            output_value=energy_loss_btu_h,
            output_name="energy_loss_btu_h",
            formula="Q_btu_h = Q_kw * 3412.142",
            units="Btu/h"
        )

        # Step 7: Calculate uncertainty if requested
        uncertainty_kw = None
        if include_uncertainty:
            uncertainty_kw = self._calculate_uncertainty(
                m_dot, delta_h, energy_loss_kw, tracker
            )

        # Get provenance record
        provenance = tracker.get_provenance_record(energy_loss_kw)

        # Apply precision
        energy_loss_kw = energy_loss_kw.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        energy_loss_btu_h = energy_loss_btu_h.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

        return BlowdownEnergyResult(
            energy_loss_rate_kw=energy_loss_kw,
            energy_loss_rate_btu_h=energy_loss_btu_h,
            blowdown_enthalpy_kj_kg=h_blowdown,
            feedwater_enthalpy_kj_kg=h_feedwater,
            enthalpy_difference_kj_kg=delta_h,
            blowdown_mass_flow_kg_h=m_dot,
            saturation_temperature_c=T_sat,
            uncertainty_kw=uncertainty_kw,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_net_energy_loss(
        self,
        gross_energy_loss_kw: float,
        recovered_energy_kw: float,
        tracker: Optional[ProvenanceTracker] = None
    ) -> ThermalResult:
        """
        Calculate net energy loss after heat recovery.

        Formula:
            Net_Loss = Gross_Loss - Recovered_Energy

        Args:
            gross_energy_loss_kw: Gross blowdown energy loss (kW)
            recovered_energy_kw: Energy recovered from heat recovery system (kW)
            tracker: Optional provenance tracker

        Returns:
            ThermalResult with net energy loss
        """
        if tracker is None:
            tracker = ProvenanceTracker(
                calculation_id="net_energy_loss",
                calculation_type="net_energy_loss",
                version=self.version
            )

        tracker.record_inputs({
            'gross_energy_loss_kw': gross_energy_loss_kw,
            'recovered_energy_kw': recovered_energy_kw
        })

        gross = Decimal(str(gross_energy_loss_kw))
        recovered = Decimal(str(recovered_energy_kw))

        # Validate inputs
        if recovered > gross:
            raise ValueError(
                f"Recovered energy ({recovered_energy_kw} kW) cannot exceed "
                f"gross loss ({gross_energy_loss_kw} kW)"
            )

        net_loss = gross - recovered

        tracker.record_step(
            operation="net_loss_calculation",
            description="Calculate net energy loss after heat recovery",
            inputs={'gross_loss': gross, 'recovered': recovered},
            output_value=net_loss,
            output_name="net_energy_loss_kw",
            formula="Net_Loss = Gross_Loss - Recovered_Energy",
            units="kW"
        )

        # Calculate recovery efficiency
        recovery_efficiency = (recovered / gross * Decimal('100')) if gross > 0 else Decimal('0')

        tracker.record_step(
            operation="recovery_efficiency",
            description="Calculate heat recovery efficiency",
            inputs={'recovered': recovered, 'gross': gross},
            output_value=recovery_efficiency,
            output_name="recovery_efficiency_percent",
            formula="Efficiency = Recovered / Gross * 100",
            units="%"
        )

        provenance = tracker.get_provenance_record(net_loss)

        return ThermalResult(
            value=net_loss.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            unit="kW",
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def get_saturated_enthalpy(
        self,
        pressure_bar: float,
        phase: str = "liquid"
    ) -> ThermalResult:
        """
        Get saturated enthalpy at given pressure.

        Public wrapper for IAPWS-IF97 saturated enthalpy calculation.

        Args:
            pressure_bar: Absolute pressure in bar
            phase: "liquid" for h_f or "vapor" for h_g

        Returns:
            ThermalResult with enthalpy value
        """
        tracker = ProvenanceTracker(
            calculation_id=f"sat_enthalpy_{pressure_bar}bar_{phase}",
            calculation_type="saturated_enthalpy",
            version=self.version,
            standard="IAPWS-IF97"
        )

        tracker.record_inputs({
            'pressure_bar': pressure_bar,
            'phase': phase
        })

        if phase == "liquid":
            h = self.iapws.get_saturated_liquid_enthalpy(pressure_bar, tracker)
        elif phase == "vapor":
            h = self.iapws.get_saturated_vapor_enthalpy(pressure_bar, tracker)
        else:
            raise ValueError(f"Invalid phase: {phase}. Must be 'liquid' or 'vapor'")

        provenance = tracker.get_provenance_record(h)

        return ThermalResult(
            value=h,
            unit="kJ/kg",
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def get_subcooled_enthalpy(
        self,
        temperature_c: float,
        pressure_bar: float
    ) -> ThermalResult:
        """
        Get subcooled liquid enthalpy.

        Public wrapper for IAPWS-IF97 subcooled enthalpy calculation.

        Args:
            temperature_c: Temperature in Celsius
            pressure_bar: Absolute pressure in bar

        Returns:
            ThermalResult with enthalpy value
        """
        tracker = ProvenanceTracker(
            calculation_id=f"subcooled_enthalpy_{temperature_c}C_{pressure_bar}bar",
            calculation_type="subcooled_enthalpy",
            version=self.version,
            standard="IAPWS-IF97"
        )

        tracker.record_inputs({
            'temperature_c': temperature_c,
            'pressure_bar': pressure_bar
        })

        h = self.iapws.get_subcooled_enthalpy(temperature_c, pressure_bar, tracker)

        provenance = tracker.get_provenance_record(h)

        return ThermalResult(
            value=h,
            unit="kJ/kg",
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def _calculate_uncertainty(
        self,
        mass_flow: Decimal,
        delta_h: Decimal,
        energy_loss: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate uncertainty in energy loss calculation.

        Uses propagation of uncertainty formula:
        u_Q = Q * sqrt((u_m/m)^2 + (u_h/h)^2)

        Args:
            mass_flow: Mass flow rate (kg/h)
            delta_h: Enthalpy difference (kJ/kg)
            energy_loss: Calculated energy loss (kW)
            tracker: Provenance tracker

        Returns:
            Uncertainty in energy loss (kW)
        """
        u_m_rel = self.uncertainty_factors['mass_flow']
        u_h_rel = self.uncertainty_factors['pressure']  # Enthalpy uncertainty from pressure

        # Propagation of uncertainty (RSS method)
        # u_Q/Q = sqrt((u_m/m)^2 + (u_h/h)^2)
        u_rel_squared = u_m_rel ** 2 + u_h_rel ** 2
        u_rel = Decimal(str(math.sqrt(float(u_rel_squared))))

        uncertainty = energy_loss * u_rel

        tracker.record_step(
            operation="uncertainty_calculation",
            description="Calculate uncertainty using propagation formula",
            inputs={
                'u_mass_flow_rel': u_m_rel,
                'u_enthalpy_rel': u_h_rel,
                'energy_loss_kw': energy_loss
            },
            output_value=uncertainty,
            output_name="uncertainty_kw",
            formula="u_Q = Q * sqrt((u_m/m)^2 + (u_h/h)^2)",
            units="kW"
        )

        return uncertainty.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


# Convenience functions for direct use

def calculate_blowdown_energy_loss(
    blowdown_mass_flow_kg_h: float,
    boiler_pressure_bar: float,
    feedwater_temp_c: float,
    feedwater_pressure_bar: float
) -> BlowdownEnergyResult:
    """
    Calculate blowdown energy loss - convenience function.

    Args:
        blowdown_mass_flow_kg_h: Blowdown mass flow rate (kg/h)
        boiler_pressure_bar: Boiler operating pressure (bar absolute)
        feedwater_temp_c: Feedwater temperature (Celsius)
        feedwater_pressure_bar: Feedwater pressure (bar absolute)

    Returns:
        BlowdownEnergyResult with complete provenance
    """
    calculator = BlowdownThermalCalculator()
    return calculator.calculate_blowdown_energy_loss(
        blowdown_mass_flow_kg_h,
        boiler_pressure_bar,
        feedwater_temp_c,
        feedwater_pressure_bar
    )


def get_saturated_enthalpy(pressure_bar: float, phase: str = "liquid") -> ThermalResult:
    """
    Get saturated enthalpy - convenience function.

    Args:
        pressure_bar: Absolute pressure in bar
        phase: "liquid" or "vapor"

    Returns:
        ThermalResult with enthalpy value
    """
    calculator = BlowdownThermalCalculator()
    return calculator.get_saturated_enthalpy(pressure_bar, phase)


def get_subcooled_enthalpy(temperature_c: float, pressure_bar: float) -> ThermalResult:
    """
    Get subcooled liquid enthalpy - convenience function.

    Args:
        temperature_c: Temperature in Celsius
        pressure_bar: Absolute pressure in bar

    Returns:
        ThermalResult with enthalpy value
    """
    calculator = BlowdownThermalCalculator()
    return calculator.get_subcooled_enthalpy(temperature_c, pressure_bar)
