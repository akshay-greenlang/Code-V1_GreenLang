"""
Unit Tests: Unit Conversion

Tests all unit conversion functions for ThermalCommand agent including:
- Temperature conversions (Celsius, Fahrenheit, Kelvin)
- Pressure conversions (bar, psi, Pa, atm)
- Flow rate conversions (m3/h, L/min, gpm)
- Energy conversions (kWh, MJ, BTU, therm)
- Mass flow conversions (kg/h, lb/h, t/h)

Reference: GL-001 Specification Section 11.1
Target Coverage: 85%+
"""

import pytest
import math
from decimal import Decimal, ROUND_HALF_UP
from typing import Callable


# =============================================================================
# Unit Conversion Classes (Simulated Production Code)
# =============================================================================

class UnitConversionError(Exception):
    """Raised when unit conversion fails."""
    def __init__(self, from_unit: str, to_unit: str, value: float, message: str = ""):
        self.from_unit = from_unit
        self.to_unit = to_unit
        self.value = value
        super().__init__(f"Cannot convert {value} from {from_unit} to {to_unit}: {message}")


class TemperatureConverter:
    """Handles temperature unit conversions."""

    KELVIN_OFFSET = 273.15

    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (celsius * 9 / 5) + 32

    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (fahrenheit - 32) * 5 / 9

    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        kelvin = celsius + TemperatureConverter.KELVIN_OFFSET
        if kelvin < 0:
            raise UnitConversionError('celsius', 'kelvin', celsius, "Result below absolute zero")
        return kelvin

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        if kelvin < 0:
            raise UnitConversionError('kelvin', 'celsius', kelvin, "Kelvin cannot be negative")
        return kelvin - TemperatureConverter.KELVIN_OFFSET

    @staticmethod
    def fahrenheit_to_kelvin(fahrenheit: float) -> float:
        """Convert Fahrenheit to Kelvin."""
        celsius = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        return TemperatureConverter.celsius_to_kelvin(celsius)

    @staticmethod
    def kelvin_to_fahrenheit(kelvin: float) -> float:
        """Convert Kelvin to Fahrenheit."""
        celsius = TemperatureConverter.kelvin_to_celsius(kelvin)
        return TemperatureConverter.celsius_to_fahrenheit(celsius)


class PressureConverter:
    """Handles pressure unit conversions."""

    # Conversion factors (to Pascals)
    BAR_TO_PA = 100000
    PSI_TO_PA = 6894.757
    ATM_TO_PA = 101325
    KPA_TO_PA = 1000
    MPA_TO_PA = 1000000

    @staticmethod
    def bar_to_psi(bar: float) -> float:
        """Convert bar to psi."""
        if bar < 0:
            raise UnitConversionError('bar', 'psi', bar, "Pressure cannot be negative")
        return bar * (PressureConverter.BAR_TO_PA / PressureConverter.PSI_TO_PA)

    @staticmethod
    def psi_to_bar(psi: float) -> float:
        """Convert psi to bar."""
        if psi < 0:
            raise UnitConversionError('psi', 'bar', psi, "Pressure cannot be negative")
        return psi * (PressureConverter.PSI_TO_PA / PressureConverter.BAR_TO_PA)

    @staticmethod
    def bar_to_pa(bar: float) -> float:
        """Convert bar to Pascals."""
        if bar < 0:
            raise UnitConversionError('bar', 'Pa', bar, "Pressure cannot be negative")
        return bar * PressureConverter.BAR_TO_PA

    @staticmethod
    def pa_to_bar(pa: float) -> float:
        """Convert Pascals to bar."""
        if pa < 0:
            raise UnitConversionError('Pa', 'bar', pa, "Pressure cannot be negative")
        return pa / PressureConverter.BAR_TO_PA

    @staticmethod
    def bar_to_atm(bar: float) -> float:
        """Convert bar to atmospheres."""
        if bar < 0:
            raise UnitConversionError('bar', 'atm', bar, "Pressure cannot be negative")
        pa = bar * PressureConverter.BAR_TO_PA
        return pa / PressureConverter.ATM_TO_PA

    @staticmethod
    def atm_to_bar(atm: float) -> float:
        """Convert atmospheres to bar."""
        if atm < 0:
            raise UnitConversionError('atm', 'bar', atm, "Pressure cannot be negative")
        pa = atm * PressureConverter.ATM_TO_PA
        return pa / PressureConverter.BAR_TO_PA

    @staticmethod
    def kpa_to_bar(kpa: float) -> float:
        """Convert kilopascals to bar."""
        if kpa < 0:
            raise UnitConversionError('kPa', 'bar', kpa, "Pressure cannot be negative")
        return (kpa * PressureConverter.KPA_TO_PA) / PressureConverter.BAR_TO_PA


class FlowRateConverter:
    """Handles volumetric flow rate conversions."""

    # Conversion factors (to m3/h)
    L_MIN_TO_M3_H = 0.06
    GPM_TO_M3_H = 0.227124707
    CFM_TO_M3_H = 1.699010796

    @staticmethod
    def m3h_to_lmin(m3h: float) -> float:
        """Convert m3/h to L/min."""
        if m3h < 0:
            raise UnitConversionError('m3/h', 'L/min', m3h, "Flow rate cannot be negative")
        return m3h / FlowRateConverter.L_MIN_TO_M3_H

    @staticmethod
    def lmin_to_m3h(lmin: float) -> float:
        """Convert L/min to m3/h."""
        if lmin < 0:
            raise UnitConversionError('L/min', 'm3/h', lmin, "Flow rate cannot be negative")
        return lmin * FlowRateConverter.L_MIN_TO_M3_H

    @staticmethod
    def m3h_to_gpm(m3h: float) -> float:
        """Convert m3/h to gallons per minute."""
        if m3h < 0:
            raise UnitConversionError('m3/h', 'gpm', m3h, "Flow rate cannot be negative")
        return m3h / FlowRateConverter.GPM_TO_M3_H

    @staticmethod
    def gpm_to_m3h(gpm: float) -> float:
        """Convert gallons per minute to m3/h."""
        if gpm < 0:
            raise UnitConversionError('gpm', 'm3/h', gpm, "Flow rate cannot be negative")
        return gpm * FlowRateConverter.GPM_TO_M3_H

    @staticmethod
    def m3h_to_cfm(m3h: float) -> float:
        """Convert m3/h to cubic feet per minute."""
        if m3h < 0:
            raise UnitConversionError('m3/h', 'cfm', m3h, "Flow rate cannot be negative")
        return m3h / FlowRateConverter.CFM_TO_M3_H

    @staticmethod
    def cfm_to_m3h(cfm: float) -> float:
        """Convert cubic feet per minute to m3/h."""
        if cfm < 0:
            raise UnitConversionError('cfm', 'm3/h', cfm, "Flow rate cannot be negative")
        return cfm * FlowRateConverter.CFM_TO_M3_H


class EnergyConverter:
    """Handles energy unit conversions."""

    # Conversion factors (to Joules)
    KWH_TO_J = 3600000
    MJ_TO_J = 1000000
    BTU_TO_J = 1055.06
    THERM_TO_J = 105506000
    KCAL_TO_J = 4184

    @staticmethod
    def kwh_to_mj(kwh: float) -> float:
        """Convert kWh to MJ."""
        if kwh < 0:
            raise UnitConversionError('kWh', 'MJ', kwh, "Energy cannot be negative")
        joules = kwh * EnergyConverter.KWH_TO_J
        return joules / EnergyConverter.MJ_TO_J

    @staticmethod
    def mj_to_kwh(mj: float) -> float:
        """Convert MJ to kWh."""
        if mj < 0:
            raise UnitConversionError('MJ', 'kWh', mj, "Energy cannot be negative")
        joules = mj * EnergyConverter.MJ_TO_J
        return joules / EnergyConverter.KWH_TO_J

    @staticmethod
    def kwh_to_btu(kwh: float) -> float:
        """Convert kWh to BTU."""
        if kwh < 0:
            raise UnitConversionError('kWh', 'BTU', kwh, "Energy cannot be negative")
        joules = kwh * EnergyConverter.KWH_TO_J
        return joules / EnergyConverter.BTU_TO_J

    @staticmethod
    def btu_to_kwh(btu: float) -> float:
        """Convert BTU to kWh."""
        if btu < 0:
            raise UnitConversionError('BTU', 'kWh', btu, "Energy cannot be negative")
        joules = btu * EnergyConverter.BTU_TO_J
        return joules / EnergyConverter.KWH_TO_J

    @staticmethod
    def kwh_to_therm(kwh: float) -> float:
        """Convert kWh to therms."""
        if kwh < 0:
            raise UnitConversionError('kWh', 'therm', kwh, "Energy cannot be negative")
        joules = kwh * EnergyConverter.KWH_TO_J
        return joules / EnergyConverter.THERM_TO_J

    @staticmethod
    def therm_to_kwh(therm: float) -> float:
        """Convert therms to kWh."""
        if therm < 0:
            raise UnitConversionError('therm', 'kWh', therm, "Energy cannot be negative")
        joules = therm * EnergyConverter.THERM_TO_J
        return joules / EnergyConverter.KWH_TO_J

    @staticmethod
    def mj_to_btu(mj: float) -> float:
        """Convert MJ to BTU."""
        if mj < 0:
            raise UnitConversionError('MJ', 'BTU', mj, "Energy cannot be negative")
        joules = mj * EnergyConverter.MJ_TO_J
        return joules / EnergyConverter.BTU_TO_J


class MassFlowConverter:
    """Handles mass flow rate conversions."""

    # Conversion factors (to kg/h)
    LB_H_TO_KG_H = 0.453592
    T_H_TO_KG_H = 1000
    G_S_TO_KG_H = 3.6

    @staticmethod
    def kgh_to_lbh(kgh: float) -> float:
        """Convert kg/h to lb/h."""
        if kgh < 0:
            raise UnitConversionError('kg/h', 'lb/h', kgh, "Mass flow cannot be negative")
        return kgh / MassFlowConverter.LB_H_TO_KG_H

    @staticmethod
    def lbh_to_kgh(lbh: float) -> float:
        """Convert lb/h to kg/h."""
        if lbh < 0:
            raise UnitConversionError('lb/h', 'kg/h', lbh, "Mass flow cannot be negative")
        return lbh * MassFlowConverter.LB_H_TO_KG_H

    @staticmethod
    def kgh_to_th(kgh: float) -> float:
        """Convert kg/h to tonnes/h."""
        if kgh < 0:
            raise UnitConversionError('kg/h', 't/h', kgh, "Mass flow cannot be negative")
        return kgh / MassFlowConverter.T_H_TO_KG_H

    @staticmethod
    def th_to_kgh(th: float) -> float:
        """Convert tonnes/h to kg/h."""
        if th < 0:
            raise UnitConversionError('t/h', 'kg/h', th, "Mass flow cannot be negative")
        return th * MassFlowConverter.T_H_TO_KG_H

    @staticmethod
    def gs_to_kgh(gs: float) -> float:
        """Convert g/s to kg/h."""
        if gs < 0:
            raise UnitConversionError('g/s', 'kg/h', gs, "Mass flow cannot be negative")
        return gs * MassFlowConverter.G_S_TO_KG_H


# =============================================================================
# Test Classes
# =============================================================================

class TestTemperatureConversion:
    """Test suite for temperature unit conversions."""

    @pytest.mark.parametrize("celsius,expected_fahrenheit", [
        (0, 32),
        (100, 212),
        (-40, -40),  # Same in both scales
        (37, 98.6),  # Body temperature
        (20, 68),    # Room temperature
    ])
    def test_celsius_to_fahrenheit(self, celsius, expected_fahrenheit):
        """Test Celsius to Fahrenheit conversion."""
        result = TemperatureConverter.celsius_to_fahrenheit(celsius)
        assert pytest.approx(result, rel=1e-4) == expected_fahrenheit

    @pytest.mark.parametrize("fahrenheit,expected_celsius", [
        (32, 0),
        (212, 100),
        (-40, -40),
        (98.6, 37),
        (68, 20),
    ])
    def test_fahrenheit_to_celsius(self, fahrenheit, expected_celsius):
        """Test Fahrenheit to Celsius conversion."""
        result = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        assert pytest.approx(result, rel=1e-4) == expected_celsius

    @pytest.mark.parametrize("celsius,expected_kelvin", [
        (0, 273.15),
        (100, 373.15),
        (-273.15, 0),  # Absolute zero
        (25, 298.15),
    ])
    def test_celsius_to_kelvin(self, celsius, expected_kelvin):
        """Test Celsius to Kelvin conversion."""
        result = TemperatureConverter.celsius_to_kelvin(celsius)
        assert pytest.approx(result, rel=1e-6) == expected_kelvin

    def test_celsius_to_kelvin_below_absolute_zero_fails(self):
        """Test that conversion below absolute zero raises error."""
        with pytest.raises(UnitConversionError) as exc_info:
            TemperatureConverter.celsius_to_kelvin(-300)

        assert "absolute zero" in str(exc_info.value).lower()

    @pytest.mark.parametrize("kelvin,expected_celsius", [
        (273.15, 0),
        (373.15, 100),
        (0, -273.15),
        (298.15, 25),
    ])
    def test_kelvin_to_celsius(self, kelvin, expected_celsius):
        """Test Kelvin to Celsius conversion."""
        result = TemperatureConverter.kelvin_to_celsius(kelvin)
        assert pytest.approx(result, rel=1e-6) == expected_celsius

    def test_negative_kelvin_fails(self):
        """Test that negative Kelvin raises error."""
        with pytest.raises(UnitConversionError) as exc_info:
            TemperatureConverter.kelvin_to_celsius(-10)

        assert "negative" in str(exc_info.value).lower()

    def test_round_trip_celsius_fahrenheit(self):
        """Test that Celsius -> Fahrenheit -> Celsius gives original value."""
        original = 25.0
        fahrenheit = TemperatureConverter.celsius_to_fahrenheit(original)
        result = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_celsius_kelvin(self):
        """Test that Celsius -> Kelvin -> Celsius gives original value."""
        original = 100.0
        kelvin = TemperatureConverter.celsius_to_kelvin(original)
        result = TemperatureConverter.kelvin_to_celsius(kelvin)
        assert pytest.approx(result, rel=1e-9) == original

    def test_fahrenheit_to_kelvin(self):
        """Test Fahrenheit to Kelvin conversion."""
        result = TemperatureConverter.fahrenheit_to_kelvin(32)  # 0C = 273.15K
        assert pytest.approx(result, rel=1e-4) == 273.15

    def test_kelvin_to_fahrenheit(self):
        """Test Kelvin to Fahrenheit conversion."""
        result = TemperatureConverter.kelvin_to_fahrenheit(273.15)  # 0C = 32F
        assert pytest.approx(result, rel=1e-4) == 32


class TestPressureConversion:
    """Test suite for pressure unit conversions."""

    @pytest.mark.parametrize("bar,expected_psi", [
        (1, 14.50377),
        (0, 0),
        (10, 145.0377),
        (0.5, 7.251885),
    ])
    def test_bar_to_psi(self, bar, expected_psi):
        """Test bar to psi conversion."""
        result = PressureConverter.bar_to_psi(bar)
        assert pytest.approx(result, rel=1e-4) == expected_psi

    @pytest.mark.parametrize("psi,expected_bar", [
        (14.50377, 1),
        (0, 0),
        (145.0377, 10),
    ])
    def test_psi_to_bar(self, psi, expected_bar):
        """Test psi to bar conversion."""
        result = PressureConverter.psi_to_bar(psi)
        assert pytest.approx(result, rel=1e-4) == expected_bar

    @pytest.mark.parametrize("bar,expected_pa", [
        (1, 100000),
        (0, 0),
        (0.1, 10000),
        (10, 1000000),
    ])
    def test_bar_to_pa(self, bar, expected_pa):
        """Test bar to Pascal conversion."""
        result = PressureConverter.bar_to_pa(bar)
        assert pytest.approx(result, rel=1e-6) == expected_pa

    @pytest.mark.parametrize("bar,expected_atm", [
        (1.01325, 1),
        (0, 0),
        (10.1325, 10),
    ])
    def test_bar_to_atm(self, bar, expected_atm):
        """Test bar to atmosphere conversion."""
        result = PressureConverter.bar_to_atm(bar)
        assert pytest.approx(result, rel=1e-4) == expected_atm

    def test_negative_pressure_fails(self):
        """Test that negative pressure raises error."""
        with pytest.raises(UnitConversionError) as exc_info:
            PressureConverter.bar_to_psi(-1)

        assert "negative" in str(exc_info.value).lower()

    def test_round_trip_bar_psi(self):
        """Test that bar -> psi -> bar gives original value."""
        original = 5.0
        psi = PressureConverter.bar_to_psi(original)
        result = PressureConverter.psi_to_bar(psi)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_bar_atm(self):
        """Test that bar -> atm -> bar gives original value."""
        original = 5.0
        atm = PressureConverter.bar_to_atm(original)
        result = PressureConverter.atm_to_bar(atm)
        assert pytest.approx(result, rel=1e-9) == original

    def test_kpa_to_bar(self):
        """Test kPa to bar conversion."""
        result = PressureConverter.kpa_to_bar(100)
        assert pytest.approx(result, rel=1e-6) == 1.0


class TestFlowRateConversion:
    """Test suite for flow rate unit conversions."""

    @pytest.mark.parametrize("m3h,expected_lmin", [
        (1, 16.6667),
        (0, 0),
        (60, 1000),
        (0.06, 1),
    ])
    def test_m3h_to_lmin(self, m3h, expected_lmin):
        """Test m3/h to L/min conversion."""
        result = FlowRateConverter.m3h_to_lmin(m3h)
        assert pytest.approx(result, rel=1e-3) == expected_lmin

    @pytest.mark.parametrize("lmin,expected_m3h", [
        (16.6667, 1),
        (0, 0),
        (1000, 60),
    ])
    def test_lmin_to_m3h(self, lmin, expected_m3h):
        """Test L/min to m3/h conversion."""
        result = FlowRateConverter.lmin_to_m3h(lmin)
        assert pytest.approx(result, rel=1e-3) == expected_m3h

    @pytest.mark.parametrize("m3h,expected_gpm", [
        (1, 4.40287),
        (0, 0),
        (10, 44.0287),
    ])
    def test_m3h_to_gpm(self, m3h, expected_gpm):
        """Test m3/h to gpm conversion."""
        result = FlowRateConverter.m3h_to_gpm(m3h)
        assert pytest.approx(result, rel=1e-3) == expected_gpm

    def test_negative_flow_rate_fails(self):
        """Test that negative flow rate raises error."""
        with pytest.raises(UnitConversionError):
            FlowRateConverter.m3h_to_lmin(-10)

    def test_round_trip_m3h_lmin(self):
        """Test that m3/h -> L/min -> m3/h gives original value."""
        original = 10.0
        lmin = FlowRateConverter.m3h_to_lmin(original)
        result = FlowRateConverter.lmin_to_m3h(lmin)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_m3h_gpm(self):
        """Test that m3/h -> gpm -> m3/h gives original value."""
        original = 10.0
        gpm = FlowRateConverter.m3h_to_gpm(original)
        result = FlowRateConverter.gpm_to_m3h(gpm)
        assert pytest.approx(result, rel=1e-9) == original

    def test_cfm_conversion(self):
        """Test CFM conversion."""
        m3h = 100.0
        cfm = FlowRateConverter.m3h_to_cfm(m3h)
        back = FlowRateConverter.cfm_to_m3h(cfm)
        assert pytest.approx(back, rel=1e-9) == m3h


class TestEnergyConversion:
    """Test suite for energy unit conversions."""

    @pytest.mark.parametrize("kwh,expected_mj", [
        (1, 3.6),
        (0, 0),
        (10, 36),
        (0.278, 1.0008),  # ~1 MJ
    ])
    def test_kwh_to_mj(self, kwh, expected_mj):
        """Test kWh to MJ conversion."""
        result = EnergyConverter.kwh_to_mj(kwh)
        assert pytest.approx(result, rel=1e-3) == expected_mj

    @pytest.mark.parametrize("mj,expected_kwh", [
        (3.6, 1),
        (0, 0),
        (36, 10),
    ])
    def test_mj_to_kwh(self, mj, expected_kwh):
        """Test MJ to kWh conversion."""
        result = EnergyConverter.mj_to_kwh(mj)
        assert pytest.approx(result, rel=1e-3) == expected_kwh

    @pytest.mark.parametrize("kwh,expected_btu", [
        (1, 3412.14),
        (0, 0),
        (10, 34121.4),
    ])
    def test_kwh_to_btu(self, kwh, expected_btu):
        """Test kWh to BTU conversion."""
        result = EnergyConverter.kwh_to_btu(kwh)
        assert pytest.approx(result, rel=1e-3) == expected_btu

    def test_kwh_to_therm(self):
        """Test kWh to therm conversion."""
        # 1 therm = 29.3 kWh approximately
        result = EnergyConverter.kwh_to_therm(29.3)
        assert pytest.approx(result, rel=1e-2) == 1.0

    def test_negative_energy_fails(self):
        """Test that negative energy raises error."""
        with pytest.raises(UnitConversionError):
            EnergyConverter.kwh_to_mj(-10)

    def test_round_trip_kwh_mj(self):
        """Test that kWh -> MJ -> kWh gives original value."""
        original = 100.0
        mj = EnergyConverter.kwh_to_mj(original)
        result = EnergyConverter.mj_to_kwh(mj)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_kwh_btu(self):
        """Test that kWh -> BTU -> kWh gives original value."""
        original = 100.0
        btu = EnergyConverter.kwh_to_btu(original)
        result = EnergyConverter.btu_to_kwh(btu)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_kwh_therm(self):
        """Test that kWh -> therm -> kWh gives original value."""
        original = 100.0
        therm = EnergyConverter.kwh_to_therm(original)
        result = EnergyConverter.therm_to_kwh(therm)
        assert pytest.approx(result, rel=1e-9) == original

    def test_mj_to_btu(self):
        """Test MJ to BTU conversion."""
        result = EnergyConverter.mj_to_btu(1)
        assert pytest.approx(result, rel=1e-3) == 947.817


class TestMassFlowConversion:
    """Test suite for mass flow unit conversions."""

    @pytest.mark.parametrize("kgh,expected_lbh", [
        (1, 2.20462),
        (0, 0),
        (100, 220.462),
        (0.453592, 1),
    ])
    def test_kgh_to_lbh(self, kgh, expected_lbh):
        """Test kg/h to lb/h conversion."""
        result = MassFlowConverter.kgh_to_lbh(kgh)
        assert pytest.approx(result, rel=1e-3) == expected_lbh

    @pytest.mark.parametrize("lbh,expected_kgh", [
        (2.20462, 1),
        (0, 0),
        (220.462, 100),
    ])
    def test_lbh_to_kgh(self, lbh, expected_kgh):
        """Test lb/h to kg/h conversion."""
        result = MassFlowConverter.lbh_to_kgh(lbh)
        assert pytest.approx(result, rel=1e-3) == expected_kgh

    @pytest.mark.parametrize("kgh,expected_th", [
        (1000, 1),
        (0, 0),
        (500, 0.5),
    ])
    def test_kgh_to_th(self, kgh, expected_th):
        """Test kg/h to t/h conversion."""
        result = MassFlowConverter.kgh_to_th(kgh)
        assert pytest.approx(result, rel=1e-6) == expected_th

    def test_negative_mass_flow_fails(self):
        """Test that negative mass flow raises error."""
        with pytest.raises(UnitConversionError):
            MassFlowConverter.kgh_to_lbh(-10)

    def test_round_trip_kgh_lbh(self):
        """Test that kg/h -> lb/h -> kg/h gives original value."""
        original = 500.0
        lbh = MassFlowConverter.kgh_to_lbh(original)
        result = MassFlowConverter.lbh_to_kgh(lbh)
        assert pytest.approx(result, rel=1e-9) == original

    def test_round_trip_kgh_th(self):
        """Test that kg/h -> t/h -> kg/h gives original value."""
        original = 500.0
        th = MassFlowConverter.kgh_to_th(original)
        result = MassFlowConverter.th_to_kgh(th)
        assert pytest.approx(result, rel=1e-9) == original

    def test_gs_to_kgh(self):
        """Test g/s to kg/h conversion."""
        result = MassFlowConverter.gs_to_kgh(1)  # 1 g/s = 3.6 kg/h
        assert pytest.approx(result, rel=1e-6) == 3.6


class TestConversionEdgeCases:
    """Test edge cases across all converters."""

    def test_zero_values_convert_to_zero(self):
        """Test that zero converts to zero for all units."""
        assert TemperatureConverter.celsius_to_fahrenheit(0) == 32  # Exception: 0C = 32F
        assert PressureConverter.bar_to_psi(0) == 0
        assert FlowRateConverter.m3h_to_lmin(0) == 0
        assert EnergyConverter.kwh_to_mj(0) == 0
        assert MassFlowConverter.kgh_to_lbh(0) == 0

    def test_very_large_values(self):
        """Test that very large values convert correctly."""
        large_value = 1e12

        # Should not raise
        result = EnergyConverter.kwh_to_mj(large_value)
        assert result == pytest.approx(large_value * 3.6, rel=1e-6)

    def test_very_small_values(self):
        """Test that very small values convert correctly."""
        small_value = 1e-12

        result = PressureConverter.bar_to_pa(small_value)
        assert result == pytest.approx(small_value * 100000, rel=1e-6)

    def test_decimal_precision(self):
        """Test that decimal precision is maintained."""
        # Test with high precision value
        value = 123.456789012345
        fahrenheit = TemperatureConverter.celsius_to_fahrenheit(value)
        back = TemperatureConverter.fahrenheit_to_celsius(fahrenheit)

        # Should maintain at least 10 decimal places of precision
        assert abs(back - value) < 1e-10

    def test_floating_point_edge_cases(self):
        """Test floating point edge cases."""
        # Test with values near floating point limits
        result = TemperatureConverter.celsius_to_kelvin(0.0000001)
        assert result > 273.15

        result = PressureConverter.bar_to_psi(0.0000001)
        assert result > 0
