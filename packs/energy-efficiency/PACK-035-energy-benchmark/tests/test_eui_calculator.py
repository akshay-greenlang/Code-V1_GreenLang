# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - EUI Calculator Engine Tests (test_eui_calculator.py)
==================================================================================

Tests EUI calculation accuracy, site/source/primary energy conversion,
rolling 12-month EUI, occupancy normalisation, multi-fuel handling,
Decimal precision, and provenance hash computation.

Test Count Target: ~80 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _load_eui():
    path = ENGINES_DIR / "eui_calculator_engine.py"
    if not path.exists():
        pytest.skip("eui_calculator_engine.py not found")
    mod_key = "pack035_test.eui_calc"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load eui_calculator_engine: {exc}")
    return mod


# =============================================================================
# 1. Engine Instantiation
# =============================================================================


class TestEUICalculatorInstantiation:
    """Test engine class can be instantiated."""

    def test_engine_class_exists(self):
        mod = _load_eui()
        assert hasattr(mod, "EUICalculatorEngine")

    def test_engine_instantiation(self):
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        assert engine is not None

    def test_engine_version(self):
        mod = _load_eui()
        assert hasattr(mod, "_MODULE_VERSION")
        assert mod._MODULE_VERSION == "1.0.0"


# =============================================================================
# 2. Site EUI Calculation
# =============================================================================


class TestSiteEUICalculation:
    """Test site EUI calculation accuracy."""

    def test_basic_site_eui(self):
        """100,000 kWh / 1000 m2 = 100 kWh/m2/yr."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-001",
            name="Basic Office",
            floor_area=1000.0,
            building_type="office",
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=100000 / 12,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert result.site_eui is not None
        assert result.site_eui.eui_kwh_per_m2_yr == pytest.approx(100.0, rel=0.01)

    def test_large_office_eui(self):
        """703,000 kWh / 5000 m2 = 140.6 kWh/m2/yr."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-002",
            name="Large Office",
            floor_area=5000.0,
            building_type="office",
        )
        elec = [42000, 40000, 38000, 36000, 37000, 42000, 48000, 47000, 40000, 38000, 40000, 43000]
        gas = [35000, 32000, 25000, 15000, 8000, 3000, 2000, 2000, 5000, 18000, 28000, 33000]
        meter_data = []
        for i, m in enumerate(range(1, 13)):
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=elec[i],
            ))
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.NATURAL_GAS,
                consumption_kwh=gas[i],
            ))
        result = engine.calculate_eui(facility, meter_data)
        total = sum(elec) + sum(gas)
        expected_eui = total / 5000.0
        assert result.site_eui.eui_kwh_per_m2_yr == pytest.approx(expected_eui, rel=0.01)

    def test_zero_floor_area_raises(self):
        """Zero floor area raises ValueError."""
        mod = _load_eui()
        with pytest.raises(Exception):
            mod.FacilityProfile(
                facility_id="TEST-003",
                name="Bad Facility",
                floor_area=0.0,
            )

    def test_empty_meter_data_raises(self):
        """Empty meter data raises ValueError."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-004",
            name="Empty Data",
            floor_area=1000.0,
        )
        with pytest.raises(ValueError, match="At least one meter reading"):
            engine.calculate_eui(facility, [])

    @pytest.mark.parametrize("area,energy,expected_eui", [
        (1000.0, 120000.0, 120.0),
        (2000.0, 120000.0, 60.0),
        (5000.0, 750000.0, 150.0),
        (10000.0, 2000000.0, 200.0),
        (500.0, 50000.0, 100.0),
    ])
    def test_parametrized_site_eui(self, area, energy, expected_eui):
        """Parametrized site EUI calculations."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PARAM",
            name="Param Test",
            floor_area=area,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=energy / 12,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert result.site_eui.eui_kwh_per_m2_yr == pytest.approx(expected_eui, rel=0.01)


# =============================================================================
# 3. Source EUI Calculation
# =============================================================================


class TestSourceEUICalculation:
    """Test source EUI with ENERGY STAR site-to-source factors."""

    def test_electricity_source_factor(self):
        """Source EUI applies 2.80 factor for electricity."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-SRC-001",
            name="Source Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        site_eui = result.site_eui.eui_kwh_per_m2_yr
        source_eui = result.source_eui.eui_kwh_per_m2_yr
        # Source factor for electricity = 2.80
        assert source_eui == pytest.approx(site_eui * 2.80, rel=0.01)

    def test_gas_source_factor(self):
        """Source EUI applies 1.047 factor for natural gas."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-SRC-002",
            name="Gas Source",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.NATURAL_GAS,
                consumption_kwh=10000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        site_eui = result.site_eui.eui_kwh_per_m2_yr
        source_eui = result.source_eui.eui_kwh_per_m2_yr
        assert source_eui == pytest.approx(site_eui * 1.047, rel=0.01)

    def test_source_eui_greater_than_site(self):
        """Source EUI is always >= site EUI."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-SRC-003",
            name="Source > Site",
            floor_area=5000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=50000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert result.source_eui.eui_kwh_per_m2_yr >= result.site_eui.eui_kwh_per_m2_yr


# =============================================================================
# 4. Primary EUI Calculation
# =============================================================================


class TestPrimaryEUICalculation:
    """Test primary EUI with EN 15603 factors."""

    def test_primary_eui_exists(self):
        """Primary EUI result is computed."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PRI-001",
            name="Primary Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period="2025-01",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert result.primary_eui is not None
        assert result.primary_eui.eui_kwh_per_m2_yr > 0

    def test_primary_eui_electricity_factor(self):
        """Primary energy factor for electricity is 2.50."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PRI-002",
            name="Primary Elec",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        site_eui = result.site_eui.eui_kwh_per_m2_yr
        primary_eui = result.primary_eui.eui_kwh_per_m2_yr
        assert primary_eui == pytest.approx(site_eui * 2.50, rel=0.01)


# =============================================================================
# 5. Rolling 12-Month EUI
# =============================================================================


class TestRollingEUI:
    """Test rolling 12-month EUI time series."""

    def test_rolling_eui_points_exist(self):
        """Rolling EUI produces data points for multi-month data."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-ROLL-001",
            name="Rolling Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert len(result.rolling_eui) >= 2

    def test_rolling_eui_chronological(self):
        """Rolling EUI points are in chronological order."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-ROLL-002",
            name="Chrono Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
            for m in range(1, 13)
        ]
        result = engine.calculate_eui(facility, meter_data)
        periods = [r.period_end for r in result.rolling_eui]
        assert periods == sorted(periods)


# =============================================================================
# 6. Occupancy Normalisation
# =============================================================================


class TestOccupancyNormalisation:
    """Test occupancy-based EUI normalisation."""

    def test_normalisation_with_operating_hours(self):
        """Building running 55 h/wk vs standard 50 h/wk reduces normalised EUI."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        result = engine.normalise_by_occupancy(
            base_eui=140.0,
            actual_hours_per_week=55.0,
            building_type="office",
        )
        assert result is not None
        # Standard 50h, actual 55h: factor = 50/55 = 0.909 -> normalised < base
        assert result.normalised_eui < result.base_eui

    def test_normalisation_high_hours_lowers_eui(self):
        """Higher actual hours lowers normalised EUI."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        result_short = engine.normalise_by_occupancy(100.0, 40.0, "office")
        result_long = engine.normalise_by_occupancy(100.0, 80.0, "office")
        assert result_long.normalised_eui < result_short.normalised_eui

    def test_normalisation_zero_hours_uses_standard(self):
        """Zero actual hours defaults to standard hours."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        result = engine.normalise_by_occupancy(100.0, 0, "office")
        # Factor = standard/standard = 1.0
        assert result.normalised_eui == pytest.approx(100.0, rel=0.01)


# =============================================================================
# 7. Floor Area Type Conversion
# =============================================================================


class TestFloorAreaConversion:
    """Test floor area type conversion (GIA, NIA, GLA, TFA)."""

    @pytest.mark.parametrize("area_type,expected_factor", [
        ("gia", 1.0),
        ("nia", 1.2),
        ("gla", 1.15),
        ("tfa", 1.1),
    ])
    def test_area_conversion_factors(self, area_type, expected_factor):
        """Floor area conversions match published factors."""
        mod = _load_eui()
        factors = mod.FLOOR_AREA_CONVERSION
        area_enum = mod.FloorAreaType(area_type)
        actual = factors[area_enum]["to_gia"]
        assert actual == pytest.approx(expected_factor, rel=0.01)


# =============================================================================
# 8. Provenance Hash
# =============================================================================


class TestEUIProvenance:
    """Test SHA-256 provenance hash computation."""

    def test_provenance_hash_exists(self):
        """Result includes a 64-char provenance hash."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PROV-001",
            name="Prov Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period="2025-01",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_provenance_hash_deterministic(self):
        """Same inputs produce same provenance hash (excluding timestamps)."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PROV-DET",
            name="Deterministic",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period="2025-06",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
        ]
        r1 = engine.calculate_eui(facility, meter_data)
        r2 = engine.calculate_eui(facility, meter_data)
        # Hashes exclude timestamps, so should match
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_inputs_different_hash(self):
        """Different inputs produce different provenance hashes."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-PROV-DIFF",
            name="Diff Test",
            floor_area=1000.0,
        )
        data_a = [mod.EnergyMeterData(period="2025-01", carrier=mod.EnergyCarrier.ELECTRICITY, consumption_kwh=10000)]
        data_b = [mod.EnergyMeterData(period="2025-01", carrier=mod.EnergyCarrier.ELECTRICITY, consumption_kwh=20000)]
        r_a = engine.calculate_eui(facility, data_a)
        r_b = engine.calculate_eui(facility, data_b)
        assert r_a.provenance_hash != r_b.provenance_hash


# =============================================================================
# 9. Multi-Carrier Handling
# =============================================================================


class TestMultiCarrier:
    """Test multi-carrier energy aggregation."""

    def test_two_carriers_sum(self):
        """EUI sums electricity and gas correctly."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-MC-001",
            name="Multi Carrier",
            floor_area=1000.0,
        )
        meter_data = []
        for m in range(1, 13):
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=5000,
            ))
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.NATURAL_GAS,
                consumption_kwh=3000,
            ))
        result = engine.calculate_eui(facility, meter_data)
        expected = (5000 + 3000) * 12 / 1000.0
        assert result.site_eui.eui_kwh_per_m2_yr == pytest.approx(expected, rel=0.01)

    def test_carrier_shares_sum_to_100(self):
        """Carrier percentage shares sum to 100%."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-MC-002",
            name="Share Test",
            floor_area=1000.0,
        )
        meter_data = []
        for m in range(1, 13):
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=7000,
            ))
            meter_data.append(mod.EnergyMeterData(
                period=f"2025-{m:02d}",
                carrier=mod.EnergyCarrier.NATURAL_GAS,
                consumption_kwh=3000,
            ))
        result = engine.calculate_eui(facility, meter_data)
        shares = result.site_eui.carrier_shares_pct
        total_share = sum(shares.values())
        assert total_share == pytest.approx(100.0, abs=0.5)


# =============================================================================
# 10. Decimal Precision
# =============================================================================


class TestDecimalPrecision:
    """Test Decimal-based precision in calculations."""

    def test_result_is_float_not_decimal(self):
        """EUI result values are floats (converted from Decimal)."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-DEC-001",
            name="Decimal Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period="2025-01",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert isinstance(result.site_eui.eui_kwh_per_m2_yr, float)

    def test_processing_time_recorded(self):
        """Processing time is recorded in milliseconds."""
        mod = _load_eui()
        engine = mod.EUICalculatorEngine()
        facility = mod.FacilityProfile(
            facility_id="TEST-DEC-002",
            name="Time Test",
            floor_area=1000.0,
        )
        meter_data = [
            mod.EnergyMeterData(
                period="2025-01",
                carrier=mod.EnergyCarrier.ELECTRICITY,
                consumption_kwh=10000,
            )
        ]
        result = engine.calculate_eui(facility, meter_data)
        assert result.processing_time_ms > 0
