# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Energy Mix Engine Tests
=========================================================

Unit tests for EnergyMixEngine (Engine 2) covering unit conversion,
mix calculation, renewable share, energy intensity, purpose breakdown,
completeness validation, and E1-5 data points.

ESRS E1-5: Energy consumption and mix.

Target: 50+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the energy_mix engine module."""
    return _load_engine("energy_mix")


@pytest.fixture
def engine(mod):
    """Create a fresh EnergyMixEngine instance."""
    return mod.EnergyMixEngine()


@pytest.fixture
def solar_entry(mod):
    """Create a solar energy consumption entry."""
    return mod.EnergyConsumptionEntry(
        source=mod.EnergySource.SOLAR,
        amount=Decimal("500"),
        unit=mod.EnergyUnit.MWH,
        purpose=mod.EnergyPurpose.ELECTRICITY,
        is_self_generated=True,
    )


@pytest.fixture
def gas_entry(mod):
    """Create a natural gas energy consumption entry."""
    return mod.EnergyConsumptionEntry(
        source=mod.EnergySource.NATURAL_GAS,
        amount=Decimal("1000"),
        unit=mod.EnergyUnit.MWH,
        purpose=mod.EnergyPurpose.HEATING,
    )


@pytest.fixture
def wind_entry(mod):
    """Create a wind energy consumption entry."""
    return mod.EnergyConsumptionEntry(
        source=mod.EnergySource.WIND,
        amount=Decimal("300"),
        unit=mod.EnergyUnit.MWH,
        purpose=mod.EnergyPurpose.ELECTRICITY,
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestEnergyEnums:
    """Tests for energy mix enums."""

    def test_energy_source_count(self, mod):
        """EnergySource has at least 12 values."""
        assert len(mod.EnergySource) >= 12

    def test_energy_category_count(self, mod):
        """EnergyCategory has 3 values."""
        assert len(mod.EnergyCategory) == 3
        values = {m.value for m in mod.EnergyCategory}
        assert values == {"fossil", "nuclear", "renewable"}

    def test_energy_unit_count(self, mod):
        """EnergyUnit has at least 4 values."""
        assert len(mod.EnergyUnit) >= 4
        values = {m.value for m in mod.EnergyUnit}
        assert {"mwh", "gj", "kwh", "tj"}.issubset(values)

    def test_energy_purpose_values(self, mod):
        """EnergyPurpose has standard categories."""
        values = {m.value for m in mod.EnergyPurpose}
        assert "heating" in values
        assert "cooling" in values
        assert "electricity" in values
        assert "transport" in values

    def test_energy_source_includes_renewables(self, mod):
        """EnergySource includes renewable sources."""
        values = {m.value for m in mod.EnergySource}
        for src in ["solar", "wind", "hydro", "geothermal", "biomass"]:
            assert src in values


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestEnergyConstants:
    """Tests for energy mix constants."""

    def test_conversion_mwh_to_gj(self, mod):
        """1 MWh = 3.6 GJ."""
        assert mod.ENERGY_CONVERSION_FACTORS["mwh"]["gj"] == Decimal("3.6")

    def test_conversion_gj_to_mwh(self, mod):
        """1 GJ = 0.277778 MWh."""
        gj_to_mwh = mod.ENERGY_CONVERSION_FACTORS["gj"]["mwh"]
        assert float(gj_to_mwh) == pytest.approx(0.277778, abs=0.001)

    def test_conversion_identity(self, mod):
        """Identity conversion (MWh to MWh) = 1."""
        assert mod.ENERGY_CONVERSION_FACTORS["mwh"]["mwh"] == Decimal("1")

    def test_source_classification_all_sources_mapped(self, mod):
        """SOURCE_CLASSIFICATION maps all EnergySource values."""
        for source in mod.EnergySource:
            assert source.value in mod.SOURCE_CLASSIFICATION

    def test_source_classification_categories(self, mod):
        """SOURCE_CLASSIFICATION uses valid EnergyCategory values."""
        for cat in mod.SOURCE_CLASSIFICATION.values():
            assert cat in list(mod.EnergyCategory)

    def test_e1_5_datapoints_exist(self, mod):
        """E1_5_DATAPOINTS is a non-empty list."""
        assert len(mod.E1_5_DATAPOINTS) >= 10


# ===========================================================================
# Unit Conversion Tests
# ===========================================================================


class TestUnitConversion:
    """Tests for energy unit conversion."""

    def test_mwh_to_gj(self, engine, mod):
        """Convert MWh to GJ (multiply by 3.6)."""
        result = engine.convert_units(
            Decimal("100"), mod.EnergyUnit.MWH, mod.EnergyUnit.GJ
        )
        assert float(result) == pytest.approx(360.0, abs=0.1)

    def test_gj_to_mwh(self, engine, mod):
        """Convert GJ to MWh."""
        result = engine.convert_units(
            Decimal("360"), mod.EnergyUnit.GJ, mod.EnergyUnit.MWH
        )
        assert float(result) == pytest.approx(100.0, abs=0.1)

    def test_identity_conversion(self, engine, mod):
        """MWh to MWh returns same value."""
        result = engine.convert_units(
            Decimal("42"), mod.EnergyUnit.MWH, mod.EnergyUnit.MWH
        )
        assert float(result) == pytest.approx(42.0, abs=0.001)

    def test_kwh_to_mwh(self, engine, mod):
        """Convert kWh to MWh (divide by 1000)."""
        result = engine.convert_units(
            Decimal("5000"), mod.EnergyUnit.KWH, mod.EnergyUnit.MWH
        )
        assert float(result) == pytest.approx(5.0, abs=0.01)

    def test_tj_to_mwh(self, engine, mod):
        """Convert TJ to MWh (multiply by 277.778)."""
        result = engine.convert_units(
            Decimal("1"), mod.EnergyUnit.TJ, mod.EnergyUnit.MWH
        )
        assert float(result) == pytest.approx(277.778, abs=0.1)

    def test_zero_amount_returns_zero(self, engine, mod):
        """Zero amount conversion returns zero."""
        result = engine.convert_units(
            Decimal("0"), mod.EnergyUnit.GJ, mod.EnergyUnit.MWH
        )
        assert result == Decimal("0")


# ===========================================================================
# Calculate Mix Tests
# ===========================================================================


class TestCalculateMix:
    """Tests for calculate_mix method."""

    def test_basic_mix(self, engine, gas_entry):
        """Basic mix with single fossil entry."""
        result = engine.calculate_mix([gas_entry])
        assert result.total_mwh > Decimal("0")
        assert result.fossil_mwh > Decimal("0")
        assert result.entry_count == 1

    def test_renewable_plus_fossil(self, engine, solar_entry, gas_entry):
        """Mix with both renewable and fossil sources."""
        result = engine.calculate_mix([solar_entry, gas_entry])
        assert result.total_mwh == result.fossil_mwh + result.nuclear_mwh + result.renewable_mwh
        assert result.renewable_mwh > Decimal("0")
        assert result.fossil_mwh > Decimal("0")

    def test_renewable_share_calculation(self, engine, solar_entry, gas_entry):
        """Renewable share is calculated correctly."""
        result = engine.calculate_mix([solar_entry, gas_entry])
        # solar=500, gas=1000, total=1500, renewable_share=500/1500=33.33%
        expected_share = Decimal("500") / Decimal("1500") * Decimal("100")
        assert float(result.renewable_share_pct) == pytest.approx(
            float(expected_share), abs=0.5
        )

    def test_single_source_mix(self, engine, solar_entry):
        """Single renewable source gives 100% renewable share."""
        result = engine.calculate_mix([solar_entry])
        assert float(result.renewable_share_pct) == pytest.approx(100.0, abs=0.1)

    def test_provenance_hash(self, engine, gas_entry):
        """Mix result has a provenance hash."""
        result = engine.calculate_mix([gas_entry])
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_empty_entries_raises(self, engine):
        """Empty entries list raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_mix([])

    def test_by_source_populated(self, engine, solar_entry, gas_entry):
        """by_source dict is populated with source values."""
        result = engine.calculate_mix([solar_entry, gas_entry])
        assert len(result.by_source) >= 2

    def test_self_generated_tracked(self, engine, solar_entry):
        """Self-generated energy is tracked separately."""
        result = engine.calculate_mix([solar_entry])
        assert result.self_generated_mwh > Decimal("0")

    def test_gj_input_converted_to_mwh(self, engine, mod):
        """Entries in GJ are converted to MWh."""
        entry = mod.EnergyConsumptionEntry(
            source=mod.EnergySource.NATURAL_GAS,
            amount=Decimal("360"),
            unit=mod.EnergyUnit.GJ,
            purpose=mod.EnergyPurpose.HEATING,
        )
        result = engine.calculate_mix([entry])
        # 360 GJ * 0.277778 = ~100 MWh
        assert float(result.total_mwh) == pytest.approx(100.0, abs=1.0)


# ===========================================================================
# Renewable Share Tests
# ===========================================================================


class TestRenewableShare:
    """Tests for renewable share calculation."""

    def test_100_pct_renewable(self, engine, solar_entry, wind_entry):
        """All-renewable mix gives 100% share."""
        result = engine.calculate_mix([solar_entry, wind_entry])
        assert float(result.renewable_share_pct) == pytest.approx(100.0, abs=0.1)

    def test_0_pct_renewable(self, engine, gas_entry):
        """All-fossil mix gives 0% renewable share."""
        result = engine.calculate_mix([gas_entry])
        assert float(result.renewable_share_pct) == pytest.approx(0.0, abs=0.1)

    def test_mixed_share(self, engine, mod):
        """50/50 fossil-renewable gives ~50% share."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("500"),
                unit=mod.EnergyUnit.MWH,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NATURAL_GAS,
                amount=Decimal("500"),
                unit=mod.EnergyUnit.MWH,
            ),
        ]
        result = engine.calculate_mix(entries)
        assert float(result.renewable_share_pct) == pytest.approx(50.0, abs=0.5)


# ===========================================================================
# Energy Intensity Tests
# ===========================================================================


class TestEnergyIntensity:
    """Tests for energy intensity calculation."""

    def test_revenue_intensity(self, engine):
        """Calculate energy intensity per revenue."""
        result = engine.calculate_energy_intensity(
            total_mwh=Decimal("15000"),
            denominator_value=Decimal("100"),
            denominator_unit="EUR_million",
        )
        # 15000 / 100 = 150 MWh/EUR_million
        assert float(result.intensity_value) == pytest.approx(150.0, abs=1.0)

    def test_per_sqm_intensity(self, engine):
        """Calculate energy intensity per square meter."""
        result = engine.calculate_energy_intensity(
            total_mwh=Decimal("1000"),
            denominator_value=Decimal("5000"),
            denominator_unit="sqm",
        )
        # 1000 / 5000 = 0.2 MWh/sqm
        assert float(result.intensity_value) == pytest.approx(0.2, abs=0.01)

    def test_zero_denominator_raises(self, engine):
        """Zero denominator raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_energy_intensity(
                total_mwh=Decimal("1000"),
                denominator_value=Decimal("0"),
                denominator_unit="EUR_million",
            )


# ===========================================================================
# Breakdown By Purpose Tests
# ===========================================================================


class TestBreakdownByPurpose:
    """Tests for energy breakdown by purpose."""

    def test_purpose_breakdown(self, engine, mod):
        """Energy is broken down by purpose."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NATURAL_GAS,
                amount=Decimal("500"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.HEATING,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("300"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.ELECTRICITY,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.DISTRICT_COOLING,
                amount=Decimal("100"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.COOLING,
            ),
        ]
        result = engine.calculate_mix(entries)
        assert len(result.by_purpose) >= 2
        assert "heating" in result.by_purpose or "HEATING" in str(result.by_purpose)

    def test_single_purpose(self, engine, gas_entry):
        """Single purpose entry."""
        result = engine.calculate_mix([gas_entry])
        assert len(result.by_purpose) >= 1


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-5 completeness validation."""

    def test_full_passes(self, engine, mod):
        """A comprehensive dataset has high completeness."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NATURAL_GAS,
                amount=Decimal("1000"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.HEATING,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("500"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.ELECTRICITY,
                is_self_generated=True,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NUCLEAR,
                amount=Decimal("200"),
                unit=mod.EnergyUnit.MWH,
                purpose=mod.EnergyPurpose.ELECTRICITY,
            ),
        ]
        result = engine.calculate_mix(entries, reporting_year=2025)
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_minimal_dataset(self, engine, gas_entry):
        """Minimal dataset still returns completeness result."""
        result = engine.calculate_mix([gas_entry])
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-5 Data Points Tests
# ===========================================================================


class TestE15Datapoints:
    """Tests for E1-5 required data point extraction."""

    def test_returns_datapoints(self, engine, solar_entry, gas_entry):
        """get_e1_5_datapoints returns required data points."""
        result = engine.calculate_mix([solar_entry, gas_entry])
        datapoints = engine.get_e1_5_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 8

    def test_total_energy_in_datapoints(self, engine, gas_entry):
        """Total energy consumption is present in data points."""
        result = engine.calculate_mix([gas_entry])
        datapoints = engine.get_e1_5_datapoints(result)
        has_total = any("total" in k.lower() for k in datapoints.keys())
        assert has_total

    def test_renewable_share_in_datapoints(self, engine, solar_entry, gas_entry):
        """Renewable share is present in data points."""
        result = engine.calculate_mix([solar_entry, gas_entry])
        datapoints = engine.get_e1_5_datapoints(result)
        has_renewable = any("renewable" in k.lower() for k in datapoints.keys())
        assert has_renewable


# ===========================================================================
# Nuclear Classification Tests
# ===========================================================================


class TestNuclearClassification:
    """Tests for nuclear energy source classification."""

    def test_nuclear_is_nuclear_category(self, mod):
        """Nuclear source is classified as nuclear category."""
        assert mod.SOURCE_CLASSIFICATION["nuclear"] == mod.EnergyCategory.NUCLEAR

    def test_nuclear_entry_tracked(self, engine, mod):
        """Nuclear energy appears in nuclear_mwh in mix result."""
        entry = mod.EnergyConsumptionEntry(
            source=mod.EnergySource.NUCLEAR,
            amount=Decimal("400"),
            unit=mod.EnergyUnit.MWH,
            purpose=mod.EnergyPurpose.ELECTRICITY,
        )
        result = engine.calculate_mix([entry])
        assert result.nuclear_mwh == Decimal("400")

    def test_nuclear_not_counted_as_renewable(self, engine, mod):
        """Nuclear energy does not count toward renewable share."""
        nuclear = mod.EnergyConsumptionEntry(
            source=mod.EnergySource.NUCLEAR,
            amount=Decimal("500"),
            unit=mod.EnergyUnit.MWH,
        )
        result = engine.calculate_mix([nuclear])
        assert float(result.renewable_share_pct) == pytest.approx(0.0, abs=0.1)


# ===========================================================================
# Source Classification Tests
# ===========================================================================


class TestSourceClassification:
    """Tests for energy source classification into categories."""

    def test_fossil_sources_classified(self, mod):
        """Fossil sources (coal, diesel, natural_gas) are classified as fossil."""
        for src in ["coal", "diesel", "natural_gas"]:
            assert mod.SOURCE_CLASSIFICATION[src] == mod.EnergyCategory.FOSSIL

    def test_renewable_sources_classified(self, mod):
        """Renewable sources are classified as renewable."""
        for src in ["solar", "wind", "hydro", "geothermal", "biomass"]:
            assert mod.SOURCE_CLASSIFICATION[src] == mod.EnergyCategory.RENEWABLE

    def test_classify_source_method(self, engine, mod):
        """classify_source returns the correct EnergyCategory."""
        assert engine.classify_source(mod.EnergySource.SOLAR) == mod.EnergyCategory.RENEWABLE
        assert engine.classify_source(mod.EnergySource.COAL) == mod.EnergyCategory.FOSSIL
        assert engine.classify_source(mod.EnergySource.NUCLEAR) == mod.EnergyCategory.NUCLEAR


# ===========================================================================
# Advanced Mix Calculation Tests
# ===========================================================================


class TestAdvancedMix:
    """Additional tests for complex mix calculations."""

    def test_three_category_mix(self, engine, mod):
        """Mix with fossil, nuclear, and renewable sums correctly."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NATURAL_GAS,
                amount=Decimal("400"),
                unit=mod.EnergyUnit.MWH,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.NUCLEAR,
                amount=Decimal("300"),
                unit=mod.EnergyUnit.MWH,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("300"),
                unit=mod.EnergyUnit.MWH,
            ),
        ]
        result = engine.calculate_mix(entries)
        assert float(result.total_mwh) == pytest.approx(1000.0, abs=0.1)
        assert float(result.fossil_mwh) == pytest.approx(400.0, abs=0.1)
        assert float(result.nuclear_mwh) == pytest.approx(300.0, abs=0.1)
        assert float(result.renewable_mwh) == pytest.approx(300.0, abs=0.1)

    def test_large_mix_scenario(self, engine, mod):
        """Mix with many entries processes correctly."""
        entries = []
        for i in range(20):
            entries.append(mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR if i % 2 == 0 else mod.EnergySource.NATURAL_GAS,
                amount=Decimal("100"),
                unit=mod.EnergyUnit.MWH,
            ))
        result = engine.calculate_mix(entries)
        assert float(result.total_mwh) == pytest.approx(2000.0, abs=0.1)
        assert result.entry_count == 20

    def test_mix_provenance_is_hex(self, engine, mod):
        """Mix result provenance hash is valid hex."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.WIND,
                amount=Decimal("250"),
                unit=mod.EnergyUnit.MWH,
            ),
        ]
        r1 = engine.calculate_mix(entries)
        assert len(r1.provenance_hash) == 64
        int(r1.provenance_hash, 16)  # Valid hex

    def test_kwh_entries_summed_correctly(self, engine, mod):
        """Multiple kWh entries are converted and summed."""
        entries = [
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("2000"),
                unit=mod.EnergyUnit.KWH,
            ),
            mod.EnergyConsumptionEntry(
                source=mod.EnergySource.SOLAR,
                amount=Decimal("3000"),
                unit=mod.EnergyUnit.KWH,
            ),
        ]
        result = engine.calculate_mix(entries)
        # 2000 + 3000 = 5000 kWh = 5 MWh
        assert float(result.total_mwh) == pytest.approx(5.0, abs=0.1)


# ===========================================================================
# Renewable Share From Values Tests
# ===========================================================================


class TestRenewableShareFromValues:
    """Tests for calculate_renewable_share_from_values helper."""

    def test_100_pct_from_values(self, engine):
        """100% renewable from raw values."""
        result = engine.calculate_renewable_share_from_values(
            renewable_mwh=Decimal("500"),
            total_mwh=Decimal("500"),
        )
        assert float(result) == pytest.approx(100.0, abs=0.1)

    def test_50_pct_from_values(self, engine):
        """50% renewable from raw values."""
        result = engine.calculate_renewable_share_from_values(
            renewable_mwh=Decimal("250"),
            total_mwh=Decimal("500"),
        )
        assert float(result) == pytest.approx(50.0, abs=0.1)

    def test_zero_total_returns_zero(self, engine):
        """Zero total MWh returns 0% renewable share."""
        result = engine.calculate_renewable_share_from_values(
            renewable_mwh=Decimal("100"),
            total_mwh=Decimal("0"),
        )
        assert float(result) == pytest.approx(0.0, abs=0.1)
