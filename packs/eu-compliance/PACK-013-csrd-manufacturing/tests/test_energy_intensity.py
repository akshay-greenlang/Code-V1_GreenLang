# -*- coding: utf-8 -*-
"""
Unit tests for EnergyIntensityEngine - PACK-013 CSRD Manufacturing Engine 2

Tests all methods of EnergyIntensityEngine with 85%+ coverage.
Validates SEC calculation, energy mix analysis, EED compliance tiers,
BAT benchmarking, decarbonization opportunities, and provenance.

Target: 38+ tests across 9 test classes.
"""

import importlib.util
import os
import sys
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines"
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


ei = _load_module("energy_intensity_engine", "energy_intensity_engine.py")

EnergyIntensityEngine = ei.EnergyIntensityEngine
EnergyIntensityConfig = ei.EnergyIntensityConfig
EnergySource = ei.EnergySource
ProductionUnit = ei.ProductionUnit
EEDTier = ei.EEDTier
FacilityEnergyData = ei.FacilityEnergyData
EnergyConsumptionData = ei.EnergyConsumptionData
ProductionVolumeData = ei.ProductionVolumeData
BenchmarkComparison = ei.BenchmarkComparison
EEDCompliance = ei.EEDCompliance
DecarbonizationOpportunity = ei.DecarbonizationOpportunity
EnergyIntensityResult = ei.EnergyIntensityResult
BAT_ENERGY_BENCHMARKS = ei.BAT_ENERGY_BENCHMARKS
DECARBONIZATION_TECHNOLOGIES = ei.DECARBONIZATION_TECHNOLOGIES
_round3 = ei._round3
_round2 = ei._round2
_mwh_to_tj = ei._mwh_to_tj
_mwh_to_mj = ei._mwh_to_mj
_compute_hash = ei._compute_hash
_safe_divide = ei._safe_divide


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_engine():
    """Create an EnergyIntensityEngine with default configuration."""
    return EnergyIntensityEngine()


@pytest.fixture
def sample_facility_energy():
    """Create a sample facility with electricity, gas, and biomass."""
    return FacilityEnergyData(
        facility_id="fac-energy-001",
        facility_name="Test Manufacturing Plant",
        sub_sector="cement",
        energy_consumption=[
            EnergyConsumptionData(
                source=EnergySource.ELECTRICITY,
                quantity_mwh=25000.0,
                cost_eur=3000000.0,
                renewable_pct=30.0,
                emission_factor_tco2_per_mwh=0.4,
            ),
            EnergyConsumptionData(
                source=EnergySource.NATURAL_GAS,
                quantity_mwh=50000.0,
                cost_eur=2500000.0,
                renewable_pct=0.0,
                emission_factor_tco2_per_mwh=0.2,
            ),
            EnergyConsumptionData(
                source=EnergySource.BIOMASS,
                quantity_mwh=10000.0,
                cost_eur=800000.0,
                renewable_pct=100.0,
                emission_factor_tco2_per_mwh=0.0,
            ),
        ],
        production_volumes=[
            ProductionVolumeData(
                product_name="Clinker",
                volume=500000.0,
                unit=ProductionUnit.TONNES,
            ),
        ],
        annual_revenue_eur=150000000.0,
    )


@pytest.fixture
def cement_energy_data():
    """Create cement-specific energy data for benchmarking."""
    return FacilityEnergyData(
        facility_name="Cement Plant A",
        sub_sector="cement",
        energy_consumption=[
            EnergyConsumptionData(
                source=EnergySource.NATURAL_GAS,
                quantity_mwh=400000.0,
                cost_eur=20000000.0,
                renewable_pct=0.0,
                emission_factor_tco2_per_mwh=0.2,
            ),
        ],
        production_volumes=[
            ProductionVolumeData(
                product_name="Clinker",
                volume=500000.0,
                unit=ProductionUnit.TONNES,
            ),
        ],
    )


@pytest.fixture
def automotive_energy_data():
    """Create automotive-specific energy data."""
    return FacilityEnergyData(
        facility_name="Auto Assembly",
        sub_sector="automotive",
        energy_consumption=[
            EnergyConsumptionData(
                source=EnergySource.ELECTRICITY,
                quantity_mwh=30000.0,
                cost_eur=4500000.0,
                renewable_pct=50.0,
                emission_factor_tco2_per_mwh=0.35,
            ),
            EnergyConsumptionData(
                source=EnergySource.NATURAL_GAS,
                quantity_mwh=15000.0,
                cost_eur=1200000.0,
                renewable_pct=0.0,
                emission_factor_tco2_per_mwh=0.2,
            ),
        ],
        production_volumes=[
            ProductionVolumeData(
                product_name="Vehicles",
                volume=50000.0,
                unit=ProductionUnit.UNITS,
            ),
        ],
        annual_revenue_eur=500000000.0,
    )


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test engine initialization with various config types."""

    def test_default(self):
        """Engine initializes with default EnergyIntensityConfig."""
        engine = EnergyIntensityEngine()
        assert engine.config is not None
        assert isinstance(engine.config, EnergyIntensityConfig)
        assert engine.config.reporting_year == 2025
        assert engine.config.include_benchmark is True
        assert engine.engine_version == "1.0.0"

    def test_with_config(self):
        """Engine initializes with an explicit EnergyIntensityConfig."""
        config = EnergyIntensityConfig(
            reporting_year=2024,
            include_eed_compliance=False,
            iso50001_certified=True,
        )
        engine = EnergyIntensityEngine(config)
        assert engine.config.reporting_year == 2024
        assert engine.config.include_eed_compliance is False
        assert engine.config.iso50001_certified is True

    def test_with_dict(self):
        """Engine initializes from a plain dictionary."""
        engine = EnergyIntensityEngine({
            "reporting_year": 2023,
            "include_benchmark": False,
        })
        assert engine.config.reporting_year == 2023
        assert engine.config.include_benchmark is False

    def test_with_none(self):
        """Engine initializes with None (defaults applied)."""
        engine = EnergyIntensityEngine(None)
        assert engine.config.reporting_year == 2025
        assert engine.config.production_unit == ProductionUnit.TONNES


class TestEnergySources:
    """Validate energy source enums and constants."""

    def test_all_sources_defined(self):
        """EnergySource enum has exactly 10 members."""
        assert len(EnergySource) == 10

    def test_energy_source_enum_values(self):
        """Key energy sources are present."""
        assert EnergySource.ELECTRICITY.value == "electricity"
        assert EnergySource.NATURAL_GAS.value == "natural_gas"
        assert EnergySource.HYDROGEN.value == "hydrogen"
        assert EnergySource.SOLAR.value == "solar"
        assert EnergySource.WIND.value == "wind"

    def test_fuel_emission_factors(self):
        """BAT benchmarks exist for key sectors."""
        assert "cement" in BAT_ENERGY_BENCHMARKS
        assert "steel_bof" in BAT_ENERGY_BENCHMARKS
        assert "aluminum" in BAT_ENERGY_BENCHMARKS
        assert "glass_flat" in BAT_ENERGY_BENCHMARKS

    def test_renewable_sources(self):
        """Solar and wind are valid energy sources."""
        assert EnergySource.SOLAR in EnergySource
        assert EnergySource.WIND in EnergySource
        assert EnergySource.BIOMASS in EnergySource


class TestSECCalculation:
    """Test Specific Energy Consumption calculations."""

    def test_sec_basic(self, default_engine):
        """SEC = energy_MJ / production_volume."""
        sec = default_engine.calculate_sec(1000.0, 500.0)
        expected = _mwh_to_mj(1000.0) / 500.0
        assert sec == pytest.approx(expected, rel=1e-6)
        assert sec == pytest.approx(7200.0, rel=1e-6)

    def test_sec_per_unit(self, default_engine):
        """SEC with unit-based production."""
        sec = default_engine.calculate_sec(500.0, 10000.0, ProductionUnit.UNITS)
        expected = _mwh_to_mj(500.0) / 10000.0
        assert sec == pytest.approx(expected, rel=1e-6)

    def test_sec_per_revenue(self, default_engine, sample_facility_energy):
        """Revenue-based energy intensity is computed."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert result.sec_mj_per_eur_revenue > 0.0

    def test_zero_production_handled(self, default_engine):
        """SEC with zero production returns 0.0 (safe divide)."""
        sec = default_engine.calculate_sec(1000.0, 0.0)
        assert sec == 0.0

    def test_multiple_products(self, default_engine):
        """Facility with multiple products computes SEC for each."""
        facility = FacilityEnergyData(
            facility_name="Multi-Product Plant",
            sub_sector="food_beverage",
            energy_consumption=[
                EnergyConsumptionData(
                    source=EnergySource.ELECTRICITY,
                    quantity_mwh=10000.0,
                    emission_factor_tco2_per_mwh=0.4,
                ),
            ],
            production_volumes=[
                ProductionVolumeData(
                    product_name="Product A",
                    volume=20000.0,
                    unit=ProductionUnit.TONNES,
                ),
                ProductionVolumeData(
                    product_name="Product B",
                    volume=10000.0,
                    unit=ProductionUnit.TONNES,
                ),
            ],
        )
        result = default_engine.calculate_energy_intensity(facility)
        assert "Product A" in result.sec_mj_per_unit
        assert "Product B" in result.sec_mj_per_unit
        assert len(result.sec_mj_per_unit) == 2

    def test_sec_precision(self, default_engine):
        """SEC values are rounded to 3 decimal places."""
        facility = FacilityEnergyData(
            facility_name="Precision Test",
            sub_sector="",
            energy_consumption=[
                EnergyConsumptionData(
                    source=EnergySource.ELECTRICITY,
                    quantity_mwh=1234.567,
                    emission_factor_tco2_per_mwh=0.4,
                ),
            ],
            production_volumes=[
                ProductionVolumeData(
                    product_name="Widget",
                    volume=1000.0,
                    unit=ProductionUnit.UNITS,
                ),
            ],
        )
        result = default_engine.calculate_energy_intensity(facility)
        sec_val = result.sec_mj_per_unit["Widget"]
        assert sec_val == _round3(sec_val)


class TestEnergyMix:
    """Test energy mix breakdown calculation."""

    def test_mix_breakdown(self, default_engine, sample_facility_energy):
        """Energy mix breakdown contains all sources."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert "electricity" in result.energy_mix_breakdown
        assert "natural_gas" in result.energy_mix_breakdown
        assert "biomass" in result.energy_mix_breakdown

    def test_renewable_share(self, default_engine, sample_facility_energy):
        """Renewable share calculated correctly."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        total_mwh = 25000.0 + 50000.0 + 10000.0
        expected_renewable_pct = (7500.0 + 10000.0) / total_mwh * 100.0
        assert result.renewable_share_pct == pytest.approx(
            _round2(expected_renewable_pct), abs=0.1
        )

    def test_total_energy_mwh(self, default_engine, sample_facility_energy):
        """Total energy MWh sums correctly."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        expected = 25000.0 + 50000.0 + 10000.0
        assert result.total_energy_mwh == pytest.approx(expected, rel=1e-3)

    def test_energy_cost(self, default_engine, sample_facility_energy):
        """Energy cost is reported per source."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        elec_data = result.energy_mix_breakdown["electricity"]
        assert elec_data["cost_eur"] == pytest.approx(3000000.0, rel=1e-3)


class TestEEDCompliance:
    """Test Energy Efficiency Directive compliance tiers."""

    def test_below_10tj_no_audit(self, default_engine):
        """Below 10 TJ: no audit required."""
        result = default_engine.assess_eed_compliance(5.0)
        assert result.tier == EEDTier.BELOW_10TJ
        assert result.audit_required is False
        assert result.iso50001_required is False
        assert result.compliant is True

    def test_audit_required_10_85tj(self, default_engine):
        """10-85 TJ: energy audit required every 4 years."""
        result = default_engine.assess_eed_compliance(50.0)
        assert result.tier == EEDTier.AUDIT_REQUIRED_10_85TJ
        assert result.audit_required is True
        assert result.iso50001_required is False

    def test_iso50001_required_above_85tj(self, default_engine):
        """Above 85 TJ: ISO 50001 required."""
        result = default_engine.assess_eed_compliance(100.0)
        assert result.tier == EEDTier.ISO50001_REQUIRED_ABOVE_85TJ
        assert result.audit_required is True
        assert result.iso50001_required is True

    def test_eed_tier_enum(self):
        """EEDTier enum has 3 values."""
        assert len(EEDTier) == 3
        assert EEDTier.BELOW_10TJ.value == "below_10tj"
        assert EEDTier.AUDIT_REQUIRED_10_85TJ.value == "audit_required_10_85tj"
        assert EEDTier.ISO50001_REQUIRED_ABOVE_85TJ.value == "iso50001_required_above_85tj"

    def test_eed_compliance_result(self, default_engine, sample_facility_energy):
        """EED compliance appears in full facility result."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert result.eed_compliance is not None
        assert result.eed_compliance.total_energy_tj > 0.0


class TestBenchmarkComparison:
    """Test BAT-AEL benchmark comparison."""

    def test_within_bat_ael(self, default_engine):
        """Facility at or below BAT is classified as at_or_below_bat."""
        bm = default_engine.compare_benchmark(2500.0, "cement")
        assert bm is not None
        assert bm.status == "at_or_below_bat"
        assert bm.percentile_rank == 95.0

    def test_above_bat_ael(self, default_engine):
        """Facility above sector average is classified as above_sector_average."""
        bm = default_engine.compare_benchmark(5000.0, "cement")
        assert bm is not None
        assert bm.status == "above_sector_average"
        assert bm.percentile_rank < 50.0

    def test_below_bat_ael(self, default_engine):
        """Facility between BAT and average is classified correctly."""
        bm = default_engine.compare_benchmark(3200.0, "cement")
        assert bm is not None
        assert bm.status == "between_bat_and_average"
        assert 50.0 < bm.percentile_rank < 95.0

    def test_percentile_ranking(self, default_engine):
        """Percentile ranking is within valid range [0, 100]."""
        bm = default_engine.compare_benchmark(4000.0, "steel_bof")
        assert bm is not None
        assert 0.0 <= bm.percentile_rank <= 100.0


class TestDecarbonization:
    """Test decarbonization opportunity identification."""

    def test_opportunities_identified(self, default_engine, sample_facility_energy):
        """Decarbonization opportunities are identified for cement."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert len(result.decarbonization_opportunities) > 0

    def test_technology_trl(self, default_engine, sample_facility_energy):
        """All technologies have TRL between 1 and 9."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        for opp in result.decarbonization_opportunities:
            assert 1 <= opp.trl <= 9

    def test_investment_payback(self, default_engine, sample_facility_energy):
        """All opportunities have positive payback period."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        for opp in result.decarbonization_opportunities:
            assert opp.payback_years > 0.0
            assert opp.investment_eur >= 0.0

    def test_co2_reduction(self, default_engine, sample_facility_energy):
        """All opportunities have non-negative CO2 reduction."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        for opp in result.decarbonization_opportunities:
            assert opp.co2_reduction_tonnes >= 0.0


class TestProvenance:
    """Test provenance hash generation and determinism."""

    def test_hash_64char(self, default_engine, sample_facility_energy):
        """Provenance hash is a 64-character hex string (SHA-256)."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_deterministic(self):
        """Same data produces the same hash."""
        data = {"energy": 1000.0, "production": 500.0}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_different_input_different_hash(self):
        """Different data produces different hashes."""
        h1 = _compute_hash({"energy": 1000.0})
        h2 = _compute_hash({"energy": 2000.0})
        assert h1 != h2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_large_facility(self, default_engine):
        """Large facility with high energy does not crash."""
        facility = FacilityEnergyData(
            facility_name="Mega Plant",
            sub_sector="steel_bof",
            energy_consumption=[
                EnergyConsumptionData(
                    source=EnergySource.ELECTRICITY,
                    quantity_mwh=5000000.0,
                    emission_factor_tco2_per_mwh=0.35,
                ),
                EnergyConsumptionData(
                    source=EnergySource.NATURAL_GAS,
                    quantity_mwh=10000000.0,
                    emission_factor_tco2_per_mwh=0.2,
                ),
            ],
            production_volumes=[
                ProductionVolumeData(
                    product_name="Steel",
                    volume=2000000.0,
                    unit=ProductionUnit.TONNES,
                ),
            ],
        )
        result = default_engine.calculate_energy_intensity(facility)
        assert result.total_energy_mwh > 0.0
        assert result.processing_time_ms >= 0.0

    def test_zero_energy(self):
        """Facility with no energy consumption raises ValueError."""
        engine = EnergyIntensityEngine()
        facility = FacilityEnergyData(
            facility_name="Empty",
            energy_consumption=[],
        )
        with pytest.raises(ValueError, match="no energy"):
            engine.calculate_energy_intensity(facility)

    def test_result_fields(self, default_engine, sample_facility_energy):
        """EnergyIntensityResult contains all expected fields."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        assert hasattr(result, "result_id")
        assert hasattr(result, "facility_id")
        assert hasattr(result, "total_energy_mwh")
        assert hasattr(result, "total_energy_tj")
        assert hasattr(result, "sec_mj_per_unit")
        assert hasattr(result, "sec_mj_per_eur_revenue")
        assert hasattr(result, "energy_mix_breakdown")
        assert hasattr(result, "renewable_share_pct")
        assert hasattr(result, "benchmark_comparison")
        assert hasattr(result, "eed_compliance")
        assert hasattr(result, "iso50001_status")
        assert hasattr(result, "decarbonization_opportunities")
        assert hasattr(result, "methodology_notes")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "engine_version")
        assert hasattr(result, "calculated_at")
        assert hasattr(result, "provenance_hash")

    def test_methodology_notes(self, default_engine, sample_facility_energy):
        """Result includes methodology notes with key information."""
        result = default_engine.calculate_energy_intensity(sample_facility_energy)
        notes_text = " ".join(result.methodology_notes)
        assert "Reporting year" in notes_text
        assert "Engine version" in notes_text
        assert "Total energy" in notes_text
        assert "Renewable share" in notes_text
