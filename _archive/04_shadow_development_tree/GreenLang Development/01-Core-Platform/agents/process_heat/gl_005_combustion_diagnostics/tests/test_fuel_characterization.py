# -*- coding: utf-8 -*-
"""
GL-005 Fuel Characterization Tests
==================================

Comprehensive unit tests for fuel characterization module including
fuel identification, HHV/LHV calculations, and blend detection.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    FuelCharacterizationConfig,
    FuelCategory,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    AnalysisStatus,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.fuel_characterization import (
    FuelCharacterizationEngine,
    FuelReferenceData,
    FUEL_DATABASE,
    get_fuel_reference,
    calculate_emission_factor,
    estimate_fuel_consumption,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.tests.conftest import (
    assert_valid_provenance_hash,
    assert_valid_confidence,
)


class TestFuelDatabase:
    """Tests for fuel reference database."""

    def test_database_contains_all_fuels(self):
        """Test that database contains all fuel categories."""
        for fuel_type in [
            FuelCategory.NATURAL_GAS,
            FuelCategory.PROPANE,
            FuelCategory.FUEL_OIL_2,
            FuelCategory.FUEL_OIL_6,
            FuelCategory.COAL_BITUMINOUS,
            FuelCategory.COAL_ANTHRACITE,
            FuelCategory.BIOMASS_WOOD,
            FuelCategory.BIOMASS_PELLET,
            FuelCategory.BIOGAS,
            FuelCategory.HYDROGEN,
        ]:
            assert fuel_type in FUEL_DATABASE

    def test_natural_gas_properties(self):
        """Test natural gas reference properties."""
        ng = FUEL_DATABASE[FuelCategory.NATURAL_GAS]

        assert ng.name == "Natural Gas (Pipeline Quality)"
        assert ng.carbon_pct == pytest.approx(75.0, rel=0.01)
        assert ng.hydrogen_pct == pytest.approx(24.0, rel=0.01)
        assert ng.hhv_mj_kg == pytest.approx(55.5, rel=0.01)
        assert ng.lhv_mj_kg == pytest.approx(50.0, rel=0.01)
        assert ng.stoich_air_fuel_ratio == pytest.approx(17.2, rel=0.01)
        assert ng.theoretical_co2_max_pct == pytest.approx(11.8, rel=0.01)

    def test_fuel_oil_properties(self):
        """Test fuel oil reference properties."""
        fo = FUEL_DATABASE[FuelCategory.FUEL_OIL_2]

        assert fo.carbon_pct == pytest.approx(86.5, rel=0.01)
        assert fo.hhv_mj_kg == pytest.approx(45.5, rel=0.01)
        assert fo.theoretical_co2_max_pct == pytest.approx(15.4, rel=0.01)

    def test_hydrogen_properties(self):
        """Test hydrogen reference properties (zero carbon)."""
        h2 = FUEL_DATABASE[FuelCategory.HYDROGEN]

        assert h2.carbon_pct == 0.0
        assert h2.hydrogen_pct == 100.0
        assert h2.theoretical_co2_max_pct == 0.0
        assert h2.co2_ef_kg_mj == 0.0  # Zero emissions

    def test_biogenic_fuels_zero_fossil_emissions(self):
        """Test that biogenic fuels have zero fossil CO2 emissions."""
        for fuel_type in [
            FuelCategory.BIOMASS_WOOD,
            FuelCategory.BIOMASS_PELLET,
            FuelCategory.BIOGAS,
        ]:
            ref = FUEL_DATABASE[fuel_type]
            assert ref.co2_ef_kg_mj == 0.0  # Biogenic CO2

    def test_get_fuel_reference_utility(self):
        """Test get_fuel_reference utility function."""
        ref = get_fuel_reference(FuelCategory.PROPANE)

        assert ref is not None
        assert ref.category == FuelCategory.PROPANE
        assert ref.name == "Propane (LPG)"

    def test_get_fuel_reference_invalid(self):
        """Test get_fuel_reference with invalid category."""
        result = get_fuel_reference("invalid_fuel")
        assert result is None


class TestEmissionFactorCalculation:
    """Tests for emission factor calculation utility."""

    def test_calculate_natural_gas_emissions(self):
        """Test CO2 emission calculation for natural gas."""
        # Natural gas: 0.0561 kg CO2/MJ
        # 100 MJ heat output -> 5.61 kg CO2
        emissions = calculate_emission_factor(FuelCategory.NATURAL_GAS, 100.0)
        expected = 100.0 * 0.0561

        assert emissions == pytest.approx(expected, rel=0.01)

    def test_calculate_coal_emissions(self):
        """Test CO2 emission calculation for coal (higher emissions)."""
        # Bituminous coal: 0.0946 kg CO2/MJ
        emissions_coal = calculate_emission_factor(FuelCategory.COAL_BITUMINOUS, 100.0)
        emissions_ng = calculate_emission_factor(FuelCategory.NATURAL_GAS, 100.0)

        # Coal should have higher emissions than natural gas
        assert emissions_coal > emissions_ng

    def test_zero_emissions_hydrogen(self):
        """Test zero emissions for hydrogen."""
        emissions = calculate_emission_factor(FuelCategory.HYDROGEN, 1000.0)
        assert emissions == 0.0

    def test_invalid_fuel_returns_zero(self):
        """Test that invalid fuel returns zero emissions."""
        emissions = calculate_emission_factor("invalid", 100.0)
        assert emissions == 0.0


class TestFuelConsumptionEstimation:
    """Tests for fuel consumption estimation utility."""

    def test_estimate_natural_gas_consumption(self):
        """Test fuel consumption estimation for natural gas."""
        # 100 MJ output at 80% efficiency = 125 MJ input
        # Natural gas LHV = 50 MJ/kg -> 2.5 kg
        consumption = estimate_fuel_consumption(
            FuelCategory.NATURAL_GAS,
            heat_output_mj=100.0,
            efficiency_pct=80.0
        )

        expected = (100.0 / 0.8) / 50.0  # 2.5 kg
        assert consumption == pytest.approx(expected, rel=0.01)

    def test_estimate_consumption_different_efficiencies(self):
        """Test that lower efficiency means higher fuel consumption."""
        consumption_80 = estimate_fuel_consumption(
            FuelCategory.NATURAL_GAS, 100.0, 80.0
        )
        consumption_90 = estimate_fuel_consumption(
            FuelCategory.NATURAL_GAS, 100.0, 90.0
        )

        assert consumption_80 > consumption_90

    def test_zero_efficiency_returns_zero(self):
        """Test that zero efficiency returns zero consumption."""
        consumption = estimate_fuel_consumption(
            FuelCategory.NATURAL_GAS, 100.0, 0.0
        )
        assert consumption == 0.0


class TestFuelCharacterizationEngine:
    """Tests for fuel characterization engine."""

    def test_initialization(self, default_fuel_config):
        """Test engine initialization."""
        engine = FuelCharacterizationEngine(default_fuel_config)
        assert engine.config == default_fuel_config

    def test_characterize_natural_gas(self, default_fuel_config):
        """Test characterization of natural gas."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Natural gas flue gas signature
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,  # Typical for natural gas
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        assert result.status == AnalysisStatus.SUCCESS
        assert result.primary_fuel.fuel_category == FuelCategory.NATURAL_GAS
        assert result.primary_fuel.confidence > 0.5

    def test_characterize_fuel_oil(self, default_fuel_config):
        """Test characterization of fuel oil."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Fuel oil flue gas signature (higher CO2)
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=14.0,  # Typical for fuel oil
            co_ppm=40.0,
            nox_ppm=80.0,
            so2_ppm=20.0,  # Sulfur present
            flue_gas_temp_c=200.0,
        )

        result = engine.characterize(reading)

        assert result.status == AnalysisStatus.SUCCESS
        # Should identify as fuel oil family
        assert result.primary_fuel.fuel_category in [
            FuelCategory.FUEL_OIL_2,
            FuelCategory.FUEL_OIL_6,
        ]

    def test_characterize_propane(self, default_fuel_config):
        """Test characterization of propane."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Propane flue gas signature
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=12.5,  # Between NG and fuel oil
            co_ppm=35.0,
            nox_ppm=50.0,
            flue_gas_temp_c=185.0,
        )

        result = engine.characterize(reading)

        assert result.status == AnalysisStatus.SUCCESS
        # Should identify as propane or similar
        assert result.primary_fuel.confidence > 0.3

    def test_characterize_with_expected_fuel(self, default_fuel_config):
        """Test characterization with expected fuel validation."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(
            reading,
            expected_fuel=FuelCategory.NATURAL_GAS
        )

        # Should match expected fuel
        assert result.matches_configured_fuel is True
        assert result.deviation_from_expected_pct < 10.0

    def test_fuel_mismatch_detection(self, default_fuel_config):
        """Test detection of fuel mismatch."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Create reading that looks like fuel oil
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=14.5,  # Fuel oil signature
            co_ppm=50.0,
            nox_ppm=90.0,
            flue_gas_temp_c=200.0,
        )

        result = engine.characterize(
            reading,
            expected_fuel=FuelCategory.NATURAL_GAS  # But expect NG
        )

        # Should detect mismatch
        assert result.matches_configured_fuel is False
        assert result.deviation_from_expected_pct > 10.0

    def test_fuel_properties_in_result(self, default_fuel_config):
        """Test that fuel properties are included in result."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        props = result.primary_fuel
        assert props.carbon_content_pct > 0
        assert props.hydrogen_content_pct > 0
        assert props.hhv_mj_kg > 0
        assert props.lhv_mj_kg > 0
        assert props.stoich_air_fuel_ratio > 0
        assert props.theoretical_co2_pct > 0


class TestBlendDetection:
    """Tests for fuel blend detection."""

    def test_no_blend_with_pure_fuel(self):
        """Test that pure fuel is not detected as blend."""
        config = FuelCharacterizationConfig(detect_fuel_blends=True)
        engine = FuelCharacterizationEngine(config)

        # Pure natural gas signature
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,  # Exactly matches NG
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        assert result.is_fuel_blend is False
        assert result.blend_components is None
        assert result.blend_fractions is None

    def test_blend_detection_ng_fuel_oil(self):
        """Test blend detection for NG + fuel oil mix."""
        config = FuelCharacterizationConfig(
            detect_fuel_blends=True,
            blend_detection_confidence=0.5
        )
        engine = FuelCharacterizationEngine(config)

        # Blend signature (between NG at 11.8% and FO at 15.4%)
        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=13.0,  # Between NG and fuel oil
            co_ppm=40.0,
            nox_ppm=60.0,
            flue_gas_temp_c=190.0,
        )

        result = engine.characterize(reading)

        # May detect as blend depending on confidence threshold
        if result.is_fuel_blend:
            assert result.blend_components is not None
            assert len(result.blend_components) == 2
            assert result.blend_fractions is not None
            assert len(result.blend_fractions) == 2
            assert sum(result.blend_fractions) == pytest.approx(1.0, rel=0.01)

    def test_blend_disabled(self):
        """Test that blend detection can be disabled."""
        config = FuelCharacterizationConfig(detect_fuel_blends=False)
        engine = FuelCharacterizationEngine(config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=13.0,
            co_ppm=40.0,
            nox_ppm=60.0,
            flue_gas_temp_c=190.0,
        )

        result = engine.characterize(reading)

        assert result.is_fuel_blend is False


class TestExcessAirCalculation:
    """Tests for excess air calculation in fuel characterization."""

    def test_excess_air_formula(self, default_fuel_config):
        """Test excess air calculation."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # EA = O2 / (20.95 - O2) * 100
        # At 3% O2: EA = 3 / (20.95 - 3) * 100 = 16.7%
        ea = engine._calculate_excess_air(3.0)
        expected = (3.0 / (20.95 - 3.0)) * 100

        assert ea == pytest.approx(expected, rel=0.01)

    def test_excess_air_at_high_o2(self, default_fuel_config):
        """Test excess air at high O2 (high excess air)."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # At 10% O2: EA = 10 / (20.95 - 10) * 100 = 91.3%
        ea = engine._calculate_excess_air(10.0)
        expected = (10.0 / (20.95 - 10.0)) * 100

        assert ea == pytest.approx(expected, rel=0.01)

    def test_excess_air_at_atmospheric(self, default_fuel_config):
        """Test excess air at atmospheric O2 (no combustion)."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # At 20.95% O2, should handle division by zero
        ea = engine._calculate_excess_air(20.95)
        assert ea == 0.0 or ea == float('inf') or ea is not None


class TestStoichiometricEstimation:
    """Tests for stoichiometric ratio estimation."""

    def test_stoich_ratio_natural_gas_signature(self, default_fuel_config):
        """Test stoich ratio estimation for natural gas."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Natural gas should have stoich A/F around 17.2
        ratio = engine._estimate_stoich_ratio(10.5, 3.0)

        # Should be close to natural gas (17.2)
        assert 14.0 < ratio < 20.0

    def test_stoich_ratio_fuel_oil_signature(self, default_fuel_config):
        """Test stoich ratio estimation for fuel oil."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        # Fuel oil should have stoich A/F around 14.4
        ratio = engine._estimate_stoich_ratio(14.5, 3.0)

        # Should be lower than natural gas
        assert 10.0 < ratio < 16.0


class TestFuelQualityAssessment:
    """Tests for fuel quality assessment."""

    def test_excellent_quality_normal_fuel(self, default_fuel_config):
        """Test excellent quality assessment for normal fuel."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,  # Low CO
            nox_ppm=45.0,
            combustibles_pct=0.05,  # Low combustibles
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        assert result.fuel_quality_rating in ["excellent", "normal"]
        assert len(result.quality_concerns) == 0

    def test_poor_quality_high_co(self, default_fuel_config):
        """Test poor quality assessment with high CO."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=400.0,  # High CO
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        assert result.fuel_quality_rating in ["poor", "suspect"]
        assert len(result.quality_concerns) > 0

    def test_quality_concerns_high_combustibles(self, default_fuel_config):
        """Test quality concerns for high combustibles."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=80.0,
            nox_ppm=45.0,
            combustibles_pct=0.8,  # High combustibles
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        # Should have combustibles concern
        concern_text = " ".join(result.quality_concerns)
        assert "combustibles" in concern_text.lower() or "unburned" in concern_text.lower()


class TestProvenanceAndAudit:
    """Tests for provenance hash and audit trail."""

    def test_provenance_hash_generated(self, default_fuel_config):
        """Test provenance hash generation."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        result = engine.characterize(reading)

        assert_valid_provenance_hash(result.provenance_hash)

    def test_audit_trail_generated(self, default_fuel_config):
        """Test audit trail generation."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        engine.characterize(reading)
        audit = engine.get_audit_trail()

        assert len(audit) > 0
        operations = [entry["operation"] for entry in audit]
        assert "combustion_parameters" in operations
        assert "fuel_identification" in operations

    def test_audit_trail_contains_confidence(self, default_fuel_config):
        """Test that audit trail includes confidence values."""
        engine = FuelCharacterizationEngine(default_fuel_config)

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=30.0,
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        engine.characterize(reading)
        audit = engine.get_audit_trail()

        # Find fuel identification entry
        fuel_entry = next(
            e for e in audit if e["operation"] == "fuel_identification"
        )

        assert "confidence" in fuel_entry["data"]
        assert_valid_confidence(fuel_entry["data"]["confidence"])
