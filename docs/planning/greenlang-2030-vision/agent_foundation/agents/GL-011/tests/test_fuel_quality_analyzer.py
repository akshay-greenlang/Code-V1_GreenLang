# -*- coding: utf-8 -*-
"""
Unit tests for FuelQualityAnalyzer.

Tests the comprehensive fuel quality analysis calculator including:
- Proximate analysis interpretation
- Ultimate analysis calculations
- Dulong/Boie heating value estimation
- Hardgrove Grindability Index correlation
- Slagging/fouling indices
- Fuel-specific CO2 emission factors
- Quality grade classification (ASTM)
- Sampling frequency optimization

Standards tested: ASTM D3172, ASTM D3176, ASTM D5865, ASTM D388
"""

import pytest
import sys
from pathlib import Path
from decimal import Decimal
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculators.fuel_quality_analyzer import (
    FuelQualityAnalyzer,
    ProximateAnalysis,
    UltimateAnalysis,
    AshComposition,
    FuelSampleInput,
    FuelQualityResult,
    HeatingValueResult,
    SlaggingFoulingResult,
    EmissionFactorResult,
    QualityDeviationAlert,
    SamplingRecommendation,
    FuelType,
    AnalysisBasis,
    QualityGrade,
    SlaggingRisk
)


class TestFuelQualityAnalyzer:
    """Test suite for FuelQualityAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> FuelQualityAnalyzer:
        """Create analyzer instance with default config."""
        return FuelQualityAnalyzer()

    @pytest.fixture
    def analyzer_boie(self) -> FuelQualityAnalyzer:
        """Create analyzer instance using Boie formula."""
        return FuelQualityAnalyzer(config={'heating_value_method': 'boie'})

    @pytest.fixture
    def bituminous_coal_proximate(self) -> ProximateAnalysis:
        """Typical bituminous coal proximate analysis."""
        return ProximateAnalysis(
            moisture_percent=Decimal("8.5"),
            volatile_matter_percent=Decimal("28.0"),
            fixed_carbon_percent=Decimal("52.0"),
            ash_percent=Decimal("11.5"),
            basis=AnalysisBasis.AS_RECEIVED
        )

    @pytest.fixture
    def bituminous_coal_ultimate(self) -> UltimateAnalysis:
        """Typical bituminous coal ultimate analysis (dry basis)."""
        return UltimateAnalysis(
            carbon_percent=Decimal("72.0"),
            hydrogen_percent=Decimal("4.8"),
            oxygen_percent=Decimal("7.5"),
            nitrogen_percent=Decimal("1.5"),
            sulfur_percent=Decimal("1.7"),
            ash_percent=Decimal("12.5"),
            basis=AnalysisBasis.DRY_BASIS
        )

    @pytest.fixture
    def ash_composition_high_slagging(self) -> AshComposition:
        """Ash composition with high slagging potential."""
        return AshComposition(
            sio2_percent=Decimal("45.0"),
            al2o3_percent=Decimal("20.0"),
            fe2o3_percent=Decimal("15.0"),
            cao_percent=Decimal("8.0"),
            mgo_percent=Decimal("2.0"),
            na2o_percent=Decimal("3.0"),
            k2o_percent=Decimal("2.0"),
            tio2_percent=Decimal("1.0"),
            so3_percent=Decimal("4.0")
        )

    @pytest.fixture
    def ash_composition_low_slagging(self) -> AshComposition:
        """Ash composition with low slagging potential."""
        return AshComposition(
            sio2_percent=Decimal("60.0"),
            al2o3_percent=Decimal("25.0"),
            fe2o3_percent=Decimal("5.0"),
            cao_percent=Decimal("3.0"),
            mgo_percent=Decimal("1.0"),
            na2o_percent=Decimal("0.5"),
            k2o_percent=Decimal("0.5"),
            tio2_percent=Decimal("2.0"),
            so3_percent=Decimal("3.0")
        )

    @pytest.fixture
    def bituminous_sample(
        self,
        bituminous_coal_proximate: ProximateAnalysis,
        bituminous_coal_ultimate: UltimateAnalysis
    ) -> FuelSampleInput:
        """Complete bituminous coal sample."""
        return FuelSampleInput(
            sample_id="COAL-2024-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ultimate=bituminous_coal_ultimate,
            sampling_date="2024-01-15",
            lot_number="LOT-A-2024",
            supplier="Quality Coal Co."
        )

    @pytest.fixture
    def biomass_proximate(self) -> ProximateAnalysis:
        """Typical biomass (wood pellets) proximate analysis."""
        return ProximateAnalysis(
            moisture_percent=Decimal("8.0"),
            volatile_matter_percent=Decimal("75.0"),
            fixed_carbon_percent=Decimal("16.0"),
            ash_percent=Decimal("1.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

    @pytest.fixture
    def biomass_sample(self, biomass_proximate: ProximateAnalysis) -> FuelSampleInput:
        """Biomass sample without ultimate analysis."""
        return FuelSampleInput(
            sample_id="BIOMASS-2024-001",
            fuel_type=FuelType.WOOD_PELLETS,
            proximate=biomass_proximate
        )

    @pytest.fixture
    def lignite_proximate(self) -> ProximateAnalysis:
        """Typical lignite proximate analysis."""
        return ProximateAnalysis(
            moisture_percent=Decimal("35.0"),
            volatile_matter_percent=Decimal("28.0"),
            fixed_carbon_percent=Decimal("25.0"),
            ash_percent=Decimal("12.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

    @pytest.fixture
    def lignite_sample(self, lignite_proximate: ProximateAnalysis) -> FuelSampleInput:
        """Lignite sample with high moisture."""
        return FuelSampleInput(
            sample_id="LIGNITE-2024-001",
            fuel_type=FuelType.LIGNITE,
            proximate=lignite_proximate
        )

    # ==========================================================================
    # Basic Functionality Tests
    # ==========================================================================

    def test_basic_analysis(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test basic fuel quality analysis completes."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result is not None
        assert isinstance(result, FuelQualityResult)
        assert result.sample_id == "COAL-2024-001"

    def test_analysis_without_ultimate(
        self,
        analyzer: FuelQualityAnalyzer,
        biomass_sample: FuelSampleInput
    ):
        """Test analysis works without ultimate analysis (estimated)."""
        result = analyzer.analyze_fuel_quality(biomass_sample)

        assert result is not None
        assert result.heating_value.hhv_mj_kg > Decimal("0")

    def test_analysis_with_ash_composition(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        bituminous_coal_ultimate: UltimateAnalysis,
        ash_composition_high_slagging: AshComposition
    ):
        """Test analysis with ash composition for slagging analysis."""
        sample = FuelSampleInput(
            sample_id="COAL-ASH-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ultimate=bituminous_coal_ultimate,
            ash_composition=ash_composition_high_slagging
        )

        result = analyzer.analyze_fuel_quality(sample)

        assert result.slagging_fouling is not None
        assert result.slagging_fouling.base_acid_ratio > Decimal("0")

    # ==========================================================================
    # Proximate Analysis Tests
    # ==========================================================================

    def test_dry_basis_conversion(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test conversion to dry basis is correct."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # Dry basis should have zero moisture
        assert result.proximate_dry_basis.moisture_percent == Decimal("0")

        # Dry basis VM should be higher than AR basis
        original_vm = bituminous_sample.proximate.volatile_matter_percent
        dry_vm = result.proximate_dry_basis.volatile_matter_percent
        assert dry_vm > original_vm

    def test_fuel_ratio_calculation(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test fuel ratio (FC/VM) calculation."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # Bituminous coal typically has fuel ratio > 1
        assert result.fuel_ratio > Decimal("1.0")
        assert result.fuel_ratio < Decimal("5.0")  # Not anthracite level

    def test_proximate_validation_total_100(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test proximate analysis validation (must sum to ~100%)."""
        invalid_proximate = ProximateAnalysis(
            moisture_percent=Decimal("10.0"),
            volatile_matter_percent=Decimal("20.0"),
            fixed_carbon_percent=Decimal("30.0"),
            ash_percent=Decimal("10.0"),  # Total = 70%, should fail
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="INVALID-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=invalid_proximate
        )

        with pytest.raises(ValueError, match="sum to"):
            analyzer.analyze_fuel_quality(sample)

    def test_negative_values_rejected(self, analyzer: FuelQualityAnalyzer):
        """Test negative values are rejected."""
        invalid_proximate = ProximateAnalysis(
            moisture_percent=Decimal("-5.0"),  # Invalid
            volatile_matter_percent=Decimal("35.0"),
            fixed_carbon_percent=Decimal("55.0"),
            ash_percent=Decimal("15.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="INVALID-002",
            fuel_type=FuelType.BITUMINOUS,
            proximate=invalid_proximate
        )

        with pytest.raises(ValueError):
            analyzer.analyze_fuel_quality(sample)

    # ==========================================================================
    # Heating Value Tests
    # ==========================================================================

    def test_dulong_hhv_calculation(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test Dulong formula HHV calculation."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # Bituminous coal typically 24-32 MJ/kg (dry basis)
        assert Decimal("20") < result.heating_value.hhv_mj_kg < Decimal("35")
        assert result.heating_value.calculation_method == "Dulong" or \
               "Dulong" in result.heating_value.calculation_method

    def test_boie_hhv_calculation(
        self,
        analyzer_boie: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test Boie formula HHV calculation."""
        result = analyzer_boie.analyze_fuel_quality(bituminous_sample)

        # Should use Boie method
        assert "Boie" in result.heating_value.calculation_method
        assert Decimal("20") < result.heating_value.hhv_mj_kg < Decimal("35")

    def test_lhv_less_than_hhv(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test LHV is always less than HHV."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.heating_value.lhv_mj_kg < result.heating_value.hhv_mj_kg

    def test_hhv_varies_by_fuel_type(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput,
        lignite_sample: FuelSampleInput,
        biomass_sample: FuelSampleInput
    ):
        """Test HHV varies appropriately by fuel type."""
        bit_result = analyzer.analyze_fuel_quality(bituminous_sample)
        lig_result = analyzer.analyze_fuel_quality(lignite_sample)
        bio_result = analyzer.analyze_fuel_quality(biomass_sample)

        # Bituminous should have highest HHV
        assert bit_result.heating_value.hhv_mj_kg > lig_result.heating_value.hhv_mj_kg

    def test_measured_hhv_used_when_provided(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        bituminous_coal_ultimate: UltimateAnalysis
    ):
        """Test measured HHV is used when provided."""
        measured_hhv = Decimal("29.5")
        sample = FuelSampleInput(
            sample_id="MEASURED-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ultimate=bituminous_coal_ultimate,
            heating_value_mj_kg=measured_hhv
        )

        result = analyzer.analyze_fuel_quality(sample)

        assert result.heating_value.hhv_mj_kg == measured_hhv
        assert "Measured" in result.heating_value.calculation_method

    # ==========================================================================
    # Slagging/Fouling Tests
    # ==========================================================================

    def test_base_acid_ratio_calculation(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        ash_composition_high_slagging: AshComposition
    ):
        """Test Base/Acid ratio calculation."""
        sample = FuelSampleInput(
            sample_id="SLAG-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ash_composition=ash_composition_high_slagging
        )

        result = analyzer.analyze_fuel_quality(sample)

        # B/A = (Fe2O3+CaO+MgO+Na2O+K2O)/(SiO2+Al2O3+TiO2)
        # = (15+8+2+3+2)/(45+20+1) = 30/66 = 0.454
        assert result.slagging_fouling is not None
        assert Decimal("0.4") < result.slagging_fouling.base_acid_ratio < Decimal("0.5")

    def test_slagging_risk_classification(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        ash_composition_high_slagging: AshComposition,
        ash_composition_low_slagging: AshComposition
    ):
        """Test slagging risk is classified correctly."""
        high_slag_sample = FuelSampleInput(
            sample_id="HIGH-SLAG-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ash_composition=ash_composition_high_slagging
        )

        low_slag_sample = FuelSampleInput(
            sample_id="LOW-SLAG-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ash_composition=ash_composition_low_slagging
        )

        high_result = analyzer.analyze_fuel_quality(high_slag_sample)
        low_result = analyzer.analyze_fuel_quality(low_slag_sample)

        # Low B/A ratio should have lower slagging risk
        assert low_result.slagging_fouling.slagging_risk.value <= high_result.slagging_fouling.slagging_risk.value

    def test_t250_temperature_estimated(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        ash_composition_high_slagging: AshComposition
    ):
        """Test T250 temperature is estimated."""
        sample = FuelSampleInput(
            sample_id="T250-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ash_composition=ash_composition_high_slagging
        )

        result = analyzer.analyze_fuel_quality(sample)

        assert result.slagging_fouling.t250_temperature_c is not None
        assert result.slagging_fouling.t250_temperature_c > Decimal("900")

    def test_recommended_actions_for_high_slagging(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis,
        ash_composition_high_slagging: AshComposition
    ):
        """Test recommended actions are provided for high slagging."""
        sample = FuelSampleInput(
            sample_id="ACTIONS-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            ash_composition=ash_composition_high_slagging
        )

        result = analyzer.analyze_fuel_quality(sample)

        # Should have recommended actions
        assert len(result.slagging_fouling.recommended_actions) > 0

    # ==========================================================================
    # Emission Factor Tests
    # ==========================================================================

    def test_co2_emission_factor_calculation(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test CO2 emission factor calculation."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # CO2 factor should be positive
        assert result.emission_factors.co2_kg_per_gj > Decimal("0")
        assert result.emission_factors.co2_kg_per_kg_fuel > Decimal("0")

        # Typical coal: 90-100 kg CO2/GJ
        assert Decimal("80") < result.emission_factors.co2_kg_per_gj < Decimal("110")

    def test_so2_emission_factor_calculation(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test SO2 emission factor calculation."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # SO2 factor depends on sulfur content
        assert result.emission_factors.so2_kg_per_gj >= Decimal("0")

    def test_biomass_lower_co2(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput,
        biomass_sample: FuelSampleInput
    ):
        """Test biomass has lower effective CO2 factor (carbon content basis)."""
        coal_result = analyzer.analyze_fuel_quality(bituminous_sample)
        bio_result = analyzer.analyze_fuel_quality(biomass_sample)

        # Biomass typically has lower carbon content
        # Note: In practice, biomass CO2 is often considered neutral
        # This test checks the calculation, not the accounting treatment
        assert bio_result.emission_factors.co2_kg_per_kg_fuel < coal_result.emission_factors.co2_kg_per_kg_fuel

    # ==========================================================================
    # Grade Classification Tests
    # ==========================================================================

    def test_grade_classification_bituminous(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test grade classification for good quality bituminous coal."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        # Good quality bituminous should be High or Standard
        assert result.grade_classification in [QualityGrade.PREMIUM, QualityGrade.HIGH, QualityGrade.STANDARD]

    def test_grade_classification_low_quality(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test grade classification for low quality fuel."""
        low_quality_proximate = ProximateAnalysis(
            moisture_percent=Decimal("20.0"),
            volatile_matter_percent=Decimal("20.0"),
            fixed_carbon_percent=Decimal("35.0"),
            ash_percent=Decimal("25.0"),  # High ash
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="LOW-QUAL-001",
            fuel_type=FuelType.LIGNITE,
            proximate=low_quality_proximate
        )

        result = analyzer.analyze_fuel_quality(sample)

        # Should be Low or Off-Spec due to high ash
        assert result.grade_classification in [QualityGrade.LOW, QualityGrade.OFF_SPEC]

    def test_quality_score_range(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test quality score is in valid range."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert Decimal("0") <= result.quality_score <= Decimal("100")

    # ==========================================================================
    # Quality Deviation Alert Tests
    # ==========================================================================

    def test_alerts_generated_for_deviations(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test alerts are generated for quality deviations."""
        high_ash_proximate = ProximateAnalysis(
            moisture_percent=Decimal("8.0"),
            volatile_matter_percent=Decimal("25.0"),
            fixed_carbon_percent=Decimal("42.0"),
            ash_percent=Decimal("25.0"),  # Very high ash
            basis=AnalysisBasis.AS_RECEIVED
        )

        high_sulfur_ultimate = UltimateAnalysis(
            carbon_percent=Decimal("65.0"),
            hydrogen_percent=Decimal("4.0"),
            oxygen_percent=Decimal("8.0"),
            nitrogen_percent=Decimal("1.5"),
            sulfur_percent=Decimal("4.5"),  # Very high sulfur
            ash_percent=Decimal("17.0"),
            basis=AnalysisBasis.DRY_BASIS
        )

        sample = FuelSampleInput(
            sample_id="ALERT-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=high_ash_proximate,
            ultimate=high_sulfur_ultimate
        )

        result = analyzer.analyze_fuel_quality(sample)

        # Should have alerts for high ash and/or sulfur
        assert len(result.alerts) > 0
        alert_params = [a.parameter for a in result.alerts]
        assert "Ash Content" in alert_params or "Sulfur Content" in alert_params

    def test_alert_includes_recommended_action(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test alerts include recommended actions."""
        high_ash_proximate = ProximateAnalysis(
            moisture_percent=Decimal("8.0"),
            volatile_matter_percent=Decimal("25.0"),
            fixed_carbon_percent=Decimal("42.0"),
            ash_percent=Decimal("25.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="ACTION-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=high_ash_proximate
        )

        result = analyzer.analyze_fuel_quality(sample)

        for alert in result.alerts:
            assert alert.recommended_action is not None
            assert len(alert.recommended_action) > 0

    # ==========================================================================
    # Sampling Frequency Tests
    # ==========================================================================

    def test_sampling_recommendation_provided(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test sampling recommendation is provided."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.sampling_recommendation is not None
        assert result.sampling_recommendation.recommended_frequency > 0

    def test_higher_frequency_for_variable_fuel(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput,
        biomass_sample: FuelSampleInput
    ):
        """Test higher sampling frequency for more variable fuel."""
        coal_result = analyzer.analyze_fuel_quality(bituminous_sample)
        bio_result = analyzer.analyze_fuel_quality(biomass_sample)

        # Biomass is typically more variable
        assert bio_result.sampling_recommendation.recommended_frequency >= \
               coal_result.sampling_recommendation.recommended_frequency

    def test_sampling_reasoning_provided(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test sampling recommendation includes reasoning."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.sampling_recommendation.reasoning is not None
        assert len(result.sampling_recommendation.reasoning) > 0

    # ==========================================================================
    # HGI Correlation Tests
    # ==========================================================================

    def test_hgi_correlation_for_coal(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test HGI is correlated for coal types."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.hardgrove_correlation is not None
        # HGI typically 20-100 for coal
        assert Decimal("20") <= result.hardgrove_correlation <= Decimal("100")

    def test_hgi_not_applicable_for_biomass(
        self,
        analyzer: FuelQualityAnalyzer,
        biomass_sample: FuelSampleInput
    ):
        """Test HGI correlation is None for non-coal fuels."""
        result = analyzer.analyze_fuel_quality(biomass_sample)

        # HGI not applicable for biomass
        assert result.hardgrove_correlation is None

    def test_measured_hgi_used(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_coal_proximate: ProximateAnalysis
    ):
        """Test measured HGI is used when provided."""
        sample = FuelSampleInput(
            sample_id="HGI-MEASURED-001",
            fuel_type=FuelType.BITUMINOUS,
            proximate=bituminous_coal_proximate,
            hardgrove_index=Decimal("58")
        )

        result = analyzer.analyze_fuel_quality(sample)

        assert result.hardgrove_correlation == Decimal("58")

    # ==========================================================================
    # Provenance and Determinism Tests
    # ==========================================================================

    def test_provenance_hash_generated(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test provenance hash is generated."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_deterministic_results(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test same inputs produce same outputs (determinism)."""
        result1 = analyzer.analyze_fuel_quality(bituminous_sample)
        result2 = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result1.heating_value.hhv_mj_kg == result2.heating_value.hhv_mj_kg
        assert result1.heating_value.lhv_mj_kg == result2.heating_value.lhv_mj_kg
        assert result1.grade_classification == result2.grade_classification
        assert result1.fuel_ratio == result2.fuel_ratio

    def test_calculation_steps_recorded(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test calculation steps are recorded for provenance."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert len(result.calculation_steps) > 0

        # Check steps have required fields
        for step in result.calculation_steps:
            assert 'step_number' in step
            assert 'operation' in step
            assert 'formula' in step

    # ==========================================================================
    # Processing Time Tests
    # ==========================================================================

    def test_processing_time_recorded(
        self,
        analyzer: FuelQualityAnalyzer,
        bituminous_sample: FuelSampleInput
    ):
        """Test processing time is recorded."""
        result = analyzer.analyze_fuel_quality(bituminous_sample)

        assert result.processing_time_ms >= Decimal("0")

    # ==========================================================================
    # Edge Case Tests
    # ==========================================================================

    def test_high_moisture_fuel(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test analysis handles high moisture fuel."""
        high_moisture_proximate = ProximateAnalysis(
            moisture_percent=Decimal("45.0"),
            volatile_matter_percent=Decimal("25.0"),
            fixed_carbon_percent=Decimal("20.0"),
            ash_percent=Decimal("10.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="HIGH-MOIST-001",
            fuel_type=FuelType.LIGNITE,
            proximate=high_moisture_proximate
        )

        result = analyzer.analyze_fuel_quality(sample)

        # Should complete successfully
        assert result is not None
        # Dry basis conversion should work
        assert result.proximate_dry_basis.volatile_matter_percent > high_moisture_proximate.volatile_matter_percent

    def test_very_high_ash_fuel(
        self,
        analyzer: FuelQualityAnalyzer
    ):
        """Test analysis handles very high ash fuel."""
        high_ash_proximate = ProximateAnalysis(
            moisture_percent=Decimal("5.0"),
            volatile_matter_percent=Decimal("20.0"),
            fixed_carbon_percent=Decimal("30.0"),
            ash_percent=Decimal("45.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="HIGH-ASH-001",
            fuel_type=FuelType.LIGNITE,
            proximate=high_ash_proximate
        )

        result = analyzer.analyze_fuel_quality(sample)

        # Should complete and flag low quality
        assert result is not None
        assert result.grade_classification == QualityGrade.OFF_SPEC

    def test_statistics_tracking(self, analyzer: FuelQualityAnalyzer):
        """Test statistics are tracked correctly."""
        stats = analyzer.get_statistics()

        assert 'analysis_count' in stats
        assert 'heating_value_method' in stats
        assert stats['analysis_count'] >= 0

    def test_unreasonable_moisture_rejected(self, analyzer: FuelQualityAnalyzer):
        """Test unreasonably high moisture is rejected."""
        extreme_moisture = ProximateAnalysis(
            moisture_percent=Decimal("70.0"),  # > 60% limit
            volatile_matter_percent=Decimal("15.0"),
            fixed_carbon_percent=Decimal("10.0"),
            ash_percent=Decimal("5.0"),
            basis=AnalysisBasis.AS_RECEIVED
        )

        sample = FuelSampleInput(
            sample_id="EXTREME-MOIST-001",
            fuel_type=FuelType.LIGNITE,
            proximate=extreme_moisture
        )

        with pytest.raises(ValueError, match="Moisture"):
            analyzer.analyze_fuel_quality(sample)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
