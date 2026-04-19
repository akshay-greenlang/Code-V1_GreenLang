# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Fouling Predictor Unit Tests

Comprehensive unit tests for fouling prediction calculations including:
- Fouling rate estimation
- Biofouling predictions
- Scale formation predictions
- Cleaning interval optimization
- Efficiency loss calculations from fouling

Test coverage target: 95%+

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from calculators.fouling_calculator import (
    FoulingCalculator,
    FoulingInput,
    FoulingOutput,
    FoulingType,
    calculate_fouling_resistance,
    calculate_biofouling_rate,
    calculate_scale_formation_rate,
    calculate_cleaning_interval,
    estimate_efficiency_loss,
    FOULING_RATE_CONSTANTS,
    BIOFOULING_FACTORS,
    SCALE_FORMATION_FACTORS,
)
from calculators.provenance import (
    ProvenanceTracker,
    ProvenanceRecord,
    verify_provenance,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def fouling_calculator():
    """Create FoulingCalculator instance."""
    return FoulingCalculator()


@pytest.fixture
def standard_fouling_input():
    """Standard fouling input for testing."""
    return FoulingInput(
        tube_material="titanium",
        cooling_water_source="cooling_tower",
        cooling_water_tds_ppm=1800.0,
        cooling_water_ph=7.8,
        cooling_water_temp_c=28.0,
        tube_velocity_m_s=2.2,
        operating_hours=4380.0,
        biocide_treatment="oxidizing",
        current_cleanliness_factor=0.85,
        design_fouling_factor_m2k_w=0.00015,
        cycles_of_concentration=4.0,
    )


@pytest.fixture
def high_fouling_input():
    """High fouling risk input for testing."""
    return FoulingInput(
        tube_material="admiralty_brass",
        cooling_water_source="river",
        cooling_water_tds_ppm=3500.0,
        cooling_water_ph=8.5,
        cooling_water_temp_c=35.0,
        tube_velocity_m_s=1.5,
        operating_hours=6000.0,
        biocide_treatment="none",
        current_cleanliness_factor=0.70,
        design_fouling_factor_m2k_w=0.00015,
        cycles_of_concentration=6.0,
    )


@pytest.fixture
def low_fouling_input():
    """Low fouling risk input for testing."""
    return FoulingInput(
        tube_material="titanium",
        cooling_water_source="seawater",
        cooling_water_tds_ppm=35000.0,  # High TDS but controlled
        cooling_water_ph=8.0,
        cooling_water_temp_c=22.0,
        tube_velocity_m_s=2.5,
        operating_hours=2000.0,
        biocide_treatment="electro_chlorination",
        current_cleanliness_factor=0.95,
        design_fouling_factor_m2k_w=0.00015,
        cycles_of_concentration=1.0,
    )


@pytest.fixture
def water_quality_samples():
    """Sample water quality data for testing."""
    return [
        {
            "source": "cooling_tower",
            "tds_ppm": 2000,
            "ph": 7.5,
            "calcium_ppm": 250,
            "alkalinity_ppm": 180,
            "silica_ppm": 25,
            "chloride_ppm": 150,
        },
        {
            "source": "river",
            "tds_ppm": 500,
            "ph": 7.2,
            "calcium_ppm": 80,
            "alkalinity_ppm": 100,
            "silica_ppm": 15,
            "chloride_ppm": 30,
        },
        {
            "source": "seawater",
            "tds_ppm": 35000,
            "ph": 8.1,
            "calcium_ppm": 400,
            "alkalinity_ppm": 120,
            "silica_ppm": 5,
            "chloride_ppm": 19000,
        },
    ]


@pytest.fixture
def historical_fouling_data():
    """Historical fouling trend data for testing."""
    base_time = datetime.utcnow()
    data = []

    for day in range(90):  # 90 days of data
        timestamp = base_time - timedelta(days=day)
        # Simulate gradual fouling
        fouling_factor = 0.00015 + (day * 0.000002)
        cleanliness = 0.95 - (day * 0.002)

        data.append({
            "timestamp": timestamp.isoformat(),
            "fouling_factor_m2k_w": min(fouling_factor, 0.0005),
            "cleanliness_factor": max(cleanliness, 0.60),
            "water_temp_c": 25.0 + (5 * math.sin(day / 7)),
            "flow_velocity_m_s": 2.2,
        })

    return data


# =============================================================================
# FOULING RATE ESTIMATION TESTS
# =============================================================================

class TestFoulingRateEstimation:
    """Test suite for fouling rate estimation."""

    @pytest.mark.unit
    def test_fouling_rate_basic(self, fouling_calculator, standard_fouling_input):
        """Test basic fouling rate estimation."""
        result, provenance = fouling_calculator.calculate(standard_fouling_input)

        assert result.fouling_rate_m2k_w_per_hour > 0
        assert provenance.provenance_hash is not None

    @pytest.mark.unit
    @pytest.mark.parametrize("tds_ppm,expected_rate_range", [
        (500, (1e-9, 5e-9)),    # Low TDS
        (1500, (5e-9, 2e-8)),   # Medium TDS
        (3000, (2e-8, 1e-7)),   # High TDS
        (5000, (5e-8, 5e-7)),   # Very high TDS
    ])
    def test_fouling_rate_vs_tds(self, fouling_calculator, tds_ppm, expected_rate_range):
        """Test fouling rate correlation with TDS."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=tds_ppm,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        assert expected_rate_range[0] <= result.fouling_rate_m2k_w_per_hour <= expected_rate_range[1]

    @pytest.mark.unit
    @pytest.mark.parametrize("temp_c,fouling_multiplier_range", [
        (20.0, (0.5, 0.8)),   # Cold - slower fouling
        (28.0, (0.9, 1.1)),   # Design - normal
        (35.0, (1.2, 1.6)),   # Warm - faster fouling
        (40.0, (1.5, 2.5)),   # Hot - much faster
    ])
    def test_fouling_rate_vs_temperature(self, fouling_calculator, temp_c, fouling_multiplier_range):
        """Test fouling rate increases with temperature."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=temp_c,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Verify temperature effect on fouling
        assert result.temperature_factor >= fouling_multiplier_range[0]
        assert result.temperature_factor <= fouling_multiplier_range[1]

    @pytest.mark.unit
    @pytest.mark.parametrize("velocity_m_s,expected_effect", [
        (1.0, "high_fouling"),    # Low velocity = high fouling
        (2.0, "normal_fouling"),  # Normal velocity
        (3.0, "low_fouling"),     # High velocity = low fouling
    ])
    def test_fouling_rate_vs_velocity(self, fouling_calculator, velocity_m_s, expected_effect):
        """Test fouling rate decreases with velocity."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=velocity_m_s,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Higher velocity should result in lower fouling rate
        if expected_effect == "high_fouling":
            assert result.velocity_factor > 1.0
        elif expected_effect == "low_fouling":
            assert result.velocity_factor < 1.0

    @pytest.mark.unit
    def test_fouling_rate_with_biocide(self, fouling_calculator):
        """Test biocide treatment effect on fouling rate."""
        base_input = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="none",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        treated_input = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        no_treatment, _ = fouling_calculator.calculate(base_input)
        with_treatment, _ = fouling_calculator.calculate(treated_input)

        # Biocide treatment should reduce fouling rate
        assert with_treatment.fouling_rate_m2k_w_per_hour < no_treatment.fouling_rate_m2k_w_per_hour


# =============================================================================
# BIOFOULING PREDICTION TESTS
# =============================================================================

class TestBiofoulingPredictions:
    """Test suite for biofouling predictions."""

    @pytest.mark.unit
    def test_biofouling_rate_basic(self, fouling_calculator, standard_fouling_input):
        """Test basic biofouling rate calculation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.biofouling_rate_m2k_w_per_hour >= 0
        assert result.biofouling_contribution_pct >= 0
        assert result.biofouling_contribution_pct <= 100

    @pytest.mark.unit
    @pytest.mark.parametrize("water_source,expected_biofouling_risk", [
        ("cooling_tower", "high"),
        ("river", "medium"),
        ("seawater", "medium"),
        ("well_water", "low"),
    ])
    def test_biofouling_by_water_source(self, fouling_calculator, water_source, expected_biofouling_risk):
        """Test biofouling risk varies by water source."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source=water_source,
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        if expected_biofouling_risk == "high":
            assert result.biofouling_risk_score >= 0.6
        elif expected_biofouling_risk == "medium":
            assert 0.3 <= result.biofouling_risk_score < 0.6
        else:
            assert result.biofouling_risk_score < 0.3

    @pytest.mark.unit
    @pytest.mark.parametrize("temp_c,expected_growth_rate", [
        (15.0, "slow"),
        (25.0, "moderate"),
        (30.0, "fast"),
        (35.0, "very_fast"),
        (45.0, "inhibited"),  # Too hot for most organisms
    ])
    def test_biofouling_temperature_dependence(self, fouling_calculator, temp_c, expected_growth_rate):
        """Test biofouling rate varies with temperature."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=temp_c,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Biofouling should follow temperature-dependent pattern
        if expected_growth_rate == "inhibited":
            assert result.biofouling_temperature_factor < 0.5
        elif expected_growth_rate == "very_fast":
            assert result.biofouling_temperature_factor > 1.2

    @pytest.mark.unit
    @pytest.mark.parametrize("biocide,effectiveness_range", [
        ("none", (0.0, 0.1)),
        ("oxidizing", (0.5, 0.8)),
        ("non_oxidizing", (0.4, 0.7)),
        ("electro_chlorination", (0.6, 0.9)),
        ("uv_treatment", (0.5, 0.8)),
    ])
    def test_biocide_effectiveness(self, fouling_calculator, biocide, effectiveness_range):
        """Test biocide treatment effectiveness on biofouling."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment=biocide,
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        assert effectiveness_range[0] <= result.biocide_effectiveness <= effectiveness_range[1]

    @pytest.mark.unit
    def test_legionella_risk_assessment(self, fouling_calculator, standard_fouling_input):
        """Test Legionella risk assessment from biofouling conditions."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        # Legionella risk should be assessed
        assert hasattr(result, 'legionella_risk_score')
        assert 0 <= result.legionella_risk_score <= 1.0

    @pytest.mark.unit
    def test_biofilm_thickness_estimation(self, fouling_calculator, high_fouling_input):
        """Test biofilm thickness estimation."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # Biofilm thickness should be estimated in micrometers
        assert result.estimated_biofilm_thickness_um >= 0
        assert result.estimated_biofilm_thickness_um < 1000  # <1mm


# =============================================================================
# SCALE FORMATION PREDICTION TESTS
# =============================================================================

class TestScaleFormationPredictions:
    """Test suite for scale formation predictions."""

    @pytest.mark.unit
    def test_scale_formation_basic(self, fouling_calculator, standard_fouling_input):
        """Test basic scale formation calculation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.scale_formation_rate_m2k_w_per_hour >= 0
        assert result.scale_contribution_pct >= 0
        assert result.scale_contribution_pct <= 100

    @pytest.mark.unit
    @pytest.mark.parametrize("tds_ppm,cycles,expected_scale_risk", [
        (500, 2.0, "low"),
        (1500, 4.0, "medium"),
        (3000, 6.0, "high"),
        (5000, 8.0, "very_high"),
    ])
    def test_scale_vs_concentration(self, fouling_calculator, tds_ppm, cycles, expected_scale_risk):
        """Test scale formation increases with concentration."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=tds_ppm,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=cycles,
        )

        result, _ = fouling_calculator.calculate(input_data)

        if expected_scale_risk == "very_high":
            assert result.scale_risk_score >= 0.8
        elif expected_scale_risk == "high":
            assert 0.6 <= result.scale_risk_score < 0.8
        elif expected_scale_risk == "medium":
            assert 0.3 <= result.scale_risk_score < 0.6
        else:
            assert result.scale_risk_score < 0.3

    @pytest.mark.unit
    @pytest.mark.parametrize("ph,scale_tendency", [
        (6.5, "low"),       # Acidic - corrosive but low scaling
        (7.5, "moderate"),  # Neutral
        (8.5, "high"),      # Alkaline - scaling tendency
        (9.0, "very_high"), # Very alkaline
    ])
    def test_scale_vs_ph(self, fouling_calculator, ph, scale_tendency):
        """Test scale formation varies with pH."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=ph,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Higher pH increases scaling tendency
        if scale_tendency == "very_high":
            assert result.ph_scale_factor > 1.3
        elif scale_tendency == "high":
            assert result.ph_scale_factor > 1.1
        elif scale_tendency == "low":
            assert result.ph_scale_factor < 0.9

    @pytest.mark.unit
    def test_langelier_saturation_index(self, fouling_calculator, standard_fouling_input):
        """Test Langelier Saturation Index calculation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        # LSI indicates scaling/corrosion tendency
        assert -3.0 <= result.langelier_saturation_index <= 3.0

        # Positive LSI = scaling, Negative = corrosion
        if result.langelier_saturation_index > 0:
            assert result.scale_tendency == "scaling"
        else:
            assert result.scale_tendency in ["neutral", "corrosive"]

    @pytest.mark.unit
    @pytest.mark.parametrize("scale_type,typical_thickness_um", [
        ("calcium_carbonate", (50, 500)),
        ("calcium_sulfate", (30, 300)),
        ("silica", (20, 200)),
        ("magnesium_silicate", (25, 250)),
    ])
    def test_scale_type_identification(self, fouling_calculator, scale_type, typical_thickness_um):
        """Test scale type identification and thickness estimation."""
        # Adjust water chemistry for different scale types
        if scale_type == "calcium_carbonate":
            ph = 8.2
            tds = 2000
        elif scale_type == "silica":
            ph = 7.0
            tds = 1500
        else:
            ph = 7.8
            tds = 2500

        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=tds,
            cooling_water_ph=ph,
            cooling_water_temp_c=35.0,
            tube_velocity_m_s=2.0,
            operating_hours=6000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.75,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=5.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Should identify primary scale type
        assert result.primary_scale_type in [
            "calcium_carbonate", "calcium_sulfate", "silica", "magnesium_silicate", "mixed"
        ]


# =============================================================================
# CLEANING INTERVAL OPTIMIZATION TESTS
# =============================================================================

class TestCleaningIntervalOptimization:
    """Test suite for cleaning interval optimization."""

    @pytest.mark.unit
    def test_cleaning_interval_basic(self, fouling_calculator, standard_fouling_input):
        """Test basic cleaning interval calculation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.recommended_cleaning_interval_hours > 0
        assert result.recommended_cleaning_interval_hours < 20000  # <~2.3 years

    @pytest.mark.unit
    def test_cleaning_interval_high_fouling(self, fouling_calculator, high_fouling_input):
        """Test cleaning interval for high fouling conditions."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # High fouling should require more frequent cleaning
        assert result.recommended_cleaning_interval_hours < 4000

    @pytest.mark.unit
    def test_cleaning_interval_low_fouling(self, fouling_calculator, low_fouling_input):
        """Test cleaning interval for low fouling conditions."""
        result, _ = fouling_calculator.calculate(low_fouling_input)

        # Low fouling allows longer intervals
        assert result.recommended_cleaning_interval_hours > 6000

    @pytest.mark.unit
    @pytest.mark.parametrize("current_cf,urgency", [
        (0.95, "none"),
        (0.85, "low"),
        (0.75, "medium"),
        (0.65, "high"),
        (0.55, "critical"),
    ])
    def test_cleaning_urgency_levels(self, fouling_calculator, current_cf, urgency):
        """Test cleaning urgency classification."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=current_cf,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        assert result.cleaning_urgency == urgency

    @pytest.mark.unit
    def test_cleaning_method_recommendation(self, fouling_calculator, standard_fouling_input):
        """Test cleaning method recommendation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        # Should recommend appropriate cleaning method
        valid_methods = [
            "mechanical_brush",
            "sponge_ball",
            "hydro_blast",
            "chemical",
            "combined",
        ]
        assert result.recommended_cleaning_method in valid_methods

    @pytest.mark.unit
    @pytest.mark.parametrize("fouling_type,expected_method", [
        ("biofouling", "chemical"),
        ("scale", "hydro_blast"),
        ("silt", "mechanical_brush"),
        ("mixed", "combined"),
    ])
    def test_cleaning_method_by_fouling_type(self, fouling_calculator, fouling_type, expected_method):
        """Test cleaning method varies by fouling type."""
        # Adjust inputs to produce different fouling types
        if fouling_type == "biofouling":
            input_data = FoulingInput(
                tube_material="titanium",
                cooling_water_source="cooling_tower",
                cooling_water_tds_ppm=1500.0,
                cooling_water_ph=7.5,
                cooling_water_temp_c=32.0,
                tube_velocity_m_s=1.8,
                operating_hours=5000.0,
                biocide_treatment="none",
                current_cleanliness_factor=0.75,
                design_fouling_factor_m2k_w=0.00015,
                cycles_of_concentration=3.0,
            )
        elif fouling_type == "scale":
            input_data = FoulingInput(
                tube_material="titanium",
                cooling_water_source="cooling_tower",
                cooling_water_tds_ppm=4000.0,
                cooling_water_ph=8.5,
                cooling_water_temp_c=35.0,
                tube_velocity_m_s=2.2,
                operating_hours=5000.0,
                biocide_treatment="oxidizing",
                current_cleanliness_factor=0.75,
                design_fouling_factor_m2k_w=0.00015,
                cycles_of_concentration=6.0,
            )
        else:
            input_data = FoulingInput(
                tube_material="titanium",
                cooling_water_source="river",
                cooling_water_tds_ppm=800.0,
                cooling_water_ph=7.2,
                cooling_water_temp_c=25.0,
                tube_velocity_m_s=2.0,
                operating_hours=5000.0,
                biocide_treatment="oxidizing",
                current_cleanliness_factor=0.75,
                design_fouling_factor_m2k_w=0.00015,
                cycles_of_concentration=1.0,
            )

        result, _ = fouling_calculator.calculate(input_data)

        # Verify appropriate method recommended
        assert result.recommended_cleaning_method in [expected_method, "combined"]

    @pytest.mark.unit
    def test_cleaning_cost_estimation(self, fouling_calculator, standard_fouling_input):
        """Test cleaning cost estimation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.estimated_cleaning_cost_usd > 0
        assert result.estimated_cleaning_cost_usd < 500000  # Reasonable range

    @pytest.mark.unit
    def test_cleaning_roi_calculation(self, fouling_calculator, high_fouling_input):
        """Test cleaning ROI calculation."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # Should calculate positive ROI for cleaning
        assert result.cleaning_roi_percent > 0

    @pytest.mark.unit
    def test_optimal_cleaning_schedule(self, fouling_calculator, standard_fouling_input):
        """Test optimal cleaning schedule generation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        # Should provide next recommended cleaning date
        assert result.next_recommended_cleaning_date is not None


# =============================================================================
# EFFICIENCY LOSS CALCULATION TESTS
# =============================================================================

class TestEfficiencyLossCalculations:
    """Test suite for efficiency loss calculations from fouling."""

    @pytest.mark.unit
    def test_efficiency_loss_basic(self, fouling_calculator, standard_fouling_input):
        """Test basic efficiency loss calculation."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.efficiency_loss_percent >= 0
        assert result.efficiency_loss_percent < 50

    @pytest.mark.unit
    @pytest.mark.parametrize("cleanliness_factor,expected_loss_range", [
        (0.95, (0, 5)),
        (0.85, (5, 15)),
        (0.75, (15, 25)),
        (0.65, (25, 35)),
        (0.55, (35, 50)),
    ])
    def test_efficiency_loss_vs_cleanliness(self, fouling_calculator, cleanliness_factor, expected_loss_range):
        """Test efficiency loss increases as cleanliness decreases."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=cleanliness_factor,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        assert expected_loss_range[0] <= result.efficiency_loss_percent <= expected_loss_range[1]

    @pytest.mark.unit
    def test_heat_rate_penalty_from_fouling(self, fouling_calculator, high_fouling_input):
        """Test heat rate penalty calculation from fouling."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # High fouling should result in heat rate penalty
        assert result.heat_rate_penalty_kj_kwh > 0
        assert result.heat_rate_penalty_kj_kwh < 500

    @pytest.mark.unit
    def test_backpressure_increase_from_fouling(self, fouling_calculator, high_fouling_input):
        """Test backpressure increase due to fouling."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # Fouling increases backpressure
        assert result.backpressure_increase_mbar > 0

    @pytest.mark.unit
    def test_power_output_loss(self, fouling_calculator, high_fouling_input):
        """Test power output loss estimation from fouling."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        assert result.power_output_loss_mw >= 0

    @pytest.mark.unit
    def test_annual_cost_of_fouling(self, fouling_calculator, standard_fouling_input):
        """Test annual cost calculation of fouling."""
        result, _ = fouling_calculator.calculate(standard_fouling_input)

        assert result.annual_fouling_cost_usd >= 0

    @pytest.mark.unit
    def test_fouling_carbon_impact(self, fouling_calculator, high_fouling_input):
        """Test carbon emission impact from fouling."""
        result, _ = fouling_calculator.calculate(high_fouling_input)

        # Fouling increases carbon emissions due to efficiency loss
        assert result.additional_carbon_emissions_tonnes_yr >= 0


# =============================================================================
# FOULING TREND ANALYSIS TESTS
# =============================================================================

class TestFoulingTrendAnalysis:
    """Test suite for fouling trend analysis."""

    @pytest.mark.unit
    def test_trend_analysis_with_historical_data(self, fouling_calculator, historical_fouling_data):
        """Test fouling trend analysis with historical data."""
        result = fouling_calculator.analyze_trend(historical_fouling_data)

        assert result.trend_direction in ["increasing", "stable", "decreasing"]
        assert result.trend_slope is not None

    @pytest.mark.unit
    def test_fouling_rate_projection(self, fouling_calculator, historical_fouling_data):
        """Test fouling rate projection."""
        result = fouling_calculator.analyze_trend(historical_fouling_data)

        # Should project future fouling
        assert result.projected_cleanliness_30_days is not None
        assert result.projected_cleanliness_60_days is not None
        assert result.projected_cleanliness_90_days is not None

    @pytest.mark.unit
    def test_cleaning_timing_prediction(self, fouling_calculator, historical_fouling_data):
        """Test cleaning timing prediction from trend."""
        result = fouling_calculator.analyze_trend(historical_fouling_data)

        # Should predict when cleaning will be needed
        assert result.days_until_cleaning_required is not None
        assert result.days_until_cleaning_required >= 0

    @pytest.mark.unit
    def test_seasonal_fouling_pattern(self, fouling_calculator, historical_fouling_data):
        """Test seasonal fouling pattern detection."""
        result = fouling_calculator.analyze_trend(historical_fouling_data)

        # May detect seasonal patterns
        assert hasattr(result, 'seasonal_pattern_detected')

    @pytest.mark.unit
    def test_anomaly_detection(self, fouling_calculator, historical_fouling_data):
        """Test anomaly detection in fouling trends."""
        # Add anomalies to data
        anomalous_data = historical_fouling_data.copy()
        anomalous_data[45]['fouling_factor_m2k_w'] = 0.001  # Sudden spike

        result = fouling_calculator.analyze_trend(anomalous_data)

        assert hasattr(result, 'anomalies_detected')


# =============================================================================
# TUBE MATERIAL COMPATIBILITY TESTS
# =============================================================================

class TestTubeMaterialCompatibility:
    """Test suite for tube material compatibility with fouling."""

    @pytest.mark.unit
    @pytest.mark.parametrize("material,water_source,compatibility", [
        ("titanium", "seawater", "excellent"),
        ("titanium", "cooling_tower", "excellent"),
        ("stainless_316", "seawater", "good"),
        ("admiralty_brass", "seawater", "poor"),
        ("admiralty_brass", "freshwater", "good"),
        ("copper_nickel_90_10", "seawater", "excellent"),
    ])
    def test_material_water_compatibility(self, fouling_calculator, material, water_source, compatibility):
        """Test tube material compatibility with water source."""
        input_data = FoulingInput(
            tube_material=material,
            cooling_water_source=water_source,
            cooling_water_tds_ppm=35000 if water_source == "seawater" else 2000,
            cooling_water_ph=8.0,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.90,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=1.0 if water_source == "seawater" else 4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        assert result.material_compatibility == compatibility

    @pytest.mark.unit
    def test_corrosion_fouling_interaction(self, fouling_calculator):
        """Test corrosion-fouling interaction for susceptible materials."""
        input_data = FoulingInput(
            tube_material="admiralty_brass",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=3000.0,
            cooling_water_ph=6.5,  # Acidic - corrosive
            cooling_water_temp_c=30.0,
            tube_velocity_m_s=2.2,
            operating_hours=4000.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=0.80,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=5.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Should warn about corrosion-fouling interaction
        assert result.corrosion_risk_score > 0.5


# =============================================================================
# PROVENANCE AND DETERMINISM TESTS
# =============================================================================

class TestFoulingProvenance:
    """Test suite for fouling calculation provenance."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, fouling_calculator, standard_fouling_input):
        """Test that provenance hash is generated."""
        result, provenance = fouling_calculator.calculate(standard_fouling_input)

        assert provenance.provenance_hash is not None
        assert len(provenance.provenance_hash) == 64

    @pytest.mark.unit
    def test_calculation_steps_recorded(self, fouling_calculator, standard_fouling_input):
        """Test that calculation steps are recorded."""
        result, provenance = fouling_calculator.calculate(standard_fouling_input)

        assert len(provenance.calculation_steps) > 0

    @pytest.mark.unit
    def test_deterministic_results(self, fouling_calculator, standard_fouling_input):
        """Test that same input produces same output."""
        result1, prov1 = fouling_calculator.calculate(standard_fouling_input)
        result2, prov2 = fouling_calculator.calculate(standard_fouling_input)

        assert result1.fouling_rate_m2k_w_per_hour == result2.fouling_rate_m2k_w_per_hour
        assert prov1.output_hash == prov2.output_hash

    @pytest.mark.unit
    def test_provenance_verification(self, fouling_calculator, standard_fouling_input):
        """Test provenance verification."""
        result, provenance = fouling_calculator.calculate(standard_fouling_input)

        is_valid = verify_provenance(provenance)
        assert is_valid is True


# =============================================================================
# STANDALONE FUNCTION TESTS
# =============================================================================

class TestStandaloneFunctions:
    """Test suite for standalone fouling functions."""

    @pytest.mark.unit
    def test_fouling_resistance_calculation(self):
        """Test fouling resistance calculation."""
        clean_u = 3500.0
        fouled_u = 3000.0

        resistance = calculate_fouling_resistance(clean_u, fouled_u)

        # R_f = 1/U_fouled - 1/U_clean
        expected = (1/fouled_u) - (1/clean_u)
        assert abs(resistance - expected) < 1e-8

    @pytest.mark.unit
    def test_biofouling_rate_calculation(self):
        """Test biofouling rate calculation."""
        temp_c = 30.0
        velocity_m_s = 2.0
        biocide_factor = 0.5

        rate = calculate_biofouling_rate(temp_c, velocity_m_s, biocide_factor)

        assert rate > 0

    @pytest.mark.unit
    def test_scale_formation_rate_calculation(self):
        """Test scale formation rate calculation."""
        tds_ppm = 2000.0
        ph = 8.0
        temp_c = 35.0
        cycles = 4.0

        rate = calculate_scale_formation_rate(tds_ppm, ph, temp_c, cycles)

        assert rate > 0

    @pytest.mark.unit
    def test_cleaning_interval_calculation(self):
        """Test cleaning interval calculation."""
        current_fouling = 0.0003
        fouling_rate = 1e-8
        max_fouling = 0.0005

        interval = calculate_cleaning_interval(current_fouling, fouling_rate, max_fouling)

        # Time to reach max fouling
        expected = (max_fouling - current_fouling) / fouling_rate
        assert abs(interval - expected) < 1.0

    @pytest.mark.unit
    def test_efficiency_loss_estimation(self):
        """Test efficiency loss estimation."""
        cleanliness_factor = 0.80

        loss = estimate_efficiency_loss(cleanliness_factor)

        # Loss should be proportional to degradation
        assert loss > 0
        assert loss < 100


# =============================================================================
# BOUNDARY CONDITION TESTS
# =============================================================================

class TestFoulingBoundaryConditions:
    """Test suite for fouling boundary conditions."""

    @pytest.mark.unit
    def test_minimum_fouling_factor(self, fouling_calculator):
        """Test minimum fouling factor (clean tubes)."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="well_water",
            cooling_water_tds_ppm=200.0,
            cooling_water_ph=7.0,
            cooling_water_temp_c=20.0,
            tube_velocity_m_s=2.5,
            operating_hours=1000.0,
            biocide_treatment="electro_chlorination",
            current_cleanliness_factor=0.99,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=1.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Very low fouling conditions
        assert result.fouling_rate_m2k_w_per_hour < 1e-8

    @pytest.mark.unit
    def test_maximum_fouling_factor(self, fouling_calculator):
        """Test maximum fouling factor (severe conditions)."""
        input_data = FoulingInput(
            tube_material="admiralty_brass",
            cooling_water_source="river",
            cooling_water_tds_ppm=5000.0,
            cooling_water_ph=9.0,
            cooling_water_temp_c=38.0,
            tube_velocity_m_s=1.0,
            operating_hours=8000.0,
            biocide_treatment="none",
            current_cleanliness_factor=0.50,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=8.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Severe fouling conditions
        assert result.fouling_rate_m2k_w_per_hour > 1e-7

    @pytest.mark.unit
    def test_zero_operating_hours(self, fouling_calculator):
        """Test with zero operating hours (new condenser)."""
        input_data = FoulingInput(
            tube_material="titanium",
            cooling_water_source="cooling_tower",
            cooling_water_tds_ppm=2000.0,
            cooling_water_ph=7.8,
            cooling_water_temp_c=28.0,
            tube_velocity_m_s=2.2,
            operating_hours=0.0,
            biocide_treatment="oxidizing",
            current_cleanliness_factor=1.0,
            design_fouling_factor_m2k_w=0.00015,
            cycles_of_concentration=4.0,
        )

        result, _ = fouling_calculator.calculate(input_data)

        # Should predict fouling rate for new condenser
        assert result.fouling_rate_m2k_w_per_hour >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
