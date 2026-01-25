# -*- coding: utf-8 -*-
"""
Unit tests for GL-016 WATERGUARD Tools.

Tests all tool functions with comprehensive coverage:
- Water chemistry analysis tools
- Scale prediction tools
- Corrosion analysis tools
- Blowdown optimization tools
- Chemical dosing tools

Author: GL-016 Test Engineering Team
Target Coverage: >85%
"""

import pytest
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools import (
    WaterTreatmentTools,
    WaterQualityAnalysis,
    BlowdownOptimization,
    ChemicalOptimization,
    ComplianceResult,
    ValidationResult,
    ScavengerType,
    AmineType,
    BoilerType
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def standard_water_data():
    """Standard water chemistry data for testing."""
    return {
        'pH': 8.5,
        'temperature': 85.0,
        'conductivity': 1200.0,
        'calcium_hardness': 150.0,
        'magnesium_hardness': 75.0,
        'alkalinity': 250.0,
        'chloride': 150.0,
        'sulfate': 100.0,
        'silica': 25.0,
        'tds': 800.0,
        'dissolved_oxygen': 0.02,
        'iron': 0.05,
        'copper': 0.01,
        'phosphate': 15.0
    }


@pytest.fixture
def high_scale_water_data():
    """Water data prone to scaling."""
    return {
        'pH': 9.0,
        'temperature': 95.0,
        'conductivity': 2000.0,
        'calcium_hardness': 350.0,
        'magnesium_hardness': 150.0,
        'alkalinity': 400.0,
        'chloride': 200.0,
        'sulfate': 150.0,
        'silica': 50.0,
        'tds': 1500.0,
        'dissolved_oxygen': 0.05,
        'iron': 0.1,
        'copper': 0.02,
        'phosphate': 10.0
    }


@pytest.fixture
def corrosive_water_data():
    """Water data prone to corrosion."""
    return {
        'pH': 6.5,
        'temperature': 80.0,
        'conductivity': 1500.0,
        'calcium_hardness': 50.0,
        'magnesium_hardness': 25.0,
        'alkalinity': 80.0,
        'chloride': 300.0,
        'sulfate': 200.0,
        'silica': 15.0,
        'tds': 900.0,
        'dissolved_oxygen': 0.2,
        'iron': 0.15,
        'copper': 0.03,
        'phosphate': 5.0
    }


@pytest.fixture
def blowdown_data():
    """Standard blowdown optimization data."""
    return {
        'makeup_conductivity': 200.0,
        'blowdown_conductivity': 2000.0,
        'tds': 2000.0,
        'alkalinity': 400.0,
        'temperature': 180.0,
        'pressure': 10.0,
        'water_cost': 0.5,
        'energy_cost': 0.08
    }


@pytest.fixture
def chemical_usage_data():
    """Current chemical usage data."""
    return {
        'phosphate': 2.0,
        'oxygen_scavenger': 1.5,
        'amine': 0.5,
        'polymer': 1.0
    }


@pytest.fixture
def water_quality_data():
    """Water quality data for chemical optimization."""
    return {
        'steam_rate': 5000.0,
        'dissolved_oxygen': 200.0,
        'hardness': 100.0,
        'volume': 20.0,
        'phosphate_residual': 30.0,
        'condensate_return_percent': 80.0
    }


@pytest.fixture
def target_data():
    """Target data for chemical optimization."""
    return {
        'phosphate_residual': 50.0,
        'condensate_pH': 8.8,
        'sludge_conditioner_need': 50.0
    }


# ============================================================================
# LSI Calculation Tests
# ============================================================================

class TestLangelierSaturationIndex:
    """Tests for Langelier Saturation Index calculation."""

    @pytest.mark.unit
    def test_lsi_basic_calculation(self):
        """Test basic LSI calculation."""
        lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
            pH=8.5,
            temperature=85.0,
            calcium_hardness=150.0,
            alkalinity=250.0,
            tds=800.0
        )

        assert isinstance(lsi, float)
        assert -5.0 <= lsi <= 5.0  # Typical LSI range

    @pytest.mark.unit
    def test_lsi_positive_scale_forming(self):
        """Test LSI positive for scale-forming water."""
        # High pH, high calcium, high alkalinity = scale-forming
        lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
            pH=9.5,
            temperature=90.0,
            calcium_hardness=300.0,
            alkalinity=400.0,
            tds=1200.0
        )

        assert lsi > 0  # Scale-forming

    @pytest.mark.unit
    def test_lsi_negative_corrosive(self):
        """Test LSI negative for corrosive water."""
        # Low pH, low calcium, low alkalinity = corrosive
        lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
            pH=6.5,
            temperature=70.0,
            calcium_hardness=30.0,
            alkalinity=50.0,
            tds=300.0
        )

        assert lsi < 0  # Corrosive

    @pytest.mark.unit
    def test_lsi_invalid_ph_raises_error(self):
        """Test LSI raises error for invalid pH."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_langelier_saturation_index(
                pH=15.0,  # Invalid
                temperature=85.0,
                calcium_hardness=150.0,
                alkalinity=250.0,
                tds=800.0
            )

    @pytest.mark.unit
    def test_lsi_invalid_temperature_raises_error(self):
        """Test LSI raises error for invalid temperature."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_langelier_saturation_index(
                pH=8.5,
                temperature=150.0,  # Invalid (>100C for liquid water)
                calcium_hardness=150.0,
                alkalinity=250.0,
                tds=800.0
            )

    @pytest.mark.unit
    def test_lsi_handles_edge_cases(self):
        """Test LSI handles edge cases."""
        # Very low values should be handled
        lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
            pH=7.0,
            temperature=25.0,
            calcium_hardness=0.5,  # Very low
            alkalinity=0.5,  # Very low
            tds=30.0  # Very low
        )

        assert isinstance(lsi, float)


# ============================================================================
# RSI Calculation Tests
# ============================================================================

class TestRyznarStabilityIndex:
    """Tests for Ryznar Stability Index calculation."""

    @pytest.mark.unit
    def test_rsi_basic_calculation(self):
        """Test basic RSI calculation."""
        rsi = WaterTreatmentTools.calculate_ryznar_stability_index(
            pH=8.5,
            pHs=7.5
        )

        assert isinstance(rsi, float)
        # RSI = 2 * pHs - pH = 2 * 7.5 - 8.5 = 6.5
        assert abs(rsi - 6.5) < 0.1

    @pytest.mark.unit
    def test_rsi_scale_forming(self):
        """Test RSI indicates scale-forming (RSI < 6.0)."""
        rsi = WaterTreatmentTools.calculate_ryznar_stability_index(
            pH=9.0,
            pHs=7.0
        )

        # RSI = 2 * 7.0 - 9.0 = 5.0
        assert rsi < 6.0  # Scale-forming

    @pytest.mark.unit
    def test_rsi_corrosive(self):
        """Test RSI indicates corrosive (RSI > 7.5)."""
        rsi = WaterTreatmentTools.calculate_ryznar_stability_index(
            pH=6.0,
            pHs=7.0
        )

        # RSI = 2 * 7.0 - 6.0 = 8.0
        assert rsi > 7.5  # Corrosive

    @pytest.mark.unit
    def test_rsi_invalid_ph_raises_error(self):
        """Test RSI raises error for invalid pH."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_ryznar_stability_index(
                pH=20.0,  # Invalid
                pHs=7.5
            )


# ============================================================================
# PSI Calculation Tests
# ============================================================================

class TestPuckoriusScalingIndex:
    """Tests for Puckorius Scaling Index calculation."""

    @pytest.mark.unit
    def test_psi_basic_calculation(self):
        """Test basic PSI calculation."""
        psi = WaterTreatmentTools.calculate_puckorius_scaling_index(
            pH=8.5,
            alkalinity=250.0,
            calcium_hardness=150.0,
            temperature=85.0
        )

        assert isinstance(psi, float)

    @pytest.mark.unit
    def test_psi_without_calcium(self):
        """Test PSI calculation without calcium hardness."""
        psi = WaterTreatmentTools.calculate_puckorius_scaling_index(
            pH=8.5,
            alkalinity=250.0
        )

        assert isinstance(psi, float)

    @pytest.mark.unit
    def test_psi_invalid_ph_raises_error(self):
        """Test PSI raises error for invalid pH."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_puckorius_scaling_index(
                pH=-1.0,  # Invalid
                alkalinity=250.0
            )


# ============================================================================
# Larson-Skold Index Tests
# ============================================================================

class TestLarsonSkoldIndex:
    """Tests for Larson-Skold Index calculation."""

    @pytest.mark.unit
    def test_lski_basic_calculation(self):
        """Test basic Larson-Skold Index calculation."""
        lski = WaterTreatmentTools.calculate_larson_skold_index(
            chloride=150.0,
            sulfate=100.0,
            alkalinity=250.0
        )

        assert isinstance(lski, float)
        assert lski >= 0

    @pytest.mark.unit
    def test_lski_low_corrosion_risk(self):
        """Test LSKI indicates low corrosion risk (< 0.2)."""
        lski = WaterTreatmentTools.calculate_larson_skold_index(
            chloride=10.0,
            sulfate=10.0,
            alkalinity=500.0
        )

        assert lski < 0.2  # Low risk

    @pytest.mark.unit
    def test_lski_high_corrosion_risk(self):
        """Test LSKI indicates high corrosion risk (> 1.0)."""
        lski = WaterTreatmentTools.calculate_larson_skold_index(
            chloride=300.0,
            sulfate=200.0,
            alkalinity=100.0
        )

        assert lski > 1.0  # High risk

    @pytest.mark.unit
    def test_lski_negative_values_raise_error(self):
        """Test LSKI raises error for negative values."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_larson_skold_index(
                chloride=-100.0,  # Invalid
                sulfate=100.0,
                alkalinity=250.0
            )


# ============================================================================
# Water Quality Analysis Tests
# ============================================================================

class TestWaterQualityAnalysis:
    """Tests for comprehensive water quality analysis."""

    @pytest.mark.unit
    def test_analyze_water_quality_returns_dataclass(self, standard_water_data):
        """Test analysis returns WaterQualityAnalysis dataclass."""
        result = WaterTreatmentTools.analyze_water_quality(standard_water_data)

        assert isinstance(result, WaterQualityAnalysis)

    @pytest.mark.unit
    def test_analyze_water_quality_has_indices(self, standard_water_data):
        """Test analysis includes all indices."""
        result = WaterTreatmentTools.analyze_water_quality(standard_water_data)

        assert hasattr(result, 'lsi_value')
        assert hasattr(result, 'rsi_value')
        assert hasattr(result, 'psi_value')
        assert hasattr(result, 'larson_skold_index')

    @pytest.mark.unit
    def test_analyze_water_quality_has_tendencies(self, standard_water_data):
        """Test analysis includes scale/corrosion tendencies."""
        result = WaterTreatmentTools.analyze_water_quality(standard_water_data)

        assert hasattr(result, 'scale_tendency')
        assert hasattr(result, 'corrosion_risk')
        assert result.scale_tendency in ['scaling', 'neutral', 'corrosive']
        assert result.corrosion_risk in ['low', 'moderate', 'high', 'severe']

    @pytest.mark.unit
    def test_analyze_water_quality_has_compliance(self, standard_water_data):
        """Test analysis includes compliance status."""
        result = WaterTreatmentTools.analyze_water_quality(standard_water_data)

        assert hasattr(result, 'compliance_status')
        assert result.compliance_status in ['PASS', 'WARNING', 'FAIL']

    @pytest.mark.unit
    def test_analyze_water_quality_has_provenance(self, standard_water_data):
        """Test analysis includes provenance hash."""
        result = WaterTreatmentTools.analyze_water_quality(standard_water_data)

        assert hasattr(result, 'provenance_hash')
        assert len(result.provenance_hash) == 64  # SHA-256 hex

    @pytest.mark.unit
    def test_high_scale_water_detected(self, high_scale_water_data):
        """Test high scale water is correctly identified."""
        result = WaterTreatmentTools.analyze_water_quality(high_scale_water_data)

        assert result.scale_tendency == 'scaling'

    @pytest.mark.unit
    def test_corrosive_water_detected(self, corrosive_water_data):
        """Test corrosive water is correctly identified."""
        result = WaterTreatmentTools.analyze_water_quality(corrosive_water_data)

        assert result.corrosion_risk in ['high', 'severe']


# ============================================================================
# Blowdown Optimization Tests
# ============================================================================

class TestBlowdownOptimization:
    """Tests for blowdown optimization tools."""

    @pytest.mark.unit
    def test_cycles_of_concentration_calculation(self):
        """Test cycles of concentration calculation."""
        cycles = WaterTreatmentTools.calculate_cycles_of_concentration(
            makeup_conductivity=200.0,
            blowdown_conductivity=2000.0
        )

        assert cycles == 10.0

    @pytest.mark.unit
    def test_cycles_zero_makeup_raises_error(self):
        """Test cycles raises error for zero makeup conductivity."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_cycles_of_concentration(
                makeup_conductivity=0.0,
                blowdown_conductivity=2000.0
            )

    @pytest.mark.unit
    def test_blowdown_rate_calculation(self):
        """Test blowdown rate calculation."""
        rate = WaterTreatmentTools.calculate_blowdown_rate(
            steam_rate=5000.0,
            cycles=5.0
        )

        assert isinstance(rate, float)
        assert rate > 0

    @pytest.mark.unit
    def test_blowdown_rate_low_cycles_raises_error(self):
        """Test blowdown rate raises error for cycles <= 1."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_blowdown_rate(
                steam_rate=5000.0,
                cycles=1.0
            )

    @pytest.mark.unit
    def test_blowdown_heat_loss_calculation(self):
        """Test blowdown heat loss calculation."""
        heat_loss = WaterTreatmentTools.calculate_blowdown_heat_loss(
            blowdown_rate=500.0,
            temperature=180.0,
            ambient_temp=25.0
        )

        assert isinstance(heat_loss, float)
        assert heat_loss > 0

    @pytest.mark.unit
    def test_optimize_blowdown_schedule(self, blowdown_data):
        """Test blowdown schedule optimization."""
        result = WaterTreatmentTools.optimize_blowdown_schedule(
            water_data=blowdown_data,
            steam_demand=5000.0
        )

        assert isinstance(result, BlowdownOptimization)
        assert result.optimal_cycles >= 3.0
        assert result.optimal_cycles <= 10.0
        assert result.recommended_blowdown_rate > 0
        assert hasattr(result, 'provenance_hash')


# ============================================================================
# Chemical Dosing Tests
# ============================================================================

class TestChemicalDosing:
    """Tests for chemical dosing calculations."""

    @pytest.mark.unit
    def test_phosphate_dosing_shock(self):
        """Test phosphate shock dosing calculation."""
        dose = WaterTreatmentTools.calculate_phosphate_dosing(
            residual_target=50.0,
            volume=20.0,
            current_level=30.0
        )

        assert isinstance(dose, float)
        assert dose >= 0

    @pytest.mark.unit
    def test_phosphate_dosing_continuous(self):
        """Test phosphate continuous dosing calculation."""
        dose = WaterTreatmentTools.calculate_phosphate_dosing(
            residual_target=50.0,
            volume=20.0,
            current_level=30.0,
            steam_rate=5000.0
        )

        assert isinstance(dose, float)
        assert dose >= 0

    @pytest.mark.unit
    def test_phosphate_at_target_returns_zero(self):
        """Test phosphate dosing returns 0 when at target."""
        dose = WaterTreatmentTools.calculate_phosphate_dosing(
            residual_target=50.0,
            volume=20.0,
            current_level=60.0  # Above target
        )

        assert dose == 0.0

    @pytest.mark.unit
    def test_oxygen_scavenger_dosing(self):
        """Test oxygen scavenger dosing calculation."""
        dose = WaterTreatmentTools.calculate_oxygen_scavenger_dosing(
            dissolved_oxygen=200.0,
            steam_rate=5000.0,
            scavenger_type=ScavengerType.SODIUM_SULFITE
        )

        assert isinstance(dose, float)
        assert dose >= 0

    @pytest.mark.unit
    def test_oxygen_scavenger_different_types(self):
        """Test different oxygen scavenger types."""
        for scavenger_type in ScavengerType:
            dose = WaterTreatmentTools.calculate_oxygen_scavenger_dosing(
                dissolved_oxygen=200.0,
                steam_rate=5000.0,
                scavenger_type=scavenger_type
            )
            assert dose >= 0

    @pytest.mark.unit
    def test_amine_dosing(self):
        """Test amine dosing calculation."""
        dose = WaterTreatmentTools.calculate_amine_dosing(
            condensate_pH_target=8.8,
            steam_rate=5000.0,
            amine_type=AmineType.NEUTRALIZING_AMINE,
            condensate_return_percent=80.0
        )

        assert isinstance(dose, float)
        assert dose >= 0

    @pytest.mark.unit
    def test_amine_invalid_ph_raises_error(self):
        """Test amine dosing raises error for invalid pH target."""
        with pytest.raises(ValueError):
            WaterTreatmentTools.calculate_amine_dosing(
                condensate_pH_target=12.0,  # Invalid
                steam_rate=5000.0
            )

    @pytest.mark.unit
    def test_polymer_dosing(self):
        """Test polymer dosing calculation."""
        dose = WaterTreatmentTools.calculate_polymer_dosing(
            sludge_conditioner_need=50.0,
            water_hardness=150.0,
            steam_rate=5000.0
        )

        assert isinstance(dose, float)
        assert dose >= 0

    @pytest.mark.unit
    def test_optimize_chemical_consumption(
        self, chemical_usage_data, water_quality_data, target_data
    ):
        """Test chemical consumption optimization."""
        result = WaterTreatmentTools.optimize_chemical_consumption(
            current_usage=chemical_usage_data,
            water_quality=water_quality_data,
            targets=target_data
        )

        assert isinstance(result, ChemicalOptimization)
        assert result.phosphate_dosing >= 0
        assert result.oxygen_scavenger_dosing >= 0
        assert result.amine_dosing >= 0
        assert result.polymer_dosing >= 0
        assert 0 <= result.optimization_score <= 100
        assert hasattr(result, 'provenance_hash')


# ============================================================================
# Scale Prediction Tests
# ============================================================================

class TestScalePrediction:
    """Tests for scale formation prediction."""

    @pytest.mark.unit
    def test_caco3_scale_prediction_positive_lsi(self):
        """Test CaCO3 scale prediction with positive LSI."""
        rate = WaterTreatmentTools.predict_calcium_carbonate_scale(
            lsi=1.5,
            temperature=85.0,
            velocity=1.0
        )

        assert rate > 0

    @pytest.mark.unit
    def test_caco3_scale_prediction_negative_lsi(self):
        """Test CaCO3 scale prediction with negative LSI."""
        rate = WaterTreatmentTools.predict_calcium_carbonate_scale(
            lsi=-1.0,
            temperature=85.0,
            velocity=1.0
        )

        assert rate == 0.0  # No scaling

    @pytest.mark.unit
    def test_silica_scale_prediction_low_risk(self):
        """Test silica scale prediction - low risk."""
        risk = WaterTreatmentTools.predict_silica_scale(
            silica_concentration=30.0,
            temperature=85.0,
            pH=8.5
        )

        assert risk == 'low'

    @pytest.mark.unit
    def test_silica_scale_prediction_high_risk(self):
        """Test silica scale prediction - high risk."""
        risk = WaterTreatmentTools.predict_silica_scale(
            silica_concentration=150.0,
            temperature=85.0,
            pH=8.5
        )

        assert risk in ['high', 'severe']


# ============================================================================
# Corrosion Prediction Tests
# ============================================================================

class TestCorrosionPrediction:
    """Tests for corrosion rate prediction."""

    @pytest.mark.unit
    def test_oxygen_corrosion_prediction(self):
        """Test oxygen corrosion prediction."""
        rate = WaterTreatmentTools.predict_oxygen_corrosion(
            dissolved_oxygen=200.0,
            temperature=70.0,
            pH=8.5
        )

        assert isinstance(rate, float)
        assert rate >= 0

    @pytest.mark.unit
    def test_acid_corrosion_prediction_low_ph(self):
        """Test acid corrosion prediction at low pH."""
        rate = WaterTreatmentTools.predict_acid_corrosion(
            pH=5.0,
            temperature=70.0
        )

        assert rate > 0

    @pytest.mark.unit
    def test_acid_corrosion_prediction_neutral_ph(self):
        """Test acid corrosion prediction at neutral pH."""
        rate = WaterTreatmentTools.predict_acid_corrosion(
            pH=7.5,
            temperature=70.0
        )

        assert rate == 0.0

    @pytest.mark.unit
    def test_corrosion_allowance_calculation(self):
        """Test corrosion allowance calculation."""
        allowance = WaterTreatmentTools.calculate_corrosion_allowance(
            material='carbon_steel',
            environment='boiler_water',
            service_life=20.0
        )

        assert isinstance(allowance, float)
        assert allowance > 0


# ============================================================================
# Compliance Checking Tests
# ============================================================================

class TestComplianceChecking:
    """Tests for compliance checking tools."""

    @pytest.mark.unit
    def test_asme_compliance_pass(self):
        """Test ASME compliance - passing conditions."""
        chemistry = {
            'pH': 11.0,
            'tds': 2500,
            'alkalinity': 500,
            'chloride': 200,
            'silica': 100,
            'hardness': 3.0
        }

        result = WaterTreatmentTools.check_asme_compliance(
            water_chemistry=chemistry,
            pressure=10.0
        )

        assert isinstance(result, ComplianceResult)
        assert result.standard == 'ASME'
        assert result.compliance_status in ['PASS', 'WARNING', 'FAIL']
        assert hasattr(result, 'provenance_hash')

    @pytest.mark.unit
    def test_asme_compliance_fail(self):
        """Test ASME compliance - failing conditions."""
        chemistry = {
            'pH': 8.0,  # Too low for low pressure
            'tds': 5000,  # Too high
            'alkalinity': 1000,  # Too high
            'chloride': 500,  # Too high
            'silica': 200,
            'hardness': 10.0
        }

        result = WaterTreatmentTools.check_asme_compliance(
            water_chemistry=chemistry,
            pressure=10.0
        )

        assert result.compliance_status == 'FAIL'
        assert len(result.violations) > 0

    @pytest.mark.unit
    def test_abma_compliance(self):
        """Test ABMA compliance checking."""
        chemistry = {
            'pH': 11.5,
            'tds': 2000,
            'phosphate_residual': 40,
            'sulfite_residual': 30
        }

        result = WaterTreatmentTools.check_abma_guidelines(
            water_chemistry=chemistry,
            boiler_type=BoilerType.FIRE_TUBE
        )

        assert isinstance(result, ComplianceResult)
        assert result.standard == 'ABMA'
        assert hasattr(result, 'provenance_hash')

    @pytest.mark.unit
    def test_treatment_program_validation_phosphate(self):
        """Test phosphate treatment program validation."""
        chemistry = {
            'pH': 11.0,
            'phosphate_residual': 45,
            'sulfite_residual': 30,
            'hardness': 1.0
        }

        result = WaterTreatmentTools.validate_treatment_program(
            program_type='phosphate',
            chemistry=chemistry
        )

        assert isinstance(result, ValidationResult)
        assert result.program_type == 'phosphate'
        assert hasattr(result, 'provenance_hash')

    @pytest.mark.unit
    def test_treatment_program_validation_avt(self):
        """Test all-volatile treatment program validation."""
        chemistry = {
            'pH': 9.4,
            'phosphate_residual': 0,
            'hardness': 0.1
        }

        result = WaterTreatmentTools.validate_treatment_program(
            program_type='all_volatile',
            chemistry=chemistry
        )

        assert isinstance(result, ValidationResult)
        assert result.program_type == 'all_volatile'


# ============================================================================
# Energy and Cost Analysis Tests
# ============================================================================

class TestEnergyCostAnalysis:
    """Tests for energy and cost analysis tools."""

    @pytest.mark.unit
    def test_blowdown_energy_savings(self):
        """Test blowdown energy savings calculation."""
        savings = WaterTreatmentTools.calculate_blowdown_energy_savings(
            before_cycles=3.0,
            after_cycles=8.0,
            steam_cost=50.0,
            steam_rate=5000.0
        )

        assert isinstance(savings, float)
        assert savings > 0  # Higher cycles = less blowdown = savings

    @pytest.mark.unit
    def test_chemical_cost_calculation(self):
        """Test chemical cost calculation."""
        cost = WaterTreatmentTools.calculate_chemical_cost(
            dosing_rates={'phosphate': 2.0, 'sulfite': 1.5},
            chemical_prices={'phosphate': 5.0, 'sulfite': 3.0}
        )

        assert cost == 14.5  # 2*5 + 1.5*3 = 10 + 4.5 = 14.5

    @pytest.mark.unit
    def test_water_treatment_roi(self):
        """Test water treatment ROI calculation."""
        roi = WaterTreatmentTools.calculate_water_treatment_roi(
            costs={'chemical': 10000, 'maintenance': 5000},
            savings={'water': 20000, 'energy': 15000, 'downtime': 10000},
            implementation_cost=50000
        )

        # Net benefit = 45000 - 15000 = 30000
        # ROI = 30000 / 50000 * 100 = 60%
        assert roi == 60.0

    @pytest.mark.unit
    def test_makeup_water_cost(self):
        """Test makeup water cost calculation."""
        cost = WaterTreatmentTools.calculate_makeup_water_cost(
            usage=100.0,
            water_price=1.0,
            treatment_cost=0.5
        )

        assert cost == 150.0  # 100 * 1.5 = 150


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestToolErrorHandling:
    """Test error handling for all tools."""

    @pytest.mark.unit
    def test_lsi_handles_zero_values(self):
        """Test LSI handles zero TDS gracefully."""
        # Should clamp to minimum
        lsi = WaterTreatmentTools.calculate_langelier_saturation_index(
            pH=8.0,
            temperature=25.0,
            calcium_hardness=0.1,
            alkalinity=0.1,
            tds=10.0  # Low but valid
        )

        assert isinstance(lsi, float)

    @pytest.mark.unit
    def test_cycles_handles_equal_conductivity(self):
        """Test cycles handles equal makeup/blowdown conductivity."""
        # Should return 1.0 with warning
        cycles = WaterTreatmentTools.calculate_cycles_of_concentration(
            makeup_conductivity=1000.0,
            blowdown_conductivity=800.0  # Less than makeup
        )

        assert cycles == 1.0


# ============================================================================
# Performance Tests
# ============================================================================

class TestToolPerformance:
    """Performance tests for tools."""

    @pytest.mark.performance
    def test_lsi_calculation_performance(self):
        """Test LSI calculation completes in reasonable time."""
        start = time.perf_counter()

        for _ in range(1000):
            WaterTreatmentTools.calculate_langelier_saturation_index(
                pH=8.5,
                temperature=85.0,
                calcium_hardness=150.0,
                alkalinity=250.0,
                tds=800.0
            )

        elapsed = time.perf_counter() - start
        assert elapsed < 1.0  # 1000 calculations in under 1 second

    @pytest.mark.performance
    def test_water_analysis_performance(self, standard_water_data):
        """Test water analysis completes in reasonable time."""
        start = time.perf_counter()

        for _ in range(100):
            WaterTreatmentTools.analyze_water_quality(standard_water_data)

        elapsed = time.perf_counter() - start
        assert elapsed < 1.0  # 100 analyses in under 1 second


# ============================================================================
# Determinism Tests
# ============================================================================

class TestToolDeterminism:
    """Test tools produce deterministic results."""

    @pytest.mark.determinism
    def test_lsi_deterministic(self):
        """Test LSI produces same results."""
        results = [
            WaterTreatmentTools.calculate_langelier_saturation_index(
                pH=8.5, temperature=85.0, calcium_hardness=150.0,
                alkalinity=250.0, tds=800.0
            )
            for _ in range(10)
        ]

        assert all(r == results[0] for r in results)

    @pytest.mark.determinism
    def test_water_analysis_deterministic(self, standard_water_data):
        """Test water analysis produces same results."""
        results = [
            WaterTreatmentTools.analyze_water_quality(standard_water_data)
            for _ in range(10)
        ]

        # All LSI values should match
        assert all(r.lsi_value == results[0].lsi_value for r in results)

        # All RSI values should match
        assert all(r.rsi_value == results[0].rsi_value for r in results)


# ============================================================================
# Integration Tests
# ============================================================================

class TestToolIntegration:
    """Integration tests for tool workflows."""

    @pytest.mark.integration
    def test_full_water_analysis_workflow(self, standard_water_data):
        """Test complete water analysis workflow."""
        # Step 1: Get water quality analysis
        analysis = WaterTreatmentTools.analyze_water_quality(standard_water_data)
        assert analysis is not None

        # Step 2: Based on LSI, predict scale
        scale_rate = WaterTreatmentTools.predict_calcium_carbonate_scale(
            lsi=analysis.lsi_value,
            temperature=standard_water_data['temperature'],
            velocity=1.0
        )
        assert scale_rate >= 0

        # Step 3: Predict corrosion
        corrosion_rate = WaterTreatmentTools.predict_oxygen_corrosion(
            dissolved_oxygen=200.0,
            temperature=standard_water_data['temperature'],
            pH=standard_water_data['pH']
        )
        assert corrosion_rate >= 0

    @pytest.mark.integration
    def test_treatment_optimization_workflow(self, blowdown_data):
        """Test treatment optimization workflow."""
        # Step 1: Optimize blowdown
        blowdown_opt = WaterTreatmentTools.optimize_blowdown_schedule(
            water_data=blowdown_data,
            steam_demand=5000.0
        )
        assert blowdown_opt is not None

        # Step 2: Calculate energy savings
        savings = WaterTreatmentTools.calculate_blowdown_energy_savings(
            before_cycles=3.0,
            after_cycles=blowdown_opt.optimal_cycles,
            steam_cost=50.0,
            steam_rate=5000.0
        )
        assert savings >= 0

    @pytest.mark.integration
    def test_compliance_and_validation_workflow(self):
        """Test compliance checking and program validation workflow."""
        chemistry = {
            'pH': 11.0,
            'tds': 2500,
            'alkalinity': 500,
            'chloride': 200,
            'silica': 100,
            'hardness': 1.0,
            'phosphate_residual': 45,
            'sulfite_residual': 30
        }

        # Step 1: Check ASME compliance
        asme_result = WaterTreatmentTools.check_asme_compliance(
            water_chemistry=chemistry,
            pressure=10.0
        )
        assert asme_result is not None

        # Step 2: Check ABMA compliance
        abma_result = WaterTreatmentTools.check_abma_guidelines(
            water_chemistry=chemistry,
            boiler_type=BoilerType.WATER_TUBE
        )
        assert abma_result is not None

        # Step 3: Validate treatment program
        validation = WaterTreatmentTools.validate_treatment_program(
            program_type='phosphate',
            chemistry=chemistry
        )
        assert validation is not None
