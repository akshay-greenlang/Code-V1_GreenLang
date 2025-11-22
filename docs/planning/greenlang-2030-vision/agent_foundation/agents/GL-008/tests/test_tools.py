# -*- coding: utf-8 -*-
"""
Comprehensive test suite for GL-008 tools.py

Tests all 7 deterministic calculation tools with:
- Unit tests for each tool
- Input validation
- Output schema validation
- Edge cases and boundary conditions
- Determinism verification
- Performance benchmarks
"""

import pytest
import numpy as np
from typing import Dict, Any
import time

# Import tools
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools import (
    SteamTrapTools,
    AcousticAnalysisResult,
    ThermalAnalysisResult,
    FailureDiagnosisResult,
    EnergyLossResult,
    MaintenancePriorityResult,
    RULPredictionResult,
    CostBenefitResult
)
from config import TrapType, FailureMode


@pytest.fixture
def tools():
    """Fixture providing SteamTrapTools instance."""
    return SteamTrapTools()


# ============================================================================
# TEST: ACOUSTIC SIGNATURE ANALYSIS
# ============================================================================

class TestAcousticAnalysis:
    """Test suite for analyze_acoustic_signature tool."""

    def test_acoustic_analysis_basic(self, tools):
        """Test basic acoustic analysis with normal signal."""
        # Create synthetic acoustic signal
        sampling_rate = 250000
        duration = 1.0
        n_samples = int(sampling_rate * duration)
        signal = np.random.randn(n_samples) * 0.1  # Low amplitude = normal

        acoustic_data = {
            'trap_id': 'TRAP-TEST-001',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data, TrapType.THERMODYNAMIC)

        # Assertions
        assert isinstance(result, AcousticAnalysisResult)
        assert result.trap_id == 'TRAP-TEST-001'
        assert 0.0 <= result.failure_probability <= 1.0
        assert 0.0 <= result.confidence_score <= 1.0
        assert isinstance(result.failure_mode, FailureMode)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_acoustic_analysis_failed_open(self, tools):
        """Test acoustic analysis with failed open signature."""
        # High amplitude signal at 30 kHz = failed open
        sampling_rate = 250000
        duration = 1.0
        t = np.linspace(0, duration, int(sampling_rate * duration))
        signal = np.sin(2 * np.pi * 30000 * t) * 2.0  # High amplitude

        acoustic_data = {
            'trap_id': 'TRAP-TEST-002',
            'signal': signal.tolist(),
            'sampling_rate_hz': sampling_rate
        }

        result = tools.analyze_acoustic_signature(acoustic_data)

        # Should detect potential failure
        assert result.signal_strength_db > 50.0
        assert result.failure_probability > 0.5

    def test_acoustic_determinism(self, tools):
        """Test that acoustic analysis is deterministic."""
        signal = np.random.randn(10000) * 0.2

        acoustic_data = {
            'trap_id': 'TRAP-DET-001',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        # Run twice with identical input
        result1 = tools.analyze_acoustic_signature(acoustic_data)
        result2 = tools.analyze_acoustic_signature(acoustic_data)

        # Results must be identical
        assert result1.failure_probability == result2.failure_probability
        assert result1.confidence_score == result2.confidence_score
        assert result1.signal_strength_db == result2.signal_strength_db
        assert result1.frequency_peak_hz == result2.frequency_peak_hz

    def test_acoustic_edge_cases(self, tools):
        """Test edge cases for acoustic analysis."""
        # Empty signal
        with pytest.raises((ValueError, IndexError)):
            tools.analyze_acoustic_signature({'trap_id': 'TEST', 'signal': []})

        # Single sample
        result = tools.analyze_acoustic_signature({
            'trap_id': 'TEST',
            'signal': [0.5],
            'sampling_rate_hz': 250000
        })
        assert result is not None


# ============================================================================
# TEST: THERMAL PATTERN ANALYSIS
# ============================================================================

class TestThermalAnalysis:
    """Test suite for analyze_thermal_pattern tool."""

    def test_thermal_analysis_normal(self, tools):
        """Test thermal analysis with normal temperature differential."""
        thermal_data = {
            'trap_id': 'TRAP-THERMAL-001',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0,  # ΔT = 20°C (normal)
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Assertions
        assert isinstance(result, ThermalAnalysisResult)
        assert result.trap_id == 'TRAP-THERMAL-001'
        assert result.temperature_differential_c == 20.0
        assert 0 <= result.trap_health_score <= 100
        assert len(result.anomalies_detected) > 0

    def test_thermal_analysis_failed_closed(self, tools):
        """Test thermal analysis with condensate backup signature."""
        thermal_data = {
            'trap_id': 'TRAP-THERMAL-002',
            'temperature_upstream_c': 180.0,
            'temperature_downstream_c': 70.0,  # ΔT = 110°C (excessive)
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect condensate pooling
        assert result.temperature_differential_c > 50.0
        assert result.condensate_pooling_detected == True

    def test_thermal_analysis_failed_open(self, tools):
        """Test thermal analysis with steam bypass signature."""
        thermal_data = {
            'trap_id': 'TRAP-THERMAL-003',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 148.0,  # ΔT = 2°C (minimal)
            'ambient_temp_c': 20.0
        }

        result = tools.analyze_thermal_pattern(thermal_data)

        # Should detect minimal differential
        assert result.temperature_differential_c < 5.0
        assert result.condensate_pooling_detected == False

    def test_thermal_determinism(self, tools):
        """Test that thermal analysis is deterministic."""
        thermal_data = {
            'trap_id': 'TRAP-DET-002',
            'temperature_upstream_c': 145.0,
            'temperature_downstream_c': 125.0
        }

        result1 = tools.analyze_thermal_pattern(thermal_data)
        result2 = tools.analyze_thermal_pattern(thermal_data)

        assert result1.trap_health_score == result2.trap_health_score
        assert result1.temperature_differential_c == result2.temperature_differential_c


# ============================================================================
# TEST: FAILURE DIAGNOSIS
# ============================================================================

class TestFailureDiagnosis:
    """Test suite for diagnose_trap_failure tool."""

    def test_diagnosis_with_all_inputs(self, tools):
        """Test diagnosis with acoustic, thermal, and operational data."""
        # Create mock results
        from dataclasses import dataclass

        # Mock acoustic result
        acoustic_result = AcousticAnalysisResult(
            trap_id='TRAP-001',
            failure_probability=0.85,
            failure_mode=FailureMode.FAILED_OPEN,
            confidence_score=0.90,
            acoustic_signature={},
            anomaly_detected=True,
            signal_strength_db=65.0,
            frequency_peak_hz=32000,
            spectral_features={},
            timestamp='2025-01-22T12:00:00',
            provenance_hash='abc123'
        )

        # Mock thermal result
        thermal_result = ThermalAnalysisResult(
            trap_id='TRAP-001',
            trap_health_score=30.0,
            temperature_upstream_c=150.0,
            temperature_downstream_c=148.0,
            temperature_differential_c=2.0,
            anomalies_detected=['Minimal temperature differential'],
            hot_spots=[],
            cold_spots=[],
            thermal_signature={},
            condensate_pooling_detected=False,
            timestamp='2025-01-22T12:00:00',
            provenance_hash='def456'
        )

        sensor_data = {
            'trap_id': 'TRAP-001',
            'pressure_upstream_psig': 100.0,
            'pressure_downstream_psig': 2.0
        }

        result = tools.diagnose_trap_failure(
            sensor_data,
            acoustic_result,
            thermal_result
        )

        # Assertions
        assert isinstance(result, FailureDiagnosisResult)
        assert result.trap_id == 'TRAP-001'
        assert isinstance(result.failure_mode, FailureMode)
        assert result.failure_severity in ['normal', 'low', 'medium', 'high', 'critical']
        assert 0.0 <= result.confidence <= 1.0
        assert result.urgency_hours > 0

    def test_diagnosis_severity_levels(self, tools):
        """Test diagnosis produces correct severity levels."""
        sensor_data = {'trap_id': 'TEST', 'pressure_upstream_psig': 100.0}

        # Critical failure
        acoustic_critical = AcousticAnalysisResult(
            trap_id='TEST', failure_probability=0.95,
            failure_mode=FailureMode.FAILED_OPEN,
            confidence_score=0.95, acoustic_signature={},
            anomaly_detected=True, signal_strength_db=70.0,
            frequency_peak_hz=30000, spectral_features={},
            timestamp='2025-01-22', provenance_hash='abc'
        )

        result = tools.diagnose_trap_failure(sensor_data, acoustic_critical, None)
        assert result.failure_severity in ['critical', 'high']


# ============================================================================
# TEST: ENERGY LOSS CALCULATION
# ============================================================================

class TestEnergyLossCalculation:
    """Test suite for calculate_energy_loss tool."""

    def test_energy_loss_basic(self, tools):
        """Test basic energy loss calculation."""
        trap_data = {
            'trap_id': 'TRAP-ENERGY-001',
            'orifice_diameter_in': 0.125,  # 1/8 inch
            'steam_pressure_psig': 100.0,
            'operating_hours_yr': 8760,
            'steam_cost_usd_per_1000lb': 8.50,
            'failure_severity': 1.0  # Complete failure
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Assertions
        assert isinstance(result, EnergyLossResult)
        assert result.steam_loss_kg_hr > 0
        assert result.steam_loss_lb_hr > 0
        assert result.energy_loss_gj_yr > 0
        assert result.cost_loss_usd_yr > 0
        assert result.co2_emissions_kg_yr > 0

    def test_energy_loss_napier_equation(self, tools):
        """Test Napier equation implementation."""
        # Known values for verification
        # W = 24.24 * P * D² * C
        # For P=100, D=0.125, C=0.7:
        # W = 24.24 * 100 * 0.015625 * 0.7 = 26.5125 lb/hr

        trap_data = {
            'trap_id': 'TRAP-NAPIER',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'failure_severity': 1.0
        }

        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        # Verify calculation
        expected_steam_loss = 24.24 * 100 * (0.125 ** 2) * 0.7
        assert abs(result.steam_loss_lb_hr - expected_steam_loss) < 0.01

    def test_energy_loss_determinism(self, tools):
        """Test energy loss calculation determinism."""
        trap_data = {
            'trap_id': 'TRAP-DET-003',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0
        }

        result1 = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        result2 = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)

        assert result1.steam_loss_kg_hr == result2.steam_loss_kg_hr
        assert result1.cost_loss_usd_yr == result2.cost_loss_usd_yr
        assert result1.co2_emissions_kg_yr == result2.co2_emissions_kg_yr


# ============================================================================
# TEST: MAINTENANCE PRIORITIZATION
# ============================================================================

class TestMaintenancePrioritization:
    """Test suite for prioritize_maintenance tool."""

    def test_prioritization_basic(self, tools):
        """Test basic fleet prioritization."""
        trap_fleet = [
            {
                'trap_id': 'TRAP-001',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 15000,
                'process_criticality': 9,
                'current_age_years': 10
            },
            {
                'trap_id': 'TRAP-002',
                'failure_mode': FailureMode.LEAKING,
                'energy_loss_usd_yr': 3000,
                'process_criticality': 5,
                'current_age_years': 3
            },
            {
                'trap_id': 'TRAP-003',
                'failure_mode': FailureMode.NORMAL,
                'energy_loss_usd_yr': 0,
                'process_criticality': 7,
                'current_age_years': 1
            }
        ]

        result = tools.prioritize_maintenance(trap_fleet)

        # Assertions
        assert isinstance(result, MaintenancePriorityResult)
        assert len(result.priority_list) == 3
        assert result.total_potential_savings_usd_yr > 0
        assert result.payback_months > 0
        assert len(result.recommended_schedule) > 0

        # Verify sorting (highest priority first)
        assert result.priority_list[0]['priority_score'] >= result.priority_list[1]['priority_score']

    def test_prioritization_roi_calculation(self, tools):
        """Test ROI calculation in prioritization."""
        trap_fleet = [
            {
                'trap_id': 'TRAP-HIGH-ROI',
                'failure_mode': FailureMode.FAILED_OPEN,
                'energy_loss_usd_yr': 20000,
                'process_criticality': 8,
                'current_age_years': 5
            }
        ]

        result = tools.prioritize_maintenance(trap_fleet)

        # High savings should result in positive ROI
        assert result.expected_roi_percent > 0
        assert result.payback_months < 24  # Less than 2 years


# ============================================================================
# TEST: RUL PREDICTION
# ============================================================================

class TestRULPrediction:
    """Test suite for predict_remaining_useful_life tool."""

    def test_rul_prediction_basic(self, tools):
        """Test basic RUL prediction."""
        condition_data = {
            'trap_id': 'TRAP-RUL-001',
            'current_age_days': 1000,
            'degradation_rate': 0.1,
            'current_health_score': 70
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # Assertions
        assert isinstance(result, RULPredictionResult)
        assert result.rul_days > 0
        assert result.rul_confidence_lower < result.rul_days < result.rul_confidence_upper
        assert len(result.failure_probability_curve) > 0

    def test_rul_weibull_calculation(self, tools):
        """Test Weibull distribution implementation."""
        condition_data = {
            'trap_id': 'TRAP-WEIBULL',
            'current_age_days': 500,
            'current_health_score': 80,
            'historical_failures': [1800, 2000, 2200]  # MTBF ≈ 2000 days
        }

        result = tools.predict_remaining_useful_life(condition_data)

        # RUL should be positive
        assert result.rul_days > 0

        # Historical MTBF should be used
        assert result.historical_mtbf_days is not None
        assert 1800 <= result.historical_mtbf_days <= 2200


# ============================================================================
# TEST: COST-BENEFIT ANALYSIS
# ============================================================================

class TestCostBenefitAnalysis:
    """Test suite for calculate_cost_benefit tool."""

    def test_cost_benefit_repair(self, tools):
        """Test cost-benefit analysis for repair action."""
        maintenance_plan = {
            'trap_id': 'TRAP-CBA-001',
            'action': 'repair',
            'annual_energy_loss_usd': 5000,
            'maintenance_cost_usd': 150,
            'expected_service_life_years': 5
        }

        result = tools.calculate_cost_benefit(maintenance_plan)

        # Assertions
        assert isinstance(result, CostBenefitResult)
        assert result.maintenance_cost_usd > 0
        assert result.annual_savings_usd > 0
        assert result.payback_months > 0
        assert result.npv_usd != 0

    def test_cost_benefit_replace(self, tools):
        """Test cost-benefit analysis for replacement action."""
        maintenance_plan = {
            'trap_id': 'TRAP-CBA-002',
            'action': 'replace',
            'annual_energy_loss_usd': 15000,
            'replacement_cost_usd': 500,
            'expected_service_life_years': 10
        }

        result = tools.calculate_cost_benefit(maintenance_plan)

        # High savings should result in positive NPV
        assert result.npv_usd > 0
        assert result.roi_percent > 0
        assert result.payback_months < 12  # Less than 1 year

    def test_cost_benefit_npv_calculation(self, tools):
        """Test NPV calculation accuracy."""
        maintenance_plan = {
            'trap_id': 'TRAP-NPV',
            'action': 'repair',
            'annual_energy_loss_usd': 1000,
            'maintenance_cost_usd': 100,
            'expected_service_life_years': 5,
            'discount_rate': 0.08
        }

        result = tools.calculate_cost_benefit(maintenance_plan)

        # NPV should be positive (savings > cost)
        assert result.npv_usd > 0


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""

    def test_acoustic_analysis_performance(self, tools, benchmark):
        """Benchmark acoustic analysis performance."""
        signal = np.random.randn(10000) * 0.2
        acoustic_data = {
            'trap_id': 'BENCHMARK',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        }

        # Should complete in < 1 second
        start = time.perf_counter()
        result = tools.analyze_acoustic_signature(acoustic_data)
        duration = time.perf_counter() - start

        assert duration < 1.0  # Less than 1 second

    def test_energy_loss_performance(self, tools):
        """Benchmark energy loss calculation performance."""
        trap_data = {
            'trap_id': 'BENCHMARK',
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0
        }

        # Should complete in < 10ms
        start = time.perf_counter()
        result = tools.calculate_energy_loss(trap_data, FailureMode.FAILED_OPEN)
        duration = time.perf_counter() - start

        assert duration < 0.01  # Less than 10ms


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_inspection_workflow(self, tools):
        """Test complete trap inspection workflow."""
        # Step 1: Acoustic analysis
        signal = np.random.randn(10000) * 0.5
        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': 'WORKFLOW-001',
            'signal': signal.tolist(),
            'sampling_rate_hz': 250000
        })

        # Step 2: Thermal analysis
        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': 'WORKFLOW-001',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 148.0
        })

        # Step 3: Diagnosis
        diagnosis_result = tools.diagnose_trap_failure(
            {'trap_id': 'WORKFLOW-001', 'pressure_upstream_psig': 100.0},
            acoustic_result,
            thermal_result
        )

        # Step 4: Energy loss (if failure detected)
        if diagnosis_result.failure_mode != FailureMode.NORMAL:
            energy_result = tools.calculate_energy_loss(
                {
                    'trap_id': 'WORKFLOW-001',
                    'orifice_diameter_in': 0.125,
                    'steam_pressure_psig': 100.0,
                    'failure_severity': diagnosis_result.confidence
                },
                diagnosis_result.failure_mode
            )

            # Verify complete workflow
            assert energy_result.cost_loss_usd_yr >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=tools", "--cov-report=html"])
