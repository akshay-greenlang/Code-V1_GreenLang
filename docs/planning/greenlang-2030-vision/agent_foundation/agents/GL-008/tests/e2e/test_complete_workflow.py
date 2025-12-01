# -*- coding: utf-8 -*-
"""
End-to-end workflow tests for GL-008 TRAPCATCHER SteamTrapInspector.

This module tests complete inspection workflows from raw sensor input
through diagnosis, energy loss calculation, and maintenance recommendation.
Validates the full agent lifecycle and data flow integrity.
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from steam_trap_inspector import SteamTrapInspector
from tools import SteamTrapTools
from config import TrapInspectorConfig, TrapType, FailureMode


@pytest.fixture
def inspector_config():
    """Create test inspector configuration."""
    return TrapInspectorConfig(
        agent_id="GL-008-E2E-TEST",
        enable_llm_classification=False,
        cache_ttl_seconds=60,
        max_concurrent_inspections=10,
        llm_temperature=0.0,
        llm_seed=42
    )


@pytest.fixture
def inspector(inspector_config):
    """Create SteamTrapInspector instance for testing."""
    return SteamTrapInspector(inspector_config)


@pytest.fixture
def tools():
    """Create SteamTrapTools instance for testing."""
    return SteamTrapTools()


@pytest.fixture
def sample_trap_data():
    """Generate sample trap data for e2e testing."""
    np.random.seed(42)
    return {
        'trap_id': 'TRAP-E2E-001',
        'trap_type': TrapType.THERMODYNAMIC,
        'location': 'Building A - Level 2 - Station 5',
        'acoustic_data': {
            'signal': (np.random.randn(50000) * 0.3).tolist(),
            'sampling_rate_hz': 250000
        },
        'thermal_data': {
            'temperature_upstream_c': 155.0,
            'temperature_downstream_c': 148.0,
            'ambient_temp_c': 22.0
        },
        'operational_data': {
            'steam_pressure_psig': 100.0,
            'orifice_diameter_in': 0.125,
            'operating_hours_yr': 8760,
            'steam_cost_usd_per_1000lb': 8.50
        },
        'maintenance_history': {
            'install_date': '2020-01-15',
            'last_inspection_date': '2024-06-01',
            'current_age_days': 1800,
            'health_score': 75
        }
    }


@pytest.mark.e2e
class TestSingleTrapInspectionWorkflow:
    """Test complete single trap inspection workflow."""

    def test_full_inspection_normal_trap(self, tools, sample_trap_data):
        """Test full inspection workflow for a normal operating trap."""
        # Step 1: Acoustic Analysis
        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': sample_trap_data['trap_id'],
            'signal': sample_trap_data['acoustic_data']['signal'],
            'sampling_rate_hz': sample_trap_data['acoustic_data']['sampling_rate_hz']
        }, sample_trap_data['trap_type'])

        assert acoustic_result is not None
        assert acoustic_result.trap_id == sample_trap_data['trap_id']
        assert acoustic_result.provenance_hash is not None
        assert len(acoustic_result.provenance_hash) == 64

        # Step 2: Thermal Analysis
        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': sample_trap_data['trap_id'],
            **sample_trap_data['thermal_data']
        })

        assert thermal_result is not None
        assert thermal_result.trap_id == sample_trap_data['trap_id']
        assert thermal_result.temperature_differential_c == 7.0

        # Step 3: Diagnosis
        diagnosis_result = tools.diagnose_trap_failure(
            {
                'trap_id': sample_trap_data['trap_id'],
                'pressure_upstream_psig': sample_trap_data['operational_data']['steam_pressure_psig']
            },
            acoustic_result,
            thermal_result
        )

        assert diagnosis_result is not None
        assert isinstance(diagnosis_result.failure_mode, FailureMode)
        assert 0.0 <= diagnosis_result.confidence <= 1.0

        # Step 4: Energy Loss Calculation (if failure detected)
        if diagnosis_result.failure_mode != FailureMode.NORMAL:
            energy_result = tools.calculate_energy_loss(
                {
                    'trap_id': sample_trap_data['trap_id'],
                    'orifice_diameter_in': sample_trap_data['operational_data']['orifice_diameter_in'],
                    'steam_pressure_psig': sample_trap_data['operational_data']['steam_pressure_psig'],
                    'steam_cost_usd_per_1000lb': sample_trap_data['operational_data']['steam_cost_usd_per_1000lb'],
                    'operating_hours_yr': sample_trap_data['operational_data']['operating_hours_yr'],
                    'failure_severity': diagnosis_result.confidence
                },
                diagnosis_result.failure_mode
            )

            assert energy_result.steam_loss_kg_hr >= 0
            assert energy_result.cost_loss_usd_yr >= 0
            assert energy_result.co2_emissions_kg_yr >= 0

        # Step 5: RUL Prediction
        rul_result = tools.predict_remaining_useful_life({
            'trap_id': sample_trap_data['trap_id'],
            'current_age_days': sample_trap_data['maintenance_history']['current_age_days'],
            'current_health_score': sample_trap_data['maintenance_history']['health_score'],
            'degradation_rate': 0.1
        })

        assert rul_result.rul_days > 0
        assert rul_result.rul_confidence_lower <= rul_result.rul_days
        assert rul_result.rul_days <= rul_result.rul_confidence_upper

    def test_full_inspection_failed_open_trap(self, tools):
        """Test full inspection workflow for a failed open trap."""
        np.random.seed(42)
        sampling_rate = 250000
        t = np.linspace(0, 1.0, sampling_rate)

        # Failed open signature: high amplitude, high frequency
        signal = 2.5 * np.sin(2 * np.pi * 32000 * t) + np.random.randn(sampling_rate) * 0.3

        trap_data = {
            'trap_id': 'TRAP-E2E-FAILED-OPEN',
            'acoustic_data': {'signal': signal.tolist(), 'sampling_rate_hz': sampling_rate},
            'thermal_data': {
                'temperature_upstream_c': 150.0,
                'temperature_downstream_c': 148.5,  # Minimal differential = steam bypass
                'ambient_temp_c': 20.0
            },
            'operational_data': {
                'steam_pressure_psig': 100.0,
                'orifice_diameter_in': 0.125
            }
        }

        # Execute full workflow
        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': trap_data['trap_id'],
            'signal': trap_data['acoustic_data']['signal'],
            'sampling_rate_hz': trap_data['acoustic_data']['sampling_rate_hz']
        })

        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': trap_data['trap_id'],
            **trap_data['thermal_data']
        })

        diagnosis_result = tools.diagnose_trap_failure(
            {'trap_id': trap_data['trap_id'], 'pressure_upstream_psig': 100.0},
            acoustic_result,
            thermal_result
        )

        # Should detect failure
        assert acoustic_result.signal_strength_db > 50.0
        assert thermal_result.temperature_differential_c < 5.0

        # Calculate energy loss
        energy_result = tools.calculate_energy_loss(
            {
                'trap_id': trap_data['trap_id'],
                'orifice_diameter_in': trap_data['operational_data']['orifice_diameter_in'],
                'steam_pressure_psig': trap_data['operational_data']['steam_pressure_psig'],
                'failure_severity': 1.0
            },
            FailureMode.FAILED_OPEN
        )

        assert energy_result.steam_loss_kg_hr > 0
        assert energy_result.cost_loss_usd_yr > 0
        assert energy_result.co2_emissions_kg_yr > 0

    def test_full_inspection_failed_closed_trap(self, tools):
        """Test full inspection workflow for a failed closed trap."""
        np.random.seed(42)

        # Failed closed: minimal acoustic signal
        signal = (np.random.randn(10000) * 0.02).tolist()

        trap_data = {
            'trap_id': 'TRAP-E2E-FAILED-CLOSED',
            'acoustic_data': {'signal': signal, 'sampling_rate_hz': 250000},
            'thermal_data': {
                'temperature_upstream_c': 180.0,
                'temperature_downstream_c': 65.0,  # Large differential = condensate backup
                'ambient_temp_c': 20.0
            }
        }

        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': trap_data['trap_id'],
            'signal': trap_data['acoustic_data']['signal'],
            'sampling_rate_hz': trap_data['acoustic_data']['sampling_rate_hz']
        })

        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': trap_data['trap_id'],
            **trap_data['thermal_data']
        })

        # Verify indicators of failed closed
        assert thermal_result.temperature_differential_c > 80.0
        assert thermal_result.condensate_pooling_detected == True
        assert thermal_result.trap_health_score < 40.0


@pytest.mark.e2e
class TestMultiTrapWorkflow:
    """Test end-to-end workflow for multiple traps (fleet)."""

    def test_fleet_inspection_workflow(self, tools):
        """Test complete fleet inspection and prioritization workflow."""
        np.random.seed(42)

        # Generate fleet of 10 traps with varying conditions
        fleet = []
        for i in range(10):
            trap = {
                'trap_id': f'TRAP-FLEET-{i:03d}',
                'failure_mode': [FailureMode.NORMAL, FailureMode.FAILED_OPEN,
                               FailureMode.LEAKING, FailureMode.FAILED_CLOSED][i % 4],
                'energy_loss_usd_yr': max(0, 15000 - i * 1000) if i % 4 != 0 else 0,
                'process_criticality': 10 - (i % 5),
                'current_age_years': 1 + (i % 12),
                'health_score': max(20, 90 - i * 5)
            }
            fleet.append(trap)

        # Prioritize maintenance
        result = tools.prioritize_maintenance(fleet)

        # Validate results
        assert len(result.priority_list) == 10
        assert result.total_potential_savings_usd_yr > 0
        assert result.expected_roi_percent > 0
        assert result.payback_months > 0

        # Verify priority ordering (highest priority first)
        for i in range(len(result.priority_list) - 1):
            assert result.priority_list[i]['priority_score'] >= result.priority_list[i + 1]['priority_score']

        # Verify schedule generation
        if hasattr(result, 'recommended_schedule'):
            assert len(result.recommended_schedule) > 0

    def test_batch_inspection_workflow(self, tools):
        """Test batch processing of multiple trap inspections."""
        np.random.seed(42)

        batch_results = []
        num_traps = 5

        for i in range(num_traps):
            signal = (np.random.randn(10000) * (0.1 + i * 0.1)).tolist()

            acoustic_result = tools.analyze_acoustic_signature({
                'trap_id': f'TRAP-BATCH-{i:03d}',
                'signal': signal,
                'sampling_rate_hz': 250000
            })

            thermal_result = tools.analyze_thermal_pattern({
                'trap_id': f'TRAP-BATCH-{i:03d}',
                'temperature_upstream_c': 150.0,
                'temperature_downstream_c': 150.0 - (i * 5),
                'ambient_temp_c': 20.0
            })

            batch_results.append({
                'trap_id': f'TRAP-BATCH-{i:03d}',
                'acoustic': acoustic_result,
                'thermal': thermal_result
            })

        # All results should be valid
        assert len(batch_results) == num_traps
        for result in batch_results:
            assert result['acoustic'].provenance_hash is not None
            assert result['thermal'].provenance_hash is not None


@pytest.mark.e2e
@pytest.mark.asyncio
class TestAsyncWorkflow:
    """Test asynchronous end-to-end workflows."""

    async def test_async_single_inspection(self, inspector):
        """Test async inspection execution."""
        np.random.seed(42)

        input_data = {
            'operation_mode': 'monitor',
            'trap_data': {
                'trap_id': 'TRAP-ASYNC-001',
                'trap_type': 'thermodynamic',
                'acoustic_data': {
                    'signal': (np.random.randn(5000) * 0.2).tolist(),
                    'sampling_rate_hz': 250000
                },
                'thermal_data': {
                    'temperature_upstream_c': 150.0,
                    'temperature_downstream_c': 130.0,
                    'ambient_temp_c': 20.0
                },
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0
            }
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'monitor'
        assert 'trap_status' in result
        assert 'analysis_results' in result
        assert result['deterministic'] == True

    async def test_async_diagnose_mode(self, inspector):
        """Test async diagnose operation mode."""
        np.random.seed(42)

        input_data = {
            'operation_mode': 'diagnose',
            'trap_data': {
                'trap_id': 'TRAP-ASYNC-DIAG',
                'acoustic_data': {
                    'signal': (np.random.randn(5000) * 0.5).tolist(),
                    'sampling_rate_hz': 250000
                },
                'thermal_data': {
                    'temperature_upstream_c': 180.0,
                    'temperature_downstream_c': 70.0
                },
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0
            }
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'diagnose'
        assert 'diagnostic_summary' in result
        assert 'impact_assessment' in result
        assert 'corrective_actions' in result

    async def test_async_predict_mode(self, inspector):
        """Test async predict operation mode."""
        input_data = {
            'operation_mode': 'predict',
            'trap_data': {
                'trap_id': 'TRAP-ASYNC-PRED',
                'current_age_days': 1500,
                'degradation_rate': 0.12,
                'current_health_score': 65,
                'process_criticality': 8
            }
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'predict'
        assert 'predictive_maintenance' in result
        assert 'maintenance_schedule' in result

    async def test_async_prioritize_mode(self, inspector):
        """Test async prioritize operation mode."""
        input_data = {
            'operation_mode': 'prioritize',
            'fleet_data': [
                {
                    'trap_id': f'TRAP-PRI-{i:03d}',
                    'failure_mode': 'failed_open' if i % 2 == 0 else 'leaking',
                    'energy_loss_usd_yr': 10000 - i * 500,
                    'process_criticality': 10 - (i % 5),
                    'current_age_years': 2 + (i % 10)
                }
                for i in range(8)
            ]
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'prioritize'
        assert 'fleet_summary' in result
        assert 'prioritized_maintenance_plan' in result
        assert 'financial_summary' in result


@pytest.mark.e2e
class TestWorkflowDataIntegrity:
    """Test data integrity throughout the workflow."""

    def test_provenance_chain_integrity(self, tools):
        """Test that provenance hashes form a valid chain."""
        np.random.seed(42)
        signal = (np.random.randn(10000) * 0.2).tolist()

        # Step 1: Acoustic
        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': 'TRAP-PROV-CHAIN',
            'signal': signal,
            'sampling_rate_hz': 250000
        })
        acoustic_hash = acoustic_result.provenance_hash

        # Step 2: Thermal
        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': 'TRAP-PROV-CHAIN',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0
        })
        thermal_hash = thermal_result.provenance_hash

        # Each step should have a unique provenance hash
        assert acoustic_hash is not None
        assert thermal_hash is not None
        assert len(acoustic_hash) == 64
        assert len(thermal_hash) == 64

        # Hashes should be different for different operations
        assert acoustic_hash != thermal_hash

    def test_trap_id_consistency(self, tools):
        """Test that trap_id is preserved throughout workflow."""
        trap_id = 'TRAP-ID-CONSISTENCY-TEST'
        np.random.seed(42)

        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': trap_id,
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        })

        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': trap_id,
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0
        })

        diagnosis_result = tools.diagnose_trap_failure(
            {'trap_id': trap_id, 'pressure_upstream_psig': 100.0},
            acoustic_result,
            thermal_result
        )

        # All results should have the same trap_id
        assert acoustic_result.trap_id == trap_id
        assert thermal_result.trap_id == trap_id
        assert diagnosis_result.trap_id == trap_id


@pytest.mark.e2e
class TestWorkflowErrorHandling:
    """Test error handling in end-to-end workflows."""

    def test_missing_acoustic_data(self, tools):
        """Test workflow handles missing acoustic data gracefully."""
        thermal_result = tools.analyze_thermal_pattern({
            'trap_id': 'TRAP-NO-ACOUSTIC',
            'temperature_upstream_c': 150.0,
            'temperature_downstream_c': 130.0
        })

        # Should still produce thermal result
        assert thermal_result is not None
        assert thermal_result.trap_id == 'TRAP-NO-ACOUSTIC'

        # Diagnosis with None acoustic should still work
        diagnosis_result = tools.diagnose_trap_failure(
            {'trap_id': 'TRAP-NO-ACOUSTIC', 'pressure_upstream_psig': 100.0},
            None,  # No acoustic data
            thermal_result
        )

        assert diagnosis_result is not None

    def test_missing_thermal_data(self, tools):
        """Test workflow handles missing thermal data gracefully."""
        np.random.seed(42)

        acoustic_result = tools.analyze_acoustic_signature({
            'trap_id': 'TRAP-NO-THERMAL',
            'signal': (np.random.randn(10000) * 0.2).tolist(),
            'sampling_rate_hz': 250000
        })

        # Should still produce acoustic result
        assert acoustic_result is not None
        assert acoustic_result.trap_id == 'TRAP-NO-THERMAL'

        # Diagnosis with None thermal should still work
        diagnosis_result = tools.diagnose_trap_failure(
            {'trap_id': 'TRAP-NO-THERMAL', 'pressure_upstream_psig': 100.0},
            acoustic_result,
            None  # No thermal data
        )

        assert diagnosis_result is not None

    def test_empty_fleet_handling(self, tools):
        """Test fleet prioritization with empty fleet."""
        with pytest.raises((ValueError, AssertionError)):
            tools.prioritize_maintenance([])


@pytest.mark.e2e
class TestWorkflowReproducibility:
    """Test that complete workflows are reproducible."""

    def test_full_workflow_reproducibility(self, tools):
        """Test that identical inputs produce identical workflow outputs."""
        np.random.seed(42)
        signal = (np.random.randn(10000) * 0.2).tolist()

        def run_complete_workflow():
            acoustic = tools.analyze_acoustic_signature({
                'trap_id': 'TRAP-REPRO-TEST',
                'signal': signal,
                'sampling_rate_hz': 250000
            })

            thermal = tools.analyze_thermal_pattern({
                'trap_id': 'TRAP-REPRO-TEST',
                'temperature_upstream_c': 150.0,
                'temperature_downstream_c': 130.0
            })

            diagnosis = tools.diagnose_trap_failure(
                {'trap_id': 'TRAP-REPRO-TEST', 'pressure_upstream_psig': 100.0},
                acoustic,
                thermal
            )

            return {
                'acoustic_hash': acoustic.provenance_hash,
                'thermal_hash': thermal.provenance_hash,
                'failure_mode': diagnosis.failure_mode,
                'confidence': diagnosis.confidence
            }

        # Run workflow twice
        result1 = run_complete_workflow()
        result2 = run_complete_workflow()

        # Results must be identical
        assert result1['acoustic_hash'] == result2['acoustic_hash']
        assert result1['thermal_hash'] == result2['thermal_hash']
        assert result1['failure_mode'] == result2['failure_mode']
        assert result1['confidence'] == result2['confidence']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "e2e"])
