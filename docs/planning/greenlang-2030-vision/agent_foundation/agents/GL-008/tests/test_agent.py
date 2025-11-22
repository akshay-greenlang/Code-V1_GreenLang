# -*- coding: utf-8 -*-
"""
Integration tests for GL-008 SteamTrapInspector agent.

Tests the main orchestrator class with all operation modes.
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from steam_trap_inspector import SteamTrapInspector
from config import TrapInspectorConfig, TrapType, FailureMode


@pytest.fixture
def config():
    """Fixture providing test configuration."""
    return TrapInspectorConfig(
        agent_id="GL-008-TEST",
        enable_llm_classification=False,  # Disable LLM for testing
        cache_ttl_seconds=60,
        max_concurrent_inspections=5
    )


@pytest.fixture
def inspector(config):
    """Fixture providing SteamTrapInspector instance."""
    return SteamTrapInspector(config)


@pytest.mark.asyncio
class TestAgentOperationModes:
    """Test all 6 operation modes."""

    async def test_monitor_mode(self, inspector):
        """Test monitor operation mode."""
        input_data = {
            'operation_mode': 'monitor',
            'trap_data': {
                'trap_id': 'TRAP-MONITOR-001',
                'trap_type': 'thermodynamic',
                'acoustic_data': {
                    'signal': np.random.randn(5000).tolist(),
                    'sampling_rate_hz': 250000
                },
                'thermal_data': {
                    'temperature_upstream_c': 150.0,
                    'temperature_downstream_c': 130.0
                },
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0
            },
            'analysis_options': {
                'inspection_method': 'multi_modal'
            }
        }

        result = await inspector.execute(input_data)

        # Verify result structure
        assert result['operation_mode'] == 'monitor'
        assert 'trap_status' in result
        assert 'analysis_results' in result
        assert 'alerts' in result
        assert 'recommendations' in result
        assert result['deterministic'] == True

    async def test_diagnose_mode(self, inspector):
        """Test diagnose operation mode."""
        input_data = {
            'operation_mode': 'diagnose',
            'trap_data': {
                'trap_id': 'TRAP-DIAGNOSE-001',
                'acoustic_data': {
                    'signal': np.random.randn(5000).tolist(),
                    'sampling_rate_hz': 250000
                },
                'thermal_data': {
                    'temperature_upstream_c': 180.0,
                    'temperature_downstream_c': 70.0  # Large delta = failed closed
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

    async def test_predict_mode(self, inspector):
        """Test predict operation mode."""
        input_data = {
            'operation_mode': 'predict',
            'trap_data': {
                'trap_id': 'TRAP-PREDICT-001',
                'current_age_days': 1000,
                'degradation_rate': 0.1,
                'current_health_score': 75,
                'process_criticality': 8
            }
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'predict'
        assert 'predictive_maintenance' in result
        assert 'maintenance_schedule' in result
        assert 'recommendations' in result

    async def test_prioritize_mode(self, inspector):
        """Test prioritize operation mode."""
        input_data = {
            'operation_mode': 'prioritize',
            'fleet_data': [
                {
                    'trap_id': 'TRAP-001',
                    'failure_mode': 'failed_open',
                    'energy_loss_usd_yr': 15000,
                    'process_criticality': 9,
                    'current_age_years': 10
                },
                {
                    'trap_id': 'TRAP-002',
                    'failure_mode': 'leaking',
                    'energy_loss_usd_yr': 3000,
                    'process_criticality': 5,
                    'current_age_years': 3
                }
            ]
        }

        result = await inspector.execute(input_data)

        assert result['operation_mode'] == 'prioritize'
        assert 'fleet_summary' in result
        assert 'prioritized_maintenance_plan' in result
        assert 'financial_summary' in result

    async def test_performance_metrics(self, inspector):
        """Test that performance metrics are tracked."""
        input_data = {
            'operation_mode': 'monitor',
            'trap_data': {
                'trap_id': 'TRAP-METRICS',
                'thermal_data': {
                    'temperature_upstream_c': 150.0,
                    'temperature_downstream_c': 130.0
                }
            }
        }

        result = await inspector.execute(input_data)

        # Check performance metrics
        assert 'performance_metrics' in result
        metrics = result['performance_metrics']
        assert 'inspections_performed' in metrics
        assert 'cache_hit_rate' in metrics
        assert 'avg_inspection_time_ms' in metrics


@pytest.mark.asyncio
class TestAgentDeterminism:
    """Test deterministic behavior."""

    async def test_deterministic_results(self, inspector):
        """Test that identical inputs produce identical outputs."""
        input_data = {
            'operation_mode': 'monitor',
            'trap_data': {
                'trap_id': 'TRAP-DET',
                'thermal_data': {
                    'temperature_upstream_c': 150.0,
                    'temperature_downstream_c': 130.0
                },
                'orifice_diameter_in': 0.125,
                'steam_pressure_psig': 100.0
            }
        }

        # Execute twice
        result1 = await inspector.execute(input_data)
        result2 = await inspector.execute(input_data)

        # Compare critical values (excluding timestamps and provenance hashes)
        assert result1['trap_status']['health_score'] == result2['trap_status']['health_score']
        assert result1['deterministic'] == True
        assert result2['deterministic'] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
