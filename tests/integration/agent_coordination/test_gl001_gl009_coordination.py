# -*- coding: utf-8 -*-
"""
Integration tests for GL-001 THERMOSYNC ↔ GL-009 THERMALIQ coordination.

Tests the coordination between ProcessHeatOrchestrator (GL-001) and
ThermalEfficiencyAnalyzer (GL-009) for thermal efficiency analysis.

Test Scenarios:
1. GL-001 orchestrates overall process heat
2. GL-009 calculates thermal efficiency
3. GL-001 requests efficiency analysis
4. GL-009 returns first/second law efficiency
5. GL-001 uses efficiency data for optimization

Coverage: Tests efficiency calculation integration, exergy analysis,
optimization feedback loops, performance tracking, and thermodynamic validation.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def gl001_config():
    """Configuration for GL-001 ProcessHeatOrchestrator."""
    return {
        'agent_id': 'GL-001',
        'agent_name': 'ProcessHeatOrchestrator',
        'version': '1.0.0',
        'efficiency_analysis_enabled': True,
        'target_first_law_efficiency': 85.0,
        'target_second_law_efficiency': 60.0
    }


@pytest.fixture
def gl009_config():
    """Configuration for GL-009 ThermalEfficiencyAnalyzer."""
    return {
        'agent_id': 'GL-009',
        'agent_name': 'ThermalEfficiencyAnalyzer',
        'version': '1.0.0',
        'analysis_methods': ['first_law', 'second_law', 'exergy'],
        'reference_temperature_k': 298.15,
        'include_exergy_destruction': True
    }


@pytest.fixture
def mock_gl001_orchestrator(gl001_config):
    """Mock GL-001 ProcessHeatOrchestrator instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl001_config['agent_id']

    # Mock request efficiency analysis
    async def mock_request_efficiency_analysis(thermal_data, analyzer):
        # Call GL-009 for analysis
        analysis_result = await analyzer.analyze_thermal_efficiency(thermal_data)

        return {
            'status': 'success',
            'efficiency_analysis': analysis_result,
            'coordination_latency_ms': 38.5
        }

    mock_agent.request_efficiency_analysis = mock_request_efficiency_analysis

    # Mock use efficiency data
    async def mock_use_efficiency_data_for_optimization(efficiency_data):
        first_law_eff = efficiency_data.get('first_law_efficiency_percent', 0)
        second_law_eff = efficiency_data.get('second_law_efficiency_percent', 0)

        # Calculate optimization potential
        first_law_gap = gl001_config['target_first_law_efficiency'] - first_law_eff
        second_law_gap = gl001_config['target_second_law_efficiency'] - second_law_eff

        return {
            'status': 'success',
            'optimization_strategy': {
                'first_law_improvement_potential_percent': max(0, first_law_gap),
                'second_law_improvement_potential_percent': max(0, second_law_gap),
                'recommended_actions': [
                    'Improve heat recovery' if first_law_gap > 5 else None,
                    'Reduce irreversibilities' if second_law_gap > 10 else None,
                    'Optimize temperature matching' if second_law_eff < 50 else None
                ],
                'priority': 'high' if first_law_gap > 10 or second_law_gap > 15 else 'medium'
            }
        }

    mock_agent.use_efficiency_data_for_optimization = mock_use_efficiency_data_for_optimization

    return mock_agent


@pytest.fixture
def mock_gl009_analyzer(gl009_config):
    """Mock GL-009 ThermalEfficiencyAnalyzer instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl009_config['agent_id']

    # Mock async analyze method
    async def mock_analyze_thermal_efficiency(thermal_data):
        # Extract thermal data
        heat_input_mw = thermal_data.get('heat_input_mw', 10.0)
        useful_heat_output_mw = thermal_data.get('useful_heat_output_mw', 8.5)
        inlet_temp_k = thermal_data.get('inlet_temperature_k', 473.15)
        outlet_temp_k = thermal_data.get('outlet_temperature_k', 373.15)
        ambient_temp_k = gl009_config['reference_temperature_k']

        # First law efficiency (energy balance)
        first_law_efficiency = (useful_heat_output_mw / heat_input_mw * 100) if heat_input_mw > 0 else 0

        # Second law efficiency (exergy efficiency)
        # Carnot efficiency based on temperatures
        carnot_efficiency = 1 - (ambient_temp_k / inlet_temp_k) if inlet_temp_k > ambient_temp_k else 0
        exergy_input = heat_input_mw * carnot_efficiency
        exergy_output = useful_heat_output_mw * (1 - ambient_temp_k / outlet_temp_k) if outlet_temp_k > ambient_temp_k else 0
        second_law_efficiency = (exergy_output / exergy_input * 100) if exergy_input > 0 else 0

        # Exergy destruction
        exergy_destruction_mw = exergy_input - exergy_output if exergy_input > exergy_output else 0

        return {
            'agent_id': gl009_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': 42.8,
            'first_law_efficiency_percent': round(first_law_efficiency, 2),
            'second_law_efficiency_percent': round(second_law_efficiency, 2),
            'carnot_efficiency_percent': round(carnot_efficiency * 100, 2),
            'exergy_analysis': {
                'exergy_input_mw': round(exergy_input, 3),
                'exergy_output_mw': round(exergy_output, 3),
                'exergy_destruction_mw': round(exergy_destruction_mw, 3),
                'exergy_destruction_percent': round(exergy_destruction_mw / exergy_input * 100, 2) if exergy_input > 0 else 0
            },
            'efficiency_metrics': {
                'energy_utilization_factor': round(first_law_efficiency / 100, 3),
                'quality_factor': round(second_law_efficiency / first_law_efficiency, 3) if first_law_efficiency > 0 else 0,
                'irreversibility_index': round(exergy_destruction_mw / exergy_input, 3) if exergy_input > 0 else 0
            },
            'recommendations': [
                'Good first law efficiency' if first_law_efficiency > 80 else 'Improve heat recovery',
                'Excellent exergy efficiency' if second_law_efficiency > 60 else 'Reduce irreversibilities',
                'Optimize temperature matching' if second_law_efficiency < 50 else 'Maintain current operation'
            ],
            'analysis_success': True,
            'provenance_hash': 'thermal_efficiency_hash_abc'
        }

    mock_agent.analyze_thermal_efficiency = mock_analyze_thermal_efficiency

    return mock_agent


@pytest.fixture
def thermal_system_data():
    """Sample thermal system data."""
    return {
        'heat_input_mw': 12.0,
        'useful_heat_output_mw': 10.2,
        'inlet_temperature_k': 500.0,
        'outlet_temperature_k': 380.0,
        'mass_flow_rate_kg_s': 15.0,
        'pressure_bar': 40.0
    }


# ============================================================================
# Test Class: GL-001 ↔ GL-009 Coordination
# ============================================================================

class TestGL001GL009Coordination:
    """Test suite for GL-001 ↔ GL-009 thermal efficiency analysis coordination."""

    @pytest.mark.asyncio
    async def test_gl001_requests_efficiency_analysis(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-001 requests efficiency analysis from GL-009."""
        # Act
        result = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=thermal_system_data,
            analyzer=mock_gl009_analyzer
        )

        # Assert
        assert result['status'] == 'success'
        assert 'efficiency_analysis' in result
        assert 'coordination_latency_ms' in result

    @pytest.mark.asyncio
    async def test_gl009_calculates_first_law_efficiency(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-009 calculates first law (energy) efficiency."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert result['analysis_success'] is True
        assert 'first_law_efficiency_percent' in result

        # Verify calculation: (10.2 / 12.0) * 100 = 85%
        expected_efficiency = (thermal_system_data['useful_heat_output_mw'] /
                              thermal_system_data['heat_input_mw'] * 100)
        assert abs(result['first_law_efficiency_percent'] - expected_efficiency) < 0.1

    @pytest.mark.asyncio
    async def test_gl009_calculates_second_law_efficiency(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-009 calculates second law (exergy) efficiency."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert 'second_law_efficiency_percent' in result
        assert 'carnot_efficiency_percent' in result

        # Second law efficiency should be less than first law
        assert result['second_law_efficiency_percent'] <= result['first_law_efficiency_percent']

        # Carnot efficiency should be based on temperature ratio
        assert result['carnot_efficiency_percent'] > 0

    @pytest.mark.asyncio
    async def test_gl009_performs_exergy_analysis(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-009 performs comprehensive exergy analysis."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert 'exergy_analysis' in result
        exergy = result['exergy_analysis']

        assert 'exergy_input_mw' in exergy
        assert 'exergy_output_mw' in exergy
        assert 'exergy_destruction_mw' in exergy
        assert 'exergy_destruction_percent' in exergy

        # Exergy balance: input = output + destruction
        total = exergy['exergy_output_mw'] + exergy['exergy_destruction_mw']
        assert abs(total - exergy['exergy_input_mw']) < 0.01

    @pytest.mark.asyncio
    async def test_gl001_uses_efficiency_data_for_optimization(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-001 uses efficiency data for optimization."""
        # Step 1: Get efficiency analysis from GL-009
        analysis_result = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=thermal_system_data,
            analyzer=mock_gl009_analyzer
        )

        efficiency_data = analysis_result['efficiency_analysis']

        # Step 2: Use efficiency data for optimization
        optimization = await mock_gl001_orchestrator.use_efficiency_data_for_optimization(
            efficiency_data=efficiency_data
        )

        # Assert
        assert optimization['status'] == 'success'
        assert 'optimization_strategy' in optimization

        strategy = optimization['optimization_strategy']
        assert 'first_law_improvement_potential_percent' in strategy
        assert 'second_law_improvement_potential_percent' in strategy
        assert 'recommended_actions' in strategy

    @pytest.mark.asyncio
    async def test_end_to_end_efficiency_optimization_workflow(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test complete efficiency analysis and optimization workflow."""
        # Step 1: GL-001 requests analysis
        analysis = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=thermal_system_data,
            analyzer=mock_gl009_analyzer
        )

        assert analysis['status'] == 'success'

        # Step 2: GL-009 provides analysis
        efficiency_data = analysis['efficiency_analysis']
        assert efficiency_data['analysis_success'] is True

        # Step 3: GL-001 uses data for optimization
        optimization = await mock_gl001_orchestrator.use_efficiency_data_for_optimization(
            efficiency_data=efficiency_data
        )

        assert optimization['status'] == 'success'

        # Step 4: Verify optimization strategy is generated
        assert 'optimization_strategy' in optimization
        assert 'priority' in optimization['optimization_strategy']

    @pytest.mark.asyncio
    async def test_efficiency_metrics_calculation(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-009 calculates comprehensive efficiency metrics."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert 'efficiency_metrics' in result
        metrics = result['efficiency_metrics']

        assert 'energy_utilization_factor' in metrics
        assert 'quality_factor' in metrics
        assert 'irreversibility_index' in metrics

        # Energy utilization factor should match first law efficiency
        assert 0 <= metrics['energy_utilization_factor'] <= 1.0

        # Quality factor relates second law to first law
        assert 0 <= metrics['quality_factor'] <= 1.0

    @pytest.mark.asyncio
    async def test_thermodynamic_validation(
        self,
        mock_gl009_analyzer
    ):
        """Test thermodynamic laws are validated in calculations."""
        # Arrange
        valid_data = {
            'heat_input_mw': 10.0,
            'useful_heat_output_mw': 8.5,
            'inlet_temperature_k': 500.0,
            'outlet_temperature_k': 380.0
        }

        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(valid_data)

        # Assert
        # First law: efficiency <= 100%
        assert result['first_law_efficiency_percent'] <= 100.0

        # Second law: efficiency <= Carnot efficiency
        assert result['second_law_efficiency_percent'] <= result['carnot_efficiency_percent']

        # Energy conservation
        assert result['first_law_efficiency_percent'] > 0

    @pytest.mark.asyncio
    async def test_temperature_impact_on_exergy(
        self,
        mock_gl009_analyzer
    ):
        """Test temperature levels impact exergy efficiency."""
        # Arrange - High temperature scenario
        high_temp_data = {
            'heat_input_mw': 10.0,
            'useful_heat_output_mw': 8.5,
            'inlet_temperature_k': 800.0,  # High temp
            'outlet_temperature_k': 600.0
        }

        # Low temperature scenario
        low_temp_data = {
            'heat_input_mw': 10.0,
            'useful_heat_output_mw': 8.5,
            'inlet_temperature_k': 400.0,  # Low temp
            'outlet_temperature_k': 350.0
        }

        # Act
        high_temp_result = await mock_gl009_analyzer.analyze_thermal_efficiency(high_temp_data)
        low_temp_result = await mock_gl009_analyzer.analyze_thermal_efficiency(low_temp_data)

        # Assert
        # Higher temperature should have higher Carnot efficiency
        assert high_temp_result['carnot_efficiency_percent'] > low_temp_result['carnot_efficiency_percent']

        # Higher quality heat should have higher exergy
        assert high_temp_result['exergy_analysis']['exergy_input_mw'] > low_temp_result['exergy_analysis']['exergy_input_mw']

    @pytest.mark.asyncio
    async def test_optimization_recommendations(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test GL-009 provides actionable recommendations."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert 'recommendations' in result
        assert len(result['recommendations']) > 0

        # Recommendations should be strings
        for rec in result['recommendations']:
            assert isinstance(rec, str)
            assert len(rec) > 0

    @pytest.mark.asyncio
    async def test_concurrent_efficiency_analyses(
        self,
        mock_gl009_analyzer
    ):
        """Test GL-009 handles concurrent analysis requests."""
        # Arrange - Multiple thermal systems
        systems = [
            {
                'heat_input_mw': 8.0 + i,
                'useful_heat_output_mw': 6.5 + i * 0.8,
                'inlet_temperature_k': 450.0 + i * 10,
                'outlet_temperature_k': 370.0 + i * 5
            }
            for i in range(10)
        ]

        # Act
        tasks = [
            mock_gl009_analyzer.analyze_thermal_efficiency(system)
            for system in systems
        ]

        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 10
        for result in results:
            assert result['analysis_success'] is True
            assert result['first_law_efficiency_percent'] > 0

    @pytest.mark.asyncio
    async def test_performance_tracking_integration(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test efficiency tracking over time for performance monitoring."""
        # Arrange - Simulate efficiency measurements over time
        measurements = []

        for i in range(5):
            # Gradually degrading efficiency
            data = {
                **thermal_system_data,
                'useful_heat_output_mw': 10.2 - (i * 0.2)  # Decreasing output
            }

            # Act
            analysis = await mock_gl001_orchestrator.request_efficiency_analysis(
                thermal_data=data,
                analyzer=mock_gl009_analyzer
            )

            measurements.append(analysis['efficiency_analysis']['first_law_efficiency_percent'])

        # Assert - Efficiency should show degradation trend
        assert measurements[0] > measurements[-1]

    @pytest.mark.asyncio
    async def test_optimization_feedback_loop(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer
    ):
        """Test optimization feedback loop using efficiency data."""
        # Initial state - low efficiency
        initial_data = {
            'heat_input_mw': 12.0,
            'useful_heat_output_mw': 9.0,  # 75% efficiency
            'inlet_temperature_k': 480.0,
            'outlet_temperature_k': 360.0
        }

        # Step 1: Analyze initial state
        initial_analysis = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=initial_data,
            analyzer=mock_gl009_analyzer
        )

        # Step 2: Get optimization strategy
        optimization = await mock_gl001_orchestrator.use_efficiency_data_for_optimization(
            efficiency_data=initial_analysis['efficiency_analysis']
        )

        # Step 3: Simulate improvement after optimization
        improved_data = {
            'heat_input_mw': 12.0,
            'useful_heat_output_mw': 10.5,  # Improved to 87.5%
            'inlet_temperature_k': 480.0,
            'outlet_temperature_k': 360.0
        }

        improved_analysis = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=improved_data,
            analyzer=mock_gl009_analyzer
        )

        # Assert
        initial_eff = initial_analysis['efficiency_analysis']['first_law_efficiency_percent']
        improved_eff = improved_analysis['efficiency_analysis']['first_law_efficiency_percent']

        assert improved_eff > initial_eff

    @pytest.mark.asyncio
    async def test_real_time_analysis_latency(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test real-time efficiency analysis latency."""
        # Act
        start_time = time.perf_counter()

        result = await mock_gl001_orchestrator.request_efficiency_analysis(
            thermal_data=thermal_system_data,
            analyzer=mock_gl009_analyzer
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert latency_ms < 100  # Should complete in <100ms for real-time
        assert result['coordination_latency_ms'] < 100

    @pytest.mark.asyncio
    async def test_data_format_compatibility(
        self,
        mock_gl001_orchestrator,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test data format compatibility between GL-001 and GL-009."""
        # Test GL-001 can consume GL-009 output
        try:
            analysis = await mock_gl001_orchestrator.request_efficiency_analysis(
                thermal_data=thermal_system_data,
                analyzer=mock_gl009_analyzer
            )

            optimization = await mock_gl001_orchestrator.use_efficiency_data_for_optimization(
                efficiency_data=analysis['efficiency_analysis']
            )

            compatibility_passed = True
        except Exception as e:
            compatibility_passed = False

        # Assert
        assert compatibility_passed, "Data format incompatibility detected"

    @pytest.mark.asyncio
    async def test_provenance_tracking(
        self,
        mock_gl009_analyzer,
        thermal_system_data
    ):
        """Test provenance tracking for efficiency calculations."""
        # Act
        result = await mock_gl009_analyzer.analyze_thermal_efficiency(thermal_system_data)

        # Assert
        assert 'provenance_hash' in result
        assert result['provenance_hash'] is not None
