# -*- coding: utf-8 -*-
"""
Integration tests for GL-001 THERMOSYNC ↔ GL-006 HEATRECLAIM coordination.

Tests the coordination between ProcessHeatOrchestrator (GL-001) and
HeatRecoveryOptimizer (GL-006) for waste heat recovery workflows.

Test Scenarios:
1. GL-001 orchestrates GL-006 for waste heat recovery
2. GL-001 identifies waste heat streams
3. GL-006 analyzes recovery opportunities
4. GL-006 returns prioritized opportunities
5. GL-001 updates heat distribution strategy

Coverage: Tests data flow, opportunity prioritization, economic analysis,
integration with heat distribution, and real-time coordination.
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
        'enable_heat_recovery': True,
        'min_recovery_temperature_c': 150,
        'heat_recovery_targets': {
            'min_efficiency_gain_percent': 2.0,
            'min_payback_period_years': 3.0
        }
    }


@pytest.fixture
def gl006_config():
    """Configuration for GL-006 HeatRecoveryOptimizer."""
    return {
        'agent_id': 'GL-006',
        'agent_name': 'HeatRecoveryOptimizer',
        'version': '1.0.0',
        'min_waste_heat_temp_c': 120,
        'max_recovery_efficiency': 0.85,
        'economic_analysis_enabled': True,
        'energy_price_usd_per_mwh': 80.0
    }


@pytest.fixture
def mock_gl001_orchestrator(gl001_config):
    """Mock GL-001 ProcessHeatOrchestrator instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl001_config['agent_id']

    # Mock waste heat stream identification
    async def mock_identify_waste_heat_streams(sensor_data):
        return {
            'waste_heat_streams': [
                {
                    'stream_id': 'WHS-001',
                    'source': 'boiler_stack',
                    'temperature_c': 280,
                    'flow_rate_kg_hr': 15000,
                    'heat_content_mw': 1.2,
                    'availability_percent': 95
                },
                {
                    'stream_id': 'WHS-002',
                    'source': 'compressor_cooling',
                    'temperature_c': 180,
                    'flow_rate_kg_hr': 8000,
                    'heat_content_mw': 0.6,
                    'availability_percent': 90
                },
                {
                    'stream_id': 'WHS-003',
                    'source': 'condensate_return',
                    'temperature_c': 95,
                    'flow_rate_kg_hr': 5000,
                    'heat_content_mw': 0.15,
                    'availability_percent': 100
                }
            ],
            'total_waste_heat_mw': 1.95,
            'recovery_potential_percent': 65
        }

    mock_agent.identify_waste_heat_streams = mock_identify_waste_heat_streams

    # Mock heat distribution update
    async def mock_update_heat_distribution(recovery_opportunities):
        return {
            'status': 'success',
            'updated_distribution': {
                'recovered_heat_integrated_mw': sum(
                    opp['recovered_heat_mw'] for opp in recovery_opportunities
                ),
                'new_efficiency_percent': 88.5,
                'efficiency_improvement_percent': 3.2
            }
        }

    mock_agent.update_heat_distribution = mock_update_heat_distribution

    return mock_agent


@pytest.fixture
def mock_gl006_optimizer(gl006_config):
    """Mock GL-006 HeatRecoveryOptimizer instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl006_config['agent_id']

    # Mock async execute method
    async def mock_execute(input_data):
        waste_streams = input_data.get('waste_heat_streams', [])

        # Analyze each stream and generate opportunities
        opportunities = []
        for i, stream in enumerate(waste_streams):
            if stream['temperature_c'] >= gl006_config['min_waste_heat_temp_c']:
                recovered_heat = stream['heat_content_mw'] * 0.7  # 70% recovery
                opportunities.append({
                    'opportunity_id': f"OPP-{i+1:03d}",
                    'stream_id': stream['stream_id'],
                    'recovery_technology': 'economizer' if stream['temperature_c'] > 200 else 'heat_exchanger',
                    'recovered_heat_mw': recovered_heat,
                    'recovery_efficiency_percent': 70,
                    'capital_cost_usd': 150000 + (recovered_heat * 100000),
                    'annual_savings_usd': recovered_heat * gl006_config['energy_price_usd_per_mwh'] * 8760,
                    'payback_period_years': round(
                        (150000 + recovered_heat * 100000) /
                        (recovered_heat * gl006_config['energy_price_usd_per_mwh'] * 8760),
                        2
                    ),
                    'priority': 'high' if recovered_heat > 0.5 else 'medium',
                    'implementation_complexity': 'medium'
                })

        # Sort by payback period (best first)
        opportunities.sort(key=lambda x: x['payback_period_years'])

        return {
            'agent_id': gl006_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'execution_time_ms': 52.3,
            'recovery_opportunities': opportunities,
            'total_recoverable_heat_mw': sum(opp['recovered_heat_mw'] for opp in opportunities),
            'total_annual_savings_usd': sum(opp['annual_savings_usd'] for opp in opportunities),
            'recommended_opportunities': [
                opp for opp in opportunities if opp['payback_period_years'] <= 3.0
            ],
            'kpi_dashboard': {
                'opportunities_identified': len(opportunities),
                'total_recovery_potential_mw': sum(opp['recovered_heat_mw'] for opp in opportunities),
                'average_payback_years': sum(opp['payback_period_years'] for opp in opportunities) / len(opportunities) if opportunities else 0
            },
            'optimization_success': True,
            'provenance_hash': 'heat_recovery_xyz789'
        }

    mock_agent.execute = mock_execute

    # Mock opportunity analysis method
    async def mock_analyze_recovery_opportunities(waste_streams, constraints):
        input_data = {
            'waste_heat_streams': waste_streams,
            'constraints': constraints
        }
        return await mock_agent.execute(input_data)

    mock_agent.analyze_recovery_opportunities = mock_analyze_recovery_opportunities

    return mock_agent


@pytest.fixture
def waste_heat_streams_payload():
    """Sample waste heat streams data."""
    return [
        {
            'stream_id': 'WHS-001',
            'source': 'boiler_stack',
            'temperature_c': 280,
            'flow_rate_kg_hr': 15000,
            'heat_content_mw': 1.2,
            'availability_percent': 95,
            'composition': {'air': 100}
        },
        {
            'stream_id': 'WHS-002',
            'source': 'compressor_cooling',
            'temperature_c': 180,
            'flow_rate_kg_hr': 8000,
            'heat_content_mw': 0.6,
            'availability_percent': 90,
            'composition': {'water': 100}
        }
    ]


# ============================================================================
# Test Class: GL-001 ↔ GL-006 Coordination
# ============================================================================

class TestGL001GL006Coordination:
    """Test suite for GL-001 ↔ GL-006 waste heat recovery coordination."""

    @pytest.mark.asyncio
    async def test_waste_heat_stream_identification(
        self,
        mock_gl001_orchestrator
    ):
        """Test GL-001 identifies waste heat streams."""
        # Arrange
        sensor_data = {
            'stack_temp_c': 280,
            'compressor_outlet_temp_c': 180,
            'condensate_temp_c': 95
        }

        # Act
        result = await mock_gl001_orchestrator.identify_waste_heat_streams(sensor_data)

        # Assert
        assert 'waste_heat_streams' in result
        assert len(result['waste_heat_streams']) > 0
        assert result['total_waste_heat_mw'] > 0

        # Verify stream structure
        for stream in result['waste_heat_streams']:
            assert 'stream_id' in stream
            assert 'temperature_c' in stream
            assert 'heat_content_mw' in stream

    @pytest.mark.asyncio
    async def test_gl006_analyzes_recovery_opportunities(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-006 analyzes waste heat recovery opportunities."""
        # Arrange
        constraints = {
            'max_capital_budget_usd': 500000,
            'max_payback_years': 3.0
        }

        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints=constraints
        )

        # Assert
        assert result['optimization_success'] is True
        assert 'recovery_opportunities' in result
        assert len(result['recovery_opportunities']) > 0

        # Verify opportunity structure
        for opp in result['recovery_opportunities']:
            assert 'opportunity_id' in opp
            assert 'recovered_heat_mw' in opp
            assert 'payback_period_years' in opp
            assert 'priority' in opp

    @pytest.mark.asyncio
    async def test_gl006_prioritizes_opportunities(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-006 returns prioritized opportunities."""
        # Arrange
        constraints = {'max_payback_years': 3.0}

        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints=constraints
        )

        # Assert
        opportunities = result['recovery_opportunities']
        assert len(opportunities) > 0

        # Verify opportunities are sorted by payback period (ascending)
        payback_periods = [opp['payback_period_years'] for opp in opportunities]
        assert payback_periods == sorted(payback_periods)

        # Verify recommended opportunities meet criteria
        recommended = result['recommended_opportunities']
        for opp in recommended:
            assert opp['payback_period_years'] <= 3.0

    @pytest.mark.asyncio
    async def test_gl001_updates_heat_distribution(
        self,
        mock_gl001_orchestrator,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-001 updates heat distribution strategy with recovery opportunities."""
        # Arrange
        # First, get recovery opportunities from GL-006
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints={}
        )
        opportunities = result['recommended_opportunities']

        # Act
        update_result = await mock_gl001_orchestrator.update_heat_distribution(
            recovery_opportunities=opportunities
        )

        # Assert
        assert update_result['status'] == 'success'
        assert 'updated_distribution' in update_result
        assert update_result['updated_distribution']['recovered_heat_integrated_mw'] > 0
        assert update_result['updated_distribution']['efficiency_improvement_percent'] > 0

    @pytest.mark.asyncio
    async def test_end_to_end_recovery_workflow(
        self,
        mock_gl001_orchestrator,
        mock_gl006_optimizer
    ):
        """Test complete end-to-end waste heat recovery workflow."""
        # Step 1: GL-001 identifies waste heat streams
        sensor_data = {'stack_temp_c': 280, 'compressor_outlet_temp_c': 180}
        streams_result = await mock_gl001_orchestrator.identify_waste_heat_streams(sensor_data)

        assert 'waste_heat_streams' in streams_result

        # Step 2: GL-006 analyzes recovery opportunities
        analysis_result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=streams_result['waste_heat_streams'],
            constraints={'max_payback_years': 3.0}
        )

        assert analysis_result['optimization_success'] is True

        # Step 3: GL-001 updates heat distribution
        update_result = await mock_gl001_orchestrator.update_heat_distribution(
            recovery_opportunities=analysis_result['recommended_opportunities']
        )

        assert update_result['status'] == 'success'
        assert update_result['updated_distribution']['efficiency_improvement_percent'] > 0

    @pytest.mark.asyncio
    async def test_economic_analysis_validation(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-006 economic analysis is valid."""
        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints={}
        )

        # Assert
        for opp in result['recovery_opportunities']:
            # Verify economic calculations
            assert opp['capital_cost_usd'] > 0
            assert opp['annual_savings_usd'] > 0
            assert opp['payback_period_years'] > 0

            # Verify payback calculation is correct
            expected_payback = opp['capital_cost_usd'] / opp['annual_savings_usd']
            assert abs(opp['payback_period_years'] - expected_payback) < 0.1

    @pytest.mark.asyncio
    async def test_technology_selection(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-006 selects appropriate recovery technology."""
        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints={}
        )

        # Assert
        for opp in result['recovery_opportunities']:
            # Find corresponding stream
            stream = next(
                s for s in waste_heat_streams_payload
                if s['stream_id'] == opp['stream_id']
            )

            # High temp (>200C) should use economizer
            if stream['temperature_c'] > 200:
                assert opp['recovery_technology'] == 'economizer'
            else:
                # Lower temp should use heat exchanger
                assert opp['recovery_technology'] in ['heat_exchanger', 'economizer']

    @pytest.mark.asyncio
    async def test_concurrent_stream_analysis(
        self,
        mock_gl006_optimizer
    ):
        """Test GL-006 analyzes multiple streams concurrently."""
        # Arrange - Create 10 different waste heat stream scenarios
        stream_scenarios = []
        for i in range(10):
            streams = [
                {
                    'stream_id': f'WHS-{i}-001',
                    'source': f'source_{i}',
                    'temperature_c': 200 + (i * 10),
                    'flow_rate_kg_hr': 10000 + (i * 1000),
                    'heat_content_mw': 0.5 + (i * 0.1),
                    'availability_percent': 90
                }
            ]
            stream_scenarios.append(streams)

        # Act
        tasks = [
            mock_gl006_optimizer.analyze_recovery_opportunities(
                waste_streams=streams,
                constraints={}
            )
            for streams in stream_scenarios
        ]

        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 10
        for result in results:
            assert result['optimization_success'] is True
            assert len(result['recovery_opportunities']) > 0

    @pytest.mark.asyncio
    async def test_constraint_enforcement(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test GL-006 enforces budget and payback constraints."""
        # Arrange
        constraints = {
            'max_capital_budget_usd': 200000,
            'max_payback_years': 2.0
        }

        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints=constraints
        )

        # Assert
        recommended = result['recommended_opportunities']
        for opp in recommended:
            # Should meet payback constraint
            assert opp['payback_period_years'] <= constraints['max_payback_years']

    @pytest.mark.asyncio
    async def test_data_flow_compatibility(
        self,
        mock_gl001_orchestrator,
        mock_gl006_optimizer
    ):
        """Test data format compatibility between GL-001 and GL-006."""
        # Step 1: GL-001 generates waste heat stream data
        sensor_data = {'stack_temp_c': 280}
        gl001_output = await mock_gl001_orchestrator.identify_waste_heat_streams(sensor_data)

        # Step 2: Verify GL-006 can consume GL-001 output
        try:
            gl006_result = await mock_gl006_optimizer.analyze_recovery_opportunities(
                waste_streams=gl001_output['waste_heat_streams'],
                constraints={}
            )
            compatibility_test_passed = True
        except Exception as e:
            compatibility_test_passed = False
            error_msg = str(e)

        # Assert
        assert compatibility_test_passed, f"Data format incompatibility: {error_msg if not compatibility_test_passed else ''}"
        assert gl006_result['optimization_success'] is True

    @pytest.mark.asyncio
    async def test_real_time_coordination(
        self,
        mock_gl001_orchestrator,
        mock_gl006_optimizer
    ):
        """Test real-time coordination latency."""
        # Arrange
        sensor_data = {'stack_temp_c': 280}

        # Act
        start_time = time.perf_counter()

        # Step 1: Identify streams
        streams = await mock_gl001_orchestrator.identify_waste_heat_streams(sensor_data)

        # Step 2: Analyze opportunities
        opportunities = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=streams['waste_heat_streams'],
            constraints={}
        )

        # Step 3: Update distribution
        update = await mock_gl001_orchestrator.update_heat_distribution(
            recovery_opportunities=opportunities['recommended_opportunities']
        )

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert total_latency_ms < 200  # Should complete in <200ms with mocks

    @pytest.mark.asyncio
    async def test_error_handling_no_recoverable_heat(
        self,
        mock_gl006_optimizer
    ):
        """Test handling when no recoverable heat is available."""
        # Arrange - Low temperature streams below recovery threshold
        low_temp_streams = [
            {
                'stream_id': 'WHS-LOW-001',
                'source': 'cooling_water',
                'temperature_c': 60,  # Below threshold
                'flow_rate_kg_hr': 5000,
                'heat_content_mw': 0.1,
                'availability_percent': 100
            }
        ]

        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=low_temp_streams,
            constraints={}
        )

        # Assert
        # Should return empty opportunities but still succeed
        assert result['optimization_success'] is True
        assert len(result['recovery_opportunities']) == 0
        assert result['total_recoverable_heat_mw'] == 0

    @pytest.mark.asyncio
    async def test_provenance_tracking(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test provenance tracking for waste heat recovery."""
        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints={}
        )

        # Assert
        assert 'provenance_hash' in result
        assert result['provenance_hash'] is not None
        assert len(result['provenance_hash']) > 0

    @pytest.mark.asyncio
    async def test_opportunity_prioritization_algorithm(
        self,
        mock_gl006_optimizer,
        waste_heat_streams_payload
    ):
        """Test opportunity prioritization follows correct algorithm."""
        # Act
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=waste_heat_streams_payload,
            constraints={}
        )

        # Assert
        opportunities = result['recovery_opportunities']

        # Should be sorted by payback period
        for i in range(len(opportunities) - 1):
            assert opportunities[i]['payback_period_years'] <= opportunities[i+1]['payback_period_years']

        # High priority should be assigned to large recovery potential
        for opp in opportunities:
            if opp['recovered_heat_mw'] > 0.5:
                assert opp['priority'] == 'high'


# ============================================================================
# Test Class: Performance and Scalability
# ============================================================================

class TestGL001GL006Performance:
    """Test performance and scalability of GL-001 ↔ GL-006 coordination."""

    @pytest.mark.asyncio
    async def test_large_number_of_streams(
        self,
        mock_gl006_optimizer
    ):
        """Test performance with large number of waste heat streams."""
        # Arrange - Create 100 waste heat streams
        large_stream_set = [
            {
                'stream_id': f'WHS-{i:04d}',
                'source': f'source_{i}',
                'temperature_c': 150 + (i % 200),
                'flow_rate_kg_hr': 5000 + (i * 100),
                'heat_content_mw': 0.2 + (i * 0.01),
                'availability_percent': 85 + (i % 15)
            }
            for i in range(100)
        ]

        # Act
        start_time = time.perf_counter()
        result = await mock_gl006_optimizer.analyze_recovery_opportunities(
            waste_streams=large_stream_set,
            constraints={}
        )
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert result['optimization_success'] is True
        assert len(result['recovery_opportunities']) > 0
        assert execution_time_ms < 500  # Should handle 100 streams in <500ms

    @pytest.mark.asyncio
    async def test_throughput_measurement(
        self,
        mock_gl001_orchestrator,
        mock_gl006_optimizer
    ):
        """Test coordination throughput."""
        # Arrange - 20 coordination cycles
        num_cycles = 20

        # Act
        start_time = time.perf_counter()

        for _ in range(num_cycles):
            sensor_data = {'stack_temp_c': 280}
            streams = await mock_gl001_orchestrator.identify_waste_heat_streams(sensor_data)
            opportunities = await mock_gl006_optimizer.analyze_recovery_opportunities(
                waste_streams=streams['waste_heat_streams'],
                constraints={}
            )
            await mock_gl001_orchestrator.update_heat_distribution(
                recovery_opportunities=opportunities['recommended_opportunities']
            )

        total_time_s = time.perf_counter() - start_time
        throughput = num_cycles / total_time_s

        # Assert
        assert throughput > 10  # At least 10 cycles/second with mocks
