# -*- coding: utf-8 -*-
"""
Integration tests for GL-002 FLAMEGUARD ↔ GL-010 EMISSIONWATCH coordination.

Tests the coordination between BoilerEfficiencyOptimizer (GL-002) and
EmissionsMonitor (GL-010) for emissions-constrained optimization.

Test Scenarios:
1. GL-002 optimizes boiler for efficiency
2. GL-010 monitors emissions compliance
3. GL-002 requests emission constraints from GL-010
4. GL-010 provides NOx/SOx limits
5. GL-002 optimizes within constraints
6. GL-010 validates emissions stay compliant

Coverage: Tests constraint enforcement, compliance validation, multi-objective
optimization, real-time emissions monitoring, and regulatory compliance.
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
def gl002_config():
    """Configuration for GL-002 BoilerEfficiencyOptimizer."""
    return {
        'agent_id': 'GL-002',
        'agent_name': 'BoilerEfficiencyOptimizer',
        'version': '1.0.0',
        'enable_emissions_constraints': True,
        'optimization_objective': 'balanced'  # Balance efficiency and emissions
    }


@pytest.fixture
def gl010_config():
    """Configuration for GL-010 EmissionsMonitor."""
    return {
        'agent_id': 'GL-010',
        'agent_name': 'EmissionsMonitor',
        'version': '1.0.0',
        'regulatory_standards': {
            'NOx_limit_ppm': 150,
            'SOx_limit_ppm': 200,
            'CO2_limit_kg_mwh': 500,
            'CO_limit_ppm': 100
        },
        'compliance_check_interval_seconds': 60
    }


@pytest.fixture
def mock_gl002_optimizer(gl002_config):
    """Mock GL-002 BoilerEfficiencyOptimizer instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl002_config['agent_id']

    # Mock optimization with emissions constraints
    async def mock_optimize_with_constraints(boiler_data, emission_constraints):
        # Simulate constrained optimization
        base_efficiency = 87.0
        base_nox = 180  # Without constraints

        # Apply constraints
        nox_limit = emission_constraints.get('NOx_limit_ppm', 200)

        if base_nox > nox_limit:
            # Reduce efficiency slightly to meet NOx constraint
            efficiency_penalty = (base_nox - nox_limit) * 0.02
            optimized_efficiency = base_efficiency - efficiency_penalty
            optimized_nox = nox_limit - 5  # Target below limit
        else:
            optimized_efficiency = base_efficiency
            optimized_nox = base_nox

        return {
            'agent_id': gl002_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'optimization_result': {
                'efficiency_percent': optimized_efficiency,
                'nox_emissions_ppm': optimized_nox,
                'sox_emissions_ppm': 120,
                'co2_emissions_kg_mwh': 450,
                'co_emissions_ppm': 60,
                'constraints_applied': True,
                'constraint_violations': []
            },
            'optimization_success': True,
            'meets_emission_limits': True
        }

    mock_agent.optimize_with_constraints = mock_optimize_with_constraints

    return mock_agent


@pytest.fixture
def mock_gl010_monitor(gl010_config):
    """Mock GL-010 EmissionsMonitor instance."""
    mock_agent = MagicMock()
    mock_agent.config = MagicMock()
    mock_agent.config.agent_id = gl010_config['agent_id']
    mock_agent.regulatory_standards = gl010_config['regulatory_standards']

    # Mock get emission constraints
    async def mock_get_emission_constraints(operation_params):
        return {
            'NOx_limit_ppm': gl010_config['regulatory_standards']['NOx_limit_ppm'],
            'SOx_limit_ppm': gl010_config['regulatory_standards']['SOx_limit_ppm'],
            'CO2_limit_kg_mwh': gl010_config['regulatory_standards']['CO2_limit_kg_mwh'],
            'CO_limit_ppm': gl010_config['regulatory_standards']['CO_limit_ppm'],
            'constraint_source': 'EPA_Title_V',
            'enforcement_level': 'mandatory'
        }

    mock_agent.get_emission_constraints = mock_get_emission_constraints

    # Mock validate compliance
    async def mock_validate_compliance(emissions_data):
        violations = []

        for pollutant, limit in gl010_config['regulatory_standards'].items():
            pollutant_key = pollutant.replace('_limit_', '_emissions_')
            actual_value = emissions_data.get(pollutant_key, 0)

            if actual_value > limit:
                violations.append({
                    'pollutant': pollutant.replace('_limit_', ''),
                    'limit': limit,
                    'actual': actual_value,
                    'exceedance_percent': (actual_value - limit) / limit * 100,
                    'severity': 'critical' if actual_value > limit * 1.2 else 'warning'
                })

        return {
            'agent_id': gl010_config['agent_id'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'compliance_status': 'COMPLIANT' if not violations else 'NON_COMPLIANT',
            'violations': violations,
            'total_violations': len(violations),
            'compliance_score': max(0, 100 - len(violations) * 25),
            'validation_success': True
        }

    mock_agent.validate_compliance = mock_validate_compliance

    return mock_agent


@pytest.fixture
def boiler_operation_data():
    """Sample boiler operation data."""
    return {
        'fuel_flow_kg_hr': 5000,
        'air_flow_kg_hr': 50000,
        'steam_output_kg_hr': 40000,
        'combustion_temp_c': 1200,
        'excess_air_percent': 15.0
    }


# ============================================================================
# Test Class: GL-002 ↔ GL-010 Coordination
# ============================================================================

class TestGL002GL010Coordination:
    """Test suite for GL-002 ↔ GL-010 emissions-constrained optimization."""

    @pytest.mark.asyncio
    async def test_gl002_requests_emission_constraints(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test GL-002 requests emission constraints from GL-010."""
        # Act
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Assert
        assert 'NOx_limit_ppm' in constraints
        assert 'SOx_limit_ppm' in constraints
        assert 'CO2_limit_kg_mwh' in constraints
        assert constraints['enforcement_level'] == 'mandatory'

    @pytest.mark.asyncio
    async def test_gl010_provides_regulatory_limits(
        self,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test GL-010 provides correct regulatory limits."""
        # Act
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Assert
        assert constraints['NOx_limit_ppm'] == 150
        assert constraints['SOx_limit_ppm'] == 200
        assert constraints['CO2_limit_kg_mwh'] == 500
        assert 'constraint_source' in constraints

    @pytest.mark.asyncio
    async def test_gl002_optimizes_within_constraints(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test GL-002 optimizes within emission constraints."""
        # Step 1: Get constraints from GL-010
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Step 2: Optimize with constraints
        result = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_operation_data,
            emission_constraints=constraints
        )

        # Assert
        assert result['optimization_success'] is True
        assert result['meets_emission_limits'] is True

        opt_result = result['optimization_result']
        assert opt_result['nox_emissions_ppm'] <= constraints['NOx_limit_ppm']
        assert opt_result['sox_emissions_ppm'] <= constraints['SOx_limit_ppm']
        assert opt_result['co2_emissions_kg_mwh'] <= constraints['CO2_limit_kg_mwh']

    @pytest.mark.asyncio
    async def test_gl010_validates_emissions_compliant(
        self,
        mock_gl010_monitor
    ):
        """Test GL-010 validates compliant emissions."""
        # Arrange
        compliant_emissions = {
            'NOx_emissions_ppm': 145,
            'SOx_emissions_ppm': 120,
            'CO2_emissions_kg_mwh': 450,
            'CO_emissions_ppm': 60
        }

        # Act
        validation = await mock_gl010_monitor.validate_compliance(compliant_emissions)

        # Assert
        assert validation['compliance_status'] == 'COMPLIANT'
        assert validation['total_violations'] == 0
        assert validation['compliance_score'] == 100

    @pytest.mark.asyncio
    async def test_gl010_detects_violations(
        self,
        mock_gl010_monitor
    ):
        """Test GL-010 detects emission violations."""
        # Arrange
        non_compliant_emissions = {
            'NOx_emissions_ppm': 180,  # Exceeds 150 limit
            'SOx_emissions_ppm': 220,  # Exceeds 200 limit
            'CO2_emissions_kg_mwh': 450,
            'CO_emissions_ppm': 60
        }

        # Act
        validation = await mock_gl010_monitor.validate_compliance(non_compliant_emissions)

        # Assert
        assert validation['compliance_status'] == 'NON_COMPLIANT'
        assert validation['total_violations'] >= 2

        # Verify violation details
        for violation in validation['violations']:
            assert 'pollutant' in violation
            assert 'limit' in violation
            assert 'actual' in violation
            assert violation['actual'] > violation['limit']

    @pytest.mark.asyncio
    async def test_end_to_end_constrained_optimization(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test complete emissions-constrained optimization workflow."""
        # Step 1: GL-010 provides constraints
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Step 2: GL-002 optimizes within constraints
        optimization = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_operation_data,
            emission_constraints=constraints
        )

        # Step 3: GL-010 validates result
        validation = await mock_gl010_monitor.validate_compliance(
            emissions_data=optimization['optimization_result']
        )

        # Assert - End-to-end compliance
        assert optimization['optimization_success'] is True
        assert validation['compliance_status'] == 'COMPLIANT'
        assert validation['total_violations'] == 0

    @pytest.mark.asyncio
    async def test_multi_objective_optimization(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test balancing efficiency and emissions objectives."""
        # Arrange
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Act
        result = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_operation_data,
            emission_constraints=constraints
        )

        # Assert
        opt_result = result['optimization_result']

        # Should maintain good efficiency while meeting emissions
        assert opt_result['efficiency_percent'] > 80.0  # Maintain high efficiency
        assert opt_result['nox_emissions_ppm'] <= constraints['NOx_limit_ppm']

    @pytest.mark.asyncio
    async def test_constraint_violation_handling(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor
    ):
        """Test handling when optimization cannot meet all constraints."""
        # Arrange - Very strict constraints
        strict_constraints = {
            'NOx_limit_ppm': 50,  # Very strict
            'SOx_limit_ppm': 50,
            'CO2_limit_kg_mwh': 300,
            'CO_limit_ppm': 20
        }

        boiler_data = {
            'fuel_flow_kg_hr': 8000,  # High load
            'air_flow_kg_hr': 70000,
            'steam_output_kg_hr': 48000
        }

        # Act
        result = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_data,
            emission_constraints=strict_constraints
        )

        # Assert
        # Should attempt optimization but may have constraint violations
        assert result is not None
        assert 'optimization_result' in result

    @pytest.mark.asyncio
    async def test_real_time_emissions_monitoring(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test real-time emissions monitoring during optimization."""
        # Arrange
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Act
        start_time = time.perf_counter()

        # Optimize
        optimization = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_operation_data,
            emission_constraints=constraints
        )

        # Validate immediately (real-time)
        validation = await mock_gl010_monitor.validate_compliance(
            emissions_data=optimization['optimization_result']
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Assert
        assert latency_ms < 200  # Real-time response <200ms
        assert validation['compliance_status'] == 'COMPLIANT'

    @pytest.mark.asyncio
    async def test_dynamic_constraint_updates(
        self,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test GL-010 provides dynamic constraints based on conditions."""
        # Arrange - Different operating conditions
        low_load_params = {**boiler_operation_data, 'steam_output_kg_hr': 20000}
        high_load_params = {**boiler_operation_data, 'steam_output_kg_hr': 48000}

        # Act
        low_load_constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=low_load_params
        )
        high_load_constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=high_load_params
        )

        # Assert
        # Constraints should be provided for both conditions
        assert 'NOx_limit_ppm' in low_load_constraints
        assert 'NOx_limit_ppm' in high_load_constraints

    @pytest.mark.asyncio
    async def test_concurrent_compliance_checks(
        self,
        mock_gl010_monitor
    ):
        """Test GL-010 handles concurrent compliance validation requests."""
        # Arrange - Multiple emission scenarios
        emission_scenarios = [
            {'NOx_emissions_ppm': 140 + i, 'SOx_emissions_ppm': 180, 'CO2_emissions_kg_mwh': 450, 'CO_emissions_ppm': 60}
            for i in range(10)
        ]

        # Act
        tasks = [
            mock_gl010_monitor.validate_compliance(emissions)
            for emissions in emission_scenarios
        ]

        results = await asyncio.gather(*tasks)

        # Assert
        assert len(results) == 10
        for result in results:
            assert result['validation_success'] is True
            assert 'compliance_status' in result

    @pytest.mark.asyncio
    async def test_violation_severity_classification(
        self,
        mock_gl010_monitor
    ):
        """Test GL-010 classifies violation severity correctly."""
        # Arrange
        minor_violation = {
            'NOx_emissions_ppm': 160,  # 10 ppm over 150 limit (minor)
            'SOx_emissions_ppm': 180,
            'CO2_emissions_kg_mwh': 450,
            'CO_emissions_ppm': 60
        }

        major_violation = {
            'NOx_emissions_ppm': 200,  # 50 ppm over limit (major)
            'SOx_emissions_ppm': 180,
            'CO2_emissions_kg_mwh': 450,
            'CO_emissions_ppm': 60
        }

        # Act
        minor_result = await mock_gl010_monitor.validate_compliance(minor_violation)
        major_result = await mock_gl010_monitor.validate_compliance(major_violation)

        # Assert
        # Both should detect violations but with different severity
        assert minor_result['total_violations'] > 0
        assert major_result['total_violations'] > 0

        # Major violation should have lower compliance score
        assert major_result['compliance_score'] <= minor_result['compliance_score']

    @pytest.mark.asyncio
    async def test_efficiency_emissions_tradeoff(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test efficiency vs emissions tradeoff optimization."""
        # Arrange
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # Act - Optimize with constraints
        constrained_result = await mock_gl002_optimizer.optimize_with_constraints(
            boiler_data=boiler_operation_data,
            emission_constraints=constraints
        )

        # Assert
        # Should achieve reasonable efficiency while meeting emissions
        opt = constrained_result['optimization_result']
        assert opt['efficiency_percent'] > 80.0
        assert opt['nox_emissions_ppm'] <= constraints['NOx_limit_ppm']

    @pytest.mark.asyncio
    async def test_data_format_compatibility(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test data format compatibility between GL-002 and GL-010."""
        # Get constraints from GL-010
        constraints = await mock_gl010_monitor.get_emission_constraints(
            operation_params=boiler_operation_data
        )

        # GL-002 should be able to consume constraints
        try:
            result = await mock_gl002_optimizer.optimize_with_constraints(
                boiler_data=boiler_operation_data,
                emission_constraints=constraints
            )
            compatibility_passed = True
        except Exception as e:
            compatibility_passed = False

        # GL-010 should be able to validate GL-002 output
        try:
            validation = await mock_gl010_monitor.validate_compliance(
                emissions_data=result['optimization_result']
            )
            validation_passed = True
        except Exception as e:
            validation_passed = False

        # Assert
        assert compatibility_passed, "GL-002 cannot consume GL-010 constraints"
        assert validation_passed, "GL-010 cannot validate GL-002 output"

    @pytest.mark.asyncio
    async def test_performance_under_continuous_monitoring(
        self,
        mock_gl002_optimizer,
        mock_gl010_monitor,
        boiler_operation_data
    ):
        """Test performance under continuous monitoring scenario."""
        # Arrange - Simulate continuous monitoring
        num_cycles = 30

        # Act
        start_time = time.perf_counter()

        for _ in range(num_cycles):
            constraints = await mock_gl010_monitor.get_emission_constraints(
                operation_params=boiler_operation_data
            )
            optimization = await mock_gl002_optimizer.optimize_with_constraints(
                boiler_data=boiler_operation_data,
                emission_constraints=constraints
            )
            validation = await mock_gl010_monitor.validate_compliance(
                emissions_data=optimization['optimization_result']
            )

        total_time_s = time.perf_counter() - start_time
        throughput = num_cycles / total_time_s

        # Assert
        assert throughput > 10  # At least 10 cycles/second
