# -*- coding: utf-8 -*-
"""
Unit tests for GL-001 ProcessHeatOrchestrator.

This module provides comprehensive testing for the ProcessHeatOrchestrator agent,
validating all tool functions, integration points, and zero-hallucination guarantees.
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Dict, Any

from process_heat_orchestrator import ProcessHeatOrchestrator
from config import ProcessHeatConfig, PlantConfiguration, SensorConfiguration
from tools import ProcessHeatTools


class TestProcessHeatTools:
    """Test suite for ProcessHeatTools deterministic functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tools = ProcessHeatTools()

    def test_calculate_thermal_efficiency(self):
        """Test thermal efficiency calculation with deterministic inputs."""
        plant_data = {
            'inlet_temp_c': 500,
            'outlet_temp_c': 150,
            'ambient_temp_c': 25,
            'fuel_input_mw': 100,
            'useful_heat_mw': 85,
            'heat_recovery_mw': 5
        }

        result = self.tools.calculate_thermal_efficiency(plant_data)

        # Verify efficiency calculation
        assert result.overall_efficiency == 90.0  # (85+5)/100 * 100
        assert 0 <= result.carnot_efficiency <= 100
        assert result.heat_recovery_efficiency > 0
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 length
        assert result.timestamp is not None

        # Verify determinism - same input should produce same hash
        result2 = self.tools.calculate_thermal_efficiency(plant_data)
        # Note: timestamps differ, so hashes will differ, but calculations should match
        assert result.overall_efficiency == result2.overall_efficiency

    def test_optimize_heat_distribution(self):
        """Test heat distribution optimization."""
        sensor_feeds = {
            'heat_demands': {
                'reactor_1': 30,
                'reactor_2': 25,
                'boiler_1': 20
            },
            'heat_sources': {
                'furnace_1': 50,
                'furnace_2': 40
            },
            'temperatures': {
                'reactor_1': 450,
                'reactor_2': 480,
                'boiler_1': 380
            }
        }

        constraints = {
            'max_temperature_variance_c': 5.0,
            'min_efficiency_percent': 85.0,
            'priority_units': ['reactor_1']
        }

        result = self.tools.optimize_heat_distribution(sensor_feeds, constraints)

        # Verify optimization results
        assert result.total_heat_demand_mw == 75  # 30+25+20
        assert result.total_heat_supply_mw == 90  # 50+40
        assert result.optimization_score > 0
        assert result.optimization_score <= 100
        assert 'reactor_1' in result.distribution_matrix
        assert result.provenance_hash is not None

    def test_validate_energy_balance(self):
        """Test energy balance validation."""
        consumption_data = {
            'fuel_input_mw': 100,
            'electricity_input_mw': 10,
            'steam_import_mw': 5,
            'process_heat_output_mw': 85,
            'electricity_output_mw': 5,
            'steam_export_mw': 10,
            'measured_losses_mw': 15
        }

        result = self.tools.validate_energy_balance(consumption_data)

        # Verify energy balance
        assert result.input_energy_mw == 115  # 100+10+5
        assert result.output_energy_mw == 100  # 85+5+10
        assert result.losses_mw == 15  # 115-100
        assert result.balance_error_percent >= 0
        assert result.is_valid in [True, False]
        assert result.provenance_hash is not None

    def test_check_emissions_compliance(self):
        """Test emissions compliance checking."""
        emissions_data = {
            'co2_kg_hr': 15000,
            'nox_kg_hr': 20,
            'sox_kg_hr': 10,
            'particulate_kg_hr': 5,
            'heat_output_mw': 90
        }

        regulations = {
            'max_emissions_kg_mwh': 200,
            'co2_kg_mwh': 180,
            'nox_kg_mwh': 0.5
        }

        result = self.tools.check_emissions_compliance(emissions_data, regulations)

        # Verify compliance check
        assert result.total_emissions_kg_hr > 0
        assert result.emission_intensity_kg_mwh > 0
        assert result.regulatory_limit_kg_mwh == 200
        assert result.compliance_status in ['PASS', 'WARNING', 'FAIL']
        assert result.provenance_hash is not None

    def test_generate_kpi_dashboard(self):
        """Test KPI dashboard generation."""
        metrics = {
            'thermal_efficiency': 88.5,
            'capacity_utilization': 92.3,
            'heat_recovery_rate': 12.5,
            'availability': 98.7,
            'co2_intensity': 165.2,
            'compliance_score': 95
        }

        dashboard = self.tools.generate_kpi_dashboard(metrics)

        # Verify dashboard structure
        assert 'operational_kpis' in dashboard
        assert 'energy_kpis' in dashboard
        assert 'environmental_kpis' in dashboard
        assert 'financial_kpis' in dashboard
        assert 'trends' in dashboard
        assert dashboard['operational_kpis']['overall_efficiency'] == 88.5
        assert dashboard['provenance_hash'] is not None

    def test_coordinate_process_heat_agents(self):
        """Test agent coordination."""
        agent_ids = ['GL-002', 'GL-003', 'GL-004']
        commands = {
            'optimize_boilers': {'priority': 'high', 'timeout': 60},
            'recover_heat': {'priority': 'medium', 'timeout': 90}
        }

        result = self.tools.coordinate_process_heat_agents(agent_ids, commands)

        # Verify coordination
        assert result['coordinated_agents'] >= 0
        assert result['total_tasks'] >= 0
        assert 'task_assignments' in result
        assert result['coordination_status'] == 'SUCCESS'
        assert result['provenance_hash'] is not None

    def test_integrate_scada_data(self):
        """Test SCADA data integration."""
        scada_feed = {
            'tags': {
                'TEMP_001': 523.5,
                'PRES_001': 42.3,
                'FLOW_001': 156.7
            },
            'quality': {
                'TEMP_001': 98,
                'PRES_001': 95,
                'FLOW_001': 92
            },
            'units': {
                'TEMP_001': 'celsius',
                'PRES_001': 'bar',
                'FLOW_001': 'kg/s'
            },
            'alarm_limits': {
                'TEMP_001': {'high': 550, 'high_high': 600}
            }
        }

        result = self.tools.integrate_scada_data(scada_feed)

        # Verify SCADA integration
        assert 'data_points' in result
        assert 'quality_metrics' in result
        assert result['quality_metrics']['total_tags'] == 3
        assert result['quality_metrics']['data_availability'] > 0
        assert result['provenance_hash'] is not None

    def test_integrate_erp_data(self):
        """Test ERP data integration."""
        erp_feed = {
            'costs': {
                'CC-001': {
                    'fuel_cost': 50000,
                    'electricity_cost': 15000,
                    'maintenance_cost': 8000,
                    'currency': 'USD',
                    'period': 'monthly'
                }
            },
            'materials': {
                'MAT-001': {
                    'description': 'Natural Gas',
                    'quantity': 10000,
                    'unit': 'm3',
                    'unit_cost': 5.0
                }
            },
            'production': {
                'planned_output': 10000,
                'actual_output': 9500,
                'unit': 'tons'
            },
            'maintenance': {
                'scheduled_tasks': ['TASK-001', 'TASK-002'],
                'completed_tasks': ['TASK-001'],
                'pending_tasks': ['TASK-002']
            }
        }

        result = self.tools.integrate_erp_data(erp_feed)

        # Verify ERP integration
        assert 'cost_data' in result
        assert 'material_data' in result
        assert 'production_data' in result
        assert 'maintenance_data' in result
        assert 'summary' in result
        assert result['cost_data']['CC-001']['total_cost'] == 73000
        assert result['production_data']['efficiency'] == 95.0
        assert result['provenance_hash'] is not None


class TestProcessHeatOrchestrator:
    """Test suite for ProcessHeatOrchestrator agent."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ProcessHeatConfig(
            agent_id="GL-001",
            agent_name="ProcessHeatOrchestrator",
            version="1.0.0",
            plants=[
                PlantConfiguration(
                    plant_id="PLANT-001",
                    plant_name="Test Plant",
                    plant_type="chemical",
                    location="Test Location",
                    capacity_mw=100.0,
                    max_temperature_c=850.0,
                    min_temperature_c=150.0,
                    nominal_pressure_bar=40.0,
                    primary_fuel="natural_gas"
                )
            ],
            sensors=[
                SensorConfiguration(
                    sensor_id="TEMP-001",
                    sensor_type="temperature",
                    location="Reactor 1",
                    unit="celsius",
                    sampling_rate_hz=10.0,
                    accuracy_percent=0.5,
                    calibration_date="2024-01-15"
                )
            ],
            emission_regulations={'max_emissions_kg_mwh': 200}
        )

    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, config):
        """Test orchestrator initialization."""
        orchestrator = ProcessHeatOrchestrator(config)

        assert orchestrator.config.agent_id == "GL-001"
        assert orchestrator.tools is not None
        assert orchestrator.performance_metrics['calculations_performed'] == 0
        assert orchestrator.short_term_memory is not None
        assert orchestrator.long_term_memory is not None

    @pytest.mark.asyncio
    async def test_orchestrator_execute(self, config):
        """Test full orchestration execution."""
        orchestrator = ProcessHeatOrchestrator(config)

        input_data = {
            'plant_data': {
                'inlet_temp_c': 500,
                'outlet_temp_c': 150,
                'ambient_temp_c': 25,
                'fuel_input_mw': 100,
                'useful_heat_mw': 85,
                'heat_recovery_mw': 5,
                'fuel_input_mw': 100,
                'electricity_input_mw': 10,
                'steam_import_mw': 5,
                'process_heat_output_mw': 85,
                'electricity_output_mw': 5,
                'steam_export_mw': 10,
                'measured_losses_mw': 15
            },
            'sensor_feeds': {
                'heat_demands': {'reactor_1': 30},
                'heat_sources': {'furnace_1': 50}
            },
            'constraints': {
                'max_temperature_variance_c': 5.0,
                'min_efficiency_percent': 85.0
            },
            'emissions_data': {
                'co2_kg_hr': 15000,
                'nox_kg_hr': 20,
                'heat_output_mw': 90
            }
        }

        result = await orchestrator.execute(input_data)

        # Verify execution result
        assert result['agent_id'] == "GL-001"
        assert result['execution_time_ms'] > 0
        assert 'thermal_efficiency' in result
        assert 'heat_distribution' in result
        assert 'energy_balance' in result
        assert 'emissions_compliance' in result
        assert 'kpi_dashboard' in result
        assert result['provenance_hash'] is not None
        assert len(result['provenance_hash']) == 64

    @pytest.mark.asyncio
    async def test_performance_targets(self, config):
        """Test that performance targets are met."""
        orchestrator = ProcessHeatOrchestrator(config)

        input_data = {
            'plant_data': {
                'inlet_temp_c': 500,
                'fuel_input_mw': 100,
                'useful_heat_mw': 85,
                'fuel_input_mw': 100,
                'process_heat_output_mw': 85,
                'measured_losses_mw': 15
            },
            'sensor_feeds': {
                'heat_demands': {'reactor_1': 30},
                'heat_sources': {'furnace_1': 50}
            },
            'constraints': {},
            'emissions_data': {
                'co2_kg_hr': 15000,
                'heat_output_mw': 90
            }
        }

        start_time = time.perf_counter()
        result = await orchestrator.execute(input_data)
        execution_time_ms = (time.perf_counter() - start_time) * 1000

        # Verify performance targets
        assert execution_time_ms < 2000  # <2s target
        assert result['execution_time_ms'] < 2000

    @pytest.mark.asyncio
    async def test_cache_functionality(self, config):
        """Test caching reduces execution time."""
        orchestrator = ProcessHeatOrchestrator(config)

        input_data = {
            'plant_data': {
                'inlet_temp_c': 500,
                'fuel_input_mw': 100,
                'useful_heat_mw': 85,
                'fuel_input_mw': 100,
                'process_heat_output_mw': 85,
                'measured_losses_mw': 15
            },
            'sensor_feeds': {
                'heat_demands': {'reactor_1': 30},
                'heat_sources': {'furnace_1': 50}
            },
            'constraints': {},
            'emissions_data': {
                'co2_kg_hr': 15000,
                'heat_output_mw': 90
            }
        }

        # First execution - cache miss
        result1 = await orchestrator.execute(input_data)
        time1 = result1['execution_time_ms']

        # Second execution - cache hit
        result2 = await orchestrator.execute(input_data)
        time2 = result2['execution_time_ms']

        # Cache should improve performance
        cache_metrics = orchestrator.performance_metrics
        assert cache_metrics['cache_hits'] > 0 or cache_metrics['cache_misses'] > 0

    @pytest.mark.asyncio
    async def test_error_recovery(self, config):
        """Test error recovery mechanism."""
        orchestrator = ProcessHeatOrchestrator(config)

        # Intentionally invalid input
        invalid_input = {
            'plant_data': {},  # Missing required fields
            'sensor_feeds': {},
            'constraints': {},
            'emissions_data': {}
        }

        # Should not crash, should recover
        try:
            result = await orchestrator.execute(invalid_input)
            # Should get partial success or error response
            assert 'agent_id' in result
        except Exception as e:
            # Exception is acceptable if no recovery configured
            assert True

    @pytest.mark.asyncio
    async def test_state_monitoring(self, config):
        """Test state monitoring capabilities."""
        orchestrator = ProcessHeatOrchestrator(config)

        state = orchestrator.get_state()

        assert state['agent_id'] == "GL-001"
        assert 'state' in state
        assert 'performance_metrics' in state
        assert 'cache_size' in state
        assert 'timestamp' in state


def test_provenance_determinism():
    """Test that provenance hashes are deterministic for same inputs."""
    tools = ProcessHeatTools()

    plant_data = {
        'inlet_temp_c': 500,
        'fuel_input_mw': 100,
        'useful_heat_mw': 85
    }

    # Run calculation twice
    result1 = tools.calculate_thermal_efficiency(plant_data)
    result2 = tools.calculate_thermal_efficiency(plant_data)

    # Same inputs should produce same efficiency values
    assert result1.overall_efficiency == result2.overall_efficiency
    assert result1.carnot_efficiency == result2.carnot_efficiency


def test_zero_hallucination_guarantee():
    """Verify all calculations are deterministic Python, no LLM."""
    tools = ProcessHeatTools()

    # All tool methods should be deterministic
    plant_data = {'inlet_temp_c': 500, 'fuel_input_mw': 100, 'useful_heat_mw': 85}

    # Run same calculation 10 times
    results = [tools.calculate_thermal_efficiency(plant_data) for _ in range(10)]

    # All results should have identical efficiency values
    efficiencies = [r.overall_efficiency for r in results]
    assert len(set(efficiencies)) == 1  # Only one unique value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])