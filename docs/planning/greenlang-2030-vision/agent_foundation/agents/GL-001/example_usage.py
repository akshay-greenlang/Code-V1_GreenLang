# -*- coding: utf-8 -*-
"""
Example usage of GL-001 ProcessHeatOrchestrator.

This script demonstrates typical usage patterns for the ProcessHeatOrchestrator
agent in production scenarios.
"""

import asyncio
import json
from datetime import datetime
from process_heat_orchestrator import ProcessHeatOrchestrator
from config import (
    ProcessHeatConfig,
    PlantConfiguration,
    SensorConfiguration,
    SCADAIntegration,
    ERPIntegration,
    OptimizationParameters
)


async def example_basic_usage():
    """Example 1: Basic thermal efficiency calculation and optimization."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Thermal Efficiency Calculation")
    print("="*80)

    # Configure the orchestrator
    config = ProcessHeatConfig(
        agent_id="GL-001",
        agent_name="ProcessHeatOrchestrator",
        version="1.0.0",
        plants=[
            PlantConfiguration(
                plant_id="PLANT-001",
                plant_name="Chemical Plant Alpha",
                plant_type="chemical",
                location="Houston, TX",
                capacity_mw=500.0,
                operating_hours_per_year=8400,
                max_temperature_c=850.0,
                min_temperature_c=150.0,
                nominal_pressure_bar=40.0,
                primary_fuel="natural_gas",
                secondary_fuels=["hydrogen", "biomass"],
                renewable_percentage=15.0
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
                calibration_date="2024-01-15",
                max_threshold=900.0,
                critical_threshold=950.0
            )
        ],
        scada_integration=SCADAIntegration(
            enabled=True,
            endpoint_url="opc.tcp://localhost:4840",
            polling_interval_seconds=5
        ),
        erp_integration=ERPIntegration(
            enabled=True,
            system_type="SAP",
            endpoint_url="https://erp.example.com/api",
            sync_interval_minutes=60
        ),
        optimization=OptimizationParameters(
            optimization_algorithm="linear_programming",
            objective_function="minimize_cost",
            max_temperature_variance_c=5.0,
            min_efficiency_percent=85.0,
            max_emissions_kg_per_mwh=200.0
        ),
        emission_regulations={
            'max_emissions_kg_mwh': 200,
            'co2_kg_mwh': 180,
            'nox_kg_mwh': 0.5,
            'sox_kg_mwh': 0.3
        }
    )

    # Initialize orchestrator
    orchestrator = ProcessHeatOrchestrator(config)

    # Prepare input data
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
            'heat_demands': {
                'reactor_1': 30,
                'reactor_2': 25,
                'boiler_1': 20,
                'dryer_1': 15
            },
            'heat_sources': {
                'furnace_1': 50,
                'furnace_2': 40,
                'heat_recovery_1': 10
            },
            'temperatures': {
                'reactor_1': 450,
                'reactor_2': 480,
                'boiler_1': 380,
                'dryer_1': 220
            }
        },
        'constraints': {
            'max_temperature_variance_c': 5.0,
            'min_efficiency_percent': 85.0,
            'max_emissions_kg_per_mwh': 200.0,
            'priority_units': ['reactor_1', 'reactor_2']
        },
        'emissions_data': {
            'co2_kg_hr': 15000,
            'nox_kg_hr': 20,
            'sox_kg_hr': 10,
            'particulate_kg_hr': 5,
            'heat_output_mw': 90
        }
    }

    # Execute orchestration
    result = await orchestrator.execute(input_data)

    # Display results
    print(f"\nAgent ID: {result['agent_id']}")
    print(f"Execution Time: {result['execution_time_ms']:.2f}ms")
    print(f"\nTHERMAL EFFICIENCY:")
    print(f"  Overall Efficiency: {result['thermal_efficiency']['overall_efficiency']}%")
    print(f"  Carnot Efficiency: {result['thermal_efficiency']['carnot_efficiency']}%")
    print(f"  Heat Recovery: {result['thermal_efficiency']['heat_recovery_efficiency']}%")
    print(f"\nHEAT DISTRIBUTION:")
    print(f"  Total Demand: {result['heat_distribution']['total_heat_demand_mw']} MW")
    print(f"  Total Supply: {result['heat_distribution']['total_heat_supply_mw']} MW")
    print(f"  Optimization Score: {result['heat_distribution']['optimization_score']}")
    print(f"  Constraints Satisfied: {result['heat_distribution']['constraints_satisfied']}")
    print(f"\nENERGY BALANCE:")
    print(f"  Input Energy: {result['energy_balance']['input_energy_mw']} MW")
    print(f"  Output Energy: {result['energy_balance']['output_energy_mw']} MW")
    print(f"  Losses: {result['energy_balance']['losses_mw']} MW")
    print(f"  Balance Error: {result['energy_balance']['balance_error_percent']}%")
    print(f"  Valid: {result['energy_balance']['is_valid']}")
    print(f"\nEMISSIONS COMPLIANCE:")
    print(f"  Total Emissions: {result['emissions_compliance']['total_emissions_kg_hr']} kg/hr")
    print(f"  Emission Intensity: {result['emissions_compliance']['emission_intensity_kg_mwh']} kg/MWh")
    print(f"  Compliance Status: {result['emissions_compliance']['compliance_status']}")
    print(f"  Margin: {result['emissions_compliance']['margin_percent']}%")
    print(f"\nPERFORMANCE METRICS:")
    print(f"  Calculations Performed: {result['performance_metrics']['calculations_performed']}")
    print(f"  Cache Hits: {result['performance_metrics']['cache_hits']}")
    print(f"  Cache Misses: {result['performance_metrics']['cache_misses']}")
    print(f"\nProvenance Hash: {result['provenance_hash']}")

    return orchestrator


async def example_multi_agent_coordination():
    """Example 2: Multi-agent coordination."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multi-Agent Coordination")
    print("="*80)

    # Use existing orchestrator from example 1
    config = ProcessHeatConfig(
        agent_id="GL-001",
        plants=[PlantConfiguration(
            plant_id="PLANT-001",
            plant_name="Test Plant",
            plant_type="chemical",
            location="Test",
            capacity_mw=100,
            max_temperature_c=850,
            min_temperature_c=150,
            nominal_pressure_bar=40,
            primary_fuel="natural_gas"
        )],
        sensors=[SensorConfiguration(
            sensor_id="TEMP-001",
            sensor_type="temperature",
            location="Test",
            unit="celsius",
            sampling_rate_hz=1.0,
            accuracy_percent=1.0,
            calibration_date="2024-01-01"
        )],
        emission_regulations={}
    )

    orchestrator = ProcessHeatOrchestrator(config)

    # Input with agent coordination
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
        },
        'coordinate_agents': True,
        'agent_ids': ['GL-002', 'GL-003', 'GL-004', 'GL-005'],
        'agent_commands': {
            'optimize_boilers': {
                'priority': 'high',
                'target_efficiency': 0.90,
                'timeout': 60
            },
            'recover_waste_heat': {
                'priority': 'medium',
                'min_recovery_mw': 10,
                'timeout': 90
            },
            'control_furnaces': {
                'priority': 'high',
                'target_temperature': 850,
                'timeout': 60
            },
            'monitor_emissions': {
                'priority': 'critical',
                'compliance_check': True,
                'timeout': 30
            }
        }
    }

    # Execute with coordination
    result = await orchestrator.execute(input_data)

    # Display coordination results
    if result.get('coordination_result'):
        coord = result['coordination_result']
        print(f"\nCOORDINATION RESULTS:")
        print(f"  Agents Coordinated: {coord['coordinated_agents']}")
        print(f"  Total Tasks: {coord['total_tasks']}")
        print(f"  Status: {coord['coordination_status']}")
        print(f"  Estimated Completion: {coord['estimated_completion_time']}")
        print(f"\nTASK ASSIGNMENTS:")
        for agent_id, tasks in coord['task_assignments'].items():
            print(f"  {agent_id}: {len(tasks)} tasks")
            for task in tasks:
                print(f"    - {task['task']} (priority: {task['priority']})")

    return orchestrator


async def example_scada_erp_integration():
    """Example 3: SCADA and ERP integration."""
    print("\n" + "="*80)
    print("EXAMPLE 3: SCADA and ERP Integration")
    print("="*80)

    config = ProcessHeatConfig(
        agent_id="GL-001",
        plants=[PlantConfiguration(
            plant_id="PLANT-001",
            plant_name="Test Plant",
            plant_type="chemical",
            location="Test",
            capacity_mw=100,
            max_temperature_c=850,
            min_temperature_c=150,
            nominal_pressure_bar=40,
            primary_fuel="natural_gas"
        )],
        sensors=[SensorConfiguration(
            sensor_id="TEMP-001",
            sensor_type="temperature",
            location="Test",
            unit="celsius",
            sampling_rate_hz=1.0,
            accuracy_percent=1.0,
            calibration_date="2024-01-01"
        )],
        emission_regulations={}
    )

    orchestrator = ProcessHeatOrchestrator(config)

    # SCADA data feed
    scada_feed = {
        'tags': {
            'TEMP_001': 523.5,
            'TEMP_002': 487.2,
            'PRES_001': 42.3,
            'PRES_002': 38.7,
            'FLOW_001': 156.7,
            'FLOW_002': 142.3,
            'ENERGY_001': 98.5
        },
        'quality': {
            'TEMP_001': 98,
            'TEMP_002': 96,
            'PRES_001': 95,
            'PRES_002': 94,
            'FLOW_001': 92,
            'FLOW_002': 91,
            'ENERGY_001': 97
        },
        'units': {
            'TEMP_001': 'celsius',
            'TEMP_002': 'celsius',
            'PRES_001': 'bar',
            'PRES_002': 'bar',
            'FLOW_001': 'kg/s',
            'FLOW_002': 'kg/s',
            'ENERGY_001': 'MW'
        },
        'alarm_limits': {
            'TEMP_001': {'high': 550, 'high_high': 600},
            'TEMP_002': {'high': 520, 'high_high': 570},
            'PRES_001': {'high': 45, 'high_high': 50}
        }
    }

    # Process SCADA data
    scada_result = await orchestrator.integrate_scada(scada_feed)
    print(f"\nSCADA INTEGRATION:")
    print(f"  Total Tags: {scada_result['quality_metrics']['total_tags']}")
    print(f"  Good Quality Tags: {scada_result['quality_metrics']['good_quality_tags']}")
    print(f"  Data Availability: {scada_result['quality_metrics']['data_availability']:.1f}%")
    print(f"  Active Alarms: {scada_result['quality_metrics']['active_alarms']}")
    if scada_result['alarms']:
        print(f"\n  ALARMS:")
        for alarm in scada_result['alarms']:
            print(f"    - {alarm['tag']}: {alarm['type']} ({alarm['severity']})")

    # ERP data feed
    erp_feed = {
        'costs': {
            'CC-001': {
                'fuel_cost': 50000,
                'electricity_cost': 15000,
                'maintenance_cost': 8000,
                'currency': 'USD',
                'period': 'monthly'
            },
            'CC-002': {
                'fuel_cost': 35000,
                'electricity_cost': 12000,
                'maintenance_cost': 6000,
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
            },
            'MAT-002': {
                'description': 'Electricity',
                'quantity': 50000,
                'unit': 'kWh',
                'unit_cost': 0.30
            }
        },
        'production': {
            'planned_output': 10000,
            'actual_output': 9500,
            'unit': 'tons'
        },
        'maintenance': {
            'scheduled_tasks': ['TASK-001', 'TASK-002', 'TASK-003'],
            'completed_tasks': ['TASK-001', 'TASK-002'],
            'pending_tasks': ['TASK-003']
        }
    }

    # Process ERP data
    erp_result = await orchestrator.integrate_erp(erp_feed)
    print(f"\nERP INTEGRATION:")
    print(f"  Total Costs: ${erp_result['summary']['total_costs']:,.2f}")
    print(f"  Material Value: ${erp_result['summary']['material_value']:,.2f}")
    print(f"  Production Efficiency: {erp_result['summary']['production_efficiency']:.1f}%")
    print(f"  Maintenance Completion: {erp_result['summary']['maintenance_completion_rate']:.1f}%")

    return orchestrator


async def example_performance_benchmark():
    """Example 4: Performance benchmarking."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Performance Benchmarking")
    print("="*80)

    config = ProcessHeatConfig(
        agent_id="GL-001",
        plants=[PlantConfiguration(
            plant_id="PLANT-001",
            plant_name="Test Plant",
            plant_type="chemical",
            location="Test",
            capacity_mw=100,
            max_temperature_c=850,
            min_temperature_c=150,
            nominal_pressure_bar=40,
            primary_fuel="natural_gas"
        )],
        sensors=[SensorConfiguration(
            sensor_id="TEMP-001",
            sensor_type="temperature",
            location="Test",
            unit="celsius",
            sampling_rate_hz=1.0,
            accuracy_percent=1.0,
            calibration_date="2024-01-01"
        )],
        emission_regulations={}
    )

    orchestrator = ProcessHeatOrchestrator(config)

    # Run multiple iterations
    iterations = 10
    execution_times = []

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

    print(f"\nRunning {iterations} iterations...")
    for i in range(iterations):
        result = await orchestrator.execute(input_data)
        execution_times.append(result['execution_time_ms'])
        print(f"  Iteration {i+1}: {result['execution_time_ms']:.2f}ms")

    # Calculate statistics
    avg_time = sum(execution_times) / len(execution_times)
    min_time = min(execution_times)
    max_time = max(execution_times)

    print(f"\nPERFORMANCE STATISTICS:")
    print(f"  Average Execution Time: {avg_time:.2f}ms")
    print(f"  Minimum Time: {min_time:.2f}ms")
    print(f"  Maximum Time: {max_time:.2f}ms")
    print(f"  Target Met (<2000ms): {'YES' if avg_time < 2000 else 'NO'}")

    # Get final state
    state = orchestrator.get_state()
    print(f"\nFINAL STATE:")
    print(f"  Total Calculations: {state['performance_metrics']['calculations_performed']}")
    print(f"  Cache Hits: {state['performance_metrics']['cache_hits']}")
    print(f"  Cache Misses: {state['performance_metrics']['cache_misses']}")
    cache_total = state['performance_metrics']['cache_hits'] + state['performance_metrics']['cache_misses']
    if cache_total > 0:
        cache_hit_rate = state['performance_metrics']['cache_hits'] / cache_total * 100
        print(f"  Cache Hit Rate: {cache_hit_rate:.1f}%")

    return orchestrator


async def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GL-001 ProcessHeatOrchestrator - Example Usage")
    print("="*80)

    # Run examples
    orchestrator1 = await example_basic_usage()
    orchestrator2 = await example_multi_agent_coordination()
    orchestrator3 = await example_scada_erp_integration()
    orchestrator4 = await example_performance_benchmark()

    # Shutdown all orchestrators
    print("\n" + "="*80)
    print("Shutting down orchestrators...")
    print("="*80)
    await orchestrator1.shutdown()
    await orchestrator2.shutdown()
    await orchestrator3.shutdown()
    await orchestrator4.shutdown()
    print("All orchestrators shut down successfully.")


if __name__ == "__main__":
    asyncio.run(main())