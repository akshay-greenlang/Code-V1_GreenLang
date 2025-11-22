# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Basic Usage Examples

This script demonstrates common use cases for the SteamTrapInspector agent.
"""

import asyncio
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from steam_trap_inspector import SteamTrapInspector
from config import TrapInspectorConfig, TrapType, FailureMode


async def example_1_basic_monitoring():
    """
    Example 1: Basic steam trap monitoring with multi-modal analysis.

    Use case: Real-time monitoring of a single trap with acoustic and thermal sensors.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Steam Trap Monitoring")
    print("="*70)

    # Initialize agent
    config = TrapInspectorConfig(
        agent_id="GL-008-EXAMPLE-1",
        enable_llm_classification=False,  # Pure deterministic mode
        monitoring_interval_seconds=300
    )

    inspector = SteamTrapInspector(config)

    # Simulate sensor data
    # In production, this would come from ultrasonic sensors and IR cameras
    acoustic_signal = np.random.randn(10000) * 0.3  # Simulated ultrasonic data

    input_data = {
        'operation_mode': 'monitor',
        'trap_data': {
            'trap_id': 'TRAP-BOILER-ROOM-001',
            'trap_type': 'thermodynamic',
            'acoustic_data': {
                'signal': acoustic_signal.tolist(),
                'sampling_rate_hz': 250000,
                'measurement_duration_sec': 5.0
            },
            'thermal_data': {
                'temperature_upstream_c': 152.0,
                'temperature_downstream_c': 128.0,
                'ambient_temp_c': 22.0
            },
            'orifice_diameter_in': 0.125,
            'steam_pressure_psig': 100.0,
            'operating_hours_yr': 8760
        }
    }

    # Execute monitoring
    result = await inspector.execute(input_data)

    # Display results
    print(f"\nTrap Status:")
    print(f"  Trap ID: {result['trap_status']['trap_id']}")
    print(f"  Health Score: {result['trap_status']['health_score']:.1f}/100")
    print(f"  Operational Status: {result['trap_status']['operational_status']}")
    print(f"  Failure Mode: {result['trap_status']['failure_mode']}")
    print(f"  Severity: {result['trap_status']['failure_severity']}")

    print(f"\nAlerts: {len(result['alerts'])}")
    for alert in result['alerts']:
        print(f"  [{alert['level']}] {alert['message']}")

    print(f"\nRecommendations:")
    print(f"  Action: {result['recommendations']['action']}")
    print(f"  Urgency: {result['recommendations']['urgency_hours']} hours")

    print(f"\nExecution Time: {result['execution_time_ms']:.2f} ms")

    await inspector.shutdown()


async def example_2_comprehensive_diagnosis():
    """
    Example 2: Comprehensive failure diagnosis for a problematic trap.

    Use case: Detailed troubleshooting of a trap showing signs of failure.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Comprehensive Failure Diagnosis")
    print("="*70)

    config = TrapInspectorConfig(agent_id="GL-008-EXAMPLE-2")
    inspector = SteamTrapInspector(config)

    # High amplitude signal = likely failed open
    failed_signal = np.sin(2 * np.pi * 32000 * np.linspace(0, 1, 10000)) * 2.0

    input_data = {
        'operation_mode': 'diagnose',
        'trap_data': {
            'trap_id': 'TRAP-PRODUCTION-LINE-A-005',
            'trap_type': 'mechanical',
            'acoustic_data': {
                'signal': failed_signal.tolist(),
                'sampling_rate_hz': 250000
            },
            'thermal_data': {
                'temperature_upstream_c': 155.0,
                'temperature_downstream_c': 153.0  # Minimal delta = steam bypass
            },
            'pressure_upstream_psig': 100.0,
            'pressure_downstream_psig': 3.0,
            'orifice_diameter_in': 0.25,  # 1/4 inch
            'steam_pressure_psig': 100.0,
            'operating_hours_yr': 8760,
            'steam_cost_usd_per_1000lb': 8.50
        }
    }

    result = await inspector.execute(input_data)

    # Display diagnostic results
    print(f"\nDiagnostic Summary:")
    print(f"  Trap ID: {result['trap_id']}")
    print(f"  Failure Mode: {result['diagnostic_summary']['failure_mode']}")
    print(f"  Root Cause: {result['diagnostic_summary']['root_cause']}")
    print(f"  Severity: {result['diagnostic_summary']['severity']}")
    print(f"  Confidence: {result['diagnostic_summary']['confidence']:.1%}")

    if result['impact_assessment']['energy_loss']:
        energy = result['impact_assessment']['energy_loss']
        print(f"\nEnergy Loss Impact:")
        print(f"  Steam Loss: {energy['steam_loss_kg_hr']:.1f} kg/hr")
        print(f"  Annual Energy Loss: {energy['energy_loss_gj_yr']:.1f} GJ/year")
        print(f"  Annual Cost: ${energy['cost_loss_usd_yr']:,.0f}/year")
        print(f"  CO2 Emissions: {energy['co2_emissions_tons_yr']:.1f} tons/year")

    if result['impact_assessment']['cost_benefit']:
        cb = result['impact_assessment']['cost_benefit']
        print(f"\nFinancial Analysis:")
        print(f"  Recommended Action: {cb['decision_recommendation']}")
        print(f"  NPV (5-year): ${cb['npv_usd']:,.0f}")
        print(f"  ROI: {cb['roi_percent']:.1f}%")
        print(f"  Payback: {cb['payback_months']:.1f} months")

    await inspector.shutdown()


async def example_3_fleet_optimization():
    """
    Example 3: Fleet-wide maintenance prioritization.

    Use case: Optimize maintenance schedule for 50+ traps across a facility.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Fleet-Wide Maintenance Optimization")
    print("="*70)

    config = TrapInspectorConfig(agent_id="GL-008-EXAMPLE-3")
    inspector = SteamTrapInspector(config)

    # Simulate fleet of 20 traps with varying conditions
    fleet_data = []
    for i in range(1, 21):
        trap = {
            'trap_id': f'TRAP-FACILITY-{i:03d}',
            'failure_mode': np.random.choice([
                'normal', 'leaking', 'failed_open', 'worn_seat'
            ]),
            'energy_loss_usd_yr': np.random.uniform(0, 20000),
            'process_criticality': np.random.randint(1, 11),
            'current_age_years': np.random.uniform(0, 15)
        }
        fleet_data.append(trap)

    input_data = {
        'operation_mode': 'prioritize',
        'fleet_data': fleet_data
    }

    result = await inspector.execute(input_data)

    # Display fleet summary
    print(f"\nFleet Summary:")
    print(f"  Total Traps: {result['fleet_summary']['total_traps']}")
    print(f"  Failures Detected: {result['fleet_summary']['failures_detected']}")
    print(f"  Total Potential Savings: ${result['fleet_summary']['total_potential_savings_usd_yr']:,.0f}/year")
    print(f"  Estimated Maintenance Cost: ${result['fleet_summary']['estimated_maintenance_cost_usd']:,.0f}")

    print(f"\nFinancial Summary:")
    print(f"  Expected ROI: {result['financial_summary']['expected_roi_percent']:.1f}%")
    print(f"  Payback Period: {result['financial_summary']['payback_months']:.1f} months")
    print(f"  Net Annual Benefit: ${result['financial_summary']['net_annual_benefit_usd']:,.0f}")

    print(f"\nTop 5 Priority Traps:")
    for i, trap in enumerate(result['prioritized_maintenance_plan']['priority_list'][:5], 1):
        print(f"  {i}. {trap['trap_id']}: "
              f"Score={trap['priority_score']:.2f}, "
              f"Loss=${trap['energy_loss_usd_yr']:,.0f}/yr, "
              f"Action={trap['recommended_action'][:50]}")

    print(f"\nMaintenance Schedule:")
    for phase in result['prioritized_maintenance_plan']['phased_schedule']:
        print(f"  Phase {phase['phase']}: {phase['description']}")
        print(f"    {phase['trap_count']} traps, "
              f"{phase['start_date'][:10]} to {phase['end_date'][:10]}")

    await inspector.shutdown()


async def example_4_predictive_maintenance():
    """
    Example 4: Predictive maintenance with RUL forecasting.

    Use case: Plan replacement schedule based on remaining useful life.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Predictive Maintenance & RUL Forecasting")
    print("="*70)

    config = TrapInspectorConfig(agent_id="GL-008-EXAMPLE-4")
    inspector = SteamTrapInspector(config)

    input_data = {
        'operation_mode': 'predict',
        'trap_data': {
            'trap_id': 'TRAP-CRITICAL-PROCESS-001',
            'current_age_days': 1500,  # 4.1 years old
            'degradation_rate': 0.12,  # Moderate degradation
            'current_health_score': 65,  # Declining health
            'historical_failures': [1800, 2000, 2100, 1950],  # Historical MTBF data
            'process_criticality': 9  # Highly critical
        }
    }

    result = await inspector.execute(input_data)

    # Display RUL predictions
    print(f"\nPredictive Maintenance Analysis:")
    print(f"  Trap ID: {result['trap_id']}")

    rul = result['predictive_maintenance']
    print(f"\nRemaining Useful Life:")
    print(f"  Predicted RUL: {rul['remaining_useful_life_days']:.0f} days ({rul['remaining_useful_life_days']/30:.1f} months)")
    print(f"  Confidence Interval (90%):")
    print(f"    Lower Bound: {rul['confidence_interval']['lower_days']:.0f} days")
    print(f"    Upper Bound: {rul['confidence_interval']['upper_days']:.0f} days")
    print(f"  Degradation Rate: {rul['degradation_rate_per_year']:.1%}/year")

    print(f"\nMaintenance Schedule:")
    schedule = result['maintenance_schedule']
    print(f"  Next Inspection: {schedule['next_inspection_date'][:10]}")
    print(f"  Inspection Frequency: Every {schedule['inspection_frequency_days']} days")
    print(f"  Replacement Planning: {schedule['replacement_planning']['recommended_date'][:10]}")
    print(f"  Confidence: {schedule['replacement_planning']['confidence']}")

    print(f"\nRecommendations:")
    recs = result['recommendations']
    print(f"  Next Inspection: {recs['next_inspection_date'][:10]}")
    print(f"  Inspection Frequency: Every {recs['inspection_frequency_days']} days")
    print(f"  Planning Horizon: {recs['replacement_planning_horizon_days']} days")

    await inspector.shutdown()


async def example_5_realtime_monitoring():
    """
    Example 5: Real-time continuous monitoring simulation.

    Use case: Monitor trap continuously and detect degradation over time.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Real-Time Continuous Monitoring")
    print("="*70)

    config = TrapInspectorConfig(
        agent_id="GL-008-EXAMPLE-5",
        enable_real_time_monitoring=True,
        monitoring_interval_seconds=60  # Check every minute
    )
    inspector = SteamTrapInspector(config)

    print("\nSimulating 5 inspection cycles...")
    print("(In production, this would run continuously)")

    for cycle in range(1, 6):
        print(f"\n--- Cycle {cycle} ---")

        # Simulate gradual degradation
        health_degradation = cycle * 5  # Health decreases each cycle
        noise_amplitude = 0.1 + (cycle * 0.1)  # Acoustic noise increases

        signal = np.random.randn(5000) * noise_amplitude

        input_data = {
            'operation_mode': 'monitor',
            'trap_data': {
                'trap_id': 'TRAP-MONITORED-CONTINUOUS-001',
                'acoustic_data': {
                    'signal': signal.tolist(),
                    'sampling_rate_hz': 250000
                },
                'thermal_data': {
                    'temperature_upstream_c': 150.0,
                    'temperature_downstream_c': 130.0 + (cycle * 2)  # Delta decreases
                }
            }
        }

        result = await inspector.execute(input_data)

        print(f"Health Score: {result['trap_status']['health_score']:.1f}/100")
        print(f"Failure Mode: {result['trap_status']['failure_mode']}")
        print(f"Execution Time: {result['execution_time_ms']:.2f} ms")
        print(f"Cache Hit Rate: {result['performance_metrics']['cache_hit_rate']:.1%}")

        # Simulate delay between cycles
        await asyncio.sleep(0.5)

    # Display final performance metrics
    print(f"\n--- Final Performance Metrics ---")
    state = inspector.get_state()
    print(f"Total Inspections: {state['performance_metrics']['inspections_performed']}")
    print(f"Average Time: {state['performance_metrics']['avg_inspection_time_ms']:.2f} ms")
    print(f"Cache Hit Rate: {state['performance_metrics']['cache_hit_rate']:.1%}")

    await inspector.shutdown()


async def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GL-008 TRAPCATCHER - Usage Examples")
    print("Automated Steam Trap Failure Detection & Diagnostics")
    print("="*70)

    try:
        await example_1_basic_monitoring()
        await example_2_comprehensive_diagnosis()
        await example_3_fleet_optimization()
        await example_4_predictive_maintenance()
        await example_5_realtime_monitoring()

        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
