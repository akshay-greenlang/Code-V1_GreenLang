"""
ISA 18.2 Alarm Management Example for Process Heat Agents

This example demonstrates how to integrate ISA 18.2 alarm management
with a Process Heat furnace agent for real-world monitoring.

Features demonstrated:
- Alarm configuration with priorities
- Processing sensor data and triggering alarms
- Operator acknowledgment and metrics
- Flood and chattering detection
- Rationalization documentation
- Performance metrics per ISA 18.2

Run: python examples/isa_18_2_alarm_example.py
"""

from datetime import datetime, timedelta
from greenlang.safety.isa_18_2_alarms import (
    AlarmManager,
    AlarmPriority,
    AlarmState,
    AlarmType,
)


def setup_furnace_alarms(manager: AlarmManager):
    """Configure typical furnace alarms per ISA 18.2."""

    # High-priority temperature alarms (emergency shutdown)
    manager.configure_alarm(
        tag='FURNACE_TEMP_HI_HI',
        description='Furnace Temperature Critical High - Shutdown Required',
        priority=AlarmPriority.EMERGENCY,
        setpoint=550.0,
        deadband=10.0,
        alarm_type=AlarmType.ANALOG_HI_HI,
        units='degC'
    )
    manager.rationalize_alarm(
        tag='FURNACE_TEMP_HI_HI',
        consequence='Furnace temperature >550°C causes refractory damage and safety hazard',
        response='Immediately reduce fuel input to zero and verify cooling system',
        response_time_sec=5
    )

    # High-priority temperature alarm (corrective action)
    manager.configure_alarm(
        tag='FURNACE_TEMP_HI',
        description='Furnace Temperature High - Reduce Fuel',
        priority=AlarmPriority.HIGH,
        setpoint=480.0,
        deadband=5.0,
        alarm_type=AlarmType.ANALOG_HI,
        units='degC'
    )
    manager.rationalize_alarm(
        tag='FURNACE_TEMP_HI',
        consequence='Temperature runaway can lead to equipment damage',
        response='Reduce fuel input rate by 10-20%',
        response_time_sec=30
    )

    # Medium-priority pressure alarm
    manager.configure_alarm(
        tag='FURNACE_PRESS_LO',
        description='Furnace Pressure Low - Check Blower',
        priority=AlarmPriority.MEDIUM,
        setpoint=50.0,
        deadband=10.0,
        alarm_type=AlarmType.ANALOG_LO,
        units='Pa'
    )
    manager.rationalize_alarm(
        tag='FURNACE_PRESS_LO',
        consequence='Low furnace pressure indicates combustion air flow problem',
        response='Verify blower operation and check for ductwork blockages',
        response_time_sec=60
    )

    # Fuel flow alarm
    manager.configure_alarm(
        tag='FUEL_FLOW_LO',
        description='Fuel Flow Low - Check Pump',
        priority=AlarmPriority.HIGH,
        setpoint=50.0,
        deadband=5.0,
        alarm_type=AlarmType.ANALOG_LO,
        units='kg/h'
    )
    manager.rationalize_alarm(
        tag='FUEL_FLOW_LO',
        consequence='Low fuel flow reduces heating capacity',
        response='Verify fuel pump and check for filter blockage',
        response_time_sec=45
    )

    # Stack temperature warning
    manager.configure_alarm(
        tag='STACK_TEMP_HI',
        description='Stack Temperature High - Efficiency Loss',
        priority=AlarmPriority.LOW,
        setpoint=350.0,
        deadband=15.0,
        alarm_type=AlarmType.ANALOG_HI,
        units='degC'
    )
    manager.rationalize_alarm(
        tag='STACK_TEMP_HI',
        consequence='High stack temperature indicates efficiency loss',
        response='Clean heat exchanger tubes if due',
        response_time_sec=300  # 5 minutes - can defer
    )


def simulate_normal_operation(manager: AlarmManager):
    """Simulate furnace operating normally."""
    print("\n" + "="*70)
    print("SCENARIO 1: Normal Furnace Operation")
    print("="*70)

    sensor_readings = {
        'FURNACE_TEMP_HI_HI': 420.0,
        'FURNACE_TEMP_HI': 420.0,
        'FURNACE_PRESS_LO': 80.0,
        'FUEL_FLOW_LO': 100.0,
        'STACK_TEMP_HI': 250.0,
    }

    print("\nProcessing sensor readings (normal operation):")
    for tag, value in sensor_readings.items():
        result = manager.process_alarm(tag, value)
        print(f"  {tag}: {value} -> {result.new_state.value}")

    metrics = manager.get_alarm_metrics()
    print(f"\nMetrics:")
    print(f"  Standing alarms: {metrics.standing_alarm_count}")
    print(f"  Alarms per 10 min: {metrics.alarms_per_10min:.1f}")
    print(f"  Operator burden: {metrics.operator_burden}")


def simulate_temperature_upset(manager: AlarmManager):
    """Simulate furnace temperature rising - operator response."""
    print("\n" + "="*70)
    print("SCENARIO 2: Temperature Upset - Operator Response")
    print("="*70)

    # Temperature starts rising
    print("\nT=0s: Temperature starts rising...")
    result = manager.process_alarm('FURNACE_TEMP_HI', 475.0)
    if result.alarm_triggered:
        print(f"  ALARM TRIGGERED: FURNACE_TEMP_HI (ID: {result.alarm_id})")

    # Temperature continues rising
    print("\nT=5s: Temperature continues rising...")
    result = manager.process_alarm('FURNACE_TEMP_HI', 485.0)
    print(f"  Still high: {result.new_state.value}")

    # Operator acknowledges
    print("\nT=10s: Operator acknowledges alarm...")
    ack = manager.acknowledge_alarm(result.alarm_id, operator_id='OP-001')
    print(f"  Acknowledged in {ack.ack_time_sec:.1f}s by {ack.ack_operator_id}")
    print(f"  Operator reduces fuel input by 15%...")

    # Temperature comes back down
    print("\nT=30s: Temperature returns to normal...")
    result = manager.process_alarm('FURNACE_TEMP_HI', 470.0)
    print(f"  Temperature stabilizing: {result.new_state.value}")

    result = manager.process_alarm('FURNACE_TEMP_HI', 465.0)
    if result.alarm_cleared:
        print(f"  ALARM CLEARED: Temperature back to safe range")

    metrics = manager.get_alarm_metrics()
    print(f"\nMetrics after incident:")
    print(f"  Standing alarms: {metrics.standing_alarm_count}")
    if metrics.avg_ack_time_sec:
        print(f"  Avg acknowledgment time: {metrics.avg_ack_time_sec:.1f}s")


def simulate_critical_condition(manager: AlarmManager):
    """Simulate emergency condition - immediate shutdown needed."""
    print("\n" + "="*70)
    print("SCENARIO 3: Emergency Condition - Immediate Shutdown")
    print("="*70)

    # Critical temperature reached
    print("\nT=0s: CRITICAL TEMPERATURE REACHED!")
    result = manager.process_alarm('FURNACE_TEMP_HI_HI', 560.0)
    if result.alarm_triggered:
        alarm_id = result.alarm_id
        print(f"  EMERGENCY ALARM: {result.new_state.value} (ID: {alarm_id})")
        print(f"  Required action: IMMEDIATE SHUTDOWN")
        print(f"  Target response time: 5 seconds")

    # Operator immediately acknowledges
    print("\nT=2s: Operator immediately acknowledges...")
    ack = manager.acknowledge_alarm(alarm_id, operator_id='OP-001')
    print(f"  Acknowledged in {ack.ack_time_sec:.1f}s")
    print(f"  Operator initiates emergency shutdown...")

    # Condition clears quickly
    print("\nT=8s: Fuel cut off - temperature dropping...")
    result = manager.process_alarm('FURNACE_TEMP_HI_HI', 520.0)
    print(f"  Cooling in progress: {result.new_state.value}")

    result = manager.process_alarm('FURNACE_TEMP_HI_HI', 480.0)
    if result.alarm_cleared:
        print(f"  SAFE: Emergency condition cleared")

    metrics = manager.get_alarm_metrics()
    print(f"\nEmergency response metrics:")
    print(f"  Acknowledgment time: {ack.ack_time_sec:.1f}s (target 5s) [OK]")
    print(f"  Total resolution time: ~8s")


def simulate_chattering_alarm(manager: AlarmManager):
    """Simulate and detect chattering (fleeting) alarm."""
    print("\n" + "="*70)
    print("SCENARIO 4: Chattering Alarm Detection")
    print("="*70)

    print("\nSimulating temperature oscillating near setpoint...")

    now = datetime.now()

    # Trigger
    print("T=0ms: Value crosses setpoint -> UNACKNOWLEDGED")
    result1 = manager.process_alarm(
        'STACK_TEMP_HI', 351.0, timestamp=now
    )

    # Clear (within 1 second - chattering!)
    print("T=500ms: Value drops below setpoint -> CLEARED")
    result2 = manager.process_alarm(
        'STACK_TEMP_HI', 349.0, timestamp=now + timedelta(milliseconds=500)
    )

    # Trigger again
    print("T=1500ms: Value crosses again -> Detected as CHATTERING")
    result3 = manager.process_alarm(
        'STACK_TEMP_HI', 351.0, timestamp=now + timedelta(milliseconds=1500)
    )

    if result3.chattering:
        print(f"\n[WARNING] CHATTERING DETECTED: {result3.chattering}")
        print("  Action: Increase deadband or add sensor damping filter")

    metrics = manager.get_alarm_metrics()
    if metrics.chattering_alarms:
        print(f"\n  Chattering alarms: {metrics.chattering_alarms}")
        print("  Recommendation: Investigate sensor stability")


def simulate_alarm_flood(manager: AlarmManager):
    """Simulate alarm flood condition."""
    print("\n" + "="*70)
    print("SCENARIO 5: Alarm Flood Detection")
    print("="*70)

    # Configure multiple alarms quickly
    print("\nSimulating 12 alarms in 10 minutes (flood threshold = 10)...\n")

    # Simulate alarms from different process variables
    alarms_to_trigger = [
        ('FURNACE_TEMP_HI', 485.0),
        ('FURNACE_PRESS_LO', 40.0),
        ('FUEL_FLOW_LO', 45.0),
        ('STACK_TEMP_HI', 360.0),
    ]

    alarm_count = 0
    for tag, value in alarms_to_trigger * 3:  # Repeat 3 times = 12 alarms
        result = manager.process_alarm(tag, value)
        if result.alarm_triggered:
            alarm_count += 1
            print(f"  Alarm {alarm_count}: {tag} = {value}")

    # Check for flood
    is_flooded, counts = manager.check_alarm_flood(threshold=10)

    if is_flooded:
        print(f"\n[WARNING] ALARM FLOOD DETECTED!")
        print(f"  Total alarms in 10-minute window: {sum(counts.values())}")
        print(f"  Priority distribution:")
        for priority, count in counts.items():
            if count > 0:
                print(f"    {priority.value}: {count}")
        print(f"\n  Action: Suppress LOW and DIAGNOSTIC priority alarms")
        print(f"  Only EMERGENCY and HIGH priority will be displayed")

    metrics = manager.get_alarm_metrics()
    print(f"\nOperator burden: {metrics.operator_burden}")
    print(f"Alarms per 10 min: {metrics.alarms_per_10min:.0f} (target <10)")


def show_metrics_summary(manager: AlarmManager):
    """Display comprehensive metrics per ISA 18.2."""
    print("\n" + "="*70)
    print("ISA 18.2 PERFORMANCE METRICS SUMMARY")
    print("="*70)

    metrics = manager.get_alarm_metrics()

    print(f"\nOperator Performance (ISA 18.2 Section 5.1):")
    print(f"  Alarms per 10 minutes:")
    print(f"    Current: {metrics.alarms_per_10min:.1f}")
    print(f"    Target: <10 [PASS]" if metrics.alarms_per_10min < 10 else f"    Target: <10 [FAIL]")

    print(f"\n  Acknowledgment rate (10 min):")
    print(f"    Current: {metrics.ack_rate_10min_pct:.1f}%")
    print(f"    Target: >90% [PASS]" if metrics.ack_rate_10min_pct > 90 else f"    Target: >90% [FAIL]")

    if metrics.avg_ack_time_sec:
        print(f"\n  Average ack time: {metrics.avg_ack_time_sec:.1f}s")

    print(f"\n  Standing alarms (unacknowledged):")
    print(f"    Current: {metrics.standing_alarm_count}")
    print(f"    Target: 0 [PASS]" if metrics.standing_alarm_count == 0 else f"    Target: 0 [FAIL]")

    print(f"\n  Stale alarms (standing >1 hour):")
    print(f"    Current: {metrics.stale_alarm_count}")
    print(f"    Target: 0 [PASS]" if metrics.stale_alarm_count == 0 else f"    Target: 0 [FAIL]")

    print(f"\n  Alarm floods (last 10 min):")
    print(f"    Current: {metrics.flood_events_10min}")
    print(f"    Target: 0 [PASS]" if metrics.flood_events_10min == 0 else f"    Target: 0 [FAIL]")

    print(f"\n  Chattering alarms:")
    print(f"    Count: {len(metrics.chattering_alarms)}")
    if metrics.chattering_alarms:
        print(f"    Tags: {', '.join(metrics.chattering_alarms)}")
    print(f"    Target: 0 [PASS]" if not metrics.chattering_alarms else f"    Target: 0 [FAIL]")

    print(f"\n  Rationalization completeness:")
    print(f"    Current: {metrics.rationalization_completeness_pct:.1f}%")
    print(f"    Target: 100% [PASS]" if metrics.rationalization_completeness_pct == 100 else f"    Target: 100% [FAIL]")

    print(f"\nOperator Burden Assessment: {metrics.operator_burden}")
    if metrics.operator_burden == "NORMAL":
        print("  [OK] Operator workload is appropriate")
    elif metrics.operator_burden == "WARNING":
        print("  [WARNING] Operator workload is elevated")
    else:
        print("  [CRITICAL] Operator is overloaded")

    # Standing alarms detail
    standing = manager.get_standing_alarms()
    if standing:
        print(f"\nStanding Alarms (sorted by priority):")
        for alarm in standing:
            print(f"  {alarm.priority.value:12s}: {alarm.tag:25s} = {alarm.value}")


def main():
    """Run all ISA 18.2 alarm management examples."""

    print("\n" + "="*70)
    print("ISA 18.2 ALARM MANAGEMENT SYSTEM - PROCESS HEAT EXAMPLE")
    print("="*70)
    print("\nReference: ISA-18.2-2016 Management of Alarms and Events")
    print("           for the Process Industries\n")

    # Initialize manager
    manager = AlarmManager(config={
        'operator_id': 'OP-HEAT-01',
        'plant_id': 'PLANT-01',
    })

    # Configure furnace alarms
    setup_furnace_alarms(manager)
    print("[OK] Furnace alarms configured (5 alarms)")
    print("[OK] All alarms rationalized per ISA 18.2 Annex D")

    # Run scenarios
    simulate_normal_operation(manager)
    simulate_temperature_upset(manager)
    simulate_critical_condition(manager)
    simulate_chattering_alarm(manager)
    simulate_alarm_flood(manager)

    # Show final metrics
    show_metrics_summary(manager)

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  1. Alarms are configured with priority and rationalization")
    print("  2. Sensor values trigger alarms based on setpoints")
    print("  3. Operators acknowledge alarms (tracked for metrics)")
    print("  4. Flood and chattering detection prevents operator overload")
    print("  5. ISA 18.2 metrics assess system and operator performance")
    print("\nFor integration with your furnace agent, use:")
    print("  manager = AlarmManager(config)")
    print("  manager.configure_alarm(tag, desc, priority, setpoint, deadband)")
    print("  result = manager.process_alarm(tag, sensor_value)")


if __name__ == '__main__':
    main()
