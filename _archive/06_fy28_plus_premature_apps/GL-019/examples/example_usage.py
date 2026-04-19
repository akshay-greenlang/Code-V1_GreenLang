# -*- coding: utf-8 -*-
"""
GL-019 HEATSCHEDULER - Example Usage

This module demonstrates how to use the ProcessHeatingScheduler agent
to optimize process heating schedules and minimize energy costs.

Examples include:
    1. Basic schedule optimization
    2. Time-of-use tariff handling
    3. Demand response integration
    4. Multi-equipment scheduling
    5. Cost savings analysis

Author: GreenLang Team
Date: December 2025
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List

# Example 1: Basic Schedule Optimization
# ============================================================================


async def example_basic_optimization():
    """
    Demonstrate basic schedule optimization with simple inputs.

    This example shows how to:
    - Create production batches with heating requirements
    - Define energy tariff structure
    - Run the optimization engine
    - Review the optimized schedule
    """
    from GL_019.config import (
        AgentConfiguration,
        TariffConfiguration,
        EquipmentConfiguration,
        TariffType,
        EquipmentType,
        EquipmentStatus,
    )
    from GL_019.process_heating_scheduler_agent import (
        ProcessHeatingSchedulerAgent,
        ProductionBatch,
        HeatingRequirements,
        Equipment,
    )

    # Define equipment
    furnace_1 = Equipment(
        equipment_id="FURNACE-001",
        equipment_type=EquipmentType.ELECTRIC_FURNACE,
        capacity_kw=500.0,
        efficiency_percent=92.0,
        status=EquipmentStatus.AVAILABLE,
        max_temperature_c=1200.0,
    )

    # Define production batches
    batches = [
        ProductionBatch(
            batch_id="BATCH-001",
            product_id="STEEL-ALLOY-A",
            quantity=1000.0,
            deadline=datetime.now(timezone.utc) + timedelta(hours=8),
            priority="HIGH",
            heating_requirements=HeatingRequirements(
                equipment_type=EquipmentType.ELECTRIC_FURNACE,
                target_temperature_c=850.0,
                duration_minutes=120,
                power_kw=450.0,
            ),
        ),
        ProductionBatch(
            batch_id="BATCH-002",
            product_id="STEEL-ALLOY-B",
            quantity=500.0,
            deadline=datetime.now(timezone.utc) + timedelta(hours=12),
            priority="MEDIUM",
            heating_requirements=HeatingRequirements(
                equipment_type=EquipmentType.ELECTRIC_FURNACE,
                target_temperature_c=750.0,
                duration_minutes=90,
                power_kw=350.0,
            ),
        ),
    ]

    # Define Time-of-Use tariff
    tariff = TariffConfiguration(
        tariff_id="TOU-INDUSTRIAL-2025",
        utility_name="Example Utility",
        tariff_type=TariffType.TIME_OF_USE,
        rates={
            "off_peak": {"rate_kwh": 0.05, "hours": list(range(0, 6)) + list(range(22, 24))},
            "mid_peak": {"rate_kwh": 0.10, "hours": list(range(6, 14)) + list(range(18, 22))},
            "on_peak": {"rate_kwh": 0.20, "hours": list(range(14, 18))},
        },
        demand_charge_rate_kw=15.0,
        demand_window_minutes=15,
    )

    # Create agent configuration
    config = AgentConfiguration(
        agent_id="GL-019",
        agent_name="HEATSCHEDULER",
        version="1.0.0",
        tariff=tariff,
        equipment=[furnace_1],
    )

    # Initialize agent
    agent = ProcessHeatingSchedulerAgent(config)

    # Run optimization
    print("=" * 60)
    print("GL-019 HEATSCHEDULER - Basic Optimization Example")
    print("=" * 60)

    result = await agent.execute(batches)

    # Display results
    print(f"\nOptimization Status: {result.optimization_status}")
    print(f"Total Energy Cost: ${result.total_cost_usd:,.2f}")
    print(f"Peak Demand: {result.peak_demand_kw:.1f} kW")
    print(f"Solver Time: {result.solver_time_seconds:.3f}s")
    print(f"\nScheduled Tasks:")

    for task in result.optimized_schedule.tasks:
        print(f"  - {task.batch_id}: {task.scheduled_start} to {task.scheduled_end}")
        print(f"    Equipment: {task.equipment_id}, Cost: ${task.estimated_cost_usd:.2f}")

    print(f"\nProvenance Hash: {result.provenance_hash[:16]}...")

    return result


# Example 2: Cost Savings Analysis
# ============================================================================


async def example_savings_analysis():
    """
    Demonstrate cost savings calculation between baseline and optimized schedules.

    This example shows how to:
    - Compare baseline (FIFO) schedule to optimized schedule
    - Calculate savings by category
    - Project annual savings
    - Calculate ROI metrics
    """
    from GL_019.calculators import (
        EnergyCostCalculator,
        SavingsCalculator,
        TariffStructure,
        TariffRate,
        TariffType,
        ScheduleComparison,
    )

    # Create tariff structure
    tariff = TariffStructure(
        tariff_id="TOU-2025",
        utility_name="Example Utility",
        tariff_type=TariffType.TIME_OF_USE,
        rates=[
            TariffRate(period_name="off_peak", rate_kwh=0.05, start_hour=0, end_hour=6),
            TariffRate(period_name="mid_peak", rate_kwh=0.10, start_hour=6, end_hour=14),
            TariffRate(period_name="on_peak", rate_kwh=0.20, start_hour=14, end_hour=18),
            TariffRate(period_name="mid_peak", rate_kwh=0.10, start_hour=18, end_hour=22),
            TariffRate(period_name="off_peak", rate_kwh=0.05, start_hour=22, end_hour=24),
        ],
        demand_charge_rate_kw=15.0,
    )

    # Initialize calculators
    cost_calculator = EnergyCostCalculator(tariff)
    savings_calculator = SavingsCalculator()

    # Define baseline schedule (production order, no optimization)
    baseline_schedule = [
        {"start_hour": 14, "duration_hours": 2, "power_kw": 500},  # On-peak
        {"start_hour": 16, "duration_hours": 2, "power_kw": 400},  # On-peak
    ]

    # Define optimized schedule (shifted to off-peak)
    optimized_schedule = [
        {"start_hour": 2, "duration_hours": 2, "power_kw": 500},  # Off-peak
        {"start_hour": 4, "duration_hours": 2, "power_kw": 400},  # Off-peak
    ]

    # Calculate baseline cost
    baseline_energy_cost = sum(
        cost_calculator.calculate_energy_cost(
            power_kw=task["power_kw"],
            start_hour=task["start_hour"],
            duration_hours=task["duration_hours"],
        )
        for task in baseline_schedule
    )
    baseline_demand = max(task["power_kw"] for task in baseline_schedule)
    baseline_demand_cost = baseline_demand * tariff.demand_charge_rate_kw
    baseline_total = baseline_energy_cost + baseline_demand_cost

    # Calculate optimized cost
    optimized_energy_cost = sum(
        cost_calculator.calculate_energy_cost(
            power_kw=task["power_kw"],
            start_hour=task["start_hour"],
            duration_hours=task["duration_hours"],
        )
        for task in optimized_schedule
    )
    optimized_demand = max(task["power_kw"] for task in optimized_schedule)
    optimized_demand_cost = optimized_demand * tariff.demand_charge_rate_kw
    optimized_total = optimized_energy_cost + optimized_demand_cost

    # Calculate savings
    savings = baseline_total - optimized_total
    savings_percent = (savings / baseline_total) * 100

    print("=" * 60)
    print("GL-019 HEATSCHEDULER - Cost Savings Analysis")
    print("=" * 60)
    print(f"\nBaseline Schedule (On-Peak):")
    print(f"  Energy Cost: ${baseline_energy_cost:.2f}")
    print(f"  Demand Charge: ${baseline_demand_cost:.2f}")
    print(f"  Total: ${baseline_total:.2f}")

    print(f"\nOptimized Schedule (Off-Peak):")
    print(f"  Energy Cost: ${optimized_energy_cost:.2f}")
    print(f"  Demand Charge: ${optimized_demand_cost:.2f}")
    print(f"  Total: ${optimized_total:.2f}")

    print(f"\nSavings:")
    print(f"  Daily Savings: ${savings:.2f} ({savings_percent:.1f}%)")
    print(f"  Annual Savings: ${savings * 250:.2f} (250 operating days)")

    # ROI calculation
    implementation_cost = 50000  # Example implementation cost
    payback_months = implementation_cost / (savings * 250 / 12)
    print(f"\nROI Analysis:")
    print(f"  Implementation Cost: ${implementation_cost:,.0f}")
    print(f"  Payback Period: {payback_months:.1f} months")


# Example 3: Demand Response Event Handling
# ============================================================================


async def example_demand_response():
    """
    Demonstrate demand response event handling and schedule adjustment.

    This example shows how to:
    - Receive a demand response event signal
    - Identify shiftable loads
    - Adjust schedule to meet curtailment target
    - Calculate incentive earnings
    """
    print("=" * 60)
    print("GL-019 HEATSCHEDULER - Demand Response Example")
    print("=" * 60)

    # Simulated demand response event
    dr_event = {
        "event_id": "DR-2025-001",
        "event_type": "ECONOMIC",
        "start_time": datetime.now(timezone.utc) + timedelta(hours=2),
        "end_time": datetime.now(timezone.utc) + timedelta(hours=4),
        "curtailment_target_kw": 200,
        "incentive_rate_kwh": 0.50,  # $/kWh for curtailed load
    }

    # Current scheduled load during DR window
    scheduled_load = [
        {"task_id": "T001", "power_kw": 150, "shiftable": True, "deadline_flexible": True},
        {"task_id": "T002", "power_kw": 100, "shiftable": True, "deadline_flexible": False},
        {"task_id": "T003", "power_kw": 200, "shiftable": False, "deadline_flexible": False},
    ]

    print(f"\nDR Event Received:")
    print(f"  Event ID: {dr_event['event_id']}")
    print(f"  Type: {dr_event['event_type']}")
    print(f"  Window: {dr_event['start_time']} to {dr_event['end_time']}")
    print(f"  Curtailment Target: {dr_event['curtailment_target_kw']} kW")
    print(f"  Incentive: ${dr_event['incentive_rate_kwh']}/kWh")

    # Identify shiftable loads
    shiftable_kw = sum(
        task["power_kw"] for task in scheduled_load
        if task["shiftable"] and task["deadline_flexible"]
    )

    print(f"\nLoad Analysis:")
    print(f"  Total Scheduled: {sum(t['power_kw'] for t in scheduled_load)} kW")
    print(f"  Shiftable Load: {shiftable_kw} kW")

    # Calculate response
    curtailment_achieved = min(shiftable_kw, dr_event["curtailment_target_kw"])
    dr_window_hours = 2  # 2-hour event
    energy_curtailed = curtailment_achieved * dr_window_hours
    incentive_earned = energy_curtailed * dr_event["incentive_rate_kwh"]

    print(f"\nDR Response:")
    print(f"  Curtailment Achieved: {curtailment_achieved} kW")
    print(f"  Energy Curtailed: {energy_curtailed} kWh")
    print(f"  Incentive Earned: ${incentive_earned:.2f}")

    # Shifted tasks
    shifted_tasks = [
        task for task in scheduled_load
        if task["shiftable"] and task["deadline_flexible"]
    ]
    print(f"\nShifted Tasks:")
    for task in shifted_tasks:
        print(f"  - {task['task_id']}: {task['power_kw']} kW shifted to post-DR window")


# Example 4: Multi-Equipment Load Balancing
# ============================================================================


async def example_load_balancing():
    """
    Demonstrate load balancing across multiple heating equipment.

    This example shows how to:
    - Distribute heating tasks across available equipment
    - Balance load to avoid demand spikes
    - Optimize equipment utilization
    """
    print("=" * 60)
    print("GL-019 HEATSCHEDULER - Load Balancing Example")
    print("=" * 60)

    # Available equipment
    equipment = [
        {"id": "FURNACE-001", "capacity_kw": 500, "efficiency": 0.92, "current_load_kw": 0},
        {"id": "FURNACE-002", "capacity_kw": 400, "efficiency": 0.90, "current_load_kw": 0},
        {"id": "BOILER-001", "capacity_kw": 300, "efficiency": 0.88, "current_load_kw": 0},
    ]

    # Tasks to schedule
    tasks = [
        {"task_id": "T001", "power_kw": 350, "duration_min": 120},
        {"task_id": "T002", "power_kw": 250, "duration_min": 90},
        {"task_id": "T003", "power_kw": 200, "duration_min": 60},
        {"task_id": "T004", "power_kw": 300, "duration_min": 150},
    ]

    print(f"\nAvailable Equipment:")
    for eq in equipment:
        print(f"  - {eq['id']}: {eq['capacity_kw']} kW capacity, {eq['efficiency']*100:.0f}% efficiency")

    print(f"\nTasks to Schedule:")
    for task in tasks:
        print(f"  - {task['task_id']}: {task['power_kw']} kW for {task['duration_min']} min")

    # Simple load balancing algorithm (first-fit decreasing)
    tasks_sorted = sorted(tasks, key=lambda t: t["power_kw"], reverse=True)
    assignments = []

    for task in tasks_sorted:
        # Find equipment with sufficient capacity
        for eq in sorted(equipment, key=lambda e: e["current_load_kw"]):
            if eq["current_load_kw"] + task["power_kw"] <= eq["capacity_kw"]:
                assignments.append({
                    "task_id": task["task_id"],
                    "equipment_id": eq["id"],
                    "power_kw": task["power_kw"],
                })
                eq["current_load_kw"] += task["power_kw"]
                break

    print(f"\nOptimized Assignments:")
    for assignment in assignments:
        print(f"  - {assignment['task_id']} -> {assignment['equipment_id']} ({assignment['power_kw']} kW)")

    print(f"\nEquipment Utilization:")
    for eq in equipment:
        utilization = (eq["current_load_kw"] / eq["capacity_kw"]) * 100
        print(f"  - {eq['id']}: {eq['current_load_kw']}/{eq['capacity_kw']} kW ({utilization:.1f}%)")

    total_demand = sum(eq["current_load_kw"] for eq in equipment)
    print(f"\nPeak Demand: {total_demand} kW")


# Main Entry Point
# ============================================================================


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GL-019 HEATSCHEDULER - Usage Examples")
    print("=" * 60 + "\n")

    # Run examples
    # await example_basic_optimization()  # Uncomment when agent is fully configured

    print("\n")
    await example_savings_analysis()

    print("\n")
    await example_demand_response()

    print("\n")
    await example_load_balancing()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
