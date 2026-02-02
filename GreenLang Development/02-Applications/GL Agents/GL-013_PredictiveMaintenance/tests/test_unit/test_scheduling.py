# -*- coding: utf-8 -*-
import pytest
from datetime import datetime, timedelta
import sys
sys.path.insert(0, str(r"c:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-013_PredictiveMaintenance"))


class TestMaintenanceScheduler:
    def test_scheduler_initialization(self):
        from scheduling.maintenance_scheduler import MaintenanceScheduler, SchedulerConfig
        config = SchedulerConfig(planning_horizon_days=30)
        scheduler = MaintenanceScheduler(config)
        assert scheduler.config.planning_horizon_days == 30
    
    def test_add_maintenance_window(self):
        from scheduling.maintenance_scheduler import MaintenanceScheduler, MaintenanceWindow
        scheduler = MaintenanceScheduler()
        window = MaintenanceWindow(
            start_time=datetime.utcnow() + timedelta(days=1),
            end_time=datetime.utcnow() + timedelta(days=1, hours=8),
        )
        scheduler.add_maintenance_window(window)
        windows = scheduler.get_available_windows(
            datetime.utcnow(),
            datetime.utcnow() + timedelta(days=7)
        )
        assert len(windows) == 1
    
    def test_schedule_task(self):
        from scheduling.maintenance_scheduler import (
            MaintenanceScheduler, ScheduledTask, SchedulePriority, MaintenanceType, ScheduleStatus
        )
        scheduler = MaintenanceScheduler()
        task = ScheduledTask(
            task_id="TSK001",
            asset_id="PUMP001",
            maintenance_type=MaintenanceType.PREDICTIVE,
            priority=SchedulePriority.HIGH,
            scheduled_start=datetime.utcnow() + timedelta(days=2),
            estimated_duration_hours=4.0,
            required_skills=["mechanical"],
            required_parts=["bearing"],
        )
        scheduled = scheduler.schedule_maintenance(task)
        assert scheduled.status == ScheduleStatus.SCHEDULED
    
    def test_optimize_schedule(self):
        from scheduling.maintenance_scheduler import (
            MaintenanceScheduler, ScheduledTask, SchedulePriority, MaintenanceType
        )
        scheduler = MaintenanceScheduler()
        tasks = [
            ScheduledTask(
                task_id=f"TSK00{i}",
                asset_id=f"PUMP00{i}",
                maintenance_type=MaintenanceType.PREDICTIVE,
                priority=SchedulePriority.MEDIUM,
                scheduled_start=datetime.utcnow(),
                estimated_duration_hours=2.0,
                required_skills=["mechanical"],
                required_parts=[],
                health_index=0.7,
            )
            for i in range(3)
        ]
        result = scheduler.optimize_schedule(tasks)
        assert len(result.schedule) == 3
        assert result.provenance_hash is not None
        assert result.optimization_score > 0


class TestInventoryPlanner:
    def test_planner_initialization(self):
        from scheduling.inventory_planner import InventoryPlanner, InventoryConfig
        config = InventoryConfig(target_service_level=0.95)
        planner = InventoryPlanner(config)
        assert planner.config.target_service_level == 0.95
    
    def test_register_part(self):
        from scheduling.inventory_planner import InventoryPlanner, SparePart, PartCriticality
        planner = InventoryPlanner()
        part = SparePart(
            part_id="BRG001",
            part_number="6205-2RS",
            description="Ball bearing 6205",
            criticality=PartCriticality.HIGH,
            unit_cost=25.50,
            lead_time_days=7,
        )
        planner.register_part(part)
        assert "BRG001" in planner._parts
    
    def test_calculate_safety_stock(self):
        from scheduling.inventory_planner import InventoryPlanner, SparePart, PartCriticality
        planner = InventoryPlanner()
        part = SparePart(
            part_id="BRG001",
            part_number="6205-2RS",
            description="Ball bearing",
            criticality=PartCriticality.HIGH,
            unit_cost=25.50,
            lead_time_days=7,
        )
        planner.register_part(part)
        safety_stock = planner.calculate_safety_stock("BRG001", demand_std=5.0)
        assert safety_stock >= 1
    
    def test_generate_replenishment_plan(self):
        from scheduling.inventory_planner import (
            InventoryPlanner, SparePart, PartCriticality, InventoryLevel
        )
        planner = InventoryPlanner()
        part = SparePart(
            part_id="BRG001",
            part_number="6205-2RS",
            description="Ball bearing",
            criticality=PartCriticality.HIGH,
            unit_cost=25.50,
            lead_time_days=7,
            supplier_ids=["SUP001"],
        )
        planner.register_part(part)
        planner.update_inventory(InventoryLevel(
            part_id="BRG001",
            quantity_on_hand=2,
            quantity_reserved=0,
            quantity_on_order=0,
            reorder_point=5,
            safety_stock=3,
        ))
        result = planner.generate_replenishment_plan({"BRG001": 5})
        assert result.provenance_hash is not None
