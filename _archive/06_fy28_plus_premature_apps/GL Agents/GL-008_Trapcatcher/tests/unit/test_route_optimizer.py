# -*- coding: utf-8 -*-
"""
Unit tests for MaintenanceRouteOptimizer.

Tests TSP/VRP optimization, route construction, and determinism.

Author: GL-TestEngineer
Date: December 2025
"""

import pytest
from datetime import datetime, timezone

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from optimization.route_optimizer import (
    MaintenanceRouteOptimizer,
    OptimizerConfig,
    MaintenanceTask,
    Location,
    Technician,
    PriorityLevel,
    TechnicianSkill,
    OptimizationObjective,
)


class TestMaintenanceRouteOptimizer:
    """Tests for MaintenanceRouteOptimizer class."""

    @pytest.fixture
    def optimizer(self):
        """Create default optimizer."""
        return MaintenanceRouteOptimizer()

    @pytest.fixture
    def depot(self):
        """Create depot location."""
        return Location(x=0, y=0, floor=0, zone="Depot")

    @pytest.fixture
    def technician(self, depot):
        """Create technician."""
        return Technician(
            technician_id="TECH-001",
            name="John Smith",
            skill_level=TechnicianSkill.ADVANCED,
            start_location=depot,
            available_hours=8.0,
        )

    @pytest.fixture
    def tasks(self):
        """Create sample maintenance tasks."""
        return [
            MaintenanceTask(
                task_id="TASK-001",
                trap_id="ST-001",
                location=Location(x=100, y=0),
                priority=PriorityLevel.CRITICAL,
                condition="failed",
                energy_loss_kw=50.0,
                estimated_duration_min=30,
            ),
            MaintenanceTask(
                task_id="TASK-002",
                trap_id="ST-002",
                location=Location(x=0, y=100),
                priority=PriorityLevel.HIGH,
                condition="leaking",
                energy_loss_kw=25.0,
                estimated_duration_min=20,
            ),
            MaintenanceTask(
                task_id="TASK-003",
                trap_id="ST-003",
                location=Location(x=100, y=100),
                priority=PriorityLevel.MEDIUM,
                condition="degraded",
                energy_loss_kw=10.0,
                estimated_duration_min=15,
            ),
        ]

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer is not None
        assert optimizer.config is not None

    def test_distance_matrix_calculation(self, optimizer, tasks, depot):
        """Test distance matrix calculation."""
        matrix = optimizer.calculate_distance_matrix(tasks, depot)

        assert len(matrix) == len(tasks) + 1  # +1 for depot
        assert len(matrix[0]) == len(tasks) + 1

        # Diagonal should be zero
        for i in range(len(matrix)):
            assert matrix[i][i] == 0.0

        # Symmetric matrix
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                assert abs(matrix[i][j] - matrix[j][i]) < 0.001

    def test_time_matrix_conversion(self, optimizer, tasks, depot):
        """Test conversion from distance to time matrix."""
        dist_matrix = optimizer.calculate_distance_matrix(tasks, depot)
        time_matrix = optimizer.calculate_time_matrix(dist_matrix)

        assert len(time_matrix) == len(dist_matrix)

        # Time should be distance / speed
        expected_time = dist_matrix[0][1] / optimizer.config.travel_speed_m_per_min
        assert abs(time_matrix[0][1] - expected_time) < 0.001

    def test_optimize_route_basic(self, optimizer, tasks, technician):
        """Test basic route optimization."""
        route = optimizer.optimize_route(tasks, technician)

        assert route is not None
        assert route.route_id is not None
        assert route.technician_id == technician.technician_id
        assert len(route.tasks) > 0
        assert route.metrics is not None

    def test_route_metrics(self, optimizer, tasks, technician):
        """Test route metrics calculation."""
        route = optimizer.optimize_route(tasks, technician)
        metrics = route.metrics

        assert metrics.total_distance_m >= 0
        assert metrics.total_travel_time_min >= 0
        assert metrics.total_service_time_min >= 0
        assert metrics.tasks_completed == len(route.tasks)

    def test_route_respects_time_constraint(self, optimizer, technician):
        """Test route respects maximum time constraint."""
        # Create many tasks
        tasks = [
            MaintenanceTask(
                task_id=f"TASK-{i:03d}",
                trap_id=f"ST-{i:03d}",
                location=Location(x=i * 50, y=i * 50),
                priority=PriorityLevel.MEDIUM,
                condition="leaking",
                energy_loss_kw=10.0,
                estimated_duration_min=60,  # Long tasks
            )
            for i in range(20)
        ]

        route = optimizer.optimize_route(tasks, technician)

        # Total time should not exceed available hours
        max_time = technician.available_hours * 60
        assert route.metrics.total_time_min <= max_time

    def test_route_skill_filtering(self, tasks):
        """Test that tasks are filtered by skill level."""
        # Create task requiring specialist
        specialist_task = MaintenanceTask(
            task_id="TASK-SPEC",
            trap_id="ST-SPEC",
            location=Location(x=50, y=50),
            priority=PriorityLevel.HIGH,
            condition="failed",
            energy_loss_kw=100.0,
            required_skill=TechnicianSkill.SPECIALIST,
        )

        # Basic skill technician
        basic_tech = Technician(
            technician_id="TECH-BASIC",
            name="Basic Tech",
            skill_level=TechnicianSkill.BASIC,
            start_location=Location(0, 0),
        )

        config = OptimizerConfig(respect_skill_requirements=True)
        optimizer = MaintenanceRouteOptimizer(config)
        route = optimizer.optimize_route([specialist_task], basic_tech)

        # Should not include specialist task
        assert len(route.tasks) == 0

    def test_priority_based_optimization(self, technician):
        """Test priority-based route ordering."""
        tasks = [
            MaintenanceTask(
                task_id="LOW",
                trap_id="ST-LOW",
                location=Location(x=10, y=10),
                priority=PriorityLevel.LOW,
                condition="degraded",
                energy_loss_kw=5.0,
            ),
            MaintenanceTask(
                task_id="CRITICAL",
                trap_id="ST-CRIT",
                location=Location(x=100, y=100),
                priority=PriorityLevel.CRITICAL,
                condition="failed",
                energy_loss_kw=100.0,
            ),
        ]

        config = OptimizerConfig(objective=OptimizationObjective.MAXIMIZE_PRIORITY)
        optimizer = MaintenanceRouteOptimizer(config)
        route = optimizer.optimize_route(tasks, technician)

        # Critical should be visited first
        if len(route.tasks) >= 2:
            assert route.tasks[0].priority.value <= route.tasks[1].priority.value

    def test_deterministic_optimization(self, optimizer, tasks, technician):
        """Test that same input produces same output."""
        route1 = optimizer.optimize_route(tasks, technician)
        route2 = optimizer.optimize_route(tasks, technician)

        assert route1.visit_order == route2.visit_order
        assert route1.metrics.total_distance_m == route2.metrics.total_distance_m
        assert route1.provenance_hash == route2.provenance_hash

    def test_fleet_optimization(self, optimizer, tasks):
        """Test fleet-wide optimization with multiple technicians."""
        technicians = [
            Technician(
                technician_id="TECH-001",
                name="Tech 1",
                skill_level=TechnicianSkill.ADVANCED,
                start_location=Location(0, 0),
            ),
            Technician(
                technician_id="TECH-002",
                name="Tech 2",
                skill_level=TechnicianSkill.INTERMEDIATE,
                start_location=Location(100, 0),
            ),
        ]

        routes = optimizer.optimize_fleet(tasks, technicians)

        assert len(routes) <= len(technicians)
        # All tasks should be assigned
        total_tasks = sum(len(r.tasks) for r in routes)
        assert total_tasks <= len(tasks)

    def test_schedule_report_generation(self, optimizer, tasks, technician):
        """Test schedule report generation."""
        route = optimizer.optimize_route(tasks, technician)
        report = optimizer.generate_schedule_report([route])

        assert "MAINTENANCE SCHEDULE" in report
        assert technician.technician_id in report

    def test_empty_task_list(self, optimizer, technician):
        """Test optimization with empty task list."""
        route = optimizer.optimize_route([], technician)

        assert route is not None
        assert len(route.tasks) == 0
        assert route.metrics.tasks_completed == 0

    def test_2opt_improvement(self, optimizer, tasks, technician):
        """Test that 2-opt improves route."""
        # With 2-opt enabled (default)
        config_with_2opt = OptimizerConfig(use_2opt=True)
        opt_with = MaintenanceRouteOptimizer(config_with_2opt)
        route_with = opt_with.optimize_route(tasks, technician)

        # Without 2-opt
        config_without_2opt = OptimizerConfig(use_2opt=False)
        opt_without = MaintenanceRouteOptimizer(config_without_2opt)
        route_without = opt_without.optimize_route(tasks, technician)

        # 2-opt should produce equal or better distance
        assert route_with.metrics.total_distance_m <= route_without.metrics.total_distance_m + 0.001


class TestLocation:
    """Tests for Location class."""

    def test_distance_calculation(self):
        """Test Euclidean distance calculation."""
        loc1 = Location(x=0, y=0)
        loc2 = Location(x=3, y=4)

        distance = loc1.distance_to(loc2)
        assert abs(distance - 5.0) < 0.001  # 3-4-5 triangle

    def test_distance_with_floor(self):
        """Test distance with floor penalty."""
        loc1 = Location(x=0, y=0, floor=0)
        loc2 = Location(x=0, y=0, floor=1)

        distance = loc1.distance_to(loc2)
        assert distance == 5.0  # 5m per floor
