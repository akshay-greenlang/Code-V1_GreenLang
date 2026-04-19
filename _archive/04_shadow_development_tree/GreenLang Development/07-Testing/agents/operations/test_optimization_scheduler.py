# -*- coding: utf-8 -*-
"""
Tests for GL-OPS-X-003: Optimization Scheduler

Tests cover:
    - Task scheduling and management
    - Carbon-optimal scheduling
    - Cost optimization
    - Grid carbon forecasting
    - Schedule conflict resolution
    - Provenance tracking

Author: GreenLang Team
"""

import pytest
from datetime import datetime, timedelta

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.operations.optimization_scheduler import (
    OptimizationScheduler,
    SchedulerInput,
    SchedulerOutput,
    OperationTask,
    ScheduleEntry,
    GridCarbonForecast,
    OptimizationGoal,
    SchedulePeriod,
    ResourceType,
)
from greenlang.utilities.determinism import DeterministicClock


class TestOptimizationSchedulerInitialization:
    """Tests for agent initialization."""

    def test_agent_creation_default_config(self):
        """Test creating agent with default configuration."""
        agent = OptimizationScheduler()

        assert agent.AGENT_ID == "GL-OPS-X-003"
        assert agent.AGENT_NAME == "Optimization Scheduler"
        assert agent.VERSION == "1.0.0"

    def test_agent_creation_custom_config(self):
        """Test creating agent with custom configuration."""
        config = AgentConfig(
            name="Custom Scheduler",
            description="Custom config test",
        )
        agent = OptimizationScheduler(config)

        assert agent.config.name == "Custom Scheduler"


class TestTaskManagement:
    """Tests for task management functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OptimizationScheduler()

    def test_add_task(self, agent):
        """Test adding a task to schedule."""
        result = agent.run({
            "operation": "add_task",
            "task": {
                "name": "Batch Processing",
                "resource_type": "compute",
                "duration_hours": 2.0,
                "energy_kwh": 500.0,
                "priority": 1,
                "flexible": True,
            }
        })

        assert result.success
        assert result.data["data"].get("added") or "task_id" in result.data["data"]

    def test_remove_task(self, agent):
        """Test removing a task from schedule."""
        # Add task first
        add_result = agent.run({
            "operation": "add_task",
            "task": {
                "name": "To Remove",
                "resource_type": "compute",
                "duration_hours": 1.0,
            }
        })

        task_id = add_result.data["data"].get("task_id")
        if task_id:
            result = agent.run({
                "operation": "remove_task",
                "task_id": task_id,
            })
            assert result.success

    def test_list_tasks(self, agent):
        """Test listing all tasks."""
        # Add a few tasks
        for i in range(3):
            agent.run({
                "operation": "add_task",
                "task": {
                    "name": f"Task {i}",
                    "resource_type": "compute",
                    "duration_hours": 1.0,
                }
            })

        result = agent.run({"operation": "list_tasks"})

        assert result.success
        assert "tasks" in result.data["data"] or "count" in result.data["data"]


class TestOptimization:
    """Tests for schedule optimization."""

    @pytest.fixture
    def agent_with_tasks(self):
        """Create agent with pre-configured tasks."""
        agent = OptimizationScheduler()

        # Add tasks with different priorities and flexibility
        tasks = [
            {"name": "High Priority", "resource_type": "compute", "duration_hours": 2.0, "priority": 1, "flexible": False},
            {"name": "Flexible Load", "resource_type": "compute", "duration_hours": 4.0, "priority": 2, "flexible": True},
            {"name": "Low Priority", "resource_type": "compute", "duration_hours": 3.0, "priority": 3, "flexible": True},
        ]

        for task in tasks:
            agent.run({"operation": "add_task", "task": task})

        return agent

    def test_optimize_for_carbon(self, agent_with_tasks):
        """Test carbon-optimal scheduling."""
        result = agent_with_tasks.run({
            "operation": "optimize",
            "optimization_goal": "carbon",
            "schedule_period": "daily",
        })

        assert result.success
        assert "schedule" in result.data["data"] or "optimized" in result.data["data"]

    def test_optimize_for_cost(self, agent_with_tasks):
        """Test cost-optimal scheduling."""
        result = agent_with_tasks.run({
            "operation": "optimize",
            "optimization_goal": "cost",
            "schedule_period": "daily",
        })

        assert result.success

    def test_optimize_balanced(self, agent_with_tasks):
        """Test balanced optimization (carbon + cost)."""
        result = agent_with_tasks.run({
            "operation": "optimize",
            "optimization_goal": "balanced",
            "schedule_period": "weekly",
        })

        assert result.success


class TestGridCarbonForecasting:
    """Tests for grid carbon forecasting."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OptimizationScheduler()

    def test_set_carbon_forecast(self, agent):
        """Test setting grid carbon forecast."""
        result = agent.run({
            "operation": "set_carbon_forecast",
            "forecast": {
                "region": "US-EAST",
                "forecasts": [
                    {"hour": 0, "carbon_intensity": 400.0},
                    {"hour": 6, "carbon_intensity": 350.0},
                    {"hour": 12, "carbon_intensity": 450.0},
                    {"hour": 18, "carbon_intensity": 380.0},
                ]
            }
        })

        assert result.success

    def test_get_carbon_forecast(self, agent):
        """Test getting grid carbon forecast."""
        # Set forecast first
        agent.run({
            "operation": "set_carbon_forecast",
            "forecast": {
                "region": "US-WEST",
                "forecasts": [
                    {"hour": 0, "carbon_intensity": 300.0},
                ]
            }
        })

        result = agent.run({
            "operation": "get_carbon_forecast",
            "region": "US-WEST",
        })

        assert result.success


class TestScheduleManagement:
    """Tests for schedule management."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OptimizationScheduler()

    def test_get_schedule(self, agent):
        """Test getting current schedule."""
        result = agent.run({
            "operation": "get_schedule",
            "schedule_period": "daily",
        })

        assert result.success

    def test_clear_schedule(self, agent):
        """Test clearing schedule."""
        # Add some tasks first
        agent.run({
            "operation": "add_task",
            "task": {
                "name": "Test Task",
                "resource_type": "compute",
                "duration_hours": 1.0,
            }
        })

        result = agent.run({"operation": "clear_schedule"})

        assert result.success


class TestProvenanceTracking:
    """Tests for provenance and audit trail."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OptimizationScheduler()

    def test_output_contains_provenance_hash(self, agent):
        """Test that all operations include provenance hash."""
        result = agent.run({
            "operation": "add_task",
            "task": {
                "name": "Provenance Test",
                "resource_type": "compute",
                "duration_hours": 1.0,
            }
        })

        assert result.success
        assert "provenance_hash" in result.data
        assert len(result.data["provenance_hash"]) == 16


class TestStatistics:
    """Tests for statistics functionality."""

    @pytest.fixture
    def agent(self):
        """Create a fresh agent for each test."""
        return OptimizationScheduler()

    def test_get_statistics(self, agent):
        """Test getting scheduler statistics."""
        result = agent.run({"operation": "get_statistics"})

        assert result.success
        assert "data" in result.data


class TestIntegration:
    """Integration tests."""

    def test_full_scheduling_workflow(self):
        """Test a complete scheduling workflow."""
        agent = OptimizationScheduler()

        # 1. Set carbon forecast
        agent.run({
            "operation": "set_carbon_forecast",
            "forecast": {
                "region": "DEFAULT",
                "forecasts": [
                    {"hour": i, "carbon_intensity": 300.0 + (i % 12) * 20}
                    for i in range(24)
                ]
            }
        })

        # 2. Add tasks
        tasks = [
            {"name": "Data Processing", "resource_type": "compute", "duration_hours": 3.0, "flexible": True},
            {"name": "Backup", "resource_type": "storage", "duration_hours": 2.0, "flexible": True},
            {"name": "Report Generation", "resource_type": "compute", "duration_hours": 1.0, "flexible": False, "priority": 1},
        ]

        for task in tasks:
            result = agent.run({"operation": "add_task", "task": task})
            assert result.success

        # 3. Optimize schedule
        optimize_result = agent.run({
            "operation": "optimize",
            "optimization_goal": "carbon",
            "schedule_period": "daily",
        })
        assert optimize_result.success

        # 4. Get schedule
        schedule_result = agent.run({
            "operation": "get_schedule",
            "schedule_period": "daily",
        })
        assert schedule_result.success

        # 5. Get statistics
        stats_result = agent.run({"operation": "get_statistics"})
        assert stats_result.success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
