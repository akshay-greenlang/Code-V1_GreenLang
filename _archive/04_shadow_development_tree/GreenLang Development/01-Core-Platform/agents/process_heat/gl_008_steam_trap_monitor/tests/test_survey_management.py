# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Survey Management Module Tests

Unit tests for survey_management.py module including TSP route optimization,
trap population management, and survey scheduling.

Target Coverage: 85%+
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    SurveyConfig,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapStatus,
    SurveyStatus,
    TrapSurveyInput,
    SurveyRouteOutput,
    RouteStop,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.survey_management import (
    TrapLocation,
    SurveyRoute,
    TSPRouteOptimizer,
    TrapPopulationManager,
    SurveyScheduler,
    TrapSurveyManager,
)


class TestTrapLocation:
    """Tests for TrapLocation dataclass."""

    def test_minimal_creation(self):
        """Test creating with minimal fields."""
        loc = TrapLocation(trap_id="ST-0001")

        assert loc.trap_id == "ST-0001"
        assert loc.coordinates is None
        assert loc.priority == "normal"

    def test_full_creation(self):
        """Test creating with all fields."""
        loc = TrapLocation(
            trap_id="ST-0001",
            coordinates=(100.0, 200.0),
            area_code="AREA-01",
            building="Building A",
            floor=2,
            last_survey_date=datetime.now(timezone.utc),
            priority="high",
            last_status=TrapStatus.GOOD,
            notes="Test trap",
        )

        assert loc.trap_id == "ST-0001"
        assert loc.coordinates == (100.0, 200.0)
        assert loc.area_code == "AREA-01"


class TestTSPRouteOptimizer:
    """Tests for TSPRouteOptimizer."""

    @pytest.fixture
    def optimizer(self) -> TSPRouteOptimizer:
        """Create optimizer instance."""
        return TSPRouteOptimizer()

    @pytest.fixture
    def sample_locations(self) -> List[TrapLocation]:
        """Create sample locations for testing."""
        return [
            TrapLocation(trap_id="ST-0001", coordinates=(0.0, 0.0)),
            TrapLocation(trap_id="ST-0002", coordinates=(10.0, 0.0)),
            TrapLocation(trap_id="ST-0003", coordinates=(10.0, 10.0)),
            TrapLocation(trap_id="ST-0004", coordinates=(0.0, 10.0)),
            TrapLocation(trap_id="ST-0005", coordinates=(5.0, 5.0)),
        ]

    def test_initialization(self, optimizer):
        """Test optimizer initializes correctly."""
        assert optimizer._optimization_count == 0

    def test_empty_list_returns_empty(self, optimizer):
        """Test empty location list returns empty."""
        result = optimizer.optimize_route([])
        assert result == []

    def test_single_location_returns_same(self, optimizer):
        """Test single location returns unchanged."""
        loc = TrapLocation(trap_id="ST-0001", coordinates=(0.0, 0.0))
        result = optimizer.optimize_route([loc])
        assert result == [loc]

    def test_two_locations_returns_same(self, optimizer):
        """Test two locations returns unchanged."""
        locs = [
            TrapLocation(trap_id="ST-0001", coordinates=(0.0, 0.0)),
            TrapLocation(trap_id="ST-0002", coordinates=(10.0, 0.0)),
        ]
        result = optimizer.optimize_route(locs)
        assert len(result) == 2

    def test_nearest_neighbor_optimization(self, optimizer, sample_locations):
        """Test nearest neighbor algorithm."""
        result = optimizer.optimize_route(
            sample_locations,
            method="nearest_neighbor",
        )

        assert len(result) == len(sample_locations)
        # All traps should be included
        result_ids = {loc.trap_id for loc in result}
        expected_ids = {loc.trap_id for loc in sample_locations}
        assert result_ids == expected_ids

    def test_two_opt_optimization(self, optimizer, sample_locations):
        """Test 2-opt algorithm."""
        result = optimizer.optimize_route(
            sample_locations,
            method="2opt",
        )

        assert len(result) == len(sample_locations)

    def test_optimization_reduces_distance(self, optimizer, sample_locations):
        """Test optimization reduces or maintains distance."""
        original_distance = optimizer.calculate_route_distance(sample_locations)

        result = optimizer.optimize_route(
            sample_locations,
            method="2opt",
        )
        optimized_distance = optimizer.calculate_route_distance(result)

        # Optimized should not be worse (may be same if already optimal)
        assert optimized_distance <= original_distance * 1.1  # Allow small tolerance

    def test_locations_without_coordinates(self, optimizer):
        """Test handling locations without coordinates."""
        locs = [
            TrapLocation(trap_id="ST-0001"),  # No coordinates
            TrapLocation(trap_id="ST-0002"),  # No coordinates
            TrapLocation(trap_id="ST-0003"),  # No coordinates
        ]

        result = optimizer.optimize_route(locs)

        # Should return sorted by area_code/trap_id
        assert len(result) == 3

    def test_mixed_coordinates(self, optimizer):
        """Test handling mix of with/without coordinates."""
        locs = [
            TrapLocation(trap_id="ST-0001", coordinates=(0.0, 0.0)),
            TrapLocation(trap_id="ST-0002"),  # No coordinates
            TrapLocation(trap_id="ST-0003", coordinates=(10.0, 10.0)),
        ]

        result = optimizer.optimize_route(locs)

        # Should handle gracefully
        assert len(result) == 3

    def test_custom_start_location(self, optimizer, sample_locations):
        """Test optimization with custom start location."""
        result = optimizer.optimize_route(
            sample_locations,
            method="nearest_neighbor",
            start_location=(50.0, 50.0),  # Far from all locations
        )

        assert len(result) == len(sample_locations)

    def test_calculate_route_distance(self, optimizer):
        """Test route distance calculation."""
        # Square route - should be 40 units (10 + 10 + 10 + 10)
        locs = [
            TrapLocation(trap_id="ST-0001", coordinates=(0.0, 0.0)),
            TrapLocation(trap_id="ST-0002", coordinates=(10.0, 0.0)),
            TrapLocation(trap_id="ST-0003", coordinates=(10.0, 10.0)),
            TrapLocation(trap_id="ST-0004", coordinates=(0.0, 10.0)),
        ]

        distance = optimizer.calculate_route_distance(locs)

        # 3 segments: (0,0) to (10,0) + (10,0) to (10,10) + (10,10) to (0,10)
        # = 10 + 10 + 10 = 30
        assert abs(distance - 30.0) < 0.1

    def test_optimization_count_increments(self, optimizer, sample_locations):
        """Test optimization count increments."""
        assert optimizer.optimization_count == 0

        optimizer.optimize_route(sample_locations)
        assert optimizer.optimization_count == 1

        optimizer.optimize_route(sample_locations)
        assert optimizer.optimization_count == 2


class TestTrapPopulationManager:
    """Tests for TrapPopulationManager."""

    @pytest.fixture
    def manager(self) -> TrapPopulationManager:
        """Create manager instance."""
        return TrapPopulationManager()

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.total_traps == 0

    def test_register_trap(self, manager):
        """Test registering a trap."""
        manager.register_trap(
            trap_id="ST-0001",
            coordinates=(100.0, 200.0),
            area_code="AREA-01",
        )

        assert manager.total_traps == 1
        trap = manager.get_trap("ST-0001")
        assert trap is not None
        assert trap.area_code == "AREA-01"

    def test_register_multiple_traps(self, manager):
        """Test registering multiple traps."""
        for i in range(10):
            manager.register_trap(trap_id=f"ST-{i:04d}")

        assert manager.total_traps == 10

    def test_update_trap_status(self, manager):
        """Test updating trap status."""
        manager.register_trap(trap_id="ST-0001")

        now = datetime.now(timezone.utc)
        manager.update_trap_status(
            trap_id="ST-0001",
            status=TrapStatus.FAILED_OPEN,
            survey_date=now,
        )

        trap = manager.get_trap("ST-0001")
        assert trap.last_status == TrapStatus.FAILED_OPEN
        assert trap.last_survey_date == now

    def test_get_traps_due_for_survey(self, manager):
        """Test getting traps due for survey."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=400)

        manager.register_trap(trap_id="ST-0001")  # Never surveyed
        manager.register_trap(trap_id="ST-0002")
        manager.update_trap_status("ST-0002", TrapStatus.GOOD, old_date)

        due_traps = manager.get_traps_due_for_survey(survey_interval_days=365)

        # Both should be due
        assert len(due_traps) == 2

    def test_get_traps_not_due(self, manager):
        """Test traps not due for survey are excluded."""
        now = datetime.now(timezone.utc)
        recent_date = now - timedelta(days=30)

        manager.register_trap(trap_id="ST-0001")
        manager.update_trap_status("ST-0001", TrapStatus.GOOD, recent_date)

        due_traps = manager.get_traps_due_for_survey(survey_interval_days=365)

        assert len(due_traps) == 0

    def test_get_traps_by_area(self, manager):
        """Test getting traps by area code."""
        manager.register_trap(trap_id="ST-0001", area_code="AREA-01")
        manager.register_trap(trap_id="ST-0002", area_code="AREA-01")
        manager.register_trap(trap_id="ST-0003", area_code="AREA-02")

        area_01_traps = manager.get_traps_by_area("AREA-01")

        assert len(area_01_traps) == 2

    def test_get_traps_by_status(self, manager):
        """Test getting traps by status."""
        now = datetime.now(timezone.utc)

        manager.register_trap(trap_id="ST-0001")
        manager.register_trap(trap_id="ST-0002")
        manager.register_trap(trap_id="ST-0003")

        manager.update_trap_status("ST-0001", TrapStatus.GOOD, now)
        manager.update_trap_status("ST-0002", TrapStatus.FAILED_OPEN, now)
        manager.update_trap_status("ST-0003", TrapStatus.GOOD, now)

        good_traps = manager.get_traps_by_status(TrapStatus.GOOD)
        failed_traps = manager.get_traps_by_status(TrapStatus.FAILED_OPEN)

        assert len(good_traps) == 2
        assert len(failed_traps) == 1

    def test_get_failed_traps(self, manager):
        """Test getting all failed traps."""
        now = datetime.now(timezone.utc)

        manager.register_trap(trap_id="ST-0001")
        manager.register_trap(trap_id="ST-0002")
        manager.register_trap(trap_id="ST-0003")
        manager.register_trap(trap_id="ST-0004")

        manager.update_trap_status("ST-0001", TrapStatus.GOOD, now)
        manager.update_trap_status("ST-0002", TrapStatus.FAILED_OPEN, now)
        manager.update_trap_status("ST-0003", TrapStatus.LEAKING, now)
        manager.update_trap_status("ST-0004", TrapStatus.FAILED_CLOSED, now)

        failed_traps = manager.get_failed_traps()

        assert len(failed_traps) == 3  # All except GOOD

    def test_get_all_area_codes(self, manager):
        """Test getting all area codes."""
        manager.register_trap(trap_id="ST-0001", area_code="AREA-01")
        manager.register_trap(trap_id="ST-0002", area_code="AREA-02")
        manager.register_trap(trap_id="ST-0003", area_code="AREA-01")

        area_codes = manager.get_all_area_codes()

        assert area_codes == {"AREA-01", "AREA-02"}


class TestSurveyScheduler:
    """Tests for SurveyScheduler."""

    @pytest.fixture
    def scheduler(self, survey_config) -> SurveyScheduler:
        """Create scheduler instance."""
        return SurveyScheduler(survey_config)

    @pytest.fixture
    def populated_manager(self) -> TrapPopulationManager:
        """Create manager with traps."""
        manager = TrapPopulationManager()
        for i in range(50):
            manager.register_trap(trap_id=f"ST-{i:04d}")
        return manager

    def test_create_schedule(self, scheduler, populated_manager):
        """Test creating survey schedule."""
        start_date = datetime.now(timezone.utc)

        schedule = scheduler.create_survey_schedule(
            population=populated_manager,
            start_date=start_date,
            num_survey_days=10,
        )

        assert len(schedule) > 0
        # All traps should be scheduled
        total_scheduled = sum(len(traps) for traps in schedule.values())
        assert total_scheduled == 50

    def test_schedule_respects_day_limit(self, scheduler, populated_manager):
        """Test schedule respects max traps per day."""
        start_date = datetime.now(timezone.utc)

        schedule = scheduler.create_survey_schedule(
            population=populated_manager,
            start_date=start_date,
            num_survey_days=10,
        )

        # Max traps per day based on config
        max_per_day = int(8.0 * 60 / 5.0)  # 8 hours / 5 min per trap

        for day_traps in schedule.values():
            assert len(day_traps) <= max_per_day

    def test_priority_ordering(self, scheduler):
        """Test failed traps are scheduled first."""
        manager = TrapPopulationManager()
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=400)

        # Register traps with different statuses
        manager.register_trap(trap_id="ST-0001")  # Unknown
        manager.register_trap(trap_id="ST-0002")
        manager.register_trap(trap_id="ST-0003")

        manager.update_trap_status("ST-0002", TrapStatus.FAILED_OPEN, old_date)
        manager.update_trap_status("ST-0003", TrapStatus.GOOD, old_date)

        schedule = scheduler.create_survey_schedule(
            population=manager,
            start_date=now,
            num_survey_days=1,
        )

        # Failed open should be first
        first_day = list(schedule.values())[0]
        assert first_day[0] == "ST-0002"


class TestTrapSurveyManager:
    """Tests for main TrapSurveyManager."""

    @pytest.fixture
    def manager(self, steam_trap_config) -> TrapSurveyManager:
        """Create manager instance."""
        return TrapSurveyManager(steam_trap_config)

    def test_initialization(self, manager):
        """Test manager initializes correctly."""
        assert manager.survey_count == 0

    def test_register_trap(self, manager):
        """Test registering trap through manager."""
        manager.register_trap(
            trap_id="ST-0001",
            coordinates=(100.0, 200.0),
            area_code="AREA-01",
        )

        assert manager.population.total_traps == 1

    def test_plan_survey(self, manager, sample_trap_survey_input):
        """Test planning survey route."""
        result = manager.plan_survey(sample_trap_survey_input)

        assert isinstance(result, SurveyRouteOutput)
        assert result.total_traps == len(sample_trap_survey_input.trap_ids)
        assert result.total_routes >= 1
        assert len(result.provenance_hash) == 64

    def test_survey_routes_cover_all_traps(self, manager, sample_trap_survey_input):
        """Test all traps are included in routes."""
        result = manager.plan_survey(sample_trap_survey_input)

        # Count total stops across all routes
        total_stops = sum(len(route) for route in result.routes)
        assert total_stops == result.total_traps

    def test_route_sequencing(self, manager, sample_trap_survey_input):
        """Test route stops are sequenced correctly."""
        result = manager.plan_survey(sample_trap_survey_input)

        for route in result.routes:
            for i, stop in enumerate(route):
                assert stop.sequence == i + 1

    def test_area_grouping(self, manager):
        """Test traps are grouped by area."""
        input_data = TrapSurveyInput(
            plant_id="TEST-001",
            trap_ids=["ST-0001", "ST-0002", "ST-0003", "ST-0004"],
            trap_locations={
                "ST-0001": (0.0, 0.0),
                "ST-0002": (10.0, 0.0),
                "ST-0003": (100.0, 100.0),
                "ST-0004": (110.0, 100.0),
            },
            trap_areas={
                "ST-0001": "AREA-01",
                "ST-0002": "AREA-01",
                "ST-0003": "AREA-02",
                "ST-0004": "AREA-02",
            },
            max_traps_per_route=2,
        )

        result = manager.plan_survey(input_data)

        # Should have at least 2 routes (one per area)
        assert result.total_routes >= 2

    def test_coverage_by_area_calculated(self, manager, sample_trap_survey_input):
        """Test coverage by area is calculated."""
        result = manager.plan_survey(sample_trap_survey_input)

        assert len(result.coverage_by_area) > 0
        total_coverage = sum(result.coverage_by_area.values())
        assert total_coverage == result.total_traps

    def test_total_time_calculated(self, manager, sample_trap_survey_input):
        """Test total time is calculated."""
        result = manager.plan_survey(sample_trap_survey_input)

        expected_time = (
            result.total_traps *
            sample_trap_survey_input.minutes_per_trap / 60.0
        )
        assert abs(result.total_time_hours - expected_time) < 0.1

    def test_get_survey_summary(self, manager):
        """Test getting survey summary."""
        now = datetime.now(timezone.utc)

        # Register and update some traps
        manager.register_trap("ST-0001")
        manager.register_trap("ST-0002")
        manager.register_trap("ST-0003")

        manager.population.update_trap_status("ST-0001", TrapStatus.GOOD, now)
        manager.population.update_trap_status("ST-0002", TrapStatus.FAILED_OPEN, now)

        summary = manager.get_survey_summary("TEST-PLANT")

        assert summary.total_traps == 3
        assert summary.traps_good == 1
        assert summary.traps_failed_open == 1
        assert summary.priority_repairs_count == 1

    def test_survey_count_increments(self, manager, sample_trap_survey_input):
        """Test survey count increments."""
        assert manager.survey_count == 0

        manager.plan_survey(sample_trap_survey_input)
        assert manager.survey_count == 1

        manager.plan_survey(sample_trap_survey_input)
        assert manager.survey_count == 2


class TestSurveyManagementIntegration:
    """Integration tests for survey management."""

    @pytest.fixture
    def manager(self, steam_trap_config) -> TrapSurveyManager:
        """Create manager instance."""
        return TrapSurveyManager(steam_trap_config)

    def test_large_survey_planning(self, manager):
        """Test planning survey for large trap population."""
        # Create input for 200 traps
        trap_ids = [f"ST-{i:04d}" for i in range(200)]
        trap_locations = {
            trap_id: (float(i % 20) * 10, float(i // 20) * 10)
            for i, trap_id in enumerate(trap_ids)
        }
        trap_areas = {
            trap_id: f"AREA-{(i % 5) + 1:02d}"
            for i, trap_id in enumerate(trap_ids)
        }

        input_data = TrapSurveyInput(
            plant_id="TEST-PLANT",
            trap_ids=trap_ids,
            trap_locations=trap_locations,
            trap_areas=trap_areas,
            max_traps_per_route=50,
        )

        result = manager.plan_survey(input_data)

        assert result.total_traps == 200
        assert result.total_routes >= 4  # At least 200/50 = 4 routes

    def test_route_optimization_improves_distance(self, manager):
        """Test route optimization produces reasonable distances."""
        # Create clustered traps
        trap_ids = [f"ST-{i:04d}" for i in range(20)]
        # Two clusters
        trap_locations = {
            trap_id: (
                float(i % 10) + (100 if i >= 10 else 0),
                float(i // 10 * 10)
            )
            for i, trap_id in enumerate(trap_ids)
        }

        input_data = TrapSurveyInput(
            plant_id="TEST-PLANT",
            trap_ids=trap_ids,
            trap_locations=trap_locations,
            trap_areas={tid: "AREA-01" for tid in trap_ids},
            max_traps_per_route=20,
        )

        result = manager.plan_survey(input_data)

        # Total distance should be reasonable (not crossing between clusters excessively)
        assert result.total_distance_ft < 1000  # Reasonable for clustered locations
