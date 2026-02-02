# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Survey Management Module

This module provides steam trap survey management including trap population
tracking, TSP (Traveling Salesman Problem) route optimization for efficient
survey execution, and survey scheduling.

Features:
    - Trap population management
    - TSP route optimization (nearest neighbor, 2-opt)
    - Survey scheduling and tracking
    - Area-based route grouping
    - Priority-based trap ordering

Standards:
    - DOE Steam System Best Practices (annual surveys)
    - Spirax Sarco Survey Program Guidelines

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.survey_management import (
    ...     TrapSurveyManager,
    ...     TSPRouteOptimizer,
    ... )
    >>> manager = TrapSurveyManager(config)
    >>> route = manager.plan_survey(trap_ids, survey_date)
    >>> print(f"Total distance: {route.total_distance_ft:.0f} ft")
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import hashlib
import heapq
import json
import logging
import math
import random

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    SteamTrapMonitorConfig,
    SurveyConfig,
)
from greenlang.agents.process_heat.gl_008_steam_trap_monitor.schemas import (
    TrapSurveyInput,
    SurveyRouteOutput,
    RouteStop,
    TrapStatusSummary,
    TrapStatus,
    SurveyStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class TrapLocation:
    """Steam trap location data."""

    trap_id: str
    coordinates: Optional[Tuple[float, float]] = None  # (lat, lon) or (x, y)
    area_code: Optional[str] = None
    building: Optional[str] = None
    floor: Optional[int] = None
    last_survey_date: Optional[datetime] = None
    priority: str = "normal"
    last_status: Optional[TrapStatus] = None
    notes: Optional[str] = None


@dataclass
class SurveyRoute:
    """Optimized survey route."""

    route_id: str
    stops: List[TrapLocation]
    total_distance: float
    estimated_time_hours: float
    area_codes: Set[str]


# =============================================================================
# TSP ROUTE OPTIMIZER
# =============================================================================

class TSPRouteOptimizer:
    """
    Traveling Salesman Problem optimizer for survey routes.

    Implements multiple algorithms for route optimization:
    - Nearest Neighbor: Fast, greedy heuristic
    - 2-opt: Local search improvement
    - Christofides: Near-optimal (not implemented, placeholder)

    All algorithms are deterministic with documented complexity.
    """

    def __init__(self) -> None:
        """Initialize the TSP optimizer."""
        self._optimization_count = 0

    def optimize_route(
        self,
        locations: List[TrapLocation],
        method: str = "nearest_neighbor",
        start_location: Optional[Tuple[float, float]] = None,
    ) -> List[TrapLocation]:
        """
        Optimize route through trap locations.

        Args:
            locations: List of trap locations
            method: Optimization method (nearest_neighbor, 2opt)
            start_location: Optional starting point

        Returns:
            Optimized list of TrapLocations in visit order
        """
        self._optimization_count += 1

        if len(locations) <= 2:
            return locations

        if method == "nearest_neighbor":
            route = self._nearest_neighbor(locations, start_location)
        elif method == "2opt":
            # Start with nearest neighbor, then improve with 2-opt
            route = self._nearest_neighbor(locations, start_location)
            route = self._two_opt(route)
        else:
            route = self._nearest_neighbor(locations, start_location)

        logger.debug(f"Route optimized: {len(route)} stops using {method}")

        return route

    def _nearest_neighbor(
        self,
        locations: List[TrapLocation],
        start: Optional[Tuple[float, float]] = None,
    ) -> List[TrapLocation]:
        """
        Nearest neighbor heuristic for TSP.

        Time complexity: O(n^2)
        Space complexity: O(n)

        Args:
            locations: List of locations
            start: Starting coordinates

        Returns:
            Ordered list of locations
        """
        if not locations:
            return []

        # Filter locations with valid coordinates
        valid_locations = [
            loc for loc in locations
            if loc.coordinates is not None
        ]

        if not valid_locations:
            # No coordinates - return by area code order
            return sorted(locations, key=lambda x: (x.area_code or "", x.trap_id))

        remaining = set(range(len(valid_locations)))
        route = []

        # Start from first location or specified start
        if start:
            # Find nearest to start
            current_coords = start
        else:
            # Start from first location
            current_idx = 0
            remaining.remove(0)
            route.append(valid_locations[0])
            current_coords = valid_locations[0].coordinates

        # Build route greedily
        while remaining:
            nearest_idx = None
            nearest_dist = float("inf")

            for idx in remaining:
                loc = valid_locations[idx]
                dist = self._distance(current_coords, loc.coordinates)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx

            if nearest_idx is not None:
                remaining.remove(nearest_idx)
                route.append(valid_locations[nearest_idx])
                current_coords = valid_locations[nearest_idx].coordinates

        # Add locations without coordinates at the end
        no_coord_locations = [
            loc for loc in locations
            if loc.coordinates is None
        ]
        route.extend(no_coord_locations)

        return route

    def _two_opt(self, route: List[TrapLocation]) -> List[TrapLocation]:
        """
        2-opt local search improvement.

        Iteratively reverses segments to improve tour length.
        Time complexity: O(n^2) per iteration, typically O(n^2 * k) total

        Args:
            route: Initial route

        Returns:
            Improved route
        """
        if len(route) < 4:
            return route

        # Extract coordinates
        coords = [
            loc.coordinates for loc in route
            if loc.coordinates is not None
        ]

        if len(coords) < 4:
            return route

        # Map indices
        coord_to_route_idx = {}
        route_indices = []
        for i, loc in enumerate(route):
            if loc.coordinates is not None:
                coord_to_route_idx[loc.coordinates] = i
                route_indices.append(i)

        n = len(route_indices)
        improved = True
        iterations = 0
        max_iterations = 100

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Calculate current distance
                    idx_i = route_indices[i]
                    idx_j = route_indices[j]
                    idx_i_prev = route_indices[i - 1]
                    idx_j_next = route_indices[(j + 1) % n] if j + 1 < n else route_indices[0]

                    loc_i = route[idx_i]
                    loc_j = route[idx_j]
                    loc_i_prev = route[idx_i_prev]
                    loc_j_next = route[idx_j_next]

                    if not all([
                        loc_i.coordinates,
                        loc_j.coordinates,
                        loc_i_prev.coordinates,
                        loc_j_next.coordinates,
                    ]):
                        continue

                    current_dist = (
                        self._distance(loc_i_prev.coordinates, loc_i.coordinates) +
                        self._distance(loc_j.coordinates, loc_j_next.coordinates)
                    )

                    new_dist = (
                        self._distance(loc_i_prev.coordinates, loc_j.coordinates) +
                        self._distance(loc_i.coordinates, loc_j_next.coordinates)
                    )

                    if new_dist < current_dist - 1e-10:
                        # Reverse segment between i and j
                        route_indices[i:j+1] = route_indices[i:j+1][::-1]
                        improved = True

        # Reconstruct route
        new_route = [route[idx] for idx in route_indices]

        # Add locations without coordinates
        no_coord = [loc for loc in route if loc.coordinates is None]
        new_route.extend(no_coord)

        return new_route

    def _distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float],
    ) -> float:
        """Calculate Euclidean distance between coordinates."""
        if coord1 is None or coord2 is None:
            return float("inf")

        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        return math.sqrt(dx * dx + dy * dy)

    def calculate_route_distance(
        self,
        route: List[TrapLocation],
    ) -> float:
        """Calculate total route distance."""
        if len(route) < 2:
            return 0.0

        total = 0.0
        for i in range(len(route) - 1):
            if route[i].coordinates and route[i + 1].coordinates:
                total += self._distance(
                    route[i].coordinates,
                    route[i + 1].coordinates,
                )

        return total

    @property
    def optimization_count(self) -> int:
        """Get optimization count."""
        return self._optimization_count


# =============================================================================
# TRAP POPULATION MANAGER
# =============================================================================

class TrapPopulationManager:
    """
    Manager for steam trap population data.

    Tracks all traps in the plant, their locations, status history,
    and survey schedule compliance.
    """

    def __init__(self) -> None:
        """Initialize the population manager."""
        self._traps: Dict[str, TrapLocation] = {}
        self._status_history: Dict[str, List[Tuple[datetime, TrapStatus]]] = {}

    def register_trap(
        self,
        trap_id: str,
        coordinates: Optional[Tuple[float, float]] = None,
        area_code: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Register a trap in the population.

        Args:
            trap_id: Unique trap identifier
            coordinates: GPS or grid coordinates
            area_code: Plant area code
            **kwargs: Additional trap attributes
        """
        location = TrapLocation(
            trap_id=trap_id,
            coordinates=coordinates,
            area_code=area_code,
            **kwargs,
        )
        self._traps[trap_id] = location
        logger.debug(f"Registered trap {trap_id}")

    def update_trap_status(
        self,
        trap_id: str,
        status: TrapStatus,
        survey_date: datetime,
    ) -> None:
        """Update trap status after survey."""
        if trap_id in self._traps:
            self._traps[trap_id].last_status = status
            self._traps[trap_id].last_survey_date = survey_date

            # Track history
            if trap_id not in self._status_history:
                self._status_history[trap_id] = []
            self._status_history[trap_id].append((survey_date, status))

    def get_traps_due_for_survey(
        self,
        survey_interval_days: int = 365,
        as_of_date: Optional[datetime] = None,
    ) -> List[TrapLocation]:
        """
        Get traps due for survey.

        Args:
            survey_interval_days: Maximum days between surveys
            as_of_date: Reference date (default: now)

        Returns:
            List of traps due for survey
        """
        if as_of_date is None:
            as_of_date = datetime.now(timezone.utc)

        due_traps = []
        cutoff = as_of_date - timedelta(days=survey_interval_days)

        for trap in self._traps.values():
            if trap.last_survey_date is None:
                due_traps.append(trap)
            elif trap.last_survey_date < cutoff:
                due_traps.append(trap)

        return due_traps

    def get_traps_by_area(
        self,
        area_code: str,
    ) -> List[TrapLocation]:
        """Get all traps in a specific area."""
        return [
            trap for trap in self._traps.values()
            if trap.area_code == area_code
        ]

    def get_traps_by_status(
        self,
        status: TrapStatus,
    ) -> List[TrapLocation]:
        """Get all traps with a specific status."""
        return [
            trap for trap in self._traps.values()
            if trap.last_status == status
        ]

    def get_failed_traps(self) -> List[TrapLocation]:
        """Get all failed traps."""
        failed_statuses = {
            TrapStatus.FAILED_OPEN,
            TrapStatus.FAILED_CLOSED,
            TrapStatus.LEAKING,
        }
        return [
            trap for trap in self._traps.values()
            if trap.last_status in failed_statuses
        ]

    def get_trap(self, trap_id: str) -> Optional[TrapLocation]:
        """Get a specific trap."""
        return self._traps.get(trap_id)

    def get_all_area_codes(self) -> Set[str]:
        """Get all unique area codes."""
        return {
            trap.area_code for trap in self._traps.values()
            if trap.area_code is not None
        }

    @property
    def total_traps(self) -> int:
        """Get total trap count."""
        return len(self._traps)


# =============================================================================
# SURVEY SCHEDULER
# =============================================================================

class SurveyScheduler:
    """
    Scheduler for steam trap surveys.

    Implements DOE-recommended annual survey scheduling with
    priority-based ordering.
    """

    def __init__(self, config: SurveyConfig) -> None:
        """
        Initialize scheduler.

        Args:
            config: Survey configuration
        """
        self.config = config

    def create_survey_schedule(
        self,
        population: TrapPopulationManager,
        start_date: datetime,
        num_survey_days: int = 10,
    ) -> Dict[date, List[str]]:
        """
        Create survey schedule for trap population.

        Args:
            population: Trap population manager
            start_date: Survey start date
            num_survey_days: Number of days for survey

        Returns:
            Dict mapping dates to trap IDs
        """
        schedule: Dict[date, List[str]] = {}

        # Get traps due for survey
        due_traps = population.get_traps_due_for_survey(
            survey_interval_days=self.config.survey_interval_months * 30
        )

        # Sort by priority (failed traps first, then by last survey date)
        def priority_key(trap: TrapLocation) -> Tuple[int, datetime]:
            priority_order = {
                TrapStatus.FAILED_OPEN: 0,
                TrapStatus.FAILED_CLOSED: 1,
                TrapStatus.LEAKING: 2,
            }
            status_priority = priority_order.get(trap.last_status, 10)
            last_survey = trap.last_survey_date or datetime.min.replace(tzinfo=timezone.utc)
            return (status_priority, last_survey)

        due_traps.sort(key=priority_key)

        # Calculate traps per day
        traps_per_day = int(
            self.config.max_hours_per_day * 60 /
            self.config.average_time_per_trap_minutes
        )

        # Assign to days
        current_date = start_date.date()
        day_count = 0
        trap_idx = 0

        while trap_idx < len(due_traps) and day_count < num_survey_days:
            day_traps = []

            for _ in range(traps_per_day):
                if trap_idx >= len(due_traps):
                    break
                day_traps.append(due_traps[trap_idx].trap_id)
                trap_idx += 1

            if day_traps:
                schedule[current_date] = day_traps

            current_date += timedelta(days=1)
            day_count += 1

        logger.info(
            f"Created survey schedule: {len(schedule)} days, "
            f"{sum(len(t) for t in schedule.values())} traps"
        )

        return schedule


# =============================================================================
# MAIN SURVEY MANAGER
# =============================================================================

class TrapSurveyManager:
    """
    Main survey management class.

    Integrates population management, route optimization, and
    survey scheduling for comprehensive trap survey programs.

    Example:
        >>> manager = TrapSurveyManager(config)
        >>> route = manager.plan_survey(
        ...     trap_ids=["ST-001", "ST-002", "ST-003"],
        ...     survey_date=datetime.now(),
        ... )
        >>> print(f"Route: {len(route.routes[0])} stops")
    """

    def __init__(self, config: SteamTrapMonitorConfig) -> None:
        """
        Initialize survey manager.

        Args:
            config: Agent configuration
        """
        self.config = config
        self._optimizer = TSPRouteOptimizer()
        self._population = TrapPopulationManager()
        self._scheduler = SurveyScheduler(config.survey)
        self._survey_count = 0

        logger.info("TrapSurveyManager initialized")

    def register_trap(
        self,
        trap_id: str,
        coordinates: Optional[Tuple[float, float]] = None,
        area_code: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Register a trap in the population."""
        self._population.register_trap(
            trap_id=trap_id,
            coordinates=coordinates,
            area_code=area_code,
            **kwargs,
        )

    def plan_survey(
        self,
        input_data: TrapSurveyInput,
    ) -> SurveyRouteOutput:
        """
        Plan optimized survey routes.

        Args:
            input_data: Survey planning input

        Returns:
            SurveyRouteOutput with optimized routes
        """
        self._survey_count += 1
        start_time = datetime.now(timezone.utc)

        # Build location list
        locations = []
        for trap_id in input_data.trap_ids:
            coords = input_data.trap_locations.get(trap_id)
            area = input_data.trap_areas.get(trap_id)

            # Check population first
            existing = self._population.get_trap(trap_id)
            if existing:
                loc = existing
                if coords:
                    loc.coordinates = coords
                if area:
                    loc.area_code = area
            else:
                loc = TrapLocation(
                    trap_id=trap_id,
                    coordinates=coords,
                    area_code=area,
                )
            locations.append(loc)

        # Group by area for multi-route planning
        areas = self._group_by_area(locations)

        # Create routes per area
        all_routes: List[List[RouteStop]] = []
        total_distance = 0.0

        for area_code, area_locations in areas.items():
            # Split into routes if too many traps
            chunks = self._split_into_routes(
                area_locations,
                input_data.max_traps_per_route,
            )

            for chunk in chunks:
                # Optimize route within chunk
                optimized = self._optimizer.optimize_route(
                    chunk,
                    method=self.config.survey.route_optimization_algorithm,
                )

                # Convert to RouteStops
                route_stops = []
                for i, loc in enumerate(optimized, 1):
                    stop = RouteStop(
                        sequence=i,
                        trap_id=loc.trap_id,
                        location=None,
                        area_code=loc.area_code,
                        gps_coordinates=loc.coordinates,
                        estimated_time_minutes=input_data.minutes_per_trap,
                        priority="high" if loc.last_status in [
                            TrapStatus.FAILED_OPEN,
                            TrapStatus.FAILED_CLOSED,
                        ] else "normal",
                    )
                    route_stops.append(stop)

                all_routes.append(route_stops)

                # Calculate distance
                route_dist = self._optimizer.calculate_route_distance(optimized)
                total_distance += route_dist

        # Calculate total time
        total_traps = len(input_data.trap_ids)
        total_time_minutes = total_traps * input_data.minutes_per_trap
        total_time_hours = total_time_minutes / 60.0

        # Coverage by area
        coverage = {}
        for loc in locations:
            area = loc.area_code or "unassigned"
            coverage[area] = coverage.get(area, 0) + 1

        # Provenance
        provenance_data = {
            "request_id": input_data.request_id,
            "total_traps": total_traps,
            "total_routes": len(all_routes),
            "timestamp": start_time.isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        logger.info(
            f"Survey planned: {len(all_routes)} routes, "
            f"{total_traps} traps, {total_time_hours:.1f} hours"
        )

        return SurveyRouteOutput(
            request_id=input_data.request_id,
            total_routes=len(all_routes),
            total_traps=total_traps,
            total_distance_ft=total_distance,
            total_time_hours=total_time_hours,
            routes=all_routes,
            optimization_method=self.config.survey.route_optimization_algorithm,
            coverage_by_area=coverage,
            provenance_hash=provenance_hash,
        )

    def _group_by_area(
        self,
        locations: List[TrapLocation],
    ) -> Dict[str, List[TrapLocation]]:
        """Group locations by area code."""
        areas: Dict[str, List[TrapLocation]] = {}

        for loc in locations:
            area = loc.area_code or "unassigned"
            if area not in areas:
                areas[area] = []
            areas[area].append(loc)

        return areas

    def _split_into_routes(
        self,
        locations: List[TrapLocation],
        max_per_route: int,
    ) -> List[List[TrapLocation]]:
        """Split locations into route-sized chunks."""
        if len(locations) <= max_per_route:
            return [locations]

        chunks = []
        for i in range(0, len(locations), max_per_route):
            chunks.append(locations[i:i + max_per_route])

        return chunks

    def get_survey_summary(
        self,
        plant_id: str,
    ) -> TrapStatusSummary:
        """
        Get summary of trap population status.

        Args:
            plant_id: Plant identifier

        Returns:
            TrapStatusSummary
        """
        # Count by status
        all_traps = list(self._population._traps.values())
        total = len(all_traps)

        status_counts = {
            TrapStatus.GOOD: 0,
            TrapStatus.FAILED_OPEN: 0,
            TrapStatus.FAILED_CLOSED: 0,
            TrapStatus.LEAKING: 0,
            TrapStatus.UNKNOWN: 0,
        }

        for trap in all_traps:
            status = trap.last_status or TrapStatus.UNKNOWN
            if status in status_counts:
                status_counts[status] += 1
            else:
                status_counts[TrapStatus.UNKNOWN] += 1

        # Calculate rates
        failed_count = (
            status_counts[TrapStatus.FAILED_OPEN] +
            status_counts[TrapStatus.FAILED_CLOSED] +
            status_counts[TrapStatus.LEAKING]
        )
        failure_rate = (failed_count / total * 100) if total > 0 else 0.0
        failed_open_rate = (
            status_counts[TrapStatus.FAILED_OPEN] / total * 100
        ) if total > 0 else 0.0

        # Survey status
        due_traps = self._population.get_traps_due_for_survey()
        surveyed = total - len(due_traps)

        if surveyed == 0:
            survey_status = SurveyStatus.NOT_STARTED
        elif surveyed >= total:
            survey_status = SurveyStatus.COMPLETED
        elif len(due_traps) > 0:
            survey_status = SurveyStatus.IN_PROGRESS
        else:
            survey_status = SurveyStatus.COMPLETED

        return TrapStatusSummary(
            plant_id=plant_id,
            total_traps=total,
            traps_good=status_counts[TrapStatus.GOOD],
            traps_failed_open=status_counts[TrapStatus.FAILED_OPEN],
            traps_failed_closed=status_counts[TrapStatus.FAILED_CLOSED],
            traps_leaking=status_counts[TrapStatus.LEAKING],
            traps_unknown=status_counts[TrapStatus.UNKNOWN],
            overall_failure_rate_pct=failure_rate,
            failed_open_rate_pct=failed_open_rate,
            survey_status=survey_status,
            traps_surveyed_this_cycle=surveyed,
            priority_repairs_count=failed_count,
        )

    @property
    def population(self) -> TrapPopulationManager:
        """Get population manager."""
        return self._population

    @property
    def survey_count(self) -> int:
        """Get survey count."""
        return self._survey_count
