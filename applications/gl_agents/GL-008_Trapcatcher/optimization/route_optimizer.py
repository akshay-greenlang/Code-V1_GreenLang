# -*- coding: utf-8 -*-
"""
Maintenance Route Optimizer for GL-008 TRAPCATCHER

TSP/VRP-based route optimization for steam trap maintenance scheduling.
Uses deterministic algorithms for reproducible, optimal route planning.

Zero-Hallucination Guarantee:
- Deterministic nearest-neighbor and 2-opt optimization
- Reproducible results with fixed random seeds
- No AI inference in any optimization path

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# ENUMERATIONS
# ============================================================================

class PriorityLevel(Enum):
    """Maintenance task priority levels."""
    CRITICAL = 1     # Failed trap, immediate action required
    HIGH = 2         # Leaking trap, action within 24 hours
    MEDIUM = 3       # Degraded performance, action within week
    LOW = 4          # Preventive maintenance
    SCHEDULED = 5    # Routine inspection


class OptimizationObjective(Enum):
    """Route optimization objectives."""
    MINIMIZE_DISTANCE = "minimize_distance"
    MINIMIZE_TIME = "minimize_time"
    MAXIMIZE_PRIORITY = "maximize_priority"
    MINIMIZE_ENERGY_LOSS = "minimize_energy_loss"
    BALANCED = "balanced"


class TechnicianSkill(Enum):
    """Technician skill levels."""
    BASIC = "basic"           # Can inspect and replace
    INTERMEDIATE = "intermediate"  # Can repair most types
    ADVANCED = "advanced"      # Can handle all trap types
    SPECIALIST = "specialist"  # Specialized for complex systems


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Location:
    """Geographic location for routing."""
    x: float  # X coordinate (meters from origin) or longitude
    y: float  # Y coordinate (meters from origin) or latitude
    floor: int = 0  # Floor level (for multi-story facilities)
    zone: str = ""  # Zone identifier
    building: str = ""  # Building identifier

    def distance_to(self, other: "Location") -> float:
        """Calculate Euclidean distance to another location."""
        dx = self.x - other.x
        dy = self.y - other.y
        # Add floor penalty (assume 5m per floor)
        dz = abs(self.floor - other.floor) * 5.0
        return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class MaintenanceTask:
    """Maintenance task for a steam trap."""
    task_id: str
    trap_id: str
    location: Location
    priority: PriorityLevel
    condition: str
    energy_loss_kw: float
    estimated_duration_min: float = 30.0
    required_skill: TechnicianSkill = TechnicianSkill.BASIC
    due_date: Optional[datetime] = None
    notes: str = ""
    parts_required: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.task_id)


@dataclass
class Technician:
    """Technician resource for maintenance."""
    technician_id: str
    name: str
    skill_level: TechnicianSkill
    start_location: Location
    available_hours: float = 8.0
    hourly_rate_usd: float = 75.0


@dataclass
class OptimizerConfig:
    """Configuration for route optimizer."""

    # Optimization parameters
    objective: OptimizationObjective = OptimizationObjective.BALANCED

    # Time constraints
    max_route_hours: float = 8.0
    travel_speed_m_per_min: float = 50.0  # ~3 km/h walking speed

    # Priority weights (for balanced objective)
    weight_distance: float = 0.3
    weight_priority: float = 0.4
    weight_energy_loss: float = 0.3

    # Optimization settings
    use_2opt: bool = True
    max_2opt_iterations: int = 1000

    # Constraints
    respect_due_dates: bool = True
    respect_skill_requirements: bool = True

    # Depot settings (start/end location)
    depot_location: Optional[Location] = None
    return_to_depot: bool = True


@dataclass
class RouteMetrics:
    """Metrics for an optimized route."""
    total_distance_m: float
    total_travel_time_min: float
    total_service_time_min: float
    total_time_min: float
    tasks_completed: int
    priority_score: float
    energy_loss_addressed_kw: float
    estimated_cost_usd: float


@dataclass
class OptimizedRoute:
    """Result of route optimization."""
    route_id: str
    technician_id: str
    tasks: List[MaintenanceTask]
    visit_order: List[int]
    metrics: RouteMetrics
    start_time: datetime
    end_time: datetime
    provenance_hash: str


# ============================================================================
# MAIN OPTIMIZER CLASS
# ============================================================================

class MaintenanceRouteOptimizer:
    """
    TSP/VRP-based route optimizer for steam trap maintenance.

    Uses deterministic algorithms (nearest-neighbor + 2-opt improvement)
    for reproducible, near-optimal route planning.

    Zero-Hallucination Guarantee:
    - All optimization uses deterministic algorithms
    - Same input always produces identical output
    - No AI/ML inference in any path

    Example:
        >>> optimizer = MaintenanceRouteOptimizer()
        >>> tasks = [MaintenanceTask(...), ...]
        >>> route = optimizer.optimize_route(tasks, technician)
        >>> print(f"Route distance: {route.metrics.total_distance_m} m")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[OptimizerConfig] = None):
        """
        Initialize route optimizer.

        Args:
            config: Optimizer configuration (optional)
        """
        self.config = config or OptimizerConfig()

    def calculate_distance_matrix(
        self,
        tasks: List[MaintenanceTask],
        depot: Optional[Location] = None
    ) -> List[List[float]]:
        """
        Calculate distance matrix between all tasks and depot.

        Args:
            tasks: List of maintenance tasks
            depot: Starting depot location

        Returns:
            NxN distance matrix (N = len(tasks) + 1 if depot)
        """
        locations = []

        # Add depot as first location if provided
        if depot:
            locations.append(depot)

        # Add task locations
        for task in tasks:
            locations.append(task.location)

        n = len(locations)
        matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = locations[i].distance_to(locations[j])
                matrix[i][j] = dist
                matrix[j][i] = dist

        return matrix

    def calculate_time_matrix(
        self,
        distance_matrix: List[List[float]]
    ) -> List[List[float]]:
        """
        Convert distance matrix to travel time matrix.

        Args:
            distance_matrix: Distance matrix in meters

        Returns:
            Time matrix in minutes
        """
        n = len(distance_matrix)
        time_matrix = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                time_matrix[i][j] = distance_matrix[i][j] / self.config.travel_speed_m_per_min

        return time_matrix

    def _nearest_neighbor(
        self,
        tasks: List[MaintenanceTask],
        distance_matrix: List[List[float]],
        has_depot: bool
    ) -> List[int]:
        """
        Construct initial route using nearest neighbor heuristic.

        Args:
            tasks: List of tasks
            distance_matrix: Distance matrix
            has_depot: Whether depot is included

        Returns:
            Visit order as list of indices
        """
        n = len(tasks)
        if n == 0:
            return []

        # Start from depot (index 0) or first task
        offset = 1 if has_depot else 0
        visited = [False] * n
        route = []

        # Current position (start at depot or first task)
        current = 0 if has_depot else 0

        for _ in range(n):
            # Find nearest unvisited task
            best_dist = float('inf')
            best_idx = -1

            for j in range(n):
                if not visited[j]:
                    task_idx = j + offset
                    dist = distance_matrix[current][task_idx]

                    # Apply priority weighting if balanced objective
                    if self.config.objective == OptimizationObjective.BALANCED:
                        priority_factor = 1.0 / tasks[j].priority.value
                        energy_factor = tasks[j].energy_loss_kw / 100.0
                        weighted_dist = (
                            dist * self.config.weight_distance -
                            priority_factor * self.config.weight_priority * 1000 -
                            energy_factor * self.config.weight_energy_loss * 1000
                        )
                    elif self.config.objective == OptimizationObjective.MAXIMIZE_PRIORITY:
                        weighted_dist = -1.0 / tasks[j].priority.value * 1000 + dist * 0.001
                    elif self.config.objective == OptimizationObjective.MINIMIZE_ENERGY_LOSS:
                        weighted_dist = -tasks[j].energy_loss_kw + dist * 0.001
                    else:
                        weighted_dist = dist

                    if weighted_dist < best_dist:
                        best_dist = weighted_dist
                        best_idx = j

            if best_idx >= 0:
                visited[best_idx] = True
                route.append(best_idx)
                current = best_idx + offset

        return route

    def _calculate_route_distance(
        self,
        route: List[int],
        distance_matrix: List[List[float]],
        has_depot: bool
    ) -> float:
        """Calculate total distance for a route."""
        if len(route) < 2:
            return 0.0

        offset = 1 if has_depot else 0
        total = 0.0

        # From depot to first task
        if has_depot:
            total += distance_matrix[0][route[0] + offset]

        # Between tasks
        for i in range(len(route) - 1):
            total += distance_matrix[route[i] + offset][route[i + 1] + offset]

        # Return to depot
        if has_depot and self.config.return_to_depot:
            total += distance_matrix[route[-1] + offset][0]

        return total

    def _2opt_improve(
        self,
        route: List[int],
        distance_matrix: List[List[float]],
        has_depot: bool
    ) -> List[int]:
        """
        Improve route using 2-opt local search.

        Args:
            route: Initial route
            distance_matrix: Distance matrix
            has_depot: Whether depot is included

        Returns:
            Improved route
        """
        if len(route) < 4:
            return route

        best_route = route[:]
        best_distance = self._calculate_route_distance(best_route, distance_matrix, has_depot)
        improved = True
        iterations = 0

        while improved and iterations < self.config.max_2opt_iterations:
            improved = False
            iterations += 1

            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    # Create new route by reversing segment between i and j
                    new_route = (
                        best_route[:i + 1] +
                        best_route[i + 1:j + 1][::-1] +
                        best_route[j + 1:]
                    )

                    new_distance = self._calculate_route_distance(
                        new_route, distance_matrix, has_depot
                    )

                    if new_distance < best_distance - 1e-6:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True

        return best_route

    def _filter_by_skill(
        self,
        tasks: List[MaintenanceTask],
        technician: Technician
    ) -> List[MaintenanceTask]:
        """Filter tasks based on technician skill level."""
        if not self.config.respect_skill_requirements:
            return tasks

        skill_levels = {
            TechnicianSkill.BASIC: 1,
            TechnicianSkill.INTERMEDIATE: 2,
            TechnicianSkill.ADVANCED: 3,
            TechnicianSkill.SPECIALIST: 4,
        }

        tech_level = skill_levels[technician.skill_level]

        return [
            task for task in tasks
            if skill_levels[task.required_skill] <= tech_level
        ]

    def _compute_provenance_hash(
        self,
        tasks: List[MaintenanceTask],
        route: List[int],
        metrics: RouteMetrics
    ) -> str:
        """Compute SHA-256 hash for route provenance."""
        data = {
            "version": self.VERSION,
            "objective": self.config.objective.value,
            "task_ids": [tasks[i].task_id for i in route],
            "total_distance": round(metrics.total_distance_m, 2),
            "total_time": round(metrics.total_time_min, 2),
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def optimize_route(
        self,
        tasks: List[MaintenanceTask],
        technician: Technician,
        start_time: Optional[datetime] = None
    ) -> OptimizedRoute:
        """
        Optimize maintenance route for a technician.

        Args:
            tasks: List of maintenance tasks
            technician: Technician to assign
            start_time: Route start time (default: now)

        Returns:
            Optimized route with metrics
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        # Filter tasks by skill
        eligible_tasks = self._filter_by_skill(tasks, technician)

        if not eligible_tasks:
            # Return empty route
            return OptimizedRoute(
                route_id=f"ROUTE-{start_time.strftime('%Y%m%d%H%M%S')}",
                technician_id=technician.technician_id,
                tasks=[],
                visit_order=[],
                metrics=RouteMetrics(
                    total_distance_m=0,
                    total_travel_time_min=0,
                    total_service_time_min=0,
                    total_time_min=0,
                    tasks_completed=0,
                    priority_score=0,
                    energy_loss_addressed_kw=0,
                    estimated_cost_usd=0,
                ),
                start_time=start_time,
                end_time=start_time,
                provenance_hash="empty",
            )

        # Use depot from config or technician start
        depot = self.config.depot_location or technician.start_location
        has_depot = depot is not None

        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(eligible_tasks, depot)
        time_matrix = self.calculate_time_matrix(distance_matrix)

        # Get initial route using nearest neighbor
        route = self._nearest_neighbor(eligible_tasks, distance_matrix, has_depot)

        # Improve with 2-opt if enabled
        if self.config.use_2opt and len(route) >= 4:
            route = self._2opt_improve(route, distance_matrix, has_depot)

        # Apply time constraint - trim route if needed
        offset = 1 if has_depot else 0
        max_time = self.config.max_route_hours * 60
        final_route = []
        cumulative_time = 0.0
        current_pos = 0  # Start at depot

        for task_idx in route:
            # Calculate time to this task
            travel_time = time_matrix[current_pos][task_idx + offset]
            service_time = eligible_tasks[task_idx].estimated_duration_min

            # Check if we can fit this task
            new_time = cumulative_time + travel_time + service_time

            # Account for return to depot
            if has_depot and self.config.return_to_depot:
                return_time = time_matrix[task_idx + offset][0]
                if new_time + return_time > max_time:
                    break

            if new_time > max_time:
                break

            final_route.append(task_idx)
            cumulative_time = new_time
            current_pos = task_idx + offset

        # Calculate final metrics
        total_distance = self._calculate_route_distance(final_route, distance_matrix, has_depot)
        total_travel_time = total_distance / self.config.travel_speed_m_per_min
        total_service_time = sum(eligible_tasks[i].estimated_duration_min for i in final_route)
        total_time = total_travel_time + total_service_time

        # Priority score (sum of inverse priorities - higher is better)
        priority_score = sum(1.0 / eligible_tasks[i].priority.value for i in final_route)

        # Energy loss addressed
        energy_loss = sum(eligible_tasks[i].energy_loss_kw for i in final_route)

        # Estimated cost
        estimated_cost = (total_time / 60.0) * technician.hourly_rate_usd

        metrics = RouteMetrics(
            total_distance_m=total_distance,
            total_travel_time_min=total_travel_time,
            total_service_time_min=total_service_time,
            total_time_min=total_time,
            tasks_completed=len(final_route),
            priority_score=priority_score,
            energy_loss_addressed_kw=energy_loss,
            estimated_cost_usd=estimated_cost,
        )

        # Get ordered tasks
        ordered_tasks = [eligible_tasks[i] for i in final_route]

        # Compute provenance
        provenance_hash = self._compute_provenance_hash(eligible_tasks, final_route, metrics)

        # Calculate end time
        end_time = start_time + timedelta(minutes=total_time)

        return OptimizedRoute(
            route_id=f"ROUTE-{start_time.strftime('%Y%m%d%H%M%S')}-{provenance_hash[:8]}",
            technician_id=technician.technician_id,
            tasks=ordered_tasks,
            visit_order=final_route,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            provenance_hash=provenance_hash,
        )

    def optimize_fleet(
        self,
        tasks: List[MaintenanceTask],
        technicians: List[Technician],
        start_time: Optional[datetime] = None
    ) -> List[OptimizedRoute]:
        """
        Optimize routes for multiple technicians (basic VRP).

        Uses greedy assignment: assigns tasks to technicians based on
        priority and proximity, then optimizes each route.

        Args:
            tasks: All maintenance tasks
            technicians: Available technicians
            start_time: Start time for all routes

        Returns:
            List of optimized routes, one per technician
        """
        if start_time is None:
            start_time = datetime.now(timezone.utc)

        if not technicians:
            return []

        # Sort tasks by priority (highest first)
        sorted_tasks = sorted(tasks, key=lambda t: (t.priority.value, -t.energy_loss_kw))

        # Assign tasks to technicians greedily
        technician_tasks: Dict[str, List[MaintenanceTask]] = {
            t.technician_id: [] for t in technicians
        }
        technician_time: Dict[str, float] = {t.technician_id: 0.0 for t in technicians}

        tech_map = {t.technician_id: t for t in technicians}

        for task in sorted_tasks:
            # Find best technician for this task
            best_tech = None
            best_score = float('inf')

            for tech in technicians:
                # Check skill
                if self.config.respect_skill_requirements:
                    skill_levels = {
                        TechnicianSkill.BASIC: 1,
                        TechnicianSkill.INTERMEDIATE: 2,
                        TechnicianSkill.ADVANCED: 3,
                        TechnicianSkill.SPECIALIST: 4,
                    }
                    if skill_levels[task.required_skill] > skill_levels[tech.skill_level]:
                        continue

                # Check time capacity
                current_time = technician_time[tech.technician_id]
                if current_time + task.estimated_duration_min > tech.available_hours * 60:
                    continue

                # Calculate score (lower is better)
                distance = tech.start_location.distance_to(task.location)
                score = distance + current_time * 0.1  # Slight penalty for busy technicians

                if score < best_score:
                    best_score = score
                    best_tech = tech

            if best_tech:
                technician_tasks[best_tech.technician_id].append(task)
                technician_time[best_tech.technician_id] += task.estimated_duration_min

        # Optimize route for each technician
        routes = []
        for tech in technicians:
            assigned_tasks = technician_tasks[tech.technician_id]
            if assigned_tasks:
                route = self.optimize_route(assigned_tasks, tech, start_time)
                routes.append(route)

        return routes

    def generate_schedule_report(self, routes: List[OptimizedRoute]) -> str:
        """
        Generate human-readable schedule report.

        Args:
            routes: List of optimized routes

        Returns:
            Formatted schedule text
        """
        if not routes:
            return "No routes generated."

        lines = [
            "=" * 80,
            "           STEAM TRAP MAINTENANCE SCHEDULE",
            "=" * 80,
            "",
        ]

        total_tasks = 0
        total_distance = 0.0
        total_energy = 0.0

        for route in routes:
            lines.append(f"TECHNICIAN: {route.technician_id}")
            lines.append(f"Route ID: {route.route_id}")
            lines.append(f"Start: {route.start_time.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"End: {route.end_time.strftime('%Y-%m-%d %H:%M')}")
            lines.append("-" * 40)

            for i, task in enumerate(route.tasks):
                lines.append(
                    f"  {i+1}. {task.trap_id} | {task.condition} | "
                    f"P{task.priority.value} | {task.energy_loss_kw:.1f} kW | "
                    f"{task.estimated_duration_min:.0f} min"
                )

            lines.append("-" * 40)
            lines.append(f"Tasks: {route.metrics.tasks_completed}")
            lines.append(f"Distance: {route.metrics.total_distance_m:.0f} m")
            lines.append(f"Time: {route.metrics.total_time_min:.0f} min")
            lines.append(f"Energy addressed: {route.metrics.energy_loss_addressed_kw:.1f} kW")
            lines.append(f"Cost: ${route.metrics.estimated_cost_usd:.2f}")
            lines.append("")

            total_tasks += route.metrics.tasks_completed
            total_distance += route.metrics.total_distance_m
            total_energy += route.metrics.energy_loss_addressed_kw

        lines.append("=" * 80)
        lines.append("SUMMARY")
        lines.append(f"Total technicians: {len(routes)}")
        lines.append(f"Total tasks: {total_tasks}")
        lines.append(f"Total distance: {total_distance:.0f} m")
        lines.append(f"Total energy addressed: {total_energy:.1f} kW")
        lines.append("=" * 80)

        return "\n".join(lines)
