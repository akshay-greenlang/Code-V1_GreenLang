"""
GL-003 UNIFIEDSTEAM - Extended Constraint Definitions

Provides advanced constraint classes for steam system optimization:
- DynamicConstraint: Time-varying constraints with schedules
- OperationalEnvelope: 2D polygon-based safe operating regions
- ProcessConstraint: Production requirements constraints
- EnvironmentalConstraint: Emissions tracking and limits
- ConstraintViolationTracker: Violation tracking and reporting

These extend the base constraints defined in constraints.py.
"""

from datetime import datetime, time as dt_time, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import uuid

from pydantic import BaseModel, Field

from .constraints import (
    ConstraintStatus,
    ConstraintSeverity,
    ConstraintCheckResult,
)


# =============================================================================
# Dynamic Constraints (Time-Varying)
# =============================================================================


class ConstraintSchedule(BaseModel):
    """Schedule definition for time-varying constraints."""

    schedule_id: str = Field(..., description="Schedule identifier")
    start_time: dt_time = Field(..., description="Schedule start time")
    end_time: dt_time = Field(..., description="Schedule end time")
    days_of_week: List[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        description="Days of week (0=Monday)"
    )
    value: float = Field(..., description="Constraint value during this schedule")
    description: str = Field(default="", description="Schedule description")

    def is_active(self, check_time: Optional[datetime] = None) -> bool:
        """
        Check if this schedule is currently active.

        Args:
            check_time: Time to check (default: now)

        Returns:
            True if schedule is active
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)

        # Check day of week
        if check_time.weekday() not in self.days_of_week:
            return False

        # Check time range
        current_time = check_time.time()
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:
            # Handles overnight schedules (e.g., 22:00 to 06:00)
            return current_time >= self.start_time or current_time <= self.end_time


class ConditionalAdjustment(BaseModel):
    """Conditional constraint adjustment based on system state."""

    condition_id: str = Field(..., description="Condition identifier")
    condition_type: str = Field(
        ..., description="Type of condition (temperature, pressure, demand)"
    )
    threshold_value: float = Field(
        ..., description="Threshold that triggers adjustment"
    )
    comparison: str = Field(
        default="greater_than",
        description="Comparison operator (greater_than, less_than, equal)"
    )
    adjustment_value: float = Field(
        ..., description="Constraint adjustment when triggered"
    )
    description: str = Field(default="", description="Condition description")

    def is_triggered(self, current_value: float) -> bool:
        """
        Check if condition is triggered.

        Args:
            current_value: Current value to check against threshold

        Returns:
            True if condition is triggered
        """
        if self.comparison == "greater_than":
            return current_value > self.threshold_value
        elif self.comparison == "less_than":
            return current_value < self.threshold_value
        elif self.comparison == "greater_equal":
            return current_value >= self.threshold_value
        elif self.comparison == "less_equal":
            return current_value <= self.threshold_value
        elif self.comparison == "equal":
            return abs(current_value - self.threshold_value) < 0.001
        return False


class DynamicConstraint(BaseModel):
    """
    Time-varying constraint with schedules and conditional adjustments.

    Supports:
    - Time-of-day schedules (peak/off-peak)
    - Day-of-week variations
    - Conditional adjustments based on system state
    - Default fallback value

    Example:
        >>> constraint = DynamicConstraint(
        ...     constraint_id="mp_header_pressure",
        ...     constraint_type="pressure",
        ...     base_value=150.0,
        ...     unit="psig",
        ...     schedules=[
        ...         ConstraintSchedule(
        ...             schedule_id="peak",
        ...             start_time=time(6, 0),
        ...             end_time=time(18, 0),
        ...             value=155.0,
        ...             description="Peak hours"
        ...         )
        ...     ]
        ... )
        >>> value, explanation = constraint.get_current_value()
    """

    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_type: str = Field(..., description="Type of constraint")
    base_value: float = Field(..., description="Base constraint value")
    unit: str = Field(default="", description="Unit of measurement")

    # Time-based schedules
    schedules: List[ConstraintSchedule] = Field(
        default_factory=list,
        description="Time-based schedule overrides"
    )

    # Conditional adjustments
    conditional_adjustments: List[ConditionalAdjustment] = Field(
        default_factory=list,
        description="Conditional adjustments based on state"
    )

    # Bounds
    min_allowed_value: Optional[float] = Field(
        default=None,
        description="Minimum allowed value after adjustments"
    )
    max_allowed_value: Optional[float] = Field(
        default=None,
        description="Maximum allowed value after adjustments"
    )

    # Metadata
    description: str = Field(default="", description="Constraint description")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def get_current_value(
        self,
        check_time: Optional[datetime] = None,
        system_state: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, str]:
        """
        Get current effective constraint value.

        Args:
            check_time: Time to evaluate (default: now)
            system_state: Current system state for conditional adjustments

        Returns:
            Tuple of (effective_value, explanation)
        """
        if check_time is None:
            check_time = datetime.now(timezone.utc)

        value = self.base_value
        explanations = [f"Base value: {self.base_value}"]

        # Check schedules (first active schedule wins)
        for schedule in self.schedules:
            if schedule.is_active(check_time):
                value = schedule.value
                explanations.append(
                    f"Schedule '{schedule.schedule_id}' active: {schedule.value}"
                )
                break

        # Apply conditional adjustments
        if system_state:
            for adj in self.conditional_adjustments:
                state_key = adj.condition_type
                if state_key in system_state:
                    if adj.is_triggered(system_state[state_key]):
                        value += adj.adjustment_value
                        explanations.append(
                            f"Condition '{adj.condition_id}' triggered: "
                            f"+{adj.adjustment_value}"
                        )

        # Apply bounds
        if self.min_allowed_value is not None and value < self.min_allowed_value:
            value = self.min_allowed_value
            explanations.append(f"Clamped to minimum: {self.min_allowed_value}")

        if self.max_allowed_value is not None and value > self.max_allowed_value:
            value = self.max_allowed_value
            explanations.append(f"Clamped to maximum: {self.max_allowed_value}")

        return value, " | ".join(explanations)

    def check(
        self,
        current_value: float,
        check_time: Optional[datetime] = None,
        system_state: Optional[Dict[str, float]] = None,
    ) -> ConstraintCheckResult:
        """
        Check if current value satisfies the dynamic constraint.

        Args:
            current_value: Current measured value
            check_time: Time to evaluate constraint at
            system_state: System state for conditional adjustments

        Returns:
            ConstraintCheckResult
        """
        limit_value, explanation = self.get_current_value(check_time, system_state)

        # Determine status based on value vs limit
        margin = current_value - limit_value
        margin_percent = (margin / limit_value * 100) if limit_value != 0 else 0

        if abs(margin_percent) < 5:
            status = ConstraintStatus.SATISFIED
            message = f"{self.constraint_id}: Within tolerance"
            severity = ConstraintSeverity.LOW
        elif margin_percent < -10:
            status = ConstraintStatus.WARNING
            message = f"{self.constraint_id}: Below target ({margin_percent:.1f}%)"
            severity = ConstraintSeverity.MEDIUM
        elif margin_percent > 10:
            status = ConstraintStatus.WARNING
            message = f"{self.constraint_id}: Above target ({margin_percent:.1f}%)"
            severity = ConstraintSeverity.MEDIUM
        else:
            status = ConstraintStatus.SATISFIED
            message = f"{self.constraint_id}: OK"
            severity = ConstraintSeverity.LOW

        return ConstraintCheckResult(
            constraint_name=self.constraint_id,
            status=status,
            severity=severity,
            current_value=current_value,
            limit_value=limit_value,
            margin_percent=margin_percent,
            message=f"{message} ({explanation})",
        )


# =============================================================================
# Operational Envelope
# =============================================================================


class EnvelopeVertex(BaseModel):
    """Vertex of an operational envelope polygon."""

    x: float = Field(..., description="X coordinate (first parameter)")
    y: float = Field(..., description="Y coordinate (second parameter)")
    label: Optional[str] = Field(default=None, description="Vertex label")


class OperationalEnvelope(BaseModel):
    """
    Defines safe operating region as a polygon in 2D parameter space.

    Used to define complex operating constraints that cannot be expressed
    as simple min/max limits. Common examples:
    - Pressure vs temperature operating envelope
    - Load vs efficiency envelope
    - Flow vs pressure drop envelope

    Example:
        >>> envelope = OperationalEnvelope(
        ...     envelope_id="boiler_pT_envelope",
        ...     name="Boiler P-T Envelope",
        ...     x_parameter="pressure",
        ...     x_unit="psig",
        ...     y_parameter="temperature",
        ...     y_unit="F",
        ...     vertices=[
        ...         EnvelopeVertex(x=100, y=350),
        ...         EnvelopeVertex(x=150, y=350),
        ...         EnvelopeVertex(x=150, y=450),
        ...         EnvelopeVertex(x=100, y=450),
        ...     ]
        ... )
        >>> inside, dist, msg = envelope.is_within_envelope(125, 400)
    """

    envelope_id: str = Field(..., description="Envelope identifier")
    name: str = Field(..., description="Envelope name")

    # Parameter definitions
    x_parameter: str = Field(..., description="X-axis parameter name")
    x_unit: str = Field(default="", description="X-axis unit")
    y_parameter: str = Field(..., description="Y-axis parameter name")
    y_unit: str = Field(default="", description="Y-axis unit")

    # Polygon vertices (clockwise or counter-clockwise order)
    vertices: List[EnvelopeVertex] = Field(
        ..., min_items=3, description="Envelope vertices"
    )

    # Safety margins
    warning_margin_percent: float = Field(
        default=10.0,
        description="Warning when within this margin of envelope boundary (%)"
    )

    # Metadata
    description: str = Field(default="", description="Envelope description")
    severity: ConstraintSeverity = Field(
        default=ConstraintSeverity.HIGH,
        description="Violation severity"
    )

    def is_within_envelope(
        self,
        x: float,
        y: float,
    ) -> Tuple[bool, float, str]:
        """
        Check if point is within the operational envelope.

        Uses ray casting algorithm for point-in-polygon test.

        Args:
            x: X coordinate value
            y: Y coordinate value

        Returns:
            Tuple of (is_inside, distance_to_boundary, message)
        """
        n = len(self.vertices)
        inside = False

        # Ray casting algorithm
        j = n - 1
        for i in range(n):
            xi, yi = self.vertices[i].x, self.vertices[i].y
            xj, yj = self.vertices[j].x, self.vertices[j].y

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        # Calculate distance to nearest edge
        min_distance = float('inf')
        for i in range(n):
            j = (i + 1) % n
            dist = self._point_to_segment_distance(
                x, y,
                self.vertices[i].x, self.vertices[i].y,
                self.vertices[j].x, self.vertices[j].y,
            )
            min_distance = min(min_distance, dist)

        if inside:
            # Calculate warning threshold based on envelope size
            x_range = max(v.x for v in self.vertices) - min(v.x for v in self.vertices)
            y_range = max(v.y for v in self.vertices) - min(v.y for v in self.vertices)
            warning_dist = min(x_range, y_range) * self.warning_margin_percent / 100

            if min_distance < warning_dist:
                message = f"Within envelope but near boundary ({min_distance:.2f} from edge)"
            else:
                message = "Operating within safe envelope"
        else:
            message = f"OUTSIDE operational envelope by {min_distance:.2f}"

        return inside, min_distance, message

    def _point_to_segment_distance(
        self,
        px: float, py: float,
        x1: float, y1: float,
        x2: float, y2: float,
    ) -> float:
        """Calculate distance from point to line segment."""
        dx = x2 - x1
        dy = y2 - y1
        length_sq = dx * dx + dy * dy

        if length_sq == 0:
            # Segment is a point
            return ((px - x1) ** 2 + (py - y1) ** 2) ** 0.5

        # Parameter t represents position along segment
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))

        # Closest point on segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy

        return ((px - closest_x) ** 2 + (py - closest_y) ** 2) ** 0.5

    def check_point(
        self,
        x: float,
        y: float,
    ) -> ConstraintCheckResult:
        """
        Check if operating point satisfies envelope constraint.

        Args:
            x: X coordinate value
            y: Y coordinate value

        Returns:
            ConstraintCheckResult
        """
        inside, distance, message = self.is_within_envelope(x, y)

        if inside:
            # Calculate warning threshold
            x_range = max(v.x for v in self.vertices) - min(v.x for v in self.vertices)
            y_range = max(v.y for v in self.vertices) - min(v.y for v in self.vertices)
            warning_dist = min(x_range, y_range) * self.warning_margin_percent / 100

            if distance < warning_dist:
                status = ConstraintStatus.WARNING
            else:
                status = ConstraintStatus.SATISFIED
            margin_percent = distance / x_range * 100 if x_range > 0 else 0
        else:
            status = ConstraintStatus.VIOLATED
            margin_percent = -distance

        return ConstraintCheckResult(
            constraint_name=self.envelope_id,
            status=status,
            severity=self.severity,
            current_value=distance,
            limit_value=0.0,  # Boundary is the limit
            margin_percent=margin_percent,
            message=message,
        )


# =============================================================================
# Process Constraints
# =============================================================================


class ProcessConstraint(BaseModel):
    """
    Production/process-related constraint.

    Defines constraints related to production requirements such as:
    - Minimum steam supply to critical processes
    - Temperature requirements for process equipment
    - Pressure requirements for production lines
    """

    constraint_id: str = Field(..., description="Constraint identifier")
    process_name: str = Field(..., description="Process or equipment name")
    constraint_type: str = Field(
        ..., description="Type (steam_supply, temperature, pressure)"
    )

    # Requirements
    minimum_value: Optional[float] = Field(
        default=None, description="Minimum required value"
    )
    maximum_value: Optional[float] = Field(
        default=None, description="Maximum allowed value"
    )
    target_value: Optional[float] = Field(
        default=None, description="Target/optimal value"
    )
    tolerance_percent: float = Field(
        default=5.0, description="Acceptable tolerance from target (%)"
    )
    unit: str = Field(default="", description="Unit of measurement")

    # Criticality
    is_critical: bool = Field(
        default=False, description="Process-critical constraint"
    )
    downtime_cost_per_hour: float = Field(
        default=0.0, ge=0, description="Cost of process downtime ($/hr)"
    )
    max_violation_duration_min: int = Field(
        default=30, description="Maximum violation duration before shutdown (min)"
    )

    # Scheduling
    active_schedule: Optional[str] = Field(
        default=None, description="Production schedule when constraint is active"
    )
    production_line: Optional[str] = Field(
        default=None, description="Associated production line"
    )

    def check(self, current_value: float) -> ConstraintCheckResult:
        """
        Check if process constraint is satisfied.

        Args:
            current_value: Current measured value

        Returns:
            ConstraintCheckResult
        """
        status = ConstraintStatus.SATISFIED
        message = f"{self.process_name}: OK"
        severity = ConstraintSeverity.CRITICAL if self.is_critical else ConstraintSeverity.HIGH
        limit_value = self.minimum_value or self.maximum_value or 0

        # Check minimum
        if self.minimum_value is not None and current_value < self.minimum_value:
            status = ConstraintStatus.VIOLATED
            message = f"{self.process_name}: Below minimum ({current_value} < {self.minimum_value})"
            limit_value = self.minimum_value

        # Check maximum
        elif self.maximum_value is not None and current_value > self.maximum_value:
            status = ConstraintStatus.VIOLATED
            message = f"{self.process_name}: Above maximum ({current_value} > {self.maximum_value})"
            limit_value = self.maximum_value

        # Check target tolerance
        elif self.target_value is not None:
            deviation_pct = abs(current_value - self.target_value) / self.target_value * 100
            if deviation_pct > self.tolerance_percent * 2:
                status = ConstraintStatus.WARNING
                message = f"{self.process_name}: Outside tolerance ({deviation_pct:.1f}% from target)"
            limit_value = self.target_value

        # Calculate margin
        margin_percent = 0.0
        if self.minimum_value is not None and self.maximum_value is not None:
            range_size = self.maximum_value - self.minimum_value
            margin_to_limit = min(
                current_value - self.minimum_value,
                self.maximum_value - current_value
            )
            margin_percent = margin_to_limit / range_size * 100 if range_size > 0 else 0
        elif self.minimum_value is not None:
            margin_percent = (current_value - self.minimum_value) / self.minimum_value * 100

        return ConstraintCheckResult(
            constraint_name=self.constraint_id,
            status=status,
            severity=severity,
            current_value=current_value,
            limit_value=limit_value,
            margin_percent=max(0, margin_percent),
            message=message,
        )


# =============================================================================
# Environmental Constraints
# =============================================================================


class EnvironmentalConstraint(BaseModel):
    """
    Environmental/emissions constraint with tracking.

    Tracks emissions against regulatory limits:
    - Hourly limits (permit conditions)
    - Daily rolling averages
    - Annual caps
    """

    constraint_id: str = Field(..., description="Constraint identifier")
    pollutant: str = Field(..., description="Pollutant name (CO2, NOx, SOx, PM)")
    unit: str = Field(default="lb/hr", description="Emission rate unit")

    # Limits
    hourly_limit: Optional[float] = Field(
        default=None, description="Hourly emission limit"
    )
    daily_limit: Optional[float] = Field(
        default=None, description="Daily emission limit"
    )
    annual_limit: Optional[float] = Field(
        default=None, description="Annual emission cap"
    )

    # Current accumulations
    hourly_accumulated: float = Field(
        default=0.0, ge=0, description="Current hour accumulation"
    )
    daily_accumulated: float = Field(
        default=0.0, ge=0, description="Current day accumulation"
    )
    annual_accumulated: float = Field(
        default=0.0, ge=0, description="Year-to-date accumulation"
    )

    # Tracking
    last_reset_hourly: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_reset_daily: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_reset_annual: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Regulatory
    permit_number: Optional[str] = Field(
        default=None, description="Associated permit number"
    )
    penalty_per_violation: float = Field(
        default=10000.0, description="Regulatory penalty per violation ($)"
    )

    def add_emission(
        self,
        emission_rate: float,
        duration_hours: float = 1.0,
    ) -> None:
        """
        Add emission to accumulators.

        Args:
            emission_rate: Emission rate in constraint units
            duration_hours: Duration of emission (hours)
        """
        emission_total = emission_rate * duration_hours

        self.hourly_accumulated += emission_total
        self.daily_accumulated += emission_total
        self.annual_accumulated += emission_total

    def reset_hourly(self) -> None:
        """Reset hourly accumulator."""
        self.hourly_accumulated = 0.0
        self.last_reset_hourly = datetime.now(timezone.utc)

    def reset_daily(self) -> None:
        """Reset daily accumulator."""
        self.daily_accumulated = 0.0
        self.last_reset_daily = datetime.now(timezone.utc)

    def reset_annual(self) -> None:
        """Reset annual accumulator."""
        self.annual_accumulated = 0.0
        self.last_reset_annual = datetime.now(timezone.utc)

    def check_all_limits(
        self,
        current_rate: Optional[float] = None,
    ) -> List[ConstraintCheckResult]:
        """
        Check all emission limits.

        Args:
            current_rate: Current emission rate (optional)

        Returns:
            List of constraint check results
        """
        results = []

        # Check hourly
        if self.hourly_limit is not None:
            value = self.hourly_accumulated
            if current_rate is not None:
                value = current_rate

            pct_used = value / self.hourly_limit * 100 if self.hourly_limit > 0 else 0

            if value > self.hourly_limit:
                status = ConstraintStatus.VIOLATED
                message = f"{self.pollutant} hourly limit EXCEEDED"
                severity = ConstraintSeverity.CRITICAL
            elif pct_used > 90:
                status = ConstraintStatus.WARNING
                message = f"{self.pollutant} hourly limit at {pct_used:.0f}%"
                severity = ConstraintSeverity.HIGH
            else:
                status = ConstraintStatus.SATISFIED
                message = f"{self.pollutant} hourly OK ({pct_used:.0f}% of limit)"
                severity = ConstraintSeverity.LOW

            results.append(ConstraintCheckResult(
                constraint_name=f"{self.constraint_id}_hourly",
                status=status,
                severity=severity,
                current_value=value,
                limit_value=self.hourly_limit,
                margin_percent=100 - pct_used,
                message=message,
            ))

        # Check daily
        if self.daily_limit is not None:
            pct_used = self.daily_accumulated / self.daily_limit * 100 if self.daily_limit > 0 else 0

            if self.daily_accumulated > self.daily_limit:
                status = ConstraintStatus.VIOLATED
                message = f"{self.pollutant} daily limit EXCEEDED"
                severity = ConstraintSeverity.CRITICAL
            elif pct_used > 80:
                status = ConstraintStatus.WARNING
                message = f"{self.pollutant} daily limit at {pct_used:.0f}%"
                severity = ConstraintSeverity.HIGH
            else:
                status = ConstraintStatus.SATISFIED
                message = f"{self.pollutant} daily OK ({pct_used:.0f}% of limit)"
                severity = ConstraintSeverity.LOW

            results.append(ConstraintCheckResult(
                constraint_name=f"{self.constraint_id}_daily",
                status=status,
                severity=severity,
                current_value=self.daily_accumulated,
                limit_value=self.daily_limit,
                margin_percent=100 - pct_used,
                message=message,
            ))

        # Check annual
        if self.annual_limit is not None:
            pct_used = self.annual_accumulated / self.annual_limit * 100 if self.annual_limit > 0 else 0

            if self.annual_accumulated > self.annual_limit:
                status = ConstraintStatus.VIOLATED
                message = f"{self.pollutant} annual cap EXCEEDED"
                severity = ConstraintSeverity.CRITICAL
            elif pct_used > 90:
                status = ConstraintStatus.WARNING
                message = f"{self.pollutant} annual cap at {pct_used:.0f}%"
                severity = ConstraintSeverity.HIGH
            else:
                status = ConstraintStatus.SATISFIED
                message = f"{self.pollutant} annual OK ({pct_used:.0f}% of cap)"
                severity = ConstraintSeverity.MEDIUM

            results.append(ConstraintCheckResult(
                constraint_name=f"{self.constraint_id}_annual",
                status=status,
                severity=severity,
                current_value=self.annual_accumulated,
                limit_value=self.annual_limit,
                margin_percent=100 - pct_used,
                message=message,
            ))

        return results

    def get_remaining_budget(self) -> Dict[str, float]:
        """Get remaining emission budget for each period."""
        return {
            "hourly": (self.hourly_limit - self.hourly_accumulated) if self.hourly_limit else float('inf'),
            "daily": (self.daily_limit - self.daily_accumulated) if self.daily_limit else float('inf'),
            "annual": (self.annual_limit - self.annual_accumulated) if self.annual_limit else float('inf'),
        }


# =============================================================================
# Constraint Violation Tracking
# =============================================================================


class ViolationRecord(BaseModel):
    """Record of a single constraint violation event."""

    violation_id: str = Field(..., description="Unique violation ID")
    constraint_id: str = Field(..., description="Violated constraint ID")
    constraint_type: str = Field(..., description="Type of constraint")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    duration_seconds: float = Field(
        default=0.0, ge=0, description="Violation duration (seconds)"
    )
    severity: ConstraintSeverity = Field(..., description="Violation severity")
    violated_value: float = Field(..., description="Value that violated constraint")
    limit_value: float = Field(..., description="Constraint limit that was violated")
    overshoot_percent: float = Field(
        default=0.0, description="Percent beyond limit"
    )
    root_cause: Optional[str] = Field(
        default=None, description="Identified root cause"
    )
    corrective_action: Optional[str] = Field(
        default=None, description="Corrective action taken"
    )
    acknowledged_by: Optional[str] = Field(
        default=None, description="Operator who acknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None, description="Acknowledgment timestamp"
    )
    resolved_at: Optional[datetime] = Field(
        default=None, description="Resolution timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

    def calculate_overshoot(self) -> None:
        """Calculate overshoot percentage."""
        if self.limit_value != 0:
            self.overshoot_percent = abs(
                (self.violated_value - self.limit_value) / self.limit_value * 100
            )


class ConstraintViolationTracker(BaseModel):
    """
    Tracks constraint violations over time for reporting and analysis.

    Provides:
    - Violation history with full audit trail
    - Statistical analysis of violations
    - Trend detection
    - Compliance reporting
    """

    tracker_id: str = Field(default="violation_tracker", description="Tracker ID")
    violations: List[ViolationRecord] = Field(
        default_factory=list, description="Violation history"
    )
    active_violations: Dict[str, ViolationRecord] = Field(
        default_factory=dict, description="Currently active violations"
    )

    # Statistics
    total_violations_24h: int = Field(default=0, description="Violations in last 24h")
    total_violations_7d: int = Field(default=0, description="Violations in last 7 days")
    total_violations_30d: int = Field(default=0, description="Violations in last 30 days")
    mean_duration_seconds: float = Field(
        default=0.0, description="Mean violation duration"
    )

    # Thresholds
    max_violations_per_day: int = Field(
        default=10, description="Maximum acceptable violations per day"
    )
    alert_threshold: int = Field(
        default=5, description="Alert after this many violations"
    )

    def record_violation(
        self,
        constraint_id: str,
        constraint_type: str,
        severity: ConstraintSeverity,
        violated_value: float,
        limit_value: float,
    ) -> ViolationRecord:
        """
        Record a new constraint violation.

        Args:
            constraint_id: ID of violated constraint
            constraint_type: Type of constraint
            severity: Violation severity
            violated_value: Value that violated the constraint
            limit_value: The constraint limit

        Returns:
            Created ViolationRecord
        """
        violation = ViolationRecord(
            violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            severity=severity,
            violated_value=violated_value,
            limit_value=limit_value,
        )
        violation.calculate_overshoot()

        # Generate provenance hash
        hash_data = (
            f"{violation.violation_id}"
            f"{constraint_id}"
            f"{violated_value}"
            f"{violation.timestamp.isoformat()}"
        )
        violation.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        # Add to history
        self.violations.append(violation)

        # Track as active
        self.active_violations[constraint_id] = violation

        # Update statistics
        self._update_statistics()

        return violation

    def resolve_violation(
        self,
        constraint_id: str,
        corrective_action: Optional[str] = None,
    ) -> Optional[ViolationRecord]:
        """
        Mark a violation as resolved.

        Args:
            constraint_id: ID of the constraint
            corrective_action: Action taken to resolve

        Returns:
            Resolved ViolationRecord or None if not found
        """
        if constraint_id not in self.active_violations:
            return None

        violation = self.active_violations.pop(constraint_id)
        violation.resolved_at = datetime.now(timezone.utc)
        violation.duration_seconds = (
            violation.resolved_at - violation.timestamp
        ).total_seconds()

        if corrective_action:
            violation.corrective_action = corrective_action

        # Update statistics
        self._update_statistics()

        return violation

    def acknowledge_violation(
        self,
        violation_id: str,
        operator: str,
    ) -> bool:
        """
        Acknowledge a violation.

        Args:
            violation_id: ID of the violation
            operator: Operator name/ID

        Returns:
            True if acknowledged, False if not found
        """
        for violation in self.violations:
            if violation.violation_id == violation_id:
                violation.acknowledged_by = operator
                violation.acknowledged_at = datetime.now(timezone.utc)
                return True
        return False

    def _update_statistics(self) -> None:
        """Update violation statistics."""
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        self.total_violations_24h = sum(
            1 for v in self.violations if v.timestamp >= day_ago
        )
        self.total_violations_7d = sum(
            1 for v in self.violations if v.timestamp >= week_ago
        )
        self.total_violations_30d = sum(
            1 for v in self.violations if v.timestamp >= month_ago
        )

        # Mean duration
        resolved = [v for v in self.violations if v.resolved_at is not None]
        if resolved:
            self.mean_duration_seconds = sum(
                v.duration_seconds for v in resolved
            ) / len(resolved)

    def get_violations_by_constraint(
        self,
        constraint_id: str,
        since: Optional[datetime] = None,
    ) -> List[ViolationRecord]:
        """Get violations for a specific constraint."""
        violations = [v for v in self.violations if v.constraint_id == constraint_id]
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        return violations

    def get_violations_by_severity(
        self,
        severity: ConstraintSeverity,
        since: Optional[datetime] = None,
    ) -> List[ViolationRecord]:
        """Get violations by severity level."""
        violations = [v for v in self.violations if v.severity == severity]
        if since:
            violations = [v for v in violations if v.timestamp >= since]
        return violations

    def get_compliance_report(
        self,
        period_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Generate compliance report for the specified period.

        Args:
            period_days: Reporting period in days

        Returns:
            Compliance report dictionary
        """
        since = datetime.now(timezone.utc) - timedelta(days=period_days)
        period_violations = [v for v in self.violations if v.timestamp >= since]

        # Group by constraint
        by_constraint: Dict[str, int] = {}
        for v in period_violations:
            by_constraint[v.constraint_id] = by_constraint.get(v.constraint_id, 0) + 1

        # Group by severity
        by_severity: Dict[str, int] = {s.value: 0 for s in ConstraintSeverity}
        for v in period_violations:
            by_severity[v.severity.value] += 1

        # Calculate compliance percentage
        total_hours = period_days * 24
        violation_hours = sum(v.duration_seconds / 3600 for v in period_violations)
        compliance_percent = (1 - violation_hours / total_hours) * 100 if total_hours > 0 else 100

        return {
            "report_period_days": period_days,
            "total_violations": len(period_violations),
            "violations_by_constraint": by_constraint,
            "violations_by_severity": by_severity,
            "total_violation_hours": round(violation_hours, 2),
            "compliance_percent": round(compliance_percent, 2),
            "mean_violation_duration_min": round(self.mean_duration_seconds / 60, 1),
            "active_violations": len(self.active_violations),
            "worst_constraint": max(by_constraint.items(), key=lambda x: x[1])[0] if by_constraint else None,
        }
