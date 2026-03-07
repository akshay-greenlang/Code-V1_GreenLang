# -*- coding: utf-8 -*-
"""
Temporal Consistency Analyzer - AGENT-EUDR-002: Geolocation Verification (Feature 4)

Analyzes temporal consistency of production plot boundaries over time. Detects
boundary expansion, contraction, centroid shifts, rapid successive changes,
and potential forest encroachment. Supports EUDR Article 10 risk assessment
by identifying suspicious boundary modifications post-cutoff (2020-12-31).

Zero-Hallucination Guarantees:
    - All calculations are deterministic (Haversine, spherical excess, centroid)
    - Boundary comparison uses pure geometry (no ML/LLM)
    - Temporal window analysis uses calendar arithmetic only
    - SHA-256 provenance hashes on all analysis results
    - No external geospatial library required for core logic

Performance Targets:
    - Single boundary analysis: <5ms
    - History analysis (100 boundary versions): <50ms

Regulatory References:
    - EUDR Article 9: Geolocation of production plots
    - EUDR Article 10: Risk assessment (boundary changes as risk signal)
    - EUDR Article 31: Record retention (5-year audit trail)
    - EUDR Cutoff Date: 31 December 2020

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-002 (Feature 4: Temporal Boundary Analysis)
Agent ID: GL-EUDR-GEO-002
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import field as dataclass_field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    BoundaryChange,
    ChangeType,
    IssueSeverity,
    TemporalChangeResult,
    ValidationIssue,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth radius in metres (WGS84 mean radius).
EARTH_RADIUS_M: float = 6_371_000.0

#: Hectares per square metre.
HA_PER_SQ_M: float = 1.0e-4

#: EUDR deforestation cutoff date.
EUDR_CUTOFF_DATE: str = "2020-12-31"

#: Threshold for significant area change (percentage).
SIGNIFICANT_AREA_CHANGE_PCT: float = 5.0

#: Threshold for significant centroid shift (metres).
SIGNIFICANT_CENTROID_SHIFT_M: float = 50.0

#: Default rapid-change detection window (days).
DEFAULT_RAPID_CHANGE_WINDOW_DAYS: int = 90

#: Maximum number of changes in a rapid-change window to trigger alert.
RAPID_CHANGE_MAX_COUNT: int = 3

#: Area expansion threshold that triggers forest encroachment check (%).
FOREST_ENCROACHMENT_AREA_THRESHOLD_PCT: float = 10.0

#: Centroid shift threshold for encroachment direction check (metres).
FOREST_ENCROACHMENT_SHIFT_THRESHOLD_M: float = 100.0

#: Tolerance for "stable" boundary (area change < this %, shift < 5m).
STABLE_AREA_TOLERANCE_PCT: float = 1.0
STABLE_SHIFT_TOLERANCE_M: float = 5.0


# ---------------------------------------------------------------------------
# TemporalConsistencyAnalyzer
# ---------------------------------------------------------------------------


class TemporalConsistencyAnalyzer:
    """Production-grade temporal boundary consistency analyzer for EUDR.

    Analyzes changes in production plot boundaries over time to detect
    suspicious modifications that may indicate deforestation. Supports:
    - Boundary expansion / contraction detection
    - Centroid shift measurement (Haversine distance)
    - Rapid successive change detection (within configurable window)
    - Forest encroachment assessment (expansion direction analysis)
    - Complete boundary change history with provenance hashes

    All calculations are deterministic with zero LLM/ML involvement.

    Example::

        analyzer = TemporalConsistencyAnalyzer()

        prev_boundary = [(-3.0, 28.0), (-3.0, 28.1), (-3.1, 28.1),
                         (-3.1, 28.0), (-3.0, 28.0)]
        new_boundary = [(-3.0, 27.99), (-3.0, 28.11), (-3.11, 28.11),
                        (-3.11, 27.99), (-3.0, 27.99)]

        result = analyzer.analyze_boundary_change(
            plot_id="PLOT-001",
            previous_boundary=prev_boundary,
            new_boundary=new_boundary,
        )
        assert result.provenance_hash != ""

    Attributes:
        rapid_change_window_days: Window in days for rapid-change detection.
        boundary_history: In-memory boundary change history by plot_id.
    """

    def __init__(
        self,
        rapid_change_window_days: int = DEFAULT_RAPID_CHANGE_WINDOW_DAYS,
        config: Any = None,
    ) -> None:
        """Initialize the TemporalConsistencyAnalyzer.

        Args:
            rapid_change_window_days: Number of days within which
                multiple boundary changes trigger a rapid-change alert.
            config: Optional GeolocationVerificationConfig instance.
                Currently reserved for future configuration extraction.
        """
        self.rapid_change_window_days = rapid_change_window_days
        # In-memory boundary history store (keyed by plot_id)
        self._boundary_history: Dict[str, List[BoundaryChange]] = defaultdict(list)
        logger.info(
            "TemporalConsistencyAnalyzer initialized: "
            "rapid_change_window=%d days",
            self.rapid_change_window_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_boundary_change(
        self,
        plot_id: str,
        previous_boundary: List[Tuple[float, float]],
        new_boundary: List[Tuple[float, float]],
        forest_boundary: Optional[List[Tuple[float, float]]] = None,
    ) -> TemporalChangeResult:
        """Analyze boundary change between two time periods -- DETERMINISTIC.

        Computes area change, centroid shift, detects expansion/contraction,
        checks for forest encroachment, and evaluates rapid-change patterns.

        Args:
            plot_id: Production plot identifier.
            previous_boundary: Previous boundary vertices (lat, lon).
            new_boundary: New boundary vertices (lat, lon).
            forest_boundary: Optional forest boundary for encroachment check.

        Returns:
            TemporalChangeResult with all analysis results.
        """
        start_time = time.monotonic()
        result = TemporalChangeResult(plot_id=plot_id)
        issues: List[ValidationIssue] = []

        # Validate inputs
        if not previous_boundary or len(previous_boundary) < 3:
            issues.append(ValidationIssue(
                code="TEMPORAL_INVALID_PREV_BOUNDARY",
                severity=IssueSeverity.HIGH,
                message="Previous boundary has fewer than 3 vertices.",
                field="previous_boundary",
            ))
            result.is_consistent = False
            result.issues = issues
            result.provenance_hash = self._compute_result_hash(result)
            return result

        if not new_boundary or len(new_boundary) < 3:
            issues.append(ValidationIssue(
                code="TEMPORAL_INVALID_NEW_BOUNDARY",
                severity=IssueSeverity.HIGH,
                message="New boundary has fewer than 3 vertices.",
                field="new_boundary",
            ))
            result.is_consistent = False
            result.issues = issues
            result.provenance_hash = self._compute_result_hash(result)
            return result

        # 1. Calculate areas
        prev_area = self._calculate_geodesic_area(previous_boundary)
        new_area = self._calculate_geodesic_area(new_boundary)

        # 2. Calculate centroids
        prev_centroid = self.calculate_centroid(previous_boundary)
        new_centroid = self.calculate_centroid(new_boundary)

        # 3. Compute area change percentage
        area_change_pct = self._calculate_area_change(prev_area, new_area)

        # 4. Compute centroid shift in metres
        centroid_shift_m = self._calculate_centroid_shift(
            prev_centroid, new_centroid
        )

        # 5. Detect change type
        change_type = self._classify_change(area_change_pct, centroid_shift_m)

        # 6. Detect expansion
        expansion = self.detect_expansion(previous_boundary, new_boundary)

        # 7. Detect shift
        shift = self.detect_shift(prev_centroid, new_centroid)

        # 8. Check forest encroachment
        forest_encroachment = False
        if expansion is not None and forest_boundary is not None:
            forest_encroachment = self._check_forest_encroachment(
                prev_centroid, new_centroid, forest_boundary
            )

        # Build boundary change record
        boundary_change = BoundaryChange(
            change_type=change_type,
            area_change_pct=round(area_change_pct, 4),
            centroid_shift_m=round(centroid_shift_m, 2),
            previous_area_ha=round(prev_area, 4),
            new_area_ha=round(new_area, 4),
            previous_centroid=prev_centroid,
            new_centroid=new_centroid,
            forest_encroachment=forest_encroachment,
        )

        # Store in history
        self._boundary_history[plot_id].append(boundary_change)

        # 9. Check for rapid changes
        rapid_change = self.detect_rapid_changes(
            plot_id,
            self._boundary_history[plot_id],
            self.rapid_change_window_days,
        )

        # 10. Generate issues
        if abs(area_change_pct) > SIGNIFICANT_AREA_CHANGE_PCT:
            direction = "expanded" if area_change_pct > 0 else "contracted"
            issues.append(ValidationIssue(
                code="TEMPORAL_SIGNIFICANT_AREA_CHANGE",
                severity=IssueSeverity.HIGH,
                message=f"Plot boundary {direction} by "
                        f"{abs(area_change_pct):.2f}% "
                        f"({prev_area:.2f} ha -> {new_area:.2f} ha).",
                field="area",
                details={
                    "previous_area_ha": round(prev_area, 4),
                    "new_area_ha": round(new_area, 4),
                    "change_pct": round(area_change_pct, 4),
                },
            ))

        if centroid_shift_m > SIGNIFICANT_CENTROID_SHIFT_M:
            issues.append(ValidationIssue(
                code="TEMPORAL_CENTROID_SHIFT",
                severity=IssueSeverity.MEDIUM,
                message=f"Plot centroid shifted by {centroid_shift_m:.1f}m "
                        f"(threshold: {SIGNIFICANT_CENTROID_SHIFT_M:.0f}m).",
                field="centroid",
                details={
                    "shift_m": round(centroid_shift_m, 2),
                    "previous_centroid": list(prev_centroid),
                    "new_centroid": list(new_centroid),
                },
            ))

        if forest_encroachment:
            issues.append(ValidationIssue(
                code="TEMPORAL_FOREST_ENCROACHMENT",
                severity=IssueSeverity.CRITICAL,
                message="Boundary expansion direction indicates potential "
                        "encroachment into forest area.",
                field="boundary",
            ))

        if rapid_change:
            issues.append(ValidationIssue(
                code="TEMPORAL_RAPID_CHANGES",
                severity=IssueSeverity.HIGH,
                message=f"Plot has undergone {len(self._boundary_history[plot_id])} "
                        f"boundary changes within "
                        f"{self.rapid_change_window_days} days.",
                field="boundary_history",
            ))

        # Determine consistency
        critical_issues = [
            i for i in issues if i.severity == IssueSeverity.CRITICAL
        ]
        high_issues = [
            i for i in issues if i.severity == IssueSeverity.HIGH
        ]
        result.is_consistent = (
            len(critical_issues) == 0 and len(high_issues) == 0
        )

        result.boundary_change = boundary_change
        result.rapid_change_detected = rapid_change
        result.change_history = list(self._boundary_history[plot_id])
        result.issues = issues
        result.provenance_hash = self._compute_result_hash(result)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.debug(
            "Temporal analysis %s for plot %s: change_type=%s, "
            "area_change=%.2f%%, shift=%.1fm, consistent=%s, "
            "issues=%d, %.2fms",
            result.analysis_id, plot_id, change_type.value,
            area_change_pct, centroid_shift_m, result.is_consistent,
            len(issues), elapsed_ms,
        )

        return result

    def detect_expansion(
        self,
        prev_vertices: List[Tuple[float, float]],
        new_vertices: List[Tuple[float, float]],
    ) -> Optional[BoundaryChange]:
        """Detect boundary expansion between two polygon versions.

        An expansion is detected when the new area exceeds the previous
        area beyond the stable tolerance threshold.

        Args:
            prev_vertices: Previous boundary vertices.
            new_vertices: New boundary vertices.

        Returns:
            BoundaryChange if expansion detected, else None.
        """
        prev_area = self._calculate_geodesic_area(prev_vertices)
        new_area = self._calculate_geodesic_area(new_vertices)
        area_change_pct = self._calculate_area_change(prev_area, new_area)

        if area_change_pct <= STABLE_AREA_TOLERANCE_PCT:
            return None

        prev_centroid = self.calculate_centroid(prev_vertices)
        new_centroid = self.calculate_centroid(new_vertices)
        shift_m = self._calculate_centroid_shift(prev_centroid, new_centroid)

        return BoundaryChange(
            change_type=ChangeType.EXPANSION,
            area_change_pct=round(area_change_pct, 4),
            centroid_shift_m=round(shift_m, 2),
            previous_area_ha=round(prev_area, 4),
            new_area_ha=round(new_area, 4),
            previous_centroid=prev_centroid,
            new_centroid=new_centroid,
        )

    def detect_shift(
        self,
        prev_centroid: Tuple[float, float],
        new_centroid: Tuple[float, float],
    ) -> Optional[BoundaryChange]:
        """Detect centroid shift between two boundary versions.

        A shift is detected when the centroid moves beyond the stable
        tolerance threshold.

        Args:
            prev_centroid: Previous centroid (lat, lon).
            new_centroid: New centroid (lat, lon).

        Returns:
            BoundaryChange if shift detected, else None.
        """
        shift_m = self._calculate_centroid_shift(prev_centroid, new_centroid)

        if shift_m <= STABLE_SHIFT_TOLERANCE_M:
            return None

        return BoundaryChange(
            change_type=ChangeType.SHIFT,
            centroid_shift_m=round(shift_m, 2),
            previous_centroid=prev_centroid,
            new_centroid=new_centroid,
        )

    def calculate_centroid(
        self, vertices: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Calculate the centroid of a polygon.

        Uses the weighted centroid formula for a simple polygon:
            C_x = (1 / 6A) * sum((x_i + x_{i+1}) * cross_i)
            C_y = (1 / 6A) * sum((y_i + y_{i+1}) * cross_i)

        where cross_i = x_i * y_{i+1} - x_{i+1} * y_i and A is the
        signed area from the shoelace formula.

        For geographic coordinates, uses lon as x and lat as y.

        Args:
            vertices: List of (lat, lon) tuples.

        Returns:
            Centroid as (lat, lon) tuple.
        """
        n = len(vertices)
        if n == 0:
            return (0.0, 0.0)

        if n <= 2:
            # Simple average for degenerate cases
            avg_lat = sum(v[0] for v in vertices) / n
            avg_lon = sum(v[1] for v in vertices) / n
            return (round(avg_lat, 8), round(avg_lon, 8))

        # Ensure ring is closed for the formula
        verts = list(vertices)
        if verts[0] != verts[-1]:
            verts.append(verts[0])

        signed_area_2 = 0.0  # 2 * signed area
        cx = 0.0  # centroid x (lon)
        cy = 0.0  # centroid y (lat)

        for i in range(len(verts) - 1):
            x_i = verts[i][1]      # lon
            y_i = verts[i][0]      # lat
            x_next = verts[i + 1][1]
            y_next = verts[i + 1][0]

            cross = x_i * y_next - x_next * y_i
            signed_area_2 += cross
            cx += (x_i + x_next) * cross
            cy += (y_i + y_next) * cross

        if abs(signed_area_2) < 1e-15:
            # Degenerate polygon -- fall back to simple average
            avg_lat = sum(v[0] for v in vertices) / n
            avg_lon = sum(v[1] for v in vertices) / n
            return (round(avg_lat, 8), round(avg_lon, 8))

        signed_area_6 = signed_area_2 * 3.0  # 6A
        cx /= signed_area_6
        cy /= signed_area_6

        return (round(cy, 8), round(cx, 8))

    def get_boundary_history(
        self, plot_id: str
    ) -> List[BoundaryChange]:
        """Retrieve the boundary change history for a plot.

        Args:
            plot_id: Production plot identifier.

        Returns:
            List of BoundaryChange records in chronological order.
        """
        return list(self._boundary_history.get(plot_id, []))

    def detect_rapid_changes(
        self,
        plot_id: str,
        changes: List[BoundaryChange],
        window_days: int,
    ) -> bool:
        """Detect rapid successive boundary changes within a time window.

        Flags a plot if it has undergone more than RAPID_CHANGE_MAX_COUNT
        boundary changes within the specified number of days.

        Args:
            plot_id: Production plot identifier.
            changes: List of BoundaryChange records.
            window_days: Time window in days to check.

        Returns:
            True if rapid changes are detected.
        """
        if len(changes) <= RAPID_CHANGE_MAX_COUNT:
            return False

        # Sort changes by detection timestamp
        sorted_changes = sorted(
            changes, key=lambda c: c.detected_at
        )

        # Sliding window: count changes within window_days
        window = timedelta(days=window_days)
        for i in range(len(sorted_changes)):
            window_start = sorted_changes[i].detected_at
            count = 0
            for j in range(i, len(sorted_changes)):
                if sorted_changes[j].detected_at - window_start <= window:
                    count += 1
                else:
                    break

            if count > RAPID_CHANGE_MAX_COUNT:
                logger.warning(
                    "Rapid boundary changes detected for plot %s: "
                    "%d changes within %d days",
                    plot_id, count, window_days,
                )
                return True

        return False

    def clear_history(self, plot_id: Optional[str] = None) -> None:
        """Clear boundary change history.

        Args:
            plot_id: If provided, clear only the specified plot's history.
                If None, clear all history.
        """
        if plot_id is not None:
            self._boundary_history.pop(plot_id, None)
            logger.info("Cleared boundary history for plot %s", plot_id)
        else:
            self._boundary_history.clear()
            logger.info("Cleared all boundary history")

    # ------------------------------------------------------------------
    # Internal: Area Calculations
    # ------------------------------------------------------------------

    def _calculate_geodesic_area(
        self, vertices: List[Tuple[float, float]]
    ) -> float:
        """Calculate geodesic area of a polygon in hectares.

        Uses the spherical excess formula (same as polygon_verifier).

        Args:
            vertices: List of (lat, lon) tuples.

        Returns:
            Area in hectares.
        """
        n = len(vertices)
        if n < 3:
            return 0.0

        # Ensure ring closure
        verts = list(vertices)
        if verts[0] != verts[-1]:
            verts.append(verts[0])
        n = len(verts)

        total_excess = 0.0
        for i in range(n - 1):
            lat1 = math.radians(verts[i][0])
            lon1 = math.radians(verts[i][1])
            lat2 = math.radians(verts[(i + 1) % (n - 1)][0])
            lon2 = math.radians(verts[(i + 1) % (n - 1)][1])

            total_excess += (lon2 - lon1) * (
                2.0 + math.sin(lat1) + math.sin(lat2)
            )

        area_sq_m = abs(total_excess) * EARTH_RADIUS_M * EARTH_RADIUS_M / 2.0
        return area_sq_m * HA_PER_SQ_M

    def _calculate_area_change(
        self, prev_area: float, new_area: float
    ) -> float:
        """Calculate percentage change in area.

        Positive values indicate expansion, negative indicate contraction.

        Args:
            prev_area: Previous area in hectares.
            new_area: New area in hectares.

        Returns:
            Percentage change (e.g., 5.0 means 5% expansion).
        """
        if prev_area <= 0:
            if new_area > 0:
                return 100.0
            return 0.0

        return ((new_area - prev_area) / prev_area) * 100.0

    def _calculate_centroid_shift(
        self,
        prev: Tuple[float, float],
        new: Tuple[float, float],
    ) -> float:
        """Calculate centroid shift distance in metres via Haversine.

        Args:
            prev: Previous centroid (lat, lon).
            new: New centroid (lat, lon).

        Returns:
            Distance in metres.
        """
        return self._haversine_distance(
            prev[0], prev[1], new[0], new[1]
        )

    # ------------------------------------------------------------------
    # Internal: Change Classification
    # ------------------------------------------------------------------

    def _classify_change(
        self, area_change_pct: float, centroid_shift_m: float
    ) -> ChangeType:
        """Classify the type of boundary change.

        Classification rules:
            - STABLE: area change < 1% AND shift < 5m
            - EXPANSION: area change > +1% (positive growth)
            - CONTRACTION: area change < -1% (negative growth)
            - SHIFT: area change < 1% BUT shift > 5m
            - RESHAPE: significant area change AND significant shift

        Args:
            area_change_pct: Percentage change in area.
            centroid_shift_m: Centroid shift in metres.

        Returns:
            ChangeType classification.
        """
        abs_area = abs(area_change_pct)
        is_area_stable = abs_area <= STABLE_AREA_TOLERANCE_PCT
        is_shift_stable = centroid_shift_m <= STABLE_SHIFT_TOLERANCE_M

        if is_area_stable and is_shift_stable:
            return ChangeType.STABLE

        if not is_area_stable and not is_shift_stable:
            # Both area and position changed significantly
            return ChangeType.RESHAPE

        if area_change_pct > STABLE_AREA_TOLERANCE_PCT:
            return ChangeType.EXPANSION

        if area_change_pct < -STABLE_AREA_TOLERANCE_PCT:
            return ChangeType.CONTRACTION

        if not is_shift_stable:
            return ChangeType.SHIFT

        return ChangeType.STABLE

    # ------------------------------------------------------------------
    # Internal: Forest Encroachment
    # ------------------------------------------------------------------

    def _check_forest_encroachment(
        self,
        prev_centroid: Tuple[float, float],
        new_centroid: Tuple[float, float],
        forest_boundary: List[Tuple[float, float]],
    ) -> bool:
        """Check whether boundary expansion direction encroaches on forest.

        Uses a simplified heuristic: if the centroid shift direction
        moves the plot closer to the forest boundary centroid, and the
        shift magnitude exceeds the encroachment threshold, flag as
        potential encroachment.

        A production implementation would use PostGIS ST_Intersects
        with actual forest cover polygons from AGENT-DATA-007.

        Args:
            prev_centroid: Previous plot centroid (lat, lon).
            new_centroid: New plot centroid (lat, lon).
            forest_boundary: Forest area boundary vertices.

        Returns:
            True if encroachment is suspected.
        """
        if not forest_boundary:
            return False

        # Calculate forest centroid
        forest_centroid = self.calculate_centroid(forest_boundary)

        # Distance from previous centroid to forest
        prev_dist_to_forest = self._haversine_distance(
            prev_centroid[0], prev_centroid[1],
            forest_centroid[0], forest_centroid[1],
        )

        # Distance from new centroid to forest
        new_dist_to_forest = self._haversine_distance(
            new_centroid[0], new_centroid[1],
            forest_centroid[0], forest_centroid[1],
        )

        # Centroid shift magnitude
        shift_m = self._haversine_distance(
            prev_centroid[0], prev_centroid[1],
            new_centroid[0], new_centroid[1],
        )

        # Encroachment condition:
        # 1. Plot moved closer to forest
        # 2. Shift is significant
        moved_closer = new_dist_to_forest < prev_dist_to_forest
        significant_shift = shift_m > FOREST_ENCROACHMENT_SHIFT_THRESHOLD_M

        # Also check if the new centroid is within 500m of forest
        within_proximity = new_dist_to_forest < 500.0

        if moved_closer and (significant_shift or within_proximity):
            logger.warning(
                "Potential forest encroachment detected: "
                "plot moved %.1fm closer to forest boundary "
                "(prev_dist=%.1fm, new_dist=%.1fm)",
                prev_dist_to_forest - new_dist_to_forest,
                prev_dist_to_forest,
                new_dist_to_forest,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Internal: Haversine Distance
    # ------------------------------------------------------------------

    def _haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate Haversine distance between two WGS84 coordinates.

        Args:
            lat1: Latitude of point 1 (degrees).
            lon1: Longitude of point 1 (degrees).
            lat2: Latitude of point 2 (degrees).
            lon2: Longitude of point 2 (degrees).

        Returns:
            Distance in metres.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)

        a = (
            math.sin(dphi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

        return EARTH_RADIUS_M * c

    # ------------------------------------------------------------------
    # Internal: Provenance Hash
    # ------------------------------------------------------------------

    def _compute_result_hash(
        self, result: TemporalChangeResult
    ) -> str:
        """Compute SHA-256 provenance hash for a temporal analysis result.

        Args:
            result: The temporal change result to hash.

        Returns:
            SHA-256 hex digest.
        """
        hash_data = {
            "module_version": _MODULE_VERSION,
            "plot_id": result.plot_id,
            "is_consistent": result.is_consistent,
            "rapid_change_detected": result.rapid_change_detected,
            "boundary_change": (
                result.boundary_change.to_dict()
                if result.boundary_change else None
            ),
            "change_history_count": len(result.change_history),
            "issue_codes": sorted([i.code for i in result.issues]),
        }
        return _compute_hash(hash_data)


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

__all__ = [
    "TemporalConsistencyAnalyzer",
    "EUDR_CUTOFF_DATE",
    "SIGNIFICANT_AREA_CHANGE_PCT",
    "SIGNIFICANT_CENTROID_SHIFT_M",
    "DEFAULT_RAPID_CHANGE_WINDOW_DAYS",
    "RAPID_CHANGE_MAX_COUNT",
]
