# -*- coding: utf-8 -*-
"""
Boundary Versioner Engine - AGENT-EUDR-006: Plot Boundary Manager (Engine 5)

Immutable boundary version management engine providing temporal point-in-time
queries, version diffing, lineage tracking, rollback with audit preservation,
and EUDR Article 31 retention compliance checking.

Zero-Hallucination Guarantees:
    - All version numbers are deterministic sequential integers.
    - Area differences are computed via float arithmetic only.
    - SHA-256 provenance hashes on all version records.
    - Binary search for temporal point-in-time queries.
    - No ML/LLM used for any computation.

Version Operations:
    create_version:     Create a new immutable boundary version.
    get_version:        Retrieve a specific version by number.
    get_latest:         Retrieve the current active version.
    get_history:        Retrieve all versions for a plot.
    get_at_date:        Temporal point-in-time query (binary search).
    get_diff:           Compute geometric diff between two versions.
    get_lineage:        Full version lineage from v1 to current.
    rollback:           Restore a prior version (creates new version).
    check_retention:    EUDR Article 31 retention compliance check.
    batch_get_at_date:  Batch temporal query across multiple plots.

Regulatory References:
    - EUDR Article 9: Geolocation boundary accuracy requirements.
    - EUDR Article 31: Record-keeping and 5-year data retention.
    - EUDR Cutoff Date: 31 December 2020 (commonly queried).

Performance Targets:
    - Version creation: <10ms per version.
    - Temporal point query: O(log n) via binary search.
    - Version diff: <50ms for standard polygon pairs.
    - Retention check: <5ms per plot.
    - Batch temporal query (1000 plots): <500ms.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-006 (Engine 5: Boundary Version Management)
Agent ID: GL-EUDR-PBM-006
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.plot_boundary.config import (
    PlotBoundaryConfig,
    get_config,
)
from greenlang.agents.eudr.plot_boundary.models import (
    BoundaryStatus,
    BoundaryVersion,
    Coordinate,
    PlotBoundary,
    RetentionStatus,
    Ring,
    VersionChangeReason,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id() -> str:
    """Generate a unique identifier using UUID4."""
    return str(uuid.uuid4())

def _polygon_area_shoelace(exterior: Ring) -> float:
    """Compute approximate polygon area using the shoelace formula.

    Uses the spherical excess approximation for WGS84 coordinates.
    Returns area in hectares. This is a fast approximate calculation
    for area diffing purposes; for precise geodesic area, use
    the Karney algorithm in polygon_manager.py.

    Args:
        exterior: Exterior ring of the polygon.

    Returns:
        Approximate area in hectares.
    """
    if len(exterior) < 4:
        return 0.0

    # Shoelace formula in lat/lon (approximate, equirectangular)
    n = len(exterior)
    area_deg2 = 0.0
    for i in range(n):
        j = (i + 1) % n
        lat_i = exterior[i].lat
        lon_i = exterior[i].lon
        lat_j = exterior[j].lat
        lon_j = exterior[j].lon
        area_deg2 += lon_i * lat_j
        area_deg2 -= lon_j * lat_i

    area_deg2 = abs(area_deg2) / 2.0

    # Convert square degrees to hectares
    # At equator: 1 degree lat ~ 111,320m, 1 degree lon ~ 111,320m
    # Rough approximation using centroid latitude for cos correction
    centroid_lat = sum(c.lat for c in exterior) / n
    cos_lat = math.cos(math.radians(centroid_lat))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * cos_lat

    area_m2 = area_deg2 * meters_per_deg_lat * meters_per_deg_lon
    return area_m2 / 10_000.0  # m2 to hectares

def _count_vertices(boundary: PlotBoundary) -> int:
    """Count total vertices in a boundary including holes.

    Args:
        boundary: Plot boundary to count vertices for.

    Returns:
        Total vertex count.
    """
    total = len(boundary.exterior)
    for hole in boundary.holes:
        total += len(hole)
    return total

# =============================================================================
# BoundaryVersioner
# =============================================================================

class BoundaryVersioner:
    """Immutable boundary version management engine.

    Maintains an in-memory store of boundary versions keyed by plot_id,
    providing temporal point-in-time queries via binary search,
    version diffing, lineage tracking, rollback with full audit trail
    preservation, and EUDR Article 31 retention compliance checking.

    All versions are immutable once created. Rollback operations create
    new versions rather than deleting or modifying existing records.

    Attributes:
        _config: Engine configuration.
        _versions: In-memory version store keyed by plot_id.

    Example:
        >>> from greenlang.agents.eudr.plot_boundary.config import PlotBoundaryConfig
        >>> config = PlotBoundaryConfig()
        >>> versioner = BoundaryVersioner(config)
        >>> version = versioner.create_version(
        ...     plot_id="PLOT-001",
        ...     boundary=some_boundary,
        ...     change_reason=VersionChangeReason.INITIAL,
        ...     changed_by="system",
        ... )
        >>> assert version.version_number == 1
    """

    def __init__(self, config: Optional[PlotBoundaryConfig] = None) -> None:
        """Initialize BoundaryVersioner.

        Args:
            config: Engine configuration. If None, uses the singleton
                configuration from environment variables.
        """
        self._config = config or get_config()
        self._versions: Dict[str, List[BoundaryVersion]] = {}
        logger.info(
            "BoundaryVersioner initialized: retention=%dy, "
            "module_version=%s",
            self._config.version_retention_years,
            _MODULE_VERSION,
        )

    # ------------------------------------------------------------------
    # Public API - Version Creation
    # ------------------------------------------------------------------

    def create_version(
        self,
        plot_id: str,
        boundary: PlotBoundary,
        change_reason: VersionChangeReason,
        changed_by: str,
    ) -> BoundaryVersion:
        """Create a new immutable boundary version.

        Computes the next sequential version number, calculates area
        differences from the previous version (if any), generates a
        SHA-256 provenance hash, marks the previous version as
        superseded, and stores the new version as ACTIVE.

        Args:
            plot_id: Unique identifier for the plot.
            boundary: Complete boundary data for this version.
            change_reason: Reason for creating this version.
            changed_by: User or system identifier.

        Returns:
            The newly created BoundaryVersion.

        Raises:
            ValueError: If plot_id or changed_by is empty.
        """
        start_time = time.monotonic()

        if not plot_id or not plot_id.strip():
            raise ValueError("plot_id must be non-empty")
        if not changed_by or not changed_by.strip():
            raise ValueError("changed_by must be non-empty")

        # Compute next version number
        next_version = self._compute_next_version(plot_id)

        # Calculate area
        current_area = _polygon_area_shoelace(boundary.exterior)

        # Calculate area diff from previous version
        area_diff_ha = 0.0
        area_diff_pct = 0.0
        previous = self._get_previous_version(plot_id)
        if previous is not None:
            area_diff_ha = current_area - previous.area_hectares
            if previous.area_hectares > 0.0:
                area_diff_pct = (
                    area_diff_ha / previous.area_hectares
                ) * 100.0

        # Generate provenance hash
        provenance_hash = self._compute_version_hash(
            boundary, next_version,
        )

        now = utcnow()

        # Mark previous version as superseded
        if previous is not None:
            self._supersede_previous(plot_id, now)

        # Create version record
        version = BoundaryVersion(
            version_id=_generate_id(),
            plot_id=plot_id,
            version_number=next_version,
            boundary=boundary,
            change_reason=change_reason,
            changed_by=changed_by,
            area_hectares=current_area,
            area_diff_hectares=area_diff_ha,
            area_diff_pct=area_diff_pct,
            status=BoundaryStatus.ACTIVE,
            provenance_hash=provenance_hash,
            created_at=now,
            valid_from=now,
            valid_until=None,
        )

        # Store immutable version record
        if plot_id not in self._versions:
            self._versions[plot_id] = []
        self._versions[plot_id].append(version)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Created version v%d for plot %s: "
            "area=%.4fha, diff=%.4fha (%.2f%%), "
            "reason=%s, by=%s, hash=%s, elapsed=%.1fms",
            next_version,
            plot_id,
            current_area,
            area_diff_ha,
            area_diff_pct,
            change_reason.value,
            changed_by,
            provenance_hash[:16],
            elapsed_ms,
        )

        return version

    # ------------------------------------------------------------------
    # Public API - Version Retrieval
    # ------------------------------------------------------------------

    def get_version(
        self,
        plot_id: str,
        version_number: int,
    ) -> BoundaryVersion:
        """Get a specific version by plot_id and version number.

        Args:
            plot_id: Unique identifier for the plot.
            version_number: Sequential version number to retrieve.

        Returns:
            The requested BoundaryVersion.

        Raises:
            ValueError: If plot_id is empty or version_number < 1.
            KeyError: If plot_id has no versions or version not found.
        """
        if not plot_id or not plot_id.strip():
            raise ValueError("plot_id must be non-empty")
        if version_number < 1:
            raise ValueError("version_number must be >= 1")

        versions = self._versions.get(plot_id)
        if not versions:
            raise KeyError(f"No versions found for plot_id={plot_id}")

        for v in versions:
            if v.version_number == version_number:
                return v

        raise KeyError(
            f"Version {version_number} not found for plot_id={plot_id}"
        )

    def get_latest(self, plot_id: str) -> BoundaryVersion:
        """Get the current (latest) active version for a plot.

        Args:
            plot_id: Unique identifier for the plot.

        Returns:
            The latest BoundaryVersion (highest version number).

        Raises:
            KeyError: If plot_id has no versions.
        """
        versions = self._versions.get(plot_id)
        if not versions:
            raise KeyError(f"No versions found for plot_id={plot_id}")

        return versions[-1]

    def get_history(self, plot_id: str) -> List[BoundaryVersion]:
        """Get all versions for a plot, ordered by version number.

        Args:
            plot_id: Unique identifier for the plot.

        Returns:
            List of all BoundaryVersion objects, oldest first.
        """
        versions = self._versions.get(plot_id)
        if not versions:
            return []

        return list(versions)

    def get_at_date(
        self,
        plot_id: str,
        target_date: datetime,
    ) -> BoundaryVersion:
        """Get the version that was active at a specific date.

        Uses binary search through version timestamps for O(log n)
        performance. The EUDR cutoff date (2020-12-31) is a commonly
        queried date.

        Args:
            plot_id: Unique identifier for the plot.
            target_date: UTC datetime to query. If timezone-naive,
                treated as UTC.

        Returns:
            The BoundaryVersion that was active at the target date.

        Raises:
            KeyError: If plot_id has no versions.
            ValueError: If target_date is before the first version.
        """
        versions = self._versions.get(plot_id)
        if not versions:
            raise KeyError(f"No versions found for plot_id={plot_id}")

        # Ensure target_date is timezone-aware
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=timezone.utc)

        # Check if target is before first version
        if target_date < versions[0].valid_from:
            raise ValueError(
                f"target_date {target_date.isoformat()} is before "
                f"first version created_at "
                f"{versions[0].valid_from.isoformat()}"
            )

        # Binary search for the version active at target_date
        idx = self._binary_search_date(versions, target_date)
        return versions[idx]

    # ------------------------------------------------------------------
    # Public API - Version Diffing
    # ------------------------------------------------------------------

    def get_diff(
        self,
        plot_id: str,
        version_a: int,
        version_b: int,
    ) -> Dict[str, Any]:
        """Compute geometric diff between two versions.

        Calculates added area (in version_b but not version_a),
        removed area (in version_a but not version_b), and unchanged
        area (intersection) using approximate polygon operations.

        Args:
            plot_id: Unique identifier for the plot.
            version_a: First version number.
            version_b: Second version number.

        Returns:
            Dictionary with diff statistics:
                - version_a: int
                - version_b: int
                - area_a_hectares: float
                - area_b_hectares: float
                - area_change_hectares: float
                - area_change_pct: float
                - vertices_a: int
                - vertices_b: int
                - vertex_change: int
                - added_area_hectares: float (approximate)
                - removed_area_hectares: float (approximate)
                - unchanged_area_hectares: float (approximate)
                - provenance_hash: str

        Raises:
            KeyError: If either version is not found.
        """
        start_time = time.monotonic()

        va = self.get_version(plot_id, version_a)
        vb = self.get_version(plot_id, version_b)

        area_a = va.area_hectares
        area_b = vb.area_hectares
        area_change = area_b - area_a
        area_change_pct = 0.0
        if area_a > 0.0:
            area_change_pct = (area_change / area_a) * 100.0

        vertices_a = _count_vertices(va.boundary)
        vertices_b = _count_vertices(vb.boundary)

        # Approximate intersection using bounding box overlap
        added, removed, unchanged = self._approximate_area_diff(
            va.boundary, vb.boundary, area_a, area_b,
        )

        diff_data = {
            "version_a": version_a,
            "version_b": version_b,
            "area_a_hectares": round(area_a, 6),
            "area_b_hectares": round(area_b, 6),
            "area_change_hectares": round(area_change, 6),
            "area_change_pct": round(area_change_pct, 4),
            "vertices_a": vertices_a,
            "vertices_b": vertices_b,
            "vertex_change": vertices_b - vertices_a,
            "added_area_hectares": round(added, 6),
            "removed_area_hectares": round(removed, 6),
            "unchanged_area_hectares": round(unchanged, 6),
        }

        diff_data["provenance_hash"] = _compute_hash(diff_data)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Computed diff for plot %s v%d->v%d: "
            "area_change=%.4fha (%.2f%%), "
            "added=%.4fha, removed=%.4fha, unchanged=%.4fha, "
            "elapsed=%.1fms",
            plot_id,
            version_a,
            version_b,
            area_change,
            area_change_pct,
            added,
            removed,
            unchanged,
            elapsed_ms,
        )

        return diff_data

    # ------------------------------------------------------------------
    # Public API - Lineage
    # ------------------------------------------------------------------

    def get_lineage(self, plot_id: str) -> List[BoundaryVersion]:
        """Get full lineage from v1 to current version.

        Returns all versions in chronological order with their
        change reasons, timestamps, and area changes.

        Args:
            plot_id: Unique identifier for the plot.

        Returns:
            Ordered list from v1 to the latest version.
        """
        return self.get_history(plot_id)

    # ------------------------------------------------------------------
    # Public API - Rollback
    # ------------------------------------------------------------------

    def rollback(
        self,
        plot_id: str,
        target_version: int,
    ) -> BoundaryVersion:
        """Rollback to a previous version by creating a new version.

        Creates a new version that restores the target version's
        boundary data. The change reason is set to CORRECTION.
        No versions are deleted -- the audit trail is fully preserved.

        Args:
            plot_id: Unique identifier for the plot.
            target_version: Version number to restore.

        Returns:
            The newly created BoundaryVersion (rollback).

        Raises:
            KeyError: If target_version is not found.
            ValueError: If target_version is the current version.
        """
        target = self.get_version(plot_id, target_version)
        latest = self.get_latest(plot_id)

        if target.version_number == latest.version_number:
            raise ValueError(
                f"Cannot rollback to current version "
                f"v{target_version} for plot {plot_id}"
            )

        logger.info(
            "Rolling back plot %s from v%d to v%d boundary",
            plot_id,
            latest.version_number,
            target_version,
        )

        # Create new version with target's boundary
        rollback_version = self.create_version(
            plot_id=plot_id,
            boundary=target.boundary,
            change_reason=VersionChangeReason.CORRECTION,
            changed_by=f"rollback_to_v{target_version}",
        )

        logger.info(
            "Rollback complete for plot %s: "
            "restored v%d boundary as v%d",
            plot_id,
            target_version,
            rollback_version.version_number,
        )

        return rollback_version

    # ------------------------------------------------------------------
    # Public API - Retention Compliance
    # ------------------------------------------------------------------

    def check_retention(self, plot_id: str) -> Dict[str, Any]:
        """Check EUDR Article 31 retention compliance for a plot.

        Analyzes all versions for the plot and determines whether
        they comply with the 5-year retention requirement. Flags
        versions approaching expiry (within 6 months).

        Args:
            plot_id: Unique identifier for the plot.

        Returns:
            Dictionary with retention status report:
                - plot_id: str
                - total_versions: int
                - retention_years: int
                - versions_compliant: int
                - versions_approaching_expiry: int
                - versions_expired: int
                - versions_permanent: int
                - overall_status: str (RetentionStatus value)
                - version_details: list of dicts
                - checked_at: str (ISO timestamp)
                - provenance_hash: str
        """
        start_time = time.monotonic()
        now = utcnow()
        retention_years = self._config.version_retention_years

        versions = self.get_history(plot_id)

        compliant = 0
        approaching = 0
        expired = 0
        permanent = 0
        version_details: List[Dict[str, Any]] = []

        for v in versions:
            status = self._classify_retention(
                v.created_at, now, retention_years,
            )

            if status == RetentionStatus.COMPLIANT:
                compliant += 1
            elif status == RetentionStatus.APPROACHING_EXPIRY:
                approaching += 1
            elif status == RetentionStatus.EXPIRED:
                expired += 1
            elif status == RetentionStatus.PERMANENT:
                permanent += 1

            version_details.append({
                "version_number": v.version_number,
                "created_at": v.created_at.isoformat(),
                "retention_status": status.value,
                "days_until_expiry": self._days_until_expiry(
                    v.created_at, now, retention_years,
                ),
            })

        # Overall status
        if expired > 0:
            overall = RetentionStatus.EXPIRED
        elif approaching > 0:
            overall = RetentionStatus.APPROACHING_EXPIRY
        else:
            overall = RetentionStatus.COMPLIANT

        report = {
            "plot_id": plot_id,
            "total_versions": len(versions),
            "retention_years": retention_years,
            "versions_compliant": compliant,
            "versions_approaching_expiry": approaching,
            "versions_expired": expired,
            "versions_permanent": permanent,
            "overall_status": overall.value,
            "version_details": version_details,
            "checked_at": now.isoformat(),
        }
        report["provenance_hash"] = _compute_hash(report)

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Retention check for plot %s: "
            "total=%d, compliant=%d, approaching=%d, "
            "expired=%d, overall=%s, elapsed=%.1fms",
            plot_id,
            len(versions),
            compliant,
            approaching,
            expired,
            overall.value,
            elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API - Batch Operations
    # ------------------------------------------------------------------

    def batch_get_at_date(
        self,
        plot_ids: List[str],
        target_date: datetime,
    ) -> Dict[str, BoundaryVersion]:
        """Batch temporal query across multiple plots.

        Retrieves the version that was active at the target date
        for each plot_id. Plots with no versions or where the
        target_date precedes the first version are silently skipped.

        Args:
            plot_ids: List of plot identifiers to query.
            target_date: UTC datetime to query.

        Returns:
            Dictionary mapping plot_id to BoundaryVersion for
            all plots that have a version at the target date.
        """
        start_time = time.monotonic()
        results: Dict[str, BoundaryVersion] = {}

        for pid in plot_ids:
            try:
                version = self.get_at_date(pid, target_date)
                results[pid] = version
            except (KeyError, ValueError):
                # Skip plots with no versions or date out of range
                logger.debug(
                    "Skipping plot %s for date %s: no matching version",
                    pid,
                    target_date.isoformat(),
                )
                continue

        elapsed_ms = (time.monotonic() - start_time) * 1000.0
        logger.info(
            "Batch temporal query: %d/%d plots matched at %s, "
            "elapsed=%.1fms",
            len(results),
            len(plot_ids),
            target_date.isoformat(),
            elapsed_ms,
        )

        return results

    # ------------------------------------------------------------------
    # Public API - Utility
    # ------------------------------------------------------------------

    @property
    def plot_count(self) -> int:
        """Return the number of plots with at least one version."""
        return len(self._versions)

    @property
    def total_version_count(self) -> int:
        """Return the total number of versions across all plots."""
        return sum(len(vs) for vs in self._versions.values())

    def has_versions(self, plot_id: str) -> bool:
        """Check if a plot has any versions.

        Args:
            plot_id: Plot identifier.

        Returns:
            True if the plot has at least one version.
        """
        return plot_id in self._versions and len(self._versions[plot_id]) > 0

    def version_count(self, plot_id: str) -> int:
        """Return the number of versions for a specific plot.

        Args:
            plot_id: Plot identifier.

        Returns:
            Number of versions, or 0 if no versions exist.
        """
        versions = self._versions.get(plot_id)
        return len(versions) if versions else 0

    def clear(self) -> None:
        """Clear all version data. Intended for testing only."""
        self._versions.clear()
        logger.info("BoundaryVersioner storage cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_next_version(self, plot_id: str) -> int:
        """Compute the next sequential version number for a plot.

        Args:
            plot_id: Plot identifier.

        Returns:
            Next version number (1 for first version).
        """
        versions = self._versions.get(plot_id)
        if not versions:
            return 1
        return versions[-1].version_number + 1

    def _get_previous_version(
        self,
        plot_id: str,
    ) -> Optional[BoundaryVersion]:
        """Get the most recent version for a plot, or None.

        Args:
            plot_id: Plot identifier.

        Returns:
            Latest BoundaryVersion or None if no versions exist.
        """
        versions = self._versions.get(plot_id)
        if not versions:
            return None
        return versions[-1]

    def _supersede_previous(
        self,
        plot_id: str,
        superseded_at: datetime,
    ) -> None:
        """Mark the current latest version as superseded.

        Sets the valid_until timestamp and changes status to
        SUPERSEDED for the current active version.

        Args:
            plot_id: Plot identifier.
            superseded_at: Timestamp when superseded.
        """
        versions = self._versions.get(plot_id)
        if not versions:
            return

        latest = versions[-1]
        if latest.status == BoundaryStatus.ACTIVE:
            # Create updated version with superseded status
            # Since Pydantic models are immutable by convention,
            # we replace the last entry
            updated = latest.model_copy(update={
                "status": BoundaryStatus.SUPERSEDED,
                "valid_until": superseded_at,
            })
            versions[-1] = updated

    def _compute_version_hash(
        self,
        boundary: PlotBoundary,
        version_number: int,
    ) -> str:
        """Compute SHA-256 provenance hash for a boundary version.

        Includes the boundary geometry, plot_id, version number,
        and module version for reproducibility.

        Args:
            boundary: Plot boundary data.
            version_number: Sequential version number.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        hash_input = {
            "plot_id": boundary.plot_id,
            "version_number": version_number,
            "module_version": _MODULE_VERSION,
            "exterior_vertex_count": len(boundary.exterior),
            "hole_count": len(boundary.holes),
            "exterior_coords": [
                {"lat": c.lat, "lon": c.lon}
                for c in boundary.exterior
            ],
            "commodity": (
                boundary.commodity.value
                if boundary.commodity else None
            ),
            "country_code": boundary.country_code,
        }
        return _compute_hash(hash_input)

    def _binary_search_date(
        self,
        versions: List[BoundaryVersion],
        target_date: datetime,
    ) -> int:
        """Binary search for the version active at a given date.

        Finds the latest version whose valid_from is <= target_date.
        Versions are assumed to be sorted by valid_from (ascending).

        Args:
            versions: Sorted list of BoundaryVersion objects.
            target_date: Target UTC datetime.

        Returns:
            Index of the version active at target_date.
        """
        low = 0
        high = len(versions) - 1
        result_idx = 0

        while low <= high:
            mid = (low + high) // 2
            mid_time = versions[mid].valid_from

            # Ensure timezone-aware comparison
            if mid_time.tzinfo is None:
                mid_time = mid_time.replace(tzinfo=timezone.utc)

            if mid_time <= target_date:
                result_idx = mid
                low = mid + 1
            else:
                high = mid - 1

        return result_idx

    def _approximate_area_diff(
        self,
        boundary_a: PlotBoundary,
        boundary_b: PlotBoundary,
        area_a: float,
        area_b: float,
    ) -> Tuple[float, float, float]:
        """Approximate area diff using bounding box intersection.

        Computes approximate added, removed, and unchanged areas
        using bounding box overlap as a proxy for geometric
        intersection. This is a fast O(1) approximation.

        Args:
            boundary_a: First boundary.
            boundary_b: Second boundary.
            area_a: Area of first boundary in hectares.
            area_b: Area of second boundary in hectares.

        Returns:
            Tuple of (added_ha, removed_ha, unchanged_ha).
        """
        # Compute bounding boxes
        bbox_a = self._bounding_box(boundary_a.exterior)
        bbox_b = self._bounding_box(boundary_b.exterior)

        # Compute intersection of bounding boxes
        overlap_fraction = self._bbox_overlap_fraction(bbox_a, bbox_b)

        # Estimate intersection area
        min_area = min(area_a, area_b)
        intersection_area = min_area * overlap_fraction

        # Compute approximate diff areas
        unchanged = intersection_area
        removed = max(0.0, area_a - intersection_area)
        added = max(0.0, area_b - intersection_area)

        return (added, removed, unchanged)

    def _bounding_box(
        self,
        ring: Ring,
    ) -> Tuple[float, float, float, float]:
        """Compute the bounding box of a ring.

        Args:
            ring: List of Coordinate objects.

        Returns:
            Tuple of (min_lat, min_lon, max_lat, max_lon).
        """
        if not ring:
            return (0.0, 0.0, 0.0, 0.0)

        lats = [c.lat for c in ring]
        lons = [c.lon for c in ring]
        return (min(lats), min(lons), max(lats), max(lons))

    def _bbox_overlap_fraction(
        self,
        bbox_a: Tuple[float, float, float, float],
        bbox_b: Tuple[float, float, float, float],
    ) -> float:
        """Compute the overlap fraction between two bounding boxes.

        Returns the ratio of the intersection area to the smaller
        bounding box area. Returns 0.0 if no overlap.

        Args:
            bbox_a: First bounding box (min_lat, min_lon, max_lat, max_lon).
            bbox_b: Second bounding box.

        Returns:
            Overlap fraction between 0.0 and 1.0.
        """
        # Intersection bounds
        int_min_lat = max(bbox_a[0], bbox_b[0])
        int_min_lon = max(bbox_a[1], bbox_b[1])
        int_max_lat = min(bbox_a[2], bbox_b[2])
        int_max_lon = min(bbox_a[3], bbox_b[3])

        if int_min_lat >= int_max_lat or int_min_lon >= int_max_lon:
            return 0.0

        int_area = (int_max_lat - int_min_lat) * (int_max_lon - int_min_lon)

        # Area of each bounding box
        area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
        area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

        min_bbox_area = min(area_a, area_b)
        if min_bbox_area <= 0.0:
            return 0.0

        return min(1.0, int_area / min_bbox_area)

    def _classify_retention(
        self,
        created_at: datetime,
        now: datetime,
        retention_years: int,
    ) -> RetentionStatus:
        """Classify the retention status of a version.

        Args:
            created_at: When the version was created.
            now: Current UTC datetime.
            retention_years: Required retention period.

        Returns:
            RetentionStatus classification.
        """
        # Ensure timezone-aware
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        expiry_date = created_at + timedelta(days=retention_years * 365)
        approaching_date = expiry_date - timedelta(days=180)

        if now >= expiry_date:
            return RetentionStatus.EXPIRED
        elif now >= approaching_date:
            return RetentionStatus.APPROACHING_EXPIRY
        else:
            return RetentionStatus.COMPLIANT

    def _days_until_expiry(
        self,
        created_at: datetime,
        now: datetime,
        retention_years: int,
    ) -> int:
        """Calculate days until a version's retention expires.

        Args:
            created_at: When the version was created.
            now: Current UTC datetime.
            retention_years: Required retention period.

        Returns:
            Days until expiry (negative if already expired).
        """
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)

        expiry_date = created_at + timedelta(days=retention_years * 365)
        delta = expiry_date - now
        return delta.days

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        return (
            f"BoundaryVersioner(plots={self.plot_count}, "
            f"total_versions={self.total_version_count})"
        )

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BoundaryVersioner",
]
