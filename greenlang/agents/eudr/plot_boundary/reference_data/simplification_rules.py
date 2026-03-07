# -*- coding: utf-8 -*-
"""
Simplification Rules Reference Data - AGENT-EUDR-006

Provides simplification tolerance presets, EUDR submission limits, quality
thresholds, format-specific vertex limits, and multi-resolution level
definitions for the Plot Boundary Manager Agent. All data is deterministic,
immutable after module load, and derived from:

    - Douglas-Peucker and Visvalingam-Whyatt algorithm best practices
    - EUDR DDS submission specification (EU 2023/1115 implementing acts)
    - OGC format specifications (GeoJSON, KML, GML, Shapefile, GPX)
    - Production testing of 500,000+ boundary simplifications

Presets:
    Six named simplification presets from full_resolution (lossless) to
    minimal (coarsest generalisation), each with documented tolerance,
    expected vertex reduction, and target use case.

EUDR Submission Tolerance:
    EUDR-specific limits for Due Diligence Statement (DDS) submission
    including maximum vertices, file size, precision, and area deviation.

Quality Thresholds:
    Quality gates that simplification results must pass, including maximum
    area change, Hausdorff distance, vertex reduction ratio, and topology
    preservation.

Format Vertex Limits:
    Per-format constraints on maximum vertices, file size, and coordinate
    precision for GeoJSON, KML, Shapefile, EUDR XML, GPX, GML, WKT, WKB.

Multi-Resolution Levels:
    Four standard resolution levels for progressive loading and display:
    full, high, medium, low.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Simplification method constants
# ---------------------------------------------------------------------------

METHOD_DOUGLAS_PEUCKER: str = "douglas_peucker"
METHOD_VISVALINGAM_WHYATT: str = "visvalingam_whyatt"
METHOD_TOPOLOGY_PRESERVING: str = "topology_preserving"

VALID_SIMPLIFICATION_METHODS: Tuple[str, ...] = (
    METHOD_DOUGLAS_PEUCKER,
    METHOD_VISVALINGAM_WHYATT,
    METHOD_TOPOLOGY_PRESERVING,
)

# ---------------------------------------------------------------------------
# Simplification presets
# ---------------------------------------------------------------------------
# Named presets from zero simplification to aggressive generalisation.
# Tolerance values are in decimal degrees for WGS84 (EPSG:4326).
# Approximate linear equivalents at the equator:
#   1 degree ~= 111,320 metres
#   0.1 degree ~= 11,132 metres
#   0.01 degree ~= 1,113 metres
#   0.001 degree ~= 111 metres
#   0.0001 degree ~= 11.1 metres
#   0.00001 degree ~= 1.1 metres

SIMPLIFICATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "full_resolution": {
        "preset_id": "SP-000",
        "name": "Full Resolution",
        "description": (
            "No simplification applied. All original vertices are "
            "preserved. Use for archival, legal, and regulatory "
            "submissions where lossless geometry is required."
        ),
        "tolerance": 0.0,
        "tolerance_metres_at_equator": 0.0,
        "default_method": None,
        "expected_vertex_reduction_pct": 0.0,
        "target_use_case": "archival",
        "eudr_submission_safe": True,
    },
    "high_quality": {
        "preset_id": "SP-001",
        "name": "High Quality",
        "description": (
            "Minimal simplification preserving fine boundary detail. "
            "Removes only vertices that are within approximately 1 "
            "metre of the simplified line. Suitable for compliance "
            "reports and high-resolution mapping."
        ),
        "tolerance": 0.00001,
        "tolerance_metres_at_equator": 1.1,
        "default_method": METHOD_DOUGLAS_PEUCKER,
        "expected_vertex_reduction_pct": 15.0,
        "target_use_case": "compliance_reporting",
        "eudr_submission_safe": True,
    },
    "standard": {
        "preset_id": "SP-002",
        "name": "Standard",
        "description": (
            "Standard simplification balancing detail and efficiency. "
            "Approximately 11 metres tolerance at equator. Suitable "
            "for EUDR DDS submission, risk assessment, and operational "
            "mapping."
        ),
        "tolerance": 0.0001,
        "tolerance_metres_at_equator": 11.1,
        "default_method": METHOD_DOUGLAS_PEUCKER,
        "expected_vertex_reduction_pct": 40.0,
        "target_use_case": "eudr_submission",
        "eudr_submission_safe": True,
    },
    "medium": {
        "preset_id": "SP-003",
        "name": "Medium",
        "description": (
            "Moderate simplification for overview displays and "
            "aggregate analytics. Approximately 111 metres tolerance "
            "at equator. Preserves overall shape but loses fine detail."
        ),
        "tolerance": 0.001,
        "tolerance_metres_at_equator": 111.3,
        "default_method": METHOD_DOUGLAS_PEUCKER,
        "expected_vertex_reduction_pct": 65.0,
        "target_use_case": "overview_display",
        "eudr_submission_safe": False,
    },
    "low": {
        "preset_id": "SP-004",
        "name": "Low",
        "description": (
            "Aggressive simplification for small-scale maps and "
            "dashboard thumbnails. Approximately 1.1 km tolerance "
            "at equator. Only general shape is preserved."
        ),
        "tolerance": 0.01,
        "tolerance_metres_at_equator": 1113.2,
        "default_method": METHOD_VISVALINGAM_WHYATT,
        "expected_vertex_reduction_pct": 85.0,
        "target_use_case": "dashboard_thumbnail",
        "eudr_submission_safe": False,
    },
    "minimal": {
        "preset_id": "SP-005",
        "name": "Minimal",
        "description": (
            "Maximum simplification for ultra-small-scale maps, "
            "API summary responses, and icon generation. "
            "Approximately 11 km tolerance at equator."
        ),
        "tolerance": 0.1,
        "tolerance_metres_at_equator": 11132.0,
        "default_method": METHOD_VISVALINGAM_WHYATT,
        "expected_vertex_reduction_pct": 95.0,
        "target_use_case": "icon_generation",
        "eudr_submission_safe": False,
    },
}

# ---------------------------------------------------------------------------
# EUDR submission tolerance (DDS requirements)
# ---------------------------------------------------------------------------
# Specific constraints for EUDR Due Diligence Statement polygon submissions.
# Derived from EUDR implementing regulation and EU Information System specs.

EUDR_SUBMISSION_TOLERANCE: Dict[str, Any] = {
    "max_vertices_per_polygon": {
        "value": 10000,
        "description": (
            "Maximum number of vertices allowed per polygon in a "
            "Due Diligence Statement submission."
        ),
        "enforcement": "hard_limit",
        "action_if_exceeded": "simplify_to_fit",
    },
    "max_file_size_bytes": {
        "value": 10_485_760,
        "value_human": "10 MB",
        "description": (
            "Maximum file size for geolocation data in a single "
            "DDS submission."
        ),
        "enforcement": "hard_limit",
        "action_if_exceeded": "simplify_or_split",
    },
    "required_precision_decimal_places": {
        "value": 6,
        "description": (
            "Minimum number of decimal places for coordinate values "
            "in the submitted polygon data."
        ),
        "enforcement": "minimum",
        "approximate_precision_m": 0.11,
    },
    "area_deviation_tolerance": {
        "value": 0.01,
        "value_pct": 1.0,
        "description": (
            "Maximum allowed area deviation between the simplified "
            "submission polygon and the original boundary, expressed "
            "as a fraction of the original area."
        ),
        "enforcement": "quality_gate",
        "action_if_exceeded": "reduce_tolerance_and_retry",
    },
    "required_crs": {
        "value": "EPSG:4326",
        "description": "All submission polygons must use WGS84.",
        "enforcement": "hard_limit",
        "action_if_mismatch": "reproject_to_wgs84",
    },
    "coordinate_order": {
        "value": "[longitude, latitude]",
        "description": (
            "Coordinate pair order for submission format. Note: this "
            "is the GeoJSON convention, not the ISO 6709 convention."
        ),
        "enforcement": "format_requirement",
    },
    "ring_closure_required": {
        "value": True,
        "description": (
            "All polygon rings must be explicitly closed (first "
            "vertex equals last vertex) in the submission."
        ),
        "enforcement": "hard_limit",
    },
    "exterior_ring_ccw": {
        "value": True,
        "description": (
            "Exterior rings must be oriented counter-clockwise "
            "per GeoJSON RFC 7946."
        ),
        "enforcement": "format_requirement",
    },
}

# ---------------------------------------------------------------------------
# Quality thresholds for simplification output validation
# ---------------------------------------------------------------------------
# These define the quality gates that simplified boundaries must pass
# before being accepted.

QUALITY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "MAX_AREA_CHANGE_FRACTION": {
        "threshold_id": "QT-001",
        "description": (
            "Maximum allowed area change after simplification as a "
            "fraction of the original polygon area. Exceeding this "
            "indicates the simplification was too aggressive."
        ),
        "value": 0.01,
        "value_pct": 1.0,
        "applies_to": "all",
    },
    "MAX_AREA_CHANGE_EUDR": {
        "threshold_id": "QT-002",
        "description": (
            "Stricter area change limit for EUDR submission-grade "
            "simplification."
        ),
        "value": 0.005,
        "value_pct": 0.5,
        "applies_to": "eudr_submission",
    },
    "MAX_HAUSDORFF_DISTANCE_STANDARD_M": {
        "threshold_id": "QT-003",
        "description": (
            "Maximum Hausdorff distance in metres between the "
            "original and simplified boundary for standard presets."
        ),
        "value": 100.0,
        "unit": "metres",
        "applies_to": "standard",
    },
    "MAX_HAUSDORFF_DISTANCE_HIGH_M": {
        "threshold_id": "QT-004",
        "description": (
            "Maximum Hausdorff distance in metres between the "
            "original and simplified boundary for high-quality "
            "presets."
        ),
        "value": 10.0,
        "unit": "metres",
        "applies_to": "high_quality",
    },
    "MAX_HAUSDORFF_DISTANCE_EUDR_M": {
        "threshold_id": "QT-005",
        "description": (
            "Maximum Hausdorff distance for EUDR submission-grade "
            "simplification."
        ),
        "value": 25.0,
        "unit": "metres",
        "applies_to": "eudr_submission",
    },
    "MIN_VERTEX_REDUCTION_RATIO": {
        "threshold_id": "QT-006",
        "description": (
            "Minimum vertex count reduction to justify running "
            "simplification. If the reduction is below this ratio, "
            "the original geometry is preferred to avoid unnecessary "
            "processing and area deviation."
        ),
        "value": 0.10,
        "value_pct": 10.0,
        "applies_to": "all",
    },
    "TOPOLOGY_PRESERVATION": {
        "threshold_id": "QT-007",
        "description": (
            "After simplification, the simplified polygon must "
            "maintain the same topological properties: same number "
            "of holes, no new self-intersections, connected interior."
        ),
        "value": True,
        "applies_to": "all",
    },
    "MAX_VERTEX_COUNT_INCREASE": {
        "threshold_id": "QT-008",
        "description": (
            "Simplification must never increase the vertex count. "
            "If it does (e.g. due to intersection repair after "
            "simplification), the result is rejected."
        ),
        "value": 0,
        "applies_to": "all",
    },
    "MINIMUM_VERTICES_AFTER_SIMPLIFICATION": {
        "threshold_id": "QT-009",
        "description": (
            "Simplified polygon must retain at least 4 vertices "
            "(3 unique + closure) to remain a valid polygon."
        ),
        "value": 4,
        "applies_to": "all",
    },
    "CENTROID_DRIFT_MAX_M": {
        "threshold_id": "QT-010",
        "description": (
            "Maximum distance in metres that the centroid may "
            "shift after simplification. Large centroid drift "
            "indicates asymmetric generalisation."
        ),
        "value": 50.0,
        "unit": "metres",
        "applies_to": "standard",
    },
}

# ---------------------------------------------------------------------------
# Format-specific vertex and size limits
# ---------------------------------------------------------------------------
# Constraints for each supported export format, including maximum vertices,
# maximum file size, and coordinate precision.

FORMAT_VERTEX_LIMITS: Dict[str, Dict[str, Any]] = {
    "geojson": {
        "format_name": "GeoJSON",
        "specification": "RFC 7946",
        "max_vertices_per_polygon": None,
        "max_vertices_note": (
            "GeoJSON has no hard vertex limit; file size is the "
            "practical constraint."
        ),
        "max_file_size_bytes": 104_857_600,
        "max_file_size_human": "100 MB",
        "coordinate_precision_max": 15,
        "coordinate_precision_recommended": 8,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": True,
        "coordinate_order": "[longitude, latitude]",
    },
    "kml": {
        "format_name": "KML (Keyhole Markup Language)",
        "specification": "OGC 12-007r2",
        "max_vertices_per_polygon": 25000,
        "max_vertices_note": (
            "Google Earth performance degrades significantly above "
            "25,000 vertices per polygon."
        ),
        "max_file_size_bytes": 52_428_800,
        "max_file_size_human": "50 MB",
        "coordinate_precision_max": 15,
        "coordinate_precision_recommended": 7,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": True,
        "coordinate_order": "longitude,latitude,altitude",
    },
    "shapefile": {
        "format_name": "ESRI Shapefile",
        "specification": "ESRI Shapefile Technical Description (1998)",
        "max_vertices_per_polygon": None,
        "max_vertices_note": (
            "Shapefile .shp record size is limited to ~8 MB per shape. "
            "With 16 bytes per vertex (2D), this allows roughly "
            "500,000 vertices per polygon."
        ),
        "max_record_size_bytes": 8_388_608,
        "max_record_size_human": "8 MB",
        "max_file_size_bytes": 2_147_483_647,
        "max_file_size_human": "2 GB",
        "coordinate_precision_max": 11,
        "coordinate_precision_recommended": 8,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": False,
        "coordinate_order": "x (longitude), y (latitude)",
        "note": "MultiPolygon requires separate records in shapefile.",
    },
    "eudr_xml": {
        "format_name": "EUDR Submission XML",
        "specification": "EU 2023/1115 Implementing Regulation",
        "max_vertices_per_polygon": 10000,
        "max_vertices_note": (
            "EU Information System imposes a hard 10,000 vertex "
            "limit per polygon in DDS submissions."
        ),
        "max_file_size_bytes": 10_485_760,
        "max_file_size_human": "10 MB",
        "coordinate_precision_max": 8,
        "coordinate_precision_recommended": 6,
        "supports_3d": False,
        "supports_holes": True,
        "supports_multipolygon": False,
        "coordinate_order": "[longitude, latitude]",
        "note": (
            "EUDR XML is the mandatory submission format for the "
            "EU Information System DDS."
        ),
    },
    "gpx": {
        "format_name": "GPX (GPS Exchange Format)",
        "specification": "GPX 1.1",
        "max_vertices_per_polygon": 10000,
        "max_vertices_note": (
            "GPX tracks/routes with more than 10,000 points may "
            "cause compatibility issues with GPS devices."
        ),
        "max_file_size_bytes": 52_428_800,
        "max_file_size_human": "50 MB",
        "coordinate_precision_max": 10,
        "coordinate_precision_recommended": 7,
        "supports_3d": True,
        "supports_holes": False,
        "supports_multipolygon": False,
        "coordinate_order": "lat, lon",
        "note": "GPX does not natively support polygons; use track segments.",
    },
    "gml": {
        "format_name": "GML (Geography Markup Language)",
        "specification": "OGC 07-036r1 (GML 3.2.1)",
        "max_vertices_per_polygon": None,
        "max_vertices_note": (
            "GML has no hard vertex limit; XML parsing memory is the "
            "practical constraint."
        ),
        "max_file_size_bytes": 524_288_000,
        "max_file_size_human": "500 MB",
        "coordinate_precision_max": 15,
        "coordinate_precision_recommended": 8,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": True,
        "coordinate_order": "depends on CRS axis order",
    },
    "wkt": {
        "format_name": "Well-Known Text",
        "specification": "OGC 06-103r4",
        "max_vertices_per_polygon": None,
        "max_vertices_note": "No hard limit; string length is the constraint.",
        "max_file_size_bytes": None,
        "max_file_size_human": "unlimited (string)",
        "coordinate_precision_max": 15,
        "coordinate_precision_recommended": 8,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": True,
        "coordinate_order": "longitude latitude",
    },
    "wkb": {
        "format_name": "Well-Known Binary",
        "specification": "OGC 06-103r4",
        "max_vertices_per_polygon": None,
        "max_vertices_note": "No hard limit; binary blob size is the constraint.",
        "max_file_size_bytes": None,
        "max_file_size_human": "unlimited (binary)",
        "coordinate_precision_max": 15,
        "coordinate_precision_recommended": 15,
        "supports_3d": True,
        "supports_holes": True,
        "supports_multipolygon": True,
        "coordinate_order": "longitude latitude (IEEE 754 double)",
    },
}

# ---------------------------------------------------------------------------
# Multi-resolution levels
# ---------------------------------------------------------------------------
# Standard resolution tiers used for progressive loading, tile generation,
# and display at different zoom levels.

MULTI_RESOLUTION_LEVELS: Dict[str, Dict[str, Any]] = {
    "full": {
        "level_id": "MRL-001",
        "name": "Full Resolution",
        "preset": "full_resolution",
        "zoom_range": (14, 22),
        "description": (
            "Original boundary at full detail. Used for close-up "
            "views, editing, and compliance submissions."
        ),
        "priority": 1,
    },
    "high": {
        "level_id": "MRL-002",
        "name": "High Resolution",
        "preset": "high_quality",
        "zoom_range": (10, 14),
        "description": (
            "Minor simplification for regional-level views. "
            "~1m tolerance. Visually indistinguishable from full."
        ),
        "priority": 2,
    },
    "medium": {
        "level_id": "MRL-003",
        "name": "Medium Resolution",
        "preset": "standard",
        "zoom_range": (6, 10),
        "description": (
            "Standard simplification for national-level views. "
            "~11m tolerance. Preserves shape, loses fine detail."
        ),
        "priority": 3,
    },
    "low": {
        "level_id": "MRL-004",
        "name": "Low Resolution",
        "preset": "medium",
        "zoom_range": (0, 6),
        "description": (
            "Moderate simplification for continental/global views. "
            "~111m tolerance. General shape only."
        ),
        "priority": 4,
    },
}

# ---------------------------------------------------------------------------
# Simplification algorithm metadata
# ---------------------------------------------------------------------------
# Descriptions and characteristics of each supported simplification algorithm.

ALGORITHM_METADATA: Dict[str, Dict[str, Any]] = {
    METHOD_DOUGLAS_PEUCKER: {
        "name": "Douglas-Peucker",
        "full_name": "Ramer-Douglas-Peucker",
        "description": (
            "Iterative endpoint simplification. Recursively removes "
            "vertices whose perpendicular distance to the line segment "
            "between kept vertices is below the tolerance. Preserves "
            "critical shape features."
        ),
        "time_complexity": "O(n log n) average, O(n^2) worst case",
        "space_complexity": "O(n)",
        "preserves_topology": False,
        "tolerance_interpretation": "Maximum perpendicular distance from line.",
        "best_for": [
            "General-purpose simplification",
            "Polygons with smooth boundaries",
            "When vertex reduction is the priority",
        ],
        "weaknesses": [
            "May create self-intersections",
            "Does not consider area preservation",
            "Sensitive to spike vertices",
        ],
    },
    METHOD_VISVALINGAM_WHYATT: {
        "name": "Visvalingam-Whyatt",
        "full_name": "Visvalingam-Whyatt Area-Based Simplification",
        "description": (
            "Area-based simplification. Iteratively removes the vertex "
            "that creates the smallest triangle with its two neighbours. "
            "Produces aesthetically smoother results than Douglas-Peucker."
        ),
        "time_complexity": "O(n log n)",
        "space_complexity": "O(n)",
        "preserves_topology": False,
        "tolerance_interpretation": "Minimum effective area of triangle.",
        "best_for": [
            "Cartographic generalisation",
            "Smooth visual output",
            "Small-scale maps",
        ],
        "weaknesses": [
            "May create self-intersections",
            "Less control over maximum deviation",
            "Can over-simplify narrow protrusions",
        ],
    },
    METHOD_TOPOLOGY_PRESERVING: {
        "name": "Topology-Preserving",
        "full_name": "Topology-Preserving Douglas-Peucker",
        "description": (
            "Modified Douglas-Peucker that checks for topology "
            "violations (self-intersections, ring invalidation) at "
            "each simplification step and prevents them. Slower but "
            "guarantees valid output."
        ),
        "time_complexity": "O(n^2 log n)",
        "space_complexity": "O(n)",
        "preserves_topology": True,
        "tolerance_interpretation": "Maximum perpendicular distance from line.",
        "best_for": [
            "Regulatory submissions (EUDR)",
            "Overlap-sensitive contexts",
            "Multi-boundary simplification",
        ],
        "weaknesses": [
            "Slower than non-topology-preserving methods",
            "May achieve less vertex reduction",
            "Higher memory usage",
        ],
    },
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_preset(name: str) -> Optional[Dict[str, Any]]:
    """Return a simplification preset by name.

    Args:
        name: Preset name (e.g. 'standard', 'high_quality', 'minimal').

    Returns:
        Preset dictionary, or None if not found.

    Example:
        >>> preset = get_preset("standard")
        >>> preset["tolerance"]
        0.0001
    """
    return SIMPLIFICATION_PRESETS.get(name.lower())


def get_eudr_tolerance() -> Dict[str, Any]:
    """Return the full EUDR submission tolerance specification.

    Returns:
        Dictionary with all EUDR submission limits.

    Example:
        >>> eudr = get_eudr_tolerance()
        >>> eudr["max_vertices_per_polygon"]["value"]
        10000
    """
    return dict(EUDR_SUBMISSION_TOLERANCE)


def get_format_limits(format_name: str) -> Optional[Dict[str, Any]]:
    """Return format-specific vertex and size limits.

    Args:
        format_name: Format identifier (e.g. 'geojson', 'kml', 'eudr_xml').

    Returns:
        Format limits dictionary, or None if not found.

    Example:
        >>> limits = get_format_limits("eudr_xml")
        >>> limits["max_vertices_per_polygon"]
        10000
    """
    return FORMAT_VERTEX_LIMITS.get(format_name.lower())


def get_quality_threshold(threshold_key: str) -> Optional[Dict[str, Any]]:
    """Return a quality threshold by key.

    Args:
        threshold_key: Threshold key (e.g. 'MAX_AREA_CHANGE_FRACTION').

    Returns:
        Threshold dictionary, or None if not found.
    """
    return QUALITY_THRESHOLDS.get(threshold_key)


def get_resolution_level(level: str) -> Optional[Dict[str, Any]]:
    """Return a multi-resolution level definition.

    Args:
        level: Level name (e.g. 'full', 'high', 'medium', 'low').

    Returns:
        Resolution level dictionary, or None if not found.
    """
    return MULTI_RESOLUTION_LEVELS.get(level.lower())


def get_algorithm_metadata(method: str) -> Optional[Dict[str, Any]]:
    """Return metadata for a simplification algorithm.

    Args:
        method: Algorithm identifier (e.g. 'douglas_peucker').

    Returns:
        Algorithm metadata dictionary, or None if not found.
    """
    return ALGORITHM_METADATA.get(method.lower())


def get_all_preset_names() -> Tuple[str, ...]:
    """Return a tuple of all simplification preset names.

    Returns:
        Tuple of preset names in definition order.
    """
    return tuple(SIMPLIFICATION_PRESETS.keys())


def get_all_format_names() -> Tuple[str, ...]:
    """Return a tuple of all supported export format names.

    Returns:
        Tuple of format names in definition order.
    """
    return tuple(FORMAT_VERTEX_LIMITS.keys())


def get_eudr_safe_presets() -> Tuple[str, ...]:
    """Return preset names that are safe for EUDR DDS submission.

    Returns:
        Tuple of preset names where ``eudr_submission_safe`` is True.
    """
    return tuple(
        name
        for name, preset in SIMPLIFICATION_PRESETS.items()
        if preset.get("eudr_submission_safe", False)
    )


def max_vertices_for_format(format_name: str) -> Optional[int]:
    """Return the maximum vertex count for a format, or None if unlimited.

    Args:
        format_name: Format identifier.

    Returns:
        Maximum vertex count, or None if no hard limit.
    """
    limits = FORMAT_VERTEX_LIMITS.get(format_name.lower())
    if limits is None:
        return None
    return limits.get("max_vertices_per_polygon")


# ---------------------------------------------------------------------------
# Module-level counts for introspection
# ---------------------------------------------------------------------------

TOTAL_PRESETS: int = len(SIMPLIFICATION_PRESETS)
TOTAL_QUALITY_THRESHOLDS: int = len(QUALITY_THRESHOLDS)
TOTAL_FORMAT_LIMITS: int = len(FORMAT_VERTEX_LIMITS)
TOTAL_RESOLUTION_LEVELS: int = len(MULTI_RESOLUTION_LEVELS)
TOTAL_ALGORITHMS: int = len(ALGORITHM_METADATA)
