# -*- coding: utf-8 -*-
"""
Boundary Standards Reference Data - AGENT-EUDR-006

Provides OGC Simple Features, ISO 19107, and EUDR Article 9 boundary rules,
validation thresholds, commodity area ranges, and country CRS defaults for the
Plot Boundary Manager Agent. All data is deterministic, immutable after module
load, and derived directly from:

    - OGC 06-103r4 (OpenGIS Implementation Standard for Geographic Information
      -- Simple Feature Access -- Part 1: Common Architecture)
    - ISO 19107:2019 Geographic Information -- Spatial Schema
    - Regulation (EU) 2023/1115 (EUDR) Articles 2, 9, 10, 31
    - FAO commodity area statistics and EUDR impact assessments

Standards Referenced:
    OGC SFA 06-103r4   - Simple Feature geometry validity rules
    ISO 19107:2019      - GM_Polygon and GM_SurfaceBoundary
    EU 2023/1115 Art. 9 - Geolocation requirements for due diligence
    EU 2023/1115 Art. 31 - Record retention (5 years)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
Status: Production Ready
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# OGC Simple Features validity rules (OGC 06-103r4)
# ---------------------------------------------------------------------------
# These rules define geometric validity for Polygon and MultiPolygon types
# under the OGC Simple Features specification, which is the canonical
# standard for geospatial data interchange.

OGC_SIMPLE_FEATURES_RULES: Dict[str, Dict[str, Any]] = {
    "RING_CLOSURE": {
        "rule_id": "OGC-SFA-001",
        "description": (
            "Every ring (exterior or interior) must be closed: the first "
            "and last coordinate pair must be identical."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": True,
        "repair_method": "Append first vertex as last vertex.",
    },
    "RING_ORIENTATION_EXTERIOR": {
        "rule_id": "OGC-SFA-002",
        "description": (
            "The exterior ring of a polygon must be oriented "
            "counter-clockwise (CCW) when viewed from above."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "warning",
        "auto_repairable": True,
        "repair_method": "Reverse vertex order of exterior ring.",
    },
    "RING_ORIENTATION_INTERIOR": {
        "rule_id": "OGC-SFA-003",
        "description": (
            "Interior rings (holes) must be oriented clockwise (CW) "
            "when viewed from above."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "warning",
        "auto_repairable": True,
        "repair_method": "Reverse vertex order of interior ring.",
    },
    "NO_SELF_INTERSECTION": {
        "rule_id": "OGC-SFA-004",
        "description": (
            "No ring of a polygon may intersect itself. A ring may "
            "touch itself at a single point (forming a figure-8) "
            "only if the touching point is a vertex of the ring."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": True,
        "repair_method": (
            "Split self-intersecting ring at intersection points "
            "and restructure into valid sub-polygons."
        ),
    },
    "HOLES_WITHIN_EXTERIOR": {
        "rule_id": "OGC-SFA-005",
        "description": (
            "Every interior ring (hole) must be entirely contained "
            "within the exterior ring."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": False,
        "repair_method": (
            "Remove offending holes or clip holes to exterior boundary."
        ),
    },
    "NO_DUPLICATE_RINGS": {
        "rule_id": "OGC-SFA-006",
        "description": (
            "A polygon must not contain duplicate rings (rings that are "
            "geometrically identical)."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": True,
        "repair_method": "Remove duplicate ring(s), keeping one copy.",
    },
    "HOLES_NOT_NESTED": {
        "rule_id": "OGC-SFA-007",
        "description": (
            "Interior rings must not be nested within each other; each "
            "hole must be directly contained by the exterior ring."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": False,
        "repair_method": (
            "Flatten nested holes into a single level or restructure "
            "as MultiPolygon."
        ),
    },
    "HOLES_NOT_TOUCHING": {
        "rule_id": "OGC-SFA-008",
        "description": (
            "Interior rings may touch the exterior ring or each other "
            "at a single point, but must not cross."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": True,
        "repair_method": "Buffer and restructure touching rings.",
    },
    "MINIMUM_VERTICES": {
        "rule_id": "OGC-SFA-009",
        "description": (
            "A valid polygon ring must have at least 4 vertices "
            "(3 unique vertices plus the closing vertex)."
        ),
        "standard": "OGC 06-103r4 Section 6.1.11.1",
        "severity": "error",
        "auto_repairable": False,
        "repair_method": "Reject polygon; insufficient geometry.",
    },
    "COORDINATE_DIMENSION": {
        "rule_id": "OGC-SFA-010",
        "description": (
            "All coordinates in a ring must have the same dimension "
            "(2D, 3D, or 4D). Mixing dimensions is invalid."
        ),
        "standard": "OGC 06-103r4 Section 6.1.2.2",
        "severity": "error",
        "auto_repairable": True,
        "repair_method": "Promote or demote coordinates to consistent dimension.",
    },
}

# ---------------------------------------------------------------------------
# EUDR geolocation requirements (Regulation (EU) 2023/1115 Article 9)
# ---------------------------------------------------------------------------
# These rules implement the specific geolocation data requirements
# set forth in the EUDR for due diligence statements.

EUDR_GEOLOCATION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "POLYGON_THRESHOLD": {
        "rule_id": "EUDR-GEO-001",
        "article": "Article 9(1)(d)",
        "description": (
            "Plots of land with an area equal to or exceeding 4 hectares "
            "must provide geolocation as polygon coordinates sufficient "
            "to describe the perimeter of the plot."
        ),
        "threshold_hectares": 4.0,
        "threshold_m2": 40000.0,
        "above_threshold": "polygon_required",
        "below_threshold": "point_sufficient",
    },
    "POINT_GEOLOCATION": {
        "rule_id": "EUDR-GEO-002",
        "article": "Article 9(1)(d)",
        "description": (
            "Plots of land with an area below 4 hectares may provide "
            "geolocation as a single latitude/longitude point (centroid "
            "or representative point)."
        ),
        "min_precision_decimal_places": 6,
        "format": "decimal_degrees",
        "crs": "EPSG:4326",
    },
    "COORDINATE_SYSTEM": {
        "rule_id": "EUDR-GEO-003",
        "article": "Article 9(1)(d)",
        "description": (
            "All geolocation data must use the WGS84 coordinate "
            "reference system (EPSG:4326)."
        ),
        "required_crs": "EPSG:4326",
        "datum": "WGS84",
        "coordinate_order": "latitude, longitude",
        "storage_order": "[longitude, latitude]",
    },
    "POSITIONAL_ACCURACY": {
        "rule_id": "EUDR-GEO-004",
        "article": "Article 9(1)(d)",
        "description": (
            "Adequate positional accuracy for EUDR compliance requires "
            "a minimum of 8 decimal places for coordinates (~1.1mm "
            "precision at the equator). Submission format requires at "
            "least 6 decimal places."
        ),
        "min_decimal_places_storage": 8,
        "min_decimal_places_submission": 6,
        "approximate_precision_m_at_equator_8dp": 0.0011,
        "approximate_precision_m_at_equator_6dp": 0.11,
    },
    "POLYGON_FORMAT": {
        "rule_id": "EUDR-GEO-005",
        "article": "Article 9(1)(d)",
        "description": (
            "Polygon boundaries must be expressed as an ordered array "
            "of [longitude, latitude] coordinate pairs forming a "
            "closed ring."
        ),
        "coordinate_pair_format": "[longitude, latitude]",
        "ring_closure": "first_vertex_equals_last_vertex",
        "orientation": "counter_clockwise_exterior",
    },
    "DDS_SUBMISSION": {
        "rule_id": "EUDR-GEO-006",
        "article": "Article 9",
        "description": (
            "Geolocation data is submitted as part of the Due Diligence "
            "Statement (DDS) to the EU Information System."
        ),
        "max_vertices_per_polygon": 10000,
        "max_submission_size_bytes": 10_485_760,
        "accepted_formats": ["json", "xml"],
    },
    "RECORD_RETENTION": {
        "rule_id": "EUDR-GEO-007",
        "article": "Article 31",
        "description": (
            "Operators shall keep due diligence statements and "
            "supporting documentation, including geolocation data, "
            "for a period of at least 5 years from the date the "
            "statement was made."
        ),
        "retention_years": 5,
        "retention_applies_to": [
            "plot_boundaries",
            "boundary_versions",
            "compliance_reports",
            "provenance_records",
        ],
    },
    "OPERATOR_OBLIGATION": {
        "rule_id": "EUDR-GEO-008",
        "article": "Article 10(2)(c)",
        "description": (
            "Risk assessment must verify that the geolocation "
            "provided is consistent with the claimed country of "
            "origin and production area."
        ),
        "verification_checks": [
            "coordinate_within_country",
            "coordinate_on_land",
            "area_consistent_with_commodity",
        ],
    },
}

# ---------------------------------------------------------------------------
# ISO 19107 spatial schema rules
# ---------------------------------------------------------------------------
# Rules derived from ISO 19107:2019 Geographic Information -- Spatial Schema
# for GM_Polygon and GM_SurfaceBoundary geometry types.

ISO_19107_RULES: Dict[str, Dict[str, Any]] = {
    "GM_POLYGON_EXTERIOR": {
        "rule_id": "ISO-19107-001",
        "description": (
            "A GM_Polygon is defined by one exterior boundary "
            "(GM_SurfaceBoundary.exterior) and zero or more interior "
            "boundaries (GM_SurfaceBoundary.interior)."
        ),
        "standard": "ISO 19107:2019 Section 6.4.12",
        "constraint": "Exactly one exterior ring required.",
    },
    "GM_SURFACE_BOUNDARY_ORIENTATION": {
        "rule_id": "ISO-19107-002",
        "description": (
            "The exterior boundary of a GM_Polygon must be oriented "
            "such that the surface is to the left of the boundary "
            "when traversed in the direction of its orientation "
            "(counter-clockwise for a right-hand rule coordinate system)."
        ),
        "standard": "ISO 19107:2019 Section 6.4.12",
        "constraint": "Exterior ring CCW, interior rings CW.",
    },
    "GM_POLYGON_PLANAR": {
        "rule_id": "ISO-19107-003",
        "description": (
            "A GM_Polygon must be planar. In geographic coordinates, "
            "this means all points lie on the surface of the Earth "
            "ellipsoid (or geoid) with consistent elevation model."
        ),
        "standard": "ISO 19107:2019 Section 6.4.12",
        "constraint": "All vertices at consistent reference surface.",
    },
    "GM_POLYGON_CONNECTED": {
        "rule_id": "ISO-19107-004",
        "description": (
            "The interior of a GM_Polygon (after removing holes) must "
            "be a connected point set."
        ),
        "standard": "ISO 19107:2019 Section 6.4.12",
        "constraint": "Single connected component after hole removal.",
    },
    "GM_CURVE_SIMPLE": {
        "rule_id": "ISO-19107-005",
        "description": (
            "Each boundary ring (GM_Curve) must be simple: it must "
            "not pass through the same point twice except at the "
            "start/end closure point."
        ),
        "standard": "ISO 19107:2019 Section 6.3.6",
        "constraint": "No self-intersections in boundary curves.",
    },
}

# ---------------------------------------------------------------------------
# Validation thresholds (default parameters)
# ---------------------------------------------------------------------------
# These define the default tolerance values used by the BoundaryValidator
# engine when checking polygon geometry quality.

VALIDATION_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "DUPLICATE_VERTEX_TOLERANCE": {
        "parameter_id": "VT-001",
        "description": (
            "Distance in metres below which consecutive vertices are "
            "considered duplicates and may be collapsed."
        ),
        "value": 0.01,
        "unit": "metres",
        "rationale": "1cm precision matches high-accuracy GNSS surveys.",
    },
    "RING_CLOSURE_TOLERANCE": {
        "parameter_id": "VT-002",
        "description": (
            "Maximum distance in metres between the first and last "
            "vertices of a ring before it is considered unclosed."
        ),
        "value": 0.1,
        "unit": "metres",
        "rationale": (
            "10cm tolerance accommodates minor rounding in coordinate "
            "transformations while ensuring topological closure."
        ),
    },
    "SPIKE_ANGLE_THRESHOLD": {
        "parameter_id": "VT-003",
        "description": (
            "Minimum interior angle in degrees at a vertex. Angles "
            "below this threshold indicate a spike artifact."
        ),
        "value": 1.0,
        "unit": "degrees",
        "rationale": (
            "1 degree is well below any plausible real-world boundary "
            "angle and catches only digitisation artifacts."
        ),
    },
    "SLIVER_ASPECT_RATIO": {
        "parameter_id": "VT-004",
        "description": (
            "Maximum ratio of polygon length to width before the "
            "polygon is flagged as a sliver. Slivers are typically "
            "digitisation artifacts from overlapping boundaries."
        ),
        "value": 20.0,
        "unit": "dimensionless",
        "rationale": (
            "A 20:1 ratio allows long narrow farm plots while "
            "catching clear sliver artifacts."
        ),
    },
    "MINIMUM_POLYGON_AREA": {
        "parameter_id": "VT-005",
        "description": (
            "Minimum polygon area in hectares. Polygons below this "
            "threshold may be digitisation errors."
        ),
        "value": 0.0001,
        "unit": "hectares",
        "equivalent_m2": 1.0,
        "rationale": (
            "1 sq metre minimum catches near-zero-area degenerate "
            "polygons while allowing micro-plots."
        ),
    },
    "MAXIMUM_POLYGON_AREA": {
        "parameter_id": "VT-006",
        "description": (
            "Maximum polygon area in hectares. Polygons exceeding "
            "this threshold are likely data entry errors or "
            "administrative boundary geometries rather than plots."
        ),
        "value": 50000.0,
        "unit": "hectares",
        "equivalent_km2": 500.0,
        "rationale": (
            "500 km2 accommodates the largest cattle ranches in "
            "Brazil while flagging country-level boundaries."
        ),
    },
    "MINIMUM_VERTEX_COUNT": {
        "parameter_id": "VT-007",
        "description": (
            "Minimum number of vertices in an exterior ring for a "
            "valid polygon. Must be at least 4 (3 unique + closure)."
        ),
        "value": 4,
        "unit": "vertices",
        "rationale": "A closed triangle is the simplest valid polygon.",
    },
    "MAXIMUM_VERTEX_COUNT": {
        "parameter_id": "VT-008",
        "description": (
            "Maximum number of vertices in a single polygon before "
            "simplification is recommended."
        ),
        "value": 100000,
        "unit": "vertices",
        "rationale": (
            "100K vertices balances high-resolution boundaries with "
            "processing performance."
        ),
    },
    "COORDINATE_PRECISION": {
        "parameter_id": "VT-009",
        "description": (
            "Number of decimal places for internal coordinate "
            "storage in WGS84 (EPSG:4326)."
        ),
        "value": 8,
        "unit": "decimal_places",
        "approximate_precision_m": 0.0011,
        "rationale": (
            "8 decimal places provides ~1.1mm precision at equator, "
            "exceeding EUDR minimum of 6 decimal places."
        ),
    },
    "SELF_INTERSECTION_BUFFER": {
        "parameter_id": "VT-010",
        "description": (
            "Buffer distance in metres used when testing for "
            "near-self-intersections caused by coordinate rounding."
        ),
        "value": 0.001,
        "unit": "metres",
        "rationale": "1mm buffer catches numerical precision artifacts.",
    },
}

# ---------------------------------------------------------------------------
# Commodity area ranges (typical plot sizes by EUDR commodity)
# ---------------------------------------------------------------------------
# Based on FAO commodity surveys, EUDR impact assessments, and industry
# benchmarks. Used for anomaly detection (flagging implausible areas).

COMMODITY_AREA_RANGES: Dict[str, Dict[str, Any]] = {
    "palm_oil": {
        "commodity_name": "Palm Oil",
        "eudr_article": "Article 1(1)(a)",
        "min_hectares": 0.5,
        "max_hectares": 50000.0,
        "typical_smallholder_ha": 2.0,
        "typical_industrial_ha": 5000.0,
        "primary_countries": ["IDN", "MYS", "NGA", "THA", "COL", "ECU"],
        "note": (
            "Smallholder plots dominate in Indonesia/Malaysia. "
            "Large industrial plantations up to 50,000 ha."
        ),
    },
    "cocoa": {
        "commodity_name": "Cocoa",
        "eudr_article": "Article 1(1)(b)",
        "min_hectares": 0.1,
        "max_hectares": 500.0,
        "typical_smallholder_ha": 1.5,
        "typical_industrial_ha": 50.0,
        "primary_countries": ["CIV", "GHA", "CMR", "NGA", "ECU", "IDN"],
        "note": (
            "Predominantly smallholder crop. Most farms 0.5-5 ha. "
            "Few industrial plantations exceed 100 ha."
        ),
    },
    "coffee": {
        "commodity_name": "Coffee",
        "eudr_article": "Article 1(1)(c)",
        "min_hectares": 0.1,
        "max_hectares": 1000.0,
        "typical_smallholder_ha": 1.0,
        "typical_industrial_ha": 100.0,
        "primary_countries": ["BRA", "VNM", "COL", "ETH", "HND", "IDN"],
        "note": (
            "Smallholder-dominated in Africa and Central America. "
            "Larger estates in Brazil and Vietnam."
        ),
    },
    "soya": {
        "commodity_name": "Soya",
        "eudr_article": "Article 1(1)(d)",
        "min_hectares": 1.0,
        "max_hectares": 100000.0,
        "typical_smallholder_ha": 50.0,
        "typical_industrial_ha": 10000.0,
        "primary_countries": ["BRA", "ARG", "USA", "PRY", "BOL"],
        "note": (
            "Large-scale mechanised agriculture dominates. "
            "Cerrado region farms commonly exceed 1,000 ha."
        ),
    },
    "rubber": {
        "commodity_name": "Rubber",
        "eudr_article": "Article 1(1)(e)",
        "min_hectares": 0.5,
        "max_hectares": 10000.0,
        "typical_smallholder_ha": 2.0,
        "typical_industrial_ha": 500.0,
        "primary_countries": ["THA", "IDN", "MYS", "VNM", "CIV", "CHN"],
        "note": (
            "Smallholders dominate in Thailand and Indonesia. "
            "Industrial plantations to 10,000 ha in Malaysia."
        ),
    },
    "cattle": {
        "commodity_name": "Cattle (Bovine Animals)",
        "eudr_article": "Article 1(1)(f)",
        "min_hectares": 1.0,
        "max_hectares": 500000.0,
        "typical_smallholder_ha": 50.0,
        "typical_industrial_ha": 50000.0,
        "primary_countries": ["BRA", "ARG", "AUS", "USA", "PRY", "URY"],
        "note": (
            "Cattle ranches in Brazil can exceed 100,000 ha. "
            "The largest estancias approach 500,000 ha."
        ),
    },
    "wood": {
        "commodity_name": "Wood / Timber",
        "eudr_article": "Article 1(1)(g)",
        "min_hectares": 0.5,
        "max_hectares": 1000000.0,
        "typical_smallholder_ha": 10.0,
        "typical_industrial_ha": 100000.0,
        "primary_countries": ["BRA", "IDN", "MYS", "RUS", "CAN", "COD"],
        "note": (
            "Forest concessions can be extremely large. "
            "Brazilian Amazon concessions up to 1 million ha."
        ),
    },
}

# ---------------------------------------------------------------------------
# Country CRS defaults
# ---------------------------------------------------------------------------
# Maps countries/regions to their preferred local CRS for metric
# area calculations. When a boundary's country is known, the system
# can auto-select the most appropriate CRS.

COUNTRY_CRS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # South America
    "BRA": {
        "country_name": "Brazil",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "SIRGAS 2000 / UTM zone {zone}S",
        "default_utm_epsg": 31983,
        "note": "Most of Brazil falls in UTM zones 21-25 South.",
    },
    "ARG": {
        "country_name": "Argentina",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "SIRGAS 2000 / UTM zone {zone}S",
        "default_utm_epsg": 32720,
        "note": "Argentina spans UTM zones 19-21 South.",
    },
    "COL": {
        "country_name": "Colombia",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32618,
        "note": "Colombia spans UTM zones 17-19 (mostly North).",
    },
    "PRY": {
        "country_name": "Paraguay",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32721,
        "note": "Paraguay falls in UTM zones 20-21 South.",
    },
    "BOL": {
        "country_name": "Bolivia",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32720,
        "note": "Bolivia spans UTM zones 19-21 South.",
    },
    "ECU": {
        "country_name": "Ecuador",
        "geographic_crs": 4674,
        "geographic_crs_name": "SIRGAS 2000",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32717,
        "note": "Ecuador straddles UTM zone 17 (mostly South).",
    },

    # Southeast Asia
    "IDN": {
        "country_name": "Indonesia",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32748,
        "note": (
            "Indonesia spans UTM zones 46-54 (mostly South). "
            "Zone 48S covers Sumatra and most of Kalimantan."
        ),
    },
    "MYS": {
        "country_name": "Malaysia",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32647,
        "note": "Peninsular Malaysia in zone 47N, Sabah/Sarawak zone 49-50N.",
    },
    "THA": {
        "country_name": "Thailand",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32647,
        "note": "Thailand spans UTM zones 47-48 North.",
    },
    "VNM": {
        "country_name": "Vietnam",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32648,
        "note": "Vietnam spans UTM zones 48-49 North.",
    },

    # West Africa
    "CIV": {
        "country_name": "Cote d'Ivoire",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32630,
        "note": "Cote d'Ivoire falls in UTM zone 30 North.",
    },
    "GHA": {
        "country_name": "Ghana",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32630,
        "note": "Ghana straddles UTM zones 30-31 North.",
    },
    "CMR": {
        "country_name": "Cameroon",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32632,
        "note": "Cameroon spans UTM zones 32-33 North.",
    },
    "NGA": {
        "country_name": "Nigeria",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32632,
        "note": "Nigeria spans UTM zones 31-33 North.",
    },

    # East Africa
    "ETH": {
        "country_name": "Ethiopia",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32637,
        "note": "Ethiopia spans UTM zones 36-38 North.",
    },
    "COD": {
        "country_name": "Democratic Republic of the Congo",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32735,
        "note": "DRC spans UTM zones 33-36 (mostly South).",
    },
    "KEN": {
        "country_name": "Kenya",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}S",
        "default_utm_epsg": 32737,
        "note": "Kenya straddles UTM zone 37 (equatorial).",
    },
    "UGA": {
        "country_name": "Uganda",
        "geographic_crs": 4326,
        "geographic_crs_name": "WGS 84",
        "projected_crs_template": "WGS 84 / UTM zone {zone}N",
        "default_utm_epsg": 32636,
        "note": "Uganda spans UTM zones 35-36 North.",
    },

    # Europe
    "DEU": {
        "country_name": "Germany",
        "geographic_crs": 4258,
        "geographic_crs_name": "ETRS89",
        "projected_crs_template": "ETRS89 / UTM zone {zone}N",
        "default_utm_epsg": 25832,
        "note": "Germany spans UTM zones 32-33 under ETRS89.",
    },
    "FRA": {
        "country_name": "France",
        "geographic_crs": 4258,
        "geographic_crs_name": "ETRS89",
        "projected_crs_template": "ETRS89 / UTM zone {zone}N",
        "default_utm_epsg": 25831,
        "note": "Metropolitan France spans UTM zones 30-32 under ETRS89.",
    },
    "ESP": {
        "country_name": "Spain",
        "geographic_crs": 4258,
        "geographic_crs_name": "ETRS89",
        "projected_crs_template": "ETRS89 / UTM zone {zone}N",
        "default_utm_epsg": 25830,
        "note": "Spain spans UTM zones 29-31 under ETRS89.",
    },
    "NLD": {
        "country_name": "Netherlands",
        "geographic_crs": 4258,
        "geographic_crs_name": "ETRS89",
        "projected_crs_template": "ETRS89 / UTM zone {zone}N",
        "default_utm_epsg": 25831,
        "note": "Netherlands in UTM zone 31 under ETRS89.",
    },

    # Australia
    "AUS": {
        "country_name": "Australia",
        "geographic_crs": 7844,
        "geographic_crs_name": "GDA2020",
        "projected_crs_template": "GDA2020 / MGA zone {zone}",
        "default_utm_epsg": 7855,
        "note": (
            "Australia uses GDA2020 / MGA zones 49-56. "
            "Zone 55 covers eastern seaboard."
        ),
    },

    # New Zealand
    "NZL": {
        "country_name": "New Zealand",
        "geographic_crs": 4167,
        "geographic_crs_name": "NZGD2000",
        "projected_crs_template": "NZGD2000 / UTM zone {zone}S",
        "default_utm_epsg": 32759,
        "note": "New Zealand spans UTM zones 58-60 South.",
    },
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------


def get_ogc_rule(rule_key: str) -> Optional[Dict[str, Any]]:
    """Return an OGC Simple Features validity rule by key.

    Args:
        rule_key: Rule key (e.g. 'RING_CLOSURE', 'NO_SELF_INTERSECTION').

    Returns:
        Rule dictionary, or None if not found.
    """
    return OGC_SIMPLE_FEATURES_RULES.get(rule_key)


def get_eudr_requirement(requirement_key: str) -> Optional[Dict[str, Any]]:
    """Return an EUDR geolocation requirement by key.

    Args:
        requirement_key: Requirement key (e.g. 'POLYGON_THRESHOLD').

    Returns:
        Requirement dictionary, or None if not found.
    """
    return EUDR_GEOLOCATION_REQUIREMENTS.get(requirement_key)


def get_iso_rule(rule_key: str) -> Optional[Dict[str, Any]]:
    """Return an ISO 19107 spatial schema rule by key.

    Args:
        rule_key: Rule key (e.g. 'GM_POLYGON_EXTERIOR').

    Returns:
        Rule dictionary, or None if not found.
    """
    return ISO_19107_RULES.get(rule_key)


def get_validation_threshold(threshold_key: str) -> Optional[Dict[str, Any]]:
    """Return a validation threshold by key.

    Args:
        threshold_key: Threshold key (e.g. 'SPIKE_ANGLE_THRESHOLD').

    Returns:
        Threshold dictionary, or None if not found.
    """
    return VALIDATION_THRESHOLDS.get(threshold_key)


def get_commodity_area_range(commodity: str) -> Optional[Dict[str, Any]]:
    """Return typical area ranges for an EUDR commodity.

    Args:
        commodity: Commodity identifier (e.g. 'palm_oil', 'cocoa').

    Returns:
        Area range dictionary, or None if not found.
    """
    return COMMODITY_AREA_RANGES.get(commodity.lower())


def get_country_crs(country_code: str) -> Optional[Dict[str, Any]]:
    """Return the default CRS settings for a country.

    Args:
        country_code: ISO 3166-1 alpha-3 country code (e.g. 'BRA', 'IDN').

    Returns:
        Country CRS dictionary, or None if not found.
    """
    return COUNTRY_CRS_DEFAULTS.get(country_code.upper())


def is_polygon_required(area_hectares: float) -> bool:
    """Determine whether a full polygon boundary is required under EUDR.

    Per Article 9(1)(d), plots >= 4 hectares require polygon boundaries;
    plots < 4 hectares may use a single coordinate point.

    Args:
        area_hectares: Plot area in hectares.

    Returns:
        True if polygon is required, False if point is sufficient.

    Example:
        >>> is_polygon_required(5.0)
        True
        >>> is_polygon_required(2.5)
        False
    """
    threshold = EUDR_GEOLOCATION_REQUIREMENTS["POLYGON_THRESHOLD"][
        "threshold_hectares"
    ]
    return area_hectares >= threshold


def get_all_auto_repairable_rules() -> List[str]:
    """Return keys of all OGC rules that support automatic repair.

    Returns:
        List of rule keys where auto_repairable is True.
    """
    return [
        key
        for key, rule in OGC_SIMPLE_FEATURES_RULES.items()
        if rule.get("auto_repairable", False)
    ]


def get_all_commodity_names() -> Tuple[str, ...]:
    """Return a tuple of all EUDR commodity identifiers.

    Returns:
        Tuple of commodity keys in sorted order.
    """
    return tuple(sorted(COMMODITY_AREA_RANGES.keys()))


def get_all_country_codes() -> Tuple[str, ...]:
    """Return a tuple of all country codes with CRS defaults.

    Returns:
        Tuple of ISO 3166-1 alpha-3 country codes in sorted order.
    """
    return tuple(sorted(COUNTRY_CRS_DEFAULTS.keys()))


# ---------------------------------------------------------------------------
# Module-level counts for introspection
# ---------------------------------------------------------------------------

TOTAL_OGC_RULES: int = len(OGC_SIMPLE_FEATURES_RULES)
TOTAL_EUDR_REQUIREMENTS: int = len(EUDR_GEOLOCATION_REQUIREMENTS)
TOTAL_ISO_RULES: int = len(ISO_19107_RULES)
TOTAL_VALIDATION_THRESHOLDS: int = len(VALIDATION_THRESHOLDS)
TOTAL_COMMODITY_RANGES: int = len(COMMODITY_AREA_RANGES)
TOTAL_COUNTRY_CRS_DEFAULTS: int = len(COUNTRY_CRS_DEFAULTS)
