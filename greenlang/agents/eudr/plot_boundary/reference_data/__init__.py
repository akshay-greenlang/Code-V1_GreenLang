# -*- coding: utf-8 -*-
"""
Reference Data Package - AGENT-EUDR-006: Plot Boundary Manager Agent

Provides built-in reference datasets for plot boundary management:
    - projection_parameters: CRS definitions, UTM zones, datum transforms, ellipsoids
    - boundary_standards: OGC/ISO/EUDR validation rules, thresholds, commodity areas
    - simplification_rules: Presets, quality gates, format limits, resolution levels

These datasets enable deterministic, zero-hallucination boundary validation,
area calculation, simplification, and compliance assessment without external
API dependencies. All data is version-tracked and provenance-auditable.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from greenlang.agents.eudr.plot_boundary.reference_data.projection_parameters import (
    CRS_DEFINITIONS,
    ELLIPSOID_PARAMETERS,
    TOTAL_CRS_COUNT,
    TOTAL_ELLIPSOIDS,
    TOTAL_TRANSFORMATIONS,
    TOTAL_UTM_ZONES,
    TRANSFORMATION_PARAMETERS,
    UTM_ZONE_PARAMETERS,
    get_all_epsg_codes,
    get_all_utm_zone_keys,
    get_central_meridian,
    get_crs_definition,
    get_ellipsoid,
    get_transformation,
    get_utm_epsg,
    get_utm_hemisphere,
    get_utm_zone,
    get_utm_zone_parameters,
    is_geographic_crs,
    is_projected_crs,
)
from greenlang.agents.eudr.plot_boundary.reference_data.boundary_standards import (
    COMMODITY_AREA_RANGES,
    COUNTRY_CRS_DEFAULTS,
    EUDR_GEOLOCATION_REQUIREMENTS,
    ISO_19107_RULES,
    OGC_SIMPLE_FEATURES_RULES,
    VALIDATION_THRESHOLDS,
    get_all_auto_repairable_rules,
    get_all_commodity_names,
    get_all_country_codes,
    get_commodity_area_range,
    get_country_crs,
    get_eudr_requirement,
    get_iso_rule,
    get_ogc_rule,
    get_validation_threshold,
    is_polygon_required,
)
from greenlang.agents.eudr.plot_boundary.reference_data.simplification_rules import (
    ALGORITHM_METADATA,
    EUDR_SUBMISSION_TOLERANCE,
    FORMAT_VERTEX_LIMITS,
    METHOD_DOUGLAS_PEUCKER,
    METHOD_TOPOLOGY_PRESERVING,
    METHOD_VISVALINGAM_WHYATT,
    MULTI_RESOLUTION_LEVELS,
    QUALITY_THRESHOLDS,
    SIMPLIFICATION_PRESETS,
    VALID_SIMPLIFICATION_METHODS,
    get_algorithm_metadata,
    get_all_format_names,
    get_all_preset_names,
    get_eudr_safe_presets,
    get_eudr_tolerance,
    get_format_limits,
    get_preset,
    get_quality_threshold,
    get_resolution_level,
    max_vertices_for_format,
)

__all__ = [
    # projection_parameters
    "CRS_DEFINITIONS",
    "UTM_ZONE_PARAMETERS",
    "TRANSFORMATION_PARAMETERS",
    "ELLIPSOID_PARAMETERS",
    "TOTAL_CRS_COUNT",
    "TOTAL_UTM_ZONES",
    "TOTAL_TRANSFORMATIONS",
    "TOTAL_ELLIPSOIDS",
    "get_crs_definition",
    "get_utm_zone",
    "get_utm_hemisphere",
    "get_utm_epsg",
    "get_utm_zone_parameters",
    "get_transformation",
    "get_ellipsoid",
    "get_central_meridian",
    "is_geographic_crs",
    "is_projected_crs",
    "get_all_epsg_codes",
    "get_all_utm_zone_keys",
    # boundary_standards
    "OGC_SIMPLE_FEATURES_RULES",
    "EUDR_GEOLOCATION_REQUIREMENTS",
    "ISO_19107_RULES",
    "VALIDATION_THRESHOLDS",
    "COMMODITY_AREA_RANGES",
    "COUNTRY_CRS_DEFAULTS",
    "get_ogc_rule",
    "get_eudr_requirement",
    "get_iso_rule",
    "get_validation_threshold",
    "get_commodity_area_range",
    "get_country_crs",
    "is_polygon_required",
    "get_all_auto_repairable_rules",
    "get_all_commodity_names",
    "get_all_country_codes",
    # simplification_rules
    "SIMPLIFICATION_PRESETS",
    "EUDR_SUBMISSION_TOLERANCE",
    "QUALITY_THRESHOLDS",
    "FORMAT_VERTEX_LIMITS",
    "MULTI_RESOLUTION_LEVELS",
    "ALGORITHM_METADATA",
    "VALID_SIMPLIFICATION_METHODS",
    "METHOD_DOUGLAS_PEUCKER",
    "METHOD_VISVALINGAM_WHYATT",
    "METHOD_TOPOLOGY_PRESERVING",
    "get_preset",
    "get_eudr_tolerance",
    "get_format_limits",
    "get_quality_threshold",
    "get_resolution_level",
    "get_algorithm_metadata",
    "get_all_preset_names",
    "get_all_format_names",
    "get_eudr_safe_presets",
    "max_vertices_for_format",
]
