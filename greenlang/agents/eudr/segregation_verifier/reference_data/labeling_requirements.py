# -*- coding: utf-8 -*-
"""
Labeling Requirements Reference Data - AGENT-EUDR-010

Label content, placement, and visual identification rules for physical
segregation verification of EUDR-compliant material.  Used by the
LabelingVerificationEngine (Engine 6) for zero-hallucination compliance
checking.

Datasets:
    LABEL_CONTENT_REQUIREMENTS:
        Required and optional fields for each of 8 label types used
        in EUDR segregation operations (compliance tag, zone sign,
        vehicle placard, container seal label, batch sticker, pallet
        marker, silo sign, processing line marker).

    COLOR_CODE_STANDARD:
        Color-coding scheme for visual identification of compliant,
        non-compliant, pending, buffer, and quarantine material
        across storage zones, containers, and labels.

    LABEL_PLACEMENT_RULES:
        Placement requirements per domain (storage, transport,
        processing) including minimum label count, positions,
        visibility distance, and material specifications.

    LABEL_DURABILITY_REQUIREMENTS:
        Minimum durability specifications for labels by environment
        (indoor, outdoor, cold, hot/humid, marine) covering fade
        resistance, adhesion, and water resistance.

    LABEL_SIZE_SPECIFICATIONS:
        Minimum dimensions for each label type to ensure legibility
        and regulatory compliance.

Lookup helpers:
    get_label_requirements(label_type) -> dict | None
    get_placement_rules(domain) -> dict | None
    get_color_meaning(color) -> str
    get_required_fields(label_type) -> list[str]
    get_optional_fields(label_type) -> list[str]
    is_field_required(label_type, field_name) -> bool
    validate_label_completeness(label_type, fields_present) -> tuple[bool, list]
    get_min_label_count(domain) -> int
    get_all_label_types() -> list[str]

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label content requirements (8 label types)
# ---------------------------------------------------------------------------

LABEL_CONTENT_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "compliance_tag": {
        "name": "Compliance Tag",
        "description": (
            "Primary compliance status label attached to individual "
            "batch containers, bags, or pallets."
        ),
        "required_fields": [
            "compliance_status",
            "batch_id",
            "commodity",
            "origin_country",
            "date_applied",
            "operator_id",
        ],
        "optional_fields": [
            "certification_ref",
            "coc_model",
            "dds_reference",
            "geolocation_hash",
            "expiry_date",
            "qr_code",
        ],
        "format_rules": {
            "batch_id_pattern": r"^[A-Z]{2,4}-\d{6,12}$",
            "date_format": "ISO-8601",
            "compliance_status_values": [
                "compliant", "non_compliant", "pending",
            ],
        },
        "tamper_evident": True,
        "removable": False,
    },
    "zone_sign": {
        "name": "Storage Zone Sign",
        "description": (
            "Permanent or semi-permanent sign posted at the entrance "
            "and inside storage zones to identify compliance status."
        ),
        "required_fields": [
            "zone_id",
            "compliance_status",
            "commodity",
            "color_code",
            "effective_date",
        ],
        "optional_fields": [
            "facility_id",
            "capacity_mt",
            "responsible_person",
            "contact_info",
            "zone_map_reference",
        ],
        "format_rules": {
            "color_code_values": [
                "green", "red", "yellow", "blue", "white",
            ],
            "date_format": "ISO-8601",
        },
        "tamper_evident": False,
        "removable": True,
    },
    "vehicle_placard": {
        "name": "Vehicle Placard",
        "description": (
            "Placard displayed on transport vehicles indicating "
            "the compliance status of the cargo being carried."
        ),
        "required_fields": [
            "vehicle_id",
            "compliance_status",
            "commodity",
            "color_code",
            "cleaning_date",
        ],
        "optional_fields": [
            "batch_id",
            "origin_country",
            "destination",
            "driver_name",
            "seal_number",
        ],
        "format_rules": {
            "color_code_values": [
                "green", "red", "yellow",
            ],
            "date_format": "ISO-8601",
        },
        "tamper_evident": False,
        "removable": True,
    },
    "container_seal_label": {
        "name": "Container Seal Label",
        "description": (
            "Label applied to container seals linking the physical "
            "seal to the batch and its compliance status."
        ),
        "required_fields": [
            "seal_number",
            "batch_id",
            "compliance_status",
            "commodity",
            "sealing_date",
            "operator_id",
        ],
        "optional_fields": [
            "container_number",
            "destination_port",
            "dds_reference",
            "qr_code",
        ],
        "format_rules": {
            "seal_number_pattern": r"^[A-Z0-9]{8,20}$",
            "date_format": "ISO-8601",
        },
        "tamper_evident": True,
        "removable": False,
    },
    "batch_sticker": {
        "name": "Batch Sticker",
        "description": (
            "Small adhesive label placed on individual bags, boxes, "
            "or units within a batch for item-level tracking."
        ),
        "required_fields": [
            "batch_id",
            "commodity",
            "compliance_status",
            "date_applied",
        ],
        "optional_fields": [
            "weight_kg",
            "origin_country",
            "lot_number",
            "barcode",
            "qr_code",
        ],
        "format_rules": {
            "date_format": "ISO-8601",
        },
        "tamper_evident": False,
        "removable": False,
    },
    "pallet_marker": {
        "name": "Pallet Marker",
        "description": (
            "Color-coded marker card or tag attached to pallets "
            "to indicate compliance status at pallet level."
        ),
        "required_fields": [
            "pallet_id",
            "batch_id",
            "compliance_status",
            "color_code",
            "commodity",
        ],
        "optional_fields": [
            "weight_kg",
            "unit_count",
            "destination",
            "date_applied",
        ],
        "format_rules": {
            "color_code_values": [
                "green", "red", "yellow",
            ],
        },
        "tamper_evident": False,
        "removable": True,
    },
    "silo_sign": {
        "name": "Silo / Tank Sign",
        "description": (
            "Identification sign on bulk storage silos, tanks, or "
            "bins indicating the compliance status of stored material."
        ),
        "required_fields": [
            "silo_id",
            "compliance_status",
            "commodity",
            "color_code",
            "last_fill_date",
        ],
        "optional_fields": [
            "capacity_mt",
            "current_volume_mt",
            "batch_ids",
            "cleaning_date",
            "responsible_person",
        ],
        "format_rules": {
            "color_code_values": [
                "green", "red", "yellow", "blue",
            ],
            "date_format": "ISO-8601",
        },
        "tamper_evident": False,
        "removable": True,
    },
    "processing_line_marker": {
        "name": "Processing Line Marker",
        "description": (
            "Status board or marker at the start of a processing "
            "line indicating current batch and compliance status."
        ),
        "required_fields": [
            "line_id",
            "compliance_status",
            "commodity",
            "color_code",
            "batch_id",
        ],
        "optional_fields": [
            "changeover_date",
            "shift_number",
            "operator_name",
            "expected_completion",
        ],
        "format_rules": {
            "color_code_values": [
                "green", "red", "yellow",
            ],
            "date_format": "ISO-8601",
        },
        "tamper_evident": False,
        "removable": True,
    },
}

TOTAL_LABEL_TYPES: int = len(LABEL_CONTENT_REQUIREMENTS)

# ---------------------------------------------------------------------------
# Color code standard
# ---------------------------------------------------------------------------

COLOR_CODE_STANDARD: Dict[str, Dict[str, Any]] = {
    "green": {
        "meaning": "EUDR-compliant / deforestation-free",
        "hex": "#22C55E",
        "rgb": (34, 197, 94),
        "pantone": "354 C",
        "application": [
            "storage_zones",
            "containers",
            "labels",
            "pallets",
            "vehicle_placards",
            "processing_line_markers",
        ],
        "priority": 1,
        "notes": "Used for confirmed compliant material with valid DDS.",
    },
    "red": {
        "meaning": "Non-compliant / deforestation-linked or unknown",
        "hex": "#EF4444",
        "rgb": (239, 68, 68),
        "pantone": "185 C",
        "application": [
            "storage_zones",
            "containers",
            "labels",
            "pallets",
            "vehicle_placards",
            "processing_line_markers",
        ],
        "priority": 2,
        "notes": (
            "Used for confirmed non-compliant or deforestation-linked "
            "material.  Material must not enter EU market."
        ),
    },
    "yellow": {
        "meaning": "Pending verification / under review",
        "hex": "#EAB308",
        "rgb": (234, 179, 8),
        "pantone": "109 C",
        "application": [
            "storage_zones",
            "containers",
            "labels",
            "pallets",
            "vehicle_placards",
            "processing_line_markers",
        ],
        "priority": 3,
        "notes": (
            "Used for material awaiting DDS verification or risk "
            "assessment.  Must not be mixed with green-coded material."
        ),
    },
    "blue": {
        "meaning": "Buffer zone / transition area",
        "hex": "#3B82F6",
        "rgb": (59, 130, 246),
        "pantone": "285 C",
        "application": [
            "storage_zones",
            "silo_signs",
        ],
        "priority": 4,
        "notes": (
            "Used for buffer areas between compliant and non-compliant "
            "zones.  No permanent storage; transition only."
        ),
    },
    "white": {
        "meaning": "Quarantine / contamination hold",
        "hex": "#F8FAFC",
        "rgb": (248, 250, 252),
        "pantone": "White",
        "application": [
            "storage_zones",
            "containers",
        ],
        "priority": 5,
        "notes": (
            "Used for material under quarantine due to contamination "
            "event or investigation.  Must not be moved or processed."
        ),
    },
}

TOTAL_COLORS: int = len(COLOR_CODE_STANDARD)

# ---------------------------------------------------------------------------
# Label placement rules per domain
# ---------------------------------------------------------------------------

LABEL_PLACEMENT_RULES: Dict[str, Dict[str, Any]] = {
    "storage": {
        "min_labels_per_zone": 2,
        "positions": ["entrance", "interior"],
        "visibility_distance_meters": 10.0,
        "min_height_meters": 1.5,
        "max_height_meters": 3.0,
        "illumination_required": True,
        "weather_protection_required": True,
        "label_types_required": ["zone_sign"],
        "label_types_optional": ["silo_sign"],
        "inspection_frequency_days": 30,
        "replacement_trigger": [
            "faded_beyond_50_percent",
            "damaged",
            "detached",
            "status_change",
        ],
        "notes": (
            "Zone signs must be visible from the zone entrance.  "
            "Interior label placed at the furthest point from entrance."
        ),
    },
    "transport": {
        "min_labels_per_vehicle": 2,
        "positions": ["cab_exterior", "cargo_door"],
        "visibility_distance_meters": 5.0,
        "min_height_meters": 1.0,
        "max_height_meters": 2.5,
        "illumination_required": False,
        "weather_protection_required": True,
        "label_types_required": ["vehicle_placard"],
        "label_types_optional": [
            "container_seal_label",
            "batch_sticker",
        ],
        "inspection_frequency_days": 1,
        "replacement_trigger": [
            "cargo_change",
            "cleaning_event",
            "damaged",
        ],
        "notes": (
            "Vehicle placard must be replaced at each cargo change.  "
            "Seal labels applied at sealing and must match manifest."
        ),
    },
    "processing": {
        "min_labels_per_line": 1,
        "positions": ["line_start"],
        "visibility_distance_meters": 5.0,
        "min_height_meters": 1.5,
        "max_height_meters": 2.5,
        "illumination_required": True,
        "weather_protection_required": False,
        "label_types_required": ["processing_line_marker"],
        "label_types_optional": ["batch_sticker", "pallet_marker"],
        "inspection_frequency_days": 1,
        "replacement_trigger": [
            "changeover",
            "batch_change",
            "shift_change",
        ],
        "notes": (
            "Processing line marker must be updated after every "
            "changeover between compliant and non-compliant runs."
        ),
    },
}

TOTAL_PLACEMENT_DOMAINS: int = len(LABEL_PLACEMENT_RULES)

# ---------------------------------------------------------------------------
# Label durability requirements by environment
# ---------------------------------------------------------------------------

LABEL_DURABILITY_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "indoor_ambient": {
        "name": "Indoor Ambient (warehouse, dry store)",
        "min_lifespan_months": 12,
        "fade_resistance": "standard",
        "adhesion_grade": "permanent",
        "water_resistance": "splash_proof",
        "temperature_range_c": (5, 40),
        "uv_resistance_required": False,
        "material": "coated_paper_or_vinyl",
    },
    "outdoor": {
        "name": "Outdoor (container yard, open storage)",
        "min_lifespan_months": 6,
        "fade_resistance": "uv_stabilized",
        "adhesion_grade": "permanent",
        "water_resistance": "waterproof",
        "temperature_range_c": (-20, 50),
        "uv_resistance_required": True,
        "material": "polypropylene_or_polyester",
    },
    "cold_storage": {
        "name": "Cold Storage (cold room, refrigerated)",
        "min_lifespan_months": 12,
        "fade_resistance": "standard",
        "adhesion_grade": "freezer_grade",
        "water_resistance": "waterproof",
        "temperature_range_c": (-30, 5),
        "uv_resistance_required": False,
        "material": "freezer_grade_vinyl",
    },
    "hot_humid": {
        "name": "Hot & Humid (tropical warehouse, plantation)",
        "min_lifespan_months": 6,
        "fade_resistance": "uv_stabilized",
        "adhesion_grade": "high_tack",
        "water_resistance": "waterproof",
        "temperature_range_c": (20, 55),
        "uv_resistance_required": True,
        "material": "polyester_with_uv_coating",
    },
    "marine": {
        "name": "Marine (vessel, port, dockside)",
        "min_lifespan_months": 3,
        "fade_resistance": "uv_stabilized",
        "adhesion_grade": "permanent",
        "water_resistance": "salt_water_resistant",
        "temperature_range_c": (-10, 45),
        "uv_resistance_required": True,
        "material": "marine_grade_polyester",
    },
}

TOTAL_DURABILITY_ENVIRONMENTS: int = len(LABEL_DURABILITY_REQUIREMENTS)

# ---------------------------------------------------------------------------
# Label size specifications (minimum dimensions)
# ---------------------------------------------------------------------------

LABEL_SIZE_SPECIFICATIONS: Dict[str, Dict[str, Any]] = {
    "compliance_tag": {
        "min_width_mm": 70,
        "min_height_mm": 50,
        "min_font_size_pt": 8,
        "min_text_contrast_ratio": 4.5,
        "barcode_required": False,
        "qr_code_min_mm": 15,
    },
    "zone_sign": {
        "min_width_mm": 400,
        "min_height_mm": 300,
        "min_font_size_pt": 36,
        "min_text_contrast_ratio": 7.0,
        "barcode_required": False,
        "qr_code_min_mm": None,
    },
    "vehicle_placard": {
        "min_width_mm": 300,
        "min_height_mm": 200,
        "min_font_size_pt": 24,
        "min_text_contrast_ratio": 7.0,
        "barcode_required": False,
        "qr_code_min_mm": None,
    },
    "container_seal_label": {
        "min_width_mm": 50,
        "min_height_mm": 30,
        "min_font_size_pt": 6,
        "min_text_contrast_ratio": 4.5,
        "barcode_required": True,
        "qr_code_min_mm": 10,
    },
    "batch_sticker": {
        "min_width_mm": 50,
        "min_height_mm": 30,
        "min_font_size_pt": 8,
        "min_text_contrast_ratio": 4.5,
        "barcode_required": True,
        "qr_code_min_mm": 10,
    },
    "pallet_marker": {
        "min_width_mm": 200,
        "min_height_mm": 150,
        "min_font_size_pt": 18,
        "min_text_contrast_ratio": 7.0,
        "barcode_required": False,
        "qr_code_min_mm": None,
    },
    "silo_sign": {
        "min_width_mm": 500,
        "min_height_mm": 400,
        "min_font_size_pt": 48,
        "min_text_contrast_ratio": 7.0,
        "barcode_required": False,
        "qr_code_min_mm": None,
    },
    "processing_line_marker": {
        "min_width_mm": 300,
        "min_height_mm": 200,
        "min_font_size_pt": 24,
        "min_text_contrast_ratio": 7.0,
        "barcode_required": False,
        "qr_code_min_mm": None,
    },
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_label_requirements(
    label_type: str,
) -> Optional[Dict[str, Any]]:
    """Return content requirements for a label type.

    Args:
        label_type: Label type identifier (compliance_tag, zone_sign,
            vehicle_placard, container_seal_label, batch_sticker,
            pallet_marker, silo_sign, processing_line_marker).

    Returns:
        Dictionary with label requirements, or None if not found.
    """
    return LABEL_CONTENT_REQUIREMENTS.get(label_type)


def get_placement_rules(
    domain: str,
) -> Optional[Dict[str, Any]]:
    """Return label placement rules for a domain.

    Args:
        domain: Segregation domain (storage, transport, processing).

    Returns:
        Dictionary with placement rules, or None if not found.
    """
    return LABEL_PLACEMENT_RULES.get(domain)


def get_color_meaning(color: str) -> str:
    """Return the meaning of a color code.

    Args:
        color: Color identifier (green, red, yellow, blue, white).

    Returns:
        Meaning string.  Returns "unknown" if color is not found.
    """
    info = COLOR_CODE_STANDARD.get(color)
    if info is None:
        return "unknown"
    return str(info["meaning"])


def get_required_fields(label_type: str) -> List[str]:
    """Return required fields for a label type.

    Args:
        label_type: Label type identifier.

    Returns:
        List of required field names.
        Returns empty list if label type is unknown.
    """
    reqs = LABEL_CONTENT_REQUIREMENTS.get(label_type)
    if reqs is None:
        return []
    return list(reqs.get("required_fields", []))


def get_optional_fields(label_type: str) -> List[str]:
    """Return optional fields for a label type.

    Args:
        label_type: Label type identifier.

    Returns:
        List of optional field names.
        Returns empty list if label type is unknown.
    """
    reqs = LABEL_CONTENT_REQUIREMENTS.get(label_type)
    if reqs is None:
        return []
    return list(reqs.get("optional_fields", []))


def is_field_required(label_type: str, field_name: str) -> bool:
    """Check whether a field is required for a label type.

    Args:
        label_type: Label type identifier.
        field_name: Field name to check.

    Returns:
        True if the field is in the required_fields list.
    """
    required = get_required_fields(label_type)
    return field_name in required


def validate_label_completeness(
    label_type: str,
    fields_present: List[str],
) -> Tuple[bool, List[str]]:
    """Validate that all required fields are present on a label.

    Args:
        label_type: Label type identifier.
        fields_present: List of field names actually present.

    Returns:
        Tuple of (is_complete, missing_fields) where is_complete
        is True if all required fields are present, and
        missing_fields lists any required fields that are absent.
    """
    required = get_required_fields(label_type)
    if not required:
        return (True, [])
    present_set = set(fields_present)
    missing = [f for f in required if f not in present_set]
    return (len(missing) == 0, missing)


def get_min_label_count(domain: str) -> int:
    """Return the minimum label count for a domain.

    Args:
        domain: Segregation domain (storage, transport, processing).

    Returns:
        Minimum label count.  Returns 1 if domain is unknown.
    """
    rules = LABEL_PLACEMENT_RULES.get(domain)
    if rules is None:
        return 1
    # Keys differ by domain
    if domain == "storage":
        return int(rules.get("min_labels_per_zone", 1))
    if domain == "transport":
        return int(rules.get("min_labels_per_vehicle", 1))
    if domain == "processing":
        return int(rules.get("min_labels_per_line", 1))
    return 1


def get_all_label_types() -> List[str]:
    """Return all supported label type identifiers.

    Returns:
        Sorted list of label type identifier strings.
    """
    return sorted(LABEL_CONTENT_REQUIREMENTS.keys())


def get_color_hex(color: str) -> str:
    """Return the hex color code for a color identifier.

    Args:
        color: Color identifier (green, red, yellow, blue, white).

    Returns:
        Hex color string (e.g. "#22C55E").
        Returns "#000000" if color is unknown.
    """
    info = COLOR_CODE_STANDARD.get(color)
    if info is None:
        return "#000000"
    return str(info["hex"])


def get_label_size(
    label_type: str,
) -> Optional[Dict[str, Any]]:
    """Return minimum size specifications for a label type.

    Args:
        label_type: Label type identifier.

    Returns:
        Dictionary with size specifications, or None if not found.
    """
    return LABEL_SIZE_SPECIFICATIONS.get(label_type)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "LABEL_CONTENT_REQUIREMENTS",
    "COLOR_CODE_STANDARD",
    "LABEL_PLACEMENT_RULES",
    "LABEL_DURABILITY_REQUIREMENTS",
    "LABEL_SIZE_SPECIFICATIONS",
    # Counts
    "TOTAL_LABEL_TYPES",
    "TOTAL_COLORS",
    "TOTAL_PLACEMENT_DOMAINS",
    "TOTAL_DURABILITY_ENVIRONMENTS",
    # Lookup helpers
    "get_label_requirements",
    "get_placement_rules",
    "get_color_meaning",
    "get_required_fields",
    "get_optional_fields",
    "is_field_required",
    "validate_label_completeness",
    "get_min_label_count",
    "get_all_label_types",
    "get_color_hex",
    "get_label_size",
]
