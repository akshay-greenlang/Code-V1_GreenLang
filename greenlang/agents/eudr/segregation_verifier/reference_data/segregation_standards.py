# -*- coding: utf-8 -*-
"""
Segregation Standards Reference Data - AGENT-EUDR-010

Segregation requirements per certification standard for physical
separation of EUDR-compliant vs non-compliant material.  These
standards define mandatory barrier types, cleaning protocols,
changeover requirements, labeling rules, and audit frequencies
for every major sustainability certification scheme used alongside
EUDR compliance.

Datasets:
    SEGREGATION_STANDARDS:
        Per-certification-scheme segregation requirements covering
        storage, transport, processing, and labeling.  Keyed by
        certification standard identifier (e.g. "FSC-STD-40-004").

    COMMODITY_SEGREGATION_REQUIREMENTS:
        Per-commodity-specific segregation requirements including
        minimum barrier type, contamination risk level, dedicated
        equipment necessity, and allergen handling.

    RISK_LEVEL_MAPPING:
        Mapping of commodity-pathway combinations to risk levels
        (critical, high, medium, low) for contamination assessment.

    MINIMUM_BARRIER_TYPES:
        Ordered hierarchy of barrier types from weakest (floor_marking)
        to strongest (separate_building), with effectiveness scores.

Lookup helpers:
    get_standard(standard_id) -> dict | None
    get_commodity_requirements(commodity) -> dict | None
    get_risk_level(commodity, pathway) -> str
    get_barrier_effectiveness(barrier_type) -> float
    get_all_standards() -> list[str]
    get_audit_frequency(standard_id) -> int
    is_dedicated_required(standard_id, domain) -> bool

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier (GL-EUDR-SGV-010)
Status: Production Ready
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Segregation standards by certification scheme
# ---------------------------------------------------------------------------

SEGREGATION_STANDARDS: Dict[str, Dict[str, Any]] = {
    # ---- FSC Chain of Custody v3.0 ----
    "FSC-STD-40-004": {
        "name": "FSC Chain of Custody",
        "version": "3.0",
        "effective_date": "2024-01-01",
        "applicable_commodities": ["wood", "rubber"],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "physical_barrier",
                "min_separation_meters": 3.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "sweep_wash",
                "cleaning_certificate_required": True,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 3,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 30,
                "flush_required": False,
                "first_run_discard": False,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": False,
                "logo_display_required": True,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "ASI",
        "penalty_for_mixing": "certificate_suspension",
    },

    # ---- RSPO Supply Chain Certification 2020 ----
    "RSPO-SCC-2020": {
        "name": "RSPO Supply Chain Certification Standard",
        "version": "2020",
        "effective_date": "2020-11-01",
        "applicable_commodities": ["oil_palm"],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "physical_barrier",
                "min_separation_meters": 5.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": True,
                "cleaning_required": True,
                "min_cleaning_method": "power_wash",
                "cleaning_certificate_required": True,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 5,
            },
            "processing": {
                "dedicated_line": True,
                "changeover_required": True,
                "min_changeover_minutes": 60,
                "flush_required": True,
                "first_run_discard": True,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": True,
                "logo_display_required": True,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "RSPO-AU",
        "penalty_for_mixing": "certificate_revocation",
    },

    # ---- ISCC 202 v4.0 ----
    "ISCC-202-v4": {
        "name": "ISCC Chain of Custody Requirements",
        "version": "4.0",
        "effective_date": "2024-03-01",
        "applicable_commodities": [
            "oil_palm", "soya", "wood", "rubber",
        ],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "wire_mesh_fence",
                "min_separation_meters": 3.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "sweep_wash",
                "cleaning_certificate_required": True,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 3,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 45,
                "flush_required": True,
                "first_run_discard": False,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": False,
                "logo_display_required": True,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "ISCC-CB",
        "penalty_for_mixing": "certificate_suspension",
    },

    # ---- UTZ / Rainforest Alliance CoC Standard ----
    "UTZ-RA-CoC-002": {
        "name": "UTZ/Rainforest Alliance Chain of Custody Standard",
        "version": "2.0",
        "effective_date": "2022-07-01",
        "applicable_commodities": ["cocoa", "coffee"],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "wire_mesh_fence",
                "min_separation_meters": 2.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "sweep_wash",
                "cleaning_certificate_required": False,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 3,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 30,
                "flush_required": False,
                "first_run_discard": False,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": False,
                "logo_display_required": True,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "RA-CB",
        "penalty_for_mixing": "certificate_suspension",
    },

    # ---- Fairtrade SOP ----
    "Fairtrade-SOP": {
        "name": "Fairtrade Standard Operating Procedures for CoC",
        "version": "1.0",
        "effective_date": "2023-01-01",
        "applicable_commodities": ["cocoa", "coffee", "soya"],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "floor_marking",
                "min_separation_meters": 2.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "sweep_wash",
                "cleaning_certificate_required": False,
                "seal_required": True,
                "cargo_history_check": False,
                "max_previous_cargoes_checked": 1,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 20,
                "flush_required": False,
                "first_run_discard": False,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": False,
                "logo_display_required": True,
                "claim_type_required": True,
                "traceability_code_required": False,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "FLOCERT",
        "penalty_for_mixing": "non_conformity_major",
    },

    # ---- ISO 22095:2020 ----
    "ISO-22095-2020": {
        "name": "ISO 22095 Chain of Custody - General Terminology and Models",
        "version": "2020",
        "effective_date": "2020-06-01",
        "applicable_commodities": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "physical_barrier",
                "min_separation_meters": 5.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "power_wash",
                "cleaning_certificate_required": True,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 5,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 60,
                "flush_required": True,
                "first_run_discard": True,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": True,
                "logo_display_required": False,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 12,
        "record_retention_years": 5,
        "certification_body_accreditation": "ISO-AB",
        "penalty_for_mixing": "non_conformity_critical",
    },

    # ---- EUDR Baseline (EU 2023/1115) ----
    "EUDR-2023-1115": {
        "name": "EU Deforestation Regulation Baseline Requirements",
        "version": "2023",
        "effective_date": "2025-12-30",
        "applicable_commodities": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
        "segregation_requirements": {
            "storage": {
                "physical_separation": True,
                "dedicated_area": True,
                "barrier_required": True,
                "min_barrier_type": "physical_barrier",
                "min_separation_meters": 5.0,
                "signage_required": True,
                "inventory_tracking": True,
                "mixed_storage_allowed": False,
            },
            "transport": {
                "dedicated_vehicle": False,
                "cleaning_required": True,
                "min_cleaning_method": "power_wash",
                "cleaning_certificate_required": True,
                "seal_required": True,
                "cargo_history_check": True,
                "max_previous_cargoes_checked": 5,
            },
            "processing": {
                "dedicated_line": False,
                "changeover_required": True,
                "min_changeover_minutes": 60,
                "flush_required": True,
                "first_run_discard": True,
                "batch_separation": True,
                "production_log_required": True,
            },
            "labeling": {
                "batch_label_required": True,
                "zone_label_required": True,
                "color_code_required": True,
                "logo_display_required": False,
                "claim_type_required": True,
                "traceability_code_required": True,
            },
        },
        "audit_frequency_months": 6,
        "record_retention_years": 5,
        "certification_body_accreditation": "EU-CA",
        "penalty_for_mixing": "market_prohibition",
    },
}

# Total standard count for assertion in tests
TOTAL_STANDARDS: int = len(SEGREGATION_STANDARDS)

# ---------------------------------------------------------------------------
# Commodity-specific segregation requirements
# ---------------------------------------------------------------------------

COMMODITY_SEGREGATION_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "min_barrier_type": "physical_barrier",
        "contamination_risk_level": "high",
        "dedicated_transport_required": True,
        "dedicated_processing_required": True,
        "allergen_handling": False,
        "temperature_controlled": True,
        "max_storage_days": 7,
        "traceability_granularity": "individual_animal",
        "special_requirements": [
            "ear_tag_verification",
            "veterinary_certificate",
            "slaughter_date_tracking",
        ],
    },
    "cocoa": {
        "min_barrier_type": "wire_mesh_fence",
        "contamination_risk_level": "medium",
        "dedicated_transport_required": False,
        "dedicated_processing_required": False,
        "allergen_handling": True,
        "temperature_controlled": False,
        "max_storage_days": 180,
        "traceability_granularity": "batch",
        "special_requirements": [
            "moisture_control",
            "pest_prevention",
            "fermentation_tracking",
        ],
    },
    "coffee": {
        "min_barrier_type": "wire_mesh_fence",
        "contamination_risk_level": "medium",
        "dedicated_transport_required": False,
        "dedicated_processing_required": False,
        "allergen_handling": False,
        "temperature_controlled": False,
        "max_storage_days": 365,
        "traceability_granularity": "lot",
        "special_requirements": [
            "moisture_control",
            "cupping_score_tracking",
            "origin_grade_separation",
        ],
    },
    "oil_palm": {
        "min_barrier_type": "physical_barrier",
        "contamination_risk_level": "critical",
        "dedicated_transport_required": True,
        "dedicated_processing_required": True,
        "allergen_handling": False,
        "temperature_controlled": True,
        "max_storage_days": 3,
        "traceability_granularity": "batch",
        "special_requirements": [
            "ffb_processing_within_24h",
            "tank_segregation",
            "pipeline_flushing",
            "acid_value_monitoring",
        ],
    },
    "rubber": {
        "min_barrier_type": "floor_marking",
        "contamination_risk_level": "low",
        "dedicated_transport_required": False,
        "dedicated_processing_required": False,
        "allergen_handling": True,
        "temperature_controlled": False,
        "max_storage_days": 365,
        "traceability_granularity": "lot",
        "special_requirements": [
            "latex_vs_dry_separation",
            "grade_classification",
        ],
    },
    "soya": {
        "min_barrier_type": "physical_barrier",
        "contamination_risk_level": "high",
        "dedicated_transport_required": False,
        "dedicated_processing_required": True,
        "allergen_handling": True,
        "temperature_controlled": False,
        "max_storage_days": 180,
        "traceability_granularity": "batch",
        "special_requirements": [
            "gmo_non_gmo_separation",
            "allergen_cross_contact_prevention",
            "extraction_solvent_tracking",
        ],
    },
    "wood": {
        "min_barrier_type": "floor_marking",
        "contamination_risk_level": "low",
        "dedicated_transport_required": False,
        "dedicated_processing_required": False,
        "allergen_handling": False,
        "temperature_controlled": False,
        "max_storage_days": 730,
        "traceability_granularity": "lot",
        "special_requirements": [
            "species_identification",
            "timber_marking",
            "log_number_tracking",
        ],
    },
}

# Total commodity count
TOTAL_COMMODITIES: int = len(COMMODITY_SEGREGATION_REQUIREMENTS)

# ---------------------------------------------------------------------------
# Risk level mapping: (commodity, pathway) -> risk level
# ---------------------------------------------------------------------------

RISK_LEVEL_MAPPING: Dict[str, Dict[str, str]] = {
    "cattle": {
        "shared_storage": "critical",
        "shared_transport": "critical",
        "shared_processing": "critical",
        "shared_equipment": "high",
        "temporal_overlap": "high",
        "adjacent_storage": "high",
        "residual_material": "critical",
        "handling_error": "high",
        "labeling_error": "medium",
        "documentation_error": "medium",
    },
    "cocoa": {
        "shared_storage": "medium",
        "shared_transport": "medium",
        "shared_processing": "high",
        "shared_equipment": "medium",
        "temporal_overlap": "medium",
        "adjacent_storage": "low",
        "residual_material": "medium",
        "handling_error": "medium",
        "labeling_error": "medium",
        "documentation_error": "low",
    },
    "coffee": {
        "shared_storage": "medium",
        "shared_transport": "medium",
        "shared_processing": "medium",
        "shared_equipment": "low",
        "temporal_overlap": "low",
        "adjacent_storage": "low",
        "residual_material": "medium",
        "handling_error": "medium",
        "labeling_error": "medium",
        "documentation_error": "low",
    },
    "oil_palm": {
        "shared_storage": "critical",
        "shared_transport": "critical",
        "shared_processing": "critical",
        "shared_equipment": "critical",
        "temporal_overlap": "high",
        "adjacent_storage": "high",
        "residual_material": "critical",
        "handling_error": "high",
        "labeling_error": "medium",
        "documentation_error": "medium",
    },
    "rubber": {
        "shared_storage": "low",
        "shared_transport": "low",
        "shared_processing": "medium",
        "shared_equipment": "low",
        "temporal_overlap": "low",
        "adjacent_storage": "low",
        "residual_material": "low",
        "handling_error": "low",
        "labeling_error": "medium",
        "documentation_error": "low",
    },
    "soya": {
        "shared_storage": "high",
        "shared_transport": "high",
        "shared_processing": "critical",
        "shared_equipment": "high",
        "temporal_overlap": "medium",
        "adjacent_storage": "medium",
        "residual_material": "high",
        "handling_error": "high",
        "labeling_error": "medium",
        "documentation_error": "medium",
    },
    "wood": {
        "shared_storage": "low",
        "shared_transport": "low",
        "shared_processing": "low",
        "shared_equipment": "low",
        "temporal_overlap": "low",
        "adjacent_storage": "low",
        "residual_material": "low",
        "handling_error": "low",
        "labeling_error": "medium",
        "documentation_error": "low",
    },
}

# ---------------------------------------------------------------------------
# Barrier type hierarchy (weakest to strongest)
# ---------------------------------------------------------------------------

MINIMUM_BARRIER_TYPES: Dict[str, Dict[str, Any]] = {
    "floor_marking": {
        "rank": 1,
        "effectiveness_score": 30.0,
        "description": "Painted lines or tape on floor",
        "physical_barrier": False,
        "prevents_liquid_contamination": False,
        "prevents_airborne_contamination": False,
        "suitable_for": ["wood", "rubber"],
    },
    "plastic_curtain": {
        "rank": 2,
        "effectiveness_score": 40.0,
        "description": "Plastic strip curtains or sheeting",
        "physical_barrier": False,
        "prevents_liquid_contamination": False,
        "prevents_airborne_contamination": True,
        "suitable_for": ["cocoa", "coffee", "wood", "rubber"],
    },
    "wire_mesh_fence": {
        "rank": 3,
        "effectiveness_score": 55.0,
        "description": "Wire mesh or chain-link fencing",
        "physical_barrier": True,
        "prevents_liquid_contamination": False,
        "prevents_airborne_contamination": False,
        "suitable_for": ["cocoa", "coffee", "soya", "wood", "rubber"],
    },
    "physical_barrier": {
        "rank": 4,
        "effectiveness_score": 70.0,
        "description": "Solid partition wall (wood, metal, or concrete block)",
        "physical_barrier": True,
        "prevents_liquid_contamination": True,
        "prevents_airborne_contamination": False,
        "suitable_for": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
    },
    "steel_partition": {
        "rank": 5,
        "effectiveness_score": 80.0,
        "description": "Steel or metal panel partition with sealed edges",
        "physical_barrier": True,
        "prevents_liquid_contamination": True,
        "prevents_airborne_contamination": True,
        "suitable_for": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
    },
    "concrete_wall": {
        "rank": 6,
        "effectiveness_score": 90.0,
        "description": "Permanent concrete or masonry wall",
        "physical_barrier": True,
        "prevents_liquid_contamination": True,
        "prevents_airborne_contamination": True,
        "suitable_for": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
    },
    "separate_room": {
        "rank": 7,
        "effectiveness_score": 95.0,
        "description": "Enclosed room within the same building",
        "physical_barrier": True,
        "prevents_liquid_contamination": True,
        "prevents_airborne_contamination": True,
        "suitable_for": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
    },
    "separate_building": {
        "rank": 8,
        "effectiveness_score": 100.0,
        "description": "Dedicated separate building or facility",
        "physical_barrier": True,
        "prevents_liquid_contamination": True,
        "prevents_airborne_contamination": True,
        "suitable_for": [
            "cattle", "cocoa", "coffee", "oil_palm",
            "rubber", "soya", "wood",
        ],
    },
}

TOTAL_BARRIER_TYPES: int = len(MINIMUM_BARRIER_TYPES)


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_standard(standard_id: str) -> Optional[Dict[str, Any]]:
    """Return segregation standard data for a given standard ID.

    Args:
        standard_id: Certification standard identifier
            (e.g. "FSC-STD-40-004").

    Returns:
        Dictionary with standard data, or None if not found.
    """
    return SEGREGATION_STANDARDS.get(standard_id)


def get_commodity_requirements(
    commodity: str,
) -> Optional[Dict[str, Any]]:
    """Return commodity-specific segregation requirements.

    Args:
        commodity: EUDR commodity (cattle, cocoa, coffee,
            oil_palm, rubber, soya, wood).

    Returns:
        Dictionary with commodity requirements, or None if not found.
    """
    return COMMODITY_SEGREGATION_REQUIREMENTS.get(commodity)


def get_risk_level(commodity: str, pathway: str) -> str:
    """Return the contamination risk level for a commodity-pathway pair.

    Args:
        commodity: EUDR commodity.
        pathway: Contamination pathway (shared_storage,
            shared_transport, etc.).

    Returns:
        Risk level string (critical, high, medium, low).
        Defaults to "medium" if the combination is unknown.
    """
    commodity_risks = RISK_LEVEL_MAPPING.get(commodity, {})
    return commodity_risks.get(pathway, "medium")


def get_barrier_effectiveness(barrier_type: str) -> float:
    """Return the effectiveness score (0-100) for a barrier type.

    Args:
        barrier_type: Barrier type identifier (floor_marking,
            plastic_curtain, wire_mesh_fence, physical_barrier,
            steel_partition, concrete_wall, separate_room,
            separate_building).

    Returns:
        Effectiveness score (0.0-100.0).
        Returns 0.0 if barrier type is unknown.
    """
    info = MINIMUM_BARRIER_TYPES.get(barrier_type)
    if info is None:
        return 0.0
    return float(info["effectiveness_score"])


def get_all_standards() -> List[str]:
    """Return all supported segregation standard IDs.

    Returns:
        Sorted list of standard identifier strings.
    """
    return sorted(SEGREGATION_STANDARDS.keys())


def get_audit_frequency(standard_id: str) -> int:
    """Return the audit frequency in months for a standard.

    Args:
        standard_id: Certification standard identifier.

    Returns:
        Audit frequency in months.  Defaults to 12 if not found.
    """
    std = SEGREGATION_STANDARDS.get(standard_id)
    if std is None:
        return 12
    return int(std.get("audit_frequency_months", 12))


def is_dedicated_required(
    standard_id: str,
    domain: str,
) -> bool:
    """Check whether a standard requires dedicated equipment for a domain.

    Args:
        standard_id: Certification standard identifier.
        domain: Segregation domain ("storage", "transport", "processing").

    Returns:
        True if dedicated equipment is required, False otherwise.
    """
    std = SEGREGATION_STANDARDS.get(standard_id)
    if std is None:
        return False
    reqs = std.get("segregation_requirements", {}).get(domain, {})
    if domain == "transport":
        return bool(reqs.get("dedicated_vehicle", False))
    if domain == "processing":
        return bool(reqs.get("dedicated_line", False))
    if domain == "storage":
        return bool(reqs.get("dedicated_area", False))
    return False


def get_barrier_rank(barrier_type: str) -> int:
    """Return the ordinal rank (1-8) for a barrier type.

    Higher rank means stronger barrier.

    Args:
        barrier_type: Barrier type identifier.

    Returns:
        Rank integer (1-8).  Returns 0 if unknown.
    """
    info = MINIMUM_BARRIER_TYPES.get(barrier_type)
    if info is None:
        return 0
    return int(info["rank"])


def meets_minimum_barrier(
    actual_barrier: str,
    required_barrier: str,
) -> bool:
    """Check if actual barrier meets or exceeds the required barrier.

    Args:
        actual_barrier: Barrier type actually in place.
        required_barrier: Minimum required barrier type.

    Returns:
        True if actual_barrier rank >= required_barrier rank.
    """
    actual_rank = get_barrier_rank(actual_barrier)
    required_rank = get_barrier_rank(required_barrier)
    return actual_rank >= required_rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Data structures
    "SEGREGATION_STANDARDS",
    "COMMODITY_SEGREGATION_REQUIREMENTS",
    "RISK_LEVEL_MAPPING",
    "MINIMUM_BARRIER_TYPES",
    # Counts
    "TOTAL_STANDARDS",
    "TOTAL_COMMODITIES",
    "TOTAL_BARRIER_TYPES",
    # Lookup helpers
    "get_standard",
    "get_commodity_requirements",
    "get_risk_level",
    "get_barrier_effectiveness",
    "get_all_standards",
    "get_audit_frequency",
    "is_dedicated_required",
    "get_barrier_rank",
    "meets_minimum_barrier",
]
