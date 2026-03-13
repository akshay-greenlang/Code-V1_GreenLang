# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-010 Segregation Verifier Agent test suite.

Provides reusable fixtures for segregation control points, storage zones,
transport vehicles, processing lines, contamination events, labels,
facility profiles, helper factories, assertion helpers, reference data
constants, and engine fixtures.

Sample Segregation Control Points (SCP):
    SCP_STORAGE_COCOA, SCP_TRANSPORT_PALM, SCP_PROCESSING_COFFEE,
    SCP_HANDLING_SOYA, SCP_LOADING_RUBBER

Sample Storage Zones:
    ZONE_COCOA_A, ZONE_COCOA_B, ZONE_PALM_C

Sample Transport Vehicles:
    VEHICLE_DEDICATED_TRUCK, VEHICLE_SHARED_CONTAINER

Sample Processing Lines:
    LINE_COCOA_ROASTING, LINE_PALM_PRESSING

Sample Contamination Events:
    CONTAMINATION_SPATIAL, CONTAMINATION_TEMPORAL

Sample Labels:
    LABEL_BIN_TAG, LABEL_ZONE_SIGN

Sample Facility Profiles:
    FACILITY_PROFILE_COCOA_WAREHOUSE, FACILITY_PROFILE_PALM_MILL

Helper Factories: make_scp(), make_zone(), make_vehicle(), make_line(),
    make_contamination(), make_label(), make_facility_profile()

Assertion Helpers: assert_valid_provenance_hash(), assert_valid_score(),
    assert_valid_compliance_status()

Reference Data Constants: SEGREGATION_METHODS, SCP_TYPES, STORAGE_TYPES,
    TRANSPORT_TYPES, PROCESSING_LINE_TYPES, CONTAMINATION_PATHWAYS,
    LABEL_TYPES, CAPABILITY_LEVELS, EUDR_COMMODITIES, REPORT_FORMATS

Engine Fixtures (8 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier Agent (GL-EUDR-SGV-010)
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
]

SEGREGATION_METHODS: List[str] = [
    "dedicated_facility",
    "physical_barrier",
    "separate_equipment",
    "temporal_separation",
    "color_coding",
    "labeling_only",
    "digital_tracking",
    "combined",
]

SCP_TYPES: List[str] = [
    "storage", "transport", "processing", "handling", "loading_unloading",
]

STORAGE_TYPES: List[str] = [
    "warehouse_bay", "silo", "tank", "cold_room", "open_yard",
    "container_yard", "bag_stack", "bulk_pile", "racking_system",
    "fumigation_chamber", "drying_floor", "covered_shed",
]

BARRIER_TYPES: List[str] = [
    "wall", "fence", "curtain", "floor_marking", "pallet_separation",
    "container", "net", "none",
]

TRANSPORT_TYPES: List[str] = [
    "truck", "container", "rail_wagon", "vessel_hold", "barge",
    "tanker", "flatbed", "refrigerated", "bulk_carrier", "air_freight",
]

CLEANING_METHODS: List[str] = [
    "dry_sweep", "compressed_air", "vacuum", "water_wash",
    "chemical_wash", "steam_clean", "fumigation",
]

PROCESSING_LINE_TYPES: List[str] = [
    "roasting", "milling", "pressing", "extraction", "refining",
    "fermentation", "drying", "sorting", "grading", "hulling",
    "crushing", "fractionation", "deodorization", "sawing", "tanning",
]

CONTAMINATION_PATHWAYS: List[str] = [
    "spatial_proximity", "temporal_proximity", "shared_equipment",
    "shared_personnel", "airborne_dust", "residual_material",
    "shared_conveyor", "shared_hopper", "shared_scale", "drainage",
]

CONTAMINATION_SEVERITIES: List[str] = [
    "low", "medium", "high", "critical",
]

LABEL_TYPES: List[str] = [
    "bin_tag", "zone_sign", "pallet_label", "container_seal",
    "vehicle_placard", "bag_stencil", "digital_display", "rfid_tag",
]

LABEL_STATUSES: List[str] = [
    "active", "expired", "damaged", "replaced", "removed",
]

LABEL_EVENT_TYPES: List[str] = [
    "applied", "verified", "damaged", "replaced", "removed",
]

STORAGE_EVENT_TYPES: List[str] = [
    "material_in", "material_out", "zone_transfer",
    "cleaning", "inspection",
]

CAPABILITY_LEVELS: List[Dict[str, Any]] = [
    {"level": 0, "name": "ad_hoc", "min_score": 0.0, "max_score": 19.99},
    {"level": 1, "name": "initial", "min_score": 20.0, "max_score": 39.99},
    {"level": 2, "name": "managed", "min_score": 40.0, "max_score": 59.99},
    {"level": 3, "name": "defined", "min_score": 60.0, "max_score": 74.99},
    {"level": 4, "name": "measured", "min_score": 75.0, "max_score": 89.99},
    {"level": 5, "name": "optimized", "min_score": 90.0, "max_score": 100.0},
]

REPORT_FORMATS: List[str] = ["json", "csv", "pdf", "eudr_xml"]

REPORT_TYPES: List[str] = ["audit", "contamination", "evidence", "trend"]

# Risk classification per segregation method
METHOD_RISK_LEVELS: Dict[str, str] = {
    "dedicated_facility": "low",
    "physical_barrier": "medium",
    "separate_equipment": "medium",
    "temporal_separation": "high",
    "color_coding": "high",
    "labeling_only": "high",
    "digital_tracking": "medium",
    "combined": "low",
}

# Risk weights per contamination pathway
PATHWAY_RISK_WEIGHTS: Dict[str, float] = {
    "spatial_proximity": 0.80,
    "temporal_proximity": 0.70,
    "shared_equipment": 0.90,
    "shared_personnel": 0.40,
    "airborne_dust": 0.50,
    "residual_material": 0.85,
    "shared_conveyor": 0.75,
    "shared_hopper": 0.80,
    "shared_scale": 0.60,
    "drainage": 0.55,
}

# Severity scores
SEVERITY_SCORES: Dict[str, float] = {
    "low": 0.25,
    "medium": 0.50,
    "high": 0.75,
    "critical": 1.00,
}

# Score weights for SCP compliance score
SCP_SCORE_WEIGHTS: Dict[str, float] = {
    "evidence": 0.30,
    "documentation": 0.25,
    "history": 0.25,
    "method": 0.20,
}

# Score weights for storage audit
STORAGE_SCORE_WEIGHTS: Dict[str, float] = {
    "barrier_quality": 0.30,
    "zone_separation": 0.25,
    "cleaning_compliance": 0.25,
    "inventory_accuracy": 0.20,
}

# Score weights for transport
TRANSPORT_SCORE_WEIGHTS: Dict[str, float] = {
    "vehicle_dedication": 0.30,
    "cleaning_verification": 0.25,
    "seal_integrity": 0.25,
    "cargo_history": 0.20,
}

# Changeover requirements per processing line type (minimum minutes)
CHANGEOVER_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "roasting": {"min_minutes": 30, "flush_required": True, "flush_kg": 5.0},
    "milling": {"min_minutes": 45, "flush_required": True, "flush_kg": 10.0},
    "pressing": {"min_minutes": 60, "flush_required": True, "flush_kg": 15.0},
    "extraction": {"min_minutes": 90, "flush_required": True, "flush_kg": 20.0},
    "refining": {"min_minutes": 120, "flush_required": True, "flush_kg": 25.0},
    "fermentation": {"min_minutes": 240, "flush_required": False, "flush_kg": 0.0},
    "drying": {"min_minutes": 60, "flush_required": False, "flush_kg": 0.0},
    "sorting": {"min_minutes": 15, "flush_required": False, "flush_kg": 0.0},
    "grading": {"min_minutes": 15, "flush_required": False, "flush_kg": 0.0},
    "hulling": {"min_minutes": 30, "flush_required": True, "flush_kg": 8.0},
    "crushing": {"min_minutes": 45, "flush_required": True, "flush_kg": 12.0},
    "fractionation": {"min_minutes": 90, "flush_required": True, "flush_kg": 20.0},
    "deodorization": {"min_minutes": 60, "flush_required": True, "flush_kg": 10.0},
    "sawing": {"min_minutes": 20, "flush_required": False, "flush_kg": 0.0},
    "tanning": {"min_minutes": 180, "flush_required": True, "flush_kg": 30.0},
}

# Equipment sharing risk weights
EQUIPMENT_SHARING_WEIGHTS: Dict[str, float] = {
    "scale": 0.60,
    "conveyor": 0.75,
    "hopper": 0.80,
    "tank": 0.90,
    "pump": 0.70,
    "dryer": 0.65,
}

# Label required content fields per label type
LABEL_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "bin_tag": ["batch_id", "commodity", "origin", "date"],
    "zone_sign": ["zone_id", "commodity", "status"],
    "pallet_label": ["batch_id", "commodity", "quantity", "origin"],
    "container_seal": ["container_id", "seal_number", "commodity"],
    "vehicle_placard": ["vehicle_id", "commodity", "route"],
    "bag_stencil": ["batch_id", "commodity", "weight"],
    "digital_display": ["zone_id", "commodity", "status", "last_updated"],
    "rfid_tag": ["tag_id", "batch_id", "commodity", "timestamp"],
}

# Certification standards for facility assessment
CERTIFICATION_STANDARDS: List[str] = ["FSC", "RSPO", "ISCC"]

# Facility assessment weights
ASSESSMENT_WEIGHTS: Dict[str, float] = {
    "layout": 0.25,
    "protocols": 0.20,
    "history": 0.20,
    "labeling": 0.15,
    "documentation": 0.20,
}

# Reverification default interval in days
DEFAULT_REVERIFICATION_DAYS: int = 90


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

# SCP IDs
SCP_ID_STORAGE_COCOA = "SCP-STOR-COC-001"
SCP_ID_TRANSPORT_PALM = "SCP-TRAN-PLM-001"
SCP_ID_PROCESSING_COFFEE = "SCP-PROC-COF-001"
SCP_ID_HANDLING_SOYA = "SCP-HAND-SOY-001"
SCP_ID_LOADING_RUBBER = "SCP-LOAD-RUB-001"

# Facility IDs
FAC_ID_WAREHOUSE_GH = "FAC-WRHS-GH-001"
FAC_ID_MILL_ID = "FAC-MILL-ID-001"
FAC_ID_FACTORY_DE = "FAC-FACT-DE-001"
FAC_ID_DEPOT_NL = "FAC-DEPT-NL-001"
FAC_ID_PORT_BR = "FAC-PORT-BR-001"

# Zone IDs
ZONE_ID_COCOA_A = "ZONE-COC-A-001"
ZONE_ID_COCOA_B = "ZONE-COC-B-001"
ZONE_ID_PALM_C = "ZONE-PLM-C-001"
ZONE_ID_MIXED_D = "ZONE-MIX-D-001"

# Vehicle IDs
VEHICLE_ID_TRUCK_01 = "VEH-TRK-001"
VEHICLE_ID_CONTAINER_01 = "VEH-CNT-001"
VEHICLE_ID_RAIL_01 = "VEH-RAL-001"

# Processing Line IDs
LINE_ID_ROASTING_01 = "LINE-RST-001"
LINE_ID_PRESSING_01 = "LINE-PRS-001"
LINE_ID_MILLING_01 = "LINE-MIL-001"

# Batch IDs
BATCH_ID_COCOA_001 = "BATCH-COC-001"
BATCH_ID_COCOA_002 = "BATCH-COC-002"
BATCH_ID_PALM_001 = "BATCH-PLM-001"
BATCH_ID_COFFEE_001 = "BATCH-COF-001"
BATCH_ID_SOYA_001 = "BATCH-SOY-001"

# Label IDs
LABEL_ID_BIN_01 = "LBL-BIN-001"
LABEL_ID_ZONE_01 = "LBL-ZON-001"
LABEL_ID_PALLET_01 = "LBL-PAL-001"


# ---------------------------------------------------------------------------
# Timestamp Helper
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    ).isoformat()


# ---------------------------------------------------------------------------
# Sample Segregation Control Points
# ---------------------------------------------------------------------------

SCP_STORAGE_COCOA: Dict[str, Any] = {
    "scp_id": SCP_ID_STORAGE_COCOA,
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "location_lat": 5.5571,
    "location_lon": -0.2013,
    "scp_type": "storage",
    "commodity": "cocoa",
    "capacity_kg": 50000.0,
    "segregation_method": "physical_barrier",
    "status": "active",
    "last_verified": _ts(days_ago=30),
    "next_verification": _ts(days_ago=-60),
    "compliance_score": 85.0,
    "evidence_refs": ["EVD-001", "EVD-002"],
    "created_at": _ts(days_ago=365),
    "metadata": {"barrier_type": "wall", "zone_count": 4},
}

SCP_TRANSPORT_PALM: Dict[str, Any] = {
    "scp_id": SCP_ID_TRANSPORT_PALM,
    "facility_id": FAC_ID_MILL_ID,
    "location_lat": -0.5200,
    "location_lon": 109.5200,
    "scp_type": "transport",
    "commodity": "palm_oil",
    "capacity_kg": 30000.0,
    "segregation_method": "dedicated_facility",
    "status": "active",
    "last_verified": _ts(days_ago=15),
    "next_verification": _ts(days_ago=-75),
    "compliance_score": 95.0,
    "evidence_refs": ["EVD-003"],
    "created_at": _ts(days_ago=200),
    "metadata": {"dedicated_fleet": True},
}

SCP_PROCESSING_COFFEE: Dict[str, Any] = {
    "scp_id": SCP_ID_PROCESSING_COFFEE,
    "facility_id": FAC_ID_FACTORY_DE,
    "location_lat": 53.5511,
    "location_lon": 9.9937,
    "scp_type": "processing",
    "commodity": "coffee",
    "capacity_kg": 20000.0,
    "segregation_method": "temporal_separation",
    "status": "active",
    "last_verified": _ts(days_ago=45),
    "next_verification": _ts(days_ago=-45),
    "compliance_score": 70.0,
    "evidence_refs": ["EVD-004", "EVD-005"],
    "created_at": _ts(days_ago=180),
    "metadata": {"changeover_minutes": 60},
}

SCP_HANDLING_SOYA: Dict[str, Any] = {
    "scp_id": SCP_ID_HANDLING_SOYA,
    "facility_id": FAC_ID_PORT_BR,
    "location_lat": -23.9608,
    "location_lon": -46.3336,
    "scp_type": "handling",
    "commodity": "soya",
    "capacity_kg": 100000.0,
    "segregation_method": "separate_equipment",
    "status": "active",
    "last_verified": _ts(days_ago=60),
    "next_verification": _ts(days_ago=-30),
    "compliance_score": 78.0,
    "evidence_refs": [],
    "created_at": _ts(days_ago=400),
    "metadata": {},
}

SCP_LOADING_RUBBER: Dict[str, Any] = {
    "scp_id": SCP_ID_LOADING_RUBBER,
    "facility_id": FAC_ID_DEPOT_NL,
    "location_lat": 51.9225,
    "location_lon": 4.4792,
    "scp_type": "loading_unloading",
    "commodity": "rubber",
    "capacity_kg": 15000.0,
    "segregation_method": "color_coding",
    "status": "active",
    "last_verified": _ts(days_ago=100),
    "next_verification": _ts(days_ago=10),
    "compliance_score": 55.0,
    "evidence_refs": ["EVD-006"],
    "created_at": _ts(days_ago=500),
    "metadata": {"color_scheme": "green_for_compliant"},
}

ALL_SAMPLE_SCPS: List[Dict[str, Any]] = [
    SCP_STORAGE_COCOA, SCP_TRANSPORT_PALM, SCP_PROCESSING_COFFEE,
    SCP_HANDLING_SOYA, SCP_LOADING_RUBBER,
]


# ---------------------------------------------------------------------------
# Sample Storage Zones
# ---------------------------------------------------------------------------

ZONE_COCOA_A: Dict[str, Any] = {
    "zone_id": ZONE_ID_COCOA_A,
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "zone_name": "Cocoa Bay A",
    "storage_type": "warehouse_bay",
    "commodity": "cocoa",
    "capacity_kg": 25000.0,
    "current_quantity_kg": 18000.0,
    "barrier_type": "wall",
    "barrier_quality_score": 90.0,
    "status": "active",
    "adjacent_zones": [ZONE_ID_COCOA_B, ZONE_ID_MIXED_D],
    "last_cleaned": _ts(days_ago=7),
    "cleaning_method": "dry_sweep",
    "metadata": {"temperature_controlled": False},
}

ZONE_COCOA_B: Dict[str, Any] = {
    "zone_id": ZONE_ID_COCOA_B,
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "zone_name": "Cocoa Bay B",
    "storage_type": "warehouse_bay",
    "commodity": "cocoa",
    "capacity_kg": 25000.0,
    "current_quantity_kg": 5000.0,
    "barrier_type": "wall",
    "barrier_quality_score": 85.0,
    "status": "active",
    "adjacent_zones": [ZONE_ID_COCOA_A],
    "last_cleaned": _ts(days_ago=3),
    "cleaning_method": "vacuum",
    "metadata": {},
}

ZONE_PALM_C: Dict[str, Any] = {
    "zone_id": ZONE_ID_PALM_C,
    "facility_id": FAC_ID_MILL_ID,
    "zone_name": "Palm Oil Tank C",
    "storage_type": "tank",
    "commodity": "palm_oil",
    "capacity_kg": 50000.0,
    "current_quantity_kg": 35000.0,
    "barrier_type": "container",
    "barrier_quality_score": 95.0,
    "status": "active",
    "adjacent_zones": [],
    "last_cleaned": _ts(days_ago=14),
    "cleaning_method": "chemical_wash",
    "metadata": {"material": "stainless_steel"},
}

ALL_SAMPLE_ZONES: List[Dict[str, Any]] = [
    ZONE_COCOA_A, ZONE_COCOA_B, ZONE_PALM_C,
]


# ---------------------------------------------------------------------------
# Sample Transport Vehicles
# ---------------------------------------------------------------------------

VEHICLE_DEDICATED_TRUCK: Dict[str, Any] = {
    "vehicle_id": VEHICLE_ID_TRUCK_01,
    "vehicle_type": "truck",
    "dedicated": True,
    "dedicated_commodity": "cocoa",
    "capacity_kg": 20000.0,
    "last_cleaned": _ts(days_ago=2),
    "cleaning_method": "dry_sweep",
    "cleaning_duration_minutes": 45,
    "cleaning_certificate": "CERT-CLN-001",
    "seal_number": "SEAL-TRK-001",
    "seal_status": "intact",
    "previous_cargo": [
        {"commodity": "cocoa", "batch_id": "BATCH-PREV-001", "compliant": True},
    ],
    "status": "available",
    "metadata": {},
}

VEHICLE_SHARED_CONTAINER: Dict[str, Any] = {
    "vehicle_id": VEHICLE_ID_CONTAINER_01,
    "vehicle_type": "container",
    "dedicated": False,
    "dedicated_commodity": None,
    "capacity_kg": 28000.0,
    "last_cleaned": _ts(days_ago=5),
    "cleaning_method": "water_wash",
    "cleaning_duration_minutes": 90,
    "cleaning_certificate": "CERT-CLN-002",
    "seal_number": "SEAL-CNT-001",
    "seal_status": "intact",
    "previous_cargo": [
        {"commodity": "soya", "batch_id": "BATCH-PREV-002", "compliant": True},
        {"commodity": "coffee", "batch_id": "BATCH-PREV-003", "compliant": False},
    ],
    "status": "available",
    "metadata": {},
}

ALL_SAMPLE_VEHICLES: List[Dict[str, Any]] = [
    VEHICLE_DEDICATED_TRUCK, VEHICLE_SHARED_CONTAINER,
]


# ---------------------------------------------------------------------------
# Sample Processing Lines
# ---------------------------------------------------------------------------

LINE_COCOA_ROASTING: Dict[str, Any] = {
    "line_id": LINE_ID_ROASTING_01,
    "facility_id": FAC_ID_FACTORY_DE,
    "line_type": "roasting",
    "commodity": "cocoa",
    "dedicated": True,
    "capacity_kg_per_hour": 500.0,
    "last_changeover": _ts(days_ago=1),
    "changeover_duration_minutes": 45,
    "flush_quantity_kg": 8.0,
    "status": "active",
    "shared_equipment": [],
    "metadata": {},
}

LINE_PALM_PRESSING: Dict[str, Any] = {
    "line_id": LINE_ID_PRESSING_01,
    "facility_id": FAC_ID_MILL_ID,
    "line_type": "pressing",
    "commodity": "palm_oil",
    "dedicated": False,
    "capacity_kg_per_hour": 2000.0,
    "last_changeover": _ts(days_ago=3),
    "changeover_duration_minutes": 75,
    "flush_quantity_kg": 20.0,
    "status": "active",
    "shared_equipment": ["scale", "conveyor"],
    "metadata": {},
}

ALL_SAMPLE_LINES: List[Dict[str, Any]] = [
    LINE_COCOA_ROASTING, LINE_PALM_PRESSING,
]


# ---------------------------------------------------------------------------
# Sample Contamination Events
# ---------------------------------------------------------------------------

CONTAMINATION_SPATIAL: Dict[str, Any] = {
    "contamination_id": "CTM-SPAT-001",
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "pathway_type": "spatial_proximity",
    "severity": "medium",
    "detected_at": _ts(days_ago=5),
    "affected_batches": [BATCH_ID_COCOA_001],
    "source_batch_id": BATCH_ID_PALM_001,
    "source_zone_id": ZONE_ID_MIXED_D,
    "affected_zone_id": ZONE_ID_COCOA_A,
    "quantity_affected_kg": 500.0,
    "status": "open",
    "root_cause": None,
    "corrective_actions": [],
    "resolved_at": None,
    "metadata": {},
}

CONTAMINATION_TEMPORAL: Dict[str, Any] = {
    "contamination_id": "CTM-TEMP-001",
    "facility_id": FAC_ID_FACTORY_DE,
    "pathway_type": "temporal_proximity",
    "severity": "low",
    "detected_at": _ts(days_ago=10),
    "affected_batches": [BATCH_ID_COFFEE_001],
    "source_batch_id": BATCH_ID_COCOA_002,
    "source_zone_id": None,
    "affected_zone_id": None,
    "quantity_affected_kg": 100.0,
    "status": "resolved",
    "root_cause": "Insufficient changeover time between batches",
    "corrective_actions": ["Extended changeover to 60 minutes"],
    "resolved_at": _ts(days_ago=8),
    "metadata": {},
}

ALL_SAMPLE_CONTAMINATIONS: List[Dict[str, Any]] = [
    CONTAMINATION_SPATIAL, CONTAMINATION_TEMPORAL,
]


# ---------------------------------------------------------------------------
# Sample Labels
# ---------------------------------------------------------------------------

LABEL_BIN_TAG: Dict[str, Any] = {
    "label_id": LABEL_ID_BIN_01,
    "label_type": "bin_tag",
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "zone_id": ZONE_ID_COCOA_A,
    "batch_id": BATCH_ID_COCOA_001,
    "content_fields": {
        "batch_id": BATCH_ID_COCOA_001,
        "commodity": "cocoa",
        "origin": "Ghana - Ashanti",
        "date": _ts(days_ago=5),
    },
    "placement": "bin_front",
    "condition": "good",
    "color_code": "green",
    "applied_at": _ts(days_ago=5),
    "expires_at": _ts(days_ago=-85),
    "status": "active",
    "metadata": {},
}

LABEL_ZONE_SIGN: Dict[str, Any] = {
    "label_id": LABEL_ID_ZONE_01,
    "label_type": "zone_sign",
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "zone_id": ZONE_ID_COCOA_A,
    "batch_id": None,
    "content_fields": {
        "zone_id": ZONE_ID_COCOA_A,
        "commodity": "cocoa",
        "status": "EUDR Compliant",
    },
    "placement": "zone_entrance",
    "condition": "good",
    "color_code": "green",
    "applied_at": _ts(days_ago=60),
    "expires_at": _ts(days_ago=-305),
    "status": "active",
    "metadata": {},
}

ALL_SAMPLE_LABELS: List[Dict[str, Any]] = [
    LABEL_BIN_TAG, LABEL_ZONE_SIGN,
]


# ---------------------------------------------------------------------------
# Sample Facility Profiles
# ---------------------------------------------------------------------------

FACILITY_PROFILE_COCOA_WAREHOUSE: Dict[str, Any] = {
    "facility_id": FAC_ID_WAREHOUSE_GH,
    "facility_name": "Accra Cocoa Warehouse",
    "facility_type": "warehouse",
    "commodities": ["cocoa"],
    "coc_models": ["SG", "IP"],
    "zone_count": 4,
    "total_capacity_kg": 100000.0,
    "segregation_methods": ["physical_barrier", "color_coding"],
    "certifications": ["UTZ"],
    "last_assessment_date": _ts(days_ago=90),
    "capability_level": 3,
    "sop_count": 12,
    "training_records": 8,
    "cleaning_schedule": "weekly",
    "incident_history": [
        {"date": _ts(days_ago=180), "type": "minor_spill", "resolved": True},
    ],
    "documentation_score": 75.0,
    "metadata": {},
}

FACILITY_PROFILE_PALM_MILL: Dict[str, Any] = {
    "facility_id": FAC_ID_MILL_ID,
    "facility_name": "Pontianak Palm Mill",
    "facility_type": "processing_mill",
    "commodities": ["palm_oil"],
    "coc_models": ["MB"],
    "zone_count": 6,
    "total_capacity_kg": 500000.0,
    "segregation_methods": ["dedicated_facility", "separate_equipment"],
    "certifications": ["RSPO", "ISCC"],
    "last_assessment_date": _ts(days_ago=45),
    "capability_level": 4,
    "sop_count": 25,
    "training_records": 20,
    "cleaning_schedule": "daily",
    "incident_history": [],
    "documentation_score": 90.0,
    "metadata": {},
}

ALL_SAMPLE_PROFILES: List[Dict[str, Any]] = [
    FACILITY_PROFILE_COCOA_WAREHOUSE, FACILITY_PROFILE_PALM_MILL,
]


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_scp(
    scp_type: str = "storage",
    commodity: str = "cocoa",
    segregation_method: str = "physical_barrier",
    facility_id: str = FAC_ID_WAREHOUSE_GH,
    capacity_kg: float = 50000.0,
    compliance_score: float = 80.0,
    scp_id: Optional[str] = None,
    status: str = "active",
    last_verified_days_ago: int = 30,
) -> Dict[str, Any]:
    """Build a segregation control point dictionary for testing.

    Args:
        scp_type: One of the 5 SCP types.
        commodity: EUDR commodity type.
        segregation_method: One of the 8 segregation methods.
        facility_id: Facility identifier.
        capacity_kg: Maximum capacity in kilograms.
        compliance_score: Current compliance score (0-100).
        scp_id: SCP identifier (auto-generated if None).
        status: SCP status (active, inactive, expired).
        last_verified_days_ago: Days since last verification.

    Returns:
        Dict with all SCP fields.
    """
    return {
        "scp_id": scp_id or f"SCP-{uuid.uuid4().hex[:12].upper()}",
        "facility_id": facility_id,
        "location_lat": 5.5571,
        "location_lon": -0.2013,
        "scp_type": scp_type,
        "commodity": commodity,
        "capacity_kg": capacity_kg,
        "segregation_method": segregation_method,
        "status": status,
        "last_verified": _ts(days_ago=last_verified_days_ago),
        "next_verification": _ts(days_ago=-(DEFAULT_REVERIFICATION_DAYS - last_verified_days_ago)),
        "compliance_score": compliance_score,
        "evidence_refs": [],
        "created_at": _ts(days_ago=365),
        "metadata": {},
    }


def make_zone(
    storage_type: str = "warehouse_bay",
    commodity: str = "cocoa",
    facility_id: str = FAC_ID_WAREHOUSE_GH,
    capacity_kg: float = 25000.0,
    current_quantity_kg: float = 10000.0,
    barrier_type: str = "wall",
    barrier_quality_score: float = 85.0,
    zone_id: Optional[str] = None,
    adjacent_zones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a storage zone dictionary for testing.

    Args:
        storage_type: One of the 12 storage types.
        commodity: EUDR commodity type.
        facility_id: Facility identifier.
        capacity_kg: Maximum capacity in kilograms.
        current_quantity_kg: Current stored quantity.
        barrier_type: Physical barrier type.
        barrier_quality_score: Barrier quality (0-100).
        zone_id: Zone identifier (auto-generated if None).
        adjacent_zones: List of adjacent zone IDs.

    Returns:
        Dict with all storage zone fields.
    """
    return {
        "zone_id": zone_id or f"ZONE-{uuid.uuid4().hex[:8].upper()}",
        "facility_id": facility_id,
        "zone_name": f"Test Zone {storage_type}",
        "storage_type": storage_type,
        "commodity": commodity,
        "capacity_kg": capacity_kg,
        "current_quantity_kg": current_quantity_kg,
        "barrier_type": barrier_type,
        "barrier_quality_score": barrier_quality_score,
        "status": "active",
        "adjacent_zones": adjacent_zones or [],
        "last_cleaned": _ts(days_ago=7),
        "cleaning_method": "dry_sweep",
        "metadata": {},
    }


def make_vehicle(
    vehicle_type: str = "truck",
    dedicated: bool = True,
    dedicated_commodity: Optional[str] = "cocoa",
    capacity_kg: float = 20000.0,
    vehicle_id: Optional[str] = None,
    cleaning_method: str = "dry_sweep",
    cleaning_duration_minutes: int = 45,
    seal_status: str = "intact",
    previous_cargo: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a transport vehicle dictionary for testing.

    Args:
        vehicle_type: One of the 10 transport types.
        dedicated: Whether vehicle is dedicated to one commodity.
        dedicated_commodity: Commodity if dedicated.
        capacity_kg: Vehicle cargo capacity.
        vehicle_id: Vehicle identifier (auto-generated if None).
        cleaning_method: Last cleaning method used.
        cleaning_duration_minutes: Duration of last cleaning.
        seal_status: Seal integrity status.
        previous_cargo: List of previous cargo records.

    Returns:
        Dict with all vehicle fields.
    """
    return {
        "vehicle_id": vehicle_id or f"VEH-{uuid.uuid4().hex[:8].upper()}",
        "vehicle_type": vehicle_type,
        "dedicated": dedicated,
        "dedicated_commodity": dedicated_commodity if dedicated else None,
        "capacity_kg": capacity_kg,
        "last_cleaned": _ts(days_ago=2),
        "cleaning_method": cleaning_method,
        "cleaning_duration_minutes": cleaning_duration_minutes,
        "cleaning_certificate": f"CERT-CLN-{uuid.uuid4().hex[:6].upper()}",
        "seal_number": f"SEAL-{uuid.uuid4().hex[:6].upper()}",
        "seal_status": seal_status,
        "previous_cargo": previous_cargo or [],
        "status": "available",
        "metadata": {},
    }


def make_line(
    line_type: str = "roasting",
    commodity: str = "cocoa",
    facility_id: str = FAC_ID_FACTORY_DE,
    dedicated: bool = True,
    capacity_kg_per_hour: float = 500.0,
    line_id: Optional[str] = None,
    shared_equipment: Optional[List[str]] = None,
    changeover_duration_minutes: Optional[int] = None,
    flush_quantity_kg: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a processing line dictionary for testing.

    Args:
        line_type: One of the 15 processing line types.
        commodity: EUDR commodity type.
        facility_id: Facility identifier.
        dedicated: Whether line is dedicated to one commodity.
        capacity_kg_per_hour: Processing throughput.
        line_id: Line identifier (auto-generated if None).
        shared_equipment: List of shared equipment types.
        changeover_duration_minutes: Override changeover duration.
        flush_quantity_kg: Override flush quantity.

    Returns:
        Dict with all processing line fields.
    """
    req = CHANGEOVER_REQUIREMENTS.get(line_type, {"min_minutes": 30, "flush_required": False, "flush_kg": 0.0})
    return {
        "line_id": line_id or f"LINE-{uuid.uuid4().hex[:8].upper()}",
        "facility_id": facility_id,
        "line_type": line_type,
        "commodity": commodity,
        "dedicated": dedicated,
        "capacity_kg_per_hour": capacity_kg_per_hour,
        "last_changeover": _ts(days_ago=1),
        "changeover_duration_minutes": changeover_duration_minutes or req["min_minutes"],
        "flush_quantity_kg": flush_quantity_kg if flush_quantity_kg is not None else req["flush_kg"],
        "status": "active",
        "shared_equipment": shared_equipment or [],
        "metadata": {},
    }


def make_contamination(
    pathway_type: str = "spatial_proximity",
    severity: str = "medium",
    facility_id: str = FAC_ID_WAREHOUSE_GH,
    affected_batches: Optional[List[str]] = None,
    source_batch_id: str = BATCH_ID_PALM_001,
    quantity_affected_kg: float = 500.0,
    contamination_id: Optional[str] = None,
    status: str = "open",
) -> Dict[str, Any]:
    """Build a contamination event dictionary for testing.

    Args:
        pathway_type: One of the 10 contamination pathways.
        severity: One of 4 severity levels.
        facility_id: Facility where contamination occurred.
        affected_batches: List of affected batch IDs.
        source_batch_id: Source batch causing contamination.
        quantity_affected_kg: Quantity affected in kilograms.
        contamination_id: Contamination identifier (auto-generated if None).
        status: Event status (open, investigating, resolved).

    Returns:
        Dict with all contamination event fields.
    """
    return {
        "contamination_id": contamination_id or f"CTM-{uuid.uuid4().hex[:8].upper()}",
        "facility_id": facility_id,
        "pathway_type": pathway_type,
        "severity": severity,
        "detected_at": _ts(days_ago=1),
        "affected_batches": affected_batches or [BATCH_ID_COCOA_001],
        "source_batch_id": source_batch_id,
        "source_zone_id": ZONE_ID_MIXED_D,
        "affected_zone_id": ZONE_ID_COCOA_A,
        "quantity_affected_kg": quantity_affected_kg,
        "status": status,
        "root_cause": None,
        "corrective_actions": [],
        "resolved_at": None,
        "metadata": {},
    }


def make_label(
    label_type: str = "bin_tag",
    facility_id: str = FAC_ID_WAREHOUSE_GH,
    zone_id: str = ZONE_ID_COCOA_A,
    batch_id: Optional[str] = BATCH_ID_COCOA_001,
    color_code: str = "green",
    label_id: Optional[str] = None,
    status: str = "active",
    content_fields: Optional[Dict[str, str]] = None,
    days_until_expiry: int = 85,
) -> Dict[str, Any]:
    """Build a label dictionary for testing.

    Args:
        label_type: One of the 8 label types.
        facility_id: Facility identifier.
        zone_id: Zone where label is placed.
        batch_id: Associated batch (optional for zone signs).
        color_code: Color code for segregation.
        label_id: Label identifier (auto-generated if None).
        status: Label status.
        content_fields: Override content fields.
        days_until_expiry: Days until label expires.

    Returns:
        Dict with all label fields.
    """
    required = LABEL_REQUIRED_FIELDS.get(label_type, [])
    if content_fields is None:
        content_fields = {}
        for field in required:
            if field == "batch_id":
                content_fields[field] = batch_id or "N/A"
            elif field == "commodity":
                content_fields[field] = "cocoa"
            elif field == "origin":
                content_fields[field] = "Ghana - Ashanti"
            elif field == "date":
                content_fields[field] = _ts(days_ago=5)
            elif field == "zone_id":
                content_fields[field] = zone_id
            elif field == "status":
                content_fields[field] = "EUDR Compliant"
            elif field == "quantity":
                content_fields[field] = "5000 kg"
            elif field == "container_id":
                content_fields[field] = "CNTR-001"
            elif field == "seal_number":
                content_fields[field] = "SEAL-001"
            elif field == "vehicle_id":
                content_fields[field] = "VEH-001"
            elif field == "route":
                content_fields[field] = "Accra-Rotterdam"
            elif field == "weight":
                content_fields[field] = "60 kg"
            elif field == "tag_id":
                content_fields[field] = "RFID-001"
            elif field == "timestamp":
                content_fields[field] = _ts()
            elif field == "last_updated":
                content_fields[field] = _ts()
            else:
                content_fields[field] = f"test-{field}"
    return {
        "label_id": label_id or f"LBL-{uuid.uuid4().hex[:8].upper()}",
        "label_type": label_type,
        "facility_id": facility_id,
        "zone_id": zone_id,
        "batch_id": batch_id,
        "content_fields": content_fields,
        "placement": "standard",
        "condition": "good",
        "color_code": color_code,
        "applied_at": _ts(days_ago=5),
        "expires_at": _ts(days_ago=-days_until_expiry),
        "status": status,
        "metadata": {},
    }


def make_facility_profile(
    facility_id: str = FAC_ID_WAREHOUSE_GH,
    facility_type: str = "warehouse",
    commodities: Optional[List[str]] = None,
    capability_level: int = 3,
    documentation_score: float = 75.0,
    sop_count: int = 12,
    incident_count: int = 0,
) -> Dict[str, Any]:
    """Build a facility profile dictionary for testing.

    Args:
        facility_id: Facility identifier.
        facility_type: Type of facility.
        commodities: List of commodities handled.
        capability_level: Current capability level (0-5).
        documentation_score: Documentation completeness score.
        sop_count: Number of SOPs.
        incident_count: Number of past incidents.

    Returns:
        Dict with all facility profile fields.
    """
    incidents = [
        {"date": _ts(days_ago=180 + i * 90), "type": "minor_spill", "resolved": True}
        for i in range(incident_count)
    ]
    return {
        "facility_id": facility_id,
        "facility_name": f"Test Facility {facility_id}",
        "facility_type": facility_type,
        "commodities": commodities or ["cocoa"],
        "coc_models": ["SG"],
        "zone_count": 4,
        "total_capacity_kg": 100000.0,
        "segregation_methods": ["physical_barrier"],
        "certifications": [],
        "last_assessment_date": _ts(days_ago=90),
        "capability_level": capability_level,
        "sop_count": sop_count,
        "training_records": sop_count,
        "cleaning_schedule": "weekly",
        "incident_history": incidents,
        "documentation_score": documentation_score,
        "metadata": {},
    }


def make_storage_event(
    event_type: str = "material_in",
    zone_id: str = ZONE_ID_COCOA_A,
    batch_id: str = BATCH_ID_COCOA_001,
    quantity_kg: float = 5000.0,
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a storage event dictionary for testing.

    Args:
        event_type: One of the 5 storage event types.
        zone_id: Zone where the event occurred.
        batch_id: Associated batch identifier.
        quantity_kg: Event quantity in kilograms.
        event_id: Event identifier (auto-generated if None).

    Returns:
        Dict with all storage event fields.
    """
    return {
        "event_id": event_id or f"SEVT-{uuid.uuid4().hex[:8].upper()}",
        "event_type": event_type,
        "zone_id": zone_id,
        "batch_id": batch_id,
        "quantity_kg": quantity_kg,
        "timestamp": _ts(),
        "performed_by": "operator-001",
        "notes": "",
        "metadata": {},
    }


def make_changeover(
    line_id: str = LINE_ID_ROASTING_01,
    from_commodity: str = "cocoa",
    to_commodity: str = "coffee",
    duration_minutes: int = 45,
    flush_kg: float = 8.0,
    changeover_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a changeover record dictionary for testing.

    Args:
        line_id: Processing line identifier.
        from_commodity: Previous commodity.
        to_commodity: Next commodity.
        duration_minutes: Changeover duration in minutes.
        flush_kg: Flush material quantity in kilograms.
        changeover_id: Changeover identifier (auto-generated if None).

    Returns:
        Dict with all changeover fields.
    """
    return {
        "changeover_id": changeover_id or f"CHG-{uuid.uuid4().hex[:8].upper()}",
        "line_id": line_id,
        "from_commodity": from_commodity,
        "to_commodity": to_commodity,
        "started_at": _ts(hours_ago=2),
        "completed_at": _ts(hours_ago=1),
        "duration_minutes": duration_minutes,
        "flush_quantity_kg": flush_kg,
        "flush_material": "inert_material",
        "cleaning_method": "dry_sweep",
        "verified_by": "supervisor-001",
        "compliant": True,
        "notes": "",
        "metadata": {},
    }


def make_label_event(
    label_id: str = LABEL_ID_BIN_01,
    event_type: str = "applied",
    event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a label event dictionary for testing.

    Args:
        label_id: Label identifier.
        event_type: One of the 5 label event types.
        event_id: Event identifier (auto-generated if None).

    Returns:
        Dict with all label event fields.
    """
    return {
        "event_id": event_id or f"LEVT-{uuid.uuid4().hex[:8].upper()}",
        "label_id": label_id,
        "event_type": event_type,
        "timestamp": _ts(),
        "performed_by": "operator-001",
        "notes": "",
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# Assertion Helpers
# ---------------------------------------------------------------------------

def compute_sha256(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance verification.

    Args:
        data: Data to hash (will be JSON-serialized).

    Returns:
        64-character hex digest string.
    """
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_valid_provenance_hash(hash_value: str) -> None:
    """Assert that a provenance hash is a valid SHA-256 hex digest.

    Args:
        hash_value: The hash string to validate.

    Raises:
        AssertionError: If hash is not a valid 64-char hex string.
    """
    assert isinstance(hash_value, str), f"Hash must be string, got {type(hash_value)}"
    assert len(hash_value) == SHA256_HEX_LENGTH, (
        f"Hash length must be {SHA256_HEX_LENGTH}, got {len(hash_value)}"
    )
    assert all(c in "0123456789abcdef" for c in hash_value), (
        "Hash must be lowercase hex characters only"
    )


def assert_valid_score(score: float, min_val: float = 0.0, max_val: float = 100.0) -> None:
    """Assert that a score is within valid bounds.

    Args:
        score: The score value to validate.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Raises:
        AssertionError: If score is out of bounds.
    """
    assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"
    assert min_val <= score <= max_val, f"Score {score} out of bounds [{min_val}, {max_val}]"


def assert_valid_compliance_status(status: str) -> None:
    """Assert that a compliance status is one of the expected values.

    Args:
        status: The status string to validate.

    Raises:
        AssertionError: If status is not a valid compliance status.
    """
    valid = {"compliant", "non_compliant", "partial", "pending", "expired"}
    assert status in valid, f"Invalid compliance status: {status}. Expected one of {valid}"


def assert_valid_capability_level(level: int) -> None:
    """Assert that a capability level is within 0-5.

    Args:
        level: The capability level to validate.

    Raises:
        AssertionError: If level is not 0-5.
    """
    assert isinstance(level, int), f"Level must be int, got {type(level)}"
    assert 0 <= level <= 5, f"Capability level {level} out of bounds [0, 5]"


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sgv_config() -> Dict[str, Any]:
    """Create a SegregationVerifierConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-SGV-010-TEST-GENESIS",
        "enable_metrics": False,
        "pool_size": 5,
        "reverification_interval_days": DEFAULT_REVERIFICATION_DAYS,
        "scp_score_weights": dict(SCP_SCORE_WEIGHTS),
        "storage_score_weights": dict(STORAGE_SCORE_WEIGHTS),
        "transport_score_weights": dict(TRANSPORT_SCORE_WEIGHTS),
        "method_risk_levels": dict(METHOD_RISK_LEVELS),
        "pathway_risk_weights": dict(PATHWAY_RISK_WEIGHTS),
        "severity_scores": dict(SEVERITY_SCORES),
        "changeover_requirements": dict(CHANGEOVER_REQUIREMENTS),
        "equipment_sharing_weights": dict(EQUIPMENT_SHARING_WEIGHTS),
        "label_required_fields": dict(LABEL_REQUIRED_FIELDS),
        "assessment_weights": dict(ASSESSMENT_WEIGHTS),
        "capability_levels": list(CAPABILITY_LEVELS),
        "certification_standards": list(CERTIFICATION_STANDARDS),
        "contamination_temporal_threshold_hours": 4,
        "contamination_spatial_threshold_meters": 10,
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.segregation_verifier.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def segregation_point_validator(sgv_config):
    """Create a SegregationPointValidator instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.segregation_point_validator import (
            SegregationPointValidator,
        )
        return SegregationPointValidator(config=sgv_config)
    except ImportError:
        pytest.skip("SegregationPointValidator not yet implemented")


@pytest.fixture
def storage_segregation_auditor(sgv_config):
    """Create a StorageSegregationAuditor instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.storage_segregation_auditor import (
            StorageSegregationAuditor,
        )
        return StorageSegregationAuditor(config=sgv_config)
    except ImportError:
        pytest.skip("StorageSegregationAuditor not yet implemented")


@pytest.fixture
def transport_segregation_tracker(sgv_config):
    """Create a TransportSegregationTracker instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.transport_segregation_tracker import (
            TransportSegregationTracker,
        )
        return TransportSegregationTracker(config=sgv_config)
    except ImportError:
        pytest.skip("TransportSegregationTracker not yet implemented")


@pytest.fixture
def processing_line_verifier(sgv_config):
    """Create a ProcessingLineVerifier instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.processing_line_verifier import (
            ProcessingLineVerifier,
        )
        return ProcessingLineVerifier(config=sgv_config)
    except ImportError:
        pytest.skip("ProcessingLineVerifier not yet implemented")


@pytest.fixture
def cross_contamination_detector(sgv_config):
    """Create a CrossContaminationDetector instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.cross_contamination_detector import (
            CrossContaminationDetector,
        )
        return CrossContaminationDetector(config=sgv_config)
    except ImportError:
        pytest.skip("CrossContaminationDetector not yet implemented")


@pytest.fixture
def labeling_verification_engine(sgv_config):
    """Create a LabelingVerificationEngine instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.labeling_verification_engine import (
            LabelingVerificationEngine,
        )
        return LabelingVerificationEngine(config=sgv_config)
    except ImportError:
        pytest.skip("LabelingVerificationEngine not yet implemented")


@pytest.fixture
def facility_assessment_engine(sgv_config):
    """Create a FacilityAssessmentEngine instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.facility_assessment_engine import (
            FacilityAssessmentEngine,
        )
        return FacilityAssessmentEngine(config=sgv_config)
    except ImportError:
        pytest.skip("FacilityAssessmentEngine not yet implemented")


@pytest.fixture
def compliance_reporter(sgv_config):
    """Create a ComplianceReporter instance for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.compliance_reporter import (
            ComplianceReporter,
        )
        return ComplianceReporter(config=sgv_config)
    except ImportError:
        pytest.skip("ComplianceReporter not yet implemented")


@pytest.fixture
def service(sgv_config):
    """Create the top-level SegregationVerifierService facade for testing."""
    try:
        from greenlang.agents.eudr.segregation_verifier.setup import (
            SegregationVerifierService,
        )
        return SegregationVerifierService(config=sgv_config)
    except ImportError:
        pytest.skip("SegregationVerifierService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_scp() -> Dict[str, Any]:
    """Return a sample segregation control point."""
    return copy.deepcopy(SCP_STORAGE_COCOA)


@pytest.fixture
def sample_storage_zone() -> Dict[str, Any]:
    """Return a sample storage zone."""
    return copy.deepcopy(ZONE_COCOA_A)


@pytest.fixture
def sample_transport_vehicle() -> Dict[str, Any]:
    """Return a sample dedicated transport vehicle."""
    return copy.deepcopy(VEHICLE_DEDICATED_TRUCK)


@pytest.fixture
def sample_processing_line() -> Dict[str, Any]:
    """Return a sample processing line."""
    return copy.deepcopy(LINE_COCOA_ROASTING)


@pytest.fixture
def sample_contamination_event() -> Dict[str, Any]:
    """Return a sample contamination event."""
    return copy.deepcopy(CONTAMINATION_SPATIAL)


@pytest.fixture
def sample_label() -> Dict[str, Any]:
    """Return a sample bin tag label."""
    return copy.deepcopy(LABEL_BIN_TAG)


@pytest.fixture
def sample_facility_profile() -> Dict[str, Any]:
    """Return a sample facility profile."""
    return copy.deepcopy(FACILITY_PROFILE_COCOA_WAREHOUSE)


@pytest.fixture(params=EUDR_COMMODITIES)
def commodity(request) -> str:
    """Parametrize across all 7 EUDR commodities."""
    return request.param


@pytest.fixture(params=SCP_TYPES)
def scp_type(request) -> str:
    """Parametrize across all 5 SCP types."""
    return request.param


@pytest.fixture(params=SEGREGATION_METHODS)
def segregation_method(request) -> str:
    """Parametrize across all 8 segregation methods."""
    return request.param
