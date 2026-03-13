# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-009 Chain of Custody Agent test suite.

Provides reusable fixtures for sample custody events, batches, batch
genealogy trees, CoC model assignments, mass balance ledger data,
transformation chains, document types, helper factories, chain builders,
assertion helpers, reference data constants, and engine fixtures.

Sample Custody Events (20+ across all 10 event types):
    TRANSFER_COCOA_GH_NL, RECEIPT_COCOA_NL, STORAGE_IN_COCOA_NL,
    STORAGE_OUT_COCOA_NL, PROCESSING_IN_COCOA_NL, PROCESSING_OUT_COCOA_NL,
    EXPORT_COCOA_GH, IMPORT_COCOA_NL, INSPECTION_COCOA_NL, SAMPLING_COCOA_NL

Sample Batches (15+ across 7 commodities):
    BATCH_COCOA_FARM_GH, BATCH_COCOA_COOP_GH, BATCH_COFFEE_FARM_CO,
    BATCH_PALM_MILL_ID, BATCH_SOYA_FARM_BR, BATCH_RUBBER_PLOT_TH,
    BATCH_WOOD_FOREST_CD, BATCH_CATTLE_RANCH_BR, etc.

Sample Genealogy Trees (10+ topologies):
    Linear, 2-way split, 3-way split, 2-batch merge, N-batch merge,
    blend, diamond, multi-step transform, deep chain, complex hybrid.

Helper Factories: make_event(), make_batch(), make_transformation(),
    make_document(), make_mass_balance_entry()

Chain Builders: build_cocoa_chain(), build_palm_oil_chain(),
    build_coffee_chain()

Assertion Helpers: assert_valid_chain(), assert_mass_conservation(),
    assert_origin_preserved()

Reference Data Constants: CONVERSION_FACTORS, DOCUMENT_REQUIREMENTS,
    COC_MODEL_RULES, EUDR_COMMODITIES

Engine Fixtures (8 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64

EUDR_COMMODITIES: List[str] = [
    "cocoa", "coffee", "palm_oil", "soya", "rubber", "cattle", "wood",
]

EVENT_TYPES: List[str] = [
    "transfer", "receipt", "storage_in", "storage_out",
    "processing_in", "processing_out", "export", "import",
    "inspection", "sampling",
]

BATCH_STATUSES: List[str] = [
    "created", "in_transit", "at_facility", "processing",
    "processed", "dispatched", "delivered", "consumed",
]

VALID_BATCH_TRANSITIONS: List[Tuple[str, str]] = [
    ("created", "in_transit"),
    ("created", "at_facility"),
    ("in_transit", "at_facility"),
    ("at_facility", "processing"),
    ("at_facility", "dispatched"),
    ("processing", "processed"),
    ("processed", "dispatched"),
    ("dispatched", "in_transit"),
    ("dispatched", "delivered"),
    ("delivered", "consumed"),
]

INVALID_BATCH_TRANSITIONS: List[Tuple[str, str]] = [
    ("consumed", "created"),
    ("consumed", "in_transit"),
    ("delivered", "processing"),
    ("created", "consumed"),
    ("processing", "dispatched"),
    ("in_transit", "processing"),
    ("processed", "in_transit"),
    ("consumed", "at_facility"),
]

COC_MODELS: List[str] = ["IP", "SG", "MB", "CB"]

COC_MODEL_LABELS: Dict[str, str] = {
    "IP": "Identity Preserved",
    "SG": "Segregated",
    "MB": "Mass Balance",
    "CB": "Controlled Blending",
}

DOCUMENT_TYPES: List[str] = [
    "bill_of_lading", "packing_list", "commercial_invoice",
    "certificate_of_origin", "phytosanitary_cert", "weight_cert",
    "quality_cert", "customs_declaration", "transport_waybill",
    "warehouse_receipt", "fumigation_cert", "insurance_cert",
    "dds_reference", "delivery_note", "purchase_order",
]

PROCESS_TYPES: List[str] = [
    "drying", "fermentation", "roasting", "milling", "refining",
    "pressing", "extraction", "fractionation", "deodorization",
    "hydrogenation", "sawing", "tanning", "spinning", "weaving",
    "slaughtering",
]

ALLOCATION_METHODS: List[str] = ["economic", "mass", "energy"]

REPORT_FORMATS: List[str] = ["json", "csv", "pdf", "eudr_xml"]


# ---------------------------------------------------------------------------
# Conversion Factors (Appendix A from PRD)
# ---------------------------------------------------------------------------

CONVERSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cocoa_beans_to_nibs": {"yield_ratio": 0.87, "commodity": "cocoa"},
    "cocoa_nibs_to_liquor": {"yield_ratio": 0.80, "commodity": "cocoa"},
    "cocoa_liquor_to_butter": {"yield_ratio": 0.45, "commodity": "cocoa"},
    "cocoa_liquor_to_powder": {"yield_ratio": 0.55, "commodity": "cocoa"},
    "palm_ffb_to_cpo": {"yield_ratio": 0.22, "commodity": "palm_oil"},
    "palm_ffb_to_pko": {"yield_ratio": 0.035, "commodity": "palm_oil"},
    "palm_cpo_to_rbd": {"yield_ratio": 0.92, "commodity": "palm_oil"},
    "coffee_cherry_to_green": {"yield_ratio": 0.18, "commodity": "coffee"},
    "coffee_green_to_roasted": {"yield_ratio": 0.82, "commodity": "coffee"},
    "soya_beans_to_oil": {"yield_ratio": 0.19, "commodity": "soya"},
    "soya_beans_to_meal": {"yield_ratio": 0.80, "commodity": "soya"},
    "rubber_latex_to_sheet": {"yield_ratio": 0.32, "commodity": "rubber"},
    "wood_log_to_sawn": {"yield_ratio": 0.50, "commodity": "wood"},
    "cattle_live_to_carcass": {"yield_ratio": 0.55, "commodity": "cattle"},
    "cattle_live_to_hide": {"yield_ratio": 0.07, "commodity": "cattle"},
}

# Acceptable loss tolerance per commodity (as fraction)
LOSS_TOLERANCES: Dict[str, float] = {
    "cocoa": 0.02,
    "coffee": 0.02,
    "palm_oil": 0.03,
    "soya": 0.01,
    "rubber": 0.03,
    "wood": 0.05,
    "cattle": 0.02,
}


# ---------------------------------------------------------------------------
# Required Documents per Event Type (Appendix B from PRD)
# ---------------------------------------------------------------------------

REQUIRED_DOCUMENTS: Dict[str, List[str]] = {
    "transfer": ["commercial_invoice", "delivery_note"],
    "export": [
        "bill_of_lading", "phytosanitary_cert",
        "certificate_of_origin", "customs_declaration",
    ],
    "import": ["bill_of_lading", "customs_declaration"],
    "processing_in": ["weight_cert", "quality_cert"],
    "processing_out": ["weight_cert"],
    "storage_in": ["warehouse_receipt", "weight_cert"],
    "storage_out": ["delivery_note", "weight_cert"],
    "inspection": ["quality_cert"],
    "receipt": ["delivery_note"],
    "sampling": [],
}

OPTIONAL_DOCUMENTS: Dict[str, List[str]] = {
    "transfer": ["transport_waybill"],
    "export": ["fumigation_cert", "insurance_cert"],
    "import": ["commercial_invoice"],
    "processing_in": ["warehouse_receipt"],
    "processing_out": ["quality_cert"],
    "storage_in": [],
    "storage_out": [],
    "inspection": [],
    "receipt": ["weight_cert"],
    "sampling": ["quality_cert"],
}


# ---------------------------------------------------------------------------
# CoC Model Rules (Appendix C from PRD)
# ---------------------------------------------------------------------------

COC_MODEL_RULES: Dict[str, Dict[str, Any]] = {
    "IP": {
        "mixing_allowed": False,
        "accounting_type": "physical",
        "credit_period_months": None,
        "certifications": ["FSC", "RSPO"],
        "description": "100% single origin, no mixing",
    },
    "SG": {
        "mixing_allowed": "compliant_only",
        "accounting_type": "physical",
        "credit_period_months": None,
        "certifications": ["FSC", "RSPO", "ISCC"],
        "description": "Compliant material kept separate from non-compliant",
    },
    "MB": {
        "mixing_allowed": True,
        "accounting_type": "accounting",
        "credit_period_months": 12,
        "certifications": ["RSPO", "ISCC", "UTZ"],
        "description": "Physical mixing OK, accounting-based tracking",
    },
    "CB": {
        "mixing_allowed": True,
        "accounting_type": "ratio",
        "credit_period_months": None,
        "certifications": ["RSPO"],
        "description": "Defined maximum blend ratio",
        "default_max_blend_ratio": 0.50,
    },
}

# Allowed CoC model transitions (can only downgrade, not upgrade)
VALID_MODEL_TRANSITIONS: List[Tuple[str, str]] = [
    ("IP", "SG"),
    ("IP", "MB"),
    ("IP", "CB"),
    ("SG", "MB"),
    ("SG", "CB"),
    ("MB", "CB"),
]

INVALID_MODEL_TRANSITIONS: List[Tuple[str, str]] = [
    ("SG", "IP"),
    ("MB", "IP"),
    ("MB", "SG"),
    ("CB", "IP"),
    ("CB", "SG"),
    ("CB", "MB"),
]

# Chain integrity risk category weights (Appendix D from PRD)
RISK_CATEGORY_WEIGHTS: Dict[str, float] = {
    "custody_gap": 0.25,
    "document_gap": 0.20,
    "mass_variance": 0.20,
    "origin_coverage": 0.20,
    "actor_verification": 0.15,
}

# Credit period lengths per certification standard
CREDIT_PERIODS: Dict[str, int] = {
    "RSPO": 3,    # months
    "FSC": 12,    # months
    "ISCC": 12,   # months
    "UTZ": 12,    # months
}


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

# Batch IDs
BATCH_ID_COCOA_FARM_GH = "BATCH-COC-FARM-GH-001"
BATCH_ID_COCOA_COOP_GH = "BATCH-COC-COOP-GH-001"
BATCH_ID_COCOA_PROC_GH = "BATCH-COC-PROC-GH-001"
BATCH_ID_COCOA_NIBS_GH = "BATCH-COC-NIBS-GH-001"
BATCH_ID_COCOA_LIQUOR_GH = "BATCH-COC-LIQR-GH-001"
BATCH_ID_COCOA_BUTTER_GH = "BATCH-COC-BUTR-GH-001"
BATCH_ID_COCOA_POWDER_GH = "BATCH-COC-PWDR-GH-001"
BATCH_ID_COFFEE_FARM_CO = "BATCH-COF-FARM-CO-001"
BATCH_ID_COFFEE_GREEN_CO = "BATCH-COF-GREN-CO-001"
BATCH_ID_PALM_MILL_ID = "BATCH-PLM-MILL-ID-001"
BATCH_ID_PALM_CPO_ID = "BATCH-PLM-CPO-ID-001"
BATCH_ID_SOYA_FARM_BR = "BATCH-SOY-FARM-BR-001"
BATCH_ID_RUBBER_PLOT_TH = "BATCH-RUB-PLOT-TH-001"
BATCH_ID_WOOD_FOREST_CD = "BATCH-WOD-FRST-CD-001"
BATCH_ID_CATTLE_RANCH_BR = "BATCH-CTL-RNCH-BR-001"

# Facility IDs
FAC_ID_COOP_GH = "FAC-COOP-GH-001"
FAC_ID_PROC_GH = "FAC-PROC-GH-001"
FAC_ID_WAREHOUSE_NL = "FAC-WRHS-NL-001"
FAC_ID_FACTORY_DE = "FAC-FACT-DE-001"
FAC_ID_MILL_ID = "FAC-MILL-ID-001"
FAC_ID_REFINERY_ID = "FAC-REFN-ID-001"
FAC_ID_MILL_CO = "FAC-MILL-CO-001"
FAC_ID_SILO_BR = "FAC-SILO-BR-001"
FAC_ID_SAWMILL_CD = "FAC-SAWM-CD-001"
FAC_ID_FEEDLOT_BR = "FAC-FEED-BR-001"

# Actor IDs
ACTOR_FARMER_GH = "ACT-FARMER-GH-001"
ACTOR_COOP_GH = "ACT-COOP-GH-001"
ACTOR_TRADER_GH = "ACT-TRADER-GH-001"
ACTOR_SHIPPER_INT = "ACT-SHIP-INT-001"
ACTOR_IMPORTER_NL = "ACT-IMP-NL-001"
ACTOR_PROCESSOR_DE = "ACT-PROC-DE-001"
ACTOR_MILL_ID = "ACT-MILL-ID-001"
ACTOR_REFINERY_ID = "ACT-REFN-ID-001"
ACTOR_EXPORTER_CO = "ACT-EXP-CO-001"
ACTOR_FARMER_CO = "ACT-FARMER-CO-001"

# Event IDs
EVT_ID_PREFIX = "EVT-COC"

# Plot IDs
PLOT_ID_COCOA_GH_1 = "PLOT-COC-GH-001"
PLOT_ID_COCOA_GH_2 = "PLOT-COC-GH-002"
PLOT_ID_COCOA_GH_3 = "PLOT-COC-GH-003"
PLOT_ID_COFFEE_CO_1 = "PLOT-COF-CO-001"
PLOT_ID_PALM_ID_1 = "PLOT-PLM-ID-001"
PLOT_ID_PALM_ID_2 = "PLOT-PLM-ID-002"
PLOT_ID_SOYA_BR_1 = "PLOT-SOY-BR-001"
PLOT_ID_RUBBER_TH_1 = "PLOT-RUB-TH-001"
PLOT_ID_WOOD_CD_1 = "PLOT-WOD-CD-001"
PLOT_ID_CATTLE_BR_1 = "PLOT-CTL-BR-001"


# ---------------------------------------------------------------------------
# Sample Origin Plots
# ---------------------------------------------------------------------------

ORIGIN_PLOT_COCOA_GH_1: Dict[str, Any] = {
    "plot_id": PLOT_ID_COCOA_GH_1,
    "gps_lat": 6.9520,
    "gps_lon": -1.9820,
    "area_ha": 3.5,
    "country_iso": "GH",
    "region": "Ashanti",
    "commodity": "cocoa",
    "deforestation_free": True,
    "verification_status": "verified",
}

ORIGIN_PLOT_COCOA_GH_2: Dict[str, Any] = {
    "plot_id": PLOT_ID_COCOA_GH_2,
    "gps_lat": 6.9550,
    "gps_lon": -1.9850,
    "area_ha": 2.0,
    "country_iso": "GH",
    "region": "Ashanti",
    "commodity": "cocoa",
    "deforestation_free": True,
    "verification_status": "verified",
}

ORIGIN_PLOT_COCOA_GH_3: Dict[str, Any] = {
    "plot_id": PLOT_ID_COCOA_GH_3,
    "gps_lat": 6.9480,
    "gps_lon": -1.9790,
    "area_ha": 4.2,
    "country_iso": "GH",
    "region": "Ashanti",
    "commodity": "cocoa",
    "deforestation_free": False,
    "verification_status": "flagged",
}

ORIGIN_PLOT_COFFEE_CO_1: Dict[str, Any] = {
    "plot_id": PLOT_ID_COFFEE_CO_1,
    "gps_lat": 5.0550,
    "gps_lon": -75.5050,
    "area_ha": 6.0,
    "country_iso": "CO",
    "region": "Caldas",
    "commodity": "coffee",
    "deforestation_free": True,
    "verification_status": "verified",
}

ORIGIN_PLOT_PALM_ID_1: Dict[str, Any] = {
    "plot_id": PLOT_ID_PALM_ID_1,
    "gps_lat": -0.5200,
    "gps_lon": 109.5200,
    "area_ha": 25.0,
    "country_iso": "ID",
    "region": "Kalimantan Barat",
    "commodity": "palm_oil",
    "deforestation_free": True,
    "verification_status": "verified",
}

ORIGIN_PLOT_PALM_ID_2: Dict[str, Any] = {
    "plot_id": PLOT_ID_PALM_ID_2,
    "gps_lat": -0.5300,
    "gps_lon": 109.5300,
    "area_ha": 18.0,
    "country_iso": "ID",
    "region": "Kalimantan Barat",
    "commodity": "palm_oil",
    "deforestation_free": True,
    "verification_status": "verified",
}


# ---------------------------------------------------------------------------
# Sample Batches (15+ across 7 commodities)
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    ).isoformat()


BATCH_COCOA_FARM_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_FARM_GH,
    "commodity": "cocoa",
    "commodity_form": "beans",
    "quantity_kg": 500.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0},
    ],
    "production_date": _ts(days_ago=30),
    "quality_grade": "grade_1",
    "status": "created",
    "created_at": _ts(days_ago=30),
    "created_by": ACTOR_FARMER_GH,
    "facility_id": None,
    "parent_batch_ids": [],
    "child_batch_ids": [],
    "coc_model": "IP",
    "metadata": {"harvest_method": "manual", "fermentation_days": 6},
}

BATCH_COCOA_COOP_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "commodity": "cocoa",
    "commodity_form": "beans",
    "quantity_kg": 5000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 30.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 20.0},
        {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0},
    ],
    "production_date": _ts(days_ago=25),
    "quality_grade": "grade_1",
    "status": "at_facility",
    "created_at": _ts(days_ago=25),
    "created_by": ACTOR_COOP_GH,
    "facility_id": FAC_ID_COOP_GH,
    "parent_batch_ids": [BATCH_ID_COCOA_FARM_GH],
    "child_batch_ids": [],
    "coc_model": "SG",
    "metadata": {},
}

BATCH_COCOA_NIBS_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_NIBS_GH,
    "commodity": "cocoa",
    "commodity_form": "nibs",
    "quantity_kg": 4350.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 30.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 20.0},
        {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0},
    ],
    "production_date": _ts(days_ago=20),
    "quality_grade": "grade_1",
    "status": "processed",
    "created_at": _ts(days_ago=20),
    "created_by": ACTOR_TRADER_GH,
    "facility_id": FAC_ID_PROC_GH,
    "parent_batch_ids": [BATCH_ID_COCOA_COOP_GH],
    "child_batch_ids": [BATCH_ID_COCOA_LIQUOR_GH],
    "coc_model": "SG",
    "metadata": {"process": "shelling"},
}

BATCH_COCOA_LIQUOR_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_LIQUOR_GH,
    "commodity": "cocoa",
    "commodity_form": "liquor",
    "quantity_kg": 3480.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 30.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 20.0},
        {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0},
    ],
    "production_date": _ts(days_ago=18),
    "quality_grade": "premium",
    "status": "processed",
    "created_at": _ts(days_ago=18),
    "created_by": ACTOR_TRADER_GH,
    "facility_id": FAC_ID_PROC_GH,
    "parent_batch_ids": [BATCH_ID_COCOA_NIBS_GH],
    "child_batch_ids": [BATCH_ID_COCOA_BUTTER_GH, BATCH_ID_COCOA_POWDER_GH],
    "coc_model": "SG",
    "metadata": {"process": "grinding"},
}

BATCH_COCOA_BUTTER_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_BUTTER_GH,
    "commodity": "cocoa",
    "commodity_form": "butter",
    "quantity_kg": 1566.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 30.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 20.0},
        {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0},
    ],
    "production_date": _ts(days_ago=16),
    "quality_grade": "premium",
    "status": "dispatched",
    "created_at": _ts(days_ago=16),
    "created_by": ACTOR_TRADER_GH,
    "facility_id": FAC_ID_PROC_GH,
    "parent_batch_ids": [BATCH_ID_COCOA_LIQUOR_GH],
    "child_batch_ids": [],
    "coc_model": "SG",
    "metadata": {"process": "pressing"},
}

BATCH_COCOA_POWDER_GH: Dict[str, Any] = {
    "batch_id": BATCH_ID_COCOA_POWDER_GH,
    "commodity": "cocoa",
    "commodity_form": "powder",
    "quantity_kg": 1914.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 30.0},
        {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 20.0},
        {"plot_id": PLOT_ID_COCOA_GH_3, "percentage": 50.0},
    ],
    "production_date": _ts(days_ago=16),
    "quality_grade": "premium",
    "status": "dispatched",
    "created_at": _ts(days_ago=16),
    "created_by": ACTOR_TRADER_GH,
    "facility_id": FAC_ID_PROC_GH,
    "parent_batch_ids": [BATCH_ID_COCOA_LIQUOR_GH],
    "child_batch_ids": [],
    "coc_model": "SG",
    "metadata": {"process": "pressing"},
}

BATCH_COFFEE_FARM_CO: Dict[str, Any] = {
    "batch_id": BATCH_ID_COFFEE_FARM_CO,
    "commodity": "coffee",
    "commodity_form": "cherry",
    "quantity_kg": 8000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COFFEE_CO_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=28),
    "quality_grade": "specialty",
    "status": "created",
    "created_at": _ts(days_ago=28),
    "created_by": ACTOR_FARMER_CO,
    "facility_id": None,
    "parent_batch_ids": [],
    "child_batch_ids": [BATCH_ID_COFFEE_GREEN_CO],
    "coc_model": "IP",
    "metadata": {"variety": "arabica", "altitude_m": 1800},
}

BATCH_COFFEE_GREEN_CO: Dict[str, Any] = {
    "batch_id": BATCH_ID_COFFEE_GREEN_CO,
    "commodity": "coffee",
    "commodity_form": "green",
    "quantity_kg": 1440.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_COFFEE_CO_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=22),
    "quality_grade": "specialty",
    "status": "processed",
    "created_at": _ts(days_ago=22),
    "created_by": ACTOR_EXPORTER_CO,
    "facility_id": FAC_ID_MILL_CO,
    "parent_batch_ids": [BATCH_ID_COFFEE_FARM_CO],
    "child_batch_ids": [],
    "coc_model": "IP",
    "metadata": {"process": "wet_processing"},
}

BATCH_PALM_MILL_ID: Dict[str, Any] = {
    "batch_id": BATCH_ID_PALM_MILL_ID,
    "commodity": "palm_oil",
    "commodity_form": "ffb",
    "quantity_kg": 50000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_PALM_ID_1, "percentage": 60.0},
        {"plot_id": PLOT_ID_PALM_ID_2, "percentage": 40.0},
    ],
    "production_date": _ts(days_ago=14),
    "quality_grade": "standard",
    "status": "at_facility",
    "created_at": _ts(days_ago=14),
    "created_by": ACTOR_MILL_ID,
    "facility_id": FAC_ID_MILL_ID,
    "parent_batch_ids": [],
    "child_batch_ids": [BATCH_ID_PALM_CPO_ID],
    "coc_model": "MB",
    "metadata": {},
}

BATCH_PALM_CPO_ID: Dict[str, Any] = {
    "batch_id": BATCH_ID_PALM_CPO_ID,
    "commodity": "palm_oil",
    "commodity_form": "cpo",
    "quantity_kg": 11000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_PALM_ID_1, "percentage": 60.0},
        {"plot_id": PLOT_ID_PALM_ID_2, "percentage": 40.0},
    ],
    "production_date": _ts(days_ago=12),
    "quality_grade": "standard",
    "status": "processed",
    "created_at": _ts(days_ago=12),
    "created_by": ACTOR_MILL_ID,
    "facility_id": FAC_ID_MILL_ID,
    "parent_batch_ids": [BATCH_ID_PALM_MILL_ID],
    "child_batch_ids": [],
    "coc_model": "MB",
    "metadata": {"ffa_pct": 3.5},
}

BATCH_SOYA_FARM_BR: Dict[str, Any] = {
    "batch_id": BATCH_ID_SOYA_FARM_BR,
    "commodity": "soya",
    "commodity_form": "beans",
    "quantity_kg": 100000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_SOYA_BR_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=40),
    "quality_grade": "grade_2",
    "status": "dispatched",
    "created_at": _ts(days_ago=40),
    "created_by": "ACT-FARMER-BR-001",
    "facility_id": FAC_ID_SILO_BR,
    "parent_batch_ids": [],
    "child_batch_ids": [],
    "coc_model": "MB",
    "metadata": {},
}

BATCH_RUBBER_PLOT_TH: Dict[str, Any] = {
    "batch_id": BATCH_ID_RUBBER_PLOT_TH,
    "commodity": "rubber",
    "commodity_form": "latex",
    "quantity_kg": 2000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_RUBBER_TH_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=10),
    "quality_grade": "grade_1",
    "status": "created",
    "created_at": _ts(days_ago=10),
    "created_by": "ACT-FARMER-TH-001",
    "facility_id": None,
    "parent_batch_ids": [],
    "child_batch_ids": [],
    "coc_model": "SG",
    "metadata": {},
}

BATCH_WOOD_FOREST_CD: Dict[str, Any] = {
    "batch_id": BATCH_ID_WOOD_FOREST_CD,
    "commodity": "wood",
    "commodity_form": "log",
    "quantity_kg": 20000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_WOOD_CD_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=60),
    "quality_grade": "standard",
    "status": "at_facility",
    "created_at": _ts(days_ago=60),
    "created_by": "ACT-LOGGER-CD-001",
    "facility_id": FAC_ID_SAWMILL_CD,
    "parent_batch_ids": [],
    "child_batch_ids": [],
    "coc_model": "IP",
    "metadata": {"species": "sapelli"},
}

BATCH_CATTLE_RANCH_BR: Dict[str, Any] = {
    "batch_id": BATCH_ID_CATTLE_RANCH_BR,
    "commodity": "cattle",
    "commodity_form": "live",
    "quantity_kg": 15000.0,
    "unit": "kg",
    "origin_plots": [
        {"plot_id": PLOT_ID_CATTLE_BR_1, "percentage": 100.0},
    ],
    "production_date": _ts(days_ago=7),
    "quality_grade": "standard",
    "status": "at_facility",
    "created_at": _ts(days_ago=7),
    "created_by": "ACT-RANCHER-BR-001",
    "facility_id": FAC_ID_FEEDLOT_BR,
    "parent_batch_ids": [],
    "child_batch_ids": [],
    "coc_model": "IP",
    "metadata": {"head_count": 30},
}

ALL_SAMPLE_BATCHES: List[Dict[str, Any]] = [
    BATCH_COCOA_FARM_GH, BATCH_COCOA_COOP_GH, BATCH_COCOA_NIBS_GH,
    BATCH_COCOA_LIQUOR_GH, BATCH_COCOA_BUTTER_GH, BATCH_COCOA_POWDER_GH,
    BATCH_COFFEE_FARM_CO, BATCH_COFFEE_GREEN_CO,
    BATCH_PALM_MILL_ID, BATCH_PALM_CPO_ID,
    BATCH_SOYA_FARM_BR, BATCH_RUBBER_PLOT_TH,
    BATCH_WOOD_FOREST_CD, BATCH_CATTLE_RANCH_BR,
]


# ---------------------------------------------------------------------------
# Sample Custody Events (20+ across all 10 event types)
# ---------------------------------------------------------------------------

def _make_event_id(suffix: str) -> str:
    """Generate a deterministic event ID."""
    return f"{EVT_ID_PREFIX}-{suffix}"


TRANSFER_COCOA_GH_NL: Dict[str, Any] = {
    "event_id": _make_event_id("XFER-COC-GH-NL-001"),
    "event_type": "transfer",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=20),
    "sender_actor_id": ACTOR_TRADER_GH,
    "receiver_actor_id": ACTOR_SHIPPER_INT,
    "source_location": {"facility_id": FAC_ID_PROC_GH, "gps_lat": 5.5571, "gps_lon": -0.2013},
    "dest_location": {"facility_id": None, "gps_lat": 5.6037, "gps_lon": -0.1870},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-INV-001", "DOC-DN-001"],
    "predecessor_event_id": None,
    "successor_event_id": _make_event_id("RCPT-COC-NL-001"),
    "status": "completed",
    "notes": "Transfer from Tema to Tema port",
}

RECEIPT_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("RCPT-COC-NL-001"),
    "event_type": "receipt",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=10),
    "sender_actor_id": ACTOR_SHIPPER_INT,
    "receiver_actor_id": ACTOR_IMPORTER_NL,
    "source_location": {"facility_id": None, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-DN-002"],
    "predecessor_event_id": _make_event_id("XFER-COC-GH-NL-001"),
    "successor_event_id": _make_event_id("STIN-COC-NL-001"),
    "status": "completed",
    "notes": "Receipt at Rotterdam warehouse",
}

STORAGE_IN_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("STIN-COC-NL-001"),
    "event_type": "storage_in",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=10, hours_ago=2),
    "sender_actor_id": ACTOR_IMPORTER_NL,
    "receiver_actor_id": ACTOR_IMPORTER_NL,
    "source_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-WR-001", "DOC-WC-001"],
    "predecessor_event_id": _make_event_id("RCPT-COC-NL-001"),
    "successor_event_id": _make_event_id("STOUT-COC-NL-001"),
    "status": "completed",
    "notes": "Stored in warehouse bay A-14",
}

STORAGE_OUT_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("STOUT-COC-NL-001"),
    "event_type": "storage_out",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=5),
    "sender_actor_id": ACTOR_IMPORTER_NL,
    "receiver_actor_id": ACTOR_PROCESSOR_DE,
    "source_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": FAC_ID_FACTORY_DE, "gps_lat": 53.5511, "gps_lon": 9.9937},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-DN-003", "DOC-WC-002"],
    "predecessor_event_id": _make_event_id("STIN-COC-NL-001"),
    "successor_event_id": None,
    "status": "completed",
    "notes": "Dispatched to Hamburg factory",
}

PROCESSING_IN_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("PRIN-COC-DE-001"),
    "event_type": "processing_in",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=4),
    "sender_actor_id": ACTOR_PROCESSOR_DE,
    "receiver_actor_id": ACTOR_PROCESSOR_DE,
    "source_location": {"facility_id": FAC_ID_FACTORY_DE, "gps_lat": 53.5511, "gps_lon": 9.9937},
    "dest_location": {"facility_id": FAC_ID_FACTORY_DE, "gps_lat": 53.5511, "gps_lon": 9.9937},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-WC-003", "DOC-QC-001"],
    "predecessor_event_id": _make_event_id("STOUT-COC-NL-001"),
    "successor_event_id": _make_event_id("PROUT-COC-DE-001"),
    "status": "completed",
    "notes": "Received at Hamburg processing line",
}

PROCESSING_OUT_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("PROUT-COC-DE-001"),
    "event_type": "processing_out",
    "batch_id": BATCH_ID_COCOA_NIBS_GH,
    "timestamp": _ts(days_ago=3),
    "sender_actor_id": ACTOR_PROCESSOR_DE,
    "receiver_actor_id": ACTOR_PROCESSOR_DE,
    "source_location": {"facility_id": FAC_ID_FACTORY_DE, "gps_lat": 53.5511, "gps_lon": 9.9937},
    "dest_location": {"facility_id": FAC_ID_FACTORY_DE, "gps_lat": 53.5511, "gps_lon": 9.9937},
    "quantity_kg": 4350.0,
    "unit": "kg",
    "document_refs": ["DOC-WC-004"],
    "predecessor_event_id": _make_event_id("PRIN-COC-DE-001"),
    "successor_event_id": None,
    "status": "completed",
    "notes": "Output: cocoa nibs after shelling",
}

EXPORT_COCOA_GH: Dict[str, Any] = {
    "event_id": _make_event_id("EXP-COC-GH-001"),
    "event_type": "export",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=18),
    "sender_actor_id": ACTOR_TRADER_GH,
    "receiver_actor_id": ACTOR_SHIPPER_INT,
    "source_location": {"facility_id": "FAC-PORT-GH", "gps_lat": 5.6131, "gps_lon": 0.0200},
    "dest_location": {"facility_id": None, "gps_lat": None, "gps_lon": None},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-BL-001", "DOC-PHY-001", "DOC-COO-001", "DOC-CD-001"],
    "predecessor_event_id": _make_event_id("XFER-COC-GH-NL-001"),
    "successor_event_id": _make_event_id("IMP-COC-NL-001"),
    "status": "completed",
    "notes": "Export from Tema Port to Rotterdam",
}

IMPORT_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("IMP-COC-NL-001"),
    "event_type": "import",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=11),
    "sender_actor_id": ACTOR_SHIPPER_INT,
    "receiver_actor_id": ACTOR_IMPORTER_NL,
    "source_location": {"facility_id": None, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": "FAC-PORT-NL", "gps_lat": 51.9225, "gps_lon": 4.4792},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-BL-001", "DOC-CD-002"],
    "predecessor_event_id": _make_event_id("EXP-COC-GH-001"),
    "successor_event_id": _make_event_id("RCPT-COC-NL-001"),
    "status": "completed",
    "notes": "Import at Rotterdam port",
}

INSPECTION_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("INSP-COC-NL-001"),
    "event_type": "inspection",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=9),
    "sender_actor_id": ACTOR_IMPORTER_NL,
    "receiver_actor_id": ACTOR_IMPORTER_NL,
    "source_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "quantity_kg": 5000.0,
    "unit": "kg",
    "document_refs": ["DOC-QC-002"],
    "predecessor_event_id": _make_event_id("STIN-COC-NL-001"),
    "successor_event_id": None,
    "status": "completed",
    "notes": "Quality inspection passed",
}

SAMPLING_COCOA_NL: Dict[str, Any] = {
    "event_id": _make_event_id("SMPL-COC-NL-001"),
    "event_type": "sampling",
    "batch_id": BATCH_ID_COCOA_COOP_GH,
    "timestamp": _ts(days_ago=9, hours_ago=1),
    "sender_actor_id": ACTOR_IMPORTER_NL,
    "receiver_actor_id": ACTOR_IMPORTER_NL,
    "source_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "dest_location": {"facility_id": FAC_ID_WAREHOUSE_NL, "gps_lat": 51.9225, "gps_lon": 4.4792},
    "quantity_kg": 0.5,
    "unit": "kg",
    "document_refs": [],
    "predecessor_event_id": _make_event_id("STIN-COC-NL-001"),
    "successor_event_id": None,
    "status": "completed",
    "notes": "Quality sampling for lab analysis",
}

ALL_SAMPLE_EVENTS: List[Dict[str, Any]] = [
    TRANSFER_COCOA_GH_NL, RECEIPT_COCOA_NL,
    STORAGE_IN_COCOA_NL, STORAGE_OUT_COCOA_NL,
    PROCESSING_IN_COCOA_NL, PROCESSING_OUT_COCOA_NL,
    EXPORT_COCOA_GH, IMPORT_COCOA_NL,
    INSPECTION_COCOA_NL, SAMPLING_COCOA_NL,
]


# ---------------------------------------------------------------------------
# Sample Documents
# ---------------------------------------------------------------------------

def _make_doc(
    doc_id: str,
    doc_type: str,
    event_ids: Optional[List[str]] = None,
    quantity_kg: Optional[float] = None,
    issuer: str = "Test Issuer",
    days_until_expiry: int = 365,
) -> Dict[str, Any]:
    """Build a document record dictionary for testing."""
    now = datetime.now(timezone.utc)
    return {
        "document_id": doc_id,
        "document_type": doc_type,
        "reference_number": f"REF-{doc_id}",
        "issuer": issuer,
        "issue_date": (now - timedelta(days=30)).isoformat(),
        "expiry_date": (now + timedelta(days=days_until_expiry)).isoformat(),
        "event_ids": event_ids or [],
        "quantity_kg": quantity_kg,
        "file_hash": hashlib.sha256(f"file-{doc_id}".encode()).hexdigest(),
        "metadata": {},
        "status": "valid",
    }


SAMPLE_DOCUMENTS: List[Dict[str, Any]] = [
    _make_doc("DOC-BL-001", "bill_of_lading", [_make_event_id("EXP-COC-GH-001")], 5000.0, "Maersk Line"),
    _make_doc("DOC-PHY-001", "phytosanitary_cert", [_make_event_id("EXP-COC-GH-001")], 5000.0, "Ghana Plant Quarantine"),
    _make_doc("DOC-COO-001", "certificate_of_origin", [_make_event_id("EXP-COC-GH-001")], 5000.0, "Ghana Cocoa Board"),
    _make_doc("DOC-CD-001", "customs_declaration", [_make_event_id("EXP-COC-GH-001")], 5000.0, "Ghana Customs"),
    _make_doc("DOC-CD-002", "customs_declaration", [_make_event_id("IMP-COC-NL-001")], 5000.0, "NL Customs"),
    _make_doc("DOC-INV-001", "commercial_invoice", [_make_event_id("XFER-COC-GH-NL-001")], 5000.0, "Ghana Cocoa Trading"),
    _make_doc("DOC-DN-001", "delivery_note", [_make_event_id("XFER-COC-GH-NL-001")], 5000.0, "Ghana Cocoa Trading"),
    _make_doc("DOC-DN-002", "delivery_note", [_make_event_id("RCPT-COC-NL-001")], 5000.0, "Shipping Co"),
    _make_doc("DOC-DN-003", "delivery_note", [_make_event_id("STOUT-COC-NL-001")], 5000.0, "PalmOil Europe BV"),
    _make_doc("DOC-WR-001", "warehouse_receipt", [_make_event_id("STIN-COC-NL-001")], 5000.0, "Rotterdam Warehouse"),
    _make_doc("DOC-WC-001", "weight_cert", [_make_event_id("STIN-COC-NL-001")], 5000.0, "SGS"),
    _make_doc("DOC-WC-002", "weight_cert", [_make_event_id("STOUT-COC-NL-001")], 5000.0, "SGS"),
    _make_doc("DOC-WC-003", "weight_cert", [_make_event_id("PRIN-COC-DE-001")], 5000.0, "SGS"),
    _make_doc("DOC-WC-004", "weight_cert", [_make_event_id("PROUT-COC-DE-001")], 4350.0, "SGS"),
    _make_doc("DOC-QC-001", "quality_cert", [_make_event_id("PRIN-COC-DE-001")], 5000.0, "SGS"),
    _make_doc("DOC-QC-002", "quality_cert", [_make_event_id("INSP-COC-NL-001")], 5000.0, "Bureau Veritas"),
]


# ---------------------------------------------------------------------------
# Sample Transformations
# ---------------------------------------------------------------------------

TRANSFORM_COCOA_BEANS_TO_NIBS: Dict[str, Any] = {
    "transformation_id": "XFRM-COC-BN-001",
    "process_type": "milling",
    "facility_id": FAC_ID_PROC_GH,
    "timestamp": _ts(days_ago=20),
    "input_batches": [{"batch_id": BATCH_ID_COCOA_COOP_GH, "quantity_kg": 5000.0}],
    "output_batches": [
        {"batch_id": BATCH_ID_COCOA_NIBS_GH, "quantity_kg": 4350.0, "product_type": "main"},
    ],
    "waste_kg": 550.0,
    "by_products": [{"type": "shell", "quantity_kg": 100.0}],
    "expected_yield_ratio": 0.87,
    "actual_yield_ratio": 0.87,
    "notes": "Cocoa beans to nibs (shelling)",
}

TRANSFORM_COCOA_NIBS_TO_LIQUOR: Dict[str, Any] = {
    "transformation_id": "XFRM-COC-NL-001",
    "process_type": "milling",
    "facility_id": FAC_ID_PROC_GH,
    "timestamp": _ts(days_ago=18),
    "input_batches": [{"batch_id": BATCH_ID_COCOA_NIBS_GH, "quantity_kg": 4350.0}],
    "output_batches": [
        {"batch_id": BATCH_ID_COCOA_LIQUOR_GH, "quantity_kg": 3480.0, "product_type": "main"},
    ],
    "waste_kg": 870.0,
    "by_products": [],
    "expected_yield_ratio": 0.80,
    "actual_yield_ratio": 0.80,
    "notes": "Cocoa nibs to liquor (grinding)",
}

TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER: Dict[str, Any] = {
    "transformation_id": "XFRM-COC-BP-001",
    "process_type": "pressing",
    "facility_id": FAC_ID_PROC_GH,
    "timestamp": _ts(days_ago=16),
    "input_batches": [{"batch_id": BATCH_ID_COCOA_LIQUOR_GH, "quantity_kg": 3480.0}],
    "output_batches": [
        {"batch_id": BATCH_ID_COCOA_BUTTER_GH, "quantity_kg": 1566.0, "product_type": "main"},
        {"batch_id": BATCH_ID_COCOA_POWDER_GH, "quantity_kg": 1914.0, "product_type": "co_product"},
    ],
    "waste_kg": 0.0,
    "by_products": [],
    "expected_yield_ratio": 1.0,
    "actual_yield_ratio": 1.0,
    "notes": "Cocoa liquor to butter + powder (pressing)",
}

ALL_SAMPLE_TRANSFORMATIONS: List[Dict[str, Any]] = [
    TRANSFORM_COCOA_BEANS_TO_NIBS,
    TRANSFORM_COCOA_NIBS_TO_LIQUOR,
    TRANSFORM_COCOA_LIQUOR_TO_BUTTER_POWDER,
]


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_event(
    event_type: str = "transfer",
    batch_id: Optional[str] = None,
    quantity_kg: float = 1000.0,
    sender_actor_id: str = ACTOR_TRADER_GH,
    receiver_actor_id: str = ACTOR_IMPORTER_NL,
    source_facility_id: str = FAC_ID_PROC_GH,
    dest_facility_id: str = FAC_ID_WAREHOUSE_NL,
    timestamp: Optional[str] = None,
    document_refs: Optional[List[str]] = None,
    event_id: Optional[str] = None,
    predecessor_event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a custody event dictionary for testing.

    Args:
        event_type: One of the 10 custody event types.
        batch_id: Associated batch identifier.
        quantity_kg: Event quantity in kilograms.
        sender_actor_id: Actor sending the goods.
        receiver_actor_id: Actor receiving the goods.
        source_facility_id: Source facility identifier.
        dest_facility_id: Destination facility identifier.
        timestamp: ISO timestamp (auto-generated if None).
        document_refs: List of document IDs linked to this event.
        event_id: Event identifier (auto-generated if None).
        predecessor_event_id: ID of previous event in chain.

    Returns:
        Dict with all custody event fields.
    """
    return {
        "event_id": event_id or f"EVT-{uuid.uuid4().hex[:12].upper()}",
        "event_type": event_type,
        "batch_id": batch_id or f"BATCH-{uuid.uuid4().hex[:12].upper()}",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "sender_actor_id": sender_actor_id,
        "receiver_actor_id": receiver_actor_id,
        "source_location": {"facility_id": source_facility_id, "gps_lat": 5.5571, "gps_lon": -0.2013},
        "dest_location": {"facility_id": dest_facility_id, "gps_lat": 51.9225, "gps_lon": 4.4792},
        "quantity_kg": quantity_kg,
        "unit": "kg",
        "document_refs": document_refs or [],
        "predecessor_event_id": predecessor_event_id,
        "successor_event_id": None,
        "status": "completed",
        "notes": "",
    }


def make_batch(
    commodity: str = "cocoa",
    commodity_form: str = "beans",
    quantity_kg: float = 1000.0,
    origin_plots: Optional[List[Dict[str, Any]]] = None,
    batch_id: Optional[str] = None,
    status: str = "created",
    coc_model: str = "SG",
    parent_batch_ids: Optional[List[str]] = None,
    facility_id: Optional[str] = None,
    created_by: str = ACTOR_FARMER_GH,
) -> Dict[str, Any]:
    """Build a batch record dictionary for testing.

    Args:
        commodity: EUDR commodity type.
        commodity_form: Physical form of the commodity.
        quantity_kg: Batch quantity in kilograms.
        origin_plots: List of origin plot allocations with percentages.
        batch_id: Batch identifier (auto-generated if None).
        status: Batch lifecycle status.
        coc_model: Chain of custody model (IP/SG/MB/CB).
        parent_batch_ids: Parent batch IDs for genealogy tracking.
        facility_id: Current facility location.
        created_by: Actor who created the batch.

    Returns:
        Dict with all batch fields.
    """
    if origin_plots is None:
        origin_plots = [{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}]
    return {
        "batch_id": batch_id or f"BATCH-{uuid.uuid4().hex[:12].upper()}",
        "commodity": commodity,
        "commodity_form": commodity_form,
        "quantity_kg": quantity_kg,
        "unit": "kg",
        "origin_plots": origin_plots,
        "production_date": datetime.now(timezone.utc).isoformat(),
        "quality_grade": "standard",
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_by": created_by,
        "facility_id": facility_id,
        "parent_batch_ids": parent_batch_ids or [],
        "child_batch_ids": [],
        "coc_model": coc_model,
        "metadata": {},
    }


def make_transformation(
    process_type: str = "milling",
    facility_id: str = FAC_ID_PROC_GH,
    input_batches: Optional[List[Dict[str, Any]]] = None,
    output_batches: Optional[List[Dict[str, Any]]] = None,
    waste_kg: float = 0.0,
    expected_yield: float = 0.87,
    actual_yield: Optional[float] = None,
    transformation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a transformation record dictionary for testing.

    Args:
        process_type: Type of processing (milling, pressing, etc.).
        facility_id: Facility where transformation occurs.
        input_batches: List of input batch references with quantities.
        output_batches: List of output batch references with quantities.
        waste_kg: Waste quantity in kilograms.
        expected_yield: Expected yield ratio from reference data.
        actual_yield: Actual yield ratio (computed if None).
        transformation_id: Transformation identifier (auto-generated if None).

    Returns:
        Dict with all transformation fields.
    """
    if input_batches is None:
        input_batches = [{"batch_id": "BATCH-IN-001", "quantity_kg": 1000.0}]
    if output_batches is None:
        output_batches = [{"batch_id": "BATCH-OUT-001", "quantity_kg": 870.0, "product_type": "main"}]
    total_in = sum(b["quantity_kg"] for b in input_batches)
    total_out = sum(b["quantity_kg"] for b in output_batches)
    computed_yield = total_out / total_in if total_in > 0 else 0.0
    return {
        "transformation_id": transformation_id or f"XFRM-{uuid.uuid4().hex[:12].upper()}",
        "process_type": process_type,
        "facility_id": facility_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_batches": input_batches,
        "output_batches": output_batches,
        "waste_kg": waste_kg,
        "by_products": [],
        "expected_yield_ratio": expected_yield,
        "actual_yield_ratio": actual_yield if actual_yield is not None else computed_yield,
        "notes": "",
    }


def make_document(
    doc_type: str = "bill_of_lading",
    event_ids: Optional[List[str]] = None,
    quantity_kg: Optional[float] = None,
    issuer: str = "Test Issuer",
    days_until_expiry: int = 365,
    doc_id: Optional[str] = None,
    status: str = "valid",
) -> Dict[str, Any]:
    """Build a document record dictionary for testing.

    Args:
        doc_type: Document type from the 15 supported types.
        event_ids: List of custody event IDs linked to this document.
        quantity_kg: Quantity referenced in the document.
        issuer: Document issuing organization.
        days_until_expiry: Days from now until document expires.
        doc_id: Document identifier (auto-generated if None).
        status: Document status (valid, expired, revoked).

    Returns:
        Dict with all document fields.
    """
    now = datetime.now(timezone.utc)
    did = doc_id or f"DOC-{uuid.uuid4().hex[:8].upper()}"
    return {
        "document_id": did,
        "document_type": doc_type,
        "reference_number": f"REF-{did}",
        "issuer": issuer,
        "issue_date": (now - timedelta(days=30)).isoformat(),
        "expiry_date": (now + timedelta(days=days_until_expiry)).isoformat(),
        "event_ids": event_ids or [],
        "quantity_kg": quantity_kg,
        "file_hash": hashlib.sha256(f"file-{did}".encode()).hexdigest(),
        "metadata": {},
        "status": status,
    }


def make_mass_balance_entry(
    facility_id: str = FAC_ID_MILL_ID,
    commodity: str = "palm_oil",
    entry_type: str = "input",
    quantity_kg: float = 1000.0,
    compliance_status: str = "compliant",
    batch_id: Optional[str] = None,
    credit_period_start: Optional[str] = None,
    entry_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a mass balance ledger entry for testing.

    Args:
        facility_id: Facility identifier.
        commodity: EUDR commodity type.
        entry_type: Either 'input' or 'output'.
        quantity_kg: Entry quantity in kilograms.
        compliance_status: Whether the material is compliant.
        batch_id: Associated batch identifier.
        credit_period_start: Start of the credit period.
        entry_id: Entry identifier (auto-generated if None).

    Returns:
        Dict with all mass balance entry fields.
    """
    now = datetime.now(timezone.utc)
    return {
        "entry_id": entry_id or f"MBE-{uuid.uuid4().hex[:12].upper()}",
        "facility_id": facility_id,
        "commodity": commodity,
        "entry_type": entry_type,
        "quantity_kg": quantity_kg,
        "compliance_status": compliance_status,
        "batch_id": batch_id or f"BATCH-{uuid.uuid4().hex[:8].upper()}",
        "date": now.isoformat(),
        "credit_period_start": credit_period_start or (now - timedelta(days=30)).isoformat(),
        "credit_period_end": (now + timedelta(days=60)).isoformat(),
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Chain Builders
# ---------------------------------------------------------------------------

def build_cocoa_chain() -> Dict[str, Any]:
    """Build a complete cocoa custody chain from farm to EU warehouse.

    Returns:
        Dict with keys: batches, events, transformations, documents.
    """
    return {
        "batches": [
            BATCH_COCOA_FARM_GH, BATCH_COCOA_COOP_GH, BATCH_COCOA_NIBS_GH,
            BATCH_COCOA_LIQUOR_GH, BATCH_COCOA_BUTTER_GH, BATCH_COCOA_POWDER_GH,
        ],
        "events": list(ALL_SAMPLE_EVENTS),
        "transformations": list(ALL_SAMPLE_TRANSFORMATIONS),
        "documents": list(SAMPLE_DOCUMENTS),
    }


def build_palm_oil_chain() -> Dict[str, Any]:
    """Build a palm oil custody chain from plantation to refinery.

    Returns:
        Dict with keys: batches, events, transformations, documents.
    """
    batch_ffb = make_batch("palm_oil", "ffb", 50000.0,
                           batch_id="BATCH-PLM-CH-FFB",
                           origin_plots=[
                               {"plot_id": PLOT_ID_PALM_ID_1, "percentage": 60.0},
                               {"plot_id": PLOT_ID_PALM_ID_2, "percentage": 40.0},
                           ])
    batch_cpo = make_batch("palm_oil", "cpo", 11000.0,
                           batch_id="BATCH-PLM-CH-CPO",
                           parent_batch_ids=["BATCH-PLM-CH-FFB"],
                           origin_plots=[
                               {"plot_id": PLOT_ID_PALM_ID_1, "percentage": 60.0},
                               {"plot_id": PLOT_ID_PALM_ID_2, "percentage": 40.0},
                           ])
    batch_rbd = make_batch("palm_oil", "rbd", 10120.0,
                           batch_id="BATCH-PLM-CH-RBD",
                           parent_batch_ids=["BATCH-PLM-CH-CPO"],
                           origin_plots=[
                               {"plot_id": PLOT_ID_PALM_ID_1, "percentage": 60.0},
                               {"plot_id": PLOT_ID_PALM_ID_2, "percentage": 40.0},
                           ])
    events = [
        make_event("receipt", "BATCH-PLM-CH-FFB", 50000.0,
                   ACTOR_MILL_ID, ACTOR_MILL_ID, FAC_ID_MILL_ID, FAC_ID_MILL_ID),
        make_event("processing_in", "BATCH-PLM-CH-FFB", 50000.0,
                   ACTOR_MILL_ID, ACTOR_MILL_ID, FAC_ID_MILL_ID, FAC_ID_MILL_ID),
        make_event("processing_out", "BATCH-PLM-CH-CPO", 11000.0,
                   ACTOR_MILL_ID, ACTOR_MILL_ID, FAC_ID_MILL_ID, FAC_ID_MILL_ID),
        make_event("transfer", "BATCH-PLM-CH-CPO", 11000.0,
                   ACTOR_MILL_ID, ACTOR_REFINERY_ID, FAC_ID_MILL_ID, FAC_ID_REFINERY_ID),
        make_event("processing_in", "BATCH-PLM-CH-CPO", 11000.0,
                   ACTOR_REFINERY_ID, ACTOR_REFINERY_ID, FAC_ID_REFINERY_ID, FAC_ID_REFINERY_ID),
        make_event("processing_out", "BATCH-PLM-CH-RBD", 10120.0,
                   ACTOR_REFINERY_ID, ACTOR_REFINERY_ID, FAC_ID_REFINERY_ID, FAC_ID_REFINERY_ID),
    ]
    transforms = [
        make_transformation("pressing", FAC_ID_MILL_ID,
                            [{"batch_id": "BATCH-PLM-CH-FFB", "quantity_kg": 50000.0}],
                            [{"batch_id": "BATCH-PLM-CH-CPO", "quantity_kg": 11000.0, "product_type": "main"}],
                            waste_kg=39000.0, expected_yield=0.22),
        make_transformation("refining", FAC_ID_REFINERY_ID,
                            [{"batch_id": "BATCH-PLM-CH-CPO", "quantity_kg": 11000.0}],
                            [{"batch_id": "BATCH-PLM-CH-RBD", "quantity_kg": 10120.0, "product_type": "main"}],
                            waste_kg=880.0, expected_yield=0.92),
    ]
    return {
        "batches": [batch_ffb, batch_cpo, batch_rbd],
        "events": events,
        "transformations": transforms,
        "documents": [],
    }


def build_coffee_chain() -> Dict[str, Any]:
    """Build a coffee custody chain from farm to roaster.

    Returns:
        Dict with keys: batches, events, transformations, documents.
    """
    batch_cherry = make_batch("coffee", "cherry", 8000.0,
                              batch_id="BATCH-COF-CH-CHER",
                              coc_model="IP",
                              origin_plots=[{"plot_id": PLOT_ID_COFFEE_CO_1, "percentage": 100.0}])
    batch_green = make_batch("coffee", "green", 1440.0,
                             batch_id="BATCH-COF-CH-GREN",
                             coc_model="IP",
                             parent_batch_ids=["BATCH-COF-CH-CHER"],
                             origin_plots=[{"plot_id": PLOT_ID_COFFEE_CO_1, "percentage": 100.0}])
    batch_roasted = make_batch("coffee", "roasted", 1180.0,
                               batch_id="BATCH-COF-CH-ROST",
                               coc_model="IP",
                               parent_batch_ids=["BATCH-COF-CH-GREN"],
                               origin_plots=[{"plot_id": PLOT_ID_COFFEE_CO_1, "percentage": 100.0}])
    events = [
        make_event("receipt", "BATCH-COF-CH-CHER", 8000.0,
                   ACTOR_FARMER_CO, ACTOR_EXPORTER_CO, "FAC-FARM-CO", FAC_ID_MILL_CO),
        make_event("processing_in", "BATCH-COF-CH-CHER", 8000.0,
                   ACTOR_EXPORTER_CO, ACTOR_EXPORTER_CO, FAC_ID_MILL_CO, FAC_ID_MILL_CO),
        make_event("processing_out", "BATCH-COF-CH-GREN", 1440.0,
                   ACTOR_EXPORTER_CO, ACTOR_EXPORTER_CO, FAC_ID_MILL_CO, FAC_ID_MILL_CO),
        make_event("export", "BATCH-COF-CH-GREN", 1440.0,
                   ACTOR_EXPORTER_CO, ACTOR_SHIPPER_INT, FAC_ID_MILL_CO, "FAC-PORT-CO"),
    ]
    transforms = [
        make_transformation("drying", FAC_ID_MILL_CO,
                            [{"batch_id": "BATCH-COF-CH-CHER", "quantity_kg": 8000.0}],
                            [{"batch_id": "BATCH-COF-CH-GREN", "quantity_kg": 1440.0, "product_type": "main"}],
                            waste_kg=6560.0, expected_yield=0.18),
    ]
    return {
        "batches": [batch_cherry, batch_green, batch_roasted],
        "events": events,
        "transformations": transforms,
        "documents": [],
    }


# ---------------------------------------------------------------------------
# Batch Genealogy Builders
# ---------------------------------------------------------------------------

def build_linear_genealogy(depth: int = 5, commodity: str = "cocoa") -> List[Dict[str, Any]]:
    """Build a linear batch genealogy chain with N levels.

    Args:
        depth: Number of batches in the chain.
        commodity: EUDR commodity type.

    Returns:
        List of batch dicts with parent/child references set.
    """
    batches = []
    qty = 10000.0
    for i in range(depth):
        bid = f"BATCH-LIN-{commodity}-{i:03d}"
        parent_ids = [batches[-1]["batch_id"]] if batches else []
        batch = make_batch(commodity, "beans", qty, batch_id=bid,
                           parent_batch_ids=parent_ids)
        if batches:
            batches[-1]["child_batch_ids"] = [bid]
        batches.append(batch)
        qty *= 0.90
    return batches


def build_split_genealogy(n_splits: int = 3, commodity: str = "cocoa") -> List[Dict[str, Any]]:
    """Build a batch genealogy with one parent splitting into N children.

    Args:
        n_splits: Number of child batches to create.
        commodity: EUDR commodity type.

    Returns:
        List of batch dicts [parent, child_0, child_1, ...].
    """
    parent = make_batch(commodity, "beans", 10000.0, batch_id="BATCH-SPLIT-PARENT")
    children = []
    child_qty = 10000.0 / n_splits
    child_ids = []
    for i in range(n_splits):
        cid = f"BATCH-SPLIT-CHILD-{i:03d}"
        child_ids.append(cid)
        child = make_batch(commodity, "beans", child_qty, batch_id=cid,
                           parent_batch_ids=["BATCH-SPLIT-PARENT"],
                           origin_plots=[
                               {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0 / n_splits * (i + 1)},
                           ])
        children.append(child)
    parent["child_batch_ids"] = child_ids
    return [parent] + children


def build_merge_genealogy(n_inputs: int = 3, commodity: str = "cocoa") -> List[Dict[str, Any]]:
    """Build a batch genealogy with N parents merging into one child.

    Args:
        n_inputs: Number of parent batches to merge.
        commodity: EUDR commodity type.

    Returns:
        List of batch dicts [parent_0, parent_1, ..., merged_child].
    """
    parents = []
    parent_ids = []
    total_qty = 0.0
    for i in range(n_inputs):
        pid = f"BATCH-MERGE-PARENT-{i:03d}"
        parent_ids.append(pid)
        qty = 1000.0 * (i + 1)
        total_qty += qty
        parent = make_batch(commodity, "beans", qty, batch_id=pid)
        parents.append(parent)
    merged = make_batch(commodity, "beans", total_qty,
                        batch_id="BATCH-MERGE-CHILD",
                        parent_batch_ids=parent_ids)
    for p in parents:
        p["child_batch_ids"] = ["BATCH-MERGE-CHILD"]
    return parents + [merged]


def build_diamond_genealogy(commodity: str = "cocoa") -> List[Dict[str, Any]]:
    """Build a diamond-shaped batch genealogy.

    Structure: root -> [mid_A, mid_B] -> merged_child

    Args:
        commodity: EUDR commodity type.

    Returns:
        List of batch dicts forming a diamond.
    """
    root = make_batch(commodity, "beans", 10000.0, batch_id="BATCH-DIAMOND-ROOT")
    mid_a = make_batch(commodity, "beans", 5000.0, batch_id="BATCH-DIAMOND-A",
                       parent_batch_ids=["BATCH-DIAMOND-ROOT"])
    mid_b = make_batch(commodity, "beans", 5000.0, batch_id="BATCH-DIAMOND-B",
                       parent_batch_ids=["BATCH-DIAMOND-ROOT"])
    merged = make_batch(commodity, "beans", 10000.0, batch_id="BATCH-DIAMOND-MERGED",
                        parent_batch_ids=["BATCH-DIAMOND-A", "BATCH-DIAMOND-B"])
    root["child_batch_ids"] = ["BATCH-DIAMOND-A", "BATCH-DIAMOND-B"]
    mid_a["child_batch_ids"] = ["BATCH-DIAMOND-MERGED"]
    mid_b["child_batch_ids"] = ["BATCH-DIAMOND-MERGED"]
    return [root, mid_a, mid_b, merged]


def build_blend_genealogy() -> List[Dict[str, Any]]:
    """Build a blend genealogy with batches from different origins.

    Returns:
        List of batch dicts [origin_A, origin_B, blended].
    """
    origin_a = make_batch("cocoa", "beans", 3000.0, batch_id="BATCH-BLEND-A",
                          origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 100.0}])
    origin_b = make_batch("cocoa", "beans", 2000.0, batch_id="BATCH-BLEND-B",
                          origin_plots=[{"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 100.0}])
    blended = make_batch("cocoa", "beans", 5000.0, batch_id="BATCH-BLEND-MIX",
                         parent_batch_ids=["BATCH-BLEND-A", "BATCH-BLEND-B"],
                         origin_plots=[
                             {"plot_id": PLOT_ID_COCOA_GH_1, "percentage": 60.0},
                             {"plot_id": PLOT_ID_COCOA_GH_2, "percentage": 40.0},
                         ])
    origin_a["child_batch_ids"] = ["BATCH-BLEND-MIX"]
    origin_b["child_batch_ids"] = ["BATCH-BLEND-MIX"]
    return [origin_a, origin_b, blended]


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


def assert_valid_chain(chain: Dict[str, Any]) -> None:
    """Assert that a custody chain has all required components.

    Args:
        chain: Dict with batches, events, transformations, documents keys.

    Raises:
        AssertionError: If chain is missing required components.
    """
    assert "batches" in chain, "Chain must include batches"
    assert "events" in chain, "Chain must include events"
    assert len(chain["batches"]) > 0, "Chain must have at least one batch"
    assert len(chain["events"]) > 0, "Chain must have at least one event"
    for batch in chain["batches"]:
        assert "batch_id" in batch, "Every batch must have a batch_id"
        assert "commodity" in batch, "Every batch must have a commodity"
    for event in chain["events"]:
        assert "event_id" in event, "Every event must have an event_id"
        assert "event_type" in event, "Every event must have an event_type"
        assert event["event_type"] in EVENT_TYPES, (
            f"Invalid event type: {event['event_type']}"
        )


def assert_mass_conservation(
    input_qty: float,
    output_qty: float,
    waste_qty: float = 0.0,
    tolerance_pct: float = 2.0,
) -> None:
    """Assert that mass is conserved within tolerance.

    Args:
        input_qty: Total input quantity.
        output_qty: Total output quantity.
        waste_qty: Total waste/loss quantity.
        tolerance_pct: Acceptable variance percentage.

    Raises:
        AssertionError: If mass conservation is violated.
    """
    total_output = output_qty + waste_qty
    if input_qty == 0:
        assert total_output == 0, "Output must be zero when input is zero"
        return
    variance_pct = abs(input_qty - total_output) / input_qty * 100.0
    assert variance_pct <= tolerance_pct, (
        f"Mass conservation violated: input={input_qty}, "
        f"output={output_qty}, waste={waste_qty}, "
        f"variance={variance_pct:.2f}% (tolerance={tolerance_pct}%)"
    )


def assert_origin_preserved(batch: Dict[str, Any]) -> None:
    """Assert that origin plot percentages sum to 100%.

    Args:
        batch: Batch dict with origin_plots field.

    Raises:
        AssertionError: If percentages do not sum to 100%.
    """
    if not batch.get("origin_plots"):
        return
    total = sum(p["percentage"] for p in batch["origin_plots"])
    assert abs(total - 100.0) < 0.01, (
        f"Origin percentages sum to {total}%, expected 100%"
    )


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


def assert_valid_completeness_score(score: float) -> None:
    """Assert that a completeness score is within [0, 100]."""
    assert isinstance(score, (int, float)), f"Score must be numeric, got {type(score)}"
    assert 0.0 <= score <= 100.0, f"Score {score} out of bounds [0, 100]"


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> Dict[str, Any]:
    """Create a ChainOfCustodyConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "max_batch_size": 10_000,
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-COC-009-TEST-GENESIS",
        "enable_metrics": False,
        "pool_size": 5,
        "custody_gap_threshold_hours": 72,
        "mass_balance_tolerance_pct": 1.0,
        "loss_tolerance_pct": 5.0,
        "credit_period_months": 12,
        "document_expiry_alert_days": [30, 14, 7],
        "risk_category_weights": dict(RISK_CATEGORY_WEIGHTS),
        "conversion_factors": dict(CONVERSION_FACTORS),
        "required_documents": dict(REQUIRED_DOCUMENTS),
        "coc_model_rules": dict(COC_MODEL_RULES),
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.chain_of_custody.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def custody_event_tracker(config):
    """Create a CustodyEventTracker instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.custody_event_tracker import (
            CustodyEventTracker,
        )
        return CustodyEventTracker(config=config)
    except ImportError:
        pytest.skip("CustodyEventTracker not yet implemented")


@pytest.fixture
def batch_lifecycle_manager(config):
    """Create a BatchLifecycleManager instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.batch_lifecycle_manager import (
            BatchLifecycleManager,
        )
        return BatchLifecycleManager(config=config)
    except ImportError:
        pytest.skip("BatchLifecycleManager not yet implemented")


@pytest.fixture
def coc_model_enforcer(config):
    """Create a CoCModelEnforcer instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.coc_model_enforcer import (
            CoCModelEnforcer,
        )
        return CoCModelEnforcer(config=config)
    except ImportError:
        pytest.skip("CoCModelEnforcer not yet implemented")


@pytest.fixture
def mass_balance_engine(config):
    """Create a MassBalanceEngine instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.mass_balance_engine import (
            MassBalanceEngine,
        )
        return MassBalanceEngine(config=config)
    except ImportError:
        pytest.skip("MassBalanceEngine not yet implemented")


@pytest.fixture
def transformation_tracker(config):
    """Create a TransformationTracker instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.transformation_tracker import (
            TransformationTracker,
        )
        return TransformationTracker(config=config)
    except ImportError:
        pytest.skip("TransformationTracker not yet implemented")


@pytest.fixture
def document_chain_verifier(config):
    """Create a DocumentChainVerifier instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.document_chain_verifier import (
            DocumentChainVerifier,
        )
        return DocumentChainVerifier(config=config)
    except ImportError:
        pytest.skip("DocumentChainVerifier not yet implemented")


@pytest.fixture
def chain_integrity_verifier(config):
    """Create a ChainIntegrityVerifier instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.chain_integrity_verifier import (
            ChainIntegrityVerifier,
        )
        return ChainIntegrityVerifier(config=config)
    except ImportError:
        pytest.skip("ChainIntegrityVerifier not yet implemented")


@pytest.fixture
def compliance_reporter(config):
    """Create a ComplianceReporter instance for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.compliance_reporter import (
            ComplianceReporter,
        )
        return ComplianceReporter(config=config)
    except ImportError:
        pytest.skip("ComplianceReporter not yet implemented")


@pytest.fixture
def service(config):
    """Create the top-level ChainOfCustodyService facade for testing."""
    try:
        from greenlang.agents.eudr.chain_of_custody.setup import (
            ChainOfCustodyService,
        )
        return ChainOfCustodyService(config=config)
    except ImportError:
        pytest.skip("ChainOfCustodyService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_events() -> List[Dict[str, Any]]:
    """Return all 10+ sample custody events."""
    return [copy.deepcopy(e) for e in ALL_SAMPLE_EVENTS]


@pytest.fixture
def sample_batches() -> List[Dict[str, Any]]:
    """Return all 14+ sample batches across 7 commodities."""
    return [copy.deepcopy(b) for b in ALL_SAMPLE_BATCHES]


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Return all sample documents."""
    return [copy.deepcopy(d) for d in SAMPLE_DOCUMENTS]


@pytest.fixture
def sample_transformations() -> List[Dict[str, Any]]:
    """Return all sample transformations."""
    return [copy.deepcopy(t) for t in ALL_SAMPLE_TRANSFORMATIONS]


@pytest.fixture
def cocoa_chain() -> Dict[str, Any]:
    """Return a complete cocoa custody chain."""
    return build_cocoa_chain()


@pytest.fixture
def palm_oil_chain() -> Dict[str, Any]:
    """Return a complete palm oil custody chain."""
    return build_palm_oil_chain()


@pytest.fixture
def coffee_chain() -> Dict[str, Any]:
    """Return a complete coffee custody chain."""
    return build_coffee_chain()


@pytest.fixture
def linear_genealogy() -> List[Dict[str, Any]]:
    """Return a 5-level linear batch genealogy."""
    return build_linear_genealogy(depth=5)


@pytest.fixture
def split_genealogy() -> List[Dict[str, Any]]:
    """Return a 3-way split batch genealogy."""
    return build_split_genealogy(n_splits=3)


@pytest.fixture
def merge_genealogy() -> List[Dict[str, Any]]:
    """Return a 3-way merge batch genealogy."""
    return build_merge_genealogy(n_inputs=3)


@pytest.fixture
def diamond_genealogy() -> List[Dict[str, Any]]:
    """Return a diamond-shaped batch genealogy."""
    return build_diamond_genealogy()


@pytest.fixture
def blend_genealogy() -> List[Dict[str, Any]]:
    """Return a blend batch genealogy."""
    return build_blend_genealogy()
