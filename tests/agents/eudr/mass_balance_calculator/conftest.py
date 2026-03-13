# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-EUDR-011 Mass Balance Calculator Agent test suite.

Provides reusable fixtures for ledgers, entries, credit periods, conversion
factors, overdraft events, loss records, carry-forward operations,
reconciliations, facility groups, consolidation reports, helper factories,
assertion helpers, reference data constants, and engine fixtures.

Sample Ledgers:
    LEDGER_COCOA_MILL_MY, LEDGER_PALM_REFINERY_ID, LEDGER_COFFEE_WAREHOUSE_NL

Sample Entries:
    ENTRY_INPUT_COCOA, ENTRY_OUTPUT_COCOA, ENTRY_ADJUSTMENT_COCOA

Sample Credit Periods:
    PERIOD_COCOA_RSPO, PERIOD_PALM_ISCC

Sample Conversion Factors:
    FACTOR_COCOA_ROASTING, FACTOR_PALM_EXTRACTION

Sample Loss Records:
    LOSS_COCOA_PROCESSING, LOSS_PALM_TRANSPORT

Sample Carry Forward Records:
    CF_COCOA_Q1_TO_Q2

Helper Factories: make_ledger(), make_entry(), make_period(), make_factor(),
    make_loss_record(), make_carry_forward(), make_reconciliation(),
    make_facility_group()

Assertion Helpers: assert_valid_provenance_hash(), assert_valid_balance(),
    assert_valid_variance(), assert_valid_score()

Reference Data Constants: REPORT_FORMATS, REPORT_TYPES, EUDR_COMMODITIES,
    SHA256_HEX_LENGTH, STANDARDS, ENTRY_TYPES, PERIOD_STATUSES,
    OVERDRAFT_MODES, OVERDRAFT_SEVERITIES, LOSS_TYPES, WASTE_TYPES,
    VARIANCE_CLASSIFICATIONS

Engine Fixtures (8 engines with pytest.skip for unimplemented)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator Agent (GL-EUDR-MBC-011)
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHA256_HEX_LENGTH: int = 64

EUDR_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood",
]

STANDARDS: List[str] = [
    "rspo", "fsc", "iscc", "utz_ra", "fairtrade", "eudr_default",
]

REPORT_FORMATS: List[str] = ["json", "csv", "pdf", "eudr_xml"]

REPORT_TYPES: List[str] = [
    "reconciliation", "consolidation", "overdraft", "variance", "evidence",
]

ENTRY_TYPES: List[str] = [
    "input", "output", "adjustment", "loss", "waste",
    "carry_forward_in", "carry_forward_out",
]

PERIOD_STATUSES: List[str] = [
    "pending", "active", "reconciling", "closed",
]

OVERDRAFT_MODES: List[str] = [
    "zero_tolerance", "percentage", "absolute",
]

OVERDRAFT_SEVERITIES: List[str] = [
    "warning", "violation", "critical",
]

LOSS_TYPES: List[str] = [
    "processing_loss", "transport_loss", "storage_loss",
    "quality_rejection", "spillage", "contamination_loss",
]

WASTE_TYPES: List[str] = [
    "by_product", "waste_material", "hazardous_waste",
]

VARIANCE_CLASSIFICATIONS: List[str] = [
    "acceptable", "warning", "violation",
]

RECONCILIATION_STATUSES: List[str] = [
    "pending", "in_progress", "completed", "signed_off", "reopened",
]

FACILITY_GROUP_TYPES: List[str] = [
    "region", "country", "commodity", "custom",
]

# Credit period days per certification standard
CREDIT_PERIOD_DAYS: Dict[str, int] = {
    "rspo": 90,
    "fsc": 365,
    "iscc": 365,
    "utz_ra": 365,
    "fairtrade": 365,
    "eudr_default": 365,
}

# Carry-forward limits per standard (percentage of period-end balance)
CARRY_FORWARD_LIMITS: Dict[str, float] = {
    "rspo": 100.0,       # Full carry-forward, expires end of receiving period
    "fsc": 100.0,        # No expiry within period
    "iscc": 100.0,       # Expires end of receiving period
    "utz_ra": 50.0,      # Max 50% carry-forward
    "fairtrade": 25.0,   # Max 25% carry-forward
    "eudr_default": 100.0,
}

# Carry-forward expiry rules per standard
CARRY_FORWARD_EXPIRY: Dict[str, Optional[str]] = {
    "rspo": "end_of_receiving_period",  # 3-month expiry
    "fsc": None,                         # No expiry within period
    "iscc": "end_of_receiving_period",
    "utz_ra": "end_of_receiving_period",
    "fairtrade": "end_of_receiving_period",
    "eudr_default": "end_of_receiving_period",
}

# Commodity-specific loss tolerances (% acceptable processing loss)
COMMODITY_LOSS_TOLERANCES: Dict[str, float] = {
    "cattle": 2.0,
    "cocoa": 5.0,
    "coffee": 4.0,
    "oil_palm": 3.0,
    "rubber": 3.5,
    "soya": 2.5,
    "wood": 3.0,
}

# Loss type max tolerances (%)
LOSS_TYPE_TOLERANCES: Dict[str, float] = {
    "processing_loss": 15.0,
    "transport_loss": 3.0,
    "storage_loss": 5.0,
    "quality_rejection": 10.0,
    "spillage": 2.0,
    "contamination_loss": 5.0,
}

# Reference conversion factors (yield ratios) by commodity and process
REFERENCE_CONVERSION_FACTORS: Dict[str, Dict[str, float]] = {
    "cocoa": {
        "fermentation": 0.92,
        "drying": 0.88,
        "roasting": 0.85,
        "winnowing": 0.80,
        "grinding": 0.98,
        "pressing": 0.45,
        "conching": 0.97,
        "tempering": 0.99,
    },
    "coffee": {
        "wet_processing": 0.60,
        "dry_processing": 0.50,
        "hulling": 0.80,
        "polishing": 0.98,
        "roasting": 0.82,
    },
    "oil_palm": {
        "sterilization": 0.95,
        "threshing": 0.65,
        "digestion": 0.90,
        "extraction": 0.22,
        "clarification": 0.95,
        "refining": 0.92,
        "fractionation": 0.90,
    },
    "wood": {
        "debarking": 0.90,
        "sawing": 0.55,
        "planing": 0.90,
        "kiln_drying": 0.92,
        "milling": 0.85,
    },
    "rubber": {
        "coagulation": 0.60,
        "sheeting": 0.95,
        "smoking": 0.88,
        "crumbling": 0.92,
    },
    "soya": {
        "cleaning": 0.98,
        "dehulling": 0.92,
        "flaking": 0.97,
        "solvent_extraction": 0.82,
        "refining": 0.92,
    },
    "cattle": {
        "slaughtering": 0.55,
        "deboning": 0.70,
        "tanning": 0.30,
    },
}

# Variance classification thresholds (%)
VARIANCE_THRESHOLDS: Dict[str, float] = {
    "acceptable": 1.0,   # <= 1%
    "warning": 3.0,       # 1% - 3%
    # > 3% = violation
}

# Overdraft tolerance thresholds
OVERDRAFT_TOLERANCE: Dict[str, Any] = {
    "zero_tolerance": {"percent": 0.0, "kg": 0.0},
    "percentage": {"percent": 5.0, "kg": 0.0},
    "absolute": {"percent": 0.0, "kg": 50.0},
}

# Default grace period in days
DEFAULT_GRACE_PERIOD_DAYS: int = 5

# Default overdraft resolution deadline in hours
DEFAULT_OVERDRAFT_RESOLUTION_HOURS: int = 48


# ---------------------------------------------------------------------------
# Pre-generated Identifiers
# ---------------------------------------------------------------------------

# Facility IDs
FAC_ID_MILL_MY = "FAC-MILL-MY-001"
FAC_ID_REFINERY_ID = "FAC-REFN-ID-001"
FAC_ID_WAREHOUSE_NL = "FAC-WRHS-NL-001"
FAC_ID_FACTORY_DE = "FAC-FACT-DE-001"
FAC_ID_PORT_BR = "FAC-PORT-BR-001"

# Batch IDs
BATCH_COCOA_001 = "BATCH-COC-001"
BATCH_COCOA_002 = "BATCH-COC-002"
BATCH_PALM_001 = "BATCH-PLM-001"
BATCH_PALM_002 = "BATCH-PLM-002"
BATCH_COFFEE_001 = "BATCH-COF-001"
BATCH_SOYA_001 = "BATCH-SOY-001"
BATCH_WOOD_001 = "BATCH-WOD-001"

# Ledger IDs
LEDGER_COCOA_001 = "LDG-COC-MY-001"
LEDGER_COCOA_002 = "LDG-COC-MY-002"
LEDGER_PALM_001 = "LDG-PLM-ID-001"
LEDGER_PALM_002 = "LDG-PLM-ID-002"
LEDGER_COFFEE_001 = "LDG-COF-NL-001"

# Period IDs
PERIOD_COCOA_Q1 = "PRD-COC-MY-Q1-2026"
PERIOD_COCOA_Q2 = "PRD-COC-MY-Q2-2026"
PERIOD_PALM_Y1 = "PRD-PLM-ID-Y1-2026"

# Entry IDs
ENTRY_ID_INPUT_001 = "ENT-INP-001"
ENTRY_ID_OUTPUT_001 = "ENT-OUT-001"
ENTRY_ID_ADJ_001 = "ENT-ADJ-001"

# Reconciliation IDs
RECON_ID_001 = "REC-COC-MY-Q1-001"
RECON_ID_002 = "REC-PLM-ID-Y1-001"

# Facility Group IDs
GROUP_ID_SOUTHEAST_ASIA = "GRP-SEA-001"
GROUP_ID_EUROPE = "GRP-EUR-001"

# Carry Forward IDs
CF_ID_001 = "CF-COC-MY-Q1Q2-001"


# ---------------------------------------------------------------------------
# Timestamp Helper
# ---------------------------------------------------------------------------

def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    """Generate ISO timestamp relative to now."""
    return (
        datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    ).isoformat()


def _ts_dt(days_ago: int = 0, hours_ago: int = 0) -> datetime:
    """Generate datetime object relative to now."""
    return datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)


# ---------------------------------------------------------------------------
# Sample Ledgers
# ---------------------------------------------------------------------------

LEDGER_COCOA_MILL_MY: Dict[str, Any] = {
    "ledger_id": LEDGER_COCOA_001,
    "facility_id": FAC_ID_MILL_MY,
    "commodity": "cocoa",
    "standard": "rspo",
    "currency_unit": "kg",
    "opening_balance_kg": Decimal("0.0"),
    "current_balance_kg": Decimal("15000.0"),
    "total_inputs_kg": Decimal("20000.0"),
    "total_outputs_kg": Decimal("4500.0"),
    "total_losses_kg": Decimal("500.0"),
    "entry_count": 12,
    "status": "active",
    "created_at": _ts(days_ago=120),
    "updated_at": _ts(days_ago=1),
    "metadata": {"mill_name": "Pahang Cocoa Mill"},
}

LEDGER_PALM_REFINERY_ID: Dict[str, Any] = {
    "ledger_id": LEDGER_PALM_001,
    "facility_id": FAC_ID_REFINERY_ID,
    "commodity": "oil_palm",
    "standard": "iscc",
    "currency_unit": "kg",
    "opening_balance_kg": Decimal("0.0"),
    "current_balance_kg": Decimal("50000.0"),
    "total_inputs_kg": Decimal("80000.0"),
    "total_outputs_kg": Decimal("25000.0"),
    "total_losses_kg": Decimal("5000.0"),
    "entry_count": 45,
    "status": "active",
    "created_at": _ts(days_ago=300),
    "updated_at": _ts(days_ago=2),
    "metadata": {"refinery_name": "Kalimantan Palm Refinery"},
}

LEDGER_COFFEE_WAREHOUSE_NL: Dict[str, Any] = {
    "ledger_id": LEDGER_COFFEE_001,
    "facility_id": FAC_ID_WAREHOUSE_NL,
    "commodity": "coffee",
    "standard": "fairtrade",
    "currency_unit": "kg",
    "opening_balance_kg": Decimal("0.0"),
    "current_balance_kg": Decimal("8000.0"),
    "total_inputs_kg": Decimal("10000.0"),
    "total_outputs_kg": Decimal("1800.0"),
    "total_losses_kg": Decimal("200.0"),
    "entry_count": 8,
    "status": "active",
    "created_at": _ts(days_ago=90),
    "updated_at": _ts(days_ago=5),
    "metadata": {"warehouse_name": "Rotterdam Coffee Warehouse"},
}

ALL_SAMPLE_LEDGERS: List[Dict[str, Any]] = [
    LEDGER_COCOA_MILL_MY, LEDGER_PALM_REFINERY_ID, LEDGER_COFFEE_WAREHOUSE_NL,
]


# ---------------------------------------------------------------------------
# Sample Entries
# ---------------------------------------------------------------------------

ENTRY_INPUT_COCOA: Dict[str, Any] = {
    "entry_id": ENTRY_ID_INPUT_001,
    "ledger_id": LEDGER_COCOA_001,
    "entry_type": "input",
    "quantity_kg": Decimal("5000.0"),
    "batch_id": BATCH_COCOA_001,
    "description": "Incoming cocoa beans from Pahang supplier",
    "timestamp": _ts(days_ago=30),
    "recorded_by": "operator-001",
    "provenance_hash": None,
    "metadata": {"supplier_id": "SUP-MY-001"},
}

ENTRY_OUTPUT_COCOA: Dict[str, Any] = {
    "entry_id": ENTRY_ID_OUTPUT_001,
    "ledger_id": LEDGER_COCOA_001,
    "entry_type": "output",
    "quantity_kg": Decimal("4000.0"),
    "batch_id": BATCH_COCOA_001,
    "description": "Processed cocoa nibs shipped to EU",
    "timestamp": _ts(days_ago=20),
    "recorded_by": "operator-002",
    "provenance_hash": None,
    "metadata": {"destination": "FAC-FACT-DE-001"},
}

ENTRY_ADJUSTMENT_COCOA: Dict[str, Any] = {
    "entry_id": ENTRY_ID_ADJ_001,
    "ledger_id": LEDGER_COCOA_001,
    "entry_type": "adjustment",
    "quantity_kg": Decimal("200.0"),
    "batch_id": BATCH_COCOA_001,
    "description": "Correction: re-weigh found +200kg",
    "timestamp": _ts(days_ago=15),
    "recorded_by": "supervisor-001",
    "provenance_hash": None,
    "metadata": {"reason": "scale_calibration"},
}

ALL_SAMPLE_ENTRIES: List[Dict[str, Any]] = [
    ENTRY_INPUT_COCOA, ENTRY_OUTPUT_COCOA, ENTRY_ADJUSTMENT_COCOA,
]


# ---------------------------------------------------------------------------
# Sample Credit Periods
# ---------------------------------------------------------------------------

PERIOD_COCOA_RSPO: Dict[str, Any] = {
    "period_id": PERIOD_COCOA_Q1,
    "facility_id": FAC_ID_MILL_MY,
    "commodity": "cocoa",
    "standard": "rspo",
    "status": "active",
    "start_date": _ts(days_ago=60),
    "end_date": _ts(days_ago=-30),
    "duration_days": 90,
    "grace_period_days": 5,
    "entry_count": 12,
    "total_inputs_kg": Decimal("20000.0"),
    "total_outputs_kg": Decimal("4500.0"),
    "closing_balance_kg": Decimal("15000.0"),
    "created_at": _ts(days_ago=60),
    "metadata": {},
}

PERIOD_PALM_ISCC: Dict[str, Any] = {
    "period_id": PERIOD_PALM_Y1,
    "facility_id": FAC_ID_REFINERY_ID,
    "commodity": "oil_palm",
    "standard": "iscc",
    "status": "active",
    "start_date": _ts(days_ago=200),
    "end_date": _ts(days_ago=-165),
    "duration_days": 365,
    "grace_period_days": 5,
    "entry_count": 45,
    "total_inputs_kg": Decimal("80000.0"),
    "total_outputs_kg": Decimal("25000.0"),
    "closing_balance_kg": Decimal("50000.0"),
    "created_at": _ts(days_ago=200),
    "metadata": {},
}

ALL_SAMPLE_PERIODS: List[Dict[str, Any]] = [
    PERIOD_COCOA_RSPO, PERIOD_PALM_ISCC,
]


# ---------------------------------------------------------------------------
# Sample Conversion Factors
# ---------------------------------------------------------------------------

FACTOR_COCOA_ROASTING: Dict[str, Any] = {
    "factor_id": "CF-COC-RST-001",
    "commodity": "cocoa",
    "process": "roasting",
    "yield_ratio": 0.85,
    "reference_ratio": 0.85,
    "deviation_percent": 0.0,
    "status": "accepted",
    "facility_id": FAC_ID_MILL_MY,
    "season": None,
    "registered_at": _ts(days_ago=60),
    "metadata": {},
}

FACTOR_PALM_EXTRACTION: Dict[str, Any] = {
    "factor_id": "CF-PLM-EXT-001",
    "commodity": "oil_palm",
    "process": "extraction",
    "yield_ratio": 0.22,
    "reference_ratio": 0.22,
    "deviation_percent": 0.0,
    "status": "accepted",
    "facility_id": FAC_ID_REFINERY_ID,
    "season": None,
    "registered_at": _ts(days_ago=180),
    "metadata": {},
}

ALL_SAMPLE_FACTORS: List[Dict[str, Any]] = [
    FACTOR_COCOA_ROASTING, FACTOR_PALM_EXTRACTION,
]


# ---------------------------------------------------------------------------
# Sample Loss Records
# ---------------------------------------------------------------------------

LOSS_COCOA_PROCESSING: Dict[str, Any] = {
    "loss_id": "LOSS-COC-PRC-001",
    "ledger_id": LEDGER_COCOA_001,
    "loss_type": "processing_loss",
    "quantity_kg": Decimal("500.0"),
    "commodity": "cocoa",
    "batch_id": BATCH_COCOA_001,
    "process_step": "roasting",
    "input_quantity_kg": Decimal("5000.0"),
    "loss_percent": 10.0,
    "tolerance_percent": 15.0,
    "within_tolerance": True,
    "recorded_at": _ts(days_ago=25),
    "metadata": {},
}

LOSS_PALM_TRANSPORT: Dict[str, Any] = {
    "loss_id": "LOSS-PLM-TRN-001",
    "ledger_id": LEDGER_PALM_001,
    "loss_type": "transport_loss",
    "quantity_kg": Decimal("150.0"),
    "commodity": "oil_palm",
    "batch_id": BATCH_PALM_001,
    "process_step": None,
    "input_quantity_kg": Decimal("10000.0"),
    "loss_percent": 1.5,
    "tolerance_percent": 3.0,
    "within_tolerance": True,
    "recorded_at": _ts(days_ago=10),
    "metadata": {"route": "Kalimantan-Rotterdam"},
}

ALL_SAMPLE_LOSSES: List[Dict[str, Any]] = [
    LOSS_COCOA_PROCESSING, LOSS_PALM_TRANSPORT,
]


# ---------------------------------------------------------------------------
# Sample Carry Forward Records
# ---------------------------------------------------------------------------

CF_COCOA_Q1_TO_Q2: Dict[str, Any] = {
    "carry_forward_id": CF_ID_001,
    "from_period_id": PERIOD_COCOA_Q1,
    "to_period_id": PERIOD_COCOA_Q2,
    "facility_id": FAC_ID_MILL_MY,
    "commodity": "cocoa",
    "standard": "rspo",
    "amount_kg": Decimal("15000.0"),
    "cap_applied": False,
    "cap_percent": 100.0,
    "status": "active",
    "expires_at": _ts(days_ago=-85),
    "created_at": _ts(days_ago=5),
    "voided_at": None,
    "metadata": {},
}

ALL_SAMPLE_CARRY_FORWARDS: List[Dict[str, Any]] = [
    CF_COCOA_Q1_TO_Q2,
]


# ---------------------------------------------------------------------------
# Sample Reconciliation Records
# ---------------------------------------------------------------------------

RECONCILIATION_COCOA_Q1: Dict[str, Any] = {
    "reconciliation_id": RECON_ID_001,
    "period_id": PERIOD_COCOA_Q1,
    "facility_id": FAC_ID_MILL_MY,
    "commodity": "cocoa",
    "standard": "rspo",
    "expected_balance_kg": Decimal("15000.0"),
    "recorded_balance_kg": Decimal("15000.0"),
    "variance_kg": Decimal("0.0"),
    "variance_percent": 0.0,
    "classification": "acceptable",
    "anomalies": [],
    "status": "completed",
    "signed_off_by": None,
    "signed_off_at": None,
    "created_at": _ts(days_ago=3),
    "metadata": {},
}

ALL_SAMPLE_RECONCILIATIONS: List[Dict[str, Any]] = [
    RECONCILIATION_COCOA_Q1,
]


# ---------------------------------------------------------------------------
# Sample Facility Groups
# ---------------------------------------------------------------------------

GROUP_SOUTHEAST_ASIA: Dict[str, Any] = {
    "group_id": GROUP_ID_SOUTHEAST_ASIA,
    "name": "Southeast Asia Operations",
    "group_type": "region",
    "facility_ids": [FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
    "commodities": ["cocoa", "oil_palm"],
    "created_at": _ts(days_ago=180),
    "metadata": {"region": "APAC"},
}

GROUP_EUROPE: Dict[str, Any] = {
    "group_id": GROUP_ID_EUROPE,
    "name": "European Operations",
    "group_type": "region",
    "facility_ids": [FAC_ID_WAREHOUSE_NL, FAC_ID_FACTORY_DE],
    "commodities": ["coffee", "cocoa"],
    "created_at": _ts(days_ago=180),
    "metadata": {"region": "EU"},
}

ALL_SAMPLE_GROUPS: List[Dict[str, Any]] = [
    GROUP_SOUTHEAST_ASIA, GROUP_EUROPE,
]


# ---------------------------------------------------------------------------
# Helper Factories
# ---------------------------------------------------------------------------

def make_ledger(
    facility_id: str = FAC_ID_MILL_MY,
    commodity: str = "cocoa",
    standard: str = "rspo",
    ledger_id: Optional[str] = None,
    opening_balance_kg: Decimal = Decimal("0.0"),
    status: str = "active",
) -> Dict[str, Any]:
    """Build a ledger dictionary for testing.

    Args:
        facility_id: Facility identifier.
        commodity: EUDR commodity type.
        standard: Certification standard.
        ledger_id: Ledger identifier (auto-generated if None).
        opening_balance_kg: Opening balance in kilograms.
        status: Ledger status (active, closed).

    Returns:
        Dict with all ledger fields.
    """
    return {
        "ledger_id": ledger_id or f"LDG-{uuid.uuid4().hex[:12].upper()}",
        "facility_id": facility_id,
        "commodity": commodity,
        "standard": standard,
        "currency_unit": "kg",
        "opening_balance_kg": opening_balance_kg,
        "current_balance_kg": opening_balance_kg,
        "total_inputs_kg": Decimal("0.0"),
        "total_outputs_kg": Decimal("0.0"),
        "total_losses_kg": Decimal("0.0"),
        "entry_count": 0,
        "status": status,
        "created_at": _ts(),
        "updated_at": _ts(),
        "metadata": {},
    }


def make_entry(
    ledger_id: str = LEDGER_COCOA_001,
    entry_type: str = "input",
    quantity_kg: Decimal = Decimal("1000.0"),
    batch_id: str = BATCH_COCOA_001,
    entry_id: Optional[str] = None,
    description: str = "Test entry",
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a ledger entry dictionary for testing.

    Args:
        ledger_id: Parent ledger identifier.
        entry_type: Entry type (input, output, adjustment, loss, waste).
        quantity_kg: Entry quantity in kilograms.
        batch_id: Associated batch identifier.
        entry_id: Entry identifier (auto-generated if None).
        description: Human-readable description.
        timestamp: Entry timestamp (defaults to now).

    Returns:
        Dict with all entry fields.
    """
    return {
        "entry_id": entry_id or f"ENT-{uuid.uuid4().hex[:8].upper()}",
        "ledger_id": ledger_id,
        "entry_type": entry_type,
        "quantity_kg": quantity_kg,
        "batch_id": batch_id,
        "description": description,
        "timestamp": timestamp or _ts(),
        "recorded_by": "test-operator",
        "provenance_hash": None,
        "metadata": {},
    }


def make_period(
    facility_id: str = FAC_ID_MILL_MY,
    commodity: str = "cocoa",
    standard: str = "rspo",
    duration_days: int = 90,
    period_id: Optional[str] = None,
    status: str = "active",
    start_days_ago: int = 60,
    grace_period_days: int = DEFAULT_GRACE_PERIOD_DAYS,
) -> Dict[str, Any]:
    """Build a credit period dictionary for testing.

    Args:
        facility_id: Facility identifier.
        commodity: EUDR commodity type.
        standard: Certification standard.
        duration_days: Period duration in days.
        period_id: Period identifier (auto-generated if None).
        status: Period status (pending, active, reconciling, closed).
        start_days_ago: Days since the period started.
        grace_period_days: Grace period in days.

    Returns:
        Dict with all period fields.
    """
    return {
        "period_id": period_id or f"PRD-{uuid.uuid4().hex[:8].upper()}",
        "facility_id": facility_id,
        "commodity": commodity,
        "standard": standard,
        "status": status,
        "start_date": _ts(days_ago=start_days_ago),
        "end_date": _ts(days_ago=start_days_ago - duration_days),
        "duration_days": duration_days,
        "grace_period_days": grace_period_days,
        "entry_count": 0,
        "total_inputs_kg": Decimal("0.0"),
        "total_outputs_kg": Decimal("0.0"),
        "closing_balance_kg": Decimal("0.0"),
        "created_at": _ts(days_ago=start_days_ago),
        "metadata": {},
    }


def make_factor(
    commodity: str = "cocoa",
    process: str = "roasting",
    yield_ratio: float = 0.85,
    factor_id: Optional[str] = None,
    facility_id: str = FAC_ID_MILL_MY,
    season: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a conversion factor dictionary for testing.

    Args:
        commodity: EUDR commodity type.
        process: Processing step name.
        yield_ratio: Output/input mass ratio.
        factor_id: Factor identifier (auto-generated if None).
        facility_id: Facility where this factor applies.
        season: Optional season qualifier (e.g. 'dry', 'wet').

    Returns:
        Dict with all factor fields.
    """
    ref = REFERENCE_CONVERSION_FACTORS.get(commodity, {}).get(process, yield_ratio)
    deviation = abs(yield_ratio - ref) / ref * 100.0 if ref else 0.0
    return {
        "factor_id": factor_id or f"CF-{uuid.uuid4().hex[:8].upper()}",
        "commodity": commodity,
        "process": process,
        "yield_ratio": yield_ratio,
        "reference_ratio": ref,
        "deviation_percent": round(deviation, 2),
        "status": "accepted" if deviation < 5.0 else "warning",
        "facility_id": facility_id,
        "season": season,
        "registered_at": _ts(),
        "metadata": {},
    }


def make_loss_record(
    ledger_id: str = LEDGER_COCOA_001,
    loss_type: str = "processing_loss",
    quantity_kg: Decimal = Decimal("100.0"),
    commodity: str = "cocoa",
    loss_id: Optional[str] = None,
    batch_id: str = BATCH_COCOA_001,
    input_quantity_kg: Decimal = Decimal("1000.0"),
    process_step: Optional[str] = "roasting",
) -> Dict[str, Any]:
    """Build a loss record dictionary for testing.

    Args:
        ledger_id: Parent ledger identifier.
        loss_type: Type of loss.
        quantity_kg: Loss quantity in kilograms.
        commodity: EUDR commodity type.
        loss_id: Loss record identifier (auto-generated if None).
        batch_id: Associated batch identifier.
        input_quantity_kg: Total input quantity for loss percentage calculation.
        process_step: Processing step where loss occurred.

    Returns:
        Dict with all loss record fields.
    """
    loss_pct = float(quantity_kg / input_quantity_kg * 100) if input_quantity_kg else 0.0
    tolerance = LOSS_TYPE_TOLERANCES.get(loss_type, 15.0)
    return {
        "loss_id": loss_id or f"LOSS-{uuid.uuid4().hex[:8].upper()}",
        "ledger_id": ledger_id,
        "loss_type": loss_type,
        "quantity_kg": quantity_kg,
        "commodity": commodity,
        "batch_id": batch_id,
        "process_step": process_step,
        "input_quantity_kg": input_quantity_kg,
        "loss_percent": round(loss_pct, 2),
        "tolerance_percent": tolerance,
        "within_tolerance": loss_pct <= tolerance,
        "recorded_at": _ts(),
        "metadata": {},
    }


def make_carry_forward(
    from_period: str = PERIOD_COCOA_Q1,
    to_period: str = PERIOD_COCOA_Q2,
    amount_kg: Decimal = Decimal("5000.0"),
    carry_forward_id: Optional[str] = None,
    facility_id: str = FAC_ID_MILL_MY,
    commodity: str = "cocoa",
    standard: str = "rspo",
    status: str = "active",
    expires_in_days: int = 90,
) -> Dict[str, Any]:
    """Build a carry-forward record dictionary for testing.

    Args:
        from_period: Source period identifier.
        to_period: Destination period identifier.
        amount_kg: Carry-forward amount in kilograms.
        carry_forward_id: Record identifier (auto-generated if None).
        facility_id: Facility identifier.
        commodity: EUDR commodity type.
        standard: Certification standard.
        status: Carry-forward status (active, expired, voided, utilized).
        expires_in_days: Days until carry-forward expires.

    Returns:
        Dict with all carry-forward fields.
    """
    cap_pct = CARRY_FORWARD_LIMITS.get(standard, 100.0)
    return {
        "carry_forward_id": carry_forward_id or f"CF-{uuid.uuid4().hex[:8].upper()}",
        "from_period_id": from_period,
        "to_period_id": to_period,
        "facility_id": facility_id,
        "commodity": commodity,
        "standard": standard,
        "amount_kg": amount_kg,
        "cap_applied": False,
        "cap_percent": cap_pct,
        "status": status,
        "expires_at": _ts(days_ago=-expires_in_days),
        "created_at": _ts(),
        "voided_at": None,
        "metadata": {},
    }


def make_reconciliation(
    period_id: str = PERIOD_COCOA_Q1,
    expected: Decimal = Decimal("15000.0"),
    recorded: Decimal = Decimal("15000.0"),
    reconciliation_id: Optional[str] = None,
    facility_id: str = FAC_ID_MILL_MY,
    commodity: str = "cocoa",
    standard: str = "rspo",
) -> Dict[str, Any]:
    """Build a reconciliation record dictionary for testing.

    Args:
        period_id: Credit period identifier.
        expected: Expected balance in kilograms.
        recorded: Recorded balance in kilograms.
        reconciliation_id: Record identifier (auto-generated if None).
        facility_id: Facility identifier.
        commodity: EUDR commodity type.
        standard: Certification standard.

    Returns:
        Dict with all reconciliation fields.
    """
    variance_kg = recorded - expected
    variance_pct = float(abs(variance_kg) / expected * 100) if expected else 0.0
    if variance_pct <= VARIANCE_THRESHOLDS["acceptable"]:
        classification = "acceptable"
    elif variance_pct <= VARIANCE_THRESHOLDS["warning"]:
        classification = "warning"
    else:
        classification = "violation"
    return {
        "reconciliation_id": reconciliation_id or f"REC-{uuid.uuid4().hex[:8].upper()}",
        "period_id": period_id,
        "facility_id": facility_id,
        "commodity": commodity,
        "standard": standard,
        "expected_balance_kg": expected,
        "recorded_balance_kg": recorded,
        "variance_kg": variance_kg,
        "variance_percent": round(variance_pct, 4),
        "classification": classification,
        "anomalies": [],
        "status": "completed",
        "signed_off_by": None,
        "signed_off_at": None,
        "created_at": _ts(),
        "metadata": {},
    }


def make_facility_group(
    name: str = "Test Group",
    facility_ids: Optional[List[str]] = None,
    group_id: Optional[str] = None,
    group_type: str = "region",
    commodities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a facility group dictionary for testing.

    Args:
        name: Human-readable group name.
        facility_ids: List of facility identifiers in the group.
        group_id: Group identifier (auto-generated if None).
        group_type: Group type (region, country, commodity, custom).
        commodities: List of commodities in the group.

    Returns:
        Dict with all facility group fields.
    """
    return {
        "group_id": group_id or f"GRP-{uuid.uuid4().hex[:8].upper()}",
        "name": name,
        "group_type": group_type,
        "facility_ids": facility_ids or [FAC_ID_MILL_MY, FAC_ID_REFINERY_ID],
        "commodities": commodities or ["cocoa", "oil_palm"],
        "created_at": _ts(),
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


def assert_valid_balance(balance: Any) -> None:
    """Assert that a balance value is a valid non-negative Decimal.

    Args:
        balance: The balance value to validate.

    Raises:
        AssertionError: If balance is not a valid non-negative Decimal.
    """
    assert isinstance(balance, (Decimal, int, float)), (
        f"Balance must be numeric, got {type(balance)}"
    )
    assert Decimal(str(balance)) >= Decimal("0"), (
        f"Balance must be >= 0, got {balance}"
    )


def assert_valid_variance(variance: Dict[str, Any]) -> None:
    """Assert that a variance record contains required fields.

    Args:
        variance: The variance dictionary to validate.

    Raises:
        AssertionError: If required fields are missing.
    """
    assert "absolute" in variance or "variance_kg" in variance, (
        "Variance must have 'absolute' or 'variance_kg' field"
    )
    assert "percentage" in variance or "variance_percent" in variance, (
        "Variance must have 'percentage' or 'variance_percent' field"
    )
    assert "classification" in variance, (
        "Variance must have 'classification' field"
    )
    classification = variance["classification"]
    assert classification in VARIANCE_CLASSIFICATIONS, (
        f"Classification must be one of {VARIANCE_CLASSIFICATIONS}, got {classification}"
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


# ---------------------------------------------------------------------------
# Configuration Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mbc_config() -> Dict[str, Any]:
    """Create a MassBalanceCalculatorConfig-compatible dictionary with test defaults."""
    return {
        "database_url": "postgresql://localhost:5432/greenlang_test",
        "redis_url": "redis://localhost:6379/1",
        "log_level": "DEBUG",
        "enable_provenance": True,
        "genesis_hash": "GL-EUDR-MBC-011-TEST-GENESIS",
        "enable_metrics": False,
        "pool_size": 5,
        "overdraft_mode": "zero_tolerance",
        "overdraft_tolerance_percent": 0.0,
        "overdraft_tolerance_kg": 0.0,
        "overdraft_resolution_hours": DEFAULT_OVERDRAFT_RESOLUTION_HOURS,
        "default_credit_period_days": 365,
        "rspo_credit_period_days": 90,
        "fsc_credit_period_days": 365,
        "iscc_credit_period_days": 365,
        "grace_period_days": DEFAULT_GRACE_PERIOD_DAYS,
        "max_carry_forward_percent": 100.0,
        "conversion_factor_warn_deviation": 0.05,
        "conversion_factor_reject_deviation": 0.15,
        "variance_acceptable_percent": 1.0,
        "variance_warning_percent": 3.0,
        "loss_validation_enabled": True,
        "by_product_credit_enabled": True,
        "batch_max_size": 500,
        "batch_concurrency": 4,
        "batch_timeout_s": 120,
        "retention_years": 5,
        "report_default_format": "json",
        "report_retention_days": 1825,
        "eudr_commodities": list(EUDR_COMMODITIES),
        "commodity_loss_tolerances": dict(COMMODITY_LOSS_TOLERANCES),
        "loss_type_tolerances": dict(LOSS_TYPE_TOLERANCES),
        "reference_conversion_factors": {
            k: dict(v) for k, v in REFERENCE_CONVERSION_FACTORS.items()
        },
        "credit_period_days": dict(CREDIT_PERIOD_DAYS),
        "carry_forward_limits": dict(CARRY_FORWARD_LIMITS),
        "variance_thresholds": dict(VARIANCE_THRESHOLDS),
        "reconciliation_auto_rollover": True,
        "anomaly_detection_enabled": True,
        "trend_window_periods": 6,
        "min_entries_for_trend": 10,
    }


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset any singleton state between tests to prevent cross-test contamination."""
    yield
    try:
        from greenlang.agents.eudr.mass_balance_calculator.config import reset_config
        reset_config()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Engine Fixtures (with graceful pytest.skip for unimplemented)
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger_manager(mbc_config):
    """Create a LedgerManager instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.ledger_manager import (
            LedgerManager,
        )
        return LedgerManager(config=mbc_config)
    except ImportError:
        pytest.skip("LedgerManager not yet implemented")


@pytest.fixture
def credit_period_engine(mbc_config):
    """Create a CreditPeriodEngine instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.credit_period_engine import (
            CreditPeriodEngine,
        )
        return CreditPeriodEngine(config=mbc_config)
    except ImportError:
        pytest.skip("CreditPeriodEngine not yet implemented")


@pytest.fixture
def conversion_factor_validator(mbc_config):
    """Create a ConversionFactorValidator instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.conversion_factor_validator import (
            ConversionFactorValidator,
        )
        return ConversionFactorValidator(config=mbc_config)
    except ImportError:
        pytest.skip("ConversionFactorValidator not yet implemented")


@pytest.fixture
def overdraft_detector(mbc_config):
    """Create an OverdraftDetector instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.overdraft_detector import (
            OverdraftDetector,
        )
        return OverdraftDetector(config=mbc_config)
    except ImportError:
        pytest.skip("OverdraftDetector not yet implemented")


@pytest.fixture
def loss_waste_tracker(mbc_config):
    """Create a LossWasteTracker instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.loss_waste_tracker import (
            LossWasteTracker,
        )
        return LossWasteTracker(config=mbc_config)
    except ImportError:
        pytest.skip("LossWasteTracker not yet implemented")


@pytest.fixture
def carry_forward_manager(mbc_config):
    """Create a CarryForwardManager instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.carry_forward_manager import (
            CarryForwardManager,
        )
        return CarryForwardManager(config=mbc_config)
    except ImportError:
        pytest.skip("CarryForwardManager not yet implemented")


@pytest.fixture
def reconciliation_engine(mbc_config):
    """Create a ReconciliationEngine instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.reconciliation_engine import (
            ReconciliationEngine,
        )
        return ReconciliationEngine(config=mbc_config)
    except ImportError:
        pytest.skip("ReconciliationEngine not yet implemented")


@pytest.fixture
def consolidation_reporter(mbc_config):
    """Create a ConsolidationReporter instance for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.consolidation_reporter import (
            ConsolidationReporter,
        )
        return ConsolidationReporter(config=mbc_config)
    except ImportError:
        pytest.skip("ConsolidationReporter not yet implemented")


@pytest.fixture
def service(mbc_config):
    """Create the top-level MassBalanceCalculatorService facade for testing."""
    try:
        from greenlang.agents.eudr.mass_balance_calculator.setup import (
            MassBalanceCalculatorService,
        )
        return MassBalanceCalculatorService(config=mbc_config)
    except ImportError:
        pytest.skip("MassBalanceCalculatorService not yet implemented")


# ---------------------------------------------------------------------------
# Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ledger() -> Dict[str, Any]:
    """Return a sample cocoa ledger."""
    return copy.deepcopy(LEDGER_COCOA_MILL_MY)


@pytest.fixture
def sample_entry() -> Dict[str, Any]:
    """Return a sample input entry."""
    return copy.deepcopy(ENTRY_INPUT_COCOA)


@pytest.fixture
def sample_period() -> Dict[str, Any]:
    """Return a sample RSPO credit period."""
    return copy.deepcopy(PERIOD_COCOA_RSPO)


@pytest.fixture
def sample_factor() -> Dict[str, Any]:
    """Return a sample cocoa roasting conversion factor."""
    return copy.deepcopy(FACTOR_COCOA_ROASTING)


@pytest.fixture
def sample_loss() -> Dict[str, Any]:
    """Return a sample processing loss record."""
    return copy.deepcopy(LOSS_COCOA_PROCESSING)


@pytest.fixture
def sample_carry_forward() -> Dict[str, Any]:
    """Return a sample carry-forward record."""
    return copy.deepcopy(CF_COCOA_Q1_TO_Q2)


@pytest.fixture
def sample_reconciliation() -> Dict[str, Any]:
    """Return a sample reconciliation record."""
    return copy.deepcopy(RECONCILIATION_COCOA_Q1)


@pytest.fixture
def sample_facility_group() -> Dict[str, Any]:
    """Return a sample facility group."""
    return copy.deepcopy(GROUP_SOUTHEAST_ASIA)


@pytest.fixture(params=EUDR_COMMODITIES)
def commodity(request) -> str:
    """Parametrize across all 7 EUDR commodities."""
    return request.param


@pytest.fixture(params=STANDARDS)
def standard(request) -> str:
    """Parametrize across all 6 certification standards."""
    return request.param


@pytest.fixture(params=ENTRY_TYPES)
def entry_type(request) -> str:
    """Parametrize across all entry types."""
    return request.param


@pytest.fixture(params=REPORT_FORMATS)
def report_format(request) -> str:
    """Parametrize across all report formats."""
    return request.param
