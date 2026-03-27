# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Shared Test Fixtures

Provides reusable fixtures for all test modules including sample entity records
(parent company, 3 subsidiaries, 1 JV, 1 associate), ownership records,
entity emissions data, boundary definitions, transfer records, M&A events,
adjustment records, and configuration fixtures.

All numeric values use Decimal for deterministic arithmetic.
SHA-256 hashes are verified where applicable.
"""

import sys
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the pack root is on sys.path so engines/ etc. are importable
# ---------------------------------------------------------------------------
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
from config.pack_config import (
    PackConfig,
    ConsolidationPackConfig,
    EntityRegistryConfig,
    OwnershipConfig,
    BoundaryConfig,
    EquityShareConfig,
    ControlApproachConfig,
    EliminationConfig,
    MnAConfig,
    AdjustmentConfig,
    GroupReportingConfig,
    AuditConfig,
    SecurityConfig,
    PerformanceConfig,
    IntegrationConfig,
    AlertConfig,
    MigrationConfig,
    EntityType,
    EntityLifecycle,
    ConsolidationApproach,
    ControlType,
    OwnershipType,
    EliminationType,
    AdjustmentType,
    MnAEventType,
    ReportingFramework,
    DataQualityTier,
    CompletionStatus,
    ApprovalStatus,
    ReportType,
    ExportFormat,
    AlertType,
    ScopeCategory,
    MaterialityThreshold,
    ProRataMethod,
    AVAILABLE_PRESETS,
    DEFAULT_ENTITY_TYPES,
    DEFAULT_OWNERSHIP_THRESHOLDS,
    DEFAULT_ELIMINATION_RULES,
    DEFAULT_MNA_RULES,
    DEFAULT_FRAMEWORK_REQUIREMENTS,
    load_preset,
    validate_config,
    get_default_config,
    list_available_presets,
    get_entity_type_defaults,
    get_ownership_threshold,
    get_elimination_rules,
    get_mna_rules,
    get_framework_requirements,
    _compute_hash,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _make_hash(data: str) -> str:
    """Compute SHA-256 hash for test provenance comparisons."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Organisation / identity fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def organization_id():
    """Sample organisation identifier."""
    return "ORG-CONSOLIDATION-001"


@pytest.fixture
def reporting_year():
    """Standard test reporting year."""
    return 2026


@pytest.fixture
def base_year():
    """Standard test base year."""
    return 2020


@pytest.fixture
def parent_entity_id():
    """Fixed parent entity ID for consistent testing."""
    return "ENT-PARENT-001"


@pytest.fixture
def sub1_entity_id():
    """Fixed subsidiary 1 entity ID."""
    return "ENT-SUB-001"


@pytest.fixture
def sub2_entity_id():
    """Fixed subsidiary 2 entity ID."""
    return "ENT-SUB-002"


@pytest.fixture
def sub3_entity_id():
    """Fixed subsidiary 3 entity ID."""
    return "ENT-SUB-003"


@pytest.fixture
def jv_entity_id():
    """Fixed joint venture entity ID."""
    return "ENT-JV-001"


@pytest.fixture
def associate_entity_id():
    """Fixed associate entity ID."""
    return "ENT-ASSOC-001"


# ---------------------------------------------------------------------------
# Pack config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Complete PackConfig dictionary for testing."""
    return {
        "company_name": "GreenTest Holdings AG",
        "consolidation_approach": "OPERATIONAL_CONTROL",
        "reporting_year": 2026,
        "base_year": 2020,
        "country": "DE",
        "currency": "EUR",
        "total_entities": 6,
        "scopes_in_scope": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET"],
        "entity_registry": {
            "max_entities": 500,
            "entity_types_enabled": [
                "SUBSIDIARY", "JOINT_VENTURE", "ASSOCIATE", "DIVISION", "BRANCH",
            ],
            "lifecycle_tracking": True,
            "hierarchy_depth_limit": 10,
        },
        "ownership": {
            "multi_tier_enabled": True,
            "max_chain_depth": 10,
            "effective_equity_method": "MULTIPLICATIVE",
            "circular_ownership_detection": True,
        },
        "boundary": {
            "consolidation_approach": "OPERATIONAL_CONTROL",
            "materiality_threshold": "FIVE_PCT",
            "materiality_threshold_pct": Decimal("0.05"),
            "de_minimis_threshold_pct": Decimal("0.01"),
            "annual_boundary_lock": True,
        },
        "elimination": {
            "elimination_enabled": True,
            "require_matching_entries": True,
            "tolerance_pct": Decimal("5.0"),
        },
        "mna": {
            "mna_tracking_enabled": True,
            "pro_rata_method": "CALENDAR_DAYS",
            "auto_base_year_recalculation": True,
        },
        "adjustment": {
            "require_justification": True,
            "require_approval": True,
        },
        "reporting": {
            "reporting_frameworks": ["CSRD_ESRS_E1", "CDP", "GRI_305"],
            "default_format": "HTML",
        },
        "audit": {
            "audit_trail_enabled": True,
            "require_sign_off": True,
            "sign_off_levels": 2,
        },
    }


@pytest.fixture
def default_pack_config():
    """Default PackConfig with all defaults."""
    return PackConfig()


@pytest.fixture
def default_consolidation_config():
    """Default ConsolidationPackConfig with all defaults."""
    return ConsolidationPackConfig()


# ---------------------------------------------------------------------------
# Entity record fixtures (parent + 3 subs + 1 JV + 1 associate)
# ---------------------------------------------------------------------------

@pytest.fixture
def parent_entity_data(parent_entity_id):
    """Parent holding company entity data dict."""
    return {
        "entity_id": parent_entity_id,
        "legal_name": "GreenTest Holdings AG",
        "entity_type": "PARENT",
        "status": "ACTIVE",
        "jurisdiction": "Zurich, Switzerland",
        "country": "CH",
        "sector_code": "6420",
        "sector_name": "Activities of holding companies",
        "tags": ["parent", "holding"],
    }


@pytest.fixture
def sub1_entity_data(sub1_entity_id, parent_entity_id):
    """Subsidiary 1: wholly owned manufacturing subsidiary."""
    return {
        "entity_id": sub1_entity_id,
        "legal_name": "GreenTest Manufacturing GmbH",
        "entity_type": "SUBSIDIARY",
        "status": "ACTIVE",
        "jurisdiction": "Munich, Germany",
        "country": "DE",
        "parent_entity_id": parent_entity_id,
        "sector_code": "2511",
        "sector_name": "Manufacture of metal structures",
        "tags": ["manufacturing", "europe"],
    }


@pytest.fixture
def sub2_entity_data(sub2_entity_id, parent_entity_id):
    """Subsidiary 2: 80% owned logistics subsidiary."""
    return {
        "entity_id": sub2_entity_id,
        "legal_name": "GreenTest Logistics Ltd",
        "entity_type": "SUBSIDIARY",
        "status": "ACTIVE",
        "jurisdiction": "London, England and Wales",
        "country": "GB",
        "parent_entity_id": parent_entity_id,
        "sector_code": "5229",
        "sector_name": "Other transportation support activities",
        "tags": ["logistics", "europe"],
    }


@pytest.fixture
def sub3_entity_data(sub3_entity_id, parent_entity_id):
    """Subsidiary 3: 60% owned US subsidiary."""
    return {
        "entity_id": sub3_entity_id,
        "legal_name": "GreenTest Americas Inc",
        "entity_type": "SUBSIDIARY",
        "status": "ACTIVE",
        "jurisdiction": "Delaware, US",
        "country": "US",
        "parent_entity_id": parent_entity_id,
        "sector_code": "2511",
        "sector_name": "Manufacture of metal structures",
        "tags": ["manufacturing", "americas"],
    }


@pytest.fixture
def jv_entity_data(jv_entity_id, parent_entity_id):
    """Joint venture: 50/50 with external partner."""
    return {
        "entity_id": jv_entity_id,
        "legal_name": "GreenTest-Partner JV BV",
        "entity_type": "JOINT_VENTURE",
        "status": "ACTIVE",
        "jurisdiction": "Amsterdam, Netherlands",
        "country": "NL",
        "parent_entity_id": parent_entity_id,
        "sector_code": "3511",
        "sector_name": "Electric power generation",
        "tags": ["joint_venture", "energy"],
    }


@pytest.fixture
def associate_entity_data(associate_entity_id, parent_entity_id):
    """Associate: 30% equity stake with significant influence."""
    return {
        "entity_id": associate_entity_id,
        "legal_name": "GreenTech Innovation AS",
        "entity_type": "ASSOCIATE",
        "status": "ACTIVE",
        "jurisdiction": "Oslo, Norway",
        "country": "NO",
        "parent_entity_id": parent_entity_id,
        "sector_code": "7211",
        "sector_name": "Research and experimental development on biotechnology",
        "tags": ["associate", "r&d"],
    }


@pytest.fixture
def all_entity_data(
    parent_entity_data,
    sub1_entity_data,
    sub2_entity_data,
    sub3_entity_data,
    jv_entity_data,
    associate_entity_data,
):
    """All 6 entity data dicts in registration order."""
    return [
        parent_entity_data,
        sub1_entity_data,
        sub2_entity_data,
        sub3_entity_data,
        jv_entity_data,
        associate_entity_data,
    ]


# ---------------------------------------------------------------------------
# Ownership record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ownership_records(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    sub3_entity_id,
    jv_entity_id,
    associate_entity_id,
):
    """Ownership links: parent owns 100/80/60/50/30 of each entity."""
    return [
        {
            "owner_entity_id": parent_entity_id,
            "target_entity_id": sub1_entity_id,
            "ownership_pct": Decimal("100"),
            "has_operational_control": True,
            "has_financial_control": True,
            "manages_operations": True,
            "directs_policies": True,
            "has_board_majority": True,
        },
        {
            "owner_entity_id": parent_entity_id,
            "target_entity_id": sub2_entity_id,
            "ownership_pct": Decimal("80"),
            "has_operational_control": True,
            "has_financial_control": True,
            "manages_operations": True,
            "directs_policies": True,
            "has_board_majority": True,
        },
        {
            "owner_entity_id": parent_entity_id,
            "target_entity_id": sub3_entity_id,
            "ownership_pct": Decimal("60"),
            "has_operational_control": True,
            "has_financial_control": True,
            "manages_operations": True,
            "directs_policies": True,
            "has_board_majority": True,
        },
        {
            "owner_entity_id": parent_entity_id,
            "target_entity_id": jv_entity_id,
            "ownership_pct": Decimal("50"),
            "has_operational_control": False,
            "has_financial_control": False,
            "manages_operations": False,
            "directs_policies": False,
            "has_board_majority": False,
        },
        {
            "owner_entity_id": parent_entity_id,
            "target_entity_id": associate_entity_id,
            "ownership_pct": Decimal("30"),
            "has_operational_control": False,
            "has_financial_control": False,
            "manages_operations": False,
            "directs_policies": False,
            "has_board_majority": False,
        },
    ]


# ---------------------------------------------------------------------------
# Entity emissions data fixtures (Scope 1, 2-loc, 2-mkt, 3)
# ---------------------------------------------------------------------------

@pytest.fixture
def entity_emissions_data(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    sub3_entity_id,
    jv_entity_id,
    associate_entity_id,
):
    """Emissions data for each entity by scope (tCO2e)."""
    return {
        parent_entity_id: {
            "scope_1": Decimal("500.00"),
            "scope_2_location": Decimal("800.00"),
            "scope_2_market": Decimal("600.00"),
            "scope_3": Decimal("2000.00"),
            "total_location": Decimal("3300.00"),
            "total_market": Decimal("3100.00"),
        },
        sub1_entity_id: {
            "scope_1": Decimal("15000.00"),
            "scope_2_location": Decimal("8000.00"),
            "scope_2_market": Decimal("6000.00"),
            "scope_3": Decimal("25000.00"),
            "total_location": Decimal("48000.00"),
            "total_market": Decimal("46000.00"),
        },
        sub2_entity_id: {
            "scope_1": Decimal("3000.00"),
            "scope_2_location": Decimal("2000.00"),
            "scope_2_market": Decimal("1800.00"),
            "scope_3": Decimal("10000.00"),
            "total_location": Decimal("15000.00"),
            "total_market": Decimal("14800.00"),
        },
        sub3_entity_id: {
            "scope_1": Decimal("8000.00"),
            "scope_2_location": Decimal("5000.00"),
            "scope_2_market": Decimal("4500.00"),
            "scope_3": Decimal("18000.00"),
            "total_location": Decimal("31000.00"),
            "total_market": Decimal("30500.00"),
        },
        jv_entity_id: {
            "scope_1": Decimal("6000.00"),
            "scope_2_location": Decimal("4000.00"),
            "scope_2_market": Decimal("3500.00"),
            "scope_3": Decimal("12000.00"),
            "total_location": Decimal("22000.00"),
            "total_market": Decimal("21500.00"),
        },
        associate_entity_id: {
            "scope_1": Decimal("1000.00"),
            "scope_2_location": Decimal("500.00"),
            "scope_2_market": Decimal("400.00"),
            "scope_3": Decimal("3000.00"),
            "total_location": Decimal("4500.00"),
            "total_market": Decimal("4400.00"),
        },
    }


@pytest.fixture
def entity_total_emissions(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    sub3_entity_id,
    jv_entity_id,
    associate_entity_id,
):
    """Flat total emissions per entity for elimination engine."""
    return {
        parent_entity_id: Decimal("3300.00"),
        sub1_entity_id: Decimal("48000.00"),
        sub2_entity_id: Decimal("15000.00"),
        sub3_entity_id: Decimal("31000.00"),
        jv_entity_id: Decimal("22000.00"),
        associate_entity_id: Decimal("4500.00"),
    }


# ---------------------------------------------------------------------------
# Transfer record fixtures (intra-group energy transfers)
# ---------------------------------------------------------------------------

@pytest.fixture
def transfer_records(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    jv_entity_id,
):
    """Intra-group energy transfers between entities."""
    return [
        {
            "reporting_year": 2026,
            "seller_entity_id": sub1_entity_id,
            "buyer_entity_id": sub2_entity_id,
            "transfer_type": "ELECTRICITY",
            "quantity": Decimal("50000"),
            "quantity_unit": "MWh",
            "seller_emissions_tco2e": Decimal("12500.00"),
            "buyer_emissions_tco2e": Decimal("12500.00"),
            "seller_scope": "SCOPE_1",
            "buyer_scope": "SCOPE_2_LOCATION",
            "intra_group_pct": Decimal("100"),
            "description": "Intra-group electricity from manufacturing CHP to logistics",
        },
        {
            "reporting_year": 2026,
            "seller_entity_id": sub1_entity_id,
            "buyer_entity_id": jv_entity_id,
            "transfer_type": "STEAM",
            "quantity": Decimal("10000"),
            "quantity_unit": "MWh_th",
            "seller_emissions_tco2e": Decimal("2000.00"),
            "buyer_emissions_tco2e": Decimal("2100.00"),
            "seller_scope": "SCOPE_1",
            "buyer_scope": "SCOPE_2_LOCATION",
            "intra_group_pct": Decimal("50"),
            "description": "Partial intra-group steam to JV (50% owned)",
        },
        {
            "reporting_year": 2026,
            "seller_entity_id": sub2_entity_id,
            "buyer_entity_id": parent_entity_id,
            "transfer_type": "WASTE",
            "quantity": Decimal("500"),
            "quantity_unit": "tonnes",
            "seller_emissions_tco2e": Decimal("250.00"),
            "buyer_emissions_tco2e": Decimal("250.00"),
            "seller_scope": "SCOPE_1",
            "buyer_scope": "SCOPE_3",
            "intra_group_pct": Decimal("100"),
            "description": "Intra-group waste transfer for processing",
        },
    ]


# ---------------------------------------------------------------------------
# M&A event fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mna_acquisition_event(parent_entity_id):
    """Acquisition event: parent acquires new subsidiary on July 1."""
    return {
        "event_id": "MNA-ACQ-001",
        "event_type": "ACQUISITION",
        "acquiring_entity_id": parent_entity_id,
        "target_entity_id": "ENT-NEWACQ-001",
        "target_entity_name": "AcquiredCo GmbH",
        "completion_date": date(2026, 7, 1),
        "reporting_year": 2026,
        "ownership_pct_acquired": Decimal("100"),
        "total_annual_emissions_tco2e": Decimal("5000.00"),
        "purchase_price_eur": Decimal("50000000.00"),
        "description": "Full acquisition of AcquiredCo GmbH",
    }


@pytest.fixture
def mna_divestiture_event(parent_entity_id, sub3_entity_id):
    """Divestiture event: parent divests sub3 on September 30."""
    return {
        "event_id": "MNA-DIV-001",
        "event_type": "DIVESTITURE",
        "divesting_entity_id": parent_entity_id,
        "target_entity_id": sub3_entity_id,
        "target_entity_name": "GreenTest Americas Inc",
        "completion_date": date(2026, 9, 30),
        "reporting_year": 2026,
        "ownership_pct_divested": Decimal("60"),
        "total_annual_emissions_tco2e": Decimal("31000.00"),
        "sale_price_eur": Decimal("80000000.00"),
        "description": "Full divestiture of US subsidiary",
    }


# ---------------------------------------------------------------------------
# Adjustment record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def adjustment_records():
    """Adjustment records for various types."""
    return [
        {
            "adjustment_id": "ADJ-001",
            "adjustment_type": "METHODOLOGY_CHANGE",
            "entity_id": "ENT-SUB-001",
            "scope": "SCOPE_1",
            "original_value_tco2e": Decimal("15000.00"),
            "adjusted_value_tco2e": Decimal("14800.00"),
            "difference_tco2e": Decimal("-200.00"),
            "justification": "Updated emission factor from DEFRA 2026",
            "status": "DRAFT",
            "submitted_by": "analyst@greentest.com",
        },
        {
            "adjustment_id": "ADJ-002",
            "adjustment_type": "ERROR_CORRECTION",
            "entity_id": "ENT-SUB-002",
            "scope": "SCOPE_2_LOCATION",
            "original_value_tco2e": Decimal("2000.00"),
            "adjusted_value_tco2e": Decimal("2150.00"),
            "difference_tco2e": Decimal("150.00"),
            "justification": "Corrected meter reading error in Q3",
            "status": "SUBMITTED",
            "submitted_by": "data-lead@greentest.com",
        },
        {
            "adjustment_id": "ADJ-003",
            "adjustment_type": "LATE_SUBMISSION",
            "entity_id": "ENT-JV-001",
            "scope": "SCOPE_3",
            "original_value_tco2e": Decimal("0.00"),
            "adjusted_value_tco2e": Decimal("12000.00"),
            "difference_tco2e": Decimal("12000.00"),
            "justification": "Late Scope 3 data received from JV partner",
            "status": "APPROVED",
            "submitted_by": "jv-manager@greentest.com",
        },
    ]


# ---------------------------------------------------------------------------
# Boundary definition fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equity_share_boundary(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    sub3_entity_id,
    jv_entity_id,
    associate_entity_id,
):
    """Boundary definition using equity share approach."""
    return {
        "boundary_id": "BND-ES-2026",
        "approach": "EQUITY_SHARE",
        "reporting_year": 2026,
        "entities": [
            {"entity_id": parent_entity_id, "equity_pct": Decimal("100"), "included": True},
            {"entity_id": sub1_entity_id, "equity_pct": Decimal("100"), "included": True},
            {"entity_id": sub2_entity_id, "equity_pct": Decimal("80"), "included": True},
            {"entity_id": sub3_entity_id, "equity_pct": Decimal("60"), "included": True},
            {"entity_id": jv_entity_id, "equity_pct": Decimal("50"), "included": True},
            {"entity_id": associate_entity_id, "equity_pct": Decimal("30"), "included": True},
        ],
        "materiality_threshold_pct": Decimal("5"),
    }


@pytest.fixture
def operational_control_boundary(
    parent_entity_id,
    sub1_entity_id,
    sub2_entity_id,
    sub3_entity_id,
    jv_entity_id,
    associate_entity_id,
):
    """Boundary definition using operational control approach."""
    return {
        "boundary_id": "BND-OC-2026",
        "approach": "OPERATIONAL_CONTROL",
        "reporting_year": 2026,
        "entities": [
            {"entity_id": parent_entity_id, "inclusion_pct": Decimal("100"), "included": True},
            {"entity_id": sub1_entity_id, "inclusion_pct": Decimal("100"), "included": True},
            {"entity_id": sub2_entity_id, "inclusion_pct": Decimal("100"), "included": True},
            {"entity_id": sub3_entity_id, "inclusion_pct": Decimal("100"), "included": True},
            {"entity_id": jv_entity_id, "inclusion_pct": Decimal("0"), "included": False},
            {"entity_id": associate_entity_id, "inclusion_pct": Decimal("0"), "included": False},
        ],
        "materiality_threshold_pct": Decimal("5"),
    }


# ---------------------------------------------------------------------------
# Multi-tier ownership for chain resolution tests
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_tier_ownership():
    """3-tier ownership: A owns 80% of B, B owns 75% of C.
    Effective A->C = 80% * 75% = 60%."""
    return {
        "entities": ["ENT-A", "ENT-B", "ENT-C"],
        "links": [
            {
                "owner_entity_id": "ENT-A",
                "target_entity_id": "ENT-B",
                "ownership_pct": Decimal("80"),
                "has_operational_control": True,
                "has_financial_control": True,
                "manages_operations": True,
                "directs_policies": True,
            },
            {
                "owner_entity_id": "ENT-B",
                "target_entity_id": "ENT-C",
                "ownership_pct": Decimal("75"),
                "has_operational_control": True,
                "has_financial_control": True,
                "manages_operations": True,
                "directs_policies": True,
            },
        ],
        "expected_effective_a_to_c": Decimal("60.0000"),
    }


# ---------------------------------------------------------------------------
# Report data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def consolidated_report_data():
    """Input data for consolidated GHG report template."""
    return {
        "company_name": "GreenTest Holdings AG",
        "reporting_period": "FY 2026",
        "consolidation_approach": "operational_control",
        "scope_1_tco2e": Decimal("26500.00"),
        "scope_2_location_tco2e": Decimal("15800.00"),
        "scope_2_market_tco2e": Decimal("12900.00"),
        "scope_3_tco2e": Decimal("55000.00"),
        "total_location_tco2e": Decimal("97300.00"),
        "total_market_tco2e": Decimal("94400.00"),
        "prior_year_location_tco2e": Decimal("100000.00"),
        "prior_year_market_tco2e": Decimal("98000.00"),
        "total_entities": 6,
        "included_entities": 4,
        "excluded_entities": 2,
        "entities": [
            {
                "entity_id": "ENT-PARENT-001",
                "entity_name": "GreenTest Holdings AG",
                "entity_type": "parent",
                "country_code": "CH",
                "control_type": "operational",
                "ownership_pct": Decimal("100"),
                "reporting_pct": Decimal("100"),
                "included": True,
            },
            {
                "entity_id": "ENT-SUB-001",
                "entity_name": "GreenTest Manufacturing GmbH",
                "entity_type": "subsidiary",
                "country_code": "DE",
                "control_type": "operational",
                "ownership_pct": Decimal("100"),
                "reporting_pct": Decimal("100"),
                "included": True,
            },
        ],
        "adjustment_waterfall": [
            {
                "label": "Raw Entity Totals",
                "scope_1_tco2e": Decimal("33500"),
                "scope_2_location_tco2e": Decimal("20300"),
                "scope_2_market_tco2e": Decimal("16800"),
                "scope_3_tco2e": Decimal("70000"),
                "total_location_tco2e": Decimal("123800"),
                "total_market_tco2e": Decimal("120300"),
            },
            {
                "label": "After Boundary Exclusions",
                "scope_1_tco2e": Decimal("26500"),
                "scope_2_location_tco2e": Decimal("15800"),
                "scope_2_market_tco2e": Decimal("12900"),
                "scope_3_tco2e": Decimal("55000"),
                "total_location_tco2e": Decimal("97300"),
                "total_market_tco2e": Decimal("94400"),
            },
        ],
        "boundary": {
            "approach": "operational_control",
            "rationale": "GHG Protocol Corporate Standard Chapter 3 - operational control approach selected as reporting entity operates all included facilities.",
            "base_year": 2020,
            "boundary_last_reviewed": "2026-01-15",
            "exclusions": ["JV excluded (no operational control)", "Associate excluded (no operational control)"],
            "materiality_threshold_pct": Decimal("5"),
        },
        "notes": [
            {
                "note_id": "NOTE-001",
                "category": "Methodology",
                "text": "Scope 2 dual reporting per GHG Protocol Scope 2 Guidance.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Workflow input fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def entity_mapping_workflow_input():
    """Input data for entity mapping workflow."""
    return {
        "organisation_id": "ORG-CONSOLIDATION-001",
        "organisation_name": "GreenTest Holdings AG",
        "reporting_year": 2026,
        "entity_data": [
            {
                "entity_name": "GreenTest Holdings AG",
                "entity_type": "parent",
                "jurisdiction": "CH",
                "direct_ownership_pct": "100",
                "source": "erp",
                "is_active": True,
            },
            {
                "entity_name": "GreenTest Manufacturing GmbH",
                "entity_type": "subsidiary",
                "jurisdiction": "DE",
                "direct_ownership_pct": "100",
                "source": "erp",
                "is_active": True,
            },
            {
                "entity_name": "GreenTest Logistics Ltd",
                "entity_type": "subsidiary",
                "jurisdiction": "GB",
                "direct_ownership_pct": "80",
                "source": "legal_registry",
                "is_active": True,
            },
            {
                "entity_name": "GreenTest-Partner JV BV",
                "entity_type": "joint_venture",
                "jurisdiction": "NL",
                "direct_ownership_pct": "50",
                "source": "manual",
                "is_active": True,
            },
        ],
        "ownership_links": [],
        "control_indicators": [],
        "emissions_estimates": [
            {"entity_id": "ent-1", "estimated_emissions_tco2e": "48000"},
            {"entity_id": "ent-2", "estimated_emissions_tco2e": "15000"},
            {"entity_id": "ent-3", "estimated_emissions_tco2e": "22000"},
            {"entity_id": "ent-4", "estimated_emissions_tco2e": "3300"},
        ],
        "materiality_threshold_pct": Decimal("1.0"),
    }
