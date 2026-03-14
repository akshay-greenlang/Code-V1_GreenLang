# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Shared Test Fixtures
=========================================================

Provides reusable pytest fixtures for all PACK-002 test modules including
multi-entity group profiles, consolidated ESRS data, intercompany
transactions, approval chains, quality gates, cross-framework mappings,
scenario configurations, stakeholders, regulatory changes, benchmarks,
webhook configurations, and mocked agent registries.

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths & sys.path setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent

# Ensure the pack root's parent directories are importable despite hyphens.
# We register the engines directory so that ``import consolidation_engine``
# works when invoked via ``importlib`` from test modules.
_engines_dir = str(PACK_ROOT / "engines")
if _engines_dir not in sys.path:
    sys.path.insert(0, _engines_dir)

# Also register PACK_ROOT so that relative imports from pack code work.
_pack_root_str = str(PACK_ROOT)
if _pack_root_str not in sys.path:
    sys.path.insert(0, _pack_root_str)
PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
ENGINES_DIR = PACK_ROOT / "engines"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Pack YAML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML_PATH


@pytest.fixture(scope="session")
def pack_yaml_raw(pack_yaml_path) -> str:
    """Return the raw text content of pack.yaml."""
    return pack_yaml_path.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary."""
    return yaml.safe_load(pack_yaml_raw)


# ---------------------------------------------------------------------------
# Professional PackConfig fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pack_config() -> Dict[str, Any]:
    """Create a Professional PackConfig with enterprise_group preset.

    Mirrors the expected output of PackConfig.load() when run with
    the enterprise_group size preset and manufacturing_pro sector
    preset, suitable for comprehensive unit-test execution.
    """
    return {
        "metadata": {
            "name": "csrd-professional",
            "version": "1.0.0",
            "category": "eu-compliance",
            "tier": "professional",
            "display_name": "CSRD Professional Pack",
        },
        "size_preset": "enterprise_group",
        "sector_preset": "manufacturing_pro",
        "reporting_year": 2025,
        "reporting_period_start": "2025-01-01",
        "reporting_period_end": "2025-12-31",
        "esrs_standards": [
            "ESRS_1", "ESRS_2",
            "E1", "E2", "E3", "E4", "E5",
            "S1", "S2", "S3", "S4",
            "G1",
        ],
        "scope3_categories_enabled": list(range(1, 16)),
        "xbrl_enabled": True,
        "language": "en",
        "languages": ["en", "de", "fr", "es", "it", "nl", "pl"],
        "consolidation": {
            "enabled": True,
            "default_approach": "operational_control",
            "intercompany_elimination": True,
            "minority_disclosures": True,
            "max_subsidiaries": 100,
        },
        "approval_workflow": {
            "enabled": True,
            "levels": 4,
            "auto_approve_threshold": 95.0,
            "escalation_timeout_hours": 48,
        },
        "quality_gates": {
            "qg1_data_completeness": {"enabled": True, "threshold": 85.0},
            "qg2_calculation_integrity": {"enabled": True, "threshold": 90.0},
            "qg3_compliance_readiness": {"enabled": True, "threshold": 80.0},
        },
        "cross_framework": {
            "cdp": True,
            "tcfd": True,
            "sbti": True,
            "eu_taxonomy": True,
            "gri": True,
            "sasb": True,
        },
        "scenario_analysis": {
            "enabled": True,
            "scenarios": ["IEA_NZE", "IEA_APS", "NGFS_ORDERLY", "NGFS_DISORDERLY"],
            "time_horizons": ["2030", "2040", "2050"],
            "monte_carlo_iterations": 10000,
        },
        "assurance": {
            "level": "reasonable",
            "standard": "ISAE_3000",
        },
        "performance_targets": {
            "full_report_max_minutes": 45,
            "consolidated_report_max_seconds": 600,
            "cross_framework_max_seconds": 240,
            "scenario_analysis_max_minutes": 30,
            "quality_gate_max_seconds": 60,
        },
        "provenance_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }


# ---------------------------------------------------------------------------
# Group profile fixture - EuroTech Holdings AG
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_group_profile() -> Dict[str, Any]:
    """Create a multi-entity group profile for EuroTech Holdings AG.

    Parent entity in Germany with 5 subsidiaries across France, Italy,
    Spain, Netherlands, and Poland. Realistic employee counts, revenue
    figures, and ownership structures for integration testing.
    """
    return {
        "group_name": "EuroTech Holdings AG",
        "legal_entity_id": "DE-HRB-789012",
        "lei": "529900XYZABCDEFGH345",
        "parent": {
            "entity_id": "eurotech-parent",
            "name": "EuroTech Holdings AG",
            "country": "DE",
            "sector": "manufacturing",
            "nace_code": "C28.1",
            "employees": 8000,
            "revenue_eur": 2_100_000_000,
            "total_assets_eur": 3_800_000_000,
            "listed": True,
            "stock_exchange": "XETRA",
            "isin": "DE000ETAG0001",
            "ownership_pct": 100.0,
            "consolidation_method": "operational_control",
            "currency": "EUR",
        },
        "subsidiaries": [
            {
                "entity_id": "eurotech-fr",
                "name": "EuroTech France SAS",
                "country": "FR",
                "employees": 1200,
                "revenue_eur": 380_000_000,
                "ownership_pct": 100.0,
                "consolidation_method": "operational_control",
                "nace_code": "C28.1",
                "currency": "EUR",
                "is_eu_entity": True,
            },
            {
                "entity_id": "eurotech-it",
                "name": "EuroTech Italia S.r.l.",
                "country": "IT",
                "employees": 800,
                "revenue_eur": 245_000_000,
                "ownership_pct": 100.0,
                "consolidation_method": "operational_control",
                "nace_code": "C28.2",
                "currency": "EUR",
                "is_eu_entity": True,
            },
            {
                "entity_id": "eurotech-es",
                "name": "EuroTech Espana S.L.",
                "country": "ES",
                "employees": 600,
                "revenue_eur": 175_000_000,
                "ownership_pct": 80.0,
                "consolidation_method": "financial_control",
                "nace_code": "C28.1",
                "currency": "EUR",
                "is_eu_entity": True,
            },
            {
                "entity_id": "eurotech-nl",
                "name": "EuroTech Nederland B.V.",
                "country": "NL",
                "employees": 400,
                "revenue_eur": 120_000_000,
                "ownership_pct": 100.0,
                "consolidation_method": "operational_control",
                "nace_code": "C26.5",
                "currency": "EUR",
                "is_eu_entity": True,
            },
            {
                "entity_id": "eurotech-pl",
                "name": "EuroTech Polska Sp. z o.o.",
                "country": "PL",
                "employees": 300,
                "revenue_eur": 65_000_000,
                "ownership_pct": 100.0,
                "consolidation_method": "operational_control",
                "nace_code": "C28.1",
                "currency": "PLN",
                "is_eu_entity": True,
            },
        ],
        "total_employees": 11300,
        "total_revenue_eur": 3_085_000_000,
        "reporting_year": 2025,
        "fiscal_year_end": "2025-12-31",
        "auditor": "KPMG AG",
        "consolidation_standard": "IFRS",
    }


# ---------------------------------------------------------------------------
# Entity ESRS data fixture
# ---------------------------------------------------------------------------

def _build_entity_esrs_data(
    entity_id: str,
    scope1_total: float,
    scope2_total: float,
    scope3_total: float,
    employees: int,
    revenue_eur: float,
    quality_score: float,
) -> Dict[str, Any]:
    """Helper to build realistic ESRS data for a single entity."""
    return {
        "entity_id": entity_id,
        "reporting_period": "2025-01-01/2025-12-31",
        "quality_score": quality_score,
        "data_points": {
            "E1-6_01_scope1_total_tco2e": scope1_total,
            "E1-6_04_scope2_location_tco2e": scope2_total * 1.15,
            "E1-6_04_scope2_market_tco2e": scope2_total,
            "E1-6_06_scope3_total_tco2e": scope3_total,
            "E1-6_06_scope3_cat1_tco2e": scope3_total * 0.42,
            "E1-6_06_scope3_cat4_tco2e": scope3_total * 0.18,
            "E1-6_06_scope3_cat6_tco2e": scope3_total * 0.05,
            "E1-6_06_scope3_cat7_tco2e": scope3_total * 0.08,
            "E1_total_energy_mwh": (scope1_total + scope2_total) * 3.2,
            "E1_renewable_energy_pct": 35.0 + (hash(entity_id) % 30),
            "S1_total_employees": employees,
            "S1_female_pct": 32.0 + (hash(entity_id) % 20),
            "S1_turnover_rate_pct": 8.0 + (hash(entity_id) % 10),
            "S1_training_hours_per_emp": 18.0 + (hash(entity_id) % 15),
            "S1_gender_pay_gap_pct": 5.0 + (hash(entity_id) % 8),
            "S1_ltir": 1.2 + (hash(entity_id) % 5) * 0.3,
            "G1_board_independence_pct": 55.0 + (hash(entity_id) % 25),
            "G1_anti_corruption_training_pct": 88.0 + (hash(entity_id) % 12),
            "revenue_eur": revenue_eur,
            "total_waste_tonnes": 250 + (hash(entity_id) % 500),
            "water_withdrawal_m3": 85000 + (hash(entity_id) % 150000),
        },
    }


@pytest.fixture
def sample_entity_data() -> Dict[str, Dict[str, Any]]:
    """Create a dict of EntityESRSData per subsidiary with realistic data.

    Each entity has unique but plausible environmental, social, and
    governance metrics. The parent entity (DE) has the largest footprint.
    """
    entities = {
        "eurotech-parent": _build_entity_esrs_data(
            "eurotech-parent", 12500.0, 8200.0, 45000.0, 8000, 2_100_000_000, 92.5,
        ),
        "eurotech-fr": _build_entity_esrs_data(
            "eurotech-fr", 4800.0, 3100.0, 18500.0, 1200, 380_000_000, 89.0,
        ),
        "eurotech-it": _build_entity_esrs_data(
            "eurotech-it", 3200.0, 2400.0, 12000.0, 800, 245_000_000, 87.5,
        ),
        "eurotech-es": _build_entity_esrs_data(
            "eurotech-es", 2100.0, 1800.0, 8500.0, 600, 175_000_000, 85.0,
        ),
        "eurotech-nl": _build_entity_esrs_data(
            "eurotech-nl", 1500.0, 1200.0, 5800.0, 400, 120_000_000, 91.0,
        ),
        "eurotech-pl": _build_entity_esrs_data(
            "eurotech-pl", 950.0, 1600.0, 3200.0, 300, 65_000_000, 83.5,
        ),
    }
    return entities


# ---------------------------------------------------------------------------
# Consolidated ESRS dataset - 100 records across all standards
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_esrs_data() -> List[Dict[str, Any]]:
    """Consolidated ESRS dataset of 100 records across all standards.

    Covers E1-E5, S1-S4, G1 with a realistic distribution of data
    points from all 6 entities in the EuroTech Holdings group.
    """
    records: List[Dict[str, Any]] = []
    entities = [
        "eurotech-parent", "eurotech-fr", "eurotech-it",
        "eurotech-es", "eurotech-nl", "eurotech-pl",
    ]
    standards_mix = [
        ("E1", "E1-6_01", "scope1_stationary", "natural_gas", "m3"),
        ("E1", "E1-6_01", "scope1_stationary", "diesel", "litres"),
        ("E1", "E1-6_02", "scope1_mobile", "diesel_fleet", "litres"),
        ("E1", "E1-6_03", "scope1_refrigerants", "R-410A", "kg"),
        ("E1", "E1-6_04", "scope2_electricity", "grid_mix", "kWh"),
        ("E1", "E1-6_06", "scope3_cat1", "steel", "tonnes"),
        ("E1", "E1-6_06", "scope3_cat4", "road_freight", "tkm"),
        ("E1", "E1-6_08", "scope3_cat6", "air_travel", "pkm"),
        ("E2", "E2-4_01", "air_pollutants", "nox_sox", "tonnes"),
        ("E3", "E3-4_01", "water_withdrawal", "freshwater", "m3"),
        ("E3", "E3-4_02", "water_recycled", "process_water", "m3"),
        ("E4", "E4-5_01", "biodiversity", "site_assessment", "count"),
        ("E5", "E5-5_01", "waste_recycling", "total_waste", "tonnes"),
        ("S1", "S1-6_01", "workforce", "headcount", "count"),
        ("S1", "S1-6_02", "workforce_diversity", "female_pct", "percent"),
        ("S1", "S1-8_01", "health_safety", "injuries", "count"),
        ("G1", "G1-1_01", "governance", "board_independence", "percent"),
    ]

    values_base = [
        850000, 42000, 15000, 8.5, 6500000, 2200, 1500000, 95000,
        4.2, 125000, 52000, 1, 380, 1850, 38.5, 3, 66.7,
    ]

    record_id = 0
    for ent_idx, entity_id in enumerate(entities):
        for dp_idx, (std, dp, cat, sub, unit) in enumerate(standards_mix):
            record_id += 1
            base_val = values_base[dp_idx % len(values_base)]
            scale = [1.0, 0.45, 0.35, 0.25, 0.18, 0.12][ent_idx]
            value = round(base_val * scale * (1.0 + (record_id % 7) * 0.02), 2)
            records.append({
                "id": f"PRO-{record_id:04d}",
                "entity_id": entity_id,
                "esrs_standard": std,
                "data_point": dp,
                "category": cat,
                "sub_category": sub,
                "value": value,
                "unit": unit,
                "source": "erp_sap" if ent_idx < 3 else "excel_manual",
                "quality_score": 0.80 + (record_id % 20) * 0.01,
                "period": "2025",
            })

    # Trim or pad to exactly 100 records
    if len(records) > 100:
        records = records[:100]
    while len(records) < 100:
        record_id += 1
        records.append({
            "id": f"PRO-{record_id:04d}",
            "entity_id": entities[record_id % len(entities)],
            "esrs_standard": "E1",
            "data_point": "E1-9_01",
            "category": "ghg_intensity",
            "sub_category": "tco2e_per_meur",
            "value": round(85.0 + record_id * 0.5, 2),
            "unit": "tCO2e/MEUR",
            "source": "calculated",
            "quality_score": 0.95,
            "period": "2025",
        })

    assert len(records) == 100
    return records


# ---------------------------------------------------------------------------
# Intercompany transactions fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_intercompany_transactions() -> List[Dict[str, Any]]:
    """Create 10 intercompany transactions for elimination testing.

    Includes revenue/cost transfers, Scope 3 emission transfers, and
    waste transfers between group entities.
    """
    return [
        {
            "transaction_id": "ICT-001",
            "from_entity": "eurotech-parent",
            "to_entity": "eurotech-fr",
            "transaction_type": "revenue",
            "amount": 45_000_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-002",
            "from_entity": "eurotech-fr",
            "to_entity": "eurotech-parent",
            "transaction_type": "cost",
            "amount": 38_000_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-003",
            "from_entity": "eurotech-parent",
            "to_entity": "eurotech-it",
            "transaction_type": "revenue",
            "amount": 22_000_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-004",
            "from_entity": "eurotech-it",
            "to_entity": "eurotech-es",
            "transaction_type": "emission_transfer",
            "amount": 1250.0,
            "scope3_category": 1,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-005",
            "from_entity": "eurotech-parent",
            "to_entity": "eurotech-nl",
            "transaction_type": "emission_transfer",
            "amount": 850.0,
            "scope3_category": 4,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-006",
            "from_entity": "eurotech-nl",
            "to_entity": "eurotech-pl",
            "transaction_type": "revenue",
            "amount": 8_500_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-007",
            "from_entity": "eurotech-fr",
            "to_entity": "eurotech-es",
            "transaction_type": "waste_transfer",
            "amount": 120.0,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-008",
            "from_entity": "eurotech-parent",
            "to_entity": "eurotech-pl",
            "transaction_type": "cost",
            "amount": 5_200_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
        {
            "transaction_id": "ICT-009",
            "from_entity": "eurotech-es",
            "to_entity": "eurotech-parent",
            "transaction_type": "emission_transfer",
            "amount": 420.0,
            "scope3_category": 1,
            "elimination_method": "partial",
        },
        {
            "transaction_id": "ICT-010",
            "from_entity": "eurotech-it",
            "to_entity": "eurotech-fr",
            "transaction_type": "revenue",
            "amount": 12_000_000,
            "scope3_category": None,
            "elimination_method": "full",
        },
    ]


# ---------------------------------------------------------------------------
# Approval chain fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_approval_chain() -> Dict[str, Any]:
    """Create a 4-level approval chain configuration.

    Levels: Preparer -> Reviewer -> Approver -> Board Sign-off
    Each level has specific approvers, timeout, and escalation rules.
    """
    return {
        "chain_id": "eurotech-approval-2025",
        "report_type": "annual_consolidated",
        "levels": [
            {
                "level": 1,
                "name": "Preparer",
                "role": "sustainability_analyst",
                "approvers": [
                    {"user_id": "anna.schmidt", "name": "Anna Schmidt", "email": "a.schmidt@eurotech.de"},
                    {"user_id": "pierre.dupont", "name": "Pierre Dupont", "email": "p.dupont@eurotech.fr"},
                ],
                "required_approvals": 1,
                "timeout_hours": 72,
                "escalation_to": "level_2",
                "auto_approve_quality_threshold": None,
            },
            {
                "level": 2,
                "name": "Reviewer",
                "role": "sustainability_manager",
                "approvers": [
                    {"user_id": "maria.weber", "name": "Maria Weber", "email": "m.weber@eurotech.de"},
                ],
                "required_approvals": 1,
                "timeout_hours": 48,
                "escalation_to": "level_3",
                "auto_approve_quality_threshold": 95.0,
            },
            {
                "level": 3,
                "name": "Approver",
                "role": "cso",
                "approvers": [
                    {"user_id": "thomas.mueller", "name": "Thomas Mueller", "email": "t.mueller@eurotech.de"},
                ],
                "required_approvals": 1,
                "timeout_hours": 48,
                "escalation_to": "level_4",
                "auto_approve_quality_threshold": None,
            },
            {
                "level": 4,
                "name": "Board Sign-off",
                "role": "board_member",
                "approvers": [
                    {"user_id": "klaus.fischer", "name": "Klaus Fischer", "email": "k.fischer@eurotech.de"},
                    {"user_id": "helga.braun", "name": "Helga Braun", "email": "h.braun@eurotech.de"},
                ],
                "required_approvals": 2,
                "timeout_hours": 120,
                "escalation_to": None,
                "auto_approve_quality_threshold": None,
            },
        ],
        "delegation_rules": {
            "enabled": True,
            "max_delegation_depth": 1,
            "require_same_role": True,
        },
        "audit_trail": True,
        "notification_channels": ["email", "slack"],
    }


# ---------------------------------------------------------------------------
# Quality gate data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_quality_gate_data() -> Dict[str, Any]:
    """Data for all 3 quality gates with realistic scores.

    QG1: Data Completeness (85% threshold)
    QG2: Calculation Integrity (90% threshold)
    QG3: Compliance Readiness (80% threshold)
    """
    return {
        "report_id": "eurotech-annual-2025",
        "entity_count": 6,
        "qg1_data_completeness": {
            "gate_id": "QG1",
            "name": "Data Completeness",
            "threshold": 85.0,
            "checks": [
                {"check_id": "QG1-01", "name": "ESRS data point coverage", "weight": 0.30, "score": 92.5},
                {"check_id": "QG1-02", "name": "Scope 1 data completeness", "weight": 0.15, "score": 95.0},
                {"check_id": "QG1-03", "name": "Scope 2 data completeness", "weight": 0.15, "score": 93.0},
                {"check_id": "QG1-04", "name": "Scope 3 data completeness", "weight": 0.10, "score": 78.0},
                {"check_id": "QG1-05", "name": "Social data completeness", "weight": 0.10, "score": 88.0},
                {"check_id": "QG1-06", "name": "Governance data completeness", "weight": 0.05, "score": 96.0},
                {"check_id": "QG1-07", "name": "Entity coverage (6/6)", "weight": 0.10, "score": 100.0},
                {"check_id": "QG1-08", "name": "Source documentation", "weight": 0.05, "score": 85.0},
            ],
        },
        "qg2_calculation_integrity": {
            "gate_id": "QG2",
            "name": "Calculation Integrity",
            "threshold": 90.0,
            "checks": [
                {"check_id": "QG2-01", "name": "Emission factor validity", "weight": 0.20, "score": 98.0},
                {"check_id": "QG2-02", "name": "Scope 1+2 dual reporting variance", "weight": 0.20, "score": 95.0},
                {"check_id": "QG2-03", "name": "Cross-entity balance (sum = group)", "weight": 0.20, "score": 100.0},
                {"check_id": "QG2-04", "name": "Unit consistency", "weight": 0.15, "score": 97.0},
                {"check_id": "QG2-05", "name": "Year-over-year variance check", "weight": 0.15, "score": 88.0},
                {"check_id": "QG2-06", "name": "Intercompany elimination accuracy", "weight": 0.10, "score": 100.0},
            ],
        },
        "qg3_compliance_readiness": {
            "gate_id": "QG3",
            "name": "Compliance Readiness",
            "threshold": 80.0,
            "checks": [
                {"check_id": "QG3-01", "name": "ESRS mandatory disclosure coverage", "weight": 0.25, "score": 91.0},
                {"check_id": "QG3-02", "name": "Validation rules pass rate", "weight": 0.20, "score": 94.0},
                {"check_id": "QG3-03", "name": "XBRL tag validity", "weight": 0.15, "score": 88.0},
                {"check_id": "QG3-04", "name": "Cross-framework consistency", "weight": 0.15, "score": 85.0},
                {"check_id": "QG3-05", "name": "Materiality alignment", "weight": 0.10, "score": 92.0},
                {"check_id": "QG3-06", "name": "Narrative quality score", "weight": 0.10, "score": 80.0},
                {"check_id": "QG3-07", "name": "Audit trail completeness", "weight": 0.05, "score": 95.0},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Cross-framework data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cross_framework_data() -> Dict[str, Any]:
    """ESRS data mapped to CDP, TCFD, SBTi, EU Taxonomy, GRI, and SASB.

    Contains mapping tables and coverage percentages for each framework.
    """
    return {
        "source_framework": "ESRS",
        "source_data_points": 185,
        "mappings": {
            "cdp": {
                "framework_name": "CDP Climate Change 2025",
                "total_questions": 142,
                "mapped_questions": 118,
                "coverage_pct": 83.1,
                "predicted_score": "A-",
                "gaps": [
                    "C2.3a - Physical risk details",
                    "C6.10 - Intensity metrics by product",
                    "C12.1c - Policy engagement details",
                ],
                "mapping_confidence": 0.87,
            },
            "tcfd": {
                "framework_name": "TCFD Recommendations",
                "total_disclosures": 11,
                "mapped_disclosures": 10,
                "coverage_pct": 90.9,
                "gaps": ["Strategy-c: Climate resilience details"],
                "pillars": {
                    "governance": {"coverage_pct": 100.0, "disclosures_mapped": 2},
                    "strategy": {"coverage_pct": 66.7, "disclosures_mapped": 2},
                    "risk_management": {"coverage_pct": 100.0, "disclosures_mapped": 3},
                    "metrics_targets": {"coverage_pct": 100.0, "disclosures_mapped": 3},
                },
            },
            "sbti": {
                "framework_name": "SBTi Corporate Net-Zero Standard v1.1",
                "near_term_target_set": True,
                "net_zero_target_set": True,
                "base_year": 2020,
                "base_year_emissions_tco2e": 95000.0,
                "current_year_emissions_tco2e": 82500.0,
                "reduction_pct": 13.2,
                "required_annual_reduction_pct": 4.2,
                "pathway": "1.5C_SDA_Manufacturing",
                "on_track": True,
                "temperature_score": 1.8,
                "coverage_pct": 92.0,
            },
            "eu_taxonomy": {
                "framework_name": "EU Taxonomy Regulation (2020/852)",
                "eligible_turnover_pct": 45.2,
                "aligned_turnover_pct": 32.8,
                "eligible_capex_pct": 52.0,
                "aligned_capex_pct": 38.5,
                "eligible_opex_pct": 28.0,
                "aligned_opex_pct": 18.2,
                "environmental_objectives": {
                    "climate_mitigation": {"eligible": True, "aligned": True},
                    "climate_adaptation": {"eligible": True, "aligned": False},
                    "water": {"eligible": False, "aligned": False},
                    "circular_economy": {"eligible": True, "aligned": True},
                    "pollution": {"eligible": False, "aligned": False},
                    "biodiversity": {"eligible": False, "aligned": False},
                },
                "coverage_pct": 88.0,
            },
            "gri": {
                "framework_name": "GRI Standards 2021",
                "total_disclosures": 120,
                "mapped_disclosures": 98,
                "coverage_pct": 81.7,
                "gaps_count": 22,
            },
            "sasb": {
                "framework_name": "SASB Industrial Machinery & Goods",
                "total_metrics": 18,
                "mapped_metrics": 14,
                "coverage_pct": 77.8,
                "gaps_count": 4,
            },
        },
        "overall_coverage_pct": 85.4,
        "provenance_hash": "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5",
    }


# ---------------------------------------------------------------------------
# Scenario analysis config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_scenario_config() -> Dict[str, Any]:
    """Configuration for 4 climate scenarios with parameters.

    IEA NZE 2050, IEA APS, NGFS Orderly, NGFS Disorderly
    """
    return {
        "analysis_id": "eurotech-scenario-2025",
        "base_year": 2025,
        "time_horizons": [2030, 2040, 2050],
        "scenarios": [
            {
                "id": "IEA_NZE",
                "name": "IEA Net Zero Emissions by 2050",
                "source": "IEA WEO 2024",
                "warming_target_c": 1.5,
                "carbon_price_2030_eur": 130,
                "carbon_price_2050_eur": 250,
                "renewable_share_2030_pct": 60.0,
                "renewable_share_2050_pct": 90.0,
                "gdp_impact_2050_pct": -1.5,
                "category": "transition",
            },
            {
                "id": "IEA_APS",
                "name": "IEA Announced Pledges Scenario",
                "source": "IEA WEO 2024",
                "warming_target_c": 1.7,
                "carbon_price_2030_eur": 75,
                "carbon_price_2050_eur": 140,
                "renewable_share_2030_pct": 45.0,
                "renewable_share_2050_pct": 70.0,
                "gdp_impact_2050_pct": -2.5,
                "category": "transition",
            },
            {
                "id": "NGFS_ORDERLY",
                "name": "NGFS Net Zero 2050 (Orderly)",
                "source": "NGFS Phase IV",
                "warming_target_c": 1.5,
                "carbon_price_2030_eur": 120,
                "carbon_price_2050_eur": 280,
                "renewable_share_2030_pct": 55.0,
                "renewable_share_2050_pct": 85.0,
                "gdp_impact_2050_pct": -1.8,
                "category": "transition",
            },
            {
                "id": "NGFS_DISORDERLY",
                "name": "NGFS Delayed Transition (Disorderly)",
                "source": "NGFS Phase IV",
                "warming_target_c": 2.0,
                "carbon_price_2030_eur": 25,
                "carbon_price_2050_eur": 350,
                "renewable_share_2030_pct": 30.0,
                "renewable_share_2050_pct": 65.0,
                "gdp_impact_2050_pct": -4.5,
                "category": "transition",
            },
        ],
        "physical_risk_params": {
            "hazards": ["flooding", "heatwave", "drought", "wildfire", "storm"],
            "rcp_scenario": "RCP4.5",
            "facility_locations": [
                {"name": "Dusseldorf", "lat": 51.2277, "lon": 6.7735, "country": "DE"},
                {"name": "Lyon", "lat": 45.7640, "lon": 4.8357, "country": "FR"},
                {"name": "Milan", "lat": 45.4642, "lon": 9.1900, "country": "IT"},
                {"name": "Madrid", "lat": 40.4168, "lon": -3.7038, "country": "ES"},
                {"name": "Amsterdam", "lat": 52.3676, "lon": 4.9041, "country": "NL"},
                {"name": "Krakow", "lat": 50.0647, "lon": 19.9450, "country": "PL"},
            ],
        },
        "monte_carlo": {
            "enabled": True,
            "iterations": 10000,
            "confidence_level": 0.95,
        },
    }


# ---------------------------------------------------------------------------
# Stakeholders fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_stakeholders() -> List[Dict[str, Any]]:
    """Create 15 stakeholders across 7 categories with salience scores.

    Categories: investors, employees, customers, suppliers, regulators,
    NGOs, communities.
    """
    return [
        # Investors (3)
        {"id": "SH-001", "name": "Deutsche Bank Asset Management", "category": "investor",
         "sub_category": "institutional_investor", "influence": 9.0, "urgency": 7.0,
         "legitimacy": 9.0, "salience_score": 8.3, "engagement_method": "survey"},
        {"id": "SH-002", "name": "Allianz Global Investors", "category": "investor",
         "sub_category": "institutional_investor", "influence": 8.5, "urgency": 6.5,
         "legitimacy": 8.5, "salience_score": 7.8, "engagement_method": "interview"},
        {"id": "SH-003", "name": "Retail Shareholders Association", "category": "investor",
         "sub_category": "retail_investor", "influence": 5.0, "urgency": 4.0,
         "legitimacy": 7.0, "salience_score": 5.3, "engagement_method": "survey"},
        # Employees (3)
        {"id": "SH-004", "name": "Works Council (Betriebsrat)", "category": "employee",
         "sub_category": "employee_representative", "influence": 8.0, "urgency": 8.0,
         "legitimacy": 9.0, "salience_score": 8.3, "engagement_method": "workshop"},
        {"id": "SH-005", "name": "IG Metall Union Representatives", "category": "employee",
         "sub_category": "trade_union", "influence": 7.5, "urgency": 7.0,
         "legitimacy": 8.5, "salience_score": 7.7, "engagement_method": "workshop"},
        {"id": "SH-006", "name": "Management Team", "category": "employee",
         "sub_category": "senior_management", "influence": 9.5, "urgency": 8.5,
         "legitimacy": 9.0, "salience_score": 9.0, "engagement_method": "interview"},
        # Customers (2)
        {"id": "SH-007", "name": "Automotive OEM Consortium", "category": "customer",
         "sub_category": "key_account", "influence": 8.0, "urgency": 7.5,
         "legitimacy": 8.0, "salience_score": 7.8, "engagement_method": "survey"},
        {"id": "SH-008", "name": "Aerospace Component Buyers", "category": "customer",
         "sub_category": "key_account", "influence": 7.0, "urgency": 6.0,
         "legitimacy": 7.5, "salience_score": 6.8, "engagement_method": "survey"},
        # Suppliers (2)
        {"id": "SH-009", "name": "Steel Suppliers Association", "category": "supplier",
         "sub_category": "tier1_supplier", "influence": 6.0, "urgency": 5.0,
         "legitimacy": 7.0, "salience_score": 6.0, "engagement_method": "questionnaire"},
        {"id": "SH-010", "name": "Chemical Suppliers Group", "category": "supplier",
         "sub_category": "tier1_supplier", "influence": 5.5, "urgency": 5.5,
         "legitimacy": 6.5, "salience_score": 5.8, "engagement_method": "questionnaire"},
        # Regulators (2)
        {"id": "SH-011", "name": "BaFin (Federal Financial Supervisory)", "category": "regulator",
         "sub_category": "financial_regulator", "influence": 9.5, "urgency": 9.0,
         "legitimacy": 10.0, "salience_score": 9.5, "engagement_method": "compliance"},
        {"id": "SH-012", "name": "European Securities & Markets Authority", "category": "regulator",
         "sub_category": "eu_regulator", "influence": 9.0, "urgency": 8.0,
         "legitimacy": 10.0, "salience_score": 9.0, "engagement_method": "compliance"},
        # NGOs (2)
        {"id": "SH-013", "name": "CDP Europe", "category": "ngo",
         "sub_category": "environmental_ngo", "influence": 7.0, "urgency": 6.0,
         "legitimacy": 8.0, "salience_score": 7.0, "engagement_method": "disclosure"},
        {"id": "SH-014", "name": "Germanwatch", "category": "ngo",
         "sub_category": "environmental_ngo", "influence": 6.0, "urgency": 5.5,
         "legitimacy": 7.5, "salience_score": 6.3, "engagement_method": "dialogue"},
        # Communities (1)
        {"id": "SH-015", "name": "Dusseldorf Community Council", "category": "community",
         "sub_category": "local_community", "influence": 4.0, "urgency": 3.0,
         "legitimacy": 7.0, "salience_score": 4.7, "engagement_method": "public_meeting"},
    ]


# ---------------------------------------------------------------------------
# Regulatory changes fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_regulatory_changes() -> List[Dict[str, Any]]:
    """Create 5 regulatory changes with varying severity levels.

    Covers ESRS amendments, EU Taxonomy updates, and national
    transposition changes.
    """
    return [
        {
            "change_id": "REG-2025-001",
            "title": "ESRS Set 2 - Sector-Specific Standards (Manufacturing)",
            "regulation": "ESRS Sector Standards",
            "authority": "EFRAG",
            "publication_date": "2025-06-15",
            "effective_date": "2026-01-01",
            "severity": "high",
            "impact_areas": ["E1", "E2", "E5"],
            "description": "New sector-specific disclosure requirements for manufacturing.",
            "action_required": True,
            "estimated_effort_days": 30,
        },
        {
            "change_id": "REG-2025-002",
            "title": "EU Taxonomy Climate Delegated Act Amendment",
            "regulation": "Regulation (EU) 2020/852",
            "authority": "European Commission",
            "publication_date": "2025-03-01",
            "effective_date": "2025-10-01",
            "severity": "medium",
            "impact_areas": ["EU_TAXONOMY"],
            "description": "Updated technical screening criteria for climate mitigation.",
            "action_required": True,
            "estimated_effort_days": 15,
        },
        {
            "change_id": "REG-2025-003",
            "title": "German CSRD Transposition - CSR-RUG Amendments",
            "regulation": "CSR-Richtlinie-Umsetzungsgesetz",
            "authority": "German Federal Government",
            "publication_date": "2025-04-15",
            "effective_date": "2025-07-01",
            "severity": "high",
            "impact_areas": ["CSRD", "ESRS"],
            "description": "German transposition adds national filing requirements.",
            "action_required": True,
            "estimated_effort_days": 20,
        },
        {
            "change_id": "REG-2025-004",
            "title": "ISAE 3410 GHG Assurance Update",
            "regulation": "ISAE 3410 (Revised)",
            "authority": "IAASB",
            "publication_date": "2025-09-01",
            "effective_date": "2026-06-01",
            "severity": "low",
            "impact_areas": ["ISAE3410"],
            "description": "Minor clarification on Scope 3 assurance procedures.",
            "action_required": False,
            "estimated_effort_days": 5,
        },
        {
            "change_id": "REG-2025-005",
            "title": "CDP 2026 Climate Change Questionnaire Update",
            "regulation": "CDP Guidance",
            "authority": "CDP",
            "publication_date": "2025-11-01",
            "effective_date": "2026-02-01",
            "severity": "medium",
            "impact_areas": ["CDP"],
            "description": "New questions on supply chain decarbonization and Scope 3.",
            "action_required": True,
            "estimated_effort_days": 10,
        },
    ]


# ---------------------------------------------------------------------------
# Benchmark data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_benchmark_data() -> Dict[str, Any]:
    """Peer comparison data for 20 metrics across the manufacturing sector.

    Includes peer average, best-in-class, and EuroTech actual values.
    """
    return {
        "sector": "manufacturing",
        "nace_codes": ["C28"],
        "peer_count": 25,
        "benchmark_date": "2025-12-31",
        "metrics": [
            {"id": "BM-01", "name": "Scope 1 Intensity (tCO2e/MEUR)", "eurotech": 10.5, "peer_avg": 14.2, "best_in_class": 6.8, "unit": "tCO2e/MEUR"},
            {"id": "BM-02", "name": "Scope 2 Intensity (tCO2e/MEUR)", "eurotech": 6.8, "peer_avg": 9.1, "best_in_class": 3.2, "unit": "tCO2e/MEUR"},
            {"id": "BM-03", "name": "Scope 3 Intensity (tCO2e/MEUR)", "eurotech": 30.1, "peer_avg": 35.5, "best_in_class": 18.0, "unit": "tCO2e/MEUR"},
            {"id": "BM-04", "name": "Total GHG Intensity", "eurotech": 47.4, "peer_avg": 58.8, "best_in_class": 28.0, "unit": "tCO2e/MEUR"},
            {"id": "BM-05", "name": "Renewable Energy Share (%)", "eurotech": 42.0, "peer_avg": 28.0, "best_in_class": 80.0, "unit": "percent"},
            {"id": "BM-06", "name": "Energy Intensity (MWh/MEUR)", "eurotech": 125.0, "peer_avg": 158.0, "best_in_class": 85.0, "unit": "MWh/MEUR"},
            {"id": "BM-07", "name": "Water Intensity (m3/MEUR)", "eurotech": 62.0, "peer_avg": 88.0, "best_in_class": 35.0, "unit": "m3/MEUR"},
            {"id": "BM-08", "name": "Waste Recycling Rate (%)", "eurotech": 72.0, "peer_avg": 55.0, "best_in_class": 92.0, "unit": "percent"},
            {"id": "BM-09", "name": "Female Employees (%)", "eurotech": 38.0, "peer_avg": 30.0, "best_in_class": 48.0, "unit": "percent"},
            {"id": "BM-10", "name": "Gender Pay Gap (%)", "eurotech": 6.5, "peer_avg": 12.0, "best_in_class": 2.0, "unit": "percent"},
            {"id": "BM-11", "name": "LTIR (per million hours)", "eurotech": 1.5, "peer_avg": 3.2, "best_in_class": 0.5, "unit": "per_million_hrs"},
            {"id": "BM-12", "name": "Training Hours per Employee", "eurotech": 22.0, "peer_avg": 18.0, "best_in_class": 40.0, "unit": "hours"},
            {"id": "BM-13", "name": "Board Independence (%)", "eurotech": 66.0, "peer_avg": 58.0, "best_in_class": 85.0, "unit": "percent"},
            {"id": "BM-14", "name": "Board Gender Diversity (%)", "eurotech": 33.0, "peer_avg": 28.0, "best_in_class": 50.0, "unit": "percent"},
            {"id": "BM-15", "name": "SBTi Target Set", "eurotech": 1.0, "peer_avg": 0.45, "best_in_class": 1.0, "unit": "boolean"},
            {"id": "BM-16", "name": "CDP Score", "eurotech": 7.0, "peer_avg": 5.5, "best_in_class": 9.0, "unit": "score_0_9"},
            {"id": "BM-17", "name": "EU Taxonomy Aligned Turnover (%)", "eurotech": 32.8, "peer_avg": 22.0, "best_in_class": 55.0, "unit": "percent"},
            {"id": "BM-18", "name": "ESRS Compliance Score (%)", "eurotech": 91.0, "peer_avg": 72.0, "best_in_class": 96.0, "unit": "percent"},
            {"id": "BM-19", "name": "Supplier Audit Coverage (%)", "eurotech": 68.0, "peer_avg": 45.0, "best_in_class": 85.0, "unit": "percent"},
            {"id": "BM-20", "name": "Anti-Corruption Training (%)", "eurotech": 95.0, "peer_avg": 82.0, "best_in_class": 100.0, "unit": "percent"},
        ],
        "esg_rating_predictions": {
            "msci": {"predicted": "AA", "peer_avg": "A", "best_in_class": "AAA"},
            "sustainalytics": {"predicted": 18.5, "peer_avg": 28.0, "best_in_class": 8.0},
            "cdp": {"predicted": "A-", "peer_avg": "B", "best_in_class": "A"},
            "iss": {"predicted": "B+", "peer_avg": "C+", "best_in_class": "A"},
        },
    }


# ---------------------------------------------------------------------------
# Webhook configuration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_webhook_config() -> List[Dict[str, Any]]:
    """Create 3 webhook subscriptions: HTTP, Slack, and email."""
    return [
        {
            "webhook_id": "WH-001",
            "name": "Compliance Dashboard",
            "type": "http",
            "url": "https://hooks.eurotech.de/compliance/events",
            "events": ["quality_gate_passed", "quality_gate_failed", "approval_completed"],
            "secret": "whsec_eurotech_compliance_2025",
            "hmac_algorithm": "sha256",
            "retry_policy": {"max_retries": 3, "backoff_seconds": [5, 30, 120]},
            "active": True,
        },
        {
            "webhook_id": "WH-002",
            "name": "Sustainability Slack Channel",
            "type": "slack",
            "url": "https://hooks.slack.com/services/T01/B02/xyz123",
            "events": ["workflow_completed", "regulatory_change_detected", "approval_required"],
            "secret": None,
            "hmac_algorithm": None,
            "retry_policy": {"max_retries": 2, "backoff_seconds": [10, 60]},
            "active": True,
        },
        {
            "webhook_id": "WH-003",
            "name": "Board Notification Email",
            "type": "email",
            "url": "mailto:board-esg@eurotech.de",
            "events": ["board_approval_required", "report_published"],
            "secret": None,
            "hmac_algorithm": None,
            "retry_policy": {"max_retries": 1, "backoff_seconds": [300]},
            "active": True,
        },
    ]


# ---------------------------------------------------------------------------
# Mock agent registry fixture (93+ agents)
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agent_registry() -> MagicMock:
    """Create a mocked agent registry with all 93+ pack agents.

    Includes all PACK-001 inherited agents (66) plus PACK-002 professional
    agents (27+) for CDP, TCFD, SBTi, EU Taxonomy, and professional engines.
    """
    registry = MagicMock()

    all_agent_ids = (
        # PACK-001 Inherited: Data agents
        [f"AGENT-DATA-{i:03d}" for i in [1, 2, 3, 8]]
        # PACK-001 Inherited: Quality agents
        + [f"AGENT-DATA-{i:03d}" for i in [10, 11, 12, 13, 19]]
        # PACK-001 Inherited: MRV Scope 1
        + [f"AGENT-MRV-{i:03d}" for i in range(1, 9)]
        # PACK-001 Inherited: MRV Scope 2
        + [f"AGENT-MRV-{i:03d}" for i in range(9, 14)]
        # PACK-001 Inherited: MRV Scope 3
        + [f"AGENT-MRV-{i:03d}" for i in range(14, 31)]
        # PACK-001 Inherited: Foundation
        + [f"AGENT-FOUND-{i:03d}" for i in range(1, 11)]
        # PACK-001 Inherited: App
        + ["GL-CSRD-APP"]
        # PACK-002: Professional Apps
        + ["GL-CDP-APP", "GL-TCFD-APP", "GL-SBTi-APP", "GL-TAXONOMY-APP"]
        # PACK-002: CDP Engines
        + ["GL-CDP-INTAKE", "GL-CDP-MAPPER", "GL-CDP-SCORER",
           "GL-CDP-CLIMATE", "GL-CDP-WATER", "GL-CDP-FORESTS"]
        # PACK-002: TCFD Engines
        + ["GL-TCFD-GOVERNANCE", "GL-TCFD-STRATEGY", "GL-TCFD-RISK",
           "GL-TCFD-METRICS", "GL-TCFD-SCENARIO", "GL-TCFD-PHYSICAL",
           "GL-TCFD-TRANSITION"]
        # PACK-002: SBTi Engines
        + ["GL-SBTi-BASELINE", "GL-SBTi-TARGET", "GL-SBTi-PATHWAY",
           "GL-SBTi-SCOPE3", "GL-SBTi-SECTOR", "GL-SBTi-VALIDATION",
           "GL-SBTi-FLAG", "GL-SBTi-FI"]
        # PACK-002: EU Taxonomy Engines
        + ["GL-TAX-ELIGIBILITY", "GL-TAX-ALIGNMENT", "GL-TAX-KPI",
           "GL-TAX-CLIMATE", "GL-TAX-ENVIRON", "GL-TAX-SOCIAL"]
        # PACK-002: Professional Engines
        + ["GL-PRO-CONSOLIDATION", "GL-PRO-APPROVAL", "GL-PRO-QUALITY-GATE",
           "GL-PRO-CROSS-FRAMEWORK", "GL-PRO-SCENARIO", "GL-PRO-BENCHMARK",
           "GL-PRO-REGULATORY"]
    )

    def is_available(agent_id: str) -> bool:
        return agent_id in all_agent_ids

    def get_agent(agent_id: str) -> MagicMock:
        if agent_id not in all_agent_ids:
            raise KeyError(f"Agent {agent_id} not registered")
        agent_mock = MagicMock()
        agent_mock.agent_id = agent_id
        agent_mock.status = "healthy"
        agent_mock.version = "1.0.0"
        return agent_mock

    registry.is_available.side_effect = is_available
    registry.get_agent.side_effect = get_agent
    registry.list_agents.return_value = all_agent_ids
    registry.health_check.return_value = {aid: "healthy" for aid in all_agent_ids}
    registry.agent_count = len(all_agent_ids)
    return registry


# ---------------------------------------------------------------------------
# Professional size presets
# ---------------------------------------------------------------------------

ENTERPRISE_GROUP_PRESET = {
    "preset_id": "enterprise_group",
    "display_name": "Enterprise Group",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4", "G1",
    ],
    "scope3_categories": list(range(1, 16)),
    "xbrl_mode": "full",
    "consolidation": {
        "enabled": True,
        "default_approach": "operational_control",
        "max_subsidiaries": 100,
        "intercompany_elimination": True,
        "minority_disclosures": True,
    },
    "approval": {
        "levels": 4,
        "auto_approve_threshold": 95.0,
        "escalation_timeout_hours": 48,
    },
    "quality_gates": {
        "qg1_threshold": 85.0,
        "qg2_threshold": 90.0,
        "qg3_threshold": 80.0,
    },
    "cross_framework": ["cdp", "tcfd", "sbti", "eu_taxonomy", "gri", "sasb"],
    "scenario_analysis": ["IEA_NZE", "IEA_APS", "NGFS_ORDERLY", "NGFS_DISORDERLY"],
    "assurance_level": "reasonable",
    "languages": ["en", "de", "fr", "es", "it", "nl", "pl"],
    "max_data_points": 5000,
}

LISTED_COMPANY_PRESET = {
    "preset_id": "listed_company",
    "display_name": "Listed Company",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4", "G1",
    ],
    "scope3_categories": list(range(1, 16)),
    "xbrl_mode": "full",
    "consolidation": {
        "enabled": True,
        "default_approach": "financial_control",
        "max_subsidiaries": 50,
        "intercompany_elimination": True,
        "minority_disclosures": True,
    },
    "approval": {
        "levels": 3,
        "auto_approve_threshold": 90.0,
        "escalation_timeout_hours": 72,
    },
    "quality_gates": {
        "qg1_threshold": 85.0,
        "qg2_threshold": 90.0,
        "qg3_threshold": 85.0,
    },
    "cross_framework": ["cdp", "tcfd", "sbti", "eu_taxonomy"],
    "scenario_analysis": ["IEA_NZE", "NGFS_ORDERLY"],
    "assurance_level": "limited",
    "languages": ["en"],
    "max_data_points": 3000,
}

FINANCIAL_INSTITUTION_PRESET = {
    "preset_id": "financial_institution",
    "display_name": "Financial Institution",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4", "G1",
    ],
    "scope3_categories": [1, 6, 7, 13, 15],
    "xbrl_mode": "full",
    "consolidation": {
        "enabled": True,
        "default_approach": "equity_share",
        "max_subsidiaries": 50,
        "intercompany_elimination": True,
        "minority_disclosures": True,
    },
    "approval": {
        "levels": 4,
        "auto_approve_threshold": 95.0,
        "escalation_timeout_hours": 48,
    },
    "quality_gates": {
        "qg1_threshold": 90.0,
        "qg2_threshold": 95.0,
        "qg3_threshold": 85.0,
    },
    "cross_framework": ["cdp", "tcfd", "sbti", "eu_taxonomy", "gri"],
    "scenario_analysis": ["IEA_NZE", "IEA_APS", "NGFS_ORDERLY", "NGFS_DISORDERLY"],
    "assurance_level": "reasonable",
    "pcaf_enabled": True,
    "gar_enabled": True,
    "fi_targets": True,
    "languages": ["en", "de"],
    "max_data_points": 4000,
}

MULTINATIONAL_PRESET = {
    "preset_id": "multinational",
    "display_name": "Multinational Corporation",
    "esrs_standards": [
        "ESRS_1", "ESRS_2",
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4", "G1",
    ],
    "scope3_categories": list(range(1, 16)),
    "xbrl_mode": "full",
    "consolidation": {
        "enabled": True,
        "default_approach": "operational_control",
        "max_subsidiaries": 200,
        "intercompany_elimination": True,
        "minority_disclosures": True,
    },
    "approval": {
        "levels": 4,
        "auto_approve_threshold": 95.0,
        "escalation_timeout_hours": 48,
    },
    "quality_gates": {
        "qg1_threshold": 85.0,
        "qg2_threshold": 90.0,
        "qg3_threshold": 80.0,
    },
    "cross_framework": ["cdp", "tcfd", "sbti", "eu_taxonomy", "gri", "sasb"],
    "scenario_analysis": ["IEA_NZE", "IEA_APS", "NGFS_ORDERLY", "NGFS_DISORDERLY"],
    "assurance_level": "reasonable",
    "multi_jurisdiction": True,
    "multi_currency": True,
    "languages": ["en", "de", "fr", "es", "it", "nl", "pl", "pt", "sv", "da"],
    "max_data_points": 10000,
}

ALL_PRO_PRESETS = {
    "enterprise_group": ENTERPRISE_GROUP_PRESET,
    "listed_company": LISTED_COMPANY_PRESET,
    "financial_institution": FINANCIAL_INSTITUTION_PRESET,
    "multinational": MULTINATIONAL_PRESET,
}


@pytest.fixture(params=["enterprise_group", "listed_company", "financial_institution", "multinational"])
def pro_preset_config(request) -> Dict[str, Any]:
    """Parameterized fixture that yields each professional size preset."""
    return ALL_PRO_PRESETS[request.param]


# ---------------------------------------------------------------------------
# Professional sector presets
# ---------------------------------------------------------------------------

MANUFACTURING_PRO_SECTOR = {
    "sector_id": "manufacturing_pro",
    "display_name": "Manufacturing Professional",
    "nace_codes": ["C10-C33"],
    "emission_focus": ["scope1_process", "scope1_fugitive", "scope1_stationary"],
    "eu_ets_integration": True,
    "cbam_preparedness": True,
    "sbti_pathway": "1.5C_SDA_Manufacturing",
    "sector_scenario_analysis": True,
}

FINANCIAL_SERVICES_PRO_SECTOR = {
    "sector_id": "financial_services_pro",
    "display_name": "Financial Services Professional",
    "nace_codes": ["K64-K66"],
    "emission_focus": ["scope3_cat15_investments", "scope3_cat6_travel"],
    "pcaf_enabled": True,
    "gar_btar_reporting": True,
    "financed_emissions": True,
    "sbti_pathway": "FI_Portfolio_1.5C",
    "stranded_asset_analysis": True,
}

TECHNOLOGY_PRO_SECTOR = {
    "sector_id": "technology_pro",
    "display_name": "Technology Professional",
    "nace_codes": ["J58-J63"],
    "emission_focus": ["scope2_electricity", "scope3_cat7_commuting"],
    "sci_pue_metrics": True,
    "data_center_optimization": True,
    "sbti_pathway": "1.5C_ACA_Technology",
}

ENERGY_PRO_SECTOR = {
    "sector_id": "energy_pro",
    "display_name": "Energy Professional",
    "nace_codes": ["B05-B09", "D35"],
    "emission_focus": ["scope1_stationary", "scope1_process", "scope1_fugitive"],
    "eu_ets_integration": True,
    "ogmp_methane": True,
    "stranded_asset_risk": True,
    "sbti_pathway": "1.5C_SDA_Power",
}

HEAVY_INDUSTRY_PRO_SECTOR = {
    "sector_id": "heavy_industry_pro",
    "display_name": "Heavy Industry Professional",
    "nace_codes": ["B05-B09", "C19-C25"],
    "emission_focus": ["scope1_process", "scope1_stationary"],
    "eu_ets_integration": True,
    "cbam_preparedness": True,
    "process_emission_intensity": True,
    "sbti_pathway": "1.5C_SDA_Heavy_Industry",
}

ALL_PRO_SECTORS = {
    "manufacturing_pro": MANUFACTURING_PRO_SECTOR,
    "financial_services_pro": FINANCIAL_SERVICES_PRO_SECTOR,
    "technology_pro": TECHNOLOGY_PRO_SECTOR,
    "energy_pro": ENERGY_PRO_SECTOR,
    "heavy_industry_pro": HEAVY_INDUSTRY_PRO_SECTOR,
}


@pytest.fixture(params=list(ALL_PRO_SECTORS.keys()))
def pro_sector_config(request) -> Dict[str, Any]:
    """Parameterized fixture that yields each professional sector config."""
    return ALL_PRO_SECTORS[request.param]


# ---------------------------------------------------------------------------
# Utility fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "pack_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# ESRS standards reference
# ---------------------------------------------------------------------------

VALID_ESRS_STANDARDS = [
    "ESRS_1", "ESRS_2",
    "E1", "E2", "E3", "E4", "E5",
    "S1", "S2", "S3", "S4",
    "G1",
]

VALID_SCOPE3_CATEGORIES = list(range(1, 16))

PROFESSIONAL_COMPLIANCE_REFS = ["CSRD", "ESRS", "ESEF", "ISAE3000", "ISAE3410", "EU_TAXONOMY"]

PROFESSIONAL_WORKFLOWS = [
    "consolidated_reporting", "cross_framework_alignment", "scenario_analysis",
    "continuous_compliance", "stakeholder_engagement", "regulatory_change_mgmt",
    "board_governance", "professional_audit",
]

PROFESSIONAL_TEMPLATES = [
    "consolidated_group_report", "cdp_questionnaire_response",
    "tcfd_disclosure", "sbti_progress_report", "eu_taxonomy_report",
    "scenario_analysis_report", "cross_framework_mapping",
    "board_governance_package", "regulatory_change_briefing",
    "professional_audit_package",
]

PROFESSIONAL_ENGINES = [
    "GL-PRO-CONSOLIDATION", "GL-PRO-APPROVAL", "GL-PRO-QUALITY-GATE",
    "GL-PRO-CROSS-FRAMEWORK", "GL-PRO-SCENARIO", "GL-PRO-BENCHMARK",
    "GL-PRO-REGULATORY",
]
