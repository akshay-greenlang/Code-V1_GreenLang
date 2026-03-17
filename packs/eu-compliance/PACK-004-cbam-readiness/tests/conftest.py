# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - Shared Test Fixtures
=========================================================

Provides reusable pytest fixtures for all PACK-004 test modules including
CBAM calculation, certificate management, quarterly reporting, supplier
management, de minimis tracking, verification, and policy compliance.

All fixtures are self-contained with no external dependencies.
Every external service is mocked via stub classes in this module.

Author: GreenLang QA Team
Version: 1.0.0
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Paths & sys.path setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent

_engines_dir = str(PACK_ROOT / "engines")
if _engines_dir not in sys.path:
    sys.path.insert(0, _engines_dir)

_pack_root_str = str(PACK_ROOT)
if _pack_root_str not in sys.path:
    sys.path.insert(0, _pack_root_str)

_config_dir = str(PACK_ROOT / "config")
if _config_dir not in sys.path:
    sys.path.insert(0, _config_dir)

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
# Utility helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of data for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    elif hasattr(data, "model_dump"):
        raw = json.dumps(data.model_dump(mode="json"), sort_keys=True, default=str)
    else:
        raw = json.dumps(str(data), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4."""
    return str(uuid.uuid4())


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
    if pack_yaml_path.exists():
        return pack_yaml_path.read_text(encoding="utf-8")
    return ""


@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary."""
    if pack_yaml_raw:
        return yaml.safe_load(pack_yaml_raw)
    return {}


# ---------------------------------------------------------------------------
# Preset / Sector / Demo loading fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def preset_files() -> Dict[str, Path]:
    """Return mapping of preset ID to file path."""
    result: Dict[str, Path] = {}
    if PRESETS_DIR.exists():
        for f in PRESETS_DIR.glob("*.yaml"):
            result[f.stem] = f
    return result


@pytest.fixture(scope="session")
def sector_files() -> Dict[str, Path]:
    """Return mapping of sector ID to file path."""
    result: Dict[str, Path] = {}
    if SECTORS_DIR.exists():
        for f in SECTORS_DIR.glob("*.yaml"):
            result[f.stem] = f
    return result


@pytest.fixture(scope="session")
def demo_config_path() -> Path:
    """Return path to demo configuration."""
    return DEMO_DIR / "demo_config.yaml"


@pytest.fixture(scope="session")
def demo_config(demo_config_path) -> Dict[str, Any]:
    """Return parsed demo configuration."""
    if demo_config_path.exists():
        return yaml.safe_load(demo_config_path.read_text(encoding="utf-8"))
    return {}


@pytest.fixture(scope="session")
def demo_imports_csv_path() -> Path:
    """Return path to demo imports CSV."""
    return DEMO_DIR / "demo_imports.csv"


@pytest.fixture(scope="session")
def demo_supplier_json_path() -> Path:
    """Return path to demo supplier JSON."""
    return DEMO_DIR / "demo_suppliers.json"


# ---------------------------------------------------------------------------
# CBAM PackConfig fixture (dict-based, no external deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cbam_config() -> Dict[str, Any]:
    """Create a full CBAMPackConfig dict with all sub-configs.

    This is a comprehensive configuration for CBAM pack testing.
    """
    return {
        "metadata": {
            "name": "cbam-readiness",
            "version": "1.0.0",
            "display_name": "CBAM Readiness Pack",
            "description": "EU Carbon Border Adjustment Mechanism compliance pack",
            "category": "eu-compliance",
            "tier": "professional",
            "author": "GreenLang Platform Team",
            "license": "Proprietary",
            "min_platform_version": "2.0.0",
            "release_date": "2026-03-14",
            "support_tier": "professional",
            "tags": [
                "cbam", "carbon-border-adjustment", "eu-compliance",
                "emission-calculation", "certificate-management",
            ],
            "compliance_references": [
                {
                    "id": "CBAM",
                    "name": "Carbon Border Adjustment Mechanism",
                    "regulation": "Regulation (EU) 2023/956",
                    "effective_date": "2023-10-01",
                    "description": "EU carbon border adjustment for imported goods",
                },
                {
                    "id": "CBAM-IR",
                    "name": "CBAM Implementing Regulation",
                    "regulation": "Implementing Regulation (EU) 2023/1773",
                    "effective_date": "2023-10-01",
                    "description": "CBAM transitional period implementing rules",
                },
                {
                    "id": "EU-ETS",
                    "name": "EU Emissions Trading System",
                    "regulation": "Directive 2003/87/EC (amended)",
                    "effective_date": "2005-01-01",
                    "description": "EU cap-and-trade system for GHG emissions",
                },
            ],
        },
        "cbam": {
            "importer": {
                "company_name": "EuroSteel Imports GmbH",
                "eori_number": "DE123456789012345",
                "registration_id": "CBAM-DE-2026-00001",
                "contact_email": "cbam@eurosteel-imports.de",
                "member_state": "DE",
                "authorized_representative": "Hans Mueller",
            },
            "goods_categories": {
                "enabled": ["cement", "steel", "aluminium", "fertilizers",
                            "electricity", "hydrogen"],
                "primary_categories": ["steel", "aluminium", "cement"],
            },
            "emission_config": {
                "default_methodology": "actual",
                "fallback_methodology": "default_values",
                "precision_decimal_places": 6,
                "unit": "tCO2e",
                "include_indirect": True,
                "gwp_source": "AR5",
            },
            "certificate_config": {
                "currency": "EUR",
                "price_source": "EU_ETS_AUCTION",
                "free_allocation_schedule": {
                    "2026": 0.975,
                    "2027": 0.925,
                    "2028": 0.850,
                    "2029": 0.775,
                    "2030": 0.515,
                    "2031": 0.390,
                    "2032": 0.265,
                    "2033": 0.140,
                    "2034": 0.000,
                },
                "surrender_deadline_months_after_year": 5,
            },
            "quarterly_config": {
                "reporting_quarters": ["Q1", "Q2", "Q3", "Q4"],
                "xml_schema_version": "1.0.0",
                "deadline_days_after_quarter": 30,
                "amendment_window_days": 60,
            },
            "supplier_config": {
                "max_installations_per_supplier": 10,
                "data_request_frequency": "quarterly",
                "quality_score_threshold": 70.0,
            },
            "deminimis_config": {
                "annual_weight_threshold_kg": 150000,
                "annual_value_threshold_eur": 150.0,
                "alert_at_pct": 80,
            },
            "verification_config": {
                "verification_body_accreditation": "DAkkS",
                "materiality_threshold_pct": 5.0,
                "statement_validity_months": 12,
            },
        },
    }


# ---------------------------------------------------------------------------
# Importer Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_importer_config() -> Dict[str, Any]:
    """Create ImporterConfig for EuroSteel Imports GmbH."""
    return {
        "company_name": "EuroSteel Imports GmbH",
        "eori_number": "DE123456789012345",
        "registration_id": "CBAM-DE-2026-00001",
        "contact_email": "cbam@eurosteel-imports.de",
        "member_state": "DE",
        "authorized_representative": "Hans Mueller",
        "address": {
            "street": "Stahlstrasse 42",
            "city": "Duisburg",
            "postal_code": "47053",
            "country": "DE",
        },
    }


# ---------------------------------------------------------------------------
# Emission Input/Result fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_emission_inputs() -> List[Dict[str, Any]]:
    """Create 10 EmissionInput fixtures (mix of steel, aluminium, cement)."""
    inputs = [
        {
            "input_id": "EI-001",
            "cn_code": "7207 11 14",
            "goods_category": "steel",
            "description": "Semi-finished products of iron",
            "origin_country": "TR",
            "weight_tonnes": 500.0,
            "supplier_id": "SUP-TR-001",
            "installation_id": "INST-TR-001",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 1.85,
            "import_date": "2026-01-15",
        },
        {
            "input_id": "EI-002",
            "cn_code": "7208 51 20",
            "goods_category": "steel",
            "description": "Flat-rolled products of iron, hot-rolled",
            "origin_country": "CN",
            "weight_tonnes": 300.0,
            "supplier_id": "SUP-CN-001",
            "installation_id": "INST-CN-001",
            "emission_methodology": "default_values",
            "specific_emission_tco2e_per_tonne": 2.30,
            "import_date": "2026-01-22",
        },
        {
            "input_id": "EI-003",
            "cn_code": "7601 10 00",
            "goods_category": "aluminium",
            "description": "Unwrought aluminium, not alloyed",
            "origin_country": "IN",
            "weight_tonnes": 200.0,
            "supplier_id": "SUP-IN-001",
            "installation_id": "INST-IN-001",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 8.50,
            "import_date": "2026-02-05",
        },
        {
            "input_id": "EI-004",
            "cn_code": "2523 29 00",
            "goods_category": "cement",
            "description": "Portland cement, grey",
            "origin_country": "TR",
            "weight_tonnes": 1000.0,
            "supplier_id": "SUP-TR-002",
            "installation_id": "INST-TR-002",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 0.65,
            "import_date": "2026-02-10",
        },
        {
            "input_id": "EI-005",
            "cn_code": "7207 20 80",
            "goods_category": "steel",
            "description": "Semi-finished products of stainless steel",
            "origin_country": "IN",
            "weight_tonnes": 150.0,
            "supplier_id": "SUP-IN-002",
            "installation_id": "INST-IN-002",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 2.10,
            "import_date": "2026-02-18",
        },
        {
            "input_id": "EI-006",
            "cn_code": "7601 20 80",
            "goods_category": "aluminium",
            "description": "Aluminium alloys, unwrought",
            "origin_country": "CN",
            "weight_tonnes": 120.0,
            "supplier_id": "SUP-CN-002",
            "installation_id": "INST-CN-002",
            "emission_methodology": "default_values",
            "specific_emission_tco2e_per_tonne": 9.20,
            "import_date": "2026-02-25",
        },
        {
            "input_id": "EI-007",
            "cn_code": "7208 10 00",
            "goods_category": "steel",
            "description": "Flat-rolled products, in coils",
            "origin_country": "TR",
            "weight_tonnes": 450.0,
            "supplier_id": "SUP-TR-001",
            "installation_id": "INST-TR-001",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 1.90,
            "import_date": "2026-03-02",
        },
        {
            "input_id": "EI-008",
            "cn_code": "2523 21 00",
            "goods_category": "cement",
            "description": "White Portland cement",
            "origin_country": "CN",
            "weight_tonnes": 800.0,
            "supplier_id": "SUP-CN-003",
            "installation_id": "INST-CN-003",
            "emission_methodology": "default_values",
            "specific_emission_tco2e_per_tonne": 0.72,
            "import_date": "2026-03-08",
        },
        {
            "input_id": "EI-009",
            "cn_code": "7604 10 10",
            "goods_category": "aluminium",
            "description": "Aluminium bars and rods",
            "origin_country": "IN",
            "weight_tonnes": 80.0,
            "supplier_id": "SUP-IN-001",
            "installation_id": "INST-IN-001",
            "emission_methodology": "actual",
            "specific_emission_tco2e_per_tonne": 7.80,
            "import_date": "2026-03-12",
        },
        {
            "input_id": "EI-010",
            "cn_code": "7209 15 00",
            "goods_category": "steel",
            "description": "Flat-rolled products, cold-rolled",
            "origin_country": "CN",
            "weight_tonnes": 250.0,
            "supplier_id": "SUP-CN-001",
            "installation_id": "INST-CN-001",
            "emission_methodology": "default_values",
            "specific_emission_tco2e_per_tonne": 2.45,
            "import_date": "2026-03-20",
        },
    ]
    return inputs


@pytest.fixture
def sample_emission_results(sample_emission_inputs) -> List[Dict[str, Any]]:
    """Pre-calculated EmissionResult fixtures matching emission inputs."""
    results = []
    for inp in sample_emission_inputs:
        total_emissions = round(
            inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"], 6
        )
        results.append({
            "input_id": inp["input_id"],
            "cn_code": inp["cn_code"],
            "goods_category": inp["goods_category"],
            "origin_country": inp["origin_country"],
            "weight_tonnes": inp["weight_tonnes"],
            "specific_emission_tco2e_per_tonne": inp["specific_emission_tco2e_per_tonne"],
            "total_emissions_tco2e": total_emissions,
            "methodology": inp["emission_methodology"],
            "calculation_date": _utcnow().strftime("%Y-%m-%d"),
            "provenance_hash": _compute_hash({
                "input_id": inp["input_id"],
                "weight": inp["weight_tonnes"],
                "ef": inp["specific_emission_tco2e_per_tonne"],
            }),
        })
    return results


# ---------------------------------------------------------------------------
# Quarterly Report fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_quarterly_report(
    sample_emission_results,
    sample_importer_config,
) -> Dict[str, Any]:
    """Create QuarterlyReport fixture for Q1 2026."""
    total_emissions = sum(r["total_emissions_tco2e"] for r in sample_emission_results)
    by_category = {}
    by_country = {}
    for r in sample_emission_results:
        cat = r["goods_category"]
        by_category[cat] = by_category.get(cat, 0.0) + r["total_emissions_tco2e"]
        country = r["origin_country"]
        by_country[country] = by_country.get(country, 0.0) + r["total_emissions_tco2e"]

    return {
        "report_id": "QR-2026-Q1-001",
        "importer": sample_importer_config,
        "reporting_period": {
            "quarter": "Q1",
            "year": 2026,
            "start_date": "2026-01-01",
            "end_date": "2026-03-31",
        },
        "total_imports": len(sample_emission_results),
        "total_weight_tonnes": sum(r["weight_tonnes"] for r in sample_emission_results),
        "total_emissions_tco2e": round(total_emissions, 6),
        "emissions_by_category": {k: round(v, 6) for k, v in by_category.items()},
        "emissions_by_country": {k: round(v, 6) for k, v in by_country.items()},
        "emission_results": sample_emission_results,
        "status": "draft",
        "xml_generated": False,
        "deadline": "2026-04-30",
        "version": 1,
        "amendment_history": [],
        "provenance_hash": _compute_hash({
            "report_id": "QR-2026-Q1-001",
            "total_emissions": round(total_emissions, 6),
        }),
    }


# ---------------------------------------------------------------------------
# Certificate Obligation fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_certificate_obligation(
    sample_emission_results,
) -> Dict[str, Any]:
    """Create CertificateObligation fixture."""
    total_emissions = sum(r["total_emissions_tco2e"] for r in sample_emission_results)
    free_allocation_pct = 0.975  # 2026 rate
    gross_obligation = total_emissions
    free_allocation = gross_obligation * free_allocation_pct
    # Turkey has a carbon pricing mechanism
    turkey_emissions = sum(
        r["total_emissions_tco2e"] for r in sample_emission_results
        if r["origin_country"] == "TR"
    )
    carbon_price_deduction = round(turkey_emissions * 0.10, 6)  # example deduction
    net_obligation = max(0, gross_obligation - free_allocation - carbon_price_deduction)
    ets_price_eur = 78.50

    return {
        "obligation_id": "OBL-2026-001",
        "year": 2026,
        "gross_obligation_tco2e": round(gross_obligation, 6),
        "free_allocation_pct": free_allocation_pct,
        "free_allocation_tco2e": round(free_allocation, 6),
        "carbon_price_deduction_tco2e": carbon_price_deduction,
        "net_obligation_tco2e": round(net_obligation, 6),
        "ets_price_eur_per_tco2e": ets_price_eur,
        "estimated_cost_eur": round(net_obligation * ets_price_eur, 2),
        "certificates_required": int(round(net_obligation, 0)),
        "surrender_deadline": "2026-05-31",
        "status": "calculated",
        "provenance_hash": _compute_hash({
            "obligation_id": "OBL-2026-001",
            "net": round(net_obligation, 6),
        }),
    }


# ---------------------------------------------------------------------------
# Supplier fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_suppliers() -> List[Dict[str, Any]]:
    """Create 3 supplier profiles with installations."""
    return [
        {
            "supplier_id": "SUP-TR-001",
            "company_name": "Eregli Demir ve Celik",
            "country": "TR",
            "eori_number": "TR987654321098765",
            "sector": "steel",
            "contact_email": "cbam@eregli-steel.com.tr",
            "status": "active",
            "registration_date": "2025-10-01",
            "installations": [
                {
                    "installation_id": "INST-TR-001",
                    "name": "Eregli Blast Furnace Complex",
                    "location": {"lat": 41.2828, "lon": 31.4164},
                    "capacity_tonnes_per_year": 5000000,
                    "un_locode": "TR ERE",
                    "emission_permit_id": "TR-ETS-001",
                },
            ],
            "quality_score": 85.0,
            "last_submission_date": "2026-03-01",
        },
        {
            "supplier_id": "SUP-CN-001",
            "company_name": "Baowu Steel Group",
            "country": "CN",
            "eori_number": None,
            "sector": "steel",
            "contact_email": "cbam@baowu.cn",
            "status": "active",
            "registration_date": "2025-11-15",
            "installations": [
                {
                    "installation_id": "INST-CN-001",
                    "name": "Shanghai Baoshan Works",
                    "location": {"lat": 31.3983, "lon": 121.4847},
                    "capacity_tonnes_per_year": 10000000,
                    "un_locode": "CN SHA",
                    "emission_permit_id": None,
                },
                {
                    "installation_id": "INST-CN-002",
                    "name": "Wuhan Iron and Steel",
                    "location": {"lat": 30.5928, "lon": 114.3052},
                    "capacity_tonnes_per_year": 8000000,
                    "un_locode": "CN WUH",
                    "emission_permit_id": None,
                },
            ],
            "quality_score": 62.0,
            "last_submission_date": "2026-02-15",
        },
        {
            "supplier_id": "SUP-IN-001",
            "company_name": "Hindalco Industries",
            "country": "IN",
            "eori_number": None,
            "sector": "aluminium",
            "contact_email": "cbam@hindalco.in",
            "status": "active",
            "registration_date": "2025-12-01",
            "installations": [
                {
                    "installation_id": "INST-IN-001",
                    "name": "Hirakud Smelter",
                    "location": {"lat": 21.5216, "lon": 83.8785},
                    "capacity_tonnes_per_year": 350000,
                    "un_locode": "IN HRK",
                    "emission_permit_id": None,
                },
            ],
            "quality_score": 72.0,
            "last_submission_date": "2026-02-28",
        },
    ]


# ---------------------------------------------------------------------------
# Emission Submission fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_emission_submissions() -> List[Dict[str, Any]]:
    """Create 5 EmissionSubmission fixtures."""
    return [
        {
            "submission_id": "SUB-001",
            "supplier_id": "SUP-TR-001",
            "installation_id": "INST-TR-001",
            "reporting_period": "Q1-2026",
            "goods_category": "steel",
            "cn_code": "7207 11 14",
            "weight_tonnes": 500.0,
            "specific_emission_tco2e_per_tonne": 1.85,
            "total_emissions_tco2e": 925.0,
            "methodology": "actual",
            "verification_status": "verified",
            "submitted_at": "2026-03-15T10:00:00Z",
            "review_status": "accepted",
            "quality_score": 92.0,
        },
        {
            "submission_id": "SUB-002",
            "supplier_id": "SUP-CN-001",
            "installation_id": "INST-CN-001",
            "reporting_period": "Q1-2026",
            "goods_category": "steel",
            "cn_code": "7208 51 20",
            "weight_tonnes": 300.0,
            "specific_emission_tco2e_per_tonne": 2.30,
            "total_emissions_tco2e": 690.0,
            "methodology": "default_values",
            "verification_status": "unverified",
            "submitted_at": "2026-03-18T14:30:00Z",
            "review_status": "pending",
            "quality_score": 65.0,
        },
        {
            "submission_id": "SUB-003",
            "supplier_id": "SUP-IN-001",
            "installation_id": "INST-IN-001",
            "reporting_period": "Q1-2026",
            "goods_category": "aluminium",
            "cn_code": "7601 10 00",
            "weight_tonnes": 200.0,
            "specific_emission_tco2e_per_tonne": 8.50,
            "total_emissions_tco2e": 1700.0,
            "methodology": "actual",
            "verification_status": "verified",
            "submitted_at": "2026-03-20T09:15:00Z",
            "review_status": "accepted",
            "quality_score": 88.0,
        },
        {
            "submission_id": "SUB-004",
            "supplier_id": "SUP-TR-002",
            "installation_id": "INST-TR-002",
            "reporting_period": "Q1-2026",
            "goods_category": "cement",
            "cn_code": "2523 29 00",
            "weight_tonnes": 1000.0,
            "specific_emission_tco2e_per_tonne": 0.65,
            "total_emissions_tco2e": 650.0,
            "methodology": "actual",
            "verification_status": "verified",
            "submitted_at": "2026-03-22T11:00:00Z",
            "review_status": "accepted",
            "quality_score": 90.0,
        },
        {
            "submission_id": "SUB-005",
            "supplier_id": "SUP-CN-002",
            "installation_id": "INST-CN-002",
            "reporting_period": "Q1-2026",
            "goods_category": "aluminium",
            "cn_code": "7601 20 80",
            "weight_tonnes": 120.0,
            "specific_emission_tco2e_per_tonne": 9.20,
            "total_emissions_tco2e": 1104.0,
            "methodology": "default_values",
            "verification_status": "unverified",
            "submitted_at": "2026-03-25T16:45:00Z",
            "review_status": "rejected",
            "quality_score": 45.0,
        },
    ]


# ---------------------------------------------------------------------------
# CN codes fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cn_codes() -> Dict[str, List[Dict[str, str]]]:
    """Dict of 50+ CN codes organized by category."""
    return {
        "cement": [
            {"code": "2523 10 00", "desc": "Cement clinkers"},
            {"code": "2523 21 00", "desc": "White Portland cement"},
            {"code": "2523 29 00", "desc": "Other Portland cement"},
            {"code": "2523 30 00", "desc": "Aluminous cement"},
            {"code": "2523 90 00", "desc": "Other hydraulic cements"},
        ],
        "steel": [
            {"code": "7206 10 00", "desc": "Ingots of iron"},
            {"code": "7207 11 14", "desc": "Semi-finished products, C<0.25%"},
            {"code": "7207 12 10", "desc": "Semi-finished products, C>=0.25%"},
            {"code": "7207 19 80", "desc": "Other semi-finished, rectangular"},
            {"code": "7207 20 80", "desc": "Stainless steel semi-finished"},
            {"code": "7208 10 00", "desc": "Flat-rolled, coils, hot-rolled"},
            {"code": "7208 25 00", "desc": "Flat-rolled, hot-rolled, w>=600mm"},
            {"code": "7208 26 00", "desc": "Flat-rolled, hot-rolled, pickled"},
            {"code": "7208 27 00", "desc": "Flat-rolled, hot-rolled, t<3mm"},
            {"code": "7208 36 00", "desc": "Flat-rolled, in coils, hot-rolled"},
            {"code": "7208 37 00", "desc": "Flat-rolled, not in coils"},
            {"code": "7208 38 00", "desc": "Flat-rolled, not in coils, t<3mm"},
            {"code": "7208 39 00", "desc": "Flat-rolled, not in coils, t<3mm o"},
            {"code": "7208 40 00", "desc": "Flat-rolled, not in coils, patterns"},
            {"code": "7208 51 20", "desc": "Flat-rolled, not in coils, t>10mm"},
            {"code": "7208 52 20", "desc": "Flat-rolled, 4.75mm<t<10mm"},
            {"code": "7209 15 00", "desc": "Cold-rolled, w>=600mm, t>=3mm"},
            {"code": "7209 16 90", "desc": "Cold-rolled, 1mm<t<3mm"},
            {"code": "7209 17 90", "desc": "Cold-rolled, 0.5mm<t<1mm"},
            {"code": "7209 18 91", "desc": "Cold-rolled, t<0.5mm"},
            {"code": "7210 11 00", "desc": "Flat-rolled, plated with tin"},
            {"code": "7211 13 00", "desc": "Flat-rolled, hot-rolled, w<600mm"},
            {"code": "7211 14 00", "desc": "Flat-rolled, hot-rolled, t>=4.75mm"},
            {"code": "7213 10 00", "desc": "Bars and rods, hot-rolled"},
            {"code": "7214 20 00", "desc": "Bars with indentations"},
            {"code": "7216 10 00", "desc": "U, I or H sections"},
        ],
        "aluminium": [
            {"code": "7601 10 00", "desc": "Unwrought aluminium, not alloyed"},
            {"code": "7601 20 20", "desc": "Aluminium alloys, primary"},
            {"code": "7601 20 80", "desc": "Aluminium alloys, secondary"},
            {"code": "7602 00 11", "desc": "Aluminium waste and scrap, turnings"},
            {"code": "7603 10 00", "desc": "Aluminium powders, non-lamellar"},
            {"code": "7604 10 10", "desc": "Aluminium bars and rods, not alloyed"},
            {"code": "7604 21 00", "desc": "Aluminium alloy hollow profiles"},
            {"code": "7604 29 10", "desc": "Aluminium alloy bars and rods"},
            {"code": "7605 11 00", "desc": "Aluminium wire, not alloyed"},
            {"code": "7605 21 00", "desc": "Aluminium alloy wire"},
            {"code": "7606 11 10", "desc": "Aluminium plates, not alloyed"},
            {"code": "7606 12 20", "desc": "Aluminium alloy plates"},
        ],
        "fertilizers": [
            {"code": "2808 00 00", "desc": "Nitric acid"},
            {"code": "2814 10 00", "desc": "Anhydrous ammonia"},
            {"code": "2814 20 00", "desc": "Ammonia in aqueous solution"},
            {"code": "3102 10 10", "desc": "Urea"},
            {"code": "3102 30 10", "desc": "Ammonium nitrate"},
            {"code": "3105 20 10", "desc": "NP fertilizers"},
        ],
        "electricity": [
            {"code": "2716 00 00", "desc": "Electrical energy"},
        ],
        "hydrogen": [
            {"code": "2804 10 00", "desc": "Hydrogen"},
        ],
    }


# ---------------------------------------------------------------------------
# ETS price fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ets_prices() -> List[Dict[str, Any]]:
    """Historical ETS prices (60-100 EUR range)."""
    prices = []
    base_date = datetime(2025, 1, 1, tzinfo=timezone.utc)
    base_price = 70.0
    for i in range(52):  # 52 weeks
        week_date = base_date + timedelta(weeks=i)
        variation = (i * 7 + 13) % 40 - 10  # -10 to +30
        price = round(base_price + variation * 0.75, 2)
        price = max(60.0, min(100.0, price))
        prices.append({
            "date": week_date.strftime("%Y-%m-%d"),
            "price_eur": price,
            "source": "EU_ETS_AUCTION",
            "volume_traded": 50000 + i * 1000,
        })
    return prices


# ---------------------------------------------------------------------------
# Compliance rules fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_compliance_rules() -> List[Dict[str, Any]]:
    """50+ CBAM compliance rules."""
    rules = []
    rule_templates = [
        ("CBAM-REG-{:03d}", "regulatory", "Regulation (EU) 2023/956 Art. {}"),
        ("CBAM-DATA-{:03d}", "data_quality", "Data completeness for {}"),
        ("CBAM-CALC-{:03d}", "calculation", "Emission calculation method {}"),
        ("CBAM-RPT-{:03d}", "reporting", "Quarterly report field {}"),
        ("CBAM-CERT-{:03d}", "certificate", "Certificate management rule {}"),
    ]
    idx = 1
    for prefix_fmt, category, desc_fmt in rule_templates:
        for j in range(1, 11):
            rules.append({
                "rule_id": prefix_fmt.format(idx),
                "category": category,
                "description": desc_fmt.format(j),
                "severity": "error" if j <= 5 else "warning",
                "applicable_goods": ["all"] if j <= 3 else ["steel", "aluminium"],
                "active": True,
            })
            idx += 1
    return rules


# ---------------------------------------------------------------------------
# Verifier fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_verifier() -> Dict[str, Any]:
    """AccreditedVerifier fixture."""
    return {
        "verifier_id": "VER-DAkkS-001",
        "company_name": "TUV Rheinland Energy GmbH",
        "accreditation_body": "DAkkS",
        "accreditation_number": "D-VS-21098-01-00",
        "accreditation_valid_until": "2027-12-31",
        "lead_verifier": "Dr. Klaus Weber",
        "contact_email": "cbam-verification@tuv.com",
        "member_state": "DE",
        "scopes": ["cement", "steel", "aluminium", "fertilizers"],
        "status": "active",
    }


# ---------------------------------------------------------------------------
# Import CSV data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_import_csv_data() -> str:
    """CSV-style import data (20 rows)."""
    header = "import_id,cn_code,goods_category,origin_country,weight_tonnes," \
             "value_eur,supplier_id,import_date,customs_declaration\n"
    rows = []
    cn_codes = [
        ("7207 11 14", "steel", "TR"), ("7208 51 20", "steel", "CN"),
        ("7601 10 00", "aluminium", "IN"), ("2523 29 00", "cement", "TR"),
        ("7207 20 80", "steel", "IN"), ("7601 20 80", "aluminium", "CN"),
        ("7208 10 00", "steel", "TR"), ("2523 21 00", "cement", "CN"),
        ("7604 10 10", "aluminium", "IN"), ("7209 15 00", "steel", "CN"),
        ("7206 10 00", "steel", "TR"), ("7605 11 00", "aluminium", "IN"),
        ("2523 10 00", "cement", "TR"), ("7208 25 00", "steel", "CN"),
        ("7601 20 20", "aluminium", "IN"), ("7208 26 00", "steel", "TR"),
        ("7604 21 00", "aluminium", "CN"), ("2523 30 00", "cement", "IN"),
        ("7213 10 00", "steel", "TR"), ("7606 11 10", "aluminium", "CN"),
    ]
    for i, (cn, cat, country) in enumerate(cn_codes, 1):
        weight = round(100 + i * 50, 1)
        value = round(weight * (150 + i * 10), 2)
        rows.append(
            f"IMP-{i:03d},{cn},{cat},{country},{weight},{value},"
            f"SUP-{country}-{(i % 3) + 1:03d},2026-{(i % 3) + 1:02d}-{(i % 28) + 1:02d},"
            f"CD-2026-{i:04d}"
        )
    return header + "\n".join(rows)


# ---------------------------------------------------------------------------
# Stub classes for external dependencies
# ---------------------------------------------------------------------------

class StubCBAMApp:
    """Stub for GL-CBAM-APP integration."""

    def __init__(self):
        self.engines: Dict[str, Any] = {}
        self.cn_codes: Dict[str, List[str]] = {}
        self.emission_factors: Dict[str, float] = {}
        self.rules: List[Dict[str, Any]] = []
        self._healthy = True

    def get_engines(self) -> Dict[str, str]:
        return {
            "cbam_calculation": "CBAMCalculationEngine v1.0",
            "certificate": "CertificateEngine v1.0",
            "quarterly_reporting": "QuarterlyReportingEngine v1.0",
            "supplier_management": "SupplierManagementEngine v1.0",
            "deminimis": "DeMinimisEngine v1.0",
            "verification": "VerificationEngine v1.0",
            "policy_compliance": "PolicyComplianceEngine v1.0",
        }

    def get_cn_codes(self, category: str = None) -> Dict[str, List[str]]:
        all_codes = {
            "steel": ["7207 11 14", "7208 51 20", "7209 15 00"],
            "aluminium": ["7601 10 00", "7601 20 80"],
            "cement": ["2523 21 00", "2523 29 00"],
        }
        if category:
            return {category: all_codes.get(category, [])}
        return all_codes

    def get_emission_factors(self, category: str) -> Dict[str, float]:
        factors = {
            "steel": {"default": 2.30, "bof": 1.85, "eaf": 0.45},
            "aluminium": {"default": 8.50, "primary": 9.20, "secondary": 1.50},
            "cement": {"default": 0.65, "clinker": 0.85, "blended": 0.50},
        }
        return factors.get(category, {})

    def get_rules(self) -> List[Dict[str, Any]]:
        return [
            {"rule_id": "R-001", "description": "CN code valid", "active": True},
            {"rule_id": "R-002", "description": "Weight positive", "active": True},
            {"rule_id": "R-003", "description": "EF in range", "active": True},
        ]

    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "healthy" if self._healthy else "unhealthy",
            "engines_loaded": 7,
            "timestamp": _utcnow().isoformat(),
        }


class StubCertificateEngine:
    """Stub for CBAM Certificate Engine."""

    FREE_ALLOCATION_SCHEDULE = {
        2026: 0.975, 2027: 0.925, 2028: 0.850, 2029: 0.775,
        2030: 0.515, 2031: 0.390, 2032: 0.265, 2033: 0.140,
        2034: 0.000,
    }

    def __init__(self):
        self.certificates: Dict[str, Dict[str, Any]] = {}
        self.holdings: List[Dict[str, Any]] = []

    def calculate_obligation(
        self,
        total_emissions_tco2e: float,
        year: int,
        carbon_price_deduction: float = 0.0,
    ) -> Dict[str, Any]:
        free_alloc_pct = self.FREE_ALLOCATION_SCHEDULE.get(year, 0.0)
        gross = total_emissions_tco2e
        free_alloc = gross * free_alloc_pct
        net = max(0.0, gross - free_alloc - carbon_price_deduction)
        return {
            "gross_obligation_tco2e": round(gross, 6),
            "free_allocation_pct": free_alloc_pct,
            "free_allocation_tco2e": round(free_alloc, 6),
            "carbon_price_deduction_tco2e": round(carbon_price_deduction, 6),
            "net_obligation_tco2e": round(net, 6),
            "certificates_required": int(round(net, 0)),
            "provenance_hash": _compute_hash({
                "gross": gross, "year": year, "deduction": carbon_price_deduction,
            }),
        }

    def estimate_cost(
        self,
        net_obligation_tco2e: float,
        ets_price_eur: float,
    ) -> Dict[str, Any]:
        cost = round(net_obligation_tco2e * ets_price_eur, 2)
        return {
            "net_obligation_tco2e": round(net_obligation_tco2e, 6),
            "ets_price_eur": ets_price_eur,
            "estimated_cost_eur": cost,
            "currency": "EUR",
        }

    def check_quarterly_holding(
        self,
        certificates_held: int,
        net_obligation_tco2e: float,
    ) -> Dict[str, Any]:
        required_pct = 0.80  # 80% quarterly holding requirement
        required = int(round(net_obligation_tco2e * required_pct, 0))
        compliant = certificates_held >= required
        return {
            "certificates_held": certificates_held,
            "required_holding": required,
            "required_pct": required_pct,
            "compliant": compliant,
            "shortfall": max(0, required - certificates_held),
        }


class StubQuarterlyEngine:
    """Stub for Quarterly Reporting Engine."""

    def __init__(self):
        self.reports: Dict[str, Dict[str, Any]] = {}

    def detect_period(self, reference_date: str) -> Dict[str, Any]:
        dt = datetime.strptime(reference_date, "%Y-%m-%d")
        quarter = (dt.month - 1) // 3 + 1
        q_start_month = (quarter - 1) * 3 + 1
        q_end_month = quarter * 3
        return {
            "quarter": f"Q{quarter}",
            "year": dt.year,
            "start_date": f"{dt.year}-{q_start_month:02d}-01",
            "end_date": f"{dt.year}-{q_end_month:02d}-{[31,30,31,31][quarter-1] if quarter < 4 else 31}",
            "deadline": f"{dt.year}-{q_end_month + 1:02d}-30"
            if q_end_month < 12
            else f"{dt.year + 1}-01-31",
        }

    def assemble_report(
        self,
        importer: Dict[str, Any],
        emission_results: List[Dict[str, Any]],
        period: Dict[str, Any],
    ) -> Dict[str, Any]:
        total_emissions = sum(r["total_emissions_tco2e"] for r in emission_results)
        report_id = f"QR-{period['year']}-{period['quarter']}-{_new_uuid()[:8]}"
        return {
            "report_id": report_id,
            "importer": importer,
            "period": period,
            "total_imports": len(emission_results),
            "total_emissions_tco2e": round(total_emissions, 6),
            "status": "assembled",
            "version": 1,
            "provenance_hash": _compute_hash({
                "report_id": report_id,
                "emissions": round(total_emissions, 6),
            }),
        }

    def generate_xml(self, report: Dict[str, Any]) -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<CBAMQuarterlyReport version="1.0.0">\n'
            f'  <ReportId>{report["report_id"]}</ReportId>\n'
            f'  <Quarter>{report["period"]["quarter"]}</Quarter>\n'
            f'  <Year>{report["period"]["year"]}</Year>\n'
            f'  <TotalEmissions unit="tCO2e">'
            f'{report["total_emissions_tco2e"]}</TotalEmissions>\n'
            f'  <TotalImports>{report["total_imports"]}</TotalImports>\n'
            f'  <ImporterName>{report["importer"]["company_name"]}</ImporterName>\n'
            f'  <ImporterEORI>{report["importer"]["eori_number"]}</ImporterEORI>\n'
            f'  <ProvenanceHash>{report["provenance_hash"]}</ProvenanceHash>\n'
            f'</CBAMQuarterlyReport>'
        )

    def validate_report(self, report: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        warnings = []
        if not report.get("importer", {}).get("eori_number"):
            errors.append("Missing EORI number")
        if report.get("total_imports", 0) == 0:
            errors.append("No imports in report")
        if report.get("total_emissions_tco2e", 0) <= 0:
            errors.append("Total emissions must be positive")
        if not report.get("period", {}).get("quarter"):
            errors.append("Missing reporting quarter")
        if report.get("status") == "draft":
            warnings.append("Report is still in draft")
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "checks_total": 4,
            "checks_passed": 4 - len(errors),
        }

    def create_amendment(
        self, report_id: str, reason: str, version: int,
    ) -> Dict[str, Any]:
        return {
            "report_id": report_id,
            "amendment_version": version + 1,
            "reason": reason,
            "status": "amended",
            "created_at": _utcnow().isoformat(),
        }


class StubSupplierPortal:
    """Stub for Supplier Management Portal."""

    def __init__(self):
        self.suppliers: Dict[str, Dict[str, Any]] = {}
        self.submissions: List[Dict[str, Any]] = []

    def register_supplier(self, data: Dict[str, Any]) -> Dict[str, Any]:
        supplier_id = data.get("supplier_id", f"SUP-{_new_uuid()[:8]}")
        supplier = {
            "supplier_id": supplier_id,
            "company_name": data.get("company_name", "Unknown"),
            "country": data.get("country", "XX"),
            "status": "active",
            "registered_at": _utcnow().isoformat(),
            "quality_score": 50.0,
        }
        self.suppliers[supplier_id] = supplier
        return supplier

    def add_installation(
        self, supplier_id: str, installation: Dict[str, Any],
    ) -> Dict[str, Any]:
        inst_id = installation.get(
            "installation_id", f"INST-{_new_uuid()[:8]}"
        )
        return {
            "installation_id": inst_id,
            "supplier_id": supplier_id,
            "name": installation.get("name", "Unnamed"),
            "status": "registered",
        }

    def submit_emission_data(
        self, supplier_id: str, data: Dict[str, Any],
    ) -> Dict[str, Any]:
        sub_id = f"SUB-{_new_uuid()[:8]}"
        submission = {
            "submission_id": sub_id,
            "supplier_id": supplier_id,
            "status": "submitted",
            "submitted_at": _utcnow().isoformat(),
            **data,
        }
        self.submissions.append(submission)
        return submission

    def review_submission(
        self, submission_id: str, decision: str, notes: str = "",
    ) -> Dict[str, Any]:
        return {
            "submission_id": submission_id,
            "decision": decision,
            "notes": notes,
            "reviewed_at": _utcnow().isoformat(),
        }

    def get_quality_score(self, supplier_id: str) -> Dict[str, Any]:
        supplier = self.suppliers.get(supplier_id, {})
        return {
            "supplier_id": supplier_id,
            "quality_score": supplier.get("quality_score", 50.0),
            "rating": "excellent" if supplier.get("quality_score", 0) >= 85
                      else "good" if supplier.get("quality_score", 0) >= 70
                      else "acceptable" if supplier.get("quality_score", 0) >= 50
                      else "poor",
        }


class StubETSFeed:
    """Stub for EU ETS price feed."""

    def __init__(self):
        self._current_price = 78.50
        self._history = []
        for i in range(52):
            self._history.append({
                "date": (datetime(2025, 1, 1) + timedelta(weeks=i)).strftime("%Y-%m-%d"),
                "price_eur": round(70.0 + (i % 20) * 1.5, 2),
            })

    def current_price(self) -> Dict[str, Any]:
        return {
            "price_eur": self._current_price,
            "currency": "EUR",
            "source": "EU_ETS_AUCTION",
            "timestamp": _utcnow().isoformat(),
        }

    def price_history(
        self, start_date: str = None, end_date: str = None,
    ) -> List[Dict[str, Any]]:
        return list(self._history)

    def price_projection(self, months: int = 12) -> List[Dict[str, Any]]:
        projections = []
        for i in range(months):
            trend = 1.0 + 0.005 * i
            projections.append({
                "month": i + 1,
                "projected_price_eur": round(self._current_price * trend, 2),
                "lower_bound": round(self._current_price * trend * 0.9, 2),
                "upper_bound": round(self._current_price * trend * 1.1, 2),
            })
        return projections

    def price_comparison(self) -> Dict[str, Any]:
        return {
            "eu_ets": self._current_price,
            "uk_ets": round(self._current_price * 0.85, 2),
            "china_ets": round(self._current_price * 0.12, 2),
            "korea_ets": round(self._current_price * 0.35, 2),
            "currency": "EUR",
        }

    def exchange_rate(self, from_currency: str, to_currency: str) -> float:
        rates = {
            ("EUR", "USD"): 1.08,
            ("EUR", "GBP"): 0.86,
            ("EUR", "TRY"): 35.20,
            ("EUR", "CNY"): 7.85,
            ("EUR", "INR"): 90.50,
            ("USD", "EUR"): 0.926,
        }
        return rates.get((from_currency, to_currency), 1.0)


class StubCustoms:
    """Stub for customs/CN code lookup service."""

    CN_CODES = {
        "7207 11 14": {"category": "steel", "desc": "Semi-finished iron"},
        "7208 51 20": {"category": "steel", "desc": "Flat-rolled, hot-rolled"},
        "7601 10 00": {"category": "aluminium", "desc": "Unwrought aluminium"},
        "7601 20 80": {"category": "aluminium", "desc": "Aluminium alloys"},
        "2523 29 00": {"category": "cement", "desc": "Portland cement"},
        "2523 21 00": {"category": "cement", "desc": "White Portland cement"},
        "7604 10 10": {"category": "aluminium", "desc": "Aluminium bars"},
        "7209 15 00": {"category": "steel", "desc": "Cold-rolled flat"},
        "2716 00 00": {"category": "electricity", "desc": "Electrical energy"},
        "2804 10 00": {"category": "hydrogen", "desc": "Hydrogen"},
    }

    def lookup_cn_code(self, code: str) -> Optional[Dict[str, Any]]:
        return self.CN_CODES.get(code)

    def validate_cn_code(self, code: str) -> Dict[str, Any]:
        valid = bool(re.match(r"^\d{4}\s\d{2}\s\d{2}$", code))
        cbam_covered = code in self.CN_CODES
        return {
            "code": code,
            "format_valid": valid,
            "cbam_covered": cbam_covered,
            "category": self.CN_CODES.get(code, {}).get("category"),
        }

    def validate_eori(self, eori: str) -> Dict[str, Any]:
        valid = bool(re.match(r"^[A-Z]{2}\d{13,17}$", eori))
        return {
            "eori": eori,
            "valid": valid,
            "member_state": eori[:2] if valid else None,
        }

    def category_lookup(self, category: str) -> List[Dict[str, Any]]:
        return [
            {"code": code, **info}
            for code, info in self.CN_CODES.items()
            if info["category"] == category
        ]

    def all_cbam_codes(self) -> List[str]:
        return sorted(self.CN_CODES.keys())


# ---------------------------------------------------------------------------
# Template render helper
# ---------------------------------------------------------------------------

def render_template_stub(
    template_id: str,
    data: Dict[str, Any],
    output_format: str = "markdown",
) -> Dict[str, Any]:
    """Stub template renderer for testing."""
    title = template_id.replace("_", " ").title()
    provenance_hash = _compute_hash({"template_id": template_id, "data": data})

    if output_format == "markdown":
        content = f"# {title}\n\n"
        for key, val in data.items():
            content += f"- **{key}**: {val}\n"
        content += f"\n---\nProvenance: {provenance_hash}\n"
    elif output_format == "html":
        content = f"<html><head><title>{title}</title></head><body>"
        content += f"<h1>{title}</h1><dl>"
        for key, val in data.items():
            content += f"<dt>{key}</dt><dd>{val}</dd>"
        content += f"</dl><footer>Provenance: {provenance_hash}</footer></body></html>"
    elif output_format == "json":
        content = json.dumps({
            "template_id": template_id,
            "title": title,
            "data": data,
            "provenance_hash": provenance_hash,
            "generated_at": _utcnow().isoformat(),
        }, indent=2, default=str)
    else:
        content = f"{title}: {json.dumps(data, default=str)}"

    return {
        "template_id": template_id,
        "format": output_format,
        "content": content,
        "provenance_hash": provenance_hash,
        "generated_at": _utcnow().isoformat(),
    }


@pytest.fixture
def template_renderer():
    """Return the template render stub function."""
    return render_template_stub


# ---------------------------------------------------------------------------
# Mock fixtures for stub classes
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cbam_app() -> StubCBAMApp:
    """Return a StubCBAMApp instance."""
    return StubCBAMApp()


@pytest.fixture
def mock_certificate_engine() -> StubCertificateEngine:
    """Return a StubCertificateEngine instance."""
    return StubCertificateEngine()


@pytest.fixture
def mock_quarterly_engine() -> StubQuarterlyEngine:
    """Return a StubQuarterlyEngine instance."""
    return StubQuarterlyEngine()


@pytest.fixture
def mock_supplier_portal() -> StubSupplierPortal:
    """Return a StubSupplierPortal instance."""
    return StubSupplierPortal()


@pytest.fixture
def mock_ets_feed() -> StubETSFeed:
    """Return a StubETSFeed instance."""
    return StubETSFeed()


@pytest.fixture
def mock_customs() -> StubCustoms:
    """Return a StubCustoms instance."""
    return StubCustoms()


# ---------------------------------------------------------------------------
# CBAM ENGINE IDS
# ---------------------------------------------------------------------------

CBAM_ENGINE_IDS = [
    "cbam_calculation",
    "certificate",
    "quarterly_reporting",
    "supplier_management",
    "deminimis",
    "verification",
    "policy_compliance",
]


# ---------------------------------------------------------------------------
# CBAM WORKFLOW IDS
# ---------------------------------------------------------------------------

CBAM_WORKFLOW_IDS = [
    "quarterly_reporting",
    "annual_declaration",
    "supplier_onboarding",
    "certificate_management",
    "verification_cycle",
    "deminimis_assessment",
    "data_collection",
]


# ---------------------------------------------------------------------------
# CBAM TEMPLATE IDS
# ---------------------------------------------------------------------------

CBAM_TEMPLATE_IDS = [
    "quarterly_report",
    "annual_declaration",
    "certificate_summary",
    "supplier_data_request",
    "verification_statement",
    "deminimis_assessment",
    "compliance_dashboard",
    "emission_calculation_detail",
]


@pytest.fixture
def cbam_engine_ids() -> List[str]:
    """Return the 7 CBAM engine IDs."""
    return list(CBAM_ENGINE_IDS)


@pytest.fixture
def cbam_workflow_ids() -> List[str]:
    """Return the 7 CBAM workflow IDs."""
    return list(CBAM_WORKFLOW_IDS)


@pytest.fixture
def cbam_template_ids() -> List[str]:
    """Return the 8 CBAM template IDs."""
    return list(CBAM_TEMPLATE_IDS)
