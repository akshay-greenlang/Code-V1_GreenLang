# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Shared Test Fixtures
=========================================================

Provides reusable pytest fixtures for all PACK-005 test modules including
certificate trading, precursor chain, multi-entity, registry API, advanced
analytics, customs automation, cross-regulation, and audit management.

All fixtures are self-contained with no external dependencies.
Every external service is mocked via stub classes in this module.

Author: GreenLang QA Team
Version: 1.0.0
"""

import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
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


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing provenance_hash field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash should be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash should be 64 hex chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Invalid hex chars in hash"


def assert_valid_uuid(value: str) -> None:
    """Verify value is a valid UUID format."""
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        raise AssertionError(f"Invalid UUID format: {value}")


def assert_decimal_precision(value: Any, places: int) -> None:
    """Verify Decimal or float has correct precision."""
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        actual_places = abs(exponent) if exponent < 0 else 0
        assert actual_places <= places, (
            f"Expected at most {places} decimal places, got {actual_places}"
        )
    elif isinstance(value, float):
        str_val = f"{value:.{places + 5}f}"
        decimal_part = str_val.split(".")[1] if "." in str_val else ""
        significant = decimal_part.rstrip("0")
        assert len(significant) <= places, (
            f"Expected at most {places} decimal places, got {len(significant)}"
        )


def generate_cn_codes(category: str, count: int) -> List[str]:
    """Generate test CN codes for a given goods category."""
    base_codes = {
        "steel": [
            "7207 11 14", "7208 51 20", "7209 15 00", "7208 10 00",
            "7207 20 80", "7208 25 00", "7208 26 00", "7210 11 00",
            "7211 13 00", "7213 10 00", "7214 20 00", "7216 10 00",
        ],
        "aluminium": [
            "7601 10 00", "7601 20 80", "7604 10 10", "7605 11 00",
            "7606 11 10", "7601 20 20", "7604 21 00", "7607 11 10",
        ],
        "cement": [
            "2523 10 00", "2523 21 00", "2523 29 00", "2523 30 00",
            "2523 90 00",
        ],
        "fertilizers": [
            "2808 00 00", "2814 10 00", "2814 20 00", "3102 10 10",
            "3102 30 10", "3105 20 10",
        ],
        "electricity": ["2716 00 00"],
        "hydrogen": ["2804 10 00"],
    }
    codes = base_codes.get(category, ["9999 99 99"])
    result = []
    for i in range(count):
        result.append(codes[i % len(codes)])
    return result


def generate_import_portfolio(
    entities: List[Dict[str, Any]], count: int
) -> List[Dict[str, Any]]:
    """Generate test import portfolio records spread across entities."""
    categories = ["steel", "aluminium", "cement", "fertilizers", "electricity", "hydrogen"]
    countries = ["TR", "CN", "IN", "UA", "BR", "ZA", "KR", "EG"]
    records = []
    for i in range(count):
        entity = entities[i % len(entities)]
        category = categories[i % len(categories)]
        cn_codes = generate_cn_codes(category, 1)
        records.append({
            "import_id": f"IMP-{i + 1:04d}",
            "entity_id": entity.get("entity_id", f"ENT-{(i % len(entities)) + 1:03d}"),
            "cn_code": cn_codes[0],
            "goods_category": category,
            "origin_country": countries[i % len(countries)],
            "weight_tonnes": round(100 + i * 25, 1),
            "value_eur": round((100 + i * 25) * (120 + i * 5), 2),
            "specific_emission_tco2e_per_tonne": round(0.5 + (i % 10) * 0.8, 4),
            "import_date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "supplier_id": f"SUP-{countries[i % len(countries)]}-{(i % 3) + 1:03d}",
        })
    return records


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
def demo_portfolio_csv_path() -> Path:
    """Return path to demo portfolio CSV."""
    return DEMO_DIR / "demo_portfolio.csv"


# ---------------------------------------------------------------------------
# CBAMCompleteConfig fixture (dict-based, no external deps)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a full CBAMCompleteConfig dict with all sub-configs populated.

    Extends PACK-004 with trading, multi-entity, registry, analytics,
    customs, cross-regulation, and audit management sub-configs.
    """
    return {
        "metadata": {
            "name": "cbam-complete",
            "version": "2.0.0",
            "display_name": "CBAM Complete Pack",
            "description": "Full CBAM compliance with trading, multi-entity, and registry integration",
            "category": "eu-compliance",
            "tier": "enterprise",
            "extends": "PACK-004-cbam-readiness",
            "author": "GreenLang Platform Team",
            "license": "Proprietary",
            "min_platform_version": "2.5.0",
            "release_date": "2026-03-14",
            "support_tier": "enterprise",
            "tags": [
                "cbam", "carbon-border-adjustment", "eu-compliance",
                "certificate-trading", "multi-entity", "registry-api",
                "customs-automation", "cross-regulation", "audit-management",
            ],
            "compliance_references": [
                {
                    "id": "CBAM",
                    "name": "Carbon Border Adjustment Mechanism",
                    "regulation": "Regulation (EU) 2023/956",
                    "effective_date": "2023-10-01",
                },
                {
                    "id": "CBAM-IR",
                    "name": "CBAM Implementing Regulation",
                    "regulation": "Implementing Regulation (EU) 2023/1773",
                    "effective_date": "2023-10-01",
                },
                {
                    "id": "EU-ETS",
                    "name": "EU Emissions Trading System",
                    "regulation": "Directive 2003/87/EC (amended)",
                    "effective_date": "2005-01-01",
                },
            ],
        },
        "trading": {
            "buying_strategies": ["market", "limit", "scheduled", "dca", "custom"],
            "valuation_methods": ["FIFO", "WAC", "MTM"],
            "resale_limit_fraction": Decimal("0.3333"),
            "resale_window_months": 12,
            "holding_compliance_threshold_pct": 50.0,
            "expiry_alert_days": [30, 60, 90],
            "default_strategy": "market",
            "default_valuation": "FIFO",
        },
        "entity_group": {
            "roles": ["parent", "subsidiary", "joint_venture", "branch"],
            "declarant_statuses": [
                "active", "pending", "suspended", "revoked", "expired", "draft",
            ],
            "consolidation_methods": ["volume", "revenue", "equal"],
            "max_entities": 50,
            "financial_guarantee_enabled": True,
        },
        "registry_api": {
            "base_url": "https://cbam-registry.ec.europa.eu/api/v1",
            "sandbox_url": "https://cbam-registry-sandbox.ec.europa.eu/api/v1",
            "mode": "sandbox",
            "retry_max": 3,
            "retry_backoff_seconds": 2,
            "poll_interval_seconds": 5,
            "poll_timeout_seconds": 120,
            "mock_mode": True,
        },
        "analytics": {
            "monte_carlo_iterations": 10000,
            "confidence_levels": [0.90, 0.95, 0.99],
            "carbon_price_models": ["gbm", "mean_reversion", "jump_diffusion"],
            "scenario_max_count": 10,
            "optimization_solver": "scipy_minimize",
        },
        "customs": {
            "cn_version": "2026",
            "anti_circumvention_rules": [
                "origin_change", "cn_reclassification",
                "scrap_ratio_anomaly", "restructuring", "minor_processing",
            ],
            "aeo_validation_enabled": True,
            "sad_parsing_enabled": True,
            "downstream_monitoring": True,
        },
        "cross_regulation": {
            "targets": ["csrd", "cdp", "sbti", "taxonomy", "ets", "eudr"],
            "data_reuse_enabled": True,
            "consistency_check_enabled": True,
            "third_country_carbon_pricing": {
                c: round(5.0 + i * 1.5, 2)
                for i, c in enumerate([
                    "TR", "CN", "IN", "UA", "BR", "ZA", "KR", "EG", "RU", "NO",
                    "IS", "GB", "JP", "CA", "AU", "NZ", "MX", "ID", "VN", "TH",
                    "PH", "MY", "SG", "CL", "AR", "CO", "PE", "TW", "PK", "BD",
                    "LK", "SA", "AE", "QA", "KW", "BH", "OM", "JO", "IL", "MA",
                    "TN", "DZ", "NG", "GH", "KE", "ET", "TZ", "UG", "MZ", "CI",
                ])
            },
        },
        "audit": {
            "evidence_retention_years": 10,
            "data_room_access_roles": ["auditor", "regulator", "compliance_officer"],
            "anomaly_detection_enabled": True,
            "penalty_rate_per_tco2e_eur": 100.0,
            "verifier_accreditation_required": True,
            "nca_correspondence_logging": True,
        },
        "cbam": {
            "importer": {
                "company_name": "EuroSteel Group GmbH",
                "eori_number": "DE123456789012345",
                "registration_id": "CBAM-DE-2026-00001",
                "contact_email": "cbam@eurosteel-group.de",
                "member_state": "DE",
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
            },
            "certificate_config": {
                "currency": "EUR",
                "price_source": "EU_ETS_AUCTION",
                "free_allocation_schedule": {
                    "2026": 0.975, "2027": 0.925, "2028": 0.850,
                    "2029": 0.775, "2030": 0.515, "2031": 0.390,
                    "2032": 0.265, "2033": 0.140, "2034": 0.000,
                },
                "surrender_deadline_months_after_year": 5,
            },
        },
        "precursor_chain": {
            "max_depth": 10,
            "allocation_methods": ["mass_based", "economic", "energy"],
            "default_allocation": "mass_based",
            "production_routes": {
                "steel": ["bf_bof", "eaf", "dri_eaf"],
                "aluminium": ["hall_heroult", "secondary_remelting"],
                "cement": ["dry_process", "wet_process"],
            },
        },
        "penalties": {
            "base_rate_per_tco2e_eur": 100.0,
            "late_surrender_multiplier": 1.5,
            "repeat_offense_multiplier": 3.0,
            "administrative_fine_min_eur": 10000.0,
            "administrative_fine_max_eur": 500000.0,
        },
    }


# ---------------------------------------------------------------------------
# Entity Group fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_entity_group() -> Dict[str, Any]:
    """Create EntityGroup with parent + 2 subsidiaries."""
    return {
        "group_id": "GRP-EUROSTEEL-001",
        "group_name": "EuroSteel Group",
        "parent": {
            "entity_id": "ENT-001",
            "legal_name": "EuroSteel Group GmbH",
            "eori_number": "DE123456789012345",
            "member_state": "DE",
            "role": "parent",
            "declarant_status": "active",
            "registration_id": "CBAM-DE-2026-00001",
        },
        "subsidiaries": [
            {
                "entity_id": "ENT-002",
                "legal_name": "EuroSteel France SAS",
                "eori_number": "FR987654321098765",
                "member_state": "FR",
                "role": "subsidiary",
                "declarant_status": "active",
                "registration_id": "CBAM-FR-2026-00015",
            },
            {
                "entity_id": "ENT-003",
                "legal_name": "EuroSteel Italia S.r.l.",
                "eori_number": "IT456789012345678",
                "member_state": "IT",
                "role": "subsidiary",
                "declarant_status": "pending",
                "registration_id": "CBAM-IT-2026-00042",
            },
        ],
        "consolidation_method": "volume",
        "financial_guarantee_eur": 500000.00,
        "created_at": "2026-01-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# Certificate Portfolio fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_certificate_portfolio() -> Dict[str, Any]:
    """Create CertificatePortfolio with 50 certificates in various states."""
    certificates = []
    statuses = ["active", "active", "active", "surrendered", "expired",
                "active", "active", "resold", "active", "cancelled"]
    base_date = datetime(2026, 1, 15, tzinfo=timezone.utc)

    for i in range(50):
        purchase_date = base_date + timedelta(days=i * 3)
        expiry_date = purchase_date + timedelta(days=365)
        status = statuses[i % len(statuses)]
        price = Decimal(str(round(70.0 + (i % 30) * 1.0, 2)))
        certificates.append({
            "certificate_id": f"CERT-2026-{i + 1:05d}",
            "purchase_date": purchase_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "price_eur": price,
            "quantity_tco2e": Decimal("1"),
            "status": status,
            "order_type": "market" if i % 3 == 0 else "limit",
            "entity_id": f"ENT-{(i % 3) + 1:03d}",
        })

    active_count = sum(1 for c in certificates if c["status"] == "active")
    total_value = sum(c["price_eur"] for c in certificates if c["status"] == "active")

    return {
        "portfolio_id": "PF-EUROSTEEL-2026",
        "entity_group_id": "GRP-EUROSTEEL-001",
        "certificates": certificates,
        "total_certificates": len(certificates),
        "active_certificates": active_count,
        "total_active_value_eur": total_value,
        "valuation_method": "FIFO",
        "as_of_date": _utcnow().strftime("%Y-%m-%d"),
    }


# ---------------------------------------------------------------------------
# Import Records fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_import_records() -> List[Dict[str, Any]]:
    """Create 20 import records across all 6 goods categories."""
    cn_code_map = {
        "steel": ["7207 11 14", "7208 51 20", "7209 15 00", "7208 10 00"],
        "aluminium": ["7601 10 00", "7601 20 80", "7604 10 10"],
        "cement": ["2523 29 00", "2523 21 00", "2523 10 00"],
        "fertilizers": ["2814 10 00", "3102 10 10", "3102 30 10"],
        "electricity": ["2716 00 00"],
        "hydrogen": ["2804 10 00"],
    }
    ef_map = {
        "steel": 1.95, "aluminium": 8.50, "cement": 0.68,
        "fertilizers": 2.10, "electricity": 0.23, "hydrogen": 10.00,
    }
    categories = list(cn_code_map.keys())
    countries = ["TR", "CN", "IN", "UA", "BR", "ZA", "KR", "EG"]
    records = []
    for i in range(20):
        cat = categories[i % len(categories)]
        codes = cn_code_map[cat]
        cn = codes[i % len(codes)]
        country = countries[i % len(countries)]
        weight = round(100 + i * 50, 1)
        ef = ef_map[cat]
        records.append({
            "import_id": f"IMP-{i + 1:03d}",
            "cn_code": cn,
            "goods_category": cat,
            "origin_country": country,
            "weight_tonnes": weight,
            "value_eur": round(weight * 150.0, 2),
            "specific_emission_tco2e_per_tonne": ef,
            "total_emissions_tco2e": round(weight * ef, 6),
            "import_date": f"2026-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "supplier_id": f"SUP-{country}-{(i % 3) + 1:03d}",
            "installation_id": f"INST-{country}-{(i % 3) + 1:03d}",
            "methodology": "actual" if i % 2 == 0 else "default_values",
        })
    return records


# ---------------------------------------------------------------------------
# Precursor Chain fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_precursor_chain() -> Dict[str, Any]:
    """Create Steel precursor chain: ore -> pig iron -> crude steel -> hot-rolled."""
    return {
        "chain_id": "PC-STEEL-001",
        "goods_category": "steel",
        "final_product": {
            "name": "Hot-rolled flat steel",
            "cn_code": "7208 51 20",
            "weight_tonnes": 1000.0,
        },
        "stages": [
            {
                "stage_id": "STG-001",
                "stage_name": "Iron Ore Mining",
                "product": "Iron ore (Fe 62%)",
                "input_weight_tonnes": 1600.0,
                "output_weight_tonnes": 1600.0,
                "emission_tco2e_per_tonne": 0.04,
                "total_emission_tco2e": 64.0,
                "allocation_method": "mass_based",
                "allocation_factor": 1.0,
                "installation_id": "INST-BR-001",
                "country": "BR",
            },
            {
                "stage_id": "STG-002",
                "stage_name": "Pig Iron Production (BF)",
                "product": "Pig iron",
                "input_weight_tonnes": 1600.0,
                "output_weight_tonnes": 1050.0,
                "emission_tco2e_per_tonne": 1.20,
                "total_emission_tco2e": 1260.0,
                "allocation_method": "mass_based",
                "allocation_factor": 0.656,
                "installation_id": "INST-TR-001",
                "country": "TR",
            },
            {
                "stage_id": "STG-003",
                "stage_name": "Crude Steel (BOF)",
                "product": "Crude steel slab",
                "input_weight_tonnes": 1050.0,
                "output_weight_tonnes": 1020.0,
                "emission_tco2e_per_tonne": 0.35,
                "total_emission_tco2e": 357.0,
                "allocation_method": "mass_based",
                "allocation_factor": 0.971,
                "installation_id": "INST-TR-001",
                "country": "TR",
            },
            {
                "stage_id": "STG-004",
                "stage_name": "Hot Rolling",
                "product": "Hot-rolled flat products",
                "input_weight_tonnes": 1020.0,
                "output_weight_tonnes": 1000.0,
                "emission_tco2e_per_tonne": 0.15,
                "total_emission_tco2e": 150.0,
                "allocation_method": "mass_based",
                "allocation_factor": 0.980,
                "installation_id": "INST-TR-001",
                "country": "TR",
            },
        ],
        "total_chain_emission_tco2e": 1831.0,
        "specific_emission_tco2e_per_tonne": 1.831,
        "production_route": "bf_bof",
        "provenance_hash": _compute_hash({"chain_id": "PC-STEEL-001"}),
    }


# ---------------------------------------------------------------------------
# Customs Declaration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_customs_declaration() -> Dict[str, Any]:
    """Create SAD (Single Administrative Document) data with 5 line items."""
    return {
        "declaration_id": "SAD-2026-DE-0001234",
        "declaration_type": "IM",
        "customs_office": "DE004600",
        "declarant_eori": "DE123456789012345",
        "date_of_acceptance": "2026-03-01",
        "country_of_dispatch": "TR",
        "line_items": [
            {
                "item_number": 1,
                "cn_code": "7207 11 14",
                "goods_description": "Semi-finished products of iron",
                "origin_country": "TR",
                "net_mass_kg": 500000,
                "statistical_value_eur": 275000.00,
                "customs_procedure": "4000",
                "additional_procedure": None,
                "cbam_applicable": True,
            },
            {
                "item_number": 2,
                "cn_code": "7208 51 20",
                "goods_description": "Flat-rolled products, hot-rolled",
                "origin_country": "TR",
                "net_mass_kg": 300000,
                "statistical_value_eur": 210000.00,
                "customs_procedure": "4000",
                "additional_procedure": None,
                "cbam_applicable": True,
            },
            {
                "item_number": 3,
                "cn_code": "7601 10 00",
                "goods_description": "Unwrought aluminium",
                "origin_country": "IN",
                "net_mass_kg": 200000,
                "statistical_value_eur": 480000.00,
                "customs_procedure": "4000",
                "additional_procedure": None,
                "cbam_applicable": True,
            },
            {
                "item_number": 4,
                "cn_code": "8471 30 00",
                "goods_description": "Portable computers (laptop)",
                "origin_country": "CN",
                "net_mass_kg": 5000,
                "statistical_value_eur": 850000.00,
                "customs_procedure": "4000",
                "additional_procedure": None,
                "cbam_applicable": False,
            },
            {
                "item_number": 5,
                "cn_code": "2523 29 00",
                "goods_description": "Portland cement, grey",
                "origin_country": "TR",
                "net_mass_kg": 1000000,
                "statistical_value_eur": 65000.00,
                "customs_procedure": "4000",
                "additional_procedure": None,
                "cbam_applicable": True,
            },
        ],
        "total_cbam_items": 4,
        "total_non_cbam_items": 1,
    }


# ---------------------------------------------------------------------------
# CBAM data for cross-regulation mapping
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_cbam_data() -> Dict[str, Any]:
    """Full CBAM dataset for cross-regulation mapping."""
    return {
        "reporting_year": 2026,
        "total_emissions_tco2e": 22500.0,
        "emissions_by_category": {
            "steel": 12500.0,
            "aluminium": 5800.0,
            "cement": 3200.0,
            "fertilizers": 800.0,
            "electricity": 150.0,
            "hydrogen": 50.0,
        },
        "total_import_value_eur": 15000000.0,
        "certificate_cost_eur": 44133.0,
        "net_obligation_tco2e": 562.5,
        "free_allocation_pct": 97.5,
        "suppliers_count": 15,
        "installations_count": 22,
        "countries_of_origin": ["TR", "CN", "IN", "UA", "BR"],
        "carbon_price_deductions": {
            "TR": {"price_eur_per_tco2e": 12.0, "emissions_tco2e": 8500.0},
            "CN": {"price_eur_per_tco2e": 8.5, "emissions_tco2e": 6200.0},
        },
        "verified": True,
        "verification_body": "TUV Rheinland Energy GmbH",
    }


# ---------------------------------------------------------------------------
# Audit Repository fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_audit_repository() -> Dict[str, Any]:
    """AuditRepository with 10 evidence records."""
    evidence_types = [
        "customs_declaration", "supplier_emission_data", "verification_statement",
        "certificate_purchase", "quarterly_report", "certificate_surrender",
        "carbon_price_deduction", "installation_permit", "calibration_record",
        "methodology_document",
    ]
    evidence_records = []
    for i in range(10):
        evidence_records.append({
            "evidence_id": f"EVD-{i + 1:04d}",
            "type": evidence_types[i],
            "description": f"Evidence record for {evidence_types[i]}",
            "file_reference": f"docs/{evidence_types[i]}/record_{i + 1}.pdf",
            "created_at": (
                datetime(2026, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=i * 10)
            ).isoformat(),
            "created_by": "compliance_officer@eurosteel-group.de",
            "hash": _compute_hash({
                "evidence_id": f"EVD-{i + 1:04d}",
                "type": evidence_types[i],
            }),
            "encrypted": i % 3 == 0,
            "retention_until": f"{2036 + (i % 5)}-12-31",
        })
    return {
        "repository_id": "AUDIT-REPO-2026",
        "entity_group_id": "GRP-EUROSTEEL-001",
        "evidence_records": evidence_records,
        "total_records": len(evidence_records),
        "chain_of_custody": [
            {
                "action": "created",
                "timestamp": "2026-01-01T00:00:00Z",
                "actor": "system",
            },
            {
                "action": "evidence_added",
                "timestamp": "2026-03-01T10:00:00Z",
                "actor": "compliance_officer@eurosteel-group.de",
                "evidence_count": 10,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Workflow Context fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow_context() -> Dict[str, Any]:
    """WorkflowContext with realistic state."""
    return {
        "workflow_id": f"WF-{_new_uuid()[:8]}",
        "workflow_type": "certificate_trading",
        "entity_group_id": "GRP-EUROSTEEL-001",
        "reporting_year": 2026,
        "started_at": _utcnow().isoformat(),
        "current_phase": "order_placement",
        "phases_completed": ["portfolio_review"],
        "phases_remaining": [
            "order_placement", "execution", "holding_check",
            "surrender_planning", "settlement",
        ],
        "status": "in_progress",
        "checkpoint": {
            "phase": "order_placement",
            "step": 2,
            "data": {"orders_submitted": 5},
        },
        "errors": [],
        "provenance_hash": _compute_hash({"workflow_type": "certificate_trading"}),
    }


# ---------------------------------------------------------------------------
# PACK-005 Engine IDs
# ---------------------------------------------------------------------------

PACK005_ENGINE_IDS = [
    "certificate_trading",
    "precursor_chain",
    "multi_entity",
    "registry_api",
    "advanced_analytics",
    "customs_automation",
    "cross_regulation",
    "audit_management",
]


# ---------------------------------------------------------------------------
# PACK-005 Workflow IDs
# ---------------------------------------------------------------------------

PACK005_WORKFLOW_IDS = [
    "certificate_trading",
    "multi_entity_consolidation",
    "registry_submission",
    "cross_regulation_sync",
    "customs_integration",
    "audit_preparation",
]


# ---------------------------------------------------------------------------
# PACK-005 Template IDs
# ---------------------------------------------------------------------------

PACK005_TEMPLATE_IDS = [
    "certificate_portfolio_report",
    "group_consolidation_report",
    "sourcing_scenario_analysis",
    "cross_regulation_mapping_report",
    "customs_integration_report",
    "audit_readiness_scorecard",
]


# ---------------------------------------------------------------------------
# PACK-005 Integration IDs
# ---------------------------------------------------------------------------

PACK005_INTEGRATION_IDS = [
    "pack_orchestrator",
    "registry_client",
    "taric_client",
    "ets_bridge",
    "cross_pack_bridge",
    "setup_wizard",
    "health_check",
]


@pytest.fixture
def engine_ids() -> List[str]:
    """Return the 8 PACK-005 engine IDs."""
    return list(PACK005_ENGINE_IDS)


@pytest.fixture
def workflow_ids() -> List[str]:
    """Return the 6 PACK-005 workflow IDs."""
    return list(PACK005_WORKFLOW_IDS)


@pytest.fixture
def template_ids() -> List[str]:
    """Return the 6 PACK-005 template IDs."""
    return list(PACK005_TEMPLATE_IDS)


@pytest.fixture
def integration_ids() -> List[str]:
    """Return the 7 PACK-005 integration IDs."""
    return list(PACK005_INTEGRATION_IDS)


# ---------------------------------------------------------------------------
# Stub classes for PACK-005 external dependencies
# ---------------------------------------------------------------------------

class StubRegistryClient:
    """Stub for CBAM Registry API client."""

    def __init__(self, mode: str = "mock"):
        self._mode = mode
        self._declarations: Dict[str, Dict[str, Any]] = {}
        self._balance: int = 100
        self._current_price: float = 78.50
        self._certificates_surrendered: int = 0

    def submit_declaration(self, declaration: Dict[str, Any]) -> Dict[str, Any]:
        decl_id = declaration.get("declaration_id", f"DECL-{_new_uuid()[:8]}")
        self._declarations[decl_id] = {**declaration, "status": "submitted"}
        return {"declaration_id": decl_id, "status": "submitted",
                "submitted_at": _utcnow().isoformat()}

    def amend_declaration(self, decl_id: str, amendments: Dict) -> Dict[str, Any]:
        if decl_id not in self._declarations:
            return {"error": "Declaration not found", "status": "rejected"}
        self._declarations[decl_id].update(amendments)
        self._declarations[decl_id]["status"] = "amended"
        return {"declaration_id": decl_id, "status": "amended"}

    def check_status(self, decl_id: str) -> Dict[str, Any]:
        decl = self._declarations.get(decl_id)
        if decl:
            return {"declaration_id": decl_id, "status": decl["status"]}
        return {"declaration_id": decl_id, "status": "not_found"}

    def purchase_certificates(self, quantity: int, price: float) -> Dict[str, Any]:
        self._balance += quantity
        return {"quantity": quantity, "price_eur": price,
                "new_balance": self._balance, "status": "purchased"}

    def surrender_certificates(self, quantity: int) -> Dict[str, Any]:
        if quantity > self._balance:
            return {"error": "Insufficient balance", "status": "rejected"}
        self._balance -= quantity
        self._certificates_surrendered += quantity
        return {"quantity": quantity, "new_balance": self._balance,
                "status": "surrendered"}

    def resell_certificates(self, quantity: int, price: float) -> Dict[str, Any]:
        if quantity > self._balance:
            return {"error": "Insufficient balance", "status": "rejected"}
        self._balance -= quantity
        return {"quantity": quantity, "price_eur": price,
                "new_balance": self._balance, "status": "resold"}

    def get_balance(self) -> Dict[str, Any]:
        return {"balance": self._balance, "as_of": _utcnow().isoformat()}

    def get_current_price(self) -> Dict[str, Any]:
        return {"price_eur": self._current_price, "source": "EU_ETS_AUCTION",
                "timestamp": _utcnow().isoformat()}

    def register_installation(self, data: Dict) -> Dict[str, Any]:
        inst_id = data.get("installation_id", f"INST-{_new_uuid()[:8]}")
        return {"installation_id": inst_id, "status": "registered"}

    def check_declarant_status(self, eori: str) -> Dict[str, Any]:
        return {"eori": eori, "status": "active", "member_state": eori[:2]}


class StubTARICClient:
    """Stub for EU TARIC/customs CN code lookup."""

    VALID_CN_CODES = {
        "7207 11 14": {"category": "steel", "desc": "Semi-finished iron"},
        "7208 51 20": {"category": "steel", "desc": "Flat-rolled hot-rolled"},
        "7209 15 00": {"category": "steel", "desc": "Cold-rolled flat"},
        "7208 10 00": {"category": "steel", "desc": "In coils hot-rolled"},
        "7601 10 00": {"category": "aluminium", "desc": "Unwrought aluminium"},
        "7601 20 80": {"category": "aluminium", "desc": "Aluminium alloys"},
        "7604 10 10": {"category": "aluminium", "desc": "Aluminium bars"},
        "2523 29 00": {"category": "cement", "desc": "Portland cement"},
        "2523 21 00": {"category": "cement", "desc": "White Portland"},
        "2523 10 00": {"category": "cement", "desc": "Cement clinkers"},
        "2814 10 00": {"category": "fertilizers", "desc": "Anhydrous ammonia"},
        "3102 10 10": {"category": "fertilizers", "desc": "Urea"},
        "2716 00 00": {"category": "electricity", "desc": "Electrical energy"},
        "2804 10 00": {"category": "hydrogen", "desc": "Hydrogen"},
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}

    def validate_cn_code(self, code: str) -> Dict[str, Any]:
        format_valid = bool(re.match(r"^\d{4}\s\d{2}\s\d{2}$", code))
        cbam_covered = code in self.VALID_CN_CODES
        category = self.VALID_CN_CODES.get(code, {}).get("category")
        result = {
            "code": code, "format_valid": format_valid,
            "cbam_covered": cbam_covered, "category": category,
        }
        self._cache[code] = result
        return result

    def get_cached(self, code: str) -> Optional[Dict[str, Any]]:
        return self._cache.get(code)

    def validate_eori(self, eori: str) -> Dict[str, Any]:
        valid = bool(re.match(r"^[A-Z]{2}\d{13,17}$", eori))
        return {"eori": eori, "valid": valid,
                "member_state": eori[:2] if valid else None}

    def check_aeo_status(self, eori: str) -> Dict[str, Any]:
        return {"eori": eori, "aeo_status": "AEOC",
                "valid_until": "2027-12-31"}


class StubETSBridge:
    """Stub for EU ETS bridge providing benchmarks and prices."""

    BENCHMARKS = {
        "steel_hot_metal": 1.328,
        "steel_eaf_carbon": 0.283,
        "steel_eaf_high_alloy": 0.352,
        "cement_clinker": 0.766,
        "aluminium_primary": 1.514,
        "fertilizer_ammonia": 1.619,
    }

    FREE_ALLOCATION_SCHEDULE = {
        2026: 97.5, 2027: 95.0, 2028: 90.0, 2029: 82.5,
        2030: 75.0, 2031: 60.0, 2032: 45.0, 2033: 30.0,
        2034: 0.0,
    }

    def get_benchmark(self, product_key: str) -> Optional[float]:
        return self.BENCHMARKS.get(product_key)

    def get_free_allocation_pct(self, year: int) -> float:
        return self.FREE_ALLOCATION_SCHEDULE.get(year, 0.0)

    def get_current_price(self) -> Dict[str, Any]:
        return {"price_eur": 78.50, "currency": "EUR",
                "source": "EU_ETS_AUCTION", "timestamp": _utcnow().isoformat()}


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
            "template_id": template_id, "title": title,
            "data": data, "provenance_hash": provenance_hash,
            "generated_at": _utcnow().isoformat(),
        }, indent=2, default=str)
    else:
        content = f"{title}: {json.dumps(data, default=str)}"

    return {
        "template_id": template_id, "format": output_format,
        "content": content, "provenance_hash": provenance_hash,
        "generated_at": _utcnow().isoformat(),
    }


@pytest.fixture
def template_renderer():
    """Return the template render stub function."""
    return render_template_stub


@pytest.fixture
def mock_registry_client() -> StubRegistryClient:
    """Return a StubRegistryClient instance."""
    return StubRegistryClient()


@pytest.fixture
def mock_taric_client() -> StubTARICClient:
    """Return a StubTARICClient instance."""
    return StubTARICClient()


@pytest.fixture
def mock_ets_bridge() -> StubETSBridge:
    """Return a StubETSBridge instance."""
    return StubETSBridge()
