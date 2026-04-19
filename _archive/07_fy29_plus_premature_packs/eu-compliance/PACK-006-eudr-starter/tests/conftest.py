# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Shared Test Fixtures
====================================================

Provides reusable pytest fixtures for all PACK-006 test modules including
sample configuration, supplier profiles, geolocation data, DDS documents,
risk scoring data, commodity classification, and workflow contexts.

All fixtures are self-contained with no external dependencies. They model
realistic EUDR (EU Deforestation Regulation) compliance scenarios across
all seven regulated commodities: cattle, cocoa, coffee, palm oil, rubber,
soya, and wood (timber).

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import json
import math
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PACK_YAML_PATH = PACK_ROOT / "pack.yaml"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
SECTORS_DIR = CONFIG_DIR / "sectors"
DEMO_DIR = CONFIG_DIR / "demo"
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


# ---------------------------------------------------------------------------
# Constants - EUDR regulation parameters
# ---------------------------------------------------------------------------

EUDR_COMMODITIES = ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"]

EUDR_CUTOFF_DATE = date(2020, 12, 31)

EUDR_HIGH_RISK_COUNTRIES = [
    "BRA", "IDN", "COD", "CMR", "GHA", "CIV", "MYS", "PNG", "LAO", "MMR",
    "NGA", "SLE", "LBR", "GTM", "HND", "PRY", "BOL", "COL", "PER", "ECU",
]

EUDR_LOW_RISK_COUNTRIES = [
    "DEU", "FRA", "GBR", "NLD", "BEL", "AUT", "CHE", "SWE", "NOR", "FIN",
    "DNK", "IRL", "LUX", "ISL", "LIE", "CZE", "SVK", "SVN", "EST", "LVA",
    "LTU", "HRV", "CAN", "JPN", "AUS", "NZL", "SGP", "KOR", "USA", "TWN",
]

EUDR_STANDARD_RISK_COUNTRIES = [
    "IND", "THA", "VNM", "PHL", "MEX", "ZAF", "KEN", "TZA", "UGA", "ETH",
    "CHN", "RUS", "TUR", "EGY", "MAR", "DZA", "TUN", "SRB", "BGR", "ROU",
]

CERTIFICATION_SCHEMES = [
    "RSPO", "FSC", "PEFC", "Rainforest_Alliance", "UTZ", "Fairtrade",
    "RTRS", "ProTerra", "ISCC", "Bonsucro",
]

CHAIN_OF_CUSTODY_MODELS = [
    "identity_preserved", "segregated", "mass_balance", "book_and_claim",
]

OPERATOR_TYPES = ["OPERATOR", "TRADER"]

COMPANY_SIZES = ["SME", "MID_MARKET", "LARGE"]

DDS_TYPES = ["STANDARD", "SIMPLIFIED"]

RISK_LEVELS = ["LOW", "STANDARD", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of arbitrary data for provenance tracking."""
    if isinstance(data, dict):
        serialized = json.dumps(data, sort_keys=True, default=str)
    elif isinstance(data, str):
        serialized = data
    else:
        serialized = str(data)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def assert_provenance_hash(result: Dict[str, Any]) -> None:
    """Verify that a result contains a valid SHA-256 provenance hash."""
    assert "provenance_hash" in result, "Result missing 'provenance_hash' field"
    h = result["provenance_hash"]
    assert isinstance(h, str), f"provenance_hash must be str, got {type(h)}"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert re.match(r"^[0-9a-f]{64}$", h), f"Invalid hex hash: {h}"


def assert_valid_uuid(value: str) -> None:
    """Verify that a value is a valid UUID4 string."""
    assert isinstance(value, str), f"Expected str, got {type(value)}"
    try:
        parsed = uuid.UUID(value, version=4)
        assert str(parsed) == value.lower(), f"UUID normalization mismatch: {value}"
    except ValueError:
        raise AssertionError(f"Invalid UUID4: {value}")


def generate_coordinates(country: str, count: int = 1) -> List[Dict[str, float]]:
    """Generate realistic WGS84 coordinates for a given country code.

    Returns a list of dicts with 'latitude' and 'longitude' keys.
    Coordinates are approximate centroids with slight offsets to simulate
    multiple plots within the same country.
    """
    centroids = {
        "BRA": (-14.235, -51.925),
        "IDN": (-0.789, 113.921),
        "CIV": (7.540, -5.547),
        "GHA": (7.946, -1.023),
        "COL": (4.571, -74.297),
        "MYS": (4.210, 101.976),
        "PER": (-9.190, -75.015),
        "CMR": (7.370, 12.354),
        "GTM": (15.783, -90.231),
        "ARG": (-38.416, -63.617),
        "DEU": (51.166, 10.452),
        "FRA": (46.228, 2.214),
        "NLD": (52.133, 5.291),
        "IND": (20.594, 78.963),
        "THA": (15.870, 100.993),
        "COD": (-4.038, 21.759),
        "PNG": (-6.315, 143.956),
        "NGA": (9.082, 8.675),
        "PRY": (-23.443, -58.444),
        "BOL": (-16.290, -63.588),
    }
    lat, lon = centroids.get(country, (0.0, 0.0))
    coords = []
    for i in range(count):
        offset_lat = (i * 0.012) % 0.5
        offset_lon = (i * 0.017) % 0.5
        coords.append({
            "latitude": round(lat + offset_lat, 6),
            "longitude": round(lon + offset_lon, 6),
        })
    return coords


# ---------------------------------------------------------------------------
# Pack YAML fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pack_yaml_path() -> Path:
    """Return the absolute path to pack.yaml."""
    return PACK_YAML_PATH


@pytest.fixture(scope="session")
def pack_yaml_raw(pack_yaml_path) -> str:
    """Return the raw text content of pack.yaml, or empty string if missing."""
    if pack_yaml_path.exists():
        return pack_yaml_path.read_text(encoding="utf-8")
    return ""


@pytest.fixture(scope="session")
def pack_yaml(pack_yaml_raw) -> Dict[str, Any]:
    """Return the parsed pack.yaml as a dictionary, or empty dict if missing."""
    if pack_yaml_raw:
        import yaml
        return yaml.safe_load(pack_yaml_raw) or {}
    return {}


# ---------------------------------------------------------------------------
# Sample configuration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create a complete EUDRStarterConfig with all sub-configs.

    Models a mid-market food importer subject to EUDR requirements,
    importing palm oil and cocoa from high-risk countries.
    """
    return {
        "metadata": {
            "name": "eudr-starter",
            "version": "1.0.0",
            "category": "eu-compliance",
            "display_name": "EUDR Starter Pack",
            "regulation": "EU Deforestation Regulation",
            "regulation_id": "Regulation (EU) 2023/1115",
            "tier": "starter",
        },
        "operator_type": "OPERATOR",
        "company_size": "MID_MARKET",
        "dds_type": "STANDARD",
        "commodities": EUDR_COMMODITIES,
        "cutoff_date": str(EUDR_CUTOFF_DATE),
        "geolocation": {
            "precision_decimals": 6,
            "coordinate_system": "WGS84",
            "polygon_required_above_ha": 4.0,
            "format": "decimal_degrees",
        },
        "risk_scoring": {
            "country_weight": 0.30,
            "supplier_weight": 0.25,
            "commodity_weight": 0.20,
            "document_weight": 0.25,
            "low_threshold": 0.30,
            "standard_threshold": 0.50,
            "high_threshold": 0.70,
            "critical_threshold": 0.85,
        },
        "certification_schemes": CERTIFICATION_SCHEMES,
        "chain_of_custody_models": CHAIN_OF_CUSTODY_MODELS,
        "country_risk_database": {
            "high_risk": EUDR_HIGH_RISK_COUNTRIES,
            "low_risk": EUDR_LOW_RISK_COUNTRIES,
            "standard_risk": EUDR_STANDARD_RISK_COUNTRIES,
        },
        "performance_targets": {
            "dds_generation_max_seconds": 60,
            "risk_assessment_max_seconds": 30,
            "geolocation_validation_max_seconds": 15,
            "health_check_max_seconds": 30,
        },
        "presets": ["palm_oil_importer", "timber_importer", "multi_commodity", "sme_trader"],
        "sectors": ["food_beverage", "forestry_paper", "cosmetics", "automotive", "retail"],
    }


# ---------------------------------------------------------------------------
# Supplier fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_supplier() -> Dict[str, Any]:
    """Create a sample supplier profile with EUDR-relevant data."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "PT Sawit Lestari",
        "country": "IDN",
        "commodity": "palm_oil",
        "eori_number": "ID1234567890",
        "address": "Jl. Raya Pekanbaru No. 42, Riau, Indonesia",
        "certifications": [
            {
                "scheme": "RSPO",
                "certificate_number": "RSPO-2024-001234",
                "valid_from": "2024-01-01",
                "valid_until": "2027-12-31",
                "scope": "identity_preserved",
                "status": "active",
            },
        ],
        "dd_status": "IN_PROGRESS",
        "risk_score": 0.62,
        "risk_level": "STANDARD",
        "data_completeness": 0.85,
        "plots": [
            {
                "plot_id": str(uuid.uuid4()),
                "latitude": -0.512345,
                "longitude": 101.456789,
                "area_hectares": 25.5,
                "polygon": [
                    [-0.510, 101.454],
                    [-0.510, 101.460],
                    [-0.515, 101.460],
                    [-0.515, 101.454],
                    [-0.510, 101.454],
                ],
            },
        ],
        "last_audit_date": "2025-06-15",
        "registration_date": "2024-03-01",
    }


@pytest.fixture
def sample_suppliers_list() -> List[Dict[str, Any]]:
    """Create a list of 5 suppliers across different commodities and risk levels."""
    suppliers = [
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "PT Sawit Lestari",
            "country": "IDN",
            "commodity": "palm_oil",
            "eori_number": "ID1234567890",
            "certifications": [{"scheme": "RSPO", "status": "active"}],
            "dd_status": "COMPLETED",
            "risk_score": 0.55,
            "risk_level": "STANDARD",
            "data_completeness": 0.92,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Maderas Tropicales S.A.",
            "country": "BRA",
            "commodity": "wood",
            "eori_number": "BR9876543210",
            "certifications": [{"scheme": "FSC", "status": "active"}],
            "dd_status": "IN_PROGRESS",
            "risk_score": 0.72,
            "risk_level": "HIGH",
            "data_completeness": 0.78,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Cocoa Gold SARL",
            "country": "CIV",
            "commodity": "cocoa",
            "eori_number": "CI5678901234",
            "certifications": [{"scheme": "Rainforest_Alliance", "status": "active"}],
            "dd_status": "IN_PROGRESS",
            "risk_score": 0.68,
            "risk_level": "HIGH",
            "data_completeness": 0.70,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Deutsche Holz GmbH",
            "country": "DEU",
            "commodity": "wood",
            "eori_number": "DE1122334455",
            "certifications": [{"scheme": "PEFC", "status": "active"}],
            "dd_status": "COMPLETED",
            "risk_score": 0.15,
            "risk_level": "LOW",
            "data_completeness": 0.98,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Soja del Sur S.A.",
            "country": "ARG",
            "commodity": "soya",
            "eori_number": "AR6677889900",
            "certifications": [],
            "dd_status": "NOT_STARTED",
            "risk_score": 0.78,
            "risk_level": "HIGH",
            "data_completeness": 0.45,
        },
    ]
    return suppliers


# ---------------------------------------------------------------------------
# Commodity-specific supplier fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_palm_oil_supplier() -> Dict[str, Any]:
    """IDN supplier with RSPO certification for palm oil."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "PT Kelapa Sawit Nusantara",
        "country": "IDN",
        "commodity": "palm_oil",
        "eori_number": "ID2233445566",
        "certifications": [
            {
                "scheme": "RSPO",
                "certificate_number": "RSPO-2024-005678",
                "valid_from": "2024-06-01",
                "valid_until": "2027-05-31",
                "scope": "mass_balance",
                "status": "active",
            },
        ],
        "dd_status": "COMPLETED",
        "risk_score": 0.52,
        "risk_level": "STANDARD",
        "data_completeness": 0.90,
        "cn_codes": ["1511 10 90", "1511 90 19"],
    }


@pytest.fixture
def sample_timber_supplier() -> Dict[str, Any]:
    """BRA supplier with FSC certification for timber."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "Madeira Verde Ltda",
        "country": "BRA",
        "commodity": "wood",
        "eori_number": "BR3344556677",
        "certifications": [
            {
                "scheme": "FSC",
                "certificate_number": "FSC-C123456",
                "valid_from": "2023-01-01",
                "valid_until": "2028-12-31",
                "scope": "segregated",
                "status": "active",
            },
        ],
        "dd_status": "IN_PROGRESS",
        "risk_score": 0.65,
        "risk_level": "HIGH",
        "data_completeness": 0.82,
        "cn_codes": ["4403 49 00", "4407 29 00"],
    }


@pytest.fixture
def sample_cocoa_supplier() -> Dict[str, Any]:
    """CIV supplier with Rainforest Alliance certification for cocoa."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "Cacao Excellence SARL",
        "country": "CIV",
        "commodity": "cocoa",
        "eori_number": "CI4455667788",
        "certifications": [
            {
                "scheme": "Rainforest_Alliance",
                "certificate_number": "RA-2024-CIV-0789",
                "valid_from": "2024-03-15",
                "valid_until": "2027-03-14",
                "scope": "segregated",
                "status": "active",
            },
        ],
        "dd_status": "IN_PROGRESS",
        "risk_score": 0.70,
        "risk_level": "HIGH",
        "data_completeness": 0.72,
        "cn_codes": ["1801 00 00", "1802 00 00"],
    }


@pytest.fixture
def sample_soy_supplier() -> Dict[str, Any]:
    """BRA supplier with RTRS certification for soya."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "AgroSoja Brasil S.A.",
        "country": "BRA",
        "commodity": "soya",
        "eori_number": "BR5566778899",
        "certifications": [
            {
                "scheme": "RTRS",
                "certificate_number": "RTRS-2024-BR-0456",
                "valid_from": "2024-07-01",
                "valid_until": "2026-06-30",
                "scope": "mass_balance",
                "status": "active",
            },
        ],
        "dd_status": "COMPLETED",
        "risk_score": 0.60,
        "risk_level": "STANDARD",
        "data_completeness": 0.88,
        "cn_codes": ["1201 90 00", "1507 10 90"],
    }


@pytest.fixture
def sample_cattle_supplier() -> Dict[str, Any]:
    """ARG supplier without standard certification for cattle."""
    return {
        "supplier_id": str(uuid.uuid4()),
        "name": "Estancia La Pampa S.A.",
        "country": "ARG",
        "commodity": "cattle",
        "eori_number": "AR6677001122",
        "certifications": [],
        "dd_status": "NOT_STARTED",
        "risk_score": 0.80,
        "risk_level": "HIGH",
        "data_completeness": 0.40,
        "cn_codes": ["0102 29 10", "0201 10 00"],
    }


# ---------------------------------------------------------------------------
# Plot and geolocation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_plot() -> Dict[str, Any]:
    """Create a plot with valid WGS84 coordinates and polygon."""
    return {
        "plot_id": str(uuid.uuid4()),
        "name": "Plot Riau-001",
        "country": "IDN",
        "latitude": -0.512345,
        "longitude": 101.456789,
        "area_hectares": 25.5,
        "coordinate_system": "WGS84",
        "precision_decimals": 6,
        "polygon": [
            [-0.510000, 101.454000],
            [-0.510000, 101.460000],
            [-0.515000, 101.460000],
            [-0.515000, 101.454000],
            [-0.510000, 101.454000],
        ],
        "land_use": "palm_oil_plantation",
        "deforestation_free_since": "2019-06-15",
        "satellite_verification_date": "2025-11-01",
    }


@pytest.fixture
def sample_plots_list() -> List[Dict[str, Any]]:
    """Create a list of 10 plots across 5 countries."""
    countries = ["IDN", "BRA", "CIV", "MYS", "COL"]
    commodities = ["palm_oil", "wood", "cocoa", "rubber", "coffee"]
    plots = []
    for i in range(10):
        country = countries[i % 5]
        commodity = commodities[i % 5]
        coords = generate_coordinates(country, 1)[0]
        area = 2.5 + (i * 3.5)
        plot_id = str(uuid.uuid4())

        polygon = None
        if area > 4.0:
            lat, lon = coords["latitude"], coords["longitude"]
            d = math.sqrt(area / 100) * 0.01
            polygon = [
                [lat - d, lon - d],
                [lat - d, lon + d],
                [lat + d, lon + d],
                [lat + d, lon - d],
                [lat - d, lon - d],
            ]

        plots.append({
            "plot_id": plot_id,
            "name": f"Plot-{country}-{i + 1:03d}",
            "country": country,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "area_hectares": round(area, 1),
            "coordinate_system": "WGS84",
            "precision_decimals": 6,
            "polygon": polygon,
            "land_use": f"{commodity}_plantation",
            "commodity": commodity,
            "deforestation_free_since": f"201{8 + (i % 3)}-0{1 + (i % 9)}-15",
        })

    return plots


@pytest.fixture
def sample_geolocation() -> Dict[str, Any]:
    """Create a valid coordinate set with polygon for geolocation verification."""
    return {
        "point": {
            "latitude": -2.345678,
            "longitude": 104.567890,
            "coordinate_system": "WGS84",
            "precision_decimals": 6,
        },
        "polygon": {
            "type": "Polygon",
            "coordinates": [[
                [104.560, -2.340],
                [104.575, -2.340],
                [104.575, -2.350],
                [104.560, -2.350],
                [104.560, -2.340],
            ]],
        },
        "area_hectares": 18.7,
        "country": "IDN",
        "is_land_based": True,
        "overlaps_protected_area": False,
    }


# ---------------------------------------------------------------------------
# DDS (Due Diligence Statement) fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dds() -> Dict[str, Any]:
    """Create a complete DDS document per EUDR Annex II."""
    operator_id = str(uuid.uuid4())
    dds_ref = f"DDS-{datetime.now().strftime('%Y%m%d')}-{operator_id[:8].upper()}"
    return {
        "dds_reference": dds_ref,
        "dds_type": "STANDARD",
        "status": "DRAFT",
        "created_at": datetime.now().isoformat(),
        "operator": {
            "name": "EcoImports B.V.",
            "country": "NLD",
            "eori_number": "NL123456789000",
            "address": "Keizersgracht 100, 1015 AA Amsterdam, Netherlands",
            "operator_type": "OPERATOR",
            "contact_email": "compliance@ecoimports.nl",
        },
        "commodities": [
            {
                "commodity": "palm_oil",
                "cn_codes": ["1511 10 90", "1511 90 19"],
                "description": "Crude palm oil and refined palm olein",
                "quantity_kg": 250000,
                "country_of_production": "IDN",
            },
        ],
        "suppliers": [
            {
                "supplier_id": str(uuid.uuid4()),
                "name": "PT Sawit Lestari",
                "country": "IDN",
                "certifications": ["RSPO"],
            },
        ],
        "geolocation": {
            "plots": [
                {
                    "plot_id": str(uuid.uuid4()),
                    "latitude": -0.512345,
                    "longitude": 101.456789,
                    "area_hectares": 25.5,
                    "polygon": [
                        [-0.510, 101.454],
                        [-0.510, 101.460],
                        [-0.515, 101.460],
                        [-0.515, 101.454],
                        [-0.510, 101.454],
                    ],
                },
            ],
        },
        "risk_assessment": {
            "country_risk": 0.65,
            "supplier_risk": 0.45,
            "commodity_risk": 0.70,
            "document_risk": 0.30,
            "composite_risk": 0.53,
            "risk_level": "STANDARD",
            "simplified_dd_eligible": False,
        },
        "cutoff_compliance": {
            "cutoff_date": "2020-12-31",
            "deforestation_free": True,
            "evidence_type": "satellite_imagery",
            "verification_date": "2025-11-01",
        },
        "evidence": [
            {
                "evidence_id": str(uuid.uuid4()),
                "type": "certificate",
                "description": "RSPO certification for PT Sawit Lestari",
                "file_reference": "RSPO-2024-001234.pdf",
            },
            {
                "evidence_id": str(uuid.uuid4()),
                "type": "satellite_analysis",
                "description": "Forest cover analysis 2018-2025",
                "file_reference": "satellite_report_riau_2025.pdf",
            },
        ],
        "annex_ii_complete": True,
        "provenance_hash": _compute_hash({"dds_ref": dds_ref, "timestamp": datetime.now().isoformat()}),
    }


# ---------------------------------------------------------------------------
# Risk data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_risk_data() -> Dict[str, Any]:
    """Create country + supplier + commodity + document risk scores."""
    return {
        "country_risk": {
            "country": "IDN",
            "score": 0.65,
            "level": "HIGH",
            "benchmark": "HIGH",
            "factors": {
                "deforestation_rate": 0.75,
                "governance_index": 0.55,
                "enforcement_score": 0.50,
                "transparency_score": 0.60,
            },
        },
        "supplier_risk": {
            "supplier_id": str(uuid.uuid4()),
            "score": 0.45,
            "level": "STANDARD",
            "factors": {
                "certification_status": 0.20,
                "audit_history": 0.35,
                "data_completeness": 0.55,
                "engagement_level": 0.40,
            },
        },
        "commodity_risk": {
            "commodity": "palm_oil",
            "score": 0.70,
            "level": "HIGH",
            "factors": {
                "deforestation_association": 0.85,
                "land_use_change_rate": 0.70,
                "supply_chain_complexity": 0.65,
                "traceability_difficulty": 0.60,
            },
        },
        "document_risk": {
            "score": 0.30,
            "level": "LOW",
            "factors": {
                "completeness": 0.90,
                "authenticity_verified": True,
                "consistency_score": 0.85,
                "age_months": 3,
            },
        },
        "composite": {
            "score": 0.53,
            "level": "STANDARD",
            "weights": {
                "country": 0.30,
                "supplier": 0.25,
                "commodity": 0.20,
                "document": 0.25,
            },
        },
    }


# ---------------------------------------------------------------------------
# Customs declaration fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_customs_declaration() -> Dict[str, Any]:
    """Create an import declaration with CN codes."""
    return {
        "declaration_id": str(uuid.uuid4()),
        "declaration_date": "2025-12-01",
        "importer": {
            "name": "EcoImports B.V.",
            "eori_number": "NL123456789000",
            "country": "NLD",
        },
        "exporter": {
            "name": "PT Sawit Lestari",
            "country": "IDN",
        },
        "items": [
            {
                "cn_code": "1511 10 90",
                "description": "Crude palm oil",
                "quantity_kg": 150000,
                "value_eur": 120000,
                "country_of_origin": "IDN",
                "eudr_covered": True,
                "commodity": "palm_oil",
            },
            {
                "cn_code": "1511 90 19",
                "description": "Palm olein, refined",
                "quantity_kg": 100000,
                "value_eur": 95000,
                "country_of_origin": "IDN",
                "eudr_covered": True,
                "commodity": "palm_oil",
            },
        ],
        "dds_reference": "DDS-20251201-ABCD1234",
        "status": "PENDING_DDS",
    }


# ---------------------------------------------------------------------------
# Workflow context fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_workflow_context() -> Dict[str, Any]:
    """Create a WorkflowContext with realistic state."""
    workflow_id = str(uuid.uuid4())
    return {
        "workflow_id": workflow_id,
        "workflow_type": "dds_generation",
        "status": "IN_PROGRESS",
        "current_phase": "risk_assessment",
        "phases_completed": ["data_collection", "geolocation_validation"],
        "phases_remaining": ["risk_assessment", "dds_assembly", "review", "submission"],
        "started_at": (datetime.now() - timedelta(hours=2)).isoformat(),
        "operator": {
            "name": "EcoImports B.V.",
            "eori_number": "NL123456789000",
        },
        "supplier_count": 5,
        "plot_count": 12,
        "commodity": "palm_oil",
        "checkpoint": {
            "phase": "geolocation_validation",
            "completed_at": (datetime.now() - timedelta(minutes=30)).isoformat(),
            "data_hash": _compute_hash({"phase": "geolocation_validation"}),
        },
        "errors": [],
        "warnings": [
            "Plot PLT-003 has only 5 decimal places precision (required: 6)",
        ],
    }


# ---------------------------------------------------------------------------
# Mock agent registry for EUDR agents
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_eudr_agent_registry() -> MagicMock:
    """Create a mocked agent registry with all EUDR pack agents."""
    registry = MagicMock()

    eudr_agent_ids = [f"AGENT-EUDR-{i:03d}" for i in range(1, 41)]
    app_ids = ["GL-EUDR-APP"]
    found_ids = [f"AGENT-FOUND-{i:03d}" for i in range(1, 11)]
    all_ids = eudr_agent_ids + app_ids + found_ids

    def is_available(agent_id: str) -> bool:
        return agent_id in all_ids

    def get_agent(agent_id: str) -> MagicMock:
        if agent_id not in all_ids:
            raise KeyError(f"Agent {agent_id} not registered")
        agent = MagicMock()
        agent.agent_id = agent_id
        agent.status = "healthy"
        agent.version = "1.0.0"
        return agent

    registry.is_available.side_effect = is_available
    registry.get_agent.side_effect = get_agent
    registry.list_agents.return_value = all_ids
    registry.health_check.return_value = {aid: "healthy" for aid in all_ids}
    return registry


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "eudr_pack_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ---------------------------------------------------------------------------
# Annex I CN code reference data
# ---------------------------------------------------------------------------

ANNEX_I_CN_CODES = {
    "cattle": [
        "0102 21 10", "0102 21 30", "0102 21 90", "0102 29 10", "0102 29 21",
        "0102 29 29", "0102 29 41", "0102 29 49", "0102 29 51", "0102 29 59",
        "0102 29 61", "0102 29 69", "0102 29 91", "0102 29 99", "0102 31 00",
        "0102 39 10", "0102 39 90", "0102 90 20", "0102 90 91", "0102 90 99",
        "0201 10 00", "0201 20 20", "0201 20 30", "0201 20 50", "0201 20 90",
        "0201 30 00", "0202 10 00", "0202 20 10", "0202 20 30", "0202 20 50",
        "0202 20 90", "0202 30 10", "0202 30 50", "0202 30 90", "0206 10 91",
        "0206 10 95", "0206 10 98",
        # Edible offal and prepared meat (EUDR Annex I extended)
        "0206 21 00", "0206 22 00", "0206 29 91", "0206 29 99",
        "0210 20 10", "0210 20 90", "0210 99 21", "0210 99 29",
        "1502 10 10", "1502 10 90", "1502 90 10", "1502 90 90",
        "1602 50 10", "1602 50 31", "1602 50 95",
        "4101 20 10", "4101 20 30", "4101 20 50", "4101 20 80",
        "4101 50 10", "4101 50 30", "4101 50 50", "4101 50 90",
        "4101 90 00",
        "4104 11 10", "4104 11 51", "4104 11 59", "4104 11 90",
        "4104 19 10", "4104 19 90", "4104 41 11", "4104 41 19",
        "4104 41 51", "4104 49 11", "4104 49 19", "4104 49 90",
    ],
    "cocoa": [
        "1801 00 00", "1802 00 00", "1803 10 00", "1803 20 00", "1804 00 00",
        "1805 00 00", "1806 10 15", "1806 10 20", "1806 10 30", "1806 10 90",
        "1806 20 10", "1806 20 30", "1806 20 50", "1806 20 70", "1806 20 80",
        "1806 20 95", "1806 31 00", "1806 32 10", "1806 32 90", "1806 90 11",
        "1806 90 19", "1806 90 31", "1806 90 39", "1806 90 50", "1806 90 60",
        "1806 90 70", "1806 90 90",
    ],
    "coffee": [
        "0901 11 00", "0901 12 00", "0901 21 00", "0901 22 00", "0901 90 10",
        "0901 90 90",
        # Coffee extracts and preparations (EUDR Annex I)
        "2101 11 00", "2101 12 92", "2101 12 98",
    ],
    "palm_oil": [
        "1511 10 10", "1511 10 90", "1511 90 11", "1511 90 19", "1511 90 91",
        "1511 90 99", "1513 21 11", "1513 21 19", "1513 21 30", "1513 21 90",
        "1513 29 11", "1513 29 19", "1513 29 30", "1513 29 50", "1513 29 90",
        # Hydrogenated palm fats and oleochemicals (EUDR Annex I)
        "1516 20 10", "1516 20 91", "1516 20 96", "1516 20 98",
        "2915 70 40", "2915 70 50", "2915 90 70",
        "2916 15 00", "2916 19 95",
        "3401 11 00", "3401 19 00", "3401 20 10", "3401 20 90",
        "3823 11 00", "3823 12 00", "3823 19 10", "3823 19 30", "3823 19 90",
        "3823 70 00",
        "3826 00 10", "3826 00 90",
    ],
    "rubber": [
        "4001 10 00", "4001 21 00", "4001 22 00", "4001 29 10", "4001 29 20",
        "4001 29 90", "4001 30 00", "4002 11 00", "4002 19 10", "4002 19 20",
        "4002 19 30", "4002 19 90", "4002 20 00", "4005 10 00", "4005 20 00",
        "4005 91 00", "4005 99 00", "4006 10 00", "4006 90 00",
        # Vulcanised rubber articles (EUDR Annex I)
        "4007 00 00",
        "4008 11 00", "4008 19 00", "4008 21 10", "4008 21 90", "4008 29 00",
        "4010 11 00", "4010 12 00", "4010 19 00",
        "4011 10 00", "4011 20 10", "4011 20 90", "4011 30 00", "4011 40 00",
        "4011 50 00", "4011 70 00", "4011 80 00", "4011 90 00",
        "4012 11 00", "4012 12 00", "4012 19 00", "4012 20 00", "4012 90 10",
        "4012 90 90",
        "4013 10 00", "4013 20 00", "4013 90 00",
        "4015 11 00", "4015 19 00", "4015 90 00",
        "4016 10 00", "4016 91 00", "4016 93 00", "4016 95 00", "4016 99 52",
        "4016 99 57", "4016 99 91", "4016 99 97",
        "4017 00 00",
    ],
    "soya": [
        "1201 10 00", "1201 90 00", "1507 10 10", "1507 10 90", "1507 90 10",
        "1507 90 90", "2304 00 00",
        # Soya flour, meal and animal feed preparations (EUDR Annex I)
        "1208 10 00",
        "2309 10 11", "2309 10 13", "2309 10 31", "2309 10 33",
        "2309 10 51", "2309 10 53",
        "2309 90 31", "2309 90 33", "2309 90 41", "2309 90 43",
        "2309 90 51", "2309 90 53",
    ],
    "wood": [
        "4401 11 00", "4401 12 00", "4401 21 00", "4401 22 00", "4401 31 00",
        "4401 32 00", "4401 39 20", "4401 39 80", "4401 41 00", "4401 49 00",
        "4403 11 00", "4403 12 00", "4403 21 10", "4403 21 90", "4403 22 00",
        "4403 23 10", "4403 23 90", "4403 24 00", "4403 25 10", "4403 25 90",
        "4403 26 00", "4403 41 00", "4403 42 00", "4403 49 00", "4403 91 00",
        "4403 93 00", "4403 94 00", "4403 95 00", "4403 96 00", "4403 97 00",
        "4403 98 00", "4403 99 00", "4407 11 10", "4407 11 20", "4407 11 90",
        "4407 12 00", "4407 19 10", "4407 19 20", "4407 19 90", "4407 21 10",
        "4407 21 90", "4407 22 10", "4407 22 90", "4407 23 10", "4407 23 90",
        "4407 25 10", "4407 25 30", "4407 25 50", "4407 25 90", "4407 26 10",
        "4407 26 30", "4407 26 50", "4407 26 90", "4407 27 10", "4407 27 30",
        "4407 27 50", "4407 27 90", "4407 28 10", "4407 28 30", "4407 28 50",
        "4407 28 90", "4407 29 20", "4407 29 25", "4407 29 30", "4407 29 60",
        "4407 29 83", "4407 29 85", "4407 29 95", "4407 91 10", "4407 91 90",
        "4407 92 00", "4407 93 10", "4407 93 40", "4407 93 91", "4407 93 99",
        "4407 94 10", "4407 94 91", "4407 94 99", "4407 95 10", "4407 95 91",
        "4407 95 99", "4407 96 10", "4407 96 91", "4407 96 99", "4407 97 10",
        "4407 97 91", "4407 97 99", "4407 99 27", "4407 99 40", "4407 99 90",
    ],
}

ALL_CN_CODES = []
for codes in ANNEX_I_CN_CODES.values():
    ALL_CN_CODES.extend(codes)
