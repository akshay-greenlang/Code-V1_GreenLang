# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Shared Test Fixtures
========================================================

Provides reusable pytest fixtures for all PACK-007 test modules including
advanced geolocation, Monte Carlo risk simulation, portfolio management,
continuous monitoring, supplier benchmarking, and all professional-tier
EUDR compliance workflows.

Extends PACK-006 fixtures with professional-tier capabilities including:
- Multi-operator portfolio management
- Monte Carlo risk simulation
- Advanced satellite monitoring
- Continuous compliance monitoring
- Supplier benchmarking and scorecards
- Regulatory change tracking
- Grievance mechanism integration
- Cross-regulation compliance

Author: GreenLang QA Team
Version: 1.0.0
"""

import hashlib
import importlib.util
import json
import math
import random
import re
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


# ---------------------------------------------------------------------------
# Dynamic import helper for hyphenated directory names
# ---------------------------------------------------------------------------

def _import_from_path(module_name: str, file_path: Path):
    """Import a module from a file path (supports hyphenated directories)."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

# Professional tier: expanded protected area buffers
PROTECTED_AREA_BUFFER_KM = [1, 3, 5, 10]

# Professional tier: satellite data sources
SATELLITE_SOURCES = ["Sentinel-1", "Sentinel-2", "Landsat-8", "MODIS", "Planet"]

# Professional tier: risk simulation parameters
SIMULATION_COUNTS = [1000, 5000, 10000]
CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]

# Professional tier: audit retention
AUDIT_RETENTION_YEARS = [3, 5, 7, 10]


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
    """Generate realistic WGS84 coordinates for a given country code."""
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
# PACK-007 Operator fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_operator_data() -> Dict[str, Any]:
    """Create a sample operator with EORI, name, and commodities."""
    return {
        "operator_id": str(uuid.uuid4()),
        "name": "GlobalTrade GmbH",
        "country": "DEU",
        "eori_number": "DE1234567890123",
        "operator_type": "OPERATOR",
        "company_size": "LARGE",
        "commodities": ["palm_oil", "cocoa", "wood"],
        "registration_date": "2024-01-15",
        "contact_email": "compliance@globaltrade.de",
        "address": "Berliner Str. 100, 10115 Berlin, Germany",
        "annual_imports_tonnes": 50000,
        "supplier_count": 25,
    }


@pytest.fixture
def sample_suppliers() -> List[Dict[str, Any]]:
    """Create 5 suppliers across different commodities and countries."""
    return [
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "PT Sawit Makmur",
            "country": "IDN",
            "commodity": "palm_oil",
            "eori_number": "ID1111222233",
            "certifications": [{"scheme": "RSPO", "status": "active"}],
            "risk_score": 0.58,
            "risk_level": "STANDARD",
            "data_completeness": 0.88,
            "performance_score": 0.75,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Cacao Premium SAS",
            "country": "CIV",
            "commodity": "cocoa",
            "eori_number": "CI9988776655",
            "certifications": [{"scheme": "Rainforest_Alliance", "status": "active"}],
            "risk_score": 0.72,
            "risk_level": "HIGH",
            "data_completeness": 0.73,
            "performance_score": 0.62,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Amazonia Timber Ltd",
            "country": "BRA",
            "commodity": "wood",
            "eori_number": "BR5544332211",
            "certifications": [{"scheme": "FSC", "status": "active"}],
            "risk_score": 0.68,
            "risk_level": "HIGH",
            "data_completeness": 0.80,
            "performance_score": 0.70,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "Java Coffee Exports",
            "country": "IDN",
            "commodity": "coffee",
            "eori_number": "ID7766554433",
            "certifications": [{"scheme": "Fairtrade", "status": "active"}],
            "risk_score": 0.45,
            "risk_level": "STANDARD",
            "data_completeness": 0.92,
            "performance_score": 0.85,
        },
        {
            "supplier_id": str(uuid.uuid4()),
            "name": "European Wood Industries",
            "country": "DEU",
            "commodity": "wood",
            "eori_number": "DE9876543210",
            "certifications": [{"scheme": "PEFC", "status": "active"}],
            "risk_score": 0.12,
            "risk_level": "LOW",
            "data_completeness": 0.98,
            "performance_score": 0.95,
        },
    ]


@pytest.fixture
def sample_plots() -> List[Dict[str, Any]]:
    """Create 10 plot geolocations with lat/lon across multiple countries."""
    countries = ["IDN", "BRA", "CIV", "MYS", "COL"]
    commodities = ["palm_oil", "wood", "cocoa", "rubber", "coffee"]
    plots = []

    for i in range(10):
        country = countries[i % 5]
        commodity = commodities[i % 5]
        coords = generate_coordinates(country, 1)[0]
        area = 5.0 + (i * 4.0)

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
            "plot_id": str(uuid.uuid4()),
            "name": f"Plot-{country}-{i + 1:03d}",
            "country": country,
            "latitude": coords["latitude"],
            "longitude": coords["longitude"],
            "area_hectares": round(area, 1),
            "polygon": polygon,
            "commodity": commodity,
            "deforestation_free_since": f"201{7 + (i % 4)}-{1 + (i % 12):02d}-15",
            "satellite_verified": True,
            "protected_area_proximity_km": round(random.uniform(0.5, 20.0), 1),
        })

    return plots


@pytest.fixture
def sample_dds() -> Dict[str, Any]:
    """Create a Due Diligence Statement with comprehensive data."""
    operator_id = str(uuid.uuid4())
    dds_ref = f"DDS-PROF-{datetime.now().strftime('%Y%m%d')}-{operator_id[:8].upper()}"
    return {
        "dds_reference": dds_ref,
        "dds_type": "STANDARD",
        "status": "SUBMITTED",
        "created_at": datetime.now().isoformat(),
        "operator": {
            "operator_id": operator_id,
            "name": "GlobalTrade GmbH",
            "eori_number": "DE1234567890123",
            "country": "DEU",
        },
        "commodities": ["palm_oil", "cocoa"],
        "total_volume_tonnes": 15000,
        "supplier_count": 12,
        "plot_count": 28,
        "risk_assessment": {
            "composite_risk": 0.58,
            "risk_level": "STANDARD",
            "monte_carlo_var_95": 0.65,
        },
        "compliance_status": "COMPLIANT",
        "provenance_hash": _compute_hash(dds_ref),
    }


@pytest.fixture
def sample_risk_data() -> Dict[str, Any]:
    """Create risk inputs for Monte Carlo simulation."""
    return {
        "country_risk": {
            "mean": 0.65,
            "std_dev": 0.10,
            "distribution": "normal",
            "min_value": 0.40,
            "max_value": 0.90,
        },
        "supplier_risk": {
            "mean": 0.55,
            "std_dev": 0.12,
            "distribution": "beta",
            "alpha": 5,
            "beta": 4,
        },
        "commodity_risk": {
            "mean": 0.70,
            "std_dev": 0.08,
            "distribution": "triangular",
            "min_value": 0.50,
            "mode": 0.70,
            "max_value": 0.85,
        },
        "document_risk": {
            "mean": 0.30,
            "std_dev": 0.15,
            "distribution": "normal",
            "min_value": 0.10,
            "max_value": 0.60,
        },
        "weights": {
            "country": 0.30,
            "supplier": 0.25,
            "commodity": 0.20,
            "document": 0.25,
        },
        "simulation_count": 10000,
        "confidence_levels": [0.90, 0.95, 0.99],
    }


@pytest.fixture
def sample_portfolio() -> List[Dict[str, Any]]:
    """Create a portfolio with 3 operators for multi-operator management."""
    return [
        {
            "operator_id": str(uuid.uuid4()),
            "name": "GlobalTrade GmbH",
            "eori_number": "DE1234567890123",
            "country": "DEU",
            "commodities": ["palm_oil", "cocoa"],
            "annual_imports_tonnes": 50000,
            "supplier_count": 25,
            "risk_score": 0.52,
        },
        {
            "operator_id": str(uuid.uuid4()),
            "name": "EcoImports B.V.",
            "eori_number": "NL9876543210987",
            "country": "NLD",
            "commodities": ["coffee", "wood"],
            "annual_imports_tonnes": 30000,
            "supplier_count": 18,
            "risk_score": 0.45,
        },
        {
            "operator_id": str(uuid.uuid4()),
            "name": "Nordic Commodities AS",
            "eori_number": "NO5544332211009",
            "country": "NOR",
            "commodities": ["wood", "soya"],
            "annual_imports_tonnes": 20000,
            "supplier_count": 12,
            "risk_score": 0.38,
        },
    ]


@pytest.fixture
def sample_complaints() -> List[Dict[str, Any]]:
    """Create 3 grievance records for grievance mechanism testing."""
    return [
        {
            "complaint_id": str(uuid.uuid4()),
            "submitted_at": (datetime.now() - timedelta(days=10)).isoformat(),
            "category": "deforestation",
            "severity": "HIGH",
            "status": "INVESTIGATING",
            "supplier_id": str(uuid.uuid4()),
            "description": "Alleged deforestation in Plot-BRA-003",
            "sla_deadline": (datetime.now() + timedelta(days=20)).isoformat(),
        },
        {
            "complaint_id": str(uuid.uuid4()),
            "submitted_at": (datetime.now() - timedelta(days=45)).isoformat(),
            "category": "data_quality",
            "severity": "MEDIUM",
            "status": "RESOLVED",
            "supplier_id": str(uuid.uuid4()),
            "description": "Incomplete geolocation data for 5 plots",
            "sla_deadline": (datetime.now() - timedelta(days=15)).isoformat(),
            "resolution": "Supplier provided updated coordinates",
        },
        {
            "complaint_id": str(uuid.uuid4()),
            "submitted_at": (datetime.now() - timedelta(days=3)).isoformat(),
            "category": "certification",
            "severity": "LOW",
            "status": "OPEN",
            "supplier_id": str(uuid.uuid4()),
            "description": "Certification expiry not updated in system",
            "sla_deadline": (datetime.now() + timedelta(days=27)).isoformat(),
        },
    ]


@pytest.fixture
def sample_audit_entries() -> List[Dict[str, Any]]:
    """Create sample audit trail entries."""
    return [
        {
            "audit_id": str(uuid.uuid4()),
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
            "event_type": "DDS_CREATED",
            "operator_id": str(uuid.uuid4()),
            "user_email": "analyst@globaltrade.de",
            "details": {"dds_reference": "DDS-PROF-20260315-ABCD1234"},
        },
        {
            "audit_id": str(uuid.uuid4()),
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
            "event_type": "RISK_ASSESSMENT_COMPLETED",
            "operator_id": str(uuid.uuid4()),
            "user_email": "analyst@globaltrade.de",
            "details": {"risk_score": 0.58, "risk_level": "STANDARD"},
        },
        {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "event_type": "DDS_SUBMITTED",
            "operator_id": str(uuid.uuid4()),
            "user_email": "compliance@globaltrade.de",
            "details": {"submission_status": "SUCCESS"},
        },
    ]


@pytest.fixture
def sample_supply_chain() -> Dict[str, Any]:
    """Create multi-tier supply chain data."""
    return {
        "operator_id": str(uuid.uuid4()),
        "commodity": "palm_oil",
        "tiers": [
            {
                "tier": 1,
                "suppliers": [
                    {
                        "supplier_id": str(uuid.uuid4()),
                        "name": "PT Sawit Direct",
                        "country": "IDN",
                        "volume_tonnes": 5000,
                    },
                ],
            },
            {
                "tier": 2,
                "suppliers": [
                    {
                        "supplier_id": str(uuid.uuid4()),
                        "name": "PT Mill Processor",
                        "country": "IDN",
                        "volume_tonnes": 5000,
                    },
                    {
                        "supplier_id": str(uuid.uuid4()),
                        "name": "Malaysian Refinery",
                        "country": "MYS",
                        "volume_tonnes": 3000,
                    },
                ],
            },
            {
                "tier": 3,
                "suppliers": [
                    {
                        "supplier_id": str(uuid.uuid4()),
                        "name": "Smallholder Cooperative A",
                        "country": "IDN",
                        "volume_tonnes": 2000,
                    },
                    {
                        "supplier_id": str(uuid.uuid4()),
                        "name": "Smallholder Cooperative B",
                        "country": "IDN",
                        "volume_tonnes": 1500,
                    },
                ],
            },
        ],
        "max_tier_depth": 3,
        "total_suppliers": 5,
    }


@pytest.fixture
def sample_regulatory_changes() -> List[Dict[str, Any]]:
    """Create sample regulatory change notifications."""
    return [
        {
            "change_id": str(uuid.uuid4()),
            "regulation": "EUDR",
            "change_type": "AMENDMENT",
            "effective_date": "2026-06-01",
            "title": "Extended commodity list - cashew nuts added",
            "impact_level": "MEDIUM",
            "description": "Cashew nuts added to Annex I regulated commodities",
            "published_date": "2026-03-01",
        },
        {
            "change_id": str(uuid.uuid4()),
            "regulation": "EUDR",
            "change_type": "GUIDANCE",
            "effective_date": "2026-04-15",
            "title": "Updated geolocation polygon guidance",
            "impact_level": "LOW",
            "description": "Clarification on polygon closure requirements",
            "published_date": "2026-03-10",
        },
    ]


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return EUDRProfessionalConfig-like dictionary."""
    return {
        "pack_id": "PACK-007",
        "pack_name": "EUDR Professional Pack",
        "tier": "professional",
        "extends": "PACK-006",
        "advanced_geolocation": {
            "sentinel_monitoring": True,
            "protected_area_buffer_km": 5.0,
            "indigenous_land_check": True,
            "forest_change_detection": True,
        },
        "scenario_risk": {
            "simulation_count": 10000,
            "confidence_levels": [0.90, 0.95, 0.99],
            "distribution_types": ["normal", "beta", "triangular"],
        },
        "satellite_monitoring": {
            "sources": ["Sentinel-1", "Sentinel-2", "Landsat-8"],
            "update_frequency_days": 14,
        },
        "continuous_monitoring": {
            "enabled": True,
            "check_interval_hours": 24,
            "alert_channels": ["email", "webhook"],
        },
        "portfolio_config": {
            "max_operators": 100,
            "shared_supplier_pool": True,
        },
        "audit_management": {
            "retention_years": 5,
            "automatic_archival": True,
        },
        "protected_area_config": {
            "buffer_km": 5,
            "check_indigenous_lands": True,
        },
        "regulatory_tracking": {
            "enabled": True,
            "sources": ["EUR-Lex", "EU_Commission"],
        },
        "grievance_config": {
            "sla_days": 30,
            "escalation_enabled": True,
        },
        "cross_regulation_config": {
            "regulations": ["EUDR", "CBAM", "CSRD"],
        },
        "supply_chain_config": {
            "max_tier_depth": 5,
            "traceability_level": "plot",
        },
        "supplier_benchmark_config": {
            "peer_group_size": 10,
            "scoring_dimensions": 6,
        },
    }


# ---------------------------------------------------------------------------
# Temporary output directory
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "eudr_professional_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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
