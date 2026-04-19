"""
PACK-047 GHG Emissions Benchmark Pack - Shared Test Fixtures
=============================================================

Provides shared fixtures for all PACK-047 test modules including
engine instances, sample data, configuration objects, peer candidates,
emissions time-series, pathway waypoints, portfolio holdings, external
dataset mock responses, and helper utilities.

All numeric fixtures use Decimal for regulatory precision.

Author: GreenLang QA Team
Date: March 2026
"""
from __future__ import annotations

import hashlib
import json
import sys
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Path setup - ensure PACK-047 root is importable
# ---------------------------------------------------------------------------
PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))


# ---------------------------------------------------------------------------
# Decimal comparison helpers
# ---------------------------------------------------------------------------


def decimal_approx(
    actual: Optional[Decimal],
    expected: Decimal,
    tolerance: Decimal = Decimal("0.000001"),
) -> bool:
    """Compare two Decimals within a given tolerance.

    Returns True if |actual - expected| <= tolerance.
    """
    if actual is None:
        return False
    return abs(actual - expected) <= tolerance


def assert_decimal_equal(
    actual: Optional[Decimal],
    expected: Decimal,
    tolerance: Decimal = Decimal("0.000001"),
    msg: str = "",
) -> None:
    """Assert two Decimals are equal within tolerance."""
    assert actual is not None, f"Actual value is None. {msg}"
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Decimal mismatch: actual={actual}, expected={expected}, "
        f"diff={diff}, tolerance={tolerance}. {msg}"
    )


def assert_decimal_gt(
    actual: Optional[Decimal],
    threshold: Decimal,
    msg: str = "",
) -> None:
    """Assert Decimal value is greater than threshold."""
    assert actual is not None, f"Actual value is None. {msg}"
    assert actual > threshold, (
        f"Expected {actual} > {threshold}. {msg}"
    )


def assert_decimal_between(
    actual: Optional[Decimal],
    lo: Decimal,
    hi: Decimal,
    msg: str = "",
) -> None:
    """Assert Decimal value is within [lo, hi]."""
    assert actual is not None, f"Actual value is None. {msg}"
    assert lo <= actual <= hi, (
        f"Expected {lo} <= {actual} <= {hi}. {msg}"
    )


def compute_test_hash(data: Any) -> str:
    """Compute SHA-256 hash from arbitrary data (for determinism testing)."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Create PackConfig-like dict with corporate_general preset defaults."""
    return {
        "pack_id": "PACK-047",
        "pack_name": "GHG Emissions Benchmark",
        "version": "1.0.0",
        "preset": "corporate_general",
        "organisation_id": "org-test-001",
        "organisation_name": "ACME Corp",
        "sector": "INDUSTRIALS",
        "sector_codes": {
            "gics": "2010",
            "nace": "C25",
            "isic": "C25",
        },
        "reporting_year": 2025,
        "base_year": 2020,
        "revenue_usd_m": Decimal("500"),
        "employees_fte": 2000,
        "scope_boundary": "scope_1_2_3",
        "gwp_version": "AR6",
        "currency": "USD",
        "peer_group": {
            "min_peers": 5,
            "max_peers": 50,
            "outlier_iqr_k": Decimal("1.5"),
            "sector_weight": Decimal("0.40"),
            "size_weight": Decimal("0.25"),
            "geo_weight": Decimal("0.20"),
            "quality_weight": Decimal("0.15"),
        },
        "pathway": {
            "primary": "IEA_NZE",
            "secondary": ["IPCC_C1", "SBTi_SDA"],
            "target_year": 2050,
        },
        "itr_method": "budget_based",
        "portfolio_metrics": ["WACI", "carbon_footprint", "carbon_intensity"],
        "output_formats": ["markdown", "html", "json", "csv", "xbrl"],
        "data_quality_framework": "PCAF",
        "transition_risk_enabled": True,
        "trajectory_window_years": 5,
        "confidence_level": Decimal("0.95"),
    }


# ---------------------------------------------------------------------------
# Peer Candidate Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_peer_candidates() -> List[Dict[str, Any]]:
    """Create 25 PeerCandidate objects across 5 sectors with varying sizes/geographies."""
    candidates = []
    sectors = [
        ("INDUSTRIALS", "C25", "2010", "Manufacturing"),
        ("ENERGY", "D35", "3510", "Power Generation"),
        ("MATERIALS", "C23", "1510", "Cement & Concrete"),
        ("TRANSPORTATION", "H49", "2030", "Road Freight"),
        ("REAL_ESTATE", "L68", "6010", "Commercial REIT"),
    ]
    geographies = ["EU_WEST", "NA_EAST", "APAC_NORTH", "EU_EAST", "NA_WEST"]
    revenue_bands = [
        Decimal("15"),    # SMALL
        Decimal("75"),    # MEDIUM
        Decimal("200"),   # LARGE
        Decimal("500"),   # ENTERPRISE
        Decimal("2000"),  # MEGA
    ]
    grid_efs = [
        Decimal("0.250"),  # EU_WEST
        Decimal("0.380"),  # NA_EAST
        Decimal("0.600"),  # APAC_NORTH
        Decimal("0.450"),  # EU_EAST
        Decimal("0.350"),  # NA_WEST
    ]
    for s_idx, (sector, nace, gics, label) in enumerate(sectors):
        for p_idx in range(5):
            peer_id = f"peer-{s_idx:02d}-{p_idx:02d}"
            candidates.append({
                "peer_id": peer_id,
                "peer_name": f"{label} Peer {p_idx + 1}",
                "sector": sector,
                "nace_code": nace,
                "gics_code": gics,
                "geography": geographies[p_idx],
                "revenue_usd_m": revenue_bands[p_idx] + Decimal(str(s_idx * 10)),
                "grid_emission_factor": grid_efs[p_idx],
                "emissions_tco2e": Decimal(str(1000 + s_idx * 500 + p_idx * 200)),
                "scope_coverage": "scope_1_2" if p_idx < 3 else "scope_1_2_3",
                "data_year": 2024 - (p_idx % 2),
                "verification_status": "verified" if p_idx < 2 else (
                    "reported" if p_idx < 4 else "estimated"
                ),
                "data_quality_score": min(p_idx + 1, 5),
            })
    return candidates


# ---------------------------------------------------------------------------
# Emissions Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_emissions_data() -> Dict[str, Any]:
    """Create 5 years of emissions data for org + 20 peers."""
    org_data = {}
    for year in range(2020, 2025):
        factor = Decimal("1") - Decimal("0.05") * Decimal(str(year - 2020))
        org_data[str(year)] = {
            "scope_1_tco2e": (Decimal("5000") * factor).quantize(Decimal("0.01")),
            "scope_2_location_tco2e": (Decimal("3000") * factor).quantize(Decimal("0.01")),
            "scope_2_market_tco2e": (Decimal("2500") * factor).quantize(Decimal("0.01")),
            "scope_3_tco2e": (Decimal("15000") * factor).quantize(Decimal("0.01")),
        }

    peer_data = {}
    for p_idx in range(20):
        peer_id = f"peer-{p_idx:03d}"
        peer_years = {}
        base_s1 = Decimal(str(2000 + p_idx * 300))
        base_s2 = Decimal(str(1000 + p_idx * 150))
        for year in range(2020, 2025):
            reduction = Decimal("1") - Decimal("0.03") * Decimal(str(year - 2020))
            peer_years[str(year)] = {
                "scope_1_tco2e": (base_s1 * reduction).quantize(Decimal("0.01")),
                "scope_2_location_tco2e": (base_s2 * reduction).quantize(Decimal("0.01")),
            }
        peer_data[peer_id] = peer_years

    return {
        "organisation": org_data,
        "peers": peer_data,
        "base_year": "2020",
        "reporting_year": "2024",
    }


# ---------------------------------------------------------------------------
# Pathway Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pathway_data() -> Dict[str, Any]:
    """Create IEA NZE + IPCC C1 + SBTi SDA waypoints."""
    return {
        "IEA_NZE": {
            "name": "IEA Net Zero by 2050",
            "base_year": 2020,
            "base_value": Decimal("100"),
            "waypoints": {
                "2025": Decimal("85"),
                "2030": Decimal("60"),
                "2035": Decimal("40"),
                "2040": Decimal("25"),
                "2045": Decimal("12"),
                "2050": Decimal("0"),
            },
            "unit": "index_base_100",
            "source": "IEA World Energy Outlook 2023",
        },
        "IPCC_C1": {
            "name": "IPCC AR6 C1 (1.5C no/low overshoot)",
            "base_year": 2020,
            "base_value": Decimal("100"),
            "waypoints": {
                "2025": Decimal("80"),
                "2030": Decimal("55"),
                "2035": Decimal("33"),
                "2040": Decimal("15"),
                "2045": Decimal("5"),
                "2050": Decimal("-5"),
            },
            "unit": "index_base_100",
            "source": "IPCC AR6 WGIII Chapter 3",
        },
        "IPCC_C2": {
            "name": "IPCC AR6 C2 (1.5C high overshoot)",
            "base_year": 2020,
            "base_value": Decimal("100"),
            "waypoints": {
                "2025": Decimal("87"),
                "2030": Decimal("65"),
                "2035": Decimal("45"),
                "2040": Decimal("28"),
                "2045": Decimal("15"),
                "2050": Decimal("5"),
            },
            "unit": "index_base_100",
            "source": "IPCC AR6 WGIII Chapter 3",
        },
        "IPCC_C3": {
            "name": "IPCC AR6 C3 (well below 2C)",
            "base_year": 2020,
            "base_value": Decimal("100"),
            "waypoints": {
                "2025": Decimal("92"),
                "2030": Decimal("75"),
                "2035": Decimal("58"),
                "2040": Decimal("42"),
                "2045": Decimal("28"),
                "2050": Decimal("18"),
            },
            "unit": "index_base_100",
            "source": "IPCC AR6 WGIII Chapter 3",
        },
        "SBTi_SDA_POWER": {
            "name": "SBTi SDA Power Generation (1.5C)",
            "base_year": 2020,
            "base_value": Decimal("0.415"),
            "waypoints": {
                "2025": Decimal("0.310"),
                "2030": Decimal("0.138"),
                "2035": Decimal("0.050"),
                "2040": Decimal("0.015"),
                "2050": Decimal("0.000"),
            },
            "unit": "tCO2e/MWh",
            "source": "SBTi SDA Tool v1.2",
        },
        "OECM": {
            "name": "One Earth Climate Model (OECM)",
            "base_year": 2020,
            "base_value": Decimal("100"),
            "waypoints": {
                "2025": Decimal("82"),
                "2030": Decimal("52"),
                "2035": Decimal("30"),
                "2040": Decimal("12"),
                "2050": Decimal("-8"),
            },
            "unit": "index_base_100",
            "source": "University of Technology Sydney (2022)",
        },
    }


# ---------------------------------------------------------------------------
# Portfolio Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_portfolio() -> List[Dict[str, Any]]:
    """Create 50-holding portfolio across 5 asset classes."""
    asset_classes = [
        "listed_equity",
        "corporate_bonds",
        "sovereign_bonds",
        "real_estate",
        "private_equity",
    ]
    holdings = []
    for ac_idx, ac in enumerate(asset_classes):
        for h_idx in range(10):
            holding_id = f"hold-{ac_idx:02d}-{h_idx:02d}"
            evic = Decimal(str(500 + h_idx * 100 + ac_idx * 50))
            outstanding = Decimal(str(2000 + h_idx * 500))
            attribution = evic / outstanding
            emissions = Decimal(str(1000 + h_idx * 200 + ac_idx * 150))
            holdings.append({
                "holding_id": holding_id,
                "holding_name": f"{ac.replace('_', ' ').title()} Holding {h_idx + 1}",
                "asset_class": ac,
                "sector": ["INDUSTRIALS", "ENERGY", "MATERIALS", "FINANCIALS", "TECHNOLOGY"][h_idx % 5],
                "geography": ["EU", "NA", "APAC", "LATAM", "MEA"][h_idx % 5],
                "investment_value_usd_m": evic,
                "enterprise_value_usd_m": outstanding,
                "ownership_share_pct": (attribution * Decimal("100")).quantize(Decimal("0.01")),
                "emissions_scope_1_2_tco2e": emissions,
                "emissions_scope_3_tco2e": emissions * Decimal("2.5"),
                "revenue_usd_m": Decimal(str(100 + h_idx * 50)),
                "pcaf_data_quality_score": min(h_idx % 5 + 1, 5),
                "weight_pct": Decimal("2.00"),  # 50 holdings * 2% = 100%
            })
    return holdings


# ---------------------------------------------------------------------------
# External Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_external_data() -> Dict[str, Any]:
    """Create CDP/TPI/GRESB mock responses."""
    return {
        "cdp": {
            "source": "CDP Climate Change 2025",
            "retrieved_at": "2025-06-15T00:00:00Z",
            "cache_ttl_hours": 24,
            "records": [
                {
                    "entity_id": "cdp-001",
                    "entity_name": "Industry Leader A",
                    "sector": "INDUSTRIALS",
                    "score": "A",
                    "scope_1_tco2e": Decimal("3500"),
                    "scope_2_tco2e": Decimal("2000"),
                    "scope_3_tco2e": Decimal("12000"),
                    "revenue_usd_m": Decimal("600"),
                    "year": 2024,
                },
                {
                    "entity_id": "cdp-002",
                    "entity_name": "Industry Peer B",
                    "sector": "INDUSTRIALS",
                    "score": "B",
                    "scope_1_tco2e": Decimal("7000"),
                    "scope_2_tco2e": Decimal("4000"),
                    "scope_3_tco2e": Decimal("25000"),
                    "revenue_usd_m": Decimal("800"),
                    "year": 2024,
                },
            ],
        },
        "tpi": {
            "source": "Transition Pathway Initiative v5.0",
            "retrieved_at": "2025-05-01T00:00:00Z",
            "cache_ttl_hours": 168,
            "records": [
                {
                    "entity_id": "tpi-001",
                    "entity_name": "Steel Major Alpha",
                    "sector": "STEEL",
                    "management_quality_score": 4,
                    "carbon_performance_alignment": "below_2c",
                    "intensity_2024": Decimal("1.85"),
                    "intensity_unit": "tCO2e/t_steel",
                    "benchmark_2030": Decimal("1.20"),
                },
                {
                    "entity_id": "tpi-002",
                    "entity_name": "Cement Corp Beta",
                    "sector": "CEMENT",
                    "management_quality_score": 3,
                    "carbon_performance_alignment": "national_pledges",
                    "intensity_2024": Decimal("0.65"),
                    "intensity_unit": "tCO2e/t_cement",
                    "benchmark_2030": Decimal("0.52"),
                },
            ],
        },
        "gresb": {
            "source": "GRESB 2025",
            "retrieved_at": "2025-10-01T00:00:00Z",
            "cache_ttl_hours": 720,
            "records": [
                {
                    "entity_id": "gresb-001",
                    "entity_name": "Office REIT Gamma",
                    "property_type": "office",
                    "gresb_score": 82,
                    "energy_intensity_kwh_m2": Decimal("125"),
                    "carbon_intensity_kgco2e_m2": Decimal("45"),
                    "floor_area_m2": Decimal("250000"),
                    "year": 2024,
                },
            ],
        },
        "crrem": {
            "source": "CRREM Decarbonisation Pathways v2.0",
            "retrieved_at": "2025-07-01T00:00:00Z",
            "cache_ttl_hours": 8760,
            "records": [
                {
                    "property_type": "office",
                    "country": "DEU",
                    "pathway_1_5c": {
                        "2025": Decimal("85"),
                        "2030": Decimal("55"),
                        "2035": Decimal("30"),
                        "2040": Decimal("12"),
                        "2050": Decimal("0"),
                    },
                    "unit": "kgCO2e/m2",
                },
            ],
        },
        "iss_esg": {
            "source": "ISS ESG Climate Solutions 2025",
            "retrieved_at": "2025-08-15T00:00:00Z",
            "cache_ttl_hours": 336,
            "records": [
                {
                    "entity_id": "iss-001",
                    "entity_name": "Global Manufacturer Delta",
                    "sector": "INDUSTRIALS",
                    "itr_scope_1_2": Decimal("2.4"),
                    "itr_scope_1_2_3": Decimal("2.8"),
                    "sbti_status": "committed",
                    "transition_risk_rating": "medium",
                },
            ],
        },
    }


# ---------------------------------------------------------------------------
# Engine Fixtures (stubbed for tests that create engines directly)
# ---------------------------------------------------------------------------


@pytest.fixture
def peer_group_engine_config() -> Dict[str, Any]:
    """Configuration dict for PeerGroupConstructionEngine."""
    return {
        "min_peers": 5,
        "max_peers": 50,
        "outlier_iqr_k": Decimal("1.5"),
        "sector_weight": Decimal("0.40"),
        "size_weight": Decimal("0.25"),
        "geo_weight": Decimal("0.20"),
        "quality_weight": Decimal("0.15"),
        "min_similarity_score": Decimal("0.30"),
        "require_same_primary_sector": True,
    }


@pytest.fixture
def normalisation_engine_config() -> Dict[str, Any]:
    """Configuration dict for ScopeNormalisationEngine."""
    return {
        "target_gwp_version": "AR6",
        "target_currency": "USD",
        "target_scope_boundary": "scope_1_2",
        "prorata_method": "linear",
        "biogenic_treatment": "exclude",
        "consolidation_approach": "operational_control",
    }


@pytest.fixture
def pathway_engine_config() -> Dict[str, Any]:
    """Configuration dict for PathwayAlignmentEngine."""
    return {
        "primary_pathway": "IEA_NZE",
        "secondary_pathways": ["IPCC_C1", "SBTi_SDA"],
        "interpolation_method": "linear",
        "base_year": 2020,
        "target_year": 2050,
    }


@pytest.fixture
def itr_engine_config() -> Dict[str, Any]:
    """Configuration dict for ImpliedTemperatureRiseEngine."""
    return {
        "method": "budget_based",
        "budget_scenario": "1.5C",
        "confidence_level": Decimal("0.95"),
        "scope_boundary": "scope_1_2",
    }


@pytest.fixture
def portfolio_engine_config() -> Dict[str, Any]:
    """Configuration dict for PortfolioBenchmarkingEngine."""
    return {
        "metrics": ["WACI", "carbon_footprint", "carbon_intensity"],
        "pcaf_version": "v3",
        "index_benchmark": "MSCI_World",
        "attribution_method": "EVIC",
    }


@pytest.fixture
def data_quality_engine_config() -> Dict[str, Any]:
    """Configuration dict for DataQualityScoringEngine."""
    return {
        "framework": "PCAF",
        "ghg_protocol_matrix": True,
        "min_quality_score": 1,
        "max_quality_score": 5,
    }


@pytest.fixture
def transition_risk_engine_config() -> Dict[str, Any]:
    """Configuration dict for TransitionRiskScoringEngine."""
    return {
        "carbon_price_usd_per_tco2e": Decimal("80"),
        "carbon_price_growth_rate_pct": Decimal("10"),
        "regulatory_risk_weight": Decimal("0.30"),
        "competitive_risk_weight": Decimal("0.25"),
        "stranding_risk_weight": Decimal("0.25"),
        "carbon_budget_risk_weight": Decimal("0.20"),
    }


@pytest.fixture
def reporting_engine_config() -> Dict[str, Any]:
    """Configuration dict for BenchmarkReportingEngine."""
    return {
        "output_formats": ["markdown", "html", "json", "csv", "xbrl"],
        "include_provenance": True,
        "include_methodology": True,
        "company_name": "ACME Corp",
        "reporting_period": "FY2025",
    }


# ---------------------------------------------------------------------------
# Workflow Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_workflow_input(sample_config, sample_peer_candidates) -> Dict[str, Any]:
    """Create sample workflow execution input."""
    return {
        "organisation_id": "org-test-001",
        "organisation_name": "ACME Corp",
        "sector": "INDUSTRIALS",
        "reporting_year": 2025,
        "base_year": 2020,
        "peer_candidates": sample_peer_candidates[:10],
        "config": sample_config,
    }


# ---------------------------------------------------------------------------
# Template Input Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_league_table_data() -> Dict[str, Any]:
    """Create sample data for league table rendering."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "FY2025",
        "peer_group_name": "EU Industrials",
        "entries": [
            {
                "rank": 1,
                "entity_name": "Best Performer Ltd",
                "emissions_tco2e": Decimal("2500"),
                "intensity": Decimal("8.33"),
                "percentile": Decimal("5"),
                "itr": Decimal("1.5"),
            },
            {
                "rank": 2,
                "entity_name": "ACME Corp",
                "emissions_tco2e": Decimal("8000"),
                "intensity": Decimal("16.00"),
                "percentile": Decimal("35"),
                "itr": Decimal("2.1"),
                "is_organisation": True,
            },
            {
                "rank": 3,
                "entity_name": "Median Peer Inc",
                "emissions_tco2e": Decimal("12000"),
                "intensity": Decimal("24.00"),
                "percentile": Decimal("50"),
                "itr": Decimal("2.5"),
            },
        ],
    }


@pytest.fixture
def sample_radar_chart_data() -> Dict[str, Any]:
    """Create sample radar chart data for multi-dimension benchmark."""
    return {
        "dimensions": [
            "Absolute Emissions",
            "Intensity (Revenue)",
            "Intensity (Physical)",
            "Pathway Alignment",
            "Data Quality",
            "Trajectory Rate",
        ],
        "organisation_values": [
            Decimal("65"),
            Decimal("72"),
            Decimal("58"),
            Decimal("80"),
            Decimal("85"),
            Decimal("70"),
        ],
        "peer_median_values": [
            Decimal("50"),
            Decimal("50"),
            Decimal("50"),
            Decimal("50"),
            Decimal("50"),
            Decimal("50"),
        ],
        "best_in_class_values": [
            Decimal("90"),
            Decimal("88"),
            Decimal("85"),
            Decimal("95"),
            Decimal("92"),
            Decimal("90"),
        ],
    }


@pytest.fixture
def sample_pathway_chart_data(sample_pathway_data) -> Dict[str, Any]:
    """Create sample pathway alignment graph data."""
    return {
        "organisation_trajectory": {
            "2020": Decimal("100"),
            "2021": Decimal("95"),
            "2022": Decimal("90"),
            "2023": Decimal("84"),
            "2024": Decimal("80"),
        },
        "pathways": {
            "IEA_NZE": sample_pathway_data["IEA_NZE"]["waypoints"],
            "IPCC_C1": sample_pathway_data["IPCC_C1"]["waypoints"],
        },
        "gap_to_pathway": {
            "IEA_NZE": Decimal("20"),
            "IPCC_C1": Decimal("25"),
        },
    }


# ---------------------------------------------------------------------------
# Pipeline / Orchestrator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pipeline_config() -> Dict[str, Any]:
    """Create sample pipeline configuration for orchestrator tests."""
    return {
        "company_name": "ACME Corp",
        "reporting_period": "2025",
        "sector": "INDUSTRIALS",
        "scope_boundary": "scope_1_2_3",
        "max_retries": 2,
        "retry_base_delay_s": 0.01,
        "enable_parallel": False,
        "timeout_per_phase_s": 30.0,
        "enable_peer_benchmarking": True,
        "enable_pathway_alignment": True,
        "enable_itr_calculation": True,
        "enable_portfolio_analysis": False,
        "enable_transition_risk": True,
    }
