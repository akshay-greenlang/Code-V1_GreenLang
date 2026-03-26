# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Shared Test Fixtures

Provides reusable fixtures for all test modules including sample site records,
facility characteristics, collection rounds, submissions, boundary definitions,
factor assignments, consolidation data, allocation configs, peer groups,
comparison results, completion data, and quality assessments.

All numeric values use Decimal for deterministic arithmetic.
"""

import sys
import os
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
    MultiSitePackConfig,
    SiteRegistryConfig,
    DataCollectionConfig,
    BoundaryConfig,
    RegionalFactorConfig,
    ConsolidationConfig,
    AllocationConfig as AllocCfg,
    ComparisonConfig,
    CompletionConfig,
    QualityConfig,
    ReportingConfig,
    SecurityConfig,
    PerformanceConfig,
    IntegrationConfig,
    AlertConfig,
    MigrationConfig,
    FacilityType as CfgFacilityType,
    FacilityLifecycle,
    ConsolidationApproach,
    OwnershipType,
    CollectionPeriodType,
    SubmissionStatus,
    DataEntryMode,
    AllocationMethod as CfgAllocationMethod,
    LandlordTenantSplit as CfgLandlordTenantSplit,
    CogenerationType,
    FactorTier,
    FactorSource,
    QualityDimension,
    QualityScore,
    ComparisonKPI,
    ReportType,
    ExportFormat,
    AlertType,
    AVAILABLE_PRESETS,
    DEFAULT_FACILITY_TYPES,
    DEFAULT_QUALITY_WEIGHTS,
    DEFAULT_ALLOCATION_PRIORITIES,
    CONSOLIDATION_APPROACH_GUIDANCE,
    REGIONAL_FACTOR_DATABASES,
    load_preset,
    validate_config,
    get_default_config,
    list_available_presets,
    get_facility_type_defaults,
    get_consolidation_guidance,
    get_regional_factor_database,
    get_quality_weights,
    get_allocation_priorities,
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
    return "ORG-MULTISITE-001"


@pytest.fixture
def reporting_year():
    """Standard test reporting year."""
    return 2026


@pytest.fixture
def base_year():
    """Standard test base year."""
    return 2020


# ---------------------------------------------------------------------------
# Pack config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    """Complete PackConfig dictionary for testing."""
    return {
        "company_name": "GreenTest Manufacturing GmbH",
        "consolidation_approach": "OPERATIONAL_CONTROL",
        "reporting_year": 2026,
        "base_year": 2020,
        "country": "DE",
        "total_sites": 5,
        "scopes_in_scope": ["SCOPE_1", "SCOPE_2"],
        "site_registry": {
            "max_sites": 500,
            "facility_types_enabled": [
                "MANUFACTURING", "OFFICE", "WAREHOUSE", "RETAIL", "DATA_CENTER",
            ],
            "lifecycle_tracking": True,
            "grouping_enabled": True,
        },
        "data_collection": {
            "collection_period": "MONTHLY",
            "data_entry_modes": ["MANUAL", "SPREADSHEET_UPLOAD", "API_PUSH"],
            "validation_strictness": "STANDARD",
            "estimation_allowed": True,
        },
        "boundary": {
            "consolidation_approach": "OPERATIONAL_CONTROL",
            "materiality_threshold": Decimal("0.05"),
            "de_minimis_threshold": Decimal("0.01"),
        },
        "consolidation": {
            "elimination_enabled": True,
            "equity_adjustment_enabled": True,
            "completeness_threshold": Decimal("0.95"),
            "reconciliation_tolerance": Decimal("0.01"),
        },
        "allocation": {
            "default_method": "FLOOR_AREA",
            "shared_services_enabled": True,
        },
        "comparison": {
            "default_kpi": "EMISSIONS_PER_M2",
            "peer_group_min_size": 3,
        },
        "completion": {
            "completeness_target": Decimal("0.95"),
        },
        "quality": {
            "minimum_quality_score": 3,
        },
        "reporting": {
            "default_format": "HTML",
        },
    }


@pytest.fixture
def default_pack_config():
    """Default PackConfig with all defaults."""
    return PackConfig()


@pytest.fixture
def default_mgmt_config():
    """Default MultiSitePackConfig with all defaults."""
    return MultiSitePackConfig()


# ---------------------------------------------------------------------------
# Site record fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_site_record():
    """Single manufacturing site record data dict for SiteRegistryEngine."""
    return {
        "site_code": "US-CHI-MFG-01",
        "site_name": "Chicago Manufacturing Plant",
        "facility_type": "MANUFACTURING",
        "legal_entity_id": "LE-001",
        "country": "US",
        "region": "Illinois",
        "city": "Chicago",
        "business_unit": "North America",
        "characteristics": {
            "floor_area_m2": Decimal("10000"),
            "headcount": 500,
            "operating_hours_per_year": 6000,
            "production_output": Decimal("50000"),
            "production_unit": "tonnes",
            "grid_region": "RFC_WEST",
        },
    }


@pytest.fixture
def sample_site_records():
    """Five diverse site records across different facility types and countries."""
    return [
        {
            "site_code": "US-CHI-MFG-01",
            "site_name": "Chicago Manufacturing Plant",
            "facility_type": "MANUFACTURING",
            "legal_entity_id": "LE-001",
            "country": "US",
            "region": "Illinois",
            "city": "Chicago",
            "business_unit": "North America",
            "characteristics": {
                "floor_area_m2": Decimal("25000"),
                "headcount": 350,
                "operating_hours_per_year": 6000,
                "production_output": Decimal("75000"),
                "production_unit": "tonnes",
                "grid_region": "RFC_WEST",
            },
        },
        {
            "site_code": "GB-LDN-OFF-01",
            "site_name": "London Head Office",
            "facility_type": "OFFICE",
            "legal_entity_id": "LE-002",
            "country": "GB",
            "region": "London",
            "city": "London",
            "business_unit": "Europe HQ",
            "characteristics": {
                "floor_area_m2": Decimal("3000"),
                "headcount": 150,
                "operating_hours_per_year": 2500,
            },
        },
        {
            "site_code": "DE-FRA-WH-01",
            "site_name": "Frankfurt Distribution Warehouse",
            "facility_type": "WAREHOUSE",
            "legal_entity_id": "LE-003",
            "country": "DE",
            "region": "Hessen",
            "city": "Frankfurt",
            "business_unit": "Europe Logistics",
            "characteristics": {
                "floor_area_m2": Decimal("8000"),
                "headcount": 60,
                "operating_hours_per_year": 4500,
            },
        },
        {
            "site_code": "US-NYC-RET-01",
            "site_name": "New York Flagship Store",
            "facility_type": "RETAIL",
            "legal_entity_id": "LE-001",
            "country": "US",
            "region": "New York",
            "city": "New York",
            "business_unit": "North America",
            "characteristics": {
                "floor_area_m2": Decimal("1500"),
                "headcount": 30,
                "operating_hours_per_year": 3500,
            },
        },
        {
            "site_code": "DE-BER-DC-01",
            "site_name": "Berlin Data Centre",
            "facility_type": "DATA_CENTER",
            "legal_entity_id": "LE-003",
            "country": "DE",
            "region": "Berlin",
            "city": "Berlin",
            "business_unit": "IT Infrastructure",
            "characteristics": {
                "floor_area_m2": Decimal("5000"),
                "headcount": 25,
                "operating_hours_per_year": 8760,
            },
        },
    ]


# ---------------------------------------------------------------------------
# Facility characteristics fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_characteristics():
    """FacilityCharacteristics-compatible dict for a manufacturing site."""
    return {
        "floor_area_m2": Decimal("10000"),
        "headcount": 500,
        "operating_hours_per_year": 6000,
        "production_output": Decimal("50000"),
        "production_unit": "tonnes",
        "grid_region": "RFC_WEST",
        "climate_zone": "Dfa",
        "electricity_provider": "ComEd",
        "gas_provider": "Peoples Gas",
    }


# ---------------------------------------------------------------------------
# Data collection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_collection_round():
    """Annual data collection round for 5 sites."""
    return {
        "round_id": "ROUND-2026-ANNUAL",
        "period_type": "ANNUAL",
        "reporting_year": 2026,
        "start_date": date(2026, 1, 1),
        "end_date": date(2026, 12, 31),
        "deadline": date(2027, 2, 28),
        "site_ids": [
            "site-001", "site-002", "site-003", "site-004", "site-005",
        ],
        "status": "OPEN",
    }


@pytest.fixture
def sample_data_entries():
    """List of DataEntry items with Decimal values for a single site submission."""
    return [
        {
            "entry_id": "entry-001",
            "source_type": "ELECTRICITY",
            "activity_data": Decimal("500000"),
            "activity_unit": "kWh",
            "emission_factor": Decimal("0.000417"),
            "emission_factor_unit": "tCO2e/kWh",
            "calculated_emissions": Decimal("208.500"),
            "scope": "SCOPE_2",
            "data_quality_score": 2,
        },
        {
            "entry_id": "entry-002",
            "source_type": "NATURAL_GAS",
            "activity_data": Decimal("100000"),
            "activity_unit": "m3",
            "emission_factor": Decimal("0.002020"),
            "emission_factor_unit": "tCO2e/m3",
            "calculated_emissions": Decimal("202.000"),
            "scope": "SCOPE_1",
            "data_quality_score": 2,
        },
        {
            "entry_id": "entry-003",
            "source_type": "DIESEL",
            "activity_data": Decimal("15000"),
            "activity_unit": "litres",
            "emission_factor": Decimal("0.002680"),
            "emission_factor_unit": "tCO2e/litre",
            "calculated_emissions": Decimal("40.200"),
            "scope": "SCOPE_1",
            "data_quality_score": 3,
        },
        {
            "entry_id": "entry-004",
            "source_type": "WASTE",
            "activity_data": Decimal("500"),
            "activity_unit": "tonnes",
            "emission_factor": Decimal("0.021000"),
            "emission_factor_unit": "tCO2e/tonne",
            "calculated_emissions": Decimal("10.500"),
            "scope": "SCOPE_3",
            "data_quality_score": 4,
        },
        {
            "entry_id": "entry-005",
            "source_type": "WATER",
            "activity_data": Decimal("25000"),
            "activity_unit": "m3",
            "emission_factor": Decimal("0.000344"),
            "emission_factor_unit": "tCO2e/m3",
            "calculated_emissions": Decimal("8.600"),
            "scope": "SCOPE_3",
            "data_quality_score": 4,
        },
    ]


@pytest.fixture
def sample_submission(sample_data_entries):
    """A site submission with 5 data entries."""
    return {
        "submission_id": "SUB-2026-SITE001",
        "site_id": "site-001",
        "round_id": "ROUND-2026-ANNUAL",
        "submitted_by": "site-manager@greentest.com",
        "submitted_at": _utcnow().isoformat(),
        "status": "SUBMITTED",
        "data_entries": sample_data_entries,
        "total_scope1": Decimal("242.200"),
        "total_scope2": Decimal("208.500"),
        "total_scope3": Decimal("19.100"),
        "grand_total": Decimal("469.800"),
        "notes": "Annual data for Chicago Manufacturing 2026",
    }


# ---------------------------------------------------------------------------
# Boundary fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_boundary():
    """BoundaryDefinition for equity share approach with 5 sites."""
    return {
        "boundary_id": "BND-2026-001",
        "organisation_id": "ORG-MULTISITE-001",
        "reporting_year": 2026,
        "consolidation_approach": "EQUITY_SHARE",
        "sites": [
            {"site_id": "site-001", "ownership_pct": Decimal("100.00"), "is_included": True},
            {"site_id": "site-002", "ownership_pct": Decimal("100.00"), "is_included": True},
            {"site_id": "site-003", "ownership_pct": Decimal("75.00"), "is_included": True},
            {"site_id": "site-004", "ownership_pct": Decimal("50.00"), "is_included": True},
            {"site_id": "site-005", "ownership_pct": Decimal("100.00"), "is_included": True},
        ],
        "materiality_threshold": Decimal("0.05"),
        "de_minimis_threshold": Decimal("0.01"),
        "is_locked": False,
    }


@pytest.fixture
def sample_entity_ownership():
    """3-level ownership hierarchy: parent + 2 subsidiaries."""
    return {
        "parent": {
            "entity_id": "LE-PARENT",
            "entity_name": "GreenTest Holdings AG",
            "ownership_pct": Decimal("100.00"),
            "sites": ["site-001", "site-002"],
        },
        "subsidiaries": [
            {
                "entity_id": "LE-SUB-001",
                "entity_name": "GreenTest Manufacturing GmbH",
                "parent_id": "LE-PARENT",
                "ownership_pct": Decimal("75.00"),
                "sites": ["site-003"],
            },
            {
                "entity_id": "LE-SUB-002",
                "entity_name": "GreenTest Logistics Ltd",
                "parent_id": "LE-PARENT",
                "ownership_pct": Decimal("50.00"),
                "sites": ["site-004", "site-005"],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Regional factor fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_factor_assignments():
    """Emission factor assignments for 5 sites with electricity grid factors."""
    return [
        {
            "site_id": "site-001",
            "country": "US",
            "source_type": "ELECTRICITY",
            "factor_value": Decimal("0.000417"),
            "factor_unit": "tCO2e/kWh",
            "factor_source": "EPA_EGRID",
            "factor_tier": "TIER_1_REGIONAL",
            "grid_region": "RFC_WEST",
            "year": 2026,
        },
        {
            "site_id": "site-002",
            "country": "GB",
            "source_type": "ELECTRICITY",
            "factor_value": Decimal("0.000207"),
            "factor_unit": "tCO2e/kWh",
            "factor_source": "DEFRA",
            "factor_tier": "TIER_2_NATIONAL",
            "grid_region": "UK_GRID",
            "year": 2026,
        },
        {
            "site_id": "site-003",
            "country": "DE",
            "source_type": "ELECTRICITY",
            "factor_value": Decimal("0.000380"),
            "factor_unit": "tCO2e/kWh",
            "factor_source": "UBA",
            "factor_tier": "TIER_2_NATIONAL",
            "grid_region": "DE_GRID",
            "year": 2026,
        },
        {
            "site_id": "site-004",
            "country": "US",
            "source_type": "ELECTRICITY",
            "factor_value": Decimal("0.000350"),
            "factor_unit": "tCO2e/kWh",
            "factor_source": "EPA_EGRID",
            "factor_tier": "TIER_1_REGIONAL",
            "grid_region": "NPCC_NY",
            "year": 2026,
        },
        {
            "site_id": "site-005",
            "country": "DE",
            "source_type": "ELECTRICITY",
            "factor_value": Decimal("0.000380"),
            "factor_unit": "tCO2e/kWh",
            "factor_source": "UBA",
            "factor_tier": "TIER_2_NATIONAL",
            "grid_region": "DE_GRID",
            "year": 2026,
        },
    ]


# ---------------------------------------------------------------------------
# Consolidation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_site_totals():
    """Five SiteTotal dicts with S1, S2, S3 in tCO2e."""
    return [
        {
            "site_id": "site-001",
            "site_name": "Chicago Manufacturing Plant",
            "scope1_tco2e": Decimal("5000.00"),
            "scope2_tco2e": Decimal("3000.00"),
            "scope3_tco2e": Decimal("10000.00"),
            "total_tco2e": Decimal("18000.00"),
            "ownership_pct": Decimal("100.00"),
        },
        {
            "site_id": "site-002",
            "site_name": "London Head Office",
            "scope1_tco2e": Decimal("100.00"),
            "scope2_tco2e": Decimal("350.00"),
            "scope3_tco2e": Decimal("800.00"),
            "total_tco2e": Decimal("1250.00"),
            "ownership_pct": Decimal("100.00"),
        },
        {
            "site_id": "site-003",
            "site_name": "Frankfurt Distribution Warehouse",
            "scope1_tco2e": Decimal("800.00"),
            "scope2_tco2e": Decimal("500.00"),
            "scope3_tco2e": Decimal("2000.00"),
            "total_tco2e": Decimal("3300.00"),
            "ownership_pct": Decimal("75.00"),
        },
        {
            "site_id": "site-004",
            "site_name": "New York Flagship Store",
            "scope1_tco2e": Decimal("150.00"),
            "scope2_tco2e": Decimal("50.00"),
            "scope3_tco2e": Decimal("200.00"),
            "total_tco2e": Decimal("400.00"),
            "ownership_pct": Decimal("50.00"),
        },
        {
            "site_id": "site-005",
            "site_name": "Berlin Data Centre",
            "scope1_tco2e": Decimal("200.00"),
            "scope2_tco2e": Decimal("2500.00"),
            "scope3_tco2e": Decimal("5000.00"),
            "total_tco2e": Decimal("7700.00"),
            "ownership_pct": Decimal("100.00"),
        },
    ]


@pytest.fixture
def sample_consolidation_run(sample_site_totals):
    """Full consolidation run result."""
    return {
        "run_id": "CONS-2026-001",
        "organisation_id": "ORG-MULTISITE-001",
        "reporting_year": 2026,
        "consolidation_approach": "EQUITY_SHARE",
        "site_totals": sample_site_totals,
        "consolidated_scope1": Decimal("5712.50"),
        "consolidated_scope2": Decimal("5587.50"),
        "consolidated_scope3": Decimal("16200.00"),
        "consolidated_total": Decimal("27500.00"),
        "completeness_pct": Decimal("100.00"),
        "sites_included": 5,
        "sites_excluded": 0,
    }


# ---------------------------------------------------------------------------
# Allocation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_allocation_config():
    """Shared services allocation config by floor area."""
    return {
        "config_id": "ALLOC-CFG-001",
        "allocation_type": "SHARED_SERVICES",
        "method": "FLOOR_AREA",
        "source_site_id": "site-HQ",
        "target_site_ids": ["site-001", "site-002", "site-003", "site-004", "site-005"],
        "allocation_keys": {
            "site-001": Decimal("25000"),
            "site-002": Decimal("3000"),
            "site-003": Decimal("8000"),
            "site-004": Decimal("1500"),
            "site-005": Decimal("5000"),
        },
        "source_emissions": Decimal("500.00"),
    }


# ---------------------------------------------------------------------------
# Comparison / benchmarking fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_peer_group():
    """Peer group of 5 manufacturing sites with KPI data."""
    return {
        "peer_group_id": "PG-MFG-001",
        "peer_group_name": "Manufacturing Sites",
        "facility_type": "MANUFACTURING",
        "sites": [
            {
                "site_id": "site-001",
                "site_name": "Chicago Plant",
                "floor_area_m2": Decimal("25000"),
                "headcount": 350,
                "total_tco2e": Decimal("18000.00"),
                "emissions_per_m2": Decimal("0.72"),
                "emissions_per_fte": Decimal("51.43"),
            },
            {
                "site_id": "site-A",
                "site_name": "Dallas Plant",
                "floor_area_m2": Decimal("20000"),
                "headcount": 280,
                "total_tco2e": Decimal("12000.00"),
                "emissions_per_m2": Decimal("0.60"),
                "emissions_per_fte": Decimal("42.86"),
            },
            {
                "site_id": "site-B",
                "site_name": "Detroit Plant",
                "floor_area_m2": Decimal("30000"),
                "headcount": 400,
                "total_tco2e": Decimal("25000.00"),
                "emissions_per_m2": Decimal("0.83"),
                "emissions_per_fte": Decimal("62.50"),
            },
            {
                "site_id": "site-C",
                "site_name": "Phoenix Plant",
                "floor_area_m2": Decimal("15000"),
                "headcount": 200,
                "total_tco2e": Decimal("10000.00"),
                "emissions_per_m2": Decimal("0.67"),
                "emissions_per_fte": Decimal("50.00"),
            },
            {
                "site_id": "site-D",
                "site_name": "Portland Plant",
                "floor_area_m2": Decimal("18000"),
                "headcount": 250,
                "total_tco2e": Decimal("8000.00"),
                "emissions_per_m2": Decimal("0.44"),
                "emissions_per_fte": Decimal("32.00"),
            },
        ],
    }


@pytest.fixture
def sample_comparison_result(sample_peer_group):
    """Comparison result with rankings."""
    sites = sample_peer_group["sites"]
    ranked = sorted(sites, key=lambda s: s["emissions_per_m2"])
    return {
        "comparison_id": "CMP-2026-001",
        "peer_group_id": "PG-MFG-001",
        "kpi": "EMISSIONS_PER_M2",
        "rankings": [
            {"rank": i + 1, "site_id": s["site_id"], "kpi_value": s["emissions_per_m2"]}
            for i, s in enumerate(ranked)
        ],
        "statistics": {
            "mean": Decimal("0.652"),
            "median": Decimal("0.67"),
            "std_dev": Decimal("0.142"),
            "min": Decimal("0.44"),
            "max": Decimal("0.83"),
            "p25": Decimal("0.60"),
            "p75": Decimal("0.72"),
        },
        "best_practice_site": "site-D",
        "worst_performer_site": "site-B",
    }


# ---------------------------------------------------------------------------
# Completion fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_completion_result():
    """Portfolio completion result at 85% completeness."""
    return {
        "assessment_id": "COMP-2026-001",
        "reporting_year": 2026,
        "total_sites": 10,
        "sites_submitted": 8,
        "sites_approved": 7,
        "sites_pending": 2,
        "sites_overdue": 1,
        "completeness_pct": Decimal("85.00"),
        "target_pct": Decimal("95.00"),
        "gap_pct": Decimal("10.00"),
        "missing_sites": ["site-009", "site-010"],
        "overdue_sites": ["site-008"],
        "estimated_missing_tco2e": Decimal("1500.00"),
    }


# ---------------------------------------------------------------------------
# Quality assessment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_quality_assessments():
    """Quality assessments for 5 sites with varying quality scores."""
    return [
        {
            "site_id": "site-001",
            "site_name": "Chicago Manufacturing Plant",
            "overall_score": 2,
            "dimension_scores": {
                "ACCURACY": Decimal("1.5"),
                "COMPLETENESS": Decimal("2.0"),
                "CONSISTENCY": Decimal("2.0"),
                "TIMELINESS": Decimal("1.0"),
                "METHODOLOGY": Decimal("2.0"),
                "DOCUMENTATION": Decimal("3.0"),
            },
            "pcaf_equivalent": 2,
            "data_sources": ["meter_reads", "invoices"],
        },
        {
            "site_id": "site-002",
            "site_name": "London Head Office",
            "overall_score": 1,
            "dimension_scores": {
                "ACCURACY": Decimal("1.0"),
                "COMPLETENESS": Decimal("1.0"),
                "CONSISTENCY": Decimal("1.0"),
                "TIMELINESS": Decimal("1.0"),
                "METHODOLOGY": Decimal("1.0"),
                "DOCUMENTATION": Decimal("2.0"),
            },
            "pcaf_equivalent": 1,
            "data_sources": ["verified_meter_reads", "audit_invoices"],
        },
        {
            "site_id": "site-003",
            "site_name": "Frankfurt Warehouse",
            "overall_score": 3,
            "dimension_scores": {
                "ACCURACY": Decimal("3.0"),
                "COMPLETENESS": Decimal("3.0"),
                "CONSISTENCY": Decimal("3.0"),
                "TIMELINESS": Decimal("2.0"),
                "METHODOLOGY": Decimal("3.0"),
                "DOCUMENTATION": Decimal("4.0"),
            },
            "pcaf_equivalent": 3,
            "data_sources": ["estimated"],
        },
        {
            "site_id": "site-004",
            "site_name": "New York Flagship Store",
            "overall_score": 4,
            "dimension_scores": {
                "ACCURACY": Decimal("4.0"),
                "COMPLETENESS": Decimal("4.0"),
                "CONSISTENCY": Decimal("3.0"),
                "TIMELINESS": Decimal("4.0"),
                "METHODOLOGY": Decimal("4.0"),
                "DOCUMENTATION": Decimal("5.0"),
            },
            "pcaf_equivalent": 4,
            "data_sources": ["extrapolated"],
        },
        {
            "site_id": "site-005",
            "site_name": "Berlin Data Centre",
            "overall_score": 2,
            "dimension_scores": {
                "ACCURACY": Decimal("2.0"),
                "COMPLETENESS": Decimal("1.0"),
                "CONSISTENCY": Decimal("2.0"),
                "TIMELINESS": Decimal("2.0"),
                "METHODOLOGY": Decimal("2.0"),
                "DOCUMENTATION": Decimal("3.0"),
            },
            "pcaf_equivalent": 2,
            "data_sources": ["meter_reads", "utility_bills"],
        },
    ]
