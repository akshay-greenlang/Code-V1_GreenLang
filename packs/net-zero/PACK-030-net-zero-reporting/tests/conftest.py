# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-030 Net Zero Reporting Pack.
================================================================

Provides pytest fixtures for all 10 engines, 8 workflows, 15 templates,
12 integrations, 8 configuration presets, and comprehensive test data
builders for multi-framework reporting, data aggregation, narrative
generation, framework mapping, XBRL tagging, dashboard generation,
assurance packaging, report compilation, validation, translation,
and format rendering.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (10 reporting engines)
    - Workflow instantiation (8 workflows)
    - PACK-021/022/028/029 mock data (prerequisite packs)
    - GL-SBTi/CDP/TCFD/GHG APP mock responses
    - Multi-framework report data (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD)
    - Narrative generation test data (4 languages)
    - Framework mapping test data (bidirectional)
    - XBRL/iXBRL tagging test data (SEC, CSRD taxonomies)
    - Dashboard configuration data (executive, investor, regulator)
    - Assurance evidence bundle data (ISAE 3410)
    - Report compilation data (sections, branding, TOC)
    - Validation schemas (7 frameworks)
    - Translation test data (EN/DE/FR/ES)
    - Format rendering data (PDF/HTML/Excel/JSON/XBRL/iXBRL)
    - Database session mocking (async PostgreSQL)
    - Redis cache mocking
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-030 Net Zero Reporting Pack
Tests:   conftest.py (~1,500 lines)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Path setup -- ensure pack root is importable
# ---------------------------------------------------------------------------

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

_REPO_ROOT = _PACK_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ENGINES_DIR = _PACK_ROOT / "engines"
WORKFLOWS_DIR = _PACK_ROOT / "workflows"
TEMPLATES_DIR = _PACK_ROOT / "templates"
INTEGRATIONS_DIR = _PACK_ROOT / "integrations"
CONFIG_DIR = _PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


# ---------------------------------------------------------------------------
# Engine imports (lazy, with graceful fallback)
# ---------------------------------------------------------------------------

try:
    from engines.data_aggregation_engine import (
        DataAggregationEngine,
        DataAggregationInput,
        DataAggregationResult,
        DataSourceType,
        ReconciliationItem,
        LineageGraph,
    )
    _HAS_DATA_AGGREGATION = True
except ImportError:
    _HAS_DATA_AGGREGATION = False

try:
    from engines.narrative_generation_engine import (
        NarrativeGenerationEngine,
        NarrativeGenerationInput,
        NarrativeGenerationResult,
        Citation,
        ConsistencyCheckResult,
        NarrativeLanguage,
    )
    _HAS_NARRATIVE_GENERATION = True
except ImportError:
    _HAS_NARRATIVE_GENERATION = False

try:
    from engines.framework_mapping_engine import (
        FrameworkMappingEngine,
        FrameworkMappingInput,
        FrameworkMappingResult,
        MetricMapping,
        MappingDirection,
        MappingConflict,
    )
    _HAS_FRAMEWORK_MAPPING = True
except ImportError:
    _HAS_FRAMEWORK_MAPPING = False

try:
    from engines.xbrl_tagging_engine import (
        XBRLTaggingEngine,
        XBRLTaggingInput,
        XBRLTaggingResult,
        XBRLTag,
        XBRLTaxonomy,
        TaxonomyValidationIssue,
    )
    _HAS_XBRL_TAGGING = True
except ImportError:
    _HAS_XBRL_TAGGING = False

try:
    from engines.dashboard_generation_engine import (
        DashboardGenerationEngine,
        DashboardGenerationInput,
        DashboardGenerationResult,
        DashboardType,
        BrandingConfig as DashboardBrandingConfig,
        FrameworkStatus,
    )
    _HAS_DASHBOARD_GENERATION = True
except ImportError:
    _HAS_DASHBOARD_GENERATION = False

try:
    from engines.assurance_packaging_engine import (
        AssurancePackagingEngine,
        AssurancePackagingInput,
        AssurancePackagingResult,
        EvidenceItem,
        ControlMatrixEntry,
        ProvenanceRecord,
    )
    _HAS_ASSURANCE_PACKAGING = True
except ImportError:
    _HAS_ASSURANCE_PACKAGING = False

try:
    from engines.report_compilation_engine import (
        ReportCompilationEngine,
        ReportCompilationInput,
        ReportCompilationResult,
        ReportMetric,
        ReportBranding,
        TableOfContents,
    )
    _HAS_REPORT_COMPILATION = True
except ImportError:
    _HAS_REPORT_COMPILATION = False

try:
    from engines.validation_engine import (
        ValidationEngine,
        ValidationInput,
        ValidationResult,
        SchemaValidationResult,
        CompletenessResult,
        ConsistencyResult,
    )
    _HAS_VALIDATION = True
except ImportError:
    _HAS_VALIDATION = False

try:
    from engines.translation_engine import (
        TranslationEngine,
        TranslationInput,
        TranslationResult,
        TerminologyReport,
        TranslationQualityTier,
        CitationReport,
    )
    _HAS_TRANSLATION = True
except ImportError:
    _HAS_TRANSLATION = False

try:
    from engines.format_rendering_engine import (
        FormatRenderingEngine,
        RenderInput,
        RenderResult,
        OutputFormat,
        BrandingConfig,
        ChartConfig,
    )
    _HAS_FORMAT_RENDERING = True
except ImportError:
    _HAS_FORMAT_RENDERING = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRAMEWORKS = ["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"]

FRAMEWORK_IDS = ["sbti", "cdp", "tcfd", "gri", "issb", "sec", "csrd"]

OUTPUT_FORMATS = ["PDF", "HTML", "Excel", "JSON", "XBRL", "iXBRL"]

LANGUAGES = ["en", "de", "fr", "es"]

SCOPES = ["scope_1", "scope_2", "scope_3"]

STAKEHOLDER_VIEWS = ["executive", "investor", "regulator", "customer", "employee"]

REPORT_STATUSES = ["draft", "review", "approved", "published"]

EVIDENCE_TYPES = ["provenance", "lineage", "methodology", "control"]

VALIDATION_SEVERITIES = ["critical", "high", "medium", "low"]

CDP_MODULES = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"]

TCFD_PILLARS = ["governance", "strategy", "risk_management", "metrics_targets"]

ESRS_E1_DISCLOSURES = ["E1-1", "E1-2", "E1-3", "E1-4", "E1-5", "E1-6", "E1-7", "E1-8", "E1-9"]

GRI_305_DISCLOSURES = ["305-1", "305-2", "305-3", "305-4", "305-5", "305-6", "305-7"]

PRESET_NAMES = [
    "csrd_focus", "cdp_alist", "tcfd_investor", "sbti_validation",
    "sec_10k", "multi_framework", "investor_relations", "assurance_ready",
]

DATA_SOURCES = [
    "PACK-021", "PACK-022", "PACK-028", "PACK-029",
    "GL-SBTi-APP", "GL-CDP-APP", "GL-TCFD-APP", "GL-GHG-APP",
]

XBRL_TAXONOMY_FRAMEWORKS = ["SEC", "CSRD"]


# ---------------------------------------------------------------------------
# Helper: Decimal assertion
# ---------------------------------------------------------------------------


def assert_decimal_close(
    actual: Decimal,
    expected: Decimal,
    tolerance: Decimal = Decimal("0.01"),
    msg: str = "",
) -> None:
    """Assert two Decimal values are within tolerance."""
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Decimal mismatch{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, diff={diff}, tol={tolerance}"
    )


def assert_decimal_equal(actual: Decimal, expected: Decimal, msg: str = "") -> None:
    """Assert two Decimal values are exactly equal."""
    assert actual == expected, (
        f"Decimal inequality{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}"
    )


def assert_decimal_positive(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is positive."""
    assert value > Decimal("0"), (
        f"Expected positive Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_decimal_non_negative(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is non-negative."""
    assert value >= Decimal("0"), (
        f"Expected non-negative Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_percentage_range(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is between 0 and 100."""
    assert Decimal("0") <= value <= Decimal("100"), (
        f"Percentage out of range{' (' + msg + ')' if msg else ''}: {value}"
    )


def assert_provenance_hash(result: Any) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash."""
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"


def assert_processing_time(result: Any, max_ms: float = 60000.0) -> None:
    """Assert processing time is within acceptable range."""
    assert hasattr(result, "processing_time_ms"), "Result missing processing_time_ms"
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    assert result.processing_time_ms < max_ms, (
        f"Processing time {result.processing_time_ms}ms exceeds {max_ms}ms"
    )


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_sha256(data: str, expected_hash: str) -> bool:
    """Verify SHA-256 hash matches expected."""
    return compute_sha256(data) == expected_hash


@contextmanager
def timed_block(label: str = "", max_seconds: float = 30.0):
    """Context manager that asserts a block completes within max_seconds."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


def assert_valid_uuid(value: str, msg: str = "") -> None:
    """Assert a string is a valid UUID."""
    try:
        uuid.UUID(str(value))
    except ValueError:
        raise AssertionError(
            f"Invalid UUID{' (' + msg + ')' if msg else ''}: {value}"
        )


def assert_valid_json(value: str, msg: str = "") -> None:
    """Assert a string is valid JSON."""
    try:
        json.loads(value)
    except (json.JSONDecodeError, TypeError):
        raise AssertionError(
            f"Invalid JSON{' (' + msg + ')' if msg else ''}: {value[:100]}"
        )


def assert_html_contains(html: str, *tags: str) -> None:
    """Assert HTML contains expected tags."""
    for tag in tags:
        assert f"<{tag}" in html or f"</{tag}>" in html, (
            f"HTML missing expected tag: {tag}"
        )


def assert_framework_valid(framework: str) -> None:
    """Assert framework is a recognized framework."""
    assert framework in FRAMEWORKS or framework.lower() in FRAMEWORK_IDS, (
        f"Unknown framework: {framework}"
    )


# ---------------------------------------------------------------------------
# Fixtures -- Organization Data
# ---------------------------------------------------------------------------


@pytest.fixture
def organization_id() -> str:
    """Return a test organization UUID."""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def reporting_period() -> Dict[str, str]:
    """Return a test reporting period."""
    return {
        "start": "2024-01-01",
        "end": "2024-12-31",
        "fiscal_year": "2024",
    }


@pytest.fixture
def organization_config() -> Dict[str, Any]:
    """Build organization reporting configuration."""
    return {
        "organization_id": "550e8400-e29b-41d4-a716-446655440000",
        "name": "GreenCorp Industries",
        "sector": "manufacturing",
        "jurisdiction": "EU",
        "frameworks_enabled": FRAMEWORKS,
        "languages": ["en", "de"],
        "branding": {
            "logo_path": "/assets/logos/greencorp.png",
            "primary_color": "#1E3A8A",
            "secondary_color": "#3B82F6",
            "font_family": "Arial, sans-serif",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Emissions Data (shared across frameworks)
# ---------------------------------------------------------------------------


@pytest.fixture
def emissions_data_2024() -> Dict[str, Any]:
    """Build GHG emissions data for reporting year 2024."""
    return {
        "reporting_year": 2024,
        "base_year": 2019,
        "scope_1_tco2e": Decimal("107500"),
        "scope_2_location_tco2e": Decimal("73100"),
        "scope_2_market_tco2e": Decimal("67080"),
        "scope_3_tco2e": Decimal("405000"),
        "scope_3_categories": {
            "cat_01_purchased_goods": Decimal("162000"),
            "cat_02_capital_goods": Decimal("31500"),
            "cat_03_fuel_energy": Decimal("25200"),
            "cat_04_upstream_transport": Decimal("37800"),
            "cat_05_waste": Decimal("10800"),
            "cat_06_business_travel": Decimal("14400"),
            "cat_07_employee_commuting": Decimal("19800"),
            "cat_08_upstream_leased": Decimal("7200"),
            "cat_09_downstream_transport": Decimal("22500"),
            "cat_10_processing": Decimal("13500"),
            "cat_11_use_of_sold": Decimal("31500"),
            "cat_12_end_of_life": Decimal("10800"),
            "cat_13_downstream_leased": Decimal("4500"),
            "cat_14_franchises": Decimal("2700"),
            "cat_15_investments": Decimal("10800"),
        },
        "total_scope_12_tco2e": Decimal("174580"),
        "total_scope_123_tco2e": Decimal("579580"),
        "revenue_m_usd": Decimal("2800"),
        "headcount": 12500,
        "intensity_tco2e_per_m_usd": Decimal("207.0"),
        "data_quality_score": Decimal("0.91"),
        "verified": True,
        "provenance_hash": compute_sha256("emissions_data_2024_greencorp"),
    }


@pytest.fixture
def baseline_2019() -> Dict[str, Any]:
    """Build PACK-021 baseline data for base year 2019."""
    return {
        "entity_name": "GreenCorp Industries",
        "base_year": 2019,
        "scope_1_tco2e": Decimal("125000"),
        "scope_2_location_tco2e": Decimal("85000"),
        "scope_2_market_tco2e": Decimal("78000"),
        "scope_3_tco2e": Decimal("450000"),
        "total_scope_12_tco2e": Decimal("203000"),
        "total_scope_123_tco2e": Decimal("653000"),
        "revenue_m_usd": Decimal("2500"),
        "headcount": 12000,
        "data_quality_score": Decimal("0.85"),
        "provenance_hash": compute_sha256("baseline_2019_greencorp"),
    }


@pytest.fixture
def target_data() -> Dict[str, Any]:
    """Build net-zero target data."""
    return {
        "target_type": "net_zero",
        "ambition": "1.5C",
        "near_term_target_year": 2030,
        "long_term_target_year": 2050,
        "scope_12_near_term_reduction_pct": Decimal("42"),
        "scope_3_near_term_reduction_pct": Decimal("25"),
        "scope_12_long_term_reduction_pct": Decimal("90"),
        "sbti_validated": True,
        "provenance_hash": compute_sha256("target_data_greencorp"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Framework-Specific Report Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sbti_report_data() -> Dict[str, Any]:
    """Build SBTi annual progress report data."""
    return {
        "framework": "SBTi",
        "target_type": "absolute",
        "base_year": 2019,
        "target_year": 2030,
        "base_year_emissions_tco2e": Decimal("203000"),
        "current_year_emissions_tco2e": Decimal("174580"),
        "reduction_achieved_pct": Decimal("14.0"),
        "reduction_required_pct": Decimal("42.0"),
        "on_track": True,
        "variance_tco2e": Decimal("-2500"),
        "next_steps": "Continue renewable procurement and fleet electrification.",
        "provenance_hash": compute_sha256("sbti_report_2024"),
    }


@pytest.fixture
def cdp_questionnaire_data() -> Dict[str, Any]:
    """Build CDP Climate Change questionnaire data (C0-C12)."""
    return {
        "framework": "CDP",
        "year": 2024,
        "modules": {
            "C0": {"organization_name": "GreenCorp Industries", "country": "Germany"},
            "C1": {"board_oversight": True, "management_responsibility": True},
            "C2": {"risk_types_identified": 5, "opportunity_types_identified": 3},
            "C3": {"strategy_influenced_by_climate": True},
            "C4": {
                "target_type": "absolute",
                "base_year": 2019,
                "target_year": 2030,
                "reduction_pct": Decimal("42"),
            },
            "C5": {"methodology": "GHG Protocol Corporate Standard"},
            "C6": {
                "scope_1_tco2e": Decimal("107500"),
                "scope_2_location_tco2e": Decimal("73100"),
                "scope_2_market_tco2e": Decimal("67080"),
            },
            "C7": {"scope_3_total_tco2e": Decimal("405000"), "categories_reported": 15},
            "C8": {"energy_consumption_mwh": Decimal("450000")},
            "C9": {"additional_metrics": True},
            "C10": {"verification_status": "third_party_verified"},
            "C11": {"carbon_pricing_used": True, "internal_price_usd": Decimal("85")},
            "C12": {"engagement_activities": 4},
        },
        "completeness_pct": Decimal("95"),
        "provenance_hash": compute_sha256("cdp_questionnaire_2024"),
    }


@pytest.fixture
def tcfd_report_data() -> Dict[str, Any]:
    """Build TCFD 4-pillar disclosure data."""
    return {
        "framework": "TCFD",
        "pillars": {
            "governance": {
                "board_oversight": "Quarterly climate risk review by board sustainability committee.",
                "management_role": "CSO reports to CEO, chairs Climate Steering Committee.",
            },
            "strategy": {
                "climate_risks": ["Physical: extreme heat", "Transition: carbon pricing"],
                "climate_opportunities": ["Energy efficiency", "Low-carbon products"],
                "scenario_analysis": {
                    "scenarios": ["1.5C (IEA NZE)", "2C (STEPS)", "3C (BAU)"],
                    "financial_impact_m_usd": Decimal("45.5"),
                },
                "resilience_assessment": "Organization is resilient under 1.5C and 2C scenarios.",
            },
            "risk_management": {
                "identification_process": "Annual climate risk assessment using TCFD framework.",
                "assessment_methodology": "Quantitative scenario analysis.",
                "integration_with_enterprise_risk": True,
            },
            "metrics_targets": {
                "scope_1_tco2e": Decimal("107500"),
                "scope_2_tco2e": Decimal("67080"),
                "scope_3_tco2e": Decimal("405000"),
                "targets": [
                    {"scope": "1+2", "year": 2030, "reduction_pct": Decimal("42")},
                    {"scope": "3", "year": 2030, "reduction_pct": Decimal("25")},
                ],
                "intensity_metric": Decimal("207.0"),
            },
        },
        "provenance_hash": compute_sha256("tcfd_report_2024"),
    }


@pytest.fixture
def gri_305_data() -> Dict[str, Any]:
    """Build GRI 305 emissions disclosure data."""
    return {
        "framework": "GRI",
        "disclosures": {
            "305-1": {"direct_emissions_tco2e": Decimal("107500"), "gases": ["CO2", "CH4", "N2O"]},
            "305-2": {"indirect_emissions_tco2e": Decimal("67080"), "method": "market-based"},
            "305-3": {"other_indirect_tco2e": Decimal("405000"), "categories": 15},
            "305-4": {"intensity_ratio": Decimal("207.0"), "denominator": "revenue_m_usd"},
            "305-5": {"reduction_tco2e": Decimal("28420"), "initiatives": 5},
            "305-6": {"ods_tonnes_cfc11eq": Decimal("0.5")},
            "305-7": {"nox_tonnes": Decimal("12.3"), "sox_tonnes": Decimal("8.7")},
        },
        "provenance_hash": compute_sha256("gri_305_2024"),
    }


@pytest.fixture
def issb_s2_data() -> Dict[str, Any]:
    """Build ISSB IFRS S2 climate disclosure data."""
    return {
        "framework": "ISSB",
        "standard": "IFRS S2",
        "governance": {"board_oversight": True, "committee": "Sustainability Committee"},
        "strategy": {"climate_risks": 5, "climate_opportunities": 3, "scenario_analysis": True},
        "risk_management": {"integrated_with_erp": True, "assessment_frequency": "annual"},
        "metrics_targets": {
            "scope_1_tco2e": Decimal("107500"),
            "scope_2_tco2e": Decimal("67080"),
            "scope_3_tco2e": Decimal("405000"),
            "industry_metrics": {"energy_intensity": Decimal("3.6")},
        },
        "provenance_hash": compute_sha256("issb_s2_2024"),
    }


@pytest.fixture
def sec_disclosure_data() -> Dict[str, Any]:
    """Build SEC 10-K climate disclosure data."""
    return {
        "framework": "SEC",
        "filing_type": "10-K",
        "fiscal_year": 2024,
        "items": {
            "item_1": {"climate_risks_in_business": "Material climate risks include..."},
            "item_1a": {"climate_risk_factors": ["Carbon pricing risk", "Physical risk"]},
            "item_7": {"climate_impact_mda": "Climate-related expenditures of $25M..."},
            "reg_sk_1502": {
                "scope_1_tco2e": Decimal("107500"),
                "scope_2_tco2e": Decimal("67080"),
            },
            "reg_sk_1504": {"targets": [{"year": 2030, "reduction_pct": Decimal("42")}]},
        },
        "xbrl_required": True,
        "attestation_level": "limited_assurance",
        "provenance_hash": compute_sha256("sec_disclosure_2024"),
    }


@pytest.fixture
def csrd_e1_data() -> Dict[str, Any]:
    """Build CSRD ESRS E1 Climate Change disclosure data."""
    return {
        "framework": "CSRD",
        "standard": "ESRS E1",
        "disclosures": {
            "E1-1": {"transition_plan": "GreenCorp has a Paris-aligned transition plan..."},
            "E1-2": {"climate_policies": ["Energy efficiency policy", "Renewable procurement"]},
            "E1-3": {"actions_resources": {"capex_m_eur": Decimal("15.0"), "opex_m_eur": Decimal("3.2")}},
            "E1-4": {"targets": [{"scope": "1+2", "year": 2030, "reduction_pct": Decimal("42")}]},
            "E1-5": {"energy_consumption_mwh": Decimal("450000"), "renewable_pct": Decimal("35")},
            "E1-6": {
                "scope_1_tco2e": Decimal("107500"),
                "scope_2_tco2e": Decimal("67080"),
                "scope_3_tco2e": Decimal("405000"),
            },
            "E1-7": {"removals_tco2e": Decimal("0"), "credits_tco2e": Decimal("0")},
            "E1-8": {"internal_carbon_price_eur": Decimal("85")},
            "E1-9": {"anticipated_financial_effects_m_eur": Decimal("45.5")},
        },
        "digital_taxonomy_required": True,
        "provenance_hash": compute_sha256("csrd_e1_2024"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Narrative Data
# ---------------------------------------------------------------------------


@pytest.fixture
def narrative_data() -> Dict[str, Any]:
    """Build narrative generation test data."""
    return {
        "section_type": "governance",
        "framework": "TCFD",
        "language": "en",
        "source_data": {
            "board_oversight": True,
            "committee_name": "Sustainability Committee",
            "meeting_frequency": "quarterly",
            "management_role": "CSO reports directly to CEO",
        },
        "citations": [
            {"id": "CIT-001", "source": "Board minutes Q4 2024", "page": 12},
            {"id": "CIT-002", "source": "Sustainability Policy v3.2", "page": 1},
        ],
        "consistency_requirements": {
            "cross_framework_refs": ["CSRD E1-1", "GRI 2-12"],
        },
    }


@pytest.fixture
def narrative_multilang() -> Dict[str, str]:
    """Build multi-language narrative samples."""
    return {
        "en": "The organization has set science-based targets aligned with the Paris Agreement.",
        "de": "Die Organisation hat wissenschaftsbasierte Ziele gesetzt, die mit dem Pariser Abkommen ubereinstimmen.",
        "fr": "L'organisation a fixe des objectifs fondes sur la science, alignes sur l'Accord de Paris.",
        "es": "La organizacion ha establecido objetivos basados en la ciencia alineados con el Acuerdo de Paris.",
    }


# ---------------------------------------------------------------------------
# Fixtures -- Framework Mapping Data
# ---------------------------------------------------------------------------


@pytest.fixture
def framework_mapping_data() -> Dict[str, Any]:
    """Build framework metric mapping test data."""
    return {
        "mappings": [
            {
                "source_framework": "TCFD",
                "target_framework": "CDP",
                "source_metric": "Scope 1 GHG emissions",
                "target_metric": "C6.1 Scope 1 emissions",
                "mapping_type": "direct",
                "confidence": Decimal("1.00"),
            },
            {
                "source_framework": "TCFD",
                "target_framework": "CSRD",
                "source_metric": "Scope 2 GHG emissions (market-based)",
                "target_metric": "E1-6 Scope 2 market-based",
                "mapping_type": "direct",
                "confidence": Decimal("0.98"),
            },
            {
                "source_framework": "GRI",
                "target_framework": "ISSB",
                "source_metric": "GRI 305-4 intensity ratio",
                "target_metric": "IFRS S2 cross-industry intensity",
                "mapping_type": "calculated",
                "confidence": Decimal("0.85"),
            },
            {
                "source_framework": "CDP",
                "target_framework": "SEC",
                "source_metric": "C4.1a absolute target",
                "target_metric": "Reg S-K 1504 targets",
                "mapping_type": "approximate",
                "confidence": Decimal("0.75"),
            },
        ],
    }


# ---------------------------------------------------------------------------
# Fixtures -- XBRL Data
# ---------------------------------------------------------------------------


@pytest.fixture
def xbrl_tag_data() -> Dict[str, Any]:
    """Build XBRL tagging test data."""
    return {
        "metrics": [
            {
                "metric_name": "Scope 1 GHG Emissions",
                "value": Decimal("107500"),
                "unit": "tCO2e",
                "xbrl_element": "esef-cor:GrossScope1GHGEmissions",
                "namespace": "http://xbrl.efrag.org/esrs/2023/core",
                "context_ref": "FY2024",
                "decimals": 0,
            },
            {
                "metric_name": "Scope 2 GHG Emissions (Market)",
                "value": Decimal("67080"),
                "unit": "tCO2e",
                "xbrl_element": "esef-cor:GrossScope2MarketBasedGHGEmissions",
                "namespace": "http://xbrl.efrag.org/esrs/2023/core",
                "context_ref": "FY2024",
                "decimals": 0,
            },
        ],
        "taxonomy_version": "ESRS_2023",
        "framework": "CSRD",
    }


@pytest.fixture
def sec_xbrl_data() -> Dict[str, Any]:
    """Build SEC XBRL tagging test data."""
    return {
        "metrics": [
            {
                "metric_name": "Scope 1 Emissions",
                "value": Decimal("107500"),
                "unit": "tCO2e",
                "xbrl_element": "us-gaap:GreenHouseGasEmissionsScope1",
                "namespace": "http://fasb.org/us-gaap/2024",
                "context_ref": "FY2024",
                "decimals": 0,
            },
        ],
        "taxonomy_version": "US-GAAP_2024",
        "framework": "SEC",
    }


# ---------------------------------------------------------------------------
# Fixtures -- Dashboard Data
# ---------------------------------------------------------------------------


@pytest.fixture
def dashboard_config() -> Dict[str, Any]:
    """Build dashboard configuration test data."""
    return {
        "view_type": "executive",
        "frameworks_displayed": FRAMEWORKS,
        "charts": [
            {"type": "heatmap", "title": "Framework Coverage", "data_source": "coverage"},
            {"type": "bar", "title": "Emissions by Scope", "data_source": "emissions"},
            {"type": "line", "title": "Progress vs Target", "data_source": "progress"},
            {"type": "countdown", "title": "Upcoming Deadlines", "data_source": "deadlines"},
        ],
        "interactive": True,
        "responsive": True,
    }


# ---------------------------------------------------------------------------
# Fixtures -- Assurance Data
# ---------------------------------------------------------------------------


@pytest.fixture
def assurance_data() -> Dict[str, Any]:
    """Build assurance evidence package test data."""
    return {
        "report_id": "550e8400-e29b-41d4-a716-446655440099",
        "audit_standard": "ISAE 3410",
        "assurance_level": "limited",
        "evidence_items": [
            {"type": "provenance", "description": "SHA-256 calculation hashes", "count": 150},
            {"type": "lineage", "description": "Data lineage diagrams", "count": 12},
            {"type": "methodology", "description": "Calculation methodology docs", "count": 7},
            {"type": "control", "description": "Control matrix items", "count": 45},
        ],
        "total_calculations_traced": 1500,
        "provenance_hash": compute_sha256("assurance_data_2024"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Branding Config
# ---------------------------------------------------------------------------


@pytest.fixture
def branding_config() -> Dict[str, Any]:
    """Build branding configuration test data."""
    return {
        "logo_path": "/assets/logos/greencorp.png",
        "primary_color": "#1E3A8A",
        "secondary_color": "#3B82F6",
        "font_family": "Arial, sans-serif",
        "style": "corporate",
        "header_text": "GreenCorp Industries - Climate Disclosure",
        "footer_text": "Confidential - GreenCorp Industries 2024",
    }


# ---------------------------------------------------------------------------
# Fixtures -- Validation Data
# ---------------------------------------------------------------------------


@pytest.fixture
def validation_schema_data() -> Dict[str, Any]:
    """Build validation schema test data."""
    return {
        "schemas": {
            "SBTi": {"required_fields": ["base_year", "target_year", "reduction_pct", "scope_coverage"]},
            "CDP": {"required_modules": CDP_MODULES, "min_completeness_pct": Decimal("80")},
            "TCFD": {"required_pillars": TCFD_PILLARS},
            "GRI": {"required_disclosures": GRI_305_DISCLOSURES},
            "ISSB": {"required_sections": ["governance", "strategy", "risk_management", "metrics_targets"]},
            "SEC": {"required_items": ["item_1", "item_1a", "item_7", "reg_sk_1502"]},
            "CSRD": {"required_disclosures": ESRS_E1_DISCLOSURES},
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Translation Data
# ---------------------------------------------------------------------------


@pytest.fixture
def translation_glossary() -> Dict[str, Dict[str, str]]:
    """Build climate terminology translation glossary."""
    return {
        "greenhouse gas emissions": {
            "de": "Treibhausgasemissionen",
            "fr": "emissions de gaz a effet de serre",
            "es": "emisiones de gases de efecto invernadero",
        },
        "carbon footprint": {
            "de": "CO2-Fussabdruck",
            "fr": "empreinte carbone",
            "es": "huella de carbono",
        },
        "net zero": {
            "de": "Netto-Null",
            "fr": "zero net",
            "es": "cero neto",
        },
        "science-based targets": {
            "de": "wissenschaftsbasierte Ziele",
            "fr": "objectifs fondes sur la science",
            "es": "objetivos basados en la ciencia",
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Deadline Data
# ---------------------------------------------------------------------------


@pytest.fixture
def framework_deadlines() -> List[Dict[str, Any]]:
    """Build framework deadline test data."""
    return [
        {"framework": "CDP", "year": 2025, "deadline": "2025-07-31", "days_remaining": 133},
        {"framework": "SBTi", "year": 2025, "deadline": "2025-12-31", "days_remaining": 286},
        {"framework": "CSRD", "year": 2024, "deadline": "2025-05-31", "days_remaining": 72},
        {"framework": "SEC", "year": 2024, "deadline": "2025-03-31", "days_remaining": 11},
        {"framework": "TCFD", "year": 2024, "deadline": "2025-06-30", "days_remaining": 102},
        {"framework": "GRI", "year": 2024, "deadline": "2025-06-30", "days_remaining": 102},
        {"framework": "ISSB", "year": 2024, "deadline": "2025-06-30", "days_remaining": 102},
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Mock Database Session & Cache
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Create a mock async database session (PostgreSQL + TimescaleDB)."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(
        fetchall=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=None),
        scalar=MagicMock(return_value=0),
    ))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.begin = MagicMock()
    return session


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=True)
    redis_client.exists = AsyncMock(return_value=False)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.hget = AsyncMock(return_value=None)
    redis_client.hset = AsyncMock(return_value=True)
    redis_client.pipeline = MagicMock(return_value=MagicMock(
        execute=AsyncMock(return_value=[]),
    ))
    return redis_client


# ---------------------------------------------------------------------------
# Fixtures -- Mock External API Clients
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pack021_api():
    """Create a mock PACK-021 API client."""
    client = MagicMock()
    client.fetch_baseline = AsyncMock(return_value={
        "base_year": 2019,
        "scope_12_tco2e": Decimal("203000"),
        "scope_3_tco2e": Decimal("450000"),
    })
    client.fetch_inventory = AsyncMock(return_value={"status": "ok"})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_pack022_api():
    """Create a mock PACK-022 API client."""
    client = MagicMock()
    client.fetch_initiatives = AsyncMock(return_value=[
        {"name": "Solar PV", "reduction_tco2e": Decimal("18000")},
        {"name": "Fleet EV", "reduction_tco2e": Decimal("15000")},
    ])
    client.fetch_macc = AsyncMock(return_value={"curves": []})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_pack028_api():
    """Create a mock PACK-028 API client."""
    client = MagicMock()
    client.fetch_pathways = AsyncMock(return_value=[])
    client.fetch_convergence = AsyncMock(return_value={})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_pack029_api():
    """Create a mock PACK-029 API client."""
    client = MagicMock()
    client.fetch_targets = AsyncMock(return_value=[])
    client.fetch_progress = AsyncMock(return_value={})
    client.fetch_variance = AsyncMock(return_value={})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_sbti_app():
    """Create a mock GL-SBTi-APP client."""
    client = MagicMock()
    client.fetch_sbti_targets = AsyncMock(return_value={
        "near_term": {"year": 2030, "reduction_pct": Decimal("42")},
        "long_term": {"year": 2050, "reduction_pct": Decimal("90")},
    })
    client.fetch_validation = AsyncMock(return_value={"valid": True})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_cdp_app():
    """Create a mock GL-CDP-APP client."""
    client = MagicMock()
    client.fetch_cdp_history = AsyncMock(return_value=[])
    client.fetch_scores = AsyncMock(return_value={"score": "A-"})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_tcfd_app():
    """Create a mock GL-TCFD-APP client."""
    client = MagicMock()
    client.fetch_scenarios = AsyncMock(return_value=[])
    client.fetch_risks = AsyncMock(return_value=[])
    client.fetch_opportunities = AsyncMock(return_value=[])
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_ghg_app():
    """Create a mock GL-GHG-APP client."""
    client = MagicMock()
    client.fetch_inventory = AsyncMock(return_value={})
    client.fetch_emission_factors = AsyncMock(return_value={})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_translation_service():
    """Create a mock translation service client."""
    client = MagicMock()
    client.translate = AsyncMock(return_value={
        "translated_text": "Translated text placeholder",
        "quality_score": Decimal("0.95"),
    })
    client.detect_language = AsyncMock(return_value="en")
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_xbrl_taxonomy_service():
    """Create a mock XBRL taxonomy service client."""
    client = MagicMock()
    client.fetch_sec_taxonomy = AsyncMock(return_value={"version": "2024", "elements": []})
    client.fetch_csrd_taxonomy = AsyncMock(return_value={"version": "ESRS_2023", "elements": []})
    client.validate_tags = AsyncMock(return_value={"valid": True, "errors": []})
    client.is_connected = MagicMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Fixtures -- Pack Paths
# ---------------------------------------------------------------------------


@pytest.fixture
def pack_root() -> Path:
    """Return the pack root directory."""
    return _PACK_ROOT


@pytest.fixture
def presets_dir() -> Path:
    """Return the presets directory."""
    return PRESETS_DIR


# ---------------------------------------------------------------------------
# Parametrized Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=FRAMEWORKS, ids=FRAMEWORKS)
def framework(request) -> str:
    """Parameterized fixture yielding each reporting framework."""
    return request.param


@pytest.fixture(params=FRAMEWORK_IDS, ids=FRAMEWORK_IDS)
def framework_id(request) -> str:
    """Parameterized fixture yielding each framework ID."""
    return request.param


@pytest.fixture(params=OUTPUT_FORMATS, ids=OUTPUT_FORMATS)
def output_format(request) -> str:
    """Parameterized fixture yielding each output format."""
    return request.param


@pytest.fixture(params=LANGUAGES, ids=LANGUAGES)
def language(request) -> str:
    """Parameterized fixture yielding each supported language."""
    return request.param


@pytest.fixture(params=STAKEHOLDER_VIEWS, ids=STAKEHOLDER_VIEWS)
def stakeholder_view(request) -> str:
    """Parameterized fixture yielding each stakeholder view type."""
    return request.param


@pytest.fixture(params=SCOPES, ids=SCOPES)
def scope(request) -> str:
    """Parameterized fixture yielding each emission scope."""
    return request.param


@pytest.fixture(params=REPORT_STATUSES, ids=REPORT_STATUSES)
def report_status(request) -> str:
    """Parameterized fixture yielding each report status."""
    return request.param


@pytest.fixture(params=PRESET_NAMES, ids=PRESET_NAMES)
def preset_name(request) -> str:
    """Parameterized fixture yielding each preset name."""
    return request.param


@pytest.fixture(params=CDP_MODULES, ids=CDP_MODULES)
def cdp_module(request) -> str:
    """Parameterized fixture yielding each CDP module."""
    return request.param


@pytest.fixture(params=TCFD_PILLARS, ids=TCFD_PILLARS)
def tcfd_pillar(request) -> str:
    """Parameterized fixture yielding each TCFD pillar."""
    return request.param


@pytest.fixture(params=ESRS_E1_DISCLOSURES, ids=ESRS_E1_DISCLOSURES)
def esrs_e1_disclosure(request) -> str:
    """Parameterized fixture yielding each ESRS E1 disclosure."""
    return request.param


@pytest.fixture(params=VALIDATION_SEVERITIES, ids=VALIDATION_SEVERITIES)
def severity(request) -> str:
    """Parameterized fixture yielding each validation severity."""
    return request.param


@pytest.fixture(params=DATA_SOURCES, ids=DATA_SOURCES)
def data_source(request) -> str:
    """Parameterized fixture yielding each data source."""
    return request.param


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------


def generate_report_sections(framework: str, count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock report sections for a framework."""
    return [
        {
            "section_id": str(uuid.uuid4()),
            "section_type": f"section_{i+1}",
            "section_order": i + 1,
            "content": f"Sample content for {framework} section {i+1}.",
            "citations": [{"id": f"CIT-{i+1:03d}", "source": f"Source {i+1}"}],
            "language": "en",
            "consistency_score": Decimal(str(round(90 + i, 2))),
        }
        for i in range(count)
    ]


def generate_report_metrics(count: int = 10) -> List[Dict[str, Any]]:
    """Generate mock report metrics."""
    metric_names = [
        "scope_1_emissions", "scope_2_location", "scope_2_market",
        "scope_3_total", "total_emissions", "intensity_revenue",
        "reduction_pct", "renewable_energy_pct", "energy_consumption",
        "carbon_price_internal",
    ]
    return [
        {
            "metric_id": str(uuid.uuid4()),
            "metric_name": metric_names[i % len(metric_names)],
            "value": Decimal(str(10000 + 5000 * i)),
            "unit": "tCO2e" if i < 5 else "various",
            "source_system": DATA_SOURCES[i % len(DATA_SOURCES)],
            "provenance_hash": compute_sha256(f"metric_{i}"),
        }
        for i in range(count)
    ]


def generate_validation_issues(count: int = 5) -> List[Dict[str, Any]]:
    """Generate mock validation issues."""
    return [
        {
            "validation_id": str(uuid.uuid4()),
            "validator": ["schema", "completeness", "consistency"][i % 3],
            "validation_type": ["error", "warning", "info"][i % 3],
            "message": f"Validation issue {i+1}: sample message",
            "field_path": f"report.section_{i+1}.field",
            "severity": VALIDATION_SEVERITIES[i % len(VALIDATION_SEVERITIES)],
            "resolved": i > count // 2,
        }
        for i in range(count)
    ]


def generate_framework_coverage(frameworks: List[str] = None) -> Dict[str, Decimal]:
    """Generate framework coverage percentages."""
    if frameworks is None:
        frameworks = FRAMEWORKS
    return {
        fw: Decimal(str(round(75 + 3.5 * i, 1)))
        for i, fw in enumerate(frameworks)
    }
