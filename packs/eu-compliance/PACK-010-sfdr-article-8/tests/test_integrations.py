# -*- coding: utf-8 -*-
"""
PACK-010 SFDR Article 8 Pack - Integration Tests
===================================================

Tests all 10 integration bridges for SFDR Article 8:
SFDRPackOrchestrator, TaxonomyPackBridge, MRVEmissionsBridge,
InvestmentScreenerBridge, PortfolioDataBridge, EETDataBridge,
RegulatoryTrackingBridge, DataQualityBridge, SFDRHealthCheck,
SFDRSetupWizard.

Self-contained: does NOT import from conftest.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import all 10 integration modules
# ---------------------------------------------------------------------------

INT_DIR = str(PACK_ROOT / "integrations")

_int_orch = _import_from_path(
    "pack010_int_orch",
    os.path.join(INT_DIR, "pack_orchestrator.py"),
)
_int_tax = _import_from_path(
    "pack010_int_tax",
    os.path.join(INT_DIR, "taxonomy_pack_bridge.py"),
)
_int_mrv = _import_from_path(
    "pack010_int_mrv",
    os.path.join(INT_DIR, "mrv_emissions_bridge.py"),
)
_int_screener = _import_from_path(
    "pack010_int_screener",
    os.path.join(INT_DIR, "investment_screener_bridge.py"),
)
_int_portfolio = _import_from_path(
    "pack010_int_portfolio",
    os.path.join(INT_DIR, "portfolio_data_bridge.py"),
)
_int_eet = _import_from_path(
    "pack010_int_eet",
    os.path.join(INT_DIR, "eet_data_bridge.py"),
)
_int_reg = _import_from_path(
    "pack010_int_reg",
    os.path.join(INT_DIR, "regulatory_tracking_bridge.py"),
)
_int_dq = _import_from_path(
    "pack010_int_dq",
    os.path.join(INT_DIR, "data_quality_bridge.py"),
)
_int_health = _import_from_path(
    "pack010_int_health",
    os.path.join(INT_DIR, "health_check.py"),
)
_int_wizard = _import_from_path(
    "pack010_int_wizard",
    os.path.join(INT_DIR, "setup_wizard.py"),
)

# Classes and configs
SFDRPackOrchestrator = _int_orch.SFDRPackOrchestrator
SFDROrchestrationConfig = _int_orch.SFDROrchestrationConfig
SFDRPipelinePhase = _int_orch.SFDRPipelinePhase
PipelineResult = _int_orch.PipelineResult

TaxonomyPackBridge = _int_tax.TaxonomyPackBridge
TaxonomyBridgeConfig = _int_tax.TaxonomyBridgeConfig
FIELD_MAPPINGS = _int_tax.FIELD_MAPPINGS

MRVEmissionsBridge = _int_mrv.MRVEmissionsBridge
MRVEmissionsBridgeConfig = _int_mrv.MRVEmissionsBridgeConfig
PAI_TO_MRV_MAP = _int_mrv.PAI_TO_MRV_MAP

InvestmentScreenerBridge = _int_screener.InvestmentScreenerBridge
InvestmentScreenerBridgeConfig = _int_screener.InvestmentScreenerBridgeConfig
EXCLUSION_CATEGORIES = _int_screener.EXCLUSION_CATEGORIES
POSITIVE_CRITERIA = _int_screener.POSITIVE_CRITERIA

PortfolioDataBridge = _int_portfolio.PortfolioDataBridge
PortfolioDataBridgeConfig = _int_portfolio.PortfolioDataBridgeConfig
HOLDING_FIELDS = _int_portfolio.HOLDING_FIELDS
DATA_CATEGORIES = _int_portfolio.DATA_CATEGORIES

EETDataBridge = _int_eet.EETDataBridge
EETDataBridgeConfig = _int_eet.EETDataBridgeConfig
EET_SFDR_FIELDS = _int_eet.EET_SFDR_FIELDS

RegulatoryTrackingBridge = _int_reg.RegulatoryTrackingBridge
RegulatoryTrackingConfig = _int_reg.RegulatoryTrackingConfig
TRACKED_REGULATIONS = _int_reg.TRACKED_REGULATIONS
REGULATORY_EVENTS = _int_reg.REGULATORY_EVENTS

DataQualityBridge = _int_dq.DataQualityBridge
DataQualityBridgeConfig = _int_dq.DataQualityBridgeConfig
QUALITY_CHECKS = _int_dq.QUALITY_CHECKS

SFDRHealthCheck = _int_health.SFDRHealthCheck
HealthCheckConfig = _int_health.HealthCheckConfig
SFDRCheckCategory = _int_health.SFDRCheckCategory

SFDRSetupWizard = _int_wizard.SFDRSetupWizard
SetupWizardConfig = _int_wizard.SetupWizardConfig
WizardStepId = _int_wizard.WizardStepId
ProductType = _int_wizard.ProductType
PresetId = _int_wizard.PresetId


# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_HOLDINGS = [
    {
        "isin": "DE0001234567",
        "name": "Green Energy Corp",
        "weight": 25.0,
        "market_value": 2500000.0,
        "sector": "Energy",
        "sector_code": "D35",
        "country": "Germany",
        "country_code": "DE",
        "esg_rating": "AA",
        "taxonomy_eligible": True,
        "taxonomy_aligned": True,
        "emissions": {"scope_1_tco2e": 100.0, "scope_2_tco2e": 50.0, "scope_3_tco2e": 200.0},
        "revenue_eur": 50000000.0,
        "enterprise_value_eur": 100000000.0,
    },
    {
        "isin": "FR0009876543",
        "name": "Social Impact Ltd",
        "weight": 25.0,
        "market_value": 2500000.0,
        "sector": "Healthcare",
        "sector_code": "Q86",
        "country": "France",
        "country_code": "FR",
        "esg_rating": "A",
        "taxonomy_eligible": True,
        "taxonomy_aligned": False,
        "emissions": {"scope_1_tco2e": 30.0, "scope_2_tco2e": 20.0, "scope_3_tco2e": 80.0},
        "revenue_eur": 30000000.0,
        "enterprise_value_eur": 60000000.0,
    },
    {
        "isin": "NL0005551234",
        "name": "Clean Tech BV",
        "weight": 25.0,
        "market_value": 2500000.0,
        "sector": "Technology",
        "sector_code": "J62",
        "country": "Netherlands",
        "country_code": "NL",
        "esg_rating": "BBB",
        "taxonomy_eligible": False,
        "taxonomy_aligned": False,
        "emissions": {"scope_1_tco2e": 10.0, "scope_2_tco2e": 15.0, "scope_3_tco2e": 40.0},
        "revenue_eur": 20000000.0,
        "enterprise_value_eur": 40000000.0,
    },
    {
        "isin": "IE0004443210",
        "name": "Balanced Corp",
        "weight": 25.0,
        "market_value": 2500000.0,
        "sector": "Financials",
        "sector_code": "K64",
        "country": "Ireland",
        "country_code": "IE",
        "esg_rating": "A",
        "taxonomy_eligible": False,
        "taxonomy_aligned": False,
        "emissions": {"scope_1_tco2e": 5.0, "scope_2_tco2e": 10.0, "scope_3_tco2e": 30.0},
        "revenue_eur": 80000000.0,
        "enterprise_value_eur": 150000000.0,
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSFDRPackOrchestrator:
    """Tests for the 10-phase SFDR pipeline orchestrator."""

    def test_instantiation_default_config(self):
        """Orchestrator can be instantiated with default config."""
        orch = SFDRPackOrchestrator()
        assert orch is not None
        assert orch.config.pack_id == "PACK-010"

    def test_instantiation_custom_config(self):
        """Orchestrator accepts a custom configuration."""
        config = SFDROrchestrationConfig(
            product_name="GL Test Fund",
            product_isin="IE00B1234567",
        )
        orch = SFDRPackOrchestrator(config)
        assert orch.config.product_name == "GL Test Fund"

    def test_execute_pipeline(self):
        """execute_pipeline returns a PipelineResult with 10 phases."""
        config = SFDROrchestrationConfig(
            product_name="GL Pipeline Test Fund",
            max_retries=0,
        )
        orch = SFDRPackOrchestrator(config)
        result = orch.execute_pipeline(SAMPLE_HOLDINGS)
        assert isinstance(result, PipelineResult)
        assert result.pack_id == "PACK-010"
        assert len(result.phase_results) == 10
        assert len(result.provenance_hash) == 64

    def test_phase_skip(self):
        """Skipping a phase marks it as SKIPPED."""
        config = SFDROrchestrationConfig(
            product_name="GL Skip Test",
            skip_phases=["audit_trail"],
            max_retries=0,
        )
        orch = SFDRPackOrchestrator(config)
        result = orch.execute_pipeline(SAMPLE_HOLDINGS)
        audit_result = result.phase_results.get("audit_trail")
        assert audit_result is not None
        assert audit_result.status.value == "skipped"


class TestTaxonomyPackBridge:
    """Tests for PACK-008 taxonomy alignment bridge."""

    def test_instantiation(self):
        """Bridge can be instantiated with default config."""
        bridge = TaxonomyPackBridge()
        assert bridge is not None

    def test_get_alignment_ratio(self):
        """get_alignment_ratio returns an AlignmentResult."""
        bridge = TaxonomyPackBridge(TaxonomyBridgeConfig(use_pack_008=False))
        result = bridge.get_alignment_ratio(SAMPLE_HOLDINGS)
        assert hasattr(result, "aligned_pct")
        assert hasattr(result, "eligible_pct")
        assert result.holdings_assessed == len(SAMPLE_HOLDINGS)

    def test_field_mappings_count(self):
        """FIELD_MAPPINGS has at least 30 entries."""
        assert len(FIELD_MAPPINGS) >= 30


class TestMRVEmissionsBridge:
    """Tests for MRV agent emissions-to-PAI routing."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = MRVEmissionsBridge()
        assert bridge is not None

    def test_pai_to_mrv_map_has_6_indicators(self):
        """PAI_TO_MRV_MAP covers PAI 1 through PAI 6."""
        assert len(PAI_TO_MRV_MAP) == 6
        for i in range(1, 7):
            assert f"pai_{i}" in PAI_TO_MRV_MAP

    def test_aggregate_portfolio_emissions(self):
        """aggregate_portfolio_emissions returns PortfolioEmissions."""
        bridge = MRVEmissionsBridge()
        result = bridge.aggregate_portfolio_emissions(SAMPLE_HOLDINGS, 10_000_000.0)
        assert result.holdings_total == 4
        assert result.total_emissions >= 0


class TestInvestmentScreenerBridge:
    """Tests for SFDR classification and screening."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = InvestmentScreenerBridge()
        assert bridge is not None

    def test_classify_product(self):
        """classify_product returns a ClassificationResult."""
        bridge = InvestmentScreenerBridge()
        result = bridge.classify_product(SAMPLE_HOLDINGS)
        assert result.holdings_screened == 4
        assert hasattr(result, "classification")
        assert len(result.provenance_hash) == 64

    def test_exclusion_categories_loaded(self):
        """EXCLUSION_CATEGORIES has at least 9 entries."""
        assert len(EXCLUSION_CATEGORIES) >= 9

    def test_positive_criteria_loaded(self):
        """POSITIVE_CRITERIA has at least 5 entries."""
        assert len(POSITIVE_CRITERIA) >= 5


class TestPortfolioDataBridge:
    """Tests for portfolio holdings ingestion."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = PortfolioDataBridge()
        assert bridge is not None

    def test_import_holdings(self):
        """import_holdings validates and returns ImportResult."""
        bridge = PortfolioDataBridge()
        result = bridge.import_holdings(SAMPLE_HOLDINGS)
        assert result.total_records == 4
        assert result.valid_records == 4
        assert result.total_weight_pct == pytest.approx(100.0, abs=1.0)
        assert len(result.provenance_hash) == 64

    def test_holding_fields_defined(self):
        """HOLDING_FIELDS has expected field names."""
        assert "isin" in HOLDING_FIELDS
        assert "weight" in HOLDING_FIELDS
        assert "sector" in HOLDING_FIELDS

    def test_data_categories_defined(self):
        """DATA_CATEGORIES has 5 entries."""
        assert len(DATA_CATEGORIES) == 5


class TestEETDataBridge:
    """Tests for European ESG Template import/export."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = EETDataBridge()
        assert bridge is not None

    def test_import_eet(self):
        """import_eet parses EET field data."""
        bridge = EETDataBridge()
        raw = {"EET_10010": "article_8", "EET_20010": "Y"}
        result = bridge.import_eet(raw)
        assert result.populated_fields >= 2
        assert hasattr(result, "provenance_hash")

    def test_export_eet(self):
        """export_eet produces serialized content."""
        bridge = EETDataBridge()
        sfdr_data = {
            "product_name": "GL Test Fund",
            "product_isin": "IE00B1234567",
            "sfdr_classification": "article_8",
        }
        result = bridge.export_eet(sfdr_data)
        assert result.total_fields > 0
        assert len(result.content) > 0

    def test_eet_sfdr_fields_has_entries(self):
        """EET_SFDR_FIELDS has at least 50 field definitions."""
        assert len(EET_SFDR_FIELDS) >= 50


class TestRegulatoryTrackingBridge:
    """Tests for SFDR regulatory updates monitoring."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = RegulatoryTrackingBridge()
        assert bridge is not None

    def test_check_updates(self):
        """check_updates returns events."""
        bridge = RegulatoryTrackingBridge()
        result = bridge.check_updates()
        assert result.total_events > 0
        assert len(result.provenance_hash) == 64

    def test_tracked_regulations_loaded(self):
        """TRACKED_REGULATIONS has at least 7 entries."""
        assert len(TRACKED_REGULATIONS) >= 7

    def test_regulatory_events_loaded(self):
        """REGULATORY_EVENTS has at least 8 events."""
        assert len(REGULATORY_EVENTS) >= 8

    def test_get_timeline(self):
        """get_timeline returns a sorted list of milestones."""
        bridge = RegulatoryTrackingBridge()
        timeline = bridge.get_timeline()
        assert len(timeline) > 0


class TestDataQualityBridge:
    """Tests for PAI data quality enforcement."""

    def test_instantiation(self):
        """Bridge can be instantiated."""
        bridge = DataQualityBridge()
        assert bridge is not None

    def test_assess_quality(self):
        """assess_quality returns a QualityAssessment."""
        bridge = DataQualityBridge()
        pai_data = {"pai_indicators": {}}
        assessment = bridge.assess_quality(pai_data, SAMPLE_HOLDINGS)
        assert hasattr(assessment, "overall_score")
        assert 0.0 <= assessment.overall_score <= 100.0
        assert len(assessment.checks) > 0

    def test_quality_checks_defined(self):
        """QUALITY_CHECKS has 15 check definitions."""
        assert len(QUALITY_CHECKS) == 15


class TestSFDRHealthCheck:
    """Tests for 20-category system verification."""

    def test_instantiation(self):
        """Health check can be instantiated."""
        health = SFDRHealthCheck(HealthCheckConfig(
            project_root=str(PROJECT_ROOT),
        ))
        assert health is not None

    def test_20_categories_defined(self):
        """SFDRCheckCategory enum has 20 values."""
        assert len(SFDRCheckCategory) == 20

    def test_run_all_checks(self):
        """run_all_checks returns a report dict."""
        health = SFDRHealthCheck(HealthCheckConfig(
            project_root=str(PROJECT_ROOT),
        ))
        report = health.run_all_checks()
        assert "health_score" in report
        assert "categories_checked" in report
        assert report["categories_checked"] == 20
        assert "provenance_hash" in report


class TestSFDRSetupWizard:
    """Tests for 8-step guided configuration wizard."""

    def test_instantiation(self):
        """Wizard can be instantiated."""
        wizard = SFDRSetupWizard()
        assert wizard is not None

    def test_wizard_step_ids(self):
        """WizardStepId enum has 8 steps."""
        assert len(WizardStepId) == 8

    def test_product_types(self):
        """ProductType enum has 6 values."""
        assert len(ProductType) == 6
        assert ProductType.UCITS.value == "ucits"
        assert ProductType.PORTFOLIO_MANAGEMENT.value == "portfolio_management"

    def test_preset_ids(self):
        """PresetId enum has at least 5 presets."""
        assert len(PresetId) >= 5
