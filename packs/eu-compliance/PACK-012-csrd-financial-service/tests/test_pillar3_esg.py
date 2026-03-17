# -*- coding: utf-8 -*-
"""
Unit tests for Pillar3ESGEngine (Engine 8)
============================================

Tests Template 1 (transition risk by sector), Template 2 (physical risk
by geography), Template 3 (real estate EPC), Template 4 (top 20 carbon-
intensive), Template 5 (taxonomy alignment), sector concentration,
maturity mismatch, and provenance hashing.

Target: 85%+ coverage, ~30 tests.
"""

import importlib.util
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "engines",
)
_ENGINE_PATH = os.path.normpath(
    os.path.join(_ENGINE_DIR, "pillar3_esg_engine.py")
)

spec = importlib.util.spec_from_file_location(
    "pillar3_esg_engine", _ENGINE_PATH,
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

Pillar3ESGEngine = mod.Pillar3ESGEngine
Pillar3Config = mod.Pillar3Config
BankingBookExposure = mod.BankingBookExposure
Pillar3Result = mod.Pillar3Result
TransitionRiskTemplate = mod.TransitionRiskTemplate
PhysicalRiskTemplate = mod.PhysicalRiskTemplate
RealEstateTemplate = mod.RealEstateTemplate
Top20CarbonExposure = mod.Top20CarbonExposure
TaxonomyAlignmentTemplate = mod.TaxonomyAlignmentTemplate
QualitativeDisclosure = mod.QualitativeDisclosure
Pillar3TemplateType = mod.Pillar3TemplateType
EPCLabel = mod.EPCLabel
NACESector = mod.NACESector
PDRange = mod.PDRange
MaturityRange = mod.MaturityRange
GeographicRegion = mod.GeographicRegion
PD_RANGE_BOUNDARIES = mod.PD_RANGE_BOUNDARIES
MATURITY_RANGE_BOUNDARIES = mod.MATURITY_RANGE_BOUNDARIES
NACE_SECTOR_LABELS = mod.NACE_SECTOR_LABELS
EPC_ORDER = mod.EPC_ORDER
CLIMATE_SENSITIVE_NACE = mod.CLIMATE_SENSITIVE_NACE
_compute_hash = mod._compute_hash
_safe_pct = mod._safe_pct
_clamp = mod._clamp
_round_val = mod._round_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> Pillar3Config:
    """Default engine configuration."""
    return Pillar3Config(
        institution_name="Test Credit Institution",
        templates_to_generate=[
            Pillar3TemplateType.TEMPLATE_1,
            Pillar3TemplateType.TEMPLATE_2,
            Pillar3TemplateType.TEMPLATE_3,
            Pillar3TemplateType.TEMPLATE_4,
            Pillar3TemplateType.TEMPLATE_5,
            Pillar3TemplateType.TEMPLATE_10,
        ],
        top_n_carbon=20,
        include_scope3=True,
    )


@pytest.fixture
def engine(default_config) -> Pillar3ESGEngine:
    """Engine instance."""
    return Pillar3ESGEngine(default_config)


@pytest.fixture
def sample_exposures() -> List[BankingBookExposure]:
    """Create a diverse set of banking book exposures."""
    return [
        # Energy sector - high carbon, with transition plan
        BankingBookExposure(
            exposure_id="p3-001",
            counterparty_name="EuroEnergy AG",
            counterparty_lei="LEI001",
            nace_code="D35",
            nace_section="D",
            country="DE",
            region=GeographicRegion.EU,
            gross_carrying_amount_eur=100_000_000.0,
            net_carrying_amount_eur=95_000_000.0,
            risk_weighted_amount_eur=80_000_000.0,
            probability_of_default=0.005,
            residual_maturity_years=7.0,
            scope1_emissions_tco2e=500_000.0,
            scope2_emissions_tco2e=50_000.0,
            scope3_emissions_tco2e=200_000.0,
            carbon_intensity=350.0,
            has_transition_plan=True,
            collateral_type="none",
            is_taxonomy_eligible=True,
            is_taxonomy_aligned=True,
            taxonomy_objective="climate_mitigation",
        ),
        # Real estate - with EPC
        BankingBookExposure(
            exposure_id="p3-002",
            counterparty_name="RE Investment Trust",
            counterparty_lei="LEI002",
            nace_code="L68",
            nace_section="L",
            country="NL",
            region=GeographicRegion.EU,
            gross_carrying_amount_eur=200_000_000.0,
            net_carrying_amount_eur=190_000_000.0,
            risk_weighted_amount_eur=100_000_000.0,
            probability_of_default=0.003,
            residual_maturity_years=15.0,
            scope1_emissions_tco2e=10_000.0,
            scope2_emissions_tco2e=30_000.0,
            carbon_intensity=45.0,
            collateral_type="real_estate",
            collateral_value_eur=250_000_000.0,
            epc_label=EPCLabel.B,
            energy_efficiency_kwh_m2=95.0,
            is_taxonomy_eligible=True,
            is_taxonomy_aligned=True,
            taxonomy_objective="climate_mitigation",
        ),
        # Manufacturing - high emissions, no plan
        BankingBookExposure(
            exposure_id="p3-003",
            counterparty_name="SteelWorks GmbH",
            counterparty_lei="LEI003",
            nace_code="C24",
            nace_section="C",
            country="DE",
            region=GeographicRegion.EU,
            gross_carrying_amount_eur=80_000_000.0,
            net_carrying_amount_eur=75_000_000.0,
            risk_weighted_amount_eur=70_000_000.0,
            probability_of_default=0.015,
            residual_maturity_years=3.0,
            scope1_emissions_tco2e=800_000.0,
            scope2_emissions_tco2e=100_000.0,
            scope3_emissions_tco2e=400_000.0,
            carbon_intensity=700.0,
            has_transition_plan=False,
            collateral_type="none",
            is_taxonomy_eligible=True,
            is_taxonomy_aligned=False,
        ),
        # Tech company - low emissions, emerging market
        BankingBookExposure(
            exposure_id="p3-004",
            counterparty_name="TechCorp India",
            counterparty_lei="LEI004",
            nace_code="J62",
            nace_section="J",
            country="IN",
            region=GeographicRegion.EMERGING,
            gross_carrying_amount_eur=30_000_000.0,
            net_carrying_amount_eur=28_000_000.0,
            risk_weighted_amount_eur=25_000_000.0,
            probability_of_default=0.025,
            residual_maturity_years=2.0,
            scope1_emissions_tco2e=500.0,
            scope2_emissions_tco2e=3_000.0,
            carbon_intensity=8.0,
            has_transition_plan=True,
            collateral_type="none",
        ),
        # Mining - fossil fuel, physical risk
        BankingBookExposure(
            exposure_id="p3-005",
            counterparty_name="CoalMining PLC",
            counterparty_lei="LEI005",
            nace_code="B05",
            nace_section="B",
            country="PL",
            region=GeographicRegion.EU,
            gross_carrying_amount_eur=50_000_000.0,
            net_carrying_amount_eur=45_000_000.0,
            risk_weighted_amount_eur=60_000_000.0,
            probability_of_default=0.08,
            residual_maturity_years=1.5,
            scope1_emissions_tco2e=1_200_000.0,
            scope2_emissions_tco2e=200_000.0,
            scope3_emissions_tco2e=3_000_000.0,
            carbon_intensity=1500.0,
            has_transition_plan=False,
            collateral_type="none",
            physical_risk_exposure=True,
            physical_hazard_type="flood",
        ),
        # Real estate - poor EPC
        BankingBookExposure(
            exposure_id="p3-006",
            counterparty_name="OldBuildings SARL",
            counterparty_lei="LEI006",
            nace_code="L68",
            nace_section="L",
            country="FR",
            region=GeographicRegion.EU,
            gross_carrying_amount_eur=60_000_000.0,
            net_carrying_amount_eur=55_000_000.0,
            risk_weighted_amount_eur=40_000_000.0,
            probability_of_default=0.01,
            residual_maturity_years=22.0,
            scope1_emissions_tco2e=5_000.0,
            scope2_emissions_tco2e=15_000.0,
            carbon_intensity=80.0,
            collateral_type="real_estate",
            collateral_value_eur=70_000_000.0,
            epc_label=EPCLabel.F,
            energy_efficiency_kwh_m2=250.0,
            physical_risk_exposure=True,
            physical_hazard_type="heatwave",
        ),
        # Agriculture - physical risk, high risk region
        BankingBookExposure(
            exposure_id="p3-007",
            counterparty_name="FarmCo Africa",
            counterparty_lei="LEI007",
            nace_code="A01",
            nace_section="A",
            country="KE",
            region=GeographicRegion.HIGH_RISK,
            gross_carrying_amount_eur=15_000_000.0,
            net_carrying_amount_eur=14_000_000.0,
            risk_weighted_amount_eur=18_000_000.0,
            probability_of_default=0.04,
            residual_maturity_years=4.0,
            scope1_emissions_tco2e=20_000.0,
            scope2_emissions_tco2e=5_000.0,
            carbon_intensity=120.0,
            has_transition_plan=False,
            physical_risk_exposure=True,
            physical_hazard_type="drought",
        ),
    ]


@pytest.fixture
def sample_qualitative_data() -> Dict[str, str]:
    """Create sample qualitative disclosure text."""
    return {
        "business_model_impact": "Climate risks materially impact our lending book.",
        "governance_framework": "Board-level ESG committee meets quarterly.",
        "risk_management_integration": "ESG factors integrated into credit risk framework.",
        "strategy_description": "Net-zero by 2050 strategy with interim 2030 targets.",
        "scenario_analysis_summary": "NGFS scenarios assessed across 3 time horizons.",
        "transition_plan_summary": "Transition plan published covering 9 NACE sectors.",
    }


# ===================================================================
# Test Class: Configuration
# ===================================================================


class TestPillar3Config:
    """Tests for Pillar3Config."""

    def test_default_config(self):
        cfg = Pillar3Config()
        assert cfg.top_n_carbon == 20
        assert cfg.include_scope3 is True
        assert len(cfg.templates_to_generate) == 6

    def test_custom_top_n(self):
        cfg = Pillar3Config(top_n_carbon=10)
        assert cfg.top_n_carbon == 10

    def test_selective_templates(self):
        cfg = Pillar3Config(
            templates_to_generate=[Pillar3TemplateType.TEMPLATE_1],
        )
        assert len(cfg.templates_to_generate) == 1


# ===================================================================
# Test Class: Engine Initialization
# ===================================================================


class TestEngineInit:
    """Tests for engine construction."""

    def test_engine_creates(self, default_config):
        eng = Pillar3ESGEngine(default_config)
        assert eng.config.institution_name == "Test Credit Institution"

    def test_climate_sensitive_nace_default(self, engine):
        assert "A" in engine._climate_sensitive
        assert "B" in engine._climate_sensitive
        assert "D" in engine._climate_sensitive
        assert "L" in engine._climate_sensitive


# ===================================================================
# Test Class: Template 1 - Transition Risk by Sector
# ===================================================================


class TestTemplate1TransitionRisk:
    """Tests for Template 1: Transition risk by sector/PD/maturity."""

    def test_template_1_generated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.transition_risk_template is not None

    def test_sector_data_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        assert len(t1.sector_data) > 0

    def test_pd_bucket_data_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        assert len(t1.pd_bucket_data) > 0

    def test_maturity_bucket_data_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        assert len(t1.maturity_bucket_data) > 0

    def test_total_gca_matches_input(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        expected_gca = sum(e.gross_carrying_amount_eur for e in sample_exposures)
        assert abs(t1.total_gca_eur - expected_gca) < 1.0

    def test_climate_sensitive_pct(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        assert 0.0 <= t1.climate_sensitive_pct <= 100.0

    def test_sector_concentration_sums_to_100(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t1 = result.transition_risk_template
        total_pct = sum(
            v.get("concentration_pct", 0.0) for v in t1.sector_data.values()
        )
        assert abs(total_pct - 100.0) < 1.0

    def test_provenance_hash_template_1(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.transition_risk_template.provenance_hash) == 64


# ===================================================================
# Test Class: Template 2 - Physical Risk by Geography
# ===================================================================


class TestTemplate2PhysicalRisk:
    """Tests for Template 2: Physical risk by geography/hazard."""

    def test_template_2_generated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.physical_risk_template is not None

    def test_geographic_data_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t2 = result.physical_risk_template
        assert len(t2.geographic_data) > 0

    def test_physical_risk_exposure_pct_in_range(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t2 = result.physical_risk_template
        assert 0.0 <= t2.physical_risk_exposure_pct <= 100.0

    def test_provenance_hash_template_2(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.physical_risk_template.provenance_hash) == 64


# ===================================================================
# Test Class: Template 3 - Real Estate EPC
# ===================================================================


class TestTemplate3RealEstate:
    """Tests for Template 3: Real estate collateral EPC distribution."""

    def test_template_3_generated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.real_estate_template is not None

    def test_epc_distribution_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t3 = result.real_estate_template
        assert len(t3.epc_distribution) > 0

    def test_real_estate_gca(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t3 = result.real_estate_template
        assert t3.total_real_estate_gca_eur > 0.0

    def test_epc_coverage_pct(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t3 = result.real_estate_template
        assert 0.0 <= t3.epc_coverage_pct <= 100.0

    def test_high_and_low_efficiency(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t3 = result.real_estate_template
        assert 0.0 <= t3.high_efficiency_pct <= 100.0
        assert 0.0 <= t3.low_efficiency_pct <= 100.0

    def test_provenance_hash_template_3(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.real_estate_template.provenance_hash) == 64


# ===================================================================
# Test Class: Template 4 - Top 20 Carbon-Intensive
# ===================================================================


class TestTemplate4Top20Carbon:
    """Tests for Template 4: Top 20 carbon-intensive counterparties."""

    def test_template_4_generated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.top_20_carbon is not None

    def test_top_20_list_populated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t4 = result.top_20_carbon
        assert len(t4.top_20_exposures) > 0
        assert len(t4.top_20_exposures) <= 20

    def test_top_20_sorted_by_emissions(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t4 = result.top_20_carbon
        if len(t4.top_20_exposures) >= 2:
            emissions_list = [
                e.get("total_emissions", 0.0) for e in t4.top_20_exposures
            ]
            assert emissions_list == sorted(emissions_list, reverse=True)

    def test_top_20_concentration_pct(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t4 = result.top_20_carbon
        assert 0.0 <= t4.top_20_concentration_pct <= 100.0

    def test_provenance_hash_template_4(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.top_20_carbon.provenance_hash) == 64


# ===================================================================
# Test Class: Template 5 - Taxonomy Alignment
# ===================================================================


class TestTemplate5TaxonomyAlignment:
    """Tests for Template 5: EU Taxonomy alignment (GAR/BTAR)."""

    def test_template_5_generated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.taxonomy_alignment is not None

    def test_gar_calculated(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t5 = result.taxonomy_alignment
        assert 0.0 <= t5.gar_pct <= 100.0

    def test_gar_numerator_leq_denominator(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t5 = result.taxonomy_alignment
        assert t5.gar_numerator_eur <= t5.gar_denominator_eur + 0.01

    def test_provenance_hash_template_5(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.taxonomy_alignment.provenance_hash) == 64


# ===================================================================
# Test Class: Template 10 - Qualitative Disclosures
# ===================================================================


class TestTemplate10Qualitative:
    """Tests for Template 10: Qualitative ESG risk disclosures."""

    def test_template_10_generated(self, engine, sample_exposures, sample_qualitative_data):
        result = engine.generate_pillar3_disclosures(
            sample_exposures, qualitative_data=sample_qualitative_data,
        )
        assert result.qualitative_disclosure is not None

    def test_sections_completed_count(self, engine, sample_exposures, sample_qualitative_data):
        result = engine.generate_pillar3_disclosures(
            sample_exposures, qualitative_data=sample_qualitative_data,
        )
        t10 = result.qualitative_disclosure
        assert t10.sections_completed == 6

    def test_completeness_100_when_all_filled(self, engine, sample_exposures, sample_qualitative_data):
        result = engine.generate_pillar3_disclosures(
            sample_exposures, qualitative_data=sample_qualitative_data,
        )
        t10 = result.qualitative_disclosure
        assert t10.completeness_pct == 100.0

    def test_empty_qualitative(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        t10 = result.qualitative_disclosure
        assert t10.sections_completed == 0
        assert t10.completeness_pct == 0.0

    def test_provenance_hash_template_10(self, engine, sample_exposures, sample_qualitative_data):
        result = engine.generate_pillar3_disclosures(
            sample_exposures, qualitative_data=sample_qualitative_data,
        )
        assert len(result.qualitative_disclosure.provenance_hash) == 64


# ===================================================================
# Test Class: Full Pillar 3 Result
# ===================================================================


class TestFullPillar3Result:
    """Tests for complete Pillar 3 disclosure result."""

    def test_result_structure(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert isinstance(result, Pillar3Result)
        assert result.total_exposures_count == len(sample_exposures)

    def test_templates_completed_count(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.templates_completed == 6

    def test_selective_templates(self, sample_exposures):
        cfg = Pillar3Config(
            templates_to_generate=[Pillar3TemplateType.TEMPLATE_1],
        )
        eng = Pillar3ESGEngine(cfg)
        result = eng.generate_pillar3_disclosures(sample_exposures)
        assert result.transition_risk_template is not None
        assert result.physical_risk_template is None
        assert result.templates_completed == 1

    def test_total_banking_book(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        expected = sum(e.gross_carrying_amount_eur for e in sample_exposures)
        assert abs(result.total_banking_book_eur - expected) < 1.0

    def test_emission_data_coverage(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert 0.0 <= result.emission_data_coverage_pct <= 100.0

    def test_epc_data_coverage(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert 0.0 <= result.epc_data_coverage_pct <= 100.0


# ===================================================================
# Test Class: Provenance and Metadata
# ===================================================================


class TestProvenanceAndMetadata:
    """Tests for provenance hash and result metadata."""

    def test_provenance_hash_is_sha256(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_processing_time_positive(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.processing_time_ms > 0.0

    def test_engine_version(self, engine, sample_exposures):
        result = engine.generate_pillar3_disclosures(sample_exposures)
        assert result.engine_version == "1.0.0"

    def test_empty_exposures(self, engine):
        result = engine.generate_pillar3_disclosures([])
        assert result.total_exposures_count == 0
        assert result.total_banking_book_eur == 0.0


# ===================================================================
# Test Class: Constants and Enums
# ===================================================================


class TestConstantsAndEnums:
    """Tests for module-level constants and enums."""

    def test_pd_range_boundaries(self):
        assert len(PD_RANGE_BOUNDARIES) == len(PDRange)

    def test_maturity_range_boundaries(self):
        assert len(MATURITY_RANGE_BOUNDARIES) == len(MaturityRange)

    def test_nace_sector_labels(self):
        assert len(NACE_SECTOR_LABELS) >= 18

    def test_epc_order(self):
        assert EPC_ORDER == ["A", "B", "C", "D", "E", "F", "G", "NONE"]

    def test_climate_sensitive_nace(self):
        assert CLIMATE_SENSITIVE_NACE == {"A", "B", "C", "D", "E", "F", "H", "L"}

    def test_epc_label_enum(self):
        assert len(EPCLabel) == 8  # A-G plus NONE
