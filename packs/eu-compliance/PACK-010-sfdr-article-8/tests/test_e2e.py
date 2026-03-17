# -*- coding: utf-8 -*-
"""
End-to-end flow tests for PACK-010 SFDR Article 8 Pack.

These tests exercise multiple engines together in realistic scenarios,
validating that data flows correctly through the complete SFDR Article 8
compliance pipeline. Each test constructs realistic input data, runs it
through multiple engines sequentially, and verifies cross-engine consistency.

Test count: 12 tests
Target: Validate complete Article 8 compliance workflows
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Path setup - import engines from PACK-010 source tree
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Module imports - engines and templates
# ---------------------------------------------------------------------------

pai_mod = _import_from_path(
    "pai_indicator_calculator",
    str(PACK_ROOT / "engines" / "pai_indicator_calculator.py"),
)
tax_mod = _import_from_path(
    "taxonomy_alignment_ratio",
    str(PACK_ROOT / "engines" / "taxonomy_alignment_ratio.py"),
)
dnsh_mod = _import_from_path(
    "sfdr_dnsh_engine",
    str(PACK_ROOT / "engines" / "sfdr_dnsh_engine.py"),
)
gov_mod = _import_from_path(
    "good_governance_engine",
    str(PACK_ROOT / "engines" / "good_governance_engine.py"),
)
esg_mod = _import_from_path(
    "esg_characteristics_engine",
    str(PACK_ROOT / "engines" / "esg_characteristics_engine.py"),
)
si_mod = _import_from_path(
    "sustainable_investment_calculator",
    str(PACK_ROOT / "engines" / "sustainable_investment_calculator.py"),
)
carbon_mod = _import_from_path(
    "portfolio_carbon_footprint",
    str(PACK_ROOT / "engines" / "portfolio_carbon_footprint.py"),
)
eet_mod = _import_from_path(
    "eet_data_engine",
    str(PACK_ROOT / "engines" / "eet_data_engine.py"),
)
annex_ii_mod = _import_from_path(
    "annex_ii_precontractual",
    str(PACK_ROOT / "templates" / "annex_ii_precontractual.py"),
)
annex_iv_mod = _import_from_path(
    "annex_iv_periodic",
    str(PACK_ROOT / "templates" / "annex_iv_periodic.py"),
)

# ---------------------------------------------------------------------------
# Aliases for readability
# ---------------------------------------------------------------------------

PAIIndicatorCalculatorEngine = pai_mod.PAIIndicatorCalculatorEngine
PAIIndicatorConfig = pai_mod.PAIIndicatorConfig
InvesteeData = pai_mod.InvesteeData
InvesteeGHGData = pai_mod.InvesteeGHGData
InvesteeSocialData = pai_mod.InvesteeSocialData
InvesteeEnvironmentalData = pai_mod.InvesteeEnvironmentalData
InvesteeEnergyData = pai_mod.InvesteeEnergyData

TaxonomyAlignmentRatioEngine = tax_mod.TaxonomyAlignmentRatioEngine
TaxonomyAlignmentConfig = tax_mod.TaxonomyAlignmentConfig
HoldingAlignmentData = tax_mod.HoldingAlignmentData
AlignmentCategory = tax_mod.AlignmentCategory
EnvironmentalObjective = tax_mod.EnvironmentalObjective

SFDRDNSHEngine = dnsh_mod.SFDRDNSHEngine
DNSHConfig = dnsh_mod.DNSHConfig
InvestmentPAIData = dnsh_mod.InvestmentPAIData
DNSHStatus = dnsh_mod.DNSHStatus

GoodGovernanceEngine = gov_mod.GoodGovernanceEngine
GovernanceConfig = gov_mod.GovernanceConfig
CompanyGovernanceData = gov_mod.CompanyGovernanceData
ManagementStructureData = gov_mod.ManagementStructureData
EmployeeRelationsData = gov_mod.EmployeeRelationsData
RemunerationData = gov_mod.RemunerationData
TaxComplianceData = gov_mod.TaxComplianceData
GovernanceStatus = gov_mod.GovernanceStatus

ESGCharacteristicsEngine = esg_mod.ESGCharacteristicsEngine
CharacteristicType = esg_mod.CharacteristicType
AttainmentStatus = esg_mod.AttainmentStatus

SustainableInvestmentCalculatorEngine = si_mod.SustainableInvestmentCalculatorEngine
SIInvestmentData = si_mod.InvestmentData
ObjectiveContribution = si_mod.ObjectiveContribution

PortfolioCarbonFootprintEngine = carbon_mod.PortfolioCarbonFootprintEngine
HoldingEmissions = carbon_mod.HoldingEmissions
ScopeCoverage = carbon_mod.ScopeCoverage

EETDataEngine = eet_mod.EETDataEngine
SFDRClassification_EET = eet_mod.SFDRClassification
ExportFormat = eet_mod.ExportFormat

AnnexIIPrecontractualTemplate = annex_ii_mod.AnnexIIPrecontractualTemplate
ProductInfo = annex_ii_mod.ProductInfo
ESCharacteristics = annex_ii_mod.ESCharacteristics
AssetAllocation = annex_ii_mod.AssetAllocation

AnnexIVPeriodicTemplate = annex_iv_mod.AnnexIVPeriodicTemplate
ReportingPeriod = annex_iv_mod.ReportingPeriod

# ---------------------------------------------------------------------------
# Common helpers for building test data
# ---------------------------------------------------------------------------

REPORTING_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
REPORTING_END = datetime(2025, 12, 31, tzinfo=timezone.utc)
TOTAL_NAV = 100_000_000.0


def _make_pai_config() -> PAIIndicatorConfig:
    """Create a standard PAI config for tests."""
    return PAIIndicatorConfig(
        reporting_period_start=REPORTING_START,
        reporting_period_end=REPORTING_END,
        total_nav_eur=TOTAL_NAV,
    )


def _make_investee(
    idx: int,
    value: float = 10_000_000.0,
    scope1: float = 5000.0,
    scope2: float = 2000.0,
) -> InvesteeData:
    """Create a minimal corporate investee holding."""
    return InvesteeData(
        investee_id=f"ISIN{idx:04d}",
        investee_name=f"Corp {idx}",
        investee_type="CORPORATE",
        value_eur=value,
        enterprise_value_eur=value * 20,
        ghg_data=InvesteeGHGData(
            scope_1_tco2eq=scope1,
            scope_2_tco2eq=scope2,
            scope_3_tco2eq=scope1 * 2,
            revenue_eur=value * 5,
        ),
        social_data=InvesteeSocialData(
            ungc_oecd_violations=False,
            has_compliance_mechanism=True,
            gender_pay_gap_pct=12.0,
            female_board_pct=35.0,
            controversial_weapons=False,
        ),
        environmental_data=InvesteeEnvironmentalData(
            biodiversity_sensitive_areas=False,
            water_emissions_tonnes=20.0,
            hazardous_waste_tonnes=50.0,
        ),
        energy_data=InvesteeEnergyData(
            non_renewable_energy_pct=45.0,
            energy_intensity_gwh_per_m_revenue=1.5,
            nace_sector="C",
        ),
    )


def _make_taxonomy_holding(
    idx: int,
    value: float = 10_000_000.0,
    aligned_rev: float = 50.0,
    primary_obj: EnvironmentalObjective = EnvironmentalObjective.CCM,
) -> HoldingAlignmentData:
    """Create a taxonomy-aligned holding."""
    return HoldingAlignmentData(
        holding_id=f"HOLD{idx:04d}",
        holding_name=f"Green Corp {idx}",
        holding_type="CORPORATE",
        value_eur=value,
        alignment_category=AlignmentCategory.ALIGNED,
        aligned_revenue_pct=aligned_rev,
        aligned_capex_pct=aligned_rev * 0.8,
        aligned_opex_pct=aligned_rev * 0.6,
        eligible_revenue_pct=aligned_rev + 10,
        eligible_capex_pct=aligned_rev * 0.9,
        eligible_opex_pct=aligned_rev * 0.7,
        primary_objective=primary_obj,
        contributing_objectives=[primary_obj],
        dnsh_passed=True,
        minimum_safeguards_passed=True,
    )


def _make_dnsh_investment(
    idx: int,
    carbon_fp: float = 200.0,
    ungc_violation: bool = False,
    weapons: bool = False,
) -> InvestmentPAIData:
    """Create investment PAI data for DNSH screening."""
    return InvestmentPAIData(
        investment_id=f"INV{idx:04d}",
        investment_name=f"Investment {idx}",
        pai_values={
            "PAI_2": carbon_fp,
            "PAI_3": carbon_fp * 1.5,
            "PAI_5": 45.0,
            "PAI_6": 2.0,
            "PAI_8": 30.0,
            "PAI_9": 100.0,
            "PAI_12": 10.0,
            "PAI_13": 35.0,
            "PAI_15": 300.0,
        },
        pai_boolean_flags={
            "PAI_4": False,
            "PAI_7": False,
            "PAI_10": ungc_violation,
            "PAI_11": True,
            "PAI_14": weapons,
            "PAI_16": False,
            "PAI_17": False,
            "PAI_18": False,
        },
    )


def _make_governance_data(
    idx: int, good: bool = True
) -> CompanyGovernanceData:
    """Create company governance data."""
    return CompanyGovernanceData(
        company_id=f"COMP{idx:04d}",
        company_name=f"Company {idx}",
        management_data=ManagementStructureData(
            has_independent_board=True,
            independent_board_pct=50.0 if good else 10.0,
            has_audit_committee=good,
            has_risk_committee=good,
            ceo_chair_separation=good,
            has_sustainability_committee=good,
            has_whistleblower_mechanism=good,
            board_meetings_per_year=8 if good else 2,
        ),
        employee_data=EmployeeRelationsData(
            ilo_core_conventions_compliance=good,
            has_health_safety_policy=good,
            lost_time_injury_rate=2.0 if good else 10.0,
            has_training_programs=good,
            living_wage_compliance=good,
            has_grievance_mechanism=good,
        ),
        remuneration_data=RemunerationData(
            has_remuneration_policy=good,
            remuneration_policy_disclosed=good,
            ceo_to_median_pay_ratio=50.0 if good else 500.0,
            has_clawback_provisions=good,
            performance_linked_pay_pct=50.0 if good else 10.0,
            esg_linked_remuneration=good,
        ),
        tax_data=TaxComplianceData(
            has_tax_strategy_disclosure=good,
            country_by_country_reporting=good,
            aggressive_tax_planning_flag=not good,
            tax_haven_exposure=not good,
            tax_transparency_score=80.0 if good else 20.0,
        ),
        ungc_signatory=good,
        ungc_violations=not good,
        oecd_violations=not good,
        has_anti_corruption_policy=good,
        has_anti_bribery_measures=good,
        corruption_controversies=0 if good else 3,
    )


def _make_carbon_holding(
    idx: int,
    value: float = 10_000_000.0,
    scope1: float = 5000.0,
    scope2: float = 2000.0,
) -> HoldingEmissions:
    """Create a holding emissions object for carbon footprint calculation."""
    return HoldingEmissions(
        holding_id=f"HOLD{idx:04d}",
        company_name=f"Corp {idx}",
        isin=f"US000{idx:04d}001",
        sector="C",
        country="DE",
        scope1=scope1,
        scope2=scope2,
        scope3=scope1 * 2,
        total_emissions=scope1 + scope2 + scope1 * 2,
        revenue=value * 5,
        evic=value * 20,
        enterprise_value=value * 18,
        market_cap=value * 15,
        total_assets=value * 25,
        total_debt=value * 5,
        total_equity=value * 10,
        holding_value=value,
        weight_pct=10.0,
    )


# ===========================================================================
# E2E Test Class
# ===========================================================================


class TestE2E:
    """End-to-end flow tests exercising multiple engines in sequence."""

    # -----------------------------------------------------------------------
    # 1. PAI calculation to report
    # -----------------------------------------------------------------------

    def test_e2e_pai_calculation_to_report(self):
        """Calculate all 18 PAI indicators, then verify result completeness."""
        config = _make_pai_config()
        engine = PAIIndicatorCalculatorEngine(config)

        holdings = [_make_investee(i, value=10_000_000.0) for i in range(10)]
        result = engine.calculate_all_pai(holdings, fund_name="E2E Test Fund")

        # All 18 indicators should be present
        assert len(result.indicators) == 18
        assert result.total_holdings == 10
        assert result.total_nav_eur == TOTAL_NAV
        assert result.fund_name == "E2E Test Fund"

        # Provenance hash present and SHA-256 length
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

        # Coverage should be non-zero since we provided data
        assert result.overall_coverage_pct > 0

        # Check that climate PAI indicators have values
        for pai_id in ["PAI_1", "PAI_2", "PAI_3"]:
            assert pai_id in result.indicators
            indicator = result.indicators[pai_id]
            assert indicator.coverage.coverage_by_count_pct > 0

    # -----------------------------------------------------------------------
    # 2. Taxonomy alignment flow
    # -----------------------------------------------------------------------

    def test_e2e_taxonomy_alignment_flow(self):
        """Calculate alignment ratio, check commitment, generate pie chart."""
        config = TaxonomyAlignmentConfig(
            total_nav_eur=TOTAL_NAV,
            reporting_date=REPORTING_END,
            pre_contractual_commitment_pct=20.0,
        )
        engine = TaxonomyAlignmentRatioEngine(config)

        # 6 aligned + 2 eligible-not-aligned + 2 not-eligible
        holdings = []
        for i in range(6):
            holdings.append(_make_taxonomy_holding(i, value=10_000_000.0))
        for i in range(6, 8):
            h = _make_taxonomy_holding(i, value=10_000_000.0)
            h.alignment_category = AlignmentCategory.ELIGIBLE_NOT_ALIGNED
            h.aligned_revenue_pct = 0.0
            holdings.append(h)
        for i in range(8, 10):
            h = _make_taxonomy_holding(i, value=10_000_000.0)
            h.alignment_category = AlignmentCategory.NOT_ELIGIBLE
            h.aligned_revenue_pct = 0.0
            h.eligible_revenue_pct = 0.0
            h.primary_objective = None
            h.contributing_objectives = []
            holdings.append(h)

        result = engine.calculate_alignment_ratio(holdings, fund_name="Green Fund")

        # Alignment should be positive
        assert result.aligned_revenue_pct > 0
        assert result.total_holdings == 10
        assert result.fund_name == "Green Fund"

        # Commitment adherence
        assert result.commitment_adherence is not None
        assert result.commitment_adherence.pre_contractual_commitment_pct == 20.0
        if result.aligned_revenue_pct >= 20.0:
            assert result.commitment_adherence.status == "COMPLIANT"

        # Pie chart data generated
        assert len(result.pie_chart_data) > 0

        # Provenance hash present
        assert len(result.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 3. Pre-contractual disclosure flow
    # -----------------------------------------------------------------------

    def test_e2e_precontractual_disclosure_flow(self):
        """Classify product, assess ESG characteristics, generate Annex II."""
        # Step 1: Define ESG characteristics
        esg_engine = ESGCharacteristicsEngine({"product_name": "GL ESG Fund"})
        chars = esg_engine.define_characteristics(
            ["climate_mitigation", "labor_rights"]
        )
        assert len(chars) == 2
        env_chars = [c for c in chars if c.characteristic_type == CharacteristicType.ENVIRONMENTAL]
        social_chars = [c for c in chars if c.characteristic_type == CharacteristicType.SOCIAL]
        assert len(env_chars) == 1
        assert len(social_chars) == 1

        # Step 2: Render Annex II template
        template = AnnexIIPrecontractualTemplate()
        template_data = {
            "product_info": {
                "product_name": "GL ESG Fund",
                "isin": "LU1234567890",
                "sfdr_classification": "article_8",
                "fund_type": "UCITS",
                "currency": "EUR",
                "management_company": "GreenLang Asset Management",
            },
            "es_characteristics": {
                "environmental": ["Climate Change Mitigation"],
                "social": ["Labor Rights Protection"],
                "binding_elements": ["Minimum 50% low carbon investments"],
                "sustainability_indicators": ["Carbon intensity reduction %"],
            },
            "asset_allocation": {
                "sustainable_pct": 30.0,
                "taxonomy_aligned_pct": 15.0,
            },
        }
        markdown = template.render_markdown(template_data)

        assert "Pre-contractual" in markdown or "pre-contractual" in markdown.lower()
        assert "GL ESG Fund" in markdown
        assert len(markdown) > 100

    # -----------------------------------------------------------------------
    # 4. Periodic reporting flow
    # -----------------------------------------------------------------------

    def test_e2e_periodic_reporting_flow(self):
        """Collect data, calculate PAI, generate Annex IV periodic report."""
        # Step 1: Calculate PAI
        config = _make_pai_config()
        pai_engine = PAIIndicatorCalculatorEngine(config)
        holdings = [_make_investee(i) for i in range(5)]
        pai_result = pai_engine.calculate_all_pai(holdings)

        assert len(pai_result.indicators) == 18

        # Step 2: Generate periodic report
        template = AnnexIVPeriodicTemplate()
        report_data = {
            "reporting_period": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "fund_name": "GL Periodic Fund",
                "isin": "LU0987654321",
                "sfdr_classification": "article_8",
            },
            "characteristic_attainments": [
                {
                    "characteristic_name": "Carbon Intensity Reduction",
                    "characteristic_type": "environmental",
                    "target": 30.0,
                    "actual": 35.0,
                    "attained_pct": 100.0,
                },
            ],
            "proportion_breakdown": {
                "sustainable_total": 25.0,
                "taxonomy_aligned": 12.0,
            },
        }
        markdown = template.render_markdown(report_data)

        assert "GL Periodic Fund" in markdown
        assert len(markdown) > 100

    # -----------------------------------------------------------------------
    # 5. DNSH and governance pipeline
    # -----------------------------------------------------------------------

    def test_e2e_dnsh_and_governance_pipeline(self):
        """Run DNSH assessment then governance check on same portfolio."""
        # Step 1: DNSH assessment
        dnsh_engine = SFDRDNSHEngine()
        investments = [_make_dnsh_investment(i) for i in range(5)]
        dnsh_result = dnsh_engine.assess_portfolio_dnsh(
            investments, portfolio_name="DNSH+Gov Test"
        )

        assert dnsh_result.total_investments == 5
        assert dnsh_result.compliance_score_pct >= 0
        assert len(dnsh_result.provenance_hash) == 64

        # Step 2: Governance check on same set
        gov_engine = GoodGovernanceEngine()
        companies = [_make_governance_data(i, good=True) for i in range(5)]
        gov_result = gov_engine.assess_portfolio_governance(
            companies, portfolio_name="DNSH+Gov Test"
        )

        assert gov_result.total_companies == 5
        assert gov_result.compliance_rate_pct > 0
        assert len(gov_result.provenance_hash) == 64

        # Cross-check: both should assess the same number
        assert dnsh_result.total_investments == gov_result.total_companies

    # -----------------------------------------------------------------------
    # 6. Sustainable investment classification
    # -----------------------------------------------------------------------

    def test_e2e_sustainable_investment_classification(self):
        """Classify investments, calculate proportions, check commitment."""
        engine = SustainableInvestmentCalculatorEngine({
            "minimum_sustainable_pct": 20.0,
            "minimum_taxonomy_pct": 5.0,
        })

        investments = []
        for i in range(10):
            inv = SIInvestmentData(
                investment_id=f"SI{i:04d}",
                company_name=f"Sustainable Corp {i}",
                nav_value=10_000_000.0,
                weight_pct=10.0,
                sector="C",
                country="DE",
                taxonomy_eligible=True if i < 7 else False,
                taxonomy_aligned_pct=60.0 if i < 4 else 0.0,
                environmental_contribution=(
                    ObjectiveContribution.CLIMATE_MITIGATION if i < 6 else None
                ),
                social_contribution=(
                    ObjectiveContribution.SOCIAL_INEQUALITY if i >= 6 and i < 8 else None
                ),
                pai_data={
                    "ghg_emissions": 5000.0,
                    "carbon_footprint": 200.0,
                    "gender_pay_gap": 8.0,
                },
                governance_data={
                    "sound_management_structures": True,
                    "employee_relations": True,
                    "remuneration_compliance": True,
                    "tax_compliance": True,
                },
            )
            investments.append(inv)

        results = engine.classify_investments(investments)
        assert len(results) == 10

        proportion = engine.calculate_proportion()
        assert proportion.total_nav > 0
        assert proportion.total_sustainable_pct >= 0
        assert proportion.taxonomy_aligned_pct >= 0

        # Provenance present
        assert len(proportion.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 7. Carbon footprint to PAI
    # -----------------------------------------------------------------------

    def test_e2e_carbon_footprint_to_pai(self):
        """Calculate WACI and carbon footprint, validate for PAI 1-3."""
        carbon_engine = PortfolioCarbonFootprintEngine()
        holdings = [_make_carbon_holding(i) for i in range(10)]

        # Calculate WACI (PAI 3)
        waci_result = carbon_engine.calculate_waci(holdings)
        assert waci_result.waci_value >= 0
        assert waci_result.total_holdings == 10
        assert len(waci_result.provenance_hash) == 64

        # Calculate carbon footprint (PAI 2)
        footprint = carbon_engine.calculate_carbon_footprint(holdings)
        assert footprint.carbon_footprint >= 0
        assert footprint.total_portfolio_value > 0
        assert len(footprint.provenance_hash) == 64

        # Both metrics should be consistent
        # WACI is intensity-based; footprint is absolute-based
        assert waci_result.waci_value > 0 or footprint.carbon_footprint > 0

    # -----------------------------------------------------------------------
    # 8. EET data export
    # -----------------------------------------------------------------------

    def test_e2e_eet_data_export(self):
        """Populate EET fields from engine results, export and validate."""
        engine = EETDataEngine()

        # Set product info
        engine.set_product_info(
            isin="LU1234567890",
            name="GL Green Equity Fund",
            reporting_date="2025-12-31",
            management_company="GreenLang AM",
            currency="EUR",
        )

        # Set SFDR classification
        engine.set_sfdr_classification(
            classification=SFDRClassification_EET.ARTICLE_8,
            promotes_environmental=True,
            promotes_social=True,
            sustainable_investment_pct=25.0,
        )

        # Populate PAI fields
        pai_fields = {
            "EET_05_001": True,
            "EET_05_020": 150.5,
            "EET_05_030": 280.0,
            "EET_04_001": 35.5,
        }
        results = engine.populate_eet_fields(pai_fields, source="engine_results")

        # All fields should populate successfully
        for field_id, success in results.items():
            assert success, f"Failed to populate {field_id}"

        # Validate
        validation = engine.validate_eet_data()
        assert validation.completeness_pct > 0
        assert validation.total_fields_checked > 0

        # Export to JSON
        export = engine.export_eet(ExportFormat.JSON)
        assert export.field_count > 0
        assert len(export.content) > 0
        assert len(export.provenance_hash) == 64

    # -----------------------------------------------------------------------
    # 9. Full Article 8 pipeline
    # -----------------------------------------------------------------------

    def test_e2e_full_article_8_pipeline(self):
        """Complete Article 8 compliance check using all 8 engines."""
        # Engine 1: PAI
        pai_engine = PAIIndicatorCalculatorEngine(_make_pai_config())
        pai_result = pai_engine.calculate_all_pai(
            [_make_investee(i) for i in range(5)]
        )
        assert len(pai_result.indicators) == 18

        # Engine 2: Taxonomy Alignment
        tax_config = TaxonomyAlignmentConfig(
            total_nav_eur=TOTAL_NAV,
            reporting_date=REPORTING_END,
        )
        tax_engine = TaxonomyAlignmentRatioEngine(tax_config)
        tax_result = tax_engine.calculate_alignment_ratio(
            [_make_taxonomy_holding(i) for i in range(5)]
        )
        assert tax_result.aligned_revenue_pct >= 0

        # Engine 3: DNSH
        dnsh_engine = SFDRDNSHEngine()
        dnsh_result = dnsh_engine.assess_portfolio_dnsh(
            [_make_dnsh_investment(i) for i in range(5)]
        )
        assert dnsh_result.total_investments == 5

        # Engine 4: Good Governance
        gov_engine = GoodGovernanceEngine()
        gov_result = gov_engine.assess_portfolio_governance(
            [_make_governance_data(i) for i in range(5)]
        )
        assert gov_result.total_companies == 5

        # Engine 5: ESG Characteristics
        esg_engine = ESGCharacteristicsEngine()
        chars = esg_engine.define_characteristics(["climate_mitigation"])
        assert len(chars) >= 1

        # Engine 6: Sustainable Investment
        si_engine = SustainableInvestmentCalculatorEngine()
        si_investments = [
            SIInvestmentData(
                investment_id=f"SI{i}",
                company_name=f"Corp {i}",
                nav_value=10_000_000.0,
                weight_pct=20.0,
                taxonomy_eligible=True,
                taxonomy_aligned_pct=40.0,
                environmental_contribution=ObjectiveContribution.CLIMATE_MITIGATION,
                governance_data={
                    "sound_management_structures": True,
                    "employee_relations": True,
                    "remuneration_compliance": True,
                    "tax_compliance": True,
                },
            )
            for i in range(5)
        ]
        classifications = si_engine.classify_investments(si_investments)
        assert len(classifications) == 5

        # Engine 7: Carbon Footprint
        carbon_engine = PortfolioCarbonFootprintEngine()
        waci = carbon_engine.calculate_waci(
            [_make_carbon_holding(i) for i in range(5)]
        )
        assert waci.waci_value >= 0

        # Engine 8: EET
        eet_engine = EETDataEngine()
        eet_engine.set_product_info(
            isin="LU0001234567",
            name="Full Pipeline Fund",
            reporting_date="2025-12-31",
        )
        validation = eet_engine.validate_eet_data()
        assert validation.total_fields_checked > 0

        # All 8 engines completed - verify cross-engine data flow is possible
        assert pai_result.total_nav_eur == tax_result.total_nav_eur

    # -----------------------------------------------------------------------
    # 10. Portfolio screening and compliance
    # -----------------------------------------------------------------------

    def test_e2e_portfolio_screening_and_compliance(self):
        """Screen portfolio holdings, check binding elements, assess compliance."""
        # DNSH screening
        dnsh_engine = SFDRDNSHEngine()
        clean_investments = [_make_dnsh_investment(i) for i in range(8)]
        dirty_investment = _make_dnsh_investment(8, ungc_violation=True, weapons=True)
        all_investments = clean_investments + [dirty_investment]

        portfolio = dnsh_engine.assess_portfolio_dnsh(
            all_investments, portfolio_name="Screening Test"
        )

        assert portfolio.total_investments == 9
        # The dirty investment should fail and be flagged for exclusion
        assert portfolio.exclusion_count >= 1
        assert portfolio.failing_investments >= 1
        assert portfolio.compliance_score_pct < 100.0

        # Category summary should have entries
        assert len(portfolio.category_summary) > 0

    # -----------------------------------------------------------------------
    # 11. Cross-engine data consistency
    # -----------------------------------------------------------------------

    def test_e2e_cross_engine_data_consistency(self):
        """Verify data flows consistently across engines."""
        # Use the same portfolio value across engines
        nav = 50_000_000.0

        # PAI engine
        pai_config = PAIIndicatorConfig(
            reporting_period_start=REPORTING_START,
            reporting_period_end=REPORTING_END,
            total_nav_eur=nav,
        )
        pai_engine = PAIIndicatorCalculatorEngine(pai_config)
        holdings = [_make_investee(i, value=10_000_000.0) for i in range(5)]
        pai_result = pai_engine.calculate_all_pai(holdings)

        # Taxonomy engine with same NAV
        tax_config = TaxonomyAlignmentConfig(
            total_nav_eur=nav,
            reporting_date=REPORTING_END,
        )
        tax_engine = TaxonomyAlignmentRatioEngine(tax_config)
        tax_holdings = [_make_taxonomy_holding(i, value=10_000_000.0) for i in range(5)]
        tax_result = tax_engine.calculate_alignment_ratio(tax_holdings)

        # NAV should be consistent
        assert pai_result.total_nav_eur == tax_result.total_nav_eur == nav
        assert pai_result.total_holdings == tax_result.total_holdings == 5

        # Both should have provenance hashes
        assert len(pai_result.provenance_hash) == 64
        assert len(tax_result.provenance_hash) == 64

        # Hashes should be different (different data)
        assert pai_result.provenance_hash != tax_result.provenance_hash

    # -----------------------------------------------------------------------
    # 12. All engines have provenance
    # -----------------------------------------------------------------------

    def test_e2e_all_engines_have_provenance(self):
        """Run all 8 engines, verify all results have SHA-256 provenance hash."""
        hashes = []

        # Engine 1: PAI
        pai_engine = PAIIndicatorCalculatorEngine(_make_pai_config())
        r1 = pai_engine.calculate_all_pai([_make_investee(0)])
        assert len(r1.provenance_hash) == 64
        hashes.append(r1.provenance_hash)

        # Engine 2: Taxonomy
        tax_config = TaxonomyAlignmentConfig(
            total_nav_eur=TOTAL_NAV, reporting_date=REPORTING_END
        )
        tax_engine = TaxonomyAlignmentRatioEngine(tax_config)
        r2 = tax_engine.calculate_alignment_ratio([_make_taxonomy_holding(0)])
        assert len(r2.provenance_hash) == 64
        hashes.append(r2.provenance_hash)

        # Engine 3: DNSH
        dnsh_engine = SFDRDNSHEngine()
        r3 = dnsh_engine.assess_dnsh(_make_dnsh_investment(0))
        assert len(r3.provenance_hash) == 64
        hashes.append(r3.provenance_hash)

        # Engine 4: Governance
        gov_engine = GoodGovernanceEngine()
        r4 = gov_engine.assess_governance(_make_governance_data(0))
        assert len(r4.provenance_hash) == 64
        hashes.append(r4.provenance_hash)

        # Engine 5: ESG Characteristics
        esg_engine = ESGCharacteristicsEngine()
        chars = esg_engine.define_characteristics(["climate_mitigation"])
        assert len(chars[0].provenance_hash) == 64
        hashes.append(chars[0].provenance_hash)

        # Engine 6: Sustainable Investment
        si_engine = SustainableInvestmentCalculatorEngine()
        si_investments = [SIInvestmentData(
            investment_id="SI0001",
            company_name="SustCorp",
            nav_value=10_000_000.0,
            weight_pct=100.0,
            taxonomy_eligible=True,
            taxonomy_aligned_pct=50.0,
            environmental_contribution=ObjectiveContribution.CLIMATE_MITIGATION,
            governance_data={
                "sound_management_structures": True,
                "employee_relations": True,
                "remuneration_compliance": True,
                "tax_compliance": True,
            },
        )]
        classifications = si_engine.classify_investments(si_investments)
        assert len(classifications[0].provenance_hash) == 64
        hashes.append(classifications[0].provenance_hash)

        # Engine 7: Carbon Footprint
        carbon_engine = PortfolioCarbonFootprintEngine()
        r7 = carbon_engine.calculate_waci([_make_carbon_holding(0)])
        assert len(r7.provenance_hash) == 64
        hashes.append(r7.provenance_hash)

        # Engine 8: EET
        eet_engine = EETDataEngine()
        eet_engine.set_product_info(
            isin="LU0001", name="Prov Test", reporting_date="2025-12-31"
        )
        r8 = eet_engine.export_eet(ExportFormat.JSON)
        assert len(r8.provenance_hash) == 64
        hashes.append(r8.provenance_hash)

        # All 8 hashes collected
        assert len(hashes) == 8

        # All hashes are valid hex strings of length 64
        for h in hashes:
            assert len(h) == 64
            int(h, 16)  # Will raise if not valid hex

        # All hashes should be unique (different engines, different data)
        assert len(set(hashes)) == 8
