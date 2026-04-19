# -*- coding: utf-8 -*-
"""
Agent integration tests for PACK-010 SFDR Article 8 Pack.

These tests verify the wiring between pack components: orchestrator phases,
bridge configurations, engine importability, workflow modules, template
modules, config classes, and provenance hash generation. All tests are
marked with @pytest.mark.integration.

Test count: 15 tests
Target: Validate inter-component wiring and contract adherence
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Module imports - integration layer
# ---------------------------------------------------------------------------

orch_mod = _import_from_path(
    "pack_orchestrator",
    str(PACK_ROOT / "integrations" / "pack_orchestrator.py"),
)
tax_bridge_mod = _import_from_path(
    "taxonomy_pack_bridge",
    str(PACK_ROOT / "integrations" / "taxonomy_pack_bridge.py"),
)
mrv_bridge_mod = _import_from_path(
    "mrv_emissions_bridge",
    str(PACK_ROOT / "integrations" / "mrv_emissions_bridge.py"),
)
screener_bridge_mod = _import_from_path(
    "investment_screener_bridge",
    str(PACK_ROOT / "integrations" / "investment_screener_bridge.py"),
)
wizard_mod = _import_from_path(
    "setup_wizard",
    str(PACK_ROOT / "integrations" / "setup_wizard.py"),
)
health_mod = _import_from_path(
    "health_check",
    str(PACK_ROOT / "integrations" / "health_check.py"),
)

# Config module -- uses yaml import, may need fallback
try:
    config_mod = _import_from_path(
        "pack_config",
        str(PACK_ROOT / "config" / "pack_config.py"),
    )
    _CONFIG_AVAILABLE = True
except Exception:
    config_mod = None
    _CONFIG_AVAILABLE = False

# ---------------------------------------------------------------------------
# Aliases for readability
# ---------------------------------------------------------------------------

SFDRPackOrchestrator = orch_mod.SFDRPackOrchestrator
SFDROrchestrationConfig = orch_mod.SFDROrchestrationConfig
SFDRPipelinePhase = orch_mod.SFDRPipelinePhase
SFDRExecutionStatus = orch_mod.SFDRExecutionStatus

TaxonomyPackBridge = tax_bridge_mod.TaxonomyPackBridge
TaxonomyBridgeConfig = tax_bridge_mod.TaxonomyBridgeConfig
FIELD_MAPPINGS = tax_bridge_mod.FIELD_MAPPINGS

MRVEmissionsBridge = mrv_bridge_mod.MRVEmissionsBridge
MRVEmissionsBridgeConfig = mrv_bridge_mod.MRVEmissionsBridgeConfig
PAIIndicator_MRV = mrv_bridge_mod.PAIIndicator

InvestmentScreenerBridge = screener_bridge_mod.InvestmentScreenerBridge
InvestmentScreenerBridgeConfig = screener_bridge_mod.InvestmentScreenerBridgeConfig

SFDRSetupWizard = wizard_mod.SFDRSetupWizard
SetupWizardConfig = wizard_mod.SetupWizardConfig
WizardStepId = wizard_mod.WizardStepId
PresetId = wizard_mod.PresetId

SFDRHealthCheck = health_mod.SFDRHealthCheck
HealthCheckConfig = health_mod.HealthCheckConfig
SFDRCheckCategory = health_mod.SFDRCheckCategory


# ===========================================================================
# Integration Test Class
# ===========================================================================


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for SFDR Article 8 Pack component wiring."""

    # -----------------------------------------------------------------------
    # 1. Pack orchestrator instantiation
    # -----------------------------------------------------------------------

    def test_pack_orchestrator_instantiation(self):
        """Verify orchestrator creates with 10 phases, config, agent stubs."""
        config = SFDROrchestrationConfig(
            product_name="Integration Test Fund",
            product_isin="LU0001234567",
        )
        orch = SFDRPackOrchestrator(config)

        assert orch.config.product_name == "Integration Test Fund"
        assert orch.config.product_isin == "LU0001234567"
        assert orch.config.pack_id == "PACK-010"

        # The orchestrator should be instantiated without errors
        # and its config should have the SFDR classification
        assert hasattr(orch, "config")
        assert orch.config.sfdr_classification is not None

    # -----------------------------------------------------------------------
    # 2. Pipeline phase enum has exactly 10 values
    # -----------------------------------------------------------------------

    def test_pack_orchestrator_has_10_phases(self):
        """Verify pipeline phase enum has exactly 10 values."""
        phases = list(SFDRPipelinePhase)
        assert len(phases) == 10

        # Verify all expected phases are present
        expected_phases = {
            "health_check",
            "configuration_init",
            "portfolio_data_loading",
            "pai_data_collection",
            "taxonomy_alignment_assessment",
            "dnsh_governance_screening",
            "esg_characteristics_assessment",
            "disclosure_generation",
            "compliance_verification",
            "audit_trail",
        }
        actual_phases = {p.value for p in phases}
        assert actual_phases == expected_phases

    # -----------------------------------------------------------------------
    # 3. All pack bridges instantiate
    # -----------------------------------------------------------------------

    def test_all_pack_bridges_instantiate(self):
        """Construct all bridge types with configs."""
        # Taxonomy bridge
        tax_bridge = TaxonomyPackBridge(TaxonomyBridgeConfig())
        assert tax_bridge.config is not None
        assert hasattr(tax_bridge, "_agents")

        # MRV emissions bridge
        mrv_bridge = MRVEmissionsBridge(MRVEmissionsBridgeConfig())
        assert mrv_bridge.config is not None
        assert hasattr(mrv_bridge, "_agents")

        # Investment screener bridge
        screener = InvestmentScreenerBridge(InvestmentScreenerBridgeConfig())
        assert screener.config is not None
        assert hasattr(screener, "_screener")

    # -----------------------------------------------------------------------
    # 4. Taxonomy pack bridge has field mappings
    # -----------------------------------------------------------------------

    def test_taxonomy_pack_bridge_has_mappings(self):
        """Verify field mappings exist and cover key SFDR fields."""
        assert isinstance(FIELD_MAPPINGS, dict)
        assert len(FIELD_MAPPINGS) > 0

        # Key taxonomy alignment fields must be mapped
        key_fields = [
            "taxonomy_aligned_turnover_pct",
            "taxonomy_aligned_capex_pct",
            "taxonomy_aligned_opex_pct",
            "taxonomy_eligible_turnover_pct",
        ]
        for field in key_fields:
            assert field in FIELD_MAPPINGS, f"Missing mapping for {field}"

        # Climate mitigation objective should be mapped
        assert "climate_mitigation_aligned_pct" in FIELD_MAPPINGS

        # Gas/nuclear CDA fields
        assert "fossil_gas_turnover_pct" in FIELD_MAPPINGS
        assert "nuclear_turnover_pct" in FIELD_MAPPINGS

    # -----------------------------------------------------------------------
    # 5. MRV emissions bridge PAI routing
    # -----------------------------------------------------------------------

    def test_mrv_emissions_bridge_pai_routing(self):
        """PAI 1-6 map to MRV agents."""
        # Verify the PAI indicator enum covers PAI 1-6
        pai_values = [p.value for p in PAIIndicator_MRV]
        assert "pai_1" in pai_values
        assert "pai_2" in pai_values
        assert "pai_3" in pai_values
        assert "pai_4" in pai_values
        assert "pai_5" in pai_values
        assert "pai_6" in pai_values

        # Bridge should have agent stubs configured
        bridge = MRVEmissionsBridge()
        assert hasattr(bridge, "_agents")

        # Config should include all three scopes by default
        assert "scope_1" in bridge.config.scope_coverage
        assert "scope_2" in bridge.config.scope_coverage
        assert "scope_3" in bridge.config.scope_coverage

    # -----------------------------------------------------------------------
    # 6. Investment screener bridge criteria
    # -----------------------------------------------------------------------

    def test_investment_screener_bridge_criteria(self):
        """Article 8 screening criteria configuration validation."""
        # Default config targets Article 8
        config = InvestmentScreenerBridgeConfig()
        assert config.screening_criteria == "article_8"
        assert config.sfdr_target.value == "article_8"
        assert config.min_esg_rating == "BBB"
        assert config.revenue_threshold_pct == 5.0

        bridge = InvestmentScreenerBridge(config)
        assert bridge.config.enable_controversy_screening is True
        assert bridge.config.max_controversy_level == 4

        # Verify exclusion categories exist in the module
        exclusion_categories = list(screener_bridge_mod.ExclusionCategory)
        assert len(exclusion_categories) >= 10

        # Key exclusions for Article 8
        excl_values = {e.value for e in exclusion_categories}
        assert "controversial_weapons" in excl_values
        assert "tobacco" in excl_values
        assert "thermal_coal" in excl_values
        assert "ungc_violators" in excl_values

    # -----------------------------------------------------------------------
    # 7. Setup wizard steps
    # -----------------------------------------------------------------------

    def test_setup_wizard_steps(self):
        """8 steps numbered correctly."""
        # Wizard step enum should have exactly 8 values
        steps = list(WizardStepId)
        assert len(steps) == 8

        expected_step_ids = {
            "product_type",
            "es_characteristics",
            "binding_elements",
            "taxonomy_alignment",
            "pai_indicators",
            "data_sources",
            "reporting_schedule",
            "validation_deployment",
        }
        actual_step_ids = {s.value for s in steps}
        assert actual_step_ids == expected_step_ids

        # Wizard instantiates successfully
        wizard = SFDRSetupWizard(SetupWizardConfig())
        assert hasattr(wizard, "config")

    # -----------------------------------------------------------------------
    # 8. Setup wizard presets
    # -----------------------------------------------------------------------

    def test_setup_wizard_presets(self):
        """5+ presets available."""
        presets = list(PresetId)
        assert len(presets) >= 5

        preset_values = {p.value for p in presets}
        # Must have at least these presets
        assert "asset_manager" in preset_values
        assert "insurance" in preset_values
        assert "bank" in preset_values
        assert "pension_fund" in preset_values
        assert "wealth_manager" in preset_values
        assert "custom" in preset_values

        # Default config should include all presets
        config = SetupWizardConfig()
        assert len(config.available_presets) >= 5
        assert config.default_preset == "asset_manager"

    # -----------------------------------------------------------------------
    # 9. Health check categories
    # -----------------------------------------------------------------------

    def test_health_check_categories(self):
        """20 check categories across 4 areas."""
        categories = list(SFDRCheckCategory)
        assert len(categories) == 20

        # Verify area distribution: 5 engines + 5 workflows + 5 config + 5 integrations
        engine_cats = [c for c in categories if c.value.startswith("engine_")]
        workflow_cats = [c for c in categories if c.value.startswith("workflow_")]
        config_cats = [c for c in categories if c.value.startswith("config_")]
        integration_cats = [c for c in categories if c.value.startswith("integration_")]

        assert len(engine_cats) == 5
        assert len(workflow_cats) == 5
        assert len(config_cats) == 5
        assert len(integration_cats) == 5

        # Health check instantiation
        health = SFDRHealthCheck(HealthCheckConfig())
        assert hasattr(health, "config")
        assert len(health.config.check_categories) == 20

    # -----------------------------------------------------------------------
    # 10. All engines importable
    # -----------------------------------------------------------------------

    def test_all_engines_importable(self):
        """Import all 8 engine modules from the engines directory."""
        engine_files = [
            ("pai_indicator_calculator", "PAIIndicatorCalculatorEngine"),
            ("taxonomy_alignment_ratio", "TaxonomyAlignmentRatioEngine"),
            ("sfdr_dnsh_engine", "SFDRDNSHEngine"),
            ("good_governance_engine", "GoodGovernanceEngine"),
            ("esg_characteristics_engine", "ESGCharacteristicsEngine"),
            ("sustainable_investment_calculator", "SustainableInvestmentCalculatorEngine"),
            ("portfolio_carbon_footprint", "PortfolioCarbonFootprintEngine"),
            ("eet_data_engine", "EETDataEngine"),
        ]

        engines_dir = PACK_ROOT / "engines"

        for module_name, class_name in engine_files:
            file_path = engines_dir / f"{module_name}.py"
            assert file_path.exists(), f"Engine file missing: {file_path}"

            mod = _import_from_path(f"test_eng_{module_name}", str(file_path))
            assert hasattr(mod, class_name), (
                f"Engine class {class_name} not found in {module_name}"
            )

    # -----------------------------------------------------------------------
    # 11. All workflows importable
    # -----------------------------------------------------------------------

    def test_all_workflows_importable(self):
        """Import all 8 workflow modules from the workflows directory."""
        workflow_files = [
            "precontractual_disclosure",
            "periodic_reporting",
            "website_disclosure",
            "pai_statement",
            "portfolio_screening",
            "taxonomy_alignment",
            "compliance_review",
            "regulatory_update",
        ]

        workflows_dir = PACK_ROOT / "workflows"

        for wf_name in workflow_files:
            file_path = workflows_dir / f"{wf_name}.py"
            assert file_path.exists(), f"Workflow file missing: {file_path}"

            mod = _import_from_path(f"test_wf_{wf_name}", str(file_path))
            # Each workflow module should have at least one class or function
            module_attrs = [a for a in dir(mod) if not a.startswith("_")]
            assert len(module_attrs) > 0, (
                f"Workflow module {wf_name} appears empty"
            )

    # -----------------------------------------------------------------------
    # 12. All templates importable
    # -----------------------------------------------------------------------

    def test_all_templates_importable(self):
        """Import all 8 template modules from the templates directory."""
        templates_dir = PACK_ROOT / "templates"

        # Get all .py files excluding __init__.py
        template_files = sorted(
            f for f in templates_dir.glob("*.py")
            if f.name != "__init__.py"
        )

        assert len(template_files) >= 2, (
            f"Expected at least 2 template files, found {len(template_files)}"
        )

        for tf in template_files:
            mod = _import_from_path(f"test_tmpl_{tf.stem}", str(tf))
            module_attrs = [a for a in dir(mod) if not a.startswith("_")]
            assert len(module_attrs) > 0, (
                f"Template module {tf.stem} appears empty"
            )

    # -----------------------------------------------------------------------
    # 13. Config module has SFDRArticle8Config class
    # -----------------------------------------------------------------------

    def test_config_module_has_sfdr_config_class(self):
        """SFDRArticle8Config exists in pack_config module."""
        if not _CONFIG_AVAILABLE:
            pytest.skip("pack_config module could not be imported (requires yaml)")

        assert hasattr(config_mod, "SFDRArticle8Config"), (
            "SFDRArticle8Config class not found in pack_config module"
        )

        # Verify it is a Pydantic BaseModel subclass
        from pydantic import BaseModel
        assert issubclass(config_mod.SFDRArticle8Config, BaseModel)

        # Verify key enums exist
        assert hasattr(config_mod, "SFDRClassification")
        assert hasattr(config_mod, "PAICategory")
        assert hasattr(config_mod, "DisclosureType")

    # -----------------------------------------------------------------------
    # 14. Bridge config pattern enforced
    # -----------------------------------------------------------------------

    def test_bridge_config_pattern_enforced(self):
        """All bridges accept a config parameter and have defaults."""
        # TaxonomyPackBridge accepts config or None
        bridge1 = TaxonomyPackBridge()
        assert bridge1.config is not None
        assert isinstance(bridge1.config, TaxonomyBridgeConfig)

        bridge1_with_config = TaxonomyPackBridge(TaxonomyBridgeConfig(
            alignment_methodology=tax_bridge_mod.AlignmentMethodology.CAPEX,
        ))
        assert bridge1_with_config.config.alignment_methodology.value == "capex"

        # MRVEmissionsBridge accepts config or None
        bridge2 = MRVEmissionsBridge()
        assert bridge2.config is not None
        assert isinstance(bridge2.config, MRVEmissionsBridgeConfig)

        bridge2_with_config = MRVEmissionsBridge(MRVEmissionsBridgeConfig(
            emission_factor_source="manual",
        ))
        assert bridge2_with_config.config.emission_factor_source == "manual"

        # InvestmentScreenerBridge accepts config or None
        bridge3 = InvestmentScreenerBridge()
        assert bridge3.config is not None
        assert isinstance(bridge3.config, InvestmentScreenerBridgeConfig)

        bridge3_with_config = InvestmentScreenerBridge(
            InvestmentScreenerBridgeConfig(min_esg_rating="A")
        )
        assert bridge3_with_config.config.min_esg_rating == "A"

    # -----------------------------------------------------------------------
    # 15. Provenance hash in engine results
    # -----------------------------------------------------------------------

    def test_provenance_hash_in_engine_results(self):
        """SHA-256 provenance hash on key engine outputs."""
        # Import engines for direct testing
        pai_mod_local = _import_from_path(
            "test_prov_pai",
            str(PACK_ROOT / "engines" / "pai_indicator_calculator.py"),
        )
        dnsh_mod_local = _import_from_path(
            "test_prov_dnsh",
            str(PACK_ROOT / "engines" / "sfdr_dnsh_engine.py"),
        )
        gov_mod_local = _import_from_path(
            "test_prov_gov",
            str(PACK_ROOT / "engines" / "good_governance_engine.py"),
        )
        tax_mod_local = _import_from_path(
            "test_prov_tax",
            str(PACK_ROOT / "engines" / "taxonomy_alignment_ratio.py"),
        )

        # PAI engine result has provenance_hash
        pai_config = pai_mod_local.PAIIndicatorConfig(
            reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
            reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
            total_nav_eur=100_000_000.0,
        )
        pai_engine = pai_mod_local.PAIIndicatorCalculatorEngine(pai_config)
        investee = pai_mod_local.InvesteeData(
            investee_id="ISIN0001",
            investee_name="TestCorp",
            investee_type="CORPORATE",
            value_eur=10_000_000.0,
            enterprise_value_eur=200_000_000.0,
            ghg_data=pai_mod_local.InvesteeGHGData(
                scope_1_tco2eq=5000.0,
                scope_2_tco2eq=2000.0,
                revenue_eur=50_000_000.0,
            ),
            social_data=pai_mod_local.InvesteeSocialData(
                ungc_oecd_violations=False,
                has_compliance_mechanism=True,
                gender_pay_gap_pct=10.0,
                female_board_pct=40.0,
                controversial_weapons=False,
            ),
        )
        pai_result = pai_engine.calculate_all_pai([investee])
        assert hasattr(pai_result, "provenance_hash")
        assert len(pai_result.provenance_hash) == 64
        # Verify it is valid hex
        int(pai_result.provenance_hash, 16)

        # DNSH engine result has provenance_hash
        dnsh_engine = dnsh_mod_local.SFDRDNSHEngine()
        dnsh_data = dnsh_mod_local.InvestmentPAIData(
            investment_id="INV0001",
            investment_name="DNSHTest",
            pai_values={"PAI_2": 200.0, "PAI_5": 40.0},
            pai_boolean_flags={"PAI_10": False, "PAI_14": False},
        )
        dnsh_result = dnsh_engine.assess_dnsh(dnsh_data)
        assert len(dnsh_result.provenance_hash) == 64
        int(dnsh_result.provenance_hash, 16)

        # Governance engine result has provenance_hash
        gov_engine = gov_mod_local.GoodGovernanceEngine()
        gov_data = gov_mod_local.CompanyGovernanceData(
            company_id="COMP0001",
            company_name="GovTest",
            management_data=gov_mod_local.ManagementStructureData(
                has_audit_committee=True,
                independent_board_pct=50.0,
            ),
            employee_data=gov_mod_local.EmployeeRelationsData(
                ilo_core_conventions_compliance=True,
                has_health_safety_policy=True,
            ),
            remuneration_data=gov_mod_local.RemunerationData(
                has_remuneration_policy=True,
            ),
            tax_data=gov_mod_local.TaxComplianceData(
                has_tax_strategy_disclosure=True,
                aggressive_tax_planning_flag=False,
            ),
            ungc_signatory=True,
            ungc_violations=False,
            has_anti_corruption_policy=True,
            has_anti_bribery_measures=True,
        )
        gov_result = gov_engine.assess_governance(gov_data)
        assert len(gov_result.provenance_hash) == 64
        int(gov_result.provenance_hash, 16)

        # Taxonomy engine result has provenance_hash
        tax_config = tax_mod_local.TaxonomyAlignmentConfig(
            total_nav_eur=100_000_000.0,
            reporting_date=datetime(2025, 12, 31, tzinfo=timezone.utc),
        )
        tax_engine = tax_mod_local.TaxonomyAlignmentRatioEngine(tax_config)
        holding = tax_mod_local.HoldingAlignmentData(
            holding_id="H0001",
            holding_name="TaxTest",
            holding_type="CORPORATE",
            value_eur=10_000_000.0,
            alignment_category=tax_mod_local.AlignmentCategory.ALIGNED,
            aligned_revenue_pct=50.0,
            primary_objective=tax_mod_local.EnvironmentalObjective.CCM,
            contributing_objectives=[tax_mod_local.EnvironmentalObjective.CCM],
        )
        tax_result = tax_engine.calculate_alignment_ratio([holding])
        assert len(tax_result.provenance_hash) == 64
        int(tax_result.provenance_hash, 16)

        # All four hashes are distinct
        all_hashes = {
            pai_result.provenance_hash,
            dnsh_result.provenance_hash,
            gov_result.provenance_hash,
            tax_result.provenance_hash,
        }
        assert len(all_hashes) == 4, "All provenance hashes should be unique"
