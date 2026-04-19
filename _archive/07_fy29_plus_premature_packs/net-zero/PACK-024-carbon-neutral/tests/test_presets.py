# -*- coding: utf-8 -*-
"""
Tests for PACK-024 Carbon Neutral Pack presets (8 presets).

Covers:
  - Corporate Neutrality Preset (8 tests)
  - SME Neutrality Preset (8 tests)
  - Event Neutrality Preset (8 tests)
  - Product Neutrality Preset (8 tests)
  - Building Neutrality Preset (8 tests)
  - Service Neutrality Preset (8 tests)
  - Project Neutrality Preset (8 tests)
  - Portfolio Neutrality Preset (8 tests)

Total: 64 tests
"""
import sys
from pathlib import Path
import pytest
import yaml

PACK_DIR = Path(__file__).resolve().parent.parent
PRESETS_DIR = PACK_DIR / "config" / "presets"
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

PRESET_NAMES = [
    "corporate_neutrality", "sme_neutrality", "event_neutrality",
    "product_neutrality", "building_neutrality", "service_neutrality",
    "project_neutrality", "portfolio_neutrality",
]


def load_preset_yaml(name):
    path = PRESETS_DIR / f"{name}.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


# ===========================================================================
# Corporate Neutrality Preset
# ===========================================================================
class TestCorporateNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("corporate_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "corporate_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "CORPORATE"
    def test_scope3_included(self, preset):
        scope = preset.get("scope_boundaries", {}).get("scope3", {})
        assert scope.get("included") is True
    def test_engines_configured(self, preset): assert "engines" in preset
    def test_footprint_enabled(self, preset): assert preset["engines"]["footprint_quantification"]["enabled"] is True
    def test_credit_quality_min(self, preset): assert preset["engines"]["credit_quality"]["min_quality_score"] >= 60
    def test_has_finance_section(self, preset): assert "finance" in preset


# ===========================================================================
# SME Neutrality Preset
# ===========================================================================
class TestSMENeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("sme_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "sme_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "SME"
    def test_scope3_excluded(self, preset):
        scope = preset.get("scope_boundaries", {}).get("scope3", {})
        assert scope.get("included") is False
    def test_simplified_engines(self, preset): assert "engines" in preset
    def test_lower_quality_threshold(self, preset): assert preset["engines"]["credit_quality"]["min_quality_score"] <= 60
    def test_smaller_supplier_count(self, preset):
        supplier = preset.get("supplier", {})
        assert supplier.get("max_suppliers", 100) <= 200
    def test_has_data_quality(self, preset): assert "data_quality" in preset


# ===========================================================================
# Event Neutrality Preset
# ===========================================================================
class TestEventNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("event_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "event_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "EVENT"
    def test_event_specific_sources(self, preset):
        scope3 = preset.get("scope_boundaries", {}).get("scope3", {})
        assert "event_specific_sources" in scope3
    def test_event_parameters(self, preset): assert "event_parameters" in preset
    def test_attendee_travel_configured(self, preset):
        params = preset.get("event_parameters", {})
        assert "max_attendees" in params
    def test_event_balance_method(self, preset):
        nb = preset.get("engines", {}).get("neutralization_balance", {})
        assert nb.get("balance_method") == "event_total"
    def test_event_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "event" in cs.get("claim_type", "").lower()


# ===========================================================================
# Product Neutrality Preset
# ===========================================================================
class TestProductNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("product_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "product_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "PRODUCT"
    def test_lca_configuration(self, preset): assert "lca" in preset
    def test_system_boundary(self, preset): assert preset["lca"]["system_boundary"] is not None
    def test_functional_unit(self, preset): assert preset["lca"]["functional_unit"] is not None
    def test_product_balance_method(self, preset):
        nb = preset.get("engines", {}).get("neutralization_balance", {})
        assert nb.get("balance_method") == "per_unit_produced"
    def test_product_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "product" in cs.get("claim_type", "").lower()


# ===========================================================================
# Building Neutrality Preset
# ===========================================================================
class TestBuildingNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("building_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "building_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "BUILDING"
    def test_building_parameters(self, preset): assert "building_parameters" in preset
    def test_crrem_alignment(self, preset):
        params = preset.get("building_parameters", {})
        assert "crrem_pathway_aligned" in params
    def test_energy_rating(self, preset):
        params = preset.get("building_parameters", {})
        assert "energy_rating" in params
    def test_building_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "building" in cs.get("claim_type", "").lower()
    def test_has_multi_building_section(self, preset): assert "multi_building" in preset


# ===========================================================================
# Service Neutrality Preset
# ===========================================================================
class TestServiceNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("service_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "service_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "SERVICE"
    def test_service_parameters(self, preset): assert "service_parameters" in preset
    def test_remote_work_configured(self, preset):
        params = preset.get("service_parameters", {})
        assert "remote_work_pct" in params
    def test_cloud_provider(self, preset):
        params = preset.get("service_parameters", {})
        assert "cloud_provider" in params
    def test_service_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "service" in cs.get("claim_type", "").lower()
    def test_scope3_included(self, preset):
        scope3 = preset.get("scope_boundaries", {}).get("scope3", {})
        assert scope3.get("included") is True


# ===========================================================================
# Project Neutrality Preset
# ===========================================================================
class TestProjectNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("project_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "project_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "PROJECT"
    def test_project_parameters(self, preset): assert "project_parameters" in preset
    def test_project_type(self, preset):
        params = preset.get("project_parameters", {})
        assert "project_type" in params
    def test_carbon_budget(self, preset):
        params = preset.get("project_parameters", {})
        assert "carbon_budget_tco2e" in params
    def test_project_balance_method(self, preset):
        nb = preset.get("engines", {}).get("neutralization_balance", {})
        assert nb.get("balance_method") == "project_total"
    def test_project_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "project" in cs.get("claim_type", "").lower()


# ===========================================================================
# Portfolio Neutrality Preset
# ===========================================================================
class TestPortfolioNeutralityPreset:
    @pytest.fixture
    def preset(self): return load_preset_yaml("portfolio_neutrality")

    def test_file_exists(self): assert (PRESETS_DIR / "portfolio_neutrality.yaml").exists()
    def test_loads_valid_yaml(self, preset): assert preset is not None
    def test_neutrality_type(self, preset): assert preset.get("neutrality_type") == "PORTFOLIO"
    def test_portfolio_parameters(self, preset): assert "portfolio_parameters" in preset
    def test_max_entities(self, preset):
        params = preset.get("portfolio_parameters", {})
        assert params.get("max_entities", 0) >= 10
    def test_intercompany_elimination(self, preset):
        params = preset.get("portfolio_parameters", {})
        assert params.get("intercompany_elimination") is True
    def test_portfolio_claim_type(self, preset):
        cs = preset.get("engines", {}).get("claims_substantiation", {})
        assert "portfolio" in cs.get("claim_type", "").lower()
    def test_entity_level_balance(self, preset):
        nb = preset.get("engines", {}).get("neutralization_balance", {})
        assert nb.get("entity_level_balance") is True
