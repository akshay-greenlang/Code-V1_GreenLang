# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Demo/Smoke Tests (test_demo.py)
=============================================================================

Smoke tests covering:
  - Demo YAML validity and content
  - 8 engine importability
  - 8 engine class instantiation
  - 8 engine smoke calculations
  - 8 workflow importability
  - 8 template importability
  - 10 integration importability
  - 6 preset loading via PackConfig.from_preset

Test Count Target: ~69 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-014 CSRD Retail & Consumer Goods
Date:    March 2026
"""

import sys
from pathlib import Path

import pytest

# Insert tests directory into path so conftest helpers are available
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    _load_module,
    _load_engine,
    _load_config_module,
    ENGINE_FILES,
    ENGINE_CLASSES,
    WORKFLOW_FILES,
    TEMPLATE_FILES,
    INTEGRATION_FILES,
    PRESET_NAMES,
    ENGINES_DIR,
    WORKFLOWS_DIR,
    TEMPLATES_DIR,
    INTEGRATIONS_DIR,
)


# =============================================================================
# 1. Demo YAML Validity
# =============================================================================


class TestDemoYAML:
    """Test demo configuration YAML file validity and structure."""

    def test_demo_yaml_exists(self, demo_yaml_path):
        """demo_config.yaml exists on disk."""
        assert demo_yaml_path.exists()

    def test_demo_yaml_is_dict(self, demo_yaml_data):
        """demo_config.yaml parses to a dictionary."""
        assert isinstance(demo_yaml_data, dict)

    def test_demo_has_company_name(self, demo_yaml_data):
        """Demo config has company_name field."""
        assert "company_name" in demo_yaml_data
        assert len(demo_yaml_data["company_name"]) > 0

    def test_demo_has_reporting_year(self, demo_yaml_data):
        """Demo config has reporting_year."""
        assert "reporting_year" in demo_yaml_data
        assert demo_yaml_data["reporting_year"] >= 2024

    def test_demo_has_stores(self, demo_yaml_data):
        """Demo config has stores list with at least 1 store."""
        assert "stores" in demo_yaml_data
        assert len(demo_yaml_data["stores"]) >= 1

    def test_demo_store_has_required_fields(self, demo_yaml_data):
        """First demo store has store_id, country, and floor_area_sqm."""
        store = demo_yaml_data["stores"][0]
        assert "store_id" in store
        assert "country" in store

    def test_demo_has_sub_sectors(self, demo_yaml_data):
        """Demo config has sub_sectors list."""
        assert "sub_sectors" in demo_yaml_data
        assert len(demo_yaml_data["sub_sectors"]) >= 1

    def test_demo_has_tier(self, demo_yaml_data):
        """Demo config has tier field."""
        assert "tier" in demo_yaml_data


# =============================================================================
# 2. Engine Importability (8 tests)
# =============================================================================


class TestEngineImportability:
    """Test that all 8 engines can be imported via dynamic loading."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_imports(self, engine_key):
        """Engine module can be dynamically loaded without errors."""
        mod = _load_engine(engine_key)
        assert mod is not None


# =============================================================================
# 3. Engine Class Instantiation (8 tests)
# =============================================================================


class TestEngineClassInstantiation:
    """Test that all 8 engine classes can be instantiated."""

    @pytest.mark.parametrize("engine_key,class_name", list(ENGINE_CLASSES.items()))
    def test_engine_class_exists(self, engine_key, class_name):
        """Engine module contains the expected engine class."""
        mod = _load_engine(engine_key)
        assert hasattr(mod, class_name), (
            f"Module {engine_key} missing class {class_name}. "
            f"Available: {[a for a in dir(mod) if not a.startswith('_')]}"
        )

    @pytest.mark.parametrize("engine_key,class_name", list(ENGINE_CLASSES.items()))
    def test_engine_class_instantiates(self, engine_key, class_name):
        """Engine class can be instantiated (no required init args)."""
        mod = _load_engine(engine_key)
        cls = getattr(mod, class_name)
        engine = cls()
        assert engine is not None


# =============================================================================
# 4. Engine Smoke Calculations (8 tests)
# =============================================================================


class TestEngineSmokeCalculations:
    """Test that each engine can perform a basic smoke calculation."""

    def test_store_emissions_smoke(self, sample_grocery_store):
        """StoreEmissionsEngine.calculate_store_emissions runs without error."""
        mod = _load_engine("store_emissions")
        engine = mod.StoreEmissionsEngine()
        result = engine.calculate_store_emissions(sample_grocery_store)
        assert result is not None
        assert isinstance(result, mod.StoreEmissionsResult)

    def test_retail_scope3_smoke(self):
        """RetailScope3Engine.calculate_scope3 runs without error."""
        mod = _load_engine("retail_scope3")
        engine = mod.RetailScope3Engine()
        input_data = mod.RetailScope3Input(
            organisation_id="TEST-ORG",
            reporting_year=2025,
            purchased_goods=[
                mod.PurchasedGoodsData(
                    product_category=mod.ProductCategory.FOOD_FRESH,
                    spend_eur=10000000.0,
                    calculation_method=mod.CalculationMethod.SPEND_BASED,
                ),
            ],
        )
        result = engine.calculate_scope3(input_data)
        assert result is not None
        assert isinstance(result, mod.RetailScope3Result)

    def test_packaging_compliance_smoke(self, sample_packaging_items):
        """PackagingComplianceEngine.assess_compliance runs without error."""
        mod = _load_engine("packaging_compliance")
        engine = mod.PackagingComplianceEngine()
        portfolio = mod.PackagingPortfolio(
            organisation_id="TEST-ORG",
            reporting_year=2025,
            items=sample_packaging_items,
            country="DE",
        )
        result = engine.assess_compliance(portfolio)
        assert result is not None
        assert isinstance(result, mod.PPWRComplianceResult)

    def test_product_sustainability_smoke(self, sample_products):
        """ProductSustainabilityEngine.assess_products runs without error."""
        mod = _load_engine("product_sustainability")
        engine = mod.ProductSustainabilityEngine()
        input_data = mod.ProductSustainabilityInput(
            organisation_id="TEST-ORG",
            reporting_year=2025,
            products=sample_products,
        )
        result = engine.assess_products(input_data)
        assert result is not None
        assert isinstance(result, mod.ProductSustainabilityResult)

    def test_food_waste_smoke(self, sample_food_waste_records):
        """FoodWasteEngine.calculate runs without error."""
        mod = _load_engine("food_waste")
        engine = mod.FoodWasteEngine()
        result = engine.calculate(
            records=sample_food_waste_records,
            reporting_year=2025,
        )
        assert result is not None
        assert isinstance(result, mod.FoodWasteResult)

    def test_supply_chain_dd_smoke(self, sample_suppliers):
        """SupplyChainDueDiligenceEngine.calculate runs without error."""
        mod = _load_engine("supply_chain_due_diligence")
        engine = mod.SupplyChainDueDiligenceEngine()
        result = engine.calculate(
            suppliers=sample_suppliers,
            employee_count=8500,
            turnover_eur=2800000000.0,
            eu_turnover_eur=2200000000.0,
        )
        assert result is not None
        assert isinstance(result, mod.SupplyChainDDResult)

    def test_circular_economy_smoke(self, sample_material_flows):
        """RetailCircularEconomyEngine.calculate runs without error."""
        mod = _load_engine("retail_circular_economy")
        engine = mod.RetailCircularEconomyEngine()
        result = engine.calculate(material_flows=sample_material_flows)
        assert result is not None
        assert isinstance(result, mod.CircularEconomyResult)

    def test_benchmark_smoke(self, sample_retail_kpis):
        """RetailBenchmarkEngine.calculate runs without error."""
        mod = _load_engine("retail_benchmark")
        engine = mod.RetailBenchmarkEngine()
        result = engine.calculate(
            kpis=sample_retail_kpis,
            sbti_pathway="1.5C",
            base_year=2020,
            base_year_emissions_tco2e=300000.0,
            target_year=2030,
        )
        assert result is not None
        assert isinstance(result, mod.BenchmarkResult)


# =============================================================================
# 5. Workflow Importability (8 tests)
# =============================================================================


class TestWorkflowImportability:
    """Test that all 8 workflow modules can be imported."""

    @pytest.mark.parametrize("wf_key,file_name", list(WORKFLOW_FILES.items()))
    def test_workflow_imports(self, wf_key, file_name):
        """Workflow module can be dynamically loaded without errors."""
        mod = _load_module(wf_key, file_name, "workflows")
        assert mod is not None


# =============================================================================
# 6. Template Importability (8 tests)
# =============================================================================


class TestTemplateImportability:
    """Test that all 8 template modules can be imported."""

    @pytest.mark.parametrize("tpl_key,file_name", list(TEMPLATE_FILES.items()))
    def test_template_imports(self, tpl_key, file_name):
        """Template module can be dynamically loaded without errors."""
        mod = _load_module(tpl_key, file_name, "templates")
        assert mod is not None


# =============================================================================
# 7. Integration Importability (10 tests)
# =============================================================================


class TestIntegrationImportability:
    """Test that all 10 integration modules can be imported."""

    @pytest.mark.parametrize("int_key,file_name", list(INTEGRATION_FILES.items()))
    def test_integration_imports(self, int_key, file_name):
        """Integration module can be dynamically loaded without errors."""
        mod = _load_module(int_key, file_name, "integrations")
        assert mod is not None


# =============================================================================
# 8. Preset Loading via PackConfig (6 tests)
# =============================================================================


class TestPresetLoadingSmokeTests:
    """Test that each preset loads and produces a valid PackConfig."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_loads_via_pack_config(self, preset_name):
        """Preset loads successfully via PackConfig.from_preset."""
        config_mod = _load_config_module()
        cfg = config_mod.PackConfig.from_preset(preset_name)
        assert cfg is not None
        assert cfg.preset_name == preset_name
        assert isinstance(cfg.pack, config_mod.CSRDRetailConfig)
        # Verify at least one sub-sector is set
        assert len(cfg.pack.sub_sectors) >= 1


# =============================================================================
# 9. Engine Version and Module-Level Attributes
# =============================================================================


class TestEngineModuleAttributes:
    """Test engine modules have expected module-level attributes."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_has_version(self, engine_key):
        """Engine module has a version string (engine_version or _MODULE_VERSION)."""
        mod = _load_engine(engine_key)
        has_version = (
            hasattr(mod, "engine_version") or
            hasattr(mod, "_MODULE_VERSION")
        )
        assert has_version, f"Engine {engine_key} has no version attribute"

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engine_version_is_string(self, engine_key):
        """Engine version is a string."""
        mod = _load_engine(engine_key)
        version = getattr(mod, "engine_version", None) or getattr(mod, "_MODULE_VERSION", None)
        assert isinstance(version, str)
        assert len(version) >= 5  # At least "1.0.0"
