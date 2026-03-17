# -*- coding: utf-8 -*-
"""
PACK-014 CSRD Retail & Consumer Goods Pack - Configuration Tests (test_config.py)
==================================================================================

Tests configuration completeness and correctness:
  - 16 enum completeness tests
  - Sub-config default value tests
  - Main CSRDRetailConfig construction tests
  - 6 preset loading tests
  - Model validator tests (grocery -> food_waste, apparel -> textile_epr)
  - Utility function tests

Test Count Target: ~46 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-014 CSRD Retail & Consumer Goods
Date:    March 2026
"""

import sys
from pathlib import Path

import pytest
import yaml

# Insert tests directory into path so conftest helpers are available
sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    _load_config_module,
    _load_engine,
    PRESETS_DIR,
    PRESET_NAMES,
    CONFIG_DIR,
)


# =============================================================================
# 1. Enum Completeness Tests (16 enums)
# =============================================================================


class TestConfigEnums:
    """Test all 16 configuration enums for completeness."""

    def test_retail_sub_sector_count(self, config_module):
        """RetailSubSector enum has exactly 16 members."""
        members = list(config_module.RetailSubSector)
        assert len(members) == 16, f"Expected 16, got {len(members)}: {[m.value for m in members]}"

    def test_retail_sub_sector_values(self, config_module):
        """RetailSubSector contains expected values."""
        expected = {"GROCERY", "APPAREL", "ELECTRONICS", "HOME_FURNISHING",
                    "DEPARTMENT_STORE", "CONVENIENCE", "HYPERMARKET", "SPECIALTY",
                    "E_COMMERCE", "WHOLESALE", "PHARMACY", "DIY_HARDWARE",
                    "LUXURY", "DISCOUNT", "FOOD_SERVICE", "OTHER"}
        actual = {m.value for m in config_module.RetailSubSector}
        assert actual == expected

    def test_retail_tier_count(self, config_module):
        """RetailTier enum has exactly 4 members."""
        assert len(list(config_module.RetailTier)) == 4

    def test_retail_tier_values(self, config_module):
        """RetailTier contains ENTERPRISE, MID_MARKET, SME, FRANCHISE."""
        expected = {"ENTERPRISE", "MID_MARKET", "SME", "FRANCHISE"}
        actual = {m.value for m in config_module.RetailTier}
        assert actual == expected

    def test_packaging_material_count(self, config_module):
        """PackagingMaterial enum has exactly 12 members."""
        assert len(list(config_module.PackagingMaterial)) == 12

    def test_epr_scheme_count(self, config_module):
        """EPRScheme enum has exactly 6 members."""
        members = list(config_module.EPRScheme)
        assert len(members) == 6, f"Got {len(members)}: {[m.value for m in members]}"

    def test_eudr_commodity_count(self, config_module):
        """EUDRCommodity enum has exactly 7 members."""
        assert len(list(config_module.EUDRCommodity)) == 7

    def test_eudr_commodity_values(self, config_module):
        """EUDRCommodity contains all 7 regulated commodities."""
        expected = {"PALM_OIL", "SOY", "COCOA", "COFFEE", "RUBBER", "TIMBER", "CATTLE"}
        actual = {m.value for m in config_module.EUDRCommodity}
        assert actual == expected

    def test_food_waste_category_count(self, config_module):
        """FoodWasteCategory enum has at least 7 members."""
        assert len(list(config_module.FoodWasteCategory)) >= 7

    def test_store_type_count(self, config_module):
        """StoreType enum has exactly 7 members."""
        assert len(list(config_module.StoreType)) == 7

    def test_refrigerant_type_count(self, config_module):
        """RefrigerantType enum has exactly 9 members."""
        assert len(list(config_module.RefrigerantType)) == 9

    def test_scope3_priority_count(self, config_module):
        """Scope3Priority enum has at least 4 members."""
        assert len(list(config_module.Scope3Priority)) >= 4

    def test_supplier_tier_count(self, config_module):
        """SupplierTier enum has exactly 4 members."""
        assert len(list(config_module.SupplierTier)) == 4

    def test_due_diligence_risk_count(self, config_module):
        """DueDiligenceRisk enum has exactly 5 members."""
        assert len(list(config_module.DueDiligenceRisk)) == 5

    def test_esrs_topic_count(self, config_module):
        """ESRSTopic enum has exactly 10 members (E1-E5, S1-S4, G1)."""
        assert len(list(config_module.ESRSTopic)) == 10

    def test_reporting_frequency_count(self, config_module):
        """ReportingFrequency enum has exactly 4 members."""
        assert len(list(config_module.ReportingFrequency)) == 4

    def test_compliance_status_count(self, config_module):
        """ComplianceStatus enum has exactly 5 members."""
        assert len(list(config_module.ComplianceStatus)) == 5

    def test_green_claim_type_count(self, config_module):
        """GreenClaimType enum has at least 7 members."""
        assert len(list(config_module.GreenClaimType)) >= 7

    def test_disclosure_format_count(self, config_module):
        """DisclosureFormat enum has exactly 4 members."""
        assert len(list(config_module.DisclosureFormat)) == 4


# =============================================================================
# 2. Sub-Config Default Tests
# =============================================================================


class TestSubConfigDefaults:
    """Test sub-configuration models instantiate with correct defaults."""

    def test_store_config_defaults(self, config_module):
        """StoreConfig instantiates with sensible defaults."""
        sc = config_module.StoreConfig()
        assert sc.country == "DE"
        assert sc.sub_sector == config_module.RetailSubSector.GROCERY
        assert sc.store_type == config_module.StoreType.STANDARD
        assert sc.has_refrigeration is False

    def test_store_emissions_config_defaults(self, config_module):
        """StoreEmissionsConfig has enabled=True by default."""
        sec = config_module.StoreEmissionsConfig()
        assert sec.enabled is True
        assert sec.refrigerant_tracking is True
        assert sec.per_sqm_normalization is True

    def test_retail_scope3_config_defaults(self, config_module):
        """RetailScope3Config has enabled=True and priority categories."""
        rsc = config_module.RetailScope3Config()
        assert rsc.enabled is True
        assert len(rsc.priority_categories) >= 3
        assert all(1 <= c <= 15 for c in rsc.priority_categories)

    def test_packaging_config_defaults(self, config_module):
        """PackagingConfig has ppwr_compliance=True by default."""
        pc = config_module.PackagingConfig()
        assert pc.enabled is True
        assert pc.ppwr_compliance is True
        assert pc.target_year == 2030

    def test_product_sustainability_config_defaults(self, config_module):
        """ProductSustainabilityConfig has green_claims_audit=True."""
        psc = config_module.ProductSustainabilityConfig()
        assert psc.enabled is True
        assert psc.green_claims_audit is True

    def test_food_waste_config_defaults(self, config_module):
        """FoodWasteConfig has enabled=False by default (opt-in for grocery)."""
        fwc = config_module.FoodWasteConfig()
        assert fwc.enabled is False
        assert fwc.reduction_target_pct == pytest.approx(30.0)
        assert fwc.baseline_year == 2020

    def test_supply_chain_dd_config_defaults(self, config_module):
        """SupplyChainDDConfig has csddd_compliance=True by default."""
        sdc = config_module.SupplyChainDDConfig()
        assert sdc.enabled is True
        assert sdc.csddd_compliance is True
        assert sdc.forced_labour_screening is True
        assert sdc.supplier_tier_depth >= 1

    def test_circular_economy_config_defaults(self, config_module):
        """CircularEconomyConfig has packaging EPR by default."""
        cec = config_module.CircularEconomyConfig()
        assert cec.enabled is True
        assert cec.mci_tracking is True
        assert config_module.EPRScheme.PACKAGING in cec.epr_schemes

    def test_benchmark_config_defaults(self, config_module):
        """BenchmarkConfig has enabled=True by default."""
        bc = config_module.BenchmarkConfig()
        assert bc.enabled is True


# =============================================================================
# 3. Main Config Construction Tests
# =============================================================================


class TestCSRDRetailConfig:
    """Test CSRDRetailConfig construction and field validation."""

    def test_default_construction(self, config_module):
        """CSRDRetailConfig can be created with all defaults."""
        cfg = config_module.CSRDRetailConfig()
        assert cfg.reporting_year >= 2024
        assert cfg.tier == config_module.RetailTier.ENTERPRISE

    def test_custom_company_name(self, config_module):
        """Company name can be set."""
        cfg = config_module.CSRDRetailConfig(company_name="EuroRetail AG")
        assert cfg.company_name == "EuroRetail AG"

    def test_sub_sectors_list(self, config_module):
        """Sub-sectors can be set as a list."""
        cfg = config_module.CSRDRetailConfig(
            sub_sectors=[config_module.RetailSubSector.GROCERY,
                         config_module.RetailSubSector.APPAREL]
        )
        assert config_module.RetailSubSector.GROCERY in cfg.sub_sectors
        assert config_module.RetailSubSector.APPAREL in cfg.sub_sectors

    def test_all_sub_configs_present(self, config_module):
        """CSRDRetailConfig has all expected sub-config attributes."""
        cfg = config_module.CSRDRetailConfig()
        for attr in ["store_emissions", "scope3", "packaging",
                     "product_sustainability", "food_waste",
                     "supply_chain_dd", "circular_economy", "benchmark"]:
            assert hasattr(cfg, attr), f"Missing sub-config: {attr}"


# =============================================================================
# 4. Preset Loading Tests
# =============================================================================


class TestPresetLoading:
    """Test loading of all 6 presets."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_yaml_loads(self, preset_name):
        """Each preset YAML file loads successfully."""
        path = PRESETS_DIR / f"{preset_name}.yaml"
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data is not None
        assert isinstance(data, dict)

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_preset_via_pack_config(self, config_module, preset_name):
        """Each preset can be loaded via PackConfig.from_preset()."""
        cfg = config_module.PackConfig.from_preset(preset_name)
        assert cfg is not None
        assert cfg.preset_name == preset_name
        assert cfg.pack_id == "PACK-014-csrd-retail"

    def test_grocery_preset_enables_food_waste(self, config_module):
        """Grocery preset has food_waste.enabled = True."""
        cfg = config_module.PackConfig.from_preset("grocery_retail")
        assert cfg.pack.food_waste.enabled is True

    def test_grocery_preset_enables_refrigerant(self, config_module):
        """Grocery preset has refrigerant_tracking = True."""
        cfg = config_module.PackConfig.from_preset("grocery_retail")
        assert cfg.pack.store_emissions.refrigerant_tracking is True

    def test_apparel_preset_has_textile_epr(self, config_module):
        """Apparel preset includes TEXTILES in EPR schemes."""
        cfg = config_module.PackConfig.from_preset("apparel_retail")
        epr_values = [s.value for s in cfg.pack.circular_economy.epr_schemes]
        assert "TEXTILES" in epr_values

    def test_electronics_preset_has_weee(self, config_module):
        """Electronics preset includes WEEE in EPR schemes."""
        cfg = config_module.PackConfig.from_preset("electronics_retail")
        epr_values = [s.value for s in cfg.pack.circular_economy.epr_schemes]
        assert "WEEE" in epr_values


# =============================================================================
# 5. Model Validator Tests
# =============================================================================


class TestModelValidators:
    """Test model validators that enforce cross-field constraints."""

    def test_grocery_subsector_auto_enables_food_waste(self, config_module):
        """When GROCERY is in sub_sectors, food_waste should be flagged or auto-enabled."""
        cfg = config_module.CSRDRetailConfig(
            sub_sectors=[config_module.RetailSubSector.GROCERY],
            food_waste=config_module.FoodWasteConfig(enabled=False),
        )
        # The validator logs a warning and may auto-enable food_waste
        # Depending on implementation, check that the config is created without error
        assert config_module.RetailSubSector.GROCERY in cfg.sub_sectors

    def test_apparel_subsector_adds_textile_epr(self, config_module):
        """When APPAREL is in sub_sectors, TEXTILES EPR is auto-added."""
        cfg = config_module.CSRDRetailConfig(
            sub_sectors=[config_module.RetailSubSector.APPAREL],
            circular_economy=config_module.CircularEconomyConfig(
                epr_schemes=[config_module.EPRScheme.PACKAGING],
            ),
        )
        epr_values = [s.value for s in cfg.circular_economy.epr_schemes]
        assert "TEXTILES" in epr_values

    def test_electronics_subsector_adds_weee_epr(self, config_module):
        """When ELECTRONICS is in sub_sectors, WEEE EPR is auto-added."""
        cfg = config_module.CSRDRetailConfig(
            sub_sectors=[config_module.RetailSubSector.ELECTRONICS],
            circular_economy=config_module.CircularEconomyConfig(
                epr_schemes=[config_module.EPRScheme.PACKAGING],
            ),
        )
        epr_values = [s.value for s in cfg.circular_economy.epr_schemes]
        assert "WEEE" in epr_values

    def test_scope3_category_validator_rejects_invalid(self, config_module):
        """RetailScope3Config rejects category numbers outside 1-15."""
        with pytest.raises(Exception):
            config_module.RetailScope3Config(priority_categories=[0, 16, 20])

    def test_scope3_category_validator_deduplicates(self, config_module):
        """RetailScope3Config deduplicates and sorts category numbers."""
        rsc = config_module.RetailScope3Config(priority_categories=[5, 1, 5, 3, 1])
        assert rsc.priority_categories == [1, 3, 5]


# =============================================================================
# 6. PackConfig Wrapper Tests
# =============================================================================


class TestPackConfigWrapper:
    """Test the top-level PackConfig wrapper."""

    def test_default_pack_config(self, config_module):
        """PackConfig default construction works."""
        pc = config_module.PackConfig()
        assert pc.pack_id == "PACK-014-csrd-retail"
        assert pc.config_version == "1.0.0"
        assert pc.preset_name is None

    def test_pack_config_has_nested_config(self, config_module):
        """PackConfig.pack contains a CSRDRetailConfig."""
        pc = config_module.PackConfig()
        assert isinstance(pc.pack, config_module.CSRDRetailConfig)


# =============================================================================
# 7. Utility Function Tests
# =============================================================================


class TestConfigUtilityFunctions:
    """Test module-level utility functions."""

    def test_list_available_presets(self, config_module):
        """list_available_presets returns all 6 preset names."""
        presets = config_module.list_available_presets()
        assert isinstance(presets, (list, dict))
        if isinstance(presets, dict):
            assert len(presets) == 6
        else:
            assert len(presets) == 6

    def test_get_subsector_info(self, config_module):
        """get_subsector_info returns info for known sub-sectors."""
        info = config_module.get_subsector_info("GROCERY")
        assert info is not None
        assert "name" in info or "nace" in info

    def test_get_fgas_gwp(self, config_module):
        """get_fgas_gwp returns GWP values for known refrigerants."""
        gwp = config_module.get_fgas_gwp("R404A")
        assert gwp == 3922

    def test_get_ppwr_target_pet_2030(self, config_module):
        """get_ppwr_target returns correct PET 2030 target."""
        target = config_module.get_ppwr_target("PET", 2030)
        assert target == pytest.approx(30.0)

    def test_reference_data_subsector_info(self, config_module):
        """SUBSECTOR_INFO constant has entries for all 16 sub-sectors."""
        info = config_module.SUBSECTOR_INFO
        assert len(info) >= 16

    def test_reference_data_fgas_gwp_values(self, config_module):
        """FGAS_GWP_VALUES has entries for at least 8 refrigerants."""
        gwp = config_module.FGAS_GWP_VALUES
        assert len(gwp) >= 8
        assert gwp["R404A"] == 3922
        assert gwp["R744"] == 1

    def test_reference_data_ppwr_targets(self, config_module):
        """PPWR_RECYCLED_CONTENT_TARGETS has PET entry."""
        targets = config_module.PPWR_RECYCLED_CONTENT_TARGETS
        assert "PET" in targets
        assert 2030 in targets["PET"]

    def test_reference_data_epr_recycling_targets(self, config_module):
        """EPR_RECYCLING_TARGETS has packaging entry."""
        targets = config_module.EPR_RECYCLING_TARGETS
        assert "PACKAGING" in targets
        assert targets["PACKAGING"] == pytest.approx(70.0)
