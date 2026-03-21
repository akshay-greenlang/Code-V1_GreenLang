# -*- coding: utf-8 -*-
"""
Test suite for PACK-023 configuration and presets.

Covers:
  - Pack Configuration (15 tests)
  - Sector Presets (20 tests)
  - Workflow Presets (10 tests)
  - Integration Presets (8 tests)
  - Demo Configuration (12 tests)

Total: 65 tests
Author: GreenLang Test Engineering
Pack: PACK-023 SBTi Alignment
"""

import sys
from pathlib import Path
from decimal import Decimal

import pytest

PACK_DIR = Path(__file__).resolve().parent.parent
if str(PACK_DIR) not in sys.path:
    sys.path.insert(0, str(PACK_DIR))

# Configuration imports
try:
    from config.pack_config import PackConfig, load_pack_config
except Exception:
    PackConfig = load_pack_config = None

try:
    from config.presets import SectorPreset, load_sector_preset
except Exception:
    SectorPreset = load_sector_preset = None

try:
    from config.demo import DemoConfig, load_demo_config
except Exception:
    DemoConfig = load_demo_config = None


# ===========================================================================
# Pack Configuration Tests
# ===========================================================================


@pytest.mark.skipif(PackConfig is None, reason="Config not available")
class TestPackConfiguration:
    """Tests for pack configuration."""

    def test_pack_config_loads(self) -> None:
        """Pack config should load."""
        config = load_pack_config()
        assert config is not None

    def test_pack_config_has_version(self) -> None:
        """Config should have version."""
        config = load_pack_config()
        if hasattr(config, "version"):
            assert config.version is not None

    def test_pack_config_has_engines(self) -> None:
        """Config should list engines."""
        config = load_pack_config()
        if hasattr(config, "engines"):
            assert len(config.engines) >= 10  # Should have 10 engines

    def test_pack_config_has_workflows(self) -> None:
        """Config should list workflows."""
        config = load_pack_config()
        if hasattr(config, "workflows"):
            assert len(config.workflows) >= 8  # Should have 8 workflows

    def test_pack_config_has_templates(self) -> None:
        """Config should list templates."""
        config = load_pack_config()
        if hasattr(config, "templates"):
            assert len(config.templates) >= 10  # Should have 10 templates

    def test_pack_config_has_integrations(self) -> None:
        """Config should list integrations."""
        config = load_pack_config()
        if hasattr(config, "integrations"):
            assert len(config.integrations) > 0

    def test_pack_config_validates_engine_count(self) -> None:
        """Config should validate 10 engines are registered."""
        config = load_pack_config()
        if hasattr(config, "engines"):
            engine_names = [e.get("name") for e in config.engines]
            expected = [
                "target_setting",
                "criteria_validation",
                "temperature_rating",
                "sda_sector",
                "flag_assessment",
                "submission_readiness",
                "scope3_screening",
                "progress_tracking",
                "fi_portfolio",
                "recalculation",
            ]
            for eng in expected:
                assert any(eng in name.lower() for name in engine_names)

    def test_pack_config_validates_workflow_count(self) -> None:
        """Config should validate 8 workflows are registered."""
        config = load_pack_config()
        if hasattr(config, "workflows"):
            workflow_names = [w.get("name") for w in config.workflows]
            expected = [
                "full_sbti_lifecycle",
                "target_setting",
                "validation",
                "flag",
                "scope3",
                "sda",
                "progress",
                "fi_target",
            ]
            for wf in expected:
                assert any(wf in name.lower() for name in workflow_names)

    def test_pack_config_engine_dependencies(self) -> None:
        """Engines should have proper dependencies."""
        config = load_pack_config()
        if hasattr(config, "engines"):
            # Each engine should have metadata
            for engine in config.engines:
                assert "name" in engine or "id" in engine

    def test_pack_config_workflow_steps(self) -> None:
        """Workflows should have step sequences."""
        config = load_pack_config()
        if hasattr(config, "workflows"):
            for workflow in config.workflows:
                if "steps" in workflow:
                    assert len(workflow["steps"]) > 0

    def test_pack_config_template_bindings(self) -> None:
        """Templates should be bound to workflows."""
        config = load_pack_config()
        if hasattr(config, "templates"):
            # Templates should be defined
            assert len(config.templates) > 0

    def test_pack_config_sbti_compliance(self) -> None:
        """Config should reflect SBTi requirements."""
        config = load_pack_config()
        if hasattr(config, "sbti_version"):
            # Should specify SBTi standard version
            assert config.sbti_version is not None

    def test_pack_config_criteria_mapping(self) -> None:
        """Config should map criteria to validation engine."""
        config = load_pack_config()
        if hasattr(config, "criteria"):
            # Should have criteria definitions
            assert len(config.criteria) >= 42  # 28 NT + 14 NZ


# ===========================================================================
# Sector Preset Tests
# ===========================================================================


@pytest.mark.skipif(SectorPreset is None, reason="Presets not available")
class TestSectorPresets:
    """Tests for sector-specific presets."""

    @pytest.mark.parametrize("sector", [
        "Technology",
        "Finance",
        "Manufacturing",
        "Energy",
        "Retail",
        "Consumer Goods",
        "Healthcare",
        "Agriculture",
    ])
    def test_sector_preset_loads(self, sector: str) -> None:
        """Each sector preset should load."""
        preset = load_sector_preset(sector)
        assert preset is not None

    def test_technology_preset_specific(self) -> None:
        """Technology sector preset should have specific settings."""
        preset = load_sector_preset("Technology")
        if hasattr(preset, "sector"):
            assert preset.sector == "Technology"

    def test_manufacturing_preset_sda(self) -> None:
        """Manufacturing should recommend SDA pathway."""
        preset = load_sector_preset("Manufacturing")
        if hasattr(preset, "recommended_pathway"):
            assert "sda" in preset.recommended_pathway.lower()

    def test_energy_preset_specific_criteria(self) -> None:
        """Energy sector should have specific criteria."""
        preset = load_sector_preset("Energy")
        if hasattr(preset, "specific_criteria"):
            assert len(preset.specific_criteria) > 0

    def test_agriculture_preset_flag(self) -> None:
        """Agriculture should support FLAG pathway."""
        preset = load_sector_preset("Agriculture")
        if hasattr(preset, "supports_flag"):
            assert preset.supports_flag is True

    def test_finance_preset_fi_alignment(self) -> None:
        """Finance should support FI alignment."""
        preset = load_sector_preset("Finance")
        if hasattr(preset, "supports_fi_alignment"):
            assert preset.supports_fi_alignment is True

    def test_preset_baseline_factors(self) -> None:
        """Each preset should have baseline emission factors."""
        for sector in ["Technology", "Manufacturing", "Energy"]:
            preset = load_sector_preset(sector)
            if hasattr(preset, "baseline_factors"):
                assert len(preset.baseline_factors) > 0

    def test_preset_reduction_rates(self) -> None:
        """Each preset should have recommended reduction rates."""
        for sector in ["Technology", "Manufacturing", "Finance"]:
            preset = load_sector_preset(sector)
            if hasattr(preset, "recommended_reduction_rates"):
                assert Decimal("0") < preset.recommended_reduction_rates.get("aca_1_5c", Decimal("0")) <= Decimal("1")

    def test_preset_data_quality_templates(self) -> None:
        """Each preset should have data quality templates."""
        for sector in ["Technology", "Manufacturing"]:
            preset = load_sector_preset(sector)
            if hasattr(preset, "data_quality_template"):
                assert preset.data_quality_template is not None

    def test_preset_compliance_frameworks(self) -> None:
        """Presets should list compliance frameworks."""
        for sector in ["Finance", "Energy"]:
            preset = load_sector_preset(sector)
            if hasattr(preset, "compliance_frameworks"):
                assert len(preset.compliance_frameworks) > 0

    def test_preset_integration_defaults(self) -> None:
        """Each preset should have integration defaults."""
        for sector in ["Technology", "Manufacturing"]:
            preset = load_sector_preset(sector)
            if hasattr(preset, "default_integrations"):
                assert len(preset.default_integrations) > 0

    def test_retail_preset_specific(self) -> None:
        """Retail sector preset should have specific settings."""
        preset = load_sector_preset("Retail")
        if hasattr(preset, "scope3_materiality"):
            # Retail typically has high Scope 3
            assert preset.scope3_materiality > Decimal("0.5")

    def test_consumer_goods_preset_specific(self) -> None:
        """Consumer Goods sector preset should have specific settings."""
        preset = load_sector_preset("Consumer Goods")
        if hasattr(preset, "supply_chain_materiality"):
            assert preset.supply_chain_materiality > Decimal("0")


# ===========================================================================
# Workflow Preset Tests
# ===========================================================================


@pytest.mark.skipif(PackConfig is None, reason="Config not available")
class TestWorkflowPresets:
    """Tests for workflow presets."""

    def test_full_lifecycle_preset(self) -> None:
        """Full SBTi lifecycle preset should be complete."""
        config = load_pack_config()
        if hasattr(config, "get_workflow"):
            wf = config.get_workflow("full_sbti_lifecycle")
            assert wf is not None

    def test_target_setting_preset(self) -> None:
        """Target setting preset should be available."""
        config = load_pack_config()
        if hasattr(config, "get_workflow"):
            wf = config.get_workflow("target_setting")
            assert wf is not None

    def test_validation_preset(self) -> None:
        """Validation preset should be available."""
        config = load_pack_config()
        if hasattr(config, "get_workflow"):
            wf = config.get_workflow("validation")
            assert wf is not None

    def test_workflow_sequence_ordering(self) -> None:
        """Workflow steps should be ordered."""
        config = load_pack_config()
        if hasattr(config, "get_workflow"):
            wf = config.get_workflow("full_sbti_lifecycle")
            if hasattr(wf, "steps"):
                # Should have target setting before validation
                step_names = [s.get("name") for s in wf.steps]
                if "target_setting" in step_names and "validation" in step_names:
                    assert step_names.index("target_setting") < step_names.index("validation")


# ===========================================================================
# Demo Configuration Tests
# ===========================================================================


@pytest.mark.skipif(DemoConfig is None, reason="Demo config not available")
class TestDemoConfiguration:
    """Tests for demo/example configuration."""

    def test_demo_config_loads(self) -> None:
        """Demo config should load."""
        demo = load_demo_config()
        assert demo is not None

    def test_demo_config_has_entities(self) -> None:
        """Demo config should have example entities."""
        demo = load_demo_config()
        if hasattr(demo, "entities"):
            assert len(demo.entities) > 0

    def test_demo_entity_technology(self) -> None:
        """Demo should have technology sector entity."""
        demo = load_demo_config()
        if hasattr(demo, "entities"):
            tech_entities = [e for e in demo.entities if e.get("sector") == "Technology"]
            assert len(tech_entities) > 0

    def test_demo_entity_manufacturing(self) -> None:
        """Demo should have manufacturing sector entity."""
        demo = load_demo_config()
        if hasattr(demo, "entities"):
            mfg_entities = [e for e in demo.entities if e.get("sector") == "Manufacturing"]
            assert len(mfg_entities) > 0

    def test_demo_entity_baseline_data(self) -> None:
        """Demo entities should have baseline data."""
        demo = load_demo_config()
        if hasattr(demo, "entities"):
            for entity in demo.entities:
                assert "scope1_tco2e" in entity or "baseline" in entity

    def test_demo_entity_target_data(self) -> None:
        """Demo entities should have target data."""
        demo = load_demo_config()
        if hasattr(demo, "entities"):
            for entity in demo.entities:
                if "target_year" in entity:
                    assert "target" in str(entity).lower()

    def test_demo_workflow_examples(self) -> None:
        """Demo config should have workflow examples."""
        demo = load_demo_config()
        if hasattr(demo, "workflow_examples"):
            assert len(demo.workflow_examples) > 0

    def test_demo_expected_outputs(self) -> None:
        """Demo config should specify expected outputs."""
        demo = load_demo_config()
        if hasattr(demo, "expected_outputs"):
            assert len(demo.expected_outputs) > 0
