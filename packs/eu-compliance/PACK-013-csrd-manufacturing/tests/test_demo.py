# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Demo and Smoke Tests (test_demo.py)
=======================================================================

Tests the demo configuration and performs smoke tests on all engines,
workflows, templates, and integrations. Ensures that:
  - Demo YAML is valid and contains expected data
  - All engine files are importable via importlib
  - All engine classes can be instantiated with default config
  - All engines produce non-None results with minimal valid data
  - All workflows, templates, and integrations are importable

Test count: 24 test definitions, ~69 collected with parametrize expansions.

All modules are loaded dynamically via importlib to avoid package install.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-013 CSRD Manufacturing
Date:    March 2026
"""

import importlib
import importlib.util
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from conftest import (
    ENGINE_CLASSES,
    ENGINE_FILES,
    INTEGRATION_FILES,
    PACK_ROOT,
    PRESET_NAMES,
    TEMPLATE_FILES,
    WORKFLOW_FILES,
    _load_config_module,
    _load_engine,
    _load_module,
)


# =============================================================================
# Test Class: Demo YAML Validation
# =============================================================================


class TestDemoYAML:
    """Tests for the demo_config.yaml file."""

    def test_demo_yaml_exists(self, demo_yaml_path: Path):
        """Verify demo_config.yaml exists on disk."""
        assert demo_yaml_path.exists(), (
            f"demo_config.yaml not found at {demo_yaml_path}"
        )

    def test_demo_yaml_valid(self, demo_yaml_data: Dict[str, Any]):
        """Verify demo_config.yaml parses as valid YAML."""
        assert isinstance(demo_yaml_data, dict)
        assert len(demo_yaml_data) > 0

    def test_demo_company_name(self, demo_yaml_data: Dict[str, Any]):
        """Verify demo company name is GreenManufacturing GmbH."""
        assert demo_yaml_data.get("company_name") == "GreenManufacturing GmbH"

    def test_demo_has_facilities(self, demo_yaml_data: Dict[str, Any]):
        """Verify demo has exactly 3 facilities."""
        facilities = demo_yaml_data.get("facilities", [])
        assert len(facilities) == 3, (
            f"Expected 3 demo facilities, got {len(facilities)}"
        )

    def test_demo_sub_sectors(self, demo_yaml_data: Dict[str, Any]):
        """Verify demo sub-sectors include cement, automotive, chemicals."""
        sub_sectors = demo_yaml_data.get("sub_sectors", [])
        expected = {"CEMENT", "AUTOMOTIVE", "CHEMICALS"}
        actual = set(sub_sectors)
        assert expected == actual, (
            f"Demo sub-sectors mismatch: expected {expected}, got {actual}"
        )

    def test_demo_engines_enabled(self, demo_yaml_data: Dict[str, Any]):
        """Verify all 8 engines are enabled in the demo configuration."""
        engine_keys = [
            "process_emissions",
            "energy_intensity",
            "product_pcf",
            "circular_economy",
            "water_pollution",
            "bat_compliance",
            "supply_chain",
            "benchmark",
        ]
        for key in engine_keys:
            section = demo_yaml_data.get(key, {})
            assert section.get("enabled", False) is True, (
                f"Engine '{key}' is not enabled in demo config"
            )


# =============================================================================
# Test Class: Preset Loading
# =============================================================================


class TestPresetLoading:
    """Tests for loading all 6 presets."""

    @pytest.mark.parametrize("preset_name", PRESET_NAMES)
    def test_demo_presets_loadable(self, preset_name: str):
        """Verify each preset YAML file can be parsed without error."""
        preset_path = PACK_ROOT / "config" / "presets" / f"{preset_name}.yaml"
        assert preset_path.exists(), f"Preset file not found: {preset_path}"

        with open(preset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert data is not None, f"Preset '{preset_name}' parsed to None"
        assert isinstance(data, dict), (
            f"Preset '{preset_name}' is not a dict"
        )


# =============================================================================
# Test Class: Engine Importability
# =============================================================================


class TestEngineImports:
    """Tests that all engine files can be imported via importlib."""

    @pytest.mark.parametrize("engine_key", list(ENGINE_FILES.keys()))
    def test_engines_importable(self, engine_key: str):
        """Verify each engine module can be loaded via importlib."""
        mod = _load_engine(engine_key)
        assert mod is not None, (
            f"Failed to load engine module: {engine_key}"
        )

    @pytest.mark.parametrize(
        "engine_key,class_name",
        list(ENGINE_CLASSES.items()),
    )
    def test_engine_classes_exist(self, engine_key: str, class_name: str):
        """Verify each engine module contains its expected class."""
        mod = _load_engine(engine_key)
        assert hasattr(mod, class_name), (
            f"Engine module '{engine_key}' does not have class '{class_name}'"
        )
        cls = getattr(mod, class_name)
        assert callable(cls), (
            f"'{class_name}' in '{engine_key}' is not callable"
        )

    @pytest.mark.parametrize(
        "engine_key,class_name",
        list(ENGINE_CLASSES.items()),
    )
    def test_engine_init_default(self, engine_key: str, class_name: str):
        """Verify each engine can be instantiated with no arguments or defaults."""
        mod = _load_engine(engine_key)
        cls = getattr(mod, class_name)

        # Engines that require specific config objects
        config_constructors = {
            "water_pollution": lambda m: m.WaterPollutionConfig(reporting_year=2025),
            "bat_compliance": lambda m: m.BATConfig(reporting_year=2025),
            "supply_chain_emissions": lambda m: m.SupplyChainConfig(reporting_year=2025),
            "manufacturing_benchmark": lambda m: m.BenchmarkConfig(
                reporting_year=2025, sub_sector="cement"
            ),
        }

        if engine_key in config_constructors:
            config = config_constructors[engine_key](mod)
            instance = cls(config)
        else:
            # Most engines accept config=None or have default config
            try:
                instance = cls()
            except TypeError:
                try:
                    instance = cls(None)
                except TypeError:
                    instance = cls(config={})

        assert instance is not None, (
            f"Failed to instantiate {class_name} from {engine_key}"
        )


# =============================================================================
# Test Class: Workflow, Template, Integration Importability
# =============================================================================


class TestModuleImports:
    """Tests that workflow, template, and integration files can be imported."""

    @pytest.mark.parametrize("workflow_key", list(WORKFLOW_FILES.keys()))
    def test_workflow_files_importable(self, workflow_key: str):
        """Verify each workflow module can be loaded via importlib."""
        file_name = WORKFLOW_FILES[workflow_key]
        mod = _load_module(workflow_key, file_name, "workflows")
        assert mod is not None, (
            f"Failed to load workflow module: {workflow_key}"
        )

    @pytest.mark.parametrize("template_key", list(TEMPLATE_FILES.keys()))
    def test_template_files_importable(self, template_key: str):
        """Verify each template module can be loaded via importlib."""
        file_name = TEMPLATE_FILES[template_key]
        mod = _load_module(template_key, file_name, "templates")
        assert mod is not None, (
            f"Failed to load template module: {template_key}"
        )

    @pytest.mark.parametrize("integration_key", list(INTEGRATION_FILES.keys()))
    def test_integration_files_importable(self, integration_key: str):
        """Verify each integration module can be loaded via importlib."""
        file_name = INTEGRATION_FILES[integration_key]
        mod = _load_module(integration_key, file_name, "integrations")
        assert mod is not None, (
            f"Failed to load integration module: {integration_key}"
        )


# =============================================================================
# Test Class: Smoke Tests - Engine Calculations
# =============================================================================


class TestSmokeEngines:
    """Smoke tests that verify each engine can execute its primary calculation.

    Each test:
      1. Creates the engine with default config
      2. Calls the main calculation method with minimal valid data
      3. Asserts the result is not None and has a provenance_hash
    """

    def test_smoke_process_emissions(self, sample_facility):
        """Smoke test: ProcessEmissionsEngine.calculate_facility_emissions."""
        mod = _load_engine("process_emissions")
        engine = mod.ProcessEmissionsEngine()
        result = engine.calculate_facility_emissions(sample_facility)

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
        assert result.total_emissions >= 0.0

    def test_smoke_energy_intensity(self, sample_energy_data):
        """Smoke test: EnergyIntensityEngine.calculate_energy_intensity."""
        mod = _load_engine("energy_intensity")
        engine = mod.EnergyIntensityEngine()
        result = engine.calculate_energy_intensity(sample_energy_data)

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
        assert result.total_energy_mwh > 0.0

    def test_smoke_product_pcf(self, sample_product, sample_bom):
        """Smoke test: ProductCarbonFootprintEngine.calculate_product_pcf."""
        mod = _load_engine("product_carbon_footprint")
        engine = mod.ProductCarbonFootprintEngine()
        # Build minimal ManufacturingProcess list required by the method
        manufacturing = [
            mod.ManufacturingProcess(
                process_name="Body Assembly",
                energy_consumption_kwh_per_unit=150.0,
                process_emissions_kgco2e_per_unit=5.0,
                waste_generated_kg_per_unit=2.0,
            ),
            mod.ManufacturingProcess(
                process_name="Paint Shop",
                energy_consumption_kwh_per_unit=80.0,
                process_emissions_kgco2e_per_unit=3.5,
                waste_generated_kg_per_unit=1.5,
            ),
        ]
        result = engine.calculate_product_pcf(
            product=sample_product,
            bom=sample_bom,
            manufacturing=manufacturing,
        )

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_smoke_circular_economy(
        self, sample_material_flows, sample_waste_streams
    ):
        """Smoke test: CircularEconomyEngine.calculate_circular_metrics."""
        mod = _load_engine("circular_economy")
        engine = mod.CircularEconomyEngine()
        result = engine.calculate_circular_metrics(
            materials=sample_material_flows,
            waste_streams=sample_waste_streams,
        )

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_smoke_water_pollution(
        self, sample_water_intake, sample_water_discharge
    ):
        """Smoke test: WaterPollutionEngine.calculate_water_balance."""
        mod = _load_engine("water_pollution")
        config = mod.WaterPollutionConfig(
            reporting_year=2025,
            production_volume=500000,
        )
        engine = mod.WaterPollutionEngine(config)
        result = engine.calculate_water_balance(
            intake=sample_water_intake,
            discharge=sample_water_discharge,
        )

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_smoke_bat_compliance(self, sample_bat_data):
        """Smoke test: BATComplianceEngine.assess_compliance."""
        mod = _load_engine("bat_compliance")
        config = mod.BATConfig(
            reporting_year=2025,
            applicable_brefs=[mod.BREFDocument.CEMENT_LIME],
        )
        engine = mod.BATComplianceEngine(config)
        result = engine.assess_compliance(sample_bat_data)

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_smoke_supply_chain(self, sample_suppliers, sample_bom_emissions):
        """Smoke test: SupplyChainEmissionsEngine.calculate_supply_chain_emissions."""
        mod = _load_engine("supply_chain_emissions")
        config = mod.SupplyChainConfig(
            reporting_year=2025,
        )
        engine = mod.SupplyChainEmissionsEngine(config)
        result = engine.calculate_supply_chain_emissions(
            suppliers=sample_suppliers,
            bom=sample_bom_emissions,
        )

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_smoke_benchmark(self, sample_facility_kpis):
        """Smoke test: ManufacturingBenchmarkEngine.benchmark_facility."""
        mod = _load_engine("manufacturing_benchmark")
        config = mod.BenchmarkConfig(
            reporting_year=2025,
            sub_sector="cement",
        )
        engine = mod.ManufacturingBenchmarkEngine(config)
        result = engine.benchmark_facility(sample_facility_kpis)

        assert result is not None
        assert hasattr(result, "provenance_hash")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64
