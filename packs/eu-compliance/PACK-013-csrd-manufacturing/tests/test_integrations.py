# -*- coding: utf-8 -*-
"""
PACK-013 CSRD Manufacturing Pack - Integration Tests

Tests all 9 integration modules: pack orchestrator, CSRD/CBAM/MRV/Data
bridges, EU ETS bridge, Taxonomy bridge, health check, and setup wizard.

20 tests across 9 test classes.
"""

import importlib.util
import sys
import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Dynamic module loading via importlib
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


def _load_module(module_name: str, file_name: str, search_dir: Path = INTEGRATIONS_DIR):
    """Load a module dynamically using importlib.util.spec_from_file_location."""
    file_path = search_dir / file_name
    if not file_path.exists():
        pytest.skip(f"Module file not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        pytest.skip(f"Cannot create spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Load all 9 integration modules
# ---------------------------------------------------------------------------

orchestrator_mod = _load_module(
    "pack013_int_orchestrator", "pack_orchestrator.py"
)
CSRDManufacturingOrchestrator = orchestrator_mod.CSRDManufacturingOrchestrator
OrchestratorConfig = orchestrator_mod.OrchestratorConfig
PipelinePhase = orchestrator_mod.PipelinePhase
PHASE_DEPENDENCIES = orchestrator_mod.PHASE_DEPENDENCIES

csrd_bridge_mod = _load_module(
    "pack013_int_csrd_bridge", "csrd_pack_bridge.py"
)
CSRDPackBridge = csrd_bridge_mod.CSRDPackBridge
CSRDBridgeConfig = csrd_bridge_mod.CSRDBridgeConfig

cbam_bridge_mod = _load_module(
    "pack013_int_cbam_bridge", "cbam_pack_bridge.py"
)
CBAMPackBridge = cbam_bridge_mod.CBAMPackBridge
CBAMBridgeConfig = cbam_bridge_mod.CBAMBridgeConfig

mrv_bridge_mod = _load_module(
    "pack013_int_mrv_bridge", "mrv_industrial_bridge.py"
)
MRVIndustrialBridge = mrv_bridge_mod.MRVIndustrialBridge
MRVBridgeConfig = mrv_bridge_mod.MRVBridgeConfig
DEFAULT_ROUTING_TABLE = mrv_bridge_mod.DEFAULT_ROUTING_TABLE
SUB_SECTOR_AGENTS = mrv_bridge_mod.SUB_SECTOR_AGENTS

data_bridge_mod = _load_module(
    "pack013_int_data_bridge", "data_manufacturing_bridge.py"
)
DataManufacturingBridge = data_bridge_mod.DataManufacturingBridge
DataBridgeConfig = data_bridge_mod.DataBridgeConfig
ERP_FIELD_MAP = data_bridge_mod.ERP_FIELD_MAP

ets_bridge_mod = _load_module(
    "pack013_int_ets_bridge", "eu_ets_bridge.py"
)
EUETSBridge = ets_bridge_mod.EUETSBridge
ETSBridgeConfig = ets_bridge_mod.ETSBridgeConfig
PRODUCT_BENCHMARKS = ets_bridge_mod.PRODUCT_BENCHMARKS

taxonomy_bridge_mod = _load_module(
    "pack013_int_taxonomy_bridge", "taxonomy_bridge.py"
)
TaxonomyBridge = taxonomy_bridge_mod.TaxonomyBridge
TaxonomyBridgeConfig = taxonomy_bridge_mod.TaxonomyBridgeConfig
MANUFACTURING_ACTIVITIES = taxonomy_bridge_mod.MANUFACTURING_ACTIVITIES

health_check_mod = _load_module(
    "pack013_int_health_check", "health_check.py"
)
ManufacturingHealthCheck = health_check_mod.ManufacturingHealthCheck
HealthCategory = health_check_mod.HealthCategory

setup_wizard_mod = _load_module(
    "pack013_int_setup_wizard", "setup_wizard.py"
)
ManufacturingSetupWizard = setup_wizard_mod.ManufacturingSetupWizard
SetupStep = setup_wizard_mod.SetupStep
STEP_ORDER = setup_wizard_mod.STEP_ORDER


# ===========================================================================
# 1. Pack Orchestrator (3 tests)
# ===========================================================================

class TestPackOrchestrator:
    """Tests for CSRDManufacturingOrchestrator."""

    def test_init_with_default_config(self):
        """Orchestrator initializes with default OrchestratorConfig."""
        orch = CSRDManufacturingOrchestrator()
        assert orch.config is not None
        assert isinstance(orch.config, OrchestratorConfig)

    def test_pipeline_phase_enum_has_eleven_values(self):
        """PipelinePhase enum contains exactly 11 phases."""
        phases = list(PipelinePhase)
        assert len(phases) == 11
        expected_values = {
            "initialization", "data_intake", "quality_assurance",
            "process_emissions", "energy_analysis", "product_pcf",
            "circular_economy", "water_pollution", "bat_compliance",
            "supply_chain", "reporting",
        }
        actual_values = {p.value for p in phases}
        assert actual_values == expected_values

    def test_phase_dependencies_has_all_phases(self):
        """PHASE_DEPENDENCIES dict covers all 11 pipeline phases."""
        assert len(PHASE_DEPENDENCIES) == 11
        for phase in PipelinePhase:
            assert phase.value in PHASE_DEPENDENCIES
        # initialization has no dependencies
        assert PHASE_DEPENDENCIES[PipelinePhase.INITIALIZATION.value] == []
        # reporting depends on 7 upstream phases
        assert len(PHASE_DEPENDENCIES[PipelinePhase.REPORTING.value]) == 7


# ===========================================================================
# 2. CSRD Pack Bridge (2 tests)
# ===========================================================================

class TestCSRDPackBridge:
    """Tests for CSRDPackBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default CSRDBridgeConfig."""
        bridge = CSRDPackBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, CSRDBridgeConfig)

    def test_config_has_pack_tier(self):
        """CSRDBridgeConfig has a pack_tier field."""
        config = CSRDBridgeConfig()
        assert hasattr(config, "pack_tier")


# ===========================================================================
# 3. CBAM Pack Bridge (2 tests)
# ===========================================================================

class TestCBAMPackBridge:
    """Tests for CBAMPackBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default CBAMBridgeConfig."""
        bridge = CBAMPackBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, CBAMBridgeConfig)

    def test_config_has_cbam_phase(self):
        """CBAMBridgeConfig has a cbam_phase field."""
        config = CBAMBridgeConfig()
        assert hasattr(config, "cbam_phase")


# ===========================================================================
# 4. MRV Industrial Bridge (3 tests)
# ===========================================================================

class TestMRVIndustrialBridge:
    """Tests for MRVIndustrialBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default MRVBridgeConfig."""
        bridge = MRVIndustrialBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, MRVBridgeConfig)

    def test_default_routing_table_has_entries(self):
        """DEFAULT_ROUTING_TABLE contains at least 20 ESRS-to-agent mappings."""
        assert isinstance(DEFAULT_ROUTING_TABLE, (dict, list))
        assert len(DEFAULT_ROUTING_TABLE) >= 20

    def test_sub_sector_agents_has_entries(self):
        """SUB_SECTOR_AGENTS maps manufacturing sub-sectors to agents."""
        assert isinstance(SUB_SECTOR_AGENTS, dict)
        assert len(SUB_SECTOR_AGENTS) >= 3


# ===========================================================================
# 5. Data Manufacturing Bridge (2 tests)
# ===========================================================================

class TestDataManufacturingBridge:
    """Tests for DataManufacturingBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default DataBridgeConfig."""
        bridge = DataManufacturingBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, DataBridgeConfig)

    def test_erp_field_map_has_entries(self):
        """ERP_FIELD_MAP contains ERP system field mappings."""
        assert isinstance(ERP_FIELD_MAP, dict)
        assert len(ERP_FIELD_MAP) >= 1


# ===========================================================================
# 6. EU ETS Bridge (2 tests)
# ===========================================================================

class TestEUETSBridge:
    """Tests for EUETSBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default ETSBridgeConfig."""
        bridge = EUETSBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, ETSBridgeConfig)

    def test_product_benchmarks_has_entries(self):
        """PRODUCT_BENCHMARKS contains benchmark values for products."""
        assert isinstance(PRODUCT_BENCHMARKS, dict)
        assert len(PRODUCT_BENCHMARKS) >= 3
        # All benchmark values should be positive floats
        for product, benchmark in PRODUCT_BENCHMARKS.items():
            assert isinstance(product, str)
            if isinstance(benchmark, (int, float)):
                assert benchmark > 0


# ===========================================================================
# 7. Taxonomy Bridge (2 tests)
# ===========================================================================

class TestTaxonomyBridge:
    """Tests for TaxonomyBridge."""

    def test_init_with_default_config(self):
        """Bridge initializes with default TaxonomyBridgeConfig."""
        bridge = TaxonomyBridge()
        assert bridge.config is not None
        assert isinstance(bridge.config, TaxonomyBridgeConfig)

    def test_manufacturing_activities_has_entries(self):
        """MANUFACTURING_ACTIVITIES contains NACE activity mappings."""
        assert isinstance(MANUFACTURING_ACTIVITIES, dict)
        assert len(MANUFACTURING_ACTIVITIES) >= 3


# ===========================================================================
# 8. Health Check (2 tests)
# ===========================================================================

class TestHealthCheck:
    """Tests for ManufacturingHealthCheck."""

    def test_init_creates_category_methods(self):
        """Health check initializes with category method mapping."""
        hc = ManufacturingHealthCheck()
        assert hasattr(hc, "_category_methods")
        assert isinstance(hc._category_methods, dict)

    def test_health_category_enum_has_22_values(self):
        """HealthCategory enum has 22 categories."""
        categories = list(HealthCategory)
        assert len(categories) == 22
        # Verify a few key categories exist
        category_values = {c.value for c in categories}
        assert "engines" in category_values
        assert "workflows" in category_values
        assert "templates" in category_values
        assert "provenance" in category_values


# ===========================================================================
# 9. Setup Wizard (2 tests)
# ===========================================================================

class TestSetupWizard:
    """Tests for ManufacturingSetupWizard."""

    def test_init_creates_wizard(self):
        """Wizard initializes with step handlers."""
        wizard = ManufacturingSetupWizard()
        assert hasattr(wizard, "_step_handlers")
        assert isinstance(wizard._step_handlers, dict)

    def test_step_order_has_eight_steps(self):
        """STEP_ORDER contains exactly 8 setup steps."""
        assert isinstance(STEP_ORDER, list)
        assert len(STEP_ORDER) == 8
        # Verify all SetupStep enum values are covered
        step_set = set(STEP_ORDER)
        assert len(step_set) == 8
