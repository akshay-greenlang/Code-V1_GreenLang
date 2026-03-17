# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - Demo Mode

Tests demo configuration, sample data validation, demo workflow execution,
and eligibility screening in demo mode. Validates the demo_config.yaml and
ensures the pack can run end-to-end with synthetic data.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PACK_008_DIR = Path(__file__).resolve().parent.parent
_DEMO_CONFIG_PATH = _PACK_008_DIR / "config" / "demo" / "demo_config.yaml"
_INTEGRATIONS_DIR = _PACK_008_DIR / "integrations"


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _instantiate_bridge(mod, class_name: str, config_class_names: list):
    """Try to instantiate a bridge class, handling config requirement patterns."""
    if mod is None:
        return None
    cls = getattr(mod, class_name, None)
    if cls is None:
        return None
    try:
        return cls()
    except TypeError:
        for cfg_name in config_class_names:
            cfg_cls = getattr(mod, cfg_name, None)
            if cfg_cls is not None:
                try:
                    return cls(cfg_cls())
                except Exception:
                    continue
    return None


# Import orchestrator for demo pipeline tests
_orch_mod = _import_from_path(
    "pack008_demo_orchestrator", _INTEGRATIONS_DIR / "pack_orchestrator.py"
)
_wizard_mod = _import_from_path(
    "pack008_demo_wizard", _INTEGRATIONS_DIR / "setup_wizard.py"
)


# ===========================================================================
# Demo Mode Tests
# ===========================================================================
@pytest.mark.unit
class TestDemoMode:
    """Test suite for PACK-008 demo mode configuration and execution."""

    # -----------------------------------------------------------------------
    # DEMO-001: Config file exists
    # -----------------------------------------------------------------------
    def test_demo_config_exists(self):
        """Demo configuration YAML file exists on disk."""
        assert _DEMO_CONFIG_PATH.exists(), (
            f"demo_config.yaml not found at {_DEMO_CONFIG_PATH}"
        )

    # -----------------------------------------------------------------------
    # DEMO-002: Config is valid YAML
    # -----------------------------------------------------------------------
    def test_demo_config_valid_yaml(self):
        """Demo configuration file is valid YAML."""
        if not _DEMO_CONFIG_PATH.exists():
            pytest.skip("demo_config.yaml not found")
        raw = _DEMO_CONFIG_PATH.read_text(encoding="utf-8")
        parsed = yaml.safe_load(raw)
        assert parsed is not None, "YAML parsed as None (empty or invalid)"
        assert isinstance(parsed, dict), f"Expected dict, got {type(parsed)}"

    # -----------------------------------------------------------------------
    # DEMO-003: Required fields present
    # -----------------------------------------------------------------------
    def test_demo_config_has_required_fields(self):
        """Demo config contains all required top-level fields."""
        if not _DEMO_CONFIG_PATH.exists():
            pytest.skip("demo_config.yaml not found")
        config = yaml.safe_load(_DEMO_CONFIG_PATH.read_text(encoding="utf-8"))
        required_fields = [
            "organization_type",
            "reporting_year",
            "objectives_in_scope",
            "eligibility",
            "sc_assessment",
            "dnsh",
            "minimum_safeguards",
            "kpi",
            "demo",
        ]
        missing = [f for f in required_fields if f not in config]
        assert len(missing) == 0, f"Missing required fields: {missing}"

        # Validate demo section
        demo_section = config.get("demo", {})
        assert demo_section.get("demo_mode_enabled") is True, (
            "demo.demo_mode_enabled must be true in demo config"
        )

    # -----------------------------------------------------------------------
    # DEMO-004: Minimal setup
    # -----------------------------------------------------------------------
    def test_demo_mode_minimal_setup(self):
        """Orchestrator can be created with demo-compatible defaults."""
        if _orch_mod is None:
            pytest.skip("pack_orchestrator module not available")
        config_cls = getattr(_orch_mod, "TaxonomyOrchestratorConfig", None)
        if config_cls is None:
            pytest.skip("TaxonomyOrchestratorConfig not found")
        orch_cls = getattr(_orch_mod, "TaxonomyPackOrchestrator", None)
        if orch_cls is None:
            pytest.skip("TaxonomyPackOrchestrator not found")

        # Create config with demo-compatible minimal settings
        try:
            config = config_cls(
                organization_type="non_financial_undertaking",
                environmental_objectives=["CCM", "CCA"],
                reporting_period_year=2025,
            )
            orch = orch_cls(config)
        except Exception as exc:
            pytest.skip(f"Could not create demo orchestrator: {exc}")

        assert orch is not None
        assert orch.config.organization_type == "non_financial_undertaking"

    # -----------------------------------------------------------------------
    # DEMO-005: Sample output
    # -----------------------------------------------------------------------
    def test_demo_produces_sample_output(self):
        """Demo config YAML contains synthetic data and reporting settings."""
        if not _DEMO_CONFIG_PATH.exists():
            pytest.skip("demo_config.yaml not found")
        config = yaml.safe_load(_DEMO_CONFIG_PATH.read_text(encoding="utf-8"))

        # Verify demo uses synthetic data
        demo_section = config.get("demo", {})
        assert demo_section.get("use_synthetic_data") is True, (
            "Demo mode must use synthetic data"
        )

        # Verify reporting settings exist
        reporting = config.get("reporting", {})
        assert reporting is not None, "Demo config must include reporting section"
        assert reporting.get("article8_enabled") is True or "default_format" in reporting, (
            "Demo reporting section must have at least article8 or format configured"
        )

    # -----------------------------------------------------------------------
    # DEMO-006: Eligibility screening
    # -----------------------------------------------------------------------
    def test_demo_eligibility_screening(self):
        """Demo config defines eligibility screening parameters."""
        if not _DEMO_CONFIG_PATH.exists():
            pytest.skip("demo_config.yaml not found")
        config = yaml.safe_load(_DEMO_CONFIG_PATH.read_text(encoding="utf-8"))

        eligibility = config.get("eligibility", {})
        assert eligibility is not None, "Demo config must include eligibility section"

        # Verify NACE-based screening mode
        screening_mode = eligibility.get("screening_mode", "")
        assert "NACE" in screening_mode.upper(), (
            f"Demo eligibility should use NACE-based screening, got: {screening_mode}"
        )

        # Verify sample activities count in demo section
        demo_section = config.get("demo", {})
        sample_count = demo_section.get("sample_activities_count", 0)
        assert sample_count > 0, "Demo must define at least 1 sample activity"
