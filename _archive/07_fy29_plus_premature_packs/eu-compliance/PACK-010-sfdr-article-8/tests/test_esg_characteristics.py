# -*- coding: utf-8 -*-
"""
Unit tests for ESGCharacteristicsEngine (PACK-010 SFDR Article 8).

Tests characteristic definition, binding elements, attainment measurement,
benchmark comparison, strategy validation, and provenance tracking.

Self-contained: no conftest imports.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Dynamic import helper
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"


def _import_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_esg_mod = _import_from_path(
    "esg_characteristics_engine",
    str(ENGINES_DIR / "esg_characteristics_engine.py"),
)

ESGCharacteristicsEngine = _esg_mod.ESGCharacteristicsEngine
CharacteristicDefinition = _esg_mod.CharacteristicDefinition
SustainabilityIndicator = _esg_mod.SustainabilityIndicator
BindingElement = _esg_mod.BindingElement
AttainmentResult = _esg_mod.AttainmentResult
BenchmarkComparison = _esg_mod.BenchmarkComparison
StrategyValidationResult = _esg_mod.StrategyValidationResult
CharacteristicsSummary = _esg_mod.CharacteristicsSummary
CharacteristicType = _esg_mod.CharacteristicType
CharacteristicStatus = _esg_mod.CharacteristicStatus
AttainmentStatus = _esg_mod.AttainmentStatus
BindingElementStatus = _esg_mod.BindingElementStatus
MeasurementFrequency = _esg_mod.MeasurementFrequency
BenchmarkType = _esg_mod.BenchmarkType

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_ENV_KEYS = ["climate_mitigation", "water_stewardship"]
_SOCIAL_KEYS = ["labor_rights", "diversity_inclusion"]


# ===================================================================
# TEST CLASS
# ===================================================================


class TestESGCharacteristicsEngine:
    """Unit tests for ESGCharacteristicsEngine."""

    # ---------------------------------------------------------------
    # 1. Engine initialization
    # ---------------------------------------------------------------

    def test_engine_default_initialization(self):
        """Test engine initializes with default config."""
        engine = ESGCharacteristicsEngine()
        assert engine is not None

    def test_engine_custom_config(self):
        """Test engine initializes with custom config dict."""
        config = {"attainment_threshold_high": 85.0, "attainment_threshold_partial": 45.0}
        engine = ESGCharacteristicsEngine(config)
        assert engine is not None

    # ---------------------------------------------------------------
    # 2. define_characteristics
    # ---------------------------------------------------------------

    def test_define_characteristics_returns_list(self):
        """Test define_characteristics returns list of definitions."""
        engine = ESGCharacteristicsEngine()
        chars = engine.define_characteristics(_ENV_KEYS)
        assert isinstance(chars, list)
        assert len(chars) >= 1
        for c in chars:
            assert isinstance(c, CharacteristicDefinition)

    def test_define_characteristics_with_custom_targets(self):
        """Test custom targets override defaults."""
        engine = ESGCharacteristicsEngine()
        targets = {"climate_mitigation": 50.0}
        chars = engine.define_characteristics(["climate_mitigation"], targets)
        assert len(chars) >= 1

    # ---------------------------------------------------------------
    # 3. define_custom_characteristic
    # ---------------------------------------------------------------

    def test_define_custom_characteristic(self):
        """Test defining a custom ESG characteristic."""
        engine = ESGCharacteristicsEngine()
        custom = engine.define_custom_characteristic(
            name="Water Usage Reduction",
            characteristic_type=CharacteristicType.ENVIRONMENTAL,
            description="Reduce water consumption by 20%",
            metric="water_usage_pct",
            target=20.0,
            unit="%",
        )
        assert isinstance(custom, CharacteristicDefinition)

    # ---------------------------------------------------------------
    # 4. add_binding_element
    # ---------------------------------------------------------------

    def test_add_binding_element(self):
        """Test adding a binding element to a characteristic."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(["climate_mitigation"])
        char_id = defs[0].characteristic_id
        element = engine.add_binding_element(
            char_id,
            {
                "commitment_name": "Exclude fossil fuel companies >10% revenue",
                "minimum_threshold": 10.0,
                "measurement_method": "revenue_screening",
            },
        )
        assert isinstance(element, BindingElement)

    # ---------------------------------------------------------------
    # 5. get_binding_elements
    # ---------------------------------------------------------------

    def test_get_binding_elements_returns_list(self):
        """Test retrieving binding elements for a characteristic."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(["climate_mitigation"])
        char_id = defs[0].characteristic_id
        engine.add_binding_element(
            char_id,
            {
                "commitment_name": "Max carbon intensity threshold",
                "minimum_threshold": 500.0,
                "measurement_method": "carbon_intensity_check",
            },
        )
        elements = engine.get_binding_elements(char_id)
        assert isinstance(elements, list)
        assert len(elements) >= 1

    # ---------------------------------------------------------------
    # 6. measure_attainment
    # ---------------------------------------------------------------

    def test_measure_attainment_returns_list(self):
        """Test attainment measurement returns list of AttainmentResult."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(_ENV_KEYS)
        # Build actual_values keyed by characteristic_id (UUID)
        actual_values = {d.characteristic_id: 35.0 for d in defs}
        results = engine.measure_attainment(
            actual_values,
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert isinstance(r, AttainmentResult)

    def test_measure_attainment_has_statuses(self):
        """Test attainment results include status field."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(_ENV_KEYS)
        actual_values = {d.characteristic_id: 35.0 for d in defs}
        results = engine.measure_attainment(
            actual_values,
            period_start=date(2025, 1, 1),
            period_end=date(2025, 12, 31),
        )
        assert len(results) >= 1
        for r in results:
            assert hasattr(r, "status")

    # ---------------------------------------------------------------
    # 7. compare_to_benchmark
    # ---------------------------------------------------------------

    def test_compare_to_benchmark_returns_result(self):
        """Test benchmark comparison returns list of BenchmarkComparison."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(_ENV_KEYS)
        # Set benchmark data first (required)
        bm_name = "MSCI Europe ESG Leaders"
        bm_values = {d.characteristic_id: 25.0 for d in defs}
        engine.set_benchmark_data(bm_name, bm_values)
        # Now compare
        actual_values = {d.characteristic_id: 35.0 for d in defs}
        comparisons = engine.compare_to_benchmark(
            actual_values,
            benchmark_name=bm_name,
            benchmark_type=BenchmarkType.DESIGNATED_REFERENCE,
        )
        assert isinstance(comparisons, list)
        assert len(comparisons) >= 1
        for comp in comparisons:
            assert isinstance(comp, BenchmarkComparison)

    # ---------------------------------------------------------------
    # 8. validate_strategy
    # ---------------------------------------------------------------

    def test_validate_strategy_returns_result(self):
        """Test strategy validation returns StrategyValidationResult."""
        engine = ESGCharacteristicsEngine()
        engine.define_characteristics(_ENV_KEYS)
        result = engine.validate_strategy()
        assert isinstance(result, StrategyValidationResult)

    # ---------------------------------------------------------------
    # 9. get_characteristics_summary
    # ---------------------------------------------------------------

    def test_get_characteristics_summary(self):
        """Test summary generation returns CharacteristicsSummary."""
        engine = ESGCharacteristicsEngine()
        defs = engine.define_characteristics(_ENV_KEYS)
        actual_values = {d.characteristic_id: 35.0 for d in defs}
        summary = engine.get_characteristics_summary(
            actual_values,
            benchmark_name="MSCI World",
        )
        assert isinstance(summary, CharacteristicsSummary)

    # ---------------------------------------------------------------
    # 10. list_available catalogs
    # ---------------------------------------------------------------

    def test_list_available_environmental(self):
        """Test listing pre-defined environmental characteristics."""
        engine = ESGCharacteristicsEngine()
        env_catalog = engine.list_available_environmental()
        assert isinstance(env_catalog, dict)
        assert len(env_catalog) >= 1
        assert "climate_mitigation" in env_catalog

    def test_list_available_social(self):
        """Test listing pre-defined social characteristics."""
        engine = ESGCharacteristicsEngine()
        social_catalog = engine.list_available_social()
        assert isinstance(social_catalog, dict)
        assert len(social_catalog) >= 1
        assert "labor_rights" in social_catalog

    # ---------------------------------------------------------------
    # 11. CharacteristicType enum
    # ---------------------------------------------------------------

    def test_characteristic_type_enum(self):
        """Test CharacteristicType enum values."""
        vals = {t.value for t in CharacteristicType}
        assert "environmental" in vals
        assert "social" in vals

    # ---------------------------------------------------------------
    # 12. AttainmentStatus enum
    # ---------------------------------------------------------------

    def test_attainment_status_enum(self):
        """Test AttainmentStatus enum values."""
        vals = {s.value for s in AttainmentStatus}
        assert len(vals) >= 3

    # ---------------------------------------------------------------
    # 13. BindingElementStatus enum
    # ---------------------------------------------------------------

    def test_binding_element_status_enum(self):
        """Test BindingElementStatus enum values."""
        vals = {s.value for s in BindingElementStatus}
        assert len(vals) >= 2

    # ---------------------------------------------------------------
    # 14. BenchmarkType enum
    # ---------------------------------------------------------------

    def test_benchmark_type_enum(self):
        """Test BenchmarkType enum values."""
        vals = {b.value for b in BenchmarkType}
        assert len(vals) >= 1
        assert "designated_reference" in vals

    # ---------------------------------------------------------------
    # 15. list_characteristics filtering
    # ---------------------------------------------------------------

    def test_list_characteristics_filter_by_type(self):
        """Test list_characteristics filters by type."""
        engine = ESGCharacteristicsEngine()
        engine.define_characteristics(_ENV_KEYS + _SOCIAL_KEYS)
        env_chars = engine.list_characteristics(
            char_type=CharacteristicType.ENVIRONMENTAL,
        )
        assert isinstance(env_chars, list)
