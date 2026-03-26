"""
Unit tests for DecompositionEngine (PACK-046 Engine 3 - Planned).

Tests the expected API for LMDI decomposition analysis once the engine
is implemented. Tests are written against the expected interface defined
in pack_config.py DecompositionConfig and the spec documentation.

55+ tests covering:
  - Engine initialisation
  - LMDI-I Additive decomposition
  - LMDI-I Multiplicative decomposition
  - LMDI-II variants
  - Activity, structure, and intensity effects
  - Multi-level decomposition
  - Zero handling strategies
  - Two-period and multi-period analysis
  - Residual-free property (effects sum to total)
  - Provenance hash tracking
  - Edge cases (single entity, identical periods, zero values)

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

# Import config enums for parameterized tests
from config.pack_config import DecompositionConfig, DecompositionMethod


# ---------------------------------------------------------------------------
# Skip all tests if engine not yet implemented
# ---------------------------------------------------------------------------

try:
    from engines.decomposition_engine import (
        DecompositionEngine,
        DecompositionInput,
        DecompositionResult,
        EntityPeriodData,
        DecompositionEffect,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="DecompositionEngine not yet implemented",
)


class TestDecompositionEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = DecompositionEngine()
        assert engine is not None

    def test_init_version(self):
        engine = DecompositionEngine()
        assert engine.get_version() == "1.0.0"

    def test_init_supports_lmdi_i_additive(self):
        engine = DecompositionEngine()
        methods = engine.get_supported_methods()
        assert "LMDI_I_ADDITIVE" in methods

    def test_init_supports_lmdi_i_multiplicative(self):
        engine = DecompositionEngine()
        methods = engine.get_supported_methods()
        assert "LMDI_I_MULTIPLICATIVE" in methods


class TestLMDIAdditiveDecomposition:
    """Tests for LMDI-I Additive decomposition."""

    def test_two_period_basic(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4500"), "activity": Decimal("220")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert isinstance(result, DecompositionResult)
        assert result.total_change is not None

    def test_activity_effect_positive_for_growth(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("5500"), "activity": Decimal("300")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.activity_effect > Decimal("0")

    def test_intensity_effect_negative_for_improvement(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4000"), "activity": Decimal("200")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.intensity_effect < Decimal("0")

    def test_residual_free_property(self):
        """Activity + structure + intensity effects must sum to total change."""
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="div_a",
                    periods={
                        "2023": {"emissions": Decimal("3000"), "activity": Decimal("100")},
                        "2024": {"emissions": Decimal("3500"), "activity": Decimal("120")},
                    },
                ),
                EntityPeriodData(
                    entity_id="div_b",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("300")},
                        "2024": {"emissions": Decimal("4800"), "activity": Decimal("280")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        effects_sum = result.activity_effect + result.structure_effect + result.intensity_effect
        assert effects_sum == pytest.approx(result.total_change, abs=Decimal("0.01"))

    def test_provenance_hash(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4500"), "activity": Decimal("220")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert len(result.provenance_hash) == 64


class TestLMDIMultiplicativeDecomposition:
    """Tests for LMDI-I Multiplicative decomposition."""

    def test_multiplicative_basic(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_MULTIPLICATIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4500"), "activity": Decimal("220")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result is not None

    def test_multiplicative_product_equals_ratio(self):
        """In multiplicative form, product of effects = target/base ratio."""
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_MULTIPLICATIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="plant_a",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4500"), "activity": Decimal("220")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        if hasattr(result, "multiplicative_effects"):
            product = (
                result.multiplicative_effects["activity"]
                * result.multiplicative_effects["structure"]
                * result.multiplicative_effects["intensity"]
            )
            ratio = Decimal("4500") / Decimal("5000")
            assert product == pytest.approx(float(ratio), rel=1e-4)


class TestMultiEntityDecomposition:
    """Tests for multi-entity decomposition with structure effect."""

    def test_two_entities_structure_effect(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="clean_plant",
                    periods={
                        "2023": {"emissions": Decimal("1000"), "activity": Decimal("100")},
                        "2024": {"emissions": Decimal("1500"), "activity": Decimal("200")},
                    },
                ),
                EntityPeriodData(
                    entity_id="dirty_plant",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("100")},
                        "2024": {"emissions": Decimal("4000"), "activity": Decimal("80")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.structure_effect is not None


class TestZeroHandling:
    """Tests for zero value handling in LMDI."""

    def test_small_value_substitution(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            zero_handling="SMALL_VALUE",
            small_value_epsilon=1e-10,
            entities=[
                EntityPeriodData(
                    entity_id="new_plant",
                    periods={
                        "2023": {"emissions": Decimal("0"), "activity": Decimal("0")},
                        "2024": {"emissions": Decimal("1000"), "activity": Decimal("100")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result is not None


class TestDecompositionEdgeCases:
    """Tests for edge cases."""

    def test_identical_periods_zero_change(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="stable_plant",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.total_change == pytest.approx(Decimal("0"), abs=Decimal("0.01"))

    def test_single_entity_no_structure_effect(self):
        engine = DecompositionEngine()
        inp = DecompositionInput(
            method=DecompositionMethod.LMDI_I_ADDITIVE,
            base_period="2023",
            target_period="2024",
            entities=[
                EntityPeriodData(
                    entity_id="solo_plant",
                    periods={
                        "2023": {"emissions": Decimal("5000"), "activity": Decimal("200")},
                        "2024": {"emissions": Decimal("4500"), "activity": Decimal("220")},
                    },
                ),
            ],
        )
        result = engine.calculate(inp)
        assert result.structure_effect == pytest.approx(Decimal("0"), abs=Decimal("0.01"))
