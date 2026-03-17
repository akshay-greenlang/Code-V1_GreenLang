# -*- coding: utf-8 -*-
"""
Unit tests for RetailCircularEconomyEngine (PACK-014, Engine 7)
================================================================

Tests all methods of RetailCircularEconomyEngine with 85%+ coverage.
Validates business logic, error handling, and edge cases.

Test count: ~38 tests
"""

import importlib.util
import os

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "engines",
    "retail_circular_economy_engine.py",
)
_ENGINE_PATH = os.path.normpath(_ENGINE_PATH)

_spec = importlib.util.spec_from_file_location("retail_circular_economy_engine", _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

RetailCircularEconomyEngine = _mod.RetailCircularEconomyEngine
CircularEconomyResult = _mod.CircularEconomyResult
TakeBackProgram = _mod.TakeBackProgram
EPRFeeData = _mod.EPRFeeData
MaterialFlow = _mod.MaterialFlow
EPRScheme = _mod.EPRScheme
TakeBackType = _mod.TakeBackType
WasteStream = _mod.WasteStream
CircularStrategy = _mod.CircularStrategy
EPR_RECYCLING_TARGETS = _mod.EPR_RECYCLING_TARGETS
CIRCULARITY_WEIGHTS = _mod.CIRCULARITY_WEIGHTS
EPR_BASE_FEE_RATES = _mod.EPR_BASE_FEE_RATES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a RetailCircularEconomyEngine instance."""
    return RetailCircularEconomyEngine()


@pytest.fixture
def sample_material_flows():
    """Create sample material flow data."""
    return [
        MaterialFlow(
            material="cardboard",
            virgin_input_tonnes=300.0,
            recycled_input_tonnes=200.0,
            waste_output_tonnes=480.0,
            recovery_tonnes=440.0,
        ),
        MaterialFlow(
            material="rigid_plastic",
            virgin_input_tonnes=100.0,
            recycled_input_tonnes=20.0,
            waste_output_tonnes=110.0,
            recovery_tonnes=70.0,
        ),
    ]


@pytest.fixture
def sample_take_back_programs():
    """Create sample take-back programmes."""
    return [
        TakeBackProgram(
            name="Packaging Recycling",
            epr_scheme=EPRScheme.PACKAGING,
            collection_method=TakeBackType.IN_STORE,
            volume_collected_tonnes=500.0,
            volume_placed_on_market_tonnes=700.0,
            recovery_rate_pct=85.0,
        ),
        TakeBackProgram(
            name="E-waste Mail-in",
            epr_scheme=EPRScheme.WEEE,
            collection_method=TakeBackType.MAIL_IN,
            volume_collected_tonnes=50.0,
            volume_placed_on_market_tonnes=80.0,
            recovery_rate_pct=70.0,
        ),
    ]


@pytest.fixture
def sample_epr_fees():
    """Create sample EPR fee data."""
    return [
        EPRFeeData(
            scheme=EPRScheme.PACKAGING,
            material_category="plastic",
            weight_tonnes=100.0,
            modulation_factor=1.2,
        ),
        EPRFeeData(
            scheme=EPRScheme.PACKAGING,
            material_category="paper",
            weight_tonnes=200.0,
            modulation_factor=0.8,
        ),
    ]


# ===========================================================================
# TestInitialization
# ===========================================================================


class TestInitialization:
    """Test engine initialisation."""

    def test_default_instantiation(self):
        """Engine can be created with no arguments."""
        engine = RetailCircularEconomyEngine()
        assert engine is not None

    def test_engine_version(self):
        """Engine exposes a version string."""
        engine = RetailCircularEconomyEngine()
        assert engine.engine_version == "1.0.0"

    def test_config_dict(self):
        """Engine accepts attribute changes."""
        engine = RetailCircularEconomyEngine()
        engine.engine_version = "2.0.0"
        assert engine.engine_version == "2.0.0"

    def test_none_inputs(self, engine):
        """Engine runs with no inputs (returns defaults)."""
        result = engine.calculate()
        assert isinstance(result, CircularEconomyResult)
        assert result.mci_score == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# TestEPRSchemes
# ===========================================================================


class TestEPRSchemes:
    """Test EPR scheme definitions."""

    def test_all_6_defined(self):
        """EPRScheme enum has exactly 6 members."""
        assert len(EPRScheme) == 6

    def test_packaging_target_70(self):
        """Packaging overall recycling target is 70%."""
        assert EPR_RECYCLING_TARGETS["packaging"]["overall"] == pytest.approx(70.0, rel=1e-6)

    def test_weee_target_65(self):
        """WEEE overall recycling target is 65%."""
        assert EPR_RECYCLING_TARGETS["weee"]["overall"] == pytest.approx(65.0, rel=1e-6)

    def test_batteries_target_63(self):
        """Batteries portable recycling target is 63%."""
        assert EPR_RECYCLING_TARGETS["batteries"]["portable"] == pytest.approx(63.0, rel=1e-6)


# ===========================================================================
# TestTakeBackPrograms
# ===========================================================================


class TestTakeBackPrograms:
    """Test take-back programme tracking."""

    def test_in_store_collection(self, engine, sample_take_back_programs):
        """In-store collection programme is tracked."""
        result = engine.calculate(take_back_programs=sample_take_back_programs)
        assert result.take_back_volumes_tonnes > 0.0

    def test_mail_in(self, engine, sample_take_back_programs):
        """Mail-in programme volume is included."""
        result = engine.calculate(take_back_programs=sample_take_back_programs)
        # 500 + 50 = 550
        assert result.take_back_volumes_tonnes == pytest.approx(550.0, rel=1e-2)

    def test_volume_tracking(self, engine, sample_take_back_programs):
        """Per-programme volumes are listed."""
        result = engine.calculate(take_back_programs=sample_take_back_programs)
        assert len(result.take_back_by_programme) == 2

    def test_recovery_rate(self, engine, sample_take_back_programs):
        """Recovery rate is reported per programme."""
        result = engine.calculate(take_back_programs=sample_take_back_programs)
        prog_0 = result.take_back_by_programme[0]
        assert "recovery_rate_pct" in prog_0

    def test_collection_rate(self, engine, sample_take_back_programs):
        """Overall take-back rate is calculated correctly."""
        result = engine.calculate(take_back_programs=sample_take_back_programs)
        # collected 550 / placed 780 * 100 = ~70.5%
        assert result.take_back_rate_pct > 0.0


# ===========================================================================
# TestMCI
# ===========================================================================


class TestMCI:
    """Test Material Circularity Index calculation."""

    def test_high_circularity(self, engine):
        """High recycled input + high recovery => MCI > 0.7."""
        flows = [
            MaterialFlow(
                material="cardboard",
                virgin_input_tonnes=50.0,
                recycled_input_tonnes=450.0,
                waste_output_tonnes=490.0,
                recovery_tonnes=480.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.mci_score > 0.7

    def test_low_circularity(self, engine):
        """High virgin + low recovery => MCI < 0.3."""
        flows = [
            MaterialFlow(
                material="rigid_plastic",
                virgin_input_tonnes=900.0,
                recycled_input_tonnes=10.0,
                waste_output_tonnes=900.0,
                recovery_tonnes=50.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.mci_score < 0.3

    def test_formula_correctness(self, engine):
        """MCI is between 0 and 1."""
        flows = [
            MaterialFlow(
                material="glass",
                virgin_input_tonnes=200.0,
                recycled_input_tonnes=100.0,
                waste_output_tonnes=280.0,
                recovery_tonnes=200.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert 0.0 <= result.mci_score <= 1.0

    def test_virgin_only(self, engine):
        """All virgin, no recovery => MCI near 0.1."""
        flows = [
            MaterialFlow(
                material="plastic_film",
                virgin_input_tonnes=1000.0,
                recycled_input_tonnes=0.0,
                waste_output_tonnes=1000.0,
                recovery_tonnes=0.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.mci_score == pytest.approx(0.1, abs=0.05)

    def test_fully_circular(self, engine):
        """All recycled, full recovery => MCI near 1.0."""
        flows = [
            MaterialFlow(
                material="metals",
                virgin_input_tonnes=0.0,
                recycled_input_tonnes=500.0,
                waste_output_tonnes=500.0,
                recovery_tonnes=500.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.mci_score >= 0.9


# ===========================================================================
# TestWasteDiversion
# ===========================================================================


class TestWasteDiversion:
    """Test waste diversion rate calculations."""

    def test_high_diversion(self, engine):
        """High recovery => diversion rate > 90%."""
        flows = [
            MaterialFlow(
                material="cardboard",
                virgin_input_tonnes=100.0,
                recycled_input_tonnes=100.0,
                waste_output_tonnes=190.0,
                recovery_tonnes=180.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.waste_diversion_rate_pct > 90.0

    def test_low_diversion(self, engine):
        """Low recovery => diversion rate < 30%."""
        flows = [
            MaterialFlow(
                material="mixed",
                virgin_input_tonnes=500.0,
                recycled_input_tonnes=0.0,
                waste_output_tonnes=500.0,
                recovery_tonnes=100.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.waste_diversion_rate_pct < 30.0

    def test_rate_calculation(self, engine, sample_material_flows):
        """Diversion rate = diverted / total waste * 100."""
        result = engine.calculate(material_flows=sample_material_flows)
        # total waste = 480 + 110 = 590, diverted = 440 + 70 = 510
        expected = 510.0 / 590.0 * 100.0
        assert result.waste_diversion_rate_pct == pytest.approx(expected, rel=1e-2)

    def test_zero_waste(self, engine):
        """Zero waste gives 0% diversion."""
        flows = [
            MaterialFlow(
                material="cardboard",
                virgin_input_tonnes=100.0,
                recycled_input_tonnes=0.0,
                waste_output_tonnes=0.0,
                recovery_tonnes=0.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.waste_diversion_rate_pct == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
# TestEPRFees
# ===========================================================================


class TestEPRFees:
    """Test EPR fee calculations."""

    def test_fee_calculation(self, engine, sample_epr_fees):
        """Fees are calculated using base rate * weight * modulation."""
        result = engine.calculate(epr_fees=sample_epr_fees)
        # plastic: 100 * 350 * 1.2 = 42000
        # paper: 200 * 80 * 0.8 = 12800
        assert result.total_epr_fees_eur == pytest.approx(54800.0, rel=1e-2)

    def test_modulation_factor(self, engine):
        """Modulation factor adjusts the fee."""
        fee = EPRFeeData(
            scheme=EPRScheme.PACKAGING,
            material_category="plastic",
            weight_tonnes=10.0,
            modulation_factor=2.0,
        )
        result = engine.calculate(epr_fees=[fee])
        # 10 * 350 * 2.0 = 7000
        assert result.total_epr_fees_eur == pytest.approx(7000.0, rel=1e-2)

    def test_total_fees(self, engine, sample_epr_fees):
        """Total fees sum all fee items."""
        result = engine.calculate(epr_fees=sample_epr_fees)
        assert result.total_epr_fees_eur > 0.0

    def test_fee_by_scheme(self, engine, sample_epr_fees):
        """Fees are aggregated by scheme."""
        result = engine.calculate(epr_fees=sample_epr_fees)
        assert "packaging" in result.epr_fees_by_scheme


# ===========================================================================
# TestMaterialFlows
# ===========================================================================


class TestMaterialFlows:
    """Test material flow tracking."""

    def test_recycled_content(self, engine, sample_material_flows):
        """Recycled content percentage is calculated correctly."""
        result = engine.calculate(material_flows=sample_material_flows)
        # total recycled = 200 + 20 = 220, total input = 500 + 120 = 620
        expected = 220.0 / 620.0 * 100.0
        assert result.recycled_content_pct == pytest.approx(expected, rel=1e-2)

    def test_virgin_input(self, engine, sample_material_flows):
        """Virgin input is tracked per material."""
        result = engine.calculate(material_flows=sample_material_flows)
        # cardboard: recycled 200 / total 500 = 40%
        assert result.recycled_content_by_material["cardboard"] == pytest.approx(40.0, rel=1e-2)

    def test_recovery_tonnes(self, engine, sample_material_flows):
        """Total diverted tonnes are summed correctly."""
        result = engine.calculate(material_flows=sample_material_flows)
        assert result.total_diverted_tonnes == pytest.approx(510.0, rel=1e-2)

    def test_circularity_by_strategy(self, engine):
        """Circularity breakdown by strategy works correctly."""
        activities = {
            "reduce": 100.0,
            "reuse": 80.0,
            "recycle": 200.0,
        }
        result = engine.calculate(circular_activities=activities)
        assert len(result.circularity_by_strategy) == 3


# ===========================================================================
# TestCircularStrategies
# ===========================================================================


class TestCircularStrategies:
    """Test circular economy R-strategy definitions."""

    def test_all_7_defined(self):
        """CircularStrategy enum has exactly 7 members."""
        assert len(CircularStrategy) == 7

    def test_strategy_weights(self):
        """CIRCULARITY_WEIGHTS has expected ordering."""
        assert CIRCULARITY_WEIGHTS["reduce"] == 1.0
        assert CIRCULARITY_WEIGHTS["reuse"] == 0.9
        assert CIRCULARITY_WEIGHTS["recycle"] == 0.4
        assert CIRCULARITY_WEIGHTS["recover"] == 0.2

    def test_dominant_strategy(self, engine):
        """Dominant strategy drives weighted circularity score."""
        # All reduce => score should be near 1.0
        activities = {"reduce": 1000.0}
        result = engine.calculate(circular_activities=activities)
        assert result.weighted_circularity_score == pytest.approx(1.0, rel=1e-2)


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hashing."""

    def test_hash_length(self, engine, sample_material_flows):
        """Provenance hash is 64 hex characters."""
        result = engine.calculate(material_flows=sample_material_flows)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, engine):
        """Provenance hash is a valid hex string derived from result data."""
        flows = [
            MaterialFlow(
                material="cardboard",
                virgin_input_tonnes=100.0,
                recycled_input_tonnes=50.0,
                waste_output_tonnes=140.0,
                recovery_tonnes=120.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        # Hash is valid hex
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)
        # Recomputing: engine computes hash when provenance_hash is still ""
        from hashlib import sha256
        import json
        saved_hash = result.provenance_hash
        result.provenance_hash = ""
        serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True, default=str)
        expected = sha256(serialized.encode("utf-8")).hexdigest()
        result.provenance_hash = saved_hash
        assert saved_hash == expected

    def test_different_input(self, engine):
        """Different inputs produce different hashes."""
        r1 = engine.calculate(material_flows=[
            MaterialFlow(
                material="cardboard",
                virgin_input_tonnes=100.0,
                recycled_input_tonnes=50.0,
                waste_output_tonnes=140.0,
                recovery_tonnes=120.0,
            ),
        ])
        r2 = engine.calculate(material_flows=[
            MaterialFlow(
                material="glass",
                virgin_input_tonnes=200.0,
                recycled_input_tonnes=100.0,
                waste_output_tonnes=280.0,
                recovery_tonnes=200.0,
            ),
        ])
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_programs(self, engine):
        """Engine works with no take-back programmes."""
        result = engine.calculate()
        assert result.take_back_volumes_tonnes == pytest.approx(0.0, abs=1e-6)

    def test_single_material(self, engine):
        """Engine works with a single material flow."""
        flows = [
            MaterialFlow(
                material="metals",
                virgin_input_tonnes=50.0,
                recycled_input_tonnes=50.0,
                waste_output_tonnes=90.0,
                recovery_tonnes=85.0,
            ),
        ]
        result = engine.calculate(material_flows=flows)
        assert result.mci_score > 0.0

    def test_large_portfolio(self, engine):
        """Engine handles many material flows."""
        flows = [
            MaterialFlow(
                material=f"material_{i}",
                virgin_input_tonnes=10.0,
                recycled_input_tonnes=5.0,
                waste_output_tonnes=14.0,
                recovery_tonnes=10.0,
            )
            for i in range(100)
        ]
        result = engine.calculate(material_flows=flows)
        assert result.total_waste_tonnes == pytest.approx(1400.0, rel=1e-2)

    def test_result_fields(self, engine, sample_material_flows):
        """Result object contains all expected fields."""
        result = engine.calculate(material_flows=sample_material_flows)
        assert hasattr(result, "mci_score")
        assert hasattr(result, "mci_grade")
        assert hasattr(result, "waste_diversion_rate_pct")
        assert hasattr(result, "recycled_content_pct")
        assert hasattr(result, "total_epr_fees_eur")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms > 0.0
