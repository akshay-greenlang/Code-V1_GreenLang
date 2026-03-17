# -*- coding: utf-8 -*-
"""
Unit tests for CircularEconomyEngine - PACK-013 CSRD Manufacturing Engine 4

Tests all methods of CircularEconomyEngine with 85%+ coverage.
Validates MCI calculation, waste hierarchy compliance, EPR scheme assessment,
recycled content tracking, waste intensity, product recyclability scoring,
and provenance hashing.

Target: 35+ tests across 8 test classes.
"""

import importlib.util
import os
import sys
import pytest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------

_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engines"
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


ce = _load_module("circular_economy_engine", "circular_economy_engine.py")

CircularEconomyEngine = ce.CircularEconomyEngine
CircularEconomyConfig = ce.CircularEconomyConfig
CircularEconomyResult = ce.CircularEconomyResult
MaterialFlowData = ce.MaterialFlowData
WasteStreamData = ce.WasteStreamData
ProductRecyclability = ce.ProductRecyclability
MCIResult = ce.MCIResult
WasteType = ce.WasteType
WasteCategory = ce.WasteCategory
WasteDestination = ce.WasteDestination
EPRScheme = ce.EPRScheme
WASTE_HIERARCHY_WEIGHTS = ce.WASTE_HIERARCHY_WEIGHTS
EPR_RECYCLING_TARGETS = ce.EPR_RECYCLING_TARGETS
CRM_RECYCLING_TARGETS = ce.CRM_RECYCLING_TARGETS
_round3 = ce._round3
_round2 = ce._round2
_compute_hash = ce._compute_hash
_safe_divide = ce._safe_divide
_safe_pct = ce._safe_pct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_engine():
    """Create a CircularEconomyEngine with default configuration."""
    return CircularEconomyEngine()


@pytest.fixture
def sample_materials():
    """Create sample material flow data (virgin + recycled steel, plastic)."""
    return [
        MaterialFlowData(
            material_name="Steel",
            virgin_input_tonnes=800.0,
            recycled_input_tonnes=200.0,
            pre_consumer_recycled_pct=10.0,
            post_consumer_recycled_pct=10.0,
        ),
        MaterialFlowData(
            material_name="Plastic",
            virgin_input_tonnes=150.0,
            recycled_input_tonnes=50.0,
            pre_consumer_recycled_pct=5.0,
            post_consumer_recycled_pct=20.0,
        ),
    ]


@pytest.fixture
def sample_waste_streams():
    """Create sample waste streams with recycling, landfill, composting, incineration."""
    return [
        WasteStreamData(
            waste_type=WasteType.METAL_SCRAP,
            waste_category=WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=120.0,
            destination=WasteDestination.RECYCLING,
            recycling_rate_pct=95.0,
            treatment_cost_eur=5000.0,
        ),
        WasteStreamData(
            waste_type=WasteType.PLASTIC,
            waste_category=WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=80.0,
            destination=WasteDestination.LANDFILL,
            recycling_rate_pct=0.0,
            treatment_cost_eur=8000.0,
        ),
        WasteStreamData(
            waste_type=WasteType.ORGANIC,
            waste_category=WasteCategory.NON_HAZARDOUS,
            quantity_tonnes=50.0,
            destination=WasteDestination.COMPOSTING,
            recycling_rate_pct=0.0,
            treatment_cost_eur=2000.0,
        ),
        WasteStreamData(
            waste_type=WasteType.CHEMICAL,
            waste_category=WasteCategory.HAZARDOUS,
            quantity_tonnes=30.0,
            destination=WasteDestination.INCINERATION,
            recycling_rate_pct=0.0,
            treatment_cost_eur=15000.0,
        ),
    ]


@pytest.fixture
def sample_products_recyclability():
    """Create sample product recyclability assessments."""
    return [
        ProductRecyclability(
            product_id="prod-001",
            product_name="Industrial Pump",
            recyclability_score_pct=75.0,
            design_for_disassembly=True,
            material_passport=True,
            substances_of_concern_count=2,
        ),
        ProductRecyclability(
            product_id="prod-002",
            product_name="Control Unit",
            recyclability_score_pct=45.0,
            design_for_disassembly=False,
            material_passport=False,
            substances_of_concern_count=5,
        ),
    ]


# ---------------------------------------------------------------------------
# Test Classes
# ---------------------------------------------------------------------------


class TestInitialization:
    """Test engine initialization."""

    def test_default_init(self):
        """Engine initializes with default CircularEconomyConfig."""
        engine = CircularEconomyEngine()
        assert engine.config is not None
        assert isinstance(engine.config, CircularEconomyConfig)
        assert engine.config.reporting_year == 2025
        assert engine.config.include_mci is True
        assert engine.engine_version == "1.0.0"

    def test_init_with_config(self):
        """Engine initializes with explicit config."""
        config = CircularEconomyConfig(
            reporting_year=2024,
            include_mci=False,
            include_epr=True,
        )
        engine = CircularEconomyEngine(config)
        assert engine.config.reporting_year == 2024
        assert engine.config.include_mci is False

    def test_init_with_dict(self):
        """Engine initializes from a dictionary."""
        engine = CircularEconomyEngine({
            "reporting_year": 2023,
            "include_product_recyclability": False,
        })
        assert engine.config.reporting_year == 2023
        assert engine.config.include_product_recyclability is False

    def test_init_with_none(self):
        """Engine initializes with None (defaults)."""
        engine = CircularEconomyEngine(None)
        assert engine.config.waste_hierarchy_compliance is True


class TestMCI:
    """Test Material Circularity Index calculation."""

    def test_mci_calculation(self, default_engine, sample_materials, sample_waste_streams):
        """MCI is computed and within [0, 1] range."""
        mci = default_engine.calculate_mci(
            sample_materials, sample_waste_streams,
            total_input=1200.0, total_recycled=250.0,
            total_waste=280.0, diverted=170.0,
        )
        assert isinstance(mci, MCIResult)
        assert 0.0 <= mci.mci_score <= 1.0

    def test_mci_fully_linear(self, default_engine):
        """Fully linear system (all virgin, all to landfill) has MCI near 0.1."""
        materials = [
            MaterialFlowData(
                material_name="Steel",
                virgin_input_tonnes=1000.0,
                recycled_input_tonnes=0.0,
            ),
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.METAL_SCRAP,
                quantity_tonnes=1000.0,
                destination=WasteDestination.LANDFILL,
            ),
        ]
        mci = default_engine.calculate_mci(
            materials, waste,
            total_input=1000.0, total_recycled=0.0,
            total_waste=1000.0, diverted=0.0,
        )
        assert mci.mci_score == pytest.approx(0.1, abs=0.05)
        assert "linear" in mci.interpretation.lower() or "low" in mci.interpretation.lower()

    def test_mci_fully_circular(self, default_engine):
        """Fully circular system (all recycled, all recycled output) has high MCI."""
        materials = [
            MaterialFlowData(
                material_name="Steel",
                virgin_input_tonnes=0.0,
                recycled_input_tonnes=1000.0,
            ),
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.METAL_SCRAP,
                quantity_tonnes=1000.0,
                destination=WasteDestination.RECYCLING,
            ),
        ]
        mci = default_engine.calculate_mci(
            materials, waste,
            total_input=1000.0, total_recycled=1000.0,
            total_waste=1000.0, diverted=1000.0,
        )
        assert mci.mci_score >= 0.9

    def test_mci_range(self, default_engine, sample_materials, sample_waste_streams):
        """MCI score clamped between 0.0 and 1.0."""
        mci = default_engine.calculate_mci(
            sample_materials, sample_waste_streams,
        )
        assert 0.0 <= mci.mci_score <= 1.0
        assert 0.0 <= mci.linear_flow_index <= 1.0

    def test_mci_formula(self, default_engine):
        """MCI formula: MCI = 1 - LFI * (0.9/X). Verify known values."""
        materials = [
            MaterialFlowData(
                material_name="Material",
                virgin_input_tonnes=500.0,
                recycled_input_tonnes=500.0,
            ),
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=200.0,
                destination=WasteDestination.RECYCLING,
            ),
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=200.0,
                destination=WasteDestination.LANDFILL,
            ),
        ]
        mci = default_engine.calculate_mci(
            materials, waste,
            total_input=1000.0, total_recycled=500.0,
            total_waste=400.0, diverted=200.0,
            utility_factor=1.0,
        )
        expected_lfi = 700.0 / 2000.0
        expected_mci = 1.0 - expected_lfi * 0.9
        assert mci.linear_flow_index == pytest.approx(expected_lfi, abs=0.01)
        assert mci.mci_score == pytest.approx(expected_mci, abs=0.01)


class TestRecycledContent:
    """Test recycled content calculations."""

    def test_recycled_input_pct(self, default_engine, sample_materials, sample_waste_streams):
        """Recycled content percentage computed correctly."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        expected_pct = (250.0 / 1200.0) * 100.0
        assert result.recycled_content_pct == pytest.approx(
            _round2(expected_pct), abs=0.1
        )

    def test_pre_consumer_vs_post_consumer(self):
        """Pre-consumer and post-consumer recycled percentages are tracked."""
        mat = MaterialFlowData(
            material_name="Steel",
            virgin_input_tonnes=700.0,
            recycled_input_tonnes=300.0,
            pre_consumer_recycled_pct=10.0,
            post_consumer_recycled_pct=20.0,
        )
        assert mat.pre_consumer_recycled_pct == 10.0
        assert mat.post_consumer_recycled_pct == 20.0
        assert mat.total_input_tonnes == 1000.0

    def test_zero_recycled(self, default_engine):
        """Zero recycled content gives 0% recycled content."""
        materials = [
            MaterialFlowData(
                material_name="Virgin Only",
                virgin_input_tonnes=500.0,
                recycled_input_tonnes=0.0,
            ),
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=50.0,
                destination=WasteDestination.LANDFILL,
            ),
        ]
        result = default_engine.calculate_circular_metrics(
            materials=materials,
            waste_streams=waste,
        )
        assert result.recycled_content_pct == 0.0

    def test_high_recycled(self, default_engine):
        """High recycled content is correctly reported."""
        materials = [
            MaterialFlowData(
                material_name="Recycled Steel",
                virgin_input_tonnes=50.0,
                recycled_input_tonnes=950.0,
            ),
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.METAL_SCRAP,
                quantity_tonnes=100.0,
                destination=WasteDestination.RECYCLING,
            ),
        ]
        result = default_engine.calculate_circular_metrics(
            materials=materials,
            waste_streams=waste,
        )
        assert result.recycled_content_pct == pytest.approx(95.0, abs=0.1)


class TestWasteHierarchy:
    """Test waste hierarchy breakdown and scoring."""

    def test_hierarchy_weights(self):
        """Waste hierarchy weights are properly ordered."""
        assert WASTE_HIERARCHY_WEIGHTS["prevention"] == 1.0
        assert WASTE_HIERARCHY_WEIGHTS["reuse"] == 0.9
        assert WASTE_HIERARCHY_WEIGHTS["recycling"] == 0.7
        assert WASTE_HIERARCHY_WEIGHTS["landfill"] == 0.0

    def test_recycling_dominant(self, default_engine):
        """Waste hierarchy score is high when recycling dominates."""
        waste = [
            WasteStreamData(
                waste_type=WasteType.METAL_SCRAP,
                quantity_tonnes=900.0,
                destination=WasteDestination.RECYCLING,
            ),
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=100.0,
                destination=WasteDestination.LANDFILL,
            ),
        ]
        hierarchy = default_engine.calculate_waste_hierarchy(waste)
        score = hierarchy["_summary"]["weighted_hierarchy_score"]
        assert score > 0.5

    def test_landfill_dominant(self, default_engine):
        """Waste hierarchy score is low when landfill dominates."""
        waste = [
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=900.0,
                destination=WasteDestination.LANDFILL,
            ),
            WasteStreamData(
                waste_type=WasteType.METAL_SCRAP,
                quantity_tonnes=100.0,
                destination=WasteDestination.RECYCLING,
            ),
        ]
        hierarchy = default_engine.calculate_waste_hierarchy(waste)
        score = hierarchy["_summary"]["weighted_hierarchy_score"]
        assert score < 0.3

    def test_waste_diversion_rate(self, default_engine, sample_materials, sample_waste_streams):
        """Waste diversion rate calculated correctly."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        expected = (170.0 / 280.0) * 100.0
        assert result.waste_diversion_rate_pct == pytest.approx(
            _round2(expected), abs=0.1
        )

    def test_total_waste(self, default_engine, sample_materials, sample_waste_streams):
        """Total waste sums all streams."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        expected_total = 120.0 + 80.0 + 50.0 + 30.0
        assert result.total_waste_generated_tonnes == pytest.approx(expected_total, rel=1e-3)


class TestEPRCompliance:
    """Test Extended Producer Responsibility compliance assessment."""

    def test_packaging_target(self, default_engine):
        """Packaging EPR target is 70% per PPWR."""
        assert EPR_RECYCLING_TARGETS["packaging"]["overall_target_pct"] == 70.0

    def test_weee_target(self, default_engine):
        """WEEE EPR target is 65%."""
        assert EPR_RECYCLING_TARGETS["weee"]["overall_target_pct"] == 65.0

    def test_batteries_target(self, default_engine):
        """Batteries EPR target is 70% per Battery Regulation."""
        assert EPR_RECYCLING_TARGETS["batteries"]["overall_target_pct"] == 70.0

    def test_epr_status(self, default_engine, sample_waste_streams):
        """EPR compliance produces structured assessment."""
        waste_with_packaging = sample_waste_streams + [
            WasteStreamData(
                waste_type=WasteType.PACKAGING,
                quantity_tonnes=100.0,
                destination=WasteDestination.RECYCLING,
            ),
            WasteStreamData(
                waste_type=WasteType.PACKAGING,
                quantity_tonnes=20.0,
                destination=WasteDestination.LANDFILL,
            ),
        ]
        epr_result = default_engine.assess_epr_compliance(
            waste_with_packaging,
            schemes=[EPRScheme.PACKAGING],
        )
        assert "packaging" in epr_result
        pkg = epr_result["packaging"]
        assert "target_pct" in pkg
        assert "actual_rate_pct" in pkg
        assert "compliant" in pkg
        assert pkg["compliant"] is True


class TestWasteIntensity:
    """Test waste intensity per revenue."""

    def test_waste_per_revenue(self, default_engine):
        """Waste intensity = waste_t / revenue_millions."""
        intensity = default_engine.calculate_waste_intensity(500.0, 100_000_000.0)
        assert intensity == pytest.approx(5.0, rel=1e-6)

    def test_waste_per_unit(self, default_engine, sample_materials, sample_waste_streams):
        """Waste intensity appears in full result."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
            annual_revenue_eur=50_000_000.0,
        )
        assert result.waste_intensity_per_revenue > 0.0
        expected = 280.0 / 50.0
        assert result.waste_intensity_per_revenue == pytest.approx(
            _round3(expected), abs=0.1
        )

    def test_zero_revenue(self, default_engine):
        """Zero revenue returns 0.0 waste intensity (safe divide)."""
        intensity = default_engine.calculate_waste_intensity(500.0, 0.0)
        assert intensity == 0.0


class TestProductRecyclability:
    """Test product recyclability scoring."""

    def test_recyclability_score(
        self, default_engine, sample_materials, sample_waste_streams,
        sample_products_recyclability
    ):
        """Product recyclability scores appear in result."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
            products=sample_products_recyclability,
        )
        assert len(result.product_recyclability_scores) == 2
        scores = {
            s["product_name"]: s["recyclability_score_pct"]
            for s in result.product_recyclability_scores
        }
        assert scores["Industrial Pump"] == 75.0
        assert scores["Control Unit"] == 45.0

    def test_design_for_disassembly(self, sample_products_recyclability):
        """Design for disassembly flag is correctly tracked."""
        pump = sample_products_recyclability[0]
        assert pump.design_for_disassembly is True
        control = sample_products_recyclability[1]
        assert control.design_for_disassembly is False

    def test_substances_of_concern(self, sample_products_recyclability):
        """Substances of concern count is tracked."""
        pump = sample_products_recyclability[0]
        assert pump.substances_of_concern_count == 2
        control = sample_products_recyclability[1]
        assert control.substances_of_concern_count == 5


class TestProvenance:
    """Test provenance hash generation."""

    def test_hash(self, default_engine, sample_materials, sample_waste_streams):
        """Result has a 64-character provenance hash."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_deterministic(self):
        """Same data produces the same hash."""
        data = {"mci": 0.65, "waste": 280.0}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_different_input(self):
        """Different data produces different hashes."""
        h1 = _compute_hash({"mci": 0.65})
        h2 = _compute_hash({"mci": 0.70})
        assert h1 != h2


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_waste_streams(self, default_engine, sample_materials):
        """Empty waste streams still produce a result (no error)."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=[],
        )
        assert result.total_waste_generated_tonnes == 0.0
        assert result.waste_diversion_rate_pct == 0.0

    def test_large_dataset(self, default_engine):
        """Large number of materials and waste streams does not crash."""
        materials = [
            MaterialFlowData(
                material_name=f"Material_{i}",
                virgin_input_tonnes=100.0,
                recycled_input_tonnes=50.0,
            )
            for i in range(100)
        ]
        waste = [
            WasteStreamData(
                waste_type=WasteType.OTHER,
                quantity_tonnes=10.0,
                destination=WasteDestination.RECYCLING,
            )
            for _ in range(200)
        ]
        result = default_engine.calculate_circular_metrics(
            materials=materials,
            waste_streams=waste,
        )
        assert result.total_material_input_tonnes > 0.0
        assert result.processing_time_ms >= 0.0

    def test_result_fields(self, default_engine, sample_materials, sample_waste_streams):
        """CircularEconomyResult contains all expected fields."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        assert hasattr(result, "result_id")
        assert hasattr(result, "total_material_input_tonnes")
        assert hasattr(result, "total_recycled_input_tonnes")
        assert hasattr(result, "recycled_content_pct")
        assert hasattr(result, "total_waste_generated_tonnes")
        assert hasattr(result, "waste_diverted_tonnes")
        assert hasattr(result, "waste_diversion_rate_pct")
        assert hasattr(result, "waste_hierarchy_breakdown")
        assert hasattr(result, "material_circularity_index")
        assert hasattr(result, "epr_compliance")
        assert hasattr(result, "product_recyclability_scores")
        assert hasattr(result, "waste_intensity_per_revenue")
        assert hasattr(result, "methodology_notes")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "engine_version")
        assert hasattr(result, "calculated_at")
        assert hasattr(result, "provenance_hash")

    def test_methodology_notes(self, default_engine, sample_materials, sample_waste_streams):
        """Result includes methodology notes with key information."""
        result = default_engine.calculate_circular_metrics(
            materials=sample_materials,
            waste_streams=sample_waste_streams,
        )
        notes_text = " ".join(result.methodology_notes)
        assert "Reporting year" in notes_text
        assert "Engine version" in notes_text
        assert "Material inflows" in notes_text
        assert "Waste generated" in notes_text
