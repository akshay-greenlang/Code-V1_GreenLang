# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - IRO Identification Engine Tests
================================================================================

Unit tests for IROIdentificationEngine (Engine 4) covering IRO catalog
access, IRO identification, classification, register building, filtering,
sector weighting, provenance hashing, and edge cases.

ESRS 1 Para 28-39: Impacts, Risks, and Opportunities identification.

Target: 45+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the iro_identification engine module."""
    return _load_engine("iro_identification")


@pytest.fixture
def engine(mod):
    """Create a fresh IROIdentificationEngine instance."""
    return mod.IROIdentificationEngine()


@pytest.fixture
def sample_iro(mod):
    """Create a sample IRO for testing."""
    return mod.IRO(
        id="IRO-001",
        name="GHG Emission Reduction Failure",
        description="Risk of failing to reduce greenhouse gas emissions",
        esrs_topic=mod.ESRSTopic.E1_CLIMATE,
        iro_type=mod.IROType.RISK,
        value_chain_stages=[mod.ValueChainStage.OWN_OPERATIONS],
        time_horizons=[mod.TimeHorizon.SHORT_TERM],
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestIROEnums:
    """Tests for IRO identification enums."""

    def test_iro_type_count(self, mod):
        """IROType has 6 values."""
        assert len(mod.IROType) == 6

    def test_iro_type_includes_core_types(self, mod):
        """IROType includes RISK and OPPORTUNITY."""
        names = {m.name for m in mod.IROType}
        assert "RISK" in names
        assert "OPPORTUNITY" in names

    def test_priority_level_count(self, mod):
        """PriorityLevel has 4 values."""
        assert len(mod.PriorityLevel) == 4

    def test_priority_level_values(self, mod):
        """PriorityLevel includes CRITICAL, HIGH, MEDIUM, LOW."""
        names = {m.name for m in mod.PriorityLevel}
        expected = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}
        assert expected.issubset(names)

    def test_esrs_topic_count(self, mod):
        """ESRSTopic has 10 values."""
        assert len(mod.ESRSTopic) == 10

    def test_value_chain_stage_count(self, mod):
        """ValueChainStage has 3 values."""
        assert len(mod.ValueChainStage) == 3

    def test_time_horizon_count(self, mod):
        """TimeHorizon has 3 values."""
        assert len(mod.TimeHorizon) == 3


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestIROConstants:
    """Tests for IRO identification constants."""

    def test_esrs_iro_catalog_has_10_topics(self, mod):
        """ESRS_IRO_CATALOG has entries for all 10 ESRS topics."""
        assert len(mod.ESRS_IRO_CATALOG) == 10

    def test_esrs_iro_catalog_total_entries(self, mod):
        """ESRS_IRO_CATALOG has 90+ total entries."""
        total = sum(len(v) for v in mod.ESRS_IRO_CATALOG.values())
        assert total >= 90

    def test_esrs_iro_catalog_e1_entries(self, mod):
        """E1 climate has 10+ IRO entries."""
        key = None
        for k in mod.ESRS_IRO_CATALOG:
            if "e1" in str(k).lower() or "climate" in str(k).lower():
                key = k
                break
        assert key is not None
        assert len(mod.ESRS_IRO_CATALOG[key]) >= 10

    def test_priority_thresholds_ordering(self, mod):
        """PRIORITY_THRESHOLDS: critical > high > medium > low."""
        thresholds = mod.PRIORITY_THRESHOLDS
        assert thresholds["critical"] > thresholds["high"]
        assert thresholds["high"] > thresholds["medium"]
        assert thresholds["medium"] > thresholds["low"]

    def test_priority_thresholds_values(self, mod):
        """PRIORITY_THRESHOLDS: critical=0.80, high=0.60, medium=0.40, low=0.00."""
        assert float(mod.PRIORITY_THRESHOLDS["critical"]) == pytest.approx(0.80)
        assert float(mod.PRIORITY_THRESHOLDS["high"]) == pytest.approx(0.60)
        assert float(mod.PRIORITY_THRESHOLDS["medium"]) == pytest.approx(0.40)
        assert float(mod.PRIORITY_THRESHOLDS["low"]) == pytest.approx(0.00)

    def test_sector_iro_priorities_has_default(self, mod):
        """SECTOR_IRO_PRIORITIES has a 'default' entry."""
        assert "default" in mod.SECTOR_IRO_PRIORITIES

    def test_sector_iro_priorities_count(self, mod):
        """SECTOR_IRO_PRIORITIES has at least 8 sectors."""
        assert len(mod.SECTOR_IRO_PRIORITIES) >= 8


# ===========================================================================
# Pydantic Model Tests
# ===========================================================================


class TestIROModel:
    """Tests for IRO Pydantic model."""

    def test_create_valid_iro(self, mod):
        """Create a valid IRO."""
        iro = mod.IRO(
            name="Climate Risk",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        assert iro.name == "Climate Risk"
        assert len(iro.id) > 0

    def test_iro_requires_name(self, mod):
        """IRO requires a name."""
        with pytest.raises(Exception):
            mod.IRO(
                name="",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
                iro_type=mod.IROType.RISK,
            )


class TestIROAssessmentModel:
    """Tests for IROAssessment Pydantic model."""

    def test_create_valid_assessment(self, mod):
        """Create a valid IROAssessment."""
        asmt = mod.IROAssessment(
            iro_id="IRO-001",
            impact_score=Decimal("0.70"),
            financial_score=Decimal("0.50"),
        )
        assert asmt.impact_score == Decimal("0.70")
        assert asmt.financial_score == Decimal("0.50")


# ===========================================================================
# Identify IROs Tests
# ===========================================================================


class TestIdentifyIROs:
    """Tests for identify_iros method."""

    def test_identify_iros_single_topic(self, engine, mod):
        """Identify IROs for a single ESRS topic."""
        iros = engine.identify_iros(topics=[mod.ESRSTopic.E1_CLIMATE])
        assert isinstance(iros, list)
        assert len(iros) >= 1

    def test_identify_iros_multiple_topics(self, engine, mod):
        """Identify IROs for multiple ESRS topics."""
        iros = engine.identify_iros(
            topics=[mod.ESRSTopic.E1_CLIMATE, mod.ESRSTopic.S1_OWN_WORKFORCE]
        )
        assert len(iros) >= 2

    def test_identify_iros_all_topics(self, engine, mod):
        """Identify IROs for all 10 ESRS topics."""
        iros = engine.identify_iros(topics=list(mod.ESRSTopic))
        assert len(iros) >= 50

    def test_identify_iros_with_value_chain_filter(self, engine, mod):
        """Filter IROs by value chain stage."""
        iros = engine.identify_iros(
            topics=[mod.ESRSTopic.E1_CLIMATE],
            value_chain_stages=[mod.ValueChainStage.OWN_OPERATIONS],
        )
        assert isinstance(iros, list)

    def test_identify_iros_returns_iro_objects(self, engine, mod):
        """Returned items have IRO-like attributes."""
        iros = engine.identify_iros(topics=[mod.ESRSTopic.E1_CLIMATE])
        for iro in iros:
            assert hasattr(iro, "name")
            assert hasattr(iro, "iro_type")


# ===========================================================================
# Classify IRO Tests
# ===========================================================================


class TestClassifyIRO:
    """Tests for classify_iro method."""

    def test_classify_critical_iro(self, engine, mod):
        """High combined score => CRITICAL priority."""
        iro = mod.IRO(
            id="CL-CRIT", name="Critical IRO",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        # classify_iro takes (iro, impact_score, financial_score)
        assessment = engine.classify_iro(
            iro,
            impact_score=Decimal("0.90"),
            financial_score=Decimal("0.85"),
        )
        assert assessment is not None
        # combined_score = max(0.90, 0.85) = 0.90 >= 0.80 => CRITICAL
        assert hasattr(assessment, "priority_level")
        assert assessment.priority_level == mod.PriorityLevel.CRITICAL

    def test_classify_low_iro(self, engine, mod):
        """Low combined score => LOW priority."""
        iro = mod.IRO(
            id="CL-LOW", name="Low IRO",
            esrs_topic=mod.ESRSTopic.G1_BUSINESS_CONDUCT,
            iro_type=mod.IROType.OPPORTUNITY,
        )
        assessment = engine.classify_iro(
            iro,
            impact_score=Decimal("0.10"),
            financial_score=Decimal("0.15"),
        )
        assert assessment is not None
        assert assessment.priority_level == mod.PriorityLevel.LOW

    def test_classify_combined_score_is_max(self, engine, mod):
        """Combined score = max(impact, financial)."""
        iro = mod.IRO(
            id="CL-MAX", name="Max Test",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        assessment = engine.classify_iro(
            iro,
            impact_score=Decimal("0.30"),
            financial_score=Decimal("0.70"),
        )
        # combined should be 0.70 (max of 0.30, 0.70)
        assert assessment.combined_score == pytest.approx(
            Decimal("0.70"), abs=Decimal("0.01")
        )

    def test_classify_material_determination(self, engine, mod):
        """IRO above threshold is marked as material."""
        iro = mod.IRO(
            id="CL-MAT", name="Material Test",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        assessment = engine.classify_iro(
            iro,
            impact_score=Decimal("0.60"),
            financial_score=Decimal("0.50"),
        )
        # max(0.60, 0.50) = 0.60 >= 0.40 threshold
        assert assessment.is_material is True

    def test_classify_not_material(self, engine, mod):
        """IRO below threshold is not material."""
        iro = mod.IRO(
            id="CL-NMAT", name="Not Material Test",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        assessment = engine.classify_iro(
            iro,
            impact_score=Decimal("0.10"),
            financial_score=Decimal("0.20"),
        )
        assert assessment.is_material is False


# ===========================================================================
# Build Register Tests
# ===========================================================================


class TestBuildRegister:
    """Tests for build_register method."""

    def _make_iros_and_assessments(self, engine, mod, count=3):
        """Helper to create IROs and classify them."""
        iros = [
            mod.IRO(
                id=f"REG-{i}", name=f"IRO {i}",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
                iro_type=mod.IROType.RISK,
            )
            for i in range(count)
        ]
        assessments = [
            engine.classify_iro(
                iros[i],
                impact_score=Decimal(str(0.30 + i * 0.20)),
                financial_score=Decimal(str(0.20 + i * 0.15)),
            )
            for i in range(count)
        ]
        return iros, assessments

    def test_build_register_basic(self, engine, mod):
        """build_register returns an IRORegister with provenance."""
        iros, assessments = self._make_iros_and_assessments(engine, mod)
        register = engine.build_register(iros, assessments)
        assert register is not None
        assert hasattr(register, "provenance_hash")
        assert len(register.provenance_hash) == 64

    def test_build_register_empty_iros_raises(self, engine, mod):
        """Empty IROs list raises ValueError."""
        with pytest.raises(ValueError):
            engine.build_register([], [])

    def test_build_register_statistics(self, engine, mod):
        """Register contains summary statistics."""
        iros, assessments = self._make_iros_and_assessments(engine, mod, count=5)
        register = engine.build_register(iros, assessments)
        assert register.total_count == 5

    def test_build_register_by_topic(self, engine, mod):
        """Register contains by_topic breakdown."""
        iros, assessments = self._make_iros_and_assessments(engine, mod)
        register = engine.build_register(iros, assessments)
        assert hasattr(register, "by_topic")
        assert isinstance(register.by_topic, dict)


# ===========================================================================
# Catalog Access Tests
# ===========================================================================


class TestCatalogAccess:
    """Tests for catalog access methods."""

    def test_get_catalog_for_topic(self, engine, mod):
        """get_catalog_for_topic returns entries for a topic."""
        entries = engine.get_catalog_for_topic(mod.ESRSTopic.E1_CLIMATE)
        assert isinstance(entries, list)
        assert len(entries) >= 5

    def test_get_catalog_summary(self, engine):
        """get_catalog_summary returns topic -> count mapping."""
        summary = engine.get_catalog_summary()
        assert isinstance(summary, dict)
        total = sum(summary.values())
        assert total >= 90

    def test_get_catalog_size(self, engine):
        """get_catalog_size returns total number of catalog entries."""
        size = engine.get_catalog_size()
        assert size >= 90


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestIROProvenanceHash:
    """Tests for IRO identification provenance hashing."""

    def test_register_hash_is_64_chars(self, engine, mod):
        """Register provenance hash is a 64-character SHA-256 hex string."""
        iros = [
            mod.IRO(
                id="PH-1", name="Hash Test",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
                iro_type=mod.IROType.RISK,
            ),
        ]
        assessments = [
            engine.classify_iro(
                iros[0],
                impact_score=Decimal("0.70"),
                financial_score=Decimal("0.60"),
            ),
        ]
        register = engine.build_register(iros, assessments)
        assert len(register.provenance_hash) == 64
        int(register.provenance_hash, 16)

    def test_register_hash_deterministic_stats(self, engine, mod):
        """Same inputs produce identical statistics."""
        iros = [
            mod.IRO(
                id="DET-1", name="Determinism Test",
                esrs_topic=mod.ESRSTopic.E1_CLIMATE,
                iro_type=mod.IROType.RISK,
            ),
        ]
        a1 = engine.classify_iro(iros[0], Decimal("0.70"), Decimal("0.60"))
        r1 = engine.build_register(iros, [a1])
        a2 = engine.classify_iro(iros[0], Decimal("0.70"), Decimal("0.60"))
        r2 = engine.build_register(iros, [a2])
        assert r1.total_count == r2.total_count

    def test_assessment_hash_is_64_chars(self, engine, mod):
        """Assessment provenance hash is a 64-character hex string."""
        iro = mod.IRO(
            id="AH-1", name="Assessment Hash",
            esrs_topic=mod.ESRSTopic.E1_CLIMATE,
            iro_type=mod.IROType.RISK,
        )
        assessment = engine.classify_iro(
            iro, Decimal("0.70"), Decimal("0.60"),
        )
        assert len(assessment.provenance_hash) == 64
        int(assessment.provenance_hash, 16)
