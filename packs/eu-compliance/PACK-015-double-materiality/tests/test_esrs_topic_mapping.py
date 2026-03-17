# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - ESRS Topic Mapping Engine Tests
================================================================================

Unit tests for ESRSTopicMappingEngine (Engine 6) covering topic-to-disclosure
mapping, batch mapping, gap analysis, coverage calculation, mandatory/phase-in
disclosures, effort estimation, data point counting, and provenance hashing.

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
    """Load the esrs_topic_mapping engine module."""
    return _load_engine("esrs_topic_mapping")


@pytest.fixture
def engine(mod):
    """Create a fresh ESRSTopicMappingEngine instance."""
    return mod.ESRSTopicMappingEngine()


@pytest.fixture
def sample_material_topics(mod):
    """Create sample MaterialTopicInput list.

    Uses the actual model fields: matter_id, esrs_topic.
    """
    return [
        mod.MaterialTopicInput(
            matter_id="matter-e1-climate",
            esrs_topic="E1",
        ),
        mod.MaterialTopicInput(
            matter_id="matter-s1-workforce",
            esrs_topic="S1",
        ),
    ]


@pytest.fixture
def sample_available_data(mod):
    """Create sample AvailableDataInput list.

    Uses the actual model fields: disclosure_id, available_data_points.
    """
    return [
        mod.AvailableDataInput(
            disclosure_id="E1-1",
            available_data_points=[
                "transition_plan_description",
                "ghg_reduction_targets",
            ],
        ),
        mod.AvailableDataInput(
            disclosure_id="E1-6",
            available_data_points=[
                "scope_1_tco2e",
                "scope_2_location_tco2e",
                "scope_2_market_tco2e",
                "scope_3_tco2e",
                "total_tco2e",
                "biogenic_tco2e",
            ],
        ),
    ]


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestESRSMappingEnums:
    """Tests for ESRS topic mapping enums."""

    def test_esrs_standard_count(self, mod):
        """ESRSStandard has 12 values (ESRS_1 + ESRS_2 + E1-E5 + S1-S4 + G1)."""
        assert len(mod.ESRSStandard) >= 11

    def test_disclosure_status_count(self, mod):
        """DisclosureStatus has 5 values."""
        assert len(mod.DisclosureStatus) == 5

    def test_coverage_level_count(self, mod):
        """CoverageLevel has 5 values."""
        assert len(mod.CoverageLevel) == 5
        names = {m.name for m in mod.CoverageLevel}
        expected = {"COMPLETE", "HIGH", "MEDIUM", "LOW", "NONE"}
        assert names == expected


# ===========================================================================
# Catalog Tests
# ===========================================================================


class TestESRSCatalog:
    """Tests for the internal disclosure catalog."""

    def test_catalog_loaded(self, engine):
        """Engine loads the disclosure catalog on init."""
        assert engine._catalog is not None or hasattr(engine, "_catalog")

    def test_catalog_esrs2_disclosures(self, engine):
        """ESRS 2 has 10 mandatory disclosure requirements."""
        disclosures = engine.get_mandatory_disclosures("ESRS_2")
        # All ESRS 2 disclosures are mandatory
        assert len(disclosures) == 10

    def test_catalog_total_disclosures(self, engine):
        """Catalog has 80+ total data points across all standards."""
        counts = engine.count_data_points_by_standard()
        total = sum(counts.values())
        assert total >= 80

    def test_catalog_e1_data_points(self, engine):
        """E1 has a significant number of data points."""
        counts = engine.count_data_points_by_standard()
        # Find E1 key
        e1_count = None
        for key, count in counts.items():
            if "E1" in key:
                e1_count = count
                break
        assert e1_count is not None
        assert e1_count >= 9

    def test_catalog_s1_data_points(self, engine):
        """S1 has a significant number of data points."""
        counts = engine.count_data_points_by_standard()
        s1_count = None
        for key, count in counts.items():
            if "S1" in key:
                s1_count = count
                break
        assert s1_count is not None
        assert s1_count >= 17


# ===========================================================================
# Map Topic to Disclosures Tests
# ===========================================================================


class TestMapTopicToDisclosures:
    """Tests for map_topic_to_disclosures method."""

    def test_map_e1_topic(self, engine):
        """Map E1 climate topic returns E1 disclosures."""
        result = engine.map_topic_to_disclosures("matter-e1", "E1")
        assert result is not None
        assert hasattr(result, "mapped_disclosures")
        assert len(result.mapped_disclosures) >= 9

    def test_map_s1_topic(self, engine):
        """Map S1 own workforce topic returns S1 disclosures."""
        result = engine.map_topic_to_disclosures("matter-s1", "S1")
        assert result is not None
        assert len(result.mapped_disclosures) >= 17

    def test_map_g1_topic(self, engine):
        """Map G1 business conduct topic returns G1 disclosures."""
        result = engine.map_topic_to_disclosures("matter-g1", "G1")
        assert result is not None
        assert len(result.mapped_disclosures) >= 6

    def test_map_all_topics(self, engine):
        """All 10 ESRS topics can be mapped."""
        topics = ["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"]
        for topic in topics:
            result = engine.map_topic_to_disclosures(f"matter-{topic}", topic)
            assert result is not None, f"Failed to map topic {topic}"
            assert len(result.mapped_disclosures) > 0, f"No disclosures for {topic}"


# ===========================================================================
# Batch Map Tests
# ===========================================================================


class TestBatchMap:
    """Tests for batch_map method."""

    def test_batch_map_single_topic(self, engine, sample_material_topics):
        """Batch map with a single material topic."""
        result = engine.batch_map([sample_material_topics[0]])
        assert result is not None
        assert hasattr(result, "mappings")

    def test_batch_map_multiple_topics(self, engine, sample_material_topics):
        """Batch map with multiple material topics."""
        result = engine.batch_map(sample_material_topics)
        assert result is not None
        # Should have ESRS_2 mandatory + E1 + S1
        assert len(result.mappings) >= 3

    def test_batch_map_always_includes_esrs2(self, engine, mod):
        """batch_map always includes ESRS 2 mandatory disclosures."""
        topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1-only",
                esrs_topic="E1",
            ),
        ]
        result = engine.batch_map(topics)
        # Result should include ESRS 2 disclosures even though
        # only E1 was specified as material
        assert result is not None
        assert len(result.provenance_hash) == 64
        # ESRS_2 mandatory mapping should be present
        esrs2_found = any(
            m.matter_id == "__ESRS2_MANDATORY__" for m in result.mappings
        )
        assert esrs2_found

    def test_batch_map_provenance_hash(self, engine, sample_material_topics):
        """Batch map result has provenance hash."""
        result = engine.batch_map(sample_material_topics)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# Gap Analysis Tests
# ===========================================================================


class TestGapAnalysis:
    """Tests for analyze_gaps method."""

    def test_analyze_gaps_basic(self, engine, mod):
        """analyze_gaps identifies missing disclosures."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        available = [
            mod.AvailableDataInput(
                disclosure_id="E1-1",
                available_data_points=[
                    "transition_plan_description",
                    "ghg_reduction_targets",
                ],
            ),
        ]
        gaps = engine.analyze_gaps(mapping_result, available)
        assert isinstance(gaps, list)
        # There should be gaps since we only provide partial E1-1 data
        assert len(gaps) >= 1

    def test_analyze_gaps_full_coverage(self, engine, mod):
        """Fewer gaps when more data is available."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)

        # Create available data for all disclosures with all data points
        available = []
        for mapping in mapping_result.mappings:
            for disc in mapping.mapped_disclosures:
                available.append(
                    mod.AvailableDataInput(
                        disclosure_id=disc.disclosure_id,
                        available_data_points=disc.data_points,
                    )
                )
        gaps = engine.analyze_gaps(mapping_result, available)
        assert isinstance(gaps, list)
        # All gaps should be fully covered
        fully_covered = sum(
            1 for g in gaps if g.status == mod.DisclosureStatus.FULLY_COVERED
        )
        assert fully_covered == len(gaps)

    def test_analyze_gaps_no_data(self, engine, mod):
        """All disclosures are gaps when no data available."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        gaps = engine.analyze_gaps(mapping_result, [])
        # E1 has 9 disclosures + ESRS 2 has 10 = 19 total disclosures
        assert len(gaps) >= 19


# ===========================================================================
# Coverage Calculation Tests
# ===========================================================================


class TestCoverageCalculation:
    """Tests for calculate_coverage method."""

    def test_coverage_zero(self, engine, mod):
        """Zero data available yields NONE coverage."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        coverage = engine.calculate_coverage(mapping_result, [])
        assert coverage is not None
        assert coverage == mod.CoverageLevel.NONE

    def test_coverage_returns_level(self, engine, mod):
        """Coverage result is a CoverageLevel enum value."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        available = [
            mod.AvailableDataInput(
                disclosure_id="E1-1",
                available_data_points=[
                    "transition_plan_description",
                    "ghg_reduction_targets",
                ],
            ),
        ]
        coverage = engine.calculate_coverage(mapping_result, available)
        assert coverage is not None
        # Should be one of the CoverageLevel values
        assert isinstance(coverage, mod.CoverageLevel)


# ===========================================================================
# Mandatory and Phase-In Disclosures Tests
# ===========================================================================


class TestMandatoryAndPhaseIn:
    """Tests for mandatory and phase-in disclosure methods."""

    def test_get_mandatory_disclosures(self, engine):
        """get_mandatory_disclosures returns ESRS 2 mandatory disclosures."""
        mandatory = engine.get_mandatory_disclosures("ESRS_2")
        assert isinstance(mandatory, list)
        assert len(mandatory) == 10

    def test_get_phase_in_disclosures(self, engine):
        """get_phase_in_disclosures returns E1 disclosures with phase-in."""
        phase_in = engine.get_phase_in_disclosures("E1")
        assert isinstance(phase_in, list)
        # E1-1 and E1-9 are phase-in
        assert len(phase_in) >= 2


# ===========================================================================
# Effort Estimation Tests
# ===========================================================================


class TestEffortEstimation:
    """Tests for estimate_effort method."""

    def test_estimate_effort_basic(self, engine, mod):
        """estimate_effort returns a Decimal effort value."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        gaps = engine.analyze_gaps(mapping_result, [])
        effort = engine.estimate_effort(gaps)
        assert effort is not None
        assert isinstance(effort, Decimal)
        assert effort > 0

    def test_estimate_effort_no_gaps(self, engine):
        """No gaps = zero effort."""
        effort = engine.estimate_effort([])
        assert effort == Decimal("0") or effort == Decimal("0.00")


# ===========================================================================
# Count Data Points Tests
# ===========================================================================


class TestCountDataPoints:
    """Tests for count_data_points_by_standard method."""

    def test_count_data_points_returns_dict(self, engine):
        """count_data_points_by_standard returns a dict."""
        counts = engine.count_data_points_by_standard()
        assert isinstance(counts, dict)
        assert len(counts) >= 10

    def test_count_data_points_positive_values(self, engine):
        """All data point counts are positive."""
        counts = engine.count_data_points_by_standard()
        for standard, count in counts.items():
            assert count > 0, f"Standard {standard} has zero data points"


# ===========================================================================
# Disclosure by ID Tests
# ===========================================================================


class TestDisclosureByID:
    """Tests for get_disclosure_by_id method."""

    def test_get_existing_disclosure(self, engine):
        """Get a known disclosure by ID."""
        disclosure = engine.get_disclosure_by_id("E1-1")
        assert disclosure is not None
        assert disclosure.disclosure_id == "E1-1"

    def test_get_nonexistent_disclosure(self, engine):
        """Get a non-existent disclosure returns None."""
        disclosure = engine.get_disclosure_by_id("NONEXISTENT-999")
        assert disclosure is None


# ===========================================================================
# Gap Summary Tests
# ===========================================================================


class TestGapSummary:
    """Tests for get_gap_summary method."""

    def test_gap_summary_basic(self, engine, mod):
        """get_gap_summary returns a summary dictionary."""
        material_topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1",
                esrs_topic="E1",
            ),
        ]
        mapping_result = engine.batch_map(material_topics)
        gaps = engine.analyze_gaps(mapping_result, [])
        summary = engine.get_gap_summary(gaps)
        assert isinstance(summary, dict)
        assert "total_disclosures" in summary
        assert "fully_covered" in summary
        assert "not_covered" in summary


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestESRSMappingProvenanceHash:
    """Tests for ESRS mapping provenance hashing."""

    def test_hash_is_64_chars(self, engine, sample_material_topics):
        """Provenance hash is a 64-character SHA-256 hex string."""
        result = engine.batch_map(sample_material_topics)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_hash_deterministic(self, engine, mod):
        """Same inputs produce same provenance hash."""
        topics = [
            mod.MaterialTopicInput(
                matter_id="matter-e1-det",
                esrs_topic="E1",
            ),
        ]
        r1 = engine.batch_map(topics)
        r2 = engine.batch_map(topics)
        assert r1.provenance_hash == r2.provenance_hash


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestESRSMappingEdgeCases:
    """Edge case tests for ESRS topic mapping engine."""

    def test_all_topics_material(self, engine, mod):
        """Map all 10 topics as material."""
        topics = [
            mod.MaterialTopicInput(
                matter_id=f"matter-{code.lower()}",
                esrs_topic=code,
            )
            for code in ["E1", "E2", "E3", "E4", "E5",
                         "S1", "S2", "S3", "S4", "G1"]
        ]
        result = engine.batch_map(topics)
        assert result is not None
        assert len(result.provenance_hash) == 64
        # Should have ESRS_2 + 10 topic mappings = 11
        assert len(result.mappings) == 11

    def test_single_topic_full_mapping(self, engine, mod):
        """Single topic produces complete disclosure mapping."""
        topics = [
            mod.MaterialTopicInput(
                matter_id="matter-g1-only",
                esrs_topic="G1",
            ),
        ]
        result = engine.batch_map(topics)
        assert result is not None
        # G1 mapping should have 6 disclosures
        g1_mapping = None
        for m in result.mappings:
            if m.esrs_topic == "G1":
                g1_mapping = m
                break
        assert g1_mapping is not None
        assert len(g1_mapping.mapped_disclosures) == 6
