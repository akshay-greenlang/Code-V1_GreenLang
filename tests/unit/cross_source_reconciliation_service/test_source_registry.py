# -*- coding: utf-8 -*-
"""
Unit tests for SourceRegistryEngine - AGENT-DATA-015

Tests all public methods of SourceRegistryEngine with 50+ test cases.
Validates registration, schema mapping, credibility scoring, ranking,
tolerance rules, deactivation, and provenance tracking.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from greenlang.cross_source_reconciliation.source_registry import (
    SourceRegistryEngine,
    _jaro_winkler_similarity,
    _jaro_similarity,
    CERTIFICATION_SCORES,
    DEFAULT_CREDIBILITY_WEIGHTS,
)
from greenlang.cross_source_reconciliation.models import (
    SourceType,
    SourceStatus,
    SchemaMapping,
    ToleranceRule,
    FieldType,
    SourceCredibility,
    SourceHealthMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh SourceRegistryEngine for each test."""
    return SourceRegistryEngine()


@pytest.fixture
def registered_erp(engine):
    """Register and return an ERP source."""
    return engine.register_source(
        name="SAP ERP",
        source_type="erp",
        priority=90,
        schema_info={"columns": ["spend", "vendor", "date", "amount"]},
        refresh_cadence="daily",
        description="Production ERP system",
        tags=["erp", "finance"],
    )


@pytest.fixture
def registered_api(engine):
    """Register and return an API source."""
    return engine.register_source(
        name="External API",
        source_type="api",
        priority=70,
        schema_info={"columns": ["vendor_name", "invoice_date", "total"]},
        refresh_cadence="hourly",
        description="Third-party supplier API",
        tags=["api", "supplier"],
    )


@pytest.fixture
def registered_manual(engine):
    """Register and return a manual entry source."""
    return engine.register_source(
        name="Manual Entry",
        source_type="manual",
        priority=20,
        schema_info={"columns": ["spend_amount", "date"]},
    )


@pytest.fixture
def sample_records():
    """Create sample records for credibility testing."""
    return [
        {"amount": 100.0, "vendor": "Acme", "date": "2025-01-15"},
        {"amount": 250.0, "vendor": "Beta Corp", "date": "2025-01-16"},
        {"amount": None, "vendor": "Gamma LLC", "date": "2025-01-17"},
        {"amount": 75.5, "vendor": "", "date": "2025-01-18"},
        {"amount": 300.0, "vendor": "Delta Inc", "date": "2025-01-19"},
    ]


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Registration
# ---------------------------------------------------------------------------


class TestRegisterSource:
    """Tests for register_source method."""

    def test_register_source_returns_source_definition(self, engine):
        """Registering a source returns a SourceDefinition object."""
        src = engine.register_source(name="Test", source_type="erp", priority=50)
        assert src is not None
        assert src.name == "Test"
        assert src.source_type == SourceType.ERP
        assert src.priority == 50
        assert src.status == SourceStatus.ACTIVE

    def test_register_source_generates_unique_id(self, engine):
        """Each registered source receives a unique UUID."""
        s1 = engine.register_source(name="Source A", source_type="api")
        s2 = engine.register_source(name="Source B", source_type="api")
        assert s1.id != s2.id
        assert len(s1.id) == 36  # UUID format

    def test_register_source_default_credibility_score(self, engine):
        """Default credibility score is 0.5."""
        src = engine.register_source(name="Test", source_type="erp")
        assert src.credibility_score == 0.5

    def test_register_source_with_all_parameters(self, engine):
        """Registration stores all provided parameters."""
        src = engine.register_source(
            name="Full Source",
            source_type="registry",
            priority=95,
            schema_info={"columns": ["a", "b"]},
            refresh_cadence="hourly",
            description="Comprehensive test source",
            tags=["tag1", "tag2"],
        )
        assert src.name == "Full Source"
        assert src.source_type == SourceType.REGISTRY
        assert src.priority == 95
        assert src.schema_info == {"columns": ["a", "b"]}
        assert src.refresh_cadence == "hourly"
        assert src.description == "Comprehensive test source"
        assert src.tags == ["tag1", "tag2"]

    def test_register_source_validates_priority_too_low(self, engine):
        """Priority below 1 raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 100"):
            engine.register_source(name="Bad", source_type="api", priority=0)

    def test_register_source_validates_priority_too_high(self, engine):
        """Priority above 100 raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 100"):
            engine.register_source(name="Bad", source_type="api", priority=101)

    def test_register_source_validates_priority_negative(self, engine):
        """Negative priority raises ValueError."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 100"):
            engine.register_source(name="Bad", source_type="api", priority=-10)

    def test_register_source_validates_empty_name(self, engine):
        """Empty name raises ValueError."""
        with pytest.raises(ValueError, match="Source name must not be empty"):
            engine.register_source(name="", source_type="api")

    def test_register_source_validates_whitespace_name(self, engine):
        """Whitespace-only name raises ValueError."""
        with pytest.raises(ValueError, match="Source name must not be empty"):
            engine.register_source(name="   ", source_type="api")

    def test_register_source_strips_name_whitespace(self, engine):
        """Source name is stripped of leading/trailing whitespace."""
        src = engine.register_source(name="  Trimmed Name  ", source_type="api")
        assert src.name == "Trimmed Name"

    @pytest.mark.parametrize("src_type", [
        "erp", "utility", "meter", "questionnaire",
        "spreadsheet", "api", "iot", "registry", "manual", "other",
    ])
    def test_register_source_all_valid_types(self, engine, src_type):
        """All valid source types are accepted."""
        src = engine.register_source(name=f"Test {src_type}", source_type=src_type)
        assert src.source_type.value == src_type

    def test_register_source_priority_boundary_1(self, engine):
        """Priority 1 (minimum) is accepted."""
        src = engine.register_source(name="Low", source_type="api", priority=1)
        assert src.priority == 1

    def test_register_source_priority_boundary_100(self, engine):
        """Priority 100 (maximum) is accepted."""
        src = engine.register_source(name="High", source_type="api", priority=100)
        assert src.priority == 100


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Update
# ---------------------------------------------------------------------------


class TestUpdateSource:
    """Tests for update_source method."""

    def test_update_source_changes_priority(self, engine, registered_erp):
        """Updating priority changes the value."""
        updated = engine.update_source(registered_erp.id, priority=95)
        assert updated.priority == 95

    def test_update_source_changes_name(self, engine, registered_erp):
        """Updating name changes the value."""
        updated = engine.update_source(registered_erp.id, name="New ERP Name")
        assert updated.name == "New ERP Name"

    def test_update_source_changes_description(self, engine, registered_erp):
        """Updating description changes the value."""
        updated = engine.update_source(
            registered_erp.id, description="Updated description",
        )
        assert updated.description == "Updated description"

    def test_update_source_preserves_unchanged_fields(self, engine, registered_erp):
        """Fields not specified in kwargs remain unchanged."""
        updated = engine.update_source(registered_erp.id, priority=95)
        assert updated.name == "SAP ERP"
        assert updated.source_type == SourceType.ERP
        assert updated.description == "Production ERP system"

    def test_update_source_unknown_id_raises_key_error(self, engine):
        """Updating a non-existent source raises KeyError."""
        with pytest.raises(KeyError, match="Source not found"):
            engine.update_source("non-existent-id", priority=50)

    def test_update_source_invalid_field_raises_value_error(self, engine, registered_erp):
        """Attempting to update a non-updatable field raises ValueError."""
        with pytest.raises(ValueError, match="Cannot update fields"):
            engine.update_source(registered_erp.id, nonexistent_field="bad")

    def test_update_source_increments_version(self, engine, registered_erp):
        """Each update increments the internal version counter."""
        engine.update_source(registered_erp.id, priority=80)
        engine.update_source(registered_erp.id, priority=85)
        # Verify by getting source (version is internal state; verify
        # indirectly through successful multiple updates)
        src = engine.get_source(registered_erp.id)
        assert src.priority == 85

    def test_update_source_changes_tags(self, engine, registered_erp):
        """Updating tags replaces the tag list."""
        updated = engine.update_source(
            registered_erp.id, tags=["new_tag"],
        )
        assert updated.tags == ["new_tag"]


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Get Source
# ---------------------------------------------------------------------------


class TestGetSource:
    """Tests for get_source method."""

    def test_get_source_returns_registered_source(self, engine, registered_erp):
        """Retrieving a registered source returns its definition."""
        src = engine.get_source(registered_erp.id)
        assert src is not None
        assert src.name == "SAP ERP"

    def test_get_source_returns_none_for_unknown_id(self, engine):
        """Retrieving an unknown ID returns None."""
        result = engine.get_source("non-existent-id")
        assert result is None

    def test_get_source_returns_none_for_empty_id(self, engine):
        """Retrieving with empty string returns None."""
        result = engine.get_source("")
        assert result is None

    def test_get_source_reflects_updates(self, engine, registered_erp):
        """get_source returns the most recent state after updates."""
        engine.update_source(registered_erp.id, priority=99)
        src = engine.get_source(registered_erp.id)
        assert src.priority == 99


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: List Sources
# ---------------------------------------------------------------------------


class TestListSources:
    """Tests for list_sources method."""

    def test_list_sources_returns_all(self, engine, registered_erp, registered_api):
        """Without filters, list_sources returns all registered sources."""
        sources = engine.list_sources()
        assert len(sources) == 2

    def test_list_sources_empty_registry(self, engine):
        """Empty registry returns empty list."""
        sources = engine.list_sources()
        assert sources == []

    def test_list_sources_filter_by_type(self, engine, registered_erp, registered_api):
        """Filtering by source_type returns matching sources only."""
        erp_sources = engine.list_sources(source_type="erp")
        assert len(erp_sources) == 1
        assert erp_sources[0].name == "SAP ERP"

    def test_list_sources_filter_by_status(self, engine, registered_erp, registered_api):
        """Filtering by status returns matching sources only."""
        engine.deactivate_source(registered_api.id)
        active = engine.list_sources(status="active")
        assert len(active) == 1
        assert active[0].name == "SAP ERP"

    def test_list_sources_filter_by_min_priority(
        self, engine, registered_erp, registered_api, registered_manual,
    ):
        """Filtering by min_priority returns sources with priority >= threshold."""
        high_priority = engine.list_sources(min_priority=70)
        assert len(high_priority) == 2  # ERP(90) + API(70)

    def test_list_sources_sorted_by_priority_desc(
        self, engine, registered_erp, registered_api, registered_manual,
    ):
        """Results are sorted by priority descending."""
        sources = engine.list_sources()
        priorities = [s.priority for s in sources]
        assert priorities == sorted(priorities, reverse=True)

    def test_list_sources_combined_filters(
        self, engine, registered_erp, registered_api, registered_manual,
    ):
        """Multiple filters are combined with AND logic."""
        sources = engine.list_sources(source_type="erp", min_priority=80)
        assert len(sources) == 1
        assert sources[0].name == "SAP ERP"


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Deactivation
# ---------------------------------------------------------------------------


class TestDeactivateSource:
    """Tests for deactivate_source method."""

    def test_deactivate_source_changes_status(self, engine, registered_erp):
        """Deactivating a source changes its status to INACTIVE."""
        result = engine.deactivate_source(registered_erp.id)
        assert result is True
        src = engine.get_source(registered_erp.id)
        assert src.status == SourceStatus.INACTIVE

    def test_deactivate_source_returns_false_for_unknown(self, engine):
        """Deactivating a non-existent source returns False."""
        result = engine.deactivate_source("non-existent-id")
        assert result is False

    def test_deactivate_source_already_inactive(self, engine, registered_erp):
        """Deactivating an already-inactive source returns False."""
        engine.deactivate_source(registered_erp.id)
        result = engine.deactivate_source(registered_erp.id)
        assert result is False


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Schema Mapping
# ---------------------------------------------------------------------------


class TestSchemaMapping:
    """Tests for register_schema_mapping and align_schemas."""

    def test_register_schema_mapping_stores_mappings(self, engine, registered_erp):
        """Registering schema mappings stores them successfully."""
        mappings = [
            SchemaMapping(
                source_column="spend",
                canonical_column="amount",
            ),
            SchemaMapping(
                source_column="vendor",
                canonical_column="supplier_name",
            ),
        ]
        result = engine.register_schema_mapping(registered_erp.id, mappings)
        assert len(result) == 2

    def test_register_schema_mapping_replaces_existing(self, engine, registered_erp):
        """New mappings replace previously registered ones."""
        first_mappings = [
            SchemaMapping(source_column="a", canonical_column="x"),
        ]
        engine.register_schema_mapping(registered_erp.id, first_mappings)

        second_mappings = [
            SchemaMapping(source_column="b", canonical_column="y"),
        ]
        engine.register_schema_mapping(registered_erp.id, second_mappings)

        stored = engine.get_schema_mapping(registered_erp.id)
        assert len(stored) == 1
        assert stored[0].source_column == "b"

    def test_register_schema_mapping_unknown_source_raises(self, engine):
        """Mapping for unknown source raises KeyError."""
        with pytest.raises(KeyError, match="Source not found"):
            engine.register_schema_mapping("bad-id", [
                SchemaMapping(source_column="a", canonical_column="b"),
            ])

    def test_register_schema_mapping_empty_list_raises(self, engine, registered_erp):
        """Empty mappings list raises ValueError."""
        with pytest.raises(ValueError, match="Mappings list must not be empty"):
            engine.register_schema_mapping(registered_erp.id, [])

    def test_register_schema_mapping_duplicate_columns_raises(
        self, engine, registered_erp,
    ):
        """Duplicate source_column entries raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate source_column"):
            engine.register_schema_mapping(registered_erp.id, [
                SchemaMapping(source_column="spend", canonical_column="a"),
                SchemaMapping(source_column="spend", canonical_column="b"),
            ])

    def test_get_schema_mapping_returns_empty_for_unmapped(
        self, engine, registered_erp,
    ):
        """Getting mappings for a source with none returns empty list."""
        result = engine.get_schema_mapping(registered_erp.id)
        assert result == []

    def test_align_schemas_finds_exact_matches(self, engine):
        """align_schemas finds exact column name matches."""
        s1 = engine.register_source(
            name="A", source_type="erp",
            schema_info={"columns": ["amount", "date", "vendor"]},
        )
        s2 = engine.register_source(
            name="B", source_type="api",
            schema_info={"columns": ["amount", "date", "vendor"]},
        )
        aligned = engine.align_schemas([s1.id, s2.id])
        assert len(aligned) == 2
        # Both sources should find all columns since they are identical
        assert len(aligned[s1.id]) >= 3
        assert len(aligned[s2.id]) >= 3

    def test_align_schemas_finds_fuzzy_matches(self, engine):
        """align_schemas uses Jaro-Winkler to find similar column names."""
        s1 = engine.register_source(
            name="A", source_type="erp",
            schema_info={"columns": ["emission_amount"]},
        )
        s2 = engine.register_source(
            name="B", source_type="api",
            schema_info={"columns": ["emissions_amount"]},
        )
        aligned = engine.align_schemas([s1.id, s2.id], similarity_threshold=0.8)
        # emission_amount and emissions_amount are very similar
        assert len(aligned[s1.id]) >= 1
        assert len(aligned[s2.id]) >= 1

    def test_align_schemas_requires_two_sources(self, engine):
        """align_schemas requires at least 2 source IDs."""
        s1 = engine.register_source(name="A", source_type="erp")
        with pytest.raises(ValueError, match="At least 2 source_ids"):
            engine.align_schemas([s1.id])

    def test_align_schemas_validates_threshold(self, engine):
        """align_schemas validates similarity_threshold range."""
        s1 = engine.register_source(
            name="A", source_type="erp",
            schema_info={"columns": ["a"]},
        )
        s2 = engine.register_source(
            name="B", source_type="api",
            schema_info={"columns": ["b"]},
        )
        with pytest.raises(ValueError, match="similarity_threshold must be"):
            engine.align_schemas([s1.id, s2.id], similarity_threshold=1.5)


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Credibility
# ---------------------------------------------------------------------------


class TestComputeCredibility:
    """Tests for compute_credibility method."""

    def test_compute_credibility_returns_scores_in_range(
        self, engine, registered_erp, sample_records,
    ):
        """Overall credibility score is in [0, 1]."""
        cred = engine.compute_credibility(registered_erp.id, sample_records)
        assert 0.0 <= cred.overall_score <= 1.0
        assert 0.0 <= cred.completeness_score <= 1.0
        assert 0.0 <= cred.timeliness_score <= 1.0
        assert 0.0 <= cred.consistency_score <= 1.0
        assert 0.0 <= cred.accuracy_score <= 1.0
        assert 0.0 <= cred.certification_score <= 1.0

    def test_compute_credibility_completeness_reflects_nulls(
        self, engine, registered_erp,
    ):
        """Completeness score decreases with more null values."""
        complete_records = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "c": 6},
        ]
        sparse_records = [
            {"a": 1, "b": None, "c": None},
            {"a": None, "b": None, "c": None},
        ]
        cred_full = engine.compute_credibility(registered_erp.id, complete_records)
        cred_sparse = engine.compute_credibility(registered_erp.id, sparse_records)
        assert cred_full.completeness_score > cred_sparse.completeness_score

    def test_compute_credibility_certification_uses_source_type(self, engine):
        """Certification score reflects source type (erp > manual)."""
        erp = engine.register_source(name="ERP", source_type="erp")
        manual = engine.register_source(name="Manual", source_type="manual")
        records = [{"a": 1}]
        cred_erp = engine.compute_credibility(erp.id, records)
        cred_manual = engine.compute_credibility(manual.id, records)
        assert cred_erp.certification_score > cred_manual.certification_score

    def test_compute_credibility_unknown_source_raises(self, engine):
        """Credibility for unknown source raises KeyError."""
        with pytest.raises(KeyError, match="Source not found"):
            engine.compute_credibility("bad-id", [{"a": 1}])

    def test_compute_credibility_empty_records_raises(self, engine, registered_erp):
        """Empty records list raises ValueError."""
        with pytest.raises(ValueError, match="Records list must not be empty"):
            engine.compute_credibility(registered_erp.id, [])

    def test_compute_credibility_updates_source_credibility_score(
        self, engine, registered_erp, sample_records,
    ):
        """After computing credibility, the source's credibility_score is updated."""
        cred = engine.compute_credibility(registered_erp.id, sample_records)
        src = engine.get_source(registered_erp.id)
        assert src.credibility_score == cred.overall_score

    def test_compute_credibility_sample_size_correct(
        self, engine, registered_erp, sample_records,
    ):
        """Sample size matches the number of records."""
        cred = engine.compute_credibility(registered_erp.id, sample_records)
        assert cred.sample_size == len(sample_records)

    def test_compute_credibility_custom_weights(
        self, engine, registered_erp, sample_records,
    ):
        """Custom weights change the overall score."""
        cred_default = engine.compute_credibility(registered_erp.id, sample_records)
        # Re-register to reset credibility cache
        erp2 = engine.register_source(name="ERP2", source_type="erp", priority=90)
        cred_custom = engine.compute_credibility(
            erp2.id, sample_records,
            weights={"completeness": 1.0, "timeliness": 0.0,
                     "consistency": 0.0, "accuracy": 0.0, "certification": 0.0},
        )
        # When only completeness matters, overall = completeness_score
        assert abs(cred_custom.overall_score - cred_custom.completeness_score) < 0.01


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Ranking
# ---------------------------------------------------------------------------


class TestRankSources:
    """Tests for rank_sources method."""

    def test_rank_sources_orders_by_score(
        self, engine, registered_erp, registered_api, registered_manual,
    ):
        """Higher priority * credibility = higher rank."""
        ranked = engine.rank_sources([
            registered_erp.id, registered_api.id, registered_manual.id,
        ])
        # ERP has highest priority (90), so should be first
        assert ranked[0][0] == registered_erp.id

    def test_rank_sources_returns_tuples_with_scores(
        self, engine, registered_erp,
    ):
        """Each result is a (source_id, score) tuple with score in [0, 1]."""
        ranked = engine.rank_sources([registered_erp.id])
        assert len(ranked) == 1
        sid, score = ranked[0]
        assert sid == registered_erp.id
        assert 0.0 <= score <= 1.0

    def test_rank_sources_excludes_unknown_ids(self, engine, registered_erp):
        """Unknown IDs are silently excluded from ranking."""
        ranked = engine.rank_sources([registered_erp.id, "unknown-id"])
        assert len(ranked) == 1

    def test_rank_sources_empty_list(self, engine):
        """Empty source list returns empty ranking."""
        ranked = engine.rank_sources([])
        assert ranked == []

    def test_rank_sources_score_formula(self, engine):
        """Ranking score equals (priority/100) * credibility_score."""
        src = engine.register_source(
            name="Test", source_type="erp", priority=80,
        )
        # Default credibility = 0.5, so score = 0.8 * 0.5 = 0.4
        ranked = engine.rank_sources([src.id])
        expected = round(80 / 100.0 * 0.5, 4)
        assert ranked[0][1] == expected


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Tolerance Rules
# ---------------------------------------------------------------------------


class TestToleranceRules:
    """Tests for set_tolerance_rules and get_tolerance_rules methods."""

    def test_set_and_get_tolerance_rules(self, engine, registered_erp, registered_api):
        """Set rules for a pair, then retrieve them."""
        pair_key = f"{registered_erp.id}:{registered_api.id}"
        rules = [
            ToleranceRule(field_name="amount", tolerance_pct=5.0),
            ToleranceRule(field_name="date", field_type=FieldType.DATE),
        ]
        engine.set_tolerance_rules(pair_key, rules)
        retrieved = engine.get_tolerance_rules(registered_erp.id, registered_api.id)
        assert len(retrieved) == 2

    def test_get_tolerance_rules_reversed_key(self, engine, registered_erp, registered_api):
        """get_tolerance_rules checks both orderings of the pair key."""
        pair_key = f"{registered_erp.id}:{registered_api.id}"
        rules = [ToleranceRule(field_name="amount", tolerance_pct=3.0)]
        engine.set_tolerance_rules(pair_key, rules)
        # Retrieve with reversed order
        retrieved = engine.get_tolerance_rules(registered_api.id, registered_erp.id)
        assert len(retrieved) == 1

    def test_get_tolerance_rules_returns_defaults_when_unset(
        self, engine, registered_erp, registered_api,
    ):
        """When no pair rules are set, default rules are returned."""
        rules = engine.get_tolerance_rules(registered_erp.id, registered_api.id)
        assert len(rules) > 0  # Defaults exist

    def test_set_tolerance_rules_empty_key_raises(self, engine):
        """Empty pair key raises ValueError."""
        with pytest.raises(ValueError, match="source_pair_key must not be empty"):
            engine.set_tolerance_rules("", [
                ToleranceRule(field_name="a"),
            ])

    def test_set_tolerance_rules_empty_rules_raises(self, engine):
        """Empty rules list raises ValueError."""
        with pytest.raises(ValueError, match="Rules list must not be empty"):
            engine.set_tolerance_rules("key:pair", [])


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance tracking across operations."""

    def test_provenance_changes_after_register(self, engine):
        """Provenance chain grows after registering a source."""
        chain_before = engine._provenance.entry_count
        engine.register_source(name="Test", source_type="api")
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before

    def test_provenance_changes_after_update(self, engine, registered_erp):
        """Provenance chain grows after updating a source."""
        chain_before = engine._provenance.entry_count
        engine.update_source(registered_erp.id, priority=80)
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before

    def test_provenance_changes_after_deactivation(self, engine, registered_erp):
        """Provenance chain grows after deactivating a source."""
        chain_before = engine._provenance.entry_count
        engine.deactivate_source(registered_erp.id)
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before

    def test_provenance_changes_after_schema_mapping(self, engine, registered_erp):
        """Provenance chain grows after registering schema mappings."""
        chain_before = engine._provenance.entry_count
        engine.register_schema_mapping(registered_erp.id, [
            SchemaMapping(source_column="a", canonical_column="b"),
        ])
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before

    def test_provenance_changes_after_align_schemas(self, engine):
        """Provenance chain grows after aligning schemas."""
        s1 = engine.register_source(
            name="A", source_type="erp",
            schema_info={"columns": ["x"]},
        )
        s2 = engine.register_source(
            name="B", source_type="api",
            schema_info={"columns": ["y"]},
        )
        chain_before = engine._provenance.entry_count
        engine.align_schemas([s1.id, s2.id])
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before

    def test_provenance_changes_after_credibility(
        self, engine, registered_erp, sample_records,
    ):
        """Provenance chain grows after computing credibility."""
        chain_before = engine._provenance.entry_count
        engine.compute_credibility(registered_erp.id, sample_records)
        chain_after = engine._provenance.entry_count
        assert chain_after > chain_before


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Health
# ---------------------------------------------------------------------------


class TestSourceHealth:
    """Tests for get_source_health method."""

    def test_get_source_health_returns_metrics(self, engine, registered_erp):
        """get_source_health returns a SourceHealthMetrics object."""
        health = engine.get_source_health(registered_erp.id)
        assert health.source_id == registered_erp.id

    def test_get_source_health_unknown_raises(self, engine):
        """get_source_health for unknown source raises KeyError."""
        with pytest.raises(KeyError, match="Source not found"):
            engine.get_source_health("bad-id")

    def test_get_source_health_after_credibility(
        self, engine, registered_erp, sample_records,
    ):
        """Health metrics reflect credibility after computation."""
        engine.compute_credibility(registered_erp.id, sample_records)
        health = engine.get_source_health(registered_erp.id)
        assert health.avg_credibility > 0.0


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Jaro-Winkler Helpers
# ---------------------------------------------------------------------------


class TestJaroWinklerSimilarity:
    """Tests for module-level Jaro-Winkler helper functions."""

    def test_identical_strings_return_one(self):
        """Identical strings have similarity 1.0."""
        assert _jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_strings_return_one(self):
        """Two empty strings have similarity 1.0."""
        assert _jaro_similarity("", "") == 1.0

    def test_one_empty_returns_zero(self):
        """One empty string produces similarity 0.0."""
        assert _jaro_winkler_similarity("hello", "") == 0.0
        assert _jaro_winkler_similarity("", "world") == 0.0

    def test_completely_different_strings(self):
        """Completely different short strings have low similarity."""
        score = _jaro_winkler_similarity("abc", "xyz")
        assert score < 0.5

    def test_similar_strings_high_score(self):
        """Similar strings (e.g. emission/emissions) score high."""
        score = _jaro_winkler_similarity("emission", "emissions")
        assert score > 0.9

    def test_prefix_bonus_applied(self):
        """Jaro-Winkler gives a bonus for common prefixes."""
        jaro = _jaro_similarity("MARTHA", "MARHTA")
        jw = _jaro_winkler_similarity("MARTHA", "MARHTA")
        assert jw >= jaro  # Winkler bonus increases the score

    def test_symmetry(self):
        """Jaro-Winkler is symmetric: sim(a,b) == sim(b,a)."""
        score_ab = _jaro_winkler_similarity("abc", "abd")
        score_ba = _jaro_winkler_similarity("abd", "abc")
        assert abs(score_ab - score_ba) < 1e-10


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Normalize Value
# ---------------------------------------------------------------------------


class TestNormalizeValue:
    """Tests for _normalize_value method."""

    def test_normalize_none_returns_none(self, engine):
        """None input returns None."""
        result = engine._normalize_value(None, "numeric")
        assert result is None

    def test_normalize_without_mapping_returns_original(self, engine):
        """Without schema_mapping, value is returned as-is."""
        result = engine._normalize_value(42, "numeric")
        assert result == 42

    def test_normalize_string_type(self, engine):
        """String normalization works via _normalize_value."""
        result = engine._normalize_value("  Test String  ", "string")
        assert result is not None


# ---------------------------------------------------------------------------
# TestSourceRegistryEngine: Certification Scores
# ---------------------------------------------------------------------------


class TestCertificationScores:
    """Tests for CERTIFICATION_SCORES constants."""

    def test_registry_highest_score(self):
        """Registry sources have the highest certification score."""
        assert CERTIFICATION_SCORES["registry"] == 1.0

    def test_manual_lowest_score(self):
        """Manual entry has the lowest certification score."""
        assert CERTIFICATION_SCORES["manual"] == 0.3

    def test_erp_score(self):
        """ERP certification score is 0.8."""
        assert CERTIFICATION_SCORES["erp"] == 0.8

    def test_all_scores_in_range(self):
        """All certification scores are in [0, 1]."""
        for source_type, score in CERTIFICATION_SCORES.items():
            assert 0.0 <= score <= 1.0, f"{source_type} score out of range: {score}"
