# -*- coding: utf-8 -*-
"""
Unit Tests for Data Lineage Tracker Models - AGENT-DATA-018

Tests all enums (14), SDK data models (16), request models (8),
constants, and Layer 1 re-exports defined in the models module.

80+ test cases covering:
  - Enum member counts and values
  - Model creation with required and optional fields
  - Model serialization via model_dump()
  - Field validation (non-empty strings, ranges 0-1, ge=0)
  - Default values
  - Pydantic extra="forbid" enforcement
  - Constants and Layer 1 re-export stubs

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pydantic
import pytest

# ---------------------------------------------------------------------------
# Restore strict model_config for model validation tests
# ---------------------------------------------------------------------------


def _restore_strict_configs():
    """Restore extra='forbid' on all Pydantic models for strict testing."""
    from greenlang.data_lineage_tracker import models as dlt_models

    for name in dir(dlt_models):
        obj = getattr(dlt_models, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, pydantic.BaseModel)
            and obj is not pydantic.BaseModel
        ):
            cfg = getattr(obj, "model_config", {})
            if isinstance(cfg, dict):
                obj.model_config = {**cfg, "extra": "forbid"}
                obj.model_rebuild(force=True)


_restore_strict_configs()

# ---------------------------------------------------------------------------
# Imports from the models module
# ---------------------------------------------------------------------------

from greenlang.data_lineage_tracker.models import (
    # Constants
    VERSION,
    MAX_ASSETS_PER_NAMESPACE,
    MAX_TRAVERSAL_DEPTH,
    MAX_EDGES_PER_ASSET,
    DEFAULT_EDGE_CONFIDENCE,
    MIN_IMPACT_CONFIDENCE,
    MAX_CHANGE_EVENTS_PER_SNAPSHOT,
    DEFAULT_PIPELINE_BATCH_SIZE,
    SCORE_TIER_THRESHOLDS,
    SUPPORTED_REPORT_FORMATS,
    IMPACT_SEVERITY_ORDER,
    # Enumerations (14)
    AssetType,
    AssetClassification,
    AssetStatus,
    TransformationType,
    EdgeType,
    TraversalDirection,
    ImpactSeverity,
    ValidationResult,
    ReportType,
    ReportFormat,
    ChangeType,
    ChangeSeverity,
    ScoreTier,
    TransformationLogicType,
    # SDK data models (16)
    DataAsset,
    TransformationEvent,
    LineageEdge,
    GraphSnapshot,
    ImpactAnalysisResult,
    ValidationReport,
    LineageReport,
    ChangeEvent,
    QualityScore,
    AuditEntry,
    GraphNode,
    GraphEdgeView,
    SubgraphResult,
    LineageChain,
    LineageStatistics,
    PipelineResult,
    # Request models (8)
    RegisterAssetRequest,
    UpdateAssetRequest,
    RecordTransformationRequest,
    CreateEdgeRequest,
    RunImpactAnalysisRequest,
    RunValidationRequest,
    GenerateReportRequest,
    RunPipelineRequest,
    # Layer 1 re-exports
    QualityDimension,
)


# ============================================================================
# TestConstants
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_version_is_string(self):
        """VERSION is a non-empty string."""
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0

    def test_max_assets_per_namespace(self):
        """MAX_ASSETS_PER_NAMESPACE is a positive integer."""
        assert isinstance(MAX_ASSETS_PER_NAMESPACE, int)
        assert MAX_ASSETS_PER_NAMESPACE > 0

    def test_max_traversal_depth(self):
        """MAX_TRAVERSAL_DEPTH is 50."""
        assert MAX_TRAVERSAL_DEPTH == 50

    def test_max_edges_per_asset(self):
        """MAX_EDGES_PER_ASSET is a positive integer."""
        assert isinstance(MAX_EDGES_PER_ASSET, int)
        assert MAX_EDGES_PER_ASSET > 0

    def test_default_edge_confidence(self):
        """DEFAULT_EDGE_CONFIDENCE is 1.0."""
        assert DEFAULT_EDGE_CONFIDENCE == 1.0

    def test_min_impact_confidence(self):
        """MIN_IMPACT_CONFIDENCE is 0.1."""
        assert MIN_IMPACT_CONFIDENCE == pytest.approx(0.1)

    def test_score_tier_thresholds(self):
        """SCORE_TIER_THRESHOLDS has 5 tiers in descending order."""
        assert isinstance(SCORE_TIER_THRESHOLDS, dict)
        assert len(SCORE_TIER_THRESHOLDS) == 5
        assert "excellent" in SCORE_TIER_THRESHOLDS
        assert "critical" in SCORE_TIER_THRESHOLDS

    def test_supported_report_formats(self):
        """SUPPORTED_REPORT_FORMATS contains expected formats."""
        assert "json" in SUPPORTED_REPORT_FORMATS
        assert "mermaid" in SUPPORTED_REPORT_FORMATS

    def test_impact_severity_order(self):
        """IMPACT_SEVERITY_ORDER is ordered low to critical."""
        assert IMPACT_SEVERITY_ORDER == ("low", "medium", "high", "critical")


# ============================================================================
# TestEnumerations
# ============================================================================


class TestAssetTypeEnum:
    """Tests for AssetType enumeration."""

    def test_member_count(self):
        """AssetType has 7 members."""
        assert len(AssetType) == 7

    @pytest.mark.parametrize("member,value", [
        ("DATASET", "dataset"),
        ("FIELD", "field"),
        ("AGENT", "agent"),
        ("PIPELINE", "pipeline"),
        ("REPORT", "report"),
        ("METRIC", "metric"),
        ("EXTERNAL_SOURCE", "external_source"),
    ])
    def test_member_values(self, member, value):
        """Each AssetType member has the correct value."""
        assert AssetType[member].value == value

    def test_is_str_enum(self):
        """AssetType members are strings."""
        assert isinstance(AssetType.DATASET, str)
        assert AssetType.DATASET == "dataset"


class TestAssetClassificationEnum:
    """Tests for AssetClassification enumeration."""

    def test_member_count(self):
        """AssetClassification has 4 members."""
        assert len(AssetClassification) == 4

    @pytest.mark.parametrize("member,value", [
        ("PUBLIC", "public"),
        ("INTERNAL", "internal"),
        ("CONFIDENTIAL", "confidential"),
        ("RESTRICTED", "restricted"),
    ])
    def test_member_values(self, member, value):
        """Each AssetClassification member has the correct value."""
        assert AssetClassification[member].value == value


class TestAssetStatusEnum:
    """Tests for AssetStatus enumeration."""

    def test_member_count(self):
        """AssetStatus has 3 members."""
        assert len(AssetStatus) == 3

    @pytest.mark.parametrize("member,value", [
        ("ACTIVE", "active"),
        ("DEPRECATED", "deprecated"),
        ("ARCHIVED", "archived"),
    ])
    def test_member_values(self, member, value):
        """Each AssetStatus member has the correct value."""
        assert AssetStatus[member].value == value


class TestTransformationTypeEnum:
    """Tests for TransformationType enumeration."""

    def test_member_count(self):
        """TransformationType has 12 members."""
        assert len(TransformationType) == 12

    @pytest.mark.parametrize("value", [
        "filter", "aggregate", "join", "calculate", "impute",
        "deduplicate", "enrich", "merge", "split", "validate",
        "normalize", "classify",
    ])
    def test_member_values(self, value):
        """Each expected value exists in TransformationType."""
        assert TransformationType(value) is not None


class TestEdgeTypeEnum:
    """Tests for EdgeType enumeration."""

    def test_member_count(self):
        """EdgeType has 2 members."""
        assert len(EdgeType) == 2

    def test_dataset_level(self):
        """EdgeType.DATASET_LEVEL has value 'dataset_level'."""
        assert EdgeType.DATASET_LEVEL.value == "dataset_level"

    def test_column_level(self):
        """EdgeType.COLUMN_LEVEL has value 'column_level'."""
        assert EdgeType.COLUMN_LEVEL.value == "column_level"


class TestTraversalDirectionEnum:
    """Tests for TraversalDirection enumeration."""

    def test_member_count(self):
        """TraversalDirection has 2 members."""
        assert len(TraversalDirection) == 2

    def test_forward(self):
        """TraversalDirection.FORWARD has value 'forward'."""
        assert TraversalDirection.FORWARD.value == "forward"

    def test_backward(self):
        """TraversalDirection.BACKWARD has value 'backward'."""
        assert TraversalDirection.BACKWARD.value == "backward"


class TestImpactSeverityEnum:
    """Tests for ImpactSeverity enumeration."""

    def test_member_count(self):
        """ImpactSeverity has 4 members."""
        assert len(ImpactSeverity) == 4

    @pytest.mark.parametrize("member,value", [
        ("CRITICAL", "critical"),
        ("HIGH", "high"),
        ("MEDIUM", "medium"),
        ("LOW", "low"),
    ])
    def test_member_values(self, member, value):
        """Each ImpactSeverity member has the correct value."""
        assert ImpactSeverity[member].value == value


class TestValidationResultEnum:
    """Tests for ValidationResult enumeration."""

    def test_member_count(self):
        """ValidationResult has 3 members."""
        assert len(ValidationResult) == 3

    def test_pass_result_value(self):
        """ValidationResult.PASS_RESULT has value 'pass'."""
        assert ValidationResult.PASS_RESULT.value == "pass"

    def test_warn_value(self):
        """ValidationResult.WARN has value 'warn'."""
        assert ValidationResult.WARN.value == "warn"

    def test_fail_value(self):
        """ValidationResult.FAIL has value 'fail'."""
        assert ValidationResult.FAIL.value == "fail"


class TestReportTypeEnum:
    """Tests for ReportType enumeration."""

    def test_member_count(self):
        """ReportType has 5 members."""
        assert len(ReportType) == 5

    @pytest.mark.parametrize("value", [
        "csrd_esrs", "ghg_protocol", "soc2", "custom", "visualization",
    ])
    def test_member_values(self, value):
        """Each expected value exists in ReportType."""
        assert ReportType(value) is not None


class TestReportFormatEnum:
    """Tests for ReportFormat enumeration."""

    def test_member_count(self):
        """ReportFormat has 7 members."""
        assert len(ReportFormat) == 7

    @pytest.mark.parametrize("value", [
        "mermaid", "dot", "json", "d3", "text", "html", "pdf",
    ])
    def test_member_values(self, value):
        """Each expected value exists in ReportFormat."""
        assert ReportFormat(value) is not None


class TestChangeTypeEnum:
    """Tests for ChangeType enumeration."""

    def test_member_count(self):
        """ChangeType has 5 members."""
        assert len(ChangeType) == 5

    @pytest.mark.parametrize("value", [
        "node_added", "node_removed", "edge_added", "edge_removed",
        "topology_changed",
    ])
    def test_member_values(self, value):
        """Each expected value exists in ChangeType."""
        assert ChangeType(value) is not None


class TestChangeSeverityEnum:
    """Tests for ChangeSeverity enumeration."""

    def test_member_count(self):
        """ChangeSeverity has 4 members."""
        assert len(ChangeSeverity) == 4

    @pytest.mark.parametrize("value", ["low", "medium", "high", "critical"])
    def test_member_values(self, value):
        """Each expected value exists in ChangeSeverity."""
        assert ChangeSeverity(value) is not None


class TestScoreTierEnum:
    """Tests for ScoreTier enumeration."""

    def test_member_count(self):
        """ScoreTier has 5 members."""
        assert len(ScoreTier) == 5

    @pytest.mark.parametrize("value", ["excellent", "good", "fair", "poor", "critical"])
    def test_member_values(self, value):
        """Each expected value exists in ScoreTier."""
        assert ScoreTier(value) is not None


class TestTransformationLogicTypeEnum:
    """Tests for TransformationLogicType enumeration."""

    def test_member_count(self):
        """TransformationLogicType has 8 members."""
        assert len(TransformationLogicType) == 8

    @pytest.mark.parametrize("value", [
        "rename", "cast", "aggregate_func", "compute", "conditional",
        "lookup", "merge_fields", "split_field",
    ])
    def test_member_values(self, value):
        """Each expected value exists in TransformationLogicType."""
        assert TransformationLogicType(value) is not None


# ============================================================================
# TestSDKDataModels
# ============================================================================


class TestDataAssetModel:
    """Tests for DataAsset SDK model."""

    def test_creation_with_required_fields(self):
        """DataAsset can be created with only required fields."""
        asset = DataAsset(
            qualified_name="emissions.scope3.spend",
            asset_type=AssetType.DATASET,
        )
        assert asset.qualified_name == "emissions.scope3.spend"
        assert asset.asset_type == AssetType.DATASET

    def test_default_values(self):
        """DataAsset defaults are set correctly."""
        asset = DataAsset(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        assert asset.display_name == ""
        assert asset.owner == ""
        assert asset.tags == {}
        assert asset.classification == AssetClassification.INTERNAL
        assert asset.status == AssetStatus.ACTIVE
        assert asset.schema_ref is None
        assert asset.description == ""
        assert asset.metadata == {}

    def test_auto_generated_id(self):
        """DataAsset generates a valid UUID for id."""
        asset = DataAsset(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        uuid.UUID(asset.id)  # Validates UUID format

    def test_timestamps_are_utc(self):
        """DataAsset created_at and updated_at are UTC."""
        asset = DataAsset(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        assert asset.created_at.tzinfo is not None
        assert asset.updated_at.tzinfo is not None

    def test_serialization(self):
        """DataAsset can be serialized to dict."""
        asset = DataAsset(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        d = asset.model_dump()
        assert isinstance(d, dict)
        assert d["qualified_name"] == "test.asset"

    def test_empty_qualified_name_raises(self):
        """DataAsset rejects empty qualified_name."""
        with pytest.raises(pydantic.ValidationError):
            DataAsset(qualified_name="", asset_type=AssetType.DATASET)

    def test_whitespace_qualified_name_raises(self):
        """DataAsset rejects whitespace-only qualified_name."""
        with pytest.raises(pydantic.ValidationError):
            DataAsset(qualified_name="   ", asset_type=AssetType.DATASET)


class TestTransformationEventModel:
    """Tests for TransformationEvent SDK model."""

    def test_creation(self):
        """TransformationEvent can be created with required fields."""
        event = TransformationEvent(
            transformation_type=TransformationType.FILTER,
        )
        assert event.transformation_type == TransformationType.FILTER

    def test_defaults(self):
        """TransformationEvent defaults are set correctly."""
        event = TransformationEvent(
            transformation_type=TransformationType.JOIN,
        )
        assert event.agent_id == ""
        assert event.pipeline_id == ""
        assert event.source_assets == []
        assert event.target_assets == []
        assert event.records_in == 0
        assert event.records_out == 0
        assert event.records_filtered == 0
        assert event.records_error == 0
        assert event.duration_ms == 0.0

    def test_negative_records_in_raises(self):
        """TransformationEvent rejects negative records_in."""
        with pytest.raises(pydantic.ValidationError):
            TransformationEvent(
                transformation_type=TransformationType.FILTER,
                records_in=-1,
            )

    def test_serialization(self):
        """TransformationEvent can be serialized."""
        event = TransformationEvent(
            transformation_type=TransformationType.AGGREGATE,
            records_in=1000,
            records_out=100,
        )
        d = event.model_dump()
        assert d["records_in"] == 1000
        assert d["records_out"] == 100


class TestLineageEdgeModel:
    """Tests for LineageEdge SDK model."""

    def test_creation(self):
        """LineageEdge can be created with required fields."""
        edge = LineageEdge(
            source_asset_id="src-1",
            target_asset_id="tgt-1",
        )
        assert edge.source_asset_id == "src-1"
        assert edge.target_asset_id == "tgt-1"

    def test_defaults(self):
        """LineageEdge defaults are set correctly."""
        edge = LineageEdge(
            source_asset_id="src-1",
            target_asset_id="tgt-1",
        )
        assert edge.edge_type == EdgeType.DATASET_LEVEL
        assert edge.confidence == 1.0
        assert edge.source_field is None
        assert edge.target_field is None

    def test_empty_source_asset_id_raises(self):
        """LineageEdge rejects empty source_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            LineageEdge(source_asset_id="", target_asset_id="tgt-1")

    def test_empty_target_asset_id_raises(self):
        """LineageEdge rejects empty target_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            LineageEdge(source_asset_id="src-1", target_asset_id="")

    def test_confidence_below_zero_raises(self):
        """LineageEdge rejects confidence < 0."""
        with pytest.raises(pydantic.ValidationError):
            LineageEdge(
                source_asset_id="src-1",
                target_asset_id="tgt-1",
                confidence=-0.1,
            )

    def test_confidence_above_one_raises(self):
        """LineageEdge rejects confidence > 1."""
        with pytest.raises(pydantic.ValidationError):
            LineageEdge(
                source_asset_id="src-1",
                target_asset_id="tgt-1",
                confidence=1.1,
            )

    def test_column_level_edge(self):
        """LineageEdge supports column_level edge type with field names."""
        edge = LineageEdge(
            source_asset_id="src-1",
            target_asset_id="tgt-1",
            edge_type=EdgeType.COLUMN_LEVEL,
            source_field="amount",
            target_field="total_amount",
        )
        assert edge.edge_type == EdgeType.COLUMN_LEVEL
        assert edge.source_field == "amount"


class TestGraphSnapshotModel:
    """Tests for GraphSnapshot SDK model."""

    def test_creation(self):
        """GraphSnapshot can be created with defaults."""
        snapshot = GraphSnapshot()
        assert snapshot.node_count == 0
        assert snapshot.edge_count == 0
        assert snapshot.coverage_score == 0.0

    def test_coverage_score_range(self):
        """GraphSnapshot rejects coverage_score > 1."""
        with pytest.raises(pydantic.ValidationError):
            GraphSnapshot(coverage_score=1.5)


class TestImpactAnalysisResultModel:
    """Tests for ImpactAnalysisResult SDK model."""

    def test_creation(self):
        """ImpactAnalysisResult can be created with required fields."""
        result = ImpactAnalysisResult(root_asset_id="asset-1")
        assert result.root_asset_id == "asset-1"
        assert result.direction == TraversalDirection.FORWARD

    def test_empty_root_asset_id_raises(self):
        """ImpactAnalysisResult rejects empty root_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            ImpactAnalysisResult(root_asset_id="")

    def test_blast_radius_range(self):
        """ImpactAnalysisResult rejects blast_radius > 1."""
        with pytest.raises(pydantic.ValidationError):
            ImpactAnalysisResult(root_asset_id="a1", blast_radius=1.5)


class TestValidationReportModel:
    """Tests for ValidationReport SDK model."""

    def test_creation(self):
        """ValidationReport can be created with defaults."""
        report = ValidationReport()
        assert report.scope == "full"
        assert report.orphan_nodes == 0
        assert report.source_coverage == 0.0

    def test_source_coverage_range(self):
        """ValidationReport rejects source_coverage > 1."""
        with pytest.raises(pydantic.ValidationError):
            ValidationReport(source_coverage=1.5)

    def test_completeness_score_range(self):
        """ValidationReport rejects completeness_score > 1."""
        with pytest.raises(pydantic.ValidationError):
            ValidationReport(completeness_score=1.1)


class TestLineageReportModel:
    """Tests for LineageReport SDK model."""

    def test_creation(self):
        """LineageReport can be created with defaults."""
        report = LineageReport()
        assert report.report_type == ReportType.CUSTOM
        assert report.format == ReportFormat.JSON

    def test_serialization(self):
        """LineageReport can be serialized."""
        report = LineageReport(content="graph LR; A-->B")
        d = report.model_dump()
        assert d["content"] == "graph LR; A-->B"


class TestChangeEventModel:
    """Tests for ChangeEvent SDK model."""

    def test_creation(self):
        """ChangeEvent can be created with required fields."""
        event = ChangeEvent(
            previous_snapshot_id="snap-1",
            current_snapshot_id="snap-2",
            change_type=ChangeType.NODE_ADDED,
            entity_id="node-1",
        )
        assert event.change_type == ChangeType.NODE_ADDED

    def test_empty_previous_snapshot_id_raises(self):
        """ChangeEvent rejects empty previous_snapshot_id."""
        with pytest.raises(pydantic.ValidationError):
            ChangeEvent(
                previous_snapshot_id="",
                current_snapshot_id="snap-2",
                change_type=ChangeType.NODE_ADDED,
                entity_id="node-1",
            )

    def test_empty_entity_id_raises(self):
        """ChangeEvent rejects empty entity_id."""
        with pytest.raises(pydantic.ValidationError):
            ChangeEvent(
                previous_snapshot_id="snap-1",
                current_snapshot_id="snap-2",
                change_type=ChangeType.NODE_ADDED,
                entity_id="",
            )

    def test_default_severity(self):
        """ChangeEvent default severity is LOW."""
        event = ChangeEvent(
            previous_snapshot_id="snap-1",
            current_snapshot_id="snap-2",
            change_type=ChangeType.EDGE_ADDED,
            entity_id="e-1",
        )
        assert event.severity == ChangeSeverity.LOW


class TestQualityScoreModel:
    """Tests for QualityScore SDK model."""

    def test_creation(self):
        """QualityScore can be created with required fields."""
        score = QualityScore(asset_id="asset-1")
        assert score.asset_id == "asset-1"

    def test_empty_asset_id_raises(self):
        """QualityScore rejects empty asset_id."""
        with pytest.raises(pydantic.ValidationError):
            QualityScore(asset_id="")

    def test_defaults(self):
        """QualityScore defaults are set correctly."""
        score = QualityScore(asset_id="asset-1")
        assert score.source_credibility == 1.0
        assert score.transformation_depth == 0
        assert score.freshness_score == 1.0
        assert score.documentation_score == 0.0
        assert score.overall_score == 0.0

    def test_overall_score_range(self):
        """QualityScore rejects overall_score > 1."""
        with pytest.raises(pydantic.ValidationError):
            QualityScore(asset_id="a1", overall_score=1.5)


class TestAuditEntryModel:
    """Tests for AuditEntry SDK model."""

    def test_creation(self):
        """AuditEntry can be created with required fields."""
        entry = AuditEntry(
            action="register_asset",
            entity_type="DataAsset",
            entity_id="a1",
        )
        assert entry.action == "register_asset"
        assert entry.actor == "system"

    def test_empty_action_raises(self):
        """AuditEntry rejects empty action."""
        with pytest.raises(pydantic.ValidationError):
            AuditEntry(action="", entity_type="DataAsset", entity_id="a1")

    def test_empty_entity_type_raises(self):
        """AuditEntry rejects empty entity_type."""
        with pytest.raises(pydantic.ValidationError):
            AuditEntry(action="register", entity_type="", entity_id="a1")

    def test_empty_entity_id_raises(self):
        """AuditEntry rejects empty entity_id."""
        with pytest.raises(pydantic.ValidationError):
            AuditEntry(action="register", entity_type="DataAsset", entity_id="")


class TestGraphNodeModel:
    """Tests for GraphNode SDK model."""

    def test_creation(self):
        """GraphNode can be created with required fields."""
        node = GraphNode(asset_id="a1")
        assert node.asset_id == "a1"
        assert node.asset_type == AssetType.DATASET

    def test_empty_asset_id_raises(self):
        """GraphNode rejects empty asset_id."""
        with pytest.raises(pydantic.ValidationError):
            GraphNode(asset_id="")


class TestGraphEdgeViewModel:
    """Tests for GraphEdgeView SDK model."""

    def test_creation(self):
        """GraphEdgeView can be created with required fields."""
        view = GraphEdgeView(
            edge_id="e1", source_id="a1", target_id="a2",
        )
        assert view.edge_id == "e1"
        assert view.confidence == 1.0

    def test_empty_edge_id_raises(self):
        """GraphEdgeView rejects empty edge_id."""
        with pytest.raises(pydantic.ValidationError):
            GraphEdgeView(edge_id="", source_id="a1", target_id="a2")

    def test_empty_source_id_raises(self):
        """GraphEdgeView rejects empty source_id."""
        with pytest.raises(pydantic.ValidationError):
            GraphEdgeView(edge_id="e1", source_id="", target_id="a2")

    def test_empty_target_id_raises(self):
        """GraphEdgeView rejects empty target_id."""
        with pytest.raises(pydantic.ValidationError):
            GraphEdgeView(edge_id="e1", source_id="a1", target_id="")

    def test_confidence_range(self):
        """GraphEdgeView rejects confidence > 1."""
        with pytest.raises(pydantic.ValidationError):
            GraphEdgeView(
                edge_id="e1", source_id="a1", target_id="a2",
                confidence=1.5,
            )


class TestSubgraphResultModel:
    """Tests for SubgraphResult SDK model."""

    def test_creation(self):
        """SubgraphResult can be created with required fields."""
        result = SubgraphResult(root_asset_id="a1")
        assert result.root_asset_id == "a1"
        assert result.nodes == []
        assert result.edges == []

    def test_empty_root_asset_id_raises(self):
        """SubgraphResult rejects empty root_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            SubgraphResult(root_asset_id="")


class TestLineageChainModel:
    """Tests for LineageChain SDK model."""

    def test_creation(self):
        """LineageChain can be created with required fields."""
        chain = LineageChain(asset_id="a1")
        assert chain.asset_id == "a1"
        assert chain.direction == TraversalDirection.BACKWARD

    def test_empty_asset_id_raises(self):
        """LineageChain rejects empty asset_id."""
        with pytest.raises(pydantic.ValidationError):
            LineageChain(asset_id="")

    def test_defaults(self):
        """LineageChain defaults are set correctly."""
        chain = LineageChain(asset_id="a1")
        assert chain.chain == []
        assert chain.depth == 0
        assert chain.total_transformations == 0


class TestLineageStatisticsModel:
    """Tests for LineageStatistics SDK model."""

    def test_creation(self):
        """LineageStatistics can be created with defaults."""
        stats = LineageStatistics()
        assert stats.total_assets == 0
        assert stats.total_transformations == 0
        assert stats.total_edges == 0

    def test_coverage_score_range(self):
        """LineageStatistics rejects coverage_score > 1."""
        with pytest.raises(pydantic.ValidationError):
            LineageStatistics(coverage_score=1.5)


class TestPipelineResultModel:
    """Tests for PipelineResult SDK model."""

    def test_creation(self):
        """PipelineResult can be created with defaults."""
        result = PipelineResult()
        assert result.stages_completed == 0
        assert result.report_generated is False
        assert result.errors == []

    def test_serialization(self):
        """PipelineResult can be serialized."""
        result = PipelineResult(
            stages_completed=5,
            assets_registered=10,
            duration_ms=250.5,
        )
        d = result.model_dump()
        assert d["stages_completed"] == 5
        assert d["duration_ms"] == 250.5


# ============================================================================
# TestRequestModels
# ============================================================================


class TestRegisterAssetRequest:
    """Tests for RegisterAssetRequest."""

    def test_creation(self):
        """RegisterAssetRequest can be created with required fields."""
        req = RegisterAssetRequest(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        assert req.qualified_name == "test.asset"

    def test_empty_qualified_name_raises(self):
        """RegisterAssetRequest rejects empty qualified_name."""
        with pytest.raises(pydantic.ValidationError):
            RegisterAssetRequest(
                qualified_name="",
                asset_type=AssetType.DATASET,
            )

    def test_optional_fields(self):
        """RegisterAssetRequest optional fields default correctly."""
        req = RegisterAssetRequest(
            qualified_name="test.asset",
            asset_type=AssetType.DATASET,
        )
        assert req.display_name == ""
        assert req.owner == ""
        assert req.tags == {}
        assert req.classification == AssetClassification.INTERNAL


class TestUpdateAssetRequest:
    """Tests for UpdateAssetRequest."""

    def test_creation_empty(self):
        """UpdateAssetRequest can be created with no fields."""
        req = UpdateAssetRequest()
        assert req.display_name is None
        assert req.owner is None
        assert req.status is None

    def test_creation_with_fields(self):
        """UpdateAssetRequest accepts optional fields."""
        req = UpdateAssetRequest(
            display_name="New Name",
            status=AssetStatus.DEPRECATED,
        )
        assert req.display_name == "New Name"
        assert req.status == AssetStatus.DEPRECATED


class TestRecordTransformationRequest:
    """Tests for RecordTransformationRequest."""

    def test_creation(self):
        """RecordTransformationRequest can be created with required fields."""
        req = RecordTransformationRequest(
            transformation_type=TransformationType.FILTER,
        )
        assert req.transformation_type == TransformationType.FILTER

    def test_defaults(self):
        """RecordTransformationRequest defaults are correct."""
        req = RecordTransformationRequest(
            transformation_type=TransformationType.JOIN,
        )
        assert req.agent_id == ""
        assert req.records_in == 0
        assert req.duration_ms == 0.0

    def test_negative_records_raises(self):
        """RecordTransformationRequest rejects negative record counts."""
        with pytest.raises(pydantic.ValidationError):
            RecordTransformationRequest(
                transformation_type=TransformationType.FILTER,
                records_in=-1,
            )


class TestCreateEdgeRequest:
    """Tests for CreateEdgeRequest."""

    def test_creation(self):
        """CreateEdgeRequest can be created with required fields."""
        req = CreateEdgeRequest(
            source_asset_id="src-1",
            target_asset_id="tgt-1",
        )
        assert req.source_asset_id == "src-1"

    def test_empty_source_raises(self):
        """CreateEdgeRequest rejects empty source_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            CreateEdgeRequest(
                source_asset_id="",
                target_asset_id="tgt-1",
            )

    def test_empty_target_raises(self):
        """CreateEdgeRequest rejects empty target_asset_id."""
        with pytest.raises(pydantic.ValidationError):
            CreateEdgeRequest(
                source_asset_id="src-1",
                target_asset_id="",
            )

    def test_confidence_range(self):
        """CreateEdgeRequest rejects confidence > 1."""
        with pytest.raises(pydantic.ValidationError):
            CreateEdgeRequest(
                source_asset_id="src-1",
                target_asset_id="tgt-1",
                confidence=1.5,
            )


class TestRunImpactAnalysisRequest:
    """Tests for RunImpactAnalysisRequest."""

    def test_creation(self):
        """RunImpactAnalysisRequest can be created with required fields."""
        req = RunImpactAnalysisRequest(asset_id="a1")
        assert req.asset_id == "a1"
        assert req.direction == TraversalDirection.FORWARD
        assert req.max_depth == 10

    def test_empty_asset_id_raises(self):
        """RunImpactAnalysisRequest rejects empty asset_id."""
        with pytest.raises(pydantic.ValidationError):
            RunImpactAnalysisRequest(asset_id="")


class TestRunValidationRequest:
    """Tests for RunValidationRequest."""

    def test_creation(self):
        """RunValidationRequest can be created with defaults."""
        req = RunValidationRequest()
        assert req.scope == "full"
        assert req.include_freshness is True
        assert req.include_coverage is True


class TestGenerateReportRequest:
    """Tests for GenerateReportRequest."""

    def test_creation(self):
        """GenerateReportRequest can be created with defaults."""
        req = GenerateReportRequest()
        assert req.report_type == ReportType.CUSTOM
        assert req.format == ReportFormat.JSON
        assert req.max_depth == 10

    def test_custom_format(self):
        """GenerateReportRequest accepts custom format."""
        req = GenerateReportRequest(format=ReportFormat.MERMAID)
        assert req.format == ReportFormat.MERMAID


class TestRunPipelineRequest:
    """Tests for RunPipelineRequest."""

    def test_creation(self):
        """RunPipelineRequest can be created with defaults."""
        req = RunPipelineRequest()
        assert req.scope == "full"
        assert req.register_assets is True
        assert req.capture_transformations is True
        assert req.run_validation is True
        assert req.generate_report is False
        assert req.report_format == ReportFormat.JSON

    def test_auto_pipeline_id(self):
        """RunPipelineRequest generates a UUID pipeline_id."""
        req = RunPipelineRequest()
        uuid.UUID(req.pipeline_id)


# ============================================================================
# TestLayer1ReExports
# ============================================================================


class TestLayer1ReExports:
    """Tests for Layer 1 re-exported types."""

    def test_quality_dimension_exists(self):
        """QualityDimension is importable (real or stub)."""
        assert QualityDimension is not None

    def test_quality_dimension_has_completeness(self):
        """QualityDimension has COMPLETENESS member."""
        assert hasattr(QualityDimension, "COMPLETENESS")

    def test_quality_dimension_has_validity(self):
        """QualityDimension has VALIDITY member."""
        assert hasattr(QualityDimension, "VALIDITY")

    def test_quality_dimension_has_consistency(self):
        """QualityDimension has CONSISTENCY member."""
        assert hasattr(QualityDimension, "CONSISTENCY")

    def test_quality_dimension_has_timeliness(self):
        """QualityDimension has TIMELINESS member."""
        assert hasattr(QualityDimension, "TIMELINESS")

    def test_quality_dimension_has_uniqueness(self):
        """QualityDimension has UNIQUENESS member."""
        assert hasattr(QualityDimension, "UNIQUENESS")

    def test_quality_dimension_has_accuracy(self):
        """QualityDimension has ACCURACY member."""
        assert hasattr(QualityDimension, "ACCURACY")

    def test_quality_dimension_member_count(self):
        """QualityDimension has at least 6 members."""
        assert len(QualityDimension) >= 6
