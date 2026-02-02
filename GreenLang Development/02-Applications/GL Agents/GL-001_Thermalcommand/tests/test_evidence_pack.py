"""
Tests for Evidence Pack Generator Module

Comprehensive test coverage for:
- Evidence pack generation
- Pack storage and retrieval
- Pack verification
- Export formats

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit.evidence_pack import (
    EvidencePack,
    EvidencePackGenerator,
    EvidencePackFormat,
    EvidencePackStatus,
    TimestampRecord,
    DatasetSummary,
    ModelVersionSummary,
    ConstraintSummary,
    SolverSummary,
    ExplainabilitySummary,
    ActionSummary,
    ImpactSummary,
    OperatorActionRecord,
)

from audit.audit_events import (
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    SolverStatus,
    ActionStatus,
    SafetyLevel,
    RecommendedAction,
    ExpectedImpact,
    InputDatasetReference,
    ModelVersionInfo,
    ConstraintInfo,
    ExplainabilityArtifact,
)


class TestTimestampRecord:
    """Tests for TimestampRecord model."""

    def test_create_timestamp_record(self):
        """Test creating timestamp record."""
        now = datetime.now(timezone.utc)
        record = TimestampRecord(
            ingestion_timestamp=now - timedelta(seconds=10),
            decision_timestamp=now - timedelta(seconds=5),
            recommendation_timestamp=now,
        )

        assert record.total_latency_ms == pytest.approx(5000, rel=0.1)

    def test_actuation_latency(self):
        """Test actuation latency calculation."""
        now = datetime.now(timezone.utc)
        record = TimestampRecord(
            ingestion_timestamp=now - timedelta(seconds=10),
            decision_timestamp=now - timedelta(seconds=5),
            recommendation_timestamp=now - timedelta(seconds=4),
            actuation_timestamp=now,
        )

        assert record.actuation_latency_ms == pytest.approx(5000, rel=0.1)

    def test_actuation_latency_none(self):
        """Test actuation latency when no actuation."""
        now = datetime.now(timezone.utc)
        record = TimestampRecord(
            ingestion_timestamp=now,
            decision_timestamp=now,
            recommendation_timestamp=now,
        )

        assert record.actuation_latency_ms is None


class TestDatasetSummary:
    """Tests for DatasetSummary model."""

    def test_create_dataset_summary(self):
        """Test creating dataset summary."""
        summary = DatasetSummary(
            datasets=[],
            total_records=1000,
            combined_hash="abc123",
        )

        assert summary.total_records == 1000


class TestSolverSummary:
    """Tests for SolverSummary model."""

    def test_create_solver_summary(self):
        """Test creating solver summary."""
        summary = SolverSummary(
            solver_name="HiGHS",
            solver_version="1.5.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
            objective_breakdown={"energy": 100000, "penalty": 25000},
        )

        assert summary.solver_name == "HiGHS"
        assert summary.solver_status == SolverStatus.OPTIMAL


class TestEvidencePack:
    """Tests for EvidencePack model."""

    @pytest.fixture
    def sample_evidence_pack(self):
        """Create sample evidence pack."""
        now = datetime.now(timezone.utc)
        return EvidencePack(
            correlation_id="corr-12345",
            decision_event_id="dec-67890",
            asset_id="boiler-001",
            timestamps=TimestampRecord(
                ingestion_timestamp=now - timedelta(seconds=10),
                decision_timestamp=now,
                recommendation_timestamp=now,
            ),
            dataset_summary=DatasetSummary(
                datasets=[],
                total_records=100,
                combined_hash="hash123",
            ),
            unit_conversion_version="1.0.0",
            constraint_summary=ConstraintSummary(
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
            ),
            model_summary=ModelVersionSummary(),
            explainability_summary=ExplainabilitySummary(),
            solver_summary=SolverSummary(
                solver_name="HiGHS",
                solver_version="1.5.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            ),
            action_summary=ActionSummary(),
            impact_summary=ImpactSummary(),
        )

    def test_create_evidence_pack(self, sample_evidence_pack):
        """Test creating evidence pack."""
        pack = sample_evidence_pack

        assert pack.correlation_id == "corr-12345"
        assert pack.status == EvidencePackStatus.DRAFT

    def test_seal_evidence_pack(self, sample_evidence_pack):
        """Test sealing evidence pack."""
        pack = sample_evidence_pack
        pack.seal()

        assert pack.status == EvidencePackStatus.SEALED
        assert pack.pack_hash is not None
        assert len(pack.pack_hash) == 64

    def test_verify_evidence_pack(self, sample_evidence_pack):
        """Test verifying evidence pack."""
        pack = sample_evidence_pack
        pack.seal()

        assert pack.verify() is True

    def test_verify_unsealed_pack(self, sample_evidence_pack):
        """Test verifying unsealed pack returns False."""
        pack = sample_evidence_pack
        # Don't seal

        assert pack.verify() is False

    def test_calculate_hash_deterministic(self, sample_evidence_pack):
        """Test hash calculation is deterministic."""
        pack = sample_evidence_pack

        hash1 = pack.calculate_hash()
        hash2 = pack.calculate_hash()

        assert hash1 == hash2


class TestEvidencePackGenerator:
    """Tests for EvidencePackGenerator."""

    @pytest.fixture
    def generator(self):
        """Create evidence pack generator."""
        return EvidencePackGenerator()

    @pytest.fixture
    def generator_with_storage(self, tmp_path):
        """Create generator with temp storage."""
        return EvidencePackGenerator(storage_path=str(tmp_path))

    @pytest.fixture
    def sample_decision_event(self):
        """Create sample decision event."""
        now = datetime.now(timezone.utc)
        return DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            facility_id="plant-001",
            ingestion_timestamp=now - timedelta(seconds=5),
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            input_datasets=[
                InputDatasetReference(
                    dataset_id="ds-001",
                    dataset_type="sensor",
                    schema_version="1.0.0",
                    data_hash="hash1",
                    record_count=100,
                    source_system="OPC-UA",
                ),
            ],
            demand_model=ModelVersionInfo(
                model_name="demand_forecast",
                model_version="2.1.0",
                model_hash="model_hash_123",
            ),
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
            objective_breakdown={"energy": 100000, "penalty": 25000},
            binding_constraints=[
                ConstraintInfo(
                    constraint_id="c-001",
                    constraint_name="max_temperature",
                    constraint_type="inequality",
                    is_binding=True,
                ),
            ],
            recommended_actions=[
                RecommendedAction(
                    action_id="act-001",
                    tag_id="TIC-001.SP",
                    asset_id="boiler-001",
                    current_value=450.0,
                    recommended_value=460.0,
                    lower_bound=400.0,
                    upper_bound=500.0,
                    unit="degF",
                ),
            ],
            expected_impact=ExpectedImpact(
                cost_delta_usd=-500.0,
                emissions_delta_kg_co2e=-50.0,
                energy_delta_mmbtu=-10.0,
                efficiency_delta_pct=2.5,
                risk_score_delta=-0.1,
                confidence_interval_lower=-600.0,
                confidence_interval_upper=-400.0,
                horizon_minutes=60,
            ),
            shap_artifacts=[
                ExplainabilityArtifact(
                    artifact_type="SHAP",
                    model_name="demand_forecast",
                    feature_importances={"temperature": 0.3, "pressure": 0.2},
                    artifact_hash="shap_hash",
                ),
            ],
        )

    @pytest.fixture
    def sample_action_events(self, sample_decision_event):
        """Create sample action events."""
        now = datetime.now(timezone.utc)
        return [
            ActionAuditEvent(
                correlation_id="corr-12345",
                asset_id="boiler-001",
                decision_event_id=str(sample_decision_event.event_id),
                decision_correlation_id="corr-12345",
                action=sample_decision_event.recommended_actions[0],
                action_status=ActionStatus.EXECUTED,
                recommended_timestamp=now - timedelta(minutes=5),
                actuation_timestamp=now,
                executed_value=460.0,
            ),
        ]

    def test_generate_evidence_pack(self, generator, sample_decision_event):
        """Test generating evidence pack from decision event."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.correlation_id == "corr-12345"
        assert pack.asset_id == "boiler-001"
        assert pack.status == EvidencePackStatus.SEALED
        assert pack.pack_hash is not None

    def test_evidence_pack_timestamps(self, generator, sample_decision_event):
        """Test timestamps are correctly captured."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.timestamps.ingestion_timestamp == sample_decision_event.ingestion_timestamp
        assert pack.timestamps.decision_timestamp == sample_decision_event.decision_timestamp

    def test_evidence_pack_datasets(self, generator, sample_decision_event):
        """Test datasets are correctly summarized."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert len(pack.dataset_summary.datasets) == 1
        assert pack.dataset_summary.total_records == 100

    def test_evidence_pack_models(self, generator, sample_decision_event):
        """Test model versions are captured."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.model_summary.demand_forecast_model is not None
        assert pack.model_summary.demand_forecast_model.model_version == "2.1.0"

    def test_evidence_pack_constraints(self, generator, sample_decision_event):
        """Test constraints are summarized."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.constraint_summary.constraint_set_id == "cs-001"
        assert len(pack.constraint_summary.binding_constraints) == 1

    def test_evidence_pack_solver(self, generator, sample_decision_event):
        """Test solver results are captured."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.solver_summary.solver_status == SolverStatus.OPTIMAL
        assert pack.solver_summary.objective_value == 125000.50

    def test_evidence_pack_explainability(self, generator, sample_decision_event):
        """Test explainability artifacts are captured."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.explainability_summary.shap_available is True
        assert len(pack.explainability_summary.shap_artifacts) == 1
        assert "temperature" in pack.explainability_summary.top_features

    def test_evidence_pack_actions(
        self, generator, sample_decision_event, sample_action_events
    ):
        """Test actions are summarized."""
        pack = generator.generate(
            decision_event=sample_decision_event,
            action_events=sample_action_events,
        )

        assert pack.action_summary.total_actions == 1
        assert pack.action_summary.executed_count == 1

    def test_evidence_pack_impact(self, generator, sample_decision_event):
        """Test impact is captured."""
        pack = generator.generate(decision_event=sample_decision_event)

        assert pack.impact_summary.expected_impact is not None
        assert pack.impact_summary.expected_impact.cost_delta_usd == -500.0


class TestEvidencePackStorage:
    """Tests for evidence pack storage."""

    @pytest.fixture
    def generator_with_storage(self, tmp_path):
        """Create generator with temp storage."""
        return EvidencePackGenerator(storage_path=str(tmp_path)), tmp_path

    @pytest.fixture
    def sample_decision_event(self):
        """Create sample decision event."""
        now = datetime.now(timezone.utc)
        return DecisionAuditEvent(
            correlation_id="corr-12345",
            asset_id="boiler-001",
            ingestion_timestamp=now - timedelta(seconds=5),
            decision_timestamp=now,
            constraint_set_id="cs-001",
            constraint_set_version="1.0.0",
            safety_boundary_policy_version="2.0.0",
            solver_status=SolverStatus.OPTIMAL,
            solve_time_ms=150.5,
            objective_value=125000.50,
        )

    def test_store_json(self, generator_with_storage, sample_decision_event):
        """Test storing evidence pack as JSON."""
        generator, tmp_path = generator_with_storage

        pack = generator.generate(decision_event=sample_decision_event)
        path = generator.store(pack, format=EvidencePackFormat.JSON)

        assert Path(path).exists()
        assert path.endswith(".json")

        # Verify JSON is valid
        with open(path, "r") as f:
            data = json.load(f)
        assert data["correlation_id"] == "corr-12345"

    def test_store_zip(self, generator_with_storage, sample_decision_event):
        """Test storing evidence pack as ZIP."""
        generator, tmp_path = generator_with_storage

        pack = generator.generate(decision_event=sample_decision_event)
        path = generator.store(pack, format=EvidencePackFormat.ZIP)

        assert Path(path).exists()
        assert path.endswith(".zip")

    def test_load_json(self, generator_with_storage, sample_decision_event):
        """Test loading evidence pack from JSON."""
        generator, tmp_path = generator_with_storage

        pack = generator.generate(decision_event=sample_decision_event)
        path = generator.store(pack, format=EvidencePackFormat.JSON)

        loaded = generator.load(path)

        assert loaded.correlation_id == pack.correlation_id
        assert loaded.pack_hash == pack.pack_hash

    def test_load_zip(self, generator_with_storage, sample_decision_event):
        """Test loading evidence pack from ZIP."""
        generator, tmp_path = generator_with_storage

        pack = generator.generate(decision_event=sample_decision_event)
        path = generator.store(pack, format=EvidencePackFormat.ZIP)

        loaded = generator.load(path)

        assert loaded.correlation_id == pack.correlation_id

    def test_store_without_path_raises(self, sample_decision_event):
        """Test storing without storage path raises error."""
        generator = EvidencePackGenerator()  # No storage path

        pack = generator.generate(decision_event=sample_decision_event)

        with pytest.raises(ValueError, match="Storage path not configured"):
            generator.store(pack)

    def test_load_nonexistent_raises(self, generator_with_storage):
        """Test loading nonexistent file raises error."""
        generator, tmp_path = generator_with_storage

        with pytest.raises(FileNotFoundError):
            generator.load(str(tmp_path / "nonexistent.json"))

    def test_verify_loaded_pack(self, generator_with_storage, sample_decision_event):
        """Test loaded pack can be verified."""
        generator, tmp_path = generator_with_storage

        pack = generator.generate(decision_event=sample_decision_event)
        path = generator.store(pack, format=EvidencePackFormat.JSON)

        loaded = generator.load(path)

        assert generator.verify_pack(loaded) is True


class TestEvidencePackListing:
    """Tests for evidence pack listing."""

    @pytest.fixture
    def generator_with_packs(self, tmp_path):
        """Create generator with multiple stored packs."""
        generator = EvidencePackGenerator(storage_path=str(tmp_path))

        # Create and store multiple packs
        now = datetime.now(timezone.utc)
        for i, asset in enumerate(["boiler-001", "boiler-002", "boiler-001"]):
            event = DecisionAuditEvent(
                correlation_id=f"corr-{i}",
                asset_id=asset,
                ingestion_timestamp=now - timedelta(hours=i),
                decision_timestamp=now - timedelta(hours=i) + timedelta(seconds=5),
                constraint_set_id="cs-001",
                constraint_set_version="1.0.0",
                safety_boundary_policy_version="2.0.0",
                solver_status=SolverStatus.OPTIMAL,
                solve_time_ms=150.5,
                objective_value=125000.50,
            )
            pack = generator.generate(decision_event=event)
            generator.store(pack, format=EvidencePackFormat.JSON)

        return generator, tmp_path

    def test_list_all_packs(self, generator_with_packs):
        """Test listing all evidence packs."""
        generator, tmp_path = generator_with_packs

        packs = generator.list_packs()

        assert len(packs) == 3

    def test_list_packs_by_asset(self, generator_with_packs):
        """Test listing packs filtered by asset."""
        generator, tmp_path = generator_with_packs

        packs = generator.list_packs(asset_id="boiler-001")

        assert len(packs) == 2
        assert all(p["asset_id"] == "boiler-001" for p in packs)


class TestOperatorActionRecord:
    """Tests for OperatorActionRecord model."""

    def test_create_operator_action_record(self):
        """Test creating operator action record."""
        now = datetime.now(timezone.utc)
        record = OperatorActionRecord(
            operator_id="op-001",
            action_type="approve",
            timestamp=now,
            authorization_level="operator",
            notes="Approved based on field conditions",
        )

        assert record.operator_id == "op-001"
        assert record.action_type == "approve"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
