# -*- coding: utf-8 -*-
"""
Unit Tests for Provenance Tracking

Tests all provenance tracking methods with 85%+ coverage.
Validates:
- SHA-256 hash generation
- Calculation step recording
- Input/output capture
- Bundle creation and sealing
- Replay validation
- 7-year retention compliance

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, date, timezone, timedelta
import hashlib
import json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audit.provenance import ProvenanceTracker, ProvenanceRecord, CalculationStep
from audit.run_bundle import (
    RunBundleBuilder,
    BundleManifest,
    BundleStatus,
    ComponentType,
    BundleComponent,
    ImmutableStorage,
    BundleReplayValidator,
)


@pytest.mark.unit
class TestProvenanceTrackerInitialization:
    """Tests for ProvenanceTracker initialization."""

    def test_default_initialization(self, provenance_tracker):
        """Test tracker initializes correctly."""
        assert provenance_tracker.run_id is not None
        assert provenance_tracker.run_id.startswith("RUN-TEST-")

    def test_tracker_starts_empty(self, provenance_tracker):
        """Test tracker starts with no records."""
        assert len(provenance_tracker.get_records()) == 0


@pytest.mark.unit
class TestProvenanceTrackerHashGeneration:
    """Tests for SHA-256 hash generation."""

    def test_hash_is_sha256(self, provenance_tracker):
        """Test generated hash is SHA-256."""
        data = {"key": "value", "number": 123}

        hash_value = provenance_tracker.compute_hash(data)

        assert len(hash_value) == 64  # SHA-256 hex length
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_hash_is_deterministic(self, provenance_tracker):
        """Test same input produces same hash."""
        data = {"key": "value", "number": 123}

        hash1 = provenance_tracker.compute_hash(data)
        hash2 = provenance_tracker.compute_hash(data)

        assert hash1 == hash2

    def test_hash_is_order_independent(self, provenance_tracker):
        """Test hash is independent of key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = provenance_tracker.compute_hash(data1)
        hash2 = provenance_tracker.compute_hash(data2)

        assert hash1 == hash2

    def test_hash_differs_for_different_data(self, provenance_tracker):
        """Test different data produces different hash."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}

        hash1 = provenance_tracker.compute_hash(data1)
        hash2 = provenance_tracker.compute_hash(data2)

        assert hash1 != hash2


@pytest.mark.unit
class TestProvenanceTrackerStepRecording:
    """Tests for calculation step recording."""

    def test_record_step(self, provenance_tracker):
        """Test recording a calculation step."""
        step = CalculationStep(
            step_number=1,
            operation="input_validation",
            input_hash="abc123",
            output_hash="def456",
            timestamp=datetime.now(timezone.utc),
            details={"validated_fields": ["fuel_type", "quantity"]},
        )

        provenance_tracker.record_step(step)

        records = provenance_tracker.get_records()
        assert len(records) == 1
        assert records[0].step_number == 1

    def test_record_multiple_steps(self, provenance_tracker):
        """Test recording multiple steps in sequence."""
        for i in range(5):
            step = CalculationStep(
                step_number=i + 1,
                operation=f"operation_{i}",
                input_hash=f"input_{i}",
                output_hash=f"output_{i}",
                timestamp=datetime.now(timezone.utc),
                details={},
            )
            provenance_tracker.record_step(step)

        records = provenance_tracker.get_records()
        assert len(records) == 5

        # Verify sequence
        for i, record in enumerate(records):
            assert record.step_number == i + 1

    def test_step_has_timestamp(self, provenance_tracker):
        """Test step records include timestamp."""
        step = CalculationStep(
            step_number=1,
            operation="test",
            input_hash="abc",
            output_hash="def",
            timestamp=datetime.now(timezone.utc),
            details={},
        )

        provenance_tracker.record_step(step)

        records = provenance_tracker.get_records()
        assert records[0].timestamp is not None


@pytest.mark.unit
class TestRunBundleBuilderInitialization:
    """Tests for RunBundleBuilder initialization."""

    def test_builder_initialization(self, run_bundle_builder):
        """Test builder initializes correctly."""
        assert run_bundle_builder.bundle_id.startswith("BUNDLE-")
        assert run_bundle_builder.is_sealed is False

    def test_builder_manifest_created(self, run_bundle_builder):
        """Test manifest is created on initialization."""
        manifest = run_bundle_builder.get_manifest()

        assert manifest.bundle_id is not None
        assert manifest.status == BundleStatus.BUILDING
        assert manifest.agent_id == "GL-011"


@pytest.mark.unit
class TestRunBundleBuilderComponents:
    """Tests for adding components to bundle."""

    def test_add_input_snapshot(self, run_bundle_builder):
        """Test adding input snapshot."""
        data = {"fuel_id": "NG-001", "quantity": 1000}

        component_id = run_bundle_builder.add_input_snapshot(
            name="fuel_data",
            data=data,
            version="1.0.0"
        )

        assert component_id is not None
        manifest = run_bundle_builder.get_manifest()
        assert len(manifest.components) == 1
        assert manifest.components[0].component_type == ComponentType.INPUT_SNAPSHOT

    def test_add_output(self, run_bundle_builder):
        """Test adding output data."""
        data = {"total_cost": 5000, "emissions_kg": 1500}

        component_id = run_bundle_builder.add_output(
            name="optimization_result",
            data=data,
            output_type="result"
        )

        assert component_id is not None
        manifest = run_bundle_builder.get_manifest()
        assert len(manifest.components) == 1
        assert manifest.components[0].component_type == ComponentType.OUTPUT_DATA

    def test_add_solver_config(self, run_bundle_builder):
        """Test adding solver configuration."""
        config = {"solver": "highs", "mip_gap": 0.01}

        component_id = run_bundle_builder.add_solver_config(
            solver_name="highs",
            config=config,
            tolerances={"mip_gap": 0.01}
        )

        assert component_id is not None
        manifest = run_bundle_builder.get_manifest()
        assert manifest.components[0].component_type == ComponentType.SOLVER_CONFIG

    def test_add_conversion_log(self, run_bundle_builder):
        """Test adding conversion log."""
        conversions = [
            {"from": "MMBtu", "to": "MJ", "factor": 1055.05585},
            {"from": "kg", "to": "lb", "factor": 2.20462},
        ]

        component_id = run_bundle_builder.add_conversion_log(conversions)

        assert component_id is not None

    def test_add_master_data(self, run_bundle_builder):
        """Test adding master data."""
        data = {"diesel": {"lhv": 43.0, "density": 840}}

        component_id = run_bundle_builder.add_master_data(
            name="fuel_properties",
            data=data,
            version="2024.1",
            effective_date=datetime.now(timezone.utc)
        )

        assert component_id is not None
        manifest = run_bundle_builder.get_manifest()
        assert manifest.components[0].component_type == ComponentType.MASTER_DATA


@pytest.mark.unit
class TestRunBundleBuilderSealing:
    """Tests for bundle sealing."""

    def test_seal_bundle(self, run_bundle_builder):
        """Test sealing a bundle."""
        # Add at least one component
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )

        manifest = run_bundle_builder.seal()

        assert manifest.status == BundleStatus.SEALED
        assert manifest.sealed_at is not None
        assert manifest.bundle_hash is not None
        assert manifest.manifest_hash is not None

    def test_seal_generates_hash(self, run_bundle_builder):
        """Test sealing generates valid SHA-256 hash."""
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )

        manifest = run_bundle_builder.seal()

        assert len(manifest.bundle_hash) == 64
        assert len(manifest.manifest_hash) == 64

    def test_cannot_add_after_seal(self, run_bundle_builder):
        """Test cannot add components after sealing."""
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )
        run_bundle_builder.seal()

        with pytest.raises(ValueError, match="sealed"):
            run_bundle_builder.add_input_snapshot(
                name="another",
                data={"key": "value2"}
            )

    def test_cannot_seal_twice(self, run_bundle_builder):
        """Test cannot seal bundle twice."""
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )
        run_bundle_builder.seal()

        with pytest.raises(ValueError, match="already sealed"):
            run_bundle_builder.seal()

    def test_cannot_seal_empty_bundle(self, run_bundle_builder):
        """Test cannot seal bundle with no components."""
        with pytest.raises(ValueError, match="empty"):
            run_bundle_builder.seal()

    def test_retention_expiry_set(self, run_bundle_builder):
        """Test retention expiry is set on seal."""
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )

        manifest = run_bundle_builder.seal()

        assert manifest.retention_expires is not None
        # Should be ~7 years from now
        years_until_expiry = (manifest.retention_expires - datetime.now(timezone.utc)).days / 365
        assert 6.9 < years_until_expiry < 7.1


@pytest.mark.unit
class TestRunBundleBuilderContentAddressing:
    """Tests for content-addressed storage."""

    def test_component_has_content_hash(self, run_bundle_builder):
        """Test each component has content hash."""
        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )

        manifest = run_bundle_builder.get_manifest()
        component = manifest.components[0]

        assert component.content_hash is not None
        assert len(component.content_hash) == 64

    def test_same_data_same_hash(self, run_bundle_builder):
        """Test same data produces same content hash."""
        data = {"key": "value"}

        run_bundle_builder.add_input_snapshot(name="test1", data=data)
        run_bundle_builder.add_input_snapshot(name="test2", data=data)

        manifest = run_bundle_builder.get_manifest()

        assert manifest.components[0].content_hash == manifest.components[1].content_hash

    def test_different_data_different_hash(self, run_bundle_builder):
        """Test different data produces different content hash."""
        run_bundle_builder.add_input_snapshot(
            name="test1",
            data={"key": "value1"}
        )
        run_bundle_builder.add_input_snapshot(
            name="test2",
            data={"key": "value2"}
        )

        manifest = run_bundle_builder.get_manifest()

        assert manifest.components[0].content_hash != manifest.components[1].content_hash


@pytest.mark.unit
class TestBundleReplayValidation:
    """Tests for bundle replay validation."""

    def test_identical_replay_passes(self, run_bundle_builder):
        """Test identical replay outputs pass validation."""
        original_data = {"cost": 5000, "emissions": 1500}

        run_bundle_builder.add_output(name="result", data=original_data)
        manifest = run_bundle_builder.seal()

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=manifest,
            replay_outputs={"result": original_data}
        )

        assert result.is_identical is True
        assert result.components_matched == result.components_validated

    def test_different_replay_fails(self, run_bundle_builder):
        """Test different replay outputs fail validation."""
        original_data = {"cost": 5000, "emissions": 1500}
        different_data = {"cost": 5001, "emissions": 1500}

        run_bundle_builder.add_output(name="result", data=original_data)
        manifest = run_bundle_builder.seal()

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=manifest,
            replay_outputs={"result": different_data}
        )

        assert result.is_identical is False
        assert len(result.mismatches) > 0


@pytest.mark.unit
class TestImmutableStorage:
    """Tests for ImmutableStorage class."""

    def test_storage_initialization(self, tmp_path):
        """Test storage initializes correctly."""
        storage = ImmutableStorage(str(tmp_path / "bundles"))

        assert storage._base_path.exists()

    def test_store_and_retrieve_bundle(self, tmp_path, run_bundle_builder):
        """Test storing and retrieving a bundle."""
        storage = ImmutableStorage(str(tmp_path / "bundles"))

        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )
        manifest = run_bundle_builder.seal()

        # Store bundle
        storage_path = storage.store_bundle(
            manifest=manifest,
            components=run_bundle_builder._components_data
        )

        # Retrieve bundle
        retrieved = storage.retrieve_bundle(manifest.bundle_hash)

        assert retrieved is not None
        assert retrieved.bundle_id == manifest.bundle_id

    def test_verify_integrity(self, tmp_path, run_bundle_builder):
        """Test bundle integrity verification."""
        storage = ImmutableStorage(str(tmp_path / "bundles"))

        run_bundle_builder.add_input_snapshot(
            name="test",
            data={"key": "value"}
        )
        manifest = run_bundle_builder.seal()

        storage.store_bundle(
            manifest=manifest,
            components=run_bundle_builder._components_data
        )

        is_valid = storage.verify_integrity(manifest.bundle_hash)

        assert is_valid is True


@pytest.mark.unit
class TestBundleStatusEnum:
    """Tests for BundleStatus enumeration."""

    def test_bundle_status_values(self):
        """Test BundleStatus enum values."""
        assert BundleStatus.BUILDING.value == "building"
        assert BundleStatus.SEALED.value == "sealed"
        assert BundleStatus.ARCHIVED.value == "archived"
        assert BundleStatus.CORRUPTED.value == "corrupted"


@pytest.mark.unit
class TestComponentTypeEnum:
    """Tests for ComponentType enumeration."""

    def test_component_type_values(self):
        """Test ComponentType enum values."""
        assert ComponentType.INPUT_SNAPSHOT.value == "input_snapshot"
        assert ComponentType.OUTPUT_DATA.value == "output_data"
        assert ComponentType.CONVERSION_LOG.value == "conversion_log"
        assert ComponentType.MASTER_DATA.value == "master_data"
        assert ComponentType.MODEL_VERSION.value == "model_version"
        assert ComponentType.SOLVER_CONFIG.value == "solver_config"


@pytest.mark.unit
class TestBundleComponentClass:
    """Tests for BundleComponent class."""

    def test_component_creation(self):
        """Test BundleComponent creation."""
        component = BundleComponent(
            component_id="test_id",
            component_type=ComponentType.INPUT_SNAPSHOT,
            filename="test.json.gz",
            content_hash="abc123" * 10 + "abcd",
            size_bytes=1024,
            compressed=True,
            metadata={"version": "1.0"}
        )

        assert component.component_id == "test_id"
        assert component.compressed is True


@pytest.mark.unit
class TestBundleManifestClass:
    """Tests for BundleManifest class."""

    def test_manifest_defaults(self):
        """Test BundleManifest default values."""
        manifest = BundleManifest(
            bundle_id="BUNDLE-TEST",
            run_id="RUN-TEST",
            agent_version="1.0.0",
            environment="test"
        )

        assert manifest.status == BundleStatus.BUILDING
        assert manifest.agent_id == "GL-011"
        assert manifest.retention_years == 7
        assert len(manifest.components) == 0


@pytest.mark.unit
class TestCalculationStepClass:
    """Tests for CalculationStep class."""

    def test_calculation_step_creation(self):
        """Test CalculationStep creation."""
        step = CalculationStep(
            step_number=1,
            operation="blend_calculation",
            input_hash="abc123",
            output_hash="def456",
            timestamp=datetime.now(timezone.utc),
            details={"method": "energy_weighted"}
        )

        assert step.step_number == 1
        assert step.operation == "blend_calculation"
        assert "method" in step.details
