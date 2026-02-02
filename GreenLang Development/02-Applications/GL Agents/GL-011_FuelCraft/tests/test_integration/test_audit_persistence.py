# -*- coding: utf-8 -*-
"""
Integration Tests for Audit Persistence

Tests audit trail persistence including:
- Run bundle storage
- Content-addressed retrieval
- Integrity verification
- 7-year retention compliance
- Replay validation

Author: GL-TestEngineer
Date: 2025-01-01
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import json
import gzip
import hashlib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from audit.run_bundle import (
    RunBundleBuilder,
    ImmutableStorage,
    BundleReplayValidator,
    BundleManifest,
    BundleStatus,
    ComponentType,
)


@pytest.mark.integration
class TestBundlePersistence:
    """Tests for bundle persistence to storage."""

    def test_persist_bundle_to_disk(self, run_bundle_builder, tmp_path):
        """Test bundle is persisted to disk."""
        storage_path = tmp_path / "bundles"

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
            storage_path=str(storage_path),
        )

        builder.add_input_snapshot(
            name="test_input",
            data={"key": "value"}
        )

        manifest = builder.seal()

        # Check bundle directory exists
        bundle_dir = storage_path / manifest.bundle_id
        assert bundle_dir.exists()

        # Check manifest file exists
        manifest_file = bundle_dir / "manifest.json"
        assert manifest_file.exists()

        # Check component files exist
        for component in manifest.components:
            component_file = bundle_dir / component.filename
            assert component_file.exists()

    def test_persist_compressed_components(self, tmp_path):
        """Test components are compressed with gzip."""
        storage_path = tmp_path / "bundles"

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
            storage_path=str(storage_path),
        )

        large_data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(100)]}
        builder.add_input_snapshot(name="large_input", data=large_data)

        manifest = builder.seal()

        # Check component is compressed
        bundle_dir = storage_path / manifest.bundle_id
        component = manifest.components[0]
        component_file = bundle_dir / component.filename

        with open(component_file, 'rb') as f:
            compressed_data = f.read()

        # Verify gzip header
        assert compressed_data[:2] == b'\x1f\x8b'

        # Decompress and verify
        decompressed = gzip.decompress(compressed_data)
        assert json.loads(decompressed) == large_data


@pytest.mark.integration
class TestImmutableStorageOperations:
    """Tests for ImmutableStorage class."""

    def test_store_bundle(self, tmp_path):
        """Test storing bundle in content-addressed storage."""
        storage = ImmutableStorage(str(tmp_path / "storage"))

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        path = storage.store_bundle(
            manifest=manifest,
            components=builder._components_data
        )

        assert path is not None
        assert Path(path).exists()

    def test_retrieve_bundle(self, tmp_path):
        """Test retrieving bundle by hash."""
        storage = ImmutableStorage(str(tmp_path / "storage"))

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        storage.store_bundle(manifest=manifest, components=builder._components_data)

        retrieved = storage.retrieve_bundle(manifest.bundle_hash)

        assert retrieved is not None
        assert retrieved.bundle_id == manifest.bundle_id
        assert retrieved.bundle_hash == manifest.bundle_hash

    def test_retrieve_nonexistent_bundle(self, tmp_path):
        """Test retrieving nonexistent bundle returns None."""
        storage = ImmutableStorage(str(tmp_path / "storage"))

        result = storage.retrieve_bundle("nonexistent_hash" * 4)

        assert result is None

    def test_verify_integrity_valid(self, tmp_path):
        """Test integrity verification for valid bundle."""
        storage = ImmutableStorage(str(tmp_path / "storage"))

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        storage.store_bundle(manifest=manifest, components=builder._components_data)

        is_valid = storage.verify_integrity(manifest.bundle_hash)

        assert is_valid is True

    def test_verify_integrity_corrupted(self, tmp_path):
        """Test integrity verification detects corruption."""
        storage = ImmutableStorage(str(tmp_path / "storage"))

        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        storage.store_bundle(manifest=manifest, components=builder._components_data)

        # Corrupt a component file
        bundle_dir = storage._base_path / manifest.bundle_hash[:2] / manifest.bundle_hash
        component_file = bundle_dir / manifest.components[0].filename

        with open(component_file, 'wb') as f:
            f.write(gzip.compress(b'{"corrupted": true}'))

        is_valid = storage.verify_integrity(manifest.bundle_hash)

        assert is_valid is False


@pytest.mark.integration
class TestReplayValidation:
    """Tests for replay validation."""

    def test_validate_identical_replay(self, tmp_path):
        """Test validation passes for identical replay."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        original_output = {"cost": 5000.0, "emissions": 1500.0}
        builder.add_output(name="result", data=original_output)
        manifest = builder.seal()

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=manifest,
            replay_outputs={"result": original_output}
        )

        assert result.is_identical is True
        assert result.components_matched == result.components_validated

    def test_validate_different_replay(self, tmp_path):
        """Test validation fails for different replay."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        original_output = {"cost": 5000.0, "emissions": 1500.0}
        builder.add_output(name="result", data=original_output)
        manifest = builder.seal()

        validator = BundleReplayValidator()
        different_output = {"cost": 5001.0, "emissions": 1500.0}  # Different cost

        result = validator.validate_replay(
            original_bundle=manifest,
            replay_outputs={"result": different_output}
        )

        assert result.is_identical is False
        assert len(result.mismatches) > 0

    def test_validate_missing_output(self, tmp_path):
        """Test validation fails for missing output."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        builder.add_output(name="result", data={"cost": 5000.0})
        manifest = builder.seal()

        validator = BundleReplayValidator()
        result = validator.validate_replay(
            original_bundle=manifest,
            replay_outputs={}  # Missing output
        )

        assert result.is_identical is False


@pytest.mark.integration
class TestRetentionCompliance:
    """Tests for 7-year retention compliance."""

    def test_retention_expiry_is_7_years(self):
        """Test retention expiry is set to 7 years."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
            retention_years=7,
        )

        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        days_until_expiry = (manifest.retention_expires - manifest.sealed_at).days
        years_until_expiry = days_until_expiry / 365

        assert 6.9 < years_until_expiry < 7.1

    def test_custom_retention_period(self):
        """Test custom retention period is respected."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
            retention_years=10,
        )

        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        days_until_expiry = (manifest.retention_expires - manifest.sealed_at).days
        years_until_expiry = days_until_expiry / 365

        assert 9.9 < years_until_expiry < 10.1


@pytest.mark.integration
class TestBundleMetadata:
    """Tests for bundle metadata."""

    def test_bundle_contains_agent_info(self):
        """Test bundle contains agent identification."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="2.1.0",
            environment="production",
        )

        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        assert manifest.agent_id == "GL-011"
        assert manifest.agent_version == "2.1.0"
        assert manifest.environment == "production"

    def test_bundle_contains_timestamps(self):
        """Test bundle contains creation timestamps."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.seal()

        assert manifest.created_at is not None
        assert manifest.sealed_at is not None
        assert manifest.sealed_at >= manifest.created_at


@pytest.mark.integration
class TestContentAddressedHashing:
    """Tests for content-addressed hashing."""

    def test_same_content_same_hash(self):
        """Test same content produces same hash."""
        builder1 = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder2 = RunBundleBuilder(
            run_id="RUN-TEST-002",
            agent_version="1.0.0",
            environment="test",
        )

        data = {"key": "value", "number": 123}

        builder1.add_input_snapshot(name="test", data=data)
        builder2.add_input_snapshot(name="test", data=data)

        manifest1 = builder1.get_manifest()
        manifest2 = builder2.get_manifest()

        # Same content should have same content hash
        assert manifest1.components[0].content_hash == manifest2.components[0].content_hash

    def test_different_content_different_hash(self):
        """Test different content produces different hash."""
        builder1 = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder2 = RunBundleBuilder(
            run_id="RUN-TEST-002",
            agent_version="1.0.0",
            environment="test",
        )

        builder1.add_input_snapshot(name="test", data={"key": "value1"})
        builder2.add_input_snapshot(name="test", data={"key": "value2"})

        manifest1 = builder1.get_manifest()
        manifest2 = builder2.get_manifest()

        # Different content should have different content hash
        assert manifest1.components[0].content_hash != manifest2.components[0].content_hash

    def test_hash_is_sha256(self):
        """Test content hash is SHA-256."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        builder.add_input_snapshot(name="test", data={"key": "value"})
        manifest = builder.get_manifest()

        hash_value = manifest.components[0].content_hash

        # SHA-256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)


@pytest.mark.integration
class TestMultipleComponentsBundle:
    """Tests for bundles with multiple components."""

    def test_bundle_with_all_component_types(self):
        """Test bundle with all component types."""
        builder = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )

        # Add all component types
        builder.add_input_snapshot(
            name="inventory",
            data={"tank_1": 100000},
            version="1.0.0"
        )
        builder.add_master_data(
            name="fuel_properties",
            data={"diesel": {"lhv": 43.0}},
            version="2024.1",
            effective_date=datetime.now(timezone.utc)
        )
        builder.add_solver_config(
            solver_name="highs",
            config={"mip_gap": 0.01},
            tolerances={"mip_gap": 0.01}
        )
        builder.add_conversion_log([
            {"from": "MMBtu", "to": "MJ", "factor": 1055.05585}
        ])
        builder.add_output(
            name="optimization_result",
            data={"cost": 50000, "status": "optimal"}
        )
        builder.add_execution_log([
            {"time": "2024-01-01T00:00:00Z", "message": "Started"}
        ])

        manifest = builder.seal()

        # Verify all components present
        component_types = {c.component_type for c in manifest.components}

        assert ComponentType.INPUT_SNAPSHOT in component_types
        assert ComponentType.MASTER_DATA in component_types
        assert ComponentType.SOLVER_CONFIG in component_types
        assert ComponentType.CONVERSION_LOG in component_types
        assert ComponentType.OUTPUT_DATA in component_types
        assert ComponentType.EXECUTION_LOG in component_types

    def test_bundle_hash_includes_all_components(self):
        """Test bundle hash depends on all components."""
        # Bundle 1 with fewer components
        builder1 = RunBundleBuilder(
            run_id="RUN-TEST-001",
            agent_version="1.0.0",
            environment="test",
        )
        builder1.add_input_snapshot(name="test", data={"key": "value"})
        manifest1 = builder1.seal()

        # Bundle 2 with more components
        builder2 = RunBundleBuilder(
            run_id="RUN-TEST-002",
            agent_version="1.0.0",
            environment="test",
        )
        builder2.add_input_snapshot(name="test", data={"key": "value"})
        builder2.add_output(name="output", data={"result": 123})
        manifest2 = builder2.seal()

        # Bundle hashes should be different
        assert manifest1.bundle_hash != manifest2.bundle_hash
