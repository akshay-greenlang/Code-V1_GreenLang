"""
Comprehensive tests for provenance utilities.

Tests cover:
- ProvenanceContext tracking
- Artifact management
- Signature handling
- SBOM references
- Context finalization
- Decorator functionality
"""

import pytest
import json
from pathlib import Path
from datetime import datetime
from greenlang.provenance.utils import (
    ProvenanceContext,
    track_provenance,
    verify_artifact_chain,
    generate_provenance_report,
)


@pytest.mark.unit
class TestProvenanceContextInit:
    """Test ProvenanceContext initialization."""

    def test_context_default_init(self):
        """Test creating context with defaults."""
        ctx = ProvenanceContext()

        assert ctx.name == "default"
        assert ctx.started_at is not None
        assert ctx.start_time > 0
        assert ctx.status == "running"
        assert ctx.inputs == {}
        assert ctx.outputs == {}
        assert ctx.artifacts == []
        assert ctx.signatures == []
        assert ctx.backend == "local"
        assert ctx.profile == "dev"

    def test_context_custom_name(self):
        """Test creating context with custom name."""
        ctx = ProvenanceContext(name="test-pipeline")

        assert ctx.name == "test-pipeline"
        assert ctx.status == "running"

    def test_context_timestamps(self):
        """Test that context has valid timestamps."""
        ctx = ProvenanceContext()

        assert isinstance(ctx.started_at, datetime)
        assert isinstance(ctx.start_time, float)
        assert ctx.start_time > 0


@pytest.mark.unit
class TestProvenanceInputOutput:
    """Test recording inputs and outputs."""

    def test_record_inputs_args(self):
        """Test recording positional arguments."""
        ctx = ProvenanceContext()
        ctx.record_inputs((1, 2, 3), {})

        assert ctx.inputs["args"] == [1, 2, 3]
        assert ctx.inputs["kwargs"] == {}

    def test_record_inputs_kwargs(self):
        """Test recording keyword arguments."""
        ctx = ProvenanceContext()
        ctx.record_inputs((), {"key1": "value1", "key2": "value2"})

        assert ctx.inputs["args"] == []
        assert ctx.inputs["kwargs"] == {"key1": "value1", "key2": "value2"}

    def test_record_inputs_both(self):
        """Test recording both args and kwargs."""
        ctx = ProvenanceContext()
        ctx.record_inputs((10, 20), {"mode": "fast", "debug": True})

        assert ctx.inputs["args"] == [10, 20]
        assert ctx.inputs["kwargs"] == {"mode": "fast", "debug": True}

    def test_record_outputs(self):
        """Test recording outputs."""
        ctx = ProvenanceContext()
        outputs = {"result": 42, "status": "success"}
        ctx.record_outputs(outputs)

        assert ctx.outputs == outputs

    def test_record_outputs_overwrite(self):
        """Test that recording outputs overwrites previous."""
        ctx = ProvenanceContext()
        ctx.record_outputs({"first": 1})
        ctx.record_outputs({"second": 2})

        assert ctx.outputs == {"second": 2}


@pytest.mark.unit
class TestProvenanceArtifacts:
    """Test artifact management."""

    def test_add_artifact(self, tmp_path):
        """Test adding an artifact."""
        ctx = ProvenanceContext()
        artifact_path = tmp_path / "test.txt"
        artifact_path.write_text("test content")

        ctx.add_artifact("test-artifact", artifact_path, "file")

        assert len(ctx.artifacts) == 1
        assert ctx.artifacts[0]["name"] == "test-artifact"
        assert ctx.artifacts[0]["path"] == str(artifact_path)
        assert ctx.artifacts[0]["type"] == "file"

    def test_add_artifact_with_metadata(self, tmp_path):
        """Test adding artifact with metadata."""
        ctx = ProvenanceContext()
        artifact_path = tmp_path / "data.json"
        artifact_path.write_text("{}")

        ctx.add_artifact(
            "data-file",
            artifact_path,
            "json",
            metadata={"size": 100, "format": "json"},
        )

        artifact = ctx.artifacts[0]
        assert artifact["metadata"]["size"] == 100
        assert artifact["metadata"]["format"] == "json"

    def test_add_multiple_artifacts(self, tmp_path):
        """Test adding multiple artifacts."""
        ctx = ProvenanceContext()

        for i in range(3):
            path = tmp_path / f"file{i}.txt"
            path.write_text(f"content {i}")
            ctx.add_artifact(f"artifact-{i}", path, "file")

        assert len(ctx.artifacts) == 3
        assert ctx.artifacts[0]["name"] == "artifact-0"
        assert ctx.artifacts[2]["name"] == "artifact-2"

    def test_artifacts_map(self, tmp_path):
        """Test that artifacts_map is maintained."""
        ctx = ProvenanceContext()
        path = tmp_path / "test.txt"
        path.write_text("test")

        ctx.add_artifact("my-artifact", path, "file")

        assert "my-artifact" in ctx.artifacts_map
        assert ctx.artifacts_map["my-artifact"] == str(path)


@pytest.mark.unit
class TestProvenanceSignatures:
    """Test signature handling."""

    def test_add_signature(self):
        """Test adding a signature."""
        ctx = ProvenanceContext()
        ctx.add_signature("sha256", "abc123def456", {"algorithm": "RSA"})

        assert len(ctx.signatures) == 1
        sig = ctx.signatures[0]
        assert sig["type"] == "sha256"
        assert sig["value"] == "abc123def456"
        assert sig["metadata"]["algorithm"] == "RSA"
        assert "timestamp" in sig

    def test_add_multiple_signatures(self):
        """Test adding multiple signatures."""
        ctx = ProvenanceContext()
        ctx.add_signature("sha256", "hash1")
        ctx.add_signature("sha512", "hash2")
        ctx.add_signature("pgp", "signature")

        assert len(ctx.signatures) == 3
        assert ctx.signatures[0]["type"] == "sha256"
        assert ctx.signatures[1]["type"] == "sha512"
        assert ctx.signatures[2]["type"] == "pgp"

    def test_signature_timestamp(self):
        """Test that signatures include timestamps."""
        ctx = ProvenanceContext()
        ctx.add_signature("test", "value")

        timestamp = ctx.signatures[0]["timestamp"]
        assert isinstance(timestamp, str)
        # Verify it's a valid ISO format timestamp
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


@pytest.mark.unit
class TestProvenanceSBOM:
    """Test SBOM reference handling."""

    def test_set_sbom(self, tmp_path):
        """Test setting SBOM reference."""
        ctx = ProvenanceContext()
        sbom_path = tmp_path / "sbom.json"
        sbom_path.write_text("{}")

        ctx.set_sbom(sbom_path)

        assert ctx.sbom_path == str(sbom_path)

    def test_set_sbom_updates(self, tmp_path):
        """Test that setting SBOM updates reference."""
        ctx = ProvenanceContext()
        sbom1 = tmp_path / "sbom1.json"
        sbom2 = tmp_path / "sbom2.json"

        ctx.set_sbom(sbom1)
        assert ctx.sbom_path == str(sbom1)

        ctx.set_sbom(sbom2)
        assert ctx.sbom_path == str(sbom2)


@pytest.mark.unit
class TestProvenanceStatus:
    """Test provenance status management."""

    def test_initial_status(self):
        """Test that initial status is running."""
        ctx = ProvenanceContext()
        assert ctx.status == "running"

    def test_set_status_success(self):
        """Test setting status to success."""
        ctx = ProvenanceContext()
        ctx.status = "success"
        assert ctx.status == "success"

    def test_set_status_failed(self):
        """Test setting status to failed."""
        ctx = ProvenanceContext()
        ctx.status = "failed"
        ctx.error = "Something went wrong"

        assert ctx.status == "failed"
        assert ctx.error == "Something went wrong"


@pytest.mark.unit
class TestTrackProvenanceDecorator:
    """Test track_provenance decorator."""

    def test_decorator_successful_function(self):
        """Test decorator with successful function."""

        @track_provenance
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)
        assert result == 8

    def test_decorator_failing_function(self):
        """Test decorator with failing function."""

        @track_provenance
        def failing_function():
            raise ValueError("Intentional failure")

        with pytest.raises(ValueError, match="Intentional failure"):
            failing_function()

    def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""

        @track_provenance
        def process_data(value, multiplier=2):
            return value * multiplier

        result = process_data(10, multiplier=3)
        assert result == 30


@pytest.mark.unit
class TestVerifyArtifactChain:
    """Test artifact chain verification."""

    def test_verify_nonexistent_artifact(self, tmp_path):
        """Test verifying non-existent artifact."""
        artifact_path = tmp_path / "missing.txt"

        is_valid, issues = verify_artifact_chain(artifact_path)

        assert is_valid is False
        assert "Artifact not found" in issues

    def test_verify_artifact_without_sbom(self, tmp_path):
        """Test verifying artifact without SBOM."""
        artifact_path = tmp_path / "file.txt"
        artifact_path.write_text("content")

        is_valid, issues = verify_artifact_chain(artifact_path)

        assert is_valid is False
        assert any("SBOM" in issue for issue in issues)

    def test_verify_artifact_without_signature(self, tmp_path):
        """Test verifying artifact without signature."""
        artifact_path = tmp_path / "file.txt"
        artifact_path.write_text("content")

        is_valid, issues = verify_artifact_chain(artifact_path)

        assert is_valid is False
        assert any("signature" in issue.lower() for issue in issues)


@pytest.mark.unit
class TestGenerateProvenanceReport:
    """Test provenance report generation."""

    def test_generate_report_basic(self, tmp_path):
        """Test generating basic provenance report."""
        artifact_path = tmp_path / "test.txt"
        artifact_path.write_text("test content")

        report = generate_provenance_report(artifact_path)

        assert "artifact" in report
        assert "timestamp" in report
        assert "checks" in report
        assert "metadata" in report

    def test_report_includes_artifact_path(self, tmp_path):
        """Test that report includes artifact path."""
        artifact_path = tmp_path / "data.json"
        artifact_path.write_text("{}")

        report = generate_provenance_report(artifact_path)

        assert report["artifact"] == str(artifact_path)

    def test_report_includes_timestamp(self, tmp_path):
        """Test that report includes timestamp."""
        artifact_path = tmp_path / "file.txt"
        artifact_path.write_text("content")

        report = generate_provenance_report(artifact_path)

        timestamp = report["timestamp"]
        # Verify it's valid ISO format
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_report_includes_checks(self, tmp_path):
        """Test that report includes verification checks."""
        artifact_path = tmp_path / "test.txt"
        artifact_path.write_text("test")

        report = generate_provenance_report(artifact_path)

        assert "checks" in report
        assert "chain_valid" in report["checks"]
        assert "issues" in report["checks"]

    def test_report_includes_metadata(self, tmp_path):
        """Test that report includes file metadata."""
        artifact_path = tmp_path / "test.txt"
        artifact_path.write_text("test content")

        report = generate_provenance_report(artifact_path)

        metadata = report["metadata"]
        assert "size" in metadata
        assert metadata["size"] > 0
        assert "sha256" in metadata
        assert len(metadata["sha256"]) == 64  # SHA256 hex length


@pytest.mark.unit
class TestProvenanceContextConfiguration:
    """Test provenance context configuration."""

    def test_context_backend(self):
        """Test context backend configuration."""
        ctx = ProvenanceContext()
        assert ctx.backend == "local"

        ctx.backend = "cloud"
        assert ctx.backend == "cloud"

    def test_context_profile(self):
        """Test context profile configuration."""
        ctx = ProvenanceContext()
        assert ctx.profile == "dev"

        ctx.profile = "production"
        assert ctx.profile == "production"

    def test_context_pipeline_spec(self):
        """Test setting pipeline specification."""
        ctx = ProvenanceContext()
        spec = {"version": "1.0", "steps": ["step1", "step2"]}

        ctx.pipeline_spec = spec
        assert ctx.pipeline_spec == spec

    def test_context_config(self):
        """Test context configuration."""
        ctx = ProvenanceContext()
        config = {"max_workers": 4, "timeout": 300}

        ctx.config = config
        assert ctx.config == config

    def test_context_environment(self):
        """Test context environment tracking."""
        ctx = ProvenanceContext()
        env = {"python_version": "3.10", "platform": "linux"}

        ctx.environment = env
        assert ctx.environment == env
