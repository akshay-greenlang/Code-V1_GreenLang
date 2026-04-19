# -*- coding: utf-8 -*-
"""
Comprehensive tests for SDK Context abstraction.

Tests cover:
- Context initialization
- Artifact management
- Step result tracking
- Context conversion (to_dict, to_result)
- File operations
"""

import pytest
import json
from pathlib import Path
from greenlang.sdk.context import Context, Artifact
from greenlang.sdk.base import Result
from greenlang.determinism import DeterministicClock


@pytest.mark.unit
class TestContextInitialization:
    """Test Context initialization."""

    def test_context_default_init(self):
        """Test creating context with defaults."""
        ctx = Context()

        assert ctx.inputs == {}
        assert ctx.data == {}
        assert ctx.profile == "dev"
        assert ctx.backend == "local"
        assert isinstance(ctx.artifacts_dir, Path)
        assert ctx.artifacts == {}
        assert ctx.steps == {}

    def test_context_with_inputs(self):
        """Test creating context with inputs."""
        inputs = {"key1": "value1", "key2": 42}
        ctx = Context(inputs=inputs)

        assert ctx.inputs == inputs
        assert ctx.data == inputs

    def test_context_with_artifacts_dir(self, tmp_path):
        """Test creating context with custom artifacts directory."""
        artifacts_dir = tmp_path / "artifacts"
        ctx = Context(artifacts_dir=artifacts_dir)

        assert ctx.artifacts_dir == artifacts_dir
        assert artifacts_dir.exists()

    def test_context_with_profile_and_backend(self):
        """Test creating context with profile and backend."""
        ctx = Context(profile="production", backend="cloud")

        assert ctx.profile == "production"
        assert ctx.backend == "cloud"

    def test_context_with_metadata(self):
        """Test creating context with metadata."""
        metadata = {"user": "alice", "project": "test"}
        ctx = Context(metadata=metadata)

        assert ctx.metadata["user"] == "alice"
        assert ctx.metadata["project"] == "test"
        assert "timestamp" in ctx.metadata

    def test_context_creates_artifacts_dir(self, tmp_path):
        """Test that context creates artifacts directory."""
        artifacts_dir = tmp_path / "new" / "nested" / "dir"
        ctx = Context(artifacts_dir=artifacts_dir)

        assert artifacts_dir.exists()
        assert artifacts_dir.is_dir()


@pytest.mark.unit
class TestArtifactClass:
    """Test Artifact dataclass."""

    def test_artifact_creation(self, tmp_path):
        """Test creating an artifact."""
        path = tmp_path / "test.txt"
        artifact = Artifact(name="test", path=path, type="file")

        assert artifact.name == "test"
        assert artifact.path == path
        assert artifact.type == "file"
        assert artifact.metadata == {}
        assert artifact.created_at is not None

    def test_artifact_with_metadata(self, tmp_path):
        """Test creating artifact with metadata."""
        path = tmp_path / "data.json"
        metadata = {"size": 1024, "format": "json"}
        artifact = Artifact(
            name="data", path=path, type="json", metadata=metadata
        )

        assert artifact.metadata == metadata

    def test_artifact_timestamp(self, tmp_path):
        """Test that artifact has timestamp."""
        path = tmp_path / "file.txt"
        artifact = Artifact(name="file", path=path, type="file")

        assert isinstance(artifact.created_at, str)
        assert len(artifact.created_at) > 0


@pytest.mark.unit
class TestContextArtifacts:
    """Test context artifact management."""

    def test_add_artifact(self, tmp_path):
        """Test adding an artifact to context."""
        ctx = Context()
        path = tmp_path / "test.txt"

        artifact = ctx.add_artifact("test", path, "file")

        assert artifact.name == "test"
        assert artifact.path == path
        assert "test" in ctx.artifacts

    def test_add_artifact_with_metadata(self, tmp_path):
        """Test adding artifact with metadata."""
        ctx = Context()
        path = tmp_path / "data.json"

        artifact = ctx.add_artifact(
            "data", path, "json", size=100, format="json"
        )

        assert artifact.metadata["size"] == 100
        assert artifact.metadata["format"] == "json"

    def test_get_artifact(self, tmp_path):
        """Test getting artifact by name."""
        ctx = Context()
        path = tmp_path / "test.txt"
        ctx.add_artifact("test", path, "file")

        artifact = ctx.get_artifact("test")

        assert artifact is not None
        assert artifact.name == "test"

    def test_get_nonexistent_artifact(self):
        """Test getting artifact that doesn't exist."""
        ctx = Context()
        artifact = ctx.get_artifact("missing")

        assert artifact is None

    def test_list_artifacts(self, tmp_path):
        """Test listing all artifacts."""
        ctx = Context()
        ctx.add_artifact("file1", tmp_path / "f1.txt", "file")
        ctx.add_artifact("file2", tmp_path / "f2.txt", "file")
        ctx.add_artifact("file3", tmp_path / "f3.txt", "file")

        names = ctx.list_artifacts()

        assert len(names) == 3
        assert "file1" in names
        assert "file2" in names
        assert "file3" in names

    def test_list_artifacts_empty(self):
        """Test listing artifacts when none exist."""
        ctx = Context()
        names = ctx.list_artifacts()

        assert names == []

    def test_remove_artifact(self, tmp_path):
        """Test removing an artifact."""
        ctx = Context()
        ctx.add_artifact("test", tmp_path / "test.txt", "file")

        result = ctx.remove_artifact("test")

        assert result is True
        assert "test" not in ctx.artifacts

    def test_remove_nonexistent_artifact(self):
        """Test removing artifact that doesn't exist."""
        ctx = Context()
        result = ctx.remove_artifact("missing")

        assert result is False


@pytest.mark.unit
class TestContextSaveArtifact:
    """Test context save_artifact functionality."""

    def test_save_json_artifact(self, tmp_path):
        """Test saving JSON artifact."""
        ctx = Context(artifacts_dir=tmp_path)
        data = {"key": "value", "number": 42}

        artifact = ctx.save_artifact("test", data, "json")

        assert artifact.name == "test"
        assert artifact.path.exists()
        assert artifact.type == "json"

        # Verify content
        with open(artifact.path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_save_text_artifact(self, tmp_path):
        """Test saving text artifact."""
        ctx = Context(artifacts_dir=tmp_path)
        content = "Hello, World!"

        artifact = ctx.save_artifact("greeting", content, "text")

        assert artifact.type == "text"
        assert artifact.path.exists()

        # Verify content
        assert artifact.path.read_text() == content

    def test_save_yaml_artifact(self, tmp_path):
        """Test saving YAML artifact."""
        ctx = Context(artifacts_dir=tmp_path)
        data = {"name": "test", "values": [1, 2, 3]}

        artifact = ctx.save_artifact("config", data, "yaml")

        assert artifact.type == "yaml"
        assert artifact.path.exists()
        assert artifact.path.suffix == ".yaml"

    def test_save_artifact_with_metadata(self, tmp_path):
        """Test saving artifact with metadata."""
        ctx = Context(artifacts_dir=tmp_path)
        data = {"test": "data"}

        artifact = ctx.save_artifact(
            "test", data, "json", author="alice", version="1.0"
        )

        assert artifact.metadata["author"] == "alice"
        assert artifact.metadata["version"] == "1.0"

    def test_save_multiple_artifacts(self, tmp_path):
        """Test saving multiple artifacts."""
        ctx = Context(artifacts_dir=tmp_path)

        ctx.save_artifact("data1", {"a": 1}, "json")
        ctx.save_artifact("data2", {"b": 2}, "json")
        ctx.save_artifact("text", "hello", "text")

        assert len(ctx.artifacts) == 3


@pytest.mark.unit
class TestContextSteps:
    """Test context step result tracking."""

    def test_add_step_result(self):
        """Test adding a step result."""
        ctx = Context()
        result = Result(success=True, data={"value": 42})

        ctx.add_step_result("step1", result)

        assert "step1" in ctx.steps
        assert ctx.steps["step1"]["success"] is True
        assert ctx.steps["step1"]["outputs"] == {"value": 42}

    def test_add_step_result_updates_data(self):
        """Test that adding step result updates context data."""
        ctx = Context()
        result = Result(success=True, data={"result": 100})

        ctx.add_step_result("calculation", result)

        assert "calculation" in ctx.data
        assert ctx.data["calculation"] == {"result": 100}

    def test_get_step_output(self):
        """Test getting output from a step."""
        ctx = Context()
        result = Result(success=True, data={"value": 10})
        ctx.add_step_result("step1", result)

        output = ctx.get_step_output("step1")

        assert output == {"value": 10}

    def test_get_nonexistent_step_output(self):
        """Test getting output from non-existent step."""
        ctx = Context()
        output = ctx.get_step_output("missing")

        assert output is None

    def test_get_all_step_outputs(self):
        """Test getting all step outputs."""
        ctx = Context()
        ctx.add_step_result("step1", Result(success=True, data={"a": 1}))
        ctx.add_step_result("step2", Result(success=True, data={"b": 2}))

        all_outputs = ctx.get_all_step_outputs()

        assert len(all_outputs) == 2
        assert all_outputs["step1"] == {"a": 1}
        assert all_outputs["step2"] == {"b": 2}

    def test_multiple_steps(self):
        """Test adding multiple step results."""
        ctx = Context()

        for i in range(5):
            result = Result(success=True, data={"step": i})
            ctx.add_step_result(f"step{i}", result)

        assert len(ctx.steps) == 5


@pytest.mark.unit
class TestContextToResult:
    """Test converting context to Result."""

    def test_to_result_all_success(self):
        """Test to_result when all steps succeed."""
        ctx = Context()
        ctx.add_step_result("step1", Result(success=True, data={"a": 1}))
        ctx.add_step_result("step2", Result(success=True, data={"b": 2}))

        result = ctx.to_result()

        assert result.success is True
        assert "step1" in result.data
        assert "step2" in result.data

    def test_to_result_with_failure(self):
        """Test to_result when a step fails."""
        ctx = Context()
        ctx.add_step_result("step1", Result(success=True, data={"a": 1}))
        ctx.add_step_result("step2", Result(success=False, error="Failed"))

        result = ctx.to_result()

        assert result.success is False

    def test_to_result_metadata(self):
        """Test that to_result includes metadata."""
        inputs = {"input": "value"}
        ctx = Context(inputs=inputs, profile="test", backend="local")

        result = ctx.to_result()

        assert "inputs" in result.metadata
        assert result.metadata["inputs"] == inputs
        assert result.metadata["profile"] == "test"
        assert result.metadata["backend"] == "local"
        assert "duration" in result.metadata

    def test_to_result_includes_artifacts(self, tmp_path):
        """Test that to_result includes artifacts."""
        ctx = Context(artifacts_dir=tmp_path)
        ctx.add_artifact("test", tmp_path / "test.txt", "file")

        result = ctx.to_result()

        assert "artifacts" in result.metadata
        assert len(result.metadata["artifacts"]) == 1


@pytest.mark.unit
class TestContextToDict:
    """Test converting context to dictionary."""

    def test_to_dict_basic(self):
        """Test converting context to dict."""
        ctx = Context(inputs={"key": "value"})
        data = ctx.to_dict()

        assert isinstance(data, dict)
        assert data["inputs"] == {"key": "value"}
        assert "artifacts_dir" in data
        assert "profile" in data
        assert "backend" in data

    def test_to_dict_includes_all_fields(self):
        """Test that to_dict includes all context fields."""
        ctx = Context(
            inputs={"input": 1},
            profile="prod",
            backend="cloud",
            metadata={"meta": "data"},
        )

        data = ctx.to_dict()

        assert "inputs" in data
        assert "profile" in data
        assert "backend" in data
        assert "metadata" in data
        assert "artifacts" in data
        assert "start_time" in data
        assert "duration" in data
        assert "steps" in data

    def test_to_dict_with_artifacts(self, tmp_path):
        """Test to_dict includes artifacts."""
        ctx = Context(artifacts_dir=tmp_path)
        ctx.add_artifact("test1", tmp_path / "t1.txt", "file")
        ctx.add_artifact("test2", tmp_path / "t2.txt", "file")

        data = ctx.to_dict()

        assert len(data["artifacts"]) == 2

    def test_to_dict_with_steps(self):
        """Test to_dict includes steps."""
        ctx = Context()
        ctx.add_step_result("step1", Result(success=True, data={"a": 1}))

        data = ctx.to_dict()

        assert "steps" in data
        assert "step1" in data["steps"]


@pytest.mark.unit
class TestContextEdgeCases:
    """Test context edge cases."""

    def test_context_data_is_alias_for_inputs(self):
        """Test that data is an alias for inputs."""
        inputs = {"key": "value"}
        ctx = Context(inputs=inputs)

        assert ctx.data == ctx.inputs
        assert ctx.data is ctx.inputs

    def test_empty_context(self):
        """Test working with empty context."""
        ctx = Context()

        assert ctx.list_artifacts() == []
        assert ctx.get_all_step_outputs() == {}

    def test_context_start_time_valid(self):
        """Test that start_time is valid."""
        from datetime import datetime

        ctx = Context()

        assert isinstance(ctx.start_time, datetime)
        assert ctx.start_time <= DeterministicClock.utcnow()
