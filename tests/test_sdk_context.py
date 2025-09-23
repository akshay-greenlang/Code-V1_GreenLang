"""
Comprehensive tests for greenlang.sdk.context module
"""

import pytest
import json
import yaml
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.sdk.context import Artifact, Context
from greenlang.sdk.base import Result


class TestArtifact:
    """Test Artifact class"""

    def test_artifact_creation_basic(self):
        """Test creating Artifact with basic parameters"""
        artifact = Artifact(
            name="test_artifact",
            path=Path("/path/to/artifact"),
            type="file"
        )

        assert artifact.name == "test_artifact"
        assert artifact.path == Path("/path/to/artifact")
        assert artifact.type == "file"
        assert isinstance(artifact.metadata, dict)
        assert len(artifact.metadata) == 0
        assert isinstance(artifact.created_at, str)

    def test_artifact_creation_with_metadata(self):
        """Test creating Artifact with metadata"""
        metadata = {"author": "test", "size": 1024}
        artifact = Artifact(
            name="test_artifact",
            path=Path("/path/to/artifact"),
            type="json",
            metadata=metadata
        )

        assert artifact.metadata == metadata

    def test_artifact_creation_with_timestamp(self):
        """Test creating Artifact with explicit timestamp"""
        timestamp = "2023-01-01T12:00:00"
        artifact = Artifact(
            name="test_artifact",
            path=Path("/path/to/artifact"),
            type="file",
            created_at=timestamp
        )

        assert artifact.created_at == timestamp

    def test_artifact_automatic_timestamp(self):
        """Test that Artifact sets automatic timestamp when not provided"""
        artifact = Artifact(
            name="test_artifact",
            path=Path("/path/to/artifact"),
            type="file"
        )

        # Should have set a timestamp automatically
        assert artifact.created_at != ""
        # Should be in ISO format
        datetime.fromisoformat(artifact.created_at.replace('Z', '+00:00'))

    def test_artifact_pydantic_validation(self):
        """Test Pydantic validation of Artifact fields"""
        # Test with valid data
        artifact = Artifact(
            name="valid_name",
            path=Path("/valid/path"),
            type="valid_type"
        )
        assert artifact.name == "valid_name"

        # Test with invalid path type (should be converted to Path)
        artifact = Artifact(
            name="test",
            path="/string/path",
            type="file"
        )
        assert isinstance(artifact.path, Path)

    def test_artifact_model_dump(self):
        """Test Artifact model serialization"""
        artifact = Artifact(
            name="test",
            path=Path("/test/path"),
            type="json",
            metadata={"key": "value"}
        )

        # Test model_dump method (Pydantic v2)
        if hasattr(artifact, 'model_dump'):
            data = artifact.model_dump()
        else:
            # Fallback to dict() for Pydantic v1
            data = artifact.dict()

        assert isinstance(data, dict)
        assert data["name"] == "test"
        assert data["type"] == "json"
        assert data["metadata"] == {"key": "value"}

    def test_artifact_path_types(self):
        """Test Artifact with different path types"""
        # String path
        artifact1 = Artifact(name="test1", path="/string/path", type="file")
        assert isinstance(artifact1.path, Path)

        # Path object
        artifact2 = Artifact(name="test2", path=Path("/path/object"), type="file")
        assert isinstance(artifact2.path, Path)

        # Relative path
        artifact3 = Artifact(name="test3", path="relative/path", type="file")
        assert isinstance(artifact3.path, Path)


class TestContext:
    """Test Context class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

    def test_context_creation_default(self):
        """Test creating Context with default parameters"""
        context = Context()

        assert isinstance(context.inputs, dict)
        assert isinstance(context.data, dict)
        assert context.inputs is context.data  # Should be aliases
        assert context.artifacts_dir == Path("out")
        assert context.profile == "dev"
        assert context.backend == "local"
        assert isinstance(context.metadata, dict)
        assert isinstance(context.artifacts, dict)
        assert isinstance(context.start_time, datetime)
        assert isinstance(context.steps, dict)

    def test_context_creation_custom(self):
        """Test creating Context with custom parameters"""
        inputs = {"key": "value"}
        metadata = {"custom": "metadata"}
        artifacts_dir = self.temp_dir / "artifacts"

        context = Context(
            inputs=inputs,
            artifacts_dir=artifacts_dir,
            profile="prod",
            backend="k8s",
            metadata=metadata
        )

        assert context.inputs == inputs
        assert context.data == inputs
        assert context.artifacts_dir == artifacts_dir
        assert context.profile == "prod"
        assert context.backend == "k8s"
        assert context.metadata["custom"] == "metadata"

    @patch('pathlib.Path.mkdir')
    def test_context_creates_artifacts_directory(self, mock_mkdir):
        """Test that Context creates artifacts directory"""
        artifacts_dir = self.temp_dir / "new_artifacts"
        context = Context(artifacts_dir=artifacts_dir)

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_context_automatic_timestamp(self):
        """Test that Context sets automatic timestamp in metadata"""
        context = Context()

        assert "timestamp" in context.metadata
        # Should be a valid ISO timestamp
        datetime.fromisoformat(context.metadata["timestamp"])

    def test_context_custom_timestamp(self):
        """Test Context with custom timestamp in metadata"""
        custom_timestamp = "2023-01-01T12:00:00"
        metadata = {"timestamp": custom_timestamp}

        context = Context(metadata=metadata)

        assert context.metadata["timestamp"] == custom_timestamp

    def test_add_artifact_basic(self):
        """Test add_artifact method"""
        context = Context()
        artifact_path = Path("/test/artifact.json")

        artifact = context.add_artifact("test_artifact", artifact_path, "json")

        assert isinstance(artifact, Artifact)
        assert artifact.name == "test_artifact"
        assert artifact.path == artifact_path
        assert artifact.type == "json"
        assert "test_artifact" in context.artifacts
        assert context.artifacts["test_artifact"] is artifact

    def test_add_artifact_with_metadata(self):
        """Test add_artifact with custom metadata"""
        context = Context()
        artifact_path = Path("/test/artifact.json")
        custom_metadata = {"author": "test", "version": "1.0"}

        artifact = context.add_artifact(
            "test_artifact",
            artifact_path,
            "json",
            **custom_metadata
        )

        assert artifact.metadata == custom_metadata

    def test_get_artifact_existing(self):
        """Test get_artifact with existing artifact"""
        context = Context()
        artifact = context.add_artifact("test", Path("/test"), "file")

        result = context.get_artifact("test")
        assert result is artifact

    def test_get_artifact_nonexistent(self):
        """Test get_artifact with non-existent artifact"""
        context = Context()

        result = context.get_artifact("nonexistent")
        assert result is None

    def test_list_artifacts(self):
        """Test list_artifacts method"""
        context = Context()
        context.add_artifact("artifact1", Path("/test1"), "file")
        context.add_artifact("artifact2", Path("/test2"), "json")

        artifacts = context.list_artifacts()
        assert set(artifacts) == {"artifact1", "artifact2"}

    def test_remove_artifact_existing(self):
        """Test remove_artifact with existing artifact"""
        context = Context()
        context.add_artifact("test", Path("/test"), "file")

        result = context.remove_artifact("test")
        assert result is True
        assert "test" not in context.artifacts

    def test_remove_artifact_nonexistent(self):
        """Test remove_artifact with non-existent artifact"""
        context = Context()

        result = context.remove_artifact("nonexistent")
        assert result is False

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_artifact_json(self, mock_json_dump, mock_file):
        """Test save_artifact with JSON type"""
        context = Context(artifacts_dir=self.temp_dir)
        content = {"key": "value", "number": 42}

        artifact = context.save_artifact("test_data", content, "json")

        assert isinstance(artifact, Artifact)
        assert artifact.name == "test_data"
        assert artifact.type == "json"
        assert str(artifact.path).endswith("test_data.json")
        mock_json_dump.assert_called_once_with(content, mock_file.return_value, indent=2)

    @patch('builtins.open', new_callable=mock_open)
    @patch('yaml.dump')
    def test_save_artifact_yaml(self, mock_yaml_dump, mock_file):
        """Test save_artifact with YAML type"""
        context = Context(artifacts_dir=self.temp_dir)
        content = {"key": "value"}

        artifact = context.save_artifact("test_data", content, "yaml")

        assert artifact.type == "yaml"
        assert str(artifact.path).endswith("test_data.yaml")
        mock_yaml_dump.assert_called_once_with(content, mock_file.return_value)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_artifact_text(self, mock_file):
        """Test save_artifact with text type"""
        context = Context(artifacts_dir=self.temp_dir)
        content = "This is text content"

        artifact = context.save_artifact("test_text", content, "text")

        assert artifact.type == "text"
        assert str(artifact.path).endswith("test_text.txt")
        mock_file.return_value.write.assert_called_once_with(content)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_artifact_csv(self, mock_file):
        """Test save_artifact with CSV type"""
        context = Context(artifacts_dir=self.temp_dir)
        content = "col1,col2\nval1,val2"

        artifact = context.save_artifact("test_data", content, "csv")

        assert artifact.type == "csv"
        assert str(artifact.path).endswith("test_data.csv")

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    def test_save_artifact_unknown_type(self, mock_json_dump, mock_file):
        """Test save_artifact with unknown type defaults to JSON"""
        context = Context(artifacts_dir=self.temp_dir)
        content = {"data": "test"}

        artifact = context.save_artifact("test", content, "unknown")

        assert artifact.type == "unknown"
        assert str(artifact.path).endswith("test.dat")
        # Should default to JSON serialization
        mock_json_dump.assert_called_once()

    def test_add_step_result_with_result_object(self):
        """Test add_step_result with Result object"""
        context = Context()
        result = Result(
            success=True,
            data={"output": "test_value"},
            metadata={"duration": 1.5}
        )

        context.add_step_result("test_step", result)

        assert "test_step" in context.steps
        step = context.steps["test_step"]
        assert step["outputs"] == {"output": "test_value"}
        assert step["success"] is True
        assert step["metadata"] == {"duration": 1.5}

        # Should also update data
        assert context.data["test_step"] == {"output": "test_value"}

    def test_add_step_result_with_dict(self):
        """Test add_step_result with dictionary result"""
        context = Context()
        result_dict = {"output": "test", "status": "ok"}

        context.add_step_result("test_step", result_dict)

        step = context.steps["test_step"]
        assert step["outputs"] == result_dict
        assert step["success"] is True  # Default
        assert step["metadata"] == {}

    def test_add_step_result_with_failed_result(self):
        """Test add_step_result with failed Result"""
        context = Context()
        result = Result(
            success=False,
            data=None,
            error="Something went wrong"
        )

        context.add_step_result("failed_step", result)

        step = context.steps["failed_step"]
        assert step["success"] is False
        assert step["outputs"] is None

    def test_get_step_output_existing(self):
        """Test get_step_output with existing step"""
        context = Context()
        result = Result(success=True, data={"value": 42})
        context.add_step_result("test_step", result)

        output = context.get_step_output("test_step")
        assert output == {"value": 42}

    def test_get_step_output_nonexistent(self):
        """Test get_step_output with non-existent step"""
        context = Context()

        output = context.get_step_output("nonexistent")
        assert output is None

    def test_get_all_step_outputs(self):
        """Test get_all_step_outputs method"""
        context = Context()

        result1 = Result(success=True, data={"step1": "output1"})
        result2 = Result(success=True, data={"step2": "output2"})

        context.add_step_result("step1", result1)
        context.add_step_result("step2", result2)

        all_outputs = context.get_all_step_outputs()
        assert all_outputs["step1"] == {"step1": "output1"}
        assert all_outputs["step2"] == {"step2": "output2"}

    def test_get_all_step_outputs_empty(self):
        """Test get_all_step_outputs with no steps"""
        context = Context()

        all_outputs = context.get_all_step_outputs()
        assert all_outputs == {}

    def test_to_result_all_successful(self):
        """Test to_result with all successful steps"""
        context = Context()

        result1 = Result(success=True, data={"output1": "value1"})
        result2 = Result(success=True, data={"output2": "value2"})

        context.add_step_result("step1", result1)
        context.add_step_result("step2", result2)

        final_result = context.to_result()

        assert isinstance(final_result, Result)
        assert final_result.success is True
        assert "step1" in final_result.data
        assert "step2" in final_result.data

    def test_to_result_with_failures(self):
        """Test to_result with some failed steps"""
        context = Context()

        result1 = Result(success=True, data={"output1": "value1"})
        result2 = Result(success=False, error="Step failed")

        context.add_step_result("step1", result1)
        context.add_step_result("step2", result2)

        final_result = context.to_result()

        assert final_result.success is False

    def test_to_result_metadata(self):
        """Test to_result metadata content"""
        inputs = {"input_key": "input_value"}
        context = Context(
            inputs=inputs,
            profile="test",
            backend="local"
        )

        # Add an artifact
        context.add_artifact("test_artifact", Path("/test"), "file")

        final_result = context.to_result()

        metadata = final_result.metadata
        assert metadata["inputs"] == inputs
        assert metadata["profile"] == "test"
        assert metadata["backend"] == "local"
        assert "duration" in metadata
        assert "artifacts" in metadata
        assert len(metadata["artifacts"]) == 1

    def test_to_dict(self):
        """Test to_dict method"""
        inputs = {"key": "value"}
        context = Context(
            inputs=inputs,
            artifacts_dir=self.temp_dir,
            profile="test",
            backend="k8s",
            metadata={"custom": "meta"}
        )

        # Add step and artifact
        context.add_step_result("test_step", Result(success=True, data={"output": "test"}))
        context.add_artifact("test_artifact", Path("/test"), "file")

        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert context_dict["inputs"] == inputs
        assert context_dict["profile"] == "test"
        assert context_dict["backend"] == "k8s"
        assert context_dict["metadata"]["custom"] == "meta"
        assert "artifacts_dir" in context_dict
        assert "start_time" in context_dict
        assert "duration" in context_dict
        assert "steps" in context_dict
        assert "artifacts" in context_dict

    def test_to_dict_artifacts_serialization(self):
        """Test to_dict properly serializes artifacts"""
        context = Context()
        context.add_artifact("test", Path("/test"), "file", metadata={"size": 1024})

        context_dict = context.to_dict()

        artifacts = context_dict["artifacts"]
        assert len(artifacts) == 1
        artifact_dict = artifacts[0]
        assert artifact_dict["name"] == "test"
        assert artifact_dict["type"] == "file"
        assert artifact_dict["metadata"]["size"] == 1024

    def test_context_duration_calculation(self):
        """Test that duration is calculated correctly"""
        import time

        context = Context()
        start_time = context.start_time

        # Simulate some time passing
        time.sleep(0.1)

        result = context.to_result()
        duration = result.metadata["duration"]

        assert isinstance(duration, float)
        assert duration > 0
        assert duration >= 0.1

    def test_context_data_updates(self):
        """Test that context.data gets updated with step results"""
        context = Context(inputs={"initial": "data"})

        # Add a step result
        result = Result(success=True, data={"step_output": "value"})
        context.add_step_result("processing_step", result)

        # Context data should now include both initial and step data
        assert context.data["initial"] == "data"
        assert context.data["processing_step"] == {"step_output": "value"}

    def test_context_step_without_data(self):
        """Test adding step result without data"""
        context = Context()
        result = Result(success=True, data=None)

        context.add_step_result("empty_step", result)

        # Should not update context.data with None
        assert "empty_step" not in context.data or context.data["empty_step"] is None

    def test_context_artifacts_directory_path_types(self):
        """Test Context with different artifacts directory path types"""
        # String path
        context1 = Context(artifacts_dir="/string/path")
        assert isinstance(context1.artifacts_dir, Path)

        # Path object
        context2 = Context(artifacts_dir=Path("/path/object"))
        assert isinstance(context2.artifacts_dir, Path)

        # Relative path
        context3 = Context(artifacts_dir="relative/path")
        assert isinstance(context3.artifacts_dir, Path)

    def test_save_artifact_with_custom_metadata(self):
        """Test save_artifact with custom metadata"""
        context = Context(artifacts_dir=self.temp_dir)
        content = {"test": "data"}

        artifact = context.save_artifact(
            "test",
            content,
            "json",
            author="test_author",
            version="1.0",
            tags=["test", "data"]
        )

        assert artifact.metadata["author"] == "test_author"
        assert artifact.metadata["version"] == "1.0"
        assert artifact.metadata["tags"] == ["test", "data"]

    def test_context_multiple_artifacts_same_name(self):
        """Test adding multiple artifacts with same name (should overwrite)"""
        context = Context()

        artifact1 = context.add_artifact("test", Path("/path1"), "file")
        artifact2 = context.add_artifact("test", Path("/path2"), "json")

        # Should have overwritten the first artifact
        assert len(context.artifacts) == 1
        assert context.artifacts["test"] is artifact2
        assert context.artifacts["test"].path == Path("/path2")
        assert context.artifacts["test"].type == "json"

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    def test_save_artifact_file_error(self, mock_file):
        """Test save_artifact with file I/O error"""
        context = Context(artifacts_dir=self.temp_dir)
        content = {"test": "data"}

        # Should raise the IOError
        with pytest.raises(IOError):
            context.save_artifact("test", content, "json")

    def test_context_empty_steps(self):
        """Test to_result with no steps"""
        context = Context()

        result = context.to_result()

        assert result.success is True  # All (zero) steps succeeded
        assert result.data == {}

    def test_context_step_result_without_success_attribute(self):
        """Test add_step_result with object that doesn't have success attribute"""
        context = Context()
        simple_result = {"output": "test"}

        context.add_step_result("simple_step", simple_result)

        step = context.steps["simple_step"]
        assert step["success"] is True  # Should default to True
        assert step["outputs"] == simple_result

    def test_artifact_in_context_serialization(self):
        """Test that artifacts are properly serialized in context methods"""
        context = Context()
        artifact = context.add_artifact("test", Path("/test/path"), "json")

        # Test both to_result and to_dict
        result = context.to_result()
        context_dict = context.to_dict()

        # Check that artifacts are serialized in both
        assert len(result.metadata["artifacts"]) == 1
        assert len(context_dict["artifacts"]) == 1

        # Artifacts should be dictionaries, not Artifact objects
        assert isinstance(result.metadata["artifacts"][0], dict)
        assert isinstance(context_dict["artifacts"][0], dict)