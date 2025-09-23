"""
Comprehensive tests for greenlang.sdk.base module
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
from abc import ABC

# Add the greenlang directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "greenlang"))

from greenlang.sdk.base import (
    Status, Metadata, Result, Agent, Pipeline, Connector,
    Dataset, Report, Transform, Validator, TInput, TOutput, TConfig
)


class TestStatus:
    """Test Status enum"""

    def test_status_values(self):
        """Test that Status enum has correct values"""
        assert Status.PENDING == "pending"
        assert Status.RUNNING == "running"
        assert Status.SUCCESS == "success"
        assert Status.FAILED == "failed"
        assert Status.SKIPPED == "skipped"

    def test_status_string_inheritance(self):
        """Test that Status inherits from str"""
        assert isinstance(Status.SUCCESS, str)
        assert Status.SUCCESS == "success"

    def test_status_comparison(self):
        """Test Status comparison operations"""
        assert Status.SUCCESS == "success"
        assert Status.FAILED != "success"
        assert Status.PENDING in ["pending", "running"]


class TestMetadata:
    """Test Metadata dataclass"""

    def test_metadata_creation(self):
        """Test creating Metadata instance"""
        metadata = Metadata(id="test", name="Test Component")
        assert metadata.id == "test"
        assert metadata.name == "Test Component"
        assert metadata.version == "0.1.0"  # Default value

    def test_metadata_with_all_fields(self):
        """Test Metadata with all fields specified"""
        created_at = datetime.now()
        metadata = Metadata(
            id="component1",
            name="Component 1",
            version="1.2.3",
            description="Test component",
            author="Test Author",
            tags=["test", "component"],
            created_at=created_at,
            updated_at=created_at
        )

        assert metadata.id == "component1"
        assert metadata.name == "Component 1"
        assert metadata.version == "1.2.3"
        assert metadata.description == "Test component"
        assert metadata.author == "Test Author"
        assert metadata.tags == ["test", "component"]
        assert metadata.created_at == created_at

    def test_metadata_to_dict(self):
        """Test Metadata.to_dict() method"""
        metadata = Metadata(
            id="test",
            name="Test",
            version="1.0.0",
            tags=["tag1", "tag2"]
        )

        result = metadata.to_dict()
        assert isinstance(result, dict)
        assert result["id"] == "test"
        assert result["name"] == "Test"
        assert result["version"] == "1.0.0"
        assert result["tags"] == ["tag1", "tag2"]
        assert "created_at" in result
        assert "updated_at" in result

    def test_metadata_default_timestamps(self):
        """Test that Metadata creates default timestamps"""
        metadata = Metadata(id="test", name="Test")
        assert isinstance(metadata.created_at, datetime)
        assert isinstance(metadata.updated_at, datetime)

    def test_metadata_default_lists(self):
        """Test that Metadata initializes default lists correctly"""
        metadata = Metadata(id="test", name="Test")
        assert isinstance(metadata.tags, list)
        assert len(metadata.tags) == 0


class TestResult:
    """Test Result dataclass"""

    def test_result_creation_success(self):
        """Test creating successful Result"""
        result = Result(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert isinstance(result.metadata, dict)

    def test_result_creation_failure(self):
        """Test creating failed Result"""
        result = Result(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.data is None

    def test_result_with_metadata(self):
        """Test Result with custom metadata"""
        metadata = {"duration": 1.5, "agent": "test"}
        result = Result(success=True, metadata=metadata)
        assert result.metadata == metadata

    def test_result_to_dict(self):
        """Test Result.to_dict() method"""
        result = Result(
            success=True,
            data={"output": "test"},
            metadata={"duration": 2.0}
        )

        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["success"] is True
        assert result_dict["data"] == {"output": "test"}
        assert result_dict["error"] is None
        assert result_dict["metadata"] == {"duration": 2.0}


class ConcreteAgent(Agent[dict, dict]):
    """Concrete Agent implementation for testing"""

    def validate(self, input_data: dict) -> bool:
        return isinstance(input_data, dict) and "required_field" in input_data

    def process(self, input_data: dict) -> dict:
        return {"processed": input_data["required_field"]}


class TestAgent:
    """Test Agent abstract base class"""

    def test_agent_is_abstract(self):
        """Test that Agent is abstract"""
        assert issubclass(Agent, ABC)

        with pytest.raises(TypeError):
            Agent()

    def test_concrete_agent_creation(self):
        """Test creating concrete Agent"""
        agent = ConcreteAgent()
        assert agent is not None
        assert isinstance(agent.metadata, Metadata)

    def test_agent_with_custom_metadata(self):
        """Test Agent with custom metadata"""
        metadata = Metadata(id="custom", name="Custom Agent", version="2.0.0")
        agent = ConcreteAgent(metadata=metadata)
        assert agent.metadata == metadata

    def test_agent_validate_method(self):
        """Test Agent validate method"""
        agent = ConcreteAgent()

        # Valid input
        valid_input = {"required_field": "value"}
        assert agent.validate(valid_input) is True

        # Invalid input
        invalid_input = {"other_field": "value"}
        assert agent.validate(invalid_input) is False

    def test_agent_process_method(self):
        """Test Agent process method"""
        agent = ConcreteAgent()
        input_data = {"required_field": "test_value"}
        output = agent.process(input_data)

        assert output == {"processed": "test_value"}

    def test_agent_run_method_success(self):
        """Test Agent run method with valid input"""
        agent = ConcreteAgent()
        input_data = {"required_field": "test_value"}

        result = agent.run(input_data)
        assert isinstance(result, Result)
        assert result.success is True
        assert result.data == {"processed": "test_value"}
        assert result.error is None

    def test_agent_run_method_validation_failure(self):
        """Test Agent run method with invalid input"""
        agent = ConcreteAgent()
        input_data = {"invalid_field": "value"}

        result = agent.run(input_data)
        assert isinstance(result, Result)
        assert result.success is False
        assert result.error == "Input validation failed"
        assert result.data is None

    def test_agent_run_method_processing_error(self):
        """Test Agent run method with processing error"""

        class FailingAgent(Agent[dict, dict]):
            def validate(self, input_data: dict) -> bool:
                return True

            def process(self, input_data: dict) -> dict:
                raise ValueError("Processing failed")

        agent = FailingAgent()
        result = agent.run({"test": "data"})

        assert isinstance(result, Result)
        assert result.success is False
        assert "Processing failed" in result.error

    def test_agent_describe_method(self):
        """Test Agent describe method"""
        agent = ConcreteAgent()
        description = agent.describe()

        assert isinstance(description, dict)
        assert "metadata" in description
        assert "input_schema" in description
        assert "output_schema" in description

    def test_agent_schema_methods(self):
        """Test Agent schema methods"""
        agent = ConcreteAgent()

        input_schema = agent.get_input_schema()
        output_schema = agent.get_output_schema()

        assert isinstance(input_schema, dict)
        assert isinstance(output_schema, dict)
        assert input_schema.get("type") == "object"
        assert output_schema.get("type") == "object"

    def test_agent_metadata_in_result(self):
        """Test that Agent includes metadata in result"""
        agent = ConcreteAgent()
        input_data = {"required_field": "test"}

        result = agent.run(input_data)
        assert "agent" in result.metadata
        assert "version" in result.metadata
        assert result.metadata["agent"] == agent.metadata.id


class ConcretePipeline(Pipeline):
    """Concrete Pipeline implementation for testing"""

    def execute(self, input_data) -> Result:
        # Simple execution that runs all agents in sequence
        current_data = input_data
        for agent in self.agents:
            agent_result = agent.run(current_data)
            if not agent_result.success:
                return agent_result
            current_data = agent_result.data

        return Result(success=True, data=current_data)


class TestPipeline:
    """Test Pipeline abstract base class"""

    def test_pipeline_is_abstract(self):
        """Test that Pipeline is abstract"""
        assert issubclass(Pipeline, ABC)

        with pytest.raises(TypeError):
            Pipeline()

    def test_concrete_pipeline_creation(self):
        """Test creating concrete Pipeline"""
        pipeline = ConcretePipeline()
        assert pipeline is not None
        assert isinstance(pipeline.metadata, Metadata)
        assert isinstance(pipeline.agents, list)
        assert len(pipeline.agents) == 0

    def test_pipeline_add_agent(self):
        """Test Pipeline add_agent method"""
        pipeline = ConcretePipeline()
        agent = ConcreteAgent()

        result = pipeline.add_agent(agent)
        assert result is pipeline  # Should return self for chaining
        assert len(pipeline.agents) == 1
        assert pipeline.agents[0] is agent

    def test_pipeline_add_multiple_agents(self):
        """Test adding multiple agents to pipeline"""
        pipeline = ConcretePipeline()
        agent1 = ConcreteAgent()
        agent2 = ConcreteAgent()

        pipeline.add_agent(agent1).add_agent(agent2)
        assert len(pipeline.agents) == 2

    def test_pipeline_execute_method(self):
        """Test Pipeline execute method"""
        pipeline = ConcretePipeline()
        agent = ConcreteAgent()
        pipeline.add_agent(agent)

        input_data = {"required_field": "test"}
        result = pipeline.execute(input_data)

        assert isinstance(result, Result)
        assert result.success is True

    def test_pipeline_describe_method(self):
        """Test Pipeline describe method"""
        pipeline = ConcretePipeline()
        agent = ConcreteAgent()
        pipeline.add_agent(agent)

        description = pipeline.describe()
        assert isinstance(description, dict)
        assert "metadata" in description
        assert "agents" in description
        assert "flow" in description
        assert len(description["agents"]) == 1

    def test_pipeline_get_flow_method(self):
        """Test Pipeline get_flow method"""
        pipeline = ConcretePipeline()
        flow = pipeline.get_flow()

        assert isinstance(flow, dict)
        assert flow.get("type") == "sequential"


class ConcreteConnector(Connector):
    """Concrete Connector implementation for testing"""

    def connect(self) -> bool:
        self.connected = True
        return True

    def disconnect(self) -> bool:
        self.connected = False
        return True

    def read(self, query) -> Result:
        if not self.connected:
            return Result(success=False, error="Not connected")
        return Result(success=True, data={"query": query, "data": "mock_data"})

    def write(self, data) -> Result:
        if not self.connected:
            return Result(success=False, error="Not connected")
        return Result(success=True, data={"written": data})


class TestConnector:
    """Test Connector abstract base class"""

    def test_connector_is_abstract(self):
        """Test that Connector is abstract"""
        assert issubclass(Connector, ABC)

        with pytest.raises(TypeError):
            Connector()

    def test_concrete_connector_creation(self):
        """Test creating concrete Connector"""
        connector = ConcreteConnector()
        assert connector is not None
        assert connector.connected is False

    def test_connector_with_config(self):
        """Test Connector with configuration"""
        config = {"host": "localhost", "port": 5432}
        connector = ConcreteConnector(config=config)
        assert connector.config == config

    def test_connector_connect_disconnect(self):
        """Test Connector connect/disconnect methods"""
        connector = ConcreteConnector()

        # Initially not connected
        assert connector.is_connected() is False

        # Connect
        result = connector.connect()
        assert result is True
        assert connector.is_connected() is True

        # Disconnect
        result = connector.disconnect()
        assert result is True
        assert connector.is_connected() is False

    def test_connector_read_method(self):
        """Test Connector read method"""
        connector = ConcreteConnector()

        # Read without connection should fail
        result = connector.read("SELECT * FROM table")
        assert result.success is False

        # Read with connection should succeed
        connector.connect()
        result = connector.read("SELECT * FROM table")
        assert result.success is True
        assert "data" in result.data

    def test_connector_write_method(self):
        """Test Connector write method"""
        connector = ConcreteConnector()
        data = {"key": "value"}

        # Write without connection should fail
        result = connector.write(data)
        assert result.success is False

        # Write with connection should succeed
        connector.connect()
        result = connector.write(data)
        assert result.success is True

    def test_connector_context_manager(self):
        """Test Connector as context manager"""
        connector = ConcreteConnector()

        with connector:
            assert connector.is_connected() is True

        assert connector.is_connected() is False

    def test_connector_context_manager_with_exception(self):
        """Test Connector context manager with exception"""
        connector = ConcreteConnector()

        try:
            with connector:
                assert connector.is_connected() is True
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still disconnect even with exception
        assert connector.is_connected() is False


class ConcreteDataset(Dataset):
    """Concrete Dataset implementation for testing"""

    def __init__(self, data=None, metadata=None):
        super().__init__(metadata)
        self._data = data or {"sample": "data"}

    def load(self):
        return self._data

    def save(self, data) -> bool:
        self._data = data
        return True

    def describe(self) -> dict:
        return {
            "rows": len(self._data) if isinstance(self._data, list) else 1,
            "columns": list(self._data.keys()) if isinstance(self._data, dict) else [],
            "type": type(self._data).__name__
        }


class TestDataset:
    """Test Dataset abstract base class"""

    def test_dataset_is_abstract(self):
        """Test that Dataset is abstract"""
        assert issubclass(Dataset, ABC)

        with pytest.raises(TypeError):
            Dataset()

    def test_concrete_dataset_creation(self):
        """Test creating concrete Dataset"""
        dataset = ConcreteDataset()
        assert dataset is not None
        assert isinstance(dataset.metadata, Metadata)

    def test_dataset_load_method(self):
        """Test Dataset load method"""
        test_data = {"test": "data", "numbers": [1, 2, 3]}
        dataset = ConcreteDataset(data=test_data)

        loaded_data = dataset.load()
        assert loaded_data == test_data

    def test_dataset_save_method(self):
        """Test Dataset save method"""
        dataset = ConcreteDataset()
        new_data = {"new": "data"}

        result = dataset.save(new_data)
        assert result is True
        assert dataset.load() == new_data

    def test_dataset_describe_method(self):
        """Test Dataset describe method"""
        dataset = ConcreteDataset()
        description = dataset.describe()

        assert isinstance(description, dict)
        assert "rows" in description or "type" in description

    def test_dataset_get_card_method(self):
        """Test Dataset get_card method"""
        dataset = ConcreteDataset()
        card = dataset.get_card()
        assert card is None  # Default implementation

    def test_dataset_get_schema_method(self):
        """Test Dataset get_schema method"""
        dataset = ConcreteDataset()
        schema = dataset.get_schema()

        assert isinstance(schema, dict)
        assert schema.get("type") == "object"

    def test_dataset_get_stats_method(self):
        """Test Dataset get_stats method"""
        dataset = ConcreteDataset()
        stats = dataset.get_stats()

        assert isinstance(stats, dict)


class ConcreteReport(Report):
    """Concrete Report implementation for testing"""

    def generate(self, data, format="markdown") -> str:
        if format == "markdown":
            return f"# Report\n\nData: {data}"
        elif format == "json":
            import json
            return json.dumps({"report": {"data": data}})
        else:
            return f"Report: {data}"

    def save(self, content: str, path: Path) -> bool:
        try:
            with open(path, 'w') as f:
                f.write(content)
            return True
        except Exception:
            return False


class TestReport:
    """Test Report abstract base class"""

    def test_report_is_abstract(self):
        """Test that Report is abstract"""
        assert issubclass(Report, ABC)

        with pytest.raises(TypeError):
            Report()

    def test_concrete_report_creation(self):
        """Test creating concrete Report"""
        report = ConcreteReport()
        assert report is not None
        assert isinstance(report.metadata, Metadata)

    def test_report_generate_method(self):
        """Test Report generate method"""
        report = ConcreteReport()
        data = {"metric": 42, "status": "success"}

        # Test markdown format
        markdown_output = report.generate(data, format="markdown")
        assert "# Report" in markdown_output
        assert str(data) in markdown_output

        # Test JSON format
        json_output = report.generate(data, format="json")
        assert "report" in json_output

    def test_report_save_method(self):
        """Test Report save method"""
        report = ConcreteReport()
        content = "# Test Report\n\nThis is a test."

        with patch("builtins.open", create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = MagicMock()

            result = report.save(content, Path("test_report.md"))
            assert result is True

    def test_report_get_templates_method(self):
        """Test Report get_templates method"""
        report = ConcreteReport()
        templates = report.get_templates()

        assert isinstance(templates, list)
        assert "default" in templates

    def test_report_get_formats_method(self):
        """Test Report get_formats method"""
        report = ConcreteReport()
        formats = report.get_formats()

        assert isinstance(formats, list)
        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats


class ConcreteTransform(Transform[dict, str]):
    """Concrete Transform implementation for testing"""

    def apply(self, data: dict) -> str:
        return str(data)


class TestTransform:
    """Test Transform abstract base class"""

    def test_transform_is_abstract(self):
        """Test that Transform is abstract"""
        assert issubclass(Transform, ABC)

        with pytest.raises(TypeError):
            Transform()

    def test_concrete_transform_creation(self):
        """Test creating concrete Transform"""
        transform = ConcreteTransform()
        assert transform is not None

    def test_transform_apply_method(self):
        """Test Transform apply method"""
        transform = ConcreteTransform()
        input_data = {"key": "value", "number": 42}

        result = transform.apply(input_data)
        assert isinstance(result, str)
        assert "key" in result

    def test_transform_callable(self):
        """Test that Transform is callable"""
        transform = ConcreteTransform()
        input_data = {"test": "data"}

        # Should be callable directly
        result = transform(input_data)
        assert isinstance(result, str)

        # Should be same as apply method
        apply_result = transform.apply(input_data)
        assert result == apply_result


class ConcreteValidator(Validator[dict]):
    """Concrete Validator implementation for testing"""

    def validate(self, data: dict) -> Result:
        if not isinstance(data, dict):
            return Result(success=False, error="Data must be a dictionary")

        if "required_field" not in data:
            return Result(success=False, error="Missing required_field")

        return Result(success=True, data=data)


class TestValidator:
    """Test Validator abstract base class"""

    def test_validator_is_abstract(self):
        """Test that Validator is abstract"""
        assert issubclass(Validator, ABC)

        with pytest.raises(TypeError):
            Validator()

    def test_concrete_validator_creation(self):
        """Test creating concrete Validator"""
        validator = ConcreteValidator()
        assert validator is not None

    def test_validator_validate_method_success(self):
        """Test Validator validate method with valid data"""
        validator = ConcreteValidator()
        valid_data = {"required_field": "value", "optional": "data"}

        result = validator.validate(valid_data)
        assert isinstance(result, Result)
        assert result.success is True

    def test_validator_validate_method_failure(self):
        """Test Validator validate method with invalid data"""
        validator = ConcreteValidator()

        # Test with wrong type
        result = validator.validate("not a dict")
        assert result.success is False
        assert "dictionary" in result.error

        # Test with missing field
        result = validator.validate({"other_field": "value"})
        assert result.success is False
        assert "required_field" in result.error

    def test_validator_callable(self):
        """Test that Validator is callable"""
        validator = ConcreteValidator()

        # Valid data
        valid_data = {"required_field": "value"}
        assert validator(valid_data) is True

        # Invalid data
        invalid_data = {"other_field": "value"}
        assert validator(invalid_data) is False


class TestTypeVariables:
    """Test type variables"""

    def test_type_variables_exist(self):
        """Test that type variables are defined"""
        assert TInput is not None
        assert TOutput is not None
        assert TConfig is not None

    def test_type_variables_usage(self):
        """Test that type variables can be used in generic classes"""

        class GenericAgent(Agent[str, int]):
            def validate(self, input_data: str) -> bool:
                return isinstance(input_data, str)

            def process(self, input_data: str) -> int:
                return len(input_data)

        agent = GenericAgent()
        result = agent.run("test")
        assert result.success is True
        assert result.data == 4