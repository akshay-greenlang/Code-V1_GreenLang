"""
GreenLang Framework - Tool Export Utilities Tests

Comprehensive test suite for tool export utilities including:
- OpenAI function format export
- Anthropic Claude tool format export
- Qwen-Agent tool format export
- OpenAPI schema generation
- JSON Schema export
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, Any

# Import modules under test
import sys
from pathlib import Path

# Add parent paths for imports
_framework_path = Path(__file__).parent.parent.parent
if str(_framework_path) not in sys.path:
    sys.path.insert(0, str(_framework_path))

from advanced.mcp_protocol import (
    ToolDefinition,
    ToolParameter,
    ToolCategory,
    SecurityLevel,
    ExecutionMode,
)
from tools.tool_export import (
    # Enums
    ExportFormat,
    # Data models
    ExportResult,
    OpenAPISchema,
    # Exporter classes
    ToolExporter,
    BatchExporter,
    # Convenience functions
    export_to_openai,
    export_to_anthropic,
    export_to_qwen,
    export_to_openapi,
    export_to_json_schema,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_calculator_tool():
    """Create a sample calculator tool definition."""
    return ToolDefinition(
        name="calculate_efficiency",
        description="Calculate thermal efficiency of a boiler system",
        parameters=[
            ToolParameter(
                name="fuel_input_kw",
                type="number",
                description="Fuel input power in kW",
                required=True,
                minimum=0,
            ),
            ToolParameter(
                name="steam_output_kw",
                type="number",
                description="Steam output power in kW",
                required=True,
                minimum=0,
            ),
            ToolParameter(
                name="include_losses",
                type="boolean",
                description="Include detailed loss breakdown",
                required=False,
                default=True,
            ),
        ],
        category=ToolCategory.CALCULATOR,
        security_level=SecurityLevel.READ_ONLY,
        version="1.0.0",
    )


@pytest.fixture
def sample_connector_tool():
    """Create a sample connector tool definition."""
    return ToolDefinition(
        name="read_sensor_data",
        description="Read data from industrial sensors via SCADA",
        parameters=[
            ToolParameter(
                name="sensor_ids",
                type="array",
                description="List of sensor IDs to read",
                required=True,
            ),
            ToolParameter(
                name="format",
                type="string",
                description="Output format",
                required=False,
                default="json",
                enum=["json", "csv", "xml"],
            ),
        ],
        category=ToolCategory.CONNECTOR,
        security_level=SecurityLevel.CONTROLLED_WRITE,
        requires_confirmation=True,
        version="2.0.0",
    )


@pytest.fixture
def exporter():
    """Create a tool exporter instance."""
    return ToolExporter()


@pytest.fixture
def tool_list(sample_calculator_tool, sample_connector_tool):
    """Create a list of sample tools."""
    return [sample_calculator_tool, sample_connector_tool]


# =============================================================================
# EXPORT RESULT TESTS
# =============================================================================

class TestExportResult:
    """Test ExportResult data model."""

    def test_export_result_creation(self):
        """Test creating an export result."""
        result = ExportResult(
            format=ExportFormat.OPENAI,
            tools=[{"name": "test"}],
            tool_count=1,
            exported_at=datetime.now(timezone.utc),
            version="1.0.0",
            provenance_hash="abc123",
        )

        assert result.format == ExportFormat.OPENAI
        assert result.tool_count == 1

    def test_export_result_to_dict(self):
        """Test converting export result to dictionary."""
        now = datetime.now(timezone.utc)
        result = ExportResult(
            format=ExportFormat.ANTHROPIC,
            tools=[{"name": "test"}],
            tool_count=1,
            exported_at=now,
            version="1.0.0",
            provenance_hash="abc123",
        )

        result_dict = result.to_dict()
        assert result_dict["format"] == "anthropic"
        assert result_dict["tool_count"] == 1
        assert result_dict["version"] == "1.0.0"

    def test_export_result_to_json(self):
        """Test converting export result to JSON."""
        result = ExportResult(
            format=ExportFormat.QWEN,
            tools=[{"name": "test"}],
            tool_count=1,
            exported_at=datetime.now(timezone.utc),
            version="1.0.0",
            provenance_hash="abc123",
        )

        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed["format"] == "qwen"


# =============================================================================
# OPENAI EXPORT TESTS
# =============================================================================

class TestOpenAIExport:
    """Test OpenAI function format export."""

    def test_basic_export(self, exporter, sample_calculator_tool):
        """Test basic OpenAI export."""
        result = exporter.to_openai([sample_calculator_tool])

        assert len(result) == 1
        func = result[0]
        assert func["type"] == "function"
        assert func["function"]["name"] == "calculate_efficiency"

    def test_parameters_structure(self, exporter, sample_calculator_tool):
        """Test OpenAI parameters structure."""
        result = exporter.to_openai([sample_calculator_tool])
        params = result[0]["function"]["parameters"]

        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        assert "fuel_input_kw" in params["properties"]
        assert "fuel_input_kw" in params["required"]

    def test_parameter_types(self, exporter, sample_calculator_tool):
        """Test parameter types are correctly mapped."""
        result = exporter.to_openai([sample_calculator_tool])
        props = result[0]["function"]["parameters"]["properties"]

        assert props["fuel_input_kw"]["type"] == "number"
        assert props["include_losses"]["type"] == "boolean"

    def test_parameter_constraints(self, exporter, sample_calculator_tool):
        """Test parameter constraints are included."""
        result = exporter.to_openai([sample_calculator_tool])
        props = result[0]["function"]["parameters"]["properties"]

        assert props["fuel_input_kw"]["minimum"] == 0
        assert props["include_losses"]["default"] is True

    def test_strict_mode(self, exporter, sample_calculator_tool):
        """Test strict mode adds required fields."""
        result = exporter.to_openai([sample_calculator_tool], strict=True)

        assert result[0]["function"]["strict"] is True
        assert result[0]["function"]["parameters"]["additionalProperties"] is False

    def test_metadata_included(self, sample_calculator_tool):
        """Test metadata is included when enabled."""
        exporter = ToolExporter(include_metadata=True)
        result = exporter.to_openai([sample_calculator_tool])

        metadata = result[0]["function"]["_greenlang_metadata"]
        assert metadata["category"] == "calculator"
        assert metadata["version"] == "1.0.0"


# =============================================================================
# ANTHROPIC EXPORT TESTS
# =============================================================================

class TestAnthropicExport:
    """Test Anthropic Claude tool format export."""

    def test_basic_export(self, exporter, sample_calculator_tool):
        """Test basic Anthropic export."""
        result = exporter.to_anthropic([sample_calculator_tool])

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "calculate_efficiency"
        assert "input_schema" in tool

    def test_input_schema_structure(self, exporter, sample_calculator_tool):
        """Test Anthropic input schema structure."""
        result = exporter.to_anthropic([sample_calculator_tool])
        schema = result[0]["input_schema"]

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_includes_cache_control(self, sample_calculator_tool):
        """Test cache control is included."""
        exporter = ToolExporter(include_metadata=True)
        result = exporter.to_anthropic([sample_calculator_tool])

        assert "cache_control" in result[0]
        assert result[0]["cache_control"]["type"] == "ephemeral"

    def test_includes_examples(self, exporter, sample_calculator_tool):
        """Test examples can be included in descriptions."""
        result = exporter.to_anthropic([sample_calculator_tool], include_examples=True)
        props = result[0]["input_schema"]["properties"]

        # Default value should be in description as example
        assert "(example:" in props["include_losses"]["description"]


# =============================================================================
# QWEN EXPORT TESTS
# =============================================================================

class TestQwenExport:
    """Test Qwen-Agent tool format export."""

    def test_basic_export(self, exporter, sample_calculator_tool):
        """Test basic Qwen export."""
        result = exporter.to_qwen([sample_calculator_tool])

        assert len(result) == 1
        tool = result[0]
        assert tool["name"] == "calculate_efficiency"
        assert "parameters" in tool
        assert isinstance(tool["parameters"], list)

    def test_human_name(self, exporter, sample_calculator_tool):
        """Test human-readable name is generated."""
        result = exporter.to_qwen([sample_calculator_tool], include_human_names=True)

        assert result[0]["name_for_human"] == "Calculate Efficiency"

    def test_parameter_format(self, exporter, sample_calculator_tool):
        """Test Qwen parameter format."""
        result = exporter.to_qwen([sample_calculator_tool])
        params = result[0]["parameters"]

        assert len(params) == 3
        param = next(p for p in params if p["name"] == "fuel_input_kw")
        assert param["type"] == "number"
        assert param["required"] is True

    def test_enum_values(self, exporter, sample_connector_tool):
        """Test enum values are included."""
        result = exporter.to_qwen([sample_connector_tool])
        params = result[0]["parameters"]

        format_param = next(p for p in params if p["name"] == "format")
        assert format_param["enum"] == ["json", "csv", "xml"]


# =============================================================================
# OPENAPI EXPORT TESTS
# =============================================================================

class TestOpenAPIExport:
    """Test OpenAPI 3.0 schema export."""

    def test_basic_structure(self, exporter, tool_list):
        """Test basic OpenAPI structure."""
        result = exporter.to_openapi(tool_list, title="Test API")

        assert result["openapi"] == "3.0.3"
        assert result["info"]["title"] == "Test API"
        assert "paths" in result
        assert "components" in result

    def test_paths_created(self, exporter, tool_list):
        """Test paths are created for each tool."""
        result = exporter.to_openapi(tool_list)
        paths = result["paths"]

        assert "/tools/calculate_efficiency" in paths
        assert "/tools/read_sensor_data" in paths

    def test_post_method(self, exporter, sample_calculator_tool):
        """Test POST method is used."""
        result = exporter.to_openapi([sample_calculator_tool])
        path = result["paths"]["/tools/calculate_efficiency"]

        assert "post" in path
        assert path["post"]["operationId"] == "calculate_efficiency"

    def test_request_body(self, exporter, sample_calculator_tool):
        """Test request body schema."""
        result = exporter.to_openapi([sample_calculator_tool])
        post = result["paths"]["/tools/calculate_efficiency"]["post"]

        assert "requestBody" in post
        assert post["requestBody"]["required"] is True
        assert "application/json" in post["requestBody"]["content"]

    def test_responses(self, exporter, sample_calculator_tool):
        """Test response schemas."""
        result = exporter.to_openapi([sample_calculator_tool])
        responses = result["paths"]["/tools/calculate_efficiency"]["post"]["responses"]

        assert "200" in responses
        assert "400" in responses
        assert "500" in responses

    def test_tags_by_category(self, exporter, tool_list):
        """Test tools are tagged by category."""
        result = exporter.to_openapi(tool_list)

        tags = [t["name"] for t in result["tags"]]
        assert "calculator" in tags
        assert "connector" in tags

    def test_security_by_level(self, exporter, tool_list):
        """Test security varies by security level."""
        result = exporter.to_openapi(tool_list)

        calc_security = result["paths"]["/tools/calculate_efficiency"]["post"]["security"]
        conn_security = result["paths"]["/tools/read_sensor_data"]["post"]["security"]

        # Read-only should have no security
        assert calc_security == []
        # Controlled write should require auth
        assert len(conn_security) > 0

    def test_schemas_generated(self, exporter, sample_calculator_tool):
        """Test component schemas are generated."""
        result = exporter.to_openapi([sample_calculator_tool])
        schemas = result["components"]["schemas"]

        assert "calculate_efficiency_request" in schemas
        assert "calculate_efficiency_response" in schemas
        assert "error_response" in schemas


# =============================================================================
# JSON SCHEMA EXPORT TESTS
# =============================================================================

class TestJSONSchemaExport:
    """Test JSON Schema export."""

    def test_basic_export(self, exporter, sample_calculator_tool):
        """Test basic JSON Schema export."""
        result = exporter.to_json_schema([sample_calculator_tool])

        assert len(result) == 1
        schema = result[0]
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"

    def test_schema_id(self, exporter, sample_calculator_tool):
        """Test schema ID is set."""
        result = exporter.to_json_schema([sample_calculator_tool])
        schema = result[0]

        assert schema["$id"] == "urn:greenlang:tool:calculate_efficiency"

    def test_title_description(self, exporter, sample_calculator_tool):
        """Test title and description."""
        result = exporter.to_json_schema([sample_calculator_tool])
        schema = result[0]

        assert schema["title"] == "calculate_efficiency"
        assert "thermal efficiency" in schema["description"]

    def test_properties(self, exporter, sample_calculator_tool):
        """Test properties are defined."""
        result = exporter.to_json_schema([sample_calculator_tool])
        schema = result[0]

        assert "properties" in schema
        assert "fuel_input_kw" in schema["properties"]
        assert "required" in schema
        assert "fuel_input_kw" in schema["required"]

    def test_additional_properties_false(self, exporter, sample_calculator_tool):
        """Test additional properties is false."""
        result = exporter.to_json_schema([sample_calculator_tool])
        schema = result[0]

        assert schema["additionalProperties"] is False


# =============================================================================
# MCP NATIVE EXPORT TESTS
# =============================================================================

class TestMCPNativeExport:
    """Test MCP native format export."""

    def test_basic_export(self, exporter, sample_calculator_tool):
        """Test basic MCP native export."""
        result = exporter.to_mcp_native([sample_calculator_tool])

        assert "tools" in result
        assert len(result["tools"]) == 1

    def test_tool_structure(self, exporter, sample_calculator_tool):
        """Test MCP tool structure."""
        result = exporter.to_mcp_native([sample_calculator_tool])
        tool = result["tools"][0]

        assert tool["name"] == "calculate_efficiency"
        assert "inputSchema" in tool
        assert tool["inputSchema"]["type"] == "object"

    def test_server_info(self, exporter, sample_calculator_tool):
        """Test server info is included."""
        result = exporter.to_mcp_native([sample_calculator_tool], include_server_info=True)

        assert "_meta" in result
        assert "serverInfo" in result["_meta"]
        assert result["_meta"]["protocolVersion"] == "2025-06-18"


# =============================================================================
# EXPORTER CLASS TESTS
# =============================================================================

class TestToolExporter:
    """Test ToolExporter class."""

    def test_export_method(self, exporter, tool_list):
        """Test generic export method."""
        result = exporter.export(tool_list, ExportFormat.OPENAI)

        assert isinstance(result, ExportResult)
        assert result.format == ExportFormat.OPENAI
        assert result.tool_count == 2
        assert result.provenance_hash != ""

    def test_export_timestamp(self, exporter, sample_calculator_tool):
        """Test export includes timestamp."""
        result = exporter.export([sample_calculator_tool], ExportFormat.ANTHROPIC)

        assert result.exported_at is not None
        assert result.exported_at <= datetime.now(timezone.utc)

    def test_export_metadata(self, sample_calculator_tool):
        """Test export includes metadata."""
        exporter = ToolExporter(include_metadata=True)
        result = exporter.export([sample_calculator_tool], ExportFormat.QWEN)

        assert result.metadata["include_metadata"] is True


# =============================================================================
# BATCH EXPORTER TESTS
# =============================================================================

class TestBatchExporter:
    """Test batch export functionality."""

    def test_export_all_formats(self, tool_list):
        """Test exporting to all formats."""
        batch = BatchExporter()
        results = batch.export_all(tool_list)

        assert len(results) == len(ExportFormat)
        assert ExportFormat.OPENAI in results
        assert ExportFormat.ANTHROPIC in results

    def test_export_selected_formats(self, tool_list):
        """Test exporting to selected formats."""
        batch = BatchExporter()
        formats = [ExportFormat.OPENAI, ExportFormat.ANTHROPIC]
        results = batch.export_all(tool_list, formats=formats)

        assert len(results) == 2
        assert ExportFormat.QWEN not in results


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test convenience export functions."""

    def test_export_to_openai(self, sample_calculator_tool):
        """Test export_to_openai function."""
        result = export_to_openai([sample_calculator_tool])

        assert isinstance(result, list)
        assert result[0]["type"] == "function"

    def test_export_to_anthropic(self, sample_calculator_tool):
        """Test export_to_anthropic function."""
        result = export_to_anthropic([sample_calculator_tool])

        assert isinstance(result, list)
        assert "input_schema" in result[0]

    def test_export_to_qwen(self, sample_calculator_tool):
        """Test export_to_qwen function."""
        result = export_to_qwen([sample_calculator_tool])

        assert isinstance(result, list)
        assert isinstance(result[0]["parameters"], list)

    def test_export_to_openapi(self, sample_calculator_tool):
        """Test export_to_openapi function."""
        result = export_to_openapi([sample_calculator_tool])

        assert isinstance(result, dict)
        assert result["openapi"] == "3.0.3"

    def test_export_to_json_schema(self, sample_calculator_tool):
        """Test export_to_json_schema function."""
        result = export_to_json_schema([sample_calculator_tool])

        assert isinstance(result, list)
        assert "$schema" in result[0]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_tool_list(self, exporter):
        """Test exporting empty tool list."""
        result = exporter.to_openai([])
        assert result == []

    def test_tool_without_parameters(self, exporter):
        """Test tool with no parameters."""
        tool = ToolDefinition(
            name="simple_tool",
            description="A tool with no parameters",
            parameters=[],
            category=ToolCategory.CALCULATOR,
        )

        result = exporter.to_openai([tool])
        assert result[0]["function"]["parameters"]["properties"] == {}
        assert result[0]["function"]["parameters"]["required"] == []

    def test_long_description_truncation(self, exporter):
        """Test long descriptions are handled."""
        tool = ToolDefinition(
            name="verbose_tool",
            description="A" * 2000,  # Very long description
            parameters=[],
            category=ToolCategory.CALCULATOR,
        )

        result = exporter.to_openai([tool])
        # OpenAI has 1024 char limit
        assert len(result[0]["function"]["description"]) <= 1024

    def test_special_characters_in_names(self, exporter):
        """Test tool names with special characters."""
        tool = ToolDefinition(
            name="calculate_co2_emissions_v2",
            description="Calculate CO2 emissions",
            parameters=[],
            category=ToolCategory.CALCULATOR,
        )

        # Should export without issues
        result = exporter.to_openai([tool])
        assert result[0]["function"]["name"] == "calculate_co2_emissions_v2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
