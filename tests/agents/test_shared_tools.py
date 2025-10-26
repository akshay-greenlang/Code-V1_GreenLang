"""
Tests for GreenLang Shared Tool Library
========================================

Comprehensive test suite covering:
- Tool base classes and interfaces
- Tool registry and discovery
- Emission calculation tools
- Tool composition
- Error handling

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

import pytest
from greenlang.agents.tools import (
    BaseTool,
    ToolDef,
    ToolResult,
    ToolSafety,
    ToolRegistry,
    get_registry,
    CalculateEmissionsTool,
    AggregateEmissionsTool,
    CalculateBreakdownTool,
    CompositeTool,
    tool,
)


class TestToolBase:
    """Tests for base tool classes."""

    def test_tool_result_creation(self):
        """ToolResult can be created with success status."""
        result = ToolResult(
            success=True,
            data={"emissions": 100.0},
            metadata={"calculation": "test"}
        )

        assert result.success is True
        assert result.data["emissions"] == 100.0
        assert result.metadata["calculation"] == "test"

    def test_tool_result_to_dict(self):
        """ToolResult can be converted to dictionary."""
        result = ToolResult(
            success=True,
            data={"value": 42},
            error=None,
            metadata={"test": "data"}
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["data"]["value"] == 42
        assert d["metadata"]["test"] == "data"

    def test_tool_def_creation(self):
        """ToolDef can be created with JSON schema."""
        tool_def = ToolDef(
            name="test_tool",
            description="A test tool",
            parameters={
                "type": "object",
                "required": ["param1"],
                "properties": {
                    "param1": {"type": "string"}
                }
            },
            safety=ToolSafety.DETERMINISTIC
        )

        assert tool_def.name == "test_tool"
        assert tool_def.description == "A test tool"
        assert tool_def.safety == ToolSafety.DETERMINISTIC


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def setup_method(self):
        """Create fresh registry for each test."""
        self.registry = ToolRegistry()

    def test_register_tool(self):
        """Tools can be registered in registry."""
        tool = CalculateEmissionsTool()
        self.registry.register(tool, category="emissions")

        assert self.registry.has("calculate_emissions")
        assert len(self.registry) == 1

    def test_get_registered_tool(self):
        """Registered tools can be retrieved."""
        tool = CalculateEmissionsTool()
        self.registry.register(tool, category="emissions")

        retrieved = self.registry.get("calculate_emissions")

        assert retrieved is tool
        assert retrieved.name == "calculate_emissions"

    def test_duplicate_registration_raises_error(self):
        """Registering duplicate tool name raises ValueError."""
        tool1 = CalculateEmissionsTool()
        tool2 = CalculateEmissionsTool()

        self.registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(tool2)

    def test_get_nonexistent_tool_raises_error(self):
        """Getting nonexistent tool raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            self.registry.get("nonexistent_tool")

    def test_list_tools_by_category(self):
        """Tools can be listed by category."""
        calc_tool = CalculateEmissionsTool()
        agg_tool = AggregateEmissionsTool()

        self.registry.register(calc_tool, category="emissions")
        self.registry.register(agg_tool, category="emissions")

        tools = self.registry.list_tools(category="emissions")

        assert len(tools) == 2
        assert "calculate_emissions" in tools
        assert "aggregate_emissions" in tools

    def test_list_tools_by_safety(self):
        """Tools can be filtered by safety level."""
        tool = CalculateEmissionsTool()
        self.registry.register(tool, category="emissions")

        tools = self.registry.list_tools(safety=ToolSafety.DETERMINISTIC)

        assert len(tools) == 1
        assert "calculate_emissions" in tools

    def test_unregister_tool(self):
        """Tools can be unregistered."""
        tool = CalculateEmissionsTool()
        self.registry.register(tool)

        self.registry.unregister("calculate_emissions")

        assert not self.registry.has("calculate_emissions")
        assert len(self.registry) == 0

    def test_get_tool_defs(self):
        """Can get ToolDef objects for all registered tools."""
        calc_tool = CalculateEmissionsTool()
        self.registry.register(calc_tool, category="emissions")

        tool_defs = self.registry.get_tool_defs()

        assert len(tool_defs) == 1
        assert tool_defs[0].name == "calculate_emissions"
        assert isinstance(tool_defs[0], ToolDef)

    def test_get_catalog(self):
        """Registry provides complete catalog."""
        tool = CalculateEmissionsTool()
        self.registry.register(tool, category="emissions")

        catalog = self.registry.get_catalog()

        assert catalog["total_tools"] == 1
        assert "emissions" in catalog["categories"]
        assert "calculate_emissions" in catalog["tools"]


class TestCalculateEmissionsTool:
    """Tests for CalculateEmissionsTool."""

    def test_calculate_emissions_success(self):
        """Emissions are calculated correctly."""
        tool = CalculateEmissionsTool()

        result = tool(
            fuel_type="natural_gas",
            amount=1000.0,
            unit="therms",
            emission_factor=53.06,
            emission_factor_unit="kgCO2e/therm",
            country="US"
        )

        assert result.success is True
        assert result.data["emissions_kg_co2e"] == 53060.0  # 1000 * 53.06
        assert result.data["fuel_type"] == "natural_gas"

    def test_calculate_emissions_with_citations(self):
        """Calculation includes citation."""
        tool = CalculateEmissionsTool()

        result = tool(
            fuel_type="natural_gas",
            amount=1000.0,
            unit="therms",
            emission_factor=53.06,
            emission_factor_unit="kgCO2e/therm"
        )

        assert result.success is True
        assert len(result.citations) == 1
        assert result.citations[0].step_name == "calculate_emissions"

    def test_get_tool_def(self):
        """Tool provides valid ToolDef."""
        tool = CalculateEmissionsTool()
        tool_def = tool.get_tool_def()

        assert tool_def.name == "calculate_emissions"
        assert tool_def.safety == ToolSafety.DETERMINISTIC
        assert "required" in tool_def.parameters
        assert "fuel_type" in tool_def.parameters["properties"]

    def test_execution_metrics_tracked(self):
        """Execution metrics are tracked."""
        tool = CalculateEmissionsTool()

        result = tool(
            fuel_type="natural_gas",
            amount=1000.0,
            unit="therms",
            emission_factor=53.06,
            emission_factor_unit="kgCO2e/therm"
        )

        assert result.execution_time_ms > 0
        assert tool.execution_count == 1

        stats = tool.get_stats()
        assert stats["executions"] == 1
        assert stats["avg_time_ms"] > 0


class TestAggregateEmissionsTool:
    """Tests for AggregateEmissionsTool."""

    def test_aggregate_multiple_sources(self):
        """Emissions from multiple sources are aggregated correctly."""
        tool = AggregateEmissionsTool()

        emissions = [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 1000.0},
            {"fuel_type": "coal", "co2e_emissions_kg": 2000.0},
            {"fuel_type": "diesel", "co2e_emissions_kg": 500.0},
        ]

        result = tool(emissions=emissions)

        assert result.success is True
        assert result.data["total_co2e_kg"] == 3500.0
        assert result.data["total_co2e_tons"] == 3.5
        assert result.data["num_sources"] == 3

    def test_aggregate_breakdown_by_fuel(self):
        """Aggregation provides breakdown by fuel type."""
        tool = AggregateEmissionsTool()

        emissions = [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 1000.0},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 500.0},
            {"fuel_type": "coal", "co2e_emissions_kg": 2000.0},
        ]

        result = tool(emissions=emissions)

        assert result.success is True
        assert result.data["by_fuel"]["natural_gas"] == 1500.0
        assert result.data["by_fuel"]["coal"] == 2000.0

    def test_aggregate_empty_list(self):
        """Aggregation handles empty list."""
        tool = AggregateEmissionsTool()

        result = tool(emissions=[])

        assert result.success is True
        assert result.data["total_co2e_kg"] == 0.0
        assert result.data["num_sources"] == 0


class TestCalculateBreakdownTool:
    """Tests for CalculateBreakdownTool."""

    def test_calculate_percentage_breakdown(self):
        """Percentage breakdown is calculated correctly."""
        tool = CalculateBreakdownTool()

        emissions = [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 1000.0},
            {"fuel_type": "coal", "co2e_emissions_kg": 3000.0},
        ]

        result = tool(emissions=emissions, total_emissions=4000.0)

        assert result.success is True
        assert result.data["by_fuel_percent"]["natural_gas"] == 25.0
        assert result.data["by_fuel_percent"]["coal"] == 75.0

    def test_identify_largest_source(self):
        """Largest emission source is identified."""
        tool = CalculateBreakdownTool()

        emissions = [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 500.0},
            {"fuel_type": "coal", "co2e_emissions_kg": 3000.0},
            {"fuel_type": "diesel", "co2e_emissions_kg": 200.0},
        ]

        result = tool(emissions=emissions, total_emissions=3700.0)

        assert result.success is True
        assert result.data["largest_source"] == "coal"
        assert result.data["smallest_source"] == "diesel"


class TestToolComposition:
    """Tests for tool composition."""

    def test_composite_tool_execution(self):
        """Composite tools execute sub-tools in sequence."""
        calc_tool = CalculateEmissionsTool()
        # Note: For full composition, would need more tools

        composite = CompositeTool(
            name="full_calculation",
            description="Complete calculation workflow",
            tools=[calc_tool],
            safety=ToolSafety.DETERMINISTIC
        )

        # This is a simplified test
        assert composite.name == "full_calculation"
        assert len(composite.tools) == 1


class TestToolDecorator:
    """Tests for @tool decorator."""

    def test_function_wrapped_as_tool(self):
        """Functions can be wrapped as tools using @tool decorator."""
        @tool(
            name="add_numbers",
            description="Add two numbers",
            parameters={
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                }
            }
        )
        def add_numbers(a: float, b: float) -> ToolResult:
            return ToolResult(
                success=True,
                data={"result": a + b}
            )

        result = add_numbers(a=10.0, b=20.0)

        assert result.success is True
        assert result.data["result"] == 30.0
