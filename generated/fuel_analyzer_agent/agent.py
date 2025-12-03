"""
Fuel Emissions Analyzer Agent - Calculates greenhouse gas emissions from fuel combustion using IPCC emission factors. Supports multiple fuel types (natural gas, diesel, gasoline, LPG) and provides complete provenance tracking for regulatory compliance.


This module implements the FuelEmissionsAnalyzerAgent for the GreenLang platform.
Generated from AgentSpec: emissions/fuel_analyzer_v1

Version: 1.0.0
License: Apache-2.0
Generated: 2025-12-03T06:23:55.531815
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import logging

from pydantic import BaseModel, Field

from greenlang_sdk.core.agent_base import SDKAgentBase, AgentResult
from greenlang_sdk.core.provenance import ProvenanceTracker

from .tools import *

logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Models
# =============================================================================

class FuelEmissionsAnalyzerAgentInput(BaseModel):
    """Input data model for FuelEmissionsAnalyzerAgent."""

    fuel_type: str = Field(..., description="Input field: fuel_type")
    quantity: float = Field(..., description="Input field: quantity")
    unit: str = Field(..., description="Input field: unit")
    region: str = Field(..., description="Input field: region")
    year: int = Field(..., description="Input field: year")


class FuelEmissionsAnalyzerAgentOutput(BaseModel):
    """Output data model for FuelEmissionsAnalyzerAgent."""

    emissions_tco2e: float = Field(..., description="Output field: emissions_tco2e")
    ef_source: str = Field(..., description="Output field: ef_source")

    # Standard provenance fields
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance hash")
    processing_time_ms: Optional[float] = Field(None, description="Processing duration")


# =============================================================================
# Agent Implementation
# =============================================================================

class FuelEmissionsAnalyzerAgent(SDKAgentBase[FuelEmissionsAnalyzerAgentInput, FuelEmissionsAnalyzerAgentOutput]):
    """
    Fuel Emissions Analyzer Agent Implementation.

    Calculates greenhouse gas emissions from fuel combustion using IPCC emission factors. Supports multiple fuel types (natural gas, diesel, gasoline, LPG) and provides complete provenance tracking for regulatory compliance.


    This agent follows GreenLang's zero-hallucination principle:
    - All calculations use deterministic tools
    - Complete provenance tracking with SHA-256 hashes
    - Full audit trail for regulatory compliance

    Attributes:
        agent_id: Unique identifier for this agent
        agent_version: Version of this agent

    Example:
        >>> agent = FuelEmissionsAnalyzerAgent()
        >>> result = await agent.run({"key": "value"})
        >>> print(result.output)
    """

    # System prompt for AI orchestration
    SYSTEM_PROMPT = """You are a climate emissions expert specialized in calculating fuel-based
greenhouse gas emissions following GHG Protocol Corporate Standard.

ZERO-HALLUCINATION RULES:
1. NEVER guess emission factors - always use lookup_emission_factor tool
2. NEVER perform calculations manually - use calculate_emissions tool
3. ALWAYS cite emission factor sources with EF URIs
4. ALWAYS validate inputs before calculation

Your workflow:
1. Validate input fuel type and quantity
2. Look up appropriate emission factor from IPCC/EPA database
3. Calculate emissions using deterministic tool
4. Return results with full provenance

Supported fuel types:
- natural_gas: Pipeline natural gas (MJ or m3)
- diesel: Diesel fuel (liters)
- gasoline: Motor gasoline (liters)
- lpg: Liquefied petroleum gas (kg)
- fuel_oil: Heavy fuel oil (liters)
"""

    def __init__(
        self,
        agent_id: str = "emissions/fuel_analyzer_v1",
        agent_version: str = "1.0.0",
        enable_provenance: bool = True,
        enable_citations: bool = True,
    ):
        """Initialize FuelEmissionsAnalyzerAgent."""
        super().__init__(
            agent_id=agent_id,
            agent_version=agent_version,
            enable_provenance=enable_provenance,
            enable_citations=enable_citations,
        )

        # Initialize tool registry
        self._tools: Dict[str, Any] = {}
        self._register_tools()

        logger.info(f"Initialized {self.agent_id} v{self.agent_version}")

    def _register_tools(self) -> None:
        """Register available tools."""
        self._tools["lookup_emission_factor"] = LookupEmissionFactorTool()
        self._tools["calculate_emissions"] = CalculateEmissionsTool()
        self._tools["validate_fuel_input"] = ValidateFuelInputTool()

    async def validate_input(
        self,
        input_data: FuelEmissionsAnalyzerAgentInput,
        context: dict
    ) -> FuelEmissionsAnalyzerAgentInput:
        """
        Validate input data against schema.

        Args:
            input_data: Raw input data
            context: Execution context

        Returns:
            Validated input data

        Raises:
            ValidationError: If validation fails
        """
        # Pydantic handles basic validation via the model
        # Add custom validation logic here
        logger.debug(f"Validating input for {self.agent_id}")

        return input_data

    async def execute(
        self,
        validated_input: FuelEmissionsAnalyzerAgentInput,
        context: dict
    ) -> FuelEmissionsAnalyzerAgentOutput:
        """
        Execute main agent logic.

        ZERO-HALLUCINATION: All calculations use deterministic tools.
        No LLM calls are made for numeric calculations.

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Processed output with provenance

        Raises:
            ExecutionError: If execution fails
        """
        start_time = datetime.utcnow()
        logger.info(f"Executing {self.agent_id}")

        try:
            # Execute core logic using tools
            result_data = await self._execute_core_logic(validated_input, context)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Build output
            output = FuelEmissionsAnalyzerAgentOutput(
                **result_data,
                processing_time_ms=processing_time,
            )

            logger.info(f"{self.agent_id} execution completed in {processing_time:.2f}ms")
            return output

        except Exception as e:
            logger.error(f"{self.agent_id} execution failed: {e}", exc_info=True)
            raise

    async def _execute_core_logic(
        self,
        input_data: FuelEmissionsAnalyzerAgentInput,
        context: dict
    ) -> Dict[str, Any]:
        """
        Execute core business logic.

        IMPORTANT: This method must use ONLY deterministic tools.
        No LLM calls for calculations.

        Override this method to implement custom logic.

        Args:
            input_data: Validated input
            context: Execution context

        Returns:
            Dictionary of output values
        """
        # TODO: Implement core logic using tools
        # Example:
        # result = await self.call_calculate_emissions(
        #     fuel_type=input_data.fuel_type,
        #     quantity=input_data.quantity
        # )
        # return {"emissions_tco2e": result["value"]}

        return {}

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Any:
        """Execute a registered tool."""
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self._tools[tool_name]
        return await tool.execute(params)

    # =========================================================================
    # Tool Methods
    # =========================================================================

    async def call_lookup_emission_factor(self, **kwargs) -> Dict[str, Any]:
        """
        Call lookup_emission_factor tool.

        Look up emission factor for a fuel type from the IPCC/EPA emission factor database. Returns the emission factor value, unit, and source citation. This is a DETERMINISTIC lookup - same inputs always return same outputs.

        """
        result = await self._execute_tool("lookup_emission_factor", kwargs)
        self.record_tool_call("lookup_emission_factor", kwargs, result)
        return result

    async def call_calculate_emissions(self, **kwargs) -> Dict[str, Any]:
        """
        Call calculate_emissions tool.

        Calculate GHG emissions from fuel combustion using emission factor and activity data. Uses deterministic formula: emissions = activity * emission_factor. All calculations are traceable and reproducible.

        """
        result = await self._execute_tool("calculate_emissions", kwargs)
        self.record_tool_call("calculate_emissions", kwargs, result)
        return result

    async def call_validate_fuel_input(self, **kwargs) -> Dict[str, Any]:
        """
        Call validate_fuel_input tool.

        Validate fuel input for physical plausibility. Checks that values are within reasonable ranges and units are compatible.

        """
        result = await self._execute_tool("validate_fuel_input", kwargs)
        self.record_tool_call("validate_fuel_input", kwargs, result)
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

async def create_agent(**kwargs) -> FuelEmissionsAnalyzerAgent:
    """Factory function to create agent instance."""
    return FuelEmissionsAnalyzerAgent(**kwargs)
