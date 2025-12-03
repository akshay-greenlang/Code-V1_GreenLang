"""
CBAM Carbon Intensity Calculator Agent - Calculates carbon intensity for CBAM-regulated goods (steel, cement, aluminum, fertilizers) and determines CBAM certificate requirements for EU imports.


This module implements the CbamCarbonIntensityCalculatorAgent for the GreenLang platform.
Generated from AgentSpec: cbam/carbon_intensity_v1

Version: 1.0.0
License: Apache-2.0
Generated: 2025-12-03T08:19:46.244413

ZERO-HALLUCINATION GUARANTEE:
- All numeric calculations use deterministic tools
- No LLM calls in the calculation path
- Complete provenance tracking with SHA-256 hashes
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import logging

from pydantic import BaseModel, Field, validator

from greenlang_sdk.core.agent_base import SDKAgentBase, AgentResult
from greenlang_sdk.core.provenance import ProvenanceTracker, ProvenanceRecord

from .tools import LookupCbamBenchmarkTool, CalculateCarbonIntensityTool
logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Models (Pydantic)
# =============================================================================

class CbamCarbonIntensityCalculatorAgentInput(BaseModel):
    """
    Input data model for CbamCarbonIntensityCalculatorAgent.

    All inputs are validated using Pydantic before processing.
    """

    product_type: str = Field(
        ..., description="Input field: product_type"
    )
    production_quantity: int = Field(
        ..., description="Input field: production_quantity"
    )
    total_emissions: int = Field(
        ..., description="Input field: total_emissions"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True


class CbamCarbonIntensityCalculatorAgentOutput(BaseModel):
    """
    Output data model for CbamCarbonIntensityCalculatorAgent.

    Includes standard provenance fields for audit trails.
    """

    carbon_intensity: float = Field(
        ...,
        description="Output field: carbon_intensity"
    )

    # Standard provenance fields
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance chain hash")
    processing_time_ms: Optional[float] = Field(None, description="Processing duration in milliseconds")
    validation_status: str = Field("PASS", description="Validation status: PASS or FAIL")

    class Config:
        """Pydantic configuration."""
        extra = "allow"  # Allow additional fields


# =============================================================================
# Agent Implementation
# =============================================================================

class CbamCarbonIntensityCalculatorAgent(SDKAgentBase[CbamCarbonIntensityCalculatorAgentInput, CbamCarbonIntensityCalculatorAgentOutput]):
    """
    CBAM Carbon Intensity Calculator Agent Implementation.

    Calculates carbon intensity for CBAM-regulated goods (steel, cement, aluminum, fertilizers) and determines CBAM certificate requirements for EU imports.


    This agent follows GreenLang's zero-hallucination principle:
    - All calculations use deterministic tools (NO LLM in calculation path)
    - Complete provenance tracking with SHA-256 hashes
    - Full audit trail for regulatory compliance
    - Citation tracking for all data sources

    Lifecycle:
        1. pre_validate  - Transform raw input
        2. validate_input - Validate against schema
        3. post_validate  - Enrich validated data
        4. pre_execute    - Setup execution
        5. execute        - Main logic (ZERO-HALLUCINATION)
        6. post_execute   - Transform output
        7. validate_output - Validate output
        8. finalize       - Cleanup and provenance

    Attributes:
        agent_id: Unique identifier for this agent
        agent_version: Semantic version string
        enable_provenance: Whether to track provenance
        enable_citations: Whether to track citations

    Example:
        >>> agent = CbamCarbonIntensityCalculatorAgent()
        >>> result = await agent.run({"key": "value"})
        >>> print(result.output)
        >>> print(result.provenance.provenance_chain)
    """

    # =========================================================================
    # Class Attributes
    # =========================================================================

    AGENT_ID = "cbam/carbon_intensity_v1"
    AGENT_VERSION = "1.0.0"

    SYSTEM_PROMPT = """You are a CBAM expert calculating carbon intensity for EU imports.\nZERO-HALLUCINATION: Always use tools for lookups and calculations.\n"""

    # Provenance configuration
    GWP_SET = "AR6GWP100"
    PIN_EMISSION_FACTORS = True
    PROVENANCE_FIELDS = ['inputs', 'outputs', 'factors', 'timestamp']

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        agent_id: str = "cbam/carbon_intensity_v1",
        agent_version: str = "1.0.0",
        enable_provenance: bool = True,
        enable_citations: bool = True,
    ):
        """
        Initialize CbamCarbonIntensityCalculatorAgent.

        Args:
            agent_id: Unique agent identifier
            agent_version: Agent version string
            enable_provenance: Enable SHA-256 provenance tracking
            enable_citations: Enable citation aggregation
        """
        super().__init__(
            agent_id=agent_id,
            agent_version=agent_version,
            enable_provenance=enable_provenance,
            enable_citations=enable_citations,
        )

        # Initialize tool instances
        self._tools: Dict[str, Any] = {}
        self._register_tools()

        logger.info(f"Initialized {self.agent_id} v{self.agent_version}")

    def _register_tools(self) -> None:
        """Register all available tools."""
        self._tools["lookup_cbam_benchmark"] = LookupCbamBenchmarkTool()
        self._tools["calculate_carbon_intensity"] = CalculateCarbonIntensityTool()

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def validate_input(
        self,
        input_data: CbamCarbonIntensityCalculatorAgentInput,
        context: dict
    ) -> CbamCarbonIntensityCalculatorAgentInput:
        """
        Validate input data against schema and business rules.

        Args:
            input_data: Input data (already Pydantic validated)
            context: Execution context

        Returns:
            Validated input data

        Raises:
            ValidationError: If business rules fail
        """
        logger.debug(f"Validating input for {self.agent_id}")

        # Pydantic handles schema validation
        # Add custom business rule validation here

        return input_data

    async def execute(
        self,
        validated_input: CbamCarbonIntensityCalculatorAgentInput,
        context: dict
    ) -> CbamCarbonIntensityCalculatorAgentOutput:
        """
        Execute main agent logic.

        ZERO-HALLUCINATION GUARANTEE:
        This method uses ONLY deterministic tools for all calculations.
        No LLM calls are made for numeric computations.

        Args:
            validated_input: Validated input data
            context: Execution context

        Returns:
            Processed output with provenance

        Raises:
            ExecutionError: If processing fails
        """
        start_time = datetime.utcnow()
        logger.info(f"Executing {self.agent_id}")

        try:
            # Execute core logic using deterministic tools
            result_data = await self._execute_core_logic(validated_input, context)

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Build output with provenance
            output = CbamCarbonIntensityCalculatorAgentOutput(
                **result_data,
                processing_time_ms=processing_time,
                validation_status="PASS",
            )

            logger.info(f"{self.agent_id} completed in {processing_time:.2f}ms")
            return output

        except Exception as e:
            logger.error(f"{self.agent_id} execution failed: {e}", exc_info=True)
            raise

    async def _execute_core_logic(
        self,
        input_data: CbamCarbonIntensityCalculatorAgentInput,
        context: dict
    ) -> Dict[str, Any]:
        """
        Execute core business logic using deterministic tools.

        IMPORTANT: This method must use ONLY registered tools.
        NO LLM calls for numeric calculations.

        Args:
            input_data: Validated input
            context: Execution context

        Returns:
            Dictionary of output field values
        """
        # TODO: Implement your business logic here
        # Example:
        #
        # # Step 1: Look up emission factor (deterministic)
        # result = await self.call_lookup_cbam_benchmark(...)
        #
        # # Step 2: Calculate emissions (deterministic)
        # emissions = activity_data * emission_factor
        #
        # # Step 3: Return results
        # return {
        #     "emissions_tco2e": emissions,
        #     "ef_uri": result["ef_uri"],
        # }

        return {}

    # =========================================================================
    # Tool Methods
    # =========================================================================

    async def call_lookup_cbam_benchmark(
        self,
        product_type: str,
    ) -> Dict[str, Any]:
        """
        Look up CBAM default benchmark values

        This is a DETERMINISTIC tool - results are reproducible and trackable.

        Args:
            product_type: product_type

        Returns:
            Tool execution result with provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If tool execution fails
        """
        params = {
            "product_type": product_type,
        }

        # Execute tool
        result = await self._tools["lookup_cbam_benchmark"].execute(params)

        # Record tool call for provenance
        self.record_tool_call("lookup_cbam_benchmark", params, result)

        return result

    async def call_calculate_carbon_intensity(
        self,
        total_emissions: float,
        production_quantity: float,
    ) -> Dict[str, Any]:
        """
        Calculate emissions per tonne of product

        This is a DETERMINISTIC tool - results are reproducible and trackable.

        Args:
            total_emissions: total_emissions
        Args:
            production_quantity: production_quantity

        Returns:
            Tool execution result with provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If tool execution fails
        """
        params = {
            "total_emissions": total_emissions,
            "production_quantity": production_quantity,
        }

        # Execute tool
        result = await self._tools["calculate_carbon_intensity"].execute(params)

        # Record tool call for provenance
        self.record_tool_call("calculate_carbon_intensity", params, result)

        return result


    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash of data for provenance."""
        import json
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# Factory Functions
# =============================================================================

async def create_agent(**kwargs) -> CbamCarbonIntensityCalculatorAgent:
    """
    Factory function to create CbamCarbonIntensityCalculatorAgent instance.

    Args:
        **kwargs: Arguments passed to CbamCarbonIntensityCalculatorAgent.__init__

    Returns:
        Initialized agent instance
    """
    return CbamCarbonIntensityCalculatorAgent(**kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "CbamCarbonIntensityCalculatorAgent",
    "CbamCarbonIntensityCalculatorAgentInput",
    "CbamCarbonIntensityCalculatorAgentOutput",
    "create_agent",
]