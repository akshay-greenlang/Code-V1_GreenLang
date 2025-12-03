"""
Building Energy Performance Calculator Agent - Calculates Energy Use Intensity (EUI) and checks compliance with Building Performance Standards for urban decarbonization.


This module implements the BuildingEnergyPerformanceCalculatorAgent for the GreenLang platform.
Generated from AgentSpec: buildings/energy_performance_v1

Version: 1.0.0
License: Apache-2.0
Generated: 2025-12-03T08:19:51.398179

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

from .tools import CalculateEuiTool, LookupBpsThresholdTool, CheckBpsComplianceTool
logger = logging.getLogger(__name__)


# =============================================================================
# Input/Output Models (Pydantic)
# =============================================================================

class BuildingEnergyPerformanceCalculatorAgentInput(BaseModel):
    """
    Input data model for BuildingEnergyPerformanceCalculatorAgent.

    All inputs are validated using Pydantic before processing.
    """

    building_type: str = Field(
        ..., description="Input field: building_type"
    )
    floor_area_sqm: int = Field(
        ..., description="Input field: floor_area_sqm"
    )
    energy_consumption_kwh: int = Field(
        ..., description="Input field: energy_consumption_kwh"
    )
    climate_zone: str = Field(
        ..., description="Input field: climate_zone"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True


class BuildingEnergyPerformanceCalculatorAgentOutput(BaseModel):
    """
    Output data model for BuildingEnergyPerformanceCalculatorAgent.

    Includes standard provenance fields for audit trails.
    """

    eui_kwh_per_sqm: int = Field(
        ...,
        description="Output field: eui_kwh_per_sqm"
    )
    bps_compliance_status: str = Field(
        ...,
        description="Output field: bps_compliance_status"
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

class BuildingEnergyPerformanceCalculatorAgent(SDKAgentBase[BuildingEnergyPerformanceCalculatorAgentInput, BuildingEnergyPerformanceCalculatorAgentOutput]):
    """
    Building Energy Performance Calculator Agent Implementation.

    Calculates Energy Use Intensity (EUI) and checks compliance with Building Performance Standards for urban decarbonization.


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
        >>> agent = BuildingEnergyPerformanceCalculatorAgent()
        >>> result = await agent.run({"key": "value"})
        >>> print(result.output)
        >>> print(result.provenance.provenance_chain)
    """

    # =========================================================================
    # Class Attributes
    # =========================================================================

    AGENT_ID = "buildings/energy_performance_v1"
    AGENT_VERSION = "1.0.0"

    SYSTEM_PROMPT = """You are a building energy expert calculating EUI and BPS compliance.\nZERO-HALLUCINATION: Always use tools for calculations and threshold lookups.\n"""

    # Provenance configuration
    GWP_SET = "AR6GWP100"
    PIN_EMISSION_FACTORS = True
    PROVENANCE_FIELDS = ['inputs', 'outputs', 'factors', 'timestamp']

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        agent_id: str = "buildings/energy_performance_v1",
        agent_version: str = "1.0.0",
        enable_provenance: bool = True,
        enable_citations: bool = True,
    ):
        """
        Initialize BuildingEnergyPerformanceCalculatorAgent.

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
        self._tools["calculate_eui"] = CalculateEuiTool()
        self._tools["lookup_bps_threshold"] = LookupBpsThresholdTool()
        self._tools["check_bps_compliance"] = CheckBpsComplianceTool()

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def validate_input(
        self,
        input_data: BuildingEnergyPerformanceCalculatorAgentInput,
        context: dict
    ) -> BuildingEnergyPerformanceCalculatorAgentInput:
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
        validated_input: BuildingEnergyPerformanceCalculatorAgentInput,
        context: dict
    ) -> BuildingEnergyPerformanceCalculatorAgentOutput:
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
            output = BuildingEnergyPerformanceCalculatorAgentOutput(
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
        input_data: BuildingEnergyPerformanceCalculatorAgentInput,
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
        # result = await self.call_calculate_eui(...)
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

    async def call_calculate_eui(
        self,
        energy_consumption_kwh: float,
        floor_area_sqm: float,
    ) -> Dict[str, Any]:
        """
        Calculate Energy Use Intensity (kWh per sqm per year)

        This is a DETERMINISTIC tool - results are reproducible and trackable.

        Args:
            energy_consumption_kwh: energy_consumption_kwh
        Args:
            floor_area_sqm: floor_area_sqm

        Returns:
            Tool execution result with provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If tool execution fails
        """
        params = {
            "energy_consumption_kwh": energy_consumption_kwh,
            "floor_area_sqm": floor_area_sqm,
        }

        # Execute tool
        result = await self._tools["calculate_eui"].execute(params)

        # Record tool call for provenance
        self.record_tool_call("calculate_eui", params, result)

        return result

    async def call_lookup_bps_threshold(
        self,
        building_type: str,
        climate_zone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Look up BPS threshold for building type

        This is a DETERMINISTIC tool - results are reproducible and trackable.

        Args:
            building_type: building_type
        Args:
            climate_zone: climate_zone

        Returns:
            Tool execution result with provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If tool execution fails
        """
        params = {
            "building_type": building_type,
            "climate_zone": climate_zone,
        }

        # Execute tool
        result = await self._tools["lookup_bps_threshold"].execute(params)

        # Record tool call for provenance
        self.record_tool_call("lookup_bps_threshold", params, result)

        return result

    async def call_check_bps_compliance(
        self,
        actual_eui: float,
        threshold_eui: float,
    ) -> Dict[str, Any]:
        """
        Check if building meets BPS threshold

        This is a DETERMINISTIC tool - results are reproducible and trackable.

        Args:
            actual_eui: actual_eui
        Args:
            threshold_eui: threshold_eui

        Returns:
            Tool execution result with provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If tool execution fails
        """
        params = {
            "actual_eui": actual_eui,
            "threshold_eui": threshold_eui,
        }

        # Execute tool
        result = await self._tools["check_bps_compliance"].execute(params)

        # Record tool call for provenance
        self.record_tool_call("check_bps_compliance", params, result)

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

async def create_agent(**kwargs) -> BuildingEnergyPerformanceCalculatorAgent:
    """
    Factory function to create BuildingEnergyPerformanceCalculatorAgent instance.

    Args:
        **kwargs: Arguments passed to BuildingEnergyPerformanceCalculatorAgent.__init__

    Returns:
        Initialized agent instance
    """
    return BuildingEnergyPerformanceCalculatorAgent(**kwargs)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "BuildingEnergyPerformanceCalculatorAgent",
    "BuildingEnergyPerformanceCalculatorAgentInput",
    "BuildingEnergyPerformanceCalculatorAgentOutput",
    "create_agent",
]