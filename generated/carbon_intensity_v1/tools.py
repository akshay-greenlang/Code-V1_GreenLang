"""
Tool implementations for CBAM Carbon Intensity Calculator.

This module provides tool wrapper classes for zero-hallucination calculations.
All tools are DETERMINISTIC and track complete provenance.

Generated from AgentSpec: cbam/carbon_intensity_v1
Version: 1.0.0

ZERO-HALLUCINATION GUARANTEE:
- All tools produce deterministic, reproducible results
- No LLM calls within tool execution
- Complete parameter validation
- Full provenance tracking with SHA-256 hashes
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import hashlib
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, Field, validator

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

from greenlang.data.cbam_benchmarks import get_cbam_database

logger = logging.getLogger(__name__)


# =============================================================================
# Base Tool Class
# =============================================================================

class BaseTool(ABC):
    """
    Base class for all deterministic tools.

    All tools must:
    - Be deterministic (same inputs -> same outputs)
    - Track provenance
    - Validate parameters
    - NOT make any LLM calls
    """

    name: str = "base_tool"
    description: str = "Base tool"
    safe: bool = True  # Safe tools are deterministic

    def __init__(self):
        """Initialize base tool."""
        self._call_count = 0
        self._last_call_time: Optional[datetime] = None

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Execution result dictionary

        Raises:
            ValueError: If parameters are invalid
            ToolExecutionError: If execution fails
        """
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """
        Validate input parameters.

        Args:
            params: Parameters to validate

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        return True

    def hash_result(self, result: Dict[str, Any]) -> str:
        """
        Calculate SHA-256 hash of result for provenance.

        Args:
            result: Result dictionary

        Returns:
            SHA-256 hash string
        """
        import json
        json_str = json.dumps(result, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# Tool Parameter Models
# =============================================================================

class LookupCbamBenchmarkToolParams(BaseModel):
    """Input parameters for lookup_cbam_benchmark."""

    product_type: str = Field(
...,
        description="product_type"
    )


class LookupCbamBenchmarkToolResult(BaseModel):
    """Output result for lookup_cbam_benchmark."""

    benchmark_value: Optional[float] = Field(
None,
        description="benchmark_value"
    )
    benchmark_unit: Optional[str] = Field(
None,
        description="benchmark_unit"
    )

    # Provenance fields
    result_hash: Optional[str] = Field(None, description="SHA-256 hash of result")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")


class CalculateCarbonIntensityToolParams(BaseModel):
    """Input parameters for calculate_carbon_intensity."""

    total_emissions: float = Field(
...,
        description="total_emissions"
    )
    production_quantity: float = Field(
...,
        description="production_quantity"
    )


class CalculateCarbonIntensityToolResult(BaseModel):
    """Output result for calculate_carbon_intensity."""

    carbon_intensity: Optional[float] = Field(
None,
        description="carbon_intensity"
    )

    # Provenance fields
    result_hash: Optional[str] = Field(None, description="SHA-256 hash of result")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")



# =============================================================================
# Tool Implementations
# =============================================================================

class LookupCbamBenchmarkTool(BaseTool):
    """
    Look up CBAM default benchmark values

    Implementation: python://greenlang.tools.cbam:lookup_benchmark
    Safe (Deterministic): True

    This tool follows zero-hallucination principles:
    - Deterministic execution
    - No LLM calls
    - Complete provenance tracking
    """

    name = "lookup_cbam_benchmark"
    description = "Look up CBAM default benchmark values"
    safe = True

    def __init__(self):
        """Initialize LookupCbamBenchmarkTool."""
        super().__init__()
        logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute lookup_cbam_benchmark tool.

        Look up CBAM default benchmark values

        Args:
            params: Tool parameters
                - product_type: string (required)

        Returns:
            Dictionary containing:
                - benchmark_value: number
                - benchmark_unit: string
                - result_hash: SHA-256 hash for provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If execution fails
        """
        self._call_count += 1
        self._last_call_time = datetime.utcnow()

        # Validate parameters
        self._validate_required_params(params)
        validated = LookupCbamBenchmarkToolParams(**params)

        try:
            # Execute deterministic logic
            result_data = await self._execute_internal(validated)

            # Calculate provenance hash
            result_hash = self.hash_result(result_data)

            # Build result
            result = {
                **result_data,
                "result_hash": result_hash,
                "executed_at": self._last_call_time.isoformat(),
            }

            logger.debug(f"Tool {self.name} executed successfully: {result_hash[:16]}...")
            return result

        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}", exc_info=True)
            raise

    def _validate_required_params(self, params: Dict[str, Any]) -> None:
        """Validate required parameters are present."""
        if "product_type" not in params or params["product_type"] is None:
            raise ValueError("Missing required parameter: product_type")

    async def _execute_internal(self, params: LookupCbamBenchmarkToolParams) -> Dict[str, Any]:
        """
        Internal execution logic - connects to CBAM benchmark database.

        ZERO-HALLUCINATION: This is a deterministic database lookup.
        Same inputs always return same outputs from EU Implementing Regulation 2023/1773.

        Args:
            params: Validated parameters

        Returns:
            Dictionary of output values
        """
        # Get database instance
        db = get_cbam_database()

        # Extract parameters
        product_type = params.product_type

        # Lookup CBAM benchmark
        benchmark = db.lookup(product_type)

        if benchmark is None:
            raise ValueError(
                f"CBAM benchmark not found for product_type={product_type}. "
                f"Available products: {', '.join(db.list_products())}"
            )

        # Return benchmark data
        return {
            "benchmark_value": benchmark.benchmark_value,
            "benchmark_unit": benchmark.unit,
            "cn_codes": benchmark.cn_codes,
            "effective_date": benchmark.effective_date,
            "source": benchmark.source,
            "production_method": benchmark.production_method,
        }


class CalculateCarbonIntensityTool(BaseTool):
    """
    Calculate emissions per tonne of product

    Implementation: python://greenlang.tools.cbam:calculate_intensity
    Safe (Deterministic): True

    This tool follows zero-hallucination principles:
    - Deterministic execution
    - No LLM calls
    - Complete provenance tracking
    """

    name = "calculate_carbon_intensity"
    description = "Calculate emissions per tonne of product"
    safe = True

    def __init__(self):
        """Initialize CalculateCarbonIntensityTool."""
        super().__init__()
        logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute calculate_carbon_intensity tool.

        Calculate emissions per tonne of product

        Args:
            params: Tool parameters
                - total_emissions: number (required)
                - production_quantity: number (required)

        Returns:
            Dictionary containing:
                - carbon_intensity: number
                - result_hash: SHA-256 hash for provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If execution fails
        """
        self._call_count += 1
        self._last_call_time = datetime.utcnow()

        # Validate parameters
        self._validate_required_params(params)
        validated = CalculateCarbonIntensityToolParams(**params)

        try:
            # Execute deterministic logic
            result_data = await self._execute_internal(validated)

            # Calculate provenance hash
            result_hash = self.hash_result(result_data)

            # Build result
            result = {
                **result_data,
                "result_hash": result_hash,
                "executed_at": self._last_call_time.isoformat(),
            }

            logger.debug(f"Tool {self.name} executed successfully: {result_hash[:16]}...")
            return result

        except Exception as e:
            logger.error(f"Tool {self.name} execution failed: {e}", exc_info=True)
            raise

    def _validate_required_params(self, params: Dict[str, Any]) -> None:
        """Validate required parameters are present."""
        if "total_emissions" not in params or params["total_emissions"] is None:
            raise ValueError("Missing required parameter: total_emissions")
        if "production_quantity" not in params or params["production_quantity"] is None:
            raise ValueError("Missing required parameter: production_quantity")

    async def _execute_internal(self, params: CalculateCarbonIntensityToolParams) -> Dict[str, Any]:
        """
        Internal execution logic - deterministic carbon intensity calculation.

        ZERO-HALLUCINATION: Uses exact formula: carbon_intensity = total_emissions / production_quantity
        All calculations are deterministic and reproducible.

        Args:
            params: Validated parameters

        Returns:
            Dictionary of output values
        """
        # Extract parameters
        total_emissions = params.total_emissions
        production_quantity = params.production_quantity

        # Validate division by zero
        if production_quantity <= 0:
            raise ValueError(
                f"Production quantity must be positive, got: {production_quantity}"
            )

        # Calculate carbon intensity (deterministic formula)
        # Formula: carbon_intensity = total_emissions / production_quantity
        carbon_intensity = total_emissions / production_quantity

        # Build calculation formula for provenance
        formula = f"{total_emissions:.4f} tCO2e / {production_quantity:.4f} tonnes = {carbon_intensity:.6f} tCO2e/tonne"

        # Return result
        return {
            "carbon_intensity": carbon_intensity,
            "carbon_intensity_unit": "tCO2e/tonne",
            "calculation_formula": formula,
            "total_emissions": total_emissions,
            "production_quantity": production_quantity,
        }



# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, type] = {
    "lookup_cbam_benchmark": LookupCbamBenchmarkTool,
    "calculate_carbon_intensity": CalculateCarbonIntensityTool,
}


def get_tool(name: str) -> Optional[BaseTool]:
    """
    Get tool instance by name.

    Args:
        name: Tool name

    Returns:
        Tool instance or None if not found
    """
    tool_class = TOOL_REGISTRY.get(name)
    if tool_class:
        return tool_class()
    return None


def list_tools() -> List[str]:
    """
    List all available tool names.

    Returns:
        List of tool names
    """
    return list(TOOL_REGISTRY.keys())


def get_tool_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get tool metadata.

    Args:
        name: Tool name

    Returns:
        Tool info dictionary or None
    """
    tool_class = TOOL_REGISTRY.get(name)
    if tool_class:
        return {
            "name": tool_class.name,
            "description": tool_class.description,
            "safe": tool_class.safe,
        }
    return None


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Base
    "BaseTool",
    # lookup_cbam_benchmark
    "LookupCbamBenchmarkTool",
    "LookupCbamBenchmarkToolParams",
    "LookupCbamBenchmarkToolResult",
    # calculate_carbon_intensity
    "CalculateCarbonIntensityTool",
    "CalculateCarbonIntensityToolParams",
    "CalculateCarbonIntensityToolResult",
    # Registry
    "TOOL_REGISTRY",
    "get_tool",
    "list_tools",
    "get_tool_info",
]