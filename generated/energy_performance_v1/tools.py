"""
Tool implementations for Building Energy Performance Calculator.

This module provides tool wrapper classes for zero-hallucination calculations.
All tools are DETERMINISTIC and track complete provenance.

Generated from AgentSpec: buildings/energy_performance_v1
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

from greenlang.data.bps_thresholds import get_bps_database

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

class CalculateEuiToolParams(BaseModel):
    """Input parameters for calculate_eui."""

    energy_consumption_kwh: float = Field(
...,
        description="energy_consumption_kwh"
    )
    floor_area_sqm: float = Field(
...,
        description="floor_area_sqm"
    )


class CalculateEuiToolResult(BaseModel):
    """Output result for calculate_eui."""

    eui_kwh_per_sqm: Optional[float] = Field(
None,
        description="eui_kwh_per_sqm"
    )

    # Provenance fields
    result_hash: Optional[str] = Field(None, description="SHA-256 hash of result")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")


class LookupBpsThresholdToolParams(BaseModel):
    """Input parameters for lookup_bps_threshold."""

    building_type: str = Field(
...,
        description="building_type"
    )
    climate_zone: Optional[str] = Field(
None,
        description="climate_zone"
    )


class LookupBpsThresholdToolResult(BaseModel):
    """Output result for lookup_bps_threshold."""

    threshold_kwh_per_sqm: Optional[float] = Field(
None,
        description="threshold_kwh_per_sqm"
    )

    # Provenance fields
    result_hash: Optional[str] = Field(None, description="SHA-256 hash of result")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")


class CheckBpsComplianceToolParams(BaseModel):
    """Input parameters for check_bps_compliance."""

    actual_eui: float = Field(
...,
        description="actual_eui"
    )
    threshold_eui: float = Field(
...,
        description="threshold_eui"
    )


class CheckBpsComplianceToolResult(BaseModel):
    """Output result for check_bps_compliance."""

    compliant: Optional[bool] = Field(
None,
        description="compliant"
    )
    gap_kwh_per_sqm: Optional[float] = Field(
None,
        description="gap_kwh_per_sqm"
    )

    # Provenance fields
    result_hash: Optional[str] = Field(None, description="SHA-256 hash of result")
    executed_at: Optional[datetime] = Field(None, description="Execution timestamp")



# =============================================================================
# Tool Implementations
# =============================================================================

class CalculateEuiTool(BaseTool):
    """
    Calculate Energy Use Intensity (kWh per sqm per year)

    Implementation: python://greenlang.tools.buildings:calculate_eui
    Safe (Deterministic): True

    This tool follows zero-hallucination principles:
    - Deterministic execution
    - No LLM calls
    - Complete provenance tracking
    """

    name = "calculate_eui"
    description = "Calculate Energy Use Intensity (kWh per sqm per year)"
    safe = True

    def __init__(self):
        """Initialize CalculateEuiTool."""
        super().__init__()
        logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute calculate_eui tool.

        Calculate Energy Use Intensity (kWh per sqm per year)

        Args:
            params: Tool parameters
                - energy_consumption_kwh: number (required)
                - floor_area_sqm: number (required)

        Returns:
            Dictionary containing:
                - eui_kwh_per_sqm: number
                - result_hash: SHA-256 hash for provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If execution fails
        """
        self._call_count += 1
        self._last_call_time = datetime.utcnow()

        # Validate parameters
        self._validate_required_params(params)
        validated = CalculateEuiToolParams(**params)

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
        if "energy_consumption_kwh" not in params or params["energy_consumption_kwh"] is None:
            raise ValueError("Missing required parameter: energy_consumption_kwh")
        if "floor_area_sqm" not in params or params["floor_area_sqm"] is None:
            raise ValueError("Missing required parameter: floor_area_sqm")

    async def _execute_internal(self, params: CalculateEuiToolParams) -> Dict[str, Any]:
        """
        Internal execution logic - deterministic EUI calculation.

        ZERO-HALLUCINATION: Uses exact formula: EUI = energy_consumption / floor_area
        All calculations are deterministic and reproducible.

        Args:
            params: Validated parameters

        Returns:
            Dictionary of output values
        """
        # Extract parameters
        energy_consumption_kwh = params.energy_consumption_kwh
        floor_area_sqm = params.floor_area_sqm

        # Validate division by zero
        if floor_area_sqm <= 0:
            raise ValueError(
                f"Floor area must be positive, got: {floor_area_sqm} sqm"
            )

        # Calculate EUI (deterministic formula)
        # Formula: EUI = energy_consumption / floor_area
        eui_kwh_per_sqm = energy_consumption_kwh / floor_area_sqm

        # Build calculation formula for provenance
        formula = f"{energy_consumption_kwh:.2f} kWh / {floor_area_sqm:.2f} sqm = {eui_kwh_per_sqm:.4f} kWh/sqm/year"

        # Return result
        return {
            "eui_kwh_per_sqm": eui_kwh_per_sqm,
            "calculation_formula": formula,
            "energy_consumption_kwh": energy_consumption_kwh,
            "floor_area_sqm": floor_area_sqm,
        }


class LookupBpsThresholdTool(BaseTool):
    """
    Look up BPS threshold for building type

    Implementation: python://greenlang.tools.buildings:lookup_threshold
    Safe (Deterministic): True

    This tool follows zero-hallucination principles:
    - Deterministic execution
    - No LLM calls
    - Complete provenance tracking
    """

    name = "lookup_bps_threshold"
    description = "Look up BPS threshold for building type"
    safe = True

    def __init__(self):
        """Initialize LookupBpsThresholdTool."""
        super().__init__()
        logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute lookup_bps_threshold tool.

        Look up BPS threshold for building type

        Args:
            params: Tool parameters
                - building_type: string (required)
                - climate_zone: string

        Returns:
            Dictionary containing:
                - threshold_kwh_per_sqm: number
                - result_hash: SHA-256 hash for provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If execution fails
        """
        self._call_count += 1
        self._last_call_time = datetime.utcnow()

        # Validate parameters
        self._validate_required_params(params)
        validated = LookupBpsThresholdToolParams(**params)

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
        if "building_type" not in params or params["building_type"] is None:
            raise ValueError("Missing required parameter: building_type")

    async def _execute_internal(self, params: LookupBpsThresholdToolParams) -> Dict[str, Any]:
        """
        Internal execution logic - connects to BPS threshold database.

        ZERO-HALLUCINATION: This is a deterministic database lookup.
        Same inputs always return same outputs from authoritative sources
        (NYC Local Law 97, ENERGY STAR, ASHRAE 90.1).

        Args:
            params: Validated parameters

        Returns:
            Dictionary of output values
        """
        # Get database instance
        db = get_bps_database()

        # Extract parameters
        building_type = params.building_type
        climate_zone = params.climate_zone

        # Lookup BPS threshold
        threshold = db.lookup(building_type, climate_zone)

        if threshold is None:
            raise ValueError(
                f"BPS threshold not found for building_type={building_type}, climate_zone={climate_zone}. "
                f"Available building types: {', '.join(db.list_building_types())}"
            )

        # Return threshold data
        return {
            "threshold_kwh_per_sqm": threshold.threshold_kwh_per_sqm,
            "ghg_threshold_kgco2e_per_sqm": threshold.ghg_threshold_kgco2e_per_sqm,
            "source": threshold.source,
            "jurisdiction": threshold.jurisdiction,
            "effective_date": threshold.effective_date,
            "climate_zone": threshold.climate_zone,
            "building_type": threshold.building_type,
        }


class CheckBpsComplianceTool(BaseTool):
    """
    Check if building meets BPS threshold

    Implementation: python://greenlang.tools.buildings:check_compliance
    Safe (Deterministic): True

    This tool follows zero-hallucination principles:
    - Deterministic execution
    - No LLM calls
    - Complete provenance tracking
    """

    name = "check_bps_compliance"
    description = "Check if building meets BPS threshold"
    safe = True

    def __init__(self):
        """Initialize CheckBpsComplianceTool."""
        super().__init__()
        logger.debug(f"Initialized tool: {self.name}")

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute check_bps_compliance tool.

        Check if building meets BPS threshold

        Args:
            params: Tool parameters
                - actual_eui: number (required)
                - threshold_eui: number (required)

        Returns:
            Dictionary containing:
                - compliant: boolean
                - gap_kwh_per_sqm: number
                - result_hash: SHA-256 hash for provenance

        Raises:
            ValueError: If required parameters are missing
            ToolExecutionError: If execution fails
        """
        self._call_count += 1
        self._last_call_time = datetime.utcnow()

        # Validate parameters
        self._validate_required_params(params)
        validated = CheckBpsComplianceToolParams(**params)

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
        if "actual_eui" not in params or params["actual_eui"] is None:
            raise ValueError("Missing required parameter: actual_eui")
        if "threshold_eui" not in params or params["threshold_eui"] is None:
            raise ValueError("Missing required parameter: threshold_eui")

    async def _execute_internal(self, params: CheckBpsComplianceToolParams) -> Dict[str, Any]:
        """
        Internal execution logic - deterministic BPS compliance check.

        ZERO-HALLUCINATION: Uses exact comparison logic:
        - compliant = (actual_eui <= threshold_eui)
        - gap = actual_eui - threshold_eui (positive = exceeds threshold)

        Args:
            params: Validated parameters

        Returns:
            Dictionary of output values
        """
        # Extract parameters
        actual_eui = params.actual_eui
        threshold_eui = params.threshold_eui

        # Validate non-negative values
        if actual_eui < 0:
            raise ValueError(f"Actual EUI cannot be negative, got: {actual_eui}")
        if threshold_eui < 0:
            raise ValueError(f"Threshold EUI cannot be negative, got: {threshold_eui}")

        # Determine compliance (deterministic comparison)
        compliant = actual_eui <= threshold_eui

        # Calculate gap (positive means exceeds threshold, needs improvement)
        gap_kwh_per_sqm = actual_eui - threshold_eui

        # Calculate percentage over/under threshold
        if threshold_eui > 0:
            percentage_difference = (gap_kwh_per_sqm / threshold_eui) * 100
        else:
            percentage_difference = 0.0

        # Return result
        return {
            "compliant": compliant,
            "gap_kwh_per_sqm": gap_kwh_per_sqm,
            "percentage_difference": percentage_difference,
            "actual_eui": actual_eui,
            "threshold_eui": threshold_eui,
            "compliance_status": "COMPLIANT" if compliant else "NON-COMPLIANT",
        }



# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, type] = {
    "calculate_eui": CalculateEuiTool,
    "lookup_bps_threshold": LookupBpsThresholdTool,
    "check_bps_compliance": CheckBpsComplianceTool,
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
    # calculate_eui
    "CalculateEuiTool",
    "CalculateEuiToolParams",
    "CalculateEuiToolResult",
    # lookup_bps_threshold
    "LookupBpsThresholdTool",
    "LookupBpsThresholdToolParams",
    "LookupBpsThresholdToolResult",
    # check_bps_compliance
    "CheckBpsComplianceTool",
    "CheckBpsComplianceToolParams",
    "CheckBpsComplianceToolResult",
    # Registry
    "TOOL_REGISTRY",
    "get_tool",
    "list_tools",
    "get_tool_info",
]