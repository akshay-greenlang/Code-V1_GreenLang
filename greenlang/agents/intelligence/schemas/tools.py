# -*- coding: utf-8 -*-
"""
Tool Definition Schemas

JSON Schema-based tool contracts for LLM function calling:
- ToolDef: Tool signature (name, description, parameters)
- ToolCall: LLM's request to execute a tool
- ToolChoice: Strategy for tool selection (auto/none/specific)
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ToolDef(BaseModel):
    """
    Tool definition for LLM function calling

    Uses JSON Schema to describe tool parameters, enabling:
    - Type validation (number, string, object, etc.)
    - Required vs. optional parameters
    - Descriptions for LLM understanding
    - Enum constraints for categorical values

    Example:
        ToolDef(
            name="calculate_fuel_emissions",
            description="Calculate CO2e emissions from fuel combustion",
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "enum": ["diesel", "gasoline", "natural_gas"],
                        "description": "Type of fuel"
                    },
                    "amount": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Fuel amount in gallons or cubic meters"
                    },
                    "region": {
                        "type": "string",
                        "description": "Geographic region code (e.g., 'CA', 'NY')"
                    }
                },
                "required": ["fuel_type", "amount", "region"]
            }
        )
    """

    name: str = Field(description="Tool name (must be valid Python identifier)")
    description: str = Field(
        description="Tool description for LLM (explain what it does and when to use it)"
    )
    parameters: Dict[str, Any] = Field(
        description="JSON Schema for tool parameters (type:object, properties, required)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "name": "get_grid_intensity",
                    "description": "Returns carbon intensity of electricity grid for a region",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "region": {
                                "type": "string",
                                "description": "Region code (e.g., 'CA', 'CAISO')",
                            },
                            "year": {
                                "type": "integer",
                                "minimum": 2000,
                                "maximum": 2030,
                                "description": "Year for grid data",
                            },
                        },
                        "required": ["region"],
                    },
                }
            ]
        },
    )


class ToolCall(BaseModel):
    """
    LLM's request to execute a tool

    Normalized format across providers (OpenAI, Anthropic, etc.):
    - id: Unique identifier for this tool call
    - name: Tool to execute (must match ToolDef.name)
    - arguments: Parameter values (validated against ToolDef.parameters)

    Example:
        ToolCall(
            id="call_abc123",
            name="calculate_fuel_emissions",
            arguments={
                "fuel_type": "diesel",
                "amount": 100,
                "region": "CA"
            }
        )
    """

    id: str = Field(description="Unique tool call ID (for result correlation)")
    name: str = Field(description="Tool name to execute")
    arguments: Dict[str, Any] = Field(
        description="Tool arguments (must match tool's JSON Schema)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": "call_001",
                    "name": "get_grid_intensity",
                    "arguments": {"region": "CA", "year": 2024},
                }
            ]
        },
    )


class ToolChoice:
    """
    Tool selection strategy constants

    - AUTO: LLM decides whether to use tools
    - NONE: Disable tool calling (text-only response)
    - REQUIRED: LLM must call at least one tool
    - Specific tool name: Force LLM to call that tool

    Usage:
        # Let LLM decide
        tool_choice = ToolChoice.AUTO

        # Force tool use
        tool_choice = ToolChoice.REQUIRED

        # Force specific tool
        tool_choice = "calculate_fuel_emissions"
    """

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"  # OpenAI: "required", Anthropic: "any"
