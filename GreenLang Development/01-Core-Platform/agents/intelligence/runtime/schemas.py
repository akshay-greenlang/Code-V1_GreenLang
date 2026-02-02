# -*- coding: utf-8 -*-
"""
Core Schemas for Tool Runtime (INTL-103)

Defines the data contracts for "no naked numbers" enforcement:
- Quantity: The ONLY allowed way to carry numeric values
- AssistantStep: Schema for LLM responses (tool_call OR final)
- Claim: Provenance link from final message to tool outputs

CTO Specification: All numerics must be wrapped in Quantity with explicit units.
Raw numbers (int/float) are forbidden in tool outputs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, Field, validator


# ============================================================================
# QUANTITY SCHEMA - The ONLY way to carry numbers
# ============================================================================

QUANTITY_SCHEMA: Dict[str, Any] = {
    "$id": "greenlang://schemas/quantity.json",
    "title": "Quantity",
    "type": "object",
    "required": ["value", "unit"],
    "properties": {
        "value": {
            "type": "number",
            "description": "Numeric value (will be normalized to canonical unit)",
        },
        "unit": {
            "type": "string",
            "description": "UCUM or domain unit (e.g., kgCO2e, kWh, %, USD)",
        },
    },
    "additionalProperties": False,
}


class Quantity(BaseModel):
    """
    Quantity: value + unit

    This is the ONLY allowed way to represent numbers in tool outputs.
    Raw numerics are forbidden - everything must have a unit.

    Examples:
        Quantity(value=100.5, unit="kgCO2e")
        Quantity(value=12.4, unit="kWh/m2")
        Quantity(value=85.0, unit="%")
    """

    value: float = Field(description="Numeric value")
    unit: str = Field(description="Unit of measurement")

    class Config:
        frozen = True  # Immutable for safety


# ============================================================================
# CLAIM SCHEMA - Provenance link from final message to tool output
# ============================================================================


class Claim(BaseModel):
    """
    Claim: Links a {{claim:i}} macro in final message to a tool output

    Used to prove that every number in the final message came from a tool.

    Attributes:
        source_call_id: ID of tool call that produced this value
        path: JSONPath to the Quantity in tool output (e.g., "$.intensity")
        quantity: The claimed Quantity value (must match resolved path)

    Example:
        Claim(
            source_call_id="tc_1",
            path="$.co2e",
            quantity=Quantity(value=1234.5, unit="kgCO2e")
        )
    """

    source_call_id: str = Field(description="Tool call ID that produced this value")
    path: str = Field(
        description="JSONPath to Quantity in tool output (e.g., $.intensity)",
        pattern=r"^\$\.[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$",
    )
    quantity: Quantity = Field(description="The claimed Quantity (must match resolved)")

    class Config:
        frozen = True


# ============================================================================
# ASSISTANT STEP SCHEMA - What the LLM can send
# ============================================================================

ASSISTANT_STEP_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "oneOf": [
        {
            # Tool call variant
            "required": ["kind", "tool_name", "arguments"],
            "properties": {
                "kind": {"const": "tool_call"},
                "tool_name": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "additionalProperties": False,
        },
        {
            # Final variant
            "required": ["kind", "final"],
            "properties": {
                "kind": {"const": "final"},
                "final": {
                    "type": "object",
                    "required": ["message"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Final message with {{claim:i}} macros only",
                        },
                        "claims": {
                            "type": "array",
                            "description": "Provenance links for all {{claim:i}} macros",
                            "items": {
                                "type": "object",
                                "required": ["source_call_id", "path", "quantity"],
                                "properties": {
                                    "source_call_id": {"type": "string"},
                                    "path": {"type": "string"},
                                    "quantity": {
                                        "type": "object",
                                        "required": ["value", "unit"],
                                        "properties": {
                                            "value": {"type": "number"},
                                            "unit": {"type": "string"},
                                        },
                                        "additionalProperties": False,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            },
            "additionalProperties": False,
        },
    ],
}


class ToolCallStep(BaseModel):
    """AssistantStep variant: tool_call"""

    kind: Literal["tool_call"] = "tool_call"
    tool_name: str
    arguments: Dict[str, Any]


class FinalStep(BaseModel):
    """AssistantStep variant: final"""

    kind: Literal["final"] = "final"
    final: "FinalPayload"


class FinalPayload(BaseModel):
    """Final message payload with claims"""

    message: str = Field(description="Message with {{claim:i}} macros")
    claims: List[Claim] = Field(
        default_factory=list, description="Provenance for macros"
    )

    @validator("message")
    def check_no_raw_digits_except_macros(cls, v):
        """Validate that message uses {{claim:i}} format for claims"""
        # Real validation happens in ToolRuntime._scan_for_naked_numbers
        # This validator just ensures the message is a string
        return v


AssistantStep = Union[ToolCallStep, FinalStep]


# Update forward refs
FinalStep.model_rebuild()
