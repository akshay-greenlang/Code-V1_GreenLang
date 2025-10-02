"""
Tool Runtime - "No Naked Numbers" Enforcement (INTL-103)

Core implementation of CTO specification:
- Tool execution with JSON Schema validation
- Unit-aware post-check (all numerics must be Quantity)
- Claims-based final assembly with {{claim:i}} macros
- Conservative digit scanner to block "naked numbers"
- Provenance tracking for every numeric value

Architecture:
    LLM → AssistantStep (tool_call|final) → ToolRuntime → Tool → Claims → Final

One-line rubric:
"No number reaches the user unless it came from a validated tool output,
 carries a recognized unit, and is tied to explicit provenance."
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal
from datetime import datetime
import re
import logging
from jsonschema import Draft202012Validator, ValidationError
from jsonpath_ng import parse as jsonpath_parse

from .schemas import Quantity, Claim, ASSISTANT_STEP_SCHEMA, QUANTITY_SCHEMA
from .errors import (
    GLValidationError,
    GLRuntimeError,
    GLSecurityError,
    GLDataError,
    GLProvenanceError,
)
from .units import UnitRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL DEFINITION
# ============================================================================


@dataclass
class Tool:
    """
    Tool definition with schemas and execution function

    Attributes:
        name: Tool name (must be unique in registry)
        description: Human-readable description for LLM
        args_schema: JSON Schema (Draft 2020-12) for arguments
        result_schema: JSON Schema for results (must use Quantity for numbers)
        fn: Callable that executes the tool
        live_required: If True, tool needs Live mode (blocked in Replay)

    Example:
        Tool(
            name="energy_intensity",
            description="Compute kWh/m2 given annual kWh and floor area",
            args_schema={
                "type": "object",
                "required": ["annual_kwh", "floor_m2"],
                "properties": {
                    "annual_kwh": {"type": "number", "minimum": 0},
                    "floor_m2": {"type": "number", "exclusiveMinimum": 0}
                }
            },
            result_schema={
                "type": "object",
                "required": ["intensity"],
                "properties": {
                    "intensity": {"$ref": "greenlang://schemas/quantity.json"}
                }
            },
            fn=lambda annual_kwh, floor_m2: {
                "intensity": {"value": annual_kwh / floor_m2, "unit": "kWh/m2"}
            }
        )
    """

    name: str
    description: str
    args_schema: Dict[str, Any]
    result_schema: Dict[str, Any]
    fn: Callable[..., Dict[str, Any]]
    live_required: bool = False


# ============================================================================
# TOOL REGISTRY
# ============================================================================


class ToolRegistry:
    """
    Tool registry for registration and lookup

    Example:
        registry = ToolRegistry()
        registry.register(energy_intensity_tool)

        tool = registry.get("energy_intensity")
        tools = registry.list()
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool"""
        if tool.name in self._tools:
            logger.warning(f"Overwriting tool: {tool.name}")

        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Tool:
        """Get tool by name"""
        if name not in self._tools:
            raise GLDataError(
                code="PATH_RESOLUTION",
                message=f"Tool '{name}' not found in registry",
                hint="Ensure tool is registered before use",
            )
        return self._tools[name]

    def list(self) -> List[Tool]:
        """List all registered tools"""
        return list(self._tools.values())

    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return list(self._tools.keys())


# ============================================================================
# TOOL RUNTIME
# ============================================================================


class ToolRuntime:
    """
    Tool runtime with "no naked numbers" enforcement

    Orchestrates:
    1. Tool call validation (args against schema)
    2. Tool execution
    3. Result validation (must use Quantity for all numbers)
    4. Provenance tracking
    5. Final assembly with claims validation
    6. Naked number scanning

    Usage:
        runtime = ToolRuntime(provider, registry, mode="Replay")
        result = runtime.run("You are a climate analyst", "What's the intensity?")

        print(result["message"])  # "Energy intensity is 10.0 kWh/m2."
        print(result["provenance"])  # [{source_call_id: "tc_1", ...}]
    """

    def __init__(
        self,
        provider: Any,  # LLM provider (OpenAI/Anthropic wrapper)
        registry: ToolRegistry,
        mode: Literal["Replay", "Live"] = "Replay",
    ):
        """
        Initialize tool runtime

        Args:
            provider: LLM provider with chat_step() and inject_tool_result()
            registry: ToolRegistry with registered tools
            mode: Execution mode (Replay blocks egress, Live allows)
        """
        self.provider = provider
        self.registry = registry
        self.mode = mode
        self.ureg = UnitRegistry()
        self.provenance: List[Dict[str, Any]] = []
        self._call_counter = 0

        # Metrics
        self._naked_number_blocks = 0
        self._replay_violations = 0

    # ========================================================================
    # MAIN LOOP
    # ========================================================================

    def run(
        self, system_prompt: str, user_msg: str, max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Main execution loop

        Steps:
        1. Initialize chat with provider
        2. Loop:
           - Get next step from LLM (tool_call or final)
           - If tool_call: execute and inject result
           - If final: validate claims and return
        3. Handle NO_NAKED_NUMBERS errors with retry

        Args:
            system_prompt: System instruction for LLM
            user_msg: User message
            max_retries: Max retries for naked number violations

        Returns:
            {
                "message": str (rendered with quantities),
                "provenance": List[Claim]
            }

        Raises:
            GLRuntimeError: If naked numbers persist after retries
        """
        # Add no-naked-numbers instruction
        full_system = (
            f"{system_prompt}\n\n"
            "IMPORTANT RULES:\n"
            "1. You must use tools to get ALL numeric values.\n"
            "2. In your final message, reference numbers via {{claim:i}} macros ONLY.\n"
            "3. Never type numeric digits directly in final message.\n"
            "4. Provide claims[] array linking each {{claim:i}} to a tool output.\n"
        )

        state = self._start_chat(full_system, user_msg)
        retry_count = 0

        while True:
            # Get next step from provider
            step = self.provider.chat_step(
                schema=ASSISTANT_STEP_SCHEMA,
                tools=self._tool_descriptors(),
                state=state,
            )

            if step["kind"] == "tool_call":
                # Execute tool call
                result = self._execute_tool_call(step)
                state = self.provider.inject_tool_result(result)
                retry_count = 0  # Reset on tool call

            else:  # kind == "final"
                try:
                    # Validate and assemble final
                    return self._finalize(step["final"])

                except GLRuntimeError as e:
                    if e.code == "NO_NAKED_NUMBERS" and retry_count < max_retries:
                        # Send repair instruction
                        repair_msg = (
                            f"❌ Error: {e.message}\n\n"
                            f"{e.hint}\n\n"
                            f"Context: {e.context}\n\n"
                            "Please either:\n"
                            "1. Call a tool to get the numeric value, OR\n"
                            "2. Use {{claim:i}} macros backed by claims[]\n\n"
                            "Try again."
                        )
                        state = self.provider.inject_error(repair_msg)
                        retry_count += 1
                        self._naked_number_blocks += 1

                    else:
                        # Max retries exceeded or other error
                        raise

    # ========================================================================
    # TOOL EXECUTION
    # ========================================================================

    def _execute_tool_call(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call with full validation

        Steps:
        1. Get tool from registry
        2. Validate args against args_schema
        3. Check mode (Live vs Replay)
        4. Execute tool function
        5. Validate output against result_schema
        6. Ensure no raw numbers (only Quantity)
        7. Index quantities for later claim resolution
        8. Record provenance

        Args:
            step: {"kind": "tool_call", "tool_name": str, "arguments": dict}

        Returns:
            {"tool_call_id": str, "output": dict}
        """
        # 1. Get tool
        tool = self.registry.get(step["tool_name"])

        # 2. Validate arguments
        self._validate(step["arguments"], tool.args_schema, "ARGS_SCHEMA")

        # 3. Check mode
        if tool.live_required and self.mode == "Replay":
            self._replay_violations += 1
            raise GLSecurityError(
                code="EGRESS_BLOCKED",
                message=f"Tool '{tool.name}' requires Live mode but runtime is in Replay",
                hint="Switch to Live mode or provide snapshot for replay",
            )

        # 4. Execute
        try:
            output = tool.fn(**step["arguments"])
        except Exception as e:
            raise GLDataError(
                code="PATH_RESOLUTION",
                message=f"Tool '{tool.name}' execution failed: {e}",
                hint=f"Check tool implementation: {tool.fn}",
            )

        # 5. Validate output
        self._validate(output, tool.result_schema, "RESULT_SCHEMA")

        # 6. Ensure no raw numbers
        self._ensure_no_raw_numbers(output)

        # 7. Index quantities
        unit_index = self._index_quantities(output)

        # 8. Generate call ID
        call_id = self._new_call_id()

        # 9. Record provenance
        provenance_entry = {
            "id": call_id,
            "tool_name": tool.name,
            "arguments": step["arguments"],
            "output": output,
            "unit_index": unit_index,
            "timestamp": datetime.utcnow().isoformat(),
            "mode": self.mode,
        }
        self.provenance.append(provenance_entry)

        logger.info(
            f"Tool executed: {tool.name} (call_id={call_id}, "
            f"quantities={len(unit_index)})"
        )

        return {"tool_call_id": call_id, "output": output}

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate(self, data: Any, schema: Dict[str, Any], error_type: str) -> None:
        """
        Validate data against JSON Schema (Draft 2020-12)

        Args:
            data: Data to validate
            schema: JSON Schema
            error_type: Error code (ARGS_SCHEMA or RESULT_SCHEMA)

        Raises:
            GLValidationError: If validation fails
        """
        # Resolve $ref if needed (simple resolution for Quantity)
        resolved_schema = self._resolve_refs(schema)

        validator = Draft202012Validator(resolved_schema)

        try:
            validator.validate(data)
        except ValidationError as e:
            raise GLValidationError(
                code=error_type,
                message=f"Schema validation failed: {e.message}",
                path=".".join(str(p) for p in e.path),
                hint=f"Check data structure against schema. Path: {list(e.path)}",
            )

    def _resolve_refs(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve $ref pointers in schema (simple implementation)

        Handles: {"$ref": "greenlang://schemas/quantity.json"}
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                if ref == "greenlang://schemas/quantity.json":
                    return QUANTITY_SCHEMA
                else:
                    return schema  # Unknown ref, keep as-is

            return {k: self._resolve_refs(v) for k, v in schema.items()}

        elif isinstance(schema, list):
            return [self._resolve_refs(item) for item in schema]

        else:
            return schema

    # ========================================================================
    # RAW NUMBER DETECTION
    # ========================================================================

    def _ensure_no_raw_numbers(self, output: Dict, path: str = "$") -> None:
        """
        Recursively scan output for raw numbers

        Raw numbers are forbidden. All numerics must be in Quantity {value, unit}.

        Args:
            output: Tool output to scan
            path: Current JSONPath (for error reporting)

        Raises:
            GLValidationError.RESULT_SCHEMA: If raw number found
        """
        if isinstance(output, dict):
            # Check if it's a Quantity (allowed)
            if "value" in output and "unit" in output and len(output) == 2:
                # Valid Quantity - check types
                if not isinstance(output["value"], (int, float)):
                    raise GLValidationError(
                        code="RESULT_SCHEMA",
                        message=f"Quantity.value must be number at {path}",
                        path=path,
                    )
                if not isinstance(output["unit"], str):
                    raise GLValidationError(
                        code="RESULT_SCHEMA",
                        message=f"Quantity.unit must be string at {path}",
                        path=path,
                    )
                return  # Valid Quantity

            # Recurse into dict
            for key, value in output.items():
                self._ensure_no_raw_numbers(value, f"{path}.{key}")

        elif isinstance(output, list):
            for i, item in enumerate(output):
                self._ensure_no_raw_numbers(item, f"{path}[{i}]")

        elif isinstance(output, (int, float)):
            # RAW NUMBER FOUND!
            raise GLValidationError(
                code="RESULT_SCHEMA",
                message=(
                    f"Raw number at {path}. All numerics must be wrapped in "
                    f'Quantity {{value: {output}, unit: "..."}}.'
                ),
                path=path,
                hint="Change tool to return Quantity instead of raw number",
            )

    # ========================================================================
    # QUANTITY INDEXING
    # ========================================================================

    def _index_quantities(self, output: Dict, path: str = "$") -> Dict[str, Quantity]:
        """
        Extract and index all Quantity objects from output

        Builds a map: JSONPath → Quantity

        Args:
            output: Tool output
            path: Current JSONPath

        Returns:
            Dict mapping paths to Quantity objects

        Example:
            output = {
                "emissions": {"value": 1234, "unit": "kgCO2e"},
                "details": {
                    "per_unit": {"value": 12.34, "unit": "kgCO2e/m2"}
                }
            }

            Result:
            {
                "$.emissions": Quantity(value=1234, unit="kgCO2e"),
                "$.details.per_unit": Quantity(value=12.34, unit="kgCO2e/m2")
            }
        """
        index = {}

        if isinstance(output, dict):
            # Check if Quantity
            if "value" in output and "unit" in output and len(output) == 2:
                index[path] = Quantity(
                    value=float(output["value"]), unit=output["unit"]
                )
                return index

            # Recurse
            for key, value in output.items():
                index.update(self._index_quantities(value, f"{path}.{key}"))

        elif isinstance(output, list):
            for i, item in enumerate(output):
                index.update(self._index_quantities(item, f"{path}[{i}]"))

        return index

    # ========================================================================
    # FINAL ASSEMBLY & NAKED NUMBER ENFORCEMENT
    # ========================================================================

    def _finalize(self, final: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize with claims validation and naked number scanning

        Steps:
        1. Validate final payload structure
        2. Resolve each claim:
           - Find tool result by source_call_id
           - Resolve JSONPath to Quantity
           - Compare claimed Quantity to resolved (must match)
        3. Format quantities for rendering
        4. Render {{claim:i}} macros
        5. Scan for naked numbers (conservative)
        6. Return final result

        Args:
            final: {"message": str, "claims": List[Claim]}

        Returns:
            {"message": str, "provenance": List[dict]}

        Raises:
            GLRuntimeError.NO_NAKED_NUMBERS: If naked numbers found
            GLDataError.PATH_RESOLUTION: If claim can't be resolved
        """
        # 1. Resolve all claims
        resolved = []
        for i, claim_dict in enumerate(final.get("claims", [])):
            # Parse as Claim
            claim = Claim(**claim_dict)

            # Find tool result
            tool_result = self._find_prov(claim.source_call_id)

            # Resolve JSONPath
            quantity_from_tool = self._resolve_jsonpath(
                tool_result["output"], claim.path
            )

            # Compare with claimed quantity
            if not self.ureg.same_quantity(quantity_from_tool, claim.quantity):
                raise GLDataError(
                    code="QUANTITY_MISMATCH",
                    message=(
                        f"Claim {i} mismatch: tool returned {quantity_from_tool}, "
                        f"but claimed {claim.quantity}"
                    ),
                    path=claim.path,
                    hint="Ensure claim matches tool output exactly (after normalization)",
                )

            # Format for rendering
            formatted = self._format_quantity(claim.quantity)
            resolved.append(formatted)

        # 2. Render macros
        rendered = self._render_with_claims(final["message"], resolved)

        # 3. Scan for naked numbers
        self._scan_for_naked_numbers(rendered, resolved)

        # 4. Return final result
        return {"message": rendered, "provenance": final.get("claims", [])}

    def _resolve_jsonpath(self, output: Dict, path: str) -> Quantity:
        """
        Resolve JSONPath to Quantity

        Args:
            output: Tool output dict
            path: JSONPath (e.g., "$.intensity")

        Returns:
            Quantity at that path

        Raises:
            GLDataError.PATH_RESOLUTION: If path invalid or doesn't point to Quantity
        """
        try:
            jsonpath_expr = jsonpath_parse(path)
            matches = jsonpath_expr.find(output)

            if not matches:
                raise GLDataError(
                    code="PATH_RESOLUTION",
                    message=f"Path '{path}' not found in output",
                    path=path,
                    hint=f"Available paths: {list(self._index_quantities(output).keys())}",
                )

            # Get first match
            value = matches[0].value

            # Must be a Quantity
            if (
                not isinstance(value, dict)
                or "value" not in value
                or "unit" not in value
            ):
                raise GLDataError(
                    code="PATH_RESOLUTION",
                    message=f"Path '{path}' did not resolve to Quantity, got: {value}",
                    path=path,
                    hint="Path must point to a Quantity {value, unit}",
                )

            return Quantity(value=float(value["value"]), unit=value["unit"])

        except GLDataError:
            raise
        except Exception as e:
            raise GLDataError(
                code="PATH_RESOLUTION",
                message=f"Invalid JSONPath '{path}': {e}",
                path=path,
                hint="Use format: $.field or $.nested.field",
            )

    def _format_quantity(self, q: Quantity) -> str:
        """
        Format quantity for display

        Normalizes to canonical unit and formats with thousand separators.

        Args:
            q: Quantity to format

        Returns:
            Formatted string (e.g., "1,234.5 kgCO2e")
        """
        # Normalize
        value, unit = self.ureg.normalize(q)

        # Format with thousand separators
        value_float = float(value)

        if abs(value_float) >= 1000:
            return f"{value_float:,.1f} {unit}"
        elif abs(value_float) >= 1:
            return f"{value_float:.2f} {unit}"
        else:
            return f"{value_float:.4f} {unit}"

    def _render_with_claims(self, template: str, resolved: List[str]) -> str:
        """
        Replace {{claim:i}} macros with resolved values

        Args:
            template: Message with {{claim:i}} macros
            resolved: List of formatted quantities

        Returns:
            Rendered message

        Raises:
            GLRuntimeError.NO_NAKED_NUMBERS: If invalid claim index
        """

        def replacer(match):
            index = int(match.group(1))
            if index >= len(resolved):
                raise GLRuntimeError(
                    code="NO_NAKED_NUMBERS",
                    message=f"Invalid claim index: {{{{claim:{index}}}}}",
                    hint=f"Only {len(resolved)} claims provided, but referenced {{{{claim:{index}}}}}",
                )
            return resolved[index]

        # Simple regex replacement
        return re.sub(r"\{\{claim:(\d+)\}\}", replacer, template)

    def _scan_for_naked_numbers(self, message: str, resolved_values: List[str]) -> None:
        """
        Scan for digits not from {{claim:i}} macros

        Conservative whitelist approach - only allow digits in specific contexts.

        Whitelisted contexts:
        - Resolved claim values: "10.00 kWh/m2" (from {{claim:i}})
        - Ordered list markers: "1. Item"
        - Version strings: "v0.4.0" (ONLY inside code blocks)
        - ISO dates: "2024-01-15"
        - ID patterns: "ID-123", "ID_456"

        Args:
            message: Final rendered message
            resolved_values: List of resolved quantity strings to exclude from scan

        Raises:
            GLRuntimeError.NO_NAKED_NUMBERS: If unwhitelisted digits found
        """
        # Build list of character ranges to exclude (resolved values)
        excluded_ranges = []
        for resolved_val in resolved_values:
            # Find all occurrences of this resolved value
            start = 0
            while True:
                pos = message.find(resolved_val, start)
                if pos == -1:
                    break
                excluded_ranges.append((pos, pos + len(resolved_val)))
                start = pos + 1

        # Find code blocks (triple backticks) and add to excluded ranges
        code_block_pattern = r"```[\s\S]*?```"
        for code_match in re.finditer(code_block_pattern, message):
            excluded_ranges.append((code_match.start(), code_match.end()))

        # Whitelist patterns (DO NOT flag these) - context-based
        whitelisted_patterns = [
            r"(?:^|\n)\d+\.\s",  # Ordered list: "1. Item" (start of line)
            r"\b\d{4}-\d{2}-\d{2}\b",  # ISO date: 2024-01-15
            r"\bID[-_]?\d+\b",  # ID: ID-123, ID_456
            r"\b\d{2}:\d{2}(:\d{2})?\b",  # Time: 14:30, 14:30:00
        ]
        # Note: Version strings (v0.4.0) are NOW ONLY allowed inside code blocks
        # They are excluded via the code_block_pattern above

        # Find all digit sequences
        digit_pattern = r"\b\d+\.?\d*\b"
        matches = re.finditer(digit_pattern, message)

        for match in matches:
            text = match.group()
            position = match.start()

            # Check if this match is within an excluded range
            # (resolved value OR code block)
            in_excluded_range = any(
                start <= position < end for start, end in excluded_ranges
            )

            if in_excluded_range:
                # This digit is part of a resolved claim value or code block - skip
                continue

            # Get context (20 chars before and after)
            context_start = max(0, position - 20)
            context_end = min(len(message), position + 20)
            context = message[context_start:context_end]

            # Check if in whitelisted context
            is_whitelisted = any(
                re.search(pattern, context) for pattern in whitelisted_patterns
            )

            if not is_whitelisted:
                raise GLRuntimeError(
                    code="NO_NAKED_NUMBERS",
                    message=f"Naked number '{text}' detected at position {position}",
                    hint=(
                        "All numeric values must come from tools via {{claim:i}} macros. "
                        "Either call a tool or remove the number."
                    ),
                    context=f"...{context}...",
                )

    def _find_prov(self, call_id: str) -> Dict[str, Any]:
        """
        Find tool result by call ID

        Args:
            call_id: Tool call ID to find

        Returns:
            Tool result dict

        Raises:
            GLProvenanceError.MISSING_TOOL_CALL: If not found
        """
        for result in self.provenance:
            if result["id"] == call_id:
                return result

        raise GLProvenanceError(
            code="MISSING_TOOL_CALL",
            message=f"Tool call '{call_id}' not found in provenance",
            hint=f"Valid call IDs: {[r['id'] for r in self.provenance]}",
        )

    # ========================================================================
    # PROVIDER INTERFACE
    # ========================================================================

    def _tool_descriptors(self) -> List[Dict[str, Any]]:
        """
        Convert tools to provider format (OpenAI/Anthropic)

        Returns:
            List of tool descriptors for LLM
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.args_schema,
            }
            for tool in self.registry.list()
        ]

    def _start_chat(self, system_prompt: str, user_msg: str) -> Any:
        """
        Initialize chat state with provider

        Args:
            system_prompt: System instruction (with no-naked-numbers rule)
            user_msg: User message

        Returns:
            Provider state object
        """
        return self.provider.init_chat(system_prompt, user_msg)

    def _new_call_id(self) -> str:
        """Generate new tool call ID"""
        self._call_counter += 1
        return f"tc_{self._call_counter}"

    # ========================================================================
    # METRICS
    # ========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get runtime metrics

        Returns:
            {
                "tool_use_rate": float,
                "naked_number_rejections": int,
                "replay_violations": int,
                "total_tool_calls": int
            }
        """
        total_steps = len(self.provenance) + 1  # +1 for final
        tool_calls = len(self.provenance)

        return {
            "tool_use_rate": tool_calls / total_steps if total_steps > 0 else 0,
            "naked_number_rejections": self._naked_number_blocks,
            "replay_violations": self._replay_violations,
            "total_tool_calls": tool_calls,
        }
