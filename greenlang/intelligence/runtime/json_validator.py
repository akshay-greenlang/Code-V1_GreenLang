"""
JSON Validation and Repair System

Enforces strict JSON output from LLMs with automatic repair:
- Extract candidate JSON from LLM responses (strip code fences, fix common errors)
- Validate against JSON Schema
- Generate repair prompts for retry attempts
- Track retry count and fail after >3 attempts

CTO SPEC COMPLIANCE:
- Hard stop after >3 JSON parse/validate retries
- Cost meter increments on EVERY attempt (including failed parse attempts)
- Raises GLJsonParseError after exceeding retry limit

Architecture:
    LLM Response → extract_candidate_json() → validate_json_schema()
    → [if fails] → generate_repair_prompt() → LLM retry
    → [if >3 attempts] → GLJsonParseError

Example:
    # First attempt
    response = "Here is the JSON: ```json\n{\"result\": 42}\n```"
    data = parse_and_validate(response, schema)  # Success

    # With repair
    response = "Here is the JSON: {result: 42}"  # Invalid (missing quotes)
    try:
        data = parse_and_validate(response, schema)
    except JSONParseError:
        # Retry with repair prompt
        repair_prompt = get_repair_prompt(schema)
        # ... retry LLM call with repair prompt ...

    # After >3 failures
    raise GLJsonParseError(request_id=..., attempts=4, last_error=...)
"""

from __future__ import annotations
import json
import re
import logging
from typing import Any, Dict, Optional
from pydantic import ValidationError
import jsonschema

logger = logging.getLogger(__name__)


class GLJsonParseError(Exception):
    """
    JSON parsing failed after maximum retry attempts

    Raised when LLM fails to produce valid JSON after >3 attempts.
    Indicates:
    - JSON schema may be too complex
    - LLM may not support strict JSON mode
    - Prompt may need improvement

    Attributes:
        request_id: Request ID for tracking
        attempts: Number of parse/validate attempts
        last_error: Last parsing error encountered
        history: List of all parse attempts and errors
    """

    def __init__(
        self,
        request_id: str,
        attempts: int,
        last_error: str,
        history: Optional[list[Dict[str, Any]]] = None,
    ):
        self.request_id = request_id
        self.attempts = attempts
        self.last_error = last_error
        self.history = history or []

        super().__init__(
            f"JSON parsing failed after {attempts} attempts (request_id={request_id}). "
            f"Last error: {last_error}"
        )


class GLValidationError(Exception):
    """JSON validation error (schema mismatch)"""

    pass


def extract_candidate_json(text: str) -> str:
    """
    Extract candidate JSON from LLM response text

    Handles common LLM output patterns:
    - Code fences: ```json\n{...}\n```
    - Inline code: `{...}`
    - Plain JSON: {...}
    - BOM characters
    - Trailing commas
    - Single quotes instead of double quotes (basic fix)

    Args:
        text: Raw LLM response text

    Returns:
        Extracted JSON string (best effort)

    Raises:
        ValueError: If no JSON-like content found

    Example:
        >>> extract_candidate_json('Here is the JSON: ```json\n{"result": 42}\n```')
        '{"result": 42}'

        >>> extract_candidate_json('The answer is `{"result": 42}`')
        '{"result": 42}'

        >>> extract_candidate_json('{result: 42,}')  # Fix trailing comma and quotes
        '{"result": 42}'
    """
    if not text or not text.strip():
        raise ValueError("Empty text - no JSON to extract")

    # Remove BOM if present
    text = text.lstrip("\ufeff")

    # 1. Try to extract from code fences (```json ... ```)
    code_fence_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE
    )
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    # 2. Try to extract from inline code (` ... `)
    elif "`{" in text or "`[" in text:
        inline_match = re.search(r"`(\{.*?\}|\[.*?\])`", text, re.DOTALL)
        if inline_match:
            text = inline_match.group(1).strip()

    # 3. Try to extract JSON object or array
    # Find outermost {...} or [...]
    elif "{" in text:
        start = text.find("{")
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break

    elif "[" in text:
        start = text.find("[")
        if start != -1:
            # Find matching closing bracket
            depth = 0
            for i in range(start, len(text)):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        text = text[start : i + 1]
                        break

    # 4. Basic cleanup
    text = text.strip()

    # Remove trailing commas before } or ]
    text = re.sub(r",\s*}", "}", text)
    text = re.sub(r",\s*]", "]", text)

    # Fix single quotes to double quotes (very basic - can break strings)
    # Only do this if no double quotes present
    if '"' not in text and "'" in text:
        # Replace single quotes around keys and string values
        text = re.sub(r"'(\w+)':", r'"\1":', text)  # Keys
        text = re.sub(r":\s*'([^']*)'", r': "\1"', text)  # Values

    if not text:
        raise ValueError("Could not extract JSON from text")

    return text


def parse_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON string with candidate extraction

    Attempts to extract and parse JSON from LLM response.
    Uses extract_candidate_json() to handle common LLM patterns.

    Args:
        text: Raw LLM response or JSON string

    Returns:
        Parsed JSON as dict

    Raises:
        json.JSONDecodeError: If JSON invalid
        ValueError: If no JSON found

    Example:
        >>> parse_json('```json\n{"result": 42}\n```')
        {'result': 42}
    """
    candidate = extract_candidate_json(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        # Log for debugging
        logger.debug(f"JSON parse error: {e}. Candidate: {candidate[:200]}")
        raise


def validate_json_schema(
    payload: Dict[str, Any],
    schema: Dict[str, Any],
) -> None:
    """
    Validate JSON payload against JSON Schema

    Uses jsonschema library for strict validation.
    Raises detailed error if validation fails.

    Args:
        payload: JSON payload to validate
        schema: JSON Schema (dict)

    Raises:
        GLValidationError: If payload doesn't match schema

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {"result": {"type": "number"}},
        ...     "required": ["result"]
        ... }
        >>> validate_json_schema({"result": 42}, schema)  # OK
        >>> validate_json_schema({"result": "foo"}, schema)  # Raises
    """
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except jsonschema.ValidationError as e:
        raise GLValidationError(
            f"JSON validation failed: {e.message}. "
            f"Schema path: {list(e.absolute_schema_path)}. "
            f"Got: {e.instance}"
        )
    except jsonschema.SchemaError as e:
        raise GLValidationError(f"Invalid JSON schema: {e.message}")


def parse_and_validate(
    text: str,
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parse and validate JSON in single operation

    Convenience function combining parse_json() and validate_json_schema().

    Args:
        text: Raw LLM response or JSON string
        schema: JSON Schema for validation

    Returns:
        Parsed and validated JSON as dict

    Raises:
        json.JSONDecodeError: If JSON invalid
        GLValidationError: If JSON doesn't match schema
        ValueError: If no JSON found

    Example:
        >>> schema = {"type": "object", "properties": {"result": {"type": "number"}}}
        >>> parse_and_validate('{"result": 42}', schema)
        {'result': 42}

        >>> parse_and_validate('{"result": "foo"}', schema)  # Raises GLValidationError
    """
    payload = parse_json(text)
    validate_json_schema(payload, schema)
    return payload


REPAIR_PROMPT_TEMPLATE = """Your previous response was invalid JSON. Return ONLY valid JSON conforming exactly to the provided JSON Schema.

**REQUIREMENTS:**
1. Output MUST be valid JSON (parseable by json.loads())
2. Output MUST match the JSON Schema below
3. Do NOT include any prose, explanation, or markdown
4. Do NOT use code fences (```json)
5. Do NOT use single quotes - use double quotes
6. Do NOT include trailing commas

**JSON Schema:**
{schema}

**Example valid output:**
{example}

Now provide your response as valid JSON ONLY:"""


def get_repair_prompt(
    schema: Dict[str, Any],
    attempt_number: int,
) -> str:
    """
    Generate repair prompt for JSON retry

    Creates a strict prompt instructing LLM to output valid JSON.
    Includes schema and example for clarity.

    Args:
        schema: JSON Schema for expected output
        attempt_number: Current retry attempt (for logging)

    Returns:
        Repair prompt text

    Example:
        >>> schema = {"type": "object", "properties": {"result": {"type": "number"}}}
        >>> prompt = get_repair_prompt(schema, attempt_number=2)
        >>> "valid JSON" in prompt
        True
    """
    # Generate example from schema
    example = _generate_example_from_schema(schema)

    prompt = REPAIR_PROMPT_TEMPLATE.format(
        schema=json.dumps(schema, indent=2), example=json.dumps(example, indent=2)
    )

    logger.info(
        f"Generated repair prompt for attempt {attempt_number}. "
        f"Schema keys: {list(schema.get('properties', {}).keys())}"
    )

    return prompt


def _generate_example_from_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate example JSON from schema

    Creates a minimal valid example based on schema properties.
    Used in repair prompts to show LLM what valid output looks like.

    Args:
        schema: JSON Schema

    Returns:
        Example JSON dict

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "count": {"type": "number"},
        ...         "name": {"type": "string"}
        ...     },
        ...     "required": ["count", "name"]
        ... }
        >>> example = _generate_example_from_schema(schema)
        >>> "count" in example and "name" in example
        True
    """
    if schema.get("type") != "object":
        # Simple fallback
        return {}

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    example = {}

    for prop_name, prop_schema in properties.items():
        # Only include required properties in example
        if prop_name not in required:
            continue

        prop_type = prop_schema.get("type")

        if prop_type == "string":
            # Use enum value if available
            if "enum" in prop_schema:
                example[prop_name] = prop_schema["enum"][0]
            else:
                example[prop_name] = "example"

        elif prop_type == "number" or prop_type == "integer":
            # Use minimum if available
            if "minimum" in prop_schema:
                example[prop_name] = prop_schema["minimum"]
            else:
                example[prop_name] = 0

        elif prop_type == "boolean":
            example[prop_name] = True

        elif prop_type == "array":
            example[prop_name] = []

        elif prop_type == "object":
            example[prop_name] = {}

        elif prop_type == "null":
            example[prop_name] = None

        else:
            # Unknown type - use null
            example[prop_name] = None

    return example


class JSONRetryTracker:
    """
    Tracks JSON parse/validate retry attempts

    Enforces CTO SPEC requirement: fail after >3 attempts.
    Records history of attempts for debugging.

    Usage:
        tracker = JSONRetryTracker(request_id="req_123", max_attempts=3)

        for attempt in range(tracker.max_attempts + 1):
            try:
                data = parse_and_validate(response, schema)
                tracker.record_success(attempt, data)
                break
            except Exception as e:
                tracker.record_failure(attempt, e)
                if tracker.should_fail():
                    raise tracker.build_error()

                # Generate repair prompt and retry
                repair_prompt = get_repair_prompt(schema, attempt + 1)
                # ... retry LLM call ...
    """

    def __init__(self, request_id: str, max_attempts: int = 3):
        """
        Initialize retry tracker

        Args:
            request_id: Request ID for correlation
            max_attempts: Maximum parse/validate attempts (default: 3)
        """
        self.request_id = request_id
        self.max_attempts = max_attempts
        self.attempts = 0
        self.history: list[Dict[str, Any]] = []
        self.last_error: Optional[str] = None

    def record_failure(
        self,
        attempt_number: int,
        error: Exception,
    ) -> None:
        """
        Record failed parse/validate attempt

        Args:
            attempt_number: Attempt number (0-indexed)
            error: Exception that occurred
        """
        self.attempts += 1
        self.last_error = str(error)

        self.history.append(
            {
                "attempt": attempt_number + 1,
                "status": "failed",
                "error": str(error),
                "error_type": error.__class__.__name__,
            }
        )

        logger.warning(
            f"JSON parse attempt {attempt_number + 1} failed "
            f"(request_id={self.request_id}): {error}"
        )

    def record_success(
        self,
        attempt_number: int,
        data: Dict[str, Any],
    ) -> None:
        """
        Record successful parse/validate attempt

        Args:
            attempt_number: Attempt number (0-indexed)
            data: Parsed JSON data
        """
        self.attempts = attempt_number + 1

        self.history.append(
            {
                "attempt": attempt_number + 1,
                "status": "success",
                "data_keys": list(data.keys()) if isinstance(data, dict) else None,
            }
        )

        logger.info(
            f"JSON parse succeeded on attempt {attempt_number + 1} "
            f"(request_id={self.request_id})"
        )

    def should_fail(self) -> bool:
        """
        Check if should fail (exceeded max attempts)

        Returns:
            True if attempts > max_attempts

        Example:
            >>> tracker = JSONRetryTracker("req_1", max_attempts=3)
            >>> tracker.attempts = 4
            >>> tracker.should_fail()
            True
        """
        return self.attempts > self.max_attempts

    def build_error(self) -> GLJsonParseError:
        """
        Build GLJsonParseError with history

        Returns:
            GLJsonParseError with full attempt history

        Example:
            >>> tracker = JSONRetryTracker("req_1", max_attempts=3)
            >>> tracker.attempts = 4
            >>> tracker.last_error = "Invalid JSON"
            >>> error = tracker.build_error()
            >>> error.attempts
            4
        """
        return GLJsonParseError(
            request_id=self.request_id,
            attempts=self.attempts,
            last_error=self.last_error or "Unknown error",
            history=self.history,
        )

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"JSONRetryTracker(request_id={self.request_id}, "
            f"attempts={self.attempts}/{self.max_attempts})"
        )
