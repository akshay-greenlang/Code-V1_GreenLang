"""
Custom Assertions
=================

Custom assertion helpers for GreenLang testing.

This module provides specialized assertions for validating agent results,
schemas, performance, caching, and LLM outputs.
"""

from typing import Any, Dict, List, Optional, Union
import time
import json
import jsonschema
from difflib import SequenceMatcher


def assert_agent_result_valid(
    result: Dict[str, Any],
    required_keys: Optional[List[str]] = None
):
    """
    Assert that an agent result is valid.

    Args:
        result: Agent execution result
        required_keys: List of required keys in result

    Raises:
        AssertionError: If result is invalid
    """
    if not isinstance(result, dict):
        raise AssertionError(f"Agent result must be a dictionary, got {type(result)}")

    if required_keys:
        missing_keys = set(required_keys) - set(result.keys())
        if missing_keys:
            raise AssertionError(f"Agent result missing required keys: {missing_keys}")


def assert_schema_valid(data: Any, schema: Union[Dict, type]):
    """
    Assert that data matches a JSON schema or Pydantic model.

    Args:
        data: Data to validate
        schema: JSON schema dict or Pydantic model class

    Raises:
        AssertionError: If validation fails
    """
    # Check if schema is a Pydantic model
    if hasattr(schema, 'model_validate'):
        try:
            schema.model_validate(data)
        except Exception as e:
            raise AssertionError(f"Pydantic validation failed: {e}")

    # Check if schema is a JSON schema
    elif isinstance(schema, dict):
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            raise AssertionError(f"JSON schema validation failed: {e}")

    else:
        raise ValueError(f"Unsupported schema type: {type(schema)}")


def assert_performance(
    execution_time: Optional[float] = None,
    memory_usage: Optional[int] = None,
    max_time: Optional[float] = None,
    max_memory: Optional[int] = None
):
    """
    Assert that performance metrics are within acceptable bounds.

    Args:
        execution_time: Actual execution time in seconds
        memory_usage: Actual memory usage in bytes
        max_time: Maximum acceptable execution time
        max_memory: Maximum acceptable memory usage

    Raises:
        AssertionError: If performance exceeds limits
    """
    if max_time is not None and execution_time is not None:
        if execution_time > max_time:
            raise AssertionError(
                f"Execution time {execution_time:.4f}s exceeded max {max_time:.4f}s"
            )

    if max_memory is not None and memory_usage is not None:
        if memory_usage > max_memory:
            raise AssertionError(
                f"Memory usage {memory_usage} bytes exceeded max {max_memory} bytes"
            )


def assert_cache_hit_rate(
    hits: int,
    misses: int,
    min_rate: float
):
    """
    Assert that cache hit rate meets minimum threshold.

    Args:
        hits: Number of cache hits
        misses: Number of cache misses
        min_rate: Minimum acceptable hit rate (0.0 to 1.0)

    Raises:
        AssertionError: If hit rate is below minimum
    """
    total = hits + misses
    if total == 0:
        raise AssertionError("No cache operations recorded")

    hit_rate = hits / total

    if hit_rate < min_rate:
        raise AssertionError(
            f"Cache hit rate {hit_rate:.2%} below minimum {min_rate:.2%}"
        )


def assert_no_hallucination(
    response: str,
    source_data: Union[str, List[str]],
    min_similarity: float = 0.7
):
    """
    Assert that LLM response is grounded in source data (no hallucination).

    Args:
        response: LLM response
        source_data: Source data or list of source documents
        min_similarity: Minimum similarity threshold (0.0 to 1.0)

    Raises:
        AssertionError: If response appears to be hallucinated
    """
    if isinstance(source_data, str):
        source_data = [source_data]

    # Check if key facts from response appear in source data
    response_sentences = response.split('.')
    grounded_count = 0

    for sentence in response_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check similarity with any source document
        max_similarity = 0
        for source in source_data:
            similarity = SequenceMatcher(None, sentence.lower(), source.lower()).ratio()
            max_similarity = max(max_similarity, similarity)

        if max_similarity >= min_similarity:
            grounded_count += 1

    # At least 70% of sentences should be grounded
    if len(response_sentences) > 0:
        grounded_ratio = grounded_count / len(response_sentences)
        if grounded_ratio < 0.7:
            raise AssertionError(
                f"Response appears to be hallucinated. Only {grounded_ratio:.2%} "
                f"of sentences are grounded in source data."
            )


def assert_deterministic(
    results: List[Any],
    tolerance: float = 0.0
):
    """
    Assert that results are deterministic (identical or within tolerance).

    Args:
        results: List of results from multiple runs
        tolerance: Acceptable difference for numeric results (0.0 = exact match)

    Raises:
        AssertionError: If results are not deterministic
    """
    if len(results) < 2:
        raise ValueError("Need at least 2 results to check determinism")

    first_result = results[0]

    for i, result in enumerate(results[1:], 1):
        if isinstance(first_result, (int, float)) and isinstance(result, (int, float)):
            # Numeric comparison with tolerance
            if abs(first_result - result) > tolerance:
                raise AssertionError(
                    f"Results are not deterministic. "
                    f"Result {i} differs from first by {abs(first_result - result)}"
                )
        else:
            # Exact comparison
            if first_result != result:
                raise AssertionError(
                    f"Results are not deterministic. "
                    f"Result {i} differs from first result"
                )


def assert_cost_within_budget(
    actual_cost: float,
    budget: float,
    currency: str = "USD"
):
    """
    Assert that cost is within budget.

    Args:
        actual_cost: Actual cost incurred
        budget: Maximum allowed budget
        currency: Currency code

    Raises:
        AssertionError: If cost exceeds budget
    """
    if actual_cost > budget:
        raise AssertionError(
            f"Cost {currency} {actual_cost:.4f} exceeded budget {currency} {budget:.4f}"
        )


def assert_token_count(
    actual_tokens: int,
    max_tokens: int
):
    """
    Assert that token count is within limit.

    Args:
        actual_tokens: Actual tokens used
        max_tokens: Maximum allowed tokens

    Raises:
        AssertionError: If tokens exceed limit
    """
    if actual_tokens > max_tokens:
        raise AssertionError(
            f"Token count {actual_tokens} exceeded limit {max_tokens}"
        )


def assert_response_contains(
    response: str,
    expected_content: Union[str, List[str]],
    case_sensitive: bool = False
):
    """
    Assert that response contains expected content.

    Args:
        response: Response text
        expected_content: Expected content or list of expected contents
        case_sensitive: Whether to perform case-sensitive matching

    Raises:
        AssertionError: If expected content is missing
    """
    if isinstance(expected_content, str):
        expected_content = [expected_content]

    if not case_sensitive:
        response = response.lower()
        expected_content = [c.lower() for c in expected_content]

    missing = []
    for content in expected_content:
        if content not in response:
            missing.append(content)

    if missing:
        raise AssertionError(
            f"Response missing expected content: {missing}"
        )


def assert_response_not_contains(
    response: str,
    forbidden_content: Union[str, List[str]],
    case_sensitive: bool = False
):
    """
    Assert that response does not contain forbidden content.

    Args:
        response: Response text
        forbidden_content: Forbidden content or list of forbidden contents
        case_sensitive: Whether to perform case-sensitive matching

    Raises:
        AssertionError: If forbidden content is found
    """
    if isinstance(forbidden_content, str):
        forbidden_content = [forbidden_content]

    if not case_sensitive:
        response = response.lower()
        forbidden_content = [c.lower() for c in forbidden_content]

    found = []
    for content in forbidden_content:
        if content in response:
            found.append(content)

    if found:
        raise AssertionError(
            f"Response contains forbidden content: {found}"
        )


def assert_json_response(response: str) -> Dict[str, Any]:
    """
    Assert that response is valid JSON and return parsed data.

    Args:
        response: Response text

    Returns:
        Parsed JSON data

    Raises:
        AssertionError: If response is not valid JSON
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Response is not valid JSON: {e}")


def assert_list_length(
    data: List[Any],
    expected_length: Optional[int] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None
):
    """
    Assert that list length meets expectations.

    Args:
        data: List to check
        expected_length: Exact expected length
        min_length: Minimum expected length
        max_length: Maximum expected length

    Raises:
        AssertionError: If length requirements not met
    """
    actual_length = len(data)

    if expected_length is not None:
        if actual_length != expected_length:
            raise AssertionError(
                f"List length {actual_length} does not match expected {expected_length}"
            )

    if min_length is not None:
        if actual_length < min_length:
            raise AssertionError(
                f"List length {actual_length} is less than minimum {min_length}"
            )

    if max_length is not None:
        if actual_length > max_length:
            raise AssertionError(
                f"List length {actual_length} exceeds maximum {max_length}"
            )


def assert_field_type(
    data: Dict[str, Any],
    field: str,
    expected_type: type
):
    """
    Assert that a field has the expected type.

    Args:
        data: Dictionary containing the field
        field: Field name
        expected_type: Expected type

    Raises:
        AssertionError: If field type is incorrect
    """
    if field not in data:
        raise AssertionError(f"Field '{field}' not found in data")

    actual_type = type(data[field])
    if not isinstance(data[field], expected_type):
        raise AssertionError(
            f"Field '{field}' has type {actual_type}, expected {expected_type}"
        )


def assert_numeric_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    inclusive: bool = True
):
    """
    Assert that a numeric value is within expected range.

    Args:
        value: Value to check
        min_value: Minimum expected value
        max_value: Maximum expected value
        inclusive: Whether range is inclusive

    Raises:
        AssertionError: If value is outside range
    """
    if min_value is not None:
        if inclusive:
            if value < min_value:
                raise AssertionError(f"Value {value} is less than minimum {min_value}")
        else:
            if value <= min_value:
                raise AssertionError(f"Value {value} is not greater than {min_value}")

    if max_value is not None:
        if inclusive:
            if value > max_value:
                raise AssertionError(f"Value {value} exceeds maximum {max_value}")
        else:
            if value >= max_value:
                raise AssertionError(f"Value {value} is not less than {max_value}")
