"""
Comprehensive tests for SDK Transform abstraction.

Tests cover:
- Transform initialization
- Apply method functionality
- Callable interface
- Type handling
- Error handling
- Composition patterns
"""

import pytest
from typing import List, Dict, Any
from greenlang.sdk.base import Transform


class DoubleTransform(Transform[int, int]):
    """Simple transform that doubles a number."""

    def apply(self, data: int) -> int:
        """Double the input value."""
        return data * 2


class StringUpperTransform(Transform[str, str]):
    """Transform that converts string to uppercase."""

    def apply(self, data: str) -> str:
        """Convert to uppercase."""
        return data.upper()


class DictFilterTransform(Transform[Dict[str, Any], Dict[str, Any]]):
    """Transform that filters dictionary keys."""

    def __init__(self, allowed_keys: List[str]):
        """Initialize with allowed keys."""
        self.allowed_keys = allowed_keys

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter dictionary to only allowed keys."""
        return {k: v for k, v in data.items() if k in self.allowed_keys}


class FailingTransform(Transform[Any, Any]):
    """Transform that always raises an exception."""

    def apply(self, data: Any) -> Any:
        """Always raise an exception."""
        raise ValueError("Transform failed intentionally")


class ChainableTransform(Transform[int, int]):
    """Transform that can be chained."""

    def __init__(self, operation: str, value: int):
        """Initialize with operation and value."""
        self.operation = operation
        self.value = value

    def apply(self, data: int) -> int:
        """Apply operation."""
        if self.operation == "add":
            return data + self.value
        elif self.operation == "multiply":
            return data * self.value
        elif self.operation == "subtract":
            return data - self.value
        else:
            return data


@pytest.mark.unit
class TestTransformBasics:
    """Test basic transform functionality."""

    def test_double_transform_apply(self):
        """Test applying a simple numeric transform."""
        transform = DoubleTransform()
        result = transform.apply(5)
        assert result == 10

    def test_double_transform_multiple_values(self):
        """Test transform with different input values."""
        transform = DoubleTransform()

        assert transform.apply(0) == 0
        assert transform.apply(-5) == -10
        assert transform.apply(100) == 200

    def test_string_transform_apply(self):
        """Test applying a string transform."""
        transform = StringUpperTransform()
        result = transform.apply("hello")
        assert result == "HELLO"

    def test_string_transform_various_inputs(self):
        """Test string transform with various inputs."""
        transform = StringUpperTransform()

        assert transform.apply("") == ""
        assert transform.apply("MiXeD") == "MIXED"
        assert transform.apply("123abc") == "123ABC"


@pytest.mark.unit
class TestTransformCallable:
    """Test transform callable interface."""

    def test_transform_is_callable(self):
        """Test that transform can be called directly."""
        transform = DoubleTransform()
        result = transform(7)
        assert result == 14

    def test_callable_equals_apply(self):
        """Test that calling transform equals apply method."""
        transform = StringUpperTransform()
        input_data = "test"

        assert transform(input_data) == transform.apply(input_data)

    def test_callable_with_complex_data(self):
        """Test callable with complex data types."""
        transform = DictFilterTransform(["a", "b"])
        data = {"a": 1, "b": 2, "c": 3}
        result = transform(data)

        assert result == {"a": 1, "b": 2}


@pytest.mark.unit
class TestDictFilterTransform:
    """Test dictionary filtering transform."""

    def test_filter_single_key(self):
        """Test filtering to single key."""
        transform = DictFilterTransform(["name"])
        data = {"name": "Alice", "age": 30, "city": "NYC"}
        result = transform.apply(data)

        assert result == {"name": "Alice"}

    def test_filter_multiple_keys(self):
        """Test filtering to multiple keys."""
        transform = DictFilterTransform(["name", "age"])
        data = {"name": "Bob", "age": 25, "city": "LA", "country": "USA"}
        result = transform.apply(data)

        assert result == {"name": "Bob", "age": 25}

    def test_filter_no_matching_keys(self):
        """Test filtering when no keys match."""
        transform = DictFilterTransform(["missing"])
        data = {"name": "Charlie", "age": 35}
        result = transform.apply(data)

        assert result == {}

    def test_filter_all_keys(self):
        """Test when all keys are allowed."""
        transform = DictFilterTransform(["a", "b", "c"])
        data = {"a": 1, "b": 2, "c": 3}
        result = transform.apply(data)

        assert result == data

    def test_filter_empty_dict(self):
        """Test filtering empty dictionary."""
        transform = DictFilterTransform(["a", "b"])
        result = transform.apply({})

        assert result == {}


@pytest.mark.unit
class TestTransformErrorHandling:
    """Test transform error handling."""

    def test_failing_transform_raises(self):
        """Test that failing transform raises exception."""
        transform = FailingTransform()

        with pytest.raises(ValueError, match="Transform failed"):
            transform.apply("any data")

    def test_failing_transform_callable_raises(self):
        """Test that failing transform raises via callable."""
        transform = FailingTransform()

        with pytest.raises(ValueError, match="Transform failed"):
            transform("any data")

    def test_transform_type_error(self):
        """Test transform with wrong type raises error."""
        transform = DoubleTransform()

        # This should raise TypeError since we can't multiply string
        with pytest.raises(TypeError):
            transform.apply("not a number")


@pytest.mark.unit
class TestTransformComposition:
    """Test composing multiple transforms."""

    def test_manual_composition(self):
        """Test manually composing transforms."""
        add_five = ChainableTransform("add", 5)
        multiply_two = ChainableTransform("multiply", 2)

        result = multiply_two(add_five(10))  # (10 + 5) * 2 = 30
        assert result == 30

    def test_three_transform_chain(self):
        """Test chaining three transforms."""
        add_10 = ChainableTransform("add", 10)
        multiply_3 = ChainableTransform("multiply", 3)
        subtract_5 = ChainableTransform("subtract", 5)

        # ((5 + 10) * 3) - 5 = 40
        result = subtract_5(multiply_3(add_10(5)))
        assert result == 40

    def test_identity_composition(self):
        """Test composing with identity-like operations."""
        add_zero = ChainableTransform("add", 0)
        multiply_one = ChainableTransform("multiply", 1)

        input_val = 42
        result = multiply_one(add_zero(input_val))
        assert result == input_val


@pytest.mark.unit
class TestTransformEdgeCases:
    """Test transform edge cases."""

    def test_transform_with_none(self):
        """Test transform behavior with None."""
        # Note: This depends on implementation details
        # Some transforms may handle None, others may not
        transform = DictFilterTransform(["key"])

        with pytest.raises((AttributeError, TypeError)):
            transform.apply(None)

    def test_transform_idempotency(self):
        """Test that applying transform twice gives expected result."""
        transform = StringUpperTransform()
        data = "hello"

        first_result = transform(data)
        second_result = transform(first_result)

        # Applying uppercase twice should give same result
        assert first_result == second_result == "HELLO"

    def test_transform_preserves_empty_values(self):
        """Test transform preserves empty but valid values."""
        transform = StringUpperTransform()

        assert transform("") == ""

    def test_chainable_operations_order(self):
        """Test that operation order matters in chaining."""
        add_5 = ChainableTransform("add", 5)
        multiply_2 = ChainableTransform("multiply", 2)

        # (10 + 5) * 2 = 30
        result1 = multiply_2(add_5(10))
        # (10 * 2) + 5 = 25
        result2 = add_5(multiply_2(10))

        assert result1 == 30
        assert result2 == 25
        assert result1 != result2


@pytest.mark.unit
class TestTransformStateful:
    """Test stateful transform behavior."""

    def test_filter_transform_state(self):
        """Test that filter transform maintains its state."""
        transform = DictFilterTransform(["a", "b"])

        # Apply to different data with same transform instance
        result1 = transform({"a": 1, "b": 2, "c": 3})
        result2 = transform({"a": 10, "b": 20, "c": 30})

        assert result1 == {"a": 1, "b": 2}
        assert result2 == {"a": 10, "b": 20}

    def test_chainable_transform_state(self):
        """Test that chainable transform maintains operation state."""
        add_7 = ChainableTransform("add", 7)

        # Multiple applications should use same operation
        assert add_7(10) == 17
        assert add_7(20) == 27
        assert add_7(0) == 7
