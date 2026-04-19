# -*- coding: utf-8 -*-
"""Simple test to verify CI setup."""
import pytest


def test_basic_math():
    """Test basic math operations."""
    assert 2 + 2 == 4
    assert 5 * 3 == 15
    assert 10 / 2 == 5


def test_string_operations():
    """Test string operations."""
    assert "hello" + " " + "world" == "hello world"
    assert "test".upper() == "TEST"
    assert len("greenlang") == 9


@pytest.mark.integration
def test_integration_example():
    """Example integration test (skipped by default in CI)."""
    # This would be skipped during CI runs
    assert True


@pytest.mark.e2e
def test_e2e_example():
    """Example end-to-end test (skipped by default in CI)."""
    # This would be skipped during CI runs
    assert True


class TestCISetup:
    """Test class to verify CI setup."""

    def test_class_method(self):
        """Test method in a class."""
        result = {"status": "success", "value": 42}
        assert result["status"] == "success"
        assert result["value"] == 42

    def test_list_operations(self):
        """Test list operations."""
        items = [1, 2, 3, 4, 5]
        assert len(items) == 5
        assert sum(items) == 15
        assert max(items) == 5