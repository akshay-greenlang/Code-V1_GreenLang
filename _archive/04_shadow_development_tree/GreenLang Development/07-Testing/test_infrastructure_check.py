"""Simple test to verify test infrastructure is working."""

import pytest


def test_basic_import():
    """Test that we can import basic modules."""
    import sys
    import os
    assert sys is not None
    assert os is not None


def test_pytest_is_working():
    """Test that pytest itself is functioning."""
    assert True


def test_simple_math():
    """Test basic arithmetic to verify execution."""
    assert 2 + 2 == 4
    assert 10 * 5 == 50


def test_fixture_access(fixtures_dir):
    """Test that we can access pytest fixtures."""
    assert fixtures_dir is not None


@pytest.mark.parametrize("a, b, expected", [
    (1, 2, 3),
    (5, 10, 15),
    (-1, 1, 0),
])
def test_parameterized(a, b, expected):
    """Test that parameterized tests work."""
    assert a + b == expected


class TestClassBasedTests:
    """Test that class-based tests work."""

    def test_in_class(self):
        """Test method in test class."""
        assert "test" in "test_string"

    def test_another_in_class(self):
        """Another test method in test class."""
        assert len("hello") == 5