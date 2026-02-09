# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Data Quality Profiler Service Unit Tests (AGENT-DATA-010)
=============================================================================

Provides shared fixtures for testing the data quality profiler config, models,
provenance tracker, metrics, dataset profiler, completeness analyzer, validity
checker, and related components.

All tests are self-contained with no external dependencies.

Includes a module-level stub for greenlang.data_quality_profiler.__init__
to bypass engine imports that may not yet be available, allowing direct
submodule imports to work.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
import sys
import types
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub the data_quality_profiler package to bypass broken __init__ imports.
# ---------------------------------------------------------------------------

_PKG_NAME = "greenlang.data_quality_profiler"

if _PKG_NAME not in sys.modules:
    import greenlang  # noqa: F401 ensure parent exists

    _stub = types.ModuleType(_PKG_NAME)
    _stub.__path__ = [
        os.path.join(os.path.dirname(greenlang.__file__), "data_quality_profiler")
    ]
    _stub.__package__ = _PKG_NAME
    _stub.__file__ = os.path.join(
        _stub.__path__[0], "__init__.py"
    )
    sys.modules[_PKG_NAME] = _stub


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_dq_env(monkeypatch):
    """Remove all GL_DQ_ env vars and reset config singleton between tests.

    This fixture runs automatically for every test in this package. It:
      1. Removes any existing GL_DQ_* environment variables.
      2. Resets the config singleton before yielding.
      3. Resets again after the test completes.
    """
    prefix = "GL_DQ_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    from greenlang.data_quality_profiler.config import reset_config
    reset_config()

    yield

    # Reset the config singleton so next test starts clean
    try:
        reset_config()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Sample dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_dataset() -> List[Dict[str, Any]]:
    """20 realistic rows with mixed columns for profiling tests.

    Columns:
      name (str), age (int), email (str), salary (float),
      date_joined (str), department (str), phone (str),
      is_active (bool), score (float), notes (str with some nulls)
    """
    return [
        {
            "name": "Alice Thompson", "age": 28, "email": "alice@greenlang.io",
            "salary": 72000.00, "date_joined": "2023-03-15",
            "department": "Engineering", "phone": "+1-555-0101",
            "is_active": True, "score": 0.92, "notes": "Lead engineer",
        },
        {
            "name": "Bob Martinez", "age": 35, "email": "bob@greenlang.io",
            "salary": 85000.00, "date_joined": "2021-07-01",
            "department": "Engineering", "phone": "+1-555-0102",
            "is_active": True, "score": 0.88, "notes": None,
        },
        {
            "name": "Carol Chen", "age": 42, "email": "carol@greenlang.io",
            "salary": 95000.00, "date_joined": "2019-11-20",
            "department": "Science", "phone": "+1-555-0103",
            "is_active": True, "score": 0.95, "notes": "Senior scientist",
        },
        {
            "name": "David Kim", "age": 31, "email": "david@greenlang.io",
            "salary": 68000.00, "date_joined": "2022-01-10",
            "department": "Operations", "phone": "+1-555-0104",
            "is_active": True, "score": 0.78, "notes": None,
        },
        {
            "name": "Eva Johansson", "age": 29, "email": "eva@greenlang.io",
            "salary": 71000.00, "date_joined": "2023-05-22",
            "department": "Engineering", "phone": "+1-555-0105",
            "is_active": True, "score": 0.85, "notes": "Frontend specialist",
        },
        {
            "name": "Frank Oduya", "age": 38, "email": "frank@greenlang.io",
            "salary": 92000.00, "date_joined": "2020-04-18",
            "department": "Science", "phone": "+1-555-0106",
            "is_active": True, "score": 0.91, "notes": "GHG analyst",
        },
        {
            "name": "Grace Liu", "age": 26, "email": "grace@greenlang.io",
            "salary": 63000.00, "date_joined": "2024-01-03",
            "department": "Operations", "phone": "+1-555-0107",
            "is_active": True, "score": 0.82, "notes": None,
        },
        {
            "name": "Hector Ruiz", "age": 44, "email": "hector@greenlang.io",
            "salary": 98000.00, "date_joined": "2018-06-12",
            "department": "Management", "phone": "+1-555-0108",
            "is_active": True, "score": 0.90, "notes": "VP Engineering",
        },
        {
            "name": "Ingrid Berg", "age": 33, "email": "ingrid@greenlang.io",
            "salary": 77000.00, "date_joined": "2022-09-08",
            "department": "Science", "phone": "+1-555-0109",
            "is_active": True, "score": 0.87, "notes": "CBAM specialist",
        },
        {
            "name": "James Okafor", "age": 30, "email": "james@greenlang.io",
            "salary": 74000.00, "date_joined": "2023-02-28",
            "department": "Engineering", "phone": "+1-555-0110",
            "is_active": True, "score": 0.84, "notes": None,
        },
        {
            "name": "Kara Singh", "age": 27, "email": "kara@greenlang.io",
            "salary": 66000.00, "date_joined": "2024-03-17",
            "department": "Operations", "phone": "+1-555-0111",
            "is_active": True, "score": 0.80, "notes": "Data ops",
        },
        {
            "name": "Leo Tanaka", "age": 39, "email": "leo@greenlang.io",
            "salary": 88000.00, "date_joined": "2020-10-05",
            "department": "Engineering", "phone": "+1-555-0112",
            "is_active": True, "score": 0.93, "notes": "Platform lead",
        },
        {
            "name": "Mia Schmidt", "age": 32, "email": "mia@greenlang.io",
            "salary": 76000.00, "date_joined": "2022-06-20",
            "department": "Science", "phone": "+1-555-0113",
            "is_active": True, "score": 0.86, "notes": None,
        },
        {
            "name": "Noah Williams", "age": 36, "email": "noah@greenlang.io",
            "salary": 82000.00, "date_joined": "2021-01-15",
            "department": "Engineering", "phone": "+1-555-0114",
            "is_active": False, "score": 0.79, "notes": "On leave",
        },
        {
            "name": "Olivia Brown", "age": 41, "email": "olivia@greenlang.io",
            "salary": 94000.00, "date_joined": "2019-08-30",
            "department": "Management", "phone": "+1-555-0115",
            "is_active": True, "score": 0.96, "notes": "Director",
        },
        {
            "name": "Pablo Mendez", "age": 25, "email": "pablo@greenlang.io",
            "salary": 60000.00, "date_joined": "2024-07-01",
            "department": "Operations", "phone": "+1-555-0116",
            "is_active": True, "score": 0.73, "notes": None,
        },
        {
            "name": "Quinn Foster", "age": 34, "email": "quinn@greenlang.io",
            "salary": 80000.00, "date_joined": "2021-12-10",
            "department": "Science", "phone": "+1-555-0117",
            "is_active": True, "score": 0.89, "notes": "Emissions lead",
        },
        {
            "name": "Rosa Petrov", "age": 37, "email": "rosa@greenlang.io",
            "salary": 86000.00, "date_joined": "2020-03-25",
            "department": "Engineering", "phone": "+1-555-0118",
            "is_active": True, "score": 0.91, "notes": None,
        },
        {
            "name": "Sam Adeyemi", "age": 29, "email": "sam@greenlang.io",
            "salary": 69000.00, "date_joined": "2023-10-08",
            "department": "Operations", "phone": "+1-555-0119",
            "is_active": True, "score": 0.81, "notes": "Supply chain",
        },
        {
            "name": "Tara Nolan", "age": 40, "email": "tara@greenlang.io",
            "salary": 91000.00, "date_joined": "2019-05-14",
            "department": "Management", "phone": "+1-555-0120",
            "is_active": True, "score": 0.94, "notes": "CTO",
        },
    ]


@pytest.fixture
def sample_dataset_with_issues() -> List[Dict[str, Any]]:
    """15 rows with intentional quality issues for negative testing.

    Issues included:
      - Null values in required fields (name, email, salary)
      - Type mismatches (age as string, salary as string)
      - Outlier values (salary = 999999, age = 200)
      - Duplicate rows (rows 5 and 6 identical)
      - Stale dates (date_joined = "2010-01-01")
      - Invalid email format ("not-an-email")
    """
    return [
        {
            "name": "Valid User", "age": 30, "email": "valid@greenlang.io",
            "salary": 75000.00, "date_joined": "2023-01-15",
            "department": "Engineering", "phone": "+1-555-0001",
            "is_active": True, "score": 0.85, "notes": "Normal row",
        },
        {
            "name": None, "age": 28, "email": "missing_name@greenlang.io",
            "salary": 70000.00, "date_joined": "2023-06-01",
            "department": "Science", "phone": "+1-555-0002",
            "is_active": True, "score": 0.80, "notes": "Missing name",
        },
        {
            "name": "Type Mismatch", "age": "twenty-five", "email": "types@greenlang.io",
            "salary": "not_a_number", "date_joined": "2022-11-20",
            "department": "Operations", "phone": "+1-555-0003",
            "is_active": True, "score": 0.70, "notes": "Type issues",
        },
        {
            "name": "Outlier Salary", "age": 35, "email": "outlier@greenlang.io",
            "salary": 999999.99, "date_joined": "2021-03-10",
            "department": "Engineering", "phone": "+1-555-0004",
            "is_active": True, "score": 0.90, "notes": "Extreme salary",
        },
        {
            "name": "Old Record", "age": 55, "email": "stale@greenlang.io",
            "salary": 65000.00, "date_joined": "2010-01-01",
            "department": "Management", "phone": "+1-555-0005",
            "is_active": False, "score": 0.60, "notes": "Very old record",
        },
        {
            "name": "Duplicate Row", "age": 32, "email": "dup@greenlang.io",
            "salary": 72000.00, "date_joined": "2022-08-15",
            "department": "Science", "phone": "+1-555-0006",
            "is_active": True, "score": 0.85, "notes": "Duplicate",
        },
        {
            "name": "Duplicate Row", "age": 32, "email": "dup@greenlang.io",
            "salary": 72000.00, "date_joined": "2022-08-15",
            "department": "Science", "phone": "+1-555-0006",
            "is_active": True, "score": 0.85, "notes": "Duplicate",
        },
        {
            "name": "Invalid Email", "age": 29, "email": "not-an-email",
            "salary": 68000.00, "date_joined": "2023-02-28",
            "department": "Operations", "phone": "+1-555-0007",
            "is_active": True, "score": 0.75, "notes": "Bad email",
        },
        {
            "name": "Missing Email", "age": 40, "email": None,
            "salary": 90000.00, "date_joined": "2020-07-12",
            "department": "Engineering", "phone": "+1-555-0008",
            "is_active": True, "score": 0.88, "notes": None,
        },
        {
            "name": "Outlier Age", "age": 200, "email": "age@greenlang.io",
            "salary": 80000.00, "date_joined": "2021-05-20",
            "department": "Science", "phone": "+1-555-0009",
            "is_active": True, "score": 0.82, "notes": "Impossible age",
        },
        {
            "name": "Missing Salary", "age": 33, "email": "nosalary@greenlang.io",
            "salary": None, "date_joined": "2022-04-10",
            "department": "Management", "phone": "+1-555-0010",
            "is_active": True, "score": 0.77, "notes": "No salary data",
        },
        {
            "name": "Negative Score", "age": 27, "email": "neg@greenlang.io",
            "salary": 62000.00, "date_joined": "2024-01-05",
            "department": "Operations", "phone": "+1-555-0011",
            "is_active": True, "score": -0.10, "notes": "Negative score",
        },
        {
            "name": "", "age": 31, "email": "empty_name@greenlang.io",
            "salary": 71000.00, "date_joined": "2023-09-18",
            "department": "Engineering", "phone": "+1-555-0012",
            "is_active": True, "score": 0.83, "notes": "Empty name",
        },
        {
            "name": "All Nulls", "age": None, "email": None,
            "salary": None, "date_joined": None,
            "department": None, "phone": None,
            "is_active": None, "score": None, "notes": None,
        },
        {
            "name": "Last Row", "age": 45, "email": "last@greenlang.io",
            "salary": 95000.00, "date_joined": "2019-12-01",
            "department": "Management", "phone": "+1-555-0014",
            "is_active": True, "score": 0.95, "notes": "Final row",
        },
    ]


# ---------------------------------------------------------------------------
# Sample Column Values Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_column_values() -> Dict[str, List[Any]]:
    """Dict mapping column names to representative value lists."""
    return {
        "name": [
            "Alice", "Bob", "Carol", "David", "Eva",
            "Frank", "Grace", "Hector", "Ingrid", "James",
        ],
        "age": [28, 35, 42, 31, 29, 38, 26, 44, 33, 30],
        "email": [
            "alice@greenlang.io", "bob@greenlang.io", "carol@greenlang.io",
            "david@greenlang.io", "eva@greenlang.io", "frank@greenlang.io",
            "grace@greenlang.io", "hector@greenlang.io", "ingrid@greenlang.io",
            "james@greenlang.io",
        ],
        "salary": [
            72000.0, 85000.0, 95000.0, 68000.0, 71000.0,
            92000.0, 63000.0, 98000.0, 77000.0, 74000.0,
        ],
        "department": [
            "Engineering", "Engineering", "Science", "Operations",
            "Engineering", "Science", "Operations", "Management",
            "Science", "Engineering",
        ],
        "is_active": [True, True, True, True, True, True, True, True, True, True],
        "score": [0.92, 0.88, 0.95, 0.78, 0.85, 0.91, 0.82, 0.90, 0.87, 0.84],
    }


# ---------------------------------------------------------------------------
# Sample Quality Rules Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_quality_rules() -> List[Dict[str, Any]]:
    """5 quality rules covering different rule types.

    Rules:
      1. Completeness rule: name column must be >= 95% non-null
      2. Range rule: age must be between 18 and 120
      3. Format rule: email must match a basic email regex
      4. Uniqueness rule: email column must have >= 99% uniqueness
      5. Freshness rule: date_joined must be within 365 days
    """
    return [
        {
            "name": "name_completeness",
            "description": "Name column must be at least 95% complete",
            "rule_type": "completeness",
            "column": "name",
            "operator": "greater_than",
            "threshold": 0.95,
            "parameters": {},
            "priority": 1,
        },
        {
            "name": "age_range",
            "description": "Age must be between 18 and 120",
            "rule_type": "range",
            "column": "age",
            "operator": "between",
            "threshold": None,
            "parameters": {"min_value": 18, "max_value": 120},
            "priority": 2,
        },
        {
            "name": "email_format",
            "description": "Email must match basic email format",
            "rule_type": "format",
            "column": "email",
            "operator": "matches",
            "threshold": None,
            "parameters": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
            "priority": 3,
        },
        {
            "name": "email_uniqueness",
            "description": "Email column must have at least 99% unique values",
            "rule_type": "uniqueness",
            "column": "email",
            "operator": "greater_than",
            "threshold": 0.99,
            "parameters": {},
            "priority": 4,
        },
        {
            "name": "date_freshness",
            "description": "Records must have been joined within 365 days",
            "rule_type": "freshness",
            "column": "date_joined",
            "operator": "less_than",
            "threshold": 365.0,
            "parameters": {"unit": "days"},
            "priority": 5,
        },
    ]


# ---------------------------------------------------------------------------
# Anomaly Data Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_anomaly_data() -> List[float]:
    """30 float values with 3 obvious outliers (values > 3 stddev).

    Normal range is approximately 96-106.
    Outliers: 500.0, 600.0, -200.0
    """
    return [
        100.0, 102.5, 98.3, 105.1, 97.8,
        103.2, 99.4, 101.7, 104.6, 96.5,
        100.8, 103.9, 97.1, 106.2, 95.4,
        102.0, 98.7, 104.3, 99.9, 101.1,
        105.5, 96.0, 103.5, 100.3, 98.0,
        500.0, 600.0, -200.0,
        101.5, 99.2,
    ]


# ---------------------------------------------------------------------------
# Additional dataset fixtures (backward compat for existing tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_numeric_data() -> List[Dict[str, Any]]:
    """Dataset with numeric columns for statistical testing."""
    return [
        {"value": 10, "temperature": 20.5, "count": 100},
        {"value": 20, "temperature": 22.3, "count": 150},
        {"value": 30, "temperature": 19.8, "count": 200},
        {"value": 40, "temperature": 25.1, "count": 250},
        {"value": 50, "temperature": 23.7, "count": 300},
        {"value": 60, "temperature": 21.0, "count": 350},
        {"value": 70, "temperature": 24.6, "count": 400},
        {"value": 80, "temperature": 18.9, "count": 450},
        {"value": 90, "temperature": 26.2, "count": 500},
        {"value": 100, "temperature": 22.0, "count": 550},
    ]


@pytest.fixture
def sample_all_null_column_data() -> List[Dict[str, Any]]:
    """Dataset where one column is entirely null."""
    return [
        {"name": "Alice", "notes": None},
        {"name": "Bob", "notes": None},
        {"name": "Charlie", "notes": None},
        {"name": "Diana", "notes": None},
        {"name": "Eve", "notes": None},
    ]


@pytest.fixture
def sample_email_data() -> List[Dict[str, Any]]:
    """Dataset with mix of valid and invalid emails."""
    return [
        {"email": "alice@example.com"},
        {"email": "bob@test.org"},
        {"email": "invalid-email"},
        {"email": "charlie@corp.net"},
        {"email": "no-at-sign"},
        {"email": "diana@mail.com"},
    ]


# ---------------------------------------------------------------------------
# Mock Prometheus Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """MagicMock for prometheus_client with Counter, Gauge, and Histogram stubs."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    mock_prom.generate_latest.return_value = (
        b"# HELP test_metric\n# TYPE test_metric counter\n"
    )
    return mock_prom
