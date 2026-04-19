# -*- coding: utf-8 -*-
"""
Shared fixtures for validation tests.
"""

import pytest
from pathlib import Path
import json


@pytest.fixture
def sample_valid_data():
    """Sample valid data for testing."""
    return {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "status": "active",
        "tags": ["developer", "python"],
        "metadata": {
            "created_at": "2024-01-01",
            "updated_at": "2024-01-15"
        }
    }


@pytest.fixture
def sample_invalid_data():
    """Sample invalid data for testing."""
    return {
        "age": -5,  # Invalid negative age
        "email": "not-an-email",  # Invalid format
        "status": "unknown"  # Invalid status
    }


@pytest.fixture
def temp_schema_file(tmp_path):
    """Create a temporary schema file."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "email"]
    }

    file_path = tmp_path / "schema.json"
    with open(file_path, 'w') as f:
        json.dump(schema, f)

    return file_path
