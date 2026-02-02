# -*- coding: utf-8 -*-
"""
Shared fixtures for I/O tests.
"""

import pytest
import json
import csv
from pathlib import Path


@pytest.fixture
def sample_records():
    """Sample records for testing."""
    return [
        {"id": 1, "name": "Alice", "age": 30, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 25, "city": "Boston"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"}
    ]


@pytest.fixture
def sample_json_object():
    """Sample JSON object."""
    return {
        "user": {
            "name": "John Doe",
            "email": "john@example.com",
            "preferences": {
                "theme": "dark",
                "language": "en"
            }
        },
        "settings": {
            "notifications": True,
            "auto_save": False
        }
    }


@pytest.fixture
def create_test_files(tmp_path):
    """Factory fixture to create test files."""
    def _create(format_type, data):
        """
        Create a test file of specified format.

        Args:
            format_type: File format (json, csv, txt, etc.)
            data: Data to write

        Returns:
            Path to created file
        """
        file_path = tmp_path / f"test_data.{format_type}"

        if format_type == "json":
            with open(file_path, 'w') as f:
                json.dump(data, f)
        elif format_type == "csv":
            with open(file_path, 'w', newline='') as f:
                if data and isinstance(data[0], dict):
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        elif format_type == "txt":
            file_path.write_text(str(data))

        return file_path

    return _create


@pytest.fixture
def large_dataset():
    """Generate large dataset for performance testing."""
    return [
        {
            "id": i,
            "name": f"User_{i}",
            "value": i * 10,
            "category": f"cat_{i % 5}"
        }
        for i in range(1000)
    ]
