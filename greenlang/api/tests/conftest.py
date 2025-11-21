# -*- coding: utf-8 -*-
"""
greenlang/api/tests/conftest.py

Pytest configuration and shared fixtures.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def test_app():
    """Application fixture for testing"""
    from greenlang.api.main import app
    return app


@pytest.fixture
def client(test_app):
    """Test client fixture"""
    return TestClient(test_app)


@pytest.fixture
def sample_calculation_request():
    """Sample calculation request fixture"""
    return {
        "fuel_type": "diesel",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US",
        "scope": "1",
        "boundary": "combustion"
    }


@pytest.fixture
def sample_batch_request():
    """Sample batch calculation request fixture"""
    return {
        "calculations": [
            {
                "fuel_type": "diesel",
                "activity_amount": 100,
                "activity_unit": "gallons",
                "geography": "US"
            },
            {
                "fuel_type": "natural_gas",
                "activity_amount": 500,
                "activity_unit": "therms",
                "geography": "US"
            }
        ]
    }
