"""
Pytest fixtures and configuration for gl-normalizer-sdk tests.
"""

import pytest
from datetime import datetime, timezone
from typing import Any, Dict

import httpx


# Test API key
TEST_API_KEY = "test-api-key-12345"


@pytest.fixture
def api_key() -> str:
    """Fixture providing test API key."""
    return TEST_API_KEY


@pytest.fixture
def base_url() -> str:
    """Fixture providing test base URL."""
    return "https://test-api.greenlang.io"


@pytest.fixture
def normalize_response() -> Dict[str, Any]:
    """Fixture providing sample normalize response."""
    return {
        "source_record_id": "test-001",
        "status": "success",
        "canonical_measurements": [
            {
                "field": "energy_consumption",
                "dimension": "energy",
                "canonical_value": 360.0,
                "canonical_unit": "MJ",
                "raw_value": 100.0,
                "raw_unit": "kWh",
                "conversion_trace": {
                    "steps": [
                        {
                            "from_unit": "kWh",
                            "to_unit": "MJ",
                            "factor": 3.6,
                            "method": "multiply",
                        }
                    ],
                    "factor_version": "2026.01.0",
                },
                "warnings": [],
            }
        ],
        "normalized_entities": [],
        "audit": {
            "normalization_event_id": "norm-evt-abc123",
            "status": "success",
        },
    }


@pytest.fixture
def batch_response() -> Dict[str, Any]:
    """Fixture providing sample batch response."""
    return {
        "summary": {
            "total": 2,
            "success": 2,
            "failed": 0,
            "warnings": 0,
        },
        "results": [
            {
                "source_record_id": "batch-0",
                "status": "success",
                "canonical_measurements": [
                    {
                        "field": None,
                        "dimension": "energy",
                        "canonical_value": 360.0,
                        "canonical_unit": "MJ",
                        "raw_value": 100.0,
                        "raw_unit": "kWh",
                        "warnings": [],
                    }
                ],
                "normalized_entities": [],
                "audit": {"normalization_event_id": "norm-evt-001", "status": "success"},
                "errors": [],
                "warnings": [],
            },
            {
                "source_record_id": "batch-1",
                "status": "success",
                "canonical_measurements": [
                    {
                        "field": None,
                        "dimension": "mass",
                        "canonical_value": 50.0,
                        "canonical_unit": "kg",
                        "raw_value": 50.0,
                        "raw_unit": "kg",
                        "warnings": [],
                    }
                ],
                "normalized_entities": [],
                "audit": {"normalization_event_id": "norm-evt-002", "status": "success"},
                "errors": [],
                "warnings": [],
            },
        ],
    }


@pytest.fixture
def entity_response() -> Dict[str, Any]:
    """Fixture providing sample entity resolution response."""
    return {
        "best_match": {
            "reference_id": "GL-FUEL-NATGAS",
            "canonical_name": "Natural gas",
            "vocabulary_version": "2026.01.0",
            "match_method": "alias",
            "confidence": 1.0,
            "needs_review": False,
            "warnings": [],
        },
        "candidates": [
            {"reference_id": "GL-FUEL-NATGAS", "canonical_name": "Natural gas", "score": 1.0},
            {"reference_id": "GL-FUEL-LNG", "canonical_name": "Liquefied natural gas", "score": 0.65},
        ],
    }


@pytest.fixture
def job_response() -> Dict[str, Any]:
    """Fixture providing sample job response."""
    return {
        "job_id": "job-abc123",
        "status": "pending",
        "progress": 0.0,
        "total_items": 1000,
        "processed_items": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "results_url": None,
        "error": None,
    }


@pytest.fixture
def vocabularies_response() -> Dict[str, Any]:
    """Fixture providing sample vocabularies response."""
    return {
        "vocabularies": [
            {
                "vocabulary_id": "fuels",
                "name": "Fuel Types",
                "version": "2026.01.0",
                "entity_type": "fuel",
                "entity_count": 150,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": "Standard fuel type vocabulary",
            },
            {
                "vocabulary_id": "materials",
                "name": "Materials",
                "version": "2026.01.0",
                "entity_type": "material",
                "entity_count": 500,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "description": "Standard materials vocabulary",
            },
        ]
    }


@pytest.fixture
def validation_error_response() -> Dict[str, Any]:
    """Fixture providing sample validation error response."""
    return {
        "errors": [
            {
                "code": "GLNORM-E200",
                "severity": "error",
                "path": "/measurements/0",
                "message": "Dimension mismatch: expected 'energy', got 'mass'",
                "expected": {"dimension": "energy"},
                "actual": {"dimension": "mass", "unit": "kg", "value": 100},
                "hint": {
                    "suggestion": "Use energy units like kWh, MJ, or GJ",
                    "docs": "gl://docs/units#energy",
                },
            }
        ]
    }


@pytest.fixture
def rate_limit_error_response() -> Dict[str, Any]:
    """Fixture providing sample rate limit error response."""
    return {
        "errors": [
            {
                "code": "GLNORM-E900",
                "severity": "error",
                "message": "Rate limit exceeded",
            }
        ],
        "retry_after": 60.0,
    }
