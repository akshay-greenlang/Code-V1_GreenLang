# -*- coding: utf-8 -*-
"""
Pytest Fixtures for PDF Extractor Service Integration Tests (AGENT-DATA-001)
=============================================================================

Provides shared fixtures for integration tests.
Overrides parent conftest's autouse fixtures that are not relevant.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest's mock_agents for PDF extractor tests.

    PDF extractor service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "pdf-extractor-integration-test",
        "environment": "test",
        "default_ocr_engine": "tesseract",
        "max_pages_per_document": 100,
        "max_file_size_mb": 50,
        "enable_line_item_extraction": True,
        "enable_cross_field_validation": True,
        "enable_document_classification": True,
    }
