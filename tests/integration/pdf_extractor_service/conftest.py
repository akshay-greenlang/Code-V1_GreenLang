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

import pytest


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
