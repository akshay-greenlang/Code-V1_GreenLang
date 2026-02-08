# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Excel Normalizer Service Integration Tests (AGENT-DATA-002)
===============================================================================

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
    """Override parent conftest's mock_agents for Excel normalizer tests.

    Excel normalizer service does not use greenlang.agents.registry, so we
    provide a simple no-op override to avoid the AttributeError from
    the parent integration conftest.
    """
    yield {}


@pytest.fixture
def integration_config() -> Dict[str, Any]:
    """Configuration for integration testing."""
    return {
        "service_name": "excel-normalizer-integration-test",
        "environment": "test",
        "max_file_size_mb": 50,
        "max_rows_per_sheet": 100000,
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "default_mapping_strategy": "fuzzy",
        "fuzzy_match_threshold": 0.75,
        "min_quality_score": 0.5,
        "completeness_weight": 0.4,
        "accuracy_weight": 0.35,
        "consistency_weight": 0.25,
    }
