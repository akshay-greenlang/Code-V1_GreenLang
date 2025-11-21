# -*- coding: utf-8 -*-
"""
GreenLang V2 - Pytest Configuration and Shared Fixtures
=======================================================

Global pytest configuration and shared test fixtures for V2 test suites.

Version: 2.0.0
Author: Testing & QA Team
"""

import pytest
import pandas as pd
import json
from pathlib import Path


# Configure pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers for V2 tests."""
    config.addinivalue_line("markers", "v2: V2 refactored agent tests")
    config.addinivalue_line("markers", "infrastructure: Infrastructure component tests")
    config.addinivalue_line("markers", "services: Shared service tests")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")


# Shared V2 fixtures
@pytest.fixture
def sample_data():
    """Sample data for V2 tests."""
    return [{"id": i, "value": i * 1.5} for i in range(100)]
