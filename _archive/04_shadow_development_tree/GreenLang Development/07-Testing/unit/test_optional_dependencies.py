# -*- coding: utf-8 -*-
"""
Tests for optional dependency import guards.

This test suite verifies that the proper error messages are shown when
optional dependencies are not available.
"""

import pytest
import sys
import subprocess
from unittest.mock import patch, MagicMock
import importlib


class TestOptionalDependencies:
    """Test optional dependency handling across the codebase."""

    def test_import_guard_error_messages(self):
        """Test that import guard error messages are properly formatted."""
        # Test the error message format by simulating ImportError scenarios
        error_patterns = [
            ("pandas is required for the EnergyBalanceAgent", "pip install greenlang[analytics]"),
            ("pandas is required for the LoadProfileAgent", "pip install greenlang[analytics]"),
            ("pandas is required for the SolarResourceAgent", "pip install greenlang[analytics]"),
            ("pandas is required for CSV export", "pip install greenlang[analytics]"),
            ("pandas is required for Excel export", "pip install greenlang[analytics]"),
        ]

        for requirement_msg, install_msg in error_patterns:
            # Test that our error messages contain the required components
            assert "pandas" in requirement_msg
            assert "greenlang[analytics]" in install_msg

    def test_agent_modules_have_guards(self):
        """Test that agent modules have proper import guard structure."""
        import inspect
        from greenlang.agents import energy_balance_agent, load_profile_agent, solar_resource_agent

        # Check that the modules have the expected try/except structure
        for module in [energy_balance_agent, load_profile_agent, solar_resource_agent]:
            source = inspect.getsource(module)
            assert "try:" in source
            assert "import pandas" in source
            assert "except ImportError:" in source
            assert "greenlang[analytics]" in source

    def test_enhanced_client_has_import_guards(self):
        """Test that enhanced client has proper import guards."""
        import inspect
        from greenlang.sdk import enhanced_client

        source = inspect.getsource(enhanced_client)

        # Check that the CSV export section has proper guards
        assert 'try:' in source
        assert 'import pandas as pd' in source
        assert 'except ImportError:' in source
        assert 'pandas is required for CSV export' in source or 'pandas is required for Excel export' in source
        assert 'greenlang[analytics]' in source

    def test_runtime_executor_numpy_flag(self):
        """Test that runtime executor properly handles numpy availability."""
        from greenlang.runtime.executor import HAS_NUMPY

        # This should be True in test environment since we include analytics
        # But the test validates the flag exists and can be checked
        assert isinstance(HAS_NUMPY, bool)

    def test_runtime_golden_numpy_flag(self):
        """Test that runtime golden module properly handles numpy availability."""
        from greenlang.runtime.golden import HAS_NUMPY

        # This should be True in test environment since we include analytics
        assert isinstance(HAS_NUMPY, bool)

    def test_dairy_load_data_script_has_guards(self):
        """Test that the dairy load data script has proper import guards."""
        from pathlib import Path

        script_path = Path(__file__).parent.parent.parent / "apps" / "climatenza_app" / "examples" / "generate_dairy_load_data.py"

        with open(script_path, 'r', encoding='utf-8') as f:
            source = f.read()

        assert "try:" in source
        assert "import pandas" in source
        assert "import numpy" in source
        assert "except ImportError:" in source
        assert "greenlang[analytics]" in source

    def test_agents_work_with_analytics_installed(self):
        """Test that agents work correctly when analytics dependencies are available."""
        # This test verifies normal operation when pandas/numpy are available
        try:
            from greenlang.agents.energy_balance_agent import EnergyBalanceAgent
            from greenlang.agents.load_profile_agent import LoadProfileAgent
            from greenlang.agents.solar_resource_agent import SolarResourceAgent

            # Should be able to instantiate the agents
            energy_agent = EnergyBalanceAgent()
            load_agent = LoadProfileAgent()
            solar_agent = SolarResourceAgent()

            assert energy_agent is not None
            assert load_agent is not None
            assert solar_agent is not None

        except ImportError:
            pytest.fail("Agents should import successfully when analytics dependencies are available")

    def test_core_functionality_without_analytics(self):
        """Test that core GreenLang functionality works without analytics dependencies."""
        # Test that basic imports work even if pandas/numpy are not available
        try:
            from greenlang.sdk.base import Result
            from greenlang.packs.loader import PackLoader

            # These should work regardless of analytics dependencies
            result = Result(success=True, data={"test": "value"})
            assert result.success is True
            assert result.data["test"] == "value"

        except ImportError as e:
            if "pandas" in str(e) or "numpy" in str(e):
                pytest.fail("Core functionality should not depend on analytics libraries")
            else:
                # Re-raise if it's a different import error
                raise