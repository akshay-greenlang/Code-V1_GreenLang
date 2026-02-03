#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Release Smoke Tests
==================

Run after PyPI publish to verify the release works correctly.

These tests validate that the GreenLang package installs correctly
and core functionality is accessible.

Usage:
    # After installing from PyPI:
    pip install greenlang-cli==0.3.0
    pytest tests/smoke/test_release_smoke.py -v

    # Or run directly:
    python tests/smoke/test_release_smoke.py

Environment Variables:
    GL_EXPECTED_VERSION: Expected version to verify (default: from pyproject.toml)
    GL_SMOKE_STRICT: Set to "1" for strict mode (fail on warnings)
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest


# =============================================================================
# Configuration
# =============================================================================

# Expected version - can be overridden via environment variable
EXPECTED_VERSION = os.environ.get("GL_EXPECTED_VERSION", "0.3.0")

# Strict mode - fail on warnings
STRICT_MODE = os.environ.get("GL_SMOKE_STRICT", "0") == "1"

# Timeout for subprocess commands (seconds)
COMMAND_TIMEOUT = 30


# =============================================================================
# Helper Functions
# =============================================================================


def run_command(
    cmd: List[str],
    timeout: int = COMMAND_TIMEOUT,
    capture_output: bool = True,
    check: bool = False,
) -> Tuple[int, str, str]:
    """
    Run a command and return (returncode, stdout, stderr).

    Args:
        cmd: Command and arguments as a list
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        check: Whether to raise on non-zero exit

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=check,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout or "", e.stderr or ""
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"


def parse_version(version_str: str) -> Optional[str]:
    """
    Extract version number from a version string.

    Args:
        version_str: String containing version (e.g., "GreenLang v0.3.0")

    Returns:
        Extracted version or None
    """
    # Match patterns like "0.3.0", "v0.3.0", "0.3.0b1", "0.3.0-beta"
    match = re.search(r"v?(\d+\.\d+\.\d+(?:[ab]\d+|\.dev\d+|[-.]?(?:alpha|beta|rc)\d*)?)", version_str, re.IGNORECASE)
    return match.group(1) if match else None


# =============================================================================
# Test Classes
# =============================================================================


class TestInstallation:
    """Tests for CLI installation and basic functionality."""

    def test_cli_installed(self):
        """Verify the 'gl' CLI command is installed and accessible."""
        returncode, stdout, stderr = run_command(["gl", "--version"])

        assert returncode == 0, (
            f"CLI --version failed with code {returncode}.\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}\n"
            f"Make sure greenlang-cli is installed: pip install greenlang-cli"
        )

    def test_cli_version_correct(self):
        """Verify the CLI reports the expected version."""
        returncode, stdout, stderr = run_command(["gl", "--version"])

        assert returncode == 0, f"CLI --version failed: {stderr}"

        # Combine stdout and stderr for version check
        output = stdout + stderr
        version = parse_version(output)

        assert version is not None, (
            f"Could not parse version from output: {output}"
        )

        # Check version matches expected (allow for version prefix variations)
        assert version.startswith(EXPECTED_VERSION.lstrip("v")), (
            f"Version mismatch: expected {EXPECTED_VERSION}, got {version}"
        )

    def test_cli_help(self):
        """Verify the 'gl --help' command works."""
        returncode, stdout, stderr = run_command(["gl", "--help"])

        assert returncode == 0, f"CLI --help failed: {stderr}"

        # Check for expected help content
        output = stdout.lower()
        assert "greenlang" in output or "infrastructure" in output, (
            f"Help output missing expected content.\nOutput: {stdout}"
        )

    def test_cli_doctor(self):
        """Verify the 'gl doctor' command works."""
        returncode, stdout, stderr = run_command(["gl", "doctor"])

        assert returncode == 0, (
            f"CLI doctor failed with code {returncode}.\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
        )

        # Check for version information in output
        output = stdout + stderr
        assert "greenlang" in output.lower() or "version" in output.lower(), (
            f"Doctor output missing expected content.\nOutput: {output}"
        )

    def test_cli_version_command(self):
        """Verify the 'gl version' subcommand works."""
        returncode, stdout, stderr = run_command(["gl", "version"])

        assert returncode == 0, f"CLI version command failed: {stderr}"

    def test_greenlang_alias(self):
        """Verify the 'greenlang' alias command works."""
        returncode, stdout, stderr = run_command(["greenlang", "--version"])

        # Note: This might fail if only 'gl' is installed
        if returncode != 0:
            pytest.skip("greenlang alias not installed (gl command available instead)")

        assert "greenlang" in (stdout + stderr).lower()


class TestImports:
    """Tests for Python module imports."""

    def test_import_greenlang(self):
        """Verify greenlang package can be imported."""
        try:
            import greenlang
            assert hasattr(greenlang, "__version__"), "greenlang missing __version__"
        except ImportError as e:
            pytest.fail(f"Failed to import greenlang: {e}")

    def test_import_greenlang_version(self):
        """Verify greenlang reports correct version."""
        import greenlang

        version = greenlang.__version__
        assert version is not None, "greenlang.__version__ is None"
        assert version.startswith(EXPECTED_VERSION.lstrip("v")), (
            f"Version mismatch: expected {EXPECTED_VERSION}, got {version}"
        )

    def test_import_agents_base(self):
        """Verify base agent classes can be imported."""
        try:
            from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
            assert BaseAgent is not None
            assert AgentConfig is not None
            assert AgentResult is not None
        except ImportError as e:
            pytest.fail(f"Failed to import agent base classes: {e}")

    def test_import_pack_loader(self):
        """Verify pack loader can be imported."""
        try:
            from greenlang.ecosystem.packs.loader import PackLoader
            assert PackLoader is not None
        except ImportError as e:
            pytest.fail(f"Failed to import PackLoader: {e}")

    def test_import_cli_module(self):
        """Verify CLI module can be imported."""
        try:
            from greenlang.cli.main import app, main
            assert app is not None
            assert main is not None
        except ImportError as e:
            pytest.fail(f"Failed to import CLI module: {e}")

    def test_import_sdk_base(self):
        """Verify SDK base classes can be imported (if available)."""
        try:
            from greenlang.integration.sdk.base import Agent, Pipeline
            assert Agent is not None
            assert Pipeline is not None
        except ImportError:
            # SDK might be optional
            pytest.skip("SDK base classes not available (optional)")

    def test_import_execution_runtime(self):
        """Verify execution runtime can be imported (if available)."""
        try:
            from greenlang.execution.runtime.backends.local import LocalBackend
            assert LocalBackend is not None
        except ImportError:
            # Runtime might be optional
            pytest.skip("Execution runtime not available (optional)")


class TestBasicFunctionality:
    """Tests for basic package functionality."""

    def test_agent_instantiation(self):
        """Verify a basic agent can be instantiated."""
        from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
        from typing import Dict, Any

        class TestAgent(BaseAgent):
            """A simple test agent."""

            def execute(self, input_data: Dict[str, Any]) -> AgentResult:
                return AgentResult(
                    success=True,
                    data={"message": "Hello from test agent"},
                )

        # Create and run agent
        config = AgentConfig(
            name="TestSmokeAgent",
            description="Smoke test agent",
            version="1.0.0",
        )
        agent = TestAgent(config)

        # Verify agent properties
        assert agent.config.name == "TestSmokeAgent"
        assert agent.config.version == "1.0.0"

        # Run agent
        result = agent.run({"test": "data"})
        assert result.success is True
        assert "message" in result.data

    def test_pack_loader_initialization(self):
        """Verify PackLoader can be initialized."""
        try:
            from greenlang.ecosystem.packs.loader import PackLoader

            loader = PackLoader()
            assert loader is not None

            # Should be able to list available packs (might be empty)
            available = loader.list_available()
            assert isinstance(available, list)
        except Exception as e:
            pytest.fail(f"PackLoader initialization failed: {e}")

    def test_config_directory_creation(self):
        """Verify GreenLang config directory can be created."""
        config_dir = Path.home() / ".greenlang"

        # Config dir should exist or be creatable
        if not config_dir.exists():
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
                assert config_dir.exists()
            except Exception as e:
                pytest.fail(f"Failed to create config directory: {e}")
        else:
            assert config_dir.is_dir()

    def test_yaml_parsing(self):
        """Verify YAML parsing works correctly."""
        import yaml

        test_yaml = """
name: test-pack
version: 1.0.0
kind: pack
description: Test pack for smoke testing
"""
        data = yaml.safe_load(test_yaml)
        assert data["name"] == "test-pack"
        assert data["version"] == "1.0.0"

    def test_pydantic_models(self):
        """Verify Pydantic models work correctly."""
        from greenlang.agents.base import AgentConfig

        config = AgentConfig(
            name="SmokeTestAgent",
            description="Test agent for smoke testing",
            version="0.1.0",
            enabled=True,
        )

        assert config.name == "SmokeTestAgent"
        assert config.enabled is True

        # Test serialization
        config_dict = config.model_dump()
        assert "name" in config_dict
        assert "description" in config_dict

    def test_json_schema_support(self):
        """Verify JSON schema validation support."""
        try:
            import jsonschema

            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["name"],
            }

            valid_data = {"name": "test", "value": 42}
            jsonschema.validate(valid_data, schema)

            # Should raise on invalid data
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"value": "not a number"}, schema)

        except ImportError:
            pytest.skip("jsonschema not available")


class TestDependencies:
    """Tests for required dependencies."""

    REQUIRED_PACKAGES = [
        "typer",
        "pydantic",
        "yaml",  # PyYAML
        "rich",
        "jsonschema",
        "httpx",
        "requests",
    ]

    @pytest.mark.parametrize("package", REQUIRED_PACKAGES)
    def test_required_package_installed(self, package: str):
        """Verify required packages are installed."""
        # Handle special cases for package names
        import_map = {
            "yaml": "yaml",  # PyYAML imports as yaml
        }
        import_name = import_map.get(package, package)

        try:
            __import__(import_name)
        except ImportError:
            pytest.fail(f"Required package not installed: {package}")

    def test_python_version(self):
        """Verify Python version meets requirements."""
        assert sys.version_info >= (3, 10), (
            f"Python 3.10+ required, got {sys.version_info.major}.{sys.version_info.minor}"
        )


class TestCLICommands:
    """Tests for specific CLI commands."""

    def test_pack_list_command(self):
        """Verify 'gl pack list' command works."""
        returncode, stdout, stderr = run_command(["gl", "pack", "list"])

        # Command should run without error (even if no packs installed)
        assert returncode == 0, (
            f"gl pack list failed: {stderr}\nstdout: {stdout}"
        )

    def test_pack_help(self):
        """Verify 'gl pack --help' shows available commands."""
        returncode, stdout, stderr = run_command(["gl", "pack", "--help"])

        assert returncode == 0, f"gl pack --help failed: {stderr}"

        output = stdout.lower()
        # Should show subcommands
        assert "list" in output or "install" in output or "validate" in output

    def test_init_help(self):
        """Verify 'gl init --help' shows available commands."""
        returncode, stdout, stderr = run_command(["gl", "init", "--help"])

        assert returncode == 0, f"gl init --help failed: {stderr}"


class TestIntegration:
    """Integration tests that verify end-to-end functionality."""

    def test_create_temp_pack_structure(self):
        """Verify a temporary pack structure can be created and loaded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pack_dir = Path(temp_dir) / "test-smoke-pack"
            pack_dir.mkdir()

            # Create pack manifest
            pack_yaml = pack_dir / "pack.yaml"
            pack_yaml.write_text("""
name: smoke-test-pack
version: 1.0.0
kind: pack
license: MIT
description: "Smoke test pack"
contents:
  agents: []
  pipelines: []
  datasets: []
""")

            # Verify file was created
            assert pack_yaml.exists()

            # Verify YAML is valid
            import yaml
            with open(pack_yaml) as f:
                manifest = yaml.safe_load(f)

            assert manifest["name"] == "smoke-test-pack"
            assert manifest["version"] == "1.0.0"

    def test_agent_lifecycle(self):
        """Test complete agent lifecycle: create, run, cleanup."""
        from greenlang.agents.base import BaseAgent, AgentConfig, AgentResult
        from typing import Dict, Any

        class LifecycleTestAgent(BaseAgent):
            """Agent that tracks its lifecycle."""

            def __init__(self, config: AgentConfig):
                self.lifecycle_events = []
                super().__init__(config)

            def initialize(self):
                self.lifecycle_events.append("initialize")

            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                self.lifecycle_events.append("validate")
                return True

            def preprocess(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
                self.lifecycle_events.append("preprocess")
                return input_data

            def execute(self, input_data: Dict[str, Any]) -> AgentResult:
                self.lifecycle_events.append("execute")
                return AgentResult(success=True, data={"processed": True})

            def postprocess(self, result: AgentResult) -> AgentResult:
                self.lifecycle_events.append("postprocess")
                return result

            def cleanup(self):
                self.lifecycle_events.append("cleanup")

        config = AgentConfig(
            name="LifecycleTestAgent",
            description="Tests agent lifecycle",
        )
        agent = LifecycleTestAgent(config)

        # Initialize is called in __init__
        assert "initialize" in agent.lifecycle_events

        # Run the agent
        result = agent.run({"test": "data"})

        # Verify lifecycle events
        assert result.success is True
        assert "validate" in agent.lifecycle_events
        assert "preprocess" in agent.lifecycle_events
        assert "execute" in agent.lifecycle_events
        assert "postprocess" in agent.lifecycle_events
        assert "cleanup" in agent.lifecycle_events


# =============================================================================
# Main Entry Point
# =============================================================================


class SmokeTestRunner:
    """Runner for executing smoke tests with detailed reporting."""

    def __init__(self):
        self.results: Dict[str, Any] = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "version": EXPECTED_VERSION,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
        }

    def run(self) -> int:
        """Run all smoke tests and return exit code."""
        print("=" * 70)
        print("GreenLang Release Smoke Test Suite")
        print("=" * 70)
        print(f"Expected Version: {EXPECTED_VERSION}")
        print(f"Python Version: {self.results['python_version']}")
        print(f"Platform: {self.results['platform']}")
        print(f"Strict Mode: {STRICT_MODE}")
        print("=" * 70)
        print()

        # Run pytest
        exit_code = pytest.main([
            __file__,
            "-v",
            "--tb=short",
            "-x" if STRICT_MODE else "",
            "--no-header",
        ])

        print()
        print("=" * 70)
        if exit_code == 0:
            print("SMOKE TEST RESULT: PASSED")
        else:
            print("SMOKE TEST RESULT: FAILED")
        print("=" * 70)

        return exit_code


if __name__ == "__main__":
    runner = SmokeTestRunner()
    sys.exit(runner.run())
