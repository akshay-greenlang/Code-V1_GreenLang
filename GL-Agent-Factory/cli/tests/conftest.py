"""
Pytest configuration and fixtures
"""

import pytest
from pathlib import Path
import tempfile
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification."""
    return {
        "metadata": {
            "id": "test-agent",
            "name": "Test Agent",
            "version": "0.1.0",
            "type": "test",
            "description": "A test agent",
        },
        "capabilities": [
            "testing",
        ],
        "architecture": {
            "framework": "greenlang",
        },
    }


@pytest.fixture
def sample_spec_file(temp_dir, sample_agent_spec):
    """Create a sample specification file."""
    spec_file = temp_dir / "test-agent.yaml"
    spec_file.write_text(yaml.dump(sample_agent_spec))
    return spec_file


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return {
        "version": "1.0",
        "defaults": {
            "output_dir": "agents",
            "test_dir": "tests",
        },
        "registry": {
            "url": "https://registry.greenlang.io",
        },
        "generator": {
            "enable_validation": True,
            "enable_tests": True,
        },
    }
