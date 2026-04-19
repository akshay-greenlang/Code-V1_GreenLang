"""
Tests for configuration management
"""

import pytest
from pathlib import Path
import tempfile
import yaml

from cli.utils.config import (
    load_config,
    save_config,
    get_config_value,
    DEFAULT_CONFIG,
)


def test_load_default_config():
    """Test loading default configuration."""
    config = load_config(Path("nonexistent.yaml"))
    assert config == DEFAULT_CONFIG


def test_save_and_load_config():
    """Test saving and loading configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config" / "factory.yaml"
        test_config = {
            "version": "1.0",
            "defaults": {
                "output_dir": "test_agents",
            },
        }

        save_config(test_config, config_path)
        assert config_path.exists()

        loaded_config = load_config(config_path)
        assert "version" in loaded_config
        assert loaded_config["defaults"]["output_dir"] == "test_agents"


def test_get_config_value():
    """Test getting nested configuration values."""
    config = {
        "defaults": {
            "output_dir": "agents",
            "test_dir": "tests",
        },
        "registry": {
            "url": "https://registry.example.com",
        },
    }

    assert get_config_value("defaults.output_dir", config=config) == "agents"
    assert get_config_value("registry.url", config=config) == "https://registry.example.com"
    assert get_config_value("nonexistent.key", "default_value", config) == "default_value"


def test_get_config_value_default():
    """Test default value when key doesn't exist."""
    config = {"key": "value"}
    assert get_config_value("nonexistent", "default", config) == "default"
