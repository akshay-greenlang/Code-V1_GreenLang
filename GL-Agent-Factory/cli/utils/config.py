"""
CLI Configuration Management

Handles loading and managing CLI configuration.
"""

from typing import Optional, Dict, Any
from pathlib import Path
import yaml


DEFAULT_CONFIG = {
    "version": "1.0",
    "defaults": {
        "output_dir": "agents",
        "test_dir": "tests",
        "spec_dir": "specs",
    },
    "registry": {
        "url": "https://registry.greenlang.io",
        "timeout": 30,
    },
    "generator": {
        "enable_validation": True,
        "enable_tests": True,
        "enable_documentation": True,
        "template": "basic",
    },
    "validation": {
        "strict_mode": False,
        "allow_unknown_fields": True,
    },
    "testing": {
        "parallel": True,
        "verbose": False,
        "coverage": True,
    },
}


def get_config_path(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the configuration file by searching up the directory tree.

    Args:
        start_dir: Starting directory for search (defaults to cwd)

    Returns:
        Path to config file if found, None otherwise
    """
    if start_dir is None:
        start_dir = Path.cwd()

    current = start_dir.resolve()

    # Search up the directory tree
    while True:
        config_file = current / "config" / "factory.yaml"
        if config_file.exists():
            return config_file

        # Alternative location
        config_file = current / "factory.yaml"
        if config_file.exists():
            return config_file

        # Check if we've reached the root
        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from file or return defaults.

    Args:
        config_path: Path to config file (will search if not provided)

    Returns:
        Configuration dictionary
    """
    # If no path provided, try to find it
    if config_path is None:
        config_path = get_config_path()

    # If still no config file, return defaults
    if config_path is None or not config_path.exists():
        return DEFAULT_CONFIG.copy()

    # Load and merge with defaults
    try:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Merge with defaults
        config = DEFAULT_CONFIG.copy()
        config.update(user_config or {})

        return config
    except Exception as e:
        # If loading fails, return defaults
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any], config_path: Path):
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_config_value(key: str, default: Any = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        key: Configuration key (e.g., "generator.enable_validation")
        default: Default value if key not found
        config: Configuration dictionary (will load if not provided)

    Returns:
        Configuration value
    """
    if config is None:
        config = load_config()

    keys = key.split(".")
    value = config

    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
            if value is None:
                return default
        else:
            return default

    return value


def update_config_value(key: str, value: Any, config_path: Optional[Path] = None):
    """
    Update a configuration value and save to file.

    Args:
        key: Configuration key (e.g., "generator.enable_validation")
        value: New value
        config_path: Path to config file (will search if not provided)
    """
    if config_path is None:
        config_path = get_config_path()
        if config_path is None:
            raise ValueError("No configuration file found")

    config = load_config(config_path)

    # Navigate to the correct nested dict
    keys = key.split(".")
    current = config

    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]

    # Set the value
    current[keys[-1]] = value

    # Save
    save_config(config, config_path)
