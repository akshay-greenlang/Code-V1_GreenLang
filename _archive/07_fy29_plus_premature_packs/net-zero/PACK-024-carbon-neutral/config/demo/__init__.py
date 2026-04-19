"""
PACK-024 Carbon Neutral Pack - Demo Configuration Module

Provides demo/sample configuration files for testing and evaluation
of the Carbon Neutral Pack without requiring live data sources.

This module contains example YAML configurations with synthetic data
that demonstrate the full capabilities of the pack across multiple
neutrality types and scenarios.

Demo Configurations:
    - demo_config.yaml: Manufacturing company (CleanTech Manufacturing GmbH)
      with comprehensive GHG footprint, carbon management plan, credit
      portfolio, and ISO 14068-1 carbon neutrality claim
"""

from pathlib import Path

# Path to demo configuration file
DEMO_CONFIG_PATH = Path(__file__).parent / "demo_config.yaml"

def load_demo_config():
    """Load the demo configuration for evaluation purposes.

    Returns:
        str: Path to demo_config.yaml file

    Example:
        >>> from packs.net_zero.PACK_024_carbon_neutral.config.demo import load_demo_config
        >>> demo_path = load_demo_config()
        >>> from packs.net_zero.PACK_024_carbon_neutral.config import PackConfig
        >>> config = PackConfig.from_yaml(demo_path)
    """
    return str(DEMO_CONFIG_PATH)

__all__ = ["DEMO_CONFIG_PATH", "load_demo_config"]
