# -*- coding: utf-8 -*-
# GL-VCCI-Carbon-APP: Scope 3 Value Chain Carbon Intelligence Platform
# Copyright © 2025 GreenLang Framework Team
#
# Enterprise-grade Scope 3 emissions tracking platform supporting all 15 GHG
# Protocol categories with zero-hallucination guarantee, AI-powered intelligence,
# and automated supplier engagement.

"""
GL-VCCI Scope 3 Platform
========================

The world's most advanced Scope 3 emissions tracking platform.

Features:
---------
- Calculate Scope 3 emissions across all 15 GHG Protocol categories
- Engage thousands of suppliers with automated data collection
- Leverage hybrid AI (zero-hallucination + intelligent estimation)
- Ensure audit compliance with SHA-256 provenance chains
- Auto-generate reports for GHG Protocol, CDP, and SBTi

Quick Start:
-----------
```python
from vcci_scope3 import Scope3Pipeline

# Initialize pipeline
pipeline = Scope3Pipeline(config_path="config/vcci_config.yaml")

# Run complete Scope 3 analysis
results = pipeline.run(
    procurement_data="procurement.csv",
    logistics_data="logistics.csv",
    supplier_data="suppliers.json"
)

# Access results
print(f"Total Scope 3 emissions: {results.total_emissions:.2f} tCO2e")
print(f"Data coverage (Tier 1/2): {results.data_coverage:.1%}")
```

CLI Usage:
---------
```bash
# Ingest procurement data
vcci intake --file procurement.csv --format csv

# Calculate Scope 3 emissions
vcci calculate --data validated_data.json --categories all

# Generate GHG Protocol report
vcci report --emissions scope3_results.json --format ghg-protocol
```

Modules:
--------
- agents: 5 core agents (Intake, Calculator, Hotspot, Engagement, Reporting)
- sdk: Python SDK for programmatic access
- cli: Command-line interface
- provenance: SHA-256 provenance chain tracking
- data: Emission factor databases
- connectors: ERP system integrations (SAP, Oracle, Workday)
- utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "GreenLang Framework Team"
__email__ = "vcci@greenlang.io"
__license__ = "Proprietary"

# Package-level imports for convenience
from typing import Optional

# Version info
VERSION = __version__
VERSION_INFO = tuple(int(x) for x in __version__.split("."))

# Package metadata
PACKAGE_NAME = "vcci-scope3-platform"
PACKAGE_DESCRIPTION = "Enterprise Scope 3 Value Chain Carbon Intelligence Platform"

# Feature flags (can be overridden by environment variables)
DEFAULT_FEATURES = {
    "entity_resolution": True,
    "llm_categorization": True,
    "supplier_engagement": True,
    "automated_reporting": True,
    "scenario_modeling": True,
    "real_time_monitoring": True,
    # Beta features
    "blockchain_provenance": False,
    "satellite_monitoring": False,
}


def get_version() -> str:
    """Get the current version of the VCCI Scope 3 Platform.

    Returns:
        str: Version string (e.g., "1.0.0")
    """
    return __version__


def get_feature_status(feature_name: str) -> bool:
    """Check if a feature is enabled.

    Args:
        feature_name: Name of the feature to check

    Returns:
        bool: True if feature is enabled, False otherwise
    """
    import os

    # Check environment variable first
    env_var = f"FEATURE_{feature_name.upper()}"
    env_value = os.getenv(env_var)

    if env_value is not None:
        return env_value.lower() in ("true", "1", "yes", "on")

    # Fall back to default
    return DEFAULT_FEATURES.get(feature_name, False)


# Lazy imports to avoid circular dependencies
# These will be imported only when explicitly requested
__all__ = [
    "VERSION",
    "VERSION_INFO",
    "PACKAGE_NAME",
    "PACKAGE_DESCRIPTION",
    "get_version",
    "get_feature_status",
]
