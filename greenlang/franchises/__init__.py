# -*- coding: utf-8 -*-
"""
Franchises Agent Package - AGENT-MRV-027

GHG Protocol Scope 3, Category 14: Franchises.
Calculates emissions from the operation of franchises not included in
Scope 1 and Scope 2 -- reported by the FRANCHISOR.

This agent covers franchise-level energy use (stationary combustion,
purchased electricity, purchased heating/cooling), refrigerant leakage,
mobile combustion from delivery fleets, and spend-based estimation when
primary data is unavailable.  It enforces double-counting prevention rules
to ensure company-owned units are excluded (DC-FRN-001).

Agent ID: GL-MRV-S3-014
Package: greenlang.franchises
API: /api/v1/franchises
DB Migration: V078
Metrics Prefix: gl_frn_
Table Prefix: gl_frn_

Calculation Methods:
    - Franchise-specific (primary energy/refrigerant data per unit)
    - Average-data (EUI benchmarks by type and climate zone)
    - Spend-based (EEIO factors applied to royalty/revenue data)
    - Hybrid (blend of methods across heterogeneous networks)

Author: GreenLang Platform Team
Date: February 2026
Status: Production Ready
"""

from typing import Any, Dict, Optional

__all__ = [
    # Engine classes
    "FranchiseDatabaseEngine",
    "FranchiseSpecificCalculatorEngine",
    "AverageDataCalculatorEngine",
    "SpendBasedCalculatorEngine",
    "HybridAggregatorEngine",
    "ComplianceCheckerEngine",
    "FranchisePipelineEngine",
    "FranchisesPipelineEngine",
    # Metadata
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Helpers
    "get_config",
    "get_version",
    "get_agent_info",
]

__version__ = "1.0.0"
__agent_id__ = "GL-MRV-S3-014"

VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Graceful imports -- each engine with try/except so the package can always
# be imported even when individual engine modules have unmet dependencies.
# ---------------------------------------------------------------------------

try:
    from greenlang.franchises.franchise_database import FranchiseDatabaseEngine
except ImportError:
    FranchiseDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.franchise_specific_calculator import FranchiseSpecificCalculatorEngine
except ImportError:
    FranchiseSpecificCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.average_data_calculator import AverageDataCalculatorEngine
except ImportError:
    AverageDataCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.spend_based_calculator import SpendBasedCalculatorEngine
except ImportError:
    SpendBasedCalculatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.hybrid_aggregator import HybridAggregatorEngine
except ImportError:
    HybridAggregatorEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.compliance_checker import ComplianceCheckerEngine
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.franchise_pipeline import FranchisePipelineEngine
except ImportError:
    FranchisePipelineEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.franchises.franchises_pipeline import FranchisesPipelineEngine
except ImportError:
    FranchisesPipelineEngine = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Agent metadata -- imported from models when available, with safe fallback.
# ---------------------------------------------------------------------------

try:
    from greenlang.franchises.models import (
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION as MODELS_VERSION,
        TABLE_PREFIX,
    )
except ImportError:
    AGENT_ID: str = "GL-MRV-S3-014"  # type: ignore[no-redef]
    AGENT_COMPONENT: str = "AGENT-MRV-027"  # type: ignore[no-redef]
    TABLE_PREFIX: str = "gl_frn_"  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

try:
    from greenlang.franchises.config import get_config
except ImportError:
    def get_config() -> Optional[Dict[str, Any]]:  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def get_version() -> str:
    """Return the semantic version string for this agent package."""
    return __version__


def get_agent_info() -> Dict[str, str]:
    """
    Return a summary dict of agent identity metadata.

    Returns:
        Dict with keys: agent_id, component, version, table_prefix, package.
    """
    return {
        "agent_id": __agent_id__,
        "component": "AGENT-MRV-027",
        "version": __version__,
        "table_prefix": "gl_frn_",
        "package": "greenlang.franchises",
        "scope": "Scope 3 Category 14",
        "description": "Franchises emissions (franchisor reporting)",
    }
