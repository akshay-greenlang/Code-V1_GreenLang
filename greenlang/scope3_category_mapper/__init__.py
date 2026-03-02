# -*- coding: utf-8 -*-
"""
AGENT-MRV-029: Scope 3 Category Mapper (Cross-Cutting MRV Agent)

Deterministically classifies organizational data (spend records, purchase
orders, bills of materials, activity data, supplier records) into the
correct GHG Protocol Scope 3 category (Category 1--15) and routes them
to the appropriate category-specific agent (AGENT-MRV-014 through
AGENT-MRV-028).

This is the first cross-cutting MRV agent and serves as the intelligent
gateway between raw organizational data and the 15 Scope 3 calculation
agents.

Agent ID: GL-MRV-X-040
Package: greenlang.scope3_category_mapper
API: /api/v1/scope3-category-mapper
DB Migration: V080
Metrics Prefix: gl_scm_
Table Prefix: gl_scm_

Engines:
    1. CategoryDatabaseEngine - NAICS/ISIC/UNSPSC/HS code mappings
    2. SpendClassifierEngine - Deterministic spend classification
    3. ActivityRouterEngine - Category agent routing
    4. BoundaryDeterminerEngine - Upstream/downstream boundaries
    5. CompletenessScreenerEngine - Category completeness analysis
    6. ComplianceCheckerEngine - Multi-framework compliance validation
    7. CategoryMapperPipelineEngine - End-to-end 10-stage pipeline

Classification Sources:
    - NAICS 2022 (1,057 codes)
    - ISIC Rev 4 (419 codes)
    - UNSPSC v26 (55,000+ codes)
    - HS 2022 (5,600+ codes)
    - GL Account ranges (200+ mappings)
    - Spend keywords (500+ terms)

Compliance Frameworks:
    - GHG Protocol Scope 3 Standard
    - ISO 14064-1:2018
    - CSRD / ESRS E1
    - CDP Climate Change Questionnaire
    - SBTi FLAG / SBTi-FI
    - California SB 253
    - SEC Climate Rule
    - ISSB S2 (IFRS)

Double-Counting Rules:
    - DC-SCM-001 through DC-SCM-010

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

__all__ = [
    # Engine classes
    "CategoryDatabaseEngine",
    "SpendClassifierEngine",
    "ActivityRouterEngine",
    "BoundaryDeterminerEngine",
    "CompletenessScreenerEngine",
    "ComplianceCheckerEngine",
    "CategoryMapperPipelineEngine",
    # Metadata constants
    "AGENT_ID",
    "AGENT_COMPONENT",
    "VERSION",
    "TABLE_PREFIX",
    # Configuration helper
    "get_config",
    # Info helpers
    "get_version",
    "get_agent_info",
]

AGENT_ID: str = "GL-MRV-X-040"
AGENT_COMPONENT: str = "AGENT-MRV-029"
VERSION: str = "1.0.0"
TABLE_PREFIX: str = "gl_scm_"

# ---------------------------------------------------------------------------
# Graceful imports -- each engine with try/except so the package can be
# imported even when optional engine dependencies are not yet installed.
# ---------------------------------------------------------------------------

try:
    from greenlang.scope3_category_mapper.category_database import CategoryDatabaseEngine
except Exception:
    CategoryDatabaseEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.spend_classifier import SpendClassifierEngine
except Exception:
    SpendClassifierEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.activity_router import ActivityRouterEngine
except Exception:
    ActivityRouterEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.boundary_determiner import BoundaryDeterminerEngine
except Exception:
    BoundaryDeterminerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.completeness_screener import CompletenessScreenerEngine
except Exception:
    CompletenessScreenerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.compliance_checker import ComplianceCheckerEngine
except Exception:
    ComplianceCheckerEngine = None  # type: ignore[assignment,misc]

try:
    from greenlang.scope3_category_mapper.category_mapper_pipeline import CategoryMapperPipelineEngine
except Exception:
    CategoryMapperPipelineEngine = None  # type: ignore[assignment,misc]

# Export configuration helper
try:
    from greenlang.scope3_category_mapper.config import get_config
except Exception:
    def get_config():  # type: ignore[misc]
        """Fallback get_config if config module is not available."""
        return None


def get_version() -> str:
    """Return the current version string for AGENT-MRV-029.

    Returns:
        Semantic version string (e.g., ``'1.0.0'``).

    Example:
        >>> get_version()
        '1.0.0'
    """
    return VERSION


def get_agent_info() -> dict:
    """Return metadata dictionary describing this agent.

    Returns:
        Dictionary with keys ``agent_id``, ``component``, ``version``,
        ``table_prefix``, ``package``, ``scope``, ``role``,
        ``engines``, ``classification_sources``, ``categories_covered``,
        and ``double_counting_rules``.

    Example:
        >>> info = get_agent_info()
        >>> info["agent_id"]
        'GL-MRV-X-040'
    """
    return {
        "agent_id": AGENT_ID,
        "component": AGENT_COMPONENT,
        "version": VERSION,
        "table_prefix": TABLE_PREFIX,
        "package": "greenlang.scope3_category_mapper",
        "scope": "Scope 3",
        "role": "Cross-Cutting -- Category Mapping & Routing",
        "engines": [
            "CategoryDatabaseEngine",
            "SpendClassifierEngine",
            "ActivityRouterEngine",
            "BoundaryDeterminerEngine",
            "CompletenessScreenerEngine",
            "ComplianceCheckerEngine",
            "CategoryMapperPipelineEngine",
        ],
        "classification_sources": [
            "NAICS 2022 (1,057 codes)",
            "ISIC Rev 4 (419 codes)",
            "UNSPSC v26 (55,000+ codes)",
            "HS 2022 (5,600+ codes)",
            "GL Account ranges (200+)",
            "Spend keywords (500+)",
        ],
        "categories_covered": list(range(1, 16)),
        "double_counting_rules": [
            f"DC-SCM-{i:03d}" for i in range(1, 11)
        ],
    }
