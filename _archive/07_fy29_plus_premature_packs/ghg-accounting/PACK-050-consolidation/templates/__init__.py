# -*- coding: utf-8 -*-
"""
PACK-050 GHG Consolidation Pack - Report Templates
=============================================================

This package provides 10 report templates for the PACK-050 GHG Consolidation
Pack. Each template supports multiple rendering formats (Markdown, HTML, JSON,
CSV) with SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. ConsolidatedGhgReport          - Group-level consolidated GHG inventory
    2. EntityBreakdownReport          - Per-entity emission contributions
    3. OwnershipStructureReport       - Corporate structure with ownership chains
    4. EquityShareReport              - Equity share approach detail
    5. EliminationLogReport           - Intercompany elimination log
    6. MnaImpactReport                - M&A adjustment impacts
    7. ScopeBreakdownReport           - Consolidated scope 1/2/3 breakdown
    8. TrendAnalysisReport            - Year-over-year consolidated trends
    9. RegulatoryDisclosureReport     - Multi-framework regulatory disclosures
    10. ConsolidationDashboard        - Interactive consolidation dashboard data

Author: GreenLang Team
Version: 50.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

__version__ = "1.0.0"
__pack__ = "PACK-050"
__pack_name__ = "GHG Consolidation Pack"
__templates_count__ = 10

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .consolidated_ghg_report import ConsolidatedGhgReport
except ImportError as e:
    logger.warning("Failed to import ConsolidatedGhgReport: %s", e)
    ConsolidatedGhgReport = None  # type: ignore[assignment,misc]

try:
    from .entity_breakdown_report import EntityBreakdownReport
except ImportError as e:
    logger.warning("Failed to import EntityBreakdownReport: %s", e)
    EntityBreakdownReport = None  # type: ignore[assignment,misc]

try:
    from .ownership_structure_report import OwnershipStructureReport
except ImportError as e:
    logger.warning("Failed to import OwnershipStructureReport: %s", e)
    OwnershipStructureReport = None  # type: ignore[assignment,misc]

try:
    from .equity_share_report import EquityShareReport
except ImportError as e:
    logger.warning("Failed to import EquityShareReport: %s", e)
    EquityShareReport = None  # type: ignore[assignment,misc]

try:
    from .elimination_log_report import EliminationLogReport
except ImportError as e:
    logger.warning("Failed to import EliminationLogReport: %s", e)
    EliminationLogReport = None  # type: ignore[assignment,misc]

try:
    from .mna_impact_report import MnaImpactReport
except ImportError as e:
    logger.warning("Failed to import MnaImpactReport: %s", e)
    MnaImpactReport = None  # type: ignore[assignment,misc]

try:
    from .scope_breakdown_report import ScopeBreakdownReport
except ImportError as e:
    logger.warning("Failed to import ScopeBreakdownReport: %s", e)
    ScopeBreakdownReport = None  # type: ignore[assignment,misc]

try:
    from .trend_analysis_report import TrendAnalysisReport
except ImportError as e:
    logger.warning("Failed to import TrendAnalysisReport: %s", e)
    TrendAnalysisReport = None  # type: ignore[assignment,misc]

try:
    from .regulatory_disclosure_report import RegulatoryDisclosureReport
except ImportError as e:
    logger.warning("Failed to import RegulatoryDisclosureReport: %s", e)
    RegulatoryDisclosureReport = None  # type: ignore[assignment,misc]

try:
    from .consolidation_dashboard import ConsolidationDashboard
except ImportError as e:
    logger.warning("Failed to import ConsolidationDashboard: %s", e)
    ConsolidationDashboard = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    "ConsolidatedGhgReport",
    "EntityBreakdownReport",
    "OwnershipStructureReport",
    "EquityShareReport",
    "EliminationLogReport",
    "MnaImpactReport",
    "ScopeBreakdownReport",
    "TrendAnalysisReport",
    "RegulatoryDisclosureReport",
    "ConsolidationDashboard",
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "consolidated_ghg_report",
        "class": ConsolidatedGhgReport,
        "description": (
            "Group-level consolidated GHG inventory report with Scope 1, "
            "Scope 2 (location and market-based), Scope 3 totals after "
            "equity adjustments, eliminations, and boundary documentation."
        ),
        "category": "consolidation",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "entity_breakdown_report",
        "class": EntityBreakdownReport,
        "description": (
            "Per-entity emission contributions with raw vs equity-adjusted "
            "emissions, waterfall chart data, top-N entities, and scope "
            "breakdown per entity."
        ),
        "category": "entity",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "ownership_structure_report",
        "class": OwnershipStructureReport,
        "description": (
            "Corporate structure visualization with entity hierarchy tree, "
            "direct and effective ownership percentages, control types, JV "
            "partner details, and organisational boundary overlay."
        ),
        "category": "structure",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "equity_share_report",
        "class": EquityShareReport,
        "description": (
            "Equity share approach details with per-entity equity percentage, "
            "multi-tier ownership chain visualization, JV split details, "
            "associate inclusion, and partner reconciliation checks."
        ),
        "category": "equity",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "elimination_log_report",
        "class": EliminationLogReport,
        "description": (
            "Intercompany elimination details with transfer register, "
            "elimination entries, matching verification status, and net "
            "impact on consolidated total."
        ),
        "category": "elimination",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "mna_impact_report",
        "class": MnaImpactReport,
        "description": (
            "M&A adjustment impacts with event timeline, pro-rata calculations, "
            "before/after boundary comparison, base year restatement, and "
            "organic vs structural growth separation."
        ),
        "category": "mna",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "scope_breakdown_report",
        "class": ScopeBreakdownReport,
        "description": (
            "Consolidated scope 1/2/3 breakdown with Scope 1 by source "
            "category, Scope 2 dual reporting, Scope 3 by category (1-15), "
            "geographic breakdown, and sector breakdown."
        ),
        "category": "scope",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "trend_analysis_report",
        "class": TrendAnalysisReport,
        "description": (
            "Year-over-year consolidated trends with absolute emissions, "
            "intensity metrics, target tracking, base year comparison, "
            "like-for-like analysis, and decomposition analysis."
        ),
        "category": "trend",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "regulatory_disclosure_report",
        "class": RegulatoryDisclosureReport,
        "description": (
            "Multi-framework disclosure template mapped to CSRD/ESRS E1, "
            "CDP Climate Change, GRI 305, TCFD Metrics, SEC Climate Rule, "
            "SBTi progress, and IFRS S2 with cross-reference table."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
    {
        "name": "consolidation_dashboard",
        "class": ConsolidationDashboard,
        "description": (
            "Interactive consolidation dashboard with KPI cards, entity "
            "contribution chart, scope pie chart, geographic heat map, "
            "M&A event timeline, alert summary, and data quality indicators."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json", "csv"],
        "version": "50.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-050 GHG Consolidation report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 report templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_template_names()
        >>> template = registry.get("consolidated_ghg_report")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                self._templates[defn["name"]] = defn

        logger.info(
            "TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with metadata."""
        return [
            {
                "name": d["name"], "description": d["description"],
                "category": d["category"], "formats": d["formats"],
                "version": d["version"],
            }
            for d in TEMPLATE_CATALOG if d["class"] is not None
        ]

    def list_template_names(self) -> List[str]:
        """List all available template names."""
        return [d["name"] for d in TEMPLATE_CATALOG if d["class"] is not None]

    def get(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get a template instance by name."""
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(f"Template '{name}' not found. Available: {available}")

        if config is not None or name not in self._instances:
            template_class = self._templates[name]["class"]
            instance = template_class(config=config)
            if config is None:
                self._instances[name] = instance
            return instance

        return self._instances[name]

    def get_template(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Alias for get()."""
        return self.get(name, config)

    def render(
        self, template_name: str, data: Dict[str, Any],
        format: str = "markdown", config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Render a template in the specified format."""
        template = self.get(template_name, config)
        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        elif format == "json":
            return template.render_json(data)
        raise ValueError(f"Unsupported format '{format}'")

    def get_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a specific template."""
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")
        defn = self._templates[name]
        return {
            "name": defn["name"], "description": defn["description"],
            "category": defn["category"], "formats": defn["formats"],
            "version": defn["version"], "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get templates filtered by category."""
        return [
            {"name": d["name"], "description": d["description"],
             "category": d["category"], "formats": d["formats"], "version": d["version"]}
            for d in TEMPLATE_CATALOG
            if d["category"] == category and d["class"] is not None
        ]

    def has_template(self, name: str) -> bool:
        """Check if a template exists."""
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._templates)

    def __repr__(self) -> str:
        return f"TemplateRegistry(templates={self.template_count}, names={self.list_template_names()})"
