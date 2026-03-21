# -*- coding: utf-8 -*-
"""
PACK-027 Enterprise Net Zero Pack - Report Templates
=======================================================

This package provides 12 report templates for the PACK-027 Enterprise Net
Zero Pack, designed for large multi-entity organisations requiring
GHG Protocol Corporate Standard reporting, SBTi target validation, CDP
A-list optimisation, TCFD/ISSB S2 disclosure, SEC climate filing, CSRD
ESRS E1 compliance, and board-level governance dashboards. All templates
support multi-format output (Markdown, HTML, JSON, Excel) with SHA-256
provenance hashing and corporate green colour scheme (#0d3b2e, #1a6b4f,
#2e8b6e).

Templates:
    1.  GHGInventoryReportTemplate          - Full GHG Protocol Corporate Standard report
    2.  SBTiTargetSubmissionTemplate        - SBTi target submission package (28+14 criteria)
    3.  CDPClimateResponseTemplate          - CDP Climate Change questionnaire (C0-C15)
    4.  TCFDReportTemplate                  - Complete TCFD / ISSB S2 disclosure
    5.  ExecutiveDashboardTemplate          - Board-level climate KPI dashboard
    6.  SupplyChainHeatmapTemplate          - Tier 1/2/3 supplier emissions heatmap
    7.  ScenarioComparisonTemplate          - 1.5C vs 2C vs BAU scenario analysis
    8.  AssuranceStatementTemplate          - ISO 14064-3 / ISAE 3410 assurance statement
    9.  BoardClimateReportTemplate          - Quarterly board climate governance report
    10. SECClimateFilingTemplate            - SEC Climate Disclosure Rule filing
    11. CSRDESRSReportTemplate              - CSRD / ESRS E1 climate report
    12. MaterialityAssessmentTemplate       - Double materiality assessment (EFRAG IG-1)

Usage:
    >>> from packs.net_zero.PACK_027_enterprise_net_zero.templates import (
    ...     TemplateRegistry,
    ...     GHGInventoryReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("ghg_inventory_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("tcfd_report", data, format="html")

    >>> # Filter by category
    >>> governance = registry.get_by_category("governance")

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

__version__ = "27.0.0"
__pack_id__ = "PACK-027"
__pack_name__ = "Enterprise Net Zero Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .ghg_inventory_report import GHGInventoryReportTemplate
except ImportError:
    GHGInventoryReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GHGInventoryReportTemplate")

try:
    from .sbti_target_submission import SBTiTargetSubmissionTemplate
except ImportError:
    SBTiTargetSubmissionTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SBTiTargetSubmissionTemplate")

try:
    from .cdp_climate_response import CDPClimateResponseTemplate
except ImportError:
    CDPClimateResponseTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CDPClimateResponseTemplate")

try:
    from .tcfd_report import TCFDReportTemplate
except ImportError:
    TCFDReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TCFDReportTemplate")

try:
    from .executive_dashboard import ExecutiveDashboardTemplate
except ImportError:
    ExecutiveDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ExecutiveDashboardTemplate")

try:
    from .supply_chain_heatmap import SupplyChainHeatmapTemplate
except ImportError:
    SupplyChainHeatmapTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SupplyChainHeatmapTemplate")

try:
    from .scenario_comparison import ScenarioComparisonTemplate
except ImportError:
    ScenarioComparisonTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ScenarioComparisonTemplate")

try:
    from .assurance_statement import AssuranceStatementTemplate
except ImportError:
    AssuranceStatementTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AssuranceStatementTemplate")

try:
    from .board_climate_report import BoardClimateReportTemplate
except ImportError:
    BoardClimateReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import BoardClimateReportTemplate")

try:
    from .sec_climate_filing import SECClimateFilingTemplate
except ImportError:
    SECClimateFilingTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SECClimateFilingTemplate")

try:
    from .csrd_esrs_report import CSRDESRSReportTemplate
except ImportError:
    CSRDESRSReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CSRDESRSReportTemplate")

try:
    from .materiality_assessment import MaterialityAssessmentTemplate
except ImportError:
    MaterialityAssessmentTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import MaterialityAssessmentTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "GHGInventoryReportTemplate",
    "SBTiTargetSubmissionTemplate",
    "CDPClimateResponseTemplate",
    "TCFDReportTemplate",
    "ExecutiveDashboardTemplate",
    "SupplyChainHeatmapTemplate",
    "ScenarioComparisonTemplate",
    "AssuranceStatementTemplate",
    "BoardClimateReportTemplate",
    "SECClimateFilingTemplate",
    "CSRDESRSReportTemplate",
    "MaterialityAssessmentTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "ghg_inventory_report",
        "class": GHGInventoryReportTemplate,
        "description": (
            "Full GHG Protocol Corporate Standard inventory report with "
            "Scope 1 (8 sources), Scope 2 (dual reporting: location and "
            "market-based), Scope 3 (all 15 categories), multi-entity "
            "consolidation, data quality matrix, base year recalculation "
            "log, and year-over-year trend analysis."
        ),
        "category": "inventory",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "sbti_target_submission",
        "class": SBTiTargetSubmissionTemplate,
        "description": (
            "SBTi target submission package validating 28 near-term criteria "
            "(C1-C28) and 14 net-zero criteria (NZ-C1 to NZ-C14) per SBTi "
            "Corporate Manual V5.3. Includes coverage analysis, pathway "
            "visualisation (1.5C/WB2C), FLAG sector assessment, and "
            "readiness score with gap identification."
        ),
        "category": "target_setting",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "cdp_climate_response",
        "class": CDPClimateResponseTemplate,
        "description": (
            "CDP Climate Change questionnaire auto-populated response "
            "covering all modules C0-C15 including governance, risks and "
            "opportunities, business strategy, emissions methodology, "
            "Scope 1/2/3 data, verification, carbon pricing, and "
            "engagement. Optimised for A-list scoring."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "tcfd_report",
        "class": TCFDReportTemplate,
        "description": (
            "Complete TCFD disclosure covering all four pillars "
            "(Governance, Strategy, Risk Management, Metrics & Targets) "
            "with ISSB S2 cross-references, scenario analysis (1.5C/2C/"
            "3C+), financial impact quantification, and transition risk "
            "assessment by business unit."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "executive_dashboard",
        "class": ExecutiveDashboardTemplate,
        "description": (
            "Board-level climate executive dashboard with RAG status "
            "indicators, 15+ KPIs across emissions/energy/water/waste, "
            "initiative tracking with budget vs actual, regulatory "
            "compliance status, peer benchmarking, and internal carbon "
            "pricing impact summary."
        ),
        "category": "governance",
        "formats": ["markdown", "html", "json"],
        "version": "27.0.0",
    },
    {
        "name": "supply_chain_heatmap",
        "class": SupplyChainHeatmapTemplate,
        "description": (
            "Tier 1/2/3 supplier emissions heatmap with geographic and "
            "commodity breakdowns, top 50 supplier scorecards, CDP "
            "supply chain and SBTi commitment tracking, engagement "
            "programme status, and Scope 3 hotspot analysis by "
            "procurement category."
        ),
        "category": "supply_chain",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "scenario_comparison",
        "class": ScenarioComparisonTemplate,
        "description": (
            "1.5C vs 2C vs BAU scenario comparison with Monte Carlo "
            "simulation results (10,000 runs, P10/P25/P50/P75/P90), "
            "sensitivity analysis (Sobol indices), investment requirements "
            "per pathway, carbon budget analysis, stranded asset risk, "
            "and marginal abatement cost curves."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "assurance_statement",
        "class": AssuranceStatementTemplate,
        "description": (
            "ISO 14064-3:2019 / ISAE 3410 assurance statement supporting "
            "limited and reasonable assurance levels. Includes management "
            "assertion, scope of engagement, work performed procedures, "
            "findings summary, materiality threshold assessment, and "
            "conclusion with verifier signature block."
        ),
        "category": "assurance",
        "formats": ["markdown", "html", "json"],
        "version": "27.0.0",
    },
    {
        "name": "board_climate_report",
        "class": BoardClimateReportTemplate,
        "description": (
            "Quarterly board climate governance report (5-10 pages) with "
            "classification header, decisions required section, KPI "
            "summary with RAG status, initiative progress tracking, "
            "regulatory horizon scan, risk register update, and "
            "recommended board resolutions."
        ),
        "category": "governance",
        "formats": ["markdown", "html", "json"],
        "version": "27.0.0",
    },
    {
        "name": "sec_climate_filing",
        "class": SECClimateFilingTemplate,
        "description": (
            "SEC Climate Disclosure Rule filing covering Reg S-K Items "
            "1501-1505 (governance, strategy, risk management, targets, "
            "transition plans) and Reg S-X Article 14 (financial statement "
            "metrics). Includes LAF attestation requirements and safe "
            "harbor statement for forward-looking disclosures."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "csrd_esrs_report",
        "class": CSRDESRSReportTemplate,
        "description": (
            "CSRD / ESRS E1 Climate Change report covering all 9 "
            "disclosure requirements (E1-1 through E1-9) with delegated "
            "regulation article references, cross-cutting standard "
            "alignment (ESRS 1/2), double materiality linkage, and "
            "limited assurance readiness assessment."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
    {
        "name": "materiality_assessment",
        "class": MaterialityAssessmentTemplate,
        "description": (
            "Double materiality assessment per EFRAG IG-1 covering "
            "financial materiality (risk/opportunity quantification) and "
            "impact materiality (environmental/social impact scoring). "
            "Includes stakeholder analysis, quantitative thresholds, "
            "disclosure framework mapping, and materiality matrix "
            "visualisation."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json", "excel"],
        "version": "27.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-027 Enterprise Net Zero Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 12 enterprise net-zero report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/Excel.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 12
        >>> template = registry.get("ghg_inventory_report")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                self._templates[defn["name"]] = defn

        logger.info(
            "PACK-027 TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates with metadata.

        Returns:
            List of template info dicts with name, description,
            category, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["class"] is not None
        ]

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template name strings.
        """
        return [defn["name"] for defn in TEMPLATE_CATALOG if defn["class"] is not None]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'ghg_inventory_report').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html,
            render_json, and optionally render_excel methods.

        Raises:
            KeyError: If template name is not found.
        """
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. Available: {available}"
            )

        if config is not None or name not in self._instances:
            template_class = self._templates[name]["class"]
            instance = template_class(config=config)
            if config is None:
                self._instances[name] = instance
            return instance

        return self._instances[name]

    def get_template(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Alias for get(). Get a template instance by name.

        Args:
            name: Template name.
            config: Optional configuration overrides.

        Returns:
            Template instance.
        """
        return self.get(name, config)

    def render(
        self,
        template_name: str,
        data: Dict[str, Any],
        format: str = "markdown",
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Render a template in the specified format.

        Convenience method that gets the template and renders in one call.

        Args:
            template_name: Template name.
            data: Report data dict.
            format: Output format ('markdown', 'html', 'json', 'excel').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json/excel).

        Raises:
            KeyError: If template name not found.
            ValueError: If format is not supported.
        """
        template = self.get(template_name, config)
        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        elif format == "json":
            return template.render_json(data)
        elif format == "excel":
            if hasattr(template, "render_excel"):
                return template.render_excel(data)
            raise ValueError(
                f"Template '{template_name}' does not support Excel format."
            )
        else:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Use 'markdown', 'html', 'json', or 'excel'."
            )

    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a specific template.

        Args:
            name: Template name.

        Returns:
            Template info dict.

        Raises:
            KeyError: If template name is not found.
        """
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found")

        defn = self._templates[name]
        return {
            "name": defn["name"],
            "description": defn["description"],
            "category": defn["category"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string. Valid categories:
                'inventory', 'target_setting', 'disclosure',
                'governance', 'supply_chain', 'analysis',
                'assurance', 'regulatory'.

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["category"] == category and defn["class"] is not None
        ]

    def has_template(self, name: str) -> bool:
        """
        Check if a template exists by name.

        Args:
            name: Template name to check.

        Returns:
            True if template exists.
        """
        return name in self._templates

    @property
    def template_count(self) -> int:
        """
        Return the number of registered templates.

        Returns:
            Template count.
        """
        return len(self._templates)

    @property
    def categories(self) -> List[str]:
        """
        Return list of unique template categories.

        Returns:
            Sorted list of category strings.
        """
        cats = set()
        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                cats.add(defn["category"])
        return sorted(cats)

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(pack='PACK-027', templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
