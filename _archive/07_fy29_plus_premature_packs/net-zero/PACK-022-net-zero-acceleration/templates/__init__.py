# -*- coding: utf-8 -*-
"""
PACK-022 Net Zero Acceleration Pack - Report Templates
=======================================================

This package provides 10 report templates for the PACK-022 Net Zero Acceleration
Pack, covering advanced net-zero strategy acceleration including multi-scenario
comparison, sectoral decarbonization pathways, supplier engagement, transition
finance, variance analysis, temperature alignment, VCMI claims, multi-entity
consolidation, assurance packaging, and executive acceleration strategy.

Each template supports three rendering formats: Markdown, HTML (with inline CSS),
and JSON. All templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. ScenarioComparisonReportTemplate      - Multi-scenario comparison with tornado charts
    2. SDAPathwayReportTemplate              - Sector-specific decarbonization pathway
    3. SupplierEngagementReportTemplate      - Supplier engagement program dashboard
    4. TransitionFinanceReportTemplate       - Climate CapEx/OpEx and Taxonomy alignment
    5. VarianceAnalysisReportTemplate        - Emissions decomposition and variance attribution
    6. TemperatureAlignmentReportTemplate    - Portfolio temperature scoring
    7. VCMIClaimsReportTemplate              - VCMI Claims Code validation
    8. MultiEntityReportTemplate             - Multi-entity consolidated emissions
    9. AssurancePackageReportTemplate        - Audit workpaper package
   10. AccelerationStrategyReportTemplate    - Executive acceleration strategy

Usage:
    >>> from packs.net_zero.PACK_022_net_zero_acceleration.templates import (
    ...     TemplateRegistry,
    ...     ScenarioComparisonReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("scenario_comparison_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

    >>> # Render via registry shortcut
    >>> result = registry.render("sda_pathway_report", data, format="html")

    >>> # Filter by category
    >>> finance_templates = registry.get_by_category("finance")

Author: GreenLang Team
Version: 22.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .scenario_comparison_report import (
        ScenarioComparisonReportTemplate,
    )
except ImportError:
    ScenarioComparisonReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ScenarioComparisonReportTemplate")

try:
    from .sda_pathway_report import (
        SDAPathwayReportTemplate,
    )
except ImportError:
    SDAPathwayReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SDAPathwayReportTemplate")

try:
    from .supplier_engagement_report import (
        SupplierEngagementReportTemplate,
    )
except ImportError:
    SupplierEngagementReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SupplierEngagementReportTemplate")

try:
    from .transition_finance_report import (
        TransitionFinanceReportTemplate,
    )
except ImportError:
    TransitionFinanceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TransitionFinanceReportTemplate")

try:
    from .variance_analysis_report import (
        VarianceAnalysisReportTemplate,
    )
except ImportError:
    VarianceAnalysisReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import VarianceAnalysisReportTemplate")

try:
    from .temperature_alignment_report import (
        TemperatureAlignmentReportTemplate,
    )
except ImportError:
    TemperatureAlignmentReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import TemperatureAlignmentReportTemplate")

try:
    from .vcmi_claims_report import (
        VCMIClaimsReportTemplate,
    )
except ImportError:
    VCMIClaimsReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import VCMIClaimsReportTemplate")

try:
    from .multi_entity_report import (
        MultiEntityReportTemplate,
    )
except ImportError:
    MultiEntityReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import MultiEntityReportTemplate")

try:
    from .assurance_package_report import (
        AssurancePackageReportTemplate,
    )
except ImportError:
    AssurancePackageReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AssurancePackageReportTemplate")

try:
    from .acceleration_strategy_report import (
        AccelerationStrategyReportTemplate,
    )
except ImportError:
    AccelerationStrategyReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AccelerationStrategyReportTemplate")


__all__ = [
    # Template classes
    "ScenarioComparisonReportTemplate",
    "SDAPathwayReportTemplate",
    "SupplierEngagementReportTemplate",
    "TransitionFinanceReportTemplate",
    "VarianceAnalysisReportTemplate",
    "TemperatureAlignmentReportTemplate",
    "VCMIClaimsReportTemplate",
    "MultiEntityReportTemplate",
    "AssurancePackageReportTemplate",
    "AccelerationStrategyReportTemplate",
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
        "name": "scenario_comparison_report",
        "class": ScenarioComparisonReportTemplate,
        "description": (
            "Multi-scenario comparison report with tornado chart data, Monte Carlo "
            "simulation results (P10/P25/median/P75/P90), sensitivity analysis with "
            "tornado ranking, decision matrix scoring cost vs risk vs ambition, "
            "recommended pathway with milestones, and key assumptions documentation."
        ),
        "category": "scenario",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "sda_pathway_report",
        "class": SDAPathwayReportTemplate,
        "description": (
            "Sector-specific decarbonization pathway report with SDA convergence "
            "curves, base year intensity profile, year-by-year convergence targets, "
            "activity growth projections, absolute emissions trajectory, ACA vs SDA "
            "comparison, IEA NZE benchmark alignment, and SBTi validation status."
        ),
        "category": "sda",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "supplier_engagement_report",
        "class": SupplierEngagementReportTemplate,
        "description": (
            "Supplier engagement program dashboard with 4-tier supplier segmentation, "
            "engagement level distribution, top 20 suppliers by emissions, RAG progress "
            "dashboard, Scope 3 impact estimation by category, engagement milestones "
            "timeline, resource allocation (team and budget), and recommendations."
        ),
        "category": "supplier",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "transition_finance_report",
        "class": TransitionFinanceReportTemplate,
        "description": (
            "Climate CapEx/OpEx and EU Taxonomy alignment report with climate "
            "investment classification, category breakdown, taxonomy alignment per "
            "activity (SC/DNSH/Min Safeguards), green bond eligibility, internal "
            "carbon pricing scenarios, NPV/IRR investment cases, climate OpEx "
            "projection, cost of inaction analysis, and ROI summary."
        ),
        "category": "finance",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "variance_analysis_report",
        "class": VarianceAnalysisReportTemplate,
        "description": (
            "Emissions decomposition and variance attribution report using Kaya/LMDI "
            "methodology with activity/intensity/structural effects, top 10 driver "
            "attribution with magnitude ranking, year-over-year waterfall data, "
            "cumulative effects since base year, rolling 1-3yr forecasts, RAG alert "
            "status with thresholds, and corrective actions required."
        ),
        "category": "variance",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "temperature_alignment_report",
        "class": TemperatureAlignmentReportTemplate,
        "description": (
            "Portfolio temperature scoring report with implied temperature rise "
            "across WATS/TETS/MOTS/EOTS aggregation methods, entity-level scores, "
            "target classification matrix, contribution analysis, temperature band "
            "distribution, what-if improvement scenarios, and CDP-WWF methodology notes."
        ),
        "category": "temperature",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "vcmi_claims_report",
        "class": VCMIClaimsReportTemplate,
        "description": (
            "VCMI Claims Code validation and certification report with 4 foundational "
            "criteria checklist (pass/fail), evidence assessment per criterion, "
            "Silver/Gold/Platinum tier requirements, ICVCM credit quality analysis, "
            "greenwashing risk flags, gap to next tier, ISO 14068-1 comparison, "
            "recommendations, and annual re-validation schedule."
        ),
        "category": "vcmi",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "multi_entity_report",
        "class": MultiEntityReportTemplate,
        "description": (
            "Multi-entity consolidated emissions report with group summary, "
            "consolidation method documentation, entity hierarchy with ownership, "
            "entity-level emissions by scope, intercompany eliminations, "
            "completeness matrix, scope split per entity, target allocation vs "
            "performance tracking, structural changes (M&A/divestiture), and "
            "base year recalculation notes with significance thresholds."
        ),
        "category": "consolidation",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "assurance_package_report",
        "class": AssurancePackageReportTemplate,
        "description": (
            "Audit workpaper package for external assurance with engagement summary, "
            "scope and boundary definition, standards applied, materiality assessment, "
            "methodology documentation per scope, sample calculation traces, data "
            "lineage map with hashes, control evidence summary, exception register, "
            "completeness matrix, change register, cross-check results, full "
            "provenance chain, and auditor notes section."
        ),
        "category": "assurance",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
    {
        "name": "acceleration_strategy_report",
        "class": AccelerationStrategyReportTemplate,
        "description": (
            "Executive-level net-zero acceleration strategy consolidating scenario "
            "analysis, SDA pathway status, supplier engagement program, climate "
            "investment plan, progress against targets with RAG, temperature alignment "
            "score, VCMI claims status, multi-entity view, key risks and mitigations, "
            "12-month implementation roadmap, and board recommendations with "
            "investment requirements."
        ),
        "category": "strategy",
        "formats": ["markdown", "html", "json"],
        "version": "22.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-022 Net Zero Acceleration Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 net-zero acceleration report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("scenario_comparison_report")
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
            "TemplateRegistry initialized with %d templates",
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
            name: Template name (e.g., 'scenario_comparison_report').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html, render_json.

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
            format: Output format ('markdown', 'html', 'json').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json).

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
        else:
            raise ValueError(
                f"Unsupported format '{format}'. Use 'markdown', 'html', or 'json'."
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
                'scenario', 'sda', 'supplier', 'finance', 'variance',
                'temperature', 'vcmi', 'consolidation', 'assurance', 'strategy'.

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

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )
