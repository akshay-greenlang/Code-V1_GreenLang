# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Report Templates
========================================================

This package provides 8 CSDDD (Corporate Sustainability Due Diligence
Directive, Directive (EU) 2024/1760) readiness report templates. Each
template supports three rendering formats: Markdown, HTML (with inline
CSS), and JSON. All templates include SHA-256 provenance hashing for
audit trail integrity.

Templates:
    1. DDReadinessReportTemplate           - Overall readiness assessment
    2. ValueChainRiskMapTemplate           - Value chain risk mapping
    3. ImpactAssessmentReportTemplate      - Adverse impact identification
    4. PreventionMitigationReportTemplate  - Prevention/mitigation tracking
    5. GrievanceMechanismReportTemplate    - Grievance mechanism assessment
    6. StakeholderEngagementReportTemplate - Stakeholder engagement report
    7. ClimateTransitionReportTemplate     - Art 22 climate transition plan
    8. CSDDDScorecardTemplate             - Executive scorecard dashboard

Usage:
    >>> from packs.eu_compliance.PACK_019_csddd_readiness.templates import (
    ...     TemplateRegistry,
    ...     DDReadinessReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("dd_readiness_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 19.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.dd_readiness_report import (
        DDReadinessReportTemplate,
    )
except ImportError:
    DDReadinessReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import DDReadinessReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.value_chain_risk_map import (
        ValueChainRiskMapTemplate,
    )
except ImportError:
    ValueChainRiskMapTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ValueChainRiskMapTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.impact_assessment_report import (
        ImpactAssessmentReportTemplate,
    )
except ImportError:
    ImpactAssessmentReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ImpactAssessmentReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.prevention_mitigation_report import (
        PreventionMitigationReportTemplate,
    )
except ImportError:
    PreventionMitigationReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import PreventionMitigationReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.grievance_mechanism_report import (
        GrievanceMechanismReportTemplate,
    )
except ImportError:
    GrievanceMechanismReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GrievanceMechanismReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.stakeholder_engagement_report import (
        StakeholderEngagementReportTemplate,
    )
except ImportError:
    StakeholderEngagementReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import StakeholderEngagementReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.climate_transition_report import (
        ClimateTransitionReportTemplate,
    )
except ImportError:
    ClimateTransitionReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClimateTransitionReportTemplate")

try:
    from packs.eu_compliance.PACK_019_csddd_readiness.templates.csddd_scorecard import (
        CSDDDScorecardTemplate,
    )
except ImportError:
    CSDDDScorecardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CSDDDScorecardTemplate")


__all__ = [
    # Template classes
    "DDReadinessReportTemplate",
    "ValueChainRiskMapTemplate",
    "ImpactAssessmentReportTemplate",
    "PreventionMitigationReportTemplate",
    "GrievanceMechanismReportTemplate",
    "StakeholderEngagementReportTemplate",
    "ClimateTransitionReportTemplate",
    "CSDDDScorecardTemplate",
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
        "name": "dd_readiness_report",
        "class": DDReadinessReportTemplate,
        "description": (
            "Overall CSDDD readiness assessment with article-by-article "
            "compliance status (Art 5-29), gap analysis, weighted readiness "
            "scoring, prioritised recommendations, and phased implementation "
            "timeline."
        ),
        "category": "readiness",
        "directive_reference": "Directive (EU) 2024/1760, Art 5-29",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "value_chain_risk_map",
        "class": ValueChainRiskMapTemplate,
        "description": (
            "Value chain risk mapping with tier-by-tier assessment, country "
            "risk overlay, sector risk distribution, hotspot identification, "
            "and risk mitigation recommendations per Art 6-8."
        ),
        "category": "risk",
        "directive_reference": "Directive (EU) 2024/1760, Art 6-8",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "impact_assessment_report",
        "class": ImpactAssessmentReportTemplate,
        "description": (
            "Adverse impact identification with severity/likelihood risk "
            "matrix, human rights and environmental impact catalogues, "
            "Art 7 prioritisation scoring, and stakeholder impact analysis."
        ),
        "category": "impact",
        "directive_reference": "Directive (EU) 2024/1760, Art 6-7",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "prevention_mitigation_report",
        "class": PreventionMitigationReportTemplate,
        "description": (
            "Prevention measures and mitigation actions with KPI-based "
            "effectiveness tracking, budget analysis, resource allocation, "
            "and gap identification per Art 8-9."
        ),
        "category": "prevention",
        "directive_reference": "Directive (EU) 2024/1760, Art 8-9",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "grievance_mechanism_report",
        "class": GrievanceMechanismReportTemplate,
        "description": (
            "Grievance mechanism assessment with UNGP Principle 31 "
            "effectiveness criteria evaluation, channel accessibility "
            "scoring, case statistics, resolution analysis, and "
            "improvement recommendations per Art 11."
        ),
        "category": "grievance",
        "directive_reference": "Directive (EU) 2024/1760, Art 11",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "stakeholder_engagement_report",
        "class": StakeholderEngagementReportTemplate,
        "description": (
            "Stakeholder engagement report with group coverage analysis, "
            "AA1000 quality assessment, activity logging, consultation "
            "outcome tracking, and engagement improvement recommendations "
            "per Art 10."
        ),
        "category": "engagement",
        "directive_reference": "Directive (EU) 2024/1760, Art 10",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "climate_transition_report",
        "class": ClimateTransitionReportTemplate,
        "description": (
            "Climate transition plan assessment with GHG reduction targets, "
            "scope-level pathway analysis, Paris Agreement 1.5C/2C alignment "
            "scoring, implementation status tracking, and milestones per "
            "Art 22."
        ),
        "category": "climate",
        "directive_reference": "Directive (EU) 2024/1760, Art 22",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
    {
        "name": "csddd_scorecard",
        "class": CSDDDScorecardTemplate,
        "description": (
            "Executive dashboard with article-by-article status grid, "
            "key metrics (10 KPIs), risk summary, period-over-period "
            "trend analysis, action item tracking, and strategic "
            "recommendations."
        ),
        "category": "scorecard",
        "directive_reference": "Directive (EU) 2024/1760",
        "formats": ["markdown", "html", "json"],
        "version": "19.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-019 CSDDD Readiness Pack report templates.

    Provides centralised discovery, instantiation, and management of all
    8 CSDDD readiness report templates. Templates can be listed, filtered
    by category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("dd_readiness_report")
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
            category, directive_reference, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "directive_reference": defn["directive_reference"],
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
            name: Template name (e.g., 'dd_readiness_report').
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
            "directive_reference": defn["directive_reference"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string (e.g., 'readiness', 'risk',
                      'impact', 'prevention', 'grievance', 'engagement',
                      'climate', 'scorecard').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "directive_reference": defn["directive_reference"],
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
