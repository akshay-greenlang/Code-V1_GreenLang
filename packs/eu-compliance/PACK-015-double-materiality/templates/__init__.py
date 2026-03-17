# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Pack - Report Templates
========================================================

This package provides 8 double materiality report templates for the
PACK-015 DMA Pack. Each template supports three rendering formats:
Markdown, HTML (with inline CSS), and JSON. All templates include
SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. ImpactMaterialityReportTemplate      - Inside-out impact assessment
    2. FinancialMaterialityReportTemplate   - Outside-in financial assessment
    3. StakeholderEngagementReportTemplate  - Stakeholder consultation & validation
    4. MaterialityMatrixReportTemplate      - 2x2 matrix with quadrant analysis
    5. IRORegisterReportTemplate            - IRO classification & prioritization
    6. ESRSDisclosureMapTemplate            - ESRS DR mapping with gap analysis
    7. DMAExecutiveSummaryTemplate          - Board-level DMA summary
    8. DMAAuditReportTemplate               - Audit-ready DMA documentation

Usage:
    >>> from packs.eu_compliance.PACK_015_double_materiality.templates import (
    ...     TemplateRegistry,
    ...     ImpactMaterialityReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("impact_materiality_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 15.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .impact_materiality_report import ImpactMaterialityReportTemplate
from .financial_materiality_report import FinancialMaterialityReportTemplate
from .stakeholder_engagement_report import StakeholderEngagementReportTemplate
from .materiality_matrix_report import MaterialityMatrixReportTemplate
from .iro_register_report import IRORegisterReportTemplate
from .esrs_disclosure_map import ESRSDisclosureMapTemplate
from .dma_executive_summary import DMAExecutiveSummaryTemplate
from .dma_audit_report import DMAAuditReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "ImpactMaterialityReportTemplate",
    "FinancialMaterialityReportTemplate",
    "StakeholderEngagementReportTemplate",
    "MaterialityMatrixReportTemplate",
    "IRORegisterReportTemplate",
    "ESRSDisclosureMapTemplate",
    "DMAExecutiveSummaryTemplate",
    "DMAAuditReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[ImpactMaterialityReportTemplate],
    Type[FinancialMaterialityReportTemplate],
    Type[StakeholderEngagementReportTemplate],
    Type[MaterialityMatrixReportTemplate],
    Type[IRORegisterReportTemplate],
    Type[ESRSDisclosureMapTemplate],
    Type[DMAExecutiveSummaryTemplate],
    Type[DMAAuditReportTemplate],
]

TemplateInstance = Union[
    ImpactMaterialityReportTemplate,
    FinancialMaterialityReportTemplate,
    StakeholderEngagementReportTemplate,
    MaterialityMatrixReportTemplate,
    IRORegisterReportTemplate,
    ESRSDisclosureMapTemplate,
    DMAExecutiveSummaryTemplate,
    DMAAuditReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "impact_materiality_report",
        "class": ImpactMaterialityReportTemplate,
        "description": (
            "Impact materiality assessment report with severity score "
            "breakdowns (scale/scope/irremediability/likelihood), ESRS "
            "topic distribution, material impact rankings, and methodology "
            "documentation per ESRS 1 Chapter 3."
        ),
        "category": "assessment",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "financial_materiality_report",
        "class": FinancialMaterialityReportTemplate,
        "description": (
            "Financial materiality assessment report with risk/opportunity "
            "analysis, KPI impact mapping (revenue, COGS, CAPEX, etc.), "
            "magnitude/likelihood/time-horizon scoring, and financial "
            "ranking per ESRS 1 Chapter 3."
        ),
        "category": "assessment",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "stakeholder_engagement_report",
        "class": StakeholderEngagementReportTemplate,
        "description": (
            "Stakeholder engagement consultation report with category "
            "coverage analysis, influence-impact priority matrix, "
            "consultation summaries, synthesized findings, and ESRS 1 "
            "sections 22-23 validation results."
        ),
        "category": "stakeholder",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "materiality_matrix_report",
        "class": MaterialityMatrixReportTemplate,
        "description": (
            "Double materiality matrix report with 2x2 quadrant analysis "
            "(double material, impact-only, financial-only, not material), "
            "visualization position data, year-over-year comparison, and "
            "sector-specific threshold application results."
        ),
        "category": "matrix",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "iro_register_report",
        "class": IRORegisterReportTemplate,
        "description": (
            "IRO (Impacts, Risks, Opportunities) register report with "
            "classification summary, detailed register table, ESRS topic "
            "distribution, value chain coverage, and composite-score "
            "prioritization."
        ),
        "category": "register",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "esrs_disclosure_map",
        "class": ESRSDisclosureMapTemplate,
        "description": (
            "ESRS disclosure mapping report with per-topic coverage "
            "summary, disclosure requirement tables (E1-G1), gap analysis "
            "with missing datapoints, effort estimates by level, and "
            "implementation timeline."
        ),
        "category": "disclosure",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "dma_executive_summary",
        "class": DMAExecutiveSummaryTemplate,
        "description": (
            "Board/management-level executive summary of the Double "
            "Materiality Assessment with key metrics dashboard, material "
            "topic highlights, matrix quadrant summary, stakeholder "
            "engagement highlights, disclosure readiness, and recommendations."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
    {
        "name": "dma_audit_report",
        "class": DMAAuditReportTemplate,
        "description": (
            "Audit-ready DMA documentation with regulatory methodology "
            "references (ESRS 1, EFRAG IG-1, CSRD), stakeholder process "
            "audit trail, scoring evidence with provenance hashes, threshold "
            "justification, validation results, and sign-off section."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "15.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-015 Double Materiality report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 DMA report templates. Templates can be listed, filtered by
    category, retrieved by name, and rendered in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("impact_materiality_report")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, TemplateInstance] = {}

        for defn in TEMPLATE_CATALOG:
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
        ]

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List of template name strings.
        """
        return [defn["name"] for defn in TEMPLATE_CATALOG]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TemplateInstance:
        """
        Get a template instance by name.

        Creates a new instance or returns a cached one. If config is
        provided, always creates a new instance.

        Args:
            name: Template name (e.g., 'impact_materiality_report').
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
    ) -> TemplateInstance:
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
            category: Category string (e.g., 'assessment', 'stakeholder',
                      'matrix', 'register', 'disclosure', 'executive', 'audit').

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
            if defn["category"] == category
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
