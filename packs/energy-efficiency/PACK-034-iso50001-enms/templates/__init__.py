# -*- coding: utf-8 -*-
"""
PACK-034 ISO 50001 Energy Management System Pack - Report Templates
====================================================================

This package provides 10 ISO 50001 EnMS report templates for the
PACK-034 ISO 50001 Energy Management System Pack. Each template supports
three rendering formats: Markdown, HTML (with inline CSS), and JSON.
All templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. EnergyPolicyTemplate              - ISO 50001 Clause 5.2 Energy Policy
    2. EnergyReviewReportTemplate        - Clause 6.3 Energy Review
    3. EnPIMethodologyTemplate           - ISO 50006 EnPI/EnB Methodology
    4. ActionPlanTemplate                - Clause 6.2 Objectives & Action Plans
    5. OperationalControlTemplate        - Clause 8.1 Operational Controls
    6. PerformanceReportTemplate         - Clause 9.1 M&V Report
    7. InternalAuditTemplate             - Clause 9.2 Internal Audit
    8. ManagementReviewTemplate          - Clause 9.3 Management Review
    9. CorrectiveActionTemplate          - Clause 10.2 NC/CA Register
    10. EnMSDocumentationTemplate        - Full EnMS Documentation Package

Usage:
    >>> from packs.energy_efficiency.PACK_034_iso50001_enms.templates import (
    ...     TemplateRegistry,
    ...     EnergyPolicyTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("energy_policy")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 34.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .energy_policy_template import EnergyPolicyTemplate
from .energy_review_report_template import EnergyReviewReportTemplate
from .enpi_methodology_template import EnPIMethodologyTemplate
from .action_plan_template import ActionPlanTemplate
from .operational_control_template import OperationalControlTemplate
from .performance_report_template import PerformanceReportTemplate
from .internal_audit_template import InternalAuditTemplate
from .management_review_template import ManagementReviewTemplate
from .corrective_action_template import CorrectiveActionTemplate
from .enms_documentation_template import EnMSDocumentationTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "EnergyPolicyTemplate",
    "EnergyReviewReportTemplate",
    "EnPIMethodologyTemplate",
    "ActionPlanTemplate",
    "OperationalControlTemplate",
    "PerformanceReportTemplate",
    "InternalAuditTemplate",
    "ManagementReviewTemplate",
    "CorrectiveActionTemplate",
    "EnMSDocumentationTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[EnergyPolicyTemplate],
    Type[EnergyReviewReportTemplate],
    Type[EnPIMethodologyTemplate],
    Type[ActionPlanTemplate],
    Type[OperationalControlTemplate],
    Type[PerformanceReportTemplate],
    Type[InternalAuditTemplate],
    Type[ManagementReviewTemplate],
    Type[CorrectiveActionTemplate],
    Type[EnMSDocumentationTemplate],
]

TemplateInstance = Union[
    EnergyPolicyTemplate,
    EnergyReviewReportTemplate,
    EnPIMethodologyTemplate,
    ActionPlanTemplate,
    OperationalControlTemplate,
    PerformanceReportTemplate,
    InternalAuditTemplate,
    ManagementReviewTemplate,
    CorrectiveActionTemplate,
    EnMSDocumentationTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "energy_policy",
        "class": EnergyPolicyTemplate,
        "description": (
            "ISO 50001 Clause 5.2 energy policy document with policy statement, "
            "scope and boundaries, commitments to continual improvement, legal "
            "compliance, EnPI improvement and information availability, objectives "
            "framework, roles and responsibilities, communication plan, and "
            "review schedule with triggers."
        ),
        "category": "policy",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "energy_review_report",
        "class": EnergyReviewReportTemplate,
        "description": (
            "ISO 50001 Clause 6.3 energy review report with executive summary, "
            "energy consumption overview by source and end use, Significant "
            "Energy Use (SEU) analysis with Pareto chart data, energy driver "
            "identification with correlation metrics, baseline status, EnPI "
            "performance comparison, improvement opportunities with payback, "
            "and data quality assessment."
        ),
        "category": "review",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "enpi_methodology",
        "class": EnPIMethodologyTemplate,
        "description": (
            "ISO 50006 EnPI/EnB methodology document with methodology overview, "
            "EnPI definitions (measured, ratio, model-based, CUSUM types), "
            "baseline establishment with model selection criteria, data "
            "requirements, measurement plan, statistical validation criteria "
            "(R-squared, CV(RMSE), p-value thresholds), reporting protocol, "
            "and review triggers."
        ),
        "category": "methodology",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "action_plan",
        "class": ActionPlanTemplate,
        "description": (
            "ISO 50001 Clause 6.2 objectives and action plans document with "
            "objectives summary, targets table with EnPI references, detailed "
            "action plans per objective with timeline and resources, financial "
            "summary (investment, savings, NPV, IRR, payback), implementation "
            "schedule with Gantt-style data, risk assessment matrix, and "
            "progress tracking framework."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "operational_control",
        "class": OperationalControlTemplate,
        "description": (
            "ISO 50001 Clause 8.1 operational controls document with SEU "
            "operating criteria, setpoint schedules for occupied/unoccupied/ "
            "seasonal modes, monitoring parameters with alert and alarm "
            "thresholds, deviation response procedures, preventive maintenance "
            "schedules, energy-efficient procurement requirements, "
            "communication of controls, and training requirements."
        ),
        "category": "operations",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "performance_report",
        "class": PerformanceReportTemplate,
        "description": (
            "ISO 50001 Clause 9.1 energy performance M&V report with executive "
            "summary, EnPI dashboard KPI cards, consumption vs baseline "
            "(normalized), CUSUM status and trend, savings summary by SEU, "
            "trend analysis, weather normalization with degree-day data, "
            "year-over-year comparison, data quality metrics, and "
            "recommendations."
        ),
        "category": "performance",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "internal_audit",
        "class": InternalAuditTemplate,
        "description": (
            "ISO 50001 Clause 9.2 internal audit report with audit information "
            "(scope, criteria, team), audit plan, clause-by-clause assessment "
            "for Clauses 4-10 with sub-clauses, findings summary table, "
            "detailed nonconformity reports with root cause analysis, "
            "opportunities for improvement, corrective action register, "
            "audit conclusions, and follow-up schedule."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "management_review",
        "class": ManagementReviewTemplate,
        "description": (
            "ISO 50001 Clause 9.3 management review report with meeting "
            "information, attendees, review of previous actions, energy "
            "performance summary with EnPI trends, policy review assessment, "
            "objectives status tracking, audit results summary, NC/CA status, "
            "resource adequacy evaluation, risks and opportunities, "
            "decisions and actions with owners, and next review date."
        ),
        "category": "management",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "corrective_action",
        "class": CorrectiveActionTemplate,
        "description": (
            "ISO 50001 Clause 10.2 nonconformity and corrective action "
            "register with NC/CA register table, detailed nonconformity "
            "records (description, clause ref, severity, root cause, "
            "correction, corrective action, verification, status), "
            "statistics summary (open/closed/overdue), trend analysis "
            "with common root causes, and effectiveness review."
        ),
        "category": "corrective",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
    {
        "name": "enms_documentation",
        "class": EnMSDocumentationTemplate,
        "description": (
            "Full EnMS documentation package report with document register "
            "of all required ISO 50001 documents, energy manual table of "
            "contents, document control procedures, records retention "
            "schedule, document status matrix, gap analysis identifying "
            "missing and outdated documents, and compliance checklist "
            "against all ISO 50001 documentation requirements."
        ),
        "category": "documentation",
        "formats": ["markdown", "html", "json"],
        "version": "34.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-034 ISO 50001 EnMS report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 ISO 50001 EnMS report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("energy_policy")
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
            name: Template name (e.g., 'energy_policy').
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
            category: Category string (e.g., 'policy', 'review',
                      'methodology', 'planning', 'operations',
                      'performance', 'audit', 'management',
                      'corrective', 'documentation').

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
