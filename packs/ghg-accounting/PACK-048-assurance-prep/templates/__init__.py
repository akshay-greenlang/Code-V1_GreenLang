# -*- coding: utf-8 -*-
"""
PACK-048 GHG Assurance Prep Pack - Report Templates
=============================================================

This package provides 10 report templates for the PACK-048 GHG Assurance
Prep Pack. Each template supports multiple rendering formats (Markdown,
HTML, JSON, and in some cases XBRL) with SHA-256 provenance hashing for
audit trail integrity.

Templates:
    1. AssuranceReadinessDashboard  - Overall readiness score, categories, gaps
    2. EvidencePackageIndex         - Evidence inventory by scope, quality, completeness
    3. ControlSelfAssessmentReport  - 25-control register, maturity heatmap, deficiencies
    4. VerifierQueryRegister        - Query/finding log, SLA, severity, escalation
    5. MaterialityAssessmentReport  - Quantitative/qualitative materiality, scope breakdown
    6. SamplingPlanReport           - Population, stratification, high-value items, stats
    7. RegulatoryRequirementReport  - Jurisdiction requirements, compliance, timeline, gaps
    8. CostTimelineReport           - Cost breakdown, level comparison, Gantt, projection
    9. ISAE3410EvidenceBundle       - 8-section ISAE 3410 mapping, evidence, XBRL
    10. MultiYearAssuranceTrend     - Multi-year readiness, findings, maturity, opinions

Usage:
    >>> from packs.ghg_accounting.PACK_048_assurance_prep.templates import (
    ...     TemplateRegistry,
    ...     AssuranceReadinessDashboard,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("assurance_readiness_dashboard")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 48.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module metadata
# ---------------------------------------------------------------------------

__version__ = "1.0.0"
__pack__ = "PACK-048"
__pack_name__ = "GHG Assurance Prep Pack"
__templates_count__ = 10

# ---------------------------------------------------------------------------
# Template imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .assurance_readiness_dashboard import AssuranceReadinessDashboard
except ImportError as e:
    logger.warning("Failed to import AssuranceReadinessDashboard: %s", e)
    AssuranceReadinessDashboard = None  # type: ignore[assignment,misc]

try:
    from .evidence_package_index import EvidencePackageIndex
except ImportError as e:
    logger.warning("Failed to import EvidencePackageIndex: %s", e)
    EvidencePackageIndex = None  # type: ignore[assignment,misc]

try:
    from .control_self_assessment_report import ControlSelfAssessmentReport
except ImportError as e:
    logger.warning("Failed to import ControlSelfAssessmentReport: %s", e)
    ControlSelfAssessmentReport = None  # type: ignore[assignment,misc]

try:
    from .verifier_query_register import VerifierQueryRegister
except ImportError as e:
    logger.warning("Failed to import VerifierQueryRegister: %s", e)
    VerifierQueryRegister = None  # type: ignore[assignment,misc]

try:
    from .materiality_assessment_report import MaterialityAssessmentReport
except ImportError as e:
    logger.warning("Failed to import MaterialityAssessmentReport: %s", e)
    MaterialityAssessmentReport = None  # type: ignore[assignment,misc]

try:
    from .sampling_plan_report import SamplingPlanReport
except ImportError as e:
    logger.warning("Failed to import SamplingPlanReport: %s", e)
    SamplingPlanReport = None  # type: ignore[assignment,misc]

try:
    from .regulatory_requirement_report import RegulatoryRequirementReport
except ImportError as e:
    logger.warning("Failed to import RegulatoryRequirementReport: %s", e)
    RegulatoryRequirementReport = None  # type: ignore[assignment,misc]

try:
    from .cost_timeline_report import CostTimelineReport
except ImportError as e:
    logger.warning("Failed to import CostTimelineReport: %s", e)
    CostTimelineReport = None  # type: ignore[assignment,misc]

try:
    from .isae_3410_evidence_bundle import ISAE3410EvidenceBundle
except ImportError as e:
    logger.warning("Failed to import ISAE3410EvidenceBundle: %s", e)
    ISAE3410EvidenceBundle = None  # type: ignore[assignment,misc]

try:
    from .multi_year_assurance_trend import MultiYearAssuranceTrend
except ImportError as e:
    logger.warning("Failed to import MultiYearAssuranceTrend: %s", e)
    MultiYearAssuranceTrend = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Exported names
# ---------------------------------------------------------------------------

__all__ = [
    # Template classes
    "AssuranceReadinessDashboard",
    "EvidencePackageIndex",
    "ControlSelfAssessmentReport",
    "VerifierQueryRegister",
    "MaterialityAssessmentReport",
    "SamplingPlanReport",
    "RegulatoryRequirementReport",
    "CostTimelineReport",
    "ISAE3410EvidenceBundle",
    "MultiYearAssuranceTrend",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
    # Type aliases
    "TemplateClass",
    "TemplateInstance",
]


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TemplateClass = Union[
    Type[AssuranceReadinessDashboard],
    Type[EvidencePackageIndex],
    Type[ControlSelfAssessmentReport],
    Type[VerifierQueryRegister],
    Type[MaterialityAssessmentReport],
    Type[SamplingPlanReport],
    Type[RegulatoryRequirementReport],
    Type[CostTimelineReport],
    Type[ISAE3410EvidenceBundle],
    Type[MultiYearAssuranceTrend],
]

TemplateInstance = Union[
    AssuranceReadinessDashboard,
    EvidencePackageIndex,
    ControlSelfAssessmentReport,
    VerifierQueryRegister,
    MaterialityAssessmentReport,
    SamplingPlanReport,
    RegulatoryRequirementReport,
    CostTimelineReport,
    ISAE3410EvidenceBundle,
    MultiYearAssuranceTrend,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "assurance_readiness_dashboard",
        "class": AssuranceReadinessDashboard,
        "description": (
            "Executive readiness dashboard with overall score (0-100), "
            "traffic light indicators, 8-10 category breakdowns, sparkline "
            "trend, top 5 remediation gaps, time-to-ready estimate, and "
            "standard-specific views (ISAE 3410, ISO 14064-3, AA1000AS)."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "evidence_package_index",
        "class": EvidencePackageIndex,
        "description": (
            "Complete evidence inventory by scope and category with quality "
            "distribution (pie chart data), completeness percentages by scope "
            "(S1/S2/S3), missing evidence highlights, SHA-256 hash per "
            "evidence item, and package version and generation timestamp."
        ),
        "category": "evidence",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "control_self_assessment_report",
        "class": ControlSelfAssessmentReport,
        "description": (
            "25-control register with design and operating effectiveness "
            "ratings, control maturity heatmap data (category x maturity "
            "level), deficiency log with severity and remediation status, "
            "control coverage summary, and improvement recommendations."
        ),
        "category": "controls",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "verifier_query_register",
        "class": VerifierQueryRegister,
        "description": (
            "IR/query/finding log table sortable by priority, status, and "
            "category with SLA compliance dashboard, outstanding query "
            "counts by priority, finding severity distribution, resolution "
            "timeline, and escalation history."
        ),
        "category": "verifier",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "materiality_assessment_report",
        "class": MaterialityAssessmentReport,
        "description": (
            "Quantitative materiality thresholds (overall, performance, "
            "clearly trivial), scope-specific materiality breakdown, "
            "qualitative factors assessment, materiality methodology "
            "narrative, and comparison to prior period materiality."
        ),
        "category": "materiality",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "sampling_plan_report",
        "class": SamplingPlanReport,
        "description": (
            "Population description with stratification table (stratum, "
            "count, value, sample size), high-value items requiring 100% "
            "testing, key items for judgmental review, sample selection "
            "methodology, and statistical parameters."
        ),
        "category": "sampling",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "regulatory_requirement_report",
        "class": RegulatoryRequirementReport,
        "description": (
            "Jurisdiction-by-jurisdiction requirements table with compliance "
            "status, timeline visualisation of requirement effective dates, "
            "gap analysis summary with action items, and multi-jurisdiction "
            "overlap analysis."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "cost_timeline_report",
        "class": CostTimelineReport,
        "description": (
            "Engagement cost breakdown by component, assurance level "
            "comparison (limited vs reasonable), timeline Gantt chart "
            "data, resource allocation by role, multi-year cost "
            "projection, and verifier RFP summary data."
        ),
        "category": "planning",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
    {
        "name": "isae_3410_evidence_bundle",
        "class": ISAE3410EvidenceBundle,
        "description": (
            "ISAE 3410-specific 8-section mapping (engagement terms, GHG "
            "statement, risk assessment, evidence, controls, analytical "
            "procedures, representations, conclusions) with cross-reference "
            "to evidence items per section and XBRL tag mapping."
        ),
        "category": "standard",
        "formats": ["markdown", "html", "json", "xbrl"],
        "version": "48.0.0",
    },
    {
        "name": "multi_year_assurance_trend",
        "class": MultiYearAssuranceTrend,
        "description": (
            "Multi-year trend analysis with readiness score evolution, "
            "finding recurrence analysis (recurring vs new vs resolved), "
            "control maturity evolution, evidence quality trend, query "
            "volume and resolution time trend, cost trajectory, and "
            "assurance opinion history."
        ),
        "category": "trend",
        "formats": ["markdown", "html", "json"],
        "version": "48.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-048 GHG Assurance Prep report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 report templates. Templates can be listed, filtered by
    category, retrieved by name, and rendered in markdown/HTML/JSON
    formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("assurance_readiness_dashboard")
        >>> md = template.render_markdown(data)
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, TemplateInstance] = {}

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
        return [
            defn["name"] for defn in TEMPLATE_CATALOG
            if defn["class"] is not None
        ]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TemplateInstance:
        """
        Get a template instance by name.

        Args:
            name: Template name (e.g., 'assurance_readiness_dashboard').
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
            category: Category string (e.g., 'executive', 'evidence',
                      'controls', 'verifier', 'materiality', 'sampling',
                      'regulatory', 'planning', 'standard', 'trend').

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
