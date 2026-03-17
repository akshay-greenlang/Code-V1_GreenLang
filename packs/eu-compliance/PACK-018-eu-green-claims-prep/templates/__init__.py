# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Report Templates
========================================================

This package provides 8 report templates for the PACK-018 EU Green Claims
Prep Pack, covering the full scope of the EU Green Claims Directive
(2023/0085) compliance preparation. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates include
SHA-256 provenance hashing for audit trail integrity.

Templates:
    1.  ClaimAssessmentReportTemplate      - Claim substantiation assessment
    2.  EvidenceDossierReportTemplate      - Verifier-ready evidence dossier
    3.  LifecycleSummaryReportTemplate     - LCA/PEF results summary
    4.  LabelComplianceReportTemplate      - Eco-label compliance audit
    5.  GreenwashingRiskReportTemplate     - Greenwashing risk screening
    6.  ComplianceGapReportTemplate        - Regulatory gap analysis
    7.  GreenClaimsScorecardTemplate       - Readiness benchmark scorecard
    8.  RegulatorySubmissionReportTemplate - Regulatory submission package

Usage:
    >>> from packs.eu_compliance.PACK_018_eu_green_claims_prep.templates import (
    ...     TemplateRegistry,
    ...     ClaimAssessmentReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("claim_assessment_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

Author: GreenLang Team
Version: 18.0.0
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .claim_assessment_report import ClaimAssessmentReportTemplate
except ImportError:
    ClaimAssessmentReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ClaimAssessmentReportTemplate")

try:
    from .evidence_dossier_report import EvidenceDossierReportTemplate
except ImportError:
    EvidenceDossierReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import EvidenceDossierReportTemplate")

try:
    from .lifecycle_summary_report import LifecycleSummaryReportTemplate
except ImportError:
    LifecycleSummaryReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import LifecycleSummaryReportTemplate")

try:
    from .label_compliance_report import LabelComplianceReportTemplate
except ImportError:
    LabelComplianceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import LabelComplianceReportTemplate")

try:
    from .greenwashing_risk_report import GreenwashingRiskReportTemplate
except ImportError:
    GreenwashingRiskReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GreenwashingRiskReportTemplate")

try:
    from .compliance_gap_report import ComplianceGapReportTemplate
except ImportError:
    ComplianceGapReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ComplianceGapReportTemplate")

try:
    from .green_claims_scorecard import GreenClaimsScorecardTemplate
except ImportError:
    GreenClaimsScorecardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GreenClaimsScorecardTemplate")

try:
    from .regulatory_submission_report import RegulatorySubmissionReportTemplate
except ImportError:
    RegulatorySubmissionReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import RegulatorySubmissionReportTemplate")


__all__ = [
    # Template classes
    "ClaimAssessmentReportTemplate",
    "EvidenceDossierReportTemplate",
    "LifecycleSummaryReportTemplate",
    "LabelComplianceReportTemplate",
    "GreenwashingRiskReportTemplate",
    "ComplianceGapReportTemplate",
    "GreenClaimsScorecardTemplate",
    "RegulatorySubmissionReportTemplate",
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
        "name": "claim_assessment_report",
        "class": ClaimAssessmentReportTemplate,
        "description": (
            "Comprehensive assessment of environmental claims against EU Green "
            "Claims Directive substantiation requirements. Evaluates claim "
            "inventory, evidence quality, risk ratings, and produces prioritised "
            "remediation recommendations."
        ),
        "category": "assessment",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "evidence_dossier_report",
        "class": EvidenceDossierReportTemplate,
        "description": (
            "Verifier-ready evidence dossier compiling all supporting documents "
            "for environmental claims. Tracks document inventory, chain of "
            "custody, validity periods, and evidence completeness."
        ),
        "category": "evidence",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "lifecycle_summary_report",
        "class": LifecycleSummaryReportTemplate,
        "description": (
            "Lifecycle assessment (LCA) and Product Environmental Footprint "
            "(PEF) summary report. Covers system boundary, impact categories, "
            "hotspot analysis, PEF scoring, and data quality ratings."
        ),
        "category": "lifecycle",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "label_compliance_report",
        "class": LabelComplianceReportTemplate,
        "description": (
            "Eco-label and certification mark compliance audit against EU Green "
            "Claims Directive Article 10 requirements. Evaluates label scheme "
            "governance, certificate validity, and identifies non-compliant labels."
        ),
        "category": "labels",
        "directive_reference": "EU Green Claims Directive 2023/0085 Article 10",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "greenwashing_risk_report",
        "class": GreenwashingRiskReportTemplate,
        "description": (
            "Greenwashing risk screening report covering the Seven Sins of "
            "Greenwashing, EU-prohibited practices, vague or misleading "
            "language detection, and high-risk claim identification with "
            "prioritised remediation actions."
        ),
        "category": "risk",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "compliance_gap_report",
        "class": ComplianceGapReportTemplate,
        "description": (
            "Regulatory compliance gap analysis mapping current green claims "
            "practices against EU Green Claims Directive requirements. Produces "
            "a risk-weighted priority matrix and phased remediation roadmap."
        ),
        "category": "gap_analysis",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "green_claims_scorecard",
        "class": GreenClaimsScorecardTemplate,
        "description": (
            "Multi-dimensional readiness benchmark scorecard covering "
            "substantiation quality, evidence management, label governance, "
            "lifecycle coverage, communication practices, and verification "
            "readiness with maturity level and trend analysis."
        ),
        "category": "scorecard",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
    {
        "name": "regulatory_submission_report",
        "class": RegulatorySubmissionReportTemplate,
        "description": (
            "Complete regulatory submission package for EU Green Claims "
            "Directive compliance. Assembles claims summary, evidence packages, "
            "verification outcomes, and formal compliance declarations for "
            "submission to Member State competent authorities."
        ),
        "category": "submission",
        "directive_reference": "EU Green Claims Directive 2023/0085",
        "formats": ["markdown", "html", "json"],
        "version": "18.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-018 EU Green Claims Prep Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 EU Green Claims report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("claim_assessment_report")
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
            name: Template name (e.g., 'claim_assessment_report').
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
            category: Category string (e.g., 'assessment', 'evidence',
                      'lifecycle', 'labels', 'risk', 'gap_analysis',
                      'scorecard', 'submission').

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
