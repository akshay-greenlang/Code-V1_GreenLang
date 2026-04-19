# -*- coding: utf-8 -*-
"""
PACK-023 SBTi Alignment Pack - Report Templates
==================================================

10 report templates for SBTi alignment documentation, covering target
summaries, validation reports, Scope 3 screening, SDA pathway analysis,
FLAG assessment, temperature ratings, progress dashboards, FI portfolio
analysis, submission packages, and framework crosswalk mapping.

Templates:
    TargetSummaryReportTemplate         -- SBTi target summary with pathway milestones
    ValidationReportTemplate            -- 42-criterion validation assessment report
    Scope3ScreeningReportTemplate       -- Scope 3 materiality screening report
    SDAPathwayReportTemplate            -- SDA intensity convergence pathway report
    FLAGAssessmentReportTemplate        -- FLAG commodity assessment report
    TemperatureRatingReportTemplate     -- Temperature rating with portfolio scores
    ProgressDashboardReportTemplate     -- Annual progress tracking dashboard
    FIPortfolioReportTemplate           -- FI portfolio target and coverage report
    SubmissionPackageReportTemplate     -- SBTi submission package documentation
    FrameworkCrosswalkReportTemplate    -- Cross-framework mapping report

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-023 SBTi Alignment Pack
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__version__: str = "1.0.0"
__pack__: str = "PACK-023"
__pack_name__: str = "SBTi Alignment Pack"
__templates_count__: int = 10

_loaded_templates: list[str] = []

# ---------------------------------------------------------------------------
# Template imports with graceful fallback
# ---------------------------------------------------------------------------

try:
    from .target_summary_report import TargetSummaryReportTemplate
    _loaded_templates.append("TargetSummaryReportTemplate")
except ImportError as e:
    logger.debug("Template (TargetSummaryReportTemplate) not available: %s", e)
    TargetSummaryReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .validation_report import ValidationReportTemplate
    _loaded_templates.append("ValidationReportTemplate")
except ImportError as e:
    logger.debug("Template (ValidationReportTemplate) not available: %s", e)
    ValidationReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .scope3_screening_report import Scope3ScreeningReportTemplate
    _loaded_templates.append("Scope3ScreeningReportTemplate")
except ImportError as e:
    logger.debug("Template (Scope3ScreeningReportTemplate) not available: %s", e)
    Scope3ScreeningReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .sda_pathway_report import SDAPathwayReportTemplate
    _loaded_templates.append("SDAPathwayReportTemplate")
except ImportError as e:
    logger.debug("Template (SDAPathwayReportTemplate) not available: %s", e)
    SDAPathwayReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .flag_assessment_report import FLAGAssessmentReportTemplate
    _loaded_templates.append("FLAGAssessmentReportTemplate")
except ImportError as e:
    logger.debug("Template (FLAGAssessmentReportTemplate) not available: %s", e)
    FLAGAssessmentReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .temperature_rating_report import TemperatureRatingReportTemplate
    _loaded_templates.append("TemperatureRatingReportTemplate")
except ImportError as e:
    logger.debug("Template (TemperatureRatingReportTemplate) not available: %s", e)
    TemperatureRatingReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .progress_dashboard_report import ProgressDashboardReportTemplate
    _loaded_templates.append("ProgressDashboardReportTemplate")
except ImportError as e:
    logger.debug("Template (ProgressDashboardReportTemplate) not available: %s", e)
    ProgressDashboardReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .fi_portfolio_report import FIPortfolioReportTemplate
    _loaded_templates.append("FIPortfolioReportTemplate")
except ImportError as e:
    logger.debug("Template (FIPortfolioReportTemplate) not available: %s", e)
    FIPortfolioReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .submission_package_report import SubmissionPackageReportTemplate
    _loaded_templates.append("SubmissionPackageReportTemplate")
except ImportError as e:
    logger.debug("Template (SubmissionPackageReportTemplate) not available: %s", e)
    SubmissionPackageReportTemplate = None  # type: ignore[assignment, misc]

try:
    from .framework_crosswalk_report import FrameworkCrosswalkReportTemplate
    _loaded_templates.append("FrameworkCrosswalkReportTemplate")
except ImportError as e:
    logger.debug("Template (FrameworkCrosswalkReportTemplate) not available: %s", e)
    FrameworkCrosswalkReportTemplate = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Template Catalog and Registry
# ---------------------------------------------------------------------------

TEMPLATE_CATALOG: list[dict] = [
    {
        "name": "target_summary",
        "class": TargetSummaryReportTemplate,
        "description": "SBTi target summary with ACA/SDA/FLAG pathway milestones and ambition assessment",
        "category": "targets",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "validation",
        "class": ValidationReportTemplate,
        "description": "42-criterion SBTi validation assessment with gap analysis and remediation guidance",
        "category": "validation",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "scope3_screening",
        "class": Scope3ScreeningReportTemplate,
        "description": "15-category Scope 3 materiality screening with coverage and priority analysis",
        "category": "screening",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "sda_pathway",
        "class": SDAPathwayReportTemplate,
        "description": "SDA intensity convergence pathway with sector benchmarks and annual milestones",
        "category": "pathways",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "flag_assessment",
        "class": FLAGAssessmentReportTemplate,
        "description": "FLAG commodity assessment with trigger evaluation and deforestation commitment",
        "category": "assessment",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "temperature_rating",
        "class": TemperatureRatingReportTemplate,
        "description": "Temperature rating with company scores, portfolio aggregation, and what-if analysis",
        "category": "scoring",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "progress_dashboard",
        "class": ProgressDashboardReportTemplate,
        "description": "Annual progress tracking dashboard with RAG status and corrective actions",
        "category": "tracking",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "fi_portfolio",
        "class": FIPortfolioReportTemplate,
        "description": "FI portfolio target analysis with PCAF scoring, coverage, and engagement tracking",
        "category": "financial",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "submission_package",
        "class": SubmissionPackageReportTemplate,
        "description": "SBTi target submission documentation package with readiness assessment",
        "category": "submission",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
    {
        "name": "framework_crosswalk",
        "class": FrameworkCrosswalkReportTemplate,
        "description": "Cross-framework mapping report for CDP, TCFD, ESRS E1, GHG Protocol, ISO 14064",
        "category": "reporting",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    },
]


class TemplateRegistry:
    """Registry for discovering, instantiating, and managing report templates.

    Provides a unified interface to list available templates, retrieve
    template instances by name, and check template availability.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_template_names()
        >>> template = registry.get("target_summary")
    """

    def __init__(self) -> None:
        """Initialize the template registry from TEMPLATE_CATALOG."""
        self._templates: dict[str, dict] = {}
        self._instances: dict[str, Any] = {}
        for defn in TEMPLATE_CATALOG:
            if defn["class"] is not None:
                self._templates[defn["name"]] = defn

    def list_templates(self) -> list[dict]:
        """Return metadata for all available templates."""
        return [
            {
                "name": d["name"],
                "description": d["description"],
                "category": d["category"],
                "formats": d["formats"],
                "version": d["version"],
            }
            for d in TEMPLATE_CATALOG
            if d["class"] is not None
        ]

    def list_template_names(self) -> list[str]:
        """Return names of all available templates."""
        return [d["name"] for d in TEMPLATE_CATALOG if d["class"] is not None]

    def get(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get a template instance by name.

        Args:
            name: Template name from TEMPLATE_CATALOG.
            config: Optional configuration dict passed to template constructor.

        Returns:
            Template instance.

        Raises:
            KeyError: If template name not found.
        """
        if name not in self._templates:
            available = ", ".join(self._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. Available: {available}"
            )
        if config is not None or name not in self._instances:
            cls = self._templates[name]["class"]
            inst = cls(config=config)
            if config is None:
                self._instances[name] = inst
            return inst
        return self._instances[name]

    def has_template(self, name: str) -> bool:
        """Check if a template is available."""
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return number of available templates."""
        return len(self._templates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: list[str] = [
    "__version__",
    "__pack__",
    "__pack_name__",
    "__templates_count__",
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "TargetSummaryReportTemplate",
    "ValidationReportTemplate",
    "Scope3ScreeningReportTemplate",
    "SDAPathwayReportTemplate",
    "FLAGAssessmentReportTemplate",
    "TemperatureRatingReportTemplate",
    "ProgressDashboardReportTemplate",
    "FIPortfolioReportTemplate",
    "SubmissionPackageReportTemplate",
    "FrameworkCrosswalkReportTemplate",
]

logger.info(
    "PACK-023 SBTi Alignment templates: %d/%d loaded",
    len(_loaded_templates),
    __templates_count__,
)
