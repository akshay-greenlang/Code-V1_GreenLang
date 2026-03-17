"""
CBAM Readiness Pack Templates - PACK-004

This module exports all 8 CBAM report template classes and provides a
TemplateRegistry for programmatic template discovery and instantiation.

Templates:
    - QuarterlyReportTemplate: Transitional quarterly reporting
    - AnnualDeclarationTemplate: Annual CBAM declaration
    - CertificateDashboardTemplate: Certificate obligation dashboard
    - SupplierScorecardTemplate: Supplier data quality scorecard
    - ComplianceStatusTemplate: Overall compliance dashboard
    - CostProjectionTemplate: Certificate cost forecast
    - DeMinimisReportTemplate: De minimis threshold tracking
    - VerificationReportTemplate: Verification status report

Example:
    >>> from packs.eu_compliance.PACK_004_cbam_readiness.templates import (
    ...     QuarterlyReportTemplate,
    ...     TemplateRegistry,
    ... )
    >>> template = QuarterlyReportTemplate()
    >>> markdown = template.render_markdown(data)

    >>> registry = TemplateRegistry()
    >>> names = registry.list_templates()
    >>> template = registry.get_template("quarterly_report")
"""

from typing import Any, Dict, List, Optional, Type, Union

from .quarterly_report import QuarterlyReportTemplate
from .annual_declaration import AnnualDeclarationTemplate
from .certificate_dashboard import CertificateDashboardTemplate
from .supplier_scorecard import SupplierScorecardTemplate
from .compliance_status import ComplianceStatusTemplate
from .cost_projection import CostProjectionTemplate
from .deminimis_report import DeMinimisReportTemplate
from .verification_report import VerificationReportTemplate


# Type alias for any template class
TemplateClass = Union[
    Type[QuarterlyReportTemplate],
    Type[AnnualDeclarationTemplate],
    Type[CertificateDashboardTemplate],
    Type[SupplierScorecardTemplate],
    Type[ComplianceStatusTemplate],
    Type[CostProjectionTemplate],
    Type[DeMinimisReportTemplate],
    Type[VerificationReportTemplate],
]

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "quarterly_report": {
        "class": QuarterlyReportTemplate,
        "name": "Quarterly Report",
        "description": (
            "CBAM transitional quarterly report with goods summaries, "
            "emission breakdowns, installation details, and compliance status."
        ),
        "category": "reporting",
        "phase": "transitional",
        "version": "1.0",
    },
    "annual_declaration": {
        "class": AnnualDeclarationTemplate,
        "name": "Annual Declaration",
        "description": (
            "Annual CBAM declaration with import summaries, certificate "
            "obligations, cost summaries, and free allocation tracking."
        ),
        "category": "reporting",
        "phase": "definitive",
        "version": "1.0",
    },
    "certificate_dashboard": {
        "class": CertificateDashboardTemplate,
        "name": "Certificate Dashboard",
        "description": (
            "Certificate obligation dashboard with KPI cards, waterfall charts, "
            "ETS price trends, and cost scenario projections."
        ),
        "category": "dashboard",
        "phase": "definitive",
        "version": "1.0",
    },
    "supplier_scorecard": {
        "class": SupplierScorecardTemplate,
        "name": "Supplier Scorecard",
        "description": (
            "Supplier data quality scorecard with quality tiers, missing data "
            "heatmaps, recommendations, and submission history."
        ),
        "category": "supplier_management",
        "phase": "both",
        "version": "1.0",
    },
    "compliance_status": {
        "class": ComplianceStatusTemplate,
        "name": "Compliance Status",
        "description": (
            "Overall compliance dashboard with regulatory timeline, scoring, "
            "obligation checklists, risk indicators, and action items."
        ),
        "category": "dashboard",
        "phase": "both",
        "version": "1.0",
    },
    "cost_projection": {
        "class": CostProjectionTemplate,
        "name": "Cost Projection",
        "description": (
            "Certificate cost forecast with scenario comparisons, annual "
            "forecasts, sensitivity analysis, and budget planning."
        ),
        "category": "financial",
        "phase": "definitive",
        "version": "1.0",
    },
    "deminimis_report": {
        "class": DeMinimisReportTemplate,
        "name": "De Minimis Report",
        "description": (
            "De minimis threshold tracking with progress bars, monthly trends, "
            "alerts, exemptions, and year-end projections."
        ),
        "category": "compliance",
        "phase": "both",
        "version": "1.0",
    },
    "verification_report": {
        "class": VerificationReportTemplate,
        "name": "Verification Report",
        "description": (
            "Verification status report with engagement overview, timeline, "
            "findings, materiality assessment, and statement tracking."
        ),
        "category": "verification",
        "phase": "definitive",
        "version": "1.0",
    },
}


class TemplateRegistry:
    """
    Registry for discovering and instantiating CBAM report templates.

    Provides a centralized catalog of all available CBAM templates with
    metadata for programmatic discovery, filtering, and instantiation.

    Attributes:
        config: Optional global configuration passed to all templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("quarterly_report")
        >>> markdown = template.render_markdown(data)

        >>> # Filter by category
        >>> dashboards = registry.list_templates(category="dashboard")
        >>> # Filter by phase
        >>> transitional = registry.list_templates(phase="transitional")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize TemplateRegistry.

        Args:
            config: Optional global configuration dictionary that will be
                passed to template constructors when instantiated via
                get_template().
        """
        self.config = config or {}
        self._catalog: Dict[str, Dict[str, Any]] = TEMPLATE_CATALOG.copy()

    def list_templates(
        self,
        category: Optional[str] = None,
        phase: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available templates with optional filtering.

        Args:
            category: Filter by category (reporting, dashboard, supplier_management,
                financial, compliance, verification).
            phase: Filter by phase (transitional, definitive, both).

        Returns:
            List of template metadata dictionaries containing:
                - key (str): Template identifier for get_template()
                - name (str): Human-readable template name
                - description (str): Template description
                - category (str): Template category
                - phase (str): CBAM phase applicability
                - version (str): Template version
        """
        result: List[Dict[str, Any]] = []

        for key, meta in self._catalog.items():
            if category and meta.get("category") != category:
                continue

            if phase:
                template_phase = meta.get("phase", "both")
                if template_phase != "both" and template_phase != phase:
                    continue

            result.append({
                "key": key,
                "name": meta["name"],
                "description": meta["description"],
                "category": meta["category"],
                "phase": meta["phase"],
                "version": meta["version"],
            })

        return result

    def get_template(
        self,
        template_key: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Instantiate and return a template by its registry key.

        Args:
            template_key: Template identifier (e.g. "quarterly_report").
            config: Optional per-template configuration. If None, uses
                the registry-level config.

        Returns:
            Instantiated template object.

        Raises:
            KeyError: If template_key is not found in the registry.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. "
                f"Available templates: {available}"
            )

        template_cls = self._catalog[template_key]["class"]
        effective_config = config if config is not None else self.config
        return template_cls(config=effective_config)

    def get_template_metadata(self, template_key: str) -> Dict[str, Any]:
        """
        Get metadata for a specific template without instantiating it.

        Args:
            template_key: Template identifier.

        Returns:
            Dictionary with template metadata (name, description, category,
            phase, version).

        Raises:
            KeyError: If template_key is not found in the registry.
        """
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(
                f"Template '{template_key}' not found. "
                f"Available templates: {available}"
            )

        meta = self._catalog[template_key]
        return {
            "key": template_key,
            "name": meta["name"],
            "description": meta["description"],
            "category": meta["category"],
            "phase": meta["phase"],
            "version": meta["version"],
        }

    def get_all_template_keys(self) -> List[str]:
        """
        Get all registered template keys.

        Returns:
            Sorted list of template key strings.
        """
        return sorted(self._catalog.keys())

    def has_template(self, template_key: str) -> bool:
        """
        Check if a template key is registered.

        Args:
            template_key: Template identifier to check.

        Returns:
            True if the template exists in the registry.
        """
        return template_key in self._catalog

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._catalog)


# Module-level exports
__all__ = [
    # Template classes
    "QuarterlyReportTemplate",
    "AnnualDeclarationTemplate",
    "CertificateDashboardTemplate",
    "SupplierScorecardTemplate",
    "ComplianceStatusTemplate",
    "CostProjectionTemplate",
    "DeMinimisReportTemplate",
    "VerificationReportTemplate",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
]
