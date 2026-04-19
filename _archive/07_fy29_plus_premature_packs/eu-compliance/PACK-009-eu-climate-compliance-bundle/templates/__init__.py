"""
EU Climate Compliance Bundle Templates - PACK-009

This module exports all 8 report template classes for the EU Climate
Compliance Bundle and provides a TemplateRegistry for programmatic
template discovery and instantiation.

Templates:
    - ConsolidatedDashboardTemplate: Multi-regulation compliance overview dashboard
    - CrossRegulationDataMapTemplate: Visual mapping of shared data fields
    - UnifiedGapAnalysisReportTemplate: Cross-framework gap inventory
    - RegulatoryCalendarReportTemplate: Timeline of regulatory deadlines
    - DataConsistencyReportTemplate: Cross-regulation data consistency analysis
    - BundleExecutiveSummaryTemplate: Board-level executive summary
    - DeduplicationSavingsReportTemplate: Data deduplication savings report
    - MultiRegulationAuditTrailTemplate: Consolidated provenance and audit trail

Example:
    >>> from packs.eu_compliance.PACK_009_eu_climate_compliance_bundle.templates import (
    ...     ConsolidatedDashboardTemplate,
    ...     TemplateRegistry,
    ... )
    >>> template = ConsolidatedDashboardTemplate()
    >>> result = template.render(data, fmt="markdown")

    >>> registry = TemplateRegistry()
    >>> names = registry.list_templates()
    >>> template = registry.get_template("consolidated_dashboard")
"""

from typing import Any, Dict, List, Optional, Type, Union

from .consolidated_dashboard import ConsolidatedDashboardTemplate
from .cross_regulation_data_map import CrossRegulationDataMapTemplate
from .unified_gap_analysis_report import UnifiedGapAnalysisReportTemplate
from .regulatory_calendar_report import RegulatoryCalendarReportTemplate
from .data_consistency_report import DataConsistencyReportTemplate
from .bundle_executive_summary import BundleExecutiveSummaryTemplate
from .deduplication_savings_report import DeduplicationSavingsReportTemplate
from .multi_regulation_audit_trail import MultiRegulationAuditTrailTemplate


# Type alias for any template class
TemplateClass = Union[
    Type[ConsolidatedDashboardTemplate],
    Type[CrossRegulationDataMapTemplate],
    Type[UnifiedGapAnalysisReportTemplate],
    Type[RegulatoryCalendarReportTemplate],
    Type[DataConsistencyReportTemplate],
    Type[BundleExecutiveSummaryTemplate],
    Type[DeduplicationSavingsReportTemplate],
    Type[MultiRegulationAuditTrailTemplate],
]

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "consolidated_dashboard": {
        "class": ConsolidatedDashboardTemplate,
        "name": "Consolidated Dashboard",
        "description": (
            "Multi-regulation compliance overview with per-regulation "
            "drill-down, traffic-light status, trend sparklines, and "
            "deadline tracking across CSRD, CBAM, EU Taxonomy, and SFDR."
        ),
        "category": "dashboard",
        "scope": "bundle",
        "version": "1.0",
    },
    "cross_regulation_data_map": {
        "class": CrossRegulationDataMapTemplate,
        "name": "Cross-Regulation Data Map",
        "description": (
            "Visual mapping of shared data fields across regulations with "
            "coverage percentages, field categories, and mapping confidence "
            "levels (exact, approximate, derived)."
        ),
        "category": "data_analysis",
        "scope": "bundle",
        "version": "1.0",
    },
    "unified_gap_analysis": {
        "class": UnifiedGapAnalysisReportTemplate,
        "name": "Unified Gap Analysis Report",
        "description": (
            "Cross-framework gap inventory sorted by multi-regulation impact "
            "score with severity breakdowns, remediation roadmap, timeline "
            "and cost estimates, and multi-regulation impact matrix."
        ),
        "category": "compliance",
        "scope": "bundle",
        "version": "1.0",
    },
    "regulatory_calendar": {
        "class": RegulatoryCalendarReportTemplate,
        "name": "Regulatory Calendar Report",
        "description": (
            "Gantt-style timeline of regulatory deadlines with "
            "cross-regulation dependencies, urgency-sorted upcoming "
            "deadlines, milestone tracking, and conflict detection."
        ),
        "category": "planning",
        "scope": "bundle",
        "version": "1.0",
    },
    "data_consistency": {
        "class": DataConsistencyReportTemplate,
        "name": "Data Consistency Report",
        "description": (
            "Consistency matrix showing agreement and conflicts across "
            "shared fields with per-field comparison, conflict severity, "
            "resolution tracking, and overall consistency score."
        ),
        "category": "data_analysis",
        "scope": "bundle",
        "version": "1.0",
    },
    "bundle_executive_summary": {
        "class": BundleExecutiveSummaryTemplate,
        "name": "Bundle Executive Summary",
        "description": (
            "Board-level overview spanning all 4 regulations with key "
            "metrics, risk flags, regulatory change alerts, strategic "
            "recommendations, and year-over-year comparison."
        ),
        "category": "executive",
        "scope": "bundle",
        "version": "1.0",
    },
    "deduplication_savings": {
        "class": DeduplicationSavingsReportTemplate,
        "name": "Deduplication Savings Report",
        "description": (
            "Data deduplication analysis with fields deduplicated count, "
            "effort reduction quantification, cost impact analysis, "
            "per-category breakdown, and before/after comparison."
        ),
        "category": "financial",
        "scope": "bundle",
        "version": "1.0",
    },
    "multi_regulation_audit_trail": {
        "class": MultiRegulationAuditTrailTemplate,
        "name": "Multi-Regulation Audit Trail",
        "description": (
            "Consolidated provenance report with cross-regulation evidence "
            "mapping, evidence reuse matrix, per-regulation completeness, "
            "SHA-256 hashes, and audit readiness scoring."
        ),
        "category": "audit",
        "scope": "bundle",
        "version": "1.0",
    },
}


class TemplateRegistry:
    """
    Registry for discovering and instantiating PACK-009 report templates.

    Provides a centralized catalog of all available EU Climate Compliance
    Bundle templates with metadata for programmatic discovery, filtering,
    and instantiation.

    Attributes:
        config: Optional global configuration passed to all templates.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("consolidated_dashboard")
        >>> markdown = template.render(data, fmt="markdown")

        >>> dashboards = registry.list_templates(category="dashboard")
        >>> all_keys = registry.get_all_template_keys()
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
        scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all available templates with optional filtering.

        Args:
            category: Filter by category (dashboard, data_analysis, compliance,
                planning, executive, financial, audit).
            scope: Filter by scope (bundle, regulation).

        Returns:
            List of template metadata dictionaries containing:
                - key (str): Template identifier for get_template()
                - name (str): Human-readable template name
                - description (str): Template description
                - category (str): Template category
                - scope (str): Template scope
                - version (str): Template version
        """
        result: List[Dict[str, Any]] = []

        for key, meta in self._catalog.items():
            if category and meta.get("category") != category:
                continue

            if scope and meta.get("scope") != scope:
                continue

            result.append({
                "key": key,
                "name": meta["name"],
                "description": meta["description"],
                "category": meta["category"],
                "scope": meta["scope"],
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
            template_key: Template identifier (e.g. "consolidated_dashboard").
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
            Dictionary with template metadata.

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
            "scope": meta["scope"],
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
    "ConsolidatedDashboardTemplate",
    "CrossRegulationDataMapTemplate",
    "UnifiedGapAnalysisReportTemplate",
    "RegulatoryCalendarReportTemplate",
    "DataConsistencyReportTemplate",
    "BundleExecutiveSummaryTemplate",
    "DeduplicationSavingsReportTemplate",
    "MultiRegulationAuditTrailTemplate",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
]
