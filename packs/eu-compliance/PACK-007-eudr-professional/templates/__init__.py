# -*- coding: utf-8 -*-
"""
PACK-007 EUDR Professional Pack - Report Templates
======================================================

This package provides 10 report templates for the PACK-007 EUDR Professional
Pack, covering advanced risk analysis through annual compliance reporting
per EU Regulation 2023/1115.

Templates:
    1.  AdvancedRiskReportTemplate           - Monte Carlo simulation results and risk projections
    2.  SatelliteMonitoringReportTemplate    - Deforestation alerts and satellite imagery analysis
    3.  SupplierBenchmarkReportTemplate      - Industry-relative supplier performance scoring
    4.  SupplyChainMapReportTemplate         - Multi-tier supply chain network visualization
    5.  PortfolioDashboardTemplate           - Multi-operator portfolio compliance dashboard
    6.  AuditReadinessReportTemplate         - CA inspection readiness and evidence inventory
    7.  ProtectedAreaReportTemplate          - WDPA/KBA overlay and buffer zone analysis
    8.  RegulatoryChangeReportTemplate       - Regulatory amendment impact and migration plan
    9.  AnnualComplianceReportTemplate       - Year-end EUDR compliance assessment
    10. GrievanceLogReportTemplate           - Stakeholder grievance tracking and resolution log

Usage:
    >>> from packs.eu_compliance.PACK_007_eudr_professional.templates import (
    ...     TemplateRegistry,
    ...     AdvancedRiskReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("advanced_risk_report")
    >>> md = template.render_markdown(data)

Author: GreenLang Team
Version: 7.0.0
Pack: PACK-007 EUDR Professional Pack
"""

__version__ = "7.0.0"
__pack_id__ = "PACK-007"
__pack_name__ = "EUDR Professional Pack"

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .advanced_risk_report import AdvancedRiskReportTemplate
except ImportError:
    AdvancedRiskReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AdvancedRiskReportTemplate")

try:
    from .satellite_monitoring_report import SatelliteMonitoringReportTemplate
except ImportError:
    SatelliteMonitoringReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SatelliteMonitoringReportTemplate")

try:
    from .supplier_benchmark_report import SupplierBenchmarkReportTemplate
except ImportError:
    SupplierBenchmarkReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SupplierBenchmarkReportTemplate")

try:
    from .supply_chain_map_report import SupplyChainMapReportTemplate
except ImportError:
    SupplyChainMapReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import SupplyChainMapReportTemplate")

try:
    from .portfolio_dashboard import PortfolioDashboardTemplate
except ImportError:
    PortfolioDashboardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import PortfolioDashboardTemplate")

try:
    from .audit_readiness_report import AuditReadinessReportTemplate
except ImportError:
    AuditReadinessReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AuditReadinessReportTemplate")

try:
    from .protected_area_report import ProtectedAreaReportTemplate
except ImportError:
    ProtectedAreaReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ProtectedAreaReportTemplate")

try:
    from .regulatory_change_report import RegulatoryChangeReportTemplate
except ImportError:
    RegulatoryChangeReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import RegulatoryChangeReportTemplate")

try:
    from .annual_compliance_report import AnnualComplianceReportTemplate
except ImportError:
    AnnualComplianceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import AnnualComplianceReportTemplate")

try:
    from .grievance_log_report import GrievanceLogReportTemplate
except ImportError:
    GrievanceLogReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import GrievanceLogReportTemplate")


__all__ = [
    # Version info
    "__version__",
    "__pack_id__",
    "__pack_name__",
    # Template classes
    "AdvancedRiskReportTemplate",
    "SatelliteMonitoringReportTemplate",
    "SupplierBenchmarkReportTemplate",
    "SupplyChainMapReportTemplate",
    "PortfolioDashboardTemplate",
    "AuditReadinessReportTemplate",
    "ProtectedAreaReportTemplate",
    "RegulatoryChangeReportTemplate",
    "AnnualComplianceReportTemplate",
    "GrievanceLogReportTemplate",
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
        "name": "advanced_risk_report",
        "class": AdvancedRiskReportTemplate,
        "description": (
            "Monte Carlo risk simulation report with scenario analysis, "
            "VaR calculations, tornado sensitivity analysis, risk projections, "
            "and mitigation recommendations for EUDR compliance risk assessment."
        ),
        "category": "risk",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "satellite_monitoring_report",
        "class": SatelliteMonitoringReportTemplate,
        "description": (
            "Satellite imagery monitoring report with deforestation alerts, "
            "fire detection events, temporal comparison, affected plot mapping, "
            "and NDVI change analysis for EUDR plot-level verification."
        ),
        "category": "monitoring",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "supplier_benchmark_report",
        "class": SupplierBenchmarkReportTemplate,
        "description": (
            "Industry-relative supplier benchmarking report with composite "
            "scoring across 7 performance dimensions, peer ranking, percentile "
            "analysis, trend tracking, and best practice identification."
        ),
        "category": "benchmarking",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "supply_chain_map_report",
        "class": SupplyChainMapReportTemplate,
        "description": (
            "Multi-tier supply chain network visualization with tier summaries, "
            "concentration risk metrics, geographic origin mapping, critical "
            "path analysis, and diversification recommendations."
        ),
        "category": "supply_chain",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "portfolio_dashboard",
        "class": PortfolioDashboardTemplate,
        "description": (
            "Multi-operator portfolio compliance dashboard with operator "
            "scoring, DDS submission tracking, shared supplier pool statistics, "
            "cost allocation breakdown, and risk aggregation across entities."
        ),
        "category": "portfolio",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "audit_readiness_report",
        "class": AuditReadinessReportTemplate,
        "description": (
            "Competent authority inspection readiness report with evidence "
            "inventory, compliance checklist, gap summary, remediation actions, "
            "mock audit results, and document retention verification."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "protected_area_report",
        "class": ProtectedAreaReportTemplate,
        "description": (
            "WDPA and Key Biodiversity Area overlay analysis report with "
            "buffer zone analysis, indigenous land flags, UNESCO/Ramsar "
            "proximity detection, and risk amplification scoring."
        ),
        "category": "geospatial",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "regulatory_change_report",
        "class": RegulatoryChangeReportTemplate,
        "description": (
            "Regulatory amendment impact assessment report with gap analysis, "
            "migration checklist, affected process mapping, implementation "
            "status tracking, and cross-regulation impact assessment."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "annual_compliance_report",
        "class": AnnualComplianceReportTemplate,
        "description": (
            "Year-end EUDR compliance assessment report with quarterly "
            "performance trends, risk evolution tracking, supplier trends, "
            "DDS submission statistics, audit findings, and next-year priorities."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
    {
        "name": "grievance_log_report",
        "class": GrievanceLogReportTemplate,
        "description": (
            "Stakeholder grievance tracking and resolution log with complaint "
            "records, investigation status, resolution records, geographic "
            "distribution, trend analysis, and overall statistics."
        ),
        "category": "grievance",
        "formats": ["markdown", "html", "json", "pdf"],
        "version": "7.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-007 EUDR Professional Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 EUDR Professional report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON/PDF.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("advanced_risk_report")
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
            "PACK-007 TemplateRegistry initialized with %d templates",
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
            name: Template name (e.g., 'advanced_risk_report').
            config: Optional configuration overrides.

        Returns:
            Template instance with render_markdown, render_html,
            render_json, and render_pdf methods.

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
            format: Output format ('markdown', 'html', 'json', 'pdf').
            config: Optional template configuration.

        Returns:
            Rendered content (str for markdown/html, dict for json,
            bytes for pdf).

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
        elif format == "pdf":
            if hasattr(template, "render_pdf"):
                return template.render_pdf(data)
            raise ValueError(
                f"Template '{template_name}' does not support PDF format."
            )
        else:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Use 'markdown', 'html', 'json', or 'pdf'."
            )

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string (e.g., 'risk', 'monitoring',
                'benchmarking', 'supply_chain', 'portfolio', 'audit',
                'geospatial', 'regulatory', 'compliance', 'grievance').

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
            f"TemplateRegistry(pack='PACK-007', templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )

    def __len__(self) -> int:
        """Return number of registered templates."""
        return self.template_count

    def __contains__(self, name: str) -> bool:
        """Check if template name is registered."""
        return self.has_template(name)

    def __iter__(self):
        """Iterate over template names."""
        return iter(self.list_template_names())
