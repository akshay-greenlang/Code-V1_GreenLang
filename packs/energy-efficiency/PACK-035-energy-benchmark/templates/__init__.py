# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark Pack - Report Templates
=====================================================

This package provides 10 energy benchmarking report templates for the
PACK-035 Energy Benchmark Pack. Each template supports three rendering
formats: Markdown, HTML (with inline CSS), and JSON. All templates
include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. EUIBenchmarkReportTemplate               - EUI calculation and benchmark comparison
    2. PeerComparisonReportTemplate             - Peer group ranking and percentile analysis
    3. SectorBenchmarkReportTemplate            - Sector-specific benchmark comparison
    4. EnergyPerformanceCertificateTemplate     - EPC/DEC rating report (EPBD compliant)
    5. PortfolioDashboardTemplate               - Multi-facility portfolio dashboard
    6. GapAnalysisReportTemplate                - End-use disaggregated gap analysis
    7. TargetTrackingReportTemplate             - Target trajectory and progress tracking
    8. RegulatoryComplianceReportTemplate       - EPBD/EED/MEES/LL97 compliance summary
    9. TrendAnalysisReportTemplate              - CUSUM and SPC trend analysis charts
   10. ExecutiveSummaryReportTemplate           - C-suite executive benchmark summary

Usage:
    >>> from packs.energy_efficiency.PACK_035_energy_benchmark.templates import (
    ...     TemplateRegistry,
    ...     EUIBenchmarkReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("eui_benchmark_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 35.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .eui_benchmark_report import EUIBenchmarkReportTemplate
from .peer_comparison_report import PeerComparisonReportTemplate
from .sector_benchmark_report import SectorBenchmarkReportTemplate
from .energy_performance_certificate import EnergyPerformanceCertificateTemplate
from .portfolio_dashboard import PortfolioDashboardTemplate
from .gap_analysis_report import GapAnalysisReportTemplate
from .target_tracking_report import TargetTrackingReportTemplate
from .regulatory_compliance_report import RegulatoryComplianceReportTemplate
from .trend_analysis_report import TrendAnalysisReportTemplate
from .executive_summary_report import ExecutiveSummaryReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "EUIBenchmarkReportTemplate",
    "PeerComparisonReportTemplate",
    "SectorBenchmarkReportTemplate",
    "EnergyPerformanceCertificateTemplate",
    "PortfolioDashboardTemplate",
    "GapAnalysisReportTemplate",
    "TargetTrackingReportTemplate",
    "RegulatoryComplianceReportTemplate",
    "TrendAnalysisReportTemplate",
    "ExecutiveSummaryReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[EUIBenchmarkReportTemplate],
    Type[PeerComparisonReportTemplate],
    Type[SectorBenchmarkReportTemplate],
    Type[EnergyPerformanceCertificateTemplate],
    Type[PortfolioDashboardTemplate],
    Type[GapAnalysisReportTemplate],
    Type[TargetTrackingReportTemplate],
    Type[RegulatoryComplianceReportTemplate],
    Type[TrendAnalysisReportTemplate],
    Type[ExecutiveSummaryReportTemplate],
]

TemplateInstance = Union[
    EUIBenchmarkReportTemplate,
    PeerComparisonReportTemplate,
    SectorBenchmarkReportTemplate,
    EnergyPerformanceCertificateTemplate,
    PortfolioDashboardTemplate,
    GapAnalysisReportTemplate,
    TargetTrackingReportTemplate,
    RegulatoryComplianceReportTemplate,
    TrendAnalysisReportTemplate,
    ExecutiveSummaryReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "eui_benchmark_report",
        "class": EUIBenchmarkReportTemplate,
        "description": (
            "Energy Use Intensity benchmark report with site/source/primary EUI "
            "calculations, accounting boundary breakdown, weather-normalised "
            "consumption, benchmark comparison against published standards, "
            "and year-over-year trend analysis."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "peer_comparison_report",
        "class": PeerComparisonReportTemplate,
        "description": (
            "Peer group comparison report with percentile ranking against "
            "ENERGY STAR, CIBSE TM46, DIN V 18599, and BPIE datasets. "
            "Includes peer group definition, statistical distribution, "
            "gap-to-median and gap-to-best-practice analysis."
        ),
        "category": "comparison",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "sector_benchmark_report",
        "class": SectorBenchmarkReportTemplate,
        "description": (
            "Sector-specific benchmark comparison report with building type "
            "classification, sub-sector breakdown, climate-adjusted benchmarks, "
            "and regulatory thresholds by sector including office, retail, "
            "healthcare, education, and industrial."
        ),
        "category": "benchmark",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "energy_performance_certificate",
        "class": EnergyPerformanceCertificateTemplate,
        "description": (
            "EPBD-compliant Energy Performance Certificate (EPC) and Display "
            "Energy Certificate (DEC) report with A-G rating, primary energy "
            "consumption, CO2 emissions, recommendations for improvement, "
            "and MEPS compliance status."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "portfolio_dashboard",
        "class": PortfolioDashboardTemplate,
        "description": (
            "Multi-facility portfolio dashboard with cross-site EUI ranking, "
            "top/bottom performer identification, portfolio-level aggregation, "
            "improvement trajectory, outlier flagging, and investment "
            "prioritisation by savings potential."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "gap_analysis_report",
        "class": GapAnalysisReportTemplate,
        "description": (
            "End-use disaggregated gap analysis report with consumption "
            "breakdown by end use (lighting, HVAC, process, plug loads, "
            "domestic hot water), gap quantification against best-practice "
            "benchmarks, and prioritised savings opportunities."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "target_tracking_report",
        "class": TargetTrackingReportTemplate,
        "description": (
            "Target trajectory and progress tracking report with baseline "
            "definition, target pathway (peer-based, absolute, SBTi-aligned), "
            "actual vs. target comparison, milestone status, and forecast "
            "to target achievement date."
        ),
        "category": "tracking",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "regulatory_compliance_report",
        "class": RegulatoryComplianceReportTemplate,
        "description": (
            "Regulatory compliance summary covering EPBD (EPC/DEC/MEPS/ZEB), "
            "EED Article 8/11 benchmarking obligations, UK MEES, NYC LL97, "
            "NABERS requirements, and deadline tracking with gap analysis "
            "and remediation recommendations."
        ),
        "category": "regulatory",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "trend_analysis_report",
        "class": TrendAnalysisReportTemplate,
        "description": (
            "CUSUM and SPC trend analysis report with cumulative sum charts, "
            "statistical process control rule violations, seasonal "
            "decomposition, trend direction assessment, and automated "
            "alert history with root cause annotations."
        ),
        "category": "analysis",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
    {
        "name": "executive_summary_report",
        "class": ExecutiveSummaryReportTemplate,
        "description": (
            "C-suite executive benchmark summary with key performance "
            "indicators, peer positioning, regulatory status, target "
            "progress, top risks, and recommended strategic actions. "
            "Designed for board-level reporting with clear visualisations."
        ),
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "35.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-035 Energy Benchmark report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 energy benchmark report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("eui_benchmark_report")
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
            name: Template name (e.g., 'eui_benchmark_report').
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
            category: Category string (e.g., 'benchmark', 'comparison',
                      'regulatory', 'dashboard', 'analysis', 'tracking',
                      'executive').

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
