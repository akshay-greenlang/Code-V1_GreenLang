# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Report Templates
==========================================================

This package provides 10 industrial energy audit report templates for the
PACK-031 Industrial Energy Audit Pack. Each template supports three
rendering formats: Markdown, HTML (with inline CSS), and JSON. All
templates include SHA-256 provenance hashing for audit trail integrity.

Templates:
    1. EnergyAuditReportTemplate            - EN 16247-compliant full audit report
    2. EnergyBaselineReportTemplate          - ISO 50006/50015 baseline establishment
    3. SavingsVerificationReportTemplate     - IPMVP-compliant M&V report
    4. EnergyManagementDashboardTemplate     - Real-time dashboard with KPIs
    5. CompressedAirReportTemplate           - ISO 11011 compressed air audit
    6. SteamSystemReportTemplate             - Steam system assessment
    7. WasteHeatRecoveryReportTemplate       - Waste heat recovery feasibility
    8. EquipmentEfficiencyReportTemplate     - Equipment-level efficiency assessment
    9. RegulatoryComplianceReportTemplate    - EED/ISO/ETS compliance summary
   10. ISO50001ReviewReportTemplate          - ISO 50001 management review package

Usage:
    >>> from packs.energy_efficiency.PACK_031_industrial_energy_audit.templates import (
    ...     TemplateRegistry,
    ...     EnergyAuditReportTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("energy_audit_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 31.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

from .energy_audit_report import EnergyAuditReportTemplate
from .energy_baseline_report import EnergyBaselineReportTemplate
from .savings_verification_report import SavingsVerificationReportTemplate
from .energy_management_dashboard import EnergyManagementDashboardTemplate
from .compressed_air_report import CompressedAirReportTemplate
from .steam_system_report import SteamSystemReportTemplate
from .waste_heat_recovery_report import WasteHeatRecoveryReportTemplate
from .equipment_efficiency_report import EquipmentEfficiencyReportTemplate
from .regulatory_compliance_report import RegulatoryComplianceReportTemplate
from .iso_50001_review_report import ISO50001ReviewReportTemplate

logger = logging.getLogger(__name__)

__all__ = [
    # Template classes
    "EnergyAuditReportTemplate",
    "EnergyBaselineReportTemplate",
    "SavingsVerificationReportTemplate",
    "EnergyManagementDashboardTemplate",
    "CompressedAirReportTemplate",
    "SteamSystemReportTemplate",
    "WasteHeatRecoveryReportTemplate",
    "EquipmentEfficiencyReportTemplate",
    "RegulatoryComplianceReportTemplate",
    "ISO50001ReviewReportTemplate",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type alias
TemplateClass = Union[
    Type[EnergyAuditReportTemplate],
    Type[EnergyBaselineReportTemplate],
    Type[SavingsVerificationReportTemplate],
    Type[EnergyManagementDashboardTemplate],
    Type[CompressedAirReportTemplate],
    Type[SteamSystemReportTemplate],
    Type[WasteHeatRecoveryReportTemplate],
    Type[EquipmentEfficiencyReportTemplate],
    Type[RegulatoryComplianceReportTemplate],
    Type[ISO50001ReviewReportTemplate],
]

TemplateInstance = Union[
    EnergyAuditReportTemplate,
    EnergyBaselineReportTemplate,
    SavingsVerificationReportTemplate,
    EnergyManagementDashboardTemplate,
    CompressedAirReportTemplate,
    SteamSystemReportTemplate,
    WasteHeatRecoveryReportTemplate,
    EquipmentEfficiencyReportTemplate,
    RegulatoryComplianceReportTemplate,
    ISO50001ReviewReportTemplate,
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "energy_audit_report",
        "class": EnergyAuditReportTemplate,
        "description": (
            "EN 16247-compliant industrial energy audit report with executive "
            "summary, facility description, energy consumption analysis, end-use "
            "breakdown, findings with savings and costs, prioritized "
            "recommendations, and implementation roadmap."
        ),
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "energy_baseline_report",
        "class": EnergyBaselineReportTemplate,
        "description": (
            "ISO 50006/50015-compliant baseline establishment report with "
            "regression models, EnPI charts data, CUSUM analysis, statistical "
            "validation, degree-day normalization results, and energy balance."
        ),
        "category": "baseline",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "savings_verification_report",
        "class": SavingsVerificationReportTemplate,
        "description": (
            "IPMVP-compliant measurement and verification report with baseline "
            "data, routine/non-routine adjustments, post-implementation "
            "measurements, verified savings, confidence intervals, and cost "
            "savings analysis."
        ),
        "category": "verification",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "energy_management_dashboard",
        "class": EnergyManagementDashboardTemplate,
        "description": (
            "Real-time energy management dashboard with KPI cards, consumption "
            "trend charts, active alerts, EnPI tracking, target progress, cost "
            "and budget tracking, and weather-normalized consumption."
        ),
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "compressed_air_report",
        "class": CompressedAirReportTemplate,
        "description": (
            "ISO 11011-compliant compressed air system audit report with system "
            "inventory, specific power analysis, leak survey results, pressure "
            "profile, VSD analysis, and optimization recommendations."
        ),
        "category": "system_audit",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "steam_system_report",
        "class": SteamSystemReportTemplate,
        "description": (
            "Steam system assessment with boiler efficiency, flue gas analysis, "
            "steam trap survey results, insulation assessment, condensate "
            "recovery analysis, flash steam recovery, and CHP feasibility."
        ),
        "category": "system_audit",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "waste_heat_recovery_report",
        "class": WasteHeatRecoveryReportTemplate,
        "description": (
            "Waste heat recovery feasibility report with heat source/sink "
            "inventory, pinch analysis results, composite curve data, technology "
            "option evaluation, ROI analysis, and implementation plan."
        ),
        "category": "feasibility",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "equipment_efficiency_report",
        "class": EquipmentEfficiencyReportTemplate,
        "description": (
            "Equipment-level efficiency assessment with motor inventory (IEC "
            "60034-30-1), pump curves, compressor profiles, boiler stack losses, "
            "fan assessment, efficiency gap analysis, and upgrade recommendations."
        ),
        "category": "equipment",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "regulatory_compliance_report",
        "class": RegulatoryComplianceReportTemplate,
        "description": (
            "Energy regulatory compliance summary covering EED obligations, "
            "audit scheduling, ISO 50001 certification status, EU ETS "
            "obligations, national requirements, and deadline tracking with "
            "gap analysis."
        ),
        "category": "compliance",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
    {
        "name": "iso_50001_review_report",
        "class": ISO50001ReviewReportTemplate,
        "description": (
            "ISO 50001:2018 Clause 9.3 management review package with EnMS "
            "performance, EnPI trends, objectives and targets status, internal "
            "audit results, nonconformity tracking, energy policy review, and "
            "continual improvement evidence."
        ),
        "category": "management",
        "formats": ["markdown", "html", "json"],
        "version": "31.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-031 Industrial Energy Audit report templates.

    Provides centralized discovery, instantiation, and management of
    all 10 energy audit report templates. Templates can be listed,
    filtered by category, retrieved by name, and rendered in
    markdown/HTML/JSON formats.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 10
        >>> template = registry.get("energy_audit_report")
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
            name: Template name (e.g., 'energy_audit_report').
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
            category: Category string (e.g., 'audit', 'baseline',
                      'verification', 'dashboard', 'system_audit',
                      'feasibility', 'equipment', 'compliance',
                      'management').

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
