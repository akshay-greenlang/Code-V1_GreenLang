# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Report Templates
========================================================

This package provides 8 report templates for the PACK-020 Battery Passport
Prep Pack, covering all key compliance areas of the EU Battery Regulation
(EU) 2023/1542. Each template supports three rendering formats: Markdown,
HTML (with inline CSS), and JSON. All templates include SHA-256 provenance
hashing for audit trail integrity.

Templates:
    1. CarbonFootprintDeclarationTemplate  - Art 7 carbon footprint declaration
    2. RecycledContentReportTemplate       - Art 8 recycled content report
    3. BatteryPassportReportTemplate       - Annex XIII full battery passport
    4. PerformanceReportTemplate           - Annex IV performance & durability
    5. DueDiligenceReportTemplate          - Art 48 supply chain due diligence
    6. LabellingComplianceReportTemplate   - Art 13-14 labelling compliance
    7. EndOfLifeReportTemplate             - Art 56-71 end-of-life management
    8. BatteryRegulationScorecardTemplate  - Executive compliance scorecard

Usage:
    >>> from packs.eu_compliance.PACK_020_battery_passport_prep.templates import (
    ...     TemplateRegistry,
    ...     CarbonFootprintDeclarationTemplate,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("carbon_footprint_declaration")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 20.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.carbon_footprint_declaration import (
        CarbonFootprintDeclarationTemplate,
    )
except ImportError:
    CarbonFootprintDeclarationTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import CarbonFootprintDeclarationTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.recycled_content_report import (
        RecycledContentReportTemplate,
    )
except ImportError:
    RecycledContentReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import RecycledContentReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.battery_passport_report import (
        BatteryPassportReportTemplate,
    )
except ImportError:
    BatteryPassportReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import BatteryPassportReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.performance_report import (
        PerformanceReportTemplate,
    )
except ImportError:
    PerformanceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import PerformanceReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.due_diligence_report import (
        DueDiligenceReportTemplate,
    )
except ImportError:
    DueDiligenceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import DueDiligenceReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.labelling_compliance_report import (
        LabellingComplianceReportTemplate,
    )
except ImportError:
    LabellingComplianceReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import LabellingComplianceReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.end_of_life_report import (
        EndOfLifeReportTemplate,
    )
except ImportError:
    EndOfLifeReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import EndOfLifeReportTemplate")

try:
    from packs.eu_compliance.PACK_020_battery_passport_prep.templates.battery_regulation_scorecard import (
        BatteryRegulationScorecardTemplate,
    )
except ImportError:
    BatteryRegulationScorecardTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import BatteryRegulationScorecardTemplate")


__all__ = [
    # Template classes
    "CarbonFootprintDeclarationTemplate",
    "RecycledContentReportTemplate",
    "BatteryPassportReportTemplate",
    "PerformanceReportTemplate",
    "DueDiligenceReportTemplate",
    "LabellingComplianceReportTemplate",
    "EndOfLifeReportTemplate",
    "BatteryRegulationScorecardTemplate",
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
        "name": "carbon_footprint_declaration",
        "class": CarbonFootprintDeclarationTemplate,
        "description": (
            "Carbon footprint declaration with lifecycle breakdown, performance "
            "class assignment, methodology reference, and threshold compliance "
            "per Article 7 of EU Battery Regulation 2023/1542."
        ),
        "category": "carbon_footprint",
        "regulation_reference": "Art 7",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "recycled_content_report",
        "class": RecycledContentReportTemplate,
        "description": (
            "Recycled content report with per-material inventory, target tracking "
            "against 2031/2036 milestones, phase compliance status, and sourcing "
            "recommendations per Article 8."
        ),
        "category": "recycled_content",
        "regulation_reference": "Art 8",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "battery_passport_report",
        "class": BatteryPassportReportTemplate,
        "description": (
            "Full battery passport covering general info, carbon footprint, "
            "supply chain DD, material composition, performance, end-of-life, "
            "and QR code data per Annex XIII."
        ),
        "category": "battery_passport",
        "regulation_reference": "Art 77-78, Annex XIII",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "performance_report",
        "class": PerformanceReportTemplate,
        "description": (
            "Performance and durability report with capacity, voltage, power, "
            "cycle life, efficiency, SoH, SoC, and overall durability rating "
            "per Article 10 and Annex IV."
        ),
        "category": "performance",
        "regulation_reference": "Art 10, Annex IV",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "due_diligence_report",
        "class": DueDiligenceReportTemplate,
        "description": (
            "Supply chain due diligence findings with supplier assessment, "
            "risk mapping, OECD 5-step compliance, audit coverage, and "
            "mitigation measures per Article 48."
        ),
        "category": "due_diligence",
        "regulation_reference": "Art 48-52",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "labelling_compliance_report",
        "class": LabellingComplianceReportTemplate,
        "description": (
            "Labelling compliance assessment with 20-element checklist, "
            "per-category status, missing element analysis, and corrective "
            "actions per Articles 13-14."
        ),
        "category": "labelling",
        "regulation_reference": "Art 13-14",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "end_of_life_report",
        "class": EndOfLifeReportTemplate,
        "description": (
            "End-of-life management report with collection rates, recycling "
            "efficiency, material recovery rates, and second-life assessment "
            "per Articles 56-71 and Annex XII."
        ),
        "category": "end_of_life",
        "regulation_reference": "Art 56-71, Annex XII",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
    {
        "name": "battery_regulation_scorecard",
        "class": BatteryRegulationScorecardTemplate,
        "description": (
            "Executive compliance dashboard scorecard with overall readiness "
            "score, article-by-article traffic lights, key metrics, regulatory "
            "timeline, and prioritized recommendations."
        ),
        "category": "scorecard",
        "regulation_reference": "EU Battery Regulation 2023/1542 (all articles)",
        "formats": ["markdown", "html", "json"],
        "version": "20.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-020 Battery Passport Prep Pack report templates.

    Provides centralized discovery, instantiation, and management of
    all 8 battery regulation compliance report templates. Templates can
    be listed, filtered by category, retrieved by name, and rendered
    in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 8
        >>> template = registry.get("carbon_footprint_declaration")
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
            category, regulation_reference, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "regulation_reference": defn["regulation_reference"],
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
            name: Template name (e.g., 'carbon_footprint_declaration').
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
            "regulation_reference": defn["regulation_reference"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string (e.g., 'carbon_footprint',
                      'recycled_content', 'battery_passport', 'performance',
                      'due_diligence', 'labelling', 'end_of_life', 'scorecard').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "regulation_reference": defn["regulation_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["category"] == category and defn["class"] is not None
        ]

    def get_by_regulation_reference(self, ref: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by regulation reference.

        Args:
            ref: Regulation reference (e.g., 'Art 7', 'Annex XIII').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "regulation_reference": defn["regulation_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if ref in defn["regulation_reference"] and defn["class"] is not None
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
