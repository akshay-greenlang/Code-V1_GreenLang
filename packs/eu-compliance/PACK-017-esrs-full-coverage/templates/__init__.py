# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - Report Templates
========================================================

This package provides 12 ESRS disclosure report templates for the
PACK-017 ESRS Full Coverage Pack, spanning all topical standards:
General Disclosures (ESRS 2), Environment (E1-E5), Social (S1-S4),
Governance (G1), plus two cross-cutting reports (Annual Report and
Compliance Scorecard). Each template supports three rendering formats:
Markdown, HTML (with inline CSS), and JSON. All templates include
SHA-256 provenance hashing for audit trail integrity.

Templates:
    1.  ESRS2GeneralReportTemplate    - ESRS 2 general disclosures
    2.  E2PollutionReportTemplate     - E2 pollution disclosures
    3.  E3WaterReportTemplate         - E3 water and marine resources
    4.  E4BiodiversityReportTemplate  - E4 biodiversity and ecosystems
    5.  E5CircularReport              - E5 resource use and circular economy
    6.  S1WorkforceReport             - S1 own workforce
    7.  S2ValueChainReport            - S2 workers in the value chain
    8.  S3CommunitiesReport           - S3 affected communities
    9.  S4ConsumersReport             - S4 consumers and end-users
    10. G1GovernanceReport            - G1 business conduct
    11. ESRSAnnualReport              - Comprehensive annual report
    12. ESRSComplianceScorecard       - Compliance dashboard

Usage:
    >>> from packs.eu_compliance.PACK_017_esrs_full_coverage.templates import (
    ...     TemplateRegistry,
    ...     ESRSAnnualReport,
    ... )
    >>> registry = TemplateRegistry()
    >>> template = registry.get("esrs_annual_report")
    >>> md = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)

    >>> # List all available templates
    >>> for info in registry.list_templates():
    ...     print(info["name"], "-", info["description"])

Author: GreenLang Team
Version: 17.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template Imports with try/except for graceful degradation
# ---------------------------------------------------------------------------

try:
    from .esrs2_general_report import (
        ESRS2GeneralReportTemplate,
    )
except ImportError:
    ESRS2GeneralReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ESRS2GeneralReportTemplate")

try:
    from .e2_pollution_report import (
        E2PollutionReportTemplate,
    )
except ImportError:
    E2PollutionReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import E2PollutionReportTemplate")

try:
    from .e3_water_report import (
        E3WaterReportTemplate,
    )
except ImportError:
    E3WaterReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import E3WaterReportTemplate")

try:
    from .e4_biodiversity_report import (
        E4BiodiversityReportTemplate,
    )
except ImportError:
    E4BiodiversityReportTemplate = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import E4BiodiversityReportTemplate")

try:
    from .e5_circular_economy_report import (
        E5CircularReport,
    )
except ImportError:
    E5CircularReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import E5CircularReport")

try:
    from .s1_workforce_report import (
        S1WorkforceReport,
    )
except ImportError:
    S1WorkforceReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import S1WorkforceReport")

try:
    from .s2_value_chain_report import (
        S2ValueChainReport,
    )
except ImportError:
    S2ValueChainReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import S2ValueChainReport")

try:
    from .s3_communities_report import (
        S3CommunitiesReport,
    )
except ImportError:
    S3CommunitiesReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import S3CommunitiesReport")

try:
    from .s4_consumers_report import (
        S4ConsumersReport,
    )
except ImportError:
    S4ConsumersReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import S4ConsumersReport")

try:
    from .g1_governance_report import (
        G1GovernanceReport,
    )
except ImportError:
    G1GovernanceReport = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import G1GovernanceReport")

try:
    from .esrs_annual_statement import (
        ESRSAnnualStatement,
    )
except ImportError:
    ESRSAnnualStatement = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ESRSAnnualStatement")

try:
    from .esrs_scorecard_report import (
        ESRSScorecard,
    )
except ImportError:
    ESRSScorecard = None  # type: ignore[assignment,misc]
    logger.warning("Failed to import ESRSScorecard")


__all__ = [
    # Template classes
    "ESRS2GeneralReportTemplate",
    "E2PollutionReportTemplate",
    "E3WaterReportTemplate",
    "E4BiodiversityReportTemplate",
    "E5CircularReport",
    "S1WorkforceReport",
    "S2ValueChainReport",
    "S3CommunitiesReport",
    "S4ConsumersReport",
    "G1GovernanceReport",
    "ESRSAnnualStatement",
    "ESRSScorecard",
    # Registry
    "TemplateRegistry",
    # Constants
    "TEMPLATE_CATALOG",
]


# Template type aliases
TemplateClass = Union[
    Type["ESRS2GeneralReportTemplate"],
    Type["E2PollutionReportTemplate"],
    Type["E3WaterReportTemplate"],
    Type["E4BiodiversityReportTemplate"],
    Type["E5CircularReport"],
    Type["S1WorkforceReport"],
    Type["S2ValueChainReport"],
    Type["S3CommunitiesReport"],
    Type["S4ConsumersReport"],
    Type["G1GovernanceReport"],
    Type["ESRSAnnualStatement"],
    Type["ESRSScorecard"],
]

TemplateInstance = Union[
    "ESRS2GeneralReportTemplate",
    "E2PollutionReportTemplate",
    "E3WaterReportTemplate",
    "E4BiodiversityReportTemplate",
    "E5CircularReport",
    "S1WorkforceReport",
    "S2ValueChainReport",
    "S3CommunitiesReport",
    "S4ConsumersReport",
    "G1GovernanceReport",
    "ESRSAnnualStatement",
    "ESRSScorecard",
]


# =============================================================================
# TEMPLATE CATALOG
# =============================================================================

TEMPLATE_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "esrs2_general_report",
        "class": ESRS2GeneralReportTemplate,
        "description": (
            "ESRS 2 general disclosures report covering governance structure "
            "(GOV-1 to GOV-5), strategy and business model (SBM-1 to SBM-3), "
            "impact/risk/opportunity identification (IRO-1, IRO-2), and "
            "overall disclosure coverage mapping."
        ),
        "category": "general",
        "esrs_reference": "ESRS 2",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "e2_pollution_report",
        "class": E2PollutionReportTemplate,
        "description": (
            "ESRS E2 pollution report covering policies (E2-1), actions (E2-2), "
            "targets (E2-3), emissions to air/water/soil (E2-4), substances "
            "of concern and SVHC (E2-5), and financial effects (E2-6)."
        ),
        "category": "environment",
        "esrs_reference": "E2",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "e3_water_report",
        "class": E3WaterReportTemplate,
        "description": (
            "ESRS E3 water and marine resources report covering policies "
            "(E3-1), actions (E3-2), targets (E3-3), water balance and stress "
            "areas (E3-4), and financial effects (E3-5)."
        ),
        "category": "environment",
        "esrs_reference": "E3",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "e4_biodiversity_report",
        "class": E4BiodiversityReportTemplate,
        "description": (
            "ESRS E4 biodiversity and ecosystems report covering transition "
            "plan (E4-1), policies (E4-2), actions (E4-3), targets (E4-4), "
            "impact metrics including land use and species (E4-5), and "
            "financial effects (E4-6)."
        ),
        "category": "environment",
        "esrs_reference": "E4",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "e5_circular_economy_report",
        "class": E5CircularReport,
        "description": (
            "ESRS E5 resource use and circular economy report covering "
            "policies (E5-1), actions (E5-2), targets (E5-3), resource "
            "inflows (E5-4), resource outflows and waste (E5-5), and "
            "financial effects (E5-6)."
        ),
        "category": "environment",
        "esrs_reference": "E5",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "s1_workforce_report",
        "class": S1WorkforceReport,
        "description": (
            "ESRS S1 own workforce report covering policies and engagement "
            "(S1-1 to S1-4), demographics (S1-6 to S1-8), health and safety "
            "(S1-14), training (S1-13), diversity and pay (S1-9, S1-16), "
            "work-life balance (S1-15), and incidents (S1-17)."
        ),
        "category": "social",
        "esrs_reference": "S1",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "s2_value_chain_report",
        "class": S2ValueChainReport,
        "description": (
            "ESRS S2 workers in the value chain report covering policies "
            "(S2-1), engagement processes (S2-2), remediation (S2-3), "
            "targets (S2-4), and impacts with actions (S2-5)."
        ),
        "category": "social",
        "esrs_reference": "S2",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "s3_communities_report",
        "class": S3CommunitiesReport,
        "description": (
            "ESRS S3 affected communities report covering policies (S3-1), "
            "engagement processes (S3-2), remediation (S3-3), targets "
            "(S3-4), and material impacts with actions (S3-5)."
        ),
        "category": "social",
        "esrs_reference": "S3",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "s4_consumers_report",
        "class": S4ConsumersReport,
        "description": (
            "ESRS S4 consumers and end-users report covering policies "
            "(S4-1), engagement (S4-2), remediation (S4-3), targets "
            "(S4-4), and material impacts with actions (S4-5)."
        ),
        "category": "social",
        "esrs_reference": "S4",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "g1_governance_report",
        "class": G1GovernanceReport,
        "description": (
            "ESRS G1 business conduct report covering corporate culture "
            "(G1-1), supplier management (G1-2), anti-corruption (G1-3), "
            "confirmed incidents (G1-4), political influence (G1-5), "
            "and payment practices (G1-6)."
        ),
        "category": "governance",
        "esrs_reference": "G1",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "esrs_annual_statement",
        "class": ESRSAnnualStatement,
        "description": (
            "Comprehensive ESRS annual sustainability statement combining all "
            "standards (ESRS 2, E1-E5, S1-S4, G1) into one integrated "
            "document with general information, environment, social, "
            "governance sections, cross-cutting metrics, and assurance readiness."
        ),
        "category": "comprehensive",
        "esrs_reference": "All ESRS",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
    {
        "name": "esrs_scorecard_report",
        "class": ESRSScorecard,
        "description": (
            "ESRS compliance scorecard providing overall compliance percentage, "
            "per-standard scores, gap analysis, materiality matrix, "
            "cross-standard consistency checks, and prioritized improvement "
            "actions for achieving full ESRS compliance."
        ),
        "category": "compliance",
        "esrs_reference": "All ESRS",
        "formats": ["markdown", "html", "json"],
        "version": "17.0.0",
    },
]


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================


class TemplateRegistry:
    """
    Registry for PACK-017 ESRS Full Coverage report templates.

    Provides centralized discovery, instantiation, and management of
    all 12 ESRS report templates. Templates can be listed, filtered
    by category or ESRS reference, retrieved by name, and rendered
    in markdown/HTML/JSON.

    Attributes:
        _templates: Internal mapping of template names to metadata.
        _instances: Cache of instantiated template objects.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> assert len(names) == 12
        >>> template = registry.get("esrs_annual_report")
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
            category, esrs_reference, formats, and version.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
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
            name: Template name (e.g., 'esrs_annual_report').
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
            "esrs_reference": defn["esrs_reference"],
            "formats": defn["formats"],
            "version": defn["version"],
            "class_name": defn["class"].__name__,
        }

    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by category.

        Args:
            category: Category string (e.g., 'general', 'environment',
                      'social', 'governance', 'comprehensive', 'compliance').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["category"] == category and defn["class"] is not None
        ]

    def get_by_esrs_reference(self, esrs_ref: str) -> List[Dict[str, Any]]:
        """
        Get templates filtered by ESRS reference.

        Args:
            esrs_ref: ESRS reference (e.g., 'E2', 'S1', 'All ESRS').

        Returns:
            List of matching template info dicts.
        """
        return [
            {
                "name": defn["name"],
                "description": defn["description"],
                "category": defn["category"],
                "esrs_reference": defn["esrs_reference"],
                "formats": defn["formats"],
                "version": defn["version"],
            }
            for defn in TEMPLATE_CATALOG
            if defn["esrs_reference"] == esrs_ref and defn["class"] is not None
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
