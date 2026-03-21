# -*- coding: utf-8 -*-
"""
PACK-001 CSRD Starter Pack - Report Template Generators
========================================================

Phase 3 report templates for CSRD compliance reporting. Each template
generates structured output in Markdown, HTML, and JSON formats with
full provenance tracking and ESRS-aligned data models.

Templates:
    - ExecutiveSummaryTemplate: Board-level 2-page CSRD summary
    - ESRSDisclosureTemplate: Full ESRS disclosure narrative (12 standards)
    - MaterialityMatrixTemplate: Double materiality assessment report
    - GHGEmissionsReportTemplate: Comprehensive GHG emissions report
    - AuditorPackageTemplate: External auditor evidence package
    - ComplianceDashboardTemplate: Real-time compliance dashboard data

Author: GreenLang Team
Version: 1.0.0
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Template 1: Executive Summary
# ---------------------------------------------------------------------------
_TEMPLATE_1_SYMBOLS: list[str] = [
    "ExecutiveSummaryTemplate",
    "ExecutiveSummaryInput",
    "ComplianceStatusEntry",
    "KeyMetricsDashboard",
    "MaterialTopicSummary",
    "RegulatoryDeadline",
    "RiskHeatmapEntry",
    "ActionItem",
]

try:
    from .executive_summary import (  # noqa: F401
        ExecutiveSummaryTemplate,
        ExecutiveSummaryInput,
        ComplianceStatusEntry,
        KeyMetricsDashboard,
        MaterialTopicSummary,
        RegulatoryDeadline,
        RiskHeatmapEntry,
        ActionItem,
    )
except ImportError as e:
    logger.debug("Template 1 (ExecutiveSummaryTemplate) not available: %s", e)
    _TEMPLATE_1_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 2: ESRS Disclosure
# ---------------------------------------------------------------------------
_TEMPLATE_2_SYMBOLS: list[str] = [
    "ESRSDisclosureTemplate",
    "ESRSDisclosureInput",
    "StandardDisclosure",
    "DisclosureRequirement",
    "MetricValue",
    "CrossReference",
    "DataQualityIndicator",
]

try:
    from .esrs_disclosure import (  # noqa: F401
        ESRSDisclosureTemplate,
        ESRSDisclosureInput,
        StandardDisclosure,
        DisclosureRequirement,
        MetricValue,
        CrossReference,
        DataQualityIndicator,
    )
except ImportError as e:
    logger.debug("Template 2 (ESRSDisclosureTemplate) not available: %s", e)
    _TEMPLATE_2_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 3: Materiality Matrix
# ---------------------------------------------------------------------------
_TEMPLATE_3_SYMBOLS: list[str] = [
    "MaterialityMatrixTemplate",
    "MaterialityMatrixInput",
    "MaterialTopic",
    "ImpactMaterialityScores",
    "FinancialMaterialityScores",
    "MatrixDataPoint",
    "StakeholderEngagement",
]

try:
    from .materiality_matrix import (  # noqa: F401
        MaterialityMatrixTemplate,
        MaterialityMatrixInput,
        MaterialTopic,
        ImpactMaterialityScores,
        FinancialMaterialityScores,
        MatrixDataPoint,
        StakeholderEngagement,
    )
except ImportError as e:
    logger.debug("Template 3 (MaterialityMatrixTemplate) not available: %s", e)
    _TEMPLATE_3_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 4: GHG Emissions Report
# ---------------------------------------------------------------------------
_TEMPLATE_4_SYMBOLS: list[str] = [
    "GHGEmissionsReportTemplate",
    "GHGEmissionsInput",
    "Scope1Breakdown",
    "Scope2Breakdown",
    "Scope3Category",
    "IntensityMetric",
    "EmissionTrend",
    "MethodologyReference",
]

try:
    from .ghg_emissions_report import (  # noqa: F401
        GHGEmissionsReportTemplate,
        GHGEmissionsInput,
        Scope1Breakdown,
        Scope2Breakdown,
        Scope3Category,
        IntensityMetric,
        EmissionTrend,
        MethodologyReference,
    )
except ImportError as e:
    logger.debug("Template 4 (GHGEmissionsReportTemplate) not available: %s", e)
    _TEMPLATE_4_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 5: Auditor Package
# ---------------------------------------------------------------------------
_TEMPLATE_5_SYMBOLS: list[str] = [
    "AuditorPackageTemplate",
    "AuditorPackageInput",
    "CalculationAuditEntry",
    "DataLineageRecord",
    "SourceDataReference",
    "ComplianceChecklistItem",
    "DataQualityAssessment",
]

try:
    from .auditor_package import (  # noqa: F401
        AuditorPackageTemplate,
        AuditorPackageInput,
        CalculationAuditEntry,
        DataLineageRecord,
        SourceDataReference,
        ComplianceChecklistItem,
        DataQualityAssessment,
    )
except ImportError as e:
    logger.debug("Template 5 (AuditorPackageTemplate) not available: %s", e)
    _TEMPLATE_5_SYMBOLS = []

# ---------------------------------------------------------------------------
# Template 6: Compliance Dashboard
# ---------------------------------------------------------------------------
_TEMPLATE_6_SYMBOLS: list[str] = [
    "ComplianceDashboardTemplate",
    "ComplianceDashboardInput",
    "StandardComplianceEntry",
    "DataCompletenessCell",
    "OutstandingAction",
    "ComplianceTrendPoint",
    "AlertEntry",
    "UpcomingDeadline",
]

try:
    from .compliance_dashboard import (  # noqa: F401
        ComplianceDashboardTemplate,
        ComplianceDashboardInput,
        StandardComplianceEntry,
        DataCompletenessCell,
        OutstandingAction,
        ComplianceTrendPoint,
        AlertEntry,
        UpcomingDeadline,
    )
except ImportError as e:
    logger.debug("Template 6 (ComplianceDashboardTemplate) not available: %s", e)
    _TEMPLATE_6_SYMBOLS = []


# ---------------------------------------------------------------------------
# Template catalog and registry
# ---------------------------------------------------------------------------
TEMPLATE_CATALOG: list[dict[str, Any]] = []

if _TEMPLATE_1_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "executive_summary",
        "class": ExecutiveSummaryTemplate,
        "description": "Board-level 2-page CSRD executive summary.",
        "category": "executive",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })
if _TEMPLATE_2_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "esrs_disclosure",
        "class": ESRSDisclosureTemplate,
        "description": "Full ESRS disclosure narrative across 12 standards.",
        "category": "disclosure",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })
if _TEMPLATE_3_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "materiality_matrix",
        "class": MaterialityMatrixTemplate,
        "description": "Double materiality assessment report.",
        "category": "materiality",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })
if _TEMPLATE_4_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "ghg_emissions_report",
        "class": GHGEmissionsReportTemplate,
        "description": "Comprehensive GHG emissions report (Scope 1/2/3).",
        "category": "emissions",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })
if _TEMPLATE_5_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "auditor_package",
        "class": AuditorPackageTemplate,
        "description": "External auditor evidence package.",
        "category": "audit",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })
if _TEMPLATE_6_SYMBOLS:
    TEMPLATE_CATALOG.append({
        "name": "compliance_dashboard",
        "class": ComplianceDashboardTemplate,
        "description": "Real-time compliance dashboard data.",
        "category": "dashboard",
        "formats": ["markdown", "html", "json"],
        "version": "1.0.0",
    })


class TemplateRegistry:
    """
    Registry for PACK-001 CSRD Starter Pack report templates.

    Provides a centralized way to discover, instantiate, and manage
    all report templates. Templates can be listed, retrieved by name,
    and instantiated with optional configuration.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = [t["name"] for t in registry.list_templates()]
        >>> template = registry.get("executive_summary")
    """

    def __init__(self) -> None:
        """Initialize TemplateRegistry with all template definitions."""
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._instances: Dict[str, Any] = {}

        for defn in TEMPLATE_CATALOG:
            self._templates[defn["name"]] = defn

        logger.info(
            "PACK-001 TemplateRegistry initialized with %d templates",
            len(self._templates),
        )

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates with metadata."""
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
        """List all available template names."""
        return [defn["name"] for defn in TEMPLATE_CATALOG]

    def get(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Get a template instance by name.

        Args:
            name: Template name (e.g., 'executive_summary').
            config: Optional configuration overrides.

        Returns:
            Template instance.

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

    def has_template(self, name: str) -> bool:
        """Check if a template exists by name."""
        return name in self._templates

    @property
    def template_count(self) -> int:
        """Return the number of registered templates."""
        return len(self._templates)

    def __repr__(self) -> str:
        """Return string representation of registry."""
        return (
            f"TemplateRegistry(templates={self.template_count}, "
            f"names={self.list_template_names()})"
        )


__all__: list[str] = [
    # Template Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    # Template classes and models
    *_TEMPLATE_1_SYMBOLS,
    *_TEMPLATE_2_SYMBOLS,
    *_TEMPLATE_3_SYMBOLS,
    *_TEMPLATE_4_SYMBOLS,
    *_TEMPLATE_5_SYMBOLS,
    *_TEMPLATE_6_SYMBOLS,
]
