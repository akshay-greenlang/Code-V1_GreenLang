# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Report Template Registry
=======================================================

Template registry and generators for EUDR compliance reporting.
Each template generates structured output in Markdown, HTML, and
JSON formats with full SHA-256 provenance tracking and EUDR-aligned
data models per Regulation (EU) 2023/1115.

Templates:
    - DDSStandardReport: Standard Due Diligence Statement (Annex II)
    - DDSSimplifiedReport: Simplified DDS for low-risk countries (Article 13)
    - ComplianceDashboard: Real-time EUDR compliance KPI dashboard
    - SupplierRiskReport: Per-supplier risk assessment report
    - CountryRiskMatrix: Country x commodity risk matrix visualization
    - GeolocationReport: Plot geolocation verification report
    - ExecutiveSummary: Board-level EUDR compliance overview

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# DDS Standard Report imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.dds_standard_report import (
    DDSStandardReport,
    DDSStandardInput,
    DDSStatus,
    CommodityType as DDS_CommodityType,
    BenchmarkClassification,
    RiskClassification,
    DDType,
    ChainOfCustodyModel,
    CertificationScheme,
    OperatorInfo,
    ProductDescription,
    QuantityInfo,
    CountryOfProduction,
    GeolocationCoordinate,
    GeolocationData,
    SupplierInfo,
    SupplyChainSummary,
    RiskScoreBreakdown,
    RiskAssessmentSummary,
    MitigationMeasure,
    EvidenceItem,
)

# ---------------------------------------------------------------------------
# DDS Simplified Report imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.dds_simplified_report import (
    DDSSimplifiedReport,
    DDSSimplifiedInput,
    SimplifiedOperatorInfo,
    SimplifiedProduct,
    SimplifiedQuantity,
    SimplifiedCountryInfo,
    SimplifiedSupplierInfo,
    SimplifiedSupplyChain,
    SimplifiedEvidenceItem,
)

# ---------------------------------------------------------------------------
# Compliance Dashboard imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.compliance_dashboard import (
    ComplianceDashboard,
    ComplianceDashboardInput,
    ComplianceOverview,
    CommodityComplianceEntry,
    RiskDistributionEntry,
    SupplierStatusSummary,
    GeolocationCoverage,
    CertificationEntry,
    DataQualityMetric,
    UpcomingDeadline as DashboardDeadline,
    RecentActivity,
    ComplianceTrafficLight,
    RiskLevel as DashboardRiskLevel,
    EventType,
)

# ---------------------------------------------------------------------------
# Supplier Risk Report imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.supplier_risk_report import (
    SupplierRiskReport,
    SupplierRiskReportInput,
    SupplierProfile,
    RiskScoreSummary,
    CountryRiskDetail,
    SupplierRiskDetail,
    CommodityRiskEntry,
    DocumentRiskDetail,
    GeolocationSummary,
    DDStatusDetail,
    Recommendation,
    RiskTrend,
    DDStatus,
    RecommendationPriority,
)

# ---------------------------------------------------------------------------
# Country Risk Matrix imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.country_risk_matrix import (
    CountryRiskMatrix,
    CountryRiskMatrixInput,
    MatrixCell,
    CountryProfile,
    CommodityHotspot,
    SupplierConcentration,
    TrendEntry,
    TrendDirection,
)

# ---------------------------------------------------------------------------
# Geolocation Report imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.geolocation_report import (
    GeolocationReport,
    GeolocationReportInput,
    ValidationSummary,
    CoordinateQualityEntry,
    PolygonAnalysisEntry,
    OverlapEntry,
    CountryDetermination,
    PlotSizeCompliance,
    CutoffDateEntry,
    GeoJSONFeature,
    MapData,
    ValidationStatus,
    PrecisionGrade,
    PolygonValidity,
    DeforestationStatus,
    ComplianceResult,
)

# ---------------------------------------------------------------------------
# Executive Summary imports
# ---------------------------------------------------------------------------
from packs.eu_compliance.PACK_006_eudr_starter.templates.executive_summary import (
    ExecutiveSummary,
    ExecutiveSummaryInput,
    HeadlineScore,
    KeyMetric,
    KeyMetricsSummary,
    CommodityStatusEntry,
    RiskExposureItem,
    FinancialImpact,
    ActionItem,
    RegulatoryMilestone,
    QuarterlyComparison,
    ReadinessGrade,
    TrafficLight,
    ActionPriority,
)


logger = logging.getLogger(__name__)

PACK_ID = "PACK-006-eudr-starter"
REGISTRY_VERSION = "1.0.0"


# =============================================================================
# ENUMS
# =============================================================================

class OutputFormat(str, Enum):
    """Supported output formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


# =============================================================================
# TEMPLATE METADATA
# =============================================================================

class TemplateInfo(BaseModel):
    """Metadata about a registered template."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    version: str = Field(..., description="Template version")
    supported_formats: List[str] = Field(
        ..., description="Supported output formats"
    )
    input_model_name: str = Field(
        ..., description="Pydantic input model class name"
    )
    template_class_name: str = Field(
        ..., description="Template class name"
    )
    eudr_articles: List[str] = Field(
        default_factory=list, description="Referenced EUDR articles"
    )
    sections_count: int = Field(0, ge=0, description="Number of report sections")


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

class TemplateRegistry:
    """Registry for all EUDR report templates.

    Provides centralized access to all 7 EUDR report templates with
    discovery, metadata inspection, and rendering capabilities.

    Templates are registered at initialization and can be accessed by
    name or enumerated with full metadata.

    Example:
        >>> registry = TemplateRegistry()
        >>> templates = registry.list_templates()
        >>> report = registry.render("dds_standard_report", data, "markdown")
    """

    # Template class registry: name -> (template_class, input_model_class, metadata)
    _TEMPLATE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
        "dds_standard_report": {
            "template_class": DDSStandardReport,
            "input_model": DDSStandardInput,
            "description": (
                "Standard Due Diligence Statement per EUDR Annex II. "
                "Primary compliance document for operators."
            ),
            "version": "1.0.0",
            "eudr_articles": [
                "Article 4", "Article 9", "Article 10", "Annex II"
            ],
            "sections_count": 14,
        },
        "dds_simplified_report": {
            "template_class": DDSSimplifiedReport,
            "input_model": DDSSimplifiedInput,
            "description": (
                "Simplified DDS for products from low-risk countries "
                "per Article 13. Reduced geolocation and risk requirements."
            ),
            "version": "1.0.0",
            "eudr_articles": ["Article 13", "Article 29"],
            "sections_count": 11,
        },
        "compliance_dashboard": {
            "template_class": ComplianceDashboard,
            "input_model": ComplianceDashboardInput,
            "description": (
                "Real-time EUDR compliance KPI dashboard with 9 widgets "
                "covering scores, commodities, risk, suppliers, and deadlines."
            ),
            "version": "1.0.0",
            "eudr_articles": [],
            "sections_count": 9,
        },
        "supplier_risk_report": {
            "template_class": SupplierRiskReport,
            "input_model": SupplierRiskReportInput,
            "description": (
                "Per-supplier risk assessment report with composite scores, "
                "trend analysis, and prioritized recommendations."
            ),
            "version": "1.0.0",
            "eudr_articles": ["Article 10"],
            "sections_count": 9,
        },
        "country_risk_matrix": {
            "template_class": CountryRiskMatrix,
            "input_model": CountryRiskMatrixInput,
            "description": (
                "Country x commodity risk matrix with color-coded risk levels, "
                "country profiles, hotspot analysis, and trend tracking."
            ),
            "version": "1.0.0",
            "eudr_articles": ["Article 29"],
            "sections_count": 6,
        },
        "geolocation_report": {
            "template_class": GeolocationReport,
            "input_model": GeolocationReportInput,
            "description": (
                "Plot geolocation verification report covering coordinate "
                "quality, polygon analysis, overlap detection, and cutoff "
                "date verification with GeoJSON output."
            ),
            "version": "1.0.0",
            "eudr_articles": ["Article 9"],
            "sections_count": 8,
        },
        "executive_summary": {
            "template_class": ExecutiveSummary,
            "input_model": ExecutiveSummaryInput,
            "description": (
                "Board-level EUDR compliance overview with readiness grade, "
                "key metrics, financial impact, and action items."
            ),
            "version": "1.0.0",
            "eudr_articles": [],
            "sections_count": 8,
        },
    }

    SUPPORTED_FORMATS = [OutputFormat.MARKDOWN, OutputFormat.HTML, OutputFormat.JSON]

    def __init__(self) -> None:
        """Initialize the template registry."""
        self._instances: Dict[str, Any] = {}
        logger.info(
            "TemplateRegistry initialized with %d templates",
            len(self._TEMPLATE_DEFINITIONS),
        )

    # --------------------------------------------------------------------- #
    # PUBLIC API
    # --------------------------------------------------------------------- #

    def list_templates(self) -> List[TemplateInfo]:
        """List all registered templates with metadata.

        Returns:
            List of TemplateInfo objects describing each template.

        Example:
            >>> registry = TemplateRegistry()
            >>> for t in registry.list_templates():
            ...     print(f"{t.name}: {t.description}")
        """
        result = []
        for name, defn in self._TEMPLATE_DEFINITIONS.items():
            result.append(
                TemplateInfo(
                    name=name,
                    description=defn["description"],
                    version=defn["version"],
                    supported_formats=[f.value for f in self.SUPPORTED_FORMATS],
                    input_model_name=defn["input_model"].__name__,
                    template_class_name=defn["template_class"].__name__,
                    eudr_articles=defn.get("eudr_articles", []),
                    sections_count=defn.get("sections_count", 0),
                )
            )
        return result

    def get_template(self, name: str) -> Any:
        """Get a template instance by name.

        Args:
            name: Template name (e.g. "dds_standard_report").

        Returns:
            Template instance with render_markdown/render_html/render_json.

        Raises:
            ValueError: If template name is not registered.

        Example:
            >>> registry = TemplateRegistry()
            >>> template = registry.get_template("dds_standard_report")
        """
        if name not in self._TEMPLATE_DEFINITIONS:
            available = ", ".join(sorted(self._TEMPLATE_DEFINITIONS.keys()))
            raise ValueError(
                f"Unknown template '{name}'. Available: {available}"
            )

        # Lazy instantiation with caching
        if name not in self._instances:
            template_class = self._TEMPLATE_DEFINITIONS[name]["template_class"]
            self._instances[name] = template_class()
            logger.debug("Instantiated template: %s", name)

        return self._instances[name]

    def get_input_model(self, name: str) -> Type[BaseModel]:
        """Get the Pydantic input model class for a template.

        Args:
            name: Template name.

        Returns:
            Pydantic BaseModel subclass for the template input.

        Raises:
            ValueError: If template name is not registered.
        """
        if name not in self._TEMPLATE_DEFINITIONS:
            available = ", ".join(sorted(self._TEMPLATE_DEFINITIONS.keys()))
            raise ValueError(
                f"Unknown template '{name}'. Available: {available}"
            )
        return self._TEMPLATE_DEFINITIONS[name]["input_model"]

    def get_template_info(self, name: str) -> TemplateInfo:
        """Get metadata for a specific template.

        Args:
            name: Template name.

        Returns:
            TemplateInfo with full metadata.

        Raises:
            ValueError: If template name is not registered.
        """
        if name not in self._TEMPLATE_DEFINITIONS:
            available = ", ".join(sorted(self._TEMPLATE_DEFINITIONS.keys()))
            raise ValueError(
                f"Unknown template '{name}'. Available: {available}"
            )
        defn = self._TEMPLATE_DEFINITIONS[name]
        return TemplateInfo(
            name=name,
            description=defn["description"],
            version=defn["version"],
            supported_formats=[f.value for f in self.SUPPORTED_FORMATS],
            input_model_name=defn["input_model"].__name__,
            template_class_name=defn["template_class"].__name__,
            eudr_articles=defn.get("eudr_articles", []),
            sections_count=defn.get("sections_count", 0),
        )

    def render(
        self,
        template_name: str,
        data: BaseModel,
        output_format: str = "markdown",
    ) -> Union[str, Dict[str, Any]]:
        """Render a template with the given data and format.

        Args:
            template_name: Name of the template to render.
            data: Validated input data (must match template's input model).
            output_format: Output format - "markdown", "html", or "json".

        Returns:
            Rendered output (string for markdown/html, dict for json).

        Raises:
            ValueError: If template name or format is invalid.
            TypeError: If data type does not match expected input model.

        Example:
            >>> registry = TemplateRegistry()
            >>> data = DDSStandardInput(...)
            >>> md = registry.render("dds_standard_report", data, "markdown")
            >>> html = registry.render("dds_standard_report", data, "html")
            >>> payload = registry.render("dds_standard_report", data, "json")
        """
        # Validate template name
        if template_name not in self._TEMPLATE_DEFINITIONS:
            available = ", ".join(sorted(self._TEMPLATE_DEFINITIONS.keys()))
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {available}"
            )

        # Validate format
        fmt = output_format.lower()
        valid_formats = {f.value for f in self.SUPPORTED_FORMATS}
        if fmt not in valid_formats:
            raise ValueError(
                f"Unsupported format '{output_format}'. "
                f"Valid formats: {', '.join(sorted(valid_formats))}"
            )

        # Validate data type
        expected_model = self._TEMPLATE_DEFINITIONS[template_name]["input_model"]
        if not isinstance(data, expected_model):
            raise TypeError(
                f"Expected {expected_model.__name__} for template "
                f"'{template_name}', got {type(data).__name__}"
            )

        # Get template and render
        template = self.get_template(template_name)

        if fmt == OutputFormat.MARKDOWN.value:
            return template.render_markdown(data)
        elif fmt == OutputFormat.HTML.value:
            return template.render_html(data)
        elif fmt == OutputFormat.JSON.value:
            return template.render_json(data)
        else:
            raise ValueError(f"Unhandled format: {fmt}")

    def template_names(self) -> List[str]:
        """Get sorted list of all registered template names.

        Returns:
            List of template name strings.
        """
        return sorted(self._TEMPLATE_DEFINITIONS.keys())

    @property
    def template_count(self) -> int:
        """Number of registered templates."""
        return len(self._TEMPLATE_DEFINITIONS)


# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

__all__ = [
    # Registry
    "TemplateRegistry",
    "TemplateInfo",
    "OutputFormat",
    # DDS Standard Report
    "DDSStandardReport",
    "DDSStandardInput",
    "DDSStatus",
    "BenchmarkClassification",
    "RiskClassification",
    "DDType",
    "ChainOfCustodyModel",
    "CertificationScheme",
    "OperatorInfo",
    "ProductDescription",
    "QuantityInfo",
    "CountryOfProduction",
    "GeolocationCoordinate",
    "GeolocationData",
    "SupplierInfo",
    "SupplyChainSummary",
    "RiskScoreBreakdown",
    "RiskAssessmentSummary",
    "MitigationMeasure",
    "EvidenceItem",
    # DDS Simplified Report
    "DDSSimplifiedReport",
    "DDSSimplifiedInput",
    "SimplifiedOperatorInfo",
    "SimplifiedProduct",
    "SimplifiedQuantity",
    "SimplifiedCountryInfo",
    "SimplifiedSupplierInfo",
    "SimplifiedSupplyChain",
    "SimplifiedEvidenceItem",
    # Compliance Dashboard
    "ComplianceDashboard",
    "ComplianceDashboardInput",
    "ComplianceOverview",
    "CommodityComplianceEntry",
    "RiskDistributionEntry",
    "SupplierStatusSummary",
    "GeolocationCoverage",
    "CertificationEntry",
    "DataQualityMetric",
    "DashboardDeadline",
    "RecentActivity",
    "ComplianceTrafficLight",
    "EventType",
    # Supplier Risk Report
    "SupplierRiskReport",
    "SupplierRiskReportInput",
    "SupplierProfile",
    "RiskScoreSummary",
    "CountryRiskDetail",
    "SupplierRiskDetail",
    "CommodityRiskEntry",
    "DocumentRiskDetail",
    "GeolocationSummary",
    "DDStatusDetail",
    "Recommendation",
    "RiskTrend",
    "DDStatus",
    "RecommendationPriority",
    # Country Risk Matrix
    "CountryRiskMatrix",
    "CountryRiskMatrixInput",
    "MatrixCell",
    "CountryProfile",
    "CommodityHotspot",
    "SupplierConcentration",
    "TrendEntry",
    "TrendDirection",
    # Geolocation Report
    "GeolocationReport",
    "GeolocationReportInput",
    "ValidationSummary",
    "CoordinateQualityEntry",
    "PolygonAnalysisEntry",
    "OverlapEntry",
    "CountryDetermination",
    "PlotSizeCompliance",
    "CutoffDateEntry",
    "GeoJSONFeature",
    "MapData",
    "ValidationStatus",
    "PrecisionGrade",
    "PolygonValidity",
    "DeforestationStatus",
    "ComplianceResult",
    # Executive Summary
    "ExecutiveSummary",
    "ExecutiveSummaryInput",
    "HeadlineScore",
    "KeyMetric",
    "KeyMetricsSummary",
    "CommodityStatusEntry",
    "RiskExposureItem",
    "FinancialImpact",
    "ActionItem",
    "RegulatoryMilestone",
    "QuarterlyComparison",
    "ReadinessGrade",
    "TrafficLight",
    "ActionPriority",
]
