"""
PACK-012 CSRD Financial Service Pack - Templates module.

Provides eight report templates for CSRD financial institution compliance
reporting. Each template supports markdown, HTML, and JSON output formats.

Templates:
    1. PCAFReportTemplate - PCAF financed emissions disclosure
    2. GARBTARReportTemplate - EU Taxonomy Art 8 DA GAR/BTAR
    3. Pillar3ESGTemplate - EBA Pillar 3 ESG ITS
    4. ClimateRiskReportTemplate - TCFD-aligned climate risk
    5. FSESRSChapterTemplate - FI-specific ESRS chapters
    6. FinancedEmissionsDashboard - Portfolio emissions dashboard
    7. InsuranceESGTemplate - Insurance ESG disclosure
    8. SBTiFIReportTemplate - SBTi-FI progress report
"""

from typing import Any, Dict, List, Optional, Type, Union

from .pcaf_report import PCAFReportTemplate, PCAFReportData
from .gar_btar_report import GARBTARReportTemplate, GARBTARReportData
from .pillar3_esg_template import Pillar3ESGTemplate, Pillar3ESGData
from .climate_risk_report import ClimateRiskReportTemplate, ClimateRiskReportData
from .fs_esrs_chapter import FSESRSChapterTemplate, FSESRSChapterData
from .financed_emissions_dashboard import FinancedEmissionsDashboard, FinancedEmissionsDashboardData
from .insurance_esg_template import InsuranceESGTemplate, InsuranceESGData
from .sbti_fi_report import SBTiFIReportTemplate, SBTiFIReportData


# Type alias for any template class in this pack
TemplateClass = Union[
    Type[PCAFReportTemplate],
    Type[GARBTARReportTemplate],
    Type[Pillar3ESGTemplate],
    Type[ClimateRiskReportTemplate],
    Type[FSESRSChapterTemplate],
    Type[FinancedEmissionsDashboard],
    Type[InsuranceESGTemplate],
    Type[SBTiFIReportTemplate],
]

# Supported output formats
SUPPORTED_FORMATS = ("markdown", "html", "json")

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "pcaf_report": {
        "class": PCAFReportTemplate,
        "display_name": "PCAF Financed Emissions Disclosure",
        "description": (
            "PCAF Standard financed emissions disclosure with asset class "
            "breakdown, data quality scores, YoY comparison, and sector attribution."
        ),
        "category": "regulatory",
        "scope": "financed_emissions",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Summary KPIs", "Asset Class Breakdown", "Sector Attribution",
            "Data Quality Distribution", "Year-over-Year Comparison",
        ],
    },
    "gar_btar_report": {
        "class": GARBTARReportTemplate,
        "display_name": "EU Taxonomy GAR/BTAR Disclosure",
        "description": (
            "EU Taxonomy Art 8 Delegated Act disclosure with GAR by objective, "
            "BTAR reconciliation, transitional/enabling activities, and EBA ITS tables."
        ),
        "category": "regulatory",
        "scope": "taxonomy",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Key Performance Indicators", "GAR by Objective",
            "Taxonomy Activity Types", "Qualitative Notes",
        ],
    },
    "pillar3_esg_template": {
        "class": Pillar3ESGTemplate,
        "display_name": "EBA Pillar 3 ESG ITS",
        "description": (
            "EBA Pillar 3 ESG ITS quantitative and qualitative templates "
            "for credit institutions covering all 10 EBA templates."
        ),
        "category": "regulatory",
        "scope": "pillar3",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Template Coverage", "Key Alignment Metrics",
            "Top 20 Carbon-Intensive Exposures", "Validation Issues",
        ],
    },
    "climate_risk_report": {
        "class": ClimateRiskReportTemplate,
        "display_name": "Climate Risk Report (TCFD-aligned)",
        "description": (
            "TCFD-aligned climate risk report with physical risk heatmaps, "
            "transition risk sectors, NGFS scenario results, and stress test impacts."
        ),
        "category": "analytics",
        "scope": "climate_risk",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Risk Overview", "Scenario Analysis Results",
            "Risk Heatmap", "Sector Risk Breakdown",
            "Governance", "Strategy", "Risk Management",
        ],
    },
    "fs_esrs_chapter": {
        "class": FSESRSChapterTemplate,
        "display_name": "FI-Specific ESRS Chapters",
        "description": (
            "Financial institution-specific ESRS chapters covering E1 (financed "
            "emissions), S1-S4 (financial inclusion, responsible lending), and "
            "G1 (board governance, ESG remuneration)."
        ),
        "category": "regulatory",
        "scope": "esrs",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "E1 - Climate (FI-Specific)", "S1-S4 - Social (FI-Specific)",
            "G1 - Governance (FI-Specific)", "Material Topics",
        ],
    },
    "financed_emissions_dashboard": {
        "class": FinancedEmissionsDashboard,
        "display_name": "Financed Emissions Dashboard",
        "description": (
            "Portfolio emissions dashboard with WACI waterfall, asset class "
            "drill-down, data quality traffic light, and top emitters."
        ),
        "category": "analytics",
        "scope": "dashboard",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Key Metrics", "WACI Waterfall", "Asset Class Drill-Down",
            "Data Quality Traffic Light", "Top Emitters", "Year-over-Year",
        ],
    },
    "insurance_esg_template": {
        "class": InsuranceESGTemplate,
        "display_name": "Insurance ESG Disclosure",
        "description": (
            "Insurance-specific ESG disclosure covering underwriting emissions, "
            "Solvency II ESG integration, and responsible underwriting practices."
        ),
        "category": "regulatory",
        "scope": "insurance",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Underwriting Emissions Summary", "Emissions by Line of Business",
            "Solvency II ESG Integration", "Responsible Underwriting",
            "Exclusion Policies", "Engagement Activities",
        ],
    },
    "sbti_fi_report": {
        "class": SBTiFIReportTemplate,
        "display_name": "SBTi-FI Progress Report",
        "description": (
            "SBTi Financial Institutions progress report covering sector targets, "
            "portfolio coverage approach, temperature rating, and NZBA tracking."
        ),
        "category": "analytics",
        "scope": "sbti",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Progress Summary", "Credibility Assessment", "Sector Targets",
            "Temperature Rating", "Gaps", "Exclusion Policies", "Engagement",
        ],
    },
}


class TemplateRegistry:
    """
    Registry for discovering, instantiating, and rendering PACK-012 templates.

    Provides a centralized catalog of all 8 CSRD Financial Service Pack
    templates with metadata for programmatic discovery, filtering,
    instantiation, and a unified render() method.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("pcaf_report")
        >>> markdown = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._catalog: Dict[str, Dict[str, Any]] = TEMPLATE_CATALOG.copy()

    def render(
        self,
        template_name: str,
        data: Dict[str, Any],
        format: str = "markdown",
        config: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Dict[str, Any]]:
        """Render a template by name in the specified format."""
        if format not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        template = self.get_template(template_name, config=config)
        if format == "markdown":
            return template.render_markdown(data)
        elif format == "html":
            return template.render_html(data)
        else:
            return template.render_json(data)

    def list_templates(
        self,
        category: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all available templates with optional filtering."""
        result: List[Dict[str, Any]] = []
        for key, meta in self._catalog.items():
            if category and meta.get("category") != category:
                continue
            if scope and meta.get("scope") != scope:
                continue
            result.append({
                "key": key,
                "display_name": meta["display_name"],
                "description": meta["description"],
                "category": meta["category"],
                "scope": meta["scope"],
                "version": meta["version"],
                "supported_formats": meta.get("supported_formats", list(SUPPORTED_FORMATS)),
                "sections": meta.get("sections", []),
            })
        return result

    def get_template(
        self,
        template_key: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Instantiate and return a template by its registry key."""
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(f"Template '{template_key}' not found. Available: {available}")
        template_cls = self._catalog[template_key]["class"]
        effective_config = config if config is not None else self.config
        return template_cls(config=effective_config)

    def get_template_metadata(self, template_key: str) -> Dict[str, Any]:
        """Get metadata for a specific template."""
        if template_key not in self._catalog:
            available = ", ".join(sorted(self._catalog.keys()))
            raise KeyError(f"Template '{template_key}' not found. Available: {available}")
        meta = self._catalog[template_key]
        return {
            "key": template_key,
            "display_name": meta["display_name"],
            "description": meta["description"],
            "category": meta["category"],
            "scope": meta["scope"],
            "version": meta["version"],
            "supported_formats": meta.get("supported_formats", list(SUPPORTED_FORMATS)),
            "sections": meta.get("sections", []),
        }

    def get_all_template_keys(self) -> List[str]:
        """Get all registered template keys."""
        return sorted(self._catalog.keys())

    def has_template(self, template_key: str) -> bool:
        """Check if a template key is registered."""
        return template_key in self._catalog

    @property
    def template_count(self) -> int:
        return len(self._catalog)

    @property
    def pack_id(self) -> str:
        return "PACK-012"

    @property
    def pack_name(self) -> str:
        return "CSRD Financial Service"


# Module-level exports
__all__ = [
    # Template classes
    "PCAFReportTemplate",
    "GARBTARReportTemplate",
    "Pillar3ESGTemplate",
    "ClimateRiskReportTemplate",
    "FSESRSChapterTemplate",
    "FinancedEmissionsDashboard",
    "InsuranceESGTemplate",
    "SBTiFIReportTemplate",
    # Data models
    "PCAFReportData",
    "GARBTARReportData",
    "Pillar3ESGData",
    "ClimateRiskReportData",
    "FSESRSChapterData",
    "FinancedEmissionsDashboardData",
    "InsuranceESGData",
    "SBTiFIReportData",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "SUPPORTED_FORMATS",
]
