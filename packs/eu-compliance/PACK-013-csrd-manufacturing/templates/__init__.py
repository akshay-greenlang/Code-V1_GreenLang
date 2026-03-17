"""
PACK-013 CSRD Manufacturing Pack - Templates module.

Provides eight report templates for CSRD manufacturing compliance
reporting. Each template supports markdown, HTML, and JSON output formats.

Templates:
    1. ProcessEmissionsReportTemplate - Process emissions breakdown
    2. ProductPCFLabelTemplate - Product carbon footprint label (DPP)
    3. EnergyPerformanceReportTemplate - Energy performance dashboard
    4. CircularEconomyReportTemplate - Circular economy metrics
    5. BATComplianceReportTemplate - BAT/IED compliance assessment
    6. WaterPollutionReportTemplate - Water & pollution disclosure
    7. ManufacturingScorecardTemplate - Sustainability scorecard
    8. DecarbonizationRoadmapTemplate - Decarbonization pathway
"""

from typing import Any, Dict, List, Optional, Type, Union

from .process_emissions_report import ProcessEmissionsReportTemplate, ProcessEmissionsReportData
from .product_pcf_label import ProductPCFLabelTemplate, ProductPCFLabelData
from .energy_performance_report import EnergyPerformanceReportTemplate, EnergyPerformanceReportData
from .circular_economy_report import CircularEconomyReportTemplate, CircularEconomyReportData
from .bat_compliance_report import BATComplianceReportTemplate, BATComplianceReportData
from .water_pollution_report import WaterPollutionReportTemplate, WaterPollutionReportData
from .manufacturing_scorecard import ManufacturingScorecardTemplate, ManufacturingScorecardData
from .decarbonization_roadmap import DecarbonizationRoadmapTemplate, DecarbonizationRoadmapData


# Type alias for any template class in this pack
TemplateClass = Union[
    Type[ProcessEmissionsReportTemplate],
    Type[ProductPCFLabelTemplate],
    Type[EnergyPerformanceReportTemplate],
    Type[CircularEconomyReportTemplate],
    Type[BATComplianceReportTemplate],
    Type[WaterPollutionReportTemplate],
    Type[ManufacturingScorecardTemplate],
    Type[DecarbonizationRoadmapTemplate],
]

# Supported output formats
SUPPORTED_FORMATS = ("markdown", "html", "json")

# Mapping of template keys to their classes and metadata
TEMPLATE_CATALOG: Dict[str, Dict[str, Any]] = {
    "process_emissions_report": {
        "class": ProcessEmissionsReportTemplate,
        "display_name": "Process Emissions Breakdown Report",
        "description": (
            "Manufacturing process emissions breakdown with facility summary, "
            "process line details, sub-sector comparison, CBAM embedded "
            "emissions, ETS benchmark, and abatement tracking."
        ),
        "category": "emissions",
        "scope": "scope1",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Scope 1 Summary", "Facility Breakdown", "Process Lines",
            "CBAM Embedded Emissions", "ETS Benchmark", "Abatement Tracking",
        ],
    },
    "product_pcf_label": {
        "class": ProductPCFLabelTemplate,
        "display_name": "Product Carbon Footprint Label",
        "description": (
            "Product carbon footprint label with total PCF, lifecycle breakdown, "
            "BOM hotspots, DPP QR code data, and ISO 14067 compliance statement."
        ),
        "category": "product",
        "scope": "lifecycle",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Product Info", "Total PCF", "Lifecycle Breakdown",
            "BOM Hotspots", "DPP Data", "ISO 14067 Compliance",
        ],
    },
    "energy_performance_report": {
        "class": EnergyPerformanceReportTemplate,
        "display_name": "Energy Performance Dashboard",
        "description": (
            "Energy performance dashboard with total energy, SEC by product, "
            "energy mix, benchmark comparison, EED compliance, decarbonization "
            "opportunities, and ISO 50001 status."
        ),
        "category": "energy",
        "scope": "facility",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Energy Summary", "Energy Mix", "SEC by Product",
            "Benchmark Comparison", "EED Compliance",
            "Decarbonization Opportunities", "ISO 50001 Status",
        ],
    },
    "circular_economy_report": {
        "class": CircularEconomyReportTemplate,
        "display_name": "Circular Economy Metrics Report",
        "description": (
            "Circular economy metrics report with material flows, MCI score, "
            "recycled content, waste hierarchy, EPR compliance, CRM tracking, "
            "and product recyclability assessment."
        ),
        "category": "circular",
        "scope": "esrs_e5",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Key Metrics", "Material Flows", "Waste Hierarchy",
            "EPR Compliance", "Critical Raw Materials", "Product Recyclability",
        ],
    },
    "bat_compliance_report": {
        "class": BATComplianceReportTemplate,
        "display_name": "BAT Compliance Assessment",
        "description": (
            "BAT compliance assessment with facility overview, parameter "
            "compliance table, gap analysis, transformation plan timeline, "
            "investment requirements, and penalty risk assessment."
        ),
        "category": "regulatory",
        "scope": "ied_bref",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Applicable BREFs", "Parameter Compliance",
            "Gap Analysis", "Transformation Plan",
            "Investment Requirements", "Penalty Risk",
        ],
    },
    "water_pollution_report": {
        "class": WaterPollutionReportTemplate,
        "display_name": "Water & Pollution Disclosure",
        "description": (
            "Water and pollution disclosure with water balance, water stress "
            "map, pollutant inventory, IED compliance, REACH SVHC tracking, "
            "and ESRS E2/E3 metrics."
        ),
        "category": "environmental",
        "scope": "esrs_e2_e3",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Water Balance", "Water Stress Assessment",
            "Pollutant Inventory", "IED Compliance",
            "REACH SVHC Tracking", "ESRS E2/E3 Metrics",
        ],
    },
    "manufacturing_scorecard": {
        "class": ManufacturingScorecardTemplate,
        "display_name": "Manufacturing Sustainability Scorecard",
        "description": (
            "Manufacturing sustainability scorecard with KPI dashboard, "
            "peer percentile rankings, SBTi alignment status, trajectory "
            "analysis, improvement priorities, and OEE sustainability overlay."
        ),
        "category": "analytics",
        "scope": "comprehensive",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "KPI Dashboard", "Peer Rankings", "SBTi Alignment",
            "Trajectory Analysis", "Improvement Priorities", "OEE Overlay",
        ],
    },
    "decarbonization_roadmap": {
        "class": DecarbonizationRoadmapTemplate,
        "display_name": "Decarbonization Pathway Report",
        "description": (
            "Decarbonization pathway report with baseline chart, technology "
            "options (MAC curve), investment timeline, annual milestones, "
            "SBTi gap analysis, and cost-benefit analysis."
        ),
        "category": "strategy",
        "scope": "decarbonization",
        "version": "1.0",
        "supported_formats": ["markdown", "html", "json"],
        "sections": [
            "Emissions Pathway", "Technology Options",
            "Investment Summary", "SBTi Alignment",
            "Cost-Benefit Analysis",
        ],
    },
}


class TemplateRegistry:
    """
    Registry for discovering, instantiating, and rendering PACK-013 templates.

    Provides a centralized catalog of all 8 CSRD Manufacturing Pack
    templates with metadata for programmatic discovery, filtering,
    instantiation, and a unified render() method.

    Example:
        >>> registry = TemplateRegistry()
        >>> names = registry.list_templates()
        >>> template = registry.get_template("process_emissions_report")
        >>> markdown = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize TemplateRegistry with optional config."""
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
        """Return number of registered templates."""
        return len(self._catalog)

    @property
    def pack_id(self) -> str:
        """Return the pack identifier."""
        return "PACK-013"

    @property
    def pack_name(self) -> str:
        """Return the pack display name."""
        return "CSRD Manufacturing"


# Module-level exports
__all__ = [
    # Template classes
    "ProcessEmissionsReportTemplate",
    "ProductPCFLabelTemplate",
    "EnergyPerformanceReportTemplate",
    "CircularEconomyReportTemplate",
    "BATComplianceReportTemplate",
    "WaterPollutionReportTemplate",
    "ManufacturingScorecardTemplate",
    "DecarbonizationRoadmapTemplate",
    # Data models
    "ProcessEmissionsReportData",
    "ProductPCFLabelData",
    "EnergyPerformanceReportData",
    "CircularEconomyReportData",
    "BATComplianceReportData",
    "WaterPollutionReportData",
    "ManufacturingScorecardData",
    "DecarbonizationRoadmapData",
    # Registry
    "TemplateRegistry",
    "TEMPLATE_CATALOG",
    "SUPPORTED_FORMATS",
]
