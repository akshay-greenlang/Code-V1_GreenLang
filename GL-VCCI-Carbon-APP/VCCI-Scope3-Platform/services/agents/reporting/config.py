"""
Scope3ReportingAgent Configuration
GL-VCCI Scope 3 Platform

Configuration settings for reporting agent.

Version: 1.0.0
Phase: 3 (Weeks 16-18)
Date: 2025-10-30
"""

from enum import Enum
from typing import Dict, Any


class ReportStandard(str, Enum):
    """Supported reporting standards."""
    ESRS_E1 = "esrs_e1"
    CDP = "cdp"
    IFRS_S2 = "ifrs_s2"
    ISO_14083 = "iso_14083"


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    HTML = "html"


class ChartType(str, Enum):
    """Chart types for visualizations."""
    PIE = "pie"
    BAR = "bar"
    LINE = "line"
    WATERFALL = "waterfall"
    HEATMAP = "heatmap"
    PARETO = "pareto"


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    STRICT = "strict"  # All checks must pass
    STANDARD = "standard"  # Core checks must pass
    LENIENT = "lenient"  # Warnings only


# ============================================================================
# QUALITY THRESHOLDS
# ============================================================================

QUALITY_THRESHOLDS = {
    "min_dqi_score": 70.0,  # Minimum average DQI score
    "min_scope_coverage": 0.80,  # 80% of Scope 3 categories
    "max_uncertainty": 0.30,  # 30% uncertainty threshold
    "min_data_completeness": 0.90,  # 90% data completeness
}


# ============================================================================
# ESRS E1 CONFIGURATION
# ============================================================================

ESRS_E1_CONFIG = {
    "required_disclosures": [
        "E1-1",  # Transition plan
        "E1-2",  # Policies
        "E1-3",  # Actions and resources
        "E1-4",  # Targets
        "E1-5",  # Energy consumption
        "E1-6",  # GHG emissions
        "E1-7",  # GHG removals
        "E1-8",  # Carbon pricing
        "E1-9",  # Financial effects
    ],
    "required_tables": [
        "ghg_emissions_by_scope",
        "scope3_by_category",
        "energy_consumption",
        "intensity_metrics",
        "year_over_year",
    ],
    "required_metrics": [
        "scope1_tco2e",
        "scope2_location_tco2e",
        "scope2_market_tco2e",
        "scope3_tco2e",
        "total_energy_mwh",
        "renewable_energy_pct",
        "intensity_per_revenue",
        "intensity_per_fte",
    ],
}


# ============================================================================
# CDP CONFIGURATION
# ============================================================================

CDP_CONFIG = {
    "version": "2024",
    "auto_population_target": 0.90,  # 90% auto-population
    "key_sections": ["C0", "C6", "C8", "C9", "C12"],
    "required_categories": [1, 4, 6],  # Minimum Scope 3 categories
    "optional_categories": [2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
}


# ============================================================================
# IFRS S2 CONFIGURATION
# ============================================================================

IFRS_S2_CONFIG = {
    "pillars": ["governance", "strategy", "risk_management", "metrics_targets"],
    "cross_industry_metrics": [
        "scope1_tco2e",
        "scope2_tco2e",
        "scope3_tco2e",
        "carbon_intensity",
    ],
    "required_sections": [
        "climate_risks",
        "climate_opportunities",
        "financial_impact",
        "resilience_analysis",
    ],
}


# ============================================================================
# ISO 14083 CONFIGURATION
# ============================================================================

ISO_14083_CONFIG = {
    "conformance_level": "full",
    "transport_modes": ["road", "rail", "sea", "air"],
    "required_elements": [
        "methodology_declaration",
        "emission_factor_sources",
        "data_quality_assessment",
        "uncertainty_quantification",
        "variance_confirmation",
    ],
    "calculation_standard": "ISO 14083:2023",
}


# ============================================================================
# CHART CONFIGURATION
# ============================================================================

CHART_CONFIG = {
    "default_style": "seaborn-v0_8-darkgrid",
    "figure_size": (12, 8),
    "dpi": 300,
    "colors": {
        "scope1": "#FF6B6B",
        "scope2": "#4ECDC4",
        "scope3": "#45B7D1",
        "category1": "#F7DC6F",
        "category4": "#BB8FCE",
        "category6": "#85C1E2",
    },
    "font": {
        "family": "sans-serif",
        "size": 11,
        "title_size": 14,
    },
}


# ============================================================================
# PDF CONFIGURATION
# ============================================================================

PDF_CONFIG = {
    "page_size": "A4",
    "margins": {
        "top": 2.5,
        "bottom": 2.5,
        "left": 2.5,
        "right": 2.5,
    },
    "header_height": 1.5,
    "footer_height": 1.0,
    "font_family": "Helvetica",
    "base_font_size": 11,
}


# ============================================================================
# EXCEL CONFIGURATION
# ============================================================================

EXCEL_CONFIG = {
    "sheets": {
        "summary": "Executive Summary",
        "scope1": "Scope 1",
        "scope2": "Scope 2",
        "scope3": "Scope 3",
        "categories": "Scope 3 Categories",
        "suppliers": "Top Suppliers",
        "data_quality": "Data Quality",
        "methodology": "Methodology",
    },
    "formatting": {
        "header_bg": "#4472C4",
        "header_font": "#FFFFFF",
        "total_bg": "#FFD966",
        "warning_bg": "#FFC7CE",
    },
}


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_CONFIG: Dict[str, Any] = {
    "agent_name": "Scope3ReportingAgent",
    "version": "1.0.0",
    "validation_level": ValidationLevel.STANDARD,
    "enable_charts": True,
    "enable_uncertainty": True,
    "enable_provenance": True,
    "quality_thresholds": QUALITY_THRESHOLDS,
    "esrs_e1": ESRS_E1_CONFIG,
    "cdp": CDP_CONFIG,
    "ifrs_s2": IFRS_S2_CONFIG,
    "iso_14083": ISO_14083_CONFIG,
    "charts": CHART_CONFIG,
    "pdf": PDF_CONFIG,
    "excel": EXCEL_CONFIG,
}


__all__ = [
    "ReportStandard",
    "ExportFormat",
    "ChartType",
    "ValidationLevel",
    "QUALITY_THRESHOLDS",
    "ESRS_E1_CONFIG",
    "CDP_CONFIG",
    "IFRS_S2_CONFIG",
    "ISO_14083_CONFIG",
    "CHART_CONFIG",
    "PDF_CONFIG",
    "EXCEL_CONFIG",
    "DEFAULT_CONFIG",
]
