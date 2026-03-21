"""
PACK-030 Net Zero Reporting Pack - Configuration Manager

This module implements the NetZeroReportingConfig and PackConfig classes that
load, merge, and validate all configuration for the Net Zero Reporting Pack.
It provides comprehensive Pydantic v2 models for multi-framework report
generation (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD), data aggregation,
narrative generation, XBRL/iXBRL tagging, assurance evidence packaging,
dashboard creation, and format rendering.

Framework Architecture (7 Frameworks):
    - SBTi: Annual progress disclosure against validated targets
    - CDP: Climate Change questionnaire (C0-C12) with A-list scoring
    - TCFD: 4-pillar disclosure (Governance/Strategy/Risk/Metrics)
    - GRI: GRI 305 emissions disclosures (305-1 through 305-7)
    - ISSB: IFRS S2 climate disclosure with industry metrics
    - SEC: 10-K climate disclosure with XBRL/iXBRL tagging
    - CSRD: ESRS E1 Climate Change with digital taxonomy

Output Formats (6 Formats):
    - PDF: Executive-ready reports with branding and citations
    - HTML: Interactive dashboards with charts and drill-down
    - Excel: Structured data tables for CDP upload preparation
    - JSON: RESTful API output with OpenAPI documentation
    - XBRL: SEC climate disclosure machine-readable tags
    - iXBRL: Inline XBRL combining human and machine readability

Narrative Generation:
    - AI-assisted drafting with citation management
    - Cross-framework consistency validation (95%+ target)
    - Multi-language support (EN, DE, FR, ES)
    - Human review workflow before publication

Assurance & Audit:
    - SHA-256 provenance on all calculations
    - Data lineage diagrams (source to report)
    - ISAE 3410 evidence bundle generation
    - Control matrix and methodology documentation

Data Aggregation:
    - PACK-021: Baseline emissions and inventory
    - PACK-022: Reduction initiatives and MACC curves
    - PACK-028: Sector pathways and convergence data
    - PACK-029: Interim targets and progress monitoring
    - GL-SBTi-APP: SBTi target data and validation results
    - GL-CDP-APP: Historical CDP responses and scores
    - GL-TCFD-APP: Scenario analysis and risk assessments
    - GL-GHG-APP: GHG inventory and emission factors

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (8 presets)
    3. Environment overrides (NZ_REPORTING_* prefix)
    4. Explicit runtime overrides

Available Presets:
    - csrd_focus: CSRD ESRS E1 Focus with digital taxonomy
    - cdp_alist: CDP A-List Target with completeness scoring
    - tcfd_investor: TCFD Investor-Grade with scenario analysis
    - sbti_validation: SBTi Validation Ready with evidence bundles
    - sec_10k: SEC 10-K Compliance with XBRL/iXBRL tagging
    - multi_framework: Multi-Framework Comprehensive (all 7)
    - investor_relations: Investor Relations Package (TCFD+ISSB)
    - assurance_ready: Assurance-Ready Package (ISAE 3410)

Regulatory Context:
    - CSRD / ESRS E1 Climate Change (EU, 2024+)
    - SEC Climate Disclosure Rules (US, proposed)
    - ISSB IFRS S2 Climate-related Disclosures (Global)
    - TCFD Recommendations (2017, updated 2023)
    - CDP Climate Change Questionnaire (2024)
    - SBTi Corporate Net-Zero Standard v1.0
    - GRI 305 Emissions (2016, updated 2022)

Cross-Pack Integration:
    - PACK-021: Net Zero Starter (baseline emissions)
    - PACK-022: Net Zero Acceleration (reduction initiatives)
    - PACK-028: Sector Pathway (sector pathways, benchmarks)
    - PACK-029: Interim Targets (interim targets, progress)

Example:
    >>> config = PackConfig.from_preset("multi_framework")
    >>> print(config.pack.frameworks_enabled)
    ['SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD']
    >>> config = PackConfig.from_preset("csrd_focus", overrides={"languages": ["en", "de"]})
    >>> print(config.pack.languages)
    ['en', 'de']
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent


# =============================================================================
# Constants
# =============================================================================

DEFAULT_REPORTING_YEAR: int = 2025
DEFAULT_BASELINE_YEAR: int = 2019
DEFAULT_NET_ZERO_YEAR: int = 2050
DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS: int = 10
DEFAULT_API_RESPONSE_TIMEOUT_MS: int = 200
DEFAULT_NARRATIVE_CONSISTENCY_TARGET_PCT: float = 95.0
DEFAULT_CACHE_HIT_RATIO_TARGET_PCT: float = 95.0
DEFAULT_RETENTION_YEARS: int = 7
DEFAULT_MAX_CONCURRENT_REPORTS: int = 8
DEFAULT_PDF_RENDER_TIMEOUT_SECONDS: int = 5
DEFAULT_XBRL_RENDER_TIMEOUT_SECONDS: int = 3

# Supported reporting frameworks
SUPPORTED_FRAMEWORKS: Dict[str, Dict[str, str]] = {
    "SBTi": {
        "full_name": "Science Based Targets initiative",
        "version": "Corporate Net-Zero Standard v1.0 (2024 update)",
        "disclosure_frequency": "Annual",
        "typical_deadline": "Rolling annual",
        "output_formats": "PDF, JSON",
        "key_sections": "Target description, base year, progress table, variance explanation",
    },
    "CDP": {
        "full_name": "CDP Climate Change Questionnaire",
        "version": "2024 Questionnaire",
        "disclosure_frequency": "Annual",
        "typical_deadline": "July 31",
        "output_formats": "Excel, PDF",
        "key_sections": "C0-C12 modules, 300+ data points",
    },
    "TCFD": {
        "full_name": "Task Force on Climate-related Financial Disclosures",
        "version": "2023 Final Recommendations",
        "disclosure_frequency": "Annual",
        "typical_deadline": "Annual report filing",
        "output_formats": "PDF, HTML",
        "key_sections": "Governance, Strategy, Risk Management, Metrics & Targets",
    },
    "GRI": {
        "full_name": "Global Reporting Initiative",
        "version": "GRI 305 (2016, updated 2022)",
        "disclosure_frequency": "Annual",
        "typical_deadline": "Annual sustainability report",
        "output_formats": "PDF, HTML",
        "key_sections": "305-1 through 305-7, Content Index",
    },
    "ISSB": {
        "full_name": "ISSB IFRS S2 Climate-related Disclosures",
        "version": "IFRS S2 (2023)",
        "disclosure_frequency": "Annual",
        "typical_deadline": "Aligned with financial reporting",
        "output_formats": "PDF, XBRL",
        "key_sections": "Governance, Strategy, Risk Management, Metrics (4-pillar)",
    },
    "SEC": {
        "full_name": "SEC Climate Disclosure Rules",
        "version": "Regulation S-K Items 1502-1506 (proposed)",
        "disclosure_frequency": "Annual",
        "typical_deadline": "10-K within 90 days of fiscal year-end",
        "output_formats": "XBRL, iXBRL, PDF",
        "key_sections": "Business description, Risk factors, MD&A, Scope 1/2 emissions",
    },
    "CSRD": {
        "full_name": "Corporate Sustainability Reporting Directive",
        "version": "ESRS E1 Climate Change (2024)",
        "disclosure_frequency": "Annual",
        "typical_deadline": "Within 5 months of fiscal year-end",
        "output_formats": "PDF, digital taxonomy",
        "key_sections": "E1-1 through E1-9 disclosure requirements",
    },
}

# Output format specifications
OUTPUT_FORMAT_SPECS: Dict[str, Dict[str, Any]] = {
    "PDF": {
        "name": "PDF Document",
        "description": "Executive-ready PDF with branding, TOC, and citations",
        "mime_type": "application/pdf",
        "renderer": "WeasyPrint",
        "max_render_time_seconds": 5,
        "supports_branding": True,
        "supports_charts": True,
    },
    "HTML": {
        "name": "Interactive HTML Dashboard",
        "description": "HTML5 dashboards with JavaScript charts and drill-down",
        "mime_type": "text/html",
        "renderer": "Jinja2 + Chart.js",
        "max_render_time_seconds": 2,
        "supports_branding": True,
        "supports_charts": True,
    },
    "Excel": {
        "name": "Excel Data Tables",
        "description": "Structured Excel with pivot tables and data validation",
        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "renderer": "openpyxl",
        "max_render_time_seconds": 2,
        "supports_branding": False,
        "supports_charts": True,
    },
    "JSON": {
        "name": "JSON API Output",
        "description": "RESTful JSON with OpenAPI/Swagger documentation",
        "mime_type": "application/json",
        "renderer": "Native Python",
        "max_render_time_seconds": 1,
        "supports_branding": False,
        "supports_charts": False,
    },
    "XBRL": {
        "name": "XBRL Machine-Readable",
        "description": "SEC/CSRD taxonomy-compliant XBRL tags",
        "mime_type": "application/xbrl+xml",
        "renderer": "lxml + XBRL taxonomy",
        "max_render_time_seconds": 3,
        "supports_branding": False,
        "supports_charts": False,
    },
    "iXBRL": {
        "name": "Inline XBRL",
        "description": "Human-readable HTML with embedded XBRL machine-readable tags",
        "mime_type": "application/xhtml+xml",
        "renderer": "lxml + Jinja2",
        "max_render_time_seconds": 3,
        "supports_branding": True,
        "supports_charts": False,
    },
}

# Supported languages for narrative generation
SUPPORTED_LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {
        "name": "English",
        "locale": "en_US",
        "translation_service": "native",
    },
    "de": {
        "name": "German",
        "locale": "de_DE",
        "translation_service": "deepl",
    },
    "fr": {
        "name": "French",
        "locale": "fr_FR",
        "translation_service": "deepl",
    },
    "es": {
        "name": "Spanish",
        "locale": "es_ES",
        "translation_service": "deepl",
    },
}

# Assurance standards mapping
ASSURANCE_STANDARDS: Dict[str, Dict[str, str]] = {
    "ISAE_3410": {
        "name": "ISAE 3410 - Assurance Engagements on Greenhouse Gas Statements",
        "scope": "GHG emissions data, calculation methodology, evidence trail",
        "levels": "limited, reasonable",
        "issuing_body": "IAASB",
    },
    "ISAE_3000": {
        "name": "ISAE 3000 - Assurance Engagements Other than Audits",
        "scope": "Sustainability information, narrative disclosures",
        "levels": "limited, reasonable",
        "issuing_body": "IAASB",
    },
    "AA1000AS": {
        "name": "AA1000 Assurance Standard v3",
        "scope": "Stakeholder engagement, materiality, responsiveness",
        "levels": "moderate, high",
        "issuing_body": "AccountAbility",
    },
}

# Evidence bundle components
EVIDENCE_BUNDLE_COMPONENTS: Dict[str, Dict[str, str]] = {
    "provenance_hashes": {
        "name": "Calculation Provenance",
        "description": "SHA-256 hashes for all metric calculations",
        "format": "JSON manifest + hash files",
    },
    "lineage_diagrams": {
        "name": "Data Lineage Diagrams",
        "description": "Visual source-to-report data flow diagrams",
        "format": "SVG + JSON graph data",
    },
    "methodology_docs": {
        "name": "Methodology Documentation",
        "description": "Calculation methodology and emission factor documentation",
        "format": "PDF + structured YAML",
    },
    "control_matrix": {
        "name": "Control Matrix",
        "description": "ISAE 3410 control objectives and evidence mapping",
        "format": "Excel + JSON",
    },
    "assumption_register": {
        "name": "Assumption Register",
        "description": "All assumptions used in calculations with justifications",
        "format": "Excel + JSON",
    },
    "change_log": {
        "name": "Change Log",
        "description": "Immutable audit trail of all data and report modifications",
        "format": "JSON + database export",
    },
}

# Branding style profiles
BRANDING_STYLES: Dict[str, Dict[str, str]] = {
    "corporate": {
        "name": "Corporate Standard",
        "description": "Professional corporate style with standard layout",
        "default_primary_color": "#1E3A8A",
        "default_secondary_color": "#3B82F6",
        "font_family": "Arial, Helvetica, sans-serif",
    },
    "executive": {
        "name": "Executive Premium",
        "description": "High-end executive style with premium layout and charts",
        "default_primary_color": "#1F2937",
        "default_secondary_color": "#6B7280",
        "font_family": "Georgia, 'Times New Roman', serif",
    },
    "investor": {
        "name": "Investor Relations",
        "description": "Financial-style layout emphasizing data and metrics",
        "default_primary_color": "#0F172A",
        "default_secondary_color": "#334155",
        "font_family": "'Segoe UI', Tahoma, sans-serif",
    },
    "regulator": {
        "name": "Regulatory Compliance",
        "description": "Formal regulatory style with reference numbering",
        "default_primary_color": "#1E3A5F",
        "default_secondary_color": "#4A7C9B",
        "font_family": "'Times New Roman', Times, serif",
    },
}

# Stakeholder view types
STAKEHOLDER_VIEW_TYPES: Dict[str, Dict[str, Any]] = {
    "investor": {
        "name": "Investor View",
        "description": "TCFD + ISSB focus, financial materiality, scenario analysis",
        "primary_frameworks": ["TCFD", "ISSB"],
        "secondary_frameworks": ["CDP", "SBTi"],
        "key_metrics": ["scope_1_2_total", "scope_3_total", "intensity_metrics",
                        "target_progress", "scenario_analysis"],
    },
    "regulator": {
        "name": "Regulator View",
        "description": "CSRD + SEC focus, compliance status, audit trail",
        "primary_frameworks": ["CSRD", "SEC"],
        "secondary_frameworks": ["GRI", "TCFD"],
        "key_metrics": ["esrs_e1_datapoints", "sec_scope_1_2", "compliance_score",
                        "validation_results"],
    },
    "customer": {
        "name": "Customer View",
        "description": "Product carbon footprint, supply chain, reduction initiatives",
        "primary_frameworks": ["GRI"],
        "secondary_frameworks": ["CDP", "SBTi"],
        "key_metrics": ["product_footprint", "supply_chain_emissions",
                        "reduction_initiatives", "carbon_label"],
    },
    "employee": {
        "name": "Employee View",
        "description": "Progress toward targets, reduction achievements, engagement",
        "primary_frameworks": ["SBTi"],
        "secondary_frameworks": ["TCFD"],
        "key_metrics": ["target_progress_pct", "annual_reduction", "initiative_impacts",
                        "facility_performance"],
    },
    "board": {
        "name": "Board & C-Suite View",
        "description": "Executive dashboard with all frameworks, risk overview",
        "primary_frameworks": ["TCFD", "CSRD", "SBTi"],
        "secondary_frameworks": ["CDP", "SEC", "ISSB", "GRI"],
        "key_metrics": ["framework_coverage_heatmap", "deadline_countdown",
                        "consistency_score", "assurance_status"],
    },
}

# Data source pack/app identifiers
DATA_SOURCE_PACKS: Dict[str, Dict[str, str]] = {
    "PACK-021": {
        "name": "Net Zero Starter Pack",
        "data_type": "Baseline emissions, GHG inventory, activity data",
        "integration_method": "REST API / database views",
    },
    "PACK-022": {
        "name": "Net Zero Acceleration Pack",
        "data_type": "Reduction initiatives, MACC curves, abatement costs",
        "integration_method": "REST API / database views",
    },
    "PACK-028": {
        "name": "Sector Pathway Pack",
        "data_type": "Sector pathways, convergence benchmarks, technology roadmaps",
        "integration_method": "REST API / database views",
    },
    "PACK-029": {
        "name": "Interim Targets Pack",
        "data_type": "Interim targets, progress monitoring, variance analysis",
        "integration_method": "REST API / database views",
    },
}

DATA_SOURCE_APPS: Dict[str, Dict[str, str]] = {
    "GL-SBTi-APP": {
        "name": "GreenLang SBTi Application",
        "data_type": "SBTi target data, validation results, submission history",
        "integration_method": "GraphQL API (OAuth 2.0)",
    },
    "GL-CDP-APP": {
        "name": "GreenLang CDP Application",
        "data_type": "Historical CDP responses, scores, peer benchmarks",
        "integration_method": "GraphQL API (OAuth 2.0)",
    },
    "GL-TCFD-APP": {
        "name": "GreenLang TCFD Application",
        "data_type": "Scenario analysis, climate risks, opportunities",
        "integration_method": "GraphQL API (OAuth 2.0)",
    },
    "GL-GHG-APP": {
        "name": "GreenLang GHG Application",
        "data_type": "GHG inventory, emission factors, activity data",
        "integration_method": "GraphQL API (OAuth 2.0)",
    },
}

# Framework deadline templates (month/day for typical reporting years)
FRAMEWORK_DEADLINE_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "SBTi": {
        "deadline_description": "Annual rolling disclosure",
        "typical_month": None,
        "notification_days": [90, 60, 30, 14, 7],
    },
    "CDP": {
        "deadline_description": "July 31 annual submission",
        "typical_month": 7,
        "typical_day": 31,
        "notification_days": [120, 90, 60, 30, 14, 7],
    },
    "TCFD": {
        "deadline_description": "Annual report filing deadline",
        "typical_month": None,
        "notification_days": [90, 60, 30, 14, 7],
    },
    "GRI": {
        "deadline_description": "Annual sustainability report publication",
        "typical_month": None,
        "notification_days": [90, 60, 30, 14],
    },
    "ISSB": {
        "deadline_description": "Aligned with financial reporting period",
        "typical_month": None,
        "notification_days": [90, 60, 30, 14, 7],
    },
    "SEC": {
        "deadline_description": "10-K filing within 90 days of fiscal year-end",
        "typical_month": None,
        "filing_window_days": 90,
        "notification_days": [90, 60, 30, 14, 7, 3],
    },
    "CSRD": {
        "deadline_description": "Within 5 months of fiscal year-end",
        "typical_month": None,
        "filing_window_months": 5,
        "notification_days": [120, 90, 60, 30, 14, 7],
    },
}

# XBRL taxonomy specifications
XBRL_TAXONOMY_SPECS: Dict[str, Dict[str, str]] = {
    "SEC": {
        "taxonomy_name": "SEC Climate Disclosure Taxonomy",
        "namespace": "https://xbrl.sec.gov/climate/2024",
        "schema_url": "https://xbrl.sec.gov/climate/2024/climate-2024.xsd",
        "version": "2024",
    },
    "CSRD": {
        "taxonomy_name": "ESRS XBRL Taxonomy",
        "namespace": "https://xbrl.efrag.org/esrs/2024",
        "schema_url": "https://xbrl.efrag.org/esrs/2024/esrs-2024.xsd",
        "version": "2024",
    },
    "ISSB": {
        "taxonomy_name": "IFRS S2 XBRL Taxonomy",
        "namespace": "https://xbrl.ifrs.org/taxonomy/2024/ifrs-s2",
        "schema_url": "https://xbrl.ifrs.org/taxonomy/2024/ifrs-s2.xsd",
        "version": "2024",
    },
}

# Consistency validation rule categories
CONSISTENCY_RULE_CATEGORIES: Dict[str, Dict[str, str]] = {
    "metric_alignment": {
        "name": "Metric Alignment",
        "description": "Same metric values reported consistently across frameworks",
    },
    "narrative_consistency": {
        "name": "Narrative Consistency",
        "description": "Qualitative statements do not contradict across frameworks",
    },
    "temporal_alignment": {
        "name": "Temporal Alignment",
        "description": "Reporting periods and base years are consistent",
    },
    "scope_boundary": {
        "name": "Scope Boundary Alignment",
        "description": "Organizational and operational boundaries match",
    },
    "methodology_consistency": {
        "name": "Methodology Consistency",
        "description": "Calculation methods described consistently",
    },
    "target_alignment": {
        "name": "Target Alignment",
        "description": "Targets and progress described consistently across frameworks",
    },
}

# Supported preset configurations
SUPPORTED_PRESETS: Dict[str, str] = {
    "csrd_focus": "CSRD ESRS E1 focus with digital taxonomy and EU compliance",
    "cdp_alist": "CDP A-List target with completeness scoring and narrative quality",
    "tcfd_investor": "TCFD investor-grade disclosure with scenario analysis",
    "sbti_validation": "SBTi validation ready with evidence bundles and 21-criteria checks",
    "sec_10k": "SEC 10-K compliance with XBRL/iXBRL tagging and attestation",
    "multi_framework": "Multi-framework comprehensive with all 7 frameworks and consistency validation",
    "investor_relations": "Investor relations package with TCFD+ISSB focus and financial metrics",
    "assurance_ready": "Assurance-ready package with full ISAE 3410 evidence bundle",
}


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: Input string to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# Enums (10 enums)
# =============================================================================


class ReportingFramework(str, Enum):
    """Supported climate disclosure reporting frameworks."""

    SBTI = "SBTi"
    CDP = "CDP"
    TCFD = "TCFD"
    GRI = "GRI"
    ISSB = "ISSB"
    SEC = "SEC"
    CSRD = "CSRD"


class OutputFormat(str, Enum):
    """Supported report output formats."""

    PDF = "PDF"
    HTML = "HTML"
    EXCEL = "Excel"
    JSON = "JSON"
    XBRL = "XBRL"
    IXBRL = "iXBRL"


class BrandingStyle(str, Enum):
    """Report branding style profiles."""

    CORPORATE = "corporate"
    EXECUTIVE = "executive"
    INVESTOR = "investor"
    REGULATOR = "regulator"


class StakeholderViewType(str, Enum):
    """Dashboard stakeholder view types."""

    INVESTOR = "investor"
    REGULATOR = "regulator"
    CUSTOMER = "customer"
    EMPLOYEE = "employee"
    BOARD = "board"


class AssuranceLevel(str, Enum):
    """External assurance engagement level."""

    NONE = "none"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class NarrativeQuality(str, Enum):
    """Narrative generation quality level."""

    STANDARD = "standard"
    HIGH = "high"
    PREMIUM = "premium"


class ConsistencyStrictness(str, Enum):
    """Cross-framework consistency validation strictness level."""

    RELAXED = "relaxed"
    STANDARD = "standard"
    STRICT = "strict"


class ReportStatus(str, Enum):
    """Report lifecycle status."""

    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    PUBLISHED = "published"


class DataSourceRequirement(str, Enum):
    """Data source availability requirement level."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"


class TranslationService(str, Enum):
    """External translation service provider."""

    NATIVE = "native"
    DEEPL = "deepl"
    GOOGLE = "google"


# =============================================================================
# Pydantic Sub-Config Models (11 models)
# =============================================================================


class FrameworkOutputConfig(BaseModel):
    """Configuration for a single framework output specification.

    Defines the output format, template, and rendering options for
    a specific framework report generation.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    format: str = Field(
        "PDF",
        description="Output format for this framework report",
    )
    template: str = Field(
        "",
        description="Template name for rendering this framework report",
    )
    enabled: bool = Field(
        True,
        description="Whether this output is enabled",
    )


class FrameworkConfig(BaseModel):
    """Configuration for framework selection and prioritization.

    Defines which frameworks are enabled, their priority order,
    and which are primary vs secondary for reporting focus.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    frameworks_enabled: List[str] = Field(
        default_factory=lambda: ["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"],
        description="List of reporting frameworks enabled for report generation",
    )
    primary_framework: Optional[str] = Field(
        None,
        description="Primary framework for reporting focus (drives layout and priority)",
    )
    secondary_frameworks: List[str] = Field(
        default_factory=list,
        description="Secondary frameworks included in reporting suite",
    )
    framework_outputs: List[FrameworkOutputConfig] = Field(
        default_factory=list,
        description="Framework-specific output format and template configurations",
    )

    @field_validator("frameworks_enabled")
    @classmethod
    def validate_frameworks(cls, v: List[str]) -> List[str]:
        """Validate that all framework names are recognized."""
        valid = set(SUPPORTED_FRAMEWORKS.keys())
        invalid = [f for f in v if f not in valid]
        if invalid:
            logger.warning(
                "Unrecognized frameworks: %s. Valid frameworks: %s",
                invalid, sorted(valid),
            )
        return v

    @field_validator("primary_framework")
    @classmethod
    def validate_primary_framework(cls, v: Optional[str]) -> Optional[str]:
        """Validate the primary framework is recognized."""
        if v is not None and v not in SUPPORTED_FRAMEWORKS:
            logger.warning(
                "Unrecognized primary_framework: %s. Valid: %s",
                v, sorted(SUPPORTED_FRAMEWORKS.keys()),
            )
        return v


class BrandingConfig(BaseModel):
    """Configuration for report branding and visual identity.

    Defines logo, colors, fonts, and style profile for rendered reports.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    style: BrandingStyle = Field(
        BrandingStyle.CORPORATE,
        description="Branding style profile (corporate, executive, investor, regulator)",
    )
    logo_path: Optional[str] = Field(
        None,
        description="Path to organization logo file (PNG, SVG)",
    )
    primary_color: str = Field(
        "#1E3A8A",
        description="Primary brand color (hex)",
    )
    secondary_color: str = Field(
        "#3B82F6",
        description="Secondary brand color (hex)",
    )
    font_family: str = Field(
        "Arial, Helvetica, sans-serif",
        description="CSS font-family for report text",
    )
    include_charts: bool = Field(
        True,
        description="Include data visualization charts in reports",
    )
    include_toc: bool = Field(
        True,
        description="Include table of contents in PDF/HTML reports",
    )
    include_page_numbers: bool = Field(
        True,
        description="Include page numbers in PDF reports",
    )
    include_headers_footers: bool = Field(
        True,
        description="Include headers and footers in PDF reports",
    )
    custom_css: Optional[str] = Field(
        None,
        description="Custom CSS for HTML/iXBRL reports",
    )

    @field_validator("primary_color", "secondary_color")
    @classmethod
    def validate_hex_color(cls, v: str) -> str:
        """Validate hex color format."""
        if not v.startswith("#") or len(v) not in (4, 7):
            logger.warning(
                "Color '%s' may not be a valid hex color. Expected format: #RGB or #RRGGBB",
                v,
            )
        return v


class DataAggregationConfig(BaseModel):
    """Configuration for multi-source data aggregation.

    Defines which data sources are required, how reconciliation works,
    and lineage tracking settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    pack_021_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.REQUIRED,
        description="PACK-021 (baseline emissions) availability requirement",
    )
    pack_022_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.REQUIRED,
        description="PACK-022 (reduction initiatives) availability requirement",
    )
    pack_028_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.RECOMMENDED,
        description="PACK-028 (sector pathways) availability requirement",
    )
    pack_029_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.REQUIRED,
        description="PACK-029 (interim targets) availability requirement",
    )
    gl_sbti_app_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.RECOMMENDED,
        description="GL-SBTi-APP availability requirement",
    )
    gl_cdp_app_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.RECOMMENDED,
        description="GL-CDP-APP availability requirement",
    )
    gl_tcfd_app_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.RECOMMENDED,
        description="GL-TCFD-APP availability requirement",
    )
    gl_ghg_app_requirement: DataSourceRequirement = Field(
        DataSourceRequirement.RECOMMENDED,
        description="GL-GHG-APP availability requirement",
    )
    auto_reconciliation: bool = Field(
        True,
        description="Automatically reconcile data from multiple sources",
    )
    reconciliation_tolerance_pct: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Percentage tolerance for source reconciliation (e.g., rounding)",
    )
    gap_detection_enabled: bool = Field(
        True,
        description="Automatically detect and flag missing data for target frameworks",
    )
    lineage_tracking_enabled: bool = Field(
        True,
        description="Track data lineage from source system to report metric",
    )
    lineage_diagram_format: str = Field(
        "svg",
        description="Format for lineage diagrams: 'svg', 'png', or 'json'",
    )
    completeness_scoring: bool = Field(
        True,
        description="Score data completeness (0-100%) for each framework",
    )
    parallel_fetching: bool = Field(
        True,
        description="Fetch data from multiple sources in parallel",
    )
    fetch_timeout_seconds: int = Field(
        30,
        ge=5,
        le=120,
        description="Timeout for individual data source fetch operations",
    )
    retry_count: int = Field(
        3,
        ge=0,
        le=10,
        description="Number of retries for failed data source connections",
    )
    circuit_breaker_threshold: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of consecutive failures before circuit breaker opens",
    )


class NarrativeConfig(BaseModel):
    """Configuration for AI-assisted narrative generation.

    Defines narrative quality, citation management, multi-language support,
    and consistency validation settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    narrative_quality: NarrativeQuality = Field(
        NarrativeQuality.HIGH,
        description="Quality level for AI-generated narrative drafts",
    )
    citation_management_enabled: bool = Field(
        True,
        description="Link narrative claims to source data with citations",
    )
    citation_format: str = Field(
        "footnote",
        description="Citation format: 'footnote', 'inline', or 'endnote'",
    )
    hyperlinked_citations: bool = Field(
        True,
        description="Generate clickable hyperlinks for citations in PDF/HTML",
    )
    consistency_validation_enabled: bool = Field(
        True,
        description="Check narratives for contradictions across frameworks",
    )
    consistency_target_pct: float = Field(
        DEFAULT_NARRATIVE_CONSISTENCY_TARGET_PCT,
        ge=80.0,
        le=100.0,
        description="Target consistency score across frameworks (0-100%)",
    )
    harmonization_suggestions: bool = Field(
        True,
        description="Suggest edits to harmonize contradicting narratives",
    )
    human_review_required: bool = Field(
        True,
        description="Require human review and approval before publication",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable multi-step review and approval workflow",
    )
    max_narrative_length_words: int = Field(
        5000,
        ge=500,
        le=20000,
        description="Maximum word count per narrative section",
    )

    @field_validator("citation_format")
    @classmethod
    def validate_citation_format(cls, v: str) -> str:
        """Validate citation format is recognized."""
        valid = {"footnote", "inline", "endnote"}
        if v not in valid:
            raise ValueError(
                f"Invalid citation_format: {v}. Must be one of: {sorted(valid)}"
            )
        return v


class TranslationConfig(BaseModel):
    """Configuration for multi-language narrative translation.

    Defines supported languages, translation service, terminology
    management, and quality validation settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for report generation (ISO 639-1 codes)",
    )
    primary_language: str = Field(
        "en",
        description="Primary language for original narrative drafting",
    )
    translation_service: TranslationService = Field(
        TranslationService.DEEPL,
        description="Translation service provider",
    )
    climate_glossary_enabled: bool = Field(
        True,
        description="Use climate-specific terminology glossary for translation consistency",
    )
    preserve_citations: bool = Field(
        True,
        description="Preserve citation links during translation",
    )
    quality_validation_enabled: bool = Field(
        True,
        description="Validate translation quality with back-translation checks",
    )
    quality_target_pct: float = Field(
        98.0,
        ge=90.0,
        le=100.0,
        description="Target translation quality score (0-100%)",
    )

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Validate language codes are supported."""
        valid = set(SUPPORTED_LANGUAGES.keys())
        invalid = [lang for lang in v if lang not in valid]
        if invalid:
            logger.warning(
                "Unrecognized language codes: %s. Supported: %s",
                invalid, sorted(valid),
            )
        return v


class XBRLConfig(BaseModel):
    """Configuration for XBRL/iXBRL tagging and taxonomy management.

    Defines taxonomy versions, validation settings, and rendering
    options for SEC and CSRD digital reporting.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    xbrl_enabled: bool = Field(
        True,
        description="Enable XBRL/iXBRL tag generation",
    )
    sec_taxonomy_version: str = Field(
        "2024",
        description="SEC climate disclosure taxonomy version",
    )
    csrd_taxonomy_version: str = Field(
        "2024",
        description="CSRD ESRS digital taxonomy version",
    )
    issb_taxonomy_version: str = Field(
        "2024",
        description="ISSB IFRS S2 taxonomy version",
    )
    taxonomy_validation_enabled: bool = Field(
        True,
        description="Validate XBRL tags against official taxonomy schemas",
    )
    ixbrl_enabled: bool = Field(
        True,
        description="Generate iXBRL (inline XBRL) human+machine-readable reports",
    )
    taxonomy_cache_enabled: bool = Field(
        True,
        description="Cache downloaded taxonomy schemas locally",
    )
    taxonomy_cache_ttl_hours: int = Field(
        168,
        ge=1,
        le=720,
        description="Cache TTL for taxonomy schemas in hours (default: 1 week)",
    )
    auto_tag_detection: bool = Field(
        True,
        description="Automatically detect appropriate XBRL tags for metrics",
    )


class AssuranceConfig(BaseModel):
    """Configuration for assurance evidence packaging and audit support.

    Defines evidence bundle components, assurance level, provenance
    tracking, and ISAE 3410 compliance settings.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="External assurance engagement level",
    )
    include_evidence_bundle: bool = Field(
        True,
        description="Generate evidence bundle for external auditors",
    )
    include_provenance: bool = Field(
        True,
        description="Include SHA-256 provenance hashes for all calculations",
    )
    include_lineage_diagrams: bool = Field(
        True,
        description="Include data lineage diagrams in evidence bundle",
    )
    include_methodology_docs: bool = Field(
        True,
        description="Include calculation methodology documentation",
    )
    include_control_matrix: bool = Field(
        True,
        description="Include ISAE 3410 control matrix in evidence bundle",
    )
    include_assumption_register: bool = Field(
        True,
        description="Include assumption register with justifications",
    )
    include_change_log: bool = Field(
        True,
        description="Include immutable change log of all modifications",
    )
    evidence_bundle_format: str = Field(
        "zip",
        description="Evidence bundle output format: 'zip' or 'tar.gz'",
    )
    isae_3410_compliance: bool = Field(
        True,
        description="Ensure evidence bundle meets ISAE 3410 requirements",
    )
    isae_3000_compliance: bool = Field(
        False,
        description="Ensure evidence bundle meets ISAE 3000 requirements",
    )

    @field_validator("evidence_bundle_format")
    @classmethod
    def validate_bundle_format(cls, v: str) -> str:
        """Validate evidence bundle format."""
        valid = {"zip", "tar.gz"}
        if v not in valid:
            raise ValueError(
                f"Invalid evidence_bundle_format: {v}. Must be one of: {sorted(valid)}"
            )
        return v


class DashboardConfig(BaseModel):
    """Configuration for interactive dashboard generation.

    Defines dashboard types, stakeholder views, chart settings,
    and interactivity options.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    executive_dashboard_enabled: bool = Field(
        True,
        description="Generate executive overview dashboard",
    )
    framework_dashboards_enabled: bool = Field(
        True,
        description="Generate per-framework detail dashboards",
    )
    stakeholder_views_enabled: bool = Field(
        True,
        description="Generate stakeholder-specific dashboard views",
    )
    default_stakeholder_view: StakeholderViewType = Field(
        StakeholderViewType.BOARD,
        description="Default stakeholder view for dashboard",
    )
    framework_coverage_heatmap: bool = Field(
        True,
        description="Include framework coverage heatmap in executive dashboard",
    )
    deadline_countdown: bool = Field(
        True,
        description="Include deadline countdown timers in dashboard",
    )
    drill_down_enabled: bool = Field(
        True,
        description="Enable drill-down from summary to detail views",
    )
    interactive_charts: bool = Field(
        True,
        description="Enable interactive JavaScript charts (Chart.js)",
    )
    responsive_design: bool = Field(
        True,
        description="Enable responsive design for mobile viewing",
    )
    export_to_pdf: bool = Field(
        True,
        description="Enable PDF export from HTML dashboards",
    )
    auto_refresh_interval_seconds: int = Field(
        0,
        ge=0,
        le=3600,
        description="Auto-refresh interval for real-time dashboards (0 = disabled)",
    )


class ValidationConfig(BaseModel):
    """Configuration for report validation and quality scoring.

    Defines schema validation, completeness checks, cross-framework
    consistency validation, and quality scoring parameters.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    schema_validation_enabled: bool = Field(
        True,
        description="Validate reports against framework JSON schemas",
    )
    completeness_validation_enabled: bool = Field(
        True,
        description="Check all required fields are present for each framework",
    )
    consistency_validation_enabled: bool = Field(
        True,
        description="Cross-framework consistency checks",
    )
    consistency_strictness: ConsistencyStrictness = Field(
        ConsistencyStrictness.STANDARD,
        description="Strictness level for consistency validation",
    )
    quality_scoring_enabled: bool = Field(
        True,
        description="Generate overall quality score (0-100%) for each report",
    )
    minimum_quality_score: float = Field(
        80.0,
        ge=0.0,
        le=100.0,
        description="Minimum quality score required for report approval",
    )
    metric_reconciliation: bool = Field(
        True,
        description="Reconcile metric values across frameworks",
    )
    metric_tolerance_pct: float = Field(
        0.1,
        ge=0.0,
        le=5.0,
        description="Tolerance for metric reconciliation differences (%)",
    )
    narrative_spell_check: bool = Field(
        True,
        description="Run spell check on generated narratives",
    )
    narrative_grammar_check: bool = Field(
        True,
        description="Run grammar check on generated narratives",
    )
    block_publish_on_critical: bool = Field(
        True,
        description="Block publication if critical validation errors exist",
    )


class NotificationConfig(BaseModel):
    """Configuration for alerting and notification channels.

    Defines email, Slack, and webhook notifications for report
    generation events, deadline reminders, and validation alerts.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    email_notifications_enabled: bool = Field(
        True,
        description="Enable email notifications for report events",
    )
    slack_notifications_enabled: bool = Field(
        False,
        description="Enable Slack notifications for report events",
    )
    teams_notifications_enabled: bool = Field(
        False,
        description="Enable Microsoft Teams notifications",
    )
    webhook_notifications_enabled: bool = Field(
        False,
        description="Enable webhook notifications for integration",
    )
    notification_recipients: List[str] = Field(
        default_factory=list,
        description="Email addresses for notification recipients",
    )
    slack_channel: Optional[str] = Field(
        None,
        description="Slack channel for notifications",
    )
    webhook_url: Optional[str] = Field(
        None,
        description="Webhook URL for notifications",
    )
    deadline_reminders: List[int] = Field(
        default_factory=lambda: [120, 90, 60, 30, 14, 7],
        description="Days before deadline to send reminders",
    )
    notify_on_generation: bool = Field(
        True,
        description="Notify when report generation completes",
    )
    notify_on_validation_failure: bool = Field(
        True,
        description="Notify when validation fails with critical errors",
    )
    notify_on_approval: bool = Field(
        True,
        description="Notify when report is approved for publication",
    )


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning.

    Defines caching, concurrency, timeout, and resource limit
    settings for report generation pipelines.
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    cache_enabled: bool = Field(
        True,
        description="Enable Redis-based caching for report data and schemas",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    cache_hit_ratio_target: float = Field(
        DEFAULT_CACHE_HIT_RATIO_TARGET_PCT,
        ge=80.0,
        le=100.0,
        description="Target cache hit ratio percentage",
    )
    max_concurrent_reports: int = Field(
        DEFAULT_MAX_CONCURRENT_REPORTS,
        ge=1,
        le=32,
        description="Maximum concurrent report generation threads",
    )
    report_generation_timeout_seconds: int = Field(
        DEFAULT_REPORT_GENERATION_TIMEOUT_SECONDS,
        ge=5,
        le=120,
        description="Maximum timeout for full report suite generation",
    )
    api_response_timeout_ms: int = Field(
        DEFAULT_API_RESPONSE_TIMEOUT_MS,
        ge=50,
        le=5000,
        description="Maximum API response time in milliseconds (p95)",
    )
    pdf_render_timeout_seconds: int = Field(
        DEFAULT_PDF_RENDER_TIMEOUT_SECONDS,
        ge=2,
        le=60,
        description="Maximum timeout for PDF rendering",
    )
    parallel_framework_execution: bool = Field(
        True,
        description="Execute framework report workflows in parallel",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        4096,
        ge=512,
        le=32768,
        description="Memory limit in MB for the report generation pipeline",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class NetZeroReportingConfig(BaseModel):
    """Main configuration model for PACK-030 Net Zero Reporting Pack.

    This is the root Pydantic v2 configuration model containing all parameters
    for multi-framework report generation, data aggregation, narrative generation,
    XBRL/iXBRL tagging, assurance evidence packaging, dashboard creation, format
    rendering, and cross-framework consistency validation.

    The model supports 7 reporting frameworks (SBTi, CDP, TCFD, GRI, ISSB, SEC,
    CSRD), 6 output formats (PDF, HTML, Excel, JSON, XBRL, iXBRL), 4 languages
    (EN, DE, FR, ES), and 5 stakeholder views (investor, regulator, customer,
    employee, board).
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_default=True,
        json_schema_extra={
            "title": "PACK-030 Net Zero Reporting Configuration",
            "description": "Configuration for multi-framework climate report generation",
        },
    )

    # --- Organization Identity ---
    organization_name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    organization_id: Optional[str] = Field(
        None,
        description="Unique organization identifier (UUID or internal code)",
    )

    # --- Temporal Settings ---
    reporting_year: int = Field(
        DEFAULT_REPORTING_YEAR,
        ge=2020,
        le=2035,
        description="Reporting year for current disclosure cycle",
    )
    baseline_year: int = Field(
        DEFAULT_BASELINE_YEAR,
        ge=2010,
        le=2030,
        description="Base year for emission baseline (GHG Protocol compliant)",
    )
    fiscal_year_end_month: int = Field(
        12,
        ge=1,
        le=12,
        description="Fiscal year-end month (1-12, default: December)",
    )

    # --- Framework Configuration ---
    frameworks: FrameworkConfig = Field(
        default_factory=FrameworkConfig,
        description="Reporting framework selection and prioritization",
    )

    # --- Output Settings ---
    output_formats: List[str] = Field(
        default_factory=lambda: ["PDF", "HTML", "Excel", "JSON", "XBRL", "iXBRL"],
        description="Enabled output formats for report rendering",
    )
    languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Languages for report generation",
    )

    # --- Sub-Configurations ---
    branding: BrandingConfig = Field(
        default_factory=BrandingConfig,
        description="Report branding and visual identity",
    )
    data_aggregation: DataAggregationConfig = Field(
        default_factory=DataAggregationConfig,
        description="Multi-source data aggregation settings",
    )
    narrative: NarrativeConfig = Field(
        default_factory=NarrativeConfig,
        description="AI-assisted narrative generation settings",
    )
    translation: TranslationConfig = Field(
        default_factory=TranslationConfig,
        description="Multi-language translation settings",
    )
    xbrl: XBRLConfig = Field(
        default_factory=XBRLConfig,
        description="XBRL/iXBRL tagging and taxonomy settings",
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig,
        description="Assurance evidence packaging settings",
    )
    dashboard: DashboardConfig = Field(
        default_factory=DashboardConfig,
        description="Interactive dashboard generation settings",
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Report validation and quality scoring settings",
    )
    notifications: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="Alerting and notification channel settings",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Runtime performance tuning settings",
    )

    # --- Pack Metadata ---
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # --- Cross-Cutting Validators ---

    @model_validator(mode="after")
    def validate_baseline_before_reporting(self) -> "NetZeroReportingConfig":
        """Ensure baseline year is before reporting year."""
        if self.baseline_year > self.reporting_year:
            raise ValueError(
                f"baseline_year ({self.baseline_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_output_formats(self) -> "NetZeroReportingConfig":
        """Validate all output formats are recognized."""
        valid = set(OUTPUT_FORMAT_SPECS.keys())
        invalid = [f for f in self.output_formats if f not in valid]
        if invalid:
            logger.warning(
                "Unrecognized output formats: %s. Valid: %s",
                invalid, sorted(valid),
            )
        return self

    @model_validator(mode="after")
    def validate_xbrl_framework_dependency(self) -> "NetZeroReportingConfig":
        """Warn if XBRL enabled but no XBRL-requiring framework is active."""
        if self.xbrl.xbrl_enabled:
            xbrl_frameworks = {"SEC", "CSRD", "ISSB"}
            enabled = set(self.frameworks.frameworks_enabled)
            if not xbrl_frameworks & enabled:
                logger.warning(
                    "XBRL tagging is enabled but no XBRL-requiring framework "
                    "(SEC, CSRD, ISSB) is in frameworks_enabled."
                )
        return self

    @model_validator(mode="after")
    def validate_language_consistency(self) -> "NetZeroReportingConfig":
        """Ensure translation languages match top-level languages."""
        if set(self.translation.languages) != set(self.languages):
            logger.warning(
                "translation.languages %s differs from top-level languages %s. "
                "Using top-level languages for consistency.",
                self.translation.languages, self.languages,
            )
            object.__setattr__(self, "languages", list(set(self.languages)))
        return self

    @model_validator(mode="after")
    def validate_assurance_with_provenance(self) -> "NetZeroReportingConfig":
        """Warn if assurance bundle enabled but provenance tracking disabled."""
        if self.assurance.include_evidence_bundle and not self.assurance.include_provenance:
            logger.warning(
                "Evidence bundle is enabled but provenance tracking is disabled. "
                "Auditors require SHA-256 provenance for ISAE 3410 compliance."
            )
        return self

    @model_validator(mode="after")
    def validate_primary_framework_in_enabled(self) -> "NetZeroReportingConfig":
        """Ensure primary framework is in the enabled list."""
        if (self.frameworks.primary_framework
                and self.frameworks.primary_framework not in self.frameworks.frameworks_enabled):
            logger.warning(
                "primary_framework '%s' is not in frameworks_enabled %s. "
                "Adding it automatically.",
                self.frameworks.primary_framework,
                self.frameworks.frameworks_enabled,
            )
            self.frameworks.frameworks_enabled.append(self.frameworks.primary_framework)
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine identifiers that should be enabled.

        Returns:
            Sorted list of enabled engine identifier strings.
        """
        engines = [
            "data_aggregation_engine",
            "report_compilation_engine",
            "validation_engine",
        ]

        if self.narrative.citation_management_enabled or self.narrative.consistency_validation_enabled:
            engines.append("narrative_generation_engine")

        if any(f in self.frameworks.frameworks_enabled for f in ["TCFD", "GRI", "ISSB"]):
            engines.append("framework_mapping_engine")

        if self.xbrl.xbrl_enabled:
            engines.append("xbrl_tagging_engine")

        if self.dashboard.executive_dashboard_enabled or self.dashboard.framework_dashboards_enabled:
            engines.append("dashboard_generation_engine")

        if self.assurance.include_evidence_bundle:
            engines.append("assurance_packaging_engine")

        if self.translation.languages and len(self.translation.languages) > 1:
            engines.append("translation_engine")

        if self.output_formats:
            engines.append("format_rendering_engine")

        return sorted(set(engines))

    def get_enabled_workflows(self) -> List[str]:
        """Return list of workflow identifiers to execute.

        Returns:
            Sorted list of enabled workflow identifier strings.
        """
        workflows = []

        framework_workflow_map = {
            "SBTi": "sbti_progress_workflow",
            "CDP": "cdp_questionnaire_workflow",
            "TCFD": "tcfd_disclosure_workflow",
            "GRI": "gri_305_workflow",
            "ISSB": "issb_ifrs_s2_workflow",
            "SEC": "sec_climate_workflow",
            "CSRD": "csrd_esrs_e1_workflow",
        }

        for framework in self.frameworks.frameworks_enabled:
            workflow = framework_workflow_map.get(framework)
            if workflow:
                workflows.append(workflow)

        if len(self.frameworks.frameworks_enabled) > 1:
            workflows.append("multi_framework_workflow")

        return sorted(set(workflows))

    def get_framework_info(self, framework: str) -> Dict[str, str]:
        """Get detailed information about a reporting framework.

        Args:
            framework: Framework identifier (SBTi, CDP, TCFD, etc.).

        Returns:
            Dictionary with framework details.
        """
        return SUPPORTED_FRAMEWORKS.get(
            framework,
            {"full_name": framework, "version": "Unknown"},
        )

    def get_output_format_info(self, format_name: str) -> Dict[str, Any]:
        """Get detailed information about an output format.

        Args:
            format_name: Output format identifier (PDF, HTML, etc.).

        Returns:
            Dictionary with format details.
        """
        return OUTPUT_FORMAT_SPECS.get(
            format_name,
            {"name": format_name, "description": "Unknown format"},
        )

    def get_stakeholder_view_info(self, view_type: str) -> Dict[str, Any]:
        """Get detailed information about a stakeholder view.

        Args:
            view_type: Stakeholder view type identifier.

        Returns:
            Dictionary with view details.
        """
        return STAKEHOLDER_VIEW_TYPES.get(
            view_type,
            {"name": view_type, "description": "Custom view"},
        )


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-030.

    Handles preset loading, environment variable overrides, and
    configuration merging. Provides SHA-256 config hashing for
    provenance tracking and JSON Schema export for API documentation.

    Example:
        >>> config = PackConfig.from_preset("multi_framework")
        >>> print(config.pack.frameworks.frameworks_enabled)
        ['SBTi', 'CDP', 'TCFD', 'GRI', 'ISSB', 'SEC', 'CSRD']
        >>> config = PackConfig.from_preset("csrd_focus", overrides={"languages": ["en", "de"]})
        >>> print(config.pack.languages)
        ['en', 'de']
    """

    model_config = ConfigDict(str_strip_whitespace=True, validate_default=True)

    pack: NetZeroReportingConfig = Field(
        default_factory=NetZeroReportingConfig,
        description="Main Net Zero Reporting configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-030-net-zero-reporting",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Loads the preset YAML file, applies environment variable overrides
        (NZ_REPORTING_* prefix), then applies any explicit runtime overrides.

        Args:
            preset_name: Name of the preset (csrd_focus, cdp_alist,
                tcfd_investor, sbti_validation, sec_10k, multi_framework,
                investor_relations, assurance_ready).
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in SUPPORTED_PRESETS.
        """
        if preset_name not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(SUPPORTED_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = _get_env_overrides("NZ_REPORTING_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = NetZeroReportingConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.

        Raises:
            FileNotFoundError: If YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = NetZeroReportingConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PackConfig":
        """Load configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            PackConfig instance.
        """
        pack_config = NetZeroReportingConfig(**config_dict)
        return cls(pack=pack_config)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Performs advisory validation beyond Pydantic's built-in validation.
        Returns warnings, not hard errors.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)

    def export_json_schema(self) -> Dict[str, Any]:
        """Export the configuration JSON Schema for API documentation.

        Returns:
            JSON Schema dictionary for the NetZeroReportingConfig model.
        """
        return NetZeroReportingConfig.model_json_schema()


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(yaml_path: Union[str, Path]) -> PackConfig:
    """Load configuration from a YAML file.

    Convenience wrapper around PackConfig.from_yaml().

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        PackConfig instance.
    """
    return PackConfig.from_yaml(yaml_path)


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    return _merge_config(base, override)


def _get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Environment variables prefixed with the given prefix are loaded and
    mapped to configuration keys. Nested keys use double underscore.

    Example:
        NZ_REPORTING_REPORTING_YEAR=2026
        NZ_REPORTING_NARRATIVE__CONSISTENCY_TARGET_PCT=98.0
        NZ_REPORTING_XBRL__SEC_TAXONOMY_VERSION=2025

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            parts = config_key.split("__")
            current = overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Parse value types
            if value.lower() in ("true", "yes", "1"):
                current[parts[-1]] = True
            elif value.lower() in ("false", "no", "0"):
                current[parts[-1]] = False
            else:
                try:
                    current[parts[-1]] = int(value)
                except ValueError:
                    try:
                        current[parts[-1]] = float(value)
                    except ValueError:
                        current[parts[-1]] = value
    return overrides


def get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Public wrapper for loading environment variable overrides.

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    return _get_env_overrides(prefix)


def validate_config(config: NetZeroReportingConfig) -> List[str]:
    """Validate a net zero reporting configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: NetZeroReportingConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization_name:
        warnings.append(
            "Organization name is empty. Set organization_name for meaningful reports."
        )

    # Check at least one framework is enabled
    if not config.frameworks.frameworks_enabled:
        warnings.append(
            "No frameworks enabled. Enable at least one framework for report generation."
        )

    # Check at least one output format is enabled
    if not config.output_formats:
        warnings.append(
            "No output formats enabled. Enable at least one format (PDF, HTML, etc.)."
        )

    # Check XBRL consistency
    if config.xbrl.xbrl_enabled:
        xbrl_formats = {"XBRL", "iXBRL"}
        if not xbrl_formats & set(config.output_formats):
            warnings.append(
                "XBRL tagging is enabled but neither XBRL nor iXBRL is in output_formats."
            )

    # Check SEC framework requires XBRL
    if "SEC" in config.frameworks.frameworks_enabled:
        if not config.xbrl.xbrl_enabled:
            warnings.append(
                "SEC framework is enabled but XBRL tagging is disabled. "
                "SEC climate disclosures require XBRL/iXBRL tagging."
            )

    # Check CSRD framework needs digital taxonomy
    if "CSRD" in config.frameworks.frameworks_enabled:
        if not config.xbrl.xbrl_enabled:
            warnings.append(
                "CSRD framework is enabled but XBRL tagging is disabled. "
                "CSRD ESRS E1 requires digital taxonomy tagging."
            )

    # Check narrative consistency
    if config.narrative.consistency_validation_enabled:
        if len(config.frameworks.frameworks_enabled) < 2:
            warnings.append(
                "Narrative consistency validation is enabled but fewer than 2 "
                "frameworks are active. Cross-framework checks require 2+ frameworks."
            )

    # Check assurance configuration
    if config.assurance.assurance_level != AssuranceLevel.NONE:
        if not config.assurance.include_evidence_bundle:
            warnings.append(
                f"Assurance level is '{config.assurance.assurance_level.value}' but "
                f"evidence bundle is disabled. Auditors require evidence packages."
            )
        if not config.assurance.include_provenance:
            warnings.append(
                f"Assurance level is '{config.assurance.assurance_level.value}' but "
                f"provenance tracking is disabled. SHA-256 hashes are required for audit."
            )

    # Check translation language availability
    if len(config.languages) > 1:
        if config.translation.translation_service == TranslationService.NATIVE:
            warnings.append(
                "Multiple languages configured but translation service is 'native'. "
                "Configure 'deepl' or 'google' for automated translation."
            )

    # Check notification configuration
    if config.notifications.email_notifications_enabled:
        if not config.notifications.notification_recipients:
            warnings.append(
                "Email notifications enabled but no recipients configured."
            )

    if config.notifications.slack_notifications_enabled:
        if not config.notifications.slack_channel:
            warnings.append(
                "Slack notifications enabled but no slack_channel configured."
            )

    if config.notifications.webhook_notifications_enabled:
        if not config.notifications.webhook_url:
            warnings.append(
                "Webhook notifications enabled but no webhook_url configured."
            )

    # Check data source requirements
    required_sources = []
    if config.data_aggregation.pack_021_requirement == DataSourceRequirement.REQUIRED:
        required_sources.append("PACK-021")
    if config.data_aggregation.pack_029_requirement == DataSourceRequirement.REQUIRED:
        required_sources.append("PACK-029")
    if required_sources:
        logger.info(
            "Required data sources for this configuration: %s",
            required_sources,
        )

    # Check dashboard with stakeholder views
    if config.dashboard.stakeholder_views_enabled:
        if not config.dashboard.executive_dashboard_enabled:
            warnings.append(
                "Stakeholder views enabled but executive dashboard disabled. "
                "Consider enabling executive_dashboard_enabled for complete coverage."
            )

    # Check performance settings
    if config.performance.report_generation_timeout_seconds < 5:
        warnings.append(
            f"report_generation_timeout_seconds ({config.performance.report_generation_timeout_seconds}s) "
            f"is very low. Multi-framework report generation may exceed this limit."
        )

    # Check parallel execution consistency
    if (config.performance.parallel_framework_execution
            and len(config.frameworks.frameworks_enabled) > config.performance.max_concurrent_reports):
        warnings.append(
            f"Parallel execution enabled with {len(config.frameworks.frameworks_enabled)} frameworks "
            f"but max_concurrent_reports is {config.performance.max_concurrent_reports}. "
            f"Some frameworks will queue."
        )

    return warnings


def get_framework_info(framework: str) -> Dict[str, str]:
    """Get reporting framework details.

    Args:
        framework: Framework identifier (SBTi, CDP, TCFD, GRI, ISSB, SEC, CSRD).

    Returns:
        Dictionary with full_name, version, disclosure_frequency, etc.
    """
    return SUPPORTED_FRAMEWORKS.get(
        framework,
        {"full_name": framework, "version": "Unknown"},
    )


def get_output_format_info(format_name: str) -> Dict[str, Any]:
    """Get output format details.

    Args:
        format_name: Format identifier (PDF, HTML, Excel, JSON, XBRL, iXBRL).

    Returns:
        Dictionary with name, description, mime_type, renderer.
    """
    return OUTPUT_FORMAT_SPECS.get(
        format_name,
        {"name": format_name, "description": "Unknown format"},
    )


def get_assurance_standard_info(standard: str) -> Dict[str, str]:
    """Get assurance standard details.

    Args:
        standard: Standard identifier (ISAE_3410, ISAE_3000, AA1000AS).

    Returns:
        Dictionary with name, scope, levels, issuing_body.
    """
    return ASSURANCE_STANDARDS.get(
        standard,
        {"name": standard, "scope": "Unknown"},
    )


def get_evidence_bundle_info(component: str) -> Dict[str, str]:
    """Get evidence bundle component details.

    Args:
        component: Component identifier (provenance_hashes, lineage_diagrams, etc.).

    Returns:
        Dictionary with name, description, format.
    """
    return EVIDENCE_BUNDLE_COMPONENTS.get(
        component,
        {"name": component, "description": "Unknown component"},
    )


def get_stakeholder_view_info(view_type: str) -> Dict[str, Any]:
    """Get stakeholder view details.

    Args:
        view_type: View type identifier (investor, regulator, customer, etc.).

    Returns:
        Dictionary with name, description, primary/secondary frameworks.
    """
    return STAKEHOLDER_VIEW_TYPES.get(
        view_type,
        {"name": view_type, "description": "Custom view"},
    )


def get_xbrl_taxonomy_info(framework: str) -> Dict[str, str]:
    """Get XBRL taxonomy specification details.

    Args:
        framework: Framework identifier (SEC, CSRD, ISSB).

    Returns:
        Dictionary with taxonomy_name, namespace, schema_url, version.
    """
    return XBRL_TAXONOMY_SPECS.get(
        framework,
        {"taxonomy_name": "Unknown", "version": "Unknown"},
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def list_supported_frameworks() -> Dict[str, str]:
    """List all supported reporting frameworks.

    Returns:
        Dictionary mapping framework codes to full names.
    """
    return {k: v["full_name"] for k, v in SUPPORTED_FRAMEWORKS.items()}


def list_output_formats() -> Dict[str, str]:
    """List all supported output formats.

    Returns:
        Dictionary mapping format codes to descriptions.
    """
    return {k: v["description"] for k, v in OUTPUT_FORMAT_SPECS.items()}


def list_supported_languages() -> Dict[str, str]:
    """List all supported languages.

    Returns:
        Dictionary mapping ISO 639-1 codes to language names.
    """
    return {k: v["name"] for k, v in SUPPORTED_LANGUAGES.items()}


def list_stakeholder_views() -> Dict[str, str]:
    """List all supported stakeholder view types.

    Returns:
        Dictionary mapping view type codes to descriptions.
    """
    return {k: v["description"] for k, v in STAKEHOLDER_VIEW_TYPES.items()}


def list_branding_styles() -> Dict[str, str]:
    """List all supported branding style profiles.

    Returns:
        Dictionary mapping style codes to descriptions.
    """
    return {k: v["description"] for k, v in BRANDING_STYLES.items()}


def list_evidence_bundle_components() -> Dict[str, str]:
    """List all evidence bundle components.

    Returns:
        Dictionary mapping component codes to descriptions.
    """
    return {k: v["description"] for k, v in EVIDENCE_BUNDLE_COMPONENTS.items()}


def list_consistency_rules() -> Dict[str, str]:
    """List all consistency validation rule categories.

    Returns:
        Dictionary mapping rule category codes to descriptions.
    """
    return {k: v["description"] for k, v in CONSISTENCY_RULE_CATEGORIES.items()}
