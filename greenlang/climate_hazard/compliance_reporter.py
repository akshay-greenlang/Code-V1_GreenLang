# -*- coding: utf-8 -*-
"""
Compliance Reporter Engine - AGENT-DATA-020 (Engine 6 of 7)

Generates climate risk compliance reports aligned with six regulatory and
voluntary disclosure frameworks: TCFD, CSRD/ESRS, EU Taxonomy, SEC Climate,
IFRS S2, and NGFS.  Supports five report types (Physical Risk Assessment,
Scenario Analysis, Adaptation Screening, Exposure Summary, Executive
Dashboard) and five output formats (JSON, HTML, Markdown, Text, CSV).

The engine maintains an in-memory report registry with full CRUD operations,
framework-specific compliance templates, evidence collection from risk/
exposure/vulnerability data, and deterministic recommendation generation
based on risk levels.

Frameworks (6):
    tcfd          - Task Force on Climate-related Financial Disclosures
    csrd_esrs     - Corporate Sustainability Reporting Directive / ESRS E1
    eu_taxonomy   - EU Taxonomy Regulation climate adaptation criteria
    sec_climate   - SEC Climate-Related Financial Disclosures (S-X/S-K)
    ifrs_s2       - IFRS S2 Sustainability-Related Financial Disclosures
    ngfs          - Network for Greening the Financial System scenarios

Report Types (5):
    PHYSICAL_RISK_ASSESSMENT  - Comprehensive hazard-by-hazard risk analysis
    SCENARIO_ANALYSIS         - Forward-looking scenario comparison
    ADAPTATION_SCREENING      - Adaptation measures screening / prioritisation
    EXPOSURE_SUMMARY          - Portfolio-level exposure concentration summary
    EXECUTIVE_DASHBOARD       - High-level KPI dashboard for board reporting

Output Formats (5):
    JSON      - Structured JSON for programmatic consumption
    HTML      - Self-contained HTML page with inline styles
    MARKDOWN  - Markdown-formatted report for documentation systems
    TEXT      - Plain-text summary for terminal or log output
    CSV       - Comma-separated values for spreadsheet import

Zero-Hallucination: All report content is assembled from deterministic
templates and input data.  Risk summaries, compliance scores, and
recommendations use transparent formulae with no LLM calls.  Every
generated report carries a SHA-256 provenance hash for tamper detection.

Example:
    >>> from greenlang.climate_hazard.compliance_reporter import ComplianceReporterEngine
    >>> engine = ComplianceReporterEngine()
    >>> report = engine.generate_report(
    ...     report_type="physical_risk_assessment",
    ...     report_format="json",
    ...     framework="tcfd",
    ...     title="Annual Physical Risk Report",
    ... )
    >>> assert report["report_id"]
    >>> assert report["provenance_hash"]

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["ComplianceReporterEngine"]

# ---------------------------------------------------------------------------
# Graceful imports -- provenance.py
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.provenance import ProvenanceTracker
except Exception:  # pragma: no cover -- fallback when module unavailable

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal fallback ProvenanceTracker when module unavailable."""

        GENESIS_HASH = hashlib.sha256(
            b"greenlang-climate-hazard-connector-genesis"
        ).hexdigest()

        def __init__(
            self, genesis_hash: str = "greenlang-climate-hazard-connector-genesis"
        ) -> None:
            self._lock = threading.Lock()
            self._chain: List[Dict[str, Any]] = []
            self._last: str = self.GENESIS_HASH

        def hash_record(self, data: Dict[str, Any]) -> str:
            s = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        def build_hash(self, data: Any) -> str:
            s = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(s.encode("utf-8")).hexdigest()

        def record(
            self,
            entity_type: str,
            action: str,
            entity_id: str,
            data: Any = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> Any:
            ts = datetime.now(timezone.utc).isoformat()
            data_hash = self.build_hash(data)
            combined = json.dumps(
                {
                    "previous": self._last,
                    "input": data_hash,
                    "output": data_hash,
                    "operation": action,
                    "timestamp": ts,
                },
                sort_keys=True,
            )
            chain_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()
            with self._lock:
                self._chain.append(
                    {
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "action": action,
                        "data_hash": data_hash,
                        "timestamp": ts,
                        "chain_hash": chain_hash,
                    }
                )
                self._last = chain_hash

            class _Entry:
                def __init__(self, hv: str) -> None:
                    self.hash_value = hv

            return _Entry(chain_hash)

        def reset(self) -> None:
            with self._lock:
                self._chain.clear()
                self._last = self.GENESIS_HASH

        @property
        def entry_count(self) -> int:
            with self._lock:
                return len(self._chain)


# ---------------------------------------------------------------------------
# Graceful imports -- metrics.py
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.metrics import (
        record_report as _record_report_metric,
        observe_pipeline_duration as _observe_pipeline_duration,
    )
except Exception:  # pragma: no cover

    def _record_report_metric(report_type: str, format: str) -> None:  # type: ignore[misc]
        """No-op metric stub."""

    def _observe_pipeline_duration(pipeline_stage: str, seconds: float) -> None:  # type: ignore[misc]
        """No-op metric stub."""


# ---------------------------------------------------------------------------
# Graceful imports -- models.py
# ---------------------------------------------------------------------------

try:
    from greenlang.climate_hazard.models import (
        ComplianceReport as _ComplianceReport,
        GenerateReportRequest as _GenerateReportRequest,
        ReportType as _ReportType,
        ReportFormat as _ReportFormat,
        RiskLevel as _RiskLevel,
        HazardType as _HazardType,
        Scenario as _Scenario,
        TimeHorizon as _TimeHorizon,
    )
except Exception:  # pragma: no cover
    _ComplianceReport = None  # type: ignore[assignment, misc]
    _GenerateReportRequest = None  # type: ignore[assignment, misc]
    _ReportType = None  # type: ignore[assignment, misc]
    _ReportFormat = None  # type: ignore[assignment, misc]
    _RiskLevel = None  # type: ignore[assignment, misc]
    _HazardType = None  # type: ignore[assignment, misc]
    _Scenario = None  # type: ignore[assignment, misc]
    _TimeHorizon = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Enumerations (local fallback-safe copies)
# ---------------------------------------------------------------------------


class ReportTypeLocal(str, Enum):
    """Report type enumeration (engine-local copy).

    Values:
        PHYSICAL_RISK_ASSESSMENT: Hazard-by-hazard analysis.
        SCENARIO_ANALYSIS: Forward-looking scenario comparison.
        ADAPTATION_SCREENING: Adaptation measure screening.
        EXPOSURE_SUMMARY: Portfolio exposure concentration summary.
        EXECUTIVE_DASHBOARD: Board-level KPI dashboard.
    """

    PHYSICAL_RISK_ASSESSMENT = "physical_risk_assessment"
    SCENARIO_ANALYSIS = "scenario_analysis"
    ADAPTATION_SCREENING = "adaptation_screening"
    EXPOSURE_SUMMARY = "exposure_summary"
    EXECUTIVE_DASHBOARD = "executive_dashboard"


class ReportFormatLocal(str, Enum):
    """Report output format enumeration (engine-local copy).

    Values:
        JSON: Structured JSON output.
        HTML: Self-contained HTML page.
        MARKDOWN: Markdown-formatted text.
        TEXT: Plain-text output.
        CSV: Comma-separated values.
    """

    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    CSV = "csv"


# Resolve enums: prefer SDK models, fall back to local copies.
ReportType = _ReportType if _ReportType is not None else ReportTypeLocal
ReportFormat = _ReportFormat if _ReportFormat is not None else ReportFormatLocal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported compliance / disclosure frameworks.
SUPPORTED_FRAMEWORKS: Tuple[str, ...] = (
    "tcfd",
    "csrd_esrs",
    "eu_taxonomy",
    "sec_climate",
    "ifrs_s2",
    "ngfs",
)

#: Valid report type strings (lower-case).
VALID_REPORT_TYPES: frozenset = frozenset(
    {
        "physical_risk_assessment",
        "scenario_analysis",
        "adaptation_screening",
        "exposure_summary",
        "executive_dashboard",
    }
)

#: Valid output format strings (lower-case).
VALID_REPORT_FORMATS: frozenset = frozenset(
    {
        "json",
        "html",
        "markdown",
        "text",
        "csv",
    }
)

#: Risk level ordering for classification (score upper bounds).
RISK_LEVEL_THRESHOLDS: List[Tuple[float, str]] = [
    (20.0, "negligible"),
    (40.0, "low"),
    (60.0, "medium"),
    (80.0, "high"),
    (100.0, "extreme"),
]

#: Recommendation urgency mapped to risk levels.
URGENCY_MAP: Dict[str, str] = {
    "negligible": "routine",
    "low": "advisory",
    "medium": "recommended",
    "high": "priority",
    "extreme": "critical",
}


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class ReportRecord:
    """In-memory record of a generated compliance report.

    Attributes:
        report_id: Unique report identifier (RPT-<uuid_hex[:12]>).
        report_type: Type of climate risk report.
        report_format: Output format of the report.
        framework: Compliance framework alignment.
        title: Human-readable report title.
        description: Brief description of the report scope.
        scope: Report scope string.
        namespace: Tenant or organizational namespace.
        asset_ids: Asset IDs covered in the report.
        hazard_types: Hazard types analysed.
        scenarios: Scenarios included.
        time_horizons: Time horizons included.
        parameters: Generation parameters.
        content: Rendered report content string.
        report_hash: SHA-256 hash of the rendered content.
        asset_count: Number of assets covered.
        hazard_count: Number of hazard types analysed.
        scenario_count: Number of scenarios included.
        risk_summary: Aggregate risk metrics.
        recommendations: Adaptation / mitigation recommendations.
        compliance_score: Framework compliance score (0-100).
        evidence_summary: Collected evidence from input data.
        generated_at: ISO 8601 UTC timestamp.
        provenance_hash: SHA-256 provenance chain hash.
    """

    report_id: str = ""
    report_type: str = "physical_risk_assessment"
    report_format: str = "json"
    framework: str = "tcfd"
    title: str = ""
    description: str = ""
    scope: str = "full"
    namespace: str = "default"
    asset_ids: List[str] = dc_field(default_factory=list)
    hazard_types: List[str] = dc_field(default_factory=list)
    scenarios: List[str] = dc_field(default_factory=list)
    time_horizons: List[str] = dc_field(default_factory=list)
    parameters: Dict[str, Any] = dc_field(default_factory=dict)
    content: str = ""
    report_hash: str = ""
    asset_count: int = 0
    hazard_count: int = 0
    scenario_count: int = 0
    risk_summary: Dict[str, Any] = dc_field(default_factory=dict)
    recommendations: List[str] = dc_field(default_factory=list)
    compliance_score: float = 0.0
    evidence_summary: Dict[str, Any] = dc_field(default_factory=dict)
    generated_at: str = ""
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the report record.
        """
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "report_format": self.report_format,
            "framework": self.framework,
            "title": self.title,
            "description": self.description,
            "scope": self.scope,
            "namespace": self.namespace,
            "asset_ids": list(self.asset_ids),
            "hazard_types": list(self.hazard_types),
            "scenarios": list(self.scenarios),
            "time_horizons": list(self.time_horizons),
            "parameters": dict(self.parameters),
            "content": self.content,
            "report_hash": self.report_hash,
            "asset_count": self.asset_count,
            "hazard_count": self.hazard_count,
            "scenario_count": self.scenario_count,
            "risk_summary": dict(self.risk_summary),
            "recommendations": list(self.recommendations),
            "compliance_score": self.compliance_score,
            "evidence_summary": dict(self.evidence_summary),
            "generated_at": self.generated_at,
            "provenance_hash": self.provenance_hash,
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    """Return current UTC datetime as ISO 8601 string."""
    return _utcnow().replace(microsecond=0).isoformat()


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short string prefix (e.g. 'RPT', 'VAL').

    Returns:
        String in the format ``<prefix>-<uuid4_hex[:12]>``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _build_provenance_hash(data: Any) -> str:
    """Build a SHA-256 hash for arbitrary data for provenance tracking.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _classify_risk_level(score: float) -> str:
    """Classify a numeric risk score (0-100) into a risk level string.

    Thresholds:
        >= 80: extreme
        >= 60: high
        >= 40: medium
        >= 20: low
        <  20: negligible

    Args:
        score: Risk score in [0, 100].

    Returns:
        Risk level string.
    """
    if score >= 80.0:
        return "extreme"
    if score >= 60.0:
        return "high"
    if score >= 40.0:
        return "medium"
    if score >= 20.0:
        return "low"
    return "negligible"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a numeric value to [lo, hi].

    Args:
        value: The value to clamp.
        lo: Lower bound (default 0.0).
        hi: Upper bound (default 100.0).

    Returns:
        Clamped value.
    """
    return max(lo, min(hi, value))


def _safe_mean(values: List[float]) -> float:
    """Compute arithmetic mean, returning 0.0 for empty lists.

    Args:
        values: List of numeric values.

    Returns:
        Arithmetic mean, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def _enum_value(val: Any) -> str:
    """Extract string value from an enum or return str(val).

    Args:
        val: Enum member or plain string.

    Returns:
        String value.
    """
    if hasattr(val, "value"):
        return str(val.value)
    return str(val)


# ---------------------------------------------------------------------------
# Framework Template Definitions
# ---------------------------------------------------------------------------

# Each framework template defines the required disclosure sections,
# descriptions, required evidence items, and compliance scoring weights.

FRAMEWORK_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "tcfd": {
        "name": "Task Force on Climate-related Financial Disclosures",
        "version": "2017 (updated 2021)",
        "description": (
            "TCFD recommendations for climate-related financial disclosures "
            "across four pillars: Governance, Strategy, Risk Management, "
            "and Metrics & Targets."
        ),
        "sections": {
            "governance": {
                "title": "Governance",
                "description": (
                    "Describe the organisation's governance around "
                    "climate-related risks and opportunities."
                ),
                "sub_sections": [
                    "Board oversight of climate-related risks",
                    "Management's role in assessing and managing climate risks",
                ],
                "required_evidence": [
                    "board_climate_oversight",
                    "management_climate_roles",
                    "climate_risk_committee",
                ],
                "weight": 0.20,
            },
            "strategy": {
                "title": "Strategy",
                "description": (
                    "Describe the actual and potential impacts of "
                    "climate-related risks and opportunities on the "
                    "organisation's businesses, strategy, and financial "
                    "planning."
                ),
                "sub_sections": [
                    "Climate-related risks and opportunities identified",
                    "Impact on business, strategy, and financial planning",
                    "Resilience of strategy under different scenarios",
                ],
                "required_evidence": [
                    "physical_risk_assessment",
                    "transition_risk_assessment",
                    "scenario_analysis",
                    "financial_impact_analysis",
                ],
                "weight": 0.30,
            },
            "risk_management": {
                "title": "Risk Management",
                "description": (
                    "Describe how the organisation identifies, assesses, "
                    "and manages climate-related risks."
                ),
                "sub_sections": [
                    "Process for identifying and assessing climate risks",
                    "Process for managing climate risks",
                    "Integration into overall risk management",
                ],
                "required_evidence": [
                    "risk_identification_process",
                    "risk_assessment_methodology",
                    "risk_management_integration",
                ],
                "weight": 0.25,
            },
            "metrics_and_targets": {
                "title": "Metrics and Targets",
                "description": (
                    "Disclose the metrics and targets used to assess and "
                    "manage relevant climate-related risks and opportunities."
                ),
                "sub_sections": [
                    "Metrics used to assess climate risks and opportunities",
                    "Scope 1, 2, and 3 GHG emissions",
                    "Targets and performance against targets",
                ],
                "required_evidence": [
                    "climate_risk_metrics",
                    "ghg_emissions_data",
                    "climate_targets",
                ],
                "weight": 0.25,
            },
        },
    },
    "csrd_esrs": {
        "name": "CSRD / ESRS E1 Climate Change",
        "version": "ESRS E1 (2023 Delegated Act)",
        "description": (
            "European Sustainability Reporting Standards E1 Climate Change "
            "disclosures under the Corporate Sustainability Reporting "
            "Directive, covering transition plans, physical and transition "
            "risks, GHG emissions, and energy."
        ),
        "sections": {
            "governance_strategy": {
                "title": "E1-1: Transition Plan for Climate Change Mitigation",
                "description": (
                    "Disclose the transition plan for climate change "
                    "mitigation including GHG reduction targets and actions."
                ),
                "sub_sections": [
                    "Transition plan overview and key assumptions",
                    "GHG emission reduction targets (Scope 1, 2, 3)",
                    "Decarbonisation levers and locked-in GHG emissions",
                    "CapEx and OpEx alignment with transition plan",
                ],
                "required_evidence": [
                    "transition_plan",
                    "ghg_targets",
                    "decarbonisation_actions",
                ],
                "weight": 0.20,
            },
            "impact_risk_opportunity": {
                "title": "E1-2: Policies for Climate Change",
                "description": (
                    "Describe policies adopted to manage material impacts, "
                    "risks and opportunities related to climate change."
                ),
                "sub_sections": [
                    "Climate change mitigation policies",
                    "Climate change adaptation policies",
                    "Energy efficiency policies",
                ],
                "required_evidence": [
                    "climate_policies",
                    "adaptation_policies",
                ],
                "weight": 0.15,
            },
            "physical_risk": {
                "title": "E1-9: Anticipated Financial Effects from Physical Risks",
                "description": (
                    "Disclose anticipated financial effects from material "
                    "physical risks including acute and chronic hazards."
                ),
                "sub_sections": [
                    "Assets at material physical risk (monetary amounts)",
                    "Proportion of assets at risk vs total assets",
                    "Adaptation actions reducing physical risk",
                    "Location of significant assets at risk",
                ],
                "required_evidence": [
                    "physical_risk_assessment",
                    "asset_exposure_data",
                    "adaptation_measures",
                    "financial_impact_quantification",
                ],
                "weight": 0.30,
            },
            "metrics_targets": {
                "title": "E1-4 to E1-8: GHG Emissions and Energy Metrics",
                "description": (
                    "Disclose GHG emissions (Scope 1, 2, 3), energy "
                    "consumption and mix, and GHG intensity metrics."
                ),
                "sub_sections": [
                    "GHG emissions (Scope 1, 2, 3 by category)",
                    "Energy consumption and mix",
                    "GHG intensity per net revenue",
                    "GHG removals and carbon credits",
                ],
                "required_evidence": [
                    "ghg_emissions_data",
                    "energy_data",
                    "intensity_metrics",
                ],
                "weight": 0.35,
            },
        },
    },
    "eu_taxonomy": {
        "name": "EU Taxonomy Regulation - Climate Adaptation",
        "version": "Delegated Act 2021/2139 (Annex II)",
        "description": (
            "EU Taxonomy technical screening criteria for substantial "
            "contribution to climate change adaptation, including climate "
            "risk and vulnerability assessment requirements."
        ),
        "sections": {
            "screening_criteria": {
                "title": "Climate Risk and Vulnerability Assessment",
                "description": (
                    "Conduct a climate risk and vulnerability assessment "
                    "to identify material physical climate risks for the "
                    "economic activity across its expected lifetime."
                ),
                "sub_sections": [
                    "Identification of physical climate risks (acute and chronic)",
                    "Assessment of materiality of identified risks",
                    "Climate projections across scenarios and time horizons",
                    "Assessment of adaptation solutions",
                ],
                "required_evidence": [
                    "physical_risk_screening",
                    "materiality_assessment",
                    "climate_projections",
                    "adaptation_solutions",
                ],
                "weight": 0.40,
            },
            "adaptation_plan": {
                "title": "Adaptation Plan Implementation",
                "description": (
                    "Implement adaptation solutions that reduce material "
                    "physical climate risks and do not adversely affect "
                    "adaptation efforts of other parties."
                ),
                "sub_sections": [
                    "Adaptation solutions identified and assessed",
                    "Implementation timeline and milestones",
                    "Monitoring and evaluation framework",
                    "Do No Significant Harm assessment",
                ],
                "required_evidence": [
                    "adaptation_plan",
                    "implementation_timeline",
                    "monitoring_framework",
                    "dnsh_assessment",
                ],
                "weight": 0.35,
            },
            "do_no_significant_harm": {
                "title": "Do No Significant Harm (DNSH) Criteria",
                "description": (
                    "Demonstrate that the economic activity does not "
                    "significantly harm other environmental objectives."
                ),
                "sub_sections": [
                    "Climate change mitigation DNSH",
                    "Water and marine resources DNSH",
                    "Circular economy DNSH",
                    "Pollution prevention DNSH",
                    "Biodiversity and ecosystems DNSH",
                ],
                "required_evidence": [
                    "dnsh_mitigation",
                    "dnsh_water",
                    "dnsh_circular",
                    "dnsh_pollution",
                    "dnsh_biodiversity",
                ],
                "weight": 0.25,
            },
        },
    },
    "sec_climate": {
        "name": "SEC Climate-Related Financial Disclosures",
        "version": "Final Rule 2024 (Reg S-X, Reg S-K)",
        "description": (
            "SEC rules requiring climate-related disclosures in registration "
            "statements and annual reports, covering governance, strategy, "
            "risk management, climate-related financial metrics, and GHG "
            "emissions."
        ),
        "sections": {
            "governance": {
                "title": "Governance of Climate-Related Risks",
                "description": (
                    "Describe the board's oversight and management's role "
                    "in assessing and managing material climate-related risks."
                ),
                "sub_sections": [
                    "Board oversight and relevant expertise",
                    "Management's role and relevant expertise",
                    "Processes for climate risk assessment",
                ],
                "required_evidence": [
                    "board_oversight",
                    "management_processes",
                ],
                "weight": 0.20,
            },
            "strategy": {
                "title": "Strategy, Business Model, and Outlook",
                "description": (
                    "Describe material climate-related risks, their actual "
                    "and potential impacts on strategy, business model, and "
                    "outlook, and any transition plan."
                ),
                "sub_sections": [
                    "Material climate risks identified (physical and transition)",
                    "Actual and potential material impacts",
                    "Scenario analysis (if used)",
                    "Transition plan (if adopted)",
                ],
                "required_evidence": [
                    "material_climate_risks",
                    "impact_assessment",
                    "scenario_analysis",
                ],
                "weight": 0.30,
            },
            "risk_management": {
                "title": "Risk Management Processes",
                "description": (
                    "Describe processes for identifying, assessing, and "
                    "managing material climate-related risks and their "
                    "integration into overall risk management."
                ),
                "sub_sections": [
                    "Risk identification and assessment processes",
                    "Risk management and mitigation processes",
                    "Integration with enterprise risk management",
                ],
                "required_evidence": [
                    "risk_identification",
                    "risk_mitigation",
                    "erm_integration",
                ],
                "weight": 0.25,
            },
            "financial_metrics": {
                "title": "Climate-Related Financial Statement Metrics",
                "description": (
                    "Disclose climate-related financial impacts in "
                    "financial statements including expenditures, impacts "
                    "on financial estimates, and GHG emissions."
                ),
                "sub_sections": [
                    "Climate-related expenditures and capitalised costs",
                    "Impact on financial estimates and assumptions",
                    "Severe weather event costs and recovery activities",
                    "GHG emissions (Scope 1 and Scope 2)",
                ],
                "required_evidence": [
                    "financial_impacts",
                    "ghg_emissions",
                    "weather_event_costs",
                ],
                "weight": 0.25,
            },
        },
    },
    "ifrs_s2": {
        "name": "IFRS S2 Climate-related Disclosures",
        "version": "ISSB IFRS S2 (June 2023)",
        "description": (
            "IFRS S2 requires disclosure of information about "
            "climate-related risks and opportunities to enable users of "
            "financial reports to assess the effects on enterprise value."
        ),
        "sections": {
            "governance": {
                "title": "Governance",
                "description": (
                    "Disclose governance processes, controls, and procedures "
                    "used to monitor, manage, and oversee climate-related "
                    "risks and opportunities."
                ),
                "sub_sections": [
                    "Governance body oversight",
                    "Management's role in governance processes",
                ],
                "required_evidence": [
                    "governance_body_oversight",
                    "management_role",
                ],
                "weight": 0.20,
            },
            "strategy": {
                "title": "Strategy",
                "description": (
                    "Disclose climate-related risks and opportunities that "
                    "could reasonably be expected to affect the entity's "
                    "cash flows, access to finance, or cost of capital."
                ),
                "sub_sections": [
                    "Climate-related risks and opportunities",
                    "Business model and value chain effects",
                    "Strategy and decision-making",
                    "Financial position, performance, and cash flows",
                    "Climate resilience assessment",
                ],
                "required_evidence": [
                    "risk_opportunity_identification",
                    "value_chain_effects",
                    "resilience_assessment",
                    "financial_effects",
                ],
                "weight": 0.30,
            },
            "risk_management": {
                "title": "Risk Management",
                "description": (
                    "Disclose the processes used to identify, assess, "
                    "prioritise, and monitor climate-related risks and "
                    "opportunities."
                ),
                "sub_sections": [
                    "Risk identification and assessment",
                    "Risk prioritisation",
                    "Risk monitoring",
                    "Integration into overall risk management",
                ],
                "required_evidence": [
                    "risk_process",
                    "prioritisation_method",
                    "monitoring_process",
                ],
                "weight": 0.25,
            },
            "metrics_and_targets": {
                "title": "Metrics and Targets",
                "description": (
                    "Disclose metrics and targets used to measure and "
                    "manage climate-related risks and opportunities."
                ),
                "sub_sections": [
                    "Cross-industry metrics (GHG, transition risks, etc.)",
                    "Industry-based metrics",
                    "Targets set and progress",
                    "Internal carbon price (if used)",
                ],
                "required_evidence": [
                    "ghg_emissions",
                    "industry_metrics",
                    "climate_targets",
                ],
                "weight": 0.25,
            },
        },
    },
    "ngfs": {
        "name": "NGFS Climate Scenarios",
        "version": "Phase IV (2023)",
        "description": (
            "Network for Greening the Financial System climate scenarios "
            "for central banks and supervisors to assess climate-related "
            "financial risks under orderly, disorderly, and hot-house "
            "world pathways."
        ),
        "sections": {
            "scenario_overview": {
                "title": "Scenario Framework Overview",
                "description": (
                    "Overview of selected NGFS scenarios and their "
                    "assumptions regarding transition and physical risk "
                    "pathways."
                ),
                "sub_sections": [
                    "Orderly scenarios (Net Zero 2050, Below 2C)",
                    "Disorderly scenarios (Delayed Transition, Divergent Net Zero)",
                    "Hot House World scenarios (NDCs, Current Policies)",
                ],
                "required_evidence": [
                    "scenario_selection",
                    "scenario_assumptions",
                ],
                "weight": 0.25,
            },
            "physical_risk_projections": {
                "title": "Physical Risk Projections",
                "description": (
                    "Climate hazard projections under selected NGFS "
                    "scenarios including temperature, precipitation, "
                    "sea level, and extreme event frequency changes."
                ),
                "sub_sections": [
                    "Temperature change projections",
                    "Precipitation change projections",
                    "Sea level rise projections",
                    "Extreme event frequency and intensity projections",
                ],
                "required_evidence": [
                    "temperature_projections",
                    "precipitation_projections",
                    "sea_level_projections",
                    "extreme_event_projections",
                ],
                "weight": 0.30,
            },
            "financial_impact": {
                "title": "Financial Impact Assessment",
                "description": (
                    "Quantified financial impacts under each scenario, "
                    "including GDP impacts, asset value changes, insurance "
                    "losses, and stranded asset risks."
                ),
                "sub_sections": [
                    "Macro-economic impacts (GDP, productivity)",
                    "Sectoral impacts and stranded assets",
                    "Insurance and re-insurance implications",
                    "Portfolio-level Value-at-Risk under scenarios",
                ],
                "required_evidence": [
                    "financial_impact_model",
                    "var_analysis",
                    "stranded_asset_assessment",
                ],
                "weight": 0.30,
            },
            "risk_management_response": {
                "title": "Risk Management Response",
                "description": (
                    "Describe risk management responses to identified "
                    "climate risks under each scenario, including stress "
                    "testing results and adaptation strategies."
                ),
                "sub_sections": [
                    "Climate stress testing methodology",
                    "Stress test results summary",
                    "Adaptation and mitigation strategies",
                ],
                "required_evidence": [
                    "stress_test_methodology",
                    "stress_test_results",
                    "adaptation_strategies",
                ],
                "weight": 0.15,
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Recommendation Templates
# ---------------------------------------------------------------------------

# Recommendations keyed by risk level, then by category.

RECOMMENDATIONS_BY_RISK_LEVEL: Dict[str, List[Dict[str, str]]] = {
    "extreme": [
        {
            "category": "immediate_action",
            "recommendation": (
                "Implement emergency climate adaptation measures for "
                "assets identified at extreme risk within the next "
                "12 months."
            ),
            "priority": "critical",
        },
        {
            "category": "asset_relocation",
            "recommendation": (
                "Evaluate feasibility of relocating or divesting assets "
                "in extreme-risk zones where adaptation is not viable."
            ),
            "priority": "critical",
        },
        {
            "category": "insurance_review",
            "recommendation": (
                "Review and increase insurance coverage for assets in "
                "extreme-risk zones; consider parametric insurance "
                "products for climate hazards."
            ),
            "priority": "critical",
        },
        {
            "category": "supply_chain",
            "recommendation": (
                "Diversify supply chain routes and suppliers to reduce "
                "concentration risk in extreme-hazard regions."
            ),
            "priority": "critical",
        },
    ],
    "high": [
        {
            "category": "adaptation_planning",
            "recommendation": (
                "Develop and fund adaptation plans for high-risk assets "
                "within the next 18 months, incorporating nature-based "
                "solutions where feasible."
            ),
            "priority": "priority",
        },
        {
            "category": "monitoring",
            "recommendation": (
                "Deploy continuous monitoring systems for high-risk "
                "hazard types with automated early warning alerts."
            ),
            "priority": "priority",
        },
        {
            "category": "financial_planning",
            "recommendation": (
                "Integrate climate risk costs into financial planning "
                "and capital allocation processes for high-risk assets."
            ),
            "priority": "priority",
        },
    ],
    "medium": [
        {
            "category": "risk_assessment",
            "recommendation": (
                "Conduct detailed asset-level climate risk assessments "
                "for medium-risk locations to identify cost-effective "
                "adaptation measures."
            ),
            "priority": "recommended",
        },
        {
            "category": "scenario_analysis",
            "recommendation": (
                "Perform scenario analysis under SSP2-4.5 and SSP5-8.5 "
                "pathways to understand risk trajectory over time."
            ),
            "priority": "recommended",
        },
    ],
    "low": [
        {
            "category": "monitoring",
            "recommendation": (
                "Maintain periodic monitoring of climate hazard indicators "
                "for low-risk assets on an annual review cycle."
            ),
            "priority": "advisory",
        },
    ],
    "negligible": [
        {
            "category": "baseline",
            "recommendation": (
                "Document baseline climate risk profile for negligible-risk "
                "assets and include in periodic portfolio reviews."
            ),
            "priority": "routine",
        },
    ],
}

# ---------------------------------------------------------------------------
# Adaptation Measures Library
# ---------------------------------------------------------------------------

ADAPTATION_MEASURES: Dict[str, List[Dict[str, str]]] = {
    "flood": [
        {"measure": "Flood barriers and levees", "effectiveness": "high"},
        {"measure": "Elevated building foundations", "effectiveness": "high"},
        {"measure": "Sustainable urban drainage systems (SuDS)", "effectiveness": "medium"},
        {"measure": "Flood early warning systems", "effectiveness": "medium"},
        {"measure": "Wetland restoration and green infrastructure", "effectiveness": "medium"},
    ],
    "drought": [
        {"measure": "Water recycling and reclamation systems", "effectiveness": "high"},
        {"measure": "Drought-resistant crop varieties", "effectiveness": "high"},
        {"measure": "Rainwater harvesting infrastructure", "effectiveness": "medium"},
        {"measure": "Water efficiency audits and retrofits", "effectiveness": "medium"},
    ],
    "extreme_heat": [
        {"measure": "Cool roofs and reflective surfaces", "effectiveness": "high"},
        {"measure": "Urban greening and shade structures", "effectiveness": "medium"},
        {"measure": "HVAC system upgrades for extreme temperatures", "effectiveness": "high"},
        {"measure": "Heat action plans and worker safety protocols", "effectiveness": "medium"},
    ],
    "wildfire": [
        {"measure": "Defensible space and vegetation management", "effectiveness": "high"},
        {"measure": "Fire-resistant building materials", "effectiveness": "high"},
        {"measure": "Wildfire detection and early warning systems", "effectiveness": "medium"},
        {"measure": "Emergency evacuation planning", "effectiveness": "medium"},
    ],
    "sea_level_rise": [
        {"measure": "Coastal protection infrastructure (seawalls, revetments)", "effectiveness": "high"},
        {"measure": "Managed retreat planning", "effectiveness": "high"},
        {"measure": "Nature-based coastal defences (mangroves, reefs)", "effectiveness": "medium"},
        {"measure": "Saline intrusion monitoring", "effectiveness": "medium"},
    ],
    "tropical_cyclone": [
        {"measure": "Building code upgrades for wind resistance", "effectiveness": "high"},
        {"measure": "Storm surge barriers", "effectiveness": "high"},
        {"measure": "Emergency preparedness and early warning", "effectiveness": "medium"},
        {"measure": "Backup power and communication systems", "effectiveness": "medium"},
    ],
    "general": [
        {"measure": "Climate risk insurance and financial hedging", "effectiveness": "medium"},
        {"measure": "Business continuity planning with climate scenarios", "effectiveness": "medium"},
        {"measure": "Supply chain diversification", "effectiveness": "medium"},
        {"measure": "Climate-resilient infrastructure design standards", "effectiveness": "high"},
    ],
}


# ===========================================================================
# ComplianceReporterEngine
# ===========================================================================


class ComplianceReporterEngine:
    """Climate risk compliance reporting engine for TCFD, CSRD/ESRS,
    EU Taxonomy, SEC Climate, IFRS S2, and NGFS frameworks.

    Generates structured compliance reports from risk, exposure, and
    vulnerability data.  Supports five report types and five output
    formats.  Maintains an in-memory report registry with CRUD
    operations, framework compliance validation, evidence collection,
    and deterministic recommendation generation.

    Thread Safety:
        All mutable state is protected by ``threading.Lock``.

    Zero-Hallucination:
        All report content is assembled from deterministic templates
        and input data.  No LLM calls for numeric computations, risk
        classifications, or compliance scoring.

    Attributes:
        _provenance: ProvenanceTracker for SHA-256 audit trails.
        _lock: Threading lock for thread-safe state access.
        _reports: In-memory store of reports keyed by report_id.
        _type_index: Index mapping report_type to list of report_ids.
        _framework_index: Index mapping framework to list of report_ids.
        _format_index: Index mapping report_format to list of report_ids.
        _namespace_index: Index mapping namespace to list of report_ids.
        _total_generated: Running count of reports generated.
        _total_validations: Running count of compliance validations.
        _total_errors: Running count of processing errors.

    Example:
        >>> engine = ComplianceReporterEngine()
        >>> report = engine.generate_report(
        ...     report_type="physical_risk_assessment",
        ...     report_format="json",
        ...     framework="tcfd",
        ...     title="Q4 Physical Risk Report",
        ... )
        >>> print(report["compliance_score"])
    """

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(
        self,
        provenance: Optional[ProvenanceTracker] = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize ComplianceReporterEngine.

        Args:
            provenance: Optional ProvenanceTracker instance.  When None,
                a new tracker is created with the given genesis_hash.
            genesis_hash: Optional genesis hash for provenance chain
                initialization.  Ignored when provenance is provided.
        """
        # Provenance tracker -- explicit None check per project convention
        if provenance is not None:
            self._provenance = provenance
        elif genesis_hash is not None:
            self._provenance = ProvenanceTracker(genesis_hash=genesis_hash)
        else:
            self._provenance = ProvenanceTracker()

        # Threading lock
        self._lock = threading.Lock()

        # In-memory stores
        self._reports: Dict[str, ReportRecord] = {}

        # Indexes for fast lookups
        self._type_index: Dict[str, List[str]] = defaultdict(list)
        self._framework_index: Dict[str, List[str]] = defaultdict(list)
        self._format_index: Dict[str, List[str]] = defaultdict(list)
        self._namespace_index: Dict[str, List[str]] = defaultdict(list)

        # Counters
        self._total_generated: int = 0
        self._total_validations: int = 0
        self._total_errors: int = 0

        logger.info(
            "ComplianceReporterEngine initialized: "
            "frameworks=%d, report_types=%d, formats=%d, "
            "provenance_entries=%d",
            len(SUPPORTED_FRAMEWORKS),
            len(VALID_REPORT_TYPES),
            len(VALID_REPORT_FORMATS),
            self._provenance.entry_count,
        )

    # ==================================================================
    # 1. generate_report
    # ==================================================================

    def generate_report(
        self,
        report_type: str = "physical_risk_assessment",
        report_format: str = "json",
        framework: str = "tcfd",
        title: str = "",
        scope: str = "full",
        asset_ids: Optional[List[str]] = None,
        hazard_types: Optional[List[str]] = None,
        scenarios: Optional[List[str]] = None,
        time_horizons: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        risk_data: Optional[List[Dict[str, Any]]] = None,
        exposure_data: Optional[List[Dict[str, Any]]] = None,
        vulnerability_data: Optional[List[Dict[str, Any]]] = None,
        include_recommendations: bool = True,
        include_maps: bool = False,
        include_projections: bool = True,
        namespace: str = "default",
        # Pipeline-style overload: accept keyword ``results`` + ``output_format``
        results: Optional[Dict[str, Any]] = None,
        output_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a climate risk compliance report.

        Assembles a structured compliance report from the provided risk,
        exposure, and vulnerability data, formatted according to the
        requested output format and aligned with the specified regulatory
        framework.

        The method supports two calling conventions:
          1. Detailed mode: supply risk_data, exposure_data, vulnerability_data
             lists directly.
          2. Pipeline mode: supply a ``results`` dict (from pipeline stages)
             and an ``output_format`` string.

        Args:
            report_type: Type of report to generate.  Must be one of
                ``physical_risk_assessment``, ``scenario_analysis``,
                ``adaptation_screening``, ``exposure_summary``, or
                ``executive_dashboard``.
            report_format: Output format.  Must be one of ``json``,
                ``html``, ``markdown``, ``text``, or ``csv``.
            framework: Compliance framework alignment string.  Must be
                one of ``tcfd``, ``csrd_esrs``, ``eu_taxonomy``,
                ``sec_climate``, ``ifrs_s2``, or ``ngfs``.
            title: Human-readable report title.  Auto-generated when
                empty.
            scope: Scope of the report (e.g. ``full``, ``portfolio``,
                ``asset:<id>``).
            asset_ids: Optional list of asset IDs to include.
            hazard_types: Optional list of hazard type strings.
            scenarios: Optional list of scenario strings.
            time_horizons: Optional list of time horizon strings.
            parameters: Extra generation parameters.
            risk_data: List of risk index / score dictionaries.
            exposure_data: List of exposure assessment dictionaries.
            vulnerability_data: List of vulnerability score dictionaries.
            include_recommendations: Whether to include recommendations.
            include_maps: Whether to include map placeholders.
            include_projections: Whether to include projection charts.
            namespace: Tenant namespace for isolation.
            results: Pipeline-style aggregated results dict.
            output_format: Pipeline-style format override.

        Returns:
            Dictionary containing the generated report with keys:
                ``report_id``, ``report_type``, ``report_format``,
                ``framework``, ``title``, ``description``, ``scope``,
                ``namespace``, ``content``, ``report_hash``,
                ``asset_count``, ``hazard_count``, ``scenario_count``,
                ``time_horizons``, ``risk_summary``, ``recommendations``,
                ``compliance_score``, ``evidence_summary``,
                ``generated_at``, ``provenance_hash``.

        Raises:
            ValueError: If report_type, report_format, or framework is
                invalid.
        """
        start_time = time.monotonic()

        # --- Resolve pipeline-style overloads ---
        if output_format is not None:
            report_format = output_format

        # Normalise enum values
        report_type_str = _enum_value(report_type).lower()
        report_format_str = _enum_value(report_format).lower()
        framework_str = _enum_value(framework).lower()

        # --- Validate inputs ---
        self._validate_report_type(report_type_str)
        self._validate_report_format(report_format_str)
        self._validate_framework(framework_str)

        # Defaults
        asset_ids = asset_ids if asset_ids is not None else []
        hazard_types_list = [_enum_value(h) for h in hazard_types] if hazard_types else []
        scenarios_list = [_enum_value(s) for s in scenarios] if scenarios else []
        time_horizons_list = [_enum_value(t) for t in time_horizons] if time_horizons else []
        parameters = parameters if parameters is not None else {}
        risk_data = risk_data if risk_data is not None else []
        exposure_data = exposure_data if exposure_data is not None else []
        vulnerability_data = vulnerability_data if vulnerability_data is not None else []

        # --- Extract data from pipeline results if provided ---
        if results is not None:
            risk_data, exposure_data, vulnerability_data = (
                self._extract_pipeline_data(results, risk_data, exposure_data, vulnerability_data)
            )

        # --- Auto-generate title if empty ---
        if not title or not title.strip():
            title = self._auto_title(report_type_str, framework_str)

        # --- Build description ---
        description = self._build_description(
            report_type_str, framework_str, scope,
        )

        # --- Collect evidence summary ---
        evidence_summary = self._collect_evidence(
            risk_data, exposure_data, vulnerability_data,
        )

        # --- Compute risk summary ---
        risk_summary = self._compute_risk_summary(
            risk_data, exposure_data, vulnerability_data,
        )

        # --- Generate recommendations ---
        recommendations: List[str] = []
        if include_recommendations:
            recommendations = self._generate_recommendations(
                risk_summary, hazard_types_list, framework_str,
            )

        # --- Compute compliance score ---
        compliance_score = self._compute_compliance_score(
            framework_str, risk_data, exposure_data, vulnerability_data,
            evidence_summary,
        )

        # --- Render content ---
        content = self._render_content(
            report_type_str, report_format_str, framework_str,
            title, description, scope, risk_summary,
            evidence_summary, recommendations, compliance_score,
            hazard_types_list, scenarios_list, time_horizons_list,
            asset_ids, include_maps, include_projections,
        )

        # --- Compute content hash ---
        report_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # --- Build record ---
        now_iso = _utcnow_iso()
        report_id = _generate_id("RPT")

        record = ReportRecord(
            report_id=report_id,
            report_type=report_type_str,
            report_format=report_format_str,
            framework=framework_str,
            title=title,
            description=description,
            scope=scope,
            namespace=namespace,
            asset_ids=list(asset_ids),
            hazard_types=hazard_types_list,
            scenarios=scenarios_list,
            time_horizons=time_horizons_list,
            parameters=dict(parameters),
            content=content,
            report_hash=report_hash,
            asset_count=len(asset_ids),
            hazard_count=len(hazard_types_list),
            scenario_count=len(scenarios_list),
            risk_summary=risk_summary,
            recommendations=recommendations,
            compliance_score=round(compliance_score, 2),
            evidence_summary=evidence_summary,
            generated_at=now_iso,
            provenance_hash="",
        )

        # --- Provenance ---
        provenance_hash = self._record_provenance(
            entity_id=report_id,
            action="generate_report",
            data=record.to_dict(),
        )
        record.provenance_hash = provenance_hash

        # --- Store ---
        with self._lock:
            self._reports[report_id] = record
            self._type_index[report_type_str].append(report_id)
            self._framework_index[framework_str].append(report_id)
            self._format_index[report_format_str].append(report_id)
            self._namespace_index[namespace].append(report_id)
            self._total_generated += 1

        # --- Metrics ---
        _record_report_metric(report_type_str, report_format_str)

        duration = time.monotonic() - start_time
        logger.info(
            "Report generated: id=%s, type=%s, framework=%s, "
            "format=%s, compliance=%.1f, duration=%.3fs",
            report_id, report_type_str, framework_str,
            report_format_str, compliance_score, duration,
        )

        return copy.deepcopy(record.to_dict())

    # ==================================================================
    # 2. get_report
    # ==================================================================

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a report by its unique identifier.

        Args:
            report_id: The unique report identifier string.

        Returns:
            Deep-copied report dictionary, or ``None`` if not found.
        """
        if not report_id:
            return None

        with self._lock:
            record = self._reports.get(report_id)

        if record is None:
            return None

        return copy.deepcopy(record.to_dict())

    # ==================================================================
    # 3. list_reports
    # ==================================================================

    def list_reports(
        self,
        report_type: Optional[str] = None,
        framework: Optional[str] = None,
        format: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List reports with optional filtering.

        Returns reports matching all provided filter criteria.  Filters
        are combined with AND logic.  Results are ordered by generation
        time (newest first) and paginated.

        Args:
            report_type: Filter by report type string.
            framework: Filter by framework string.
            format: Filter by output format string.
            namespace: Filter by namespace.
            limit: Maximum number of results to return (default 100).
            offset: Number of results to skip (default 0).

        Returns:
            List of deep-copied report dictionaries matching filters.
        """
        with self._lock:
            candidates = list(self._reports.values())

        # Apply filters
        if report_type is not None:
            rt = _enum_value(report_type).lower()
            candidates = [r for r in candidates if r.report_type == rt]

        if framework is not None:
            fw = _enum_value(framework).lower()
            candidates = [r for r in candidates if r.framework == fw]

        if format is not None:
            fmt = _enum_value(format).lower()
            candidates = [r for r in candidates if r.report_format == fmt]

        if namespace is not None:
            candidates = [r for r in candidates if r.namespace == namespace]

        # Sort by generation time descending (newest first)
        candidates.sort(key=lambda r: r.generated_at, reverse=True)

        # Paginate
        page = candidates[offset: offset + limit]

        return [copy.deepcopy(r.to_dict()) for r in page]

    # ==================================================================
    # 4. delete_report
    # ==================================================================

    def delete_report(self, report_id: str) -> bool:
        """Delete a report from the registry.

        Args:
            report_id: The unique report identifier to delete.

        Returns:
            ``True`` if the report was found and deleted, ``False``
            otherwise.
        """
        if not report_id:
            return False

        with self._lock:
            record = self._reports.pop(report_id, None)

        if record is None:
            return False

        # Remove from indexes
        with self._lock:
            self._remove_from_index(self._type_index, record.report_type, report_id)
            self._remove_from_index(self._framework_index, record.framework, report_id)
            self._remove_from_index(self._format_index, record.report_format, report_id)
            self._remove_from_index(self._namespace_index, record.namespace, report_id)

        self._record_provenance(
            entity_id=report_id,
            action="delete_report",
            data={"report_id": report_id, "deleted": True},
        )

        logger.info("Report deleted: id=%s", report_id)
        return True

    # ==================================================================
    # 5. get_framework_template
    # ==================================================================

    def get_framework_template(self, framework: str) -> Dict[str, Any]:
        """Return the compliance template for a framework.

        Args:
            framework: Framework identifier string (e.g. ``tcfd``,
                ``csrd_esrs``).

        Returns:
            Deep-copied dictionary containing the framework template
            with name, version, description, and sections.

        Raises:
            ValueError: If the framework is not recognized.
        """
        fw = _enum_value(framework).lower()
        self._validate_framework(fw)

        template = FRAMEWORK_TEMPLATES.get(fw)
        if template is None:
            raise ValueError(f"No template found for framework '{fw}'")

        return copy.deepcopy(template)

    # ==================================================================
    # 6. list_frameworks
    # ==================================================================

    def list_frameworks(self) -> List[Dict[str, Any]]:
        """Return metadata for all supported compliance frameworks.

        Returns:
            List of dictionaries, each containing ``framework_id``,
            ``name``, ``version``, ``description``, and
            ``section_count``.
        """
        result: List[Dict[str, Any]] = []
        for fw_id, template in FRAMEWORK_TEMPLATES.items():
            result.append({
                "framework_id": fw_id,
                "name": template["name"],
                "version": template["version"],
                "description": template["description"],
                "section_count": len(template.get("sections", {})),
            })
        return result

    # ==================================================================
    # 7. validate_compliance
    # ==================================================================

    def validate_compliance(
        self,
        framework: str,
        risk_data: Optional[List[Dict[str, Any]]] = None,
        exposure_data: Optional[List[Dict[str, Any]]] = None,
        vulnerability_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Validate data completeness against a framework's requirements.

        Checks whether the provided data satisfies the evidence
        requirements defined in the framework template.  Returns a
        section-by-section compliance breakdown with an overall score.

        Args:
            framework: Framework identifier string.
            risk_data: Risk assessment data.
            exposure_data: Exposure assessment data.
            vulnerability_data: Vulnerability assessment data.

        Returns:
            Dictionary with keys:
                ``framework`` - framework identifier.
                ``overall_score`` - compliance score (0-100).
                ``overall_status`` - PASS / PARTIAL / FAIL.
                ``sections`` - per-section compliance details.
                ``missing_evidence`` - list of missing evidence items.
                ``recommendations`` - suggestions to improve compliance.
                ``validated_at`` - ISO timestamp.
                ``provenance_hash`` - SHA-256 provenance hash.

        Raises:
            ValueError: If framework is not recognized.
        """
        start_time = time.monotonic()

        fw = _enum_value(framework).lower()
        self._validate_framework(fw)

        risk_data = risk_data if risk_data is not None else []
        exposure_data = exposure_data if exposure_data is not None else []
        vulnerability_data = vulnerability_data if vulnerability_data is not None else []

        template = FRAMEWORK_TEMPLATES[fw]
        sections_result: Dict[str, Dict[str, Any]] = {}
        all_missing: List[str] = []
        weighted_score = 0.0

        # Determine what evidence is available
        available_evidence = self._determine_available_evidence(
            risk_data, exposure_data, vulnerability_data,
        )

        for section_id, section_def in template["sections"].items():
            required = section_def.get("required_evidence", [])
            weight = section_def.get("weight", 0.0)
            found = [e for e in required if e in available_evidence]
            missing = [e for e in required if e not in available_evidence]
            all_missing.extend(missing)

            section_score = (len(found) / len(required) * 100.0) if required else 100.0
            section_status = "PASS" if section_score >= 80.0 else (
                "PARTIAL" if section_score >= 40.0 else "FAIL"
            )

            sections_result[section_id] = {
                "title": section_def["title"],
                "score": round(section_score, 2),
                "status": section_status,
                "weight": weight,
                "required_evidence": required,
                "found_evidence": found,
                "missing_evidence": missing,
            }

            weighted_score += section_score * weight

        # Overall
        overall_score = round(_clamp(weighted_score, 0.0, 100.0), 2)
        overall_status = "PASS" if overall_score >= 80.0 else (
            "PARTIAL" if overall_score >= 40.0 else "FAIL"
        )

        # Compliance improvement recommendations
        compliance_recs: List[str] = []
        if all_missing:
            compliance_recs.append(
                f"Provide data for {len(all_missing)} missing evidence "
                f"item(s) to improve compliance: {', '.join(all_missing[:5])}"
                + ("..." if len(all_missing) > 5 else "")
            )
        if overall_score < 80.0:
            compliance_recs.append(
                f"Overall compliance score is {overall_score}%. "
                f"Target >= 80% for full framework alignment."
            )

        now_iso = _utcnow_iso()
        validation_id = _generate_id("VAL")

        result: Dict[str, Any] = {
            "validation_id": validation_id,
            "framework": fw,
            "framework_name": template["name"],
            "overall_score": overall_score,
            "overall_status": overall_status,
            "sections": sections_result,
            "missing_evidence": list(set(all_missing)),
            "available_evidence": sorted(available_evidence),
            "recommendations": compliance_recs,
            "validated_at": now_iso,
            "provenance_hash": "",
        }

        provenance_hash = self._record_provenance(
            entity_id=validation_id,
            action="validate_compliance",
            data=result,
        )
        result["provenance_hash"] = provenance_hash

        with self._lock:
            self._total_validations += 1

        duration = time.monotonic() - start_time
        logger.info(
            "Compliance validated: framework=%s, score=%.1f, "
            "status=%s, missing=%d, duration=%.3fs",
            fw, overall_score, overall_status, len(all_missing), duration,
        )

        return result

    # ==================================================================
    # 8. get_statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine operational statistics.

        Returns:
            Dictionary with counts and summary statistics:
                ``total_reports`` - number of reports in store.
                ``total_generated`` - total reports ever generated.
                ``total_validations`` - total compliance validations.
                ``total_errors`` - accumulated error count.
                ``report_type_distribution`` - count per report type.
                ``framework_distribution`` - count per framework.
                ``format_distribution`` - count per output format.
                ``avg_compliance_score`` - average compliance score.
                ``provenance_entries`` - provenance entry count.
        """
        with self._lock:
            reports = list(self._reports.values())
            stats: Dict[str, Any] = {
                "total_reports": len(self._reports),
                "total_generated": self._total_generated,
                "total_validations": self._total_validations,
                "total_errors": self._total_errors,
            }

        # Report type distribution
        type_dist: Dict[str, int] = {}
        for r in reports:
            type_dist[r.report_type] = type_dist.get(r.report_type, 0) + 1
        stats["report_type_distribution"] = type_dist

        # Framework distribution
        fw_dist: Dict[str, int] = {}
        for r in reports:
            fw_dist[r.framework] = fw_dist.get(r.framework, 0) + 1
        stats["framework_distribution"] = fw_dist

        # Format distribution
        fmt_dist: Dict[str, int] = {}
        for r in reports:
            fmt_dist[r.report_format] = fmt_dist.get(r.report_format, 0) + 1
        stats["format_distribution"] = fmt_dist

        # Average compliance score
        scores = [r.compliance_score for r in reports]
        stats["avg_compliance_score"] = round(_safe_mean(scores), 2)

        # Provenance entry count
        if self._provenance is not None:
            try:
                stats["provenance_entries"] = self._provenance.entry_count
            except Exception:
                stats["provenance_entries"] = 0
        else:
            stats["provenance_entries"] = 0

        return copy.deepcopy(stats)

    # ==================================================================
    # 9. clear
    # ==================================================================

    def clear(self) -> None:
        """Reset all engine state to initial (empty) condition.

        Clears the report registry, all indexes, and provenance
        tracker.  Resets all counters to zero.  Intended for testing
        and teardown.
        """
        with self._lock:
            self._reports.clear()
            self._type_index.clear()
            self._framework_index.clear()
            self._format_index.clear()
            self._namespace_index.clear()
            self._total_generated = 0
            self._total_validations = 0
            self._total_errors = 0

        if self._provenance is not None:
            try:
                self._provenance.reset()
            except Exception:
                pass

        self._record_provenance(
            entity_id="compliance-reporter",
            action="clear_engine",
            data={"cleared": True},
        )

        logger.info("ComplianceReporterEngine cleared to initial state")

    # ==================================================================
    # 10. export_reports
    # ==================================================================

    def export_reports(self) -> List[Dict[str, Any]]:
        """Export all stored reports as a list of dictionaries.

        Returns:
            List of deep-copied report dictionaries, oldest first.
        """
        with self._lock:
            reports = list(self._reports.values())

        reports.sort(key=lambda r: r.generated_at)
        return [copy.deepcopy(r.to_dict()) for r in reports]

    # ==================================================================
    # 11. import_reports
    # ==================================================================

    def import_reports(
        self,
        reports: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Import reports from a list of dictionaries.

        Existing reports with matching IDs are overwritten.

        Args:
            reports: List of report record dictionaries.

        Returns:
            Dictionary with ``imported`` count, ``skipped`` count,
            and ``errors`` list.
        """
        imported = 0
        skipped = 0
        errors: List[str] = []

        for rpt_dict in reports:
            try:
                report_id = rpt_dict.get("report_id", "")
                if not report_id:
                    report_id = _generate_id("RPT")

                record = ReportRecord(
                    report_id=report_id,
                    report_type=rpt_dict.get("report_type", "physical_risk_assessment"),
                    report_format=rpt_dict.get("report_format", "json"),
                    framework=rpt_dict.get("framework", "tcfd"),
                    title=rpt_dict.get("title", ""),
                    description=rpt_dict.get("description", ""),
                    scope=rpt_dict.get("scope", "full"),
                    namespace=rpt_dict.get("namespace", "default"),
                    asset_ids=rpt_dict.get("asset_ids", []),
                    hazard_types=rpt_dict.get("hazard_types", []),
                    scenarios=rpt_dict.get("scenarios", []),
                    time_horizons=rpt_dict.get("time_horizons", []),
                    parameters=rpt_dict.get("parameters", {}),
                    content=rpt_dict.get("content", ""),
                    report_hash=rpt_dict.get("report_hash", ""),
                    asset_count=rpt_dict.get("asset_count", 0),
                    hazard_count=rpt_dict.get("hazard_count", 0),
                    scenario_count=rpt_dict.get("scenario_count", 0),
                    risk_summary=rpt_dict.get("risk_summary", {}),
                    recommendations=rpt_dict.get("recommendations", []),
                    compliance_score=rpt_dict.get("compliance_score", 0.0),
                    evidence_summary=rpt_dict.get("evidence_summary", {}),
                    generated_at=rpt_dict.get("generated_at", _utcnow_iso()),
                    provenance_hash=rpt_dict.get("provenance_hash", ""),
                )

                with self._lock:
                    self._reports[report_id] = record
                    self._type_index[record.report_type].append(report_id)
                    self._framework_index[record.framework].append(report_id)
                    self._format_index[record.report_format].append(report_id)
                    self._namespace_index[record.namespace].append(report_id)

                imported += 1

            except (ValueError, TypeError, KeyError) as exc:
                rid = rpt_dict.get("report_id", "unknown")
                logger.warning("Import failed for report %s: %s", rid, str(exc))
                errors.append(f"Report {rid}: {str(exc)}")
                skipped += 1
                with self._lock:
                    self._total_errors += 1

        self._record_provenance(
            entity_id="compliance-reporter",
            action="import_reports",
            data={"imported": imported, "skipped": skipped},
        )

        logger.info(
            "Reports imported: imported=%d, skipped=%d, errors=%d",
            imported, skipped, len(errors),
        )

        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
        }

    # ==================================================================
    # 12. get_supported_report_types
    # ==================================================================

    def get_supported_report_types(self) -> List[str]:
        """Return the list of supported report type strings.

        Returns:
            Sorted list of valid report type strings.
        """
        return sorted(VALID_REPORT_TYPES)

    # ==================================================================
    # 13. get_supported_formats
    # ==================================================================

    def get_supported_formats(self) -> List[str]:
        """Return the list of supported output format strings.

        Returns:
            Sorted list of valid output format strings.
        """
        return sorted(VALID_REPORT_FORMATS)

    # ==================================================================
    # 14. get_adaptation_measures
    # ==================================================================

    def get_adaptation_measures(
        self,
        hazard_type: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Return adaptation measures, optionally filtered by hazard type.

        Args:
            hazard_type: Optional hazard type to filter by.  When None,
                returns all measures keyed by hazard type.

        Returns:
            Dictionary mapping hazard type (or ``general``) to a list
            of measure dictionaries with ``measure`` and
            ``effectiveness`` keys.
        """
        if hazard_type is not None:
            ht = _enum_value(hazard_type).lower()
            measures = ADAPTATION_MEASURES.get(ht, [])
            general = ADAPTATION_MEASURES.get("general", [])
            return {ht: copy.deepcopy(measures), "general": copy.deepcopy(general)}

        return copy.deepcopy(ADAPTATION_MEASURES)

    # ==================================================================
    # 15. get_recommendations
    # ==================================================================

    def get_recommendations(
        self,
        risk_level: str = "medium",
    ) -> List[Dict[str, str]]:
        """Return recommendation templates for a given risk level.

        Args:
            risk_level: Risk level string (negligible, low, medium,
                high, extreme).

        Returns:
            List of recommendation dictionaries with ``category``,
            ``recommendation``, and ``priority`` keys.
        """
        level = risk_level.lower()
        recs = RECOMMENDATIONS_BY_RISK_LEVEL.get(level, [])
        return copy.deepcopy(recs)

    # ==================================================================
    # Dunder methods
    # ==================================================================

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            String showing report count and framework count.
        """
        with self._lock:
            report_c = len(self._reports)
        return (
            f"ComplianceReporterEngine("
            f"reports={report_c}, "
            f"frameworks={len(SUPPORTED_FRAMEWORKS)}, "
            f"provenance={'on' if self._provenance else 'off'})"
        )

    def __len__(self) -> int:
        """Return the total number of stored reports.

        Returns:
            Integer count of reports in the store.
        """
        with self._lock:
            return len(self._reports)

    # ==================================================================
    # Private: Provenance
    # ==================================================================

    def _record_provenance(
        self,
        entity_id: str,
        action: str,
        data: Any,
    ) -> str:
        """Record a provenance entry and return the hash.

        Args:
            entity_id: Entity identifier for provenance.
            action: Action label (e.g. generate_report).
            data: Data payload to hash.

        Returns:
            SHA-256 provenance hash string.
        """
        if self._provenance is None:
            return _build_provenance_hash(data)

        try:
            entry = self._provenance.record(
                entity_type="compliance_report",
                action=action,
                entity_id=entity_id,
                data=data,
            )
            return getattr(entry, "hash_value", _build_provenance_hash(data))
        except Exception as exc:
            logger.warning(
                "Provenance recording failed for %s/%s: %s",
                entity_id, action, exc,
            )
            return _build_provenance_hash(data)

    # ==================================================================
    # Private: Validation helpers
    # ==================================================================

    @staticmethod
    def _validate_report_type(report_type: str) -> None:
        """Validate that report_type is a recognised value.

        Args:
            report_type: Report type string to validate.

        Raises:
            ValueError: If report_type is not recognised.
        """
        if report_type not in VALID_REPORT_TYPES:
            raise ValueError(
                f"report_type must be one of {sorted(VALID_REPORT_TYPES)}, "
                f"got '{report_type}'"
            )

    @staticmethod
    def _validate_report_format(report_format: str) -> None:
        """Validate that report_format is a recognised value.

        Args:
            report_format: Report format string to validate.

        Raises:
            ValueError: If report_format is not recognised.
        """
        if report_format not in VALID_REPORT_FORMATS:
            raise ValueError(
                f"report_format must be one of {sorted(VALID_REPORT_FORMATS)}, "
                f"got '{report_format}'"
            )

    @staticmethod
    def _validate_framework(framework: str) -> None:
        """Validate that framework is a recognised value.

        Args:
            framework: Framework string to validate.

        Raises:
            ValueError: If framework is not recognised.
        """
        if framework not in SUPPORTED_FRAMEWORKS:
            raise ValueError(
                f"framework must be one of {list(SUPPORTED_FRAMEWORKS)}, "
                f"got '{framework}'"
            )

    @staticmethod
    def _remove_from_index(
        index: Dict[str, List[str]],
        key: str,
        value: str,
    ) -> None:
        """Remove a value from an index list under a key.

        Args:
            index: The index dictionary.
            key: The index key.
            value: The value to remove.
        """
        if key in index:
            try:
                index[key].remove(value)
            except ValueError:
                pass

    # ==================================================================
    # Private: Data extraction
    # ==================================================================

    @staticmethod
    def _extract_pipeline_data(
        results: Dict[str, Any],
        risk_data: List[Dict[str, Any]],
        exposure_data: List[Dict[str, Any]],
        vulnerability_data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Extract risk, exposure, and vulnerability data from pipeline results.

        Looks for data under standard pipeline stage keys.

        Args:
            results: Pipeline results dictionary.
            risk_data: Existing risk data (extended, not replaced).
            exposure_data: Existing exposure data (extended).
            vulnerability_data: Existing vulnerability data (extended).

        Returns:
            Tuple of (risk_data, exposure_data, vulnerability_data).
        """
        # Risk data from risk_calculation stage
        risk_stage = results.get("risk_calculation", {})
        if isinstance(risk_stage, dict):
            risk_items = risk_stage.get("risk_indices", [])
            if isinstance(risk_items, list):
                risk_data = list(risk_data) + risk_items

        # Exposure data from exposure_assessment stage
        exposure_stage = results.get("exposure_assessment", {})
        if isinstance(exposure_stage, dict):
            exp_items = exposure_stage.get("assessments", [])
            if isinstance(exp_items, list):
                exposure_data = list(exposure_data) + exp_items

        # Vulnerability data from vulnerability_scoring stage
        vuln_stage = results.get("vulnerability_scoring", {})
        if isinstance(vuln_stage, dict):
            vuln_items = vuln_stage.get("scores", [])
            if isinstance(vuln_items, list):
                vulnerability_data = list(vulnerability_data) + vuln_items

        return risk_data, exposure_data, vulnerability_data

    # ==================================================================
    # Private: Auto-title generation
    # ==================================================================

    @staticmethod
    def _auto_title(report_type: str, framework: str) -> str:
        """Generate a default report title from type and framework.

        Args:
            report_type: Report type string.
            framework: Framework string.

        Returns:
            Human-readable title string.
        """
        type_labels = {
            "physical_risk_assessment": "Physical Risk Assessment",
            "scenario_analysis": "Scenario Analysis",
            "adaptation_screening": "Adaptation Screening",
            "exposure_summary": "Exposure Summary",
            "executive_dashboard": "Executive Dashboard",
        }
        fw_labels = {
            "tcfd": "TCFD",
            "csrd_esrs": "CSRD/ESRS",
            "eu_taxonomy": "EU Taxonomy",
            "sec_climate": "SEC Climate",
            "ifrs_s2": "IFRS S2",
            "ngfs": "NGFS",
        }
        type_label = type_labels.get(report_type, report_type.replace("_", " ").title())
        fw_label = fw_labels.get(framework, framework.upper())
        return f"{fw_label} {type_label} Report"

    # ==================================================================
    # Private: Description builder
    # ==================================================================

    @staticmethod
    def _build_description(
        report_type: str,
        framework: str,
        scope: str,
    ) -> str:
        """Build a report description string.

        Args:
            report_type: Report type string.
            framework: Framework string.
            scope: Report scope string.

        Returns:
            Description string.
        """
        template = FRAMEWORK_TEMPLATES.get(framework, {})
        fw_name = template.get("name", framework)
        return (
            f"Climate risk {report_type.replace('_', ' ')} report "
            f"aligned with {fw_name}, scope: {scope}."
        )

    # ==================================================================
    # Private: Evidence collection
    # ==================================================================

    @staticmethod
    def _collect_evidence(
        risk_data: List[Dict[str, Any]],
        exposure_data: List[Dict[str, Any]],
        vulnerability_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collect evidence summary from input data.

        Aggregates key statistics from the provided data for inclusion
        in the report evidence section.

        Args:
            risk_data: Risk assessment data.
            exposure_data: Exposure assessment data.
            vulnerability_data: Vulnerability assessment data.

        Returns:
            Dictionary with evidence counts and summaries.
        """
        risk_scores = [
            float(r.get("risk_score", r.get("composite_score", 0.0)))
            for r in risk_data
        ]
        exposure_scores = [
            float(e.get("composite_score", e.get("exposure_score", 0.0)))
            for e in exposure_data
        ]
        vuln_scores = [
            float(v.get("vulnerability_score", 0.0))
            for v in vulnerability_data
        ]

        # Hazard types seen
        hazard_types_seen: set = set()
        for r in risk_data:
            ht = r.get("hazard_type", "")
            if ht:
                hazard_types_seen.add(_enum_value(ht))
        for e in exposure_data:
            ht = e.get("hazard_type", "")
            if ht:
                hazard_types_seen.add(_enum_value(ht))

        return {
            "risk_data_count": len(risk_data),
            "exposure_data_count": len(exposure_data),
            "vulnerability_data_count": len(vulnerability_data),
            "risk_score_avg": round(_safe_mean(risk_scores), 2),
            "risk_score_max": round(max(risk_scores), 2) if risk_scores else 0.0,
            "risk_score_min": round(min(risk_scores), 2) if risk_scores else 0.0,
            "exposure_score_avg": round(_safe_mean(exposure_scores), 2),
            "exposure_score_max": round(max(exposure_scores), 2) if exposure_scores else 0.0,
            "vulnerability_score_avg": round(_safe_mean(vuln_scores), 2),
            "vulnerability_score_max": round(max(vuln_scores), 2) if vuln_scores else 0.0,
            "hazard_types_identified": sorted(hazard_types_seen),
            "total_data_points": len(risk_data) + len(exposure_data) + len(vulnerability_data),
        }

    # ==================================================================
    # Private: Risk summary computation
    # ==================================================================

    @staticmethod
    def _compute_risk_summary(
        risk_data: List[Dict[str, Any]],
        exposure_data: List[Dict[str, Any]],
        vulnerability_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate risk summary from input data.

        Args:
            risk_data: Risk assessment data.
            exposure_data: Exposure assessment data.
            vulnerability_data: Vulnerability assessment data.

        Returns:
            Dictionary with aggregate risk metrics.
        """
        risk_scores = [
            float(r.get("risk_score", r.get("composite_score", 0.0)))
            for r in risk_data
        ]
        vuln_scores = [
            float(v.get("vulnerability_score", 0.0))
            for v in vulnerability_data
        ]

        # Overall risk score: weighted average of risk and vulnerability
        all_scores = risk_scores + vuln_scores
        overall_score = round(_safe_mean(all_scores), 2)
        overall_level = _classify_risk_level(overall_score)

        # Risk level distribution
        level_dist: Dict[str, int] = {
            "negligible": 0, "low": 0, "medium": 0, "high": 0, "extreme": 0,
        }
        for s in all_scores:
            lvl = _classify_risk_level(s)
            level_dist[lvl] = level_dist.get(lvl, 0) + 1

        # Hazard-level breakdown
        hazard_scores: Dict[str, List[float]] = defaultdict(list)
        for r in risk_data:
            ht = _enum_value(r.get("hazard_type", "unknown"))
            hazard_scores[ht].append(
                float(r.get("risk_score", r.get("composite_score", 0.0)))
            )

        hazard_summary: Dict[str, Dict[str, Any]] = {}
        for ht, scores in hazard_scores.items():
            hazard_summary[ht] = {
                "count": len(scores),
                "avg_score": round(_safe_mean(scores), 2),
                "max_score": round(max(scores), 2) if scores else 0.0,
                "risk_level": _classify_risk_level(_safe_mean(scores)),
            }

        return {
            "overall_risk_score": overall_score,
            "overall_risk_level": overall_level,
            "total_risk_assessments": len(risk_data),
            "total_exposure_assessments": len(exposure_data),
            "total_vulnerability_scores": len(vulnerability_data),
            "risk_level_distribution": level_dist,
            "hazard_summary": hazard_summary,
            "high_risk_count": level_dist.get("high", 0) + level_dist.get("extreme", 0),
            "urgency": URGENCY_MAP.get(overall_level, "routine"),
        }

    # ==================================================================
    # Private: Recommendation generation
    # ==================================================================

    @staticmethod
    def _generate_recommendations(
        risk_summary: Dict[str, Any],
        hazard_types: List[str],
        framework: str,
    ) -> List[str]:
        """Generate adaptation and mitigation recommendations.

        Deterministic recommendation generation based on risk level
        and hazard types.  No LLM calls.

        Args:
            risk_summary: Computed risk summary.
            hazard_types: Hazard types identified.
            framework: Compliance framework.

        Returns:
            List of recommendation strings.
        """
        recommendations: List[str] = []
        overall_level = risk_summary.get("overall_risk_level", "negligible")

        # Risk-level recommendations
        level_recs = RECOMMENDATIONS_BY_RISK_LEVEL.get(overall_level, [])
        for rec in level_recs:
            recommendations.append(rec["recommendation"])

        # Hazard-specific adaptation measures
        for ht in hazard_types:
            ht_lower = _enum_value(ht).lower()
            measures = ADAPTATION_MEASURES.get(ht_lower, [])
            for m in measures[:2]:
                rec_text = (
                    f"[{ht_lower}] Consider implementing: "
                    f"{m['measure']} (effectiveness: {m['effectiveness']})"
                )
                recommendations.append(rec_text)

        # Framework-specific recommendations
        fw_recs = {
            "tcfd": (
                "Ensure TCFD-aligned disclosure covers all four pillars: "
                "Governance, Strategy, Risk Management, and Metrics & Targets."
            ),
            "csrd_esrs": (
                "Quantify anticipated financial effects from material "
                "physical risks as required by ESRS E1-9."
            ),
            "eu_taxonomy": (
                "Complete climate risk and vulnerability assessment as "
                "required by EU Taxonomy Annex II for climate adaptation "
                "substantial contribution."
            ),
            "sec_climate": (
                "Disclose material climate-related expenditures and severe "
                "weather event costs in financial statements as required "
                "by SEC climate rules."
            ),
            "ifrs_s2": (
                "Assess climate resilience under multiple scenarios as "
                "required by IFRS S2 for strategy disclosures."
            ),
            "ngfs": (
                "Conduct climate stress tests under orderly, disorderly, "
                "and hot-house world NGFS scenarios."
            ),
        }
        fw_rec = fw_recs.get(framework)
        if fw_rec:
            recommendations.append(fw_rec)

        return recommendations

    # ==================================================================
    # Private: Compliance score computation
    # ==================================================================

    @staticmethod
    def _compute_compliance_score(
        framework: str,
        risk_data: List[Dict[str, Any]],
        exposure_data: List[Dict[str, Any]],
        vulnerability_data: List[Dict[str, Any]],
        evidence_summary: Dict[str, Any],
    ) -> float:
        """Compute a compliance readiness score (0-100).

        The score reflects how complete the provided data is relative
        to the framework's requirements.  It is deterministic:
        score = data_completeness * 50 + evidence_breadth * 50.

        Args:
            framework: Framework identifier.
            risk_data: Risk data.
            exposure_data: Exposure data.
            vulnerability_data: Vulnerability data.
            evidence_summary: Evidence summary dict.

        Returns:
            Compliance score in [0, 100].
        """
        # Data completeness component (0-50)
        has_risk = 1.0 if risk_data else 0.0
        has_exposure = 1.0 if exposure_data else 0.0
        has_vuln = 1.0 if vulnerability_data else 0.0
        data_completeness = ((has_risk + has_exposure + has_vuln) / 3.0) * 50.0

        # Evidence breadth component (0-50)
        template = FRAMEWORK_TEMPLATES.get(framework, {})
        sections = template.get("sections", {})
        total_required = 0
        total_available = 0

        # Count total required evidence items and estimate available
        available_set = set()
        if risk_data:
            available_set.update([
                "physical_risk_assessment", "risk_identification",
                "risk_assessment_methodology", "material_climate_risks",
                "risk_process", "physical_risk_screening",
                "risk_identification_process",
            ])
        if exposure_data:
            available_set.update([
                "asset_exposure_data", "climate_projections",
                "scenario_analysis", "scenario_selection",
                "scenario_assumptions",
            ])
        if vulnerability_data:
            available_set.update([
                "vulnerability_assessment", "adaptation_measures",
                "resilience_assessment", "impact_assessment",
                "financial_impact_analysis",
            ])

        for section_def in sections.values():
            required = section_def.get("required_evidence", [])
            total_required += len(required)
            for e in required:
                if e in available_set:
                    total_available += 1

        evidence_breadth = 0.0
        if total_required > 0:
            evidence_breadth = (total_available / total_required) * 50.0

        return _clamp(data_completeness + evidence_breadth, 0.0, 100.0)

    # ==================================================================
    # Private: Available evidence detection
    # ==================================================================

    @staticmethod
    def _determine_available_evidence(
        risk_data: List[Dict[str, Any]],
        exposure_data: List[Dict[str, Any]],
        vulnerability_data: List[Dict[str, Any]],
    ) -> set:
        """Determine which evidence items are available from data.

        Maps the presence and content of input data to evidence item
        identifiers used by framework templates.

        Args:
            risk_data: Risk assessment data.
            exposure_data: Exposure assessment data.
            vulnerability_data: Vulnerability assessment data.

        Returns:
            Set of available evidence item strings.
        """
        available: set = set()

        if risk_data:
            available.update([
                "physical_risk_assessment",
                "physical_risk_screening",
                "risk_identification_process",
                "risk_assessment_methodology",
                "risk_identification",
                "risk_process",
                "material_climate_risks",
                "climate_risk_metrics",
            ])

        if exposure_data:
            available.update([
                "asset_exposure_data",
                "climate_projections",
                "scenario_analysis",
                "scenario_selection",
                "scenario_assumptions",
            ])

        if vulnerability_data:
            available.update([
                "vulnerability_assessment",
                "adaptation_measures",
                "adaptation_solutions",
                "resilience_assessment",
                "impact_assessment",
                "financial_impact_analysis",
            ])

        if risk_data and exposure_data:
            available.update([
                "risk_management_integration",
                "monitoring_process",
                "risk_mitigation",
                "erm_integration",
            ])

        if risk_data and vulnerability_data:
            available.update([
                "financial_effects",
                "financial_impacts",
                "financial_impact_model",
                "materiality_assessment",
            ])

        if risk_data and exposure_data and vulnerability_data:
            available.update([
                "stress_test_methodology",
                "stress_test_results",
                "adaptation_strategies",
                "value_chain_effects",
            ])

        return available

    # ==================================================================
    # Private: Content rendering
    # ==================================================================

    def _render_content(
        self,
        report_type: str,
        report_format: str,
        framework: str,
        title: str,
        description: str,
        scope: str,
        risk_summary: Dict[str, Any],
        evidence_summary: Dict[str, Any],
        recommendations: List[str],
        compliance_score: float,
        hazard_types: List[str],
        scenarios: List[str],
        time_horizons: List[str],
        asset_ids: List[str],
        include_maps: bool,
        include_projections: bool,
    ) -> str:
        """Render the report content in the specified format.

        Dispatches to format-specific renderers.

        Args:
            report_type: Report type string.
            report_format: Output format string.
            framework: Framework string.
            title: Report title.
            description: Report description.
            scope: Report scope.
            risk_summary: Computed risk summary.
            evidence_summary: Collected evidence summary.
            recommendations: List of recommendation strings.
            compliance_score: Compliance readiness score.
            hazard_types: Hazard types list.
            scenarios: Scenarios list.
            time_horizons: Time horizons list.
            asset_ids: Asset IDs list.
            include_maps: Whether to include map placeholders.
            include_projections: Whether to include projection placeholders.

        Returns:
            Rendered content string.
        """
        # Build common data structure for all renderers
        data = {
            "report_type": report_type,
            "framework": framework,
            "title": title,
            "description": description,
            "scope": scope,
            "compliance_score": compliance_score,
            "risk_summary": risk_summary,
            "evidence_summary": evidence_summary,
            "recommendations": recommendations,
            "hazard_types": hazard_types,
            "scenarios": scenarios,
            "time_horizons": time_horizons,
            "asset_count": len(asset_ids),
            "include_maps": include_maps,
            "include_projections": include_projections,
            "generated_at": _utcnow_iso(),
        }

        # Get framework template sections
        template = FRAMEWORK_TEMPLATES.get(framework, {})
        data["framework_name"] = template.get("name", framework)
        data["framework_version"] = template.get("version", "")
        data["sections"] = template.get("sections", {})

        if report_format == "json":
            return self._render_json(data)
        elif report_format == "html":
            return self._render_html(data)
        elif report_format == "markdown":
            return self._render_markdown(data)
        elif report_format == "text":
            return self._render_text(data)
        elif report_format == "csv":
            return self._render_csv(data)
        else:
            return self._render_json(data)

    # ------------------------------------------------------------------
    # JSON renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_json(data: Dict[str, Any]) -> str:
        """Render report content as formatted JSON.

        Args:
            data: Report data structure.

        Returns:
            JSON-formatted string.
        """
        return json.dumps(data, indent=2, sort_keys=False, default=str)

    # ------------------------------------------------------------------
    # HTML renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_html(data: Dict[str, Any]) -> str:
        """Render report content as self-contained HTML.

        Args:
            data: Report data structure.

        Returns:
            HTML string with inline styles.
        """
        title = data.get("title", "Climate Risk Report")
        fw_name = data.get("framework_name", "")
        fw_version = data.get("framework_version", "")
        description = data.get("description", "")
        compliance = data.get("compliance_score", 0.0)
        risk_summary = data.get("risk_summary", {})
        overall_level = risk_summary.get("overall_risk_level", "negligible")
        overall_score = risk_summary.get("overall_risk_score", 0.0)
        recommendations = data.get("recommendations", [])
        sections = data.get("sections", {})
        generated_at = data.get("generated_at", "")

        # Colour mapping for risk levels
        level_colors = {
            "negligible": "#28a745",
            "low": "#6f42c1",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "extreme": "#dc3545",
        }
        level_color = level_colors.get(overall_level, "#6c757d")

        # Build sections HTML
        sections_html = ""
        for sec_id, sec_def in sections.items():
            sec_title = sec_def.get("title", sec_id)
            sec_desc = sec_def.get("description", "")
            sub_secs = sec_def.get("sub_sections", [])
            sub_html = "".join(f"<li>{s}</li>" for s in sub_secs)
            sections_html += (
                f'<div class="section">'
                f'<h3>{sec_title}</h3>'
                f'<p>{sec_desc}</p>'
                f'<ul>{sub_html}</ul>'
                f'</div>'
            )

        # Build recommendations HTML
        recs_html = ""
        if recommendations:
            recs_items = "".join(f"<li>{r}</li>" for r in recommendations)
            recs_html = (
                f'<div class="section">'
                f'<h3>Recommendations</h3>'
                f'<ol>{recs_items}</ol>'
                f'</div>'
            )

        html = (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<title>{title}</title>'
            f'<style>'
            f'body{{font-family:Arial,sans-serif;margin:40px;color:#333}}'
            f'h1{{color:#1a5276}}'
            f'h2{{color:#2c3e50;border-bottom:2px solid #eee;padding-bottom:8px}}'
            f'h3{{color:#34495e}}'
            f'.meta{{background:#f8f9fa;padding:16px;border-radius:4px;margin:16px 0}}'
            f'.section{{margin:24px 0;padding:16px;border-left:4px solid #3498db}}'
            f'.score-badge{{display:inline-block;padding:8px 16px;border-radius:20px;'
            f'color:#fff;font-weight:bold;background:{level_color}}}'
            f'</style>'
            f'</head><body>'
            f'<h1>{title}</h1>'
            f'<div class="meta">'
            f'<p><strong>Framework:</strong> {fw_name} ({fw_version})</p>'
            f'<p><strong>Description:</strong> {description}</p>'
            f'<p><strong>Generated:</strong> {generated_at}</p>'
            f'</div>'
            f'<h2>Risk Summary</h2>'
            f'<p>Overall Risk Score: <strong>{overall_score}</strong> '
            f'<span class="score-badge">{overall_level.upper()}</span></p>'
            f'<p>Compliance Readiness Score: <strong>{compliance:.1f}%</strong></p>'
            f'<h2>Framework Sections</h2>'
            f'{sections_html}'
            f'{recs_html}'
            f'</body></html>'
        )
        return html

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_markdown(data: Dict[str, Any]) -> str:
        """Render report content as Markdown.

        Args:
            data: Report data structure.

        Returns:
            Markdown-formatted string.
        """
        lines: List[str] = []
        title = data.get("title", "Climate Risk Report")
        fw_name = data.get("framework_name", "")
        fw_version = data.get("framework_version", "")
        description = data.get("description", "")
        compliance = data.get("compliance_score", 0.0)
        risk_summary = data.get("risk_summary", {})
        overall_level = risk_summary.get("overall_risk_level", "negligible")
        overall_score = risk_summary.get("overall_risk_score", 0.0)
        recommendations = data.get("recommendations", [])
        sections = data.get("sections", {})
        generated_at = data.get("generated_at", "")

        lines.append(f"# {title}")
        lines.append("")
        lines.append(f"**Framework:** {fw_name} ({fw_version})")
        lines.append(f"**Description:** {description}")
        lines.append(f"**Generated:** {generated_at}")
        lines.append("")

        lines.append("## Risk Summary")
        lines.append("")
        lines.append(f"- **Overall Risk Score:** {overall_score}")
        lines.append(f"- **Overall Risk Level:** {overall_level.upper()}")
        lines.append(f"- **Compliance Readiness:** {compliance:.1f}%")
        lines.append(f"- **Total Risk Assessments:** {risk_summary.get('total_risk_assessments', 0)}")
        lines.append(f"- **Total Exposure Assessments:** {risk_summary.get('total_exposure_assessments', 0)}")
        lines.append(f"- **Total Vulnerability Scores:** {risk_summary.get('total_vulnerability_scores', 0)}")
        lines.append(f"- **High/Extreme Risk Count:** {risk_summary.get('high_risk_count', 0)}")
        lines.append("")

        # Risk level distribution
        level_dist = risk_summary.get("risk_level_distribution", {})
        if level_dist:
            lines.append("### Risk Level Distribution")
            lines.append("")
            lines.append("| Level | Count |")
            lines.append("|-------|-------|")
            for level, count in level_dist.items():
                lines.append(f"| {level} | {count} |")
            lines.append("")

        # Framework sections
        lines.append("## Framework Sections")
        lines.append("")
        for sec_id, sec_def in sections.items():
            sec_title = sec_def.get("title", sec_id)
            sec_desc = sec_def.get("description", "")
            sub_secs = sec_def.get("sub_sections", [])

            lines.append(f"### {sec_title}")
            lines.append("")
            lines.append(sec_desc)
            lines.append("")
            for s in sub_secs:
                lines.append(f"- {s}")
            lines.append("")

        # Recommendations
        if recommendations:
            lines.append("## Recommendations")
            lines.append("")
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Text renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_text(data: Dict[str, Any]) -> str:
        """Render report content as plain text.

        Args:
            data: Report data structure.

        Returns:
            Plain-text string.
        """
        lines: List[str] = []
        title = data.get("title", "Climate Risk Report")
        fw_name = data.get("framework_name", "")
        description = data.get("description", "")
        compliance = data.get("compliance_score", 0.0)
        risk_summary = data.get("risk_summary", {})
        overall_level = risk_summary.get("overall_risk_level", "negligible")
        overall_score = risk_summary.get("overall_risk_score", 0.0)
        recommendations = data.get("recommendations", [])
        generated_at = data.get("generated_at", "")

        separator = "=" * 72
        sub_sep = "-" * 72

        lines.append(separator)
        lines.append(f"  {title}")
        lines.append(separator)
        lines.append(f"  Framework:   {fw_name}")
        lines.append(f"  Description: {description}")
        lines.append(f"  Generated:   {generated_at}")
        lines.append(sub_sep)
        lines.append("")
        lines.append("RISK SUMMARY")
        lines.append(sub_sep)
        lines.append(f"  Overall Risk Score:  {overall_score}")
        lines.append(f"  Overall Risk Level:  {overall_level.upper()}")
        lines.append(f"  Compliance Score:    {compliance:.1f}%")
        lines.append(f"  Risk Assessments:    {risk_summary.get('total_risk_assessments', 0)}")
        lines.append(f"  Exposure Assessments: {risk_summary.get('total_exposure_assessments', 0)}")
        lines.append(f"  Vulnerability Scores: {risk_summary.get('total_vulnerability_scores', 0)}")
        lines.append(f"  High/Extreme Count:  {risk_summary.get('high_risk_count', 0)}")
        lines.append("")

        if recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append(sub_sep)
            for i, rec in enumerate(recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        lines.append(separator)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV renderer
    # ------------------------------------------------------------------

    @staticmethod
    def _render_csv(data: Dict[str, Any]) -> str:
        """Render report summary as CSV.

        Produces a CSV with one row per key metric.

        Args:
            data: Report data structure.

        Returns:
            CSV-formatted string.
        """
        lines: List[str] = []
        risk_summary = data.get("risk_summary", {})
        evidence = data.get("evidence_summary", {})

        lines.append("metric,value")
        lines.append(f"title,\"{data.get('title', '')}\"")
        lines.append(f"framework,\"{data.get('framework_name', '')}\"")
        lines.append(f"report_type,\"{data.get('report_type', '')}\"")
        lines.append(f"scope,\"{data.get('scope', '')}\"")
        lines.append(f"compliance_score,{data.get('compliance_score', 0.0)}")
        lines.append(f"overall_risk_score,{risk_summary.get('overall_risk_score', 0.0)}")
        lines.append(f"overall_risk_level,\"{risk_summary.get('overall_risk_level', '')}\"")
        lines.append(f"total_risk_assessments,{risk_summary.get('total_risk_assessments', 0)}")
        lines.append(f"total_exposure_assessments,{risk_summary.get('total_exposure_assessments', 0)}")
        lines.append(f"total_vulnerability_scores,{risk_summary.get('total_vulnerability_scores', 0)}")
        lines.append(f"high_risk_count,{risk_summary.get('high_risk_count', 0)}")
        lines.append(f"risk_score_avg,{evidence.get('risk_score_avg', 0.0)}")
        lines.append(f"risk_score_max,{evidence.get('risk_score_max', 0.0)}")
        lines.append(f"exposure_score_avg,{evidence.get('exposure_score_avg', 0.0)}")
        lines.append(f"vulnerability_score_avg,{evidence.get('vulnerability_score_avg', 0.0)}")
        lines.append(f"asset_count,{data.get('asset_count', 0)}")
        lines.append(f"hazard_types,\"{';'.join(data.get('hazard_types', []))}\"")
        lines.append(f"generated_at,\"{data.get('generated_at', '')}\"")
        lines.append(f"recommendation_count,{len(data.get('recommendations', []))}")

        # Risk level distribution rows
        level_dist = risk_summary.get("risk_level_distribution", {})
        for level, count in level_dist.items():
            lines.append(f"risk_level_{level},{count}")

        return "\n".join(lines)
