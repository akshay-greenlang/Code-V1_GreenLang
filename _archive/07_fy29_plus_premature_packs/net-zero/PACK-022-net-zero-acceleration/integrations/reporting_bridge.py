# -*- coding: utf-8 -*-
"""
ReportingBridge - Cross-Framework Reporting with ISAE 3410 Assurance for PACK-022
====================================================================================

Extended reporting bridge with ISAE 3410 assurance report mapping, SDA pathway
disclosure support, and multi-entity consolidation for cross-framework reporting.
Maps net-zero data to CDP Climate Change, TCFD, ESRS E1, GHG Protocol, and
ISAE 3410 assurance disclosures.

Framework Mappings:
    CDP Climate Change  -- Sections C4 (Targets), C6 (Emissions), C7 (Breakdown)
    TCFD                -- Metrics & Targets pillar
    ESRS E1             -- Climate change disclosure requirements
    GHG Protocol        -- Corporate Standard and Scope 3 Standard
    ISAE 3410           -- Assurance engagement on GHG statements

Functions:
    - map_to_cdp()                     -- Map to CDP Climate Change
    - map_to_tcfd()                    -- Map to TCFD Metrics & Targets
    - map_to_esrs_e1()                 -- Map to ESRS E1 disclosures
    - map_to_ghg_protocol()            -- Map to GHG Protocol
    - map_to_assurance_report()        -- Map to ISAE 3410 assurance
    - generate_multi_framework_report()-- Generate consolidated report

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-022 Net Zero Acceleration Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Agent Stubs
# ---------------------------------------------------------------------------

class _AgentStub:
    """Stub for unavailable reporting app modules."""

    def __init__(self, app_name: str) -> None:
        self._app_name = app_name
        self._available = False

    def __getattr__(self, name: str) -> Any:
        def _stub_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "app": self._app_name,
                "method": name,
                "status": "degraded",
                "message": f"{self._app_name} not available, using stub",
            }
        return _stub_method

def _try_import_app(app_id: str, module_path: str) -> Any:
    """Try to import a reporting app with graceful fallback.

    Args:
        app_id: App identifier.
        module_path: Python module path.

    Returns:
        Imported module or _AgentStub if unavailable.
    """
    try:
        import importlib

        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("App %s not available, using stub", app_id)
        return _AgentStub(app_id)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportingFramework(str, Enum):
    """Supported reporting frameworks."""

    CDP_CLIMATE = "cdp_climate"
    TCFD = "tcfd"
    ESRS_E1 = "esrs_e1"
    GHG_PROTOCOL = "ghg_protocol"
    ISAE_3410 = "isae_3410"

class MappingStatus(str, Enum):
    """Framework mapping status."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    NOT_STARTED = "not_started"
    FAILED = "failed"

class AssuranceLevel(str, Enum):
    """ISAE 3410 assurance levels."""

    LIMITED = "limited"
    REASONABLE = "reasonable"
    NONE = "none"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class ReportingBridgeConfig(BaseModel):
    """Configuration for the Reporting Bridge."""

    pack_id: str = Field(default="PACK-022")
    enable_provenance: bool = Field(default=True)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    organization_name: str = Field(default="")
    frameworks_enabled: List[str] = Field(
        default_factory=lambda: [
            "cdp_climate", "tcfd", "esrs_e1", "ghg_protocol", "isae_3410",
        ],
    )
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    multi_entity: bool = Field(default=False)

class FrameworkMappingResult(BaseModel):
    """Result of mapping net-zero data to a single framework."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    framework: str = Field(default="")
    mapping_status: MappingStatus = Field(default=MappingStatus.NOT_STARTED)
    sections_mapped: List[Dict[str, Any]] = Field(default_factory=list)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    data_gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class AssuranceReportResult(BaseModel):
    """Result of ISAE 3410 assurance mapping."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    scope_of_engagement: List[str] = Field(default_factory=list)
    subject_matter: List[Dict[str, Any]] = Field(default_factory=list)
    criteria_used: List[str] = Field(default_factory=list)
    evidence_gathered: List[str] = Field(default_factory=list)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    recommendations: List[str] = Field(default_factory=list)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

class MultiFrameworkReportResult(BaseModel):
    """Result of multi-framework report generation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    frameworks_mapped: List[str] = Field(default_factory=list)
    framework_results: Dict[str, FrameworkMappingResult] = Field(default_factory=dict)
    assurance_result: Optional[AssuranceReportResult] = Field(None)
    overall_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cross_framework_consistency: bool = Field(default=False)
    total_data_gaps: int = Field(default=0)
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# App Routing
# ---------------------------------------------------------------------------

REPORTING_APPS: Dict[str, Dict[str, str]] = {
    "cdp_app": {"name": "GL-CDP-APP", "module": "greenlang.apps.cdp"},
    "tcfd_app": {"name": "GL-TCFD-APP", "module": "greenlang.apps.tcfd"},
    "csrd_app": {"name": "GL-CSRD-APP", "module": "greenlang.apps.csrd"},
    "ghg_app": {"name": "GL-GHG-APP", "module": "greenlang.apps.ghg"},
    "iso14064_app": {"name": "GL-ISO14064-APP", "module": "greenlang.apps.iso14064"},
}

# CDP Climate Change sections relevant to net-zero
CDP_SECTIONS: Dict[str, Dict[str, Any]] = {
    "C4": {
        "name": "Targets and Performance",
        "subsections": [
            "C4.1 - Emissions reduction targets",
            "C4.1a - Absolute target details",
            "C4.1b - Intensity target details",
            "C4.2 - Net-zero target",
            "C4.2a - Net-zero target details",
            "C4.2b - Progress to net-zero",
            "C4.2c - SDA pathway progress",
        ],
        "net_zero_fields": [
            "target_year", "base_year", "pathway", "scopes_covered",
            "reduction_pct", "offset_strategy", "interim_targets",
            "sda_sector", "sda_convergence_year",
        ],
    },
    "C6": {
        "name": "Emissions Data",
        "subsections": [
            "C6.1 - Scope 1 emissions",
            "C6.2 - Scope 1 breakdown",
            "C6.3 - Scope 2 location-based",
            "C6.4 - Scope 2 market-based",
            "C6.5 - Scope 3 emissions",
        ],
        "net_zero_fields": [
            "scope1_tco2e", "scope2_location_tco2e", "scope2_market_tco2e",
            "scope3_tco2e", "scope3_by_category",
        ],
    },
    "C7": {
        "name": "Emissions Breakdown",
        "subsections": [
            "C7.1 - Scope 1 breakdown by GHG type",
            "C7.2 - Scope 1 breakdown by facility",
            "C7.3a - Scope 3 Category 1",
            "C7.5 - Scope 3 gross total",
            "C7.9 - Scope 3 breakdown methodology",
        ],
        "net_zero_fields": [
            "emissions_by_ghg", "emissions_by_facility",
            "scope3_methodology", "scope3_category_details",
        ],
    },
}

# TCFD Metrics & Targets disclosures
TCFD_DISCLOSURES: Dict[str, Dict[str, Any]] = {
    "metrics_a": {
        "name": "GHG emissions Scope 1/2/3",
        "fields": ["scope1_tco2e", "scope2_tco2e", "scope3_tco2e"],
    },
    "metrics_b": {
        "name": "Climate-related risks and opportunities",
        "fields": ["transition_risks", "physical_risks", "opportunities"],
    },
    "targets_a": {
        "name": "Targets to manage climate risks",
        "fields": ["near_term_target", "long_term_target", "net_zero_target"],
    },
    "targets_b": {
        "name": "Performance against targets",
        "fields": ["reduction_achieved_pct", "on_track", "gap_analysis"],
    },
    "targets_c": {
        "name": "Transition plan metrics",
        "fields": ["sda_pathway", "temperature_score", "supplier_engagement_pct"],
    },
}

# ESRS E1 disclosure requirements
ESRS_E1_DISCLOSURES: Dict[str, Dict[str, Any]] = {
    "E1-1": {"name": "Transition plan", "fields": ["targets", "actions", "resources", "timeline"]},
    "E1-2": {"name": "Policies related to climate change mitigation", "fields": ["policies"]},
    "E1-3": {"name": "Actions and resources", "fields": ["actions", "capex", "opex"]},
    "E1-4": {"name": "Targets related to climate change", "fields": ["targets", "base_year", "pathway"]},
    "E1-5": {"name": "Energy consumption and mix", "fields": ["energy_consumption", "renewable_pct"]},
    "E1-6": {"name": "Scope 1/2/3 emissions", "fields": ["scope1", "scope2_location", "scope2_market", "scope3"]},
    "E1-7": {"name": "GHG removals and carbon credits", "fields": ["removals", "credits", "offsets"]},
    "E1-8": {"name": "Internal carbon pricing", "fields": ["carbon_price", "methodology"]},
    "E1-9": {"name": "Anticipated financial effects", "fields": ["financial_effects"]},
}

# ISAE 3410 assurance engagement structure
ISAE_3410_SECTIONS: Dict[str, Dict[str, Any]] = {
    "scope_engagement": {
        "name": "Scope of the engagement",
        "fields": [
            "reporting_entity", "reporting_period", "consolidation_approach",
            "scopes_included", "assurance_level",
        ],
    },
    "subject_matter": {
        "name": "GHG statement subject matter",
        "fields": [
            "scope1_tco2e", "scope2_tco2e", "scope3_tco2e", "total_tco2e",
            "base_year_emissions", "reduction_targets",
        ],
    },
    "criteria": {
        "name": "Applicable criteria",
        "fields": [
            "ghg_protocol_corporate", "ghg_protocol_scope3", "iso14064_1",
            "emission_factor_sources",
        ],
    },
    "evidence": {
        "name": "Evidence and procedures",
        "fields": [
            "analytical_procedures", "inquiry", "inspection",
            "recalculation", "observation", "data_tracing",
        ],
    },
    "findings": {
        "name": "Findings and conclusion",
        "fields": [
            "material_misstatements", "data_quality_issues",
            "boundary_completeness", "methodology_consistency",
        ],
    },
}

# ---------------------------------------------------------------------------
# ReportingBridge
# ---------------------------------------------------------------------------

class ReportingBridge:
    """Cross-framework reporting bridge for PACK-022 Net Zero Acceleration.

    Extended reporting bridge with ISAE 3410 assurance support, SDA pathway
    disclosure, and multi-entity consolidation.

    Attributes:
        config: Bridge configuration.
        _apps: Dict of loaded reporting app modules/stubs.

    Example:
        >>> bridge = ReportingBridge(ReportingBridgeConfig(reporting_year=2025))
        >>> cdp = bridge.map_to_cdp(net_zero_data)
        >>> assert cdp.mapping_status == MappingStatus.COMPLETE
    """

    def __init__(self, config: Optional[ReportingBridgeConfig] = None) -> None:
        """Initialize ReportingBridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or ReportingBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._apps: Dict[str, Any] = {}
        for app_id, info in REPORTING_APPS.items():
            self._apps[app_id] = _try_import_app(app_id, info["module"])

        available = sum(
            1 for a in self._apps.values() if not isinstance(a, _AgentStub)
        )
        self.logger.info(
            "ReportingBridge initialized: %d/%d apps available, year=%d, assurance=%s",
            available, len(self._apps),
            self.config.reporting_year, self.config.assurance_level.value,
        )

    # -------------------------------------------------------------------------
    # Framework Mapping Methods
    # -------------------------------------------------------------------------

    def map_to_cdp(
        self,
        net_zero_data: Dict[str, Any],
    ) -> FrameworkMappingResult:
        """Map net-zero data to CDP Climate Change questionnaire sections.

        Maps to CDP sections C4 (Targets), C6 (Emissions), C7 (Breakdown).
        Extended with SDA pathway fields for PACK-022.

        Args:
            net_zero_data: Dict with emissions, targets, and pathway data.

        Returns:
            FrameworkMappingResult with CDP mapping.
        """
        return self._map_framework(
            ReportingFramework.CDP_CLIMATE,
            CDP_SECTIONS,
            net_zero_data,
            key_field="net_zero_fields",
        )

    def map_to_tcfd(
        self,
        net_zero_data: Dict[str, Any],
    ) -> FrameworkMappingResult:
        """Map net-zero data to TCFD Metrics & Targets pillar.

        Extended with transition plan metrics for PACK-022.

        Args:
            net_zero_data: Dict with emissions, targets, and pathway data.

        Returns:
            FrameworkMappingResult with TCFD mapping.
        """
        return self._map_framework(
            ReportingFramework.TCFD,
            TCFD_DISCLOSURES,
            net_zero_data,
            key_field="fields",
        )

    def map_to_esrs_e1(
        self,
        net_zero_data: Dict[str, Any],
    ) -> FrameworkMappingResult:
        """Map net-zero data to ESRS E1 (Climate) disclosures.

        Args:
            net_zero_data: Dict with emissions, targets, and pathway data.

        Returns:
            FrameworkMappingResult with ESRS E1 mapping.
        """
        return self._map_framework(
            ReportingFramework.ESRS_E1,
            ESRS_E1_DISCLOSURES,
            net_zero_data,
            key_field="fields",
        )

    def map_to_ghg_protocol(
        self,
        net_zero_data: Dict[str, Any],
    ) -> FrameworkMappingResult:
        """Map net-zero data to GHG Protocol Corporate Standard.

        Args:
            net_zero_data: Dict with emissions data.

        Returns:
            FrameworkMappingResult with GHG Protocol mapping.
        """
        start = time.monotonic()
        result = FrameworkMappingResult(
            framework=ReportingFramework.GHG_PROTOCOL.value,
        )

        try:
            ghg_fields = [
                "organizational_boundary", "consolidation_approach",
                "scope1_tco2e", "scope2_location_tco2e", "scope2_market_tco2e",
                "scope3_tco2e", "scope3_by_category", "base_year",
                "base_year_emissions", "recalculation_policy",
                "emission_factors_source", "methodology_description",
            ]

            data_gaps: List[str] = []
            mapped_count = 0

            for field in ghg_fields:
                if field in net_zero_data and net_zero_data[field] is not None:
                    mapped_count += 1
                else:
                    data_gaps.append(f"GHG Protocol: {field}")

            result.sections_mapped = [{
                "section": "GHG Protocol Corporate Standard",
                "fields_mapped": mapped_count,
                "fields_total": len(ghg_fields),
            }]
            result.data_gaps = data_gaps
            result.completeness_pct = round(
                (mapped_count / len(ghg_fields) * 100.0) if ghg_fields else 0.0, 1
            )

            result.mapping_status = self._determine_mapping_status(
                result.completeness_pct
            )
            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            result.mapping_status = MappingStatus.FAILED
            self.logger.error("GHG Protocol mapping failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Assurance Mapping
    # -------------------------------------------------------------------------

    def map_to_assurance_report(
        self,
        net_zero_data: Dict[str, Any],
    ) -> AssuranceReportResult:
        """Map net-zero data to ISAE 3410 assurance engagement structure.

        Evaluates readiness for limited or reasonable assurance of
        GHG statements based on data availability, methodology
        documentation, and control environment.

        Args:
            net_zero_data: Dict with emissions, methodology, and controls data.

        Returns:
            AssuranceReportResult with readiness assessment.
        """
        start = time.monotonic()
        result = AssuranceReportResult(
            assurance_level=self.config.assurance_level,
        )

        try:
            # Evaluate scope of engagement
            scope_fields = ISAE_3410_SECTIONS["scope_engagement"]["fields"]
            scope_available = [
                f for f in scope_fields
                if f in net_zero_data and net_zero_data[f] is not None
            ]
            result.scope_of_engagement = scope_available

            # Evaluate subject matter
            sm_fields = ISAE_3410_SECTIONS["subject_matter"]["fields"]
            subject_matter = []
            for field in sm_fields:
                value = net_zero_data.get(field)
                subject_matter.append({
                    "field": field,
                    "available": value is not None,
                    "value": value,
                })
            result.subject_matter = subject_matter

            # Evaluate criteria
            criteria_fields = ISAE_3410_SECTIONS["criteria"]["fields"]
            result.criteria_used = [
                f for f in criteria_fields
                if f in net_zero_data and net_zero_data[f] is not None
            ]

            # Evaluate evidence
            evidence_fields = ISAE_3410_SECTIONS["evidence"]["fields"]
            result.evidence_gathered = [
                f for f in evidence_fields
                if f in net_zero_data and net_zero_data[f] is not None
            ]

            # Evaluate findings
            findings: List[Dict[str, Any]] = []
            findings_fields = ISAE_3410_SECTIONS["findings"]["fields"]
            for field in findings_fields:
                value = net_zero_data.get(field)
                if value is not None:
                    findings.append({
                        "area": field,
                        "finding": value if isinstance(value, str) else "Data available",
                        "severity": "info",
                    })
                else:
                    findings.append({
                        "area": field,
                        "finding": f"Missing: {field}",
                        "severity": "warning",
                    })
            result.findings = findings

            # Calculate readiness score
            total_fields = sum(
                len(section["fields"])
                for section in ISAE_3410_SECTIONS.values()
            )
            available_count = (
                len(scope_available)
                + sum(1 for sm in subject_matter if sm["available"])
                + len(result.criteria_used)
                + len(result.evidence_gathered)
                + sum(1 for f in findings if f["severity"] == "info")
            )
            if total_fields > 0:
                result.readiness_score = round(
                    (available_count / total_fields) * 100.0, 1
                )

            # Recommendations based on assurance level
            if self.config.assurance_level == AssuranceLevel.REASONABLE:
                if result.readiness_score < 80.0:
                    result.recommendations.append(
                        "Readiness score below 80% for reasonable assurance; "
                        "consider starting with limited assurance"
                    )
                if not result.criteria_used:
                    result.recommendations.append(
                        "Document applicable criteria (GHG Protocol, ISO 14064-1) "
                        "for reasonable assurance"
                    )
            elif self.config.assurance_level == AssuranceLevel.LIMITED:
                if result.readiness_score < 50.0:
                    result.recommendations.append(
                        "Readiness score below 50% for limited assurance; "
                        "complete subject matter documentation first"
                    )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Assurance mapping failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Multi-Framework Report
    # -------------------------------------------------------------------------

    def generate_multi_framework_report(
        self,
        net_zero_data: Dict[str, Any],
    ) -> MultiFrameworkReportResult:
        """Generate a consolidated multi-framework report.

        Extended with ISAE 3410 assurance for PACK-022.

        Args:
            net_zero_data: Dict with all net-zero data.

        Returns:
            MultiFrameworkReportResult with all framework mappings.
        """
        start = time.monotonic()
        result = MultiFrameworkReportResult()

        try:
            framework_handlers = {
                ReportingFramework.CDP_CLIMATE.value: self.map_to_cdp,
                ReportingFramework.TCFD.value: self.map_to_tcfd,
                ReportingFramework.ESRS_E1.value: self.map_to_esrs_e1,
                ReportingFramework.GHG_PROTOCOL.value: self.map_to_ghg_protocol,
            }

            total_completeness = 0.0
            total_gaps = 0
            mapped_frameworks: List[str] = []

            for fw_name in self.config.frameworks_enabled:
                if fw_name == ReportingFramework.ISAE_3410.value:
                    continue  # Handled separately
                handler = framework_handlers.get(fw_name)
                if handler is None:
                    continue

                fw_result = handler(net_zero_data)
                result.framework_results[fw_name] = fw_result
                mapped_frameworks.append(fw_name)
                total_completeness += fw_result.completeness_pct
                total_gaps += len(fw_result.data_gaps)

            result.frameworks_mapped = mapped_frameworks
            if mapped_frameworks:
                result.overall_completeness_pct = round(
                    total_completeness / len(mapped_frameworks), 1
                )
            result.total_data_gaps = total_gaps

            # ISAE 3410 assurance if enabled
            if ReportingFramework.ISAE_3410.value in self.config.frameworks_enabled:
                result.assurance_result = self.map_to_assurance_report(net_zero_data)
                mapped_frameworks.append(ReportingFramework.ISAE_3410.value)

            # Cross-framework consistency check
            emissions_values: set = set()
            for fw_name, fw_result in result.framework_results.items():
                for section in fw_result.sections_mapped:
                    if "scope1_tco2e" in section.get("mapped_fields", []):
                        emissions_values.add(fw_name)
            result.cross_framework_consistency = (
                len(emissions_values) <= 1
                or len(emissions_values) == len([
                    f for f in mapped_frameworks
                    if f != ReportingFramework.ISAE_3410.value
                ])
            )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            self.logger.error("Multi-framework report failed: %s", exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    # -------------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------------

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with app availability information.
        """
        available = sum(
            1 for a in self._apps.values() if not isinstance(a, _AgentStub)
        )
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "assurance_level": self.config.assurance_level.value,
            "frameworks_enabled": self.config.frameworks_enabled,
            "total_apps": len(self._apps),
            "available_apps": available,
            "cdp_sections": len(CDP_SECTIONS),
            "tcfd_disclosures": len(TCFD_DISCLOSURES),
            "esrs_e1_disclosures": len(ESRS_E1_DISCLOSURES),
            "isae_3410_sections": len(ISAE_3410_SECTIONS),
        }

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _map_framework(
        self,
        framework: ReportingFramework,
        sections_def: Dict[str, Dict[str, Any]],
        net_zero_data: Dict[str, Any],
        key_field: str = "fields",
    ) -> FrameworkMappingResult:
        """Generic framework mapping implementation.

        Args:
            framework: Target framework.
            sections_def: Section definitions with field lists.
            net_zero_data: Data to map.
            key_field: Key name for the fields list in section definitions.

        Returns:
            FrameworkMappingResult with mapping.
        """
        start = time.monotonic()
        result = FrameworkMappingResult(framework=framework.value)

        try:
            sections: List[Dict[str, Any]] = []
            data_gaps: List[str] = []
            total_fields = 0
            mapped_fields = 0

            for section_id, section_info in sections_def.items():
                field_list = section_info.get(key_field, [])
                section_mapped = []
                for field in field_list:
                    total_fields += 1
                    if field in net_zero_data and net_zero_data[field] is not None:
                        section_mapped.append(field)
                        mapped_fields += 1
                    else:
                        data_gaps.append(
                            f"{framework.value.upper()} {section_id}: {field}"
                        )

                sections.append({
                    "section_id": section_id,
                    "section_name": section_info.get("name", section_id),
                    "fields_mapped": len(section_mapped),
                    "fields_total": len(field_list),
                    "mapped_fields": section_mapped,
                })

            result.sections_mapped = sections
            result.data_gaps = data_gaps
            result.completeness_pct = round(
                (mapped_fields / total_fields * 100.0) if total_fields > 0 else 0.0, 1
            )

            result.mapping_status = self._determine_mapping_status(
                result.completeness_pct
            )

            if data_gaps:
                result.recommendations.append(
                    f"Complete {len(data_gaps)} missing {framework.value} data points"
                )

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            result.mapping_status = MappingStatus.FAILED
            self.logger.error("%s mapping failed: %s", framework.value, exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)
        return result

    @staticmethod
    def _determine_mapping_status(completeness_pct: float) -> MappingStatus:
        """Determine mapping status from completeness percentage.

        Args:
            completeness_pct: Completeness percentage.

        Returns:
            MappingStatus enum value.
        """
        if completeness_pct >= 80.0:
            return MappingStatus.COMPLETE
        elif completeness_pct >= 40.0:
            return MappingStatus.PARTIAL
        else:
            return MappingStatus.NOT_STARTED
