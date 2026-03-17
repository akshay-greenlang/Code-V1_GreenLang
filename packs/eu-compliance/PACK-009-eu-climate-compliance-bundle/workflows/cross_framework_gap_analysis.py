# -*- coding: utf-8 -*-
"""
Cross-Framework Gap Analysis Workflow
==========================================

Four-phase workflow that runs gap analysis across all four constituent
regulation packs (CSRD, CBAM, EU Taxonomy, EUDR), maps gaps across
frameworks, scores them by severity and cross-regulation breadth,
and generates a unified remediation plan with timeline.

Phases:
    1. IndividualGapScans - Run gap analysis in each constituent pack
    2. CrossMapping - Map gaps across frameworks
    3. ImpactScoring - Score by severity and cross-regulation breadth
    4. RemediationPlanning - Generate unified remediation plan

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import math
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Status of a workflow phase."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"


class RegulationPack(str, Enum):
    """Constituent regulation packs in the bundle."""
    CSRD = "CSRD"
    CBAM = "CBAM"
    EU_TAXONOMY = "EU_TAXONOMY"
    EUDR = "EUDR"


class GapSeverity(str, Enum):
    """Severity of a compliance gap."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFORMATIONAL = "INFORMATIONAL"


class GapCategory(str, Enum):
    """Category of compliance gap."""
    DATA_MISSING = "DATA_MISSING"
    PROCESS_MISSING = "PROCESS_MISSING"
    CONTROL_MISSING = "CONTROL_MISSING"
    DOCUMENTATION_GAP = "DOCUMENTATION_GAP"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    GOVERNANCE_GAP = "GOVERNANCE_GAP"
    TECHNOLOGY_GAP = "TECHNOLOGY_GAP"
    EXPERTISE_GAP = "EXPERTISE_GAP"


class RemediationPriority(str, Enum):
    """Priority for remediation actions."""
    IMMEDIATE = "IMMEDIATE"
    SHORT_TERM = "SHORT_TERM"
    MEDIUM_TERM = "MEDIUM_TERM"
    LONG_TERM = "LONG_TERM"


# =============================================================================
# GAP ANALYSIS DEFINITIONS
# =============================================================================


PACK_GAP_CATEGORIES: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"gap_id": "CSRD-G01", "area": "ghg_emissions_measurement", "category": "DATA_MISSING", "requirement": "ESRS E1-6: Scope 1, 2, 3 emissions measurement", "fields_needed": ["scope1_tco2e", "scope2_tco2e", "scope3_tco2e"]},
        {"gap_id": "CSRD-G02", "area": "energy_data", "category": "DATA_MISSING", "requirement": "ESRS E1-5: Energy consumption and mix reporting", "fields_needed": ["energy_mwh", "renewable_pct"]},
        {"gap_id": "CSRD-G03", "area": "double_materiality", "category": "PROCESS_MISSING", "requirement": "ESRS 1: Double materiality assessment process", "fields_needed": ["materiality_assessment"]},
        {"gap_id": "CSRD-G04", "area": "transition_plan", "category": "DOCUMENTATION_GAP", "requirement": "ESRS E1-1: Climate transition plan", "fields_needed": ["transition_plan_text"]},
        {"gap_id": "CSRD-G05", "area": "governance_structure", "category": "GOVERNANCE_GAP", "requirement": "ESRS G1: Sustainability governance structure", "fields_needed": ["governance_doc"]},
        {"gap_id": "CSRD-G06", "area": "assurance_readiness", "category": "CONTROL_MISSING", "requirement": "CSRD Art. 34: Limited assurance readiness", "fields_needed": ["assurance_status"]},
        {"gap_id": "CSRD-G07", "area": "data_management", "category": "TECHNOLOGY_GAP", "requirement": "Systematic sustainability data collection", "fields_needed": ["data_system_status"]},
        {"gap_id": "CSRD-G08", "area": "value_chain_data", "category": "CAPABILITY_GAP", "requirement": "ESRS: Value chain data collection", "fields_needed": ["value_chain_data"]},
        {"gap_id": "CSRD-G09", "area": "biodiversity_data", "category": "DATA_MISSING", "requirement": "ESRS E4: Biodiversity metrics", "fields_needed": ["biodiversity_assessment"]},
        {"gap_id": "CSRD-G10", "area": "social_metrics", "category": "DATA_MISSING", "requirement": "ESRS S1: Workforce metrics", "fields_needed": ["workforce_data"]},
    ],
    RegulationPack.CBAM.value: [
        {"gap_id": "CBAM-G01", "area": "embedded_emissions", "category": "DATA_MISSING", "requirement": "Art. 7: Actual embedded emissions data", "fields_needed": ["scope1_tco2e", "scope2_tco2e", "production_volume"]},
        {"gap_id": "CBAM-G02", "area": "supplier_engagement", "category": "PROCESS_MISSING", "requirement": "Art. 8: Supplier emissions data collection process", "fields_needed": ["supplier_data"]},
        {"gap_id": "CBAM-G03", "area": "cn_classification", "category": "CAPABILITY_GAP", "requirement": "CN code classification expertise", "fields_needed": ["cn_codes"]},
        {"gap_id": "CBAM-G04", "area": "verification_process", "category": "CONTROL_MISSING", "requirement": "Art. 8: Emissions verification process", "fields_needed": ["verification_status"]},
        {"gap_id": "CBAM-G05", "area": "reporting_system", "category": "TECHNOLOGY_GAP", "requirement": "Quarterly CBAM reporting system", "fields_needed": ["reporting_system_status"]},
        {"gap_id": "CBAM-G06", "area": "certificate_mgmt", "category": "PROCESS_MISSING", "requirement": "Art. 22: Certificate purchase and surrender process", "fields_needed": ["certificate_process"]},
        {"gap_id": "CBAM-G07", "area": "carbon_price_tracking", "category": "DATA_MISSING", "requirement": "Art. 9: Origin country carbon price tracking", "fields_needed": ["carbon_price_origin"]},
        {"gap_id": "CBAM-G08", "area": "declarant_registration", "category": "PROCESS_MISSING", "requirement": "Art. 5: Authorized declarant registration", "fields_needed": ["declarant_status"]},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"gap_id": "TAX-G01", "area": "eligibility_screening", "category": "PROCESS_MISSING", "requirement": "Art. 8: Activity eligibility screening process", "fields_needed": ["taxonomy_activities"]},
        {"gap_id": "TAX-G02", "area": "sc_criteria", "category": "CAPABILITY_GAP", "requirement": "Technical screening criteria assessment capability", "fields_needed": ["sc_assessment"]},
        {"gap_id": "TAX-G03", "area": "dnsh_assessment", "category": "PROCESS_MISSING", "requirement": "Art. 17: DNSH assessment process", "fields_needed": ["dnsh_results"]},
        {"gap_id": "TAX-G04", "area": "kpi_calculation", "category": "DATA_MISSING", "requirement": "Revenue/CapEx/OpEx KPI data", "fields_needed": ["revenue", "capex", "opex"]},
        {"gap_id": "TAX-G05", "area": "minimum_safeguards", "category": "GOVERNANCE_GAP", "requirement": "Art. 18: Minimum safeguards policies", "fields_needed": ["safeguards_status"]},
        {"gap_id": "TAX-G06", "area": "activity_mapping", "category": "CAPABILITY_GAP", "requirement": "Business activity to NACE code mapping", "fields_needed": ["nace_codes"]},
        {"gap_id": "TAX-G07", "area": "climate_data", "category": "DATA_MISSING", "requirement": "Climate adaptation vulnerability data", "fields_needed": ["adaptation_data"]},
        {"gap_id": "TAX-G08", "area": "third_party_verification", "category": "CONTROL_MISSING", "requirement": "Third-party taxonomy alignment verification", "fields_needed": ["verification_provider"]},
    ],
    RegulationPack.EUDR.value: [
        {"gap_id": "EUDR-G01", "area": "geolocation_data", "category": "DATA_MISSING", "requirement": "Art. 9(d): Plot-level geolocation data", "fields_needed": ["geolocations"]},
        {"gap_id": "EUDR-G02", "area": "supply_chain_map", "category": "CAPABILITY_GAP", "requirement": "Full supply chain traceability mapping", "fields_needed": ["supply_chain_map"]},
        {"gap_id": "EUDR-G03", "area": "satellite_monitoring", "category": "TECHNOLOGY_GAP", "requirement": "Satellite-based deforestation monitoring", "fields_needed": ["satellite_data"]},
        {"gap_id": "EUDR-G04", "area": "risk_assessment", "category": "PROCESS_MISSING", "requirement": "Art. 10: Country and supplier risk assessment", "fields_needed": ["risk_scores"]},
        {"gap_id": "EUDR-G05", "area": "dd_statement", "category": "DOCUMENTATION_GAP", "requirement": "Art. 4: Due diligence statement preparation", "fields_needed": ["dd_statement"]},
        {"gap_id": "EUDR-G06", "area": "legality_verification", "category": "CONTROL_MISSING", "requirement": "Art. 3: Production legality verification", "fields_needed": ["legality_docs"]},
        {"gap_id": "EUDR-G07", "area": "commodity_tracing", "category": "PROCESS_MISSING", "requirement": "Commodity segregation and tracing process", "fields_needed": ["commodity_tracing"]},
        {"gap_id": "EUDR-G08", "area": "staff_training", "category": "EXPERTISE_GAP", "requirement": "EUDR compliance training for procurement", "fields_needed": ["training_status"]},
    ],
}

CROSS_FRAMEWORK_GAP_LINKS: Dict[str, List[str]] = {
    "ghg_emissions_measurement": ["CSRD-G01", "CBAM-G01"],
    "supplier_engagement": ["CBAM-G02", "EUDR-G02"],
    "governance_structure": ["CSRD-G05", "TAX-G05"],
    "verification_process": ["CBAM-G04", "TAX-G08", "EUDR-G06"],
    "data_management_systems": ["CSRD-G07", "CBAM-G05"],
    "carbon_pricing_tracking": ["CSRD-G01", "CBAM-G07"],
    "transition_and_dnsh": ["CSRD-G04", "TAX-G03"],
    "biodiversity_and_deforestation": ["CSRD-G09", "EUDR-G03"],
    "value_chain_traceability": ["CSRD-G08", "EUDR-G02", "CBAM-G02"],
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    records_processed: int = Field(default=0)


class WorkflowResult(BaseModel):
    """Complete result from a multi-phase workflow execution."""
    workflow_id: str = Field(...)
    workflow_name: str = Field(...)
    status: WorkflowStatus = Field(...)
    started_at: datetime = Field(...)
    completed_at: Optional[datetime] = Field(None)
    total_duration_seconds: float = Field(default=0.0)
    phases: List[PhaseResult] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class WorkflowConfig(BaseModel):
    """Configuration for cross-framework gap analysis workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    available_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data fields currently available in the organization"
    )
    existing_processes: List[str] = Field(
        default_factory=list,
        description="List of process/capability identifiers already in place"
    )
    remediation_start_date: Optional[str] = Field(
        None,
        description="ISO date for remediation timeline start"
    )
    budget_eur: Optional[float] = Field(
        None, ge=0,
        description="Available budget for remediation in EUR"
    )
    skip_phases: List[str] = Field(default_factory=list)


class CrossFrameworkGapAnalysisResult(WorkflowResult):
    """Result from cross-framework gap analysis workflow."""
    total_gaps: int = Field(default=0)
    cross_framework_gaps: int = Field(default=0)
    critical_gaps: int = Field(default=0)
    remediation_actions: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CrossFrameworkGapAnalysisWorkflow:
    """
    Four-phase cross-framework gap analysis workflow.

    Runs individual gap scans, maps across frameworks, scores by
    severity, and generates remediation plans.

    Example:
        >>> wf = CrossFrameworkGapAnalysisWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     available_data={"scope1_tco2e": 15000.0}
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "cross_framework_gap_analysis"

    PHASE_ORDER = [
        "individual_gap_scans",
        "cross_mapping",
        "impact_scoring",
        "remediation_planning",
    ]

    def __init__(self) -> None:
        """Initialize the cross-framework gap analysis workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> CrossFrameworkGapAnalysisResult:
        """
        Execute the four-phase cross-framework gap analysis workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            CrossFrameworkGapAnalysisResult with gap analysis outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting cross-framework gap analysis %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "individual_gap_scans": self._phase_individual_gap_scans,
            "cross_mapping": self._phase_cross_mapping,
            "impact_scoring": self._phase_impact_scoring,
            "remediation_planning": self._phase_remediation_planning,
        }

        for phase_name in self.PHASE_ORDER:
            if phase_name in config.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                continue

            try:
                phase_result = phase_methods[phase_name](config)
                completed_phases.append(phase_result)
                if phase_result.status == PhaseStatus.COMPLETED:
                    self._phase_outputs[phase_name] = phase_result.outputs
                elif phase_result.status == PhaseStatus.FAILED:
                    overall_status = WorkflowStatus.FAILED
                    break
            except Exception as exc:
                logger.error("Phase '%s' raised: %s", phase_name, exc, exc_info=True)
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL

        completed_at = datetime.utcnow()
        summary = self._build_summary()
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        return CrossFrameworkGapAnalysisResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            total_gaps=summary.get("total_gaps", 0),
            cross_framework_gaps=summary.get("cross_framework_gaps", 0),
            critical_gaps=summary.get("critical_gaps", 0),
            remediation_actions=summary.get("remediation_actions", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Individual Gap Scans
    # -------------------------------------------------------------------------

    def _phase_individual_gap_scans(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Run gap analysis in each constituent pack.

        Evaluates each gap category against available data and processes
        to identify which gaps exist.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            pack_gaps: Dict[str, List[Dict[str, Any]]] = {}
            total_gaps = 0
            total_no_gap = 0

            for pack in config.target_packs:
                pack_name = pack.value
                gap_defs = PACK_GAP_CATEGORIES.get(pack_name, [])
                pack_results: List[Dict[str, Any]] = []

                for gap_def in gap_defs:
                    gap_result = self._evaluate_gap(
                        gap_def, config.available_data, config.existing_processes
                    )
                    gap_result["pack"] = pack_name
                    pack_results.append(gap_result)

                    if gap_result["is_gap"]:
                        total_gaps += 1
                    else:
                        total_no_gap += 1

                pack_gaps[pack_name] = pack_results

            outputs["pack_gaps"] = pack_gaps
            outputs["total_gaps"] = total_gaps
            outputs["total_no_gap"] = total_no_gap
            outputs["per_pack_gap_counts"] = {
                pack: sum(1 for g in gaps if g["is_gap"])
                for pack, gaps in pack_gaps.items()
            }
            outputs["per_pack_total_counts"] = {
                pack: len(gaps) for pack, gaps in pack_gaps.items()
            }

            gap_by_category: Dict[str, int] = {}
            for pack_results in pack_gaps.values():
                for g in pack_results:
                    if g["is_gap"]:
                        cat = g.get("category", "UNKNOWN")
                        gap_by_category[cat] = gap_by_category.get(cat, 0) + 1
            outputs["gaps_by_category"] = gap_by_category

            logger.info(
                "Individual gap scans complete: %d gaps found out of %d checks",
                total_gaps, total_gaps + total_no_gap,
            )

            status = PhaseStatus.COMPLETED
            records = total_gaps + total_no_gap

        except Exception as exc:
            logger.error("Individual gap scans failed: %s", exc, exc_info=True)
            errors.append(f"Individual gap scans failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="individual_gap_scans",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _evaluate_gap(
        self,
        gap_def: Dict[str, Any],
        available_data: Dict[str, Any],
        existing_processes: List[str],
    ) -> Dict[str, Any]:
        """Evaluate whether a gap exists based on available data and processes."""
        gap_id = gap_def["gap_id"]
        area = gap_def["area"]
        category = gap_def["category"]
        fields_needed = gap_def.get("fields_needed", [])

        fields_present = sum(1 for f in fields_needed if f in available_data)
        fields_total = max(len(fields_needed), 1)
        coverage = fields_present / fields_total

        process_related = area in existing_processes
        has_capability = f"{area}_capability" in existing_processes

        if category in ("DATA_MISSING",):
            is_gap = coverage < 1.0
            gap_pct = 1.0 - coverage
        elif category in ("PROCESS_MISSING", "CONTROL_MISSING"):
            is_gap = not process_related
            gap_pct = 0.0 if process_related else 1.0
        elif category in ("DOCUMENTATION_GAP",):
            is_gap = coverage < 1.0 and not process_related
            gap_pct = 1.0 - coverage if is_gap else 0.0
        elif category in ("CAPABILITY_GAP", "EXPERTISE_GAP"):
            is_gap = not has_capability and coverage < 0.5
            gap_pct = 1.0 - coverage if is_gap else 0.0
        elif category in ("GOVERNANCE_GAP",):
            is_gap = not process_related and coverage < 1.0
            gap_pct = 1.0 if is_gap else 0.0
        elif category in ("TECHNOLOGY_GAP",):
            is_gap = not has_capability
            gap_pct = 1.0 if is_gap else 0.0
        else:
            is_gap = coverage < 1.0
            gap_pct = 1.0 - coverage

        return {
            "gap_id": gap_id,
            "area": area,
            "category": category,
            "requirement": gap_def["requirement"],
            "is_gap": is_gap,
            "gap_percentage": round(gap_pct, 4),
            "fields_needed": fields_needed,
            "fields_present": fields_present,
            "fields_total": fields_total,
            "data_coverage": round(coverage, 4),
            "assessed_at": datetime.utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Phase 2: Cross-Mapping
    # -------------------------------------------------------------------------

    def _phase_cross_mapping(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Map gaps across frameworks.

        Identifies gaps that affect multiple regulations and groups
        them into cross-framework gap clusters.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            scan_out = self._phase_outputs.get("individual_gap_scans", {})
            pack_gaps = scan_out.get("pack_gaps", {})

            all_gap_ids: Set[str] = set()
            gap_lookup: Dict[str, Dict[str, Any]] = {}
            for pack_results in pack_gaps.values():
                for gap in pack_results:
                    if gap["is_gap"]:
                        all_gap_ids.add(gap["gap_id"])
                        gap_lookup[gap["gap_id"]] = gap

            cross_framework_clusters: List[Dict[str, Any]] = []
            gap_ids_in_clusters: Set[str] = set()

            for theme, linked_gap_ids in CROSS_FRAMEWORK_GAP_LINKS.items():
                active_gaps = [
                    gid for gid in linked_gap_ids if gid in all_gap_ids
                ]
                if len(active_gaps) < 2:
                    continue

                packs_affected = set()
                for gid in active_gaps:
                    gap_info = gap_lookup.get(gid, {})
                    packs_affected.add(gap_info.get("pack", ""))

                if len(packs_affected) < 2:
                    continue

                cluster = {
                    "cluster_id": str(uuid.uuid4()),
                    "theme": theme,
                    "gap_ids": active_gaps,
                    "packs_affected": sorted(packs_affected),
                    "breadth": len(packs_affected),
                    "gap_details": [
                        {
                            "gap_id": gid,
                            "pack": gap_lookup[gid].get("pack", ""),
                            "area": gap_lookup[gid].get("area", ""),
                            "category": gap_lookup[gid].get("category", ""),
                        }
                        for gid in active_gaps
                    ],
                }
                cross_framework_clusters.append(cluster)
                gap_ids_in_clusters.update(active_gaps)

            pack_only_gaps = all_gap_ids - gap_ids_in_clusters

            outputs["cross_framework_clusters"] = cross_framework_clusters
            outputs["cross_framework_count"] = len(cross_framework_clusters)
            outputs["gaps_in_clusters"] = len(gap_ids_in_clusters)
            outputs["pack_only_gaps"] = sorted(pack_only_gaps)
            outputs["pack_only_count"] = len(pack_only_gaps)

            max_breadth = max(
                (c["breadth"] for c in cross_framework_clusters), default=0
            )
            outputs["max_cross_regulation_breadth"] = max_breadth

            logger.info(
                "Cross-mapping complete: %d clusters, %d cross-framework gaps, %d pack-only",
                len(cross_framework_clusters), len(gap_ids_in_clusters), len(pack_only_gaps),
            )

            status = PhaseStatus.COMPLETED
            records = len(cross_framework_clusters)

        except Exception as exc:
            logger.error("Cross-mapping failed: %s", exc, exc_info=True)
            errors.append(f"Cross-mapping failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="cross_mapping",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Impact Scoring
    # -------------------------------------------------------------------------

    def _phase_impact_scoring(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Score gaps by severity and cross-regulation breadth.

        Assigns an impact score combining regulatory risk, breadth
        across frameworks, and remediation complexity.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            scan_out = self._phase_outputs.get("individual_gap_scans", {})
            mapping_out = self._phase_outputs.get("cross_mapping", {})
            pack_gaps = scan_out.get("pack_gaps", {})
            clusters = mapping_out.get("cross_framework_clusters", [])

            gap_id_to_cluster: Dict[str, Dict[str, Any]] = {}
            for cluster in clusters:
                for gid in cluster["gap_ids"]:
                    gap_id_to_cluster[gid] = cluster

            scored_gaps: List[Dict[str, Any]] = []
            for pack_results in pack_gaps.values():
                for gap in pack_results:
                    if not gap["is_gap"]:
                        continue

                    gap_id = gap["gap_id"]
                    cluster = gap_id_to_cluster.get(gap_id)
                    breadth = cluster["breadth"] if cluster else 1

                    severity, impact_score = self._calculate_impact(
                        gap, breadth
                    )

                    scored_gap = {
                        "gap_id": gap_id,
                        "pack": gap.get("pack", ""),
                        "area": gap["area"],
                        "category": gap["category"],
                        "requirement": gap["requirement"],
                        "severity": severity,
                        "impact_score": round(impact_score, 4),
                        "cross_regulation_breadth": breadth,
                        "gap_percentage": gap["gap_percentage"],
                        "cluster_theme": cluster["theme"] if cluster else None,
                    }
                    scored_gaps.append(scored_gap)

            scored_gaps.sort(key=lambda g: g["impact_score"], reverse=True)

            severity_counts = {}
            for sg in scored_gaps:
                sev = sg["severity"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            outputs["scored_gaps"] = scored_gaps
            outputs["total_scored"] = len(scored_gaps)
            outputs["severity_counts"] = severity_counts
            outputs["critical_count"] = severity_counts.get(GapSeverity.CRITICAL.value, 0)
            outputs["high_count"] = severity_counts.get(GapSeverity.HIGH.value, 0)
            outputs["medium_count"] = severity_counts.get(GapSeverity.MEDIUM.value, 0)
            outputs["low_count"] = severity_counts.get(GapSeverity.LOW.value, 0)

            if scored_gaps:
                outputs["max_impact_score"] = scored_gaps[0]["impact_score"]
                outputs["avg_impact_score"] = round(
                    sum(g["impact_score"] for g in scored_gaps) / len(scored_gaps), 4
                )
                outputs["top_5_gaps"] = scored_gaps[:5]
            else:
                outputs["max_impact_score"] = 0.0
                outputs["avg_impact_score"] = 0.0
                outputs["top_5_gaps"] = []

            logger.info(
                "Impact scoring complete: %d gaps scored, %d critical, %d high",
                len(scored_gaps),
                severity_counts.get(GapSeverity.CRITICAL.value, 0),
                severity_counts.get(GapSeverity.HIGH.value, 0),
            )

            status = PhaseStatus.COMPLETED
            records = len(scored_gaps)

        except Exception as exc:
            logger.error("Impact scoring failed: %s", exc, exc_info=True)
            errors.append(f"Impact scoring failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="impact_scoring",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _calculate_impact(
        self,
        gap: Dict[str, Any],
        breadth: int,
    ) -> Tuple[str, float]:
        """
        Calculate severity and impact score for a gap.

        Impact formula: base_risk * breadth_multiplier * gap_percentage
        """
        category = gap.get("category", "")
        gap_pct = gap.get("gap_percentage", 1.0)

        category_risk: Dict[str, float] = {
            GapCategory.DATA_MISSING.value: 0.7,
            GapCategory.PROCESS_MISSING.value: 0.8,
            GapCategory.CONTROL_MISSING.value: 0.85,
            GapCategory.DOCUMENTATION_GAP.value: 0.5,
            GapCategory.CAPABILITY_GAP.value: 0.65,
            GapCategory.GOVERNANCE_GAP.value: 0.75,
            GapCategory.TECHNOLOGY_GAP.value: 0.6,
            GapCategory.EXPERTISE_GAP.value: 0.55,
        }
        base_risk = category_risk.get(category, 0.5)

        breadth_multiplier = 1.0 + (breadth - 1) * 0.3

        impact_score = base_risk * breadth_multiplier * gap_pct
        impact_score = min(impact_score, 1.0)

        if impact_score >= 0.8:
            severity = GapSeverity.CRITICAL.value
        elif impact_score >= 0.6:
            severity = GapSeverity.HIGH.value
        elif impact_score >= 0.4:
            severity = GapSeverity.MEDIUM.value
        elif impact_score >= 0.2:
            severity = GapSeverity.LOW.value
        else:
            severity = GapSeverity.INFORMATIONAL.value

        return severity, impact_score

    # -------------------------------------------------------------------------
    # Phase 4: Remediation Planning
    # -------------------------------------------------------------------------

    def _phase_remediation_planning(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Generate unified remediation plan with timeline.

        Creates actionable remediation items prioritized by impact
        score, with estimated effort, cost, and timeline.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            scoring_out = self._phase_outputs.get("impact_scoring", {})
            scored_gaps = scoring_out.get("scored_gaps", [])

            if config.remediation_start_date:
                try:
                    start_date = datetime.fromisoformat(
                        config.remediation_start_date.replace("Z", "+00:00")
                    )
                except ValueError:
                    start_date = datetime.utcnow()
            else:
                start_date = datetime.utcnow()

            remediation_actions: List[Dict[str, Any]] = []
            current_date = start_date
            total_effort_days = 0
            total_estimated_cost = 0.0

            for gap in scored_gaps:
                priority = self._determine_priority(gap["severity"])
                effort_days, estimated_cost = self._estimate_effort(
                    gap["category"], gap["cross_regulation_breadth"]
                )

                action_start = current_date
                action_end = action_start + timedelta(days=effort_days)

                action = {
                    "action_id": str(uuid.uuid4()),
                    "gap_id": gap["gap_id"],
                    "pack": gap["pack"],
                    "area": gap["area"],
                    "category": gap["category"],
                    "severity": gap["severity"],
                    "impact_score": gap["impact_score"],
                    "priority": priority,
                    "description": f"Remediate {gap['area']}: {gap['requirement']}",
                    "effort_days": effort_days,
                    "estimated_cost_eur": estimated_cost,
                    "start_date": action_start.strftime("%Y-%m-%d"),
                    "end_date": action_end.strftime("%Y-%m-%d"),
                    "cross_regulation_benefit": gap["cross_regulation_breadth"] > 1,
                    "packs_benefited": gap["cross_regulation_breadth"],
                    "cluster_theme": gap.get("cluster_theme"),
                }
                remediation_actions.append(action)

                total_effort_days += effort_days
                total_estimated_cost += estimated_cost

                if priority in (RemediationPriority.IMMEDIATE.value, RemediationPriority.SHORT_TERM.value):
                    current_date = action_end

            by_priority: Dict[str, int] = {}
            for action in remediation_actions:
                p = action["priority"]
                by_priority[p] = by_priority.get(p, 0) + 1

            cross_benefit_actions = [
                a for a in remediation_actions if a["cross_regulation_benefit"]
            ]

            outputs["remediation_actions"] = remediation_actions
            outputs["total_actions"] = len(remediation_actions)
            outputs["by_priority"] = by_priority
            outputs["total_effort_days"] = total_effort_days
            outputs["total_estimated_cost_eur"] = round(total_estimated_cost, 2)
            outputs["cross_benefit_actions"] = len(cross_benefit_actions)
            outputs["plan_start_date"] = start_date.strftime("%Y-%m-%d")
            outputs["plan_end_date"] = current_date.strftime("%Y-%m-%d")

            if config.budget_eur is not None:
                outputs["budget_eur"] = config.budget_eur
                outputs["budget_sufficient"] = total_estimated_cost <= config.budget_eur
                outputs["budget_gap_eur"] = max(
                    0, total_estimated_cost - config.budget_eur
                )
                if total_estimated_cost > config.budget_eur:
                    warnings.append(
                        f"Estimated cost EUR {total_estimated_cost:,.2f} "
                        f"exceeds budget EUR {config.budget_eur:,.2f}"
                    )

            outputs["executive_summary"] = {
                "total_gaps": len(scored_gaps),
                "total_remediation_actions": len(remediation_actions),
                "critical_actions": by_priority.get(RemediationPriority.IMMEDIATE.value, 0),
                "estimated_duration_days": total_effort_days,
                "estimated_total_cost_eur": round(total_estimated_cost, 2),
                "cross_regulation_efficiencies": len(cross_benefit_actions),
            }

            logger.info(
                "Remediation planning complete: %d actions, %d days, EUR %.2f",
                len(remediation_actions), total_effort_days, total_estimated_cost,
            )

            status = PhaseStatus.COMPLETED
            records = len(remediation_actions)

        except Exception as exc:
            logger.error("Remediation planning failed: %s", exc, exc_info=True)
            errors.append(f"Remediation planning failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="remediation_planning",
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
            records_processed=records,
        )

    def _determine_priority(self, severity: str) -> str:
        """Determine remediation priority from gap severity."""
        priority_map = {
            GapSeverity.CRITICAL.value: RemediationPriority.IMMEDIATE.value,
            GapSeverity.HIGH.value: RemediationPriority.SHORT_TERM.value,
            GapSeverity.MEDIUM.value: RemediationPriority.MEDIUM_TERM.value,
            GapSeverity.LOW.value: RemediationPriority.LONG_TERM.value,
            GapSeverity.INFORMATIONAL.value: RemediationPriority.LONG_TERM.value,
        }
        return priority_map.get(severity, RemediationPriority.MEDIUM_TERM.value)

    def _estimate_effort(
        self,
        category: str,
        breadth: int,
    ) -> Tuple[int, float]:
        """
        Estimate effort in days and cost in EUR for a remediation action.

        Deterministic estimation based on gap category and breadth.
        """
        base_effort: Dict[str, int] = {
            GapCategory.DATA_MISSING.value: 15,
            GapCategory.PROCESS_MISSING.value: 25,
            GapCategory.CONTROL_MISSING.value: 20,
            GapCategory.DOCUMENTATION_GAP.value: 10,
            GapCategory.CAPABILITY_GAP.value: 30,
            GapCategory.GOVERNANCE_GAP.value: 20,
            GapCategory.TECHNOLOGY_GAP.value: 40,
            GapCategory.EXPERTISE_GAP.value: 15,
        }
        base_cost_per_day = 800.0

        effort_days = base_effort.get(category, 20)
        effort_days = int(effort_days * (1.0 + (breadth - 1) * 0.15))
        estimated_cost = effort_days * base_cost_per_day

        return effort_days, estimated_cost

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        scans = self._phase_outputs.get("individual_gap_scans", {})
        mapping = self._phase_outputs.get("cross_mapping", {})
        scoring = self._phase_outputs.get("impact_scoring", {})
        remediation = self._phase_outputs.get("remediation_planning", {})

        return {
            "total_gaps": scans.get("total_gaps", 0),
            "gaps_by_category": scans.get("gaps_by_category", {}),
            "cross_framework_gaps": mapping.get("gaps_in_clusters", 0),
            "cross_framework_clusters": mapping.get("cross_framework_count", 0),
            "critical_gaps": scoring.get("critical_count", 0),
            "high_gaps": scoring.get("high_count", 0),
            "avg_impact_score": scoring.get("avg_impact_score", 0.0),
            "remediation_actions": remediation.get("total_actions", 0),
            "estimated_effort_days": remediation.get("total_effort_days", 0),
            "estimated_cost_eur": remediation.get("total_estimated_cost_eur", 0.0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
