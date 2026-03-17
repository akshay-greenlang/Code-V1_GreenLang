# -*- coding: utf-8 -*-
"""
Consolidated Reporting Workflow
====================================

Four-phase workflow that collects results from all four constituent regulation
packs (CSRD, CBAM, EU Taxonomy, EUDR), maps overlapping disclosures to avoid
duplication, generates consolidated and per-regulation reports, and produces
filing-ready packages for each regulatory authority.

Phases:
    1. ResultsCollection - Collect results from all 4 packs
    2. CrossMapping - Map overlapping disclosures to avoid duplication
    3. ReportGeneration - Generate consolidated + per-regulation reports
    4. FilingPackage - Produce filing-ready packages per regulation

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

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


class ReportFormat(str, Enum):
    """Available report output formats."""
    PDF = "PDF"
    XHTML = "XHTML"
    XLSX = "XLSX"
    JSON = "JSON"
    XML = "XML"


class FilingStatus(str, Enum):
    """Filing package readiness status."""
    READY = "READY"
    INCOMPLETE = "INCOMPLETE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# =============================================================================
# DISCLOSURE MAPPINGS
# =============================================================================


PACK_DISCLOSURES: Dict[str, List[Dict[str, Any]]] = {
    RegulationPack.CSRD.value: [
        {"disclosure_id": "CSRD-E1-6", "topic": "ghg_emissions", "section": "ESRS E1-6", "description": "Gross Scope 1, 2, 3 GHG emissions", "data_fields": ["scope1_tco2e", "scope2_tco2e", "scope3_tco2e"]},
        {"disclosure_id": "CSRD-E1-5", "topic": "energy_consumption", "section": "ESRS E1-5", "description": "Energy consumption and mix", "data_fields": ["energy_mwh", "renewable_pct"]},
        {"disclosure_id": "CSRD-E1-8", "topic": "carbon_pricing", "section": "ESRS E1-8", "description": "Internal carbon pricing", "data_fields": ["carbon_price_eur"]},
        {"disclosure_id": "CSRD-E1-1", "topic": "transition_plan", "section": "ESRS E1-1", "description": "Transition plan for climate change mitigation", "data_fields": ["transition_plan_text"]},
        {"disclosure_id": "CSRD-E3-4", "topic": "water_consumption", "section": "ESRS E3-4", "description": "Water consumption", "data_fields": ["water_m3"]},
        {"disclosure_id": "CSRD-E4-4", "topic": "biodiversity", "section": "ESRS E4-4", "description": "Biodiversity and ecosystems", "data_fields": ["biodiversity_assessment"]},
        {"disclosure_id": "CSRD-E5-5", "topic": "waste_management", "section": "ESRS E5-5", "description": "Resource outflows including waste", "data_fields": ["waste_tonnes"]},
        {"disclosure_id": "CSRD-G1-1", "topic": "governance", "section": "ESRS G1-1", "description": "Business conduct policies", "data_fields": ["governance_doc"]},
    ],
    RegulationPack.CBAM.value: [
        {"disclosure_id": "CBAM-EMB", "topic": "embedded_emissions", "section": "CBAM Art. 7", "description": "Embedded emissions per goods category", "data_fields": ["scope1_tco2e", "scope2_tco2e", "specific_emissions"]},
        {"disclosure_id": "CBAM-IMP", "topic": "import_volumes", "section": "CBAM Art. 6", "description": "Import volumes by CN code", "data_fields": ["import_tonnes", "cn_codes"]},
        {"disclosure_id": "CBAM-CERT", "topic": "certificates", "section": "CBAM Art. 22", "description": "CBAM certificates purchased and surrendered", "data_fields": ["certs_purchased", "certs_surrendered", "cert_cost_eur"]},
        {"disclosure_id": "CBAM-CPD", "topic": "carbon_price_deduction", "section": "CBAM Art. 9", "description": "Carbon price paid in country of origin", "data_fields": ["carbon_price_origin_eur"]},
        {"disclosure_id": "CBAM-SUP", "topic": "supplier_data", "section": "CBAM Art. 8", "description": "Supplier installation data and verification", "data_fields": ["supplier_ids", "verification_status"]},
    ],
    RegulationPack.EU_TAXONOMY.value: [
        {"disclosure_id": "TAX-REV", "topic": "taxonomy_revenue", "section": "Art. 8 Delegated Act", "description": "Taxonomy-eligible and aligned revenue", "data_fields": ["eligible_revenue_eur", "aligned_revenue_eur"]},
        {"disclosure_id": "TAX-CAP", "topic": "taxonomy_capex", "section": "Art. 8 Delegated Act", "description": "Taxonomy-eligible and aligned CapEx", "data_fields": ["eligible_capex_eur", "aligned_capex_eur"]},
        {"disclosure_id": "TAX-OPX", "topic": "taxonomy_opex", "section": "Art. 8 Delegated Act", "description": "Taxonomy-eligible and aligned OpEx", "data_fields": ["eligible_opex_eur", "aligned_opex_eur"]},
        {"disclosure_id": "TAX-DNSH", "topic": "dnsh_assessment", "section": "Art. 17", "description": "DNSH assessment across 6 objectives", "data_fields": ["scope1_tco2e", "scope2_tco2e", "water_m3", "waste_tonnes"]},
        {"disclosure_id": "TAX-MSS", "topic": "minimum_safeguards", "section": "Art. 18", "description": "Minimum social safeguards", "data_fields": ["safeguards_status"]},
    ],
    RegulationPack.EUDR.value: [
        {"disclosure_id": "EUDR-COM", "topic": "commodity_traceability", "section": "EUDR Art. 9", "description": "Commodity identification and traceability", "data_fields": ["commodity_type", "supply_chain_map"]},
        {"disclosure_id": "EUDR-GEO", "topic": "geolocation", "section": "EUDR Art. 9(d)", "description": "Geolocation of production plots", "data_fields": ["geolocations"]},
        {"disclosure_id": "EUDR-DFS", "topic": "deforestation_status", "section": "EUDR Art. 3", "description": "Deforestation-free status verification", "data_fields": ["deforestation_free_status"]},
        {"disclosure_id": "EUDR-RSK", "topic": "risk_assessment", "section": "EUDR Art. 10", "description": "Risk assessment results", "data_fields": ["risk_score", "country_risk"]},
        {"disclosure_id": "EUDR-DDS", "topic": "due_diligence", "section": "EUDR Art. 8", "description": "Due diligence statement", "data_fields": ["dd_statement"]},
    ],
}

DISCLOSURE_OVERLAP_MAP: Dict[str, List[str]] = {
    "ghg_emissions": ["CSRD-E1-6", "CBAM-EMB", "TAX-DNSH"],
    "carbon_pricing": ["CSRD-E1-8", "CBAM-CERT", "CBAM-CPD"],
    "water_consumption": ["CSRD-E3-4", "TAX-DNSH"],
    "waste_management": ["CSRD-E5-5", "TAX-DNSH"],
    "supplier_data": ["CBAM-SUP", "EUDR-COM"],
    "governance": ["CSRD-G1-1", "TAX-MSS"],
}

FILING_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    RegulationPack.CSRD.value: {
        "authority": "National Competent Authority (ESMA)",
        "format": "XHTML with iXBRL tags",
        "frequency": "Annual",
        "deadline_month": 4,
        "required_disclosures": ["CSRD-E1-6", "CSRD-E1-5", "CSRD-E1-1", "CSRD-G1-1"],
    },
    RegulationPack.CBAM.value: {
        "authority": "European Commission CBAM Authority",
        "format": "CBAM Transitional Registry XML",
        "frequency": "Quarterly + Annual",
        "deadline_month": 1,
        "required_disclosures": ["CBAM-EMB", "CBAM-IMP", "CBAM-CERT"],
    },
    RegulationPack.EU_TAXONOMY.value: {
        "authority": "National Competent Authority (financial supervisor)",
        "format": "Structured tables in annual report",
        "frequency": "Annual",
        "deadline_month": 4,
        "required_disclosures": ["TAX-REV", "TAX-CAP", "TAX-DNSH"],
    },
    RegulationPack.EUDR.value: {
        "authority": "EU Information System (national portal)",
        "format": "Due Diligence Statement XML",
        "frequency": "Per-shipment + Annual review",
        "deadline_month": 12,
        "required_disclosures": ["EUDR-COM", "EUDR-GEO", "EUDR-DFS", "EUDR-DDS"],
    },
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
    """Configuration for consolidated reporting workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    pack_results: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Pre-computed results from each pack"
    )
    report_formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.JSON]
    )
    include_consolidated: bool = Field(
        default=True,
        description="Generate a single consolidated report in addition to per-regulation"
    )
    skip_phases: List[str] = Field(default_factory=list)


class ConsolidatedReportingResult(WorkflowResult):
    """Result from consolidated reporting workflow."""
    disclosures_collected: int = Field(default=0)
    overlapping_disclosures: int = Field(default=0)
    reports_generated: int = Field(default=0)
    filing_packages_ready: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ConsolidatedReportingWorkflow:
    """
    Four-phase consolidated reporting workflow.

    Collects results from all constituent packs, maps overlapping
    disclosures, generates reports, and produces filing packages.

    Example:
        >>> wf = ConsolidatedReportingWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "consolidated_reporting"

    PHASE_ORDER = [
        "results_collection",
        "cross_mapping",
        "report_generation",
        "filing_package",
    ]

    def __init__(self) -> None:
        """Initialize the consolidated reporting workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> ConsolidatedReportingResult:
        """
        Execute the four-phase consolidated reporting workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            ConsolidatedReportingResult with reporting outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting consolidated reporting %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "results_collection": self._phase_results_collection,
            "cross_mapping": self._phase_cross_mapping,
            "report_generation": self._phase_report_generation,
            "filing_package": self._phase_filing_package,
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

        return ConsolidatedReportingResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            disclosures_collected=summary.get("disclosures_collected", 0),
            overlapping_disclosures=summary.get("overlapping_disclosures", 0),
            reports_generated=summary.get("reports_generated", 0),
            filing_packages_ready=summary.get("filing_packages_ready", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Results Collection
    # -------------------------------------------------------------------------

    def _phase_results_collection(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Collect results from all 4 constituent packs.

        Gathers disclosure data points from each pack, either from
        pre-computed results or by simulating collection.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collected_disclosures: Dict[str, List[Dict[str, Any]]] = {}
            total_disclosures = 0

            for pack in config.target_packs:
                pack_name = pack.value
                pack_disc = PACK_DISCLOSURES.get(pack_name, [])
                pack_results = config.pack_results.get(pack_name, {})

                enriched: List[Dict[str, Any]] = []
                for disc in pack_disc:
                    disc_data = dict(disc)
                    disc_data["pack"] = pack_name
                    disc_data["has_data"] = self._check_disclosure_data(
                        disc, pack_results
                    )
                    disc_data["data_values"] = self._extract_disclosure_values(
                        disc, pack_results
                    )
                    disc_data["collected_at"] = datetime.utcnow().isoformat()
                    enriched.append(disc_data)

                collected_disclosures[pack_name] = enriched
                total_disclosures += len(enriched)

            outputs["collected_disclosures"] = collected_disclosures
            outputs["total_disclosures"] = total_disclosures
            outputs["per_pack_counts"] = {
                pack: len(discs)
                for pack, discs in collected_disclosures.items()
            }

            with_data = sum(
                1
                for pack_discs in collected_disclosures.values()
                for d in pack_discs
                if d.get("has_data")
            )
            outputs["disclosures_with_data"] = with_data
            outputs["disclosures_without_data"] = total_disclosures - with_data

            if with_data < total_disclosures:
                warnings.append(
                    f"{total_disclosures - with_data} disclosures missing data"
                )

            logger.info(
                "Results collection complete: %d disclosures, %d with data",
                total_disclosures, with_data,
            )

            status = PhaseStatus.COMPLETED
            records = total_disclosures

        except Exception as exc:
            logger.error("Results collection failed: %s", exc, exc_info=True)
            errors.append(f"Results collection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="results_collection",
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

    def _check_disclosure_data(
        self,
        disclosure: Dict[str, Any],
        pack_results: Dict[str, Any],
    ) -> bool:
        """Check if pack results contain data for a disclosure."""
        data_fields = disclosure.get("data_fields", [])
        if not pack_results:
            return False
        return any(f in pack_results for f in data_fields)

    def _extract_disclosure_values(
        self,
        disclosure: Dict[str, Any],
        pack_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract relevant data values from pack results for a disclosure."""
        data_fields = disclosure.get("data_fields", [])
        values: Dict[str, Any] = {}
        for field in data_fields:
            if field in pack_results:
                values[field] = pack_results[field]
        return values

    # -------------------------------------------------------------------------
    # Phase 2: Cross-Mapping
    # -------------------------------------------------------------------------

    def _phase_cross_mapping(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Map overlapping disclosures to avoid duplication.

        Identifies disclosure topics that appear in multiple regulations
        and designates a primary source to avoid conflicting reports.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_out = self._phase_outputs.get("results_collection", {})
            collected = collection_out.get("collected_disclosures", {})

            overlap_groups: List[Dict[str, Any]] = []
            total_overlapping_ids: Set[str] = set()

            for topic, disc_ids in DISCLOSURE_OVERLAP_MAP.items():
                group_disclosures: List[Dict[str, Any]] = []
                for pack_name, pack_discs in collected.items():
                    for disc in pack_discs:
                        if disc["disclosure_id"] in disc_ids:
                            group_disclosures.append(disc)

                if len(group_disclosures) < 2:
                    continue

                primary = self._select_primary_disclosure(group_disclosures)
                secondary_ids = [
                    d["disclosure_id"]
                    for d in group_disclosures
                    if d["disclosure_id"] != primary["disclosure_id"]
                ]
                total_overlapping_ids.update(
                    d["disclosure_id"] for d in group_disclosures
                )

                overlap_groups.append({
                    "topic": topic,
                    "disclosure_ids": [d["disclosure_id"] for d in group_disclosures],
                    "packs_involved": list({d["pack"] for d in group_disclosures}),
                    "primary_disclosure": primary["disclosure_id"],
                    "primary_pack": primary["pack"],
                    "secondary_disclosures": secondary_ids,
                    "data_reconciliation": "primary_source_used",
                })

            outputs["overlap_groups"] = overlap_groups
            outputs["total_overlap_groups"] = len(overlap_groups)
            outputs["overlapping_disclosure_ids"] = sorted(total_overlapping_ids)
            outputs["overlapping_count"] = len(total_overlapping_ids)

            dedup_map: Dict[str, str] = {}
            for group in overlap_groups:
                primary_id = group["primary_disclosure"]
                for sec_id in group["secondary_disclosures"]:
                    dedup_map[sec_id] = primary_id
            outputs["deduplication_map"] = dedup_map

            logger.info(
                "Cross-mapping complete: %d overlap groups, %d overlapping disclosures",
                len(overlap_groups), len(total_overlapping_ids),
            )

            status = PhaseStatus.COMPLETED
            records = len(overlap_groups)

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

    def _select_primary_disclosure(
        self,
        disclosures: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Select the primary disclosure from overlapping group.

        Priority: CSRD > EU_TAXONOMY > CBAM > EUDR (most comprehensive first).
        """
        pack_priority = {
            RegulationPack.CSRD.value: 0,
            RegulationPack.EU_TAXONOMY.value: 1,
            RegulationPack.CBAM.value: 2,
            RegulationPack.EUDR.value: 3,
        }
        sorted_discs = sorted(
            disclosures,
            key=lambda d: pack_priority.get(d.get("pack", ""), 99),
        )
        return sorted_discs[0]

    # -------------------------------------------------------------------------
    # Phase 3: Report Generation
    # -------------------------------------------------------------------------

    def _phase_report_generation(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Generate consolidated and per-regulation reports.

        Creates one consolidated cross-regulation report and individual
        reports for each regulation in the requested formats.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_out = self._phase_outputs.get("results_collection", {})
            mapping_out = self._phase_outputs.get("cross_mapping", {})
            collected = collection_out.get("collected_disclosures", {})
            dedup_map = mapping_out.get("deduplication_map", {})

            generated_reports: List[Dict[str, Any]] = []

            for pack in config.target_packs:
                pack_name = pack.value
                pack_discs = collected.get(pack_name, [])

                for fmt in config.report_formats:
                    report = self._generate_pack_report(
                        pack_name, pack_discs, fmt.value, dedup_map,
                        config.organization_id, config.reporting_year,
                    )
                    generated_reports.append(report)

            if config.include_consolidated:
                for fmt in config.report_formats:
                    consolidated_report = self._generate_consolidated_report(
                        collected, fmt.value, dedup_map,
                        config.organization_id, config.reporting_year,
                    )
                    generated_reports.append(consolidated_report)

            outputs["generated_reports"] = generated_reports
            outputs["reports_count"] = len(generated_reports)
            outputs["per_regulation_count"] = sum(
                1 for r in generated_reports if r["report_type"] == "per_regulation"
            )
            outputs["consolidated_count"] = sum(
                1 for r in generated_reports if r["report_type"] == "consolidated"
            )

            logger.info(
                "Report generation complete: %d reports generated",
                len(generated_reports),
            )

            status = PhaseStatus.COMPLETED
            records = len(generated_reports)

        except Exception as exc:
            logger.error("Report generation failed: %s", exc, exc_info=True)
            errors.append(f"Report generation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="report_generation",
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

    def _generate_pack_report(
        self,
        pack_name: str,
        disclosures: List[Dict[str, Any]],
        report_format: str,
        dedup_map: Dict[str, str],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Generate a report for a single regulation pack."""
        active_disclosures = [
            d for d in disclosures
            if d["disclosure_id"] not in dedup_map
        ]
        referenced_disclosures = [
            d for d in disclosures
            if d["disclosure_id"] in dedup_map
        ]

        report_id = str(uuid.uuid4())
        return {
            "report_id": report_id,
            "report_type": "per_regulation",
            "pack": pack_name,
            "format": report_format,
            "organization_id": org_id,
            "reporting_year": year,
            "active_disclosures": len(active_disclosures),
            "cross_referenced_disclosures": len(referenced_disclosures),
            "total_disclosures": len(disclosures),
            "file_reference": f"report_{pack_name.lower()}_{year}_{report_id[:8]}.{report_format.lower()}",
            "generated_at": datetime.utcnow().isoformat(),
            "page_count_estimate": max(5, len(active_disclosures) * 3),
        }

    def _generate_consolidated_report(
        self,
        all_disclosures: Dict[str, List[Dict[str, Any]]],
        report_format: str,
        dedup_map: Dict[str, str],
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Generate the consolidated cross-regulation report."""
        total_discs = sum(len(v) for v in all_disclosures.values())
        deduped_count = total_discs - len(dedup_map)

        report_id = str(uuid.uuid4())
        return {
            "report_id": report_id,
            "report_type": "consolidated",
            "pack": "ALL",
            "format": report_format,
            "organization_id": org_id,
            "reporting_year": year,
            "regulations_covered": list(all_disclosures.keys()),
            "total_disclosures": total_discs,
            "deduplicated_disclosures": deduped_count,
            "savings_from_dedup": len(dedup_map),
            "file_reference": f"report_consolidated_{year}_{report_id[:8]}.{report_format.lower()}",
            "generated_at": datetime.utcnow().isoformat(),
            "page_count_estimate": max(20, deduped_count * 3),
        }

    # -------------------------------------------------------------------------
    # Phase 4: Filing Package
    # -------------------------------------------------------------------------

    def _phase_filing_package(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Produce filing-ready packages per regulation.

        Creates submission-ready packages with the correct format,
        metadata, and validation status for each regulatory authority.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            collection_out = self._phase_outputs.get("results_collection", {})
            report_out = self._phase_outputs.get("report_generation", {})
            collected = collection_out.get("collected_disclosures", {})
            reports = report_out.get("generated_reports", [])

            filing_packages: List[Dict[str, Any]] = []

            for pack in config.target_packs:
                pack_name = pack.value
                filing_req = FILING_REQUIREMENTS.get(pack_name, {})
                pack_discs = collected.get(pack_name, [])
                pack_reports = [
                    r for r in reports
                    if r.get("pack") == pack_name
                ]

                required_ids = set(filing_req.get("required_disclosures", []))
                present_ids = {d["disclosure_id"] for d in pack_discs if d.get("has_data")}
                missing_required = required_ids - present_ids

                if not missing_required:
                    filing_status = FilingStatus.READY.value
                elif len(missing_required) < len(required_ids):
                    filing_status = FilingStatus.INCOMPLETE.value
                    warnings.append(
                        f"{pack_name}: missing required disclosures: "
                        f"{', '.join(sorted(missing_required))}"
                    )
                else:
                    filing_status = FilingStatus.VALIDATION_FAILED.value
                    warnings.append(
                        f"{pack_name}: all required disclosures missing"
                    )

                package_id = str(uuid.uuid4())
                filing_packages.append({
                    "package_id": package_id,
                    "pack": pack_name,
                    "authority": filing_req.get("authority", "Unknown"),
                    "submission_format": filing_req.get("format", "Unknown"),
                    "frequency": filing_req.get("frequency", "Unknown"),
                    "deadline_month": filing_req.get("deadline_month", 0),
                    "filing_status": filing_status,
                    "required_disclosures": sorted(required_ids),
                    "present_disclosures": sorted(present_ids),
                    "missing_disclosures": sorted(missing_required),
                    "attached_reports": [
                        r.get("file_reference", "") for r in pack_reports
                    ],
                    "organization_id": config.organization_id,
                    "reporting_year": config.reporting_year,
                    "prepared_at": datetime.utcnow().isoformat(),
                    "file_reference": f"filing_{pack_name.lower()}_{config.reporting_year}_{package_id[:8]}.zip",
                })

            outputs["filing_packages"] = filing_packages
            outputs["packages_ready"] = sum(
                1 for p in filing_packages
                if p["filing_status"] == FilingStatus.READY.value
            )
            outputs["packages_incomplete"] = sum(
                1 for p in filing_packages
                if p["filing_status"] == FilingStatus.INCOMPLETE.value
            )
            outputs["packages_failed"] = sum(
                1 for p in filing_packages
                if p["filing_status"] == FilingStatus.VALIDATION_FAILED.value
            )

            logger.info(
                "Filing packages complete: %d ready, %d incomplete, %d failed",
                outputs["packages_ready"],
                outputs["packages_incomplete"],
                outputs["packages_failed"],
            )

            status = PhaseStatus.COMPLETED
            records = len(filing_packages)

        except Exception as exc:
            logger.error("Filing package failed: %s", exc, exc_info=True)
            errors.append(f"Filing package failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="filing_package",
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
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        collection = self._phase_outputs.get("results_collection", {})
        mapping = self._phase_outputs.get("cross_mapping", {})
        reporting = self._phase_outputs.get("report_generation", {})
        filing = self._phase_outputs.get("filing_package", {})

        return {
            "disclosures_collected": collection.get("total_disclosures", 0),
            "disclosures_with_data": collection.get("disclosures_with_data", 0),
            "overlapping_disclosures": mapping.get("overlapping_count", 0),
            "overlap_groups": mapping.get("total_overlap_groups", 0),
            "reports_generated": reporting.get("reports_count", 0),
            "filing_packages_ready": filing.get("packages_ready", 0),
            "filing_packages_incomplete": filing.get("packages_incomplete", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
