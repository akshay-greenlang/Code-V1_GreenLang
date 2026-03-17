# -*- coding: utf-8 -*-
"""
Cross-Regulation Sync Workflow
=================================

Four-phase sync workflow triggered by CBAM data changes. Detects deltas
in CBAM import data, certificates, and calculations, then maps those
changes to affected regulations using the CrossRegulationEngine. Generates
regulation-specific data exports and notifies responsible compliance teams.

Target Regulations:
    - CSRD ESRS E1: GHG emissions disclosures (Scope 3 Category 1 upstream)
    - CDP C6/C7/C11: Climate change questionnaire sections
    - SBTi: Scope 3 Category 1 (Purchased Goods & Services) targets
    - EU Taxonomy: Climate change mitigation DNSH criteria
    - EU ETS: Benchmark data, free allocation adjustments
    - EUDR: Supply chain overlap for deforestation-linked commodities

Sync Triggers:
    - New CBAM import records added
    - CBAM calculations updated (embedded emissions recalculated)
    - Certificate purchases or surrenders executed
    - Declaration submitted or status changed
    - Supplier emission data updated

Phases:
    1. ChangeDetection - Identify CBAM data deltas since last sync
    2. Mapping - Transform CBAM data to regulation-specific formats
    3. OutputGeneration - Generate exports and consistency reports
    4. Notification - Alert compliance teams of updated data

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

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


class ChangeType(str, Enum):
    """Type of CBAM data change."""
    NEW_IMPORT = "NEW_IMPORT"
    UPDATED_CALCULATION = "UPDATED_CALCULATION"
    CERTIFICATE_PURCHASE = "CERTIFICATE_PURCHASE"
    CERTIFICATE_SURRENDER = "CERTIFICATE_SURRENDER"
    DECLARATION_SUBMITTED = "DECLARATION_SUBMITTED"
    SUPPLIER_DATA_UPDATED = "SUPPLIER_DATA_UPDATED"
    VERIFICATION_COMPLETED = "VERIFICATION_COMPLETED"


class TargetRegulation(str, Enum):
    """Regulation that may be affected by CBAM data changes."""
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    CDP_CLIMATE = "CDP_CLIMATE"
    SBTI_SCOPE3 = "SBTI_SCOPE3"
    EU_TAXONOMY = "EU_TAXONOMY"
    EU_ETS = "EU_ETS"
    EUDR = "EUDR"


class ExportFormat(str, Enum):
    """Output format for regulation exports."""
    JSON = "JSON"
    CSV = "CSV"
    XML = "XML"


class ConflictSeverity(str, Enum):
    """Severity of cross-regulation data conflict."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


# =============================================================================
# REGULATION MAPPINGS
# =============================================================================

# Which change types affect which regulations
CHANGE_REGULATION_MAP: Dict[str, List[str]] = {
    ChangeType.NEW_IMPORT.value: [
        TargetRegulation.CSRD_ESRS_E1.value,
        TargetRegulation.CDP_CLIMATE.value,
        TargetRegulation.SBTI_SCOPE3.value,
        TargetRegulation.EU_TAXONOMY.value,
        TargetRegulation.EUDR.value,
    ],
    ChangeType.UPDATED_CALCULATION.value: [
        TargetRegulation.CSRD_ESRS_E1.value,
        TargetRegulation.CDP_CLIMATE.value,
        TargetRegulation.SBTI_SCOPE3.value,
    ],
    ChangeType.CERTIFICATE_PURCHASE.value: [
        TargetRegulation.EU_ETS.value,
        TargetRegulation.CSRD_ESRS_E1.value,
    ],
    ChangeType.CERTIFICATE_SURRENDER.value: [
        TargetRegulation.EU_ETS.value,
        TargetRegulation.CSRD_ESRS_E1.value,
    ],
    ChangeType.DECLARATION_SUBMITTED.value: [
        TargetRegulation.CSRD_ESRS_E1.value,
        TargetRegulation.CDP_CLIMATE.value,
    ],
    ChangeType.SUPPLIER_DATA_UPDATED.value: [
        TargetRegulation.CSRD_ESRS_E1.value,
        TargetRegulation.CDP_CLIMATE.value,
        TargetRegulation.SBTI_SCOPE3.value,
        TargetRegulation.EUDR.value,
    ],
    ChangeType.VERIFICATION_COMPLETED.value: [
        TargetRegulation.CSRD_ESRS_E1.value,
    ],
}

# Regulation-specific data sections
REGULATION_SECTIONS: Dict[str, Dict[str, str]] = {
    TargetRegulation.CSRD_ESRS_E1.value: {
        "section": "ESRS E1 - Climate Change",
        "data_points": "E1-6 GHG emissions, E1-8 Internal carbon pricing",
        "scope": "Scope 3 Category 1 (Purchased Goods upstream emissions)",
    },
    TargetRegulation.CDP_CLIMATE.value: {
        "section": "CDP Climate Change",
        "data_points": "C6 Emissions data, C7 Emissions breakdown, C11 Carbon pricing",
        "scope": "Scope 3 upstream, carbon pricing mechanisms",
    },
    TargetRegulation.SBTI_SCOPE3.value: {
        "section": "SBTi Target Setting",
        "data_points": "Scope 3 Category 1 base year and target year",
        "scope": "Purchased Goods & Services boundary",
    },
    TargetRegulation.EU_TAXONOMY.value: {
        "section": "EU Taxonomy Climate Mitigation",
        "data_points": "DNSH criteria, carbon intensity benchmarks",
        "scope": "Climate change mitigation substantial contribution",
    },
    TargetRegulation.EU_ETS.value: {
        "section": "EU ETS",
        "data_points": "Benchmark data, free allocation, CBAM cross-reference",
        "scope": "ETS-CBAM interaction, free allocation phase-out",
    },
    TargetRegulation.EUDR.value: {
        "section": "EU Deforestation Regulation",
        "data_points": "Supply chain overlap, commodity traceability",
        "scope": "Deforestation-linked commodities in CBAM imports",
    },
}


# =============================================================================
# DATA MODELS - SHARED
# =============================================================================


class WorkflowContext(BaseModel):
    """Shared state passed between workflow phases."""
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: str = Field(...)
    execution_timestamp: datetime = Field(default_factory=datetime.utcnow)
    config: Dict[str, Any] = Field(default_factory=dict)
    phase_states: Dict[str, PhaseStatus] = Field(default_factory=dict)
    phase_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def set_phase_output(self, phase_name: str, outputs: Dict[str, Any]) -> None:
        """Store phase outputs for downstream consumption."""
        self.phase_outputs[phase_name] = outputs

    def get_phase_output(self, phase_name: str) -> Dict[str, Any]:
        """Retrieve outputs from a previous phase."""
        return self.phase_outputs.get(phase_name, {})

    def mark_phase(self, phase_name: str, status: PhaseStatus) -> None:
        """Record phase status for checkpoint/resume."""
        self.phase_states[phase_name] = status

    def is_phase_completed(self, phase_name: str) -> bool:
        """Check if a phase has already completed."""
        return self.phase_states.get(phase_name) == PhaseStatus.COMPLETED


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


# =============================================================================
# DATA MODELS - CROSS-REGULATION SYNC
# =============================================================================


class CbamChange(BaseModel):
    """A single CBAM data change record."""
    change_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    change_type: ChangeType = Field(...)
    timestamp: str = Field(default="")
    entity_id: Optional[str] = Field(None)
    record_id: Optional[str] = Field(None)
    previous_value: Optional[Dict[str, Any]] = Field(None)
    new_value: Optional[Dict[str, Any]] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CrossRegulationSyncInput(BaseModel):
    """Input configuration for cross-regulation sync."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2026, le=2050)
    last_sync_timestamp: Optional[str] = Field(
        None, description="ISO timestamp of last successful sync"
    )
    changes: List[CbamChange] = Field(
        default_factory=list,
        description="Explicit changes (if not auto-detected)"
    )
    target_regulations: List[TargetRegulation] = Field(
        default_factory=lambda: list(TargetRegulation),
        description="Regulations to sync to"
    )
    cbam_data_snapshot: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current CBAM data for mapping"
    )
    export_formats: List[ExportFormat] = Field(
        default_factory=lambda: [ExportFormat.JSON]
    )
    notify_teams: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Regulation -> list of contact emails"
    )
    skip_phases: List[str] = Field(default_factory=list)


class CrossRegulationSyncResult(WorkflowResult):
    """Complete result from cross-regulation sync."""
    changes_detected: int = Field(default=0)
    regulations_affected: int = Field(default=0)
    exports_generated: int = Field(default=0)
    conflicts_found: int = Field(default=0)
    notifications_sent: int = Field(default=0)


# =============================================================================
# PHASE IMPLEMENTATIONS
# =============================================================================


class ChangeDetectionPhase:
    """
    Phase 1: Change Detection.

    Identifies new or modified CBAM data since the last sync. Computes
    the delta (new imports, updated calculations, certificate changes)
    and determines which target regulations are affected.
    """

    PHASE_NAME = "change_detection"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute change detection phase.

        Args:
            context: Workflow context with CBAM data and sync history.

        Returns:
            PhaseResult with detected changes and affected regulations.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            explicit_changes = config.get("changes", [])
            last_sync = config.get("last_sync_timestamp")
            target_regs = config.get("target_regulations", [])
            cbam_data = config.get("cbam_data_snapshot", {})

            # Use explicit changes if provided, otherwise detect
            if explicit_changes:
                detected_changes = explicit_changes
            else:
                detected_changes = await self._detect_changes(
                    cbam_data, last_sync
                )

            # Classify changes by type
            by_type: Dict[str, List[Dict[str, Any]]] = {}
            for change in detected_changes:
                ct = change.get("change_type", "")
                if ct not in by_type:
                    by_type[ct] = []
                by_type[ct].append(change)

            outputs["changes_detected"] = len(detected_changes)
            outputs["changes_by_type"] = {
                k: len(v) for k, v in by_type.items()
            }
            outputs["change_details"] = detected_changes

            # Determine affected regulations
            affected_regs: Set[str] = set()
            for change in detected_changes:
                ct = change.get("change_type", "")
                mapped_regs = CHANGE_REGULATION_MAP.get(ct, [])
                for reg in mapped_regs:
                    if not target_regs or reg in target_regs:
                        affected_regs.add(reg)

            outputs["affected_regulations"] = sorted(affected_regs)
            outputs["regulations_count"] = len(affected_regs)

            # Per-regulation change summary
            reg_changes: Dict[str, List[str]] = {}
            for reg in affected_regs:
                reg_change_types = []
                for ct, changes in by_type.items():
                    if reg in CHANGE_REGULATION_MAP.get(ct, []):
                        reg_change_types.append(ct)
                reg_changes[reg] = reg_change_types
            outputs["regulation_change_map"] = reg_changes

            if not detected_changes:
                warnings.append("No CBAM data changes detected since last sync")

            outputs["sync_window"] = {
                "from": last_sync or "initial_sync",
                "to": datetime.utcnow().isoformat(),
            }

            status = PhaseStatus.COMPLETED
            records = len(detected_changes)

        except Exception as exc:
            logger.error("ChangeDetection failed: %s", exc, exc_info=True)
            errors.append(f"Change detection failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
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

    async def _detect_changes(
        self,
        cbam_data: Dict[str, Any],
        last_sync: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        Auto-detect CBAM data changes since last sync.

        In production, this queries the CBAM data store for records
        modified after last_sync timestamp.
        """
        logger.info(
            "Auto-detecting CBAM changes since %s",
            last_sync or "beginning",
        )
        return []


class MappingPhase:
    """
    Phase 2: Mapping.

    Applies regulation-specific mapping rules via the CrossRegulationEngine
    to transform CBAM data into the format required by each affected
    target regulation.
    """

    PHASE_NAME = "mapping"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute regulation mapping phase.

        Args:
            context: Workflow context with detected changes.

        Returns:
            PhaseResult with per-regulation mapped data.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            detection = context.get_phase_output("change_detection")
            affected_regs = detection.get("affected_regulations", [])
            cbam_data = config.get("cbam_data_snapshot", {})
            changes = detection.get("change_details", [])

            mapped_data: Dict[str, Dict[str, Any]] = {}
            mapping_errors: List[Dict[str, Any]] = []

            for reg in affected_regs:
                try:
                    reg_mapped = await self._map_to_regulation(
                        reg, cbam_data, changes
                    )
                    mapped_data[reg] = reg_mapped
                    logger.info(
                        "Mapped CBAM data to %s: %d data points",
                        reg, reg_mapped.get("data_point_count", 0),
                    )
                except Exception as exc:
                    mapping_errors.append({
                        "regulation": reg,
                        "error": str(exc),
                    })
                    warnings.append(
                        f"Mapping to {reg} failed: {str(exc)}"
                    )

            outputs["mapped_regulations"] = list(mapped_data.keys())
            outputs["mapped_data"] = mapped_data
            outputs["mapping_errors"] = mapping_errors
            outputs["successful_mappings"] = len(mapped_data)
            outputs["failed_mappings"] = len(mapping_errors)

            status = PhaseStatus.COMPLETED
            records = len(mapped_data)

        except Exception as exc:
            logger.error("Mapping failed: %s", exc, exc_info=True)
            errors.append(f"Regulation mapping failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
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

    async def _map_to_regulation(
        self,
        regulation: str,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Map CBAM data to a specific regulation format.

        In production, this delegates to the CrossRegulationEngine
        which holds the mapping rules for each regulation.
        """
        reg_info = REGULATION_SECTIONS.get(regulation, {})

        if regulation == TargetRegulation.CSRD_ESRS_E1.value:
            return self._map_csrd_esrs_e1(cbam_data, changes, reg_info)
        elif regulation == TargetRegulation.CDP_CLIMATE.value:
            return self._map_cdp_climate(cbam_data, changes, reg_info)
        elif regulation == TargetRegulation.SBTI_SCOPE3.value:
            return self._map_sbti_scope3(cbam_data, changes, reg_info)
        elif regulation == TargetRegulation.EU_TAXONOMY.value:
            return self._map_eu_taxonomy(cbam_data, changes, reg_info)
        elif regulation == TargetRegulation.EU_ETS.value:
            return self._map_eu_ets(cbam_data, changes, reg_info)
        elif regulation == TargetRegulation.EUDR.value:
            return self._map_eudr(cbam_data, changes, reg_info)
        else:
            return {"regulation": regulation, "data_point_count": 0}

    def _map_csrd_esrs_e1(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to CSRD ESRS E1 format."""
        emissions = cbam_data.get("total_embedded_emissions_tco2e", 0)
        certificates = cbam_data.get("certificates_purchased", 0)
        cert_cost = cbam_data.get("certificate_cost_eur", 0)
        return {
            "regulation": TargetRegulation.CSRD_ESRS_E1.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "E1_6_scope3_cat1_upstream": emissions,
                "E1_8_internal_carbon_price": cert_cost / max(emissions, 1),
                "E1_8_cbam_certificates_purchased": certificates,
                "E1_8_cbam_cost_eur": cert_cost,
            },
            "data_point_count": 4,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }

    def _map_cdp_climate(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to CDP Climate Change format."""
        emissions = cbam_data.get("total_embedded_emissions_tco2e", 0)
        by_category = cbam_data.get("by_goods_category", {})
        return {
            "regulation": TargetRegulation.CDP_CLIMATE.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "C6_scope3_cat1_emissions_tco2e": emissions,
                "C7_emissions_by_category": by_category,
                "C11_carbon_pricing_cbam_cost": cbam_data.get(
                    "certificate_cost_eur", 0
                ),
                "C11_carbon_pricing_type": "CBAM_CERTIFICATE",
            },
            "data_point_count": 4,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }

    def _map_sbti_scope3(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to SBTi Scope 3 format."""
        emissions = cbam_data.get("total_embedded_emissions_tco2e", 0)
        return {
            "regulation": TargetRegulation.SBTI_SCOPE3.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "scope3_cat1_purchased_goods_tco2e": emissions,
                "scope3_cat1_cbam_covered_tco2e": emissions,
                "methodology": "CBAM_verified_embedded_emissions",
            },
            "data_point_count": 3,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }

    def _map_eu_taxonomy(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to EU Taxonomy format."""
        emissions = cbam_data.get("total_embedded_emissions_tco2e", 0)
        imports_tonnes = cbam_data.get("total_imports_tonnes", 0)
        intensity = emissions / max(imports_tonnes, 1)
        return {
            "regulation": TargetRegulation.EU_TAXONOMY.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "climate_mitigation_carbon_intensity": intensity,
                "cbam_covered_imports_tonnes": imports_tonnes,
                "cbam_embedded_emissions_tco2e": emissions,
                "dnsh_assessment_source": "CBAM_embedded_emissions",
            },
            "data_point_count": 4,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }

    def _map_eu_ets(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to EU ETS format."""
        return {
            "regulation": TargetRegulation.EU_ETS.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "cbam_certificates_purchased": cbam_data.get(
                    "certificates_purchased", 0
                ),
                "cbam_certificates_surrendered": cbam_data.get(
                    "certificates_surrendered", 0
                ),
                "ets_price_at_purchase_eur": cbam_data.get(
                    "ets_price_eur", 0
                ),
                "free_allocation_deduction_tco2e": cbam_data.get(
                    "free_allocation_deduction_tco2e", 0
                ),
            },
            "data_point_count": 4,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }

    def _map_eudr(
        self,
        cbam_data: Dict[str, Any],
        changes: List[Dict[str, Any]],
        reg_info: Dict[str, str],
    ) -> Dict[str, Any]:
        """Map CBAM data to EUDR format for supply chain overlap."""
        suppliers = cbam_data.get("suppliers", [])
        commodities = cbam_data.get("by_goods_category", {})
        return {
            "regulation": TargetRegulation.EUDR.value,
            "section": reg_info.get("section", ""),
            "data_points": {
                "cbam_suppliers_count": len(suppliers),
                "cbam_commodities": list(commodities.keys()),
                "supply_chain_overlap_assessment": "pending",
                "deforestation_risk_commodities": [],
            },
            "data_point_count": 4,
            "source": "CBAM_PACK_005",
            "change_count": len(changes),
        }


class OutputGenerationPhase:
    """
    Phase 3: Output Generation.

    Generates regulation-specific data exports in requested formats
    (JSON, CSV) and creates a cross-regulation consistency report
    flagging any conflicts between regulation requirements.
    """

    PHASE_NAME = "output_generation"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute output generation phase.

        Args:
            context: Workflow context with mapped data.

        Returns:
            PhaseResult with export manifests and consistency report.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            mapping = context.get_phase_output("mapping")
            mapped_data = mapping.get("mapped_data", {})
            export_formats = config.get("export_formats", ["JSON"])
            org_id = config.get("organization_id", "")
            year = config.get("reporting_year", 0)

            # Generate exports
            exports: List[Dict[str, Any]] = []
            for reg, data in mapped_data.items():
                for fmt in export_formats:
                    export = await self._generate_export(
                        reg, data, fmt, org_id, year
                    )
                    exports.append(export)

            outputs["exports"] = exports
            outputs["exports_count"] = len(exports)

            # Consistency report
            consistency = self._check_consistency(mapped_data)
            outputs["consistency_report"] = consistency
            outputs["conflicts_found"] = len(
                consistency.get("conflicts", [])
            )

            if consistency.get("conflicts"):
                for conflict in consistency["conflicts"]:
                    severity = conflict.get("severity", "WARNING")
                    if severity == ConflictSeverity.CRITICAL.value:
                        errors.append(
                            f"Cross-regulation conflict: "
                            f"{conflict.get('description', '')}"
                        )
                    else:
                        warnings.append(
                            f"Cross-regulation {severity}: "
                            f"{conflict.get('description', '')}"
                        )

            status = PhaseStatus.COMPLETED
            records = len(exports)

        except Exception as exc:
            logger.error("OutputGeneration failed: %s", exc, exc_info=True)
            errors.append(f"Output generation failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
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

    async def _generate_export(
        self,
        regulation: str,
        data: Dict[str, Any],
        export_format: str,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """Generate a single regulation export."""
        export_id = str(uuid.uuid4())
        return {
            "export_id": export_id,
            "regulation": regulation,
            "format": export_format,
            "organization_id": org_id,
            "reporting_year": year,
            "data_point_count": data.get("data_point_count", 0),
            "generated_at": datetime.utcnow().isoformat(),
            "file_reference": (
                f"cbam_sync_{regulation.lower()}_{year}_{export_id[:8]}"
                f".{export_format.lower()}"
            ),
        }

    def _check_consistency(
        self, mapped_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check cross-regulation consistency of mapped data.

        Verifies that the same underlying CBAM data is represented
        consistently across different regulations.
        """
        conflicts: List[Dict[str, Any]] = []

        # Extract emissions values across regulations
        emissions_values: Dict[str, float] = {}
        for reg, data in mapped_data.items():
            dp = data.get("data_points", {})
            for key, val in dp.items():
                if isinstance(val, (int, float)) and "emission" in key.lower():
                    emissions_values[f"{reg}.{key}"] = val

        # Check CSRD vs CDP emissions consistency
        csrd_data = mapped_data.get(TargetRegulation.CSRD_ESRS_E1.value, {})
        cdp_data = mapped_data.get(TargetRegulation.CDP_CLIMATE.value, {})
        csrd_emissions = csrd_data.get("data_points", {}).get(
            "E1_6_scope3_cat1_upstream", 0
        )
        cdp_emissions = cdp_data.get("data_points", {}).get(
            "C6_scope3_cat1_emissions_tco2e", 0
        )
        if csrd_emissions > 0 and cdp_emissions > 0:
            if abs(csrd_emissions - cdp_emissions) > 0.01:
                conflicts.append({
                    "severity": ConflictSeverity.CRITICAL.value,
                    "regulations": [
                        TargetRegulation.CSRD_ESRS_E1.value,
                        TargetRegulation.CDP_CLIMATE.value,
                    ],
                    "field": "scope3_cat1_emissions",
                    "description": (
                        f"CSRD ESRS E1 emissions ({csrd_emissions:.4f}) "
                        f"differs from CDP C6 ({cdp_emissions:.4f})"
                    ),
                    "csrd_value": csrd_emissions,
                    "cdp_value": cdp_emissions,
                })

        # Check CSRD vs SBTi Scope 3 consistency
        sbti_data = mapped_data.get(TargetRegulation.SBTI_SCOPE3.value, {})
        sbti_emissions = sbti_data.get("data_points", {}).get(
            "scope3_cat1_purchased_goods_tco2e", 0
        )
        if csrd_emissions > 0 and sbti_emissions > 0:
            if abs(csrd_emissions - sbti_emissions) > 0.01:
                conflicts.append({
                    "severity": ConflictSeverity.WARNING.value,
                    "regulations": [
                        TargetRegulation.CSRD_ESRS_E1.value,
                        TargetRegulation.SBTI_SCOPE3.value,
                    ],
                    "field": "scope3_cat1_emissions",
                    "description": (
                        f"CSRD emissions ({csrd_emissions:.4f}) differs "
                        f"from SBTi ({sbti_emissions:.4f}). Boundary "
                        f"differences may explain variance."
                    ),
                })

        return {
            "checked_at": datetime.utcnow().isoformat(),
            "regulations_checked": list(mapped_data.keys()),
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
            "consistent": len(conflicts) == 0,
        }


class SyncNotificationPhase:
    """
    Phase 4: Notification.

    Alerts regulation-specific compliance teams of updated data,
    provides links to updated reports, and logs sync completion
    for the audit trail.
    """

    PHASE_NAME = "notification"

    async def execute(self, context: WorkflowContext) -> PhaseResult:
        """
        Execute notification phase.

        Args:
            context: Workflow context with exports and consistency data.

        Returns:
            PhaseResult with notification delivery results.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            config = context.config
            org_id = config.get("organization_id", "")
            year = config.get("reporting_year", 0)
            notify_teams = config.get("notify_teams", {})
            detection = context.get_phase_output("change_detection")
            export_output = context.get_phase_output("output_generation")
            affected_regs = detection.get("affected_regulations", [])
            exports = export_output.get("exports", [])
            consistency = export_output.get("consistency_report", {})

            # Send per-regulation notifications
            notifications: List[Dict[str, Any]] = []
            for reg in affected_regs:
                contacts = notify_teams.get(reg, [])
                if not contacts:
                    contacts = notify_teams.get("default", [])

                reg_exports = [
                    e for e in exports if e.get("regulation") == reg
                ]
                reg_info = REGULATION_SECTIONS.get(reg, {})

                notification = {
                    "notification_id": str(uuid.uuid4()),
                    "regulation": reg,
                    "section": reg_info.get("section", reg),
                    "recipients": contacts,
                    "subject": (
                        f"CBAM Data Sync Update - {reg_info.get('section', reg)} "
                        f"({year})"
                    ),
                    "body": {
                        "changes_detected": detection.get(
                            "changes_detected", 0
                        ),
                        "data_points_updated": sum(
                            e.get("data_point_count", 0)
                            for e in reg_exports
                        ),
                        "exports": [
                            e.get("file_reference", "")
                            for e in reg_exports
                        ],
                        "consistency_status": (
                            "CONSISTENT" if consistency.get("consistent", True)
                            else "CONFLICTS_DETECTED"
                        ),
                    },
                    "sent_at": datetime.utcnow().isoformat(),
                    "delivery_status": (
                        "sent" if contacts else "no_recipients"
                    ),
                }
                notifications.append(notification)

                if not contacts:
                    warnings.append(
                        f"No notification recipients configured for {reg}"
                    )

            outputs["notifications"] = notifications
            outputs["notifications_sent"] = sum(
                1 for n in notifications
                if n.get("delivery_status") == "sent"
            )
            outputs["notifications_skipped"] = sum(
                1 for n in notifications
                if n.get("delivery_status") == "no_recipients"
            )

            # Audit trail log
            sync_log = {
                "log_id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "organization_id": org_id,
                "reporting_year": year,
                "workflow_id": context.workflow_id,
                "changes_synced": detection.get("changes_detected", 0),
                "regulations_updated": affected_regs,
                "exports_generated": len(exports),
                "conflicts_found": consistency.get("conflict_count", 0),
                "notifications_sent": outputs["notifications_sent"],
            }
            outputs["sync_audit_log"] = sync_log

            status = PhaseStatus.COMPLETED

        except Exception as exc:
            logger.error("Notification failed: %s", exc, exc_info=True)
            errors.append(f"Notification failed: {str(exc)}")
            status = PhaseStatus.FAILED

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name=self.PHASE_NAME,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
            outputs=outputs,
            errors=errors,
            warnings=warnings,
            provenance_hash=_hash_data(outputs),
        )


# =============================================================================
# WORKFLOW ORCHESTRATOR
# =============================================================================


class CrossRegulationSyncWorkflow:
    """
    Four-phase cross-regulation sync workflow.

    Triggered by CBAM data changes, this workflow detects deltas,
    maps data to affected regulations, generates exports, and
    notifies compliance teams. Supports checkpoint/resume.

    Attributes:
        workflow_id: Unique execution identifier.
        _phases: Ordered phase executors.
        _progress_callback: Optional progress notification callback.

    Example:
        >>> wf = CrossRegulationSyncWorkflow()
        >>> input_data = CrossRegulationSyncInput(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     changes=[CbamChange(change_type=ChangeType.NEW_IMPORT)],
        ... )
        >>> result = await wf.run(input_data)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "cross_regulation_sync"

    PHASE_ORDER = [
        "change_detection",
        "mapping",
        "output_generation",
        "notification",
    ]

    def __init__(
        self,
        progress_callback: Optional[Callable[[str, str, float], None]] = None,
    ) -> None:
        """
        Initialize cross-regulation sync workflow.

        Args:
            progress_callback: Optional callback(phase, message, pct).
        """
        self.workflow_id: str = str(uuid.uuid4())
        self._progress_callback = progress_callback
        self._phases: Dict[str, Any] = {
            "change_detection": ChangeDetectionPhase(),
            "mapping": MappingPhase(),
            "output_generation": OutputGenerationPhase(),
            "notification": SyncNotificationPhase(),
        }

    async def run(
        self, input_data: CrossRegulationSyncInput
    ) -> CrossRegulationSyncResult:
        """
        Execute the 4-phase cross-regulation sync workflow.

        Args:
            input_data: Validated workflow input configuration.

        Returns:
            CrossRegulationSyncResult with sync outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting cross-regulation sync %s for org=%s year=%d",
            self.workflow_id, input_data.organization_id,
            input_data.reporting_year,
        )

        context = WorkflowContext(
            workflow_id=self.workflow_id,
            organization_id=input_data.organization_id,
            config=self._build_config(input_data),
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING

        for idx, phase_name in enumerate(self.PHASE_ORDER):
            if phase_name in input_data.skip_phases:
                skip_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.SKIPPED,
                    provenance_hash=_hash_data({"skipped": True}),
                )
                completed_phases.append(skip_result)
                context.mark_phase(phase_name, PhaseStatus.SKIPPED)
                continue

            if context.is_phase_completed(phase_name):
                continue

            pct = idx / len(self.PHASE_ORDER)
            self._notify_progress(phase_name, f"Starting: {phase_name}", pct)
            context.mark_phase(phase_name, PhaseStatus.RUNNING)

            try:
                phase_result = await self._phases[phase_name].execute(context)
                completed_phases.append(phase_result)

                if phase_result.status == PhaseStatus.COMPLETED:
                    context.set_phase_output(phase_name, phase_result.outputs)
                    context.mark_phase(phase_name, PhaseStatus.COMPLETED)
                else:
                    context.mark_phase(phase_name, phase_result.status)
                    if phase_name == "change_detection":
                        overall_status = WorkflowStatus.FAILED
                        break

                context.errors.extend(phase_result.errors)
                context.warnings.extend(phase_result.warnings)

            except Exception as exc:
                logger.error(
                    "Phase '%s' raised: %s", phase_name, exc, exc_info=True
                )
                error_result = PhaseResult(
                    phase_name=phase_name,
                    status=PhaseStatus.FAILED,
                    errors=[str(exc)],
                    provenance_hash=_hash_data({"error": str(exc)}),
                )
                completed_phases.append(error_result)
                context.mark_phase(phase_name, PhaseStatus.FAILED)
                overall_status = WorkflowStatus.FAILED
                break

        if overall_status == WorkflowStatus.RUNNING:
            all_ok = all(
                p.status in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED)
                for p in completed_phases
            )
            overall_status = (
                WorkflowStatus.COMPLETED if all_ok else WorkflowStatus.PARTIAL
            )

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()
        summary = self._build_summary(context)
        provenance = _hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in completed_phases],
        })

        self._notify_progress(
            "workflow", f"Workflow {overall_status.value}", 1.0
        )

        return CrossRegulationSyncResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=total_duration,
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            changes_detected=summary.get("changes_detected", 0),
            regulations_affected=summary.get("regulations_affected", 0),
            exports_generated=summary.get("exports_generated", 0),
            conflicts_found=summary.get("conflicts_found", 0),
            notifications_sent=summary.get("notifications_sent", 0),
        )

    def _build_config(
        self, input_data: CrossRegulationSyncInput
    ) -> Dict[str, Any]:
        """Transform input model to config dict for phases."""
        return {
            "organization_id": input_data.organization_id,
            "reporting_year": input_data.reporting_year,
            "last_sync_timestamp": input_data.last_sync_timestamp,
            "changes": [c.model_dump() for c in input_data.changes],
            "target_regulations": [
                r.value for r in input_data.target_regulations
            ],
            "cbam_data_snapshot": input_data.cbam_data_snapshot,
            "export_formats": [f.value for f in input_data.export_formats],
            "notify_teams": input_data.notify_teams,
        }

    def _build_summary(self, context: WorkflowContext) -> Dict[str, Any]:
        """Build workflow summary from phase outputs."""
        detection = context.get_phase_output("change_detection")
        export_out = context.get_phase_output("output_generation")
        notif_out = context.get_phase_output("notification")
        return {
            "changes_detected": detection.get("changes_detected", 0),
            "regulations_affected": detection.get("regulations_count", 0),
            "exports_generated": export_out.get("exports_count", 0),
            "conflicts_found": export_out.get("conflicts_found", 0),
            "notifications_sent": notif_out.get("notifications_sent", 0),
        }

    def _notify_progress(
        self, phase: str, message: str, pct: float
    ) -> None:
        """Send progress notification via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(phase, message, min(pct, 1.0))
            except Exception:
                logger.debug("Progress callback failed for phase=%s", phase)


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
