# -*- coding: utf-8 -*-
"""
Data Consistency Reconciliation Workflow
=============================================

Four-phase workflow that extracts shared data points from all four
constituent regulation packs (CSRD, CBAM, EU Taxonomy, EUDR), compares
them for consistency, flags discrepancies with severity ratings, and
auto-resolves or creates manual resolution tasks.

Phases:
    1. Extract - Extract shared data points from all packs
    2. Compare - Compare extracted data for consistency
    3. Flag - Flag discrepancies with severity ratings
    4. Resolve - Auto-resolve or create manual resolution tasks

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import logging
import uuid
from datetime import datetime
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


class DiscrepancySeverity(str, Enum):
    """Severity of a data discrepancy."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ResolutionMethod(str, Enum):
    """Method used to resolve a discrepancy."""
    AUTO_LATEST = "AUTO_LATEST"
    AUTO_AVERAGE = "AUTO_AVERAGE"
    AUTO_PRIMARY_SOURCE = "AUTO_PRIMARY_SOURCE"
    AUTO_TOLERANCE = "AUTO_TOLERANCE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    UNRESOLVED = "UNRESOLVED"


class DataPointType(str, Enum):
    """Type of shared data point."""
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    DATE = "DATE"
    BOOLEAN = "BOOLEAN"
    ENUM = "ENUM"


# =============================================================================
# SHARED DATA POINT DEFINITIONS
# =============================================================================


SHARED_DATA_POINTS: List[Dict[str, Any]] = [
    {"point_id": "SDP-001", "name": "Scope 1 GHG Emissions", "field": "scope1_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "packs": ["CSRD", "CBAM", "EU_TAXONOMY"], "tolerance_pct": 1.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-002", "name": "Scope 2 GHG Emissions", "field": "scope2_tco2e", "data_type": "NUMERIC", "unit": "tCO2e", "packs": ["CSRD", "CBAM", "EU_TAXONOMY"], "tolerance_pct": 1.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-003", "name": "Total Energy Consumption", "field": "energy_mwh", "data_type": "NUMERIC", "unit": "MWh", "packs": ["CSRD", "EU_TAXONOMY"], "tolerance_pct": 2.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-004", "name": "Renewable Energy Percentage", "field": "renewable_pct", "data_type": "NUMERIC", "unit": "%", "packs": ["CSRD", "EU_TAXONOMY"], "tolerance_pct": 1.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-005", "name": "Water Consumption", "field": "water_m3", "data_type": "NUMERIC", "unit": "m3", "packs": ["CSRD", "EU_TAXONOMY"], "tolerance_pct": 3.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-006", "name": "Import Volume", "field": "import_tonnes", "data_type": "NUMERIC", "unit": "tonnes", "packs": ["CBAM", "EUDR"], "tolerance_pct": 0.5, "primary_source": "CBAM", "auto_resolvable": True},
    {"point_id": "SDP-007", "name": "Country of Origin", "field": "country_of_origin", "data_type": "TEXT", "unit": None, "packs": ["CBAM", "EUDR"], "tolerance_pct": 0.0, "primary_source": "CBAM", "auto_resolvable": False},
    {"point_id": "SDP-008", "name": "Supplier Identifier", "field": "supplier_id", "data_type": "TEXT", "unit": None, "packs": ["CBAM", "EUDR"], "tolerance_pct": 0.0, "primary_source": "CBAM", "auto_resolvable": False},
    {"point_id": "SDP-009", "name": "Carbon Price (EUR)", "field": "carbon_price_eur", "data_type": "NUMERIC", "unit": "EUR", "packs": ["CSRD", "CBAM"], "tolerance_pct": 0.1, "primary_source": "CBAM", "auto_resolvable": True},
    {"point_id": "SDP-010", "name": "Reporting Period Start", "field": "reporting_period_start", "data_type": "DATE", "unit": None, "packs": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"], "tolerance_pct": 0.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-011", "name": "Reporting Period End", "field": "reporting_period_end", "data_type": "DATE", "unit": None, "packs": ["CSRD", "CBAM", "EU_TAXONOMY", "EUDR"], "tolerance_pct": 0.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-012", "name": "Waste Generated", "field": "waste_tonnes", "data_type": "NUMERIC", "unit": "tonnes", "packs": ["CSRD", "EU_TAXONOMY"], "tolerance_pct": 3.0, "primary_source": "CSRD", "auto_resolvable": True},
    {"point_id": "SDP-013", "name": "Governance Documentation Status", "field": "governance_status", "data_type": "ENUM", "unit": None, "packs": ["CSRD", "EU_TAXONOMY"], "tolerance_pct": 0.0, "primary_source": "CSRD", "auto_resolvable": False},
    {"point_id": "SDP-014", "name": "Embedded Emissions per Tonne", "field": "embedded_emissions_per_tonne", "data_type": "NUMERIC", "unit": "tCO2e/t", "packs": ["CBAM", "CSRD"], "tolerance_pct": 2.0, "primary_source": "CBAM", "auto_resolvable": True},
    {"point_id": "SDP-015", "name": "Supply Chain Risk Score", "field": "supply_chain_risk", "data_type": "NUMERIC", "unit": "score", "packs": ["EUDR", "CBAM"], "tolerance_pct": 5.0, "primary_source": "EUDR", "auto_resolvable": True},
]


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
    """Configuration for data consistency reconciliation workflow."""
    organization_id: str = Field(...)
    reporting_year: int = Field(..., ge=2024, le=2050)
    target_packs: List[RegulationPack] = Field(
        default_factory=lambda: list(RegulationPack)
    )
    pack_data: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-pack data snapshots keyed by pack name"
    )
    auto_resolve_enabled: bool = Field(
        default=True,
        description="Enable automatic resolution of discrepancies within tolerance"
    )
    custom_tolerances: Dict[str, float] = Field(
        default_factory=dict,
        description="Override tolerance_pct for specific data point IDs"
    )
    resolution_assignees: Dict[str, str] = Field(
        default_factory=dict,
        description="Pack name -> assignee for manual resolution tasks"
    )
    skip_phases: List[str] = Field(default_factory=list)


class DataConsistencyReconciliationResult(WorkflowResult):
    """Result from data consistency reconciliation workflow."""
    data_points_compared: int = Field(default=0)
    discrepancies_found: int = Field(default=0)
    auto_resolved: int = Field(default=0)
    manual_tasks_created: int = Field(default=0)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DataConsistencyReconciliationWorkflow:
    """
    Four-phase data consistency reconciliation workflow.

    Extracts shared data, compares for consistency, flags discrepancies,
    and resolves them automatically or creates manual tasks.

    Example:
        >>> wf = DataConsistencyReconciliationWorkflow()
        >>> config = WorkflowConfig(
        ...     organization_id="org-123",
        ...     reporting_year=2026,
        ...     pack_data={
        ...         "CSRD": {"scope1_tco2e": 15000.0},
        ...         "CBAM": {"scope1_tco2e": 15050.0},
        ...     }
        ... )
        >>> result = wf.execute(config)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    WORKFLOW_NAME = "data_consistency_reconciliation"

    PHASE_ORDER = [
        "extract",
        "compare",
        "flag",
        "resolve",
    ]

    def __init__(self) -> None:
        """Initialize the data consistency reconciliation workflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self._phase_outputs: Dict[str, Dict[str, Any]] = {}

    def execute(self, config: WorkflowConfig) -> DataConsistencyReconciliationResult:
        """
        Execute the four-phase data consistency reconciliation workflow.

        Args:
            config: Validated workflow configuration.

        Returns:
            DataConsistencyReconciliationResult with reconciliation outcomes.
        """
        started_at = datetime.utcnow()
        logger.info(
            "Starting data consistency reconciliation %s for org=%s year=%d",
            self.workflow_id, config.organization_id, config.reporting_year,
        )

        completed_phases: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        phase_methods = {
            "extract": self._phase_extract,
            "compare": self._phase_compare,
            "flag": self._phase_flag,
            "resolve": self._phase_resolve,
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

        return DataConsistencyReconciliationResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            started_at=started_at,
            completed_at=completed_at,
            total_duration_seconds=(completed_at - started_at).total_seconds(),
            phases=completed_phases,
            summary=summary,
            provenance_hash=provenance,
            data_points_compared=summary.get("data_points_compared", 0),
            discrepancies_found=summary.get("discrepancies_found", 0),
            auto_resolved=summary.get("auto_resolved", 0),
            manual_tasks_created=summary.get("manual_tasks_created", 0),
        )

    # -------------------------------------------------------------------------
    # Phase 1: Extract
    # -------------------------------------------------------------------------

    def _phase_extract(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 1: Extract shared data points from all packs.

        Retrieves the value of each shared data point from each
        constituent pack's data snapshot.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            active_packs = {p.value for p in config.target_packs}
            extracted: List[Dict[str, Any]] = []

            for sdp in SHARED_DATA_POINTS:
                relevant_packs = [p for p in sdp["packs"] if p in active_packs]
                if len(relevant_packs) < 2:
                    continue

                pack_values: Dict[str, Any] = {}
                pack_timestamps: Dict[str, str] = {}
                missing_packs: List[str] = []

                for pack_name in relevant_packs:
                    pack_data = config.pack_data.get(pack_name, {})
                    field = sdp["field"]

                    if field in pack_data:
                        pack_values[pack_name] = pack_data[field]
                        pack_timestamps[pack_name] = pack_data.get(
                            f"{field}_timestamp",
                            datetime.utcnow().isoformat()
                        )
                    else:
                        missing_packs.append(pack_name)

                if len(pack_values) < 2:
                    if missing_packs:
                        warnings.append(
                            f"Data point '{sdp['name']}' missing from: "
                            f"{', '.join(missing_packs)}"
                        )
                    continue

                extracted.append({
                    "point_id": sdp["point_id"],
                    "name": sdp["name"],
                    "field": sdp["field"],
                    "data_type": sdp["data_type"],
                    "unit": sdp.get("unit"),
                    "packs": relevant_packs,
                    "pack_values": pack_values,
                    "pack_timestamps": pack_timestamps,
                    "missing_packs": missing_packs,
                    "tolerance_pct": config.custom_tolerances.get(
                        sdp["point_id"], sdp["tolerance_pct"]
                    ),
                    "primary_source": sdp["primary_source"],
                    "auto_resolvable": sdp["auto_resolvable"],
                    "extracted_at": datetime.utcnow().isoformat(),
                })

            outputs["extracted_points"] = extracted
            outputs["total_extracted"] = len(extracted)
            outputs["points_with_missing"] = sum(
                1 for e in extracted if e["missing_packs"]
            )

            by_type: Dict[str, int] = {}
            for e in extracted:
                dt = e["data_type"]
                by_type[dt] = by_type.get(dt, 0) + 1
            outputs["by_data_type"] = by_type

            logger.info(
                "Extraction complete: %d shared data points extracted",
                len(extracted),
            )

            status = PhaseStatus.COMPLETED
            records = len(extracted)

        except Exception as exc:
            logger.error("Extraction failed: %s", exc, exc_info=True)
            errors.append(f"Extraction failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="extract",
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
    # Phase 2: Compare
    # -------------------------------------------------------------------------

    def _phase_compare(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 2: Compare extracted data points for consistency.

        Performs pairwise comparisons between pack values for each
        shared data point and calculates variance metrics.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            extract_out = self._phase_outputs.get("extract", {})
            extracted = extract_out.get("extracted_points", [])

            comparisons: List[Dict[str, Any]] = []

            for point in extracted:
                comparison = self._compare_point(point)
                comparisons.append(comparison)

            outputs["comparisons"] = comparisons
            outputs["total_compared"] = len(comparisons)
            outputs["consistent_count"] = sum(
                1 for c in comparisons if c["is_consistent"]
            )
            outputs["inconsistent_count"] = sum(
                1 for c in comparisons if not c["is_consistent"]
            )

            logger.info(
                "Comparison complete: %d points, %d consistent, %d inconsistent",
                len(comparisons),
                outputs["consistent_count"],
                outputs["inconsistent_count"],
            )

            status = PhaseStatus.COMPLETED
            records = len(comparisons)

        except Exception as exc:
            logger.error("Comparison failed: %s", exc, exc_info=True)
            errors.append(f"Comparison failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="compare",
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

    def _compare_point(self, point: Dict[str, Any]) -> Dict[str, Any]:
        """Compare values of a single data point across packs."""
        data_type = point["data_type"]
        pack_values = point["pack_values"]
        tolerance_pct = point["tolerance_pct"]

        if data_type == DataPointType.NUMERIC.value:
            return self._compare_numeric(point, pack_values, tolerance_pct)
        elif data_type == DataPointType.TEXT.value:
            return self._compare_text(point, pack_values)
        elif data_type == DataPointType.DATE.value:
            return self._compare_date(point, pack_values)
        elif data_type == DataPointType.ENUM.value:
            return self._compare_text(point, pack_values)
        elif data_type == DataPointType.BOOLEAN.value:
            return self._compare_text(point, pack_values)
        else:
            return self._compare_text(point, pack_values)

    def _compare_numeric(
        self,
        point: Dict[str, Any],
        pack_values: Dict[str, Any],
        tolerance_pct: float,
    ) -> Dict[str, Any]:
        """Compare numeric values across packs."""
        values = {}
        for pack, val in pack_values.items():
            if isinstance(val, (int, float)):
                values[pack] = float(val)

        if len(values) < 2:
            return {
                "point_id": point["point_id"],
                "name": point["name"],
                "data_type": point["data_type"],
                "pack_values": pack_values,
                "is_consistent": True,
                "variance_pct": 0.0,
                "max_deviation": 0.0,
                "comparison_type": "numeric",
                "pairwise_results": [],
            }

        all_vals = list(values.values())
        mean_val = sum(all_vals) / len(all_vals)
        max_val = max(all_vals)
        min_val = min(all_vals)

        if mean_val > 0:
            variance_pct = ((max_val - min_val) / mean_val) * 100
        else:
            variance_pct = 0.0 if max_val == min_val else 100.0

        is_consistent = variance_pct <= tolerance_pct

        pairwise: List[Dict[str, Any]] = []
        pack_names = sorted(values.keys())
        for i in range(len(pack_names)):
            for j in range(i + 1, len(pack_names)):
                p1, p2 = pack_names[i], pack_names[j]
                v1, v2 = values[p1], values[p2]
                diff = abs(v1 - v2)
                ref = max(abs(v1), abs(v2), 1e-10)
                pair_pct = (diff / ref) * 100

                pairwise.append({
                    "pack_a": p1,
                    "pack_b": p2,
                    "value_a": v1,
                    "value_b": v2,
                    "absolute_diff": round(diff, 6),
                    "percentage_diff": round(pair_pct, 4),
                    "within_tolerance": pair_pct <= tolerance_pct,
                })

        return {
            "point_id": point["point_id"],
            "name": point["name"],
            "data_type": point["data_type"],
            "pack_values": pack_values,
            "is_consistent": is_consistent,
            "variance_pct": round(variance_pct, 4),
            "max_deviation": round(max_val - min_val, 6),
            "mean_value": round(mean_val, 6),
            "tolerance_pct": tolerance_pct,
            "comparison_type": "numeric",
            "pairwise_results": pairwise,
        }

    def _compare_text(
        self,
        point: Dict[str, Any],
        pack_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare text/enum/boolean values across packs."""
        normalized = {}
        for pack, val in pack_values.items():
            normalized[pack] = str(val).strip().upper()

        unique_values = set(normalized.values())
        is_consistent = len(unique_values) <= 1

        pairwise: List[Dict[str, Any]] = []
        pack_names = sorted(normalized.keys())
        for i in range(len(pack_names)):
            for j in range(i + 1, len(pack_names)):
                p1, p2 = pack_names[i], pack_names[j]
                pairwise.append({
                    "pack_a": p1,
                    "pack_b": p2,
                    "value_a": pack_values[p1],
                    "value_b": pack_values[p2],
                    "match": normalized[p1] == normalized[p2],
                })

        return {
            "point_id": point["point_id"],
            "name": point["name"],
            "data_type": point["data_type"],
            "pack_values": pack_values,
            "is_consistent": is_consistent,
            "unique_value_count": len(unique_values),
            "comparison_type": "text",
            "pairwise_results": pairwise,
        }

    def _compare_date(
        self,
        point: Dict[str, Any],
        pack_values: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compare date values across packs."""
        normalized = {}
        for pack, val in pack_values.items():
            if isinstance(val, str):
                normalized[pack] = val[:10]
            elif isinstance(val, datetime):
                normalized[pack] = val.strftime("%Y-%m-%d")
            else:
                normalized[pack] = str(val)

        unique_dates = set(normalized.values())
        is_consistent = len(unique_dates) <= 1

        pairwise: List[Dict[str, Any]] = []
        pack_names = sorted(normalized.keys())
        for i in range(len(pack_names)):
            for j in range(i + 1, len(pack_names)):
                p1, p2 = pack_names[i], pack_names[j]
                pairwise.append({
                    "pack_a": p1,
                    "pack_b": p2,
                    "value_a": normalized[p1],
                    "value_b": normalized[p2],
                    "match": normalized[p1] == normalized[p2],
                })

        return {
            "point_id": point["point_id"],
            "name": point["name"],
            "data_type": point["data_type"],
            "pack_values": pack_values,
            "is_consistent": is_consistent,
            "unique_date_count": len(unique_dates),
            "comparison_type": "date",
            "pairwise_results": pairwise,
        }

    # -------------------------------------------------------------------------
    # Phase 3: Flag
    # -------------------------------------------------------------------------

    def _phase_flag(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 3: Flag discrepancies with severity ratings.

        Assigns severity to each inconsistency based on the magnitude
        of the discrepancy and the criticality of the data point.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            compare_out = self._phase_outputs.get("compare", {})
            comparisons = compare_out.get("comparisons", [])
            extract_out = self._phase_outputs.get("extract", {})
            extracted = extract_out.get("extracted_points", [])
            point_lookup = {e["point_id"]: e for e in extracted}

            discrepancies: List[Dict[str, Any]] = []

            for comp in comparisons:
                if comp.get("is_consistent", True):
                    continue

                point_id = comp["point_id"]
                point_info = point_lookup.get(point_id, {})

                severity = self._assess_severity(comp, point_info)

                disc = {
                    "discrepancy_id": str(uuid.uuid4()),
                    "point_id": point_id,
                    "name": comp["name"],
                    "data_type": comp["data_type"],
                    "severity": severity,
                    "pack_values": comp["pack_values"],
                    "comparison_type": comp.get("comparison_type", ""),
                    "pairwise_results": comp.get("pairwise_results", []),
                    "primary_source": point_info.get("primary_source", ""),
                    "auto_resolvable": point_info.get("auto_resolvable", False),
                    "tolerance_pct": point_info.get("tolerance_pct", 0.0),
                    "flagged_at": datetime.utcnow().isoformat(),
                }

                if comp.get("comparison_type") == "numeric":
                    disc["variance_pct"] = comp.get("variance_pct", 0.0)
                    disc["max_deviation"] = comp.get("max_deviation", 0.0)
                    disc["mean_value"] = comp.get("mean_value", 0.0)
                else:
                    disc["unique_value_count"] = comp.get(
                        "unique_value_count",
                        comp.get("unique_date_count", 0)
                    )

                discrepancies.append(disc)

                if severity == DiscrepancySeverity.CRITICAL.value:
                    errors.append(
                        f"Critical discrepancy: {comp['name']} across packs"
                    )
                elif severity == DiscrepancySeverity.HIGH.value:
                    warnings.append(
                        f"High discrepancy: {comp['name']}"
                    )

            discrepancies.sort(
                key=lambda d: {
                    DiscrepancySeverity.CRITICAL.value: 0,
                    DiscrepancySeverity.HIGH.value: 1,
                    DiscrepancySeverity.MEDIUM.value: 2,
                    DiscrepancySeverity.LOW.value: 3,
                    DiscrepancySeverity.INFO.value: 4,
                }.get(d["severity"], 5)
            )

            severity_counts: Dict[str, int] = {}
            for d in discrepancies:
                sev = d["severity"]
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

            outputs["discrepancies"] = discrepancies
            outputs["total_discrepancies"] = len(discrepancies)
            outputs["severity_counts"] = severity_counts
            outputs["critical_count"] = severity_counts.get(DiscrepancySeverity.CRITICAL.value, 0)
            outputs["high_count"] = severity_counts.get(DiscrepancySeverity.HIGH.value, 0)
            outputs["auto_resolvable_count"] = sum(
                1 for d in discrepancies if d["auto_resolvable"]
            )
            outputs["manual_required_count"] = sum(
                1 for d in discrepancies if not d["auto_resolvable"]
            )

            logger.info(
                "Flagging complete: %d discrepancies, %d critical, %d high",
                len(discrepancies),
                severity_counts.get(DiscrepancySeverity.CRITICAL.value, 0),
                severity_counts.get(DiscrepancySeverity.HIGH.value, 0),
            )

            status = PhaseStatus.COMPLETED
            records = len(discrepancies)

        except Exception as exc:
            logger.error("Flagging failed: %s", exc, exc_info=True)
            errors.append(f"Flagging failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="flag",
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

    def _assess_severity(
        self,
        comparison: Dict[str, Any],
        point_info: Dict[str, Any],
    ) -> str:
        """Assess severity of a discrepancy."""
        comp_type = comparison.get("comparison_type", "")
        tolerance = point_info.get("tolerance_pct", 0.0)

        if comp_type == "numeric":
            variance = comparison.get("variance_pct", 0.0)
            if variance > tolerance * 10:
                return DiscrepancySeverity.CRITICAL.value
            elif variance > tolerance * 5:
                return DiscrepancySeverity.HIGH.value
            elif variance > tolerance * 2:
                return DiscrepancySeverity.MEDIUM.value
            elif variance > tolerance:
                return DiscrepancySeverity.LOW.value
            else:
                return DiscrepancySeverity.INFO.value
        else:
            unique_count = comparison.get(
                "unique_value_count",
                comparison.get("unique_date_count", 1)
            )
            packs_involved = len(comparison.get("pack_values", {}))
            if unique_count == packs_involved:
                return DiscrepancySeverity.CRITICAL.value
            elif unique_count > 2:
                return DiscrepancySeverity.HIGH.value
            else:
                return DiscrepancySeverity.MEDIUM.value

    # -------------------------------------------------------------------------
    # Phase 4: Resolve
    # -------------------------------------------------------------------------

    def _phase_resolve(self, config: WorkflowConfig) -> PhaseResult:
        """
        Phase 4: Auto-resolve or create manual resolution tasks.

        Applies resolution strategies for auto-resolvable discrepancies
        and creates task entries for those requiring manual review.
        """
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        try:
            flag_out = self._phase_outputs.get("flag", {})
            discrepancies = flag_out.get("discrepancies", [])

            resolved: List[Dict[str, Any]] = []
            manual_tasks: List[Dict[str, Any]] = []

            for disc in discrepancies:
                if disc["auto_resolvable"] and config.auto_resolve_enabled:
                    resolution = self._auto_resolve(disc)
                    resolved.append(resolution)
                else:
                    task = self._create_manual_task(disc, config)
                    manual_tasks.append(task)

            outputs["resolved"] = resolved
            outputs["auto_resolved_count"] = len(resolved)
            outputs["manual_tasks"] = manual_tasks
            outputs["manual_task_count"] = len(manual_tasks)

            resolution_methods: Dict[str, int] = {}
            for r in resolved:
                method = r["resolution_method"]
                resolution_methods[method] = resolution_methods.get(method, 0) + 1
            outputs["resolution_methods"] = resolution_methods

            outputs["reconciliation_report"] = {
                "workflow_id": self.workflow_id,
                "organization_id": config.organization_id,
                "reporting_year": config.reporting_year,
                "total_discrepancies": len(discrepancies),
                "auto_resolved": len(resolved),
                "manual_review_needed": len(manual_tasks),
                "completed_at": datetime.utcnow().isoformat(),
            }

            if manual_tasks:
                warnings.append(
                    f"{len(manual_tasks)} discrepancies require manual review"
                )

            logger.info(
                "Resolution complete: %d auto-resolved, %d manual tasks",
                len(resolved), len(manual_tasks),
            )

            status = PhaseStatus.COMPLETED
            records = len(resolved) + len(manual_tasks)

        except Exception as exc:
            logger.error("Resolution failed: %s", exc, exc_info=True)
            errors.append(f"Resolution failed: {str(exc)}")
            status = PhaseStatus.FAILED
            records = 0

        completed_at = datetime.utcnow()
        return PhaseResult(
            phase_name="resolve",
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

    def _auto_resolve(self, disc: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-resolve a discrepancy using the appropriate strategy."""
        comp_type = disc.get("comparison_type", "")
        primary_source = disc.get("primary_source", "")
        pack_values = disc.get("pack_values", {})

        if comp_type == "numeric" and disc.get("variance_pct", 0) <= disc.get("tolerance_pct", 0) * 2:
            method = ResolutionMethod.AUTO_TOLERANCE.value
            resolved_value = disc.get("mean_value", 0.0)
            explanation = (
                f"Variance {disc.get('variance_pct', 0):.4f}% within 2x tolerance "
                f"({disc.get('tolerance_pct', 0)}%). Using mean value."
            )
        elif primary_source and primary_source in pack_values:
            method = ResolutionMethod.AUTO_PRIMARY_SOURCE.value
            resolved_value = pack_values[primary_source]
            explanation = f"Using primary source value from {primary_source}."
        elif comp_type == "numeric":
            values = [v for v in pack_values.values() if isinstance(v, (int, float))]
            if values:
                resolved_value = round(sum(values) / len(values), 6)
                method = ResolutionMethod.AUTO_AVERAGE.value
                explanation = f"Using average of {len(values)} pack values."
            else:
                resolved_value = list(pack_values.values())[0] if pack_values else None
                method = ResolutionMethod.AUTO_LATEST.value
                explanation = "Using first available value."
        else:
            if primary_source and primary_source in pack_values:
                resolved_value = pack_values[primary_source]
                method = ResolutionMethod.AUTO_PRIMARY_SOURCE.value
                explanation = f"Text value: using primary source {primary_source}."
            else:
                resolved_value = list(pack_values.values())[0] if pack_values else None
                method = ResolutionMethod.AUTO_LATEST.value
                explanation = "Using first available value as fallback."

        return {
            "resolution_id": str(uuid.uuid4()),
            "discrepancy_id": disc["discrepancy_id"],
            "point_id": disc["point_id"],
            "name": disc["name"],
            "resolution_method": method,
            "resolved_value": resolved_value,
            "original_values": pack_values,
            "explanation": explanation,
            "applied_to_packs": list(pack_values.keys()),
            "resolved_at": datetime.utcnow().isoformat(),
        }

    def _create_manual_task(
        self,
        disc: Dict[str, Any],
        config: WorkflowConfig,
    ) -> Dict[str, Any]:
        """Create a manual resolution task for a discrepancy."""
        primary_pack = disc.get("primary_source", "")
        assignee = config.resolution_assignees.get(
            primary_pack,
            config.resolution_assignees.get("default", "unassigned"),
        )

        severity_priority = {
            DiscrepancySeverity.CRITICAL.value: "P1",
            DiscrepancySeverity.HIGH.value: "P2",
            DiscrepancySeverity.MEDIUM.value: "P3",
            DiscrepancySeverity.LOW.value: "P4",
            DiscrepancySeverity.INFO.value: "P5",
        }

        return {
            "task_id": str(uuid.uuid4()),
            "discrepancy_id": disc["discrepancy_id"],
            "point_id": disc["point_id"],
            "name": disc["name"],
            "severity": disc["severity"],
            "priority": severity_priority.get(disc["severity"], "P3"),
            "assignee": assignee,
            "pack_values": disc["pack_values"],
            "primary_source": primary_pack,
            "description": (
                f"Manual review required for '{disc['name']}': "
                f"values differ across {len(disc['pack_values'])} packs"
            ),
            "resolution_method": ResolutionMethod.MANUAL_REVIEW.value,
            "status": "OPEN",
            "created_at": datetime.utcnow().isoformat(),
            "due_date": None,
        }

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    def _build_summary(self) -> Dict[str, Any]:
        """Build workflow summary from all phase outputs."""
        extract = self._phase_outputs.get("extract", {})
        compare = self._phase_outputs.get("compare", {})
        flag = self._phase_outputs.get("flag", {})
        resolve = self._phase_outputs.get("resolve", {})

        return {
            "data_points_extracted": extract.get("total_extracted", 0),
            "data_points_compared": compare.get("total_compared", 0),
            "consistent_points": compare.get("consistent_count", 0),
            "inconsistent_points": compare.get("inconsistent_count", 0),
            "discrepancies_found": flag.get("total_discrepancies", 0),
            "critical_discrepancies": flag.get("critical_count", 0),
            "auto_resolved": resolve.get("auto_resolved_count", 0),
            "manual_tasks_created": resolve.get("manual_task_count", 0),
        }


# =============================================================================
# UTILITIES
# =============================================================================


def _hash_data(data: Any) -> str:
    """Compute SHA-256 provenance hash of arbitrary data."""
    serialized = str(data).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
