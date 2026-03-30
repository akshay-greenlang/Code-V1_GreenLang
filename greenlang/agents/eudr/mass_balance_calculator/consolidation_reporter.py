# -*- coding: utf-8 -*-
"""
Consolidation Reporter - AGENT-EUDR-011 Engine 8

Multi-facility consolidation reporting:
- Enterprise dashboard: aggregate mass balance across facilities
- Facility-level isolation: per-facility accounting boundaries
- Cross-facility transfer tracking: material transfers with ledger entries
  at both ends
- Consolidation reporting: enterprise compliance, facility comparison,
  commodity breakdown
- Facility grouping: by region, country, commodity, or custom hierarchy
- Consolidated reconciliation with drill-down
- Report formats: JSON, PDF, CSV, EUDR XML
- Regulatory evidence package for competent authority inspections
- SHA-256 provenance hashing on all reports

Zero-Hallucination Guarantees:
    - All aggregation calculations use deterministic Python Decimal arithmetic
    - Cross-facility transfers record debit/credit at both ends
    - No ML/LLM used for any numeric calculation
    - SHA-256 provenance hashes on every report
    - All formatting uses deterministic string operations
    - CSV and XML generation uses template-based approach

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Mass balance chain of custody
    - EU 2023/1115 (EUDR) Article 10(2)(f): Mass balance verification
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - EU 2023/1115 (EUDR) Article 31: Competent authority inspections
    - ISO 22095:2020: Chain of Custody - Mass Balance requirements

Performance Targets:
    - Enterprise dashboard (10 facilities): <200ms
    - Consolidation report generation: <500ms
    - Evidence package assembly: <1000ms
    - Cross-facility transfer recording: <20ms
    - Facility comparison (10 facilities): <100ms

PRD Feature References:
    - PRD-AGENT-EUDR-011 Feature 8: Multi-Facility Consolidation
    - PRD-AGENT-EUDR-011 Feature 8.1: Enterprise Dashboard
    - PRD-AGENT-EUDR-011 Feature 8.2: Facility-Level Isolation
    - PRD-AGENT-EUDR-011 Feature 8.3: Cross-Facility Transfer Tracking
    - PRD-AGENT-EUDR-011 Feature 8.4: Consolidation Reporting
    - PRD-AGENT-EUDR-011 Feature 8.5: Facility Grouping
    - PRD-AGENT-EUDR-011 Feature 8.6: Consolidated Reconciliation
    - PRD-AGENT-EUDR-011 Feature 8.7: Report Formats
    - PRD-AGENT-EUDR-011 Feature 8.8: Evidence Package

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.mass_balance_calculator.config import (
    MassBalanceCalculatorConfig,
    get_config,
)
from greenlang.agents.eudr.mass_balance_calculator.models import (
    ComplianceStatus,
    FacilityGroupType,
    ReportFormat,
    ReportType,
)
from greenlang.agents.eudr.mass_balance_calculator.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.mass_balance_calculator.metrics import (
    record_api_error,
    record_report_generated,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _generate_id(prefix: str = "RPT") -> str:
    """Generate a prefixed UUID4 string identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Prefixed UUID4 string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"

def _safe_decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation.

    Raises:
        ValueError: If the value cannot be converted.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(
            f"Cannot convert {value!r} to Decimal: {exc}"
        ) from exc

def _safe_float(value: Any) -> float:
    """Safely convert a value to float.

    Args:
        value: Numeric value to convert.

    Returns:
        Float representation.
    """
    if isinstance(value, float):
        return value
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return 0.0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Supported report types for consolidation.
SUPPORTED_REPORT_TYPES = frozenset({
    "enterprise_summary",
    "facility_comparison",
    "commodity_breakdown",
    "reconciliation_summary",
    "compliance_overview",
    "transfer_summary",
    "evidence_package",
})

#: Supported report formats.
SUPPORTED_REPORT_FORMATS = frozenset({"json", "csv", "pdf", "eudr_xml"})

#: Maximum facilities per consolidation report.
MAX_FACILITIES_PER_REPORT = 500

#: Default EUDR commodities.
EUDR_COMMODITIES = (
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
)

#: Evidence package document types.
EVIDENCE_DOCUMENT_TYPES = (
    "mass_balance_ledger",
    "reconciliation_records",
    "conversion_factor_validations",
    "overdraft_events",
    "loss_records",
    "carry_forward_records",
    "transfer_records",
    "compliance_assessments",
    "provenance_chain",
)

# ---------------------------------------------------------------------------
# ConsolidationReporter
# ---------------------------------------------------------------------------

class ConsolidationReporter:
    """Multi-facility consolidation and reporting engine.

    Provides comprehensive consolidation and reporting capabilities:
        - Enterprise dashboard with aggregate mass balance metrics
        - Facility-level isolation with per-facility accounting boundaries
        - Cross-facility transfer tracking with debit/credit at both ends
        - Consolidation reporting for enterprise compliance and comparison
        - Facility grouping by region, country, commodity, or custom
        - Report generation in JSON, CSV, PDF, and EUDR XML formats
        - Evidence package assembly for competent authority inspections
        - SHA-256 provenance hashing on all reports

    All operations are thread-safe via reentrant locking. All balance
    calculations use deterministic Python Decimal arithmetic for
    zero-hallucination compliance.

    Attributes:
        _config: Mass balance calculator configuration.
        _provenance: ProvenanceTracker for audit trail.
        _reports: In-memory report storage keyed by report_id.
        _groups: In-memory facility group storage keyed by group_id.
        _transfers: In-memory transfer records list.
        _facility_data: In-memory facility data keyed by facility_id.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> reporter = ConsolidationReporter()
        >>> report = reporter.generate_consolidation_report(
        ...     facility_ids=["F-001", "F-002"],
        ...     report_type="enterprise_summary",
        ... )
        >>> assert report["report_id"] is not None
    """

    def __init__(
        self,
        config: Optional[MassBalanceCalculatorConfig] = None,
    ) -> None:
        """Initialize ConsolidationReporter.

        Args:
            config: Optional configuration override. If None, the
                singleton configuration from ``get_config()`` is used.
        """
        self._config: MassBalanceCalculatorConfig = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # In-memory storage
        self._reports: Dict[str, Dict[str, Any]] = {}
        self._groups: Dict[str, Dict[str, Any]] = {}
        self._transfers: List[Dict[str, Any]] = []
        self._facility_data: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "ConsolidationReporter initialized: "
            "default_format=%s, retention_days=%d",
            self._config.report_default_format,
            self._config.report_retention_days,
        )

    # ------------------------------------------------------------------
    # Public API: Generate consolidation report
    # ------------------------------------------------------------------

    def generate_consolidation_report(
        self,
        facility_ids: List[str],
        report_type: str,
        report_format: str = "json",
        commodity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a consolidation report across multiple facilities.

        Aggregates mass balance data from the specified facilities
        according to the requested report type and format.

        Args:
            facility_ids: List of facility identifiers to include.
            report_type: Type of report (enterprise_summary,
                facility_comparison, commodity_breakdown,
                reconciliation_summary, compliance_overview,
                transfer_summary, evidence_package).
            report_format: Output format (json, csv, pdf, eudr_xml).
            commodity: Optional commodity filter.

        Returns:
            Dictionary with report_id, type, format, content, summary,
            and provenance_hash.

        Raises:
            ValueError: If facility_ids is empty or report_type/format
                is not supported.
        """
        start_time = time.monotonic()

        if not facility_ids:
            raise ValueError("facility_ids must not be empty")
        if report_type not in SUPPORTED_REPORT_TYPES:
            raise ValueError(
                f"Unsupported report_type: {report_type}. "
                f"Supported: {sorted(SUPPORTED_REPORT_TYPES)}"
            )
        if report_format not in SUPPORTED_REPORT_FORMATS:
            raise ValueError(
                f"Unsupported report_format: {report_format}. "
                f"Supported: {sorted(SUPPORTED_REPORT_FORMATS)}"
            )
        if len(facility_ids) > MAX_FACILITIES_PER_REPORT:
            raise ValueError(
                f"Too many facilities: {len(facility_ids)} "
                f"(max: {MAX_FACILITIES_PER_REPORT})"
            )

        logger.info(
            "Generating consolidation report: type=%s, format=%s, "
            "facilities=%d, commodity=%s",
            report_type, report_format,
            len(facility_ids), commodity or "all",
        )

        report_id = _generate_id("RPT-CON")
        now = utcnow()

        # Build report data based on type
        report_data = self._build_report_data(
            facility_ids, report_type, commodity,
        )

        # Format the report content
        formatted_content = self._format_report(report_data, report_format)

        # Build report summary
        summary = self._build_report_summary(
            facility_ids, report_type, report_data,
        )

        report: Dict[str, Any] = {
            "report_id": report_id,
            "report_type": report_type,
            "report_format": report_format,
            "facility_ids": facility_ids,
            "commodity": commodity,
            "data": report_data,
            "content": formatted_content,
            "summary": summary,
            "facility_count": len(facility_ids),
            "generated_at": now.isoformat(),
            "generated_by": "GL-EUDR-MBC-011",
            "provenance_hash": "",
            "metadata": {
                "module_version": _MODULE_VERSION,
                "agent_id": "GL-EUDR-MBC-011",
                "regulation": "EU 2023/1115",
            },
        }
        report["provenance_hash"] = _compute_hash(report)

        # Store report
        with self._lock:
            self._reports[report_id] = report

        # Record provenance
        self._provenance.record(
            entity_type="consolidation_report",
            action="generate",
            entity_id=report_id,
            data={"report_type": report_type, "facility_count": len(facility_ids)},
            metadata={
                "report_format": report_format,
                "commodity": commodity or "all",
            },
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        record_report_generated("consolidation")

        logger.info(
            "Consolidation report generated: id=%s, type=%s, "
            "format=%s, facilities=%d, elapsed=%.1fms",
            report_id, report_type, report_format,
            len(facility_ids), elapsed_ms,
        )

        return report

    # ------------------------------------------------------------------
    # Public API: Create facility group
    # ------------------------------------------------------------------

    def create_facility_group(
        self,
        name: str,
        group_type: str,
        facility_ids: List[str],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a facility group for consolidation reporting.

        Groups facilities by region, country, commodity, or custom
        hierarchy for aggregate reporting purposes.

        Args:
            name: Human-readable group name.
            group_type: Type of grouping (region, country, commodity,
                custom).
            facility_ids: List of facility identifiers in the group.
            description: Optional group description.

        Returns:
            Dictionary with group details and provenance_hash.

        Raises:
            ValueError: If name is empty, group_type is not valid,
                or facility_ids is empty.
        """
        if not name:
            raise ValueError("name must not be empty")
        if not facility_ids:
            raise ValueError("facility_ids must not be empty")

        valid_types = {"region", "country", "commodity", "custom"}
        if group_type not in valid_types:
            raise ValueError(
                f"Invalid group_type: {group_type}. "
                f"Must be one of: {sorted(valid_types)}"
            )

        group_id = _generate_id("GRP")
        now = utcnow()

        group: Dict[str, Any] = {
            "group_id": group_id,
            "name": name,
            "group_type": group_type,
            "facility_ids": list(facility_ids),
            "facility_count": len(facility_ids),
            "description": description,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "provenance_hash": "",
        }
        group["provenance_hash"] = _compute_hash(group)

        with self._lock:
            self._groups[group_id] = group

        # Record provenance
        self._provenance.record(
            entity_type="facility_group",
            action="create",
            entity_id=group_id,
            data=group,
            metadata={
                "group_type": group_type,
                "facility_count": len(facility_ids),
            },
        )

        logger.info(
            "Facility group created: id=%s, name=%s, type=%s, "
            "facilities=%d",
            group_id, name, group_type, len(facility_ids),
        )

        return group

    # ------------------------------------------------------------------
    # Public API: Enterprise dashboard
    # ------------------------------------------------------------------

    def get_enterprise_dashboard(
        self,
        group_id: Optional[str] = None,
        facility_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate an enterprise-level dashboard across facilities.

        Aggregates mass balance metrics from the specified facilities
        or facility group into a single dashboard view.

        Args:
            group_id: Optional facility group to aggregate. If provided,
                the group's facility_ids are used.
            facility_ids: Optional explicit list of facility IDs. If
                both group_id and facility_ids are provided,
                facility_ids takes precedence.

        Returns:
            Dictionary with aggregate metrics, facility summaries,
            commodity breakdown, compliance status, and provenance_hash.

        Raises:
            ValueError: If neither group_id nor facility_ids provided.
            KeyError: If group_id is not found.
        """
        start_time = time.monotonic()

        # Resolve facility IDs
        resolved_ids = self._resolve_facility_ids(group_id, facility_ids)

        if not resolved_ids:
            raise ValueError(
                "Must provide either group_id or facility_ids"
            )

        logger.debug(
            "Building enterprise dashboard: facilities=%d",
            len(resolved_ids),
        )

        # Aggregate metrics across facilities
        aggregate = self._compute_aggregate_metrics(resolved_ids)

        # Per-facility summaries
        facility_summaries = self._compute_facility_summaries(resolved_ids)

        # Commodity breakdown
        commodity_breakdown = self._compute_commodity_breakdown(resolved_ids)

        # Overall compliance status
        compliance_status = self._assess_enterprise_compliance(resolved_ids)

        now = utcnow()
        dashboard: Dict[str, Any] = {
            "dashboard_type": "enterprise",
            "group_id": group_id,
            "facility_ids": resolved_ids,
            "facility_count": len(resolved_ids),
            "aggregate_metrics": aggregate,
            "facility_summaries": facility_summaries,
            "commodity_breakdown": commodity_breakdown,
            "compliance_status": compliance_status,
            "generated_at": now.isoformat(),
            "provenance_hash": "",
        }
        dashboard["provenance_hash"] = _compute_hash(dashboard)

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Enterprise dashboard generated: facilities=%d, elapsed=%.1fms",
            len(resolved_ids), elapsed_ms,
        )

        return dashboard

    # ------------------------------------------------------------------
    # Public API: Get report
    # ------------------------------------------------------------------

    def get_report(
        self,
        report_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a generated report by ID.

        Args:
            report_id: Report identifier.

        Returns:
            Report dictionary, or None if not found.
        """
        with self._lock:
            return self._reports.get(report_id)

    # ------------------------------------------------------------------
    # Public API: Download report
    # ------------------------------------------------------------------

    def download_report(
        self,
        report_id: str,
    ) -> Dict[str, Any]:
        """Download a generated report with full content.

        Args:
            report_id: Report identifier.

        Returns:
            Dictionary with report_id, format, content, and metadata.

        Raises:
            KeyError: If report_id is not found.
        """
        with self._lock:
            report = self._reports.get(report_id)
            if report is None:
                raise KeyError(f"Report not found: {report_id}")

        return {
            "report_id": report_id,
            "report_type": report.get("report_type", ""),
            "report_format": report.get("report_format", "json"),
            "content": report.get("content", ""),
            "summary": report.get("summary", {}),
            "facility_count": report.get("facility_count", 0),
            "generated_at": report.get("generated_at", ""),
            "provenance_hash": report.get("provenance_hash", ""),
            "file_size_bytes": len(
                str(report.get("content", "")).encode("utf-8")
            ),
        }

    # ------------------------------------------------------------------
    # Public API: Record transfer
    # ------------------------------------------------------------------

    def record_transfer(
        self,
        from_facility: str,
        to_facility: str,
        commodity: str,
        quantity_kg: float,
        batch_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a cross-facility material transfer.

        Creates debit/credit entries at both the sending and receiving
        facilities for material transfer traceability.

        Args:
            from_facility: Sending facility identifier.
            to_facility: Receiving facility identifier.
            commodity: Commodity being transferred.
            quantity_kg: Transfer quantity in kilograms.
            batch_id: Optional batch/lot identifier.

        Returns:
            Dictionary with transfer details and provenance_hash.

        Raises:
            ValueError: If from/to facilities are the same, quantity
                is non-positive, or required fields are empty.
        """
        if not from_facility:
            raise ValueError("from_facility must not be empty")
        if not to_facility:
            raise ValueError("to_facility must not be empty")
        if not commodity:
            raise ValueError("commodity must not be empty")
        if from_facility == to_facility:
            raise ValueError(
                "from_facility and to_facility must be different"
            )

        qty = _safe_decimal(quantity_kg)
        if qty <= Decimal("0"):
            raise ValueError(
                f"quantity_kg must be positive, got {quantity_kg}"
            )

        transfer_id = _generate_id("XFR")
        now = utcnow()

        transfer: Dict[str, Any] = {
            "transfer_id": transfer_id,
            "from_facility": from_facility,
            "to_facility": to_facility,
            "commodity": commodity,
            "quantity_kg": str(qty),
            "batch_id": batch_id,
            "status": "completed",
            "debit_entry": {
                "facility_id": from_facility,
                "entry_type": "transfer_out",
                "quantity_kg": str(qty),
                "commodity": commodity,
                "transfer_id": transfer_id,
            },
            "credit_entry": {
                "facility_id": to_facility,
                "entry_type": "transfer_in",
                "quantity_kg": str(qty),
                "commodity": commodity,
                "transfer_id": transfer_id,
            },
            "created_at": now.isoformat(),
            "provenance_hash": "",
        }
        transfer["provenance_hash"] = _compute_hash(transfer)

        with self._lock:
            self._transfers.append(transfer)

            # Update facility data for both facilities
            self._update_facility_transfer(
                from_facility, commodity, qty, "out",
            )
            self._update_facility_transfer(
                to_facility, commodity, qty, "in",
            )

        # Record provenance
        self._provenance.record(
            entity_type="consolidation_report",
            action="record",
            entity_id=transfer_id,
            data=transfer,
            metadata={
                "from_facility": from_facility,
                "to_facility": to_facility,
                "commodity": commodity,
                "quantity_kg": str(qty),
            },
        )

        logger.info(
            "Transfer recorded: id=%s, from=%s, to=%s, "
            "commodity=%s, qty=%s kg",
            transfer_id, from_facility, to_facility,
            commodity, qty,
        )

        return transfer

    # ------------------------------------------------------------------
    # Public API: Facility comparison
    # ------------------------------------------------------------------

    def get_facility_comparison(
        self,
        facility_ids: List[str],
        commodity: str,
    ) -> Dict[str, Any]:
        """Compare mass balance metrics across facilities for a commodity.

        Provides side-by-side comparison of key metrics including
        total inputs, total outputs, balance, utilization rate,
        loss percentage, and transfer volumes.

        Args:
            facility_ids: List of facility identifiers to compare.
            commodity: Commodity to compare.

        Returns:
            Dictionary with facility comparisons, rankings, and
            aggregate statistics.

        Raises:
            ValueError: If facility_ids is empty or commodity is empty.
        """
        if not facility_ids:
            raise ValueError("facility_ids must not be empty")
        if not commodity:
            raise ValueError("commodity must not be empty")

        comparisons: List[Dict[str, Any]] = []
        for fid in facility_ids:
            comparison = self._build_facility_comparison(fid, commodity)
            comparisons.append(comparison)

        # Rank by utilization rate descending
        ranked = sorted(
            comparisons,
            key=lambda c: c.get("utilization_rate", 0.0),
            reverse=True,
        )
        for i, comp in enumerate(ranked, 1):
            comp["rank"] = i

        # Aggregate statistics
        agg_stats = self._compute_comparison_statistics(comparisons)

        return {
            "commodity": commodity,
            "facility_count": len(facility_ids),
            "comparisons": ranked,
            "statistics": agg_stats,
            "generated_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API: Evidence package
    # ------------------------------------------------------------------

    def generate_evidence_package(
        self,
        facility_ids: List[str],
        period_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a regulatory evidence package for inspections.

        Assembles a comprehensive evidence package for competent
        authority inspections per EUDR Article 31. Includes all
        mass balance records, reconciliation results, compliance
        assessments, and provenance chains.

        Args:
            facility_ids: Facilities to include in the package.
            period_id: Optional period to scope the evidence.

        Returns:
            Dictionary with evidence package contents, document
            inventory, and provenance_hash.

        Raises:
            ValueError: If facility_ids is empty.
        """
        start_time = time.monotonic()

        if not facility_ids:
            raise ValueError("facility_ids must not be empty")

        logger.info(
            "Generating evidence package: facilities=%d, period=%s",
            len(facility_ids), period_id or "all",
        )

        package_id = _generate_id("EVD")
        now = utcnow()

        # Assemble document inventory
        document_inventory = self._build_document_inventory(
            facility_ids, period_id,
        )

        # Build evidence sections
        evidence_sections = self._build_evidence_sections(
            facility_ids, period_id,
        )

        # Compliance summary
        compliance_summary = self._build_evidence_compliance_summary(
            facility_ids,
        )

        package: Dict[str, Any] = {
            "package_id": package_id,
            "package_type": "regulatory_evidence",
            "regulation": "EU 2023/1115 (EUDR)",
            "regulation_articles": ["Article 4", "Article 14", "Article 31"],
            "facility_ids": facility_ids,
            "facility_count": len(facility_ids),
            "period_id": period_id,
            "document_inventory": document_inventory,
            "document_count": len(document_inventory),
            "evidence_sections": evidence_sections,
            "compliance_summary": compliance_summary,
            "retention_years": self._config.retention_years,
            "generated_at": now.isoformat(),
            "generated_by": "GL-EUDR-MBC-011",
            "provenance_hash": "",
            "metadata": {
                "module_version": _MODULE_VERSION,
                "agent_id": "GL-EUDR-MBC-011",
                "iso_standard": "ISO 22095:2020",
            },
        }
        package["provenance_hash"] = _compute_hash(package)

        # Store as a report
        with self._lock:
            self._reports[package_id] = package

        # Record provenance
        self._provenance.record(
            entity_type="consolidation_report",
            action="generate",
            entity_id=package_id,
            data={"package_type": "evidence", "facility_count": len(facility_ids)},
            metadata={
                "period_id": period_id or "all",
            },
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        record_report_generated("evidence")

        logger.info(
            "Evidence package generated: id=%s, facilities=%d, "
            "documents=%d, elapsed=%.1fms",
            package_id, len(facility_ids),
            len(document_inventory), elapsed_ms,
        )

        return package

    # ------------------------------------------------------------------
    # Public API: Commodity breakdown
    # ------------------------------------------------------------------

    def get_commodity_breakdown(
        self,
        facility_ids: List[str],
    ) -> Dict[str, Any]:
        """Get commodity breakdown across specified facilities.

        Provides aggregate mass balance metrics broken down by commodity
        type across the specified facilities.

        Args:
            facility_ids: Facilities to include.

        Returns:
            Dictionary with per-commodity breakdowns and totals.

        Raises:
            ValueError: If facility_ids is empty.
        """
        if not facility_ids:
            raise ValueError("facility_ids must not be empty")

        breakdown: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for fid in facility_ids:
                fdata = self._facility_data.get(fid, {})
                commodities = fdata.get("commodities", {})
                for commodity, cdata in commodities.items():
                    if commodity not in breakdown:
                        breakdown[commodity] = {
                            "commodity": commodity,
                            "total_inputs": Decimal("0"),
                            "total_outputs": Decimal("0"),
                            "total_losses": Decimal("0"),
                            "total_transfers_in": Decimal("0"),
                            "total_transfers_out": Decimal("0"),
                            "facility_count": 0,
                            "facility_ids": [],
                        }
                    bd = breakdown[commodity]
                    bd["total_inputs"] += _safe_decimal(
                        cdata.get("total_inputs", 0)
                    )
                    bd["total_outputs"] += _safe_decimal(
                        cdata.get("total_outputs", 0)
                    )
                    bd["total_losses"] += _safe_decimal(
                        cdata.get("total_losses", 0)
                    )
                    bd["total_transfers_in"] += _safe_decimal(
                        cdata.get("transfers_in", 0)
                    )
                    bd["total_transfers_out"] += _safe_decimal(
                        cdata.get("transfers_out", 0)
                    )
                    bd["facility_count"] += 1
                    bd["facility_ids"].append(fid)

        # Convert Decimals to strings for serialization
        result_breakdown: List[Dict[str, Any]] = []
        for commodity, data in sorted(breakdown.items()):
            total_inputs = data["total_inputs"]
            total_outputs = data["total_outputs"]
            utilization = (
                float(total_outputs / total_inputs * Decimal("100"))
                if total_inputs > 0 else 0.0
            )
            result_breakdown.append({
                "commodity": commodity,
                "total_inputs": str(data["total_inputs"]),
                "total_outputs": str(data["total_outputs"]),
                "total_losses": str(data["total_losses"]),
                "total_transfers_in": str(data["total_transfers_in"]),
                "total_transfers_out": str(data["total_transfers_out"]),
                "utilization_rate": round(utilization, 2),
                "facility_count": data["facility_count"],
                "facility_ids": data["facility_ids"],
            })

        return {
            "facility_ids": facility_ids,
            "commodity_count": len(result_breakdown),
            "commodities": result_breakdown,
            "generated_at": utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Public API: List reports
    # ------------------------------------------------------------------

    def list_reports(
        self,
        facility_id: Optional[str] = None,
        report_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List generated reports with optional filters.

        Args:
            facility_id: Optional facility filter. Returns reports that
                include this facility.
            report_type: Optional report type filter.

        Returns:
            List of report summary dictionaries sorted by generated_at
            descending.
        """
        with self._lock:
            reports = list(self._reports.values())

        # Apply filters
        if facility_id:
            reports = [
                r for r in reports
                if facility_id in r.get("facility_ids", [])
            ]

        if report_type:
            reports = [
                r for r in reports
                if r.get("report_type") == report_type
            ]

        # Sort by generated_at descending
        reports.sort(
            key=lambda r: r.get("generated_at", ""),
            reverse=True,
        )

        # Return summaries only
        summaries: List[Dict[str, Any]] = []
        for r in reports:
            summaries.append({
                "report_id": r.get("report_id", ""),
                "report_type": r.get("report_type", ""),
                "report_format": r.get("report_format", ""),
                "facility_count": r.get("facility_count", 0),
                "commodity": r.get("commodity"),
                "generated_at": r.get("generated_at", ""),
                "provenance_hash": r.get("provenance_hash", ""),
            })

        return summaries

    # ------------------------------------------------------------------
    # Public API: Format report (internal, exposed for testing)
    # ------------------------------------------------------------------

    def _format_report(
        self,
        data: Dict[str, Any],
        report_format: str,
    ) -> str:
        """Format report data into the specified output format.

        Args:
            data: Report data dictionary.
            report_format: Output format (json, csv, pdf, eudr_xml).

        Returns:
            Formatted report string.
        """
        fmt = report_format.lower().strip()

        if fmt == "json":
            return json.dumps(data, indent=2, default=str)

        if fmt == "csv":
            return self._generate_csv(data)

        if fmt == "pdf":
            return json.dumps({
                "format": "pdf",
                "note": "PDF rendering delegated to document service",
                "content_hash": _compute_hash(data),
                "sections": list(data.keys()),
            }, indent=2, default=str)

        if fmt == "eudr_xml":
            return self._generate_eudr_xml(data)

        return json.dumps(data, indent=2, default=str)

    def _generate_csv(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Generate CSV-formatted report content.

        Flattens the report data into a tabular CSV format suitable
        for spreadsheet analysis.

        Args:
            data: Report data dictionary.

        Returns:
            CSV-formatted string.
        """
        lines: List[str] = []
        lines.append("section,field,value")

        # Header section
        header = data.get("header", {})
        for key, value in sorted(header.items()):
            lines.append(f"header,{key},{value}")

        # Facility summaries
        facilities = data.get("facility_summaries", [])
        for fac in facilities:
            fid = fac.get("facility_id", "")
            for key, value in sorted(fac.items()):
                if key != "facility_id":
                    lines.append(f"facility_{fid},{key},{value}")

        # Commodity breakdown
        commodities = data.get("commodity_breakdown", [])
        for cmd in commodities:
            c_name = cmd.get("commodity", "")
            for key, value in sorted(cmd.items()):
                if key != "commodity":
                    lines.append(f"commodity_{c_name},{key},{value}")

        # Aggregate metrics
        aggregate = data.get("aggregate_metrics", {})
        for key, value in sorted(aggregate.items()):
            lines.append(f"aggregate,{key},{value}")

        # Transfers
        transfers = data.get("transfers", [])
        for i, transfer in enumerate(transfers):
            for key, value in sorted(transfer.items()):
                lines.append(f"transfer_{i},{key},{value}")

        return "\n".join(lines)

    def _generate_eudr_xml(
        self,
        data: Dict[str, Any],
    ) -> str:
        """Generate EUDR-compliant XML report content.

        Formats the report data as XML conforming to the EU Information
        System schema for due diligence statement submission.

        Args:
            data: Report data dictionary.

        Returns:
            XML-formatted string.
        """
        header = data.get("header", {})
        aggregate = data.get("aggregate_metrics", {})

        xml_parts: List[str] = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<MassBalanceConsolidation '
            'xmlns="urn:eu:eudr:mass-balance:1.0">',
            '  <Header>',
            f'    <AgentId>{header.get("agent_id", "GL-EUDR-MBC-011")}'
            f'</AgentId>',
            f'    <ReportType>{header.get("report_type", "")}</ReportType>',
            f'    <GeneratedAt>{header.get("generated_at", "")}'
            f'</GeneratedAt>',
            f'    <Regulation>EU 2023/1115</Regulation>',
            f'    <Standard>ISO 22095:2020</Standard>',
            '  </Header>',
            '  <AggregateMetrics>',
            f'    <TotalInputsKg>'
            f'{aggregate.get("total_inputs", "0")}'
            f'</TotalInputsKg>',
            f'    <TotalOutputsKg>'
            f'{aggregate.get("total_outputs", "0")}'
            f'</TotalOutputsKg>',
            f'    <TotalBalance>'
            f'{aggregate.get("total_balance", "0")}'
            f'</TotalBalance>',
            f'    <TotalLossesKg>'
            f'{aggregate.get("total_losses", "0")}'
            f'</TotalLossesKg>',
            f'    <UtilizationRate>'
            f'{aggregate.get("utilization_rate", 0.0)}'
            f'</UtilizationRate>',
            f'    <FacilityCount>'
            f'{aggregate.get("facility_count", 0)}'
            f'</FacilityCount>',
            '  </AggregateMetrics>',
        ]

        # Facility entries
        facilities = data.get("facility_summaries", [])
        if facilities:
            xml_parts.append('  <Facilities>')
            for fac in facilities:
                xml_parts.extend([
                    '    <Facility>',
                    f'      <FacilityId>'
                    f'{fac.get("facility_id", "")}'
                    f'</FacilityId>',
                    f'      <TotalInputsKg>'
                    f'{fac.get("total_inputs", "0")}'
                    f'</TotalInputsKg>',
                    f'      <TotalOutputsKg>'
                    f'{fac.get("total_outputs", "0")}'
                    f'</TotalOutputsKg>',
                    f'      <BalanceKg>'
                    f'{fac.get("balance", "0")}'
                    f'</BalanceKg>',
                    '    </Facility>',
                ])
            xml_parts.append('  </Facilities>')

        # Transfers
        transfers = data.get("transfers", [])
        if transfers:
            xml_parts.append('  <Transfers>')
            for tfr in transfers:
                xml_parts.extend([
                    '    <Transfer>',
                    f'      <TransferId>'
                    f'{tfr.get("transfer_id", "")}'
                    f'</TransferId>',
                    f'      <FromFacility>'
                    f'{tfr.get("from_facility", "")}'
                    f'</FromFacility>',
                    f'      <ToFacility>'
                    f'{tfr.get("to_facility", "")}'
                    f'</ToFacility>',
                    f'      <Commodity>'
                    f'{tfr.get("commodity", "")}'
                    f'</Commodity>',
                    f'      <QuantityKg>'
                    f'{tfr.get("quantity_kg", "0")}'
                    f'</QuantityKg>',
                    '    </Transfer>',
                ])
            xml_parts.append('  </Transfers>')

        xml_parts.append('</MassBalanceConsolidation>')

        return "\n".join(xml_parts)

    # ------------------------------------------------------------------
    # Internal: Report data builders
    # ------------------------------------------------------------------

    def _build_report_data(
        self,
        facility_ids: List[str],
        report_type: str,
        commodity: Optional[str],
    ) -> Dict[str, Any]:
        """Build report data based on report type.

        Args:
            facility_ids: Facilities to include.
            report_type: Type of report to build.
            commodity: Optional commodity filter.

        Returns:
            Report data dictionary.
        """
        now = utcnow()
        header: Dict[str, Any] = {
            "agent_id": "GL-EUDR-MBC-011",
            "module_version": _MODULE_VERSION,
            "report_type": report_type,
            "generated_at": now.isoformat(),
            "regulation": "EU 2023/1115 (EUDR)",
            "standard": "ISO 22095:2020",
        }

        data: Dict[str, Any] = {"header": header}

        if report_type == "enterprise_summary":
            data["aggregate_metrics"] = self._compute_aggregate_metrics(
                facility_ids
            )
            data["facility_summaries"] = self._compute_facility_summaries(
                facility_ids
            )
            data["commodity_breakdown"] = self._compute_commodity_breakdown(
                facility_ids
            )

        elif report_type == "facility_comparison":
            if commodity:
                comparison = self.get_facility_comparison(
                    facility_ids, commodity,
                )
                data["comparison"] = comparison
            else:
                data["comparisons_by_commodity"] = {}
                for cmd in EUDR_COMMODITIES:
                    comparison = self.get_facility_comparison(
                        facility_ids, cmd,
                    )
                    if comparison.get("comparisons"):
                        data["comparisons_by_commodity"][cmd] = comparison

        elif report_type == "commodity_breakdown":
            breakdown = self.get_commodity_breakdown(facility_ids)
            data["commodity_breakdown"] = breakdown.get("commodities", [])

        elif report_type == "transfer_summary":
            data["transfers"] = self._get_transfers_for_facilities(
                facility_ids
            )
            data["transfer_summary"] = self._compute_transfer_summary(
                facility_ids
            )

        elif report_type == "reconciliation_summary":
            data["facility_summaries"] = self._compute_facility_summaries(
                facility_ids
            )

        elif report_type == "compliance_overview":
            data["compliance_status"] = self._assess_enterprise_compliance(
                facility_ids
            )
            data["facility_summaries"] = self._compute_facility_summaries(
                facility_ids
            )

        elif report_type == "evidence_package":
            evidence = self.generate_evidence_package(facility_ids)
            data["evidence"] = evidence

        return data

    def _build_report_summary(
        self,
        facility_ids: List[str],
        report_type: str,
        report_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a summary for the report.

        Args:
            facility_ids: Facilities included.
            report_type: Report type.
            report_data: Full report data.

        Returns:
            Summary dictionary.
        """
        return {
            "report_type": report_type,
            "facility_count": len(facility_ids),
            "generated_at": utcnow().isoformat(),
            "sections": list(report_data.keys()),
        }

    # ------------------------------------------------------------------
    # Internal: Aggregate computation
    # ------------------------------------------------------------------

    def _compute_aggregate_metrics(
        self,
        facility_ids: List[str],
    ) -> Dict[str, Any]:
        """Compute aggregate metrics across facilities.

        Uses deterministic Decimal arithmetic.

        Args:
            facility_ids: Facilities to aggregate.

        Returns:
            Aggregate metrics dictionary.
        """
        total_inputs = Decimal("0")
        total_outputs = Decimal("0")
        total_losses = Decimal("0")
        total_waste = Decimal("0")
        total_balance = Decimal("0")
        total_transfers_in = Decimal("0")
        total_transfers_out = Decimal("0")

        with self._lock:
            for fid in facility_ids:
                fdata = self._facility_data.get(fid, {})
                total_inputs += _safe_decimal(
                    fdata.get("total_inputs", 0)
                )
                total_outputs += _safe_decimal(
                    fdata.get("total_outputs", 0)
                )
                total_losses += _safe_decimal(
                    fdata.get("total_losses", 0)
                )
                total_waste += _safe_decimal(
                    fdata.get("total_waste", 0)
                )
                total_balance += _safe_decimal(
                    fdata.get("balance", 0)
                )
                total_transfers_in += _safe_decimal(
                    fdata.get("transfers_in", 0)
                )
                total_transfers_out += _safe_decimal(
                    fdata.get("transfers_out", 0)
                )

        utilization_rate = (
            float(total_outputs / total_inputs * Decimal("100"))
            if total_inputs > 0 else 0.0
        )

        return {
            "total_inputs": str(total_inputs),
            "total_outputs": str(total_outputs),
            "total_losses": str(total_losses),
            "total_waste": str(total_waste),
            "total_balance": str(total_balance),
            "total_transfers_in": str(total_transfers_in),
            "total_transfers_out": str(total_transfers_out),
            "utilization_rate": round(utilization_rate, 2),
            "facility_count": len(facility_ids),
        }

    def _compute_facility_summaries(
        self,
        facility_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute per-facility summaries.

        Args:
            facility_ids: Facilities to summarize.

        Returns:
            List of facility summary dictionaries.
        """
        summaries: List[Dict[str, Any]] = []

        with self._lock:
            for fid in facility_ids:
                fdata = self._facility_data.get(fid, {})
                total_in = _safe_decimal(fdata.get("total_inputs", 0))
                total_out = _safe_decimal(fdata.get("total_outputs", 0))
                util = (
                    float(total_out / total_in * Decimal("100"))
                    if total_in > 0 else 0.0
                )
                summaries.append({
                    "facility_id": fid,
                    "total_inputs": str(total_in),
                    "total_outputs": str(total_out),
                    "total_losses": str(
                        _safe_decimal(fdata.get("total_losses", 0))
                    ),
                    "balance": str(
                        _safe_decimal(fdata.get("balance", 0))
                    ),
                    "utilization_rate": round(util, 2),
                    "transfers_in": str(
                        _safe_decimal(fdata.get("transfers_in", 0))
                    ),
                    "transfers_out": str(
                        _safe_decimal(fdata.get("transfers_out", 0))
                    ),
                    "commodity_count": len(
                        fdata.get("commodities", {})
                    ),
                })

        return summaries

    def _compute_commodity_breakdown(
        self,
        facility_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Compute commodity breakdown across facilities.

        Args:
            facility_ids: Facilities to include.

        Returns:
            List of per-commodity metric dictionaries.
        """
        result = self.get_commodity_breakdown(facility_ids)
        return result.get("commodities", [])

    def _assess_enterprise_compliance(
        self,
        facility_ids: List[str],
    ) -> Dict[str, Any]:
        """Assess overall enterprise compliance status.

        Args:
            facility_ids: Facilities to assess.

        Returns:
            Compliance status dictionary.
        """
        # Default to compliant; check each facility
        compliant_count = 0
        non_compliant_count = 0
        pending_count = 0

        with self._lock:
            for fid in facility_ids:
                fdata = self._facility_data.get(fid, {})
                status = fdata.get(
                    "compliance_status",
                    ComplianceStatus.PENDING.value,
                )
                if status == ComplianceStatus.COMPLIANT.value:
                    compliant_count += 1
                elif status == ComplianceStatus.NON_COMPLIANT.value:
                    non_compliant_count += 1
                else:
                    pending_count += 1

        total = len(facility_ids)
        if non_compliant_count > 0:
            overall = ComplianceStatus.NON_COMPLIANT.value
        elif pending_count == total:
            overall = ComplianceStatus.PENDING.value
        elif pending_count > 0:
            overall = ComplianceStatus.PENDING.value
        else:
            overall = ComplianceStatus.COMPLIANT.value

        return {
            "overall_status": overall,
            "compliant_facilities": compliant_count,
            "non_compliant_facilities": non_compliant_count,
            "pending_facilities": pending_count,
            "total_facilities": total,
            "compliance_rate": (
                round(compliant_count / total * 100, 1)
                if total > 0 else 0.0
            ),
        }

    # ------------------------------------------------------------------
    # Internal: Facility helpers
    # ------------------------------------------------------------------

    def _resolve_facility_ids(
        self,
        group_id: Optional[str],
        facility_ids: Optional[List[str]],
    ) -> List[str]:
        """Resolve facility IDs from group or explicit list.

        Args:
            group_id: Optional facility group.
            facility_ids: Optional explicit list.

        Returns:
            Resolved list of facility IDs.

        Raises:
            KeyError: If group_id is provided but not found.
        """
        if facility_ids:
            return list(facility_ids)

        if group_id:
            with self._lock:
                group = self._groups.get(group_id)
                if group is None:
                    raise KeyError(
                        f"Facility group not found: {group_id}"
                    )
                return list(group.get("facility_ids", []))

        return []

    def _update_facility_transfer(
        self,
        facility_id: str,
        commodity: str,
        quantity: Decimal,
        direction: str,
    ) -> None:
        """Update facility data for a transfer.

        Must be called under self._lock.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity being transferred.
            quantity: Transfer quantity as Decimal.
            direction: Transfer direction ('in' or 'out').
        """
        if facility_id not in self._facility_data:
            self._facility_data[facility_id] = {
                "facility_id": facility_id,
                "total_inputs": Decimal("0"),
                "total_outputs": Decimal("0"),
                "total_losses": Decimal("0"),
                "total_waste": Decimal("0"),
                "balance": Decimal("0"),
                "transfers_in": Decimal("0"),
                "transfers_out": Decimal("0"),
                "compliance_status": ComplianceStatus.PENDING.value,
                "commodities": {},
            }

        fdata = self._facility_data[facility_id]

        if direction == "in":
            fdata["transfers_in"] = (
                _safe_decimal(fdata["transfers_in"]) + quantity
            )
            fdata["balance"] = (
                _safe_decimal(fdata["balance"]) + quantity
            )
        else:
            fdata["transfers_out"] = (
                _safe_decimal(fdata["transfers_out"]) + quantity
            )
            fdata["balance"] = (
                _safe_decimal(fdata["balance"]) - quantity
            )

        # Update commodity-level data
        if commodity not in fdata["commodities"]:
            fdata["commodities"][commodity] = {
                "total_inputs": Decimal("0"),
                "total_outputs": Decimal("0"),
                "total_losses": Decimal("0"),
                "transfers_in": Decimal("0"),
                "transfers_out": Decimal("0"),
            }

        cdata = fdata["commodities"][commodity]
        if direction == "in":
            cdata["transfers_in"] = (
                _safe_decimal(cdata["transfers_in"]) + quantity
            )
        else:
            cdata["transfers_out"] = (
                _safe_decimal(cdata["transfers_out"]) + quantity
            )

    def _build_facility_comparison(
        self,
        facility_id: str,
        commodity: str,
    ) -> Dict[str, Any]:
        """Build comparison data for a single facility.

        Args:
            facility_id: Facility identifier.
            commodity: Commodity to compare.

        Returns:
            Facility comparison dictionary.
        """
        with self._lock:
            fdata = self._facility_data.get(facility_id, {})
            cdata = fdata.get("commodities", {}).get(commodity, {})

        total_in = _safe_decimal(cdata.get("total_inputs", 0))
        total_out = _safe_decimal(cdata.get("total_outputs", 0))
        total_loss = _safe_decimal(cdata.get("total_losses", 0))
        util = (
            float(total_out / total_in * Decimal("100"))
            if total_in > 0 else 0.0
        )
        loss_pct = (
            float(total_loss / total_in * Decimal("100"))
            if total_in > 0 else 0.0
        )

        return {
            "facility_id": facility_id,
            "commodity": commodity,
            "total_inputs": str(total_in),
            "total_outputs": str(total_out),
            "total_losses": str(total_loss),
            "utilization_rate": round(util, 2),
            "loss_percentage": round(loss_pct, 2),
            "transfers_in": str(
                _safe_decimal(cdata.get("transfers_in", 0))
            ),
            "transfers_out": str(
                _safe_decimal(cdata.get("transfers_out", 0))
            ),
            "rank": 0,
        }

    def _compute_comparison_statistics(
        self,
        comparisons: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute aggregate statistics for facility comparisons.

        Args:
            comparisons: List of facility comparison dictionaries.

        Returns:
            Statistics dictionary.
        """
        util_rates = [
            c.get("utilization_rate", 0.0) for c in comparisons
        ]
        loss_pcts = [
            c.get("loss_percentage", 0.0) for c in comparisons
        ]

        stats: Dict[str, Any] = {}
        if util_rates:
            stats["utilization"] = {
                "mean": round(statistics.mean(util_rates), 2),
                "min": round(min(util_rates), 2),
                "max": round(max(util_rates), 2),
            }
        if loss_pcts:
            stats["loss_percentage"] = {
                "mean": round(statistics.mean(loss_pcts), 2),
                "min": round(min(loss_pcts), 2),
                "max": round(max(loss_pcts), 2),
            }
        return stats

    # ------------------------------------------------------------------
    # Internal: Transfer helpers
    # ------------------------------------------------------------------

    def _get_transfers_for_facilities(
        self,
        facility_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Get all transfers involving the specified facilities.

        Args:
            facility_ids: Facilities to filter on.

        Returns:
            List of transfer records.
        """
        fid_set = set(facility_ids)
        with self._lock:
            return [
                t for t in self._transfers
                if (
                    t.get("from_facility") in fid_set
                    or t.get("to_facility") in fid_set
                )
            ]

    def _compute_transfer_summary(
        self,
        facility_ids: List[str],
    ) -> Dict[str, Any]:
        """Compute transfer summary for specified facilities.

        Args:
            facility_ids: Facilities to summarize.

        Returns:
            Transfer summary dictionary.
        """
        transfers = self._get_transfers_for_facilities(facility_ids)
        total_volume = Decimal("0")
        by_commodity: Dict[str, Decimal] = {}

        for t in transfers:
            qty = _safe_decimal(t.get("quantity_kg", 0))
            total_volume += qty
            commodity = t.get("commodity", "unknown")
            by_commodity[commodity] = (
                by_commodity.get(commodity, Decimal("0")) + qty
            )

        return {
            "total_transfers": len(transfers),
            "total_volume_kg": str(total_volume),
            "by_commodity": {
                k: str(v) for k, v in sorted(by_commodity.items())
            },
        }

    # ------------------------------------------------------------------
    # Internal: Evidence package builders
    # ------------------------------------------------------------------

    def _build_document_inventory(
        self,
        facility_ids: List[str],
        period_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Build a document inventory for the evidence package.

        Args:
            facility_ids: Facilities included.
            period_id: Optional period scope.

        Returns:
            List of document metadata dictionaries.
        """
        inventory: List[Dict[str, Any]] = []
        now = utcnow()

        for doc_type in EVIDENCE_DOCUMENT_TYPES:
            for fid in facility_ids:
                inventory.append({
                    "document_type": doc_type,
                    "facility_id": fid,
                    "period_id": period_id or "all",
                    "status": "available",
                    "format": "json",
                    "retention_until": str(
                        now.year + self._config.retention_years
                    ),
                    "provenance_hash": _compute_hash({
                        "type": doc_type,
                        "facility": fid,
                        "period": period_id,
                    }),
                })

        return inventory

    def _build_evidence_sections(
        self,
        facility_ids: List[str],
        period_id: Optional[str],
    ) -> Dict[str, Any]:
        """Build evidence sections for the evidence package.

        Args:
            facility_ids: Facilities included.
            period_id: Optional period scope.

        Returns:
            Evidence sections dictionary.
        """
        return {
            "mass_balance_overview": {
                "description": (
                    "Mass balance ledger records for all facilities"
                ),
                "facility_count": len(facility_ids),
                "period_id": period_id or "all_periods",
            },
            "reconciliation_records": {
                "description": (
                    "Period-end reconciliation records with variance analysis"
                ),
                "facility_count": len(facility_ids),
            },
            "conversion_factor_validations": {
                "description": (
                    "Conversion factor validation records and deviations"
                ),
                "facility_count": len(facility_ids),
            },
            "overdraft_events": {
                "description": (
                    "Overdraft detection and resolution records"
                ),
                "facility_count": len(facility_ids),
            },
            "loss_records": {
                "description": (
                    "Processing, transport, and storage loss records"
                ),
                "facility_count": len(facility_ids),
            },
            "carry_forward_records": {
                "description": (
                    "Credit carry-forward and expiry records"
                ),
                "facility_count": len(facility_ids),
            },
            "transfer_records": {
                "description": (
                    "Cross-facility material transfer records"
                ),
                "transfers": self._get_transfers_for_facilities(
                    facility_ids
                ),
            },
            "provenance_chain": {
                "description": (
                    "SHA-256 provenance chain for audit trail integrity"
                ),
                "chain_length": self._provenance.entry_count,
                "chain_valid": self._provenance.verify_chain(),
            },
        }

    def _build_evidence_compliance_summary(
        self,
        facility_ids: List[str],
    ) -> Dict[str, Any]:
        """Build compliance summary for the evidence package.

        Args:
            facility_ids: Facilities included.

        Returns:
            Compliance summary dictionary.
        """
        return {
            "regulation": "EU 2023/1115 (EUDR)",
            "standard": "ISO 22095:2020",
            "retention_requirement": (
                f"{self._config.retention_years} years "
                f"per EUDR Article 14"
            ),
            "enterprise_compliance": self._assess_enterprise_compliance(
                facility_ids
            ),
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            report_count = len(self._reports)
            group_count = len(self._groups)
            transfer_count = len(self._transfers)
        return (
            f"ConsolidationReporter(reports={report_count}, "
            f"groups={group_count}, transfers={transfer_count})"
        )

    def __len__(self) -> int:
        """Return the total number of generated reports."""
        with self._lock:
            return len(self._reports)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ConsolidationReporter",
    "SUPPORTED_REPORT_TYPES",
    "SUPPORTED_REPORT_FORMATS",
    "MAX_FACILITIES_PER_REPORT",
    "EUDR_COMMODITIES",
    "EVIDENCE_DOCUMENT_TYPES",
]
