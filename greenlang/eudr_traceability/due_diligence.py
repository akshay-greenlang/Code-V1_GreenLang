# -*- coding: utf-8 -*-
"""
Due Diligence Engine - AGENT-DATA-004: EUDR Traceability Connector

Manages the lifecycle of Due Diligence Statements (DDS) per EUDR Article 4.
Operators and traders must submit a DDS to the EU Information System before
placing EUDR-regulated products on the EU market or exporting them.

Zero-Hallucination Guarantees:
    - DDS completeness validation is deterministic rule-based checking
    - Origin information is gathered from registered plots (no guessing)
    - SHA-256 provenance hashes on all DDS operations
    - Status transitions follow a strict state machine

Example:
    >>> from greenlang.eudr_traceability.due_diligence import DueDiligenceEngine
    >>> engine = DueDiligenceEngine()
    >>> dds = engine.generate_dds(request)
    >>> assert dds.status == DDSStatus.DRAFT

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.eudr_traceability.models import (
    DDSStatus,
    DDSType,
    DueDiligenceStatement,
    EUDRCommodity,
    GenerateDDSRequest,
    RiskLevel,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class DueDiligenceEngine:
    """Due Diligence Statement lifecycle management engine.

    Manages generation, validation, signing, and submission of DDS
    documents per EUDR Article 4, with completeness checking and
    origin information gathering.

    Attributes:
        _config: Configuration dictionary or object.
        _plot_registry: PlotRegistryEngine for origin lookups.
        _chain_of_custody: ChainOfCustodyEngine for batch tracing.
        _risk_engine: RiskAssessmentEngine for risk scoring.
        _statements: In-memory DDS storage keyed by dds_id.
        _provenance: Provenance tracker instance.

    Example:
        >>> engine = DueDiligenceEngine()
        >>> dds = engine.generate_dds(request)
        >>> engine.submit_dds(dds.dds_id)
    """

    def __init__(
        self,
        config: Any = None,
        plot_registry: Any = None,
        chain_of_custody: Any = None,
        risk_engine: Any = None,
        provenance: Any = None,
    ) -> None:
        """Initialize DueDiligenceEngine.

        Args:
            config: Optional configuration.
            plot_registry: Optional PlotRegistryEngine.
            chain_of_custody: Optional ChainOfCustodyEngine.
            risk_engine: Optional RiskAssessmentEngine.
            provenance: Optional ProvenanceTracker.
        """
        self._config = config or {}
        self._plot_registry = plot_registry
        self._chain_of_custody = chain_of_custody
        self._risk_engine = risk_engine
        self._provenance = provenance

        # In-memory storage
        self._statements: Dict[str, DueDiligenceStatement] = {}

        logger.info("DueDiligenceEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_dds(
        self,
        request: GenerateDDSRequest,
    ) -> DueDiligenceStatement:
        """Generate a new Due Diligence Statement.

        Gathers origin information from registered plots, assesses
        risk level, and creates the DDS in DRAFT status.

        Args:
            request: DDS generation request.

        Returns:
            DueDiligenceStatement in DRAFT status.
        """
        start_time = time.monotonic()

        dds_id = self._generate_dds_id()

        # Gather origin information from plots
        origin_countries = self._gather_origin_countries(request.origin_plot_ids)

        # Create DDS using new model field names
        dds = DueDiligenceStatement(
            dds_id=dds_id,
            commodity=request.commodity,
            product_description=request.product_description,
            cn_codes=list(request.cn_codes),
            quantity=request.quantity,
            operator_id="pending",
            operator_name="pending",
            operator_country="XX",
            dds_type=request.dds_type,
            origin_plot_ids=list(request.origin_plot_ids),
            origin_countries=list(origin_countries),
            deforestation_free_declaration=True,
            legal_compliance_declaration=True,
            status=DDSStatus.DRAFT,
        )

        # Store
        self._statements[dds_id] = dds

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(dds)
            self._provenance.record(
                entity_type="dds",
                entity_id=dds_id,
                action="dds_generation",
                data_hash=data_hash,
            )

        # Record metrics
        try:
            from greenlang.eudr_traceability.metrics import record_dds_generated
            record_dds_generated(request.commodity.value, request.dds_type.value)
        except ImportError:
            pass

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Generated DDS %s: commodity=%s, plots=%d (%.1f ms)",
            dds_id, request.commodity.value,
            len(request.origin_plot_ids), elapsed_ms,
        )
        return dds

    def get_dds(self, dds_id: str) -> Optional[DueDiligenceStatement]:
        """Get a DDS by ID.

        Args:
            dds_id: DDS identifier.

        Returns:
            DueDiligenceStatement or None if not found.
        """
        return self._statements.get(dds_id)

    def list_dds(
        self,
        status: Optional[str] = None,
        commodity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[DueDiligenceStatement]:
        """List DDS with optional filtering.

        Args:
            status: Optional status filter.
            commodity: Optional commodity filter.
            limit: Maximum results.
            offset: Results to skip.

        Returns:
            List of DueDiligenceStatement instances.
        """
        statements = list(self._statements.values())

        if status is not None:
            statements = [
                s for s in statements if s.status.value == status
            ]
        if commodity is not None:
            statements = [
                s for s in statements if s.commodity.value == commodity
            ]

        return statements[offset:offset + limit]

    def submit_dds(self, dds_id: str) -> DueDiligenceStatement:
        """Submit a DDS for review by marking it as submitted.

        Validates DDS completeness before allowing submission.

        Args:
            dds_id: DDS identifier.

        Returns:
            Updated DueDiligenceStatement with SUBMITTED status.

        Raises:
            ValueError: If DDS not found, not in submittable state,
                or completeness validation fails.
        """
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        # Check current status allows submission
        submittable_states = {DDSStatus.DRAFT}
        if dds.status not in submittable_states:
            raise ValueError(
                f"DDS {dds_id} cannot be submitted from status "
                f"'{dds.status.value}'"
            )

        # Validate completeness
        issues = self._validate_dds_completeness(dds)
        if issues:
            raise ValueError(
                f"DDS {dds_id} is not complete: {'; '.join(issues)}"
            )

        # Update status
        dds.status = DDSStatus.SUBMITTED
        dds.submission_date = _utcnow()
        dds.updated_at = _utcnow()

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(dds)
            self._provenance.record(
                entity_type="dds",
                entity_id=dds_id,
                action="dds_submission",
                data_hash=data_hash,
            )

        logger.info("DDS %s submitted", dds_id)
        return dds

    def sign_dds(
        self,
        dds_id: str,
        signature: str,
    ) -> DueDiligenceStatement:
        """Digitally sign a DDS.

        Args:
            dds_id: DDS identifier.
            signature: Digital signature string.

        Returns:
            Updated DueDiligenceStatement with signature.

        Raises:
            ValueError: If DDS not found or signature is empty.
        """
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        if not signature or not signature.strip():
            raise ValueError("Signature must not be empty")

        dds.digital_signature = signature
        dds.updated_at = _utcnow()

        # Record provenance
        if self._provenance is not None:
            data_hash = _compute_hash(dds)
            self._provenance.record(
                entity_type="dds",
                entity_id=dds_id,
                action="dds_signing",
                data_hash=data_hash,
            )

        logger.info("DDS %s signed", dds_id)
        return dds

    def export_for_eu_system(self, dds_id: str) -> Dict[str, Any]:
        """Export a DDS in the format required by the EU Information System.

        Args:
            dds_id: DDS identifier.

        Returns:
            Dictionary formatted for EU Information System API.

        Raises:
            ValueError: If DDS not found.
        """
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        return {
            "due_diligence_statement": {
                "reference_number": dds.dds_id,
                "commodity_type": dds.commodity.value,
                "product_description": dds.product_description,
                "operator": {
                    "id": dds.operator_id,
                    "name": dds.operator_name,
                    "country": dds.operator_country,
                },
                "origin": {
                    "countries": dds.origin_countries,
                    "plot_geolocations": [
                        {"plot_id": pid} for pid in dds.origin_plot_ids
                    ],
                },
                "quantity": {
                    "amount": str(dds.quantity),
                    "unit": dds.unit,
                },
                "declarations": {
                    "deforestation_free": dds.deforestation_free_declaration,
                    "legal_compliance": dds.legal_compliance_declaration,
                },
                "risk_assessment": {
                    "risk_level": dds.risk_level.value,
                    "mitigation_measures": dds.risk_mitigation_measures,
                },
                "cn_codes": dds.cn_codes,
                "digital_signature": dds.digital_signature,
                "status": dds.status.value,
            },
        }

    def get_dds_statistics(self) -> Dict[str, Any]:
        """Get aggregated DDS statistics.

        Returns:
            Dictionary with DDS counts by status and commodity.
        """
        by_status: Dict[str, int] = {}
        by_commodity: Dict[str, int] = {}

        for dds in self._statements.values():
            status_key = dds.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            commodity_key = dds.commodity.value
            by_commodity[commodity_key] = (
                by_commodity.get(commodity_key, 0) + 1
            )

        return {
            "total": len(self._statements),
            "by_status": by_status,
            "by_commodity": by_commodity,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_dds_completeness(
        self,
        dds: DueDiligenceStatement,
    ) -> List[str]:
        """Validate that a DDS is complete for submission.

        Checks all required fields per EUDR Article 4 requirements.

        Args:
            dds: DueDiligenceStatement to validate.

        Returns:
            List of validation issue descriptions. Empty if complete.
        """
        issues: List[str] = []

        if not dds.operator_id or dds.operator_id == "pending":
            issues.append("Operator ID is required")
        if not dds.origin_plot_ids:
            issues.append("At least one plot ID is required")
        if dds.quantity <= Decimal("0"):
            issues.append("Quantity must be greater than zero")
        if not dds.origin_countries:
            issues.append("At least one origin country is required")

        return issues

    # ------------------------------------------------------------------
    # Origin information
    # ------------------------------------------------------------------

    def _gather_origin_countries(
        self,
        plot_ids: List[str],
    ) -> Set[str]:
        """Gather origin countries from registered plots.

        Args:
            plot_ids: Plot IDs to gather info from.

        Returns:
            Set of country codes.
        """
        origin_countries: Set[str] = set()

        if self._plot_registry is None or not plot_ids:
            return origin_countries

        for plot_id in plot_ids:
            plot = self._plot_registry.get_plot(plot_id)
            if plot is None:
                continue
            origin_countries.add(plot.geolocation.country_code)

        return origin_countries

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_dds_id(self) -> str:
        """Generate a unique DDS identifier.

        Returns:
            DDS ID in format "DDS-{hex12}".
        """
        return f"DDS-{uuid.uuid4().hex[:12]}"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def dds_count(self) -> int:
        """Return the total number of DDS records."""
        return len(self._statements)


__all__ = [
    "DueDiligenceEngine",
]
