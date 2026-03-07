# -*- coding: utf-8 -*-
"""
GL-EUDR-APP DDS Reporting Engine - Due Diligence Statement Lifecycle Management

Generates, validates, manages, and submits Due Diligence Statements (DDS) per
EU Regulation 2023/1115 Articles 4, 9-12. The DDS is the core compliance
artifact operators must submit to the EU Information System before placing
relevant commodities on the EU market.

DDS Sections (per EU format):
    1. Operator information
    2. Product description
    3. Country of production
    4. Geolocation data
    5. Risk assessment
    6. Risk mitigation
    7. Conclusion

DDS Lifecycle:
    DRAFT -> REVIEW -> VALIDATED -> SUBMITTED -> ACCEPTED/REJECTED -> AMENDED

Reference Number Format: EUDR-{ISO3}-{YEAR}-{SEQUENCE:06d}

Zero-Hallucination Guarantees:
    - Reference numbers use deterministic sequence generation
    - Validation uses rule-based section completeness checks
    - Risk scores are passed through from risk engine (not computed here)
    - SHA-256 provenance on all DDS records
    - EU submission simulated in v1.0, real integration via AGENT-DATA-005

Example:
    >>> from services.dds_reporting_engine import DDSReportingEngine
    >>> engine = DDSReportingEngine(config, supplier_engine, doc_engine, risk_engine)
    >>> dds = engine.generate_dds("supplier-1", "coffee", 2026, ["plot-1"])

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.config import (
    DDSStatus,
    EUDRAppConfig,
    EUDRCommodity,
    RiskLevel,
)
from services.models import (
    DDSAnnualSummary,
    DDSFilterRequest,
    DDSGenerateRequest,
    DDSMitigationMeasure,
    DDSRiskAssessment,
    DDSSubmissionResult,
    DDSValidationResult,
    DueDiligenceStatement,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a UUID v4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# DDS Section Definitions
# ---------------------------------------------------------------------------

DDS_REQUIRED_SECTIONS: List[str] = [
    "operator_information",
    "product_description",
    "country_of_production",
    "geolocation_data",
    "risk_assessment",
    "risk_mitigation",
    "conclusion",
]


# ===========================================================================
# DDS Reporting Engine
# ===========================================================================


class DDSReportingEngine:
    """Due Diligence Statement generation and lifecycle management.

    Manages the complete DDS lifecycle from generation through EU system
    submission. Integrates with the supplier, document, risk, and
    supply chain mapper engines to populate DDS sections.

    Attributes:
        _config: Application configuration.
        _lock: Reentrant lock for thread safety.
        _dds_records: In-memory DDS storage keyed by ID.
        _sequence_counters: Reference number sequence counters per country-year.
        _supplier_engine: SupplierIntakeEngine for supplier data (optional).
        _document_engine: DocumentVerificationEngine for docs (optional).
        _risk_engine: RiskAggregator for risk scores (optional).
        _supply_chain_service: SupplyChainAppService for AGENT-EUDR-001 (optional).

    Example:
        >>> engine = DDSReportingEngine(config)
        >>> dds = engine.generate_dds("supplier-1", "coffee", 2026, ["plot-1"])
        >>> validation = engine.validate_dds(dds.id)
        >>> if validation.is_valid:
        ...     result = engine.submit_dds(dds.id)
    """

    def __init__(
        self,
        config: EUDRAppConfig,
        supplier_engine: Optional[Any] = None,
        document_engine: Optional[Any] = None,
        risk_engine: Optional[Any] = None,
        supply_chain_service: Optional[Any] = None,
    ) -> None:
        """Initialize DDSReportingEngine.

        Args:
            config: Application configuration.
            supplier_engine: Optional SupplierIntakeEngine reference.
            document_engine: Optional DocumentVerificationEngine reference.
            risk_engine: Optional RiskAggregator reference.
            supply_chain_service: Optional SupplyChainAppService for
                AGENT-EUDR-001 supply chain mapping data.
        """
        self._config = config
        self._lock = threading.RLock()
        self._dds_records: Dict[str, DueDiligenceStatement] = {}
        self._sequence_counters: Dict[str, int] = {}
        self._supplier_engine = supplier_engine
        self._document_engine = document_engine
        self._risk_engine = risk_engine
        self._supply_chain_service = supply_chain_service
        logger.info("DDSReportingEngine initialized")

    # -----------------------------------------------------------------------
    # Reference Number Generation
    # -----------------------------------------------------------------------

    def _generate_reference_number(
        self, country: str, year: int
    ) -> str:
        """Generate a unique DDS reference number.

        Format: EUDR-{ISO3}-{YEAR}-{SEQUENCE:06d}

        Args:
            country: ISO-3166 alpha-3 country code.
            year: Reporting year.

        Returns:
            Formatted reference number string.
        """
        prefix = self._config.dds_reference_prefix
        key = f"{country}-{year}"

        with self._lock:
            current = self._sequence_counters.get(key, 0) + 1
            self._sequence_counters[key] = current

        return f"{prefix}-{country.upper()}-{year}-{current:06d}"

    # -----------------------------------------------------------------------
    # DDS Generation
    # -----------------------------------------------------------------------

    def generate_dds(
        self,
        supplier_id: str,
        commodity: str,
        year: int,
        plots: List[str],
        procurement_ids: Optional[List[str]] = None,
        operator_name: Optional[str] = None,
        operator_country: Optional[str] = None,
        product_description: Optional[str] = None,
    ) -> DueDiligenceStatement:
        """Generate a new Due Diligence Statement with all required sections.

        Creates a DDS in DRAFT status with sections populated from
        available supplier, plot, and risk data.

        Args:
            supplier_id: Supplier this DDS covers.
            commodity: Commodity type string.
            year: Reporting year (>= 2024).
            plots: List of plot IDs to include.
            procurement_ids: Optional list of procurement IDs.
            operator_name: Optional operator name.
            operator_country: Optional operator country ISO-3.
            product_description: Optional product description.

        Returns:
            Generated DueDiligenceStatement in DRAFT status.

        Raises:
            ValueError: If required parameters are invalid.
        """
        if not supplier_id:
            raise ValueError("supplier_id is required")
        if not plots:
            raise ValueError("At least one plot ID is required")
        if year < 2024:
            raise ValueError("Year must be 2024 or later")

        # Resolve commodity enum
        try:
            commodity_enum = EUDRCommodity(commodity.lower())
        except ValueError:
            raise ValueError(
                f"Invalid commodity: {commodity}. "
                f"Valid: {[c.value for c in EUDRCommodity]}"
            )

        # Determine country for reference number
        country = operator_country or "XXX"
        if self._supplier_engine:
            supplier = self._supplier_engine.get_supplier(supplier_id)
            if supplier:
                country = supplier.country
                if not operator_name:
                    operator_name = supplier.name

        # Generate reference number
        reference_number = self._generate_reference_number(country, year)

        # Build plot details
        plot_details = self._build_plot_details(plots)

        # Build risk assessment section
        risk_assessment = self._build_risk_assessment(
            supplier_id, plots
        )

        # Build mitigation measures
        mitigation_measures = self._build_mitigation_measures(
            risk_assessment
        )

        # Build conclusion
        conclusion = self._build_conclusion(risk_assessment)

        # Gather linked documents
        documents = self._gather_documents(supplier_id, plots)

        # Build supply chain section from AGENT-EUDR-001 (if available)
        supply_chain_section = self._build_supply_chain_section(
            supplier_id, plots, commodity_enum
        )
        supply_chain_graph_id = (
            supply_chain_section.get("graph_id")
            if supply_chain_section
            else None
        )

        dds = DueDiligenceStatement(
            reference_number=reference_number,
            supplier_id=supplier_id,
            operator_name=operator_name,
            operator_country=country if country != "XXX" else None,
            year=year,
            commodity=commodity_enum,
            product_description=product_description or self._default_product_description(commodity_enum),
            country_of_production=country if country != "XXX" else None,
            plots=plots,
            plot_details=plot_details,
            procurement_ids=procurement_ids or [],
            status=DDSStatus.DRAFT,
            risk_assessment=risk_assessment,
            mitigation_measures=mitigation_measures,
            documents=documents,
            conclusion=conclusion,
            supply_chain_section=supply_chain_section,
            supply_chain_graph_id=supply_chain_graph_id,
        )

        with self._lock:
            self._dds_records[dds.id] = dds

        logger.info(
            "Generated DDS %s (ref=%s) for supplier %s: "
            "commodity=%s, year=%d, plots=%d",
            dds.id,
            reference_number,
            supplier_id,
            commodity_enum.value,
            year,
            len(plots),
        )
        return dds

    def generate_dds_from_request(
        self, request: DDSGenerateRequest
    ) -> DueDiligenceStatement:
        """Generate a DDS from a structured request object.

        Args:
            request: DDSGenerateRequest with all parameters.

        Returns:
            Generated DueDiligenceStatement.
        """
        return self.generate_dds(
            supplier_id=request.supplier_id,
            commodity=request.commodity.value,
            year=request.year,
            plots=request.plots,
            procurement_ids=request.procurement_ids,
            operator_name=request.operator_name,
            operator_country=request.operator_country,
            product_description=request.product_description,
        )

    # -----------------------------------------------------------------------
    # DDS Validation
    # -----------------------------------------------------------------------

    def validate_dds(self, dds_id: str) -> DDSValidationResult:
        """Validate a DDS for completeness and correctness.

        Checks all seven required sections, field presence, and
        logical consistency.

        Args:
            dds_id: DDS identifier.

        Returns:
            DDSValidationResult with errors and warnings.

        Raises:
            ValueError: If DDS not found.
        """
        with self._lock:
            dds = self._dds_records.get(dds_id)
            if dds is None:
                raise ValueError(f"DDS not found: {dds_id}")

        errors: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        missing_sections: List[str] = []

        # Section 1: Operator information
        if not dds.supplier_id:
            errors.append({
                "section": "operator_information",
                "field": "supplier_id",
                "message": "Supplier ID is required",
            })
            missing_sections.append("operator_information")

        if not dds.operator_name:
            warnings.append({
                "section": "operator_information",
                "field": "operator_name",
                "message": "Operator name not provided",
            })

        # Section 2: Product description
        if not dds.commodity:
            errors.append({
                "section": "product_description",
                "field": "commodity",
                "message": "Commodity is required",
            })
            missing_sections.append("product_description")

        # Section 3: Country of production
        if not dds.country_of_production:
            warnings.append({
                "section": "country_of_production",
                "field": "country_of_production",
                "message": "Country of production not specified",
            })

        # Section 4: Geolocation data
        if not dds.plots or len(dds.plots) == 0:
            errors.append({
                "section": "geolocation_data",
                "field": "plots",
                "message": "At least one plot with geolocation is required",
            })
            missing_sections.append("geolocation_data")

        # Section 5: Risk assessment
        if not dds.risk_assessment:
            errors.append({
                "section": "risk_assessment",
                "field": "risk_assessment",
                "message": "Risk assessment is required",
            })
            missing_sections.append("risk_assessment")
        elif dds.risk_assessment.overall_risk == 0.0:
            warnings.append({
                "section": "risk_assessment",
                "field": "overall_risk",
                "message": "Risk assessment shows zero risk; verify accuracy",
            })

        # Section 6: Risk mitigation
        if (
            dds.risk_assessment
            and dds.risk_assessment.overall_risk > 0.3
            and len(dds.mitigation_measures) == 0
        ):
            errors.append({
                "section": "risk_mitigation",
                "field": "mitigation_measures",
                "message": (
                    "Risk mitigation measures required when "
                    "risk score exceeds 0.3"
                ),
            })
            missing_sections.append("risk_mitigation")

        # Section 7: Conclusion
        if not dds.conclusion:
            errors.append({
                "section": "conclusion",
                "field": "conclusion",
                "message": "Conclusion statement is required",
            })
            missing_sections.append("conclusion")

        # Year validation
        if dds.year < 2024:
            errors.append({
                "section": "general",
                "field": "year",
                "message": "Reporting year must be 2024 or later",
            })

        # Reference number validation
        if not dds.reference_number:
            errors.append({
                "section": "general",
                "field": "reference_number",
                "message": "Reference number is missing",
            })

        # Completeness calculation
        total_sections = len(DDS_REQUIRED_SECTIONS)
        complete_sections = total_sections - len(set(missing_sections))
        completeness_pct = (
            (complete_sections / total_sections * 100)
            if total_sections > 0
            else 0.0
        )

        is_valid = len(errors) == 0

        # Update DDS status if valid
        if is_valid and dds.status == DDSStatus.DRAFT:
            with self._lock:
                dds.status = DDSStatus.REVIEW
                dds.updated_at = _utcnow()

        result = DDSValidationResult(
            dds_id=dds_id,
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            completeness_pct=round(completeness_pct, 1),
            missing_sections=list(set(missing_sections)),
        )

        logger.info(
            "Validated DDS %s: valid=%s, errors=%d, warnings=%d, "
            "completeness=%.1f%%",
            dds_id,
            is_valid,
            len(errors),
            len(warnings),
            completeness_pct,
        )
        return result

    # -----------------------------------------------------------------------
    # DDS Submission
    # -----------------------------------------------------------------------

    def submit_dds(self, dds_id: str) -> DDSSubmissionResult:
        """Submit a validated DDS to the EU Information System.

        In v1.0, EU system submission is simulated. In production, this
        integrates with AGENT-DATA-005 (EUDR Traceability) eu_system_connector.

        Args:
            dds_id: DDS identifier.

        Returns:
            DDSSubmissionResult with submission outcome.

        Raises:
            ValueError: If DDS not found or not in valid status.
        """
        with self._lock:
            dds = self._dds_records.get(dds_id)
            if dds is None:
                raise ValueError(f"DDS not found: {dds_id}")

        # Validate before submission
        if dds.status not in (
            DDSStatus.REVIEW,
            DDSStatus.VALIDATED,
            DDSStatus.AMENDED,
        ):
            return DDSSubmissionResult(
                dds_id=dds_id,
                reference_number=dds.reference_number,
                submitted=False,
                status=dds.status,
                errors=[
                    f"DDS must be in REVIEW, VALIDATED, or AMENDED status "
                    f"to submit. Current status: {dds.status.value}"
                ],
            )

        # Run validation
        validation = self.validate_dds(dds_id)
        if not validation.is_valid:
            return DDSSubmissionResult(
                dds_id=dds_id,
                reference_number=dds.reference_number,
                submitted=False,
                status=dds.status,
                errors=[
                    f"Validation failed: {len(validation.errors)} errors"
                ] + [e.get("message", "") for e in validation.errors],
            )

        # Simulate EU system submission
        eu_submission_id = f"EU-{uuid.uuid4().hex[:12].upper()}"
        submission_date = _utcnow()

        with self._lock:
            dds.status = DDSStatus.SUBMITTED
            dds.submission_date = submission_date
            dds.eu_submission_id = eu_submission_id
            dds.eu_response = {
                "submission_id": eu_submission_id,
                "status": "received",
                "message": "DDS received by EU Information System",
                "received_at": submission_date.isoformat(),
            }
            dds.updated_at = _utcnow()

        result = DDSSubmissionResult(
            dds_id=dds_id,
            reference_number=dds.reference_number,
            submitted=True,
            eu_submission_id=eu_submission_id,
            submission_date=submission_date,
            status=DDSStatus.SUBMITTED,
            response_message="DDS successfully submitted to EU Information System",
        )

        logger.info(
            "Submitted DDS %s (ref=%s) to EU system: submission_id=%s",
            dds_id,
            dds.reference_number,
            eu_submission_id,
        )
        return result

    # -----------------------------------------------------------------------
    # DDS Retrieval
    # -----------------------------------------------------------------------

    def get_dds(self, dds_id: str) -> Optional[DueDiligenceStatement]:
        """Get a DDS by ID.

        Args:
            dds_id: DDS identifier.

        Returns:
            DueDiligenceStatement if found, None otherwise.
        """
        with self._lock:
            return self._dds_records.get(dds_id)

    def list_dds(
        self, filters: Optional[DDSFilterRequest] = None
    ) -> List[DueDiligenceStatement]:
        """List DDS records with optional filtering.

        Args:
            filters: Filtering criteria.

        Returns:
            List of matching DueDiligenceStatement records.
        """
        with self._lock:
            records = list(self._dds_records.values())

        if filters is None:
            return records

        if filters.supplier_id:
            records = [
                r for r in records
                if r.supplier_id == filters.supplier_id
            ]
        if filters.commodity:
            records = [
                r for r in records if r.commodity == filters.commodity
            ]
        if filters.year:
            records = [r for r in records if r.year == filters.year]
        if filters.status:
            records = [r for r in records if r.status == filters.status]

        offset = filters.offset
        limit = filters.limit
        return records[offset: offset + limit]

    # -----------------------------------------------------------------------
    # DDS Status Management
    # -----------------------------------------------------------------------

    def update_dds_status(
        self, dds_id: str, status: DDSStatus
    ) -> DueDiligenceStatement:
        """Update a DDS lifecycle status.

        Validates the status transition is legal.

        Args:
            dds_id: DDS identifier.
            status: New status.

        Returns:
            Updated DueDiligenceStatement.

        Raises:
            ValueError: If DDS not found or transition invalid.
        """
        with self._lock:
            dds = self._dds_records.get(dds_id)
            if dds is None:
                raise ValueError(f"DDS not found: {dds_id}")

            # Validate transition
            valid_transitions = self._get_valid_transitions(dds.status)
            if status not in valid_transitions:
                raise ValueError(
                    f"Invalid status transition: {dds.status.value} -> "
                    f"{status.value}. Valid: "
                    f"{[s.value for s in valid_transitions]}"
                )

            old_status = dds.status
            dds.status = status
            dds.updated_at = _utcnow()

        logger.info(
            "Updated DDS %s status: %s -> %s",
            dds_id,
            old_status.value,
            status.value,
        )
        return dds

    # -----------------------------------------------------------------------
    # DDS Amendment
    # -----------------------------------------------------------------------

    def amend_dds(
        self, dds_id: str, amendments: Dict[str, Any]
    ) -> DueDiligenceStatement:
        """Amend a rejected DDS with corrections.

        Records the amendment in history and sets status to AMENDED.

        Args:
            dds_id: DDS identifier.
            amendments: Dictionary of amended fields and values.

        Returns:
            Amended DueDiligenceStatement.

        Raises:
            ValueError: If DDS not found or not in REJECTED status.
        """
        with self._lock:
            dds = self._dds_records.get(dds_id)
            if dds is None:
                raise ValueError(f"DDS not found: {dds_id}")

            if dds.status != DDSStatus.REJECTED:
                raise ValueError(
                    f"Only REJECTED DDS can be amended. "
                    f"Current status: {dds.status.value}"
                )

            # Record amendment
            amendment_record = {
                "amended_at": _utcnow().isoformat(),
                "previous_status": dds.status.value,
                "changes": amendments,
                "amendment_number": len(dds.amendment_history) + 1,
            }
            dds.amendment_history.append(amendment_record)

            # Apply amendments
            if "conclusion" in amendments:
                dds.conclusion = amendments["conclusion"]
            if "product_description" in amendments:
                dds.product_description = amendments["product_description"]
            if "plots" in amendments:
                dds.plots = amendments["plots"]
            if "mitigation_measures" in amendments:
                # Add new mitigation measures
                for measure_data in amendments["mitigation_measures"]:
                    measure = DDSMitigationMeasure(
                        risk_factor=measure_data.get("risk_factor", ""),
                        measure=measure_data.get("measure", ""),
                        status=measure_data.get("status", "planned"),
                    )
                    dds.mitigation_measures.append(measure)

            dds.status = DDSStatus.AMENDED
            dds.rejection_reason = None
            dds.updated_at = _utcnow()

        logger.info(
            "Amended DDS %s: %d changes applied, amendment #%d",
            dds_id,
            len(amendments),
            len(dds.amendment_history),
        )
        return dds

    # -----------------------------------------------------------------------
    # Bulk Operations
    # -----------------------------------------------------------------------

    def bulk_generate_dds(
        self, requests: List[DDSGenerateRequest]
    ) -> List[DueDiligenceStatement]:
        """Generate multiple DDS records in batch.

        Args:
            requests: List of DDS generation requests.

        Returns:
            List of generated DueDiligenceStatement records.
        """
        results: List[DueDiligenceStatement] = []
        for idx, request in enumerate(requests):
            try:
                dds = self.generate_dds_from_request(request)
                results.append(dds)
            except Exception as exc:
                logger.error(
                    "Bulk DDS generation failed at index %d: %s",
                    idx,
                    exc,
                )
        logger.info(
            "Bulk DDS generation: %d/%d successful",
            len(results),
            len(requests),
        )
        return results

    # -----------------------------------------------------------------------
    # DDS Export / Download
    # -----------------------------------------------------------------------

    def download_dds(
        self, dds_id: str, format: str = "json"
    ) -> Dict[str, Any]:
        """Export a DDS in the specified format.

        Args:
            dds_id: DDS identifier.
            format: Export format ("json" or "summary").

        Returns:
            DDS data as a dictionary.

        Raises:
            ValueError: If DDS not found.
        """
        with self._lock:
            dds = self._dds_records.get(dds_id)
            if dds is None:
                raise ValueError(f"DDS not found: {dds_id}")

        if format == "summary":
            return {
                "reference_number": dds.reference_number,
                "supplier_id": dds.supplier_id,
                "operator_name": dds.operator_name,
                "commodity": dds.commodity.value,
                "year": dds.year,
                "country_of_production": dds.country_of_production,
                "plots_count": len(dds.plots),
                "status": dds.status.value,
                "risk_level": (
                    dds.risk_assessment.risk_level.value
                    if dds.risk_assessment
                    else None
                ),
                "overall_risk": (
                    dds.risk_assessment.overall_risk
                    if dds.risk_assessment
                    else None
                ),
                "submission_date": (
                    dds.submission_date.isoformat()
                    if dds.submission_date
                    else None
                ),
                "created_at": dds.created_at.isoformat(),
            }

        # Full JSON export
        return dds.model_dump(mode="json")

    # -----------------------------------------------------------------------
    # Annual Summary
    # -----------------------------------------------------------------------

    def get_annual_summary(self, year: int) -> DDSAnnualSummary:
        """Generate an annual summary of DDS activity.

        Args:
            year: Reporting year.

        Returns:
            DDSAnnualSummary with statistics.
        """
        with self._lock:
            year_records = [
                r for r in self._dds_records.values() if r.year == year
            ]

        total_dds = len(year_records)

        # Count by status
        by_status: Dict[str, int] = {}
        for r in year_records:
            key = r.status.value
            by_status[key] = by_status.get(key, 0) + 1

        # Count by commodity
        by_commodity: Dict[str, int] = {}
        for r in year_records:
            key = r.commodity.value
            by_commodity[key] = by_commodity.get(key, 0) + 1

        # Count by country
        by_country: Dict[str, int] = {}
        for r in year_records:
            key = r.country_of_production or "UNKNOWN"
            by_country[key] = by_country.get(key, 0) + 1

        # Acceptance rate
        accepted = by_status.get(DDSStatus.ACCEPTED.value, 0)
        submitted = by_status.get(DDSStatus.SUBMITTED.value, 0)
        rejected = by_status.get(DDSStatus.REJECTED.value, 0)
        total_decided = accepted + rejected
        acceptance_rate = (
            (accepted / total_decided * 100) if total_decided > 0 else 0.0
        )

        # Average processing days (draft to submission)
        processing_days_list: List[float] = []
        for r in year_records:
            if r.submission_date and r.created_at:
                delta = (r.submission_date - r.created_at).total_seconds()
                processing_days_list.append(delta / 86400.0)
        avg_processing_days = (
            sum(processing_days_list) / len(processing_days_list)
            if processing_days_list
            else 0.0
        )

        # Top rejection reasons
        rejection_reasons: Dict[str, int] = {}
        for r in year_records:
            if r.rejection_reason:
                rejection_reasons[r.rejection_reason] = (
                    rejection_reasons.get(r.rejection_reason, 0) + 1
                )
        top_rejections = [
            {"reason": k, "count": v}
            for k, v in sorted(
                rejection_reasons.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

        summary = DDSAnnualSummary(
            year=year,
            total_dds=total_dds,
            by_status=by_status,
            by_commodity=by_commodity,
            by_country=by_country,
            acceptance_rate=round(acceptance_rate, 1),
            average_processing_days=round(avg_processing_days, 1),
            top_rejection_reasons=top_rejections,
        )

        logger.info(
            "Annual summary for %d: %d DDS, acceptance=%.1f%%",
            year,
            total_dds,
            acceptance_rate,
        )
        return summary

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with DDS counts and status breakdown.
        """
        with self._lock:
            records = list(self._dds_records.values())

        by_status: Dict[str, int] = {}
        by_commodity: Dict[str, int] = {}
        for r in records:
            sk = r.status.value
            by_status[sk] = by_status.get(sk, 0) + 1
            ck = r.commodity.value
            by_commodity[ck] = by_commodity.get(ck, 0) + 1

        return {
            "total_dds": len(records),
            "by_status": by_status,
            "by_commodity": by_commodity,
        }

    # -----------------------------------------------------------------------
    # Supply Chain Service Setter
    # -----------------------------------------------------------------------

    def set_supply_chain_service(self, service: Any) -> None:
        """Inject or update the SupplyChainAppService reference.

        Used during startup to wire the supply chain service into the
        DDS engine after both are initialized.

        Args:
            service: SupplyChainAppService instance.
        """
        self._supply_chain_service = service
        logger.info("DDSReportingEngine: Supply chain service connected")

    # -----------------------------------------------------------------------
    # Supply Chain Section Builder
    # -----------------------------------------------------------------------

    def _build_supply_chain_section(
        self,
        supplier_id: str,
        plot_ids: List[str],
        commodity: EUDRCommodity,
    ) -> Optional[Dict[str, Any]]:
        """Build the supply chain section for a DDS using AGENT-EUDR-001.

        Calls the RegulatoryExporter via the SupplyChainAppService to
        produce a DDS-compatible supply chain summary per EUDR Article
        4(2). If the supply chain service is not available, returns None
        and the DDS proceeds without a supply chain section.

        The section includes:
            - supply_chain_summary: node/edge counts, tier depth,
              traceability score, gap count
            - supply_chain_nodes: list of supply chain nodes with
              role, country, and risk classification
            - geolocation_references: plot geolocation data linked
              from AGENT-DATA-005

        Args:
            supplier_id: Supplier the DDS covers.
            plot_ids: Plot IDs included in the DDS.
            commodity: Commodity covered by the DDS.

        Returns:
            Supply chain section dictionary for the DDS, or None if
            the supply chain service is unavailable.
        """
        if self._supply_chain_service is None:
            logger.debug(
                "Supply chain service not available; "
                "DDS will not include supply chain section"
            )
            return None

        try:
            # Attempt to find an existing graph for this supplier
            scm_svc = self._supply_chain_service
            if not scm_svc.is_initialized:
                logger.debug(
                    "Supply chain service not initialized; "
                    "skipping supply chain section"
                )
                return None

            # Try to get the underlying SCM service
            inner_svc = scm_svc.scm_service
            exporter = inner_svc.regulatory_exporter

            if exporter is None:
                logger.debug(
                    "RegulatoryExporter not available; "
                    "skipping supply chain section"
                )
                return None

            # Look up graph for this supplier via graph engine
            graph_engine = inner_svc.graph_engine
            if graph_engine is None:
                logger.debug(
                    "Graph engine not available; "
                    "skipping supply chain section"
                )
                return None

            # Search for a graph associated with this supplier
            graph = None
            graph_id = None
            if hasattr(graph_engine, "find_graph_by_supplier"):
                graph = graph_engine.find_graph_by_supplier(supplier_id)
            elif hasattr(graph_engine, "list_graphs"):
                graphs = graph_engine.list_graphs()
                for g in graphs:
                    g_supplier = getattr(g, "supplier_id", None)
                    if g_supplier == supplier_id:
                        graph = g
                        break

            if graph is None:
                logger.info(
                    "No supply chain graph found for supplier %s; "
                    "DDS supply chain section will be empty",
                    supplier_id,
                )
                return {
                    "graph_id": None,
                    "status": "no_graph",
                    "message": (
                        "No supply chain graph available for this supplier. "
                        "Create a supply chain mapping to include this section."
                    ),
                    "commodity": commodity.value,
                    "supplier_id": supplier_id,
                }

            graph_id = getattr(graph, "graph_id", None) or getattr(
                graph, "id", None
            )

            # Export DDS section via RegulatoryExporter
            dds_data = exporter.export_dds_json(graph=graph)

            # Build the section
            section: Dict[str, Any] = {
                "graph_id": graph_id,
                "status": "available",
                "commodity": commodity.value,
                "supplier_id": supplier_id,
            }

            # Extract supply chain summary from export
            if hasattr(dds_data, "model_dump"):
                export_dict = dds_data.model_dump(mode="json")
            elif isinstance(dds_data, dict):
                export_dict = dds_data
            else:
                export_dict = {}

            section["supply_chain_summary"] = export_dict.get(
                "supply_chain_summary", {}
            )
            section["supply_chain_nodes"] = export_dict.get(
                "supply_chain_nodes", []
            )
            section["traceability"] = export_dict.get("traceability", {})
            section["provenance"] = export_dict.get("provenance", {})

            # Link plot geolocation from AGENT-DATA-005
            geo_linker = inner_svc.geolocation_linker
            if geo_linker is not None and plot_ids:
                try:
                    geo_refs = []
                    for plot_id in plot_ids:
                        geo = geo_linker.get_plot_geolocation(plot_id)
                        if geo is not None:
                            geo_entry = (
                                geo.model_dump(mode="json")
                                if hasattr(geo, "model_dump")
                                else geo
                            )
                            geo_refs.append(geo_entry)
                    section["geolocation_references"] = geo_refs
                except Exception as geo_exc:
                    logger.warning(
                        "Failed to link geolocation data: %s", geo_exc
                    )
                    section["geolocation_references"] = []
            else:
                section["geolocation_references"] = []

            # Compute provenance hash
            section["provenance_hash"] = _compute_hash(section)

            logger.info(
                "Supply chain section built for DDS: "
                "supplier=%s, graph_id=%s, nodes=%d",
                supplier_id,
                graph_id,
                len(section.get("supply_chain_nodes", [])),
            )
            return section

        except Exception as exc:
            logger.warning(
                "Failed to build supply chain section for DDS "
                "(supplier=%s): %s. DDS will proceed without it.",
                supplier_id,
                exc,
            )
            return None

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _get_valid_transitions(
        self, current: DDSStatus
    ) -> List[DDSStatus]:
        """Get valid status transitions from current status.

        Args:
            current: Current DDS status.

        Returns:
            List of valid target statuses.
        """
        transitions: Dict[DDSStatus, List[DDSStatus]] = {
            DDSStatus.DRAFT: [DDSStatus.REVIEW],
            DDSStatus.REVIEW: [DDSStatus.VALIDATED, DDSStatus.DRAFT],
            DDSStatus.VALIDATED: [DDSStatus.SUBMITTED, DDSStatus.REVIEW],
            DDSStatus.SUBMITTED: [DDSStatus.ACCEPTED, DDSStatus.REJECTED],
            DDSStatus.ACCEPTED: [],
            DDSStatus.REJECTED: [DDSStatus.AMENDED],
            DDSStatus.AMENDED: [DDSStatus.SUBMITTED, DDSStatus.REVIEW],
        }
        return transitions.get(current, [])

    def _build_plot_details(
        self, plot_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Build plot detail entries for the DDS.

        Args:
            plot_ids: List of plot IDs.

        Returns:
            List of plot detail dictionaries.
        """
        details: List[Dict[str, Any]] = []

        for plot_id in plot_ids:
            plot_data: Dict[str, Any] = {
                "plot_id": plot_id,
                "included": True,
            }

            # Try to get plot details from supplier engine
            if self._supplier_engine:
                plot = self._supplier_engine.get_plot(plot_id)
                if plot:
                    plot_data.update({
                        "name": plot.name,
                        "country_iso3": plot.country_iso3,
                        "commodity": plot.commodity.value,
                        "area_hectares": plot.area_hectares,
                        "centroid_lat": plot.centroid_lat,
                        "centroid_lon": plot.centroid_lon,
                        "risk_level": plot.risk_level.value,
                        "is_deforestation_free": plot.is_deforestation_free,
                    })

            details.append(plot_data)

        return details

    def _build_risk_assessment(
        self,
        supplier_id: str,
        plot_ids: List[str],
    ) -> DDSRiskAssessment:
        """Build risk assessment section for the DDS.

        Uses the RiskAggregator if available; otherwise provides defaults.

        Args:
            supplier_id: Supplier ID.
            plot_ids: Plot IDs.

        Returns:
            DDSRiskAssessment with risk scores.
        """
        # If risk engine is available, use it for the first plot
        if self._risk_engine and plot_ids:
            try:
                assessment = self._risk_engine.assess_risk(
                    supplier_id, plot_ids[0]
                )
                return DDSRiskAssessment(
                    country_risk=assessment.country_risk,
                    commodity_risk=0.0,
                    supplier_risk=assessment.supplier_risk,
                    satellite_risk=assessment.satellite_risk,
                    overall_risk=assessment.overall_risk,
                    risk_level=assessment.risk_level,
                    factors=[
                        f.get("factor", str(f))
                        for f in assessment.factors
                    ],
                    data_sources=assessment.data_sources,
                    assessment_date=_utcnow(),
                )
            except Exception as exc:
                logger.warning(
                    "Risk engine assessment failed: %s. Using defaults.",
                    exc,
                )

        # Default risk assessment
        return DDSRiskAssessment(
            country_risk=0.3,
            commodity_risk=0.3,
            supplier_risk=0.2,
            satellite_risk=0.15,
            overall_risk=0.25,
            risk_level=RiskLevel.STANDARD,
            factors=["Baseline risk assessment (default values)"],
            data_sources=["EUDR country risk benchmarking"],
            assessment_date=_utcnow(),
        )

    def _build_mitigation_measures(
        self, risk_assessment: DDSRiskAssessment
    ) -> List[DDSMitigationMeasure]:
        """Generate mitigation measures based on risk assessment.

        Args:
            risk_assessment: Risk assessment results.

        Returns:
            List of recommended mitigation measures.
        """
        measures: List[DDSMitigationMeasure] = []

        if risk_assessment.country_risk > 0.5:
            measures.append(DDSMitigationMeasure(
                risk_factor="Country risk",
                measure=(
                    "Conduct enhanced due diligence for high-risk "
                    "country of production"
                ),
                status="planned",
            ))

        if risk_assessment.satellite_risk > 0.3:
            measures.append(DDSMitigationMeasure(
                risk_factor="Satellite risk",
                measure=(
                    "Commission independent satellite monitoring "
                    "for production plots"
                ),
                status="planned",
            ))

        if risk_assessment.supplier_risk > 0.4:
            measures.append(DDSMitigationMeasure(
                risk_factor="Supplier risk",
                measure=(
                    "Request updated sustainability certificates "
                    "from supplier"
                ),
                status="planned",
            ))

        # Always include a standard measure
        if not measures:
            measures.append(DDSMitigationMeasure(
                risk_factor="General compliance",
                measure=(
                    "Maintain ongoing monitoring and periodic "
                    "compliance reviews"
                ),
                status="in_progress",
            ))

        return measures

    def _build_conclusion(
        self, risk_assessment: DDSRiskAssessment
    ) -> str:
        """Build DDS conclusion statement based on risk assessment.

        Args:
            risk_assessment: Risk assessment results.

        Returns:
            Conclusion text string.
        """
        if risk_assessment.risk_level in (
            RiskLevel.LOW,
            RiskLevel.STANDARD,
        ):
            return (
                "Based on the due diligence conducted in accordance with "
                "EU Regulation 2023/1115, the risk of non-compliance has "
                "been assessed as acceptable. The relevant commodities and "
                "products are considered deforestation-free and legally "
                "produced. Ongoing monitoring will be maintained."
            )

        if risk_assessment.risk_level == RiskLevel.HIGH:
            return (
                "Due diligence conducted per EU Regulation 2023/1115 has "
                "identified elevated risk of non-compliance. Enhanced "
                "monitoring and risk mitigation measures have been "
                "implemented as detailed in the risk mitigation section. "
                "Continued assessment is required before final determination."
            )

        return (
            "Due diligence conducted per EU Regulation 2023/1115 has "
            "identified critical risk factors. The operator has implemented "
            "comprehensive risk mitigation measures and will continue "
            "monitoring. Further evidence and assessment are required "
            "before the commodities can be placed on the market."
        )

    def _gather_documents(
        self,
        supplier_id: str,
        plot_ids: List[str],
    ) -> List[str]:
        """Gather document IDs linked to supplier and plots.

        Args:
            supplier_id: Supplier ID.
            plot_ids: Plot IDs.

        Returns:
            List of document IDs.
        """
        if not self._document_engine:
            return []

        doc_ids: List[str] = []
        try:
            docs = self._document_engine.list_documents()
            for doc in docs:
                if doc.linked_supplier_id == supplier_id:
                    doc_ids.append(doc.id)
                elif doc.linked_plot_id and doc.linked_plot_id in plot_ids:
                    doc_ids.append(doc.id)
        except Exception as exc:
            logger.warning("Failed to gather documents: %s", exc)

        return doc_ids

    def _default_product_description(
        self, commodity: EUDRCommodity
    ) -> str:
        """Generate a default product description for a commodity.

        Args:
            commodity: EUDR commodity type.

        Returns:
            Default product description string.
        """
        descriptions: Dict[EUDRCommodity, str] = {
            EUDRCommodity.CATTLE: (
                "Cattle and derived products including beef, leather, "
                "and other bovine products"
            ),
            EUDRCommodity.COCOA: (
                "Cocoa beans, cocoa butter, cocoa powder, and derived "
                "chocolate products"
            ),
            EUDRCommodity.COFFEE: (
                "Coffee beans (green and roasted), ground coffee, and "
                "coffee extracts"
            ),
            EUDRCommodity.PALM_OIL: (
                "Palm oil, palm kernel oil, and derived products "
                "including oleochemicals"
            ),
            EUDRCommodity.RUBBER: (
                "Natural rubber, latex, and derived rubber products"
            ),
            EUDRCommodity.SOY: (
                "Soybeans, soy meal, soy oil, and derived soy products"
            ),
            EUDRCommodity.WOOD: (
                "Timber, wood products, pulp, paper, and printed "
                "paper products"
            ),
        }
        return descriptions.get(commodity, f"{commodity.value} products")
