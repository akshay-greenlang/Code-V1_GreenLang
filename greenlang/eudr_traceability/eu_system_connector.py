# -*- coding: utf-8 -*-
"""
EU Information System Connector Engine - AGENT-DATA-004: EUDR Traceability

Provides integration with the EU Information System for submitting
Due Diligence Statements per EUDR Article 12. Supports:

- DDS submission preparation and validation
- EU system format transformation
- Submission workflow (prepare -> validate -> submit -> confirm)
- Sandbox and production environment support
- Bulk DDS submission
- Retry handling for failed submissions
- Submission status tracking and audit logging

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EUSystemConnector:
    """EU Information System integration engine.

    Handles the full lifecycle of submitting Due Diligence Statements
    to the EU Information System as required by EUDR Article 12.

    Supports sandbox (simulated) and production modes. In sandbox mode,
    submissions are processed locally with simulated EU system responses.

    Attributes:
        config: EUDRTraceabilityConfig instance.
        _submissions: In-memory submission record storage.
        _submission_index: Index by DDS ID for quick lookup.
    """

    # Maximum retry attempts for failed submissions
    MAX_RETRIES = 3

    # Required fields for EU system submission
    REQUIRED_EU_FIELDS = {
        "operator_id", "operator_name", "operator_country",
        "commodity", "product_description", "cn_codes",
        "quantity", "origin_countries", "origin_plots",
        "deforestation_free_declaration", "legal_compliance_declaration",
    }

    def __init__(self, config: Optional[Any] = None):
        """Initialize EU System Connector.

        Args:
            config: EUDRTraceabilityConfig instance. If None, uses defaults.
        """
        if config is None:
            from greenlang.eudr_traceability.config import get_config
            config = get_config()
        self.config = config

        # In-memory storage
        self._submissions: Dict[str, Dict[str, Any]] = {}
        self._submission_index_by_dds: Dict[str, List[str]] = {}

        logger.info(
            "EUSystemConnector initialized: sandbox=%s, url=%s",
            getattr(config, 'eu_system_sandbox', True),
            getattr(config, 'eu_system_url', ''),
        )

    # =========================================================================
    # Submission Management
    # =========================================================================

    def prepare_submission(
        self,
        dds_id: str,
        dds_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare a DDS for submission to the EU Information System.

        Validates the DDS data, formats it for the EU system, and creates
        a submission record for tracking.

        Args:
            dds_id: Due Diligence Statement ID.
            dds_data: DDS data dictionary with all required fields.

        Returns:
            EUSubmissionRecord as dictionary.

        Raises:
            ValueError: If DDS data is invalid or incomplete.
        """
        # Validate EU format
        validation_errors = self.validate_eu_format(dds_data)
        if validation_errors:
            raise ValueError(
                f"DDS data validation failed: {'; '.join(validation_errors)}"
            )

        submission_id = self._generate_submission_id()
        now = datetime.utcnow()

        # Format for EU system
        eu_payload = self.format_for_eu_system(dds_data)

        record = {
            "submission_id": submission_id,
            "dds_id": dds_id,
            "submission_status": "pending",
            "eu_reference": None,
            "submitted_at": now.isoformat(),
            "response_at": None,
            "error_message": None,
            "retry_count": 0,
            "request_payload": eu_payload,
            "response_payload": None,
            "created_at": now.isoformat(),
            "metadata": {},
        }

        # Store
        self._submissions[submission_id] = record
        if dds_id not in self._submission_index_by_dds:
            self._submission_index_by_dds[dds_id] = []
        self._submission_index_by_dds[dds_id].append(submission_id)

        # Track metrics
        try:
            from greenlang.eudr_traceability.metrics import record_eu_submission
            record_eu_submission(status="pending")
        except (ImportError, Exception):
            pass

        logger.info("Prepared submission %s for DDS %s", submission_id, dds_id)
        return record

    def submit_to_eu(self, submission_id: str) -> Dict[str, Any]:
        """Submit a prepared DDS to the EU Information System.

        In sandbox mode, simulates the EU system response.
        In production mode, would make HTTP call to EU system API.

        Args:
            submission_id: Submission record ID.

        Returns:
            Updated EUSubmissionRecord as dictionary.

        Raises:
            ValueError: If submission not found or already submitted.
        """
        record = self._submissions.get(submission_id)
        if not record:
            raise ValueError(f"Submission not found: {submission_id}")

        if record["submission_status"] == "accepted":
            raise ValueError(f"Submission already accepted: {submission_id}")

        now = datetime.utcnow()
        is_sandbox = getattr(self.config, 'eu_system_sandbox', True)

        if is_sandbox:
            # Simulated EU system response
            response = self._simulate_eu_response(record)
        else:
            # Production EU system call (placeholder)
            response = self._call_eu_system(record)

        record["submitted_at"] = now.isoformat()
        record["response_at"] = now.isoformat()
        record["response_payload"] = response
        record["submission_status"] = response.get("status", "error")
        record["eu_reference"] = response.get("eu_reference")
        record["error_message"] = response.get("error_message")

        # Track metrics
        try:
            from greenlang.eudr_traceability.metrics import record_eu_submission
            record_eu_submission(status=record["submission_status"])
        except (ImportError, Exception):
            pass

        logger.info(
            "Submitted %s: status=%s, ref=%s",
            submission_id,
            record["submission_status"],
            record["eu_reference"],
        )
        return record

    def get_submission_status(
        self,
        submission_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get submission status by ID.

        Args:
            submission_id: Submission record ID.

        Returns:
            EUSubmissionRecord or None if not found.
        """
        return self._submissions.get(submission_id)

    def list_submissions(
        self,
        dds_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List submission records with optional filters.

        Args:
            dds_id: Filter by DDS ID.
            status: Filter by submission status.
            limit: Maximum results to return.
            offset: Number of results to skip.

        Returns:
            List of EUSubmissionRecord dictionaries.
        """
        if dds_id:
            sub_ids = self._submission_index_by_dds.get(dds_id, [])
            results = [self._submissions[sid] for sid in sub_ids
                       if sid in self._submissions]
        else:
            results = list(self._submissions.values())

        if status:
            results = [r for r in results if r["submission_status"] == status]

        # Sort by submitted_at descending
        results.sort(key=lambda r: r.get("submitted_at", ""), reverse=True)
        return results[offset:offset + limit]

    def retry_submission(self, submission_id: str) -> Dict[str, Any]:
        """Retry a failed submission.

        Increments retry count and resubmits. Fails if max retries exceeded.

        Args:
            submission_id: Submission record ID.

        Returns:
            Updated EUSubmissionRecord.

        Raises:
            ValueError: If submission not found or max retries exceeded.
        """
        record = self._submissions.get(submission_id)
        if not record:
            raise ValueError(f"Submission not found: {submission_id}")

        if record["retry_count"] >= self.MAX_RETRIES:
            record["submission_status"] = "error"
            record["error_message"] = (
                f"Max retries ({self.MAX_RETRIES}) exceeded"
            )
            raise ValueError(
                f"Max retries exceeded for submission: {submission_id}"
            )

        record["retry_count"] += 1
        record["submission_status"] = "pending"
        record["error_message"] = None

        logger.info(
            "Retrying submission %s (attempt %d/%d)",
            submission_id,
            record["retry_count"],
            self.MAX_RETRIES,
        )
        return self.submit_to_eu(submission_id)

    def bulk_submit(
        self,
        dds_submissions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Submit multiple DDS in bulk.

        Args:
            dds_submissions: List of {dds_id, dds_data} dictionaries.

        Returns:
            List of EUSubmissionRecord dictionaries.
        """
        results = []
        for item in dds_submissions:
            dds_id = item.get("dds_id", "")
            dds_data = item.get("dds_data", {})
            try:
                record = self.prepare_submission(dds_id, dds_data)
                record = self.submit_to_eu(record["submission_id"])
                results.append(record)
            except (ValueError, Exception) as e:
                results.append({
                    "dds_id": dds_id,
                    "submission_status": "error",
                    "error_message": str(e),
                })
        return results

    # =========================================================================
    # Format & Validation
    # =========================================================================

    def format_for_eu_system(
        self,
        dds_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Transform DDS data into EU Information System API format.

        Args:
            dds_data: Raw DDS data dictionary.

        Returns:
            Formatted data for EU system API.
        """
        return {
            "eudr_version": "1.0",
            "submission_type": "due_diligence_statement",
            "operator": {
                "id": dds_data.get("operator_id", ""),
                "name": dds_data.get("operator_name", ""),
                "country": dds_data.get("operator_country", ""),
                "eori_number": dds_data.get("operator_eori"),
            },
            "product": {
                "commodity": dds_data.get("commodity", ""),
                "description": dds_data.get("product_description", ""),
                "cn_codes": dds_data.get("cn_codes", []),
                "quantity_kg": str(dds_data.get("quantity", "0")),
                "unit": dds_data.get("unit", "kg"),
            },
            "traceability": {
                "origin_countries": dds_data.get("origin_countries", []),
                "production_plots": [
                    {"plot_id": pid}
                    for pid in dds_data.get("origin_plot_ids", [])
                ],
            },
            "declarations": {
                "deforestation_free": dds_data.get(
                    "deforestation_free_declaration", False
                ),
                "legal_compliance": dds_data.get(
                    "legal_compliance_declaration", False
                ),
                "due_diligence_performed": True,
            },
            "risk_assessment": {
                "level": dds_data.get("risk_level", "standard"),
                "mitigation_measures": dds_data.get(
                    "risk_mitigation_measures", []
                ),
            },
            "metadata": {
                "submission_timestamp": datetime.utcnow().isoformat(),
                "system": "GreenLang Climate OS",
                "agent_id": "GL-DATA-EUDR-001",
                "content_hash": self._compute_content_hash(dds_data),
            },
        }

    def validate_eu_format(
        self,
        dds_data: Dict[str, Any],
    ) -> List[str]:
        """Validate DDS data for EU system format compliance.

        Args:
            dds_data: DDS data dictionary to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        for field in self.REQUIRED_EU_FIELDS:
            value = dds_data.get(field)
            if value is None or value == "" or value == []:
                errors.append(f"Missing required field: {field}")

        # Validate operator country is ISO 3166-1 alpha-2
        country = dds_data.get("operator_country", "")
        if country and len(country) != 2:
            errors.append(
                f"Invalid operator_country: must be 2-letter ISO code, "
                f"got '{country}'"
            )

        # Validate CN codes format
        cn_codes = dds_data.get("cn_codes", [])
        for cn in cn_codes:
            if not isinstance(cn, str) or len(cn) < 4:
                errors.append(f"Invalid CN code format: {cn}")

        # Validate quantity is positive
        try:
            qty = Decimal(str(dds_data.get("quantity", 0)))
            if qty <= 0:
                errors.append("Quantity must be positive")
        except (ValueError, TypeError):
            errors.append("Invalid quantity value")

        # Validate origin plots exist
        plots = dds_data.get("origin_plot_ids", [])
        if not plots:
            errors.append("At least one origin plot is required")

        return errors

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _simulate_eu_response(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate an EU Information System response for sandbox mode.

        Args:
            record: Submission record.

        Returns:
            Simulated EU system response dictionary.
        """
        eu_ref = f"EU-{uuid.uuid4().hex[:16].upper()}"
        return {
            "status": "accepted",
            "eu_reference": eu_ref,
            "message": "Due diligence statement accepted (sandbox mode)",
            "timestamp": datetime.utcnow().isoformat(),
            "error_message": None,
        }

    def _call_eu_system(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Make HTTP call to EU Information System API.

        This is a placeholder for production integration.
        Actual implementation would use httpx or aiohttp.

        Args:
            record: Submission record with request_payload.

        Returns:
            EU system response dictionary.
        """
        # Production implementation would go here
        logger.warning(
            "Production EU system integration not yet implemented. "
            "Using sandbox simulation."
        )
        return self._simulate_eu_response(record)

    def _compute_content_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of DDS content for integrity.

        Args:
            data: DDS data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Select key fields for hashing
        hash_data = {
            "operator_id": data.get("operator_id", ""),
            "commodity": data.get("commodity", ""),
            "cn_codes": sorted(data.get("cn_codes", [])),
            "quantity": str(data.get("quantity", "0")),
            "origin_plot_ids": sorted(data.get("origin_plot_ids", [])),
            "deforestation_free": data.get(
                "deforestation_free_declaration", False
            ),
            "legal_compliance": data.get(
                "legal_compliance_declaration", False
            ),
        }
        content = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _generate_submission_id(self) -> str:
        """Generate unique submission ID.

        Returns:
            Submission ID in format SUB-XXXXXXXXXXXX.
        """
        return f"SUB-{uuid.uuid4().hex[:12].upper()}"

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get submission statistics.

        Returns:
            Dictionary with submission counts by status.
        """
        status_counts: Dict[str, int] = {}
        for record in self._submissions.values():
            status = record["submission_status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_submissions": len(self._submissions),
            "by_status": status_counts,
            "unique_dds": len(self._submission_index_by_dds),
        }
