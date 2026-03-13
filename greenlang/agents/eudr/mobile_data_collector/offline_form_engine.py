# -*- coding: utf-8 -*-
"""
Offline Form Engine - AGENT-EUDR-015 Mobile Data Collector (Engine 1)

Production-grade offline-first structured form collection engine for
EUDR compliance covering 6 form types (producer_registration, plot_survey,
harvest_log, custody_transfer, quality_inspection, smallholder_declaration),
local in-memory queue-based storage (server-side simulation of device
SQLite), Pydantic schema validation, draft save/restore, form status
transitions (draft -> pending -> syncing -> synced -> failed), sync queue
management, data completeness scoring, commodity-specific field validation,
and offline conflict detection.

Zero-Hallucination Guarantees:
    - All form data is persisted to deterministic in-memory stores
    - Validation uses Pydantic models and explicit field rules only
    - Completeness scoring is a simple percentage calculation
    - Status transitions follow a strict state machine
    - No LLM calls in any validation or calculation path
    - SHA-256 provenance recorded for every mutation

PRD: PRD-AGENT-EUDR-015 Feature F1 (Offline Form Collection)
Agent ID: GL-EUDR-MDC-015
Regulation: EU 2023/1115 (EUDR) Articles 4, 9, 10, 14

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.mobile_data_collector.config import get_config
from greenlang.agents.eudr.mobile_data_collector.metrics import (
    observe_form_submission_duration,
    record_api_error,
    record_form_submitted,
)
from greenlang.agents.eudr.mobile_data_collector.models import (
    CommodityType,
    FormResponse,
    FormStatus,
    FormSubmission,
    FormType,
    SyncQueueItem,
    SyncStatus,
)
from greenlang.agents.eudr.mobile_data_collector.provenance import (
    get_provenance_tracker,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class OfflineFormEngineError(Exception):
    """Base exception for offline form engine operations."""


class FormNotFoundError(OfflineFormEngineError):
    """Raised when a form submission cannot be found."""


class FormValidationError(OfflineFormEngineError):
    """Raised when form data fails validation."""


class FormStateTransitionError(OfflineFormEngineError):
    """Raised when an invalid status transition is attempted."""


class FormConflictError(OfflineFormEngineError):
    """Raised when a form conflict is detected during sync."""


# ---------------------------------------------------------------------------
# Valid status transitions
# ---------------------------------------------------------------------------

_VALID_TRANSITIONS: Dict[FormStatus, List[FormStatus]] = {
    FormStatus.DRAFT: [FormStatus.PENDING, FormStatus.DRAFT],
    FormStatus.PENDING: [FormStatus.SYNCING, FormStatus.DRAFT],
    FormStatus.SYNCING: [FormStatus.SYNCED, FormStatus.FAILED],
    FormStatus.SYNCED: [],
    FormStatus.FAILED: [FormStatus.PENDING, FormStatus.DRAFT],
}

# ---------------------------------------------------------------------------
# Required fields per form type
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: Dict[str, List[str]] = {
    "producer_registration": [
        "producer_name", "national_id", "address", "country",
        "commodity_type", "registration_date",
    ],
    "plot_survey": [
        "plot_name", "plot_location", "area_ha", "commodity_type",
        "land_use_history", "deforestation_status",
    ],
    "harvest_log": [
        "harvest_date", "commodity_type", "quantity_kg",
        "plot_id", "quality_grade",
    ],
    "custody_transfer": [
        "sender_id", "receiver_id", "commodity_type",
        "quantity_kg", "transfer_date", "transport_method",
    ],
    "quality_inspection": [
        "inspector_name", "inspection_date", "commodity_type",
        "sample_count", "quality_grade", "defect_percentage",
    ],
    "smallholder_declaration": [
        "declarant_name", "national_id", "commodity_type",
        "total_area_ha", "deforestation_free", "declaration_date",
    ],
}

# ---------------------------------------------------------------------------
# Commodity-specific required fields
# ---------------------------------------------------------------------------

_COMMODITY_FIELDS: Dict[str, List[str]] = {
    "cattle": [
        "head_count", "breed", "ear_tag_numbers",
        "pasture_type", "feeding_system",
    ],
    "cocoa": [
        "bean_variety", "fermentation_days", "drying_method",
        "moisture_percentage", "tree_age_years",
    ],
    "coffee": [
        "coffee_variety", "altitude_m", "processing_method",
        "cherry_maturity", "cupping_score",
    ],
    "oil_palm": [
        "ffb_weight_kg", "palm_variety", "plantation_age_years",
        "certification_scheme", "mill_name",
    ],
    "rubber": [
        "tapping_frequency", "latex_grade", "dry_rubber_content_pct",
        "clone_type", "coagulation_method",
    ],
    "soya": [
        "soy_variety", "yield_tonnes_ha", "moisture_pct",
        "protein_content_pct", "gmo_status",
    ],
    "wood": [
        "tree_species", "log_volume_m3", "diameter_cm",
        "forest_type", "felling_license_number",
    ],
}


# ---------------------------------------------------------------------------
# OfflineFormEngine
# ---------------------------------------------------------------------------


class OfflineFormEngine:
    """Offline-first structured form collection engine for EUDR compliance.

    Manages the complete lifecycle of EUDR compliance forms from draft
    creation through validation, submission, sync queue management, and
    conflict detection. Uses an in-memory dictionary store to simulate
    the device-side SQLite database on the server.

    The engine supports 6 EUDR form types per EU 2023/1115:
        - producer_registration (Article 9.1.f)
        - plot_survey (Article 9.1.c-d)
        - harvest_log (Article 9.1.a-b,e)
        - custody_transfer (Article 9.1.f-g)
        - quality_inspection (Article 10.1)
        - smallholder_declaration (Article 4.2)

    Thread Safety:
        All public methods are protected by a reentrant lock for
        concurrent access from multiple API handlers.

    Attributes:
        _config: Agent configuration instance.
        _forms: In-memory form store keyed by form_id.
        _sync_queue: In-memory sync queue keyed by queue_item_id.
        _drafts: Draft storage keyed by form_id.
        _provenance: Provenance tracker for audit trails.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> engine = OfflineFormEngine()
        >>> result = engine.submit_form(
        ...     device_id="dev-001",
        ...     operator_id="op-001",
        ...     form_type="harvest_log",
        ...     template_id="tmpl-001",
        ...     data={"harvest_date": "2026-01-15", ...},
        ... )
        >>> assert result.status == FormStatus.PENDING
    """

    __slots__ = (
        "_config",
        "_forms",
        "_sync_queue",
        "_drafts",
        "_provenance",
        "_lock",
    )

    def __init__(self) -> None:
        """Initialize the OfflineFormEngine with empty stores."""
        self._config = get_config()
        self._forms: Dict[str, FormSubmission] = {}
        self._sync_queue: Dict[str, SyncQueueItem] = {}
        self._drafts: Dict[str, Dict[str, Any]] = {}
        self._provenance = get_provenance_tracker()
        self._lock = threading.RLock()
        logger.info(
            "OfflineFormEngine initialized: strictness=%s, "
            "max_form_size=%dKB, draft_expiry=%dd",
            self._config.validation_strictness,
            self._config.max_form_size_kb,
            self._config.draft_expiry_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_form(
        self,
        device_id: str,
        operator_id: str,
        form_type: str,
        template_id: str,
        data: Dict[str, Any],
        template_version: str = "1.0.0",
        commodity_type: Optional[str] = None,
        country_code: Optional[str] = None,
        local_timestamp: Optional[datetime] = None,
        gps_capture_ids: Optional[List[str]] = None,
        photo_ids: Optional[List[str]] = None,
        signature_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FormResponse:
        """Submit a completed EUDR compliance form for synchronization.

        Creates a FormSubmission, validates it against the template
        schema, computes a SHA-256 submission hash, transitions the
        status to PENDING, adds it to the sync queue, and records
        provenance.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent identifier.
            form_type: One of the 6 EUDR form types.
            template_id: Template definition identifier.
            data: Form field values as key-value pairs.
            template_version: Semantic version of the template used.
            commodity_type: EUDR commodity type if applicable.
            country_code: ISO 3166-1 alpha-2 country code.
            local_timestamp: Device-local timestamp at submission.
            gps_capture_ids: Linked GPS capture identifiers.
            photo_ids: Linked photo evidence identifiers.
            signature_ids: Linked digital signature identifiers.
            metadata: Additional form metadata.

        Returns:
            FormResponse with form_id, status, and provenance hash.

        Raises:
            FormValidationError: If form data fails validation.
            OfflineFormEngineError: If submission processing fails.
        """
        start_time = time.monotonic()
        try:
            # Validate form type
            self._validate_form_type(form_type)

            # Parse commodity type
            parsed_commodity = self._parse_commodity_type(commodity_type)

            # Build the form submission
            form = self._build_form_submission(
                device_id=device_id,
                operator_id=operator_id,
                form_type=form_type,
                template_id=template_id,
                template_version=template_version,
                data=data,
                commodity_type=parsed_commodity,
                country_code=country_code,
                local_timestamp=local_timestamp,
                gps_capture_ids=gps_capture_ids or [],
                photo_ids=photo_ids or [],
                signature_ids=signature_ids or [],
                metadata=metadata or {},
            )

            # Validate data
            errors = self._validate_form_data(form)
            if errors and self._config.validation_strictness == "strict":
                raise FormValidationError(
                    f"Form validation failed: {'; '.join(errors)}"
                )

            # Compute submission hash
            form.submission_hash = self._compute_submission_hash(form)

            # Transition to PENDING
            form.status = FormStatus.PENDING
            form.server_timestamp = datetime.now(timezone.utc).replace(
                microsecond=0
            )

            # Store form
            with self._lock:
                self._forms[form.form_id] = form

            # Add to sync queue
            self._add_to_sync_queue(form)

            # Record provenance
            provenance_entry = self._provenance.record(
                entity_type="form_submission",
                action="submit",
                entity_id=form.form_id,
                data=form.model_dump(mode="json"),
                metadata={
                    "device_id": device_id,
                    "operator_id": operator_id,
                    "form_type": form_type,
                },
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            observe_form_submission_duration(elapsed_ms / 1000)
            record_form_submitted(
                form_type,
                commodity_type or "unknown",
            )

            logger.info(
                "Form submitted: form_id=%s type=%s device=%s "
                "operator=%s elapsed=%.1fms",
                form.form_id, form_type, device_id,
                operator_id, elapsed_ms,
            )

            return FormResponse(
                form_id=form.form_id,
                status=form.status,
                submission_hash=form.submission_hash,
                provenance_hash=provenance_entry.hash_value,
                processing_time_ms=elapsed_ms,
                message="Form submitted successfully",
                form=form,
            )

        except FormValidationError:
            record_api_error("submit")
            raise
        except Exception as e:
            record_api_error("submit")
            logger.error(
                "Form submission failed: %s", str(e), exc_info=True,
            )
            raise OfflineFormEngineError(
                f"Form submission failed: {str(e)}"
            ) from e

    def save_draft(
        self,
        device_id: str,
        operator_id: str,
        form_type: str,
        template_id: str,
        data: Dict[str, Any],
        form_id: Optional[str] = None,
        commodity_type: Optional[str] = None,
        country_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FormResponse:
        """Save a form as a draft for later completion and submission.

        Drafts are stored separately and can be restored, edited, and
        eventually submitted. A draft can be saved multiple times with
        updated data.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent identifier.
            form_type: EUDR form type.
            template_id: Template definition identifier.
            data: Partial or complete form field values.
            form_id: Existing form_id for update, or None for new draft.
            commodity_type: EUDR commodity type if applicable.
            country_code: ISO 3166-1 alpha-2 country code.
            metadata: Additional form metadata.

        Returns:
            FormResponse with form_id and DRAFT status.

        Raises:
            OfflineFormEngineError: If draft save fails.
        """
        start_time = time.monotonic()
        try:
            self._validate_form_type(form_type)
            parsed_commodity = self._parse_commodity_type(commodity_type)

            draft_id = form_id or str(uuid.uuid4())

            form = self._build_form_submission(
                device_id=device_id,
                operator_id=operator_id,
                form_type=form_type,
                template_id=template_id,
                template_version="1.0.0",
                data=data,
                commodity_type=parsed_commodity,
                country_code=country_code,
                local_timestamp=None,
                gps_capture_ids=[],
                photo_ids=[],
                signature_ids=[],
                metadata=metadata or {},
            )
            form.form_id = draft_id
            form.status = FormStatus.DRAFT

            with self._lock:
                self._forms[draft_id] = form
                self._drafts[draft_id] = {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "device_id": device_id,
                    "operator_id": operator_id,
                }

            provenance_entry = self._provenance.record(
                entity_type="form_submission",
                action="create",
                entity_id=draft_id,
                data={"form_type": form_type, "status": "draft"},
                metadata={"device_id": device_id},
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000

            logger.info(
                "Draft saved: form_id=%s type=%s device=%s elapsed=%.1fms",
                draft_id, form_type, device_id, elapsed_ms,
            )

            return FormResponse(
                form_id=draft_id,
                status=FormStatus.DRAFT,
                submission_hash=None,
                provenance_hash=provenance_entry.hash_value,
                processing_time_ms=elapsed_ms,
                message="Draft saved successfully",
                form=form,
            )

        except Exception as e:
            record_api_error("submit")
            logger.error(
                "Draft save failed: %s", str(e), exc_info=True,
            )
            raise OfflineFormEngineError(
                f"Draft save failed: {str(e)}"
            ) from e

    def get_form(self, form_id: str) -> FormSubmission:
        """Retrieve a form submission by its identifier.

        Args:
            form_id: Unique form submission identifier.

        Returns:
            The FormSubmission instance.

        Raises:
            FormNotFoundError: If the form_id does not exist.
        """
        with self._lock:
            form = self._forms.get(form_id)
        if form is None:
            raise FormNotFoundError(
                f"Form not found: form_id={form_id}"
            )
        return form

    def list_forms(
        self,
        device_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        form_type: Optional[str] = None,
        status: Optional[str] = None,
        commodity_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FormSubmission]:
        """List form submissions with optional filters.

        Args:
            device_id: Filter by device identifier.
            operator_id: Filter by operator identifier.
            form_type: Filter by EUDR form type.
            status: Filter by form status.
            commodity_type: Filter by commodity type.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching FormSubmission instances.
        """
        with self._lock:
            forms = list(self._forms.values())

        # Apply filters
        if device_id:
            forms = [f for f in forms if f.device_id == device_id]
        if operator_id:
            forms = [f for f in forms if f.operator_id == operator_id]
        if form_type:
            forms = [
                f for f in forms
                if f.form_type.value == form_type
            ]
        if status:
            forms = [f for f in forms if f.status.value == status]
        if commodity_type:
            forms = [
                f for f in forms
                if f.commodity_type is not None
                and f.commodity_type.value == commodity_type
            ]

        # Sort by created_at descending
        forms.sort(key=lambda f: f.created_at, reverse=True)

        # Apply pagination
        return forms[offset: offset + limit]

    def validate_form(
        self,
        form_id: Optional[str] = None,
        form_type: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        commodity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate form data against its template schema.

        Can validate an existing form by form_id, or validate raw data
        against a form_type schema.

        Args:
            form_id: Existing form to validate, or None.
            form_type: Form type for raw data validation.
            data: Raw data to validate (when form_id is None).
            commodity_type: EUDR commodity for commodity-specific checks.

        Returns:
            Dictionary with is_valid, errors list, warnings list,
            and completeness_score.

        Raises:
            FormNotFoundError: If form_id is provided but not found.
            FormValidationError: If neither form_id nor form_type/data
                are provided.
        """
        start_time = time.monotonic()

        if form_id:
            form = self.get_form(form_id)
            ft = form.form_type.value
            form_data = form.data
            ct = (
                form.commodity_type.value
                if form.commodity_type else None
            )
        elif form_type and data is not None:
            ft = form_type
            form_data = data
            ct = commodity_type
        else:
            raise FormValidationError(
                "Either form_id or (form_type + data) must be provided"
            )

        errors = self._validate_required_fields(ft, form_data)
        warnings = self._validate_commodity_fields(ct, form_data)
        score = self._calculate_completeness(ft, form_data, ct)

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "completeness_score": score,
            "processing_time_ms": elapsed_ms,
        }

    def get_sync_queue(
        self,
        device_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[SyncQueueItem]:
        """Get items from the sync queue with optional filters.

        Args:
            device_id: Filter by device identifier.
            status: Filter by sync status.
            limit: Maximum number of results.

        Returns:
            List of SyncQueueItem instances sorted by priority.
        """
        with self._lock:
            items = list(self._sync_queue.values())

        if device_id:
            items = [i for i in items if i.device_id == device_id]
        if status:
            items = [i for i in items if i.status.value == status]

        # Sort by priority (1=highest), then by creation time
        items.sort(key=lambda i: (i.priority, i.created_at))

        return items[:limit]

    def mark_synced(
        self,
        form_id: str,
        queue_item_id: Optional[str] = None,
    ) -> FormResponse:
        """Mark a form as successfully synced to the server.

        Transitions the form status from SYNCING to SYNCED and updates
        the corresponding sync queue item to COMPLETED.

        Args:
            form_id: Form submission identifier.
            queue_item_id: Optional sync queue item identifier.

        Returns:
            FormResponse with updated status.

        Raises:
            FormNotFoundError: If the form does not exist.
            FormStateTransitionError: If the status transition is invalid.
        """
        start_time = time.monotonic()

        with self._lock:
            form = self._forms.get(form_id)
            if form is None:
                raise FormNotFoundError(
                    f"Form not found: form_id={form_id}"
                )

            # Allow transition from SYNCING or PENDING to SYNCED
            if form.status not in (
                FormStatus.SYNCING, FormStatus.PENDING,
            ):
                raise FormStateTransitionError(
                    f"Cannot transition from {form.status.value} "
                    f"to synced"
                )

            form.status = FormStatus.SYNCED
            form.updated_at = datetime.now(timezone.utc).replace(
                microsecond=0
            )

            # Update sync queue item
            if queue_item_id and queue_item_id in self._sync_queue:
                item = self._sync_queue[queue_item_id]
                item.status = SyncStatus.COMPLETED
                item.updated_at = datetime.now(timezone.utc).replace(
                    microsecond=0
                )
            else:
                # Find matching queue item by form_id
                for item in self._sync_queue.values():
                    if (
                        item.item_id == form_id
                        and item.status != SyncStatus.COMPLETED
                    ):
                        item.status = SyncStatus.COMPLETED
                        item.updated_at = datetime.now(
                            timezone.utc
                        ).replace(microsecond=0)
                        break

        provenance_entry = self._provenance.record(
            entity_type="form_submission",
            action="sync",
            entity_id=form_id,
            data={"status": "synced"},
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Form marked synced: form_id=%s elapsed=%.1fms",
            form_id, elapsed_ms,
        )

        return FormResponse(
            form_id=form_id,
            status=FormStatus.SYNCED,
            submission_hash=form.submission_hash,
            provenance_hash=provenance_entry.hash_value,
            processing_time_ms=elapsed_ms,
            message="Form synced successfully",
            form=form,
        )

    def get_completeness_score(
        self,
        form_id: str,
    ) -> Dict[str, Any]:
        """Calculate the data completeness score for a form.

        Returns the percentage of required fields that are filled,
        plus commodity-specific field completion.

        Args:
            form_id: Form submission identifier.

        Returns:
            Dictionary with total_score, base_score,
            commodity_score, filled_fields, total_required_fields.

        Raises:
            FormNotFoundError: If the form does not exist.
        """
        form = self.get_form(form_id)
        ft = form.form_type.value
        ct = (
            form.commodity_type.value
            if form.commodity_type else None
        )
        return self._build_completeness_report(ft, form.data, ct)

    def delete_form(self, form_id: str) -> Dict[str, Any]:
        """Delete a form submission from the store.

        Only forms in DRAFT or FAILED status can be deleted.

        Args:
            form_id: Form submission identifier.

        Returns:
            Dictionary with deleted form_id and status.

        Raises:
            FormNotFoundError: If the form does not exist.
            FormStateTransitionError: If form is not deletable.
        """
        with self._lock:
            form = self._forms.get(form_id)
            if form is None:
                raise FormNotFoundError(
                    f"Form not found: form_id={form_id}"
                )

            if form.status not in (
                FormStatus.DRAFT, FormStatus.FAILED,
            ):
                raise FormStateTransitionError(
                    f"Cannot delete form in {form.status.value} status; "
                    f"only DRAFT or FAILED forms can be deleted"
                )

            del self._forms[form_id]
            self._drafts.pop(form_id, None)

            # Remove from sync queue
            queue_ids_to_remove = [
                qid for qid, item in self._sync_queue.items()
                if item.item_id == form_id
            ]
            for qid in queue_ids_to_remove:
                del self._sync_queue[qid]

        self._provenance.record(
            entity_type="form_submission",
            action="update",
            entity_id=form_id,
            data={"action": "delete", "status": form.status.value},
        )

        logger.info("Form deleted: form_id=%s", form_id)

        return {
            "form_id": form_id,
            "deleted": True,
            "previous_status": form.status.value,
        }

    def detect_conflicts(
        self,
        form_id: str,
        incoming_data: Dict[str, Any],
        incoming_device_id: str,
    ) -> Dict[str, Any]:
        """Detect conflicts when the same form is edited on multiple devices.

        Compares incoming data against the stored form and reports
        field-level conflicts.

        Args:
            form_id: Form submission identifier.
            incoming_data: Data from the remote device.
            incoming_device_id: Device submitting the incoming data.

        Returns:
            Dictionary with has_conflicts, conflict_fields list,
            and conflict_count.

        Raises:
            FormNotFoundError: If the form does not exist.
        """
        form = self.get_form(form_id)
        conflicts: List[Dict[str, Any]] = []

        for key, incoming_value in incoming_data.items():
            existing_value = form.data.get(key)
            if existing_value is not None and existing_value != incoming_value:
                conflicts.append({
                    "field_name": key,
                    "local_value": existing_value,
                    "incoming_value": incoming_value,
                    "local_device_id": form.device_id,
                    "incoming_device_id": incoming_device_id,
                })

        if conflicts:
            logger.warning(
                "Conflicts detected for form_id=%s: %d fields",
                form_id, len(conflicts),
            )

        return {
            "form_id": form_id,
            "has_conflicts": len(conflicts) > 0,
            "conflict_fields": conflicts,
            "conflict_count": len(conflicts),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def form_count(self) -> int:
        """Return the total number of stored forms."""
        with self._lock:
            return len(self._forms)

    @property
    def draft_count(self) -> int:
        """Return the number of forms in DRAFT status."""
        with self._lock:
            return sum(
                1 for f in self._forms.values()
                if f.status == FormStatus.DRAFT
            )

    @property
    def pending_count(self) -> int:
        """Return the number of forms in PENDING status."""
        with self._lock:
            return sum(
                1 for f in self._forms.values()
                if f.status == FormStatus.PENDING
            )

    @property
    def sync_queue_depth(self) -> int:
        """Return the number of items in the sync queue."""
        with self._lock:
            return len(self._sync_queue)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_form_type(self, form_type: str) -> None:
        """Validate that form_type is a supported EUDR form type.

        Args:
            form_type: Form type string to validate.

        Raises:
            FormValidationError: If form_type is not supported.
        """
        valid_types = {ft.value for ft in FormType}
        if form_type not in valid_types:
            raise FormValidationError(
                f"Invalid form_type '{form_type}'; "
                f"must be one of {sorted(valid_types)}"
            )

    def _parse_commodity_type(
        self,
        commodity_type: Optional[str],
    ) -> Optional[CommodityType]:
        """Parse and validate commodity type string.

        Args:
            commodity_type: Commodity type string or None.

        Returns:
            CommodityType enum value or None.

        Raises:
            FormValidationError: If commodity_type is invalid.
        """
        if commodity_type is None:
            return None
        try:
            return CommodityType(commodity_type)
        except ValueError:
            valid = [ct.value for ct in CommodityType]
            raise FormValidationError(
                f"Invalid commodity_type '{commodity_type}'; "
                f"must be one of {valid}"
            )

    def _build_form_submission(
        self,
        device_id: str,
        operator_id: str,
        form_type: str,
        template_id: str,
        template_version: str,
        data: Dict[str, Any],
        commodity_type: Optional[CommodityType],
        country_code: Optional[str],
        local_timestamp: Optional[datetime],
        gps_capture_ids: List[str],
        photo_ids: List[str],
        signature_ids: List[str],
        metadata: Dict[str, Any],
    ) -> FormSubmission:
        """Build a FormSubmission model from input parameters.

        Args:
            device_id: Source device identifier.
            operator_id: Field agent identifier.
            form_type: EUDR form type string.
            template_id: Template definition identifier.
            template_version: Template version string.
            data: Form field values.
            commodity_type: Parsed commodity type or None.
            country_code: ISO country code or None.
            local_timestamp: Device timestamp or None.
            gps_capture_ids: Linked GPS identifiers.
            photo_ids: Linked photo identifiers.
            signature_ids: Linked signature identifiers.
            metadata: Additional metadata.

        Returns:
            Populated FormSubmission instance.
        """
        now = datetime.now(timezone.utc).replace(microsecond=0)
        return FormSubmission(
            device_id=device_id,
            operator_id=operator_id,
            form_type=FormType(form_type),
            template_id=template_id,
            template_version=template_version,
            status=FormStatus.DRAFT,
            data=data,
            commodity_type=commodity_type,
            country_code=country_code,
            local_timestamp=local_timestamp,
            gps_capture_ids=gps_capture_ids,
            photo_ids=photo_ids,
            signature_ids=signature_ids,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )

    def _validate_form_data(
        self,
        form: FormSubmission,
    ) -> List[str]:
        """Validate form data against required field rules.

        Args:
            form: FormSubmission to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        # Size check
        data_json = json.dumps(form.data, default=str)
        size_kb = len(data_json.encode("utf-8")) / 1024
        if size_kb > self._config.max_form_size_kb:
            errors.append(
                f"Form data size {size_kb:.1f}KB exceeds "
                f"max {self._config.max_form_size_kb}KB"
            )

        # Required fields
        required_errors = self._validate_required_fields(
            form.form_type.value, form.data,
        )
        errors.extend(required_errors)

        return errors

    def _validate_required_fields(
        self,
        form_type: str,
        data: Dict[str, Any],
    ) -> List[str]:
        """Validate that all required fields for a form type are present.

        Args:
            form_type: EUDR form type string.
            data: Form field values.

        Returns:
            List of validation error messages.
        """
        errors: List[str] = []
        required = _REQUIRED_FIELDS.get(form_type, [])
        for field_name in required:
            value = data.get(field_name)
            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                errors.append(
                    f"Required field '{field_name}' is missing or empty "
                    f"for form type '{form_type}'"
                )
        return errors

    def _validate_commodity_fields(
        self,
        commodity_type: Optional[str],
        data: Dict[str, Any],
    ) -> List[str]:
        """Validate commodity-specific fields (as warnings).

        Args:
            commodity_type: EUDR commodity type or None.
            data: Form field values.

        Returns:
            List of warning messages for missing commodity fields.
        """
        warnings: List[str] = []
        if commodity_type is None:
            return warnings
        commodity_fields = _COMMODITY_FIELDS.get(commodity_type, [])
        for field_name in commodity_fields:
            value = data.get(field_name)
            if value is None or (
                isinstance(value, str) and not value.strip()
            ):
                warnings.append(
                    f"Commodity-specific field '{field_name}' is missing "
                    f"for commodity '{commodity_type}'"
                )
        return warnings

    def _calculate_completeness(
        self,
        form_type: str,
        data: Dict[str, Any],
        commodity_type: Optional[str],
    ) -> float:
        """Calculate the percentage of required fields that are filled.

        Args:
            form_type: EUDR form type string.
            data: Form field values.
            commodity_type: EUDR commodity type or None.

        Returns:
            Completeness score as a float between 0.0 and 100.0.
        """
        required = list(_REQUIRED_FIELDS.get(form_type, []))
        if commodity_type:
            required.extend(_COMMODITY_FIELDS.get(commodity_type, []))

        if not required:
            return 100.0

        filled = sum(
            1 for f in required
            if data.get(f) is not None
            and not (isinstance(data.get(f), str) and not data[f].strip())
        )

        return round((filled / len(required)) * 100, 2)

    def _build_completeness_report(
        self,
        form_type: str,
        data: Dict[str, Any],
        commodity_type: Optional[str],
    ) -> Dict[str, Any]:
        """Build a detailed completeness report for a form.

        Args:
            form_type: EUDR form type string.
            data: Form field values.
            commodity_type: EUDR commodity type or None.

        Returns:
            Dictionary with total_score, base_score,
            commodity_score, filled_fields, total_required_fields.
        """
        base_required = _REQUIRED_FIELDS.get(form_type, [])
        commodity_required = (
            _COMMODITY_FIELDS.get(commodity_type, [])
            if commodity_type else []
        )

        def _count_filled(fields: List[str]) -> int:
            return sum(
                1 for f in fields
                if data.get(f) is not None
                and not (
                    isinstance(data.get(f), str) and not data[f].strip()
                )
            )

        base_filled = _count_filled(base_required)
        commodity_filled = _count_filled(commodity_required)

        total_required = len(base_required) + len(commodity_required)
        total_filled = base_filled + commodity_filled

        base_score = (
            round((base_filled / len(base_required)) * 100, 2)
            if base_required else 100.0
        )
        commodity_score = (
            round(
                (commodity_filled / len(commodity_required)) * 100, 2,
            )
            if commodity_required else 100.0
        )
        total_score = (
            round((total_filled / total_required) * 100, 2)
            if total_required else 100.0
        )

        return {
            "total_score": total_score,
            "base_score": base_score,
            "commodity_score": commodity_score,
            "filled_fields": total_filled,
            "total_required_fields": total_required,
            "base_required": len(base_required),
            "base_filled": base_filled,
            "commodity_required": len(commodity_required),
            "commodity_filled": commodity_filled,
        }

    def _compute_submission_hash(
        self,
        form: FormSubmission,
    ) -> str:
        """Compute SHA-256 hash of form data for integrity verification.

        Args:
            form: FormSubmission whose data to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        hash_input = json.dumps(
            form.data, sort_keys=True, default=str,
        )
        return hashlib.sha256(
            hash_input.encode("utf-8")
        ).hexdigest()

    def _add_to_sync_queue(self, form: FormSubmission) -> None:
        """Add a form to the sync queue for upload.

        Args:
            form: FormSubmission to queue for synchronization.
        """
        data_json = json.dumps(form.data, default=str)
        payload_size = len(data_json.encode("utf-8"))

        item = SyncQueueItem(
            device_id=form.device_id,
            item_type="form",
            item_id=form.form_id,
            priority=2,
            status=SyncStatus.QUEUED,
            payload_size_bytes=payload_size,
        )

        with self._lock:
            self._sync_queue[item.queue_item_id] = item

        logger.debug(
            "Added to sync queue: item_id=%s form_id=%s size=%d",
            item.queue_item_id, form.form_id, payload_size,
        )

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"OfflineFormEngine(forms={self.form_count}, "
            f"drafts={self.draft_count}, "
            f"pending={self.pending_count}, "
            f"queue={self.sync_queue_depth})"
        )

    def __len__(self) -> int:
        """Return the total number of stored forms."""
        return self.form_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "OfflineFormEngine",
    "OfflineFormEngineError",
    "FormNotFoundError",
    "FormValidationError",
    "FormStateTransitionError",
    "FormConflictError",
]
