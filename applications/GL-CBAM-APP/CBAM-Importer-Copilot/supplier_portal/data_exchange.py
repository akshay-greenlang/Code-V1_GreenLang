# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal - Data Exchange Service

Manages the bi-directional data exchange between EU importers and
third-country suppliers.  Handles access request workflows, authorization
management, data retrieval for importers, bulk export, and a full audit
trail of who accessed what and when.

All access control is enforced at the service layer: importers can only
retrieve data from suppliers that have explicitly approved their access
request.

Reference:
  - EU CBAM Regulation 2023/956, Art. 6-8 (authorised CBAM declarant)
  - EU Implementing Regulation 2023/1773, Art. 10 (data exchange)

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import uuid
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

from supplier_portal.models import (
    AccessEvent,
    AccessRequest,
    AccessRequestStatus,
    CBAMSector,
    EmissionsDataSubmission,
    ExportFormat,
    Installation,
    SubmissionStatus,
    _utc_now,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================


class DataExchangeError(Exception):
    """Base exception for data exchange operations."""


class AccessDeniedError(DataExchangeError):
    """Raised when an importer lacks access to a supplier's data."""


class AccessRequestNotFoundError(DataExchangeError):
    """Raised when a referenced access request does not exist."""


class DuplicateAccessRequestError(DataExchangeError):
    """Raised when an importer already has a pending request."""


# ============================================================================
# JSON ENCODER
# ============================================================================


class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder for Decimal and datetime serialization."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if hasattr(obj, "model_dump"):
            return obj.model_dump(mode="json")
        return super().default(obj)


# ============================================================================
# DATA EXCHANGE SERVICE
# ============================================================================


class DataExchangeService:
    """
    Service for managing importer-supplier data exchange.

    Implements the full access request lifecycle:
      1. Importer requests data access (request_data_access)
      2. Supplier reviews and approves/denies (approve_data_access)
      3. Importer retrieves authorized data (get_supplier_emissions_for_importer)
      4. Supplier can revoke at any time (revoke_data_access)

    All data access events are recorded in the audit log for CBAM compliance.

    Thread Safety:
      All public methods acquire self._lock (RLock) before mutating state.

    Args:
        registry: SupplierRegistryEngine instance.
        submissions: EmissionsSubmissionEngine instance.

    Example:
        >>> service = DataExchangeService(registry, submissions)
        >>> request = service.request_data_access(
        ...     importer_id="IMP-001",
        ...     supplier_id="SUP-ABC",
        ...     scope="emissions_data",
        ...     purpose="CBAM quarterly report Q1 2026",
        ... )
        >>> service.approve_data_access("SUP-ABC", request.request_id)
        >>> data = service.get_supplier_emissions_for_importer(
        ...     "IMP-001", "SUP-ABC", "2026Q1"
        ... )
    """

    def __init__(self, registry: Any, submissions: Any) -> None:
        """Initialize data exchange service."""
        self._registry = registry
        self._submissions = submissions
        self._lock = threading.RLock()
        self._access_requests: Dict[str, AccessRequest] = {}
        self._approved_access: Dict[str, Dict[str, _ApprovedAccess]] = {}
        self._audit_log: List[AccessEvent] = []
        self._notifications: List[Dict[str, Any]] = []
        logger.info("DataExchangeService initialized")

    # ------------------------------------------------------------------
    # Data Retrieval for Importers
    # ------------------------------------------------------------------

    def get_supplier_emissions_for_importer(
        self,
        importer_id: str,
        supplier_id: str,
        period: Optional[str] = None,
    ) -> List[EmissionsDataSubmission]:
        """
        Retrieve authorized emissions data for an importer.

        Returns only submissions from installations the importer has been
        authorized to access, filtered to accepted/submitted statuses.

        Args:
            importer_id: EU importer identifier.
            supplier_id: Supplier identifier.
            period: Optional reporting period filter (YYYYQN).

        Returns:
            List of EmissionsDataSubmission the importer is authorized to see.

        Raises:
            AccessDeniedError: If the importer does not have access.
        """
        with self._lock:
            access = self._check_access(importer_id, supplier_id)

            # Get all submissions for the supplier
            all_submissions = self._submissions.get_submissions(
                supplier_id=supplier_id
            )

            # Filter to authorized installations
            authorized_inst = access.installation_ids
            if authorized_inst:
                all_submissions = [
                    s
                    for s in all_submissions
                    if s.installation_id in authorized_inst
                ]

            # Filter to period
            if period:
                all_submissions = [
                    s for s in all_submissions if s.reporting_period == period
                ]

            # Only show accepted or submitted
            visible_statuses = {
                SubmissionStatus.ACCEPTED,
                SubmissionStatus.SUBMITTED,
            }
            result = [
                s
                for s in all_submissions
                if s.submission_status in visible_statuses
            ]

            # Record access event
            self._record_access_event(
                actor_id=importer_id,
                actor_type="importer",
                action="view",
                resource_type="emissions_data",
                resource_id=supplier_id,
                supplier_id=supplier_id,
                details={
                    "period": period,
                    "results_count": len(result),
                },
            )

            logger.info(
                "Importer %s retrieved %d submissions from supplier %s (period=%s)",
                importer_id,
                len(result),
                supplier_id,
                period,
            )

        return result

    def search_third_country_installations(
        self,
        country: Optional[str] = None,
        sector: Optional[CBAMSector] = None,
        name: Optional[str] = None,
    ) -> List[Installation]:
        """
        Search the registry for third-country installations.

        This is a public search endpoint that does not require an active
        access grant.  Returns only active, verified (or pending) installations.

        Args:
            country: ISO 3166-1 alpha-2 country code filter.
            sector: CBAM sector filter.
            name: Case-insensitive name substring match.

        Returns:
            List of matching Installation objects.
        """
        with self._lock:
            all_suppliers = self._registry.get_all_suppliers()
            results: List[Installation] = []

            for profile in all_suppliers:
                try:
                    installations = self._registry.get_installations(
                        profile.supplier_id
                    )
                except Exception:
                    continue

                for inst in installations:
                    if not inst.is_active:
                        continue

                    # Apply country filter
                    if country and inst.country.upper() != country.upper():
                        continue

                    # Apply sector filter
                    if sector and sector not in inst.cbam_sectors:
                        continue

                    # Apply name filter
                    if name and name.lower() not in inst.name.lower():
                        continue

                    results.append(inst)

            logger.info(
                "Installation search: %d results (country=%s, sector=%s, name=%s)",
                len(results),
                country,
                sector,
                name,
            )

        return results

    # ------------------------------------------------------------------
    # Access Request Workflow
    # ------------------------------------------------------------------

    def request_data_access(
        self,
        importer_id: str,
        supplier_id: str,
        scope: str = "emissions_data",
        purpose: str = "",
        installation_ids: Optional[List[str]] = None,
        access_duration_days: int = 365,
    ) -> AccessRequest:
        """
        Create a data access request from an importer to a supplier.

        The request starts in PENDING status and must be approved by the
        supplier before the importer can retrieve data.

        Args:
            importer_id: EU importer identifier.
            supplier_id: Target supplier identifier.
            scope: Scope of access (emissions_data, verification_reports, all).
            purpose: Reason for the access request.
            installation_ids: Specific installations (None = all).
            access_duration_days: Duration of access in days (1-730).

        Returns:
            The created AccessRequest.

        Raises:
            DuplicateAccessRequestError: If a pending request already exists.
        """
        with self._lock:
            # Verify supplier exists
            self._registry.get_supplier(supplier_id)

            # Check for existing pending request
            for req in self._access_requests.values():
                if (
                    req.importer_id == importer_id
                    and req.supplier_id == supplier_id
                    and req.status == AccessRequestStatus.PENDING
                ):
                    raise DuplicateAccessRequestError(
                        f"Importer '{importer_id}' already has a pending "
                        f"access request for supplier '{supplier_id}'"
                    )

            request = AccessRequest(
                importer_id=importer_id,
                importer_name=importer_id,
                supplier_id=supplier_id,
                installation_ids=installation_ids,
                scope=scope,
                purpose=purpose,
                status=AccessRequestStatus.PENDING,
                requested_at=datetime.now(timezone.utc),
                access_duration_days=access_duration_days,
            )

            self._access_requests[request.request_id] = request

            self._record_access_event(
                actor_id=importer_id,
                actor_type="importer",
                action="access_requested",
                resource_type="access_request",
                resource_id=request.request_id,
                supplier_id=supplier_id,
                details={
                    "scope": scope,
                    "purpose": purpose,
                    "installations": installation_ids,
                },
            )

            logger.info(
                "Access request created: %s (importer=%s -> supplier=%s)",
                request.request_id,
                importer_id,
                supplier_id,
            )

        return request

    def approve_data_access(
        self,
        supplier_id: str,
        request_id: str,
        restrictions: Optional[List[str]] = None,
        approved_installation_ids: Optional[List[str]] = None,
    ) -> bool:
        """
        Supplier approves an importer's data access request.

        Creates an active access grant that allows the importer to
        retrieve the supplier's emissions data.

        Args:
            supplier_id: Supplier performing the approval.
            request_id: ID of the access request to approve.
            restrictions: Optional restrictions on the access.
            approved_installation_ids: Subset of installations to approve.

        Returns:
            True if the request was approved.

        Raises:
            AccessRequestNotFoundError: If the request does not exist.
            DataExchangeError: If the request belongs to a different supplier.
        """
        with self._lock:
            request = self._get_request_or_raise(request_id)

            if request.supplier_id != supplier_id:
                raise DataExchangeError(
                    f"Access request '{request_id}' does not belong to "
                    f"supplier '{supplier_id}'"
                )

            if request.status != AccessRequestStatus.PENDING:
                raise DataExchangeError(
                    f"Access request '{request_id}' is in "
                    f"'{request.status.value}' state, cannot approve"
                )

            # Update request status
            request = request.model_copy(
                update={
                    "status": AccessRequestStatus.APPROVED,
                    "resolved_at": datetime.now(timezone.utc),
                    "restrictions": restrictions,
                }
            )
            self._access_requests[request_id] = request

            # Determine installations to authorize
            inst_ids = approved_installation_ids or request.installation_ids
            if inst_ids is None:
                # All installations
                try:
                    installations = self._registry.get_installations(supplier_id)
                    inst_ids = [i.installation_id for i in installations]
                except Exception:
                    inst_ids = []

            # Create approved access record
            expires_at = datetime.now(timezone.utc) + timedelta(
                days=request.access_duration_days
            )
            access = _ApprovedAccess(
                importer_id=request.importer_id,
                supplier_id=supplier_id,
                installation_ids=set(inst_ids) if inst_ids else None,
                scope=request.scope,
                approved_at=datetime.now(timezone.utc),
                expires_at=expires_at,
                restrictions=restrictions or [],
                is_active=True,
            )

            if supplier_id not in self._approved_access:
                self._approved_access[supplier_id] = {}
            self._approved_access[supplier_id][request.importer_id] = access

            # Also link in the registry
            try:
                self._registry.link_supplier_to_importer(
                    supplier_id=supplier_id,
                    importer_id=request.importer_id,
                    authorization_token=request.request_id,
                )
            except Exception as e:
                logger.warning(
                    "Could not link importer in registry: %s", e
                )

            self._record_access_event(
                actor_id=supplier_id,
                actor_type="supplier",
                action="access_approved",
                resource_type="access_request",
                resource_id=request_id,
                supplier_id=supplier_id,
                details={
                    "importer_id": request.importer_id,
                    "installations": inst_ids,
                    "expires_at": expires_at.isoformat(),
                },
            )

            logger.info(
                "Access approved: %s (importer=%s, supplier=%s, expires=%s)",
                request_id,
                request.importer_id,
                supplier_id,
                expires_at.isoformat(),
            )

        return True

    def deny_data_access(
        self,
        supplier_id: str,
        request_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Supplier denies an importer's data access request.

        Args:
            supplier_id: Supplier performing the denial.
            request_id: ID of the access request to deny.
            reason: Optional reason for denial.

        Returns:
            True if the request was denied.

        Raises:
            AccessRequestNotFoundError: If the request does not exist.
            DataExchangeError: If the request does not belong to this supplier.
        """
        with self._lock:
            request = self._get_request_or_raise(request_id)

            if request.supplier_id != supplier_id:
                raise DataExchangeError(
                    f"Access request '{request_id}' does not belong to "
                    f"supplier '{supplier_id}'"
                )

            if request.status != AccessRequestStatus.PENDING:
                raise DataExchangeError(
                    f"Access request '{request_id}' is in "
                    f"'{request.status.value}' state, cannot deny"
                )

            request = request.model_copy(
                update={
                    "status": AccessRequestStatus.DENIED,
                    "resolved_at": datetime.now(timezone.utc),
                    "notes": reason,
                }
            )
            self._access_requests[request_id] = request

            self._record_access_event(
                actor_id=supplier_id,
                actor_type="supplier",
                action="access_denied",
                resource_type="access_request",
                resource_id=request_id,
                supplier_id=supplier_id,
                details={
                    "importer_id": request.importer_id,
                    "reason": reason,
                },
            )

            logger.info(
                "Access denied: %s (importer=%s, supplier=%s)",
                request_id,
                request.importer_id,
                supplier_id,
            )

        return True

    def revoke_data_access(
        self,
        supplier_id: str,
        importer_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Revoke an importer's data access to a supplier.

        Args:
            supplier_id: Supplier revoking access.
            importer_id: Importer whose access is being revoked.
            reason: Optional reason for revocation.

        Returns:
            True if access was revoked, False if no active access existed.
        """
        with self._lock:
            supplier_access = self._approved_access.get(supplier_id, {})
            access = supplier_access.get(importer_id)

            if access is None or not access.is_active:
                return False

            access.is_active = False
            del supplier_access[importer_id]

            # Unlink in registry
            try:
                self._registry.unlink_supplier_from_importer(
                    supplier_id, importer_id
                )
            except Exception as e:
                logger.warning(
                    "Could not unlink importer in registry: %s", e
                )

            # Update any approved requests to revoked
            for req in self._access_requests.values():
                if (
                    req.importer_id == importer_id
                    and req.supplier_id == supplier_id
                    and req.status == AccessRequestStatus.APPROVED
                ):
                    updated = req.model_copy(
                        update={
                            "status": AccessRequestStatus.REVOKED,
                            "notes": reason or "Access revoked by supplier",
                        }
                    )
                    self._access_requests[req.request_id] = updated

            self._record_access_event(
                actor_id=supplier_id,
                actor_type="supplier",
                action="access_revoked",
                resource_type="access_grant",
                resource_id=f"{supplier_id}:{importer_id}",
                supplier_id=supplier_id,
                details={
                    "importer_id": importer_id,
                    "reason": reason,
                },
            )

            logger.info(
                "Access revoked: importer=%s from supplier=%s",
                importer_id,
                supplier_id,
            )

        return True

    # ------------------------------------------------------------------
    # Bulk Export
    # ------------------------------------------------------------------

    def bulk_export_supplier_data(
        self,
        importer_id: str,
        period: Optional[str] = None,
        fmt: ExportFormat = ExportFormat.JSON,
    ) -> bytes:
        """
        Bulk export emissions data from all linked suppliers for an importer.

        Args:
            importer_id: EU importer identifier.
            period: Optional reporting period filter (YYYYQN).
            fmt: Export format (JSON, CSV, or XML).

        Returns:
            Byte content of the exported data.
        """
        with self._lock:
            all_data: List[Dict[str, Any]] = []

            # Find all suppliers this importer has access to
            for supplier_id, access_map in self._approved_access.items():
                access = access_map.get(importer_id)
                if access is None or not access.is_active:
                    continue

                # Check expiry
                if access.expires_at < datetime.now(timezone.utc):
                    continue

                try:
                    submissions = self.get_supplier_emissions_for_importer(
                        importer_id, supplier_id, period
                    )
                    for sub in submissions:
                        all_data.append({
                            "supplier_id": supplier_id,
                            **sub.model_dump(mode="json"),
                        })
                except AccessDeniedError:
                    continue

            self._record_access_event(
                actor_id=importer_id,
                actor_type="importer",
                action="bulk_export",
                resource_type="emissions_data",
                resource_id="bulk",
                supplier_id="multiple",
                details={
                    "period": period,
                    "format": fmt.value,
                    "record_count": len(all_data),
                },
            )

            if fmt == ExportFormat.JSON:
                return self._export_json(all_data)
            elif fmt == ExportFormat.CSV:
                return self._export_csv(all_data)
            elif fmt == ExportFormat.XML:
                return self._export_xml(all_data)
            else:
                return self._export_json(all_data)

    # ------------------------------------------------------------------
    # Audit & Notifications
    # ------------------------------------------------------------------

    def get_access_audit_log(
        self,
        supplier_id: str,
        limit: int = 100,
    ) -> List[AccessEvent]:
        """
        Retrieve the data access audit log for a supplier.

        Shows all access events: who accessed what data and when.

        Args:
            supplier_id: Supplier identifier.
            limit: Maximum entries to return.

        Returns:
            List of AccessEvent objects, newest first.
        """
        with self._lock:
            entries = [
                e
                for e in self._audit_log
                if e.supplier_id == supplier_id
            ]
            # Newest first
            entries.sort(key=lambda e: e.timestamp, reverse=True)
            return entries[:limit]

    def get_access_requests(
        self,
        supplier_id: Optional[str] = None,
        importer_id: Optional[str] = None,
        status: Optional[AccessRequestStatus] = None,
    ) -> List[AccessRequest]:
        """
        Retrieve access requests with optional filters.

        Args:
            supplier_id: Filter by supplier.
            importer_id: Filter by importer.
            status: Filter by request status.

        Returns:
            List of matching AccessRequest objects.
        """
        with self._lock:
            results = list(self._access_requests.values())

            if supplier_id:
                results = [r for r in results if r.supplier_id == supplier_id]

            if importer_id:
                results = [r for r in results if r.importer_id == importer_id]

            if status:
                results = [r for r in results if r.status == status]

            return results

    def get_access_request(self, request_id: str) -> AccessRequest:
        """
        Retrieve a single access request by ID.

        Args:
            request_id: Access request identifier.

        Returns:
            The AccessRequest.

        Raises:
            AccessRequestNotFoundError: If not found.
        """
        with self._lock:
            return self._get_request_or_raise(request_id)

    def notify_data_update(
        self,
        supplier_id: str,
        submission_id: str,
    ) -> List[str]:
        """
        Notify all linked importers that new/updated data is available.

        Args:
            supplier_id: Supplier whose data was updated.
            submission_id: ID of the updated submission.

        Returns:
            List of importer IDs that were notified.
        """
        with self._lock:
            notified: List[str] = []
            supplier_access = self._approved_access.get(supplier_id, {})

            for importer_id, access in supplier_access.items():
                if not access.is_active:
                    continue
                if access.expires_at < datetime.now(timezone.utc):
                    continue

                notification = {
                    "notification_id": str(uuid.uuid4()),
                    "type": "data_update",
                    "supplier_id": supplier_id,
                    "importer_id": importer_id,
                    "submission_id": submission_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": (
                        f"Supplier {supplier_id} has updated emissions data "
                        f"(submission {submission_id})"
                    ),
                }
                self._notifications.append(notification)
                notified.append(importer_id)

                logger.info(
                    "Notification sent to importer %s: data update from %s",
                    importer_id,
                    supplier_id,
                )

            self._record_access_event(
                actor_id=supplier_id,
                actor_type="supplier",
                action="data_update_notified",
                resource_type="submission",
                resource_id=submission_id,
                supplier_id=supplier_id,
                details={
                    "notified_importers": notified,
                },
            )

        return notified

    def get_notifications(
        self,
        importer_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve notifications, optionally filtered.

        Args:
            importer_id: Filter by importer.
            supplier_id: Filter by supplier.
            limit: Maximum entries to return.

        Returns:
            List of notification dictionaries.
        """
        with self._lock:
            results = list(self._notifications)

            if importer_id:
                results = [
                    n for n in results if n.get("importer_id") == importer_id
                ]

            if supplier_id:
                results = [
                    n for n in results if n.get("supplier_id") == supplier_id
                ]

            results.sort(
                key=lambda n: n.get("timestamp", ""),
                reverse=True,
            )

            return results[:limit]

    # ------------------------------------------------------------------
    # Reset (for testing)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all in-memory stores. For testing only."""
        with self._lock:
            self._access_requests.clear()
            self._approved_access.clear()
            self._audit_log.clear()
            self._notifications.clear()
            logger.warning("DataExchangeService stores reset")

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _check_access(
        self, importer_id: str, supplier_id: str
    ) -> "_ApprovedAccess":
        """
        Verify that an importer has active access to a supplier's data.

        Raises:
            AccessDeniedError: If access is not granted or has expired.
        """
        supplier_access = self._approved_access.get(supplier_id, {})
        access = supplier_access.get(importer_id)

        if access is None or not access.is_active:
            raise AccessDeniedError(
                f"Importer '{importer_id}' does not have access to "
                f"supplier '{supplier_id}' data"
            )

        # Check expiry
        if access.expires_at < datetime.now(timezone.utc):
            access.is_active = False
            raise AccessDeniedError(
                f"Access for importer '{importer_id}' to supplier "
                f"'{supplier_id}' has expired"
            )

        return access

    def _get_request_or_raise(self, request_id: str) -> AccessRequest:
        """Retrieve access request or raise AccessRequestNotFoundError."""
        request = self._access_requests.get(request_id)
        if request is None:
            raise AccessRequestNotFoundError(
                f"Access request '{request_id}' not found"
            )
        return request

    def _record_access_event(
        self,
        actor_id: str,
        actor_type: str,
        action: str,
        resource_type: str,
        resource_id: str,
        supplier_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a data access event in the audit log."""
        event = AccessEvent(
            timestamp=datetime.now(timezone.utc),
            actor_id=actor_id,
            actor_type=actor_type,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            supplier_id=supplier_id,
            details=details,
        )
        self._audit_log.append(event)

    def _export_json(self, data: List[Dict[str, Any]]) -> bytes:
        """Export data as JSON bytes."""
        return json.dumps(
            {"data": data, "count": len(data)},
            cls=_DecimalEncoder,
            indent=2,
        ).encode("utf-8")

    def _export_csv(self, data: List[Dict[str, Any]]) -> bytes:
        """Export data as CSV bytes."""
        import csv
        import io

        output = io.StringIO()
        if not data:
            return b""

        # Use keys from the first record as headers
        headers = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for record in data:
            # Flatten complex values
            flat = {}
            for k, v in record.items():
                if isinstance(v, (dict, list)):
                    flat[k] = json.dumps(v, cls=_DecimalEncoder)
                else:
                    flat[k] = v
            writer.writerow(flat)

        return output.getvalue().encode("utf-8")

    def _export_xml(self, data: List[Dict[str, Any]]) -> bytes:
        """Export data as XML bytes."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append("<CBAMExport>")

        for record in data:
            lines.append("  <Record>")
            for key, value in record.items():
                safe_key = key.replace(" ", "_")
                safe_value = _xml_escape(str(value) if value is not None else "")
                lines.append(f"    <{safe_key}>{safe_value}</{safe_key}>")
            lines.append("  </Record>")

        lines.append("</CBAMExport>")
        return "\n".join(lines).encode("utf-8")


# ============================================================================
# INTERNAL DATA CLASSES
# ============================================================================


class _ApprovedAccess:
    """Internal record of an approved data access grant."""

    __slots__ = (
        "importer_id",
        "supplier_id",
        "installation_ids",
        "scope",
        "approved_at",
        "expires_at",
        "restrictions",
        "is_active",
    )

    def __init__(
        self,
        importer_id: str,
        supplier_id: str,
        installation_ids: Optional[set],
        scope: str,
        approved_at: datetime,
        expires_at: datetime,
        restrictions: List[str],
        is_active: bool = True,
    ) -> None:
        self.importer_id = importer_id
        self.supplier_id = supplier_id
        self.installation_ids = installation_ids
        self.scope = scope
        self.approved_at = approved_at
        self.expires_at = expires_at
        self.restrictions = restrictions
        self.is_active = is_active


# ============================================================================
# MODULE HELPERS
# ============================================================================


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
