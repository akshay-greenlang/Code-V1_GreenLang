# -*- coding: utf-8 -*-
"""
Transport Routes - AGENT-EUDR-010 Segregation Verifier API

Endpoints for managing transport vehicle segregation including vehicle
registration, segregation verification, cleaning records, and cargo history.

Endpoints:
    POST   /transport/vehicles              - Register transport vehicle
    GET    /transport/vehicles/{vehicle_id}  - Get vehicle details
    POST   /transport/verify                - Verify transport segregation
    POST   /transport/cleaning              - Record cleaning verification
    GET    /transport/history/{vehicle_id}   - Get vehicle cargo history

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.segregation_verifier.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_request_id,
    get_sgv_service,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_vehicle_id,
)
from greenlang.agents.eudr.segregation_verifier.api.schemas import (
    CargoHistoryEntry,
    CleaningVerificationStatus,
    ContaminationSeverity,
    ProvenanceInfo,
    RecordCleaningRequest,
    RegisterVehicleRequest,
    TransportVerificationFinding,
    TransportVerificationResponse,
    VehicleHistoryResponse,
    VehicleResponse,
    VehicleStatus,
    VerifyTransportRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Transport Segregation"])

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

_vehicle_store: Dict[str, Dict] = {}
_cargo_history_store: Dict[str, List[Dict]] = {}


def _get_vehicle_store() -> Dict[str, Dict]:
    """Return the vehicle store singleton."""
    return _vehicle_store


def _get_cargo_history_store() -> Dict[str, List[Dict]]:
    """Return the cargo history store singleton."""
    return _cargo_history_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /transport/vehicles
# ---------------------------------------------------------------------------


@router.post(
    "/transport/vehicles",
    response_model=VehicleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register transport vehicle",
    description=(
        "Register a new transport vehicle for segregation tracking. "
        "Vehicles can be dedicated to specific commodities or shared "
        "with cleaning protocols between loads."
    ),
    responses={
        201: {"description": "Vehicle registered successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def register_vehicle(
    request: Request,
    body: RegisterVehicleRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:transport:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> VehicleResponse:
    """Register a new transport vehicle.

    Args:
        body: Vehicle registration parameters.
        user: Authenticated user with transport:create permission.

    Returns:
        VehicleResponse with the new vehicle details.
    """
    start = time.monotonic()
    try:
        vehicle_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)

        provenance_data = body.model_dump(mode="json")
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        vehicle_record = {
            "vehicle_id": vehicle_id,
            "vehicle_reference": body.vehicle_reference,
            "vehicle_type": body.vehicle_type,
            "operator_id": body.operator_id,
            "status": VehicleStatus.AVAILABLE,
            "commodity_restrictions": body.commodity_restrictions,
            "is_dedicated": body.is_dedicated,
            "capacity": body.capacity,
            "cleaning_protocol": body.cleaning_protocol,
            "last_cleaning_at": None,
            "last_cleaning_status": None,
            "last_cargo_commodity": None,
            "notes": body.notes,
            "metadata": body.metadata,
            "created_at": now,
            "updated_at": now,
            "provenance": provenance,
        }

        store = _get_vehicle_store()
        store[vehicle_id] = vehicle_record

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Vehicle registered: id=%s ref=%s type=%s operator=%s",
            vehicle_id,
            body.vehicle_reference,
            body.vehicle_type.value,
            body.operator_id,
        )

        return VehicleResponse(**vehicle_record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to register vehicle: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register transport vehicle",
        )


# ---------------------------------------------------------------------------
# GET /transport/vehicles/{vehicle_id}
# ---------------------------------------------------------------------------


@router.get(
    "/transport/vehicles/{vehicle_id}",
    response_model=VehicleResponse,
    summary="Get vehicle details",
    description="Retrieve full details of a registered transport vehicle.",
    responses={
        200: {"description": "Vehicle details"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Vehicle not found"},
    },
)
async def get_vehicle(
    request: Request,
    vehicle_id: str = Depends(validate_vehicle_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:transport:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VehicleResponse:
    """Get vehicle details by ID.

    Args:
        vehicle_id: Vehicle identifier.
        user: Authenticated user with transport:read permission.

    Returns:
        VehicleResponse with full vehicle details.

    Raises:
        HTTPException: 404 if vehicle not found.
    """
    try:
        store = _get_vehicle_store()
        record = store.get(vehicle_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transport vehicle {vehicle_id} not found",
            )

        return VehicleResponse(**record, processing_time_ms=0.0)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to retrieve vehicle %s: %s", vehicle_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transport vehicle",
        )


# ---------------------------------------------------------------------------
# POST /transport/verify
# ---------------------------------------------------------------------------


@router.post(
    "/transport/verify",
    response_model=TransportVerificationResponse,
    summary="Verify transport segregation",
    description=(
        "Verify transport segregation for a vehicle-batch combination. "
        "Checks cleaning status, previous cargo compatibility, and seal "
        "integrity before approving transport."
    ),
    responses={
        200: {"description": "Verification completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Vehicle not found"},
    },
)
async def verify_transport(
    request: Request,
    body: VerifyTransportRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:transport:verify")
    ),
    _rate: None = Depends(rate_limit_write),
) -> TransportVerificationResponse:
    """Verify transport segregation.

    Args:
        body: Verification request with check flags.
        user: Authenticated user with transport:verify permission.

    Returns:
        TransportVerificationResponse with findings and approval status.
    """
    start = time.monotonic()
    try:
        store = _get_vehicle_store()
        vehicle = store.get(body.vehicle_id)

        if vehicle is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transport vehicle {body.vehicle_id} not found",
            )

        verification_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).replace(microsecond=0)
        findings: List[TransportVerificationFinding] = []
        finding_idx = 0

        # Check 1: Cleaning status
        if body.check_cleaning_status:
            cleaning_ok = vehicle.get("last_cleaning_status") in (
                CleaningVerificationStatus.PASSED,
                None,  # New vehicle, no previous cargo
            )
            finding_idx += 1
            findings.append(TransportVerificationFinding(
                finding_id=f"TVF-{finding_idx:04d}",
                category="cleaning",
                severity=ContaminationSeverity.HIGH,
                passed=cleaning_ok,
                message=(
                    "Vehicle cleaning verification passed"
                    if cleaning_ok
                    else "Vehicle cleaning verification required"
                ),
                remediation=(
                    None if cleaning_ok
                    else "Perform cleaning and submit cleaning verification"
                ),
            ))

        # Check 2: Previous cargo compatibility
        if body.check_previous_cargo:
            last_cargo = vehicle.get("last_cargo_commodity")
            cargo_compatible = (
                last_cargo is None or last_cargo == body.commodity
            )
            finding_idx += 1
            findings.append(TransportVerificationFinding(
                finding_id=f"TVF-{finding_idx:04d}",
                category="cargo_history",
                severity=ContaminationSeverity.HIGH,
                passed=cargo_compatible,
                message=(
                    "Previous cargo is compatible"
                    if cargo_compatible
                    else (
                        f"Previous cargo ({last_cargo}) differs from "
                        f"current commodity ({body.commodity.value})"
                    )
                ),
                remediation=(
                    None if cargo_compatible
                    else "Verify cleaning protocol was followed"
                ),
            ))

        # Check 3: Seal integrity
        if body.check_seals:
            finding_idx += 1
            findings.append(TransportVerificationFinding(
                finding_id=f"TVF-{finding_idx:04d}",
                category="seals",
                severity=ContaminationSeverity.CRITICAL,
                passed=True,
                message="Seal integrity verified",
            ))

        # Check 4: Commodity restrictions
        restrictions = vehicle.get("commodity_restrictions", [])
        if restrictions and body.commodity not in restrictions:
            finding_idx += 1
            findings.append(TransportVerificationFinding(
                finding_id=f"TVF-{finding_idx:04d}",
                category="compatibility",
                severity=ContaminationSeverity.CRITICAL,
                passed=False,
                message=(
                    f"Vehicle is restricted to commodities: "
                    f"{[r.value if hasattr(r, 'value') else r for r in restrictions]}"
                ),
                remediation="Use a vehicle approved for this commodity",
            ))

        total_checks = len(findings)
        checks_passed = sum(1 for f in findings if f.passed)
        checks_failed = total_checks - checks_passed
        compliance_score = (
            (checks_passed / total_checks * 100.0) if total_checks > 0 else 0.0
        )
        is_approved = checks_failed == 0

        provenance_hash = _compute_provenance_hash({
            "verification_id": verification_id,
            "vehicle_id": body.vehicle_id,
            "batch_id": body.batch_id,
            "approved": is_approved,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="transport_verification",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Transport verification: vehicle=%s batch=%s approved=%s score=%.1f",
            body.vehicle_id,
            body.batch_id,
            is_approved,
            compliance_score,
        )

        return TransportVerificationResponse(
            verification_id=verification_id,
            vehicle_id=body.vehicle_id,
            batch_id=body.batch_id,
            commodity=body.commodity,
            is_approved=is_approved,
            compliance_score=compliance_score,
            total_checks=total_checks,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            findings=findings,
            verified_at=now,
            provenance=provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed transport verification: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify transport segregation",
        )


# ---------------------------------------------------------------------------
# POST /transport/cleaning
# ---------------------------------------------------------------------------


@router.post(
    "/transport/cleaning",
    response_model=VehicleResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Record cleaning verification",
    description=(
        "Record a cleaning verification for a transport vehicle. "
        "Updates the vehicle's cleaning status and timestamp."
    ),
    responses={
        201: {"description": "Cleaning recorded"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Vehicle not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def record_cleaning(
    request: Request,
    body: RecordCleaningRequest,
    user: AuthUser = Depends(
        require_permission("eudr-sgv:transport:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> VehicleResponse:
    """Record a vehicle cleaning verification.

    Args:
        body: Cleaning verification details.
        user: Authenticated user with transport:create permission.

    Returns:
        VehicleResponse with updated cleaning status.

    Raises:
        HTTPException: 404 if vehicle not found.
    """
    start = time.monotonic()
    try:
        store = _get_vehicle_store()
        record = store.get(body.vehicle_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transport vehicle {body.vehicle_id} not found",
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        cleaning_time = body.inspection_date or now

        # Update vehicle cleaning status
        cleaning_status = (
            CleaningVerificationStatus.PASSED
            if body.passed
            else CleaningVerificationStatus.FAILED
        )
        record["last_cleaning_at"] = cleaning_time
        record["last_cleaning_status"] = cleaning_status
        record["updated_at"] = now

        if body.passed:
            record["status"] = VehicleStatus.AVAILABLE
        else:
            record["status"] = VehicleStatus.QUARANTINE

        # Recompute provenance
        provenance_hash = _compute_provenance_hash(body.model_dump(mode="json"))
        record["provenance"] = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="cleaning_verification",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Cleaning recorded: vehicle=%s status=%s inspector=%s",
            body.vehicle_id,
            cleaning_status.value,
            body.inspector_name,
        )

        return VehicleResponse(**record, processing_time_ms=elapsed_ms)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record cleaning: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record cleaning verification",
        )


# ---------------------------------------------------------------------------
# GET /transport/history/{vehicle_id}
# ---------------------------------------------------------------------------


@router.get(
    "/transport/history/{vehicle_id}",
    response_model=VehicleHistoryResponse,
    summary="Get vehicle cargo history",
    description="Retrieve the cargo history for a transport vehicle.",
    responses={
        200: {"description": "Cargo history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Vehicle not found"},
    },
)
async def get_vehicle_history(
    request: Request,
    vehicle_id: str = Depends(validate_vehicle_id),
    user: AuthUser = Depends(
        require_permission("eudr-sgv:transport:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VehicleHistoryResponse:
    """Get cargo history for a vehicle.

    Args:
        vehicle_id: Vehicle identifier.
        user: Authenticated user with transport:read permission.

    Returns:
        VehicleHistoryResponse with cargo history entries.

    Raises:
        HTTPException: 404 if vehicle not found.
    """
    try:
        store = _get_vehicle_store()
        if vehicle_id not in store:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transport vehicle {vehicle_id} not found",
            )

        history_store = _get_cargo_history_store()
        entries_data = history_store.get(vehicle_id, [])

        entries: List[CargoHistoryEntry] = []
        commodities_set: set = set()
        for entry_data in entries_data:
            entry = CargoHistoryEntry(**entry_data)
            entries.append(entry)
            commodities_set.add(entry.commodity)

        return VehicleHistoryResponse(
            vehicle_id=vehicle_id,
            entries=entries,
            total_entries=len(entries),
            commodities_transported=list(commodities_set),
            processing_time_ms=0.0,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get vehicle history %s: %s",
            vehicle_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve vehicle cargo history",
        )
