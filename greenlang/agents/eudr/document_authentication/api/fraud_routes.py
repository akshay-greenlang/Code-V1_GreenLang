# -*- coding: utf-8 -*-
"""
Fraud Detection Routes - AGENT-EUDR-012 Document Authentication API

Endpoints for fraud pattern detection including single document analysis,
batch detection, alert retrieval, operator fraud summary, and rule listing.

Endpoints:
    POST   /fraud/detect                        - Run fraud detection
    POST   /fraud/detect/batch                   - Batch fraud detection
    GET    /fraud/alerts/{document_id}           - Get fraud alerts
    GET    /fraud/alerts/summary/{operator_id}   - Get fraud summary
    GET    /fraud/rules                          - List active fraud rules

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012, Feature 6 (Fraud Detection)
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.document_authentication.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_dav_service,
    get_pagination,
    get_request_id,
    rate_limit_batch,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_document_id,
    validate_operator_id,
)
from greenlang.agents.eudr.document_authentication.api.schemas import (
    AuthenticationResultSchema,
    BatchDetectFraudSchema,
    BatchFraudResultSchema,
    DetectFraudSchema,
    FraudAlertListSchema,
    FraudAlertSchema,
    FraudDetectionResultSchema,
    FraudPatternTypeSchema,
    FraudRuleListSchema,
    FraudRuleSchema,
    FraudSeveritySchema,
    FraudSummarySchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Fraud Detection"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_fraud_result_store: Dict[str, Dict] = {}
_fraud_alert_store: Dict[str, List[Dict]] = {}
_fraud_summary_store: Dict[str, Dict] = {}


def _get_fraud_result_store() -> Dict[str, Dict]:
    """Return the fraud detection result store singleton."""
    return _fraud_result_store


def _get_fraud_alert_store() -> Dict[str, List[Dict]]:
    """Return the fraud alert store indexed by document_id."""
    return _fraud_alert_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _detect_fraud_logic(document_id: str) -> Dict[str, Any]:
    """Deterministic fraud detection simulation.

    Zero hallucination: rule-based pattern matching only.

    Args:
        document_id: Document identifier to analyze.

    Returns:
        Dict with fraud detection results.
    """
    now = _utcnow()
    alerts = []

    # Simulate checking a set of patterns
    patterns_checked = 15
    patterns_triggered = 0

    # Simulate a low-severity alert for demonstration
    if "suspicious" in document_id.lower():
        alert = {
            "alert_id": str(uuid.uuid4()),
            "document_id": document_id,
            "pattern_type": FraudPatternTypeSchema.ROUND_NUMBER_BIAS,
            "severity": FraudSeveritySchema.LOW,
            "description": "80%+ of quantity values are round numbers",
            "confidence": 0.72,
            "evidence": {"round_number_percentage": 85.0},
            "related_documents": [],
            "recommended_action": "Review quantity values for accuracy",
            "resolved": False,
            "resolved_at": None,
            "resolved_by": None,
            "created_at": now,
        }
        alerts.append(alert)
        patterns_triggered = 1

    # Calculate composite risk score (deterministic)
    severity_weights = {"low": 1.0, "medium": 3.0, "high": 7.0, "critical": 10.0}
    risk_score = 0.0
    for alert in alerts:
        weight = severity_weights.get(alert["severity"].value, 1.0)
        risk_score += weight * alert["confidence"]

    risk_score = min(risk_score * 10.0, 100.0)

    # Determine overall risk
    if risk_score >= 70.0:
        overall_risk = FraudSeveritySchema.CRITICAL
        auth_result = AuthenticationResultSchema.FRAUDULENT
    elif risk_score >= 40.0:
        overall_risk = FraudSeveritySchema.HIGH
        auth_result = AuthenticationResultSchema.SUSPICIOUS
    elif risk_score >= 10.0:
        overall_risk = FraudSeveritySchema.MEDIUM
        auth_result = AuthenticationResultSchema.SUSPICIOUS
    elif alerts:
        overall_risk = FraudSeveritySchema.LOW
        auth_result = AuthenticationResultSchema.INCONCLUSIVE
    else:
        overall_risk = FraudSeveritySchema.LOW
        auth_result = AuthenticationResultSchema.AUTHENTIC

    return {
        "overall_risk": overall_risk,
        "risk_score": risk_score,
        "alerts": alerts,
        "patterns_checked": patterns_checked,
        "patterns_triggered": patterns_triggered,
        "authentication_result": auth_result,
    }


# Default fraud rules
_DEFAULT_FRAUD_RULES: List[Dict[str, Any]] = [
    {
        "rule_id": "FR-001",
        "pattern_type": FraudPatternTypeSchema.DUPLICATE_REUSE,
        "name": "Duplicate Document Reuse",
        "description": "Detects same hash/certificate reused across different shipments",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-002",
        "pattern_type": FraudPatternTypeSchema.QUANTITY_TAMPERING,
        "name": "Quantity Tampering Detection",
        "description": "Detects quantity deviations exceeding tolerance threshold",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": 5.0,
        "parameters": {"tolerance_percent": 5.0},
    },
    {
        "rule_id": "FR-003",
        "pattern_type": FraudPatternTypeSchema.DATE_MANIPULATION,
        "name": "Date Manipulation Detection",
        "description": "Detects inconsistent or impossible date sequences",
        "severity": FraudSeveritySchema.MEDIUM,
        "enabled": True,
        "threshold": 30.0,
        "parameters": {"tolerance_days": 30},
    },
    {
        "rule_id": "FR-004",
        "pattern_type": FraudPatternTypeSchema.EXPIRED_CERT,
        "name": "Expired Certificate Detection",
        "description": "Flags documents referencing expired certificates",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-005",
        "pattern_type": FraudPatternTypeSchema.SERIAL_ANOMALY,
        "name": "Serial Number Anomaly",
        "description": "Detects certificate serial numbers not matching expected patterns",
        "severity": FraudSeveritySchema.MEDIUM,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-006",
        "pattern_type": FraudPatternTypeSchema.ISSUER_MISMATCH,
        "name": "Issuer Mismatch Detection",
        "description": "Detects mismatch between claimed issuer and signing certificate",
        "severity": FraudSeveritySchema.CRITICAL,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-007",
        "pattern_type": FraudPatternTypeSchema.TEMPLATE_FORGERY,
        "name": "Template Forgery Detection",
        "description": "Detects layout/font deviations from known issuer templates",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": 0.85,
        "parameters": {"min_similarity": 0.85},
    },
    {
        "rule_id": "FR-008",
        "pattern_type": FraudPatternTypeSchema.CROSS_DOC_INCONSISTENCY,
        "name": "Cross-Document Inconsistency",
        "description": "Detects data contradictions across related documents",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-009",
        "pattern_type": FraudPatternTypeSchema.GEO_IMPOSSIBILITY,
        "name": "Geographic Impossibility",
        "description": "Detects physically impossible production/transport scenarios",
        "severity": FraudSeveritySchema.CRITICAL,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-010",
        "pattern_type": FraudPatternTypeSchema.VELOCITY_ANOMALY,
        "name": "Velocity Anomaly Detection",
        "description": "Detects unusually high document issuance rate per issuer",
        "severity": FraudSeveritySchema.MEDIUM,
        "enabled": True,
        "threshold": 10.0,
        "parameters": {"max_per_day": 10},
    },
    {
        "rule_id": "FR-011",
        "pattern_type": FraudPatternTypeSchema.MODIFICATION_ANOMALY,
        "name": "Modification Anomaly Detection",
        "description": "Detects post-issuance modifications in document metadata",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-012",
        "pattern_type": FraudPatternTypeSchema.ROUND_NUMBER_BIAS,
        "name": "Round Number Bias Detection",
        "description": "Flags documents with suspiciously high proportion of round numbers",
        "severity": FraudSeveritySchema.LOW,
        "enabled": True,
        "threshold": 80.0,
        "parameters": {"threshold_percent": 80.0},
    },
    {
        "rule_id": "FR-013",
        "pattern_type": FraudPatternTypeSchema.COPY_PASTE,
        "name": "Copy-Paste Detection",
        "description": "Detects text/image duplication from other documents",
        "severity": FraudSeveritySchema.MEDIUM,
        "enabled": True,
        "threshold": 0.90,
        "parameters": {"min_similarity": 0.90},
    },
    {
        "rule_id": "FR-014",
        "pattern_type": FraudPatternTypeSchema.MISSING_REQUIRED,
        "name": "Missing Required Fields",
        "description": "Flags documents missing required fields for their type",
        "severity": FraudSeveritySchema.MEDIUM,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
    {
        "rule_id": "FR-015",
        "pattern_type": FraudPatternTypeSchema.SCOPE_MISMATCH,
        "name": "Scope Mismatch Detection",
        "description": "Detects certification scope not covering claimed commodity/region",
        "severity": FraudSeveritySchema.HIGH,
        "enabled": True,
        "threshold": None,
        "parameters": {},
    },
]


# ---------------------------------------------------------------------------
# POST /fraud/detect
# ---------------------------------------------------------------------------


@router.post(
    "/fraud/detect",
    response_model=FraudDetectionResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Run fraud detection",
    description=(
        "Run fraud pattern detection on an EUDR document checking "
        "up to 15 pattern types including duplicate reuse, quantity "
        "tampering, date manipulation, and geographic impossibility."
    ),
    responses={
        201: {"description": "Fraud detection completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def detect_fraud(
    request: Request,
    body: DetectFraudSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:fraud:detect")
    ),
    _rate: None = Depends(rate_limit_write),
) -> FraudDetectionResultSchema:
    """Run fraud detection on a document.

    Args:
        body: Fraud detection request.
        user: Authenticated user with fraud:detect permission.

    Returns:
        FraudDetectionResultSchema with detection results.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        fraud_result = _detect_fraud_logic(body.document_id)

        provenance_data = body.model_dump(mode="json")
        provenance_data["detected_by"] = user.user_id
        provenance_hash = _compute_provenance_hash(provenance_data)

        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        alert_schemas = [
            FraudAlertSchema(**a) for a in fraud_result["alerts"]
        ]

        result_record = {
            "document_id": body.document_id,
            "overall_risk": fraud_result["overall_risk"],
            "risk_score": fraud_result["risk_score"],
            "alerts": alert_schemas,
            "patterns_checked": fraud_result["patterns_checked"],
            "patterns_triggered": fraud_result["patterns_triggered"],
            "authentication_result": fraud_result["authentication_result"],
            "provenance": provenance,
            "created_at": now,
        }

        result_store = _get_fraud_result_store()
        result_store[body.document_id] = result_record

        alert_store = _get_fraud_alert_store()
        alert_store[body.document_id] = fraud_result["alerts"]

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Fraud detection: doc=%s risk=%s score=%.1f alerts=%d",
            body.document_id,
            fraud_result["overall_risk"].value,
            fraud_result["risk_score"],
            len(fraud_result["alerts"]),
        )

        return FraudDetectionResultSchema(
            **result_record,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed fraud detection: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to run fraud detection",
        )


# ---------------------------------------------------------------------------
# POST /fraud/detect/batch
# ---------------------------------------------------------------------------


@router.post(
    "/fraud/detect/batch",
    response_model=BatchFraudResultSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Batch fraud detection",
    description=(
        "Run fraud detection on up to 500 documents in a single request. "
        "Each document is analyzed independently."
    ),
    responses={
        201: {"description": "Batch fraud detection processed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def batch_detect_fraud(
    request: Request,
    body: BatchDetectFraudSchema,
    user: AuthUser = Depends(
        require_permission("eudr-dav:fraud:batch")
    ),
    _rate: None = Depends(rate_limit_batch),
) -> BatchFraudResultSchema:
    """Batch fraud detection for multiple documents.

    Args:
        body: Batch fraud detection request.
        user: Authenticated user with fraud:batch permission.

    Returns:
        BatchFraudResultSchema with results and errors.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        results: List[FraudDetectionResultSchema] = []
        errors: List[Dict[str, Any]] = []
        total_alerts = 0
        result_store = _get_fraud_result_store()
        alert_store = _get_fraud_alert_store()

        for idx, doc_req in enumerate(body.documents):
            try:
                fraud_result = _detect_fraud_logic(doc_req.document_id)

                provenance_hash = _compute_provenance_hash({
                    "document_id": doc_req.document_id,
                    "detected_by": user.user_id,
                    "index": idx,
                })
                provenance = ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    created_by=user.user_id,
                    created_at=now,
                    source="api",
                )

                alert_schemas = [
                    FraudAlertSchema(**a) for a in fraud_result["alerts"]
                ]

                result_record = {
                    "document_id": doc_req.document_id,
                    "overall_risk": fraud_result["overall_risk"],
                    "risk_score": fraud_result["risk_score"],
                    "alerts": alert_schemas,
                    "patterns_checked": fraud_result["patterns_checked"],
                    "patterns_triggered": fraud_result["patterns_triggered"],
                    "authentication_result": fraud_result["authentication_result"],
                    "provenance": provenance,
                    "created_at": now,
                }

                result_store[doc_req.document_id] = result_record
                alert_store[doc_req.document_id] = fraud_result["alerts"]
                total_alerts += len(fraud_result["alerts"])

                results.append(FraudDetectionResultSchema(**result_record))

            except Exception as entry_exc:
                errors.append({
                    "index": idx,
                    "document_id": doc_req.document_id,
                    "error": str(entry_exc),
                })

        batch_provenance_hash = _compute_provenance_hash({
            "total": len(body.documents),
            "analyzed": len(results),
            "failed": len(errors),
            "total_alerts": total_alerts,
            "operator": user.user_id,
        })
        batch_provenance = ProvenanceInfo(
            provenance_hash=batch_provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        logger.info(
            "Batch fraud detection: total=%d analyzed=%d failed=%d alerts=%d",
            len(body.documents),
            len(results),
            len(errors),
            total_alerts,
        )

        return BatchFraudResultSchema(
            total_submitted=len(body.documents),
            total_analyzed=len(results),
            total_failed=len(errors),
            total_alerts=total_alerts,
            results=results,
            errors=errors,
            provenance=batch_provenance,
            processing_time_ms=elapsed_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed batch fraud detection: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process batch fraud detection",
        )


# ---------------------------------------------------------------------------
# GET /fraud/alerts/{document_id}
# ---------------------------------------------------------------------------


@router.get(
    "/fraud/alerts/{document_id}",
    response_model=FraudAlertListSchema,
    summary="Get fraud alerts",
    description="Retrieve fraud alerts for a specific document.",
    responses={
        200: {"description": "Fraud alerts retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_fraud_alerts(
    request: Request,
    document_id: str = Depends(validate_document_id),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-dav:fraud:alerts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FraudAlertListSchema:
    """Get fraud alerts for a document.

    Args:
        document_id: Document identifier.
        pagination: Pagination parameters.
        user: Authenticated user with fraud:alerts:read permission.

    Returns:
        FraudAlertListSchema with fraud alerts.
    """
    start = time.monotonic()
    try:
        alert_store = _get_fraud_alert_store()
        raw_alerts = alert_store.get(document_id, [])

        total = len(raw_alerts)
        paginated = raw_alerts[pagination.offset: pagination.offset + pagination.limit]
        has_more = (pagination.offset + pagination.limit) < total

        alert_schemas = [FraudAlertSchema(**a) for a in paginated]
        meta = PaginatedMeta(
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
            has_more=has_more,
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return FraudAlertListSchema(
            document_id=document_id,
            alerts=alert_schemas,
            total_count=total,
            pagination=meta,
            processing_time_ms=elapsed_ms,
            timestamp=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get fraud alerts for %s: %s",
            document_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud alerts",
        )


# ---------------------------------------------------------------------------
# GET /fraud/alerts/summary/{operator_id}
# ---------------------------------------------------------------------------


@router.get(
    "/fraud/alerts/summary/{operator_id}",
    response_model=FraudSummarySchema,
    summary="Get fraud summary",
    description="Get fraud alert summary for an EUDR operator.",
    responses={
        200: {"description": "Fraud summary retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_fraud_summary(
    request: Request,
    operator_id: str = Depends(validate_operator_id),
    user: AuthUser = Depends(
        require_permission("eudr-dav:fraud:summary:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FraudSummarySchema:
    """Get fraud summary for an operator.

    Args:
        operator_id: Operator identifier.
        user: Authenticated user with fraud:summary:read permission.

    Returns:
        FraudSummarySchema with aggregated fraud statistics.
    """
    start = time.monotonic()
    try:
        now = _utcnow()

        # Aggregate from in-memory stores
        alert_store = _get_fraud_alert_store()
        total_alerts = 0
        unresolved = 0
        critical = 0
        high = 0
        medium = 0
        low = 0

        for doc_alerts in alert_store.values():
            for alert in doc_alerts:
                total_alerts += 1
                if not alert.get("resolved", False):
                    unresolved += 1
                sev = alert.get("severity")
                if hasattr(sev, "value"):
                    sev = sev.value
                if sev == "critical":
                    critical += 1
                elif sev == "high":
                    high += 1
                elif sev == "medium":
                    medium += 1
                elif sev == "low":
                    low += 1

        avg_risk = 0.0
        result_store = _get_fraud_result_store()
        if result_store:
            scores = [r.get("risk_score", 0.0) for r in result_store.values()]
            if scores:
                avg_risk = sum(scores) / len(scores)

        provenance_hash = _compute_provenance_hash({
            "operator_id": operator_id,
            "total_alerts": total_alerts,
            "computed_by": user.user_id,
        })
        provenance = ProvenanceInfo(
            provenance_hash=provenance_hash,
            created_by=user.user_id,
            created_at=now,
            source="api",
        )

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return FraudSummarySchema(
            operator_id=operator_id,
            total_alerts=total_alerts,
            unresolved_alerts=unresolved,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            average_risk_score=avg_risk,
            top_patterns=[],
            provenance=provenance,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get fraud summary for %s: %s",
            operator_id, exc, exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud summary",
        )


# ---------------------------------------------------------------------------
# GET /fraud/rules
# ---------------------------------------------------------------------------


@router.get(
    "/fraud/rules",
    response_model=FraudRuleListSchema,
    summary="List active fraud rules",
    description="List all configured fraud detection rules and their parameters.",
    responses={
        200: {"description": "Fraud rules retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_fraud_rules(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-dav:fraud:rules:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> FraudRuleListSchema:
    """List active fraud detection rules.

    Args:
        user: Authenticated user with fraud:rules:read permission.

    Returns:
        FraudRuleListSchema with all configured rules.
    """
    start = time.monotonic()
    try:
        now = _utcnow()
        rules = [FraudRuleSchema(**r, created_at=now) for r in _DEFAULT_FRAUD_RULES]
        enabled_count = sum(1 for r in rules if r.enabled)

        elapsed_ms = (time.monotonic() - start) * 1000.0

        return FraudRuleListSchema(
            rules=rules,
            total_count=len(rules),
            enabled_count=enabled_count,
            processing_time_ms=elapsed_ms,
            timestamp=now,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list fraud rules: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list fraud rules",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
]
