"""
Due Diligence Statement (DDS) API Routes for GL-EUDR-APP v1.0

Manages the lifecycle of EU Deforestation Regulation Due Diligence
Statements: generation, validation, submission to the EU Information
System (simulated), amendments, downloads (JSON/XML), and annual
summaries.

Reference format: EUDR-{country_iso3}-{year}-{sequence:06d}

Prefix: /api/v1/dds
Tags: Due Diligence Statements
"""

import uuid
import math
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/dds", tags=["Due Diligence Statements"])

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DDSStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"


class DownloadFormat(str, Enum):
    JSON = "json"
    XML = "xml"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class DDSGenerateRequest(BaseModel):
    """Request to generate a new Due Diligence Statement.

    Example::

        {
            "supplier_id": "sup_abc123",
            "commodity": "soya",
            "year": 2025,
            "plot_ids": ["plot_001", "plot_002"]
        }
    """

    supplier_id: str = Field(..., description="Supplier ID")
    commodity: str = Field(..., description="EUDR-regulated commodity")
    year: int = Field(..., ge=2020, le=2050, description="Reporting year")
    plot_ids: List[str] = Field(
        ..., min_length=1, description="Plot IDs included in this DDS"
    )
    operator_name: Optional[str] = Field(None, description="EU operator / trader name")
    operator_country: Optional[str] = Field(None, description="EU operator country code")


class DDSAmendRequest(BaseModel):
    """Request to amend a submitted DDS."""

    reason: str = Field(..., min_length=1, max_length=1000, description="Reason for amendment")
    updated_plot_ids: Optional[List[str]] = Field(
        None, description="Revised plot IDs (if changed)"
    )
    updated_commodity: Optional[str] = Field(None, description="Revised commodity")
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")


class BulkDDSGenerateRequest(BaseModel):
    """Request to generate DDS for multiple suppliers at once.

    Example::

        {
            "requests": [
                {"supplier_id": "sup_abc", "commodity": "soya", "year": 2025, "plot_ids": ["p1"]},
                {"supplier_id": "sup_def", "commodity": "cocoa", "year": 2025, "plot_ids": ["p2"]}
            ]
        }
    """

    requests: List[DDSGenerateRequest] = Field(
        ..., min_length=1, max_length=100, description="List of DDS generation requests"
    )


class DDSValidationIssue(BaseModel):
    """A single validation issue found in a DDS."""

    field: str
    severity: str = Field(..., description="error | warning")
    message: str


class DDSResponse(BaseModel):
    """Response model for a single DDS."""

    dds_id: str = Field(..., description="Internal DDS identifier")
    reference: str = Field(
        ..., description="Official reference: EUDR-{country}-{year}-{seq}"
    )
    supplier_id: str
    commodity: str
    year: int
    plot_ids: List[str]
    operator_name: Optional[str] = None
    operator_country: Optional[str] = None
    status: DDSStatus
    completeness_score: float = Field(
        ..., ge=0, le=100, description="Completeness percentage"
    )
    validation_issues: List[DDSValidationIssue] = Field(default_factory=list)
    submitted_at: Optional[datetime] = None
    eu_reference_number: Optional[str] = Field(
        None, description="Reference assigned by EU Information System"
    )
    amendment_history: List[Dict] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class DDSListResponse(BaseModel):
    """Paginated list of DDS records."""

    items: List[DDSResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class DDSValidationResponse(BaseModel):
    """Result of DDS completeness validation."""

    dds_id: str
    reference: str
    is_complete: bool
    completeness_score: float
    issues: List[DDSValidationIssue]
    ready_to_submit: bool


class DDSSubmitResponse(BaseModel):
    """Result of DDS submission to EU system (simulated)."""

    dds_id: str
    reference: str
    status: DDSStatus
    eu_reference_number: str
    submitted_at: datetime
    message: str


class BulkDDSGenerateResponse(BaseModel):
    """Response for bulk DDS generation."""

    total_requested: int
    total_created: int
    total_failed: int
    created: List[DDSResponse]
    errors: List[Dict]


class DDSAnnualSummary(BaseModel):
    """Annual DDS summary statistics."""

    year: int
    total_dds: int
    by_status: Dict[str, int]
    by_commodity: Dict[str, int]
    average_completeness: float
    submitted_count: int
    accepted_count: int
    rejected_count: int


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_dds_store: Dict[str, dict] = {}
_dds_sequence: Dict[str, int] = {}  # key: "{country}-{year}" -> counter


def _next_reference(country: str, year: int) -> str:
    """Generate next DDS reference number."""
    key = f"{country.upper()}-{year}"
    seq = _dds_sequence.get(key, 0) + 1
    _dds_sequence[key] = seq
    return f"EUDR-{country.upper()}-{year}-{seq:06d}"


def _compute_completeness(record: dict) -> tuple:
    """Compute DDS completeness score and issues."""
    issues: List[DDSValidationIssue] = []
    checks = 0
    passed = 0

    # Check supplier_id
    checks += 1
    if record.get("supplier_id"):
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="supplier_id", severity="error", message="Supplier ID is required"
        ))

    # Check commodity
    checks += 1
    allowed = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
    if record.get("commodity") and record["commodity"].lower() in allowed:
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="commodity", severity="error", message="Valid EUDR commodity required"
        ))

    # Check plot_ids
    checks += 1
    if record.get("plot_ids") and len(record["plot_ids"]) > 0:
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="plot_ids", severity="error", message="At least one plot must be linked"
        ))

    # Check operator_name
    checks += 1
    if record.get("operator_name"):
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="operator_name", severity="warning",
            message="Operator name recommended for submission",
        ))

    # Check operator_country
    checks += 1
    if record.get("operator_country"):
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="operator_country", severity="warning",
            message="Operator country recommended for submission",
        ))

    # Check year
    checks += 1
    if record.get("year") and 2020 <= record["year"] <= 2050:
        passed += 1
    else:
        issues.append(DDSValidationIssue(
            field="year", severity="error", message="Year must be between 2020 and 2050"
        ))

    score = round((passed / checks) * 100, 1) if checks > 0 else 0.0
    return score, issues


def _build_dds_response(data: dict) -> DDSResponse:
    return DDSResponse(**data)


def _dds_to_xml(record: dict) -> str:
    """Convert DDS record to basic XML representation."""
    plots_xml = "\n".join(
        f"      <PlotId>{pid}</PlotId>" for pid in record.get("plot_ids", [])
    )
    amendments_xml = ""
    for amend in record.get("amendment_history", []):
        amendments_xml += f"""
    <Amendment>
      <Reason>{amend.get('reason', '')}</Reason>
      <Date>{amend.get('date', '')}</Date>
    </Amendment>"""

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<DueDiligenceStatement xmlns="urn:eu:eudr:dds:v1">
  <DDSId>{record['dds_id']}</DDSId>
  <Reference>{record['reference']}</Reference>
  <SupplierId>{record['supplier_id']}</SupplierId>
  <Commodity>{record['commodity']}</Commodity>
  <Year>{record['year']}</Year>
  <Status>{record['status']}</Status>
  <CompletenessScore>{record['completeness_score']}</CompletenessScore>
  <OperatorName>{record.get('operator_name', '')}</OperatorName>
  <OperatorCountry>{record.get('operator_country', '')}</OperatorCountry>
  <Plots>
{plots_xml}
  </Plots>
  <EUReference>{record.get('eu_reference_number', '')}</EUReference>
  <SubmittedAt>{record.get('submitted_at', '')}</SubmittedAt>
  <CreatedAt>{record['created_at']}</CreatedAt>
  <UpdatedAt>{record['updated_at']}</UpdatedAt>
  <Amendments>{amendments_xml}
  </Amendments>
</DueDiligenceStatement>"""
    return xml


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=DDSResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate DDS",
    description="Generate a new Due Diligence Statement for a supplier and commodity.",
)
async def generate_dds(body: DDSGenerateRequest) -> DDSResponse:
    """
    Generate a new DDS in DRAFT status.

    Assigns a unique reference in the format EUDR-{country}-{year}-{seq}.
    Computes an initial completeness score.

    Returns:
        201 with generated DDS record.
    """
    now = datetime.now(timezone.utc)
    dds_id = f"dds_{uuid.uuid4().hex[:12]}"

    # Derive country from operator or default to supplier context
    country = (body.operator_country or "EU").upper()[:3]
    reference = _next_reference(country, body.year)

    record = {
        "dds_id": dds_id,
        "reference": reference,
        "supplier_id": body.supplier_id,
        "commodity": body.commodity.lower(),
        "year": body.year,
        "plot_ids": body.plot_ids,
        "operator_name": body.operator_name,
        "operator_country": body.operator_country,
        "status": DDSStatus.DRAFT,
        "completeness_score": 0.0,
        "validation_issues": [],
        "submitted_at": None,
        "eu_reference_number": None,
        "amendment_history": [],
        "created_at": now,
        "updated_at": now,
    }

    score, issues = _compute_completeness(record)
    record["completeness_score"] = score
    record["validation_issues"] = [i.model_dump() for i in issues]

    _dds_store[dds_id] = record
    logger.info("DDS generated: %s (ref=%s)", dds_id, reference)
    return _build_dds_response(record)


@router.get(
    "/{dds_id}",
    response_model=DDSResponse,
    summary="Get DDS",
    description="Retrieve a Due Diligence Statement by its identifier.",
)
async def get_dds(dds_id: str) -> DDSResponse:
    """
    Fetch a single DDS by ID.

    Returns:
        200 with DDS record.

    Raises:
        404 if DDS not found.
    """
    record = _dds_store.get(dds_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DDS '{dds_id}' not found",
        )
    return _build_dds_response(record)


@router.get(
    "/",
    response_model=DDSListResponse,
    summary="List DDS",
    description="List Due Diligence Statements with filtering and pagination.",
)
async def list_dds(
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    dds_status: Optional[DDSStatus] = Query(None, alias="status", description="Filter by status"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    year: Optional[int] = Query(None, description="Filter by reporting year"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> DDSListResponse:
    """
    Retrieve a paginated list of DDS records.

    Returns:
        200 with paginated DDS list.
    """
    results = list(_dds_store.values())

    if supplier_id:
        results = [d for d in results if d["supplier_id"] == supplier_id]
    if dds_status:
        results = [d for d in results if d["status"] == dds_status.value]
    if commodity:
        results = [d for d in results if d["commodity"] == commodity.lower()]
    if year:
        results = [d for d in results if d["year"] == year]

    results.sort(key=lambda d: d["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    page_items = results[start : start + limit]

    return DDSListResponse(
        items=[_build_dds_response(d) for d in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.post(
    "/{dds_id}/validate",
    response_model=DDSValidationResponse,
    summary="Validate DDS",
    description="Validate DDS completeness and readiness for submission.",
)
async def validate_dds(dds_id: str) -> DDSValidationResponse:
    """
    Run completeness checks on a DDS.

    Evaluates required fields, commodity validity, plot linkage, and
    operator information. Updates stored completeness_score.

    Returns:
        200 with validation results.

    Raises:
        404 if DDS not found.
    """
    record = _dds_store.get(dds_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DDS '{dds_id}' not found",
        )

    score, issues = _compute_completeness(record)
    record["completeness_score"] = score
    record["validation_issues"] = [i.model_dump() for i in issues]

    has_errors = any(i.severity == "error" for i in issues)
    is_complete = score >= 100.0 and not has_errors
    ready = score >= 80.0 and not has_errors

    if is_complete and record["status"] == DDSStatus.DRAFT:
        record["status"] = DDSStatus.VALIDATED

    record["updated_at"] = datetime.now(timezone.utc)

    return DDSValidationResponse(
        dds_id=dds_id,
        reference=record["reference"],
        is_complete=is_complete,
        completeness_score=score,
        issues=issues,
        ready_to_submit=ready,
    )


@router.post(
    "/{dds_id}/submit",
    response_model=DDSSubmitResponse,
    summary="Submit DDS",
    description="Submit DDS to the EU Information System (simulated for v1.0).",
)
async def submit_dds(dds_id: str) -> DDSSubmitResponse:
    """
    Submit a DDS to the EU EUDR Information System.

    In v1.0, this is simulated: an EU reference number is generated and
    the status transitions to SUBMITTED. Production integration would
    call the actual EU API.

    Returns:
        200 with submission confirmation.

    Raises:
        400 if DDS is not ready (completeness < 80% or has errors).
        404 if DDS not found.
    """
    record = _dds_store.get(dds_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DDS '{dds_id}' not found",
        )

    if record["status"] in (DDSStatus.SUBMITTED, DDSStatus.ACCEPTED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DDS already in status '{record['status']}'; cannot re-submit",
        )

    # Validate first
    score, issues = _compute_completeness(record)
    has_errors = any(i.severity == "error" for i in issues)
    if has_errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DDS has validation errors and cannot be submitted. Score: {score}",
        )

    now = datetime.now(timezone.utc)
    eu_ref = f"EU-{uuid.uuid4().hex[:8].upper()}"

    record["status"] = DDSStatus.SUBMITTED
    record["submitted_at"] = now
    record["eu_reference_number"] = eu_ref
    record["completeness_score"] = score
    record["validation_issues"] = [i.model_dump() for i in issues]
    record["updated_at"] = now

    logger.info("DDS submitted: %s -> EU ref %s", dds_id, eu_ref)

    return DDSSubmitResponse(
        dds_id=dds_id,
        reference=record["reference"],
        status=DDSStatus.SUBMITTED,
        eu_reference_number=eu_ref,
        submitted_at=now,
        message="DDS submitted to EU Information System successfully (simulated)",
    )


@router.get(
    "/{dds_id}/download",
    summary="Download DDS",
    description="Download a DDS in JSON or XML format.",
)
async def download_dds(
    dds_id: str,
    format: DownloadFormat = Query(DownloadFormat.JSON, description="Download format: json or xml"),
):
    """
    Download a DDS in the requested format.

    Returns:
        200 with DDS content as JSON or XML.

    Raises:
        404 if DDS not found.
    """
    record = _dds_store.get(dds_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DDS '{dds_id}' not found",
        )

    if format == DownloadFormat.XML:
        xml_content = _dds_to_xml(record)
        return JSONResponse(
            content={"format": "xml", "content": xml_content},
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{record["reference"]}.xml"'
            },
        )

    # JSON format
    response_data = _build_dds_response(record).model_dump(mode="json")
    return JSONResponse(
        content={"format": "json", "content": response_data},
        headers={
            "Content-Disposition": f'attachment; filename="{record["reference"]}.json"'
        },
    )


@router.post(
    "/{dds_id}/amend",
    response_model=DDSResponse,
    summary="Amend DDS",
    description="Amend a previously submitted DDS with updated information.",
)
async def amend_dds(dds_id: str, body: DDSAmendRequest) -> DDSResponse:
    """
    Amend a submitted DDS.

    Creates an amendment record, updates the DDS fields, and sets
    status to AMENDED. The original submission data is preserved in
    the amendment history.

    Returns:
        200 with amended DDS record.

    Raises:
        400 if DDS has not been submitted yet.
        404 if DDS not found.
    """
    record = _dds_store.get(dds_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DDS '{dds_id}' not found",
        )

    if record["status"] not in (DDSStatus.SUBMITTED, DDSStatus.ACCEPTED, DDSStatus.REJECTED):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only submitted, accepted, or rejected DDS can be amended",
        )

    now = datetime.now(timezone.utc)
    amendment = {
        "amendment_id": f"amend_{uuid.uuid4().hex[:8]}",
        "reason": body.reason,
        "notes": body.notes,
        "previous_status": record["status"],
        "previous_plot_ids": record["plot_ids"][:],
        "previous_commodity": record["commodity"],
        "date": now.isoformat(),
    }

    if body.updated_plot_ids is not None:
        record["plot_ids"] = body.updated_plot_ids
    if body.updated_commodity is not None:
        record["commodity"] = body.updated_commodity.lower()

    record["amendment_history"].append(amendment)
    record["status"] = DDSStatus.AMENDED
    record["updated_at"] = now

    # Re-compute completeness
    score, issues = _compute_completeness(record)
    record["completeness_score"] = score
    record["validation_issues"] = [i.model_dump() for i in issues]

    logger.info("DDS amended: %s (reason: %s)", dds_id, body.reason)
    return _build_dds_response(record)


@router.post(
    "/bulk-generate",
    response_model=BulkDDSGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk generate DDS",
    description="Generate Due Diligence Statements for multiple suppliers in one request.",
)
async def bulk_generate_dds(body: BulkDDSGenerateRequest) -> BulkDDSGenerateResponse:
    """
    Bulk generate DDS for multiple suppliers.

    Each request is processed independently; failures do not block
    successful entries.

    Returns:
        201 with bulk generation summary.
    """
    created: List[DDSResponse] = []
    errors: List[Dict] = []

    for idx, req in enumerate(body.requests):
        try:
            now = datetime.now(timezone.utc)
            dds_id = f"dds_{uuid.uuid4().hex[:12]}"
            country = (req.operator_country or "EU").upper()[:3]
            reference = _next_reference(country, req.year)

            record = {
                "dds_id": dds_id,
                "reference": reference,
                "supplier_id": req.supplier_id,
                "commodity": req.commodity.lower(),
                "year": req.year,
                "plot_ids": req.plot_ids,
                "operator_name": req.operator_name,
                "operator_country": req.operator_country,
                "status": DDSStatus.DRAFT,
                "completeness_score": 0.0,
                "validation_issues": [],
                "submitted_at": None,
                "eu_reference_number": None,
                "amendment_history": [],
                "created_at": now,
                "updated_at": now,
            }

            score, issues = _compute_completeness(record)
            record["completeness_score"] = score
            record["validation_issues"] = [i.model_dump() for i in issues]

            _dds_store[dds_id] = record
            created.append(_build_dds_response(record))

        except Exception as exc:
            errors.append({
                "index": idx,
                "supplier_id": req.supplier_id,
                "error": str(exc),
            })

    logger.info(
        "Bulk DDS generation: %d created, %d failed out of %d",
        len(created),
        len(errors),
        len(body.requests),
    )

    return BulkDDSGenerateResponse(
        total_requested=len(body.requests),
        total_created=len(created),
        total_failed=len(errors),
        created=created,
        errors=errors,
    )


@router.get(
    "/summary/{year}",
    response_model=DDSAnnualSummary,
    summary="Annual DDS summary",
    description="Get annual summary statistics for Due Diligence Statements.",
)
async def get_annual_summary(year: int) -> DDSAnnualSummary:
    """
    Aggregate DDS statistics for the given year.

    Returns:
        200 with annual summary including counts by status and commodity.
    """
    year_records = [d for d in _dds_store.values() if d["year"] == year]

    by_status: Dict[str, int] = {}
    by_commodity: Dict[str, int] = {}
    total_score = 0.0

    for d in year_records:
        s = d["status"] if isinstance(d["status"], str) else d["status"].value
        by_status[s] = by_status.get(s, 0) + 1
        by_commodity[d["commodity"]] = by_commodity.get(d["commodity"], 0) + 1
        total_score += d.get("completeness_score", 0.0)

    avg_completeness = round(total_score / len(year_records), 1) if year_records else 0.0

    return DDSAnnualSummary(
        year=year,
        total_dds=len(year_records),
        by_status=by_status,
        by_commodity=by_commodity,
        average_completeness=avg_completeness,
        submitted_count=by_status.get("submitted", 0),
        accepted_count=by_status.get("accepted", 0),
        rejected_count=by_status.get("rejected", 0),
    )
