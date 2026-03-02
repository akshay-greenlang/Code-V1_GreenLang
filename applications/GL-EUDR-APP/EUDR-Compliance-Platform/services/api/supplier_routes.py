"""
Supplier Management API Routes for GL-EUDR-APP v1.0

Provides CRUD operations for suppliers subject to EU Deforestation
Regulation (EUDR) compliance. Includes bulk import, compliance status
tracking, and risk summary endpoints.

Prefix: /api/v1/suppliers
Tags: Suppliers
"""

import uuid
import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/suppliers", tags=["Suppliers"])

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Address(BaseModel):
    """Physical address of a supplier facility."""

    street: Optional[str] = Field(None, description="Street address line")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State or province")
    postal_code: Optional[str] = Field(None, description="Postal / ZIP code")
    country: str = Field(..., description="ISO-3166 country code (2-letter)")


class SupplierCreateRequest(BaseModel):
    """Request body for creating a new supplier.

    Example::

        {
            "name": "Amazonia Timber Co.",
            "country": "BR",
            "tax_id": "12.345.678/0001-90",
            "commodities": ["timber", "soy"],
            "address": {"city": "Manaus", "country": "BR"}
        }
    """

    name: str = Field(..., min_length=1, max_length=255, description="Legal entity name")
    country: str = Field(
        ..., min_length=2, max_length=3, description="ISO country code (2 or 3-letter)"
    )
    tax_id: Optional[str] = Field(None, max_length=100, description="Tax identification number")
    commodities: List[str] = Field(
        ...,
        min_length=1,
        description="EUDR commodities: cattle, cocoa, coffee, oil_palm, rubber, soya, wood",
    )
    address: Optional[Address] = Field(None, description="Primary business address")

    @field_validator("commodities")
    @classmethod
    def validate_commodities(cls, v: List[str]) -> List[str]:
        allowed = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        for c in v:
            if c.lower() not in allowed:
                raise ValueError(
                    f"Invalid commodity '{c}'. Allowed: {sorted(allowed)}"
                )
        return [c.lower() for c in v]


class SupplierUpdateRequest(BaseModel):
    """Request body for updating an existing supplier."""

    name: Optional[str] = Field(None, min_length=1, max_length=255)
    country: Optional[str] = Field(None, min_length=2, max_length=3)
    tax_id: Optional[str] = Field(None, max_length=100)
    commodities: Optional[List[str]] = None
    address: Optional[Address] = None

    @field_validator("commodities")
    @classmethod
    def validate_commodities(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return v
        allowed = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        for c in v:
            if c.lower() not in allowed:
                raise ValueError(
                    f"Invalid commodity '{c}'. Allowed: {sorted(allowed)}"
                )
        return [c.lower() for c in v]


class SupplierResponse(BaseModel):
    """Response model for a single supplier record."""

    supplier_id: str = Field(..., description="Unique supplier identifier")
    name: str
    country: str
    tax_id: Optional[str] = None
    commodities: List[str]
    address: Optional[Address] = None
    compliance_status: str = Field(
        "pending", description="pending | compliant | non_compliant | under_review"
    )
    risk_level: str = Field("unknown", description="low | medium | high | critical | unknown")
    created_at: datetime
    updated_at: datetime


class SupplierListResponse(BaseModel):
    """Paginated list of suppliers."""

    items: List[SupplierResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class BulkImportRecord(BaseModel):
    """Single record in a bulk import payload."""

    name: str = Field(..., min_length=1, max_length=255)
    country: str = Field(..., min_length=2, max_length=3)
    tax_id: Optional[str] = None
    commodities: List[str]
    address: Optional[Address] = None


class BulkImportRequest(BaseModel):
    """Request body for bulk supplier import.

    Example::

        {
            "records": [
                {"name": "Supplier A", "country": "BR", "commodities": ["soya"]},
                {"name": "Supplier B", "country": "ID", "commodities": ["oil_palm"]}
            ]
        }
    """

    records: List[BulkImportRecord] = Field(
        ..., min_length=1, max_length=1000, description="Supplier records to import"
    )


class BulkImportResponse(BaseModel):
    """Response for bulk import operation."""

    total_submitted: int
    total_created: int
    total_failed: int
    created_ids: List[str]
    errors: List[Dict]


class ComplianceStatusResponse(BaseModel):
    """Compliance status details for a supplier."""

    supplier_id: str
    supplier_name: str
    compliance_status: str
    dds_submitted: int = Field(0, description="Number of DDS submitted")
    dds_pending: int = Field(0, description="Number of DDS pending")
    documents_verified: int = Field(0, description="Verified supporting documents")
    documents_pending: int = Field(0, description="Documents awaiting verification")
    last_assessment_date: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list, description="Outstanding compliance issues")


class RiskSummaryResponse(BaseModel):
    """Risk summary for a supplier."""

    supplier_id: str
    supplier_name: str
    overall_risk_level: str
    risk_score: float = Field(..., ge=0, le=100, description="Composite risk score 0-100")
    country_risk: str
    commodity_risks: Dict[str, str] = Field(
        default_factory=dict, description="Per-commodity risk levels"
    )
    deforestation_alerts: int = Field(0, description="Active deforestation alerts on plots")
    last_assessment_date: Optional[datetime] = None


class DeleteResponse(BaseModel):
    """Confirmation of a deletion."""

    supplier_id: str
    deleted: bool
    message: str


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_suppliers: Dict[str, dict] = {}


def _build_supplier_response(data: dict) -> SupplierResponse:
    """Convert internal dict to SupplierResponse model."""
    return SupplierResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/",
    response_model=SupplierResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create supplier",
    description="Register a new supplier in the EUDR compliance system.",
)
async def create_supplier(body: SupplierCreateRequest) -> SupplierResponse:
    """
    Create a new supplier record.

    - Validates commodities against EUDR-regulated list.
    - Assigns unique supplier_id and sets initial compliance_status to 'pending'.

    Returns:
        201 with created supplier record.

    Raises:
        422 if validation fails.
    """
    now = datetime.now(timezone.utc)
    supplier_id = f"sup_{uuid.uuid4().hex[:12]}"

    record = {
        "supplier_id": supplier_id,
        "name": body.name,
        "country": body.country.upper(),
        "tax_id": body.tax_id,
        "commodities": body.commodities,
        "address": body.address.model_dump() if body.address else None,
        "compliance_status": "pending",
        "risk_level": "unknown",
        "created_at": now,
        "updated_at": now,
    }
    _suppliers[supplier_id] = record
    logger.info("Supplier created: %s (%s)", supplier_id, body.name)
    return _build_supplier_response(record)


@router.get(
    "/{supplier_id}",
    response_model=SupplierResponse,
    summary="Get supplier",
    description="Retrieve a single supplier by its identifier.",
)
async def get_supplier(supplier_id: str) -> SupplierResponse:
    """
    Fetch supplier details by ID.

    Returns:
        200 with supplier record.

    Raises:
        404 if supplier not found.
    """
    record = _suppliers.get(supplier_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )
    return _build_supplier_response(record)


@router.put(
    "/{supplier_id}",
    response_model=SupplierResponse,
    summary="Update supplier",
    description="Update fields on an existing supplier record.",
)
async def update_supplier(supplier_id: str, body: SupplierUpdateRequest) -> SupplierResponse:
    """
    Partially update a supplier.

    Only fields provided in the request body are updated; omitted fields
    retain their current values.

    Returns:
        200 with updated supplier record.

    Raises:
        404 if supplier not found.
    """
    record = _suppliers.get(supplier_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )

    update_data = body.model_dump(exclude_unset=True)
    if "address" in update_data and update_data["address"] is not None:
        update_data["address"] = body.address.model_dump()

    for key, value in update_data.items():
        record[key] = value

    record["updated_at"] = datetime.now(timezone.utc)
    if "country" in update_data:
        record["country"] = record["country"].upper()

    logger.info("Supplier updated: %s", supplier_id)
    return _build_supplier_response(record)


@router.get(
    "/",
    response_model=SupplierListResponse,
    summary="List suppliers",
    description="List suppliers with filtering, search, and pagination.",
)
async def list_suppliers(
    country: Optional[str] = Query(None, description="Filter by country code"),
    commodity: Optional[str] = Query(None, description="Filter by commodity"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level"),
    compliance_status: Optional[str] = Query(None, description="Filter by compliance status"),
    search: Optional[str] = Query(None, description="Search by name (case-insensitive)"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> SupplierListResponse:
    """
    Retrieve a paginated, filterable list of suppliers.

    Supports filtering by country, commodity, risk level, and compliance
    status. Free-text search matches against supplier name.

    Returns:
        200 with paginated supplier list.
    """
    results = list(_suppliers.values())

    # Apply filters
    if country:
        results = [s for s in results if s["country"].upper() == country.upper()]
    if commodity:
        results = [
            s for s in results if commodity.lower() in [c.lower() for c in s["commodities"]]
        ]
    if risk_level:
        results = [s for s in results if s["risk_level"] == risk_level.lower()]
    if compliance_status:
        results = [s for s in results if s["compliance_status"] == compliance_status.lower()]
    if search:
        search_lower = search.lower()
        results = [s for s in results if search_lower in s["name"].lower()]

    # Sort by created_at descending
    results.sort(key=lambda s: s["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    end = start + limit
    page_items = results[start:end]

    return SupplierListResponse(
        items=[_build_supplier_response(s) for s in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.post(
    "/bulk-import",
    response_model=BulkImportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Bulk import suppliers",
    description="Import multiple supplier records from a structured payload.",
)
async def bulk_import_suppliers(body: BulkImportRequest) -> BulkImportResponse:
    """
    Bulk import suppliers from a list of records.

    Processes each record independently -- valid records are created even
    if some records fail validation.

    Returns:
        201 with import summary including created IDs and per-record errors.
    """
    created_ids: List[str] = []
    errors: List[Dict] = []
    now = datetime.now(timezone.utc)

    allowed_commodities = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}

    for idx, rec in enumerate(body.records):
        try:
            # Validate commodities
            for c in rec.commodities:
                if c.lower() not in allowed_commodities:
                    raise ValueError(f"Invalid commodity '{c}'")

            supplier_id = f"sup_{uuid.uuid4().hex[:12]}"
            record = {
                "supplier_id": supplier_id,
                "name": rec.name,
                "country": rec.country.upper(),
                "tax_id": rec.tax_id,
                "commodities": [c.lower() for c in rec.commodities],
                "address": rec.address.model_dump() if rec.address else None,
                "compliance_status": "pending",
                "risk_level": "unknown",
                "created_at": now,
                "updated_at": now,
            }
            _suppliers[supplier_id] = record
            created_ids.append(supplier_id)
        except Exception as exc:
            errors.append({"index": idx, "name": rec.name, "error": str(exc)})

    logger.info(
        "Bulk import: %d created, %d failed out of %d",
        len(created_ids),
        len(errors),
        len(body.records),
    )

    return BulkImportResponse(
        total_submitted=len(body.records),
        total_created=len(created_ids),
        total_failed=len(errors),
        created_ids=created_ids,
        errors=errors,
    )


@router.get(
    "/{supplier_id}/compliance",
    response_model=ComplianceStatusResponse,
    summary="Get compliance status",
    description="Retrieve EUDR compliance status for a specific supplier.",
)
async def get_supplier_compliance(supplier_id: str) -> ComplianceStatusResponse:
    """
    Fetch compliance status including DDS counts, document verification
    progress, and outstanding issues.

    Returns:
        200 with compliance status details.

    Raises:
        404 if supplier not found.
    """
    record = _suppliers.get(supplier_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )

    # In v1.0, return baseline compliance data from in-memory state.
    issues: List[str] = []
    if record["compliance_status"] == "pending":
        issues.append("No Due Diligence Statement submitted")
        issues.append("Supplier risk assessment not completed")

    return ComplianceStatusResponse(
        supplier_id=supplier_id,
        supplier_name=record["name"],
        compliance_status=record["compliance_status"],
        dds_submitted=0,
        dds_pending=0,
        documents_verified=0,
        documents_pending=0,
        last_assessment_date=None,
        issues=issues,
    )


@router.get(
    "/{supplier_id}/risk",
    response_model=RiskSummaryResponse,
    summary="Get risk summary",
    description="Retrieve risk assessment summary for a specific supplier.",
)
async def get_supplier_risk(supplier_id: str) -> RiskSummaryResponse:
    """
    Fetch aggregated risk summary including country risk, per-commodity
    risk levels, and deforestation alert count.

    Returns:
        200 with risk summary.

    Raises:
        404 if supplier not found.
    """
    record = _suppliers.get(supplier_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )

    # Country-level risk heuristic (v1.0 placeholder)
    high_risk_countries = {"BR", "ID", "CO", "MY", "CG", "CD", "CM", "GH", "CI"}
    country_risk = "high" if record["country"] in high_risk_countries else "standard"

    commodity_risks = {}
    for c in record["commodities"]:
        if c in ("oil_palm", "soya", "cattle"):
            commodity_risks[c] = "high"
        elif c in ("cocoa", "coffee", "rubber"):
            commodity_risks[c] = "medium"
        else:
            commodity_risks[c] = "low"

    # Composite score: higher for high-risk country + commodity combination
    base_score = 50.0 if country_risk == "high" else 25.0
    commodity_scores = {"high": 20, "medium": 10, "low": 5}
    if commodity_risks:
        avg_commodity = sum(
            commodity_scores.get(r, 5) for r in commodity_risks.values()
        ) / len(commodity_risks)
        risk_score = min(100.0, base_score + avg_commodity)
    else:
        risk_score = base_score

    if risk_score >= 70:
        overall = "high"
    elif risk_score >= 40:
        overall = "medium"
    else:
        overall = "low"

    return RiskSummaryResponse(
        supplier_id=supplier_id,
        supplier_name=record["name"],
        overall_risk_level=overall,
        risk_score=round(risk_score, 1),
        country_risk=country_risk,
        commodity_risks=commodity_risks,
        deforestation_alerts=0,
        last_assessment_date=None,
    )


@router.delete(
    "/{supplier_id}",
    response_model=DeleteResponse,
    summary="Delete supplier",
    description="Remove a supplier record from the system.",
)
async def delete_supplier(supplier_id: str) -> DeleteResponse:
    """
    Soft-delete a supplier.

    In v1.0 this performs a hard delete from in-memory storage.

    Returns:
        200 with deletion confirmation.

    Raises:
        404 if supplier not found.
    """
    record = _suppliers.pop(supplier_id, None)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found",
        )

    logger.info("Supplier deleted: %s (%s)", supplier_id, record["name"])
    return DeleteResponse(
        supplier_id=supplier_id,
        deleted=True,
        message=f"Supplier '{record['name']}' deleted successfully",
    )
