"""
GL-Taxonomy-APP Portfolio Management API

Manages investment and lending portfolios for EU Taxonomy alignment
analysis.  Supports portfolio CRUD, individual holding management,
bulk upload from CSV/Excel, and organizational portfolio listing.

Portfolio Types:
    - Investment Portfolio: Equity, bonds, funds
    - Lending Portfolio: Loans, mortgages, project finance
    - Mixed Portfolio: Combined investment and lending

Used by both non-financial undertakings (investment holdings) and
financial institutions (banking book, trading book).
"""

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter(prefix="/api/v1/taxonomy/portfolios", tags=["Portfolio Management"])


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PortfolioType(str, Enum):
    """Portfolio type classification."""
    INVESTMENT = "investment"
    LENDING = "lending"
    MIXED = "mixed"


class HoldingType(str, Enum):
    """Holding/exposure type."""
    EQUITY = "equity"
    CORPORATE_BOND = "corporate_bond"
    COVERED_BOND = "covered_bond"
    LOAN_NFC = "loan_nfc"
    MORTGAGE_RESIDENTIAL = "mortgage_residential"
    MORTGAGE_COMMERCIAL = "mortgage_commercial"
    PROJECT_FINANCE = "project_finance"
    VEHICLE_LOAN = "vehicle_loan"
    FUND = "fund"


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------

class CreatePortfolioRequest(BaseModel):
    """Create a portfolio."""
    org_id: str = Field(...)
    portfolio_name: str = Field(..., min_length=1, max_length=300)
    portfolio_type: PortfolioType = Field(...)
    currency: str = Field("EUR", max_length=3)
    reporting_date: str = Field(..., description="ISO date")
    description: Optional[str] = Field(None, max_length=2000)

    class Config:
        json_schema_extra = {
            "example": {
                "org_id": "bank_001",
                "portfolio_name": "Green Lending Book Q4 2025",
                "portfolio_type": "lending",
                "currency": "EUR",
                "reporting_date": "2025-12-31",
            }
        }


class UpdatePortfolioRequest(BaseModel):
    """Update a portfolio."""
    portfolio_name: Optional[str] = Field(None, max_length=300)
    description: Optional[str] = Field(None, max_length=2000)
    reporting_date: Optional[str] = None


class AddHoldingsRequest(BaseModel):
    """Add holdings to a portfolio."""
    holdings: List[Dict[str, Any]] = Field(
        ..., min_length=1, max_length=5000,
        description="List of {name, type, amount_eur, nace_code, epc_rating, ...}",
    )


class UploadPortfolioRequest(BaseModel):
    """Upload portfolio from file."""
    org_id: str = Field(...)
    portfolio_name: str = Field(...)
    file_format: str = Field(..., description="csv or excel")
    column_mapping: Optional[Dict[str, str]] = Field(None, description="Source column to target field mapping")


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class PortfolioResponse(BaseModel):
    """Portfolio record."""
    portfolio_id: str
    org_id: str
    portfolio_name: str
    portfolio_type: str
    currency: str
    reporting_date: str
    total_value_eur: float
    holdings_count: int
    description: Optional[str]
    created_at: datetime
    updated_at: datetime


class HoldingResponse(BaseModel):
    """Portfolio holding."""
    holding_id: str
    portfolio_id: str
    counterparty_name: str
    holding_type: str
    amount_eur: float
    nace_code: Optional[str]
    taxonomy_eligible: bool
    taxonomy_aligned: bool
    aligned_amount_eur: float
    epc_rating: Optional[str]
    added_at: datetime


class HoldingsListResponse(BaseModel):
    """Holdings list with pagination."""
    portfolio_id: str
    holdings: List[HoldingResponse]
    total_count: int
    total_value_eur: float
    aligned_value_eur: float
    alignment_pct: float
    generated_at: datetime


class UploadResponse(BaseModel):
    """Portfolio upload result."""
    portfolio_id: str
    org_id: str
    portfolio_name: str
    rows_processed: int
    rows_accepted: int
    rows_rejected: int
    validation_errors: List[str]
    total_value_eur: float
    generated_at: datetime


class PortfolioListResponse(BaseModel):
    """Organization portfolio list."""
    org_id: str
    portfolios: List[PortfolioResponse]
    total_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# In-Memory Store
# ---------------------------------------------------------------------------

_portfolios: Dict[str, Dict[str, Any]] = {}
_holdings: Dict[str, List[Dict[str, Any]]] = {}


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _now() -> datetime:
    return datetime.utcnow()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/",
    response_model=PortfolioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create portfolio",
    description="Create a new portfolio for taxonomy alignment analysis.",
)
async def create_portfolio(request: CreatePortfolioRequest) -> PortfolioResponse:
    """Create a portfolio."""
    portfolio_id = _generate_id("pf")
    now = _now()
    data = {
        "portfolio_id": portfolio_id,
        "org_id": request.org_id,
        "portfolio_name": request.portfolio_name,
        "portfolio_type": request.portfolio_type.value,
        "currency": request.currency,
        "reporting_date": request.reporting_date,
        "total_value_eur": 0,
        "holdings_count": 0,
        "description": request.description,
        "created_at": now,
        "updated_at": now,
    }
    _portfolios[portfolio_id] = data
    _holdings[portfolio_id] = []
    return PortfolioResponse(**data)


@router.get(
    "/{portfolio_id}",
    response_model=PortfolioResponse,
    summary="Get portfolio",
    description="Retrieve a portfolio by ID.",
)
async def get_portfolio(portfolio_id: str) -> PortfolioResponse:
    """Get portfolio by ID."""
    portfolio = _portfolios.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found.")
    return PortfolioResponse(**portfolio)


@router.put(
    "/{portfolio_id}",
    response_model=PortfolioResponse,
    summary="Update portfolio",
    description="Update portfolio metadata.",
)
async def update_portfolio(portfolio_id: str, request: UpdatePortfolioRequest) -> PortfolioResponse:
    """Update portfolio."""
    portfolio = _portfolios.get(portfolio_id)
    if not portfolio:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found.")

    updates = request.model_dump(exclude_unset=True)
    portfolio.update(updates)
    portfolio["updated_at"] = _now()
    return PortfolioResponse(**portfolio)


@router.delete(
    "/{portfolio_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete portfolio",
    description="Delete a portfolio and all its holdings.",
)
async def delete_portfolio(portfolio_id: str) -> None:
    """Delete portfolio."""
    if portfolio_id not in _portfolios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found.")
    del _portfolios[portfolio_id]
    _holdings.pop(portfolio_id, None)
    return None


@router.post(
    "/{portfolio_id}/holdings",
    response_model=List[HoldingResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Add holdings",
    description="Add one or more holdings to a portfolio.",
)
async def add_holdings(
    portfolio_id: str,
    request: AddHoldingsRequest,
) -> List[HoldingResponse]:
    """Add holdings to portfolio."""
    if portfolio_id not in _portfolios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found.")

    results = []
    total_added = 0

    for h in request.holdings:
        holding_id = _generate_id("hld")
        amount = float(h.get("amount_eur", 0))
        nace = h.get("nace_code")
        epc = h.get("epc_rating")

        # Simplified alignment logic
        eligible = nace is not None
        aligned = False
        aligned_amount = 0.0

        if epc and epc in ("A", "B"):
            eligible = True
            aligned = True
            aligned_amount = amount
        elif h.get("taxonomy_aligned", False):
            aligned = True
            aligned_amount = amount

        entry = {
            "holding_id": holding_id,
            "portfolio_id": portfolio_id,
            "counterparty_name": h.get("name", "Unknown"),
            "holding_type": h.get("type", "loan_nfc"),
            "amount_eur": amount,
            "nace_code": nace,
            "taxonomy_eligible": eligible,
            "taxonomy_aligned": aligned,
            "aligned_amount_eur": aligned_amount,
            "epc_rating": epc,
            "added_at": _now(),
        }
        _holdings[portfolio_id].append(entry)
        results.append(HoldingResponse(**entry))
        total_added += amount

    # Update portfolio totals
    portfolio = _portfolios[portfolio_id]
    portfolio["holdings_count"] = len(_holdings[portfolio_id])
    portfolio["total_value_eur"] = sum(h["amount_eur"] for h in _holdings[portfolio_id])
    portfolio["updated_at"] = _now()

    return results


@router.get(
    "/{portfolio_id}/holdings",
    response_model=HoldingsListResponse,
    summary="List holdings",
    description="List all holdings in a portfolio.",
)
async def list_holdings(
    portfolio_id: str,
    holding_type: Optional[str] = Query(None),
    aligned_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> HoldingsListResponse:
    """List portfolio holdings."""
    if portfolio_id not in _portfolios:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Portfolio {portfolio_id} not found.")

    holdings = _holdings.get(portfolio_id, [])
    if holding_type:
        holdings = [h for h in holdings if h["holding_type"] == holding_type]
    if aligned_only:
        holdings = [h for h in holdings if h["taxonomy_aligned"]]

    total_value = sum(h["amount_eur"] for h in holdings)
    aligned_value = sum(h["aligned_amount_eur"] for h in holdings)
    paginated = holdings[offset:offset + limit]

    return HoldingsListResponse(
        portfolio_id=portfolio_id,
        holdings=[HoldingResponse(**h) for h in paginated],
        total_count=len(holdings),
        total_value_eur=round(total_value, 2),
        aligned_value_eur=round(aligned_value, 2),
        alignment_pct=round((aligned_value / total_value) * 100, 1) if total_value > 0 else 0,
        generated_at=_now(),
    )


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload portfolio file (CSV/Excel)",
    description="Upload a portfolio from a CSV or Excel file.",
)
async def upload_portfolio(request: UploadPortfolioRequest) -> UploadResponse:
    """Upload portfolio from file."""
    if request.file_format not in ("csv", "excel"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{request.file_format}'. Use csv or excel.",
        )

    portfolio_id = _generate_id("pf")
    now = _now()

    # Simulate file processing
    rows_processed = 150
    rows_accepted = 142
    rows_rejected = 8
    total_value = 2500000000.0

    _portfolios[portfolio_id] = {
        "portfolio_id": portfolio_id,
        "org_id": request.org_id,
        "portfolio_name": request.portfolio_name,
        "portfolio_type": "mixed",
        "currency": "EUR",
        "reporting_date": "2025-12-31",
        "total_value_eur": total_value,
        "holdings_count": rows_accepted,
        "description": f"Uploaded from {request.file_format} file",
        "created_at": now,
        "updated_at": now,
    }
    _holdings[portfolio_id] = []

    return UploadResponse(
        portfolio_id=portfolio_id,
        org_id=request.org_id,
        portfolio_name=request.portfolio_name,
        rows_processed=rows_processed,
        rows_accepted=rows_accepted,
        rows_rejected=rows_rejected,
        validation_errors=[
            "Row 15: Missing NACE code",
            "Row 28: Invalid amount (negative value)",
            "Row 42: Unknown holding type 'unknown_type'",
            "Row 67: Missing counterparty name",
            "Rows 89-92: Duplicate entries detected",
            "Row 105: EPC rating out of range",
            "Row 131: Missing currency code",
            "Row 148: Amount exceeds single-exposure limit",
        ],
        total_value_eur=total_value,
        generated_at=now,
    )


@router.get(
    "/{org_id}/list",
    response_model=PortfolioListResponse,
    summary="List org portfolios",
    description="List all portfolios for an organization.",
)
async def list_org_portfolios(
    org_id: str,
    portfolio_type: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> PortfolioListResponse:
    """List org portfolios."""
    portfolios = [p for p in _portfolios.values() if p["org_id"] == org_id]
    if portfolio_type:
        portfolios = [p for p in portfolios if p["portfolio_type"] == portfolio_type]
    portfolios.sort(key=lambda p: p["created_at"], reverse=True)

    return PortfolioListResponse(
        org_id=org_id,
        portfolios=[PortfolioResponse(**p) for p in portfolios[:limit]],
        total_count=len(portfolios),
        generated_at=_now(),
    )
