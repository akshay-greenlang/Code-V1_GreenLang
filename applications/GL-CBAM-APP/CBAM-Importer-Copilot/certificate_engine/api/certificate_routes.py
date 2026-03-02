# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Certificate Engine API Routes v1.1

FastAPI router implementing the CBAM certificate management API.
Provides endpoints for certificate obligation calculation, EU ETS price
retrieval, free allocation benchmarks, and carbon price deduction workflow.

Per EU CBAM Regulation 2023/956:
  - Articles 21-24: Certificate obligations, pricing, holding, surrender
  - Article 26: Carbon price deduction for origin-country payments
  - Article 31: Free allocation phase-out schedule (2026-2034)

All calculation endpoints use deterministic Decimal arithmetic (ZERO HALLUCINATION).

Endpoints:
    POST /obligations/calculate                     - Calculate annual obligation
    GET  /obligations/{importer_id}/{year}          - Get obligation summary
    GET  /obligations/{importer_id}/{year}/breakdown/cn      - By CN code
    GET  /obligations/{importer_id}/{year}/breakdown/country - By country
    POST /obligations/{importer_id}/{year}/project           - Cost projection
    GET  /holdings/{importer_id}/{year}/{quarter}   - Quarterly holding check
    POST /holdings/{importer_id}/{year}/record      - Record certificates held
    GET  /ets-price/current                         - Get current ETS price
    GET  /ets-price/weekly/{date}                   - Get weekly price
    GET  /ets-price/quarterly/{year}/{quarter}      - Quarterly average
    GET  /ets-price/annual/{year}                   - Annual average
    GET  /ets-price/history                         - Date range history
    GET  /ets-price/trend                           - Price trend analysis
    POST /ets-price/manual                          - Manual price entry
    POST /ets-price/import                          - Bulk price import
    GET  /free-allocation/schedule                  - Phase-out schedule
    GET  /free-allocation/benchmarks                - All product benchmarks
    GET  /free-allocation/{cn_code}/{year}          - Specific allocation factor
    PUT  /free-allocation/{cn_code}/{year}          - Update benchmark
    GET  /free-allocation/compare                   - Year-over-year comparison
    POST /deductions/register                       - Register carbon price deduction
    GET  /deductions/{importer_id}/{year}           - Get deductions
    GET  /deductions/detail/{deduction_id}          - Get specific deduction
    POST /deductions/{deduction_id}/verify          - Verify deduction
    POST /deductions/{deduction_id}/approve         - Approve deduction
    POST /deductions/{deduction_id}/reject          - Reject deduction
    POST /deductions/{deduction_id}/evidence        - Add evidence doc
    GET  /deductions/{importer_id}/{year}/summary   - Deduction summary
    GET  /country-pricing/{country}                 - Country carbon pricing info

Version: 1.1.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Path, Body
from pydantic import BaseModel, Field

from ..models import (
    CarbonPricingScheme,
    CertificateObligation,
    CertificateSummary,
    DeductionStatus,
    QuarterlyHolding,
    compute_sha256,
    quantize_decimal,
)
from ..certificate_calculator import CertificateCalculatorEngine
from ..ets_price_service import ETSPriceService
from ..free_allocation import FreeAllocationEngine
from ..carbon_price_deduction import CarbonPriceDeductionEngine

logger = logging.getLogger(__name__)

# ============================================================================
# ROUTER DEFINITION
# ============================================================================

router = APIRouter(
    prefix="/api/v1/certificates",
    tags=["certificates"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

# ---- Obligation Endpoints ----

class ShipmentInput(BaseModel):
    """Individual shipment for obligation calculation."""

    cn_code: str = Field(
        ...,
        min_length=6,
        max_length=10,
        description="Combined Nomenclature code (6-10 digits)"
    )
    quantity_mt: float = Field(
        ...,
        gt=0,
        description="Imported quantity in metric tonnes"
    )
    embedded_emissions_tCO2e: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total embedded emissions in tCO2e (if known)"
    )
    embedded_emissions_per_mt: Optional[float] = Field(
        default=None,
        ge=0,
        description="Specific embedded emissions in tCO2e per tonne (if total not provided)"
    )
    country_of_origin: str = Field(
        default="",
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )


class ObligationCalculateRequest(BaseModel):
    """Request body for calculating annual certificate obligation."""

    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI number or internal identifier"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Obligation year"
    )
    shipments: List[ShipmentInput] = Field(
        ...,
        min_length=1,
        description="List of shipments for the obligation calculation"
    )


class ObligationResponse(BaseModel):
    """Response for certificate obligation endpoints."""

    status: str = Field(default="success")
    obligation: Dict[str, Any] = Field(
        ...,
        description="Certificate obligation details"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time in milliseconds"
    )


class SummaryResponse(BaseModel):
    """Response for obligation summary endpoint."""

    status: str = Field(default="success")
    summary: Dict[str, Any] = Field(
        ...,
        description="Certificate obligation summary"
    )


class BreakdownResponse(BaseModel):
    """Response for obligation breakdown endpoints."""

    status: str = Field(default="success")
    importer_id: str = Field(..., description="Importer identifier")
    year: int = Field(..., description="Reference year")
    breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Breakdown records"
    )


class CostProjectionRequest(BaseModel):
    """Request body for cost projection."""

    ets_price_forecast: Optional[float] = Field(
        default=None,
        gt=0,
        description="ETS price forecast EUR/tCO2e (uses current price if omitted)"
    )


class CostProjectionResponse(BaseModel):
    """Response for cost projection endpoint."""

    status: str = Field(default="success")
    projection: Dict[str, Any] = Field(
        ...,
        description="Cost projection with low/mid/high scenarios"
    )


# ---- Holdings Endpoints ----

class HoldingResponse(BaseModel):
    """Response for quarterly holding check."""

    status: str = Field(default="success")
    holding: Dict[str, Any] = Field(
        ...,
        description="Quarterly holding details"
    )


class RecordHoldingRequest(BaseModel):
    """Request body for recording certificates held."""

    certificates_held: float = Field(
        ...,
        ge=0,
        description="Number of CBAM certificates currently held"
    )


class RecordHoldingResponse(BaseModel):
    """Response for recording certificates held."""

    status: str = Field(default="success")
    importer_id: str = Field(..., description="Importer identifier")
    year: int = Field(..., description="Reference year")
    certificates_held: str = Field(..., description="Certificates now on record")
    message: str = Field(default="")


# ---- ETS Price Endpoints ----

class ETSPriceResponse(BaseModel):
    """Response for ETS price endpoints."""

    status: str = Field(default="success")
    price: Dict[str, Any] = Field(
        ...,
        description="ETS price data"
    )


class ETSPriceHistoryResponse(BaseModel):
    """Response for ETS price history."""

    status: str = Field(default="success")
    count: int = Field(default=0, description="Number of records returned")
    prices: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Historical price records"
    )


class ETSTrendResponse(BaseModel):
    """Response for ETS price trend analysis."""

    status: str = Field(default="success")
    trend: Dict[str, Any] = Field(
        ...,
        description="Trend analysis results"
    )


class ManualPriceRequest(BaseModel):
    """Request body for manual ETS price entry."""

    date: str = Field(
        ...,
        description="Price date in ISO format (YYYY-MM-DD)"
    )
    price_eur_per_tco2e: float = Field(
        ...,
        gt=0,
        description="Price in EUR per tCO2e"
    )


class BulkPriceImportRequest(BaseModel):
    """Request body for bulk price import."""

    prices: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        description="List of price records with date, price, and optional period fields"
    )


class BulkImportResponse(BaseModel):
    """Response for bulk price import."""

    status: str = Field(default="success")
    imported: int = Field(default=0, description="Records imported successfully")
    total: int = Field(default=0, description="Total records provided")
    message: str = Field(default="")


# ---- Free Allocation Endpoints ----

class FreeAllocationScheduleResponse(BaseModel):
    """Response for phase-out schedule."""

    status: str = Field(default="success")
    schedule: Dict[str, Any] = Field(
        ...,
        description="Year-by-year phase-out schedule"
    )


class BenchmarkListResponse(BaseModel):
    """Response for product benchmark listing."""

    status: str = Field(default="success")
    benchmarks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Product benchmark records"
    )


class AllocationFactorResponse(BaseModel):
    """Response for specific allocation factor."""

    status: str = Field(default="success")
    factor: Dict[str, Any] = Field(
        ...,
        description="Allocation factor details"
    )


class UpdateBenchmarkRequest(BaseModel):
    """Request body for updating a product benchmark."""

    benchmark_value_tco2e: float = Field(
        ...,
        gt=0,
        description="New benchmark value in tCO2e per tonne of product"
    )


class AllocationCompareResponse(BaseModel):
    """Response for year-over-year comparison."""

    status: str = Field(default="success")
    comparison: Dict[str, Any] = Field(
        ...,
        description="Year-over-year comparison details"
    )


# ---- Deduction Endpoints ----

class RegisterDeductionRequest(BaseModel):
    """Request body for registering a carbon price deduction."""

    deduction_id: str = Field(
        ...,
        min_length=5,
        description="Unique deduction identifier (e.g. CPD-2026-NL123-001)"
    )
    importer_id: str = Field(
        ...,
        min_length=1,
        description="Importer EORI or internal identifier"
    )
    installation_id: str = Field(
        ...,
        min_length=1,
        description="Installation identifier in country of origin"
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    )
    pricing_scheme: CarbonPricingScheme = Field(
        ...,
        description="Type of carbon pricing mechanism"
    )
    carbon_price_paid_local: float = Field(
        ...,
        ge=0,
        description="Amount paid in local currency"
    )
    currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="ISO 4217 currency code"
    )
    tonnes_covered: float = Field(
        ...,
        gt=0,
        description="Tonnes of CO2e covered by this payment"
    )
    evidence_docs: List[str] = Field(
        default_factory=list,
        description="List of evidence document references"
    )
    year: int = Field(
        ...,
        ge=2023,
        le=2099,
        description="Reference year for this deduction"
    )


class DeductionResponse(BaseModel):
    """Response for a single deduction."""

    status: str = Field(default="success")
    deduction: Dict[str, Any] = Field(
        ...,
        description="Deduction details"
    )


class DeductionListResponse(BaseModel):
    """Response for listing deductions."""

    status: str = Field(default="success")
    count: int = Field(default=0, description="Number of deductions")
    deductions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Deduction records"
    )


class VerifyDeductionRequest(BaseModel):
    """Request body for verifying a deduction."""

    verified_by: str = Field(
        ...,
        min_length=1,
        description="Identifier of the verifier"
    )


class ApproveDeductionRequest(BaseModel):
    """Request body for approving a deduction."""

    approved_by: str = Field(
        ...,
        min_length=1,
        description="Identifier of the approver"
    )


class RejectDeductionRequest(BaseModel):
    """Request body for rejecting a deduction."""

    rejected_by: str = Field(
        ...,
        min_length=1,
        description="Identifier of the rejector"
    )
    reason: str = Field(
        default="",
        description="Reason for rejection"
    )


class AddEvidenceRequest(BaseModel):
    """Request body for adding an evidence document."""

    document_ref: str = Field(
        ...,
        min_length=1,
        description="Document reference (filename, URL, or ID)"
    )


class DeductionSummaryResponse(BaseModel):
    """Response for deduction summary."""

    status: str = Field(default="success")
    summary: Dict[str, Any] = Field(
        ...,
        description="Deduction summary with breakdowns"
    )


class CountryPricingResponse(BaseModel):
    """Response for country carbon pricing info."""

    status: str = Field(default="success")
    pricing: Dict[str, Any] = Field(
        ...,
        description="Country carbon pricing details"
    )


# ============================================================================
# HELPER: Serialize Pydantic models to dict
# ============================================================================

def _obligation_to_dict(obl: CertificateObligation) -> Dict[str, Any]:
    """Convert CertificateObligation to a JSON-safe dict."""
    return {
        "obligation_id": obl.obligation_id,
        "importer_id": obl.importer_id,
        "year": obl.year,
        "cn_code": obl.cn_code,
        "country_of_origin": obl.country_of_origin,
        "quantity_mt": str(obl.quantity_mt),
        "embedded_emissions_tCO2e": str(obl.embedded_emissions_tCO2e),
        "gross_certificates_required": str(obl.gross_certificates_required),
        "free_allocation_deduction": str(obl.free_allocation_deduction),
        "carbon_price_deduction_eur": str(obl.carbon_price_deduction_eur),
        "carbon_price_deduction_tCO2e": str(obl.carbon_price_deduction_tCO2e),
        "net_certificates_required": str(obl.net_certificates_required),
        "certificate_cost_eur": str(obl.certificate_cost_eur),
        "ets_price_used": str(obl.ets_price_used),
        "calculation_date": str(obl.calculation_date),
        "provenance_hash": obl.provenance_hash,
    }


def _summary_to_dict(summary: CertificateSummary) -> Dict[str, Any]:
    """Convert CertificateSummary to a JSON-safe dict."""
    return {
        "importer_id": summary.importer_id,
        "year": summary.year,
        "total_gross": str(summary.total_gross),
        "total_free_allocation": str(summary.total_free_allocation),
        "total_carbon_deductions": str(summary.total_carbon_deductions),
        "total_net": str(summary.total_net),
        "total_cost_eur": str(summary.total_cost_eur),
        "quarterly_holdings_required": str(summary.quarterly_holdings_required),
        "certificates_held": str(summary.certificates_held),
        "shortfall": str(summary.shortfall),
        "ets_price_used": str(summary.ets_price_used),
        "obligations_by_cn": summary.obligations_by_cn,
        "provenance_hash": summary.provenance_hash,
    }


def _holding_to_dict(holding: QuarterlyHolding) -> Dict[str, Any]:
    """Convert QuarterlyHolding to a JSON-safe dict."""
    return {
        "quarter": holding.quarter,
        "year": holding.year,
        "importer_id": holding.importer_id,
        "holding_required": str(holding.holding_required),
        "certificates_held": str(holding.certificates_held),
        "compliant": holding.compliant,
        "shortfall": str(holding.shortfall),
    }


def _deduction_to_dict(d: Any) -> Dict[str, Any]:
    """Convert CarbonPriceDeduction to a JSON-safe dict."""
    return {
        "deduction_id": d.deduction_id,
        "importer_id": d.importer_id,
        "installation_id": d.installation_id,
        "country": d.country,
        "pricing_scheme": d.pricing_scheme.value,
        "carbon_price_paid_eur": str(d.carbon_price_paid_eur),
        "carbon_price_paid_local": str(d.carbon_price_paid_local),
        "exchange_rate": str(d.exchange_rate),
        "currency": d.currency,
        "tonnes_covered": str(d.tonnes_covered),
        "deduction_per_tonne_eur": str(d.deduction_per_tonne_eur),
        "evidence_docs": d.evidence_docs,
        "verification_status": d.verification_status.value,
        "verified_by": d.verified_by,
        "verified_at": str(d.verified_at) if d.verified_at else None,
        "year": d.year,
        "provenance_hash": d.provenance_hash,
    }


# ============================================================================
# ENGINE SINGLETONS
# ============================================================================

def _get_calculator() -> CertificateCalculatorEngine:
    """Get the singleton CertificateCalculatorEngine."""
    return CertificateCalculatorEngine()


def _get_ets_service() -> ETSPriceService:
    """Get the singleton ETSPriceService."""
    return ETSPriceService()


def _get_free_alloc() -> FreeAllocationEngine:
    """Get a FreeAllocationEngine instance."""
    return FreeAllocationEngine()


def _get_deduction_engine() -> CarbonPriceDeductionEngine:
    """Get the singleton CarbonPriceDeductionEngine."""
    return CarbonPriceDeductionEngine()


# ============================================================================
# OBLIGATION ENDPOINTS
# ============================================================================

@router.post(
    "/obligations/calculate",
    response_model=ObligationResponse,
    summary="Calculate annual certificate obligation",
    description=(
        "Calculate the full CBAM certificate obligation for an importer. "
        "Computes gross certificates, applies free allocation and carbon "
        "price deductions, and determines net obligation and estimated cost. "
        "Per Regulation 2023/956 Articles 21-26."
    ),
)
async def calculate_obligation(
    request: ObligationCalculateRequest,
) -> ObligationResponse:
    """Calculate annual CBAM certificate obligation."""
    start_time = time.time()

    try:
        # Convert shipment inputs to dicts for the engine
        shipments = []
        for s in request.shipments:
            shipment_dict: Dict[str, Any] = {
                "cn_code": s.cn_code,
                "quantity_mt": s.quantity_mt,
                "country_of_origin": s.country_of_origin,
            }
            if s.embedded_emissions_tCO2e is not None:
                shipment_dict["embedded_emissions_tCO2e"] = s.embedded_emissions_tCO2e
            elif s.embedded_emissions_per_mt is not None:
                shipment_dict["embedded_emissions_per_mt"] = s.embedded_emissions_per_mt
            else:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Each shipment must provide either "
                        "embedded_emissions_tCO2e or embedded_emissions_per_mt"
                    ),
                )
            shipments.append(shipment_dict)

        calculator = _get_calculator()
        obligation = calculator.calculate_annual_obligation(
            importer_id=request.importer_id,
            year=request.year,
            shipments=shipments,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "API: obligation calculated, id=%s, net=%s, cost=EUR %s, in %.1fms",
            obligation.obligation_id, obligation.net_certificates_required,
            obligation.certificate_cost_eur, duration_ms,
        )

        return ObligationResponse(
            status="success",
            obligation=_obligation_to_dict(obligation),
            provenance_hash=obligation.provenance_hash,
            processing_time_ms=round(duration_ms, 2),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Obligation calculation validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Obligation calculation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/obligations/{importer_id}/{year}",
    response_model=SummaryResponse,
    summary="Get obligation summary",
    description=(
        "Get the full certificate obligation summary for an importer and "
        "year. Includes totals, deductions, quarterly holding requirements, "
        "and CN code breakdown."
    ),
)
async def get_obligation_summary(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Obligation year"),
) -> SummaryResponse:
    """Get certificate obligation summary."""
    start_time = time.time()

    try:
        calculator = _get_calculator()
        summary = calculator.get_obligation_summary(importer_id, year)

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "API: summary retrieved, importer=%s, year=%d, in %.1fms",
            importer_id, year, duration_ms,
        )

        return SummaryResponse(
            status="success",
            summary=_summary_to_dict(summary),
        )

    except ValueError as exc:
        logger.warning("Summary retrieval error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Summary retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/obligations/{importer_id}/{year}/breakdown/cn",
    response_model=BreakdownResponse,
    summary="Obligation breakdown by CN code",
    description="Get certificate obligation breakdown by Combined Nomenclature code.",
)
async def get_breakdown_by_cn(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Obligation year"),
) -> BreakdownResponse:
    """Get obligation breakdown by CN code."""
    try:
        calculator = _get_calculator()
        obligations = calculator.breakdown_by_cn_code(importer_id, year)

        breakdown = [_obligation_to_dict(obl) for obl in obligations]

        return BreakdownResponse(
            status="success",
            importer_id=importer_id,
            year=year,
            breakdown=breakdown,
        )

    except Exception as exc:
        logger.error("CN breakdown failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/obligations/{importer_id}/{year}/breakdown/country",
    response_model=BreakdownResponse,
    summary="Obligation breakdown by country",
    description="Get certificate obligation breakdown by country of origin.",
)
async def get_breakdown_by_country(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Obligation year"),
) -> BreakdownResponse:
    """Get obligation breakdown by country of origin."""
    try:
        calculator = _get_calculator()
        breakdown = calculator.breakdown_by_country(importer_id, year)

        return BreakdownResponse(
            status="success",
            importer_id=importer_id,
            year=year,
            breakdown=breakdown,
        )

    except Exception as exc:
        logger.error("Country breakdown failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/obligations/{importer_id}/{year}/project",
    response_model=CostProjectionResponse,
    summary="Project annual certificate cost",
    description=(
        "Project annual CBAM certificate cost with low/mid/high ETS price "
        "scenarios. Optionally provide a price forecast; otherwise the "
        "current ETS price is used as baseline."
    ),
)
async def project_cost(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Projection year"),
    request: CostProjectionRequest = Body(...),
) -> CostProjectionResponse:
    """Project annual CBAM certificate cost."""
    try:
        calculator = _get_calculator()

        forecast = None
        if request.ets_price_forecast is not None:
            forecast = Decimal(str(request.ets_price_forecast))

        projection = calculator.project_annual_cost(
            importer_id=importer_id,
            year=year,
            ets_price_forecast=forecast,
        )

        return CostProjectionResponse(
            status="success",
            projection=projection,
        )

    except ValueError as exc:
        logger.warning("Cost projection error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Cost projection failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# HOLDING ENDPOINTS
# ============================================================================

@router.get(
    "/holdings/{importer_id}/{year}/{quarter}",
    response_model=HoldingResponse,
    summary="Check quarterly holding compliance",
    description=(
        "Check whether an importer meets the quarterly holding requirement. "
        "Per Regulation Article 23, importers must hold certificates covering "
        "at least 50% of their estimated annual obligation."
    ),
)
async def check_quarterly_holding(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
    quarter: str = Path(
        ...,
        pattern="^Q[1-4]$",
        description="Quarter identifier (Q1/Q2/Q3/Q4)"
    ),
) -> HoldingResponse:
    """Check quarterly certificate holding compliance."""
    try:
        calculator = _get_calculator()
        holding = calculator.check_quarterly_compliance(
            importer_id=importer_id,
            year=year,
            quarter=quarter,
        )

        return HoldingResponse(
            status="success",
            holding=_holding_to_dict(holding),
        )

    except ValueError as exc:
        logger.warning("Quarterly holding check error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Quarterly holding check failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/holdings/{importer_id}/{year}/record",
    response_model=RecordHoldingResponse,
    summary="Record certificates held",
    description="Record the number of CBAM certificates currently held by an importer.",
)
async def record_certificates_held(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
    request: RecordHoldingRequest = Body(...),
) -> RecordHoldingResponse:
    """Record certificates held by an importer."""
    try:
        calculator = _get_calculator()
        count = Decimal(str(request.certificates_held))
        calculator.record_certificates_held(
            importer_id=importer_id,
            year=year,
            count=count,
        )

        logger.info(
            "API: certificates held recorded, importer=%s, year=%d, count=%s",
            importer_id, year, count,
        )

        return RecordHoldingResponse(
            status="success",
            importer_id=importer_id,
            year=year,
            certificates_held=str(quantize_decimal(count, places=3)),
            message=f"Recorded {count} certificates held for {importer_id} in {year}",
        )

    except ValueError as exc:
        logger.warning("Record holding error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Record holding failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# ETS PRICE ENDPOINTS
# ============================================================================

@router.get(
    "/ets-price/current",
    response_model=ETSPriceResponse,
    summary="Get current ETS price",
    description=(
        "Get the latest available EU ETS auction price. "
        "Per Regulation Article 22, CBAM certificate prices are derived "
        "from the weekly average of EU ETS allowance auction closing prices."
    ),
)
async def get_current_ets_price() -> ETSPriceResponse:
    """Get the current EU ETS price."""
    try:
        service = _get_ets_service()
        price = service.get_current_price()

        return ETSPriceResponse(
            status="success",
            price={
                "date": str(price.date),
                "price_eur_per_tco2e": str(price.price_eur_per_tco2e),
                "source": price.source.value,
                "volume_weighted": price.volume_weighted,
                "period": price.period,
            },
        )

    except ValueError as exc:
        logger.warning("Current ETS price error: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Current ETS price failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/ets-price/weekly/{price_date}",
    response_model=ETSPriceResponse,
    summary="Get weekly ETS price",
    description="Get the EU ETS price for the week containing the specified date.",
)
async def get_weekly_ets_price(
    price_date: str = Path(
        ...,
        description="Date in ISO format (YYYY-MM-DD)"
    ),
) -> ETSPriceResponse:
    """Get weekly EU ETS price for a specific date."""
    try:
        target_date = date.fromisoformat(price_date)
        service = _get_ets_service()
        price = service.get_weekly_price(target_date)

        return ETSPriceResponse(
            status="success",
            price={
                "date": str(price.date),
                "price_eur_per_tco2e": str(price.price_eur_per_tco2e),
                "source": price.source.value,
                "volume_weighted": price.volume_weighted,
                "period": price.period,
                "requested_date": price_date,
            },
        )

    except ValueError as exc:
        logger.warning("Weekly ETS price error: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Weekly ETS price failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/ets-price/quarterly/{year}/{quarter}",
    response_model=ETSPriceResponse,
    summary="Get quarterly average ETS price",
    description="Get the volume-weighted quarterly average EU ETS price.",
)
async def get_quarterly_ets_price(
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
    quarter: int = Path(..., ge=1, le=4, description="Quarter number (1-4)"),
) -> ETSPriceResponse:
    """Get quarterly average EU ETS price."""
    try:
        service = _get_ets_service()
        price = service.get_quarterly_average(year, quarter)

        return ETSPriceResponse(
            status="success",
            price={
                "date": str(price.date),
                "price_eur_per_tco2e": str(price.price_eur_per_tco2e),
                "source": price.source.value,
                "volume_weighted": price.volume_weighted,
                "period": price.period,
                "year": year,
                "quarter": quarter,
            },
        )

    except ValueError as exc:
        logger.warning("Quarterly ETS price error: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Quarterly ETS price failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/ets-price/annual/{year}",
    response_model=ETSPriceResponse,
    summary="Get annual average ETS price",
    description="Get the annual average EU ETS price for a given year.",
)
async def get_annual_ets_price(
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
) -> ETSPriceResponse:
    """Get annual average EU ETS price."""
    try:
        service = _get_ets_service()
        price = service.get_annual_average(year)

        return ETSPriceResponse(
            status="success",
            price={
                "date": str(price.date),
                "price_eur_per_tco2e": str(price.price_eur_per_tco2e),
                "source": price.source.value,
                "volume_weighted": price.volume_weighted,
                "period": price.period,
                "year": year,
            },
        )

    except ValueError as exc:
        logger.warning("Annual ETS price error: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Annual ETS price failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/ets-price/history",
    response_model=ETSPriceHistoryResponse,
    summary="Get ETS price history",
    description="Get historical EU ETS prices within a date range.",
)
async def get_ets_price_history(
    start_date: str = Query(
        ...,
        description="Start date in ISO format (YYYY-MM-DD)"
    ),
    end_date: str = Query(
        ...,
        description="End date in ISO format (YYYY-MM-DD)"
    ),
) -> ETSPriceHistoryResponse:
    """Get EU ETS price history for a date range."""
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)

        if start > end:
            raise HTTPException(
                status_code=422,
                detail="start_date must be on or before end_date",
            )

        service = _get_ets_service()
        prices = service.get_price_history(start, end)

        price_records = [
            {
                "date": str(p.date),
                "price_eur_per_tco2e": str(p.price_eur_per_tco2e),
                "source": p.source.value,
                "volume_weighted": p.volume_weighted,
                "period": p.period,
            }
            for p in prices
        ]

        return ETSPriceHistoryResponse(
            status="success",
            count=len(price_records),
            prices=price_records,
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning("Price history error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Price history failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/ets-price/trend",
    response_model=ETSTrendResponse,
    summary="Analyze ETS price trend",
    description=(
        "Analyze EU ETS price trend over recent periods. "
        "Returns moving average, volatility, min/max, direction, "
        "and percentage change."
    ),
)
async def get_ets_price_trend(
    periods: int = Query(
        default=12,
        ge=2,
        le=104,
        description="Number of most recent data points to analyze"
    ),
) -> ETSTrendResponse:
    """Analyze EU ETS price trend."""
    try:
        service = _get_ets_service()
        trend = service.get_price_trend(periods)

        return ETSTrendResponse(
            status="success",
            trend=trend,
        )

    except Exception as exc:
        logger.error("Price trend analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/ets-price/manual",
    response_model=ETSPriceResponse,
    summary="Manual ETS price entry",
    description=(
        "Manually set an EU ETS price (admin operation). "
        "Use this for corrections or when automated feeds are unavailable."
    ),
)
async def set_manual_ets_price(
    request: ManualPriceRequest = Body(...),
) -> ETSPriceResponse:
    """Manually set an EU ETS price."""
    try:
        price_date = date.fromisoformat(request.date)
        price_value = Decimal(str(request.price_eur_per_tco2e))

        service = _get_ets_service()
        price = service.set_price(price_date, price_value)

        logger.info(
            "API: manual ETS price set, date=%s, EUR %s",
            price_date, price_value,
        )

        return ETSPriceResponse(
            status="success",
            price={
                "date": str(price.date),
                "price_eur_per_tco2e": str(price.price_eur_per_tco2e),
                "source": price.source.value,
                "volume_weighted": price.volume_weighted,
                "period": price.period,
            },
        )

    except ValueError as exc:
        logger.warning("Manual price entry error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Manual price entry failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/ets-price/import",
    response_model=BulkImportResponse,
    summary="Bulk ETS price import",
    description=(
        "Bulk import EU ETS prices from an external feed. "
        "Each record should have date, price, and optional period fields."
    ),
)
async def import_ets_prices(
    request: BulkPriceImportRequest = Body(...),
) -> BulkImportResponse:
    """Bulk import EU ETS prices."""
    try:
        service = _get_ets_service()
        imported = service.import_price_feed(request.prices)

        logger.info(
            "API: bulk ETS price import, %d/%d records imported",
            imported, len(request.prices),
        )

        return BulkImportResponse(
            status="success",
            imported=imported,
            total=len(request.prices),
            message=f"Successfully imported {imported} of {len(request.prices)} price records",
        )

    except Exception as exc:
        logger.error("Bulk price import failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# FREE ALLOCATION ENDPOINTS
# ============================================================================

@router.get(
    "/free-allocation/schedule",
    response_model=FreeAllocationScheduleResponse,
    summary="Get free allocation phase-out schedule",
    description=(
        "Get the full free allocation phase-out schedule from 2025 to 2035. "
        "Per Regulation Article 31, free allocation declines from 97.5% in "
        "2026 to 0% in 2034."
    ),
)
async def get_phase_out_schedule() -> FreeAllocationScheduleResponse:
    """Get the free allocation phase-out schedule."""
    try:
        engine = _get_free_alloc()
        schedule = engine.get_phase_out_schedule()

        # Convert int keys to string for JSON compatibility
        schedule_str = {str(k): v for k, v in schedule.items()}

        return FreeAllocationScheduleResponse(
            status="success",
            schedule=schedule_str,
        )

    except Exception as exc:
        logger.error("Phase-out schedule failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/free-allocation/benchmarks",
    response_model=BenchmarkListResponse,
    summary="Get all product benchmarks",
    description=(
        "Get all CBAM product benchmarks with EU ETS free allocation values. "
        "Based on EU ETS Benchmark Decision 2021/927."
    ),
)
async def get_all_benchmarks() -> BenchmarkListResponse:
    """Get all product benchmarks."""
    try:
        engine = _get_free_alloc()
        factors = engine.get_benchmark_values()

        benchmarks = [
            {
                "product_benchmark": f.product_benchmark,
                "benchmark_value_tCO2e": str(f.benchmark_value_tCO2e),
                "allocation_percentage": str(f.allocation_percentage),
                "effective_allocation_tCO2e": str(f.effective_allocation_tCO2e),
                "phase": f.phase.value,
                "phase_label": f.phase.label,
                "year": f.year,
                "cn_codes": f.cn_codes,
            }
            for f in factors
        ]

        return BenchmarkListResponse(
            status="success",
            benchmarks=benchmarks,
        )

    except Exception as exc:
        logger.error("Benchmark listing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/free-allocation/{cn_code}/{year}",
    response_model=AllocationFactorResponse,
    summary="Get allocation factor for CN code and year",
    description=(
        "Get the free allocation factor for a specific product (CN code) "
        "and year. Returns benchmark value, allocation percentage, and "
        "effective deduction."
    ),
)
async def get_allocation_factor(
    cn_code: str = Path(
        ...,
        min_length=6,
        max_length=10,
        description="Combined Nomenclature code"
    ),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
) -> AllocationFactorResponse:
    """Get allocation factor for a specific CN code and year."""
    try:
        engine = _get_free_alloc()
        factor = engine.get_allocation_factor(cn_code, year)

        if factor is None:
            raise HTTPException(
                status_code=404,
                detail=f"No benchmark found for CN code '{cn_code}' in year {year}",
            )

        return AllocationFactorResponse(
            status="success",
            factor={
                "product_benchmark": factor.product_benchmark,
                "benchmark_value_tCO2e": str(factor.benchmark_value_tCO2e),
                "allocation_percentage": str(factor.allocation_percentage),
                "effective_allocation_tCO2e": str(factor.effective_allocation_tCO2e),
                "phase": factor.phase.value,
                "phase_label": factor.phase.label,
                "year": factor.year,
                "cn_codes": factor.cn_codes,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Allocation factor lookup failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put(
    "/free-allocation/{cn_code}/{year}",
    response_model=AllocationFactorResponse,
    summary="Update product benchmark",
    description=(
        "Update or override the EU ETS benchmark value for a product "
        "and year (admin operation)."
    ),
)
async def update_benchmark(
    cn_code: str = Path(
        ...,
        min_length=6,
        max_length=10,
        description="Combined Nomenclature code"
    ),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
    request: UpdateBenchmarkRequest = Body(...),
) -> AllocationFactorResponse:
    """Update a product benchmark value."""
    try:
        engine = _get_free_alloc()
        value = Decimal(str(request.benchmark_value_tco2e))

        factor = engine.update_benchmark(cn_code, year, value)

        logger.info(
            "API: benchmark updated, CN=%s, year=%d, value=%s",
            cn_code, year, value,
        )

        return AllocationFactorResponse(
            status="success",
            factor={
                "product_benchmark": factor.product_benchmark,
                "benchmark_value_tCO2e": str(factor.benchmark_value_tCO2e),
                "allocation_percentage": str(factor.allocation_percentage),
                "effective_allocation_tCO2e": str(factor.effective_allocation_tCO2e),
                "phase": factor.phase.value,
                "phase_label": factor.phase.label,
                "year": factor.year,
                "cn_codes": factor.cn_codes,
            },
        )

    except ValueError as exc:
        logger.warning("Benchmark update error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Benchmark update failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/free-allocation/compare",
    response_model=AllocationCompareResponse,
    summary="Compare allocation between years",
    description=(
        "Compare free allocation between two years for a given CN code. "
        "Shows the year-over-year change in CBAM obligation as free "
        "allocation declines."
    ),
)
async def compare_allocation_years(
    cn_code: str = Query(
        ...,
        min_length=6,
        max_length=10,
        description="Combined Nomenclature code"
    ),
    year_from: int = Query(
        ...,
        ge=2023,
        le=2099,
        description="Starting year"
    ),
    year_to: int = Query(
        ...,
        ge=2023,
        le=2099,
        description="Ending year"
    ),
) -> AllocationCompareResponse:
    """Compare free allocation between two years."""
    try:
        engine = _get_free_alloc()
        comparison = engine.compare_allocation_years(cn_code, year_from, year_to)

        if "error" in comparison:
            raise HTTPException(status_code=404, detail=comparison["error"])

        return AllocationCompareResponse(
            status="success",
            comparison=comparison,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Allocation comparison failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# DEDUCTION ENDPOINTS
# ============================================================================

@router.post(
    "/deductions/register",
    response_model=DeductionResponse,
    summary="Register carbon price deduction",
    description=(
        "Register a new carbon price deduction claim. "
        "Per Regulation Article 26, importers may deduct carbon prices "
        "effectively paid in the country of origin. The deduction is "
        "created in PENDING status and requires verification."
    ),
)
async def register_deduction(
    request: RegisterDeductionRequest = Body(...),
) -> DeductionResponse:
    """Register a new carbon price deduction."""
    start_time = time.time()

    try:
        engine = _get_deduction_engine()
        deduction = engine.register_deduction(
            deduction_id=request.deduction_id,
            importer_id=request.importer_id,
            installation_id=request.installation_id,
            country=request.country,
            pricing_scheme=request.pricing_scheme,
            carbon_price_paid_local=Decimal(str(request.carbon_price_paid_local)),
            currency=request.currency,
            tonnes_covered=Decimal(str(request.tonnes_covered)),
            evidence_docs=request.evidence_docs,
            year=request.year,
        )

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "API: deduction registered, id=%s, EUR %s, in %.1fms",
            deduction.deduction_id, deduction.carbon_price_paid_eur, duration_ms,
        )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except ValueError as exc:
        logger.warning("Deduction registration error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Deduction registration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/deductions/{importer_id}/{year}",
    response_model=DeductionListResponse,
    summary="Get deductions for importer and year",
    description="Get all carbon price deductions for an importer and year.",
)
async def get_deductions(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
) -> DeductionListResponse:
    """Get deductions for an importer and year."""
    try:
        engine = _get_deduction_engine()
        deductions = engine.get_deductions(importer_id, year)

        records = [_deduction_to_dict(d) for d in deductions]

        return DeductionListResponse(
            status="success",
            count=len(records),
            deductions=records,
        )

    except Exception as exc:
        logger.error("Deduction listing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/deductions/detail/{deduction_id}",
    response_model=DeductionResponse,
    summary="Get specific deduction",
    description="Get details for a specific carbon price deduction by ID.",
)
async def get_deduction_detail(
    deduction_id: str = Path(
        ...,
        min_length=5,
        description="Deduction identifier"
    ),
) -> DeductionResponse:
    """Get a specific deduction by ID."""
    try:
        engine = _get_deduction_engine()
        deduction = engine.get_deduction(deduction_id)

        if deduction is None:
            raise HTTPException(
                status_code=404,
                detail=f"Deduction '{deduction_id}' not found",
            )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Deduction detail failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/deductions/{deduction_id}/verify",
    response_model=DeductionResponse,
    summary="Verify a deduction",
    description=(
        "Mark a deduction as verified (PENDING -> VERIFIED). "
        "Per Article 26(2), the competent authority must verify "
        "the carbon price was effectively paid."
    ),
)
async def verify_deduction(
    deduction_id: str = Path(
        ...,
        min_length=5,
        description="Deduction identifier"
    ),
    request: VerifyDeductionRequest = Body(...),
) -> DeductionResponse:
    """Verify a carbon price deduction."""
    try:
        engine = _get_deduction_engine()
        deduction = engine.verify_deduction(
            deduction_id=deduction_id,
            verified_by=request.verified_by,
        )

        logger.info(
            "API: deduction verified, id=%s, by=%s",
            deduction_id, request.verified_by,
        )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except ValueError as exc:
        logger.warning("Deduction verification error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Deduction verification failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/deductions/{deduction_id}/approve",
    response_model=DeductionResponse,
    summary="Approve a deduction",
    description=(
        "Approve a verified deduction (VERIFIED -> APPROVED). "
        "The deduction becomes eligible for certificate obligation reduction."
    ),
)
async def approve_deduction(
    deduction_id: str = Path(
        ...,
        min_length=5,
        description="Deduction identifier"
    ),
    request: ApproveDeductionRequest = Body(...),
) -> DeductionResponse:
    """Approve a carbon price deduction."""
    try:
        engine = _get_deduction_engine()
        deduction = engine.approve_deduction(
            deduction_id=deduction_id,
            approved_by=request.approved_by,
        )

        logger.info(
            "API: deduction approved, id=%s, by=%s",
            deduction_id, request.approved_by,
        )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except ValueError as exc:
        logger.warning("Deduction approval error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Deduction approval failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/deductions/{deduction_id}/reject",
    response_model=DeductionResponse,
    summary="Reject a deduction",
    description=(
        "Reject a deduction (PENDING or VERIFIED -> REJECTED). "
        "The deduction will not be eligible for obligation reduction."
    ),
)
async def reject_deduction(
    deduction_id: str = Path(
        ...,
        min_length=5,
        description="Deduction identifier"
    ),
    request: RejectDeductionRequest = Body(...),
) -> DeductionResponse:
    """Reject a carbon price deduction."""
    try:
        engine = _get_deduction_engine()
        deduction = engine.reject_deduction(
            deduction_id=deduction_id,
            rejected_by=request.rejected_by,
            reason=request.reason,
        )

        logger.info(
            "API: deduction rejected, id=%s, by=%s, reason=%s",
            deduction_id, request.rejected_by, request.reason,
        )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except ValueError as exc:
        logger.warning("Deduction rejection error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Deduction rejection failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post(
    "/deductions/{deduction_id}/evidence",
    response_model=DeductionResponse,
    summary="Add evidence document",
    description=(
        "Add an evidence document reference to a deduction. "
        "Per Implementing Regulation 2023/1773 Article 7, importers must "
        "provide evidence of the carbon price effectively paid."
    ),
)
async def add_evidence(
    deduction_id: str = Path(
        ...,
        min_length=5,
        description="Deduction identifier"
    ),
    request: AddEvidenceRequest = Body(...),
) -> DeductionResponse:
    """Add an evidence document to a deduction."""
    try:
        engine = _get_deduction_engine()
        deduction = engine.add_evidence(
            deduction_id=deduction_id,
            document_ref=request.document_ref,
        )

        logger.info(
            "API: evidence added, deduction=%s, doc=%s",
            deduction_id, request.document_ref,
        )

        return DeductionResponse(
            status="success",
            deduction=_deduction_to_dict(deduction),
        )

    except ValueError as exc:
        logger.warning("Add evidence error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Add evidence failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/deductions/{importer_id}/{year}/summary",
    response_model=DeductionSummaryResponse,
    summary="Get deduction summary",
    description=(
        "Get a detailed summary of all deductions for an importer and year. "
        "Includes breakdowns by country and pricing scheme, with totals "
        "for eligible, pending, and rejected deductions."
    ),
)
async def get_deduction_summary(
    importer_id: str = Path(..., min_length=1, description="Importer identifier"),
    year: int = Path(..., ge=2023, le=2099, description="Reference year"),
) -> DeductionSummaryResponse:
    """Get deduction summary for an importer and year."""
    try:
        engine = _get_deduction_engine()
        summary = engine.get_deduction_summary(importer_id, year)

        return DeductionSummaryResponse(
            status="success",
            summary=summary,
        )

    except Exception as exc:
        logger.error("Deduction summary failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# COUNTRY PRICING ENDPOINT
# ============================================================================

@router.get(
    "/country-pricing/{country}",
    response_model=CountryPricingResponse,
    summary="Get country carbon pricing info",
    description=(
        "Get carbon pricing information for a specific country. "
        "Returns scheme type, effective price, currency, and "
        "eligibility for CBAM Article 26 deduction."
    ),
)
async def get_country_pricing(
    country: str = Path(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code"
    ),
) -> CountryPricingResponse:
    """Get country carbon pricing information."""
    try:
        engine = _get_deduction_engine()
        pricing = engine.get_country_carbon_pricing(country)

        return CountryPricingResponse(
            status="success",
            pricing=pricing,
        )

    except Exception as exc:
        logger.error("Country pricing lookup failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
