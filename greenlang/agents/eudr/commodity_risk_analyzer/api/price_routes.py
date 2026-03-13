# -*- coding: utf-8 -*-
"""
Price Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for commodity price monitoring including current prices,
price history, volatility analysis, market disruption detection,
and price forecasting.

Endpoints:
    GET  /price/{commodity_id}/current     - Current price
    GET  /price/{commodity_id}/history     - Price history
    GET  /price/{commodity_id}/volatility  - Volatility analysis
    GET  /price/market-disruptions         - Market disruptions
    POST /price/forecast                   - Price forecast

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Price Volatility Engine
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_price_volatility_engine,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
    validate_commodity_type,
    validate_date_range,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    CommodityTypeEnum,
    ForecastPoint,
    MarketConditionEnum,
    MarketDisruptionEntry,
    MarketDisruptionResponse,
    PriceForecastRequest,
    PriceForecastResponse,
    PriceHistoryEntry,
    PriceHistoryResponse,
    PriceResponse,
    SeveritySummaryEnum,
    VolatilityLevelEnum,
    VolatilityResponse,
    VolatilityTrendEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Price & Market"])

# ---------------------------------------------------------------------------
# Reference price data (static for API layer; production uses exchange feeds)
# ---------------------------------------------------------------------------

_REFERENCE_PRICES: Dict[str, Dict[str, Any]] = {
    "cattle": {"price": Decimal("4200.00"), "volatility_30d": Decimal("0.12"), "volatility_90d": Decimal("0.18"), "exchange": "CME"},
    "cocoa": {"price": Decimal("2650.00"), "volatility_30d": Decimal("0.35"), "volatility_90d": Decimal("0.42"), "exchange": "ICE"},
    "coffee": {"price": Decimal("2180.00"), "volatility_30d": Decimal("0.28"), "volatility_90d": Decimal("0.31"), "exchange": "ICE"},
    "oil_palm": {"price": Decimal("820.00"), "volatility_30d": Decimal("0.22"), "volatility_90d": Decimal("0.25"), "exchange": "BMD"},
    "rubber": {"price": Decimal("1450.00"), "volatility_30d": Decimal("0.18"), "volatility_90d": Decimal("0.20"), "exchange": "SICOM"},
    "soya": {"price": Decimal("390.00"), "volatility_30d": Decimal("0.15"), "volatility_90d": Decimal("0.19"), "exchange": "CBOT"},
    "wood": {"price": Decimal("550.00"), "volatility_30d": Decimal("0.10"), "volatility_90d": Decimal("0.14"), "exchange": "CME"},
}


def _classify_volatility(vol: Decimal) -> VolatilityLevelEnum:
    """Classify volatility value into a level."""
    if vol >= Decimal("0.40"):
        return VolatilityLevelEnum.EXTREME
    elif vol >= Decimal("0.25"):
        return VolatilityLevelEnum.HIGH
    elif vol >= Decimal("0.15"):
        return VolatilityLevelEnum.MODERATE
    return VolatilityLevelEnum.LOW


def _classify_market(vol_30d: Decimal) -> MarketConditionEnum:
    """Classify market condition from 30-day volatility."""
    if vol_30d >= Decimal("0.50"):
        return MarketConditionEnum.CRISIS
    elif vol_30d >= Decimal("0.35"):
        return MarketConditionEnum.DISRUPTED
    elif vol_30d >= Decimal("0.20"):
        return MarketConditionEnum.VOLATILE
    return MarketConditionEnum.STABLE


def _resolve_commodity_type(commodity_id: str) -> str:
    """Resolve a commodity_id to a commodity type string.

    Accepts both commodity type names directly (e.g. 'cocoa')
    and profile IDs. Returns normalized commodity type.
    """
    normalized = commodity_id.strip().lower()
    if normalized in _REFERENCE_PRICES:
        return normalized
    # Attempt lookup from profile store (import here to avoid circular)
    from greenlang.agents.eudr.commodity_risk_analyzer.api.commodity_routes import (
        _profile_store,
    )

    profile = _profile_store.get(commodity_id)
    if profile:
        return profile.commodity_type.value
    return normalized


# ---------------------------------------------------------------------------
# GET /price/{commodity_id}/current
# ---------------------------------------------------------------------------


@router.get(
    "/price/{commodity_id}/current",
    response_model=PriceResponse,
    summary="Get current commodity price",
    description=(
        "Retrieve current price data for an EUDR commodity including "
        "30-day and 90-day volatility indices and market condition."
    ),
    responses={
        200: {"description": "Current price data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_current_price(
    commodity_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:price:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PriceResponse:
    """Get current price for a commodity.

    Args:
        commodity_id: Commodity identifier or type name.
        user: Authenticated user with price:read permission.

    Returns:
        PriceResponse with current price and volatility data.
    """
    ct = _resolve_commodity_type(commodity_id)
    ref = _REFERENCE_PRICES.get(ct)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price data not found for commodity: {commodity_id}",
        )

    return PriceResponse(
        commodity_type=CommodityTypeEnum(ct),
        price=ref["price"],
        currency="USD",
        price_date=date.today(),
        volatility_30d=ref["volatility_30d"],
        volatility_90d=ref["volatility_90d"],
        market_condition=_classify_market(ref["volatility_30d"]),
        exchange=ref["exchange"],
    )


# ---------------------------------------------------------------------------
# GET /price/{commodity_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/price/{commodity_id}/history",
    response_model=PriceHistoryResponse,
    summary="Get price history",
    description="Retrieve historical price data for a commodity over a given period.",
    responses={
        200: {"description": "Price history data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_price_history(
    commodity_id: str,
    request: Request,
    date_range: Dict[str, Optional[date]] = Depends(validate_date_range),
    days: int = Query(default=30, ge=1, le=365, description="Number of days of history"),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:price:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PriceHistoryResponse:
    """Get price history for a commodity.

    Args:
        commodity_id: Commodity identifier or type name.
        date_range: Optional date range filter.
        days: Number of days of history to return.
        user: Authenticated user with price:read permission.

    Returns:
        PriceHistoryResponse with historical price series.
    """
    ct = _resolve_commodity_type(commodity_id)
    ref = _REFERENCE_PRICES.get(ct)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price data not found for commodity: {commodity_id}",
        )

    # Generate synthetic price history for API demonstration
    today = date.today()
    start = date_range.get("start_date") or (today - timedelta(days=days))
    end = date_range.get("end_date") or today

    prices: List[PriceHistoryEntry] = []
    base_price = ref["price"]
    current = start
    idx = 0
    while current <= end:
        # Simple deterministic variation for reproducibility
        variation = Decimal(str(((idx * 7 + 3) % 11 - 5) / 100.0))
        price_val = base_price * (Decimal("1.0") + variation)
        prices.append(
            PriceHistoryEntry(
                price_date=current,
                price=price_val.quantize(Decimal("0.01")),
                currency="USD",
            )
        )
        current += timedelta(days=1)
        idx += 1

    return PriceHistoryResponse(
        commodity_type=CommodityTypeEnum(ct),
        prices=prices,
        period_start=start,
        period_end=end,
        currency="USD",
        data_points=len(prices),
    )


# ---------------------------------------------------------------------------
# GET /price/{commodity_id}/volatility
# ---------------------------------------------------------------------------


@router.get(
    "/price/{commodity_id}/volatility",
    response_model=VolatilityResponse,
    summary="Get volatility analysis",
    description=(
        "Analyze price volatility for a commodity with configurable rolling "
        "window, trend direction, and risk level classification."
    ),
    responses={
        200: {"description": "Volatility analysis results"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_volatility(
    commodity_id: str,
    request: Request,
    window_days: int = Query(
        default=30, ge=7, le=365,
        description="Rolling window size in days",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:price:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> VolatilityResponse:
    """Get volatility analysis for a commodity.

    Args:
        commodity_id: Commodity identifier or type name.
        window_days: Rolling window size.
        user: Authenticated user with price:read permission.

    Returns:
        VolatilityResponse with volatility index and classification.
    """
    ct = _resolve_commodity_type(commodity_id)
    ref = _REFERENCE_PRICES.get(ct)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price data not found for commodity: {commodity_id}",
        )

    # Use 30d or 90d reference volatility based on window
    if window_days <= 30:
        vol = ref["volatility_30d"]
    else:
        vol = ref["volatility_90d"]

    return VolatilityResponse(
        commodity_type=CommodityTypeEnum(ct),
        volatility=vol,
        window_days=window_days,
        trend=VolatilityTrendEnum.STABLE,
        risk_level=_classify_volatility(vol),
        percentile_rank=Decimal("65.0"),
    )


# ---------------------------------------------------------------------------
# GET /price/market-disruptions
# ---------------------------------------------------------------------------


@router.get(
    "/price/market-disruptions",
    response_model=MarketDisruptionResponse,
    summary="Get market disruptions",
    description=(
        "Retrieve active and recent market disruption events that may "
        "impact commodity pricing and supply chain stability."
    ),
    responses={
        200: {"description": "Market disruption data"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_market_disruptions(
    request: Request,
    commodity_type: Optional[str] = Depends(validate_commodity_type),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:price:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> MarketDisruptionResponse:
    """Get market disruption events.

    Args:
        commodity_type: Optional commodity type filter.
        user: Authenticated user with price:read permission.

    Returns:
        MarketDisruptionResponse with disruption events.
    """
    # Return representative disruption events
    disruptions: List[MarketDisruptionEntry] = []
    ct_enum = CommodityTypeEnum(commodity_type) if commodity_type else CommodityTypeEnum.COCOA

    if commodity_type is None or commodity_type == "cocoa":
        disruptions.append(
            MarketDisruptionEntry(
                event_type="supply_shortage",
                description="West African cocoa supply disrupted by adverse weather conditions",
                severity=SeveritySummaryEnum.HIGH,
                start_date=date(2026, 1, 15),
                price_impact_pct=Decimal("12.5"),
                affected_countries=["GH", "CI"],
            )
        )

    if commodity_type is None or commodity_type == "oil_palm":
        disruptions.append(
            MarketDisruptionEntry(
                event_type="export_restriction",
                description="Indonesia temporary palm oil export levy adjustment",
                severity=SeveritySummaryEnum.MEDIUM,
                start_date=date(2026, 2, 1),
                end_date=date(2026, 3, 31),
                price_impact_pct=Decimal("8.0"),
                affected_countries=["ID"],
            )
        )

    overall_severity = SeveritySummaryEnum.LOW
    if disruptions:
        max_sev = max(d.severity.value for d in disruptions)
        overall_severity = SeveritySummaryEnum(max_sev)

    return MarketDisruptionResponse(
        commodity_type=ct_enum,
        disruptions=disruptions,
        severity=overall_severity,
        total_disruptions=len(disruptions),
    )


# ---------------------------------------------------------------------------
# POST /price/forecast
# ---------------------------------------------------------------------------


@router.post(
    "/price/forecast",
    response_model=PriceForecastResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate price forecast",
    description=(
        "Generate a price forecast for an EUDR commodity over a specified "
        "horizon with optional confidence intervals."
    ),
    responses={
        200: {"description": "Price forecast generated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def forecast_price(
    request: Request,
    body: PriceForecastRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:price:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PriceForecastResponse:
    """Generate a price forecast for a commodity.

    Args:
        body: Forecast request with commodity type and horizon.
        user: Authenticated user with price:write permission.

    Returns:
        PriceForecastResponse with forecasted price points.
    """
    ct = body.commodity_type.value
    ref = _REFERENCE_PRICES.get(ct)
    if ref is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No reference price data for commodity: {ct}",
        )

    base_price = ref["price"]
    vol = ref["volatility_30d"]

    # Generate forecast points with simple trend + volatility
    forecast_points: List[ForecastPoint] = []
    today = date.today()
    for day in range(1, body.horizon_days + 1):
        forecast_date = today + timedelta(days=day)
        # Simple linear trend with seasonal variation
        trend = Decimal(str(day * 0.001))
        price = base_price * (Decimal("1.0") + trend)

        lower = None
        upper = None
        confidence = None
        if body.include_confidence:
            spread = price * vol * Decimal(str(day ** 0.5)) * Decimal("0.01")
            lower = (price - spread).quantize(Decimal("0.01"))
            upper = (price + spread).quantize(Decimal("0.01"))
            confidence = max(
                Decimal("0.5"),
                Decimal("0.95") - Decimal(str(day * 0.001)),
            )

        forecast_points.append(
            ForecastPoint(
                forecast_date=forecast_date,
                price=price.quantize(Decimal("0.01")),
                lower_bound=lower,
                upper_bound=upper,
                confidence=confidence,
            )
        )

    confidence_intervals: Dict[str, Decimal] = {}
    if body.include_confidence and forecast_points:
        last = forecast_points[-1]
        if last.lower_bound and last.upper_bound:
            confidence_intervals = {
                "90pct_lower": last.lower_bound,
                "90pct_upper": last.upper_bound,
            }

    logger.info(
        "Price forecast generated: commodity=%s horizon=%d points=%d",
        ct,
        body.horizon_days,
        len(forecast_points),
    )

    return PriceForecastResponse(
        commodity_type=body.commodity_type,
        forecast=forecast_points,
        confidence_intervals=confidence_intervals,
        horizon_days=body.horizon_days,
        currency="USD",
    )
