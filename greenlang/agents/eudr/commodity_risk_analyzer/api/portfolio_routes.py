# -*- coding: utf-8 -*-
"""
Portfolio Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for multi-commodity portfolio risk aggregation including
portfolio analysis, concentration index calculation, diversification
scoring, and portfolio summary.

Endpoints:
    POST /portfolio/analyze          - Analyze portfolio
    GET  /portfolio/concentration    - Concentration index
    GET  /portfolio/diversification  - Diversification score
    GET  /portfolio/summary          - Portfolio summary

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Portfolio Risk Aggregator Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    get_portfolio_risk_aggregator,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    CommodityBreakdownEntry,
    CommodityShareEntry,
    CommodityTypeEnum,
    ConcentrationResponse,
    DiversificationResponse,
    DiversificationSuggestion,
    PortfolioAnalyzeRequest,
    PortfolioResponse,
    PortfolioSummaryResponse,
    ProvenanceInfo,
    SeveritySummaryEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Portfolio Aggregation"])

# ---------------------------------------------------------------------------
# In-memory portfolio store (replaced by database in production)
# ---------------------------------------------------------------------------

_portfolio_store: Dict[str, PortfolioResponse] = {}
_latest_portfolio_id: Optional[str] = None


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _compute_hhi(shares: List[Decimal]) -> Decimal:
    """Compute Herfindahl-Hirschman Index from market shares.

    HHI = sum(share_i^2) where shares are proportions summing to 1.0.
    Range: 1/n (perfect diversification) to 1.0 (perfect concentration).

    Args:
        shares: List of share proportions (0-1).

    Returns:
        HHI value between 0 and 1.
    """
    if not shares:
        return Decimal("0.0")
    return sum(s * s for s in shares)


def _classify_concentration(hhi: Decimal) -> str:
    """Classify HHI concentration level.

    Args:
        hhi: HHI value (0-1).

    Returns:
        Classification string.
    """
    if hhi >= Decimal("0.5"):
        return "highly_concentrated"
    elif hhi >= Decimal("0.25"):
        return "concentrated"
    elif hhi >= Decimal("0.15"):
        return "moderate"
    return "diversified"


def _compute_diversification_score(hhi: Decimal, commodity_count: int) -> Decimal:
    """Compute diversification score from HHI and commodity count.

    Higher score = more diversified. Score of 1.0 = perfectly diversified.

    Args:
        hhi: HHI value (0-1).
        commodity_count: Number of distinct commodities.

    Returns:
        Diversification score (0-1).
    """
    if commodity_count <= 1:
        return Decimal("0.0")

    # Inverse HHI scaled by commodity count potential
    max_diversity = Decimal("1.0") / Decimal(str(commodity_count))
    diversity_ratio = max_diversity / hhi if hhi > Decimal("0") else Decimal("1.0")
    return min(Decimal("1.0"), diversity_ratio).quantize(Decimal("0.01"))


# ---------------------------------------------------------------------------
# POST /portfolio/analyze
# ---------------------------------------------------------------------------


@router.post(
    "/portfolio/analyze",
    response_model=PortfolioResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze commodity portfolio",
    description=(
        "Perform a comprehensive risk analysis on a multi-commodity portfolio "
        "including HHI concentration indexing, diversification scoring, "
        "total risk exposure quantification, and optimization recommendations."
    ),
    responses={
        200: {"description": "Portfolio analysis completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def analyze_portfolio(
    request: Request,
    body: PortfolioAnalyzeRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:portfolio:write")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> PortfolioResponse:
    """Analyze a multi-commodity portfolio for risk.

    Args:
        body: Portfolio analysis request with positions.
        user: Authenticated user with portfolio:write permission.

    Returns:
        PortfolioResponse with concentration, diversification, and risk breakdown.
    """
    global _latest_portfolio_id
    start = time.monotonic()
    try:
        portfolio_id = str(uuid.uuid4())

        # Compute total volume
        total_volume = sum(p.volume for p in body.commodity_positions)
        if total_volume <= Decimal("0"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Total portfolio volume must be greater than zero",
            )

        # Compute shares and risk breakdown
        shares: List[Decimal] = []
        breakdown: List[CommodityBreakdownEntry] = []
        total_risk_weighted = Decimal("0.0")

        for pos in body.commodity_positions:
            share_pct = (pos.volume / total_volume * Decimal("100.0")).quantize(
                Decimal("0.01")
            )
            share_prop = pos.volume / total_volume
            shares.append(share_prop)

            # Use provided risk or default
            risk_score = pos.risk_score or Decimal("50.0")
            risk_contribution = (share_prop * risk_score).quantize(Decimal("0.01"))
            total_risk_weighted += share_prop * risk_score

            breakdown.append(
                CommodityBreakdownEntry(
                    commodity_type=pos.commodity_type,
                    volume=pos.volume,
                    share_pct=share_pct,
                    risk_score=risk_score,
                    risk_contribution=risk_contribution,
                )
            )

        # Compute HHI and diversification
        hhi = _compute_hhi(shares).quantize(Decimal("0.0001"))
        commodity_count = len(body.commodity_positions)
        div_score = _compute_diversification_score(hhi, commodity_count)
        total_risk = min(Decimal("100.0"), total_risk_weighted).quantize(
            Decimal("0.01")
        )

        # Generate recommendations
        recommendations: List[str] = []
        classification = _classify_concentration(hhi)

        if classification in ("concentrated", "highly_concentrated"):
            top = max(breakdown, key=lambda b: b.share_pct)
            recommendations.append(
                f"Portfolio is {classification} with {top.commodity_type.value} "
                f"representing {top.share_pct}% of volume. Consider diversifying."
            )
        if commodity_count < 3:
            recommendations.append(
                "Consider adding more commodity types to reduce concentration risk."
            )

        high_risk = [b for b in breakdown if b.risk_score >= Decimal("70.0")]
        if high_risk:
            names = [b.commodity_type.value for b in high_risk]
            recommendations.append(
                f"High-risk commodities detected: {', '.join(names)}. "
                f"Increase due diligence or consider substitution."
            )

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            body.model_dump_json(), f"{portfolio_id}:{total_risk}"
        )

        response = PortfolioResponse(
            portfolio_id=portfolio_id,
            portfolio_name=body.portfolio_name,
            concentration_index=hhi,
            diversification_score=div_score,
            total_risk_exposure=total_risk,
            commodity_breakdown=breakdown,
            recommendations=recommendations,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

        _portfolio_store[portfolio_id] = response
        _latest_portfolio_id = portfolio_id

        logger.info(
            "Portfolio analyzed: id=%s name=%s hhi=%s risk=%s commodities=%d",
            portfolio_id,
            body.portfolio_name,
            hhi,
            total_risk,
            commodity_count,
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Portfolio analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Portfolio analysis failed",
        )


# ---------------------------------------------------------------------------
# GET /portfolio/concentration
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/concentration",
    response_model=ConcentrationResponse,
    summary="Get portfolio concentration index",
    description=(
        "Retrieve the Herfindahl-Hirschman Index (HHI) concentration analysis "
        "for the most recently analyzed portfolio."
    ),
    responses={
        200: {"description": "Concentration analysis"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No portfolio analyzed yet"},
    },
)
async def get_concentration(
    request: Request,
    portfolio_id: Optional[str] = Query(
        None,
        description="Portfolio ID. Uses latest if not specified.",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:portfolio:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ConcentrationResponse:
    """Get concentration index for a portfolio.

    Args:
        portfolio_id: Optional portfolio ID (uses latest if omitted).
        user: Authenticated user with portfolio:read permission.

    Returns:
        ConcentrationResponse with HHI and commodity shares.
    """
    pid = portfolio_id or _latest_portfolio_id
    if pid is None or pid not in _portfolio_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio analysis found. Run /portfolio/analyze first.",
        )

    portfolio = _portfolio_store[pid]

    # Build commodity shares
    commodity_shares: List[CommodityShareEntry] = [
        CommodityShareEntry(
            commodity_type=b.commodity_type,
            share_pct=b.share_pct,
            volume=b.volume,
        )
        for b in portfolio.commodity_breakdown
    ]

    # Find top commodity
    top_commodity = None
    if commodity_shares:
        top = max(commodity_shares, key=lambda s: s.share_pct)
        top_commodity = top.commodity_type

    return ConcentrationResponse(
        hhi_index=portfolio.concentration_index,
        classification=_classify_concentration(portfolio.concentration_index),
        commodity_shares=commodity_shares,
        top_commodity=top_commodity,
    )


# ---------------------------------------------------------------------------
# GET /portfolio/diversification
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/diversification",
    response_model=DiversificationResponse,
    summary="Get portfolio diversification score",
    description=(
        "Retrieve diversification analysis for the most recently analyzed "
        "portfolio with improvement suggestions."
    ),
    responses={
        200: {"description": "Diversification analysis"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No portfolio analyzed yet"},
    },
)
async def get_diversification(
    request: Request,
    portfolio_id: Optional[str] = Query(
        None,
        description="Portfolio ID. Uses latest if not specified.",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:portfolio:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> DiversificationResponse:
    """Get diversification analysis for a portfolio.

    Args:
        portfolio_id: Optional portfolio ID (uses latest if omitted).
        user: Authenticated user with portfolio:read permission.

    Returns:
        DiversificationResponse with score and improvement suggestions.
    """
    pid = portfolio_id or _latest_portfolio_id
    if pid is None or pid not in _portfolio_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio analysis found. Run /portfolio/analyze first.",
        )

    portfolio = _portfolio_store[pid]
    div_score = portfolio.diversification_score
    commodity_count = len(portfolio.commodity_breakdown)

    # Describe current state
    if div_score >= Decimal("0.8"):
        state = f"Well diversified across {commodity_count} commodities"
    elif div_score >= Decimal("0.5"):
        state = f"Moderately diversified across {commodity_count} commodities"
    elif div_score >= Decimal("0.3"):
        state = f"Poorly diversified across {commodity_count} commodities"
    else:
        state = f"Highly concentrated with {commodity_count} commodity(ies)"

    # Generate improvement suggestions
    suggestions: List[DiversificationSuggestion] = []

    if commodity_count < 4:
        suggestions.append(
            DiversificationSuggestion(
                action="Add new commodity types to the portfolio",
                impact="Reduces concentration risk and HHI index",
                priority=SeveritySummaryEnum.HIGH,
            )
        )

    if portfolio.concentration_index > Decimal("0.25"):
        top = max(portfolio.commodity_breakdown, key=lambda b: b.share_pct)
        suggestions.append(
            DiversificationSuggestion(
                action=f"Reduce {top.commodity_type.value} share from {top.share_pct}%",
                impact="Lower HHI concentration and balanced risk distribution",
                priority=SeveritySummaryEnum.MEDIUM,
            )
        )

    if div_score < Decimal("0.5"):
        suggestions.append(
            DiversificationSuggestion(
                action="Equalize volume distribution across commodities",
                impact="Improved diversification score and reduced single-commodity dependency",
                priority=SeveritySummaryEnum.HIGH,
            )
        )

    return DiversificationResponse(
        score=div_score,
        current_state=state,
        improvement_suggestions=suggestions,
        commodity_count=commodity_count,
    )


# ---------------------------------------------------------------------------
# GET /portfolio/summary
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/summary",
    response_model=PortfolioSummaryResponse,
    summary="Get portfolio summary",
    description="Retrieve a high-level summary of the most recently analyzed portfolio.",
    responses={
        200: {"description": "Portfolio summary"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "No portfolio analyzed yet"},
    },
)
async def get_portfolio_summary(
    request: Request,
    portfolio_id: Optional[str] = Query(
        None,
        description="Portfolio ID. Uses latest if not specified.",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:portfolio:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> PortfolioSummaryResponse:
    """Get portfolio summary.

    Args:
        portfolio_id: Optional portfolio ID (uses latest if omitted).
        user: Authenticated user with portfolio:read permission.

    Returns:
        PortfolioSummaryResponse with high-level metrics.
    """
    pid = portfolio_id or _latest_portfolio_id
    if pid is None or pid not in _portfolio_store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No portfolio analysis found. Run /portfolio/analyze first.",
        )

    portfolio = _portfolio_store[pid]

    # Compute aggregates
    total_volume = sum(b.volume for b in portfolio.commodity_breakdown)
    commodity_count = len(portfolio.commodity_breakdown)

    # Average risk score (volume-weighted)
    avg_risk = portfolio.total_risk_exposure

    # Highest risk commodity
    highest_risk = None
    if portfolio.commodity_breakdown:
        top = max(portfolio.commodity_breakdown, key=lambda b: b.risk_score)
        highest_risk = top.commodity_type

    return PortfolioSummaryResponse(
        portfolio_name=portfolio.portfolio_name,
        total_volume=total_volume,
        commodity_count=commodity_count,
        average_risk_score=avg_risk,
        hhi_index=portfolio.concentration_index,
        diversification_score=portfolio.diversification_score,
        highest_risk_commodity=highest_risk,
    )
