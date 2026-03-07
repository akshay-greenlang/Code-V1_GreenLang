# -*- coding: utf-8 -*-
"""
Accuracy Scoring Routes - AGENT-EUDR-002 Geolocation Verification API

Endpoints for retrieving, summarizing, and configuring accuracy scores.
Scores are computed by the AccuracyScoringEngine using Decimal arithmetic
for bit-perfect reproducibility.

Endpoints:
    GET  /scores/{plot_id}          - Get current accuracy score for a plot
    GET  /scores/{plot_id}/history  - Get score history for a plot
    GET  /scores/summary            - Aggregate score statistics
    PUT  /scores/weights            - Update score weights (admin only)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-002 Geolocation Verification Agent (GL-EUDR-GEO-002)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.geolocation_verification.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_accuracy_scorer,
    get_pagination,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
)
from greenlang.agents.eudr.geolocation_verification.api.schemas import (
    AccuracyScoreResponse,
    PaginatedMeta,
    ScoreHistoryResponse,
    ScoreSummaryResponse,
    ScoreWeightsResponse,
    ScoreWeightsUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Accuracy Scoring"])


# ---------------------------------------------------------------------------
# GET /scores/{plot_id}
# ---------------------------------------------------------------------------


@router.get(
    "/scores/{plot_id}",
    response_model=AccuracyScoreResponse,
    summary="Get current accuracy score for a plot",
    description=(
        "Retrieve the most recent composite accuracy score for a production "
        "plot. The score is computed from six weighted sub-components: "
        "coordinate precision, polygon quality, country match, protected "
        "area status, deforestation status, and temporal consistency. "
        "All scores use Decimal arithmetic for bit-perfect reproducibility."
    ),
    responses={
        200: {"description": "Accuracy score"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Score not found"},
    },
)
async def get_accuracy_score(
    plot_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:scores:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> AccuracyScoreResponse:
    """Get the current accuracy score for a plot.

    Args:
        plot_id: Plot identifier.
        user: Authenticated user with scores:read permission.

    Returns:
        AccuracyScoreResponse with composite score and sub-component breakdown.

    Raises:
        HTTPException: 404 if no score found for plot.
    """
    logger.info(
        "Accuracy score request: user=%s plot_id=%s",
        user.user_id,
        plot_id,
    )

    try:
        scorer = get_accuracy_scorer()
        score = scorer.get_latest_score(plot_id=plot_id)

        if score is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No accuracy score found for plot {plot_id}",
            )

        return AccuracyScoreResponse(
            score_id=score.score_id,
            plot_id=plot_id,
            total_score=float(score.total_score),
            coordinate_precision_score=float(score.coordinate_precision_score),
            polygon_quality_score=float(score.polygon_quality_score),
            country_match_score=float(score.country_match_score),
            protected_area_score=float(score.protected_area_score),
            deforestation_score=float(score.deforestation_score),
            temporal_consistency_score=float(score.temporal_consistency_score),
            quality_tier=score.quality_tier.value,
            weights_used={k: float(v) for k, v in score.weights_used.items()},
            provenance_hash=score.provenance_hash,
            scored_at=score.scored_at,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Accuracy score retrieval failed: plot_id=%s error=%s",
            plot_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No accuracy score found for plot {plot_id}",
        )


# ---------------------------------------------------------------------------
# GET /scores/{plot_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/scores/{plot_id}/history",
    response_model=ScoreHistoryResponse,
    summary="Get score history for a plot",
    description=(
        "Retrieve the full scoring history for a production plot with "
        "pagination. Returns scores ordered by date with the most "
        "recent first. Useful for tracking score trends over time."
    ),
    responses={
        200: {"description": "Score history"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_score_history(
    plot_id: str,
    request: Request,
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:scores:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ScoreHistoryResponse:
    """Get score history for a plot with pagination.

    Args:
        plot_id: Plot identifier.
        pagination: Limit and offset for pagination.
        user: Authenticated user with scores:read permission.

    Returns:
        ScoreHistoryResponse with paginated score history.
    """
    logger.info(
        "Score history request: user=%s plot_id=%s limit=%d offset=%d",
        user.user_id,
        plot_id,
        pagination.limit,
        pagination.offset,
    )

    try:
        scorer = get_accuracy_scorer()
        history = scorer.get_score_history(
            plot_id=plot_id,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        scores: List[AccuracyScoreResponse] = []
        total = 0

        if history is not None:
            total = getattr(history, "total", 0)
            raw_scores = getattr(history, "scores", [])
            for s in raw_scores:
                scores.append(
                    AccuracyScoreResponse(
                        score_id=s.score_id,
                        plot_id=plot_id,
                        total_score=float(s.total_score),
                        coordinate_precision_score=float(s.coordinate_precision_score),
                        polygon_quality_score=float(s.polygon_quality_score),
                        country_match_score=float(s.country_match_score),
                        protected_area_score=float(s.protected_area_score),
                        deforestation_score=float(s.deforestation_score),
                        temporal_consistency_score=float(s.temporal_consistency_score),
                        quality_tier=s.quality_tier.value,
                        weights_used={k: float(v) for k, v in s.weights_used.items()},
                        provenance_hash=s.provenance_hash,
                        scored_at=s.scored_at,
                    )
                )

        return ScoreHistoryResponse(
            plot_id=plot_id,
            total_scores=total,
            scores=scores,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as exc:
        logger.error(
            "Score history retrieval failed: plot_id=%s error=%s",
            plot_id,
            exc,
            exc_info=True,
        )
        # Return empty history rather than 500
        return ScoreHistoryResponse(
            plot_id=plot_id,
            total_scores=0,
            scores=[],
            meta=PaginatedMeta(
                total=0,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=False,
            ),
        )


# ---------------------------------------------------------------------------
# GET /scores/summary
# ---------------------------------------------------------------------------


@router.get(
    "/scores/summary",
    response_model=ScoreSummaryResponse,
    summary="Get aggregate score statistics",
    description=(
        "Retrieve aggregate accuracy score statistics across all plots "
        "accessible to the authenticated user. Includes average, median, "
        "min, max scores, quality tier distribution, and average sub-scores "
        "by component."
    ),
    responses={
        200: {"description": "Aggregate score statistics"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_score_summary(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:scores:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> ScoreSummaryResponse:
    """Get aggregate score statistics across all plots.

    Args:
        user: Authenticated user with scores:read permission.

    Returns:
        ScoreSummaryResponse with aggregate statistics and distributions.
    """
    logger.info(
        "Score summary request: user=%s operator=%s",
        user.user_id,
        user.operator_id,
    )

    try:
        scorer = get_accuracy_scorer()

        summary = scorer.get_summary(
            operator_id=user.operator_id or user.user_id,
        )

        if summary is None:
            # Return empty summary
            from greenlang.agents.eudr.geolocation_verification.config import get_config
            cfg = get_config()
            return ScoreSummaryResponse(
                total_plots_scored=0,
                average_score=0.0,
                median_score=0.0,
                min_score=0.0,
                max_score=0.0,
                quality_distribution={"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
                average_sub_scores={},
                current_weights=cfg.score_weights,
            )

        return ScoreSummaryResponse(
            total_plots_scored=getattr(summary, "total_plots_scored", 0),
            average_score=getattr(summary, "average_score", 0.0),
            median_score=getattr(summary, "median_score", 0.0),
            min_score=getattr(summary, "min_score", 0.0),
            max_score=getattr(summary, "max_score", 0.0),
            quality_distribution=getattr(
                summary, "quality_distribution",
                {"gold": 0, "silver": 0, "bronze": 0, "fail": 0},
            ),
            average_sub_scores=getattr(summary, "average_sub_scores", {}),
            current_weights=getattr(summary, "current_weights", {}),
        )

    except Exception as exc:
        logger.error(
            "Score summary failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        # Return empty summary rather than 500
        return ScoreSummaryResponse(
            total_plots_scored=0,
            average_score=0.0,
            median_score=0.0,
            min_score=0.0,
            max_score=0.0,
        )


# ---------------------------------------------------------------------------
# PUT /scores/weights
# ---------------------------------------------------------------------------


@router.put(
    "/scores/weights",
    response_model=ScoreWeightsResponse,
    summary="Update accuracy score weights (admin only)",
    description=(
        "Update the component weights used for computing composite accuracy "
        "scores. Requires admin role. Weights must contain exactly six keys "
        "(precision, polygon, country, protected, deforestation, temporal) "
        "and must sum to 1.0. Updated weights take effect for all "
        "subsequent scoring operations."
    ),
    responses={
        200: {"description": "Weights updated"},
        400: {"model": ErrorResponse, "description": "Invalid weights"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Admin role required"},
    },
)
async def update_score_weights(
    body: ScoreWeightsUpdateRequest,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-geolocation:scores:admin")
    ),
    _rate: None = Depends(rate_limit_write),
) -> ScoreWeightsResponse:
    """Update accuracy score component weights.

    Only users with the admin role can update weights. The new weights
    are validated for correct keys and sum, then applied to all
    subsequent scoring operations.

    Args:
        body: Weight update request with new weight values.
        user: Authenticated admin user with scores:admin permission.

    Returns:
        ScoreWeightsResponse with previous and new weights.

    Raises:
        HTTPException: 400 if weights invalid, 403 if not admin.
    """
    logger.info(
        "Score weights update: user=%s new_weights=%s",
        user.user_id,
        body.weights,
    )

    try:
        from greenlang.agents.eudr.geolocation_verification.config import (
            get_config,
            set_config,
        )

        cfg = get_config()
        previous_weights = dict(cfg.score_weights)

        # Update config with new weights
        cfg.score_weights = dict(body.weights)

        # Update the scorer instance if it exists
        scorer = get_accuracy_scorer()
        if hasattr(scorer, "update_weights"):
            scorer.update_weights(body.weights)

        now = datetime.now(timezone.utc).replace(microsecond=0)

        logger.info(
            "Score weights updated: user=%s previous=%s new=%s",
            user.user_id,
            previous_weights,
            body.weights,
        )

        return ScoreWeightsResponse(
            status="updated",
            previous_weights=previous_weights,
            new_weights=dict(body.weights),
            updated_at=now,
        )

    except ValueError as exc:
        logger.warning(
            "Score weights update error: user=%s error=%s",
            user.user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(
            "Score weights update failed: user=%s error=%s",
            user.user_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Score weights update failed due to an internal error",
        )
