# -*- coding: utf-8 -*-
"""
Feedback API Endpoints for GL-002 BoilerEfficiencyOptimizer

This module provides FastAPI endpoints for feedback submission and retrieval.
Implements RESTful API with proper error handling and validation.

Example:
    >>> from fastapi import FastAPI
    >>> app = FastAPI()
    >>> feedback_router = create_feedback_router(db_url="postgresql://...")
    >>> app.include_router(feedback_router, prefix="/api/v1/feedback")
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Path, Body, Depends
from fastapi.responses import JSONResponse
import logging

from .feedback_models import (
from greenlang.determinism import DeterministicClock
    OptimizationFeedback,
    FeedbackStats,
    SatisfactionTrend,
    FeedbackSummary,
    FeedbackCategory
)
from .feedback_collector import FeedbackCollector

logger = logging.getLogger(__name__)


def create_feedback_router(db_url: str) -> APIRouter:
    """
    Create FastAPI router for feedback endpoints.

    Args:
        db_url: PostgreSQL database connection URL

    Returns:
        Configured FastAPI router
    """
    router = APIRouter(tags=["feedback"])
    collector = FeedbackCollector(db_url)

    @router.on_event("startup")
    async def startup():
        """Initialize database connection on startup."""
        await collector.initialize()
        logger.info("Feedback API initialized")

    @router.on_event("shutdown")
    async def shutdown():
        """Close database connection on shutdown."""
        await collector.close()
        logger.info("Feedback API shutdown")

    @router.post(
        "/optimization/{optimization_id}",
        response_model=Dict[str, Any],
        status_code=201,
        summary="Submit optimization feedback",
        description="Submit user feedback for a specific optimization recommendation"
    )
    async def submit_feedback(
        optimization_id: str = Path(..., description="Unique optimization identifier"),
        rating: int = Body(..., ge=1, le=5, description="Star rating (1-5)"),
        comment: Optional[str] = Body(None, description="Optional detailed comment"),
        actual_savings: Optional[float] = Body(None, ge=0, description="Actual savings in kWh"),
        predicted_savings: Optional[float] = Body(None, ge=0, description="Predicted savings in kWh"),
        category: FeedbackCategory = Body(FeedbackCategory.OTHER, description="Feedback category"),
        user_id: str = Body(..., description="User identifier"),
        metadata: Dict[str, Any] = Body(default_factory=dict, description="Additional context")
    ) -> Dict[str, Any]:
        """
        Submit feedback for an optimization recommendation.

        This endpoint collects user satisfaction ratings, comments, and actual results
        to enable continuous improvement of optimization algorithms.
        """
        try:
            # Create feedback object
            feedback = OptimizationFeedback(
                optimization_id=optimization_id,
                rating=rating,
                comment=comment,
                actual_savings=actual_savings,
                predicted_savings=predicted_savings,
                category=category,
                user_id=user_id,
                metadata=metadata
            )

            # Store feedback
            result = await collector.collect_feedback(feedback)

            logger.info(f"Feedback submitted successfully for optimization {optimization_id}")

            return {
                "status": "success",
                "message": "Feedback submitted successfully",
                "data": result
            }

        except ValueError as e:
            logger.warning(f"Invalid feedback submission: {e}")
            raise HTTPException(status_code=400, detail=str(e))

        except RuntimeError as e:
            logger.error(f"Feedback submission failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to submit feedback")

        except Exception as e:
            logger.error(f"Unexpected error in feedback submission: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get(
        "/stats",
        response_model=FeedbackStats,
        summary="Get feedback statistics",
        description="Retrieve aggregated feedback statistics for a time period"
    )
    async def get_statistics(
        days: int = Query(30, ge=1, le=365, description="Number of days to include"),
        category: Optional[FeedbackCategory] = Query(None, description="Filter by category")
    ) -> FeedbackStats:
        """
        Get aggregated feedback statistics.

        Returns metrics including average rating, NPS score, and distribution.
        """
        try:
            stats = await collector.get_stats(days=days, category=category)

            logger.info(f"Statistics retrieved: {days} days, avg_rating={stats.average_rating}")

            return stats

        except RuntimeError as e:
            logger.error(f"Failed to get statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

        except Exception as e:
            logger.error(f"Unexpected error getting statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get(
        "/recent",
        response_model=List[OptimizationFeedback],
        summary="Get recent feedback",
        description="Retrieve recent feedback submissions with optional filtering"
    )
    async def get_recent(
        limit: int = Query(10, ge=1, le=100, description="Maximum number of results"),
        rating: Optional[int] = Query(None, ge=1, le=5, description="Filter by rating")
    ) -> List[OptimizationFeedback]:
        """
        Get recent feedback submissions.

        Returns list of feedback sorted by timestamp (newest first).
        """
        try:
            feedback_list = await collector.get_recent_feedback(
                limit=limit,
                rating_filter=rating
            )

            logger.info(f"Retrieved {len(feedback_list)} recent feedback items")

            return feedback_list

        except RuntimeError as e:
            logger.error(f"Failed to get recent feedback: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve feedback")

        except Exception as e:
            logger.error(f"Unexpected error getting recent feedback: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get(
        "/trends",
        response_model=List[SatisfactionTrend],
        summary="Get satisfaction trends",
        description="Retrieve time-series satisfaction trends with moving averages"
    )
    async def get_trends(
        days: int = Query(90, ge=7, le=365, description="Number of days to include")
    ) -> List[SatisfactionTrend]:
        """
        Get satisfaction trends over time.

        Returns daily satisfaction metrics with 7-day and 30-day moving averages.
        """
        try:
            trends = await collector.get_satisfaction_trends(days=days)

            logger.info(f"Retrieved {len(trends)} trend records")

            return trends

        except RuntimeError as e:
            logger.error(f"Failed to get trends: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve trends")

        except Exception as e:
            logger.error(f"Unexpected error getting trends: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @router.get(
        "/health",
        summary="Health check",
        description="Check feedback system health status"
    )
    async def health_check() -> Dict[str, Any]:
        """
        Health check endpoint.

        Returns system status and basic metrics.
        """
        try:
            # Get basic stats to verify database connectivity
            stats = await collector.get_stats(days=1)

            return {
                "status": "healthy",
                "timestamp": DeterministicClock.utcnow().isoformat(),
                "database": "connected",
                "recent_feedback_count": stats.total_feedback_count
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "timestamp": DeterministicClock.utcnow().isoformat(),
                    "error": str(e)
                }
            )

    return router


# Standalone function for batch feedback submission
async def submit_feedback_batch(
    collector: FeedbackCollector,
    feedback_list: List[OptimizationFeedback]
) -> Dict[str, Any]:
    """
    Submit multiple feedback items in batch.

    Args:
        collector: FeedbackCollector instance
        feedback_list: List of feedback items to submit

    Returns:
        Batch submission results with success/failure counts
    """
    results = {
        "total": len(feedback_list),
        "successful": 0,
        "failed": 0,
        "errors": []
    }

    for feedback in feedback_list:
        try:
            await collector.collect_feedback(feedback)
            results["successful"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({
                "optimization_id": feedback.optimization_id,
                "error": str(e)
            })
            logger.error(f"Failed to submit feedback for {feedback.optimization_id}: {e}")

    logger.info(
        f"Batch feedback submission complete: {results['successful']}/{results['total']} successful"
    )

    return results
