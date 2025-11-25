# -*- coding: utf-8 -*-
"""
Rating and Review System

Implements Wilson score-based rating calculation, review moderation,
and helpful vote system for marketplace agents.
"""

import math
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from greenlang.determinism import DeterministicClock
from greenlang.marketplace.models import (
    MarketplaceAgent,
    AgentReview,
    AgentInstall,
    AgentPurchase,
    ReviewStatus,
)

logger = logging.getLogger(__name__)


class ReviewSortBy(str, Enum):
    """Review sorting options"""
    MOST_HELPFUL = "most_helpful"
    NEWEST = "newest"
    HIGHEST_RATING = "highest_rating"
    LOWEST_RATING = "lowest_rating"
    VERIFIED_ONLY = "verified_only"


class FlagReason(str, Enum):
    """Review flag reasons"""
    SPAM = "spam"
    OFFENSIVE = "offensive"
    OFF_TOPIC = "off_topic"
    DUPLICATE = "duplicate"
    FAKE = "fake"
    OTHER = "other"


@dataclass
class RatingDistribution:
    """Rating distribution statistics"""
    total_count: int
    average: float
    wilson_score: float
    ratings: Dict[int, int]  # {1: count, 2: count, ...}
    percentages: Dict[int, float]  # {1: percent, 2: percent, ...}


@dataclass
class ReviewValidation:
    """Review validation result"""
    valid: bool
    errors: List[str]
    warnings: List[str]


def calculate_wilson_score(positive: int, total: int, confidence: float = 0.95) -> float:
    """
    Calculate Wilson score confidence interval lower bound.

    This is a better ranking metric than simple average because it accounts
    for the number of ratings. Agents with few perfect ratings will rank
    lower than agents with many good ratings.

    Args:
        positive: Number of positive ratings (4-5 stars)
        total: Total number of ratings
        confidence: Confidence level (default 95%)

    Returns:
        Wilson score lower bound (0-1)
    """
    if total == 0:
        return 0.0

    # Z-score for confidence level (1.96 for 95%)
    z = 1.96 if confidence == 0.95 else 1.645

    p_hat = positive / total
    denominator = 1 + (z * z) / total

    center = p_hat + (z * z) / (2 * total)
    adjustment = z * math.sqrt((p_hat * (1 - p_hat) + (z * z) / (4 * total)) / total)

    lower_bound = (center - adjustment) / denominator

    return max(0.0, min(1.0, lower_bound))


def calculate_weighted_rating(ratings: Dict[int, int], min_ratings: int = 5) -> float:
    """
    Calculate weighted average rating with Bayesian average.

    Adds virtual ratings to prevent manipulation from few extreme ratings.

    Args:
        ratings: Dictionary of {star_count: number_of_ratings}
        min_ratings: Minimum number of ratings to consider reliable

    Returns:
        Weighted average rating (1-5)
    """
    if not ratings:
        return 0.0

    total_ratings = sum(ratings.values())

    if total_ratings == 0:
        return 0.0

    # Calculate actual average
    weighted_sum = sum(stars * count for stars, count in ratings.items())
    actual_average = weighted_sum / total_ratings

    # If we have enough ratings, use actual average
    if total_ratings >= min_ratings:
        return actual_average

    # Otherwise, blend with prior (assume 3.5 average for new agents)
    prior_average = 3.5
    prior_weight = min_ratings - total_ratings

    bayesian_average = (
        (weighted_sum + prior_average * prior_weight) /
        (total_ratings + prior_weight)
    )

    return bayesian_average


class RatingSystem:
    """
    Rating system manager.

    Handles rating calculations, updates, and statistics.
    """

    def __init__(self, session: Session):
        self.session = session

    def calculate_agent_ratings(self, agent_id: str) -> RatingDistribution:
        """
        Calculate complete rating distribution for an agent.

        Args:
            agent_id: Agent UUID

        Returns:
            RatingDistribution with all statistics
        """
        # Get all approved reviews
        reviews = self.session.query(AgentReview).filter(
            and_(
                AgentReview.agent_id == agent_id,
                AgentReview.status == ReviewStatus.APPROVED.value
            )
        ).all()

        if not reviews:
            return RatingDistribution(
                total_count=0,
                average=0.0,
                wilson_score=0.0,
                ratings={1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                percentages={1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
            )

        # Count ratings
        rating_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for review in reviews:
            rating_counts[review.rating] = rating_counts.get(review.rating, 0) + 1

        total_count = len(reviews)

        # Calculate average
        weighted_average = calculate_weighted_rating(rating_counts)

        # Calculate Wilson score (consider 4-5 stars as "positive")
        positive = rating_counts.get(4, 0) + rating_counts.get(5, 0)
        wilson = calculate_wilson_score(positive, total_count)

        # Calculate percentages
        percentages = {
            stars: (count / total_count * 100) if total_count > 0 else 0.0
            for stars, count in rating_counts.items()
        }

        return RatingDistribution(
            total_count=total_count,
            average=weighted_average,
            wilson_score=wilson,
            ratings=rating_counts,
            percentages=percentages
        )

    def update_agent_ratings(self, agent_id: str) -> None:
        """
        Update agent rating statistics in database.

        Args:
            agent_id: Agent UUID
        """
        agent = self.session.query(MarketplaceAgent).filter(
            MarketplaceAgent.id == agent_id
        ).first()

        if not agent:
            logger.error(f"Agent {agent_id} not found")
            return

        distribution = self.calculate_agent_ratings(agent_id)

        # Update agent
        agent.rating_avg = distribution.average
        agent.rating_count = distribution.total_count
        agent.wilson_score = distribution.wilson_score
        agent.rating_1_count = distribution.ratings.get(1, 0)
        agent.rating_2_count = distribution.ratings.get(2, 0)
        agent.rating_3_count = distribution.ratings.get(3, 0)
        agent.rating_4_count = distribution.ratings.get(4, 0)
        agent.rating_5_count = distribution.ratings.get(5, 0)

        self.session.commit()
        logger.info(f"Updated ratings for agent {agent_id}: avg={distribution.average:.2f}, wilson={distribution.wilson:.2f}")

    def get_trending_agents(self, days: int = 7, limit: int = 10) -> List[MarketplaceAgent]:
        """
        Get trending agents based on recent ratings.

        Args:
            days: Number of days to look back
            limit: Maximum number of agents to return

        Returns:
            List of trending agents
        """
        since = DeterministicClock.utcnow() - timedelta(days=days)

        # Subquery for recent review counts
        recent_reviews = self.session.query(
            AgentReview.agent_id,
            func.count(AgentReview.id).label('recent_count'),
            func.avg(AgentReview.rating).label('recent_avg')
        ).filter(
            and_(
                AgentReview.created_at >= since,
                AgentReview.status == ReviewStatus.APPROVED.value
            )
        ).group_by(AgentReview.agent_id).subquery()

        # Join with agents and order by recent activity
        agents = self.session.query(MarketplaceAgent).join(
            recent_reviews,
            MarketplaceAgent.id == recent_reviews.c.agent_id
        ).filter(
            MarketplaceAgent.status == "published"
        ).order_by(
            recent_reviews.c.recent_count.desc(),
            recent_reviews.c.recent_avg.desc()
        ).limit(limit).all()

        return agents

    def get_top_rated_agents(self, limit: int = 10, min_ratings: int = 5) -> List[MarketplaceAgent]:
        """
        Get top-rated agents using Wilson score.

        Args:
            limit: Maximum number of agents to return
            min_ratings: Minimum number of ratings required

        Returns:
            List of top-rated agents
        """
        agents = self.session.query(MarketplaceAgent).filter(
            and_(
                MarketplaceAgent.status == "published",
                MarketplaceAgent.rating_count >= min_ratings
            )
        ).order_by(
            MarketplaceAgent.wilson_score.desc(),
            MarketplaceAgent.rating_count.desc()
        ).limit(limit).all()

        return agents


class ReviewModerator:
    """
    Review moderation system.

    Handles review validation, spam detection, and moderation queue.
    """

    def __init__(self, session: Session):
        self.session = session

    def validate_review(
        self,
        user_id: str,
        agent_id: str,
        rating: int,
        review_text: Optional[str] = None
    ) -> ReviewValidation:
        """
        Validate a review submission.

        Checks for:
        - Duplicate reviews
        - Rate limiting
        - Required purchase/install
        - Content validation

        Args:
            user_id: User UUID
            agent_id: Agent UUID
            rating: Star rating (1-5)
            review_text: Optional review text

        Returns:
            ReviewValidation result
        """
        errors = []
        warnings = []

        # Check for existing review
        existing = self.session.query(AgentReview).filter(
            and_(
                AgentReview.user_id == user_id,
                AgentReview.agent_id == agent_id
            )
        ).first()

        if existing:
            errors.append("You have already reviewed this agent")

        # Check rating range
        if not (1 <= rating <= 5):
            errors.append("Rating must be between 1 and 5")

        # Check if user has installed/purchased agent
        has_installed = self.session.query(AgentInstall).filter(
            and_(
                AgentInstall.user_id == user_id,
                AgentInstall.agent_id == agent_id
            )
        ).first()

        has_purchased = self.session.query(AgentPurchase).filter(
            and_(
                AgentPurchase.user_id == user_id,
                AgentPurchase.agent_id == agent_id,
                AgentPurchase.status == "completed"
            )
        ).first()

        if not has_installed and not has_purchased:
            warnings.append("You haven't installed this agent. Install it to verify your review.")

        # Check rate limiting (max 10 reviews per day)
        today_start = DeterministicClock.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        today_reviews = self.session.query(func.count(AgentReview.id)).filter(
            and_(
                AgentReview.user_id == user_id,
                AgentReview.created_at >= today_start
            )
        ).scalar()

        if today_reviews >= 10:
            errors.append("Daily review limit reached. Please try again tomorrow.")

        # Validate review text if provided
        if review_text:
            if len(review_text) < 10:
                errors.append("Review text must be at least 10 characters")
            elif len(review_text) > 5000:
                errors.append("Review text must be less than 5000 characters")

            # Simple spam detection
            if self._is_spam(review_text):
                errors.append("Review appears to be spam")

        return ReviewValidation(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )

    def _is_spam(self, text: str) -> bool:
        """
        Simple spam detection.

        Args:
            text: Review text

        Returns:
            True if text appears to be spam
        """
        spam_indicators = [
            "http://",
            "https://",
            "www.",
            "click here",
            "buy now",
            "limited time",
            "act now",
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in spam_indicators)

    def submit_review(
        self,
        user_id: str,
        agent_id: str,
        rating: int,
        title: Optional[str] = None,
        review_text: Optional[str] = None
    ) -> Tuple[bool, Optional[AgentReview], List[str]]:
        """
        Submit a new review.

        Args:
            user_id: User UUID
            agent_id: Agent UUID
            rating: Star rating (1-5)
            title: Optional review title
            review_text: Optional review text

        Returns:
            Tuple of (success, review, errors)
        """
        # Validate review
        validation = self.validate_review(user_id, agent_id, rating, review_text)

        if not validation.valid:
            return False, None, validation.errors

        # Check verification status
        has_purchased = self.session.query(AgentPurchase).filter(
            and_(
                AgentPurchase.user_id == user_id,
                AgentPurchase.agent_id == agent_id,
                AgentPurchase.status == "completed"
            )
        ).first()

        has_installed = self.session.query(AgentInstall).filter(
            and_(
                AgentInstall.user_id == user_id,
                AgentInstall.agent_id == agent_id
            )
        ).first()

        # Create review
        review = AgentReview(
            agent_id=agent_id,
            user_id=user_id,
            rating=rating,
            title=title,
            review_text=review_text,
            verified_purchase=bool(has_purchased),
            verified_install=bool(has_installed),
            status=ReviewStatus.APPROVED.value  # Auto-approve for now
        )

        self.session.add(review)
        self.session.commit()

        # Update agent ratings
        rating_system = RatingSystem(self.session)
        rating_system.update_agent_ratings(agent_id)

        logger.info(f"Review submitted by {user_id} for agent {agent_id}: {rating} stars")

        return True, review, []

    def flag_review(
        self,
        review_id: str,
        user_id: str,
        reason: FlagReason,
        details: Optional[str] = None
    ) -> bool:
        """
        Flag a review as inappropriate.

        Args:
            review_id: Review UUID
            user_id: User who flagged
            reason: Flag reason
            details: Optional details

        Returns:
            True if flagged successfully
        """
        review = self.session.query(AgentReview).filter(
            AgentReview.id == review_id
        ).first()

        if not review:
            logger.error(f"Review {review_id} not found")
            return False

        # Increment flag count
        review.flagged_count = (review.flagged_count or 0) + 1

        # Store flag reason
        if not review.flag_reasons:
            review.flag_reasons = []
        review.flag_reasons.append({
            "user_id": str(user_id),
            "reason": reason.value,
            "details": details,
            "timestamp": DeterministicClock.utcnow().isoformat()
        })

        # Auto-hide if too many flags
        if review.flagged_count >= 5:
            review.status = ReviewStatus.FLAGGED.value
            logger.warning(f"Review {review_id} auto-flagged after {review.flagged_count} reports")

        self.session.commit()
        return True

    def vote_helpful(self, review_id: str, user_id: str, helpful: bool = True) -> bool:
        """
        Vote if a review is helpful.

        Args:
            review_id: Review UUID
            user_id: User who voted
            helpful: True for helpful, False for not helpful

        Returns:
            True if voted successfully
        """
        review = self.session.query(AgentReview).filter(
            AgentReview.id == review_id
        ).first()

        if not review:
            return False

        # In a real system, track individual votes to prevent duplicates
        # For now, just increment counters
        if helpful:
            review.helpful_count = (review.helpful_count or 0) + 1
        else:
            review.not_helpful_count = (review.not_helpful_count or 0) + 1

        self.session.commit()
        return True

    def get_reviews(
        self,
        agent_id: str,
        sort_by: ReviewSortBy = ReviewSortBy.MOST_HELPFUL,
        limit: int = 20,
        offset: int = 0,
        verified_only: bool = False
    ) -> List[AgentReview]:
        """
        Get reviews for an agent with sorting.

        Args:
            agent_id: Agent UUID
            sort_by: Sort order
            limit: Maximum number of reviews
            offset: Pagination offset
            verified_only: Only show verified reviews

        Returns:
            List of reviews
        """
        query = self.session.query(AgentReview).filter(
            and_(
                AgentReview.agent_id == agent_id,
                AgentReview.status == ReviewStatus.APPROVED.value
            )
        )

        if verified_only:
            query = query.filter(
                or_(
                    AgentReview.verified_purchase == True,
                    AgentReview.verified_install == True
                )
            )

        # Apply sorting
        if sort_by == ReviewSortBy.MOST_HELPFUL:
            query = query.order_by(AgentReview.helpful_count.desc())
        elif sort_by == ReviewSortBy.NEWEST:
            query = query.order_by(AgentReview.created_at.desc())
        elif sort_by == ReviewSortBy.HIGHEST_RATING:
            query = query.order_by(AgentReview.rating.desc(), AgentReview.created_at.desc())
        elif sort_by == ReviewSortBy.LOWEST_RATING:
            query = query.order_by(AgentReview.rating.asc(), AgentReview.created_at.desc())

        reviews = query.limit(limit).offset(offset).all()
        return reviews

    def respond_to_review(
        self,
        review_id: str,
        author_id: str,
        response: str
    ) -> bool:
        """
        Allow agent author to respond to a review.

        Args:
            review_id: Review UUID
            author_id: Agent author UUID
            response: Response text

        Returns:
            True if response added successfully
        """
        review = self.session.query(AgentReview).join(
            MarketplaceAgent,
            AgentReview.agent_id == MarketplaceAgent.id
        ).filter(
            and_(
                AgentReview.id == review_id,
                MarketplaceAgent.author_id == author_id
            )
        ).first()

        if not review:
            logger.error(f"Review {review_id} not found or author {author_id} not authorized")
            return False

        review.author_response = response
        review.author_response_at = DeterministicClock.utcnow()

        self.session.commit()
        logger.info(f"Author response added to review {review_id}")
        return True


class ReviewAnalytics:
    """
    Review analytics and insights.

    Provides aggregate statistics and insights about reviews.
    """

    def __init__(self, session: Session):
        self.session = session

    def get_review_trends(
        self,
        agent_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get review trends over time.

        Args:
            agent_id: Agent UUID
            days: Number of days to analyze

        Returns:
            Dictionary with trend data
        """
        since = DeterministicClock.utcnow() - timedelta(days=days)

        reviews = self.session.query(AgentReview).filter(
            and_(
                AgentReview.agent_id == agent_id,
                AgentReview.created_at >= since,
                AgentReview.status == ReviewStatus.APPROVED.value
            )
        ).all()

        if not reviews:
            return {
                "total_reviews": 0,
                "average_rating": 0.0,
                "trend": "neutral"
            }

        # Calculate weekly averages
        weekly_averages = []
        for week in range(days // 7):
            week_start = since + timedelta(weeks=week)
            week_end = week_start + timedelta(weeks=1)

            week_reviews = [
                r for r in reviews
                if week_start <= r.created_at < week_end
            ]

            if week_reviews:
                avg = sum(r.rating for r in week_reviews) / len(week_reviews)
                weekly_averages.append(avg)

        # Determine trend
        trend = "neutral"
        if len(weekly_averages) >= 2:
            if weekly_averages[-1] > weekly_averages[0]:
                trend = "improving"
            elif weekly_averages[-1] < weekly_averages[0]:
                trend = "declining"

        return {
            "total_reviews": len(reviews),
            "average_rating": sum(r.rating for r in reviews) / len(reviews),
            "weekly_averages": weekly_averages,
            "trend": trend
        }
