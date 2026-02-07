# -*- coding: utf-8 -*-
"""
Security Training Security Scorer - SEC-010

Calculates employee security posture scores based on multiple factors:
training completion, phishing resistance, MFA enablement, password hygiene,
and security incident history.

Classes:
    - SecurityScorer: Main class for security score calculation

Features:
    - Weighted composite scoring (0-100)
    - Component breakdown for transparency
    - Team and organization leaderboards
    - At-risk user identification
    - Personalized improvement suggestions

Example:
    >>> from greenlang.infrastructure.security_training.security_scorer import (
    ...     SecurityScorer,
    ... )
    >>> scorer = SecurityScorer(completion_tracker)
    >>> score = await scorer.calculate_score(user_id)
    >>> breakdown = await scorer.get_score_breakdown(user_id)
    >>> leaderboard = await scorer.get_leaderboard(team_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.infrastructure.security_training.models import (
    SecurityScore,
)
from greenlang.infrastructure.security_training.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Score Components
# ---------------------------------------------------------------------------


@dataclass
class ScoreComponent:
    """A single component of the security score.

    Attributes:
        name: Component name.
        weight: Weight in overall score (0.0-1.0).
        raw_score: Raw component score (0-100).
        weighted_score: Weighted contribution to total.
        details: Additional details about the score.
    """

    name: str
    weight: float
    raw_score: float
    weighted_score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreBreakdown:
    """Complete breakdown of a user's security score.

    Attributes:
        user_id: User identifier.
        total_score: Overall security score (0-100).
        components: Individual component scores.
        calculated_at: When the score was calculated.
        trend: Score change from previous calculation.
        suggestions: Personalized improvement suggestions.
    """

    user_id: str
    total_score: int
    components: List[ScoreComponent]
    calculated_at: datetime
    trend: str  # "up", "down", "stable"
    suggestions: List[str]


@dataclass
class LeaderboardEntry:
    """Entry in the security score leaderboard.

    Attributes:
        rank: Position in leaderboard.
        user_id: User identifier.
        score: Security score.
        trend: Score trend.
    """

    rank: int
    user_id: str
    score: int
    trend: str


# ---------------------------------------------------------------------------
# Score Component Weights
# ---------------------------------------------------------------------------

SCORE_COMPONENTS = {
    "training_completion": {
        "weight": 0.30,
        "description": "Training course completion rate",
        "max_points": 100,
    },
    "phishing_resistance": {
        "weight": 0.25,
        "description": "Resistance to phishing simulations",
        "max_points": 100,
    },
    "mfa_enabled": {
        "weight": 0.20,
        "description": "Multi-factor authentication usage",
        "max_points": 100,
    },
    "password_age": {
        "weight": 0.10,
        "description": "Password hygiene and age",
        "max_points": 100,
    },
    "security_incidents": {
        "weight": 0.15,
        "description": "Security incident history",
        "max_points": 100,
    },
}


# ---------------------------------------------------------------------------
# Security Scorer Class
# ---------------------------------------------------------------------------


class SecurityScorer:
    """Security posture scorer for employees.

    Calculates composite security scores based on multiple factors and
    provides leaderboards, at-risk identification, and improvement suggestions.

    Attributes:
        _scores: Cache of calculated scores by user_id.
        _user_data: Mock user data for scoring (in prod, from IAM/HR).
        _phishing_data: Mock phishing results (in prod, from PhishingSimulator).
        _incident_data: Mock incident data (in prod, from incident system).

    Example:
        >>> scorer = SecurityScorer()
        >>> score = await scorer.calculate_score("user-1")
        >>> print(f"Security Score: {score.score}")
    """

    def __init__(self) -> None:
        """Initialize the security scorer."""
        self._config = get_config()

        # Caches (in production, backed by database)
        self._scores: Dict[str, SecurityScore] = {}
        self._previous_scores: Dict[str, int] = {}

        # Mock data sources (in production, integrated with other systems)
        self._user_training_data: Dict[str, Dict[str, Any]] = {}
        self._user_phishing_data: Dict[str, Dict[str, Any]] = {}
        self._user_mfa_status: Dict[str, bool] = {}
        self._user_password_data: Dict[str, Dict[str, Any]] = {}
        self._user_incident_data: Dict[str, List[Dict[str, Any]]] = {}
        self._user_teams: Dict[str, str] = {}

        logger.info(
            "SecurityScorer initialized with weights: training=%.0f%%, "
            "phishing=%.0f%%, mfa=%.0f%%, password=%.0f%%, incidents=%.0f%%",
            self._config.score_weight_training * 100,
            self._config.score_weight_phishing * 100,
            self._config.score_weight_mfa * 100,
            self._config.score_weight_password * 100,
            self._config.score_weight_incidents * 100,
        )

    async def calculate_score(self, user_id: str) -> SecurityScore:
        """Calculate security score for a user.

        Computes a weighted composite score from all security factors.

        Args:
            user_id: User identifier.

        Returns:
            SecurityScore with overall score and component breakdown.
        """
        components: Dict[str, float] = {}

        # Calculate each component
        training_score = await self._calculate_training_score(user_id)
        phishing_score = await self._calculate_phishing_score(user_id)
        mfa_score = await self._calculate_mfa_score(user_id)
        password_score = await self._calculate_password_score(user_id)
        incident_score = await self._calculate_incident_score(user_id)

        components["training_completion"] = training_score
        components["phishing_resistance"] = phishing_score
        components["mfa_enabled"] = mfa_score
        components["password_age"] = password_score
        components["security_incidents"] = incident_score

        # Calculate weighted total
        total = (
            training_score * self._config.score_weight_training
            + phishing_score * self._config.score_weight_phishing
            + mfa_score * self._config.score_weight_mfa
            + password_score * self._config.score_weight_password
            + incident_score * self._config.score_weight_incidents
        )

        # Round to integer
        final_score = round(total)

        # Get previous score for trend
        previous = self._previous_scores.get(user_id)

        # Create score object
        score = SecurityScore(
            user_id=user_id,
            score=final_score,
            components=components,
            calculated_at=datetime.now(timezone.utc),
            previous_score=previous,
        )

        # Update caches
        if user_id in self._scores:
            self._previous_scores[user_id] = self._scores[user_id].score
        self._scores[user_id] = score

        logger.info(
            "Calculated security score for %s: %d (training=%d, phishing=%d, "
            "mfa=%d, password=%d, incidents=%d)",
            user_id,
            final_score,
            training_score,
            phishing_score,
            mfa_score,
            password_score,
            incident_score,
        )

        return score

    async def get_score_breakdown(self, user_id: str) -> ScoreBreakdown:
        """Get detailed score breakdown for a user.

        Args:
            user_id: User identifier.

        Returns:
            ScoreBreakdown with all components and suggestions.
        """
        # Get or calculate score
        score = self._scores.get(user_id)
        if score is None:
            score = await self.calculate_score(user_id)

        # Build component details
        components: List[ScoreComponent] = []
        for name, info in SCORE_COMPONENTS.items():
            raw_score = score.components.get(name, 0)
            weight = getattr(self._config, f"score_weight_{name.split('_')[0]}", info["weight"])

            components.append(
                ScoreComponent(
                    name=name,
                    weight=weight,
                    raw_score=raw_score,
                    weighted_score=raw_score * weight,
                    details={"description": info["description"]},
                )
            )

        # Determine trend
        if score.previous_score is None:
            trend = "stable"
        elif score.score > score.previous_score:
            trend = "up"
        elif score.score < score.previous_score:
            trend = "down"
        else:
            trend = "stable"

        # Generate suggestions
        suggestions = await self.get_improvement_suggestions(user_id)

        return ScoreBreakdown(
            user_id=user_id,
            total_score=score.score,
            components=components,
            calculated_at=score.calculated_at,
            trend=trend,
            suggestions=suggestions,
        )

    async def get_leaderboard(
        self,
        team_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[LeaderboardEntry]:
        """Get security score leaderboard.

        Args:
            team_id: Optional team filter.
            limit: Maximum entries to return.

        Returns:
            List of LeaderboardEntry sorted by score descending.
        """
        # Get relevant users
        if team_id:
            user_ids = [
                uid for uid, tid in self._user_teams.items() if tid == team_id
            ]
        else:
            user_ids = list(self._scores.keys())

        # Calculate scores for users without cached scores
        for user_id in user_ids:
            if user_id not in self._scores:
                await self.calculate_score(user_id)

        # Sort by score
        scored_users: List[Tuple[str, int, str]] = []
        for user_id in user_ids:
            score = self._scores.get(user_id)
            if score:
                # Determine trend
                previous = self._previous_scores.get(user_id)
                if previous is None:
                    trend = "stable"
                elif score.score > previous:
                    trend = "up"
                elif score.score < previous:
                    trend = "down"
                else:
                    trend = "stable"

                scored_users.append((user_id, score.score, trend))

        # Sort descending
        scored_users.sort(key=lambda x: x[1], reverse=True)

        # Build leaderboard
        leaderboard: List[LeaderboardEntry] = []
        for rank, (user_id, score, trend) in enumerate(scored_users[:limit], 1):
            leaderboard.append(
                LeaderboardEntry(
                    rank=rank,
                    user_id=user_id,
                    score=score,
                    trend=trend,
                )
            )

        return leaderboard

    async def get_organization_average(self) -> float:
        """Get organization-wide average security score.

        Returns:
            Average score (0.0-100.0).
        """
        if not self._scores:
            return 0.0

        total = sum(s.score for s in self._scores.values())
        return total / len(self._scores)

    async def identify_at_risk_users(
        self,
        threshold: Optional[int] = None,
    ) -> List[Tuple[str, int, List[str]]]:
        """Identify users with scores below threshold.

        Args:
            threshold: Score threshold (default from config).

        Returns:
            List of (user_id, score, weak_areas).
        """
        if threshold is None:
            threshold = self._config.at_risk_score_threshold

        at_risk: List[Tuple[str, int, List[str]]] = []

        for user_id, score in self._scores.items():
            if score.score < threshold:
                # Identify weak areas
                weak_areas = []
                for name, component_score in score.components.items():
                    if component_score < 70:
                        weak_areas.append(name)

                at_risk.append((user_id, score.score, weak_areas))

        # Sort by score ascending (worst first)
        at_risk.sort(key=lambda x: x[1])

        logger.info(
            "Identified %d at-risk users (threshold=%d)",
            len(at_risk),
            threshold,
        )

        return at_risk

    async def get_improvement_suggestions(
        self,
        user_id: str,
    ) -> List[str]:
        """Get personalized improvement suggestions.

        Args:
            user_id: User identifier.

        Returns:
            List of suggestion strings.
        """
        score = self._scores.get(user_id)
        if score is None:
            return ["Complete your security score assessment."]

        suggestions: List[str] = []

        # Check each component and suggest improvements
        if score.components.get("training_completion", 100) < 80:
            suggestions.append(
                "Complete your required security training courses to improve "
                "your training score."
            )

        if score.components.get("phishing_resistance", 100) < 80:
            suggestions.append(
                "Review phishing recognition training. Remember to report "
                "suspicious emails instead of clicking links."
            )

        if score.components.get("mfa_enabled", 100) < 100:
            suggestions.append(
                "Enable multi-factor authentication on your account for "
                "significantly improved security."
            )

        if score.components.get("password_age", 100) < 80:
            suggestions.append(
                "Consider updating your password and using a password manager "
                "for unique, strong passwords."
            )

        if score.components.get("security_incidents", 100) < 80:
            suggestions.append(
                "Your security incident history is affecting your score. "
                "Review security policies and best practices."
            )

        if not suggestions:
            suggestions.append(
                "Great job! Maintain your security practices and help "
                "colleagues improve their security posture."
            )

        return suggestions

    # ---------------------------------------------------------------------------
    # Component Score Calculations
    # ---------------------------------------------------------------------------

    async def _calculate_training_score(self, user_id: str) -> float:
        """Calculate training completion component score."""
        data = self._user_training_data.get(user_id)
        if data is None:
            return 50.0  # Default for no data

        completion_rate = data.get("completion_rate", 0.0)
        return min(100.0, completion_rate * 100)

    async def _calculate_phishing_score(self, user_id: str) -> float:
        """Calculate phishing resistance component score.

        Scoring:
        - Never clicked: 100
        - Clicked but reported: 75
        - Clicked: 50
        - Entered credentials: 25
        """
        data = self._user_phishing_data.get(user_id)
        if data is None:
            return 75.0  # Default for no test data

        if data.get("credentials_entered"):
            return 25.0
        if data.get("clicked") and not data.get("reported"):
            return 50.0
        if data.get("clicked") and data.get("reported"):
            return 75.0
        if data.get("reported"):
            return 100.0

        return 85.0  # No interaction

    async def _calculate_mfa_score(self, user_id: str) -> float:
        """Calculate MFA component score."""
        mfa_enabled = self._user_mfa_status.get(user_id)
        if mfa_enabled is None:
            return 0.0  # Unknown = assume not enabled

        return 100.0 if mfa_enabled else 0.0

    async def _calculate_password_score(self, user_id: str) -> float:
        """Calculate password hygiene component score.

        Based on password age and complexity indicators.
        """
        data = self._user_password_data.get(user_id)
        if data is None:
            return 50.0  # Default

        score = 100.0

        # Deduct for old password
        age_days = data.get("age_days", 0)
        if age_days > 365:
            score -= 30
        elif age_days > 180:
            score -= 15

        # Deduct for weak password indicators
        if data.get("reused"):
            score -= 25
        if data.get("common_pattern"):
            score -= 20

        return max(0.0, score)

    async def _calculate_incident_score(self, user_id: str) -> float:
        """Calculate security incident component score.

        Based on number and severity of past incidents.
        """
        incidents = self._user_incident_data.get(user_id, [])
        if not incidents:
            return 100.0  # No incidents = perfect score

        score = 100.0

        for incident in incidents:
            severity = incident.get("severity", "low")
            if severity == "critical":
                score -= 30
            elif severity == "high":
                score -= 20
            elif severity == "medium":
                score -= 10
            else:
                score -= 5

        return max(0.0, score)

    # ---------------------------------------------------------------------------
    # Data Management (for testing/integration)
    # ---------------------------------------------------------------------------

    def set_training_data(
        self,
        user_id: str,
        completion_rate: float,
    ) -> None:
        """Set training data for a user."""
        self._user_training_data[user_id] = {"completion_rate": completion_rate}

    def set_phishing_data(
        self,
        user_id: str,
        clicked: bool = False,
        reported: bool = False,
        credentials_entered: bool = False,
    ) -> None:
        """Set phishing data for a user."""
        self._user_phishing_data[user_id] = {
            "clicked": clicked,
            "reported": reported,
            "credentials_entered": credentials_entered,
        }

    def set_mfa_status(self, user_id: str, enabled: bool) -> None:
        """Set MFA status for a user."""
        self._user_mfa_status[user_id] = enabled

    def set_password_data(
        self,
        user_id: str,
        age_days: int = 0,
        reused: bool = False,
        common_pattern: bool = False,
    ) -> None:
        """Set password data for a user."""
        self._user_password_data[user_id] = {
            "age_days": age_days,
            "reused": reused,
            "common_pattern": common_pattern,
        }

    def set_incident_data(
        self,
        user_id: str,
        incidents: List[Dict[str, Any]],
    ) -> None:
        """Set incident data for a user."""
        self._user_incident_data[user_id] = incidents

    def set_user_team(self, user_id: str, team_id: str) -> None:
        """Set team membership for a user."""
        self._user_teams[user_id] = team_id


__all__ = [
    "LeaderboardEntry",
    "SCORE_COMPONENTS",
    "ScoreBreakdown",
    "ScoreComponent",
    "SecurityScorer",
]
