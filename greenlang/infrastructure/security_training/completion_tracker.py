# -*- coding: utf-8 -*-
"""
Security Training Completion Tracker - SEC-010

Tracks training completion status, compliance rates, and sends reminders
for overdue training. Provides individual user progress, team compliance,
and organization-wide statistics.

Classes:
    - CompletionTracker: Main class for tracking training completion

Features:
    - Course start and completion tracking
    - User progress aggregation
    - Team and organization compliance rates
    - Overdue training reminders
    - Compliance reporting

Example:
    >>> from greenlang.infrastructure.security_training.completion_tracker import (
    ...     CompletionTracker,
    ... )
    >>> tracker = CompletionTracker(library, mapper)
    >>> await tracker.start_course(user_id, course_id)
    >>> progress = await tracker.get_user_progress(user_id)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.infrastructure.security_training.models import (
    Certificate,
    TrainingCompletion,
    UserProgress,
)
from greenlang.infrastructure.security_training.content_library import ContentLibrary
from greenlang.infrastructure.security_training.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class TeamCompliance:
    """Team training compliance statistics.

    Attributes:
        team_id: Team identifier.
        team_name: Team display name.
        total_members: Total team members.
        compliant_members: Members with complete training.
        overdue_members: Members with overdue training.
        compliance_rate: Compliance percentage (0.0-1.0).
        average_score: Average assessment score.
    """

    team_id: str
    team_name: str
    total_members: int
    compliant_members: int
    overdue_members: int
    compliance_rate: float
    average_score: float


@dataclass
class OrganizationCompliance:
    """Organization-wide compliance statistics.

    Attributes:
        total_employees: Total employees.
        compliant_employees: Employees with complete training.
        overdue_employees: Employees with overdue training.
        compliance_rate: Overall compliance rate.
        average_score: Average assessment score.
        by_course: Compliance broken down by course.
        by_team: Compliance broken down by team.
    """

    total_employees: int
    compliant_employees: int
    overdue_employees: int
    compliance_rate: float
    average_score: float
    by_course: Dict[str, float]
    by_team: Dict[str, TeamCompliance]


@dataclass
class ReminderResult:
    """Result of sending training reminders.

    Attributes:
        total_reminders: Total reminders sent.
        email_count: Reminders sent via email.
        slack_count: Reminders sent via Slack.
        user_ids: List of users who received reminders.
    """

    total_reminders: int
    email_count: int
    slack_count: int
    user_ids: List[str]


# ---------------------------------------------------------------------------
# Completion Tracker Class
# ---------------------------------------------------------------------------


class CompletionTracker:
    """Training completion and compliance tracker.

    Tracks training progress for users, teams, and the organization.
    Handles course start/completion recording, progress queries, and
    reminder sending.

    Attributes:
        library: ContentLibrary for course information.
        _completions: Cache of completions by user_id.
        _certificates: Cache of certificates by user_id.

    Example:
        >>> tracker = CompletionTracker(library)
        >>> await tracker.start_course("user-1", "owasp_top_10")
        >>> await tracker.record_completion("user-1", "owasp_top_10", 85, True)
        >>> progress = await tracker.get_user_progress("user-1")
    """

    def __init__(
        self,
        library: ContentLibrary,
        completions: Optional[Dict[str, List[TrainingCompletion]]] = None,
        certificates: Optional[Dict[str, List[Certificate]]] = None,
    ) -> None:
        """Initialize the completion tracker.

        Args:
            library: ContentLibrary for course access.
            completions: Optional pre-loaded completions (user_id -> list).
            certificates: Optional pre-loaded certificates (user_id -> list).
        """
        self.library = library
        self._config = get_config()

        # Caches (in production, backed by database)
        self._completions: Dict[str, List[TrainingCompletion]] = completions or {}
        self._certificates: Dict[str, List[Certificate]] = certificates or {}

        # Team memberships (in production, from IAM/HR system)
        self._team_members: Dict[str, List[str]] = {}
        self._user_teams: Dict[str, str] = {}

        logger.info("CompletionTracker initialized")

    async def start_course(
        self,
        user_id: str,
        course_id: str,
    ) -> TrainingCompletion:
        """Record that a user has started a course.

        Args:
            user_id: User identifier.
            course_id: Course identifier.

        Returns:
            Created TrainingCompletion record.
        """
        # Check if already started
        existing = await self._get_latest_completion(user_id, course_id)
        if existing and existing.completed_at is None:
            logger.info(
                "User %s already in progress on course %s",
                user_id,
                course_id,
            )
            return existing

        completion = TrainingCompletion(
            user_id=user_id,
            course_id=course_id,
            started_at=datetime.now(timezone.utc),
        )

        if user_id not in self._completions:
            self._completions[user_id] = []
        self._completions[user_id].append(completion)

        logger.info("User %s started course %s", user_id, course_id)
        return completion

    async def record_completion(
        self,
        user_id: str,
        course_id: str,
        score: int,
        passed: bool,
        certificate: Optional[Certificate] = None,
    ) -> TrainingCompletion:
        """Record that a user has completed a course.

        Args:
            user_id: User identifier.
            course_id: Course identifier.
            score: Assessment score.
            passed: Whether the user passed.
            certificate: Optional certificate if issued.

        Returns:
            Updated TrainingCompletion record.
        """
        completion = await self._get_latest_completion(user_id, course_id)

        if completion is None:
            # Create new completion if not started
            completion = await self.start_course(user_id, course_id)

        completion.completed_at = datetime.now(timezone.utc)
        completion.score = score
        completion.passed = passed
        completion.certificate_id = certificate.id if certificate else None
        completion.attempts += 1

        # Store certificate
        if certificate:
            if user_id not in self._certificates:
                self._certificates[user_id] = []
            self._certificates[user_id].append(certificate)

        logger.info(
            "User %s completed course %s: score=%d, passed=%s",
            user_id,
            course_id,
            score,
            passed,
        )

        return completion

    async def get_user_progress(
        self,
        user_id: str,
        roles: Optional[List[str]] = None,
    ) -> UserProgress:
        """Get aggregated training progress for a user.

        Args:
            user_id: User identifier.
            roles: Optional user roles for curriculum determination.

        Returns:
            UserProgress with completion statistics.
        """
        completions = self._completions.get(user_id, [])
        certificates = self._certificates.get(user_id, [])

        # Get required courses (default to all-employee courses if no roles)
        from greenlang.infrastructure.security_training.curriculum_mapper import (
            ALL_EMPLOYEE_COURSES,
            ROLE_CURRICULA,
        )

        required_courses: set[str] = set(ALL_EMPLOYEE_COURSES)
        if roles:
            for role in roles:
                if role.lower() in ROLE_CURRICULA:
                    required_courses.update(ROLE_CURRICULA[role.lower()])

        # Calculate statistics
        completed_courses: set[str] = set()
        in_progress_courses: set[str] = set()
        overdue_courses: set[str] = set()
        expiring_soon: List[str] = []
        scores: List[int] = []

        # Build maps of latest completions and valid certificates
        latest_completions: Dict[str, TrainingCompletion] = {}
        valid_certificates: Dict[str, Certificate] = {}

        for c in completions:
            if c.course_id not in latest_completions:
                latest_completions[c.course_id] = c
            elif c.completed_at and (
                not latest_completions[c.course_id].completed_at
                or c.completed_at > latest_completions[c.course_id].completed_at
            ):
                latest_completions[c.course_id] = c

        now = datetime.now(timezone.utc)
        for cert in certificates:
            if cert.expires_at > now:
                if (
                    cert.course_id not in valid_certificates
                    or cert.issued_at > valid_certificates[cert.course_id].issued_at
                ):
                    valid_certificates[cert.course_id] = cert

        # Analyze each required course
        for course_id in required_courses:
            completion = latest_completions.get(course_id)
            certificate = valid_certificates.get(course_id)

            if completion and completion.passed:
                # Check if still valid
                if certificate and certificate.expires_at > now:
                    completed_courses.add(course_id)
                    if completion.score:
                        scores.append(completion.score)

                    # Check if expiring soon
                    days_until_expiry = (certificate.expires_at - now).days
                    if days_until_expiry <= 30:
                        expiring_soon.append(course_id)
                else:
                    # Expired
                    overdue_courses.add(course_id)
            elif completion and completion.completed_at is None:
                # In progress
                in_progress_courses.add(course_id)
            else:
                # Not started = overdue (for mandatory courses)
                overdue_courses.add(course_id)

        total_required = len(required_courses)
        total_completed = len(completed_courses)
        completion_rate = total_completed / total_required if total_required > 0 else 0.0
        average_score = sum(scores) / len(scores) if scores else None

        return UserProgress(
            user_id=user_id,
            total_required=total_required,
            total_completed=total_completed,
            total_in_progress=len(in_progress_courses),
            total_overdue=len(overdue_courses),
            completion_rate=completion_rate,
            average_score=average_score,
            certificates=[c.id for c in valid_certificates.values()],
            expiring_soon=expiring_soon,
            security_score=None,  # Calculated by SecurityScorer
        )

    async def get_team_compliance(
        self,
        team_id: str,
    ) -> TeamCompliance:
        """Get compliance statistics for a team.

        Args:
            team_id: Team identifier.

        Returns:
            TeamCompliance statistics.
        """
        members = self._team_members.get(team_id, [])

        if not members:
            return TeamCompliance(
                team_id=team_id,
                team_name=f"Team {team_id}",
                total_members=0,
                compliant_members=0,
                overdue_members=0,
                compliance_rate=0.0,
                average_score=0.0,
            )

        compliant = 0
        overdue = 0
        scores: List[float] = []

        for user_id in members:
            progress = await self.get_user_progress(user_id)

            if progress.completion_rate >= self._config.compliance_threshold:
                compliant += 1
            if progress.total_overdue > 0:
                overdue += 1
            if progress.average_score:
                scores.append(progress.average_score)

        return TeamCompliance(
            team_id=team_id,
            team_name=f"Team {team_id}",
            total_members=len(members),
            compliant_members=compliant,
            overdue_members=overdue,
            compliance_rate=compliant / len(members) if members else 0.0,
            average_score=sum(scores) / len(scores) if scores else 0.0,
        )

    async def get_organization_compliance(self) -> OrganizationCompliance:
        """Get organization-wide compliance statistics.

        Returns:
            OrganizationCompliance with overall and breakdown statistics.
        """
        all_users = set(self._completions.keys())

        if not all_users:
            return OrganizationCompliance(
                total_employees=0,
                compliant_employees=0,
                overdue_employees=0,
                compliance_rate=0.0,
                average_score=0.0,
                by_course={},
                by_team={},
            )

        compliant = 0
        overdue = 0
        scores: List[float] = []
        course_completions: Dict[str, int] = {}

        for user_id in all_users:
            progress = await self.get_user_progress(user_id)

            if progress.completion_rate >= self._config.compliance_threshold:
                compliant += 1
            if progress.total_overdue > 0:
                overdue += 1
            if progress.average_score:
                scores.append(progress.average_score)

        # Calculate per-course compliance
        all_courses = await self.library.list_courses()
        for course in all_courses:
            completed_count = 0
            for user_id in all_users:
                completions = self._completions.get(user_id, [])
                for c in completions:
                    if c.course_id == course.id and c.passed:
                        completed_count += 1
                        break
            course_completions[course.id] = completed_count

        by_course = {
            course_id: count / len(all_users) if all_users else 0.0
            for course_id, count in course_completions.items()
        }

        # Calculate per-team compliance
        by_team = {}
        for team_id in self._team_members.keys():
            by_team[team_id] = await self.get_team_compliance(team_id)

        return OrganizationCompliance(
            total_employees=len(all_users),
            compliant_employees=compliant,
            overdue_employees=overdue,
            compliance_rate=compliant / len(all_users) if all_users else 0.0,
            average_score=sum(scores) / len(scores) if scores else 0.0,
            by_course=by_course,
            by_team=by_team,
        )

    async def send_reminders(self) -> ReminderResult:
        """Send reminders for overdue and upcoming training.

        Sends reminders based on configuration settings for days
        before due date.

        Returns:
            ReminderResult with counts of reminders sent.
        """
        all_users = set(self._completions.keys())
        reminded_users: List[str] = []
        email_count = 0
        slack_count = 0

        for user_id in all_users:
            progress = await self.get_user_progress(user_id)

            # Check for overdue training
            if progress.total_overdue > 0:
                if self._config.reminder_email_enabled:
                    await self._send_email_reminder(user_id, "overdue")
                    email_count += 1
                if self._config.reminder_slack_enabled:
                    await self._send_slack_reminder(user_id, "overdue")
                    slack_count += 1
                reminded_users.append(user_id)

            # Check for expiring training
            elif progress.expiring_soon:
                if self._config.reminder_email_enabled:
                    await self._send_email_reminder(user_id, "expiring")
                    email_count += 1
                if self._config.reminder_slack_enabled:
                    await self._send_slack_reminder(user_id, "expiring")
                    slack_count += 1
                reminded_users.append(user_id)

        logger.info(
            "Sent %d training reminders (email=%d, slack=%d)",
            len(reminded_users),
            email_count,
            slack_count,
        )

        return ReminderResult(
            total_reminders=len(reminded_users),
            email_count=email_count,
            slack_count=slack_count,
            user_ids=reminded_users,
        )

    async def get_compliance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Generate compliance report for a date range.

        Args:
            start_date: Report start date (default: 30 days ago).
            end_date: Report end date (default: now).

        Returns:
            Report data dictionary.
        """
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        org_compliance = await self.get_organization_compliance()

        # Count completions in date range
        completions_in_range = 0
        for user_id, completions in self._completions.items():
            for c in completions:
                if c.completed_at and start_date <= c.completed_at <= end_date:
                    completions_in_range += 1

        return {
            "report_id": f"COMPLIANCE-{start_date.strftime('%Y%m%d')}",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_employees": org_compliance.total_employees,
                "compliant_employees": org_compliance.compliant_employees,
                "compliance_rate_pct": round(org_compliance.compliance_rate * 100, 1),
                "target_rate_pct": self._config.compliance_threshold * 100,
                "average_score": round(org_compliance.average_score, 1),
            },
            "completions_in_period": completions_in_range,
            "by_course": {
                course_id: round(rate * 100, 1)
                for course_id, rate in org_compliance.by_course.items()
            },
            "by_team": {
                team_id: {
                    "compliance_rate_pct": round(tc.compliance_rate * 100, 1),
                    "total_members": tc.total_members,
                    "compliant_members": tc.compliant_members,
                }
                for team_id, tc in org_compliance.by_team.items()
            },
            "status": (
                "COMPLIANT"
                if org_compliance.compliance_rate >= self._config.compliance_threshold
                else "NON_COMPLIANT"
            ),
        }

    async def get_overdue_users(self) -> List[Tuple[str, List[str]]]:
        """Get list of users with overdue training.

        Returns:
            List of (user_id, list of overdue course IDs).
        """
        overdue_users: List[Tuple[str, List[str]]] = []

        for user_id in self._completions.keys():
            progress = await self.get_user_progress(user_id)
            if progress.total_overdue > 0:
                # Get specific overdue courses
                overdue_courses = await self._get_overdue_courses(user_id)
                if overdue_courses:
                    overdue_users.append((user_id, overdue_courses))

        return overdue_users

    async def _get_latest_completion(
        self,
        user_id: str,
        course_id: str,
    ) -> Optional[TrainingCompletion]:
        """Get the latest completion record for a user-course pair."""
        completions = self._completions.get(user_id, [])
        latest = None

        for c in completions:
            if c.course_id == course_id:
                if latest is None:
                    latest = c
                elif c.started_at > latest.started_at:
                    latest = c

        return latest

    async def _get_overdue_courses(self, user_id: str) -> List[str]:
        """Get list of overdue course IDs for a user."""
        from greenlang.infrastructure.security_training.curriculum_mapper import (
            ALL_EMPLOYEE_COURSES,
        )

        overdue: List[str] = []
        completions = self._completions.get(user_id, [])
        completed_courses = {c.course_id for c in completions if c.passed}

        for course_id in ALL_EMPLOYEE_COURSES:
            if course_id not in completed_courses:
                overdue.append(course_id)

        return overdue

    async def _send_email_reminder(
        self,
        user_id: str,
        reminder_type: str,
    ) -> None:
        """Send email reminder (stub - integrate with SES)."""
        logger.info(
            "Email reminder sent to %s: type=%s",
            user_id,
            reminder_type,
        )

    async def _send_slack_reminder(
        self,
        user_id: str,
        reminder_type: str,
    ) -> None:
        """Send Slack reminder (stub - integrate with Slack API)."""
        logger.info(
            "Slack reminder sent to %s: type=%s",
            user_id,
            reminder_type,
        )

    def set_team_members(
        self,
        team_id: str,
        member_ids: List[str],
    ) -> None:
        """Set team membership (for testing/caching).

        Args:
            team_id: Team identifier.
            member_ids: List of user IDs in the team.
        """
        self._team_members[team_id] = member_ids
        for user_id in member_ids:
            self._user_teams[user_id] = team_id


__all__ = [
    "CompletionTracker",
    "OrganizationCompliance",
    "ReminderResult",
    "TeamCompliance",
]
