# -*- coding: utf-8 -*-
"""
Security Training Curriculum Mapper - SEC-010

Maps training curricula to user roles and tracks training requirements.
Determines which courses are required, recommended, and overdue for each user
based on their roles, completed training, and certificate expiration.

Classes:
    - CurriculumMapper: Maps roles to training paths and tracks requirements

Example:
    >>> from greenlang.infrastructure.security_training.curriculum_mapper import (
    ...     CurriculumMapper,
    ... )
    >>> from greenlang.infrastructure.security_training.content_library import (
    ...     ContentLibrary,
    ... )
    >>> library = ContentLibrary()
    >>> mapper = CurriculumMapper(library)
    >>> curriculum = await mapper.get_curriculum(user)
    >>> required = await mapper.get_required_training(user)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Protocol

from greenlang.infrastructure.security_training.models import (
    Certificate,
    Course,
    TrainingCompletion,
)
from greenlang.infrastructure.security_training.content_library import ContentLibrary
from greenlang.infrastructure.security_training.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User Protocol
# ---------------------------------------------------------------------------


class UserProtocol(Protocol):
    """Protocol for user objects passed to curriculum mapper.

    Users must have an id and roles property for curriculum mapping.
    """

    @property
    def id(self) -> str:
        """User identifier."""
        ...

    @property
    def roles(self) -> List[str]:
        """User's roles for curriculum mapping."""
        ...


# ---------------------------------------------------------------------------
# Role Curricula Definitions
# ---------------------------------------------------------------------------


ROLE_CURRICULA: Dict[str, List[str]] = {
    "developer": [
        "secure_coding_fundamentals",
        "owasp_top_10",
        "secure_code_review",
        "dependency_security",
    ],
    "devops": [
        "infrastructure_security",
        "secrets_management",
        "container_security",
        "incident_response",
    ],
    "sre": [
        "infrastructure_security",
        "secrets_management",
        "container_security",
        "incident_response",
    ],
    "security": [
        "secure_coding_fundamentals",
        "owasp_top_10",
        "infrastructure_security",
        "secrets_management",
        "incident_response",
    ],
    "data_engineer": [
        "data_classification",
        "secure_coding_fundamentals",
    ],
    "manager": [
        # Managers get base training only
    ],
}

# Courses required for ALL employees regardless of role
ALL_EMPLOYEE_COURSES = [
    "security_awareness",
    "phishing_recognition",
    "password_hygiene",
    "data_classification",
]


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class CurriculumItem:
    """A single item in a user's training curriculum.

    Attributes:
        course: The course definition.
        is_required: Whether this course is mandatory.
        is_completed: Whether the user has completed this course.
        is_expired: Whether the completion has expired.
        is_overdue: Whether the course is past due date.
        completion: The user's completion record if exists.
        certificate: The certificate if issued.
        due_date: When the training is due (if applicable).
    """

    course: Course
    is_required: bool = True
    is_completed: bool = False
    is_expired: bool = False
    is_overdue: bool = False
    completion: Optional[TrainingCompletion] = None
    certificate: Optional[Certificate] = None
    due_date: Optional[datetime] = None


@dataclass
class UserCurriculum:
    """A user's complete training curriculum.

    Attributes:
        user_id: User identifier.
        items: List of curriculum items.
        total_required: Total required courses.
        total_completed: Completed courses.
        total_overdue: Overdue courses.
        completion_rate: Completion percentage (0.0-1.0).
    """

    user_id: str
    items: List[CurriculumItem]
    total_required: int = 0
    total_completed: int = 0
    total_overdue: int = 0
    completion_rate: float = 0.0


# ---------------------------------------------------------------------------
# Curriculum Mapper Class
# ---------------------------------------------------------------------------


class CurriculumMapper:
    """Maps training curricula to user roles.

    Determines which courses are required for each user based on their roles,
    and tracks completion status against those requirements.

    Attributes:
        library: ContentLibrary for course access.
        _completions: Cache of user completions (in production, from DB).
        _certificates: Cache of certificates (in production, from DB).

    Example:
        >>> library = ContentLibrary()
        >>> mapper = CurriculumMapper(library)
        >>> required = await mapper.get_required_training(user)
    """

    def __init__(
        self,
        library: ContentLibrary,
        completions: Optional[Dict[str, List[TrainingCompletion]]] = None,
        certificates: Optional[Dict[str, List[Certificate]]] = None,
    ) -> None:
        """Initialize the curriculum mapper.

        Args:
            library: ContentLibrary instance for course access.
            completions: Optional pre-loaded user completions (user_id -> completions).
            certificates: Optional pre-loaded certificates (user_id -> certificates).
        """
        self.library = library
        self._completions: Dict[str, List[TrainingCompletion]] = completions or {}
        self._certificates: Dict[str, List[Certificate]] = certificates or {}
        self._config = get_config()

        logger.info("CurriculumMapper initialized")

    def _get_role_courses(self, roles: List[str]) -> List[str]:
        """Get all course IDs required for given roles.

        Args:
            roles: List of user roles.

        Returns:
            List of unique course IDs for all roles.
        """
        course_ids: set[str] = set()

        # All employees get base courses
        course_ids.update(ALL_EMPLOYEE_COURSES)

        # Add role-specific courses
        for role in roles:
            role_lower = role.lower()
            if role_lower in ROLE_CURRICULA:
                course_ids.update(ROLE_CURRICULA[role_lower])

        return list(course_ids)

    def _get_user_completions(self, user_id: str) -> Dict[str, TrainingCompletion]:
        """Get user's completions indexed by course_id.

        Args:
            user_id: User identifier.

        Returns:
            Dict mapping course_id to most recent completion.
        """
        completions = self._completions.get(user_id, [])
        result: Dict[str, TrainingCompletion] = {}

        for completion in completions:
            # Keep the most recent completion for each course
            if completion.course_id not in result:
                result[completion.course_id] = completion
            elif (
                completion.completed_at
                and (
                    not result[completion.course_id].completed_at
                    or completion.completed_at > result[completion.course_id].completed_at
                )
            ):
                result[completion.course_id] = completion

        return result

    def _get_user_certificates(self, user_id: str) -> Dict[str, Certificate]:
        """Get user's valid certificates indexed by course_id.

        Args:
            user_id: User identifier.

        Returns:
            Dict mapping course_id to certificate (most recent valid).
        """
        certificates = self._certificates.get(user_id, [])
        now = datetime.now(timezone.utc)
        result: Dict[str, Certificate] = {}

        for cert in certificates:
            if cert.expires_at > now:  # Only valid certificates
                if cert.course_id not in result or cert.issued_at > result[cert.course_id].issued_at:
                    result[cert.course_id] = cert

        return result

    def _is_completion_expired(
        self,
        completion: Optional[TrainingCompletion],
        certificate: Optional[Certificate],
    ) -> bool:
        """Check if a completion/certificate has expired.

        Args:
            completion: User's completion record.
            certificate: User's certificate if issued.

        Returns:
            True if expired, False otherwise.
        """
        if certificate is None:
            # No certificate = consider expired if completion is old
            if completion is None or completion.completed_at is None:
                return True

            # Check if completion is older than training cycle
            age = datetime.now(timezone.utc) - completion.completed_at
            return age.days > self._config.training_cycle_days

        # Check certificate expiration
        return certificate.expires_at < datetime.now(timezone.utc)

    async def get_curriculum(self, user: UserProtocol) -> UserCurriculum:
        """Get complete curriculum for a user.

        Determines all required and optional courses, their completion status,
        and calculates overall compliance.

        Args:
            user: User object with id and roles.

        Returns:
            UserCurriculum with all curriculum items and statistics.
        """
        course_ids = self._get_role_courses(user.roles)
        completions = self._get_user_completions(user.id)
        certificates = self._get_user_certificates(user.id)

        items: List[CurriculumItem] = []
        total_required = 0
        total_completed = 0
        total_overdue = 0

        for course_id in course_ids:
            course = await self.library.get_course(course_id)
            if course is None:
                continue

            completion = completions.get(course_id)
            certificate = certificates.get(course_id)

            is_completed = (
                completion is not None
                and completion.passed is True
                and not self._is_completion_expired(completion, certificate)
            )
            is_expired = (
                completion is not None
                and completion.passed is True
                and self._is_completion_expired(completion, certificate)
            )

            # Calculate due date (certificate expiry or training cycle end)
            due_date = None
            if certificate:
                due_date = certificate.expires_at
            elif is_expired and completion and completion.completed_at:
                due_date = completion.completed_at + timedelta(
                    days=self._config.training_cycle_days
                )

            is_overdue = due_date is not None and due_date < datetime.now(timezone.utc)

            item = CurriculumItem(
                course=course,
                is_required=course.is_mandatory,
                is_completed=is_completed,
                is_expired=is_expired,
                is_overdue=is_overdue,
                completion=completion,
                certificate=certificate,
                due_date=due_date,
            )
            items.append(item)

            if course.is_mandatory:
                total_required += 1
                if is_completed:
                    total_completed += 1
                if is_overdue or (not is_completed and is_expired):
                    total_overdue += 1

        completion_rate = total_completed / total_required if total_required > 0 else 0.0

        return UserCurriculum(
            user_id=user.id,
            items=items,
            total_required=total_required,
            total_completed=total_completed,
            total_overdue=total_overdue,
            completion_rate=completion_rate,
        )

    async def get_required_training(self, user: UserProtocol) -> List[Course]:
        """Get mandatory courses not yet completed or expired.

        Args:
            user: User object with id and roles.

        Returns:
            List of Course objects that need to be completed.
        """
        curriculum = await self.get_curriculum(user)

        required: List[Course] = []
        for item in curriculum.items:
            if item.is_required and (not item.is_completed or item.is_expired):
                required.append(item.course)

        return required

    async def get_recommended_training(self, user: UserProtocol) -> List[Course]:
        """Get recommended courses based on role and gaps.

        Suggests courses beyond required training based on:
        - Courses related to user's role but not mandatory
        - Courses that would improve skills in weak areas

        Args:
            user: User object with id and roles.

        Returns:
            List of recommended Course objects.
        """
        curriculum = await self.get_curriculum(user)
        completed_ids = {
            item.course.id for item in curriculum.items if item.is_completed
        }

        # Get all courses from the library
        all_courses = await self.library.list_courses()

        recommended: List[Course] = []
        for course in all_courses:
            # Skip if already in curriculum or completed
            if course.id in completed_ids:
                continue
            if any(item.course.id == course.id for item in curriculum.items):
                continue

            # Recommend if it matches user's role
            for role in user.roles:
                role_lower = role.lower()
                if (
                    course.role_required == role_lower
                    or any(tag in course.tags for tag in ["security", role_lower])
                ):
                    recommended.append(course)
                    break

        return recommended[:5]  # Limit recommendations

    async def get_overdue_training(self, user: UserProtocol) -> List[Course]:
        """Get overdue mandatory courses.

        Args:
            user: User object with id and roles.

        Returns:
            List of overdue Course objects.
        """
        curriculum = await self.get_curriculum(user)

        overdue: List[Course] = []
        for item in curriculum.items:
            if item.is_required and item.is_overdue:
                overdue.append(item.course)

        return overdue

    async def get_expiring_soon(
        self,
        user: UserProtocol,
        days: int = 30,
    ) -> List[CurriculumItem]:
        """Get courses with certificates expiring within specified days.

        Args:
            user: User object with id and roles.
            days: Number of days to look ahead.

        Returns:
            List of CurriculumItems with expiring certificates.
        """
        curriculum = await self.get_curriculum(user)
        threshold = datetime.now(timezone.utc) + timedelta(days=days)

        expiring: List[CurriculumItem] = []
        for item in curriculum.items:
            if item.certificate and item.certificate.expires_at < threshold:
                expiring.append(item)

        return expiring

    async def check_prerequisites(
        self,
        user: UserProtocol,
        course_id: str,
    ) -> tuple[bool, List[str]]:
        """Check if user has completed prerequisites for a course.

        Args:
            user: User object with id and roles.
            course_id: Course to check prerequisites for.

        Returns:
            Tuple of (prerequisites_met, missing_course_ids).
        """
        course = await self.library.get_course(course_id)
        if course is None:
            return False, []

        if not course.prerequisites:
            return True, []

        completions = self._get_user_completions(user.id)
        certificates = self._get_user_certificates(user.id)

        missing: List[str] = []
        for prereq_id in course.prerequisites:
            completion = completions.get(prereq_id)
            certificate = certificates.get(prereq_id)

            if completion is None or completion.passed is not True:
                missing.append(prereq_id)
            elif self._is_completion_expired(completion, certificate):
                missing.append(prereq_id)

        return len(missing) == 0, missing

    def add_completion(
        self,
        user_id: str,
        completion: TrainingCompletion,
    ) -> None:
        """Add a completion record (for testing/caching).

        Args:
            user_id: User identifier.
            completion: Completion record to add.
        """
        if user_id not in self._completions:
            self._completions[user_id] = []
        self._completions[user_id].append(completion)

    def add_certificate(
        self,
        user_id: str,
        certificate: Certificate,
    ) -> None:
        """Add a certificate (for testing/caching).

        Args:
            user_id: User identifier.
            certificate: Certificate to add.
        """
        if user_id not in self._certificates:
            self._certificates[user_id] = []
        self._certificates[user_id].append(certificate)


__all__ = [
    "ALL_EMPLOYEE_COURSES",
    "CurriculumItem",
    "CurriculumMapper",
    "ROLE_CURRICULA",
    "UserCurriculum",
    "UserProtocol",
]
