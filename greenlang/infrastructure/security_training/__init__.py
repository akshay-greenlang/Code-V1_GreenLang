# -*- coding: utf-8 -*-
"""
GreenLang Security Training Platform - SEC-010

Production-grade security training and phishing simulation platform for the
GreenLang Climate OS. Provides role-based training curricula, interactive
assessments, phishing campaign management, and employee security scoring.

Components:
    - ContentLibrary: Training course content management
    - CurriculumMapper: Role-based training path assignment
    - AssessmentEngine: Quiz generation and grading with certification
    - PhishingSimulator: Automated phishing simulation campaigns
    - CompletionTracker: Training progress and compliance tracking
    - SecurityScorer: Employee security posture scoring

Public API:
    - ContentLibrary: Manages training course catalog and content
    - CurriculumMapper: Maps roles to required training paths
    - AssessmentEngine: Handles quiz generation, grading, and certificates
    - PhishingSimulator: Runs phishing simulation campaigns
    - CompletionTracker: Tracks training completion and compliance
    - SecurityScorer: Calculates employee security scores

Example:
    >>> from greenlang.infrastructure.security_training import (
    ...     ContentLibrary, CurriculumMapper, AssessmentEngine,
    ...     PhishingSimulator, CompletionTracker, SecurityScorer,
    ... )
    >>> library = ContentLibrary()
    >>> courses = library.list_courses(role_filter="developer")
    >>> mapper = CurriculumMapper(library)
    >>> curriculum = await mapper.get_curriculum(user)
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.security_training.models import (
    CampaignMetrics,
    CampaignStatus,
    Certificate,
    Course,
    CourseContent,
    Module,
    PhishingCampaign,
    PhishingResult,
    Question,
    QuizSubmission,
    SecurityScore,
    TemplateType,
    TrainingCompletion,
    UserProgress,
)
from greenlang.infrastructure.security_training.config import (
    TrainingConfig,
    get_config,
    reset_config,
)
from greenlang.infrastructure.security_training.content_library import ContentLibrary
from greenlang.infrastructure.security_training.curriculum_mapper import CurriculumMapper
from greenlang.infrastructure.security_training.assessment_engine import AssessmentEngine
from greenlang.infrastructure.security_training.phishing_simulator import PhishingSimulator
from greenlang.infrastructure.security_training.completion_tracker import CompletionTracker
from greenlang.infrastructure.security_training.security_scorer import SecurityScorer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "TrainingConfig",
    "get_config",
    "reset_config",
    # Models
    "CampaignMetrics",
    "CampaignStatus",
    "Certificate",
    "Course",
    "CourseContent",
    "Module",
    "PhishingCampaign",
    "PhishingResult",
    "Question",
    "QuizSubmission",
    "SecurityScore",
    "TemplateType",
    "TrainingCompletion",
    "UserProgress",
    # Services
    "ContentLibrary",
    "CurriculumMapper",
    "AssessmentEngine",
    "PhishingSimulator",
    "CompletionTracker",
    "SecurityScorer",
]
