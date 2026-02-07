# -*- coding: utf-8 -*-
"""
Audit Project Management Module - SEC-009 Phase 8

This module provides comprehensive audit project management for SOC 2 Type II
preparation. Includes timeline management, task tracking, milestone monitoring,
and status reporting for audit coordination.

Submodules:
    - timeline: Audit timeline and milestone management
    - tasks: Task tracking and kanban-style workflow
    - status: Status reporting and KPI calculation

Public API:
    - AuditTimeline: Timeline and milestone orchestration
    - AuditTaskManager: Task management and assignment
    - AuditStatusReporter: Status reports and executive summaries
    - AuditPhase: Project phases enumeration
    - AuditProject: Core project data model

Example:
    >>> from greenlang.infrastructure.soc2_preparation.project import (
    ...     AuditTimeline,
    ...     AuditTaskManager,
    ...     AuditStatusReporter,
    ... )
    >>> timeline = AuditTimeline(config)
    >>> project = await timeline.create_timeline(project_create)
    >>> await timeline.add_milestone(project.project_id, milestone)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.soc2_preparation.project.timeline import (
    AuditTimeline,
    AuditPhase,
    AuditProject,
    AuditProjectCreate,
    AuditMilestone,
    MilestoneCreate,
    DelayedMilestone,
    TimelineView,
)
from greenlang.infrastructure.soc2_preparation.project.tasks import (
    AuditTaskManager,
    AuditTask,
    TaskCreate,
    TaskStatus,
    TaskBoard,
)
from greenlang.infrastructure.soc2_preparation.project.status import (
    AuditStatusReporter,
    ProjectKPIs,
    WeeklyReport,
)

logger = logging.getLogger(__name__)

__all__ = [
    # Timeline
    "AuditTimeline",
    "AuditPhase",
    "AuditProject",
    "AuditProjectCreate",
    "AuditMilestone",
    "MilestoneCreate",
    "DelayedMilestone",
    "TimelineView",
    # Tasks
    "AuditTaskManager",
    "AuditTask",
    "TaskCreate",
    "TaskStatus",
    "TaskBoard",
    # Status
    "AuditStatusReporter",
    "ProjectKPIs",
    "WeeklyReport",
]
