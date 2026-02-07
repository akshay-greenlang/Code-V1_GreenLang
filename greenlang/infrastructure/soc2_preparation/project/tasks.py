# -*- coding: utf-8 -*-
"""
Audit Task Manager Module - SEC-009 Phase 8

Provides task management and kanban-style workflow tracking for SOC 2 audit
projects. Supports task creation, assignment, status tracking, and workload
analysis.

Task Status Flow:
    TODO -> IN_PROGRESS -> DONE

Classes:
    - TaskStatus: Task workflow states
    - TaskPriority: Task priority levels
    - AuditTask: Core task data model
    - TaskBoard: Kanban board view
    - AuditTaskManager: Task orchestration class

Example:
    >>> manager = AuditTaskManager(config)
    >>> task = await manager.create_task(TaskCreate(
    ...     project_id="proj-123",
    ...     title="Collect access logs",
    ...     description="Export access logs for CC6.1 testing",
    ...     category="evidence_collection",
    ... ))
    >>> await manager.assign_task(task.task_id, assignee_id="user-456")
    >>> await manager.update_status(task.task_id, "in_progress")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Task workflow states for kanban-style tracking."""

    TODO = "todo"
    """Task is pending and not yet started."""

    IN_PROGRESS = "in_progress"
    """Task is actively being worked on."""

    BLOCKED = "blocked"
    """Task is blocked waiting on external dependency."""

    IN_REVIEW = "in_review"
    """Task is complete and awaiting review."""

    DONE = "done"
    """Task is completed and verified."""

    CANCELLED = "cancelled"
    """Task was cancelled."""


class TaskPriority(str, Enum):
    """Task priority levels."""

    CRITICAL = "critical"
    """Must be completed immediately - audit blocker."""

    HIGH = "high"
    """Important - should be completed this week."""

    MEDIUM = "medium"
    """Standard priority - complete by target date."""

    LOW = "low"
    """Nice to have - complete as time allows."""


class TaskCategory(str, Enum):
    """Task categories for SOC 2 audit work."""

    EVIDENCE_COLLECTION = "evidence_collection"
    """Gathering evidence for control testing."""

    CONTROL_TESTING = "control_testing"
    """Executing control tests."""

    FINDING_REMEDIATION = "finding_remediation"
    """Addressing audit findings."""

    DOCUMENTATION = "documentation"
    """Creating or updating documentation."""

    WALKTHROUGH = "walkthrough"
    """Control walkthrough preparation."""

    REVIEW = "review"
    """Review and approval tasks."""

    COMMUNICATION = "communication"
    """Auditor communication and meetings."""

    OTHER = "other"
    """Other audit-related tasks."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class TaskCreate(BaseModel):
    """Input model for creating a new task."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    project_id: str = Field(
        ...,
        description="ID of the parent audit project.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Task title.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed task description.",
    )
    category: TaskCategory = Field(
        default=TaskCategory.OTHER,
        description="Task category.",
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level.",
    )
    due_date: Optional[date] = Field(
        default=None,
        description="Task due date.",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the assignee.",
    )
    milestone_id: Optional[str] = Field(
        default=None,
        description="Associated milestone ID.",
    )
    control_id: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Associated control ID (e.g., CC6.1).",
    )
    estimated_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="Estimated hours to complete.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Task tags for filtering.",
    )


class AuditTask(BaseModel):
    """Core audit task data model.

    Attributes:
        task_id: Unique identifier for the task.
        project_id: Parent project ID.
        title: Task title.
        description: Detailed description.
        category: Task category.
        status: Current task status.
        priority: Task priority level.
        due_date: Target completion date.
        assignee_id: User ID of the assignee.
        milestone_id: Associated milestone ID.
        control_id: Associated control ID.
        estimated_hours: Estimated hours to complete.
        actual_hours: Actual hours spent.
        tags: Task tags.
        metadata: Additional metadata.
        created_at: Creation timestamp.
        updated_at: Last modification timestamp.
        completed_at: Completion timestamp.
        created_by: User ID of the creator.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the task.",
    )
    project_id: str = Field(
        ...,
        description="Parent project ID.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Task title.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed task description.",
    )
    category: TaskCategory = Field(
        default=TaskCategory.OTHER,
        description="Task category.",
    )
    status: TaskStatus = Field(
        default=TaskStatus.TODO,
        description="Current task status.",
    )
    priority: TaskPriority = Field(
        default=TaskPriority.MEDIUM,
        description="Task priority level.",
    )
    due_date: Optional[date] = Field(
        default=None,
        description="Target completion date.",
    )
    assignee_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="User ID of the assignee.",
    )
    milestone_id: Optional[str] = Field(
        default=None,
        description="Associated milestone ID.",
    )
    control_id: Optional[str] = Field(
        default=None,
        max_length=32,
        description="Associated control ID (e.g., CC6.1).",
    )
    estimated_hours: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1000.0,
        description="Estimated hours to complete.",
    )
    actual_hours: float = Field(
        default=0.0,
        ge=0.0,
        description="Actual hours spent.",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Task tags for filtering.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional task metadata.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last modification timestamp (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion timestamp (UTC).",
    )
    created_by: str = Field(
        default="",
        max_length=256,
        description="User ID of the creator.",
    )

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase and remove duplicates."""
        seen: set[str] = set()
        result: List[str] = []
        for tag in v:
            t = tag.strip().lower()
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    @property
    def is_overdue(self) -> bool:
        """Check if the task is past its due date and not completed."""
        if self.status == TaskStatus.DONE:
            return False
        if self.due_date is None:
            return False
        return date.today() > self.due_date

    @property
    def days_until_due(self) -> Optional[int]:
        """Calculate days until due date (negative if overdue)."""
        if self.due_date is None:
            return None
        return (self.due_date - date.today()).days


class TaskBoard(BaseModel):
    """Kanban board view of tasks.

    Attributes:
        project_id: Project ID for this board.
        todo: Tasks in TODO status.
        in_progress: Tasks in IN_PROGRESS status.
        blocked: Tasks in BLOCKED status.
        in_review: Tasks in IN_REVIEW status.
        done: Tasks in DONE status.
        total_tasks: Total number of tasks.
        completed_percentage: Percentage of tasks completed.
    """

    model_config = ConfigDict(extra="forbid")

    project_id: str = Field(..., description="Project ID for this board.")
    todo: List[AuditTask] = Field(
        default_factory=list,
        description="Tasks in TODO status.",
    )
    in_progress: List[AuditTask] = Field(
        default_factory=list,
        description="Tasks in IN_PROGRESS status.",
    )
    blocked: List[AuditTask] = Field(
        default_factory=list,
        description="Tasks in BLOCKED status.",
    )
    in_review: List[AuditTask] = Field(
        default_factory=list,
        description="Tasks in IN_REVIEW status.",
    )
    done: List[AuditTask] = Field(
        default_factory=list,
        description="Tasks in DONE status.",
    )
    total_tasks: int = Field(default=0, description="Total number of tasks.")
    completed_percentage: float = Field(
        default=0.0,
        description="Percentage of tasks completed.",
    )


# ---------------------------------------------------------------------------
# Task Manager
# ---------------------------------------------------------------------------


class AuditTaskManager:
    """Task management and workflow tracking for audit projects.

    Provides task creation, assignment, status tracking, and workload
    analysis for SOC 2 audit coordination.

    Attributes:
        config: Configuration instance.
        _tasks: In-memory task storage.

    Example:
        >>> manager = AuditTaskManager(config)
        >>> task = await manager.create_task(task_create)
        >>> await manager.assign_task(task.task_id, "user-123")
    """

    def __init__(self, config: Any = None) -> None:
        """Initialize AuditTaskManager.

        Args:
            config: Optional configuration instance.
        """
        self.config = config
        self._tasks: Dict[str, AuditTask] = {}
        logger.info("AuditTaskManager initialized")

    async def create_task(self, task: TaskCreate) -> AuditTask:
        """Create a new audit task.

        Args:
            task: Task creation parameters.

        Returns:
            The created AuditTask.
        """
        start_time = datetime.now(timezone.utc)

        audit_task = AuditTask(
            project_id=task.project_id,
            title=task.title,
            description=task.description,
            category=task.category,
            priority=task.priority,
            due_date=task.due_date,
            assignee_id=task.assignee_id,
            milestone_id=task.milestone_id,
            control_id=task.control_id,
            estimated_hours=task.estimated_hours,
            tags=task.tags,
        )

        self._tasks[audit_task.task_id] = audit_task

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Task created: id=%s, title='%s', project=%s, "
            "category=%s, priority=%s, elapsed=%.2fms",
            audit_task.task_id,
            audit_task.title,
            audit_task.project_id,
            audit_task.category.value,
            audit_task.priority.value,
            elapsed_ms,
        )

        return audit_task

    async def assign_task(
        self,
        task_id: str,
        assignee_id: str,
    ) -> None:
        """Assign a task to a user.

        Args:
            task_id: ID of the task to assign.
            assignee_id: User ID of the assignee.

        Raises:
            ValueError: If task not found.
        """
        task = await self._get_task(task_id)

        old_assignee = task.assignee_id
        task.assignee_id = assignee_id
        task.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Task assigned: task=%s, assignee=%s (was %s)",
            task_id,
            assignee_id,
            old_assignee,
        )

    async def update_status(
        self,
        task_id: str,
        status: str,
    ) -> None:
        """Update a task's status.

        Args:
            task_id: ID of the task to update.
            status: New status value.

        Raises:
            ValueError: If task not found or invalid status.
        """
        task = await self._get_task(task_id)

        old_status = task.status
        task.status = TaskStatus(status)
        task.updated_at = datetime.now(timezone.utc)

        # Set completed_at if transitioning to DONE
        if task.status == TaskStatus.DONE and old_status != TaskStatus.DONE:
            task.completed_at = datetime.now(timezone.utc)

        logger.info(
            "Task status updated: task=%s, status=%s (was %s)",
            task_id,
            task.status.value,
            old_status.value,
        )

    def get_task_board(self, project_id: str) -> TaskBoard:
        """Get a kanban board view of tasks for a project.

        Args:
            project_id: ID of the project.

        Returns:
            TaskBoard with tasks organized by status.
        """
        project_tasks = [t for t in self._tasks.values() if t.project_id == project_id]

        # Organize by status
        todo = [t for t in project_tasks if t.status == TaskStatus.TODO]
        in_progress = [t for t in project_tasks if t.status == TaskStatus.IN_PROGRESS]
        blocked = [t for t in project_tasks if t.status == TaskStatus.BLOCKED]
        in_review = [t for t in project_tasks if t.status == TaskStatus.IN_REVIEW]
        done = [t for t in project_tasks if t.status == TaskStatus.DONE]

        # Sort each column by priority then due date
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }

        def sort_key(task: AuditTask) -> tuple:
            p = priority_order.get(task.priority, 99)
            d = task.due_date or date.max
            return (p, d)

        todo.sort(key=sort_key)
        in_progress.sort(key=sort_key)
        blocked.sort(key=sort_key)
        in_review.sort(key=sort_key)
        done.sort(key=lambda t: t.completed_at or t.updated_at, reverse=True)

        # Calculate metrics
        total = len(project_tasks)
        completed = len(done)
        completed_pct = (completed / total * 100) if total > 0 else 0.0

        return TaskBoard(
            project_id=project_id,
            todo=todo,
            in_progress=in_progress,
            blocked=blocked,
            in_review=in_review,
            done=done,
            total_tasks=total,
            completed_percentage=completed_pct,
        )

    def calculate_progress(self, project_id: str) -> float:
        """Calculate completion percentage for a project.

        Args:
            project_id: ID of the project.

        Returns:
            Completion percentage (0-100).
        """
        project_tasks = [t for t in self._tasks.values() if t.project_id == project_id]

        if not project_tasks:
            return 0.0

        completed = sum(1 for t in project_tasks if t.status == TaskStatus.DONE)
        return (completed / len(project_tasks)) * 100

    def get_assignee_workload(self, assignee_id: str) -> List[AuditTask]:
        """Get all tasks assigned to a user.

        Args:
            assignee_id: User ID of the assignee.

        Returns:
            List of AuditTask objects assigned to the user.
        """
        tasks = [
            t
            for t in self._tasks.values()
            if t.assignee_id == assignee_id and t.status not in (TaskStatus.DONE, TaskStatus.CANCELLED)
        ]

        # Sort by priority and due date
        priority_order = {
            TaskPriority.CRITICAL: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 3,
        }

        tasks.sort(
            key=lambda t: (priority_order.get(t.priority, 99), t.due_date or date.max)
        )

        return tasks

    async def get_task(self, task_id: str) -> Optional[AuditTask]:
        """Get a task by ID.

        Args:
            task_id: ID of the task.

        Returns:
            AuditTask if found, None otherwise.
        """
        return self._tasks.get(task_id)

    async def list_tasks(
        self,
        project_id: Optional[str] = None,
        status: Optional[TaskStatus] = None,
        category: Optional[TaskCategory] = None,
        assignee_id: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
    ) -> List[AuditTask]:
        """List tasks with optional filtering.

        Args:
            project_id: Filter by project ID.
            status: Filter by status.
            category: Filter by category.
            assignee_id: Filter by assignee.
            priority: Filter by priority.

        Returns:
            List of matching AuditTask objects.
        """
        tasks = list(self._tasks.values())

        if project_id is not None:
            tasks = [t for t in tasks if t.project_id == project_id]

        if status is not None:
            tasks = [t for t in tasks if t.status == status]

        if category is not None:
            tasks = [t for t in tasks if t.category == category]

        if assignee_id is not None:
            tasks = [t for t in tasks if t.assignee_id == assignee_id]

        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]

        return tasks

    async def log_time(
        self,
        task_id: str,
        hours: float,
    ) -> None:
        """Log time spent on a task.

        Args:
            task_id: ID of the task.
            hours: Hours to add.

        Raises:
            ValueError: If task not found or invalid hours.
        """
        if hours <= 0:
            raise ValueError("Hours must be positive.")

        task = await self._get_task(task_id)
        task.actual_hours += hours
        task.updated_at = datetime.now(timezone.utc)

        logger.info(
            "Time logged: task=%s, hours=%.2f, total=%.2f",
            task_id,
            hours,
            task.actual_hours,
        )

    async def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[TaskPriority] = None,
        due_date: Optional[date] = None,
        tags: Optional[List[str]] = None,
    ) -> AuditTask:
        """Update task fields.

        Args:
            task_id: ID of the task to update.
            title: New title (optional).
            description: New description (optional).
            priority: New priority (optional).
            due_date: New due date (optional).
            tags: New tags (optional).

        Returns:
            Updated AuditTask.

        Raises:
            ValueError: If task not found.
        """
        task = await self._get_task(task_id)

        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = priority
        if due_date is not None:
            task.due_date = due_date
        if tags is not None:
            task.tags = tags

        task.updated_at = datetime.now(timezone.utc)

        logger.info("Task updated: id=%s", task_id)
        return task

    async def delete_task(self, task_id: str) -> None:
        """Delete a task.

        Args:
            task_id: ID of the task to delete.

        Raises:
            ValueError: If task not found.
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task '{task_id}' not found.")

        del self._tasks[task_id]
        logger.info("Task deleted: id=%s", task_id)

    # -----------------------------------------------------------------------
    # Private Methods
    # -----------------------------------------------------------------------

    async def _get_task(self, task_id: str) -> AuditTask:
        """Get a task or raise ValueError if not found."""
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")
        return task


__all__ = [
    "TaskStatus",
    "TaskPriority",
    "TaskCategory",
    "TaskCreate",
    "AuditTask",
    "TaskBoard",
    "AuditTaskManager",
]
