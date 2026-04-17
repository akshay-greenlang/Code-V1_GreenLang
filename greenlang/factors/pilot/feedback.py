# -*- coding: utf-8 -*-
"""
Pilot feedback collection and analysis (F093).

Collects structured feedback from design partners and generates
actionable insights for product improvement.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    COVERAGE_GAP = "coverage_gap"
    API_USABILITY = "api_usability"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    PRICING = "pricing"
    OTHER = "other"


class FeedbackPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeedbackStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


@dataclass
class FeedbackEntry:
    """A single feedback entry from a pilot partner."""

    feedback_id: str
    partner_id: str
    category: FeedbackCategory
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    status: FeedbackStatus = FeedbackStatus.NEW
    title: str = ""
    description: str = ""
    factor_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    resolved_at: str = ""
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "partner_id": self.partner_id,
            "category": self.category.value,
            "priority": self.priority.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "factor_ids": self.factor_ids,
            "tags": self.tags,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


class FeedbackCollector:
    """
    Collects and manages feedback from pilot partners.

    Supports CRUD operations, status tracking, and querying by
    partner/category/priority/status.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, FeedbackEntry] = {}

    def submit(
        self,
        partner_id: str,
        category: FeedbackCategory,
        title: str,
        description: str = "",
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        factor_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> FeedbackEntry:
        """Submit new feedback."""
        entry = FeedbackEntry(
            feedback_id=f"fb_{uuid.uuid4().hex[:12]}",
            partner_id=partner_id,
            category=category,
            priority=priority,
            title=title,
            description=description,
            factor_ids=factor_ids or [],
            tags=tags or [],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._entries[entry.feedback_id] = entry
        logger.info(
            "Feedback submitted: %s from %s [%s/%s]",
            entry.feedback_id, partner_id, category.value, priority.value,
        )
        return entry

    def get(self, feedback_id: str) -> Optional[FeedbackEntry]:
        return self._entries.get(feedback_id)

    def acknowledge(self, feedback_id: str) -> Optional[FeedbackEntry]:
        entry = self._entries.get(feedback_id)
        if entry:
            entry.status = FeedbackStatus.ACKNOWLEDGED
        return entry

    def resolve(self, feedback_id: str, notes: str = "") -> Optional[FeedbackEntry]:
        entry = self._entries.get(feedback_id)
        if entry:
            entry.status = FeedbackStatus.RESOLVED
            entry.resolved_at = datetime.now(timezone.utc).isoformat()
            entry.resolution_notes = notes
        return entry

    def list_by_partner(self, partner_id: str) -> List[FeedbackEntry]:
        return [e for e in self._entries.values() if e.partner_id == partner_id]

    def list_by_category(self, category: FeedbackCategory) -> List[FeedbackEntry]:
        return [e for e in self._entries.values() if e.category == category]

    def list_by_status(self, status: FeedbackStatus) -> List[FeedbackEntry]:
        return [e for e in self._entries.values() if e.status == status]

    def list_open(self) -> List[FeedbackEntry]:
        return [
            e for e in self._entries.values()
            if e.status in (FeedbackStatus.NEW, FeedbackStatus.ACKNOWLEDGED, FeedbackStatus.IN_PROGRESS)
        ]

    def all_entries(self) -> List[FeedbackEntry]:
        return list(self._entries.values())


class FeedbackAnalyzer:
    """
    Analyzes collected feedback for actionable insights.

    Provides:
      - Category distribution
      - Priority heat map
      - Partner satisfaction proxy
      - Top issues ranking
    """

    def __init__(self, collector: FeedbackCollector) -> None:
        self._collector = collector

    def category_distribution(self) -> Dict[str, int]:
        """Count feedback entries by category."""
        dist: Dict[str, int] = defaultdict(int)
        for entry in self._collector.all_entries():
            dist[entry.category.value] += 1
        return dict(dist)

    def priority_distribution(self) -> Dict[str, int]:
        """Count feedback entries by priority."""
        dist: Dict[str, int] = defaultdict(int)
        for entry in self._collector.all_entries():
            dist[entry.priority.value] += 1
        return dict(dist)

    def resolution_rate(self) -> float:
        """Percentage of feedback that has been resolved."""
        entries = self._collector.all_entries()
        if not entries:
            return 0.0
        resolved = sum(1 for e in entries if e.status == FeedbackStatus.RESOLVED)
        return resolved / len(entries)

    def top_issues(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Rank open issues by priority (critical first)."""
        priority_order = {
            FeedbackPriority.CRITICAL: 0,
            FeedbackPriority.HIGH: 1,
            FeedbackPriority.MEDIUM: 2,
            FeedbackPriority.LOW: 3,
        }
        open_entries = self._collector.list_open()
        ranked = sorted(open_entries, key=lambda e: priority_order.get(e.priority, 9))
        return [e.to_dict() for e in ranked[:limit]]

    def partner_satisfaction_proxy(self) -> Dict[str, float]:
        """
        Estimate satisfaction per partner (0-100).

        Higher = fewer critical/high issues and better resolution rate.
        """
        entries = self._collector.all_entries()
        by_partner: Dict[str, List[FeedbackEntry]] = defaultdict(list)
        for e in entries:
            by_partner[e.partner_id].append(e)

        scores: Dict[str, float] = {}
        for pid, partner_entries in by_partner.items():
            total = len(partner_entries)
            critical = sum(1 for e in partner_entries if e.priority == FeedbackPriority.CRITICAL)
            high = sum(1 for e in partner_entries if e.priority == FeedbackPriority.HIGH)
            resolved = sum(1 for e in partner_entries if e.status == FeedbackStatus.RESOLVED)

            # Penalty for critical/high, bonus for resolution
            base = 80.0
            base -= critical * 15
            base -= high * 5
            base += (resolved / max(total, 1)) * 20
            scores[pid] = max(0.0, min(100.0, round(base, 1)))

        return scores

    def full_report(self) -> Dict[str, Any]:
        """Generate comprehensive feedback analysis report."""
        return {
            "total_feedback": len(self._collector.all_entries()),
            "open_count": len(self._collector.list_open()),
            "resolution_rate": round(self.resolution_rate(), 4),
            "category_distribution": self.category_distribution(),
            "priority_distribution": self.priority_distribution(),
            "top_issues": self.top_issues(10),
            "partner_satisfaction": self.partner_satisfaction_proxy(),
        }
