# -*- coding: utf-8 -*-
"""
ConflictDetectorEngine - Validation Rule Conflict Detection Engine

This module implements the ConflictDetectorEngine for the Validation Rule
Engine (AGENT-DATA-019, GL-DATA-X-022). It is Engine 4 of 7 in the
validation rule engine pipeline.

The engine detects contradictory, overlapping, or redundant validation
rules across a rule set. It analyses rules grouped by column to limit
O(n^2) pair-wise comparisons and produces structured conflict reports
with severity grading, resolution suggestions, and SHA-256 provenance
hashes for complete audit trail coverage.

Zero-Hallucination Guarantees:
    - All conflict determinations are deterministic rule comparisons.
    - Range overlap and contradiction checks use Python arithmetic only.
    - No LLM calls for any numeric or logical conflict detection.
    - SHA-256 provenance chains every conflict report for tamper-evident audit.

Conflict Types (5):
    - RANGE_OVERLAP: Two RANGE rules on the same column with overlapping
      bounds that may cause ambiguous validation results.
    - RANGE_CONTRADICTION: Two RANGE rules on the same column with no valid
      intersection, making it impossible for any value to satisfy both.
    - FORMAT_CONFLICT: Two FORMAT rules on the same column with incompatible
      regex patterns that no string can simultaneously match.
    - SEVERITY_INCONSISTENCY: Rules with identical conditions assigned
      different severity levels, causing inconsistent enforcement.
    - REDUNDANCY: One rule is a logical subset of another (its range is
      entirely within the other's), or both rules have identical conditions.

Conflict Severity:
    - high: Contradictions (no valid values satisfy all rules).
    - medium: Overlaps (ambiguous but not necessarily contradictory).
    - low: Redundancies (wasteful but not harmful).

Example:
    >>> from greenlang.validation_rule_engine.conflict_detector import (
    ...     ConflictDetectorEngine,
    ... )
    >>> # Assuming a RuleRegistryEngine is available
    >>> engine = ConflictDetectorEngine(registry=rule_registry)
    >>> report = engine.detect_all_conflicts()
    >>> print(report["total_conflicts"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Engine: 4 of 7 -- ConflictDetectorEngine
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graceful import of ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.provenance import ProvenanceTracker
except ImportError:  # pragma: no cover -- fallback when provenance not yet built
    try:
        from greenlang.schema_migration.provenance import ProvenanceTracker
    except ImportError:

        class ProvenanceTracker:  # type: ignore[no-redef]
            """Minimal stub when provenance module is unavailable."""

            def __init__(self, genesis_hash: str = "stub") -> None:
                self._entries: list = []

            def record(
                self,
                entity_type: str,
                entity_id: str,
                action: str,
                data: Any = None,
            ) -> Any:
                """No-op record."""
                return None

            def build_hash(self, data: Any) -> str:
                """Build a SHA-256 hash."""
                serialized = json.dumps(data, sort_keys=True, default=str)
                return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

            @property
            def entry_count(self) -> int:
                return len(self._entries)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ConflictType(str, Enum):
    """Classification of detected rule conflict.

    RANGE_OVERLAP: Two RANGE rules share overlapping numeric bounds.
    RANGE_CONTRADICTION: Two RANGE rules have no valid intersection.
    FORMAT_CONFLICT: Two FORMAT rules have incompatible regex patterns.
    SEVERITY_INCONSISTENCY: Identical conditions with different severities.
    REDUNDANCY: One rule is a logical subset of another.
    """

    RANGE_OVERLAP = "range_overlap"
    RANGE_CONTRADICTION = "range_contradiction"
    FORMAT_CONFLICT = "format_conflict"
    SEVERITY_INCONSISTENCY = "severity_inconsistency"
    REDUNDANCY = "redundancy"


class ConflictSeverity(str, Enum):
    """Severity grade for a detected conflict.

    HIGH: Contradiction -- no valid values can satisfy all rules.
    MEDIUM: Overlap -- ambiguous validation results possible.
    LOW: Redundancy -- wasteful but not harmful.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ResolutionType(str, Enum):
    """Available conflict resolution strategies.

    KEEP_A: Retain rule A, deactivate or remove rule B.
    KEEP_B: Retain rule B, deactivate or remove rule A.
    MERGE: Merge the two rules into a single unified rule.
    IGNORE: Acknowledge the conflict but take no corrective action.
    """

    KEEP_A = "keep_a"
    KEEP_B = "keep_b"
    MERGE = "merge"
    IGNORE = "ignore"


# ---------------------------------------------------------------------------
# Severity ordering constants
# ---------------------------------------------------------------------------

#: Maps severity name to numeric rank for comparisons (higher = more severe).
_SEVERITY_RANK: Dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

#: Conflict type to default severity mapping.
_CONFLICT_TYPE_SEVERITY: Dict[str, str] = {
    ConflictType.RANGE_CONTRADICTION: ConflictSeverity.HIGH,
    ConflictType.FORMAT_CONFLICT: ConflictSeverity.MEDIUM,
    ConflictType.RANGE_OVERLAP: ConflictSeverity.MEDIUM,
    ConflictType.SEVERITY_INCONSISTENCY: ConflictSeverity.MEDIUM,
    ConflictType.REDUNDANCY: ConflictSeverity.LOW,
}

#: Maximum number of conflicts to store before warning.
MAX_STORED_CONFLICTS: int = 50_000

#: Default limit for list_conflicts pagination.
DEFAULT_LIST_LIMIT: int = 100


# ---------------------------------------------------------------------------
# Helper: UTC timestamp
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# ConflictDetectorEngine
# ---------------------------------------------------------------------------


class ConflictDetectorEngine:
    """Engine 4 of 7: Rule Conflict Detector for AGENT-DATA-019.

    Detects contradictory, overlapping, and redundant validation rules
    within a rule registry. Groups rules by column to reduce comparison
    complexity and produces structured conflict reports with SHA-256
    provenance hashes.

    Thread Safety:
        All public methods acquire ``self._lock`` before mutating shared
        state. The engine is safe for concurrent use across multiple threads.

    Zero-Hallucination:
        All conflict detection uses deterministic Python arithmetic and
        string comparison. No LLM calls for any numeric or logical
        determination.

    Attributes:
        _registry: Reference to the RuleRegistryEngine containing all rules.
        _conflicts: Dict mapping conflict_id to stored conflict records.
        _lock: Thread lock protecting shared state mutations.
        _provenance: ProvenanceTracker for SHA-256 audit chains.
        _stats: Running statistics updated after every detection pass.
        _resolutions: Dict mapping conflict_id to resolution records.

    Example:
        >>> engine = ConflictDetectorEngine(registry=rule_registry)
        >>> report = engine.detect_all_conflicts()
        >>> report["total_conflicts"]
        0
    """

    def __init__(
        self,
        registry: Any = None,
        provenance: Optional[ProvenanceTracker] = None,
        genesis_hash: Optional[str] = None,
    ) -> None:
        """Initialize ConflictDetectorEngine.

        Args:
            registry: A RuleRegistryEngine instance that provides access
                to registered validation rules via ``list_rules()``,
                ``get_rule()``, and ``get_rules_by_column()`` methods.
                If ``None``, an internal stub registry is created.
            provenance: Optional ProvenanceTracker instance for SHA-256
                audit chains. If ``None``, a new tracker is created.
            genesis_hash: Optional genesis hash for provenance tracker
                creation when no ``provenance`` is given.

        Example:
            >>> engine = ConflictDetectorEngine(registry=rule_registry)
            >>> engine.get_statistics()["total_detections"]
            0
        """
        self._registry: Any = registry
        self._conflicts: Dict[str, Dict[str, Any]] = {}
        self._lock: threading.Lock = threading.Lock()
        if provenance is not None:
            self._provenance: ProvenanceTracker = provenance
        elif genesis_hash is not None:
            self._provenance = ProvenanceTracker(genesis_hash=genesis_hash)
        else:
            self._provenance = ProvenanceTracker()
        self._stats: Dict[str, Any] = self._initial_stats()
        self._resolutions: Dict[str, Dict[str, Any]] = {}

        logger.info(
            "ConflictDetectorEngine initialized; registry_id=%s provenance_id=%s",
            id(self._registry),
            id(self._provenance),
        )

    # ------------------------------------------------------------------
    # 1. detect_all_conflicts
    # ------------------------------------------------------------------

    def detect_all_conflicts(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all conflict detectors and produce a unified conflict report.

        Groups rules by column to limit O(n^2) pair-wise comparison
        complexity. Each column group is checked for range overlaps,
        range contradictions, format conflicts, severity inconsistencies,
        and redundancies.

        Args:
            rule_ids: Optional list of rule IDs to analyse. When ``None``,
                all active rules in the registry are analysed.

        Returns:
            Dict with the following keys:

            - ``conflict_id`` (str): Unique ID for this detection report.
            - ``total_conflicts`` (int): Total number of conflicts found.
            - ``conflicts`` (List[Dict]): List of individual conflict records.
            - ``severity_distribution`` (Dict[str, int]): Count by severity.
            - ``type_distribution`` (Dict[str, int]): Count by conflict type.
            - ``recommendations`` (List[str]): High-level action items.
            - ``duration_ms`` (float): Processing time in milliseconds.
            - ``provenance_hash`` (str): SHA-256 hash of the full report.
            - ``analyzed_rules`` (int): Number of rules analysed.
            - ``analyzed_columns`` (int): Number of unique columns analysed.
            - ``detected_at`` (str): ISO-8601 UTC timestamp.

        Example:
            >>> report = engine.detect_all_conflicts()
            >>> report["total_conflicts"]
            0
        """
        start_ns = time.monotonic_ns()
        report_id = str(uuid.uuid4())

        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)

        all_conflicts: List[Dict[str, Any]] = []

        # Run each detector type across column groups
        range_overlaps = self._detect_range_overlaps_internal(rules, column_groups)
        range_contradictions = self._detect_range_contradictions_internal(
            rules, column_groups
        )
        format_conflicts = self._detect_format_conflicts_internal(
            rules, column_groups
        )
        severity_issues = self._detect_severity_inconsistencies_internal(
            rules, column_groups
        )
        redundancies = self._detect_redundancies_internal(rules, column_groups)
        conditional_conflicts = self._detect_conditional_conflicts_internal(rules)

        all_conflicts.extend(range_overlaps)
        all_conflicts.extend(range_contradictions)
        all_conflicts.extend(format_conflicts)
        all_conflicts.extend(severity_issues)
        all_conflicts.extend(redundancies)
        all_conflicts.extend(conditional_conflicts)

        # Store each conflict
        with self._lock:
            for conflict in all_conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        severity_dist = self._compute_severity_distribution(all_conflicts)
        type_dist = self._compute_type_distribution(all_conflicts)
        recommendations = self._generate_report_recommendations(
            all_conflicts, severity_dist
        )

        elapsed_ms = (time.monotonic_ns() - start_ns) / 1_000_000.0

        report_data = {
            "conflict_id": report_id,
            "total_conflicts": len(all_conflicts),
            "conflicts": all_conflicts,
            "severity_distribution": severity_dist,
            "type_distribution": type_dist,
            "recommendations": recommendations,
            "duration_ms": round(elapsed_ms, 3),
            "analyzed_rules": len(rules),
            "analyzed_columns": len(column_groups),
            "detected_at": _utcnow().isoformat(),
        }

        provenance_hash = self._compute_report_hash(report_data)
        report_data["provenance_hash"] = provenance_hash

        # Update statistics
        with self._lock:
            self._stats["total_detections"] += 1
            self._stats["total_conflicts_found"] += len(all_conflicts)
            self._stats["total_rules_analyzed"] += len(rules)
            for sev, count in severity_dist.items():
                key = f"conflicts_{sev}"
                self._stats[key] = self._stats.get(key, 0) + count

        self._provenance.record(
            entity_type="conflict_report",
            entity_id=report_id,
            action="detect_all_completed",
            data={
                "total_conflicts": len(all_conflicts),
                "analyzed_rules": len(rules),
                "provenance_hash": provenance_hash,
            },
        )

        logger.info(
            "detect_all_conflicts completed: report_id=%s total=%d "
            "rules=%d columns=%d duration_ms=%.1f",
            report_id,
            len(all_conflicts),
            len(rules),
            len(column_groups),
            elapsed_ms,
        )
        return report_data

    # ------------------------------------------------------------------
    # 2. detect_range_overlaps
    # ------------------------------------------------------------------

    def detect_range_overlaps(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find RANGE rules on the same column with overlapping bounds.

        Two RANGE rules overlap when their intervals share at least one
        common value. For example, RANGE [0, 100] and RANGE [50, 200]
        overlap on the interval [50, 100].

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.
                When ``None``, all active rules are analysed.

        Returns:
            List of conflict dicts, each containing:

            - ``conflict_id`` (str): Unique conflict identifier.
            - ``conflict_type`` (str): ``"range_overlap"``.
            - ``severity`` (str): ``"medium"``.
            - ``rule_a_id`` (str): ID of the first overlapping rule.
            - ``rule_b_id`` (str): ID of the second overlapping rule.
            - ``column`` (str): Column where the overlap occurs.
            - ``range_a`` (Dict): ``{"min": ..., "max": ...}`` for rule A.
            - ``range_b`` (Dict): ``{"min": ..., "max": ...}`` for rule B.
            - ``overlap_range`` (Dict): ``{"min": ..., "max": ...}``
              defining the overlap interval.
            - ``description`` (str): Human-readable conflict explanation.
            - ``detected_at`` (str): ISO-8601 UTC timestamp.

        Example:
            >>> overlaps = engine.detect_range_overlaps()
            >>> for o in overlaps:
            ...     print(o["column"], o["overlap_range"])
        """
        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)
        conflicts = self._detect_range_overlaps_internal(rules, column_groups)

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug("detect_range_overlaps: found %d overlaps", len(conflicts))
        return conflicts

    # ------------------------------------------------------------------
    # 3. detect_range_contradictions
    # ------------------------------------------------------------------

    def detect_range_contradictions(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find RANGE rules on the same column with no valid intersection.

        Two RANGE rules contradict each other when both are required
        (severity >= HIGH) and their intervals have no common values.
        For example, RANGE [0, 10] and RANGE [20, 30] on the same
        required column make it impossible for any value to pass both.

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.

        Returns:
            List of conflict dicts with ``conflict_type`` = ``"range_contradiction"``
            and ``severity`` = ``"high"``.

        Example:
            >>> contradictions = engine.detect_range_contradictions()
            >>> len(contradictions)
            0
        """
        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)
        conflicts = self._detect_range_contradictions_internal(
            rules, column_groups
        )

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug(
            "detect_range_contradictions: found %d contradictions",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # 4. detect_format_conflicts
    # ------------------------------------------------------------------

    def detect_format_conflicts(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find FORMAT rules on the same column with incompatible patterns.

        Two FORMAT rules are considered conflicting when their regex
        patterns produce non-overlapping match sets for typical inputs.
        For example, ``^\\d{3}$`` (exactly 3 digits) and ``^[A-Z]{2}\\d+``
        (2 uppercase letters followed by digits) cannot both match the
        same string.

        Detection is heuristic: the engine generates a small set of
        probe strings from each pattern's character class and tests
        cross-matching. Patterns that cannot cross-match are flagged.

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.

        Returns:
            List of conflict dicts with ``conflict_type`` = ``"format_conflict"``
            and ``severity`` = ``"medium"``.

        Example:
            >>> format_issues = engine.detect_format_conflicts()
        """
        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)
        conflicts = self._detect_format_conflicts_internal(rules, column_groups)

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug(
            "detect_format_conflicts: found %d format conflicts",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # 5. detect_severity_inconsistencies
    # ------------------------------------------------------------------

    def detect_severity_inconsistencies(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find rules with identical conditions but different severities.

        Two rules are considered condition-identical when they share the
        same rule type, column, operator, and threshold/pattern. If such
        rules are assigned different severity levels, enforcement becomes
        unpredictable.

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.

        Returns:
            List of conflict dicts with ``conflict_type`` =
            ``"severity_inconsistency"`` and ``severity`` = ``"medium"``.

        Example:
            >>> issues = engine.detect_severity_inconsistencies()
        """
        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)
        conflicts = self._detect_severity_inconsistencies_internal(
            rules, column_groups
        )

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug(
            "detect_severity_inconsistencies: found %d issues",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # 6. detect_redundancies
    # ------------------------------------------------------------------

    def detect_redundancies(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find rules that are logical subsets or duplicates of other rules.

        A rule is redundant when:
        - Its RANGE is entirely within another rule's RANGE on the same
          column (subset redundancy).
        - Its conditions are identical to another rule (duplicate).

        Redundant rules waste evaluation time and increase maintenance
        burden without adding validation value.

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.

        Returns:
            List of conflict dicts with ``conflict_type`` = ``"redundancy"``
            and ``severity`` = ``"low"``.

        Example:
            >>> redundant = engine.detect_redundancies()
        """
        rules = self._resolve_rules(rule_ids)
        column_groups = self._group_rules_by_column(rules)
        conflicts = self._detect_redundancies_internal(rules, column_groups)

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug(
            "detect_redundancies: found %d redundancies",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # 7. detect_conditional_conflicts
    # ------------------------------------------------------------------

    def detect_conditional_conflicts(
        self,
        rule_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find CONDITIONAL rules with contradictory predicates.

        Two CONDITIONAL rules conflict when they share the same predicate
        condition (e.g., IF country="US") but specify contradictory
        consequents (e.g., THEN currency="USD" vs THEN currency="EUR").

        Args:
            rule_ids: Optional list of rule IDs to restrict analysis.

        Returns:
            List of conflict dicts with ``conflict_type`` =
            ``"conditional_conflict"`` and ``severity`` = ``"high"``.

        Example:
            >>> cond_conflicts = engine.detect_conditional_conflicts()
        """
        rules = self._resolve_rules(rule_ids)
        conflicts = self._detect_conditional_conflicts_internal(rules)

        with self._lock:
            for conflict in conflicts:
                self._conflicts[conflict["conflict_id"]] = conflict

        logger.debug(
            "detect_conditional_conflicts: found %d conflicts",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # 8. analyze_rule_set_conflicts
    # ------------------------------------------------------------------

    def analyze_rule_set_conflicts(
        self,
        set_id: str,
    ) -> Dict[str, Any]:
        """Run all conflict detectors on rules within a specific rule set.

        Retrieves rules belonging to ``set_id`` from the registry and
        passes them through ``detect_all_conflicts`` for comprehensive
        analysis.

        Args:
            set_id: The rule set identifier to analyse. Must be non-empty.

        Returns:
            Same report structure as :meth:`detect_all_conflicts`, with
            an additional ``"set_id"`` key.

        Raises:
            ValueError: If ``set_id`` is empty or blank.

        Example:
            >>> report = engine.analyze_rule_set_conflicts("emissions_v2")
            >>> report["set_id"]
            'emissions_v2'
        """
        if not set_id or not set_id.strip():
            raise ValueError("set_id must be non-empty")

        rule_ids = self._get_rule_ids_for_set(set_id)
        report = self.detect_all_conflicts(rule_ids=rule_ids)
        report["set_id"] = set_id

        self._provenance.record(
            entity_type="conflict_report",
            entity_id=report["conflict_id"],
            action="rule_set_analysis_completed",
            data={"set_id": set_id, "total_conflicts": report["total_conflicts"]},
        )

        logger.info(
            "analyze_rule_set_conflicts: set_id=%s rules=%d conflicts=%d",
            set_id,
            report["analyzed_rules"],
            report["total_conflicts"],
        )
        return report

    # ------------------------------------------------------------------
    # 9. get_conflict
    # ------------------------------------------------------------------

    def get_conflict(
        self,
        conflict_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored conflict record by its unique identifier.

        Args:
            conflict_id: The UUID string of the conflict to retrieve.

        Returns:
            The stored conflict dict, or ``None`` if not found.

        Example:
            >>> conflict = engine.get_conflict("some-uuid")
            >>> conflict is None
            True
        """
        with self._lock:
            result = self._conflicts.get(conflict_id)

        if result is None:
            logger.debug("get_conflict: conflict_id=%s not found", conflict_id)
        return result

    # ------------------------------------------------------------------
    # 10. list_conflicts
    # ------------------------------------------------------------------

    def list_conflicts(
        self,
        conflict_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = DEFAULT_LIST_LIMIT,
    ) -> List[Dict[str, Any]]:
        """List stored conflict records with optional filtering.

        Results are ordered by detection timestamp (most recent first).

        Args:
            conflict_type: Optional filter by conflict type string.
                Must be one of the ``ConflictType`` values if provided.
            severity: Optional filter by severity string (``"high"``,
                ``"medium"``, or ``"low"``).
            limit: Maximum number of results to return. Must be >= 1.
                Defaults to ``DEFAULT_LIST_LIMIT`` (100).

        Returns:
            List of conflict dicts matching the filters, most recent first.

        Raises:
            ValueError: If ``limit`` is less than 1.

        Example:
            >>> conflicts = engine.list_conflicts(severity="high", limit=10)
        """
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        with self._lock:
            all_conflicts = list(self._conflicts.values())

        # Apply filters
        if conflict_type is not None:
            all_conflicts = [
                c for c in all_conflicts
                if c.get("conflict_type") == conflict_type
            ]

        if severity is not None:
            all_conflicts = [
                c for c in all_conflicts
                if c.get("severity") == severity
            ]

        # Sort by detected_at descending (most recent first)
        all_conflicts.sort(
            key=lambda c: c.get("detected_at", ""),
            reverse=True,
        )

        page = all_conflicts[:limit]
        logger.debug(
            "list_conflicts: total_matching=%d limit=%d returned=%d",
            len(all_conflicts),
            limit,
            len(page),
        )
        return page

    # ------------------------------------------------------------------
    # 11. resolve_conflict
    # ------------------------------------------------------------------

    def resolve_conflict(
        self,
        conflict_id: str,
        resolution: str,
    ) -> Dict[str, Any]:
        """Apply a resolution to a detected conflict.

        Marks the conflict as resolved and records the resolution
        strategy. Does not automatically modify or delete rules; callers
        must take follow-up action based on the resolution type.

        Valid resolution strategies:
            - ``"keep_a"``: Retain rule A, recommend deactivating rule B.
            - ``"keep_b"``: Retain rule B, recommend deactivating rule A.
            - ``"merge"``: Merge both rules into a single unified rule.
            - ``"ignore"``: Acknowledge the conflict but take no action.

        Args:
            conflict_id: The UUID of the conflict to resolve.
            resolution: The resolution strategy string.

        Returns:
            Dict with:

            - ``"conflict_id"`` (str): The resolved conflict ID.
            - ``"resolution"`` (str): The applied resolution strategy.
            - ``"status"`` (str): ``"resolved"``.
            - ``"resolved_at"`` (str): ISO-8601 UTC timestamp.
            - ``"provenance_hash"`` (str): SHA-256 hash of the resolution.

        Raises:
            ValueError: If ``conflict_id`` is not found, or ``resolution``
                is not a recognised strategy.

        Example:
            >>> result = engine.resolve_conflict("cf-123", "keep_a")
            >>> result["status"]
            'resolved'
        """
        valid_resolutions = {r.value for r in ResolutionType}
        if resolution not in valid_resolutions:
            raise ValueError(
                f"resolution must be one of {sorted(valid_resolutions)}, "
                f"got '{resolution}'"
            )

        with self._lock:
            conflict = self._conflicts.get(conflict_id)
            if conflict is None:
                raise ValueError(f"conflict_id '{conflict_id}' not found")

            resolved_at = _utcnow().isoformat()

            resolution_record = {
                "conflict_id": conflict_id,
                "resolution": resolution,
                "status": "resolved",
                "resolved_at": resolved_at,
                "original_conflict": conflict,
            }

            provenance_hash = self._compute_hash({
                "conflict_id": conflict_id,
                "resolution": resolution,
                "resolved_at": resolved_at,
            })
            resolution_record["provenance_hash"] = provenance_hash

            # Mark the conflict as resolved in-place
            conflict["status"] = "resolved"
            conflict["resolution"] = resolution
            conflict["resolved_at"] = resolved_at

            self._resolutions[conflict_id] = resolution_record
            self._stats["total_resolutions"] += 1

        self._provenance.record(
            entity_type="conflict_resolution",
            entity_id=conflict_id,
            action="conflict_resolved",
            data={
                "resolution": resolution,
                "provenance_hash": provenance_hash,
            },
        )

        logger.info(
            "resolve_conflict: conflict_id=%s resolution=%s",
            conflict_id,
            resolution,
        )
        return resolution_record

    # ------------------------------------------------------------------
    # 12. get_resolution_suggestions
    # ------------------------------------------------------------------

    def get_resolution_suggestions(
        self,
        conflict_id: str,
    ) -> List[Dict[str, Any]]:
        """Generate auto-resolution suggestions for a specific conflict.

        Suggestions are based on the conflict type and severity. Each
        suggestion includes a strategy name, rationale, and confidence
        score indicating the engine's certainty that the suggestion
        will resolve the conflict without side effects.

        Args:
            conflict_id: The UUID of the conflict to generate suggestions for.

        Returns:
            List of suggestion dicts, each containing:

            - ``"strategy"`` (str): Resolution strategy name.
            - ``"rationale"`` (str): Human-readable justification.
            - ``"confidence"`` (float): Score 0.0-1.0 indicating certainty.
            - ``"priority"`` (int): Lower number = higher priority.

        Raises:
            ValueError: If ``conflict_id`` is not found.

        Example:
            >>> suggestions = engine.get_resolution_suggestions("cf-123")
            >>> suggestions[0]["strategy"]
            'keep_a'
        """
        with self._lock:
            conflict = self._conflicts.get(conflict_id)

        if conflict is None:
            raise ValueError(f"conflict_id '{conflict_id}' not found")

        conflict_type = conflict.get("conflict_type", "")
        suggestions = self._build_suggestions_for_type(conflict_type, conflict)

        logger.debug(
            "get_resolution_suggestions: conflict_id=%s type=%s suggestions=%d",
            conflict_id,
            conflict_type,
            len(suggestions),
        )
        return suggestions

    # ------------------------------------------------------------------
    # 13. get_statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return running statistics for all detection operations.

        Returns:
            Dict with:

            - ``"total_detections"`` (int): Number of detection runs.
            - ``"total_conflicts_found"`` (int): Cumulative conflict count.
            - ``"total_rules_analyzed"`` (int): Cumulative rules analysed.
            - ``"total_resolutions"`` (int): Number of conflicts resolved.
            - ``"conflicts_high"`` (int): Count of high-severity conflicts.
            - ``"conflicts_medium"`` (int): Count of medium-severity conflicts.
            - ``"conflicts_low"`` (int): Count of low-severity conflicts.
            - ``"stored_conflicts"`` (int): Current in-memory conflict count.
            - ``"stored_resolutions"`` (int): Current resolution count.
            - ``"provenance_entries"`` (int): Provenance tracker entry count.

        Example:
            >>> stats = engine.get_statistics()
            >>> stats["total_detections"]
            0
        """
        with self._lock:
            stats = dict(self._stats)
            stats["stored_conflicts"] = len(self._conflicts)
            stats["stored_resolutions"] = len(self._resolutions)

        stats["provenance_entries"] = self._provenance.entry_count
        logger.debug("get_statistics: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # 14. clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all stored conflicts, resolutions, and reset statistics.

        Provenance entries are NOT cleared (the audit trail is immutable).
        Intended for test teardown to prevent state leakage.

        Example:
            >>> engine.clear()
            >>> engine.get_statistics()["total_detections"]
            0
        """
        with self._lock:
            self._conflicts.clear()
            self._resolutions.clear()
            self._stats = self._initial_stats()

        logger.info(
            "ConflictDetectorEngine state cleared (provenance preserved)"
        )

    # ==================================================================
    # INTERNAL: Detection Implementations
    # ==================================================================

    # ------------------------------------------------------------------
    # Internal: detect_range_overlaps
    # ------------------------------------------------------------------

    def _detect_range_overlaps_internal(
        self,
        rules: List[Dict[str, Any]],
        column_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect range overlaps within column groups.

        For each column with multiple RANGE rules, checks every pair
        for overlapping intervals. An overlap exists when
        max(min_a, min_b) <= min(max_a, max_b).

        Args:
            rules: Full list of resolved rules (unused, kept for signature
                consistency).
            column_groups: Rules grouped by column name.

        Returns:
            List of range overlap conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for column, col_rules in column_groups.items():
            range_rules = [
                r for r in col_rules
                if self._get_rule_type(r) == "range"
            ]
            if len(range_rules) < 2:
                continue

            for i in range(len(range_rules)):
                for j in range(i + 1, len(range_rules)):
                    rule_a = range_rules[i]
                    rule_b = range_rules[j]
                    pair_key = self._make_pair_key(rule_a, rule_b)

                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    range_a = self._extract_range(rule_a)
                    range_b = self._extract_range(rule_b)

                    if range_a is None or range_b is None:
                        continue

                    overlap = self._compute_range_overlap(range_a, range_b)
                    if overlap is not None:
                        # Check if it is a full contradiction (handled separately)
                        # Overlaps are only flagged when there IS a valid intersection
                        conflict = self._build_conflict(
                            conflict_type=ConflictType.RANGE_OVERLAP,
                            severity=ConflictSeverity.MEDIUM,
                            rule_a_id=self._get_rule_id(rule_a),
                            rule_b_id=self._get_rule_id(rule_b),
                            column=column,
                            description=(
                                f"RANGE rules on column '{column}' have "
                                f"overlapping bounds: [{range_a['min']}, "
                                f"{range_a['max']}] and [{range_b['min']}, "
                                f"{range_b['max']}] overlap on "
                                f"[{overlap['min']}, {overlap['max']}]."
                            ),
                            details={
                                "range_a": range_a,
                                "range_b": range_b,
                                "overlap_range": overlap,
                            },
                        )
                        conflicts.append(conflict)

        logger.debug(
            "_detect_range_overlaps_internal: checked %d columns, "
            "found %d overlaps",
            len(column_groups),
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # Internal: detect_range_contradictions
    # ------------------------------------------------------------------

    def _detect_range_contradictions_internal(
        self,
        rules: List[Dict[str, Any]],
        column_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect range contradictions within column groups.

        A contradiction exists when two RANGE rules on the same column
        have no valid intersection AND both rules are required (severity
        >= HIGH). This makes it impossible for any value to pass both
        validations.

        Args:
            rules: Full list of resolved rules.
            column_groups: Rules grouped by column name.

        Returns:
            List of range contradiction conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for column, col_rules in column_groups.items():
            range_rules = [
                r for r in col_rules
                if self._get_rule_type(r) == "range"
            ]
            if len(range_rules) < 2:
                continue

            for i in range(len(range_rules)):
                for j in range(i + 1, len(range_rules)):
                    rule_a = range_rules[i]
                    rule_b = range_rules[j]
                    pair_key = self._make_pair_key(rule_a, rule_b)

                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    # Both must be required (severity >= HIGH)
                    sev_a = self._get_rule_severity(rule_a)
                    sev_b = self._get_rule_severity(rule_b)
                    if (
                        _SEVERITY_RANK.get(sev_a, 0) < _SEVERITY_RANK.get("high", 3)
                        or _SEVERITY_RANK.get(sev_b, 0) < _SEVERITY_RANK.get("high", 3)
                    ):
                        continue

                    range_a = self._extract_range(rule_a)
                    range_b = self._extract_range(rule_b)

                    if range_a is None or range_b is None:
                        continue

                    overlap = self._compute_range_overlap(range_a, range_b)
                    if overlap is None:
                        # No valid intersection -- contradiction
                        conflict = self._build_conflict(
                            conflict_type=ConflictType.RANGE_CONTRADICTION,
                            severity=ConflictSeverity.HIGH,
                            rule_a_id=self._get_rule_id(rule_a),
                            rule_b_id=self._get_rule_id(rule_b),
                            column=column,
                            description=(
                                f"RANGE rules on column '{column}' are "
                                f"contradictory: [{range_a['min']}, "
                                f"{range_a['max']}] and [{range_b['min']}, "
                                f"{range_b['max']}] have no valid "
                                f"intersection. No value can satisfy both."
                            ),
                            details={
                                "range_a": range_a,
                                "range_b": range_b,
                                "severity_a": sev_a,
                                "severity_b": sev_b,
                            },
                        )
                        conflicts.append(conflict)

        logger.debug(
            "_detect_range_contradictions_internal: found %d contradictions",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # Internal: detect_format_conflicts
    # ------------------------------------------------------------------

    def _detect_format_conflicts_internal(
        self,
        rules: List[Dict[str, Any]],
        column_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect format (regex pattern) conflicts within column groups.

        For each column with multiple FORMAT rules, compares their regex
        patterns heuristically. Two patterns are flagged as conflicting
        when:
        1. They have fundamentally different character class anchors
           (e.g., one requires only digits, the other only letters).
        2. They have conflicting length constraints from anchored
           quantifiers.

        The engine uses static analysis of regex structure rather than
        brute-force string generation to keep detection deterministic
        and efficient.

        Args:
            rules: Full list of resolved rules.
            column_groups: Rules grouped by column name.

        Returns:
            List of format conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for column, col_rules in column_groups.items():
            format_rules = [
                r for r in col_rules
                if self._get_rule_type(r) == "format"
            ]
            if len(format_rules) < 2:
                continue

            for i in range(len(format_rules)):
                for j in range(i + 1, len(format_rules)):
                    rule_a = format_rules[i]
                    rule_b = format_rules[j]
                    pair_key = self._make_pair_key(rule_a, rule_b)

                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    pattern_a = self._get_rule_pattern(rule_a)
                    pattern_b = self._get_rule_pattern(rule_b)

                    if not pattern_a or not pattern_b:
                        continue

                    if self._are_patterns_incompatible(pattern_a, pattern_b):
                        conflict = self._build_conflict(
                            conflict_type=ConflictType.FORMAT_CONFLICT,
                            severity=ConflictSeverity.MEDIUM,
                            rule_a_id=self._get_rule_id(rule_a),
                            rule_b_id=self._get_rule_id(rule_b),
                            column=column,
                            description=(
                                f"FORMAT rules on column '{column}' have "
                                f"incompatible patterns: '{pattern_a}' and "
                                f"'{pattern_b}' cannot both be satisfied "
                                f"by any single string value."
                            ),
                            details={
                                "pattern_a": pattern_a,
                                "pattern_b": pattern_b,
                            },
                        )
                        conflicts.append(conflict)

        logger.debug(
            "_detect_format_conflicts_internal: found %d format conflicts",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # Internal: detect_severity_inconsistencies
    # ------------------------------------------------------------------

    def _detect_severity_inconsistencies_internal(
        self,
        rules: List[Dict[str, Any]],
        column_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect rules with identical conditions but different severities.

        Two rules are condition-identical when they share the same
        rule_type, column, operator, and threshold/pattern values. If
        their severity levels differ, validation enforcement becomes
        inconsistent.

        Args:
            rules: Full list of resolved rules.
            column_groups: Rules grouped by column name.

        Returns:
            List of severity inconsistency conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for column, col_rules in column_groups.items():
            if len(col_rules) < 2:
                continue

            # Group by condition fingerprint within each column
            condition_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for rule in col_rules:
                fingerprint = self._compute_condition_fingerprint(rule)
                condition_groups[fingerprint].append(rule)

            for fingerprint, group in condition_groups.items():
                if len(group) < 2:
                    continue

                # Check for severity differences within the same-condition group
                severities_seen: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for rule in group:
                    sev = self._get_rule_severity(rule)
                    severities_seen[sev].append(rule)

                if len(severities_seen) < 2:
                    continue  # All same severity

                # Generate conflicts for each pair with different severity
                severity_keys = sorted(severities_seen.keys())
                for si in range(len(severity_keys)):
                    for sj in range(si + 1, len(severity_keys)):
                        sev_i = severity_keys[si]
                        sev_j = severity_keys[sj]
                        for rule_a in severities_seen[sev_i]:
                            for rule_b in severities_seen[sev_j]:
                                pair_key = self._make_pair_key(rule_a, rule_b)
                                if pair_key in seen_pairs:
                                    continue
                                seen_pairs.add(pair_key)

                                conflict = self._build_conflict(
                                    conflict_type=ConflictType.SEVERITY_INCONSISTENCY,
                                    severity=ConflictSeverity.MEDIUM,
                                    rule_a_id=self._get_rule_id(rule_a),
                                    rule_b_id=self._get_rule_id(rule_b),
                                    column=column,
                                    description=(
                                        f"Rules on column '{column}' with "
                                        f"identical conditions have different "
                                        f"severities: '{sev_i}' vs '{sev_j}'."
                                    ),
                                    details={
                                        "severity_a": sev_i,
                                        "severity_b": sev_j,
                                        "condition_fingerprint": fingerprint,
                                    },
                                )
                                conflicts.append(conflict)

        logger.debug(
            "_detect_severity_inconsistencies_internal: found %d issues",
            len(conflicts),
        )
        return conflicts

    # ------------------------------------------------------------------
    # Internal: detect_redundancies
    # ------------------------------------------------------------------

    def _detect_redundancies_internal(
        self,
        rules: List[Dict[str, Any]],
        column_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Detect redundant (subset or duplicate) rules within column groups.

        A rule A is redundant relative to rule B when:
        1. Both are RANGE rules on the same column and A's range is
           entirely contained within B's range (subset).
        2. Both rules have identical condition fingerprints (duplicate).

        Args:
            rules: Full list of resolved rules.
            column_groups: Rules grouped by column name.

        Returns:
            List of redundancy conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for column, col_rules in column_groups.items():
            if len(col_rules) < 2:
                continue

            # Check for exact duplicate conditions
            conflicts.extend(
                self._find_exact_duplicates(column, col_rules, seen_pairs)
            )

            # Check for RANGE subset redundancies
            range_rules = [
                r for r in col_rules
                if self._get_rule_type(r) == "range"
            ]
            if len(range_rules) >= 2:
                conflicts.extend(
                    self._find_range_subsets(column, range_rules, seen_pairs)
                )

        logger.debug(
            "_detect_redundancies_internal: found %d redundancies",
            len(conflicts),
        )
        return conflicts

    def _find_exact_duplicates(
        self,
        column: str,
        col_rules: List[Dict[str, Any]],
        seen_pairs: Set[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        """Find rules with identical condition fingerprints.

        Args:
            column: Column name for the group.
            col_rules: All rules targeting this column.
            seen_pairs: Mutable set of already-seen pair keys.

        Returns:
            List of redundancy conflict dicts for exact duplicates.
        """
        conflicts: List[Dict[str, Any]] = []
        condition_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for rule in col_rules:
            fingerprint = self._compute_condition_fingerprint(rule)
            condition_groups[fingerprint].append(rule)

        for fingerprint, group in condition_groups.items():
            if len(group) < 2:
                continue

            # Also need same severity to be a true duplicate
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    rule_a = group[i]
                    rule_b = group[j]

                    if self._get_rule_severity(rule_a) != self._get_rule_severity(rule_b):
                        continue  # Different severity = not a true duplicate

                    pair_key = self._make_pair_key(rule_a, rule_b)
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    conflict = self._build_conflict(
                        conflict_type=ConflictType.REDUNDANCY,
                        severity=ConflictSeverity.LOW,
                        rule_a_id=self._get_rule_id(rule_a),
                        rule_b_id=self._get_rule_id(rule_b),
                        column=column,
                        description=(
                            f"Rules on column '{column}' are exact "
                            f"duplicates with identical conditions and "
                            f"severity."
                        ),
                        details={
                            "redundancy_type": "exact_duplicate",
                            "condition_fingerprint": fingerprint,
                        },
                    )
                    conflicts.append(conflict)

        return conflicts

    def _find_range_subsets(
        self,
        column: str,
        range_rules: List[Dict[str, Any]],
        seen_pairs: Set[Tuple[str, str]],
    ) -> List[Dict[str, Any]]:
        """Find RANGE rules where one range is entirely within another.

        Rule A is a subset of Rule B when:
            B.min <= A.min AND A.max <= B.max

        Args:
            column: Column name for the group.
            range_rules: RANGE rules targeting this column.
            seen_pairs: Mutable set of already-seen pair keys.

        Returns:
            List of redundancy conflict dicts for range subsets.
        """
        conflicts: List[Dict[str, Any]] = []

        for i in range(len(range_rules)):
            for j in range(i + 1, len(range_rules)):
                rule_a = range_rules[i]
                rule_b = range_rules[j]
                pair_key = self._make_pair_key(rule_a, rule_b)

                if pair_key in seen_pairs:
                    continue

                range_a = self._extract_range(rule_a)
                range_b = self._extract_range(rule_b)

                if range_a is None or range_b is None:
                    continue

                subset_info = self._check_range_subset(range_a, range_b)
                if subset_info is not None:
                    seen_pairs.add(pair_key)
                    subset_rule_id = (
                        self._get_rule_id(rule_a)
                        if subset_info["subset"] == "a"
                        else self._get_rule_id(rule_b)
                    )
                    superset_rule_id = (
                        self._get_rule_id(rule_b)
                        if subset_info["subset"] == "a"
                        else self._get_rule_id(rule_a)
                    )

                    conflict = self._build_conflict(
                        conflict_type=ConflictType.REDUNDANCY,
                        severity=ConflictSeverity.LOW,
                        rule_a_id=self._get_rule_id(rule_a),
                        rule_b_id=self._get_rule_id(rule_b),
                        column=column,
                        description=(
                            f"RANGE rule '{subset_rule_id}' on column "
                            f"'{column}' is entirely contained within "
                            f"rule '{superset_rule_id}' and is redundant."
                        ),
                        details={
                            "redundancy_type": "range_subset",
                            "subset_rule_id": subset_rule_id,
                            "superset_rule_id": superset_rule_id,
                            "range_a": range_a,
                            "range_b": range_b,
                        },
                    )
                    conflicts.append(conflict)

        return conflicts

    # ------------------------------------------------------------------
    # Internal: detect_conditional_conflicts
    # ------------------------------------------------------------------

    def _detect_conditional_conflicts_internal(
        self,
        rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect CONDITIONAL rules with contradictory predicates.

        Groups CONDITIONAL rules by their predicate (IF condition) and
        checks whether the consequents (THEN actions) contradict each
        other. For example, IF country="US" THEN currency="USD" vs
        IF country="US" THEN currency="EUR" is a contradiction.

        Args:
            rules: Full list of resolved rules.

        Returns:
            List of conditional conflict dicts.
        """
        conflicts: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        conditional_rules = [
            r for r in rules
            if self._get_rule_type(r) == "conditional"
        ]

        if len(conditional_rules) < 2:
            return conflicts

        # Group by predicate fingerprint
        predicate_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for rule in conditional_rules:
            predicate_fp = self._compute_predicate_fingerprint(rule)
            predicate_groups[predicate_fp].append(rule)

        for pred_fp, group in predicate_groups.items():
            if len(group) < 2:
                continue

            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    rule_a = group[i]
                    rule_b = group[j]
                    pair_key = self._make_pair_key(rule_a, rule_b)

                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)

                    if self._are_consequents_contradictory(rule_a, rule_b):
                        then_a = self._get_rule_consequent(rule_a)
                        then_b = self._get_rule_consequent(rule_b)
                        predicate_desc = self._get_rule_predicate_description(rule_a)

                        conflict = self._build_conflict(
                            conflict_type="conditional_conflict",
                            severity=ConflictSeverity.HIGH,
                            rule_a_id=self._get_rule_id(rule_a),
                            rule_b_id=self._get_rule_id(rule_b),
                            column=self._get_rule_column(rule_a),
                            description=(
                                f"CONDITIONAL rules share predicate "
                                f"'{predicate_desc}' but have "
                                f"contradictory consequents: "
                                f"'{then_a}' vs '{then_b}'."
                            ),
                            details={
                                "predicate_fingerprint": pred_fp,
                                "consequent_a": then_a,
                                "consequent_b": then_b,
                            },
                        )
                        conflicts.append(conflict)

        logger.debug(
            "_detect_conditional_conflicts_internal: found %d conflicts",
            len(conflicts),
        )
        return conflicts

    # ==================================================================
    # INTERNAL: Rule Resolution and Grouping
    # ==================================================================

    def _resolve_rules(
        self,
        rule_ids: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        """Resolve rule IDs to rule dicts from the registry.

        When ``rule_ids`` is ``None``, returns all active rules from
        the registry. Otherwise, fetches each rule by ID.

        Args:
            rule_ids: Optional list of specific rule IDs to fetch.

        Returns:
            List of rule dicts.
        """
        if rule_ids is None:
            return self._get_all_active_rules()

        rules: List[Dict[str, Any]] = []
        for rule_id in rule_ids:
            rule = self._get_rule_by_id(rule_id)
            if rule is not None:
                rules.append(rule)
            else:
                logger.warning(
                    "_resolve_rules: rule_id=%s not found in registry",
                    rule_id,
                )
        return rules

    def _get_all_active_rules(self) -> List[Dict[str, Any]]:
        """Retrieve all active rules from the registry.

        Attempts to call ``registry.list_rules(status="active")`` first.
        Falls back to ``registry.list_rules()`` if the status parameter
        is not supported.

        Returns:
            List of rule dicts from the registry.
        """
        try:
            if hasattr(self._registry, "list_rules"):
                try:
                    rules = self._registry.list_rules(status="active")
                except TypeError:
                    rules = self._registry.list_rules()
                if isinstance(rules, list):
                    return rules
                if isinstance(rules, dict):
                    return rules.get("rules", [])
        except Exception as exc:
            logger.warning(
                "_get_all_active_rules: failed to fetch from registry: %s",
                exc,
            )
        return []

    def _get_rule_by_id(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single rule from the registry by its ID.

        Args:
            rule_id: The rule identifier to fetch.

        Returns:
            The rule dict, or ``None`` if not found.
        """
        try:
            if hasattr(self._registry, "get_rule"):
                return self._registry.get_rule(rule_id)
        except Exception as exc:
            logger.warning(
                "_get_rule_by_id: failed for rule_id=%s: %s",
                rule_id,
                exc,
            )
        return None

    def _get_rule_ids_for_set(self, set_id: str) -> Optional[List[str]]:
        """Retrieve rule IDs belonging to a specific rule set.

        Attempts ``registry.get_rules_by_set(set_id)``. Falls back
        to filtering all rules by set_id if the dedicated method
        is unavailable.

        Args:
            set_id: The rule set identifier.

        Returns:
            List of rule IDs, or ``None`` to indicate all rules.
        """
        try:
            if hasattr(self._registry, "get_rules_by_set"):
                result = self._registry.get_rules_by_set(set_id)
                if isinstance(result, list):
                    return [self._get_rule_id(r) for r in result]
        except Exception as exc:
            logger.warning(
                "_get_rule_ids_for_set: failed for set_id=%s: %s",
                set_id,
                exc,
            )

        # Fallback: filter all rules by set_id attribute
        try:
            all_rules = self._get_all_active_rules()
            return [
                self._get_rule_id(r) for r in all_rules
                if r.get("set_id") == set_id or r.get("rule_set_id") == set_id
            ]
        except Exception:
            return None

    def _group_rules_by_column(
        self,
        rules: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group rules by their target column name.

        Rules without a column attribute are placed in a special
        ``"__global__"`` group.

        Args:
            rules: List of rule dicts to group.

        Returns:
            Dict mapping column name to list of rules targeting that column.
        """
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for rule in rules:
            column = self._get_rule_column(rule)
            groups[column].append(rule)

        logger.debug(
            "_group_rules_by_column: %d rules -> %d column groups",
            len(rules),
            len(groups),
        )
        return dict(groups)

    # ==================================================================
    # INTERNAL: Rule Accessor Helpers
    # ==================================================================

    def _get_rule_id(self, rule: Dict[str, Any]) -> str:
        """Extract the rule ID from a rule dict.

        Supports ``"id"``, ``"rule_id"``, and ``"ruleId"`` key names.

        Args:
            rule: Rule dict.

        Returns:
            Rule ID string, or ``"unknown"`` if no ID is found.
        """
        for key in ("id", "rule_id", "ruleId"):
            value = rule.get(key)
            if value is not None:
                return str(value)
        return "unknown"

    def _get_rule_type(self, rule: Dict[str, Any]) -> str:
        """Extract the rule type from a rule dict.

        Supports ``"type"``, ``"rule_type"``, and ``"ruleType"`` key names.
        Returns lowercase for consistent comparison.

        Args:
            rule: Rule dict.

        Returns:
            Lowercase rule type string, or ``""`` if not found.
        """
        for key in ("type", "rule_type", "ruleType"):
            value = rule.get(key)
            if value is not None:
                return str(value).lower()
        return ""

    def _get_rule_column(self, rule: Dict[str, Any]) -> str:
        """Extract the target column from a rule dict.

        Supports ``"column"``, ``"field"``, ``"field_name"``, and
        ``"fieldName"`` key names.

        Args:
            rule: Rule dict.

        Returns:
            Column name string, or ``"__global__"`` if not found.
        """
        for key in ("column", "field", "field_name", "fieldName"):
            value = rule.get(key)
            if value is not None:
                return str(value)
        return "__global__"

    def _get_rule_severity(self, rule: Dict[str, Any]) -> str:
        """Extract the severity level from a rule dict.

        Args:
            rule: Rule dict.

        Returns:
            Lowercase severity string, or ``"medium"`` as default.
        """
        value = rule.get("severity", "medium")
        return str(value).lower()

    def _get_rule_pattern(self, rule: Dict[str, Any]) -> str:
        """Extract the regex pattern from a FORMAT rule dict.

        Supports ``"pattern"``, ``"regex"``, and ``"format_pattern"``
        key names. Also checks nested ``"config"`` and ``"parameters"``.

        Args:
            rule: Rule dict.

        Returns:
            Pattern string, or ``""`` if not found.
        """
        for key in ("pattern", "regex", "format_pattern"):
            value = rule.get(key)
            if value is not None:
                return str(value)

        # Check nested config/parameters
        for container_key in ("config", "parameters", "params"):
            container = rule.get(container_key)
            if isinstance(container, dict):
                for key in ("pattern", "regex", "format_pattern"):
                    value = container.get(key)
                    if value is not None:
                        return str(value)

        return ""

    def _get_rule_operator(self, rule: Dict[str, Any]) -> str:
        """Extract the operator from a rule dict.

        Args:
            rule: Rule dict.

        Returns:
            Operator string, or ``""`` if not found.
        """
        for key in ("operator", "op"):
            value = rule.get(key)
            if value is not None:
                return str(value).lower()

        config = rule.get("config") or rule.get("parameters") or {}
        if isinstance(config, dict):
            for key in ("operator", "op"):
                value = config.get(key)
                if value is not None:
                    return str(value).lower()

        return ""

    def _get_rule_threshold(self, rule: Dict[str, Any]) -> Optional[Any]:
        """Extract the threshold value from a rule dict.

        Args:
            rule: Rule dict.

        Returns:
            Threshold value, or ``None`` if not found.
        """
        for key in ("threshold", "value"):
            value = rule.get(key)
            if value is not None:
                return value

        config = rule.get("config") or rule.get("parameters") or {}
        if isinstance(config, dict):
            for key in ("threshold", "value"):
                value = config.get(key)
                if value is not None:
                    return value

        return None

    def _get_rule_consequent(self, rule: Dict[str, Any]) -> str:
        """Extract the consequent (THEN clause) from a CONDITIONAL rule.

        Args:
            rule: Rule dict.

        Returns:
            Consequent string, or ``""`` if not found.
        """
        for key in ("consequent", "then", "then_value", "expected_value"):
            value = rule.get(key)
            if value is not None:
                return str(value)

        config = rule.get("config") or rule.get("parameters") or {}
        if isinstance(config, dict):
            for key in ("consequent", "then", "then_value", "expected_value"):
                value = config.get(key)
                if value is not None:
                    return str(value)

        return ""

    def _get_rule_predicate_description(self, rule: Dict[str, Any]) -> str:
        """Extract a human-readable description of the predicate (IF clause).

        Args:
            rule: Rule dict.

        Returns:
            Predicate description string.
        """
        for key in ("predicate", "condition", "if_clause", "when"):
            value = rule.get(key)
            if value is not None:
                if isinstance(value, dict):
                    return json.dumps(value, sort_keys=True, default=str)
                return str(value)

        config = rule.get("config") or rule.get("parameters") or {}
        if isinstance(config, dict):
            for key in ("predicate", "condition", "if_clause", "when"):
                value = config.get(key)
                if value is not None:
                    if isinstance(value, dict):
                        return json.dumps(value, sort_keys=True, default=str)
                    return str(value)

        return ""

    # ==================================================================
    # INTERNAL: Range Analysis
    # ==================================================================

    def _extract_range(
        self,
        rule: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Extract min/max bounds from a RANGE rule dict.

        Supports multiple key naming conventions: ``"min"``/``"max"``,
        ``"min_value"``/``"max_value"``, or nested in ``"config"``/
        ``"parameters"``.

        Args:
            rule: Rule dict expected to define a numeric range.

        Returns:
            Dict with ``"min"`` and ``"max"`` float keys, or ``None``
            if bounds cannot be extracted.
        """
        min_val: Optional[float] = None
        max_val: Optional[float] = None

        # Direct keys
        for min_key in ("min", "min_value", "minimum", "lower_bound"):
            value = rule.get(min_key)
            if value is not None:
                try:
                    min_val = float(value)
                    break
                except (ValueError, TypeError):
                    pass

        for max_key in ("max", "max_value", "maximum", "upper_bound"):
            value = rule.get(max_key)
            if value is not None:
                try:
                    max_val = float(value)
                    break
                except (ValueError, TypeError):
                    pass

        # Check nested config/parameters
        if min_val is None or max_val is None:
            for container_key in ("config", "parameters", "params", "range"):
                container = rule.get(container_key)
                if not isinstance(container, dict):
                    continue

                if min_val is None:
                    for min_key in ("min", "min_value", "minimum", "lower_bound"):
                        value = container.get(min_key)
                        if value is not None:
                            try:
                                min_val = float(value)
                                break
                            except (ValueError, TypeError):
                                pass

                if max_val is None:
                    for max_key in ("max", "max_value", "maximum", "upper_bound"):
                        value = container.get(max_key)
                        if value is not None:
                            try:
                                max_val = float(value)
                                break
                            except (ValueError, TypeError):
                                pass

        if min_val is None or max_val is None:
            return None

        # Ensure min <= max
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        return {"min": min_val, "max": max_val}

    def _compute_range_overlap(
        self,
        range_a: Dict[str, float],
        range_b: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        """Compute the overlapping interval of two numeric ranges.

        Two ranges [a_min, a_max] and [b_min, b_max] overlap when
        max(a_min, b_min) <= min(a_max, b_max).

        Args:
            range_a: Dict with ``"min"`` and ``"max"`` keys.
            range_b: Dict with ``"min"`` and ``"max"`` keys.

        Returns:
            Dict with ``"min"`` and ``"max"`` defining the overlap, or
            ``None`` if there is no overlap.
        """
        overlap_min = max(range_a["min"], range_b["min"])
        overlap_max = min(range_a["max"], range_b["max"])

        if overlap_min <= overlap_max:
            return {"min": overlap_min, "max": overlap_max}
        return None

    def _check_range_subset(
        self,
        range_a: Dict[str, float],
        range_b: Dict[str, float],
    ) -> Optional[Dict[str, str]]:
        """Check whether one range is entirely contained within another.

        Range A is a subset of Range B when:
            B.min <= A.min AND A.max <= B.max

        The reverse (B subset of A) is also checked.

        Args:
            range_a: Dict with ``"min"`` and ``"max"`` keys.
            range_b: Dict with ``"min"`` and ``"max"`` keys.

        Returns:
            Dict with ``"subset"`` key indicating ``"a"`` or ``"b"``,
            or ``None`` if neither is a subset.
        """
        # Check if A is a subset of B
        if range_b["min"] <= range_a["min"] and range_a["max"] <= range_b["max"]:
            # But not if they are identical (that is a duplicate, not subset)
            if range_a["min"] != range_b["min"] or range_a["max"] != range_b["max"]:
                return {"subset": "a"}

        # Check if B is a subset of A
        if range_a["min"] <= range_b["min"] and range_b["max"] <= range_a["max"]:
            if range_a["min"] != range_b["min"] or range_a["max"] != range_b["max"]:
                return {"subset": "b"}

        return None

    # ==================================================================
    # INTERNAL: Format Analysis
    # ==================================================================

    def _are_patterns_incompatible(
        self,
        pattern_a: str,
        pattern_b: str,
    ) -> bool:
        """Determine whether two regex patterns are mutually incompatible.

        Uses heuristic analysis of the pattern structure to detect
        fundamental incompatibilities without brute-force string generation.

        Checks performed:
        1. Both patterns are anchored (^...$) with conflicting character
           classes (e.g., digits-only vs letters-only).
        2. Fixed-length anchored patterns with different lengths.
        3. Patterns that compile but have structurally exclusive
           leading character classes.

        Args:
            pattern_a: First regex pattern string.
            pattern_b: Second regex pattern string.

        Returns:
            ``True`` if the patterns are likely incompatible; ``False``
            if compatibility cannot be determined or patterns may be
            compatible.
        """
        # Validate both patterns compile
        try:
            re.compile(pattern_a)
            re.compile(pattern_b)
        except re.error:
            return False  # Cannot analyse broken patterns

        anchored_a = self._is_fully_anchored(pattern_a)
        anchored_b = self._is_fully_anchored(pattern_b)

        if not (anchored_a and anchored_b):
            return False  # Non-anchored patterns may overlap

        # Extract dominant character class from each pattern
        class_a = self._extract_dominant_char_class(pattern_a)
        class_b = self._extract_dominant_char_class(pattern_b)

        if class_a and class_b and class_a != class_b:
            # Different dominant classes in fully anchored patterns
            if self._are_char_classes_disjoint(class_a, class_b):
                return True

        # Check fixed-length conflicts
        len_a = self._extract_fixed_length(pattern_a)
        len_b = self._extract_fixed_length(pattern_b)

        if len_a is not None and len_b is not None and len_a != len_b:
            return True

        return False

    def _is_fully_anchored(self, pattern: str) -> bool:
        """Check whether a pattern is anchored at both start and end.

        Args:
            pattern: Regex pattern string.

        Returns:
            ``True`` if pattern starts with ``^`` and ends with ``$``.
        """
        stripped = pattern.strip()
        return stripped.startswith("^") and stripped.endswith("$")

    def _extract_dominant_char_class(self, pattern: str) -> str:
        """Extract the dominant character class from an anchored pattern.

        Returns a simplified class label:
        - ``"digits"`` for patterns like ``\\d+``, ``[0-9]+``
        - ``"alpha_upper"`` for patterns like ``[A-Z]+``
        - ``"alpha_lower"`` for patterns like ``[a-z]+``
        - ``"alpha"`` for patterns like ``[A-Za-z]+``
        - ``""`` if no dominant class can be determined.

        Args:
            pattern: Regex pattern string (expected to be anchored).

        Returns:
            Simplified class label string.
        """
        # Remove anchors for analysis
        inner = pattern.strip()
        if inner.startswith("^"):
            inner = inner[1:]
        if inner.endswith("$"):
            inner = inner[:-1]

        if not inner:
            return ""

        # Check for digits-only pattern
        digits_patterns = (
            r"\d", "[0-9]", r"\d+", "[0-9]+",
            r"\d{", "[0-9]{",
        )
        if any(inner.startswith(dp) for dp in digits_patterns):
            # Check the whole pattern is digits
            non_digit = re.sub(r"\\d|\\d\+|\\d\{[^}]*\}|\[0-9\]|\[0-9\]\+|\[0-9\]\{[^}]*\}", "", inner)
            if not non_digit or all(c in "{}+*?|,()" for c in non_digit):
                return "digits"

        # Check for uppercase letters only
        if inner.startswith("[A-Z]"):
            return "alpha_upper"

        # Check for lowercase letters only
        if inner.startswith("[a-z]"):
            return "alpha_lower"

        # Check for mixed alpha
        if inner.startswith("[A-Za-z]") or inner.startswith("[a-zA-Z]"):
            return "alpha"

        return ""

    def _extract_fixed_length(self, pattern: str) -> Optional[int]:
        """Extract the fixed length from an anchored, fixed-length pattern.

        Recognises patterns like ``^\\d{3}$`` (length 3), ``^[A-Z]{5}$``
        (length 5), and ``^.{10}$`` (length 10).

        Args:
            pattern: Regex pattern string.

        Returns:
            Fixed length as int, or ``None`` if the pattern does not
            specify a fixed length.
        """
        inner = pattern.strip()
        if inner.startswith("^"):
            inner = inner[1:]
        if inner.endswith("$"):
            inner = inner[:-1]

        # Match patterns like \d{3}, [A-Z]{5}, .{10}
        match = re.match(r"^(?:\\[dDwWsS]|\[\^?[^\]]+\]|\.)\{(\d+)\}$", inner)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass

        # Match literal characters (no quantifiers)
        if re.match(r"^[a-zA-Z0-9]+$", inner):
            return len(inner)

        return None

    def _are_char_classes_disjoint(
        self,
        class_a: str,
        class_b: str,
    ) -> bool:
        """Check whether two simplified character classes are disjoint.

        Args:
            class_a: First class label (e.g., ``"digits"``).
            class_b: Second class label (e.g., ``"alpha_upper"``).

        Returns:
            ``True`` if no character can belong to both classes.
        """
        disjoint_pairs: Set[Tuple[str, str]] = {
            ("digits", "alpha_upper"),
            ("digits", "alpha_lower"),
            ("digits", "alpha"),
            ("alpha_upper", "alpha_lower"),
        }

        pair = tuple(sorted([class_a, class_b]))
        return pair in disjoint_pairs

    # ==================================================================
    # INTERNAL: Condition Fingerprinting
    # ==================================================================

    def _compute_condition_fingerprint(self, rule: Dict[str, Any]) -> str:
        """Compute a deterministic fingerprint for a rule's conditions.

        The fingerprint encodes rule_type, column, operator, and
        threshold/pattern so that rules with identical conditions
        produce the same fingerprint regardless of severity or metadata.

        Args:
            rule: Rule dict.

        Returns:
            SHA-256 hex digest of the condition components.
        """
        components = {
            "type": self._get_rule_type(rule),
            "column": self._get_rule_column(rule),
            "operator": self._get_rule_operator(rule),
            "threshold": str(self._get_rule_threshold(rule)),
            "pattern": self._get_rule_pattern(rule),
        }

        # Add range bounds if present
        range_bounds = self._extract_range(rule)
        if range_bounds is not None:
            components["range_min"] = str(range_bounds["min"])
            components["range_max"] = str(range_bounds["max"])

        serialized = json.dumps(components, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _compute_predicate_fingerprint(self, rule: Dict[str, Any]) -> str:
        """Compute a fingerprint for the predicate (IF clause) of a CONDITIONAL rule.

        Only the predicate condition is included, not the consequent,
        so that rules with the same predicate but different consequents
        can be grouped and compared.

        Args:
            rule: CONDITIONAL rule dict.

        Returns:
            SHA-256 hex digest of the predicate components (truncated to 16 chars).
        """
        predicate_data: Dict[str, Any] = {}

        for key in ("predicate", "condition", "if_clause", "when"):
            value = rule.get(key)
            if value is not None:
                predicate_data["predicate"] = value
                break

        config = rule.get("config") or rule.get("parameters") or {}
        if isinstance(config, dict) and not predicate_data:
            for key in ("predicate", "condition", "if_clause", "when"):
                value = config.get(key)
                if value is not None:
                    predicate_data["predicate"] = value
                    break

        # Include the predicate column for extra specificity
        predicate_data["column"] = self._get_rule_column(rule)

        serialized = json.dumps(predicate_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _are_consequents_contradictory(
        self,
        rule_a: Dict[str, Any],
        rule_b: Dict[str, Any],
    ) -> bool:
        """Check whether two CONDITIONAL rules have contradictory consequents.

        Two consequents are contradictory when they target the same
        field/column but specify different required values. For example,
        ``THEN currency="USD"`` vs ``THEN currency="EUR"``.

        Args:
            rule_a: First CONDITIONAL rule dict.
            rule_b: Second CONDITIONAL rule dict.

        Returns:
            ``True`` if the consequents are contradictory.
        """
        consequent_a = self._get_rule_consequent(rule_a)
        consequent_b = self._get_rule_consequent(rule_b)

        if not consequent_a or not consequent_b:
            return False

        # Different consequent values on the same predicate
        return consequent_a != consequent_b

    # ==================================================================
    # INTERNAL: Conflict Building
    # ==================================================================

    def _build_conflict(
        self,
        conflict_type: str,
        severity: str,
        rule_a_id: str,
        rule_b_id: str,
        column: str,
        description: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a structured conflict record.

        Args:
            conflict_type: Conflict type string.
            severity: Severity level string.
            rule_a_id: ID of the first rule in the conflict pair.
            rule_b_id: ID of the second rule in the conflict pair.
            column: Column where the conflict was detected.
            description: Human-readable conflict explanation.
            details: Optional additional details dict.

        Returns:
            Complete conflict record dict with provenance hash.
        """
        conflict_id = str(uuid.uuid4())
        detected_at = _utcnow().isoformat()

        conflict: Dict[str, Any] = {
            "conflict_id": conflict_id,
            "conflict_type": conflict_type if isinstance(conflict_type, str) else conflict_type.value,
            "severity": severity if isinstance(severity, str) else severity.value,
            "rule_a_id": rule_a_id,
            "rule_b_id": rule_b_id,
            "column": column,
            "description": description,
            "detected_at": detected_at,
            "status": "open",
        }

        if details is not None:
            conflict.update(details)

        provenance_hash = self._compute_hash({
            "conflict_id": conflict_id,
            "conflict_type": conflict.get("conflict_type"),
            "rule_a_id": rule_a_id,
            "rule_b_id": rule_b_id,
            "column": column,
            "detected_at": detected_at,
        })
        conflict["provenance_hash"] = provenance_hash

        self._provenance.record(
            entity_type="conflict",
            entity_id=conflict_id,
            action="conflict_detected",
            data={
                "conflict_type": conflict.get("conflict_type"),
                "severity": conflict.get("severity"),
                "provenance_hash": provenance_hash,
            },
        )

        return conflict

    def _make_pair_key(
        self,
        rule_a: Dict[str, Any],
        rule_b: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Create a canonical pair key for deduplication.

        The pair key is a tuple of sorted rule IDs so that (A, B) and
        (B, A) produce the same key.

        Args:
            rule_a: First rule dict.
            rule_b: Second rule dict.

        Returns:
            Sorted tuple of rule ID strings.
        """
        id_a = self._get_rule_id(rule_a)
        id_b = self._get_rule_id(rule_b)
        return tuple(sorted([id_a, id_b]))  # type: ignore[return-value]

    # ==================================================================
    # INTERNAL: Report Helpers
    # ==================================================================

    def _compute_severity_distribution(
        self,
        conflicts: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Compute a severity distribution for a list of conflicts.

        Args:
            conflicts: List of conflict dicts.

        Returns:
            Dict mapping severity string to count.
        """
        dist: Dict[str, int] = defaultdict(int)
        for conflict in conflicts:
            sev = conflict.get("severity", "unknown")
            dist[sev] += 1
        return dict(dist)

    def _compute_type_distribution(
        self,
        conflicts: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """Compute a conflict type distribution for a list of conflicts.

        Args:
            conflicts: List of conflict dicts.

        Returns:
            Dict mapping conflict type string to count.
        """
        dist: Dict[str, int] = defaultdict(int)
        for conflict in conflicts:
            ct = conflict.get("conflict_type", "unknown")
            dist[ct] += 1
        return dict(dist)

    def _generate_report_recommendations(
        self,
        conflicts: List[Dict[str, Any]],
        severity_dist: Dict[str, int],
    ) -> List[str]:
        """Generate high-level recommendations based on detected conflicts.

        Args:
            conflicts: List of conflict dicts.
            severity_dist: Severity distribution dict.

        Returns:
            List of human-readable recommendation strings.
        """
        recommendations: List[str] = []

        if not conflicts:
            recommendations.append(
                "No conflicts detected. The rule set is internally consistent."
            )
            return recommendations

        high_count = severity_dist.get("high", 0)
        medium_count = severity_dist.get("medium", 0)
        low_count = severity_dist.get("low", 0)

        if high_count > 0:
            recommendations.append(
                f"CRITICAL: {high_count} high-severity conflict(s) detected. "
                f"These include contradictions where no valid value can satisfy "
                f"all rules. Resolve these immediately before deploying the "
                f"rule set."
            )

        if medium_count > 0:
            recommendations.append(
                f"WARNING: {medium_count} medium-severity conflict(s) detected. "
                f"These include overlapping ranges, format conflicts, and "
                f"severity inconsistencies that may cause ambiguous validation "
                f"results. Review and resolve before production use."
            )

        if low_count > 0:
            recommendations.append(
                f"INFO: {low_count} low-severity conflict(s) detected. "
                f"These are redundant rules that waste evaluation time. "
                f"Consider consolidating or removing duplicate rules."
            )

        # Type-specific recommendations
        type_dist = self._compute_type_distribution(conflicts)

        if type_dist.get(ConflictType.RANGE_CONTRADICTION, 0) > 0:
            recommendations.append(
                "Range contradictions exist. Review RANGE rules on the same "
                "column to ensure at least one valid value can pass all rules."
            )

        if type_dist.get(ConflictType.SEVERITY_INCONSISTENCY, 0) > 0:
            recommendations.append(
                "Severity inconsistencies detected. Standardize severity "
                "levels for rules with identical conditions to ensure "
                "consistent enforcement."
            )

        if type_dist.get(ConflictType.REDUNDANCY, 0) > 0:
            recommendations.append(
                "Redundant rules found. Consider consolidating subset rules "
                "into a single broader rule, or removing exact duplicates."
            )

        return recommendations

    def _build_suggestions_for_type(
        self,
        conflict_type: str,
        conflict: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build resolution suggestions based on conflict type.

        Args:
            conflict_type: Conflict type string.
            conflict: The conflict dict.

        Returns:
            Ordered list of suggestion dicts.
        """
        suggestions: List[Dict[str, Any]] = []

        if conflict_type == ConflictType.RANGE_OVERLAP:
            suggestions.extend([
                {
                    "strategy": "merge",
                    "rationale": (
                        "Merge the overlapping ranges into a single rule "
                        "that covers the union of both ranges, eliminating "
                        "ambiguity."
                    ),
                    "confidence": 0.85,
                    "priority": 1,
                },
                {
                    "strategy": "keep_a",
                    "rationale": (
                        "Keep the first rule and adjust or remove the "
                        "second rule to eliminate overlap."
                    ),
                    "confidence": 0.70,
                    "priority": 2,
                },
                {
                    "strategy": "keep_b",
                    "rationale": (
                        "Keep the second rule and adjust or remove the "
                        "first rule to eliminate overlap."
                    ),
                    "confidence": 0.70,
                    "priority": 3,
                },
                {
                    "strategy": "ignore",
                    "rationale": (
                        "Acknowledge the overlap but take no action. This "
                        "is acceptable if the overlapping range is "
                        "intentional."
                    ),
                    "confidence": 0.40,
                    "priority": 4,
                },
            ])

        elif conflict_type == ConflictType.RANGE_CONTRADICTION:
            suggestions.extend([
                {
                    "strategy": "keep_a",
                    "rationale": (
                        "Keep rule A and remove rule B. This resolves the "
                        "contradiction by accepting rule A's range as "
                        "authoritative."
                    ),
                    "confidence": 0.75,
                    "priority": 1,
                },
                {
                    "strategy": "keep_b",
                    "rationale": (
                        "Keep rule B and remove rule A. This resolves the "
                        "contradiction by accepting rule B's range as "
                        "authoritative."
                    ),
                    "confidence": 0.75,
                    "priority": 2,
                },
                {
                    "strategy": "merge",
                    "rationale": (
                        "Create a new rule with a range that covers the "
                        "union of both ranges, if the intent was to "
                        "accept values in either range."
                    ),
                    "confidence": 0.50,
                    "priority": 3,
                },
            ])

        elif conflict_type == ConflictType.FORMAT_CONFLICT:
            suggestions.extend([
                {
                    "strategy": "keep_a",
                    "rationale": (
                        "Keep pattern A as the canonical format rule and "
                        "remove pattern B."
                    ),
                    "confidence": 0.70,
                    "priority": 1,
                },
                {
                    "strategy": "keep_b",
                    "rationale": (
                        "Keep pattern B as the canonical format rule and "
                        "remove pattern A."
                    ),
                    "confidence": 0.70,
                    "priority": 2,
                },
                {
                    "strategy": "merge",
                    "rationale": (
                        "Combine both patterns using alternation "
                        "(pattern_a|pattern_b) if the column should "
                        "accept either format."
                    ),
                    "confidence": 0.60,
                    "priority": 3,
                },
            ])

        elif conflict_type == ConflictType.SEVERITY_INCONSISTENCY:
            sev_a = conflict.get("severity_a", "unknown")
            sev_b = conflict.get("severity_b", "unknown")
            higher_sev = sev_a if _SEVERITY_RANK.get(sev_a, 0) > _SEVERITY_RANK.get(sev_b, 0) else sev_b

            suggestions.extend([
                {
                    "strategy": "keep_a",
                    "rationale": (
                        f"Standardize both rules to severity '{sev_a}' "
                        f"by updating rule B."
                    ),
                    "confidence": 0.65,
                    "priority": 2,
                },
                {
                    "strategy": "keep_b",
                    "rationale": (
                        f"Standardize both rules to severity '{sev_b}' "
                        f"by updating rule A."
                    ),
                    "confidence": 0.65,
                    "priority": 2,
                },
                {
                    "strategy": "merge",
                    "rationale": (
                        f"Merge into a single rule using the higher "
                        f"severity '{higher_sev}' for consistent "
                        f"enforcement."
                    ),
                    "confidence": 0.80,
                    "priority": 1,
                },
            ])

        elif conflict_type == ConflictType.REDUNDANCY:
            redundancy_type = conflict.get("redundancy_type", "")
            if redundancy_type == "exact_duplicate":
                suggestions.extend([
                    {
                        "strategy": "keep_a",
                        "rationale": (
                            "Remove the duplicate rule B. Both rules are "
                            "identical; keeping A is sufficient."
                        ),
                        "confidence": 0.95,
                        "priority": 1,
                    },
                    {
                        "strategy": "keep_b",
                        "rationale": (
                            "Remove the duplicate rule A. Both rules are "
                            "identical; keeping B is sufficient."
                        ),
                        "confidence": 0.95,
                        "priority": 2,
                    },
                ])
            else:
                # Range subset
                subset_id = conflict.get("subset_rule_id", "")
                superset_id = conflict.get("superset_rule_id", "")
                keep_strategy = (
                    "keep_a"
                    if superset_id == conflict.get("rule_a_id")
                    else "keep_b"
                )
                suggestions.extend([
                    {
                        "strategy": keep_strategy,
                        "rationale": (
                            f"Remove the subset rule '{subset_id}' and "
                            f"keep the broader rule '{superset_id}' which "
                            f"already covers all values."
                        ),
                        "confidence": 0.90,
                        "priority": 1,
                    },
                    {
                        "strategy": "ignore",
                        "rationale": (
                            "Keep both rules if the subset rule enforces "
                            "a stricter severity on a narrower range "
                            "intentionally."
                        ),
                        "confidence": 0.50,
                        "priority": 2,
                    },
                ])

        else:
            # Generic suggestions for unknown conflict types
            suggestions.extend([
                {
                    "strategy": "keep_a",
                    "rationale": "Keep rule A and review rule B for removal.",
                    "confidence": 0.50,
                    "priority": 1,
                },
                {
                    "strategy": "keep_b",
                    "rationale": "Keep rule B and review rule A for removal.",
                    "confidence": 0.50,
                    "priority": 2,
                },
                {
                    "strategy": "ignore",
                    "rationale": "Acknowledge the conflict without action.",
                    "confidence": 0.30,
                    "priority": 3,
                },
            ])

        # Sort by priority
        suggestions.sort(key=lambda s: s["priority"])
        return suggestions

    # ==================================================================
    # INTERNAL: Hashing and Statistics
    # ==================================================================

    def _compute_hash(self, data: Any) -> str:
        """Compute a SHA-256 hash for arbitrary data.

        Args:
            data: Any JSON-serializable object.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _compute_report_hash(self, report_data: Dict[str, Any]) -> str:
        """Compute a SHA-256 provenance hash for a complete conflict report.

        Excludes the ``provenance_hash`` field itself (which has not yet
        been set) to avoid circular dependency.

        Args:
            report_data: The full report dict (before provenance_hash is set).

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        # Create a copy without provenance_hash to avoid circular hashing
        hashable = {k: v for k, v in report_data.items() if k != "provenance_hash"}
        return self._compute_hash(hashable)

    def _initial_stats(self) -> Dict[str, Any]:
        """Create the initial statistics dict.

        Returns:
            Dict with all statistics keys initialized to zero.
        """
        return {
            "total_detections": 0,
            "total_conflicts_found": 0,
            "total_rules_analyzed": 0,
            "total_resolutions": 0,
            "conflicts_high": 0,
            "conflicts_medium": 0,
            "conflicts_low": 0,
        }


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------

__all__ = [
    "ConflictDetectorEngine",
]
