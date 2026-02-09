# -*- coding: utf-8 -*-
"""
Category Rule Engine - AGENT-DATA-009: Spend Data Categorizer
===============================================================

Custom categorisation rule engine supporting six match types with
priority-based evaluation. Enables organisations to define, import,
export, and evaluate custom rules for spend category assignment.

Supports:
    - Six match types: EXACT, CONTAINS, REGEX, FUZZY, STARTS_WITH, ENDS_WITH
    - Priority-based evaluation (highest priority wins)
    - Rule CRUD (create, read, update, delete)
    - Batch rule evaluation against spend records
    - Rule effectiveness tracking (match count statistics)
    - Bulk import and export of rules
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all mutations

Zero-Hallucination Guarantees:
    - All rule evaluation is deterministic (string matching / regex)
    - Fuzzy matching uses deterministic LCS-based similarity
    - No LLM or ML model in rule evaluation path
    - Priority ordering is deterministic (highest wins, stable sort)
    - SHA-256 provenance hashes for audit trails

Example:
    >>> from greenlang.spend_categorizer.category_rule import CategoryRuleEngine
    >>> engine = CategoryRuleEngine()
    >>> rule = engine.create_rule(
    ...     name="Travel Agency Rule",
    ...     match_type="CONTAINS",
    ...     pattern="travel",
    ...     target_category="business_travel",
    ... )
    >>> result = engine.evaluate_rules({"description": "Corporate travel booking"})
    >>> print(result)  # ("business_travel", 0.92, rule.rule_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
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
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "MatchType",
    "CategoryRule",
    "CategoryRuleEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "rule") -> str:
    """Generate a unique identifier with a prefix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Match types
# ---------------------------------------------------------------------------


class MatchType:
    """Supported match types for category rules.

    Constants:
        EXACT: Exact string match (case-insensitive).
        CONTAINS: Substring match (case-insensitive).
        REGEX: Regular expression match.
        FUZZY: Fuzzy similarity match (LCS-based, threshold 0.75).
        STARTS_WITH: Prefix match (case-insensitive).
        ENDS_WITH: Suffix match (case-insensitive).
    """

    EXACT = "EXACT"
    CONTAINS = "CONTAINS"
    REGEX = "REGEX"
    FUZZY = "FUZZY"
    STARTS_WITH = "STARTS_WITH"
    ENDS_WITH = "ENDS_WITH"

    ALL = [EXACT, CONTAINS, REGEX, FUZZY, STARTS_WITH, ENDS_WITH]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class CategoryRule(BaseModel):
    """A custom categorisation rule."""

    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    match_type: str = Field(..., description="Match type (EXACT, CONTAINS, REGEX, FUZZY, STARTS_WITH, ENDS_WITH)")
    pattern: str = Field(..., description="Pattern to match against record text")
    target_category: str = Field(..., description="Category to assign when rule matches")
    target_scope3: Optional[int] = Field(None, ge=0, le=15, description="Scope 3 category override")
    match_field: str = Field(default="description", description="Record field to match against")
    priority: int = Field(default=100, ge=1, le=10000, description="Rule priority (higher = evaluated first)")
    confidence: float = Field(default=0.90, ge=0.0, le=1.0, description="Confidence score when rule matches")
    active: bool = Field(default=True, description="Whether rule is active")
    match_count: int = Field(default=0, ge=0, description="Number of times rule has matched")
    fuzzy_threshold: float = Field(default=0.75, ge=0.0, le=1.0, description="Threshold for fuzzy matching")
    created_at: str = Field(default="", description="Creation timestamp ISO")
    updated_at: str = Field(default="", description="Last update timestamp ISO")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# CategoryRuleEngine
# ---------------------------------------------------------------------------


class CategoryRuleEngine:
    """Custom categorisation rule engine.

    Manages a priority-ordered set of rules for spend category
    assignment. Rules are evaluated in priority order (highest first)
    and the first matching rule wins.

    Supports six match types and tracks match counts for
    effectiveness reporting.

    Attributes:
        _config: Configuration dictionary.
        _rules: In-memory rule storage keyed by rule_id.
        _lock: Threading lock for thread-safe mutations.
        _stats: Cumulative evaluation statistics.

    Example:
        >>> engine = CategoryRuleEngine()
        >>> rule = engine.create_rule("fuel", "CONTAINS", "diesel", "fuel_energy")
        >>> cat, conf, rid = engine.evaluate_rules({"description": "diesel delivery"})
        >>> assert cat == "fuel_energy"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CategoryRuleEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``default_priority``: int (default 100)
                - ``default_confidence``: float (default 0.90)
                - ``fuzzy_threshold``: float (default 0.75)
        """
        self._config = config or {}
        self._default_priority: int = self._config.get("default_priority", 100)
        self._default_confidence: float = self._config.get("default_confidence", 0.90)
        self._fuzzy_threshold: float = self._config.get("fuzzy_threshold", 0.75)
        self._rules: Dict[str, CategoryRule] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, Any] = {
            "rules_created": 0,
            "rules_deleted": 0,
            "evaluations_performed": 0,
            "matches_found": 0,
            "no_match_count": 0,
            "by_match_type": {},
            "errors": 0,
        }
        logger.info(
            "CategoryRuleEngine initialised: default_priority=%d, "
            "default_confidence=%.2f, fuzzy_threshold=%.2f",
            self._default_priority,
            self._default_confidence,
            self._fuzzy_threshold,
        )

    # ------------------------------------------------------------------
    # Public API - CRUD
    # ------------------------------------------------------------------

    def create_rule(
        self,
        name: str,
        match_type: str,
        pattern: str,
        target_category: str,
        priority: Optional[int] = None,
        confidence: Optional[float] = None,
        match_field: str = "description",
        target_scope3: Optional[int] = None,
        fuzzy_threshold: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CategoryRule:
        """Create a new categorisation rule.

        Args:
            name: Human-readable rule name.
            match_type: One of EXACT, CONTAINS, REGEX, FUZZY,
                STARTS_WITH, ENDS_WITH.
            pattern: Pattern string to match.
            target_category: Category to assign on match.
            priority: Evaluation priority (higher = first). Defaults
                to default_priority.
            confidence: Confidence score on match. Defaults to
                default_confidence.
            match_field: Record field to match against (default "description").
            target_scope3: Optional Scope 3 category override (1-15).
            fuzzy_threshold: Similarity threshold for FUZZY match type.
            metadata: Optional additional metadata.

        Returns:
            Created CategoryRule.

        Raises:
            ValueError: If match_type is not valid or pattern is empty.
        """
        match_type_upper = match_type.upper().strip()
        if match_type_upper not in MatchType.ALL:
            raise ValueError(
                f"Invalid match_type '{match_type}'. "
                f"Must be one of: {MatchType.ALL}"
            )
        if not pattern:
            raise ValueError("Pattern must not be empty")

        # Validate regex patterns
        if match_type_upper == MatchType.REGEX:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"Invalid regex pattern '{pattern}': {exc}"
                ) from exc

        rid = _generate_id("rule")
        now_iso = _utcnow().isoformat()

        provenance_hash = self._compute_provenance(
            rid, name, match_type_upper, pattern, target_category, now_iso,
        )

        rule = CategoryRule(
            rule_id=rid,
            name=name,
            match_type=match_type_upper,
            pattern=pattern,
            target_category=target_category,
            target_scope3=target_scope3,
            match_field=match_field,
            priority=priority or self._default_priority,
            confidence=confidence or self._default_confidence,
            active=True,
            match_count=0,
            fuzzy_threshold=fuzzy_threshold or self._fuzzy_threshold,
            created_at=now_iso,
            updated_at=now_iso,
            provenance_hash=provenance_hash,
            metadata=metadata or {},
        )

        with self._lock:
            self._rules[rid] = rule
            self._stats["rules_created"] += 1

        logger.info(
            "Created rule %s: name='%s', type=%s, pattern='%s', "
            "target='%s', priority=%d",
            rid, name, match_type_upper, pattern,
            target_category, rule.priority,
        )
        return rule

    def get_rule(self, rule_id: str) -> Optional[CategoryRule]:
        """Retrieve a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            CategoryRule or None if not found.
        """
        return self._rules.get(rule_id)

    def list_rules(
        self,
        priority: Optional[int] = None,
        active_only: bool = True,
        limit: int = 50,
    ) -> List[CategoryRule]:
        """List rules with optional filtering.

        Args:
            priority: Optional minimum priority filter.
            active_only: If True, only return active rules.
            limit: Maximum results.

        Returns:
            List of CategoryRule objects sorted by priority descending.
        """
        rules = list(self._rules.values())

        if active_only:
            rules = [r for r in rules if r.active]

        if priority is not None:
            rules = [r for r in rules if r.priority >= priority]

        # Sort by priority descending (highest first)
        rules.sort(key=lambda r: r.priority, reverse=True)

        return rules[:limit]

    def update_rule(self, rule_id: str, **kwargs: Any) -> CategoryRule:
        """Update an existing rule.

        Args:
            rule_id: Rule identifier.
            **kwargs: Fields to update (name, pattern, match_type,
                target_category, priority, confidence, active,
                match_field, target_scope3, fuzzy_threshold, metadata).

        Returns:
            Updated CategoryRule.

        Raises:
            ValueError: If rule not found or invalid field values.
        """
        rule = self._rules.get(rule_id)
        if rule is None:
            raise ValueError(f"Rule not found: {rule_id}")

        now_iso = _utcnow().isoformat()
        update_data = rule.model_dump()

        # Apply updates
        allowed_fields = {
            "name", "pattern", "match_type", "target_category",
            "priority", "confidence", "active", "match_field",
            "target_scope3", "fuzzy_threshold", "metadata",
        }
        for key, value in kwargs.items():
            if key in allowed_fields:
                if key == "match_type":
                    value = value.upper().strip()
                    if value not in MatchType.ALL:
                        raise ValueError(f"Invalid match_type: {value}")
                update_data[key] = value

        update_data["updated_at"] = now_iso
        update_data["provenance_hash"] = self._compute_provenance(
            rule_id, update_data["name"], update_data["match_type"],
            update_data["pattern"], update_data["target_category"], now_iso,
        )

        updated_rule = CategoryRule(**update_data)

        with self._lock:
            self._rules[rule_id] = updated_rule

        logger.info("Updated rule %s", rule_id)
        return updated_rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                self._stats["rules_deleted"] += 1
                logger.info("Deleted rule %s", rule_id)
                return True
        return False

    # ------------------------------------------------------------------
    # Public API - Evaluation
    # ------------------------------------------------------------------

    def evaluate_rules(
        self,
        record: Dict[str, Any],
    ) -> Tuple[Optional[str], float, Optional[str]]:
        """Evaluate all active rules against a spend record.

        Rules are evaluated in priority order (highest first).
        The first matching rule wins.

        Args:
            record: Spend record dict.

        Returns:
            Tuple of (target_category, confidence, rule_id).
            Returns (None, 0.0, None) if no rules match.
        """
        start = time.monotonic()

        # Get active rules sorted by priority descending
        rules = self.list_rules(active_only=True, limit=10000)

        for rule in rules:
            if self._evaluate_single_rule(rule, record):
                # Match found
                with self._lock:
                    # Increment match count
                    stored = self._rules.get(rule.rule_id)
                    if stored:
                        update_data = stored.model_dump()
                        update_data["match_count"] = stored.match_count + 1
                        self._rules[rule.rule_id] = CategoryRule(**update_data)
                    self._stats["evaluations_performed"] += 1
                    self._stats["matches_found"] += 1
                    mt_counts = self._stats["by_match_type"]
                    mt_counts[rule.match_type] = mt_counts.get(rule.match_type, 0) + 1

                elapsed = (time.monotonic() - start) * 1000
                logger.debug(
                    "Rule %s matched: target=%s, conf=%.2f (%.1f ms)",
                    rule.rule_id, rule.target_category,
                    rule.confidence, elapsed,
                )
                return rule.target_category, rule.confidence, rule.rule_id

        # No match
        with self._lock:
            self._stats["evaluations_performed"] += 1
            self._stats["no_match_count"] += 1

        return None, 0.0, None

    def apply_rules(
        self,
        records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply rules to a batch of records.

        For each record, evaluates all rules and adds the resulting
        ``rule_category``, ``rule_confidence``, and ``rule_id`` fields.

        Args:
            records: List of spend record dicts.

        Returns:
            The input records with rule results added.
        """
        start = time.monotonic()
        matched = 0

        for rec in records:
            cat, conf, rid = self.evaluate_rules(rec)
            rec["rule_category"] = cat
            rec["rule_confidence"] = conf
            rec["rule_id"] = rid
            if cat is not None:
                matched += 1

        elapsed = (time.monotonic() - start) * 1000
        logger.info(
            "Applied rules to %d records: %d matched (%.1f ms)",
            len(records), matched, elapsed,
        )
        return records

    # ------------------------------------------------------------------
    # Public API - Import / Export
    # ------------------------------------------------------------------

    def import_rules(self, rules_data: List[Dict[str, Any]]) -> int:
        """Bulk import rules from a list of dicts.

        Each dict should have at least ``name``, ``match_type``,
        ``pattern``, and ``target_category``.

        Args:
            rules_data: List of rule definition dicts.

        Returns:
            Number of rules successfully imported.
        """
        imported = 0
        for rule_def in rules_data:
            try:
                self.create_rule(
                    name=str(rule_def.get("name", "")),
                    match_type=str(rule_def.get("match_type", "CONTAINS")),
                    pattern=str(rule_def.get("pattern", "")),
                    target_category=str(rule_def.get("target_category", "")),
                    priority=rule_def.get("priority"),
                    confidence=rule_def.get("confidence"),
                    match_field=str(rule_def.get("match_field", "description")),
                    target_scope3=rule_def.get("target_scope3"),
                    fuzzy_threshold=rule_def.get("fuzzy_threshold"),
                    metadata=rule_def.get("metadata"),
                )
                imported += 1
            except (ValueError, TypeError) as exc:
                logger.warning("Failed to import rule: %s", exc)
                with self._lock:
                    self._stats["errors"] += 1

        logger.info("Imported %d of %d rules", imported, len(rules_data))
        return imported

    def export_rules(self) -> List[Dict[str, Any]]:
        """Export all rules as a list of dicts.

        Returns:
            List of rule dicts suitable for import.
        """
        rules = list(self._rules.values())
        rules.sort(key=lambda r: r.priority, reverse=True)
        return [r.model_dump() for r in rules]

    # ------------------------------------------------------------------
    # Public API - Effectiveness
    # ------------------------------------------------------------------

    def get_effectiveness(
        self,
        rule_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get rule match effectiveness statistics.

        Args:
            rule_id: Optional specific rule ID. If None, returns
                aggregate statistics for all rules.

        Returns:
            Dict with match counts, rates, and rankings.
        """
        if rule_id:
            rule = self._rules.get(rule_id)
            if rule is None:
                return {"error": f"Rule not found: {rule_id}"}
            return {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "match_type": rule.match_type,
                "pattern": rule.pattern,
                "target_category": rule.target_category,
                "match_count": rule.match_count,
                "active": rule.active,
                "priority": rule.priority,
            }

        # Aggregate effectiveness
        rules = list(self._rules.values())
        total_matches = sum(r.match_count for r in rules)
        active_count = sum(1 for r in rules if r.active)
        inactive_count = len(rules) - active_count
        never_matched = sum(1 for r in rules if r.match_count == 0 and r.active)

        # Top matching rules
        top_rules = sorted(rules, key=lambda r: r.match_count, reverse=True)[:10]
        top_list = [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "match_count": r.match_count,
                "target_category": r.target_category,
            }
            for r in top_rules
        ]

        # By match type
        by_type: Dict[str, int] = {}
        for r in rules:
            by_type[r.match_type] = by_type.get(r.match_type, 0) + r.match_count

        return {
            "total_rules": len(rules),
            "active_rules": active_count,
            "inactive_rules": inactive_count,
            "total_matches": total_matches,
            "never_matched_rules": never_matched,
            "by_match_type": by_type,
            "top_rules": top_list,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return cumulative engine statistics.

        Returns:
            Dictionary with evaluation counters and breakdowns.
        """
        with self._lock:
            stats = dict(self._stats)
            stats["by_match_type"] = dict(self._stats["by_match_type"])
        stats["total_rules"] = len(self._rules)
        stats["active_rules"] = sum(1 for r in self._rules.values() if r.active)
        return stats

    # ------------------------------------------------------------------
    # Internal - Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_rule(
        self,
        rule: CategoryRule,
        record: Dict[str, Any],
    ) -> bool:
        """Evaluate a single rule against a record.

        Args:
            rule: Rule to evaluate.
            record: Spend record dict.

        Returns:
            True if the rule matches.
        """
        # Get the text to match against
        text = str(record.get(rule.match_field, "")).strip()
        if not text:
            # Try common fallback fields
            for field in ("description", "vendor_name", "category"):
                text = str(record.get(field, "")).strip()
                if text:
                    break

        if not text:
            return False

        pattern = rule.pattern

        if rule.match_type == MatchType.EXACT:
            return text.lower() == pattern.lower()

        elif rule.match_type == MatchType.CONTAINS:
            return pattern.lower() in text.lower()

        elif rule.match_type == MatchType.STARTS_WITH:
            return text.lower().startswith(pattern.lower())

        elif rule.match_type == MatchType.ENDS_WITH:
            return text.lower().endswith(pattern.lower())

        elif rule.match_type == MatchType.REGEX:
            try:
                return bool(re.search(pattern, text, re.IGNORECASE))
            except re.error:
                logger.warning("Regex error in rule %s: %s", rule.rule_id, pattern)
                return False

        elif rule.match_type == MatchType.FUZZY:
            similarity = _string_similarity(text, pattern)
            return similarity >= rule.fuzzy_threshold

        return False

    # ------------------------------------------------------------------
    # Internal - Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(
        self,
        rule_id: str,
        name: str,
        match_type: str,
        pattern: str,
        target: str,
        timestamp: str,
    ) -> str:
        """Compute SHA-256 provenance hash for a rule.

        Args:
            rule_id: Rule identifier.
            name: Rule name.
            match_type: Match type.
            pattern: Match pattern.
            target: Target category.
            timestamp: Mutation timestamp.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps({
            "rule_id": rule_id,
            "name": name,
            "match_type": match_type,
            "pattern": pattern,
            "target": target,
            "timestamp": timestamp,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _string_similarity(a: str, b: str) -> float:
    """Compute normalised LCS-based string similarity.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Similarity in [0, 1].
    """
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    a_lower = a.lower().strip()
    b_lower = b.lower().strip()

    if a_lower == b_lower:
        return 1.0

    m, n = len(a_lower), len(b_lower)
    if m == 0 or n == 0:
        return 0.0

    # LCS DP
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a_lower[i - 1] == b_lower[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr

    lcs_len = prev[n]
    return (2.0 * lcs_len) / (m + n)
