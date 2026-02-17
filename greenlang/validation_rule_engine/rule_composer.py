# -*- coding: utf-8 -*-
"""
Rule Composer Engine - AGENT-DATA-019: Validation Rule Engine (GL-DATA-X-022)
=============================================================================

Engine 2 of 7 -- RuleComposerEngine.

Builds compound rules from atomic rules using logical operators (AND, OR, NOT),
manages named rule sets with SemVer versioning, supports rule templates and
inheritance (child rule sets extending parent sets with overrides), and maintains
a rule dependency DAG with topological sort (Kahn's algorithm) for evaluation
ordering and DFS cycle detection.

Capabilities:
    - AND/OR/NOT composition of atomic rules into compound expressions
    - Max nesting depth enforcement (configurable, default 10)
    - Flatten compound rules to their constituent atomic rule IDs
    - Structure validation for compound rule trees
    - Named rule sets with SemVer versioning (auto-bump on mutation)
    - Rule set CRUD with SLA threshold management
    - Rule templates (abstract rule set patterns)
    - Template instantiation with parameter overrides
    - Rule set inheritance (child extends parent, override/add rules)
    - Inheritance chain traversal
    - Rule dependency graph (DAG) with Kahn's topological sort
    - DFS cycle detection in dependency graphs
    - Rule set comparison (diff: added/removed/modified rules)
    - Thread-safe with reentrant locking
    - SHA-256 provenance hashes on every mutating operation

Zero-Hallucination Guarantees:
    - All composition logic is deterministic tree construction
    - SemVer arithmetic is integer tuple operations
    - Topological sort is Kahn's algorithm (deterministic BFS)
    - Cycle detection is iterative DFS (no recursion stack overflow risk)
    - No LLM or ML models in any composition, evaluation, or graph path
    - SHA-256 provenance hashes for complete audit trails
    - Thread-safe with reentrant locking

Example:
    >>> from greenlang.validation_rule_engine.rule_composer import RuleComposerEngine
    >>> # Assume registry is a RuleRegistryEngine instance with rules registered
    >>> composer = RuleComposerEngine(registry=registry)
    >>> compound = composer.create_compound_rule(
    ...     name="valid_emission",
    ...     operator="AND",
    ...     rule_ids=["rule_001", "rule_002"],
    ...     description="Both rules must pass",
    ... )
    >>> print(compound["compound_id"], compound["operator"])
    >>> rule_set = composer.create_rule_set(
    ...     name="GHG Scope 1 Validation",
    ...     description="All rules for Scope 1 emissions",
    ...     rule_ids=["rule_001", "rule_002"],
    ...     compound_rule_ids=[compound["compound_id"]],
    ... )
    >>> print(rule_set["set_id"], rule_set["version"])

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-019 Validation Rule Engine (GL-DATA-X-022)
Status: Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "RuleComposerEngine",
    "MAX_COMPOUND_DEPTH",
    "VALID_COMPOUND_OPERATORS",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Valid logical operators for compound rules.
VALID_OPERATORS: FrozenSet[str] = frozenset({"AND", "OR", "NOT"})

#: Public alias for valid compound operators.
VALID_COMPOUND_OPERATORS: FrozenSet[str] = VALID_OPERATORS

#: Operators requiring two or more child rules.
MULTI_CHILD_OPERATORS: FrozenSet[str] = frozenset({"AND", "OR"})

#: SemVer bump type strings.
BUMP_MAJOR = "major"
BUMP_MINOR = "minor"
BUMP_PATCH = "patch"

#: Default maximum nesting depth for compound rules.
DEFAULT_MAX_NESTING_DEPTH = 10

#: Public alias for maximum compound rule nesting depth.
MAX_COMPOUND_DEPTH: int = DEFAULT_MAX_NESTING_DEPTH

#: Default maximum rules per set.
DEFAULT_MAX_RULES_PER_SET = 500

#: Default SLA thresholds for rule sets.
DEFAULT_SLA_THRESHOLDS: Dict[str, float] = {
    "critical_pass_rate": 0.99,
    "overall_pass_rate": 0.95,
    "warn_pass_rate": 0.80,
}

#: ID prefixes for generated identifiers.
_PREFIX_COMPOUND = "CMP"
_PREFIX_RULE_SET = "RS"
_PREFIX_TEMPLATE = "TPL"
_PREFIX_DEPENDENCY = "DEP"


# ---------------------------------------------------------------------------
# Graceful import: ProvenanceTracker
# ---------------------------------------------------------------------------

try:
    from greenlang.validation_rule_engine.provenance import (
        ProvenanceTracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    logger.info(
        "ProvenanceTracker not available; using built-in SHA-256 tracking only."
    )

    class ProvenanceTracker:  # type: ignore[no-redef]
        """Minimal fallback ProvenanceTracker when the real module is absent."""

        def __init__(self) -> None:
            self._log: List[Dict[str, Any]] = []

        def record(
            self,
            entity_type: str,
            entity_id: str,
            action: str,
            metadata: Any = None,
        ) -> Any:
            """Record a provenance entry and return a stub object."""
            ts = _utcnow().isoformat()
            data_str = json.dumps(metadata, sort_keys=True, default=str) if metadata else "null"
            chain_hash = hashlib.sha256(
                f"{entity_type}:{entity_id}:{action}:{data_str}:{ts}".encode()
            ).hexdigest()
            entry = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "action": action,
                "hash_value": chain_hash,
                "timestamp": ts,
            }
            self._log.append(entry)
            return type("ProvenanceEntry", (), entry)()

        def build_hash(self, data: Any) -> str:
            """Return SHA-256 hash of JSON-serialised data."""
            return hashlib.sha256(
                json.dumps(data, sort_keys=True, default=str).encode()
            ).hexdigest()


# ---------------------------------------------------------------------------
# Graceful import: Prometheus metrics
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore
    _PROMETHEUS_AVAILABLE = True

    vre_compound_rules_created_total = Counter(
        "gl_vre_compound_rules_created_total",
        "Total compound rules created",
        labelnames=["operator"],
    )
    vre_rule_sets_created_total = Counter(
        "gl_vre_rule_sets_created_total",
        "Total rule sets created",
        labelnames=["pack_type"],
    )
    vre_rule_sets_updated_total = Counter(
        "gl_vre_rule_sets_updated_total",
        "Total rule set updates",
    )
    vre_templates_created_total = Counter(
        "gl_vre_templates_created_total",
        "Total templates created",
    )
    vre_templates_instantiated_total = Counter(
        "gl_vre_templates_instantiated_total",
        "Total template instantiations",
    )
    vre_dependencies_added_total = Counter(
        "gl_vre_dependencies_added_total",
        "Total rule dependencies added",
    )
    vre_cycles_detected_total = Counter(
        "gl_vre_cycles_detected_total",
        "Total dependency cycles detected",
    )
    vre_active_compound_rules = Gauge(
        "gl_vre_active_compound_rules",
        "Current number of compound rules",
    )
    vre_active_rule_sets_gauge = Gauge(
        "gl_vre_active_rule_sets",
        "Current number of rule sets",
    )
    vre_composer_operation_duration = Histogram(
        "gl_vre_composer_operation_duration_seconds",
        "Duration of composer operations in seconds",
        labelnames=["operation"],
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )

except (ImportError, ValueError):
    _PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed or metrics already registered; "
        "rule composer metrics disabled."
    )

    class _NoOpCounter:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpCounter":
            return self
        def inc(self, _: float = 1) -> None:
            pass

    class _NoOpGauge:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpGauge":
            return self
        def set(self, _: float) -> None:
            pass
        def inc(self, _: float = 1) -> None:
            pass
        def dec(self, _: float = 1) -> None:
            pass

    class _NoOpHistogram:  # type: ignore
        def labels(self, **_: Any) -> "_NoOpHistogram":
            return self
        def observe(self, _: float) -> None:
            pass

    vre_compound_rules_created_total = _NoOpCounter()  # type: ignore
    vre_rule_sets_created_total = _NoOpCounter()  # type: ignore
    vre_rule_sets_updated_total = _NoOpCounter()  # type: ignore
    vre_templates_created_total = _NoOpCounter()  # type: ignore
    vre_templates_instantiated_total = _NoOpCounter()  # type: ignore
    vre_dependencies_added_total = _NoOpCounter()  # type: ignore
    vre_cycles_detected_total = _NoOpCounter()  # type: ignore
    vre_active_compound_rules = _NoOpGauge()  # type: ignore
    vre_active_rule_sets_gauge = _NoOpGauge()  # type: ignore
    vre_composer_operation_duration = _NoOpHistogram()  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed for determinism.

    Returns:
        Timezone-aware datetime at second precision in UTC.
    """
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: Short uppercase string prepended to the random hex segment.

    Returns:
        String of the form ``{prefix}-{hex12}``.

    Example:
        >>> _generate_id("CMP")
        'CMP-3f9a1b2c4d5e'
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _build_sha256(data: Any) -> str:
    """Build a deterministic SHA-256 hash from any JSON-serialisable value.

    Args:
        data: JSON-serialisable value (dict, list, str, int, etc.).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_version(version_string: str) -> Tuple[int, int, int]:
    """Parse a SemVer string into its (major, minor, patch) integer tuple.

    Args:
        version_string: A semantic version string in the form ``"X.Y.Z"``.

    Returns:
        A three-tuple of non-negative integers (major, minor, patch).

    Raises:
        ValueError: If the string does not conform to the ``X.Y.Z`` format.
    """
    parts = version_string.strip().split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': expected 3 parts."
        )
    try:
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError as exc:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': all parts must be integers."
        ) from exc
    if major < 0 or minor < 0 or patch < 0:
        raise ValueError(
            f"Invalid SemVer string '{version_string}': parts must be non-negative."
        )
    return major, minor, patch


def _bump_version(current_version: str, bump_type: str) -> str:
    """Increment a SemVer string according to the specified bump type.

    Rules:
        - ``"major"``: increment major, reset minor and patch to 0.
        - ``"minor"``: increment minor, reset patch to 0.
        - ``"patch"``: increment patch only.

    Args:
        current_version: Current SemVer string (e.g. ``"1.0.0"``).
        bump_type: One of ``"major"``, ``"minor"``, or ``"patch"``.

    Returns:
        New SemVer string after the bump.

    Raises:
        ValueError: If ``bump_type`` is not a recognised value.
    """
    major, minor, patch = _parse_version(current_version)
    if bump_type == BUMP_MAJOR:
        return f"{major + 1}.0.0"
    if bump_type == BUMP_MINOR:
        return f"{major}.{minor + 1}.0"
    if bump_type == BUMP_PATCH:
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(
        f"Unknown bump_type '{bump_type}'. Must be one of: major, minor, patch."
    )


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    """Normalize, deduplicate, and sort a list of tags.

    Args:
        tags: Raw tag list, or None.

    Returns:
        Sorted list of unique, lowercased, stripped tag strings.
    """
    if not tags:
        return []
    seen: Set[str] = set()
    result: List[str] = []
    for tag in tags:
        normalized = tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


# ---------------------------------------------------------------------------
# RuleComposerEngine
# ---------------------------------------------------------------------------


class RuleComposerEngine:
    """Compound rule composition, rule set management, and dependency graphing.

    Engine 2 of 7 in the Validation Rule Engine pipeline. Takes a reference
    to a ``RuleRegistryEngine`` (Engine 1) so that compound rules and rule
    sets can validate that their constituent rule IDs actually exist.

    Thread-safe: all public methods acquire ``self._lock`` before reading
    or mutating internal state.

    Attributes:
        _registry: Reference to the RuleRegistryEngine for rule existence checks.
        _provenance: ProvenanceTracker for SHA-256 audit trails.
        _compound_rules: In-memory store of compound rules keyed by compound_id.
        _rule_sets: In-memory store of rule sets keyed by set_id.
        _rule_set_versions: Version history per set_id.
        _templates: In-memory store of rule templates keyed by template_id.
        _dependencies: Adjacency list for rule dependency graph.
        _max_nesting_depth: Maximum allowed nesting depth for compound rules.
        _max_rules_per_set: Maximum allowed rules in a single rule set.
        _lock: Reentrant lock for thread safety.

    Example:
        >>> composer = RuleComposerEngine(registry=registry)
        >>> compound = composer.create_compound_rule(
        ...     name="both_pass",
        ...     operator="AND",
        ...     rule_ids=["rule_001", "rule_002"],
        ... )
        >>> assert compound["operator"] == "AND"
    """

    def __init__(
        self,
        registry: Any,
        provenance: Optional[ProvenanceTracker] = None,
        max_nesting_depth: int = DEFAULT_MAX_NESTING_DEPTH,
        max_rules_per_set: int = DEFAULT_MAX_RULES_PER_SET,
    ) -> None:
        """Initialize RuleComposerEngine with a registry reference.

        Args:
            registry: A RuleRegistryEngine instance (or any object that
                exposes a ``get_rule(rule_id) -> Optional[dict]`` method).
            provenance: Optional ProvenanceTracker for audit trails. When
                ``None`` a new tracker is created.
            max_nesting_depth: Maximum allowed nesting depth for compound
                rules. Defaults to 10.
            max_rules_per_set: Maximum allowed number of rules in a single
                rule set. Defaults to 500.
        """
        self._registry = registry
        self._provenance: ProvenanceTracker = (
            provenance if provenance is not None else ProvenanceTracker()
        )
        self._max_nesting_depth: int = max_nesting_depth
        self._max_rules_per_set: int = max_rules_per_set

        # In-memory stores
        self._compound_rules: Dict[str, Dict[str, Any]] = {}
        self._rule_sets: Dict[str, Dict[str, Any]] = {}
        self._rule_set_versions: Dict[str, List[Dict[str, Any]]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "RuleComposerEngine initialized: max_nesting_depth=%d, "
            "max_rules_per_set=%d",
            self._max_nesting_depth,
            self._max_rules_per_set,
        )

    # ======================================================================
    # COMPOUND RULE METHODS (1-4)
    # ======================================================================

    def create_compound_rule(
        self,
        name: str,
        operator: str,
        rule_ids: List[str],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a compound rule from atomic or other compound rule IDs.

        A compound rule combines two or more rules (AND, OR) or negates
        exactly one rule (NOT) into a logical expression tree.

        Args:
            name: Human-readable name for the compound rule.
            operator: Logical operator. Must be one of ``AND``, ``OR``,
                or ``NOT``.
            rule_ids: List of rule IDs (atomic or compound) to compose.
                AND/OR require 2+ IDs; NOT requires exactly 1.
            description: Optional human-readable description.

        Returns:
            Dictionary representing the created compound rule with keys:
            ``compound_id``, ``name``, ``operator``, ``rule_ids``,
            ``description``, ``created_at``, ``updated_at``,
            ``provenance_hash``, ``nesting_depth``.

        Raises:
            ValueError: If the operator is invalid, rule_ids count violates
                operator constraints, any rule_id does not exist, or nesting
                depth exceeds the configured maximum.
        """
        start_time = time.monotonic()

        # Validate operator
        normalized_operator = operator.upper().strip()
        if normalized_operator not in VALID_OPERATORS:
            raise ValueError(
                f"Invalid operator '{operator}'. Must be one of: "
                f"{sorted(VALID_OPERATORS)}"
            )

        # Validate rule_ids count
        if not rule_ids:
            raise ValueError("rule_ids must not be empty.")

        if normalized_operator == "NOT":
            if len(rule_ids) != 1:
                raise ValueError(
                    f"NOT operator requires exactly 1 rule_id, "
                    f"got {len(rule_ids)}."
                )
        elif len(rule_ids) < 2:
            raise ValueError(
                f"{normalized_operator} operator requires 2 or more "
                f"rule_ids, got {len(rule_ids)}."
            )

        # Validate all rule_ids exist (atomic or compound)
        with self._lock:
            for rule_id in rule_ids:
                if not self._rule_exists(rule_id):
                    raise ValueError(
                        f"Rule ID '{rule_id}' does not exist in the "
                        f"registry or compound rules."
                    )

            # Calculate nesting depth
            depth = self._calculate_nesting_depth(rule_ids)
            if depth >= self._max_nesting_depth:
                raise ValueError(
                    f"Compound rule nesting depth {depth + 1} exceeds "
                    f"maximum allowed depth of {self._max_nesting_depth}."
                )

            # Build the compound rule record
            compound_id = _generate_id(_PREFIX_COMPOUND)
            now = _utcnow().isoformat()

            compound_rule: Dict[str, Any] = {
                "compound_id": compound_id,
                "name": name.strip(),
                "operator": normalized_operator,
                "rule_ids": list(rule_ids),
                "description": description.strip() if description else "",
                "created_at": now,
                "updated_at": now,
                "nesting_depth": depth + 1,
                "provenance_hash": "",
            }

            # Compute provenance hash
            provenance_hash = _build_sha256(compound_rule)
            compound_rule["provenance_hash"] = provenance_hash

            self._compound_rules[compound_id] = compound_rule

        # Record provenance
        self._provenance.record(
            entity_type="compound_rule",
            entity_id=compound_id,
            action="compound_rule_composed",
            metadata=compound_rule,
        )

        # Metrics
        vre_compound_rules_created_total.labels(
            operator=normalized_operator
        ).inc()
        vre_active_compound_rules.set(len(self._compound_rules))
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="create_compound_rule"
        ).observe(duration)

        logger.info(
            "Created compound rule '%s' (id=%s, operator=%s, "
            "children=%d, depth=%d) in %.4fs",
            name,
            compound_id,
            normalized_operator,
            len(rule_ids),
            depth + 1,
            duration,
        )
        return copy.deepcopy(compound_rule)

    def get_compound_rule(
        self, compound_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a compound rule by its ID.

        Args:
            compound_id: The unique identifier of the compound rule.

        Returns:
            A deep copy of the compound rule dictionary, or ``None`` if
            no compound rule with the given ID exists.
        """
        with self._lock:
            rule = self._compound_rules.get(compound_id)
            if rule is None:
                return None
            return copy.deepcopy(rule)

    def flatten_compound_rule(self, compound_id: str) -> List[str]:
        """Flatten a compound rule tree into its constituent atomic rule IDs.

        Traverses the compound rule tree depth-first and collects all
        leaf-level (atomic) rule IDs, deduplicating while preserving
        first-seen order.

        Args:
            compound_id: The compound rule ID to flatten.

        Returns:
            Ordered list of unique atomic rule IDs that make up the
            compound rule.

        Raises:
            ValueError: If the compound_id does not exist.
        """
        start_time = time.monotonic()

        with self._lock:
            if compound_id not in self._compound_rules:
                raise ValueError(
                    f"Compound rule '{compound_id}' does not exist."
                )

            seen: Set[str] = set()
            result: List[str] = []
            self._flatten_recursive(compound_id, seen, result)

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="flatten_compound_rule"
        ).observe(duration)

        logger.debug(
            "Flattened compound rule '%s' to %d atomic rules in %.4fs",
            compound_id,
            len(result),
            duration,
        )
        return result

    def evaluate_compound_structure(
        self, compound_id: str
    ) -> Dict[str, Any]:
        """Validate that a compound rule tree is well-formed.

        Checks:
            - The compound_id exists
            - All child rule_ids exist (recursively)
            - No circular references
            - Nesting depth within limits
            - Operator arity constraints are satisfied

        Args:
            compound_id: The compound rule ID to validate.

        Returns:
            Dictionary with keys:
            ``valid`` (bool), ``compound_id`` (str),
            ``total_atomic_rules`` (int), ``max_depth`` (int),
            ``errors`` (list of str), ``warnings`` (list of str).
        """
        start_time = time.monotonic()
        errors: List[str] = []
        warnings: List[str] = []

        with self._lock:
            if compound_id not in self._compound_rules:
                return {
                    "valid": False,
                    "compound_id": compound_id,
                    "total_atomic_rules": 0,
                    "max_depth": 0,
                    "errors": [
                        f"Compound rule '{compound_id}' does not exist."
                    ],
                    "warnings": [],
                }

            visited: Set[str] = set()
            max_depth = self._validate_tree(
                compound_id, visited, errors, warnings, current_depth=0
            )

            # Flatten to count atomic rules
            seen: Set[str] = set()
            atomic: List[str] = []
            self._flatten_recursive(compound_id, seen, atomic)

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="evaluate_compound_structure"
        ).observe(duration)

        result = {
            "valid": len(errors) == 0,
            "compound_id": compound_id,
            "total_atomic_rules": len(atomic),
            "max_depth": max_depth,
            "errors": errors,
            "warnings": warnings,
        }

        logger.info(
            "Evaluated compound structure '%s': valid=%s, depth=%d, "
            "atomic_rules=%d, errors=%d in %.4fs",
            compound_id,
            result["valid"],
            max_depth,
            len(atomic),
            len(errors),
            duration,
        )
        return result

    # ======================================================================
    # RULE SET METHODS (5-12)
    # ======================================================================

    def create_rule_set(
        self,
        name: str,
        description: str,
        rule_ids: List[str],
        compound_rule_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        sla_thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Create a named rule set with versioning and SLA thresholds.

        A rule set is a named, versioned collection of atomic and/or
        compound rules with shared metadata and SLA thresholds for
        pass/warn/fail evaluation.

        Args:
            name: Human-readable name for the rule set.
            description: Description of the rule set's purpose.
            rule_ids: List of atomic rule IDs to include.
            compound_rule_ids: Optional list of compound rule IDs to include.
            tags: Optional list of tags for categorization.
            sla_thresholds: Optional SLA thresholds. Defaults to
                ``{"critical_pass_rate": 0.99, "overall_pass_rate": 0.95,
                "warn_pass_rate": 0.80}``.

        Returns:
            Dictionary representing the created rule set with keys:
            ``set_id``, ``name``, ``description``, ``version``,
            ``rule_ids``, ``compound_rule_ids``, ``tags``,
            ``sla_thresholds``, ``parent_set_id``, ``created_at``,
            ``updated_at``, ``provenance_hash``, ``status``,
            ``rule_count``.

        Raises:
            ValueError: If the name is empty, any rule_id does not exist,
                or the total rule count exceeds the maximum.
        """
        start_time = time.monotonic()

        if not name or not name.strip():
            raise ValueError("Rule set name must not be empty.")

        compound_rule_ids = compound_rule_ids if compound_rule_ids is not None else []
        resolved_tags = _normalize_tags(tags)
        resolved_sla = self._resolve_sla_thresholds(sla_thresholds)

        with self._lock:
            # Validate all rule_ids exist
            for rule_id in rule_ids:
                if not self._rule_exists(rule_id):
                    raise ValueError(
                        f"Rule ID '{rule_id}' does not exist in the "
                        f"registry or compound rules."
                    )

            # Validate all compound_rule_ids exist
            for cmp_id in compound_rule_ids:
                if cmp_id not in self._compound_rules:
                    raise ValueError(
                        f"Compound rule ID '{cmp_id}' does not exist."
                    )

            # Check max rules per set
            total_count = len(rule_ids) + len(compound_rule_ids)
            if total_count > self._max_rules_per_set:
                raise ValueError(
                    f"Total rule count {total_count} exceeds maximum "
                    f"of {self._max_rules_per_set} rules per set."
                )

            # Build the rule set record
            set_id = _generate_id(_PREFIX_RULE_SET)
            now = _utcnow().isoformat()
            version = "1.0.0"

            rule_set: Dict[str, Any] = {
                "set_id": set_id,
                "name": name.strip(),
                "description": description.strip() if description else "",
                "version": version,
                "rule_ids": list(rule_ids),
                "compound_rule_ids": list(compound_rule_ids),
                "tags": resolved_tags,
                "sla_thresholds": resolved_sla,
                "parent_set_id": None,
                "created_at": now,
                "updated_at": now,
                "status": "active",
                "rule_count": total_count,
                "provenance_hash": "",
            }

            # Compute provenance hash
            provenance_hash = _build_sha256(rule_set)
            rule_set["provenance_hash"] = provenance_hash

            self._rule_sets[set_id] = rule_set

            # Store initial version snapshot
            self._rule_set_versions[set_id] = [
                self._create_version_snapshot(rule_set)
            ]

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_set_created",
            metadata=rule_set,
        )

        # Metrics
        vre_rule_sets_created_total.labels(pack_type="custom").inc()
        vre_active_rule_sets_gauge.set(len(self._rule_sets))
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="create_rule_set"
        ).observe(duration)

        logger.info(
            "Created rule set '%s' (id=%s, v=%s, rules=%d, "
            "compound=%d, tags=%s) in %.4fs",
            name,
            set_id,
            version,
            len(rule_ids),
            len(compound_rule_ids),
            resolved_tags,
            duration,
        )
        return copy.deepcopy(rule_set)

    def get_rule_set(self, set_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a rule set by its ID.

        Args:
            set_id: The unique identifier of the rule set.

        Returns:
            A deep copy of the rule set dictionary, or ``None`` if no
            rule set with the given ID exists.
        """
        with self._lock:
            rule_set = self._rule_sets.get(set_id)
            if rule_set is None:
                return None
            return copy.deepcopy(rule_set)

    def update_rule_set(
        self, set_id: str, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """Update a rule set's fields with automatic SemVer bump.

        Updatable fields: ``name``, ``description``, ``tags``,
        ``sla_thresholds``, ``status``. Changes to ``rule_ids`` or
        ``compound_rule_ids`` should use ``add_rules_to_set`` /
        ``remove_rules_from_set`` for proper tracking.

        Version bump classification:
            - ``name`` or ``status`` change: minor bump.
            - ``description``, ``tags``, ``sla_thresholds`` change: patch bump.

        Args:
            set_id: The rule set ID to update.
            **kwargs: Fields to update. Supported keys: ``name``,
                ``description``, ``tags``, ``sla_thresholds``, ``status``.

        Returns:
            Updated rule set dictionary, or ``None`` if the set_id
            does not exist.

        Raises:
            ValueError: If an invalid field is provided or the status
                transition is invalid.
        """
        start_time = time.monotonic()

        allowed_fields = {"name", "description", "tags", "sla_thresholds", "status"}
        invalid_fields = set(kwargs.keys()) - allowed_fields
        if invalid_fields:
            raise ValueError(
                f"Cannot update fields: {sorted(invalid_fields)}. "
                f"Allowed: {sorted(allowed_fields)}"
            )

        with self._lock:
            if set_id not in self._rule_sets:
                return None

            rule_set = self._rule_sets[set_id]
            bump_type = BUMP_PATCH
            changes_made = False

            # Apply updates
            if "name" in kwargs:
                new_name = kwargs["name"].strip()
                if new_name != rule_set["name"]:
                    rule_set["name"] = new_name
                    bump_type = BUMP_MINOR
                    changes_made = True

            if "status" in kwargs:
                new_status = kwargs["status"]
                if new_status != rule_set["status"]:
                    self._validate_status_transition(
                        rule_set["status"], new_status
                    )
                    rule_set["status"] = new_status
                    bump_type = BUMP_MINOR
                    changes_made = True

            if "description" in kwargs:
                new_desc = kwargs["description"].strip() if kwargs["description"] else ""
                if new_desc != rule_set["description"]:
                    rule_set["description"] = new_desc
                    changes_made = True

            if "tags" in kwargs:
                new_tags = _normalize_tags(kwargs["tags"])
                if new_tags != rule_set["tags"]:
                    rule_set["tags"] = new_tags
                    changes_made = True

            if "sla_thresholds" in kwargs:
                new_sla = self._resolve_sla_thresholds(kwargs["sla_thresholds"])
                if new_sla != rule_set["sla_thresholds"]:
                    rule_set["sla_thresholds"] = new_sla
                    changes_made = True

            if not changes_made:
                return copy.deepcopy(rule_set)

            # Bump version
            rule_set["version"] = _bump_version(rule_set["version"], bump_type)
            rule_set["updated_at"] = _utcnow().isoformat()
            rule_set["provenance_hash"] = _build_sha256(rule_set)

            # Store version snapshot
            self._rule_set_versions[set_id].append(
                self._create_version_snapshot(rule_set)
            )

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_set_updated",
            metadata={"changes": kwargs, "new_version": rule_set["version"]},
        )

        # Metrics
        vre_rule_sets_updated_total.inc()
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="update_rule_set"
        ).observe(duration)

        logger.info(
            "Updated rule set '%s' (id=%s, v=%s, bump=%s) in %.4fs",
            rule_set["name"],
            set_id,
            rule_set["version"],
            bump_type,
            duration,
        )
        return copy.deepcopy(rule_set)

    def delete_rule_set(self, set_id: str) -> bool:
        """Delete a rule set by its ID.

        Removes the rule set and its version history from the in-memory
        store. This operation is permanent within the current session.

        Args:
            set_id: The rule set ID to delete.

        Returns:
            ``True`` if the rule set was found and deleted, ``False``
            if no rule set with the given ID existed.
        """
        start_time = time.monotonic()

        with self._lock:
            if set_id not in self._rule_sets:
                logger.warning(
                    "Attempted to delete non-existent rule set '%s'", set_id
                )
                return False

            deleted_set = self._rule_sets.pop(set_id)
            self._rule_set_versions.pop(set_id, None)

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_set_deleted",
            metadata={"name": deleted_set["name"], "version": deleted_set["version"]},
        )

        # Metrics
        vre_active_rule_sets_gauge.set(len(self._rule_sets))
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="delete_rule_set"
        ).observe(duration)

        logger.info(
            "Deleted rule set '%s' (id=%s) in %.4fs",
            deleted_set["name"],
            set_id,
            duration,
        )
        return True

    def add_rules_to_set(
        self, set_id: str, rule_ids: List[str]
    ) -> Dict[str, Any]:
        """Add rules to an existing rule set.

        Deduplicates against already-present rule IDs. Triggers a minor
        version bump.

        Args:
            set_id: The rule set ID.
            rule_ids: List of rule IDs to add.

        Returns:
            Updated rule set dictionary.

        Raises:
            ValueError: If the set_id does not exist, any rule_id does
                not exist, or adding would exceed the maximum rule count.
        """
        start_time = time.monotonic()

        with self._lock:
            if set_id not in self._rule_sets:
                raise ValueError(f"Rule set '{set_id}' does not exist.")

            rule_set = self._rule_sets[set_id]

            # Validate all new rule_ids exist
            for rule_id in rule_ids:
                if not self._rule_exists(rule_id):
                    raise ValueError(
                        f"Rule ID '{rule_id}' does not exist in the "
                        f"registry or compound rules."
                    )

            # Deduplicate
            existing_ids = set(rule_set["rule_ids"])
            new_ids = [rid for rid in rule_ids if rid not in existing_ids]

            if not new_ids:
                logger.debug(
                    "No new rules to add to set '%s'; all already present.",
                    set_id,
                )
                return copy.deepcopy(rule_set)

            # Check capacity
            new_total = (
                len(rule_set["rule_ids"])
                + len(rule_set["compound_rule_ids"])
                + len(new_ids)
            )
            if new_total > self._max_rules_per_set:
                raise ValueError(
                    f"Adding {len(new_ids)} rules would bring total to "
                    f"{new_total}, exceeding maximum of "
                    f"{self._max_rules_per_set}."
                )

            # Apply additions
            rule_set["rule_ids"].extend(new_ids)
            rule_set["rule_count"] = (
                len(rule_set["rule_ids"]) + len(rule_set["compound_rule_ids"])
            )
            rule_set["version"] = _bump_version(rule_set["version"], BUMP_MINOR)
            rule_set["updated_at"] = _utcnow().isoformat()
            rule_set["provenance_hash"] = _build_sha256(rule_set)

            # Store version snapshot
            self._rule_set_versions[set_id].append(
                self._create_version_snapshot(rule_set)
            )

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_added_to_set",
            metadata={"added_rule_ids": new_ids},
        )

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="add_rules_to_set"
        ).observe(duration)

        logger.info(
            "Added %d rules to set '%s' (id=%s, v=%s) in %.4fs",
            len(new_ids),
            rule_set["name"],
            set_id,
            rule_set["version"],
            duration,
        )
        return copy.deepcopy(rule_set)

    def remove_rules_from_set(
        self, set_id: str, rule_ids: List[str]
    ) -> Dict[str, Any]:
        """Remove rules from an existing rule set.

        Silently ignores rule IDs that are not in the set. Triggers a
        minor version bump if any rules were actually removed.

        Args:
            set_id: The rule set ID.
            rule_ids: List of rule IDs to remove.

        Returns:
            Updated rule set dictionary.

        Raises:
            ValueError: If the set_id does not exist.
        """
        start_time = time.monotonic()

        with self._lock:
            if set_id not in self._rule_sets:
                raise ValueError(f"Rule set '{set_id}' does not exist.")

            rule_set = self._rule_sets[set_id]
            ids_to_remove = set(rule_ids)

            # Remove from rule_ids
            original_rule_count = len(rule_set["rule_ids"])
            rule_set["rule_ids"] = [
                rid for rid in rule_set["rule_ids"]
                if rid not in ids_to_remove
            ]
            removed_from_rules = original_rule_count - len(rule_set["rule_ids"])

            # Remove from compound_rule_ids
            original_compound_count = len(rule_set["compound_rule_ids"])
            rule_set["compound_rule_ids"] = [
                cid for cid in rule_set["compound_rule_ids"]
                if cid not in ids_to_remove
            ]
            removed_from_compound = original_compound_count - len(
                rule_set["compound_rule_ids"]
            )

            total_removed = removed_from_rules + removed_from_compound

            if total_removed == 0:
                logger.debug(
                    "No rules removed from set '%s'; none of the IDs were present.",
                    set_id,
                )
                return copy.deepcopy(rule_set)

            # Update metadata
            rule_set["rule_count"] = (
                len(rule_set["rule_ids"]) + len(rule_set["compound_rule_ids"])
            )
            rule_set["version"] = _bump_version(rule_set["version"], BUMP_MINOR)
            rule_set["updated_at"] = _utcnow().isoformat()
            rule_set["provenance_hash"] = _build_sha256(rule_set)

            # Store version snapshot
            self._rule_set_versions[set_id].append(
                self._create_version_snapshot(rule_set)
            )

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_removed_from_set",
            metadata={
                "removed_rule_ids": list(ids_to_remove),
                "actually_removed": total_removed,
            },
        )

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="remove_rules_from_set"
        ).observe(duration)

        logger.info(
            "Removed %d rules from set '%s' (id=%s, v=%s) in %.4fs",
            total_removed,
            rule_set["name"],
            set_id,
            rule_set["version"],
            duration,
        )
        return copy.deepcopy(rule_set)

    def list_rule_sets(
        self,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List rule sets with optional tag filtering and pagination.

        Args:
            tags: Optional list of tags to filter by. A rule set matches
                if it contains all specified tags (AND logic).
            limit: Maximum number of results to return. Defaults to 100.
            offset: Number of results to skip. Defaults to 0.

        Returns:
            List of rule set dictionaries sorted by creation time
            (newest first), paginated by offset and limit.
        """
        with self._lock:
            result = list(self._rule_sets.values())

        # Filter by tags if provided
        if tags:
            normalized_filter_tags = set(_normalize_tags(tags))
            result = [
                rs for rs in result
                if normalized_filter_tags.issubset(set(rs.get("tags", [])))
            ]

        # Sort by created_at descending (newest first)
        result.sort(key=lambda rs: rs.get("created_at", ""), reverse=True)

        # Paginate
        paginated = result[offset: offset + limit]

        return [copy.deepcopy(rs) for rs in paginated]

    def get_rule_set_versions(self, set_id: str) -> List[Dict[str, Any]]:
        """Retrieve the version history of a rule set.

        Args:
            set_id: The rule set ID.

        Returns:
            List of version snapshot dictionaries sorted by version
            (oldest first). Returns an empty list if the set_id does
            not exist.
        """
        with self._lock:
            versions = self._rule_set_versions.get(set_id, [])
            return [copy.deepcopy(v) for v in versions]

    # ======================================================================
    # TEMPLATE & INHERITANCE METHODS (13-16)
    # ======================================================================

    def create_template(
        self,
        name: str,
        rule_definitions: List[Dict[str, Any]],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a rule template (abstract rule set pattern).

        A template defines a set of rule definitions that can be
        instantiated into concrete rule sets with optional parameter
        overrides. Templates do not require rule IDs to exist in the
        registry; they store rule definitions as patterns.

        Args:
            name: Human-readable name for the template.
            rule_definitions: List of rule definition dictionaries. Each
                definition should contain at minimum ``rule_type`` and
                ``parameters`` keys.
            description: Optional description of the template.

        Returns:
            Dictionary representing the created template with keys:
            ``template_id``, ``name``, ``description``,
            ``rule_definitions``, ``created_at``, ``updated_at``,
            ``provenance_hash``, ``instantiation_count``.

        Raises:
            ValueError: If the name is empty or rule_definitions is empty.
        """
        start_time = time.monotonic()

        if not name or not name.strip():
            raise ValueError("Template name must not be empty.")
        if not rule_definitions:
            raise ValueError("Template must contain at least one rule definition.")

        template_id = _generate_id(_PREFIX_TEMPLATE)
        now = _utcnow().isoformat()

        with self._lock:
            template: Dict[str, Any] = {
                "template_id": template_id,
                "name": name.strip(),
                "description": description.strip() if description else "",
                "rule_definitions": copy.deepcopy(rule_definitions),
                "created_at": now,
                "updated_at": now,
                "instantiation_count": 0,
                "provenance_hash": "",
            }

            provenance_hash = _build_sha256(template)
            template["provenance_hash"] = provenance_hash

            self._templates[template_id] = template

        # Record provenance
        self._provenance.record(
            entity_type="template",
            entity_id=template_id,
            action="template_created",
            metadata=template,
        )

        # Metrics
        vre_templates_created_total.inc()
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="create_template"
        ).observe(duration)

        logger.info(
            "Created template '%s' (id=%s, definitions=%d) in %.4fs",
            name,
            template_id,
            len(rule_definitions),
            duration,
        )
        return copy.deepcopy(template)

    def instantiate_template(
        self,
        template_id: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Instantiate a rule set from a template with optional overrides.

        Creates a concrete rule set by processing the template's rule
        definitions. The caller may provide overrides to modify specific
        rule definitions (keyed by index or rule_type) or global
        parameters like ``name``, ``description``, ``tags``, and
        ``sla_thresholds``.

        Args:
            template_id: The template ID to instantiate.
            overrides: Optional dictionary of overrides. Supported keys:
                ``name`` (str), ``description`` (str), ``tags`` (list),
                ``sla_thresholds`` (dict), ``rule_overrides`` (dict
                mapping definition index to parameter overrides).

        Returns:
            Dictionary representing the instantiated rule set (same
            structure as ``create_rule_set`` output), plus a
            ``template_id`` field linking to the source template.

        Raises:
            ValueError: If the template_id does not exist.
        """
        start_time = time.monotonic()
        overrides = overrides if overrides is not None else {}

        with self._lock:
            if template_id not in self._templates:
                raise ValueError(
                    f"Template '{template_id}' does not exist."
                )

            template = self._templates[template_id]
            definitions = copy.deepcopy(template["rule_definitions"])

            # Apply rule-level overrides
            rule_overrides = overrides.get("rule_overrides", {})
            for idx_str, override_params in rule_overrides.items():
                idx = int(idx_str)
                if 0 <= idx < len(definitions):
                    definitions[idx].update(override_params)

            # Build set metadata from template + overrides
            set_name = overrides.get(
                "name", f"{template['name']} - Instance"
            )
            set_description = overrides.get(
                "description",
                f"Instantiated from template '{template['name']}'"
            )
            set_tags = overrides.get("tags", [])
            set_sla = overrides.get("sla_thresholds", None)

            # Increment instantiation count
            template["instantiation_count"] += 1
            template["updated_at"] = _utcnow().isoformat()

        # Build the rule set record directly (rules are definition-based,
        # not registry-based, so we store definitions in the set)
        set_id = _generate_id(_PREFIX_RULE_SET)
        now = _utcnow().isoformat()
        resolved_sla = self._resolve_sla_thresholds(set_sla)
        resolved_tags = _normalize_tags(set_tags)

        with self._lock:
            rule_set: Dict[str, Any] = {
                "set_id": set_id,
                "name": set_name.strip() if isinstance(set_name, str) else set_name,
                "description": (
                    set_description.strip()
                    if isinstance(set_description, str)
                    else set_description
                ),
                "version": "1.0.0",
                "rule_ids": [],
                "compound_rule_ids": [],
                "rule_definitions": definitions,
                "tags": resolved_tags,
                "sla_thresholds": resolved_sla,
                "parent_set_id": None,
                "template_id": template_id,
                "created_at": now,
                "updated_at": now,
                "status": "active",
                "rule_count": len(definitions),
                "provenance_hash": "",
            }

            provenance_hash = _build_sha256(rule_set)
            rule_set["provenance_hash"] = provenance_hash

            self._rule_sets[set_id] = rule_set
            self._rule_set_versions[set_id] = [
                self._create_version_snapshot(rule_set)
            ]

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_set_created",
            metadata={
                "template_id": template_id,
                "overrides": overrides,
            },
        )

        # Metrics
        vre_templates_instantiated_total.inc()
        vre_rule_sets_created_total.labels(pack_type="template").inc()
        vre_active_rule_sets_gauge.set(len(self._rule_sets))
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="instantiate_template"
        ).observe(duration)

        logger.info(
            "Instantiated template '%s' as rule set '%s' (id=%s, "
            "definitions=%d) in %.4fs",
            template["name"],
            set_name,
            set_id,
            len(definitions),
            duration,
        )
        return copy.deepcopy(rule_set)

    def create_child_rule_set(
        self,
        parent_set_id: str,
        name: str,
        override_rules: Optional[Dict[str, str]] = None,
        additional_rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a child rule set that inherits from a parent.

        The child rule set contains all rules from the parent, with the
        ability to override specific rules (replace a parent rule ID with
        a child rule ID) and add additional rules.

        Args:
            parent_set_id: The parent rule set ID to inherit from.
            name: Human-readable name for the child rule set.
            override_rules: Optional dictionary mapping parent rule IDs
                to replacement child rule IDs. Each replacement ID must
                exist in the registry.
            additional_rules: Optional list of additional rule IDs to
                include beyond the inherited set.

        Returns:
            Dictionary representing the child rule set with
            ``parent_set_id`` pointing to the parent.

        Raises:
            ValueError: If the parent set does not exist, override rule
                IDs are invalid, or additional rules do not exist.
        """
        start_time = time.monotonic()
        override_rules = override_rules if override_rules is not None else {}
        additional_rules = additional_rules if additional_rules is not None else []

        with self._lock:
            if parent_set_id not in self._rule_sets:
                raise ValueError(
                    f"Parent rule set '{parent_set_id}' does not exist."
                )

            parent = self._rule_sets[parent_set_id]

            # Build inherited rule_ids with overrides applied
            inherited_rule_ids: List[str] = []
            for rule_id in parent["rule_ids"]:
                if rule_id in override_rules:
                    replacement_id = override_rules[rule_id]
                    if not self._rule_exists(replacement_id):
                        raise ValueError(
                            f"Override rule ID '{replacement_id}' does not "
                            f"exist in the registry or compound rules."
                        )
                    inherited_rule_ids.append(replacement_id)
                else:
                    inherited_rule_ids.append(rule_id)

            # Validate and add additional rules
            for rule_id in additional_rules:
                if not self._rule_exists(rule_id):
                    raise ValueError(
                        f"Additional rule ID '{rule_id}' does not exist "
                        f"in the registry or compound rules."
                    )

            # Deduplicate additional rules against inherited
            existing_ids = set(inherited_rule_ids)
            deduped_additional = [
                rid for rid in additional_rules
                if rid not in existing_ids
            ]
            final_rule_ids = inherited_rule_ids + deduped_additional

            # Inherit compound rules from parent (no overrides on compound)
            inherited_compound = list(parent["compound_rule_ids"])

            # Check capacity
            total_count = len(final_rule_ids) + len(inherited_compound)
            if total_count > self._max_rules_per_set:
                raise ValueError(
                    f"Total rule count {total_count} exceeds maximum "
                    f"of {self._max_rules_per_set}."
                )

            # Build child rule set
            set_id = _generate_id(_PREFIX_RULE_SET)
            now = _utcnow().isoformat()

            child_set: Dict[str, Any] = {
                "set_id": set_id,
                "name": name.strip(),
                "description": (
                    f"Child of '{parent['name']}' (id={parent_set_id})"
                ),
                "version": "1.0.0",
                "rule_ids": final_rule_ids,
                "compound_rule_ids": inherited_compound,
                "tags": list(parent.get("tags", [])),
                "sla_thresholds": copy.deepcopy(parent["sla_thresholds"]),
                "parent_set_id": parent_set_id,
                "created_at": now,
                "updated_at": now,
                "status": "active",
                "rule_count": total_count,
                "provenance_hash": "",
            }

            provenance_hash = _build_sha256(child_set)
            child_set["provenance_hash"] = provenance_hash

            self._rule_sets[set_id] = child_set
            self._rule_set_versions[set_id] = [
                self._create_version_snapshot(child_set)
            ]

        # Record provenance
        self._provenance.record(
            entity_type="rule_set",
            entity_id=set_id,
            action="rule_set_created",
            metadata={
                "parent_set_id": parent_set_id,
                "overrides": override_rules,
                "additional_rules": additional_rules,
            },
        )

        # Metrics
        vre_rule_sets_created_total.labels(pack_type="inherited").inc()
        vre_active_rule_sets_gauge.set(len(self._rule_sets))
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="create_child_rule_set"
        ).observe(duration)

        logger.info(
            "Created child rule set '%s' (id=%s, parent=%s, "
            "inherited=%d, overrides=%d, additional=%d) in %.4fs",
            name,
            set_id,
            parent_set_id,
            len(inherited_rule_ids),
            len(override_rules),
            len(deduped_additional),
            duration,
        )
        return copy.deepcopy(child_set)

    def get_inheritance_chain(self, set_id: str) -> List[Dict[str, Any]]:
        """Retrieve the inheritance chain for a rule set.

        Walks from the given set to its parent, grandparent, etc. until
        a root set (no parent) is reached. Includes cycle detection to
        prevent infinite loops.

        Args:
            set_id: The starting rule set ID.

        Returns:
            List of rule set summary dictionaries (``set_id``, ``name``,
            ``version``, ``parent_set_id``) ordered from the given set
            to the root ancestor. Returns an empty list if the set_id
            does not exist.
        """
        with self._lock:
            chain: List[Dict[str, Any]] = []
            visited: Set[str] = set()
            current_id: Optional[str] = set_id

            while current_id is not None:
                if current_id in visited:
                    logger.warning(
                        "Cycle detected in inheritance chain at '%s'",
                        current_id,
                    )
                    break

                if current_id not in self._rule_sets:
                    if not chain:
                        return []
                    break

                visited.add(current_id)
                rs = self._rule_sets[current_id]
                chain.append({
                    "set_id": rs["set_id"],
                    "name": rs["name"],
                    "version": rs["version"],
                    "parent_set_id": rs.get("parent_set_id"),
                })
                current_id = rs.get("parent_set_id")

            return chain

    # ======================================================================
    # DEPENDENCY GRAPH METHODS (17-19)
    # ======================================================================

    def add_rule_dependency(
        self, rule_id: str, depends_on_rule_id: str
    ) -> Dict[str, Any]:
        """Add a dependency between two rules.

        Declares that ``rule_id`` depends on ``depends_on_rule_id``,
        meaning the latter must pass before the former is evaluated.

        Args:
            rule_id: The rule that has the dependency.
            depends_on_rule_id: The rule that must pass first.

        Returns:
            Dictionary with keys: ``rule_id``, ``depends_on``,
            ``dependency_id``, ``created_at``, ``provenance_hash``.

        Raises:
            ValueError: If either rule does not exist, the dependency
                is self-referential, or adding it would create a cycle.
        """
        start_time = time.monotonic()

        if rule_id == depends_on_rule_id:
            raise ValueError(
                f"A rule cannot depend on itself: '{rule_id}'."
            )

        with self._lock:
            # Validate both rules exist
            if not self._rule_exists(rule_id):
                raise ValueError(
                    f"Rule ID '{rule_id}' does not exist."
                )
            if not self._rule_exists(depends_on_rule_id):
                raise ValueError(
                    f"Rule ID '{depends_on_rule_id}' does not exist."
                )

            # Check for duplicate
            if depends_on_rule_id in self._dependencies.get(rule_id, set()):
                logger.debug(
                    "Dependency '%s' -> '%s' already exists.",
                    rule_id,
                    depends_on_rule_id,
                )
                return {
                    "rule_id": rule_id,
                    "depends_on": depends_on_rule_id,
                    "dependency_id": f"{_PREFIX_DEPENDENCY}-existing",
                    "created_at": _utcnow().isoformat(),
                    "provenance_hash": _build_sha256(
                        {"rule_id": rule_id, "depends_on": depends_on_rule_id}
                    ),
                    "status": "already_exists",
                }

            # Tentatively add the dependency, then check for cycles
            self._dependencies[rule_id].add(depends_on_rule_id)

            # Collect all nodes involved in the dependency graph
            all_nodes: Set[str] = set()
            for node, deps in self._dependencies.items():
                all_nodes.add(node)
                all_nodes.update(deps)

            cycles = self._detect_cycles_in_graph(all_nodes)
            if cycles:
                # Rollback
                self._dependencies[rule_id].discard(depends_on_rule_id)
                if not self._dependencies[rule_id]:
                    del self._dependencies[rule_id]
                vre_cycles_detected_total.inc()
                raise ValueError(
                    f"Adding dependency '{rule_id}' -> "
                    f"'{depends_on_rule_id}' would create a cycle: "
                    f"{cycles[0]}"
                )

        # Build result
        dep_id = _generate_id(_PREFIX_DEPENDENCY)
        now = _utcnow().isoformat()
        result = {
            "rule_id": rule_id,
            "depends_on": depends_on_rule_id,
            "dependency_id": dep_id,
            "created_at": now,
            "provenance_hash": _build_sha256(
                {"rule_id": rule_id, "depends_on": depends_on_rule_id}
            ),
            "status": "created",
        }

        # Record provenance
        self._provenance.record(
            entity_type="rule_dependency",
            entity_id=dep_id,
            action="dependency_added",
            metadata=result,
        )

        # Metrics
        vre_dependencies_added_total.inc()
        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="add_rule_dependency"
        ).observe(duration)

        logger.info(
            "Added dependency: '%s' depends on '%s' (id=%s) in %.4fs",
            rule_id,
            depends_on_rule_id,
            dep_id,
            duration,
        )
        return result

    def get_evaluation_order(self, rule_ids: List[str]) -> List[str]:
        """Compute evaluation order using topological sort (Kahn's algorithm).

        Given a list of rule IDs, returns them sorted such that all
        dependencies of a rule appear before the rule itself. Rules
        without dependencies maintain their original relative order.

        Args:
            rule_ids: List of rule IDs to sort.

        Returns:
            Topologically sorted list of rule IDs.

        Raises:
            ValueError: If a cycle is detected among the given rules.
        """
        start_time = time.monotonic()

        with self._lock:
            # Build subgraph for only the requested rules
            rule_id_set = set(rule_ids)

            # adjacency: rule -> set of rules it depends on (predecessors)
            in_degree: Dict[str, int] = {rid: 0 for rid in rule_ids}
            adjacency: Dict[str, List[str]] = defaultdict(list)

            for rid in rule_ids:
                deps = self._dependencies.get(rid, set())
                for dep in deps:
                    if dep in rule_id_set:
                        # dep -> rid (dep must come before rid)
                        adjacency[dep].append(rid)
                        in_degree[rid] = in_degree.get(rid, 0) + 1

            # Kahn's algorithm
            queue: deque[str] = deque()
            for rid in rule_ids:
                if in_degree.get(rid, 0) == 0:
                    queue.append(rid)

            sorted_result: List[str] = []
            while queue:
                node = queue.popleft()
                sorted_result.append(node)
                for neighbor in adjacency.get(node, []):
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

        # Check for cycles (not all nodes processed)
        if len(sorted_result) != len(rule_ids):
            remaining = set(rule_ids) - set(sorted_result)
            raise ValueError(
                f"Cycle detected among rules: {sorted(remaining)}. "
                f"Cannot determine evaluation order."
            )

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="get_evaluation_order"
        ).observe(duration)

        logger.debug(
            "Computed evaluation order for %d rules in %.4fs",
            len(rule_ids),
            duration,
        )
        return sorted_result

    def detect_dependency_cycles(self, rule_ids: List[str]) -> List[List[str]]:
        """Detect cycles in the dependency graph among the given rules.

        Uses iterative DFS with explicit stack to avoid recursion stack
        overflow on deeply nested graphs.

        Args:
            rule_ids: List of rule IDs to check for cycles.

        Returns:
            List of cycles, where each cycle is a list of rule IDs
            forming a circular dependency. Returns an empty list if
            no cycles are found.
        """
        start_time = time.monotonic()

        with self._lock:
            rule_id_set = set(rule_ids)
            cycles = self._detect_cycles_in_subgraph(rule_id_set)

        if cycles:
            vre_cycles_detected_total.inc()

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="detect_dependency_cycles"
        ).observe(duration)

        logger.info(
            "Cycle detection for %d rules: found %d cycles in %.4fs",
            len(rule_ids),
            len(cycles),
            duration,
        )
        return cycles

    # ======================================================================
    # UTILITY METHODS (20-22)
    # ======================================================================

    def compare_rule_sets(
        self, set_id_a: str, set_id_b: str
    ) -> Dict[str, Any]:
        """Compare two rule sets and return the diff.

        Computes the set difference of rule IDs and compound rule IDs
        between the two rule sets, and identifies metadata differences.

        Args:
            set_id_a: First rule set ID.
            set_id_b: Second rule set ID.

        Returns:
            Dictionary with keys:
            ``set_id_a``, ``set_id_b``,
            ``added_rules`` (in B but not A),
            ``removed_rules`` (in A but not B),
            ``common_rules`` (in both),
            ``added_compound_rules``,
            ``removed_compound_rules``,
            ``common_compound_rules``,
            ``metadata_diffs`` (list of changed field names),
            ``a_version``, ``b_version``.

        Raises:
            ValueError: If either set_id does not exist.
        """
        start_time = time.monotonic()

        with self._lock:
            if set_id_a not in self._rule_sets:
                raise ValueError(
                    f"Rule set '{set_id_a}' does not exist."
                )
            if set_id_b not in self._rule_sets:
                raise ValueError(
                    f"Rule set '{set_id_b}' does not exist."
                )

            set_a = self._rule_sets[set_id_a]
            set_b = self._rule_sets[set_id_b]

        # Rule ID diff
        rules_a = set(set_a["rule_ids"])
        rules_b = set(set_b["rule_ids"])
        added_rules = sorted(rules_b - rules_a)
        removed_rules = sorted(rules_a - rules_b)
        common_rules = sorted(rules_a & rules_b)

        # Compound rule ID diff
        compound_a = set(set_a["compound_rule_ids"])
        compound_b = set(set_b["compound_rule_ids"])
        added_compound = sorted(compound_b - compound_a)
        removed_compound = sorted(compound_a - compound_b)
        common_compound = sorted(compound_a & compound_b)

        # Metadata diff
        metadata_diffs: List[str] = []
        compare_fields = ["name", "description", "tags", "sla_thresholds", "status"]
        for field_name in compare_fields:
            if set_a.get(field_name) != set_b.get(field_name):
                metadata_diffs.append(field_name)

        result = {
            "set_id_a": set_id_a,
            "set_id_b": set_id_b,
            "a_version": set_a["version"],
            "b_version": set_b["version"],
            "added_rules": added_rules,
            "removed_rules": removed_rules,
            "common_rules": common_rules,
            "added_compound_rules": added_compound,
            "removed_compound_rules": removed_compound,
            "common_compound_rules": common_compound,
            "metadata_diffs": metadata_diffs,
            "total_changes": (
                len(added_rules) + len(removed_rules)
                + len(added_compound) + len(removed_compound)
                + len(metadata_diffs)
            ),
        }

        duration = time.monotonic() - start_time
        vre_composer_operation_duration.labels(
            operation="compare_rule_sets"
        ).observe(duration)

        logger.info(
            "Compared rule sets '%s' (v%s) vs '%s' (v%s): "
            "+%d/-%d rules, +%d/-%d compound, %d metadata diffs in %.4fs",
            set_id_a,
            set_a["version"],
            set_id_b,
            set_b["version"],
            len(added_rules),
            len(removed_rules),
            len(added_compound),
            len(removed_compound),
            len(metadata_diffs),
            duration,
        )
        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Return statistics about the composer's current state.

        Returns:
            Dictionary with keys: ``compound_rules_count``,
            ``rule_sets_count``, ``templates_count``,
            ``dependencies_count``, ``total_dependency_edges``,
            ``avg_rules_per_set``, ``max_nesting_depth_configured``,
            ``max_nesting_depth_actual``, ``rule_sets_with_parents``,
            ``rule_sets_from_templates``.
        """
        with self._lock:
            compound_count = len(self._compound_rules)
            set_count = len(self._rule_sets)
            template_count = len(self._templates)
            dep_node_count = len(self._dependencies)
            total_edges = sum(
                len(deps) for deps in self._dependencies.values()
            )

            # Average rules per set
            if set_count > 0:
                total_rules = sum(
                    rs["rule_count"] for rs in self._rule_sets.values()
                )
                avg_rules = total_rules / set_count
            else:
                avg_rules = 0.0

            # Max actual nesting depth
            max_actual_depth = 0
            for cmp in self._compound_rules.values():
                depth = cmp.get("nesting_depth", 0)
                if depth > max_actual_depth:
                    max_actual_depth = depth

            # Count sets with parents
            sets_with_parents = sum(
                1 for rs in self._rule_sets.values()
                if rs.get("parent_set_id") is not None
            )

            # Count sets from templates
            sets_from_templates = sum(
                1 for rs in self._rule_sets.values()
                if rs.get("template_id") is not None
            )

        return {
            "compound_rules_count": compound_count,
            "rule_sets_count": set_count,
            "templates_count": template_count,
            "dependencies_count": dep_node_count,
            "total_dependency_edges": total_edges,
            "avg_rules_per_set": round(avg_rules, 2),
            "max_nesting_depth_configured": self._max_nesting_depth,
            "max_nesting_depth_actual": max_actual_depth,
            "rule_sets_with_parents": sets_with_parents,
            "rule_sets_from_templates": sets_from_templates,
        }

    def clear(self) -> None:
        """Clear all in-memory state.

        Removes all compound rules, rule sets, version histories,
        templates, and dependencies. Primarily intended for testing.
        """
        with self._lock:
            self._compound_rules.clear()
            self._rule_sets.clear()
            self._rule_set_versions.clear()
            self._templates.clear()
            self._dependencies.clear()

        # Reset metrics
        vre_active_compound_rules.set(0)
        vre_active_rule_sets_gauge.set(0)

        logger.info("RuleComposerEngine cleared all in-memory state.")

    # ======================================================================
    # INTERNAL HELPERS
    # ======================================================================

    def _rule_exists(self, rule_id: str) -> bool:
        """Check if a rule exists in the registry or compound rules.

        Must be called while holding ``self._lock``.

        Args:
            rule_id: The rule ID to check.

        Returns:
            ``True`` if the rule exists in the registry (via
            ``get_rule()``) or in the compound rules store.
        """
        # Check compound rules first (O(1) lookup)
        if rule_id in self._compound_rules:
            return True

        # Check registry (duck-typed: expects get_rule(rule_id) method)
        try:
            result = self._registry.get_rule(rule_id)
            return result is not None
        except (AttributeError, TypeError):
            logger.warning(
                "Registry does not support get_rule(); "
                "assuming rule '%s' does not exist.",
                rule_id,
            )
            return False

    def _calculate_nesting_depth(self, rule_ids: List[str]) -> int:
        """Calculate the maximum nesting depth among a set of rule IDs.

        Atomic rules have depth 0. Compound rules have depth equal to
        their stored ``nesting_depth`` value.

        Must be called while holding ``self._lock``.

        Args:
            rule_ids: List of rule IDs (atomic or compound).

        Returns:
            Maximum nesting depth found among the given rules.
        """
        max_depth = 0
        for rule_id in rule_ids:
            cmp = self._compound_rules.get(rule_id)
            if cmp is not None:
                depth = cmp.get("nesting_depth", 0)
                if depth > max_depth:
                    max_depth = depth
        return max_depth

    def _flatten_recursive(
        self,
        rule_id: str,
        seen: Set[str],
        result: List[str],
    ) -> None:
        """Recursively flatten a compound or atomic rule to atomic IDs.

        Must be called while holding ``self._lock``.

        Args:
            rule_id: The rule ID to flatten.
            seen: Set of already-visited rule IDs for deduplication.
            result: Accumulator list for atomic rule IDs.
        """
        if rule_id in seen:
            return

        cmp = self._compound_rules.get(rule_id)
        if cmp is None:
            # This is an atomic rule
            seen.add(rule_id)
            result.append(rule_id)
            return

        # This is a compound rule; recurse into children
        seen.add(rule_id)
        for child_id in cmp["rule_ids"]:
            self._flatten_recursive(child_id, seen, result)

    def _validate_tree(
        self,
        compound_id: str,
        visited: Set[str],
        errors: List[str],
        warnings: List[str],
        current_depth: int,
    ) -> int:
        """Recursively validate a compound rule tree structure.

        Must be called while holding ``self._lock``.

        Args:
            compound_id: The compound rule ID to validate.
            visited: Set of already-visited compound IDs for cycle detection.
            errors: Accumulator for error messages.
            warnings: Accumulator for warning messages.
            current_depth: Current recursion depth.

        Returns:
            Maximum depth reached during validation.
        """
        if compound_id in visited:
            errors.append(
                f"Circular reference detected at compound rule "
                f"'{compound_id}'."
            )
            return current_depth

        visited.add(compound_id)
        cmp = self._compound_rules.get(compound_id)

        if cmp is None:
            errors.append(
                f"Compound rule '{compound_id}' does not exist."
            )
            return current_depth

        # Check depth limit
        if current_depth >= self._max_nesting_depth:
            errors.append(
                f"Nesting depth {current_depth} exceeds maximum "
                f"of {self._max_nesting_depth} at compound "
                f"'{compound_id}'."
            )
            return current_depth

        # Validate operator arity
        operator = cmp["operator"]
        child_count = len(cmp["rule_ids"])
        if operator == "NOT" and child_count != 1:
            errors.append(
                f"NOT operator at '{compound_id}' has {child_count} "
                f"children; expected exactly 1."
            )
        elif operator in ("AND", "OR") and child_count < 2:
            errors.append(
                f"{operator} operator at '{compound_id}' has "
                f"{child_count} children; expected 2 or more."
            )

        # Warn on large fan-out
        if child_count > 20:
            warnings.append(
                f"Compound rule '{compound_id}' has {child_count} "
                f"children; consider splitting for readability."
            )

        # Recurse into compound children
        max_depth = current_depth
        for child_id in cmp["rule_ids"]:
            if child_id in self._compound_rules:
                child_depth = self._validate_tree(
                    child_id, visited, errors, warnings, current_depth + 1
                )
                if child_depth > max_depth:
                    max_depth = child_depth
            else:
                # Atomic rule; validate existence
                if not self._rule_exists(child_id):
                    errors.append(
                        f"Child rule '{child_id}' of compound "
                        f"'{compound_id}' does not exist."
                    )

        return max_depth

    def _detect_cycles_in_graph(
        self, nodes: Set[str]
    ) -> List[List[str]]:
        """Detect cycles in the full dependency graph using iterative DFS.

        Must be called while holding ``self._lock``.

        Args:
            nodes: Set of all node IDs in the graph.

        Returns:
            List of cycles found, each represented as a list of node IDs.
        """
        return self._detect_cycles_in_subgraph(nodes)

    def _detect_cycles_in_subgraph(
        self, nodes: Set[str]
    ) -> List[List[str]]:
        """Detect cycles using iterative DFS with three-color marking.

        Uses WHITE (unvisited), GRAY (in current path), BLACK (fully
        processed) coloring to identify back edges that form cycles.

        Must be called while holding ``self._lock``.

        Args:
            nodes: Set of node IDs to check.

        Returns:
            List of detected cycles.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {node: WHITE for node in nodes}
        parent: Dict[str, Optional[str]] = {node: None for node in nodes}
        cycles: List[List[str]] = []

        for start_node in nodes:
            if color.get(start_node, WHITE) != WHITE:
                continue

            # Iterative DFS using explicit stack
            # Stack entries: (node, iterator_over_successors, is_entering)
            stack: List[Tuple[str, Any, bool]] = [
                (start_node, None, True)
            ]

            while stack:
                node, successors_iter, is_entering = stack[-1]

                if is_entering:
                    color[node] = GRAY
                    deps = self._dependencies.get(node, set())
                    relevant_deps = [d for d in deps if d in nodes]
                    stack[-1] = (node, iter(relevant_deps), False)
                    continue

                try:
                    neighbor = next(successors_iter)
                    neighbor_color = color.get(neighbor, WHITE)

                    if neighbor_color == GRAY:
                        # Back edge found: reconstruct cycle
                        cycle = [neighbor]
                        for s_node, _, _ in reversed(stack):
                            cycle.append(s_node)
                            if s_node == neighbor:
                                break
                        cycle.reverse()
                        cycles.append(cycle)
                    elif neighbor_color == WHITE:
                        parent[neighbor] = node
                        stack.append((neighbor, None, True))

                except StopIteration:
                    color[node] = BLACK
                    stack.pop()

        return cycles

    def _resolve_sla_thresholds(
        self, sla_thresholds: Optional[Dict[str, float]]
    ) -> Dict[str, float]:
        """Resolve SLA thresholds with defaults for missing keys.

        Args:
            sla_thresholds: Caller-provided thresholds, or None.

        Returns:
            Complete SLA thresholds dictionary with all three keys.
        """
        defaults = copy.deepcopy(DEFAULT_SLA_THRESHOLDS)
        if sla_thresholds is None:
            return defaults
        # Merge caller values over defaults
        for key, value in sla_thresholds.items():
            if key in defaults:
                if not isinstance(value, (int, float)):
                    logger.warning(
                        "SLA threshold '%s' has non-numeric value %r; "
                        "using default.",
                        key,
                        value,
                    )
                    continue
                if not 0.0 <= value <= 1.0:
                    logger.warning(
                        "SLA threshold '%s' value %.4f outside [0, 1]; "
                        "using default.",
                        key,
                        value,
                    )
                    continue
                defaults[key] = float(value)
            else:
                # Accept additional custom thresholds
                if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
                    defaults[key] = float(value)
        return defaults

    def _validate_status_transition(
        self, current_status: str, new_status: str
    ) -> None:
        """Validate a rule set status transition.

        Allowed transitions:
            - ``active`` -> ``deprecated``
            - ``active`` -> ``archived``
            - ``deprecated`` -> ``archived``
            - ``draft`` -> ``active``

        Args:
            current_status: Current status.
            new_status: Desired new status.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed_transitions: Dict[str, Set[str]] = {
            "draft": {"active"},
            "active": {"deprecated", "archived"},
            "deprecated": {"archived"},
            "archived": set(),
        }

        allowed = allowed_transitions.get(current_status, set())
        if new_status not in allowed:
            raise ValueError(
                f"Invalid status transition: '{current_status}' -> "
                f"'{new_status}'. Allowed from '{current_status}': "
                f"{sorted(allowed) if allowed else 'none (terminal state)'}."
            )

    def _create_version_snapshot(
        self, rule_set: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a version history snapshot of a rule set.

        Args:
            rule_set: The current state of the rule set.

        Returns:
            Dictionary with version metadata and a snapshot of the
            rule set at this version.
        """
        return {
            "version": rule_set["version"],
            "snapshot": copy.deepcopy(rule_set),
            "timestamp": rule_set.get("updated_at", _utcnow().isoformat()),
            "provenance_hash": rule_set.get("provenance_hash", ""),
        }
