# -*- coding: utf-8 -*-
"""
Data Classifier - AGENT-FOUND-006: Access & Policy Guard

Classifies resources by data sensitivity level using built-in PII
detection patterns and domain-specific rules. Supports clearance
checking against the CLASSIFICATION_HIERARCHY.

Zero-Hallucination Guarantees:
    - Pattern-based classification is deterministic
    - No ML or probabilistic classification decisions
    - Classification hierarchy is a static lookup table
    - All results are reproducible

Example:
    >>> from greenlang.access_guard.classifier import DataClassifier
    >>> from greenlang.access_guard.models import Resource, Principal
    >>> classifier = DataClassifier()
    >>> level = classifier.classify(resource)
    >>> allowed, reason = classifier.check_clearance(principal, resource)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from greenlang.access_guard.models import (
    CLASSIFICATION_HIERARCHY,
    DataClassification,
    Principal,
    Resource,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in PII detection patterns
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: List[Dict[str, Any]] = [
    # PII patterns
    {
        "name": "ssn",
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "classification": DataClassification.RESTRICTED,
        "description": "US Social Security Number",
    },
    {
        "name": "passport",
        "pattern": r"\b[A-Z]{1,2}\d{6,9}\b",
        "classification": DataClassification.RESTRICTED,
        "description": "Passport number pattern",
    },
    {
        "name": "credit_card",
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "classification": DataClassification.RESTRICTED,
        "description": "Credit card number",
    },
    {
        "name": "email",
        "pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "classification": DataClassification.CONFIDENTIAL,
        "description": "Email address",
    },
    {
        "name": "phone",
        "pattern": r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "classification": DataClassification.CONFIDENTIAL,
        "description": "Phone number",
    },
    # Domain-specific keyword patterns
    {
        "name": "financial_keywords",
        "pattern": r"\b(?:bank_account|routing_number|tax_id|ein)\b",
        "classification": DataClassification.RESTRICTED,
        "description": "Financial identifier keywords",
    },
    {
        "name": "emission_keywords",
        "pattern": r"\b(?:scope_[123]_emission|ghg_inventory|carbon_footprint)\b",
        "classification": DataClassification.INTERNAL,
        "description": "Emission data keywords",
    },
    {
        "name": "compliance_keywords",
        "pattern": r"\b(?:sox_report|audit_finding|regulatory_violation)\b",
        "classification": DataClassification.CONFIDENTIAL,
        "description": "Compliance-sensitive keywords",
    },
    {
        "name": "personal_keywords",
        "pattern": r"\b(?:date_of_birth|dob|social_security|ssn|passport_number)\b",
        "classification": DataClassification.RESTRICTED,
        "description": "Personal data keywords",
    },
]


# ---------------------------------------------------------------------------
# DataClassifier
# ---------------------------------------------------------------------------


class DataClassifier:
    """Classifies resources by data sensitivity using pattern matching.

    Uses a two-phase classification approach:
    1. Resource-type based domain classification
    2. Content-based PII pattern detection (elevates classification)

    The classifier is thread-safe. Custom patterns can be registered
    at runtime via ``register_pattern()``.

    Attributes:
        _patterns: List of classification patterns (built-in + custom).

    Example:
        >>> classifier = DataClassifier()
        >>> level = classifier.classify(resource)
        >>> print(level)  # DataClassification.CONFIDENTIAL
    """

    def __init__(self) -> None:
        """Initialize the DataClassifier with built-in patterns."""
        self._patterns: List[Dict[str, Any]] = list(_BUILTIN_PATTERNS)
        self._compiled: Dict[str, re.Pattern[str]] = {}
        self._lock = threading.Lock()

        # Pre-compile all patterns
        for entry in self._patterns:
            self._compiled[entry["name"]] = re.compile(
                entry["pattern"], re.IGNORECASE,
            )

        logger.info(
            "DataClassifier initialized with %d patterns", len(self._patterns),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, resource: Resource) -> DataClassification:
        """Classify a resource based on type and content patterns.

        Phase 1: Domain-based classification using resource_type.
        Phase 2: Pattern-based PII scan on resource attributes
                 (can only elevate, never lower classification).

        Args:
            resource: The resource to classify.

        Returns:
            The determined DataClassification level.
        """
        classification = resource.classification

        # Phase 1: Domain-based elevation
        classification = self._classify_by_domain(resource, classification)

        # Phase 2: Pattern-based PII scan
        classification = self._classify_by_patterns(resource, classification)

        logger.debug(
            "Classified resource %s as %s",
            resource.resource_id, classification.value,
        )
        return classification

    def check_clearance(
        self, principal: Principal, resource: Resource,
    ) -> Tuple[bool, Optional[str]]:
        """Check if a principal has sufficient clearance for a resource.

        Compares the principal's clearance_level against the resource's
        classification using the CLASSIFICATION_HIERARCHY ordering.

        Args:
            principal: The requesting principal.
            resource: The resource being accessed.

        Returns:
            Tuple of (cleared, denial_reason). denial_reason is None
            when cleared is True.
        """
        principal_level = CLASSIFICATION_HIERARCHY.get(
            principal.clearance_level, 0,
        )
        resource_level = CLASSIFICATION_HIERARCHY.get(
            resource.classification, 0,
        )

        if resource_level > principal_level:
            reason = (
                f"Insufficient clearance: principal has "
                f"'{principal.clearance_level.value}' but resource "
                f"requires '{resource.classification.value}'"
            )
            return False, reason

        return True, None

    def get_classification_hierarchy(self) -> Dict[str, int]:
        """Get the classification hierarchy as a serializable dict.

        Returns:
            Dictionary mapping classification name to numeric level.
        """
        return {k.value: v for k, v in CLASSIFICATION_HIERARCHY.items()}

    def register_pattern(
        self,
        pattern: str,
        classification: DataClassification,
        name: Optional[str] = None,
        description: str = "",
    ) -> None:
        """Register a custom classification pattern at runtime.

        Args:
            pattern: Regex pattern string.
            classification: Classification to assign on match.
            name: Optional pattern name (auto-generated if None).
            description: Human-readable description.
        """
        entry_name = name or f"custom_{len(self._patterns)}"

        entry: Dict[str, Any] = {
            "name": entry_name,
            "pattern": pattern,
            "classification": classification,
            "description": description,
        }

        with self._lock:
            self._patterns.append(entry)
            self._compiled[entry_name] = re.compile(pattern, re.IGNORECASE)

        logger.info(
            "Registered classification pattern: %s -> %s",
            entry_name, classification.value,
        )

    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all registered classification patterns.

        Returns:
            List of pattern dictionaries with name, pattern, classification,
            and description.
        """
        results: List[Dict[str, Any]] = []
        for entry in self._patterns:
            results.append({
                "name": entry["name"],
                "pattern": entry["pattern"],
                "classification": entry["classification"].value
                    if isinstance(entry["classification"], DataClassification)
                    else entry["classification"],
                "description": entry.get("description", ""),
            })
        return results

    @property
    def count(self) -> int:
        """Return the number of registered patterns."""
        return len(self._patterns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_by_domain(
        self,
        resource: Resource,
        current: DataClassification,
    ) -> DataClassification:
        """Elevate classification based on resource type domain rules.

        Args:
            resource: The resource to classify.
            current: Current classification level.

        Returns:
            Possibly elevated classification level.
        """
        resource_type_lower = resource.resource_type.lower()
        current_level = CLASSIFICATION_HIERARCHY.get(current, 0)

        # Financial data
        if any(
            kw in resource_type_lower
            for kw in ("financial", "payment", "invoice", "billing")
        ):
            confidential_level = CLASSIFICATION_HIERARCHY[DataClassification.CONFIDENTIAL]
            if current_level < confidential_level:
                return DataClassification.CONFIDENTIAL

        # Emission data
        if any(
            kw in resource_type_lower
            for kw in ("emission", "ghg", "carbon")
        ):
            internal_level = CLASSIFICATION_HIERARCHY[DataClassification.INTERNAL]
            if current_level < internal_level:
                return DataClassification.INTERNAL

        # Compliance/audit data
        if any(
            kw in resource_type_lower
            for kw in ("compliance", "audit", "regulatory")
        ):
            confidential_level = CLASSIFICATION_HIERARCHY[DataClassification.CONFIDENTIAL]
            if current_level < confidential_level:
                return DataClassification.CONFIDENTIAL

        # Personal/PII data
        if any(
            kw in resource_type_lower
            for kw in ("personal", "pii", "employee", "hr")
        ):
            restricted_level = CLASSIFICATION_HIERARCHY[DataClassification.RESTRICTED]
            if current_level < restricted_level:
                return DataClassification.RESTRICTED

        return current

    def _classify_by_patterns(
        self,
        resource: Resource,
        current: DataClassification,
    ) -> DataClassification:
        """Elevate classification by scanning resource attributes with patterns.

        Scans the string representation of resource attributes against
        all registered patterns. Can only elevate, never lower.

        Args:
            resource: The resource to classify.
            current: Current classification level.

        Returns:
            Possibly elevated classification level.
        """
        current_level = CLASSIFICATION_HIERARCHY.get(current, 0)
        attributes_str = str(resource.attributes).lower()
        resource_id_str = resource.resource_id.lower()
        combined_text = f"{attributes_str} {resource_id_str}"

        for entry in self._patterns:
            pattern_classification = entry["classification"]
            pattern_level = CLASSIFICATION_HIERARCHY.get(pattern_classification, 0)

            # Only check if this pattern could elevate the classification
            if pattern_level <= current_level:
                continue

            compiled = self._compiled.get(entry["name"])
            if compiled and compiled.search(combined_text):
                current = pattern_classification
                current_level = pattern_level
                logger.debug(
                    "Pattern '%s' elevated classification to %s",
                    entry["name"], current.value,
                )

        return current


__all__ = [
    "DataClassifier",
]
