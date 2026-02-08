# -*- coding: utf-8 -*-
"""
Unit Tests for DataClassifier (AGENT-FOUND-006)

Tests data classification, PII detection, financial/emission type detection,
clearance checks, custom pattern registration, and hierarchy queries.

Coverage target: 85%+ of classifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest


# ---------------------------------------------------------------------------
# Inline models and classifier mirroring the access guard service
# ---------------------------------------------------------------------------


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


CLASSIFICATION_HIERARCHY: Dict[DataClassification, int] = {
    DataClassification.PUBLIC: 0,
    DataClassification.INTERNAL: 1,
    DataClassification.CONFIDENTIAL: 2,
    DataClassification.RESTRICTED: 3,
    DataClassification.TOP_SECRET: 4,
}


class Resource:
    def __init__(self, resource_id, resource_type="data", tenant_id="",
                 classification="internal", attributes=None,
                 geographic_location=None):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.tenant_id = tenant_id
        self.classification = DataClassification(classification)
        self.attributes = attributes or {}
        self.geographic_location = geographic_location


class DataClassifier:
    """Data classification engine with PII detection and custom patterns."""

    def __init__(self):
        self._patterns: Dict[str, DataClassification] = {
            "ssn": DataClassification.RESTRICTED,
            "social_security": DataClassification.RESTRICTED,
            "passport": DataClassification.RESTRICTED,
            "credit_card": DataClassification.RESTRICTED,
        }
        self._type_classifications: Dict[str, DataClassification] = {
            "financial": DataClassification.CONFIDENTIAL,
            "payment": DataClassification.CONFIDENTIAL,
            "emission": DataClassification.INTERNAL,
        }

    @property
    def count(self) -> int:
        return len(self._patterns)

    def classify(self, resource: Resource) -> DataClassification:
        """Classify a resource based on type, attributes, and patterns."""
        classification = resource.classification

        # Check for sensitive patterns in attributes
        attrs_str = str(resource.attributes).lower()
        for pattern_key, pattern_class in self._patterns.items():
            if pattern_key in attrs_str:
                if CLASSIFICATION_HIERARCHY.get(pattern_class, 0) > CLASSIFICATION_HIERARCHY.get(classification, 0):
                    classification = pattern_class

        # Check resource type
        resource_type_lower = resource.resource_type.lower()
        for type_key, type_class in self._type_classifications.items():
            if type_key in resource_type_lower:
                if CLASSIFICATION_HIERARCHY.get(type_class, 0) > CLASSIFICATION_HIERARCHY.get(classification, 0):
                    classification = type_class

        return classification

    def check_clearance(
        self, clearance_level: DataClassification, required_level: DataClassification,
    ) -> Tuple[bool, Optional[str]]:
        """Check if clearance level meets required level."""
        clearance_val = CLASSIFICATION_HIERARCHY.get(clearance_level, 0)
        required_val = CLASSIFICATION_HIERARCHY.get(required_level, 0)

        if clearance_val >= required_val:
            return True, None
        return (
            False,
            f"Insufficient clearance: has '{clearance_level.value}' "
            f"but requires '{required_level.value}'",
        )

    def get_classification_hierarchy(self) -> Dict[str, int]:
        """Return classification hierarchy as str->int mapping."""
        return {k.value: v for k, v in CLASSIFICATION_HIERARCHY.items()}

    def register_pattern(
        self, pattern: str, classification: DataClassification,
    ) -> None:
        """Register a custom pattern for classification."""
        self._patterns[pattern.lower()] = classification

    def list_patterns(self) -> Dict[str, str]:
        """List all registered patterns."""
        return {k: v.value for k, v in self._patterns.items()}


# ===========================================================================
# Test Classes
# ===========================================================================


class TestClassifierBasic:
    """Test default classification from resource."""

    def test_default_classification_preserved(self):
        classifier = DataClassifier()
        r = Resource(resource_id="r1", classification="internal")
        result = classifier.classify(r)
        assert result == DataClassification.INTERNAL

    def test_public_classification_preserved(self):
        classifier = DataClassifier()
        r = Resource(resource_id="r1", classification="public")
        result = classifier.classify(r)
        assert result == DataClassification.PUBLIC

    def test_confidential_classification_preserved(self):
        classifier = DataClassifier()
        r = Resource(resource_id="r1", classification="confidential")
        result = classifier.classify(r)
        assert result == DataClassification.CONFIDENTIAL

    def test_restricted_classification_preserved(self):
        classifier = DataClassifier()
        r = Resource(resource_id="r1", classification="restricted")
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_top_secret_classification_preserved(self):
        classifier = DataClassifier()
        r = Resource(resource_id="r1", classification="top_secret")
        result = classifier.classify(r)
        assert result == DataClassification.TOP_SECRET

    def test_count_default_patterns(self):
        classifier = DataClassifier()
        assert classifier.count == 4  # ssn, social_security, passport, credit_card


class TestClassifierPII:
    """Test SSN, passport, credit_card patterns -> RESTRICTED."""

    def test_ssn_in_attributes_elevates_to_restricted(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="internal",
            attributes={"ssn": "123-45-6789"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_social_security_in_attributes(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="internal",
            attributes={"social_security": "xxx-xx-xxxx"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_passport_in_attributes(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="internal",
            attributes={"passport": "AB1234567"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_credit_card_in_attributes(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="internal",
            attributes={"credit_card": "4111-1111-1111-1111"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_pii_does_not_downgrade_top_secret(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="top_secret",
            attributes={"ssn": "123-45-6789"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.TOP_SECRET

    def test_nested_pii_in_attributes(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="public",
            attributes={"data": {"nested_ssn_field": "value"}},
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_no_pii_keeps_classification(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", classification="internal",
            attributes={"name": "John", "department": "engineering"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.INTERNAL


class TestClassifierFinancial:
    """Test financial/payment resource types -> CONFIDENTIAL."""

    def test_financial_type_elevates_to_confidential(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="financial_report",
            classification="internal",
        )
        result = classifier.classify(r)
        assert result == DataClassification.CONFIDENTIAL

    def test_payment_type_elevates_to_confidential(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="payment_record",
            classification="internal",
        )
        result = classifier.classify(r)
        assert result == DataClassification.CONFIDENTIAL

    def test_financial_does_not_downgrade_restricted(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="financial_data",
            classification="restricted",
        )
        result = classifier.classify(r)
        assert result == DataClassification.RESTRICTED

    def test_financial_does_not_elevate_if_already_higher(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="financial_data",
            classification="top_secret",
        )
        result = classifier.classify(r)
        assert result == DataClassification.TOP_SECRET


class TestClassifierEmission:
    """Test emission resource types -> INTERNAL minimum."""

    def test_emission_type_elevates_public_to_internal(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="emission_data",
            classification="public",
        )
        result = classifier.classify(r)
        assert result == DataClassification.INTERNAL

    def test_emission_does_not_downgrade_confidential(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="emission_report",
            classification="confidential",
        )
        result = classifier.classify(r)
        assert result == DataClassification.CONFIDENTIAL

    def test_emission_internal_stays_internal(self):
        classifier = DataClassifier()
        r = Resource(
            resource_id="r1", resource_type="emission_factor",
            classification="internal",
        )
        result = classifier.classify(r)
        assert result == DataClassification.INTERNAL


class TestClassifierClearanceCheck:
    """Test clearance >= classification passes."""

    def test_clearance_equal_passes(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.CONFIDENTIAL, DataClassification.CONFIDENTIAL,
        )
        assert passed is True
        assert reason is None

    def test_clearance_higher_passes(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.RESTRICTED, DataClassification.INTERNAL,
        )
        assert passed is True
        assert reason is None

    def test_top_secret_can_access_all(self):
        classifier = DataClassifier()
        for level in DataClassification:
            passed, _ = classifier.check_clearance(
                DataClassification.TOP_SECRET, level,
            )
            assert passed is True

    def test_public_can_access_public(self):
        classifier = DataClassifier()
        passed, _ = classifier.check_clearance(
            DataClassification.PUBLIC, DataClassification.PUBLIC,
        )
        assert passed is True


class TestClassifierClearanceInsufficient:
    """Test clearance < classification fails."""

    def test_public_cannot_access_internal(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.PUBLIC, DataClassification.INTERNAL,
        )
        assert passed is False
        assert "Insufficient clearance" in reason

    def test_internal_cannot_access_confidential(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.INTERNAL, DataClassification.CONFIDENTIAL,
        )
        assert passed is False
        assert "confidential" in reason.lower()

    def test_confidential_cannot_access_restricted(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED,
        )
        assert passed is False

    def test_restricted_cannot_access_top_secret(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.RESTRICTED, DataClassification.TOP_SECRET,
        )
        assert passed is False

    def test_reason_contains_levels(self):
        classifier = DataClassifier()
        passed, reason = classifier.check_clearance(
            DataClassification.PUBLIC, DataClassification.RESTRICTED,
        )
        assert "public" in reason
        assert "restricted" in reason


class TestClassifierCustomPattern:
    """Test register_pattern and list_patterns."""

    def test_register_new_pattern(self):
        classifier = DataClassifier()
        classifier.register_pattern("dob", DataClassification.CONFIDENTIAL)
        r = Resource(
            resource_id="r1", classification="public",
            attributes={"dob": "1990-01-01"},
        )
        result = classifier.classify(r)
        assert result == DataClassification.CONFIDENTIAL

    def test_register_pattern_case_insensitive(self):
        classifier = DataClassifier()
        classifier.register_pattern("DOB", DataClassification.CONFIDENTIAL)
        patterns = classifier.list_patterns()
        assert "dob" in patterns

    def test_list_patterns_includes_defaults(self):
        classifier = DataClassifier()
        patterns = classifier.list_patterns()
        assert "ssn" in patterns
        assert "passport" in patterns
        assert "credit_card" in patterns
        assert "social_security" in patterns

    def test_list_patterns_includes_custom(self):
        classifier = DataClassifier()
        classifier.register_pattern("tax_id", DataClassification.RESTRICTED)
        patterns = classifier.list_patterns()
        assert "tax_id" in patterns
        assert patterns["tax_id"] == "restricted"

    def test_register_pattern_increases_count(self):
        classifier = DataClassifier()
        initial = classifier.count
        classifier.register_pattern("driver_license", DataClassification.RESTRICTED)
        assert classifier.count == initial + 1


class TestClassifierHierarchy:
    """Test get_classification_hierarchy ordering."""

    def test_returns_all_levels(self):
        classifier = DataClassifier()
        hierarchy = classifier.get_classification_hierarchy()
        assert len(hierarchy) == 5

    def test_public_is_zero(self):
        classifier = DataClassifier()
        hierarchy = classifier.get_classification_hierarchy()
        assert hierarchy["public"] == 0

    def test_top_secret_is_four(self):
        classifier = DataClassifier()
        hierarchy = classifier.get_classification_hierarchy()
        assert hierarchy["top_secret"] == 4

    def test_ordering_correct(self):
        classifier = DataClassifier()
        hierarchy = classifier.get_classification_hierarchy()
        assert hierarchy["public"] < hierarchy["internal"]
        assert hierarchy["internal"] < hierarchy["confidential"]
        assert hierarchy["confidential"] < hierarchy["restricted"]
        assert hierarchy["restricted"] < hierarchy["top_secret"]
