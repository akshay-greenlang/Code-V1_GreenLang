# -*- coding: utf-8 -*-
"""
Test suite for audit_trail_lineage models - AGENT-MRV-030.

Tests all enumerations, dataclass models, and serialization helpers
for the Audit Trail & Lineage Agent (GL-MRV-X-042).

Coverage:
- AuditEventType enum (12 values)
- AuditEventRecord dataclass (frozen, to_dict, Decimal serialization)
- _decimal_serializer helper (Decimal, datetime, Enum, unsupported)
- _VALID_EVENT_TYPES, _VALID_SCOPES, _VALID_CATEGORIES sets
- Constants: AGENT_ID, ENGINE_ID, GENESIS_HASH, TABLE_PREFIX
- Module-level __init__.py metadata constants and helper functions

Target: ~120 tests, 85%+ coverage.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.audit_trail_lineage.audit_event_engine import (
        AuditEventType,
        AuditEventRecord,
        _decimal_serializer,
        _VALID_EVENT_TYPES,
        _VALID_SCOPES,
        _VALID_CATEGORIES,
        AGENT_ID as ENGINE_AGENT_ID,
        AGENT_COMPONENT as ENGINE_AGENT_COMPONENT,
        ENGINE_ID,
        ENGINE_VERSION,
        TABLE_PREFIX as ENGINE_TABLE_PREFIX,
        GENESIS_HASH,
        HASH_ALGORITHM,
        ENCODING,
        _ZERO,
        _ONE,
        _QUANT_2DP,
        _QUANT_4DP,
        _DEFAULT_QUERY_LIMIT,
        _DEFAULT_QUERY_OFFSET,
        _MAX_QUERY_LIMIT,
        _MAX_BATCH_SIZE,
    )
    ENGINE_MODELS_AVAILABLE = True
except ImportError:
    ENGINE_MODELS_AVAILABLE = False

try:
    from greenlang.agents.mrv.audit_trail_lineage import (
        AGENT_ID,
        AGENT_COMPONENT,
        VERSION,
        TABLE_PREFIX,
        get_version,
        get_agent_info,
    )
    INIT_AVAILABLE = True
except ImportError:
    INIT_AVAILABLE = False

_SKIP_ENGINE = pytest.mark.skipif(
    not ENGINE_MODELS_AVAILABLE,
    reason="audit_event_engine models not available",
)

_SKIP_INIT = pytest.mark.skipif(
    not INIT_AVAILABLE,
    reason="audit_trail_lineage __init__ not available",
)


# ==============================================================================
# MODULE-LEVEL METADATA CONSTANTS TESTS
# ==============================================================================


@_SKIP_INIT
class TestModuleMetadata:
    """Test __init__.py metadata constants and helper functions."""

    def test_agent_id_value(self):
        """Test AGENT_ID is GL-MRV-X-042."""
        assert AGENT_ID == "GL-MRV-X-042"

    def test_agent_component_value(self):
        """Test AGENT_COMPONENT is AGENT-MRV-030."""
        assert AGENT_COMPONENT == "AGENT-MRV-030"

    def test_version_value(self):
        """Test VERSION is 1.0.0."""
        assert VERSION == "1.0.0"

    def test_table_prefix_value(self):
        """Test TABLE_PREFIX is gl_atl_."""
        assert TABLE_PREFIX == "gl_atl_"

    def test_table_prefix_ends_with_underscore(self):
        """Test TABLE_PREFIX ends with underscore."""
        assert TABLE_PREFIX.endswith("_")

    def test_get_version_returns_string(self):
        """Test get_version() returns a string."""
        v = get_version()
        assert isinstance(v, str)
        assert v == "1.0.0"

    def test_get_version_semver_format(self):
        """Test get_version() returns valid semver format."""
        v = get_version()
        parts = v.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_get_agent_info_returns_dict(self):
        """Test get_agent_info() returns a dictionary."""
        info = get_agent_info()
        assert isinstance(info, dict)

    def test_get_agent_info_agent_id(self):
        """Test get_agent_info() contains correct agent_id."""
        info = get_agent_info()
        assert info["agent_id"] == "GL-MRV-X-042"

    def test_get_agent_info_component(self):
        """Test get_agent_info() contains correct component."""
        info = get_agent_info()
        assert info["component"] == "AGENT-MRV-030"

    def test_get_agent_info_version(self):
        """Test get_agent_info() contains correct version."""
        info = get_agent_info()
        assert info["version"] == "1.0.0"

    def test_get_agent_info_table_prefix(self):
        """Test get_agent_info() contains correct table_prefix."""
        info = get_agent_info()
        assert info["table_prefix"] == "gl_atl_"

    def test_get_agent_info_package(self):
        """Test get_agent_info() contains correct package."""
        info = get_agent_info()
        assert info["package"] == "greenlang.agents.mrv.audit_trail_lineage"

    def test_get_agent_info_scope(self):
        """Test get_agent_info() scope covers all three scopes."""
        info = get_agent_info()
        assert "Scope 1" in info["scope"]
        assert "2" in info["scope"]
        assert "3" in info["scope"]

    def test_get_agent_info_role(self):
        """Test get_agent_info() role is Cross-Cutting."""
        info = get_agent_info()
        assert "Cross-Cutting" in info["role"]

    def test_get_agent_info_engines_count(self):
        """Test get_agent_info() lists 7 engines."""
        info = get_agent_info()
        assert len(info["engines"]) == 7

    def test_get_agent_info_engines_names(self):
        """Test get_agent_info() engine names are correct."""
        info = get_agent_info()
        expected_engines = [
            "AuditEventEngine",
            "LineageGraphEngine",
            "EvidencePackagerEngine",
            "ComplianceTracerEngine",
            "ChangeDetectorEngine",
            "ComplianceCheckerEngine",
            "AuditTrailPipelineEngine",
        ]
        assert info["engines"] == expected_engines

    def test_get_agent_info_compliance_frameworks(self):
        """Test get_agent_info() lists compliance frameworks."""
        info = get_agent_info()
        frameworks = info["compliance_frameworks"]
        assert isinstance(frameworks, list)
        assert len(frameworks) >= 9

    def test_get_agent_info_compliance_has_ghg_protocol(self):
        """Test compliance frameworks include GHG Protocol."""
        info = get_agent_info()
        assert "GHG Protocol Corporate Standard" in info["compliance_frameworks"]

    def test_get_agent_info_compliance_has_iso(self):
        """Test compliance frameworks include ISO 14064."""
        info = get_agent_info()
        assert "ISO 14064-1:2018" in info["compliance_frameworks"]

    def test_get_agent_info_compliance_has_csrd(self):
        """Test compliance frameworks include CSRD ESRS E1."""
        info = get_agent_info()
        assert "CSRD ESRS E1" in info["compliance_frameworks"]

    def test_get_agent_info_capabilities(self):
        """Test get_agent_info() lists capabilities."""
        info = get_agent_info()
        assert isinstance(info["capabilities"], list)
        assert len(info["capabilities"]) >= 7

    def test_get_agent_info_capability_hash_chain(self):
        """Test capabilities include SHA-256 hash-chained events."""
        info = get_agent_info()
        sha_cap = [c for c in info["capabilities"] if "SHA-256" in c]
        assert len(sha_cap) >= 1

    def test_get_agent_info_capability_lineage(self):
        """Test capabilities include lineage DAG traversal."""
        info = get_agent_info()
        dag_cap = [c for c in info["capabilities"] if "lineage" in c.lower() or "DAG" in c]
        assert len(dag_cap) >= 1


# ==============================================================================
# ENGINE CONSTANTS TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestEngineConstants:
    """Test constants defined in audit_event_engine module."""

    def test_engine_agent_id(self):
        """Test ENGINE AGENT_ID matches module-level constant."""
        assert ENGINE_AGENT_ID == "GL-MRV-X-042"

    def test_engine_agent_component(self):
        """Test ENGINE AGENT_COMPONENT."""
        assert ENGINE_AGENT_COMPONENT == "AGENT-MRV-030"

    def test_engine_id(self):
        """Test ENGINE_ID follows naming convention."""
        assert ENGINE_ID == "gl_atl_audit_event_engine"
        assert ENGINE_ID.startswith("gl_atl_")

    def test_engine_version(self):
        """Test ENGINE_VERSION is 1.0.0."""
        assert ENGINE_VERSION == "1.0.0"

    def test_engine_table_prefix(self):
        """Test ENGINE TABLE_PREFIX is gl_atl_."""
        assert ENGINE_TABLE_PREFIX == "gl_atl_"

    def test_genesis_hash(self):
        """Test GENESIS_HASH is the expected seed value."""
        assert GENESIS_HASH == "greenlang-atl-genesis-v1"

    def test_genesis_hash_nonempty(self):
        """Test GENESIS_HASH is non-empty."""
        assert len(GENESIS_HASH) >= 5

    def test_hash_algorithm(self):
        """Test HASH_ALGORITHM is sha256."""
        assert HASH_ALGORITHM == "sha256"

    def test_encoding(self):
        """Test ENCODING is utf-8."""
        assert ENCODING == "utf-8"

    def test_decimal_zero(self):
        """Test _ZERO is Decimal('0')."""
        assert _ZERO == Decimal("0")

    def test_decimal_one(self):
        """Test _ONE is Decimal('1')."""
        assert _ONE == Decimal("1")

    def test_quant_2dp(self):
        """Test _QUANT_2DP is Decimal('0.01')."""
        assert _QUANT_2DP == Decimal("0.01")

    def test_quant_4dp(self):
        """Test _QUANT_4DP is Decimal('0.0001')."""
        assert _QUANT_4DP == Decimal("0.0001")

    def test_default_query_limit(self):
        """Test default query limit is 1000."""
        assert _DEFAULT_QUERY_LIMIT == 1000

    def test_default_query_offset(self):
        """Test default query offset is 0."""
        assert _DEFAULT_QUERY_OFFSET == 0

    def test_max_query_limit(self):
        """Test max query limit is 10000."""
        assert _MAX_QUERY_LIMIT == 10000

    def test_max_batch_size(self):
        """Test max batch size is 5000."""
        assert _MAX_BATCH_SIZE == 5000


# ==============================================================================
# AUDIT EVENT TYPE ENUM TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestAuditEventTypeEnum:
    """Test AuditEventType enumeration."""

    def test_enum_has_12_members(self):
        """Test AuditEventType has exactly 12 members."""
        assert len(AuditEventType) == 12

    def test_data_ingested(self):
        """Test DATA_INGESTED enum value."""
        assert AuditEventType.DATA_INGESTED.value == "DATA_INGESTED"

    def test_data_validated(self):
        """Test DATA_VALIDATED enum value."""
        assert AuditEventType.DATA_VALIDATED.value == "DATA_VALIDATED"

    def test_data_transformed(self):
        """Test DATA_TRANSFORMED enum value."""
        assert AuditEventType.DATA_TRANSFORMED.value == "DATA_TRANSFORMED"

    def test_emission_factor_resolved(self):
        """Test EMISSION_FACTOR_RESOLVED enum value."""
        assert AuditEventType.EMISSION_FACTOR_RESOLVED.value == "EMISSION_FACTOR_RESOLVED"

    def test_calculation_started(self):
        """Test CALCULATION_STARTED enum value."""
        assert AuditEventType.CALCULATION_STARTED.value == "CALCULATION_STARTED"

    def test_calculation_completed(self):
        """Test CALCULATION_COMPLETED enum value."""
        assert AuditEventType.CALCULATION_COMPLETED.value == "CALCULATION_COMPLETED"

    def test_calculation_failed(self):
        """Test CALCULATION_FAILED enum value."""
        assert AuditEventType.CALCULATION_FAILED.value == "CALCULATION_FAILED"

    def test_compliance_checked(self):
        """Test COMPLIANCE_CHECKED enum value."""
        assert AuditEventType.COMPLIANCE_CHECKED.value == "COMPLIANCE_CHECKED"

    def test_report_generated(self):
        """Test REPORT_GENERATED enum value."""
        assert AuditEventType.REPORT_GENERATED.value == "REPORT_GENERATED"

    def test_provenance_sealed(self):
        """Test PROVENANCE_SEALED enum value."""
        assert AuditEventType.PROVENANCE_SEALED.value == "PROVENANCE_SEALED"

    def test_manual_override(self):
        """Test MANUAL_OVERRIDE enum value."""
        assert AuditEventType.MANUAL_OVERRIDE.value == "MANUAL_OVERRIDE"

    def test_chain_verified(self):
        """Test CHAIN_VERIFIED enum value."""
        assert AuditEventType.CHAIN_VERIFIED.value == "CHAIN_VERIFIED"

    def test_enum_is_str_subclass(self):
        """Test AuditEventType is a string enum."""
        assert issubclass(AuditEventType, str)
        assert issubclass(AuditEventType, Enum)

    def test_enum_string_conversion(self):
        """Test enum members can be used as strings."""
        assert str(AuditEventType.DATA_INGESTED) == "AuditEventType.DATA_INGESTED"
        assert AuditEventType.DATA_INGESTED == "DATA_INGESTED"

    def test_enum_value_equality(self):
        """Test enum value equals its string representation."""
        for member in AuditEventType:
            assert member.value == member.name

    def test_enum_membership_by_value(self):
        """Test membership check by value."""
        assert "DATA_INGESTED" in [e.value for e in AuditEventType]
        assert "INVALID_TYPE" not in [e.value for e in AuditEventType]

    def test_enum_creation_by_value(self):
        """Test creating enum from value string."""
        evt = AuditEventType("CALCULATION_COMPLETED")
        assert evt == AuditEventType.CALCULATION_COMPLETED

    def test_enum_invalid_value_raises(self):
        """Test creating enum with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            AuditEventType("INVALID_VALUE")

    def test_all_values_in_valid_set(self):
        """Test all enum values exist in _VALID_EVENT_TYPES set."""
        for member in AuditEventType:
            assert member.value in _VALID_EVENT_TYPES

    def test_valid_event_types_matches_enum(self):
        """Test _VALID_EVENT_TYPES set has same count as enum."""
        assert len(_VALID_EVENT_TYPES) == len(AuditEventType)


# ==============================================================================
# VALID SCOPES TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestValidScopes:
    """Test _VALID_SCOPES set."""

    def test_valid_scopes_count(self):
        """Test there are exactly 3 valid scopes."""
        assert len(_VALID_SCOPES) == 3

    def test_scope_1_valid(self):
        """Test scope_1 is valid."""
        assert "scope_1" in _VALID_SCOPES

    def test_scope_2_valid(self):
        """Test scope_2 is valid."""
        assert "scope_2" in _VALID_SCOPES

    def test_scope_3_valid(self):
        """Test scope_3 is valid."""
        assert "scope_3" in _VALID_SCOPES

    def test_invalid_scope_not_in_set(self):
        """Test invalid scope values are not in the set."""
        assert "scope_4" not in _VALID_SCOPES
        assert "Scope 1" not in _VALID_SCOPES
        assert "" not in _VALID_SCOPES


# ==============================================================================
# VALID CATEGORIES TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestValidCategories:
    """Test _VALID_CATEGORIES set."""

    def test_valid_categories_count(self):
        """Test there are exactly 15 valid categories."""
        assert len(_VALID_CATEGORIES) == 15

    def test_categories_range_1_to_15(self):
        """Test categories span integers 1 through 15."""
        assert _VALID_CATEGORIES == set(range(1, 16))

    def test_category_1_valid(self):
        """Test category 1 is valid."""
        assert 1 in _VALID_CATEGORIES

    def test_category_15_valid(self):
        """Test category 15 is valid."""
        assert 15 in _VALID_CATEGORIES

    def test_category_0_invalid(self):
        """Test category 0 is invalid."""
        assert 0 not in _VALID_CATEGORIES

    def test_category_16_invalid(self):
        """Test category 16 is invalid."""
        assert 16 not in _VALID_CATEGORIES

    def test_category_negative_invalid(self):
        """Test negative category is invalid."""
        assert -1 not in _VALID_CATEGORIES


# ==============================================================================
# AUDIT EVENT RECORD DATACLASS TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestAuditEventRecord:
    """Test AuditEventRecord frozen dataclass."""

    def _make_record(self, **overrides: Any) -> "AuditEventRecord":
        """Helper to create an AuditEventRecord with defaults."""
        defaults = {
            "event_id": "atl-test001",
            "event_type": "DATA_INGESTED",
            "agent_id": "GL-MRV-S1-001",
            "scope": "scope_1",
            "category": None,
            "organization_id": "org-test-001",
            "reporting_year": 2025,
            "calculation_id": "calc-001",
            "data_quality_score": Decimal("0.85"),
            "payload": {"rows": 500},
            "prev_event_hash": "a" * 64,
            "event_hash": "b" * 64,
            "chain_position": 0,
            "timestamp": "2025-01-15T10:30:00+00:00",
            "metadata": {"source": "test"},
        }
        defaults.update(overrides)
        return AuditEventRecord(**defaults)

    def test_record_creation(self):
        """Test AuditEventRecord can be created with valid data."""
        record = self._make_record()
        assert record.event_id == "atl-test001"
        assert record.event_type == "DATA_INGESTED"

    def test_record_frozen_event_id(self):
        """Test AuditEventRecord event_id cannot be modified."""
        record = self._make_record()
        with pytest.raises(AttributeError):
            record.event_id = "changed"  # type: ignore[misc]

    def test_record_frozen_event_hash(self):
        """Test AuditEventRecord event_hash cannot be modified."""
        record = self._make_record()
        with pytest.raises(AttributeError):
            record.event_hash = "tampered"  # type: ignore[misc]

    def test_record_frozen_payload(self):
        """Test AuditEventRecord payload reference cannot be replaced."""
        record = self._make_record()
        with pytest.raises(AttributeError):
            record.payload = {"hacked": True}  # type: ignore[misc]

    def test_record_frozen_timestamp(self):
        """Test AuditEventRecord timestamp cannot be modified."""
        record = self._make_record()
        with pytest.raises(AttributeError):
            record.timestamp = "2099-01-01T00:00:00Z"  # type: ignore[misc]

    def test_record_frozen_chain_position(self):
        """Test AuditEventRecord chain_position cannot be modified."""
        record = self._make_record()
        with pytest.raises(AttributeError):
            record.chain_position = 999  # type: ignore[misc]

    def test_record_to_dict(self):
        """Test to_dict returns proper dictionary."""
        record = self._make_record()
        d = record.to_dict()
        assert isinstance(d, dict)
        assert d["event_id"] == "atl-test001"
        assert d["event_type"] == "DATA_INGESTED"

    def test_record_to_dict_decimal_serialized(self):
        """Test to_dict serializes Decimal as string."""
        record = self._make_record(data_quality_score=Decimal("0.85"))
        d = record.to_dict()
        assert d["data_quality_score"] == "0.85"
        assert isinstance(d["data_quality_score"], str)

    def test_record_to_dict_all_fields_present(self):
        """Test to_dict includes all 15 fields."""
        record = self._make_record()
        d = record.to_dict()
        expected_keys = {
            "event_id", "event_type", "agent_id", "scope", "category",
            "organization_id", "reporting_year", "calculation_id",
            "data_quality_score", "payload", "prev_event_hash",
            "event_hash", "chain_position", "timestamp", "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_record_scope_none(self):
        """Test AuditEventRecord accepts None scope."""
        record = self._make_record(scope=None)
        assert record.scope is None

    def test_record_category_none(self):
        """Test AuditEventRecord accepts None category."""
        record = self._make_record(category=None)
        assert record.category is None

    def test_record_category_integer(self):
        """Test AuditEventRecord accepts integer category."""
        record = self._make_record(scope="scope_3", category=6)
        assert record.category == 6

    def test_record_calculation_id_none(self):
        """Test AuditEventRecord accepts None calculation_id."""
        record = self._make_record(calculation_id=None)
        assert record.calculation_id is None

    def test_record_empty_payload(self):
        """Test AuditEventRecord accepts empty payload dict."""
        record = self._make_record(payload={})
        assert record.payload == {}

    def test_record_empty_metadata(self):
        """Test AuditEventRecord accepts empty metadata dict."""
        record = self._make_record(metadata={})
        assert record.metadata == {}


# ==============================================================================
# DECIMAL SERIALIZER TESTS
# ==============================================================================


@_SKIP_ENGINE
class TestDecimalSerializer:
    """Test _decimal_serializer JSON hook function."""

    def test_serializes_decimal(self):
        """Test Decimal is serialized to string."""
        result = _decimal_serializer(Decimal("3.14"))
        assert result == "3.14"
        assert isinstance(result, str)

    def test_serializes_decimal_zero(self):
        """Test Decimal zero is serialized correctly."""
        result = _decimal_serializer(Decimal("0"))
        assert result == "0"

    def test_serializes_decimal_negative(self):
        """Test negative Decimal is serialized correctly."""
        result = _decimal_serializer(Decimal("-1.5"))
        assert result == "-1.5"

    def test_serializes_datetime(self):
        """Test datetime is serialized to ISO format."""
        dt = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        result = _decimal_serializer(dt)
        assert "2025-01-15" in result
        assert "10:30:00" in result

    def test_serializes_enum(self):
        """Test Enum is serialized to its value."""
        result = _decimal_serializer(AuditEventType.DATA_INGESTED)
        assert result == "DATA_INGESTED"

    def test_unsupported_type_raises_type_error(self):
        """Test unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _decimal_serializer(set([1, 2, 3]))

    def test_unsupported_type_bytes(self):
        """Test bytes raises TypeError."""
        with pytest.raises(TypeError):
            _decimal_serializer(b"bytes data")

    def test_unsupported_type_object(self):
        """Test arbitrary object raises TypeError."""
        with pytest.raises(TypeError):
            _decimal_serializer(object())

    def test_serializes_high_precision_decimal(self):
        """Test high-precision Decimal is preserved."""
        result = _decimal_serializer(Decimal("1.23456789012345"))
        assert result == "1.23456789012345"
