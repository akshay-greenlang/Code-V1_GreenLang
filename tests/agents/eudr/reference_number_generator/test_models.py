# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- models.py

Tests all 12 enumerations (including 27 MemberStateCode values),
15+ Pydantic v2 models, constants, and model validation. 70+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.models import (
    AGENT_ID,
    AGENT_VERSION,
    EUDR_COMMODITIES,
    VALID_MEMBER_STATES,
    AuditAction,
    BatchGenerationRequest,
    BatchRequest,
    BatchStatus,
    ChecksumAlgorithm,
    CollisionRecord,
    FormatRule,
    FormatTemplate,
    FormatVersion,
    GenerationMode,
    GenerationRequest,
    GenerationResponse,
    HealthStatus,
    MemberStateCode,
    ReferenceNumber,
    ReferenceNumberComponents,
    ReferenceNumberStatus,
    RevocationReason,
    SequenceCounter,
    SequenceOverflowStrategy,
    SequenceStatus,
    TransferReason,
    TransferRecord,
    ValidationLog,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
    ValidatorType,
)


# ====================================================================
# Test: Enum -- ReferenceNumberStatus
# ====================================================================


class TestReferenceNumberStatusEnum:
    """Test ReferenceNumberStatus enum values."""

    def test_member_count(self):
        assert len(ReferenceNumberStatus) == 7

    @pytest.mark.parametrize("member,value", [
        (ReferenceNumberStatus.RESERVED, "reserved"),
        (ReferenceNumberStatus.ACTIVE, "active"),
        (ReferenceNumberStatus.USED, "used"),
        (ReferenceNumberStatus.EXPIRED, "expired"),
        (ReferenceNumberStatus.REVOKED, "revoked"),
        (ReferenceNumberStatus.TRANSFERRED, "transferred"),
        (ReferenceNumberStatus.CANCELLED, "cancelled"),
    ])
    def test_status_values(self, member, value):
        assert member.value == value

    def test_is_str_enum(self):
        assert isinstance(ReferenceNumberStatus.ACTIVE, str)


# ====================================================================
# Test: Enum -- MemberStateCode (27 values)
# ====================================================================


class TestMemberStateCodeEnum:
    """Test MemberStateCode enum has all 27 EU member states."""

    def test_member_count(self):
        assert len(MemberStateCode) == 27

    @pytest.mark.parametrize("code", [
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
        "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
        "PL", "PT", "RO", "SK", "SI", "ES", "SE",
    ])
    def test_member_state_exists(self, code):
        member = MemberStateCode(code)
        assert member.value == code

    def test_is_str_enum(self):
        assert isinstance(MemberStateCode.DE, str)

    def test_all_codes_uppercase_two_letter(self):
        for ms in MemberStateCode:
            assert len(ms.value) == 2
            assert ms.value.isupper()


# ====================================================================
# Test: Enum -- ChecksumAlgorithm
# ====================================================================


class TestChecksumAlgorithmEnum:
    """Test ChecksumAlgorithm enum values."""

    def test_member_count(self):
        assert len(ChecksumAlgorithm) == 4

    @pytest.mark.parametrize("member,value", [
        (ChecksumAlgorithm.LUHN, "luhn"),
        (ChecksumAlgorithm.ISO7064, "iso7064"),
        (ChecksumAlgorithm.CRC16, "crc16"),
        (ChecksumAlgorithm.MODULO97, "modulo97"),
    ])
    def test_algorithm_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- ValidationResult
# ====================================================================


class TestValidationResultEnum:
    """Test ValidationResult enum values."""

    def test_member_count(self):
        assert len(ValidationResult) == 9

    @pytest.mark.parametrize("member,value", [
        (ValidationResult.VALID, "valid"),
        (ValidationResult.INVALID_FORMAT, "invalid_format"),
        (ValidationResult.INVALID_CHECKSUM, "invalid_checksum"),
        (ValidationResult.INVALID_MEMBER_STATE, "invalid_member_state"),
        (ValidationResult.INVALID_SEQUENCE, "invalid_sequence"),
        (ValidationResult.EXPIRED, "expired"),
        (ValidationResult.REVOKED, "revoked"),
        (ValidationResult.NOT_FOUND, "not_found"),
        (ValidationResult.UNKNOWN, "unknown"),
    ])
    def test_result_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- BatchStatus
# ====================================================================


class TestBatchStatusEnum:
    """Test BatchStatus enum values."""

    def test_member_count(self):
        assert len(BatchStatus) == 7

    @pytest.mark.parametrize("member,value", [
        (BatchStatus.PENDING, "pending"),
        (BatchStatus.IN_PROGRESS, "in_progress"),
        (BatchStatus.COMPLETED, "completed"),
        (BatchStatus.PARTIAL, "partial"),
        (BatchStatus.FAILED, "failed"),
        (BatchStatus.CANCELLED, "cancelled"),
        (BatchStatus.TIMEOUT, "timeout"),
    ])
    def test_batch_status_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- TransferReason
# ====================================================================


class TestTransferReasonEnum:
    """Test TransferReason enum values."""

    def test_member_count(self):
        assert len(TransferReason) == 5

    @pytest.mark.parametrize("member,value", [
        (TransferReason.OWNERSHIP_CHANGE, "ownership_change"),
        (TransferReason.MERGER_ACQUISITION, "merger_acquisition"),
        (TransferReason.OPERATIONAL_TRANSFER, "operational_transfer"),
        (TransferReason.REGULATORY_REASSIGNMENT, "regulatory_reassignment"),
        (TransferReason.ERROR_CORRECTION, "error_correction"),
    ])
    def test_transfer_reason_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- RevocationReason
# ====================================================================


class TestRevocationReasonEnum:
    """Test RevocationReason enum values."""

    def test_member_count(self):
        assert len(RevocationReason) == 7

    @pytest.mark.parametrize("member,value", [
        (RevocationReason.FRAUD, "fraud"),
        (RevocationReason.NON_COMPLIANCE, "non_compliance"),
        (RevocationReason.DUPLICATE, "duplicate"),
        (RevocationReason.DATA_ERROR, "data_error"),
        (RevocationReason.OPERATOR_REQUEST, "operator_request"),
        (RevocationReason.REGULATORY_ORDER, "regulatory_order"),
        (RevocationReason.SYSTEM_ERROR, "system_error"),
    ])
    def test_revocation_reason_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- SequenceOverflowStrategy
# ====================================================================


class TestSequenceOverflowStrategyEnum:
    """Test SequenceOverflowStrategy enum values."""

    def test_member_count(self):
        assert len(SequenceOverflowStrategy) == 3

    @pytest.mark.parametrize("member,value", [
        (SequenceOverflowStrategy.EXTEND, "extend"),
        (SequenceOverflowStrategy.REJECT, "reject"),
        (SequenceOverflowStrategy.ROLLOVER, "rollover"),
    ])
    def test_overflow_strategy_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- FormatVersion
# ====================================================================


class TestFormatVersionEnum:
    """Test FormatVersion enum values."""

    def test_member_count(self):
        assert len(FormatVersion) == 3

    @pytest.mark.parametrize("member,value", [
        (FormatVersion.V1_0, "1.0"),
        (FormatVersion.V1_1, "1.1"),
        (FormatVersion.V2_0, "2.0"),
    ])
    def test_format_version_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- ValidatorType
# ====================================================================


class TestValidatorTypeEnum:
    """Test ValidatorType enum values."""

    def test_member_count(self):
        assert len(ValidatorType) == 7

    @pytest.mark.parametrize("member,value", [
        (ValidatorType.FORMAT, "format"),
        (ValidatorType.CHECKSUM, "checksum"),
        (ValidatorType.MEMBER_STATE, "member_state"),
        (ValidatorType.SEQUENCE, "sequence"),
        (ValidatorType.EXPIRATION, "expiration"),
        (ValidatorType.EXISTENCE, "existence"),
        (ValidatorType.LIFECYCLE, "lifecycle"),
    ])
    def test_validator_type_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- AuditAction
# ====================================================================


class TestAuditActionEnum:
    """Test AuditAction enum values."""

    def test_member_count(self):
        assert len(AuditAction) == 10

    @pytest.mark.parametrize("member,value", [
        (AuditAction.GENERATE, "generate"),
        (AuditAction.VALIDATE, "validate"),
        (AuditAction.ACTIVATE, "activate"),
        (AuditAction.USE, "use"),
        (AuditAction.EXPIRE, "expire"),
        (AuditAction.REVOKE, "revoke"),
        (AuditAction.TRANSFER, "transfer"),
        (AuditAction.CANCEL, "cancel"),
        (AuditAction.BATCH_GENERATE, "batch_generate"),
        (AuditAction.VERIFY, "verify"),
    ])
    def test_audit_action_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Enum -- GenerationMode
# ====================================================================


class TestGenerationModeEnum:
    """Test GenerationMode enum values."""

    def test_member_count(self):
        assert len(GenerationMode) == 5

    @pytest.mark.parametrize("member,value", [
        (GenerationMode.SINGLE, "single"),
        (GenerationMode.BATCH, "batch"),
        (GenerationMode.RESERVED, "reserved"),
        (GenerationMode.SEQUENTIAL, "sequential"),
        (GenerationMode.IDEMPOTENT, "idempotent"),
    ])
    def test_generation_mode_values(self, member, value):
        assert member.value == value


# ====================================================================
# Test: Constants
# ====================================================================


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-RNG-038"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_valid_member_states_count(self):
        assert len(VALID_MEMBER_STATES) == 27

    def test_valid_member_states_matches_enum(self):
        enum_values = [ms.value for ms in MemberStateCode]
        assert set(VALID_MEMBER_STATES) == set(enum_values)

    def test_eudr_commodities_count(self):
        assert len(EUDR_COMMODITIES) == 7

    @pytest.mark.parametrize("commodity", [
        "cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood",
    ])
    def test_eudr_commodity_exists(self, commodity):
        assert commodity in EUDR_COMMODITIES


# ====================================================================
# Test: Model -- ReferenceNumberComponents
# ====================================================================


class TestReferenceNumberComponentsModel:
    """Test ReferenceNumberComponents Pydantic model."""

    def test_creation(self, sample_components):
        assert sample_components.prefix == "EUDR"
        assert sample_components.member_state == "DE"
        assert sample_components.year == 2026
        assert sample_components.operator_code == "OP001"
        assert sample_components.sequence == 1
        assert sample_components.checksum == "7"

    def test_member_state_min_length(self):
        with pytest.raises(Exception):
            ReferenceNumberComponents(
                prefix="EUDR", member_state="D", year=2026,
                operator_code="OP001", sequence=1, checksum="7",
            )

    def test_year_below_minimum(self):
        with pytest.raises(Exception):
            ReferenceNumberComponents(
                prefix="EUDR", member_state="DE", year=2023,
                operator_code="OP001", sequence=1, checksum="7",
            )

    def test_year_above_maximum(self):
        with pytest.raises(Exception):
            ReferenceNumberComponents(
                prefix="EUDR", member_state="DE", year=2100,
                operator_code="OP001", sequence=1, checksum="7",
            )

    def test_negative_sequence(self):
        with pytest.raises(Exception):
            ReferenceNumberComponents(
                prefix="EUDR", member_state="DE", year=2026,
                operator_code="OP001", sequence=-1, checksum="7",
            )

    def test_extra_fields_ignored(self):
        comp = ReferenceNumberComponents(
            prefix="EUDR", member_state="DE", year=2026,
            operator_code="OP001", sequence=1, checksum="7",
            extra_field="should_be_ignored",
        )
        assert not hasattr(comp, "extra_field")


# ====================================================================
# Test: Model -- ReferenceNumber
# ====================================================================


class TestReferenceNumberModel:
    """Test ReferenceNumber Pydantic model."""

    def test_creation(self, sample_reference_number):
        assert sample_reference_number.reference_number == "EUDR-DE-2026-OP001-000001-7"
        assert sample_reference_number.operator_id == "OP-001"
        assert sample_reference_number.status == ReferenceNumberStatus.ACTIVE
        assert sample_reference_number.format_version == "1.0"
        assert sample_reference_number.checksum_algorithm == "luhn"

    def test_default_status_is_active(self, sample_components):
        ref = ReferenceNumber(
            reference_id="test-id",
            reference_number="EUDR-DE-2026-OP001-000001-7",
            components=sample_components,
            operator_id="OP-001",
        )
        assert ref.status == ReferenceNumberStatus.ACTIVE

    def test_commodity_optional(self, sample_components):
        ref = ReferenceNumber(
            reference_id="test-id",
            reference_number="EUDR-DE-2026-OP001-000001-7",
            components=sample_components,
            operator_id="OP-001",
            commodity=None,
        )
        assert ref.commodity is None

    def test_provenance_hash_default_empty(self, sample_components):
        ref = ReferenceNumber(
            reference_id="test-id",
            reference_number="EUDR-DE-2026-OP001-000001-7",
            components=sample_components,
            operator_id="OP-001",
        )
        assert ref.provenance_hash == ""


# ====================================================================
# Test: Model -- SequenceCounter
# ====================================================================


class TestSequenceCounterModel:
    """Test SequenceCounter Pydantic model."""

    def test_creation(self, sample_sequence_counter):
        assert sample_sequence_counter.operator_id == "OP-001"
        assert sample_sequence_counter.member_state == "DE"
        assert sample_sequence_counter.year == 2026
        assert sample_sequence_counter.current_value == 100
        assert sample_sequence_counter.max_value == 999999

    def test_default_overflow_strategy(self, sample_sequence_counter):
        assert sample_sequence_counter.overflow_strategy == SequenceOverflowStrategy.EXTEND

    def test_reserved_count_default_zero(self):
        counter = SequenceCounter(
            counter_id="test",
            operator_id="OP-001",
            member_state="DE",
            year=2026,
            current_value=0,
            max_value=999999,
        )
        assert counter.reserved_count == 0


# ====================================================================
# Test: Model -- BatchRequest
# ====================================================================


class TestBatchRequestModel:
    """Test BatchRequest Pydantic model."""

    def test_creation(self, sample_batch_request):
        assert sample_batch_request.operator_id == "OP-001"
        assert sample_batch_request.member_state == "DE"
        assert sample_batch_request.count == 10
        assert sample_batch_request.status == BatchStatus.PENDING

    def test_count_must_be_positive(self):
        with pytest.raises(Exception):
            BatchRequest(
                batch_id="test",
                operator_id="OP-001",
                member_state="DE",
                count=0,
            )

    def test_count_max_10000(self):
        with pytest.raises(Exception):
            BatchRequest(
                batch_id="test",
                operator_id="OP-001",
                member_state="DE",
                count=10001,
            )

    def test_reference_numbers_default_empty(self, sample_batch_request):
        assert sample_batch_request.reference_numbers == []


# ====================================================================
# Test: Model -- ValidationLog
# ====================================================================


class TestValidationLogModel:
    """Test ValidationLog Pydantic model."""

    def test_creation(self, sample_validation_log):
        assert sample_validation_log.is_valid is True
        assert sample_validation_log.result == ValidationResult.VALID
        assert len(sample_validation_log.checks_performed) == 3

    def test_validated_by_default(self, sample_validation_log):
        assert sample_validation_log.validated_by == AGENT_ID


# ====================================================================
# Test: Model -- FormatRule
# ====================================================================


class TestFormatRuleModel:
    """Test FormatRule Pydantic model."""

    def test_creation(self, sample_format_rule):
        assert sample_format_rule.member_state == "DE"
        assert sample_format_rule.country_name == "Germany"
        assert sample_format_rule.prefix == "EUDR"
        assert sample_format_rule.separator == "-"
        assert sample_format_rule.sequence_digits == 6

    def test_sequence_digits_min(self):
        with pytest.raises(Exception):
            FormatRule(
                member_state="DE", country_name="Germany",
                sequence_digits=3,
            )

    def test_sequence_digits_max(self):
        with pytest.raises(Exception):
            FormatRule(
                member_state="DE", country_name="Germany",
                sequence_digits=11,
            )


# ====================================================================
# Test: Model -- CollisionRecord
# ====================================================================


class TestCollisionRecordModel:
    """Test CollisionRecord Pydantic model."""

    def test_creation(self, sample_collision_record):
        assert sample_collision_record.operator_id == "OP-001"
        assert sample_collision_record.attempt_number == 1
        assert sample_collision_record.resolved is True
        assert sample_collision_record.resolution_method == "next_sequence"

    def test_resolved_default_false(self):
        record = CollisionRecord(
            collision_id="test",
            reference_number="EUDR-DE-2026-OP001-000001-7",
            operator_id="OP-001",
            attempt_number=1,
        )
        assert record.resolved is False


# ====================================================================
# Test: Model -- TransferRecord
# ====================================================================


class TestTransferRecordModel:
    """Test TransferRecord Pydantic model."""

    def test_creation(self, sample_transfer_record):
        assert sample_transfer_record.from_operator_id == "OP-001"
        assert sample_transfer_record.to_operator_id == "OP-002"
        assert sample_transfer_record.reason == TransferReason.OWNERSHIP_CHANGE


# ====================================================================
# Test: Model -- GenerationRequest / GenerationResponse
# ====================================================================


class TestGenerationModels:
    """Test GenerationRequest and GenerationResponse models."""

    def test_generation_request(self, sample_generation_request):
        assert sample_generation_request.operator_id == "OP-001"
        assert sample_generation_request.member_state == "DE"
        assert sample_generation_request.commodity == "coffee"
        assert sample_generation_request.idempotency_key == "idem-key-001"

    def test_generation_response(self, sample_generation_response):
        assert sample_generation_response.status == ReferenceNumberStatus.ACTIVE
        assert sample_generation_response.format_version == "1.0"
        assert sample_generation_response.checksum_algorithm == "luhn"

    def test_batch_generation_request(self, sample_batch_generation_request):
        assert sample_batch_generation_request.operator_id == "OP-001"
        assert sample_batch_generation_request.count == 50


# ====================================================================
# Test: Model -- Validation Request / Response
# ====================================================================


class TestValidationModels:
    """Test ValidationRequest and ValidationResponse models."""

    def test_validation_request(self, sample_validation_request):
        assert sample_validation_request.reference_number == "EUDR-DE-2026-OP001-000001-7"
        assert sample_validation_request.check_existence is True
        assert sample_validation_request.check_lifecycle is True

    def test_validation_response_creation(self):
        resp = ValidationResponse(
            reference_number="EUDR-DE-2026-OP001-000001-7",
            is_valid=True,
            result=ValidationResult.VALID,
            validated_at=datetime.now(timezone.utc).isoformat(),
        )
        assert resp.is_valid is True


# ====================================================================
# Test: Model -- SequenceStatus
# ====================================================================


class TestSequenceStatusModel:
    """Test SequenceStatus Pydantic model."""

    def test_creation(self, sample_sequence_status):
        assert sample_sequence_status.operator_id == "OP-001"
        assert sample_sequence_status.current_value == 100
        assert sample_sequence_status.available == 999899
        assert sample_sequence_status.utilization_percent == pytest.approx(0.01)


# ====================================================================
# Test: Model -- FormatTemplate
# ====================================================================


class TestFormatTemplateModel:
    """Test FormatTemplate Pydantic model."""

    def test_creation(self, sample_format_template):
        assert sample_format_template.template_id == "tpl-001"
        assert sample_format_template.version == "1.0"
        assert "{prefix}" in sample_format_template.pattern
        assert len(sample_format_template.components) == 6


# ====================================================================
# Test: Model -- HealthStatus
# ====================================================================


class TestHealthStatusModel:
    """Test HealthStatus Pydantic model."""

    def test_creation(self, sample_health_status):
        assert sample_health_status.agent_id == AGENT_ID
        assert sample_health_status.status == "healthy"
        assert sample_health_status.version == AGENT_VERSION

    def test_defaults(self):
        hs = HealthStatus()
        assert hs.agent_id == AGENT_ID
        assert hs.status == "healthy"
        assert hs.database is False
        assert hs.redis is False
        assert hs.uptime_seconds == 0.0
        assert hs.engines == {}
