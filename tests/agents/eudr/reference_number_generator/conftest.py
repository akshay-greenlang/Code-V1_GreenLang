# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-038 Reference Number Generator tests.

Provides reusable test fixtures for config, models, provenance, reference
numbers, sequence counters, batch requests, format rules for all 27 EU
member states, validation logs, engine fixtures, and mock Redis client
across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    EU_MEMBER_STATES,
    ReferenceNumberGeneratorConfig,
    reset_config,
)
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
from greenlang.agents.eudr.reference_number_generator.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> ReferenceNumberGeneratorConfig:
    """Create a default ReferenceNumberGeneratorConfig instance."""
    return ReferenceNumberGeneratorConfig()


@pytest.fixture
def custom_config() -> ReferenceNumberGeneratorConfig:
    """Create a config with non-default values for testing overrides."""
    return ReferenceNumberGeneratorConfig(
        db_host="db.test.local",
        db_port=5433,
        db_name="greenlang_test",
        db_pool_min=1,
        db_pool_max=5,
        redis_host="redis.test.local",
        redis_port=6380,
        reference_prefix="TEST",
        default_member_state="FR",
        separator="_",
        sequence_digits=8,
        sequence_start=100,
        sequence_end=99999999,
        checksum_algorithm="iso7064",
        checksum_length=2,
        max_batch_size=500,
        collision_max_retries=3,
        default_expiration_months=6,
        retention_years=7,
    )


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Reference Number Component fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_components() -> ReferenceNumberComponents:
    """Create sample reference number components."""
    return ReferenceNumberComponents(
        prefix="EUDR",
        member_state="DE",
        year=2026,
        operator_code="OP001",
        sequence=1,
        checksum="7",
    )


@pytest.fixture
def sample_components_fr() -> ReferenceNumberComponents:
    """Create sample French reference number components."""
    return ReferenceNumberComponents(
        prefix="EUDR",
        member_state="FR",
        year=2026,
        operator_code="FROPS01",
        sequence=42,
        checksum="3",
    )


# ---------------------------------------------------------------------------
# Reference Number fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_reference_number(sample_components: ReferenceNumberComponents) -> ReferenceNumber:
    """Create a sample ReferenceNumber model."""
    return ReferenceNumber(
        reference_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        components=sample_components,
        operator_id="OP-001",
        commodity="coffee",
        status=ReferenceNumberStatus.ACTIVE,
        format_version="1.0",
        checksum_algorithm="luhn",
        generated_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(days=365),
        provenance_hash="a" * 64,
    )


@pytest.fixture
def sample_reference_numbers() -> List[str]:
    """Provide a list of sample reference number strings."""
    return [
        "EUDR-DE-2026-OP001-000001-7",
        "EUDR-FR-2026-FROPS-000042-3",
        "EUDR-IT-2026-ITOP1-000100-5",
        "EUDR-ES-2026-ESOP2-000200-1",
        "EUDR-NL-2026-NLOP3-000300-9",
    ]


# ---------------------------------------------------------------------------
# Sequence Counter fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sequence_counter() -> SequenceCounter:
    """Create a sample SequenceCounter model."""
    return SequenceCounter(
        counter_id=str(uuid.uuid4()),
        operator_id="OP-001",
        member_state="DE",
        year=2026,
        current_value=100,
        max_value=999999,
        reserved_count=0,
        overflow_strategy=SequenceOverflowStrategy.EXTEND,
    )


@pytest.fixture
def sample_sequence_status() -> SequenceStatus:
    """Create a sample SequenceStatus model."""
    return SequenceStatus(
        operator_id="OP-001",
        member_state="DE",
        year=2026,
        current_value=100,
        max_value=999999,
        available=999899,
        utilization_percent=0.01,
        overflow_strategy="extend",
    )


# ---------------------------------------------------------------------------
# Batch fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_batch_request() -> BatchRequest:
    """Create a sample BatchRequest model."""
    return BatchRequest(
        batch_id=str(uuid.uuid4()),
        operator_id="OP-001",
        member_state="DE",
        commodity="coffee",
        count=10,
        status=BatchStatus.PENDING,
    )


@pytest.fixture
def sample_batch_generation_request() -> BatchGenerationRequest:
    """Create a sample BatchGenerationRequest model."""
    return BatchGenerationRequest(
        operator_id="OP-001",
        member_state="DE",
        count=50,
        commodity="cocoa",
    )


# ---------------------------------------------------------------------------
# Validation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_validation_log() -> ValidationLog:
    """Create a sample ValidationLog model."""
    return ValidationLog(
        validation_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        result=ValidationResult.VALID,
        checks_performed=[
            {"check": "format", "passed": True, "message": "Valid format"},
            {"check": "checksum", "passed": True, "message": "Checksum valid"},
            {"check": "member_state", "passed": True, "message": "Valid member state: DE"},
        ],
        is_valid=True,
    )


@pytest.fixture
def sample_validation_request() -> ValidationRequest:
    """Create a sample ValidationRequest model."""
    return ValidationRequest(
        reference_number="EUDR-DE-2026-OP001-000001-7",
        check_existence=True,
        check_lifecycle=True,
    )


# ---------------------------------------------------------------------------
# Format Rule fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_format_rule() -> FormatRule:
    """Create a sample FormatRule model."""
    return FormatRule(
        member_state="DE",
        country_name="Germany",
        prefix="EUDR",
        separator="-",
        sequence_digits=6,
        checksum_algorithm="luhn",
        format_version="1.0",
        example="EUDR-DE-2026-OP001-000001-7",
    )


@pytest.fixture
def all_member_state_format_rules() -> List[FormatRule]:
    """Create format rules for all 27 EU member states."""
    rules = []
    for code, name in EU_MEMBER_STATES.items():
        rules.append(
            FormatRule(
                member_state=code,
                country_name=name,
                prefix="EUDR",
                separator="-",
                sequence_digits=6,
                checksum_algorithm="luhn",
                format_version="1.0",
                example=f"EUDR-{code}-2026-OP001-000001-0",
            )
        )
    return rules


# ---------------------------------------------------------------------------
# Collision fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_collision_record() -> CollisionRecord:
    """Create a sample CollisionRecord model."""
    return CollisionRecord(
        collision_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        operator_id="OP-001",
        attempt_number=1,
        resolved=True,
        resolution_method="next_sequence",
    )


# ---------------------------------------------------------------------------
# Transfer fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_transfer_record() -> TransferRecord:
    """Create a sample TransferRecord model."""
    return TransferRecord(
        transfer_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        from_operator_id="OP-001",
        to_operator_id="OP-002",
        reason=TransferReason.OWNERSHIP_CHANGE,
        authorized_by="admin@greenlang.io",
        provenance_hash="b" * 64,
    )


# ---------------------------------------------------------------------------
# Generation Request/Response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_generation_request() -> GenerationRequest:
    """Create a sample GenerationRequest model."""
    return GenerationRequest(
        operator_id="OP-001",
        member_state="DE",
        commodity="coffee",
        idempotency_key="idem-key-001",
    )


@pytest.fixture
def sample_generation_response() -> GenerationResponse:
    """Create a sample GenerationResponse model."""
    return GenerationResponse(
        reference_id=str(uuid.uuid4()),
        reference_number="EUDR-DE-2026-OP001-000001-7",
        operator_id="OP-001",
        member_state="DE",
        status=ReferenceNumberStatus.ACTIVE,
        format_version="1.0",
        checksum_algorithm="luhn",
        generated_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=365)).isoformat(),
        provenance_hash="c" * 64,
    )


# ---------------------------------------------------------------------------
# Format Template fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_format_template() -> FormatTemplate:
    """Create a sample FormatTemplate model."""
    return FormatTemplate(
        template_id="tpl-001",
        version="1.0",
        pattern="{prefix}{sep}{ms}{sep}{year}{sep}{operator}{sep}{sequence}{sep}{checksum}",
        components=["prefix", "member_state", "year", "operator_code", "sequence", "checksum"],
        regex_pattern=r"^EUDR-[A-Z]{2}-\d{4}-[A-Z0-9]{1,10}-\d{6}-\d{1}$",
        example="EUDR-DE-2026-OP001-000001-7",
    )


# ---------------------------------------------------------------------------
# Health Status fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_health_status() -> HealthStatus:
    """Create a sample HealthStatus model."""
    return HealthStatus(
        agent_id=AGENT_ID,
        status="healthy",
        version=AGENT_VERSION,
        engines={
            "number_generator": "available",
            "format_validator": "available",
            "sequence_manager": "available",
        },
        database=True,
        redis=True,
        uptime_seconds=3600.0,
        active_references=1000,
        total_generated=5000,
    )


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def number_generator_engine(sample_config):
    """Create a NumberGenerator engine instance."""
    from greenlang.agents.eudr.reference_number_generator.number_generator import (
        NumberGenerator,
    )
    return NumberGenerator(config=sample_config)


@pytest.fixture
def format_validator_engine(sample_config):
    """Create a FormatValidator engine instance."""
    from greenlang.agents.eudr.reference_number_generator.format_validator import (
        FormatValidator,
    )
    return FormatValidator(config=sample_config)


@pytest.fixture
def sequence_manager_engine(sample_config):
    """Create a SequenceManager engine instance."""
    from greenlang.agents.eudr.reference_number_generator.sequence_manager import (
        SequenceManager,
    )
    return SequenceManager(config=sample_config)


# ---------------------------------------------------------------------------
# Mock Redis client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_redis_client() -> MagicMock:
    """Create a mock Redis client for distributed lock testing."""
    client = MagicMock()
    client.incr = MagicMock(return_value=1)
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    client.delete = MagicMock(return_value=1)
    client.expire = MagicMock(return_value=True)
    client.setnx = MagicMock(return_value=True)
    client.pipeline = MagicMock()
    client.exists = MagicMock(return_value=0)
    return client


# ---------------------------------------------------------------------------
# EU Member States convenience fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def eu_member_state_codes() -> List[str]:
    """Provide all 27 EU member state codes."""
    return list(EU_MEMBER_STATES.keys())


@pytest.fixture
def eu_member_state_map() -> Dict[str, str]:
    """Provide full EU member state code to country name mapping."""
    return dict(EU_MEMBER_STATES)


# ---------------------------------------------------------------------------
# Commodity list fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def eudr_commodities() -> List[str]:
    """Provide all EUDR commodities."""
    return list(EUDR_COMMODITIES)
