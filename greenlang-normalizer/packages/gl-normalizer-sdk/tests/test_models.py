"""
Unit tests for SDK data models.
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError as PydanticValidationError

from gl_normalizer.models import (
    PolicyMode,
    BatchMode,
    MatchMethod,
    EntityType,
    JobStatus,
    ClientConfig,
    ReferenceConditions,
    NormalizeMetadata,
    NormalizeRequest,
    EntityHints,
    EntityRequest,
    ConversionStep,
    ConversionTrace,
    Warning,
    NormalizeResult,
    EntityResult,
    AuditInfo,
    BatchSummary,
    BatchItemResult,
    BatchResult,
    Job,
    Vocabulary,
    VocabularyEntry,
)


class TestEnums:
    """Test enum definitions."""

    def test_policy_mode_values(self) -> None:
        """Test PolicyMode enum values."""
        assert PolicyMode.STRICT.value == "STRICT"
        assert PolicyMode.LENIENT.value == "LENIENT"

    def test_batch_mode_values(self) -> None:
        """Test BatchMode enum values."""
        assert BatchMode.PARTIAL.value == "PARTIAL"
        assert BatchMode.ALL_OR_NOTHING.value == "ALL_OR_NOTHING"

    def test_match_method_values(self) -> None:
        """Test MatchMethod enum values."""
        assert MatchMethod.EXACT_ID.value == "exact_id"
        assert MatchMethod.ALIAS.value == "alias"
        assert MatchMethod.FUZZY.value == "fuzzy"
        assert MatchMethod.LLM_CANDIDATE.value == "llm_candidate"

    def test_entity_type_values(self) -> None:
        """Test EntityType enum values."""
        assert EntityType.FUEL.value == "fuel"
        assert EntityType.MATERIAL.value == "material"
        assert EntityType.PROCESS.value == "process"

    def test_job_status_values(self) -> None:
        """Test JobStatus enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestClientConfig:
    """Test ClientConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = ClientConfig()
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.retry_max_delay == 30.0
        assert config.enable_cache is True
        assert config.cache_ttl == 300
        assert config.pool_connections == 10
        assert config.pool_maxsize == 100

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = ClientConfig(
            timeout=60.0,
            max_retries=5,
            enable_cache=False,
        )
        assert config.timeout == 60.0
        assert config.max_retries == 5
        assert config.enable_cache is False

    def test_validation_timeout_positive(self) -> None:
        """Test timeout must be positive."""
        with pytest.raises(PydanticValidationError):
            ClientConfig(timeout=0)

    def test_validation_max_retries_range(self) -> None:
        """Test max_retries must be in range."""
        with pytest.raises(PydanticValidationError):
            ClientConfig(max_retries=11)

    def test_immutable(self) -> None:
        """Test config is immutable."""
        config = ClientConfig()
        with pytest.raises(PydanticValidationError):
            config.timeout = 60.0  # type: ignore


class TestReferenceConditions:
    """Test ReferenceConditions model."""

    def test_default_values(self) -> None:
        """Test default reference conditions."""
        conditions = ReferenceConditions()
        assert conditions.temperature_c == 0.0
        assert conditions.pressure_kpa == 101.325

    def test_custom_values(self) -> None:
        """Test custom reference conditions."""
        conditions = ReferenceConditions(temperature_c=15.0, pressure_kpa=100.0)
        assert conditions.temperature_c == 15.0
        assert conditions.pressure_kpa == 100.0

    def test_alias_support(self) -> None:
        """Test alias field names."""
        conditions = ReferenceConditions(temperature_C=25.0, pressure_kPa=110.0)
        assert conditions.temperature_c == 25.0
        assert conditions.pressure_kpa == 110.0


class TestNormalizeMetadata:
    """Test NormalizeMetadata model."""

    def test_empty_metadata(self) -> None:
        """Test empty metadata."""
        metadata = NormalizeMetadata()
        assert metadata.locale is None
        assert metadata.reference_conditions is None
        assert metadata.gwp_version is None

    def test_full_metadata(self) -> None:
        """Test full metadata."""
        metadata = NormalizeMetadata(
            locale="en-US",
            reference_conditions=ReferenceConditions(),
            gwp_version="AR6",
            notes="Test note",
        )
        assert metadata.locale == "en-US"
        assert metadata.reference_conditions is not None
        assert metadata.gwp_version == "AR6"

    def test_gwp_version_validation(self) -> None:
        """Test GWP version must be AR5 or AR6."""
        with pytest.raises(PydanticValidationError):
            NormalizeMetadata(gwp_version="AR4")


class TestNormalizeRequest:
    """Test NormalizeRequest model."""

    def test_minimal_request(self) -> None:
        """Test minimal request with required fields only."""
        request = NormalizeRequest(value=100.0, unit="kWh")
        assert request.value == 100.0
        assert request.unit == "kWh"
        assert request.target_unit is None
        assert request.expected_dimension is None

    def test_full_request(self) -> None:
        """Test full request with all fields."""
        request = NormalizeRequest(
            value=100.0,
            unit="kWh",
            target_unit="MJ",
            expected_dimension="energy",
            field="energy_consumption",
            metadata=NormalizeMetadata(gwp_version="AR6"),
        )
        assert request.target_unit == "MJ"
        assert request.expected_dimension == "energy"
        assert request.field == "energy_consumption"
        assert request.metadata is not None

    def test_unit_whitespace_stripped(self) -> None:
        """Test that whitespace is stripped from unit strings."""
        request = NormalizeRequest(value=100.0, unit="  kWh  ")
        assert request.unit == "kWh"

    def test_unit_validation_min_length(self) -> None:
        """Test unit must have minimum length."""
        with pytest.raises(PydanticValidationError):
            NormalizeRequest(value=100.0, unit="")

    def test_unit_validation_max_length(self) -> None:
        """Test unit must not exceed maximum length."""
        with pytest.raises(PydanticValidationError):
            NormalizeRequest(value=100.0, unit="x" * 501)


class TestEntityRequest:
    """Test EntityRequest model."""

    def test_minimal_request(self) -> None:
        """Test minimal entity request."""
        request = EntityRequest(entity_type=EntityType.FUEL, raw_name="Natural gas")
        assert request.entity_type == EntityType.FUEL
        assert request.raw_name == "Natural gas"

    def test_full_request(self) -> None:
        """Test full entity request."""
        request = EntityRequest(
            entity_type=EntityType.FUEL,
            raw_name="Natural gas",
            raw_code="NG-001",
            field="fuel_type",
            hints=EntityHints(region="EU", sector="energy"),
        )
        assert request.raw_code == "NG-001"
        assert request.hints is not None
        assert request.hints.region == "EU"


class TestConversionStep:
    """Test ConversionStep model."""

    def test_basic_step(self) -> None:
        """Test basic conversion step."""
        step = ConversionStep(
            from_unit="kWh",
            to_unit="MJ",
            factor=3.6,
            method="multiply",
        )
        assert step.from_unit == "kWh"
        assert step.to_unit == "MJ"
        assert step.factor == 3.6
        assert step.method == "multiply"

    def test_step_with_reference_conditions(self) -> None:
        """Test step with reference conditions."""
        step = ConversionStep(
            from_unit="Nm3",
            to_unit="m3",
            factor=1.0,
            method="basis_conversion",
            reference_conditions={"T": 273.15, "P": 101.325},
        )
        assert step.reference_conditions is not None
        assert step.reference_conditions["T"] == 273.15


class TestNormalizeResult:
    """Test NormalizeResult model."""

    def test_basic_result(self) -> None:
        """Test basic normalize result."""
        result = NormalizeResult(
            dimension="energy",
            canonical_value=360.0,
            canonical_unit="MJ",
            raw_value=100.0,
            raw_unit="kWh",
        )
        assert result.canonical_value == 360.0
        assert result.canonical_unit == "MJ"
        assert result.dimension == "energy"

    def test_result_with_trace(self) -> None:
        """Test result with conversion trace."""
        result = NormalizeResult(
            dimension="energy",
            canonical_value=360.0,
            canonical_unit="MJ",
            raw_value=100.0,
            raw_unit="kWh",
            conversion_trace=ConversionTrace(
                steps=[
                    ConversionStep(
                        from_unit="kWh",
                        to_unit="MJ",
                        factor=3.6,
                        method="multiply",
                    )
                ],
                factor_version="2026.01.0",
            ),
        )
        assert result.conversion_trace is not None
        assert len(result.conversion_trace.steps) == 1


class TestEntityResult:
    """Test EntityResult model."""

    def test_basic_result(self) -> None:
        """Test basic entity result."""
        result = EntityResult(
            entity_type=EntityType.FUEL,
            raw_name="Nat Gas",
            reference_id="GL-FUEL-NATGAS",
            canonical_name="Natural gas",
            vocabulary_version="2026.01.0",
            match_method=MatchMethod.ALIAS,
            confidence=1.0,
        )
        assert result.reference_id == "GL-FUEL-NATGAS"
        assert result.canonical_name == "Natural gas"
        assert result.confidence == 1.0
        assert result.needs_review is False

    def test_result_needs_review(self) -> None:
        """Test result that needs review."""
        result = EntityResult(
            entity_type=EntityType.FUEL,
            raw_name="Unknown Fuel",
            reference_id="GL-FUEL-UNKNOWN",
            canonical_name="Unknown",
            vocabulary_version="2026.01.0",
            match_method=MatchMethod.FUZZY,
            confidence=0.7,
            needs_review=True,
        )
        assert result.needs_review is True
        assert result.match_method == MatchMethod.FUZZY

    def test_confidence_validation(self) -> None:
        """Test confidence must be between 0 and 1."""
        with pytest.raises(PydanticValidationError):
            EntityResult(
                entity_type=EntityType.FUEL,
                raw_name="Test",
                reference_id="GL-FUEL-TEST",
                canonical_name="Test",
                vocabulary_version="2026.01.0",
                match_method=MatchMethod.EXACT_ID,
                confidence=1.5,  # Invalid
            )


class TestBatchResult:
    """Test BatchResult model."""

    def test_batch_result(self) -> None:
        """Test batch result."""
        result = BatchResult(
            summary=BatchSummary(total=2, success=2, failed=0),
            results=[
                BatchItemResult(
                    source_record_id="batch-0",
                    status="success",
                    canonical_measurements=[],
                    normalized_entities=[],
                ),
                BatchItemResult(
                    source_record_id="batch-1",
                    status="success",
                    canonical_measurements=[],
                    normalized_entities=[],
                ),
            ],
        )
        assert result.summary.total == 2
        assert result.summary.success == 2
        assert len(result.results) == 2


class TestJob:
    """Test Job model."""

    def test_pending_job(self) -> None:
        """Test pending job."""
        job = Job(
            job_id="job-abc123",
            status=JobStatus.PENDING,
            total_items=1000,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.results_url is None

    def test_completed_job(self) -> None:
        """Test completed job."""
        now = datetime.now(timezone.utc)
        job = Job(
            job_id="job-abc123",
            status=JobStatus.COMPLETED,
            progress=100.0,
            total_items=1000,
            processed_items=1000,
            created_at=now,
            updated_at=now,
            completed_at=now,
            results_url="https://api.greenlang.io/v1/jobs/job-abc123/results",
        )
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.results_url is not None


class TestVocabulary:
    """Test Vocabulary model."""

    def test_vocabulary(self) -> None:
        """Test vocabulary model."""
        vocab = Vocabulary(
            vocabulary_id="fuels",
            name="Fuel Types",
            version="2026.01.0",
            entity_type=EntityType.FUEL,
            entity_count=150,
            created_at=datetime.now(timezone.utc),
            description="Standard fuel type vocabulary",
        )
        assert vocab.vocabulary_id == "fuels"
        assert vocab.entity_count == 150


class TestVocabularyEntry:
    """Test VocabularyEntry model."""

    def test_active_entry(self) -> None:
        """Test active vocabulary entry."""
        entry = VocabularyEntry(
            reference_id="GL-FUEL-NATGAS",
            canonical_name="Natural gas",
            entity_type=EntityType.FUEL,
            aliases=["Nat Gas", "Natural-gas", "NG"],
            attributes={"composition": "CH4"},
        )
        assert entry.status == "active"
        assert entry.replaced_by is None
        assert len(entry.aliases) == 3

    def test_deprecated_entry(self) -> None:
        """Test deprecated vocabulary entry."""
        entry = VocabularyEntry(
            reference_id="GL-FUEL-OLD",
            canonical_name="Old Fuel",
            entity_type=EntityType.FUEL,
            status="deprecated",
            replaced_by="GL-FUEL-NEW",
        )
        assert entry.status == "deprecated"
        assert entry.replaced_by == "GL-FUEL-NEW"
