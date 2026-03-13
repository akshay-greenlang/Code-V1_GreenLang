# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- api.py

Tests API-level behavior for reference number generation, validation,
batch processing, retrieval, lifecycle management, health endpoints,
and concurrent request handling. 60+ tests.

These tests exercise API-level behavior using the engines directly.
Once api.py (FastAPI router) is implemented, tests can use
httpx.AsyncClient / TestClient to test HTTP endpoints.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    EU_MEMBER_STATES,
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.format_validator import (
    FormatValidator,
)
from greenlang.agents.eudr.reference_number_generator.models import (
    AGENT_ID,
    AGENT_VERSION,
    BatchGenerationRequest,
    BatchRequest,
    BatchStatus,
    GenerationRequest,
    GenerationResponse,
    HealthStatus,
    ReferenceNumberStatus,
    ValidationRequest,
    ValidationResponse,
    ValidationResult,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)
from greenlang.agents.eudr.reference_number_generator.sequence_manager import (
    SequenceManager,
)


# ====================================================================
# Test: Single Reference Generation API
# ====================================================================


class TestGenerateAPI:
    """Test single reference generation API-level behavior."""

    @pytest.mark.asyncio
    async def test_generate_single_reference(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE")
        assert result["reference_number"] is not None
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_generate_returns_all_fields(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE", commodity="coffee")
        assert "reference_id" in result
        assert "reference_number" in result
        assert "components" in result
        assert "operator_id" in result
        assert "commodity" in result
        assert "status" in result
        assert "format_version" in result
        assert "checksum_algorithm" in result
        assert "generated_at" in result
        assert "expires_at" in result
        assert "provenance_hash" in result

    @pytest.mark.asyncio
    async def test_generate_with_idempotency_key(self):
        engine = NumberGenerator()
        r1 = await engine.generate("OP-001", "DE", idempotency_key="key-001")
        r2 = await engine.generate("OP-001", "DE", idempotency_key="key-001")
        assert r1["reference_number"] == r2["reference_number"]

    @pytest.mark.asyncio
    async def test_generate_invalid_operator_returns_error(self):
        engine = NumberGenerator()
        with pytest.raises(ValueError, match="operator_id"):
            await engine.generate("", "DE")

    @pytest.mark.asyncio
    async def test_generate_invalid_member_state_returns_error(self):
        engine = NumberGenerator()
        with pytest.raises(ValueError, match="member_state"):
            await engine.generate("OP-001", "XX")

    @pytest.mark.asyncio
    async def test_generate_request_model(self):
        req = GenerationRequest(
            operator_id="OP-001",
            member_state="DE",
            commodity="coffee",
            idempotency_key="key-001",
        )
        engine = NumberGenerator()
        result = await engine.generate(
            req.operator_id,
            req.member_state,
            commodity=req.commodity,
            idempotency_key=req.idempotency_key,
        )
        assert result["operator_id"] == "OP-001"

    @pytest.mark.asyncio
    async def test_generate_response_model_compatible(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE")
        # Verify result can construct GenerationResponse
        resp = GenerationResponse(
            reference_id=result["reference_id"],
            reference_number=result["reference_number"],
            operator_id=result["operator_id"],
            member_state=result["components"]["member_state"],
            generated_at=result["generated_at"],
            expires_at=result["expires_at"],
            provenance_hash=result["provenance_hash"],
        )
        assert resp.reference_number == result["reference_number"]


# ====================================================================
# Test: Validation API
# ====================================================================


class TestValidateAPI:
    """Test reference number validation API-level behavior."""

    @pytest.mark.asyncio
    async def test_validate_valid_reference(self):
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_validate_returns_checks(self):
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert "checks" in vresult
        assert isinstance(vresult["checks"], list)
        assert len(vresult["checks"]) >= 4

    @pytest.mark.asyncio
    async def test_validate_invalid_format(self):
        val = FormatValidator()
        vresult = await val.validate("INVALID-REF-NUMBER")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_response_model_compatible(self):
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        resp = ValidationResponse(
            reference_number=vresult["reference_number"],
            is_valid=vresult["is_valid"],
            result=ValidationResult(vresult["result"]),
            checks=vresult["checks"],
            validated_at=vresult["validated_at"],
        )
        assert resp.is_valid is True

    @pytest.mark.asyncio
    async def test_validate_request_model(self):
        req = ValidationRequest(
            reference_number="EUDR-DE-2026-OP001-000001-7",
            check_existence=True,
            check_lifecycle=True,
        )
        assert req.reference_number == "EUDR-DE-2026-OP001-000001-7"


# ====================================================================
# Test: Batch Generation API
# ====================================================================


class TestBatchAPI:
    """Test batch generation API-level behavior."""

    @pytest.mark.asyncio
    async def test_batch_generate_10(self):
        engine = NumberGenerator()
        results = []
        for _ in range(10):
            result = await engine.generate("OP-001", "DE")
            results.append(result["reference_number"])
        assert len(results) == 10
        assert len(set(results)) == 10

    @pytest.mark.asyncio
    async def test_batch_generate_50_with_commodity(self):
        engine = NumberGenerator()
        results = []
        for _ in range(50):
            result = await engine.generate("OP-001", "DE", commodity="coffee")
            results.append(result)
        assert all(r["commodity"] == "coffee" for r in results)

    @pytest.mark.asyncio
    async def test_batch_request_model(self):
        req = BatchGenerationRequest(
            operator_id="OP-001",
            member_state="DE",
            count=100,
            commodity="cocoa",
        )
        assert req.count == 100
        assert req.commodity == "cocoa"

    @pytest.mark.asyncio
    async def test_batch_status_model(self):
        batch = BatchRequest(
            batch_id=str(uuid.uuid4()),
            operator_id="OP-001",
            member_state="DE",
            count=10,
            status=BatchStatus.COMPLETED,
            generated_count=10,
            reference_numbers=["ref-1", "ref-2"],
            completed_at=datetime.now(timezone.utc),
        )
        assert batch.status == BatchStatus.COMPLETED
        assert batch.generated_count == 10


# ====================================================================
# Test: Retrieval API
# ====================================================================


class TestRetrievalAPI:
    """Test reference number retrieval API-level behavior."""

    @pytest.mark.asyncio
    async def test_get_reference_by_number(self):
        engine = NumberGenerator()
        result = await engine.generate("OP-001", "DE")
        found = await engine.get_reference(result["reference_number"])
        assert found is not None
        assert found["operator_id"] == "OP-001"

    @pytest.mark.asyncio
    async def test_get_reference_not_found(self):
        engine = NumberGenerator()
        found = await engine.get_reference("NONEXISTENT")
        assert found is None

    @pytest.mark.asyncio
    async def test_list_all_references(self):
        engine = NumberGenerator()
        for _ in range(5):
            await engine.generate("OP-001", "DE")
        refs = await engine.list_references()
        assert len(refs) == 5

    @pytest.mark.asyncio
    async def test_list_filter_by_operator(self):
        engine = NumberGenerator()
        await engine.generate("OP-001", "DE")
        await engine.generate("OP-002", "FR")
        refs = await engine.list_references(operator_id="OP-001")
        assert len(refs) == 1

    @pytest.mark.asyncio
    async def test_list_filter_by_status(self):
        engine = NumberGenerator()
        await engine.generate("OP-001", "DE")
        refs = await engine.list_references(status="active")
        assert len(refs) == 1

    @pytest.mark.asyncio
    async def test_list_filter_by_member_state(self):
        engine = NumberGenerator()
        await engine.generate("OP-001", "DE")
        await engine.generate("OP-001", "FR")
        refs = await engine.list_references(member_state="FR")
        assert len(refs) == 1


# ====================================================================
# Test: Health Check API
# ====================================================================


class TestHealthAPI:
    """Test health check API-level behavior."""

    @pytest.mark.asyncio
    async def test_number_generator_health(self):
        engine = NumberGenerator()
        health = await engine.health_check()
        assert health["status"] == "available"

    @pytest.mark.asyncio
    async def test_format_validator_health(self):
        engine = FormatValidator()
        health = await engine.health_check()
        assert health["status"] == "available"
        assert health["member_state_rules"] == "27"

    @pytest.mark.asyncio
    async def test_sequence_manager_health(self):
        engine = SequenceManager()
        health = await engine.health_check()
        assert health["status"] == "available"

    def test_health_status_model(self):
        hs = HealthStatus(
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
            active_references=100,
            total_generated=500,
        )
        assert hs.status == "healthy"
        assert len(hs.engines) == 3


# ====================================================================
# Test: Concurrent Generation API
# ====================================================================


class TestConcurrentGenerationAPI:
    """Test concurrent reference generation at API level."""

    @pytest.mark.asyncio
    async def test_100_concurrent_requests(self):
        engine = NumberGenerator()
        tasks = [
            engine.generate(f"OP-{i:04d}", "DE")
            for i in range(100)
        ]
        results = await asyncio.gather(*tasks)
        refs = {r["reference_number"] for r in results}
        assert len(refs) == 100

    @pytest.mark.asyncio
    async def test_concurrent_multi_state(self):
        engine = NumberGenerator()
        states = list(EU_MEMBER_STATES.keys())[:10]
        tasks = [
            engine.generate("OP-001", ms)
            for ms in states
        ]
        results = await asyncio.gather(*tasks)
        refs = {r["reference_number"] for r in results}
        assert len(refs) == 10

    @pytest.mark.asyncio
    async def test_concurrent_generate_and_validate(self):
        gen = NumberGenerator()
        val = FormatValidator()

        # Generate concurrently
        gen_tasks = [gen.generate(f"OP-{i:03d}", "DE") for i in range(20)]
        gen_results = await asyncio.gather(*gen_tasks)

        # Validate concurrently
        val_tasks = [
            val.validate(r["reference_number"]) for r in gen_results
        ]
        val_results = await asyncio.gather(*val_tasks)

        assert all(v["is_valid"] for v in val_results)


# ====================================================================
# Test: Sequence Management API
# ====================================================================


class TestSequenceAPI:
    """Test sequence management API-level behavior."""

    @pytest.mark.asyncio
    async def test_sequence_status(self):
        engine = SequenceManager()
        await engine.next_sequence("OP-001", "DE", 2026)
        status = await engine.get_sequence_status("OP-001", "DE", 2026)
        assert status["current_value"] == 1

    @pytest.mark.asyncio
    async def test_sequence_reserve(self):
        engine = SequenceManager()
        reserved = await engine.reserve_sequences("OP-001", "DE", 2026, 10)
        assert len(reserved) == 10

    @pytest.mark.asyncio
    async def test_sequence_reset(self):
        engine = SequenceManager()
        await engine.next_sequence("OP-001", "DE", 2026)
        result = await engine.reset_sequence("OP-001", "DE", 2026)
        assert result is True

    @pytest.mark.asyncio
    async def test_sequence_list_counters(self):
        engine = SequenceManager()
        await engine.next_sequence("OP-001", "DE", 2026)
        await engine.next_sequence("OP-002", "FR", 2026)
        counters = await engine.list_counters()
        assert len(counters) == 2


# ====================================================================
# Test: Error Handling API
# ====================================================================


class TestErrorHandlingAPI:
    """Test error handling at API level."""

    @pytest.mark.asyncio
    async def test_empty_operator_id_error(self):
        engine = NumberGenerator()
        with pytest.raises(ValueError):
            await engine.generate("", "DE")

    @pytest.mark.asyncio
    async def test_invalid_member_state_error(self):
        engine = NumberGenerator()
        with pytest.raises(ValueError):
            await engine.generate("OP-001", "XX")

    @pytest.mark.asyncio
    async def test_sequence_exhaustion_error(self):
        config = ReferenceNumberGeneratorConfig(
            sequence_start=1,
            sequence_end=2,
            sequence_overflow_strategy="reject",
        )
        engine = NumberGenerator(config=config)
        await engine.generate("OP-001", "DE")
        await engine.generate("OP-001", "DE")
        with pytest.raises(RuntimeError):
            await engine.generate("OP-001", "DE")

    @pytest.mark.asyncio
    async def test_reserve_invalid_count_error(self):
        engine = SequenceManager()
        with pytest.raises(ValueError):
            await engine.reserve_sequences("OP-001", "DE", 2026, 0)

    @pytest.mark.asyncio
    async def test_reserve_exceeds_max_batch_error(self):
        engine = SequenceManager()
        with pytest.raises(ValueError):
            await engine.reserve_sequences(
                "OP-001", "DE", 2026,
                engine.config.max_batch_size + 1
            )


# ====================================================================
# Test: Full Workflow API
# ====================================================================


class TestFullWorkflowAPI:
    """Test end-to-end workflows at API level."""

    @pytest.mark.asyncio
    async def test_generate_validate_retrieve_workflow(self):
        gen = NumberGenerator()
        val = FormatValidator()

        # 1. Generate
        result = await gen.generate("OP-001", "DE", commodity="coffee")
        ref = result["reference_number"]

        # 2. Validate
        vresult = await val.validate(ref)
        assert vresult["is_valid"] is True

        # 3. Retrieve
        found = await gen.get_reference(ref)
        assert found is not None
        assert found["status"] == "active"
        assert found["commodity"] == "coffee"

    @pytest.mark.asyncio
    async def test_batch_workflow(self):
        gen = NumberGenerator()
        seq = SequenceManager()
        val = FormatValidator()

        # 1. Reserve sequences
        reserved = await seq.reserve_sequences("OP-001", "DE", 2026, 10)
        assert len(reserved) == 10

        # 2. Generate references
        refs = []
        for _ in range(10):
            result = await gen.generate("OP-001", "DE")
            refs.append(result["reference_number"])

        # 3. Validate all
        for ref in refs:
            vresult = await val.validate(ref)
            assert vresult["is_valid"] is True

        # 4. Check uniqueness
        assert len(set(refs)) == 10
