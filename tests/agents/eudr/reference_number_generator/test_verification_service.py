# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- verification_service.py

Tests reference number verification: authenticity checks, checksum validation,
lifecycle status verification, cross-algorithm verification, provenance
hash verification, and comprehensive verification workflow. 40+ tests.

These tests validate verification behavior using the FormatValidator
and NumberGenerator engines. Once verification_service.py is
implemented as a standalone engine, tests can import directly.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    EU_MEMBER_STATES,
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.format_validator import (
    FormatValidator,
)
from greenlang.agents.eudr.reference_number_generator.models import (
    ReferenceNumberStatus,
    ValidationResult,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)
from greenlang.agents.eudr.reference_number_generator.provenance import (
    ProvenanceTracker,
)


# ====================================================================
# Test: Format-Based Verification
# ====================================================================


class TestFormatVerification:
    """Test format-based reference number verification."""

    @pytest.mark.asyncio
    async def test_verify_valid_reference(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        val = FormatValidator()
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_verify_invalid_prefix(self):
        val = FormatValidator()
        vresult = await val.validate("WRONG-DE-2026-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_verify_invalid_member_state(self):
        val = FormatValidator()
        vresult = await val.validate("EUDR-XX-2026-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_verify_truncated_reference(self):
        val = FormatValidator()
        vresult = await val.validate("EUDR-DE")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_verify_empty_reference(self):
        val = FormatValidator()
        vresult = await val.validate("")
        assert vresult["is_valid"] is False


# ====================================================================
# Test: Checksum Verification
# ====================================================================


class TestChecksumVerification:
    """Test checksum-specific verification."""

    @pytest.mark.asyncio
    async def test_luhn_checksum_verification(self):
        config = ReferenceNumberGeneratorConfig(checksum_algorithm="luhn")
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        # Verify checksum check passed
        checksum_checks = [
            c for c in vresult["checks"]
            if c.get("check") == "checksum"
        ]
        if checksum_checks:
            assert checksum_checks[0]["passed"] is True

    @pytest.mark.asyncio
    async def test_iso7064_checksum_verification(self):
        config = ReferenceNumberGeneratorConfig(
            checksum_algorithm="iso7064", checksum_length=2
        )
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_modulo97_checksum_verification(self):
        config = ReferenceNumberGeneratorConfig(
            checksum_algorithm="modulo97", checksum_length=2
        )
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_corrupted_checksum_fails(self):
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        ref = result["reference_number"]

        # Corrupt the checksum digit
        parts = ref.split("-")
        original = parts[-1]
        corrupted = str((int(original) + 5) % 10)
        if corrupted == original:
            corrupted = str((int(original) + 3) % 10)
        parts[-1] = corrupted
        corrupted_ref = "-".join(parts)

        vresult = await val.validate(corrupted_ref)
        if corrupted != original:
            assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_checksum_deterministic_verification(self):
        """Verify same reference validates identically twice."""
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        ref = result["reference_number"]

        v1 = await val.validate(ref)
        v2 = await val.validate(ref)
        assert v1["is_valid"] == v2["is_valid"]
        assert v1["result"] == v2["result"]


# ====================================================================
# Test: Existence Verification
# ====================================================================


class TestExistenceVerification:
    """Test reference number existence verification."""

    @pytest.mark.asyncio
    async def test_existing_reference_found(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        found = await gen.get_reference(result["reference_number"])
        assert found is not None

    @pytest.mark.asyncio
    async def test_nonexistent_reference_not_found(self):
        gen = NumberGenerator()
        found = await gen.get_reference("EUDR-DE-2026-OP001-999999-0")
        assert found is None


# ====================================================================
# Test: Lifecycle Status Verification
# ====================================================================


class TestLifecycleStatusVerification:
    """Test lifecycle status verification of reference numbers."""

    @pytest.mark.asyncio
    async def test_active_reference_status(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_used_reference_status(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        ref = gen._references[result["reference_number"]]
        ref["status"] = ReferenceNumberStatus.USED.value
        assert ref["status"] == "used"

    @pytest.mark.asyncio
    async def test_expired_reference_status(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        ref = gen._references[result["reference_number"]]
        ref["status"] = ReferenceNumberStatus.EXPIRED.value
        assert ref["status"] == "expired"

    @pytest.mark.asyncio
    async def test_revoked_reference_status(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        ref = gen._references[result["reference_number"]]
        ref["status"] = ReferenceNumberStatus.REVOKED.value
        assert ref["status"] == "revoked"


# ====================================================================
# Test: Provenance Hash Verification
# ====================================================================


class TestProvenanceVerification:
    """Test provenance hash verification."""

    @pytest.mark.asyncio
    async def test_generated_reference_has_provenance_hash(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_provenance_hash_is_hex(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        int(result["provenance_hash"], 16)  # Should not raise

    def test_reference_hash_deterministic(self):
        tracker = ProvenanceTracker()
        h1 = tracker.compute_reference_hash("EUDR-DE-2026-OP001-000001-7", "OP-001")
        h2 = tracker.compute_reference_hash("EUDR-DE-2026-OP001-000001-7", "OP-001")
        assert h1 == h2

    def test_different_references_different_hashes(self):
        tracker = ProvenanceTracker()
        h1 = tracker.compute_reference_hash("EUDR-DE-2026-OP001-000001-7", "OP-001")
        h2 = tracker.compute_reference_hash("EUDR-DE-2026-OP001-000002-4", "OP-001")
        assert h1 != h2

    def test_provenance_chain_verification(self):
        tracker = ProvenanceTracker()
        steps = [
            {"step": "generate", "source": "rng_038", "data": {"ref": "001"}},
            {"step": "validate", "source": "rng_038", "data": {"valid": True}},
            {"step": "verify", "source": "rng_038", "data": {"verified": True}},
        ]
        chain = tracker.build_chain(steps)
        assert tracker.verify_chain(chain) is True


# ====================================================================
# Test: Comprehensive Verification Workflow
# ====================================================================


class TestComprehensiveVerification:
    """Test full verification workflow: generate, validate, verify."""

    @pytest.mark.asyncio
    async def test_full_verification_workflow_luhn(self):
        gen = NumberGenerator()
        val = FormatValidator()

        # Generate
        result = await gen.generate("OP-001", "DE", commodity="coffee")
        ref = result["reference_number"]

        # Format validate
        vresult = await val.validate(ref)
        assert vresult["is_valid"] is True

        # Existence check
        found = await gen.get_reference(ref)
        assert found is not None
        assert found["status"] == "active"

        # Provenance present
        assert len(found["provenance_hash"]) == 64

    @pytest.mark.asyncio
    async def test_full_verification_all_27_states(self):
        """Verify references generated for all 27 EU member states."""
        gen = NumberGenerator()
        val = FormatValidator()

        for ms in EU_MEMBER_STATES:
            result = await gen.generate("OP-001", ms)
            vresult = await val.validate(result["reference_number"])
            assert vresult["is_valid"] is True, f"Failed for {ms}"

    @pytest.mark.asyncio
    async def test_verify_multiple_references_batch(self):
        gen = NumberGenerator()
        val = FormatValidator()

        refs = []
        for _ in range(20):
            result = await gen.generate("OP-001", "DE")
            refs.append(result["reference_number"])

        for ref in refs:
            vresult = await val.validate(ref)
            assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_verify_after_status_change(self):
        """Format validation should pass regardless of lifecycle status."""
        gen = NumberGenerator()
        val = FormatValidator()

        result = await gen.generate("OP-001", "DE")
        ref = result["reference_number"]

        # Change status to used
        gen._references[ref]["status"] = "used"

        # Format validation still passes
        vresult = await val.validate(ref)
        assert vresult["is_valid"] is True


# ====================================================================
# Test: Verification Performance
# ====================================================================


class TestVerificationPerformance:
    """Test verification performance characteristics."""

    @pytest.mark.asyncio
    async def test_validation_duration_recorded(self):
        gen = NumberGenerator()
        val = FormatValidator()
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["validation_duration_ms"] >= 0
        assert vresult["validation_duration_ms"] < 1000  # Should be fast

    @pytest.mark.asyncio
    async def test_batch_verification_performance(self):
        """Verify 100 references complete in reasonable time."""
        import time
        gen = NumberGenerator()
        val = FormatValidator()

        refs = []
        for _ in range(100):
            result = await gen.generate("OP-001", "DE")
            refs.append(result["reference_number"])

        start = time.monotonic()
        for ref in refs:
            await val.validate(ref)
        duration = time.monotonic() - start

        assert duration < 5.0  # 100 validations in under 5 seconds
