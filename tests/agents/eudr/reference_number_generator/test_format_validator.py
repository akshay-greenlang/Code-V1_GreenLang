# -*- coding: utf-8 -*-
"""
Unit tests for AGENT-EUDR-038 Reference Number Generator -- format_validator.py

Tests reference number format validation across all 27 EU member states,
checksum verification (Luhn, ISO 7064), component parsing, regex patterns,
invalid formats, and edge cases. 50+ tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.reference_number_generator.config import (
    EU_MEMBER_STATES,
    ReferenceNumberGeneratorConfig,
)
from greenlang.agents.eudr.reference_number_generator.format_validator import (
    FormatValidator,
)
from greenlang.agents.eudr.reference_number_generator.models import (
    ValidationResult,
)
from greenlang.agents.eudr.reference_number_generator.number_generator import (
    NumberGenerator,
)


# ====================================================================
# Test: Initialization
# ====================================================================


class TestFormatValidatorInit:
    """Test FormatValidator initialization."""

    def test_init_with_default_config(self):
        engine = FormatValidator()
        assert engine.config is not None
        assert engine.validation_count == 0

    def test_init_with_custom_config(self, custom_config):
        engine = FormatValidator(config=custom_config)
        assert engine.config.reference_prefix == "TEST"

    def test_init_builds_format_rules(self, format_validator_engine):
        rules = format_validator_engine._format_rules
        assert len(rules) == 27

    def test_init_format_rules_for_all_member_states(self, format_validator_engine):
        for code in EU_MEMBER_STATES:
            assert code in format_validator_engine._format_rules


# ====================================================================
# Test: validate() - Valid References
# ====================================================================


class TestValidateValidReferences:
    """Test validation of correctly formatted reference numbers."""

    @pytest.mark.asyncio
    async def test_validate_generated_reference(self):
        """Generate a reference, then validate it -- must pass."""
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        ref = result["reference_number"]

        validator = FormatValidator()
        vresult = await validator.validate(ref)
        assert vresult["is_valid"] is True
        assert vresult["result"] == ValidationResult.VALID.value

    @pytest.mark.asyncio
    async def test_validate_increments_counter(self, format_validator_engine):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        await format_validator_engine.validate(result["reference_number"])
        assert format_validator_engine.validation_count == 1

    @pytest.mark.asyncio
    async def test_validate_returns_checks(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        validator = FormatValidator()
        vresult = await validator.validate(result["reference_number"])
        assert "checks" in vresult
        assert len(vresult["checks"]) >= 4

    @pytest.mark.asyncio
    async def test_validate_duration_recorded(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        validator = FormatValidator()
        vresult = await validator.validate(result["reference_number"])
        assert "validation_duration_ms" in vresult
        assert vresult["validation_duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_validate_timestamp_present(self):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        validator = FormatValidator()
        vresult = await validator.validate(result["reference_number"])
        assert "validated_at" in vresult


# ====================================================================
# Test: validate() - Invalid References
# ====================================================================


class TestValidateInvalidReferences:
    """Test validation of incorrectly formatted reference numbers."""

    @pytest.mark.asyncio
    async def test_validate_empty_string(self, format_validator_engine):
        vresult = await format_validator_engine.validate("")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_whitespace_string(self, format_validator_engine):
        vresult = await format_validator_engine.validate("   ")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_wrong_prefix(self, format_validator_engine):
        vresult = await format_validator_engine.validate("WRONG-DE-2026-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_invalid_member_state(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-XX-2026-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_too_few_parts(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-2026")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_too_many_parts(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-2026-OP001-000001-7-EXTRA")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_non_numeric_year(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-ABCD-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_year_out_of_range_low(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-2023-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_year_out_of_range_high(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-2100-OP001-000001-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_non_numeric_sequence(self, format_validator_engine):
        vresult = await format_validator_engine.validate("EUDR-DE-2026-OP001-ABCDEF-7")
        assert vresult["is_valid"] is False

    @pytest.mark.asyncio
    async def test_validate_wrong_checksum(self):
        """Generate a valid reference, alter the checksum, and verify it fails."""
        gen = NumberGenerator()
        result = await gen.generate("OP-001", "DE")
        ref = result["reference_number"]
        # Alter checksum
        parts = ref.split("-")
        original_checksum = parts[-1]
        wrong_checksum = str((int(original_checksum) + 1) % 10)
        parts[-1] = wrong_checksum
        altered_ref = "-".join(parts)

        validator = FormatValidator()
        vresult = await validator.validate(altered_ref)
        # Could be invalid checksum or overall invalid
        if original_checksum != wrong_checksum:
            assert vresult["is_valid"] is False


# ====================================================================
# Test: _parse_components
# ====================================================================


class TestParseComponents:
    """Test reference number component parsing."""

    def test_parse_valid_components(self, format_validator_engine):
        result = format_validator_engine._parse_components("EUDR-DE-2026-OP001-000001-7")
        assert result is not None
        assert result["prefix"] == "EUDR"
        assert result["member_state"] == "DE"
        assert result["year"] == 2026
        assert result["operator_code"] == "OP001"
        assert result["sequence"] == 1
        assert result["checksum"] == "7"

    def test_parse_returns_none_for_too_few_parts(self, format_validator_engine):
        result = format_validator_engine._parse_components("EUDR-DE-2026")
        assert result is None

    def test_parse_returns_none_for_bad_year(self, format_validator_engine):
        result = format_validator_engine._parse_components("EUDR-DE-XXXX-OP001-000001-7")
        assert result is None

    def test_parse_returns_none_for_bad_sequence(self, format_validator_engine):
        result = format_validator_engine._parse_components("EUDR-DE-2026-OP001-ABCDEF-7")
        assert result is None


# ====================================================================
# Test: Checksum Algorithms
# ====================================================================


class TestFormatValidatorChecksums:
    """Test checksum computation in FormatValidator."""

    def test_to_numeric_digits(self):
        assert FormatValidator._to_numeric("123") == "123"

    def test_to_numeric_letters(self):
        assert FormatValidator._to_numeric("AB") == "1011"

    def test_luhn_checksum_single_digit(self):
        result = FormatValidator._luhn_checksum("7992739871")
        assert len(result) == 1
        assert result.isdigit()

    def test_luhn_checksum_deterministic(self):
        r1 = FormatValidator._luhn_checksum("7992739871")
        r2 = FormatValidator._luhn_checksum("7992739871")
        assert r1 == r2

    def test_iso7064_checksum_two_digits(self):
        result = FormatValidator._iso7064_checksum("123456789")
        assert len(result) == 2
        assert result.isdigit()

    def test_recompute_checksum_matches_generator(self):
        """FormatValidator checksum must match NumberGenerator checksum."""
        gen = NumberGenerator()
        val = FormatValidator()

        # Use same inputs
        numeric_gen = gen._to_numeric("DE2026OP0011")
        numeric_val = val._to_numeric("DE2026OP0011")
        assert numeric_gen == numeric_val

        luhn_gen = gen._luhn_checksum(numeric_gen)
        luhn_val = val._luhn_checksum(numeric_val)
        assert luhn_gen == luhn_val


# ====================================================================
# Test: get_format_rules
# ====================================================================


class TestGetFormatRules:
    """Test format rule retrieval."""

    def test_get_format_rules_de(self, format_validator_engine):
        rule = format_validator_engine.get_format_rules("DE")
        assert rule is not None
        assert rule["member_state"] == "DE"
        assert rule["country_name"] == "Germany"
        assert rule["prefix"] == "EUDR"

    def test_get_format_rules_case_insensitive(self, format_validator_engine):
        rule = format_validator_engine.get_format_rules("de")
        assert rule is not None
        assert rule["member_state"] == "DE"

    def test_get_format_rules_nonexistent(self, format_validator_engine):
        rule = format_validator_engine.get_format_rules("XX")
        assert rule is None

    def test_get_all_format_rules(self, format_validator_engine):
        all_rules = format_validator_engine.get_all_format_rules()
        assert len(all_rules) == 27

    def test_format_rule_has_regex(self, format_validator_engine):
        rule = format_validator_engine.get_format_rules("DE")
        assert "regex" in rule
        assert len(rule["regex"]) > 0

    def test_format_rule_has_example(self, format_validator_engine):
        rule = format_validator_engine.get_format_rules("DE")
        assert "example" in rule
        assert "EUDR" in rule["example"]


# ====================================================================
# Test: All 27 Member States Validation
# ====================================================================


class TestAllMemberStatesValidation:
    """Test validation for references generated across all 27 EU member states."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("member_state", list(EU_MEMBER_STATES.keys()))
    async def test_validate_generated_for_each_state(self, member_state):
        gen = NumberGenerator()
        result = await gen.generate("OP-001", member_state)
        ref = result["reference_number"]

        validator = FormatValidator()
        vresult = await validator.validate(ref)
        assert vresult["is_valid"] is True, (
            f"Validation failed for {member_state}: {vresult}"
        )


# ====================================================================
# Test: Health Check
# ====================================================================


class TestFormatValidatorHealthCheck:
    """Test engine health check."""

    @pytest.mark.asyncio
    async def test_health_check(self, format_validator_engine):
        health = await format_validator_engine.health_check()
        assert health["status"] == "available"
        assert health["member_state_rules"] == "27"
        assert "total_validations" in health


# ====================================================================
# Test: Cross-Algorithm Validation
# ====================================================================


class TestCrossAlgorithmValidation:
    """Test validation with different checksum algorithms."""

    @pytest.mark.asyncio
    async def test_luhn_generate_and_validate(self):
        config = ReferenceNumberGeneratorConfig(checksum_algorithm="luhn")
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_iso7064_generate_and_validate(self):
        config = ReferenceNumberGeneratorConfig(
            checksum_algorithm="iso7064", checksum_length=2
        )
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True

    @pytest.mark.asyncio
    async def test_modulo97_generate_and_validate(self):
        config = ReferenceNumberGeneratorConfig(
            checksum_algorithm="modulo97", checksum_length=2
        )
        gen = NumberGenerator(config=config)
        val = FormatValidator(config=config)
        result = await gen.generate("OP-001", "DE")
        vresult = await val.validate(result["reference_number"])
        assert vresult["is_valid"] is True
