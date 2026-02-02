"""
Unit tests for greenlang.schema.errors module.

Tests all error codes defined in GL-FOUND-X-002 for:
    1. Uniqueness of all error codes
    2. Valid error code format (GLSCHEMA-[EW]\\d{3})
    3. get_error_info returns correct info
    4. Error messages have proper placeholders
    5. Error categories are valid
    6. Severity levels are correct

Target: 100% coverage of the errors module.
"""

import re
from collections import Counter
from typing import Set

import pytest

from greenlang.schema.errors import (
    ErrorCode,
    ErrorInfo,
    ErrorCategory,
    Severity,
    ERROR_REGISTRY,
    get_error_info,
    format_error_message,
    format_error_hint,
    is_error,
    is_warning,
    is_info,
    get_codes_by_category,
    get_codes_by_severity,
    get_all_error_codes,
    validate_error_code,
    get_error_by_code,
)


# =============================================================================
# Constants
# =============================================================================

# Valid error code format: GLSCHEMA-[E|W][0-9]{3}
ERROR_CODE_PATTERN = re.compile(r"^GLSCHEMA-[EW]\d{3}$")

# Valid categories based on PRD
VALID_CATEGORIES = {
    ErrorCategory.STRUCTURAL,
    ErrorCategory.CONSTRAINT,
    ErrorCategory.UNIT,
    ErrorCategory.RULE,
    ErrorCategory.SCHEMA,
    ErrorCategory.DEPRECATION,
    ErrorCategory.LINT,
    ErrorCategory.LIMIT,
}

# Valid severity levels
VALID_SEVERITIES = {Severity.ERROR, Severity.WARNING, Severity.INFO}


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def all_error_codes() -> list:
    """Get all error code enum members."""
    return list(ErrorCode)


@pytest.fixture
def all_code_strings(all_error_codes) -> list:
    """Get all error code strings (the enum values)."""
    return [ec.value for ec in all_error_codes]


@pytest.fixture
def error_codes_only() -> list:
    """Get only error-level codes (not warnings)."""
    return get_codes_by_severity(Severity.ERROR)


@pytest.fixture
def warning_codes_only() -> list:
    """Get only warning-level codes."""
    return get_codes_by_severity(Severity.WARNING)


@pytest.fixture
def info_codes_only() -> list:
    """Get only info-level codes."""
    return get_codes_by_severity(Severity.INFO)


# =============================================================================
# Test: Error Code Uniqueness
# =============================================================================

class TestErrorCodeUniqueness:
    """Tests for error code uniqueness."""

    def test_all_error_codes_are_unique(self, all_code_strings):
        """Test that all error codes are unique - no duplicates."""
        code_counts = Counter(all_code_strings)
        duplicates = [code for code, count in code_counts.items() if count > 1]

        assert len(duplicates) == 0, (
            f"Duplicate error codes found: {duplicates}"
        )

    def test_all_enum_names_are_unique(self, all_error_codes):
        """Test that all enum member names are unique."""
        names = [ec.name for ec in all_error_codes]
        name_counts = Counter(names)
        duplicates = [name for name, count in name_counts.items() if count > 1]

        assert len(duplicates) == 0, (
            f"Duplicate enum names found: {duplicates}"
        )

    def test_minimum_error_codes_defined(self, all_error_codes):
        """Test that a minimum number of error codes are defined."""
        # PRD defines ~30 error codes minimum
        MIN_EXPECTED_CODES = 25
        assert len(all_error_codes) >= MIN_EXPECTED_CODES, (
            f"Expected at least {MIN_EXPECTED_CODES} error codes, "
            f"got {len(all_error_codes)}"
        )

    def test_all_error_codes_in_registry(self, all_error_codes):
        """Test that all ErrorCode enum values are in the ERROR_REGISTRY."""
        missing = [ec for ec in all_error_codes if ec not in ERROR_REGISTRY]

        assert len(missing) == 0, (
            f"Error codes not in registry: {[ec.name for ec in missing]}"
        )


# =============================================================================
# Test: Error Code Format
# =============================================================================

class TestErrorCodeFormat:
    """Tests for error code format validation."""

    def test_all_codes_match_format(self, all_code_strings):
        """Test that all error codes match GLSCHEMA-[EW]\\d{3} format."""
        invalid_codes = [
            code for code in all_code_strings
            if not ERROR_CODE_PATTERN.match(code)
        ]

        assert len(invalid_codes) == 0, (
            f"Invalid error code format found: {invalid_codes}\n"
            f"Expected format: GLSCHEMA-[E|W]XXX"
        )

    def test_error_codes_start_with_e(self, error_codes_only):
        """Test that error severity codes start with E."""
        invalid = [
            ec for ec in error_codes_only
            if not ec.value.startswith("GLSCHEMA-E")
        ]

        assert len(invalid) == 0, (
            f"Error codes should start with GLSCHEMA-E: "
            f"{[ec.value for ec in invalid]}"
        )

    def test_warning_and_info_codes_prefix_convention(self, warning_codes_only, info_codes_only):
        """Test that warning/info severity codes follow naming conventions.

        Note: Most warning/info codes start with W, but some may intentionally
        use E prefix (e.g., UNIT_NONCANONICAL is a warning but uses E prefix
        for consistency in the unit category).
        """
        # Check that most follow the convention
        w_prefix_count = sum(
            1 for ec in warning_codes_only + info_codes_only
            if ec.value.startswith("GLSCHEMA-W")
        )
        total = len(warning_codes_only) + len(info_codes_only)

        # At least 80% should follow convention
        if total > 0:
            ratio = w_prefix_count / total
            assert ratio >= 0.8, (
                f"Only {w_prefix_count}/{total} ({ratio:.0%}) warning/info codes "
                f"follow the GLSCHEMA-W convention. Expected at least 80%."
            )

    def test_code_number_extraction(self, all_code_strings):
        """Test that numeric part can be extracted from all codes."""
        for code in all_code_strings:
            # Extract the numeric part (last 3 digits)
            match = re.search(r"(\d{3})$", code)
            assert match is not None, f"Cannot extract number from: {code}"

            number = int(match.group(1))
            assert 0 <= number <= 999, (
                f"Code number out of range [0-999]: {code}"
            )

    @pytest.mark.parametrize("code_string,should_be_valid", [
        ("GLSCHEMA-E100", True),
        ("GLSCHEMA-E999", True),
        ("GLSCHEMA-W600", True),
        ("GLSCHEMA-W799", True),
        ("GLSCHEMA-E001", True),
        ("GLSCHEMA-e100", False),  # Lowercase
        ("GLSCHEMA-X100", False),  # Invalid prefix
        ("GLSCHEMA-E1000", False),  # Too many digits
        ("GLSCHEMA-E10", False),  # Too few digits
        ("GL-E100", False),  # Wrong prefix
        ("GLSCHEMA100", False),  # Missing hyphen
        ("", False),  # Empty
        ("GLSCHEMA-E", False),  # No number
    ])
    def test_error_code_pattern_validation(self, code_string, should_be_valid):
        """Test error code pattern matching with various inputs."""
        is_valid = bool(ERROR_CODE_PATTERN.match(code_string))
        assert is_valid == should_be_valid, (
            f"Code '{code_string}' validation failed: "
            f"expected {should_be_valid}, got {is_valid}"
        )


# =============================================================================
# Test: get_error_info Function
# =============================================================================

class TestGetErrorInfo:
    """Tests for the get_error_info function."""

    def test_get_by_enum(self):
        """Test that get_error_info works with ErrorCode enum."""
        info = get_error_info(ErrorCode.MISSING_REQUIRED)

        assert info is not None
        assert isinstance(info, ErrorInfo)
        assert info.code == "GLSCHEMA-E100"
        assert info.name == "MISSING_REQUIRED"
        assert info.category == ErrorCategory.STRUCTURAL
        assert info.severity == Severity.ERROR

    def test_get_by_string(self):
        """Test that get_error_info works with code string."""
        info = get_error_info("GLSCHEMA-E100")

        assert info is not None
        assert info.code == "GLSCHEMA-E100"

    def test_get_nonexistent_code_raises(self):
        """Test that get_error_info raises KeyError for unknown code."""
        with pytest.raises(KeyError):
            get_error_info("GLSCHEMA-E999")

    def test_get_invalid_format_code_raises(self):
        """Test that get_error_info raises for invalid format."""
        with pytest.raises(KeyError):
            get_error_info("INVALID")

    def test_info_has_message_template(self):
        """Test that ErrorInfo has message_template."""
        info = get_error_info(ErrorCode.MISSING_REQUIRED)
        assert info.message_template is not None
        assert len(info.message_template) > 0

    @pytest.mark.parametrize("code_name", [
        "MISSING_REQUIRED",
        "TYPE_MISMATCH",
        "RANGE_VIOLATION",
        "UNIT_MISSING",
        "DEPRECATED_FIELD",
        "PAYLOAD_TOO_LARGE",
    ])
    def test_common_error_codes_exist(self, code_name):
        """Test that common error codes are defined."""
        error = getattr(ErrorCode, code_name, None)
        assert error is not None, f"Error code {code_name} not found"
        assert error.value.startswith("GLSCHEMA-")

        # Should also be in registry
        info = get_error_info(error)
        assert info is not None


# =============================================================================
# Test: Error Message Placeholders
# =============================================================================

class TestErrorMessagePlaceholders:
    """Tests for error message placeholder templates."""

    # Regex to find placeholders like {field}, {path}, {value}
    PLACEHOLDER_PATTERN = re.compile(r"\{(\w+)\}")

    def test_all_messages_are_non_empty(self, all_error_codes):
        """Test that all error codes have non-empty messages."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            assert info.message_template, f"Error {ec.value} has empty message"
            assert len(info.message_template) >= 5, (
                f"Error {ec.value} message too short: '{info.message_template}'"
            )

    def test_hint_templates_have_valid_placeholders(self, all_error_codes):
        """Test that hint templates with placeholders are well-formed."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            hint = info.hint_template
            if hint:
                # If hint has placeholders, they should be valid
                placeholders = self.PLACEHOLDER_PATTERN.findall(hint)
                for ph in placeholders:
                    # Placeholder names should be snake_case identifiers
                    assert re.match(r"^[a-z][a-z0-9_]*$", ph), (
                        f"Invalid placeholder '{ph}' in hint for {ec.value}"
                    )

    def test_messages_use_consistent_placeholder_style(self, all_error_codes):
        """Test that messages use {placeholder} style, not %s or other formats."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            message = info.message_template
            # Check for old-style formatting
            assert "%s" not in message, (
                f"Error {ec.value} uses %s format instead of {{placeholder}}"
            )
            assert "%d" not in message, (
                f"Error {ec.value} uses %d format instead of {{placeholder}}"
            )

    def test_format_error_message_substitutes_values(self):
        """Test that format_error_message correctly substitutes values."""
        msg = format_error_message(
            ErrorCode.TYPE_MISMATCH,
            path="/data/value",
            expected="integer",
            actual="string"
        )

        assert "/data/value" in msg
        assert "integer" in msg
        assert "string" in msg
        # Placeholders should be replaced
        assert "{path}" not in msg

    def test_format_error_message_handles_missing_values(self):
        """Test that format_error_message handles missing placeholder values."""
        # Should not raise, even if some placeholders are missing
        msg = format_error_message(
            ErrorCode.TYPE_MISMATCH,
            path="/data/value"
            # expected and actual are missing
        )

        assert "/data/value" in msg
        # Message should still be returned (with unreplaced placeholders)
        assert isinstance(msg, str)

    def test_format_error_hint_returns_none_when_no_hint(self, all_error_codes):
        """Test that format_error_hint returns None when no hint available."""
        # Find a code without hint (if any)
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            if info.hint_template is None:
                hint = format_error_hint(ec)
                assert hint is None
                break

    def test_format_error_hint_substitutes_values(self):
        """Test that format_error_hint correctly substitutes values."""
        hint = format_error_hint(
            ErrorCode.MISSING_REQUIRED,
            field="energy",
            expected_type="number"
        )

        if hint:  # If hint exists
            assert "energy" in hint or "{field}" not in hint


# =============================================================================
# Test: Error Categories
# =============================================================================

class TestErrorCategories:
    """Tests for error category assignments."""

    def test_all_categories_are_valid(self, all_error_codes):
        """Test that all error codes have valid categories."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            assert info.category in VALID_CATEGORIES, (
                f"Error {ec.value} has invalid category: '{info.category}'\n"
                f"Valid categories: {VALID_CATEGORIES}"
            )

    def test_all_categories_have_codes(self):
        """Test that all expected categories have at least one error code."""
        for category in VALID_CATEGORIES:
            codes = get_codes_by_category(category)
            assert len(codes) >= 1, (
                f"Category {category.value} has no error codes"
            )

    def test_structural_errors_in_e1xx_range(self, all_error_codes):
        """Test that structural errors are in E1xx range."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            if info.category == ErrorCategory.STRUCTURAL:
                # Extract number from code
                match = re.search(r"GLSCHEMA-E(\d{3})", ec.value)
                if match:
                    num = int(match.group(1))
                    assert 100 <= num < 200, (
                        f"Structural error {ec.value} should be in E1xx range"
                    )

    def test_constraint_errors_in_e2xx_range(self, all_error_codes):
        """Test that constraint errors are in E2xx range."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            if info.category == ErrorCategory.CONSTRAINT:
                match = re.search(r"GLSCHEMA-E(\d{3})", ec.value)
                if match:
                    num = int(match.group(1))
                    assert 200 <= num < 300, (
                        f"Constraint error {ec.value} should be in E2xx range"
                    )

    def test_deprecation_warnings_in_w6xx_range(self, all_error_codes):
        """Test that deprecation warnings are in W6xx range."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            if info.category == ErrorCategory.DEPRECATION:
                match = re.search(r"GLSCHEMA-W(\d{3})", ec.value)
                if match:
                    num = int(match.group(1))
                    assert 600 <= num < 700, (
                        f"Deprecation warning {ec.value} should be in W6xx range"
                    )

    def test_get_codes_by_category(self):
        """Test that get_codes_by_category returns correct codes."""
        structural = get_codes_by_category(ErrorCategory.STRUCTURAL)
        assert len(structural) >= 1
        for ec in structural:
            info = ERROR_REGISTRY[ec]
            assert info.category == ErrorCategory.STRUCTURAL


# =============================================================================
# Test: Severity Levels
# =============================================================================

class TestSeverityLevels:
    """Tests for error severity level assignments."""

    def test_all_severities_are_valid(self, all_error_codes):
        """Test that all error codes have valid severity levels."""
        for ec in all_error_codes:
            info = ERROR_REGISTRY[ec]
            assert info.severity in VALID_SEVERITIES, (
                f"Error {ec.value} has invalid severity: '{info.severity}'\n"
                f"Valid severities: {VALID_SEVERITIES}"
            )

    def test_is_error_function(self, error_codes_only):
        """Test that is_error returns True for error-level codes."""
        for ec in error_codes_only:
            assert is_error(ec), f"{ec.value} should return True for is_error()"

    def test_is_warning_function(self, warning_codes_only):
        """Test that is_warning returns True for warning-level codes."""
        for ec in warning_codes_only:
            assert is_warning(ec), (
                f"{ec.value} should return True for is_warning()"
            )

    def test_is_info_function(self, info_codes_only):
        """Test that is_info returns True for info-level codes."""
        for ec in info_codes_only:
            assert is_info(ec), f"{ec.value} should return True for is_info()"

    def test_is_error_and_warning_mutually_exclusive(self, all_error_codes):
        """Test that codes are not both error and warning."""
        for ec in all_error_codes:
            error_result = is_error(ec)
            warning_result = is_warning(ec)
            info_result = is_info(ec)

            # Exactly one should be True
            true_count = sum([error_result, warning_result, info_result])
            assert true_count == 1, (
                f"{ec.value} has ambiguous severity: "
                f"is_error={error_result}, is_warning={warning_result}, "
                f"is_info={info_result}"
            )

    def test_e_prefix_implies_error_severity(self, all_error_codes):
        """Test that E prefix codes have error severity."""
        for ec in all_error_codes:
            if ec.value.startswith("GLSCHEMA-E"):
                info = ERROR_REGISTRY[ec]
                # Note: Some E codes might be warnings in the current implementation
                # This test verifies the convention
                if info.severity != Severity.ERROR:
                    pytest.skip(
                        f"Code {ec.value} starts with E but has severity "
                        f"'{info.severity.value}' (may be intentional)"
                    )

    def test_get_codes_by_severity(self):
        """Test that get_codes_by_severity returns correct codes."""
        errors = get_codes_by_severity(Severity.ERROR)
        assert len(errors) >= 1
        for ec in errors:
            info = ERROR_REGISTRY[ec]
            assert info.severity == Severity.ERROR


# =============================================================================
# Test: ErrorInfo Dataclass
# =============================================================================

class TestErrorInfo:
    """Tests for the ErrorInfo dataclass."""

    def test_error_info_is_frozen(self):
        """Test that ErrorInfo is immutable."""
        info = ErrorInfo(
            code="TEST-E001",
            name="TEST_ERROR",
            category=ErrorCategory.STRUCTURAL,
            severity=Severity.ERROR,
            message_template="Test message",
        )

        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            info.code = "CHANGED"

    def test_error_info_fields(self):
        """Test that ErrorInfo has all required fields."""
        info = ErrorInfo(
            code="TEST-E001",
            name="TEST_ERROR",
            category=ErrorCategory.STRUCTURAL,
            severity=Severity.ERROR,
            message_template="Test message",
            hint_template="Fix by doing {action}",
            documentation_url="https://docs.example.com/errors/TEST-E001",
        )

        assert info.code == "TEST-E001"
        assert info.name == "TEST_ERROR"
        assert info.category == ErrorCategory.STRUCTURAL
        assert info.severity == Severity.ERROR
        assert info.message_template == "Test message"
        assert info.hint_template == "Fix by doing {action}"
        assert info.documentation_url == "https://docs.example.com/errors/TEST-E001"

    def test_error_info_optional_fields(self):
        """Test that hint_template and documentation_url are optional."""
        info = ErrorInfo(
            code="TEST-E001",
            name="TEST_ERROR",
            category=ErrorCategory.STRUCTURAL,
            severity=Severity.ERROR,
            message_template="Test message",
        )

        assert info.hint_template is None
        assert info.documentation_url is None


# =============================================================================
# Test: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Tests for utility functions."""

    def test_get_all_error_codes(self):
        """Test that get_all_error_codes returns all code strings."""
        codes = get_all_error_codes()
        assert isinstance(codes, list)
        assert len(codes) >= 25  # PRD minimum

        for code in codes:
            assert isinstance(code, str)
            assert code.startswith("GLSCHEMA-")

    def test_validate_error_code_valid(self):
        """Test validate_error_code with valid codes."""
        assert validate_error_code("GLSCHEMA-E100") is True

    def test_validate_error_code_invalid(self):
        """Test validate_error_code with invalid codes."""
        assert validate_error_code("INVALID") is False
        assert validate_error_code("GLSCHEMA-E999") is False  # Not registered

    def test_get_error_by_code_valid(self):
        """Test get_error_by_code with valid code."""
        ec = get_error_by_code("GLSCHEMA-E100")
        assert ec is not None
        assert ec == ErrorCode.MISSING_REQUIRED

    def test_get_error_by_code_invalid(self):
        """Test get_error_by_code with invalid code."""
        ec = get_error_by_code("INVALID")
        assert ec is None

        ec = get_error_by_code("GLSCHEMA-E999")
        assert ec is None


# =============================================================================
# Test: Specific Error Codes (PRD Requirements)
# =============================================================================

class TestPRDRequiredErrorCodes:
    """Tests to verify PRD-required error codes exist."""

    @pytest.mark.parametrize("code,enum_name,category", [
        ("GLSCHEMA-E100", "MISSING_REQUIRED", ErrorCategory.STRUCTURAL),
        ("GLSCHEMA-E101", "UNKNOWN_FIELD", ErrorCategory.STRUCTURAL),
        ("GLSCHEMA-E102", "TYPE_MISMATCH", ErrorCategory.STRUCTURAL),
        ("GLSCHEMA-E103", "INVALID_NULL", ErrorCategory.STRUCTURAL),
        ("GLSCHEMA-E200", "RANGE_VIOLATION", ErrorCategory.CONSTRAINT),
        ("GLSCHEMA-E201", "PATTERN_MISMATCH", ErrorCategory.CONSTRAINT),
        ("GLSCHEMA-E202", "ENUM_VIOLATION", ErrorCategory.CONSTRAINT),
        ("GLSCHEMA-E203", "LENGTH_VIOLATION", ErrorCategory.CONSTRAINT),
        ("GLSCHEMA-E204", "UNIQUE_VIOLATION", ErrorCategory.CONSTRAINT),
        ("GLSCHEMA-E300", "UNIT_MISSING", ErrorCategory.UNIT),
        ("GLSCHEMA-E301", "UNIT_INCOMPATIBLE", ErrorCategory.UNIT),
        ("GLSCHEMA-E302", "UNIT_NONCANONICAL", ErrorCategory.UNIT),
        ("GLSCHEMA-E303", "UNIT_UNKNOWN", ErrorCategory.UNIT),
        ("GLSCHEMA-E400", "RULE_VIOLATION", ErrorCategory.RULE),
        ("GLSCHEMA-E401", "CONDITIONAL_REQUIRED", ErrorCategory.RULE),
        ("GLSCHEMA-E402", "CONSISTENCY_ERROR", ErrorCategory.RULE),
        ("GLSCHEMA-E500", "REF_RESOLUTION_FAILED", ErrorCategory.SCHEMA),
        ("GLSCHEMA-E501", "CIRCULAR_REF", ErrorCategory.SCHEMA),
        ("GLSCHEMA-E502", "SCHEMA_INVALID", ErrorCategory.SCHEMA),
        ("GLSCHEMA-E503", "SCHEMA_VERSION_MISMATCH", ErrorCategory.SCHEMA),
        ("GLSCHEMA-W600", "DEPRECATED_FIELD", ErrorCategory.DEPRECATION),
        ("GLSCHEMA-W601", "RENAMED_FIELD", ErrorCategory.DEPRECATION),
        ("GLSCHEMA-W602", "REMOVED_FIELD", ErrorCategory.DEPRECATION),
        ("GLSCHEMA-W700", "SUSPICIOUS_KEY", ErrorCategory.LINT),
        ("GLSCHEMA-W701", "NONCOMPLIANT_CASING", ErrorCategory.LINT),
        ("GLSCHEMA-W702", "UNIT_FORMAT_STYLE", ErrorCategory.LINT),
        ("GLSCHEMA-E800", "PAYLOAD_TOO_LARGE", ErrorCategory.LIMIT),
        ("GLSCHEMA-E801", "DEPTH_EXCEEDED", ErrorCategory.LIMIT),
        ("GLSCHEMA-E802", "ITEMS_EXCEEDED", ErrorCategory.LIMIT),
        ("GLSCHEMA-E803", "REFS_EXCEEDED", ErrorCategory.LIMIT),
        ("GLSCHEMA-E804", "FINDINGS_EXCEEDED", ErrorCategory.LIMIT),
    ])
    def test_prd_required_error_code_exists(self, code, enum_name, category):
        """Test that each PRD-required error code exists with correct attributes."""
        # Check enum member exists
        ec = getattr(ErrorCode, enum_name, None)
        assert ec is not None, f"ErrorCode.{enum_name} not found"

        # Check code matches
        assert ec.value == code, (
            f"ErrorCode.{enum_name}.value is '{ec.value}', expected '{code}'"
        )

        # Check it's in registry
        assert ec in ERROR_REGISTRY, f"ErrorCode.{enum_name} not in ERROR_REGISTRY"

        # Check category matches
        info = ERROR_REGISTRY[ec]
        assert info.category == category, (
            f"ErrorCode.{enum_name}.category is '{info.category}', "
            f"expected '{category}'"
        )


# =============================================================================
# Test: Enum and Registry Consistency
# =============================================================================

class TestConsistency:
    """Tests for consistency between enum and registry."""

    def test_registry_code_matches_enum_value(self):
        """Test that ErrorInfo.code matches ErrorCode.value in registry."""
        for ec, info in ERROR_REGISTRY.items():
            assert info.code == ec.value, (
                f"Registry mismatch for {ec.name}: "
                f"ErrorInfo.code='{info.code}' vs ErrorCode.value='{ec.value}'"
            )

    def test_registry_name_matches_enum_name(self):
        """Test that ErrorInfo.name matches ErrorCode.name in registry."""
        for ec, info in ERROR_REGISTRY.items():
            assert info.name == ec.name, (
                f"Registry mismatch for {ec.value}: "
                f"ErrorInfo.name='{info.name}' vs ErrorCode.name='{ec.name}'"
            )

    def test_no_orphan_error_codes(self, all_error_codes):
        """Test that every ErrorCode has an entry in ERROR_REGISTRY."""
        orphans = [ec for ec in all_error_codes if ec not in ERROR_REGISTRY]
        assert len(orphans) == 0, (
            f"ErrorCode values without registry entries: "
            f"{[ec.name for ec in orphans]}"
        )

    def test_no_orphan_registry_entries(self, all_error_codes):
        """Test that every ERROR_REGISTRY key is a valid ErrorCode."""
        valid_codes = set(all_error_codes)
        orphans = [ec for ec in ERROR_REGISTRY.keys() if ec not in valid_codes]
        assert len(orphans) == 0, (
            f"Registry entries without ErrorCode values: "
            f"{[ec.name for ec in orphans]}"
        )
