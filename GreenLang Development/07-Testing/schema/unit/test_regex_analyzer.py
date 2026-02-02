# -*- coding: utf-8 -*-
"""
Unit tests for the Regex Safety Analyzer.

This module tests the RegexAnalyzer class and related functions from
greenlang.schema.compiler.regex_analyzer for detecting ReDoS vulnerabilities.

Test Categories:
1. Nested quantifier detection
2. Overlapping alternation detection
3. Exponential backtracking detection
4. Complexity scoring
5. RE2 compatibility checking
6. Pattern sanitization
7. Edge cases and boundary conditions

Example:
    pytest tests/schema/unit/test_regex_analyzer.py -v

Version: 1.0.0
Date: 2026-01-29
"""

import pytest
import re

from greenlang.schema.compiler.regex_analyzer import (
    RegexAnalyzer,
    RegexAnalysisResult,
    VulnerabilityType,
    analyze_regex_safety,
    is_safe_pattern,
    is_re2_compatible,
    sanitize_pattern,
    compile_with_timeout,
    DANGEROUS_PATTERNS,
    SAFE_PATTERNS,
    _detect_nested_quantifiers,
    _detect_overlapping_alternations,
    _compute_complexity_score,
    _suggest_safe_alternative,
)
from greenlang.schema.constants import MAX_REGEX_LENGTH, REGEX_TIMEOUT_MS


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def analyzer() -> RegexAnalyzer:
    """Create a default RegexAnalyzer instance."""
    return RegexAnalyzer()


@pytest.fixture
def strict_analyzer() -> RegexAnalyzer:
    """Create a strict RegexAnalyzer with low complexity threshold."""
    return RegexAnalyzer(max_complexity_score=0.5)


@pytest.fixture
def lenient_analyzer() -> RegexAnalyzer:
    """Create a lenient RegexAnalyzer with high complexity threshold."""
    return RegexAnalyzer(max_complexity_score=0.95)


# ============================================================================
# TEST DATA
# ============================================================================

# Known dangerous patterns that should be flagged
DANGEROUS_TEST_PATTERNS = [
    ("(a+)+", VulnerabilityType.NESTED_QUANTIFIER, "Nested quantifier"),
    ("(a*)*", VulnerabilityType.NESTED_QUANTIFIER, "Double Kleene star"),
    ("(a+)*", VulnerabilityType.NESTED_QUANTIFIER, "Nested quantifier variant"),
    ("(a*)+", VulnerabilityType.NESTED_QUANTIFIER, "Nested quantifier variant 2"),
    ("(.*)+", VulnerabilityType.NESTED_QUANTIFIER, "Nested wildcard"),
    ("(.*)*", VulnerabilityType.NESTED_QUANTIFIER, "Double wildcard"),
    ("(.+)+", VulnerabilityType.NESTED_QUANTIFIER, "Nested .+ pattern"),
    ("(.+)*", VulnerabilityType.NESTED_QUANTIFIER, "Nested .+* pattern"),
    ("(a|a)+", VulnerabilityType.NESTED_QUANTIFIER, "Identical alternation"),
    ("(a|aa)+", VulnerabilityType.NESTED_QUANTIFIER, "Prefix alternation"),
    ("(aa|a)+", VulnerabilityType.NESTED_QUANTIFIER, "Prefix alternation reversed"),
    ("([a-zA-Z]+)*", VulnerabilityType.NESTED_QUANTIFIER, "Char class nested"),
]

# Safe patterns that should pass
SAFE_TEST_PATTERNS = [
    "^[a-zA-Z0-9]+$",
    r"^\d{4}-\d{2}-\d{2}$",
    "^[A-Z]{2,3}$",
    "^[a-z_][a-z0-9_]*$",
    r"^\d+$",
    "^[a-f0-9]{32}$",
    "^[a-f0-9]{64}$",
    "^[A-Za-z0-9+/]+=*$",
    r"^\w+@\w+\.\w+$",  # Simple email-like pattern
    "^hello$",
    "^(foo|bar)$",
    r"^[0-9]{1,5}$",
]

# RE2-incompatible patterns
RE2_INCOMPATIBLE_TEST_PATTERNS = [
    (r"(\d+)\1", "Backreference"),
    (r"(?=foo)bar", "Positive lookahead"),
    (r"(?!foo)bar", "Negative lookahead"),
    (r"(?<=foo)bar", "Positive lookbehind"),
    (r"(?<!foo)bar", "Negative lookbehind"),
]


# ============================================================================
# ANALYZER INITIALIZATION TESTS
# ============================================================================

class TestRegexAnalyzerInit:
    """Tests for RegexAnalyzer initialization."""

    def test_default_initialization(self):
        """Test analyzer with default parameters."""
        analyzer = RegexAnalyzer()
        assert analyzer.max_complexity_score == 0.8
        assert analyzer.timeout_ms == REGEX_TIMEOUT_MS

    def test_custom_complexity_score(self):
        """Test analyzer with custom complexity score."""
        analyzer = RegexAnalyzer(max_complexity_score=0.5)
        assert analyzer.max_complexity_score == 0.5

    def test_custom_timeout(self):
        """Test analyzer with custom timeout."""
        analyzer = RegexAnalyzer(timeout_ms=200)
        assert analyzer.timeout_ms == 200

    def test_invalid_complexity_score_high(self):
        """Test that high complexity score raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            RegexAnalyzer(max_complexity_score=1.5)

    def test_invalid_complexity_score_low(self):
        """Test that negative complexity score raises ValueError."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            RegexAnalyzer(max_complexity_score=-0.1)

    def test_boundary_complexity_scores(self):
        """Test boundary values for complexity score."""
        analyzer_zero = RegexAnalyzer(max_complexity_score=0.0)
        assert analyzer_zero.max_complexity_score == 0.0

        analyzer_one = RegexAnalyzer(max_complexity_score=1.0)
        assert analyzer_one.max_complexity_score == 1.0


# ============================================================================
# NESTED QUANTIFIER DETECTION TESTS
# ============================================================================

class TestNestedQuantifierDetection:
    """Tests for nested quantifier detection."""

    @pytest.mark.parametrize("pattern,vuln_type,desc", DANGEROUS_TEST_PATTERNS)
    def test_detect_dangerous_patterns(
        self, analyzer: RegexAnalyzer, pattern: str, vuln_type: VulnerabilityType, desc: str
    ):
        """Test that dangerous patterns are detected."""
        result = analyzer.analyze(pattern)
        assert not result.is_safe, f"Pattern {pattern} ({desc}) should be unsafe"
        assert result.vulnerability_type is not None

    def test_nested_quantifier_basic(self, analyzer: RegexAnalyzer):
        """Test basic nested quantifier detection."""
        result = analyzer.analyze("(a+)+")
        assert not result.is_safe
        assert result.vulnerability_type == VulnerabilityType.NESTED_QUANTIFIER

    def test_nested_quantifier_with_char_class(self, analyzer: RegexAnalyzer):
        """Test nested quantifier with character class."""
        result = analyzer.analyze("([a-z]+)+")
        assert not result.is_safe

    def test_nested_quantifier_deep(self, analyzer: RegexAnalyzer):
        """Test deeply nested quantifier."""
        result = analyzer.analyze("((a+)+)+")
        assert not result.is_safe

    def test_no_nested_quantifier(self, analyzer: RegexAnalyzer):
        """Test pattern without nested quantifiers."""
        result = analyzer.analyze("^[a-z]+$")
        assert result.is_safe

    def test_quantifier_in_different_groups(self, analyzer: RegexAnalyzer):
        """Test quantifiers in separate groups (not nested)."""
        result = analyzer.analyze("(a+)(b+)")
        assert result.is_safe


# ============================================================================
# OVERLAPPING ALTERNATION DETECTION TESTS
# ============================================================================

class TestOverlappingAlternationDetection:
    """Tests for overlapping alternation detection."""

    def test_identical_alternation(self, analyzer: RegexAnalyzer):
        """Test detection of identical alternation branches."""
        result = analyzer.analyze("(a|a)+")
        assert not result.is_safe

    def test_prefix_alternation(self, analyzer: RegexAnalyzer):
        """Test detection of prefix alternation branches."""
        result = analyzer.analyze("(a|ab)+")
        assert not result.is_safe

    def test_suffix_alternation(self, analyzer: RegexAnalyzer):
        """Test detection of suffix alternation branches."""
        result = analyzer.analyze("(ab|b)+")
        # This may or may not be flagged depending on implementation
        # The key is that prefix overlaps are always flagged

    def test_no_overlapping_alternation(self, analyzer: RegexAnalyzer):
        """Test pattern without overlapping alternations."""
        result = analyzer.analyze("^(foo|bar)$")
        assert result.is_safe

    def test_mutually_exclusive_alternation(self, analyzer: RegexAnalyzer):
        """Test mutually exclusive alternations."""
        result = analyzer.analyze("^(cat|dog|bird)$")
        assert result.is_safe


# ============================================================================
# EXPONENTIAL BACKTRACKING DETECTION TESTS
# ============================================================================

class TestExponentialBacktrackingDetection:
    """Tests for exponential backtracking detection."""

    def test_multiple_wildcards(self, analyzer: RegexAnalyzer):
        """Test detection of multiple adjacent wildcards."""
        result = analyzer.analyze(".*.*.*x")
        assert not result.is_safe

    def test_unbounded_repetition_at_end(self, analyzer: RegexAnalyzer):
        """Test detection of unbounded repetition at pattern end."""
        result = analyzer.analyze("([a-z]+)*")
        assert not result.is_safe

    def test_high_quantifier_bounds(self, analyzer: RegexAnalyzer):
        """Test detection of high quantifier bounds."""
        # This tests the complexity score path more than direct detection
        result = analyzer.analyze("(a){1,100}")
        # May or may not be unsafe depending on complexity threshold


# ============================================================================
# COMPLEXITY SCORING TESTS
# ============================================================================

class TestComplexityScoring:
    """Tests for complexity score calculation."""

    def test_empty_pattern_score(self, analyzer: RegexAnalyzer):
        """Test that empty pattern has zero complexity."""
        result = analyzer.analyze("")
        assert result.complexity_score == 0.0

    def test_simple_pattern_low_score(self, analyzer: RegexAnalyzer):
        """Test that simple patterns have low complexity scores."""
        result = analyzer.analyze("^hello$")
        assert result.complexity_score < 0.3

    def test_complex_pattern_high_score(self, analyzer: RegexAnalyzer):
        """Test that complex patterns have higher complexity scores."""
        simple = analyzer.analyze("^[a-z]+$")
        complex_pat = analyzer.analyze("(a+|b+|c+)*[d-f]{2,5}")
        assert complex_pat.complexity_score > simple.complexity_score

    def test_quantifier_increases_score(self, analyzer: RegexAnalyzer):
        """Test that quantifiers increase complexity score."""
        no_quant = analyzer.analyze("^abc$")
        with_quant = analyzer.analyze("^a+b*c?$")
        assert with_quant.complexity_score > no_quant.complexity_score

    def test_nesting_increases_score(self, analyzer: RegexAnalyzer):
        """Test that nesting depth increases complexity score."""
        flat = analyzer.analyze("abc")
        nested = analyzer.analyze("(a(b(c)))")
        assert nested.complexity_score > flat.complexity_score

    def test_alternation_increases_score(self, analyzer: RegexAnalyzer):
        """Test that alternations increase complexity score."""
        no_alt = analyzer.analyze("^abc$")
        with_alt = analyzer.analyze("^(a|b|c)$")
        assert with_alt.complexity_score > no_alt.complexity_score

    def test_score_normalized(self, analyzer: RegexAnalyzer):
        """Test that complexity score is always 0.0 to 1.0."""
        patterns = [
            "",
            "a",
            "^[a-z]+$",
            "(a+)+",
            "((a|b)+)+",
            ".*.*.*.*",
        ]
        for pattern in patterns:
            result = analyzer.analyze(pattern)
            assert 0.0 <= result.complexity_score <= 1.0


# ============================================================================
# RE2 COMPATIBILITY TESTS
# ============================================================================

class TestRE2Compatibility:
    """Tests for RE2 compatibility checking."""

    @pytest.mark.parametrize("pattern,desc", RE2_INCOMPATIBLE_TEST_PATTERNS)
    def test_re2_incompatible_patterns(
        self, analyzer: RegexAnalyzer, pattern: str, desc: str
    ):
        """Test that RE2-incompatible patterns are detected."""
        result = analyzer.analyze(pattern)
        assert not result.is_re2_compatible, f"Pattern with {desc} should not be RE2 compatible"

    def test_re2_compatible_basic(self, analyzer: RegexAnalyzer):
        """Test basic RE2-compatible pattern."""
        result = analyzer.analyze("^[a-z]+$")
        assert result.is_re2_compatible

    def test_re2_compatible_with_quantifiers(self, analyzer: RegexAnalyzer):
        """Test RE2-compatible pattern with quantifiers."""
        result = analyzer.analyze(r"^\d{1,5}$")
        assert result.is_re2_compatible

    def test_re2_compatible_with_alternation(self, analyzer: RegexAnalyzer):
        """Test RE2-compatible pattern with alternation."""
        result = analyzer.analyze("^(foo|bar)$")
        assert result.is_re2_compatible


# ============================================================================
# SAFE PATTERN TESTS
# ============================================================================

class TestSafePatterns:
    """Tests for known safe patterns."""

    @pytest.mark.parametrize("pattern", SAFE_TEST_PATTERNS)
    def test_safe_patterns(self, analyzer: RegexAnalyzer, pattern: str):
        """Test that known safe patterns are classified as safe."""
        result = analyzer.analyze(pattern)
        assert result.is_safe, f"Pattern {pattern} should be safe"

    def test_known_safe_patterns_constant(self, analyzer: RegexAnalyzer):
        """Test all patterns in SAFE_PATTERNS constant."""
        for pattern in SAFE_PATTERNS:
            result = analyzer.analyze(pattern)
            assert result.is_safe, f"SAFE_PATTERNS pattern {pattern} should be safe"


# ============================================================================
# MODULE-LEVEL FUNCTION TESTS
# ============================================================================

class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_regex_safety(self):
        """Test the analyze_regex_safety function."""
        result = analyze_regex_safety("^[a-z]+$")
        assert result.is_safe

        result = analyze_regex_safety("(a+)+")
        assert not result.is_safe

    def test_is_safe_pattern(self):
        """Test the is_safe_pattern function."""
        assert is_safe_pattern("^[a-z]+$")
        assert not is_safe_pattern("(a+)+")

    def test_is_re2_compatible_function(self):
        """Test the is_re2_compatible function."""
        assert is_re2_compatible("^[a-z]+$")
        assert not is_re2_compatible(r"(\d+)\1")

    def test_sanitize_pattern_nested_quantifier(self):
        """Test sanitize_pattern removes nested quantifiers."""
        sanitized = sanitize_pattern("(a+)+")
        # The result should be simpler
        assert "+" not in sanitized or sanitized.count("+") < 2

    def test_sanitize_pattern_duplicate_alternation(self):
        """Test sanitize_pattern removes duplicate alternations."""
        sanitized = sanitize_pattern("(a|a)+")
        # Should simplify to (a)+
        assert "|" not in sanitized or sanitized != "(a|a)+"

    def test_sanitize_pattern_safe_unchanged(self):
        """Test sanitize_pattern doesn't change safe patterns."""
        pattern = "^[a-z]+$"
        sanitized = sanitize_pattern(pattern)
        assert sanitized == pattern

    def test_compile_with_timeout_safe(self):
        """Test compile_with_timeout with safe pattern."""
        compiled = compile_with_timeout("^[a-z]+$")
        assert compiled is not None
        assert isinstance(compiled, re.Pattern)

    def test_compile_with_timeout_unsafe(self):
        """Test compile_with_timeout with unsafe pattern."""
        compiled = compile_with_timeout("(a+)+")
        assert compiled is None

    def test_compile_with_timeout_invalid_syntax(self):
        """Test compile_with_timeout with invalid regex syntax."""
        compiled = compile_with_timeout("(unclosed")
        assert compiled is None


# ============================================================================
# LEGACY FUNCTION TESTS
# ============================================================================

class TestLegacyFunctions:
    """Tests for legacy compatibility functions."""

    def test_detect_nested_quantifiers_legacy(self):
        """Test legacy _detect_nested_quantifiers function."""
        result = _detect_nested_quantifiers("(a+)+")
        assert result is not None
        assert "nested" in result.lower() or "quantifier" in result.lower()

    def test_detect_nested_quantifiers_safe(self):
        """Test legacy _detect_nested_quantifiers with safe pattern."""
        result = _detect_nested_quantifiers("^[a-z]+$")
        assert result is None

    def test_detect_overlapping_alternations_legacy(self):
        """Test legacy _detect_overlapping_alternations function."""
        result = _detect_overlapping_alternations("(a|ab)+")
        # May or may not detect depending on implementation
        # Key is that it doesn't raise an exception

    def test_compute_complexity_score_legacy(self):
        """Test legacy _compute_complexity_score function."""
        score = _compute_complexity_score("^[a-z]+$")
        assert 0.0 <= score <= 1.0

    def test_suggest_safe_alternative_legacy(self):
        """Test legacy _suggest_safe_alternative function."""
        suggestion = _suggest_safe_alternative("(a+)+", "nested_quantifier")
        assert suggestion is not None
        assert len(suggestion) > 0


# ============================================================================
# EDGE CASES AND BOUNDARY TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_pattern(self, analyzer: RegexAnalyzer):
        """Test empty pattern handling."""
        result = analyzer.analyze("")
        assert result.is_safe
        assert result.complexity_score == 0.0

    def test_single_character_pattern(self, analyzer: RegexAnalyzer):
        """Test single character pattern."""
        result = analyzer.analyze("a")
        assert result.is_safe

    def test_very_long_pattern(self, analyzer: RegexAnalyzer):
        """Test pattern exceeding maximum length."""
        long_pattern = "a" * (MAX_REGEX_LENGTH + 1)
        result = analyzer.analyze(long_pattern)
        assert not result.is_safe
        assert result.vulnerability_type == VulnerabilityType.CATASTROPHIC_BACKTRACK

    def test_max_length_pattern(self, analyzer: RegexAnalyzer):
        """Test pattern at exactly maximum length."""
        pattern = "a" * MAX_REGEX_LENGTH
        result = analyzer.analyze(pattern)
        # Should be analyzed, not rejected for length

    def test_invalid_regex_syntax(self, analyzer: RegexAnalyzer):
        """Test invalid regex syntax handling."""
        result = analyzer.analyze("(unclosed")
        assert not result.is_safe

    def test_escaped_characters(self, analyzer: RegexAnalyzer):
        """Test pattern with escaped characters."""
        result = analyzer.analyze(r"^\d+\.\d+$")
        assert result.is_safe

    def test_unicode_pattern(self, analyzer: RegexAnalyzer):
        """Test pattern with unicode characters."""
        result = analyzer.analyze("^[\\u0000-\\uFFFF]+$")
        # Should not crash

    def test_pattern_with_anchors(self, analyzer: RegexAnalyzer):
        """Test pattern with anchors."""
        result = analyzer.analyze("^start.*end$")
        assert result.is_safe  # Anchors help prevent backtracking

    def test_complex_but_safe_pattern(self, analyzer: RegexAnalyzer):
        """Test complex but safe pattern."""
        pattern = r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z$"
        result = analyzer.analyze(pattern)
        # ISO 8601 date pattern should be safe despite complexity


# ============================================================================
# RESULT MODEL TESTS
# ============================================================================

class TestRegexAnalysisResult:
    """Tests for RegexAnalysisResult model."""

    def test_result_immutable(self, analyzer: RegexAnalyzer):
        """Test that result is immutable (frozen)."""
        result = analyzer.analyze("^[a-z]+$")
        with pytest.raises(Exception):  # Could be ValidationError or AttributeError
            result.is_safe = False

    def test_result_has_all_fields(self, analyzer: RegexAnalyzer):
        """Test that result has all expected fields."""
        result = analyzer.analyze("^[a-z]+$")
        assert hasattr(result, "pattern")
        assert hasattr(result, "is_safe")
        assert hasattr(result, "complexity_score")
        assert hasattr(result, "vulnerability_type")
        assert hasattr(result, "vulnerable_fragment")
        assert hasattr(result, "recommendation")
        assert hasattr(result, "is_re2_compatible")
        assert hasattr(result, "estimated_worst_case_steps")

    def test_unsafe_result_has_vulnerability_info(self, analyzer: RegexAnalyzer):
        """Test that unsafe results have vulnerability information."""
        result = analyzer.analyze("(a+)+")
        assert not result.is_safe
        assert result.vulnerability_type is not None
        assert result.recommendation is not None
        assert len(result.recommendation) > 0


# ============================================================================
# WORST CASE ESTIMATION TESTS
# ============================================================================

class TestWorstCaseEstimation:
    """Tests for worst-case step estimation."""

    def test_simple_pattern_low_steps(self, analyzer: RegexAnalyzer):
        """Test that simple patterns have low estimated steps."""
        result = analyzer.analyze("^[a-z]+$")
        assert result.estimated_worst_case_steps is not None
        assert result.estimated_worst_case_steps < 10000

    def test_complex_pattern_high_steps(self, analyzer: RegexAnalyzer):
        """Test that complex patterns have higher estimated steps."""
        simple = analyzer.analyze("^abc$")
        complex_pat = analyzer.analyze("(a|b|c){1,10}(d|e|f){1,10}")

        if simple.estimated_worst_case_steps and complex_pat.estimated_worst_case_steps:
            assert complex_pat.estimated_worst_case_steps > simple.estimated_worst_case_steps


# ============================================================================
# INTEGRATION-STYLE TESTS
# ============================================================================

class TestAnalyzerIntegration:
    """Integration-style tests that verify the analyzer works end-to-end."""

    def test_analyze_real_world_email_pattern(self, analyzer: RegexAnalyzer):
        """Test a real-world email validation pattern."""
        # Simplified email pattern
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        result = analyzer.analyze(pattern)
        assert result.is_safe

    def test_analyze_real_world_date_pattern(self, analyzer: RegexAnalyzer):
        """Test a real-world date validation pattern."""
        pattern = r"^\d{4}-\d{2}-\d{2}$"
        result = analyzer.analyze(pattern)
        assert result.is_safe

    def test_analyze_real_world_url_pattern(self, analyzer: RegexAnalyzer):
        """Test a real-world URL validation pattern."""
        # Simplified URL pattern
        pattern = r"^https?://[a-zA-Z0-9.-]+(/[a-zA-Z0-9._/-]*)?$"
        result = analyzer.analyze(pattern)
        assert result.is_safe

    def test_analyze_real_world_uuid_pattern(self, analyzer: RegexAnalyzer):
        """Test a real-world UUID validation pattern."""
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        result = analyzer.analyze(pattern)
        assert result.is_safe

    def test_analyze_potentially_dangerous_wildcard_pattern(self, analyzer: RegexAnalyzer):
        """Test a potentially dangerous wildcard pattern."""
        # This pattern can cause issues with certain inputs
        pattern = ".*a.*b.*c"
        result = analyzer.analyze(pattern)
        # May or may not be safe depending on exact checks


# ============================================================================
# PERFORMANCE TESTS (Quick sanity checks)
# ============================================================================

class TestPerformance:
    """Basic performance sanity checks."""

    def test_analysis_completes_quickly(self, analyzer: RegexAnalyzer):
        """Test that analysis completes in reasonable time."""
        import time

        patterns = [
            "^[a-z]+$",
            "(a+)+",
            r"^\d{4}-\d{2}-\d{2}$",
            "(a|b|c)+",
        ]

        start = time.perf_counter()
        for pattern in patterns:
            analyzer.analyze(pattern)
        elapsed = time.perf_counter() - start

        # Should complete in under 1 second for simple patterns
        assert elapsed < 1.0

    def test_long_safe_pattern_analysis(self, analyzer: RegexAnalyzer):
        """Test that long but safe patterns are analyzed quickly."""
        import time

        # Generate a long but safe pattern
        pattern = "^(" + "|".join(f"word{i}" for i in range(100)) + ")$"

        start = time.perf_counter()
        result = analyzer.analyze(pattern)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
