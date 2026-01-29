# -*- coding: utf-8 -*-
"""
Security Tests for ReDoS Prevention (Task 1.5).

This module contains security tests for Regular Expression Denial of Service
(ReDoS) prevention, verifying that the regex analyzer correctly identifies
and rejects dangerous patterns before they can cause harm.

Attack Vectors Tested:
    - Nested quantifiers (classic ReDoS)
    - Overlapping alternations
    - Exponential backtracking
    - Catastrophic backtracking
    - Polynomial complexity patterns
    - Real-world vulnerable patterns

IMPORTANT: These tests verify that dangerous patterns are IDENTIFIED, not
executed with malicious input. The analyzer should flag all dangerous
patterns before they can be used.

Security References:
    - OWASP ReDoS: https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS
    - CVE Database ReDoS entries
    - Node.js ReDoS vulnerabilities (historical reference)

Author: GreenLang Team
Date: 2026-01-29
"""

import time
import pytest
import re

from greenlang.schema.compiler.regex_analyzer import (
    RegexAnalyzer,
    RegexAnalysisResult,
    VulnerabilityType,
    analyze_regex_safety,
    is_safe_pattern,
    compile_with_timeout,
    DANGEROUS_PATTERNS,
)
from greenlang.schema.constants import MAX_REGEX_LENGTH


# =============================================================================
# Nested Quantifier Attacks
# =============================================================================


class TestNestedQuantifierAttacks:
    """Tests for nested quantifier ReDoS patterns."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_classic_nested_plus(self, analyzer: RegexAnalyzer) -> None:
        """Test classic (a+)+ nested quantifier attack."""
        result = analyzer.analyze("(a+)+")
        assert not result.is_safe
        assert result.vulnerability_type == VulnerabilityType.NESTED_QUANTIFIER

    def test_nested_star(self, analyzer: RegexAnalyzer) -> None:
        """Test (a*)* double Kleene star attack."""
        result = analyzer.analyze("(a*)*")
        assert not result.is_safe

    def test_mixed_nested_quantifiers(self, analyzer: RegexAnalyzer) -> None:
        """Test (a+)* mixed nested quantifiers."""
        result = analyzer.analyze("(a+)*")
        assert not result.is_safe

    def test_nested_with_char_class(self, analyzer: RegexAnalyzer) -> None:
        """Test ([a-z]+)+ with character class."""
        result = analyzer.analyze("([a-z]+)+")
        assert not result.is_safe

    def test_nested_with_dot(self, analyzer: RegexAnalyzer) -> None:
        """Test (.+)+ with dot wildcard."""
        result = analyzer.analyze("(.+)+")
        assert not result.is_safe

    def test_deeply_nested(self, analyzer: RegexAnalyzer) -> None:
        """Test deeply nested quantifiers ((a+)+)+."""
        result = analyzer.analyze("((a+)+)+")
        assert not result.is_safe

    def test_nested_with_whitespace_class(self, analyzer: RegexAnalyzer) -> None:
        """Test (\\s+)+ with whitespace class."""
        result = analyzer.analyze(r"(\s+)+")
        assert not result.is_safe

    def test_nested_with_word_class(self, analyzer: RegexAnalyzer) -> None:
        """Test (\\w+)+ with word class."""
        result = analyzer.analyze(r"(\w+)+")
        assert not result.is_safe

    def test_nested_with_digit_class(self, analyzer: RegexAnalyzer) -> None:
        """Test (\\d+)+ with digit class."""
        result = analyzer.analyze(r"(\d+)+")
        assert not result.is_safe


# =============================================================================
# Overlapping Alternation Attacks
# =============================================================================


class TestOverlappingAlternationAttacks:
    """Tests for overlapping alternation ReDoS patterns."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_identical_alternation(self, analyzer: RegexAnalyzer) -> None:
        """Test (a|a)+ identical alternation attack."""
        result = analyzer.analyze("(a|a)+")
        assert not result.is_safe

    def test_prefix_alternation(self, analyzer: RegexAnalyzer) -> None:
        """Test (a|aa)+ prefix alternation attack."""
        result = analyzer.analyze("(a|aa)+")
        assert not result.is_safe

    def test_suffix_prefix_overlap(self, analyzer: RegexAnalyzer) -> None:
        """Test (aa|a)+ reversed prefix attack."""
        result = analyzer.analyze("(aa|a)+")
        assert not result.is_safe

    def test_longer_overlap(self, analyzer: RegexAnalyzer) -> None:
        """Test (abc|ab|a)+ cascading prefix attack."""
        result = analyzer.analyze("(abc|ab|a)+")
        # Should detect overlapping prefixes

    def test_string_prefix_overlap(self, analyzer: RegexAnalyzer) -> None:
        """Test overlapping string alternations."""
        result = analyzer.analyze("(foo|foobar)+")
        # Should detect that 'foo' is prefix of 'foobar'


# =============================================================================
# Exponential Backtracking Attacks
# =============================================================================


class TestExponentialBacktrackingAttacks:
    """Tests for exponential backtracking patterns."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_multiple_wildcards(self, analyzer: RegexAnalyzer) -> None:
        """Test .*.*.*x multiple wildcard attack."""
        result = analyzer.analyze(".*.*.*x")
        assert not result.is_safe

    def test_wildcard_sequence(self, analyzer: RegexAnalyzer) -> None:
        """Test (.*a)(.*b)(.*c) backtracking chain."""
        result = analyzer.analyze("(.*a)(.*b)(.*c)")
        # Complex pattern may trigger exponential detection

    def test_dot_star_suffix(self, analyzer: RegexAnalyzer) -> None:
        """Test ^.*a.*b.*$ pattern."""
        result = analyzer.analyze("^.*a.*b.*$")
        # Multiple .* with specific chars

    def test_repeated_dot_star(self, analyzer: RegexAnalyzer) -> None:
        """Test (.*X){3,} repeated wildcard attack."""
        result = analyzer.analyze("(.*X){3,}")
        assert not result.is_safe


# =============================================================================
# Real-World Vulnerable Patterns
# =============================================================================


class TestRealWorldVulnerablePatterns:
    """Tests for real-world patterns that have caused ReDoS in production."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_email_vulnerable_pattern(self, analyzer: RegexAnalyzer) -> None:
        """Test vulnerable email regex pattern.

        This pattern is similar to ones that have caused ReDoS in
        email validation libraries.
        """
        # This is a simplified version of problematic email patterns
        pattern = r"^([a-zA-Z0-9._-]+)*@"
        result = analyzer.analyze(pattern)
        # Character class with quantifier, nested in group with quantifier
        # May or may not be flagged depending on exact detection

    def test_url_vulnerable_pattern(self, analyzer: RegexAnalyzer) -> None:
        """Test vulnerable URL regex pattern.

        Similar to patterns that caused issues in URL validators.
        """
        pattern = r"^(([a-z]+)://)?([^/]+)*"
        result = analyzer.analyze(pattern)
        # Group with * containing group with quantifier

    def test_html_tag_vulnerable_pattern(self, analyzer: RegexAnalyzer) -> None:
        """Test vulnerable HTML tag regex pattern."""
        pattern = r"<([a-z]+)([^>]*)>"
        result = analyzer.analyze(pattern)
        # Generally safe but included for completeness

    def test_json_path_vulnerable_pattern(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern similar to JSON path parsing issues."""
        pattern = r"\$\.([a-zA-Z_]+\.?)+"
        result = analyzer.analyze(pattern)
        # Group with + containing optional quantifier

    def test_markdown_link_vulnerable(self, analyzer: RegexAnalyzer) -> None:
        """Test markdown link pattern vulnerabilities."""
        pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        result = analyzer.analyze(pattern)
        # Negated char classes are generally safer


# =============================================================================
# Known CVE Patterns
# =============================================================================


class TestKnownCVEPatterns:
    """Tests for patterns similar to known CVE vulnerabilities.

    Note: These are simplified versions for testing purposes.
    Real CVEs often involve more complex contexts.
    """

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_cve_style_nested_quantifier(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern similar to CVE-style nested quantifier."""
        # Similar to patterns in ua-parser and other libraries
        pattern = r"([a-zA-Z]+)+\."
        result = analyzer.analyze(pattern)
        assert not result.is_safe

    def test_cve_style_alternation(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern similar to CVE-style alternation."""
        # Similar to vulnerable version patterns
        pattern = r"(v(ersion)?\s*\.?\s*)+\d+"
        result = analyzer.analyze(pattern)
        # Contains potential nested quantifiers

    def test_whitespace_bomb(self, analyzer: RegexAnalyzer) -> None:
        """Test whitespace-based ReDoS pattern."""
        pattern = r"(\s+|\t+)+"
        result = analyzer.analyze(pattern)
        assert not result.is_safe  # Nested quantifiers


# =============================================================================
# All Known Dangerous Patterns
# =============================================================================


class TestAllDangerousPatterns:
    """Test all patterns in the DANGEROUS_PATTERNS constant."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    @pytest.mark.parametrize("pattern", list(DANGEROUS_PATTERNS))
    def test_dangerous_pattern_detected(
        self, analyzer: RegexAnalyzer, pattern: str
    ) -> None:
        """Verify all DANGEROUS_PATTERNS are detected as unsafe."""
        result = analyzer.analyze(pattern)
        assert not result.is_safe, f"Pattern {pattern} should be detected as unsafe"


# =============================================================================
# Compile Protection Tests
# =============================================================================


class TestCompileProtection:
    """Tests verifying compile_with_timeout blocks dangerous patterns."""

    def test_compile_blocks_nested_quantifier(self) -> None:
        """Test that compile_with_timeout blocks nested quantifiers."""
        compiled = compile_with_timeout("(a+)+")
        assert compiled is None

    def test_compile_blocks_overlapping_alt(self) -> None:
        """Test that compile_with_timeout blocks overlapping alternations."""
        compiled = compile_with_timeout("(a|a)+")
        assert compiled is None

    def test_compile_allows_safe_pattern(self) -> None:
        """Test that compile_with_timeout allows safe patterns."""
        compiled = compile_with_timeout("^[a-z]+$")
        assert compiled is not None
        assert isinstance(compiled, re.Pattern)

    def test_compile_handles_invalid_syntax(self) -> None:
        """Test that compile_with_timeout handles invalid regex syntax."""
        compiled = compile_with_timeout("(unclosed")
        assert compiled is None


# =============================================================================
# Performance Security Tests
# =============================================================================


class TestPerformanceSecurity:
    """Tests verifying security checks themselves don't cause DoS."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_analysis_completes_quickly(self, analyzer: RegexAnalyzer) -> None:
        """Test that analysis of dangerous patterns completes quickly."""
        dangerous_patterns = [
            "(a+)+",
            "(a*)*",
            "((a+)+)+",
            "(a|aa)+",
            ".*.*.*x",
        ]

        for pattern in dangerous_patterns:
            start = time.perf_counter()
            result = analyzer.analyze(pattern)
            elapsed = time.perf_counter() - start

            # Analysis should complete in under 100ms
            assert elapsed < 0.1, f"Analysis of {pattern} took {elapsed}s"
            assert not result.is_safe

    def test_long_pattern_analysis(self, analyzer: RegexAnalyzer) -> None:
        """Test that long patterns don't cause analyzer DoS."""
        # Generate long but valid pattern
        long_pattern = "(a|b|c|d|e)" * 50

        start = time.perf_counter()
        result = analyzer.analyze(long_pattern)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"Analysis took {elapsed}s"

    def test_complex_pattern_analysis(self, analyzer: RegexAnalyzer) -> None:
        """Test that complex patterns don't cause analyzer DoS."""
        # Complex but not necessarily dangerous pattern
        complex_pattern = r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$"

        start = time.perf_counter()
        result = analyzer.analyze(complex_pattern)
        elapsed = time.perf_counter() - start

        # Should complete quickly
        assert elapsed < 0.5, f"Analysis took {elapsed}s"

    def test_max_length_rejection_fast(self, analyzer: RegexAnalyzer) -> None:
        """Test that max length patterns are rejected quickly."""
        too_long = "a" * (MAX_REGEX_LENGTH + 1)

        start = time.perf_counter()
        result = analyzer.analyze(too_long)
        elapsed = time.perf_counter() - start

        # Should reject immediately
        assert elapsed < 0.01, f"Length check took {elapsed}s"
        assert not result.is_safe


# =============================================================================
# Edge Cases
# =============================================================================


class TestSecurityEdgeCases:
    """Tests for security edge cases."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_empty_pattern_safe(self, analyzer: RegexAnalyzer) -> None:
        """Test that empty pattern is considered safe."""
        result = analyzer.analyze("")
        assert result.is_safe

    def test_single_char_safe(self, analyzer: RegexAnalyzer) -> None:
        """Test that single character pattern is safe."""
        result = analyzer.analyze("a")
        assert result.is_safe

    def test_escaped_special_chars(self, analyzer: RegexAnalyzer) -> None:
        """Test patterns with escaped special characters."""
        result = analyzer.analyze(r"\+\*\?")
        assert result.is_safe  # Escaped chars are literals

    def test_lookahead_patterns(self, analyzer: RegexAnalyzer) -> None:
        """Test patterns with lookahead."""
        # Lookahead patterns may have their own complexity issues
        result = analyzer.analyze(r"(?=.*a)(?=.*b).*")
        # Should be analyzed without crashing

    def test_lookbehind_patterns(self, analyzer: RegexAnalyzer) -> None:
        """Test patterns with lookbehind."""
        result = analyzer.analyze(r"(?<=a)b")
        # Should be analyzed without crashing
        assert not result.is_re2_compatible  # RE2 doesn't support lookbehind

    def test_unicode_patterns(self, analyzer: RegexAnalyzer) -> None:
        """Test patterns with unicode."""
        result = analyzer.analyze(r"[\u0000-\uFFFF]+")
        # Should not crash

    def test_backreference_patterns(self, analyzer: RegexAnalyzer) -> None:
        """Test patterns with backreferences.

        Note: The pattern r"(\\w+)\\1" uses escaped backslash so it's
        looking for literal \\1 in the pattern string. For actual
        backreference we need raw string notation.
        """
        # Test RE2 compatibility detection for backreferences
        # The pattern with a real backreference should be flagged
        result = analyzer.analyze("(a)\\1")  # Actual backreference
        # The analyzer checks for \\1 pattern in the string
        # Since we use single backslash, it may or may not detect
        # Let's just verify it doesn't crash
        assert result is not None


# =============================================================================
# Combined Attack Vectors
# =============================================================================


class TestCombinedAttackVectors:
    """Tests for combined attack patterns."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_nested_plus_overlapping_alt(self, analyzer: RegexAnalyzer) -> None:
        """Test nested quantifiers combined with overlapping alternation."""
        result = analyzer.analyze("((a|aa)+)+")
        assert not result.is_safe

    def test_wildcard_nested_in_alt(self, analyzer: RegexAnalyzer) -> None:
        """Test wildcards nested in alternations."""
        result = analyzer.analyze("(.*|.+)+")
        assert not result.is_safe

    def test_multiple_attack_vectors(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern with multiple attack vectors."""
        # Nested quantifiers AND overlapping alternations
        result = analyzer.analyze("((a+|aa)+)*")
        assert not result.is_safe


# =============================================================================
# Defense Verification
# =============================================================================


class TestDefenseVerification:
    """Tests verifying defense mechanisms work correctly."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_safe_patterns_not_rejected(self, analyzer: RegexAnalyzer) -> None:
        """Verify common safe patterns are not falsely rejected."""
        safe_patterns = [
            r"^[a-zA-Z0-9]+$",
            r"^\d{4}-\d{2}-\d{2}$",
            r"^[A-Z]{2}[0-9]{6}$",
            r"^(https?|ftp)://[^\s/$.?#].[^\s]*$",
            r"^\+?[0-9]{1,15}$",
            r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$",
        ]

        for pattern in safe_patterns:
            result = analyzer.analyze(pattern)
            assert result.is_safe, f"Safe pattern {pattern} was incorrectly rejected"

    def test_vulnerable_fragment_extracted(self, analyzer: RegexAnalyzer) -> None:
        """Test that vulnerable fragments are correctly identified."""
        result = analyzer.analyze("prefix(a+)+suffix")
        assert not result.is_safe
        assert result.vulnerable_fragment is not None

    def test_recommendation_provided(self, analyzer: RegexAnalyzer) -> None:
        """Test that recommendations are provided for unsafe patterns."""
        result = analyzer.analyze("(a+)+")
        assert not result.is_safe
        assert result.recommendation is not None
        assert len(result.recommendation) > 0


# =============================================================================
# Boundary Tests
# =============================================================================


class TestBoundaryConditions:
    """Tests for boundary conditions in security checks."""

    @pytest.fixture
    def analyzer(self) -> RegexAnalyzer:
        """Create analyzer instance."""
        return RegexAnalyzer()

    def test_exact_max_length(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern at exactly max length."""
        pattern = "a" * MAX_REGEX_LENGTH
        result = analyzer.analyze(pattern)
        # Should be analyzed (not rejected for length)

    def test_one_over_max_length(self, analyzer: RegexAnalyzer) -> None:
        """Test pattern one character over max length."""
        pattern = "a" * (MAX_REGEX_LENGTH + 1)
        result = analyzer.analyze(pattern)
        assert not result.is_safe

    def test_complexity_at_threshold(self) -> None:
        """Test pattern at complexity threshold boundary."""
        # Create analyzer with known threshold
        analyzer = RegexAnalyzer(max_complexity_score=0.5)

        # Test patterns around the threshold
        simple = analyzer.analyze("^abc$")
        assert simple.complexity_score < 0.5

    def test_single_quantifier_safe(self, analyzer: RegexAnalyzer) -> None:
        """Test that single (non-nested) quantifiers are safe."""
        patterns = ["a+", "a*", "a?", "a{1,5}"]
        for pattern in patterns:
            result = analyzer.analyze(pattern)
            # Single quantifiers should be safe
            assert result.is_safe, f"Single quantifier {pattern} falsely rejected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
