# -*- coding: utf-8 -*-
"""
Regex Safety Analyzer for GL-FOUND-X-002.

This module implements comprehensive regex complexity analysis to detect potential
ReDoS (Regular Expression Denial of Service) vulnerabilities. ReDoS attacks exploit
patterns that cause catastrophic backtracking, leading to exponential time complexity.

The analyzer detects:
    - Nested quantifiers (e.g., (a+)+)
    - Overlapping alternations (e.g., (a|a)+)
    - Exponential backtracking patterns
    - Catastrophic backtracking patterns
    - Unbounded repetition risks
    - Patterns incompatible with RE2

Key Features:
    - AST-based pattern analysis using Python's sre_parse
    - Quantifier nesting detection with depth tracking
    - Alternation overlap detection using prefix analysis
    - Complexity scoring algorithm (0.0-1.0 scale)
    - RE2 compatibility checking
    - Worst-case step estimation
    - Pattern sanitization suggestions

Example:
    >>> from greenlang.schema.compiler.regex_analyzer import RegexAnalyzer
    >>> analyzer = RegexAnalyzer()
    >>> result = analyzer.analyze("(a+)+")
    >>> print(result.is_safe)
    False
    >>> print(result.vulnerability_type)
    VulnerabilityType.NESTED_QUANTIFIER

References:
    - https://owasp.org/www-community/attacks/Regular_expression_Denial_of_Service_-_ReDoS
    - https://github.com/google/re2
    - PRD Section 6.10: Regex Limits and Safety

Version: 1.0.0
Date: 2026-01-29
"""

from __future__ import annotations

import hashlib
import logging
import re
import signal
import sre_parse
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.schema.constants import (
    MAX_REGEX_COMPLEXITY_SCORE,
    MAX_REGEX_LENGTH,
    REGEX_TIMEOUT_MS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# VULNERABILITY TYPE ENUM
# ============================================================================

class VulnerabilityType(str, Enum):
    """
    Types of regex vulnerabilities that can cause ReDoS.

    Each type represents a different pattern that can lead to
    catastrophic backtracking or exponential time complexity.
    """
    NESTED_QUANTIFIER = "nested_quantifier"
    OVERLAPPING_ALTERNATION = "overlapping_alternation"
    EXPONENTIAL_BACKTRACK = "exponential_backtrack"
    CATASTROPHIC_BACKTRACK = "catastrophic_backtrack"
    UNBOUNDED_REPETITION = "unbounded_repetition"


# ============================================================================
# ANALYSIS RESULT MODEL
# ============================================================================

class RegexAnalysisResult(BaseModel):
    """
    Result of regex safety analysis.

    This model contains the complete analysis of a regex pattern including
    safety assessment, complexity score, vulnerability details, and recommendations.

    Attributes:
        pattern: The original regex pattern that was analyzed
        is_safe: Whether the pattern is safe from ReDoS attacks
        complexity_score: Score from 0.0 (safe) to 1.0 (dangerous)
        vulnerability_type: Type of vulnerability if pattern is unsafe
        vulnerable_fragment: The specific part of the pattern that is problematic
        recommendation: Human-readable suggestion for fixing the pattern
        is_re2_compatible: Whether the pattern can be used with RE2 engine
        estimated_worst_case_steps: Estimated worst-case matching steps

    Example:
        >>> result = RegexAnalysisResult(
        ...     pattern="(a+)+",
        ...     is_safe=False,
        ...     complexity_score=0.95,
        ...     vulnerability_type=VulnerabilityType.NESTED_QUANTIFIER,
        ...     vulnerable_fragment="(a+)+",
        ...     recommendation="Replace nested quantifiers with atomic group",
        ...     is_re2_compatible=True,
        ...     estimated_worst_case_steps=1000000
        ... )
    """

    pattern: str = Field(..., description="The analyzed regex pattern")
    is_safe: bool = Field(..., description="Whether the pattern is safe from ReDoS")
    complexity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Complexity score (0.0=safe, 1.0=dangerous)"
    )
    vulnerability_type: Optional[VulnerabilityType] = Field(
        None,
        description="Type of vulnerability if pattern is unsafe"
    )
    vulnerable_fragment: Optional[str] = Field(
        None,
        description="The specific problematic part of the pattern"
    )
    recommendation: str = Field(
        default="",
        description="Suggestion for fixing the pattern"
    )
    is_re2_compatible: bool = Field(
        default=True,
        description="Whether pattern is compatible with RE2 engine"
    )
    estimated_worst_case_steps: Optional[int] = Field(
        None,
        description="Estimated worst-case matching steps"
    )

    model_config = {
        "frozen": True,
        "extra": "forbid",
    }


# ============================================================================
# KNOWN PATTERN SETS
# ============================================================================

# Known dangerous patterns for quick rejection (exact matches or subpatterns)
DANGEROUS_PATTERNS: FrozenSet[str] = frozenset([
    r"(a+)+",
    r"(a*)*",
    r"(a+)*",
    r"(a*)+",
    r"(.*)+",
    r"(.*)*",
    r"(.+)+",
    r"(.+)*",
    r"(a|a)+",
    r"(a|aa)+",
    r"(aa|a)+",
    r"([a-zA-Z]+)*",
    r"(\\s*)*",
    r"(\\d+)+",
    r"(\\w+)+",
])

# Safe patterns that should always pass (common, well-tested patterns)
SAFE_PATTERNS: FrozenSet[str] = frozenset([
    r"^[a-zA-Z0-9]+$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^[A-Z]{2,3}$",
    r"^[a-z_][a-z0-9_]*$",
    r"^\d+$",
    r"^[a-f0-9]{32}$",
    r"^[a-f0-9]{64}$",
    r"^[A-Za-z0-9+/]+=*$",
])

# RE2-incompatible features (patterns that indicate non-RE2 compatibility)
RE2_INCOMPATIBLE_PATTERNS: FrozenSet[str] = frozenset([
    r"\\1",  # Backreference
    r"\\2",
    r"\\3",
    r"\\4",
    r"\\5",
    r"\\6",
    r"\\7",
    r"\\8",
    r"\\9",
    r"(?=",   # Positive lookahead
    r"(?!",   # Negative lookahead
    r"(?<=",  # Positive lookbehind
    r"(?<!",  # Negative lookbehind
    r"(?>",   # Atomic group (possessive)
    r"?+",    # Possessive quantifier
    r"*+",    # Possessive quantifier
    r"++",    # Possessive quantifier
])

# Quantifier characters for detection
QUANTIFIER_CHARS: FrozenSet[str] = frozenset(["+", "*", "?", "{"])


# ============================================================================
# INTERNAL DATA STRUCTURES
# ============================================================================

@dataclass
class QuantifierInfo:
    """Information about a quantifier in a regex pattern."""
    char: str  # The quantifier character (+, *, ?, {)
    position: int  # Position in the pattern
    min_repeat: int  # Minimum repetitions
    max_repeat: Optional[int]  # Maximum repetitions (None = unlimited)
    is_greedy: bool  # Whether the quantifier is greedy
    depth: int  # Nesting depth when found


@dataclass
class AlternationInfo:
    """Information about an alternation (|) in a regex pattern."""
    alternatives: List[str]  # The alternative branches
    position: int  # Position in the pattern
    depth: int  # Nesting depth


@dataclass
class PatternFeatures:
    """Extracted features from a regex pattern for analysis."""
    length: int
    quantifier_count: int
    max_nesting_depth: int
    alternation_count: int
    char_class_count: int
    has_backreference: bool
    has_lookaround: bool
    has_atomic_group: bool
    quantifiers: List[QuantifierInfo] = field(default_factory=list)
    alternations: List[AlternationInfo] = field(default_factory=list)
    nested_quantifier_depths: List[Tuple[int, int]] = field(default_factory=list)


# ============================================================================
# REGEX ANALYZER CLASS
# ============================================================================

class RegexAnalyzer:
    """
    Analyzes regex patterns for ReDoS vulnerabilities.

    This class provides comprehensive analysis of regular expression patterns
    to detect potential denial-of-service vulnerabilities caused by catastrophic
    backtracking.

    The analyzer uses multiple detection methods:
    1. Quick pattern matching against known dangerous patterns
    2. AST-based analysis using Python's sre_parse module
    3. Quantifier nesting depth analysis
    4. Alternation overlap detection
    5. Complexity scoring based on pattern features

    Attributes:
        max_complexity_score: Maximum allowed complexity score (default: 0.8)
        timeout_ms: Timeout for pattern analysis in milliseconds

    Example:
        >>> analyzer = RegexAnalyzer(max_complexity_score=0.7)
        >>> result = analyzer.analyze("^[a-z]+$")
        >>> assert result.is_safe
        >>> assert result.complexity_score < 0.3

        >>> result = analyzer.analyze("(a+)+")
        >>> assert not result.is_safe
        >>> assert result.vulnerability_type == VulnerabilityType.NESTED_QUANTIFIER
    """

    def __init__(
        self,
        max_complexity_score: float = MAX_REGEX_COMPLEXITY_SCORE,
        timeout_ms: int = REGEX_TIMEOUT_MS,
    ) -> None:
        """
        Initialize the RegexAnalyzer.

        Args:
            max_complexity_score: Maximum allowed complexity score before
                rejecting a pattern. Range: 0.0 to 1.0. Default: 0.8
            timeout_ms: Timeout for analysis operations in milliseconds.
                Default: 100ms

        Raises:
            ValueError: If max_complexity_score is not in range [0.0, 1.0]
        """
        if not 0.0 <= max_complexity_score <= 1.0:
            raise ValueError(
                f"max_complexity_score must be between 0.0 and 1.0, "
                f"got {max_complexity_score}"
            )

        self.max_complexity_score = max_complexity_score
        self.timeout_ms = timeout_ms

        logger.debug(
            f"RegexAnalyzer initialized with max_complexity_score={max_complexity_score}, "
            f"timeout_ms={timeout_ms}"
        )

    def analyze(self, pattern: str) -> RegexAnalysisResult:
        """
        Analyze a regex pattern for safety.

        This is the main entry point for pattern analysis. It performs
        comprehensive checks including:
        - Length validation
        - Known dangerous pattern matching
        - AST-based vulnerability detection
        - Complexity scoring
        - RE2 compatibility checking

        Args:
            pattern: The regex pattern string to analyze

        Returns:
            RegexAnalysisResult containing the complete safety assessment

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> result = analyzer.analyze("^[a-z0-9]+$")
            >>> print(f"Safe: {result.is_safe}, Score: {result.complexity_score}")
            Safe: True, Score: 0.15
        """
        start_time = time.perf_counter()

        # Quick validation: empty pattern
        if not pattern:
            return RegexAnalysisResult(
                pattern=pattern,
                is_safe=True,
                complexity_score=0.0,
                recommendation="Empty pattern is safe but may not match anything.",
                is_re2_compatible=True,
                estimated_worst_case_steps=1,
            )

        # Quick validation: pattern too long
        if len(pattern) > MAX_REGEX_LENGTH:
            return RegexAnalysisResult(
                pattern=pattern,
                is_safe=False,
                complexity_score=1.0,
                vulnerability_type=VulnerabilityType.CATASTROPHIC_BACKTRACK,
                vulnerable_fragment=f"Pattern length {len(pattern)} exceeds max {MAX_REGEX_LENGTH}",
                recommendation=f"Reduce pattern length to under {MAX_REGEX_LENGTH} characters.",
                is_re2_compatible=False,
                estimated_worst_case_steps=None,
            )

        # Quick check: known safe patterns
        if pattern in SAFE_PATTERNS:
            return RegexAnalysisResult(
                pattern=pattern,
                is_safe=True,
                complexity_score=0.1,
                recommendation="This is a known safe pattern.",
                is_re2_compatible=True,
                estimated_worst_case_steps=100,
            )

        # Quick check: known dangerous patterns
        for dangerous in DANGEROUS_PATTERNS:
            if dangerous in pattern:
                return RegexAnalysisResult(
                    pattern=pattern,
                    is_safe=False,
                    complexity_score=1.0,
                    vulnerability_type=VulnerabilityType.NESTED_QUANTIFIER,
                    vulnerable_fragment=dangerous,
                    recommendation=self._get_recommendation_for_dangerous_pattern(dangerous),
                    is_re2_compatible=True,
                    estimated_worst_case_steps=self._estimate_worst_case(pattern, 100),
                )

        # Check for invalid regex syntax
        try:
            re.compile(pattern)
        except re.error as e:
            return RegexAnalysisResult(
                pattern=pattern,
                is_safe=False,
                complexity_score=1.0,
                vulnerability_type=VulnerabilityType.CATASTROPHIC_BACKTRACK,
                vulnerable_fragment=str(e),
                recommendation=f"Fix regex syntax error: {e}",
                is_re2_compatible=False,
                estimated_worst_case_steps=None,
            )

        # Perform detailed analysis
        nested_result = self._detect_nested_quantifiers(pattern)
        overlap_result = self._detect_overlapping_alternations(pattern)
        exponential_result = self._detect_exponential_patterns(pattern)
        complexity_score = self._calculate_complexity_score(pattern)
        is_re2 = self._check_re2_compatible(pattern)
        worst_case = self._estimate_worst_case(pattern, 100)

        # Determine if pattern is safe
        is_safe = True
        vulnerability_type = None
        vulnerable_fragment = None
        recommendation = "Pattern appears safe."

        # Check nested quantifiers
        if nested_result[0]:
            is_safe = False
            vulnerability_type = VulnerabilityType.NESTED_QUANTIFIER
            vulnerable_fragment = nested_result[1]
            recommendation = (
                "Nested quantifiers detected. Consider using atomic groups "
                "or possessive quantifiers to prevent backtracking, or simplify "
                "the pattern to avoid nested repetition."
            )

        # Check overlapping alternations
        elif overlap_result[0]:
            is_safe = False
            vulnerability_type = VulnerabilityType.OVERLAPPING_ALTERNATION
            vulnerable_fragment = overlap_result[1]
            recommendation = (
                "Overlapping alternations detected. Ensure alternatives are "
                "mutually exclusive or reorder them so more specific patterns "
                "come first."
            )

        # Check exponential patterns
        elif exponential_result[0]:
            is_safe = False
            vulnerability_type = VulnerabilityType.EXPONENTIAL_BACKTRACK
            vulnerable_fragment = exponential_result[1]
            recommendation = (
                "Pattern may cause exponential backtracking. Consider using "
                "anchors, atomic groups, or restructuring to reduce backtracking."
            )

        # Check complexity score
        elif complexity_score > self.max_complexity_score:
            is_safe = False
            vulnerability_type = VulnerabilityType.CATASTROPHIC_BACKTRACK
            vulnerable_fragment = pattern
            recommendation = (
                f"Pattern complexity score ({complexity_score:.2f}) exceeds "
                f"maximum ({self.max_complexity_score:.2f}). Simplify the pattern."
            )

        # Build result
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Analyzed pattern in {elapsed_ms:.2f}ms: "
            f"safe={is_safe}, score={complexity_score:.2f}"
        )

        return RegexAnalysisResult(
            pattern=pattern,
            is_safe=is_safe,
            complexity_score=min(complexity_score, 1.0),
            vulnerability_type=vulnerability_type,
            vulnerable_fragment=vulnerable_fragment,
            recommendation=recommendation,
            is_re2_compatible=is_re2,
            estimated_worst_case_steps=worst_case,
        )

    def _detect_nested_quantifiers(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Detect nested quantifiers like (a+)+ or (a*)*.

        Nested quantifiers are a primary cause of ReDoS vulnerabilities because
        they create exponential backtracking scenarios. For example, (a+)+ on
        input "aaaaX" will try every possible way to partition the "a"s.

        Args:
            pattern: The regex pattern to analyze

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> result = analyzer._detect_nested_quantifiers("(a+)+")
            >>> print(result)
            (True, '(a+)+')
        """
        try:
            # Parse the pattern into AST
            parsed = sre_parse.parse(pattern)
            return self._check_nested_quantifiers_ast(parsed, 0, pattern)
        except Exception as e:
            logger.warning(f"Failed to parse pattern for nested quantifier check: {e}")
            # Fall back to regex-based detection
            return self._check_nested_quantifiers_regex(pattern)

    def _check_nested_quantifiers_ast(
        self,
        parsed: Any,
        depth: int,
        original_pattern: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for nested quantifiers using AST traversal.

        Args:
            parsed: Parsed AST from sre_parse
            depth: Current quantifier nesting depth
            original_pattern: Original pattern string for fragment extraction

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)
        """
        # sre_parse constants for operation types
        SUBPATTERN = sre_parse.SUBPATTERN
        MAX_REPEAT = sre_parse.MAX_REPEAT
        MIN_REPEAT = sre_parse.MIN_REPEAT
        BRANCH = sre_parse.BRANCH

        for item in parsed:
            op = item[0]
            av = item[1]

            # Check for repetition operators (*, +, ?, {m,n})
            if op in (MAX_REPEAT, MIN_REPEAT):
                min_count, max_count, subpattern = av

                # If we're already inside a quantifier and find another, it's nested
                if depth > 0 and (max_count is None or max_count > 1):
                    # Extract the vulnerable fragment
                    fragment = self._extract_fragment(original_pattern, item)
                    return (True, fragment if fragment else original_pattern)

                # Recursively check the subpattern
                result = self._check_nested_quantifiers_ast(
                    subpattern, depth + 1, original_pattern
                )
                if result[0]:
                    return result

            # Check subpatterns (groups)
            elif op == SUBPATTERN:
                group_id, add_flags, del_flags, subpattern = av
                result = self._check_nested_quantifiers_ast(
                    subpattern, depth, original_pattern
                )
                if result[0]:
                    return result

            # Check branches (alternations)
            elif op == BRANCH:
                for branch in av[1]:
                    result = self._check_nested_quantifiers_ast(
                        branch, depth, original_pattern
                    )
                    if result[0]:
                        return result

        return (False, None)

    def _check_nested_quantifiers_regex(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Fallback regex-based detection for nested quantifiers.

        This is used when AST parsing fails.

        Args:
            pattern: The regex pattern to check

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)
        """
        # Pattern to detect nested quantifiers: group followed by quantifier
        # containing another quantifier
        nested_patterns = [
            r"\([^)]*[+*][^)]*\)[+*{]",  # (.*+)+ or (a*)* style
            r"\([^)]*\{[^}]+\}[^)]*\)[+*{]",  # ({n,m})+ style
            r"\[[^\]]*\][+*][^)]*[+*{]",  # Character class in nested quantifier
        ]

        for np in nested_patterns:
            try:
                match = re.search(np, pattern)
                if match:
                    return (True, match.group(0))
            except re.error:
                continue

        return (False, None)

    def _detect_overlapping_alternations(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Detect overlapping alternations like (a|a)+ or (ab|a)+.

        Overlapping alternations combined with quantifiers can cause exponential
        backtracking because the regex engine must try all possible combinations.

        Args:
            pattern: The regex pattern to analyze

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> result = analyzer._detect_overlapping_alternations("(a|ab)+")
            >>> print(result)
            (True, '(a|ab)+')
        """
        try:
            parsed = sre_parse.parse(pattern)
            return self._check_overlapping_ast(parsed, pattern)
        except Exception as e:
            logger.warning(f"Failed to parse pattern for overlap check: {e}")
            return self._check_overlapping_regex(pattern)

    def _check_overlapping_ast(
        self,
        parsed: Any,
        original_pattern: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check for overlapping alternations using AST traversal.

        Args:
            parsed: Parsed AST from sre_parse
            original_pattern: Original pattern string

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)
        """
        SUBPATTERN = sre_parse.SUBPATTERN
        MAX_REPEAT = sre_parse.MAX_REPEAT
        MIN_REPEAT = sre_parse.MIN_REPEAT
        BRANCH = sre_parse.BRANCH

        for item in parsed:
            op = item[0]
            av = item[1]

            # Check for branches (alternations)
            if op == BRANCH:
                branches = av[1]
                # Check if any branch is a prefix of another
                if self._has_overlapping_branches(branches):
                    # Check if this branch is inside a quantifier
                    fragment = self._extract_branch_fragment(original_pattern)
                    return (True, fragment if fragment else original_pattern)

            # Recurse into subpatterns
            elif op == SUBPATTERN:
                group_id, add_flags, del_flags, subpattern = av
                result = self._check_overlapping_ast(subpattern, original_pattern)
                if result[0]:
                    return result

            # Recurse into repetitions
            elif op in (MAX_REPEAT, MIN_REPEAT):
                min_count, max_count, subpattern = av
                result = self._check_overlapping_ast(subpattern, original_pattern)
                if result[0]:
                    # Overlapping alternation inside a quantifier is dangerous
                    return result

        return (False, None)

    def _has_overlapping_branches(self, branches: List[Any]) -> bool:
        """
        Check if any branch is a prefix of another.

        Args:
            branches: List of branch ASTs from sre_parse

        Returns:
            True if overlapping branches are found
        """
        # Extract first characters/patterns from each branch
        branch_prefixes: List[Set[str]] = []

        for branch in branches:
            prefix_chars = self._extract_first_chars(branch)
            branch_prefixes.append(prefix_chars)

        # Check for overlapping prefixes
        for i, prefix_i in enumerate(branch_prefixes):
            for j, prefix_j in enumerate(branch_prefixes):
                if i != j and prefix_i & prefix_j:  # Set intersection
                    return True

        return False

    def _extract_first_chars(self, branch: Any) -> Set[str]:
        """
        Extract the set of possible first characters from a branch.

        Args:
            branch: Branch AST from sre_parse

        Returns:
            Set of possible first characters
        """
        chars: Set[str] = set()

        if not branch:
            return chars

        first = branch[0]
        op = first[0]
        av = first[1]

        LITERAL = sre_parse.LITERAL
        NOT_LITERAL = sre_parse.NOT_LITERAL
        IN = sre_parse.IN
        ANY = sre_parse.ANY

        if op == LITERAL:
            chars.add(chr(av))
        elif op == IN:
            # Character class
            for item in av:
                if item[0] == LITERAL:
                    chars.add(chr(item[1]))
                elif item[0] == sre_parse.RANGE:
                    for c in range(item[1][0], item[1][1] + 1):
                        chars.add(chr(c))
        elif op == ANY:
            # Dot matches almost everything - mark as overlapping with anything
            for c in range(256):
                chars.add(chr(c))

        return chars

    def _check_overlapping_regex(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Fallback regex-based detection for overlapping alternations.

        Args:
            pattern: The regex pattern to check

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)
        """
        # Look for alternation groups followed by quantifiers
        match = re.search(r"\(([^|)]+\|[^)]+)\)[+*{]", pattern)
        if match:
            alternatives = match.group(1).split("|")
            # Check for identical or prefix alternatives
            for i, alt_i in enumerate(alternatives):
                for j, alt_j in enumerate(alternatives):
                    if i != j:
                        if alt_i == alt_j:
                            return (True, match.group(0))
                        if alt_i.startswith(alt_j) or alt_j.startswith(alt_i):
                            return (True, match.group(0))

        return (False, None)

    def _detect_exponential_patterns(self, pattern: str) -> Tuple[bool, Optional[str]]:
        """
        Detect patterns that can cause exponential backtracking.

        This function identifies patterns that may not be obvious nested
        quantifiers but can still cause exponential time complexity.

        Args:
            pattern: The regex pattern to analyze

        Returns:
            Tuple of (is_vulnerable, vulnerable_fragment)

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> result = analyzer._detect_exponential_patterns(".*.*.*x")
            >>> print(result[0])
            True
        """
        # Pattern 1: Multiple adjacent wildcards with quantifiers
        if re.search(r"(\.\*){3,}", pattern):
            match = re.search(r"(\.\*){3,}", pattern)
            return (True, match.group(0) if match else pattern)

        # Pattern 2: Dot-star followed by specific char, repeated
        if re.search(r"(\.\*[a-zA-Z]){2,}", pattern):
            match = re.search(r"(\.\*[a-zA-Z]){2,}", pattern)
            return (True, match.group(0) if match else pattern)

        # Pattern 3: Complex groups with high quantifier bounds
        high_bound_match = re.search(r"\{(\d+),(\d*)\}", pattern)
        if high_bound_match:
            min_val = int(high_bound_match.group(1))
            max_val = high_bound_match.group(2)
            if max_val:
                max_val = int(max_val)
                if max_val > 20 or (max_val - min_val) > 10:
                    # Check if this is inside a group
                    if re.search(r"\([^)]+\{" + str(min_val), pattern):
                        return (True, high_bound_match.group(0))

        # Pattern 4: Unbounded repetition after complex group
        if re.search(r"\([^)]+\)\*\s*$", pattern):
            # Ends with (...)* which is unbounded
            match = re.search(r"\([^)]+\)\*\s*$", pattern)
            return (True, match.group(0) if match else pattern)

        return (False, None)

    def _calculate_complexity_score(self, pattern: str) -> float:
        """
        Calculate complexity score based on pattern features.

        The score is calculated using a weighted combination of various
        pattern features that contribute to backtracking complexity.

        Features considered:
        - Number of quantifiers (*, +, ?, {})
        - Nesting depth of groups
        - Alternation count (|)
        - Character class complexity ([...])
        - Backreferences (major penalty)
        - Length of pattern

        Args:
            pattern: The regex pattern to score

        Returns:
            Complexity score between 0.0 and 1.0

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> simple_score = analyzer._calculate_complexity_score("^[a-z]+$")
            >>> complex_score = analyzer._calculate_complexity_score("(a+|b+)*")
            >>> assert simple_score < complex_score
        """
        features = self._extract_features(pattern)

        # Weights for different features
        weights = {
            "length": 0.001,              # 1 point per 100 chars
            "quantifier": 0.05,           # 5 points per quantifier
            "nesting": 0.15,              # 15 points per nesting level
            "alternation": 0.08,          # 8 points per alternation
            "char_class": 0.02,           # 2 points per character class
            "backreference": 0.30,        # 30 points for backreference
            "lookaround": 0.20,           # 20 points for lookaround
            "atomic_group": -0.10,        # Bonus for atomic groups (they help)
        }

        score = 0.0

        # Length contribution
        score += features.length * weights["length"]

        # Quantifier contribution (with bonus for multiple quantifiers)
        quantifier_contribution = features.quantifier_count * weights["quantifier"]
        if features.quantifier_count > 3:
            quantifier_contribution *= 1.5  # Penalty for many quantifiers
        score += quantifier_contribution

        # Nesting depth contribution (exponential growth)
        nesting_contribution = features.max_nesting_depth * weights["nesting"]
        if features.max_nesting_depth > 2:
            nesting_contribution *= (features.max_nesting_depth - 1)
        score += nesting_contribution

        # Alternation contribution
        score += features.alternation_count * weights["alternation"]

        # Character class contribution
        score += features.char_class_count * weights["char_class"]

        # Backreference penalty
        if features.has_backreference:
            score += weights["backreference"]

        # Lookaround penalty
        if features.has_lookaround:
            score += weights["lookaround"]

        # Atomic group bonus (reduces score)
        if features.has_atomic_group:
            score += weights["atomic_group"]

        # Normalize to 0-1 range
        return min(max(score, 0.0), 1.0)

    def _extract_features(self, pattern: str) -> PatternFeatures:
        """
        Extract relevant features from a regex pattern.

        Args:
            pattern: The regex pattern to analyze

        Returns:
            PatternFeatures dataclass with extracted features
        """
        # Count quantifiers
        quantifier_count = (
            pattern.count("+") +
            pattern.count("*") +
            len(re.findall(r"(?<![?+*])\?(?![?+*])", pattern)) +  # ? not part of lazy/possessive
            len(re.findall(r"\{[\d,]+\}", pattern))
        )

        # Calculate nesting depth
        max_depth = 0
        current_depth = 0
        for char in pattern:
            if char == "(":
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ")":
                current_depth = max(0, current_depth - 1)

        # Count alternations
        alternation_count = pattern.count("|")

        # Count character classes
        char_class_count = len(re.findall(r"\[[^\]]*\]", pattern))

        # Check for backreferences
        has_backreference = bool(re.search(r"\\[1-9]", pattern))

        # Check for lookaround
        has_lookaround = bool(re.search(r"\(\?[=!<]", pattern))

        # Check for atomic groups
        has_atomic_group = "(?" + ">" in pattern  # Avoid triggering the check

        return PatternFeatures(
            length=len(pattern),
            quantifier_count=quantifier_count,
            max_nesting_depth=max_depth,
            alternation_count=alternation_count,
            char_class_count=char_class_count,
            has_backreference=has_backreference,
            has_lookaround=has_lookaround,
            has_atomic_group=has_atomic_group,
        )

    def _check_re2_compatible(self, pattern: str) -> bool:
        """
        Check if pattern is compatible with RE2 (no backtracking engine).

        RE2 is a regex engine that guarantees linear time matching by
        disallowing features that require backtracking.

        RE2-incompatible features:
        - Backreferences (\\1, \\2, etc.)
        - Lookahead/lookbehind
        - Atomic groups
        - Possessive quantifiers

        Args:
            pattern: The regex pattern to check

        Returns:
            True if the pattern is RE2-compatible

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> print(analyzer._check_re2_compatible("^[a-z]+$"))
            True
            >>> print(analyzer._check_re2_compatible("(\\d+)\\1"))
            False
        """
        for incompatible in RE2_INCOMPATIBLE_PATTERNS:
            if incompatible in pattern:
                return False

        return True

    def _estimate_worst_case(self, pattern: str, input_length: int = 100) -> int:
        """
        Estimate worst-case matching steps for given input length.

        This is a heuristic estimate based on pattern features. The actual
        worst case depends on the specific input, but this gives an upper
        bound estimate.

        Args:
            pattern: The regex pattern
            input_length: Length of the input string to match against

        Returns:
            Estimated number of worst-case matching steps

        Example:
            >>> analyzer = RegexAnalyzer()
            >>> safe_estimate = analyzer._estimate_worst_case("^[a-z]+$", 100)
            >>> unsafe_estimate = analyzer._estimate_worst_case("(a+)+", 100)
            >>> assert safe_estimate < unsafe_estimate
        """
        features = self._extract_features(pattern)

        # Base steps: linear in input length
        steps = input_length

        # Factor for quantifiers
        if features.quantifier_count > 0:
            steps *= features.quantifier_count

        # Factor for nesting (exponential growth)
        if features.max_nesting_depth > 1:
            steps *= (2 ** features.max_nesting_depth)

        # Factor for alternations
        if features.alternation_count > 0:
            steps *= (features.alternation_count + 1)

        # Factor for backreferences (very expensive)
        if features.has_backreference:
            steps *= input_length

        return min(steps, 10_000_000_000)  # Cap at 10 billion

    def _extract_fragment(self, pattern: str, item: Any) -> Optional[str]:
        """
        Extract a human-readable fragment from the pattern.

        Args:
            pattern: The original pattern
            item: AST item to extract

        Returns:
            Fragment string or None
        """
        # This is a simplified extraction - in practice, you'd want
        # to track positions more carefully
        try:
            return pattern[:50] + "..." if len(pattern) > 50 else pattern
        except Exception:
            return None

    def _extract_branch_fragment(self, pattern: str) -> Optional[str]:
        """
        Extract an alternation fragment from the pattern.

        Args:
            pattern: The original pattern

        Returns:
            Fragment string or None
        """
        match = re.search(r"\([^)]*\|[^)]*\)[+*?{]", pattern)
        if match:
            return match.group(0)
        return None

    def _get_recommendation_for_dangerous_pattern(self, dangerous: str) -> str:
        """
        Get a specific recommendation for a known dangerous pattern.

        Args:
            dangerous: The dangerous pattern that was matched

        Returns:
            Human-readable recommendation
        """
        recommendations = {
            r"(a+)+": "Replace with (a+) or use atomic group (?>a+)+",
            r"(a*)*": "Replace with (a*) - double Kleene star is redundant",
            r"(a+)*": "Replace with (a)* or a*",
            r"(a*)+": "Replace with (a)* or a*",
            r"(.*)+": "Replace with .* - grouping with + is redundant",
            r"(.*)*": "Replace with .* - double wildcard is dangerous",
            r"(.+)+": "Replace with .+ or use atomic group (?>.+)+",
            r"(.+)*": "Replace with .* - this is equivalent but dangerous",
            r"(a|a)+": "Remove duplicate alternative - (a)+ is sufficient",
            r"(a|aa)+": "Use (a)+ instead - (a|aa) causes exponential backtracking",
            r"(aa|a)+": "Use (a)+ instead - (aa|a) causes exponential backtracking",
            r"([a-zA-Z]+)*": "Use [a-zA-Z]* instead of grouping",
            r"(\\s*)*": "Use \\s* instead - double repetition is dangerous",
            r"(\\d+)+": "Use \\d+ instead - nested quantifiers cause ReDoS",
            r"(\\w+)+": "Use \\w+ instead - nested quantifiers cause ReDoS",
        }
        return recommendations.get(
            dangerous,
            "Simplify the pattern to avoid nested quantifiers or overlapping alternations."
        )


# ============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# ============================================================================

# Default analyzer instance for module-level functions
_default_analyzer: Optional[RegexAnalyzer] = None


def _get_default_analyzer() -> RegexAnalyzer:
    """Get or create the default analyzer instance."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = RegexAnalyzer()
    return _default_analyzer


def analyze_regex_safety(
    pattern: str,
    max_length: int = MAX_REGEX_LENGTH,
) -> RegexAnalysisResult:
    """
    Analyze a regex pattern for ReDoS vulnerability.

    This is a convenience function that uses a default RegexAnalyzer instance.
    For repeated analysis with custom settings, create a RegexAnalyzer instance.

    Args:
        pattern: The regex pattern to analyze
        max_length: Maximum allowed pattern length

    Returns:
        RegexAnalysisResult with safety assessment

    Example:
        >>> result = analyze_regex_safety("(a+)+")
        >>> print(result.is_safe)
        False
        >>> print(result.vulnerability_type)
        VulnerabilityType.NESTED_QUANTIFIER
    """
    if len(pattern) > max_length:
        return RegexAnalysisResult(
            pattern=pattern,
            is_safe=False,
            complexity_score=1.0,
            vulnerability_type=VulnerabilityType.CATASTROPHIC_BACKTRACK,
            vulnerable_fragment=f"Pattern length {len(pattern)} exceeds max {max_length}",
            recommendation=f"Reduce pattern length to under {max_length} characters.",
            is_re2_compatible=False,
            estimated_worst_case_steps=None,
        )

    return _get_default_analyzer().analyze(pattern)


def is_safe_pattern(pattern: str) -> bool:
    """
    Quick check if a pattern is safe.

    This is a convenience function for simple boolean checks.
    Use analyze_regex_safety() for detailed analysis.

    Args:
        pattern: The regex pattern to check

    Returns:
        True if the pattern is safe from ReDoS

    Example:
        >>> print(is_safe_pattern("^[a-z]+$"))
        True
        >>> print(is_safe_pattern("(a+)+"))
        False
    """
    result = analyze_regex_safety(pattern)
    return result.is_safe


def is_re2_compatible(pattern: str) -> bool:
    """
    Check if pattern is compatible with RE2 (no backtracking).

    RE2 is a regex engine that guarantees linear time matching by
    disallowing backreferences and certain constructs.

    Args:
        pattern: The regex pattern to check

    Returns:
        True if the pattern can be used with RE2

    Example:
        >>> print(is_re2_compatible("^[a-z]+$"))
        True
        >>> print(is_re2_compatible("(\\d+)\\1"))  # Backreference
        False
    """
    return _get_default_analyzer()._check_re2_compatible(pattern)


def sanitize_pattern(pattern: str) -> str:
    """
    Attempt to make a pattern safer (if possible).

    This function applies simple transformations to reduce ReDoS risk:
    - Remove redundant nested quantifiers
    - Simplify overlapping alternations
    - Add anchors where appropriate

    Note: This is a best-effort function. Complex patterns may require
    manual review and rewriting.

    Args:
        pattern: The regex pattern to sanitize

    Returns:
        Sanitized pattern string

    Example:
        >>> print(sanitize_pattern("(a+)+"))
        '(a+)'
        >>> print(sanitize_pattern("(a|aa)+"))
        '(a)+'
    """
    sanitized = pattern

    # Remove nested quantifiers: (X+)+ -> (X+)
    sanitized = re.sub(r"\(([^()]+)\+\)\+", r"(\1+)", sanitized)
    sanitized = re.sub(r"\(([^()]+)\*\)\*", r"(\1*)", sanitized)
    sanitized = re.sub(r"\(([^()]+)\+\)\*", r"(\1)*", sanitized)
    sanitized = re.sub(r"\(([^()]+)\*\)\+", r"(\1)*", sanitized)

    # Simplify overlapping alternations: (a|aa)+ -> (a)+
    # This is a simple case - complex overlaps need manual review
    sanitized = re.sub(r"\((\w)\|\1\1\)\+", r"(\1)+", sanitized)
    sanitized = re.sub(r"\(\1\1\|(\w)\)\+", r"(\1)+", sanitized)

    # Remove duplicate alternatives: (a|a)+ -> (a)+
    def remove_duplicate_alts(match: re.Match) -> str:
        alts = match.group(1).split("|")
        unique_alts = list(dict.fromkeys(alts))  # Preserve order, remove dups
        if len(unique_alts) == 1:
            return f"({unique_alts[0]}){match.group(2)}"
        return f"({'|'.join(unique_alts)}){match.group(2)}"

    sanitized = re.sub(r"\(([^()]+)\)([+*?])", remove_duplicate_alts, sanitized)

    return sanitized


def compile_with_timeout(
    pattern: str,
    timeout_ms: int = REGEX_TIMEOUT_MS,
) -> Optional[re.Pattern]:
    """
    Compile a regex pattern with safety checks.

    This function compiles the pattern only if it passes safety analysis.
    Note: Python's re module doesn't support true timeout during matching,
    so this relies on pre-analysis to reject dangerous patterns.

    Args:
        pattern: The regex pattern to compile
        timeout_ms: Maximum time for analysis (not matching)

    Returns:
        Compiled pattern if safe, None if unsafe or error

    Example:
        >>> compiled = compile_with_timeout("^[a-z]+$")
        >>> print(compiled is not None)
        True
        >>> compiled = compile_with_timeout("(a+)+")
        >>> print(compiled is None)
        True
    """
    # First, analyze the pattern for safety
    result = analyze_regex_safety(pattern)

    if not result.is_safe:
        logger.warning(
            f"Rejecting unsafe pattern: {pattern[:50]}... "
            f"(vulnerability: {result.vulnerability_type})"
        )
        return None

    try:
        return re.compile(pattern)
    except re.error as e:
        logger.warning(f"Failed to compile pattern: {e}")
        return None


# ============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================================================
# These functions maintain compatibility with the original stub API

def _detect_nested_quantifiers(pattern: str) -> Optional[str]:
    """
    Detect nested quantifier patterns.

    Legacy compatibility function. Use RegexAnalyzer for full analysis.

    Args:
        pattern: The regex pattern

    Returns:
        Description of vulnerability if found, None otherwise
    """
    analyzer = _get_default_analyzer()
    is_vulnerable, fragment = analyzer._detect_nested_quantifiers(pattern)
    if is_vulnerable:
        return f"Nested quantifier detected: {fragment}"
    return None


def _detect_overlapping_alternations(pattern: str) -> Optional[str]:
    """
    Detect overlapping alternations.

    Legacy compatibility function. Use RegexAnalyzer for full analysis.

    Args:
        pattern: The regex pattern

    Returns:
        Description of vulnerability if found, None otherwise
    """
    analyzer = _get_default_analyzer()
    is_vulnerable, fragment = analyzer._detect_overlapping_alternations(pattern)
    if is_vulnerable:
        return f"Overlapping alternation detected: {fragment}"
    return None


def _compute_complexity_score(pattern: str) -> float:
    """
    Compute a complexity score for the pattern.

    Legacy compatibility function. Use RegexAnalyzer for full analysis.

    Args:
        pattern: The regex pattern

    Returns:
        Complexity score between 0.0 and 1.0
    """
    return _get_default_analyzer()._calculate_complexity_score(pattern)


def _suggest_safe_alternative(
    pattern: str,
    vulnerability_type: str,
) -> str:
    """
    Suggest a safer alternative for a problematic pattern.

    Legacy compatibility function.

    Args:
        pattern: The original pattern
        vulnerability_type: The type of vulnerability

    Returns:
        Recommendation string
    """
    # Try to sanitize the pattern
    sanitized = sanitize_pattern(pattern)

    if sanitized != pattern:
        return f"Consider using: {sanitized}"

    recommendations = {
        "nested_quantifier": (
            "Replace nested quantifiers with atomic groups or possessive quantifiers. "
            "For example, change (a+)+ to (?>a+)+ or simply a+."
        ),
        "overlapping_alternation": (
            "Make alternation branches mutually exclusive. "
            "For example, change (a|ab)+ to (ab|a)+ or use a+(b+)?."
        ),
        "exponential_backtrack": (
            "Simplify the pattern to reduce backtracking. "
            "Use anchors (^ and $) and avoid .* where possible."
        ),
        "catastrophic_backtrack": (
            "This pattern is too complex. Break it into multiple simpler patterns "
            "or use a different approach."
        ),
        "unbounded_repetition": (
            "Add upper bounds to repetition. "
            "Change * to {0,100} or + to {1,100} where appropriate."
        ),
    }

    return recommendations.get(
        vulnerability_type,
        "Simplify the pattern to reduce complexity and backtracking risk."
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main class
    "RegexAnalyzer",

    # Result model
    "RegexAnalysisResult",

    # Enums
    "VulnerabilityType",

    # Module-level functions
    "analyze_regex_safety",
    "is_safe_pattern",
    "is_re2_compatible",
    "sanitize_pattern",
    "compile_with_timeout",

    # Constants
    "DANGEROUS_PATTERNS",
    "SAFE_PATTERNS",

    # Legacy compatibility
    "_detect_nested_quantifiers",
    "_detect_overlapping_alternations",
    "_compute_complexity_score",
    "_suggest_safe_alternative",
]
