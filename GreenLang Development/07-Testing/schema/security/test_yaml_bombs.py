# -*- coding: utf-8 -*-
"""
Security Tests for YAML Bomb Prevention - GL-FOUND-X-002.

This module contains security tests specifically for YAML bomb prevention,
including various attack vectors that could cause memory exhaustion or
denial of service through YAML parsing.

Attack Vectors Tested:
    - Billion Laughs (exponential entity expansion)
    - Quadratic Blowup (repeated aliases)
    - Deep recursion (anchor chains)
    - Large arrays with aliases
    - Merge key attacks
    - Recursive anchors (self-referencing)
    - Timing attacks (must reject quickly)

IMPORTANT: These tests verify that attacks are BLOCKED, not executed.
The parser should reject all malicious payloads before they can cause harm.

Security References:
    - https://en.wikipedia.org/wiki/Billion_laughs_attack
    - https://cwe.mitre.org/data/definitions/776.html
    - PRD Section 6.10: Security Limits

Author: GreenLang Team
Date: 2026-01-29
Version: 1.0.0
"""

import pytest

from greenlang.schema.compiler.parser import (
    ParseError,
    parse_payload,
    SafeYAMLLoader,
)
from greenlang.schema.errors import ErrorCode


# =============================================================================
# Classic Billion Laughs Attack
# =============================================================================


class TestBillionLaughsAttack:
    """Tests for billion laughs (exponential expansion) attack prevention."""

    def test_basic_billion_laughs(self) -> None:
        """Test classic billion laughs pattern is rejected."""
        yaml_bomb = """
lol1: &lol1 "lol"
lol2: &lol2 [*lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1, *lol1]
lol3: &lol3 [*lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2, *lol2]
lol4: &lol4 [*lol3, *lol3, *lol3, *lol3, *lol3, *lol3, *lol3, *lol3, *lol3, *lol3]
lol5: &lol5 [*lol4, *lol4, *lol4, *lol4, *lol4, *lol4, *lol4, *lol4, *lol4, *lol4]
lol6: &lol6 [*lol5, *lol5, *lol5, *lol5, *lol5, *lol5, *lol5, *lol5, *lol5, *lol5]
lol7: &lol7 [*lol6, *lol6, *lol6, *lol6, *lol6, *lol6, *lol6, *lol6, *lol6, *lol6]
lol8: &lol8 [*lol7, *lol7, *lol7, *lol7, *lol7, *lol7, *lol7, *lol7, *lol7, *lol7]
lol9: [*lol8, *lol8, *lol8, *lol8, *lol8, *lol8, *lol8, *lol8, *lol8, *lol8]
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        # Should fail on anchor detection, not explode
        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_mini_billion_laughs(self) -> None:
        """Test smaller version of billion laughs is still blocked."""
        yaml_bomb = """
a: &a [1, 2, 3]
b: &b [*a, *a]
c: [*b, *b]
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_billion_laughs_with_strings(self) -> None:
        """Test billion laughs using string expansion."""
        yaml_bomb = """
s1: &s1 "AAAAAAAAAA"
s2: &s2 [*s1, *s1, *s1, *s1, *s1, *s1, *s1, *s1, *s1, *s1]
s3: &s3 [*s2, *s2, *s2, *s2, *s2, *s2, *s2, *s2, *s2, *s2]
result: *s3
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Quadratic Blowup Attack
# =============================================================================


class TestQuadraticBlowupAttack:
    """Tests for quadratic blowup attack prevention."""

    def test_repeated_alias_references(self) -> None:
        """Test payload with many alias references to same anchor."""
        # Create payload with many references to one large anchor
        yaml_bomb = """
large_data: &big_data
  - "This is a reasonably large string that will be repeated"
  - "Another line of data that expands the payload"
  - "Yet another line to make it bigger"
refs:
  r1: *big_data
  r2: *big_data
  r3: *big_data
  r4: *big_data
  r5: *big_data
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_linear_alias_chain(self) -> None:
        """Test linear chain of aliases."""
        yaml_bomb = """
a: &a {value: 1}
b: &b {nested: *a}
c: &c {nested: *b}
d: &d {nested: *c}
e: {nested: *d}
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Anchor Placement Attacks
# =============================================================================


class TestAnchorPlacementAttacks:
    """Tests for various anchor placement patterns."""

    def test_scalar_anchor(self) -> None:
        """Test anchor on scalar value is rejected."""
        yaml_bomb = """
key: &anchor_name "value"
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_sequence_anchor(self) -> None:
        """Test anchor on sequence is rejected."""
        yaml_bomb = """
items: &items
  - one
  - two
  - three
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_mapping_anchor(self) -> None:
        """Test anchor on mapping is rejected."""
        yaml_bomb = """
mapping: &mapping
  key1: value1
  key2: value2
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_nested_anchor(self) -> None:
        """Test anchor in nested structure is rejected."""
        yaml_bomb = """
outer:
  inner: &inner
    key: value
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# YAML Merge Key Attacks
# =============================================================================


class TestMergeKeyAttacks:
    """Tests for YAML merge key attack prevention."""

    def test_simple_merge_key(self) -> None:
        """Test simple merge key usage is rejected."""
        yaml_bomb = """
base: &base
  name: base
  value: 1

extended:
  <<: *base
  extra: data
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_multiple_merge_keys(self) -> None:
        """Test multiple merge keys is rejected."""
        yaml_bomb = """
base1: &base1
  a: 1
base2: &base2
  b: 2
combined:
  <<: [*base1, *base2]
  c: 3
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_recursive_merge(self) -> None:
        """Test recursive merge pattern is rejected."""
        yaml_bomb = """
level1: &level1
  a: 1
level2: &level2
  <<: *level1
  b: 2
level3:
  <<: *level2
  c: 3
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Alias Reference Attacks
# =============================================================================


class TestAliasReferenceAttacks:
    """Tests for alias reference patterns."""

    def test_standalone_alias(self) -> None:
        """Test document starting with alias reference."""
        # This would fail anyway since there's no anchor defined first
        yaml_bomb = "*undefined_anchor"

        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        # Should fail parsing
        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_alias_in_key_position(self) -> None:
        """Test alias as mapping key."""
        yaml_bomb = """
anchor: &key "actual_key"
*key: value
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_alias_in_list(self) -> None:
        """Test alias reference in list."""
        yaml_bomb = """
item: &item "data"
list:
  - *item
  - *item
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Edge Cases
# =============================================================================


class TestSecurityEdgeCases:
    """Tests for security edge cases."""

    def test_anchor_name_with_special_chars(self) -> None:
        """Test anchor with special characters."""
        yaml_bomb = """
data: &special-anchor_123
  key: value
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_empty_anchor_reference(self) -> None:
        """Test referencing anchor on empty value."""
        yaml_bomb = """
empty: &empty
ref: *empty
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_null_anchor(self) -> None:
        """Test anchor on null value."""
        yaml_bomb = """
null_val: &null_anchor ~
ref: *null_anchor
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_boolean_anchor(self) -> None:
        """Test anchor on boolean value."""
        yaml_bomb = """
flag: &flag_anchor true
ref: *flag_anchor
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Combined Attack Patterns
# =============================================================================


class TestCombinedAttacks:
    """Tests for combined attack patterns."""

    def test_deep_nesting_with_anchors(self) -> None:
        """Test deep nesting combined with anchors."""
        yaml_bomb = """
level1: &l1
  level2: &l2
    level3: &l3
      data: [*l1, *l2]
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value

    def test_anchor_within_large_structure(self) -> None:
        """Test anchor hidden within large structure."""
        yaml_bomb = """
data:
  section1:
    a: 1
    b: 2
  section2:
    c: 3
    hidden: &hidden
      malicious: data
  section3:
    d: 4
    ref: *hidden
"""
        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_bomb)

        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value


# =============================================================================
# Safe YAML Verification
# =============================================================================


class TestSafeYAMLVerification:
    """Tests verifying safe YAML parsing works correctly."""

    def test_valid_yaml_still_works(self) -> None:
        """Test that valid YAML without anchors still works."""
        valid_yaml = """
name: test
items:
  - one
  - two
  - three
nested:
  key: value
  number: 42
  flag: true
  nothing: null
"""
        result = parse_payload(valid_yaml)
        assert result.format == "yaml"
        assert result.data["name"] == "test"
        assert result.data["items"] == ["one", "two", "three"]
        assert result.data["nested"]["key"] == "value"

    def test_complex_valid_yaml(self) -> None:
        """Test complex valid YAML structure."""
        valid_yaml = """
apiVersion: v1
kind: ConfigMap
metadata:
  name: test-config
  namespace: default
data:
  setting1: value1
  setting2: value2
  config.yaml: |
    nested:
      yaml: content
      array:
        - item1
        - item2
"""
        result = parse_payload(valid_yaml)
        assert result.format == "yaml"
        assert result.data["apiVersion"] == "v1"

    def test_multiline_strings_safe(self) -> None:
        """Test multiline strings without anchors."""
        valid_yaml = """
literal: |
  Line one
  Line two
  Line three
folded: >
  This is a long
  folded string
  that spans lines
"""
        result = parse_payload(valid_yaml)
        assert "Line one" in result.data["literal"]


# =============================================================================
# Performance Bounds
# =============================================================================


class TestSecurityPerformance:
    """Tests verifying security checks don't introduce performance issues."""

    def test_rejection_is_fast(self) -> None:
        """Test that malicious payloads are rejected quickly."""
        import time

        yaml_bomb = """
a: &a ["lol","lol","lol","lol","lol","lol","lol","lol","lol"]
b: &b [*a,*a,*a,*a,*a,*a,*a,*a,*a]
c: &c [*b,*b,*b,*b,*b,*b,*b,*b,*b]
"""
        start = time.perf_counter()

        with pytest.raises(ParseError):
            parse_payload(yaml_bomb)

        elapsed = time.perf_counter() - start

        # Should reject almost immediately (not expand the bomb)
        assert elapsed < 0.5, f"Rejection took too long: {elapsed}s"

    def test_large_safe_yaml_performance(self) -> None:
        """Test that large safe YAML doesn't trigger false positives."""
        # Create large but safe YAML
        items = [f"item_{i}: value_{i}" for i in range(100)]
        large_yaml = "data:\n  " + "\n  ".join(items)

        result = parse_payload(large_yaml)
        assert result.format == "yaml"
        assert "data" in result.data


# =============================================================================
# Recursive Anchor Tests
# =============================================================================


class TestRecursiveAnchors:
    """Tests for recursive anchor detection and rejection."""

    def test_self_referencing_anchor_rejected(self) -> None:
        """Test self-referencing anchor patterns are rejected."""
        yaml_self_ref = """
recursive: &self
  child: *self
"""
        with pytest.raises(ParseError):
            parse_payload(yaml_self_ref)

    def test_mutually_recursive_anchors_rejected(self) -> None:
        """Test mutually recursive A->B->A patterns are rejected."""
        yaml_mutual = """
a: &a
  ref_b: *b
b: &b
  ref_a: *a
"""
        with pytest.raises(ParseError):
            parse_payload(yaml_mutual)

    def test_deeply_nested_alias_chain_rejected(self) -> None:
        """Test deep chain of aliases is rejected."""
        # Build a chain: l0 -> l1 -> l2 -> ... -> l19
        lines = ["l0: &l0 base"]
        for i in range(1, 20):
            lines.append(f"l{i}: &l{i} [*l{i-1}]")
        yaml_chain = "\n".join(lines)

        with pytest.raises(ParseError):
            parse_payload(yaml_chain)


# =============================================================================
# Timing Attack Tests
# =============================================================================


class TestTimingAttacks:
    """Tests ensuring attacks are rejected quickly (no exponential time)."""

    def test_billion_laughs_rejected_quickly(self) -> None:
        """Test billion laughs doesn't cause exponential time."""
        import time

        # Classic 9-level billion laughs
        yaml_bomb = """
l0: &l0 ["lol"]
l1: &l1 [*l0,*l0,*l0,*l0,*l0,*l0,*l0,*l0,*l0,*l0]
l2: &l2 [*l1,*l1,*l1,*l1,*l1,*l1,*l1,*l1,*l1,*l1]
l3: &l3 [*l2,*l2,*l2,*l2,*l2,*l2,*l2,*l2,*l2,*l2]
l4: &l4 [*l3,*l3,*l3,*l3,*l3,*l3,*l3,*l3,*l3,*l3]
l5: [*l4,*l4,*l4,*l4,*l4,*l4,*l4,*l4,*l4,*l4]
"""
        start = time.perf_counter()
        with pytest.raises(ParseError):
            parse_payload(yaml_bomb)
        elapsed = time.perf_counter() - start

        # Must complete in under 1 second (should be milliseconds)
        assert elapsed < 1.0, f"Attack took {elapsed:.2f}s - too slow!"

    def test_repeated_attacks_dont_leak_resources(self) -> None:
        """Test that repeated attack attempts don't cause resource leaks."""
        yaml_attack = "bomb: &b [*b, *b, *b]"

        for _ in range(100):
            with pytest.raises(ParseError):
                parse_payload(yaml_attack)

        # If we get here without crashing or hanging, resources are handled


# =============================================================================
# Error Message Tests
# =============================================================================


class TestErrorMessages:
    """Tests for security-related error message quality."""

    def test_anchor_error_mentions_security(self) -> None:
        """Test that anchor errors explain security reasoning."""
        yaml_with_anchor = "data: &test value"

        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_with_anchor)

        error_msg = str(exc_info.value).lower()
        # Should mention security or billion laughs or disabled
        assert any(term in error_msg for term in ["security", "disabled", "billion"])

    def test_error_includes_anchor_name(self) -> None:
        """Test that errors include the specific anchor name found."""
        yaml_with_anchor = "mydata: &myanchor value"

        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_with_anchor)

        assert "myanchor" in str(exc_info.value)

    def test_error_code_is_correct(self) -> None:
        """Test that correct error code is returned for anchor detection."""
        yaml_with_anchor = "data: &a [1]"

        with pytest.raises(ParseError) as exc_info:
            parse_payload(yaml_with_anchor)

        # Should be the schema parse error code
        assert exc_info.value.code == ErrorCode.SCHEMA_PARSE_ERROR.value
