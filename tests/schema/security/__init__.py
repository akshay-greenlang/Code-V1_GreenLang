# -*- coding: utf-8 -*-
"""
Security tests for GL-FOUND-X-002 (GreenLang Schema Compiler & Validator).

This package contains comprehensive security-focused tests for:

    - test_yaml_bombs.py: YAML bomb (billion laughs) attack prevention
        - Exponential entity expansion
        - Recursive anchor chains
        - Quadratic blowup attacks
        - Anchor/alias rejection

    - test_redos.py: ReDoS (regex denial of service) prevention
        - Nested quantifier attacks
        - Overlapping alternation attacks
        - Exponential backtracking patterns
        - Real-world CVE-like patterns
        - Complexity scoring validation

    - test_schema_bombs.py: Schema bomb attack prevention
        - Deep $ref chain attacks
        - Circular reference detection
        - Exponential $ref expansion
        - Self-referencing schemas
        - Mutually recursive schemas

    - test_input_limits.py: Input limit enforcement
        - Large payload rejection
        - Deep nesting detection
        - Array item count limits
        - Object property limits
        - Total node count limits
        - Size-before-parsing validation

    - test_path_traversal.py: Path traversal prevention
        - Directory traversal (../) attacks
        - Absolute path injection
        - URL-encoded traversal attempts
        - Null byte injection
        - URI scheme abuse
        - JSON Pointer safety

Security Testing Philosophy:
    These tests verify that malicious inputs are REJECTED safely and quickly,
    not that they cause harmful behavior. The validator must:

    1. Detect attacks early (before resource exhaustion)
    2. Fail fast with clear error messages
    3. Never hang or crash on malicious input
    4. Complete within timeout limits
    5. Return appropriate error codes

Usage:
    Run all security tests:
        pytest tests/schema/security/ -v -m security

    Run with timeout enforcement:
        pytest tests/schema/security/ -v --timeout=30

    Run specific attack category:
        pytest tests/schema/security/test_yaml_bombs.py -v

Error Codes:
    Security-related GLSCHEMA error codes:
    - GLSCHEMA-E507: YAML anchor/alias detected (security disabled)
    - GLSCHEMA-E800: Payload too large
    - GLSCHEMA-E801: Nesting depth exceeded
    - GLSCHEMA-E803: Maximum $ref expansions exceeded
    - GLSCHEMA-E805: Node count exceeded

References:
    - OWASP: https://owasp.org/www-community/attacks/
    - CWE-400: Uncontrolled Resource Consumption
    - CWE-776: Improper Restriction of Recursive Entity References
    - CWE-22: Improper Limitation of a Pathname to a Restricted Directory
    - PRD Section 6.10: Limits and Safety

Author: GreenLang Security Testing Team
Date: 2026-01-29
Version: 1.0.0
"""

from __future__ import annotations

# Package metadata
__all__ = [
    "test_yaml_bombs",
    "test_redos",
    "test_schema_bombs",
    "test_input_limits",
    "test_path_traversal",
]
