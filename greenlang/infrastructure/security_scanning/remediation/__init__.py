# -*- coding: utf-8 -*-
"""
Remediation Module - SEC-007

Automated remediation capabilities for security findings including
dependency version bumps, secret rotation triggers, configuration
hardening, and GitHub PR creation.

Exports:
    - AutoFixGenerator: Generate automated fixes for vulnerabilities
    - FixPR: Pull request fix model
    - RotationTask: Secret rotation task model
    - ConfigPatch: Configuration patch model
    - FixStatus: Fix status enumeration

Example:
    >>> from greenlang.infrastructure.security_scanning.remediation import (
    ...     AutoFixGenerator,
    ... )
    >>> generator = AutoFixGenerator(config)
    >>> fix = await generator.generate_dependency_fix(vulnerability)
    >>> pr_url = await generator.create_github_pr(fix)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from greenlang.infrastructure.security_scanning.remediation.auto_fix import (
    AutoFixGenerator,
    AutoFixConfig,
    FixPR,
    RotationTask,
    ConfigPatch,
    FixStatus,
    FixType,
)

__all__ = [
    "AutoFixGenerator",
    "AutoFixConfig",
    "FixPR",
    "RotationTask",
    "ConfigPatch",
    "FixStatus",
    "FixType",
]

__version__ = "1.0.0"
