# -*- coding: utf-8 -*-
"""
Security Scanners Package - SEC-007

Provides scanner implementations for various security scanning tools.
Each scanner implements the BaseScanner interface for unified orchestration.

Scanner Categories:
    - SAST: Static Application Security Testing (Bandit, Semgrep, CodeQL)
    - SCA: Software Composition Analysis (Trivy, Snyk, pip-audit, Safety)
    - Secrets: Secret Detection (Gitleaks, TruffleHog, detect-secrets)
    - Container: Container Image Scanning (Trivy, Grype, Cosign)
    - IaC: Infrastructure as Code (TFSec, Checkov, Kubeconform)
    - DAST: Dynamic Application Security Testing (OWASP ZAP)

Example:
    >>> from greenlang.infrastructure.security_scanning.scanners import (
    ...     BanditScanner,
    ...     TrivyScanner,
    ...     GitleaksScanner,
    ... )
    >>> scanner = BanditScanner(config)
    >>> result = await scanner.scan("/path/to/code")

Author: GreenLang Security Team
Date: February 2026
"""

from greenlang.infrastructure.security_scanning.scanners.base import (
    BaseScanner,
    ScannerExecutionError,
    ScannerTimeoutError,
    ScannerNotFoundError,
)

__all__ = [
    # Base
    "BaseScanner",
    "ScannerExecutionError",
    "ScannerTimeoutError",
    "ScannerNotFoundError",
]

# Lazy imports for specific scanners to avoid import overhead
# Use explicit imports when needed:
# from greenlang.infrastructure.security_scanning.scanners.sast import BanditScanner
