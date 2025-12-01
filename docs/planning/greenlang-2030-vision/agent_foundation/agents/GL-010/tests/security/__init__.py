# -*- coding: utf-8 -*-
"""
GL-010 CARBONSCOPE Security Tests Package.

Security test suite for the CarbonAccountingEngine covering:
- Input validation (SQL injection, command injection, path traversal)
- Authentication and authorization (RBAC)
- Data protection (secrets, provenance integrity)
- Audit compliance (tampering detection)
- Emission factor validation
- GHG reporting data integrity
- Safety interlocks for emission exceedances

Standards Compliance:
- OWASP Top 10 Application Security Risks
- NIST 800-53 Security Controls
- EPA 40 CFR Part 75 Data Integrity
- GHG Protocol Audit Requirements

Author: GreenLang Foundation Security Team
Version: 1.0.0
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "GreenLang Foundation Security Team"

# Test directory path
SECURITY_TESTS_DIR = Path(__file__).parent
