# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Security Tests Package.

Security validation tests covering OWASP Top 10, input validation,
authentication, authorization, and safety interlocks.

Security Standards:
    - OWASP Top 10
    - IEC 62443 - Industrial Cybersecurity
    - ISA/IEC 62443 - Security for Industrial Automation

Author: GreenLang Industrial Optimization Team
Agent ID: GL-012
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-012"

SECURITY_MARKERS = [
    "security",
    "input_validation",
    "authentication",
    "authorization",
    "audit",
    "safety_interlock",
]
