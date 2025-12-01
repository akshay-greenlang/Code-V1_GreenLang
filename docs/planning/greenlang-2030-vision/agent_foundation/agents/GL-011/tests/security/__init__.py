# -*- coding: utf-8 -*-
"""
GL-011 FUELCRAFT Security Tests Package.

Comprehensive security validation tests covering:
- Input validation and sanitization
- Injection attack prevention (SQL, command, path traversal)
- Fuel composition limit validation
- Safety interlock tests for fuel system controls
- Authentication and authorization
- Data protection and audit compliance
- Concurrent access safety

Security Standards:
    - OWASP Top 10 coverage
    - IEC 62443 Industrial Security
    - API 2350 Tank Overfill Protection
    - NFPA 30 Flammable Liquids Code

Author: GL-SecurityAuditor
Agent ID: GL-011
Version: 1.0.0
"""

__version__ = "1.0.0"
__agent_id__ = "GL-011"

# Security test constants
MAX_STRING_LENGTH = 1000
MAX_FUEL_HEATING_VALUE_MJ_KG = 150.0  # Physical maximum (hydrogen ~120)
MIN_FUEL_HEATING_VALUE_MJ_KG = 0.0
MAX_PERCENTAGE = 100.0
MIN_PERCENTAGE = 0.0
MAX_EMISSION_FACTOR = 500.0  # kg CO2e/GJ physical maximum
