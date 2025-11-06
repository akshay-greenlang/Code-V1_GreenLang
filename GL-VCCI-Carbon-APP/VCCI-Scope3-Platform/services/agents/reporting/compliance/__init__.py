"""
Compliance and Validation Components
GL-VCCI Scope 3 Platform
"""

from .validator import ComplianceValidator
from .audit_trail import AuditTrailGenerator

__all__ = ["ComplianceValidator", "AuditTrailGenerator"]
