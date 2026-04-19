# -*- coding: utf-8 -*-
"""
Backward compatibility shim for greenlang.validation.framework.
Use greenlang.governance.validation.framework instead.
"""

# Re-export all classes from the new location
from greenlang.governance.validation.framework import (
    ValidationFramework,
    ValidationResult,
    ValidationError,
    ValidationSeverity,
)

__all__ = [
    "ValidationFramework",
    "ValidationResult",
    "ValidationError",
    "ValidationSeverity",
]
