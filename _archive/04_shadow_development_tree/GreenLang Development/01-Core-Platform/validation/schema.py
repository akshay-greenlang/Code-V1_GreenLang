# -*- coding: utf-8 -*-
"""
Backward compatibility shim for greenlang.validation.schema.
Use greenlang.governance.validation.schema instead.
"""

# Re-export all classes from the new location
from greenlang.governance.validation.schema import (
    SchemaValidator,
    SchemaValidationError,
)

__all__ = [
    "SchemaValidator",
    "SchemaValidationError",
]
