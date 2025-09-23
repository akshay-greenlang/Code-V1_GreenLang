"""
Compatibility shim for core.greenlang -> greenlang migration
This module will be deprecated in v0.3.0
"""

import warnings

warnings.warn(
    "Importing from 'core.greenlang' is deprecated. Use 'import greenlang' instead. "
    "This compatibility layer will be removed in v0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical greenlang package
