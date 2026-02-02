# -*- coding: utf-8 -*-
"""
DEPRECATED: GreenLang Registry Package

This module has been consolidated into greenlang.greenlang_registry.
This file now provides backward-compatible re-exports with deprecation warnings.

Please update your imports:
    OLD: from greenlang.registry import OCIClient
    NEW: from greenlang.greenlang_registry.clients import OCIClient
    OR:  from greenlang.greenlang_registry import OCIClient

The OCI client has been moved to greenlang_registry/clients/ to be part of
the full-featured registry service which includes:
- Agent versioning and metadata repository
- Multi-tenant governance
- Lifecycle state management
- API-first design
- OCI registry client functionality

This re-export will be removed in version 2.0.0.
"""

import warnings

# Backward-compatible re-exports
from greenlang.greenlang_registry.clients import (
    OCIManifest,
    OCIDescriptor,
    OCIAuth,
    OCIClient,
)

# Issue deprecation warning on import
warnings.warn(
    "greenlang.registry is deprecated. "
    "Import from greenlang.greenlang_registry.clients instead. "
    "This compatibility layer will be removed in version 2.0.0.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "OCIManifest",
    "OCIDescriptor",
    "OCIAuth",
    "OCIClient",
]
