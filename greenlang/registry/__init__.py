# -*- coding: utf-8 -*-
"""
GreenLang Registry Package

This module provides OCI (Open Container Initiative) registry client functionality
for interacting with container registries.
"""

from .oci_client import (
    OCIManifest,
    OCIDescriptor,
    OCIAuth,
    OCIClient,
)

__all__ = [
    "OCIManifest",
    "OCIDescriptor",
    "OCIAuth",
    "OCIClient",
]
