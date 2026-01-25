# -*- coding: utf-8 -*-
"""
GreenLang Registry Clients

OCI (Open Container Initiative) client for interacting with container registries.
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
