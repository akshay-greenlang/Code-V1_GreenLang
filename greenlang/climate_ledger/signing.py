# -*- coding: utf-8 -*-
"""
Climate Ledger - Artifact Signing
===================================

Re-exports artifact signing and verification from
``greenlang.utilities.provenance.signing`` for the v3 Climate Ledger
product surface.

Exported symbols:

- ``sign_artifact`` -- sign a file artifact with the secure provider
- ``verify_artifact`` -- verify an artifact's cryptographic signature

Example::

    >>> from greenlang.climate_ledger.signing import sign_artifact, verify_artifact
    >>> sig = sign_artifact(Path("report.pdf"))
    >>> ok, info = verify_artifact(Path("report.pdf"))

Author: GreenLang Platform Team
Date: April 2026
Status: Production Ready
"""

from __future__ import annotations

# Re-exports from the canonical signing module -- no new logic needed.
from greenlang.utilities.provenance.signing import (
    sign_artifact,
    verify_artifact,
)

__all__ = [
    "sign_artifact",
    "verify_artifact",
]
