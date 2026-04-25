# -*- coding: utf-8 -*-
"""GreenLang Factors v0.1 alpha release pipeline (Wave E / TaskCreate #23).

This package houses the staging/production publish flow. Records flow:

    seed file --> AlphaProvenanceGate --> namespace='staging'
                                            |
                                            +-- methodology lead review
                                            |
                                            v
                                        namespace='production'

CTO doc reference: §19.1 (FY27 Q1 alpha publish pipeline; manual flip).
"""
from __future__ import annotations

from greenlang.factors.release.alpha_publisher import (
    AlphaPublisher,
    AlphaPublisherError,
    StagingDiff,
)

# alpha_edition_manifest is a sibling deliverable owned by another task.
# Re-export its public surface when available so this package's
# `__all__` is the union of both. Guard the import so the publisher
# remains usable even if the manifest module hasn't landed yet.
try:  # pragma: no cover - exercised only when manifest module exists
    from greenlang.factors.release.alpha_edition_manifest import (
        AlphaEditionManifest,
        FactorManifestEntry,
        SourceManifestEntry,
        build_manifest,
        canonical_json_bytes,
        verify_manifest,
        write_manifest,
    )

    __all__ = [
        "AlphaEditionManifest",
        "AlphaPublisher",
        "AlphaPublisherError",
        "FactorManifestEntry",
        "SourceManifestEntry",
        "StagingDiff",
        "build_manifest",
        "canonical_json_bytes",
        "verify_manifest",
        "write_manifest",
    ]
except ImportError:
    __all__ = [
        "AlphaPublisher",
        "AlphaPublisherError",
        "StagingDiff",
    ]
