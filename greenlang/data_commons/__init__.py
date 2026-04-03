# -*- coding: utf-8 -*-
"""
GreenLang Data Commons - Shared utilities for data layer agents.

This package provides base classes and helpers that eliminate duplicate
boilerplate across the 20 data-layer agent modules.

Modules:
    config_base:  BaseDataConfig dataclass and thread-safe singleton factory.
    metrics:      MetricsFactory for Prometheus metric registration.
    provenance:   ProvenanceTracker with SHA-256 chain hashing.
    router_base:  Standard FastAPI router factory and service dependencies.
    hash_utils:   Deterministic hashing utilities (compute_hash, etc.).

Example:
    >>> from greenlang.data_commons.config_base import BaseDataConfig
    >>> from greenlang.data_commons.provenance import ProvenanceTracker
    >>> from greenlang.data_commons.metrics import MetricsFactory
    >>> from greenlang.data_commons.router_base import create_standard_router
    >>> from greenlang.data_commons.hash_utils import compute_hash
"""

# -- config_base -------------------------------------------------------------
from greenlang.data_commons.config_base import (
    BaseDataConfig,
    EnvReader,
    create_config_singleton,
)

# -- metrics -----------------------------------------------------------------
from greenlang.data_commons.metrics import (
    CONFIDENCE_BUCKETS,
    DURATION_BUCKETS,
    LONG_DURATION_BUCKETS,
    PROMETHEUS_AVAILABLE,
    MetricsFactory,
)

# -- provenance --------------------------------------------------------------
from greenlang.data_commons.provenance import (
    ProvenanceTracker,
)

# -- router_base -------------------------------------------------------------
from greenlang.data_commons.router_base import (
    FASTAPI_AVAILABLE,
    create_standard_router,
    error_response,
    get_service_dependency,
)

# -- hash_utils --------------------------------------------------------------
from greenlang.data_commons.hash_utils import (
    build_hash,
    compute_hash,
    deterministic_id,
    file_hash,
)

__all__ = [
    # config_base
    "BaseDataConfig",
    "EnvReader",
    "create_config_singleton",
    # metrics
    "PROMETHEUS_AVAILABLE",
    "DURATION_BUCKETS",
    "CONFIDENCE_BUCKETS",
    "LONG_DURATION_BUCKETS",
    "MetricsFactory",
    # provenance
    "ProvenanceTracker",
    # router_base
    "FASTAPI_AVAILABLE",
    "create_standard_router",
    "get_service_dependency",
    "error_response",
    # hash_utils
    "compute_hash",
    "deterministic_id",
    "file_hash",
    "build_hash",
]
