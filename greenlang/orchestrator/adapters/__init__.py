# -*- coding: utf-8 -*-
"""
Legacy Agent Adapters
======================

Provides GLIP v1 wrappers for legacy HTTP-based agents.

Components:
    - HttpLegacyAdapter: HTTP to GLIP v1 translation
    - AdapterConfig: Configuration for adapters
    - HttpMethod: Supported HTTP methods
    - AuthType: Authentication types
    - AdapterContainerEntrypoint: Container entrypoint for adapters
    - create_adapter_config: Factory function

Author: GreenLang Team
"""

from greenlang.orchestrator.adapters.http_legacy_adapter import (
    HttpLegacyAdapter,
    AdapterConfig,
    HttpMethod,
    AuthType,
    AdapterContainerEntrypoint,
    create_adapter_config,
)

__all__ = [
    "HttpLegacyAdapter",
    "AdapterConfig",
    "HttpMethod",
    "AuthType",
    "AdapterContainerEntrypoint",
    "create_adapter_config",
]
