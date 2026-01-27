# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Errors Module
====================================

GL-FOUND-X-001: Comprehensive error taxonomy and structured error system
for the GreenLang orchestrator.

This module provides:
- Error classification (TRANSIENT, RESOURCE, USER_CONFIG, POLICY_DENIED, AGENT_BUG, INFRASTRUCTURE)
- Structured error codes (GL-E-*)
- Suggested fixes with actionable remediation
- Error registry with mappings
- CLI and JSON formatting

Author: GreenLang Team
Version: 2.0.0
"""

from greenlang.orchestrator.errors.taxonomy import (
    # Enums
    ErrorClass,
    FixType,
    RetryPolicy,
    # Error codes
    ErrorCode,
    # Models
    SuggestedFix,
    OrchestrationError,
    ErrorMetadata,
    # Registry
    ErrorRegistry,
    # Formatters
    format_error_cli,
    format_error_json,
    format_error_markdown,
    # Factory functions
    create_error,
    create_validation_error,
    create_resource_error,
    create_policy_error,
    create_infrastructure_error,
    # Utilities
    get_error_class_for_http_status,
    get_error_class_for_k8s_exit_code,
    serialize_error_chain,
)

__all__ = [
    # Enums
    "ErrorClass",
    "FixType",
    "RetryPolicy",
    # Error codes
    "ErrorCode",
    # Models
    "SuggestedFix",
    "OrchestrationError",
    "ErrorMetadata",
    # Registry
    "ErrorRegistry",
    # Formatters
    "format_error_cli",
    "format_error_json",
    "format_error_markdown",
    # Factory functions
    "create_error",
    "create_validation_error",
    "create_resource_error",
    "create_policy_error",
    "create_infrastructure_error",
    # Utilities
    "get_error_class_for_http_status",
    "get_error_class_for_k8s_exit_code",
    "serialize_error_chain",
]

__version__ = "2.0.0"
