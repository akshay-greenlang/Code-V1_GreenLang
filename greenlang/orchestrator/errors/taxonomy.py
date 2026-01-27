# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Error Taxonomy - GL-FOUND-X-001

Comprehensive error classification system with structured error codes,
suggested fixes, and formatting for CLI/JSON output.

This module provides:
- ErrorClass enum for error classification
- ErrorCode constants (GL-E-*)
- OrchestrationError structured model with suggested fixes
- Error registry mapping codes to fixes
- CLI and JSON formatting utilities
- K8s exit code and HTTP status code mappings

Features:
- TRANSIENT: Network timeouts, 5xx, rate limits - retry allowed
- RESOURCE: OOM, node eviction, quota exceeded - bounded retry
- USER_CONFIG: Invalid params, missing dataset - no retry
- POLICY_DENIED: Permission/policy failures - no retry
- AGENT_BUG: Deterministic crash, 4xx - no retry
- INFRASTRUCTURE: K8s, S3 failures - depends on specific error

Author: GreenLang Team
Date: 2026-01-27
Version: 2.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class ErrorClass(str, Enum):
    """
    Classification of orchestration errors.

    Each class has different retry semantics and user remediation paths.
    """

    TRANSIENT = "TRANSIENT"
    """Network timeouts, 5xx errors, rate limits. Retry: YES with backoff."""

    RESOURCE = "RESOURCE"
    """OOM, node eviction, quota exceeded. Retry: BOUNDED (with resource increase)."""

    USER_CONFIG = "USER_CONFIG"
    """Invalid parameters, missing dataset, schema errors. Retry: NO (fix config)."""

    POLICY_DENIED = "POLICY_DENIED"
    """Permission denied, OPA policy failures. Retry: NO (request access)."""

    AGENT_BUG = "AGENT_BUG"
    """Deterministic crash, 4xx client errors. Retry: NO (code fix required)."""

    INFRASTRUCTURE = "INFRASTRUCTURE"
    """K8s, S3, database failures. Retry: DEPENDS on specific failure."""


class FixType(str, Enum):
    """Type of suggested fix for an error."""

    RESOURCE_CHANGE = "RESOURCE_CHANGE"
    """Increase memory, CPU, timeout, or other resource limits."""

    PARAM_CHANGE = "PARAM_CHANGE"
    """Modify pipeline parameters or configuration values."""

    POLICY_REQUEST = "POLICY_REQUEST"
    """Request policy exception or permission grant."""

    CODE_FIX = "CODE_FIX"
    """Fix agent code or pipeline definition."""

    RETRY = "RETRY"
    """Simply retry the operation (for transient errors)."""

    DATA_FIX = "DATA_FIX"
    """Fix input data or dataset configuration."""

    INFRASTRUCTURE_FIX = "INFRASTRUCTURE_FIX"
    """Infrastructure team intervention required."""

    CONTACT_SUPPORT = "CONTACT_SUPPORT"
    """Contact support team for assistance."""


class RetryPolicy(str, Enum):
    """Retry policy for an error class."""

    NO_RETRY = "NO_RETRY"
    """Do not retry - error requires user intervention."""

    IMMEDIATE_RETRY = "IMMEDIATE_RETRY"
    """Retry immediately without backoff."""

    EXPONENTIAL_BACKOFF = "EXPONENTIAL_BACKOFF"
    """Retry with exponential backoff."""

    BOUNDED_RETRY = "BOUNDED_RETRY"
    """Retry up to N times with resource adjustment."""

    CONDITIONAL_RETRY = "CONDITIONAL_RETRY"
    """Retry only if specific conditions are met."""


# =============================================================================
# ERROR CODES
# =============================================================================


class ErrorCode:
    """
    GreenLang orchestrator error codes (GL-E-*).

    All error codes follow the pattern: GL-E-{CATEGORY}-{SPECIFIC}
    """

    # -------------------------------------------------------------------------
    # YAML/Configuration Errors (USER_CONFIG)
    # -------------------------------------------------------------------------
    GL_E_YAML_INVALID = "GL-E-YAML-INVALID"
    """YAML syntax error or invalid structure."""

    GL_E_YAML_SCHEMA = "GL-E-YAML-SCHEMA"
    """YAML does not match expected schema."""

    GL_E_YAML_VERSION = "GL-E-YAML-VERSION"
    """Unsupported pipeline YAML version."""

    # -------------------------------------------------------------------------
    # DAG Errors (USER_CONFIG)
    # -------------------------------------------------------------------------
    GL_E_DAG_CYCLE = "GL-E-DAG-CYCLE"
    """Cycle detected in pipeline DAG."""

    GL_E_DAG_ORPHAN = "GL-E-DAG-ORPHAN"
    """Orphaned node in DAG (no path to sink)."""

    GL_E_DAG_MISSING_DEP = "GL-E-DAG-MISSING-DEP"
    """Missing dependency reference in DAG."""

    GL_E_DAG_DUPLICATE_ID = "GL-E-DAG-DUPLICATE-ID"
    """Duplicate step ID in DAG."""

    # -------------------------------------------------------------------------
    # Agent Errors (USER_CONFIG / AGENT_BUG)
    # -------------------------------------------------------------------------
    GL_E_AGENT_NOT_FOUND = "GL-E-AGENT-NOT-FOUND"
    """Agent not found in registry."""

    GL_E_AGENT_VERSION = "GL-E-AGENT-VERSION"
    """Specified agent version not available."""

    GL_E_AGENT_DEPRECATED = "GL-E-AGENT-DEPRECATED"
    """Agent is deprecated and cannot be used."""

    GL_E_AGENT_CRASH = "GL-E-AGENT-CRASH"
    """Agent crashed during execution."""

    GL_E_AGENT_TIMEOUT = "GL-E-AGENT-TIMEOUT"
    """Agent execution timed out."""

    GL_E_AGENT_OUTPUT_INVALID = "GL-E-AGENT-OUTPUT-INVALID"
    """Agent output does not match expected schema."""

    # -------------------------------------------------------------------------
    # Parameter Errors (USER_CONFIG)
    # -------------------------------------------------------------------------
    GL_E_PARAM_MISSING = "GL-E-PARAM-MISSING"
    """Required parameter is missing."""

    GL_E_PARAM_TYPE = "GL-E-PARAM-TYPE"
    """Parameter has incorrect type."""

    GL_E_PARAM_RANGE = "GL-E-PARAM-RANGE"
    """Parameter value out of allowed range."""

    GL_E_PARAM_ENUM = "GL-E-PARAM-ENUM"
    """Parameter value not in allowed enum values."""

    GL_E_PARAM_FORMAT = "GL-E-PARAM-FORMAT"
    """Parameter value has incorrect format."""

    # -------------------------------------------------------------------------
    # Kubernetes Errors (RESOURCE / INFRASTRUCTURE)
    # -------------------------------------------------------------------------
    GL_E_K8S_JOB_OOM = "GL-E-K8S-JOB-OOM"
    """Kubernetes job killed due to Out-Of-Memory."""

    GL_E_K8S_JOB_TIMEOUT = "GL-E-K8S-JOB-TIMEOUT"
    """Kubernetes job exceeded active deadline."""

    GL_E_K8S_IMAGEPULL = "GL-E-K8S-IMAGEPULL"
    """Failed to pull container image."""

    GL_E_K8S_EVICTION = "GL-E-K8S-EVICTION"
    """Pod evicted due to node pressure."""

    GL_E_K8S_QUOTA = "GL-E-K8S-QUOTA"
    """Namespace quota exceeded."""

    GL_E_K8S_SCHEDULING = "GL-E-K8S-SCHEDULING"
    """Failed to schedule pod (insufficient resources)."""

    GL_E_K8S_NODE_FAILURE = "GL-E-K8S-NODE-FAILURE"
    """Node failure during execution."""

    GL_E_K8S_NETWORK = "GL-E-K8S-NETWORK"
    """Kubernetes network error."""

    GL_E_K8S_API = "GL-E-K8S-API"
    """Kubernetes API server error."""

    # -------------------------------------------------------------------------
    # S3/Storage Errors (INFRASTRUCTURE / POLICY_DENIED)
    # -------------------------------------------------------------------------
    GL_E_S3_ACCESS_DENIED = "GL-E-S3-ACCESS-DENIED"
    """S3 access denied (permissions)."""

    GL_E_S3_NOT_FOUND = "GL-E-S3-NOT-FOUND"
    """S3 object or bucket not found."""

    GL_E_S3_TIMEOUT = "GL-E-S3-TIMEOUT"
    """S3 operation timed out."""

    GL_E_S3_QUOTA = "GL-E-S3-QUOTA"
    """S3 storage quota exceeded."""

    GL_E_S3_INTEGRITY = "GL-E-S3-INTEGRITY"
    """S3 data integrity check failed."""

    GL_E_STORAGE_CORRUPTED = "GL-E-STORAGE-CORRUPTED"
    """Artifact storage data corrupted."""

    # -------------------------------------------------------------------------
    # Policy Errors (POLICY_DENIED)
    # -------------------------------------------------------------------------
    GL_E_OPA_DENY = "GL-E-OPA-DENY"
    """OPA policy denied the operation."""

    GL_E_YAML_POLICY = "GL-E-YAML-POLICY"
    """YAML policy rule violated."""

    GL_E_RBAC_DENIED = "GL-E-RBAC-DENIED"
    """RBAC permission denied."""

    GL_E_TENANT_ISOLATION = "GL-E-TENANT-ISOLATION"
    """Tenant isolation policy violated."""

    GL_E_DATA_CLASSIFICATION = "GL-E-DATA-CLASSIFICATION"
    """Data classification policy violated."""

    # -------------------------------------------------------------------------
    # Quota Errors (RESOURCE)
    # -------------------------------------------------------------------------
    GL_E_QUOTA_EXCEEDED = "GL-E-QUOTA-EXCEEDED"
    """General quota exceeded."""

    GL_E_QUOTA_CPU = "GL-E-QUOTA-CPU"
    """CPU quota exceeded."""

    GL_E_QUOTA_MEMORY = "GL-E-QUOTA-MEMORY"
    """Memory quota exceeded."""

    GL_E_QUOTA_GPU = "GL-E-QUOTA-GPU"
    """GPU quota exceeded."""

    GL_E_QUOTA_RUNS = "GL-E-QUOTA-RUNS"
    """Concurrent run quota exceeded."""

    GL_E_QUOTA_STORAGE = "GL-E-QUOTA-STORAGE"
    """Storage quota exceeded."""

    # -------------------------------------------------------------------------
    # Network/API Errors (TRANSIENT)
    # -------------------------------------------------------------------------
    GL_E_NETWORK_TIMEOUT = "GL-E-NETWORK-TIMEOUT"
    """Network connection timed out."""

    GL_E_NETWORK_DNS = "GL-E-NETWORK-DNS"
    """DNS resolution failed."""

    GL_E_NETWORK_REFUSED = "GL-E-NETWORK-REFUSED"
    """Connection refused."""

    GL_E_API_RATE_LIMIT = "GL-E-API-RATE-LIMIT"
    """API rate limit exceeded."""

    GL_E_API_5XX = "GL-E-API-5XX"
    """API server error (5xx)."""

    GL_E_API_UNAVAILABLE = "GL-E-API-UNAVAILABLE"
    """API service unavailable."""

    # -------------------------------------------------------------------------
    # Database Errors (INFRASTRUCTURE)
    # -------------------------------------------------------------------------
    GL_E_DB_CONNECTION = "GL-E-DB-CONNECTION"
    """Database connection failed."""

    GL_E_DB_TIMEOUT = "GL-E-DB-TIMEOUT"
    """Database query timed out."""

    GL_E_DB_DEADLOCK = "GL-E-DB-DEADLOCK"
    """Database deadlock detected."""

    GL_E_DB_INTEGRITY = "GL-E-DB-INTEGRITY"
    """Database integrity constraint violated."""

    # -------------------------------------------------------------------------
    # Provenance/Audit Errors (AGENT_BUG)
    # -------------------------------------------------------------------------
    GL_E_PROVENANCE_MISSING = "GL-E-PROVENANCE-MISSING"
    """Required provenance data missing."""

    GL_E_PROVENANCE_MISMATCH = "GL-E-PROVENANCE-MISMATCH"
    """Provenance hash mismatch detected."""

    GL_E_AUDIT_CHAIN_BROKEN = "GL-E-AUDIT-CHAIN-BROKEN"
    """Hash chain integrity compromised."""

    # -------------------------------------------------------------------------
    # Data Errors (USER_CONFIG / DATA)
    # -------------------------------------------------------------------------
    GL_E_DATA_NOT_FOUND = "GL-E-DATA-NOT-FOUND"
    """Input dataset not found."""

    GL_E_DATA_SCHEMA = "GL-E-DATA-SCHEMA"
    """Input data schema mismatch."""

    GL_E_DATA_EMPTY = "GL-E-DATA-EMPTY"
    """Input dataset is empty."""

    GL_E_DATA_CORRUPTED = "GL-E-DATA-CORRUPTED"
    """Input data is corrupted."""

    GL_E_DATA_SIZE = "GL-E-DATA-SIZE"
    """Input data exceeds size limit."""

    # -------------------------------------------------------------------------
    # Internal Errors (AGENT_BUG / INFRASTRUCTURE)
    # -------------------------------------------------------------------------
    GL_E_INTERNAL = "GL-E-INTERNAL"
    """Internal orchestrator error."""

    GL_E_ASSERTION = "GL-E-ASSERTION"
    """Internal assertion failed."""

    GL_E_UNEXPECTED = "GL-E-UNEXPECTED"
    """Unexpected error occurred."""


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SuggestedFix(BaseModel):
    """
    A suggested fix for an orchestration error.

    Provides actionable remediation guidance for users.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "example": {
                "fix_type": "RESOURCE_CHANGE",
                "field": "resources.memory",
                "recommended_value": "4Gi",
                "description": "Increase memory limit to handle larger datasets",
            }
        },
    )

    fix_type: FixType = Field(
        ...,
        description="Type of fix (RESOURCE_CHANGE, PARAM_CHANGE, POLICY_REQUEST, etc.)",
    )

    field: Optional[str] = Field(
        None,
        description="Pipeline field to modify (e.g., 'resources.memory', 'params.batch_size')",
    )

    recommended_value: Optional[Any] = Field(
        None,
        description="Recommended value for the field",
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Human-readable description of the fix",
    )

    cli_command: Optional[str] = Field(
        None,
        description="CLI command to apply the fix (if applicable)",
    )

    doc_link: Optional[str] = Field(
        None,
        description="Link to documentation for this fix",
    )


class ErrorMetadata(BaseModel):
    """
    Metadata about an error code.

    Used by the ErrorRegistry to store information about each error code.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = Field(..., description="Error code (GL-E-*)")
    error_class: ErrorClass = Field(..., description="Error classification")
    retry_policy: RetryPolicy = Field(..., description="Retry policy for this error")
    default_message: str = Field(..., description="Default error message template")
    default_fixes: List[SuggestedFix] = Field(
        default_factory=list, description="Default suggested fixes"
    )
    severity: str = Field(
        default="ERROR", description="Severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    category: str = Field(default="general", description="Error category for grouping")


class OrchestrationError(BaseModel):
    """
    Structured orchestration error with comprehensive context.

    This model provides all information needed for user-friendly error
    presentation and automated remediation.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "code": "GL-E-K8S-JOB-OOM",
                "error_class": "RESOURCE",
                "message": "Agent 'emissions_calculator' was killed due to OOM (exit code 137)",
                "details": {
                    "step_id": "calculate_scope1",
                    "agent_name": "emissions_calculator",
                    "exit_code": 137,
                    "memory_limit": "2Gi",
                    "memory_peak": "2.1Gi",
                },
                "suggested_fixes": [
                    {
                        "fix_type": "RESOURCE_CHANGE",
                        "field": "steps.calculate_scope1.resources.memory",
                        "recommended_value": "4Gi",
                        "description": "Double the memory limit for this step",
                    }
                ],
                "links": {
                    "run_viewer": "https://greenlang.io/runs/abc123",
                    "logs": "https://greenlang.io/runs/abc123/logs",
                    "trace": "https://greenlang.io/runs/abc123/trace",
                },
            }
        },
    )

    code: str = Field(
        ...,
        pattern=r"^GL-E-[A-Z0-9-]+$",
        description="Error code (GL-E-*)",
    )

    error_class: ErrorClass = Field(
        ...,
        description="Error classification (TRANSIENT, RESOURCE, USER_CONFIG, etc.)",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Human-readable error message",
    )

    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Error-specific details (step_id, agent_name, etc.)",
    )

    suggested_fixes: List[SuggestedFix] = Field(
        default_factory=list,
        description="List of suggested fixes for this error",
    )

    links: Dict[str, str] = Field(
        default_factory=dict,
        description="Related links (run_viewer, logs, trace, docs)",
    )

    retry_policy: RetryPolicy = Field(
        default=RetryPolicy.NO_RETRY,
        description="Retry policy for this error",
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred (UTC)",
    )

    run_id: Optional[str] = Field(
        None,
        description="Pipeline run ID",
    )

    step_id: Optional[str] = Field(
        None,
        description="Step ID where the error occurred",
    )

    agent_name: Optional[str] = Field(
        None,
        description="Agent name that produced the error",
    )

    trace_id: Optional[str] = Field(
        None,
        description="Distributed trace ID for correlation",
    )

    correlation_id: Optional[str] = Field(
        None,
        description="Correlation ID for grouping related errors",
    )

    cause: Optional[str] = Field(
        None,
        description="Root cause description (if known)",
    )

    stack_trace: Optional[str] = Field(
        None,
        description="Stack trace for debugging (may be redacted in production)",
    )

    @field_validator("code")
    @classmethod
    def validate_error_code(cls, v: str) -> str:
        """Validate error code format."""
        if not v.startswith("GL-E-"):
            raise ValueError(f"Error code must start with 'GL-E-': {v}")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return self.model_dump_json(indent=indent, exclude_none=True)

    def get_error_hash(self) -> str:
        """
        Generate a hash for error deduplication.

        Uses code, step_id, and agent_name for uniqueness.
        """
        hash_input = f"{self.code}:{self.step_id or ''}:{self.agent_name or ''}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


# =============================================================================
# ERROR REGISTRY
# =============================================================================


class ErrorRegistry:
    """
    Registry mapping error codes to metadata and suggested fixes.

    This registry provides:
    - Error code to ErrorClass mapping
    - Default suggested fixes for each error
    - HTTP status code to ErrorClass mapping
    - K8s exit code to ErrorClass mapping
    """

    _error_metadata: Dict[str, ErrorMetadata] = {}
    _http_status_mapping: Dict[int, ErrorClass] = {}
    _k8s_exit_code_mapping: Dict[int, str] = {}
    _initialized: bool = False

    @classmethod
    def initialize(cls) -> None:
        """Initialize the error registry with all error metadata."""
        if cls._initialized:
            return

        cls._register_yaml_errors()
        cls._register_dag_errors()
        cls._register_agent_errors()
        cls._register_param_errors()
        cls._register_k8s_errors()
        cls._register_s3_errors()
        cls._register_policy_errors()
        cls._register_quota_errors()
        cls._register_network_errors()
        cls._register_db_errors()
        cls._register_provenance_errors()
        cls._register_data_errors()
        cls._register_internal_errors()
        cls._register_http_mappings()
        cls._register_k8s_exit_codes()

        cls._initialized = True
        logger.info(f"ErrorRegistry initialized with {len(cls._error_metadata)} error codes")

    @classmethod
    def _register_error(
        cls,
        code: str,
        error_class: ErrorClass,
        retry_policy: RetryPolicy,
        default_message: str,
        default_fixes: Optional[List[SuggestedFix]] = None,
        severity: str = "ERROR",
        category: str = "general",
    ) -> None:
        """Register an error code with its metadata."""
        cls._error_metadata[code] = ErrorMetadata(
            code=code,
            error_class=error_class,
            retry_policy=retry_policy,
            default_message=default_message,
            default_fixes=default_fixes or [],
            severity=severity,
            category=category,
        )

    @classmethod
    def _register_yaml_errors(cls) -> None:
        """Register YAML/configuration errors."""
        cls._register_error(
            ErrorCode.GL_E_YAML_INVALID,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Invalid YAML syntax: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Check YAML syntax using a validator (e.g., yamllint)",
                    doc_link="https://docs.greenlang.io/pipelines/yaml-syntax",
                )
            ],
            category="configuration",
        )
        cls._register_error(
            ErrorCode.GL_E_YAML_SCHEMA,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "YAML does not match pipeline schema: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Validate YAML against pipeline schema",
                    cli_command="greenlang pipeline validate <file>",
                    doc_link="https://docs.greenlang.io/pipelines/schema",
                )
            ],
            category="configuration",
        )
        cls._register_error(
            ErrorCode.GL_E_YAML_VERSION,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Unsupported pipeline version: {version}. Supported: {supported}",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="version",
                    description="Update pipeline version to a supported version",
                    doc_link="https://docs.greenlang.io/pipelines/versioning",
                )
            ],
            category="configuration",
        )

    @classmethod
    def _register_dag_errors(cls) -> None:
        """Register DAG errors."""
        cls._register_error(
            ErrorCode.GL_E_DAG_CYCLE,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Cycle detected in pipeline DAG: {cycle}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Remove circular dependency between steps. Check step dependencies.",
                    doc_link="https://docs.greenlang.io/pipelines/dag",
                )
            ],
            category="dag",
        )
        cls._register_error(
            ErrorCode.GL_E_DAG_ORPHAN,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Orphaned step '{step_id}' has no path to pipeline output",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Connect orphaned step to pipeline output or remove it",
                )
            ],
            category="dag",
        )
        cls._register_error(
            ErrorCode.GL_E_DAG_MISSING_DEP,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Step '{step_id}' references missing dependency '{dep_id}'",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    field="steps.{step_id}.depends_on",
                    description="Add missing step or fix dependency reference",
                )
            ],
            category="dag",
        )
        cls._register_error(
            ErrorCode.GL_E_DAG_DUPLICATE_ID,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Duplicate step ID: '{step_id}'",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Rename one of the duplicate steps to have a unique ID",
                )
            ],
            category="dag",
        )

    @classmethod
    def _register_agent_errors(cls) -> None:
        """Register agent errors."""
        cls._register_error(
            ErrorCode.GL_E_AGENT_NOT_FOUND,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Agent '{agent_name}' not found in registry",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.agent",
                    description="Check agent name spelling or use 'greenlang agent list' to see available agents",
                    cli_command="greenlang agent list",
                )
            ],
            category="agent",
        )
        cls._register_error(
            ErrorCode.GL_E_AGENT_VERSION,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Agent '{agent_name}' version '{version}' not available. Available: {available}",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.agent_version",
                    description="Use an available agent version",
                    cli_command="greenlang agent versions {agent_name}",
                )
            ],
            category="agent",
        )
        cls._register_error(
            ErrorCode.GL_E_AGENT_DEPRECATED,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Agent '{agent_name}' is deprecated. Replacement: '{replacement}'",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.agent",
                    description="Migrate to the replacement agent",
                    doc_link="https://docs.greenlang.io/agents/deprecation",
                )
            ],
            category="agent",
        )
        cls._register_error(
            ErrorCode.GL_E_AGENT_CRASH,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Agent '{agent_name}' crashed: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Check agent logs for stack trace. Report issue if bug confirmed.",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact support with run ID and error details",
                ),
            ],
            severity="CRITICAL",
            category="agent",
        )
        cls._register_error(
            ErrorCode.GL_E_AGENT_TIMEOUT,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Agent '{agent_name}' execution timed out after {timeout}s",
            [
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    field="steps.{step_id}.timeout",
                    description="Increase timeout for this step",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Reduce input data size or enable chunked processing",
                ),
            ],
            category="agent",
        )
        cls._register_error(
            ErrorCode.GL_E_AGENT_OUTPUT_INVALID,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Agent '{agent_name}' output does not match schema: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Agent output schema violation. Check agent implementation.",
                )
            ],
            category="agent",
        )

    @classmethod
    def _register_param_errors(cls) -> None:
        """Register parameter errors."""
        cls._register_error(
            ErrorCode.GL_E_PARAM_MISSING,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Required parameter '{param}' is missing for step '{step_id}'",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.params.{param}",
                    description="Add the required parameter",
                )
            ],
            category="parameter",
        )
        cls._register_error(
            ErrorCode.GL_E_PARAM_TYPE,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Parameter '{param}' has incorrect type. Expected {expected}, got {actual}",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.params.{param}",
                    description="Change parameter value to correct type",
                )
            ],
            category="parameter",
        )
        cls._register_error(
            ErrorCode.GL_E_PARAM_RANGE,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Parameter '{param}' value {value} out of range [{min}, {max}]",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.params.{param}",
                    description="Adjust parameter value to be within allowed range",
                )
            ],
            category="parameter",
        )
        cls._register_error(
            ErrorCode.GL_E_PARAM_ENUM,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Parameter '{param}' value '{value}' not in allowed values: {allowed}",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.params.{param}",
                    description="Use one of the allowed values",
                )
            ],
            category="parameter",
        )
        cls._register_error(
            ErrorCode.GL_E_PARAM_FORMAT,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Parameter '{param}' has invalid format. Expected: {format}",
            [
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.params.{param}",
                    description="Fix parameter format to match expected pattern",
                )
            ],
            category="parameter",
        )

    @classmethod
    def _register_k8s_errors(cls) -> None:
        """Register Kubernetes errors."""
        cls._register_error(
            ErrorCode.GL_E_K8S_JOB_OOM,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Kubernetes job killed due to OOM (exit code 137). Memory limit: {memory_limit}",
            [
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    field="steps.{step_id}.resources.memory",
                    description="Increase memory limit. Recommended: double current limit.",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Enable streaming or reduce batch size to lower memory usage",
                ),
            ],
            severity="CRITICAL",
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_JOB_TIMEOUT,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Kubernetes job exceeded active deadline of {deadline}s",
            [
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    field="steps.{step_id}.resources.timeout",
                    description="Increase job timeout",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    field="steps.{step_id}.resources.cpu",
                    description="Increase CPU allocation to speed up processing",
                ),
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_IMAGEPULL,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Failed to pull image '{image}': {error}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Transient issue - retry will be attempted automatically",
                ),
                SuggestedFix(
                    fix_type=FixType.INFRASTRUCTURE_FIX,
                    description="Check image registry connectivity and credentials",
                ),
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_EVICTION,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Pod evicted due to node pressure: {reason}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Job will be rescheduled automatically",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    description="Request QoS Guaranteed by setting resource requests = limits",
                ),
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_QUOTA,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Namespace quota exceeded: {resource} ({used}/{limit})",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request quota increase for namespace",
                    cli_command="greenlang quota request --resource {resource} --amount {requested}",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    description="Reduce resource requests or wait for other jobs to complete",
                ),
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_SCHEDULING,
            ErrorClass.RESOURCE,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Failed to schedule pod: {reason}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Cluster may be congested - retry with backoff",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    description="Reduce resource requests to fit available capacity",
                ),
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_NODE_FAILURE,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Node failure during execution: {node}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Job will be rescheduled on healthy node",
                )
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_NETWORK,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Kubernetes network error: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Transient network issue - retrying automatically",
                )
            ],
            category="kubernetes",
        )
        cls._register_error(
            ErrorCode.GL_E_K8S_API,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Kubernetes API server error: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="API server may be temporarily overloaded - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact platform team if error persists",
                ),
            ],
            category="kubernetes",
        )

    @classmethod
    def _register_s3_errors(cls) -> None:
        """Register S3/storage errors."""
        cls._register_error(
            ErrorCode.GL_E_S3_ACCESS_DENIED,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "S3 access denied: {bucket}/{key}",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request S3 bucket access permissions",
                    doc_link="https://docs.greenlang.io/storage/permissions",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Verify bucket name and key path are correct",
                ),
            ],
            category="storage",
        )
        cls._register_error(
            ErrorCode.GL_E_S3_NOT_FOUND,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "S3 object not found: {bucket}/{key}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Verify object exists and path is correct",
                    cli_command="aws s3 ls s3://{bucket}/{key}",
                )
            ],
            category="storage",
        )
        cls._register_error(
            ErrorCode.GL_E_S3_TIMEOUT,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "S3 operation timed out: {operation}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Transient issue - retrying automatically",
                )
            ],
            category="storage",
        )
        cls._register_error(
            ErrorCode.GL_E_S3_QUOTA,
            ErrorClass.RESOURCE,
            RetryPolicy.NO_RETRY,
            "S3 storage quota exceeded",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request storage quota increase",
                ),
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Clean up old artifacts to free storage",
                    cli_command="greenlang artifact cleanup --older-than 30d",
                ),
            ],
            category="storage",
        )
        cls._register_error(
            ErrorCode.GL_E_S3_INTEGRITY,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "S3 data integrity check failed: expected {expected}, got {actual}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Re-upload artifact from source",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact support if corruption persists",
                ),
            ],
            severity="CRITICAL",
            category="storage",
        )
        cls._register_error(
            ErrorCode.GL_E_STORAGE_CORRUPTED,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.NO_RETRY,
            "Artifact storage data corrupted: {artifact_id}",
            [
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact support - data recovery may be needed",
                )
            ],
            severity="CRITICAL",
            category="storage",
        )

    @classmethod
    def _register_policy_errors(cls) -> None:
        """Register policy errors."""
        cls._register_error(
            ErrorCode.GL_E_OPA_DENY,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "OPA policy denied: {policy} - {reason}",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request policy exception or modify pipeline to comply",
                    doc_link="https://docs.greenlang.io/governance/policies",
                ),
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Review policy requirements and modify pipeline accordingly",
                ),
            ],
            category="policy",
        )
        cls._register_error(
            ErrorCode.GL_E_YAML_POLICY,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "YAML policy rule violated: {rule}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Modify pipeline YAML to comply with policy",
                    doc_link="https://docs.greenlang.io/governance/yaml-policies",
                )
            ],
            category="policy",
        )
        cls._register_error(
            ErrorCode.GL_E_RBAC_DENIED,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "RBAC permission denied: {action} on {resource}",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request required role/permission from admin",
                    cli_command="greenlang access request --role {role}",
                )
            ],
            category="policy",
        )
        cls._register_error(
            ErrorCode.GL_E_TENANT_ISOLATION,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "Tenant isolation policy violated: cannot access {resource}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Use resources within your tenant namespace",
                )
            ],
            severity="CRITICAL",
            category="policy",
        )
        cls._register_error(
            ErrorCode.GL_E_DATA_CLASSIFICATION,
            ErrorClass.POLICY_DENIED,
            RetryPolicy.NO_RETRY,
            "Data classification policy violated: {classification} data cannot be used in {context}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Use appropriately classified data for this pipeline",
                    doc_link="https://docs.greenlang.io/governance/data-classification",
                )
            ],
            severity="CRITICAL",
            category="policy",
        )

    @classmethod
    def _register_quota_errors(cls) -> None:
        """Register quota errors."""
        cls._register_error(
            ErrorCode.GL_E_QUOTA_EXCEEDED,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Quota exceeded: {quota_type} ({used}/{limit})",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request quota increase",
                    cli_command="greenlang quota request --type {quota_type} --amount {requested}",
                )
            ],
            category="quota",
        )
        cls._register_error(
            ErrorCode.GL_E_QUOTA_CPU,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "CPU quota exceeded: {used}/{limit} cores",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request CPU quota increase",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    description="Reduce CPU requests per step",
                ),
            ],
            category="quota",
        )
        cls._register_error(
            ErrorCode.GL_E_QUOTA_MEMORY,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Memory quota exceeded: {used}/{limit}",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request memory quota increase",
                ),
                SuggestedFix(
                    fix_type=FixType.RESOURCE_CHANGE,
                    description="Reduce memory requests per step",
                ),
            ],
            category="quota",
        )
        cls._register_error(
            ErrorCode.GL_E_QUOTA_GPU,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "GPU quota exceeded: {used}/{limit} GPUs",
            [
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request GPU quota increase",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Consider using CPU-only mode if applicable",
                ),
            ],
            category="quota",
        )
        cls._register_error(
            ErrorCode.GL_E_QUOTA_RUNS,
            ErrorClass.RESOURCE,
            RetryPolicy.BOUNDED_RETRY,
            "Concurrent run quota exceeded: {used}/{limit} runs",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Wait for existing runs to complete",
                ),
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request concurrent run quota increase",
                ),
            ],
            category="quota",
        )
        cls._register_error(
            ErrorCode.GL_E_QUOTA_STORAGE,
            ErrorClass.RESOURCE,
            RetryPolicy.NO_RETRY,
            "Storage quota exceeded: {used}/{limit}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Clean up old artifacts",
                    cli_command="greenlang artifact cleanup --older-than 30d",
                ),
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request storage quota increase",
                ),
            ],
            category="quota",
        )

    @classmethod
    def _register_network_errors(cls) -> None:
        """Register network/API errors."""
        cls._register_error(
            ErrorCode.GL_E_NETWORK_TIMEOUT,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Network connection timed out: {endpoint}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Transient issue - retrying automatically",
                )
            ],
            category="network",
        )
        cls._register_error(
            ErrorCode.GL_E_NETWORK_DNS,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "DNS resolution failed: {hostname}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="DNS may be temporarily unavailable - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.INFRASTRUCTURE_FIX,
                    description="Check DNS configuration if error persists",
                ),
            ],
            category="network",
        )
        cls._register_error(
            ErrorCode.GL_E_NETWORK_REFUSED,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Connection refused: {endpoint}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Service may be restarting - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.INFRASTRUCTURE_FIX,
                    description="Check service health if error persists",
                ),
            ],
            category="network",
        )
        cls._register_error(
            ErrorCode.GL_E_API_RATE_LIMIT,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "API rate limit exceeded: {api} (retry after {retry_after}s)",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Rate limit will reset - retrying with backoff",
                ),
                SuggestedFix(
                    fix_type=FixType.POLICY_REQUEST,
                    description="Request rate limit increase if consistently hitting limits",
                ),
            ],
            category="network",
        )
        cls._register_error(
            ErrorCode.GL_E_API_5XX,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "API server error ({status}): {endpoint}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Server error - retrying automatically",
                )
            ],
            category="network",
        )
        cls._register_error(
            ErrorCode.GL_E_API_UNAVAILABLE,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "API service unavailable: {service}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Service may be under maintenance - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Check service status page or contact support if prolonged",
                ),
            ],
            category="network",
        )

    @classmethod
    def _register_db_errors(cls) -> None:
        """Register database errors."""
        cls._register_error(
            ErrorCode.GL_E_DB_CONNECTION,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Database connection failed: {database}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Connection may be temporarily unavailable - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.INFRASTRUCTURE_FIX,
                    description="Check database health if error persists",
                ),
            ],
            category="database",
        )
        cls._register_error(
            ErrorCode.GL_E_DB_TIMEOUT,
            ErrorClass.TRANSIENT,
            RetryPolicy.EXPONENTIAL_BACKOFF,
            "Database query timed out after {timeout}s",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Query may have been slow - retrying",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Reduce query scope or add filters",
                ),
            ],
            category="database",
        )
        cls._register_error(
            ErrorCode.GL_E_DB_DEADLOCK,
            ErrorClass.TRANSIENT,
            RetryPolicy.IMMEDIATE_RETRY,
            "Database deadlock detected",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Deadlock resolved - retrying immediately",
                )
            ],
            category="database",
        )
        cls._register_error(
            ErrorCode.GL_E_DB_INTEGRITY,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Database integrity constraint violated: {constraint}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Check data for duplicates or invalid references",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact support if constraint is unexpected",
                ),
            ],
            category="database",
        )

    @classmethod
    def _register_provenance_errors(cls) -> None:
        """Register provenance/audit errors."""
        cls._register_error(
            ErrorCode.GL_E_PROVENANCE_MISSING,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Required provenance data missing: {field}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Agent must include provenance metadata in output",
                    doc_link="https://docs.greenlang.io/provenance",
                )
            ],
            severity="CRITICAL",
            category="provenance",
        )
        cls._register_error(
            ErrorCode.GL_E_PROVENANCE_MISMATCH,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Provenance hash mismatch: expected {expected}, got {actual}",
            [
                SuggestedFix(
                    fix_type=FixType.CODE_FIX,
                    description="Data may have been modified - check agent implementation",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Report potential data integrity issue",
                ),
            ],
            severity="CRITICAL",
            category="provenance",
        )
        cls._register_error(
            ErrorCode.GL_E_AUDIT_CHAIN_BROKEN,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Hash chain integrity compromised at block {block}",
            [
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Report audit chain break - investigation required",
                )
            ],
            severity="CRITICAL",
            category="provenance",
        )

    @classmethod
    def _register_data_errors(cls) -> None:
        """Register data errors."""
        cls._register_error(
            ErrorCode.GL_E_DATA_NOT_FOUND,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Input dataset not found: {dataset}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Verify dataset exists and path is correct",
                    cli_command="greenlang dataset list",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    field="steps.{step_id}.input",
                    description="Update input dataset reference",
                ),
            ],
            category="data",
        )
        cls._register_error(
            ErrorCode.GL_E_DATA_SCHEMA,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Input data schema mismatch: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Validate input data against expected schema",
                    cli_command="greenlang data validate --schema {schema} {file}",
                )
            ],
            category="data",
        )
        cls._register_error(
            ErrorCode.GL_E_DATA_EMPTY,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Input dataset is empty: {dataset}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Provide non-empty dataset or check upstream steps",
                )
            ],
            category="data",
        )
        cls._register_error(
            ErrorCode.GL_E_DATA_CORRUPTED,
            ErrorClass.INFRASTRUCTURE,
            RetryPolicy.NO_RETRY,
            "Input data is corrupted: {dataset}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Re-upload or regenerate corrupted dataset",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Contact support if corruption source is unclear",
                ),
            ],
            severity="CRITICAL",
            category="data",
        )
        cls._register_error(
            ErrorCode.GL_E_DATA_SIZE,
            ErrorClass.USER_CONFIG,
            RetryPolicy.NO_RETRY,
            "Input data exceeds size limit: {size} > {limit}",
            [
                SuggestedFix(
                    fix_type=FixType.DATA_FIX,
                    description="Reduce dataset size or split into chunks",
                ),
                SuggestedFix(
                    fix_type=FixType.PARAM_CHANGE,
                    description="Enable chunked processing mode",
                ),
            ],
            category="data",
        )

    @classmethod
    def _register_internal_errors(cls) -> None:
        """Register internal errors."""
        cls._register_error(
            ErrorCode.GL_E_INTERNAL,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Internal orchestrator error: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Report internal error with run ID and details",
                )
            ],
            severity="CRITICAL",
            category="internal",
        )
        cls._register_error(
            ErrorCode.GL_E_ASSERTION,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Internal assertion failed: {assertion}",
            [
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Report assertion failure - this is a bug",
                )
            ],
            severity="CRITICAL",
            category="internal",
        )
        cls._register_error(
            ErrorCode.GL_E_UNEXPECTED,
            ErrorClass.AGENT_BUG,
            RetryPolicy.NO_RETRY,
            "Unexpected error occurred: {error}",
            [
                SuggestedFix(
                    fix_type=FixType.RETRY,
                    description="Try running again - may be transient",
                ),
                SuggestedFix(
                    fix_type=FixType.CONTACT_SUPPORT,
                    description="Report if error persists",
                ),
            ],
            severity="CRITICAL",
            category="internal",
        )

    @classmethod
    def _register_http_mappings(cls) -> None:
        """Register HTTP status code to ErrorClass mappings."""
        # Transient (retry)
        cls._http_status_mapping[408] = ErrorClass.TRANSIENT  # Request Timeout
        cls._http_status_mapping[425] = ErrorClass.TRANSIENT  # Too Early
        cls._http_status_mapping[429] = ErrorClass.TRANSIENT  # Too Many Requests
        cls._http_status_mapping[500] = ErrorClass.TRANSIENT  # Internal Server Error
        cls._http_status_mapping[502] = ErrorClass.TRANSIENT  # Bad Gateway
        cls._http_status_mapping[503] = ErrorClass.TRANSIENT  # Service Unavailable
        cls._http_status_mapping[504] = ErrorClass.TRANSIENT  # Gateway Timeout

        # User config (no retry)
        cls._http_status_mapping[400] = ErrorClass.USER_CONFIG  # Bad Request
        cls._http_status_mapping[404] = ErrorClass.USER_CONFIG  # Not Found
        cls._http_status_mapping[405] = ErrorClass.USER_CONFIG  # Method Not Allowed
        cls._http_status_mapping[406] = ErrorClass.USER_CONFIG  # Not Acceptable
        cls._http_status_mapping[409] = ErrorClass.USER_CONFIG  # Conflict
        cls._http_status_mapping[410] = ErrorClass.USER_CONFIG  # Gone
        cls._http_status_mapping[411] = ErrorClass.USER_CONFIG  # Length Required
        cls._http_status_mapping[413] = ErrorClass.USER_CONFIG  # Payload Too Large
        cls._http_status_mapping[414] = ErrorClass.USER_CONFIG  # URI Too Long
        cls._http_status_mapping[415] = ErrorClass.USER_CONFIG  # Unsupported Media Type
        cls._http_status_mapping[422] = ErrorClass.USER_CONFIG  # Unprocessable Entity

        # Policy denied (no retry)
        cls._http_status_mapping[401] = ErrorClass.POLICY_DENIED  # Unauthorized
        cls._http_status_mapping[403] = ErrorClass.POLICY_DENIED  # Forbidden
        cls._http_status_mapping[407] = ErrorClass.POLICY_DENIED  # Proxy Auth Required
        cls._http_status_mapping[451] = ErrorClass.POLICY_DENIED  # Unavailable For Legal

        # Infrastructure (depends)
        cls._http_status_mapping[501] = ErrorClass.INFRASTRUCTURE  # Not Implemented
        cls._http_status_mapping[505] = ErrorClass.INFRASTRUCTURE  # HTTP Version Not Supported

    @classmethod
    def _register_k8s_exit_codes(cls) -> None:
        """Register Kubernetes exit code mappings."""
        # Standard exit codes
        cls._k8s_exit_code_mapping[0] = "SUCCESS"
        cls._k8s_exit_code_mapping[1] = ErrorCode.GL_E_AGENT_CRASH
        cls._k8s_exit_code_mapping[2] = ErrorCode.GL_E_AGENT_CRASH

        # Signal-based exit codes (128 + signal number)
        cls._k8s_exit_code_mapping[137] = ErrorCode.GL_E_K8S_JOB_OOM  # SIGKILL (9)
        cls._k8s_exit_code_mapping[143] = ErrorCode.GL_E_K8S_JOB_TIMEOUT  # SIGTERM (15)
        cls._k8s_exit_code_mapping[139] = ErrorCode.GL_E_AGENT_CRASH  # SIGSEGV (11)
        cls._k8s_exit_code_mapping[134] = ErrorCode.GL_E_AGENT_CRASH  # SIGABRT (6)

        # Custom exit codes (application-specific)
        cls._k8s_exit_code_mapping[100] = ErrorCode.GL_E_PARAM_MISSING
        cls._k8s_exit_code_mapping[101] = ErrorCode.GL_E_DATA_NOT_FOUND
        cls._k8s_exit_code_mapping[102] = ErrorCode.GL_E_DATA_SCHEMA
        cls._k8s_exit_code_mapping[103] = ErrorCode.GL_E_AGENT_OUTPUT_INVALID

    @classmethod
    def get_metadata(cls, code: str) -> Optional[ErrorMetadata]:
        """Get metadata for an error code."""
        cls.initialize()
        return cls._error_metadata.get(code)

    @classmethod
    def get_error_class(cls, code: str) -> ErrorClass:
        """Get error class for an error code."""
        cls.initialize()
        metadata = cls._error_metadata.get(code)
        if metadata:
            return metadata.error_class
        return ErrorClass.AGENT_BUG  # Default for unknown errors

    @classmethod
    def get_retry_policy(cls, code: str) -> RetryPolicy:
        """Get retry policy for an error code."""
        cls.initialize()
        metadata = cls._error_metadata.get(code)
        if metadata:
            return metadata.retry_policy
        return RetryPolicy.NO_RETRY  # Default for unknown errors

    @classmethod
    def get_default_fixes(cls, code: str) -> List[SuggestedFix]:
        """Get default suggested fixes for an error code."""
        cls.initialize()
        metadata = cls._error_metadata.get(code)
        if metadata:
            return list(metadata.default_fixes)
        return []

    @classmethod
    def get_http_error_class(cls, status_code: int) -> ErrorClass:
        """Map HTTP status code to error class."""
        cls.initialize()
        return cls._http_status_mapping.get(status_code, ErrorClass.AGENT_BUG)

    @classmethod
    def get_k8s_error_code(cls, exit_code: int) -> Optional[str]:
        """Map K8s exit code to error code."""
        cls.initialize()
        result = cls._k8s_exit_code_mapping.get(exit_code)
        if result == "SUCCESS":
            return None
        return result

    @classmethod
    def list_codes(cls, category: Optional[str] = None) -> List[str]:
        """List all error codes, optionally filtered by category."""
        cls.initialize()
        codes = []
        for code, metadata in cls._error_metadata.items():
            if category is None or metadata.category == category:
                codes.append(code)
        return sorted(codes)

    @classmethod
    def list_categories(cls) -> List[str]:
        """List all error categories."""
        cls.initialize()
        categories = set()
        for metadata in cls._error_metadata.values():
            categories.add(metadata.category)
        return sorted(categories)


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================


def format_error_cli(error: OrchestrationError, color: bool = True) -> str:
    """
    Format an OrchestrationError for CLI output.

    Args:
        error: The error to format
        color: Whether to include ANSI color codes

    Returns:
        Formatted error string for terminal display
    """
    # ANSI color codes
    if color:
        RED = "\033[91m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"
    else:
        RED = YELLOW = CYAN = BOLD = DIM = RESET = ""

    lines = []

    # Error header
    error_class_colors = {
        ErrorClass.TRANSIENT: YELLOW,
        ErrorClass.RESOURCE: YELLOW,
        ErrorClass.USER_CONFIG: RED,
        ErrorClass.POLICY_DENIED: RED,
        ErrorClass.AGENT_BUG: RED,
        ErrorClass.INFRASTRUCTURE: YELLOW,
    }
    error_color = error_class_colors.get(error.error_class, RED)

    lines.append(f"{error_color}{BOLD}Error: {error.code}{RESET}")
    lines.append(f"{DIM}Class: {error.error_class.value}{RESET}")
    lines.append("")
    lines.append(error.message)

    # Details
    if error.details:
        lines.append("")
        lines.append(f"{CYAN}Details:{RESET}")
        for key, value in error.details.items():
            lines.append(f"  {key}: {value}")

    # Context
    context_parts = []
    if error.run_id:
        context_parts.append(f"Run: {error.run_id}")
    if error.step_id:
        context_parts.append(f"Step: {error.step_id}")
    if error.agent_name:
        context_parts.append(f"Agent: {error.agent_name}")

    if context_parts:
        lines.append("")
        lines.append(f"{DIM}{' | '.join(context_parts)}{RESET}")

    # Suggested fixes
    if error.suggested_fixes:
        lines.append("")
        lines.append(f"{CYAN}Suggested Fixes:{RESET}")
        for i, fix in enumerate(error.suggested_fixes, 1):
            lines.append(f"  {i}. [{fix.fix_type.value}] {fix.description}")
            if fix.field:
                lines.append(f"     Field: {fix.field}")
            if fix.recommended_value is not None:
                lines.append(f"     Recommended: {fix.recommended_value}")
            if fix.cli_command:
                lines.append(f"     Command: {BOLD}{fix.cli_command}{RESET}")
            if fix.doc_link:
                lines.append(f"     Docs: {fix.doc_link}")

    # Links
    if error.links:
        lines.append("")
        lines.append(f"{CYAN}Links:{RESET}")
        for name, url in error.links.items():
            lines.append(f"  {name}: {url}")

    # Retry info
    if error.retry_policy != RetryPolicy.NO_RETRY:
        lines.append("")
        lines.append(f"{YELLOW}Retry Policy: {error.retry_policy.value}{RESET}")

    return "\n".join(lines)


def format_error_json(
    error: OrchestrationError,
    include_stack_trace: bool = False,
) -> Dict[str, Any]:
    """
    Format an OrchestrationError as JSON for CI/CD pipelines.

    Args:
        error: The error to format
        include_stack_trace: Whether to include stack trace

    Returns:
        Dictionary suitable for JSON serialization
    """
    result: Dict[str, Any] = {
        "error": {
            "code": error.code,
            "class": error.error_class.value,
            "message": error.message,
            "retry_policy": error.retry_policy.value,
            "timestamp": error.timestamp.isoformat() + "Z",
        },
        "context": {
            "run_id": error.run_id,
            "step_id": error.step_id,
            "agent_name": error.agent_name,
            "trace_id": error.trace_id,
            "correlation_id": error.correlation_id,
        },
        "details": error.details,
        "suggested_fixes": [
            {
                "type": fix.fix_type.value,
                "field": fix.field,
                "recommended_value": fix.recommended_value,
                "description": fix.description,
                "cli_command": fix.cli_command,
                "doc_link": fix.doc_link,
            }
            for fix in error.suggested_fixes
        ],
        "links": error.links,
    }

    # Remove None values from context
    result["context"] = {k: v for k, v in result["context"].items() if v is not None}

    # Include stack trace if requested
    if include_stack_trace and error.stack_trace:
        result["debug"] = {"stack_trace": error.stack_trace}

    # Add cause if present
    if error.cause:
        result["error"]["cause"] = error.cause

    return result


def format_error_markdown(error: OrchestrationError) -> str:
    """
    Format an OrchestrationError as Markdown for reports/issues.

    Args:
        error: The error to format

    Returns:
        Markdown-formatted error string
    """
    lines = []

    # Header
    lines.append(f"## Error: {error.code}")
    lines.append("")
    lines.append(f"**Class:** `{error.error_class.value}`")
    lines.append(f"**Retry Policy:** `{error.retry_policy.value}`")
    lines.append(f"**Timestamp:** {error.timestamp.isoformat()}Z")
    lines.append("")

    # Message
    lines.append("### Message")
    lines.append("")
    lines.append(f"> {error.message}")
    lines.append("")

    # Context
    context_items = []
    if error.run_id:
        context_items.append(f"- **Run ID:** `{error.run_id}`")
    if error.step_id:
        context_items.append(f"- **Step ID:** `{error.step_id}`")
    if error.agent_name:
        context_items.append(f"- **Agent:** `{error.agent_name}`")
    if error.trace_id:
        context_items.append(f"- **Trace ID:** `{error.trace_id}`")

    if context_items:
        lines.append("### Context")
        lines.append("")
        lines.extend(context_items)
        lines.append("")

    # Details
    if error.details:
        lines.append("### Details")
        lines.append("")
        lines.append("| Key | Value |")
        lines.append("|-----|-------|")
        for key, value in error.details.items():
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")

    # Suggested Fixes
    if error.suggested_fixes:
        lines.append("### Suggested Fixes")
        lines.append("")
        for i, fix in enumerate(error.suggested_fixes, 1):
            lines.append(f"#### {i}. {fix.fix_type.value}")
            lines.append("")
            lines.append(fix.description)
            if fix.field:
                lines.append(f"- **Field:** `{fix.field}`")
            if fix.recommended_value is not None:
                lines.append(f"- **Recommended Value:** `{fix.recommended_value}`")
            if fix.cli_command:
                lines.append(f"- **Command:** `{fix.cli_command}`")
            if fix.doc_link:
                lines.append(f"- **Documentation:** [{fix.doc_link}]({fix.doc_link})")
            lines.append("")

    # Links
    if error.links:
        lines.append("### Related Links")
        lines.append("")
        for name, url in error.links.items():
            lines.append(f"- [{name}]({url})")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_error(
    code: str,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    suggested_fixes: Optional[List[SuggestedFix]] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    links: Optional[Dict[str, str]] = None,
    **message_kwargs: Any,
) -> OrchestrationError:
    """
    Create an OrchestrationError from an error code.

    Uses the ErrorRegistry to populate default values.

    Args:
        code: Error code (GL-E-*)
        message: Custom message (uses default if not provided)
        details: Error-specific details
        suggested_fixes: Custom fixes (uses defaults if not provided)
        run_id: Pipeline run ID
        step_id: Step ID
        agent_name: Agent name
        links: Related links
        **message_kwargs: Values to format into the message template

    Returns:
        Configured OrchestrationError
    """
    ErrorRegistry.initialize()
    metadata = ErrorRegistry.get_metadata(code)

    if metadata:
        error_class = metadata.error_class
        retry_policy = metadata.retry_policy
        default_message = metadata.default_message
        default_fixes = list(metadata.default_fixes)
    else:
        error_class = ErrorClass.AGENT_BUG
        retry_policy = RetryPolicy.NO_RETRY
        default_message = "Unknown error: {code}"
        default_fixes = []
        logger.warning(f"Unknown error code: {code}")

    # Format message
    if message is None:
        try:
            message = default_message.format(code=code, **message_kwargs)
        except KeyError as e:
            message = f"{default_message} (missing key: {e})"

    # Use provided fixes or defaults
    final_fixes = suggested_fixes if suggested_fixes is not None else default_fixes

    return OrchestrationError(
        code=code,
        error_class=error_class,
        message=message,
        details=details or {},
        suggested_fixes=final_fixes,
        retry_policy=retry_policy,
        run_id=run_id,
        step_id=step_id,
        agent_name=agent_name,
        links=links or {},
    )


def create_validation_error(
    error: str,
    step_id: Optional[str] = None,
    field: Optional[str] = None,
    **kwargs: Any,
) -> OrchestrationError:
    """Create a validation error (YAML, schema, param)."""
    if "yaml" in error.lower() or "syntax" in error.lower():
        code = ErrorCode.GL_E_YAML_INVALID
    elif "schema" in error.lower():
        code = ErrorCode.GL_E_YAML_SCHEMA
    else:
        code = ErrorCode.GL_E_PARAM_MISSING

    details = {"error": error}
    if field:
        details["field"] = field

    return create_error(
        code=code,
        step_id=step_id,
        details=details,
        error=error,
        **kwargs,
    )


def create_resource_error(
    resource_type: str,
    used: Optional[str] = None,
    limit: Optional[str] = None,
    step_id: Optional[str] = None,
    **kwargs: Any,
) -> OrchestrationError:
    """Create a resource error (OOM, quota, timeout)."""
    resource_to_code = {
        "memory": ErrorCode.GL_E_K8S_JOB_OOM,
        "oom": ErrorCode.GL_E_K8S_JOB_OOM,
        "timeout": ErrorCode.GL_E_K8S_JOB_TIMEOUT,
        "cpu": ErrorCode.GL_E_QUOTA_CPU,
        "gpu": ErrorCode.GL_E_QUOTA_GPU,
        "storage": ErrorCode.GL_E_QUOTA_STORAGE,
    }

    code = resource_to_code.get(resource_type.lower(), ErrorCode.GL_E_QUOTA_EXCEEDED)

    details: Dict[str, Any] = {"resource_type": resource_type}
    if used:
        details["used"] = used
    if limit:
        details["limit"] = limit

    return create_error(
        code=code,
        step_id=step_id,
        details=details,
        memory_limit=limit,
        used=used,
        limit=limit,
        **kwargs,
    )


def create_policy_error(
    policy: str,
    reason: str,
    step_id: Optional[str] = None,
    **kwargs: Any,
) -> OrchestrationError:
    """Create a policy error (OPA, RBAC, data classification)."""
    policy_lower = policy.lower()

    if "opa" in policy_lower:
        code = ErrorCode.GL_E_OPA_DENY
    elif "rbac" in policy_lower:
        code = ErrorCode.GL_E_RBAC_DENIED
    elif "tenant" in policy_lower:
        code = ErrorCode.GL_E_TENANT_ISOLATION
    elif "classification" in policy_lower or "data" in policy_lower:
        code = ErrorCode.GL_E_DATA_CLASSIFICATION
    else:
        code = ErrorCode.GL_E_YAML_POLICY

    return create_error(
        code=code,
        step_id=step_id,
        details={"policy": policy, "reason": reason},
        policy=policy,
        reason=reason,
        **kwargs,
    )


def create_infrastructure_error(
    component: str,
    error: str,
    step_id: Optional[str] = None,
    **kwargs: Any,
) -> OrchestrationError:
    """Create an infrastructure error (K8s, S3, DB)."""
    component_lower = component.lower()

    if "k8s" in component_lower or "kubernetes" in component_lower:
        code = ErrorCode.GL_E_K8S_API
    elif "s3" in component_lower:
        code = ErrorCode.GL_E_S3_TIMEOUT
    elif "db" in component_lower or "database" in component_lower:
        code = ErrorCode.GL_E_DB_CONNECTION
    else:
        code = ErrorCode.GL_E_INTERNAL

    return create_error(
        code=code,
        step_id=step_id,
        details={"component": component, "error": error},
        error=error,
        **kwargs,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_error_class_for_http_status(status_code: int) -> ErrorClass:
    """
    Get error class for an HTTP status code.

    Args:
        status_code: HTTP status code (e.g., 500, 403)

    Returns:
        Appropriate ErrorClass
    """
    return ErrorRegistry.get_http_error_class(status_code)


def get_error_class_for_k8s_exit_code(exit_code: int) -> tuple[Optional[str], ErrorClass]:
    """
    Get error code and class for a Kubernetes exit code.

    Args:
        exit_code: Container exit code (e.g., 137 for OOM)

    Returns:
        Tuple of (error_code, error_class)
    """
    ErrorRegistry.initialize()
    error_code = ErrorRegistry.get_k8s_error_code(exit_code)

    if error_code is None:
        return None, ErrorClass.AGENT_BUG  # Unknown exit code

    metadata = ErrorRegistry.get_metadata(error_code)
    if metadata:
        return error_code, metadata.error_class

    return error_code, ErrorClass.AGENT_BUG


def serialize_error_chain(errors: List[OrchestrationError]) -> Dict[str, Any]:
    """
    Serialize a chain of related errors.

    Useful for representing cascading failures.

    Args:
        errors: List of related errors (ordered from root cause to effect)

    Returns:
        Dictionary with error chain representation
    """
    if not errors:
        return {"errors": [], "count": 0}

    root_error = errors[0]
    chain = []

    for i, error in enumerate(errors):
        chain.append({
            "index": i,
            "code": error.code,
            "class": error.error_class.value,
            "message": error.message,
            "step_id": error.step_id,
            "is_root_cause": i == 0,
        })

    return {
        "root_cause": format_error_json(root_error),
        "chain": chain,
        "count": len(errors),
        "summary": f"{len(errors)} errors starting with {root_error.code}",
    }


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Initialize registry on module load
ErrorRegistry.initialize()


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
