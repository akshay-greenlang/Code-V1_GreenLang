# -*- coding: utf-8 -*-
"""
GreenLang Orchestrator Module
=============================

GL-FOUND-X-001: The core orchestration layer for GreenLang Climate OS.

This module provides:
- Pipeline YAML compilation and planning
- DAG execution with GLIP v1 protocol
- K8s Job execution backend
- S3 artifact management
- OPA/YAML policy governance
- Hash-chained audit trail
- FastAPI control plane API
- CLI interface
- Pipeline Template Engine (FR-005)

Submodules:
    - executors: Execution backends (K8s, local, legacy)
    - artifacts: Artifact storage (S3, local)
    - governance: Policy engine (OPA + YAML)
    - audit: Event store and audit trail
    - errors: Error taxonomy and structured errors
    - adapters: Legacy agent adapters
    - api: FastAPI routes
    - schemas: JSON schemas for validation
    - template_engine: Pipeline templates and modules (FR-005)

Author: GreenLang Team
Version: 2.1.0
"""

from greenlang.orchestrator.pipeline_schema import (
    # Core models
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    PipelineDefaults,
    StepDefinition,
    ParameterDefinition,
    PolicyAttachment,
    # Enums
    ParameterType,
    PolicySeverity as PipelinePolicySeverity,
    StepType,
    DataClassification,
    # Legacy models
    ResourceRequirements,
    ArtifactDefinition,
    RunConfig,
    StepResult,
    ExecutionContext,
    # Constants
    SUPPORTED_API_VERSION,
    AGENT_ID_PATTERN,
    # Functions
    load_pipeline_yaml,
    load_pipeline_file,
    validate_agent_id,
    extract_template_references,
)

from greenlang.orchestrator.governance import (
    PolicyEngine,
    PolicyEngineConfig,
    PolicyDecision,
    PolicyReason,
    ApprovalRequirement,
    PolicyAction,
    PolicySeverity,
    EvaluationPoint,
    PolicyBundle,
    YAMLRule,
    CostBudget,
    DataResidencyRule,
)

# Executors (always available - no external deps)
from greenlang.orchestrator.executors.base import (
    ExecutorBackend,
    RunContext as GLIPRunContext,
    StepResult as GLIPStepResult,
    ExecutionStatus as StepStatus,  # Alias for compatibility
    StepMetadata,
    ResourceProfile as GLIPResourceProfile,
    ExecutionResult as GLIPExecutionResult,
)

# GLIP Orchestrator (imports gracefully handle missing dependencies)
try:
    from greenlang.orchestrator.glip_orchestrator import (
        GLIPOrchestrator,
        GLIPOrchestratorConfig,
        GLIPRunConfig,
        GLIPExecutionMode,
        GLIPStepContext,
        create_glip_orchestrator,
        # P1 Features
        ApprovalStatus,
        ApprovalRequest,
        FanOutSpec,
        ConcurrencyConfig,
    )
    GLIP_AVAILABLE = True
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"GLIP Orchestrator not available: {e}")
    GLIPOrchestrator = None
    GLIPOrchestratorConfig = None
    GLIPRunConfig = None
    GLIPExecutionMode = None
    GLIPStepContext = None
    create_glip_orchestrator = None
    # P1 Features
    ApprovalStatus = None
    ApprovalRequest = None
    FanOutSpec = None
    ConcurrencyConfig = None
    GLIP_AVAILABLE = False

try:
    from greenlang.orchestrator.executors.k8s_executor import (
        K8sExecutor,
        K8sExecutorConfig,
    )
    K8S_AVAILABLE = True
except ImportError:
    K8sExecutor = None
    K8sExecutorConfig = None
    K8S_AVAILABLE = False

# Artifacts
from greenlang.orchestrator.artifacts.base import (
    ArtifactStore,
    ArtifactMetadata,
    ArtifactManifest,
)

try:
    from greenlang.orchestrator.artifacts.s3_store import (
        S3ArtifactStore,
        S3StoreConfig,
    )
    S3_AVAILABLE = True
except ImportError:
    S3ArtifactStore = None
    S3StoreConfig = None
    S3_AVAILABLE = False

# Audit
try:
    from greenlang.orchestrator.audit.event_store import (
        EventStore as AuditEventStore,
        RunEvent as AuditEvent,
        EventType,
    )
    # Config alias
    AuditEventStoreConfig = None  # Not yet implemented
    AUDIT_AVAILABLE = True
except ImportError:
    AuditEventStore = None
    AuditEventStoreConfig = None
    AuditEvent = None
    EventType = None
    AUDIT_AVAILABLE = False

# Adapters
try:
    from greenlang.orchestrator.adapters.http_legacy_adapter import (
        HttpLegacyAdapter,
        AdapterConfig,
    )
    ADAPTERS_AVAILABLE = True
except ImportError:
    HttpLegacyAdapter = None
    AdapterConfig = None
    ADAPTERS_AVAILABLE = False

# Template Engine (FR-005)
from greenlang.orchestrator.template_engine import (
    # Core models
    PipelineTemplate,
    TemplateParameter,
    TemplateStep,
    TemplateImport,
    ExpandedStep,
    TemplateExpansionResult,
    # Enums
    TemplateParameterType,
    TemplateStatus,
    # Registry and resolver
    TemplateRegistry,
    TemplateResolver,
    # Constants
    MAX_TEMPLATE_NESTING_DEPTH,
    # Functions
    load_template_yaml,
    load_template_file,
    create_template_registry,
)

__all__ = [
    # Pipeline Schema - Core Models
    "PipelineDefinition",
    "PipelineMetadata",
    "PipelineSpec",
    "PipelineDefaults",
    "StepDefinition",
    "ParameterDefinition",
    "PolicyAttachment",
    # Pipeline Schema - Enums
    "ParameterType",
    "PipelinePolicySeverity",
    "StepType",
    "DataClassification",
    # Pipeline Schema - Legacy Models
    "ResourceRequirements",
    "ArtifactDefinition",
    "RunConfig",
    "StepResult",
    "ExecutionContext",
    # Pipeline Schema - Constants
    "SUPPORTED_API_VERSION",
    "AGENT_ID_PATTERN",
    # Pipeline Schema - Functions
    "load_pipeline_yaml",
    "load_pipeline_file",
    "validate_agent_id",
    "extract_template_references",
    # Policy Engine
    "PolicyEngine",
    "PolicyEngineConfig",
    "PolicyDecision",
    "PolicyReason",
    "ApprovalRequirement",
    "PolicyAction",
    "PolicySeverity",
    "EvaluationPoint",
    "PolicyBundle",
    "YAMLRule",
    "CostBudget",
    "DataResidencyRule",
    # GLIP v1 Orchestrator
    "GLIPOrchestrator",
    "GLIPOrchestratorConfig",
    "GLIPRunConfig",
    "GLIPExecutionMode",
    "GLIPStepContext",
    "create_glip_orchestrator",
    # GLIP v1 P1 Features
    "ApprovalStatus",
    "ApprovalRequest",
    "FanOutSpec",
    "ConcurrencyConfig",
    # Executors
    "ExecutorBackend",
    "GLIPRunContext",
    "GLIPStepResult",
    "StepStatus",
    "StepMetadata",
    "GLIPResourceProfile",
    "K8sExecutor",
    "K8sExecutorConfig",
    # Artifacts
    "ArtifactStore",
    "ArtifactMetadata",
    "ArtifactManifest",
    "S3ArtifactStore",
    "S3StoreConfig",
    # Audit
    "AuditEventStore",
    "AuditEventStoreConfig",
    "AuditEvent",
    "EventType",
    # Adapters
    "HttpLegacyAdapter",
    "AdapterConfig",
    # Availability flags
    "GLIP_AVAILABLE",
    "K8S_AVAILABLE",
    "S3_AVAILABLE",
    "AUDIT_AVAILABLE",
    "ADAPTERS_AVAILABLE",
    # Template Engine (FR-005)
    "PipelineTemplate",
    "TemplateParameter",
    "TemplateStep",
    "TemplateImport",
    "ExpandedStep",
    "TemplateExpansionResult",
    "TemplateParameterType",
    "TemplateStatus",
    "TemplateRegistry",
    "TemplateResolver",
    "MAX_TEMPLATE_NESTING_DEPTH",
    "load_template_yaml",
    "load_template_file",
    "create_template_registry",
    # ================================================================
    # AGENT-FOUND-001: DAG Orchestrator (added February 2026)
    # ================================================================
    # DAG Configuration
    "OrchestratorConfig",
    "get_orchestrator_config",
    "set_orchestrator_config",
    "reset_orchestrator_config",
    # DAG Models - Enums
    "DAGExecutionStatus",
    "DAGNodeStatus",
    "DAGOnFailureStrategy",
    "DAGNodeOnFailure",
    "DAGRetryStrategyType",
    "DAGOnTimeout",
    # DAG Models - Policy dataclasses
    "DAGRetryPolicy",
    "DAGTimeoutPolicy",
    # DAG Models - Core definitions
    "DAGNode",
    "DAGWorkflow",
    # DAG Models - Execution results
    "NodeExecutionResult",
    "NodeProvenance",
    "DAGExecutionTrace",
    "DAGCheckpoint",
    # DAG Builder
    "DAGBuilder",
    # DAG Validator
    "DAGValidationError",
    "validate_dag_workflow",
    "detect_dag_cycles",
    "find_dag_unreachable_nodes",
    # Topological Sort
    "dag_topological_sort",
    "dag_level_grouping",
    "get_dag_roots",
    "get_dag_sinks",
    # DAG Executor
    "DAGExecutor",
    "DAGExecutionContext",
    "DAGExecutionOptions",
    # Node Runner
    "DAGNodeRunner",
    # Checkpoint Store
    "DAGCheckpointStore",
    "MemoryDAGCheckpointStore",
    "FileDAGCheckpointStore",
    "create_dag_checkpoint_store",
    # Provenance
    "DAGProvenanceTracker",
    # Determinism
    "DeterministicScheduler",
    # Metrics
    "DAG_PROMETHEUS_AVAILABLE",
    "record_dag_execution",
    "record_dag_node_execution",
    "record_dag_node_retry",
    "record_dag_node_timeout",
    # Setup
    "DAGOrchestrator",
    "configure_dag_orchestrator",
    "get_dag_orchestrator",
    # DAG Availability flag
    "DAG_ORCHESTRATOR_AVAILABLE",
]

__version__ = "2.2.0"


# =========================================================================
# AGENT-FOUND-001: DAG Orchestrator Imports (new module, February 2026)
# =========================================================================

try:
    # Configuration
    from greenlang.orchestrator.config import (
        OrchestratorConfig,
        get_config as get_orchestrator_config,
        set_config as set_orchestrator_config,
        reset_config as reset_orchestrator_config,
    )

    # Models - Enums (aliased to avoid name collisions with GLIP enums)
    from greenlang.orchestrator.models import (
        ExecutionStatus as DAGExecutionStatus,
        NodeStatus as DAGNodeStatus,
        DAGOnFailure as DAGOnFailureStrategy,
        OnFailure as DAGNodeOnFailure,
        RetryStrategyType as DAGRetryStrategyType,
        OnTimeout as DAGOnTimeout,
    )

    # Models - Policy dataclasses
    from greenlang.orchestrator.models import (
        RetryPolicy as DAGRetryPolicy,
        TimeoutPolicy as DAGTimeoutPolicy,
    )

    # Models - Core definitions
    from greenlang.orchestrator.models import (
        DAGNode,
        DAGWorkflow,
    )

    # Models - Execution results
    from greenlang.orchestrator.models import (
        NodeExecutionResult,
        NodeProvenance,
        ExecutionTrace as DAGExecutionTrace,
        DAGCheckpoint,
    )

    # Builder
    from greenlang.orchestrator.dag_builder import DAGBuilder

    # Validator
    from greenlang.orchestrator.dag_validator import (
        ValidationError as DAGValidationError,
        validate_dag as validate_dag_workflow,
        detect_cycles as detect_dag_cycles,
        find_unreachable_nodes as find_dag_unreachable_nodes,
    )

    # Topological Sort
    from greenlang.orchestrator.topological_sort import (
        topological_sort as dag_topological_sort,
        level_grouping as dag_level_grouping,
        get_roots as get_dag_roots,
        get_sinks as get_dag_sinks,
    )

    # Executor
    from greenlang.orchestrator.dag_executor import (
        DAGExecutor,
        ExecutionContext as DAGExecutionContext,
        ExecutionOptions as DAGExecutionOptions,
    )

    # Node Runner
    from greenlang.orchestrator.node_runner import (
        NodeRunner as DAGNodeRunner,
    )

    # Checkpoint Store
    from greenlang.orchestrator.checkpoint_store import (
        DAGCheckpointStore,
        MemoryDAGCheckpointStore,
        FileDAGCheckpointStore,
        create_checkpoint_store as create_dag_checkpoint_store,
    )

    # Provenance
    from greenlang.orchestrator.provenance import (
        ProvenanceTracker as DAGProvenanceTracker,
    )

    # Determinism
    from greenlang.orchestrator.determinism import (
        DeterministicScheduler,
    )

    # Metrics
    from greenlang.orchestrator.metrics import (
        PROMETHEUS_AVAILABLE as DAG_PROMETHEUS_AVAILABLE,
        record_dag_execution,
        record_node_execution as record_dag_node_execution,
        record_node_retry as record_dag_node_retry,
        record_node_timeout as record_dag_node_timeout,
    )

    # Setup
    from greenlang.orchestrator.dag_setup import (
        DAGOrchestrator,
        configure_dag_orchestrator,
        get_dag_orchestrator,
    )

    DAG_ORCHESTRATOR_AVAILABLE = True

except ImportError as _dag_err:
    import logging as _log
    _log.getLogger(__name__).debug(
        "DAG Orchestrator (AGENT-FOUND-001) not fully available: %s",
        _dag_err,
    )
    DAG_ORCHESTRATOR_AVAILABLE = False
