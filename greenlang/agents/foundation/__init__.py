# -*- coding: utf-8 -*-
"""GreenLang Foundation Layer Agents - updated for SDK migration."""

# GL-FOUND-X-001: GreenLang Orchestrator (full SDK package)
from greenlang.agents.foundation.orchestrator import (
    PipelineDefinition, PipelineMetadata, PipelineSpec, StepDefinition,
    PolicyEngine as OrchestratorPolicyEngine, PolicyDecision,
    ExecutorBackend, ArtifactStore, ArtifactMetadata,
    PipelineTemplate, TemplateRegistry, TemplateResolver,
    DAG_ORCHESTRATOR_AVAILABLE, GLIP_AVAILABLE,
)

if DAG_ORCHESTRATOR_AVAILABLE:
    from greenlang.agents.foundation.orchestrator import (
        DAGOrchestrator, DAGExecutor, DAGBuilder, DAGNode, DAGWorkflow,
        DAGNodeRunner, DAGCheckpointStore, DeterministicScheduler,
        configure_dag_orchestrator, get_dag_orchestrator,
    )

# GL-FOUND-X-002: Schema Compiler & Validator (Layer 1 + SDK)
from greenlang.agents.foundation.schema_compiler import (
    SchemaCompilerAgent, SchemaRegistry, SchemaRegistryEntry,
    TypeCoercionEngine, UnitConsistencyChecker, FixSuggestionGenerator,
    SchemaCompilerInput, SchemaCompilerOutput, FixSuggestion,
    FixSuggestionType, CoercionType, CoercionRecord, UnitInfo,
    SchemaType, UNIT_FAMILIES, UNIT_CONVERSIONS,
)

from greenlang.agents.foundation.schema import (
    SchemaService, configure_schema_service, get_schema_service,
    validate as schema_validate, validate_batch as schema_validate_batch,
    compile_schema, CompiledSchema,
)

# GL-FOUND-X-003: Unit & Reference Normalizer (Layer 1 + SDK)
from greenlang.agents.foundation.unit_normalizer import (
    UnitNormalizerAgent, UnitDimension, GHGType,
    ConversionRequest, ConversionResult,
    GHGConversionRequest, GHGConversionResult,
    FuelStandardizationRequest, FuelStandardizationResult,
    MaterialStandardizationRequest, MaterialStandardizationResult,
    ReferenceIDRequest, ReferenceIDResult,
    CurrencyConversionRequest, CurrencyConversionResult,
    NormalizerInput, NormalizerOutput,
    GWP_AR6_100, GWP_AR5_100, GWP_AR4_100,
)

from greenlang.agents.foundation.normalizer import (
    NormalizerService, configure_normalizer_service, get_normalizer_service,
    UnitConverter, EntityResolver,
)

# GL-FOUND-X-004: Assumptions Registry Agent (Layer 1 + SDK)
from greenlang.agents.foundation.assumptions_registry import (
    AssumptionsRegistryAgent, AssumptionDataType, AssumptionCategory,
    ScenarioType, ChangeType, ValidationSeverity, ValidationRule,
    ValidationResult, AssumptionMetadata, AssumptionVersion,
    Assumption, Scenario, ChangeLogEntry, DependencyNode,
    AssumptionsRegistryInput, AssumptionsRegistryOutput,
)

from greenlang.agents.foundation.assumptions import (
    AssumptionsService, configure_assumptions_service, get_assumptions_service,
    AssumptionRegistry, ScenarioManager, AssumptionValidator,
)

# GL-FOUND-X-005: Citations & Evidence Agent (Layer 1 + SDK)
from greenlang.agents.foundation.citations_agent import (
    CitationsEvidenceAgent, Citation, CitationMetadata,
    CitationType, SourceAuthority,
    VerificationStatus as CitationVerificationStatus,
    RegulatoryFramework, EvidenceItem, EvidenceType, EvidencePackage,
    MethodologyReference, RegulatoryRequirement, DataSourceAttribution,
    CitationsAgentInput, CitationsAgentOutput,
)

from greenlang.agents.foundation.citations import (
    CitationsService, configure_citations_service, get_citations_service,
    CitationRegistry, EvidenceManager, VerificationEngine,
)

# GL-FOUND-X-006: Access & Policy Guard Agent (Layer 1 + SDK)
from greenlang.agents.foundation.policy_guard import (
    PolicyGuardAgent, AccessDecision, PolicyType, DataClassification,
    RoleType, AuditEventType, Principal, Resource, AccessRequest,
    PolicyRule, Policy, AccessDecisionResult, AuditEvent,
    RateLimitConfig, PolicyGuardConfig, ComplianceReport,
    PolicySimulationResult, PolicyEngine, RateLimiter,
    CLASSIFICATION_HIERARCHY, DEFAULT_ROLE_PERMISSIONS,
)

from greenlang.agents.foundation.access_guard import (
    AccessGuardService, configure_access_guard, get_access_guard,
    DataClassifier, AuditLogger,
)

# GL-FOUND-X-007: PII Redaction & Minimization Agent
from greenlang.agents.foundation.pii_redaction import (
    PIIRedactionAgent, PIIType, RedactionStrategy, ComplianceFramework,
    DetectionConfidence, AuditAction, PIIMatch, RedactedMatch,
    TokenEntry, RedactionPolicy, PIIRedactionInput, PIIRedactionOutput,
    AuditLogEntry,
)

# GL-FOUND-X-008: Quality Gate & Test Harness Agent (full SDK)
from greenlang.agents.foundation.qa_test_harness import (
    TestCaseInput, TestCaseResult, TestSuiteInput, TestSuiteResult,
    TestStatus, TestCategory, SeverityLevel, TestAssertion,
    PerformanceBenchmark, CoverageReport, GoldenFileSpec,
    TestFixture, COMMON_FIXTURES,
    QATestHarnessService, configure_qa_test_harness, get_qa_test_harness,
    TestRunner, AssertionEngine, GoldenFileManager,
    RegressionDetector, PerformanceBenchmarker, CoverageTracker,
    ReportGenerator,
)
QATestHarnessAgent = QATestHarnessService  # Backward-compatible alias

# GL-FOUND-X-009: Observability & Telemetry Agent (full SDK)
from greenlang.agents.foundation.observability_agent import (
    ObservabilityInput, ObservabilityOutput, MetricType, MetricValue,
    MetricDefinition, TraceContext, SpanDefinition, LogEntry,
    AlertRule, Alert, AlertSeverity, AlertStatus, HealthStatus, HealthCheck,
    ObservabilityAgentService, configure_observability_agent, get_observability_agent,
    MetricsCollector, TraceManager, LogAggregator, AlertEvaluator,
    HealthChecker as ObsHealthChecker, DashboardProvider, SLOTracker,
)
ObservabilityAgent = ObservabilityAgentService  # Backward-compatible alias

# GL-FOUND-X-010: Agent Registry & Versioning Agent (full SDK)
from greenlang.agents.foundation.agent_registry import (
    AgentLayer, SectorClassification, AgentHealthStatus,
    CapabilityCategory, SemanticVersion, AgentCapability,
    AgentVariant, AgentDependency, AgentMetadataEntry,
    RegistryQueryInput, RegistryQueryOutput,
    DependencyResolutionInput, DependencyResolutionOutput,
    AgentRegistryService, configure_agent_registry, get_agent_registry,
    AgentRegistry, HealthChecker as RegistryHealthChecker,
    DependencyResolver, CapabilityMatcher,
)
VersionedAgentRegistry = AgentRegistry  # Backward-compatible alias

# Supplementary: Reproducibility Agent (Layer 1 + SDK)
from greenlang.agents.foundation.reproducibility_agent import (
    ReproducibilityAgent, ReproducibilityInput, ReproducibilityOutput,
    ReproducibilityReport, VerificationStatus, DriftSeverity,
    NonDeterminismSource, EnvironmentFingerprint, SeedConfiguration,
    VersionManifest, VersionPin, ReplayConfiguration,
    VerificationCheck, DriftDetection,
)

from greenlang.agents.foundation.reproducibility import (
    ReproducibilityService, configure_reproducibility, get_reproducibility,
    ArtifactHasher, DeterminismVerifier, DriftDetector, ReplayEngine,
)

