# -*- coding: utf-8 -*-
"""
Pytest Fixtures for GreenLang Orchestrator Determinism Tests

Provides shared fixtures for testing the GL-FOUND-X-001 Orchestrator:
- Mock orchestrator with deterministic configuration
- Mock artifact store (S3 mock)
- Mock K8s executor
- Sample pipeline definitions
- Deterministic clock fixture
- In-memory event store

All fixtures ensure deterministic behavior by:
- Using DeterministicClock for timestamps
- Using content-addressable hashing
- Providing reproducible test data

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# GreenLang imports
from greenlang.utilities.determinism import (
    DeterministicClock,
    freeze_time,
    unfreeze_time,
    deterministic_id,
)

# Orchestrator imports
from greenlang.orchestrator.pipeline_schema import (
    PipelineDefinition,
    PipelineMetadata,
    PipelineSpec,
    PipelineDefaults,
    StepDefinition,
    ParameterDefinition,
    ParameterType,
)
from greenlang.orchestrator.executors.base import (
    ExecutionStatus,
    ResourceProfile,
    RunContext,
    StepResult,
    StepMetadata,
    ExecutionResult,
    ArtifactReference,
)
from greenlang.orchestrator.artifacts.base import (
    ArtifactStore,
    ArtifactMetadata,
    ArtifactManifest,
    ArtifactType,
    RetentionPolicy,
)
from greenlang.orchestrator.audit.event_store import (
    EventType,
    RunEvent,
    InMemoryEventStore,
    GENESIS_HASH,
)


# =============================================================================
# FIXED TIMESTAMPS FOR DETERMINISTIC TESTING
# =============================================================================

FROZEN_TIME = datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
FROZEN_TIME_ISO = "2026-01-15T10:30:00+00:00"


# =============================================================================
# DETERMINISTIC CLOCK FIXTURES
# =============================================================================

@pytest.fixture
def frozen_clock():
    """
    Fixture that freezes the DeterministicClock for deterministic testing.

    All timestamps during the test will return the same frozen time.
    Automatically unfreezes after the test.

    Usage:
        def test_something(frozen_clock):
            ts1 = DeterministicClock.now()
            ts2 = DeterministicClock.now()
            assert ts1 == ts2  # Both are frozen time
    """
    freeze_time(FROZEN_TIME)
    yield FROZEN_TIME
    unfreeze_time()


@pytest.fixture
def deterministic_clock_context():
    """
    Context manager fixture for frozen clock.

    Returns a context manager that can be used to freeze time temporarily.
    """
    return DeterministicClock.frozen


# =============================================================================
# MOCK ARTIFACT STORE
# =============================================================================

class MockArtifactStore(ArtifactStore):
    """
    In-memory mock artifact store for testing.

    Provides deterministic artifact storage with:
    - Content-addressable checksums
    - Reproducible URIs
    - In-memory storage (no external dependencies)

    Attributes:
        _artifacts: Dict mapping URIs to artifact data
        _metadata: Dict mapping URIs to ArtifactMetadata
        bucket: Mock S3 bucket name
    """

    def __init__(self, bucket: str = "test-artifacts"):
        """Initialize mock artifact store."""
        self._artifacts: Dict[str, bytes] = {}
        self._metadata: Dict[str, ArtifactMetadata] = {}
        self.bucket = bucket

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum of data."""
        return hashlib.sha256(data).hexdigest()

    async def write_input_context(
        self,
        run_id: str,
        step_id: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> ArtifactMetadata:
        """Write input context to mock store."""
        data = json.dumps(context, sort_keys=True).encode("utf-8")
        uri = self.generate_input_uri(self.bucket, tenant_id, run_id, step_id)
        checksum = self._compute_checksum(data)

        self._artifacts[uri] = data

        metadata = ArtifactMetadata(
            artifact_id=f"{run_id}-{step_id}-input",
            uri=uri,
            artifact_type=ArtifactType.INPUT_CONTEXT,
            run_id=run_id,
            step_id=step_id,
            name="input.json",
            checksum=checksum,
            size_bytes=len(data),
            media_type="application/json",
            created_at=DeterministicClock.now(timezone.utc),
        )
        self._metadata[uri] = metadata
        return metadata

    async def read_result(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read result from mock store."""
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, "result.json"
        )
        if uri in self._artifacts:
            return json.loads(self._artifacts[uri].decode("utf-8"))
        return None

    async def read_step_metadata(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read step metadata from mock store."""
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, "step_metadata.json"
        )
        if uri in self._artifacts:
            return json.loads(self._artifacts[uri].decode("utf-8"))
        return None

    async def list_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> ArtifactManifest:
        """List artifacts for a run/step."""
        prefix = f"s3://{self.bucket}/"
        if tenant_id:
            prefix += f"{tenant_id}/"
        prefix += f"runs/{run_id}/"
        if step_id:
            prefix += f"steps/{step_id}/"

        artifacts = [
            meta for uri, meta in self._metadata.items()
            if uri.startswith(prefix)
        ]

        total_size = sum(a.size_bytes for a in artifacts)

        return ArtifactManifest(
            run_id=run_id,
            step_id=step_id,
            artifacts=artifacts,
            total_size_bytes=total_size,
            artifact_count=len(artifacts),
            manifest_checksum=self._compute_checksum(
                json.dumps([a.uri for a in artifacts]).encode()
            ),
        )

    async def write_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        data: bytes,
        media_type: str,
        tenant_id: str,
    ) -> ArtifactMetadata:
        """Write artifact to mock store."""
        if hasattr(data, 'read'):
            data = data.read()

        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, name
        )
        checksum = self._compute_checksum(data)

        self._artifacts[uri] = data

        metadata = ArtifactMetadata(
            artifact_id=f"{run_id}-{step_id}-{name}",
            uri=uri,
            artifact_type=ArtifactType.DATA,
            run_id=run_id,
            step_id=step_id,
            name=name,
            checksum=checksum,
            size_bytes=len(data),
            media_type=media_type,
            created_at=DeterministicClock.now(timezone.utc),
        )
        self._metadata[uri] = metadata
        return metadata

    async def read_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
    ) -> Optional[bytes]:
        """Read artifact from mock store."""
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, name
        )
        return self._artifacts.get(uri)

    async def get_presigned_url(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Generate mock presigned URL."""
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, name
        )
        return f"{uri}?X-Mock-Signature=test&expires={expires_in_seconds}"

    async def delete_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        """Delete artifacts from mock store."""
        prefix = f"s3://{self.bucket}/"
        if tenant_id:
            prefix += f"{tenant_id}/"
        prefix += f"runs/{run_id}/"
        if step_id:
            prefix += f"steps/{step_id}/"

        to_delete = [uri for uri in self._artifacts.keys() if uri.startswith(prefix)]
        for uri in to_delete:
            del self._artifacts[uri]
            self._metadata.pop(uri, None)

        return len(to_delete)

    async def verify_checksum(
        self,
        uri: str,
        expected_checksum: str,
    ) -> bool:
        """Verify artifact checksum."""
        if uri not in self._artifacts:
            return False
        actual = self._compute_checksum(self._artifacts[uri])
        return actual == expected_checksum

    async def apply_retention_policy(
        self,
        tenant_id: str,
        policy: RetentionPolicy,
    ) -> int:
        """Apply retention policy (mock - does nothing)."""
        return 0

    # Test helper methods

    def store_mock_result(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
        result: Dict[str, Any],
    ) -> str:
        """Helper to store a mock result for testing."""
        data = json.dumps(result, sort_keys=True).encode("utf-8")
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, "result.json"
        )
        self._artifacts[uri] = data
        return self._compute_checksum(data)

    def store_mock_step_metadata(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Helper to store mock step metadata for testing."""
        data = json.dumps(metadata, sort_keys=True).encode("utf-8")
        uri = self.generate_artifact_uri(
            self.bucket, tenant_id, run_id, step_id, "step_metadata.json"
        )
        self._artifacts[uri] = data
        return self._compute_checksum(data)


@pytest.fixture
def mock_artifact_store():
    """Fixture providing a mock artifact store."""
    return MockArtifactStore(bucket="test-greenlang-artifacts")


# =============================================================================
# MOCK K8S EXECUTOR
# =============================================================================

class MockK8sExecutor:
    """
    Mock K8s executor for deterministic testing.

    Simulates K8s Job execution with:
    - Deterministic execution times
    - Configurable success/failure scenarios
    - Mock artifact reading

    Attributes:
        artifact_store: Mock artifact store for results
        execution_results: Pre-configured execution outcomes
        execution_delay_ms: Simulated execution time
    """

    def __init__(
        self,
        artifact_store: MockArtifactStore,
        default_status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
        execution_delay_ms: float = 100.0,
    ):
        """Initialize mock K8s executor."""
        self.artifact_store = artifact_store
        self.default_status = default_status
        self.execution_delay_ms = execution_delay_ms
        self._step_results: Dict[str, ExecutionStatus] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def configure_step_result(
        self,
        step_id: str,
        status: ExecutionStatus,
        exit_code: Optional[int] = None,
    ):
        """Configure the result for a specific step."""
        self._step_results[step_id] = (status, exit_code)

    async def execute(
        self,
        context: RunContext,
        container_image: str,
        resources: ResourceProfile,
        namespace: str,
        input_uri: str,
        output_uri: str,
    ) -> ExecutionResult:
        """Execute a mock K8s job."""
        started_at = DeterministicClock.now(timezone.utc)

        # Record execution for verification
        self._execution_history.append({
            "step_id": context.step_id,
            "run_id": context.run_id,
            "idempotency_key": context.idempotency_key,
            "container_image": container_image,
            "input_uri": input_uri,
            "output_uri": output_uri,
        })

        # Get configured result or default
        if context.step_id in self._step_results:
            status, exit_code = self._step_results[context.step_id]
        else:
            status = self.default_status
            exit_code = 0 if status == ExecutionStatus.SUCCEEDED else 1

        # Read result from artifact store if succeeded
        result = None
        metadata = None
        error_message = None
        error_code = None

        if status == ExecutionStatus.SUCCEEDED:
            result_data = await self.artifact_store.read_result(
                run_id=context.run_id,
                step_id=context.step_id,
                tenant_id=context.tenant_id,
            )
            if result_data:
                result = StepResult(
                    success=True,
                    data=result_data.get("outputs", {}),
                    artifacts={},
                )

            metadata_data = await self.artifact_store.read_step_metadata(
                run_id=context.run_id,
                step_id=context.step_id,
                tenant_id=context.tenant_id,
            )
            if metadata_data:
                metadata = StepMetadata(**metadata_data)
        else:
            error_message = f"Step {context.step_id} failed with status {status.value}"
            error_code = "GL-E-MOCK-FAILURE"

        completed_at = DeterministicClock.now(timezone.utc)

        return ExecutionResult(
            step_id=context.step_id,
            status=status,
            result=result,
            metadata=metadata,
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=self.execution_delay_ms,
            error_message=error_message,
            error_code=error_code,
            exit_code=exit_code,
            input_uri=input_uri,
            output_uri=output_uri,
        )

    async def cancel(self, step_id: str, namespace: str) -> bool:
        """Cancel a mock job (always succeeds)."""
        return True

    async def get_status(self, step_id: str, namespace: str) -> ExecutionStatus:
        """Get status of a mock job."""
        if step_id in self._step_results:
            return self._step_results[step_id][0]
        return self.default_status

    async def get_logs(
        self,
        step_id: str,
        namespace: str,
        tail_lines: Optional[int] = None,
    ) -> str:
        """Get mock logs."""
        return f"[MOCK] Execution logs for step {step_id}"

    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get list of all executed jobs for verification."""
        return self._execution_history.copy()

    def clear_history(self):
        """Clear execution history."""
        self._execution_history.clear()


@pytest.fixture
def mock_k8s_executor(mock_artifact_store):
    """Fixture providing a mock K8s executor."""
    return MockK8sExecutor(artifact_store=mock_artifact_store)


# =============================================================================
# SAMPLE PIPELINE DEFINITIONS
# =============================================================================

@pytest.fixture
def simple_pipeline_def() -> PipelineDefinition:
    """
    Simple single-step pipeline for basic tests.

    Pipeline structure:
        ingest -> (output)
    """
    return PipelineDefinition(
        apiVersion="greenlang/v1",
        kind="Pipeline",
        metadata=PipelineMetadata(
            name="simple-test-pipeline",
            namespace="test",
            version="1.0.0",
            labels={"env": "test"},
        ),
        spec=PipelineSpec(
            parameters={
                "input_uri": ParameterDefinition(
                    type=ParameterType.STRING,
                    required=True,
                    description="Input data URI",
                ),
            },
            defaults=PipelineDefaults(retries=0, timeoutSeconds=300),
            steps=[
                StepDefinition(
                    id="ingest",
                    agent="GL-DATA-X-001",
                    with_={"uri": "{{ params.input_uri }}"},
                    outputs={"dataset": "$.artifact.dataset_uri"},
                ),
            ],
        ),
    )


@pytest.fixture
def multi_step_pipeline_def() -> PipelineDefinition:
    """
    Multi-step pipeline with dependencies for DAG tests.

    Pipeline structure:
        ingest -> transform -> validate
                     |
                     v
                  export
    """
    return PipelineDefinition(
        apiVersion="greenlang/v1",
        kind="Pipeline",
        metadata=PipelineMetadata(
            name="multi-step-pipeline",
            namespace="test",
            version="1.0.0",
            labels={"env": "test", "complexity": "medium"},
        ),
        spec=PipelineSpec(
            parameters={
                "input_uri": ParameterDefinition(
                    type=ParameterType.STRING,
                    required=True,
                ),
                "output_format": ParameterDefinition(
                    type=ParameterType.STRING,
                    default="parquet",
                ),
            },
            defaults=PipelineDefaults(retries=1, timeoutSeconds=600),
            steps=[
                StepDefinition(
                    id="ingest",
                    agent="GL-DATA-X-001",
                    with_={"uri": "{{ params.input_uri }}"},
                    outputs={"raw_data": "$.artifact.raw_uri"},
                ),
                StepDefinition(
                    id="transform",
                    agent="GL-CALC-A-001",
                    dependsOn=["ingest"],
                    with_={"data_uri": "{{ steps.ingest.outputs.raw_data }}"},
                    outputs={"transformed": "$.artifact.transformed_uri"},
                ),
                StepDefinition(
                    id="validate",
                    agent="GL-DATA-V-001",
                    dependsOn=["transform"],
                    with_={"data_uri": "{{ steps.transform.outputs.transformed }}"},
                    outputs={"validation_report": "$.validation.report_uri"},
                ),
                StepDefinition(
                    id="export",
                    agent="GL-DATA-E-001",
                    dependsOn=["transform"],
                    with_={
                        "data_uri": "{{ steps.transform.outputs.transformed }}",
                        "format": "{{ params.output_format }}",
                    },
                    outputs={"export_uri": "$.artifact.export_uri"},
                ),
            ],
        ),
    )


@pytest.fixture
def carbon_calc_pipeline_def() -> PipelineDefinition:
    """
    Carbon calculation pipeline for emissions testing.

    Pipeline structure:
        collect_activity -> calculate_emissions -> generate_report
    """
    return PipelineDefinition(
        apiVersion="greenlang/v1",
        kind="Pipeline",
        metadata=PipelineMetadata(
            name="carbon-footprint-calc",
            namespace="sustainability",
            version="2.0.0",
            labels={"domain": "scope3", "regulation": "sb253"},
        ),
        spec=PipelineSpec(
            parameters={
                "facility_id": ParameterDefinition(
                    type=ParameterType.STRING,
                    required=True,
                ),
                "reporting_period": ParameterDefinition(
                    type=ParameterType.STRING,
                    required=True,
                ),
                "emission_factor_source": ParameterDefinition(
                    type=ParameterType.STRING,
                    default="EPA",
                ),
            },
            defaults=PipelineDefaults(retries=2, timeoutSeconds=1800),
            steps=[
                StepDefinition(
                    id="collect-activity",
                    agent="GL-DATA-C-001",
                    with_={
                        "facility_id": "{{ params.facility_id }}",
                        "period": "{{ params.reporting_period }}",
                    },
                    outputs={"activity_data": "$.artifact.activity_uri"},
                ),
                StepDefinition(
                    id="calculate-emissions",
                    agent="GL-CALC-E-001",
                    dependsOn=["collect-activity"],
                    with_={
                        "activity_uri": "{{ steps.collect-activity.outputs.activity_data }}",
                        "ef_source": "{{ params.emission_factor_source }}",
                    },
                    outputs={
                        "emissions": "$.artifact.emissions_uri",
                        "provenance": "$.provenance.hash",
                    },
                ),
                StepDefinition(
                    id="generate-report",
                    agent="GL-REPORT-X-001",
                    dependsOn=["calculate-emissions"],
                    with_={
                        "emissions_uri": "{{ steps.calculate-emissions.outputs.emissions }}",
                        "provenance_hash": "{{ steps.calculate-emissions.outputs.provenance }}",
                    },
                    outputs={"report": "$.artifact.report_uri"},
                ),
            ],
        ),
    )


# =============================================================================
# EVENT STORE FIXTURE
# =============================================================================

@pytest.fixture
def event_store():
    """Fixture providing an in-memory event store."""
    return InMemoryEventStore()


# =============================================================================
# RUN CONTEXT FACTORY
# =============================================================================

@pytest.fixture
def run_context_factory(frozen_clock):
    """
    Factory fixture for creating deterministic RunContext instances.

    Returns a function that creates RunContext with deterministic values.
    """
    def _create_context(
        run_id: str = "run-test-001",
        step_id: str = "step-001",
        pipeline_id: str = "test-pipeline",
        tenant_id: str = "tenant-test",
        agent_id: str = "GL-TEST-X-001",
        agent_version: str = "1.0.0",
        params: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, ArtifactReference]] = None,
        idempotency_key: Optional[str] = None,
        retry_attempt: int = 0,
    ) -> RunContext:
        """Create a RunContext with deterministic values."""
        # Generate deterministic IDs
        if idempotency_key is None:
            key_content = f"{run_id}:{step_id}:{retry_attempt}"
            idempotency_key = hashlib.sha256(key_content.encode()).hexdigest()[:32]

        trace_id = deterministic_id(f"trace-{run_id}")
        span_id = deterministic_id(f"span-{run_id}-{step_id}")
        log_id = deterministic_id(f"log-{run_id}-{step_id}")

        return RunContext(
            run_id=run_id,
            step_id=step_id,
            pipeline_id=pipeline_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            agent_version=agent_version,
            schema_version="1.0",
            params=params or {},
            inputs=inputs or {},
            permissions_context={},
            deadline_ts=None,
            timeout_seconds=300,
            retry_attempt=retry_attempt,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            span_id=span_id,
            log_correlation_id=log_id,
        )

    return _create_context


# =============================================================================
# MOCK ORCHESTRATOR FIXTURE
# =============================================================================

@pytest.fixture
def mock_glip_orchestrator_components(
    mock_artifact_store,
    mock_k8s_executor,
    event_store,
):
    """
    Fixture providing all mock components for the GLIP orchestrator.

    Returns a dict with:
    - artifact_store: MockArtifactStore
    - k8s_executor: MockK8sExecutor
    - event_store: InMemoryEventStore
    """
    return {
        "artifact_store": mock_artifact_store,
        "k8s_executor": mock_k8s_executor,
        "event_store": event_store,
    }


# =============================================================================
# HELPER FUNCTIONS AS FIXTURES
# =============================================================================

@pytest.fixture
def compute_content_hash():
    """
    Fixture providing a function to compute content-addressable hash.

    This ensures consistent hashing across all tests.
    """
    def _compute_hash(content: Any) -> str:
        """Compute SHA-256 hash of content."""
        if isinstance(content, str):
            data = content.encode("utf-8")
        elif isinstance(content, bytes):
            data = content
        else:
            data = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    return _compute_hash


@pytest.fixture
def assert_deterministic_hash(compute_content_hash):
    """
    Fixture providing an assertion helper for deterministic hashing.

    Verifies that computing a hash multiple times yields the same result.
    """
    def _assert_deterministic(content: Any, iterations: int = 10):
        """Assert that hash computation is deterministic."""
        hashes = [compute_content_hash(content) for _ in range(iterations)]
        assert len(set(hashes)) == 1, f"Hash not deterministic: {set(hashes)}"
        return hashes[0]

    return _assert_deterministic
