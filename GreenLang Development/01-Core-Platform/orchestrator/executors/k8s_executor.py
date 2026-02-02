# -*- coding: utf-8 -*-
"""
Kubernetes Job Executor Backend
================================

GLIP v1 execution backend using Kubernetes Jobs.

Each step runs as an isolated K8s Job with:
- Resource limits from agent registry
- Environment variables (GL_INPUT_URI, GL_OUTPUT_URI)
- Deadline/timeout enforcement
- Cancellation support

Author: GreenLang Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

# Optional kubernetes dependency - use TYPE_CHECKING for type hints
if TYPE_CHECKING:
    from kubernetes_asyncio import client as k8s_client
    from kubernetes_asyncio.client import ApiException as K8sApiException
    from kubernetes_asyncio.watch import Watch as K8sWatch

# Runtime import with fallback
try:
    from kubernetes_asyncio import client, config
    from kubernetes_asyncio.client import ApiException
    from kubernetes_asyncio.watch import Watch
    KUBERNETES_AVAILABLE = True
except ImportError:
    client = None  # type: ignore
    config = None  # type: ignore
    ApiException = Exception  # type: ignore
    Watch = None  # type: ignore
    KUBERNETES_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "kubernetes_asyncio not available. Install with: pip install kubernetes_asyncio"
    )

from greenlang.orchestrator.artifacts.base import ArtifactStore
from greenlang.orchestrator.executors.base import (
    ExecutionResult,
    ExecutionStatus,
    ExecutorBackend,
    ResourceProfile,
    RunContext,
    StepMetadata,
    StepResult,
)
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class K8sExecutorConfig:
    """Configuration for K8s executor."""

    def __init__(
        self,
        in_cluster: bool = True,
        kubeconfig_path: Optional[str] = None,
        service_account: str = "greenlang-executor",
        image_pull_policy: str = "IfNotPresent",
        image_pull_secrets: Optional[List[str]] = None,
        default_namespace: str = "greenlang",
        job_ttl_seconds_after_finished: int = 3600,
        active_deadline_seconds: int = 3600,
        backoff_limit: int = 0,  # We handle retries at orchestrator level
        restart_policy: str = "Never",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
        node_selector: Optional[Dict[str, str]] = None,
        tolerations: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Initialize K8s executor configuration.

        Args:
            in_cluster: Whether running inside K8s cluster
            kubeconfig_path: Path to kubeconfig (if not in_cluster)
            service_account: Service account for Jobs
            image_pull_policy: Image pull policy
            image_pull_secrets: Image pull secrets
            default_namespace: Default namespace for Jobs
            job_ttl_seconds_after_finished: TTL for Job cleanup
            active_deadline_seconds: Maximum Job duration
            backoff_limit: K8s retry limit (0 = no K8s retries)
            restart_policy: Pod restart policy
            labels: Additional labels for Jobs
            annotations: Additional annotations for Jobs
            node_selector: Node selector for Jobs
            tolerations: Tolerations for Jobs
        """
        self.in_cluster = in_cluster
        self.kubeconfig_path = kubeconfig_path
        self.service_account = service_account
        self.image_pull_policy = image_pull_policy
        self.image_pull_secrets = image_pull_secrets or []
        self.default_namespace = default_namespace
        self.job_ttl_seconds_after_finished = job_ttl_seconds_after_finished
        self.active_deadline_seconds = active_deadline_seconds
        self.backoff_limit = backoff_limit
        self.restart_policy = restart_policy
        self.labels = labels or {}
        self.annotations = annotations or {}
        self.node_selector = node_selector or {}
        self.tolerations = tolerations or []


class K8sExecutor(ExecutorBackend):
    """
    Kubernetes Job execution backend for GLIP v1.

    Executes agents as K8s Jobs with standardized environment:
    - GL_INPUT_URI: S3 URI for input.json
    - GL_OUTPUT_URI: S3 URI prefix for outputs

    Usage:
        config = K8sExecutorConfig(in_cluster=True)
        executor = K8sExecutor(config, artifact_store)

        result = await executor.execute(
            context=run_context,
            container_image="greenlang/agent-mrv:1.0.0",
            resources=resource_profile,
            namespace="tenant-acme",
            input_uri="s3://bucket/runs/123/steps/456/input.json",
            output_uri="s3://bucket/runs/123/steps/456/",
        )
    """

    def __init__(
        self,
        config: K8sExecutorConfig,
        artifact_store: ArtifactStore,
    ):
        """
        Initialize K8s executor.

        Args:
            config: K8s executor configuration
            artifact_store: Artifact store for reading results
        """
        self.config = config
        self.artifact_store = artifact_store
        self._initialized = False
        self._batch_api: Optional[Any] = None  # client.BatchV1Api when available
        self._core_api: Optional[Any] = None  # client.CoreV1Api when available
        logger.info(f"Initialized K8sExecutor: namespace={config.default_namespace}")

    async def _ensure_initialized(self):
        """Ensure K8s client is initialized."""
        if self._initialized:
            return

        if self.config.in_cluster:
            config.load_incluster_config()
        else:
            await config.load_kube_config(config_file=self.config.kubeconfig_path)

        self._batch_api = client.BatchV1Api()
        self._core_api = client.CoreV1Api()
        self._initialized = True
        logger.debug("K8s client initialized")

    def _generate_job_name(self, step_id: str) -> str:
        """Generate unique Job name from step ID."""
        # K8s names must be DNS-compatible: lowercase, max 63 chars
        short_id = step_id.replace("_", "-")[:50]
        suffix = uuid.uuid4().hex[:8]
        return f"gl-{short_id}-{suffix}".lower()

    def _build_job_spec(
        self,
        context: RunContext,
        container_image: str,
        resources: ResourceProfile,
        namespace: str,
        input_uri: str,
        output_uri: str,
    ) -> Any:  # Returns client.V1Job when kubernetes_asyncio is available
        """Build K8s Job specification."""
        job_name = self._generate_job_name(context.step_id)

        # Environment variables
        env_vars = [
            client.V1EnvVar(name="GL_INPUT_URI", value=input_uri),
            client.V1EnvVar(name="GL_OUTPUT_URI", value=output_uri),
            client.V1EnvVar(name="GL_RUN_ID", value=context.run_id),
            client.V1EnvVar(name="GL_STEP_ID", value=context.step_id),
            client.V1EnvVar(name="GL_PIPELINE_ID", value=context.pipeline_id),
            client.V1EnvVar(name="GL_AGENT_ID", value=context.agent_id),
            client.V1EnvVar(name="GL_AGENT_VERSION", value=context.agent_version),
            client.V1EnvVar(name="GL_TENANT_ID", value=context.tenant_id),
            client.V1EnvVar(name="GL_IDEMPOTENCY_KEY", value=context.idempotency_key),
            client.V1EnvVar(name="GL_RETRY_ATTEMPT", value=str(context.retry_attempt)),
            client.V1EnvVar(name="GL_TIMEOUT_SECONDS", value=str(context.timeout_seconds)),
            client.V1EnvVar(name="GL_TRACE_ID", value=context.trace_id),
            client.V1EnvVar(name="GL_SPAN_ID", value=context.span_id),
            client.V1EnvVar(name="GL_LOG_CORRELATION_ID", value=context.log_correlation_id),
        ]

        # Resource requirements
        resource_reqs = client.V1ResourceRequirements(
            requests={
                "cpu": resources.cpu_request,
                "memory": resources.memory_request,
            },
            limits={
                "cpu": resources.cpu_limit,
                "memory": resources.memory_limit,
            },
        )

        if resources.gpu_count > 0:
            resource_reqs.limits[resources.gpu_resource_key] = str(resources.gpu_count)
            resource_reqs.requests[resources.gpu_resource_key] = str(resources.gpu_count)

        # Container spec
        container = client.V1Container(
            name="agent",
            image=container_image,
            image_pull_policy=self.config.image_pull_policy,
            env=env_vars,
            resources=resource_reqs,
            command=["gl-agent", "run"],  # GLIP v1 standard entrypoint
        )

        # Pod spec
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy=self.config.restart_policy,
            service_account_name=self.config.service_account,
            node_selector=self.config.node_selector or None,
            tolerations=[
                client.V1Toleration(**t) for t in self.config.tolerations
            ] if self.config.tolerations else None,
        )

        # Add image pull secrets
        if self.config.image_pull_secrets:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=secret)
                for secret in self.config.image_pull_secrets
            ]

        # Job labels
        labels = {
            "app.kubernetes.io/name": "greenlang-agent",
            "app.kubernetes.io/component": "executor",
            "greenlang.io/run-id": context.run_id,
            "greenlang.io/step-id": context.step_id,
            "greenlang.io/agent-id": context.agent_id,
            "greenlang.io/tenant-id": context.tenant_id,
            **self.config.labels,
        }

        # Job annotations
        annotations = {
            "greenlang.io/idempotency-key": context.idempotency_key,
            "greenlang.io/trace-id": context.trace_id,
            **self.config.annotations,
        }

        # Calculate active deadline
        if context.deadline_ts:
            remaining = (context.deadline_ts - DeterministicClock.now()).total_seconds()
            active_deadline = max(int(remaining), 60)  # At least 60 seconds
        else:
            active_deadline = context.timeout_seconds

        # Job spec
        job_spec = client.V1JobSpec(
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(
                    labels=labels,
                    annotations=annotations,
                ),
                spec=pod_spec,
            ),
            backoff_limit=self.config.backoff_limit,
            active_deadline_seconds=active_deadline,
            ttl_seconds_after_finished=self.config.job_ttl_seconds_after_finished,
        )

        # Job object
        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=namespace,
                labels=labels,
                annotations=annotations,
            ),
            spec=job_spec,
        )

        return job

    async def execute(
        self,
        context: RunContext,
        container_image: str,
        resources: ResourceProfile,
        namespace: str,
        input_uri: str,
        output_uri: str,
    ) -> ExecutionResult:
        """Execute a GLIP v1 agent as a K8s Job."""
        await self._ensure_initialized()

        job = self._build_job_spec(
            context=context,
            container_image=container_image,
            resources=resources,
            namespace=namespace,
            input_uri=input_uri,
            output_uri=output_uri,
        )

        job_name = job.metadata.name
        started_at = DeterministicClock.now()

        logger.info(f"Creating K8s Job: {namespace}/{job_name} for step {context.step_id}")

        try:
            # Create the Job
            await self._batch_api.create_namespaced_job(
                namespace=namespace,
                body=job,
            )

            # Wait for Job completion
            status, exit_code = await self._wait_for_job_completion(
                job_name=job_name,
                namespace=namespace,
                timeout_seconds=context.timeout_seconds,
            )

            completed_at = DeterministicClock.now()
            duration_ms = (completed_at - started_at).total_seconds() * 1000

            # Read results from artifact store
            result = None
            metadata = None
            error_message = None
            error_code = None

            if status == ExecutionStatus.SUCCEEDED:
                # Parse tenant_id from output_uri
                # Format: s3://bucket/tenant_id/runs/run_id/steps/step_id/
                parts = output_uri.replace("s3://", "").split("/")
                if len(parts) >= 5:
                    tenant_id = parts[1]
                else:
                    tenant_id = context.tenant_id

                result_data = await self.artifact_store.read_result(
                    run_id=context.run_id,
                    step_id=context.step_id,
                    tenant_id=tenant_id,
                )
                if result_data:
                    result = StepResult(**result_data)

                metadata_data = await self.artifact_store.read_step_metadata(
                    run_id=context.run_id,
                    step_id=context.step_id,
                    tenant_id=tenant_id,
                )
                if metadata_data:
                    metadata = StepMetadata(**metadata_data)

            elif status == ExecutionStatus.FAILED:
                error_message = f"Job failed with exit code {exit_code}"
                error_code = self._exit_code_to_error_code(exit_code)

            elif status == ExecutionStatus.TIMEOUT:
                error_message = f"Job timed out after {context.timeout_seconds}s"
                error_code = "GL-E-K8S-JOB-TIMEOUT"

            elif status == ExecutionStatus.CANCELED:
                error_message = "Job was canceled"
                error_code = "GL-E-K8S-JOB-CANCELED"

            return ExecutionResult(
                step_id=context.step_id,
                status=status,
                result=result,
                metadata=metadata,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
                error_message=error_message,
                error_code=error_code,
                exit_code=exit_code,
                input_uri=input_uri,
                output_uri=output_uri,
            )

        except ApiException as e:
            logger.error(f"K8s API error for Job {job_name}: {e}")
            return ExecutionResult(
                step_id=context.step_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=DeterministicClock.now(),
                error_message=f"K8s API error: {e.reason}",
                error_code="GL-E-K8S-API-ERROR",
                input_uri=input_uri,
                output_uri=output_uri,
            )

        except Exception as e:
            logger.error(f"Unexpected error executing Job {job_name}: {e}", exc_info=True)
            return ExecutionResult(
                step_id=context.step_id,
                status=ExecutionStatus.FAILED,
                started_at=started_at,
                completed_at=DeterministicClock.now(),
                error_message=str(e),
                error_code="GL-E-K8S-EXECUTOR-ERROR",
                input_uri=input_uri,
                output_uri=output_uri,
            )

    async def _wait_for_job_completion(
        self,
        job_name: str,
        namespace: str,
        timeout_seconds: int,
    ) -> tuple[ExecutionStatus, Optional[int]]:
        """Wait for Job to complete and return status."""
        deadline = DeterministicClock.now() + timedelta(seconds=timeout_seconds + 60)

        while DeterministicClock.now() < deadline:
            try:
                job = await self._batch_api.read_namespaced_job_status(
                    name=job_name,
                    namespace=namespace,
                )

                status = job.status

                # Check completion conditions
                if status.succeeded and status.succeeded > 0:
                    return ExecutionStatus.SUCCEEDED, 0

                if status.failed and status.failed > 0:
                    # Get exit code from pod
                    exit_code = await self._get_pod_exit_code(job_name, namespace)
                    return ExecutionStatus.FAILED, exit_code

                # Check for deadline exceeded
                for condition in (status.conditions or []):
                    if condition.type == "Failed" and condition.reason == "DeadlineExceeded":
                        return ExecutionStatus.TIMEOUT, None

                # Still running, wait a bit
                await asyncio.sleep(2)

            except ApiException as e:
                if e.status == 404:
                    # Job was deleted (canceled)
                    return ExecutionStatus.CANCELED, None
                raise

        # Timeout waiting
        return ExecutionStatus.TIMEOUT, None

    async def _get_pod_exit_code(self, job_name: str, namespace: str) -> Optional[int]:
        """Get exit code from Job's pod."""
        try:
            pods = await self._core_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}",
            )

            for pod in pods.items:
                if pod.status and pod.status.container_statuses:
                    for container_status in pod.status.container_statuses:
                        if container_status.state and container_status.state.terminated:
                            return container_status.state.terminated.exit_code

        except Exception as e:
            logger.warning(f"Failed to get pod exit code for {job_name}: {e}")

        return None

    def _exit_code_to_error_code(self, exit_code: Optional[int]) -> str:
        """Map exit code to error code."""
        if exit_code is None:
            return "GL-E-K8S-JOB-UNKNOWN"

        # Common exit codes
        exit_code_map = {
            137: "GL-E-K8S-JOB-OOM",  # SIGKILL (usually OOM)
            143: "GL-E-K8S-JOB-SIGTERM",  # SIGTERM
            1: "GL-E-AGENT-ERROR",
            2: "GL-E-AGENT-USAGE-ERROR",
        }

        return exit_code_map.get(exit_code, f"GL-E-K8S-EXIT-{exit_code}")

    async def cancel(self, step_id: str, namespace: str) -> bool:
        """Cancel a running Job."""
        await self._ensure_initialized()

        try:
            # Find Job by step_id label
            jobs = await self._batch_api.list_namespaced_job(
                namespace=namespace,
                label_selector=f"greenlang.io/step-id={step_id}",
            )

            if not jobs.items:
                logger.warning(f"No Job found for step {step_id}")
                return False

            for job in jobs.items:
                # Delete Job (this also deletes pods)
                await self._batch_api.delete_namespaced_job(
                    name=job.metadata.name,
                    namespace=namespace,
                    body=client.V1DeleteOptions(
                        propagation_policy="Background",
                    ),
                )
                logger.info(f"Canceled Job: {namespace}/{job.metadata.name}")

            return True

        except ApiException as e:
            logger.error(f"Failed to cancel Job for step {step_id}: {e}")
            return False

    async def get_status(self, step_id: str, namespace: str) -> ExecutionStatus:
        """Get status of a Job."""
        await self._ensure_initialized()

        try:
            jobs = await self._batch_api.list_namespaced_job(
                namespace=namespace,
                label_selector=f"greenlang.io/step-id={step_id}",
            )

            if not jobs.items:
                return ExecutionStatus.PENDING

            job = jobs.items[0]
            status = job.status

            if status.succeeded and status.succeeded > 0:
                return ExecutionStatus.SUCCEEDED
            if status.failed and status.failed > 0:
                return ExecutionStatus.FAILED
            if status.active and status.active > 0:
                return ExecutionStatus.RUNNING

            return ExecutionStatus.SCHEDULED

        except ApiException as e:
            if e.status == 404:
                return ExecutionStatus.PENDING
            raise

    async def get_logs(
        self,
        step_id: str,
        namespace: str,
        tail_lines: Optional[int] = None,
    ) -> str:
        """Get logs from a Job's pod."""
        await self._ensure_initialized()

        try:
            # Find Job
            jobs = await self._batch_api.list_namespaced_job(
                namespace=namespace,
                label_selector=f"greenlang.io/step-id={step_id}",
            )

            if not jobs.items:
                return ""

            job_name = jobs.items[0].metadata.name

            # Find pod
            pods = await self._core_api.list_namespaced_pod(
                namespace=namespace,
                label_selector=f"job-name={job_name}",
            )

            if not pods.items:
                return ""

            pod_name = pods.items[0].metadata.name

            # Get logs
            logs = await self._core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container="agent",
                tail_lines=tail_lines,
            )

            return logs

        except ApiException as e:
            logger.warning(f"Failed to get logs for step {step_id}: {e}")
            return ""


def create_k8s_executor(
    artifact_store: ArtifactStore,
    in_cluster: bool = True,
    kubeconfig_path: Optional[str] = None,
    namespace: str = "greenlang",
) -> K8sExecutor:
    """
    Factory function to create K8s executor.

    Args:
        artifact_store: Artifact store for reading results
        in_cluster: Whether running inside K8s cluster
        kubeconfig_path: Path to kubeconfig (if not in_cluster)
        namespace: Default namespace for Jobs

    Returns:
        Configured K8sExecutor instance
    """
    config = K8sExecutorConfig(
        in_cluster=in_cluster,
        kubeconfig_path=kubeconfig_path,
        default_namespace=namespace,
    )
    return K8sExecutor(config, artifact_store)
