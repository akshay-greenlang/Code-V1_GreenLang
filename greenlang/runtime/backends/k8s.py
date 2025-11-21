# -*- coding: utf-8 -*-
"""
Kubernetes Backend for GreenLang Pipeline Execution
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    Backend,
    ExecutionContext,
    Pipeline,
    PipelineStep,
    ExecutionStatus,
    ExecutionResult,
)

logger = logging.getLogger(__name__)

# Try to import kubernetes client
try:
    from kubernetes import client
    from kubernetes import config as k8s_config
    from kubernetes.client.rest import ApiException

    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    logger.warning(
        "Kubernetes client not available. Install with: pip install kubernetes"
    )


class KubernetesBackend(Backend):
    """Execute pipelines as Kubernetes Jobs"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kubernetes backend

        Args:
            config: Backend configuration including:
                - namespace: Kubernetes namespace (default: "default")
                - image: Default executor image
                - service_account: Service account name
                - node_selector: Node selector labels
                - tolerations: Pod tolerations
                - image_pull_secrets: Image pull secrets
                - in_cluster: Whether running inside cluster
        """
        super().__init__(config)

        if not KUBERNETES_AVAILABLE:
            raise ImportError("Kubernetes client is required for KubernetesBackend")

        self.namespace = config.get("namespace", "default")
        self.default_image = config.get("image", "greenlang/executor:latest")
        self.service_account = config.get("service_account")
        self.node_selector = config.get("node_selector", {})
        self.tolerations = config.get("tolerations", [])
        self.image_pull_secrets = config.get("image_pull_secrets", [])

        # Initialize Kubernetes client
        if config.get("in_cluster", False):
            k8s_config.load_incluster_config()
        else:
            k8s_config.load_kube_config()

        self.batch_v1 = client.BatchV1Api()
        self.core_v1 = client.CoreV1Api()

        # Track running jobs
        self.jobs = {}

        logger.info(f"KubernetesBackend initialized for namespace: {self.namespace}")

    def execute(self, pipeline: Pipeline, context: ExecutionContext) -> ExecutionResult:
        """
        Execute pipeline as Kubernetes Job

        Args:
            pipeline: Pipeline to execute
            context: Execution context

        Returns:
            ExecutionResult
        """
        # Validate pipeline
        errors = self.validate_pipeline(pipeline)
        if errors:
            return ExecutionResult(
                run_id=context.run_id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.FAILED,
                start_time=DeterministicClock.utcnow(),
                errors=errors,
            )

        # Create job for each step (or orchestrate with workflow engine)
        start_time = DeterministicClock.utcnow()

        try:
            if len(pipeline.steps) == 1:
                # Single step - create one job
                job_name = self._create_job(pipeline.steps[0], pipeline, context)
                result = self._wait_for_job(job_name, context.run_id)
            else:
                # Multiple steps - create workflow
                result = self._create_workflow(pipeline, context)

            end_time = DeterministicClock.utcnow()
            duration = (end_time - start_time).total_seconds()

            return ExecutionResult(
                run_id=context.run_id,
                pipeline_name=pipeline.name,
                status=result["status"],
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                outputs=result.get("outputs", {}),
                logs=result.get("logs", []),
                errors=result.get("errors", []),
                metadata={"job_names": result.get("job_names", [])},
            )

        except Exception as e:
            logger.error(f"Failed to execute pipeline: {e}")
            return ExecutionResult(
                run_id=context.run_id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=DeterministicClock.utcnow(),
                errors=[str(e)],
            )

    def _create_job(
        self, step: PipelineStep, pipeline: Pipeline, context: ExecutionContext
    ) -> str:
        """Create Kubernetes Job for a pipeline step"""

        # Generate unique job name
        job_name = f"gl-{pipeline.name}-{step.name}-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}".lower()
        job_name = job_name.replace("_", "-")[:63]  # K8s name limit

        # Prepare environment variables
        env_vars = []
        env = self.prepare_environment(context)
        env.update(step.env)

        for key, value in env.items():
            env_vars.append(client.V1EnvVar(name=key, value=str(value)))

        # Add context as JSON
        env_vars.append(
            client.V1EnvVar(name="GL_CONTEXT", value=json.dumps(context.to_dict()))
        )

        # Add capabilities to environment
        capabilities = getattr(context, 'capabilities', {})
        env_vars.append(
            client.V1EnvVar(name="GL_CAPABILITIES", value=json.dumps(capabilities))
        )

        # Create security context based on capabilities
        security_context = self._create_security_context(capabilities)

        # Prepare container with security context
        container = client.V1Container(
            name="executor",
            image=step.image or self.default_image,
            command=step.command,
            args=step.args,
            env=env_vars,
            resources=self._create_resource_requirements(step),
            security_context=security_context,
        )

        # Add volume mounts if specified
        if context.volumes:
            container.volume_mounts = self._create_volume_mounts(context.volumes)

        # Create pod security context
        pod_security_context = self._create_pod_security_context(capabilities)

        # Prepare pod spec with security settings
        pod_spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never",
            service_account_name=self.service_account or pipeline.service_account,
            node_selector=self.node_selector,
            tolerations=self._create_tolerations(),
            security_context=pod_security_context,
        )

        # Create NetworkPolicy if network is restricted
        if not capabilities.get("net", {}).get("allow"):
            self._create_network_policy(job_name, pipeline.name, context)

        # Add image pull secrets
        if self.image_pull_secrets:
            pod_spec.image_pull_secrets = [
                client.V1LocalObjectReference(name=secret)
                for secret in self.image_pull_secrets
            ]

        # Add volumes if specified
        if context.volumes:
            pod_spec.volumes = self._create_volumes(context.volumes)

        # Create job
        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                name=job_name,
                namespace=self.namespace,
                labels={
                    "app": "greenlang",
                    "pipeline": pipeline.name,
                    "step": step.name,
                    "run-id": context.run_id,
                    **context.labels,
                },
                annotations={
                    "pipeline.greenlang.io/name": pipeline.name,
                    "pipeline.greenlang.io/step": step.name,
                    "pipeline.greenlang.io/run-id": context.run_id,
                    **context.annotations,
                },
            ),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={
                            "app": "greenlang",
                            "pipeline": pipeline.name,
                            "step": step.name,
                        }
                    ),
                    spec=pod_spec,
                ),
                backoff_limit=step.retry_count,
                active_deadline_seconds=step.timeout,
            ),
        )

        # Create the job
        response = self.batch_v1.create_namespaced_job(
            namespace=self.namespace, body=job
        )

        # Track the job
        self.jobs[context.run_id] = {
            "name": job_name,
            "pipeline": pipeline.name,
            "step": step.name,
            "created_at": DeterministicClock.utcnow(),
        }

        logger.info(f"Created Kubernetes job: {job_name}")
        return job_name

    def _create_workflow(
        self, pipeline: Pipeline, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Create workflow for multi-step pipeline"""

        # This is a simplified workflow implementation
        # In production, you might use Argo Workflows or Tekton

        job_names = []
        results = {
            "status": ExecutionStatus.RUNNING,
            "outputs": {},
            "logs": [],
            "errors": [],
            "job_names": job_names,
        }

        # Execute steps in dependency order
        completed_steps = set()

        while len(completed_steps) < len(pipeline.steps):
            # Find steps that can be executed
            ready_steps = []
            for step in pipeline.steps:
                if step.name not in completed_steps:
                    # Check if dependencies are satisfied
                    if all(dep in completed_steps for dep in step.depends_on):
                        ready_steps.append(step)

            if not ready_steps:
                # No steps ready - might be a dependency cycle
                results["status"] = ExecutionStatus.FAILED
                results["errors"].append(
                    "Dependency cycle detected or unsatisfiable dependencies"
                )
                break

            # Execute ready steps in parallel
            step_jobs = {}
            for step in ready_steps:
                job_name = self._create_job(step, pipeline, context)
                job_names.append(job_name)
                step_jobs[step.name] = job_name

            # Wait for steps to complete
            for step_name, job_name in step_jobs.items():
                step_result = self._wait_for_job(job_name, context.run_id)

                if step_result["status"] == ExecutionStatus.SUCCEEDED:
                    completed_steps.add(step_name)
                    results["logs"].extend(step_result.get("logs", []))
                    results["outputs"][step_name] = step_result.get("outputs", {})
                else:
                    results["status"] = ExecutionStatus.FAILED
                    results["errors"].extend(step_result.get("errors", []))
                    results["errors"].append(f"Step {step_name} failed")
                    return results

        results["status"] = ExecutionStatus.SUCCEEDED
        return results

    def _wait_for_job(
        self, job_name: str, run_id: str, timeout: int = 3600
    ) -> Dict[str, Any]:
        """Wait for job to complete"""

        start_time = time.time()
        result = {
            "status": ExecutionStatus.UNKNOWN,
            "logs": [],
            "outputs": {},
            "errors": [],
        }

        while time.time() - start_time < timeout:
            try:
                job = self.batch_v1.read_namespaced_job(
                    name=job_name, namespace=self.namespace
                )

                if job.status.succeeded:
                    result["status"] = ExecutionStatus.SUCCEEDED
                    result["logs"] = self._get_pod_logs(job_name)
                    break
                elif job.status.failed:
                    result["status"] = ExecutionStatus.FAILED
                    result["logs"] = self._get_pod_logs(job_name)
                    result["errors"].append(f"Job {job_name} failed")
                    break

                time.sleep(5)

            except ApiException as e:
                logger.error(f"Error checking job status: {e}")
                result["status"] = ExecutionStatus.FAILED
                result["errors"].append(str(e))
                break

        if result["status"] == ExecutionStatus.UNKNOWN:
            result["status"] = ExecutionStatus.FAILED
            result["errors"].append(f"Job {job_name} timed out after {timeout} seconds")

        return result

    def _get_pod_logs(self, job_name: str) -> List[str]:
        """Get logs from job pods"""

        logs = []

        try:
            # Find pods for the job
            pods = self.core_v1.list_namespaced_pod(
                namespace=self.namespace, label_selector=f"job-name={job_name}"
            )

            for pod in pods.items:
                try:
                    pod_logs = self.core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name, namespace=self.namespace
                    )
                    logs.extend(pod_logs.split("\n"))
                except ApiException as e:
                    logger.error(f"Failed to get logs for pod {pod.metadata.name}: {e}")

        except ApiException as e:
            logger.error(f"Failed to list pods for job {job_name}: {e}")

        return logs

    def get_status(self, run_id: str) -> ExecutionStatus:
        """Get execution status"""

        if run_id not in self.jobs:
            return ExecutionStatus.UNKNOWN

        job_info = self.jobs[run_id]

        try:
            job = self.batch_v1.read_namespaced_job(
                name=job_info["name"], namespace=self.namespace
            )

            if job.status.succeeded:
                return ExecutionStatus.SUCCEEDED
            elif job.status.failed:
                return ExecutionStatus.FAILED
            elif job.status.active:
                return ExecutionStatus.RUNNING
            else:
                return ExecutionStatus.PENDING

        except ApiException as e:
            logger.error(f"Failed to get job status: {e}")
            return ExecutionStatus.UNKNOWN

    def get_logs(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """Get execution logs"""

        if run_id not in self.jobs:
            return []

        job_info = self.jobs[run_id]
        return self._get_pod_logs(job_info["name"])

    def cancel(self, run_id: str) -> bool:
        """Cancel execution"""

        if run_id not in self.jobs:
            return False

        job_info = self.jobs[run_id]

        try:
            # Delete the job
            self.batch_v1.delete_namespaced_job(
                name=job_info["name"],
                namespace=self.namespace,
                propagation_policy="Background",
            )

            logger.info(f"Cancelled job: {job_info['name']}")
            return True

        except ApiException as e:
            logger.error(f"Failed to cancel job: {e}")
            return False

    def cleanup(self, run_id: str) -> bool:
        """Cleanup execution resources"""

        if run_id not in self.jobs:
            return False

        job_info = self.jobs[run_id]

        try:
            # Delete the job and its pods
            self.batch_v1.delete_namespaced_job(
                name=job_info["name"],
                namespace=self.namespace,
                propagation_policy="Foreground",
            )

            # Remove from tracking
            del self.jobs[run_id]

            logger.info(f"Cleaned up job: {job_info['name']}")
            return True

        except ApiException as e:
            logger.error(f"Failed to cleanup job: {e}")
            return False

    def _create_resource_requirements(
        self, step: PipelineStep
    ) -> client.V1ResourceRequirements:
        """Create resource requirements for container"""

        if not step.resources:
            return None

        requirements = client.V1ResourceRequirements()

        # Set requests
        requests = {}
        if step.resources.cpu:
            requests["cpu"] = step.resources.cpu
        if step.resources.memory:
            requests["memory"] = step.resources.memory
        if step.resources.ephemeral_storage:
            requests["ephemeral-storage"] = step.resources.ephemeral_storage
        if step.resources.gpu:
            requests["nvidia.com/gpu"] = step.resources.gpu

        if requests:
            requirements.requests = requests

        # Set limits (same as requests for predictable behavior)
        requirements.limits = requests.copy()

        return requirements

    def _create_tolerations(self) -> List[client.V1Toleration]:
        """Create pod tolerations"""

        tolerations = []

        for tol in self.tolerations:
            toleration = client.V1Toleration(
                key=tol.get("key"),
                operator=tol.get("operator", "Equal"),
                value=tol.get("value"),
                effect=tol.get("effect"),
            )
            tolerations.append(toleration)

        return tolerations

    def _create_volumes(
        self, volume_specs: List[Dict[str, Any]]
    ) -> List[client.V1Volume]:
        """Create pod volumes"""

        volumes = []

        for spec in volume_specs:
            volume = client.V1Volume(name=spec["name"])

            if spec["type"] == "configMap":
                volume.config_map = client.V1ConfigMapVolumeSource(name=spec["source"])
            elif spec["type"] == "secret":
                volume.secret = client.V1SecretVolumeSource(secret_name=spec["source"])
            elif spec["type"] == "persistentVolumeClaim":
                volume.persistent_volume_claim = (
                    client.V1PersistentVolumeClaimVolumeSource(
                        claim_name=spec["source"]
                    )
                )
            elif spec["type"] == "emptyDir":
                volume.empty_dir = client.V1EmptyDirVolumeSource(
                    medium=spec.get("medium", ""), size_limit=spec.get("sizeLimit")
                )

            volumes.append(volume)

        return volumes

    def _create_volume_mounts(
        self, volume_specs: List[Dict[str, Any]]
    ) -> List[client.V1VolumeMount]:
        """Create container volume mounts"""

        mounts = []

        for spec in volume_specs:
            mount = client.V1VolumeMount(
                name=spec["name"],
                mount_path=spec["mountPath"],
                read_only=spec.get("readOnly", False),
            )
            mounts.append(mount)

        return mounts

    def _create_security_context(
        self, capabilities: Dict[str, Any]
    ) -> client.V1SecurityContext:
        """
        Create container security context based on capabilities

        Args:
            capabilities: Pack capabilities

        Returns:
            V1SecurityContext for container
        """
        security_context = client.V1SecurityContext(
            run_as_non_root=True,
            allow_privilege_escalation=False,
            read_only_root_filesystem=self._should_use_readonly_root(capabilities),
            capabilities=client.V1Capabilities(
                drop=["ALL"],  # Drop all capabilities by default
                add=self._get_required_capabilities(capabilities)
            )
        )

        return security_context

    def _create_pod_security_context(
        self, capabilities: Dict[str, Any]
    ) -> client.V1PodSecurityContext:
        """
        Create pod security context

        Args:
            capabilities: Pack capabilities

        Returns:
            V1PodSecurityContext
        """
        pod_security_context = client.V1PodSecurityContext(
            run_as_non_root=True,
            run_as_user=1000,  # Non-root user ID
            run_as_group=1000,
            fs_group=1000,
            seccomp_profile=client.V1SeccompProfile(
                type="RuntimeDefault"  # Use runtime's default seccomp profile
            )
        )

        return pod_security_context

    def _create_network_policy(
        self, job_name: str, pipeline_name: str, context: ExecutionContext
    ):
        """
        Create NetworkPolicy to deny network for job

        Args:
            job_name: Job name
            pipeline_name: Pipeline name
            context: Execution context
        """
        try:
            from kubernetes import client as k8s_client
            networking_v1 = k8s_client.NetworkingV1Api()

            # Create deny-all network policy
            network_policy = k8s_client.V1NetworkPolicy(
                metadata=k8s_client.V1ObjectMeta(
                    name=f"{job_name}-netpol",
                    namespace=self.namespace,
                    labels={
                        "app": "greenlang",
                        "pipeline": pipeline_name,
                        "run-id": context.run_id,
                    }
                ),
                spec=k8s_client.V1NetworkPolicySpec(
                    pod_selector=k8s_client.V1LabelSelector(
                        match_labels={
                            "app": "greenlang",
                            "pipeline": pipeline_name,
                        }
                    ),
                    policy_types=["Ingress", "Egress"],
                    # Empty ingress/egress lists = deny all
                    ingress=[],
                    egress=[]
                )
            )

            networking_v1.create_namespaced_network_policy(
                namespace=self.namespace,
                body=network_policy
            )

            logger.info(f"Created NetworkPolicy to deny network for job: {job_name}")

        except Exception as e:
            logger.warning(f"Failed to create NetworkPolicy: {e}")

    def _get_required_capabilities(
        self, capabilities: Dict[str, Any]
    ) -> List[str]:
        """
        Get Linux capabilities required based on pack capabilities

        Args:
            capabilities: Pack capabilities

        Returns:
            List of Linux capabilities to add
        """
        linux_caps = []

        # Check subprocess capability
        if capabilities.get("subprocess", {}).get("allow"):
            linux_caps.append("SYS_PTRACE")  # For subprocess debugging

        # Check network capability
        if capabilities.get("net", {}).get("allow"):
            linux_caps.append("NET_RAW")  # For raw sockets if needed

        return linux_caps if linux_caps else []

    def _should_use_readonly_root(self, capabilities: Dict[str, Any]) -> bool:
        """
        Determine if container should use read-only root filesystem

        Args:
            capabilities: Pack capabilities

        Returns:
            True if read-only root should be used
        """
        # Check filesystem capability
        fs_cap = capabilities.get("fs", {})
        if isinstance(fs_cap, dict) and fs_cap.get("allow"):
            # Check if root write is needed
            write_paths = fs_cap.get("write_paths", [])
            # If any write path is not under /tmp, need writable root
            if any(not path.startswith("/tmp") for path in write_paths if path):
                return False
        # Default to read-only root for security
        return True
