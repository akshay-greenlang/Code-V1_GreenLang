"""
Docker Backend for GreenLang Pipeline Execution
"""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import tempfile

from .base import (
    Backend,
    ExecutionContext,
    Pipeline,
    PipelineStep,
    ExecutionStatus,
    ExecutionResult,
)

logger = logging.getLogger(__name__)

# Try to import docker client
try:
    import docker
    from docker.errors import ContainerError, ImageNotFound

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    logger.warning("Docker client not available. Install with: pip install docker")


class DockerBackend(Backend):
    """Execute pipelines as Docker containers"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Docker backend

        Args:
            config: Backend configuration including:
                - docker_host: Docker daemon URL
                - network: Docker network name
                - volumes: Volume mappings
                - registry: Docker registry URL
                - auth_config: Registry authentication
        """
        super().__init__(config)

        if not DOCKER_AVAILABLE:
            raise ImportError("Docker client is required for DockerBackend")

        # Initialize Docker client
        docker_host = config.get("docker_host")
        if docker_host:
            self.client = docker.DockerClient(base_url=docker_host)
        else:
            self.client = docker.from_env()

        self.network = config.get("network", "bridge")
        self.volumes = config.get("volumes", {})
        self.registry = config.get("registry")
        self.auth_config = config.get("auth_config")

        # Track running containers
        self.containers = {}
        self.container_logs = {}

        logger.info(f"DockerBackend initialized with network: {self.network}")

    def execute(self, pipeline: Pipeline, context: ExecutionContext) -> ExecutionResult:
        """
        Execute pipeline as Docker containers

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
                start_time=datetime.utcnow(),
                errors=errors,
            )

        start_time = datetime.utcnow()

        try:
            if len(pipeline.steps) == 1:
                # Single step
                result = self._run_step(pipeline.steps[0], pipeline, context)
            else:
                # Multiple steps
                result = self._run_workflow(pipeline, context)

            end_time = datetime.utcnow()
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
                metadata={"containers": result.get("containers", [])},
            )

        except Exception as e:
            logger.error(f"Failed to execute pipeline: {e}")
            return ExecutionResult(
                run_id=context.run_id,
                pipeline_name=pipeline.name,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.utcnow(),
                errors=[str(e)],
            )

    def _run_step(
        self, step: PipelineStep, pipeline: Pipeline, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run a single pipeline step"""

        container_name = f"gl-{pipeline.name}-{step.name}-{context.run_id[:8]}".lower()

        # Prepare environment
        environment = self.prepare_environment(context)
        environment.update(step.env)
        environment["GL_CONTEXT"] = json.dumps(context.to_dict())

        # Prepare volumes
        volumes = self._prepare_volumes(context)

        # Pull image if needed
        image = step.image or "greenlang/executor:latest"
        try:
            self.client.images.get(image)
        except ImageNotFound:
            logger.info(f"Pulling image: {image}")
            self.client.images.pull(image, auth_config=self.auth_config)

        # Prepare command
        command = step.command + step.args

        try:
            # Run container
            logger.info(f"Starting container: {container_name}")
            container = self.client.containers.run(
                image=image,
                command=command,
                name=container_name,
                environment=environment,
                volumes=volumes,
                network=self.network,
                detach=True,
                remove=False,
                labels={
                    "greenlang.pipeline": pipeline.name,
                    "greenlang.step": step.name,
                    "greenlang.run_id": context.run_id,
                },
            )

            # Track container
            self.containers[context.run_id] = {
                "container": container,
                "name": container_name,
                "step": step.name,
            }

            # Wait for completion with timeout
            exit_code = self._wait_for_container(container, step.timeout)

            # Get logs
            logs = container.logs(timestamps=False).decode("utf-8").split("\n")
            self.container_logs[context.run_id] = logs

            # Determine status
            if exit_code == 0:
                status = ExecutionStatus.SUCCEEDED
                errors = []
            else:
                status = ExecutionStatus.FAILED
                errors = [f"Container exited with code {exit_code}"]

            # Cleanup
            container.remove(force=True)

            return {
                "status": status,
                "logs": logs,
                "errors": errors,
                "outputs": {"exit_code": exit_code},
                "containers": [container_name],
            }

        except ContainerError as e:
            logger.error(f"Container error: {e}")
            return {
                "status": ExecutionStatus.FAILED,
                "logs": [e.stderr.decode("utf-8")] if e.stderr else [],
                "errors": [str(e)],
                "outputs": {},
                "containers": [container_name],
            }
        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            return {
                "status": ExecutionStatus.FAILED,
                "logs": [],
                "errors": [str(e)],
                "outputs": {},
                "containers": [],
            }

    def _run_workflow(
        self, pipeline: Pipeline, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run multi-step workflow"""

        results = {
            "status": ExecutionStatus.RUNNING,
            "logs": [],
            "errors": [],
            "outputs": {},
            "containers": [],
        }

        completed_steps = set()
        step_outputs = {}

        while len(completed_steps) < len(pipeline.steps):
            # Find ready steps
            ready_steps = []
            for step in pipeline.steps:
                if step.name not in completed_steps:
                    if all(dep in completed_steps for dep in step.depends_on):
                        ready_steps.append(step)

            if not ready_steps:
                results["status"] = ExecutionStatus.FAILED
                results["errors"].append(
                    "Dependency cycle or unsatisfiable dependencies"
                )
                break

            # Run ready steps in parallel
            threads = []
            step_results = {}

            for step in ready_steps:

                def run_step_thread(s, r):
                    r[s.name] = self._run_step(s, pipeline, context)

                thread = threading.Thread(
                    target=run_step_thread, args=(step, step_results)
                )
                thread.start()
                threads.append(thread)

            # Wait for all threads
            for thread in threads:
                thread.join()

            # Process results
            for step_name, result in step_results.items():
                if result["status"] == ExecutionStatus.SUCCEEDED:
                    completed_steps.add(step_name)
                    results["logs"].extend(result.get("logs", []))
                    results["containers"].extend(result.get("containers", []))
                    step_outputs[step_name] = result.get("outputs", {})
                else:
                    results["status"] = ExecutionStatus.FAILED
                    results["errors"].extend(result.get("errors", []))
                    results["errors"].append(f"Step {step_name} failed")
                    return results

        results["status"] = ExecutionStatus.SUCCEEDED
        results["outputs"] = step_outputs
        return results

    def _wait_for_container(self, container, timeout: int) -> int:
        """Wait for container to complete"""

        start_time = time.time()

        while time.time() - start_time < timeout:
            container.reload()

            if container.status in ["exited", "dead"]:
                return container.attrs["State"]["ExitCode"]

            time.sleep(1)

        # Timeout - kill container
        logger.warning(f"Container {container.name} timed out after {timeout}s")
        container.kill()
        container.reload()

        return -1

    def _prepare_volumes(self, context: ExecutionContext) -> Dict[str, Dict[str, str]]:
        """Prepare volume mappings"""

        volumes = {}

        # Add configured volumes
        for host_path, container_path in self.volumes.items():
            volumes[host_path] = {"bind": container_path, "mode": "rw"}

        # Add context volumes
        for vol in context.volumes:
            if vol["type"] == "bind":
                volumes[vol["source"]] = {
                    "bind": vol["target"],
                    "mode": vol.get("mode", "rw"),
                }

        # Create temp directory for outputs
        temp_dir = tempfile.mkdtemp(prefix="greenlang-")
        volumes[temp_dir] = {"bind": "/tmp/outputs", "mode": "rw"}

        return volumes

    def get_status(self, run_id: str) -> ExecutionStatus:
        """Get execution status"""

        if run_id not in self.containers:
            return ExecutionStatus.UNKNOWN

        container_info = self.containers[run_id]
        container = container_info["container"]

        try:
            container.reload()

            if container.status == "running":
                return ExecutionStatus.RUNNING
            elif container.status == "exited":
                exit_code = container.attrs["State"]["ExitCode"]
                if exit_code == 0:
                    return ExecutionStatus.SUCCEEDED
                else:
                    return ExecutionStatus.FAILED
            else:
                return ExecutionStatus.UNKNOWN

        except Exception as e:
            logger.error(f"Failed to get container status: {e}")
            return ExecutionStatus.UNKNOWN

    def get_logs(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """Get execution logs"""

        if run_id in self.container_logs:
            return self.container_logs[run_id]

        if run_id in self.containers:
            container_info = self.containers[run_id]
            container = container_info["container"]

            try:
                logs = container.logs(timestamps=False).decode("utf-8").split("\n")
                return logs
            except Exception as e:
                logger.error(f"Failed to get container logs: {e}")
                return []

        return []

    def cancel(self, run_id: str) -> bool:
        """Cancel execution"""

        if run_id not in self.containers:
            return False

        container_info = self.containers[run_id]
        container = container_info["container"]

        try:
            container.stop(timeout=10)
            container.remove(force=True)
            logger.info(f"Cancelled container: {container_info['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel container: {e}")
            return False

    def cleanup(self, run_id: str) -> bool:
        """Cleanup execution resources"""

        if run_id not in self.containers:
            return False

        container_info = self.containers[run_id]
        container = container_info["container"]

        try:
            container.remove(force=True)
            del self.containers[run_id]

            if run_id in self.container_logs:
                del self.container_logs[run_id]

            logger.info(f"Cleaned up container: {container_info['name']}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup container: {e}")
            return False
