# -*- coding: utf-8 -*-
"""
Local Backend for GreenLang Pipeline Execution (Development)
"""

import json
import logging
import subprocess
import time
import tempfile
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import (
from greenlang.determinism import DeterministicClock
    Backend,
    ExecutionContext,
    Pipeline,
    PipelineStep,
    ExecutionStatus,
    ExecutionResult,
)

logger = logging.getLogger(__name__)


class LocalBackend(Backend):
    """Execute pipelines locally using subprocess (for development/testing)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Local backend

        Args:
            config: Backend configuration including:
                - working_dir: Working directory for execution
                - python_path: Python executable path
                - venv_path: Virtual environment path
                - shell: Whether to use shell execution
        """
        super().__init__(config)

        self.working_dir = Path(config.get("working_dir", tempfile.gettempdir()))
        self.python_path = config.get("python_path", sys.executable)
        self.venv_path = config.get("venv_path")
        self.use_shell = config.get("shell", False)

        # Track running processes
        self.processes = {}
        self.process_logs = {}
        self.process_outputs = {}

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"LocalBackend initialized with working_dir: {self.working_dir}")

    def execute(self, pipeline: Pipeline, context: ExecutionContext) -> ExecutionResult:
        """
        Execute pipeline locally

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

        start_time = DeterministicClock.utcnow()

        try:
            if len(pipeline.steps) == 1:
                # Single step
                result = self._run_step(pipeline.steps[0], pipeline, context)
            else:
                # Multiple steps
                result = self._run_workflow(pipeline, context)

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
                metadata={"processes": result.get("processes", [])},
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

    def _run_step(
        self, step: PipelineStep, pipeline: Pipeline, context: ExecutionContext
    ) -> Dict[str, Any]:
        """Run a single pipeline step"""

        step_id = f"{context.run_id}_{step.name}"

        # Prepare environment
        env = os.environ.copy()
        env.update(self.prepare_environment(context))
        env.update(step.env)
        env["GL_CONTEXT"] = json.dumps(context.to_dict())

        # Activate virtual environment if specified
        if self.venv_path:
            if os.name == "nt":  # Windows
                env["PATH"] = f"{self.venv_path}/Scripts;{env['PATH']}"
            else:  # Unix
                env["PATH"] = f"{self.venv_path}/bin:{env['PATH']}"

        # Prepare command
        if step.image and "python" in step.image:
            # Python command
            command = [self.python_path] + step.command + step.args
        else:
            # General command
            command = step.command + step.args

        # Create working directory for step
        step_dir = self.working_dir / pipeline.name / step.name
        step_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        stdout_file = step_dir / "stdout.log"
        stderr_file = step_dir / "stderr.log"

        try:
            logger.info(f"Starting process for step: {step.name}")

            # Start process
            with open(stdout_file, "w") as stdout, open(stderr_file, "w") as stderr:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=str(step_dir),
                    stdout=stdout,
                    stderr=stderr,
                    shell=self.use_shell,
                )

                # Track process
                self.processes[step_id] = {
                    "process": process,
                    "step": step.name,
                    "stdout_file": stdout_file,
                    "stderr_file": stderr_file,
                }

                # Wait for completion with timeout
                try:
                    return_code = process.wait(timeout=step.timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    return_code = -1
                    logger.warning(f"Process for step {step.name} timed out")

            # Read logs
            with open(stdout_file, "r") as f:
                stdout_logs = f.read().split("\n")
            with open(stderr_file, "r") as f:
                stderr_logs = f.read().split("\n")

            all_logs = stdout_logs + stderr_logs
            self.process_logs[step_id] = all_logs

            # Determine status
            if return_code == 0:
                status = ExecutionStatus.SUCCEEDED
                errors = []
            else:
                status = ExecutionStatus.FAILED
                errors = [f"Process exited with code {return_code}"] + stderr_logs

            # Read outputs if available
            outputs = self._read_step_outputs(step_dir)
            self.process_outputs[step_id] = outputs

            return {
                "status": status,
                "logs": all_logs,
                "errors": errors,
                "outputs": outputs,
                "processes": [step_id],
            }

        except Exception as e:
            logger.error(f"Failed to run process: {e}")
            return {
                "status": ExecutionStatus.FAILED,
                "logs": [],
                "errors": [str(e)],
                "outputs": {},
                "processes": [],
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
            "processes": [],
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

            # Run ready steps (sequentially for local backend to avoid resource conflicts)
            for step in ready_steps:
                # Pass previous step outputs as environment
                step_context = ExecutionContext(**context.to_dict())
                step_context.parameters.update({"step_outputs": step_outputs})

                result = self._run_step(step, pipeline, step_context)

                if result["status"] == ExecutionStatus.SUCCEEDED:
                    completed_steps.add(step.name)
                    results["logs"].extend(result.get("logs", []))
                    results["processes"].extend(result.get("processes", []))
                    step_outputs[step.name] = result.get("outputs", {})
                else:
                    results["status"] = ExecutionStatus.FAILED
                    results["errors"].extend(result.get("errors", []))
                    results["errors"].append(f"Step {step.name} failed")
                    return results

        results["status"] = ExecutionStatus.SUCCEEDED
        results["outputs"] = step_outputs
        return results

    def _read_step_outputs(self, step_dir: Path) -> Dict[str, Any]:
        """Read step outputs from files"""

        outputs = {}

        # Look for output files
        output_file = step_dir / "outputs.json"
        if output_file.exists():
            try:
                with open(output_file, "r") as f:
                    outputs = json.load(f)
            except Exception as e:
                logger.error(f"Failed to read outputs: {e}")

        # Look for result files
        for file in step_dir.glob("*.result"):
            try:
                with open(file, "r") as f:
                    outputs[file.stem] = f.read()
            except Exception as e:
                logger.error(f"Failed to read result file {file}: {e}")

        return outputs

    def get_status(self, run_id: str) -> ExecutionStatus:
        """Get execution status"""

        # Check all processes for this run
        for step_id, proc_info in self.processes.items():
            if step_id.startswith(run_id):
                process = proc_info["process"]

                poll = process.poll()
                if poll is None:
                    return ExecutionStatus.RUNNING
                elif poll == 0:
                    continue  # Check other steps
                else:
                    return ExecutionStatus.FAILED

        # All processes completed successfully
        return ExecutionStatus.SUCCEEDED

    def get_logs(self, run_id: str, step_name: Optional[str] = None) -> List[str]:
        """Get execution logs"""

        logs = []

        if step_name:
            step_id = f"{run_id}_{step_name}"
            if step_id in self.process_logs:
                return self.process_logs[step_id]
            elif step_id in self.processes:
                # Read from files
                proc_info = self.processes[step_id]
                try:
                    with open(proc_info["stdout_file"], "r") as f:
                        logs.extend(f.read().split("\n"))
                    with open(proc_info["stderr_file"], "r") as f:
                        logs.extend(f.read().split("\n"))
                except Exception as e:
                    logger.error(f"Failed to read logs: {e}")
        else:
            # Get all logs for run
            for step_id, step_logs in self.process_logs.items():
                if step_id.startswith(run_id):
                    logs.extend(step_logs)

        return logs

    def cancel(self, run_id: str) -> bool:
        """Cancel execution"""

        cancelled = False

        for step_id, proc_info in list(self.processes.items()):
            if step_id.startswith(run_id):
                process = proc_info["process"]

                try:
                    if process.poll() is None:
                        process.terminate()
                        time.sleep(2)
                        if process.poll() is None:
                            process.kill()

                        logger.info(f"Cancelled process for step: {proc_info['step']}")
                        cancelled = True
                except Exception as e:
                    logger.error(f"Failed to cancel process: {e}")

        return cancelled

    def cleanup(self, run_id: str) -> bool:
        """Cleanup execution resources"""

        cleaned = False

        # Remove from tracking
        for step_id in list(self.processes.keys()):
            if step_id.startswith(run_id):
                del self.processes[step_id]
                cleaned = True

        for step_id in list(self.process_logs.keys()):
            if step_id.startswith(run_id):
                del self.process_logs[step_id]

        for step_id in list(self.process_outputs.keys()):
            if step_id.startswith(run_id):
                del self.process_outputs[step_id]

        # Optionally clean up files
        if self.config.get("cleanup_files", False):
            for pipeline_dir in sorted(self.working_dir.iterdir()):
                if pipeline_dir.is_dir():
                    for step_dir in sorted(pipeline_dir.iterdir()):
                        if run_id in str(step_dir):
                            import shutil

                            shutil.rmtree(step_dir)
                            logger.info(f"Cleaned up directory: {step_dir}")

        return cleaned
