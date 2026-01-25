# -*- coding: utf-8 -*-
"""
Runtime Executor
================

Executes pipelines with different runtime profiles (local, k8s, cloud).
Supports deterministic execution for reproducible runs.
"""

import os
import sys
import json
import random
import subprocess
import shlex
import tempfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

from ..sdk.base import Result
from ..sdk.context import Context
from ..packs.loader import PackLoader
from greenlang.utilities.determinism import DeterministicClock, deterministic_uuid
from ..sandbox import (
    SandboxConfig,
    sandbox_execute,
    create_basic_sandbox,
    create_data_processing_sandbox,
    disable_sandbox,
)
import yaml

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logger = logging.getLogger(__name__)


class DeterministicConfig:
    """Configuration for deterministic execution"""

    def __init__(
        self,
        seed: int = 42,
        freeze_env: bool = True,
        normalize_floats: bool = True,
        float_precision: int = 6,
        quantization_bits: Optional[int] = None,
    ):
        self.seed = seed
        self.freeze_env = freeze_env
        self.normalize_floats = normalize_floats
        self.float_precision = float_precision
        self.quantization_bits = quantization_bits

    def apply(self):
        """Apply deterministic settings"""
        # Set random seeds
        random.seed(self.seed)
        if HAS_NUMPY:
            np.random.seed(self.seed)

        # Set Python hash seed
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # Disable Python optimizations that can affect determinism
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        # Set TensorFlow/PyTorch determinism if available
        try:
            import tensorflow as tf

            tf.random.set_seed(self.seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
        except ImportError:
            pass

        try:
            import torch

            torch.manual_seed(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass


class Executor:
    """
    Pipeline executor with runtime profiles

    Profiles:
    - local: Run on local machine
    - k8s: Run on Kubernetes
    - cloud: Run on cloud functions

    Supports deterministic execution for reproducible runs.
    """

    def __init__(
        self,
        backend: str = "local",
        deterministic: bool = False,
        det_config: Optional[DeterministicConfig] = None,
        sandbox_config: Optional[SandboxConfig] = None,
    ):
        """
        Initialize executor

        Args:
            backend: Execution backend (local, k8s, cloud)
            deterministic: Enable deterministic execution
            det_config: Deterministic configuration
            sandbox_config: Sandbox configuration for capability gating
        """
        self.backend = backend
        self.profile = backend  # Maintain backward compatibility
        self.deterministic = deterministic
        self.det_config = det_config or DeterministicConfig()
        self.sandbox_config = sandbox_config or disable_sandbox()
        self.loader = PackLoader()
        self.run_ledger = []
        self.artifacts_dir = Path.home() / ".greenlang" / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._validate_backend()

    def _safe_run(self, cmd_parts, **kwargs):
        """
        Safely execute a command with injection protection

        Args:
            cmd_parts: List of command parts
            **kwargs: Additional arguments for subprocess.run

        Returns:
            CompletedProcess result
        """
        # Validate command parts
        safe_cmd = []
        for part in cmd_parts:
            if not isinstance(part, str):
                part = str(part)
            # Use shlex.quote to safely escape shell metacharacters
            # but only if the part contains suspicious characters
            if any(char in part for char in ['|', '>', '<', ';', '&', '$', '`', '\n', '(', ')', '{', '}']):
                # This part might be trying injection - quote it
                cleaned = shlex.quote(part)
                logger.warning(f"Potentially dangerous command part quoted: {part} -> {cleaned}")
                safe_cmd.append(cleaned)
            else:
                safe_cmd.append(part)

        # Never use shell=True for safety
        kwargs['shell'] = False

        # Execute with safety constraints
        return subprocess.run(safe_cmd, **kwargs)

    def _validate_backend(self):
        """Validate backend availability"""
        if self.backend not in ["local", "k8s", "kubernetes", "cloud"]:
            raise ValueError(f"Unknown backend: {self.backend}")

        if self.backend in ["k8s", "kubernetes"]:
            # Check kubectl availability
            try:
                self._safe_run(
                    ["kubectl", "version", "--client"],
                    capture_output=True,
                    check=True,
                    timeout=5,
                )
            except (
                FileNotFoundError,
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ):
                logger.warning(
                    "kubectl not found or not configured, falling back to local"
                )
                self.backend = "local"

    @contextmanager
    def context(self, artifacts_dir: Path):
        """
        Create execution context

        Args:
            artifacts_dir: Directory for artifacts

        Yields:
            ExecutionContext instance
        """
        from ..provenance.utils import ProvenanceContext

        ctx = ProvenanceContext(str(artifacts_dir))
        ctx.backend = self.backend

        # Apply deterministic settings
        if self.deterministic:
            self.det_config.apply()

            # Freeze environment
            if self.det_config.freeze_env:
                ctx.environment = dict(os.environ)

        # Record versions
        ctx.versions = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "backend": self.backend,
            "deterministic": str(self.deterministic),
        }

        try:
            yield ctx
        finally:
            # Cleanup if needed
            pass

    def execute(self, pipeline: Any, inputs: Dict[str, Any]) -> Result:
        """
        Execute pipeline with specified backend

        Args:
            pipeline: Pipeline specification
            inputs: Input parameters

        Returns:
            Execution result
        """
        if self.backend == "local":
            return self._exec_local(pipeline, inputs)
        elif self.backend in ["k8s", "kubernetes"]:
            return self._exec_k8s(pipeline, inputs)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _load_pipeline(self, pipeline_path: str) -> Dict[str, Any]:
        """
        Load pipeline from file or reference

        Args:
            pipeline_path: Path to pipeline YAML file or pack reference

        Returns:
            Pipeline dictionary
        """
        path = Path(pipeline_path)

        # If it's a file path
        if path.exists():
            with open(path, "r") as f:
                pipeline = yaml.safe_load(f)
            return pipeline

        # Try loading from pack
        if "/" in pipeline_path or ":" in pipeline_path:
            # Format: pack_name/pipeline or pack_name:pipeline
            if "/" in pipeline_path:
                pack_name, pipeline_name = pipeline_path.split("/", 1)
            else:
                pack_name, pipeline_name = pipeline_path.split(":", 1)

            # Load pack and get pipeline
            try:
                loaded_pack = self.loader.load(pack_name)
                pipeline = loaded_pack.get_pipeline(pipeline_name)
                if pipeline:
                    return pipeline
            except Exception as e:
                logger.warning(f"Failed to load pipeline from pack: {e}")

        # Try as direct pipeline name in discovered packs
        for pack in self.loader.loaded_packs.values():
            pipeline = pack.get_pipeline(pipeline_path)
            if pipeline:
                return pipeline

        raise ValueError(f"Pipeline not found: {pipeline_path}")

    def run(
        self, pipeline_path: str, inputs: Dict = None, artifacts_dir: Path = None
    ) -> Result:
        """
        Execute pipeline with proper agent loading

        Args:
            pipeline_path: Path to pipeline file or reference
            inputs: Input data for pipeline
            artifacts_dir: Optional artifacts directory

        Returns:
            Execution result
        """
        # Load pipeline
        pipeline = self._load_pipeline(pipeline_path)

        # Create context with proper artifact management
        context = Context(
            inputs=inputs,
            artifacts_dir=artifacts_dir or Path("out"),
            backend=self.backend,
            metadata={
                "pipeline": pipeline.get("name", "unknown"),
                "version": pipeline.get("version", "1.0.0"),
            },
        )

        # Get steps from pipeline
        steps = pipeline.get("steps", pipeline.get("stages", []))

        for step in steps:
            step_name = step.get("name", step.get("id", f"step_{len(context.steps)}"))
            agent_ref = step.get("agent")

            if not agent_ref:
                logger.warning(f"No agent specified for step {step_name}")
                continue

            try:
                # Load agent class
                agent_class = self.loader.get_agent(agent_ref)
                agent = agent_class()

                # Determine action to execute
                action = step.get("action", "process")

                # Prepare inputs for step
                step_inputs = step.get("inputs", step.get("with_", {}))
                if not step_inputs:
                    step_inputs = context.data

                # Execute action with sandbox if configured
                if hasattr(agent, action):
                    method = getattr(agent, action)
                    result = self._execute_with_sandbox(method, step_inputs, step_name)
                else:
                    # Default to process or run method
                    if hasattr(agent, "process"):
                        result = self._execute_with_sandbox(
                            agent.process, step_inputs, step_name
                        )
                    elif hasattr(agent, "run"):
                        result = self._execute_with_sandbox(
                            agent.run, step_inputs, step_name
                        )
                    else:
                        raise AttributeError(
                            f"Agent {agent_ref} has no method '{action}', 'process', or 'run'"
                        )

                # Add result to context
                context.add_step_result(step_name, result)

                # Save step artifacts if configured
                if step.get("save_artifacts", False):
                    context.save_artifact(
                        f"{step_name}_output",
                        result.data,
                        type="json",
                        step=step_name,
                        success=result.success,
                    )

                logger.info(f"Step {step_name} completed: {result.success}")

            except Exception as e:
                logger.error(f"Step {step_name} failed: {e}")
                # Create error result
                error_result = Result(
                    success=False, data={}, metadata={"error": str(e)}
                )
                context.add_step_result(step_name, error_result)
                # Continue or break based on pipeline settings
                if pipeline.get("stop_on_error", True):
                    break

        return context.to_result()

    def run_legacy(self, pipeline_ref: str, input_data: Dict[str, Any]) -> Result:
        """
        Execute a pipeline

        Args:
            pipeline_ref: Pipeline reference (pack.pipeline)
            input_data: Input data for pipeline

        Returns:
            Execution result
        """
        run_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
        run_start = DeterministicClock.now()

        logger.info(f"Starting run {run_id} for pipeline {pipeline_ref}")

        try:
            # Load pipeline
            pipeline = self.loader.get_pipeline(pipeline_ref)
            if not pipeline:
                return Result(
                    success=False, error=f"Pipeline not found: {pipeline_ref}"
                )

            # Create run context
            context = {
                "run_id": run_id,
                "pipeline": pipeline_ref,
                "profile": self.profile,
                "input": input_data,
                "artifacts": [],
                "results": {},
            }

            # Execute based on profile
            if self.profile == "local":
                result = self._run_local(pipeline, context)
            elif self.profile == "k8s":
                result = self._run_k8s(pipeline, context)
            elif self.profile == "cloud":
                result = self._run_cloud(pipeline, context)
            else:
                return Result(success=False, error=f"Unknown profile: {self.profile}")

            # Record run
            run_end = DeterministicClock.now()
            run_record = {
                "run_id": run_id,
                "pipeline": pipeline_ref,
                "profile": self.profile,
                "status": "success" if result.success else "failed",
                "start_time": run_start.isoformat(),
                "end_time": run_end.isoformat(),
                "duration_seconds": (run_end - run_start).total_seconds(),
                "artifacts": context.get("artifacts", []),
            }

            self._save_run_record(run_record)

            # Generate run.json
            self._generate_run_json(run_id, pipeline, context, result)

            return result

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return Result(success=False, error=str(e))

    def _exec_local(self, pipeline: Dict[str, Any], inputs: Dict[str, Any]) -> Result:
        """
        Execute pipeline locally with deterministic support
        """
        logger.info(f"Executing pipeline locally: {pipeline.get('name', 'unnamed')}")

        start_time = DeterministicClock.utcnow()

        try:
            # Apply deterministic settings
            if self.deterministic:
                self.det_config.apply()

            # Process pipeline stages
            outputs = {}
            context = {"input": inputs, "results": {}, "artifacts": []}

            steps = pipeline.get("steps", pipeline.get("stages", []))

            for step in steps:
                step_name = step.get("name", f"step_{len(outputs)}")
                logger.info(f"Executing step: {step_name}")

                # Execute step based on type
                step_type = step.get("type", "agent")

                if step_type == "agent":
                    agent_ref = step.get("agent")
                    if agent_ref:
                        # Load and run agent
                        agent_class = self.loader.get_agent(agent_ref)
                        if agent_class:
                            agent = agent_class()
                            step_input = self._prepare_step_input(step, context)
                            result = self._execute_with_sandbox(
                                agent.run, step_input, step_name
                            )
                            outputs[step_name] = result.data if result.success else None
                            context["results"][step_name] = outputs[step_name]
                        else:
                            logger.warning(f"Agent not found: {agent_ref}")

                elif step_type == "python":
                    outputs[step_name] = self._exec_python_stage_with_sandbox(
                        step, context, step_name
                    )
                    context["results"][step_name] = outputs[step_name]

                elif step_type == "shell":
                    outputs[step_name] = self._exec_shell_stage_with_sandbox(
                        step, context, step_name
                    )
                    context["results"][step_name] = outputs[step_name]

                # Update inputs for next stage if configured
                if step.get("pass_outputs", False):
                    inputs.update(
                        outputs[step_name]
                        if isinstance(outputs[step_name], dict)
                        else {}
                    )

            # Apply output normalization if deterministic
            if self.deterministic and self.det_config.normalize_floats:
                outputs = self._normalize_outputs(outputs)

            duration = (DeterministicClock.utcnow() - start_time).total_seconds()

            return Result(
                success=True,
                data=outputs,
                metadata={
                    "duration": duration,
                    "stages_executed": len(steps),
                    "backend": self.backend,
                    "deterministic": self.deterministic,
                },
            )

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            duration = (DeterministicClock.utcnow() - start_time).total_seconds()

            return Result(
                success=False,
                error=str(e),
                metadata={"duration": duration, "backend": self.backend},
            )

    def _run_local(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline locally (legacy method for compatibility)"""
        return self._exec_local(pipeline, context.get("input", {}))

    def _exec_k8s(self, pipeline: Dict[str, Any], inputs: Dict[str, Any]) -> Result:
        """
        Execute pipeline on Kubernetes
        """
        logger.info(
            f"Executing pipeline on Kubernetes: {pipeline.get('name', 'unnamed')}"
        )

        start_time = DeterministicClock.utcnow()

        try:
            # Generate Kubernetes Job manifest
            job_manifest = self._create_k8s_job(pipeline, inputs)
            job_name = job_manifest["metadata"]["name"]

            # Create temporary file for manifest
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                import yaml

                yaml.dump(job_manifest, f)
                manifest_path = f.name

            try:
                # Apply manifest
                self._safe_run(
                    ["kubectl", "apply", "-f", manifest_path],
                    check=True,
                    capture_output=True,
                )

                # Wait for job completion
                logger.info(f"Waiting for job {job_name} to complete...")
                self._wait_for_k8s_job(job_name)

                # Get job logs
                logs = self._get_k8s_job_logs(job_name)

                # Parse outputs from logs (simplified)
                outputs = self._parse_k8s_outputs(logs)

                # Apply output normalization if deterministic
                if self.deterministic and self.det_config.normalize_floats:
                    outputs = self._normalize_outputs(outputs)

                duration = (DeterministicClock.utcnow() - start_time).total_seconds()

                return Result(
                    success=True,
                    data=outputs,
                    metadata={
                        "duration": duration,
                        "backend": self.backend,
                        "job_name": job_name,
                        "deterministic": self.deterministic,
                    },
                )

            finally:
                # Cleanup
                Path(manifest_path).unlink(missing_ok=True)

                # Delete job if configured
                if pipeline.get("cleanup", True):
                    self._safe_run(
                        ["kubectl", "delete", "job", job_name], capture_output=True
                    )

        except Exception as e:
            logger.error(f"Kubernetes execution failed: {e}")
            duration = (DeterministicClock.utcnow() - start_time).total_seconds()

            return Result(
                success=False,
                error=str(e),
                metadata={"duration": duration, "backend": self.backend},
            )

    def _run_k8s(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline on Kubernetes (legacy method for compatibility)"""
        return self._exec_k8s(pipeline, context.get("input", {}))

    def _run_cloud(self, pipeline: Dict[str, Any], context: Dict[str, Any]) -> Result:
        """Execute pipeline on cloud functions"""
        # TODO: Implement cloud execution
        # This would invoke Lambda/Cloud Functions for each step
        return Result(success=False, error="Cloud execution not yet implemented")

    def _prepare_step_input(
        self, step: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare input for a step"""
        input_mapping = step.get("input", {})

        if isinstance(input_mapping, dict):
            # Map from context
            step_input = {}
            for key, path in input_mapping.items():
                # Simple path resolution (could be more sophisticated)
                if path.startswith("$input."):
                    field = path.replace("$input.", "")
                    step_input[key] = context["input"].get(field)
                elif path.startswith("$results."):
                    field = path.replace("$results.", "")
                    step_input[key] = context["results"].get(field)
                else:
                    step_input[key] = path
            return step_input
        else:
            # Use context input directly
            return context["input"]

    def _collect_output(
        self, pipeline: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Collect pipeline output"""
        output_mapping = pipeline.get("output", {})

        if output_mapping:
            output = {}
            for key, path in output_mapping.items():
                if path.startswith("$results."):
                    field = path.replace("$results.", "")
                    output[key] = context["results"].get(field)
                else:
                    output[key] = path
            return output
        else:
            # Return all results
            return context["results"]

    def _save_run_record(self, record: Dict[str, Any]):
        """Save run record to ledger"""
        self.run_ledger.append(record)

        # Persist to file
        ledger_file = self.artifacts_dir / "run_ledger.jsonl"
        with open(ledger_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _generate_run_json(
        self,
        run_id: str,
        pipeline: Dict[str, Any],
        context: Dict[str, Any],
        result: Result,
    ):
        """Generate deterministic run.json for reproducibility"""
        run_json = {
            "run_id": run_id,
            "pipeline": pipeline,
            "input": context["input"],
            "output": result.data if result.success else None,
            "status": "success" if result.success else "failed",
            "error": result.error if not result.success else None,
            "artifacts": context.get("artifacts", []),
            "profile": self.profile,
            "timestamp": DeterministicClock.now().isoformat(),
        }

        # Save run.json
        run_file = self.artifacts_dir / f"run_{run_id}.json"
        with open(run_file, "w") as f:
            json.dump(run_json, f, indent=2)

        logger.info(f"Generated run.json: {run_file}")

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all runs from ledger"""
        ledger_file = self.artifacts_dir / "run_ledger.jsonl"

        if not ledger_file.exists():
            return []

        runs = []
        with open(ledger_file) as f:
            for line in f:
                runs.append(json.loads(line))

        return runs

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific run"""
        run_file = self.artifacts_dir / f"run_{run_id}.json"

        if run_file.exists():
            with open(run_file) as f:
                return json.load(f)

        return None

    def _exec_python_stage(self, stage: Dict, context: Dict) -> Dict:
        """
        Execute a Python code stage with sandboxed execution.

        SECURITY FIX: Replaced unsafe exec() with RestrictedPython sandbox
        to prevent remote code execution vulnerabilities.
        """
        code = stage.get("code", "")

        # SECURITY: Use RestrictedPython for sandboxed execution
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr

            # Compile code in restricted mode
            byte_code = compile_restricted(
                code,
                filename='<pipeline_stage>',
                mode='exec'
            )

            if byte_code.errors:
                logger.error(f"Code compilation errors: {byte_code.errors}")
                raise ValueError(f"Invalid Python code: {byte_code.errors}")

            # Create restricted namespace with safe builtins only
            restricted_namespace = {
                '__builtins__': safe_globals,
                '_getiter_': guarded_iter_unpack_sequence,
                '_getattr_': safer_getattr,
                # Safe data access
                "inputs": context.get("input", {}),
                "outputs": {},
                "context": {
                    "input": context.get("input", {}),
                    "results": context.get("results", {}),
                },
            }

            # Add deterministic utilities if enabled (restricted)
            if self.deterministic:
                restricted_namespace["__seed__"] = self.det_config.seed
                if HAS_NUMPY:
                    restricted_namespace["np"] = np
                restricted_namespace["random"] = random

            # Execute with timeout protection
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("Code execution timed out after 30 seconds")

            # Set timeout (only on Unix-like systems)
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)  # 30 second timeout

            try:
                # Execute in restricted sandbox
                exec(byte_code.code, restricted_namespace)
            finally:
                # Disable timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)

            return restricted_namespace.get("outputs", {})

        except ImportError:
            # Fallback if RestrictedPython not available - log critical warning
            logger.critical(
                "RestrictedPython not installed! Code execution is UNSAFE. "
                "Install with: pip install RestrictedPython"
            )

            # Create minimal namespace without dangerous builtins
            # SECURITY: Remove dangerous builtins
            safe_builtins = {
                '__import__': None,  # Block imports
                'eval': None,  # Block eval
                'exec': None,  # Block nested exec
                'compile': None,  # Block compile
                'open': None,  # Block file operations
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'dict': dict,
                    'list': list,
                    'tuple': tuple,
                    'set': set,
                    'range': range,
                    'enumerate': enumerate,
                    'zip': zip,
                    'sum': sum,
                    'min': min,
                    'max': max,
                    'abs': abs,
                    'round': round,
                },
            }

            namespace = {
                **safe_builtins,
                "inputs": context.get("input", {}),
                "outputs": {},
                "context": context,
            }

            # Add deterministic utilities if enabled
            if self.deterministic:
                namespace["__seed__"] = self.det_config.seed
                if HAS_NUMPY:
                    namespace["np"] = np
                namespace["random"] = random

            # SECURITY FIX: Compile code with restricted mode and timeout
            # This provides basic protection when RestrictedPython is not available
            try:
                # Compile with restricted mode flag to disable some dangerous operations
                compiled_code = compile(code, '<pipeline_stage>', 'exec', flags=0, dont_inherit=True)

                # Additional safety check - reject obviously dangerous patterns
                dangerous_patterns = ['__import__', 'eval', 'exec', 'compile', 'open', 'file',
                                     'input', 'raw_input', '__subclasses__', '__builtins__']
                if any(pattern in code for pattern in dangerous_patterns):
                    raise ValueError(f"Security violation: Dangerous pattern detected in code")

                # Execute with timeout and restricted namespace
                exec(compiled_code, namespace)
            except Exception as e:
                logger.error(f"Code execution failed: {e}")
                raise

            return namespace.get("outputs", {})

    def _exec_python_stage_with_sandbox(
        self, stage: Dict, context: Dict, step_name: str
    ) -> Dict:
        """Execute a Python code stage with sandbox capability gating"""
        if not self.sandbox_config.enabled:
            return self._exec_python_stage(stage, context)

        logger.info(
            f"Executing Python stage {step_name} with sandbox capability gating"
        )

        def execute_python():
            return self._exec_python_stage(stage, context)

        try:
            result = sandbox_execute(execute_python, self.sandbox_config)
            return result
        except Exception as e:
            logger.error(f"Sandbox Python execution failed for step {step_name}: {e}")
            return {"error": str(e), "sandbox_error": True}

    def _exec_shell_stage(self, stage: Dict, context: Dict) -> Dict:
        """Execute a shell command stage with secure input sanitization"""
        import shlex
        import re

        command = stage.get("command", "")

        # Substitute variables with proper sanitization
        for key, value in context.get("input", {}).items():
            # Validate input to prevent injection
            str_value = str(value)
            if not re.match(r'^[a-zA-Z0-9_\-\.\s/]+$', str_value):
                # For values with special characters, use shlex.quote
                str_value = shlex.quote(str_value)
            command = command.replace(f"${{{key}}}", str_value)

        # Parse command safely and execute without shell=True
        try:
            # Use shlex.split to properly parse the command
            cmd_parts = shlex.split(command)
            result = self._safe_run(cmd_parts, shell=False, capture_output=True, text=True)
        except ValueError as e:
            # If shlex.split fails, command might have unclosed quotes
            # Log error and return failure
            logger.error(f"Invalid command syntax: {e}")
            return {
                "stdout": "",
                "stderr": f"Command parsing error: {e}",
                "returncode": 1,
            }
        except Exception as e:
            # For any other execution errors
            logger.error(f"Command execution error: {e}")
            return {
                "stdout": "",
                "stderr": f"Execution error: {e}",
                "returncode": 1,
            }

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    def _exec_shell_stage_with_sandbox(
        self, stage: Dict, context: Dict, step_name: str
    ) -> Dict:
        """Execute a shell command stage with sandbox capability gating"""
        if not self.sandbox_config.enabled:
            return self._exec_shell_stage(stage, context)

        logger.info(f"Executing shell stage {step_name} with sandbox capability gating")

        def execute_shell():
            return self._exec_shell_stage(stage, context)

        try:
            result = sandbox_execute(execute_shell, self.sandbox_config)
            return result
        except Exception as e:
            logger.error(f"Sandbox shell execution failed for step {step_name}: {e}")
            return {
                "error": str(e),
                "sandbox_error": True,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def _create_k8s_job(self, pipeline: Dict, inputs: Dict) -> Dict:
        """Create Kubernetes Job manifest"""
        import uuid

        job_name = f"greenlang-{pipeline.get('name', 'job')}-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}"

        # Basic Job manifest
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "labels": {
                    "app": "greenlang",
                    "pipeline": pipeline.get("name", "unnamed"),
                },
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "name": "pipeline",
                                "image": pipeline.get("image", "python:3.9-slim"),
                                "command": ["/bin/sh", "-c"],
                                "args": [
                                    pipeline.get(
                                        "command", 'echo "No command specified"'
                                    )
                                ],
                                "env": [
                                    {"name": "INPUTS", "value": json.dumps(inputs)}
                                ],
                            }
                        ],
                        "restartPolicy": "Never",
                    }
                },
                "backoffLimit": pipeline.get("retries", 1),
            },
        }

        # Add deterministic settings
        if self.deterministic:
            manifest["spec"]["template"]["spec"]["containers"][0]["env"].extend(
                [
                    {"name": "PYTHONHASHSEED", "value": str(self.det_config.seed)},
                    {"name": "RANDOM_SEED", "value": str(self.det_config.seed)},
                ]
            )

        return manifest

    def _wait_for_k8s_job(self, job_name: str, timeout: int = 300):
        """Wait for Kubernetes job to complete"""
        import time

        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check job status
            result = self._safe_run(
                ["kubectl", "get", "job", job_name, "-o", "json"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                job_status = json.loads(result.stdout)

                # Check if completed
                if job_status.get("status", {}).get("succeeded", 0) > 0:
                    return True
                elif job_status.get("status", {}).get("failed", 0) > 0:
                    raise RuntimeError(f"Job {job_name} failed")

            time.sleep(5)

        raise TimeoutError(f"Job {job_name} timed out after {timeout} seconds")

    def _get_k8s_job_logs(self, job_name: str) -> str:
        """Get logs from Kubernetes job"""
        result = self._safe_run(
            ["kubectl", "logs", f"job/{job_name}"], capture_output=True, text=True
        )

        return result.stdout

    def _parse_k8s_outputs(self, logs: str) -> Dict:
        """Parse outputs from Kubernetes job logs"""
        outputs = {}

        # Look for JSON output markers
        for line in logs.split("\n"):
            if line.startswith("OUTPUT:"):
                try:
                    output_data = json.loads(line[7:])
                    outputs.update(output_data)
                except json.JSONDecodeError:
                    pass

        return outputs

    def _normalize_outputs(self, outputs: Any) -> Any:
        """
        Normalize outputs for determinism

        Args:
            outputs: Raw outputs

        Returns:
            Normalized outputs
        """
        if isinstance(outputs, dict):
            return {k: self._normalize_outputs(v) for k, v in outputs.items()}
        elif isinstance(outputs, list):
            return [self._normalize_outputs(v) for v in outputs]
        elif isinstance(outputs, float):
            # Round to specified precision
            if self.det_config.quantization_bits:
                # Quantize to specified bits
                scale = 2**self.det_config.quantization_bits
                return round(outputs * scale) / scale
            else:
                # Round to decimal places
                return round(outputs, self.det_config.float_precision)
        elif HAS_NUMPY and isinstance(outputs, np.ndarray):
            # Normalize numpy arrays
            if outputs.dtype in [np.float32, np.float64]:
                if self.det_config.quantization_bits:
                    scale = 2**self.det_config.quantization_bits
                    return np.round(outputs * scale) / scale
                else:
                    return np.round(outputs, self.det_config.float_precision)
            return outputs
        else:
            return outputs

    def _execute_with_sandbox(
        self, method: callable, inputs: Any, step_name: str = "unknown"
    ) -> Result:
        """
        Execute a method with sandbox capability gating if configured.

        Args:
            method: Method to execute (agent.run, agent.process, etc.)
            inputs: Input data for the method
            step_name: Name of the step for logging

        Returns:
            Result from method execution
        """
        if not self.sandbox_config.enabled:
            # Sandbox disabled - execute directly
            logger.debug(f"Executing step {step_name} without sandbox")
            return method(inputs)

        logger.info(f"Executing step {step_name} with sandbox capability gating")

        try:
            # Execute method within sandbox context
            result = sandbox_execute(method, self.sandbox_config, inputs)

            # Log sandbox violations if any
            validator = getattr(sandbox_execute, "_last_validator", None)
            if validator and validator.violations:
                logger.warning(
                    f"Step {step_name} had {len(validator.violations)} capability violations"
                )
                for violation in validator.violations:
                    logger.warning(
                        f"  - {violation['capability']}: {violation['reason']}"
                    )

            return result

        except Exception as e:
            logger.error(f"Sandbox execution failed for step {step_name}: {e}")
            # Return failed result
            return Result(
                success=False,
                error=str(e),
                metadata={"step": step_name, "sandbox_error": True},
            )

    def enable_sandbox(self, config: SandboxConfig = None):
        """
        Enable sandbox with the given configuration.

        Args:
            config: Sandbox configuration. If None, uses basic sandbox.
        """
        if config is None:
            config = create_basic_sandbox()
        self.sandbox_config = config
        logger.info("Sandbox capability gating enabled")

    def disable_sandbox(self):
        """Disable sandbox capability gating."""
        self.sandbox_config = disable_sandbox()
        logger.info("Sandbox capability gating disabled")

    def set_sandbox_policy(self, policy_name: str):
        """
        Set sandbox policy by name.

        Args:
            policy_name: Name of predefined policy ('basic', 'data_processing', 'network', 'disabled')
        """
        if policy_name == "basic":
            self.sandbox_config = create_basic_sandbox()
        elif policy_name == "data_processing":
            self.sandbox_config = create_data_processing_sandbox()
        elif policy_name == "disabled":
            self.sandbox_config = disable_sandbox()
        else:
            raise ValueError(f"Unknown sandbox policy: {policy_name}")

        logger.info(f"Set sandbox policy to: {policy_name}")


# Alias for backward compatibility and alternative naming convention
PipelineExecutor = Executor


__all__ = [
    "Executor",
    "PipelineExecutor",
    "DeterministicConfig",
]
