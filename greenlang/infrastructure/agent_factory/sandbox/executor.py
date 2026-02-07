"""
Sandbox Executor - Agent Factory Sandbox (INFRA-010)

Provides process-level isolation for agent execution. Each agent
invocation runs in a clean subprocess with filtered environment
variables, an isolated working directory, and bounded stdout/stderr
capture.

Classes:
    - SandboxResult: Outcome of a sandboxed execution.
    - SandboxExecutor: Core sandbox execution engine.

Example:
    >>> executor = SandboxExecutor()
    >>> result = await executor.execute(
    ...     agent_module="greenlang.agents.intake",
    ...     input_data={"file": "report.csv"},
    ...     limits=ResourceLimits(),
    ... )
    >>> print(result.exit_code, result.duration_ms)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum bytes of stdout/stderr to capture per invocation
_MAX_OUTPUT_BYTES: int = 10 * 1024 * 1024  # 10 MB

# Environment variable prefixes allowed in the sandbox (v1 scope).
# Only GL_* and OTEL_* are prefix-matched.  Everything else must be in
# the explicit allowlist below.
_SAFE_ENV_PREFIXES: tuple[str, ...] = (
    "GL_",
    "OTEL_",
)

# Explicit allowlist of individual environment variable names that are
# safe to pass through even though they do not match _SAFE_ENV_PREFIXES.
_ALLOWED_ENV_KEYS: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "TZ",
    "PYTHONPATH",
    "PYTHONHASHSEED",
    "VIRTUAL_ENV",
})

# Environment variables that are never passed through
_BLOCKED_ENV_KEYS: Set[str] = {
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "DATABASE_URL",
    "REDIS_URL",
    "POSTGRES_PASSWORD",
    "API_KEY",
    "SECRET_KEY",
    "PRIVATE_KEY",
}


# ---------------------------------------------------------------------------
# Resource Usage (from sandbox)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceUsage:
    """Resource usage reported by a sandboxed execution.

    Attributes:
        cpu_time_ms: CPU time consumed in milliseconds.
        memory_peak_mb: Peak memory usage in megabytes.
        disk_read_mb: Disk read in megabytes.
        disk_write_mb: Disk write in megabytes.
    """

    cpu_time_ms: float = 0.0
    memory_peak_mb: float = 0.0
    disk_read_mb: float = 0.0
    disk_write_mb: float = 0.0


# ---------------------------------------------------------------------------
# Sandbox Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of a sandboxed agent execution.

    Attributes:
        stdout: Captured standard output (truncated to MAX_OUTPUT_BYTES).
        stderr: Captured standard error (truncated to MAX_OUTPUT_BYTES).
        exit_code: Process exit code (0 = success).
        duration_ms: Wall-clock duration in milliseconds.
        resource_usage: Resource usage metrics from the execution.
        agent_module: The agent module that was executed.
        input_hash: SHA-256 hash of the serialised input data.
        output_hash: SHA-256 hash of the captured stdout.
        work_dir: Path to the temporary working directory (cleaned up).
        error_classification: Human-readable error category if exit_code != 0.
    """

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float
    resource_usage: ResourceUsage
    agent_module: str
    input_hash: str
    output_hash: str
    work_dir: str
    error_classification: str = ""


# ---------------------------------------------------------------------------
# Sandbox Executor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Process-level isolation engine for agent execution.

    Each invocation runs in a clean subprocess with:
    - Filtered environment variables (only safe prefixes).
    - An isolated temporary working directory.
    - Bounded stdout/stderr capture.
    - Configurable resource limits (via ResourceLimits).

    Attributes:
        python_executable: Path to the Python interpreter for subprocesses.
    """

    def __init__(
        self,
        python_executable: Optional[str] = None,
        allowed_env_prefixes: Optional[tuple[str, ...]] = None,
    ) -> None:
        """Initialize the sandbox executor.

        Args:
            python_executable: Path to the Python binary. Defaults to sys.executable.
            allowed_env_prefixes: Additional safe env var prefixes to pass through.
        """
        self.python_executable = python_executable or sys.executable
        self._env_prefixes = _SAFE_ENV_PREFIXES
        if allowed_env_prefixes:
            self._env_prefixes = self._env_prefixes + allowed_env_prefixes
        logger.info(
            "SandboxExecutor initialized (python=%s)", self.python_executable,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent_module: str,
        input_data: Dict[str, Any],
        limits: Optional[Any] = None,
        timeout_s: Optional[float] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> SandboxResult:
        """Execute an agent module in a sandboxed subprocess.

        Args:
            agent_module: Fully qualified Python module path of the agent.
            input_data: JSON-serialisable input data for the agent.
            limits: Optional ResourceLimits instance.
            timeout_s: Execution timeout in seconds. Defaults to 600s.
            extra_env: Additional environment variables to inject.

        Returns:
            SandboxResult with execution outcome.
        """
        effective_timeout = timeout_s or 600.0
        if limits is not None and hasattr(limits, "max_execution_seconds"):
            effective_timeout = min(effective_timeout, limits.max_execution_seconds)

        # Serialise input and compute hash
        input_json = json.dumps(input_data, sort_keys=True, default=str)
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()

        # Create isolated working directory
        work_dir = tempfile.mkdtemp(prefix="gl_sandbox_")
        input_path = os.path.join(work_dir, "input.json")

        try:
            # Write input data to the working directory
            with open(input_path, "w", encoding="utf-8") as f:
                f.write(input_json)

            # Build clean environment
            env = self._build_clean_env(extra_env)

            # Build the subprocess command
            runner_script = self._build_runner_script(agent_module, input_path)
            script_path = os.path.join(work_dir, "_runner.py")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(runner_script)

            cmd = [self.python_executable, script_path]

            logger.info(
                "SandboxExecutor: starting '%s' (timeout=%.0fs, work_dir=%s)",
                agent_module, effective_timeout, work_dir,
            )
            start_time = time.perf_counter()

            # Execute subprocess
            stdout, stderr, exit_code = await self._run_subprocess(
                cmd, env, work_dir, effective_timeout,
            )

            duration_ms = (time.perf_counter() - start_time) * 1000
            output_hash = hashlib.sha256(stdout.encode()).hexdigest()

            # Classify errors
            error_class = self._classify_error(exit_code, stderr)

            resource_usage = self._estimate_resource_usage(duration_ms)

            result = SandboxResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_ms=duration_ms,
                resource_usage=resource_usage,
                agent_module=agent_module,
                input_hash=input_hash,
                output_hash=output_hash,
                work_dir=work_dir,
                error_classification=error_class,
            )

            log_level = logging.INFO if exit_code == 0 else logging.ERROR
            logger.log(
                log_level,
                "SandboxExecutor: '%s' completed (exit=%d, %.1fms, class=%s)",
                agent_module, exit_code, duration_ms, error_class or "OK",
            )

            return result

        finally:
            # Clean up the working directory
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception as exc:
                logger.warning(
                    "SandboxExecutor: failed to clean work_dir %s: %s",
                    work_dir, exc,
                )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _build_clean_env(
        self,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """Build a filtered environment for the subprocess.

        Only passes through environment variables that match the prefix
        allowlist (``GL_*``, ``OTEL_*``) or the explicit key allowlist.
        Known sensitive keys are always blocked.

        Args:
            extra_env: Additional variables to inject.

        Returns:
            Filtered environment dictionary.
        """
        env: Dict[str, str] = {}
        for key, value in os.environ.items():
            upper = key.upper()
            if upper in _BLOCKED_ENV_KEYS:
                continue
            # Accept if the key matches a safe prefix
            if any(upper.startswith(p) for p in self._env_prefixes):
                env[key] = value
                continue
            # Accept if the key is in the explicit allowlist
            if upper in _ALLOWED_ENV_KEYS:
                env[key] = value

        # Ensure minimal PATH is always present
        if "PATH" not in env:
            env["PATH"] = os.environ.get("PATH", "")

        if extra_env:
            for key, value in extra_env.items():
                if key.upper() not in _BLOCKED_ENV_KEYS:
                    env[key] = value

        return env

    @staticmethod
    def _build_runner_script(agent_module: str, input_path: str) -> str:
        """Generate a Python runner script for the subprocess.

        Args:
            agent_module: Fully qualified module path.
            input_path: Path to the input JSON file.

        Returns:
            Python script content as a string.
        """
        return f'''\
import json
import sys
import importlib

def main():
    with open({input_path!r}, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    try:
        mod = importlib.import_module({agent_module!r})
    except ImportError as e:
        print(f"ERROR: Cannot import module: {{e}}", file=sys.stderr)
        sys.exit(2)

    # Look for a standard entry point
    entry = getattr(mod, "run", None) or getattr(mod, "main", None)
    if entry is None:
        print("ERROR: Module has no 'run' or 'main' function", file=sys.stderr)
        sys.exit(3)

    try:
        result = entry(input_data)
        if result is not None:
            print(json.dumps(result, default=str))
    except Exception as e:
        print(f"ERROR: Agent execution failed: {{e}}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    async def _run_subprocess(
        self,
        cmd: List[str],
        env: Dict[str, str],
        cwd: str,
        timeout_s: float,
    ) -> tuple[str, str, int]:
        """Run a subprocess with timeout enforcement.

        Sends SIGTERM first, waits 5s, then SIGKILL if still running.

        Args:
            cmd: Command and arguments.
            env: Environment variables.
            cwd: Working directory.
            timeout_s: Timeout in seconds.

        Returns:
            Tuple of (stdout, stderr, exit_code).
        """
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "SandboxExecutor: subprocess timed out after %.0fs, terminating",
                timeout_s,
            )
            # Graceful termination
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("SandboxExecutor: SIGTERM failed, sending SIGKILL")
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                pass

            stdout_bytes = b""
            stderr_bytes = b"TIMEOUT: execution exceeded limit"
            return (
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
                -1,
            )

        # Truncate output
        stdout_str = stdout_bytes[:_MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="replace"
        )
        stderr_str = stderr_bytes[:_MAX_OUTPUT_BYTES].decode(
            "utf-8", errors="replace"
        )

        return stdout_str, stderr_str, process.returncode or 0

    @staticmethod
    def _classify_error(exit_code: int, stderr: str) -> str:
        """Classify the error based on exit code and stderr content.

        Args:
            exit_code: Process exit code.
            stderr: Captured stderr.

        Returns:
            Human-readable error classification string.
        """
        if exit_code == 0:
            return ""
        if exit_code == -1:
            return "timeout"
        if exit_code == 2:
            return "import_error"
        if exit_code == 3:
            return "missing_entry_point"
        if "MemoryError" in stderr:
            return "out_of_memory"
        if "PermissionError" in stderr:
            return "permission_denied"
        if "FileNotFoundError" in stderr:
            return "file_not_found"
        return "execution_error"

    @staticmethod
    def _estimate_resource_usage(duration_ms: float) -> ResourceUsage:
        """Estimate resource usage from available process metrics.

        This is a best-effort estimation. For accurate metrics, use
        platform-specific profiling (e.g., cgroups on Linux).

        Args:
            duration_ms: Execution duration in milliseconds.

        Returns:
            Estimated ResourceUsage.
        """
        return ResourceUsage(
            cpu_time_ms=duration_ms,
            memory_peak_mb=0.0,
            disk_read_mb=0.0,
            disk_write_mb=0.0,
        )


__all__ = [
    "ResourceUsage",
    "SandboxExecutor",
    "SandboxResult",
]
