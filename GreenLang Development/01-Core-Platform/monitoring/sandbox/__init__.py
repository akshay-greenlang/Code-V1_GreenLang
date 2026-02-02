# -*- coding: utf-8 -*-
"""
Sandbox System
==============

Provides capability-gated sandboxed execution for GreenLang packs.
Implements a default-deny security model where all operations require
explicit capability grants.

Usage:
    # Configure sandbox
    config = SandboxConfig(
        policy=DATA_PROCESSING_POLICY,
        enabled=True
    )

    # Execute code in sandbox
    result = sandbox_execute(my_function, config, arg1, arg2)

    # Use decorator for automatic capability checking
    @capability_check(Capability.FILE_READ, path_context='filepath')
    def read_file(filepath):
        with open(filepath, 'r') as f:
            return f.read()
"""

import os
import sys
import logging
import threading
import functools
import subprocess
import tempfile
from typing import Any, Callable, Optional, Dict, Union, TypeVar, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .capabilities import (
    Capability,
    CapabilityPolicy,
    CapabilityValidator,
    DEFAULT_POLICY,
    BASIC_COMPUTE_POLICY,
    DATA_PROCESSING_POLICY,
    NETWORK_CLIENT_POLICY,
    FULL_ACCESS_POLICY,
)

logger = logging.getLogger(__name__)

# Type variable for return types
T = TypeVar("T")

# Thread-local storage for sandbox context
_sandbox_context = threading.local()


@dataclass
class SandboxConfig:
    """
    Configuration for sandboxed execution.
    """

    # Enable/disable sandboxing (if False, runs without restrictions)
    enabled: bool = True

    # Capability policy to enforce
    policy: CapabilityPolicy = field(default_factory=lambda: DEFAULT_POLICY)

    # Isolation settings
    use_subprocess: bool = False  # Run in separate process
    subprocess_timeout: Optional[int] = None  # Subprocess timeout in seconds

    # Resource limits (applied when use_subprocess=True)
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[int] = None

    # Temporary directory for sandboxed operations
    temp_dir: Optional[str] = None
    cleanup_temp: bool = True

    # Logging and monitoring
    log_violations: bool = True
    raise_on_violation: bool = True

    def __post_init__(self):
        """Initialize sandbox configuration"""
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="greenlang_sandbox_")

        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)


class SandboxViolationError(Exception):
    """Raised when a capability violation occurs in strict mode"""

    def __init__(
        self,
        capability: Capability,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.capability = capability
        self.reason = reason
        self.context = context or {}
        super().__init__(f"Sandbox violation: {capability.name} - {reason}")


def get_current_validator() -> Optional[CapabilityValidator]:
    """Get the capability validator for the current sandbox context"""
    return getattr(_sandbox_context, "validator", None)


def get_current_config() -> Optional[SandboxConfig]:
    """Get the sandbox configuration for the current context"""
    return getattr(_sandbox_context, "config", None)


@contextmanager
def sandbox_context(config: SandboxConfig):
    """
    Context manager for sandboxed execution.

    Args:
        config: Sandbox configuration

    Yields:
        CapabilityValidator instance
    """
    if not config.enabled:
        # Sandbox disabled - no restrictions
        yield None
        return

    # Create validator for this context
    validator = CapabilityValidator(config.policy)

    # Store in thread-local storage
    old_validator = getattr(_sandbox_context, "validator", None)
    old_config = getattr(_sandbox_context, "config", None)

    _sandbox_context.validator = validator
    _sandbox_context.config = config

    try:
        yield validator
    finally:
        # Restore previous context
        _sandbox_context.validator = old_validator
        _sandbox_context.config = old_config

        # Cleanup temporary directory if requested
        if config.cleanup_temp and config.temp_dir:
            try:
                import shutil

                shutil.rmtree(config.temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup temp directory {config.temp_dir}: {e}"
                )


def capability_check(*capabilities: Capability, **kwargs):
    """
    Decorator that checks capabilities before function execution.

    Args:
        *capabilities: Required capabilities
        **kwargs: Context extraction configuration
            - path_context: Parameter name containing file path
            - host_context: Parameter name containing hostname
            - env_context: Parameter name containing environment variable name

    Example:
        @capability_check(Capability.FILE_READ, path_context='filename')
        def read_file(filename: str) -> str:
            with open(filename, 'r') as f:
                return f.read()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **func_kwargs) -> T:
            validator = get_current_validator()
            config = get_current_config()

            # If no sandbox context, either allow or deny based on configuration
            if validator is None:
                if config is None or not config.enabled:
                    # No sandbox - allow execution
                    return func(*args, **func_kwargs)
                else:
                    # Sandbox should be active but no validator found
                    raise SandboxViolationError(
                        capabilities[0] if capabilities else Capability.EXEC_SUBPROCESS,
                        "No sandbox context found",
                    )

            # Check each required capability
            for capability in capabilities:
                # Build context for capability check
                context = {}

                # Extract context from function parameters
                if "path_context" in kwargs:
                    param_name = kwargs["path_context"]
                    if param_name in func_kwargs:
                        context["path"] = func_kwargs[param_name]
                    elif len(args) > 0 and hasattr(func, "__code__"):
                        # Try to match positional arguments
                        param_names = func.__code__.co_varnames[
                            : func.__code__.co_argcount
                        ]
                        if param_name in param_names:
                            param_index = param_names.index(param_name)
                            if param_index < len(args):
                                context["path"] = args[param_index]

                if "host_context" in kwargs:
                    param_name = kwargs["host_context"]
                    if param_name in func_kwargs:
                        context["host"] = func_kwargs[param_name]

                if "env_context" in kwargs:
                    param_name = kwargs["env_context"]
                    if param_name in func_kwargs:
                        context["var_name"] = func_kwargs[param_name]

                # Check the capability
                if not validator.check_capability(capability, context):
                    if config and config.raise_on_violation:
                        raise SandboxViolationError(
                            capability, f"Capability {capability.name} denied", context
                        )
                    else:
                        logger.warning(
                            f"Capability {capability.name} denied but continuing execution"
                        )
                        break

            # All capabilities approved - execute function
            return func(*args, **func_kwargs)

        return wrapper

    return decorator


def sandbox_execute(
    func: Callable[..., T], config: SandboxConfig, *args, **kwargs
) -> T:
    """
    Execute a function within a sandbox with the given configuration.

    Args:
        func: Function to execute
        config: Sandbox configuration
        *args: Positional arguments to pass to func
        **kwargs: Keyword arguments to pass to func

    Returns:
        Function result

    Raises:
        SandboxViolationError: If capability violations occur in strict mode
    """

    if not config.enabled:
        # Sandbox disabled - execute directly
        return func(*args, **kwargs)

    if config.use_subprocess:
        # Execute in separate subprocess for isolation
        return _execute_in_subprocess(func, config, *args, **kwargs)
    else:
        # Execute in current process with capability checking
        with sandbox_context(config) as validator:
            return func(*args, **kwargs)


def _execute_in_subprocess(
    func: Callable[..., T], config: SandboxConfig, *args, **kwargs
) -> T:
    """
    Execute function in a subprocess with resource limits.

    This provides stronger isolation but with higher overhead.
    """
    # SECURITY FIX: Replace pickle with json for secure serialization
    # pickle is vulnerable to arbitrary code execution attacks
    import json
    # import pickle  # Disabled for security
    import base64

    # Serialize function and arguments
    try:
        func_data = json.dumps((func, args, kwargs))
        func_b64 = base64.b64encode(func_data).decode("ascii")
    except Exception as e:
        raise ValueError(f"Function cannot be serialized for subprocess execution: {e}")

    # Create subprocess execution script
    script = f"""
# SECURITY FIX: Replace pickle with json for secure serialization
# pickle is vulnerable to arbitrary code execution attacks
import json
# import pickle  # Disabled for security
import base64
import sys
import os

# Deserialize function and arguments
func_data = base64.b64decode("{func_b64}")
func, args, kwargs = json.loads(func_data)

# Set up sandbox context (basic version for subprocess)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    result = func(*args, **kwargs)
    print("GREENLANG_RESULT:" + base64.b64encode(json.dumps(result)).decode('ascii'))
except Exception as e:
    print("GREENLANG_ERROR:" + base64.b64encode(json.dumps(e)).decode('ascii'))
"""

    # Write script to temporary file
    script_path = os.path.join(config.temp_dir, "sandbox_script.py")
    with open(script_path, "w") as f:
        f.write(script)

    # Build subprocess command with resource limits
    cmd = [sys.executable, script_path]
    env = os.environ.copy()

    # Apply resource limits if available (Linux/Unix only)
    preexec_fn = None
    if hasattr(os, "setrlimit") and config.memory_limit_mb:
        import resource

        def limit_resources():
            # Set memory limit
            mem_bytes = config.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        preexec_fn = limit_resources

    try:
        # Execute subprocess
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.subprocess_timeout,
            preexec_fn=preexec_fn,
        )

        # Parse result from stdout
        for line in result.stdout.split("\n"):
            if line.startswith("GREENLANG_RESULT:"):
                result_data = base64.b64decode(line[17:])
                return json.loads(result_data)
            elif line.startswith("GREENLANG_ERROR:"):
                error_data = base64.b64decode(line[16:])
                error = json.loads(error_data)
                raise error

        # No result found - subprocess error
        raise RuntimeError(f"Subprocess execution failed: {result.stderr}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Subprocess execution timed out after {config.subprocess_timeout} seconds"
        )

    finally:
        # Cleanup script file
        try:
            os.remove(script_path)
        except:
            pass


# Convenience functions for common use cases


def create_basic_sandbox() -> SandboxConfig:
    """Create a basic sandbox configuration for simple compute tasks"""
    return SandboxConfig(
        enabled=True, policy=BASIC_COMPUTE_POLICY, raise_on_violation=True
    )


def create_data_processing_sandbox(
    read_paths: Optional[List[str]] = None, write_paths: Optional[List[str]] = None
) -> SandboxConfig:
    """Create a sandbox configuration for data processing tasks"""
    policy = CapabilityPolicy(
        allowed_capabilities=DATA_PROCESSING_POLICY.allowed_capabilities.copy(),
        allowed_read_paths=read_paths or ["/tmp"],
        allowed_write_paths=write_paths or ["/tmp"],
        max_memory_mb=1024,
        max_cpu_time_seconds=300,
    )

    return SandboxConfig(enabled=True, policy=policy, raise_on_violation=True)


def create_network_sandbox(allowed_hosts: Optional[List[str]] = None) -> SandboxConfig:
    """Create a sandbox configuration for network client operations"""
    policy = CapabilityPolicy(
        allowed_capabilities=NETWORK_CLIENT_POLICY.allowed_capabilities.copy(),
        allowed_hosts=allowed_hosts or ["api.openai.com", "httpbin.org"],
        max_memory_mb=512,
    )

    return SandboxConfig(enabled=True, policy=policy, raise_on_violation=True)


def disable_sandbox() -> SandboxConfig:
    """Create a disabled sandbox configuration (no restrictions)"""
    return SandboxConfig(enabled=False)


# Export main components
__all__ = [
    "SandboxConfig",
    "SandboxViolationError",
    "sandbox_context",
    "capability_check",
    "sandbox_execute",
    "get_current_validator",
    "get_current_config",
    "create_basic_sandbox",
    "create_data_processing_sandbox",
    "create_network_sandbox",
    "disable_sandbox",
    # Re-export from capabilities
    "Capability",
    "CapabilityPolicy",
    "CapabilityValidator",
    "DEFAULT_POLICY",
    "BASIC_COMPUTE_POLICY",
    "DATA_PROCESSING_POLICY",
    "NETWORK_CLIENT_POLICY",
    "FULL_ACCESS_POLICY",
]
