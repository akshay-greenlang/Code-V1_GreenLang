# -*- coding: utf-8 -*-
"""
GreenLang Provenance - Environment Module
Environment capture for reproducibility and audit trails.
"""

import os
import platform
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION ENVIRONMENT CAPTURE
# ============================================================================

def get_environment_info() -> Dict[str, Any]:
    """
    Capture complete execution environment for reproducibility.

    Records all relevant system and Python environment details needed
    to reproduce the exact execution environment. Critical for
    regulatory compliance and debugging.

    Returns:
        Dictionary with environment details:
        {
            "timestamp": str (ISO 8601),
            "python": {...},
            "system": {...},
            "process": {...},
            "greenlang": {...}
        }

    Example:
        >>> env = get_environment_info()
        >>> print(f"Python: {env['python']['version']}")
        >>> print(f"OS: {env['system']['os']} {env['system']['release']}")
    """
    env_info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # Python environment
        "python": get_python_info(),

        # System environment
        "system": get_system_info(),

        # Process information
        "process": get_process_info(),

        # GreenLang environment
        "greenlang": get_greenlang_info()
    }

    return env_info


def get_python_info() -> Dict[str, Any]:
    """
    Get Python interpreter information.

    Returns:
        Dictionary with Python details
    """
    return {
        "version": sys.version,
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
            "releaselevel": sys.version_info.releaselevel,
            "serial": sys.version_info.serial
        },
        "implementation": platform.python_implementation(),
        "compiler": platform.python_compiler(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "platform": sys.platform,
        "max_size": sys.maxsize,
        "byte_order": sys.byteorder
    }


def get_system_info() -> Dict[str, Any]:
    """
    Get system/OS information.

    Returns:
        Dictionary with system details
    """
    system_info = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        "hostname": platform.node(),
        "platform": platform.platform()
    }

    # Add CPU count if available
    try:
        import multiprocessing
        system_info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        system_info["cpu_count"] = "unknown"

    return system_info


def get_process_info() -> Dict[str, Any]:
    """
    Get current process information.

    Returns:
        Dictionary with process details
    """
    process_info = {
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
        "locale": os.getenv("LANG") or os.getenv("LC_ALL") or "unknown"
    }

    # Add parent PID if available
    try:
        process_info["ppid"] = os.getppid()
    except AttributeError:
        process_info["ppid"] = "unknown"

    return process_info


def get_greenlang_info() -> Dict[str, Any]:
    """
    Get GreenLang framework information.

    Returns:
        Dictionary with GreenLang version and configuration
    """
    greenlang_info = {
        "framework_version": "1.0.0",
        "provenance_version": "1.0.0"
    }

    # Try to get actual GreenLang version
    try:
        import greenlang
        if hasattr(greenlang, '__version__'):
            greenlang_info["framework_version"] = greenlang.__version__
    except ImportError:
        pass

    # Get environment variables
    greenlang_info["environment_vars"] = {
        "GREENLANG_HOME": os.getenv("GREENLANG_HOME"),
        "GREENLANG_CONFIG": os.getenv("GREENLANG_CONFIG"),
        "GREENLANG_CACHE_DIR": os.getenv("GREENLANG_CACHE_DIR"),
    }

    return greenlang_info


# ============================================================================
# DEPENDENCY VERSION TRACKING
# ============================================================================

def get_dependency_versions(
    packages: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Get versions of critical dependencies.

    Captures exact versions of Python packages used in the pipeline.
    Essential for reproducibility and regulatory compliance.

    Args:
        packages: Optional list of package names to check.
                 If None, checks common GreenLang dependencies.

    Returns:
        Dictionary mapping package name to version:
        {
            "pandas": "2.0.3",
            "pydantic": "2.1.1",
            ...
        }

    Example:
        >>> deps = get_dependency_versions()
        >>> print(f"Using pandas {deps.get('pandas', 'unknown')}")

        >>> # Check specific packages
        >>> deps = get_dependency_versions(["numpy", "scipy"])
    """
    if packages is None:
        # Default critical packages for GreenLang
        packages = [
            "pandas",
            "pydantic",
            "jsonschema",
            "pyyaml",
            "openpyxl",
            "numpy",
            "pyarrow",
            "lxml",
            "tqdm",
            "pytest"
        ]

    dependencies = {}

    for package in packages:
        dependencies[package] = _get_package_version(package)

    return dependencies


def _get_package_version(package_name: str) -> str:
    """
    Get version of a single package.

    Args:
        package_name: Name of package

    Returns:
        Version string or "unknown" if not installed
    """
    try:
        # Try importlib.metadata (Python 3.8+)
        try:
            from importlib.metadata import version
            return version(package_name)
        except ImportError:
            # Fallback to pkg_resources
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
    except Exception:
        return "unknown"


def get_all_installed_packages() -> Dict[str, str]:
    """
    Get all installed Python packages and their versions.

    Returns:
        Dictionary of all installed packages

    Example:
        >>> all_packages = get_all_installed_packages()
        >>> print(f"Total packages: {len(all_packages)}")
    """
    packages = {}

    try:
        # Try importlib.metadata (Python 3.8+)
        try:
            from importlib.metadata import distributions
            for dist in distributions():
                packages[dist.name] = dist.version
        except ImportError:
            # Fallback to pkg_resources
            import pkg_resources
            for dist in pkg_resources.working_set:
                packages[dist.project_name] = dist.version
    except Exception as e:
        logger.warning(f"Could not enumerate packages: {e}")

    return packages


def capture_environment_snapshot() -> Dict[str, Any]:
    """
    Capture a complete environment snapshot.

    Combines environment info, dependencies, and system details
    into a single comprehensive snapshot.

    Returns:
        Complete environment snapshot

    Example:
        >>> snapshot = capture_environment_snapshot()
        >>> # Save for later reproducibility
        >>> import json
        >>> with open("environment.json", "w") as f:
        ...     json.dump(snapshot, f, indent=2)
    """
    return {
        "environment": get_environment_info(),
        "dependencies": get_dependency_versions(),
        "all_packages": get_all_installed_packages(),
        "env_variables": dict(os.environ),
        "snapshot_timestamp": datetime.now(timezone.utc).isoformat()
    }


def compare_environments(env1: Dict[str, Any], env2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two environment snapshots.

    Args:
        env1: First environment snapshot
        env2: Second environment snapshot

    Returns:
        Dictionary with differences:
        {
            "python_version_match": bool,
            "os_match": bool,
            "dependency_differences": {...},
            "summary": str
        }

    Example:
        >>> env1 = capture_environment_snapshot()
        >>> # Later...
        >>> env2 = capture_environment_snapshot()
        >>> diff = compare_environments(env1, env2)
        >>> if not diff["python_version_match"]:
        ...     print("Warning: Python versions differ!")
    """
    differences = {
        "python_version_match": False,
        "os_match": False,
        "dependency_differences": {},
        "env_var_differences": {},
        "summary": ""
    }

    # Compare Python versions
    py1 = env1.get("environment", {}).get("python", {}).get("version_info", {})
    py2 = env2.get("environment", {}).get("python", {}).get("version_info", {})
    differences["python_version_match"] = (
        py1.get("major") == py2.get("major") and
        py1.get("minor") == py2.get("minor") and
        py1.get("micro") == py2.get("micro")
    )

    # Compare OS
    sys1 = env1.get("environment", {}).get("system", {})
    sys2 = env2.get("environment", {}).get("system", {})
    differences["os_match"] = (
        sys1.get("os") == sys2.get("os") and
        sys1.get("release") == sys2.get("release")
    )

    # Compare dependencies
    deps1 = env1.get("dependencies", {})
    deps2 = env2.get("dependencies", {})

    for pkg in set(deps1.keys()) | set(deps2.keys()):
        v1 = deps1.get(pkg, "not installed")
        v2 = deps2.get(pkg, "not installed")
        if v1 != v2:
            differences["dependency_differences"][pkg] = {
                "env1": v1,
                "env2": v2
            }

    # Create summary
    summary_parts = []
    if not differences["python_version_match"]:
        summary_parts.append("Python versions differ")
    if not differences["os_match"]:
        summary_parts.append("Operating systems differ")
    if differences["dependency_differences"]:
        summary_parts.append(f"{len(differences['dependency_differences'])} package version differences")

    if summary_parts:
        differences["summary"] = "; ".join(summary_parts)
    else:
        differences["summary"] = "Environments match"

    return differences
