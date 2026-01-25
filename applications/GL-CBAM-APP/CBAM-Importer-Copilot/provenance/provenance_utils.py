# -*- coding: utf-8 -*-
"""
CBAM Importer Copilot - Provenance Utilities

Enterprise-grade provenance tracking for regulatory compliance.

This module provides utilities for creating complete audit trails that
meet EU CBAM regulatory requirements for data integrity and traceability.

Version: 1.0.0
Author: GreenLang CBAM Team
"""

import hashlib
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


# ============================================================================
# FILE INTEGRITY (SHA256 HASHING)
# ============================================================================

def hash_file(file_path: str, algorithm: str = "sha256") -> Dict[str, Any]:
    """
    Calculate cryptographic hash of a file for integrity verification.

    This creates a unique fingerprint of the input file that can be used
    to verify that the file hasn't been tampered with. Required for
    EU CBAM regulatory compliance.

    Args:
        file_path: Path to file to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Dictionary with hash details:
        {
            "file_path": str,
            "file_name": str,
            "file_size_bytes": int,
            "hash_algorithm": str,
            "hash_value": str (hex),
            "hash_timestamp": str (ISO 8601),
            "verification": str (how to verify)
        }

    Example:
        >>> hash_info = hash_file("shipments.csv")
        >>> print(f"SHA256: {hash_info['hash_value']}")
        >>> # Later, verify integrity:
        >>> new_hash = hash_file("shipments.csv")
        >>> assert new_hash['hash_value'] == hash_info['hash_value']
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Select hash algorithm
    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    # Read file in chunks (memory-efficient for large files)
    chunk_size = 65536  # 64 KB chunks

    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    hash_value = hasher.hexdigest()
    file_size = file_path.stat().st_size

    return {
        "file_path": str(file_path.absolute()),
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "hash_algorithm": algorithm.upper(),
        "hash_value": hash_value,
        "hash_timestamp": datetime.now(timezone.utc).isoformat(),
        "verification": f"{algorithm}sum {file_path.name}",
        "human_readable_size": _format_bytes(file_size)
    }


def _format_bytes(bytes_size: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


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
        "python": {
            "version": sys.version,
            "version_info": {
                "major": sys.version_info.major,
                "minor": sys.version_info.minor,
                "micro": sys.version_info.micro,
                "releaselevel": sys.version_info.releaselevel
            },
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "executable": sys.executable,
            "prefix": sys.prefix
        },

        # System environment
        "system": {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0],
            "hostname": platform.node()
        },

        # Process information
        "process": {
            "pid": os.getpid(),
            "cwd": os.getcwd(),
            "user": os.getenv("USER") or os.getenv("USERNAME") or "unknown",
            "locale": os.getenv("LANG") or os.getenv("LC_ALL") or "unknown"
        },

        # GreenLang environment
        "greenlang": {
            "cbam_copilot_version": "1.0.0",
            "pack_version": "1.0.0"
        }
    }

    return env_info


# ============================================================================
# DEPENDENCY VERSION TRACKING
# ============================================================================

def get_dependency_versions() -> Dict[str, str]:
    """
    Get versions of all critical dependencies.

    Captures exact versions of all Python packages used in the pipeline.
    Essential for reproducibility and regulatory compliance.

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
    """
    dependencies = {}

    # Core dependencies
    critical_packages = [
        "pandas",
        "pydantic",
        "jsonschema",
        "pyyaml",
        "openpyxl"
    ]

    for package in critical_packages:
        try:
            # Try importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import version
                dependencies[package] = version(package)
            except ImportError:
                # Fallback to pkg_resources
                import pkg_resources
                dependencies[package] = pkg_resources.get_distribution(package).version
        except Exception:
            dependencies[package] = "unknown"

    return dependencies


# ============================================================================
# PROVENANCE RECORD
# ============================================================================

@dataclass
class ProvenanceRecord:
    """
    Complete provenance record for a CBAM report.

    This dataclass captures all information needed to audit and reproduce
    a CBAM report generation.

    Attributes:
        report_id: Unique report identifier
        generated_at: Timestamp (ISO 8601)
        input_file_hash: SHA256 hash of input file
        environment: Execution environment details
        dependencies: Package versions
        configuration: Configuration snapshot
        agent_execution: Agent execution details
        data_lineage: Data transformation lineage
        validation_results: Validation outcomes
    """
    report_id: str
    generated_at: str
    input_file_hash: Dict[str, Any]
    environment: Dict[str, Any]
    dependencies: Dict[str, str]
    configuration: Dict[str, Any]
    agent_execution: List[Dict[str, Any]]
    data_lineage: List[Dict[str, Any]]
    validation_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str):
        """Save provenance record to file."""
        with open(path, 'w') as f:
            f.write(self.to_json())


# ============================================================================
# PROVENANCE CREATION
# ============================================================================

def create_provenance_record(
    report_id: str,
    input_file: str,
    configuration: Dict[str, Any],
    agent_executions: List[Dict[str, Any]],
    validation_results: Dict[str, Any]
) -> ProvenanceRecord:
    """
    Create a complete provenance record for a report.

    Args:
        report_id: Unique report identifier
        input_file: Path to input file
        configuration: Configuration used
        agent_executions: List of agent execution details
        validation_results: Validation outcomes

    Returns:
        ProvenanceRecord with complete audit trail

    Example:
        >>> provenance = create_provenance_record(
        ...     report_id="CBAM-2025Q4-001",
        ...     input_file="shipments.csv",
        ...     configuration={...},
        ...     agent_executions=[...],
        ...     validation_results={...}
        ... )
        >>> provenance.save("provenance.json")
    """
    # Hash input file
    input_hash = hash_file(input_file)

    # Capture environment
    environment = get_environment_info()

    # Get dependency versions
    dependencies = get_dependency_versions()

    # Create data lineage
    data_lineage = _create_data_lineage(input_file, agent_executions)

    # Create provenance record
    provenance = ProvenanceRecord(
        report_id=report_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        input_file_hash=input_hash,
        environment=environment,
        dependencies=dependencies,
        configuration=configuration,
        agent_execution=agent_executions,
        data_lineage=data_lineage,
        validation_results=validation_results
    )

    return provenance


def _create_data_lineage(
    input_file: str,
    agent_executions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Create data lineage showing data flow through pipeline.

    Args:
        input_file: Original input file
        agent_executions: Agent execution details

    Returns:
        List of lineage events
    """
    lineage = []

    # Input event
    lineage.append({
        "step": 0,
        "stage": "input",
        "description": "Raw shipment data ingested",
        "file": input_file,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })

    # Agent transformations
    for idx, execution in enumerate(agent_executions, 1):
        lineage.append({
            "step": idx,
            "stage": execution.get("agent_name", f"agent_{idx}"),
            "description": execution.get("description", "Data transformation"),
            "input_records": execution.get("input_records", 0),
            "output_records": execution.get("output_records", 0),
            "timestamp": execution.get("end_time", datetime.now(timezone.utc).isoformat())
        })

    return lineage


# ============================================================================
# PROVENANCE VALIDATION
# ============================================================================

def validate_provenance(
    provenance: ProvenanceRecord,
    input_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Validate a provenance record.

    Checks:
    1. Input file hash matches (if file provided)
    2. All required fields present
    3. Timestamps are valid
    4. Agent execution chain is complete

    Args:
        provenance: ProvenanceRecord to validate
        input_file: Optional input file to verify hash

    Returns:
        Validation result:
        {
            "is_valid": bool,
            "checks": [...],
            "errors": [...],
            "warnings": [...]
        }

    Example:
        >>> provenance = ProvenanceRecord.from_json("provenance.json")
        >>> result = validate_provenance(provenance, input_file="shipments.csv")
        >>> if result['is_valid']:
        ...     print("Provenance verified ✓")
    """
    checks = []
    errors = []
    warnings = []

    # Check 1: Input file hash (if file provided)
    if input_file:
        try:
            current_hash = hash_file(input_file)
            if current_hash['hash_value'] == provenance.input_file_hash['hash_value']:
                checks.append({
                    "check": "input_file_hash",
                    "status": "pass",
                    "message": "Input file hash matches provenance record"
                })
            else:
                errors.append({
                    "check": "input_file_hash",
                    "status": "fail",
                    "message": "Input file hash does NOT match provenance record",
                    "expected": provenance.input_file_hash['hash_value'],
                    "actual": current_hash['hash_value']
                })
        except Exception as e:
            warnings.append({
                "check": "input_file_hash",
                "status": "warning",
                "message": f"Could not verify input file hash: {e}"
            })

    # Check 2: Required fields
    required_fields = [
        "report_id",
        "generated_at",
        "input_file_hash",
        "environment",
        "dependencies",
        "configuration",
        "agent_execution",
        "data_lineage",
        "validation_results"
    ]

    missing_fields = []
    for field in required_fields:
        if not hasattr(provenance, field) or getattr(provenance, field) is None:
            missing_fields.append(field)

    if not missing_fields:
        checks.append({
            "check": "required_fields",
            "status": "pass",
            "message": "All required fields present"
        })
    else:
        errors.append({
            "check": "required_fields",
            "status": "fail",
            "message": f"Missing required fields: {', '.join(missing_fields)}"
        })

    # Check 3: Timestamps valid
    try:
        generated_at = datetime.fromisoformat(provenance.generated_at.replace('Z', '+00:00'))
        checks.append({
            "check": "timestamp_valid",
            "status": "pass",
            "message": f"Timestamp is valid: {generated_at}"
        })
    except Exception as e:
        errors.append({
            "check": "timestamp_valid",
            "status": "fail",
            "message": f"Invalid timestamp: {e}"
        })

    # Check 4: Agent execution chain
    if provenance.agent_execution and len(provenance.agent_execution) >= 3:
        checks.append({
            "check": "agent_chain",
            "status": "pass",
            "message": f"Complete agent chain ({len(provenance.agent_execution)} agents)"
        })
    else:
        warnings.append({
            "check": "agent_chain",
            "status": "warning",
            "message": f"Incomplete agent chain ({len(provenance.agent_execution or [])} agents, expected 3)"
        })

    # Determine overall validity
    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "checks_passed": len(checks),
        "errors": len(errors),
        "warnings": len(warnings),
        "checks": checks,
        "error_details": errors,
        "warning_details": warnings,
        "summary": f"{'✓ Valid' if is_valid else '✗ Invalid'} provenance record ({len(checks)} checks passed, {len(errors)} errors, {len(warnings)} warnings)"
    }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_audit_report(provenance: ProvenanceRecord) -> str:
    """
    Generate human-readable audit report from provenance record.

    Args:
        provenance: ProvenanceRecord to report on

    Returns:
        Markdown-formatted audit report

    Example:
        >>> report = generate_audit_report(provenance)
        >>> print(report)
        >>> # Or save to file:
        >>> with open("audit_report.md", "w") as f:
        ...     f.write(report)
    """
    report_lines = []

    report_lines.append("# CBAM PROVENANCE AUDIT REPORT")
    report_lines.append("")
    report_lines.append(f"**Report ID:** {provenance.report_id}")
    report_lines.append(f"**Generated:** {provenance.generated_at}")
    report_lines.append("")

    # Input File Integrity
    report_lines.append("## Input File Integrity")
    report_lines.append("")
    report_lines.append(f"- **File:** {provenance.input_file_hash['file_name']}")
    report_lines.append(f"- **Size:** {provenance.input_file_hash['human_readable_size']}")
    report_lines.append(f"- **SHA256:** `{provenance.input_file_hash['hash_value']}`")
    report_lines.append(f"- **Hashed:** {provenance.input_file_hash['hash_timestamp']}")
    report_lines.append("")

    # Execution Environment
    report_lines.append("## Execution Environment")
    report_lines.append("")
    py_version = provenance.environment['python']['version_info']
    report_lines.append(f"- **Python:** {py_version['major']}.{py_version['minor']}.{py_version['micro']}")
    report_lines.append(f"- **OS:** {provenance.environment['system']['os']} {provenance.environment['system']['release']}")
    report_lines.append(f"- **Machine:** {provenance.environment['system']['machine']}")
    report_lines.append("")

    # Dependencies
    report_lines.append("## Dependencies")
    report_lines.append("")
    for pkg, version in provenance.dependencies.items():
        report_lines.append(f"- {pkg}: {version}")
    report_lines.append("")

    # Agent Execution
    report_lines.append("## Agent Execution")
    report_lines.append("")
    for execution in provenance.agent_execution:
        report_lines.append(f"### {execution.get('agent_name', 'Unknown Agent')}")
        report_lines.append(f"- Started: {execution.get('start_time', 'N/A')}")
        report_lines.append(f"- Ended: {execution.get('end_time', 'N/A')}")
        report_lines.append(f"- Duration: {execution.get('duration_seconds', 'N/A')}s")
        report_lines.append("")

    # Data Lineage
    report_lines.append("## Data Lineage")
    report_lines.append("")
    for event in provenance.data_lineage:
        report_lines.append(f"{event['step']}. **{event['stage']}**: {event['description']}")
    report_lines.append("")

    # Validation
    report_lines.append("## Validation Results")
    report_lines.append("")
    report_lines.append(f"- **Status:** {'✓ PASS' if provenance.validation_results.get('is_valid') else '✗ FAIL'}")
    report_lines.append(f"- **Errors:** {len(provenance.validation_results.get('errors', []))}")
    report_lines.append(f"- **Warnings:** {len(provenance.validation_results.get('warnings', []))}")
    report_lines.append("")

    return "\n".join(report_lines)


# ============================================================================
# END OF PROVENANCE UTILS
# ============================================================================
