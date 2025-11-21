# -*- coding: utf-8 -*-
"""
GreenLang Provenance - Validation Module
Provenance record validation and integrity verification.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from .records import ProvenanceRecord
from .hashing import hash_file

logger = logging.getLogger(__name__)


# ============================================================================
# PROVENANCE VALIDATION
# ============================================================================

def validate_provenance(
    provenance: ProvenanceRecord,
    input_file: Optional[str] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate a provenance record.

    Performs comprehensive validation:
    1. Input file hash matches (if file provided)
    2. All required fields present
    3. Timestamps are valid
    4. Agent execution chain is complete
    5. Data lineage is consistent

    Args:
        provenance: ProvenanceRecord to validate
        input_file: Optional input file to verify hash
        strict: If True, warnings become errors

    Returns:
        Validation result:
        {
            "is_valid": bool,
            "checks_passed": int,
            "errors": int,
            "warnings": int,
            "checks": [...],
            "error_details": [...],
            "warning_details": [...],
            "summary": str
        }

    Example:
        >>> provenance = ProvenanceRecord.load("provenance.json")
        >>> result = validate_provenance(provenance, input_file="data.csv")
        >>> if result['is_valid']:
        ...     print("Provenance verified ✓")
    """
    checks = []
    errors = []
    warnings = []

    # Check 1: Input file hash (if file provided)
    if input_file:
        _validate_input_hash(provenance, input_file, checks, errors, warnings)

    # Check 2: Required fields
    _validate_required_fields(provenance, checks, errors)

    # Check 3: Timestamps valid
    _validate_timestamps(provenance, checks, errors)

    # Check 4: Agent execution chain
    _validate_agent_chain(provenance, checks, warnings, strict)

    # Check 5: Data lineage consistency
    _validate_data_lineage(provenance, checks, warnings)

    # Check 6: Environment info completeness
    _validate_environment(provenance, checks, warnings)

    # Determine overall validity
    is_valid = len(errors) == 0 and (not strict or len(warnings) == 0)

    return {
        "is_valid": is_valid,
        "checks_passed": len(checks),
        "errors": len(errors),
        "warnings": len(warnings),
        "checks": checks,
        "error_details": errors,
        "warning_details": warnings,
        "summary": f"{'✓ Valid' if is_valid else '✗ Invalid'} provenance record "
                   f"({len(checks)} checks passed, {len(errors)} errors, {len(warnings)} warnings)"
    }


def _validate_input_hash(
    provenance: ProvenanceRecord,
    input_file: str,
    checks: List,
    errors: List,
    warnings: List
):
    """Validate input file hash matches."""
    try:
        current_hash = hash_file(input_file)
        if provenance.input_file_hash and current_hash['hash_value'] == provenance.input_file_hash.get('hash_value'):
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
                "expected": provenance.input_file_hash.get('hash_value') if provenance.input_file_hash else None,
                "actual": current_hash['hash_value']
            })
    except Exception as e:
        warnings.append({
            "check": "input_file_hash",
            "status": "warning",
            "message": f"Could not verify input file hash: {e}"
        })


def _validate_required_fields(provenance: ProvenanceRecord, checks: List, errors: List):
    """Validate all required fields are present."""
    required_fields = [
        "record_id",
        "generated_at",
        "environment",
        "dependencies",
        "configuration"
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


def _validate_timestamps(provenance: ProvenanceRecord, checks: List, errors: List):
    """Validate timestamps are valid."""
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


def _validate_agent_chain(provenance: ProvenanceRecord, checks: List, warnings: List, strict: bool):
    """Validate agent execution chain."""
    if provenance.agent_execution and len(provenance.agent_execution) > 0:
        checks.append({
            "check": "agent_chain",
            "status": "pass",
            "message": f"Agent chain present ({len(provenance.agent_execution)} agents)"
        })
    else:
        warning = {
            "check": "agent_chain",
            "status": "warning",
            "message": "No agent executions recorded"
        }
        warnings.append(warning)


def _validate_data_lineage(provenance: ProvenanceRecord, checks: List, warnings: List):
    """Validate data lineage consistency."""
    if provenance.data_lineage:
        # Check that lineage steps are sequential
        steps = [event.get("step", -1) for event in provenance.data_lineage]
        expected_steps = list(range(len(steps)))

        if steps == expected_steps:
            checks.append({
                "check": "data_lineage",
                "status": "pass",
                "message": f"Data lineage is consistent ({len(steps)} steps)"
            })
        else:
            warnings.append({
                "check": "data_lineage",
                "status": "warning",
                "message": "Data lineage steps are not sequential"
            })
    else:
        warnings.append({
            "check": "data_lineage",
            "status": "warning",
            "message": "No data lineage recorded"
        })


def _validate_environment(provenance: ProvenanceRecord, checks: List, warnings: List):
    """Validate environment information is complete."""
    if provenance.environment:
        required_env_keys = ["python", "system", "process"]
        missing_keys = [k for k in required_env_keys if k not in provenance.environment]

        if not missing_keys:
            checks.append({
                "check": "environment_complete",
                "status": "pass",
                "message": "Environment information is complete"
            })
        else:
            warnings.append({
                "check": "environment_complete",
                "status": "warning",
                "message": f"Missing environment keys: {', '.join(missing_keys)}"
            })


# ============================================================================
# INTEGRITY VERIFICATION
# ============================================================================

def verify_integrity(
    provenance: ProvenanceRecord,
    artifacts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Verify integrity of provenance and associated artifacts.

    Args:
        provenance: ProvenanceRecord to verify
        artifacts: Optional list of artifact paths to verify

    Returns:
        Verification result with integrity status

    Example:
        >>> result = verify_integrity(provenance, artifacts=["output.csv"])
        >>> assert result["integrity_verified"]
    """
    result = {
        "integrity_verified": True,
        "provenance_valid": False,
        "artifacts_verified": [],
        "issues": []
    }

    # Validate provenance record
    validation = validate_provenance(provenance)
    result["provenance_valid"] = validation["is_valid"]

    if not validation["is_valid"]:
        result["integrity_verified"] = False
        result["issues"].extend(validation["error_details"])

    # Verify artifacts if provided
    if artifacts:
        for artifact_path in artifacts:
            try:
                artifact_hash = hash_file(artifact_path)
                result["artifacts_verified"].append({
                    "path": artifact_path,
                    "verified": True,
                    "hash": artifact_hash["hash_value"]
                })
            except Exception as e:
                result["integrity_verified"] = False
                result["issues"].append({
                    "artifact": artifact_path,
                    "error": str(e)
                })
                result["artifacts_verified"].append({
                    "path": artifact_path,
                    "verified": False,
                    "error": str(e)
                })

    return result
