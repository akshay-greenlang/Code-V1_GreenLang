"""
Run ledger for deterministic execution tracking
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def stable_hash(obj: Any) -> str:
    """
    Create stable hash of object using deterministic JSON serialization
    
    Args:
        obj: Object to hash
    
    Returns:
        SHA-256 hex digest
    """
    # Convert to JSON with sorted keys and no whitespace
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def write_run_ledger(result: Any, ctx: Any, output_path: Optional[Path] = None) -> Path:
    """
    Write deterministic run ledger with stable hashing
    
    Args:
        result: Execution result object
        ctx: Execution context with pipeline spec, inputs, etc.
        output_path: Optional output path (defaults to out/run.json)
    
    Returns:
        Path to written ledger file
    """
    if output_path is None:
        output_path = Path("out/run.json")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build deterministic record
    record = {
        "version": "1.0.0",
        "kind": "greenlang-run-ledger",
        "metadata": {
            "started_at": getattr(ctx, "started_at", datetime.utcnow()).isoformat(),
            "finished_at": datetime.utcnow().isoformat(),
            "duration": time.time() - getattr(ctx, "start_time", time.time()),
            "status": "success" if getattr(result, "success", True) else "failed"
        },
        "spec": {
            "pipeline_hash": stable_hash(getattr(ctx, "pipeline_spec", {})),
            "inputs_hash": stable_hash(getattr(ctx, "inputs", {})),
            "config_hash": stable_hash(getattr(ctx, "config", {})),
            "artifacts": getattr(ctx, "artifacts_map", {}),
            "versions": getattr(ctx, "versions", {}),
            "sbom_ref": getattr(ctx, "sbom_path", None),
            "signatures": getattr(ctx, "signatures", [])
        },
        "execution": {
            "backend": getattr(ctx, "backend", "local"),
            "profile": getattr(ctx, "profile", "dev"),
            "environment": getattr(ctx, "environment", {})
        },
        "outputs": getattr(result, "outputs", {}),
        "metrics": getattr(result, "metrics", {})
    }
    
    # Add artifacts with hashes
    artifacts = []
    if hasattr(ctx, "artifacts"):
        for artifact in ctx.artifacts:
            artifact_info = {
                "name": getattr(artifact, "name", "unknown"),
                "path": str(getattr(artifact, "path", "")),
                "type": getattr(artifact, "type", "file"),
                "hash": None,
                "metadata": getattr(artifact, "metadata", {})
            }
            
            # Calculate artifact hash if file exists
            artifact_path = Path(artifact_info["path"])
            if artifact_path.exists() and artifact_path.is_file():
                artifact_info["hash"] = _calculate_file_hash(artifact_path)
            
            artifacts.append(artifact_info)
    
    record["spec"]["artifacts_list"] = artifacts
    
    # Add error details if failed
    if not getattr(result, "success", True):
        record["error"] = {
            "message": getattr(result, "error", "Unknown error"),
            "type": getattr(result, "error_type", "ExecutionError"),
            "traceback": getattr(result, "traceback", None)
        }
    
    # Calculate ledger hash (excluding timestamp fields)
    ledger_data = record.copy()
    ledger_data.pop("metadata", None)  # Remove timestamped metadata
    record["spec"]["ledger_hash"] = stable_hash(ledger_data)
    
    # Write deterministically (sorted keys for consistency)
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    
    return output_path


def _calculate_file_hash(file_path: Path) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to file
    
    Returns:
        SHA-256 hex digest
    """
    hasher = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def verify_run_ledger(ledger_path: Path) -> bool:
    """
    Verify a run ledger's integrity
    
    Args:
        ledger_path: Path to ledger JSON file
    
    Returns:
        True if ledger is valid and unmodified
    """
    if not ledger_path.exists():
        return False
    
    with open(ledger_path) as f:
        ledger = json.load(f)
    
    # Get stored hash
    stored_hash = ledger.get("spec", {}).get("ledger_hash")
    if not stored_hash:
        return False
    
    # Recalculate hash (excluding metadata)
    ledger_data = ledger.copy()
    ledger_data.pop("metadata", None)
    ledger_data["spec"] = ledger_data["spec"].copy()
    ledger_data["spec"].pop("ledger_hash", None)
    
    calculated_hash = stable_hash(ledger_data)
    
    return calculated_hash == stored_hash


def read_run_ledger(ledger_path: Path) -> Dict[str, Any]:
    """
    Read and validate a run ledger
    
    Args:
        ledger_path: Path to ledger JSON file
    
    Returns:
        Ledger data dictionary
    
    Raises:
        ValueError: If ledger is invalid or corrupted
    """
    if not ledger_path.exists():
        raise ValueError(f"Ledger not found: {ledger_path}")
    
    with open(ledger_path) as f:
        ledger = json.load(f)
    
    # Verify integrity
    if not verify_run_ledger(ledger_path):
        raise ValueError(f"Ledger integrity check failed: {ledger_path}")
    
    return ledger


def compare_runs(ledger1_path: Path, ledger2_path: Path) -> Dict[str, Any]:
    """
    Compare two run ledgers for reproducibility
    
    Args:
        ledger1_path: Path to first ledger
        ledger2_path: Path to second ledger
    
    Returns:
        Comparison results with differences
    """
    ledger1 = read_run_ledger(ledger1_path)
    ledger2 = read_run_ledger(ledger2_path)
    
    comparison = {
        "identical": True,
        "differences": []
    }
    
    # Compare pipeline hashes
    if ledger1["spec"]["pipeline_hash"] != ledger2["spec"]["pipeline_hash"]:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "pipeline_hash",
            "ledger1": ledger1["spec"]["pipeline_hash"],
            "ledger2": ledger2["spec"]["pipeline_hash"]
        })
    
    # Compare input hashes
    if ledger1["spec"]["inputs_hash"] != ledger2["spec"]["inputs_hash"]:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "inputs_hash",
            "ledger1": ledger1["spec"]["inputs_hash"],
            "ledger2": ledger2["spec"]["inputs_hash"]
        })
    
    # Compare config hashes
    if ledger1["spec"]["config_hash"] != ledger2["spec"]["config_hash"]:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "config_hash",
            "ledger1": ledger1["spec"]["config_hash"],
            "ledger2": ledger2["spec"]["config_hash"]
        })
    
    # Compare outputs (should be identical for reproducible runs)
    outputs1_hash = stable_hash(ledger1.get("outputs", {}))
    outputs2_hash = stable_hash(ledger2.get("outputs", {}))
    
    if outputs1_hash != outputs2_hash:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "outputs",
            "ledger1_hash": outputs1_hash,
            "ledger2_hash": outputs2_hash
        })
    
    # Compare metrics
    metrics1_hash = stable_hash(ledger1.get("metrics", {}))
    metrics2_hash = stable_hash(ledger2.get("metrics", {}))
    
    if metrics1_hash != metrics2_hash:
        comparison["identical"] = False
        comparison["differences"].append({
            "field": "metrics",
            "ledger1_hash": metrics1_hash,
            "ledger2_hash": metrics2_hash
        })
    
    return comparison