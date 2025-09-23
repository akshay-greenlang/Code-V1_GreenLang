"""
Run ledger for deterministic execution tracking and audit trail
"""

import json
import hashlib
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


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
            "status": "success" if getattr(result, "success", True) else "failed",
        },
        "spec": {
            "pipeline_hash": stable_hash(getattr(ctx, "pipeline_spec", {})),
            "inputs_hash": stable_hash(getattr(ctx, "inputs", {})),
            "config_hash": stable_hash(getattr(ctx, "config", {})),
            "artifacts": getattr(ctx, "artifacts_map", {}),
            "versions": getattr(ctx, "versions", {}),
            "sbom_ref": getattr(ctx, "sbom_path", None),
            "signatures": getattr(ctx, "signatures", []),
        },
        "execution": {
            "backend": getattr(ctx, "backend", "local"),
            "profile": getattr(ctx, "profile", "dev"),
            "environment": getattr(ctx, "environment", {}),
        },
        "outputs": getattr(result, "outputs", {}),
        "metrics": getattr(result, "metrics", {}),
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
                "metadata": getattr(artifact, "metadata", {}),
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
            "traceback": getattr(result, "traceback", None),
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

    with open(file_path, "rb") as f:
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

    comparison = {"identical": True, "differences": []}

    # Compare pipeline hashes
    if ledger1["spec"]["pipeline_hash"] != ledger2["spec"]["pipeline_hash"]:
        comparison["identical"] = False
        comparison["differences"].append(
            {
                "field": "pipeline_hash",
                "ledger1": ledger1["spec"]["pipeline_hash"],
                "ledger2": ledger2["spec"]["pipeline_hash"],
            }
        )

    # Compare input hashes
    if ledger1["spec"]["inputs_hash"] != ledger2["spec"]["inputs_hash"]:
        comparison["identical"] = False
        comparison["differences"].append(
            {
                "field": "inputs_hash",
                "ledger1": ledger1["spec"]["inputs_hash"],
                "ledger2": ledger2["spec"]["inputs_hash"],
            }
        )

    # Compare config hashes
    if ledger1["spec"]["config_hash"] != ledger2["spec"]["config_hash"]:
        comparison["identical"] = False
        comparison["differences"].append(
            {
                "field": "config_hash",
                "ledger1": ledger1["spec"]["config_hash"],
                "ledger2": ledger2["spec"]["config_hash"],
            }
        )

    # Compare outputs (should be identical for reproducible runs)
    outputs1_hash = stable_hash(ledger1.get("outputs", {}))
    outputs2_hash = stable_hash(ledger2.get("outputs", {}))

    if outputs1_hash != outputs2_hash:
        comparison["identical"] = False
        comparison["differences"].append(
            {
                "field": "outputs",
                "ledger1_hash": outputs1_hash,
                "ledger2_hash": outputs2_hash,
            }
        )

    # Compare metrics
    metrics1_hash = stable_hash(ledger1.get("metrics", {}))
    metrics2_hash = stable_hash(ledger2.get("metrics", {}))

    if metrics1_hash != metrics2_hash:
        comparison["identical"] = False
        comparison["differences"].append(
            {
                "field": "metrics",
                "ledger1_hash": metrics1_hash,
                "ledger2_hash": metrics2_hash,
            }
        )

    return comparison


class RunLedger:
    """
    Append-only ledger for tracking all pipeline executions

    Provides audit trail, compliance tracking, and execution history
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        """
        Initialize run ledger

        Args:
            ledger_path: Path to ledger file (defaults to ~/.greenlang/ledger.jsonl)
        """
        if ledger_path is None:
            gl_home = Path.home() / ".greenlang"
            gl_home.mkdir(parents=True, exist_ok=True)
            self.ledger_path = gl_home / "ledger.jsonl"
        else:
            self.ledger_path = Path(ledger_path)
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize ledger file if it doesn't exist
        if not self.ledger_path.exists():
            self.ledger_path.touch()
            logger.info(f"Created new ledger: {self.ledger_path}")

    def record_run(
        self,
        pipeline: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a pipeline execution in the ledger

        Args:
            pipeline: Pipeline name or reference
            inputs: Input data for the pipeline
            outputs: Output data from the pipeline
            metadata: Additional metadata (duration, backend, etc.)

        Returns:
            Run ID (UUID)
        """
        # Generate unique run ID
        run_id = str(uuid.uuid4())

        # Create ledger entry
        entry = {
            "id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline": pipeline,
            "input_hash": hashlib.sha256(
                json.dumps(inputs, sort_keys=True).encode()
            ).hexdigest(),
            "output_hash": hashlib.sha256(
                json.dumps(outputs, sort_keys=True).encode()
            ).hexdigest(),
            "metadata": metadata or {},
        }

        # Add system metadata
        entry["metadata"].update(
            {"recorded_at": datetime.utcnow().isoformat(), "ledger_version": "1.0.0"}
        )

        # Append to ledger (JSONL format - one JSON object per line)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        logger.info(f"Recorded run {run_id} for pipeline {pipeline}")
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific run by ID

        Args:
            run_id: Run UUID

        Returns:
            Run entry or None if not found
        """
        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("id") == run_id:
                        return entry
        return None

    def list_runs(
        self,
        pipeline: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List runs from the ledger

        Args:
            pipeline: Filter by pipeline name (optional)
            limit: Maximum number of entries to return
            since: Only return runs since this timestamp

        Returns:
            List of run entries
        """
        runs = []

        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    # Apply filters
                    if pipeline and entry.get("pipeline") != pipeline:
                        continue

                    if since:
                        entry_time = datetime.fromisoformat(entry["timestamp"])
                        if entry_time < since:
                            continue

                    runs.append(entry)

                    if len(runs) >= limit:
                        break

        # Return in reverse chronological order
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        return runs[:limit]

    def find_duplicate_runs(
        self, input_hash: str, pipeline: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find runs with the same input hash (for deduplication)

        Args:
            input_hash: SHA-256 hash of inputs
            pipeline: Filter by pipeline name (optional)

        Returns:
            List of matching run entries
        """
        matches = []

        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)

                    if entry.get("input_hash") == input_hash:
                        if not pipeline or entry.get("pipeline") == pipeline:
                            matches.append(entry)

        return matches

    def get_statistics(
        self, pipeline: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get execution statistics from the ledger

        Args:
            pipeline: Filter by pipeline name (optional)
            days: Number of days to look back

        Returns:
            Statistics dictionary
        """
        since = datetime.utcnow() - timedelta(days=days)
        runs = self.list_runs(pipeline=pipeline, limit=10000, since=since)

        if not runs:
            return {
                "total_runs": 0,
                "unique_inputs": 0,
                "unique_outputs": 0,
                "average_per_day": 0,
                "pipelines": [],
            }

        # Collect statistics
        input_hashes = set()
        output_hashes = set()
        pipelines = set()

        for run in runs:
            input_hashes.add(run["input_hash"])
            output_hashes.add(run["output_hash"])
            pipelines.add(run["pipeline"])

        return {
            "total_runs": len(runs),
            "unique_inputs": len(input_hashes),
            "unique_outputs": len(output_hashes),
            "average_per_day": len(runs) / days,
            "pipelines": list(pipelines),
            "period_days": days,
            "since": since.isoformat(),
            "latest_run": runs[0]["timestamp"] if runs else None,
        }

    def verify_reproducibility(
        self, input_hash: str, output_hash: str, pipeline: str
    ) -> bool:
        """
        Verify if outputs are reproducible for given inputs

        Args:
            input_hash: Expected input hash
            output_hash: Expected output hash
            pipeline: Pipeline name

        Returns:
            True if all runs with same inputs produce same outputs
        """
        runs = self.find_duplicate_runs(input_hash, pipeline)

        if not runs:
            return False

        # Check if all runs with same input produce same output
        for run in runs:
            if run["output_hash"] != output_hash:
                logger.warning(
                    f"Reproducibility issue: Run {run['id']} produced "
                    f"different output for same input"
                )
                return False

        return True

    def export_to_json(
        self, output_path: Path, pipeline: Optional[str] = None, days: int = 30
    ) -> Path:
        """
        Export ledger entries to JSON file

        Args:
            output_path: Path to output JSON file
            pipeline: Filter by pipeline name (optional)
            days: Number of days to export

        Returns:
            Path to exported file
        """
        since = datetime.utcnow() - timedelta(days=days)
        runs = self.list_runs(pipeline=pipeline, limit=10000, since=since)

        export_data = {
            "version": "1.0.0",
            "exported_at": datetime.utcnow().isoformat(),
            "ledger_path": str(self.ledger_path),
            "filters": {"pipeline": pipeline, "days": days, "since": since.isoformat()},
            "runs": runs,
            "statistics": self.get_statistics(pipeline, days),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported {len(runs)} runs to {output_path}")
        return output_path

    def clean_old_entries(self, days_to_keep: int = 90) -> int:
        """
        Clean old entries from the ledger (compliance/storage management)

        Args:
            days_to_keep: Number of days of history to keep

        Returns:
            Number of entries removed
        """
        cutoff = datetime.utcnow() - timedelta(days=days_to_keep)

        # Read all entries
        kept_entries = []
        removed_count = 0

        with open(self.ledger_path, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry["timestamp"])

                    if entry_time >= cutoff:
                        kept_entries.append(entry)
                    else:
                        removed_count += 1

        # Rewrite ledger with kept entries
        with open(self.ledger_path, "w") as f:
            for entry in kept_entries:
                f.write(json.dumps(entry) + "\n")

        logger.info(f"Cleaned {removed_count} old entries from ledger")
        return removed_count


# Import timedelta for the statistics method
from datetime import timedelta
