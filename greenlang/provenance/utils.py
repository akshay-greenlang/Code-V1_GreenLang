# -*- coding: utf-8 -*-
"""
Provenance utilities for tracking and verification
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


def track_provenance(func):
    """
    Decorator to automatically track provenance for functions

    Usage:
        @track_provenance
        def my_pipeline(inputs):
            # Pipeline code
            return results
    """

    def wrapper(*args, **kwargs):
        # Create provenance context
        from . import ProvenanceContext

        ctx = ProvenanceContext()

        # Record inputs
        ctx.record_inputs(args, kwargs)

        # Execute function
        try:
            result = func(*args, **kwargs)
            ctx.record_outputs(result)
            ctx.status = "success"
        except Exception as e:
            ctx.status = "failed"
            ctx.error = str(e)
            raise
        finally:
            # Write provenance records
            ctx.finalize()

        return result

    return wrapper


class ProvenanceContext:
    """
    Context manager for provenance tracking
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.started_at = DeterministicClock.utcnow()
        self.start_time = DeterministicClock.utcnow().timestamp()
        self.inputs = {}
        self.outputs = {}
        self.artifacts = []
        self.artifacts_map = {}
        self.versions = {}
        self.signatures = []
        self.sbom_path = None
        self.pipeline_spec = {}
        self.config = {}
        self.backend = "local"
        self.profile = "dev"
        self.environment = {}
        self.status = "running"
        self.error = None

    def record_inputs(self, args: tuple, kwargs: dict):
        """Record function inputs"""
        self.inputs = {"args": list(args), "kwargs": kwargs}

    def record_outputs(self, outputs: Any):
        """Record function outputs"""
        self.outputs = outputs

    def add_artifact(
        self, name: str, path: Path, artifact_type: str = "file", metadata: Dict = None
    ):
        """Add an artifact to the context"""
        artifact = {
            "name": name,
            "path": str(path),
            "type": artifact_type,
            "metadata": metadata or {},
        }
        self.artifacts.append(artifact)
        self.artifacts_map[name] = str(path)

    def add_signature(self, sig_type: str, value: str, metadata: Dict = None):
        """Add a signature to the context"""
        signature = {
            "type": sig_type,
            "value": value,
            "metadata": metadata or {},
            "timestamp": DeterministicClock.utcnow().isoformat(),
        }
        self.signatures.append(signature)

    def set_sbom(self, sbom_path: Path):
        """Set SBOM reference"""
        self.sbom_path = str(sbom_path)

    def finalize(self):
        """Finalize provenance tracking and write records"""
        from .ledger import write_run_ledger

        # Create result object
        result = type(
            "Result",
            (),
            {
                "success": self.status == "success",
                "outputs": self.outputs,
                "metrics": {},
                "error": self.error,
            },
        )()

        # Write run ledger
        ledger_path = write_run_ledger(result, self)
        logger.info(f"Provenance ledger written to: {ledger_path}")

        return ledger_path


def record_seed_info(
    ctx: ProvenanceContext,
    spec: dict,
    seed_root: int,
    seed_path: str = "",
    seed_child: Optional[int] = None,
    spec_type: str = "scenario"
) -> dict:
    """
    Record seed information for deterministic replay.

    Integrates with existing GreenLang provenance system to track
    RNG seeds, spec hashes, and seed derivation paths for reproducibility.

    Args:
        ctx: Provenance context (from executor or SDK)
        spec: Scenario/pipeline/agent specification dict
        seed_root: Root seed value (0 to 2^64-1)
        seed_path: Hierarchical path (e.g., "scenario:foo|param:bar|trial:42")
        seed_child: Derived child seed (optional)
        spec_type: Type of spec ("scenario", "pipeline", "agent")

    Returns:
        Dictionary of recorded seed info

    Example:
        >>> from greenlang.provenance.utils import ProvenanceContext, record_seed_info
        >>> ctx = ProvenanceContext("my_scenario")
        >>> spec = {"name": "baseline_sweep", "parameters": [...]}
        >>> seed_info = record_seed_info(
        ...     ctx=ctx,
        ...     spec=spec,
        ...     seed_root=42,
        ...     seed_path="scenario:baseline_sweep|param:temperature|trial:0",
        ...     seed_child=123456789
        ... )
        >>> print(seed_info["spec_hash"][:16])
        'a1b2c3d4e5f67890'
    """
    from .ledger import stable_hash

    # Calculate spec hash using existing infrastructure
    spec_hash = stable_hash(spec)

    # Build seed info structure
    seed_info = {
        f"spec_hash_{spec_type}": spec_hash,
        "seed_root": seed_root,
        "seed_path": seed_path or "root",
        "seed_child": seed_child,
        "spec_type": spec_type,
        "recorded_at": DeterministicClock.utcnow().isoformat()
    }

    # Store in context metadata (extends existing pattern)
    if not hasattr(ctx, "metadata"):
        ctx.metadata = {}

    ctx.metadata.setdefault("seed_tracking", {})
    ctx.metadata["seed_tracking"].update(seed_info)

    # Also add as artifact for discoverability
    seed_path_safe = seed_path.replace(":", "_").replace("|", "_") if seed_path else "root"
    ctx.add_artifact(
        name=f"seed_info_{seed_path_safe}",
        path=Path(".greenlang") / "seed_info.json",
        artifact_type="seed_metadata",
        metadata=seed_info
    )

    logger.info(
        f"Recorded seed info: spec_hash={spec_hash[:8]}..., "
        f"root={seed_root}, path={seed_path}, child={seed_child}"
    )

    return seed_info


def derive_child_seed(parent_seed: int, seed_path: str) -> int:
    """
    Derive a child seed from parent seed and hierarchical path.

    Uses deterministic hashing to ensure reproducibility.
    This is a convenience wrapper around GLRNG's derivation logic.

    Args:
        parent_seed: Parent seed value (0 to 2^64-1)
        seed_path: Hierarchical path (e.g., "param:temperature|trial:0")

    Returns:
        Derived child seed (64-bit unsigned integer)

    Example:
        >>> child = derive_child_seed(42, "scenario:foo|param:bar")
        >>> child  # Deterministic output
        12345678901234567
    """
    import hashlib

    # Create deterministic derivation (matches GLRNG implementation)
    seed_string = f"{parent_seed}:{seed_path}"
    hash_bytes = hashlib.sha256(seed_string.encode()).digest()

    # Convert first 8 bytes to 64-bit integer
    child_seed = int.from_bytes(hash_bytes[:8], byteorder='little', signed=False)

    return child_seed


def verify_artifact_chain(artifact_path: Path) -> Tuple[bool, List[str]]:
    """
    Verify complete artifact chain (SBOM, signatures, hashes)

    Args:
        artifact_path: Path to artifact

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    # Check if artifact exists
    if not artifact_path.exists():
        return False, ["Artifact not found"]

    # Check for SBOM
    sbom_path = artifact_path.parent / f"{artifact_path.name}.sbom.json"
    if not sbom_path.exists():
        sbom_path = artifact_path.parent / "sbom.spdx.json"

    if not sbom_path.exists():
        issues.append("No SBOM found")
    else:
        # Verify SBOM
        from .sbom import verify_sbom

        if not verify_sbom(sbom_path, artifact_path.parent):
            issues.append("SBOM verification failed")

    # Check for signatures
    sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
    if not sig_path.exists():
        issues.append("No signature found")
    else:
        # Verify signature
        from .signing import verify_artifact

        if not verify_artifact(artifact_path, sig_path):
            issues.append("Signature verification failed")

    # Check for cosign signature
    cosign_sig = artifact_path.parent / f"{artifact_path.name}.sig"
    if cosign_sig.exists():
        from .sign import cosign_verify

        if not cosign_verify(str(artifact_path)):
            issues.append("Cosign verification failed")

    return len(issues) == 0, issues


def generate_provenance_report(artifact_path: Path) -> Dict[str, Any]:
    """
    Generate comprehensive provenance report for an artifact

    Args:
        artifact_path: Path to artifact

    Returns:
        Provenance report dictionary
    """
    report = {
        "artifact": str(artifact_path),
        "timestamp": DeterministicClock.utcnow().isoformat(),
        "checks": {},
        "metadata": {},
    }

    # Verify artifact chain
    is_valid, issues = verify_artifact_chain(artifact_path)
    report["checks"]["chain_valid"] = is_valid
    report["checks"]["issues"] = issues

    # Add SBOM info
    sbom_path = artifact_path.parent / f"{artifact_path.name}.sbom.json"
    if not sbom_path.exists():
        sbom_path = artifact_path.parent / "sbom.spdx.json"

    if sbom_path.exists():
        with open(sbom_path) as f:
            sbom = json.load(f)

        report["sbom"] = {
            "format": sbom.get("bomFormat", "unknown"),
            "version": sbom.get("specVersion", "unknown"),
            "components": len(sbom.get("components", [])),
            "vulnerabilities": len(sbom.get("vulnerabilities", [])),
        }

    # Add signature info
    sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
    if sig_path.exists():
        with open(sig_path) as f:
            sig = json.load(f)

        report["signature"] = {
            "version": sig.get("version", "unknown"),
            "algorithm": sig.get("spec", {})
            .get("signature", {})
            .get("algorithm", "unknown"),
            "timestamp": sig.get("metadata", {}).get("timestamp", "unknown"),
        }

    # Add file metadata
    if artifact_path.exists():
        stat = artifact_path.stat()
        report["metadata"] = {
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }

        # Calculate hash
        import hashlib

        hasher = hashlib.sha256()
        with open(artifact_path, "rb") as f:
            hasher.update(f.read())
        report["metadata"]["sha256"] = hasher.hexdigest()

    return report


def check_reproducibility(run1_path: Path, run2_path: Path) -> Dict[str, Any]:
    """
    Check if two runs are reproducible

    Args:
        run1_path: Path to first run ledger
        run2_path: Path to second run ledger

    Returns:
        Reproducibility report
    """
    from .ledger import compare_runs

    comparison = compare_runs(run1_path, run2_path)

    report = {
        "reproducible": comparison["identical"],
        "timestamp": DeterministicClock.utcnow().isoformat(),
        "runs": {"run1": str(run1_path), "run2": str(run2_path)},
        "differences": comparison["differences"],
    }

    # Add recommendations
    if not comparison["identical"]:
        report["recommendations"] = []

        for diff in comparison["differences"]:
            if diff["field"] == "pipeline_hash":
                report["recommendations"].append(
                    "Pipeline code has changed between runs"
                )
            elif diff["field"] == "inputs_hash":
                report["recommendations"].append("Input data differs between runs")
            elif diff["field"] == "config_hash":
                report["recommendations"].append("Configuration differs between runs")
            elif diff["field"] == "outputs":
                report["recommendations"].append(
                    "Outputs differ - check for non-deterministic operations"
                )

    return report


def export_provenance_bundle(
    artifact_path: Path, output_path: Optional[Path] = None
) -> Path:
    """
    Export complete provenance bundle as a tar archive

    Args:
        artifact_path: Path to artifact
        output_path: Optional output path for bundle

    Returns:
        Path to created bundle
    """
    import tarfile
    import tempfile

    if output_path is None:
        output_path = artifact_path.parent / f"{artifact_path.name}.provenance.tar.gz"

    with tarfile.open(output_path, "w:gz") as tar:
        # Add the artifact itself
        tar.add(artifact_path, arcname=artifact_path.name)

        # Add SBOM
        sbom_path = artifact_path.parent / f"{artifact_path.name}.sbom.json"
        if not sbom_path.exists():
            sbom_path = artifact_path.parent / "sbom.spdx.json"
        if sbom_path.exists():
            tar.add(sbom_path, arcname=sbom_path.name)

        # Add signatures
        sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
        if sig_path.exists():
            tar.add(sig_path, arcname=sig_path.name)

        # Add cosign signatures
        cosign_sig = artifact_path.parent / f"{artifact_path.name}.sig"
        if cosign_sig.exists():
            tar.add(cosign_sig, arcname=f"cosign_{cosign_sig.name}")

        cosign_cert = artifact_path.parent / f"{artifact_path.name}.pem"
        if cosign_cert.exists():
            tar.add(cosign_cert, arcname=f"cosign_{cosign_cert.name}")

        # Add run ledger if exists
        ledger_path = artifact_path.parent / "run.json"
        if ledger_path.exists():
            tar.add(ledger_path, arcname="run.json")

        # Generate and add provenance report
        report = generate_provenance_report(artifact_path)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f, indent=2)
            temp_report = f.name

        tar.add(temp_report, arcname="provenance_report.json")
        Path(temp_report).unlink()

    logger.info(f"Provenance bundle created: {output_path}")
    return output_path


def import_provenance_bundle(
    bundle_path: Path, extract_to: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Import and verify a provenance bundle

    Args:
        bundle_path: Path to provenance bundle
        extract_to: Optional extraction directory

    Returns:
        Dictionary of extracted file paths
    """
    import tarfile

    if extract_to is None:
        extract_to = bundle_path.parent / bundle_path.stem

    extract_to.mkdir(parents=True, exist_ok=True)

    extracted = {}

    with tarfile.open(bundle_path, "r:gz") as tar:
        for member in tar.getmembers():
            tar.extract(member, extract_to)
            extracted[member.name] = extract_to / member.name

    # Verify the extracted artifact
    artifact_files = [
        f for f in extracted.keys() if not f.endswith((".json", ".sig", ".pem"))
    ]
    if artifact_files:
        artifact_path = extracted[artifact_files[0]]
        is_valid, issues = verify_artifact_chain(artifact_path)

        if not is_valid:
            logger.warning(f"Provenance verification issues: {issues}")

    return extracted
