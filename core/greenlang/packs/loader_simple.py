"""
Simple pack loader for PR1 - validates manifest and referenced files.
"""

from pathlib import Path
import yaml
from .manifest import PackManifest


def load_manifest(path: str) -> PackManifest:
    """
    Load and validate a pack manifest.

    Args:
        path: Path to pack directory containing pack.yaml

    Returns:
        Validated PackManifest instance

    Raises:
        FileNotFoundError: If pack.yaml or referenced files are missing
        ValueError: If manifest is invalid
    """
    root = Path(path)

    # Load pack.yaml
    manifest_file = root / "pack.yaml"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Missing pack.yaml in {root}")

    # Parse YAML
    data = yaml.safe_load(manifest_file.read_text())

    # Validate with Pydantic
    pm = PackManifest(**data)

    # Check existence of all referenced files
    missing_files = []

    # Check pipelines
    for pipeline in pm.contents.pipelines:
        if not (root / pipeline).exists():
            missing_files.append(f"Pipeline: {pipeline}")

    # Check agents
    for agent in pm.contents.agents:
        if not (root / agent).exists():
            missing_files.append(f"Agent: {agent}")

    # Check datasets
    for dataset in pm.contents.datasets:
        if not (root / dataset).exists():
            missing_files.append(f"Dataset: {dataset}")

    # Check reports
    for report in pm.contents.reports:
        if not (root / report).exists():
            missing_files.append(f"Report: {report}")

    # Check tests
    for test in pm.tests:
        # Tests can be glob patterns, so check if any files match
        test_files = list(root.glob(test))
        if not test_files:
            missing_files.append(f"Test pattern: {test}")

    # Check card file
    if not (root / pm.card).exists():
        missing_files.append(f"Card: {pm.card}")

    # Check SBOM if specified
    if pm.security.sbom and not (root / pm.security.sbom).exists():
        missing_files.append(f"SBOM: {pm.security.sbom}")

    # Check signature files
    for sig in pm.security.signatures:
        if not (root / sig).exists():
            missing_files.append(f"Signature: {sig}")

    # Raise error if any files are missing
    if missing_files:
        error_msg = f"Missing files in pack {pm.name}:\n"
        for missing in missing_files:
            error_msg += f"  - {missing}\n"
        raise FileNotFoundError(error_msg)

    return pm


def validate_pack(path: str) -> tuple[bool, list[str]]:
    """
    Validate a pack directory.

    Args:
        path: Path to pack directory

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    try:
        manifest = load_manifest(path)
        return True, []
    except FileNotFoundError as e:
        errors.append(str(e))
    except ValueError as e:
        errors.append(f"Invalid manifest: {e}")
    except Exception as e:
        errors.append(f"Unexpected error: {e}")

    return False, errors
