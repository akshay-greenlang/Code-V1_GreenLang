"""
Cosign signing wrapper
"""

import subprocess
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def cosign_sign(path: str, recursive: bool = True) -> None:
    """
    Sign artifacts using cosign with keyless (OIDC) if available

    Args:
        path: Path to artifact or directory to sign
        recursive: If True, sign all files recursively
    """
    if not _has_cosign():
        logger.warning("cosign not installed, skipping signing")
        return

    path = Path(path)

    # Build cosign command
    cmd = ["cosign", "sign-blob"]

    # Use keyless signing if available (requires OIDC)
    if _supports_keyless():
        cmd.extend(["--yes"])  # Auto-confirm
        logger.info("Using keyless signing with OIDC")
    else:
        # Fall back to key-based signing if configured
        key_path = _get_signing_key()
        if key_path:
            cmd.extend(["--key", str(key_path)])
            logger.info(f"Using key-based signing with {key_path}")
        else:
            logger.warning("No signing key configured, skipping signing")
            return

    # Sign files
    if path.is_file():
        _sign_file(path, cmd)
    elif path.is_dir() and recursive:
        # Sign all files in directory
        for file_path in path.rglob("*"):
            if file_path.is_file() and not _should_skip(file_path):
                _sign_file(file_path, cmd)
    elif path.is_dir():
        # Create and sign directory manifest
        manifest_path = _create_directory_manifest(path)
        _sign_file(manifest_path, cmd)


def cosign_verify(path: str, recursive: bool = True) -> bool:
    """
    Verify signatures using cosign

    Args:
        path: Path to artifact or directory
        recursive: If True, verify all files recursively

    Returns:
        True if all signatures are valid
    """
    if not _has_cosign():
        logger.warning("cosign not installed, cannot verify")
        return False

    path = Path(path)

    # Build verify command
    cmd = ["cosign", "verify-blob"]

    # Add certificate identity if configured
    identity = _get_verify_identity()
    if identity:
        cmd.extend(["--certificate-identity", identity])
        cmd.extend(["--certificate-oidc-issuer", _get_oidc_issuer()])

    # Verify files
    if path.is_file():
        return _verify_file(path, cmd)
    elif path.is_dir() and recursive:
        # Verify all signed files
        all_valid = True
        for sig_path in path.rglob("*.sig"):
            file_path = sig_path.with_suffix("")
            if file_path.exists():
                if not _verify_file(file_path, cmd):
                    all_valid = False
        return all_valid
    elif path.is_dir():
        # Verify directory manifest
        manifest_path = path / ".manifest.json"
        if manifest_path.exists():
            return _verify_file(manifest_path, cmd)

    return False


def _has_cosign() -> bool:
    """Check if cosign is installed"""
    try:
        result = subprocess.run(["cosign", "version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _supports_keyless() -> bool:
    """Check if keyless signing is available"""
    import os

    # Check for OIDC/workload identity
    return any(
        [
            os.getenv("COSIGN_EXPERIMENTAL"),
            os.getenv("SIGSTORE_ID_TOKEN"),
            os.getenv("GITHUB_ACTIONS"),  # GitHub OIDC
            os.getenv("GOOGLE_SERVICE_ACCOUNT"),  # GCP workload identity
            os.getenv("AWS_ROLE_ARN"),  # AWS IRSA
        ]
    )


def _get_signing_key() -> Optional[Path]:
    """Get signing key path from environment or config"""
    import os

    key_path = os.getenv("COSIGN_KEY")
    if key_path:
        return Path(key_path)

    # Check standard locations
    locations = [
        Path.home() / ".cosign" / "key.pem",
        Path.home() / ".greenlang" / "signing" / "key.pem",
    ]

    for loc in locations:
        if loc.exists():
            return loc

    return None


def _get_verify_identity() -> Optional[str]:
    """Get certificate identity for verification"""
    import os

    return os.getenv("COSIGN_CERTIFICATE_IDENTITY")


def _get_oidc_issuer() -> str:
    """Get OIDC issuer for verification"""
    import os

    return os.getenv(
        "COSIGN_CERTIFICATE_OIDC_ISSUER", "https://oauth2.sigstore.dev/auth"
    )


def _sign_file(file_path: Path, base_cmd: list) -> bool:
    """Sign a single file"""
    try:
        # Sign the file
        cmd = base_cmd.copy()
        cmd.append(str(file_path))

        # Output signature to .sig file
        sig_path = file_path.parent / f"{file_path.name}.sig"
        cmd.extend(["--output-signature", str(sig_path)])

        # If certificate is generated, save it
        cert_path = file_path.parent / f"{file_path.name}.pem"
        cmd.extend(["--output-certificate", str(cert_path)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Signed: {file_path}")
            return True
        else:
            logger.error(f"Failed to sign {file_path}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error signing {file_path}: {e}")
        return False


def _verify_file(file_path: Path, base_cmd: list) -> bool:
    """Verify a single file signature"""
    try:
        sig_path = file_path.parent / f"{file_path.name}.sig"
        cert_path = file_path.parent / f"{file_path.name}.pem"

        if not sig_path.exists():
            logger.warning(f"No signature found for {file_path}")
            return False

        cmd = base_cmd.copy()
        cmd.extend(["--signature", str(sig_path)])

        if cert_path.exists():
            cmd.extend(["--certificate", str(cert_path)])

        cmd.append(str(file_path))

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Verified: {file_path}")
            return True
        else:
            logger.error(f"Verification failed for {file_path}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error verifying {file_path}: {e}")
        return False


def _should_skip(file_path: Path) -> bool:
    """Check if file should be skipped for signing"""
    skip_patterns = [
        "*.sig",
        "*.pem",
        "*.key",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        ".git",
        ".env",
    ]

    for pattern in skip_patterns:
        if pattern in str(file_path):
            return True

    return False


def _create_directory_manifest(directory: Path) -> Path:
    """Create a manifest of directory contents for signing"""
    import hashlib

    manifest = {"version": "1.0", "directory": str(directory), "files": []}

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and not _should_skip(file_path):
            hasher = hashlib.sha256()
            with open(file_path, "rb") as f:
                hasher.update(f.read())

            manifest["files"].append(
                {
                    "path": str(file_path.relative_to(directory)),
                    "size": file_path.stat().st_size,
                    "sha256": hasher.hexdigest(),
                }
            )

    manifest_path = directory / ".manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return manifest_path


# Re-export for compatibility
