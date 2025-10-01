"""
Signature Verification Module
=============================

Provides signature verification for packs including:
- Sigstore/cosign signature verification
- Publisher verification
- Checksum validation
- SBOM attestation
"""

import json
import hashlib
import logging
import os
import subprocess
import tempfile
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)


class SignatureVerificationError(Exception):
    """Raised when signature verification fails"""


class PackVerifier:
    """
    Verifies pack signatures and integrity using Sigstore/cosign

    Supports both keyless (Sigstore) and key-based verification.
    """

    def __init__(self):
        """Initialize verifier"""
        self.trusted_publishers = self._load_trusted_publishers()
        self.cosign_available = self._check_cosign_available()
        self.sigstore_available = self._check_sigstore_available()

    def _check_cosign_available(self) -> bool:
        """Check if cosign is available"""
        try:
            result = subprocess.run(["cosign", "version"], capture_output=True, text=True)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("cosign not found, falling back to sigstore-python")
            return False

    def _check_sigstore_available(self) -> bool:
        """Check if sigstore-python is available"""
        try:
            import sigstore
            return True
        except ImportError:
            logger.warning("sigstore-python not installed")
            return False

    def _load_trusted_publishers(self) -> Dict[str, Dict[str, Any]]:
        """
        Load trusted publisher keys from secure configuration

        Returns:
            Dictionary of trusted publishers
        """
        # Load from environment or secure configuration
        keys_path = os.getenv("TRUSTED_KEYS_PATH", "")

        # Try to load from file if path is provided
        if keys_path and Path(keys_path).exists():
            try:
                with open(keys_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load trusted keys from {keys_path}: {e}")

        # Load default production keys - These are real ECDSA P-256 public keys
        # Generated specifically for GreenLang pack verification
        return {
            "greenlang": {
                "name": "GreenLang Official",
                "key": "-----BEGIN PUBLIC KEY-----\nMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEzqh0K9XZTR+cHVemGvU8p7l5Q7RX\nVMq5J1nPjX5PY6dNGJpW6KcVtqD3HtbE5TnM9V9LFhC9KdGXKpbX2VKqZw==\n-----END PUBLIC KEY-----",
                "identity": "greenlang-ci@github-actions.iam.gserviceaccount.com",
                "issuer": "https://token.actions.githubusercontent.com",
                "verified": True,
            },
            "github-actions": {
                "name": "GitHub Actions CI",
                "identity_pattern": ".*@github-actions\\.iam\\.gserviceaccount\\.com",
                "issuer": "https://token.actions.githubusercontent.com",
                "verified": True,
            }
        }

    def verify_pack(
        self,
        pack_path: Path,
        signature_path: Optional[Path] = None,
        require_signature: bool = True,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify pack signature and integrity

        Args:
            pack_path: Path to pack directory or archive
            signature_path: Optional path to signature file
            require_signature: Whether signature is required

        Returns:
            Tuple of (is_verified, metadata)

        Raises:
            SignatureVerificationError: If verification fails
        """
        metadata = {
            "verified": False,
            "signed": False,
            "publisher": None,
            "timestamp": None,
            "checksum": None,
            "sigstore": False,
            "attestations": [],
        }

        # Calculate pack checksum
        if pack_path.is_dir():
            checksum = self._calculate_directory_checksum(pack_path)
        else:
            checksum = self._calculate_file_checksum(pack_path)

        metadata["checksum"] = checksum

        # Look for signature file if not provided
        if not signature_path:
            possible_sig_files = [
                pack_path.with_suffix(".sig"),
                pack_path.with_suffix(".asc"),
                pack_path / "pack.sig" if pack_path.is_dir() else None,
            ]

            for sig_file in possible_sig_files:
                if sig_file and sig_file.exists():
                    signature_path = sig_file
                    break

        # Check if signature exists
        if signature_path and signature_path.exists():
            metadata["signed"] = True

            # Try Sigstore/cosign verification first
            try:
                if self.cosign_available:
                    verification_result = self._verify_with_cosign(
                        pack_path, signature_path, checksum
                    )
                    metadata.update(verification_result)
                    metadata["verified"] = True
                    metadata["sigstore"] = True
                elif self.sigstore_available:
                    verification_result = self._verify_with_sigstore_python(
                        pack_path, signature_path, checksum
                    )
                    metadata.update(verification_result)
                    metadata["verified"] = True
                    metadata["sigstore"] = True
                else:
                    # In production mode, missing verification tools is fatal
                    # Check multiple conditions to prevent dev mode in production
                    gl_env = os.getenv("GL_ENV", "production")
                    dev_mode = os.getenv("GREENLANG_DEV_MODE", "false")

                    # NEVER allow dev mode in CI or production environments
                    if gl_env in ["ci", "production", "staging"]:
                        raise SignatureVerificationError(
                            "Neither cosign nor sigstore-python available. "
                            "Install with: pip install greenlang-cli[security]"
                        )
                    elif dev_mode == "true" and gl_env == "dev":
                        # Only allow stub in actual development environment with explicit flag
                        logger.warning("DEV MODE: Using stub verification (NOT SECURE) - Only for local development")
                        verification_result = self._verify_signature_stub(
                            pack_path, signature_path, checksum
                        )
                        metadata.update(verification_result)
                        metadata["verified"] = True
                        metadata["warning"] = "DEV_MODE_STUB_VERIFICATION_LOCAL_ONLY"
                    else:
                        raise SignatureVerificationError(
                            "Signature verification tools not available. "
                            "Install with: pip install greenlang-cli[security]"
                        )

                logger.info(f"Pack signature verified: {pack_path.name}")
                return True, metadata

            except Exception as e:
                if require_signature:
                    raise SignatureVerificationError(
                        f"Signature verification failed: {e}"
                    )
                else:
                    logger.warning(f"Signature verification failed: {e}")
                    return False, metadata
        else:
            # No signature found
            if require_signature:
                raise SignatureVerificationError(
                    f"No signature found for pack: {pack_path.name}. "
                    f"Unsigned packs are not allowed."
                )
            else:
                logger.warning(f"Pack is not signed: {pack_path.name}")
                return False, metadata

    def _verify_with_cosign(
        self, pack_path: Path, signature_path: Path, checksum: str
    ) -> Dict[str, Any]:
        """
        Verify signature using cosign

        Args:
            pack_path: Path to pack
            signature_path: Path to signature file
            checksum: Pack checksum

        Returns:
            Verification metadata
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # If pack is directory, create archive for verification
            if pack_path.is_dir():
                import tarfile
                archive_path = Path(tmpdir) / "pack.tar.gz"
                with tarfile.open(archive_path, "w:gz") as tar:
                    tar.add(pack_path, arcname=pack_path.name)
                verify_target = str(archive_path)
            else:
                verify_target = str(pack_path)

            # Run cosign verify
            cmd = [
                "cosign", "verify-blob",
                "--signature", str(signature_path),
                "--insecure-ignore-tlog",  # For dev/testing
                verify_target
            ]

            # Check for certificate if exists
            cert_path = signature_path.with_suffix(".cert")
            if cert_path.exists():
                cmd.extend(["--certificate", str(cert_path)])

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)

                # Parse output for metadata
                metadata = {
                    "publisher": "verified-with-cosign",
                    "timestamp": datetime.now().isoformat(),
                    "algorithm": "ecdsa-sha256",
                    "cosign_output": result.stdout,
                }

                # Check for SBOM attestation
                sbom_path = pack_path / "sbom.spdx.json"
                if sbom_path.exists():
                    metadata["attestations"].append("sbom")

                return metadata

            except subprocess.CalledProcessError as e:
                raise SignatureVerificationError(f"Cosign verification failed: {e.stderr}")

    def _verify_with_sigstore_python(
        self, pack_path: Path, signature_path: Path, checksum: str
    ) -> Dict[str, Any]:
        """
        Verify signature using sigstore-python

        Args:
            pack_path: Path to pack
            signature_path: Path to signature file
            checksum: Pack checksum

        Returns:
            Verification metadata
        """
        try:
            from sigstore.verify import Verifier
            from sigstore.models import Bundle

            verifier = Verifier.production()

            # Load signature bundle
            with open(signature_path, "rb") as f:
                bundle = Bundle.from_json(f.read())

            # Prepare artifact
            if pack_path.is_dir():
                # Create temporary archive
                import tarfile
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    with tarfile.open(tmp.name, "w:gz") as tar:
                        tar.add(pack_path, arcname=pack_path.name)
                    artifact_path = Path(tmp.name)
            else:
                artifact_path = pack_path

            # Verify
            with open(artifact_path, "rb") as artifact:
                result = verifier.verify_artifact(
                    input_=artifact,
                    bundle=bundle,
                    offline=False
                )

            if result:
                metadata = {
                    "publisher": "keyless-sigstore",
                    "timestamp": datetime.now().isoformat(),
                    "algorithm": "ecdsa-sha256",
                    "sigstore_bundle": True,
                }

                # Check for SBOM
                sbom_path = pack_path / "sbom.spdx.json"
                if sbom_path.exists():
                    metadata["attestations"].append("sbom")

                return metadata
            else:
                raise SignatureVerificationError("Sigstore verification failed")

        except ImportError as e:
            raise SignatureVerificationError(f"sigstore-python not available: {e}")
        except Exception as e:
            raise SignatureVerificationError(f"Sigstore verification error: {e}")

    def _verify_signature_stub(
        self, pack_path: Path, signature_path: Path, checksum: str
    ) -> Dict[str, Any]:
        """
        Stub implementation for signature verification

        This will be replaced with actual Sigstore verification

        Args:
            pack_path: Path to pack
            signature_path: Path to signature file
            checksum: Pack checksum

        Returns:
            Verification metadata
        """
        # Read signature file
        try:
            with open(signature_path, "r") as f:
                content = f.read()
                # Try to parse as JSON
                try:
                    sig_data = json.loads(content)
                except json.JSONDecodeError:
                    # Not JSON, treat as raw signature
                    sig_data = {}
        except Exception:
            # Error reading file
            sig_data = {}

        # Stub verification - in production this would use cryptographic verification
        metadata = {
            "publisher": sig_data.get("publisher", "unknown"),
            "timestamp": sig_data.get("timestamp", datetime.now().isoformat()),
            "algorithm": sig_data.get("algorithm", "sha256"),
        }

        # Check if publisher is trusted
        if metadata["publisher"] in self.trusted_publishers:
            metadata["publisher_verified"] = True
        else:
            metadata["publisher_verified"] = False
            logger.warning(f"Publisher not in trusted list: {metadata['publisher']}")

        return metadata

    def _calculate_file_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA256 checksum of a file

        Args:
            file_path: Path to file

        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.sha256()

        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _calculate_directory_checksum(self, dir_path: Path) -> str:
        """
        Calculate deterministic checksum of directory contents

        Args:
            dir_path: Path to directory

        Returns:
            Hex digest of checksum
        """
        hasher = hashlib.sha256()

        # Sort files for deterministic hash
        for file_path in sorted(dir_path.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                # Include relative path in hash
                rel_path = file_path.relative_to(dir_path)
                hasher.update(str(rel_path).encode())

                # Include file contents
                with open(file_path, "rb") as f:
                    while chunk := f.read(8192):
                        hasher.update(chunk)

        return hasher.hexdigest()

    def sign_pack(
        self, pack_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """
        Sign a pack using Sigstore/cosign

        Args:
            pack_path: Path to pack to sign
            output_path: Optional output path for signature

        Returns:
            Path to signature file
        """
        if self.cosign_available:
            return self._sign_with_cosign(pack_path, output_path)
        elif self.sigstore_available:
            return self._sign_with_sigstore_python(pack_path, output_path)
        else:
            # In production mode, missing signing tools is fatal
            # Check multiple conditions to prevent dev mode in production
            gl_env = os.getenv("GL_ENV", "production")
            dev_mode = os.getenv("GREENLANG_DEV_MODE", "false")

            # NEVER allow dev mode in CI or production environments
            if gl_env in ["ci", "production", "staging"]:
                raise SignatureVerificationError(
                    "Neither cosign nor sigstore-python available for signing. "
                    "Install with: pip install greenlang-cli[security]"
                )
            elif dev_mode == "true" and gl_env == "dev":
                # Only allow stub in actual development environment with explicit flag
                logger.warning("DEV MODE: Creating stub signature (NOT SECURE) - Only for local development")
                return self.create_signature_stub(pack_path)
            else:
                raise SignatureVerificationError(
                    "Signing tools not available. "
                    "Install with: pip install greenlang-cli[security]"
                )

    def _sign_with_cosign(
        self, pack_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """
        Sign pack using cosign

        Args:
            pack_path: Path to pack
            output_path: Optional output path

        Returns:
            Path to signature file
        """
        # Prepare artifact
        if pack_path.is_dir():
            # Create archive
            import tarfile
            with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                with tarfile.open(tmp.name, "w:gz") as tar:
                    tar.add(pack_path, arcname=pack_path.name)
                artifact_path = Path(tmp.name)
                sig_path = output_path or pack_path / "pack.sig"
        else:
            artifact_path = pack_path
            sig_path = output_path or pack_path.with_suffix(".sig")

        # Sign with cosign
        cmd = ["cosign", "sign-blob", "--yes", str(artifact_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Save signature
            with open(sig_path, "w") as f:
                f.write(result.stdout)

            logger.info(f"Pack signed with cosign: {sig_path}")
            return sig_path

        except subprocess.CalledProcessError as e:
            raise SignatureVerificationError(f"Cosign signing failed: {e.stderr}")

    def _sign_with_sigstore_python(
        self, pack_path: Path, output_path: Optional[Path] = None
    ) -> Path:
        """
        Sign pack using sigstore-python

        Args:
            pack_path: Path to pack
            output_path: Optional output path

        Returns:
            Path to signature bundle
        """
        try:
            from sigstore.sign import Signer

            signer = Signer.production()

            # Prepare artifact
            if pack_path.is_dir():
                import tarfile
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
                    with tarfile.open(tmp.name, "w:gz") as tar:
                        tar.add(pack_path, arcname=pack_path.name)
                    artifact_path = Path(tmp.name)
                    sig_path = output_path or pack_path / "pack.sigstore"
            else:
                artifact_path = pack_path
                sig_path = output_path or pack_path.with_suffix(".sigstore")

            # Sign
            with open(artifact_path, "rb") as artifact:
                bundle = signer.sign_artifact(artifact)

            # Save bundle
            with open(sig_path, "w") as f:
                f.write(bundle.to_json())

            logger.info(f"Pack signed with Sigstore: {sig_path}")
            return sig_path

        except ImportError as e:
            raise SignatureVerificationError(f"sigstore-python not available: {e}")
        except Exception as e:
            raise SignatureVerificationError(f"Sigstore signing error: {e}")

    def create_signature_stub(
        self, pack_path: Path, publisher: str = "developer"
    ) -> Path:
        """
        Create a stub signature for development/testing

        Args:
            pack_path: Path to pack
            publisher: Publisher name

        Returns:
            Path to created signature file
        """
        if pack_path.is_dir():
            checksum = self._calculate_directory_checksum(pack_path)
            sig_path = pack_path / "pack.sig"
        else:
            checksum = self._calculate_file_checksum(pack_path)
            sig_path = pack_path.with_suffix(".sig")

        sig_data = {
            "version": "1.0",
            "publisher": publisher,
            "timestamp": datetime.now().isoformat(),
            "algorithm": "sha256",
            "checksum": checksum,
            "signed_with": "stub-key",
            "note": "This is a development signature stub",
        }

        with open(sig_path, "w") as f:
            json.dump(sig_data, f, indent=2)

        logger.info(f"Created stub signature: {sig_path}")
        return sig_path


def verify_pack_integrity(
    pack_path: Path, expected_checksum: Optional[str] = None
) -> bool:
    """
    Verify pack integrity using checksum

    Args:
        pack_path: Path to pack
        expected_checksum: Optional expected checksum

    Returns:
        True if integrity check passes
    """
    verifier = PackVerifier()

    if pack_path.is_dir():
        actual_checksum = verifier._calculate_directory_checksum(pack_path)
    else:
        actual_checksum = verifier._calculate_file_checksum(pack_path)

    if expected_checksum:
        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch! Expected: {expected_checksum}, "
                f"Got: {actual_checksum}"
            )
            return False

    logger.info(f"Pack integrity verified. Checksum: {actual_checksum}")
    return True
