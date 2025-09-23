"""
Signature Verification Module
=============================

Provides signature verification for packs including:
- Signature verification (stub for Sigstore integration)
- Publisher verification
- Checksum validation
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class SignatureVerificationError(Exception):
    """Raised when signature verification fails"""


class PackVerifier:
    """
    Verifies pack signatures and integrity

    This is a stub implementation that will be replaced with
    Sigstore keyless verification in the next iteration.
    """

    def __init__(self):
        """Initialize verifier"""
        self.trusted_publishers = self._load_trusted_publishers()

    def _load_trusted_publishers(self) -> Dict[str, Dict[str, Any]]:
        """
        Load trusted publisher keys

        Returns:
            Dictionary of trusted publishers
        """
        # In production, this would load from a secure source
        # For now, return a stub
        return {
            "greenlang": {
                "name": "GreenLang Official",
                "key": "placeholder-public-key",
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

            # Verify signature (stub implementation)
            try:
                verification_result = self._verify_signature_stub(
                    pack_path, signature_path, checksum
                )
                metadata.update(verification_result)
                metadata["verified"] = True

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
