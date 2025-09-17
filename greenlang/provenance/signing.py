"""
Signature Verification for GreenLang Packs
==========================================

Provides cryptographic signature verification for pack integrity and authenticity.
SECURITY: Default-deny - unsigned packs are rejected unless explicitly allowed.
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64

logger = logging.getLogger(__name__)


class SignatureVerifier(ABC):
    """
    Abstract base class for signature verification

    All verifiers must implement the verify method to check
    artifact signatures against a certificate chain.
    """

    @abstractmethod
    def verify(self, artifact: bytes, signature: bytes,
               cert_chain: Optional[bytes] = None) -> bool:
        """
        Verify artifact signature

        Args:
            artifact: The artifact bytes to verify
            signature: The signature bytes
            cert_chain: Optional certificate chain for verification

        Returns:
            True if signature is valid, False otherwise
        """
        pass

    @abstractmethod
    def get_verifier_info(self) -> Dict[str, Any]:
        """Get information about this verifier"""
        pass


class DevKeyVerifier(SignatureVerifier):
    """
    Development key verifier for testing

    SECURITY WARNING: This is for development/testing only!
    Uses ephemeral keys generated at runtime - no hardcoded keys.
    """

    def __init__(self):
        """Initialize with ephemeral keypair for testing"""
        logger.warning("⚠️  DevKeyVerifier initialized - FOR DEVELOPMENT ONLY!")
        logger.warning("⚠️  This verifier uses ephemeral keys and provides no real security!")

        # Generate ephemeral keypair at runtime - no hardcoded keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()

        # Store public key in PEM format for distribution
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    def sign(self, artifact: bytes) -> bytes:
        """
        Sign an artifact (for testing only)

        Args:
            artifact: The artifact to sign

        Returns:
            Base64-encoded signature
        """
        # Hash the artifact
        digest = hashlib.sha256(artifact).digest()

        # Sign the hash
        signature = self.private_key.sign(
            digest,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )

        # Return base64-encoded signature
        return base64.b64encode(signature)

    def verify(self, artifact: bytes, signature: bytes,
               cert_chain: Optional[bytes] = None) -> bool:
        """
        Verify artifact signature using dev key

        Args:
            artifact: The artifact bytes to verify
            signature: The base64-encoded signature bytes
            cert_chain: Ignored for dev verifier

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Decode base64 signature
            sig_bytes = base64.b64decode(signature)

            # Hash the artifact
            digest = hashlib.sha256(artifact).digest()

            # Verify signature
            self.public_key.verify(
                sig_bytes,
                digest,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )

            logger.info("✅ Signature verified (DEV KEY)")
            return True

        except InvalidSignature:
            logger.error("❌ Invalid signature")
            return False
        except Exception as e:
            logger.error(f"❌ Signature verification failed: {e}")
            return False

    def get_verifier_info(self) -> Dict[str, Any]:
        """Get information about this verifier"""
        return {
            "type": "DevKeyVerifier",
            "warning": "FOR DEVELOPMENT ONLY - NO REAL SECURITY",
            "public_key_fingerprint": hashlib.sha256(self.public_key_pem).hexdigest()[:16]
        }


class SigstoreVerifier(SignatureVerifier):
    """
    Production signature verifier using Sigstore

    Sigstore provides keyless signing and transparency for software artifacts.
    This will be implemented in Week 1 of the security sprint.
    """

    def __init__(self):
        """Initialize Sigstore verifier"""
        raise NotImplementedError(
            "SigstoreVerifier will be implemented in Week 1. "
            "Use DevKeyVerifier for testing or pass --allow-unsigned flag."
        )

    def verify(self, artifact: bytes, signature: bytes,
               cert_chain: Optional[bytes] = None) -> bool:
        """
        Verify artifact using Sigstore

        This will:
        1. Verify the signature against Fulcio certificate
        2. Check transparency log inclusion in Rekor
        3. Validate certificate chain to Sigstore root
        """
        raise NotImplementedError("Sigstore verification coming in Week 1")

    def get_verifier_info(self) -> Dict[str, Any]:
        """Get information about this verifier"""
        return {
            "type": "SigstoreVerifier",
            "status": "NOT_IMPLEMENTED",
            "planned": "Week 1 of security sprint"
        }


class UnsignedPackError(Exception):
    """Raised when attempting to install unsigned pack without override"""
    pass


def create_verifier(verifier_type: str = "dev") -> SignatureVerifier:
    """
    Factory function to create appropriate verifier

    Args:
        verifier_type: Type of verifier ("dev" or "sigstore")

    Returns:
        SignatureVerifier instance
    """
    if verifier_type == "dev":
        return DevKeyVerifier()
    elif verifier_type == "sigstore":
        return SigstoreVerifier()
    else:
        raise ValueError(f"Unknown verifier type: {verifier_type}")


def verify_pack_signature(pack_path: Path,
                         verifier: Optional[SignatureVerifier] = None) -> Tuple[bool, str]:
    """
    Verify a pack's signature

    Args:
        pack_path: Path to pack file
        verifier: Optional verifier to use (defaults to DevKeyVerifier)

    Returns:
        (is_valid, message) tuple
    """
    if not verifier:
        verifier = DevKeyVerifier()

    # Look for signature file
    sig_path = Path(str(pack_path) + ".sig")

    if not sig_path.exists():
        return False, f"No signature file found at {sig_path}"

    try:
        # Read pack and signature
        pack_bytes = pack_path.read_bytes()
        sig_bytes = sig_path.read_bytes()

        # Verify signature
        if verifier.verify(pack_bytes, sig_bytes):
            return True, "Signature verified successfully"
        else:
            return False, "Signature verification failed"

    except Exception as e:
        return False, f"Error verifying signature: {e}"


def sign_pack(pack_path: Path, signer: Optional[DevKeyVerifier] = None) -> Path:
    """
    Sign a pack file (development only)

    Args:
        pack_path: Path to pack file
        signer: Optional DevKeyVerifier to use

    Returns:
        Path to signature file
    """
    if not signer:
        signer = DevKeyVerifier()

    # Read pack bytes
    pack_bytes = pack_path.read_bytes()

    # Generate signature
    signature = signer.sign(pack_bytes)

    # Write signature file
    sig_path = Path(str(pack_path) + ".sig")
    sig_path.write_bytes(signature)

    logger.info(f"✅ Pack signed: {sig_path}")
    return sig_path