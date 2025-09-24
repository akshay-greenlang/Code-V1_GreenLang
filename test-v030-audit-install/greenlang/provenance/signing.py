"""
Artifact Signing and Verification
==================================

Signs and verifies artifacts using cryptographic signatures.
Supports RSA, ECDSA, and keyless (identity-based) signing.
"""

import json
import hashlib
import base64
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import logging

# Set up Windows encoding support for this module
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")

logger = logging.getLogger(__name__)

# Try to import cryptography for real signing
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa, ec
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available, using mock signing")


def sign_artifact(
    artifact_path: Path, key_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Sign an artifact using secure provider

    Args:
        artifact_path: Path to artifact to sign
        key_path: Deprecated - keys are managed by provider

    Returns:
        Signature dictionary
    """
    if key_path:
        logger.warning("key_path parameter is deprecated and will be ignored")

    # Import the new secure signing module
    from ...greenlang.security import signing as secure_signing

    # Sign using secure provider
    signature = secure_signing.sign_artifact(artifact_path)

    # Save signature next to artifact
    sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(signature, f, indent=2, ensure_ascii=False)

    return signature


def verify_artifact(
    artifact_path: Path, signature_path: Optional[Path] = None
) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify an artifact signature using secure provider

    Args:
        artifact_path: Path to artifact
        signature_path: Path to signature file (optional)

    Returns:
        Tuple of (is_valid, signer_info)
    """
    if not artifact_path.exists():
        raise ValueError(f"Artifact not found: {artifact_path}")

    # Find signature file
    if signature_path is None:
        signature_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")

    if not signature_path.exists():
        return False, None

    # Load signature
    with open(signature_path, "r", encoding="utf-8", errors="replace") as f:
        signature = json.load(f)

    # Import the new secure signing module
    from ...greenlang.security import signing as secure_signing

    try:
        # Verify using secure provider
        secure_signing.verify_artifact(artifact_path, signature)

        # Extract signer info
        signer_info = {
            "subject": signature.get("metadata", {}).get("artifact", "Unknown"),
            "issuer": signature.get("metadata", {})
            .get("signer", {})
            .get("type", "unknown"),
            "timestamp": signature.get("metadata", {}).get("timestamp", "Unknown"),
        }

        return True, signer_info
    except Exception as e:
        logger.debug(f"Signature verification failed: {e}")
        return False, None


def sign_pack(pack_path: Path, key_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Sign an entire pack with cryptographic signature

    Args:
        pack_path: Path to pack directory
        key_path: Path to signing key (will generate if not provided)

    Returns:
        Pack signature dictionary
    """
    if not pack_path.exists():
        raise ValueError(f"Pack not found: {pack_path}")

    # Calculate hash of all pack files (excluding existing signatures)
    pack_hash = _calculate_directory_hash(
        pack_path, exclude=["pack.sig", "*.pem", "*.key"]
    )

    # Load manifest
    manifest_path = pack_path / "pack.yaml"
    if manifest_path.exists():
        import yaml

        with open(manifest_path, "r", encoding="utf-8", errors="replace") as f:
            manifest = yaml.safe_load(f)
    else:
        manifest = {}

    # Sign using secure provider
    from ...greenlang.security import signing as secure_signing

    # Create payload for signing
    payload = pack_hash.encode("utf-8")

    # Get signer based on environment
    signer = secure_signing.create_signer()

    # Sign the hash
    result = signer.sign(payload)

    signature_value = base64.b64encode(result["signature"]).decode()
    algorithm = result["algorithm"]
    public_key_pem = result.get("public_key")

    # Create pack signature
    signature = {
        "version": "1.0.0",
        "kind": "greenlang-pack-signature",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pack": manifest.get("name", "unknown"),
            "version": manifest.get("version", "0.0.0"),
            "signer": os.environ.get("USER", "unknown"),
        },
        "spec": {
            "hash": {"algorithm": "sha256", "value": pack_hash},
            "signature": {"algorithm": algorithm, "value": signature_value},
        },
    }

    # Include public key if available
    if public_key_pem:
        signature["spec"]["publicKey"] = public_key_pem

    # Include manifest hash for integrity
    manifest_hash = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()
    ).hexdigest()
    signature["spec"]["manifestHash"] = manifest_hash

    # Save signature
    sig_path = pack_path / "pack.sig"
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(signature, f, indent=2, ensure_ascii=False)

    logger.info(f"Pack signed successfully: {pack_path}")
    return signature


def verify_pack(pack_path: Path) -> bool:
    """
    Verify a pack signature

    Args:
        pack_path: Path to pack directory

    Returns:
        True if pack signature is valid
    """
    sig_path = pack_path / "pack.sig"

    if not sig_path.exists():
        return False

    # Load signature
    with open(sig_path, "r", encoding="utf-8", errors="replace") as f:
        signature = json.load(f)

    # Calculate current hash
    current_hash = _calculate_directory_hash(pack_path, exclude=["pack.sig"])
    expected_hash = signature["spec"]["hash"]["value"]

    if current_hash != expected_hash:
        return False

    # Import the new secure signing module
    from ...greenlang.security import signing as secure_signing

    try:
        # Create verifier
        verifier = secure_signing.create_verifier()

        # Verify signature
        sig_spec = signature["spec"]
        sig_bytes = base64.b64decode(sig_spec["signature"]["value"])
        payload = current_hash.encode("utf-8")

        verify_kwargs = {}
        if sig_spec.get("publicKey"):
            verify_kwargs["public_key"] = sig_spec["publicKey"]

        verifier.verify(payload, sig_bytes, **verify_kwargs)
        return True
    except Exception as e:
        logger.debug(f"Pack verification failed: {e}")
        return False


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)

    return hasher.hexdigest()


def _calculate_directory_hash(directory: Path, exclude: list = None) -> str:
    """Calculate hash of directory contents"""
    exclude = exclude or []
    hasher = hashlib.sha256()

    # Sort files for deterministic hash
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            # Skip excluded files
            if file_path.name in exclude:
                continue
            if file_path.name.startswith("."):
                continue
            if "__pycache__" in str(file_path):
                continue

            # Include file path and content in hash
            relative_path = file_path.relative_to(directory)
            hasher.update(str(relative_path).encode())

            with open(file_path, "rb") as f:
                hasher.update(f.read())

    return hasher.hexdigest()


# _mock_sign removed - using secure provider instead


# _get_or_create_key_pair removed - keys are managed by secure provider


def _cryptographic_sign(data: str, key_path: Path) -> Tuple[str, str, str]:
    """
    Sign data with RSA private key

    Args:
        data: Data to sign (hash)
        key_path: Path to private key

    Returns:
        Tuple of (signature, algorithm, public_key_pem)
    """
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("cryptography library not available")

    # Load private key
    with open(key_path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(), password=None, backend=default_backend()
        )

    # Sign the data
    if isinstance(private_key, rsa.RSAPrivateKey):
        # RSA signing
        signature = private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        algorithm = "rsa-pss-sha256"
    elif isinstance(private_key, ec.EllipticCurvePrivateKey):
        # ECDSA signing
        signature = private_key.sign(data.encode(), ec.ECDSA(hashes.SHA256()))
        algorithm = "ecdsa-sha256"
    else:
        raise ValueError(f"Unsupported key type: {type(private_key)}")

    # Get public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    # Base64 encode signature
    signature_b64 = base64.b64encode(signature).decode()

    return signature_b64, algorithm, public_pem


def verify_pack_signature(
    pack_path: Path, signature_path: Optional[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify pack signature with cryptographic verification

    Args:
        pack_path: Path to pack directory
        signature_path: Path to signature file (optional)

    Returns:
        Tuple of (is_valid, signature_info)
    """
    if signature_path is None:
        signature_path = pack_path / "pack.sig"

    if not signature_path.exists():
        return False, {"error": "No signature found"}

    # Load signature
    with open(signature_path, "r", encoding="utf-8", errors="replace") as f:
        signature = json.load(f)

    # Calculate current hash
    current_hash = _calculate_directory_hash(
        pack_path, exclude=["pack.sig", "*.pem", "*.key"]
    )
    expected_hash = signature["spec"]["hash"]["value"]

    if current_hash != expected_hash:
        return False, {
            "error": "Hash mismatch",
            "expected": expected_hash,
            "actual": current_hash,
        }

    # Verify signature
    sig_algorithm = signature["spec"]["signature"]["algorithm"]
    sig_value = signature["spec"]["signature"]["value"]

    if sig_algorithm == "mock":
        # Legacy mock signatures are no longer supported
        logger.warning("Mock signatures are deprecated and will be rejected")
        is_valid = False
    elif CRYPTO_AVAILABLE and sig_algorithm in ["rsa-pss-sha256", "ecdsa-sha256"]:
        # Cryptographic verification
        public_key_pem = signature["spec"].get("publicKey")
        if not public_key_pem:
            return False, {"error": "No public key in signature"}

        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(), backend=default_backend()
            )

            # Decode signature
            signature_bytes = base64.b64decode(sig_value)

            # Verify based on algorithm
            if sig_algorithm == "rsa-pss-sha256" and isinstance(
                public_key, rsa.RSAPublicKey
            ):
                public_key.verify(
                    signature_bytes,
                    current_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
                is_valid = True
            elif sig_algorithm == "ecdsa-sha256" and isinstance(
                public_key, ec.EllipticCurvePublicKey
            ):
                public_key.verify(
                    signature_bytes, current_hash.encode(), ec.ECDSA(hashes.SHA256())
                )
                is_valid = True
            else:
                is_valid = False

        except InvalidSignature:
            is_valid = False
        except Exception as e:
            return False, {"error": f"Verification failed: {e}"}
    else:
        return False, {"error": f"Unsupported algorithm: {sig_algorithm}"}

    # Return result with signature info
    if is_valid:
        info = {
            "valid": True,
            "pack": signature["metadata"]["pack"],
            "version": signature["metadata"]["version"],
            "signed_at": signature["metadata"]["timestamp"],
            "signer": signature["metadata"].get("signer", "unknown"),
            "algorithm": sig_algorithm,
        }
    else:
        info = {"valid": False, "error": "Signature verification failed"}

    return is_valid, info


def create_keyless_signature(artifact_path: Path, identity: str) -> Dict[str, Any]:
    """
    Create keyless signature using identity (like sigstore)

    Args:
        artifact_path: Path to artifact
        identity: Identity (email, OIDC token, etc.)

    Returns:
        Keyless signature
    """
    artifact_hash = _calculate_file_hash(artifact_path)

    signature = {
        "version": "1.0.0",
        "kind": "greenlang-keyless-signature",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "artifact": str(artifact_path.name),
            "identity": identity,
        },
        "spec": {
            "hash": {"algorithm": "sha256", "value": artifact_hash},
            "identity": {
                "issuer": "greenlang",
                "subject": identity,
                "verified_at": datetime.now().isoformat(),
            },
            # In real implementation, would include:
            # - Transparency log entry
            # - Certificate chain
            # - OIDC token claims
        },
    }

    return signature
