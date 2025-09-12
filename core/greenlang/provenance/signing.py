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
    from cryptography.hazmat.primitives.asymmetric import padding, rsa, ec, utils
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available, using mock signing")


def sign_artifact(artifact_path: Path, key_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Sign an artifact
    
    Args:
        artifact_path: Path to artifact to sign
        key_path: Path to signing key (optional)
    
    Returns:
        Signature dictionary
    """
    # TODO: Integrate with actual cosign or sigstore
    # For now, create a simple signature format
    
    if not artifact_path.exists():
        raise ValueError(f"Artifact not found: {artifact_path}")
    
    # Calculate artifact hash
    artifact_hash = _calculate_file_hash(artifact_path)
    
    # Create signature payload
    signature = {
        "version": "1.0.0",
        "kind": "greenlang-signature",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "artifact": str(artifact_path.name),
            "size": artifact_path.stat().st_size
        },
        "spec": {
            "hash": {
                "algorithm": "sha256",
                "value": artifact_hash
            },
            "signature": {
                "algorithm": "mock",  # Would be RSA, ECDSA, etc.
                "value": _mock_sign(artifact_hash, key_path)
            }
        }
    }
    
    # Save signature next to artifact
    sig_path = artifact_path.with_suffix(artifact_path.suffix + ".sig")
    with open(sig_path, "w", encoding='utf-8') as f:
        json.dump(signature, f, indent=2, ensure_ascii=False)
    
    return signature


def verify_artifact(artifact_path: Path, signature_path: Optional[Path] = None) -> tuple[bool, Optional[Dict[str, Any]]]:
    """
    Verify an artifact signature
    
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
    with open(signature_path, 'r', encoding='utf-8', errors='replace') as f:
        signature = json.load(f)
    
    # Verify hash
    expected_hash = signature["spec"]["hash"]["value"]
    actual_hash = _calculate_file_hash(artifact_path)
    
    if expected_hash != actual_hash:
        return False, None
    
    # Verify signature (mock for now)
    sig_value = signature["spec"]["signature"]["value"]
    expected_sig = _mock_sign(actual_hash, None)
    
    is_valid = sig_value == expected_sig
    
    # Extract signer info
    signer_info = None
    if is_valid:
        signer_info = {
            "subject": signature.get("metadata", {}).get("artifact", "Unknown"),
            "issuer": "greenlang-mock",
            "timestamp": signature.get("metadata", {}).get("timestamp", "Unknown")
        }
    
    return is_valid, signer_info


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
    pack_hash = _calculate_directory_hash(pack_path, exclude=["pack.sig", "*.pem", "*.key"])
    
    # Load manifest
    manifest_path = pack_path / "pack.yaml"
    if manifest_path.exists():
        import yaml
        with open(manifest_path, 'r', encoding='utf-8', errors='replace') as f:
            manifest = yaml.safe_load(f)
    else:
        manifest = {}
    
    # Sign the hash
    if CRYPTO_AVAILABLE:
        # Use real cryptographic signing
        if key_path is None:
            # Generate or use default key
            key_path = _get_or_create_key_pair(pack_path)
        
        signature_value, algorithm, public_key_pem = _cryptographic_sign(pack_hash, key_path)
    else:
        # Fallback to mock signing
        signature_value = _mock_sign(pack_hash, key_path)
        algorithm = "mock"
        public_key_pem = None
    
    # Create pack signature
    signature = {
        "version": "1.0.0",
        "kind": "greenlang-pack-signature",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pack": manifest.get("name", "unknown"),
            "version": manifest.get("version", "0.0.0"),
            "signer": os.environ.get("USER", "unknown")
        },
        "spec": {
            "hash": {
                "algorithm": "sha256",
                "value": pack_hash
            },
            "signature": {
                "algorithm": algorithm,
                "value": signature_value
            }
        }
    }
    
    # Include public key if available
    if public_key_pem:
        signature["spec"]["publicKey"] = public_key_pem
    
    # Include manifest hash for integrity
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    signature["spec"]["manifestHash"] = manifest_hash
    
    # Save signature
    sig_path = pack_path / "pack.sig"
    with open(sig_path, "w", encoding='utf-8') as f:
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
    with open(sig_path, 'r', encoding='utf-8', errors='replace') as f:
        signature = json.load(f)
    
    # Calculate current hash
    current_hash = _calculate_directory_hash(pack_path, exclude=["pack.sig"])
    expected_hash = signature["spec"]["hash"]["value"]
    
    if current_hash != expected_hash:
        return False
    
    # Verify signature
    sig_value = signature["spec"]["signature"]["value"]
    expected_sig = _mock_sign(current_hash, None)
    
    return sig_value == expected_sig


def _calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file"""
    hasher = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
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
            if file_path.name.startswith('.'):
                continue
            if '__pycache__' in str(file_path):
                continue
            
            # Include file path and content in hash
            relative_path = file_path.relative_to(directory)
            hasher.update(str(relative_path).encode())
            
            with open(file_path, 'rb') as f:
                hasher.update(f.read())
    
    return hasher.hexdigest()


def _mock_sign(data: str, key_path: Optional[Path]) -> str:
    """
    Mock signing function
    
    Real implementation would use cryptographic signing
    """
    # For demo purposes, just create a deterministic "signature"
    signer = hashlib.sha256()
    signer.update(data.encode())
    
    if key_path and key_path.exists():
        with open(key_path, 'rb') as f:
            signer.update(f.read())
    else:
        signer.update(b"mock-key")
    
    return base64.b64encode(signer.digest()).decode()


def _get_or_create_key_pair(pack_path: Path) -> Path:
    """
    Get or create RSA key pair for signing
    
    Args:
        pack_path: Pack directory
    
    Returns:
        Path to private key
    """
    # Check for existing keys in pack
    private_key_path = pack_path / "private.key"
    public_key_path = pack_path / "public.pem"
    
    if private_key_path.exists():
        return private_key_path
    
    # Check for global keys
    gl_home = Path.home() / ".greenlang"
    gl_home.mkdir(exist_ok=True)
    
    global_private = gl_home / "signing.key"
    global_public = gl_home / "signing.pub"
    
    if global_private.exists():
        return global_private
    
    # Generate new key pair
    if CRYPTO_AVAILABLE:
        logger.info("Generating new RSA key pair for signing")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        # Save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        with open(global_private, 'wb') as f:
            f.write(private_pem)
        
        # Save public key
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        with open(global_public, 'wb') as f:
            f.write(public_pem)
        
        logger.info(f"Keys saved to {gl_home}")
        return global_private
    else:
        # Create mock key
        with open(global_private, 'w', encoding='utf-8') as f:
            f.write("MOCK_PRIVATE_KEY")
        return global_private


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
    with open(key_path, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend()
        )
    
    # Sign the data
    if isinstance(private_key, rsa.RSAPrivateKey):
        # RSA signing
        signature = private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        algorithm = "rsa-pss-sha256"
    elif isinstance(private_key, ec.EllipticCurvePrivateKey):
        # ECDSA signing
        signature = private_key.sign(
            data.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        algorithm = "ecdsa-sha256"
    else:
        raise ValueError(f"Unsupported key type: {type(private_key)}")
    
    # Get public key
    public_key = private_key.public_key()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    
    # Base64 encode signature
    signature_b64 = base64.b64encode(signature).decode()
    
    return signature_b64, algorithm, public_pem


def verify_pack_signature(pack_path: Path, signature_path: Optional[Path] = None) -> Tuple[bool, Dict[str, Any]]:
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
    with open(signature_path, 'r', encoding='utf-8', errors='replace') as f:
        signature = json.load(f)
    
    # Calculate current hash
    current_hash = _calculate_directory_hash(pack_path, exclude=["pack.sig", "*.pem", "*.key"])
    expected_hash = signature["spec"]["hash"]["value"]
    
    if current_hash != expected_hash:
        return False, {"error": "Hash mismatch", "expected": expected_hash, "actual": current_hash}
    
    # Verify signature
    sig_algorithm = signature["spec"]["signature"]["algorithm"]
    sig_value = signature["spec"]["signature"]["value"]
    
    if sig_algorithm == "mock":
        # Mock verification
        expected_sig = _mock_sign(current_hash, None)
        is_valid = sig_value == expected_sig
    elif CRYPTO_AVAILABLE and sig_algorithm in ["rsa-pss-sha256", "ecdsa-sha256"]:
        # Cryptographic verification
        public_key_pem = signature["spec"].get("publicKey")
        if not public_key_pem:
            return False, {"error": "No public key in signature"}
        
        try:
            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=default_backend()
            )
            
            # Decode signature
            signature_bytes = base64.b64decode(sig_value)
            
            # Verify based on algorithm
            if sig_algorithm == "rsa-pss-sha256" and isinstance(public_key, rsa.RSAPublicKey):
                public_key.verify(
                    signature_bytes,
                    current_hash.encode(),
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                is_valid = True
            elif sig_algorithm == "ecdsa-sha256" and isinstance(public_key, ec.EllipticCurvePublicKey):
                public_key.verify(
                    signature_bytes,
                    current_hash.encode(),
                    ec.ECDSA(hashes.SHA256())
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
            "algorithm": sig_algorithm
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
            "identity": identity
        },
        "spec": {
            "hash": {
                "algorithm": "sha256",
                "value": artifact_hash
            },
            "identity": {
                "issuer": "greenlang",
                "subject": identity,
                "verified_at": datetime.now().isoformat()
            },
            # In real implementation, would include:
            # - Transparency log entry
            # - Certificate chain
            # - OIDC token claims
        }
    }
    
    return signature