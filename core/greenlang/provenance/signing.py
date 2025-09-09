"""
Artifact Signing and Verification
==================================

Signs and verifies artifacts using cosign-style signatures.
"""

import json
import hashlib
import base64
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


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
    with open(sig_path, "w") as f:
        json.dump(signature, f, indent=2)
    
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
    with open(signature_path) as f:
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
    Sign an entire pack
    
    Args:
        pack_path: Path to pack directory
        key_path: Path to signing key
    
    Returns:
        Pack signature
    """
    # Calculate hash of all pack files
    pack_hash = _calculate_directory_hash(pack_path)
    
    # Load manifest
    manifest_path = pack_path / "pack.yaml"
    if manifest_path.exists():
        import yaml
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
    else:
        manifest = {}
    
    # Create pack signature
    signature = {
        "version": "1.0.0",
        "kind": "greenlang-pack-signature",
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "pack": manifest.get("name", "unknown"),
            "version": manifest.get("version", "0.0.0")
        },
        "spec": {
            "hash": {
                "algorithm": "sha256",
                "value": pack_hash
            },
            "signature": {
                "algorithm": "mock",
                "value": _mock_sign(pack_hash, key_path)
            },
            "manifest": manifest
        }
    }
    
    # Save signature
    sig_path = pack_path / "pack.sig"
    with open(sig_path, "w") as f:
        json.dump(signature, f, indent=2)
    
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
    with open(sig_path) as f:
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