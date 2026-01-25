# -*- coding: utf-8 -*-
"""
Secure Signing Provider Abstraction for GreenLang
=================================================

Provides a secure signing abstraction with multiple provider implementations:
- SigstoreKeylessSigner: For CI/CD using OIDC (GitHub Actions)
- EphemeralKeypairSigner: For tests using runtime-generated keys
- ExternalKMSSigner: Placeholder for KMS integration

NO HARDCODED KEYS - All keys are either ephemeral or externally managed.
"""

import os
import hashlib
import base64
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass
from greenlang.utilities.determinism import DeterministicClock
from greenlang.intelligence import ChatMessage

# Import KMS providers
try:
    from .kms import (
        create_kms_provider,
        detect_kms_provider,
        KMSConfig,
        KMSProviderError,
        KMSSigningError,
    )
    KMS_AVAILABLE = True
except ImportError:
    KMS_AVAILABLE = False
    logger.info("KMS providers not available")

logger = logging.getLogger(__name__)

# Try to import cryptography
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import (
        rsa,
        ec,
        ed25519,
        padding,
    )
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available, signing will be limited")

# Try to import sigstore
try:
    from sigstore.sign import Signer as SigstoreSigner
    from sigstore.verify import Verifier as SigstoreVerifier

    SIGSTORE_AVAILABLE = True
except ImportError:
    SIGSTORE_AVAILABLE = False
    logger.info("sigstore library not available, keyless signing disabled")


class SignResult(TypedDict):
    """Result from signing operation"""

    signature: bytes
    cert_chain: Optional[bytes]  # for Sigstore bundle
    transparency_entry: Optional[str]  # Rekor entry ID/URL
    algorithm: str
    timestamp: str
    public_key: Optional[str]  # PEM encoded public key for verification


@dataclass
class SigningConfig:
    """Configuration for signing operations"""

    mode: str  # 'keyless', 'ephemeral', 'kms', 'disabled'
    kms_key_id: Optional[str] = None
    sigstore_audience: Optional[str] = None
    sigstore_staging: bool = False  # Use staging endpoints for testing
    rekor_url: Optional[str] = None
    fulcio_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "SigningConfig":
        """Load configuration from environment"""
        mode = os.environ.get(
            "GL_SIGNING_MODE", "keyless" if os.environ.get("CI") else "ephemeral"
        )

        # Detect if we're in CI and have OIDC available
        if mode == "keyless" and not os.environ.get("CI"):
            logger.info(
                "Keyless signing requested but not in CI, falling back to ephemeral"
            )
            mode = "ephemeral"

        return cls(
            mode=mode,
            kms_key_id=os.environ.get("GL_KMS_KEY_ID"),
            sigstore_audience=os.environ.get("GL_SIGSTORE_AUDIENCE", "sigstore"),
            sigstore_staging=os.environ.get("GL_SIGSTORE_STAGING", "").lower()
            in ("1", "true"),
            rekor_url=os.environ.get("GL_REKOR_URL"),
            fulcio_url=os.environ.get("GL_FULCIO_URL"),
        )


class Signer(ABC):
    """Abstract base class for signing providers"""

    @abstractmethod
    def sign(self, payload: bytes) -> SignResult:
        """
        Sign a payload

        Args:
            payload: Data to sign

        Returns:
            SignResult with signature and metadata
        """

    @abstractmethod
    def get_signer_info(self) -> Dict[str, Any]:
        """Get information about this signer"""


class Verifier(ABC):
    """Abstract base class for verification providers"""

    @abstractmethod
    def verify(self, payload: bytes, signature: bytes, **kwargs) -> None:
        """
        Verify a signature

        Args:
            payload: Original data that was signed
            signature: Signature to verify
            **kwargs: Additional verification parameters (cert_chain, public_key, etc.)

        Raises:
            InvalidSignature: If verification fails
        """

    @abstractmethod
    def get_verifier_info(self) -> Dict[str, Any]:
        """Get information about this verifier"""


class SigstoreKeylessSigner(Signer):
    """
    Sigstore keyless signing using OIDC identity

    This signer uses Sigstore's keyless signing flow:
    1. Obtain OIDC token from CI provider (GitHub Actions)
    2. Request certificate from Fulcio
    3. Sign with ephemeral key
    4. Log to Rekor transparency log
    """

    def __init__(self, config: Optional[SigningConfig] = None):
        """Initialize Sigstore keyless signer"""
        if not SIGSTORE_AVAILABLE:
            raise RuntimeError("sigstore library not installed: pip install sigstore")

        self.config = config or SigningConfig.from_env()

        # Detect if we're in a supported CI environment
        if not os.environ.get("CI"):
            raise RuntimeError(
                "Sigstore keyless signing requires CI environment with OIDC"
            )

        # Check for GitHub Actions OIDC
        if os.environ.get("GITHUB_ACTIONS") != "true":
            logger.warning("Not in GitHub Actions, Sigstore signing may not work")

        self.staging = self.config.sigstore_staging
        logger.info(f"Initialized Sigstore keyless signer (staging={self.staging})")

    def sign(self, payload: bytes) -> SignResult:
        """Sign using Sigstore keyless flow"""
        try:
            # Create Sigstore signer
            signer = (
                SigstoreSigner.production()
                if not self.staging
                else SigstoreSigner.staging()
            )

            # Sign the payload
            result = signer.sign(payload)

            # Extract bundle information
            bundle = result.to_bundle()

            return SignResult(
                signature=base64.b64decode(bundle.message_signature.signature),
                cert_chain=(
                    bundle.verification_material.x509_certificate_chain.certificates[
                        0
                    ].raw_bytes
                    if bundle.verification_material.x509_certificate_chain
                    else None
                ),
                transparency_entry=(
                    bundle.verification_material.tlog_entries[0].log_id.key_id
                    if bundle.verification_material.tlog_entries
                    else None
                ),
                algorithm="sigstore-keyless",
                timestamp=DeterministicClock.now().isoformat(),
                public_key=None,  # Keyless doesn't use persistent keys
            )

        except Exception as e:
            logger.error(f"Sigstore signing failed: {e}")
            raise RuntimeError(f"Failed to sign with Sigstore: {e}")

    def get_signer_info(self) -> Dict[str, Any]:
        """Get signer information"""
        return {
            "type": "sigstore-keyless",
            "staging": self.staging,
            "ci_provider": os.environ.get("CI_NAME", "unknown"),
            "repository": os.environ.get("GITHUB_REPOSITORY", "unknown"),
            "workflow": os.environ.get("GITHUB_WORKFLOW", "unknown"),
        }


class EphemeralKeypairSigner(Signer):
    """
    Ephemeral key pair signer for testing

    Generates a new Ed25519 key pair for each instance.
    Keys exist only in memory and are never persisted.
    """

    def __init__(self):
        """Initialize with ephemeral Ed25519 keypair"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not installed")

        # Generate ephemeral Ed25519 keypair
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()

        # Store public key in PEM format
        self.public_key_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        logger.debug("Generated ephemeral Ed25519 keypair for testing")

    def sign(self, payload: bytes) -> SignResult:
        """Sign with ephemeral key"""
        # Sign the payload directly with Ed25519
        signature = self.private_key.sign(payload)

        return SignResult(
            signature=signature,
            cert_chain=None,
            transparency_entry=None,
            algorithm="ed25519",
            timestamp=DeterministicClock.now().isoformat(),
            public_key=self.public_key_pem,
        )

    def get_signer_info(self) -> Dict[str, Any]:
        """Get signer information"""
        return {
            "type": "ephemeral-ed25519",
            "key_fingerprint": hashlib.sha256(self.public_key_pem.encode()).hexdigest()[
                :16
            ],
            "test_mode": True,
        }


class ExternalKMSSigner(Signer):
    """
    External KMS signer with support for multiple cloud providers

    Supports:
    - AWS KMS
    - Azure Key Vault
    - Google Cloud KMS

    Features:
    - Automatic provider detection
    - Key caching with TTL
    - Retry logic with exponential backoff
    - Async signing support
    """

    def __init__(self, config: Optional[SigningConfig] = None):
        """Initialize KMS signer"""
        if not KMS_AVAILABLE:
            raise RuntimeError(
                "KMS support not available. Install required provider: "
                "pip install boto3  # for AWS\n"
                "pip install azure-keyvault-keys azure-identity  # for Azure\n"
                "pip install google-cloud-kms  # for GCP"
            )

        self.config = config or SigningConfig.from_env()

        if not self.config.kms_key_id:
            raise ValueError("KMS key ID required for KMS signing")

        # Create KMS configuration
        kms_config = KMSConfig(
            provider=os.environ.get("GL_KMS_PROVIDER", ""),
            key_id=self.config.kms_key_id,
            region=os.environ.get("GL_KMS_REGION", os.environ.get("AWS_REGION")),
            cache_ttl_seconds=int(os.environ.get("GL_KMS_CACHE_TTL", "300")),
            max_retries=int(os.environ.get("GL_KMS_MAX_RETRIES", "3")),
            timeout_seconds=int(os.environ.get("GL_KMS_TIMEOUT", "30")),
        )

        # Add provider-specific configuration
        kms_config.aws_profile = os.environ.get("AWS_PROFILE")
        kms_config.azure_tenant_id = os.environ.get("AZURE_TENANT_ID")
        kms_config.azure_vault_url = os.environ.get("AZURE_KEY_VAULT_URL")
        kms_config.gcp_project_id = os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT"))
        kms_config.gcp_location_id = os.environ.get("GCP_LOCATION", "global")
        kms_config.gcp_keyring_id = os.environ.get("GCP_KEYRING")

        # Create KMS provider
        try:
            self.kms_provider = create_kms_provider(kms_config)
            self.provider_name = self.kms_provider.config.provider
            logger.info(f"Initialized {self.provider_name.upper()} KMS signer with key {self.config.kms_key_id}")
        except KMSProviderError as e:
            raise RuntimeError(f"Failed to initialize KMS provider: {e}")

    def sign(self, payload: bytes) -> SignResult:
        """Sign using external KMS"""
        try:
            # Sign using KMS provider
            kms_result = self.kms_provider.sign(payload)

            # Convert to our SignResult format
            return SignResult(
                signature=kms_result["signature"],
                cert_chain=None,  # KMS doesn't use certificates
                transparency_entry=None,  # No transparency log for KMS
                algorithm=kms_result["algorithm"],
                timestamp=kms_result["timestamp"],
                public_key=None,  # Will be fetched separately if needed
            )

        except KMSSigningError as e:
            logger.error(f"KMS signing failed: {e}")
            raise RuntimeError(f"Failed to sign with KMS: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during KMS signing: {e}")
            raise RuntimeError(f"Unexpected KMS signing error: {e}")

    def get_signer_info(self) -> Dict[str, Any]:
        """Get signer information"""
        info = {
            "type": "kms",
            "provider": self.provider_name,
            "key_id": self.config.kms_key_id,
        }

        # Add provider-specific info
        try:
            key_info = self.kms_provider.get_cached_key_info()
            info.update({
                "algorithm": key_info.algorithm.value,
                "enabled": key_info.enabled,
                "rotation_enabled": key_info.rotation_enabled,
                "created_at": key_info.created_at.isoformat() if key_info.created_at else None,
            })
        except Exception as e:
            logger.warning(f"Could not fetch key info: {e}")

        return info


class SigstoreBundleVerifier(Verifier):
    """Verifier for Sigstore bundles"""

    def __init__(self, staging: bool = False):
        """Initialize Sigstore verifier"""
        if not SIGSTORE_AVAILABLE:
            raise RuntimeError("sigstore library not installed")

        self.staging = staging

    def verify(self, payload: bytes, signature: bytes, **kwargs) -> None:
        """Verify Sigstore bundle"""
        cert_chain = kwargs.get("cert_chain")
        transparency_entry = kwargs.get("transparency_entry")

        if not cert_chain:
            raise ValueError("Certificate chain required for Sigstore verification")

        try:
            # Create verifier
            verifier = (
                SigstoreVerifier.production()
                if not self.staging
                else SigstoreVerifier.staging()
            )

            # TODO: Reconstruct bundle from components and verify
            # This requires more complex bundle reconstruction
            logger.info("Sigstore verification would happen here")

        except Exception as e:
            raise InvalidSignature(f"Sigstore verification failed: {e}")

    def get_verifier_info(self) -> Dict[str, Any]:
        """Get verifier information"""
        return {"type": "sigstore-bundle", "staging": self.staging}


class DetachedSigVerifier(Verifier):
    """Verifier for detached signatures with public key"""

    def verify(self, payload: bytes, signature: bytes, **kwargs) -> None:
        """Verify detached signature"""
        public_key_pem = kwargs.get("public_key")
        algorithm = kwargs.get("algorithm", "ed25519")

        if not public_key_pem:
            raise ValueError("Public key required for detached signature verification")

        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library not installed")

        try:
            # Load public key
            if isinstance(public_key_pem, str):
                public_key_pem = public_key_pem.encode()

            public_key = serialization.load_pem_public_key(
                public_key_pem, backend=default_backend()
            )

            # Verify based on algorithm
            if algorithm == "ed25519" and isinstance(
                public_key, ed25519.Ed25519PublicKey
            ):
                public_key.verify(signature, payload)
            elif algorithm.startswith("rsa") and isinstance(
                public_key, rsa.RSAPublicKey
            ):
                public_key.verify(
                    signature,
                    payload,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH,
                    ),
                    hashes.SHA256(),
                )
            elif algorithm.startswith("ecdsa") and isinstance(
                public_key, ec.EllipticCurvePublicKey
            ):
                public_key.verify(signature, payload, ec.ECDSA(hashes.SHA256()))
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

        except InvalidSignature:
            raise InvalidSignature("Signature verification failed")
        except Exception as e:
            raise InvalidSignature(f"Verification failed: {e}")

    def get_verifier_info(self) -> Dict[str, Any]:
        """Get verifier information"""
        return {
            "type": "detached-signature",
            "supports": ["ed25519", "rsa-pss", "ecdsa"],
        }


def create_signer(config: Optional[SigningConfig] = None) -> Signer:
    """
    Create a signer based on configuration

    Args:
        config: Optional signing configuration

    Returns:
        Configured signer instance
    """
    config = config or SigningConfig.from_env()

    if config.mode == "disabled":
        raise RuntimeError("Signing is disabled")
    elif config.mode == "keyless":
        return SigstoreKeylessSigner(config)
    elif config.mode == "ephemeral":
        return EphemeralKeypairSigner()
    elif config.mode == "kms":
        return ExternalKMSSigner(config)
    else:
        raise ValueError(f"Unknown signing mode: {config.mode}")


def create_verifier(signature_type: str = "auto", **kwargs) -> Verifier:
    """
    Create a verifier based on signature type

    Args:
        signature_type: Type of signature ('sigstore', 'detached', 'auto')
        **kwargs: Additional verifier parameters

    Returns:
        Configured verifier instance
    """
    if signature_type == "sigstore" or (
        signature_type == "auto" and kwargs.get("cert_chain")
    ):
        return SigstoreBundleVerifier(staging=kwargs.get("staging", False))
    else:
        return DetachedSigVerifier()


def sign_artifact(
    artifact_path: Path, signer: Optional[Signer] = None
) -> Dict[str, Any]:
    """
    Sign an artifact file

    Args:
        artifact_path: Path to artifact
        signer: Optional signer instance (creates default if not provided)

    Returns:
        Signature dictionary with metadata
    """
    if not artifact_path.exists():
        raise ValueError(f"Artifact not found: {artifact_path}")

    # Read artifact
    artifact_bytes = artifact_path.read_bytes()

    # Create signer if not provided
    if signer is None:
        signer = create_signer()

    # Sign the artifact
    result = signer.sign(artifact_bytes)

    # Create signature structure
    signature = {
        "version": "2.0.0",
        "kind": "greenlang-signature",
        "metadata": {
            "timestamp": result["timestamp"],
            "artifact": str(artifact_path.name),
            "size": len(artifact_bytes),
            "hash": {
                "algorithm": "sha256",
                "value": hashlib.sha256(artifact_bytes).hexdigest(),
            },
        },
        "spec": {
            "signature": {
                "algorithm": result["algorithm"],
                "value": base64.b64encode(result["signature"]).decode(),
            }
        },
    }

    # Add optional fields
    if result.get("public_key"):
        signature["spec"]["publicKey"] = result["public_key"]
    if result.get("cert_chain"):
        signature["spec"]["certChain"] = base64.b64encode(result["cert_chain"]).decode()
    if result.get("transparency_entry"):
        signature["spec"]["transparencyLog"] = result["transparency_entry"]

    # Add signer info
    signature["metadata"]["signer"] = signer.get_signer_info()

    return signature


def verify_artifact(
    artifact_path: Path, signature: Dict[str, Any], verifier: Optional[Verifier] = None
) -> bool:
    """
    Verify an artifact signature

    Args:
        artifact_path: Path to artifact
        signature: Signature dictionary
        verifier: Optional verifier instance

    Returns:
        True if valid, raises InvalidSignature otherwise
    """
    if not artifact_path.exists():
        raise ValueError(f"Artifact not found: {artifact_path}")

    # Read artifact
    artifact_bytes = artifact_path.read_bytes()

    # Verify hash first
    expected_hash = signature["metadata"]["hash"]["value"]
    actual_hash = hashlib.sha256(artifact_bytes).hexdigest()

    if expected_hash != actual_hash:
        raise InvalidSignature(
            f"Hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    # Extract signature components
    sig_spec = signature["spec"]
    sig_bytes = base64.b64decode(sig_spec["signature"]["value"])
    algorithm = sig_spec["signature"]["algorithm"]

    # Create verifier if not provided
    if verifier is None:
        verifier = create_verifier(
            signature_type="sigstore" if sig_spec.get("certChain") else "detached"
        )

    # Prepare verification kwargs
    verify_kwargs = {"algorithm": algorithm}

    if sig_spec.get("publicKey"):
        verify_kwargs["public_key"] = sig_spec["publicKey"]
    if sig_spec.get("certChain"):
        verify_kwargs["cert_chain"] = base64.b64decode(sig_spec["certChain"])
    if sig_spec.get("transparencyLog"):
        verify_kwargs["transparency_entry"] = sig_spec["transparencyLog"]

    # Verify signature
    verifier.verify(artifact_bytes, sig_bytes, **verify_kwargs)

    return True
