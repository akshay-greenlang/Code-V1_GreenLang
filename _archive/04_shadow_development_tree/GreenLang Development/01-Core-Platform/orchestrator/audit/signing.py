# -*- coding: utf-8 -*-
"""
Cryptographic Signing for Tamper-Evident Audit Trail (FR-694)

This module implements cryptographic signing capabilities for the GreenLang
Orchestrator audit trail, providing tamper-evident signatures for audit packages.

Features:
- Abstract SigningProvider for pluggable key management
- LocalKeySigner using Ed25519/RSA with cryptography library
- KMSSigner placeholder for AWS KMS integration
- VaultSigner placeholder for HashiCorp Vault integration
- SignatureVerifier for signature verification
- Key rotation support with key_id tracking

Author: GreenLang Team
Version: 1.0.0
GL-FOUND-X-001: Tamper-Evident Audit Signing (FR-694)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND ENUMS
# ============================================================================

class SignatureAlgorithm(str, Enum):
    """Supported cryptographic signature algorithms."""
    ED25519 = "Ed25519"
    RSA_PSS_SHA256 = "RSA-PSS-SHA256"
    RSA_PSS_SHA384 = "RSA-PSS-SHA384"
    RSA_PSS_SHA512 = "RSA-PSS-SHA512"
    ECDSA_P256_SHA256 = "ECDSA-P256-SHA256"
    ECDSA_P384_SHA384 = "ECDSA-P384-SHA384"


class KeyType(str, Enum):
    """Type of cryptographic key."""
    ED25519 = "ed25519"
    RSA_2048 = "rsa-2048"
    RSA_3072 = "rsa-3072"
    RSA_4096 = "rsa-4096"
    ECDSA_P256 = "ecdsa-p256"
    ECDSA_P384 = "ecdsa-p384"


class ProviderType(str, Enum):
    """Signing provider types."""
    LOCAL = "local"
    AWS_KMS = "aws_kms"
    HASHICORP_VAULT = "hashicorp_vault"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_KMS = "gcp_kms"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class SignatureMetadata:
    """Metadata attached to a cryptographic signature."""
    key_id: str
    algorithm: SignatureAlgorithm
    signed_at: datetime
    provider: ProviderType
    key_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm.value,
            "signed_at": self.signed_at.isoformat(),
            "provider": self.provider.value,
            "key_version": self.key_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignatureMetadata":
        """Create metadata from dictionary."""
        return cls(
            key_id=data["key_id"],
            algorithm=SignatureAlgorithm(data["algorithm"]),
            signed_at=datetime.fromisoformat(data["signed_at"]),
            provider=ProviderType(data["provider"]),
            key_version=data.get("key_version"),
        )


@dataclass
class SignatureBundle:
    """Complete signature bundle with metadata for audit packages."""
    signature: str  # Base64-encoded
    metadata: SignatureMetadata
    package_hash: str  # SHA-256 hex digest

    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to dictionary for JSON serialization."""
        return {
            "signature": self.signature,
            "metadata": self.metadata.to_dict(),
            "package_hash": self.package_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignatureBundle":
        """Create bundle from dictionary."""
        return cls(
            signature=data["signature"],
            metadata=SignatureMetadata.from_dict(data["metadata"]),
            package_hash=data["package_hash"],
        )

    def to_json(self) -> str:
        """Serialize bundle to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, json_str: str) -> "SignatureBundle":
        """Deserialize bundle from JSON string."""
        return cls.from_dict(json.loads(json_str))


class PublicKeyInfo(BaseModel):
    """Public key information for signature verification."""
    key_id: str = Field(..., description="Unique key identifier")
    algorithm: SignatureAlgorithm = Field(..., description="Key algorithm")
    public_key_pem: str = Field(..., description="PEM-encoded public key")
    created_at: datetime = Field(..., description="Key creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Key expiration")
    is_active: bool = Field(True, description="Whether key is active")

    model_config = {"frozen": True}


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SigningError(Exception):
    """Base exception for signing operations."""
    pass


class KeyNotFoundError(SigningError):
    """Raised when a signing key is not found."""
    def __init__(self, key_id: str):
        super().__init__(f"Signing key not found: {key_id}")
        self.key_id = key_id


class SignatureVerificationError(SigningError):
    """Raised when signature verification fails."""
    def __init__(self, message: str, key_id: Optional[str] = None):
        super().__init__(message)
        self.key_id = key_id


class ProviderConfigError(SigningError):
    """Raised when provider configuration is invalid."""
    pass


class KeyRotationError(SigningError):
    """Raised when key rotation fails."""
    pass


# ============================================================================
# ABSTRACT SIGNING PROVIDER
# ============================================================================

class SigningProvider(ABC):
    """Abstract base class for cryptographic signing providers."""

    @property
    @abstractmethod
    def key_id(self) -> str:
        """Get the current active key identifier."""
        ...

    @property
    @abstractmethod
    def algorithm(self) -> SignatureAlgorithm:
        """Get the signature algorithm used by this provider."""
        ...

    @property
    @abstractmethod
    def provider_type(self) -> ProviderType:
        """Get the provider type."""
        ...

    @abstractmethod
    def sign(self, data: bytes) -> bytes:
        """Sign data with the private key."""
        ...

    @abstractmethod
    def get_public_key(self) -> bytes:
        """Get the public key in PEM format."""
        ...

    def sign_with_metadata(self, data: bytes) -> SignatureBundle:
        """Sign data and return a complete signature bundle with metadata."""
        package_hash = hashlib.sha256(data).hexdigest()
        signature_bytes = self.sign(data)
        metadata = SignatureMetadata(
            key_id=self.key_id,
            algorithm=self.algorithm,
            signed_at=datetime.now(timezone.utc),
            provider=self.provider_type,
            key_version=getattr(self, 'key_version', None),
        )
        return SignatureBundle(
            signature=base64.b64encode(signature_bytes).decode('ascii'),
            metadata=metadata,
            package_hash=package_hash,
        )

    def get_public_key_info(self) -> PublicKeyInfo:
        """Get public key information for verification distribution."""
        return PublicKeyInfo(
            key_id=self.key_id,
            algorithm=self.algorithm,
            public_key_pem=self.get_public_key().decode('utf-8'),
            created_at=getattr(self, 'created_at', datetime.now(timezone.utc)),
            expires_at=getattr(self, 'expires_at', None),
            is_active=getattr(self, 'is_active', True),
        )


# ============================================================================
# LOCAL KEY SIGNER
# ============================================================================

class LocalKeySigner(SigningProvider):
    """Local file-based signing provider using the cryptography library."""

    def __init__(
        self,
        private_key: Any,
        public_key: Any,
        key_id: str,
        algorithm: SignatureAlgorithm,
        created_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
    ):
        self._private_key = private_key
        self._public_key = public_key
        self._key_id = key_id
        self._algorithm = algorithm
        self.created_at = created_at or datetime.now(timezone.utc)
        self.expires_at = expires_at
        self.is_active = True
        logger.info(f"Initialized LocalKeySigner: key_id={key_id}")

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def algorithm(self) -> SignatureAlgorithm:
        return self._algorithm

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.LOCAL

    @classmethod
    def generate_ed25519(cls, key_id: str, expires_at: Optional[datetime] = None) -> "LocalKeySigner":
        """Generate a new Ed25519 key pair."""
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        logger.info(f"Generated new Ed25519 key pair: key_id={key_id}")
        return cls(private_key, public_key, key_id, SignatureAlgorithm.ED25519, expires_at=expires_at)

    @classmethod
    def generate_rsa(
        cls,
        key_id: str,
        key_size: int = 3072,
        hash_algorithm: str = "SHA256",
        expires_at: Optional[datetime] = None
    ) -> "LocalKeySigner":
        """Generate a new RSA key pair for RSA-PSS signing."""
        try:
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        if key_size not in (2048, 3072, 4096):
            raise ValueError(f"Invalid RSA key size: {key_size}")
        algo_map = {
            "SHA256": SignatureAlgorithm.RSA_PSS_SHA256,
            "SHA384": SignatureAlgorithm.RSA_PSS_SHA384,
            "SHA512": SignatureAlgorithm.RSA_PSS_SHA512
        }
        if hash_algorithm not in algo_map:
            raise ValueError(f"Invalid hash algorithm: {hash_algorithm}")
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
        public_key = private_key.public_key()
        logger.info(f"Generated new RSA-{key_size} key pair: key_id={key_id}")
        return cls(private_key, public_key, key_id, algo_map[hash_algorithm], expires_at=expires_at)

    @classmethod
    def generate_ecdsa(cls, key_id: str, curve: str = "P256", expires_at: Optional[datetime] = None) -> "LocalKeySigner":
        """Generate a new ECDSA key pair."""
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        curve_map = {
            "P256": (ec.SECP256R1(), SignatureAlgorithm.ECDSA_P256_SHA256),
            "P384": (ec.SECP384R1(), SignatureAlgorithm.ECDSA_P384_SHA384)
        }
        if curve not in curve_map:
            raise ValueError(f"Invalid curve: {curve}")
        curve_obj, algorithm = curve_map[curve]
        private_key = ec.generate_private_key(curve_obj)
        public_key = private_key.public_key()
        logger.info(f"Generated new ECDSA-{curve} key pair: key_id={key_id}")
        return cls(private_key, public_key, key_id, algorithm, expires_at=expires_at)

    @classmethod
    def load_from_file(
        cls,
        private_key_path: Union[str, Path],
        key_id: str,
        algorithm: SignatureAlgorithm,
        password: Optional[bytes] = None
    ) -> "LocalKeySigner":
        """Load a signing key from a PEM file."""
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        path = Path(private_key_path)
        if not path.exists():
            raise FileNotFoundError(f"Private key file not found: {path}")
        try:
            with open(path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=password)
            public_key = private_key.public_key()
            logger.info(f"Loaded key from file: key_id={key_id}, path={path}")
            return cls(private_key, public_key, key_id, algorithm)
        except Exception as e:
            logger.error(f"Failed to load key from {path}: {e}")
            raise SigningError(f"Failed to load key: {e}") from e

    def save_private_key(self, path: Union[str, Path], password: Optional[bytes] = None) -> None:
        """Save the private key to a PEM file."""
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        try:
            encryption = (
                serialization.BestAvailableEncryption(password)
                if password else serialization.NoEncryption()
            )
            pem_data = self._private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=encryption
            )
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                f.write(pem_data)
            logger.info(f"Saved private key to: {path}")
        except Exception as e:
            logger.error(f"Failed to save private key: {e}")
            raise SigningError(f"Failed to save private key: {e}") from e

    def sign(self, data: bytes) -> bytes:
        """Sign data with the private key."""
        try:
            if self._algorithm == SignatureAlgorithm.ED25519:
                return self._private_key.sign(data)
            elif self._algorithm in (
                SignatureAlgorithm.RSA_PSS_SHA256,
                SignatureAlgorithm.RSA_PSS_SHA384,
                SignatureAlgorithm.RSA_PSS_SHA512
            ):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                hash_map = {
                    SignatureAlgorithm.RSA_PSS_SHA256: hashes.SHA256(),
                    SignatureAlgorithm.RSA_PSS_SHA384: hashes.SHA384(),
                    SignatureAlgorithm.RSA_PSS_SHA512: hashes.SHA512()
                }
                return self._private_key.sign(
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hash_map[self._algorithm]),
                        salt_length=padding.PSS.AUTO
                    ),
                    hash_map[self._algorithm]
                )
            elif self._algorithm in (
                SignatureAlgorithm.ECDSA_P256_SHA256,
                SignatureAlgorithm.ECDSA_P384_SHA384
            ):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import ec
                hash_map = {
                    SignatureAlgorithm.ECDSA_P256_SHA256: hashes.SHA256(),
                    SignatureAlgorithm.ECDSA_P384_SHA384: hashes.SHA384()
                }
                return self._private_key.sign(data, ec.ECDSA(hash_map[self._algorithm]))
            else:
                raise SigningError(f"Unsupported algorithm: {self._algorithm}")
        except Exception as e:
            logger.error(f"Signing failed: {e}")
            raise SigningError(f"Signing failed: {e}") from e

    def get_public_key(self) -> bytes:
        """Get the public key in PEM format."""
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )


# ============================================================================
# AWS KMS SIGNER
# ============================================================================

class KMSSigner(SigningProvider):
    """AWS KMS-based signing provider."""

    def __init__(
        self,
        key_id: str,
        region: str = "us-east-1",
        algorithm: SignatureAlgorithm = SignatureAlgorithm.ECDSA_P256_SHA256,
        profile_name: Optional[str] = None
    ):
        self._key_id = key_id
        self._region = region
        self._algorithm = algorithm
        self._profile_name = profile_name
        self._client = None
        logger.info(f"Initialized KMSSigner: key_id={key_id}, region={region}")

    @property
    def key_id(self) -> str:
        return self._key_id

    @property
    def algorithm(self) -> SignatureAlgorithm:
        return self._algorithm

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.AWS_KMS

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError("boto3 required for KMSSigner. Install: pip install boto3")
            session_kwargs = {}
            if self._profile_name:
                session_kwargs["profile_name"] = self._profile_name
            session = boto3.Session(**session_kwargs)
            self._client = session.client("kms", region_name=self._region)
        return self._client

    def sign(self, data: bytes) -> bytes:
        """Sign data using AWS KMS."""
        try:
            client = self._get_client()
            kms_algo_map = {
                SignatureAlgorithm.ECDSA_P256_SHA256: "ECDSA_SHA_256",
                SignatureAlgorithm.ECDSA_P384_SHA384: "ECDSA_SHA_384",
                SignatureAlgorithm.RSA_PSS_SHA256: "RSASSA_PSS_SHA_256",
                SignatureAlgorithm.RSA_PSS_SHA384: "RSASSA_PSS_SHA_384",
                SignatureAlgorithm.RSA_PSS_SHA512: "RSASSA_PSS_SHA_512"
            }
            if self._algorithm not in kms_algo_map:
                raise SigningError(f"Algorithm {self._algorithm} not supported by KMS")
            response = client.sign(
                KeyId=self._key_id,
                Message=data,
                MessageType="RAW",
                SigningAlgorithm=kms_algo_map[self._algorithm]
            )
            return response["Signature"]
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"KMS signing failed: {e}")
            raise SigningError(f"KMS signing failed: {e}") from e

    def get_public_key(self) -> bytes:
        """Get the public key from KMS."""
        try:
            client = self._get_client()
            response = client.get_public_key(KeyId=self._key_id)
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.serialization import load_der_public_key
            public_key = load_der_public_key(response["PublicKey"])
            return public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        except ImportError as e:
            raise ImportError(str(e))
        except Exception as e:
            logger.error(f"Failed to get KMS public key: {e}")
            raise SigningError(f"Failed to get KMS public key: {e}") from e


# ============================================================================
# HASHICORP VAULT SIGNER
# ============================================================================

class VaultSigner(SigningProvider):
    """HashiCorp Vault Transit secrets engine signing provider."""

    def __init__(
        self,
        vault_url: str,
        key_name: str,
        token: Optional[str] = None,
        mount_point: str = "transit",
        algorithm: SignatureAlgorithm = SignatureAlgorithm.ED25519,
        key_version: Optional[int] = None
    ):
        self._vault_url = vault_url
        self._key_name = key_name
        self._token = token
        self._mount_point = mount_point
        self._algorithm = algorithm
        self._key_version = key_version
        self._client = None
        logger.info(f"Initialized VaultSigner: key_name={key_name}")

    @property
    def key_id(self) -> str:
        return self._key_name

    @property
    def algorithm(self) -> SignatureAlgorithm:
        return self._algorithm

    @property
    def provider_type(self) -> ProviderType:
        return ProviderType.HASHICORP_VAULT

    @property
    def key_version(self) -> Optional[str]:
        return str(self._key_version) if self._key_version else None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import hvac
            except ImportError:
                raise ImportError("hvac required for VaultSigner. Install: pip install hvac")
            self._client = hvac.Client(url=self._vault_url, token=self._token)
        return self._client

    def sign(self, data: bytes) -> bytes:
        """Sign data using Vault Transit engine."""
        try:
            client = self._get_client()
            hash_algo_map = {
                SignatureAlgorithm.ED25519: "sha2-256",
                SignatureAlgorithm.RSA_PSS_SHA256: "sha2-256",
                SignatureAlgorithm.RSA_PSS_SHA384: "sha2-384",
                SignatureAlgorithm.RSA_PSS_SHA512: "sha2-512",
                SignatureAlgorithm.ECDSA_P256_SHA256: "sha2-256",
                SignatureAlgorithm.ECDSA_P384_SHA384: "sha2-384"
            }
            input_data = base64.b64encode(data).decode("utf-8")
            kwargs = {
                "name": self._key_name,
                "plaintext": input_data,
                "hash_algorithm": hash_algo_map.get(self._algorithm, "sha2-256"),
                "mount_point": self._mount_point
            }
            if self._key_version:
                kwargs["key_version"] = self._key_version
            response = client.secrets.transit.sign_data(**kwargs)
            signature_data = response["data"]["signature"]
            parts = signature_data.split(":")
            signature_b64 = parts[2] if len(parts) >= 3 else signature_data
            return base64.b64decode(signature_b64)
        except ImportError:
            raise
        except Exception as e:
            logger.error(f"Vault signing failed: {e}")
            raise SigningError(f"Vault signing failed: {e}") from e

    def get_public_key(self) -> bytes:
        """Get the public key from Vault."""
        try:
            client = self._get_client()
            response = client.secrets.transit.read_key(
                name=self._key_name,
                mount_point=self._mount_point
            )
            keys = response["data"]["keys"]
            if self._key_version:
                key_data = keys.get(str(self._key_version), {})
            else:
                latest_version = max(int(v) for v in keys.keys())
                key_data = keys[str(latest_version)]
            public_key_pem = key_data.get("public_key", "")
            return public_key_pem.encode("utf-8")
        except ImportError as e:
            raise ImportError(str(e))
        except Exception as e:
            logger.error(f"Failed to get Vault public key: {e}")
            raise SigningError(f"Failed to get Vault public key: {e}") from e


# ============================================================================
# SIGNATURE VERIFIER
# ============================================================================

class SignatureVerifier:
    """Signature verification service for audit packages."""

    def __init__(self):
        self._public_keys: Dict[str, Tuple[Any, SignatureAlgorithm]] = {}
        self._key_info: Dict[str, PublicKeyInfo] = {}
        logger.info("Initialized SignatureVerifier")

    def add_public_key(
        self,
        key_id: str,
        public_key: bytes,
        algorithm: SignatureAlgorithm
    ) -> None:
        """Add a public key to the verifier registry."""
        try:
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            raise ImportError("cryptography library required. Install: pip install cryptography")
        try:
            loaded_key = serialization.load_pem_public_key(public_key)
            self._public_keys[key_id] = (loaded_key, algorithm)
            self._key_info[key_id] = PublicKeyInfo(
                key_id=key_id,
                algorithm=algorithm,
                public_key_pem=public_key.decode("utf-8"),
                created_at=datetime.now(timezone.utc)
            )
            logger.info(f"Added public key to verifier: key_id={key_id}")
        except Exception as e:
            logger.error(f"Failed to add public key: {e}")
            raise SigningError(f"Failed to add public key: {e}") from e

    def add_public_key_info(self, key_info: PublicKeyInfo) -> None:
        """Add a public key using PublicKeyInfo."""
        self.add_public_key(
            key_id=key_info.key_id,
            public_key=key_info.public_key_pem.encode("utf-8"),
            algorithm=key_info.algorithm
        )
        self._key_info[key_info.key_id] = key_info

    def get_key_info(self, key_id: str) -> Optional[PublicKeyInfo]:
        """Get information about a registered key."""
        return self._key_info.get(key_id)

    def list_keys(self) -> List[PublicKeyInfo]:
        """List all registered public keys."""
        return list(self._key_info.values())

    def verify(self, data: bytes, signature: bytes, key_id: str) -> bool:
        """Verify a signature against registered public key."""
        if key_id not in self._public_keys:
            raise KeyNotFoundError(key_id)
        public_key, algorithm = self._public_keys[key_id]
        try:
            if algorithm == SignatureAlgorithm.ED25519:
                public_key.verify(signature, data)
                return True
            elif algorithm in (
                SignatureAlgorithm.RSA_PSS_SHA256,
                SignatureAlgorithm.RSA_PSS_SHA384,
                SignatureAlgorithm.RSA_PSS_SHA512
            ):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import padding
                hash_map = {
                    SignatureAlgorithm.RSA_PSS_SHA256: hashes.SHA256(),
                    SignatureAlgorithm.RSA_PSS_SHA384: hashes.SHA384(),
                    SignatureAlgorithm.RSA_PSS_SHA512: hashes.SHA512()
                }
                public_key.verify(
                    signature,
                    data,
                    padding.PSS(
                        mgf=padding.MGF1(hash_map[algorithm]),
                        salt_length=padding.PSS.AUTO
                    ),
                    hash_map[algorithm]
                )
                return True
            elif algorithm in (
                SignatureAlgorithm.ECDSA_P256_SHA256,
                SignatureAlgorithm.ECDSA_P384_SHA384
            ):
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.asymmetric import ec
                hash_map = {
                    SignatureAlgorithm.ECDSA_P256_SHA256: hashes.SHA256(),
                    SignatureAlgorithm.ECDSA_P384_SHA384: hashes.SHA384()
                }
                public_key.verify(signature, data, ec.ECDSA(hash_map[algorithm]))
                return True
            else:
                logger.error(f"Unsupported algorithm: {algorithm}")
                return False
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False

    def verify_bundle(self, data: bytes, bundle: SignatureBundle) -> bool:
        """Verify a signature bundle (checks hash and signature)."""
        computed_hash = hashlib.sha256(data).hexdigest()
        if computed_hash != bundle.package_hash:
            raise SignatureVerificationError(
                f"Package hash mismatch: expected {bundle.package_hash}, computed {computed_hash}",
                key_id=bundle.metadata.key_id
            )
        signature_bytes = base64.b64decode(bundle.signature)
        return self.verify(data=data, signature=signature_bytes, key_id=bundle.metadata.key_id)


# ============================================================================
# SIGNING PROVIDER FACTORY
# ============================================================================

class SigningProviderFactory:
    """Factory for creating signing providers from configuration."""

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> SigningProvider:
        """Create a signing provider from configuration dictionary."""
        provider_type = config.get("provider", "local").lower()
        if provider_type == "local":
            return cls._create_local_signer(config)
        elif provider_type == "aws_kms":
            return cls._create_kms_signer(config)
        elif provider_type == "hashicorp_vault":
            return cls._create_vault_signer(config)
        else:
            raise ProviderConfigError(f"Unknown provider type: {provider_type}")

    @classmethod
    def _create_local_signer(cls, config: Dict[str, Any]) -> LocalKeySigner:
        key_id = config.get("key_id")
        if not key_id:
            raise ProviderConfigError("key_id is required for local provider")
        algorithm_str = config.get("algorithm", "Ed25519")
        try:
            algorithm = SignatureAlgorithm(algorithm_str)
        except ValueError:
            raise ProviderConfigError(f"Invalid algorithm: {algorithm_str}")
        private_key_path = config.get("private_key_path")
        if private_key_path:
            password = config.get("key_password")
            if password:
                password = password.encode("utf-8")
            return LocalKeySigner.load_from_file(private_key_path, key_id, algorithm, password)
        else:
            if algorithm == SignatureAlgorithm.ED25519:
                return LocalKeySigner.generate_ed25519(key_id)
            elif algorithm.value.startswith("RSA"):
                key_size = config.get("key_size", 3072)
                hash_algo = algorithm.value.split("-")[-1]
                return LocalKeySigner.generate_rsa(key_id, key_size, hash_algo)
            elif algorithm.value.startswith("ECDSA"):
                curve = "P256" if "P256" in algorithm.value else "P384"
                return LocalKeySigner.generate_ecdsa(key_id, curve)
            else:
                raise ProviderConfigError(f"Cannot generate key for algorithm: {algorithm}")

    @classmethod
    def _create_kms_signer(cls, config: Dict[str, Any]) -> KMSSigner:
        key_id = config.get("key_id")
        if not key_id:
            raise ProviderConfigError("key_id is required for AWS KMS provider")
        algorithm_str = config.get("algorithm", "ECDSA-P256-SHA256")
        try:
            algorithm = SignatureAlgorithm(algorithm_str)
        except ValueError:
            raise ProviderConfigError(f"Invalid algorithm: {algorithm_str}")
        return KMSSigner(key_id, config.get("region", "us-east-1"), algorithm, config.get("profile_name"))

    @classmethod
    def _create_vault_signer(cls, config: Dict[str, Any]) -> VaultSigner:
        vault_url = config.get("vault_url")
        if not vault_url:
            raise ProviderConfigError("vault_url is required for Vault provider")
        key_name = config.get("key_name")
        if not key_name:
            raise ProviderConfigError("key_name is required for Vault provider")
        algorithm_str = config.get("algorithm", "Ed25519")
        try:
            algorithm = SignatureAlgorithm(algorithm_str)
        except ValueError:
            raise ProviderConfigError(f"Invalid algorithm: {algorithm_str}")
        return VaultSigner(
            vault_url, key_name, config.get("token"),
            config.get("mount_point", "transit"), algorithm, config.get("key_version")
        )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "SignatureAlgorithm", "KeyType", "ProviderType",
    "SignatureMetadata", "SignatureBundle", "PublicKeyInfo",
    "SigningError", "KeyNotFoundError", "SignatureVerificationError",
    "ProviderConfigError", "KeyRotationError",
    "SigningProvider", "LocalKeySigner", "KMSSigner", "VaultSigner",
    "SignatureVerifier", "SigningProviderFactory",
]
