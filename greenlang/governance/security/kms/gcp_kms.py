"""
Google Cloud KMS Provider Implementation
=========================================

Integrates with Google Cloud Key Management Service for secure key operations.
"""

import base64
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .base_kms import (
    BaseKMSProvider,
    KMSConfig,
    KMSSignResult,
    KMSKeyInfo,
    KMSProviderError,
    KMSKeyNotFoundError,
    KMSSigningError,
    KMSKeyRotationError,
    KeyAlgorithm,
    SigningAlgorithm,
)

logger = logging.getLogger(__name__)

# Try to import Google Cloud SDK
try:
    from google.cloud import kms
    from google.api_core.exceptions import NotFound, GoogleAPIError
    from google.oauth2 import service_account
    import crc32c
    GCP_SDK_AVAILABLE = True
except ImportError:
    GCP_SDK_AVAILABLE = False
    logger.warning("Google Cloud SDK not available - GCP KMS support disabled")

# Try to import async support
try:
    from google.cloud import kms_v1
    from google.api_core import retry_async
    import aiohttp
    GCP_ASYNC_AVAILABLE = True
except ImportError:
    GCP_ASYNC_AVAILABLE = False
    logger.info("GCP async support not available")


class GCPCloudKMSProvider(BaseKMSProvider):
    """
    Google Cloud KMS provider implementation

    Supports:
    - Symmetric encryption keys
    - Asymmetric signing keys (RSA, EC)
    - Hardware Security Module (HSM) keys
    - Key rotation and versioning
    - Global and regional keys
    """

    # Map our algorithms to GCP algorithms
    ALGORITHM_MAP = {
        SigningAlgorithm.RSASSA_PSS_SHA256: "RSA_SIGN_PSS_2048_SHA256",
        SigningAlgorithm.RSASSA_PSS_SHA384: "RSA_SIGN_PSS_3072_SHA384",
        SigningAlgorithm.RSASSA_PSS_SHA512: "RSA_SIGN_PSS_4096_SHA512",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA256: "RSA_SIGN_PKCS1_2048_SHA256",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA384: "RSA_SIGN_PKCS1_3072_SHA384",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA512: "RSA_SIGN_PKCS1_4096_SHA512",
        SigningAlgorithm.ECDSA_SHA256: "EC_SIGN_P256_SHA256",
        SigningAlgorithm.ECDSA_SHA384: "EC_SIGN_P384_SHA384",
        # GCP doesn't support P521 ECDSA
    }

    # Map GCP algorithms to our key algorithms
    PURPOSE_ALGORITHM_MAP = {
        "RSA_SIGN_PSS_2048_SHA256": KeyAlgorithm.RSA_2048,
        "RSA_SIGN_PSS_3072_SHA384": KeyAlgorithm.RSA_3072,
        "RSA_SIGN_PSS_4096_SHA512": KeyAlgorithm.RSA_4096,
        "RSA_SIGN_PKCS1_2048_SHA256": KeyAlgorithm.RSA_2048,
        "RSA_SIGN_PKCS1_3072_SHA384": KeyAlgorithm.RSA_3072,
        "RSA_SIGN_PKCS1_4096_SHA512": KeyAlgorithm.RSA_4096,
        "EC_SIGN_P256_SHA256": KeyAlgorithm.ECDSA_P256,
        "EC_SIGN_P384_SHA384": KeyAlgorithm.ECDSA_P384,
    }

    def __init__(self, config: KMSConfig):
        """Initialize GCP Cloud KMS provider"""
        if not GCP_SDK_AVAILABLE:
            raise KMSProviderError(
                "Google Cloud SDK is required for GCP KMS: pip install google-cloud-kms crc32c"
            )

        super().__init__(config)

        # Parse GCP-specific configuration
        self.project_id = config.gcp_project_id
        self.location_id = config.gcp_location_id or "global"
        self.keyring_id = config.gcp_keyring_id

        if not self.project_id:
            raise ValueError("GCP project ID required (gcp_project_id)")
        if not self.keyring_id:
            raise ValueError("GCP keyring ID required (gcp_keyring_id)")

        # Parse key name from key_id
        self.key_name = self._parse_key_name(config.key_id)

        # Set async availability
        self.config.async_enabled = self.config.async_enabled and GCP_ASYNC_AVAILABLE

    def _parse_key_name(self, key_id: str) -> str:
        """Parse key name from key ID"""
        if not key_id:
            raise ValueError("Key ID required")

        # If it's a full resource name, extract the key name
        if key_id.startswith("projects/"):
            parts = key_id.split("/")
            if len(parts) >= 8 and parts[4] == "keyRings" and parts[6] == "cryptoKeys":
                return parts[7]
            return key_id

        # Otherwise assume it's just the key name
        return key_id

    def _get_key_resource_name(self, key_name: Optional[str] = None) -> str:
        """Build full GCP resource name for a key"""
        key_name = key_name or self.key_name
        return (
            f"projects/{self.project_id}/locations/{self.location_id}/"
            f"keyRings/{self.keyring_id}/cryptoKeys/{key_name}"
        )

    def _get_key_version_resource_name(self, key_name: str, version: str = "1") -> str:
        """Build full GCP resource name for a key version"""
        key_resource = self._get_key_resource_name(key_name)
        return f"{key_resource}/cryptoKeyVersions/{version}"

    def _create_client(self) -> Any:
        """Create GCP KMS client"""
        try:
            # Create client with credentials if provided
            if hasattr(self.config, 'gcp_credentials_path'):
                credentials = service_account.Credentials.from_service_account_file(
                    getattr(self.config, 'gcp_credentials_path')
                )
                client = kms.KeyManagementServiceClient(credentials=credentials)
            else:
                # Use default credentials
                client = kms.KeyManagementServiceClient()

            # Test connectivity by getting key
            key_name = self._get_key_resource_name()
            try:
                key = client.get_crypto_key(request={"name": key_name})
                logger.info(f"Connected to GCP KMS: {key_name}")
            except NotFound:
                raise KMSKeyNotFoundError(f"Key {self.key_name} not found")
            except GoogleAPIError as e:
                raise KMSProviderError(f"Failed to connect to GCP KMS: {e}")

            return client

        except Exception as e:
            raise KMSProviderError(f"Failed to create GCP KMS client: {e}")

    def _create_async_client(self) -> Any:
        """Create async GCP KMS client"""
        # GCP doesn't have native async client, would need custom implementation
        return None

    def _crc32c_checksum(self, data: bytes) -> int:
        """Calculate CRC32C checksum for data integrity"""
        if not GCP_SDK_AVAILABLE:
            # Fallback to simple checksum if crc32c not available
            return hash(data) & 0xFFFFFFFF

        try:
            import crc32c
            return int.from_bytes(crc32c.crc32c(data).to_bytes(4, 'big'), 'big')
        except ImportError:
            # Fallback to simple checksum
            return hash(data) & 0xFFFFFFFF

    def get_key_info(self, key_id: Optional[str] = None) -> KMSKeyInfo:
        """Get information about a GCP KMS key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name
        resource_name = self._get_key_resource_name(key_name)

        # Check cache
        cached = self.cache.get(resource_name)
        if cached:
            return cached

        try:
            # Get the key
            key = self.retry_with_backoff(
                self.client.get_crypto_key,
                request={"name": resource_name}
            )

            # Get primary version
            primary_version = None
            if hasattr(key, 'primary') and key.primary:
                primary_version = key.primary.name.split('/')[-1]

            # Determine algorithm from key purpose and algorithm
            algorithm = KeyAlgorithm.RSA_2048  # Default
            if hasattr(key, 'version_template') and key.version_template:
                gcp_algorithm = key.version_template.algorithm
                algorithm = self.PURPOSE_ALGORITHM_MAP.get(
                    kms.CryptoKeyVersion.CryptoKeyVersionAlgorithm(gcp_algorithm).name,
                    KeyAlgorithm.RSA_2048
                )

            # Get public key if it's an asymmetric key
            public_key = None
            if key.purpose == kms.CryptoKey.CryptoKeyPurpose.ASYMMETRIC_SIGN:
                try:
                    # Get the public key for the primary version
                    version_name = self._get_key_version_resource_name(key_name, primary_version or "1")
                    public_key_response = self.client.get_public_key(
                        request={"name": version_name}
                    )
                    public_key = public_key_response.pem.encode() if public_key_response.pem else None
                except Exception as e:
                    logger.warning(f"Could not retrieve public key: {e}")

            key_info = KMSKeyInfo(
                key_id=key_name,
                key_arn=resource_name,
                algorithm=algorithm,
                created_at=key.create_time if hasattr(key, 'create_time') else datetime.utcnow(),
                enabled=True,  # GCP keys don't have enabled/disabled state like AWS
                rotation_enabled=bool(key.rotation_period) if hasattr(key, 'rotation_period') else False,
                key_version=primary_version,
                public_key=public_key,
                metadata={
                    "purpose": kms.CryptoKey.CryptoKeyPurpose(key.purpose).name,
                    "protection_level": key.version_template.protection_level if key.version_template else None,
                    "labels": dict(key.labels) if key.labels else {},
                    "import_only": bool(key.import_only) if hasattr(key, 'import_only') else False,
                }
            )

            # Cache the result
            self.cache.set(resource_name, key_info)

            return key_info

        except NotFound:
            raise KMSKeyNotFoundError(f"Key {key_name} not found in GCP KMS")
        except Exception as e:
            raise KMSProviderError(f"Failed to get key info: {e}")

    def sign(self, data: bytes, key_id: Optional[str] = None,
             algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Sign data using GCP KMS key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        # Get key info to determine version
        key_info = self.get_cached_key_info(key_name)
        version = key_info.key_version or "1"

        # Build version resource name
        version_name = self._get_key_version_resource_name(key_name, version)

        # Hash the data based on algorithm
        if algorithm:
            if "SHA256" in algorithm.value:
                digest = hashlib.sha256(data).digest()
                digest_obj = {"sha256": digest}
            elif "SHA384" in algorithm.value:
                digest = hashlib.sha384(data).digest()
                digest_obj = {"sha384": digest}
            elif "SHA512" in algorithm.value:
                digest = hashlib.sha512(data).digest()
                digest_obj = {"sha512": digest}
            else:
                digest = hashlib.sha256(data).digest()
                digest_obj = {"sha256": digest}
        else:
            # Default to SHA256
            digest = hashlib.sha256(data).digest()
            digest_obj = {"sha256": digest}

        try:
            # Calculate CRC32C for data integrity
            digest_crc32c = self._crc32c_checksum(
                next(iter(digest_obj.values()))
            )

            # Sign the digest
            request = {
                "name": version_name,
                "digest": digest_obj,
                "digest_crc32c": digest_crc32c,
            }

            response = self.retry_with_backoff(
                self.client.asymmetric_sign,
                request=request
            )

            # Verify response integrity
            if hasattr(response, 'signature_crc32c'):
                if response.signature_crc32c != self._crc32c_checksum(response.signature):
                    raise KMSSigningError("Signature CRC32C verification failed")

            # Verify the signature was created by the expected key version
            if hasattr(response, 'name') and response.name != version_name:
                logger.warning(f"Signature created by different version: {response.name}")

            return KMSSignResult(
                signature=response.signature,
                key_id=key_name,
                algorithm=str(algorithm) if algorithm else "SHA256",
                timestamp=datetime.utcnow().isoformat(),
                key_version=version,
                provider="gcp"
            )

        except NotFound:
            raise KMSKeyNotFoundError(f"Key version {version_name} not found")
        except GoogleAPIError as e:
            raise KMSSigningError(f"GCP KMS signing failed: {e}")
        except Exception as e:
            raise KMSSigningError(f"Unexpected error during signing: {e}")

    def verify(self, data: bytes, signature: bytes,
               key_id: Optional[str] = None,
               algorithm: Optional[SigningAlgorithm] = None) -> bool:
        """Verify signature using GCP KMS key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        # Get public key
        key_info = self.get_cached_key_info(key_name)
        if not key_info.public_key:
            # Fetch public key
            version = key_info.key_version or "1"
            version_name = self._get_key_version_resource_name(key_name, version)

            try:
                public_key_response = self.client.get_public_key(
                    request={"name": version_name}
                )
                public_key_pem = public_key_response.pem
            except Exception as e:
                raise KMSSigningError(f"Failed to get public key: {e}")
        else:
            public_key_pem = key_info.public_key.decode() if isinstance(key_info.public_key, bytes) else key_info.public_key

        # Use cryptography library to verify
        try:
            from cryptography.hazmat.primitives import serialization, hashes
            from cryptography.hazmat.primitives.asymmetric import padding, utils
            from cryptography.hazmat.backends import default_backend
            from cryptography.exceptions import InvalidSignature

            # Load public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode() if isinstance(public_key_pem, str) else public_key_pem,
                backend=default_backend()
            )

            # Hash the data
            if algorithm and "SHA384" in algorithm.value:
                hash_algo = hashes.SHA384()
            elif algorithm and "SHA512" in algorithm.value:
                hash_algo = hashes.SHA512()
            else:
                hash_algo = hashes.SHA256()

            digest = hashes.Hash(hash_algo, backend=default_backend())
            digest.update(data)
            hashed = digest.finalize()

            # Verify based on key type
            from cryptography.hazmat.primitives.asymmetric import rsa, ec

            if isinstance(public_key, rsa.RSAPublicKey):
                # Determine padding based on algorithm
                if algorithm and "PSS" in algorithm.value:
                    pad = padding.PSS(
                        mgf=padding.MGF1(hash_algo),
                        salt_length=padding.PSS.DIGEST_LENGTH
                    )
                else:
                    pad = padding.PKCS1v15()

                public_key.verify(signature, data, pad, hash_algo)
            elif isinstance(public_key, ec.EllipticCurvePublicKey):
                public_key.verify(signature, data, ec.ECDSA(hash_algo))
            else:
                raise KMSSigningError(f"Unsupported key type: {type(public_key)}")

            return True

        except InvalidSignature:
            return False
        except ImportError:
            raise KMSProviderError("cryptography library required for verification")
        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False

    def rotate_key(self, key_id: Optional[str] = None) -> str:
        """Rotate a GCP KMS key by creating a new version"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name
        resource_name = self._get_key_resource_name(key_name)

        try:
            # Get current key
            key = self.client.get_crypto_key(request={"name": resource_name})

            # Create new key version
            request = {
                "parent": resource_name,
                "crypto_key_version": {
                    "state": kms.CryptoKeyVersion.CryptoKeyVersionState.ENABLED
                }
            }

            new_version = self.retry_with_backoff(
                self.client.create_crypto_key_version,
                request=request
            )

            # Set as primary version
            key.primary = new_version
            update_mask = {"paths": ["primary"]}

            self.client.update_crypto_key(
                request={
                    "crypto_key": key,
                    "update_mask": update_mask
                }
            )

            # Invalidate cache
            self.cache.invalidate(resource_name)

            # Extract version number
            version_num = new_version.name.split('/')[-1]
            logger.info(f"Created new key version for {key_name}: {version_num}")

            return version_num

        except NotFound:
            raise KMSKeyNotFoundError(f"Key {key_name} not found")
        except GoogleAPIError as e:
            raise KMSKeyRotationError(f"GCP KMS rotation failed: {e}")
        except Exception as e:
            raise KMSKeyRotationError(f"Unexpected error during rotation: {e}")

    async def sign_async(self, data: bytes, key_id: Optional[str] = None,
                        algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Async signing using GCP KMS (falls back to sync)"""
        # GCP doesn't have native async support, fall back to sync
        return self.sign(data, key_id, algorithm)

    def list_keys(self, limit: int = 100) -> List[str]:
        """
        List available keys in the keyring

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of key names
        """
        try:
            parent = f"projects/{self.project_id}/locations/{self.location_id}/keyRings/{self.keyring_id}"

            key_names = []
            count = 0

            # List crypto keys
            request = {"parent": parent}
            page_result = self.client.list_crypto_keys(request=request)

            for key in page_result:
                # Extract key name from full resource name
                key_name = key.name.split('/')[-1]
                key_names.append(key_name)
                count += 1

                if count >= limit:
                    break

            return key_names

        except Exception as e:
            raise KMSProviderError(f"Failed to list keys: {e}")

    def enable_rotation(self, key_id: Optional[str] = None,
                       rotation_period_days: int = 90) -> bool:
        """
        Enable automatic rotation for a key

        Args:
            key_id: Key identifier
            rotation_period_days: Days between rotations

        Returns:
            True if successful
        """
        key_name = self._parse_key_name(key_id) if key_id else self.key_name
        resource_name = self._get_key_resource_name(key_name)

        try:
            # Get current key
            key = self.client.get_crypto_key(request={"name": resource_name})

            # Set rotation period (in seconds)
            from google.protobuf.duration_pb2 import Duration
            rotation_period = Duration()
            rotation_period.seconds = rotation_period_days * 24 * 3600

            key.rotation_period = rotation_period

            # Update the key
            update_mask = {"paths": ["rotation_period"]}
            self.client.update_crypto_key(
                request={
                    "crypto_key": key,
                    "update_mask": update_mask
                }
            )

            # Invalidate cache
            self.cache.invalidate(resource_name)

            logger.info(f"Enabled rotation for {key_name} with period {rotation_period_days} days")
            return True

        except Exception as e:
            raise KMSKeyRotationError(f"Failed to enable rotation: {e}")

    def destroy_key_version(self, key_id: Optional[str] = None,
                           version: str = None) -> bool:
        """
        Schedule destruction of a key version

        Args:
            key_id: Key identifier
            version: Version to destroy

        Returns:
            True if successful
        """
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        if not version:
            raise ValueError("Version required for destruction")

        version_name = self._get_key_version_resource_name(key_name, version)

        try:
            # Schedule destruction
            response = self.retry_with_backoff(
                self.client.destroy_crypto_key_version,
                request={"name": version_name}
            )

            logger.warning(f"Scheduled destruction of key version {version_name}")
            return response.state == kms.CryptoKeyVersion.CryptoKeyVersionState.DESTROY_SCHEDULED

        except Exception as e:
            raise KMSProviderError(f"Failed to destroy key version: {e}")