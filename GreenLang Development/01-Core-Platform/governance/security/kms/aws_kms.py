"""
AWS KMS Provider Implementation
================================

Integrates with AWS Key Management Service for secure key operations.
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

# Try to import boto3
try:
    import boto3
    from botocore.exceptions import (
        ClientError,
        BotoCoreError,
        NoCredentialsError,
    )
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - AWS KMS support disabled")

# Try to import aioboto3 for async support
try:
    import aioboto3
    AIOBOTO3_AVAILABLE = True
except ImportError:
    AIOBOTO3_AVAILABLE = False
    logger.info("aioboto3 not available - async AWS KMS support disabled")


class AWSKMSProvider(BaseKMSProvider):
    """
    AWS KMS provider implementation

    Supports:
    - Customer Master Keys (CMKs)
    - Asymmetric signing keys (RSA, ECC)
    - Symmetric encryption keys
    - Key rotation
    - Multi-region keys
    """

    # Map our algorithms to AWS algorithms
    ALGORITHM_MAP = {
        SigningAlgorithm.RSASSA_PSS_SHA256: "RSASSA_PSS_SHA_256",
        SigningAlgorithm.RSASSA_PSS_SHA384: "RSASSA_PSS_SHA_384",
        SigningAlgorithm.RSASSA_PSS_SHA512: "RSASSA_PSS_SHA_512",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA256: "RSASSA_PKCS1_V1_5_SHA_256",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA384: "RSASSA_PKCS1_V1_5_SHA_384",
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA512: "RSASSA_PKCS1_V1_5_SHA_512",
        SigningAlgorithm.ECDSA_SHA256: "ECDSA_SHA_256",
        SigningAlgorithm.ECDSA_SHA384: "ECDSA_SHA_384",
        SigningAlgorithm.ECDSA_SHA512: "ECDSA_SHA_512",
    }

    # Map AWS key specs to our key algorithms
    KEY_SPEC_MAP = {
        "RSA_2048": KeyAlgorithm.RSA_2048,
        "RSA_3072": KeyAlgorithm.RSA_3072,
        "RSA_4096": KeyAlgorithm.RSA_4096,
        "ECC_NIST_P256": KeyAlgorithm.ECDSA_P256,
        "ECC_NIST_P384": KeyAlgorithm.ECDSA_P384,
        "ECC_NIST_P521": KeyAlgorithm.ECDSA_P521,
    }

    def __init__(self, config: KMSConfig):
        """Initialize AWS KMS provider"""
        if not BOTO3_AVAILABLE:
            raise KMSProviderError("boto3 is required for AWS KMS: pip install boto3")

        super().__init__(config)

        # Validate AWS-specific config
        if not config.key_id:
            raise ValueError("AWS KMS key ID or ARN required")

        # Set async availability based on library
        self.config.async_enabled = self.config.async_enabled and AIOBOTO3_AVAILABLE

    def _create_client(self) -> Any:
        """Create AWS KMS client"""
        try:
            session_kwargs = {}
            if self.config.aws_profile:
                session_kwargs["profile_name"] = self.config.aws_profile

            session = boto3.Session(**session_kwargs)

            client_kwargs = {
                "service_name": "kms",
            }

            if self.config.region:
                client_kwargs["region_name"] = self.config.region

            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            client = session.client(**client_kwargs)

            # Test connectivity
            try:
                client.describe_key(KeyId=self.config.key_id)
                logger.info(f"Connected to AWS KMS in region {client.meta.region_name}")
            except ClientError as e:
                if e.response["Error"]["Code"] == "NotFoundException":
                    raise KMSKeyNotFoundError(f"Key {self.config.key_id} not found")
                raise KMSProviderError(f"Failed to connect to AWS KMS: {e}")

            return client

        except NoCredentialsError:
            raise KMSProviderError(
                "AWS credentials not found. Configure AWS CLI or set environment variables."
            )
        except Exception as e:
            raise KMSProviderError(f"Failed to create AWS KMS client: {e}")

    def _create_async_client(self) -> Any:
        """Create async AWS KMS client"""
        if not AIOBOTO3_AVAILABLE:
            return None

        try:
            session_kwargs = {}
            if self.config.aws_profile:
                session_kwargs["profile_name"] = self.config.aws_profile

            session = aioboto3.Session(**session_kwargs)

            client_kwargs = {
                "service_name": "kms",
            }

            if self.config.region:
                client_kwargs["region_name"] = self.config.region

            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            # Return context manager for async client
            return session.client(**client_kwargs)

        except Exception as e:
            logger.warning(f"Failed to create async AWS KMS client: {e}")
            return None

    def get_key_info(self, key_id: Optional[str] = None) -> KMSKeyInfo:
        """Get information about an AWS KMS key"""
        key_id = key_id or self.config.key_id

        # Check cache
        cached = self.cache.get(key_id)
        if cached:
            return cached

        try:
            # Describe the key
            response = self.retry_with_backoff(
                self.client.describe_key,
                KeyId=key_id
            )

            key_metadata = response["KeyMetadata"]

            # Get public key if it's an asymmetric key
            public_key = None
            if key_metadata.get("KeyUsage") == "SIGN_VERIFY":
                try:
                    public_key_response = self.client.get_public_key(KeyId=key_id)
                    public_key = public_key_response["PublicKey"]
                except ClientError:
                    logger.warning(f"Could not retrieve public key for {key_id}")

            # Map key spec to our algorithm enum
            key_spec = key_metadata.get("KeySpec", "UNKNOWN")
            algorithm = self.KEY_SPEC_MAP.get(key_spec, KeyAlgorithm.RSA_2048)

            key_info = KMSKeyInfo(
                key_id=key_metadata["KeyId"],
                key_arn=key_metadata["Arn"],
                algorithm=algorithm,
                created_at=key_metadata["CreationDate"],
                enabled=key_metadata["Enabled"],
                rotation_enabled=key_metadata.get("KeyRotationEnabled", False),
                public_key=public_key,
                metadata={
                    "key_usage": key_metadata.get("KeyUsage"),
                    "key_state": key_metadata.get("KeyState"),
                    "key_spec": key_spec,
                    "multi_region": key_metadata.get("MultiRegion", False),
                    "description": key_metadata.get("Description"),
                }
            )

            # Cache the result
            self.cache.set(key_id, key_info)

            return key_info

        except ClientError as e:
            if e.response["Error"]["Code"] == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key {key_id} not found in AWS KMS")
            raise KMSProviderError(f"Failed to get key info: {e}")
        except Exception as e:
            raise KMSProviderError(f"Unexpected error getting key info: {e}")

    def sign(self, data: bytes, key_id: Optional[str] = None,
             algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Sign data using AWS KMS key"""
        key_id = key_id or self.config.key_id

        # Determine signing algorithm
        if algorithm:
            aws_algorithm = self.ALGORITHM_MAP.get(algorithm)
            if not aws_algorithm:
                raise KMSSigningError(f"Unsupported algorithm for AWS KMS: {algorithm}")
        else:
            # Use default based on key type
            key_info = self.get_cached_key_info(key_id)
            if key_info.algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_3072, KeyAlgorithm.RSA_4096]:
                aws_algorithm = "RSASSA_PSS_SHA_256"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P256:
                aws_algorithm = "ECDSA_SHA_256"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P384:
                aws_algorithm = "ECDSA_SHA_384"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P521:
                aws_algorithm = "ECDSA_SHA_512"
            else:
                aws_algorithm = "RSASSA_PSS_SHA_256"  # Default

        try:
            # Hash the data first (AWS KMS requires pre-hashed data for some algorithms)
            if "SHA_256" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha256(data).digest()
            elif "SHA_384" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha384(data).digest()
            elif "SHA_512" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha512(data).digest()
            else:
                message_type = "RAW"
                message = data

            # Sign the data
            response = self.retry_with_backoff(
                self.client.sign,
                KeyId=key_id,
                Message=message,
                MessageType=message_type,
                SigningAlgorithm=aws_algorithm
            )

            return KMSSignResult(
                signature=response["Signature"],
                key_id=response["KeyId"],
                algorithm=response["SigningAlgorithm"],
                timestamp=datetime.utcnow().isoformat(),
                key_version=None,  # AWS doesn't expose key version in sign response
                provider="aws"
            )

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key {key_id} not found")
            elif error_code == "DisabledException":
                raise KMSSigningError(f"Key {key_id} is disabled")
            elif error_code == "KeyUnavailableException":
                raise KMSSigningError(f"Key {key_id} is unavailable")
            raise KMSSigningError(f"AWS KMS signing failed: {e}")
        except Exception as e:
            raise KMSSigningError(f"Unexpected error during signing: {e}")

    def verify(self, data: bytes, signature: bytes,
               key_id: Optional[str] = None,
               algorithm: Optional[SigningAlgorithm] = None) -> bool:
        """Verify signature using AWS KMS key"""
        key_id = key_id or self.config.key_id

        # Determine signing algorithm
        if algorithm:
            aws_algorithm = self.ALGORITHM_MAP.get(algorithm)
            if not aws_algorithm:
                raise KMSSigningError(f"Unsupported algorithm for AWS KMS: {algorithm}")
        else:
            # Try to detect from key
            key_info = self.get_cached_key_info(key_id)
            if key_info.algorithm in [KeyAlgorithm.RSA_2048, KeyAlgorithm.RSA_3072, KeyAlgorithm.RSA_4096]:
                aws_algorithm = "RSASSA_PSS_SHA_256"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P256:
                aws_algorithm = "ECDSA_SHA_256"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P384:
                aws_algorithm = "ECDSA_SHA_384"
            elif key_info.algorithm == KeyAlgorithm.ECDSA_P521:
                aws_algorithm = "ECDSA_SHA_512"
            else:
                aws_algorithm = "RSASSA_PSS_SHA_256"

        try:
            # Hash the data if needed
            if "SHA_256" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha256(data).digest()
            elif "SHA_384" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha384(data).digest()
            elif "SHA_512" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha512(data).digest()
            else:
                message_type = "RAW"
                message = data

            # Verify the signature
            response = self.retry_with_backoff(
                self.client.verify,
                KeyId=key_id,
                Message=message,
                MessageType=message_type,
                Signature=signature,
                SigningAlgorithm=aws_algorithm
            )

            return response["SignatureValid"]

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key {key_id} not found")
            elif error_code == "KMSInvalidSignatureException":
                return False
            raise KMSSigningError(f"AWS KMS verification failed: {e}")
        except Exception as e:
            raise KMSSigningError(f"Unexpected error during verification: {e}")

    def rotate_key(self, key_id: Optional[str] = None) -> str:
        """Enable automatic key rotation for AWS KMS key"""
        key_id = key_id or self.config.key_id

        try:
            # Enable key rotation
            self.retry_with_backoff(
                self.client.enable_key_rotation,
                KeyId=key_id
            )

            # Invalidate cache for this key
            self.cache.invalidate(key_id)

            # Get rotation status
            response = self.client.get_key_rotation_status(KeyId=key_id)

            if response["KeyRotationEnabled"]:
                logger.info(f"Key rotation enabled for {key_id}")
                return key_id  # AWS manages versions internally
            else:
                raise KMSKeyRotationError(f"Failed to enable rotation for {key_id}")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NotFoundException":
                raise KMSKeyNotFoundError(f"Key {key_id} not found")
            elif error_code == "UnsupportedOperationException":
                raise KMSKeyRotationError(f"Key {key_id} does not support rotation")
            raise KMSKeyRotationError(f"AWS KMS rotation failed: {e}")
        except Exception as e:
            raise KMSKeyRotationError(f"Unexpected error during rotation: {e}")

    async def sign_async(self, data: bytes, key_id: Optional[str] = None,
                        algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Async signing using AWS KMS"""
        if not self.config.async_enabled or not self.async_client:
            # Fall back to sync
            return self.sign(data, key_id, algorithm)

        key_id = key_id or self.config.key_id

        # Determine signing algorithm
        if algorithm:
            aws_algorithm = self.ALGORITHM_MAP.get(algorithm)
            if not aws_algorithm:
                raise KMSSigningError(f"Unsupported algorithm for AWS KMS: {algorithm}")
        else:
            # Use default
            aws_algorithm = "RSASSA_PSS_SHA_256"

        try:
            # Hash the data
            if "SHA_256" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha256(data).digest()
            elif "SHA_384" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha384(data).digest()
            elif "SHA_512" in aws_algorithm:
                message_type = "DIGEST"
                message = hashlib.sha512(data).digest()
            else:
                message_type = "RAW"
                message = data

            # Use async client
            async with self.async_client as client:
                response = await client.sign(
                    KeyId=key_id,
                    Message=message,
                    MessageType=message_type,
                    SigningAlgorithm=aws_algorithm
                )

                return KMSSignResult(
                    signature=response["Signature"],
                    key_id=response["KeyId"],
                    algorithm=response["SigningAlgorithm"],
                    timestamp=datetime.utcnow().isoformat(),
                    key_version=None,
                    provider="aws"
                )

        except Exception as e:
            raise KMSSigningError(f"Async signing failed: {e}")

    def create_data_key(self, key_id: Optional[str] = None,
                       key_spec: str = "AES_256") -> Dict[str, bytes]:
        """
        Generate a data encryption key (DEK) for envelope encryption

        Args:
            key_id: Master key ID
            key_spec: Key specification (AES_256 or AES_128)

        Returns:
            Dict with 'plaintext' and 'ciphertext' keys
        """
        key_id = key_id or self.config.key_id

        try:
            response = self.retry_with_backoff(
                self.client.generate_data_key,
                KeyId=key_id,
                KeySpec=key_spec
            )

            return {
                "plaintext": response["Plaintext"],
                "ciphertext": response["CiphertextBlob"]
            }

        except ClientError as e:
            raise KMSProviderError(f"Failed to generate data key: {e}")

    def list_keys(self, limit: int = 100) -> List[str]:
        """
        List available KMS keys

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of key IDs
        """
        try:
            paginator = self.client.get_paginator("list_keys")
            key_ids = []

            for page in paginator.paginate(Limit=min(limit, 1000)):
                for key in page["Keys"]:
                    key_ids.append(key["KeyId"])
                    if len(key_ids) >= limit:
                        break
                if len(key_ids) >= limit:
                    break

            return key_ids

        except ClientError as e:
            raise KMSProviderError(f"Failed to list keys: {e}")