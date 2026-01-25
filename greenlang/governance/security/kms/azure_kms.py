"""
Azure Key Vault Provider Implementation
========================================

Integrates with Azure Key Vault for secure key operations.
"""

import base64
import hashlib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse

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

# Try to import Azure SDK
try:
    from azure.identity import DefaultAzureCredential, ClientSecretCredential
    from azure.keyvault.keys import KeyClient, KeyVaultKey
    from azure.keyvault.keys.crypto import CryptographyClient, SignatureAlgorithm as AzureSignAlgorithm
    from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("Azure SDK not available - Azure Key Vault support disabled")

# Try to import async Azure SDK
try:
    from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
    from azure.keyvault.keys.aio import KeyClient as AsyncKeyClient
    from azure.keyvault.keys.crypto.aio import CryptographyClient as AsyncCryptographyClient
    AZURE_ASYNC_AVAILABLE = True
except ImportError:
    AZURE_ASYNC_AVAILABLE = False
    logger.info("Azure async SDK not available - async Azure Key Vault support disabled")


class AzureKeyVaultProvider(BaseKMSProvider):
    """
    Azure Key Vault provider implementation

    Supports:
    - RSA keys (2048, 3072, 4096)
    - Elliptic Curve keys (P-256, P-384, P-521)
    - Key rotation and versioning
    - Hardware Security Module (HSM) backed keys
    """

    # Map our algorithms to Azure algorithms
    ALGORITHM_MAP = {
        SigningAlgorithm.RSASSA_PSS_SHA256: AzureSignAlgorithm.ps256 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.RSASSA_PSS_SHA384: AzureSignAlgorithm.ps384 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.RSASSA_PSS_SHA512: AzureSignAlgorithm.ps512 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA256: AzureSignAlgorithm.rs256 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA384: AzureSignAlgorithm.rs384 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.RSASSA_PKCS1_V1_5_SHA512: AzureSignAlgorithm.rs512 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.ECDSA_SHA256: AzureSignAlgorithm.es256 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.ECDSA_SHA384: AzureSignAlgorithm.es384 if AZURE_SDK_AVAILABLE else None,
        SigningAlgorithm.ECDSA_SHA512: AzureSignAlgorithm.es512 if AZURE_SDK_AVAILABLE else None,
    }

    # Map Azure key types to our key algorithms
    KEY_TYPE_MAP = {
        "RSA": {
            2048: KeyAlgorithm.RSA_2048,
            3072: KeyAlgorithm.RSA_3072,
            4096: KeyAlgorithm.RSA_4096,
        },
        "EC": {
            "P-256": KeyAlgorithm.ECDSA_P256,
            "P-384": KeyAlgorithm.ECDSA_P384,
            "P-521": KeyAlgorithm.ECDSA_P521,
        }
    }

    def __init__(self, config: KMSConfig):
        """Initialize Azure Key Vault provider"""
        if not AZURE_SDK_AVAILABLE:
            raise KMSProviderError(
                "Azure SDK is required for Azure Key Vault: pip install azure-keyvault-keys azure-identity"
            )

        super().__init__(config)

        # Parse vault URL from config or key ID
        self.vault_url = self._parse_vault_url(config)
        if not self.vault_url:
            raise ValueError("Azure Key Vault URL required (in azure_vault_url or key_id)")

        # Parse key name from key ID
        self.key_name = self._parse_key_name(config.key_id)
        if not self.key_name:
            raise ValueError("Key name required in key_id")

        # Set async availability
        self.config.async_enabled = self.config.async_enabled and AZURE_ASYNC_AVAILABLE

        # Store credential for reuse
        self._credential = None
        self._async_credential = None

    def _parse_vault_url(self, config: KMSConfig) -> Optional[str]:
        """Parse vault URL from config or key ID"""
        if config.azure_vault_url:
            return config.azure_vault_url

        # Try to parse from key_id if it's a full URL
        if config.key_id and config.key_id.startswith("https://"):
            parsed = urlparse(config.key_id)
            return f"{parsed.scheme}://{parsed.netloc}"

        return None

    def _parse_key_name(self, key_id: str) -> Optional[str]:
        """Parse key name from key ID"""
        if not key_id:
            return None

        # If it's a URL, extract the key name
        if key_id.startswith("https://"):
            parsed = urlparse(key_id)
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) >= 2 and path_parts[0] == "keys":
                return path_parts[1]
            return None

        # Otherwise assume it's just the key name
        return key_id

    def _create_credential(self):
        """Create Azure credential"""
        if self._credential:
            return self._credential

        # Use explicit credentials if provided
        if self.config.azure_tenant_id and hasattr(self.config, 'azure_client_id'):
            self._credential = ClientSecretCredential(
                tenant_id=self.config.azure_tenant_id,
                client_id=getattr(self.config, 'azure_client_id'),
                client_secret=getattr(self.config, 'azure_client_secret')
            )
        else:
            # Use default credential chain
            self._credential = DefaultAzureCredential()

        return self._credential

    def _create_client(self) -> Any:
        """Create Azure Key Vault client"""
        try:
            credential = self._create_credential()
            client = KeyClient(vault_url=self.vault_url, credential=credential)

            # Test connectivity
            try:
                key = client.get_key(self.key_name)
                logger.info(f"Connected to Azure Key Vault: {self.vault_url}")
            except ResourceNotFoundError:
                raise KMSKeyNotFoundError(f"Key {self.key_name} not found in vault")
            except HttpResponseError as e:
                raise KMSProviderError(f"Failed to connect to Azure Key Vault: {e}")

            return client

        except Exception as e:
            raise KMSProviderError(f"Failed to create Azure Key Vault client: {e}")

    def _create_async_client(self) -> Any:
        """Create async Azure Key Vault client"""
        if not AZURE_ASYNC_AVAILABLE:
            return None

        try:
            if not self._async_credential:
                # Use explicit credentials if provided
                if self.config.azure_tenant_id and hasattr(self.config, 'azure_client_id'):
                    self._async_credential = AsyncDefaultAzureCredential()
                else:
                    self._async_credential = AsyncDefaultAzureCredential()

            return AsyncKeyClient(vault_url=self.vault_url, credential=self._async_credential)

        except Exception as e:
            logger.warning(f"Failed to create async Azure Key Vault client: {e}")
            return None

    def _get_crypto_client(self, key: KeyVaultKey) -> Any:
        """Get cryptography client for a key"""
        credential = self._create_credential()
        return CryptographyClient(key, credential=credential)

    def get_key_info(self, key_id: Optional[str] = None) -> KMSKeyInfo:
        """Get information about an Azure Key Vault key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        # Check cache
        cached = self.cache.get(key_name)
        if cached:
            return cached

        try:
            # Get the key
            key = self.retry_with_backoff(
                self.client.get_key,
                key_name
            )

            # Determine algorithm from key type
            if key.key_type == "RSA":
                size = key.properties.key_size or 2048
                algorithm = self.KEY_TYPE_MAP["RSA"].get(size, KeyAlgorithm.RSA_2048)
            elif key.key_type == "EC":
                curve = key.properties.curve_name
                algorithm = self.KEY_TYPE_MAP["EC"].get(curve, KeyAlgorithm.ECDSA_P256)
            else:
                algorithm = KeyAlgorithm.RSA_2048  # Default

            # Get public key if available
            public_key = None
            if hasattr(key.key, 'n') and hasattr(key.key, 'e'):  # RSA
                # Construct PEM from modulus and exponent
                # This is simplified - actual implementation would need proper ASN.1 encoding
                pass
            elif hasattr(key.key, 'x') and hasattr(key.key, 'y'):  # EC
                # Construct PEM from curve points
                pass

            key_info = KMSKeyInfo(
                key_id=key.name,
                key_arn=key.id,  # Full key identifier
                algorithm=algorithm,
                created_at=key.properties.created_on or datetime.utcnow(),
                enabled=key.properties.enabled,
                rotation_enabled=False,  # Azure doesn't have automatic rotation like AWS
                key_version=key.properties.version,
                public_key=public_key,
                metadata={
                    "vault_url": self.vault_url,
                    "key_type": key.key_type,
                    "key_operations": key.key_operations,
                    "hsm_protected": key.properties.hsm_protected,
                    "recoverable": key.properties.recoverable_days is not None,
                    "tags": key.properties.tags,
                }
            )

            # Cache the result
            self.cache.set(key_name, key_info)

            return key_info

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key {key_name} not found in Azure Key Vault")
        except Exception as e:
            raise KMSProviderError(f"Failed to get key info: {e}")

    def sign(self, data: bytes, key_id: Optional[str] = None,
             algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Sign data using Azure Key Vault key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        # Get the key
        try:
            key = self.client.get_key(key_name)
        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key {key_name} not found")
        except Exception as e:
            raise KMSSigningError(f"Failed to get key: {e}")

        # Determine signing algorithm
        if algorithm:
            azure_algorithm = self.ALGORITHM_MAP.get(algorithm)
            if not azure_algorithm:
                raise KMSSigningError(f"Unsupported algorithm for Azure: {algorithm}")
        else:
            # Use default based on key type
            if key.key_type == "RSA":
                azure_algorithm = AzureSignAlgorithm.rs256
            elif key.key_type == "EC":
                if "P-384" in str(key.properties.curve_name):
                    azure_algorithm = AzureSignAlgorithm.es384
                elif "P-521" in str(key.properties.curve_name):
                    azure_algorithm = AzureSignAlgorithm.es512
                else:
                    azure_algorithm = AzureSignAlgorithm.es256
            else:
                azure_algorithm = AzureSignAlgorithm.rs256

        try:
            # Get crypto client for the key
            crypto_client = self._get_crypto_client(key)

            # Hash the data based on algorithm
            if azure_algorithm in [AzureSignAlgorithm.es256, AzureSignAlgorithm.rs256,
                                  AzureSignAlgorithm.ps256]:
                digest = hashlib.sha256(data).digest()
            elif azure_algorithm in [AzureSignAlgorithm.es384, AzureSignAlgorithm.rs384,
                                    AzureSignAlgorithm.ps384]:
                digest = hashlib.sha384(data).digest()
            elif azure_algorithm in [AzureSignAlgorithm.es512, AzureSignAlgorithm.rs512,
                                    AzureSignAlgorithm.ps512]:
                digest = hashlib.sha512(data).digest()
            else:
                digest = data

            # Sign the digest
            result = self.retry_with_backoff(
                crypto_client.sign,
                algorithm=azure_algorithm,
                digest=digest
            )

            return KMSSignResult(
                signature=result.signature,
                key_id=key.name,
                algorithm=str(result.algorithm),
                timestamp=datetime.utcnow().isoformat(),
                key_version=key.properties.version,
                provider="azure"
            )

        except Exception as e:
            raise KMSSigningError(f"Azure Key Vault signing failed: {e}")

    def verify(self, data: bytes, signature: bytes,
               key_id: Optional[str] = None,
               algorithm: Optional[SigningAlgorithm] = None) -> bool:
        """Verify signature using Azure Key Vault key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        # Get the key
        try:
            key = self.client.get_key(key_name)
        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key {key_name} not found")
        except Exception as e:
            raise KMSSigningError(f"Failed to get key: {e}")

        # Determine signing algorithm
        if algorithm:
            azure_algorithm = self.ALGORITHM_MAP.get(algorithm)
            if not azure_algorithm:
                raise KMSSigningError(f"Unsupported algorithm for Azure: {algorithm}")
        else:
            # Use default based on key type
            if key.key_type == "RSA":
                azure_algorithm = AzureSignAlgorithm.rs256
            else:
                azure_algorithm = AzureSignAlgorithm.es256

        try:
            # Get crypto client for the key
            crypto_client = self._get_crypto_client(key)

            # Hash the data based on algorithm
            if azure_algorithm in [AzureSignAlgorithm.es256, AzureSignAlgorithm.rs256,
                                  AzureSignAlgorithm.ps256]:
                digest = hashlib.sha256(data).digest()
            elif azure_algorithm in [AzureSignAlgorithm.es384, AzureSignAlgorithm.rs384,
                                    AzureSignAlgorithm.ps384]:
                digest = hashlib.sha384(data).digest()
            elif azure_algorithm in [AzureSignAlgorithm.es512, AzureSignAlgorithm.rs512,
                                    AzureSignAlgorithm.ps512]:
                digest = hashlib.sha512(data).digest()
            else:
                digest = data

            # Verify the signature
            result = self.retry_with_backoff(
                crypto_client.verify,
                algorithm=azure_algorithm,
                digest=digest,
                signature=signature
            )

            return result.is_valid

        except Exception as e:
            # Invalid signature might raise an exception
            logger.debug(f"Signature verification failed: {e}")
            return False

    def rotate_key(self, key_id: Optional[str] = None) -> str:
        """Create a new version of an Azure Key Vault key"""
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        try:
            # Get current key properties
            current_key = self.client.get_key(key_name)

            # Create new key version
            new_key = self.retry_with_backoff(
                self.client.create_key,
                name=key_name,
                key_type=current_key.key_type,
                key_operations=current_key.key_operations,
                enabled=True,
                tags={"rotated": "true", "previous_version": current_key.properties.version}
            )

            # Invalidate cache for this key
            self.cache.invalidate(key_name)

            logger.info(f"Created new key version for {key_name}: {new_key.properties.version}")
            return new_key.properties.version

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key {key_name} not found")
        except Exception as e:
            raise KMSKeyRotationError(f"Azure Key Vault rotation failed: {e}")

    async def sign_async(self, data: bytes, key_id: Optional[str] = None,
                        algorithm: Optional[SigningAlgorithm] = None) -> KMSSignResult:
        """Async signing using Azure Key Vault"""
        if not self.config.async_enabled or not self.async_client:
            # Fall back to sync
            return self.sign(data, key_id, algorithm)

        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        try:
            async with self.async_client as client:
                # Get the key
                key = await client.get_key(key_name)

                # Create async crypto client
                async_crypto = AsyncCryptographyClient(
                    key,
                    credential=self._async_credential
                )

                # Determine algorithm
                if algorithm:
                    azure_algorithm = self.ALGORITHM_MAP.get(algorithm)
                else:
                    azure_algorithm = AzureSignAlgorithm.rs256

                # Hash the data
                if azure_algorithm in [AzureSignAlgorithm.es256, AzureSignAlgorithm.rs256,
                                      AzureSignAlgorithm.ps256]:
                    digest = hashlib.sha256(data).digest()
                elif azure_algorithm in [AzureSignAlgorithm.es384, AzureSignAlgorithm.rs384,
                                        AzureSignAlgorithm.ps384]:
                    digest = hashlib.sha384(data).digest()
                else:
                    digest = hashlib.sha512(data).digest()

                # Sign
                result = await async_crypto.sign(
                    algorithm=azure_algorithm,
                    digest=digest
                )

                return KMSSignResult(
                    signature=result.signature,
                    key_id=key.name,
                    algorithm=str(result.algorithm),
                    timestamp=datetime.utcnow().isoformat(),
                    key_version=key.properties.version,
                    provider="azure"
                )

        except Exception as e:
            raise KMSSigningError(f"Async signing failed: {e}")

    def list_keys(self, limit: int = 100) -> List[str]:
        """
        List available keys in the vault

        Args:
            limit: Maximum number of keys to return

        Returns:
            List of key names
        """
        try:
            key_names = []
            count = 0

            for key_properties in self.client.list_properties_of_keys():
                if not key_properties.enabled:
                    continue  # Skip disabled keys

                key_names.append(key_properties.name)
                count += 1

                if count >= limit:
                    break

            return key_names

        except Exception as e:
            raise KMSProviderError(f"Failed to list keys: {e}")

    def backup_key(self, key_id: Optional[str] = None) -> bytes:
        """
        Backup a key from Azure Key Vault

        Args:
            key_id: Key identifier

        Returns:
            Backup blob
        """
        key_name = self._parse_key_name(key_id) if key_id else self.key_name

        try:
            backup = self.retry_with_backoff(
                self.client.backup_key,
                key_name
            )
            return backup

        except ResourceNotFoundError:
            raise KMSKeyNotFoundError(f"Key {key_name} not found")
        except Exception as e:
            raise KMSProviderError(f"Failed to backup key: {e}")

    def restore_key(self, backup: bytes) -> str:
        """
        Restore a key from backup

        Args:
            backup: Backup blob

        Returns:
            Restored key name
        """
        try:
            restored_key = self.retry_with_backoff(
                self.client.restore_key_backup,
                backup
            )
            return restored_key.name

        except Exception as e:
            raise KMSProviderError(f"Failed to restore key: {e}")