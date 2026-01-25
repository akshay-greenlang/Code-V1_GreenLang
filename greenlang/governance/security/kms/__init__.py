"""
KMS (Key Management Service) Integration for GreenLang
======================================================

Provides secure key management and signing operations using cloud KMS providers:
- AWS KMS
- Azure Key Vault
- Google Cloud KMS

Features:
- Automatic provider detection from environment
- Key caching with TTL to reduce API calls
- Async signing support
- Batch signing operations
- Comprehensive error handling
- Key rotation support
"""

from .base_kms import (
    BaseKMSProvider,
    KMSConfig,
    KMSSignResult,
    KMSKeyInfo,
    KMSProviderError,
    KMSKeyNotFoundError,
    KMSSigningError,
)
from .aws_kms import AWSKMSProvider
from .azure_kms import AzureKeyVaultProvider
from .gcp_kms import GCPCloudKMSProvider
from .factory import create_kms_provider, detect_kms_provider

__all__ = [
    # Base classes
    "BaseKMSProvider",
    "KMSConfig",
    "KMSSignResult",
    "KMSKeyInfo",
    # Providers
    "AWSKMSProvider",
    "AzureKeyVaultProvider",
    "GCPCloudKMSProvider",
    # Factory
    "create_kms_provider",
    "detect_kms_provider",
    # Exceptions
    "KMSProviderError",
    "KMSKeyNotFoundError",
    "KMSSigningError",
]

__version__ = "1.0.0"