"""
KMS Provider Factory
====================

Automatic detection and creation of KMS providers based on environment.
"""

import os
import logging
from typing import Optional, Type
from urllib.parse import urlparse

from .base_kms import BaseKMSProvider, KMSConfig, KMSProviderError
from .aws_kms import AWSKMSProvider
from .azure_kms import AzureKeyVaultProvider
from .gcp_kms import GCPCloudKMSProvider

logger = logging.getLogger(__name__)


def detect_kms_provider() -> Optional[str]:
    """
    Automatically detect KMS provider from environment

    Returns:
        Provider name ('aws', 'azure', 'gcp') or None
    """
    # Check for AWS
    if any([
        os.environ.get("AWS_ACCESS_KEY_ID"),
        os.environ.get("AWS_PROFILE"),
        os.environ.get("AWS_ROLE_ARN"),
        os.path.exists(os.path.expanduser("~/.aws/credentials")),
    ]):
        logger.info("Detected AWS environment")
        return "aws"

    # Check for Azure
    if any([
        os.environ.get("AZURE_CLIENT_ID"),
        os.environ.get("AZURE_TENANT_ID"),
        os.environ.get("MSI_ENDPOINT"),  # Managed Identity
        os.environ.get("IDENTITY_ENDPOINT"),  # Managed Identity
    ]):
        logger.info("Detected Azure environment")
        return "azure"

    # Check for GCP
    if any([
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        os.environ.get("GOOGLE_CLOUD_PROJECT"),
        os.environ.get("GCP_PROJECT"),
        os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json")),
    ]):
        logger.info("Detected GCP environment")
        return "gcp"

    # Check if running in specific cloud environments
    if os.path.exists("/var/run/secrets/kubernetes.io"):
        # Running in Kubernetes, check for cloud-specific annotations
        if os.environ.get("AWS_REGION"):
            return "aws"
        elif os.environ.get("AZURE_RESOURCE_GROUP"):
            return "azure"
        elif os.environ.get("GCP_PROJECT_ID"):
            return "gcp"

    logger.warning("Could not detect KMS provider from environment")
    return None


def detect_provider_from_key_id(key_id: str) -> Optional[str]:
    """
    Detect provider from key ID format

    Args:
        key_id: Key identifier

    Returns:
        Provider name or None
    """
    if not key_id:
        return None

    # AWS ARN format
    if key_id.startswith("arn:aws:kms:"):
        return "aws"

    # Azure Key Vault URL
    if ".vault.azure.net" in key_id or key_id.startswith("https://") and "vault" in key_id:
        return "azure"

    # GCP resource name
    if key_id.startswith("projects/") and "/keyRings/" in key_id:
        return "gcp"

    # Check for provider hints in environment
    if os.environ.get("GL_KMS_PROVIDER"):
        return os.environ.get("GL_KMS_PROVIDER").lower()

    return None


def create_kms_provider(config: Optional[KMSConfig] = None,
                       provider: Optional[str] = None) -> BaseKMSProvider:
    """
    Create a KMS provider instance

    Args:
        config: KMS configuration (created from environment if not provided)
        provider: Provider name ('aws', 'azure', 'gcp') or None for auto-detect

    Returns:
        KMS provider instance

    Raises:
        KMSProviderError: If provider cannot be determined or created
    """
    # Create config from environment if not provided
    if config is None:
        config = KMSConfig.from_env() if hasattr(KMSConfig, 'from_env') else create_config_from_env()

    # Determine provider
    if provider:
        provider = provider.lower()
    else:
        # Try to detect from key ID first
        provider = detect_provider_from_key_id(config.key_id)

        # Fall back to environment detection
        if not provider:
            provider = detect_kms_provider()

        # Finally check explicit config
        if not provider and config.provider:
            provider = config.provider.lower()

    if not provider:
        raise KMSProviderError(
            "Could not determine KMS provider. Set GL_KMS_PROVIDER environment variable "
            "or specify provider in configuration."
        )

    # Update config with detected provider
    config.provider = provider

    # Create provider instance
    provider_map: dict[str, Type[BaseKMSProvider]] = {
        "aws": AWSKMSProvider,
        "azure": AzureKeyVaultProvider,
        "gcp": GCPCloudKMSProvider,
    }

    provider_class = provider_map.get(provider)
    if not provider_class:
        raise KMSProviderError(f"Unknown KMS provider: {provider}")

    try:
        logger.info(f"Creating {provider.upper()} KMS provider")
        return provider_class(config)
    except Exception as e:
        raise KMSProviderError(f"Failed to create {provider} KMS provider: {e}")


def create_config_from_env() -> KMSConfig:
    """
    Create KMS configuration from environment variables

    Environment variables:
    - GL_KMS_PROVIDER: Provider name (aws, azure, gcp)
    - GL_KMS_KEY_ID: Key identifier
    - GL_KMS_REGION: Region for AWS
    - GL_KMS_ENDPOINT: Custom endpoint URL
    - GL_KMS_CACHE_TTL: Cache TTL in seconds (default: 300)
    - GL_KMS_MAX_RETRIES: Maximum retries (default: 3)
    - GL_KMS_TIMEOUT: Timeout in seconds (default: 30)
    - GL_KMS_ASYNC_ENABLED: Enable async support (default: true)

    AWS-specific:
    - AWS_PROFILE: AWS profile to use
    - AWS_REGION: AWS region

    Azure-specific:
    - AZURE_TENANT_ID: Azure AD tenant ID
    - AZURE_KEY_VAULT_URL: Key Vault URL
    - AZURE_CLIENT_ID: Service principal client ID
    - AZURE_CLIENT_SECRET: Service principal secret

    GCP-specific:
    - GCP_PROJECT_ID: GCP project ID
    - GCP_LOCATION: GCP location (default: global)
    - GCP_KEYRING: GCP keyring name
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account JSON
    """
    # Basic configuration
    config = KMSConfig(
        provider=os.environ.get("GL_KMS_PROVIDER", ""),
        key_id=os.environ.get("GL_KMS_KEY_ID", ""),
        region=os.environ.get("GL_KMS_REGION", os.environ.get("AWS_REGION")),
        endpoint_url=os.environ.get("GL_KMS_ENDPOINT"),
        cache_ttl_seconds=int(os.environ.get("GL_KMS_CACHE_TTL", "300")),
        max_retries=int(os.environ.get("GL_KMS_MAX_RETRIES", "3")),
        timeout_seconds=int(os.environ.get("GL_KMS_TIMEOUT", "30")),
        async_enabled=os.environ.get("GL_KMS_ASYNC_ENABLED", "true").lower() in ("true", "1", "yes"),
    )

    # AWS-specific
    config.aws_profile = os.environ.get("AWS_PROFILE")

    # Azure-specific
    config.azure_tenant_id = os.environ.get("AZURE_TENANT_ID")
    config.azure_vault_url = os.environ.get("AZURE_KEY_VAULT_URL")

    # GCP-specific
    config.gcp_project_id = os.environ.get("GCP_PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT"))
    config.gcp_location_id = os.environ.get("GCP_LOCATION", "global")
    config.gcp_keyring_id = os.environ.get("GCP_KEYRING")

    return config


def list_available_providers() -> list[str]:
    """
    List available KMS providers based on installed dependencies

    Returns:
        List of available provider names
    """
    available = []

    # Check AWS
    try:
        import boto3
        available.append("aws")
    except ImportError:
        pass

    # Check Azure
    try:
        from azure.keyvault.keys import KeyClient
        available.append("azure")
    except ImportError:
        pass

    # Check GCP
    try:
        from google.cloud import kms
        available.append("gcp")
    except ImportError:
        pass

    return available


def get_provider_requirements(provider: str) -> list[str]:
    """
    Get pip requirements for a specific provider

    Args:
        provider: Provider name

    Returns:
        List of pip package names
    """
    requirements = {
        "aws": ["boto3", "botocore", "aioboto3"],  # aioboto3 optional for async
        "azure": [
            "azure-keyvault-keys",
            "azure-identity",
            "azure-core",
        ],
        "gcp": [
            "google-cloud-kms",
            "crc32c",  # For data integrity
        ],
    }

    return requirements.get(provider.lower(), [])