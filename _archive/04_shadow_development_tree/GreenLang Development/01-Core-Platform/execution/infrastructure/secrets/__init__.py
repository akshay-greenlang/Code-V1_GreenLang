"""
GreenLang Secrets Management Infrastructure
============================================

HashiCorp Vault integration for secure secrets management.

This module provides the secrets management infrastructure for the GreenLang
Process Heat platform, including Vault client operations and automatic
credential rotation.

Components:
- VaultClient: High-level Vault client with retry logic and caching
- SecretsRotationManager: Automatic credential and certificate rotation
- VaultConfig: Configuration for Vault connectivity

Example:
    >>> from greenlang.infrastructure.secrets import VaultClient
    >>> client = VaultClient()
    >>> secret = await client.get_secret("process-heat/database")
"""

from greenlang.infrastructure.secrets.vault_client import (
    VaultClient,
    VaultConfig,
    VaultAuthMethod,
    VaultSecret,
    DatabaseCredentials,
    AWSCredentials,
    Certificate,
)
from greenlang.infrastructure.secrets.secrets_rotation import (
    SecretsRotationManager,
    RotationConfig,
    RotationResult,
    RotationType,
)

__all__ = [
    # Vault Client
    'VaultClient',
    'VaultConfig',
    'VaultAuthMethod',
    'VaultSecret',
    'DatabaseCredentials',
    'AWSCredentials',
    'Certificate',
    # Rotation
    'SecretsRotationManager',
    'RotationConfig',
    'RotationResult',
    'RotationType',
]

__version__ = '1.0.0'
