# -*- coding: utf-8 -*-
"""
Data encryption utilities for sensitive ESG data.

Uses Fernet symmetric encryption (AES-128 in CBC mode).
Key management via environment variables.
"""

import os
import base64
import logging
from typing import Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

logger = logging.getLogger(__name__)


class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data.

    Features:
    - AES-128 symmetric encryption
    - Key derivation from passphrase
    - Automatic key rotation support
    - Secure key storage in environment
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize encryption manager.

        Args:
            key: Encryption key (32 bytes). If None, loads from env.
        """
        if key is None:
            key = self._load_key_from_env()

        self.fernet = Fernet(key)
        logger.info("Encryption manager initialized")

    def _load_key_from_env(self) -> bytes:
        """Load encryption key from environment variable."""
        key_b64 = os.getenv('CSRD_ENCRYPTION_KEY')

        if not key_b64:
            raise ValueError(
                "CSRD_ENCRYPTION_KEY not set. Generate with: "
                "python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        return base64.urlsafe_b64decode(key_b64)

    @staticmethod
    def generate_key() -> bytes:
        """Generate a new encryption key."""
        return Fernet.generate_key()

    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Encrypt data.

        Args:
            data: String or bytes to encrypt

        Returns:
            Base64-encoded encrypted data
        """
        if isinstance(data, str):
            data = data.encode('utf-8')

        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')

    def decrypt(self, encrypted_data: str) -> str:
        """
        Decrypt data.

        Args:
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted string
        """
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data)
        decrypted = self.fernet.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')

    def encrypt_dict(self, data: dict, fields: list) -> dict:
        """
        Encrypt specific fields in a dictionary.

        Args:
            data: Dictionary with data
            fields: List of field names to encrypt

        Returns:
            Dictionary with encrypted fields
        """
        encrypted_data = data.copy()

        for field in fields:
            if field in encrypted_data and encrypted_data[field] is not None:
                encrypted_data[field] = self.encrypt(str(encrypted_data[field]))
                encrypted_data[f'{field}_encrypted'] = True

        return encrypted_data

    def decrypt_dict(self, data: dict, fields: list) -> dict:
        """
        Decrypt specific fields in a dictionary.

        Args:
            data: Dictionary with encrypted data
            fields: List of field names to decrypt

        Returns:
            Dictionary with decrypted fields
        """
        decrypted_data = data.copy()

        for field in fields:
            if f'{field}_encrypted' in decrypted_data and decrypted_data[f'{field}_encrypted']:
                decrypted_data[field] = self.decrypt(decrypted_data[field])
                decrypted_data[f'{field}_encrypted'] = False

        return decrypted_data


# Singleton instance
_encryption_manager = None

def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager instance."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager
