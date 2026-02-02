# -*- coding: utf-8 -*-
"""
Key Provider Interface for GreenLang
====================================

Provides a clean abstraction for key management without embedded secrets.
All keys are loaded from environment variables or generated ephemerally.
"""

from dataclasses import dataclass
from typing import Protocol


class KeyProvider(Protocol):
    """Protocol for key providers"""

    def get_private_key_pem(self) -> bytes: ...
    def get_public_key_pem(self) -> bytes: ...


@dataclass
class EnvKeyProvider:
    """Key provider that loads keys from environment variables"""

    priv_var: str = "GL_SIGNING_PRIVATE_KEY_PEM"
    pub_var: str = "GL_SIGNING_PUBLIC_KEY_PEM"

    def get_private_key_pem(self) -> bytes:
        """Get private key from environment"""
        import os

        val = os.getenv(self.priv_var)
        if not val:
            raise RuntimeError(f"Missing signing private key env: {self.priv_var}")
        return val.encode()

    def get_public_key_pem(self) -> bytes:
        """Get public key from environment"""
        import os

        val = os.getenv(self.pub_var)
        if not val:
            raise RuntimeError(f"Missing signing public key env: {self.pub_var}")
        return val.encode()
