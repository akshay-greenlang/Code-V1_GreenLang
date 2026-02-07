# -*- coding: utf-8 -*-
"""
JWKS Endpoint - JWT Authentication Service (SEC-001)

Provides the current RSA public key in standard JSON Web Key Set (JWKS)
format so that external services can validate GreenLang-issued JWTs
without sharing the private key.

The JWKS response follows RFC 7517 and includes the ``kid`` (key ID),
``kty`` (key type), ``use`` (signature), ``alg`` (RS256), and the RSA
public key parameters ``n`` (modulus) and ``e`` (exponent) encoded as
Base64url.

Classes:
    - JWKSProvider: Builds and caches the JWKS response.

Example:
    >>> provider = JWKSProvider(public_key_pem=pem_bytes)
    >>> jwks = provider.get_jwks()
    >>> jwks["keys"][0]["kty"]
    'RSA'

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base64url helpers (no padding, URL-safe)
# ---------------------------------------------------------------------------


def _base64url_encode(data: bytes) -> str:
    """Encode bytes to Base64url without padding.

    Args:
        data: Raw bytes to encode.

    Returns:
        Base64url-encoded string (no ``=`` padding).
    """
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _int_to_base64url(value: int) -> str:
    """Encode a positive integer to Base64url (big-endian, unsigned).

    Args:
        value: Non-negative integer.

    Returns:
        Base64url-encoded string of the integer's big-endian byte
        representation.
    """
    byte_length = (value.bit_length() + 7) // 8
    raw = value.to_bytes(byte_length, byteorder="big")
    return _base64url_encode(raw)


# ---------------------------------------------------------------------------
# JWKSProvider
# ---------------------------------------------------------------------------


class JWKSProvider:
    """Manages JWKS (JSON Web Key Set) for public key distribution.

    Converts an RSA public key (PEM format) into a standard JWKS JSON
    structure that can be served at ``/.well-known/jwks.json``.

    The JWKS is computed once and cached for the lifetime of the provider
    instance.  When key rotation occurs, a new ``JWKSProvider`` should
    be created or :meth:`refresh` called.

    Attributes:
        kid: Key ID included in the JWKS ``kid`` field.

    Example:
        >>> pem = open("public.pem", "rb").read()
        >>> provider = JWKSProvider(public_key_pem=pem, key_id="v1")
        >>> print(provider.get_jwks())
    """

    def __init__(
        self,
        public_key_pem: Optional[bytes] = None,
        jwt_handler: Any = None,
        key_id: str = "greenlang-rsa-1",
    ) -> None:
        """Initialize the JWKS provider.

        Exactly one of *public_key_pem* or *jwt_handler* must be provided.
        If *jwt_handler* is given, its ``public_key_pem`` attribute is read.

        Args:
            public_key_pem: RSA public key in PEM-encoded bytes.
            jwt_handler: An object with a ``public_key_pem`` attribute.
            key_id: Key ID (``kid``) for the JWK entry.

        Raises:
            ValueError: If neither *public_key_pem* nor *jwt_handler* is given.
        """
        self.kid = key_id
        self._cached_jwks: Optional[Dict[str, Any]] = None

        if public_key_pem is not None:
            self._public_key_pem = public_key_pem
        elif jwt_handler is not None:
            pem = getattr(jwt_handler, "public_key_pem", None)
            if pem is None:
                raise ValueError(
                    "jwt_handler must have a 'public_key_pem' attribute"
                )
            self._public_key_pem = (
                pem if isinstance(pem, bytes) else pem.encode("utf-8")
            )
        else:
            raise ValueError(
                "Either 'public_key_pem' or 'jwt_handler' must be provided"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_jwks(self) -> Dict[str, Any]:
        """Return the JWKS JSON containing the current public key.

        The result follows RFC 7517 and is suitable for serving at
        ``/.well-known/jwks.json``.

        Returns:
            A dict of the form ``{"keys": [<JWK>]}``.
        """
        if self._cached_jwks is not None:
            return self._cached_jwks

        jwk = self._rsa_public_key_to_jwk(self._public_key_pem, self.kid)
        self._cached_jwks = {"keys": [jwk]}

        logger.info("JWKS built for kid=%s", self.kid)
        return self._cached_jwks

    def refresh(self, public_key_pem: Optional[bytes] = None) -> None:
        """Invalidate the cached JWKS, optionally replacing the public key.

        Call this after key rotation so that subsequent :meth:`get_jwks`
        calls return the new key.

        Args:
            public_key_pem: New PEM-encoded public key, or ``None`` to
                keep the existing key but regenerate the JWK.
        """
        if public_key_pem is not None:
            self._public_key_pem = public_key_pem

        self._cached_jwks = None
        logger.info("JWKS cache invalidated for kid=%s", self.kid)

    # ------------------------------------------------------------------
    # Internal: PEM -> JWK conversion
    # ------------------------------------------------------------------

    def _rsa_public_key_to_jwk(
        self,
        public_key_pem: bytes,
        kid: str,
    ) -> Dict[str, Any]:
        """Convert an RSA PEM public key to a JWK dictionary.

        Uses ``cryptography`` to parse the PEM and extract the modulus
        (``n``) and exponent (``e``), then encodes them as Base64url per
        RFC 7518 Section 6.3.

        Args:
            public_key_pem: PEM-encoded RSA public key.
            kid: Key ID for the ``kid`` field.

        Returns:
            JWK dictionary with fields ``kty``, ``use``, ``alg``, ``kid``,
            ``n``, and ``e``.

        Raises:
            ImportError: If ``cryptography`` is not installed.
            ValueError: If the PEM cannot be parsed as an RSA public key.
        """
        try:
            from cryptography.hazmat.primitives.serialization import (
                load_pem_public_key,
            )
            from cryptography.hazmat.primitives.asymmetric.rsa import (
                RSAPublicNumbers,
            )
        except ImportError as exc:
            raise ImportError(
                "The 'cryptography' package is required for JWKS generation. "
                "Install it with: pip install cryptography"
            ) from exc

        if isinstance(public_key_pem, str):
            public_key_pem = public_key_pem.encode("utf-8")

        public_key = load_pem_public_key(public_key_pem)
        public_numbers = public_key.public_numbers()  # type: ignore[union-attr]

        if not isinstance(public_numbers, RSAPublicNumbers):
            raise ValueError("Provided key is not an RSA public key")

        n_b64 = _int_to_base64url(public_numbers.n)
        e_b64 = _int_to_base64url(public_numbers.e)

        jwk: Dict[str, str] = {
            "kty": "RSA",
            "use": "sig",
            "alg": "RS256",
            "kid": kid,
            "n": n_b64,
            "e": e_b64,
        }

        logger.debug(
            "Converted RSA public key to JWK: kid=%s n_len=%d",
            kid,
            len(n_b64),
        )
        return jwk

    # ------------------------------------------------------------------
    # Utility: get individual key by kid
    # ------------------------------------------------------------------

    def get_key_by_kid(self, kid: str) -> Optional[Dict[str, Any]]:
        """Look up a specific key in the JWKS by its ``kid``.

        Args:
            kid: The Key ID to search for.

        Returns:
            The matching JWK dictionary, or ``None`` if not found.
        """
        jwks = self.get_jwks()
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return key
        return None
