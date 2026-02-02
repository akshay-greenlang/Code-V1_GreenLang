# -*- coding: utf-8 -*-
"""
Ephemeral Key Generation for Testing
====================================

Generates ephemeral keys in memory for testing signing/verification.
Keys are never persisted to disk.
"""

from typing import Tuple

def generate_ephemeral_keypair() -> Tuple[bytes, bytes]:
    """
    Generate an ephemeral Ed25519 keypair for testing

    Returns:
        Tuple of (private_key_pem, public_key_pem)
    """
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives import serialization

        # Generate ephemeral Ed25519 keypair
        priv = ed25519.Ed25519PrivateKey.generate()
        pub = priv.public_key()

        # Serialize to PEM format
        priv_pem = priv.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

        pub_pem = pub.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return priv_pem, pub_pem

    except ImportError:
        # Fallback to mock keys if cryptography not available
        # These are clearly marked as test-only
        import base64
        mock_priv = b"-----BEGIN PRIVATE KEY-----\nTEST_ONLY_NOT_REAL\n-----END PRIVATE KEY-----"
        mock_pub = b"-----BEGIN PUBLIC KEY-----\nTEST_ONLY_NOT_REAL\n-----END PUBLIC KEY-----"
        return mock_priv, mock_pub