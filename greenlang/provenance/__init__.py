"""
Provenance and Supply Chain Security
====================================

Provides cryptographic verification, SBOM validation, and
supply chain attestations for GreenLang packs.
"""

from .signing import (
    SignatureVerifier,
    DevKeyVerifier,
    SigstoreVerifier,
    UnsignedPackError,
    create_verifier,
    verify_pack_signature,
    sign_pack
)

__all__ = [
    'SignatureVerifier',
    'DevKeyVerifier',
    'SigstoreVerifier',
    'UnsignedPackError',
    'create_verifier',
    'verify_pack_signature',
    'sign_pack',
]