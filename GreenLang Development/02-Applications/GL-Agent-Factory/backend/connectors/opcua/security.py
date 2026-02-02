"""
OPC-UA Security Module for GreenLang Process Heat Agents.

This module handles all security aspects of OPC-UA connections including:
- X.509 certificate generation and management
- Certificate validation and trust management
- Authentication token handling
- Secure channel establishment
- Cryptographic operations

Security best practices:
- Never hardcode credentials
- Use system certificate stores or secure vaults
- Validate server certificates
- Use strong security policies (Basic256Sha256 minimum)

Usage:
    from connectors.opcua.security import CertificateManager, SecurityManager

    # Generate client certificates
    cert_manager = CertificateManager(cert_dir="/path/to/certs")
    cert, key = await cert_manager.generate_client_certificate(
        common_name="GreenLang-Agent",
        organization="GreenLang"
    )

    # Configure security
    security = SecurityManager(cert_manager)
    await security.setup_security(
        policy=SecurityPolicy.BASIC256SHA256,
        mode=MessageSecurityMode.SIGN_AND_ENCRYPT
    )
"""

import asyncio
import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from .types import (
    SecurityPolicy,
    MessageSecurityMode,
    AuthenticationType,
    ConnectionConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Certificate Configuration
# =============================================================================


class CertificateType(str, Enum):
    """Types of X.509 certificates."""

    CLIENT = "client"
    SERVER = "server"
    CA = "ca"


class KeyType(str, Enum):
    """Cryptographic key types."""

    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    ECDSA_P256 = "ecdsa_p256"
    ECDSA_P384 = "ecdsa_p384"


@dataclass
class CertificateConfig:
    """Configuration for certificate generation."""

    common_name: str
    organization: str = "GreenLang"
    organizational_unit: str = "Process Heat Agents"
    country: str = "US"
    state: str = "California"
    locality: str = "San Francisco"
    validity_days: int = 365
    key_type: KeyType = KeyType.RSA_2048
    key_size: int = 2048
    application_uri: str = "urn:greenlang:opcua:client"
    dns_names: List[str] = field(default_factory=list)
    ip_addresses: List[str] = field(default_factory=list)


class CertificateInfo(BaseModel):
    """Information about a certificate."""

    serial_number: str = Field(..., description="Certificate serial number")
    subject: str = Field(..., description="Certificate subject")
    issuer: str = Field(..., description="Certificate issuer")
    not_before: datetime = Field(..., description="Valid from date")
    not_after: datetime = Field(..., description="Valid until date")
    fingerprint_sha1: str = Field(..., description="SHA-1 fingerprint")
    fingerprint_sha256: str = Field(..., description="SHA-256 fingerprint")
    application_uri: Optional[str] = Field(None, description="OPC-UA application URI")
    key_type: str = Field(..., description="Key type (RSA, ECDSA)")
    key_size: int = Field(..., description="Key size in bits")
    is_self_signed: bool = Field(False, description="Whether certificate is self-signed")
    is_expired: bool = Field(False, description="Whether certificate is expired")
    days_until_expiry: int = Field(..., description="Days until expiration")


# =============================================================================
# Certificate Manager
# =============================================================================


class CertificateManager:
    """
    Manages X.509 certificates for OPC-UA security.

    Handles certificate generation, loading, validation, and trust management.
    Uses cryptography library for X.509 operations.
    """

    def __init__(
        self,
        cert_dir: Union[str, Path] = "certs",
        trusted_dir: Optional[Union[str, Path]] = None,
        rejected_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize certificate manager.

        Args:
            cert_dir: Directory for storing certificates
            trusted_dir: Directory for trusted certificates
            rejected_dir: Directory for rejected certificates
        """
        self.cert_dir = Path(cert_dir)
        self.trusted_dir = Path(trusted_dir) if trusted_dir else self.cert_dir / "trusted"
        self.rejected_dir = Path(rejected_dir) if rejected_dir else self.cert_dir / "rejected"

        # Ensure directories exist
        self._ensure_directories()

        # Certificate cache
        self._cert_cache: Dict[str, bytes] = {}
        self._key_cache: Dict[str, bytes] = {}

    def _ensure_directories(self) -> None:
        """Create certificate directories if they don't exist."""
        for directory in [self.cert_dir, self.trusted_dir, self.rejected_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured certificate directory: {directory}")

    async def generate_client_certificate(
        self,
        config: CertificateConfig,
        output_name: str = "client",
    ) -> Tuple[bytes, bytes]:
        """
        Generate a self-signed client certificate and private key.

        Args:
            config: Certificate configuration
            output_name: Base name for output files

        Returns:
            Tuple of (certificate_pem, private_key_pem)
        """
        try:
            # Import cryptography library
            from cryptography import x509
            from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa, ec
            from cryptography.hazmat.backends import default_backend

            logger.info(f"Generating client certificate: {config.common_name}")

            # Generate private key
            if config.key_type in (KeyType.RSA_2048, KeyType.RSA_4096):
                key_size = 4096 if config.key_type == KeyType.RSA_4096 else 2048
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend(),
                )
            else:
                curve = ec.SECP384R1() if config.key_type == KeyType.ECDSA_P384 else ec.SECP256R1()
                private_key = ec.generate_private_key(curve, default_backend())

            # Build subject name
            subject = x509.Name([
                x509.NameAttribute(NameOID.COUNTRY_NAME, config.country),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, config.state),
                x509.NameAttribute(NameOID.LOCALITY_NAME, config.locality),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, config.organization),
                x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, config.organizational_unit),
                x509.NameAttribute(NameOID.COMMON_NAME, config.common_name),
            ])

            # Build certificate
            now = datetime.utcnow()
            builder = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(subject)  # Self-signed
                .public_key(private_key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now)
                .not_valid_after(now + timedelta(days=config.validity_days))
            )

            # Add extensions
            # Basic Constraints
            builder = builder.add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )

            # Key Usage
            builder = builder.add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=True,
                    data_encipherment=True,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )

            # Extended Key Usage (OPC-UA client)
            builder = builder.add_extension(
                x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                    ExtendedKeyUsageOID.SERVER_AUTH,
                ]),
                critical=False,
            )

            # Subject Alternative Name
            san_entries = [x509.UniformResourceIdentifier(config.application_uri)]
            for dns_name in config.dns_names:
                san_entries.append(x509.DNSName(dns_name))
            for ip_addr in config.ip_addresses:
                from ipaddress import ip_address
                san_entries.append(x509.IPAddress(ip_address(ip_addr)))

            builder = builder.add_extension(
                x509.SubjectAlternativeName(san_entries),
                critical=False,
            )

            # Sign the certificate
            certificate = builder.sign(
                private_key,
                hashes.SHA256(),
                default_backend(),
            )

            # Serialize to PEM format
            cert_pem = certificate.public_bytes(serialization.Encoding.PEM)
            key_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )

            # Save to files
            cert_path = self.cert_dir / f"{output_name}.pem"
            key_path = self.cert_dir / f"{output_name}_key.pem"

            cert_path.write_bytes(cert_pem)
            key_path.write_bytes(key_pem)

            # Set restrictive permissions on key file
            os.chmod(key_path, 0o600)

            logger.info(f"Generated certificate: {cert_path}")
            logger.info(f"Generated private key: {key_path}")

            # Cache the certificates
            self._cert_cache[output_name] = cert_pem
            self._key_cache[output_name] = key_pem

            return cert_pem, key_pem

        except ImportError:
            logger.warning("cryptography library not available, using placeholder certificates")
            return await self._generate_placeholder_certificate(config, output_name)

    async def _generate_placeholder_certificate(
        self,
        config: CertificateConfig,
        output_name: str,
    ) -> Tuple[bytes, bytes]:
        """
        Generate placeholder certificate when cryptography library unavailable.

        This should only be used for development/testing purposes.
        """
        logger.warning("Using placeholder certificate - NOT FOR PRODUCTION USE")

        # Placeholder PEM content (for testing only)
        cert_pem = b"""-----BEGIN CERTIFICATE-----
PLACEHOLDER_CERTIFICATE_FOR_DEVELOPMENT_ONLY
-----END CERTIFICATE-----
"""
        key_pem = b"""-----BEGIN PRIVATE KEY-----
PLACEHOLDER_KEY_FOR_DEVELOPMENT_ONLY
-----END PRIVATE KEY-----
"""

        # Save placeholder files
        cert_path = self.cert_dir / f"{output_name}.pem"
        key_path = self.cert_dir / f"{output_name}_key.pem"

        cert_path.write_bytes(cert_pem)
        key_path.write_bytes(key_pem)

        return cert_pem, key_pem

    async def load_certificate(self, cert_path: Union[str, Path]) -> bytes:
        """
        Load a certificate from file.

        Args:
            cert_path: Path to certificate file

        Returns:
            Certificate bytes in PEM or DER format
        """
        path = Path(cert_path)
        if not path.exists():
            raise FileNotFoundError(f"Certificate not found: {cert_path}")

        cert_data = path.read_bytes()
        logger.debug(f"Loaded certificate: {path}")
        return cert_data

    async def load_private_key(
        self,
        key_path: Union[str, Path],
        password: Optional[bytes] = None,
    ) -> bytes:
        """
        Load a private key from file.

        Args:
            key_path: Path to private key file
            password: Optional password for encrypted keys

        Returns:
            Private key bytes
        """
        path = Path(key_path)
        if not path.exists():
            raise FileNotFoundError(f"Private key not found: {key_path}")

        key_data = path.read_bytes()
        logger.debug(f"Loaded private key: {path}")
        return key_data

    async def get_certificate_info(
        self,
        cert_data: bytes,
    ) -> CertificateInfo:
        """
        Extract information from a certificate.

        Args:
            cert_data: Certificate bytes (PEM or DER)

        Returns:
            Certificate information
        """
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import hashes

            # Load certificate
            if b"-----BEGIN" in cert_data:
                cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            else:
                cert = x509.load_der_x509_certificate(cert_data, default_backend())

            # Calculate fingerprints
            sha1_fingerprint = cert.fingerprint(hashes.SHA1()).hex()
            sha256_fingerprint = cert.fingerprint(hashes.SHA256()).hex()

            # Get key info
            public_key = cert.public_key()
            key_type = type(public_key).__name__
            key_size = public_key.key_size if hasattr(public_key, "key_size") else 0

            # Check expiration
            now = datetime.utcnow()
            is_expired = now > cert.not_valid_after_utc.replace(tzinfo=None)
            days_until_expiry = (cert.not_valid_after_utc.replace(tzinfo=None) - now).days

            # Extract application URI from SAN
            application_uri = None
            try:
                san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
                for name in san.value:
                    if isinstance(name, x509.UniformResourceIdentifier):
                        application_uri = name.value
                        break
            except x509.ExtensionNotFound:
                pass

            return CertificateInfo(
                serial_number=format(cert.serial_number, "x"),
                subject=cert.subject.rfc4514_string(),
                issuer=cert.issuer.rfc4514_string(),
                not_before=cert.not_valid_before_utc.replace(tzinfo=None),
                not_after=cert.not_valid_after_utc.replace(tzinfo=None),
                fingerprint_sha1=sha1_fingerprint,
                fingerprint_sha256=sha256_fingerprint,
                application_uri=application_uri,
                key_type=key_type,
                key_size=key_size,
                is_self_signed=cert.subject == cert.issuer,
                is_expired=is_expired,
                days_until_expiry=days_until_expiry,
            )

        except ImportError:
            logger.warning("cryptography library not available")
            return CertificateInfo(
                serial_number="unknown",
                subject="unknown",
                issuer="unknown",
                not_before=datetime.utcnow(),
                not_after=datetime.utcnow() + timedelta(days=365),
                fingerprint_sha1="unknown",
                fingerprint_sha256="unknown",
                key_type="unknown",
                key_size=0,
                days_until_expiry=365,
            )

    async def add_trusted_certificate(
        self,
        cert_data: bytes,
        name: Optional[str] = None,
    ) -> Path:
        """
        Add a certificate to the trusted store.

        Args:
            cert_data: Certificate bytes
            name: Optional name for the certificate file

        Returns:
            Path to saved certificate
        """
        if not name:
            # Generate name from fingerprint
            fingerprint = hashlib.sha256(cert_data).hexdigest()[:16]
            name = f"trusted_{fingerprint}"

        cert_path = self.trusted_dir / f"{name}.pem"
        cert_path.write_bytes(cert_data)
        logger.info(f"Added trusted certificate: {cert_path}")
        return cert_path

    async def remove_trusted_certificate(self, cert_path: Union[str, Path]) -> bool:
        """
        Remove a certificate from the trusted store.

        Args:
            cert_path: Path to certificate to remove

        Returns:
            True if removed successfully
        """
        path = Path(cert_path)
        if path.exists() and path.parent == self.trusted_dir:
            path.unlink()
            logger.info(f"Removed trusted certificate: {path}")
            return True
        return False

    async def reject_certificate(
        self,
        cert_data: bytes,
        reason: str = "Rejected by user",
    ) -> Path:
        """
        Move a certificate to the rejected store.

        Args:
            cert_data: Certificate bytes
            reason: Reason for rejection

        Returns:
            Path to rejected certificate
        """
        fingerprint = hashlib.sha256(cert_data).hexdigest()[:16]
        cert_path = self.rejected_dir / f"rejected_{fingerprint}.pem"
        cert_path.write_bytes(cert_data)

        # Write reason to companion file
        reason_path = self.rejected_dir / f"rejected_{fingerprint}.reason"
        reason_path.write_text(f"{datetime.utcnow().isoformat()}: {reason}")

        logger.info(f"Rejected certificate: {cert_path}, reason: {reason}")
        return cert_path

    async def is_certificate_trusted(self, cert_data: bytes) -> bool:
        """
        Check if a certificate is in the trusted store.

        Args:
            cert_data: Certificate bytes to check

        Returns:
            True if certificate is trusted
        """
        fingerprint = hashlib.sha256(cert_data).hexdigest()

        for cert_file in self.trusted_dir.glob("*.pem"):
            trusted_data = cert_file.read_bytes()
            trusted_fingerprint = hashlib.sha256(trusted_data).hexdigest()
            if fingerprint == trusted_fingerprint:
                return True

        return False

    async def is_certificate_rejected(self, cert_data: bytes) -> bool:
        """
        Check if a certificate is in the rejected store.

        Args:
            cert_data: Certificate bytes to check

        Returns:
            True if certificate is rejected
        """
        fingerprint = hashlib.sha256(cert_data).hexdigest()

        for cert_file in self.rejected_dir.glob("*.pem"):
            rejected_data = cert_file.read_bytes()
            rejected_fingerprint = hashlib.sha256(rejected_data).hexdigest()
            if fingerprint == rejected_fingerprint:
                return True

        return False

    async def list_trusted_certificates(self) -> List[CertificateInfo]:
        """
        List all trusted certificates.

        Returns:
            List of certificate information
        """
        certificates = []
        for cert_file in self.trusted_dir.glob("*.pem"):
            cert_data = cert_file.read_bytes()
            info = await self.get_certificate_info(cert_data)
            certificates.append(info)
        return certificates


# =============================================================================
# Security Manager
# =============================================================================


class SecurityManager:
    """
    Manages security configuration for OPC-UA connections.

    Handles security policy selection, authentication, and secure channel setup.
    """

    # Security policy URIs
    SECURITY_POLICY_URIS = {
        SecurityPolicy.NONE: "http://opcfoundation.org/UA/SecurityPolicy#None",
        SecurityPolicy.BASIC128RSA15: "http://opcfoundation.org/UA/SecurityPolicy#Basic128Rsa15",
        SecurityPolicy.BASIC256: "http://opcfoundation.org/UA/SecurityPolicy#Basic256",
        SecurityPolicy.BASIC256SHA256: "http://opcfoundation.org/UA/SecurityPolicy#Basic256Sha256",
        SecurityPolicy.AES128_SHA256_RSAOAEP: "http://opcfoundation.org/UA/SecurityPolicy#Aes128_Sha256_RsaOaep",
        SecurityPolicy.AES256_SHA256_RSAPSS: "http://opcfoundation.org/UA/SecurityPolicy#Aes256_Sha256_RsaPss",
    }

    # Minimum recommended security policy
    MINIMUM_SECURE_POLICY = SecurityPolicy.BASIC256SHA256

    def __init__(
        self,
        certificate_manager: CertificateManager,
        allow_insecure: bool = False,
    ):
        """
        Initialize security manager.

        Args:
            certificate_manager: Certificate manager instance
            allow_insecure: Whether to allow insecure connections (not recommended)
        """
        self.cert_manager = certificate_manager
        self.allow_insecure = allow_insecure

        self._client_certificate: Optional[bytes] = None
        self._client_private_key: Optional[bytes] = None
        self._server_certificate: Optional[bytes] = None

    async def setup_security(
        self,
        config: ConnectionConfig,
    ) -> Dict[str, Any]:
        """
        Set up security configuration for a connection.

        Args:
            config: Connection configuration

        Returns:
            Security parameters for OPC-UA client
        """
        security_params: Dict[str, Any] = {
            "security_policy": self.SECURITY_POLICY_URIS.get(config.security_policy),
            "security_mode": config.security_mode.value,
            "certificate": None,
            "private_key": None,
            "server_certificate": None,
            "user_token": None,
        }

        # Validate security policy
        if config.security_policy != SecurityPolicy.NONE:
            if not self.allow_insecure and config.security_policy.value < self.MINIMUM_SECURE_POLICY.value:
                logger.warning(
                    f"Security policy {config.security_policy} is below recommended minimum "
                    f"({self.MINIMUM_SECURE_POLICY}). Consider using a stronger policy."
                )

        # Load or generate client certificate
        if config.security_policy != SecurityPolicy.NONE:
            if config.certificate_path and config.private_key_path:
                # Load existing certificates
                self._client_certificate = await self.cert_manager.load_certificate(
                    config.certificate_path
                )
                self._client_private_key = await self.cert_manager.load_private_key(
                    config.private_key_path
                )
            else:
                # Generate new certificates
                cert_config = CertificateConfig(
                    common_name=config.application_name,
                    application_uri=config.application_uri,
                )
                self._client_certificate, self._client_private_key = (
                    await self.cert_manager.generate_client_certificate(cert_config)
                )

            security_params["certificate"] = self._client_certificate
            security_params["private_key"] = self._client_private_key

            # Load server certificate if provided
            if config.server_certificate_path:
                self._server_certificate = await self.cert_manager.load_certificate(
                    config.server_certificate_path
                )
                security_params["server_certificate"] = self._server_certificate

        # Set up user authentication
        if config.authentication_type == AuthenticationType.USERNAME_PASSWORD:
            if not config.username or not config.password:
                raise ValueError("Username and password required for UsernamePassword authentication")
            security_params["user_token"] = {
                "type": "username",
                "username": config.username,
                "password": config.password,
            }

        elif config.authentication_type == AuthenticationType.CERTIFICATE:
            security_params["user_token"] = {
                "type": "certificate",
                "certificate": self._client_certificate,
            }

        elif config.authentication_type == AuthenticationType.ANONYMOUS:
            security_params["user_token"] = {
                "type": "anonymous",
            }

        logger.info(
            f"Security configured: policy={config.security_policy.value}, "
            f"mode={config.security_mode.value}, auth={config.authentication_type.value}"
        )

        return security_params

    async def validate_server_certificate(
        self,
        server_certificate: bytes,
        endpoint_url: str,
        auto_trust: bool = False,
    ) -> bool:
        """
        Validate a server certificate.

        Args:
            server_certificate: Server certificate bytes
            endpoint_url: Expected endpoint URL
            auto_trust: Whether to auto-trust unknown certificates

        Returns:
            True if certificate is valid and trusted
        """
        # Check if already rejected
        if await self.cert_manager.is_certificate_rejected(server_certificate):
            logger.warning("Server certificate is in rejected list")
            return False

        # Check if trusted
        if await self.cert_manager.is_certificate_trusted(server_certificate):
            logger.debug("Server certificate is trusted")
            return True

        # Get certificate info
        cert_info = await self.cert_manager.get_certificate_info(server_certificate)

        # Check expiration
        if cert_info.is_expired:
            logger.error(f"Server certificate is expired: {cert_info.not_after}")
            await self.cert_manager.reject_certificate(
                server_certificate,
                reason="Certificate expired"
            )
            return False

        # Check if certificate will expire soon
        if cert_info.days_until_expiry < 30:
            logger.warning(
                f"Server certificate expires in {cert_info.days_until_expiry} days"
            )

        # Validate application URI if present
        if cert_info.application_uri:
            # Extract hostname from endpoint URL
            from urllib.parse import urlparse
            parsed = urlparse(endpoint_url)
            expected_host = parsed.hostname

            # Basic validation - in production, more thorough checks needed
            if expected_host and expected_host not in cert_info.application_uri:
                logger.warning(
                    f"Application URI mismatch: {cert_info.application_uri} vs {endpoint_url}"
                )

        # Handle unknown certificate
        if auto_trust:
            logger.warning("Auto-trusting unknown server certificate")
            await self.cert_manager.add_trusted_certificate(server_certificate)
            return True

        logger.warning(
            f"Unknown server certificate: {cert_info.subject}, "
            f"fingerprint: {cert_info.fingerprint_sha256[:16]}..."
        )
        return False

    def get_security_policy_uri(self, policy: SecurityPolicy) -> str:
        """
        Get the URI for a security policy.

        Args:
            policy: Security policy enum

        Returns:
            Security policy URI string
        """
        return self.SECURITY_POLICY_URIS.get(
            policy,
            self.SECURITY_POLICY_URIS[SecurityPolicy.NONE]
        )

    @staticmethod
    def generate_nonce(length: int = 32) -> bytes:
        """
        Generate a cryptographic nonce.

        Args:
            length: Nonce length in bytes

        Returns:
            Random nonce bytes
        """
        return secrets.token_bytes(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """
        Hash a password using PBKDF2.

        Args:
            password: Password to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations=100000,
        )

        return password_hash, salt


# =============================================================================
# Credential Manager (for secrets management)
# =============================================================================


class CredentialManager:
    """
    Manages credentials for OPC-UA authentication.

    Supports multiple credential storage backends:
    - Environment variables
    - File-based (encrypted)
    - HashiCorp Vault
    - AWS Secrets Manager
    """

    def __init__(
        self,
        backend: str = "env",
        vault_url: Optional[str] = None,
        vault_token: Optional[str] = None,
    ):
        """
        Initialize credential manager.

        Args:
            backend: Credential backend (env, vault, aws)
            vault_url: HashiCorp Vault URL
            vault_token: Vault authentication token
        """
        self.backend = backend
        self.vault_url = vault_url
        self.vault_token = vault_token

        self._cache: Dict[str, str] = {}

    async def get_credential(self, key: str) -> Optional[str]:
        """
        Retrieve a credential by key.

        Args:
            key: Credential key

        Returns:
            Credential value or None
        """
        # Check cache first
        if key in self._cache:
            return self._cache[key]

        value = None

        if self.backend == "env":
            value = os.environ.get(key)

        elif self.backend == "vault":
            value = await self._get_from_vault(key)

        elif self.backend == "aws":
            value = await self._get_from_aws(key)

        if value:
            self._cache[key] = value

        return value

    async def _get_from_vault(self, key: str) -> Optional[str]:
        """Get credential from HashiCorp Vault."""
        if not self.vault_url or not self.vault_token:
            logger.warning("Vault not configured")
            return None

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vault_url}/v1/secret/data/{key}",
                    headers={"X-Vault-Token": self.vault_token},
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("data", {}).get("data", {}).get("value")
        except Exception as e:
            logger.error(f"Failed to get credential from Vault: {e}")

        return None

    async def _get_from_aws(self, key: str) -> Optional[str]:
        """Get credential from AWS Secrets Manager."""
        try:
            import boto3

            client = boto3.client("secretsmanager")
            response = client.get_secret_value(SecretId=key)
            return response.get("SecretString")
        except Exception as e:
            logger.error(f"Failed to get credential from AWS: {e}")

        return None

    def clear_cache(self) -> None:
        """Clear the credential cache."""
        self._cache.clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Configuration
    "CertificateType",
    "KeyType",
    "CertificateConfig",
    "CertificateInfo",
    # Certificate management
    "CertificateManager",
    # Security management
    "SecurityManager",
    # Credential management
    "CredentialManager",
]
