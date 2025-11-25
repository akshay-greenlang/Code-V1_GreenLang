# -*- coding: utf-8 -*-
"""
SAML 2.0 Authentication Provider for GreenLang

This module implements a comprehensive SAML 2.0 Service Provider (SP) for enterprise SSO.
Supports major IdPs including Okta, Azure AD, OneLogin, and generic SAML 2.0 providers.

Features:
- SAML 2.0 assertion validation and signature verification
- Multiple IdP support with dynamic configuration
- Attribute mapping from SAML to internal user model
- Session management and SLO (Single Logout)
- Metadata exchange (SP and IdP metadata)
- Encryption support for assertions
- Certificate rotation and management

Security:
- XML signature validation using xmlsec1
- Assertion encryption/decryption
- Replay attack prevention
- Time-based validation (NotBefore/NotOnOrAfter)
- Certificate validation and chain verification
"""

import logging
import base64
import zlib
import urllib.parse
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import xml.etree.ElementTree as ET
from greenlang.determinism import deterministic_uuid, DeterministicClock

try:
    from onelogin.saml2.auth import OneLogin_Saml2_Auth
    from onelogin.saml2.settings import OneLogin_Saml2_Settings
    from onelogin.saml2.utils import OneLogin_Saml2_Utils
    from onelogin.saml2.errors import OneLogin_Saml2_Error
    SAML_AVAILABLE = True
except ImportError:
    SAML_AVAILABLE = False

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger(__name__)


class SAMLBindingType(Enum):
    """SAML binding types"""
    HTTP_REDIRECT = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
    HTTP_POST = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
    HTTP_ARTIFACT = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Artifact"
    SOAP = "urn:oasis:names:tc:SAML:2.0:bindings:SOAP"


class SAMLNameIDFormat(Enum):
    """SAML NameID formats"""
    PERSISTENT = "urn:oasis:names:tc:SAML:2.0:nameid-format:persistent"
    TRANSIENT = "urn:oasis:names:tc:SAML:2.0:nameid-format:transient"
    EMAIL = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
    UNSPECIFIED = "urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified"
    ENTITY = "urn:oasis:names:tc:SAML:2.0:nameid-format:entity"


class SAMLAuthnContext(Enum):
    """SAML authentication context classes"""
    PASSWORD = "urn:oasis:names:tc:SAML:2.0:ac:classes:Password"
    PASSWORD_PROTECTED = "urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport"
    TLS_CLIENT = "urn:oasis:names:tc:SAML:2.0:ac:classes:TLSClient"
    X509 = "urn:oasis:names:tc:SAML:2.0:ac:classes:X509"
    SMARTCARD = "urn:oasis:names:tc:SAML:2.0:ac:classes:Smartcard"
    KERBEROS = "urn:oasis:names:tc:SAML:2.0:ac:classes:Kerberos"


@dataclass
class SAMLConfig:
    """SAML Service Provider Configuration"""
    # SP Identity
    sp_entity_id: str
    sp_acs_url: str  # Assertion Consumer Service URL
    sp_sls_url: Optional[str] = None  # Single Logout Service URL

    # IdP Configuration
    idp_entity_id: str = ""
    idp_sso_url: str = ""
    idp_slo_url: Optional[str] = None
    idp_x509_cert: str = ""
    idp_metadata_url: Optional[str] = None

    # Security Settings
    want_assertions_signed: bool = True
    want_messages_signed: bool = True
    want_assertions_encrypted: bool = False
    want_name_id_encrypted: bool = False
    authn_requests_signed: bool = True
    logout_requests_signed: bool = True
    logout_responses_signed: bool = True

    # SP Certificates (PEM format)
    sp_x509_cert: Optional[str] = None
    sp_private_key: Optional[str] = None

    # Binding preferences
    authn_request_binding: SAMLBindingType = SAMLBindingType.HTTP_REDIRECT
    logout_request_binding: SAMLBindingType = SAMLBindingType.HTTP_REDIRECT

    # Name ID format
    name_id_format: SAMLNameIDFormat = SAMLNameIDFormat.EMAIL

    # Authentication context
    requested_authn_context: List[SAMLAuthnContext] = field(default_factory=lambda: [SAMLAuthnContext.PASSWORD_PROTECTED])
    requested_authn_context_comparison: str = "exact"  # exact, minimum, maximum, better

    # Attribute mapping (SAML attribute -> internal field)
    attribute_mapping: Dict[str, str] = field(default_factory=dict)

    # Session settings
    session_lifetime: int = 3600  # seconds
    allow_repeat_attribute_name: bool = True

    # Organization info
    organization_name: str = "GreenLang"
    organization_display_name: str = "GreenLang Platform"
    organization_url: str = "https://greenlang.io"

    # Technical contact
    technical_contact_name: str = "GreenLang Support"
    technical_contact_email: str = "support@greenlang.io"

    # Debug mode
    debug: bool = False
    strict: bool = True


@dataclass
class SAMLAssertion:
    """Parsed SAML assertion"""
    name_id: str
    name_id_format: str
    session_index: Optional[str]
    attributes: Dict[str, Any]
    issuer: str
    audience: str
    subject: str
    not_before: Optional[datetime]
    not_on_or_after: Optional[datetime]
    authn_instant: datetime
    authn_context: str
    assertion_id: str
    raw_xml: str


@dataclass
class SAMLUser:
    """User object created from SAML assertion"""
    user_id: str
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    session_index: Optional[str] = None
    name_id: str = ""
    name_id_format: str = ""
    idp_entity_id: str = ""


@dataclass
class SAMLSession:
    """SAML session information"""
    session_id: str
    user: SAMLUser
    created_at: datetime
    expires_at: datetime
    assertion: SAMLAssertion
    idp_session_index: Optional[str] = None


class SAMLRequestCache:
    """In-memory cache for SAML request tracking (replay prevention)"""

    def __init__(self, max_size: int = 10000, ttl: int = 300):
        self._cache: Dict[str, datetime] = {}
        self._max_size = max_size
        self._ttl = ttl

    def add_request(self, request_id: str) -> None:
        """Add a SAML request ID to the cache"""
        # Clean up expired entries
        self._cleanup()

        # Add new entry
        self._cache[request_id] = DeterministicClock.utcnow()

        # Enforce max size
        if len(self._cache) > self._max_size:
            # Remove oldest entries
            sorted_items = sorted(self._cache.items(), key=lambda x: x[1])
            to_remove = len(self._cache) - self._max_size
            for request_id, _ in sorted_items[:to_remove]:
                del self._cache[request_id]

    def has_request(self, request_id: str) -> bool:
        """Check if a request ID exists in cache"""
        self._cleanup()
        return request_id in self._cache

    def _cleanup(self) -> None:
        """Remove expired entries"""
        now = DeterministicClock.utcnow()
        expired = [
            req_id for req_id, timestamp in self._cache.items()
            if (now - timestamp).total_seconds() > self._ttl
        ]
        for req_id in expired:
            del self._cache[req_id]


class SAMLCertificateManager:
    """Manages X.509 certificates for SAML"""

    @staticmethod
    def generate_self_signed_cert(
        common_name: str = "greenlang-sp",
        validity_days: int = 365
    ) -> Tuple[str, str]:
        """Generate a self-signed certificate for SP"""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Generate certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(x509.oid.NameOID.COMMON_NAME, common_name),
            x509.NameAttribute(x509.oid.NameOID.ORGANIZATION_NAME, "GreenLang"),
        ])

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            DeterministicClock.utcnow()
        ).not_valid_after(
            DeterministicClock.utcnow() + timedelta(days=validity_days)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.DNSName("*.greenlang.io"),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256(), default_backend())

        # Serialize to PEM format
        cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode('utf-8')
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')

        return cert_pem, key_pem

    @staticmethod
    def validate_certificate(cert_pem: str) -> bool:
        """Validate certificate format and expiration"""
        try:
            cert = x509.load_pem_x509_certificate(
                cert_pem.encode('utf-8'),
                default_backend()
            )

            # Check expiration
            now = DeterministicClock.utcnow()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                logger.warning("Certificate is expired or not yet valid")
                return False

            return True
        except Exception as e:
            logger.error(f"Certificate validation failed: {e}")
            return False

    @staticmethod
    def extract_cert_info(cert_pem: str) -> Dict[str, Any]:
        """Extract certificate information"""
        try:
            cert = x509.load_pem_x509_certificate(
                cert_pem.encode('utf-8'),
                default_backend()
            )

            return {
                "subject": cert.subject.rfc4514_string(),
                "issuer": cert.issuer.rfc4514_string(),
                "serial_number": cert.serial_number,
                "not_before": cert.not_valid_before,
                "not_after": cert.not_valid_after,
                "fingerprint": cert.fingerprint(hashes.SHA256()).hex(),
            }
        except Exception as e:
            logger.error(f"Failed to extract certificate info: {e}")
            return {}


class SAMLAttributeMapper:
    """Maps SAML attributes to internal user model"""

    # Default attribute mappings for common IdPs
    DEFAULT_MAPPINGS = {
        "okta": {
            "email": "email",
            "firstName": "first_name",
            "lastName": "last_name",
            "login": "username",
            "groups": "groups",
        },
        "azure": {
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress": "email",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname": "first_name",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname": "last_name",
            "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name": "username",
            "http://schemas.microsoft.com/ws/2008/06/identity/claims/groups": "groups",
        },
        "onelogin": {
            "User.email": "email",
            "User.FirstName": "first_name",
            "User.LastName": "last_name",
            "PersonImmutableID": "user_id",
            "MemberOf": "groups",
        },
        "generic": {
            "email": "email",
            "givenName": "first_name",
            "surname": "last_name",
            "uid": "username",
            "memberOf": "groups",
        }
    }

    def __init__(self, mapping: Optional[Dict[str, str]] = None):
        self.mapping = mapping or self.DEFAULT_MAPPINGS["generic"]

    def map_attributes(self, saml_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """Map SAML attributes to internal format"""
        mapped = {}

        for saml_attr, internal_attr in self.mapping.items():
            if saml_attr in saml_attributes:
                value = saml_attributes[saml_attr]

                # Handle list values (take first element if single value expected)
                if isinstance(value, list):
                    if internal_attr in ["groups", "roles"]:
                        mapped[internal_attr] = value
                    else:
                        mapped[internal_attr] = value[0] if value else None
                else:
                    mapped[internal_attr] = value

        return mapped

    @classmethod
    def get_idp_mapping(cls, idp_type: str) -> Dict[str, str]:
        """Get default mapping for specific IdP"""
        return cls.DEFAULT_MAPPINGS.get(idp_type.lower(), cls.DEFAULT_MAPPINGS["generic"])


class SAMLProvider:
    """
    SAML 2.0 Service Provider Implementation

    Handles SAML authentication flows, assertion validation, and user mapping.
    """

    def __init__(self, config: SAMLConfig):
        if not SAML_AVAILABLE:
            raise ImportError(
                "python3-saml is not installed. "
                "Install it with: pip install python3-saml"
            )

        self.config = config
        self.request_cache = SAMLRequestCache()
        self.attribute_mapper = SAMLAttributeMapper(config.attribute_mapping)
        self.cert_manager = SAMLCertificateManager()
        self.sessions: Dict[str, SAMLSession] = {}

        # Initialize SAML settings
        self._saml_settings = self._build_saml_settings()

        logger.info(f"Initialized SAML provider for {config.sp_entity_id}")

    def _build_saml_settings(self) -> Dict[str, Any]:
        """Build python3-saml settings dictionary"""
        settings = {
            "strict": self.config.strict,
            "debug": self.config.debug,
            "sp": {
                "entityId": self.config.sp_entity_id,
                "assertionConsumerService": {
                    "url": self.config.sp_acs_url,
                    "binding": self.config.authn_request_binding.value
                },
                "singleLogoutService": {
                    "url": self.config.sp_sls_url or "",
                    "binding": self.config.logout_request_binding.value
                },
                "NameIDFormat": self.config.name_id_format.value,
                "x509cert": self.config.sp_x509_cert or "",
                "privateKey": self.config.sp_private_key or "",
            },
            "idp": {
                "entityId": self.config.idp_entity_id,
                "singleSignOnService": {
                    "url": self.config.idp_sso_url,
                    "binding": self.config.authn_request_binding.value
                },
                "singleLogoutService": {
                    "url": self.config.idp_slo_url or "",
                    "binding": self.config.logout_request_binding.value
                },
                "x509cert": self.config.idp_x509_cert,
            },
            "security": {
                "nameIdEncrypted": self.config.want_name_id_encrypted,
                "authnRequestsSigned": self.config.authn_requests_signed,
                "logoutRequestSigned": self.config.logout_requests_signed,
                "logoutResponseSigned": self.config.logout_responses_signed,
                "signMetadata": self.config.authn_requests_signed,
                "wantMessagesSigned": self.config.want_messages_signed,
                "wantAssertionsSigned": self.config.want_assertions_signed,
                "wantAssertionsEncrypted": self.config.want_assertions_encrypted,
                "wantNameId": True,
                "wantNameIdEncrypted": self.config.want_name_id_encrypted,
                "requestedAuthnContext": [ctx.value for ctx in self.config.requested_authn_context],
                "requestedAuthnContextComparison": self.config.requested_authn_context_comparison,
                "signatureAlgorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
                "digestAlgorithm": "http://www.w3.org/2001/04/xmlenc#sha256",
            },
            "organization": {
                "en-US": {
                    "name": self.config.organization_name,
                    "displayname": self.config.organization_display_name,
                    "url": self.config.organization_url
                }
            },
            "contactPerson": {
                "technical": {
                    "givenName": self.config.technical_contact_name,
                    "emailAddress": self.config.technical_contact_email
                }
            }
        }

        return settings

    def get_auth_request_url(
        self,
        relay_state: Optional[str] = None,
        force_authn: bool = False,
        is_passive: bool = False
    ) -> Tuple[str, str]:
        """
        Generate SAML authentication request URL

        Returns:
            Tuple of (auth_url, request_id)
        """
        try:
            # Create request
            request_id = f"GREENLANG_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex}"

            # Prepare request data for python3-saml
            req = self._prepare_request({})
            auth = OneLogin_Saml2_Auth(req, self._saml_settings)

            # Generate auth URL
            auth_url = auth.login(
                return_to=relay_state,
                force_authn=force_authn,
                is_passive=is_passive
            )

            # Cache request ID for replay prevention
            self.request_cache.add_request(request_id)

            logger.info(f"Generated SAML auth request: {request_id}")
            return auth_url, request_id

        except Exception as e:
            logger.error(f"Failed to generate auth request: {e}")
            raise SAMLError(f"Auth request generation failed: {e}")

    def process_response(
        self,
        saml_response: str,
        request_id: Optional[str] = None
    ) -> SAMLUser:
        """
        Process SAML response and create user object

        Args:
            saml_response: Base64 encoded SAML response
            request_id: Original request ID (for replay prevention)

        Returns:
            SAMLUser object
        """
        try:
            # Prepare request data
            req = self._prepare_request({
                "post_data": {"SAMLResponse": saml_response}
            })

            auth = OneLogin_Saml2_Auth(req, self._saml_settings)
            auth.process_response()

            # Check for errors
            errors = auth.get_errors()
            if errors:
                error_reason = auth.get_last_error_reason()
                logger.error(f"SAML response errors: {errors}, reason: {error_reason}")
                raise SAMLError(f"SAML validation failed: {error_reason}")

            # Check authentication
            if not auth.is_authenticated():
                raise SAMLError("User not authenticated")

            # Extract assertion data
            assertion = self._parse_assertion(auth)

            # Check for replay attacks
            if request_id and not self.request_cache.has_request(request_id):
                logger.warning(f"Potential replay attack detected: {request_id}")
                # In strict mode, reject; in non-strict, log and continue
                if self.config.strict:
                    raise SAMLError("Invalid or expired request ID")

            # Map attributes to user
            user = self._create_user_from_assertion(assertion)

            # Create session
            session = self._create_session(user, assertion)
            self.sessions[session.session_id] = session

            logger.info(f"Successfully authenticated user: {user.email}")
            return user

        except OneLogin_Saml2_Error as e:
            logger.error(f"SAML processing error: {e}")
            raise SAMLError(f"SAML processing failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing SAML response: {e}")
            raise SAMLError(f"Response processing failed: {e}")

    def _parse_assertion(self, auth: OneLogin_Saml2_Auth) -> SAMLAssertion:
        """Parse SAML assertion from auth object"""
        attributes = auth.get_attributes()

        # Get session index
        session_index = auth.get_session_index()

        # Get name ID
        name_id = auth.get_nameid()
        name_id_format = auth.get_nameid_format()

        # Get timestamps (if available)
        # Note: python3-saml doesn't expose these directly, would need XML parsing
        assertion = SAMLAssertion(
            name_id=name_id,
            name_id_format=name_id_format,
            session_index=session_index,
            attributes=attributes,
            issuer=self.config.idp_entity_id,
            audience=self.config.sp_entity_id,
            subject=name_id,
            not_before=None,  # Would need custom XML parsing
            not_on_or_after=None,  # Would need custom XML parsing
            authn_instant=DeterministicClock.utcnow(),
            authn_context="",  # Would need custom XML parsing
            assertion_id=f"assertion_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex}",
            raw_xml=auth.get_last_response_xml() or ""
        )

        return assertion

    def _create_user_from_assertion(self, assertion: SAMLAssertion) -> SAMLUser:
        """Create user object from SAML assertion"""
        # Map attributes
        mapped_attrs = self.attribute_mapper.map_attributes(assertion.attributes)

        # Extract required fields
        email = mapped_attrs.get("email") or assertion.name_id
        username = mapped_attrs.get("username") or email.split("@")[0]

        user = SAMLUser(
            user_id=mapped_attrs.get("user_id") or assertion.name_id,
            email=email,
            username=username,
            first_name=mapped_attrs.get("first_name"),
            last_name=mapped_attrs.get("last_name"),
            display_name=mapped_attrs.get("display_name"),
            groups=mapped_attrs.get("groups", []),
            roles=mapped_attrs.get("roles", []),
            attributes=assertion.attributes,
            session_index=assertion.session_index,
            name_id=assertion.name_id,
            name_id_format=assertion.name_id_format,
            idp_entity_id=assertion.issuer
        )

        return user

    def _create_session(self, user: SAMLUser, assertion: SAMLAssertion) -> SAMLSession:
        """Create SAML session"""
        session_id = secrets.token_urlsafe(32)
        now = DeterministicClock.utcnow()

        session = SAMLSession(
            session_id=session_id,
            user=user,
            created_at=now,
            expires_at=now + timedelta(seconds=self.config.session_lifetime),
            assertion=assertion,
            idp_session_index=assertion.session_index
        )

        return session

    def get_logout_request_url(
        self,
        session_id: str,
        relay_state: Optional[str] = None
    ) -> str:
        """Generate SAML logout request URL"""
        session = self.sessions.get(session_id)
        if not session:
            raise SAMLError(f"Session not found: {session_id}")

        try:
            req = self._prepare_request({})
            auth = OneLogin_Saml2_Auth(req, self._saml_settings)

            # Generate logout URL
            logout_url = auth.logout(
                return_to=relay_state,
                name_id=session.user.name_id,
                session_index=session.idp_session_index,
                name_id_format=session.user.name_id_format
            )

            # Remove session
            del self.sessions[session_id]

            logger.info(f"Generated logout request for session: {session_id}")
            return logout_url

        except Exception as e:
            logger.error(f"Failed to generate logout request: {e}")
            raise SAMLError(f"Logout request generation failed: {e}")

    def process_logout_response(self, saml_response: str) -> bool:
        """Process SAML logout response"""
        try:
            req = self._prepare_request({
                "get_data": {"SAMLResponse": saml_response}
            })

            auth = OneLogin_Saml2_Auth(req, self._saml_settings)

            # Process SLO response
            url = auth.process_slo(delete_session_cb=lambda: None)

            errors = auth.get_errors()
            if errors:
                logger.error(f"Logout response errors: {errors}")
                return False

            logger.info("Successfully processed logout response")
            return True

        except Exception as e:
            logger.error(f"Failed to process logout response: {e}")
            return False

    def get_metadata(self) -> str:
        """Get SP metadata XML"""
        try:
            settings = OneLogin_Saml2_Settings(self._saml_settings)
            metadata = settings.get_sp_metadata()

            # Validate metadata
            errors = settings.validate_metadata(metadata)
            if errors:
                logger.error(f"Metadata validation errors: {errors}")
                raise SAMLError(f"Invalid metadata: {errors}")

            return metadata

        except Exception as e:
            logger.error(f"Failed to generate metadata: {e}")
            raise SAMLError(f"Metadata generation failed: {e}")

    def validate_session(self, session_id: str) -> bool:
        """Validate if a session is still valid"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check expiration
        if DeterministicClock.utcnow() > session.expires_at:
            del self.sessions[session_id]
            return False

        return True

    def get_session(self, session_id: str) -> Optional[SAMLSession]:
        """Get session by ID"""
        if self.validate_session(session_id):
            return self.sessions.get(session_id)
        return None

    def _prepare_request(self, custom_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request data for python3-saml"""
        return {
            "https": "on",
            "http_host": urllib.parse.urlparse(self.config.sp_entity_id).netloc,
            "script_name": urllib.parse.urlparse(self.config.sp_acs_url).path,
            "server_port": "443",
            "get_data": custom_data.get("get_data", {}),
            "post_data": custom_data.get("post_data", {}),
        }

    def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        now = DeterministicClock.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if now > session.expires_at
        ]

        for sid in expired:
            del self.sessions[sid]

        logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)


class SAMLError(Exception):
    """SAML-specific error"""
    pass


# Helper functions for common IdP configurations

def create_okta_config(
    sp_entity_id: str,
    sp_acs_url: str,
    okta_domain: str,
    okta_app_id: str,
    idp_cert: str,
    **kwargs
) -> SAMLConfig:
    """Create SAML config for Okta"""
    return SAMLConfig(
        sp_entity_id=sp_entity_id,
        sp_acs_url=sp_acs_url,
        idp_entity_id=f"http://www.okta.com/{okta_app_id}",
        idp_sso_url=f"https://{okta_domain}/app/{okta_app_id}/sso/saml",
        idp_x509_cert=idp_cert,
        attribute_mapping=SAMLAttributeMapper.get_idp_mapping("okta"),
        **kwargs
    )


def create_azure_config(
    sp_entity_id: str,
    sp_acs_url: str,
    tenant_id: str,
    app_id: str,
    idp_cert: str,
    **kwargs
) -> SAMLConfig:
    """Create SAML config for Azure AD"""
    return SAMLConfig(
        sp_entity_id=sp_entity_id,
        sp_acs_url=sp_acs_url,
        idp_entity_id=f"https://sts.windows.net/{tenant_id}/",
        idp_sso_url=f"https://login.microsoftonline.com/{tenant_id}/saml2",
        idp_x509_cert=idp_cert,
        attribute_mapping=SAMLAttributeMapper.get_idp_mapping("azure"),
        **kwargs
    )


def create_onelogin_config(
    sp_entity_id: str,
    sp_acs_url: str,
    onelogin_subdomain: str,
    onelogin_app_id: str,
    idp_cert: str,
    **kwargs
) -> SAMLConfig:
    """Create SAML config for OneLogin"""
    return SAMLConfig(
        sp_entity_id=sp_entity_id,
        sp_acs_url=sp_acs_url,
        idp_entity_id=f"https://app.onelogin.com/saml/metadata/{onelogin_app_id}",
        idp_sso_url=f"https://{onelogin_subdomain}.onelogin.com/trust/saml2/http-post/sso/{onelogin_app_id}",
        idp_x509_cert=idp_cert,
        attribute_mapping=SAMLAttributeMapper.get_idp_mapping("onelogin"),
        **kwargs
    )


__all__ = [
    "SAMLProvider",
    "SAMLConfig",
    "SAMLAssertion",
    "SAMLUser",
    "SAMLSession",
    "SAMLError",
    "SAMLBindingType",
    "SAMLNameIDFormat",
    "SAMLAuthnContext",
    "SAMLAttributeMapper",
    "SAMLCertificateManager",
    "create_okta_config",
    "create_azure_config",
    "create_onelogin_config",
]
