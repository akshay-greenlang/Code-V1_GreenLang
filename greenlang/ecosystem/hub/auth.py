# -*- coding: utf-8 -*-
"""
Authentication Module for GreenLang Hub
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
from ..security.http import SecureHTTPSession
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization
import jwt
from greenlang.utilities.determinism import DeterministicClock

try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)


class HubAuth:
    """Authentication handler for GreenLang Hub"""

    TOKEN_EXPIRY_DAYS = 30
    KEYRING_SERVICE = "greenlang-hub"

    def __init__(
        self,
        username: Optional[str] = None,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize authentication

        Args:
            username: Hub username
            token: Authentication token
            api_key: API key for service accounts
        """
        self.username = username
        self.token = token
        self.api_key = api_key

        # Config directory for storing auth data
        self.config_dir = Path.home() / ".greenlang" / "auth"
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load saved credentials if not provided
        if not token and not api_key:
            self._load_credentials()

    def get_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for requests

        Returns:
            Dictionary of headers
        """
        headers = {}

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        elif self.api_key:
            headers["X-API-Key"] = self.api_key

        if self.username:
            headers["X-Username"] = self.username

        return headers

    def login(
        self,
        username: str,
        password: str,
        registry_url: str = "https://hub.greenlang.io",
    ) -> bool:
        """
        Login to hub and obtain token

        Args:
            username: Username
            password: Password
            registry_url: Registry URL

        Returns:
            True if successful
        """
        try:
            logger.info(f"Logging in as {username}")

            with SecureHTTPSession() as session:
                response = session.post(
                    f"{registry_url}/api/v1/auth/login",
                    json={"username": username, "password": password},
                )

            response.raise_for_status()

            data = response.json()
            self.token = data.get("token")
            self.username = username

            # Save credentials
            self._save_credentials()

            logger.info("Login successful")
            return True

        except Exception as e:
            logger.error(f"Login failed: {e.response.status_code} - {e.response.text}")
            return False
        except Exception as e:
            logger.error(f"Login error: {e}")
            return False

    def logout(self):
        """Logout and clear credentials"""
        logger.info("Logging out")

        self.token = None
        self.api_key = None
        self.username = None

        # Clear saved credentials
        self._clear_credentials()

        logger.info("Logged out successfully")

    def refresh_token(self, registry_url: str = "https://hub.greenlang.io") -> bool:
        """
        Refresh authentication token

        Args:
            registry_url: Registry URL

        Returns:
            True if successful
        """
        if not self.token:
            logger.error("No token to refresh")
            return False

        try:
            logger.info("Refreshing token")

            with SecureHTTPSession() as session:
                response = session.post(
                    f"{registry_url}/api/v1/auth/refresh", headers=self.get_headers()
                )

            response.raise_for_status()

            data = response.json()
            self.token = data.get("token")

            # Save new token
            self._save_credentials()

            logger.info("Token refreshed successfully")
            return True

        except Exception as e:
            logger.error(f"Token refresh failed: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return False

    def register(
        self,
        username: str,
        email: str,
        password: str,
        registry_url: str = "https://hub.greenlang.io",
    ) -> bool:
        """
        Register new account

        Args:
            username: Desired username
            email: Email address
            password: Password
            registry_url: Registry URL

        Returns:
            True if successful
        """
        try:
            logger.info(f"Registering new account: {username}")

            with SecureHTTPSession() as session:
                response = session.post(
                    f"{registry_url}/api/v1/auth/register",
                    json={"username": username, "email": email, "password": password},
                )

            response.raise_for_status()

            logger.info("Registration successful")

            # Auto-login after registration
            return self.login(username, password, registry_url)

        except Exception as e:
            logger.error(
                f"Registration failed: {e.response.status_code} - {e.response.text}"
            )
            return False
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False

    def create_api_key(
        self, name: str, registry_url: str = "https://hub.greenlang.io"
    ) -> Optional[str]:
        """
        Create new API key

        Args:
            name: Key name/description
            registry_url: Registry URL

        Returns:
            API key if successful
        """
        if not self.token:
            logger.error("Authentication required to create API key")
            return None

        try:
            logger.info(f"Creating API key: {name}")

            with SecureHTTPSession() as session:
                response = session.post(
                    f"{registry_url}/api/v1/auth/api-keys",
                    headers=self.get_headers(),
                    json={"name": name},
                )

            response.raise_for_status()

            data = response.json()
            api_key = data.get("key")

            logger.info("API key created successfully")
            return api_key

        except Exception as e:
            logger.error(f"API key creation failed: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"API key creation error: {e}")
            return None

    def verify_token(self) -> bool:
        """
        Verify if current token is valid

        Returns:
            True if valid
        """
        if not self.token:
            return False

        try:
            # Decode JWT token (without verification for expiry check)
            # SECURITY FIX: Always verify JWT signature in production
            import os
            if os.getenv("GREENLANG_DEBUG_MODE") == "true":
                logger.warning("JWT signature verification disabled in DEBUG mode only!")
                payload = jwt.decode(self.token, options={"verify_signature": False})
            else:
                # Production mode - always verify signature
                secret_key = os.getenv("JWT_SECRET_KEY", "default-secret-key-change-in-production")
                payload = jwt.decode(self.token, secret_key, algorithms=["HS256"])

            # Check expiration
            exp = payload.get("exp")
            if exp:
                expiry = datetime.fromtimestamp(exp)
                if expiry < DeterministicClock.utcnow():
                    logger.warning("Token expired")
                    return False

            return True

        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return False

    def _save_credentials(self):
        """Save credentials securely"""
        try:
            # Use keyring for secure storage if available
            if KEYRING_AVAILABLE:
                if self.token:
                    keyring.set_password(self.KEYRING_SERVICE, "token", self.token)
                if self.username:
                    keyring.set_password(
                        self.KEYRING_SERVICE, "username", self.username
                    )
                if self.api_key:
                    keyring.set_password(self.KEYRING_SERVICE, "api_key", self.api_key)
            else:
                raise Exception("Keyring not available")

            logger.debug("Credentials saved securely")

        except Exception as e:
            logger.warning(f"Failed to save to keyring, using file: {e}")

            # Fallback to file storage
            creds_file = self.config_dir / "credentials.json"
            creds = {
                "username": self.username,
                "token": self.token,
                "api_key": self.api_key,
                "updated": DeterministicClock.utcnow().isoformat(),
            }

            with open(creds_file, "w") as f:
                json.dump(creds, f, indent=2)

            # Set restrictive permissions
            import os

            os.chmod(creds_file, 0o600)

    def _load_credentials(self):
        """Load saved credentials"""
        try:
            # Try keyring first
            if KEYRING_AVAILABLE:
                self.token = keyring.get_password(self.KEYRING_SERVICE, "token")
                self.username = keyring.get_password(self.KEYRING_SERVICE, "username")
                self.api_key = keyring.get_password(self.KEYRING_SERVICE, "api_key")
            else:
                raise Exception("Keyring not available")

            if self.token or self.api_key:
                logger.debug("Credentials loaded from keyring")
                return

        except Exception as e:
            logger.debug(f"Keyring not available: {e}")

        # Fallback to file
        creds_file = self.config_dir / "credentials.json"
        if creds_file.exists():
            try:
                with open(creds_file, "r") as f:
                    creds = json.load(f)

                self.username = creds.get("username")
                self.token = creds.get("token")
                self.api_key = creds.get("api_key")

                logger.debug("Credentials loaded from file")

            except Exception as e:
                logger.error(f"Failed to load credentials: {e}")

    def _clear_credentials(self):
        """Clear saved credentials"""
        try:
            # Clear from keyring
            if KEYRING_AVAILABLE:
                keyring.delete_password(self.KEYRING_SERVICE, "token")
                keyring.delete_password(self.KEYRING_SERVICE, "username")
                keyring.delete_password(self.KEYRING_SERVICE, "api_key")
        except:
            pass

        # Clear file
        creds_file = self.config_dir / "credentials.json"
        if creds_file.exists():
            creds_file.unlink()

        logger.debug("Credentials cleared")


class PackSigner:
    """Digital signature handler for packs"""

    def __init__(self, private_key_path: Optional[Path] = None):
        """
        Initialize pack signer

        Args:
            private_key_path: Path to private key file
        """
        self.private_key_path = private_key_path
        self.private_key = None
        self.public_key = None

        if private_key_path and private_key_path.exists():
            self._load_keys()

    def generate_keys(self, key_size: int = 2048) -> tuple:
        """
        Generate new RSA key pair

        Args:
            key_size: RSA key size

        Returns:
            Tuple of (private_key, public_key)
        """
        logger.info(f"Generating {key_size}-bit RSA key pair")

        self.private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size
        )
        self.public_key = self.private_key.public_key()

        return self.private_key, self.public_key

    def save_keys(self, private_path: Path, public_path: Optional[Path] = None):
        """
        Save key pair to files

        Args:
            private_path: Path for private key
            public_path: Path for public key (optional)
        """
        if not self.private_key:
            raise ValueError("No keys to save")

        # Save private key
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        private_path.parent.mkdir(parents=True, exist_ok=True)
        with open(private_path, "wb") as f:
            f.write(private_pem)

        # Set restrictive permissions
        import os

        os.chmod(private_path, 0o600)

        logger.info(f"Private key saved to {private_path}")

        # Save public key if path provided
        if public_path:
            public_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            with open(public_path, "wb") as f:
                f.write(public_pem)

            logger.info(f"Public key saved to {public_path}")

    def sign_pack(self, pack_data: bytes) -> Dict[str, Any]:
        """
        Sign pack data

        Args:
            pack_data: Pack content to sign

        Returns:
            Signature dictionary
        """
        if not self.private_key:
            raise ValueError("Private key not loaded")

        # Calculate hash
        digest = hashes.Hash(hashes.SHA256())
        digest.update(pack_data)
        pack_hash = digest.finalize()

        # Sign hash
        signature = self.private_key.sign(
            pack_hash,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )

        return {
            "algorithm": "RSA-PSS-SHA256",
            "signature": signature.hex(),
            "hash": pack_hash.hex(),
            "timestamp": DeterministicClock.utcnow().isoformat(),
        }

    def verify_signature(
        self,
        pack_data: bytes,
        signature: Dict[str, Any],
        public_key_pem: Optional[bytes] = None,
    ) -> bool:
        """
        Verify pack signature

        Args:
            pack_data: Pack content
            signature: Signature dictionary
            public_key_pem: Public key PEM (optional)

        Returns:
            True if valid
        """
        try:
            # Load public key if provided
            if public_key_pem:
                public_key = serialization.load_pem_public_key(public_key_pem)
            else:
                public_key = self.public_key

            if not public_key:
                logger.error("No public key available for verification")
                return False

            # Calculate hash
            digest = hashes.Hash(hashes.SHA256())
            digest.update(pack_data)
            pack_hash = digest.finalize()

            # Verify signature
            sig_bytes = bytes.fromhex(signature["signature"])
            public_key.verify(
                sig_bytes,
                pack_hash,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )

            logger.info("Signature verification successful")
            return True

        except Exception as e:
            logger.error(f"Signature verification failed: {e}")
            return False

    def _load_keys(self):
        """Load keys from file"""
        try:
            with open(self.private_key_path, "rb") as f:
                self.private_key = serialization.load_pem_private_key(
                    f.read(), password=None
                )
            self.public_key = self.private_key.public_key()

            logger.info("Keys loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            raise
