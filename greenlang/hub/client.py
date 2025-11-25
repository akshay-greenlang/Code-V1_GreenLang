# -*- coding: utf-8 -*-
"""
Hub Client for GreenLang Registry
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

from greenlang.security.http import SecureHTTPSession

from .archive import create_pack_archive, extract_pack_archive
from .manifest import load_manifest, save_manifest, validate_manifest
from .auth import HubAuth
from greenlang.determinism import DeterministicClock
from greenlang.intelligence import ChatMessage

# Import policy enforcer for security checks
try:
    from greenlang.policy.enforcer import check_install

    POLICY_ENABLED = True
except ImportError:
    logger.warning("Policy enforcer not available - running without policy checks")
    POLICY_ENABLED = False

    def check_install(*args, **kwargs):
        pass


logger = logging.getLogger(__name__)


class HubClient:
    """Client for interacting with GreenLang Hub Registry"""

    DEFAULT_REGISTRY_URL = "https://hub.greenlang.in"
    TIMEOUT = (10.0, 30.0)  # (connect_timeout, read_timeout)

    def __init__(
        self,
        registry_url: str = None,
        auth: Optional[HubAuth] = None,
        timeout: Optional[tuple] = None,
    ):
        """
        Initialize Hub Client

        Args:
            registry_url: Registry URL (defaults to hub.greenlang.io)
            auth: Authentication handler
            timeout: Request timeout settings as (connect_timeout, read_timeout)
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self.auth = auth
        self.timeout = timeout or self.TIMEOUT

        # Setup secure HTTP session
        self.session = SecureHTTPSession(
            timeout=self.timeout, max_retries=3, backoff_factor=0.3
        )

        # Set default headers
        self.default_headers = {
            "User-Agent": "GreenLang-Hub-Client/1.0",
            "Accept": "application/json",
        }

        # Apply authentication if provided
        if self.auth:
            self.default_headers.update(self.auth.get_headers())

        # Local cache directory
        self.cache_dir = Path.home() / ".greenlang" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Hub client initialized for {self.registry_url}")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def push(
        self,
        pack_path: Path,
        signature: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Push pack to registry

        Args:
            pack_path: Path to pack directory or archive
            signature: Optional digital signature
            tags: Optional tags for the pack
            description: Optional description

        Returns:
            Response data from registry
        """
        pack_path = Path(pack_path)

        if not pack_path.exists():
            raise FileNotFoundError(f"Pack not found: {pack_path}")

        # Load and validate manifest
        manifest = load_manifest(pack_path)
        validate_manifest(manifest)

        # Policy check BEFORE pushing
        if POLICY_ENABLED:
            # Add metadata for policy check
            manifest_dict = manifest.dict() if hasattr(manifest, "dict") else manifest
            manifest_dict["signature_verified"] = signature is not None
            manifest_dict["publisher"] = manifest_dict.get("publisher", "unknown")

            try:
                check_install(
                    manifest_dict,
                    str(pack_path),
                    stage="publish",
                    permissive=False,  # Strict for publishing
                )
                logger.info("✓ Pack passed policy checks for publishing")
            except RuntimeError as e:
                logger.error(f"Policy check failed: {e}")
                raise ValueError(f"Publishing blocked by policy: {e}")

        # Create archive if directory
        if pack_path.is_dir():
            logger.info(f"Creating archive for {pack_path}")
            archive_path = create_pack_archive(pack_path)
        else:
            archive_path = pack_path

        # Prepare upload data
        upload_data = {
            "manifest": json.dumps(manifest.dict()),
            "timestamp": DeterministicClock.utcnow().isoformat(),
        }

        if signature:
            upload_data["signature"] = json.dumps(signature)

        if tags:
            upload_data["tags"] = json.dumps(tags)

        if description:
            upload_data["description"] = description

        # Upload pack
        try:
            with open(archive_path, "rb") as f:
                files = {"pack": (archive_path.name, f, "application/x-tar")}

                logger.info(f"Pushing {manifest.get('name', 'pack')} to registry")
                # Note: files parameter handling needs adjustment for requests library
                response = self.session.post(
                    f"{self.registry_url}/api/v1/packs",
                    files=files,
                    data=upload_data,
                    headers=self.default_headers,
                )

                result = response.json()
                logger.info(f"Successfully pushed pack: {result.get('id', 'unknown')}")

                return result

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"Failed to push pack: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"Error pushing pack: {e}")
            raise
        finally:
            # Cleanup temporary archive if created
            if pack_path.is_dir() and archive_path.exists():
                archive_path.unlink()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def pull(
        self,
        pack_ref: str,
        output_dir: Optional[Path] = None,
        verify_signature: bool = True,
    ) -> Path:
        """
        Pull pack from registry

        Args:
            pack_ref: Pack reference (name@version or ID)
            output_dir: Output directory (defaults to ~/.greenlang/packs)
            verify_signature: Whether to verify pack signature

        Returns:
            Path to extracted pack
        """
        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
        else:
            output_dir = Path.home() / ".greenlang" / "packs"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Check cache first
        cached_path = self._check_cache(pack_ref)
        if cached_path and cached_path.exists():
            logger.info(f"Using cached pack: {pack_ref}")
            return cached_path

        try:
            # Download pack
            logger.info(f"Pulling pack: {pack_ref}")
            response = self.session.get(
                f"{self.registry_url}/api/v1/packs/{pack_ref}",
                headers=self.default_headers,
            )

            # Parse response
            pack_data = response.json()
            pack_content = pack_data.get("content")
            manifest = pack_data.get("manifest")
            signature = pack_data.get("signature")

            if not pack_content:
                # Download binary content separately
                download_url = pack_data.get("download_url")
                if download_url:
                    content_response = self.session.get(
                        download_url, headers=self.default_headers
                    )
                    pack_content = content_response.content
                else:
                    raise ValueError("No pack content or download URL provided")

            # Verify signature if required
            signature_verified = False
            if verify_signature and signature:
                signature_verified = self._verify_signature(pack_content, signature)
                if not signature_verified:
                    raise ValueError("Signature verification failed")
            elif not verify_signature:
                logger.warning(
                    "⚠️  Skipping signature verification (--no-verify flag used)"
                )

            # Policy check BEFORE extraction
            if POLICY_ENABLED and manifest:
                # Add metadata for policy check
                manifest["signature_verified"] = signature_verified
                manifest["publisher"] = manifest.get(
                    "publisher", pack_data.get("publisher", "unknown")
                )

                try:
                    check_install(
                        manifest,
                        str(output_dir),
                        stage="add",
                        permissive=False,  # Always strict - no permissive mode allowed
                    )
                    logger.info("✓ Pack passed policy checks")
                except RuntimeError as e:
                    logger.error(f"Policy check failed: {e}")
                    raise ValueError(f"Installation blocked by policy: {e}")

            # Extract pack
            pack_name = manifest.get("name", pack_ref.replace("@", "_"))
            pack_dir = output_dir / pack_name

            logger.info(f"Extracting pack to {pack_dir}")
            extract_pack_archive(pack_content, pack_dir)

            # Save manifest
            if manifest:
                save_manifest(pack_dir, manifest)

            # Cache the pack
            self._cache_pack(pack_ref, pack_dir)

            logger.info(f"Successfully pulled pack to {pack_dir}")
            return pack_dir

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"Failed to pull pack: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"Failed to pull pack: {e}")
            raise
        except Exception as e:
            logger.error(f"Error pulling pack: {e}")
            raise

    def search(
        self,
        query: str = None,
        tags: Optional[List[str]] = None,
        author: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Search for packs in registry

        Args:
            query: Search query string
            tags: Filter by tags
            author: Filter by author
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of pack metadata
        """
        params = {"limit": limit, "offset": offset}

        if query:
            params["q"] = query

        if tags:
            params["tags"] = ",".join(tags)

        if author:
            params["author"] = author

        try:
            logger.info(f"Searching registry with query: {query}")
            response = self.session.get(
                f"{self.registry_url}/api/v1/packs/search",
                params=params,
                headers=self.default_headers,
            )

            results = response.json()
            logger.info(f"Found {len(results)} packs")

            return results

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"Search failed: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"Search failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error searching packs: {e}")
            raise

    def list_packs(
        self, user: Optional[str] = None, limit: int = 50, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List packs from registry

        Args:
            user: Filter by user/organization
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of pack metadata
        """
        params = {"limit": limit, "offset": offset}

        if user:
            params["user"] = user

        try:
            logger.info("Listing packs from registry")
            response = self.session.get(
                f"{self.registry_url}/api/v1/packs",
                params=params,
                headers=self.default_headers,
            )

            packs = response.json()
            logger.info(f"Listed {len(packs)} packs")

            return packs

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"List failed: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"List failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Error listing packs: {e}")
            raise

    def get_pack_info(self, pack_ref: str) -> Dict[str, Any]:
        """
        Get detailed information about a pack

        Args:
            pack_ref: Pack reference (name@version or ID)

        Returns:
            Pack metadata and details
        """
        try:
            logger.info(f"Getting info for pack: {pack_ref}")
            response = self.session.get(
                f"{self.registry_url}/api/v1/packs/{pack_ref}/info",
                headers=self.default_headers,
            )

            return response.json()

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"Failed to get pack info: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"Failed to get pack info: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting pack info: {e}")
            raise

    def delete_pack(self, pack_ref: str) -> bool:
        """
        Delete a pack from registry (requires authentication)

        Args:
            pack_ref: Pack reference to delete

        Returns:
            True if successful
        """
        if not self.auth:
            raise ValueError("Authentication required to delete packs")

        try:
            logger.info(f"Deleting pack: {pack_ref}")
            response = self.session.delete(
                f"{self.registry_url}/api/v1/packs/{pack_ref}",
                headers=self.default_headers,
            )

            logger.info(f"Successfully deleted pack: {pack_ref}")
            return True

        except Exception as e:
            if hasattr(e, "response"):
                logger.error(
                    f"Failed to delete pack: {e.response.status_code} - {e.response.text}"
                )
            else:
                logger.error(f"Failed to delete pack: {e}")
            raise
        except Exception as e:
            logger.error(f"Error deleting pack: {e}")
            raise

    def _check_cache(self, pack_ref: str) -> Optional[Path]:
        """Check if pack exists in local cache"""
        cache_key = pack_ref.replace("@", "_").replace("/", "_")
        cached_path = self.cache_dir / cache_key

        if cached_path.exists():
            # Check if cache is still valid (e.g., not too old)
            manifest_path = cached_path / "manifest.json"
            if manifest_path.exists():
                return cached_path

        return None

    def _cache_pack(self, pack_ref: str, pack_path: Path):
        """Cache pack locally"""
        cache_key = pack_ref.replace("@", "_").replace("/", "_")
        cache_path = self.cache_dir / cache_key

        if cache_path.exists():
            import shutil

            shutil.rmtree(cache_path)

        # Copy to cache
        import shutil

        shutil.copytree(pack_path, cache_path)

        logger.debug(f"Cached pack {pack_ref} at {cache_path}")

    def _verify_signature(self, content: bytes, signature: Dict) -> bool:
        """
        Verify pack signature using cryptographic verification.

        Supports multiple signature algorithms:
        - Ed25519: EdDSA signature scheme
        - RSA-PSS: RSA with PSS padding (RSA-2048, RSA-4096)
        - ECDSA: Elliptic Curve Digital Signature Algorithm (P-256, P-384)
        - Sigstore: Keyless verification with certificate chain

        Args:
            content: The pack content bytes to verify
            signature: Signature dictionary containing:
                - spec.signature.algorithm: Algorithm used (ed25519, rsa-pss, ecdsa, sigstore-keyless)
                - spec.signature.value: Base64-encoded signature bytes
                - spec.publicKey: PEM-encoded public key (for non-Sigstore)
                - spec.certChain: Base64-encoded certificate chain (for Sigstore)
                - metadata.hash.value: Expected SHA-256 hash of content (optional)

        Returns:
            True if signature is valid

        Raises:
            ValueError: If signature verification fails or signature format is invalid
        """
        import base64
        import hashlib

        # Try to import verification modules
        try:
            from greenlang.security.signing import (
                DetachedSigVerifier,
                SigstoreBundleVerifier,
                CRYPTO_AVAILABLE,
                SIGSTORE_AVAILABLE,
            )
        except ImportError:
            logger.error("Security signing module not available")
            raise ValueError("Security module not available for signature verification")

        # Validate signature structure
        if not isinstance(signature, dict):
            logger.error("Invalid signature format: not a dictionary")
            raise ValueError("Invalid signature format: expected dictionary")

        # Handle different signature formats (v1.0 and v2.0)
        sig_spec = signature.get("spec", signature)
        sig_info = sig_spec.get("signature", sig_spec)

        # Extract algorithm
        algorithm = sig_info.get("algorithm")
        if not algorithm:
            logger.error("No algorithm specified in signature")
            raise ValueError("No algorithm specified in signature")

        # Extract signature value
        sig_value = sig_info.get("value")
        if not sig_value:
            logger.error("No signature value found")
            raise ValueError("No signature value found in signature")

        try:
            sig_bytes = base64.b64decode(sig_value)
        except Exception as e:
            logger.error(f"Failed to decode signature: {e}")
            raise ValueError(f"Invalid base64 signature value: {e}")

        # Verify content hash if provided (additional integrity check)
        metadata = signature.get("metadata", {})
        hash_info = metadata.get("hash", {})
        expected_hash = hash_info.get("value")

        if expected_hash:
            actual_hash = hashlib.sha256(content).hexdigest()
            if actual_hash != expected_hash:
                logger.error(
                    f"Content hash mismatch: expected {expected_hash}, got {actual_hash}"
                )
                raise ValueError("Content hash verification failed")
            logger.debug("Content hash verified successfully")

        # Verify signature based on algorithm
        try:
            if algorithm == "sigstore-keyless":
                # Sigstore keyless verification with certificate chain
                if not SIGSTORE_AVAILABLE:
                    logger.error("Sigstore library not available for keyless verification")
                    raise ValueError(
                        "Sigstore not available. Install with: pip install sigstore"
                    )

                cert_chain_b64 = sig_spec.get("certChain")
                if not cert_chain_b64:
                    raise ValueError("Certificate chain required for Sigstore verification")

                cert_chain = base64.b64decode(cert_chain_b64)
                transparency_entry = sig_spec.get("transparencyLog")

                verifier = SigstoreBundleVerifier(staging=False)
                verifier.verify(
                    content,
                    sig_bytes,
                    cert_chain=cert_chain,
                    transparency_entry=transparency_entry,
                )
                logger.info("Sigstore signature verified successfully")

            elif algorithm in ("ed25519", "rsa-pss", "ecdsa", "ecdsa-sha256"):
                # Detached signature verification with public key
                if not CRYPTO_AVAILABLE:
                    logger.error("Cryptography library not available")
                    raise ValueError(
                        "Cryptography not available. Install with: pip install cryptography"
                    )

                public_key_pem = sig_spec.get("publicKey")
                if not public_key_pem:
                    # Try to fetch public key from hub if key_id is provided
                    key_id = sig_spec.get("keyId") or metadata.get("signer", {}).get(
                        "key_fingerprint"
                    )
                    if key_id:
                        public_key_pem = self._fetch_public_key_from_hub(key_id)
                    else:
                        raise ValueError(
                            "Public key required for detached signature verification"
                        )

                verifier = DetachedSigVerifier()
                verifier.verify(
                    content, sig_bytes, public_key=public_key_pem, algorithm=algorithm
                )
                logger.info(f"{algorithm.upper()} signature verified successfully")

            else:
                logger.error(f"Unsupported signature algorithm: {algorithm}")
                raise ValueError(f"Unsupported signature algorithm: {algorithm}")

            return True

        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            # Handle cryptography InvalidSignature and other errors
            error_type = type(e).__name__
            if error_type == "InvalidSignature":
                logger.error("Signature verification failed: invalid signature")
                raise ValueError("Signature verification failed: invalid signature")
            else:
                logger.error(f"Signature verification error: {e}")
                raise ValueError(f"Signature verification failed: {e}")

    def _fetch_public_key_from_hub(self, key_id: str) -> str:
        """
        Fetch public key from hub registry for verification.

        Args:
            key_id: Key identifier (fingerprint or key ID)

        Returns:
            PEM-encoded public key string

        Raises:
            ValueError: If key cannot be fetched
        """
        try:
            logger.debug(f"Fetching public key {key_id} from hub")
            response = self.session.get(
                f"{self.registry_url}/api/v1/keys/{key_id}",
                headers=self.default_headers,
            )

            key_data = response.json()
            public_key = key_data.get("public_key")

            if not public_key:
                raise ValueError(f"No public key returned for key_id: {key_id}")

            logger.debug(f"Successfully fetched public key {key_id}")
            return public_key

        except Exception as e:
            logger.error(f"Failed to fetch public key {key_id}: {e}")
            raise ValueError(f"Failed to fetch public key from hub: {e}")

    def close(self):
        """Close HTTP client"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
