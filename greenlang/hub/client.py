"""
Hub Client for GreenLang Registry
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .archive import create_pack_archive, extract_pack_archive
from .manifest import load_manifest, save_manifest, validate_manifest
from .auth import HubAuth

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

    DEFAULT_REGISTRY_URL = "https://hub.greenlang.io"
    TIMEOUT = httpx.Timeout(30.0, connect=10.0)

    def __init__(
        self,
        registry_url: str = None,
        auth: Optional[HubAuth] = None,
        timeout: Optional[httpx.Timeout] = None,
    ):
        """
        Initialize Hub Client

        Args:
            registry_url: Registry URL (defaults to hub.greenlang.io)
            auth: Authentication handler
            timeout: Request timeout settings
        """
        self.registry_url = registry_url or self.DEFAULT_REGISTRY_URL
        self.auth = auth
        self.timeout = timeout or self.TIMEOUT

        # Setup HTTP client with retry logic
        self.session = httpx.Client(
            base_url=self.registry_url,
            timeout=self.timeout,
            headers={
                "User-Agent": "GreenLang-Hub-Client/1.0",
                "Accept": "application/json",
            },
        )

        # Apply authentication if provided
        if self.auth:
            self.session.headers.update(self.auth.get_headers())

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
            "timestamp": datetime.utcnow().isoformat(),
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
                response = self.session.post(
                    "/api/v1/packs", files=files, data=upload_data
                )

                response.raise_for_status()

                result = response.json()
                logger.info(f"Successfully pushed pack: {result.get('id', 'unknown')}")

                return result

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to push pack: {e.response.status_code} - {e.response.text}"
            )
            raise
        except Exception as e:
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
            response = self.session.get(f"/api/v1/packs/{pack_ref}")
            response.raise_for_status()

            # Parse response
            pack_data = response.json()
            pack_content = pack_data.get("content")
            manifest = pack_data.get("manifest")
            signature = pack_data.get("signature")

            if not pack_content:
                # Download binary content separately
                download_url = pack_data.get("download_url")
                if download_url:
                    content_response = self.session.get(download_url)
                    content_response.raise_for_status()
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
                        permissive=not verify_signature,  # Permissive if --no-verify
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

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to pull pack: {e.response.status_code} - {e.response.text}"
            )
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
            response = self.session.get("/api/v1/packs/search", params=params)
            response.raise_for_status()

            results = response.json()
            logger.info(f"Found {len(results)} packs")

            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"Search failed: {e.response.status_code} - {e.response.text}")
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
            response = self.session.get("/api/v1/packs", params=params)
            response.raise_for_status()

            packs = response.json()
            logger.info(f"Listed {len(packs)} packs")

            return packs

        except httpx.HTTPStatusError as e:
            logger.error(f"List failed: {e.response.status_code} - {e.response.text}")
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
            response = self.session.get(f"/api/v1/packs/{pack_ref}/info")
            response.raise_for_status()

            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to get pack info: {e.response.status_code} - {e.response.text}"
            )
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
            response = self.session.delete(f"/api/v1/packs/{pack_ref}")
            response.raise_for_status()

            logger.info(f"Successfully deleted pack: {pack_ref}")
            return True

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to delete pack: {e.response.status_code} - {e.response.text}"
            )
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
        """Verify pack signature"""
        # TODO: Implement actual signature verification
        # This would use cryptographic libraries to verify the signature
        logger.warning("Signature verification not yet implemented")
        return True

    def close(self):
        """Close HTTP client"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
