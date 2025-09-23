"""
Native OCI Registry Client for GreenLang

This module provides a native Python implementation for interacting with
OCI (Open Container Initiative) compliant registries without external dependencies.
"""

import json
import hashlib
import base64
import tarfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import urllib.parse
import urllib.request
import urllib.error
import ssl
import re

logger = logging.getLogger(__name__)


@dataclass
class OCIManifest:
    """OCI Image Manifest representation"""

    schema_version: int = 2
    media_type: str = "application/vnd.oci.image.manifest.v1+json"
    config: Dict[str, Any] = field(default_factory=dict)
    layers: List[Dict[str, Any]] = field(default_factory=list)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert manifest to JSON"""
        return json.dumps(
            {
                "schemaVersion": self.schema_version,
                "mediaType": self.media_type,
                "config": self.config,
                "layers": self.layers,
                "annotations": self.annotations,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, data: str) -> "OCIManifest":
        """Create manifest from JSON"""
        obj = json.loads(data)
        return cls(
            schema_version=obj.get("schemaVersion", 2),
            media_type=obj.get(
                "mediaType", "application/vnd.oci.image.manifest.v1+json"
            ),
            config=obj.get("config", {}),
            layers=obj.get("layers", []),
            annotations=obj.get("annotations", {}),
        )


@dataclass
class OCIDescriptor:
    """OCI Content Descriptor"""

    media_type: str
    digest: str
    size: int
    urls: Optional[List[str]] = None
    annotations: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert descriptor to dictionary"""
        result = {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
        }
        if self.urls:
            result["urls"] = self.urls
        if self.annotations:
            result["annotations"] = self.annotations
        return result


class OCIAuth:
    """Handle OCI registry authentication"""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.username = username
        self.password = password
        self.token = token
        self._auth_cache: Dict[str, str] = {}

    def get_auth_header(
        self, registry: str, scope: Optional[str] = None
    ) -> Optional[str]:
        """Get authentication header for registry"""
        if self.token:
            return f"Bearer {self.token}"

        if self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            return f"Basic {encoded}"

        # Check for cached bearer token
        if registry in self._auth_cache:
            return f"Bearer {self._auth_cache[registry]}"

        return None

    def authenticate(self, registry: str, www_authenticate: str) -> bool:
        """Handle WWW-Authenticate challenge"""
        # Parse WWW-Authenticate header
        if www_authenticate.startswith("Bearer "):
            # Extract realm, service, and scope
            params = self._parse_www_authenticate(www_authenticate)
            if "realm" in params:
                # Request bearer token
                token = self._request_bearer_token(
                    params["realm"], params.get("service"), params.get("scope")
                )
                if token:
                    self._auth_cache[registry] = token
                    return True

        return False

    def _parse_www_authenticate(self, header: str) -> Dict[str, str]:
        """Parse WWW-Authenticate header"""
        params = {}
        # Remove "Bearer " prefix
        header = header[7:]

        # Parse key="value" pairs
        pattern = r'(\w+)="([^"]*)"'
        matches = re.findall(pattern, header)
        for key, value in matches:
            params[key] = value

        return params

    def _request_bearer_token(
        self, realm: str, service: Optional[str], scope: Optional[str]
    ) -> Optional[str]:
        """Request bearer token from auth endpoint"""
        # Build token request URL
        params = {}
        if service:
            params["service"] = service
        if scope:
            params["scope"] = scope

        if self.username and self.password:
            params["account"] = self.username

        url = f"{realm}?{urllib.parse.urlencode(params)}"

        # Make token request
        headers = {}
        if self.username and self.password:
            credentials = f"{self.username}:{self.password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read())
                return data.get("token") or data.get("access_token")
        except Exception:
            return None


class OCIClient:
    """
    Native OCI registry client for push/pull operations.

    This implementation provides basic OCI registry operations without
    requiring external tools like ORAS or Docker.
    """

    def __init__(
        self,
        registry: str = "ghcr.io",
        auth: Optional[OCIAuth] = None,
        insecure: bool = False,
        insecure_transport: bool = False,
    ):
        """
        Initialize OCI client.

        Args:
            registry: Registry URL (default: ghcr.io)
            auth: Authentication credentials
            insecure: Allow insecure HTTPS connections (DANGEROUS - dev only)
            insecure_transport: Allow HTTP instead of HTTPS (DANGEROUS - dev only)
        """
        self.registry = registry.rstrip("/")
        self.auth = auth or OCIAuth()
        self.insecure = insecure
        self.insecure_transport = insecure_transport

        # SECURITY: Enforce HTTPS - HTTP is never allowed
        if self.registry.startswith("http://"):
            raise ValueError(
                "SECURITY: HTTP registries are not allowed. "
                "Use HTTPS for all registry connections."
            )
        elif not self.registry.startswith("https://") and "://" not in self.registry:
            # Prepend https:// if no protocol specified
            self.registry = f"https://{self.registry}"

        # Create SSL context - always verify certificates
        self.ssl_context = ssl.create_default_context()
        if insecure:
            raise ValueError(
                "SECURITY: Insecure TLS is not allowed. "
                "SSL/TLS verification is mandatory for all connections."
            )

    def _make_request(
        self,
        method: str,
        url: str,
        data: Optional[bytes] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, bytes, Dict[str, str]]:
        """Make HTTP request to registry"""
        headers = headers or {}

        # Add authentication if available
        auth_header = self.auth.get_auth_header(self.registry)
        if auth_header:
            headers["Authorization"] = auth_header

        # Create request
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            # Make request
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                return response.code, response.read(), dict(response.headers)
        except urllib.error.HTTPError as e:
            # Handle authentication challenge
            if e.code == 401 and "WWW-Authenticate" in e.headers:
                if self.auth.authenticate(self.registry, e.headers["WWW-Authenticate"]):
                    # Retry with new auth
                    auth_header = self.auth.get_auth_header(self.registry)
                    if auth_header:
                        headers["Authorization"] = auth_header
                        req = urllib.request.Request(
                            url, data=data, headers=headers, method=method
                        )
                        with urllib.request.urlopen(
                            req, context=self.ssl_context
                        ) as response:
                            return (
                                response.code,
                                response.read(),
                                dict(response.headers),
                            )

            return e.code, e.read(), dict(e.headers)

    def _calculate_digest(self, data: bytes) -> str:
        """Calculate SHA256 digest"""
        return f"sha256:{hashlib.sha256(data).hexdigest()}"

    def push_blob(
        self,
        namespace: str,
        name: str,
        data: bytes,
        media_type: str = "application/octet-stream",
    ) -> OCIDescriptor:
        """
        Push a blob to the registry.

        Args:
            namespace: Repository namespace
            name: Repository name
            data: Blob data
            media_type: Content media type

        Returns:
            OCIDescriptor for the pushed blob
        """
        digest = self._calculate_digest(data)
        size = len(data)

        # Check if blob exists
        url = f"https://{self.registry}/v2/{namespace}/{name}/blobs/{digest}"
        code, _, _ = self._make_request("HEAD", url)

        if code == 200:
            # Blob already exists
            return OCIDescriptor(media_type=media_type, digest=digest, size=size)

        # Initiate upload
        url = f"https://{self.registry}/v2/{namespace}/{name}/blobs/uploads/"
        code, _, headers = self._make_request("POST", url)

        if code not in [202, 201]:
            raise RuntimeError(f"Failed to initiate blob upload: {code}")

        # Get upload URL
        upload_url = headers.get("Location", "")
        if not upload_url:
            raise RuntimeError("No upload URL provided")

        # Complete upload
        if not upload_url.startswith("http"):
            upload_url = f"https://{self.registry}{upload_url}"

        upload_url = f"{upload_url}&digest={digest}"

        headers = {"Content-Type": media_type, "Content-Length": str(size)}

        code, _, _ = self._make_request("PUT", upload_url, data=data, headers=headers)

        if code not in [201, 204]:
            raise RuntimeError(f"Failed to upload blob: {code}")

        return OCIDescriptor(media_type=media_type, digest=digest, size=size)

    def pull_blob(self, namespace: str, name: str, digest: str) -> bytes:
        """
        Pull a blob from the registry.

        Args:
            namespace: Repository namespace
            name: Repository name
            digest: Blob digest

        Returns:
            Blob data
        """
        url = f"https://{self.registry}/v2/{namespace}/{name}/blobs/{digest}"
        code, data, _ = self._make_request("GET", url)

        if code != 200:
            raise RuntimeError(f"Failed to pull blob: {code}")

        return data

    def push_manifest(
        self, namespace: str, name: str, tag: str, manifest: OCIManifest
    ) -> str:
        """
        Push a manifest to the registry.

        Args:
            namespace: Repository namespace
            name: Repository name
            tag: Image tag
            manifest: OCI manifest

        Returns:
            Manifest digest
        """
        manifest_json = manifest.to_json().encode()
        digest = self._calculate_digest(manifest_json)

        url = f"https://{self.registry}/v2/{namespace}/{name}/manifests/{tag}"
        headers = {
            "Content-Type": manifest.media_type,
            "Content-Length": str(len(manifest_json)),
        }

        code, _, _ = self._make_request("PUT", url, data=manifest_json, headers=headers)

        if code not in [201, 204]:
            raise RuntimeError(f"Failed to push manifest: {code}")

        return digest

    def pull_manifest(self, namespace: str, name: str, reference: str) -> OCIManifest:
        """
        Pull a manifest from the registry.

        Args:
            namespace: Repository namespace
            name: Repository name
            reference: Tag or digest

        Returns:
            OCI manifest
        """
        url = f"https://{self.registry}/v2/{namespace}/{name}/manifests/{reference}"
        headers = {"Accept": "application/vnd.oci.image.manifest.v1+json"}

        code, data, _ = self._make_request("GET", url, headers=headers)

        if code != 200:
            raise RuntimeError(f"Failed to pull manifest: {code}")

        return OCIManifest.from_json(data.decode())

    def push_pack(
        self,
        namespace: str,
        name: str,
        tag: str,
        pack_path: Path,
        annotations: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Push a GreenLang pack to the registry.

        Args:
            namespace: Repository namespace
            name: Pack name
            tag: Pack version
            pack_path: Path to pack directory
            annotations: OCI annotations

        Returns:
            Manifest digest
        """
        # Create tar archive of pack
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            with tarfile.open(tmp.name, "w:gz") as tar:
                tar.add(pack_path, arcname=".")

            tmp.seek(0)
            pack_data = tmp.read()

        # Push pack as blob
        pack_descriptor = self.push_blob(
            namespace,
            name,
            pack_data,
            media_type="application/vnd.greenlang.pack.v1.tar+gzip",
        )

        # Create config
        config = {
            "architecture": "unknown",
            "os": "unknown",
            "rootfs": {"type": "layers", "diff_ids": []},
            "config": {"Labels": annotations or {}},
        }
        config_json = json.dumps(config).encode()

        # Push config blob
        config_descriptor = self.push_blob(
            namespace,
            name,
            config_json,
            media_type="application/vnd.oci.image.config.v1+json",
        )

        # Create manifest
        manifest = OCIManifest(
            config={
                "mediaType": config_descriptor.media_type,
                "digest": config_descriptor.digest,
                "size": config_descriptor.size,
            },
            layers=[pack_descriptor.to_dict()],
            annotations=annotations or {},
        )

        # Push manifest
        return self.push_manifest(namespace, name, tag, manifest)

    def pull_pack(
        self, namespace: str, name: str, reference: str, output_dir: Path
    ) -> Path:
        """
        Pull a GreenLang pack from the registry.

        Args:
            namespace: Repository namespace
            name: Pack name
            reference: Tag or digest
            output_dir: Output directory

        Returns:
            Path to extracted pack
        """
        # Pull manifest
        manifest = self.pull_manifest(namespace, name, reference)

        # Pull pack layer
        if not manifest.layers:
            raise RuntimeError("No layers in manifest")

        layer = manifest.layers[0]
        pack_data = self.pull_blob(namespace, name, layer["digest"])

        # Extract pack
        output_dir.mkdir(parents=True, exist_ok=True)
        pack_dir = output_dir / name

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp.write(pack_data)
            tmp.flush()

            with tarfile.open(tmp.name, "r:gz") as tar:
                tar.extractall(pack_dir)

        return pack_dir

    def list_tags(self, namespace: str, name: str) -> List[str]:
        """
        List tags for a repository.

        Args:
            namespace: Repository namespace
            name: Repository name

        Returns:
            List of tags
        """
        url = f"https://{self.registry}/v2/{namespace}/{name}/tags/list"
        code, data, _ = self._make_request("GET", url)

        if code != 200:
            raise RuntimeError(f"Failed to list tags: {code}")

        result = json.loads(data)
        return result.get("tags", [])

    def delete_manifest(self, namespace: str, name: str, digest: str) -> bool:
        """
        Delete a manifest from the registry.

        Args:
            namespace: Repository namespace
            name: Repository name
            digest: Manifest digest

        Returns:
            True if successful
        """
        url = f"https://{self.registry}/v2/{namespace}/{name}/manifests/{digest}"
        code, _, _ = self._make_request("DELETE", url)

        return code in [202, 204]


def create_client(
    registry: str = "ghcr.io",
    username: Optional[str] = None,
    password: Optional[str] = None,
    token: Optional[str] = None,
    insecure: bool = False,
    insecure_transport: bool = False,
) -> OCIClient:
    """
    Create an OCI client with authentication.

    Args:
        registry: Registry URL (HTTPS enforced by default)
        username: Username for basic auth
        password: Password for basic auth
        token: Bearer token
        insecure: Allow insecure TLS/SSL (requires GL_DEBUG_INSECURE=1)
        insecure_transport: Allow HTTP transport (requires GL_DEBUG_INSECURE=1)

    Returns:
        Configured OCIClient
    """
    auth = OCIAuth(username=username, password=password, token=token)
    return OCIClient(
        registry=registry,
        auth=auth,
        insecure=insecure,
        insecure_transport=insecure_transport,
    )
