# -*- coding: utf-8 -*-
"""
Pack Installer
==============

Handles installation of packs from various sources:
- PyPI (pip install)
- Local directories
- GitHub repositories
- GreenLang Hub
"""

import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
from urllib.parse import urlparse
import requests

from .manifest import PackManifest
from .registry import PackRegistry, InstalledPack
from ..security import (
    create_secure_session,
    validate_url,
    validate_git_url,
    safe_download,
    safe_extract_archive,
    validate_pack_structure,
    PackVerifier,
    SignatureVerificationError,
)

logger = logging.getLogger(__name__)


class PackInstaller:
    """
    Installs packs from various sources
    """

    def __init__(self, registry: Optional[PackRegistry] = None):
        """
        Initialize installer

        Args:
            registry: Pack registry (creates new if not provided)
        """
        self.registry = registry or PackRegistry()
        self.cache_dir = Path.home() / ".greenlang" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = create_secure_session()
        self.verifier = PackVerifier()

    def install_pack(
        self,
        path: str,
        allow_unsigned: bool = False,
        force: bool = False,
    ) -> InstalledPack:
        """
        Install a pack with security validation

        Args:
            path: Path to pack directory or archive
            allow_unsigned: Allow unsigned packs (dev mode only)
            force: Force reinstall if already exists

        Returns:
            InstalledPack metadata

        Raises:
            SignatureVerificationError: If pack is unsigned and allow_unsigned is False
            ValueError: If pack capabilities are invalid
        """
        # Convert to Path
        pack_path = Path(path)

        # Check signature unless explicitly allowed
        verify = not allow_unsigned

        if not allow_unsigned:
            logger.info("Verifying pack signature (signed-only mode)")
        else:
            logger.warning("⚠️  SECURITY WARNING: Installing unsigned pack (dev mode)")

        # Install based on path type
        if pack_path.is_dir():
            installed = self._install_from_local(pack_path, verify=verify)
        elif pack_path.is_file():
            installed = self._install_from_archive(pack_path, verify=verify)
        else:
            # Try as a source string (PyPI, GitHub, etc.)
            installed = self.install(str(path), verify=verify, force=force)

        # Validate capabilities
        if installed and installed.path:
            manifest_path = Path(installed.path) / "pack.yaml"
            if manifest_path.exists():
                manifest = PackManifest.from_yaml(Path(installed.path))
                if manifest.capabilities:
                    issues = self._validate_capabilities(manifest.capabilities)
                    if issues:
                        logger.warning(f"Capability validation issues: {issues}")
                        if not allow_unsigned:  # Strict mode
                            # Uninstall the pack
                            self.uninstall(installed.name)
                            raise ValueError(f"Pack capabilities validation failed: {issues}")

        return installed

    def _validate_capabilities(self, capabilities) -> List[str]:
        """
        Validate pack capabilities for security

        Args:
            capabilities: Capabilities object from manifest

        Returns:
            List of validation issues (empty if valid)
        """
        issues = []

        # Check subprocess capabilities
        if hasattr(capabilities, 'subprocess') and capabilities.subprocess:
            if capabilities.subprocess.allow:
                # Check for dangerous binaries
                dangerous_binaries = [
                    '/bin/sh', '/bin/bash', '/usr/bin/python', '/usr/bin/perl',
                    '/usr/bin/ruby', '/usr/bin/sudo', 'sudo', 'sh', 'bash',
                    '/usr/bin/curl', 'curl', 'wget', '/usr/bin/wget'
                ]
                for binary in capabilities.subprocess.allowlist:
                    if binary in dangerous_binaries:
                        issues.append(f"Dangerous binary in allowlist: {binary}")

        # Check filesystem capabilities
        if hasattr(capabilities, 'fs') and capabilities.fs:
            if capabilities.fs.allow:
                # Check for root filesystem access
                dangerous_paths = ['/', '/*', '/**', '/etc', '/etc/*', '/root', '/root/*']

                if hasattr(capabilities.fs, 'write'):
                    if isinstance(capabilities.fs.write, dict) and 'allowlist' in capabilities.fs.write:
                        for path in capabilities.fs.write['allowlist']:
                            if path in dangerous_paths:
                                issues.append("Root filesystem write access is not allowed")
                            if '..' in path:
                                issues.append(f"Path traversal detected: {path}")

                if hasattr(capabilities.fs, 'read'):
                    if isinstance(capabilities.fs.read, dict) and 'allowlist' in capabilities.fs.read:
                        for path in capabilities.fs.read['allowlist']:
                            if '..' in path:
                                issues.append(f"Path traversal detected: {path}")

                # Check write_paths and read_paths
                if hasattr(capabilities.fs, 'write_paths'):
                    for path in capabilities.fs.write_paths:
                        if path in dangerous_paths:
                            issues.append("Root filesystem write access is not allowed")
                        if '..' in path:
                            issues.append(f"Path traversal detected: {path}")

                if hasattr(capabilities.fs, 'read_paths'):
                    for path in capabilities.fs.read_paths:
                        if '..' in path:
                            issues.append(f"Path traversal detected: {path}")

        # Check network capabilities
        if hasattr(capabilities, 'net') and capabilities.net:
            if capabilities.net.allow:
                # Check egress allowlist
                if not capabilities.net.egress_allowlist:
                    issues.append("Network capability enabled without egress allowlist")

        return issues

    def install(
        self,
        source: str,
        version: Optional[str] = None,
        force: bool = False,
        verify: bool = True,
    ) -> InstalledPack:
        """
        Install a pack from any source

        Args:
            source: Pack source (name, path, URL, or git repo)
            version: Optional version constraint
            force: Force reinstall if already exists
            verify: Verify pack integrity

        Returns:
            InstalledPack metadata

        Examples:
            # Install from PyPI
            installer.install("greenlang-boiler-solar")
            installer.install("greenlang-boiler-solar==0.2.0")

            # Install from local directory
            installer.install("./packs/boiler-solar")

            # Install from GitHub
            installer.install("github:greenlang/packs/boiler-solar")

            # Install from Hub
            installer.install("hub:boiler-solar")
        """
        # Check if already installed
        existing = self.registry.get(self._extract_pack_name(source))
        if existing and not force:
            logger.info(f"Pack already installed: {existing.name} v{existing.version}")
            return existing

        # Determine source type and install
        if source.startswith("github:"):
            return self._install_from_github(source, version, verify)
        elif source.startswith("hub:"):
            return self._install_from_hub(source, version, verify)
        elif source.startswith(("http://", "https://")):
            return self._install_from_url(source, verify)
        elif Path(source).exists():
            return self._install_from_local(Path(source), verify)
        else:
            # Assume PyPI package
            return self._install_from_pip(source, version, verify)

    def _extract_pack_name(self, source: str) -> str:
        """Extract pack name from source"""
        if ":" in source:
            # Remove prefix (github:, hub:, etc)
            source = source.split(":", 1)[1]

        if "/" in source:
            # Take last component
            source = source.split("/")[-1]

        # Remove version specifiers
        for op in ["==", ">=", "<=", ">", "<", "~="]:
            if op in source:
                source = source.split(op)[0]

        return source.strip()

    def _install_from_pip(
        self, package: str, version: Optional[str] = None, verify: bool = True
    ) -> InstalledPack:
        """
        Install pack from PyPI using pip

        Args:
            package: Package name
            version: Optional version constraint
            verify: Verify after install

        Returns:
            InstalledPack metadata
        """
        # Construct pip install command
        if version:
            if not any(op in version for op in ["==", ">=", "<=", ">", "<", "~="]):
                version = f"=={version}"
            install_spec = f"{package}{version}"
        else:
            install_spec = package

        logger.info(f"Installing from PyPI: {install_spec}")

        # Run pip install
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", install_spec]
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to install {install_spec}: {e}")

        # Force rediscovery of entry points
        self.registry._discover_entry_points()

        # Get the installed pack
        pack_name = self._extract_pack_name(package)
        installed = self.registry.get(pack_name)

        if not installed:
            # Try with greenlang- prefix
            installed = self.registry.get(f"greenlang-{pack_name}")

        if not installed:
            raise RuntimeError(f"Pack not found after installation: {pack_name}")

        logger.info(f"Successfully installed: {installed.name} v{installed.version}")
        return installed

    def _install_from_local(self, path: Path, verify: bool = True) -> InstalledPack:
        """
        Install pack from local directory

        Args:
            path: Path to pack directory
            verify: Verify pack integrity

        Returns:
            InstalledPack metadata
        """
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Check for pack.yaml
        manifest_path = path / "pack.yaml"
        if not manifest_path.exists():
            raise ValueError(f"No pack.yaml found at {path}")

        logger.info(f"Installing from local directory: {path}")

        # Load and validate manifest
        manifest = PackManifest.from_yaml(path)

        # Copy to packs directory
        dest_dir = Path.home() / ".greenlang" / "packs" / manifest.name
        dest_dir.parent.mkdir(parents=True, exist_ok=True)

        if dest_dir.exists():
            logger.warning(f"Removing existing pack at {dest_dir}")
            shutil.rmtree(dest_dir)

        shutil.copytree(path, dest_dir)

        # Register with registry
        installed = self.registry.register(dest_dir, verify=verify)

        logger.info(f"Successfully installed: {installed.name} v{installed.version}")
        return installed

    def _install_from_github(
        self, repo_spec: str, version: Optional[str] = None, verify: bool = True
    ) -> InstalledPack:
        """
        Install pack from GitHub repository

        Args:
            repo_spec: GitHub repo spec (github:owner/repo/pack-name)
            version: Optional version/tag
            verify: Verify pack

        Returns:
            InstalledPack metadata
        """
        # Parse repo spec
        parts = repo_spec.replace("github:", "").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub spec: {repo_spec}")

        owner = parts[0]
        repo = parts[1]
        pack_name = parts[2] if len(parts) > 2 else None

        # Construct download URL
        branch = version or "main"
        if pack_name:
            url = f"https://github.com/{owner}/{repo}/archive/{branch}.tar.gz"
        else:
            url = f"https://github.com/{owner}/{repo}/archive/{branch}.tar.gz"

        # Validate Git URL
        validate_git_url(url)

        logger.info(f"Downloading from GitHub: {url}")

        # Download to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "pack.tar.gz"

            # Secure download
            safe_download(url, str(archive_path), session=self.session)

            # Extract archive safely
            extract_dir = Path(tmpdir) / "extracted"
            safe_extract_archive(archive_path, extract_dir)

            # Find pack directory
            extracted = list(extract_dir.glob("*/"))
            if not extracted:
                raise RuntimeError("No files extracted from archive")

            repo_dir = extracted[0]
            if pack_name:
                pack_dir = repo_dir / pack_name
            else:
                pack_dir = repo_dir

            if not (pack_dir / "pack.yaml").exists():
                # Search for pack.yaml
                pack_yamls = list(repo_dir.glob("**/pack.yaml"))
                if pack_yamls:
                    pack_dir = pack_yamls[0].parent
                else:
                    raise RuntimeError("No pack.yaml found in repository")

            # Install from extracted directory
            return self._install_from_local(pack_dir, verify)

    def _install_from_hub(
        self, hub_spec: str, version: Optional[str] = None, verify: bool = True
    ) -> InstalledPack:
        """
        Install pack from GreenLang Hub

        Args:
            hub_spec: Hub spec (hub:pack-name)
            version: Optional version
            verify: Verify pack

        Returns:
            InstalledPack metadata
        """
        pack_name = hub_spec.replace("hub:", "")

        # Hub API endpoint (configurable)
        hub_url = "https://hub.greenlang.ai"

        # Get pack metadata
        metadata_url = f"{hub_url}/api/packs/{pack_name}"
        if version:
            metadata_url += f"?version={version}"

        # Validate Hub URL
        validate_url(metadata_url)

        logger.info(f"Fetching metadata from Hub: {metadata_url}")

        try:
            response = self.session.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch from Hub: {e}")

        # Download pack archive
        download_url = metadata.get("download_url")
        if not download_url:
            raise RuntimeError("No download URL in Hub response")

        # Validate download URL
        validate_url(download_url)

        logger.info(f"Downloading pack: {download_url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / f"{pack_name}.glpack"

            # Secure download
            safe_download(download_url, str(archive_path), session=self.session)

            # Extract and install
            return self._install_from_archive(archive_path, verify)

    def _install_from_url(self, url: str, verify: bool = True) -> InstalledPack:
        """
        Install pack from direct URL

        Args:
            url: Direct URL to pack archive
            verify: Verify pack

        Returns:
            InstalledPack metadata
        """
        # Validate URL
        validate_url(url)

        logger.info(f"Downloading from URL: {url}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Determine filename from URL
            parsed = urlparse(url)
            filename = Path(parsed.path).name or "pack.archive"
            archive_path = Path(tmpdir) / filename

            # Secure download
            safe_download(url, str(archive_path), session=self.session)

            # Install from archive
            return self._install_from_archive(archive_path, verify)

    def _install_from_archive(
        self, archive_path: Path, verify: bool = True
    ) -> InstalledPack:
        """
        Install pack from archive file

        Args:
            archive_path: Path to archive (.glpack, .tar.gz, .zip)
            verify: Verify pack

        Returns:
            InstalledPack metadata
        """
        # Verify signature if required
        if verify:
            try:
                verified, metadata = self.verifier.verify_pack(archive_path)
                if not verified:
                    logger.warning(f"Pack signature not verified: {archive_path.name}")
            except SignatureVerificationError as e:
                logger.error(f"Signature verification failed: {e}")
                raise

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()

            # Safe extraction with path traversal protection
            safe_extract_archive(archive_path, extract_dir)

            # Find pack.yaml
            pack_yamls = list(extract_dir.glob("**/pack.yaml"))
            if not pack_yamls:
                raise RuntimeError("No pack.yaml found in archive")

            pack_dir = pack_yamls[0].parent

            # Validate pack structure
            validate_pack_structure(pack_dir)

            # Install from extracted directory
            return self._install_from_local(pack_dir, verify)

    def uninstall(self, pack_name: str) -> bool:
        """
        Uninstall a pack

        Args:
            pack_name: Name of pack to uninstall

        Returns:
            True if uninstalled successfully
        """
        pack = self.registry.get(pack_name)
        if not pack:
            logger.warning(f"Pack not found: {pack_name}")
            return False

        # If installed via pip, use pip uninstall
        if pack.location.startswith("entry_point:") or "site-packages" in pack.location:
            # Try to uninstall via pip
            package_name = f"greenlang-{pack_name}"
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "uninstall", "-y", package_name]
                )
                logger.info(f"Uninstalled via pip: {package_name}")
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to uninstall via pip: {package_name}")

        # Remove from local directory if exists
        if Path(pack.location).exists() and "site-packages" not in pack.location:
            shutil.rmtree(pack.location)
            logger.info(f"Removed local files: {pack.location}")

        # Unregister from registry
        self.registry.unregister(pack_name)

        logger.info(f"Successfully uninstalled: {pack_name}")
        return True

    def update(self, pack_name: str, version: Optional[str] = None) -> InstalledPack:
        """
        Update a pack to latest or specified version

        Args:
            pack_name: Name of pack to update
            version: Optional target version

        Returns:
            Updated InstalledPack metadata
        """
        pack = self.registry.get(pack_name)
        if not pack:
            raise ValueError(f"Pack not found: {pack_name}")

        logger.info(f"Updating {pack_name} from v{pack.version}")

        # Determine source for update
        if pack.location.startswith("entry_point:") or "site-packages" in pack.location:
            # Update via pip
            return self._install_from_pip(
                f"greenlang-{pack_name}", version, verify=True
            )
        else:
            # For local packs, need to know original source
            # This would require storing source info in registry
            logger.warning("Cannot update local pack without source information")
            raise NotImplementedError("Local pack updates not yet supported")

    def list_available(self, source: str = "pypi") -> List[Dict[str, Any]]:
        """
        List available packs from a source

        Args:
            source: Source to list from (pypi, hub)

        Returns:
            List of available packs with metadata
        """
        if source == "pypi":
            return self._list_pypi_packs()
        elif source == "hub":
            return self._list_hub_packs()
        else:
            raise ValueError(f"Unknown source: {source}")

    def _list_pypi_packs(self) -> List[Dict[str, Any]]:
        """List available packs from PyPI"""
        # Search PyPI for greenlang packs
        try:
            import xmlrpc.client

            client = xmlrpc.client.ServerProxy("https://pypi.org/pypi")

            # Search for packages with greenlang prefix
            results = []
            for package in client.search({"name": "greenlang-"}):
                results.append(
                    {
                        "name": package["name"],
                        "version": package["version"],
                        "summary": package["summary"],
                        "source": "pypi",
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Failed to search PyPI: {e}")
            return []

    def _list_hub_packs(self) -> List[Dict[str, Any]]:
        """List available packs from Hub"""
        hub_url = "https://hub.greenlang.ai"

        try:
            from greenlang.security.http import get as secure_get

            api_url = f"{hub_url}/api/packs"
            response = secure_get(api_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from Hub: {e}")
            return []
