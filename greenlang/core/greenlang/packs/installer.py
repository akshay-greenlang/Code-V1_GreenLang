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

        Raises:
            ValueError: If pack not found or version comparison fails
            RuntimeError: If update or rollback fails
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
            # Handle local pack updates
            return self._update_local_pack(pack, version)

    def _update_local_pack(
        self, pack: InstalledPack, target_version: Optional[str] = None
    ) -> InstalledPack:
        """
        Update a locally installed pack with backup and rollback support.

        This method implements a safe update process:
        1. Read current pack version from local installation
        2. Compare with available/target version
        3. Backup current pack
        4. Install new version
        5. Rollback on failure

        Args:
            pack: Currently installed pack metadata
            target_version: Optional specific version to update to

        Returns:
            Updated InstalledPack metadata

        Raises:
            ValueError: If version comparison fails or no update available
            RuntimeError: If update fails and rollback is required
        """
        import hashlib
        from datetime import datetime

        pack_dir = Path(pack.location)
        if not pack_dir.exists():
            raise ValueError(f"Pack directory not found: {pack_dir}")

        # Step 1: Read current pack version from local installation
        current_version = pack.version
        current_manifest = PackManifest.from_yaml(pack_dir)
        logger.info(f"Current local pack version: {current_version}")

        # Step 2: Determine target version and compare
        if target_version:
            # User specified a target version
            if not self._is_newer_version(target_version, current_version):
                logger.info(
                    f"Target version {target_version} is not newer than "
                    f"current version {current_version}"
                )
                raise ValueError(
                    f"Target version {target_version} is not newer than "
                    f"installed version {current_version}"
                )
            new_version = target_version
        else:
            # Check manifest metadata for update source
            source_info = self._get_pack_source_info(pack)
            if source_info:
                available_version = self._check_available_version(
                    pack.name, source_info
                )
                if available_version and self._is_newer_version(
                    available_version, current_version
                ):
                    new_version = available_version
                    logger.info(f"Found newer version available: {new_version}")
                else:
                    logger.info(f"Pack {pack.name} is already at latest version")
                    return pack
            else:
                # No source info - require explicit version
                raise ValueError(
                    f"Cannot determine update source for local pack {pack.name}. "
                    "Please specify a target version or reinstall from a known source."
                )

        # Step 3: Create backup of current pack
        backup_dir = self._create_pack_backup(pack_dir, pack.name, current_version)
        logger.info(f"Created backup at: {backup_dir}")

        # Step 4: Attempt to install new version
        try:
            # Try to update from original source if known
            source_info = self._get_pack_source_info(pack)
            if source_info:
                updated_pack = self._install_from_source(
                    source_info, new_version, verify=True
                )
            else:
                # Perform in-place version update if pack files are available locally
                updated_pack = self._perform_inplace_update(
                    pack, pack_dir, new_version
                )

            logger.info(
                f"Successfully updated {pack.name} from v{current_version} "
                f"to v{updated_pack.version}"
            )

            # Clean up backup on success (optional: keep for safety)
            self._cleanup_backup(backup_dir, keep_days=7)

            return updated_pack

        except Exception as e:
            # Step 5: Rollback on failure
            logger.error(f"Update failed: {e}. Initiating rollback...")
            self._rollback_pack(pack_dir, backup_dir, pack.name)
            logger.info(f"Rollback completed. Pack restored to v{current_version}")
            raise RuntimeError(
                f"Failed to update {pack.name} to v{new_version}: {e}. "
                f"Pack has been rolled back to v{current_version}."
            ) from e

    def _is_newer_version(self, version_a: str, version_b: str) -> bool:
        """
        Compare two semantic versions to determine if version_a is newer than version_b.

        Args:
            version_a: First version string (e.g., "1.2.3")
            version_b: Second version string (e.g., "1.2.0")

        Returns:
            True if version_a is strictly newer than version_b
        """
        def parse_version(v: str) -> tuple:
            """Parse version string into comparable tuple."""
            # Remove pre-release and build metadata for comparison
            base_version = v.split("-")[0].split("+")[0]
            parts = base_version.split(".")
            # Pad with zeros to ensure comparable length
            numeric_parts = []
            for part in parts[:3]:  # major, minor, patch
                try:
                    numeric_parts.append(int(part))
                except ValueError:
                    numeric_parts.append(0)
            # Pad to 3 parts
            while len(numeric_parts) < 3:
                numeric_parts.append(0)
            return tuple(numeric_parts)

        try:
            parsed_a = parse_version(version_a)
            parsed_b = parse_version(version_b)
            return parsed_a > parsed_b
        except Exception as e:
            logger.warning(f"Version comparison failed: {e}")
            return False

    def _get_pack_source_info(self, pack: InstalledPack) -> Optional[Dict[str, Any]]:
        """
        Extract source information from pack metadata.

        Args:
            pack: Installed pack metadata

        Returns:
            Dictionary with source type and location, or None if unknown
        """
        manifest = pack.manifest
        metadata = manifest.get("metadata", {})

        # Check for repository URL in metadata
        if metadata:
            repository = metadata.get("repository")
            if repository:
                if "github.com" in repository:
                    return {"type": "github", "url": repository}
                elif "hub.greenlang.ai" in repository:
                    return {"type": "hub", "name": pack.name}
                else:
                    return {"type": "url", "url": repository}

        # Check for homepage
        homepage = metadata.get("homepage") if metadata else None
        if homepage and "github.com" in homepage:
            return {"type": "github", "url": homepage}

        return None

    def _check_available_version(
        self, pack_name: str, source_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Check for available version from source.

        Args:
            pack_name: Name of the pack
            source_info: Source information dictionary

        Returns:
            Latest available version string, or None if check fails
        """
        source_type = source_info.get("type")

        try:
            if source_type == "hub":
                # Query Hub API for latest version
                hub_url = "https://hub.greenlang.ai"
                response = self.session.get(f"{hub_url}/api/packs/{pack_name}/latest")
                if response.ok:
                    return response.json().get("version")

            elif source_type == "github":
                # Query GitHub API for latest release
                url = source_info.get("url", "")
                # Parse owner/repo from GitHub URL
                parts = url.replace("https://github.com/", "").split("/")
                if len(parts) >= 2:
                    owner, repo = parts[0], parts[1].replace(".git", "")
                    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
                    response = self.session.get(api_url)
                    if response.ok:
                        tag = response.json().get("tag_name", "")
                        # Remove 'v' prefix if present
                        return tag.lstrip("v")

        except Exception as e:
            logger.warning(f"Failed to check available version: {e}")

        return None

    def _create_pack_backup(
        self, pack_dir: Path, pack_name: str, version: str
    ) -> Path:
        """
        Create a backup of the current pack installation.

        Args:
            pack_dir: Path to current pack directory
            pack_name: Name of the pack
            version: Current version being backed up

        Returns:
            Path to backup directory
        """
        from datetime import datetime

        backup_base = Path.home() / ".greenlang" / "backups"
        backup_base.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{pack_name}_v{version}_{timestamp}"
        backup_dir = backup_base / backup_name

        logger.info(f"Creating backup of {pack_name} v{version} to {backup_dir}")
        shutil.copytree(pack_dir, backup_dir)

        # Create backup metadata file
        backup_metadata = {
            "pack_name": pack_name,
            "version": version,
            "original_location": str(pack_dir),
            "backup_timestamp": timestamp,
            "backup_hash": self.registry._calculate_directory_hash(backup_dir),
        }

        import json
        with open(backup_dir / ".backup_metadata.json", "w") as f:
            json.dump(backup_metadata, f, indent=2)

        return backup_dir

    def _rollback_pack(
        self, pack_dir: Path, backup_dir: Path, pack_name: str
    ) -> None:
        """
        Rollback pack to backup version.

        Args:
            pack_dir: Current pack directory to restore
            backup_dir: Backup directory to restore from
            pack_name: Name of the pack
        """
        logger.warning(f"Rolling back {pack_name} from backup at {backup_dir}")

        # Verify backup integrity before rollback
        backup_metadata_path = backup_dir / ".backup_metadata.json"
        if backup_metadata_path.exists():
            import json
            with open(backup_metadata_path) as f:
                backup_metadata = json.load(f)

            # Verify backup hash
            current_backup_hash = self.registry._calculate_directory_hash(backup_dir)
            expected_hash = backup_metadata.get("backup_hash")

            if expected_hash and current_backup_hash != expected_hash:
                logger.error("Backup integrity check failed - hash mismatch")
                raise RuntimeError(
                    "Backup integrity verification failed. Manual intervention required."
                )

        # Remove failed installation
        if pack_dir.exists():
            shutil.rmtree(pack_dir)

        # Restore from backup (exclude metadata file)
        shutil.copytree(
            backup_dir,
            pack_dir,
            ignore=shutil.ignore_patterns(".backup_metadata.json"),
        )

        # Re-register the restored pack
        self.registry.register(pack_dir, verify=True)

        logger.info(f"Rollback completed for {pack_name}")

    def _cleanup_backup(self, backup_dir: Path, keep_days: int = 7) -> None:
        """
        Clean up old backups, keeping recent ones.

        Args:
            backup_dir: Path to the backup just created
            keep_days: Number of days to keep backups
        """
        from datetime import datetime, timedelta

        backup_base = backup_dir.parent
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for backup in backup_base.iterdir():
            if not backup.is_dir():
                continue

            # Check backup age from metadata
            metadata_file = backup / ".backup_metadata.json"
            if metadata_file.exists():
                try:
                    import json
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    timestamp_str = metadata.get("backup_timestamp", "")
                    if timestamp_str:
                        backup_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        if backup_date < cutoff_date:
                            logger.info(f"Removing old backup: {backup}")
                            shutil.rmtree(backup)
                except Exception as e:
                    logger.warning(f"Could not process backup {backup}: {e}")

    def _install_from_source(
        self, source_info: Dict[str, Any], version: str, verify: bool = True
    ) -> InstalledPack:
        """
        Install pack from known source.

        Args:
            source_info: Source information dictionary
            version: Version to install
            verify: Whether to verify pack integrity

        Returns:
            Installed pack metadata
        """
        source_type = source_info.get("type")

        if source_type == "hub":
            pack_name = source_info.get("name")
            return self._install_from_hub(f"hub:{pack_name}", version, verify)

        elif source_type == "github":
            url = source_info.get("url", "")
            # Convert GitHub URL to github: spec
            parts = url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1].replace(".git", "")
                return self._install_from_github(f"github:{owner}/{repo}", version, verify)

        elif source_type == "url":
            url = source_info.get("url", "")
            return self._install_from_url(url, verify)

        raise ValueError(f"Unknown source type: {source_type}")

    def _perform_inplace_update(
        self, pack: InstalledPack, pack_dir: Path, new_version: str
    ) -> InstalledPack:
        """
        Perform in-place version update when no external source is available.

        This is typically used when the pack files have been manually updated
        in the pack directory and only the registry needs to be refreshed.

        Args:
            pack: Current installed pack
            pack_dir: Pack directory path
            new_version: New version to register

        Returns:
            Updated InstalledPack metadata
        """
        # Re-read manifest to get updated information
        manifest = PackManifest.from_yaml(pack_dir)

        # Verify the manifest version matches expected
        if manifest.version != new_version:
            raise ValueError(
                f"Manifest version ({manifest.version}) does not match "
                f"target version ({new_version}). Please update pack.yaml first."
            )

        # Validate pack structure
        errors = manifest.validate_files(pack_dir)
        if errors:
            raise ValueError(f"Pack validation failed: {', '.join(errors)}")

        # Re-register with updated information
        updated_pack = self.registry.register(pack_dir, verify=True)

        logger.info(
            f"In-place update completed for {pack.name}: "
            f"v{pack.version} -> v{updated_pack.version}"
        )

        return updated_pack

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

            response = secure_get(f"{hub_url}/api/packs")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from Hub: {e}")
            return []
