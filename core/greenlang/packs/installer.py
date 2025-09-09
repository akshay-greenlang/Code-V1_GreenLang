"""
Pack Installer
==============

Handles installation of packs from various sources:
- PyPI (pip install)
- Local directories
- GitHub repositories
- GreenLang Hub
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil
import tarfile
import zipfile
from urllib.parse import urlparse
import requests

from .manifest import PackManifest
from .registry import PackRegistry, InstalledPack

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
    
    def install(self, source: str, version: Optional[str] = None, 
                force: bool = False, verify: bool = True) -> InstalledPack:
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
        elif source.startswith("http"):
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
    
    def _install_from_pip(self, package: str, version: Optional[str] = None, 
                         verify: bool = True) -> InstalledPack:
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
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", install_spec
            ])
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
    
    def _install_from_github(self, repo_spec: str, version: Optional[str] = None,
                           verify: bool = True) -> InstalledPack:
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
        
        logger.info(f"Downloading from GitHub: {url}")
        
        # Download to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "pack.tar.gz"
            
            # Download archive
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract archive
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(tmpdir)
            
            # Find pack directory
            extracted = list(Path(tmpdir).glob("*/"))
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
    
    def _install_from_hub(self, hub_spec: str, version: Optional[str] = None,
                         verify: bool = True) -> InstalledPack:
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
        
        logger.info(f"Fetching metadata from Hub: {metadata_url}")
        
        try:
            response = requests.get(metadata_url)
            response.raise_for_status()
            metadata = response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to fetch from Hub: {e}")
        
        # Download pack archive
        download_url = metadata.get("download_url")
        if not download_url:
            raise RuntimeError("No download URL in Hub response")
        
        logger.info(f"Downloading pack: {download_url}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / f"{pack_name}.glpack"
            
            # Download
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
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
        logger.info(f"Downloading from URL: {url}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Determine filename from URL
            parsed = urlparse(url)
            filename = Path(parsed.path).name or "pack.archive"
            archive_path = Path(tmpdir) / filename
            
            # Download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(archive_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Install from archive
            return self._install_from_archive(archive_path, verify)
    
    def _install_from_archive(self, archive_path: Path, verify: bool = True) -> InstalledPack:
        """
        Install pack from archive file
        
        Args:
            archive_path: Path to archive (.glpack, .tar.gz, .zip)
            verify: Verify pack
        
        Returns:
            InstalledPack metadata
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir) / "extracted"
            extract_dir.mkdir()
            
            # Extract based on file type
            if archive_path.suffix == ".glpack" or archive_path.name.endswith(".tar.gz"):
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_dir)
            elif archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            # Find pack.yaml
            pack_yamls = list(extract_dir.glob("**/pack.yaml"))
            if not pack_yamls:
                raise RuntimeError("No pack.yaml found in archive")
            
            pack_dir = pack_yamls[0].parent
            
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
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", package_name
                ])
                logger.info(f"Uninstalled via pip: {package_name}")
            except subprocess.CalledProcessError:
                logger.warning(f"Failed to uninstall via pip: {package_name}")
        
        # Remove from local directory if exists
        if Path(pack.location).exists() and not "site-packages" in pack.location:
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
            return self._install_from_pip(f"greenlang-{pack_name}", version, verify=True)
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
                results.append({
                    "name": package["name"],
                    "version": package["version"],
                    "summary": package["summary"],
                    "source": "pypi"
                })
            
            return results
        except Exception as e:
            logger.error(f"Failed to search PyPI: {e}")
            return []
    
    def _list_hub_packs(self) -> List[Dict[str, Any]]:
        """List available packs from Hub"""
        hub_url = "https://hub.greenlang.ai"
        
        try:
            response = requests.get(f"{hub_url}/api/packs")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch from Hub: {e}")
            return []