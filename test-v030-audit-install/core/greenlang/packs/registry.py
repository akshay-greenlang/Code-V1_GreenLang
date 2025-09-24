"""
Pack Registry
=============

Manages installed packs and provides discovery mechanisms.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import importlib.metadata

from .manifest import PackManifest


logger = logging.getLogger(__name__)


@dataclass
class InstalledPack:
    """Metadata for an installed pack"""

    name: str
    version: str
    location: str  # Path or entry point
    manifest: Dict[str, Any]
    installed_at: str
    hash: str  # SHA256 of pack contents
    verified: bool = False
    signature: Optional[str] = None


class PackRegistry:
    """
    Central registry for installed packs

    Tracks:
    - Locally installed packs (from filesystem)
    - Packs installed via pip (entry points)
    - Remote packs from Hub
    """

    def __init__(self, registry_dir: Optional[Path] = None):
        """
        Initialize registry

        Args:
            registry_dir: Directory for registry data (default: ~/.greenlang/registry)
        """
        self.registry_dir = registry_dir or Path.home() / ".greenlang" / "registry"
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "packs.json"
        self.packs: Dict[str, InstalledPack] = {}

        # Load existing registry
        self._load_registry()

        # Discover packs
        self._discover_entry_points()
        self._discover_local_packs()

    def _load_registry(self):
        """Load registry from disk"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)
                    for pack_data in data.get("packs", []):
                        pack = InstalledPack(**pack_data)
                        self.packs[pack.name] = pack
                logger.info(f"Loaded {len(self.packs)} packs from registry")
            except Exception as e:
                logger.error(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save registry to disk"""
        try:
            data = {
                "version": "0.1.0",
                "updated_at": datetime.now().isoformat(),
                "packs": [asdict(pack) for pack in self.packs.values()],
            }
            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.packs)} packs to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _discover_entry_points(self):
        """Discover packs installed as Python packages"""
        try:
            # Look for greenlang.packs entry points
            import sys

            if sys.version_info >= (3, 10):
                eps = importlib.metadata.entry_points(group="greenlang.packs")
            else:
                eps = importlib.metadata.entry_points().get("greenlang.packs", [])

            for ep in eps:
                try:
                    # Load the entry point - should return path to pack.yaml
                    manifest_path_func = ep.load()

                    # Get manifest path
                    if callable(manifest_path_func):
                        manifest_path = Path(manifest_path_func())
                    else:
                        manifest_path = Path(manifest_path_func)

                    if not manifest_path.exists():
                        logger.warning(
                            f"Manifest path does not exist for {ep.name}: {manifest_path}"
                        )
                        continue

                    # Load manifest from the path
                    pack_dir = manifest_path.parent
                    manifest = PackManifest.from_yaml(pack_dir)

                    # Register the pack
                    installed_pack = InstalledPack(
                        name=manifest.name,
                        version=manifest.version,
                        location=str(pack_dir),
                        manifest=manifest.model_dump(),
                        installed_at=datetime.now().isoformat(),
                        hash=self._calculate_directory_hash(pack_dir),
                        verified=True,  # Entry points are pip-installed, assume verified
                    )

                    # Use manifest name as key (not entry point name)
                    self.packs[manifest.name] = installed_pack
                    logger.info(
                        f"Discovered pack from entry point: {manifest.name} v{manifest.version}"
                    )

                except Exception as e:
                    logger.error(f"Failed to load entry point {ep.name}: {e}")

        except Exception as e:
            logger.debug(f"No entry points found: {e}")

    def _discover_local_packs(self):
        """Discover packs in local directories"""
        # Check standard pack directories
        pack_dirs = [
            Path.cwd() / "packs",  # Current directory
            Path.home() / ".greenlang" / "packs",  # User directory
            Path("/opt/greenlang/packs"),  # System directory (Linux/Mac)
        ]

        for pack_dir in pack_dirs:
            if not pack_dir.exists():
                continue

            # Look for subdirectories with pack.yaml
            for subdir in pack_dir.iterdir():
                if not subdir.is_dir():
                    continue

                manifest_path = subdir / "pack.yaml"
                if manifest_path.exists():
                    try:
                        manifest = PackManifest.from_yaml(manifest_path)

                        # Calculate hash of pack contents
                        pack_hash = self._calculate_directory_hash(subdir)

                        installed_pack = InstalledPack(
                            name=manifest.name,
                            version=manifest.version,
                            location=str(subdir),
                            manifest=manifest.model_dump(),
                            installed_at=datetime.now().isoformat(),
                            hash=pack_hash,
                            verified=False,  # Local packs need verification
                        )

                        self.packs[manifest.name] = installed_pack
                        logger.info(
                            f"Discovered local pack: {manifest.name} at {subdir}"
                        )

                    except Exception as e:
                        logger.error(f"Failed to load pack from {subdir}: {e}")

    def _calculate_hash(self, content: str) -> str:
        """Calculate SHA256 hash of content"""
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_directory_hash(self, directory: Path) -> str:
        """Calculate hash of directory contents"""
        hasher = hashlib.sha256()

        # Sort files for deterministic hash
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and not file_path.name.startswith("."):
                hasher.update(file_path.name.encode())
                with open(file_path, "rb") as f:
                    hasher.update(f.read())

        return hasher.hexdigest()

    def register(self, pack_path: Path, verify: bool = True) -> InstalledPack:
        """
        Register a new pack

        Args:
            pack_path: Path to pack directory
            verify: Whether to verify pack integrity

        Returns:
            InstalledPack metadata
        """
        manifest_path = pack_path / "pack.yaml"
        if not manifest_path.exists():
            raise ValueError(f"No pack.yaml found at {pack_path}")

        manifest = PackManifest.from_yaml(manifest_path)

        # Validate files exist
        errors = manifest.validate_files(pack_path)
        if errors:
            raise ValueError(f"Pack validation failed: {', '.join(errors)}")

        # Calculate hash
        pack_hash = self._calculate_directory_hash(pack_path)

        # Create installed pack record
        installed_pack = InstalledPack(
            name=manifest.name,
            version=manifest.version,
            location=str(pack_path),
            manifest=manifest.model_dump(),
            installed_at=datetime.now().isoformat(),
            hash=pack_hash,
            verified=verify,
        )

        # Register
        self.packs[manifest.name] = installed_pack
        self._save_registry()

        logger.info(f"Registered pack: {manifest.name} v{manifest.version}")
        return installed_pack

    def unregister(self, pack_name: str):
        """Remove a pack from registry"""
        if pack_name in self.packs:
            del self.packs[pack_name]
            self._save_registry()
            logger.info(f"Unregistered pack: {pack_name}")
        else:
            raise ValueError(f"Pack not found: {pack_name}")

    def get(
        self, pack_name: str, version: Optional[str] = None
    ) -> Optional[InstalledPack]:
        """
        Get installed pack by name and optionally version

        Args:
            pack_name: Name of the pack
            version: Optional version to match

        Returns:
            InstalledPack if found, None otherwise
        """
        pack = self.packs.get(pack_name)
        if pack and version:
            # Check if version matches
            if pack.version != version:
                return None
        return pack

    def list(self, kind: Optional[str] = None) -> List[InstalledPack]:
        """
        List all installed packs

        Args:
            kind: Filter by pack kind (pack, dataset, connector)

        Returns:
            List of installed packs
        """
        packs = list(self.packs.values())

        if kind:
            packs = [p for p in packs if p.manifest.get("kind") == kind]

        return packs

    def search(self, query: str) -> List[InstalledPack]:
        """Search for packs by name or description"""
        results = []
        query_lower = query.lower()

        for pack in self.packs.values():
            if (
                query_lower in pack.name.lower()
                or query_lower in pack.manifest.get("description", "").lower()
            ):
                results.append(pack)

        return results

    def verify(self, pack_name: str) -> bool:
        """
        Verify pack integrity

        Args:
            pack_name: Name of pack to verify

        Returns:
            True if pack is verified
        """
        pack = self.get(pack_name)
        if not pack:
            raise ValueError(f"Pack not found: {pack_name}")

        if pack.location.startswith("entry_point:"):
            # Entry points are pre-verified
            return True

        # Recalculate hash and compare
        pack_dir = Path(pack.location)
        current_hash = self._calculate_directory_hash(pack_dir)

        if current_hash == pack.hash:
            pack.verified = True
            self._save_registry()
            return True
        else:
            logger.warning(f"Pack verification failed: {pack_name}")
            return False

    def list_pipelines(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all pipelines from installed packs

        Returns:
            Dictionary mapping pack names to their pipelines
        """
        pipelines = {}

        for pack in self.packs.values():
            pack_pipelines = []

            # Get pipelines from contents
            if pack.manifest.get("contents"):
                for pipeline_file in pack.manifest["contents"].get("pipelines", []):
                    pack_pipelines.append(
                        {
                            "name": pipeline_file.replace(".yaml", "").replace(
                                ".yml", ""
                            ),
                            "file": pipeline_file,
                            "description": f"Pipeline from {pack.name}",
                        }
                    )

            if pack_pipelines:
                pipelines[pack.name] = pack_pipelines

        return pipelines

    def get_dependencies(self, pack_name: str) -> List[str]:
        """Get dependencies for a pack"""
        pack = self.get(pack_name)
        if not pack:
            return []

        deps = []
        for dep in pack.manifest.get("dependencies", []):
            # Handle both string and dict format
            if isinstance(dep, str):
                deps.append(dep)
            elif isinstance(dep, dict):
                deps.append(f"{dep['name']}{dep.get('version', '')}")

        return deps
