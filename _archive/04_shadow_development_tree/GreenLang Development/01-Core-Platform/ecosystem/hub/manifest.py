# -*- coding: utf-8 -*-
"""
Manifest Handling Utilities for GreenLang Packs
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator
import hashlib
import yaml
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PackDependency(BaseModel):
    """Pack dependency specification"""

    name: str
    version: str
    source: Optional[str] = None
    optional: bool = False


class PackAuthor(BaseModel):
    """Pack author information"""

    name: str
    email: Optional[str] = None
    url: Optional[str] = None


class PackManifest(BaseModel):
    """Pack manifest schema"""

    # Required fields
    name: str = Field(..., description="Pack name")
    version: str = Field(..., description="Pack version (semver)")
    description: str = Field(..., description="Pack description")

    # Optional metadata
    author: Optional[PackAuthor] = None
    license: Optional[str] = None
    homepage: Optional[str] = None
    repository: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

    # Pack contents
    entry_point: Optional[str] = Field(None, description="Main entry point")
    modules: List[str] = Field(default_factory=list, description="Included modules")
    resources: List[str] = Field(default_factory=list, description="Resource files")

    # Dependencies
    dependencies: List[PackDependency] = Field(default_factory=list)
    dev_dependencies: List[PackDependency] = Field(default_factory=list)

    # Compatibility
    greenlang_version: Optional[str] = Field(
        None, description="Required GreenLang version"
    )
    python_version: Optional[str] = Field(None, description="Required Python version")
    platform: Optional[List[str]] = Field(None, description="Supported platforms")

    # Build info
    build_date: Optional[datetime] = None
    build_number: Optional[int] = None
    checksum: Optional[str] = None

    # Registry metadata
    published: bool = False
    downloads: int = 0
    stars: int = 0

    @validator("version")
    def validate_version(cls, v):
        """Validate semantic version"""
        import re

        pattern = r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid version format: {v}")
        return v

    @validator("name")
    def validate_name(cls, v):
        """Validate pack name"""
        import re

        if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", v):
            raise ValueError(f"Invalid pack name: {v}")
        return v

    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert to dictionary with JSON serialization support"""
        data = super().dict(**kwargs)

        # Convert datetime to ISO format
        if data.get("build_date"):
            data["build_date"] = data["build_date"].isoformat()

        # Convert Pydantic models to dicts
        if data.get("author"):
            data["author"] = (
                data["author"]
                if isinstance(data["author"], dict)
                else data["author"].dict()
            )

        if data.get("dependencies"):
            data["dependencies"] = [
                dep if isinstance(dep, dict) else dep.dict()
                for dep in data["dependencies"]
            ]

        if data.get("dev_dependencies"):
            data["dev_dependencies"] = [
                dep if isinstance(dep, dict) else dep.dict()
                for dep in data["dev_dependencies"]
            ]

        return data


def load_manifest(pack_path: Path) -> PackManifest:
    """
    Load manifest from pack directory or archive

    Args:
        pack_path: Path to pack directory or manifest file

    Returns:
        PackManifest object
    """
    pack_path = Path(pack_path)

    # Determine manifest path
    if pack_path.is_file():
        if pack_path.name in ["manifest.json", "manifest.yaml", "manifest.yml"]:
            manifest_path = pack_path
        else:
            raise ValueError(f"Not a manifest file: {pack_path}")
    else:
        # Look for manifest in directory
        manifest_path = None
        for name in [
            "manifest.json",
            "manifest.yaml",
            "manifest.yml",
            "greenlang.json",
        ]:
            candidate = pack_path / name
            if candidate.exists():
                manifest_path = candidate
                break

        if not manifest_path:
            raise FileNotFoundError(f"No manifest found in {pack_path}")

    logger.info(f"Loading manifest from {manifest_path}")

    try:
        # Load based on file extension
        if manifest_path.suffix in [".yaml", ".yml"]:
            with open(manifest_path, "r") as f:
                data = yaml.safe_load(f)
        else:
            with open(manifest_path, "r") as f:
                data = json.load(f)

        # Parse into model
        manifest = PackManifest(**data)

        logger.info(f"Loaded manifest for {manifest.name} v{manifest.version}")
        return manifest

    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        raise


def save_manifest(pack_path: Path, manifest, format: str = "json") -> Path:
    """
    Save manifest to pack directory

    Args:
        pack_path: Pack directory path
        manifest: Manifest object or dictionary
        format: Output format ('json' or 'yaml')

    Returns:
        Path to saved manifest file
    """
    pack_path = Path(pack_path)
    pack_path.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if isinstance(manifest, PackManifest):
        data = manifest.dict()
    else:
        data = manifest

    # Determine output file
    if format == "yaml":
        manifest_file = pack_path / "manifest.yaml"
        with open(manifest_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        manifest_file = pack_path / "manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(data, f, indent=2)

    logger.info(f"Saved manifest to {manifest_file}")
    return manifest_file


def validate_manifest(manifest) -> bool:
    """
    Validate manifest data

    Args:
        manifest: Manifest to validate

    Returns:
        True if valid

    Raises:
        ValueError: If manifest is invalid
    """
    try:
        if isinstance(manifest, dict):
            # Parse into model for validation
            PackManifest(**manifest)
        elif not isinstance(manifest, PackManifest):
            raise ValueError("Invalid manifest type")

        logger.info("Manifest validation successful")
        return True

    except Exception as e:
        logger.error(f"Manifest validation failed: {e}")
        raise ValueError(f"Invalid manifest: {e}")


def create_manifest(
    pack_path: Path,
    name: Optional[str] = None,
    version: str = "0.1.0",
    description: Optional[str] = None,
    author: Optional[str] = None,
    **kwargs,
) -> PackManifest:
    """
    Create a new manifest for a pack

    Args:
        pack_path: Path to pack directory
        name: Pack name (defaults to directory name)
        version: Pack version
        description: Pack description
        author: Author name
        **kwargs: Additional manifest fields

    Returns:
        Created PackManifest
    """
    pack_path = Path(pack_path)

    if not pack_path.exists():
        raise FileNotFoundError(f"Pack directory not found: {pack_path}")

    # Default values
    if not name:
        name = pack_path.name

    if not description:
        description = f"{name} GreenLang pack"

    # Scan pack contents
    modules = []
    resources = []

    for file in pack_path.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(pack_path)

            if file.suffix == ".py":
                modules.append(str(rel_path))
            elif file.suffix in [".json", ".yaml", ".yml", ".txt", ".md"]:
                resources.append(str(rel_path))

    # Create manifest
    manifest_data = {
        "name": name,
        "version": version,
        "description": description,
        "modules": modules,
        "resources": resources,
        "build_date": DeterministicClock.utcnow(),
        **kwargs,
    }

    if author:
        if isinstance(author, str):
            manifest_data["author"] = {"name": author}
        else:
            manifest_data["author"] = author

    manifest = PackManifest(**manifest_data)

    logger.info(f"Created manifest for {manifest.name}")
    return manifest


def update_manifest(manifest_path: Path, updates: Dict[str, Any]) -> PackManifest:
    """
    Update existing manifest with new values

    Args:
        manifest_path: Path to manifest file
        updates: Dictionary of updates

    Returns:
        Updated PackManifest
    """
    # Load existing manifest
    manifest = load_manifest(manifest_path)

    # Apply updates
    manifest_dict = manifest.dict()
    manifest_dict.update(updates)

    # Re-validate
    updated_manifest = PackManifest(**manifest_dict)

    # Save back
    save_manifest(manifest_path.parent, updated_manifest)

    logger.info(f"Updated manifest at {manifest_path}")
    return updated_manifest


def merge_manifests(base: PackManifest, override: PackManifest) -> PackManifest:
    """
    Merge two manifests (override takes precedence)

    Args:
        base: Base manifest
        override: Override manifest

    Returns:
        Merged PackManifest
    """
    base_dict = base.dict()
    override_dict = override.dict()

    # Deep merge
    merged = {**base_dict}

    for key, value in override_dict.items():
        if value is not None:
            if isinstance(value, list) and key in merged:
                # Merge lists
                merged[key] = list(set(merged.get(key, []) + value))
            else:
                merged[key] = value

    return PackManifest(**merged)


def calculate_manifest_checksum(manifest: PackManifest) -> str:
    """
    Calculate checksum for manifest content

    Args:
        manifest: Manifest object

    Returns:
        SHA256 checksum
    """
    # Serialize to JSON for consistent hashing
    content = json.dumps(manifest.dict(), sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def compare_manifests(
    manifest1: PackManifest, manifest2: PackManifest
) -> Dict[str, Any]:
    """
    Compare two manifests and return differences

    Args:
        manifest1: First manifest
        manifest2: Second manifest

    Returns:
        Dictionary of differences
    """
    dict1 = manifest1.dict()
    dict2 = manifest2.dict()

    differences = {}

    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        val1 = dict1.get(key)
        val2 = dict2.get(key)

        if val1 != val2:
            differences[key] = {"old": val1, "new": val2}

    return differences
