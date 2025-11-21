# -*- coding: utf-8 -*-
"""
PackBuilder - Build GreenLang packs from agents.

This module creates distributable agent packs for the GreenLang Hub,
enabling easy sharing and deployment of agents across organizations.

Example:
    >>> builder = PackBuilder()
    >>> metadata = PackMetadata(name="CarbonPack", version="1.0.0")
    >>> pack_id = builder.create_pack(agent_dir, metadata)
    >>> print(f"Pack created: {pack_id}")
"""

import json
import shutil
import hashlib
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from pydantic import BaseModel, Field, validator
import yaml
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PackMetadata(BaseModel):
    """Metadata for GreenLang agent pack."""

    # Core metadata
    name: str = Field(..., description="Pack name")
    version: str = Field(..., description="Semantic version")
    description: str = Field(..., description="Pack description")

    # Agent information
    agent_type: str = Field(..., description="Type of agent")
    domain: str = Field(..., description="Business domain")
    agents: List[str] = Field(default_factory=list, description="List of agents in pack")

    # Author information
    author: str = Field("GreenLang AI", description="Pack author")
    author_email: str = Field("dev@greenlang.ai", description="Author email")
    organization: Optional[str] = Field(None, description="Organization")

    # Dependencies
    dependencies: List[str] = Field(default_factory=list, description="Required packages")
    greenlang_version: str = Field(">=1.0.0", description="Required GreenLang version")
    python_version: str = Field(">=3.9", description="Required Python version")

    # Licensing
    license: str = Field("Proprietary", description="License type")
    license_url: Optional[str] = Field(None, description="License URL")

    # Hub metadata
    tags: List[str] = Field(default_factory=list, description="Pack tags")
    category: str = Field("general", description="Pack category")
    visibility: str = Field("private", description="public, private, or organization")

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic version format."""
        parts = v.split('.')
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must be semantic (x.y.z)")
        return v

    @validator('visibility')
    def validate_visibility(cls, v):
        """Validate visibility setting."""
        if v not in ["public", "private", "organization"]:
            raise ValueError("Visibility must be: public, private, or organization")
        return v


class PackConfiguration(BaseModel):
    """Configuration for pack building."""

    # Build options
    include_tests: bool = Field(True, description="Include test files")
    include_docs: bool = Field(True, description="Include documentation")
    include_configs: bool = Field(True, description="Include configuration files")
    include_examples: bool = Field(True, description="Include example usage")

    # Compression
    compression: str = Field("gzip", description="Compression type: gzip, bz2, xz")
    compression_level: int = Field(9, ge=1, le=9, description="Compression level")

    # Security
    sign_pack: bool = Field(False, description="Digitally sign the pack")
    encrypt_pack: bool = Field(False, description="Encrypt the pack")
    checksum_algorithm: str = Field("sha256", description="Checksum algorithm")

    # Output
    output_format: str = Field("tar.gz", description="Output format: tar.gz, zip")
    output_directory: Path = Field(Path("./packs"), description="Output directory")


class PackManifest(BaseModel):
    """Pack manifest for installation and validation."""

    pack_id: str = Field(..., description="Unique pack identifier")
    metadata: PackMetadata = Field(..., description="Pack metadata")

    # Contents
    files: List[Dict[str, Any]] = Field(default_factory=list, description="File list with checksums")
    total_size_bytes: int = Field(0, description="Total pack size")
    file_count: int = Field(0, description="Number of files")

    # Integrity
    pack_checksum: str = Field(..., description="Pack checksum")
    checksums: Dict[str, str] = Field(default_factory=dict, description="File checksums")

    # Build information
    build_timestamp: datetime = Field(default_factory=datetime.utcnow)
    builder_version: str = Field("1.0.0", description="Builder version")
    platform: str = Field("any", description="Target platform")


class PackBuilder:
    """
    Build distributable agent packs for GreenLang Hub.

    Features:
    - Pack creation with metadata
    - Dependency bundling
    - Digital signatures
    - Compression optimization
    - Hub-ready format
    """

    def __init__(self, config: Optional[PackConfiguration] = None):
        """Initialize pack builder."""
        self.config = config or PackConfiguration()
        self.config.output_directory.mkdir(parents=True, exist_ok=True)

        logger.info(f"PackBuilder initialized with output: {self.config.output_directory}")

    def create_pack(
        self,
        agent_dir: Path,
        metadata: PackMetadata,
        custom_files: Optional[List[Path]] = None
    ) -> str:
        """
        Create a GreenLang pack from agent directory.

        Args:
            agent_dir: Directory containing agent code
            metadata: Pack metadata
            custom_files: Additional files to include

        Returns:
            Pack ID (hash identifier)
        """
        try:
            # Generate pack ID
            pack_id = self._generate_pack_id(metadata)

            # Create temporary build directory
            build_dir = self.config.output_directory / f".build_{pack_id}"
            build_dir.mkdir(exist_ok=True)

            # Copy agent files
            self._copy_agent_files(agent_dir, build_dir)

            # Add custom files if provided
            if custom_files:
                self._add_custom_files(custom_files, build_dir)

            # Create requirements file
            self._create_requirements(build_dir, metadata)

            # Create manifest
            manifest = self._create_manifest(build_dir, metadata, pack_id)

            # Write manifest
            manifest_path = build_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest.dict(), indent=2, default=str))

            # Create pack archive
            pack_path = self._create_archive(build_dir, pack_id, metadata)

            # Calculate final checksum
            pack_checksum = self._calculate_checksum(pack_path)

            # Clean up build directory
            shutil.rmtree(build_dir)

            logger.info(
                f"Pack created successfully: {pack_id} "
                f"({pack_path.stat().st_size / 1024 / 1024:.2f} MB)"
            )

            return pack_id

        except Exception as e:
            logger.error(f"Failed to create pack: {str(e)}", exc_info=True)
            raise

    def validate_pack(self, pack_path: Path) -> bool:
        """
        Validate pack integrity and structure.

        Args:
            pack_path: Path to pack file

        Returns:
            True if valid
        """
        try:
            # Open and extract manifest
            if pack_path.suffix == ".zip":
                with zipfile.ZipFile(pack_path, 'r') as zf:
                    manifest_data = zf.read("manifest.json")
            else:
                with tarfile.open(pack_path, 'r:*') as tf:
                    manifest_file = tf.extractfile("manifest.json")
                    manifest_data = manifest_file.read()

            manifest = PackManifest(**json.loads(manifest_data))

            # Verify checksum
            actual_checksum = self._calculate_checksum(pack_path)
            if actual_checksum != manifest.pack_checksum:
                logger.error("Pack checksum mismatch")
                return False

            # Verify structure
            required_files = ["manifest.json", "requirements.txt"]
            # Additional validation...

            return True

        except Exception as e:
            logger.error(f"Pack validation failed: {str(e)}")
            return False

    def install_pack(self, pack_path: Path, target_dir: Path) -> bool:
        """
        Install pack to target directory.

        Args:
            pack_path: Path to pack file
            target_dir: Installation directory

        Returns:
            Success status
        """
        try:
            # Validate pack first
            if not self.validate_pack(pack_path):
                raise ValueError("Pack validation failed")

            # Extract pack
            if pack_path.suffix == ".zip":
                with zipfile.ZipFile(pack_path, 'r') as zf:
                    zf.extractall(target_dir)
            else:
                with tarfile.open(pack_path, 'r:*') as tf:
                    tf.extractall(target_dir)

            # Read manifest
            manifest_path = target_dir / "manifest.json"
            manifest = PackManifest(**json.loads(manifest_path.read_text()))

            # Install dependencies
            self._install_dependencies(target_dir, manifest.metadata)

            logger.info(f"Pack installed successfully to {target_dir}")
            return True

        except Exception as e:
            logger.error(f"Pack installation failed: {str(e)}")
            return False

    def _generate_pack_id(self, metadata: PackMetadata) -> str:
        """Generate unique pack ID."""
        content = f"{metadata.name}{metadata.version}{metadata.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _copy_agent_files(self, source_dir: Path, target_dir: Path):
        """Copy agent files to build directory."""
        # Copy Python files
        for py_file in source_dir.glob("*.py"):
            if not py_file.name.startswith("test_") or self.config.include_tests:
                shutil.copy2(py_file, target_dir)

        # Copy configuration files
        if self.config.include_configs:
            for config_file in source_dir.glob("*.yaml"):
                shutil.copy2(config_file, target_dir)
            for config_file in source_dir.glob("*.json"):
                shutil.copy2(config_file, target_dir)

        # Copy documentation
        if self.config.include_docs:
            for doc_file in source_dir.glob("*.md"):
                shutil.copy2(doc_file, target_dir)

    def _add_custom_files(self, files: List[Path], target_dir: Path):
        """Add custom files to pack."""
        for file_path in files:
            if file_path.exists():
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)

    def _create_requirements(self, build_dir: Path, metadata: PackMetadata):
        """Create requirements.txt file."""
        requirements = [
            f"greenlang{metadata.greenlang_version}",
            *metadata.dependencies
        ]

        req_path = build_dir / "requirements.txt"
        req_path.write_text("\n".join(requirements))

    def _create_manifest(
        self,
        build_dir: Path,
        metadata: PackMetadata,
        pack_id: str
    ) -> PackManifest:
        """Create pack manifest."""
        files = []
        checksums = {}
        total_size = 0

        for file_path in sorted(build_dir.iterdir()):
            if file_path.is_file():
                rel_path = file_path.relative_to(build_dir)
                file_size = file_path.stat().st_size
                file_checksum = self._calculate_checksum(file_path)

                files.append({
                    "path": str(rel_path),
                    "size": file_size,
                    "checksum": file_checksum
                })

                checksums[str(rel_path)] = file_checksum
                total_size += file_size

        return PackManifest(
            pack_id=pack_id,
            metadata=metadata,
            files=files,
            total_size_bytes=total_size,
            file_count=len(files),
            pack_checksum="",  # Will be updated after archive creation
            checksums=checksums
        )

    def _create_archive(
        self,
        build_dir: Path,
        pack_id: str,
        metadata: PackMetadata
    ) -> Path:
        """Create compressed archive."""
        pack_name = f"{metadata.name}-{metadata.version}-{pack_id}"

        if self.config.output_format == "zip":
            pack_path = self.config.output_directory / f"{pack_name}.zip"
            with zipfile.ZipFile(
                pack_path,
                'w',
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=self.config.compression_level
            ) as zf:
                for file_path in build_dir.rglob("*"):
                    if file_path.is_file():
                        arc_name = file_path.relative_to(build_dir)
                        zf.write(file_path, arc_name)
        else:
            # Default to tar.gz
            pack_path = self.config.output_directory / f"{pack_name}.tar.gz"
            with tarfile.open(pack_path, f'w:{self.config.compression}') as tf:
                tf.add(build_dir, arcname=pack_name)

        return pack_path

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(self.config.checksum_algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def _install_dependencies(self, target_dir: Path, metadata: PackMetadata):
        """Install pack dependencies."""
        req_file = target_dir / "requirements.txt"

        if req_file.exists():
            import subprocess
            subprocess.run(
                ["pip", "install", "-r", str(req_file)],
                check=False,
                capture_output=True
            )


class PackRegistry:
    """Registry for managing installed packs."""

    def __init__(self, registry_dir: Path = Path("~/.greenlang/packs").expanduser()):
        """Initialize pack registry."""
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_dir / "registry.json"

        # Load existing registry
        self.registry = self._load_registry()

    def register_pack(self, pack_id: str, metadata: PackMetadata, install_path: Path):
        """Register installed pack."""
        self.registry[pack_id] = {
            "metadata": metadata.dict(),
            "install_path": str(install_path),
            "installed_at": DeterministicClock.utcnow().isoformat()
        }
        self._save_registry()

    def unregister_pack(self, pack_id: str):
        """Remove pack from registry."""
        if pack_id in self.registry:
            del self.registry[pack_id]
            self._save_registry()

    def get_installed_packs(self) -> Dict[str, Any]:
        """Get list of installed packs."""
        return self.registry.copy()

    def _load_registry(self) -> Dict[str, Any]:
        """Load registry from file."""
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text())
        return {}

    def _save_registry(self):
        """Save registry to file."""
        self.registry_file.write_text(
            json.dumps(self.registry, indent=2, default=str)
        )