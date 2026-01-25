# -*- coding: utf-8 -*-
"""
Centralized Artifact Manager for GreenLang

This module provides a unified artifact management system for tracking,
storing, and retrieving artifacts throughout pipeline execution.
"""

import json
import hashlib
import shutil
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import mimetypes
from greenlang.utilities.determinism import DeterministicClock


class ArtifactType(Enum):
    """Types of artifacts in the system"""

    INPUT = "input"
    OUTPUT = "output"
    LOG = "log"
    REPORT = "report"
    SBOM = "sbom"
    PROVENANCE = "provenance"
    MODEL = "model"
    DATASET = "dataset"
    CONFIG = "config"
    CHECKPOINT = "checkpoint"
    METADATA = "metadata"
    UNKNOWN = "unknown"


class ArtifactStorage(Enum):
    """Storage backends for artifacts"""

    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact"""

    artifact_id: str
    name: str
    artifact_type: ArtifactType
    size: int
    hash: str
    created_at: datetime
    created_by: Optional[str] = None
    pipeline_id: Optional[str] = None
    run_id: Optional[str] = None
    step_name: Optional[str] = None
    mime_type: Optional[str] = None
    encoding: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)
    parent_artifacts: List[str] = field(default_factory=list)
    child_artifacts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        data = asdict(self)
        data["artifact_type"] = self.artifact_type.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactMetadata":
        """Create metadata from dictionary"""
        data["artifact_type"] = ArtifactType(data["artifact_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Artifact:
    """Represents an artifact in the system"""

    metadata: ArtifactMetadata
    storage_path: Path
    storage_type: ArtifactStorage = ArtifactStorage.LOCAL

    def exists(self) -> bool:
        """Check if artifact exists in storage"""
        if self.storage_type == ArtifactStorage.LOCAL:
            return self.storage_path.exists()

        # NOTE: Cloud storage implementation pending
        # When implementing S3/Azure/GCS:
        # if self.storage_type == ArtifactStorage.S3:
        #     import boto3
        #     s3 = boto3.client('s3')
        #     try:
        #         s3.head_object(Bucket=bucket, Key=key)
        #         return True
        #     except:
        #         return False
        # elif self.storage_type == ArtifactStorage.AZURE:
        #     from azure.storage.blob import BlobServiceClient
        #     blob_client = BlobServiceClient.from_connection_string(conn_str)
        #     return blob_client.get_blob_client(container, blob).exists()

        return False

    def get_size(self) -> int:
        """Get size of artifact"""
        if self.storage_type == ArtifactStorage.LOCAL and self.exists():
            return self.storage_path.stat().st_size
        return self.metadata.size

    def read(self, mode: str = "rb") -> Union[str, bytes]:
        """Read artifact content"""
        if self.storage_type == ArtifactStorage.LOCAL:
            with open(self.storage_path, mode) as f:
                return f.read()

        # NOTE: Cloud storage read implementation pending
        # When implementing:
        # if self.storage_type == ArtifactStorage.S3:
        #     import boto3
        #     s3 = boto3.client('s3')
        #     obj = s3.get_object(Bucket=bucket, Key=key)
        #     return obj['Body'].read()
        # elif self.storage_type == ArtifactStorage.AZURE:
        #     blob_client = BlobServiceClient.from_connection_string(conn_str)
        #     downloader = blob_client.get_blob_client(container, blob).download_blob()
        #     return downloader.readall()

        raise NotImplementedError(f"Storage type {self.storage_type} not implemented")

    def write(self, content: Union[str, bytes], mode: str = "wb") -> None:
        """Write artifact content"""
        if self.storage_type == ArtifactStorage.LOCAL:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, mode) as f:
                f.write(content)
        else:
            # NOTE: Cloud storage write implementation pending
            # When implementing:
            # if self.storage_type == ArtifactStorage.S3:
            #     import boto3
            #     s3 = boto3.client('s3')
            #     s3.put_object(Bucket=bucket, Key=key, Body=content)
            # elif self.storage_type == ArtifactStorage.AZURE:
            #     blob_client = BlobServiceClient.from_connection_string(conn_str)
            #     blob_client.get_blob_client(container, blob).upload_blob(content, overwrite=True)
            # elif self.storage_type == ArtifactStorage.GCS:
            #     from google.cloud import storage
            #     client = storage.Client()
            #     bucket = client.bucket(bucket_name)
            #     blob = bucket.blob(blob_name)
            #     blob.upload_from_string(content)

            raise NotImplementedError(
                f"Storage type {self.storage_type} not implemented"
            )


class ArtifactManager:
    """
    Centralized artifact manager for tracking and managing artifacts
    throughout the pipeline execution lifecycle.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        storage_type: ArtifactStorage = ArtifactStorage.LOCAL,
    ):
        """
        Initialize the artifact manager.

        Args:
            base_dir: Base directory for storing artifacts
            storage_type: Type of storage backend to use
        """
        self.base_dir = base_dir or Path.cwd() / ".greenlang" / "artifacts"
        self.storage_type = storage_type
        self.artifacts: Dict[str, Artifact] = {}
        self.metadata_dir = self.base_dir / ".metadata"

        # Create directories if using local storage
        if storage_type == ArtifactStorage.LOCAL:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Load existing metadata
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load existing artifact metadata from disk"""
        if self.storage_type == ArtifactStorage.LOCAL and self.metadata_dir.exists():
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file, "r") as f:
                        data = json.load(f)
                    metadata = ArtifactMetadata.from_dict(data)
                    storage_path = self.base_dir / data.get(
                        "storage_path", metadata.artifact_id
                    )
                    artifact = Artifact(metadata, storage_path, self.storage_type)
                    self.artifacts[metadata.artifact_id] = artifact
                except Exception as e:
                    print(f"Warning: Could not load metadata from {metadata_file}: {e}")

    def _save_metadata(self, artifact: Artifact) -> None:
        """Save artifact metadata to disk"""
        if self.storage_type == ArtifactStorage.LOCAL:
            metadata_file = self.metadata_dir / f"{artifact.metadata.artifact_id}.json"
            data = artifact.metadata.to_dict()
            data["storage_path"] = str(artifact.storage_path.relative_to(self.base_dir))
            with open(metadata_file, "w") as f:
                json.dump(data, f, indent=2)

    def _calculate_hash(self, content: Union[str, bytes, Path]) -> str:
        """Calculate SHA256 hash of content"""
        hasher = hashlib.sha256()

        if isinstance(content, Path):
            with open(content, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        elif isinstance(content, str):
            hasher.update(content.encode("utf-8"))
        else:
            hasher.update(content)

        return hasher.hexdigest()

    def _generate_artifact_id(
        self, name: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a unique artifact ID"""
        # Create a deterministic ID based on name and context
        id_parts = [name]
        if context:
            if "pipeline_id" in context:
                id_parts.append(context["pipeline_id"])
            if "run_id" in context:
                id_parts.append(context["run_id"])
            if "step_name" in context:
                id_parts.append(context["step_name"])

        id_parts.append(DeterministicClock.utcnow().isoformat())
        id_string = "_".join(id_parts)

        # Create a short hash for the ID
        return hashlib.md5(id_string.encode()).hexdigest()[:12]

    def _detect_artifact_type(
        self, file_path: Path, hint: Optional[ArtifactType] = None
    ) -> ArtifactType:
        """Detect artifact type from file extension and content"""
        if hint:
            return hint

        # Check by extension
        ext = file_path.suffix.lower()
        type_mapping = {
            ".json": ArtifactType.CONFIG,
            ".yaml": ArtifactType.CONFIG,
            ".yml": ArtifactType.CONFIG,
            ".log": ArtifactType.LOG,
            ".txt": ArtifactType.LOG,
            ".pdf": ArtifactType.REPORT,
            ".html": ArtifactType.REPORT,
            ".md": ArtifactType.REPORT,
            ".spdx": ArtifactType.SBOM,
            ".sbom": ArtifactType.SBOM,
            ".csv": ArtifactType.DATASET,
            ".parquet": ArtifactType.DATASET,
            ".pkl": ArtifactType.MODEL,
            ".h5": ArtifactType.MODEL,
            ".ckpt": ArtifactType.CHECKPOINT,
        }

        # Check for SBOM patterns in filename
        if "sbom" in file_path.name.lower():
            return ArtifactType.SBOM
        if "provenance" in file_path.name.lower() or "ledger" in file_path.name.lower():
            return ArtifactType.PROVENANCE

        return type_mapping.get(ext, ArtifactType.UNKNOWN)

    def create_artifact(
        self,
        name: str,
        content: Union[str, bytes, Path],
        artifact_type: Optional[ArtifactType] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Artifact:
        """
        Create and store a new artifact.

        Args:
            name: Name of the artifact
            content: Content of the artifact (string, bytes, or path to file)
            artifact_type: Type of artifact (auto-detected if not provided)
            context: Execution context (pipeline_id, run_id, step_name, etc.)
            tags: Tags to associate with the artifact
            properties: Additional properties

        Returns:
            Created Artifact object
        """
        # Generate artifact ID
        artifact_id = self._generate_artifact_id(name, context)

        # Determine storage path
        storage_subdir = artifact_type.value if artifact_type else "unknown"
        if context and "run_id" in context:
            storage_subdir = f"runs/{context['run_id']}/{storage_subdir}"
        storage_path = self.base_dir / storage_subdir / f"{artifact_id}_{name}"

        # Handle different content types
        if isinstance(content, Path):
            # File path provided
            if not artifact_type:
                artifact_type = self._detect_artifact_type(content)

            # Copy file to storage
            if self.storage_type == ArtifactStorage.LOCAL:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(content, storage_path)

            file_hash = self._calculate_hash(content)
            size = content.stat().st_size
            mime_type, encoding = mimetypes.guess_type(str(content))
        else:
            # Content provided directly
            if not artifact_type:
                artifact_type = ArtifactType.OUTPUT

            # Write content to storage
            if self.storage_type == ArtifactStorage.LOCAL:
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                mode = "w" if isinstance(content, str) else "wb"
                with open(storage_path, mode) as f:
                    f.write(content)

            file_hash = self._calculate_hash(content)
            size = len(content) if isinstance(content, str) else len(content)
            mime_type, encoding = mimetypes.guess_type(name)

        # Create metadata
        metadata = ArtifactMetadata(
            artifact_id=artifact_id,
            name=name,
            artifact_type=artifact_type,
            size=size,
            hash=file_hash,
            created_at=DeterministicClock.utcnow(),
            created_by=context.get("user") if context else None,
            pipeline_id=context.get("pipeline_id") if context else None,
            run_id=context.get("run_id") if context else None,
            step_name=context.get("step_name") if context else None,
            mime_type=mime_type,
            encoding=encoding,
            tags=tags or {},
            properties=properties or {},
        )

        # Create artifact
        artifact = Artifact(metadata, storage_path, self.storage_type)

        # Store artifact
        self.artifacts[artifact_id] = artifact
        self._save_metadata(artifact)

        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Get an artifact by ID"""
        return self.artifacts.get(artifact_id)

    def get_artifacts_by_type(self, artifact_type: ArtifactType) -> List[Artifact]:
        """Get all artifacts of a specific type"""
        return [
            a
            for a in self.artifacts.values()
            if a.metadata.artifact_type == artifact_type
        ]

    def get_artifacts_by_run(self, run_id: str) -> List[Artifact]:
        """Get all artifacts from a specific run"""
        return [a for a in self.artifacts.values() if a.metadata.run_id == run_id]

    def get_artifacts_by_pipeline(self, pipeline_id: str) -> List[Artifact]:
        """Get all artifacts from a specific pipeline"""
        return [
            a for a in self.artifacts.values() if a.metadata.pipeline_id == pipeline_id
        ]

    def get_artifacts_by_step(self, run_id: str, step_name: str) -> List[Artifact]:
        """Get all artifacts from a specific step in a run"""
        return [
            a
            for a in self.artifacts.values()
            if a.metadata.run_id == run_id and a.metadata.step_name == step_name
        ]

    def search_artifacts(
        self,
        name_pattern: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> List[Artifact]:
        """Search for artifacts by name pattern, tags, or properties"""
        results = list(self.artifacts.values())

        # Filter by name pattern
        if name_pattern:
            import re

            pattern = re.compile(name_pattern)
            results = [a for a in results if pattern.search(a.metadata.name)]

        # Filter by tags
        if tags:
            results = [
                a
                for a in results
                if all(a.metadata.tags.get(k) == v for k, v in tags.items())
            ]

        # Filter by properties
        if properties:
            results = [
                a
                for a in results
                if all(a.metadata.properties.get(k) == v for k, v in properties.items())
            ]

        return results

    def link_artifacts(self, parent_id: str, child_id: str) -> bool:
        """Create a parent-child relationship between artifacts"""
        parent = self.get_artifact(parent_id)
        child = self.get_artifact(child_id)

        if not parent or not child:
            return False

        # Update relationships
        if child_id not in parent.metadata.child_artifacts:
            parent.metadata.child_artifacts.append(child_id)
            self._save_metadata(parent)

        if parent_id not in child.metadata.parent_artifacts:
            child.metadata.parent_artifacts.append(parent_id)
            self._save_metadata(child)

        return True

    def get_lineage(
        self, artifact_id: str, direction: str = "both"
    ) -> Dict[str, List[str]]:
        """
        Get artifact lineage (parents and/or children).

        Args:
            artifact_id: ID of the artifact
            direction: "parents", "children", or "both"

        Returns:
            Dictionary with 'parents' and/or 'children' lists
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return {}

        result = {}

        if direction in ["parents", "both"]:
            parents = []
            to_visit = list(artifact.metadata.parent_artifacts)
            visited = set()

            while to_visit:
                parent_id = to_visit.pop(0)
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                parents.append(parent_id)

                parent_artifact = self.get_artifact(parent_id)
                if parent_artifact:
                    to_visit.extend(parent_artifact.metadata.parent_artifacts)

            result["parents"] = parents

        if direction in ["children", "both"]:
            children = []
            to_visit = list(artifact.metadata.child_artifacts)
            visited = set()

            while to_visit:
                child_id = to_visit.pop(0)
                if child_id in visited:
                    continue
                visited.add(child_id)
                children.append(child_id)

                child_artifact = self.get_artifact(child_id)
                if child_artifact:
                    to_visit.extend(child_artifact.metadata.child_artifacts)

            result["children"] = children

        return result

    def delete_artifact(self, artifact_id: str) -> bool:
        """Delete an artifact"""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return False

        # Remove from storage
        if (
            self.storage_type == ArtifactStorage.LOCAL
            and artifact.storage_path.exists()
        ):
            artifact.storage_path.unlink()

        # Remove metadata
        metadata_file = self.metadata_dir / f"{artifact_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()

        # Remove from cache
        del self.artifacts[artifact_id]

        # Update relationships
        for parent_id in artifact.metadata.parent_artifacts:
            parent = self.get_artifact(parent_id)
            if parent and artifact_id in parent.metadata.child_artifacts:
                parent.metadata.child_artifacts.remove(artifact_id)
                self._save_metadata(parent)

        for child_id in artifact.metadata.child_artifacts:
            child = self.get_artifact(child_id)
            if child and artifact_id in child.metadata.parent_artifacts:
                child.metadata.parent_artifacts.remove(artifact_id)
                self._save_metadata(child)

        return True

    def cleanup_old_artifacts(self, days: int = 30) -> int:
        """Remove artifacts older than specified days"""
        cutoff = DeterministicClock.utcnow().timestamp() - (days * 24 * 60 * 60)
        removed = 0

        for artifact_id in list(self.artifacts.keys()):
            artifact = self.artifacts[artifact_id]
            if artifact.metadata.created_at.timestamp() < cutoff:
                if self.delete_artifact(artifact_id):
                    removed += 1

        return removed

    def export_metadata(self, output_file: Path) -> None:
        """Export all artifact metadata to a JSON file"""
        metadata_list = [a.metadata.to_dict() for a in self.artifacts.values()]
        with open(output_file, "w") as f:
            json.dump(metadata_list, f, indent=2)

    def import_metadata(self, input_file: Path) -> int:
        """Import artifact metadata from a JSON file"""
        with open(input_file, "r") as f:
            metadata_list = json.load(f)

        imported = 0
        for data in metadata_list:
            try:
                metadata = ArtifactMetadata.from_dict(data)
                storage_path = self.base_dir / data.get(
                    "storage_path", metadata.artifact_id
                )
                artifact = Artifact(metadata, storage_path, self.storage_type)
                self.artifacts[metadata.artifact_id] = artifact
                self._save_metadata(artifact)
                imported += 1
            except Exception as e:
                print(
                    f"Warning: Could not import artifact {data.get('artifact_id')}: {e}"
                )

        return imported

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored artifacts"""
        total_size = sum(a.metadata.size for a in self.artifacts.values())
        by_type = {}

        for artifact in self.artifacts.values():
            type_name = artifact.metadata.artifact_type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "size": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["size"] += artifact.metadata.size

        return {
            "total_artifacts": len(self.artifacts),
            "total_size": total_size,
            "by_type": by_type,
            "storage_type": self.storage_type.value,
            "base_directory": str(self.base_dir),
        }


# Global artifact manager instance
_artifact_manager: Optional[ArtifactManager] = None


def get_artifact_manager(base_dir: Optional[Path] = None) -> ArtifactManager:
    """Get or create the global artifact manager instance"""
    global _artifact_manager
    if _artifact_manager is None:
        _artifact_manager = ArtifactManager(base_dir)
    return _artifact_manager


def set_artifact_manager(manager: ArtifactManager) -> None:
    """Set the global artifact manager instance"""
    global _artifact_manager
    _artifact_manager = manager
