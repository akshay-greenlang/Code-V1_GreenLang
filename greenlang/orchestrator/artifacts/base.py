# -*- coding: utf-8 -*-
"""
Artifact Store Base Interface
==============================

Defines the protocol for GLIP v1 artifact storage.

Author: GreenLang Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, BinaryIO, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ArtifactType(str, Enum):
    """Types of artifacts in GLIP v1."""
    INPUT_CONTEXT = "input_context"  # input.json (RunContext)
    RESULT = "result"  # result.json (StepResult)
    METADATA = "metadata"  # step_metadata.json
    DATA = "data"  # artifacts/* files
    ERROR = "error"  # error.json (on failure)


class ArtifactMetadata(BaseModel):
    """Metadata for a stored artifact."""
    artifact_id: str = Field(..., description="Unique artifact identifier")
    uri: str = Field(..., description="Full URI to artifact")
    artifact_type: ArtifactType = Field(..., description="Type of artifact")

    # Identity
    run_id: str = Field(..., description="Run ID")
    step_id: str = Field(..., description="Step ID")
    name: str = Field(..., description="Artifact name/path")

    # Integrity
    checksum: str = Field(..., description="SHA-256 checksum")
    size_bytes: int = Field(..., description="Size in bytes")
    media_type: str = Field(default="application/octet-stream", description="MIME type")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

    # Provenance
    producer_agent_id: Optional[str] = Field(None, description="Agent that produced this")
    producer_agent_version: Optional[str] = Field(None, description="Agent version")

    # Lineage
    consumed_by_steps: List[str] = Field(default_factory=list, description="Steps that consumed this")


class ArtifactManifest(BaseModel):
    """Manifest of all artifacts for a run/step."""
    run_id: str = Field(..., description="Run ID")
    step_id: Optional[str] = Field(None, description="Step ID (None for run-level)")
    artifacts: List[ArtifactMetadata] = Field(default_factory=list, description="List of artifacts")
    total_size_bytes: int = Field(default=0, description="Total size")
    artifact_count: int = Field(default=0, description="Number of artifacts")
    manifest_checksum: str = Field(default="", description="Checksum of manifest")


class RetentionPolicy(BaseModel):
    """Artifact retention policy."""
    retention_days: int = Field(default=90, description="Days to retain artifacts")
    legal_hold: bool = Field(default=False, description="Whether artifacts are under legal hold")
    archive_after_days: Optional[int] = Field(None, description="Days before archiving to cold storage")


class ArtifactStore(ABC):
    """
    Abstract base class for GLIP v1 artifact storage.

    Implementations:
        - S3ArtifactStore: AWS S3 / S3-compatible
        - LocalArtifactStore: Local filesystem (dev)
    """

    @abstractmethod
    async def write_input_context(
        self,
        run_id: str,
        step_id: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> ArtifactMetadata:
        """
        Write input.json (RunContext) to artifact store.

        Args:
            run_id: Run ID
            step_id: Step ID
            context: RunContext as dict
            tenant_id: Tenant ID for path isolation

        Returns:
            ArtifactMetadata for the written file
        """
        pass

    @abstractmethod
    async def read_result(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read result.json from artifact store.

        Args:
            run_id: Run ID
            step_id: Step ID
            tenant_id: Tenant ID

        Returns:
            StepResult as dict, or None if not found
        """
        pass

    @abstractmethod
    async def read_step_metadata(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Read step_metadata.json from artifact store.

        Args:
            run_id: Run ID
            step_id: Step ID
            tenant_id: Tenant ID

        Returns:
            StepMetadata as dict, or None if not found
        """
        pass

    @abstractmethod
    async def list_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> ArtifactManifest:
        """
        List all artifacts for a run or step.

        Args:
            run_id: Run ID
            step_id: Optional step ID filter
            tenant_id: Tenant ID

        Returns:
            ArtifactManifest with all matching artifacts
        """
        pass

    @abstractmethod
    async def write_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        data: Union[bytes, BinaryIO],
        media_type: str,
        tenant_id: str,
    ) -> ArtifactMetadata:
        """
        Write an arbitrary artifact file.

        Args:
            run_id: Run ID
            step_id: Step ID
            name: Artifact name/path
            data: File content
            media_type: MIME type
            tenant_id: Tenant ID

        Returns:
            ArtifactMetadata for the written file
        """
        pass

    @abstractmethod
    async def read_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
    ) -> Optional[bytes]:
        """
        Read an artifact file.

        Args:
            run_id: Run ID
            step_id: Step ID
            name: Artifact name/path
            tenant_id: Tenant ID

        Returns:
            File content as bytes, or None if not found
        """
        pass

    @abstractmethod
    async def get_presigned_url(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """
        Get a presigned URL for direct artifact access.

        Args:
            run_id: Run ID
            step_id: Step ID
            name: Artifact name/path
            tenant_id: Tenant ID
            expires_in_seconds: URL expiration time

        Returns:
            Presigned URL string
        """
        pass

    @abstractmethod
    async def delete_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        """
        Delete artifacts for a run or step.

        Args:
            run_id: Run ID
            step_id: Optional step ID (None = entire run)
            tenant_id: Tenant ID

        Returns:
            Number of artifacts deleted
        """
        pass

    @abstractmethod
    async def verify_checksum(
        self,
        uri: str,
        expected_checksum: str,
    ) -> bool:
        """
        Verify artifact integrity by checksum.

        Args:
            uri: Artifact URI
            expected_checksum: Expected SHA-256 checksum

        Returns:
            True if checksum matches
        """
        pass

    @abstractmethod
    async def apply_retention_policy(
        self,
        tenant_id: str,
        policy: RetentionPolicy,
    ) -> int:
        """
        Apply retention policy to artifacts.

        Args:
            tenant_id: Tenant ID
            policy: Retention policy to apply

        Returns:
            Number of artifacts affected
        """
        pass

    def generate_artifact_uri(
        self,
        bucket: str,
        tenant_id: str,
        run_id: str,
        step_id: str,
        name: str,
    ) -> str:
        """
        Generate standard artifact URI.

        Format: s3://{bucket}/{tenant_id}/runs/{run_id}/steps/{step_id}/{name}
        """
        return f"s3://{bucket}/{tenant_id}/runs/{run_id}/steps/{step_id}/{name}"

    def generate_input_uri(
        self,
        bucket: str,
        tenant_id: str,
        run_id: str,
        step_id: str,
    ) -> str:
        """Generate GL_INPUT_URI for a step."""
        return self.generate_artifact_uri(bucket, tenant_id, run_id, step_id, "input.json")

    def generate_output_uri_prefix(
        self,
        bucket: str,
        tenant_id: str,
        run_id: str,
        step_id: str,
    ) -> str:
        """Generate GL_OUTPUT_URI prefix for a step."""
        return f"s3://{bucket}/{tenant_id}/runs/{run_id}/steps/{step_id}/"
