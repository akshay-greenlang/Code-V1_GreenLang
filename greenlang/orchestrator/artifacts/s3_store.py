# -*- coding: utf-8 -*-
"""
S3 Artifact Store Implementation
=================================

AWS S3 / S3-compatible artifact storage for GLIP v1.

Supports:
    - AWS S3
    - MinIO (local development)
    - Any S3-compatible storage

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import io
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, BinaryIO, Dict, List, Optional, Union

# Optional boto3 dependency
try:
    import aioboto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    AIOBOTO3_AVAILABLE = True
except ImportError:
    aioboto3 = None
    Config = None
    ClientError = Exception
    AIOBOTO3_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "aioboto3 not available. Install with: pip install aioboto3"
    )

from pydantic import Field

from greenlang.orchestrator.artifacts.base import (
    ArtifactManifest,
    ArtifactMetadata,
    ArtifactStore,
    ArtifactType,
    RetentionPolicy,
)
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class S3Config:
    """Configuration for S3 artifact store."""

    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        use_ssl: bool = True,
        verify_ssl: bool = True,
        max_pool_connections: int = 50,
        connect_timeout: int = 10,
        read_timeout: int = 30,
    ):
        """
        Initialize S3 configuration.

        Args:
            bucket: S3 bucket name
            region: AWS region
            endpoint_url: Custom endpoint (for MinIO)
            access_key_id: AWS access key (None = use default credentials)
            secret_access_key: AWS secret key
            use_ssl: Whether to use SSL
            verify_ssl: Whether to verify SSL certificates
            max_pool_connections: Connection pool size
            connect_timeout: Connection timeout seconds
            read_timeout: Read timeout seconds
        """
        self.bucket = bucket
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl
        self.max_pool_connections = max_pool_connections
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def get_boto_config(self) -> Config:
        """Get boto3 config object."""
        return Config(
            max_pool_connections=self.max_pool_connections,
            connect_timeout=self.connect_timeout,
            read_timeout=self.read_timeout,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )


class S3ArtifactStore(ArtifactStore):
    """
    S3-based artifact store for GLIP v1.

    Implements the ArtifactStore interface using AWS S3 or S3-compatible storage.

    Usage:
        config = S3Config(bucket="greenlang-artifacts", region="us-west-2")
        store = S3ArtifactStore(config)

        # Write input context
        metadata = await store.write_input_context(
            run_id="run_123",
            step_id="step_456",
            context=run_context.model_dump(),
            tenant_id="acme"
        )

        # Read result
        result = await store.read_result(
            run_id="run_123",
            step_id="step_456",
            tenant_id="acme"
        )
    """

    def __init__(self, config: S3Config):
        """
        Initialize S3 artifact store.

        Args:
            config: S3 configuration
        """
        self.config = config
        self._session = aioboto3.Session()
        logger.info(f"Initialized S3ArtifactStore: bucket={config.bucket}, region={config.region}")

    def _get_client_kwargs(self) -> Dict[str, Any]:
        """Get kwargs for S3 client creation."""
        kwargs = {
            "region_name": self.config.region,
            "config": self.config.get_boto_config(),
            "use_ssl": self.config.use_ssl,
            "verify": self.config.verify_ssl,
        }

        if self.config.endpoint_url:
            kwargs["endpoint_url"] = self.config.endpoint_url

        if self.config.access_key_id and self.config.secret_access_key:
            kwargs["aws_access_key_id"] = self.config.access_key_id
            kwargs["aws_secret_access_key"] = self.config.secret_access_key

        return kwargs

    def _get_key(
        self,
        tenant_id: str,
        run_id: str,
        step_id: str,
        name: str,
    ) -> str:
        """Generate S3 object key."""
        return f"{tenant_id}/runs/{run_id}/steps/{step_id}/{name}"

    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum."""
        return hashlib.sha256(data).hexdigest()

    async def write_input_context(
        self,
        run_id: str,
        step_id: str,
        context: Dict[str, Any],
        tenant_id: str,
    ) -> ArtifactMetadata:
        """Write input.json (RunContext) to S3."""
        data = json.dumps(context, sort_keys=True, default=str, indent=2).encode("utf-8")
        key = self._get_key(tenant_id, run_id, step_id, "input.json")
        checksum = self._compute_checksum(data)

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType="application/json",
                Metadata={
                    "checksum": checksum,
                    "run_id": run_id,
                    "step_id": step_id,
                    "artifact_type": ArtifactType.INPUT_CONTEXT.value,
                },
            )

        uri = f"s3://{self.config.bucket}/{key}"
        logger.debug(f"Wrote input context: {uri}")

        return ArtifactMetadata(
            artifact_id=str(uuid.uuid4()),
            uri=uri,
            artifact_type=ArtifactType.INPUT_CONTEXT,
            run_id=run_id,
            step_id=step_id,
            name="input.json",
            checksum=checksum,
            size_bytes=len(data),
            media_type="application/json",
            created_at=DeterministicClock.now(),
        )

    async def read_result(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read result.json from S3."""
        key = self._get_key(tenant_id, run_id, step_id, "result.json")

        try:
            async with self._session.client("s3", **self._get_client_kwargs()) as s3:
                response = await s3.get_object(
                    Bucket=self.config.bucket,
                    Key=key,
                )
                body = await response["Body"].read()
                return json.loads(body.decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Result not found: {key}")
                return None
            raise

    async def read_step_metadata(
        self,
        run_id: str,
        step_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Read step_metadata.json from S3."""
        key = self._get_key(tenant_id, run_id, step_id, "step_metadata.json")

        try:
            async with self._session.client("s3", **self._get_client_kwargs()) as s3:
                response = await s3.get_object(
                    Bucket=self.config.bucket,
                    Key=key,
                )
                body = await response["Body"].read()
                return json.loads(body.decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Step metadata not found: {key}")
                return None
            raise

    async def list_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> ArtifactManifest:
        """List all artifacts for a run or step."""
        if tenant_id and step_id:
            prefix = f"{tenant_id}/runs/{run_id}/steps/{step_id}/"
        elif tenant_id:
            prefix = f"{tenant_id}/runs/{run_id}/"
        else:
            prefix = f"runs/{run_id}/"

        artifacts = []
        total_size = 0

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=self.config.bucket,
                Prefix=prefix,
            ):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    size = obj["Size"]
                    total_size += size

                    # Parse key to extract components
                    parts = key.split("/")
                    if len(parts) >= 5:
                        artifact_tenant = parts[0]
                        artifact_run = parts[2]
                        artifact_step = parts[4]
                        artifact_name = "/".join(parts[5:])
                    else:
                        continue

                    # Determine artifact type
                    if artifact_name == "input.json":
                        artifact_type = ArtifactType.INPUT_CONTEXT
                    elif artifact_name == "result.json":
                        artifact_type = ArtifactType.RESULT
                    elif artifact_name == "step_metadata.json":
                        artifact_type = ArtifactType.METADATA
                    elif artifact_name == "error.json":
                        artifact_type = ArtifactType.ERROR
                    else:
                        artifact_type = ArtifactType.DATA

                    # Get object metadata for checksum
                    head_response = await s3.head_object(
                        Bucket=self.config.bucket,
                        Key=key,
                    )
                    checksum = head_response.get("Metadata", {}).get("checksum", "")

                    artifacts.append(ArtifactMetadata(
                        artifact_id=str(uuid.uuid4()),
                        uri=f"s3://{self.config.bucket}/{key}",
                        artifact_type=artifact_type,
                        run_id=artifact_run,
                        step_id=artifact_step,
                        name=artifact_name,
                        checksum=checksum,
                        size_bytes=size,
                        media_type=head_response.get("ContentType", "application/octet-stream"),
                        created_at=obj.get("LastModified", DeterministicClock.now()),
                    ))

        # Compute manifest checksum
        manifest_content = json.dumps(
            [a.model_dump() for a in artifacts],
            sort_keys=True,
            default=str
        )
        manifest_checksum = self._compute_checksum(manifest_content.encode())

        return ArtifactManifest(
            run_id=run_id,
            step_id=step_id,
            artifacts=artifacts,
            total_size_bytes=total_size,
            artifact_count=len(artifacts),
            manifest_checksum=manifest_checksum,
        )

    async def write_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        data: Union[bytes, BinaryIO],
        media_type: str,
        tenant_id: str,
    ) -> ArtifactMetadata:
        """Write an arbitrary artifact file."""
        if isinstance(data, BinaryIO):
            data = data.read()

        key = self._get_key(tenant_id, run_id, step_id, f"artifacts/{name}")
        checksum = self._compute_checksum(data)

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            await s3.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType=media_type,
                Metadata={
                    "checksum": checksum,
                    "run_id": run_id,
                    "step_id": step_id,
                    "artifact_type": ArtifactType.DATA.value,
                },
            )

        uri = f"s3://{self.config.bucket}/{key}"
        logger.debug(f"Wrote artifact: {uri}")

        return ArtifactMetadata(
            artifact_id=str(uuid.uuid4()),
            uri=uri,
            artifact_type=ArtifactType.DATA,
            run_id=run_id,
            step_id=step_id,
            name=name,
            checksum=checksum,
            size_bytes=len(data),
            media_type=media_type,
            created_at=DeterministicClock.now(),
        )

    async def read_artifact(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
    ) -> Optional[bytes]:
        """Read an artifact file."""
        key = self._get_key(tenant_id, run_id, step_id, f"artifacts/{name}")

        try:
            async with self._session.client("s3", **self._get_client_kwargs()) as s3:
                response = await s3.get_object(
                    Bucket=self.config.bucket,
                    Key=key,
                )
                return await response["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug(f"Artifact not found: {key}")
                return None
            raise

    async def get_presigned_url(
        self,
        run_id: str,
        step_id: str,
        name: str,
        tenant_id: str,
        expires_in_seconds: int = 3600,
    ) -> str:
        """Get a presigned URL for direct artifact access."""
        key = self._get_key(tenant_id, run_id, step_id, name)

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            url = await s3.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.config.bucket,
                    "Key": key,
                },
                ExpiresIn=expires_in_seconds,
            )
            return url

    async def delete_artifacts(
        self,
        run_id: str,
        step_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> int:
        """Delete artifacts for a run or step."""
        if tenant_id and step_id:
            prefix = f"{tenant_id}/runs/{run_id}/steps/{step_id}/"
        elif tenant_id:
            prefix = f"{tenant_id}/runs/{run_id}/"
        else:
            prefix = f"runs/{run_id}/"

        deleted_count = 0

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=self.config.bucket,
                Prefix=prefix,
            ):
                objects = page.get("Contents", [])
                if objects:
                    await s3.delete_objects(
                        Bucket=self.config.bucket,
                        Delete={
                            "Objects": [{"Key": obj["Key"]} for obj in objects],
                            "Quiet": True,
                        },
                    )
                    deleted_count += len(objects)

        logger.info(f"Deleted {deleted_count} artifacts for {prefix}")
        return deleted_count

    async def verify_checksum(
        self,
        uri: str,
        expected_checksum: str,
    ) -> bool:
        """Verify artifact integrity by checksum."""
        # Parse S3 URI
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}")

        parts = uri[5:].split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI: {uri}")

        bucket, key = parts

        try:
            async with self._session.client("s3", **self._get_client_kwargs()) as s3:
                response = await s3.get_object(
                    Bucket=bucket,
                    Key=key,
                )
                body = await response["Body"].read()
                actual_checksum = self._compute_checksum(body)
                return actual_checksum == expected_checksum
        except ClientError:
            return False

    async def apply_retention_policy(
        self,
        tenant_id: str,
        policy: RetentionPolicy,
    ) -> int:
        """Apply retention policy to artifacts."""
        if policy.legal_hold:
            logger.info(f"Skipping retention for {tenant_id}: legal hold active")
            return 0

        cutoff_date = DeterministicClock.now() - timedelta(days=policy.retention_days)
        prefix = f"{tenant_id}/runs/"
        deleted_count = 0

        async with self._session.client("s3", **self._get_client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")

            async for page in paginator.paginate(
                Bucket=self.config.bucket,
                Prefix=prefix,
            ):
                objects_to_delete = [
                    {"Key": obj["Key"]}
                    for obj in page.get("Contents", [])
                    if obj.get("LastModified", DeterministicClock.now()) < cutoff_date
                ]

                if objects_to_delete:
                    await s3.delete_objects(
                        Bucket=self.config.bucket,
                        Delete={
                            "Objects": objects_to_delete,
                            "Quiet": True,
                        },
                    )
                    deleted_count += len(objects_to_delete)

        logger.info(f"Applied retention policy for {tenant_id}: deleted {deleted_count} artifacts")
        return deleted_count

    async def health_check(self) -> bool:
        """Check S3 connectivity."""
        try:
            async with self._session.client("s3", **self._get_client_kwargs()) as s3:
                await s3.head_bucket(Bucket=self.config.bucket)
                return True
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return False


def create_s3_artifact_store(
    bucket: str,
    region: str = "us-east-1",
    endpoint_url: Optional[str] = None,
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
) -> S3ArtifactStore:
    """
    Factory function to create S3 artifact store.

    Args:
        bucket: S3 bucket name
        region: AWS region
        endpoint_url: Custom endpoint (for MinIO)
        access_key_id: AWS access key (None = use default credentials)
        secret_access_key: AWS secret key

    Returns:
        Configured S3ArtifactStore instance
    """
    config = S3Config(
        bucket=bucket,
        region=region,
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
    return S3ArtifactStore(config)


def create_local_minio_store(
    bucket: str = "greenlang-dev",
    endpoint_url: str = "http://localhost:9000",
    access_key: str = "minioadmin",
    secret_key: str = "minioadmin",
) -> S3ArtifactStore:
    """
    Factory function for local MinIO development store.

    Args:
        bucket: Bucket name
        endpoint_url: MinIO endpoint
        access_key: MinIO access key
        secret_key: MinIO secret key

    Returns:
        S3ArtifactStore configured for local MinIO
    """
    config = S3Config(
        bucket=bucket,
        region="us-east-1",
        endpoint_url=endpoint_url,
        access_key_id=access_key,
        secret_access_key=secret_key,
        use_ssl=False,
        verify_ssl=False,
    )
    return S3ArtifactStore(config)
