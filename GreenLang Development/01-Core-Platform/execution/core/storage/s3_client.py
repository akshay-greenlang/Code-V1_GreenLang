# -*- coding: utf-8 -*-
"""
S3 Storage Client for GreenLang Artifact Management
====================================================

INFRA-006: Implementation of boto3 S3 storage client for artifact management.

This module provides:
    - S3 bucket operations (upload, download, delete, list)
    - Multipart upload for large files
    - Automatic retry with exponential backoff
    - Server-side encryption (SSE-S3/SSE-KMS)
    - Presigned URL generation for temporary access

Author: GreenLang Team
Version: 1.0.0
"""

import io
import logging
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, Union

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


class S3StorageConfig:
    """Configuration for S3 storage client"""

    def __init__(
        self,
        bucket_name: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        use_ssl: bool = True,
        verify_ssl: bool = True,
        max_retries: int = 3,
        connect_timeout: int = 10,
        read_timeout: int = 30,
        multipart_threshold: int = 8 * 1024 * 1024,  # 8MB
        multipart_chunksize: int = 8 * 1024 * 1024,  # 8MB
    ):
        """
        Initialize S3 storage configuration.

        Args:
            bucket_name: S3 bucket name
            region: AWS region
            endpoint_url: Custom endpoint URL (for S3-compatible services)
            access_key_id: AWS access key ID (uses environment/IAM role if not provided)
            secret_access_key: AWS secret access key
            kms_key_id: KMS key ID for SSE-KMS encryption
            use_ssl: Whether to use SSL
            verify_ssl: Whether to verify SSL certificates
            max_retries: Maximum number of retries for failed requests
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            multipart_threshold: Size threshold for multipart uploads
            multipart_chunksize: Chunk size for multipart uploads
        """
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.kms_key_id = kms_key_id
        self.use_ssl = use_ssl
        self.verify_ssl = verify_ssl
        self.max_retries = max_retries
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.multipart_threshold = multipart_threshold
        self.multipart_chunksize = multipart_chunksize


class S3StorageClient:
    """
    S3 storage client for GreenLang artifact management.

    Provides upload, download, delete, and list operations with
    automatic retry, encryption, and multipart upload support.
    """

    def __init__(self, config: S3StorageConfig):
        """
        Initialize S3 storage client.

        Args:
            config: S3 storage configuration
        """
        self.config = config
        self._client: Optional[Any] = None
        self._resource: Optional[Any] = None

    @property
    def client(self) -> Any:
        """Get or create boto3 S3 client (lazy initialization)"""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def resource(self) -> Any:
        """Get or create boto3 S3 resource (lazy initialization)"""
        if self._resource is None:
            self._resource = self._create_resource()
        return self._resource

    def _create_client(self) -> Any:
        """Create boto3 S3 client with retry configuration"""
        boto_config = Config(
            retries={
                "max_attempts": self.config.max_retries,
                "mode": "adaptive",
            },
            connect_timeout=self.config.connect_timeout,
            read_timeout=self.config.read_timeout,
        )

        client_kwargs: Dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.config.region,
            "config": boto_config,
            "use_ssl": self.config.use_ssl,
            "verify": self.config.verify_ssl,
        }

        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url

        if self.config.access_key_id and self.config.secret_access_key:
            client_kwargs["aws_access_key_id"] = self.config.access_key_id
            client_kwargs["aws_secret_access_key"] = self.config.secret_access_key

        return boto3.client(**client_kwargs)

    def _create_resource(self) -> Any:
        """Create boto3 S3 resource"""
        resource_kwargs: Dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.config.region,
            "use_ssl": self.config.use_ssl,
            "verify": self.config.verify_ssl,
        }

        if self.config.endpoint_url:
            resource_kwargs["endpoint_url"] = self.config.endpoint_url

        if self.config.access_key_id and self.config.secret_access_key:
            resource_kwargs["aws_access_key_id"] = self.config.access_key_id
            resource_kwargs["aws_secret_access_key"] = self.config.secret_access_key

        return boto3.resource(**resource_kwargs)

    def _get_encryption_args(self) -> Dict[str, str]:
        """Get encryption arguments for S3 operations"""
        if self.config.kms_key_id:
            return {
                "ServerSideEncryption": "aws:kms",
                "SSEKMSKeyId": self.config.kms_key_id,
            }
        return {"ServerSideEncryption": "AES256"}

    def exists(self, key: str) -> bool:
        """
        Check if an object exists in S3.

        Args:
            key: S3 object key

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.head_object(Bucket=self.config.bucket_name, Key=key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            logger.error(f"Error checking object existence: {e}")
            raise

    def upload(
        self,
        key: str,
        content: Union[bytes, str, BinaryIO, Path],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> bool:
        """
        Upload content to S3.

        Args:
            key: S3 object key
            content: Content to upload (bytes, string, file-like object, or path)
            content_type: MIME type of the content
            metadata: Additional metadata to store with the object

        Returns:
            True if upload successful
        """
        try:
            extra_args: Dict[str, Any] = self._get_encryption_args()

            if content_type:
                extra_args["ContentType"] = content_type

            if metadata:
                extra_args["Metadata"] = metadata

            # Handle different content types
            if isinstance(content, Path):
                # File path - use multipart upload for large files
                file_size = content.stat().st_size
                if file_size > self.config.multipart_threshold:
                    return self._multipart_upload(key, content, extra_args)
                with open(content, "rb") as f:
                    self.client.put_object(
                        Bucket=self.config.bucket_name,
                        Key=key,
                        Body=f.read(),
                        **extra_args,
                    )
            elif isinstance(content, str):
                # String content
                self.client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=content.encode("utf-8"),
                    **extra_args,
                )
            elif isinstance(content, bytes):
                # Bytes content
                self.client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=content,
                    **extra_args,
                )
            else:
                # File-like object
                self.client.put_object(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    Body=content.read(),
                    **extra_args,
                )

            logger.info(f"Successfully uploaded {key} to S3")
            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to upload {key} to S3: {e}")
            raise

    def _multipart_upload(
        self, key: str, file_path: Path, extra_args: Dict[str, Any]
    ) -> bool:
        """
        Perform multipart upload for large files.

        Args:
            key: S3 object key
            file_path: Path to file to upload
            extra_args: Additional arguments for upload

        Returns:
            True if upload successful
        """
        try:
            # Create multipart upload
            response = self.client.create_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=key,
                **extra_args,
            )
            upload_id = response["UploadId"]

            parts: List[Dict[str, Any]] = []
            part_number = 1

            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(self.config.multipart_chunksize)
                    if not chunk:
                        break

                    # Upload part
                    part_response = self.client.upload_part(
                        Bucket=self.config.bucket_name,
                        Key=key,
                        UploadId=upload_id,
                        PartNumber=part_number,
                        Body=chunk,
                    )

                    parts.append({
                        "PartNumber": part_number,
                        "ETag": part_response["ETag"],
                    })
                    part_number += 1

            # Complete multipart upload
            self.client.complete_multipart_upload(
                Bucket=self.config.bucket_name,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(f"Successfully completed multipart upload for {key}")
            return True

        except (BotoCoreError, ClientError) as e:
            # Abort multipart upload on failure
            try:
                self.client.abort_multipart_upload(
                    Bucket=self.config.bucket_name,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                pass
            logger.error(f"Failed multipart upload for {key}: {e}")
            raise

    def download(self, key: str) -> bytes:
        """
        Download content from S3.

        Args:
            key: S3 object key

        Returns:
            Downloaded content as bytes
        """
        try:
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )
            content = response["Body"].read()
            logger.info(f"Successfully downloaded {key} from S3")
            return content

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to download {key} from S3: {e}")
            raise

    def download_to_file(self, key: str, file_path: Path) -> bool:
        """
        Download S3 object to a file.

        Args:
            key: S3 object key
            file_path: Local path to save file

        Returns:
            True if download successful
        """
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                Bucket=self.config.bucket_name,
                Key=key,
                Filename=str(file_path),
            )
            logger.info(f"Successfully downloaded {key} to {file_path}")
            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to download {key} to {file_path}: {e}")
            raise

    def delete(self, key: str) -> bool:
        """
        Delete an object from S3.

        Args:
            key: S3 object key

        Returns:
            True if deletion successful
        """
        try:
            self.client.delete_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )
            logger.info(f"Successfully deleted {key} from S3")
            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete {key} from S3: {e}")
            raise

    def delete_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Delete multiple objects from S3.

        Args:
            keys: List of S3 object keys to delete

        Returns:
            Dictionary with 'Deleted' and 'Errors' lists
        """
        try:
            objects = [{"Key": key} for key in keys]
            response = self.client.delete_objects(
                Bucket=self.config.bucket_name,
                Delete={"Objects": objects, "Quiet": False},
            )
            deleted = response.get("Deleted", [])
            errors = response.get("Errors", [])

            logger.info(f"Deleted {len(deleted)} objects, {len(errors)} errors")
            return {"Deleted": deleted, "Errors": errors}

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to delete multiple objects: {e}")
            raise

    def list_objects(
        self,
        prefix: str = "",
        max_keys: int = 1000,
        continuation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List objects in S3 bucket.

        Args:
            prefix: Key prefix to filter objects
            max_keys: Maximum number of keys to return
            continuation_token: Token for pagination

        Returns:
            Dictionary with 'Contents', 'IsTruncated', and 'NextContinuationToken'
        """
        try:
            kwargs: Dict[str, Any] = {
                "Bucket": self.config.bucket_name,
                "Prefix": prefix,
                "MaxKeys": max_keys,
            }

            if continuation_token:
                kwargs["ContinuationToken"] = continuation_token

            response = self.client.list_objects_v2(**kwargs)

            return {
                "Contents": response.get("Contents", []),
                "IsTruncated": response.get("IsTruncated", False),
                "NextContinuationToken": response.get("NextContinuationToken"),
            }

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to list objects with prefix {prefix}: {e}")
            raise

    def get_object_metadata(self, key: str) -> Dict[str, Any]:
        """
        Get metadata for an S3 object.

        Args:
            key: S3 object key

        Returns:
            Object metadata dictionary
        """
        try:
            response = self.client.head_object(
                Bucket=self.config.bucket_name,
                Key=key,
            )
            return {
                "ContentLength": response.get("ContentLength"),
                "ContentType": response.get("ContentType"),
                "LastModified": response.get("LastModified"),
                "ETag": response.get("ETag"),
                "Metadata": response.get("Metadata", {}),
            }

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to get metadata for {key}: {e}")
            raise

    def generate_presigned_url(
        self,
        key: str,
        operation: str = "get_object",
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for temporary access.

        Args:
            key: S3 object key
            operation: Operation to presign ('get_object' or 'put_object')
            expires_in: URL expiration time in seconds

        Returns:
            Presigned URL string
        """
        try:
            url = self.client.generate_presigned_url(
                ClientMethod=operation,
                Params={
                    "Bucket": self.config.bucket_name,
                    "Key": key,
                },
                ExpiresIn=expires_in,
            )
            logger.info(f"Generated presigned URL for {key}")
            return url

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            raise

    def copy(self, source_key: str, dest_key: str) -> bool:
        """
        Copy an object within S3.

        Args:
            source_key: Source object key
            dest_key: Destination object key

        Returns:
            True if copy successful
        """
        try:
            copy_source = {
                "Bucket": self.config.bucket_name,
                "Key": source_key,
            }
            extra_args = self._get_encryption_args()

            self.client.copy_object(
                Bucket=self.config.bucket_name,
                Key=dest_key,
                CopySource=copy_source,
                **extra_args,
            )
            logger.info(f"Successfully copied {source_key} to {dest_key}")
            return True

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Failed to copy {source_key} to {dest_key}: {e}")
            raise

    def get_bucket_size(self, prefix: str = "") -> int:
        """
        Calculate total size of objects in bucket (or prefix).

        Args:
            prefix: Optional prefix to filter objects

        Returns:
            Total size in bytes
        """
        total_size = 0
        continuation_token = None

        while True:
            result = self.list_objects(
                prefix=prefix,
                max_keys=1000,
                continuation_token=continuation_token,
            )

            for obj in result.get("Contents", []):
                total_size += obj.get("Size", 0)

            if not result.get("IsTruncated"):
                break

            continuation_token = result.get("NextContinuationToken")

        return total_size


# Singleton instance for default S3 client
_default_s3_client: Optional[S3StorageClient] = None


def get_s3_client(
    bucket_name: Optional[str] = None,
    region: Optional[str] = None,
    **kwargs: Any,
) -> S3StorageClient:
    """
    Get or create the default S3 storage client.

    Args:
        bucket_name: S3 bucket name (uses env var GREENLANG_S3_BUCKET if not provided)
        region: AWS region (uses env var AWS_REGION if not provided)
        **kwargs: Additional configuration options

    Returns:
        S3StorageClient instance
    """
    global _default_s3_client

    import os

    if _default_s3_client is None:
        config = S3StorageConfig(
            bucket_name=bucket_name or os.environ.get("GREENLANG_S3_BUCKET", "greenlang-artifacts"),
            region=region or os.environ.get("AWS_REGION", "us-east-1"),
            kms_key_id=os.environ.get("GREENLANG_KMS_KEY_ID"),
            **kwargs,
        )
        _default_s3_client = S3StorageClient(config)

    return _default_s3_client


def set_s3_client(client: S3StorageClient) -> None:
    """Set the default S3 storage client instance."""
    global _default_s3_client
    _default_s3_client = client
