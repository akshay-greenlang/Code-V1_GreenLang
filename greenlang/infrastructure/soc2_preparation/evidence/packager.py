# -*- coding: utf-8 -*-
"""
Evidence Packager - SEC-009 Phase 3

Creates structured evidence packages for auditor delivery.

Package Structure:
    evidence_package/
    ├── manifest.json           # Package metadata and integrity hashes
    ├── CC1/                    # Control Environment
    │   ├── CC1.1/
    │   │   ├── evidence.json   # Evidence index for this criterion
    │   │   ├── policies/       # Policy documents
    │   │   ├── screenshots/    # Screenshot evidence
    │   │   └── attestations/   # Signed attestations
    │   └── ...
    ├── CC6/                    # Logical Access Controls
    │   └── ...
    ├── populations/            # Sample populations for testing
    │   ├── access_requests.csv
    │   ├── change_tickets.csv
    │   └── incidents.csv
    └── hashes.sha256           # SHA-256 checksums for all files

Features:
    - Creates auditor-friendly directory structure
    - Generates comprehensive manifest with metadata
    - Calculates SHA-256 hashes for integrity verification
    - Uploads packages to S3 with versioning
    - Generates presigned URLs for auditor access

Example:
    >>> packager = EvidencePackager(config)
    >>> package = await packager.create_package(
    ...     criteria=["CC6.1", "CC6.2", "CC6.3"],
    ...     period=DateRange(start=audit_start, end=audit_end)
    ... )
    >>> s3_key = await packager.upload_to_s3(package)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from greenlang.infrastructure.soc2_preparation.evidence.models import (
    DateRange,
    Evidence,
    EvidencePackage,
    EvidencePackageManifest,
    EvidenceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class EvidencePackagerConfig(BaseModel):
    """Configuration for the evidence packager."""

    # S3 settings
    s3_bucket: str = Field(
        default="greenlang-audit-evidence",
        description="S3 bucket for evidence packages",
    )
    s3_prefix: str = Field(
        default="evidence-packages",
        description="S3 key prefix for packages",
    )
    s3_region: str = Field(default="us-east-1")
    kms_key_id: Optional[str] = Field(
        default=None,
        description="KMS key for server-side encryption",
    )

    # Package settings
    include_raw_content: bool = Field(
        default=True,
        description="Include raw content in evidence files",
    )
    compress_package: bool = Field(
        default=True,
        description="Create compressed ZIP archive",
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum file size in MB",
    )
    presigned_url_expiry_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        le=720,  # 30 days
    )

    # Metadata
    organization_name: str = Field(default="GreenLang")
    audit_firm: Optional[str] = Field(default=None)
    prepared_by: str = Field(default="GreenLang SOC 2 Automation")


# ---------------------------------------------------------------------------
# Evidence Packager
# ---------------------------------------------------------------------------


class EvidencePackager:
    """Creates structured evidence packages for SOC 2 auditors.

    Organizes evidence into a hierarchical directory structure,
    generates manifests with integrity hashes, and uploads
    packages to S3 for secure auditor access.

    Example:
        >>> config = EvidencePackagerConfig(s3_bucket="audit-evidence")
        >>> packager = EvidencePackager(config)
        >>> package = await packager.create_package(
        ...     criteria=["CC6.1", "CC6.2"],
        ...     period=date_range,
        ...     evidence=collected_evidence
        ... )
    """

    def __init__(self, config: EvidencePackagerConfig) -> None:
        """Initialize the evidence packager.

        Args:
            config: Packager configuration.
        """
        self.config = config
        self._s3_client: Any = None

    async def create_package(
        self,
        criteria: List[str],
        period: DateRange,
        evidence: Optional[Dict[str, List[Evidence]]] = None,
        populations: Optional[Dict[str, List[Any]]] = None,
        package_name: Optional[str] = None,
    ) -> EvidencePackage:
        """Create an evidence package for the specified criteria.

        Args:
            criteria: List of SOC 2 criterion IDs to include.
            period: Audit period date range.
            evidence: Dictionary mapping criterion ID to evidence list.
            populations: Dictionary mapping population name to items.
            package_name: Optional custom package name.

        Returns:
            Complete EvidencePackage with manifest and file references.
        """
        if evidence is None:
            evidence = {}
        if populations is None:
            populations = {}

        # Generate package metadata
        package_id = uuid4()
        if package_name is None:
            package_name = f"SOC2_Evidence_{period.start.strftime('%Y%m%d')}_{period.end.strftime('%Y%m%d')}"

        # Create temporary directory for package assembly
        temp_dir = Path(tempfile.mkdtemp(prefix="evidence_package_"))

        try:
            # Build directory structure
            package_dir = temp_dir / package_name
            package_dir.mkdir(parents=True)

            # Track all file hashes
            evidence_hashes: Dict[str, str] = {}
            all_evidence: List[Evidence] = []
            total_size = 0

            # Create criterion directories and evidence files
            for criterion_id in criteria:
                # Parse criterion category (e.g., CC6 from CC6.1)
                category = self._get_category(criterion_id)

                criterion_dir = package_dir / category / criterion_id
                criterion_dir.mkdir(parents=True, exist_ok=True)

                # Create subdirectories for evidence types
                (criterion_dir / "policies").mkdir(exist_ok=True)
                (criterion_dir / "screenshots").mkdir(exist_ok=True)
                (criterion_dir / "attestations").mkdir(exist_ok=True)
                (criterion_dir / "logs").mkdir(exist_ok=True)
                (criterion_dir / "configurations").mkdir(exist_ok=True)

                # Write evidence for this criterion
                criterion_evidence = evidence.get(criterion_id, [])
                all_evidence.extend(criterion_evidence)

                evidence_index: List[Dict[str, Any]] = []

                for idx, item in enumerate(criterion_evidence):
                    # Determine target directory based on evidence type
                    subdir = self._get_evidence_subdir(item.evidence_type)
                    target_dir = criterion_dir / subdir

                    # Write evidence file
                    file_name = f"{item.evidence_type.value}_{idx:04d}.json"
                    file_path = target_dir / file_name

                    evidence_data = {
                        "evidence_id": str(item.evidence_id),
                        "title": item.title,
                        "description": item.description,
                        "source": item.source.value,
                        "collected_at": item.collected_at.isoformat(),
                        "period_start": item.period_start.isoformat() if item.period_start else None,
                        "period_end": item.period_end.isoformat() if item.period_end else None,
                        "metadata": item.metadata,
                    }

                    if self.config.include_raw_content and item.content:
                        evidence_data["content"] = item.content

                    file_content = json.dumps(evidence_data, indent=2, default=str)
                    file_path.write_text(file_content)

                    # Calculate hash
                    file_hash = self._calculate_hash(file_content.encode())
                    relative_path = str(file_path.relative_to(package_dir))
                    evidence_hashes[relative_path] = file_hash
                    total_size += len(file_content)

                    evidence_index.append({
                        "evidence_id": str(item.evidence_id),
                        "file": file_name,
                        "title": item.title,
                        "hash": file_hash,
                    })

                # Write criterion evidence index
                index_path = criterion_dir / "evidence.json"
                index_content = json.dumps({
                    "criterion_id": criterion_id,
                    "evidence_count": len(criterion_evidence),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "items": evidence_index,
                }, indent=2)
                index_path.write_text(index_content)

                index_hash = self._calculate_hash(index_content.encode())
                evidence_hashes[str(index_path.relative_to(package_dir))] = index_hash

            # Create populations directory and export CSVs
            populations_dir = package_dir / "populations"
            populations_dir.mkdir(exist_ok=True)

            population_files: Dict[str, str] = {}
            for pop_name, pop_items in populations.items():
                csv_path = populations_dir / f"{pop_name}.csv"
                csv_content = self._export_population_csv(pop_items)
                csv_path.write_text(csv_content)

                csv_hash = self._calculate_hash(csv_content.encode())
                relative_path = str(csv_path.relative_to(package_dir))
                evidence_hashes[relative_path] = csv_hash
                population_files[pop_name] = relative_path
                total_size += len(csv_content)

            # Write hashes file
            hashes_content = self._generate_hashes_file(evidence_hashes)
            hashes_path = package_dir / "hashes.sha256"
            hashes_path.write_text(hashes_content)

            # Calculate package hash (hash of all hashes)
            package_hash = self._calculate_hash(hashes_content.encode())

            # Generate manifest
            manifest = self._generate_manifest(
                package_id=package_id,
                package_name=package_name,
                criteria=criteria,
                period=period,
                evidence_count=len(all_evidence),
                total_size=total_size,
                package_hash=package_hash,
                evidence_hashes=evidence_hashes,
            )

            # Write manifest
            manifest_path = package_dir / "manifest.json"
            manifest_content = json.dumps(manifest.model_dump(), indent=2, default=str)
            manifest_path.write_text(manifest_content)

            # Create ZIP archive if configured
            if self.config.compress_package:
                zip_path = temp_dir / f"{package_name}.zip"
                self._create_zip_archive(package_dir, zip_path)
                archive_path = str(zip_path)
            else:
                archive_path = str(package_dir)

            # Create package result
            package = EvidencePackage(
                manifest=manifest,
                evidence=all_evidence,
                populations=population_files,
                s3_location=None,
                download_url=None,
            )

            # Store archive path for upload
            package.manifest.metadata["archive_path"] = archive_path  # type: ignore

            logger.info(
                f"Created evidence package: {package_name} with "
                f"{len(all_evidence)} evidence items, {total_size} bytes"
            )

            return package

        except Exception as exc:
            logger.error(f"Failed to create evidence package: {exc}")
            raise
        finally:
            # Note: We keep temp_dir for upload, caller should cleanup
            pass

    async def upload_to_s3(
        self,
        package: EvidencePackage,
    ) -> str:
        """Upload evidence package to S3.

        Args:
            package: Evidence package to upload.

        Returns:
            S3 key of the uploaded package.
        """
        try:
            import aioboto3

            session = aioboto3.Session()
            archive_path = package.manifest.metadata.get("archive_path")

            if not archive_path or not Path(archive_path).exists():
                raise ValueError("Package archive not found")

            # Generate S3 key
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            file_ext = ".zip" if archive_path.endswith(".zip") else ""
            s3_key = f"{self.config.s3_prefix}/{timestamp}_{package.manifest.package_id}{file_ext}"

            # Upload to S3
            extra_args: Dict[str, Any] = {
                "ContentType": "application/zip" if file_ext else "application/octet-stream",
                "Metadata": {
                    "package_id": str(package.manifest.package_id),
                    "package_hash": package.manifest.package_hash or "",
                    "evidence_count": str(package.manifest.evidence_count),
                },
            }

            if self.config.kms_key_id:
                extra_args["ServerSideEncryption"] = "aws:kms"
                extra_args["SSEKMSKeyId"] = self.config.kms_key_id
            else:
                extra_args["ServerSideEncryption"] = "AES256"

            async with session.client("s3", region_name=self.config.s3_region) as s3:
                with open(archive_path, "rb") as f:
                    await s3.put_object(
                        Bucket=self.config.s3_bucket,
                        Key=s3_key,
                        Body=f.read(),
                        **extra_args,
                    )

                # Generate presigned URL
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self.config.s3_bucket,
                        "Key": s3_key,
                    },
                    ExpiresIn=self.config.presigned_url_expiry_hours * 3600,
                )

            # Update package with S3 location
            package.s3_location = f"s3://{self.config.s3_bucket}/{s3_key}"
            package.download_url = url

            # Cleanup temp files
            self._cleanup_temp_files(archive_path)

            logger.info(f"Uploaded evidence package to {package.s3_location}")

            return s3_key

        except ImportError:
            logger.error("aioboto3 not available for S3 upload")
            raise
        except Exception as exc:
            logger.error(f"Failed to upload package to S3: {exc}")
            raise

    def _generate_manifest(
        self,
        package_id: UUID,
        package_name: str,
        criteria: List[str],
        period: DateRange,
        evidence_count: int,
        total_size: int,
        package_hash: str,
        evidence_hashes: Dict[str, str],
    ) -> EvidencePackageManifest:
        """Generate package manifest with metadata.

        Args:
            package_id: Unique package identifier.
            package_name: Human-readable package name.
            criteria: List of criteria included.
            period: Audit period.
            evidence_count: Number of evidence items.
            total_size: Total size in bytes.
            package_hash: SHA-256 hash of package.
            evidence_hashes: Map of file paths to hashes.

        Returns:
            Package manifest.
        """
        return EvidencePackageManifest(
            package_id=package_id,
            package_name=package_name,
            audit_period=period,
            criteria=criteria,
            evidence_count=evidence_count,
            total_size_bytes=total_size,
            created_at=datetime.now(timezone.utc),
            created_by=self.config.prepared_by,
            package_hash=package_hash,
            evidence_hashes=evidence_hashes,
            metadata={
                "organization": self.config.organization_name,
                "audit_firm": self.config.audit_firm,
                "package_version": "1.0",
                "schema_version": "SEC-009-v1",
            },
        )

    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content.

        Args:
            content: Content to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(content).hexdigest()

    def _calculate_package_hash(self, package_path: str) -> str:
        """Calculate SHA-256 hash of entire package.

        Args:
            package_path: Path to package directory or archive.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        hasher = hashlib.sha256()
        path = Path(package_path)

        if path.is_file():
            # Hash the archive file
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
        else:
            # Hash all files in directory
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            hasher.update(chunk)

        return hasher.hexdigest()

    def _get_category(self, criterion_id: str) -> str:
        """Extract category from criterion ID.

        Args:
            criterion_id: Criterion ID (e.g., "CC6.1").

        Returns:
            Category prefix (e.g., "CC6").
        """
        # Split on first digit after letters
        for i, c in enumerate(criterion_id):
            if c.isdigit():
                # Find where the point is
                point_idx = criterion_id.find(".", i)
                if point_idx > 0:
                    return criterion_id[:point_idx]
                return criterion_id[:i+1]
        return criterion_id

    def _get_evidence_subdir(self, evidence_type: EvidenceType) -> str:
        """Get subdirectory name for evidence type.

        Args:
            evidence_type: Type of evidence.

        Returns:
            Subdirectory name.
        """
        type_to_subdir = {
            EvidenceType.POLICY: "policies",
            EvidenceType.PROCEDURE: "policies",
            EvidenceType.SCREENSHOT: "screenshots",
            EvidenceType.LOG_EXPORT: "logs",
            EvidenceType.CONFIGURATION: "configurations",
            EvidenceType.ATTESTATION: "attestations",
            EvidenceType.TICKET: "logs",
            EvidenceType.CODE_CHANGE: "logs",
            EvidenceType.ACCESS_REVIEW: "attestations",
            EvidenceType.SECURITY_SCAN: "logs",
            EvidenceType.INCIDENT_REPORT: "logs",
            EvidenceType.TRAINING_RECORD: "attestations",
            EvidenceType.VENDOR_ASSESSMENT: "attestations",
            EvidenceType.PENETRATION_TEST: "logs",
            EvidenceType.AUDIT_REPORT: "attestations",
            EvidenceType.METRIC_EXPORT: "logs",
            EvidenceType.BACKUP_VERIFICATION: "logs",
            EvidenceType.RECOVERY_TEST: "logs",
        }
        return type_to_subdir.get(evidence_type, "logs")

    def _generate_hashes_file(self, hashes: Dict[str, str]) -> str:
        """Generate SHA-256 hashes file content.

        Args:
            hashes: Map of file paths to hashes.

        Returns:
            Hashes file content in standard format.
        """
        lines = []
        for path, hash_value in sorted(hashes.items()):
            lines.append(f"{hash_value}  {path}")
        return "\n".join(lines) + "\n"

    def _export_population_csv(self, items: List[Any]) -> str:
        """Export population items to CSV format.

        Args:
            items: List of items to export.

        Returns:
            CSV content as string.
        """
        if not items:
            return ""

        output = io.StringIO()

        # Determine columns from first item
        if isinstance(items[0], dict):
            fieldnames = list(items[0].keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item)
        elif hasattr(items[0], "__dict__"):
            fieldnames = list(items[0].__dict__.keys())
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for item in items:
                writer.writerow(item.__dict__)
        else:
            # Simple list of values
            writer = csv.writer(output)
            writer.writerow(["value"])
            for item in items:
                writer.writerow([str(item)])

        return output.getvalue()

    def _create_zip_archive(self, source_dir: Path, zip_path: Path) -> None:
        """Create ZIP archive of directory.

        Args:
            source_dir: Source directory to archive.
            zip_path: Path for output ZIP file.
        """
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir.parent)
                    zf.write(file_path, arcname)

        logger.debug(f"Created ZIP archive: {zip_path}")

    def _cleanup_temp_files(self, archive_path: str) -> None:
        """Cleanup temporary files after upload.

        Args:
            archive_path: Path to archive file or directory.
        """
        try:
            path = Path(archive_path)
            if path.is_file():
                # Remove the archive and its parent temp directory
                temp_dir = path.parent
                path.unlink()
                if temp_dir.name.startswith("evidence_package_"):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except Exception as exc:
            logger.warning(f"Failed to cleanup temp files: {exc}")

    async def list_packages(
        self,
        prefix: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List evidence packages in S3.

        Args:
            prefix: Optional prefix filter.
            limit: Maximum number of packages to return.

        Returns:
            List of package metadata dictionaries.
        """
        try:
            import aioboto3

            session = aioboto3.Session()
            packages: List[Dict[str, Any]] = []

            s3_prefix = f"{self.config.s3_prefix}/{prefix}" if prefix else self.config.s3_prefix

            async with session.client("s3", region_name=self.config.s3_region) as s3:
                paginator = s3.get_paginator("list_objects_v2")

                async for page in paginator.paginate(
                    Bucket=self.config.s3_bucket,
                    Prefix=s3_prefix,
                    MaxKeys=limit,
                ):
                    for obj in page.get("Contents", []):
                        packages.append({
                            "key": obj["Key"],
                            "size": obj["Size"],
                            "last_modified": obj["LastModified"].isoformat(),
                            "s3_uri": f"s3://{self.config.s3_bucket}/{obj['Key']}",
                        })

            return packages

        except Exception as exc:
            logger.error(f"Failed to list packages: {exc}")
            return []

    async def get_presigned_url(
        self,
        s3_key: str,
        expiry_hours: Optional[int] = None,
    ) -> str:
        """Generate presigned URL for package download.

        Args:
            s3_key: S3 object key.
            expiry_hours: URL expiration in hours.

        Returns:
            Presigned download URL.
        """
        try:
            import aioboto3

            session = aioboto3.Session()
            expiry = expiry_hours or self.config.presigned_url_expiry_hours

            async with session.client("s3", region_name=self.config.s3_region) as s3:
                url = await s3.generate_presigned_url(
                    "get_object",
                    Params={
                        "Bucket": self.config.s3_bucket,
                        "Key": s3_key,
                    },
                    ExpiresIn=expiry * 3600,
                )

            return url

        except Exception as exc:
            logger.error(f"Failed to generate presigned URL: {exc}")
            raise
