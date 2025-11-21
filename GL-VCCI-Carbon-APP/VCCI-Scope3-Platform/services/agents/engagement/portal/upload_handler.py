# -*- coding: utf-8 -*-
"""
Upload handler for supplier data submissions.

Supports CSV, Excel, JSON uploads with validation.
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import json
import csv
from io import StringIO

from ..models import DataUpload, UploadStatus, ValidationResult
from ..exceptions import FileFormatError, UploadValidationError
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock


logger = logging.getLogger(__name__)


class UploadHandler:
    """
    Handles supplier data uploads with validation and processing.

    Features:
    - Multi-format support (CSV, Excel, JSON)
    - Real-time validation
    - Progress tracking
    - Data quality scoring
    """

    def __init__(self, max_file_size_mb: int = 50):
        """
        Initialize upload handler.

        Args:
            max_file_size_mb: Maximum file size in MB
        """
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.supported_formats = ["csv", "xlsx", "json", "xml"]
        self.uploads: Dict[str, DataUpload] = {}
        logger.info("UploadHandler initialized")

    def initiate_upload(
        self,
        supplier_id: str,
        campaign_id: str,
        file_name: str,
        file_size: int,
        file_type: str
    ) -> DataUpload:
        """
        Initiate data upload.

        Args:
            supplier_id: Supplier identifier
            campaign_id: Campaign identifier
            file_name: Name of uploaded file
            file_size: File size in bytes
            file_type: File type (csv, xlsx, json)

        Returns:
            Data upload record

        Raises:
            FileFormatError: If file format not supported
        """
        # Validate file size
        if file_size > self.max_file_size_bytes:
            raise FileFormatError(
                file_type,
                self.supported_formats
            )

        # Validate file type
        if file_type.lower() not in self.supported_formats:
            raise FileFormatError(file_type, self.supported_formats)

        # Generate upload ID
        upload_id = f"upload_{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:16]}"

        upload = DataUpload(
            upload_id=upload_id,
            supplier_id=supplier_id,
            campaign_id=campaign_id,
            file_name=file_name,
            file_type=file_type.lower(),
            file_size_bytes=file_size,
            status=UploadStatus.IN_PROGRESS
        )

        self.uploads[upload_id] = upload

        logger.info(
            f"Initiated upload {upload_id} for supplier {supplier_id}: "
            f"{file_name} ({file_size} bytes)"
        )

        return upload

    def process_csv(
        self,
        upload_id: str,
        csv_content: str
    ) -> Dict[str, Any]:
        """
        Process CSV file upload.

        Args:
            upload_id: Upload identifier
            csv_content: CSV file content

        Returns:
            Parsed data
        """
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError(f"Upload {upload_id} not found")

        try:
            # Parse CSV
            csv_file = StringIO(csv_content)
            reader = csv.DictReader(csv_file)
            records = list(reader)

            upload.records_count = len(records)
            upload.status = UploadStatus.VALIDATING

            logger.info(f"Parsed {len(records)} records from CSV upload {upload_id}")

            return {"records": records, "count": len(records)}

        except Exception as e:
            upload.status = UploadStatus.FAILED
            upload.validation_errors.append(f"CSV parsing error: {str(e)}")
            logger.error(f"Failed to parse CSV upload {upload_id}: {e}")
            raise

    def process_json(
        self,
        upload_id: str,
        json_content: str
    ) -> Dict[str, Any]:
        """
        Process JSON file upload.

        Args:
            upload_id: Upload identifier
            json_content: JSON file content

        Returns:
            Parsed data
        """
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError(f"Upload {upload_id} not found")

        try:
            # Parse JSON
            data = json.loads(json_content)

            # Expect array of records
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
            else:
                raise ValueError("Expected array of records or {records: [...]}")

            upload.records_count = len(records)
            upload.status = UploadStatus.VALIDATING

            logger.info(f"Parsed {len(records)} records from JSON upload {upload_id}")

            return {"records": records, "count": len(records)}

        except Exception as e:
            upload.status = UploadStatus.FAILED
            upload.validation_errors.append(f"JSON parsing error: {str(e)}")
            logger.error(f"Failed to parse JSON upload {upload_id}: {e}")
            raise

    def complete_upload(
        self,
        upload_id: str,
        validation_result: ValidationResult
    ) -> DataUpload:
        """
        Complete upload with validation result.

        Args:
            upload_id: Upload identifier
            validation_result: Validation result

        Returns:
            Updated upload record
        """
        upload = self.uploads.get(upload_id)
        if not upload:
            raise ValueError(f"Upload {upload_id} not found")

        upload.validated_at = DeterministicClock.utcnow()
        upload.validation_errors = validation_result.errors
        upload.data_quality_score = validation_result.data_quality_score

        if validation_result.is_valid:
            upload.status = UploadStatus.COMPLETED
            logger.info(
                f"Completed upload {upload_id} with DQI {validation_result.data_quality_score:.2f}"
            )
        else:
            upload.status = UploadStatus.FAILED
            logger.warning(
                f"Upload {upload_id} failed validation with {len(validation_result.errors)} errors"
            )

        return upload

    def get_upload(self, upload_id: str) -> Optional[DataUpload]:
        """
        Get upload record.

        Args:
            upload_id: Upload identifier

        Returns:
            Upload record or None
        """
        return self.uploads.get(upload_id)

    def get_supplier_uploads(
        self,
        supplier_id: str
    ) -> List[DataUpload]:
        """
        Get all uploads for supplier.

        Args:
            supplier_id: Supplier identifier

        Returns:
            List of uploads
        """
        return [
            upload for upload in self.uploads.values()
            if upload.supplier_id == supplier_id
        ]

    def get_campaign_uploads(
        self,
        campaign_id: str
    ) -> List[DataUpload]:
        """
        Get all uploads for campaign.

        Args:
            campaign_id: Campaign identifier

        Returns:
            List of uploads
        """
        return [
            upload for upload in self.uploads.values()
            if upload.campaign_id == campaign_id
        ]

    def get_upload_statistics(
        self,
        campaign_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get upload statistics.

        Args:
            campaign_id: Optional campaign ID to filter by

        Returns:
            Upload statistics
        """
        uploads = (
            self.get_campaign_uploads(campaign_id) if campaign_id
            else list(self.uploads.values())
        )

        total = len(uploads)
        completed = len([u for u in uploads if u.status == UploadStatus.COMPLETED])
        failed = len([u for u in uploads if u.status == UploadStatus.FAILED])
        in_progress = len([u for u in uploads if u.status == UploadStatus.IN_PROGRESS])

        # Average DQI
        dqi_scores = [u.data_quality_score for u in uploads if u.data_quality_score is not None]
        avg_dqi = sum(dqi_scores) / len(dqi_scores) if dqi_scores else 0.0

        # Total records
        total_records = sum(u.records_count for u in uploads)

        return {
            "total_uploads": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "average_dqi": avg_dqi,
            "total_records": total_records,
        }
