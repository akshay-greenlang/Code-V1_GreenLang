"""
Base ETL Pipeline Framework
===========================

Abstract base classes and infrastructure for building ETL pipelines.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
Created: 2025-12-04
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import asyncio
import hashlib
import json
from pathlib import Path
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = logging.getLogger(__name__)

T = TypeVar('T')


class PipelineStage(str, Enum):
    """ETL pipeline stages."""
    INITIALIZE = "initialize"
    EXTRACT = "extract"
    VALIDATE_SOURCE = "validate_source"
    TRANSFORM = "transform"
    VALIDATE_TARGET = "validate_target"
    LOAD_STAGING = "load_staging"
    LOAD_PRODUCTION = "load_production"
    CLEANUP = "cleanup"
    COMPLETE = "complete"
    FAILED = "failed"


class LoadMode(str, Enum):
    """Data loading modes."""
    FULL = "full"  # Truncate and reload
    INCREMENTAL = "incremental"  # Append/merge new records
    MERGE = "merge"  # Upsert (insert or update)
    APPEND = "append"  # Insert only


@dataclass
class PipelineMetrics:
    """Pipeline execution metrics."""
    records_extracted: int = 0
    records_transformed: int = 0
    records_valid: int = 0
    records_invalid: int = 0
    records_loaded: int = 0
    records_skipped: int = 0
    records_updated: int = 0
    bytes_processed: int = 0
    duration_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    pipeline_name: str = Field(..., description="Pipeline name")
    source_name: str = Field(..., description="Data source name")
    load_mode: LoadMode = Field(default=LoadMode.MERGE)
    batch_size: int = Field(default=1000, ge=1, le=100000)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=5, ge=1, le=300)
    timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    enable_validation: bool = Field(default=True)
    enable_audit: bool = Field(default=True)
    staging_table: Optional[str] = None
    target_table: str = Field(default="emission_factors")
    error_threshold_percent: float = Field(default=5.0, ge=0, le=100)
    checkpoint_enabled: bool = Field(default=True)
    checkpoint_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True


class PipelineResult(BaseModel):
    """Pipeline execution result."""
    pipeline_name: str
    run_id: str
    status: str  # success, failed, partial
    stage: PipelineStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checkpoints: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class PipelineCheckpoint:
    """Checkpoint management for pipeline recovery."""

    def __init__(self, checkpoint_path: str, pipeline_name: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.pipeline_name = pipeline_name
        self.checkpoint_file = self.checkpoint_path / f"{pipeline_name}_checkpoint.json"

    def save(self, stage: PipelineStage, data: Dict[str, Any]) -> None:
        """Save checkpoint data."""
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "pipeline_name": self.pipeline_name,
            "stage": stage.value,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        logger.debug(f"Checkpoint saved at stage {stage}")

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint data."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def clear(self) -> None:
        """Clear checkpoint data."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.debug("Checkpoint cleared")


class BasePipeline(ABC, Generic[T]):
    """
    Abstract base class for ETL pipelines.

    Provides:
    - Structured ETL workflow (extract -> validate -> transform -> load)
    - Error handling with retries
    - Checkpoint/recovery support
    - Metrics collection
    - Audit logging
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline."""
        self.config = config
        self.metrics = PipelineMetrics()
        self.run_id = self._generate_run_id()
        self.current_stage = PipelineStage.INITIALIZE
        self.checkpoint = None

        if config.checkpoint_enabled and config.checkpoint_path:
            self.checkpoint = PipelineCheckpoint(
                config.checkpoint_path,
                config.pipeline_name
            )

        logger.info(f"Pipeline {config.pipeline_name} initialized with run_id={self.run_id}")

    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        content = f"{self.config.pipeline_name}:{datetime.utcnow().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    async def run(self) -> PipelineResult:
        """
        Execute the full ETL pipeline.

        Returns:
            PipelineResult with execution details
        """
        started_at = datetime.utcnow()
        result = PipelineResult(
            pipeline_name=self.config.pipeline_name,
            run_id=self.run_id,
            status="running",
            stage=PipelineStage.INITIALIZE,
            started_at=started_at,
        )

        try:
            # Check for existing checkpoint
            if self.checkpoint:
                checkpoint_data = self.checkpoint.load()
                if checkpoint_data:
                    logger.info(f"Resuming from checkpoint: {checkpoint_data['stage']}")
                    # Resume logic would go here

            # Stage 1: Extract
            self.current_stage = PipelineStage.EXTRACT
            result.stage = self.current_stage
            logger.info(f"[{self.run_id}] Starting extraction")
            raw_data = await self._extract_with_retry()
            self.metrics.records_extracted = len(raw_data)
            logger.info(f"[{self.run_id}] Extracted {len(raw_data)} records")

            if self.checkpoint:
                self.checkpoint.save(self.current_stage, {"records_extracted": len(raw_data)})

            # Stage 2: Validate source data
            if self.config.enable_validation:
                self.current_stage = PipelineStage.VALIDATE_SOURCE
                result.stage = self.current_stage
                logger.info(f"[{self.run_id}] Validating source data")
                valid_data, invalid_data = await self.validate_source(raw_data)
                self.metrics.records_valid = len(valid_data)
                self.metrics.records_invalid = len(invalid_data)

                # Check error threshold
                if len(raw_data) > 0:
                    error_rate = (len(invalid_data) / len(raw_data)) * 100
                    if error_rate > self.config.error_threshold_percent:
                        raise ValueError(
                            f"Error rate {error_rate:.1f}% exceeds threshold "
                            f"{self.config.error_threshold_percent}%"
                        )

                logger.info(f"[{self.run_id}] Validation: {len(valid_data)} valid, {len(invalid_data)} invalid")
            else:
                valid_data = raw_data

            # Stage 3: Transform
            self.current_stage = PipelineStage.TRANSFORM
            result.stage = self.current_stage
            logger.info(f"[{self.run_id}] Transforming data")
            transformed_data = await self.transform(valid_data)
            self.metrics.records_transformed = len(transformed_data)
            logger.info(f"[{self.run_id}] Transformed {len(transformed_data)} records")

            if self.checkpoint:
                self.checkpoint.save(self.current_stage, {"records_transformed": len(transformed_data)})

            # Stage 4: Validate target data
            if self.config.enable_validation:
                self.current_stage = PipelineStage.VALIDATE_TARGET
                result.stage = self.current_stage
                logger.info(f"[{self.run_id}] Validating transformed data")
                final_data, rejected_data = await self.validate_target(transformed_data)
                self.metrics.records_skipped += len(rejected_data)
            else:
                final_data = transformed_data

            # Stage 5: Load to staging (if configured)
            if self.config.staging_table:
                self.current_stage = PipelineStage.LOAD_STAGING
                result.stage = self.current_stage
                logger.info(f"[{self.run_id}] Loading to staging table")
                await self.load_staging(final_data)

            # Stage 6: Load to production
            self.current_stage = PipelineStage.LOAD_PRODUCTION
            result.stage = self.current_stage
            logger.info(f"[{self.run_id}] Loading to production table")
            load_result = await self.load_production(final_data)
            self.metrics.records_loaded = load_result.get("inserted", 0)
            self.metrics.records_updated = load_result.get("updated", 0)
            logger.info(
                f"[{self.run_id}] Loaded {self.metrics.records_loaded} records, "
                f"updated {self.metrics.records_updated} records"
            )

            # Stage 7: Cleanup
            self.current_stage = PipelineStage.CLEANUP
            result.stage = self.current_stage
            await self.cleanup()

            # Complete
            self.current_stage = PipelineStage.COMPLETE
            result.stage = self.current_stage
            result.status = "success"

            if self.checkpoint:
                self.checkpoint.clear()

        except Exception as e:
            self.current_stage = PipelineStage.FAILED
            result.stage = self.current_stage
            result.status = "failed"
            result.errors.append(str(e))
            self.metrics.errors.append(str(e))
            logger.error(f"[{self.run_id}] Pipeline failed: {e}")
            raise

        finally:
            completed_at = datetime.utcnow()
            result.completed_at = completed_at
            result.duration_seconds = (completed_at - started_at).total_seconds()
            self.metrics.duration_seconds = result.duration_seconds

            result.metrics = {
                "records_extracted": self.metrics.records_extracted,
                "records_transformed": self.metrics.records_transformed,
                "records_valid": self.metrics.records_valid,
                "records_invalid": self.metrics.records_invalid,
                "records_loaded": self.metrics.records_loaded,
                "records_updated": self.metrics.records_updated,
                "records_skipped": self.metrics.records_skipped,
                "duration_seconds": self.metrics.duration_seconds,
            }
            result.warnings = self.metrics.warnings

            # Audit log
            if self.config.enable_audit:
                await self._log_audit(result)

        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _extract_with_retry(self) -> List[T]:
        """Extract with retry logic."""
        return await self.extract()

    @abstractmethod
    async def extract(self) -> List[T]:
        """
        Extract data from source.

        Returns:
            List of raw data records
        """
        pass

    async def validate_source(self, data: List[T]) -> tuple[List[T], List[T]]:
        """
        Validate source data.

        Args:
            data: Raw extracted data

        Returns:
            Tuple of (valid_records, invalid_records)
        """
        valid = []
        invalid = []
        for record in data:
            if self._is_valid_source_record(record):
                valid.append(record)
            else:
                invalid.append(record)
        return valid, invalid

    def _is_valid_source_record(self, record: T) -> bool:
        """Check if source record is valid. Override in subclass."""
        return record is not None

    @abstractmethod
    async def transform(self, data: List[T]) -> List[Dict[str, Any]]:
        """
        Transform data to target schema.

        Args:
            data: Validated source data

        Returns:
            List of transformed records
        """
        pass

    async def validate_target(self, data: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate transformed data against target schema.

        Args:
            data: Transformed data

        Returns:
            Tuple of (valid_records, rejected_records)
        """
        valid = []
        rejected = []
        for record in data:
            if self._is_valid_target_record(record):
                valid.append(record)
            else:
                rejected.append(record)
        return valid, rejected

    def _is_valid_target_record(self, record: Dict[str, Any]) -> bool:
        """Check if target record is valid. Override in subclass."""
        required_fields = ['factor_id', 'factor_value', 'factor_unit']
        return all(field in record and record[field] is not None for field in required_fields)

    async def load_staging(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Load data to staging table.

        Args:
            data: Validated data

        Returns:
            Load statistics
        """
        # Default implementation - override in subclass
        return {"loaded": len(data)}

    @abstractmethod
    async def load_production(self, data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Load data to production table.

        Args:
            data: Final validated data

        Returns:
            Load statistics (inserted, updated, etc.)
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup resources after pipeline execution."""
        pass

    async def _log_audit(self, result: PipelineResult) -> None:
        """Log pipeline execution to audit table."""
        logger.info(
            f"Pipeline {result.pipeline_name} completed: "
            f"status={result.status}, "
            f"records_loaded={result.metrics.get('records_loaded', 0)}, "
            f"duration={result.duration_seconds:.2f}s"
        )


class DeadLetterQueue:
    """Dead Letter Queue for failed records."""

    def __init__(self, queue_path: str):
        self.queue_path = Path(queue_path)
        self.queue_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        record: Dict[str, Any],
        error: str,
        pipeline_name: str,
        stage: PipelineStage
    ) -> None:
        """Write failed record to DLQ."""
        dlq_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_name": pipeline_name,
            "stage": stage.value,
            "error": error,
            "record": record,
        }

        filename = f"dlq_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
        filepath = self.queue_path / filename

        with open(filepath, 'w') as f:
            json.dump(dlq_entry, f, indent=2, default=str)

        logger.warning(f"Record written to DLQ: {filepath}")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all records from DLQ."""
        records = []
        for filepath in self.queue_path.glob("dlq_*.json"):
            with open(filepath, 'r') as f:
                records.append(json.load(f))
        return records

    def replay(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific record for replay."""
        for filepath in self.queue_path.glob(f"dlq_*{record_id}*.json"):
            with open(filepath, 'r') as f:
                return json.load(f)
        return None

    def clear(self) -> int:
        """Clear all records from DLQ."""
        count = 0
        for filepath in self.queue_path.glob("dlq_*.json"):
            filepath.unlink()
            count += 1
        return count
