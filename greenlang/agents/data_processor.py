"""
GreenLang Data Processing Agent
Specialized base class for data transformation and batch processing operations.
"""

from typing import Any, Dict, List, Optional, Callable, Union, Iterator
from pydantic import BaseModel, Field, validator
from abc import abstractmethod
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .base import BaseAgent, AgentConfig, AgentResult, AgentMetrics

logger = logging.getLogger(__name__)


class DataProcessorConfig(AgentConfig):
    """Configuration for data processing agents."""
    batch_size: int = Field(default=1000, description="Number of records to process in a batch")
    parallel_workers: int = Field(default=1, description="Number of parallel workers for batch processing")
    enable_progress: bool = Field(default=True, description="Show progress bar during processing")
    collect_errors: bool = Field(default=True, description="Collect errors instead of failing immediately")
    max_errors: int = Field(default=100, description="Maximum number of errors before aborting")
    validate_records: bool = Field(default=True, description="Validate each record before processing")

    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    @validator('parallel_workers')
    def validate_workers(cls, v):
        if v <= 0:
            raise ValueError("parallel_workers must be positive")
        if v > 32:
            raise ValueError("parallel_workers cannot exceed 32")
        return v


class ProcessingError(BaseModel):
    """Record of a processing error."""
    record_id: Union[int, str] = Field(..., description="ID of the failed record")
    record_data: Dict[str, Any] = Field(..., description="The failed record data")
    error_message: str = Field(..., description="Error message")
    timestamp: datetime = Field(default_factory=datetime.now, description="When error occurred")


class DataProcessorResult(AgentResult):
    """Enhanced result from data processing with detailed metrics."""
    records_processed: int = Field(default=0, description="Number of records successfully processed")
    records_failed: int = Field(default=0, description="Number of records that failed")
    errors: List[ProcessingError] = Field(default_factory=list, description="Collected errors")
    batches_processed: int = Field(default=0, description="Number of batches processed")


class BaseDataProcessor(BaseAgent):
    """
    Base class for data processing agents.

    Provides:
    - Batch processing with configurable batch size
    - Parallel processing with thread pool
    - Record-level validation and transformation
    - Error collection and reporting
    - Progress tracking
    - Automatic metrics collection

    Example:
        class CSVProcessor(BaseDataProcessor):
            def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
                # Transform a single record
                record['price'] = float(record['price']) * 1.1
                return record

            def validate_record(self, record: Dict[str, Any]) -> bool:
                # Validate a single record
                return 'price' in record and 'quantity' in record
    """

    def __init__(self, config: Optional[DataProcessorConfig] = None):
        """Initialize data processor with configuration."""
        if config is None:
            config = DataProcessorConfig(
                name=self.__class__.__name__,
                description=self.__class__.__doc__ or "Data processor agent"
            )
        super().__init__(config)
        self.config: DataProcessorConfig = config

    @abstractmethod
    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single record.
        Must be implemented by subclasses.

        Args:
            record: Input record as dictionary

        Returns:
            Processed record as dictionary
        """
        pass

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate a single record.
        Override to add custom validation logic.

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def transform_record(self, record: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Transform a single record with error handling.

        Args:
            record: Input record
            index: Record index

        Returns:
            Transformed record

        Raises:
            Exception if processing fails and error collection is disabled
        """
        try:
            # Validate if enabled
            if self.config.validate_records and not self.validate_record(record):
                raise ValueError(f"Record validation failed for record {index}")

            # Process the record
            result = self.process_record(record)

            # Track success
            self.stats.increment("records_processed")

            return result

        except Exception as e:
            self.stats.increment("records_failed")

            if not self.config.collect_errors:
                raise

            # Record the error
            error = ProcessingError(
                record_id=index,
                record_data=record,
                error_message=str(e)
            )
            self.logger.warning(f"Failed to process record {index}: {str(e)}")

            return {"_error": error}

    def process_batch(self, batch: List[Dict[str, Any]], batch_index: int) -> List[Dict[str, Any]]:
        """
        Process a batch of records.

        Args:
            batch: List of records to process
            batch_index: Index of this batch

        Returns:
            List of processed records
        """
        results = []
        start_index = batch_index * self.config.batch_size

        for i, record in enumerate(batch):
            record_index = start_index + i
            processed = self.transform_record(record, record_index)
            results.append(processed)

        self.stats.increment("batches_processed")
        return results

    def process_batches_parallel(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process batches in parallel using thread pool.

        Args:
            batches: List of batches to process

        Returns:
            List of all processed records
        """
        all_results = []

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            # Submit all batches
            futures = {
                executor.submit(self.process_batch, batch, i): i
                for i, batch in enumerate(batches)
            }

            # Collect results with progress bar
            iterator = as_completed(futures)
            if self.config.enable_progress:
                iterator = tqdm(iterator, total=len(batches), desc="Processing batches")

            for future in iterator:
                batch_results = future.result()
                all_results.extend(batch_results)

                # Check error threshold
                errors = self.stats.custom_counters.get("records_failed", 0)
                if errors >= self.config.max_errors:
                    self.logger.error(f"Exceeded max errors ({self.config.max_errors}), aborting")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

        return all_results

    def process_batches_sequential(self, batches: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process batches sequentially.

        Args:
            batches: List of batches to process

        Returns:
            List of all processed records
        """
        all_results = []

        iterator = enumerate(batches)
        if self.config.enable_progress:
            iterator = tqdm(iterator, total=len(batches), desc="Processing batches")

        for i, batch in iterator:
            batch_results = self.process_batch(batch, i)
            all_results.extend(batch_results)

            # Check error threshold
            errors = self.stats.custom_counters.get("records_failed", 0)
            if errors >= self.config.max_errors:
                self.logger.error(f"Exceeded max errors ({self.config.max_errors}), aborting")
                break

        return all_results

    def create_batches(self, records: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Split records into batches.

        Args:
            records: List of records to batch

        Returns:
            List of batches
        """
        batches = []
        for i in range(0, len(records), self.config.batch_size):
            batch = records[i:i + self.config.batch_size]
            batches.append(batch)
        return batches

    def execute(self, input_data: Dict[str, Any]) -> DataProcessorResult:
        """
        Execute data processing on input records.

        Args:
            input_data: Must contain 'records' key with list of records to process

        Returns:
            DataProcessorResult with processed records and metrics
        """
        # Extract records from input
        records = input_data.get("records", [])
        if not records:
            return DataProcessorResult(
                success=False,
                error="No records provided in input_data['records']"
            )

        self.logger.info(f"Processing {len(records)} records with batch size {self.config.batch_size}")

        # Create batches
        batches = self.create_batches(records)

        # Process batches (parallel or sequential)
        if self.config.parallel_workers > 1:
            processed_records = self.process_batches_parallel(batches)
        else:
            processed_records = self.process_batches_sequential(batches)

        # Separate successful results from errors
        successful_records = []
        errors = []

        for record in processed_records:
            if "_error" in record:
                errors.append(record["_error"])
            else:
                successful_records.append(record)

        # Create result
        records_processed = self.stats.custom_counters.get("records_processed", 0)
        records_failed = self.stats.custom_counters.get("records_failed", 0)
        batches_processed = self.stats.custom_counters.get("batches_processed", 0)

        success = records_failed < self.config.max_errors

        result = DataProcessorResult(
            success=success,
            data={"records": successful_records},
            records_processed=records_processed,
            records_failed=records_failed,
            errors=errors,
            batches_processed=batches_processed,
            metadata={
                "total_input_records": len(records),
                "total_output_records": len(successful_records),
                "batch_size": self.config.batch_size,
                "parallel_workers": self.config.parallel_workers
            }
        )

        if not success:
            result.error = f"Processing failed with {records_failed} errors (max: {self.config.max_errors})"

        return result

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate that input contains records."""
        if "records" not in input_data:
            self.logger.error("Input data must contain 'records' key")
            return False

        records = input_data["records"]
        if not isinstance(records, list):
            self.logger.error("'records' must be a list")
            return False

        return True

    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get detailed processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        stats = self.get_stats()
        stats["processing"] = {
            "records_processed": self.stats.custom_counters.get("records_processed", 0),
            "records_failed": self.stats.custom_counters.get("records_failed", 0),
            "batches_processed": self.stats.custom_counters.get("batches_processed", 0),
            "success_rate": self._calculate_success_rate()
        }
        return stats

    def _calculate_success_rate(self) -> float:
        """Calculate record-level success rate."""
        processed = self.stats.custom_counters.get("records_processed", 0)
        failed = self.stats.custom_counters.get("records_failed", 0)
        total = processed + failed

        if total == 0:
            return 0.0

        return round((processed / total) * 100, 2)
