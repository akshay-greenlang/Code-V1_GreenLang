# -*- coding: utf-8 -*-
"""
GreenLang Infrastructure - Agent Templates
===========================================

Template classes for batch processing agents.
Provides base implementations for common agent patterns.

Author: GreenLang Infrastructure Team
"""

from typing import List, Dict, Any, Optional, Generic, TypeVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class BatchResult:
    """Result of a batch processing operation."""
    total_items: int
    successful_items: int
    failed_items: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_items == 0:
            return 0.0
        return self.successful_items / self.total_items

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class BatchAgent(ABC, Generic[T, R]):
    """
    Template for batch processing agents.

    Provides infrastructure for processing large datasets in batches
    with error handling, progress tracking, and parallel processing support.

    Type Parameters:
        T: Input item type
        R: Result item type
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_workers: int = 4,
        use_async: bool = False,
        continue_on_error: bool = True
    ):
        """
        Initialize batch agent.

        Args:
            batch_size: Number of items to process in each batch
            max_workers: Maximum number of parallel workers
            use_async: Use async processing
            continue_on_error: Continue processing if an item fails
        """
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.use_async = use_async
        self.continue_on_error = continue_on_error
        self._executor: Optional[ThreadPoolExecutor] = None
        logger.info(f"Initialized BatchAgent with batch_size={batch_size}, max_workers={max_workers}")

    @abstractmethod
    def process_item(self, item: T) -> R:
        """
        Process a single item.

        Args:
            item: Item to process

        Returns:
            Processed result

        Raises:
            Exception: If processing fails
        """
        pass

    def validate_item(self, item: T) -> bool:
        """
        Validate an item before processing.

        Override this method to add custom validation.

        Args:
            item: Item to validate

        Returns:
            True if valid, False otherwise
        """
        return True

    def pre_batch_hook(self, items: List[T]) -> None:
        """
        Hook called before processing a batch.

        Override to add pre-processing logic.

        Args:
            items: Items in the batch
        """
        pass

    def post_batch_hook(self, items: List[T], results: List[R]) -> None:
        """
        Hook called after processing a batch.

        Override to add post-processing logic.

        Args:
            items: Items that were processed
            results: Processing results
        """
        pass

    def process_batch(self, items: List[T]) -> BatchResult:
        """
        Process a batch of items.

        Args:
            items: Items to process

        Returns:
            BatchResult with processing outcomes
        """
        result = BatchResult(total_items=len(items))

        # Pre-processing hook
        self.pre_batch_hook(items)

        # Process items
        for i, item in enumerate(items):
            try:
                # Validate item
                if not self.validate_item(item):
                    result.failed_items += 1
                    result.errors.append({
                        "index": i,
                        "error": "Validation failed",
                        "item": str(item)[:100]  # Truncate for logging
                    })
                    if not self.continue_on_error:
                        break
                    continue

                # Process item
                processed = self.process_item(item)
                result.results.append(processed)
                result.successful_items += 1

            except Exception as e:
                result.failed_items += 1
                result.errors.append({
                    "index": i,
                    "error": str(e),
                    "item": str(item)[:100]
                })
                logger.error(f"Error processing item {i}: {e}")

                if not self.continue_on_error:
                    break

        # Post-processing hook
        self.post_batch_hook(items, result.results)

        # Finalize result
        result.end_time = datetime.now()
        return result

    def process_all(self, items: List[T]) -> BatchResult:
        """
        Process all items in batches.

        Args:
            items: All items to process

        Returns:
            Combined BatchResult
        """
        combined_result = BatchResult(total_items=len(items))

        # Process in batches
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            logger.info(f"Processing batch {i//self.batch_size + 1} ({len(batch)} items)")

            batch_result = self.process_batch(batch)

            # Combine results
            combined_result.successful_items += batch_result.successful_items
            combined_result.failed_items += batch_result.failed_items
            combined_result.errors.extend(batch_result.errors)
            combined_result.results.extend(batch_result.results)

        combined_result.end_time = datetime.now()

        logger.info(
            f"Batch processing complete: {combined_result.successful_items}/{combined_result.total_items} successful "
            f"({combined_result.success_rate:.1%} success rate)"
        )

        return combined_result

    async def process_batch_async(self, items: List[T]) -> BatchResult:
        """
        Process a batch of items asynchronously.

        Args:
            items: Items to process

        Returns:
            BatchResult with processing outcomes
        """
        result = BatchResult(total_items=len(items))

        # Pre-processing hook
        self.pre_batch_hook(items)

        # Process items concurrently
        tasks = []
        for i, item in enumerate(items):
            if self.validate_item(item):
                tasks.append(self._process_item_async(i, item))
            else:
                result.failed_items += 1
                result.errors.append({
                    "index": i,
                    "error": "Validation failed",
                    "item": str(item)[:100]
                })

        # Wait for all tasks
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, task_result in enumerate(task_results):
                if isinstance(task_result, Exception):
                    result.failed_items += 1
                    result.errors.append({
                        "index": i,
                        "error": str(task_result),
                        "item": str(items[i])[:100]
                    })
                else:
                    result.results.append(task_result)
                    result.successful_items += 1

        # Post-processing hook
        self.post_batch_hook(items, result.results)

        result.end_time = datetime.now()
        return result

    async def _process_item_async(self, index: int, item: T) -> R:
        """Process an item asynchronously."""
        try:
            # Run in executor if process_item is not async
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.process_item, item)
        except Exception as e:
            logger.error(f"Error processing item {index}: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None