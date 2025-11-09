"""
Parallel Processing Engine
GL-VCCI Scope 3 Platform - Concurrent Processing Optimization

This module provides advanced parallel processing capabilities:
- AsyncIO concurrent processing
- Process pool for CPU-bound tasks
- Thread pool for I/O-bound tasks
- Backpressure and rate limiting
- Dynamic concurrency adjustment

Performance Targets:
- Throughput: 100,000 suppliers/hour
- Concurrency: Up to 100 concurrent tasks
- CPU utilization: <70%
- Memory: <8GB

Version: 1.0.0
Team: Performance & Batch Processing (Team 5)
Date: 2025-11-09
"""

import logging
import asyncio
import multiprocessing
from typing import List, Dict, Any, Callable, TypeVar, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import time
import psutil

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# PARALLEL PROCESSING CONFIGURATION
# ============================================================================

@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    # Concurrency settings
    max_concurrent_tasks: int = 50  # Max concurrent async tasks
    max_workers: int = None  # CPU cores (None = auto-detect)
    max_threads: int = 20  # Thread pool size

    # Performance settings
    enable_backpressure: bool = True
    backpressure_threshold: int = 100  # Queue size threshold
    enable_rate_limiting: bool = True
    max_rate_per_second: int = 1000

    # Resource limits
    max_cpu_percent: float = 70.0  # Max CPU utilization
    max_memory_percent: float = 80.0  # Max memory utilization

    # Monitoring
    enable_progress_tracking: bool = True
    log_interval: int = 10  # Log every N tasks

    def __post_init__(self):
        if self.max_workers is None:
            # Auto-detect CPU cores
            self.max_workers = multiprocessing.cpu_count()


# Optimized configurations
HIGH_THROUGHPUT_CONFIG = ParallelConfig(
    max_concurrent_tasks=100,
    max_workers=multiprocessing.cpu_count(),
    max_threads=50,
    enable_backpressure=True,
    max_rate_per_second=2000
)

BALANCED_CONFIG = ParallelConfig(
    max_concurrent_tasks=50,
    max_workers=multiprocessing.cpu_count() // 2,
    max_threads=20,
    enable_backpressure=True,
    max_rate_per_second=1000
)

LOW_RESOURCE_CONFIG = ParallelConfig(
    max_concurrent_tasks=10,
    max_workers=2,
    max_threads=5,
    enable_backpressure=True,
    max_rate_per_second=100
)


# ============================================================================
# ASYNC PARALLEL PROCESSOR
# ============================================================================

class AsyncParallelProcessor:
    """
    High-performance async parallel processor.

    Features:
    - Concurrent task execution with asyncio.gather()
    - Semaphore-based concurrency control
    - Backpressure handling
    - Resource monitoring
    - Progress tracking
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize async parallel processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or BALANCED_CONFIG
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

    async def process_parallel(
        self,
        items: List[T],
        processor: Callable[[T], R],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[R]:
        """
        Process items in parallel using asyncio.

        Args:
            items: List of items to process
            processor: Async function to process each item
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0

        logger.info(
            f"Starting parallel processing: {len(items)} items, "
            f"max_concurrent={self.config.max_concurrent_tasks}"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def process_with_semaphore(item: T, index: int) -> Optional[R]:
            """Process single item with semaphore control"""
            async with semaphore:
                try:
                    # Check resource limits
                    if self.config.enable_backpressure:
                        await self._check_resource_limits()

                    # Process item
                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(item)
                    else:
                        result = processor(item)

                    self.processed_count += 1

                    # Progress callback
                    if progress_callback:
                        progress_callback(index + 1, len(items))

                    # Log progress
                    if (
                        self.config.enable_progress_tracking and
                        self.processed_count % self.config.log_interval == 0
                    ):
                        elapsed = time.time() - self.start_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0

                        logger.info(
                            f"Progress: {self.processed_count}/{len(items)} "
                            f"({rate:.1f} items/sec)"
                        )

                    return result

                except Exception as e:
                    logger.error(
                        f"Error processing item {index}: {e}",
                        exc_info=True
                    )
                    self.error_count += 1
                    return None

        # Create tasks for all items
        tasks = [
            process_with_semaphore(item, i)
            for i, item in enumerate(items)
        ]

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        valid_results = [
            r for r in results
            if r is not None and not isinstance(r, Exception)
        ]

        # Log final statistics
        elapsed = time.time() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0

        logger.info(
            f"Parallel processing completed: "
            f"{self.processed_count} successful, "
            f"{self.error_count} errors, "
            f"{elapsed:.2f}s, "
            f"{rate:.1f} items/sec"
        )

        return valid_results

    async def process_batches_parallel(
        self,
        batches: List[List[T]],
        batch_processor: Callable[[List[T]], R]
    ) -> List[R]:
        """
        Process multiple batches in parallel.

        Args:
            batches: List of batches to process
            batch_processor: Async function to process each batch

        Returns:
            List of batch results
        """
        logger.info(
            f"Processing {len(batches)} batches in parallel "
            f"(max_concurrent={self.config.max_concurrent_tasks})"
        )

        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        async def process_batch_with_semaphore(batch: List[T], batch_num: int):
            async with semaphore:
                try:
                    logger.debug(f"Processing batch {batch_num + 1}/{len(batches)}")

                    if asyncio.iscoroutinefunction(batch_processor):
                        result = await batch_processor(batch)
                    else:
                        result = batch_processor(batch)

                    return result

                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} failed: {e}")
                    return None

        # Process all batches
        tasks = [
            process_batch_with_semaphore(batch, i)
            for i, batch in enumerate(batches)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter valid results
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    async def _check_resource_limits(self):
        """Check system resource limits and apply backpressure"""
        # Check CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent > self.config.max_cpu_percent:
            logger.warning(
                f"High CPU utilization: {cpu_percent}%, "
                "applying backpressure"
            )
            await asyncio.sleep(0.5)

        # Check memory utilization
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > self.config.max_memory_percent:
            logger.warning(
                f"High memory utilization: {memory_percent}%, "
                "applying backpressure"
            )
            await asyncio.sleep(1.0)


# ============================================================================
# MULTIPROCESSING PARALLEL PROCESSOR
# ============================================================================

class MultiprocessingParallelProcessor:
    """
    Parallel processor using multiprocessing for CPU-bound tasks.

    Use for:
    - CPU-intensive calculations
    - Tasks that benefit from multiple processes
    - GIL-free parallel execution
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize multiprocessing processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or BALANCED_CONFIG

    def process_parallel(
        self,
        items: List[T],
        processor: Callable[[T], R]
    ) -> List[R]:
        """
        Process items in parallel using ProcessPoolExecutor.

        Args:
            items: List of items to process
            processor: Function to process each item

        Returns:
            List of results
        """
        logger.info(
            f"Starting multiprocessing: {len(items)} items, "
            f"{self.config.max_workers} workers"
        )

        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(processor, item): i
                for i, item in enumerate(items)
            }

            # Collect results
            results = []
            completed = 0

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Log progress
                    if completed % self.config.log_interval == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(items)} "
                            f"({rate:.1f} items/sec)"
                        )

                except Exception as e:
                    logger.error(f"Task failed: {e}")

        elapsed = time.time() - start_time
        rate = len(results) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Multiprocessing completed: {len(results)} results in {elapsed:.2f}s "
            f"({rate:.1f} items/sec)"
        )

        return results


# ============================================================================
# THREAD POOL PARALLEL PROCESSOR
# ============================================================================

class ThreadPoolParallelProcessor:
    """
    Parallel processor using threading for I/O-bound tasks.

    Use for:
    - I/O-intensive operations (network, disk)
    - API calls
    - Database operations
    """

    def __init__(self, config: Optional[ParallelConfig] = None):
        """
        Initialize thread pool processor.

        Args:
            config: Parallel processing configuration
        """
        self.config = config or BALANCED_CONFIG

    def process_parallel(
        self,
        items: List[T],
        processor: Callable[[T], R]
    ) -> List[R]:
        """
        Process items in parallel using ThreadPoolExecutor.

        Args:
            items: List of items to process
            processor: Function to process each item

        Returns:
            List of results
        """
        logger.info(
            f"Starting thread pool processing: {len(items)} items, "
            f"{self.config.max_threads} threads"
        )

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.config.max_threads) as executor:
            # Submit all tasks
            futures = {
                executor.submit(processor, item): i
                for i, item in enumerate(items)
            }

            # Collect results
            results = []
            completed = 0

            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    # Log progress
                    if completed % self.config.log_interval == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{len(items)} "
                            f"({rate:.1f} items/sec)"
                        )

                except Exception as e:
                    logger.error(f"Task failed: {e}")

        elapsed = time.time() - start_time
        rate = len(results) / elapsed if elapsed > 0 else 0

        logger.info(
            f"Thread pool completed: {len(results)} results in {elapsed:.2f}s "
            f"({rate:.1f} items/sec)"
        )

        return results


# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    """
    Token bucket rate limiter for controlling throughput.

    Features:
    - Token bucket algorithm
    - Configurable rate
    - Async/await support
    """

    def __init__(self, max_rate_per_second: int):
        """
        Initialize rate limiter.

        Args:
            max_rate_per_second: Maximum operations per second
        """
        self.max_rate = max_rate_per_second
        self.tokens = max_rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire token (wait if necessary)"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens
            self.tokens = min(
                self.max_rate,
                self.tokens + elapsed * self.max_rate
            )

            self.last_update = now

            # Wait if no tokens available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.max_rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1


# ============================================================================
# ADAPTIVE CONCURRENCY CONTROLLER
# ============================================================================

class AdaptiveConcurrencyController:
    """
    Dynamically adjust concurrency based on system metrics.

    Features:
    - Monitor CPU and memory
    - Adjust concurrency automatically
    - Prevent resource exhaustion
    """

    def __init__(
        self,
        initial_concurrency: int = 10,
        min_concurrency: int = 1,
        max_concurrency: int = 100,
        target_cpu_percent: float = 60.0
    ):
        """
        Initialize adaptive controller.

        Args:
            initial_concurrency: Starting concurrency level
            min_concurrency: Minimum concurrency
            max_concurrency: Maximum concurrency
            target_cpu_percent: Target CPU utilization
        """
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.target_cpu = target_cpu_percent

    async def adjust_concurrency(self) -> int:
        """
        Adjust concurrency based on current metrics.

        Returns:
            New concurrency level
        """
        # Get current CPU utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Adjust concurrency
        if cpu_percent > self.target_cpu + 10:
            # Decrease concurrency if CPU too high
            self.current_concurrency = max(
                self.min_concurrency,
                int(self.current_concurrency * 0.9)
            )
            logger.info(
                f"Decreasing concurrency to {self.current_concurrency} "
                f"(CPU: {cpu_percent}%)"
            )

        elif cpu_percent < self.target_cpu - 10:
            # Increase concurrency if CPU too low
            self.current_concurrency = min(
                self.max_concurrency,
                int(self.current_concurrency * 1.1)
            )
            logger.info(
                f"Increasing concurrency to {self.current_concurrency} "
                f"(CPU: {cpu_percent}%)"
            )

        return self.current_concurrency


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

EXAMPLE_USAGE = """
# ============================================================================
# Parallel Processing Usage Examples
# ============================================================================

# Example 1: Async Parallel Processing
# ----------------------------------------------------------------------------
from processing.parallel_processor import AsyncParallelProcessor, HIGH_THROUGHPUT_CONFIG

processor = AsyncParallelProcessor(config=HIGH_THROUGHPUT_CONFIG)

# Process 100K items concurrently
items = [{"supplier_id": f"SUP-{i}"} for i in range(100000)]

results = await processor.process_parallel(
    items,
    processor=calculate_single_emission,
    progress_callback=lambda current, total: print(f"{current}/{total}")
)


# Example 2: Multiprocessing for CPU-Bound Tasks
# ----------------------------------------------------------------------------
from processing.parallel_processor import MultiprocessingParallelProcessor

processor = MultiprocessingParallelProcessor()

# Process CPU-intensive calculations
results = processor.process_parallel(
    items,
    processor=cpu_intensive_calculation
)


# Example 3: Thread Pool for I/O-Bound Tasks
# ----------------------------------------------------------------------------
from processing.parallel_processor import ThreadPoolParallelProcessor

processor = ThreadPoolParallelProcessor()

# Process API calls in parallel
results = processor.process_parallel(
    api_urls,
    processor=fetch_from_api
)


# Example 4: Rate Limiting
# ----------------------------------------------------------------------------
from processing.parallel_processor import RateLimiter

rate_limiter = RateLimiter(max_rate_per_second=1000)

async def process_with_rate_limit(item):
    await rate_limiter.acquire()
    return await process_item(item)

results = await asyncio.gather(*[process_with_rate_limit(item) for item in items])


# Example 5: Adaptive Concurrency
# ----------------------------------------------------------------------------
from processing.parallel_processor import AdaptiveConcurrencyController

controller = AdaptiveConcurrencyController(
    initial_concurrency=10,
    target_cpu_percent=60.0
)

# Adjust concurrency dynamically
new_concurrency = await controller.adjust_concurrency()

# Use new concurrency level
processor = AsyncParallelProcessor()
processor.config.max_concurrent_tasks = new_concurrency
"""


__all__ = [
    'ParallelConfig',
    'AsyncParallelProcessor',
    'MultiprocessingParallelProcessor',
    'ThreadPoolParallelProcessor',
    'RateLimiter',
    'AdaptiveConcurrencyController',
    'HIGH_THROUGHPUT_CONFIG',
    'BALANCED_CONFIG',
    'LOW_RESOURCE_CONFIG',
]
