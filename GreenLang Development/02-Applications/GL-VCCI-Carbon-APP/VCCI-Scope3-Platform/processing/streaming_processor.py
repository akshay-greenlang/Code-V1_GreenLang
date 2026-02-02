# -*- coding: utf-8 -*-
"""
Streaming Data Processor
GL-VCCI Scope 3 Platform - High-Performance Batch Processing

This module provides streaming processing capabilities for 100K+ suppliers/hour:
- Async generators for memory-efficient processing
- Database cursor streaming
- CSV streaming without loading full file
- Chunked processing for large datasets
- Backpressure handling

Performance Targets:
- Throughput: 100,000 suppliers/hour (1,666/min)
- Memory: <8GB for 100K suppliers
- Latency: P95 <500ms per supplier

Version: 1.0.0
Team: Performance & Batch Processing (Team 5)
Date: 2025-11-09
"""

import logging
import asyncio
import csv
import aiofiles
from typing import AsyncIterator, Iterator, List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)


# ============================================================================
# STREAMING CONFIGURATION
# ============================================================================

@dataclass
class StreamConfig:
    """Streaming processor configuration"""
    chunk_size: int = 1000  # Records per chunk
    buffer_size: int = 5000  # Buffer for backpressure
    max_memory_mb: int = 8000  # 8GB max memory
    enable_backpressure: bool = True
    prefetch_chunks: int = 2  # Number of chunks to prefetch


# ============================================================================
# ASYNC STREAMING PROCESSOR
# ============================================================================

class AsyncStreamingProcessor:
    """
    High-performance streaming processor using async generators.

    Features:
    - Memory-efficient streaming
    - Async generators for non-blocking I/O
    - Backpressure handling
    - Progress tracking
    - Error recovery
    """

    def __init__(self, config: Optional[StreamConfig] = None):
        """
        Initialize streaming processor.

        Args:
            config: Streaming configuration
        """
        self.config = config or StreamConfig()
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None

    async def stream_from_database(
        self,
        session,
        query,
        chunk_size: Optional[int] = None
    ) -> AsyncIterator[List[Any]]:
        """
        Stream results from database in chunks using server-side cursor.

        Args:
            session: Database session
            query: SQLAlchemy query
            chunk_size: Records per chunk

        Yields:
            Chunks of database records
        """
        chunk_size = chunk_size or self.config.chunk_size

        logger.info(f"Starting database streaming with chunk_size={chunk_size}")

        # Use server-side cursor for streaming
        # This prevents loading all results into memory
        result = await session.stream(query)

        chunk = []
        async for row in result:
            chunk.append(row)

            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []

        # Yield remaining records
        if chunk:
            yield chunk

    async def stream_from_csv(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
        skip_header: bool = True
    ) -> AsyncIterator[List[Dict[str, Any]]]:
        """
        Stream CSV file in chunks without loading entire file.

        Args:
            file_path: Path to CSV file
            chunk_size: Records per chunk
            skip_header: Whether to skip header row

        Yields:
            Chunks of CSV records as dictionaries
        """
        chunk_size = chunk_size or self.config.chunk_size

        logger.info(f"Starting CSV streaming: {file_path}")

        async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
            # Read header
            header_line = await f.readline()
            headers = header_line.strip().split(',')

            chunk = []
            line_num = 1

            async for line in f:
                line_num += 1

                try:
                    # Parse CSV line
                    values = line.strip().split(',')

                    if len(values) != len(headers):
                        logger.warning(
                            f"Line {line_num}: Column count mismatch "
                            f"(expected {len(headers)}, got {len(values)})"
                        )
                        continue

                    # Create record dictionary
                    record = dict(zip(headers, values))
                    chunk.append(record)

                    # Yield chunk when full
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []

                        # Allow other tasks to run
                        await asyncio.sleep(0)

                except Exception as e:
                    logger.error(f"Error parsing line {line_num}: {e}")
                    self.error_count += 1
                    continue

            # Yield remaining records
            if chunk:
                yield chunk

        logger.info(
            f"CSV streaming completed: {line_num} lines, "
            f"{self.error_count} errors"
        )

    async def stream_from_list(
        self,
        records: List[Any],
        chunk_size: Optional[int] = None
    ) -> AsyncIterator[List[Any]]:
        """
        Stream from in-memory list in chunks.

        Args:
            records: List of records
            chunk_size: Records per chunk

        Yields:
            Chunks of records
        """
        chunk_size = chunk_size or self.config.chunk_size

        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            yield chunk

            # Allow other tasks to run
            await asyncio.sleep(0)

    async def process_stream(
        self,
        stream: AsyncIterator[List[Any]],
        processor: Callable[[List[Any]], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Any]:
        """
        Process streaming data with a processing function.

        Args:
            stream: Async iterator of data chunks
            processor: Function to process each chunk
            progress_callback: Optional callback for progress updates

        Returns:
            List of processing results
        """
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0

        results = []
        chunk_num = 0

        async for chunk in stream:
            chunk_num += 1

            try:
                # Process chunk
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(chunk)
                else:
                    result = processor(chunk)

                results.append(result)
                self.processed_count += len(chunk)

                # Progress callback
                if progress_callback:
                    progress_callback(chunk_num, self.processed_count)

                # Log progress
                if chunk_num % 10 == 0:
                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / elapsed if elapsed > 0 else 0

                    logger.info(
                        f"Progress: {self.processed_count} records, "
                        f"{rate:.1f} rec/sec, "
                        f"chunk {chunk_num}"
                    )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num}: {e}")
                self.error_count += len(chunk)
                continue

        # Final statistics
        elapsed = time.time() - self.start_time
        rate = self.processed_count / elapsed if elapsed > 0 else 0

        logger.info(
            f"Streaming completed: "
            f"{self.processed_count} records processed, "
            f"{self.error_count} errors, "
            f"{elapsed:.2f}s, "
            f"{rate:.1f} rec/sec"
        )

        return results

    async def process_with_backpressure(
        self,
        stream: AsyncIterator[List[Any]],
        processor: Callable[[List[Any]], Any],
        max_concurrent: int = 10
    ) -> List[Any]:
        """
        Process stream with backpressure control.

        Limits concurrent processing to prevent overwhelming system.

        Args:
            stream: Async iterator of data chunks
            processor: Function to process each chunk
            max_concurrent: Maximum concurrent chunk processing

        Returns:
            List of processing results
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        tasks = []

        async def process_with_semaphore(chunk, chunk_num):
            """Process chunk with semaphore control"""
            async with semaphore:
                try:
                    logger.debug(f"Processing chunk {chunk_num}")

                    if asyncio.iscoroutinefunction(processor):
                        result = await processor(chunk)
                    else:
                        result = processor(chunk)

                    self.processed_count += len(chunk)
                    return result

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {e}")
                    self.error_count += len(chunk)
                    return None

        # Create tasks for all chunks
        chunk_num = 0
        async for chunk in stream:
            chunk_num += 1
            task = asyncio.create_task(process_with_semaphore(chunk, chunk_num))
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        return [r for r in results if r is not None and not isinstance(r, Exception)]


# ============================================================================
# DATABASE CHUNK ITERATOR
# ============================================================================

class DatabaseChunkIterator:
    """
    Efficient database chunk iterator using LIMIT/OFFSET.

    Prevents loading entire result set into memory.
    """

    def __init__(
        self,
        session,
        query,
        chunk_size: int = 1000
    ):
        """
        Initialize chunk iterator.

        Args:
            session: Database session
            query: SQLAlchemy query
            chunk_size: Records per chunk
        """
        self.session = session
        self.query = query
        self.chunk_size = chunk_size
        self.offset = 0

    async def __aiter__(self):
        """Async iterator protocol"""
        return self

    async def __anext__(self) -> List[Any]:
        """Get next chunk"""
        # Add LIMIT and OFFSET to query
        chunk_query = (
            self.query
            .limit(self.chunk_size)
            .offset(self.offset)
        )

        # Execute query
        result = await self.session.execute(chunk_query)
        rows = result.scalars().all()

        # Stop iteration if no more rows
        if not rows:
            raise StopAsyncIteration

        # Update offset for next chunk
        self.offset += self.chunk_size

        return rows


# ============================================================================
# MEMORY-EFFICIENT GENERATORS
# ============================================================================

async def generate_supplier_batches(
    session,
    tenant_id: str,
    batch_size: int = 1000
) -> AsyncIterator[List[Dict[str, Any]]]:
    """
    Generate supplier batches from database without loading all.

    Args:
        session: Database session
        tenant_id: Tenant identifier
        batch_size: Records per batch

    Yields:
        Batches of supplier records
    """
    from sqlalchemy import select, text

    offset = 0

    while True:
        # Query with LIMIT/OFFSET
        query = text("""
            SELECT
                supplier_id,
                supplier_name,
                duns_number,
                lei_code,
                spend_usd
            FROM suppliers
            WHERE tenant_id = :tenant_id
            ORDER BY supplier_id
            LIMIT :limit OFFSET :offset
        """)

        result = await session.execute(
            query,
            {
                "tenant_id": tenant_id,
                "limit": batch_size,
                "offset": offset
            }
        )

        rows = result.fetchall()

        # Stop if no more rows
        if not rows:
            break

        # Convert to dictionaries
        batch = [
            {
                "supplier_id": row[0],
                "supplier_name": row[1],
                "duns_number": row[2],
                "lei_code": row[3],
                "spend_usd": row[4]
            }
            for row in rows
        ]

        yield batch

        offset += batch_size

        # Allow other tasks to run
        await asyncio.sleep(0)


def csv_line_generator(file_path: str) -> Iterator[str]:
    """
    Generator for reading CSV line by line.

    Args:
        file_path: Path to CSV file

    Yields:
        Individual CSV lines
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)

        for line in f:
            yield line.strip()


async def async_csv_generator(file_path: str) -> AsyncIterator[Dict[str, Any]]:
    """
    Async generator for CSV records.

    Args:
        file_path: Path to CSV file

    Yields:
        Individual CSV records as dictionaries
    """
    async with aiofiles.open(file_path, mode='r', encoding='utf-8') as f:
        # Read and parse header
        header_line = await f.readline()
        headers = header_line.strip().split(',')

        # Stream records
        async for line in f:
            values = line.strip().split(',')

            if len(values) == len(headers):
                yield dict(zip(headers, values))


# ============================================================================
# CHUNKING UTILITIES
# ============================================================================

def chunk_list(items: List[Any], chunk_size: int) -> Iterator[List[Any]]:
    """
    Split list into chunks.

    Args:
        items: List to chunk
        chunk_size: Size of each chunk

    Yields:
        Chunks of items
    """
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


async def async_chunk_processor(
    items: List[Any],
    processor: Callable[[Any], Any],
    chunk_size: int = 1000,
    max_concurrent: int = 10
) -> List[Any]:
    """
    Process items in chunks with concurrency control.

    Args:
        items: Items to process
        processor: Function to process each item
        chunk_size: Items per chunk
        max_concurrent: Maximum concurrent chunks

    Returns:
        List of results
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_chunk(chunk: List[Any]) -> List[Any]:
        async with semaphore:
            if asyncio.iscoroutinefunction(processor):
                return await asyncio.gather(*[processor(item) for item in chunk])
            else:
                return [processor(item) for item in chunk]

    # Create chunks
    chunks = list(chunk_list(items, chunk_size))

    # Process all chunks
    chunk_results = await asyncio.gather(*[process_chunk(chunk) for chunk in chunks])

    # Flatten results
    return [item for chunk in chunk_results for item in chunk]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

EXAMPLE_USAGE = """
# ============================================================================
# Streaming Processor Usage Examples
# ============================================================================

# Example 1: Stream from CSV
# ----------------------------------------------------------------------------
from processing.streaming_processor import AsyncStreamingProcessor

processor = AsyncStreamingProcessor()

# Stream and process CSV
async def process_csv():
    async for chunk in processor.stream_from_csv(
        'suppliers.csv',
        chunk_size=1000
    ):
        # Process chunk
        results = await calculate_emissions(chunk)
        await save_results(results)


# Example 2: Stream from Database
# ----------------------------------------------------------------------------
from processing.streaming_processor import generate_supplier_batches

async def process_suppliers(session, tenant_id):
    async for batch in generate_supplier_batches(
        session,
        tenant_id,
        batch_size=1000
    ):
        # Process batch
        emissions = await calculator.calculate_batch(batch, category=1)
        await save_emissions(emissions)


# Example 3: Process with Backpressure
# ----------------------------------------------------------------------------
processor = AsyncStreamingProcessor()

# Create stream
stream = processor.stream_from_csv('large_file.csv')

# Process with backpressure control
results = await processor.process_with_backpressure(
    stream,
    processor=calculate_emissions,
    max_concurrent=10  # Limit concurrent processing
)


# Example 4: Database Chunk Iterator
# ----------------------------------------------------------------------------
from processing.streaming_processor import DatabaseChunkIterator
from sqlalchemy import select

# Create query
query = select(Supplier).where(Supplier.tenant_id == tenant_id)

# Iterate in chunks
async for chunk in DatabaseChunkIterator(session, query, chunk_size=1000):
    await process_chunk(chunk)


# Example 5: Memory-Efficient CSV Processing
# ----------------------------------------------------------------------------
from processing.streaming_processor import async_csv_generator

# Process one record at a time
async for record in async_csv_generator('suppliers.csv'):
    result = await calculate_single_emission(record)
    await save_result(result)
"""


__all__ = [
    "StreamConfig",
    "AsyncStreamingProcessor",
    "DatabaseChunkIterator",
    "generate_supplier_batches",
    "csv_line_generator",
    "async_csv_generator",
    "chunk_list",
    "async_chunk_processor",
]
