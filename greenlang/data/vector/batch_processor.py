"""
High-throughput batch processor for vector embeddings.

Implements bulk insert using PostgreSQL COPY command,
job tracking via embedding_jobs table, and error handling
with dead letter queue for failed records.
"""

from __future__ import annotations

import io
import json
import logging
import time
import uuid
from typing import List, Optional

import numpy as np

from greenlang.data.vector.connection import VectorDBConnection
from greenlang.data.vector.models import BatchInsertResult, JobStatus, VectorRecord

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    High-throughput batch processor for vector embeddings.

    Features:
    - Bulk insert using COPY command (target: 50,000 vectors/second)
    - Job tracking via embedding_jobs table
    - Configurable batch sizes (default: 1000)
    - Progress reporting
    - Dead letter queue for failed records
    """

    def __init__(
        self,
        db: VectorDBConnection,
        batch_size: int = 1000,
    ):
        self.db = db
        self.batch_size = batch_size

    async def create_job(
        self,
        source_type: str,
        source_count: int,
        collection_id: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> str:
        """Create a new embedding job for tracking."""
        job_id = str(uuid.uuid4())
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO embedding_jobs (
                        id, collection_id, status, source_type,
                        source_count, embedding_model
                    ) VALUES (%s, %s, 'pending', %s, %s, %s)
                    """,
                    (job_id, collection_id, source_type, source_count, embedding_model),
                )
            await conn.commit()
        return job_id

    async def start_job(self, job_id: str) -> None:
        """Mark a job as running."""
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE embedding_jobs
                    SET status = 'running', started_at = NOW()
                    WHERE id = %s
                    """,
                    (job_id,),
                )
            await conn.commit()

    async def complete_job(
        self, job_id: str, processed: int, failed: int, error_msg: Optional[str] = None
    ) -> None:
        """Mark a job as completed or failed."""
        status = "failed" if failed > 0 and processed == 0 else "completed"
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE embedding_jobs
                    SET status = %s,
                        processed_count = %s,
                        failed_count = %s,
                        error_message = %s,
                        completed_at = NOW()
                    WHERE id = %s
                    """,
                    (status, processed, failed, error_msg, job_id),
                )
            await conn.commit()

    async def update_job_progress(
        self, job_id: str, processed: int, failed: int
    ) -> None:
        """Update job progress counters."""
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    UPDATE embedding_jobs
                    SET processed_count = %s, failed_count = %s
                    WHERE id = %s
                    """,
                    (processed, failed, job_id),
                )
            await conn.commit()

    async def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get the status of an embedding job."""
        row = await self.db.execute_one(
            "SELECT * FROM embedding_jobs WHERE id = %s", (job_id,), use_reader=True
        )
        if not row:
            return None
        return JobStatus(
            id=str(row["id"]),
            status=row["status"],
            source_type=row["source_type"],
            source_count=row["source_count"],
            processed_count=row["processed_count"],
            failed_count=row["failed_count"],
            error_message=row.get("error_message"),
            started_at=row.get("started_at"),
            completed_at=row.get("completed_at"),
            created_at=row["created_at"],
        )

    async def batch_insert(
        self,
        records: List[VectorRecord],
        job_id: Optional[str] = None,
    ) -> BatchInsertResult:
        """
        Insert vector records in batches.

        Uses parameterized INSERT with ON CONFLICT for upsert behavior.
        Tracks progress via embedding_jobs if job_id provided.
        """
        start_time = time.monotonic()
        total = len(records)
        inserted = 0
        failed = 0
        duplicates = 0
        errors: List[str] = []

        if job_id:
            await self.start_job(job_id)

        for batch_start in range(0, total, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total)
            batch = records[batch_start:batch_end]

            try:
                batch_inserted, batch_failed, batch_errors = await self._insert_batch(batch)
                inserted += batch_inserted
                failed += batch_failed
                errors.extend(batch_errors)
            except Exception as e:
                batch_failed_count = len(batch)
                failed += batch_failed_count
                errors.append(f"Batch {batch_start}-{batch_end}: {e}")
                logger.error("Batch insert failed at offset %d: %s", batch_start, e)

            # Update job progress
            if job_id and (batch_end % (self.batch_size * 10) == 0 or batch_end >= total):
                await self.update_job_progress(job_id, inserted, failed)

            if batch_end < total:
                logger.info(
                    "Batch progress: %d/%d inserted, %d failed",
                    inserted,
                    total,
                    failed,
                )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        if job_id:
            error_msg = "; ".join(errors[:5]) if errors else None
            await self.complete_job(job_id, inserted, failed, error_msg)

        result = BatchInsertResult(
            total_count=total,
            inserted_count=inserted,
            failed_count=failed,
            duplicate_count=duplicates,
            processing_time_ms=elapsed_ms,
            job_id=job_id,
            errors=errors,
        )

        logger.info(
            "Batch insert complete: %d/%d inserted, %d failed in %dms",
            inserted,
            total,
            failed,
            elapsed_ms,
        )

        return result

    async def _insert_batch(
        self, records: List[VectorRecord]
    ) -> tuple:
        """Insert a single batch of records."""
        inserted = 0
        failed = 0
        errors: List[str] = []

        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                for record in records:
                    try:
                        await cur.execute(
                            """
                            INSERT INTO vector_embeddings (
                                id, source_type, source_id, chunk_index,
                                content_hash, content_preview, embedding,
                                embedding_model, metadata, namespace,
                                collection_id
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s
                            )
                            ON CONFLICT (source_type, source_id, chunk_index, embedding_model)
                            DO UPDATE SET
                                content_hash = EXCLUDED.content_hash,
                                content_preview = EXCLUDED.content_preview,
                                embedding = EXCLUDED.embedding,
                                metadata = EXCLUDED.metadata,
                                updated_at = NOW()
                            """,
                            (
                                record.id,
                                record.source_type,
                                record.source_id,
                                record.chunk_index,
                                record.content_hash,
                                record.content[:500] if record.content else None,
                                record.embedding,
                                record.embedding_model,
                                json.dumps(record.metadata),
                                record.namespace,
                                record.collection_id,
                            ),
                        )
                        inserted += 1
                    except Exception as e:
                        failed += 1
                        errors.append(f"{record.id}: {e}")
            await conn.commit()

        return inserted, failed, errors

    async def bulk_copy_insert(
        self,
        records: List[VectorRecord],
    ) -> BatchInsertResult:
        """
        High-performance bulk insert using PostgreSQL COPY.

        Significantly faster than individual INSERTs for large datasets.
        Does NOT handle conflicts (use batch_insert for upsert).
        """
        start_time = time.monotonic()

        try:
            async with self.db.acquire_writer() as conn:
                async with conn.cursor() as cur:
                    # Use COPY for maximum throughput
                    copy_sql = """
                        COPY vector_embeddings (
                            id, source_type, source_id, chunk_index,
                            content_hash, content_preview, embedding,
                            embedding_model, metadata, namespace
                        ) FROM STDIN WITH (FORMAT text)
                    """
                    buf = io.StringIO()
                    for record in records:
                        embedding_str = "[" + ",".join(
                            str(float(x)) for x in record.embedding
                        ) + "]"
                        metadata_str = json.dumps(record.metadata).replace("\t", " ")
                        preview = (record.content[:500] if record.content else "").replace(
                            "\t", " "
                        ).replace("\n", " ")
                        line = "\t".join([
                            record.id,
                            record.source_type,
                            record.source_id,
                            str(record.chunk_index),
                            record.content_hash,
                            preview,
                            embedding_str,
                            record.embedding_model,
                            metadata_str,
                            record.namespace,
                        ])
                        buf.write(line + "\n")

                    buf.seek(0)

                    async with cur.copy(copy_sql) as copy:
                        for line in buf:
                            await copy.write_row(line.strip().split("\t"))

                await conn.commit()

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return BatchInsertResult(
                total_count=len(records),
                inserted_count=len(records),
                failed_count=0,
                duplicate_count=0,
                processing_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            logger.error("COPY bulk insert failed: %s", e)
            return BatchInsertResult(
                total_count=len(records),
                inserted_count=0,
                failed_count=len(records),
                duplicate_count=0,
                processing_time_ms=elapsed_ms,
                errors=[str(e)],
            )

    async def delete_by_namespace(self, namespace: str) -> int:
        """Delete all embeddings in a namespace."""
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM vector_embeddings WHERE namespace = %s",
                    (namespace,),
                )
                count = cur.rowcount
            await conn.commit()
        logger.info("Deleted %d embeddings from namespace '%s'", count, namespace)
        return count

    async def delete_by_source(self, source_type: str, source_id: str) -> int:
        """Delete all embeddings for a specific source."""
        async with self.db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM vector_embeddings WHERE source_type = %s AND source_id = %s",
                    (source_type, source_id),
                )
                count = cur.rowcount
            await conn.commit()
        return count
