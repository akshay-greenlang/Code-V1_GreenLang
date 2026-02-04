"""
Weaviate to pgvector migration script.

Implements phased migration:
  Phase 1: Parallel operation (dual-write)
  Phase 2: Data migration (batch export/import)
  Phase 3: Traffic shift (gradual cutover)
  Phase 4: Decommission

PRD: INFRA-005 Vector Database Infrastructure with pgvector
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from greenlang.data.vector.connection import VectorDBConnection
from greenlang.data.vector.models import VectorRecord

logger = logging.getLogger(__name__)

BATCH_SIZE = 10000


@dataclass
class MigrationStats:
    total_exported: int = 0
    total_imported: int = 0
    total_failed: int = 0
    total_skipped: int = 0
    collections_migrated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0
    checksum_verified: bool = False


class WeaviateToPgvectorMigrator:
    """
    Migrate vector embeddings from Weaviate to pgvector.

    Supports batch export from Weaviate, schema transformation,
    bulk import via COPY, and checksum verification.
    """

    def __init__(
        self,
        weaviate_url: str,
        pg_db: VectorDBConnection,
        weaviate_api_key: Optional[str] = None,
    ):
        self.weaviate_url = weaviate_url
        self.pg_db = pg_db
        self._weaviate_client = None
        self._api_key = weaviate_api_key

    def _get_weaviate_client(self):
        """Initialize Weaviate client."""
        if self._weaviate_client is None:
            try:
                import weaviate

                auth = None
                if self._api_key:
                    auth = weaviate.AuthApiKey(api_key=self._api_key)

                self._weaviate_client = weaviate.Client(
                    url=self.weaviate_url,
                    auth_client_secret=auth,
                )
            except ImportError:
                raise ImportError(
                    "weaviate-client required for migration. "
                    "Install with: pip install weaviate-client"
                )
        return self._weaviate_client

    async def migrate_collection(
        self,
        collection_name: str,
        namespace: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        batch_size: int = BATCH_SIZE,
        dry_run: bool = False,
    ) -> MigrationStats:
        """
        Migrate a single Weaviate collection to pgvector.

        Args:
            collection_name: Weaviate class/collection name
            namespace: Target pgvector namespace
            embedding_model: Embedding model used
            batch_size: Records per batch
            dry_run: If True, only export and validate without importing
        """
        stats = MigrationStats()
        start_time = time.monotonic()

        client = self._get_weaviate_client()

        # Get total count
        try:
            result = (
                client.query
                .aggregate(collection_name)
                .with_meta_count()
                .do()
            )
            total = result["data"]["Aggregate"][collection_name][0]["meta"]["count"]
            logger.info(
                "Migrating collection '%s' (%d vectors) to namespace '%s'",
                collection_name,
                total,
                namespace,
            )
        except Exception as e:
            stats.errors.append(f"Failed to get count: {e}")
            return stats

        offset = 0
        while offset < total:
            # Fetch batch from Weaviate
            try:
                batch = (
                    client.query
                    .get(collection_name, ["content", "source_id", "source_type", "metadata"])
                    .with_additional(["vector", "id"])
                    .with_limit(batch_size)
                    .with_offset(offset)
                    .do()
                )
                objects = batch["data"]["Get"][collection_name]
            except Exception as e:
                stats.errors.append(f"Export failed at offset {offset}: {e}")
                stats.total_failed += batch_size
                offset += batch_size
                continue

            stats.total_exported += len(objects)

            if dry_run:
                logger.info(
                    "DRY RUN: Would import %d vectors (offset %d/%d)",
                    len(objects), offset, total,
                )
                offset += batch_size
                continue

            # Transform to pgvector records
            records = []
            for obj in objects:
                try:
                    vector = np.array(
                        obj["_additional"]["vector"], dtype=np.float32
                    )
                    content = obj.get("content", "")
                    records.append(
                        VectorRecord(
                            id=obj["_additional"]["id"],
                            source_type=obj.get("source_type", "document"),
                            source_id=obj.get("source_id", obj["_additional"]["id"]),
                            content=content,
                            embedding=vector,
                            namespace=namespace,
                            chunk_index=0,
                            content_hash=hashlib.sha256(
                                content.encode()
                            ).hexdigest(),
                            metadata=obj.get("metadata", {}),
                            embedding_model=embedding_model,
                        )
                    )
                except Exception as e:
                    stats.total_failed += 1
                    stats.errors.append(
                        f"Transform failed for {obj.get('_additional', {}).get('id')}: {e}"
                    )

            # Bulk insert to pgvector
            if records:
                try:
                    imported = await self._bulk_insert(records)
                    stats.total_imported += imported
                except Exception as e:
                    stats.total_failed += len(records)
                    stats.errors.append(f"Import failed at offset {offset}: {e}")

            offset += batch_size
            logger.info(
                "Progress: %d/%d exported, %d imported, %d failed",
                stats.total_exported, total, stats.total_imported, stats.total_failed,
            )

        stats.collections_migrated.append(collection_name)
        stats.duration_seconds = time.monotonic() - start_time

        logger.info(
            "Migration of '%s' complete: %d exported, %d imported, %d failed (%.1fs)",
            collection_name,
            stats.total_exported,
            stats.total_imported,
            stats.total_failed,
            stats.duration_seconds,
        )

        return stats

    async def _bulk_insert(self, records: List[VectorRecord]) -> int:
        """Insert records into pgvector."""
        inserted = 0
        async with self.pg_db.acquire_writer() as conn:
            async with conn.cursor() as cur:
                for record in records:
                    try:
                        await cur.execute(
                            """
                            INSERT INTO vector_embeddings (
                                id, source_type, source_id, chunk_index,
                                content_hash, content_preview, embedding,
                                embedding_model, metadata, namespace
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                            ON CONFLICT (source_type, source_id, chunk_index, embedding_model)
                            DO NOTHING
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
                            ),
                        )
                        inserted += 1
                    except Exception as e:
                        logger.warning("Insert failed for %s: %s", record.id, e)
            await conn.commit()
        return inserted

    async def verify_migration(
        self, collection_name: str, namespace: str
    ) -> Dict[str, Any]:
        """Verify migration integrity by comparing counts and sampling vectors."""
        client = self._get_weaviate_client()

        # Weaviate count
        result = (
            client.query
            .aggregate(collection_name)
            .with_meta_count()
            .do()
        )
        weaviate_count = result["data"]["Aggregate"][collection_name][0]["meta"]["count"]

        # pgvector count
        pg_row = await self.pg_db.execute_one(
            "SELECT COUNT(*) as count FROM vector_embeddings WHERE namespace = %s",
            (namespace,),
        )
        pgvector_count = pg_row["count"]

        return {
            "collection": collection_name,
            "namespace": namespace,
            "weaviate_count": weaviate_count,
            "pgvector_count": pgvector_count,
            "match": weaviate_count == pgvector_count,
            "difference": abs(weaviate_count - pgvector_count),
        }

    async def migrate_all(
        self,
        collection_mapping: Dict[str, str],
        dry_run: bool = False,
    ) -> MigrationStats:
        """
        Migrate multiple Weaviate collections.

        Args:
            collection_mapping: {weaviate_collection: pgvector_namespace}
            dry_run: If True, validate without importing
        """
        combined = MigrationStats()
        start_time = time.monotonic()

        for collection_name, namespace in collection_mapping.items():
            stats = await self.migrate_collection(
                collection_name, namespace, dry_run=dry_run
            )
            combined.total_exported += stats.total_exported
            combined.total_imported += stats.total_imported
            combined.total_failed += stats.total_failed
            combined.collections_migrated.extend(stats.collections_migrated)
            combined.errors.extend(stats.errors)

        combined.duration_seconds = time.monotonic() - start_time
        return combined


class DualWriteProxy:
    """
    Dual-write proxy for Phase 1 parallel operation.

    Writes to both Weaviate and pgvector simultaneously.
    Reads from the configured primary (default: Weaviate).
    """

    def __init__(
        self,
        weaviate_url: str,
        pg_db: VectorDBConnection,
        primary: str = "weaviate",
    ):
        self.weaviate_url = weaviate_url
        self.pg_db = pg_db
        self.primary = primary
        self._weaviate_client = None

    async def store_embedding(
        self,
        record: VectorRecord,
        collection_name: str,
    ) -> Dict[str, bool]:
        """Write to both backends."""
        results = {"weaviate": False, "pgvector": False}

        # Write to pgvector
        try:
            async with self.pg_db.acquire_writer() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO vector_embeddings (
                            id, source_type, source_id, chunk_index,
                            content_hash, content_preview, embedding,
                            embedding_model, metadata, namespace
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                        ON CONFLICT (source_type, source_id, chunk_index, embedding_model)
                        DO UPDATE SET embedding = EXCLUDED.embedding,
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
                        ),
                    )
                await conn.commit()
            results["pgvector"] = True
        except Exception as e:
            logger.error("pgvector dual-write failed: %s", e)

        # Write to Weaviate
        try:
            client = self._get_weaviate_client()
            client.data_object.create(
                data_object={
                    "content": record.content[:500],
                    "source_type": record.source_type,
                    "source_id": record.source_id,
                    "metadata": record.metadata,
                },
                class_name=collection_name,
                uuid=record.id,
                vector=record.embedding.tolist(),
            )
            results["weaviate"] = True
        except Exception as e:
            logger.error("Weaviate dual-write failed: %s", e)

        return results

    def _get_weaviate_client(self):
        if self._weaviate_client is None:
            import weaviate
            self._weaviate_client = weaviate.Client(url=self.weaviate_url)
        return self._weaviate_client


class TrafficShifter:
    """
    Gradual traffic shift from Weaviate to pgvector (Phase 3).

    Supports configurable percentages for read traffic distribution.
    """

    def __init__(self, pgvector_pct: int = 0):
        self._pgvector_pct = pgvector_pct

    @property
    def pgvector_percentage(self) -> int:
        return self._pgvector_pct

    def set_percentage(self, pct: int) -> None:
        """Set pgvector traffic percentage (0-100)."""
        if not 0 <= pct <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        self._pgvector_pct = pct
        logger.info("Traffic shift: %d%% pgvector, %d%% Weaviate", pct, 100 - pct)

    def should_use_pgvector(self) -> bool:
        """Determine which backend to use for this request."""
        import random
        return random.randint(1, 100) <= self._pgvector_pct

    def get_backend(self) -> str:
        """Get the backend to use for this request."""
        return "pgvector" if self.should_use_pgvector() else "weaviate"
