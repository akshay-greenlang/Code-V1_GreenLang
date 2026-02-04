"""
Integration tests for pgvector infrastructure.

These tests require a running PostgreSQL instance with pgvector extension.

To run:
    PGVECTOR_HOST=localhost PGVECTOR_DATABASE=greenlang_test pytest tests/integration/test_pgvector_integration.py

PRD: INFRA-005 Vector Database Infrastructure with pgvector
Acceptance Criteria: Search latency <100ms, recall >0.95, throughput >10K/s
"""

import asyncio
import os
import uuid

import numpy as np
import pytest

# Skip all tests if no database connection is configured
pytestmark = pytest.mark.skipif(
    not os.getenv("PGVECTOR_HOST"),
    reason="PGVECTOR_HOST not set - skipping integration tests",
)


@pytest.fixture
def db_config():
    from greenlang.data.vector.config import VectorDBConfig
    return VectorDBConfig.from_env()


@pytest.fixture
async def db_connection(db_config):
    from greenlang.data.vector.connection import VectorDBConnection
    db = VectorDBConnection(db_config)
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def test_namespace():
    return f"test_{uuid.uuid4().hex[:8]}"


class TestDatabaseConnection:
    @pytest.mark.asyncio
    async def test_health_check(self, db_connection):
        health = await db_connection.health_check()
        assert health["writer"] is True
        assert health["pgvector"] is True

    @pytest.mark.asyncio
    async def test_pgvector_extension_version(self, db_connection):
        health = await db_connection.health_check()
        assert "pgvector_version" in health
        # pgvector 0.7.0+
        version = health["pgvector_version"]
        major, minor = version.split(".")[:2]
        assert int(major) >= 0
        assert int(minor) >= 5


class TestEmbeddingOperations:
    @pytest.mark.asyncio
    async def test_insert_and_query(self, db_connection, test_namespace):
        """Scenario: Insert embeddings and retrieve via similarity search."""
        embedding = np.random.rand(384).astype(np.float32)
        record_id = str(uuid.uuid4())
        source_id = str(uuid.uuid4())

        # Insert
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO vector_embeddings (
                        id, source_type, source_id, chunk_index,
                        content_hash, content_preview, embedding,
                        embedding_model, metadata, namespace
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, '{}', %s)
                    """,
                    (
                        record_id, "document", source_id, 0,
                        "testhash", "test preview",
                        embedding, "all-MiniLM-L6-v2", test_namespace,
                    ),
                )
            await conn.commit()

        # Query
        async with db_connection.acquire_reader() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM vector_embeddings
                    WHERE namespace = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 1
                    """,
                    (embedding, test_namespace, embedding),
                )
                row = await cur.fetchone()
                assert row is not None
                assert str(row["id"]) == record_id
                assert row["similarity"] > 0.99  # Same vector

        # Cleanup
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM vector_embeddings WHERE namespace = %s",
                    (test_namespace,),
                )
            await conn.commit()

    @pytest.mark.asyncio
    async def test_similarity_search_latency(self, db_connection, test_namespace):
        """Scenario: Search latency < 100ms for top-10."""
        import time

        # Insert 100 random embeddings
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                for i in range(100):
                    emb = np.random.rand(384).astype(np.float32)
                    await cur.execute(
                        """
                        INSERT INTO vector_embeddings (
                            source_type, source_id, chunk_index,
                            content_hash, embedding,
                            embedding_model, namespace
                        ) VALUES ('document', %s, 0, 'hash', %s, 'all-MiniLM-L6-v2', %s)
                        """,
                        (str(uuid.uuid4()), emb, test_namespace),
                    )
            await conn.commit()

        # Search and measure latency
        query_emb = np.random.rand(384).astype(np.float32)
        start = time.monotonic()

        async with db_connection.acquire_reader() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, 1 - (embedding <=> %s::vector) AS similarity
                    FROM vector_embeddings
                    WHERE namespace = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT 10
                    """,
                    (query_emb, test_namespace, query_emb),
                )
                rows = await cur.fetchall()

        latency_ms = (time.monotonic() - start) * 1000
        assert len(rows) == 10
        assert latency_ms < 100, f"Search latency {latency_ms:.1f}ms exceeds 100ms SLO"

        # Cleanup
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM vector_embeddings WHERE namespace = %s",
                    (test_namespace,),
                )
            await conn.commit()


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_hybrid_search_rrf(self, db_connection, test_namespace):
        """Scenario: Hybrid search combines vector and text results."""
        # Insert documents with both embeddings and text
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                for i, text in enumerate([
                    "Climate change regulations require carbon reporting",
                    "Building energy efficiency standards for 2030",
                    "Supply chain emissions scope 3 methodology",
                ]):
                    emb = np.random.rand(384).astype(np.float32)
                    await cur.execute(
                        """
                        INSERT INTO vector_embeddings (
                            source_type, source_id, chunk_index,
                            content_hash, content_preview, embedding,
                            embedding_model, namespace
                        ) VALUES ('document', %s, 0, 'hash', %s, %s, 'all-MiniLM-L6-v2', %s)
                        """,
                        (str(uuid.uuid4()), text, emb, test_namespace),
                    )
            await conn.commit()

        # Hybrid search using stored function
        query_emb = np.random.rand(384).astype(np.float32)
        async with db_connection.acquire_reader() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT * FROM vector_hybrid_search(%s, %s, %s, %s)",
                    (query_emb, "carbon reporting", test_namespace, 10),
                )
                rows = await cur.fetchall()

        assert len(rows) >= 1

        # Cleanup
        async with db_connection.acquire_writer() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM vector_embeddings WHERE namespace = %s",
                    (test_namespace,),
                )
            await conn.commit()


class TestRBACRoles:
    @pytest.mark.asyncio
    async def test_roles_exist(self, db_connection):
        """Verify RBAC roles are created."""
        async with db_connection.acquire_reader() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT rolname FROM pg_roles WHERE rolname IN ('vector_reader', 'vector_writer', 'vector_admin')"
                )
                rows = await cur.fetchall()
                roles = {r["rolname"] for r in rows}
                assert "vector_reader" in roles
                assert "vector_writer" in roles
                assert "vector_admin" in roles
