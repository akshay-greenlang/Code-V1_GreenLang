"""
FastAPI router for Vector Search API endpoints.

Provides REST API for embedding generation, vector storage,
similarity search, hybrid search, and administration operations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/vectors", tags=["vectors"])


# ============================================================================
# Request/Response Models
# ============================================================================


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=10000)
    model_name: Optional[str] = Field(None, description="Embedding model override")
    namespace: str = Field("default", max_length=100)


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
    count: int
    processing_time_ms: int


class EmbedAndStoreRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=10000)
    source_type: str = Field("document", max_length=50)
    source_id: Optional[str] = None
    namespace: str = Field("default", max_length=100)
    metadata: Optional[Dict[str, Any]] = None
    collection_name: Optional[str] = None
    model_name: Optional[str] = None


class EmbedAndStoreResponse(BaseModel):
    total_count: int
    inserted_count: int
    failed_count: int
    duplicate_count: int
    processing_time_ms: int
    job_id: Optional[str] = None


class SearchRequestModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    namespace: str = Field("default", max_length=100)
    top_k: int = Field(10, ge=1, le=1000)
    threshold: float = Field(0.7, ge=0.0, le=1.0)
    source_type: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    ef_search: Optional[int] = Field(None, ge=10, le=1000)


class SearchMatchModel(BaseModel):
    id: str
    source_type: str
    source_id: str
    chunk_index: int
    content_preview: Optional[str]
    metadata: Dict[str, Any]
    similarity: float
    vector_rank: Optional[int] = None
    text_rank: Optional[int] = None
    rrf_score: Optional[float] = None


class SearchResponse(BaseModel):
    matches: List[SearchMatchModel]
    query_text: str
    total_results: int
    latency_ms: int
    search_type: str
    namespace: str


class HybridSearchRequestModel(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    namespace: str = Field("default", max_length=100)
    top_k: int = Field(10, ge=1, le=1000)
    rrf_k: int = Field(60, ge=1, le=200)


class CollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    namespace: str = Field("default", max_length=100)
    embedding_model: str = Field("all-MiniLM-L6-v2", max_length=100)
    dimensions: int = Field(384, ge=1, le=4096)
    distance_metric: str = Field("cosine")


class CollectionResponse(BaseModel):
    id: str
    name: str
    namespace: str
    embedding_model: str
    dimensions: int
    distance_metric: str
    vector_count: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class StatsResponse(BaseModel):
    total_embeddings: int
    namespace_counts: Dict[str, int]
    indexes: List[Dict[str, Any]]
    search_stats_1h: Dict[str, Any]


class JobStatusResponse(BaseModel):
    id: str
    status: str
    source_type: str
    source_count: int
    processed_count: int
    failed_count: int
    progress_pct: float
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: str


class HealthResponse(BaseModel):
    status: str
    writer: bool
    reader: bool
    pgvector: bool
    pgvector_version: Optional[str] = None


class BatchDeleteRequest(BaseModel):
    namespace: Optional[str] = None
    source_type: Optional[str] = None
    source_id: Optional[str] = None


class BatchDeleteResponse(BaseModel):
    deleted_count: int


# ============================================================================
# Dependency injection placeholder
# ============================================================================

# These will be set during app startup via configure_vector_api()
_embedding_service = None
_search_engine = None
_batch_processor = None
_index_manager = None
_db = None


def configure_vector_api(
    embedding_service,
    search_engine,
    batch_processor,
    index_manager,
    db,
):
    """Configure the vector API with service dependencies."""
    global _embedding_service, _search_engine, _batch_processor, _index_manager, _db
    _embedding_service = embedding_service
    _search_engine = search_engine
    _batch_processor = batch_processor
    _index_manager = index_manager
    _db = db


def _get_embedding_service():
    if _embedding_service is None:
        raise HTTPException(503, "Embedding service not configured")
    return _embedding_service


def _get_search_engine():
    if _search_engine is None:
        raise HTTPException(503, "Search engine not configured")
    return _search_engine


def _get_batch_processor():
    if _batch_processor is None:
        raise HTTPException(503, "Batch processor not configured")
    return _batch_processor


def _get_index_manager():
    if _index_manager is None:
        raise HTTPException(503, "Index manager not configured")
    return _index_manager


def _get_db():
    if _db is None:
        raise HTTPException(503, "Database not configured")
    return _db


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for a list of texts."""
    svc = _get_embedding_service()
    from greenlang.data.vector.models import EmbeddingRequest

    result = await svc.embed(
        EmbeddingRequest(
            texts=request.texts,
            namespace=request.namespace,
            model_name=request.model_name,
        )
    )
    return EmbedResponse(
        embeddings=result.embeddings.tolist(),
        model=result.model,
        dimensions=result.dimensions,
        count=result.count,
        processing_time_ms=result.processing_time_ms,
    )


@router.post("/embed-and-store", response_model=EmbedAndStoreResponse)
async def embed_and_store(request: EmbedAndStoreRequest):
    """Generate embeddings and store in pgvector."""
    svc = _get_embedding_service()
    from greenlang.data.vector.models import EmbeddingRequest

    result = await svc.embed_and_store(
        EmbeddingRequest(
            texts=request.texts,
            namespace=request.namespace,
            source_type=request.source_type,
            source_id=request.source_id,
            metadata=request.metadata,
            collection_name=request.collection_name,
            model_name=request.model_name,
        )
    )
    return EmbedAndStoreResponse(
        total_count=result.total_count,
        inserted_count=result.inserted_count,
        failed_count=result.failed_count,
        duplicate_count=result.duplicate_count,
        processing_time_ms=result.processing_time_ms,
        job_id=result.job_id,
    )


@router.post("/search", response_model=SearchResponse)
async def similarity_search(request: SearchRequestModel):
    """Perform similarity search using pgvector."""
    engine = _get_search_engine()
    from greenlang.data.vector.models import SearchRequest

    if request.source_type or request.metadata_filter:
        result = await engine.filtered_search(
            SearchRequest(
                query=request.query,
                namespace=request.namespace,
                top_k=request.top_k,
                threshold=request.threshold,
                source_type=request.source_type,
                metadata_filter=request.metadata_filter,
                ef_search=request.ef_search,
            )
        )
    else:
        result = await engine.similarity_search(
            SearchRequest(
                query=request.query,
                namespace=request.namespace,
                top_k=request.top_k,
                threshold=request.threshold,
                ef_search=request.ef_search,
            )
        )

    return SearchResponse(
        matches=[
            SearchMatchModel(
                id=m.id,
                source_type=m.source_type,
                source_id=m.source_id,
                chunk_index=m.chunk_index,
                content_preview=m.content_preview,
                metadata=m.metadata,
                similarity=m.similarity,
            )
            for m in result.matches
        ],
        query_text=result.query_text,
        total_results=result.total_results,
        latency_ms=result.latency_ms,
        search_type=result.search_type,
        namespace=result.namespace,
    )


@router.post("/hybrid-search", response_model=SearchResponse)
async def hybrid_search(request: HybridSearchRequestModel):
    """Perform hybrid search combining vector + full-text with RRF."""
    engine = _get_search_engine()
    from greenlang.data.vector.models import HybridSearchRequest

    result = await engine.hybrid_search(
        HybridSearchRequest(
            query=request.query,
            namespace=request.namespace,
            top_k=request.top_k,
            rrf_k=request.rrf_k,
        )
    )

    return SearchResponse(
        matches=[
            SearchMatchModel(
                id=m.id,
                source_type=m.source_type,
                source_id=m.source_id,
                chunk_index=m.chunk_index,
                content_preview=m.content_preview,
                metadata=m.metadata,
                similarity=m.similarity,
                vector_rank=m.vector_rank,
                text_rank=m.text_rank,
                rrf_score=m.rrf_score,
            )
            for m in result.matches
        ],
        query_text=result.query_text,
        total_results=result.total_results,
        latency_ms=result.latency_ms,
        search_type=result.search_type,
        namespace=result.namespace,
    )


@router.post("/collections", response_model=CollectionResponse, status_code=201)
async def create_collection(request: CollectionCreateRequest):
    """Create a new embedding collection."""
    db = _get_db()
    import uuid

    collection_id = str(uuid.uuid4())
    await db.execute(
        """
        INSERT INTO embedding_collections (id, name, description, namespace,
            embedding_model, dimensions, distance_metric)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            collection_id,
            request.name,
            request.description,
            request.namespace,
            request.embedding_model,
            request.dimensions,
            request.distance_metric,
        ),
    )

    row = await db.execute_one(
        "SELECT * FROM embedding_collections WHERE id = %s", (collection_id,)
    )
    return _collection_from_row(row)


@router.get("/collections", response_model=List[CollectionResponse])
async def list_collections(namespace: Optional[str] = Query(None)):
    """List all embedding collections."""
    db = _get_db()
    if namespace:
        rows = await db.execute(
            "SELECT * FROM embedding_collections WHERE namespace = %s ORDER BY name",
            (namespace,),
            use_reader=True,
        )
    else:
        rows = await db.execute(
            "SELECT * FROM embedding_collections ORDER BY name",
            use_reader=True,
        )
    return [_collection_from_row(r) for r in rows]


@router.get("/stats", response_model=StatsResponse)
async def get_stats(namespace: Optional[str] = Query(None)):
    """Get vector database statistics."""
    engine = _get_search_engine()
    stats = await engine.get_stats(namespace=namespace)
    return StatsResponse(**stats)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of an embedding job."""
    processor = _get_batch_processor()
    job = await processor.get_job_status(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        source_type=job.source_type,
        source_count=job.source_count,
        processed_count=job.processed_count,
        failed_count=job.failed_count,
        progress_pct=job.progress_pct,
        error_message=job.error_message,
        started_at=str(job.started_at) if job.started_at else None,
        completed_at=str(job.completed_at) if job.completed_at else None,
        created_at=str(job.created_at),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check vector database health."""
    db = _get_db()
    health = await db.health_check()
    status = "healthy" if health.get("writer") and health.get("pgvector") else "unhealthy"
    return HealthResponse(
        status=status,
        writer=health.get("writer", False),
        reader=health.get("reader", False),
        pgvector=health.get("pgvector", False),
        pgvector_version=health.get("pgvector_version"),
    )


@router.delete("/embeddings", response_model=BatchDeleteResponse)
async def delete_embeddings(request: BatchDeleteRequest):
    """Delete embeddings by namespace or source."""
    processor = _get_batch_processor()
    if request.source_type and request.source_id:
        count = await processor.delete_by_source(request.source_type, request.source_id)
    elif request.namespace:
        count = await processor.delete_by_namespace(request.namespace)
    else:
        raise HTTPException(400, "Must specify namespace or source_type + source_id")
    return BatchDeleteResponse(deleted_count=count)


@router.get("/indexes")
async def list_indexes():
    """List all vector indexes with size and usage info."""
    mgr = _get_index_manager()
    return await mgr.get_index_health()


@router.post("/indexes/rebuild/{index_name}")
async def rebuild_index(index_name: str):
    """Rebuild a specific vector index."""
    mgr = _get_index_manager()
    await mgr.rebuild_index(index_name)
    return {"status": "rebuilt", "index": index_name}


def _collection_from_row(row: dict) -> CollectionResponse:
    return CollectionResponse(
        id=str(row["id"]),
        name=row["name"],
        namespace=row["namespace"],
        embedding_model=row["embedding_model"],
        dimensions=row["dimensions"],
        distance_metric=row["distance_metric"],
        vector_count=row.get("vector_count", 0),
        metadata=row.get("metadata", {}),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
