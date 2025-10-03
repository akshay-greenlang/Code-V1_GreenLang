"""
Pydantic schemas for RAG system components.

Core data models:
- DocMeta: Document metadata with version, publisher, checksums
- Chunk: Document chunk with section hierarchy and embeddings
- RAGCitation: Audit-ready citation for regulatory compliance
- QueryResult: Retrieval results with MMR scores and provenance
"""

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, HttpUrl, validator


class DocMeta(BaseModel):
    """
    Document metadata for regulatory compliance and audit trails.

    Tracks document provenance, version, publisher, and integrity hashes
    for citation and verification purposes.
    """

    doc_id: str = Field(
        description="Unique document identifier (UUID v5 from source URI/path)"
    )
    title: str = Field(description="Document title")
    collection: str = Field(
        description="Collection name (for allowlisting, e.g., 'ghg_protocol_corp')"
    )

    # Source information
    source_uri: Optional[str] = Field(
        default=None, description="Original document URI (http/https/file/gl)"
    )
    source_path: Optional[str] = Field(
        default=None, description="Local file path if applicable"
    )

    # Publishing information
    publisher: Optional[str] = Field(
        default=None, description="Publisher/standards body (e.g., 'WRI/WBCSD', 'IPCC')"
    )
    publication_date: Optional[date] = Field(
        default=None, description="Original publication date (ISO 8601)"
    )
    revision_date: Optional[date] = Field(
        default=None, description="Last revision date (for errata/amendments)"
    )
    version: Optional[str] = Field(
        default=None, description="Document version (e.g., '1.05', 'AR6 WG3')"
    )

    # Integrity and provenance
    content_hash: str = Field(
        description="SHA-256 hash of original document (for verification)"
    )
    doc_hash: str = Field(
        description="SHA-256 hash of canonicalized text + metadata (for determinism)"
    )

    # Ingestion metadata
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when document was ingested (UTC)",
    )
    ingested_by: Optional[str] = Field(
        default=None, description="User or system that ingested document"
    )
    pipeline_version: Optional[str] = Field(
        default=None, description="Ingestion pipeline version (for reproducibility)"
    )

    # Document properties
    total_chunks: Optional[int] = Field(
        default=None, description="Total number of chunks generated"
    )
    total_pages: Optional[int] = Field(default=None, description="Page count (for PDFs)")
    file_size_bytes: Optional[int] = Field(
        default=None, description="Original file size"
    )

    # Extra metadata (flexible for climate-specific fields)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., standard_id, compliance_level)",
    )

    @validator("collection")
    def validate_collection(cls, v):
        """Validate collection name (alphanumeric + underscore/hyphen only)."""
        import re

        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError("Collection name must be alphanumeric (+ underscore/hyphen)")
        if len(v) > 64:
            raise ValueError("Collection name must be ≤64 characters")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "doc_id": "a3f5b2c8-d1e6-5f9a-8b7c-2d5e8f1a4b7c",
                "title": "GHG Protocol Corporate Accounting and Reporting Standard",
                "collection": "ghg_protocol_corp",
                "source_uri": "https://ghgprotocol.org/sites/default/files/standards/ghg-protocol-revised.pdf",
                "publisher": "WRI/WBCSD",
                "publication_date": "2004-09-01",
                "revision_date": "2015-03-24",
                "version": "1.05",
                "content_hash": "a3f5b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4",
                "doc_hash": "b2c8d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2",
                "ingested_at": "2025-10-02T14:23:00Z",
                "ingested_by": "climate_science_team",
                "pipeline_version": "0.4.0",
                "total_chunks": 347,
                "total_pages": 112,
            }
        }


class Chunk(BaseModel):
    """
    Document chunk with section hierarchy and embedding metadata.

    Represents a semantically meaningful piece of a document with:
    - Stable chunk ID for deterministic retrieval
    - Section hierarchy (e.g., "Chapter 7 > 7.3.1")
    - Page/paragraph references for citations
    - Embedding hash for verification
    """

    chunk_id: str = Field(
        description="Deterministic UUID v5 from (doc_id, section_path, start_offset)"
    )
    doc_id: str = Field(description="Document identifier (foreign key to DocMeta)")

    # Section hierarchy
    section_path: str = Field(
        description="Hierarchical section path (e.g., 'Chapter 7 > 7.3.1 > Emission Factors')"
    )
    section_hash: str = Field(
        description="SHA-256 hash of section text + path (for citation verification)"
    )

    # Position within document
    page_start: Optional[int] = Field(
        default=None, description="Starting page number (1-indexed)"
    )
    page_end: Optional[int] = Field(default=None, description="Ending page number")
    paragraph: Optional[int] = Field(
        default=None, description="Paragraph index within section (0-indexed)"
    )
    start_char: int = Field(description="Character offset where chunk starts in document")
    end_char: int = Field(description="Character offset where chunk ends")

    # Content
    text: str = Field(description="Chunk text content")
    token_count: int = Field(
        description="Number of tokens (for budget tracking and context limits)"
    )

    # Embedding metadata (vector NOT stored in this model)
    embedding_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of embedding vector (for determinism verification)",
    )
    embedding_model: Optional[str] = Field(
        default=None, description="Embedding model used (e.g., 'all-MiniLM-L6-v2')"
    )

    # Extra metadata (tables, formulas, etc.)
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (e.g., table_data, formula_latex, footnotes)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "c8d1e6f9-a4b7-5c2d-8e1f-4a7b2c5d8e1f",
                "doc_id": "a3f5b2c8-d1e6-5f9a-8b7c-2d5e8f1a4b7c",
                "section_path": "Chapter 7 > Section 7.3 > 7.3.1 Emission Factors",
                "section_hash": "d1e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8",
                "page_start": 45,
                "page_end": 46,
                "paragraph": 2,
                "start_char": 12450,
                "end_char": 13100,
                "text": "For stationary combustion sources, the emission factor...",
                "token_count": 128,
                "embedding_hash": "e6f9a4b7c2d5e8f1",
                "embedding_model": "all-MiniLM-L6-v2",
            }
        }


class RAGCitation(BaseModel):
    """
    Audit-ready citation for regulatory compliance.

    Includes full provenance chain:
    - Document title, publisher, version
    - Section hierarchy and page numbers
    - Source URI with section anchor
    - Checksum for integrity verification
    """

    # Document information
    doc_title: str = Field(description="Document title")
    publisher: Optional[str] = Field(default=None, description="Publisher/standards body")
    version: Optional[str] = Field(default=None, description="Document version")
    publication_date: Optional[date] = Field(
        default=None, description="Publication date"
    )

    # Section information
    section_path: str = Field(description="Hierarchical section path")
    section_hash: str = Field(description="Section hash (for verification)")
    page_number: Optional[int] = Field(default=None, description="Page number")
    paragraph: Optional[int] = Field(default=None, description="Paragraph index")

    # Source information
    uri: Optional[str] = Field(default=None, description="Source URI")
    uri_fragment: Optional[str] = Field(
        default=None, description="URI fragment (e.g., #section_7_3_1)"
    )

    # Integrity
    checksum: str = Field(description="SHA-256 checksum of source document (first 8 chars)")

    # Formatted citation string
    formatted: str = Field(
        description="Human-readable citation string (for display and reports)"
    )

    # Relevance score (from retrieval)
    relevance_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Relevance score from retrieval"
    )

    @classmethod
    def from_chunk(cls, chunk: Chunk, doc_meta: DocMeta, relevance_score: float = 0.0):
        """
        Create citation from chunk and document metadata.

        Args:
            chunk: Document chunk
            doc_meta: Document metadata
            relevance_score: Relevance score from retrieval

        Returns:
            RAGCitation instance
        """
        # Format URI with section anchor
        uri_fragment = None
        if doc_meta.source_uri:
            # Create fragment from section_path (e.g., #section_7_3_1)
            section_id = chunk.section_path.replace(" ", "_").replace(">", "")
            uri_fragment = f"#{section_id}"

        # Create formatted citation string
        formatted_parts = []

        # Title and version
        title_part = doc_meta.title
        if doc_meta.version:
            title_part += f" v{doc_meta.version}"
        formatted_parts.append(title_part)

        # Publisher and date
        if doc_meta.publisher and doc_meta.publication_date:
            formatted_parts.append(
                f"({doc_meta.publisher}, {doc_meta.publication_date.isoformat()})"
            )
        elif doc_meta.publisher:
            formatted_parts.append(f"({doc_meta.publisher})")

        # Section and page
        section_part = chunk.section_path
        if chunk.paragraph is not None:
            section_part += f", para {chunk.paragraph}"
        if chunk.page_start:
            section_part += f", p.{chunk.page_start}"
        formatted_parts.append(section_part)

        # URI
        if doc_meta.source_uri:
            uri_str = doc_meta.source_uri
            if uri_fragment:
                uri_str += uri_fragment
            formatted_parts.append(uri_str)

        # Checksum
        formatted_parts.append(f"SHA256:{doc_meta.content_hash[:8]}")

        formatted = ", ".join(formatted_parts)

        return cls(
            doc_title=doc_meta.title,
            publisher=doc_meta.publisher,
            version=doc_meta.version,
            publication_date=doc_meta.publication_date,
            section_path=chunk.section_path,
            section_hash=chunk.section_hash,
            page_number=chunk.page_start,
            paragraph=chunk.paragraph,
            uri=doc_meta.source_uri,
            uri_fragment=uri_fragment,
            checksum=doc_meta.content_hash[:8],
            formatted=formatted,
            relevance_score=relevance_score,
        )

    class Config:
        json_schema_extra = {
            "example": {
                "doc_title": "GHG Protocol Corporate Standard",
                "publisher": "WRI/WBCSD",
                "version": "1.05",
                "publication_date": "2015-03-24",
                "section_path": "Chapter 7 > 7.3.1",
                "section_hash": "d1e6f9a4b7c2d5e8",
                "page_number": 45,
                "paragraph": 2,
                "uri": "https://ghgprotocol.org/standard.pdf",
                "uri_fragment": "#Chapter_7_7.3.1",
                "checksum": "a3f5b2c8",
                "formatted": "GHG Protocol Corporate Standard v1.05 (WRI/WBCSD, 2015-03-24), Chapter 7 > 7.3.1, para 2, p.45, https://ghgprotocol.org/standard.pdf#Chapter_7_7.3.1, SHA256:a3f5b2c8",
                "relevance_score": 0.87,
            }
        }


class QueryResult(BaseModel):
    """
    Result from RAG retrieval with MMR-diversified chunks and citations.
    """

    # Query information
    query: str = Field(description="Original query string")
    query_hash: str = Field(
        description="SHA-256 hash of query + params (for caching)"
    )

    # Retrieved chunks
    chunks: List[Chunk] = Field(description="Retrieved document chunks")
    citations: List[RAGCitation] = Field(
        description="Audit-ready citations for each chunk"
    )

    # Retrieval metadata
    relevance_scores: List[float] = Field(
        description="Relevance scores for each chunk (0.0-1.0)"
    )
    retrieval_method: Literal["similarity", "mmr", "hybrid"] = Field(
        description="Retrieval method used"
    )
    search_time_ms: int = Field(description="Search time in milliseconds")

    # Cost tracking
    total_tokens: int = Field(
        description="Total tokens in retrieved chunks (for context budget)"
    )
    total_chunks: int = Field(description="Number of chunks retrieved")

    # Collections searched
    collections_searched: List[str] = Field(
        description="Collections that were searched"
    )

    # Extra metadata
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retrieval metadata (e.g., fetch_k, lambda_mult for MMR)",
    )

    @validator("chunks", "citations", "relevance_scores")
    def validate_lists_same_length(cls, v, values):
        """Ensure chunks, citations, and scores have same length."""
        if "chunks" in values and len(v) != len(values["chunks"]):
            raise ValueError("chunks, citations, and relevance_scores must have same length")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "emission factors for stationary combustion",
                "query_hash": "e6f9a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1a4b7c2d5e8f1",
                "chunks": [],  # List of Chunk objects
                "citations": [],  # List of RAGCitation objects
                "relevance_scores": [0.92, 0.87, 0.81, 0.76, 0.68],
                "retrieval_method": "mmr",
                "search_time_ms": 42,
                "total_tokens": 640,
                "total_chunks": 5,
                "collections_searched": ["ghg_protocol_corp", "ipcc_ar6_wg3"],
                "extra": {"fetch_k": 30, "lambda_mult": 0.5},
            }
        }


class IngestionManifest(BaseModel):
    """
    Audit trail for document ingestion (MANIFEST.json).

    Tracks who ingested what, when, and how for regulatory compliance.
    """

    manifest_version: str = Field(default="1.0", description="Manifest schema version")
    collection_id: str = Field(description="Collection identifier")

    # Documents ingested
    documents: List[DocMeta] = Field(description="List of ingested documents")

    # Ingestion metadata
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When ingestion completed (UTC)"
    )
    ingestion_duration_seconds: Optional[float] = Field(
        default=None, description="Total ingestion time"
    )

    # Pipeline configuration
    pipeline_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ingestion pipeline configuration (chunking params, etc.)",
    )

    # Transformations applied
    transformations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of transformations (table extraction, formula parsing, etc.)",
    )

    # Approval (for governance)
    approved_by: Optional[List[str]] = Field(
        default=None, description="List of approvers (for CSRB governance)"
    )
    approval_date: Optional[datetime] = Field(
        default=None, description="Approval timestamp"
    )

    # Vector store metadata
    vector_store_type: str = Field(description="Vector store used (FAISS/ChromaDB/Weaviate)")
    vector_store_config: Dict[str, Any] = Field(
        default_factory=dict, description="Vector store configuration"
    )
    total_embeddings: int = Field(description="Total number of embeddings generated")

    class Config:
        json_schema_extra = {
            "example": {
                "manifest_version": "1.0",
                "collection_id": "ghg_protocol_corp_v1.05",
                "documents": [],  # List of DocMeta objects
                "ingestion_timestamp": "2025-10-02T15:00:00Z",
                "ingestion_duration_seconds": 127.5,
                "pipeline_config": {
                    "chunk_size": 512,
                    "chunk_overlap": 64,
                    "chunking_strategy": "token_aware",
                },
                "transformations": [
                    {"type": "table_extraction", "tool": "Camelot", "version": "0.10.1"},
                    {"type": "section_path_extraction", "tool": "ClimateDocParser"},
                ],
                "approved_by": ["climate_scientist_1", "audit_lead"],
                "approval_date": "2025-10-03T10:00:00Z",
                "vector_store_type": "FAISS",
                "vector_store_config": {"index_type": "IndexFlatL2", "dimension": 384},
                "total_embeddings": 347,
            }
        }
