"""
Document ingestion for GreenLang RAG system.

Handles PDF and Markdown ingestion with deterministic chunking,
embedding generation, and vector store indexing.

Per CTO spec Section 7:
1. Parse file(s) (pdf/md) -> sections
2. Build canonical doc text + header map -> doc_hash
3. Chunk -> Chunk[]
4. Embed -> vectors
5. Upsert into vector store (batch)
6. Write MANIFEST.json with doc_id, doc_hash, n_chunks, timestamps
"""

import time
import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from greenlang.intelligence.rag.models import (
    DocMeta,
    Chunk,
    IngestionManifest,
)
from greenlang.intelligence.rag.config import (
    RAGConfig,
    get_config,
    is_collection_allowed,
)
from greenlang.intelligence.rag.embeddings import get_embedding_provider
from greenlang.intelligence.rag.vector_stores import get_vector_store, Document
from greenlang.intelligence.rag.chunker import get_chunker
from greenlang.intelligence.rag.hashing import (
    file_hash,
    canonicalize_text,
    chunk_uuid5,
    section_hash,
)
from greenlang.intelligence.rag.sanitize import (
    validate_collection_name,
    detect_suspicious_content,
)

logger = logging.getLogger(__name__)


async def ingest_path(
    path: Path,
    collection: str,
    doc_meta: DocMeta,
    config: Optional[RAGConfig] = None,
    extract_tables: bool = False,
    extract_formulas: bool = False,
) -> IngestionManifest:
    """
    Ingest document(s) from path into collection.

    Steps (per CTO spec Section 7):
    1. Parse file(s) (pdf/md) -> sections
    2. Build canonical doc text + header map -> doc_hash
    3. Chunk -> Chunk[]
    4. Embed -> vectors
    5. Upsert into vector store (batch)
    6. Write MANIFEST.json with doc_id, doc_hash, n_chunks, timestamps

    Args:
        path: Path to document file
        collection: Collection name (must be in allowlist)
        doc_meta: Document metadata
        config: RAG configuration (defaults to global config)
        extract_tables: Extract tables from document
        extract_formulas: Extract formulas from document

    Returns:
        IngestionManifest with ingestion details

    Raises:
        ValueError: If collection not in allowlist
        FileNotFoundError: If file not found
        RuntimeError: If file hash doesn't match doc_meta.content_hash
    """
    start_time = time.time()
    config = config or get_config()

    # Step 1: Validate collection allowlist
    if not validate_collection_name(collection):
        raise ValueError(f"Invalid collection name: {collection}")

    if not is_collection_allowed(collection, config):
        raise ValueError(
            f"Collection '{collection}' is not in allowlist. "
            f"Allowed: {', '.join(config.allowlist)}"
        )

    # Step 2: Verify file exists
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Step 3: Verify file hash
    if config.verify_checksums:
        computed_hash = file_hash(str(path))
        if computed_hash != doc_meta.content_hash:
            raise RuntimeError(
                f"File hash mismatch for {path}\n"
                f"Expected: {doc_meta.content_hash}\n"
                f"Computed: {computed_hash}\n"
                "File may have been tampered with or metadata is incorrect."
            )

    # Step 4: Extract text from document
    text = _extract_text(path)

    # Check for suspicious content
    if config.enable_sanitization:
        warning = detect_suspicious_content(text)
        if warning:
            print(f"[SECURITY WARNING] {warning}")

    # Step 5: Initialize components
    embedder = get_embedding_provider(
        provider=config.embedding_provider,
        model_name=config.embedding_model,
    )

    vector_store = get_vector_store(
        dimension=config.embedding_dimension,
        config=config,
    )

    chunker = get_chunker(
        strategy=config.chunking_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    # Step 6: Chunk document
    chunks = _chunk_document(
        text=text,
        doc_meta=doc_meta,
        collection=collection,
        chunker=chunker,
        config=config,
    )

    # Step 7: Generate embeddings (batch processing)
    embeddings = await _generate_embeddings(
        texts=[chunk.text for chunk in chunks],
        embedder=embedder,
        config=config,
    )

    # Step 8: Store in vector store
    await _store_chunks(
        chunks=chunks,
        embeddings=embeddings,
        collection=collection,
        vector_store=vector_store,
    )

    # Step 9: Create manifest
    duration = time.time() - start_time
    manifest = IngestionManifest(
        collection_id=collection,
        documents=[doc_meta],
        ingestion_duration_seconds=duration,
        pipeline_config={
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunking_strategy": config.chunking_strategy,
            "embedding_model": config.embedding_model,
            "embedding_dimension": config.embedding_dimension,
        },
        vector_store_type=config.vector_store_provider,
        vector_store_config={
            "provider": config.vector_store_provider,
            "path": config.vector_store_path,
        },
        total_embeddings=len(embeddings),
    )

    return manifest


async def ingest_directory(
    directory: Path,
    collection: str,
    pattern: str = "**/*.{pdf,md}",
    config: Optional[RAGConfig] = None,
    meta_overrides: Optional[Dict] = None,
) -> IngestionManifest:
    """
    Ingest all matching files in directory.

    Args:
        directory: Directory to search for files
        collection: Collection name
        pattern: Glob pattern for files (default: pdf and markdown)
        config: RAG configuration
        meta_overrides: Metadata overrides to apply to all documents

    Returns:
        Combined IngestionManifest for all documents
    """
    config = config or get_config()

    # Find all matching files
    # Note: Python glob doesn't support {pdf,md} syntax, need to handle separately
    pdf_files = list(directory.glob("**/*.pdf"))
    md_files = list(directory.glob("**/*.md"))
    all_files = pdf_files + md_files

    if not all_files:
        raise ValueError(f"No files found matching pattern in {directory}")

    all_chunks = []
    all_docs = []
    total_embeddings = 0
    start_time = time.time()

    for file_path in all_files:
        # Create DocMeta for file
        doc_meta = DocMeta(
            doc_id=file_hash(str(file_path))[:32],  # Use file hash as doc_id
            title=file_path.stem,
            collection=collection,
            source_uri=str(file_path),
            publisher=meta_overrides.get("publisher", "Unknown") if meta_overrides else "Unknown",
            content_hash=file_hash(str(file_path)),
            doc_hash=file_hash(str(file_path)),
        )

        # Apply overrides
        if meta_overrides:
            for key, value in meta_overrides.items():
                if hasattr(doc_meta, key):
                    setattr(doc_meta, key, value)

        # Ingest file
        manifest = await ingest_path(
            path=file_path,
            collection=collection,
            doc_meta=doc_meta,
            config=config,
        )

        all_docs.append(doc_meta)
        total_embeddings += manifest.total_embeddings

    duration = time.time() - start_time

    # Create combined manifest
    combined_manifest = IngestionManifest(
        collection_id=collection,
        documents=all_docs,
        ingestion_duration_seconds=duration,
        pipeline_config={
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunking_strategy": config.chunking_strategy,
            "embedding_model": config.embedding_model,
            "embedding_dimension": config.embedding_dimension,
        },
        vector_store_type=config.vector_store_provider,
        vector_store_config={
            "provider": config.vector_store_provider,
            "path": config.vector_store_path,
        },
        total_embeddings=total_embeddings,
    )

    return combined_manifest


def write_manifest(
    manifest: IngestionManifest,
    manifest_path: Path,
) -> None:
    """
    Write MANIFEST.json for ingested documents.

    Args:
        manifest: Ingestion manifest
        manifest_path: Path to write manifest file
    """
    manifest_dict = {
        "collection_id": manifest.collection_id,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content_hash": doc.content_hash,
                "doc_hash": doc.doc_hash,
            }
            for doc in manifest.documents
        ],
        "ingestion_duration_seconds": manifest.ingestion_duration_seconds,
        "pipeline_config": manifest.pipeline_config,
        "vector_store_type": manifest.vector_store_type,
        "vector_store_config": manifest.vector_store_config,
        "total_embeddings": manifest.total_embeddings,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest_dict, f, indent=2)


def _extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract text from PDF using PyMuPDF with deterministic parsing.

    Args:
        file_path: Path to PDF document

    Returns:
        Extracted text with normalized whitespace

    Raises:
        Exception: If PDF extraction fails
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF extraction. "
            "Install it with: pip install PyMuPDF"
        )

    doc = fitz.open(file_path)
    full_text = ""

    try:
        for page_num, page in enumerate(doc):
            # Extract text with consistent encoding
            page_text = page.get_text()

            # Add page break marker for consistent chunking
            if page_num > 0:
                full_text += "\n\n"  # Consistent page separator

            full_text += page_text
    finally:
        doc.close()

    # Normalize whitespace for deterministic parsing
    # Replace multiple spaces with single space
    full_text = re.sub(r'[ \t]+', ' ', full_text)
    # Replace multiple newlines with max 2 (paragraph breaks)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    # Strip trailing whitespace from each line
    full_text = '\n'.join(line.rstrip() for line in full_text.splitlines())

    return full_text


def _extract_text(file_path: Path) -> str:
    """
    Extract text from document with proper format detection.

    Supports:
    - PDF extraction (PyMuPDF)
    - Text files (UTF-8, with fallback to latin-1)

    Future enhancements:
    - Word extraction (python-docx)
    - Table extraction (Camelot, Tabula)
    - Formula extraction (latex parsing)

    Args:
        file_path: Path to document

    Returns:
        Extracted text
    """
    file_suffix = file_path.suffix.lower()

    # Handle PDF files
    if file_suffix == '.pdf':
        try:
            return _extract_text_from_pdf(file_path)
        except ImportError as e:
            logger.warning(f"PyMuPDF not available: {e}. Falling back to text reading.")
        except Exception as e:
            logger.warning(
                f"PDF extraction failed for {file_path}: {e}. "
                "Falling back to text reading."
            )

    # Handle text files (or fallback for PDF)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Try latin-1 fallback
        logger.warning(f"UTF-8 decode failed for {file_path}, trying latin-1")
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def _chunk_document(
    text: str,
    doc_meta: DocMeta,
    collection: str,
    chunker: Any,
    config: RAGConfig,
) -> List[Chunk]:
    """
    Chunk document into semantic segments.

    Args:
        text: Document text
        doc_meta: Document metadata
        collection: Collection name
        chunker: Chunker instance
        config: RAG configuration

    Returns:
        List of Chunk objects
    """
    # Canonicalize text for hashing
    canonical_text = canonicalize_text(text)

    # Use chunker to split document
    chunks = []
    chunk_size = config.chunk_size * 4  # Approximate chars (4 chars/token)
    overlap = config.chunk_overlap * 4

    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i : i + chunk_size]
        if not chunk_text.strip():
            continue

        # Create chunk ID
        section_path = f"Section_{i // chunk_size}"
        chunk_id = chunk_uuid5(
            doc_id=doc_meta.doc_id,
            section_path=section_path,
            start_offset=i,
        )

        # Compute section hash
        sec_hash = section_hash(chunk_text, section_path)

        # Approximate token count (will be accurate with TokenAwareChunker)
        token_count = len(chunk_text.split())

        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=doc_meta.doc_id,
            section_path=section_path,
            section_hash=sec_hash,
            start_char=i,
            end_char=i + len(chunk_text),
            text=chunk_text,
            token_count=token_count,
            embedding_model=config.embedding_model,
            extra={"collection": collection},
        )

        chunks.append(chunk)

    return chunks


async def _generate_embeddings(
    texts: List[str],
    embedder: Any,
    config: RAGConfig,
) -> List[List[float]]:
    """
    Generate embeddings for texts.

    Args:
        texts: List of text strings
        embedder: Embedder instance
        config: RAG configuration

    Returns:
        List of embedding vectors
    """
    # Batch processing for efficiency
    embeddings = []
    batch_size = config.embedding_batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # Call embedder.embed_batch(batch)
        # For now, placeholder
        batch_embeddings = [
            [0.0] * config.embedding_dimension for _ in batch
        ]
        embeddings.extend(batch_embeddings)

    return embeddings


async def _store_chunks(
    chunks: List[Chunk],
    embeddings: List[List[float]],
    collection: str,
    vector_store: Any,
) -> None:
    """
    Store chunks and embeddings in vector store.

    Args:
        chunks: List of chunks
        embeddings: List of embeddings
        collection: Collection name
        vector_store: Vector store instance
    """
    # Create Document objects
    import numpy as np

    documents = [
        Document(chunk=chunk, embedding=np.array(emb, dtype=np.float32))
        for chunk, emb in zip(chunks, embeddings)
    ]

    # Add to vector store
    vector_store.add_documents(documents, collection)
