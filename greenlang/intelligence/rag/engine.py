"""
Main RAG engine orchestrating document ingestion and retrieval.

INTL-104: RAG v1 - Deterministic, policy-safe retrieval with:
- Document ingestion pipeline (chunk → embed → index)
- Two-stage retrieval (fetch_k → MMR → top_k)
- Audit-ready citations with provenance
- Collection allowlist enforcement
- Sanitization for prompt injection defense

Architecture:
    User Query → Sanitize → Embed → Retrieve → MMR → Citations → Sanitize → Return

Security:
- Collection allowlist enforcement (config.py)
- Input sanitization (sanitize.py)
- Network isolation in replay mode (determinism.py)
"""

import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
import hashlib

from greenlang.intelligence.rag.models import (
    DocMeta,
    Chunk,
    RAGCitation,
    QueryResult,
    IngestionManifest,
)
from greenlang.intelligence.rag.config import (
    RAGConfig,
    get_config,
    enforce_allowlist,
    is_collection_allowed,
)
from greenlang.intelligence.rag.hashing import (
    file_hash,
    doc_hash,
    section_hash,
    chunk_uuid5,
    canonicalize_text,
    query_hash,
)
from greenlang.intelligence.rag.sanitize import (
    sanitize_rag_input,
    sanitize_for_prompt,
    validate_collection_name,
    detect_suspicious_content,
)
from greenlang.intelligence.rag.determinism import DeterministicRAG


class RAGEngine:
    """
    Main RAG engine for document ingestion and retrieval.

    Orchestrates:
    - Embedder: Text → vector embeddings
    - Vector store: Embedding storage and similarity search
    - Retriever: MMR-based diversified retrieval
    - Chunker: Document → semantic chunks
    - Determinism: Replay mode with caching

    Example:
        >>> config = RAGConfig(mode="live")
        >>> engine = RAGEngine(config)
        >>> manifest = await engine.ingest_document(
        ...     file_path=Path("ghg_protocol.pdf"),
        ...     collection="ghg_protocol_corp",
        ...     doc_meta=DocMeta(...)
        ... )
        >>> result = await engine.query("emission factors", top_k=6)
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG engine.

        Args:
            config: RAG configuration (defaults to global config)
        """
        self.config = config or get_config()

        # Initialize components (will be imported once available)
        self.embedder = None
        self.vector_store = None
        self.retriever = None
        self.chunker = None

        # Initialize deterministic wrapper if in replay mode
        self.deterministic_wrapper = None
        if self.config.mode == "replay":
            self.deterministic_wrapper = DeterministicRAG(
                mode="replay",
                config=self.config,
            )

        # Store document metadata in memory (will move to persistent store later)
        self._doc_metadata: Dict[str, DocMeta] = {}

    def _initialize_components(self):
        """
        Initialize RAG components (lazy loading).

        This method imports and initializes:
        - Embedder (MiniLM, OpenAI, or Anthropic)
        - Vector store (FAISS, ChromaDB, or Weaviate)
        - Retriever (MMR-based)
        - Chunker (token-aware, character, or sentence)
        """
        if self.embedder is not None:
            return  # Already initialized

        # Import components
        try:
            from greenlang.intelligence.rag.embeddings import get_embedding_provider
            from greenlang.intelligence.rag.vector_stores import get_vector_store
            from greenlang.intelligence.rag.retrievers import get_retriever

            # Initialize embedder (factory function only takes config)
            self.embedder = get_embedding_provider(config=self.config)

            # Initialize vector store (factory function takes dimension and config)
            self.vector_store = get_vector_store(
                dimension=self.config.embedding_dimension,
                config=self.config,
            )

            # Initialize retriever (if using MMR retrieval method)
            if self.config.retrieval_method == "mmr":
                self.retriever = get_retriever(
                    vector_store=self.vector_store,
                    retrieval_method=self.config.retrieval_method,
                    fetch_k=self.config.default_fetch_k,
                    top_k=self.config.default_top_k,
                    lambda_mult=self.config.mmr_lambda,
                )
            else:
                # Simple similarity retrieval
                self.retriever = get_retriever(
                    vector_store=self.vector_store,
                    retrieval_method=self.config.retrieval_method,
                    top_k=self.config.default_top_k,
                )

            # Chunker initialization - skip for now, using simple chunking in _chunk_document
            # Will be enhanced later when chunker module is ready
            self.chunker = None

        except ImportError as e:
            # Components not yet available - provide helpful error
            raise ImportError(
                f"RAG components not yet available: {e}\n"
                "The core components (embedders, vector_stores, retrievers) "
                "may need additional dependencies installed."
            )

    async def ingest_document(
        self,
        file_path: Path,
        collection: str,
        doc_meta: DocMeta,
        extract_tables: bool = False,
        extract_formulas: bool = False,
    ) -> IngestionManifest:
        """
        Ingest a document into the RAG system.

        Pipeline:
        1. Validate collection allowlist
        2. Verify file hash matches doc_meta.content_hash
        3. Extract text (PDF, Word, etc.)
        4. Chunk document (TokenAwareChunker)
        5. Generate embeddings (batch processing)
        6. Store in vector store with metadata
        7. Create MANIFEST.json for audit trail

        Args:
            file_path: Path to document file
            collection: Collection name (must be in allowlist)
            doc_meta: Document metadata
            extract_tables: Extract tables (future enhancement)
            extract_formulas: Extract formulas (future enhancement)

        Returns:
            IngestionManifest with audit trail

        Raises:
            ValueError: If collection not in allowlist
            FileNotFoundError: If file not found
            RuntimeError: If file hash doesn't match doc_meta.content_hash
        """
        start_time = time.time()

        # Step 1: Validate collection allowlist
        if not validate_collection_name(collection):
            raise ValueError(f"Invalid collection name: {collection}")

        if not is_collection_allowed(collection, self.config):
            raise ValueError(
                f"Collection '{collection}' is not in allowlist. "
                f"Allowed: {', '.join(self.config.allowlist)}"
            )

        # Step 2: Verify file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 3: Verify file hash
        if self.config.verify_checksums:
            computed_hash = file_hash(str(file_path))
            if computed_hash != doc_meta.content_hash:
                raise RuntimeError(
                    f"File hash mismatch for {file_path}\n"
                    f"Expected: {doc_meta.content_hash}\n"
                    f"Computed: {computed_hash}\n"
                    "File may have been tampered with or metadata is incorrect."
                )

        # Step 4: Initialize components
        self._initialize_components()

        # Step 5: Extract text (placeholder - will be enhanced with PDF/Word support)
        # For now, assume text file
        text = self._extract_text(file_path)

        # Check for suspicious content
        if self.config.enable_sanitization:
            warning = detect_suspicious_content(text)
            if warning:
                print(f"[SECURITY WARNING] {warning}")

        # Step 6: Chunk document
        chunks = self._chunk_document(
            text=text,
            doc_meta=doc_meta,
            collection=collection,
        )

        # Step 7: Generate embeddings (batch processing)
        embeddings = await self._generate_embeddings([chunk.text for chunk in chunks])

        # Step 8: Store in vector store
        await self._store_chunks(
            chunks=chunks,
            embeddings=embeddings,
            collection=collection,
        )

        # Step 9: Store document metadata
        self._doc_metadata[doc_meta.doc_id] = doc_meta

        # Step 10: Create manifest
        duration = time.time() - start_time
        manifest = IngestionManifest(
            collection_id=collection,
            documents=[doc_meta],
            ingestion_duration_seconds=duration,
            pipeline_config={
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "chunking_strategy": self.config.chunking_strategy,
                "embedding_model": self.config.embedding_model,
                "embedding_dimension": self.config.embedding_dimension,
            },
            vector_store_type=self.config.vector_store_provider,
            vector_store_config={
                "provider": self.config.vector_store_provider,
                "path": self.config.vector_store_path,
            },
            total_embeddings=len(embeddings),
        )

        return manifest

    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text from document.

        Future enhancements:
        - PDF extraction (PyMuPDF, pdfplumber)
        - Word extraction (python-docx)
        - Table extraction (Camelot, Tabula)
        - Formula extraction (latex parsing)

        Args:
            file_path: Path to document

        Returns:
            Extracted text
        """
        # For now, assume UTF-8 text file
        # Will be enhanced with PDF/Word support by other agent
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            # Try latin-1 fallback
            with open(file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _chunk_document(
        self,
        text: str,
        doc_meta: DocMeta,
        collection: str,
    ) -> List[Chunk]:
        """
        Chunk document into semantic segments.

        Args:
            text: Document text
            doc_meta: Document metadata
            collection: Collection name

        Returns:
            List of Chunk objects
        """
        # Canonicalize text for hashing
        canonical_text = canonicalize_text(text)

        # Use chunker to split document
        # For now, simple chunking by character count
        # Will be replaced with TokenAwareChunker by other agent
        chunks = []
        chunk_size = self.config.chunk_size * 4  # Approximate chars (4 chars/token)
        overlap = self.config.chunk_overlap * 4

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
                embedding_model=self.config.embedding_model,
            )

            chunks.append(chunk)

        return chunks

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        # Use embedder to generate embeddings (async)
        if not texts:
            return []

        embeddings_np = await self.embedder.embed(texts)

        # Convert numpy arrays to lists for serialization
        embeddings = [emb.tolist() for emb in embeddings_np]

        return embeddings

    async def _store_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection: str,
    ) -> None:
        """
        Store chunks and embeddings in vector store.

        Args:
            chunks: List of chunks
            embeddings: List of embeddings
            collection: Collection name
        """
        import numpy as np
        from greenlang.intelligence.rag.vector_stores import Document

        # Create Document objects for vector store
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            # Add collection to chunk metadata
            chunk.extra["collection"] = collection

            # Create Document wrapper
            doc = Document(
                chunk=chunk,
                embedding=np.array(embedding, dtype=np.float32),
            )
            documents.append(doc)

        # Add documents to vector store
        self.vector_store.add_documents(documents, collection=collection)

    async def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        collections: Optional[List[str]] = None,
        fetch_k: Optional[int] = None,
        mmr_lambda: Optional[float] = None,
    ) -> QueryResult:
        """
        Query the RAG system.

        Pipeline:
        1. Sanitize input query
        2. Enforce collection allowlist
        3. Check deterministic wrapper (replay mode)
        4. Embed query
        5. Fetch candidates from vector store (fetch_k results)
        6. Apply MMR for diversity (re-rank to top_k)
        7. Generate citations with provenance
        8. Sanitize retrieved text
        9. Return QueryResult

        Args:
            query: Query string
            top_k: Number of results to return (default: config.default_top_k)
            collections: Collections to search (default: all allowed)
            fetch_k: Number of candidates for MMR (default: config.default_fetch_k)
            mmr_lambda: MMR lambda parameter (default: config.mmr_lambda)

        Returns:
            QueryResult with chunks, citations, and metadata

        Raises:
            ValueError: If collection not in allowlist
        """
        start_time = time.time()

        # Set defaults
        top_k = top_k or self.config.default_top_k
        fetch_k = fetch_k or self.config.default_fetch_k
        mmr_lambda = mmr_lambda if mmr_lambda is not None else self.config.mmr_lambda
        collections = collections or self.config.allowlist

        # Step 1: Sanitize input query
        if self.config.enable_sanitization:
            original_query = query
            query = sanitize_for_prompt(query, max_length=512)

            # Check for suspicious content
            warning = detect_suspicious_content(query)
            if warning:
                print(f"[SECURITY WARNING] Query sanitization: {warning}")

        # Step 2: Enforce collection allowlist
        enforce_allowlist(collections, self.config)

        # Step 3: Check deterministic wrapper (replay mode)
        if self.deterministic_wrapper:
            return self.deterministic_wrapper.search(
                query=query,
                k=top_k,
                collections=collections,
                fetch_k=fetch_k,
                mmr_lambda=mmr_lambda,
                engine=self,
            )

        # Step 4: Perform actual search (live/record mode)
        return await self._real_search(
            query=query,
            top_k=top_k,
            collections=collections,
            fetch_k=fetch_k,
            mmr_lambda=mmr_lambda,
        )

    async def _real_search(
        self,
        query: str,
        top_k: int,
        collections: List[str],
        fetch_k: int,
        mmr_lambda: float,
    ) -> QueryResult:
        """
        Perform actual search (without caching).

        This method is called by both live/record mode and deterministic wrapper.

        Args:
            query: Query string (already sanitized)
            top_k: Number of results to return
            collections: Collections to search
            fetch_k: Number of candidates for MMR
            mmr_lambda: MMR lambda parameter

        Returns:
            QueryResult
        """
        start_time = time.time()

        # Initialize components
        self._initialize_components()

        # Step 1: Embed query
        query_embedding = await self._embed_query(query)

        # Step 2: Fetch candidates from vector store (fetch_k results)
        # Returns List[Document] with chunks and embeddings
        candidate_documents = await self._fetch_candidates(
            query_embedding=query_embedding,
            collections=collections,
            k=fetch_k,
        )

        # Step 3: Apply MMR for diversity
        if self.config.retrieval_method == "mmr" and len(candidate_documents) > top_k:
            selected_chunks, scores = await self._apply_mmr(
                query_embedding=query_embedding,
                candidates=candidate_documents,
                k=top_k,
                lambda_mult=mmr_lambda,
            )
        else:
            # Use top candidates without MMR - extract chunks from documents
            selected_chunks = [doc.chunk for doc in candidate_documents[:top_k]]
            scores = [1.0] * len(selected_chunks)  # Placeholder scores

        # Step 4: Generate citations
        citations = self._generate_citations(selected_chunks, scores)

        # Step 5: Sanitize retrieved text
        if self.config.enable_sanitization:
            for chunk in selected_chunks:
                chunk.text = sanitize_rag_input(
                    chunk.text,
                    strict=self.config.strict_sanitization,
                )

        # Step 6: Compute metadata
        total_tokens = sum(chunk.token_count for chunk in selected_chunks)
        search_time_ms = int((time.time() - start_time) * 1000)

        # Compute query hash
        qhash = query_hash(
            query,
            {
                "k": top_k,
                "collections": sorted(collections),
                "fetch_k": fetch_k,
                "mmr_lambda": mmr_lambda,
            },
        )

        # Create QueryResult
        result = QueryResult(
            query=query,
            query_hash=qhash,
            chunks=selected_chunks,
            citations=citations,
            relevance_scores=scores,
            retrieval_method=self.config.retrieval_method,
            search_time_ms=search_time_ms,
            total_tokens=total_tokens,
            total_chunks=len(selected_chunks),
            collections_searched=collections,
            extra={
                "fetch_k": fetch_k,
                "mmr_lambda": mmr_lambda,
            },
        )

        return result

    async def _embed_query(self, query: str) -> List[float]:
        """
        Embed query string.

        Args:
            query: Query string

        Returns:
            Embedding vector
        """
        # Use embedder to embed query
        embeddings_np = await self.embedder.embed([query])

        # Return first (and only) embedding as list
        return embeddings_np[0].tolist()

    async def _fetch_candidates(
        self,
        query_embedding: List[float],
        collections: List[str],
        k: int,
    ):
        """
        Fetch candidate documents from vector store.

        Args:
            query_embedding: Query embedding vector
            collections: Collections to search
            k: Number of candidates to fetch

        Returns:
            List of candidate Document objects (with chunks and embeddings)
        """
        import numpy as np

        # Convert query embedding to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Search vector store (returns Documents with embeddings)
        documents = self.vector_store.similarity_search(
            query_embedding=query_vec,
            k=k,
            collections=collections,
        )

        return documents

    async def _apply_mmr(
        self,
        query_embedding: List[float],
        candidates,  # List[Document] with chunks and embeddings
        k: int,
        lambda_mult: float,
    ) -> tuple:
        """
        Apply MMR for diversity.

        Args:
            query_embedding: Query embedding
            candidates: Candidate Document objects (with embeddings)
            k: Number of results to select
            lambda_mult: MMR lambda (0=diversity, 1=relevance)

        Returns:
            Tuple of (selected_chunks, scores)
        """
        import numpy as np
        from greenlang.intelligence.rag.retrievers import mmr_retrieval

        # Convert query embedding to numpy array
        query_vec = np.array(query_embedding, dtype=np.float32)

        # Apply MMR retrieval
        results = mmr_retrieval(
            query_embedding=query_vec,
            candidates=candidates,  # Documents with embeddings
            lambda_mult=lambda_mult,
            k=k,
        )

        # Extract chunks and scores from results
        selected_chunks = [doc.chunk for doc, score in results]
        scores = [score for doc, score in results]

        return selected_chunks, scores

    def _generate_citations(
        self,
        chunks: List[Chunk],
        scores: List[float],
    ) -> List[RAGCitation]:
        """
        Generate citations for chunks.

        Args:
            chunks: Retrieved chunks
            scores: Relevance scores

        Returns:
            List of RAGCitation objects
        """
        citations = []

        for chunk, score in zip(chunks, scores):
            # Get document metadata
            doc_meta = self._doc_metadata.get(chunk.doc_id)

            if doc_meta:
                citation = RAGCitation.from_chunk(
                    chunk=chunk,
                    doc_meta=doc_meta,
                    relevance_score=score,
                )
            else:
                # Fallback: create minimal citation
                citation = RAGCitation(
                    doc_title=f"Document {chunk.doc_id}",
                    section_path=chunk.section_path,
                    section_hash=chunk.section_hash,
                    checksum="unknown",
                    formatted=f"{chunk.section_path} (Document {chunk.doc_id})",
                    relevance_score=score,
                )

            citations.append(citation)

        return citations

    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.

        Args:
            collection: Collection name

        Returns:
            Dict with stats (num_documents, num_chunks, total_tokens, etc.)
        """
        # Placeholder - will query vector store
        return {
            "collection": collection,
            "num_documents": 0,
            "num_chunks": 0,
            "total_tokens": 0,
            "embedding_dimension": self.config.embedding_dimension,
        }

    def list_collections(self) -> List[str]:
        """
        List all available collections.

        Returns:
            List of collection names
        """
        # Placeholder - will query vector store
        return self.config.allowlist

    async def delete_collection(self, collection: str) -> None:
        """
        Delete a collection.

        Args:
            collection: Collection name

        Raises:
            ValueError: If collection not found
        """
        if not is_collection_allowed(collection, self.config):
            raise ValueError(f"Collection '{collection}' is not in allowlist")

        # Will call self.vector_store.delete_collection(collection)
        # Placeholder
        pass
