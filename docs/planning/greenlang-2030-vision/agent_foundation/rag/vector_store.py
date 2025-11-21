# -*- coding: utf-8 -*-
"""
Vector Store Abstraction for GreenLang RAG System
Unified interface for multiple vector databases with hybrid search
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from greenlang.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with embeddings to the store"""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs"""
        pass

    @abstractmethod
    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update existing documents"""
        pass

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        alpha: float = 0.5
    ) -> Tuple[List[Any], List[float]]:
        """
        Hybrid search combining vector and keyword search
        alpha: weight for vector search (0 = keyword only, 1 = vector only)
        """
        # Default implementation uses only vector search
        return self.similarity_search(query_embedding, top_k, filters)


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store for efficient similarity search"""

    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "IndexFlatL2",
        index_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Initialize index
        if index_path and Path(index_path).exists():
            self.index = self.faiss.read_index(index_path)
            self.documents = self._load_metadata()
        else:
            self.index = self._create_index()
            self.documents = []

        # Document metadata storage
        self.id_to_idx = {}
        self.idx_to_id = {}

    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == "IndexFlatL2":
            return self.faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IndexFlatIP":
            return self.faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "IndexIVFFlat":
            quantizer = self.faiss.IndexFlatL2(self.dimension)
            index = self.faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            return index
        elif self.index_type == "IndexHNSWFlat":
            index = self.faiss.IndexHNSWFlat(self.dimension, 32)
            return index
        else:
            return self.faiss.IndexFlatL2(self.dimension)

    def _load_metadata(self) -> List[Any]:
        """Load document metadata from file"""
        if self.metadata_path and Path(self.metadata_path).exists():
            with open(self.metadata_path, 'r') as f:
                return json.load(f)
        return []

    def _save_metadata(self):
        """Save document metadata to file"""
        if self.metadata_path:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.documents, f, default=str)

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents with embeddings to FAISS index"""

        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")

        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(deterministic_uuid(__name__, str(DeterministicClock.now()))) for _ in range(len(documents))]

        # Normalize embeddings if using inner product
        if self.index_type == "IndexFlatIP":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Add to index
        start_idx = len(self.documents)
        self.index.add(embeddings.astype('float32'))

        # Store documents and mappings
        for i, (doc, doc_id) in enumerate(zip(documents, ids)):
            idx = start_idx + i
            self.documents.append(doc)
            self.id_to_idx[doc_id] = idx
            self.idx_to_id[idx] = doc_id

        # Save if paths are specified
        if self.index_path:
            self.faiss.write_index(self.index, self.index_path)
        if self.metadata_path:
            self._save_metadata()

        logger.info(f"Added {len(documents)} documents to FAISS index")
        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents using FAISS"""

        if self.index.ntotal == 0:
            return [], []

        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Normalize if using inner product
        if self.index_type == "IndexFlatIP":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        # Filter results
        documents = []
        scores = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]

            # Apply filters if provided
            if filters:
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                elif isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                else:
                    metadata = {}

                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue

            documents.append(doc)
            # Convert distance to similarity score (0-1 range)
            if self.index_type == "IndexFlatL2":
                score = 1.0 / (1.0 + dist)
            else:  # Inner product
                score = float(dist)
            scores.append(score)

        return documents, scores

    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs (Note: FAISS doesn't support deletion directly)"""
        logger.warning("FAISS doesn't support direct deletion. Consider rebuilding index.")
        # Mark documents as deleted in metadata
        for doc_id in ids:
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]
                if idx < len(self.documents):
                    self.documents[idx] = None
        return True

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update existing documents"""
        for i, doc_id in enumerate(ids):
            if doc_id in self.id_to_idx:
                idx = self.id_to_idx[doc_id]

                if documents and i < len(documents):
                    self.documents[idx] = documents[i]

                if embeddings is not None and i < embeddings.shape[0]:
                    # FAISS doesn't support direct update, need to reconstruct
                    logger.warning("FAISS embedding update requires index reconstruction")

        if self.metadata_path:
            self._save_metadata()

        return True

    def save(self):
        """Save index and metadata to disk"""
        if self.index_path:
            self.faiss.write_index(self.index, self.index_path)
        if self.metadata_path:
            self._save_metadata()


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store implementation"""

    def __init__(
        self,
        collection_name: str = "greenlang_rag",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None
    ):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")

        self.collection_name = collection_name

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to ChromaDB"""

        if ids is None:
            import uuid
            ids = [str(deterministic_uuid(__name__, str(DeterministicClock.now()))) for _ in range(len(documents))]

        # Prepare documents for ChromaDB
        texts = []
        metadatas = []

        for doc in documents:
            if hasattr(doc, 'content'):
                texts.append(doc.content)
                metadatas.append(doc.metadata if hasattr(doc, 'metadata') else {})
            elif isinstance(doc, dict):
                texts.append(doc.get('content', doc.get('text', '')))
                metadatas.append(doc.get('metadata', {}))
            else:
                texts.append(str(doc))
                metadatas.append({})

        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents to ChromaDB")
        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents in ChromaDB"""

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filters
        )

        documents = []
        scores = []

        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                # Reconstruct document object
                from rag_system import Document
                doc = Document(
                    content=doc_text,
                    metadata=results['metadatas'][0][i] if results['metadatas'] else {},
                    doc_id=results['ids'][0][i] if results['ids'] else None
                )
                documents.append(doc)

                # Convert distance to similarity score
                if results['distances'] and results['distances'][0]:
                    distance = results['distances'][0][i]
                    score = 1.0 / (1.0 + distance)
                else:
                    score = 1.0
                scores.append(score)

        return documents, scores

    def delete(self, ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update documents in ChromaDB"""
        try:
            update_params = {"ids": ids}

            if documents:
                texts = []
                for doc in documents:
                    if hasattr(doc, 'content'):
                        texts.append(doc.content)
                    elif isinstance(doc, dict):
                        texts.append(doc.get('content', doc.get('text', '')))
                    else:
                        texts.append(str(doc))
                update_params["documents"] = texts

            if embeddings is not None:
                update_params["embeddings"] = embeddings.tolist()

            if metadata:
                update_params["metadatas"] = metadata

            self.collection.update(**update_params)
            logger.info(f"Updated {len(ids)} documents in ChromaDB")
            return True
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation"""

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str = "greenlang-rag",
        dimension: int = 384,
        metric: str = "cosine",
        namespace: Optional[str] = None
    ):
        try:
            import pinecone
        except ImportError:
            raise ImportError("Pinecone not installed. Install with: pip install pinecone-client")

        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)

        # Create or get index
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric
            )

        self.index = pinecone.Index(index_name)
        self.namespace = namespace
        self.dimension = dimension

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to Pinecone"""

        if ids is None:
            import uuid
            ids = [str(deterministic_uuid(__name__, str(DeterministicClock.now()))) for _ in range(len(documents))]

        # Prepare vectors for Pinecone
        vectors = []
        for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
            metadata = {}

            if hasattr(doc, 'content'):
                metadata['content'] = doc.content[:1000]  # Pinecone metadata size limit
                if hasattr(doc, 'metadata'):
                    metadata.update(doc.metadata)
            elif isinstance(doc, dict):
                metadata = doc.get('metadata', {})
                metadata['content'] = doc.get('content', doc.get('text', ''))[:1000]
            else:
                metadata['content'] = str(doc)[:1000]

            vectors.append({
                'id': doc_id,
                'values': embedding.tolist(),
                'metadata': metadata
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)

        logger.info(f"Added {len(documents)} documents to Pinecone")
        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents in Pinecone"""

        # Query index
        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            filter=filters,
            namespace=self.namespace,
            include_metadata=True
        )

        documents = []
        scores = []

        for match in results['matches']:
            # Reconstruct document
            from rag_system import Document
            metadata = match.get('metadata', {})
            content = metadata.pop('content', '')

            doc = Document(
                content=content,
                metadata=metadata,
                doc_id=match['id']
            )
            documents.append(doc)
            scores.append(match['score'])

        return documents, scores

    def delete(self, ids: List[str]) -> bool:
        """Delete documents from Pinecone"""
        try:
            self.index.delete(ids=ids, namespace=self.namespace)
            logger.info(f"Deleted {len(ids)} documents from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update documents in Pinecone (via upsert)"""
        if not documents and embeddings is None and not metadata:
            return True

        # Fetch existing vectors
        existing = self.index.fetch(ids=ids, namespace=self.namespace)

        vectors = []
        for i, doc_id in enumerate(ids):
            if doc_id in existing['vectors']:
                vector_data = {'id': doc_id}

                # Use new embedding or keep existing
                if embeddings is not None and i < embeddings.shape[0]:
                    vector_data['values'] = embeddings[i].tolist()
                else:
                    vector_data['values'] = existing['vectors'][doc_id]['values']

                # Update metadata
                new_metadata = existing['vectors'][doc_id].get('metadata', {})
                if metadata and i < len(metadata):
                    new_metadata.update(metadata[i])
                if documents and i < len(documents):
                    doc = documents[i]
                    if hasattr(doc, 'content'):
                        new_metadata['content'] = doc.content[:1000]
                    elif isinstance(doc, dict):
                        new_metadata['content'] = doc.get('content', doc.get('text', ''))[:1000]

                vector_data['metadata'] = new_metadata
                vectors.append(vector_data)

        if vectors:
            self.index.upsert(vectors=vectors, namespace=self.namespace)
            logger.info(f"Updated {len(vectors)} documents in Pinecone")

        return True


class HybridVectorStore(VectorStore):
    """
    Hybrid vector store combining multiple backends
    Supports both vector and keyword search with result fusion
    """

    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: Optional[Any] = None,
        fusion_algorithm: str = "rrf"  # Reciprocal Rank Fusion
    ):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.fusion_algorithm = fusion_algorithm

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to both stores"""
        ids = self.vector_store.add_documents(documents, embeddings, ids)

        if self.keyword_store:
            # Add to keyword store (e.g., Elasticsearch, BM25)
            self._add_to_keyword_store(documents, ids)

        return ids

    def _add_to_keyword_store(self, documents: List[Any], ids: List[str]):
        """Add documents to keyword store (placeholder)"""
        # This would integrate with Elasticsearch, Whoosh, or BM25
        pass

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Vector-only search"""
        return self.vector_store.similarity_search(query_embedding, top_k, filters)

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        alpha: float = 0.5
    ) -> Tuple[List[Any], List[float]]:
        """
        Hybrid search combining vector and keyword results
        alpha: weight for vector search (0 = keyword only, 1 = vector only)
        """
        # Get vector search results
        vector_docs, vector_scores = self.vector_store.similarity_search(
            query_embedding, top_k * 2, filters
        )

        if not self.keyword_store:
            return vector_docs[:top_k], vector_scores[:top_k]

        # Get keyword search results
        keyword_docs, keyword_scores = self._keyword_search(
            query_text, top_k * 2, filters
        )

        # Fuse results
        if self.fusion_algorithm == "rrf":
            return self._reciprocal_rank_fusion(
                vector_docs, vector_scores,
                keyword_docs, keyword_scores,
                top_k, alpha
            )
        else:
            return self._weighted_fusion(
                vector_docs, vector_scores,
                keyword_docs, keyword_scores,
                top_k, alpha
            )

    def _keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Keyword search (placeholder)"""
        # This would integrate with actual keyword search
        return [], []

    def _reciprocal_rank_fusion(
        self,
        docs1: List[Any], scores1: List[float],
        docs2: List[Any], scores2: List[float],
        top_k: int, alpha: float,
        k: int = 60
    ) -> Tuple[List[Any], List[float]]:
        """
        Reciprocal Rank Fusion for combining search results
        k: constant for RRF formula (typically 60)
        """
        # Create document scores map
        doc_scores = {}

        # Add vector search results
        for rank, (doc, score) in enumerate(zip(docs1, scores1)):
            doc_id = doc.doc_id if hasattr(doc, 'doc_id') else str(doc)
            rrf_score = alpha * (1.0 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # Add keyword search results
        for rank, (doc, score) in enumerate(zip(docs2, scores2)):
            doc_id = doc.doc_id if hasattr(doc, 'doc_id') else str(doc)
            rrf_score = (1 - alpha) * (1.0 / (k + rank + 1))
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Retrieve documents
        final_docs = []
        final_scores = []
        doc_map = {doc.doc_id if hasattr(doc, 'doc_id') else str(doc): doc
                   for doc in docs1 + docs2}

        for doc_id, score in sorted_docs[:top_k]:
            if doc_id in doc_map:
                final_docs.append(doc_map[doc_id])
                final_scores.append(score)

        return final_docs, final_scores

    def _weighted_fusion(
        self,
        docs1: List[Any], scores1: List[float],
        docs2: List[Any], scores2: List[float],
        top_k: int, alpha: float
    ) -> Tuple[List[Any], List[float]]:
        """Simple weighted fusion of results"""
        # Normalize scores
        if scores1:
            scores1 = np.array(scores1) / max(scores1)
        if scores2:
            scores2 = np.array(scores2) / max(scores2)

        # Combine with weights
        combined = []
        for doc, score in zip(docs1, scores1):
            combined.append((doc, alpha * score))
        for doc, score in zip(docs2, scores2):
            combined.append((doc, (1 - alpha) * score))

        # Sort and return top k
        combined.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in combined[:top_k]]
        final_scores = [score for _, score in combined[:top_k]]

        return final_docs, final_scores

    def delete(self, ids: List[str]) -> bool:
        """Delete from both stores"""
        success = self.vector_store.delete(ids)
        if self.keyword_store:
            # Delete from keyword store
            pass
        return success

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update in both stores"""
        success = self.vector_store.update(ids, documents, embeddings, metadata)
        if self.keyword_store and documents:
            # Update keyword store
            pass
        return success


class WeaviateVectorStore(VectorStore):
    """Weaviate vector store implementation"""

    def __init__(
        self,
        url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        class_name: str = "GreenLangDocument",
        vectorizer: str = "none",
        dimension: int = 768,
        distance_metric: str = "cosine"
    ):
        try:
            import weaviate
            from weaviate.auth import AuthApiKey
        except ImportError:
            raise ImportError("Weaviate not installed. Install with: pip install weaviate-client")

        # Initialize client
        auth_config = AuthApiKey(api_key=api_key) if api_key else None
        self.client = weaviate.Client(
            url=url,
            auth_client_secret=auth_config
        )

        self.class_name = class_name
        self.dimension = dimension

        # Create schema if it doesn't exist
        self._create_schema(vectorizer, distance_metric)

    def _create_schema(self, vectorizer: str, distance_metric: str):
        """Create Weaviate schema for documents"""
        schema = {
            "class": self.class_name,
            "vectorizer": vectorizer,
            "moduleConfig": {},
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Document content"
                },
                {
                    "name": "source",
                    "dataType": ["string"],
                    "description": "Document source"
                },
                {
                    "name": "metadata",
                    "dataType": ["text"],
                    "description": "JSON metadata"
                },
                {
                    "name": "doc_id",
                    "dataType": ["string"],
                    "description": "Document ID"
                },
                {
                    "name": "chunk_id",
                    "dataType": ["int"],
                    "description": "Chunk ID"
                },
                {
                    "name": "confidence_score",
                    "dataType": ["number"],
                    "description": "Confidence score"
                }
            ],
            "vectorIndexConfig": {
                "distance": distance_metric,
                "vectorCacheMaxObjects": 1000000
            }
        }

        # Check if class exists
        existing = self.client.schema.get()
        class_exists = any(c["class"] == self.class_name for c in existing.get("classes", []))

        if not class_exists:
            self.client.schema.create_class(schema)
            logger.info(f"Created Weaviate class: {self.class_name}")

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to Weaviate"""
        if ids is None:
            import uuid
            ids = [str(deterministic_uuid(__name__, str(DeterministicClock.now()))) for _ in range(len(documents))]

        batch = self.client.batch.configure(batch_size=100, timeout_retries=3)

        with batch as batch_client:
            for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
                properties = {}

                if hasattr(doc, 'content'):
                    properties['content'] = doc.content
                    properties['source'] = getattr(doc, 'source', '')
                    properties['metadata'] = json.dumps(getattr(doc, 'metadata', {}))
                    properties['doc_id'] = getattr(doc, 'doc_id', doc_id)
                    properties['chunk_id'] = getattr(doc, 'chunk_id', i)
                    properties['confidence_score'] = getattr(doc, 'confidence_score', 1.0)
                elif isinstance(doc, dict):
                    properties = {
                        'content': doc.get('content', doc.get('text', '')),
                        'source': doc.get('source', ''),
                        'metadata': json.dumps(doc.get('metadata', {})),
                        'doc_id': doc_id,
                        'chunk_id': i,
                        'confidence_score': doc.get('confidence_score', 1.0)
                    }
                else:
                    properties = {
                        'content': str(doc),
                        'doc_id': doc_id,
                        'chunk_id': i
                    }

                batch_client.add_data_object(
                    properties,
                    self.class_name,
                    uuid=doc_id,
                    vector=embedding.tolist()
                )

        logger.info(f"Added {len(documents)} documents to Weaviate")
        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents in Weaviate"""
        from document_processor import Document, DocumentMetadata

        # Build query
        near_vector = {"vector": query_embedding.tolist()}

        query = (
            self.client.query
            .get(self.class_name, ["content", "source", "metadata", "doc_id", "chunk_id", "confidence_score"])
            .with_near_vector(near_vector)
            .with_limit(top_k)
            .with_additional(["distance", "id"])
        )

        # Add filters if provided
        if filters:
            where_filter = self._build_where_filter(filters)
            query = query.with_where(where_filter)

        # Execute query
        result = query.do()

        documents = []
        scores = []

        if result and "data" in result:
            for item in result["data"]["Get"][self.class_name]:
                # Parse metadata
                metadata_dict = {}
                if item.get("metadata"):
                    try:
                        metadata_dict = json.loads(item["metadata"])
                    except:
                        pass

                # Create document
                metadata = DocumentMetadata(
                    source=item.get("source", ""),
                    doc_type="unknown",
                    custom_metadata=metadata_dict
                )

                doc = Document(
                    content=item.get("content", ""),
                    metadata=metadata,
                    doc_id=item.get("doc_id"),
                    chunk_id=item.get("chunk_id"),
                    confidence_score=item.get("confidence_score", 1.0)
                )

                documents.append(doc)

                # Convert distance to similarity score
                distance = item.get("_additional", {}).get("distance", 0)
                score = 1.0 / (1.0 + distance)
                scores.append(score)

        return documents, scores

    def _build_where_filter(self, filters: Dict) -> Dict:
        """Build Weaviate where filter from dictionary"""
        where_conditions = []

        for key, value in filters.items():
            where_conditions.append({
                "path": [key],
                "operator": "Equal",
                "valueString": str(value)
            })

        if len(where_conditions) == 1:
            return where_conditions[0]
        else:
            return {
                "operator": "And",
                "operands": where_conditions
            }

    def delete(self, ids: List[str]) -> bool:
        """Delete documents from Weaviate"""
        try:
            for doc_id in ids:
                self.client.data_object.delete(doc_id, class_name=self.class_name)
            logger.info(f"Deleted {len(ids)} documents from Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update documents in Weaviate"""
        try:
            for i, doc_id in enumerate(ids):
                properties = {}

                if documents and i < len(documents):
                    doc = documents[i]
                    if hasattr(doc, 'content'):
                        properties['content'] = doc.content
                    elif isinstance(doc, dict):
                        properties['content'] = doc.get('content', doc.get('text', ''))

                if metadata and i < len(metadata):
                    properties['metadata'] = json.dumps(metadata[i])

                if properties:
                    self.client.data_object.update(
                        properties,
                        class_name=self.class_name,
                        uuid=doc_id
                    )

                # Update vector if provided
                if embeddings is not None and i < embeddings.shape[0]:
                    self.client.data_object.replace(
                        properties,
                        class_name=self.class_name,
                        uuid=doc_id,
                        vector=embeddings[i].tolist()
                    )

            logger.info(f"Updated {len(ids)} documents in Weaviate")
            return True
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation"""

    def __init__(
        self,
        url: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "greenlang_rag",
        dimension: int = 768,
        distance: str = "Cosine",
        on_disk: bool = False
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct
        except ImportError:
            raise ImportError("Qdrant client not installed. Install with: pip install qdrant-client")

        self.dimension = dimension
        self.collection_name = collection_name

        # Import models
        self.Distance = Distance
        self.VectorParams = VectorParams
        self.PointStruct = PointStruct

        # Initialize client
        self.client = QdrantClient(
            url=url,
            port=port,
            api_key=api_key
        )

        # Create collection if it doesn't exist
        self._create_collection(distance, on_disk)

    def _create_collection(self, distance: str, on_disk: bool):
        """Create Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if self.collection_name not in collection_names:
            distance_map = {
                "Cosine": self.Distance.COSINE,
                "Euclidean": self.Distance.EUCLID,
                "Dot": self.Distance.DOT,
            }

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(distance, self.Distance.COSINE)
                ),
                on_disk_payload=on_disk
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to Qdrant"""
        if ids is None:
            import uuid
            ids = [str(deterministic_uuid(__name__, str(DeterministicClock.now()))) for _ in range(len(documents))]

        points = []

        for i, (doc, embedding, doc_id) in enumerate(zip(documents, embeddings, ids)):
            payload = {}

            if hasattr(doc, 'content'):
                payload['content'] = doc.content
                payload['source'] = getattr(doc, 'source', '')
                payload['metadata'] = getattr(doc, 'metadata', {})
                payload['doc_id'] = getattr(doc, 'doc_id', doc_id)
                payload['chunk_id'] = getattr(doc, 'chunk_id', i)
                payload['confidence_score'] = getattr(doc, 'confidence_score', 1.0)
            elif isinstance(doc, dict):
                payload = doc.copy()
                payload['doc_id'] = doc_id
                payload['chunk_id'] = i
            else:
                payload = {
                    'content': str(doc),
                    'doc_id': doc_id,
                    'chunk_id': i
                }

            # Create point
            point = self.PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        logger.info(f"Added {len(documents)} documents to Qdrant")
        return ids

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> Tuple[List[Any], List[float]]:
        """Search for similar documents in Qdrant"""
        from document_processor import Document, DocumentMetadata
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Build filter if provided
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )

        documents = []
        scores = []

        for result in results:
            payload = result.payload

            # Create document
            metadata_dict = payload.get('metadata', {})
            if isinstance(metadata_dict, dict):
                metadata = DocumentMetadata(
                    source=payload.get('source', ''),
                    doc_type="unknown",
                    custom_metadata=metadata_dict
                )
            else:
                metadata = DocumentMetadata(
                    source=payload.get('source', ''),
                    doc_type="unknown"
                )

            doc = Document(
                content=payload.get('content', ''),
                metadata=metadata,
                doc_id=payload.get('doc_id'),
                chunk_id=payload.get('chunk_id'),
                confidence_score=payload.get('confidence_score', 1.0)
            )

            documents.append(doc)
            scores.append(result.score)

        return documents, scores

    def delete(self, ids: List[str]) -> bool:
        """Delete documents from Qdrant"""
        try:
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids)
            )
            logger.info(f"Deleted {len(ids)} documents from Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False

    def update(
        self,
        ids: List[str],
        documents: Optional[List[Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        metadata: Optional[List[Dict]] = None
    ) -> bool:
        """Update documents in Qdrant"""
        try:
            points = []

            for i, doc_id in enumerate(ids):
                # Get existing point
                existing = self.client.retrieve(
                    collection_name=self.collection_name,
                    ids=[doc_id],
                    with_payload=True,
                    with_vectors=True
                )

                if not existing:
                    continue

                point_data = existing[0]
                payload = point_data.payload
                vector = point_data.vector

                # Update payload
                if documents and i < len(documents):
                    doc = documents[i]
                    if hasattr(doc, 'content'):
                        payload['content'] = doc.content
                    elif isinstance(doc, dict):
                        payload.update(doc)

                if metadata and i < len(metadata):
                    payload['metadata'] = metadata[i]

                # Update vector if provided
                if embeddings is not None and i < embeddings.shape[0]:
                    vector = embeddings[i].tolist()

                # Create updated point
                point = self.PointStruct(
                    id=doc_id,
                    vector=vector,
                    payload=payload
                )
                points.append(point)

            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            logger.info(f"Updated {len(points)} documents in Qdrant")
            return True
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False


def create_vector_store(
    store_type: str = "faiss",
    **kwargs
) -> VectorStore:
    """Factory function to create vector store instances"""

    if store_type == "faiss":
        return FAISSVectorStore(**kwargs)
    elif store_type == "chroma":
        return ChromaDBVectorStore(**kwargs)
    elif store_type == "pinecone":
        return PineconeVectorStore(**kwargs)
    elif store_type == "weaviate":
        return WeaviateVectorStore(**kwargs)
    elif store_type == "qdrant":
        return QdrantVectorStore(**kwargs)
    elif store_type == "hybrid":
        # Create hybrid store with FAISS as default vector store
        vector_store = FAISSVectorStore(
            dimension=kwargs.get('dimension', 384)
        )
        return HybridVectorStore(vector_store, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")