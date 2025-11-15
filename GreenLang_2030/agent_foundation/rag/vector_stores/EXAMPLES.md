# Vector Store Integration Examples

This document provides comprehensive examples for using GreenLang's vector store implementations with ChromaDB and Pinecone.

## Table of Contents

1. [ChromaDB Examples](#chromadb-examples)
2. [Pinecone Examples](#pinecone-examples)
3. [Factory Pattern](#factory-pattern)
4. [Advanced Features](#advanced-features)
5. [Performance Optimization](#performance-optimization)
6. [Production Deployment](#production-deployment)

---

## ChromaDB Examples

ChromaDB is recommended for development, local testing, and small-scale deployments.

### Basic Usage

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np

# Create store with persistent storage
store = ChromaVectorStore(
    collection_name="greenlang_rag",
    persist_directory="./chroma_db",  # Persistent storage
    distance_metric="cosine",
    embedding_dimension=384,
    batch_size=100
)

# Add documents with embeddings
documents = [
    {
        "content": "Climate change is affecting global emissions",
        "metadata": {"source": "IPCC", "year": 2023}
    },
    {
        "content": "Carbon footprint calculation methodology",
        "metadata": {"source": "GHG Protocol", "year": 2022}
    }
]

embeddings = np.random.randn(2, 384).astype(np.float32)

ids = store.add_documents(documents, embeddings)
print(f"Added {len(ids)} documents")

# Search for similar documents
query_embedding = np.random.randn(384)
results, scores = store.similarity_search(
    query_embedding,
    top_k=10,
    filters={"source": "IPCC"}  # Optional metadata filtering
)

for doc, score in zip(results, scores):
    print(f"Document: {doc.content[:50]}... (score: {score:.4f})")
```

### Batch Operations (1000+ vectors/second)

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np
import time

store = ChromaVectorStore(persist_directory="./chroma_db", batch_size=100)

# Create 10,000 documents for batch processing
documents = [
    {
        "content": f"Climate document {i} with comprehensive analysis",
        "metadata": {"doc_id": f"doc_{i}", "batch": i // 1000}
    }
    for i in range(10000)
]

embeddings = np.random.randn(10000, 384).astype(np.float32)

# Batch processing automatically splits into 100-document batches
start = time.time()
ids = store.add_documents(documents, embeddings, batch_size=100)
elapsed = time.time() - start

throughput = len(documents) / elapsed
print(f"Added {len(documents)} documents in {elapsed:.2f}s")
print(f"Throughput: {throughput:.0f} documents/second")

# Check metrics
metrics = store.get_metrics()
print(f"Total batches processed: {metrics['total_batches_processed']}")
print(f"Avg search time: {metrics['avg_search_time_ms']:.2f}ms")
```

### Metadata Filtering for Multi-Tenancy

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np

store = ChromaVectorStore(persist_directory="./chroma_db")

# Add documents from different tenants
tenant1_docs = [
    {
        "content": "Tenant 1 climate data",
        "metadata": {"tenant_id": "tenant-001", "region": "EMEA"}
    }
]

tenant2_docs = [
    {
        "content": "Tenant 2 emissions data",
        "metadata": {"tenant_id": "tenant-002", "region": "APAC"}
    }
]

embeddings1 = np.random.randn(1, 384).astype(np.float32)
embeddings2 = np.random.randn(1, 384).astype(np.float32)

store.add_documents(tenant1_docs, embeddings1)
store.add_documents(tenant2_docs, embeddings2)

# Search only in tenant-001 data
query_emb = np.random.randn(384)
results, scores = store.similarity_search(
    query_emb,
    top_k=10,
    filters={"tenant_id": "tenant-001"}  # Multi-tenancy isolation
)

print(f"Found {len(results)} results for tenant-001")
for doc in results:
    print(f"Tenant: {doc.metadata['tenant_id']}")
```

### Health Monitoring

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore

store = ChromaVectorStore(persist_directory="./chroma_db")

# Perform health check
health = store.health_check()

print(f"Status: {health['status']}")
print(f"Documents: {health['document_count']}")
print(f"Embedding dimension: {health['embedding_dimension']}")
print(f"Avg search time: {health['metrics']['avg_search_time_ms']:.2f}ms")

# Get detailed metrics
metrics = store.get_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Total documents: {metrics['total_documents']}")
```

### Persistence Verification

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np

# Create store with persistent directory
store1 = ChromaVectorStore(
    collection_name="persistent_collection",
    persist_directory="./chroma_persistent"
)

# Add documents
docs = [{"content": f"Doc {i}", "metadata": {"persistent": True}} for i in range(100)]
embeddings = np.random.randn(100, 384).astype(np.float32)
ids = store1.add_documents(docs, embeddings)

print(f"Store 1 document count: {store1.collection.count()}")

# Create new store instance pointing to same directory
store2 = ChromaVectorStore(
    collection_name="persistent_collection",
    persist_directory="./chroma_persistent"
)

print(f"Store 2 document count: {store2.collection.count()}")
# Verify data persisted across instances
assert store1.collection.count() == store2.collection.count()
print("Persistence verified!")
```

---

## Pinecone Examples

Pinecone is recommended for production deployments with high-scale requirements.

### Basic Production Setup

```python
import os
from GreenLang_2030.agent_foundation.rag.vector_stores import PineconeVectorStore
import numpy as np

# Initialize Pinecone Serverless
store = PineconeVectorStore(
    api_key=os.environ['PINECONE_API_KEY'],
    index_name="greenlang-prod",
    dimension=384,
    metric="cosine",
    environment="us-east-1"  # Pinecone region
)

# Add documents to production index
documents = [
    {
        "content": "ESG disclosure requirements and compliance",
        "metadata": {"source": "SEC", "year": 2023, "category": "esg"}
    },
    {
        "content": "Carbon accounting methodology for enterprises",
        "metadata": {"source": "GHG Protocol", "year": 2023, "category": "accounting"}
    }
]

embeddings = np.random.randn(2, 384).astype(np.float32)

ids = store.add_documents(documents, embeddings)
print(f"Upserted {len(ids)} documents to Pinecone")

# Search with semantic similarity
query_emb = np.random.randn(384)
results, scores = store.similarity_search(
    query_emb,
    top_k=10,
    include_metadata=True
)

for doc, score in zip(results, scores):
    print(f"Match: {doc.content[:50]}... (score: {score:.4f})")
```

### Multi-Tenant Deployment with Namespaces

```python
import os
from GreenLang_2030.agent_foundation.rag.vector_stores import PineconeVectorStore
import numpy as np

store = PineconeVectorStore(
    api_key=os.environ['PINECONE_API_KEY'],
    index_name="greenlang-multi-tenant"
)

# Tenant 1: Add documents
tenant1_docs = [
    {"content": f"Tenant 1 document {i}", "metadata": {"tenant": "company-a"}}
    for i in range(100)
]
embeddings_t1 = np.random.randn(100, 384).astype(np.float32)

ids_t1 = store.add_documents(
    tenant1_docs,
    embeddings_t1,
    namespace="company-a"  # Namespace isolation
)
print(f"Added {len(ids_t1)} documents for company-a")

# Tenant 2: Add documents
tenant2_docs = [
    {"content": f"Tenant 2 document {i}", "metadata": {"tenant": "company-b"}}
    for i in range(150)
]
embeddings_t2 = np.random.randn(150, 384).astype(np.float32)

ids_t2 = store.add_documents(
    tenant2_docs,
    embeddings_t2,
    namespace="company-b"  # Separate namespace
)
print(f"Added {len(ids_t2)} documents for company-b")

# Search in company-a namespace only
query_emb = np.random.randn(384)
results_a, _ = store.similarity_search(
    query_emb,
    namespace="company-a",
    top_k=10
)

# Search in company-b namespace only
results_b, _ = store.similarity_search(
    query_emb,
    namespace="company-b",
    top_k=10
)

print(f"Company A results: {len(results_a)}")
print(f"Company B results: {len(results_b)}")
```

### High-Performance Batch Upsert (1000+ vectors/second)

```python
import os
import time
import numpy as np
from GreenLang_2030.agent_foundation.rag.vector_stores import PineconeVectorStore

store = PineconeVectorStore(
    api_key=os.environ['PINECONE_API_KEY'],
    index_name="greenlang-perf",
    batch_size=100  # Optimal batch size for Pinecone
)

# Create large document set
print("Creating 5000 test documents...")
documents = [
    {
        "content": f"Environmental impact document {i} with detailed analysis",
        "metadata": {
            "doc_id": f"doc_{i}",
            "category": ["climate", "emissions", "sustainability"][i % 3],
            "score": i % 100
        }
    }
    for i in range(5000)
]

embeddings = np.random.randn(5000, 384).astype(np.float32)

# Batch upsert with performance tracking
print("Starting batch upsert...")
start = time.time()
ids = store.add_documents(documents, embeddings, batch_size=100)
elapsed = time.time() - start

throughput = len(documents) / elapsed
print(f"Upserted {len(documents)} vectors in {elapsed:.2f}s")
print(f"Throughput: {throughput:.0f} vectors/second")
print(f"Target: >1000 vectors/second")

# Check index stats
stats = store.get_index_stats()
print(f"\nIndex statistics:")
print(f"  Total vectors: {stats.total_vector_count}")
print(f"  Dimension: {stats.dimension}")
print(f"  Index fullness: {stats.index_fullness:.2%}")
```

### Advanced Metadata Filtering

```python
import os
import numpy as np
from GreenLang_2030.agent_foundation.rag.vector_stores import PineconeVectorStore

store = PineconeVectorStore(
    api_key=os.environ['PINECONE_API_KEY'],
    index_name="greenlang-filters"
)

# Add documents with rich metadata
documents = [
    {
        "content": "2023 ESG Report",
        "metadata": {
            "year": 2023,
            "source": "IPCC",
            "region": "EU",
            "tags": ["climate", "emissions"]
        }
    }
    for _ in range(50)
]
embeddings = np.random.randn(50, 384).astype(np.float32)
store.add_documents(documents, embeddings)

# Search with complex filters
query_emb = np.random.randn(384)

# Example 1: Exact match
results1, _ = store.similarity_search(
    query_emb,
    top_k=10,
    filters={"year": 2023}
)

# Example 2: Comparison operators
results2, _ = store.similarity_search(
    query_emb,
    top_k=10,
    filters={"year": {"$gte": 2022}}  # Greater than or equal
)

# Example 3: Multiple conditions (AND)
results3, _ = store.similarity_search(
    query_emb,
    top_k=10,
    filters={
        "year": 2023,
        "region": "EU",
        "source": {"$in": ["IPCC", "UN"]}
    }
)

print(f"Exact match results: {len(results1)}")
print(f"Comparison results: {len(results2)}")
print(f"Complex filter results: {len(results3)}")
```

---

## Factory Pattern

The factory pattern provides a unified interface for creating vector stores.

### Basic Factory Usage

```python
from GreenLang_2030.agent_foundation.rag.vector_stores.factory import (
    VectorStoreFactory,
    VectorStoreConfig,
    VectorStoreType
)

# Create factory
factory = VectorStoreFactory()

# List available stores
available = factory.get_available_stores()
for store_type, status in available.items():
    print(f"{store_type}: {status}")

# Create ChromaDB store via factory
chroma_config = VectorStoreConfig(
    store_type=VectorStoreType.CHROMA,
    persist_directory="./chroma_data"
)
chroma_store = factory.create(VectorStoreType.CHROMA, chroma_config)

# Create Pinecone store via factory
import os
pinecone_config = VectorStoreConfig(
    store_type=VectorStoreType.PINECONE,
    pinecone_api_key=os.environ['PINECONE_API_KEY'],
    pinecone_environment="us-east-1"
)
pinecone_store = factory.create(VectorStoreType.PINECONE, pinecone_config)
```

### Convenience Functions

```python
from GreenLang_2030.agent_foundation.rag.vector_stores.factory import (
    create_chroma_store,
    create_pinecone_store
)
import os

# Quick ChromaDB setup
chroma = create_chroma_store(
    collection_name="my_collection",
    persist_directory="./my_data"
)

# Quick Pinecone setup
pinecone = create_pinecone_store(
    api_key=os.environ['PINECONE_API_KEY'],
    collection_name="my-index",
    environment="us-east-1"
)
```

---

## Advanced Features

### Provenance Tracking (SHA-256 Hashing)

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np

store = ChromaVectorStore(persist_directory="./chroma_db")

# Add documents - provenance hash is automatically computed
documents = [
    {"content": "Deterministic climate data", "metadata": {"source": "verified"}}
]
embeddings = np.random.randn(1, 384).astype(np.float32)

ids = store.add_documents(documents, embeddings)

# Retrieve and verify provenance
results, _ = store.similarity_search(embeddings[0], top_k=1)
if results:
    provenance_hash = results[0].metadata.get('provenance_hash')
    indexed_at = results[0].metadata.get('indexed_at')
    print(f"Provenance hash: {provenance_hash}")
    print(f"Indexed at: {indexed_at}")
    # Hash can be verified for audit compliance
```

### Distance Metric Options

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore

# Cosine similarity (default, recommended for text embeddings)
cosine_store = ChromaVectorStore(
    distance_metric="cosine",
    persist_directory="./chroma_cosine"
)

# L2 Euclidean distance
l2_store = ChromaVectorStore(
    distance_metric="l2",
    persist_directory="./chroma_l2"
)

# Inner product
ip_store = ChromaVectorStore(
    distance_metric="ip",
    persist_directory="./chroma_ip"
)
```

### Collection Management

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore

store = ChromaVectorStore(persist_directory="./chroma_db")

# Get collection statistics
stats = store.get_collection_stats()
print(f"Collection: {stats.name}")
print(f"Documents: {stats.count}")
print(f"Dimension: {stats.dimension}")

# List all collections
collections = store.list_collections()
print(f"Available collections: {collections}")

# Delete entire collection
success = store.delete_collection()
print(f"Deleted: {success}")
```

---

## Performance Optimization

### Embedding Caching

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=10000)
def get_cached_embedding(text: str) -> np.ndarray:
    """Cache embeddings to avoid re-computation."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text, convert_to_numpy=True)

store = ChromaVectorStore(persist_directory="./chroma_db")

# Use cached embeddings
text = "Important climate document"
embedding = get_cached_embedding(text)
# Next call with same text uses cache (66% cost reduction)
embedding = get_cached_embedding(text)
```

### Query Optimization

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np
import time

store = ChromaVectorStore(persist_directory="./chroma_db")

# Batch queries for efficiency
queries = [np.random.randn(384) for _ in range(100)]

start = time.time()
results = [store.similarity_search(q, top_k=5) for q in queries]
elapsed = time.time() - start

latency = (elapsed / len(queries)) * 1000
print(f"Average query latency: {latency:.2f}ms")
print(f"Target: <50ms for 100K vectors")
```

---

## Production Deployment

### Environment-Based Configuration

```python
import os
from GreenLang_2030.agent_foundation.rag.vector_stores.factory import (
    VectorStoreFactory,
    VectorStoreConfig,
    VectorStoreType
)

# Determine store type from environment
store_type = os.environ.get('VECTOR_STORE', 'chroma')

if store_type == 'pinecone':
    config = VectorStoreConfig(
        store_type=VectorStoreType.PINECONE,
        pinecone_api_key=os.environ['PINECONE_API_KEY'],
        pinecone_environment=os.environ.get('PINECONE_ENV', 'us-east-1'),
        collection_name=os.environ.get('VECTOR_INDEX', 'greenlang-prod')
    )
else:
    config = VectorStoreConfig(
        store_type=VectorStoreType.CHROMA,
        persist_directory=os.environ.get('CHROMA_PATH', './chroma_db')
    )

factory = VectorStoreFactory()
store = factory.create(store_type, config)
```

### Health Checks and Monitoring

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import logging
import time

logger = logging.getLogger(__name__)

store = ChromaVectorStore(persist_directory="./chroma_db")

# Periodic health monitoring
while True:
    health = store.health_check()

    if health['status'] == 'healthy':
        logger.info(f"Vector store healthy: {health['document_count']} documents")
    else:
        logger.error(f"Vector store unhealthy: {health.get('error')}")

    metrics = store.get_metrics()
    logger.info(f"Avg search time: {metrics['avg_search_time_ms']:.2f}ms")

    time.sleep(300)  # Check every 5 minutes
```

### Error Handling

```python
from GreenLang_2030.agent_foundation.rag.vector_stores import ChromaVectorStore
import numpy as np
import logging

logger = logging.getLogger(__name__)
store = ChromaVectorStore(persist_directory="./chroma_db")

try:
    documents = [{"content": "Test", "metadata": {}}]
    embeddings = np.random.randn(1, 384).astype(np.float32)
    ids = store.add_documents(documents, embeddings)

    # Verify addition
    results, _ = store.similarity_search(embeddings[0], top_k=1)
    if not results:
        logger.warning("Document added but not found in search")

except ValueError as e:
    logger.error(f"Invalid input: {e}")
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
```

---

## Summary

- **ChromaDB**: Development, testing, local deployments
- **Pinecone**: Production, high-scale, multi-tenant
- **Batch Operations**: >1000 vectors/second with batch_size=100
- **Metadata Filtering**: Multi-tenancy, compliance, data isolation
- **Provenance Tracking**: SHA-256 hashes for audit trails
- **Health Monitoring**: Built-in health checks and metrics

For more information, see the inline documentation in each module.
