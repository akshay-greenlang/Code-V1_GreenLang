# Vector Store Quick Start Guide

Get up and running with GreenLang's vector stores in 5 minutes.

## Installation

```bash
# Install dependencies
pip install chromadb>=0.4.0 pinecone-client>=3.0.0 sentence-transformers>=2.2.0

# Or use the main requirements.txt
pip install -e .
```

## 30-Second Example

### ChromaDB (Development)

```python
from vector_stores import create_chroma_store
import numpy as np

# Create store
store = create_chroma_store(persist_directory="./chroma_db")

# Add documents
docs = [
    {"content": "Climate change impacts", "metadata": {"source": "IPCC"}},
    {"content": "Carbon emissions", "metadata": {"source": "IEA"}},
]
embeddings = np.random.randn(2, 384).astype(np.float32)

ids = store.add_documents(docs, embeddings)

# Search
query_emb = np.random.randn(384)
results, scores = store.similarity_search(query_emb, top_k=2)

# Done! Results ready
for doc, score in zip(results, scores):
    print(f"{doc.content} (score: {score:.4f})")
```

### Pinecone (Production)

```python
from vector_stores import create_pinecone_store
import os

# Create store
store = create_pinecone_store(
    api_key=os.environ['PINECONE_API_KEY'],
    collection_name="my-index"
)

# Add documents
docs = [{"content": "Sustainability metrics"}]
embeddings = np.random.randn(1, 384).astype(np.float32)

ids = store.add_documents(docs, embeddings)

# Search
results, scores = store.similarity_search(query_emb, top_k=5)
```

## Common Tasks

### 1. Batch Add Documents

```python
store.add_documents(
    documents=large_doc_list,  # 1000+ documents
    embeddings=embeddings_array,
    batch_size=100  # Processes in batches for efficiency
)
```

### 2. Search with Filters

```python
# ChromaDB
results, scores = store.similarity_search(
    query_embedding,
    top_k=10,
    filters={"source": "IPCC", "year": 2023}
)

# Pinecone
results, scores = store.similarity_search(
    query_embedding,
    filters={"year": {"$gte": 2023}}
)
```

### 3. Multi-Tenant Isolation

```python
# Pinecone: Use namespaces
store.add_documents(
    docs,
    embeddings,
    namespace="company-a"  # Isolated per tenant
)

# Search in specific namespace
store.similarity_search(
    query_emb,
    namespace="company-a"
)
```

### 4. Delete Documents

```python
success = store.delete(["doc_id_1", "doc_id_2"])
```

### 5. Update Documents

```python
success = store.update(
    ids=["doc_id_1"],
    documents=[{"content": "Updated content"}],
    metadata=[{"updated": True}]
)
```

### 6. Health Check

```python
health = store.health_check()
print(f"Status: {health['status']}")
print(f"Documents: {health['document_count']}")

# Get metrics
metrics = store.get_metrics()
print(f"Avg search: {metrics['avg_search_time_ms']:.2f}ms")
```

## Configuration

### ChromaDB Configuration

```python
from vector_stores import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="my_collection",      # Collection name
    persist_directory="./chroma_db",      # Where to store data
    distance_metric="cosine",              # cosine, l2, ip
    embedding_dimension=384,               # Dimension of embeddings
    batch_size=100                         # Batch size for operations
)
```

### Pinecone Configuration

```python
from vector_stores import PineconeVectorStore

store = PineconeVectorStore(
    api_key="your-api-key",               # Pinecone API key
    index_name="my-index",                # Index name
    dimension=384,                        # Embedding dimension
    metric="cosine",                      # cosine, euclidean, dotproduct
    namespace="tenant-1",                 # Default namespace
    cloud="aws",                          # aws, gcp, azure
    region="us-east-1",                   # Cloud region
    batch_size=100                        # Optimal: 100 for Pinecone
)
```

### Factory Pattern

```python
from vector_stores.factory import VectorStoreFactory, VectorStoreConfig

config = VectorStoreConfig(
    store_type="chroma",
    collection_name="my_collection",
    persist_directory="./data"
)

factory = VectorStoreFactory()
store = factory.create("chroma", config)
```

## Performance Tips

1. **Batch Size**: Use 100 for optimal throughput
2. **Distance Metric**: Use "cosine" for text embeddings
3. **Caching**: Cache embeddings to reduce computation
4. **Query Limits**: Use `top_k=10` unless you need more
5. **Metadata Filtering**: Filter at query time, not post-processing

## Troubleshooting

### "ImportError: No module named chromadb"

```bash
pip install chromadb>=0.4.0
```

### "Pinecone API key invalid"

```bash
export PINECONE_API_KEY="your-actual-key"
python your_script.py
```

### "Slow search performance"

```python
# Check collection size
stats = store.get_collection_stats()

# Reduce top_k if not needed
results, scores = store.similarity_search(query_emb, top_k=5)

# Enable query caching
```

### "Out of memory with large batches"

```python
# Reduce batch size
store.add_documents(docs, embeddings, batch_size=50)
```

## Environment Variables

```bash
# ChromaDB
export CHROMA_PATH=./chroma_db
export CHROMA_COLLECTION=greenlang_rag

# Pinecone
export PINECONE_API_KEY=your-key
export PINECONE_ENV=us-east-1
export PINECONE_INDEX=greenlang-prod

# General
export VECTOR_STORE=chroma  # or pinecone
```

## Next Steps

1. Read the full [EXAMPLES.md](EXAMPLES.md) for advanced usage
2. Check [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for deployment
3. Review [test_vector_stores.py](test_vector_stores.py) for test patterns

## Support

- **Documentation**: See EXAMPLES.md and PRODUCTION_READINESS.md
- **Testing**: Run pytest on test_vector_stores.py
- **Issues**: Check logs in console or use health_check()
