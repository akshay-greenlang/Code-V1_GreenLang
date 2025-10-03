# Weaviate Integration for GreenLang RAG

Complete Weaviate vector store integration for INTL-104 RAG v1.

## Overview

This integration provides a production-ready vector store backend for the GreenLang RAG system using Weaviate, a self-hosted vector database.

### Features

- **Self-hosted deployment**: Full control over data and infrastructure
- **Collection-based filtering**: Security through allowlisting
- **Batch operations**: Efficient ingestion with dynamic sizing
- **KNN search**: Fast similarity search with L2 distance
- **Persistent storage**: Data survives container restarts
- **Health monitoring**: Built-in health checks and statistics

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   RAGEngine                         │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              WeaviateProvider                       │
│  - add_documents()                                  │
│  - similarity_search()                              │
│  - save/load config                                 │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              WeaviateClient                         │
│  - Connection management                            │
│  - Schema creation (idempotent)                     │
│  - Batch operations                                 │
│  - Health checks                                    │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Weaviate Server                        │
│  - Docker container                                 │
│  - Port 8080 (HTTP) / 50051 (gRPC)                 │
│  - Persistent volume                                │
└─────────────────────────────────────────────────────┘
```

## Schema

The Chunk class schema (matches CTO spec Section 6):

```python
{
    "class": "Chunk",
    "properties": [
        {"name": "chunk_id", "dataType": ["string"]},      # UUID v5
        {"name": "collection", "dataType": ["string"]},     # For filtering
        {"name": "doc_id", "dataType": ["string"]},
        {"name": "title", "dataType": ["string"]},
        {"name": "publisher", "dataType": ["string"]},
        {"name": "year", "dataType": ["string"]},
        {"name": "version", "dataType": ["string"]},
        {"name": "section_path", "dataType": ["string"]},
        {"name": "section_hash", "dataType": ["string"]},
        {"name": "page_start", "dataType": ["int"]},
        {"name": "page_end", "dataType": ["int"]},
        {"name": "para_index", "dataType": ["int"]},
        {"name": "text", "dataType": ["text"]},
    ],
    "vectorizer": "none"  # We provide embeddings externally
}
```

## Installation

### 1. Install Dependencies

```bash
# Install weaviate-client
pip install weaviate-client>=3.25.0

# Or install all LLM dependencies
pip install -e ".[llm]"
```

### 2. Start Weaviate

```bash
# Navigate to Weaviate docker directory
cd docker/weaviate

# Start Weaviate container
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f weaviate
```

### 3. Verify Health

```bash
# Check Weaviate health endpoint
curl http://localhost:8080/v1/.well-known/ready

# Should return: {"status": "ok"}
```

## Configuration

### Environment Variables

```bash
# Weaviate endpoint
export WEAVIATE_ENDPOINT="http://localhost:8080"

# API key (optional, not needed for local dev)
export WEAVIATE_API_KEY=""

# Set vector store provider
export GL_VECTOR_STORE="weaviate"
```

### RAGConfig

```python
from greenlang.intelligence.rag import RAGConfig

config = RAGConfig(
    mode="live",
    vector_store_provider="weaviate",
    weaviate_endpoint="http://localhost:8080",
    allowlist=["ghg_protocol_corp", "test_collection"],
    embedding_provider="minilm",
    embedding_dimension=384,
)
```

## Usage

### Basic Usage

```python
from greenlang.intelligence.rag import (
    RAGConfig,
    get_vector_store,
    Document,
    Chunk,
)
import numpy as np

# Create config
config = RAGConfig(
    vector_store_provider="weaviate",
    weaviate_endpoint="http://localhost:8080",
)

# Get vector store
store = get_vector_store(dimension=384, config=config)

# Create document
chunk = Chunk(
    chunk_id="chunk-001",
    doc_id="doc-001",
    section_path="Section 1",
    section_hash="abc123",
    page_start=1,
    text="Sample text about climate.",
    # ... other fields
)

embedding = np.random.randn(384).astype(np.float32)
doc = Document(chunk=chunk, embedding=embedding)

# Add to Weaviate
store.add_documents([doc], collection="test_collection")

# Search
query_embedding = np.random.randn(384).astype(np.float32)
results = store.similarity_search(
    query_embedding=query_embedding,
    k=5,
    collections=["test_collection"],
)

# View results
for result in results:
    print(f"Chunk: {result.chunk.chunk_id}")
    print(f"Distance: {result.metadata['distance']}")
    print(f"Text: {result.chunk.text[:100]}")
```

### With RAGEngine

```python
from greenlang.intelligence.rag import RAGEngine, RAGConfig

config = RAGConfig(
    vector_store_provider="weaviate",
    weaviate_endpoint="http://localhost:8080",
)

engine = RAGEngine(config)

# Ingest documents
# engine.ingest(...)

# Query
# result = await engine.query("emission factors")
```

## Testing

### Run Example

```bash
# Start Weaviate first
cd docker/weaviate
docker-compose up -d

# Run example
cd ../..
python examples/rag/weaviate_example.py
```

Expected output:
```
======================================================================
GreenLang RAG - Weaviate Integration Example
======================================================================

[1] Checking Weaviate health...
   ✓ Weaviate is running and healthy
   - Endpoint: http://localhost:8080
   - Total objects: 0

[2] Creating RAG configuration with Weaviate...
   - Vector store: weaviate
   ...
```

### Manual Testing

```bash
# Check Weaviate schema
curl http://localhost:8080/v1/schema

# Get object count
curl http://localhost:8080/v1/objects?limit=1

# Query GraphQL
curl -X POST http://localhost:8080/v1/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query": "{Aggregate{Chunk{meta{count}}}}"}'
```

### Integration Tests

```python
import pytest
from greenlang.intelligence.rag import WeaviateProvider, RAGConfig

@pytest.fixture
def weaviate_provider():
    config = RAGConfig(
        vector_store_provider="weaviate",
        weaviate_endpoint="http://localhost:8080",
    )
    provider = WeaviateProvider(dimension=384, config=config)

    # Clean up after test
    yield provider

    # Optional: delete test data
    provider.weaviate_client.delete_schema()

def test_add_documents(weaviate_provider):
    # Test document addition
    pass

def test_similarity_search(weaviate_provider):
    # Test search functionality
    pass
```

## Troubleshooting

### Weaviate not starting

```bash
# Check logs
docker-compose logs weaviate

# Common issues:
# - Port 8080 already in use
# - Insufficient memory (needs ~2GB)
# - Data corruption (try: docker-compose down -v)
```

### Connection timeout

```python
# Increase timeout
from greenlang.intelligence.rag.weaviate_client import WeaviateClient

client = WeaviateClient(
    endpoint="http://localhost:8080",
    timeout_config=60000,  # 60 seconds
    startup_period=60,
)
```

### Schema errors

```python
# Reset schema (WARNING: deletes all data)
client = WeaviateClient()
client.delete_schema()
client.ensure_schema()
```

### Collection not allowed

```
ValueError: Collection 'xyz' is not allowed. Allowed: ghg_protocol_corp, ...
```

Solution: Add collection to allowlist:

```python
config = RAGConfig(
    allowlist=["ghg_protocol_corp", "xyz"],
)
```

## Performance

### Benchmarks (local dev)

- **Ingestion**: ~1000 docs/sec (batch size 100)
- **Search**: ~10ms for k=10 (10k total docs)
- **Memory**: ~2GB for Weaviate container
- **Disk**: ~100MB per 10k chunks (384-dim embeddings)

### Optimization Tips

1. **Batch size**: Increase for large ingestions (100-500)
2. **Parallel queries**: Use async/await for multiple queries
3. **Collection filtering**: Pre-filter to reduce search space
4. **Schema indexing**: Enable inverted index for text fields
5. **Resource limits**: Adjust GOMEMLIMIT in docker-compose.yml

## Production Deployment

### Docker Compose (Production)

```yaml
services:
  weaviate:
    image: semitechnologies/weaviate:1.25.5
    environment:
      AUTHENTICATION_APIKEY_ENABLED: 'true'
      AUTHENTICATION_APIKEY_ALLOWED_KEYS: '${WEAVIATE_API_KEY}'
      GOMAXPROCS: '4'
      GOMEMLIMIT: '8GiB'
    resources:
      limits:
        memory: 8Gi
        cpus: '4'
```

### Kubernetes

See `docker/weaviate/k8s/` for Kubernetes manifests.

### Cloud Deployment

- **AWS**: Use ECS/EKS with EBS volumes
- **GCP**: Use GKE with Persistent Disks
- **Azure**: Use AKS with Azure Disks

## Monitoring

### Metrics

```python
# Get stats
stats = provider.get_stats()
print(f"Total docs: {stats['total_documents_added']}")
print(f"Collections: {stats['collections_added']}")

# Weaviate metrics
weaviate_stats = stats['weaviate_stats']
print(f"Total objects: {weaviate_stats['total_objects']}")
```

### Prometheus

Weaviate exposes metrics at `:2112/metrics`:

```bash
curl http://localhost:2112/metrics
```

Key metrics:
- `weaviate_object_count_total`
- `weaviate_query_duration_ms`
- `weaviate_batch_duration_ms`

## Comparison: Weaviate vs FAISS

| Feature | Weaviate | FAISS |
|---------|----------|-------|
| **Persistence** | Yes (disk) | No (memory/manual save) |
| **Scalability** | Horizontal | Vertical |
| **Setup** | Docker required | Pure Python |
| **Performance** | ~10ms | ~1ms |
| **Production Ready** | Yes | No (needs wrapper) |
| **Best For** | Production | Dev/Testing |

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Python Client Docs](https://weaviate.io/developers/weaviate/client-libraries/python)
- [Docker Deployment](https://weaviate.io/developers/weaviate/installation/docker-compose)
- [CTO Spec Section 6](../../docs/INTL-104-RAG-v1-spec.md#section-6-vector-store)

## Support

For issues or questions:
- GitHub Issues: https://github.com/greenlang/greenlang/issues
- Discord: https://discord.gg/greenlang
- Email: support@greenlang.io
