# Weaviate Quick Start Guide

Get up and running with Weaviate in 5 minutes.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with GreenLang installed

## Step 1: Start Weaviate (30 seconds)

```bash
cd docker/weaviate
docker-compose up -d
```

Wait for health check:
```bash
# Should return HTTP 200
curl http://localhost:8080/v1/.well-known/ready
```

## Step 2: Install Dependencies (1 minute)

```bash
pip install weaviate-client>=3.25.0
# Or: pip install -e ".[llm]"
```

## Step 3: Run Example (30 seconds)

```bash
python examples/rag/weaviate_example.py
```

## Step 4: Use in Your Code (2 minutes)

```python
from greenlang.intelligence.rag import RAGConfig, get_vector_store
import numpy as np

# Configure
config = RAGConfig(
    vector_store_provider="weaviate",
    weaviate_endpoint="http://localhost:8080",
    allowlist=["test_collection"],
)

# Create store
store = get_vector_store(dimension=384, config=config)

# Add documents (with your actual chunks and embeddings)
from greenlang.intelligence.rag import Document, Chunk

chunk = Chunk(
    chunk_id="test-001",
    doc_id="doc-001",
    section_path="Test Section",
    section_hash="abc",
    text="Test text",
    start_char=0,
    end_char=9,
    token_count=2,
)

embedding = np.random.randn(384).astype(np.float32)
doc = Document(chunk=chunk, embedding=embedding)

store.add_documents([doc], collection="test_collection")

# Search
query_emb = np.random.randn(384).astype(np.float32)
results = store.similarity_search(query_emb, k=5, collections=["test_collection"])

print(f"Found {len(results)} results")
for r in results:
    print(f"  - {r.chunk.chunk_id}: {r.chunk.text[:50]}")
```

## Step 5: Clean Up

```bash
cd docker/weaviate
docker-compose down

# To also delete data:
docker-compose down -v
```

## Troubleshooting

**Port 8080 already in use?**
```bash
# Edit docker/weaviate/docker-compose.yml
# Change: "8080:8080" -> "8081:8080"
# Update endpoint: WEAVIATE_ENDPOINT="http://localhost:8081"
```

**Connection timeout?**
```bash
# Wait longer for startup (can take 30s on first run)
docker-compose logs -f weaviate
```

**Schema errors?**
```python
from greenlang.intelligence.rag.weaviate_client import WeaviateClient
client = WeaviateClient()
client.delete_schema()
client.ensure_schema()
```

## Next Steps

1. Read full documentation: `examples/rag/WEAVIATE_SETUP.md`
2. Integrate with RAGEngine
3. Use with real embeddings (MiniLM/OpenAI)
4. Set up production deployment

## Common Commands

```bash
# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Stop
docker-compose stop

# Start
docker-compose start

# Delete everything
docker-compose down -v
```

## Environment Variables

```bash
# Optional: Set in .env file
export WEAVIATE_ENDPOINT="http://localhost:8080"
export GL_VECTOR_STORE="weaviate"
export GL_RAG_ALLOWLIST="test_collection,ghg_protocol_corp"
```

## Performance Tips

- Use batch size 100+ for bulk ingestion
- Enable collection filtering to speed up search
- Increase memory limit in docker-compose.yml for large datasets
- Use persistent volumes to avoid data loss

## Security Notes

- Default setup uses anonymous access (dev only)
- For production, enable API key authentication
- Use allowlist to restrict collections
- Run in private network/VPN

## Support

- Full docs: `examples/rag/WEAVIATE_SETUP.md`
- Issues: https://github.com/greenlang/greenlang/issues
- Discord: https://discord.gg/greenlang
