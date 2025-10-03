# RAG System Troubleshooting Guide

**Document:** INTL-104 RAG System Troubleshooting
**Version:** 1.0
**Last Updated:** 2025-10-03

## Table of Contents

1. [Common Issues](#common-issues)
2. [Error Messages](#error-messages)
3. [Performance Issues](#performance-issues)
4. [Configuration Problems](#configuration-problems)
5. [Testing Issues](#testing-issues)
6. [Debug Mode](#debug-mode)

---

## Common Issues

### Weaviate Connection Failures

**Symptoms:**
- Error: "Failed to create Weaviate client"
- Error: "Weaviate not ready after N attempts"
- Timeouts during initialization
- Health check failures

**Root Cause:**
- Weaviate server not running
- Incorrect endpoint URL
- Network connectivity issues
- Weaviate startup time exceeded
- Firewall blocking connection

**Solution Steps:**

1. **Verify Weaviate is running:**
   ```bash
   # Check Docker container status
   docker ps | grep weaviate

   # Check Weaviate health endpoint
   curl http://localhost:8080/v1/.well-known/ready
   ```

2. **Check endpoint configuration:**
   ```python
   # Verify WEAVIATE_ENDPOINT environment variable
   echo $WEAVIATE_ENDPOINT

   # Should be: http://localhost:8080
   # Or your custom endpoint
   ```

3. **Increase startup period:**
   ```python
   from greenlang.intelligence.rag.weaviate_client import WeaviateClient

   client = WeaviateClient(
       endpoint="http://localhost:8080",
       startup_period=60,  # Increase to 60 seconds
       timeout_config=60000  # 60 second timeout
   )
   ```

4. **Check network connectivity:**
   ```bash
   # Test connection
   ping localhost
   telnet localhost 8080
   ```

**Prevention Tips:**
- Always start Weaviate before running RAG operations
- Use docker-compose for consistent startup
- Set appropriate startup_period based on system resources
- Monitor Weaviate logs for startup issues

---

### Embedding Errors

**Symptoms:**
- Error: "Cannot embed empty text list"
- Error: "sentence-transformers not installed"
- Error: "Failed to load model"
- Slow embedding generation
- Out of memory errors

**Root Cause:**
- Missing dependencies (sentence-transformers, torch)
- Empty or invalid input text
- Model download failures (network issues)
- Insufficient memory for model
- GPU/CPU configuration issues

**Solution Steps:**

1. **Install dependencies:**
   ```bash
   pip install sentence-transformers torch faiss-cpu
   ```

2. **Verify input text:**
   ```python
   texts = ["Sample text"]  # Must not be empty
   embeddings = await embedder.embed(texts)
   ```

3. **Download model manually:**
   ```python
   from sentence_transformers import SentenceTransformer

   # Download model (requires network)
   model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
   ```

4. **Configure for CPU-only:**
   ```python
   config = RAGConfig(
       embedding_provider="minilm",
       mode="replay"  # Forces CPU-only
   )
   ```

5. **Handle memory issues:**
   ```python
   config = RAGConfig(
       embedding_batch_size=16,  # Reduce batch size
   )
   ```

**Prevention Tips:**
- Pre-download models before offline deployment
- Validate input text before embedding
- Use appropriate batch sizes for your system
- Monitor memory usage during embedding
- Use CPU-only mode for determinism

---

### Vector Store Errors

**Symptoms:**
- Error: "FAISS index not found"
- Error: "Metadata not found"
- Error: "Embedding dimension mismatch"
- Error: "Document has no embedding"
- Index corruption

**Root Cause:**
- Vector store not built/saved
- Missing metadata files
- Dimension mismatch between embeddings and index
- Corrupted index files
- Incorrect file paths

**Solution Steps:**

1. **Verify vector store exists:**
   ```bash
   ls -la knowledge_base/vector_store/
   # Should contain: index.faiss, metadata.pkl
   ```

2. **Rebuild vector store:**
   ```python
   from greenlang.intelligence.rag import get_vector_store

   vector_store = get_vector_store(dimension=384)
   # Re-add documents
   vector_store.add_documents(docs, collection="my_collection")
   vector_store.save(Path("knowledge_base/vector_store"))
   ```

3. **Check dimension consistency:**
   ```python
   # Verify embedding dimension matches vector store
   config = RAGConfig(
       embedding_dimension=384,  # Must match model output
   )
   ```

4. **Validate documents before adding:**
   ```python
   for doc in documents:
       if doc.embedding is None:
           raise ValueError(f"Document {doc.chunk.chunk_id} missing embedding")
       if len(doc.embedding) != 384:
           raise ValueError(f"Dimension mismatch: {len(doc.embedding)}")
   ```

**Prevention Tips:**
- Always save vector store after building
- Keep metadata and index files together
- Use consistent embedding dimensions
- Validate all documents before batch operations
- Backup vector store regularly

---

### Query Failures

**Symptoms:**
- Error: "Vector store is empty"
- Error: "Weaviate search failed"
- No results returned
- Incorrect results
- Query timeouts

**Root Cause:**
- Empty vector store
- Collection filter mismatch
- Weaviate connectivity issues
- Invalid query embedding
- Network timeout

**Solution Steps:**

1. **Verify vector store has data:**
   ```python
   stats = vector_store.get_stats()
   print(f"Total documents: {stats['total_documents']}")
   ```

2. **Check collection names:**
   ```python
   # Verify collection exists and matches query
   collections = ["ghg_protocol_corp", "ipcc_ar6_wg3"]
   results = vector_store.similarity_search(
       query_embedding=emb,
       collections=collections  # Must match ingested collections
   )
   ```

3. **Validate query embedding:**
   ```python
   if len(query_embedding) != 384:
       raise ValueError("Query embedding dimension mismatch")
   ```

4. **Increase timeout:**
   ```python
   config = RAGConfig(
       query_timeout_seconds=60  # Increase from default 30
   )
   ```

**Prevention Tips:**
- Verify vector store is populated before querying
- Use consistent collection names
- Test queries with known documents
- Monitor query performance
- Set appropriate timeouts

---

### Ingestion Failures

**Symptoms:**
- Error: "File not found"
- Error: "File hash mismatch"
- Error: "Collection not in allowlist"
- Error: "Invalid collection name"
- Slow ingestion speed

**Root Cause:**
- Missing or incorrect file paths
- File tampering/modification
- Collection not in allowlist
- Invalid collection name format
- Large batch sizes

**Solution Steps:**

1. **Verify file exists:**
   ```python
   from pathlib import Path

   file_path = Path("documents/report.pdf")
   if not file_path.exists():
       raise FileNotFoundError(f"File not found: {file_path}")
   ```

2. **Disable hash verification (for testing):**
   ```python
   config = RAGConfig(
       verify_checksums=False  # Disable for testing
   )
   ```

3. **Add collection to allowlist:**
   ```python
   config = RAGConfig(
       allowlist=[
           "ghg_protocol_corp",
           "ipcc_ar6_wg3",
           "my_new_collection"  # Add your collection
       ]
   )
   ```

4. **Validate collection name:**
   ```python
   import re

   collection = "my_collection_123"  # Must be alphanumeric + underscore/hyphen
   if not re.match(r"^[a-zA-Z0-9_-]+$", collection):
       raise ValueError(f"Invalid collection name: {collection}")
   ```

5. **Optimize batch size:**
   ```python
   # For Weaviate
   result = weaviate_client.batch_add_objects(
       objects=objects,
       batch_size=50  # Reduce if experiencing issues
   )
   ```

**Prevention Tips:**
- Validate file paths before ingestion
- Use absolute paths
- Add collections to allowlist before ingestion
- Use standard collection naming conventions
- Monitor batch operation performance

---

## Error Messages

### Collection Not Allowed

**Error:**
```
ValueError: Collection 'my_collection' is not in allowlist.
Allowed collections: ghg_protocol_corp, ipcc_ar6_wg3, gl_docs
```

**Meaning:** The collection you're trying to access is not in the security allowlist.

**Fix:**
```python
# Add to allowlist in configuration
config = RAGConfig(
    allowlist=[
        "ghg_protocol_corp",
        "ipcc_ar6_wg3",
        "gl_docs",
        "my_collection"  # Add your collection
    ]
)

# Or via environment variable
export GL_RAG_ALLOWLIST="ghg_protocol_corp,ipcc_ar6_wg3,my_collection"
```

---

### Network Isolation Errors

**Error:**
```
RuntimeError: No cached result found for query in replay mode.
Cannot make network calls in replay mode.
```

**Meaning:** Replay mode enforces strict offline operation. The query result is not in cache.

**Fix:**

1. **Switch to live mode (if appropriate):**
   ```python
   config = RAGConfig(
       mode="live"  # Allows network access
   )
   ```

2. **Pre-populate cache (for replay mode):**
   ```python
   # First, run in record mode to build cache
   config = RAGConfig(
       mode="record"
   )

   det = DeterministicRAG(
       mode="record",
       cache_path=Path("cache.json"),
       config=config
   )

   # Execute queries to populate cache
   result = det.query(q="test query", top_k=5, collections=["test"])

   # Then switch to replay mode
   config.mode = "replay"
   ```

---

### Hash Mismatch Errors

**Error:**
```
RuntimeError: File hash mismatch for documents/report.pdf
Expected: abc123...
Computed: def456...
File may have been tampered with or metadata is incorrect.
```

**Meaning:** The file's content hash doesn't match the expected hash in metadata.

**Fix:**

1. **Update content_hash in metadata:**
   ```python
   from greenlang.intelligence.rag.hashing import file_hash

   computed_hash = file_hash("documents/report.pdf")
   doc_meta.content_hash = computed_hash
   doc_meta.doc_hash = computed_hash
   ```

2. **Disable verification (testing only):**
   ```python
   config = RAGConfig(
       verify_checksums=False
   )
   ```

3. **Verify file hasn't been modified:**
   ```bash
   # Check file modification time
   ls -l documents/report.pdf

   # Verify file integrity
   sha256sum documents/report.pdf
   ```

---

### Dimension Mismatch Errors

**Error:**
```
ValueError: Embedding dimension mismatch: expected 384, got 768
```

**Meaning:** The embedding vector dimension doesn't match the vector store configuration.

**Fix:**

1. **Use consistent embedding model:**
   ```python
   # all-MiniLM-L6-v2 -> 384 dimensions
   # all-mpnet-base-v2 -> 768 dimensions

   config = RAGConfig(
       embedding_model="sentence-transformers/all-MiniLM-L6-v2",
       embedding_dimension=384  # Must match model output
   )
   ```

2. **Rebuild vector store with correct dimension:**
   ```python
   vector_store = get_vector_store(
       dimension=768,  # Match your embedding model
       config=config
   )
   ```

---

## Performance Issues

### Slow Ingestion

**Symptoms:**
- Ingestion takes hours for large document sets
- High CPU/memory usage
- System becomes unresponsive

**Root Cause:**
- Large batch sizes
- Inefficient chunking
- Single-threaded processing
- Large embedding batches

**Solution Steps:**

1. **Optimize batch sizes:**
   ```python
   config = RAGConfig(
       embedding_batch_size=32,  # Reduce from 64
   )

   # For Weaviate
   batch_size = 50  # Reduce from 100
   ```

2. **Use smaller chunks:**
   ```python
   config = RAGConfig(
       chunk_size=256,  # Reduce from 512
       chunk_overlap=32  # Reduce from 64
   )
   ```

3. **Monitor progress:**
   ```python
   import logging
   logging.basicConfig(level=logging.INFO)

   # Will show progress during ingestion
   ```

4. **Process files incrementally:**
   ```python
   # Ingest one file at a time instead of batch
   for file_path in document_files:
       manifest = await ingest_path(
           path=file_path,
           collection=collection,
           doc_meta=create_doc_meta(file_path),
           config=config
       )
   ```

**Prevention Tips:**
- Start with small batch sizes and tune upward
- Profile ingestion to identify bottlenecks
- Use chunking strategy appropriate for content
- Monitor system resources during ingestion

---

### Slow Queries

**Symptoms:**
- Query takes > 10 seconds
- Weaviate timeouts
- UI becomes unresponsive

**Root Cause:**
- Large vector store (millions of vectors)
- High fetch_k for MMR
- Network latency to Weaviate
- Inefficient similarity search

**Solution Steps:**

1. **Reduce fetch_k:**
   ```python
   config = RAGConfig(
       default_fetch_k=10,  # Reduce from 30
       default_top_k=3      # Reduce from 6
   )
   ```

2. **Use collection filtering:**
   ```python
   # Filter by specific collections to reduce search space
   results = retriever.retrieve(
       query_embedding=emb,
       collections=["ghg_protocol_corp"]  # Instead of all collections
   )
   ```

3. **Optimize Weaviate:**
   ```python
   # Use local Weaviate instance for faster queries
   config = RAGConfig(
       weaviate_endpoint="http://localhost:8080"  # Local instead of remote
   )
   ```

4. **Cache frequent queries:**
   ```python
   # Implement query caching
   query_cache = {}
   cache_key = hash(query_text)

   if cache_key in query_cache:
       return query_cache[cache_key]

   result = await query(q=query_text, ...)
   query_cache[cache_key] = result
   ```

**Prevention Tips:**
- Set reasonable fetch_k and top_k values
- Use collection filtering when possible
- Deploy Weaviate close to application
- Implement caching for frequent queries
- Monitor query performance metrics

---

### Memory Issues

**Symptoms:**
- Out of memory errors
- System swap usage high
- Process killed by OS
- Gradual memory increase

**Root Cause:**
- Large embedding batches
- Loading entire vector store in memory
- Memory leaks in long-running processes
- Too many concurrent operations

**Solution Steps:**

1. **Reduce batch sizes:**
   ```python
   config = RAGConfig(
       embedding_batch_size=8,  # Reduce significantly
   )
   ```

2. **Use streaming for large operations:**
   ```python
   # Process documents in batches instead of all at once
   batch_size = 10
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i+batch_size]
       vector_store.add_documents(batch, collection)
   ```

3. **Use Weaviate instead of FAISS:**
   ```python
   # Weaviate stores vectors on disk, not in memory
   config = RAGConfig(
       vector_store_provider="weaviate"
   )
   ```

4. **Monitor memory usage:**
   ```python
   import psutil

   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

**Prevention Tips:**
- Use appropriate batch sizes for your system
- Choose vector store based on dataset size
- Monitor memory usage in production
- Implement cleanup for long-running processes
- Use generators for large datasets

---

### Batch Size Tuning

**Guidance:**

| System RAM | Embedding Batch | Ingestion Batch | Fetch K |
|------------|----------------|-----------------|---------|
| 8 GB       | 8-16           | 25-50          | 10-20   |
| 16 GB      | 16-32          | 50-100         | 20-30   |
| 32 GB      | 32-64          | 100-200        | 30-50   |
| 64+ GB     | 64-128         | 200-500        | 50-100  |

**Test script:**
```python
import time
import psutil

def benchmark_batch_size(batch_sizes):
    for batch_size in batch_sizes:
        start_mem = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()

        # Run ingestion with batch_size
        config = RAGConfig(embedding_batch_size=batch_size)
        # ... ingest documents ...

        end_time = time.time()
        end_mem = psutil.Process().memory_info().rss / 1024 / 1024

        print(f"Batch size {batch_size}:")
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Memory: {end_mem - start_mem:.2f}MB")
```

---

## Configuration Problems

### Invalid Config Values

**Symptoms:**
- Error: "Invalid mode: xyz"
- Error: "Invalid collection name"
- Validation errors on startup

**Root Cause:**
- Typos in configuration
- Invalid enum values
- Out of range values

**Solution Steps:**

1. **Use valid mode values:**
   ```python
   config = RAGConfig(
       mode="replay"  # Must be "replay" or "live"
   )
   ```

2. **Validate collection names:**
   ```python
   # Only alphanumeric, underscore, hyphen allowed
   # Max 64 characters
   collection = "ghg_protocol_corp"  # Valid
   collection = "ghg protocol"        # Invalid (space)
   collection = "ghg@protocol"        # Invalid (@ symbol)
   ```

3. **Check value ranges:**
   ```python
   config = RAGConfig(
       mmr_lambda=0.5,  # Must be 0.0 to 1.0
       chunk_size=512,  # Positive integer
       chunk_overlap=64  # Less than chunk_size
   )
   ```

**Prevention Tips:**
- Use IDE with type checking
- Validate configuration on startup
- Use configuration schemas
- Document valid values

---

### Missing Environment Variables

**Symptoms:**
- Falls back to defaults
- API key not found warnings
- Unexpected behavior

**Root Cause:**
- Environment variables not set
- .env file not loaded
- Typos in variable names

**Solution Steps:**

1. **Create .env file:**
   ```bash
   # .env
   OPENAI_API_KEY=sk-...
   WEAVIATE_ENDPOINT=http://localhost:8080
   GL_MODE=replay
   GL_RAG_ALLOWLIST="ghg_protocol_corp,ipcc_ar6_wg3"
   ```

2. **Load environment variables:**
   ```python
   from dotenv import load_dotenv
   load_dotenv()

   import os
   print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
   ```

3. **Set in code (for testing):**
   ```python
   import os
   os.environ['GL_MODE'] = 'replay'
   os.environ['GL_RAG_ALLOWLIST'] = 'ghg_protocol_corp,ipcc_ar6_wg3'
   ```

**Prevention Tips:**
- Use .env file for development
- Document required environment variables
- Provide defaults where appropriate
- Validate critical variables on startup

---

### Provider Not Found

**Symptoms:**
- Error: "Unknown embedding provider"
- Error: "Unknown vector store provider"
- ImportError for provider modules

**Root Cause:**
- Typo in provider name
- Unsupported provider
- Missing dependencies

**Solution Steps:**

1. **Use valid provider names:**
   ```python
   # Embedding providers
   embedding_provider="minilm"    # Valid
   embedding_provider="openai"    # Valid
   embedding_provider="anthropic" # Not yet supported

   # Vector store providers
   vector_store_provider="faiss"    # Valid
   vector_store_provider="weaviate" # Valid
   vector_store_provider="chromadb" # Not yet implemented
   ```

2. **Install provider dependencies:**
   ```bash
   # For minilm
   pip install sentence-transformers torch

   # For openai
   pip install openai

   # For faiss
   pip install faiss-cpu  # or faiss-gpu

   # For weaviate
   pip install weaviate-client
   ```

3. **Check provider availability:**
   ```python
   try:
       embedder = get_embedding_provider(config)
       print(f"Embedder available: {embedder.name}")
   except ValueError as e:
       print(f"Embedder not available: {e}")
   ```

**Prevention Tips:**
- Check documentation for supported providers
- Install all dependencies
- Use try-except for provider initialization
- Fall back to default providers

---

## Testing Issues

### Test Failures

**Common test failures and solutions:**

#### MMR Diversity Test Fails

**Symptoms:**
- MMR diversity not higher than similarity
- Improvement percentage below 30%

**Fix:**
```python
# Ensure synthetic corpus has clear groups
# Check lambda parameter
config = RAGConfig(
    mmr_lambda=0.5  # Balance between relevance and diversity
)

# Verify embeddings are properly clustered
# Use debug logging to inspect results
```

#### Hash Verification Test Fails

**Symptoms:**
- Chunk IDs don't match expected
- Section hashes inconsistent

**Fix:**
```python
# Ensure deterministic hashing
from greenlang.intelligence.rag.hashing import chunk_uuid5

# Verify inputs are stable
chunk_id = chunk_uuid5(
    doc_id="stable_id",
    section_path="stable_path",
    start_offset=0  # Must be exact
)
```

#### Network Isolation Test Fails

**Symptoms:**
- Network calls detected in replay mode
- Mock assertions fail

**Fix:**
```python
# Ensure all network calls are properly mocked
with patch("urllib.request.urlopen") as mock_urllib, \
     patch("requests.request") as mock_requests, \
     patch("http.client.HTTPConnection") as mock_http:

    # Run test
    # Verify no calls made
    assert not mock_urllib.called
```

---

### Mock Issues

**Symptoms:**
- Mocks not working as expected
- Real calls being made in tests
- Mock side effects incorrect

**Solution Steps:**

1. **Patch at correct level:**
   ```python
   # Patch where imported, not where defined
   # Wrong:
   @patch("sentence_transformers.SentenceTransformer")

   # Correct:
   @patch("greenlang.intelligence.rag.embeddings.SentenceTransformer")
   ```

2. **Use proper mock objects:**
   ```python
   from unittest.mock import MagicMock, AsyncMock

   # For async functions
   mock_embedder = AsyncMock()
   mock_embedder.embed.return_value = [np.zeros(384)]

   # For sync functions
   mock_vector_store = MagicMock()
   mock_vector_store.add_documents.return_value = None
   ```

3. **Verify mock calls:**
   ```python
   mock_func.assert_called_once()
   mock_func.assert_called_with(expected_arg)
   assert mock_func.call_count == 3
   ```

**Prevention Tips:**
- Use pytest fixtures for common mocks
- Verify mocks are applied correctly
- Check mock call counts and arguments
- Use spy pattern when needed

---

### Async Issues in Tests

**Symptoms:**
- RuntimeError: "Event loop is closed"
- Async functions not awaited
- Test hangs indefinitely

**Solution Steps:**

1. **Use pytest-asyncio:**
   ```python
   import pytest

   @pytest.mark.asyncio
   async def test_async_function():
       result = await async_function()
       assert result is not None
   ```

2. **Use asyncio.run for sync tests:**
   ```python
   import asyncio

   def test_sync_wrapper():
       result = asyncio.run(async_function())
       assert result is not None
   ```

3. **Handle event loop properly:**
   ```python
   import asyncio

   # Create new event loop for each test
   loop = asyncio.new_event_loop()
   asyncio.set_event_loop(loop)

   try:
       result = loop.run_until_complete(async_function())
   finally:
       loop.close()
   ```

**Prevention Tips:**
- Use pytest-asyncio plugin
- Mark async tests with @pytest.mark.asyncio
- Don't mix sync and async code incorrectly
- Close event loops properly

---

## Debug Mode

### How to Enable Debug Logging

**Basic logging:**
```python
import logging

# Enable debug logging for RAG system
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or for specific modules
logging.getLogger('greenlang.intelligence.rag').setLevel(logging.DEBUG)
```

**Component-specific logging:**
```python
# Embeddings
logging.getLogger('greenlang.intelligence.rag.embeddings').setLevel(logging.DEBUG)

# Vector stores
logging.getLogger('greenlang.intelligence.rag.vector_stores').setLevel(logging.DEBUG)

# Weaviate client
logging.getLogger('greenlang.intelligence.rag.weaviate_client').setLevel(logging.DEBUG)

# Query engine
logging.getLogger('greenlang.intelligence.rag.engine').setLevel(logging.DEBUG)
```

**Log to file:**
```python
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_debug.log'),
        logging.StreamHandler()
    ]
)
```

---

### How to Inspect Embeddings

**View embedding vectors:**
```python
import numpy as np

embeddings = await embedder.embed(["Sample text"])
emb = embeddings[0]

print(f"Embedding shape: {emb.shape}")
print(f"Embedding dtype: {emb.dtype}")
print(f"First 10 values: {emb[:10]}")
print(f"L2 norm: {np.linalg.norm(emb):.6f}")
print(f"Min value: {emb.min():.6f}")
print(f"Max value: {emb.max():.6f}")
print(f"Mean value: {emb.mean():.6f}")
```

**Compare embeddings:**
```python
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

emb1 = embeddings[0]
emb2 = embeddings[1]

similarity = cosine_similarity(emb1, emb2)
print(f"Cosine similarity: {similarity:.4f}")
```

**Visualize embeddings (2D projection):**
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Embedding Visualization')
plt.show()
```

---

### How to Verify Hashes

**Check file hash:**
```python
from greenlang.intelligence.rag.hashing import file_hash

hash_value = file_hash("documents/report.pdf")
print(f"File hash: {hash_value}")
print(f"Hash length: {len(hash_value)}")  # Should be 64 (SHA-256)
```

**Check chunk hash:**
```python
from greenlang.intelligence.rag.hashing import chunk_uuid5, section_hash

# Chunk UUID (deterministic)
chunk_id = chunk_uuid5(
    doc_id="test_doc",
    section_path="Section_1",
    start_offset=0
)
print(f"Chunk ID: {chunk_id}")

# Section hash
sec_hash = section_hash("Sample text", "Section_1")
print(f"Section hash: {sec_hash}")
```

**Verify hash consistency:**
```python
# Compute hash multiple times
hash1 = file_hash("documents/report.pdf")
hash2 = file_hash("documents/report.pdf")

assert hash1 == hash2, "Hashes should be deterministic"
print(f"Hash consistency verified")
```

---

### How to Check Determinism

**Verify embedding determinism:**
```python
config = RAGConfig(mode="replay")
embedder = get_embedding_provider(config)

# Generate embeddings multiple times
emb1 = await embedder.embed(["Sample text"])
emb2 = await embedder.embed(["Sample text"])

# Compare
diff = np.abs(emb1[0] - emb2[0]).max()
print(f"Max difference: {diff}")

if diff < 1e-6:
    print("Embeddings are deterministic")
else:
    print(f"WARNING: Embeddings differ by {diff}")
```

**Verify query determinism:**
```python
# Run same query multiple times
results1 = await query(q="test query", top_k=5, collections=["test"])
results2 = await query(q="test query", top_k=5, collections=["test"])

# Compare chunk IDs
ids1 = [chunk.chunk_id for chunk in results1.chunks]
ids2 = [chunk.chunk_id for chunk in results2.chunks]

if ids1 == ids2:
    print("Query results are deterministic")
else:
    print("WARNING: Query results differ")
    print(f"First run:  {ids1}")
    print(f"Second run: {ids2}")
```

**Verify vector store determinism:**
```python
# Build vector store twice
vector_store1 = get_vector_store(dimension=384)
vector_store1.add_documents(docs, collection="test")

vector_store2 = get_vector_store(dimension=384)
vector_store2.add_documents(docs, collection="test")

# Compare results
results1 = vector_store1.similarity_search(query_emb, k=5)
results2 = vector_store2.similarity_search(query_emb, k=5)

# Should return same documents in same order
assert len(results1) == len(results2)
for doc1, doc2 in zip(results1, results2):
    assert doc1.chunk.chunk_id == doc2.chunk.chunk_id

print("Vector store operations are deterministic")
```

---

## Quick Reference

### Most Common Issues

1. **Weaviate not running** → Start Weaviate: `docker-compose up -d`
2. **Missing dependencies** → Install: `pip install sentence-transformers faiss-cpu weaviate-client`
3. **Collection not allowed** → Add to allowlist in configuration
4. **Empty vector store** → Run ingestion before querying
5. **Dimension mismatch** → Use consistent embedding model and dimension

### Key Configuration Settings

```python
config = RAGConfig(
    mode="replay",                    # replay or live
    embedding_provider="minilm",      # minilm, openai
    embedding_dimension=384,          # Match model output
    vector_store_provider="faiss",    # faiss or weaviate
    chunk_size=512,                   # Tokens per chunk
    chunk_overlap=64,                 # Overlap in tokens
    embedding_batch_size=32,          # Batch size for embeddings
    default_top_k=6,                  # Results to return
    default_fetch_k=30,               # Candidates for MMR
    mmr_lambda=0.5,                   # Diversity vs relevance
)
```

### Debug Checklist

- [ ] Enable debug logging
- [ ] Check environment variables
- [ ] Verify file paths are correct
- [ ] Confirm Weaviate is running (if used)
- [ ] Validate collection names
- [ ] Check embedding dimensions
- [ ] Verify vector store exists
- [ ] Test with simple queries first
- [ ] Monitor memory usage
- [ ] Check for error logs

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check logs:** Enable debug logging to see detailed error messages
2. **Run tests:** Execute test suite to verify system components
3. **Verify setup:** Review installation and configuration
4. **Isolate issue:** Test individual components separately
5. **Consult docs:** Review RAG_IMPLEMENTATION_GUIDE.md and INTL-104_implementation.md

For additional support, refer to:
- RAG Implementation Guide: `RAG_IMPLEMENTATION_GUIDE.md`
- INTL-104 Specification: `docs/rag/INTL-104_implementation.md`
- Test suite: `tests/rag/`
- Example code: `examples/rag/`
