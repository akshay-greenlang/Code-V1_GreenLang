# Weaviate Setup for GreenLang RAG

This directory contains the Docker Compose configuration for running Weaviate as the vector database for GreenLang's RAG (Retrieval-Augmented Generation) system.

## Quick Start

### 1. Start Weaviate

```bash
# From project root
docker-compose -f docker/weaviate/docker-compose.yml up -d

# Or use the CLI
gl rag up
```

### 2. Verify Health

```bash
# Check container status
docker ps | grep weaviate

# Health check
curl http://localhost:8080/v1/.well-known/ready

# Expected response:
# HTTP 200 OK
```

### 3. Stop Weaviate

```bash
docker-compose -f docker/weaviate/docker-compose.yml down

# To remove data volumes
docker-compose -f docker/weaviate/docker-compose.yml down -v
```

## Configuration

### Environment Variables

Create a `.env` file from the example:

```bash
cp docker/weaviate/.env.example docker/weaviate/.env
```

Key settings:
- `WEAVIATE_ENDPOINT`: Weaviate URL (default: http://localhost:8080)
- `WEAVIATE_API_KEY`: API key for authentication (empty for dev)
- `WEAVIATE_BATCH_SIZE`: Batch size for ingestion (default: 100)

### Resource Limits

Default limits (suitable for dev/CI):
- **CPU**: 2 cores (GOMAXPROCS=2)
- **Memory**: 2 GiB (GOMEMLIMIT=2GiB)

For production, adjust in `docker-compose.yml`:
```yaml
environment:
  GOMAXPROCS: '4'
  GOMEMLIMIT: '8GiB'
```

## Data Persistence

Weaviate data is persisted in `./weaviate_data/` directory:

```
docker/weaviate/
  weaviate_data/
    meta.json
    index/
    wal/
```

**⚠️ Important**: This directory is excluded from git (see `.gitignore`).

## Schema Management

Weaviate schemas are auto-created by GreenLang on first use. See `greenlang/rag/weaviate_client.py` for schema definitions.

### Chunk Class Schema

```python
{
  "class": "Chunk",
  "vectorizer": "none",  # We supply embeddings
  "properties": [
    {"name": "chunk_id", "dataType": ["string"]},
    {"name": "collection", "dataType": ["string"]},
    {"name": "doc_id", "dataType": ["string"]},
    {"name": "title", "dataType": ["string"]},
    {"name": "section_path", "dataType": ["string"]},
    {"name": "section_hash", "dataType": ["string"]},
    {"name": "page_start", "dataType": ["int"]},
    {"name": "page_end", "dataType": ["int"]},
    {"name": "text", "dataType": ["text"]},
    # ... more fields
  ]
}
```

## Troubleshooting

### Weaviate won't start

**Check logs**:
```bash
docker-compose -f docker/weaviate/docker-compose.yml logs weaviate
```

**Common issues**:
- Port 8080 already in use: Change `ports` in docker-compose.yml
- Insufficient memory: Increase `GOMEMLIMIT`
- Data corruption: Remove volumes and restart

### Connection refused

**Verify network**:
```bash
# Check if Weaviate is listening
netstat -an | grep 8080

# Test connectivity
curl -v http://localhost:8080/v1/.well-known/ready
```

### Schema errors

**Reset schema**:
```python
from greenlang.rag.weaviate_client import WeaviateClient

client = WeaviateClient()
client.delete_schema()  # ⚠️ Deletes all data!
client.ensure_schema()  # Recreate
```

## Performance Tuning

### For Development (Default)
- 1-2 vCPU, 2-4 GB RAM
- Single node, no sharding
- Suitable for <100k chunks

### For Production
```yaml
environment:
  GOMAXPROCS: '8'
  GOMEMLIMIT: '16GiB'
  PERSISTENCE_LSM_CACHE_SIZE: '4GB'
```

### Batch Optimization

For large ingestions:
```python
config = RAGConfig(
    weaviate_batch_size=200,  # Increase batch size
    weaviate_batch_dynamic=True  # Auto-adjust
)
```

## Monitoring

### Metrics Endpoint

```bash
# Prometheus metrics
curl http://localhost:8080/metrics
```

### Query Performance

```bash
# Get schema statistics
curl http://localhost:8080/v1/schema/Chunk

# Check shard status
curl http://localhost:8080/v1/schema/Chunk/shards
```

## Security

### Production Deployment

**Enable API Key Authentication**:

```yaml
environment:
  AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'false'
  AUTHENTICATION_APIKEY_ENABLED: 'true'
  AUTHENTICATION_APIKEY_ALLOWED_KEYS: 'your-secret-key-here'
```

**Set API key in GreenLang**:
```bash
export WEAVIATE_API_KEY=your-secret-key-here
```

### Network Security

For production, bind to localhost only:
```yaml
ports:
  - "127.0.0.1:8080:8080"  # Only accessible locally
```

Use a reverse proxy (nginx, Caddy) for external access with TLS.

## Backup & Restore

### Backup

```bash
# Stop Weaviate
docker-compose -f docker/weaviate/docker-compose.yml stop

# Backup data directory
tar -czf weaviate_backup_$(date +%Y%m%d).tar.gz docker/weaviate/weaviate_data/

# Restart
docker-compose -f docker/weaviate/docker-compose.yml start
```

### Restore

```bash
# Stop Weaviate
docker-compose -f docker/weaviate/docker-compose.yml down

# Remove old data
rm -rf docker/weaviate/weaviate_data/

# Extract backup
tar -xzf weaviate_backup_20251003.tar.gz

# Start Weaviate
docker-compose -f docker/weaviate/docker-compose.yml up -d
```

## References

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Docker Compose Reference](https://weaviate.io/developers/weaviate/installation/docker-compose)
- [Schema Configuration](https://weaviate.io/developers/weaviate/config-refs/schema)
- [Performance Tuning](https://weaviate.io/developers/weaviate/configuration/performance)
