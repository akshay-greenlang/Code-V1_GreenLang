# GreenLang Agent Registry - Setup Guide

Complete setup guide for the GreenLang Agent Registry infrastructure.

## Overview

The Agent Registry provides:
- **PostgreSQL Database** for agent metadata, versions, and certifications
- **REST API** (FastAPI) for agent operations
- **Python SDK Client** for programmatic access
- **CLI Commands** for agent publishing and management
- **Docker Compose** setup for local development

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agent Registry Stack                     │
├─────────────────────────────────────────────────────────────┤
│  CLI (gl agent)                                              │
│    ↓                                                          │
│  SDK Client (RegistryClient)                                 │
│    ↓                                                          │
│  Registry API (FastAPI + asyncpg)                            │
│    ↓                                                          │
│  PostgreSQL Database                                         │
│  Redis Cache                                                 │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start (Docker Compose)

### 1. Start Services

```bash
# Start all services (PostgreSQL, Redis, Registry API)
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f registry-api

# Check service health
curl http://localhost:8000/health
```

### 2. Access Services

- **Registry API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432 (user: greenlang, password: greenlang)
- **Redis**: localhost:6379
- **pgAdmin**: http://localhost:5050 (admin@greenlang.io / admin)
- **Redis Commander**: http://localhost:8081

### 3. Initialize Database

The database schema is automatically initialized on first startup via Docker entrypoint.

To manually initialize:

```bash
# Connect to PostgreSQL
docker exec -it greenlang-postgres psql -U greenlang -d greenlang_registry

# Run schema
\i /docker-entrypoint-initdb.d/01-schema.sql
```

## Manual Setup (No Docker)

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install fastapi uvicorn asyncpg httpx pydantic

# Install PostgreSQL (macOS)
brew install postgresql
brew services start postgresql

# Install Redis (macOS)
brew install redis
brew services start redis
```

### 2. Create Database

```bash
# Create database
createdb greenlang_registry

# Initialize schema
psql greenlang_registry < core/greenlang/registry/schema.sql
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.registry .env

# Edit configuration
export DB_HOST=localhost
export DB_PORT=5432
export DB_USER=greenlang
export DB_PASSWORD=greenlang
export DB_NAME=greenlang_registry
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### 4. Start Registry API

```bash
# Start API server
uvicorn greenlang.registry.api:app --host 0.0.0.0 --port 8000 --reload

# Or with multiple workers
uvicorn greenlang.registry.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## CLI Usage

### Configure Registry URL

```bash
# Set registry URL
export GL_REGISTRY_URL=http://localhost:8000

# Optional: Set API key for authentication
export GL_REGISTRY_API_KEY=your-api-key
```

### Publish Agent

```bash
# Publish agent from .glpack file
gl agent publish thermosync.glpack \
  --version 1.0.0 \
  --namespace greenlang \
  --author "GreenLang Team" \
  --description "Temperature monitoring agent"

# Publish from directory
gl agent publish ./agents/thermosync/ \
  --version 1.0.1 \
  --namespace greenlang
```

### List Agents

```bash
# List all agents
gl agent list

# Filter by namespace
gl agent list --namespace greenlang

# Search agents
gl agent list --search "temperature"

# JSON output
gl agent list --json
```

### View Agent Details

```bash
# By agent ID
gl agent info 12345678-1234-1234-1234-123456789abc

# By namespace/name
gl agent info greenlang/thermosync

# JSON output
gl agent info greenlang/thermosync --json
```

### Submit Certification

```bash
# Certify agent version
gl agent certify <agent_id> \
  --version 1.0.0 \
  --dimension security \
  --status passed \
  --score 95.0 \
  --certified-by GL-CERT
```

## Python SDK Usage

### Basic Usage

```python
from greenlang_sdk.registry import RegistryClient

# Initialize client
client = RegistryClient(base_url="http://localhost:8000")

# Register new agent
agent = client.register(
    name="thermosync",
    namespace="greenlang",
    description="Temperature monitoring agent",
    author="GreenLang Team"
)
print(f"Agent registered: {agent['id']}")

# Publish version
version = client.publish_version(
    agent_id=agent["id"],
    version="1.0.0",
    pack_path="/path/to/thermosync.glpack"
)
print(f"Version published: {version['version']}")

# List agents
agents = client.list_agents(namespace="greenlang")
for agent in agents["agents"]:
    print(f"- {agent['namespace']}/{agent['name']}")

# Submit certification
cert = client.certify(
    agent_id=agent["id"],
    version="1.0.0",
    dimension="security",
    status="passed",
    score=95.0
)
print(f"Certification submitted: {cert['dimension']} - {cert['status']}")

# Close client
client.close()
```

### Async Usage

```python
from greenlang_sdk.registry import AsyncRegistryClient
import asyncio

async def main():
    async with AsyncRegistryClient(base_url="http://localhost:8000") as client:
        # Register agent
        agent = await client.register(
            name="thermosync",
            namespace="greenlang"
        )

        # Publish version
        version = await client.publish_version(
            agent_id=agent["id"],
            version="1.0.0",
            pack_path="/path/to/pack.glpack"
        )

        print(f"Published {agent['name']} v{version['version']}")

asyncio.run(main())
```

## API Endpoints

### Agent Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents` | Register new agent |
| GET | `/api/v1/agents` | List agents (with filtering) |
| GET | `/api/v1/agents/{id}` | Get agent details |

### Version Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents/{id}/versions` | Publish new version |
| GET | `/api/v1/agents/{id}/versions` | List versions |
| GET | `/api/v1/agents/{id}/versions/{version}` | Get version details |

### Certification Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/agents/{id}/certify` | Submit certification |
| GET | `/api/v1/agents/{id}/certifications` | List certifications |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/ready` | Readiness check |

## Database Schema

### Tables

- **agents**: Core agent metadata
- **agent_versions**: Version history and packs
- **agent_certifications**: GL-CERT certifications
- **agent_tags**: Searchable tags
- **agent_downloads**: Usage analytics

### Views

- **agent_latest_versions**: Latest version per agent
- **agent_summary**: Agent summary with stats

See `core/greenlang/registry/schema.sql` for complete schema.

## Monitoring & Observability

### Prometheus Metrics

```python
# Custom metrics endpoint (to be implemented)
# GET /metrics
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Readiness (includes database check)
curl http://localhost:8000/api/v1/ready
```

### Logs

```bash
# Docker Compose logs
docker-compose -f docker-compose.dev.yml logs -f registry-api

# Filter by severity
docker-compose -f docker-compose.dev.yml logs registry-api | grep ERROR
```

## Production Deployment

### Environment Variables

Production configuration:

```bash
# Database
DB_HOST=prod-db.example.com
DB_PORT=5432
DB_USER=greenlang_prod
DB_PASSWORD=<secure-password>
DB_NAME=greenlang_registry

# Redis
REDIS_HOST=prod-redis.example.com
REDIS_PORT=6379

# API
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=warning

# Security
API_KEY=<secure-api-key>
JWT_SECRET_KEY=<secure-jwt-key>
```

### Kubernetes Deployment

See `docs/kubernetes/registry-deployment.yaml` for Kubernetes manifests.

### Security Considerations

1. **API Authentication**: Enable API key authentication
2. **Database Encryption**: Use SSL/TLS for database connections
3. **Secrets Management**: Use AWS Secrets Manager or HashiCorp Vault
4. **Network Security**: Deploy in private VPC with security groups
5. **Rate Limiting**: Implement rate limiting on API endpoints

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Check connection
docker exec -it greenlang-postgres psql -U greenlang -d greenlang_registry

# View logs
docker logs greenlang-postgres
```

### API Not Starting

```bash
# Check dependencies
pip list | grep -E "fastapi|uvicorn|asyncpg"

# Check environment variables
env | grep -E "DB_|REDIS_"

# View API logs
docker logs greenlang-registry-api
```

### CLI Connection Errors

```bash
# Verify registry URL
echo $GL_REGISTRY_URL

# Test connection
curl http://localhost:8000/health

# Enable debug mode
GL_REGISTRY_URL=http://localhost:8000 gl agent list --help
```

## Development

### Running Tests

```bash
# Unit tests
pytest core/greenlang/registry/tests/

# Integration tests
pytest core/greenlang/registry/tests/test_integration.py

# With coverage
pytest --cov=core/greenlang/registry --cov-report=html
```

### Database Migrations

```bash
# Create migration (future: Alembic)
alembic revision --autogenerate -m "Add new column"

# Apply migration
alembic upgrade head
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/greenlang/core/issues
- Documentation: https://docs.greenlang.io
- Email: support@greenlang.io
