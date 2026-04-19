# GreenLang Platform - Unified Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08
**Status:** Ready for Deployment

---

## Overview

This guide provides step-by-step instructions for deploying the entire GreenLang platform as a unified system with all three applications (CBAM, CSRD, VCCI) sharing infrastructure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GreenLang Platform                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Applications:                                               │
│  ├─ CBAM API (Port 8001)                                    │
│  ├─ CSRD Web (Port 8002)                                    │
│  └─ VCCI Backend (Port 8000) + Worker                       │
│                                                             │
│  Shared Infrastructure:                                      │
│  ├─ PostgreSQL (Port 5432) - Shared DB with app schemas    │
│  ├─ Redis (Port 6379) - Shared cache                        │
│  ├─ RabbitMQ (Port 5672/15672) - Message queue             │
│  └─ Weaviate (Port 8080) - Vector database                  │
│                                                             │
│  Monitoring:                                                 │
│  ├─ Prometheus (Port 9090)                                  │
│  ├─ Grafana (Port 3000)                                     │
│  └─ pgAdmin (Port 5050)                                     │
│                                                             │
│  Network: greenlang-platform (172.26.0.0/16)                │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

- **OS:** Linux, macOS, or Windows with WSL2
- **RAM:** Minimum 16 GB (32 GB recommended)
- **Disk:** 50 GB free space
- **CPU:** 4+ cores

### Software Requirements

- Docker Engine 24.0+
- Docker Compose 2.20+
- Python 3.10+ (for validation scripts)
- Git

### Required API Keys

- Anthropic API Key (for CSRD and VCCI)
- OpenAI API Key (for CSRD and VCCI)
- Pinecone API Key (optional, for CSRD)

---

## Pre-Deployment Checklist

- [ ] Docker and Docker Compose installed
- [ ] API keys obtained
- [ ] Firewall rules configured (if applicable)
- [ ] Sufficient system resources available
- [ ] Network ports available (8000-8002, 3000, 5432, 6379, 5672, 8080, 9090, 15672)

---

## Deployment Steps

### Step 1: Clone and Navigate

```bash
cd /path/to/Code-V1_GreenLang/deployment
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your values
nano .env  # or vim, code, etc.
```

**Critical values to update:**
- `POSTGRES_PASSWORD`
- `REDIS_PASSWORD`
- `RABBITMQ_PASSWORD`
- `SHARED_SECRET_KEY` (min 32 characters)
- `SHARED_JWT_SECRET` (min 32 characters)
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`

**Generate secure secrets:**
```bash
# Generate random secret keys
openssl rand -hex 32  # For SHARED_SECRET_KEY
openssl rand -hex 32  # For SHARED_JWT_SECRET
```

### Step 3: Verify Configuration Files

```bash
# Verify docker-compose file
docker-compose -f docker-compose-unified.yml config

# Should show parsed configuration without errors
```

### Step 4: Build Images (Optional but Recommended)

```bash
# Build CBAM
docker build -t greenlang/cbam-app:latest ../GL-CBAM-APP/CBAM-Importer-Copilot/

# Build CSRD
docker build -t greenlang/csrd-app:latest ../GL-CSRD-APP/CSRD-Reporting-Platform/

# Build VCCI Backend
docker build -t greenlang/vcci-backend:latest -f ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/backend/Dockerfile ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/

# Build VCCI Worker
docker build -t greenlang/vcci-worker:latest -f ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/worker/Dockerfile ../GL-VCCI-Carbon-APP/VCCI-Scope3-Platform/
```

### Step 5: Start Infrastructure First

```bash
# Start shared infrastructure only
docker-compose -f docker-compose-unified.yml up -d postgres redis rabbitmq weaviate

# Wait for services to be healthy (about 30 seconds)
docker-compose -f docker-compose-unified.yml ps
```

**Verify infrastructure:**
```bash
# Check PostgreSQL
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "\dn"

# Should show schemas: cbam, csrd, vcci, shared, public
```

### Step 6: Start Applications

```bash
# Start all three applications
docker-compose -f docker-compose-unified.yml up -d cbam-api csrd-web vcci-backend vcci-worker

# Watch logs
docker-compose -f docker-compose-unified.yml logs -f
```

### Step 7: Start Monitoring Stack

```bash
# Start Prometheus and Grafana
docker-compose -f docker-compose-unified.yml up -d prometheus grafana

# Optional: Start pgAdmin for database management
docker-compose -f docker-compose-unified.yml --profile admin up -d pgadmin
```

### Step 8: Verify Deployment

```bash
# Check all services are running
docker-compose -f docker-compose-unified.yml ps

# All services should show "healthy" or "running"
```

**Health Check URLs:**
- CBAM API: http://localhost:8001/health
- CSRD Web: http://localhost:8002/health
- VCCI Backend: http://localhost:8000/health/live
- Prometheus: http://localhost:9090/-/healthy
- Grafana: http://localhost:3000/api/health

```bash
# Quick health check
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8000/health/live
```

### Step 9: Run Integration Tests

```bash
# Install Python dependencies for validation script
pip install requests psycopg2-binary redis pika

# Run validation
python validate_integration.py

# Should show all tests passing
```

---

## Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| CBAM API | http://localhost:8001 | N/A (API endpoints) |
| CBAM API Docs | http://localhost:8001/docs | N/A |
| CSRD Web | http://localhost:8002 | N/A (API endpoints) |
| CSRD API Docs | http://localhost:8002/docs | N/A |
| VCCI Backend | http://localhost:8000 | N/A (API endpoints) |
| VCCI API Docs | http://localhost:8000/docs | N/A |
| Grafana | http://localhost:3000 | admin / greenlang2024 |
| Prometheus | http://localhost:9090 | N/A |
| RabbitMQ UI | http://localhost:15672 | greenlang / greenlang_rabbit_2024 |
| pgAdmin | http://localhost:5050 | admin@greenlang.com / greenlang_admin_2024 |

---

## Database Access

### Via psql (from host)

```bash
psql -h localhost -p 5432 -U greenlang_admin -d greenlang_platform
```

### Via Docker

```bash
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform
```

### Schema Access

Each application has its own schema:

```sql
-- CBAM tables
SET search_path TO cbam;
\dt

-- CSRD tables
SET search_path TO csrd;
\dt

-- VCCI tables
SET search_path TO vcci;
\dt

-- Shared tables
SET search_path TO public;
\dt

-- Integration views
SET search_path TO shared;
\dv
```

---

## Cross-App Integration

### Shared Authentication

All apps use the same JWT secret. A token generated by one app is valid on all apps.

**Create user:**
```bash
# Via database
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
INSERT INTO users (email, password_hash, first_name, last_name, org_id)
VALUES ('user@example.com', '\$2b\$12\$...', 'Test', 'User', '00000000-0000-0000-0000-000000000001');
"
```

### Message Queue Integration

Apps communicate via RabbitMQ for async events:

**Example: VCCI publishes emissions calculation**
```python
# In VCCI app
import pika

connection = pika.BlockingConnection(...)
channel = connection.channel()
channel.basic_publish(
    exchange='',
    routing_key='emissions.calculated',
    body=json.dumps({
        'org_id': '...',
        'total_emissions': 12345.67
    })
)
```

**CSRD consumes:**
```python
# In CSRD app
def callback(ch, method, properties, body):
    data = json.loads(body)
    # Store emissions data for reporting
    ...

channel.basic_consume(
    queue='emissions.calculated',
    on_message_callback=callback
)
```

---

## Monitoring

### Prometheus Targets

View all scraped targets:
```
http://localhost:9090/targets
```

Expected targets:
- cbam-app
- csrd-app
- vcci-backend
- postgres
- redis
- rabbitmq
- weaviate

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin / greenlang2024
3. Navigate to Dashboards
4. View "GreenLang Platform - Unified Dashboard"

### Metrics Examples

```promql
# Total HTTP requests across all apps
sum(rate(http_requests_total[5m])) by (app)

# Database connections
sum(pg_stat_database_numbackends) by (datname)

# Redis memory usage
redis_memory_used_bytes

# RabbitMQ queue depth
rabbitmq_queue_messages
```

---

## Troubleshooting

### Services Won't Start

```bash
# Check logs
docker-compose -f docker-compose-unified.yml logs <service-name>

# Common issues:
# - Port already in use
# - Insufficient memory
# - Missing environment variables
```

### Database Connection Errors

```bash
# Verify PostgreSQL is running
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin

# Check connection string in app logs
docker-compose -f docker-compose-unified.yml logs cbam-api | grep DATABASE_URL
```

### Network Issues

```bash
# Verify network
docker network inspect greenlang-platform

# Ensure all containers are on same network
docker-compose -f docker-compose-unified.yml ps --format json | jq -r '.[].Networks'
```

### Reset Everything (CAUTION)

```bash
# Stop all services
docker-compose -f docker-compose-unified.yml down

# Remove volumes (deletes all data!)
docker-compose -f docker-compose-unified.yml down -v

# Restart fresh
docker-compose -f docker-compose-unified.yml up -d
```

---

## Performance Tuning

### PostgreSQL

Edit `docker-compose-unified.yml` PostgreSQL command section:

```yaml
command:
  - "-c"
  - "max_connections=500"  # Increase if needed
  - "-c"
  - "shared_buffers=1GB"   # Increase for more RAM
```

### Redis

```yaml
command: >
  redis-server
  --maxmemory 2gb  # Increase cache size
```

---

## Security Checklist

- [ ] Change all default passwords in `.env`
- [ ] Use strong, random secrets (min 32 characters)
- [ ] Enable HTTPS in production (use nginx reverse proxy)
- [ ] Restrict database access to application network only
- [ ] Enable Redis authentication
- [ ] Use firewall rules to restrict external access
- [ ] Regularly update Docker images
- [ ] Monitor logs for suspicious activity

---

## Backup and Recovery

### Database Backup

```bash
# Backup entire database
docker-compose -f docker-compose-unified.yml exec -T postgres pg_dump -U greenlang_admin greenlang_platform > backup_$(date +%Y%m%d).sql

# Backup specific schema
docker-compose -f docker-compose-unified.yml exec -T postgres pg_dump -U greenlang_admin -n cbam greenlang_platform > backup_cbam_$(date +%Y%m%d).sql
```

### Restore Database

```bash
# Restore from backup
cat backup_20250108.sql | docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin greenlang_platform
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)

```bash
# Scale CBAM API to 3 instances
docker-compose -f docker-compose-unified.yml up -d --scale cbam-api=3

# Add load balancer (nginx, traefik, etc.)
```

### Vertical Scaling (More Resources)

Edit docker-compose to add resource limits:

```yaml
cbam-api:
  deploy:
    resources:
      limits:
        cpus: '2'
        memory: 4G
      reservations:
        cpus: '1'
        memory: 2G
```

---

## Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild image
docker build -t greenlang/cbam-app:latest ../GL-CBAM-APP/CBAM-Importer-Copilot/

# Restart service
docker-compose -f docker-compose-unified.yml up -d cbam-api
```

### View Logs

```bash
# All services
docker-compose -f docker-compose-unified.yml logs -f

# Specific service
docker-compose -f docker-compose-unified.yml logs -f cbam-api

# Last 100 lines
docker-compose -f docker-compose-unified.yml logs --tail=100 vcci-backend
```

### Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune
```

---

## Production Deployment

For production, consider:

1. **Use Kubernetes** instead of Docker Compose
   - Refer to `k8s/` directories in each app
   - Use provided Helm charts

2. **External Database** - Use managed PostgreSQL (AWS RDS, GCP Cloud SQL)

3. **External Cache** - Use managed Redis (AWS ElastiCache, Redis Cloud)

4. **Load Balancer** - Use cloud load balancer or nginx

5. **SSL/TLS** - Configure HTTPS with Let's Encrypt

6. **Secrets Management** - Use Vault, AWS Secrets Manager, etc.

7. **Monitoring** - Send metrics to Datadog, New Relic, etc.

---

## Support

For issues or questions:
- Review logs: `docker-compose logs`
- Check documentation in each app's directory
- Run validation script: `python validate_integration.py`

---

## Next Steps

After successful deployment:

1. [ ] Create test organization and users
2. [ ] Test each application individually
3. [ ] Test cross-app integration (VCCI → CSRD data flow)
4. [ ] Configure monitoring alerts
5. [ ] Set up automated backups
6. [ ] Review security settings
7. [ ] Load test the platform
8. [ ] Document custom configurations

---

**End of Deployment Guide**
