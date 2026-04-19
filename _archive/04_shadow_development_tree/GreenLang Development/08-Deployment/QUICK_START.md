# GreenLang Platform - Quick Start Guide

**5-Minute Deployment Guide**

---

## TL;DR - Just Get It Running

```bash
cd deployment
cp .env.example .env
# Edit .env with your API keys
./deploy.sh start
```

That's it! Platform will be running at:
- CBAM: http://localhost:8001
- CSRD: http://localhost:8002
- VCCI: http://localhost:8000

---

## What Gets Deployed?

```
┌─────────────────────────────────────────┐
│     GreenLang Unified Platform          │
├─────────────────────────────────────────┤
│                                         │
│  🔧 Shared Infrastructure               │
│  ├─ PostgreSQL (single database)       │
│  ├─ Redis (shared cache)                │
│  ├─ RabbitMQ (message queue)            │
│  └─ Weaviate (vector DB)                │
│                                         │
│  📱 Applications                         │
│  ├─ CBAM API (port 8001)                │
│  ├─ CSRD Web (port 8002)                │
│  └─ VCCI Backend (port 8000)            │
│                                         │
│  📊 Monitoring                           │
│  ├─ Prometheus (port 9090)              │
│  └─ Grafana (port 3000)                 │
│                                         │
└─────────────────────────────────────────┘
```

---

## Prerequisites

- Docker + Docker Compose
- 16 GB RAM minimum
- Ports available: 8000-8002, 3000, 5432, 6379, 5672, 8080, 9090
- API Keys: Anthropic + OpenAI

---

## Step 1: Configure

```bash
cd deployment
cp .env.example .env
```

**Edit .env and set:**
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
SHARED_JWT_SECRET=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -hex 16)
```

---

## Step 2: Deploy

```bash
./deploy.sh start
```

Wait ~2 minutes for all services to start.

---

## Step 3: Verify

```bash
./deploy.sh health
```

Should show all services healthy:
- ✓ CBAM API is healthy
- ✓ CSRD Web is healthy
- ✓ VCCI Backend is healthy
- ✓ Prometheus is healthy
- ✓ Grafana is healthy

---

## Step 4: Access

Open in browser:
- **Grafana Dashboard:** http://localhost:3000
  - Login: admin / greenlang2024
  - View unified platform metrics

- **API Documentation:**
  - CBAM: http://localhost:8001/docs
  - CSRD: http://localhost:8002/docs
  - VCCI: http://localhost:8000/docs

---

## Common Commands

```bash
# View logs
./deploy.sh logs

# Check status
./deploy.sh status

# Stop platform
./deploy.sh stop

# Restart
./deploy.sh restart

# Run tests
python validate_integration.py
```

---

## Default Credentials

**Admin User:**
- Email: admin@greenlang.com
- Password: admin123
- Access: All three apps

**Grafana:**
- User: admin
- Password: greenlang2024

**RabbitMQ UI:** http://localhost:15672
- User: greenlang
- Password: greenlang_rabbit_2024

---

## Troubleshooting

**Services won't start?**
```bash
# Check if ports are free
./deploy.sh status

# Check logs for errors
./deploy.sh logs
```

**Database connection errors?**
```bash
# Verify PostgreSQL is running
docker-compose -f docker-compose-unified.yml exec postgres pg_isready
```

**Out of memory?**
```bash
# Check system resources
docker stats
```

---

## What's Next?

1. Read full guide: `DEPLOYMENT_GUIDE.md`
2. Review architecture: `PLATFORM_INTEGRATION_DEPLOYMENT_REPORT.md`
3. Test integration: `python validate_integration.py`
4. Configure monitoring alerts
5. Set up backups

---

## Architecture at a Glance

**Shared Database:**
```sql
greenlang_platform
├─ public (shared: users, orgs, roles)
├─ cbam (CBAM tables)
├─ csrd (CSRD tables)
└─ vcci (VCCI tables)
```

**Network:**
```
172.26.0.0/16 (greenlang-platform)
├─ Infrastructure: 172.26.0.10-19
├─ Applications:   172.26.0.100-119
└─ Monitoring:     172.26.0.200-219
```

**Integration:**
- Shared JWT authentication
- Cross-app message queue (RabbitMQ)
- REST API integration
- Unified monitoring

---

## Key Features

✓ **One-Command Deployment** - `./deploy.sh start`
✓ **Shared Infrastructure** - Single DB, cache, queue
✓ **Unified Auth** - One token works on all apps
✓ **Cross-App Events** - RabbitMQ message queue
✓ **Unified Monitoring** - Prometheus + Grafana
✓ **Health Checks** - Auto-healing containers
✓ **Production Ready** - Optimized configuration

---

## File Structure

```
deployment/
├─ docker-compose-unified.yml   # Main deployment file
├─ .env.example                  # Configuration template
├─ deploy.sh                     # Deployment script
├─ validate_integration.py       # Integration tests
├─ DEPLOYMENT_GUIDE.md          # Full documentation
├─ init/
│  └─ shared_db_schema.sql      # Database schema
└─ monitoring/
   ├─ prometheus-unified.yml    # Metrics config
   ├─ alerts-unified.yml        # Alert rules
   └─ grafana-provisioning/     # Dashboard config
```

---

## Need Help?

**Full Documentation:** `DEPLOYMENT_GUIDE.md`
**Integration Report:** `PLATFORM_INTEGRATION_DEPLOYMENT_REPORT.md`
**Logs:** `./deploy.sh logs`
**Health Status:** `./deploy.sh health`

---

**Happy Deploying!**
