# GL-CSRD-APP Production Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-08
**Status:** Production Ready ✅

---

## 🚀 Quick Start

Choose your deployment method based on your infrastructure:

| Method | Best For | Time | Complexity |
|--------|----------|------|------------|
| [Docker Compose](#docker-compose-deployment) | Development, Small Production | 5 min | ⭐ Easy |
| [Kubernetes](#kubernetes-deployment) | Enterprise Production | 15 min | ⭐⭐⭐ Advanced |
| [Manual Installation](#manual-installation) | Custom Deployments | 30 min | ⭐⭐ Moderate |

---

## 📋 Prerequisites

### All Deployments
- **OS**: Linux (Ubuntu 22.04+), macOS (12+), Windows Server 2019+
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 50GB free disk space
- **Network**: Internet access for pulling dependencies

### Docker-Specific
- Docker Engine 20.10+
- Docker Compose 2.0+

### Kubernetes-Specific
- Kubernetes cluster 1.24+
- kubectl configured
- Helm 3.0+ (optional but recommended)
- Ingress controller (nginx, traefik)
- cert-manager for TLS (optional)

### Required Credentials
- Anthropic API key (Claude)
- OpenAI API key (optional)
- Pinecone API key (optional)
- Database credentials
- Encryption keys

---

## 🐳 Docker Compose Deployment

### Step 1: Clone Repository

```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform
```

### Step 2: Configure Environment

```bash
# Copy production environment template
cp .env.production.example .env.production

# Edit with your actual values
nano .env.production
```

**Required Environment Variables:**

```bash
# Database
DATABASE_URL=postgresql://csrd_user:YOUR_STRONG_PASSWORD@db:5432/csrd_db

# Redis
REDIS_URL=redis://redis:6379/0

# AI/LLM API Keys
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
OPENAI_API_KEY=sk-YOUR_KEY_HERE

# Security (generate these!)
SECRET_KEY=$(openssl rand -base64 32)
CSRD_ENCRYPTION_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')

# Application
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKERS=4
```

### Step 3: Generate Security Keys

```bash
# Generate secret key for JWT/sessions
openssl rand -base64 32

# Generate encryption key for sensitive data
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add these to your `.env.production` file.

### Step 4: Start Services

```bash
# Start all services (web, database, redis, weaviate)
docker-compose --env-file .env.production up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f web
```

### Step 5: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0","timestamp":"...","uptime_seconds":...}

# Readiness check
curl http://localhost:8000/ready

# API documentation
open http://localhost:8000/docs
```

### Step 6: Access Services

| Service | URL | Default Credentials |
|---------|-----|---------------------|
| API | http://localhost:8000 | - |
| API Docs | http://localhost:8000/docs | - |
| pgAdmin | http://localhost:5050 | admin@csrd.local / admin |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | - |

### Step 7: Run Initial Tests

```bash
# Execute quick start example
docker-compose exec web python examples/quick_start.py

# Run test suite (975 tests!)
docker-compose exec web pytest tests/ -v --tb=short

# Run specific test categories
docker-compose exec web pytest tests/ -m unit -v
docker-compose exec web pytest tests/ -m integration -v
```

### Docker Compose Profiles

```bash
# Minimal deployment (API + DB + Redis only)
docker-compose up -d

# With pgAdmin for database management
docker-compose --profile admin up -d

# With monitoring (Prometheus + Grafana)
docker-compose --profile monitoring up -d

# Full production stack
docker-compose --profile production up -d
```

### Maintenance Commands

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: Data loss)
docker-compose down -v

# Update to latest version
git pull
docker-compose build --no-cache
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Scale API servers
docker-compose up -d --scale web=3

# Backup database
docker-compose exec db pg_dump -U csrd_user csrd_db > backup_$(date +%Y%m%d).sql

# Restore database
cat backup_20250108.sql | docker-compose exec -T db psql -U csrd_user csrd_db

# Backup volumes
docker run --rm -v csrd-postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

---

## ☸️ Kubernetes Deployment

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Ingress Controller                    │
│                  (TLS Termination)                       │
└─────────────────┬───────────────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼──────┐
│  CSRD API      │  │   Grafana   │
│  (3-20 pods)   │  │  Monitoring │
│  Auto-scaling  │  └─────────────┘
└───────┬────────┘
        │
┌───────┴────────────────────┐
│                            │
▼                            ▼
┌──────────────┐  ┌─────────────────┐
│  PostgreSQL  │  │  Redis + Weaviate│
│ (StatefulSet)│  │   (StatefulSets) │
└──────────────┘  └─────────────────┘
```

### Step 1: Prepare Kubernetes Cluster

```bash
# Verify kubectl access
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl create namespace production
kubectl config set-context --current --namespace=production

# Label nodes (optional, for node affinity)
kubectl label nodes node-1 workload=csrd
kubectl label nodes node-1 tier=api
```

### Step 2: Create Secrets

```bash
# Method 1: From environment variables
kubectl create secret generic csrd-secrets \
  --from-literal=database-url="postgresql://csrd_user:PASSWORD@csrd-postgresql:5432/csrd_db" \
  --from-literal=redis-url="redis://csrd-redis:6379/0" \
  --from-literal=weaviate-url="http://csrd-weaviate:8080" \
  --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  --from-literal=encryption-key="$CSRD_ENCRYPTION_KEY" \
  --from-literal=secret-key="$SECRET_KEY" \
  --namespace=production

# Method 2: From file
cp deployment/k8s/secrets.yaml.example deployment/k8s/secrets.yaml
# Edit secrets.yaml with actual values
kubectl apply -f deployment/k8s/secrets.yaml

# Verify secrets
kubectl get secrets -n production
```

### Step 3: Deploy Infrastructure (Databases)

```bash
# Deploy PostgreSQL
kubectl apply -f deployment/k8s/statefulset.yaml

# Wait for PostgreSQL to be ready
kubectl wait --for=condition=ready pod -l component=database --timeout=300s -n production

# Deploy Redis
kubectl rollout status statefulset/csrd-redis -n production

# Deploy Weaviate
kubectl rollout status statefulset/csrd-weaviate -n production

# Verify all databases are running
kubectl get statefulsets -n production
kubectl get pods -n production
```

### Step 4: Deploy Application

```bash
# Apply ConfigMaps
kubectl apply -f deployment/k8s/configmap.yaml

# Apply Services
kubectl apply -f deployment/k8s/service.yaml

# Deploy application
kubectl apply -f deployment/k8s/deployment.yaml

# Wait for deployment
kubectl rollout status deployment/csrd-app -n production
```

### Step 5: Configure Auto-Scaling

```bash
# Deploy Horizontal Pod Autoscaler
kubectl apply -f deployment/k8s/hpa.yaml

# Verify HPA
kubectl get hpa csrd-hpa -n production

# Expected output:
# NAME       REFERENCE             TARGETS         MINPODS   MAXPODS   REPLICAS
# csrd-hpa   Deployment/csrd-app   15%/70%, 20%/80%   3         20        3
```

### Step 6: Configure Ingress (HTTPS)

```bash
# Option 1: Manual TLS certificate
kubectl create secret tls csrd-tls \
  --cert=path/to/tls.crt \
  --key=path/to/tls.key \
  --namespace=production

# Option 2: cert-manager (recommended)
# Install cert-manager first
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Create ClusterIssuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@yourdomain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Deploy Ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Get Ingress IP
kubectl get ingress csrd-ingress -n production
```

### Step 7: Verify Production Deployment

```bash
# Check all resources
kubectl get all -n production

# Check pod logs
kubectl logs -f deployment/csrd-app -n production

# Check events
kubectl get events --sort-by='.lastTimestamp' -n production

# Port forward for local testing
kubectl port-forward svc/csrd-service 8000:80 -n production

# Test health endpoint
curl http://localhost:8000/health

# Test via ingress (replace with your domain)
curl https://csrd.yourdomain.com/health
```

### Kubernetes Monitoring

```bash
# View pod resource usage
kubectl top pods -n production

# View node resource usage
kubectl top nodes

# View HPA status
kubectl get hpa -n production -w

# View pod events
kubectl describe pod <pod-name> -n production

# Exec into pod for debugging
kubectl exec -it <pod-name> -n production -- /bin/bash

# View application logs
kubectl logs -f deployment/csrd-app -n production --tail=100
```

### Kubernetes Maintenance

```bash
# Update application image
kubectl set image deployment/csrd-app \
  csrd-app=greenlang/csrd-app:v1.1.0 \
  -n production

# Rollback deployment
kubectl rollout undo deployment/csrd-app -n production

# Scale manually (override HPA)
kubectl scale deployment csrd-app --replicas=5 -n production

# Restart pods (rolling restart)
kubectl rollout restart deployment/csrd-app -n production

# Delete and redeploy
kubectl delete -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/deployment.yaml

# Drain node for maintenance
kubectl drain node-1 --ignore-daemonsets --delete-emptydir-data
# ... perform maintenance ...
kubectl uncordon node-1
```

---

## 🔧 Manual Installation

### Step 1: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y \
  python3.11 python3.11-venv python3.11-dev \
  postgresql-15 postgresql-contrib \
  redis-server \
  build-essential libpq-dev \
  git curl
```

**macOS:**
```bash
brew install python@3.11 postgresql@15 redis
brew services start postgresql@15
brew services start redis
```

**Windows:**
- Install Python 3.11 from python.org
- Install PostgreSQL 15 from postgresql.org
- Install Redis from github.com/microsoftarchive/redis

### Step 2: Create Database

```bash
# Start PostgreSQL (if not running)
sudo systemctl start postgresql  # Linux
brew services start postgresql@15  # macOS

# Create database and user
sudo -u postgres psql
```

```sql
-- In PostgreSQL shell
CREATE USER csrd_user WITH PASSWORD 'your_secure_password_here';
CREATE DATABASE csrd_db OWNER csrd_user;
GRANT ALL PRIVILEGES ON DATABASE csrd_db TO csrd_user;

-- Enable required extensions
\c csrd_db
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

\q
```

### Step 3: Install Application

```bash
# Clone repository
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install production server
pip install gunicorn uvicorn[standard]
```

### Step 4: Configure Environment

```bash
# Copy environment template
cp .env.production.example .env.production

# Edit configuration
nano .env.production
```

Set these critical values:
```bash
DATABASE_URL=postgresql://csrd_user:your_password@localhost:5432/csrd_db
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY
SECRET_KEY=$(openssl rand -base64 32)
CSRD_ENCRYPTION_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')
```

### Step 5: Initialize Database

```bash
# Load environment variables
source .env.production

# Run database migrations (if using Alembic)
# alembic upgrade head

# Or run initialization script
# python scripts/init_db.py
```

### Step 6: Start Services

You'll need **3 terminal windows**:

**Terminal 1 - Redis:**
```bash
redis-server
```

**Terminal 2 - Application:**
```bash
source venv/bin/activate
source .env.production

# Option 1: Uvicorn (development-like)
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4

# Option 2: Gunicorn (production)
gunicorn api.server:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --access-logfile - \
  --error-logfile -
```

**Terminal 3 - Monitoring:**
```bash
# Monitor logs
tail -f logs/csrd.log

# Or monitor processes
watch -n 1 'ps aux | grep -E "python|uvicorn|gunicorn"'
```

### Step 7: Create Systemd Service (Linux Production)

```bash
# Create systemd service file
sudo nano /etc/systemd/system/csrd-app.service
```

```ini
[Unit]
Description=CSRD/ESRS Digital Reporting Platform
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=csrd
Group=csrd
WorkingDirectory=/opt/csrd-app
Environment="PATH=/opt/csrd-app/venv/bin"
EnvironmentFile=/opt/csrd-app/.env.production
ExecStart=/opt/csrd-app/venv/bin/gunicorn api.server:app \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 300 \
  --access-logfile /var/log/csrd/access.log \
  --error-logfile /var/log/csrd/error.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable csrd-app
sudo systemctl start csrd-app
sudo systemctl status csrd-app

# View logs
sudo journalctl -u csrd-app -f
```

---

## 🔒 Production Security Checklist

### Pre-Deployment

- [ ] **Strong Passwords**: All passwords are 32+ characters, unique
- [ ] **API Keys**: Production API keys configured (not test keys)
- [ ] **Encryption**: Encryption key generated and backed up securely
- [ ] **TLS/SSL**: HTTPS certificates configured
- [ ] **Database**: PostgreSQL uses SSL/TLS connections
- [ ] **Firewall**: Ports restricted (only 80/443 publicly accessible)
- [ ] **CORS**: Restricted to known domains only
- [ ] **Rate Limiting**: Enabled and configured
- [ ] **Debug Mode**: Disabled in production
- [ ] **Error Traces**: Detailed traces disabled
- [ ] **Secrets**: Never committed to Git
- [ ] **Backups**: Automated backup strategy configured
- [ ] **Monitoring**: Sentry/Prometheus/Grafana configured
- [ ] **Logging**: Centralized log aggregation setup
- [ ] **Access Control**: Principle of least privilege applied

### Post-Deployment

- [ ] **Health Checks**: All endpoints responding correctly
- [ ] **Auto-Scaling**: HPA working (Kubernetes)
- [ ] **Backups**: First backup successful
- [ ] **Monitoring**: Dashboards displaying data
- [ ] **Alerts**: Alert rules configured
- [ ] **Rollback**: Rollback procedure tested
- [ ] **Documentation**: Runbook created
- [ ] **Incident Response**: Plan documented
- [ ] **On-Call**: Rotation schedule set
- [ ] **Security Scan**: Vulnerability scan passed
- [ ] **Load Test**: Performance validated
- [ ] **Disaster Recovery**: DR plan tested

---

## 📊 Monitoring & Observability

### Health Endpoints

```bash
# Application health (liveness)
curl http://localhost:8000/health
# Response: {"status":"healthy","version":"1.0.0","uptime_seconds":1234}

# Readiness check (dependencies)
curl http://localhost:8000/ready
# Response: {"status":"ready","database":"connected","redis":"connected","weaviate":"connected"}

# Prometheus metrics
curl http://localhost:8000/metrics
# Response: Prometheus-compatible metrics
```

### Key Metrics to Monitor

**Application Metrics:**
- Request rate (requests/sec): Target < 1000/sec per pod
- Response time (p50, p95, p99): Target p95 < 500ms
- Error rate (%): Target < 1%
- Active connections: Monitor for leaks

**Business Metrics:**
- Reports generated: Count
- Calculations processed: Count (975 per report)
- Data quality score: Average (target > 95%)
- Pipeline execution time: Seconds (target < 300s)

**Infrastructure Metrics:**
- CPU utilization: Target < 70%
- Memory usage: Target < 80%
- Disk I/O: ops/sec
- Network throughput: MB/s

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m]))

# Response time (95th percentile)
histogram_quantile(0.95,
  rate(http_request_duration_seconds_bucket[5m]))

# Pipeline throughput
rate(csrd_records_processed_total[5m])

# Pod CPU usage
container_cpu_usage_seconds_total{namespace="production",pod=~"csrd-app.*"}

# Pod memory usage
container_memory_usage_bytes{namespace="production",pod=~"csrd-app.*"}
```

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (default: admin/admin)

**Pre-configured dashboards:**
1. **Application Overview**: Request rates, latencies, errors
2. **Business Metrics**: Reports, calculations, quality scores
3. **Infrastructure**: CPU, memory, disk, network
4. **Database**: Query performance, connections, cache hit rate
5. **Kubernetes**: Pod metrics, HPA status, cluster health

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) automatically:

1. **On every push**: Run linting, tests (975 tests), security scans
2. **On develop branch**: Deploy to staging
3. **On master/main branch**: Deploy to production with canary
4. **On tags (v*)**: Create GitHub release

### Triggering Deployments

**Staging Deployment:**
```bash
git checkout develop
git merge feature/new-feature
git push origin develop
# ✅ Automatically deploys to staging
```

**Production Deployment:**
```bash
# Create and push tag
git tag -a v1.1.0 -m "Release v1.1.0 - New features"
git push origin v1.1.0
# ✅ Automatically deploys to production with canary
```

**Manual Deployment:**
```bash
# Trigger workflow manually via GitHub UI
# Actions > CI/CD Pipeline > Run workflow > Select environment
```

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: Code Quality (Ruff, MyPy, Bandit)             │
└─────────────┬───────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│  Stage 2: Tests (975 tests, Coverage)                   │
└─────────────┬───────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────┐
│  Stage 3: Build Docker Image + Security Scan            │
└─────────────┬───────────────────────────────────────────┘
              │
        ┌─────┴──────┐
        │            │
┌───────▼─────┐  ┌──▼───────────┐
│  Deploy     │  │  Deploy      │
│  Staging    │  │  Production  │
│             │  │  (Canary)    │
└─────────────┘  └──────────────┘
```

---

## 🆘 Troubleshooting

### Common Issues

#### Issue: Application won't start

```bash
# Docker Compose
docker-compose logs web

# Kubernetes
kubectl logs deployment/csrd-app -n production

# Common causes:
# 1. Database connection failed
# 2. Missing environment variables
# 3. Port already in use
```

**Solutions:**
```bash
# Check database connectivity
docker-compose exec web python -c "import psycopg2; psycopg2.connect('postgresql://...')"

# Check environment variables
docker-compose exec web env | grep -E "DATABASE|REDIS|API_KEY"

# Check port availability
lsof -i :8000  # Linux/macOS
netstat -ano | findstr :8000  # Windows
```

#### Issue: Database connection error

```bash
# Test database connectivity
psql postgresql://csrd_user:password@localhost:5432/csrd_db

# Check database is running
docker-compose ps db  # Docker
kubectl get pods -l component=database -n production  # K8s

# Check database logs
docker-compose logs db
kubectl logs statefulset/csrd-postgresql -n production
```

#### Issue: Out of memory (OOM)

```bash
# Check memory usage
docker stats  # Docker
kubectl top pods -n production  # Kubernetes

# Increase memory limits (docker-compose.yml)
services:
  web:
    deploy:
      resources:
        limits:
          memory: 4G

# Increase memory limits (Kubernetes)
kubectl edit deployment csrd-app -n production
# Update resources.limits.memory to 8Gi
```

#### Issue: Slow performance

```bash
# Check metrics
curl http://localhost:9090  # Prometheus
curl http://localhost:8000/metrics  # App metrics

# Common causes:
# 1. Insufficient resources (CPU/Memory)
# 2. Database queries not indexed
# 3. Too few workers
# 4. Large request payloads

# Solutions:
# - Scale horizontally (more pods)
# - Add database indexes
# - Increase worker count
# - Enable caching
```

#### Issue: Tests failing

```bash
# Run tests with verbose output
pytest tests/ -vv --tb=long

# Run specific failing test
pytest tests/test_calculator_agent.py::test_ghg_calculation -vv

# Check test dependencies
pip install pytest pytest-cov pytest-asyncio pytest-mock

# Clear test cache
pytest --cache-clear
rm -rf .pytest_cache __pycache__
```

#### Issue: SSL/TLS certificate errors

```bash
# Check certificate expiration
openssl x509 -in /path/to/cert.crt -noout -dates

# Renew Let's Encrypt certificate (cert-manager)
kubectl delete secret csrd-tls -n production
# cert-manager will automatically recreate

# Manual certificate renewal
certbot renew
```

### Debug Mode

```bash
# Enable debug logging (Docker Compose)
docker-compose exec web sh -c "export LOG_LEVEL=DEBUG && python -m api.server"

# Enable debug logging (Kubernetes)
kubectl set env deployment/csrd-app LOG_LEVEL=DEBUG -n production

# Execute interactive shell
docker-compose exec web /bin/bash
kubectl exec -it <pod-name> -n production -- /bin/bash

# Run Python REPL for debugging
docker-compose exec web python
>>> from api.server import app
>>> # Debug here...
```

### Performance Profiling

```bash
# Install profiling tools
pip install py-spy memory_profiler

# Profile CPU
py-spy top --pid $(pgrep -f uvicorn)

# Profile memory
python -m memory_profiler api/server.py

# Generate flame graph
py-spy record -o profile.svg --duration 60 -- python -m api.server
```

---

## 📞 Support & Resources

### Documentation
- **Project README**: [README.md](README.md)
- **API Reference**: http://localhost:8000/docs
- **Development Guide**: [COMPLETE_DEVELOPMENT_GUIDE.md](COMPLETE_DEVELOPMENT_GUIDE.md)
- **Testing Guide**: [TESTING_GUIDE.md](TESTING_GUIDE.md)

### Getting Help

1. **Check Documentation**: Start with this guide and other docs
2. **Search Issues**: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
3. **Create Issue**: Include:
   - Environment details (OS, versions)
   - Error logs (sanitize secrets!)
   - Steps to reproduce
   - Expected vs actual behavior

### Team Contacts
- **Technical Support**: support@greenlang.com
- **Security Issues**: security@greenlang.com
- **General Inquiries**: info@greenlang.com

---

## 📚 Additional Resources

### Training Materials

**For Developers:**
1. Review architecture documentation
2. Set up local development environment
3. Run through quick start examples
4. Write and run tests
5. Contribute features

**For DevOps:**
1. Review this deployment guide
2. Set up staging environment
3. Configure monitoring and alerts
4. Test rollback procedures
5. Document runbooks

**For Business Users:**
1. Review product documentation
2. Complete user training
3. Run example reports
4. Provide feedback

### Best Practices

**Deployment:**
- Always deploy to staging first
- Run smoke tests after deployment
- Monitor metrics for 30 minutes post-deploy
- Keep rollback plan ready
- Document all changes

**Monitoring:**
- Set up alerts for critical metrics
- Review dashboards daily
- Investigate anomalies immediately
- Keep historical data for trends
- Document incidents and resolutions

**Security:**
- Rotate credentials every 90 days
- Review access logs weekly
- Scan for vulnerabilities monthly
- Update dependencies regularly
- Conduct security audits quarterly

---

## 📝 Changelog

### v1.0.0 (2025-11-08)
- ✅ Initial production-ready deployment infrastructure
- ✅ Docker Compose with Weaviate and pgAdmin
- ✅ Complete Kubernetes manifests
- ✅ CI/CD pipeline with 975 test automation
- ✅ FastAPI server implementation
- ✅ Comprehensive deployment documentation
- ✅ Production environment template
- ✅ Security hardening and best practices

---

**Deployment Guide v1.0.0**
**Last Updated:** 2025-11-08
**Status:** Production Ready ✅
**Test Coverage:** 975 tests passing

For questions or issues, please refer to the [support section](#support--resources) above.
