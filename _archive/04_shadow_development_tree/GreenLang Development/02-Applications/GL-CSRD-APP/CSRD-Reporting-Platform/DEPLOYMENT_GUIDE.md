# GL-CSRD-APP Production Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-18
**Status:** Production Ready

---

## 🚀 Quick Start Deployment

Choose your deployment method:
1. [Docker Compose (Fastest)](#docker-compose-deployment) - 5 minutes
2. [Kubernetes (Production)](#kubernetes-deployment) - 15 minutes
3. [Manual Installation](#manual-installation) - 30 minutes

---

## 🐳 Docker Compose Deployment

**Best for:** Development, staging, small production deployments

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 20GB disk space

### Step-by-Step

**1. Clone Repository**
```bash
git clone https://github.com/greenlang/GL-CSRD-APP
cd GL-CSRD-APP/CSRD-Reporting-Platform
```

**2. Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
nano .env
```

Required environment variables:
```bash
# Database
DATABASE_URL=postgresql://csrd_user:YOUR_SECURE_PASSWORD@db:5432/csrd_db

# Redis
REDIS_URL=redis://redis:6379/0

# API Keys
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE

# Application
LOG_LEVEL=INFO
WORKERS=4
ENVIRONMENT=production
```

**3. Start Services**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f web
```

**4. Verify Deployment**
```bash
# Health check
curl http://localhost:8000/health

# Expected response: {"status": "healthy", "version": "1.0.0"}
```

**5. Access Services**
- Application: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

**6. Run Initial Tests**
```bash
# Execute quick start example
docker-compose exec web python examples/quick_start.py

# Run test suite
docker-compose exec web pytest tests/ -v
```

### Maintenance Commands

```bash
# Stop all services
docker-compose down

# Update to latest version
git pull
docker-compose pull
docker-compose up -d

# View logs
docker-compose logs -f [service_name]

# Backup database
docker-compose exec db pg_dump -U csrd_user csrd_db > backup.sql

# Restore database
cat backup.sql | docker-compose exec -T db psql -U csrd_user csrd_db
```

---

## ☸️ Kubernetes Deployment

**Best for:** Production, high availability, auto-scaling

### Prerequisites
- Kubernetes cluster 1.24+
- kubectl configured
- 16GB RAM minimum (cluster)
- 100GB storage
- Helm 3.0+ (optional)

### Step-by-Step

**1. Create Namespace**
```bash
kubectl create namespace production
kubectl config set-context --current --namespace=production
```

**2. Create Secrets**
```bash
# Copy secrets template
cp deployment/k8s/secrets.yaml.example deployment/k8s/secrets.yaml

# Edit with actual values
nano deployment/k8s/secrets.yaml

# Apply secrets
kubectl apply -f deployment/k8s/secrets.yaml
```

**3. Deploy Application**
```bash
# Deploy all resources
kubectl apply -f deployment/k8s/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services
```

**4. Wait for Rollout**
```bash
# Monitor rollout
kubectl rollout status deployment/csrd-app

# Expected: "deployment "csrd-app" successfully rolled out"
```

**5. Verify Deployment**
```bash
# Get service external IP
kubectl get svc csrd-service

# Health check (replace EXTERNAL-IP)
curl http://EXTERNAL-IP/health
```

**6. Configure Auto-Scaling**
```bash
# Check HPA status
kubectl get hpa csrd-hpa

# Expected output shows current/target replicas
```

**7. Set up Monitoring**
```bash
# Install Prometheus (if not already installed)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Access Grafana
kubectl port-forward svc/prometheus-grafana 3000:80
# Open http://localhost:3000 (admin/prom-operator)
```

### Kubernetes Maintenance

```bash
# View logs
kubectl logs -f deployment/csrd-app

# Scale manually
kubectl scale deployment csrd-app --replicas=5

# Rolling update
kubectl set image deployment/csrd-app csrd-app=greenlang/csrd-app:v1.1.0

# Rollback deployment
kubectl rollout undo deployment/csrd-app

# Execute commands in pod
kubectl exec -it csrd-app-xxxx -- /bin/bash

# Port forward for debugging
kubectl port-forward deployment/csrd-app 8000:8000
```

---

## 🔧 Manual Installation

**Best for:** Custom deployments, development

### Prerequisites
- Python 3.10+
- PostgreSQL 15+
- Redis 7+
- 4GB RAM minimum
- Linux/macOS/Windows

### Step-by-Step

**1. Install System Dependencies**

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv postgresql-15 redis-server
```

**macOS:**
```bash
brew install python@3.11 postgresql@15 redis
```

**Windows:**
- Install Python 3.11 from python.org
- Install PostgreSQL from postgresql.org
- Install Redis from github.com/microsoftarchive/redis

**2. Set Up Database**
```bash
# Start PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql@15  # macOS

# Create database and user
sudo -u postgres psql
```

```sql
CREATE USER csrd_user WITH PASSWORD 'your_secure_password';
CREATE DATABASE csrd_db OWNER csrd_user;
GRANT ALL PRIVILEGES ON DATABASE csrd_db TO csrd_user;
\q
```

**3. Install Application**
```bash
# Clone repository
git clone https://github.com/greenlang/GL-CSRD-APP
cd GL-CSRD-APP/CSRD-Reporting-Platform

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**4. Configure Environment**
```bash
# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://csrd_user:your_secure_password@localhost:5432/csrd_db
REDIS_URL=redis://localhost:6379/0
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
LOG_LEVEL=INFO
ENVIRONMENT=production
EOF
```

**5. Initialize Database**
```bash
# Run migrations (if applicable)
python scripts/init_db.py
```

**6. Start Services**

**Terminal 1 - Redis:**
```bash
redis-server
```

**Terminal 2 - Application:**
```bash
source venv/bin/activate
gunicorn --bind 0.0.0.0:8000 --workers 4 --timeout 300 app:app
```

**Terminal 3 - Celery Worker (optional):**
```bash
source venv/bin/activate
celery -A tasks worker --loglevel=info
```

**7. Verify Installation**
```bash
curl http://localhost:8000/health

# Run tests
pytest tests/ -v
```

---

## 🔒 Production Security Checklist

### Pre-Deployment

- [ ] Change all default passwords
- [ ] Generate strong secrets (use `openssl rand -base64 32`)
- [ ] Configure SSL/TLS certificates
- [ ] Set up firewall rules
- [ ] Enable database encryption at rest
- [ ] Configure backup strategy
- [ ] Review and harden CORS settings
- [ ] Enable rate limiting
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation

### Post-Deployment

- [ ] Verify health checks responding
- [ ] Test auto-scaling (if Kubernetes)
- [ ] Verify backups running
- [ ] Check monitoring dashboards
- [ ] Test rollback procedure
- [ ] Document incident response plan
- [ ] Set up on-call rotation
- [ ] Conduct security scan
- [ ] Review access logs
- [ ] Test disaster recovery

---

## 📊 Monitoring & Observability

### Health Endpoints

```bash
# Application health
GET /health
Response: {"status": "healthy", "version": "1.0.0", "timestamp": "..."}

# Readiness check
GET /ready
Response: {"status": "ready", "database": "connected", "redis": "connected"}

# Metrics (Prometheus format)
GET /metrics
Response: Prometheus-compatible metrics
```

### Key Metrics to Monitor

**Application Metrics:**
- Request rate (requests/sec)
- Response time (p50, p95, p99)
- Error rate (%)
- Active connections

**Business Metrics:**
- Reports generated (count)
- Calculations processed (count)
- Data quality score (0-100)
- Pipeline execution time (seconds)

**Infrastructure Metrics:**
- CPU utilization (%)
- Memory usage (%)
- Disk I/O (ops/sec)
- Network throughput (MB/s)

### Prometheus Queries

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# Response time (95th percentile)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Pipeline throughput
rate(csrd_records_processed_total[5m])
```

---

## 🔄 CI/CD Integration

### GitHub Actions (Already Configured)

The application includes three workflows:

**1. test.yml** - Runs on every push
- Unit tests
- Integration tests
- Code coverage

**2. quality_gates.yml** - Runs on every push/PR
- Code linting
- Security scanning
- Performance benchmarks

**3. deploy.yml** - Runs on main branch/tags
- Builds Docker image
- Deploys to staging (develop branch)
- Deploys to production (tags)
- Blue-green deployment

### Triggering Deployments

**Staging:**
```bash
git checkout develop
git merge feature-branch
git push origin develop
# Auto-deploys to staging
```

**Production:**
```bash
# Create release tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
# Auto-deploys to production with blue-green
```

---

## 🆘 Troubleshooting

### Common Issues

**Issue: Application won't start**
```bash
# Check logs
docker-compose logs web  # Docker Compose
kubectl logs deployment/csrd-app  # Kubernetes

# Common causes:
# - Database connection failed
# - Missing environment variables
# - Port already in use
```

**Issue: Database connection error**
```bash
# Test database connectivity
psql postgresql://csrd_user:password@localhost:5432/csrd_db

# Check database is running
docker-compose ps db  # Docker Compose
kubectl get pods | grep postgres  # Kubernetes
```

**Issue: Out of memory**
```bash
# Check memory usage
docker stats  # Docker
kubectl top pods  # Kubernetes

# Solution: Increase memory limits or scale horizontally
```

**Issue: Slow performance**
```bash
# Check metrics
curl http://localhost:9090  # Prometheus
curl http://localhost:8000/metrics  # App metrics

# Common causes:
# - Insufficient resources
# - Database not indexed
# - Too few workers
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose logs -f web

# Execute interactive shell
docker-compose exec web /bin/bash
kubectl exec -it csrd-app-xxxx -- /bin/bash
```

---

## 📞 Support

### Resources
- **Documentation:** /docs directory
- **API Reference:** /docs/API_REFERENCE.md
- **GitHub Issues:** https://github.com/greenlang/GL-CSRD-APP/issues
- **Email:** support@greenlang.com

### Getting Help

1. Check documentation
2. Search GitHub issues
3. Create new issue with:
   - Environment details
   - Error logs
   - Steps to reproduce
   - Expected vs actual behavior

---

## 🎓 Training & Onboarding

### For Developers
1. Read COMPLETE_DEVELOPMENT_GUIDE.md
2. Run local development setup
3. Review code architecture
4. Complete quick start example
5. Write and run tests

### For DevOps
1. Review this deployment guide
2. Set up staging environment
3. Configure monitoring
4. Test rollback procedures
5. Document runbooks

### For Business Users
1. Review README.md
2. Complete user training
3. Run example reports
4. Review output formats
5. Provide feedback

---

**Deployment Guide v1.0.0**
**Last Updated:** 2025-10-18
**Status:** Production Ready ✅
