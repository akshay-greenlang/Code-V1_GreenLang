# GL-CBAM-APP Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the GL-CBAM-APP (GreenLang CBAM Importer Copilot) in production environments. The application supports multiple deployment methods:

- **Docker Compose** - Quick local/single-server deployment
- **Kubernetes** - Scalable cloud-native deployment
- **Manual** - Traditional server deployment

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker Compose)](#quick-start-docker-compose)
3. [Production Deployment (Kubernetes)](#production-deployment-kubernetes)
4. [Manual Deployment](#manual-deployment)
5. [Configuration](#configuration)
6. [Security Hardening](#security-hardening)
7. [Monitoring & Observability](#monitoring--observability)
8. [Backup & Disaster Recovery](#backup--disaster-recovery)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4 GB
- Disk: 50 GB
- OS: Ubuntu 20.04+, RHEL 8+, or equivalent

**Recommended (Production):**
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 100+ GB SSD
- OS: Ubuntu 22.04 LTS

### Software Requirements

1. **Docker & Docker Compose**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose

   # Verify installation
   docker --version
   docker-compose --version
   ```

2. **Kubernetes (for K8s deployment)**
   - kubectl 1.28+
   - Kubernetes cluster 1.28+
   - Helm 3.12+ (optional)

3. **Git**
   ```bash
   sudo apt-get update
   sudo apt-get install git -y
   ```

---

## Quick Start (Docker Compose)

### 1. Clone Repository

```bash
git clone https://github.com/greenlang/GL-CBAM-APP.git
cd GL-CBAM-APP/CBAM-Importer-Copilot
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.production.example .env

# Edit configuration
nano .env
```

**Required changes in `.env`:**
- `POSTGRES_PASSWORD` - Set strong database password
- `REDIS_PASSWORD` - Set strong Redis password
- `SECRET_KEY` - Generate with: `openssl rand -base64 32`
- `JWT_SECRET_KEY` - Generate with: `openssl rand -base64 32`
- `CORS_ORIGINS` - Set your frontend URL(s)

### 3. Start Services

```bash
# Pull images and start services
docker-compose up -d

# View logs
docker-compose logs -f api

# Check service status
docker-compose ps
```

### 4. Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs

# pgAdmin (database UI)
open http://localhost:5050
```

### 5. Run Database Migrations (First Time)

```bash
# Access API container
docker-compose exec api bash

# Run migrations (if applicable)
# python migrate.py

# Exit container
exit
```

### 6. Test CBAM Pipeline

```bash
docker-compose exec api python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output /app/output/test_report.json \
  --importer-name "Test Company BV" \
  --importer-country NL \
  --importer-eori NL123456789 \
  --declarant-name "John Doe" \
  --declarant-position "Compliance Officer"
```

---

## Production Deployment (Kubernetes)

### 1. Prerequisites

**Install kubectl:**
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
kubectl version --client
```

**Install Helm (optional):**
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
helm version
```

**Install cert-manager (for TLS):**
```bash
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
```

**Install NGINX Ingress Controller:**
```bash
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml
```

### 2. Configure Kubernetes Cluster

```bash
# Verify cluster connection
kubectl cluster-info
kubectl get nodes

# Create namespace
kubectl apply -f k8s/configmap.yaml  # This creates the namespace
```

### 3. Configure Secrets

**Important:** Never commit secrets to version control!

```bash
# Generate secrets
SECRET_KEY=$(openssl rand -base64 32)
JWT_SECRET_KEY=$(openssl rand -base64 32)
DB_PASSWORD=$(openssl rand -base64 24)
REDIS_PASSWORD=$(openssl rand -base64 24)

# Create secrets
kubectl create secret generic cbam-api-secrets \
  --from-literal=SECRET_KEY="${SECRET_KEY}" \
  --from-literal=JWT_SECRET_KEY="${JWT_SECRET_KEY}" \
  --from-literal=POSTGRES_PASSWORD="${DB_PASSWORD}" \
  --from-literal=REDIS_PASSWORD="${REDIS_PASSWORD}" \
  --from-literal=DATABASE_URL="postgresql://cbam_user:${DB_PASSWORD}@cbam-postgres:5432/cbam_db" \
  --from-literal=REDIS_URL="redis://:${REDIS_PASSWORD}@cbam-redis:6379/0" \
  -n gl-cbam

# Verify secrets created
kubectl get secrets -n gl-cbam
```

### 4. Update ConfigMaps

```bash
# Edit configmap with your specific values
kubectl edit configmap cbam-api-config -n gl-cbam

# Update CORS_ORIGINS, API URLs, etc.
```

### 5. Deploy PostgreSQL

Create `k8s/postgres-statefulset.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: cbam-postgres
  namespace: gl-cbam
spec:
  serviceName: cbam-postgres-headless
  replicas: 1
  selector:
    matchLabels:
      app: cbam-postgres
  template:
    metadata:
      labels:
        app: cbam-postgres
        component: database
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          ports:
            - containerPort: 5432
          envFrom:
            - secretRef:
                name: cbam-postgres-secrets
          volumeMounts:
            - name: postgres-data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
    - metadata:
        name: postgres-data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 50Gi
```

```bash
kubectl apply -f k8s/postgres-statefulset.yaml
```

### 6. Deploy Redis

Create `k8s/redis-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cbam-redis
  namespace: gl-cbam
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cbam-redis
  template:
    metadata:
      labels:
        app: cbam-redis
        component: cache
    spec:
      containers:
        - name: redis
          image: redis:7-alpine
          ports:
            - containerPort: 6379
          command:
            - redis-server
            - --requirepass
            - $(REDIS_PASSWORD)
          env:
            - name: REDIS_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: cbam-api-secrets
                  key: REDIS_PASSWORD
          volumeMounts:
            - name: redis-data
              mountPath: /data
      volumes:
        - name: redis-data
          persistentVolumeClaim:
            claimName: cbam-redis-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: cbam-redis-data
  namespace: gl-cbam
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

```bash
kubectl apply -f k8s/redis-deployment.yaml
```

### 7. Deploy Application

```bash
# Apply all Kubernetes manifests
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml

# Wait for deployment to complete
kubectl rollout status deployment/cbam-api -n gl-cbam

# Check pod status
kubectl get pods -n gl-cbam -l app=cbam-api

# View logs
kubectl logs -f deployment/cbam-api -n gl-cbam
```

### 8. Configure DNS

```bash
# Get external IP
EXTERNAL_IP=$(kubectl get ingress cbam-api-ingress -n gl-cbam -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "Add DNS A record:"
echo "api.cbam.greenlang.com -> ${EXTERNAL_IP}"
```

### 9. Verify Deployment

```bash
# Check all resources
kubectl get all -n gl-cbam

# Test endpoints
curl https://api.cbam.greenlang.com/health
curl https://api.cbam.greenlang.com/api/v1/health/detailed

# View ingress
kubectl get ingress -n gl-cbam

# Check TLS certificate
kubectl get certificate -n gl-cbam
```

---

## Manual Deployment

### 1. Install Python and Dependencies

```bash
# Install Python 3.11
sudo apt-get update
sudo apt-get install python3.11 python3.11-venv python3-pip -y

# Create virtual environment
python3.11 -m venv /opt/cbam-venv
source /opt/cbam-venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn psycopg2-binary redis
```

### 2. Install PostgreSQL

```bash
# Install PostgreSQL 16
sudo apt-get install postgresql-16 postgresql-contrib-16 -y

# Start service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE cbam_db;
CREATE USER cbam_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE cbam_db TO cbam_user;
\q
EOF
```

### 3. Install Redis

```bash
# Install Redis
sudo apt-get install redis-server -y

# Configure Redis
sudo nano /etc/redis/redis.conf
# Set: requirepass your_redis_password

# Restart Redis
sudo systemctl restart redis-server
sudo systemctl enable redis-server
```

### 4. Configure Application

```bash
# Clone repository
git clone https://github.com/greenlang/GL-CBAM-APP.git
cd GL-CBAM-APP/CBAM-Importer-Copilot

# Create .env file
cp .env.production.example .env
nano .env

# Set proper values
```

### 5. Create Systemd Service

Create `/etc/systemd/system/cbam-api.service`:

```ini
[Unit]
Description=GL-CBAM-APP API Service
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=cbam
Group=cbam
WorkingDirectory=/opt/GL-CBAM-APP/CBAM-Importer-Copilot
Environment="PATH=/opt/cbam-venv/bin"
EnvironmentFile=/opt/GL-CBAM-APP/CBAM-Importer-Copilot/.env
ExecStart=/opt/cbam-venv/bin/gunicorn \
  --bind 0.0.0.0:8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --timeout 120 \
  --access-logfile /var/log/cbam/access.log \
  --error-logfile /var/log/cbam/error.log \
  api.main:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Create user and directories
sudo useradd -r -s /bin/bash cbam
sudo mkdir -p /var/log/cbam /opt/GL-CBAM-APP
sudo chown -R cbam:cbam /var/log/cbam /opt/GL-CBAM-APP

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cbam-api
sudo systemctl start cbam-api

# Check status
sudo systemctl status cbam-api
```

### 6. Configure NGINX Reverse Proxy

```bash
# Install NGINX
sudo apt-get install nginx -y

# Create config
sudo nano /etc/nginx/sites-available/cbam-api
```

```nginx
server {
    listen 80;
    server_name api.cbam.greenlang.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        # Body size
        client_max_body_size 100M;
    }

    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/cbam-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 7. Install SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx -y

# Obtain certificate
sudo certbot --nginx -d api.cbam.greenlang.com

# Auto-renewal is set up automatically
sudo certbot renew --dry-run
```

---

## Configuration

### Environment Variables

All configuration is done via environment variables. See `.env.production.example` for full list.

**Critical variables:**
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `SECRET_KEY` - Application secret key
- `JWT_SECRET_KEY` - JWT signing key
- `CORS_ORIGINS` - Allowed frontend origins

### Application Settings

Edit `k8s/configmap.yaml` (Kubernetes) or `.env` (Docker/Manual) to configure:
- API settings
- Feature flags
- Performance tuning
- CBAM-specific settings

---

## Security Hardening

### 1. Generate Strong Secrets

```bash
# Secret key (32+ characters)
openssl rand -base64 32

# JWT secret
openssl rand -base64 32

# Database password
openssl rand -base64 24
```

### 2. Enable HTTPS/TLS

- **Kubernetes**: Configured via Ingress with cert-manager
- **Docker**: Use reverse proxy (NGINX/Traefik) with Let's Encrypt
- **Manual**: NGINX with certbot

### 3. Configure Firewall

```bash
# UFW (Ubuntu)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Deny direct database access from outside
sudo ufw deny 5432/tcp
sudo ufw deny 6379/tcp
```

### 4. Secure Database

```bash
# PostgreSQL: Edit pg_hba.conf
sudo nano /etc/postgresql/16/main/pg_hba.conf

# Only allow local connections
# local   all   all   md5
# host    all   all   127.0.0.1/32   md5
```

### 5. Enable Rate Limiting

Set in `.env`:
```bash
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

### 6. Configure CORS

```bash
# Restrict to your frontend domain(s)
CORS_ORIGINS=https://cbam.greenlang.com,https://app.greenlang.com
```

---

## Monitoring & Observability

### 1. Health Checks

```bash
# Basic health
curl https://api.cbam.greenlang.com/health

# Detailed health (includes DB, Redis)
curl https://api.cbam.greenlang.com/api/v1/health/detailed
```

### 2. Prometheus Metrics

Enable metrics in `.env`:
```bash
ENABLE_METRICS=true
METRICS_PORT=9090
```

Access metrics:
```bash
curl http://localhost:9090/metrics
```

### 3. Log Aggregation

**Kubernetes:**
```bash
# Stream logs
kubectl logs -f deployment/cbam-api -n gl-cbam

# Last 100 lines
kubectl logs --tail=100 deployment/cbam-api -n gl-cbam

# Export logs
kubectl logs deployment/cbam-api -n gl-cbam > logs.txt
```

**Docker:**
```bash
# View logs
docker-compose logs -f api

# Export logs
docker-compose logs api > logs.txt
```

### 4. Sentry Integration (Error Tracking)

Configure in `.env`:
```bash
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0.1
```

---

## Backup & Disaster Recovery

### 1. Database Backup

**Automated backup script:**

```bash
#!/bin/bash
# /opt/scripts/backup-postgres.sh

BACKUP_DIR="/opt/backups/postgres"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="cbam_db_${DATE}.sql.gz"

# Create backup
docker-compose exec -T postgres pg_dump -U cbam_user cbam_db | gzip > "${BACKUP_DIR}/${BACKUP_FILE}"

# Delete backups older than 30 days
find ${BACKUP_DIR} -name "*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}"
```

**Cron job:**
```bash
# Daily backup at 2 AM
0 2 * * * /opt/scripts/backup-postgres.sh >> /var/log/cbam-backup.log 2>&1
```

### 2. Restore from Backup

```bash
# Docker
gunzip < backup.sql.gz | docker-compose exec -T postgres psql -U cbam_user cbam_db

# Kubernetes
gunzip < backup.sql.gz | kubectl exec -i cbam-postgres-0 -n gl-cbam -- psql -U cbam_user cbam_db
```

### 3. Disaster Recovery Plan

1. **Database**: Daily automated backups to S3/cloud storage
2. **Configuration**: Version controlled in Git
3. **Secrets**: Stored in vault (HashiCorp Vault, AWS Secrets Manager)
4. **Application**: Containerized, can be redeployed quickly
5. **RTO**: < 1 hour
6. **RPO**: < 24 hours

---

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs api
kubectl logs deployment/cbam-api -n gl-cbam

# Common issues:
# 1. Database connection failed
# 2. Redis connection failed
# 3. Missing secrets
# 4. Port already in use
```

### Database Connection Issues

```bash
# Test connection
docker-compose exec api python -c "
from sqlalchemy import create_engine
engine = create_engine('${DATABASE_URL}')
conn = engine.connect()
print('✅ Database connected!')
"
```

### Redis Connection Issues

```bash
# Test Redis
docker-compose exec redis redis-cli ping
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD} ping
```

### Performance Issues

```bash
# Check resource usage
docker stats
kubectl top pods -n gl-cbam

# Increase workers
docker-compose up -d --scale api=5

# Kubernetes autoscaling
kubectl get hpa -n gl-cbam
```

---

## Maintenance

### Update Application

**Docker:**
```bash
# Pull latest code
git pull

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d
```

**Kubernetes:**
```bash
# Update image
kubectl set image deployment/cbam-api api=ghcr.io/greenlang/gl-cbam-app:v1.1.0 -n gl-cbam

# Rollout status
kubectl rollout status deployment/cbam-api -n gl-cbam

# Rollback if needed
kubectl rollout undo deployment/cbam-api -n gl-cbam
```

### Database Migrations

```bash
# Docker
docker-compose exec api alembic upgrade head

# Kubernetes
kubectl exec -it deployment/cbam-api -n gl-cbam -- alembic upgrade head
```

### Clear Redis Cache

```bash
# Docker
docker-compose exec redis redis-cli FLUSHALL

# Kubernetes
kubectl exec -it deployment/cbam-redis -n gl-cbam -- redis-cli FLUSHALL
```

---

## Support & Resources

- **Documentation**: https://docs.greenlang.com/cbam
- **GitHub**: https://github.com/greenlang/GL-CBAM-APP
- **Issues**: https://github.com/greenlang/GL-CBAM-APP/issues
- **Email**: cbam@greenlang.com

---

## License

Copyright 2024 GreenLang. All rights reserved.
