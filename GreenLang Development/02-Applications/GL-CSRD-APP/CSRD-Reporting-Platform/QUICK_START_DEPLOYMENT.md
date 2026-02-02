# Quick Start Deployment Guide

**Get GL-CSRD-APP running in production in under 10 minutes!**

---

## 🚀 Fastest Path: Docker Compose (5 minutes)

### Prerequisites
- Docker 20.10+ and Docker Compose 2.0+
- 8GB RAM minimum
- Anthropic API key

### Steps

1. **Clone & Navigate**
```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform
```

2. **Configure Environment**
```bash
cp .env.production.example .env.production
```

Edit `.env.production` and set these **required** values:
```bash
# Generate secret key
openssl rand -base64 32

# Generate encryption key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set in .env.production
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_KEY_HERE
SECRET_KEY=<your-generated-secret-key>
CSRD_ENCRYPTION_KEY=<your-generated-encryption-key>
```

3. **Start All Services**
```bash
docker-compose --env-file .env.production up -d
```

4. **Verify It's Running**
```bash
# Check health
curl http://localhost:8000/health

# Expected: {"status":"healthy","version":"1.0.0",...}
```

5. **Access Services**
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050 (admin@csrd.local / admin)
- **Grafana**: http://localhost:3000 (admin / admin)

**Done!** Your CSRD platform is running. 🎉

---

## ☸️ Production Path: Kubernetes (15 minutes)

### Prerequisites
- Kubernetes cluster 1.24+
- kubectl configured
- 16GB RAM minimum (cluster)

### Steps

1. **Clone Repository**
```bash
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform/deployment/k8s
```

2. **Create Namespace**
```bash
kubectl create namespace production
kubectl config set-context --current --namespace=production
```

3. **Configure Secrets**
```bash
# Generate keys
SECRET_KEY=$(openssl rand -base64 32)
ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# Create secret
kubectl create secret generic csrd-secrets \
  --from-literal=database-url="postgresql://csrd_user:CHANGE_PASSWORD@csrd-postgresql:5432/csrd_db" \
  --from-literal=redis-url="redis://csrd-redis:6379/0" \
  --from-literal=weaviate-url="http://csrd-weaviate:8080" \
  --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
  --from-literal=encryption-key="$ENCRYPTION_KEY" \
  --from-literal=secret-key="$SECRET_KEY" \
  --namespace=production
```

4. **Deploy All Components**
```bash
kubectl apply -f configmap.yaml
kubectl apply -f statefulset.yaml
kubectl apply -f service.yaml
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
kubectl apply -f ingress.yaml
```

5. **Wait for Deployment**
```bash
kubectl rollout status deployment/csrd-app -n production
```

6. **Verify**
```bash
kubectl get all -n production

# Test health endpoint
kubectl run test --rm -i --restart=Never \
  --image=curlimages/curl:latest \
  --namespace=production \
  -- curl -f http://csrd-service/health
```

**Done!** Your CSRD platform is running on Kubernetes. 🎉

---

## 📊 What's Running?

After deployment, you have:

### Services
| Service | Purpose | Port |
|---------|---------|------|
| CSRD API | Main application | 8000 |
| PostgreSQL | Primary database | 5432 |
| Redis | Caching | 6379 |
| Weaviate | Vector DB (RAG) | 8080 |
| pgAdmin | DB Management | 5050 |
| Prometheus | Metrics | 9090 |
| Grafana | Dashboards | 3000 |

### Features Enabled
- ✅ Full CSRD/ESRS pipeline (6 agents)
- ✅ 975 metrics calculations
- ✅ AI-powered materiality assessment
- ✅ XBRL/iXBRL generation
- ✅ Data encryption
- ✅ Auto-scaling (K8s only)
- ✅ Health monitoring
- ✅ API documentation

---

## 🧪 Test Your Deployment

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. API Documentation
Open browser: http://localhost:8000/docs

### 3. Run Test Suite
```bash
# Docker Compose
docker-compose exec web pytest tests/ -v

# Kubernetes
kubectl exec -it deployment/csrd-app -n production -- pytest tests/ -v
```

### 4. Process Sample Data
```bash
# Docker Compose
docker-compose exec web python examples/quick_start.py

# Kubernetes
kubectl exec -it deployment/csrd-app -n production -- python examples/quick_start.py
```

---

## 🔧 Common Commands

### Docker Compose

```bash
# View logs
docker-compose logs -f web

# Restart services
docker-compose restart

# Stop all
docker-compose down

# Update & restart
git pull && docker-compose up -d --build

# Backup database
docker-compose exec db pg_dump -U csrd_user csrd_db > backup.sql
```

### Kubernetes

```bash
# View logs
kubectl logs -f deployment/csrd-app -n production

# Restart pods
kubectl rollout restart deployment/csrd-app -n production

# Scale manually
kubectl scale deployment csrd-app --replicas=5 -n production

# Get pod status
kubectl get pods -n production

# Exec into pod
kubectl exec -it <pod-name> -n production -- /bin/bash
```

---

## 🆘 Quick Troubleshooting

### Issue: Service won't start

**Docker Compose:**
```bash
docker-compose logs web
# Check for errors in output
```

**Kubernetes:**
```bash
kubectl describe pod <pod-name> -n production
kubectl logs <pod-name> -n production
```

### Issue: Can't connect to database

**Check database is running:**
```bash
# Docker Compose
docker-compose ps db

# Kubernetes
kubectl get pods -l component=database -n production
```

**Test connection:**
```bash
# Docker Compose
docker-compose exec db psql -U csrd_user -d csrd_db -c "SELECT 1"

# Kubernetes
kubectl exec -it statefulset/csrd-postgresql -n production -- psql -U csrd_user -d csrd_db -c "SELECT 1"
```

### Issue: Out of memory

**Check resource usage:**
```bash
# Docker
docker stats

# Kubernetes
kubectl top pods -n production
kubectl top nodes
```

**Solution:** Increase memory limits in docker-compose.yml or deployment.yaml

---

## 📚 Next Steps

1. **Review Full Documentation**
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
   - [deployment/k8s/README.md](deployment/k8s/README.md) - Kubernetes details
   - [DEPLOYMENT_INFRASTRUCTURE_COMPLETE.md](DEPLOYMENT_INFRASTRUCTURE_COMPLETE.md) - Infrastructure overview

2. **Configure Production Settings**
   - Set up domain and TLS certificates
   - Configure backups
   - Set up monitoring alerts
   - Review security settings

3. **Run Full Test Suite**
   - Execute all 975 tests
   - Verify calculations
   - Test report generation

4. **Enable Monitoring**
   - Access Grafana dashboards
   - Set up alerts in Prometheus
   - Configure Sentry (optional)

5. **Plan Scaling**
   - Review resource usage
   - Configure auto-scaling
   - Plan capacity

---

## 📞 Get Help

- **Documentation**: See DEPLOYMENT.md for full guide
- **Issues**: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- **Email**: support@greenlang.com

---

**Quick Start Guide** v1.0.0
**Last Updated:** 2025-11-08
**Status:** Production Ready ✅
