# GreenLang Emission Factor API - Deployment Guide

## Overview

This guide covers deploying the GreenLang Emission Factor REST API to production environments.

**API Features:**
- 327+ emission factors (US, EU, UK, and growing)
- Multi-gas breakdown (CO2, CH4, N2O)
- Scope 1, 2, 3 calculations
- Batch processing (up to 100 calculations)
- Redis caching for <50ms response times
- Rate limiting (1000 req/min)
- 99.9% uptime target

**Performance Targets:**
- Response time: <50ms (95th percentile)
- Throughput: 1000 requests/second
- Availability: 99.9% uptime
- Horizontal scaling: Yes

---

## Quick Start

### Local Development

```bash
# 1. Install dependencies
cd greenlang/api
pip install -r requirements.txt

# 2. Run the API
uvicorn greenlang.api.main:app --reload --port 8000

# 3. Access the API
# API: http://localhost:8000
# Docs: http://localhost:8000/api/docs
# ReDoc: http://localhost:8000/api/redoc
```

### Docker (Single Container)

```bash
# Build image
docker build -t greenlang-api:1.0.0 -f greenlang/api/Dockerfile .

# Run container
docker run -p 8000:8000 greenlang-api:1.0.0

# Access API at http://localhost:8000
```

### Docker Compose (API + Redis)

```bash
# Start all services
cd greenlang/api
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

---

## Production Deployment

### Architecture

```
┌─────────────┐
│ Load        │
│ Balancer    │
│ (nginx/ALB) │
└──────┬──────┘
       │
       ├───────────┬───────────┬───────────┐
       │           │           │           │
┌──────▼──────┐ ┌──▼────────┐ ┌──▼────────┐
│ API         │ │ API       │ │ API       │
│ Instance 1  │ │ Instance 2│ │ Instance 3│
│ (4 workers) │ │ (4 workers)│ │ (4 workers)│
└──────┬──────┘ └──┬────────┘ └──┬────────┘
       │           │              │
       └───────────┼──────────────┘
                   │
            ┌──────▼──────┐
            │ Redis       │
            │ Cluster     │
            │ (cache)     │
            └─────────────┘
```

### Prerequisites

1. **Docker & Docker Compose**
   ```bash
   docker --version  # Should be 20.10+
   docker-compose --version  # Should be 1.29+
   ```

2. **Redis** (for caching)
   - Redis 7.x
   - Managed service (AWS ElastiCache, Azure Cache, etc.) OR
   - Self-hosted cluster

3. **Environment Variables**
   ```bash
   REDIS_URL=redis://your-redis-host:6379
   CACHE_ENABLED=true
   LOG_LEVEL=info
   WORKERS=4
   ```

### Deployment Options

#### Option 1: AWS ECS (Fargate)

1. **Build and push Docker image to ECR:**
   ```bash
   # Authenticate to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ECR_URL

   # Build image
   docker build -t greenlang-api:1.0.0 -f greenlang/api/Dockerfile .

   # Tag image
   docker tag greenlang-api:1.0.0 YOUR_ECR_URL/greenlang-api:1.0.0

   # Push to ECR
   docker push YOUR_ECR_URL/greenlang-api:1.0.0
   ```

2. **Create ECS Task Definition:**
   ```json
   {
     "family": "greenlang-api",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "1024",
     "memory": "2048",
     "containerDefinitions": [
       {
         "name": "greenlang-api",
         "image": "YOUR_ECR_URL/greenlang-api:1.0.0",
         "portMappings": [
           {
             "containerPort": 8000,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "REDIS_URL",
             "value": "redis://your-elasticache-endpoint:6379"
           },
           {
             "name": "CACHE_ENABLED",
             "value": "true"
           },
           {
             "name": "LOG_LEVEL",
             "value": "info"
           }
         ],
         "healthCheck": {
           "command": ["CMD-SHELL", "curl -f http://localhost:8000/api/v1/health || exit 1"],
           "interval": 30,
           "timeout": 10,
           "retries": 3,
           "startPeriod": 40
         },
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/greenlang-api",
             "awslogs-region": "us-east-1",
             "awslogs-stream-prefix": "api"
           }
         }
       }
     ]
   }
   ```

3. **Create ECS Service:**
   ```bash
   aws ecs create-service \
     --cluster greenlang-cluster \
     --service-name greenlang-api \
     --task-definition greenlang-api:1 \
     --desired-count 3 \
     --launch-type FARGATE \
     --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
     --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=greenlang-api,containerPort=8000"
   ```

4. **Setup Application Load Balancer:**
   - Create ALB with health check on `/api/v1/health`
   - Configure SSL certificate (AWS ACM)
   - Create target group (port 8000)
   - Setup auto-scaling based on CPU/memory

#### Option 2: Kubernetes (GKE, EKS, AKS)

1. **Create Kubernetes manifests:**

   **deployment.yaml**
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: greenlang-api
     labels:
       app: greenlang-api
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: greenlang-api
     template:
       metadata:
         labels:
           app: greenlang-api
       spec:
         containers:
         - name: api
           image: YOUR_REGISTRY/greenlang-api:1.0.0
           ports:
           - containerPort: 8000
           env:
           - name: REDIS_URL
             value: "redis://redis-service:6379"
           - name: CACHE_ENABLED
             value: "true"
           - name: LOG_LEVEL
             value: "info"
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "2Gi"
               cpu: "1000m"
           livenessProbe:
             httpGet:
               path: /api/v1/health
               port: 8000
             initialDelaySeconds: 40
             periodSeconds: 30
           readinessProbe:
             httpGet:
               path: /api/v1/health
               port: 8000
             initialDelaySeconds: 20
             periodSeconds: 10
   ```

   **service.yaml**
   ```yaml
   apiVersion: v1
   kind: Service
   metadata:
     name: greenlang-api-service
   spec:
     selector:
       app: greenlang-api
     ports:
     - protocol: TCP
       port: 80
       targetPort: 8000
     type: LoadBalancer
   ```

   **hpa.yaml** (Horizontal Pod Autoscaler)
   ```yaml
   apiVersion: autoscaling/v2
   kind: HorizontalPodAutoscaler
   metadata:
     name: greenlang-api-hpa
   spec:
     scaleTargetRef:
       apiVersion: apps/v1
       kind: Deployment
       name: greenlang-api
     minReplicas: 3
     maxReplicas: 10
     metrics:
     - type: Resource
       resource:
         name: cpu
         target:
           type: Utilization
           averageUtilization: 70
     - type: Resource
       resource:
         name: memory
         target:
           type: Utilization
           averageUtilization: 80
   ```

2. **Deploy to Kubernetes:**
   ```bash
   kubectl apply -f deployment.yaml
   kubectl apply -f service.yaml
   kubectl apply -f hpa.yaml

   # Check status
   kubectl get pods
   kubectl get svc
   kubectl get hpa
   ```

#### Option 3: Google Cloud Run

```bash
# Build and deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT/greenlang-api
gcloud run deploy greenlang-api \
  --image gcr.io/YOUR_PROJECT/greenlang-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars REDIS_URL=redis://your-redis-host:6379,CACHE_ENABLED=true \
  --min-instances 3 \
  --max-instances 10 \
  --memory 2Gi \
  --cpu 2
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` | No |
| `CACHE_ENABLED` | Enable/disable caching | `true` | No |
| `LOG_LEVEL` | Logging level (info, debug, warning, error) | `info` | No |
| `WORKERS` | Number of Uvicorn workers | `4` | No |
| `PORT` | Server port | `8000` | No |
| `HOST` | Server host | `0.0.0.0` | No |

### Scaling Configuration

**Vertical Scaling (Per Instance):**
- CPU: 1-2 cores recommended
- Memory: 2GB recommended (1GB minimum)
- Workers: 4 workers per instance (CPU cores * 2)

**Horizontal Scaling:**
- Minimum instances: 3 (high availability)
- Maximum instances: 10-20 (based on load)
- Auto-scale trigger: 70% CPU or 80% memory
- Scale-up: Add 1 instance when threshold exceeded for 2 minutes
- Scale-down: Remove 1 instance when below 30% for 5 minutes

---

## Monitoring & Observability

### Health Checks

```bash
# Health check endpoint
curl http://localhost:8000/api/v1/health

# Expected response (200 OK):
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-11-19T10:30:00Z",
  "database": "connected",
  "cache": "available",
  "uptime_seconds": 86400.0
}
```

### Metrics Endpoints

```bash
# API statistics
curl http://localhost:8000/api/v1/stats

# Coverage statistics
curl http://localhost:8000/api/v1/stats/coverage
```

### Logging

**Log Format:** JSON structured logs

**Log Levels:**
- `INFO`: Normal operations, API requests
- `WARNING`: Cache misses, fallback usage
- `ERROR`: Calculation failures, database errors
- `DEBUG`: Detailed request/response data

**Log Aggregation:**
- AWS: CloudWatch Logs
- GCP: Cloud Logging
- Azure: Application Insights
- Self-hosted: ELK Stack, Grafana Loki

### Monitoring Setup (Prometheus + Grafana)

1. **Add Prometheus metrics endpoint:**
   ```bash
   pip install prometheus-fastapi-instrumentator
   ```

2. **Configure Prometheus scraping:**
   ```yaml
   scrape_configs:
     - job_name: 'greenlang-api'
       static_configs:
         - targets: ['api-service:8000']
       metrics_path: /metrics
   ```

3. **Key metrics to monitor:**
   - Request rate (requests/second)
   - Response time (p50, p95, p99)
   - Error rate (4xx, 5xx)
   - Cache hit rate
   - Database query time
   - Memory usage
   - CPU usage

---

## Performance Optimization

### Caching Strategy

**Redis Configuration:**
```bash
# Cache settings
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

**Cache TTL:**
- Emission factors: 1 hour (3600s)
- Statistics: 5 minutes (300s)
- Search results: 15 minutes (900s)

**Cache Warming:**
- Common factors pre-loaded on startup
- Factors: diesel, gasoline, natural_gas, electricity (US, UK, EU)

### Database Optimization

- Use in-memory database for factors (fast lookups)
- Implement connection pooling (if using PostgreSQL/MySQL)
- Index on: `fuel_type`, `geography`, `scope`, `boundary`

### Rate Limiting

**Current Limits:**
- Factor queries: 1000/minute per IP
- Calculations: 500/minute per IP
- Batch calculations: 100/minute per IP
- Search: 500/minute per IP

**Adjust for production:**
```python
# In main.py
@limiter.limit("2000/minute")  # Increase for production
async def list_factors(...):
    ...
```

---

## Security

### API Authentication (Production TODO)

**Option 1: JWT Tokens**
```python
# Implement JWT validation
from jose import jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return User(**payload)
```

**Option 2: API Keys**
```python
# Implement API key validation
async def validate_api_key(api_key: str = Header(...)):
    if api_key not in valid_keys:
        raise HTTPException(status_code=401)
```

### HTTPS/TLS

**Always use HTTPS in production:**
- Terminate SSL at load balancer (recommended)
- OR use SSL certificates in Uvicorn:
  ```bash
  uvicorn greenlang.api.main:app \
    --ssl-keyfile=/path/to/key.pem \
    --ssl-certfile=/path/to/cert.pem
  ```

### CORS Configuration

**Update allowed origins in production:**
```python
# In main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.greenlang.io",
        "https://dashboard.greenlang.io"
    ],  # Specific domains only
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest greenlang/api/tests/ -v

# Run with coverage
pytest greenlang/api/tests/ -v --cov=greenlang.api --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Load Testing

**Using Apache Bench:**
```bash
# 1000 requests, 10 concurrent
ab -n 1000 -c 10 http://localhost:8000/api/v1/factors

# POST request
ab -n 1000 -c 10 -p calc.json -T application/json \
   http://localhost:8000/api/v1/calculate
```

**Using Locust:**
```python
# locustfile.py
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 2)

    @task(3)
    def list_factors(self):
        self.client.get("/api/v1/factors")

    @task(2)
    def calculate(self):
        self.client.post("/api/v1/calculate", json={
            "fuel_type": "diesel",
            "activity_amount": 100,
            "activity_unit": "gallons",
            "geography": "US"
        })

# Run: locust -f locustfile.py --host http://localhost:8000
```

---

## Troubleshooting

### Common Issues

**1. API not starting:**
```bash
# Check logs
docker logs greenlang-api

# Common causes:
# - Port 8000 already in use
# - Missing dependencies
# - Redis connection failed
```

**2. Slow response times:**
```bash
# Check cache status
curl http://localhost:8000/api/v1/stats | jq '.cache_stats'

# Restart Redis
docker-compose restart redis
```

**3. High memory usage:**
```bash
# Check container stats
docker stats greenlang-api

# Reduce workers or cache size
# In docker-compose.yml:
environment:
  - WORKERS=2
```

**4. Rate limit errors:**
```bash
# Increase rate limits in production
# Or implement per-user limits with API keys
```

---

## Maintenance

### Update Emission Factors

```python
# Add new factor to database
from greenlang.data.emission_factor_database import EmissionFactorDatabase
from greenlang.data.emission_factor_record import EmissionFactorRecord

db = EmissionFactorDatabase()

# Load from JSON file
db._load_custom_factors()  # Loads from data directory

# OR add programmatically
new_factor = EmissionFactorRecord(...)
db.add_factor_record(new_factor)

# Save to file
db.save_to_directory("./custom_factors")
```

### Database Backups

```bash
# Export all factors to JSON
curl http://localhost:8000/api/v1/factors?limit=500 > factors_backup.json

# Or from Python
python -c "
from greenlang.data.emission_factor_database import EmissionFactorDatabase
db = EmissionFactorDatabase()
db.save_to_directory('./backups/factors')
"
```

### Version Upgrades

```bash
# 1. Pull latest code
git pull origin master

# 2. Rebuild Docker image
docker build -t greenlang-api:1.1.0 -f greenlang/api/Dockerfile .

# 3. Update deployment (zero-downtime)
# For Kubernetes:
kubectl set image deployment/greenlang-api api=greenlang-api:1.1.0

# For ECS:
aws ecs update-service --cluster greenlang-cluster \
  --service greenlang-api --force-new-deployment
```

---

## API Usage Examples

### List Factors

```bash
curl "http://localhost:8000/api/v1/factors?fuel_type=diesel&geography=US"
```

### Get Specific Factor

```bash
curl "http://localhost:8000/api/v1/factors/EF:US:diesel:2024:v1"
```

### Calculate Emissions

```bash
curl -X POST "http://localhost:8000/api/v1/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "fuel_type": "diesel",
    "activity_amount": 100,
    "activity_unit": "gallons",
    "geography": "US"
  }'
```

### Batch Calculate

```bash
curl -X POST "http://localhost:8000/api/v1/calculate/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "calculations": [
      {
        "fuel_type": "diesel",
        "activity_amount": 100,
        "activity_unit": "gallons",
        "geography": "US"
      },
      {
        "fuel_type": "natural_gas",
        "activity_amount": 500,
        "activity_unit": "therms",
        "geography": "US"
      }
    ]
  }'
```

---

## Support

**Documentation:** https://docs.greenlang.io/api
**Issues:** https://github.com/greenlang/issues
**Email:** support@greenlang.io

---

## License

Apache 2.0 - See LICENSE file for details
