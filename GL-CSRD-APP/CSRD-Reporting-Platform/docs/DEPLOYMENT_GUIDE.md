# CSRD Platform - Deployment Guide

**Production Deployment Guide for the CSRD/ESRS Digital Reporting Platform**

Version 1.0.0 | Last Updated: 2025-10-18

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Installation Methods](#installation-methods)
4. [Configuration](#configuration)
5. [Security & Compliance](#security--compliance)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring & Logging](#monitoring--logging)
8. [Backup & Recovery](#backup--recovery)
9. [CI/CD Integration](#cicd-integration)
10. [Cloud Deployment](#cloud-deployment)
11. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

**For Processing <10,000 Data Points:**

| Component | Specification |
|-----------|---------------|
| **CPU** | 2 cores @ 2.4 GHz |
| **RAM** | 4 GB |
| **Storage** | 10 GB available |
| **OS** | Windows 10+, macOS 10.15+, Ubuntu 20.04+ |
| **Python** | 3.11+ |
| **Network** | Internet connection for LLM API (materiality only) |

**For Processing >10,000 Data Points:**

| Component | Specification |
|-----------|---------------|
| **CPU** | 4+ cores @ 2.8 GHz |
| **RAM** | 8 GB+ (16 GB recommended) |
| **Storage** | 50 GB available (SSD recommended) |
| **OS** | Ubuntu 22.04 LTS (recommended for production) |
| **Python** | 3.11+ |
| **Network** | Stable internet connection |

### Recommended Production Setup

| Component | Specification |
|-----------|---------------|
| **CPU** | 8 cores @ 3.0 GHz+ |
| **RAM** | 16 GB (32 GB for >50K data points) |
| **Storage** | 100 GB SSD |
| **OS** | Ubuntu 22.04 LTS Server |
| **Python** | 3.11 or 3.12 |
| **Database** | PostgreSQL 14+ (optional) |
| **Backup** | Automated daily backups |

### Software Dependencies

**Required:**
- Python 3.11 or higher
- pip (Python package manager)
- OpenSSL 1.1.1+

**Optional:**
- Docker 24.0+
- PostgreSQL 14+ (for data persistence)
- Redis 7.0+ (for caching)
- Nginx (for reverse proxy)

---

## Environment Setup

### Python Environment

**1. System Python Installation**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# macOS (using Homebrew)
brew install python@3.11

# Windows
# Download from https://www.python.org/downloads/
```

**2. Virtual Environment Setup**

```bash
# Create virtual environment
python3.11 -m venv /opt/csrd-platform/venv

# Activate virtual environment
source /opt/csrd-platform/venv/bin/activate  # Linux/macOS
# or
C:\opt\csrd-platform\venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

**3. Install Platform**

```bash
# Clone repository
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git
cd Code-V1_GreenLang/GL-CSRD-APP/CSRD-Reporting-Platform

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from sdk.csrd_sdk import csrd_build_report; print('✓ Installation successful')"
```

---

## Installation Methods

### Method 1: Standard Installation

**Best for:** Development, testing, single-server deployments

```bash
# 1. Create directory structure
sudo mkdir -p /opt/csrd-platform/{app,data,config,logs,output}

# 2. Clone repository
cd /opt/csrd-platform/app
git clone https://github.com/akshay-greenlang/Code-V1_GreenLang.git .
cd GL-CSRD-APP/CSRD-Reporting-Platform

# 3. Create virtual environment
python3.11 -m venv /opt/csrd-platform/venv

# 4. Install dependencies
source /opt/csrd-platform/venv/bin/activate
pip install -r requirements.txt

# 5. Configure
cp config/csrd_config.example.yaml /opt/csrd-platform/config/csrd_config.yaml
# Edit configuration
nano /opt/csrd-platform/config/csrd_config.yaml

# 6. Test installation
python examples/quick_start.py
```

---

### Method 2: Docker Deployment

**Best for:** Containerized deployments, microservices, cloud

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /app/output /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CSRD_OUTPUT_DIR=/app/output

# Expose port (if needed for API)
EXPOSE 8000

# Default command
CMD ["python", "examples/quick_start.py"]
```

**Build and Run:**

```bash
# Build image
docker build -t csrd-platform:1.0.0 .

# Run container
docker run -d \
  --name csrd-platform \
  -v /opt/csrd-data:/app/data \
  -v /opt/csrd-output:/app/output \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e CSRD_COMPANY_NAME="Acme Corp" \
  csrd-platform:1.0.0

# View logs
docker logs -f csrd-platform
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  csrd-platform:
    build: .
    image: csrd-platform:1.0.0
    container_name: csrd-platform
    restart: unless-stopped
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CSRD_COMPANY_NAME=${CSRD_COMPANY_NAME}
      - CSRD_COMPANY_LEI=${CSRD_COMPANY_LEI}
    networks:
      - csrd-network

  postgres:
    image: postgres:14
    container_name: csrd-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=csrd_db
      - POSTGRES_USER=csrd_user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - csrd-network

volumes:
  postgres-data:

networks:
  csrd-network:
    driver: bridge
```

**Start Services:**

```bash
docker-compose up -d
```

---

### Method 3: Kubernetes Deployment

**Best for:** Large-scale deployments, enterprise

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csrd-platform
  namespace: csrd
spec:
  replicas: 3
  selector:
    matchLabels:
      app: csrd-platform
  template:
    metadata:
      labels:
        app: csrd-platform
    spec:
      containers:
      - name: csrd-platform
        image: csrd-platform:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: csrd-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: data
          mountPath: /app/data
        - name: output
          mountPath: /app/output
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: csrd-data-pvc
      - name: output
        persistentVolumeClaim:
          claimName: csrd-output-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: csrd-platform
  namespace: csrd
spec:
  selector:
    app: csrd-platform
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**Deploy:**

```bash
# Create namespace
kubectl create namespace csrd

# Create secrets
kubectl create secret generic csrd-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  -n csrd

# Apply deployment
kubectl apply -f deployment.yaml
```

---

## Configuration

### Environment Variables

**Production .env File:**

```bash
# Company Information
CSRD_COMPANY_NAME="Acme Manufacturing EU B.V."
CSRD_COMPANY_LEI="549300ABC123DEF456GH"
CSRD_COMPANY_COUNTRY="NL"
CSRD_SECTOR="Manufacturing"
CSRD_REPORTING_YEAR=2024

# LLM Configuration
OPENAI_API_KEY="sk-..."
# or
ANTHROPIC_API_KEY="sk-ant-..."

# Paths
CSRD_DATA_DIR="/opt/csrd-platform/data"
CSRD_OUTPUT_DIR="/opt/csrd-platform/output"
CSRD_CONFIG_PATH="/opt/csrd-platform/config/csrd_config.yaml"

# Database (optional)
DB_HOST="localhost"
DB_PORT=5432
DB_NAME="csrd_db"
DB_USERNAME="csrd_user"
DB_PASSWORD="secure_password"

# Performance
CSRD_MAX_WORKERS=4
CSRD_BATCH_SIZE=1000

# Logging
LOG_LEVEL="INFO"
LOG_FILE="/opt/csrd-platform/logs/csrd.log"
```

**Load Environment Variables:**

```bash
# Load from .env file
export $(cat .env | xargs)

# Or use direnv (recommended)
echo "dotenv" > .envrc
direnv allow
```

---

### Production Configuration

**File: `/opt/csrd-platform/config/csrd_config.yaml`**

```yaml
# Production CSRD Configuration

company:
  name: "Acme Manufacturing EU B.V."
  lei: "549300ABC123DEF456GH"
  country: "NL"
  sector: "Manufacturing"
  reporting_year: 2024

paths:
  esrs_data_points: "/opt/csrd-platform/app/data/esrs_data_points.json"
  emission_factors: "/opt/csrd-platform/app/data/emission_factors.json"
  compliance_rules: "/opt/csrd-platform/app/rules/compliance_rules.yaml"
  output_dir: "/opt/csrd-platform/output"

agents:
  intake:
    enabled: true
    data_quality_threshold: 0.85  # Stricter for production

  materiality:
    enabled: true
    requires_human_review: true
    impact_threshold: 6.0  # Higher bar
    financial_threshold: 6.0

  calculator:
    enabled: true
    zero_hallucination: true
    deterministic: true
    verification: true  # Enable double-checking

  audit:
    enabled: true
    execute_all_rules: true
    generate_auditor_package: true

llm:
  openai:
    api_key_env_var: "OPENAI_API_KEY"
    default_model: "gpt-4o"
    max_retries: 5  # More retries in production
    timeout_seconds: 120

performance:
  total_pipeline_max_minutes: 30
  parallel_processing: true  # Enable for production
  max_workers: 4

logging:
  log_level: "INFO"
  log_to_file: true
  log_file: "/opt/csrd-platform/logs/csrd.log"
  log_rotation: "daily"
  log_retention_days: 90

security:
  encrypt_sensitive_data: true
  audit_trail_retention_days: 2555  # 7 years
```

---

## Security & Compliance

### Data Security

**1. Encryption at Rest**

```bash
# Encrypt sensitive files
openssl enc -aes-256-cbc -salt \
  -in company_profile.json \
  -out company_profile.json.enc \
  -pass pass:$ENCRYPTION_KEY

# Decrypt when needed
openssl enc -d -aes-256-cbc \
  -in company_profile.json.enc \
  -out company_profile.json \
  -pass pass:$ENCRYPTION_KEY
```

**2. API Key Security**

```bash
# Never commit API keys to Git
echo ".env" >> .gitignore
echo "*.key" >> .gitignore

# Use environment variables
export OPENAI_API_KEY=$(cat /secure/openai.key)

# Or use secrets management
# AWS Secrets Manager
aws secretsmanager get-secret-value \
  --secret-id csrd/openai-api-key \
  --query SecretString \
  --output text
```

**3. Access Control**

```bash
# Set proper file permissions
sudo chown -R csrd:csrd /opt/csrd-platform
sudo chmod 700 /opt/csrd-platform/config
sudo chmod 600 /opt/csrd-platform/config/*.yaml
sudo chmod 700 /opt/csrd-platform/output
```

---

### GDPR Compliance

**1. Data Minimization**

```python
# Don't include personal data in ESG data
# Remove PII before processing
import pandas as pd

df = pd.read_csv("esg_data.csv")

# Remove personal identifiers
df = df.drop(columns=['employee_name', 'email', 'phone'], errors='ignore')

df.to_csv("esg_data_anonymized.csv", index=False)
```

**2. Audit Trail Retention**

```yaml
# Configure retention in csrd_config.yaml
security:
  audit_trail_retention_days: 2555  # 7 years as per CSRD requirements
```

**3. Data Access Logging**

```python
import logging

# Configure audit logging
audit_logger = logging.getLogger('csrd.audit')
audit_logger.addHandler(logging.FileHandler('/var/log/csrd/audit.log'))

def log_data_access(user, file, action):
    audit_logger.info(f"User: {user}, File: {file}, Action: {action}")
```

---

## Performance Tuning

### Optimization Strategies

**1. Parallel Processing**

```yaml
# Enable in config
pipeline:
  parallel_processing: true
  max_workers: 4  # Number of CPU cores
```

**2. Batch Processing**

```python
from sdk.csrd_sdk import csrd_build_report

# Process in batches
batch_size = 1000
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    report = csrd_build_report(
        esg_data=batch,
        company_profile="company.json",
        output_dir=f"output/batch_{i}"
    )
```

**3. Caching**

```python
# Cache validation results
import redis
import json

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def validate_with_cache(data_hash, data):
    # Check cache
    cached = redis_client.get(f"validation:{data_hash}")
    if cached:
        return json.loads(cached)

    # Validate
    result = csrd_validate_data(data)

    # Cache result
    redis_client.setex(
        f"validation:{data_hash}",
        86400,  # 24 hours
        json.dumps(result)
    )

    return result
```

---

### Resource Limits

**Docker Resource Limits:**

```bash
docker run -d \
  --name csrd-platform \
  --memory="8g" \
  --memory-swap="8g" \
  --cpus="4" \
  --pids-limit=100 \
  csrd-platform:1.0.0
```

**Kubernetes Resource Limits:**

```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

---

## Monitoring & Logging

### Application Logging

**Configure Structured Logging:**

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName
        })

# Configure logger
logger = logging.getLogger('csrd')
handler = logging.FileHandler('/var/log/csrd/app.log')
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

### Metrics Collection

**Using Prometheus:**

```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
reports_generated = Counter('csrd_reports_total', 'Total reports generated')
processing_time = Histogram('csrd_processing_seconds', 'Report processing time')

# Instrument code
@processing_time.time()
def generate_report():
    report = csrd_build_report(...)
    reports_generated.inc()
    return report

# Start metrics server
start_http_server(9090)
```

---

### Health Checks

**Health Check Endpoint:**

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/ready')
def ready():
    # Check if system is ready
    checks = {
        'database': check_database(),
        'llm_api': check_llm_api(),
        'disk_space': check_disk_space()
    }

    all_healthy = all(checks.values())

    return jsonify({
        'ready': all_healthy,
        'checks': checks
    }), 200 if all_healthy else 503

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## Backup & Recovery

### Backup Strategy

**1. Daily Backups**

```bash
#!/bin/bash
# /opt/csrd-platform/scripts/backup.sh

BACKUP_DIR="/backup/csrd"
DATE=$(date +%Y%m%d)

# Backup data
tar -czf $BACKUP_DIR/data-$DATE.tar.gz /opt/csrd-platform/data

# Backup configuration
tar -czf $BACKUP_DIR/config-$DATE.tar.gz /opt/csrd-platform/config

# Backup output
tar -czf $BACKUP_DIR/output-$DATE.tar.gz /opt/csrd-platform/output

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://csrd-backups/$DATE/
```

**2. Schedule with Cron**

```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/csrd-platform/scripts/backup.sh
```

---

### Disaster Recovery

**Recovery Procedure:**

```bash
#!/bin/bash
# /opt/csrd-platform/scripts/restore.sh

BACKUP_DATE=$1

# Stop services
docker-compose down

# Restore data
tar -xzf /backup/csrd/data-$BACKUP_DATE.tar.gz -C /

# Restore configuration
tar -xzf /backup/csrd/config-$BACKUP_DATE.tar.gz -C /

# Start services
docker-compose up -d

echo "Recovery complete for backup: $BACKUP_DATE"
```

---

## CI/CD Integration

### GitHub Actions

**.github/workflows/deploy.yml:**

```yaml
name: Deploy CSRD Platform

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest

      - name: Run tests
        run: pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t csrd-platform:${{ github.sha }} .

      - name: Push to registry
        run: |
          docker tag csrd-platform:${{ github.sha }} registry.example.com/csrd-platform:latest
          docker push registry.example.com/csrd-platform:latest

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/csrd-platform \
            csrd-platform=registry.example.com/csrd-platform:latest
```

---

## Cloud Deployment

### AWS Deployment

**1. EC2 Instance**

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name csrd-key \
  --security-group-ids sg-xxxxx \
  --subnet-id subnet-xxxxx \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=CSRD-Platform}]'

# SSH and install
ssh -i csrd-key.pem ubuntu@<instance-ip>
# Follow standard installation steps
```

**2. ECS Deployment**

```json
{
  "family": "csrd-platform",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [{
    "name": "csrd-platform",
    "image": "registry.example.com/csrd-platform:latest",
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "environment": [
      {"name": "CSRD_COMPANY_NAME", "value": "Acme Corp"}
    ],
    "secrets": [
      {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."}
    ]
  }]
}
```

---

### Azure Deployment

**1. Azure Container Instance**

```bash
az container create \
  --resource-group csrd-rg \
  --name csrd-platform \
  --image registry.example.com/csrd-platform:latest \
  --cpu 4 \
  --memory 8 \
  --environment-variables CSRD_COMPANY_NAME="Acme Corp" \
  --secure-environment-variables OPENAI_API_KEY="sk-..."
```

---

### GCP Deployment

**1. Cloud Run**

```bash
gcloud run deploy csrd-platform \
  --image registry.example.com/csrd-platform:latest \
  --platform managed \
  --region europe-west1 \
  --memory 8Gi \
  --cpu 4 \
  --set-env-vars CSRD_COMPANY_NAME="Acme Corp" \
  --set-secrets OPENAI_API_KEY=openai-key:latest
```

---

## Troubleshooting

### Common Production Issues

**1. Out of Memory**

```bash
# Check memory usage
free -h

# Increase swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**2. Slow Performance**

```bash
# Check system resources
top
htop

# Check disk I/O
iostat -x 1

# Optimize Python
export PYTHONOPTIMIZE=1
```

**3. API Rate Limits**

```python
# Implement exponential backoff
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_llm_api():
    # API call
    pass
```

---

## Production Checklist

- [ ] System requirements met
- [ ] Python 3.11+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Configuration file created and validated
- [ ] Environment variables set
- [ ] API keys secured
- [ ] File permissions set correctly
- [ ] Logging configured
- [ ] Monitoring in place
- [ ] Backup strategy implemented
- [ ] Health checks configured
- [ ] Disaster recovery plan documented
- [ ] Security audit completed
- [ ] Load testing performed
- [ ] Documentation updated

---

## Support

For production deployment support:
- Email: enterprise@greenlang.io
- Documentation: https://greenlang.io/docs
- Professional Services: Available for custom deployments

---

**Last Updated:** 2025-10-18
**Version:** 1.0.0
**License:** MIT
