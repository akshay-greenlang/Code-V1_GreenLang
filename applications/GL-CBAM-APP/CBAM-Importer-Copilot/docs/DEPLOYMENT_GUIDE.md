# CBAM Importer Copilot - Deployment Guide

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**Target Audience:** DevOps engineers, system administrators, IT managers

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [Production Deployment](#production-deployment)
5. [Docker Deployment](#docker-deployment)
6. [Cloud Deployment](#cloud-deployment)
7. [Configuration Management](#configuration-management)
8. [Security Hardening](#security-hardening)
9. [Monitoring & Logging](#monitoring--logging)
10. [Backup & Recovery](#backup--recovery)
11. [Scaling & Performance](#scaling--performance)

---

## Overview

### Deployment Architectures

This tool supports multiple deployment patterns:

1. **Local Desktop** - Individual compliance officers
2. **Shared Server** - Team collaboration
3. **Docker Container** - Isolated environment
4. **Cloud VM** - AWS/Azure/GCP
5. **Kubernetes** - Enterprise scale

### Infrastructure Sizing

| Deployment Type | Users | Shipments/Month | Resources |
|----------------|-------|-----------------|-----------|
| **Desktop** | 1 | <10K | 4GB RAM, 2 CPU |
| **Small Team** | 5-10 | <50K | 8GB RAM, 4 CPU |
| **Department** | 10-50 | <500K | 16GB RAM, 8 CPU |
| **Enterprise** | 50+ | >500K | 32GB+ RAM, 16+ CPU |

---

## System Requirements

### Minimum Requirements

**Hardware:**
- CPU: 2 cores @ 2.0 GHz
- RAM: 4 GB
- Disk: 10 GB free space
- Network: Internet for initial setup only

**Software:**
- Python: 3.9 or higher
- Operating System: Linux, macOS, or Windows
- Git: 2.30 or higher (for installation)

### Recommended Production Setup

**Hardware:**
- CPU: 8 cores @ 3.0 GHz
- RAM: 16 GB
- Disk: 100 GB SSD
- Network: 1 Gbps

**Software:**
- Python: 3.11+ (tested on 3.11.5)
- Operating System: Ubuntu 22.04 LTS
- Database: PostgreSQL 15 (for audit log storage)
- Container: Docker 24+ (optional)

### Performance by Configuration

| Configuration | 10K Shipments | 100K Shipments | 1M Shipments |
|---------------|---------------|----------------|--------------|
| Minimum (4GB) | ~30s | ~5min | ~45min |
| Recommended (16GB) | ~10s | ~2min | ~15min |
| High (32GB) | ~5s | ~30s | ~5min |

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

**For: Individual users, small teams**

```bash
# Step 1: Clone repository
git clone https://github.com/greenlang/cbam-importer-copilot.git
cd cbam-importer-copilot

# Step 2: Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Step 4: Verify installation
gl cbam --version
# Expected output: cbam-importer-copilot 1.0.0

# Step 5: Run example
gl cbam report examples/demo_shipments.csv \
  --importer-name "Acme Steel BV" \
  --importer-country "NL" \
  --importer-eori "NL123456789012"

# Step 6: Verify output
ls output/
# Expected: cbam_report.json, cbam_summary.md
```

**Time to complete:** ~5 minutes

**Pros:**
- ✅ Full control over environment
- ✅ Easy to customize
- ✅ Works offline after initial setup

**Cons:**
- ❌ Manual dependency management
- ❌ Environment conflicts possible

---

### Method 2: Docker Installation (Isolated)

**For: Production environments, team deployments**

```bash
# Step 1: Build Docker image
docker build -t cbam-copilot:1.0.0 .

# Step 2: Run container
docker run -v $(pwd)/data:/app/data \
           -v $(pwd)/output:/app/output \
           cbam-copilot:1.0.0 \
           gl cbam report /app/data/shipments.csv \
           --importer-name "Acme Steel BV" \
           --importer-country "NL" \
           --importer-eori "NL123456789012"

# Step 3: Verify output
ls output/
```

**Time to complete:** ~10 minutes (including build)

**Pros:**
- ✅ Complete isolation
- ✅ Reproducible environment
- ✅ Easy to version
- ✅ No dependency conflicts

**Cons:**
- ❌ Requires Docker knowledge
- ❌ Slightly slower startup

**Dockerfile:**

```dockerfile
# Dockerfile
FROM python:3.11.5-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Verify installation
RUN gl cbam --version

# Create output directory
RUN mkdir -p /app/output

# Set entrypoint
ENTRYPOINT ["gl", "cbam"]
CMD ["--help"]
```

---

### Method 3: System-Wide Installation

**For: Shared servers, multi-user systems**

```bash
# Step 1: Install system-wide (requires sudo)
sudo pip3 install -e .

# Step 2: Verify installation
gl cbam --version

# Step 3: All users can now use
gl cbam report ~/data/shipments.csv ...
```

**Pros:**
- ✅ Available to all users
- ✅ Single installation
- ✅ System PATH integration

**Cons:**
- ❌ Requires admin privileges
- ❌ Potential version conflicts
- ❌ Harder to upgrade

---

## Production Deployment

### Architecture: Shared Application Server

```
┌─────────────────────────────────────────────────────────────────┐
│                      USERS (Compliance Team)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ User 1   │  │ User 2   │  │ User 3   │  │ User N   │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
└───────┼─────────────┼─────────────┼─────────────┼───────────────┘
        │             │             │             │
        └─────────────┴─────────────┴─────────────┘
                          │
                          ↓ SSH / Web Interface
┌─────────────────────────────────────────────────────────────────┐
│              APPLICATION SERVER (cbam-server-01)                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CBAM Importer Copilot Application                        │  │
│  │  - Multi-tenant configuration                             │  │
│  │  - User-specific output directories                       │  │
│  │  - Audit logging                                          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────┐  ┌───────────────────┐                  │
│  │  Data Storage     │  │  PostgreSQL       │                  │
│  │  /data/           │  │  (Audit logs)     │                  │
│  └───────────────────┘  └───────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   BACKUP & ARCHIVE                               │
│  - Daily backups to S3/Azure Blob                               │
│  - 4-year retention for compliance                              │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Production Deployment

#### 1. Server Provisioning

```bash
# Ubuntu 22.04 LTS recommended
# Instance: 8 vCPU, 16GB RAM, 100GB SSD

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3.11 python3.11-venv python3-pip git postgresql-15

# Create application user
sudo useradd -m -s /bin/bash cbam
sudo passwd cbam
```

#### 2. Application Installation

```bash
# As cbam user
sudo su - cbam

# Create application directory
mkdir -p /home/cbam/app
cd /home/cbam/app

# Clone repository
git clone https://github.com/greenlang/cbam-importer-copilot.git
cd cbam-importer-copilot

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install application
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
gl cbam --version
```

#### 3. Directory Structure

```bash
# Create production directory structure
mkdir -p /home/cbam/{data,output,config,logs,backups}

# Set permissions
chmod 750 /home/cbam/{data,output,config,logs,backups}
chown -R cbam:cbam /home/cbam
```

**Structure:**

```
/home/cbam/
├── app/                       # Application code
│   └── cbam-importer-copilot/
├── data/                      # Input files
│   ├── entity_1/
│   ├── entity_2/
│   └── shared/
├── output/                    # Generated reports
│   ├── 2025/
│   │   ├── Q1/
│   │   ├── Q2/
│   │   └── Q3/
│   └── archive/
├── config/                    # Configuration files
│   ├── entity_1.cbam.yaml
│   ├── entity_2.cbam.yaml
│   └── global.yaml
├── logs/                      # Application logs
│   └── cbam.log
└── backups/                   # Local backups
    └── daily/
```

#### 4. Configuration Management

```bash
# Create global configuration
cat > /home/cbam/config/global.yaml <<EOF
# Global CBAM configuration
paths:
  cn_codes: "/home/cbam/app/cbam-importer-copilot/data/cn_codes.json"
  rules: "/home/cbam/app/cbam-importer-copilot/rules/cbam_rules.yaml"
  suppliers: "/home/cbam/data/shared/suppliers.yaml"

output:
  directory: "/home/cbam/output"
  format: "both"
  include_provenance: true

performance:
  chunk_size: 5000
  parallel_workers: 4
  memory_limit_mb: 8192

logging:
  level: "INFO"
  file: "/home/cbam/logs/cbam.log"
  rotation: "daily"
  retention_days: 90
EOF
```

#### 5. Multi-Tenant Setup

```bash
# Create entity-specific configs
cat > /home/cbam/config/acme_nl.cbam.yaml <<EOF
importer:
  name: "Acme Steel EU BV"
  country: "NL"
  eori: "NL123456789012"

declarant:
  name: "John Smith"
  position: "Compliance Officer"
  email: "john.smith@acme.eu"

# Inherit global paths and settings
extends: "/home/cbam/config/global.yaml"
EOF

# Usage
gl cbam report /home/cbam/data/entity_1/shipments.csv \
  --config /home/cbam/config/acme_nl.cbam.yaml
```

#### 6. Systemd Service (Optional)

**For automated processing:**

```ini
# /etc/systemd/system/cbam-processor.service
[Unit]
Description=CBAM Report Processor
After=network.target

[Service]
Type=simple
User=cbam
WorkingDirectory=/home/cbam/app/cbam-importer-copilot
Environment="PATH=/home/cbam/app/cbam-importer-copilot/venv/bin"
ExecStart=/home/cbam/app/cbam-importer-copilot/venv/bin/python3 -m cbam_processor
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable cbam-processor
sudo systemctl start cbam-processor
sudo systemctl status cbam-processor
```

---

## Docker Deployment

### Production Docker Compose

**For: Team deployments with database**

```yaml
# docker-compose.yml
version: '3.8'

services:
  cbam-app:
    build: .
    image: cbam-copilot:1.0.0
    container_name: cbam-app
    volumes:
      - ./data:/app/data
      - ./output:/app/output
      - ./config:/app/config
      - ./logs:/app/logs
    environment:
      - CBAM_LOG_LEVEL=INFO
      - CBAM_DATABASE_URL=postgresql://cbam:password@db:5432/cbam
    depends_on:
      - db
    networks:
      - cbam-network
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    container_name: cbam-db
    environment:
      - POSTGRES_USER=cbam
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=cbam
    volumes:
      - db-data:/var/lib/postgresql/data
    networks:
      - cbam-network
    restart: unless-stopped

  backup:
    image: postgres:15-alpine
    container_name: cbam-backup
    volumes:
      - ./backups:/backups
      - db-data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=cbam
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=cbam
    command: >
      sh -c "while true; do
        pg_dump -U cbam -h db cbam > /backups/cbam_\$(date +%Y%m%d_%H%M%S).sql
        find /backups -name 'cbam_*.sql' -mtime +7 -delete
        sleep 86400
      done"
    depends_on:
      - db
    networks:
      - cbam-network
    restart: unless-stopped

volumes:
  db-data:

networks:
  cbam-network:
    driver: bridge
```

**Deployment:**

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f cbam-app

# Run report
docker-compose exec cbam-app gl cbam report /app/data/shipments.csv ...

# Stop services
docker-compose down
```

---

## Cloud Deployment

### AWS Deployment

#### Architecture: EC2 + S3

```
┌─────────────────────────────────────────────────────────────────┐
│                          USERS                                   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ↓ HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                      AWS LOAD BALANCER                           │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                    EC2 Instance (t3.xlarge)                      │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  CBAM Application                                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────┐      │      ┌──────────────────┐         │
│  │  EBS Volume      │      ↓      │  CloudWatch      │         │
│  │  (Application)   │      │      │  (Monitoring)    │         │
│  └──────────────────┘      │      └──────────────────┘         │
└────────────────────────────┼─────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                         S3 BUCKETS                               │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  Input Data      │  │  Output Reports  │                    │
│  │  (cbam-input)    │  │  (cbam-output)   │                    │
│  └──────────────────┘  └──────────────────┘                    │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │  Backups         │  │  Audit Logs      │                    │
│  │  (cbam-backup)   │  │  (cbam-audit)    │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

#### Step-by-Step AWS Deployment

**1. Create S3 Buckets:**

```bash
# Create buckets
aws s3 mb s3://cbam-input
aws s3 mb s3://cbam-output
aws s3 mb s3://cbam-backup

# Enable versioning
aws s3api put-bucket-versioning \
  --bucket cbam-output \
  --versioning-configuration Status=Enabled

# Enable encryption
aws s3api put-bucket-encryption \
  --bucket cbam-output \
  --server-side-encryption-configuration '{
    "Rules": [{
      "ApplyServerSideEncryptionByDefault": {
        "SSEAlgorithm": "AES256"
      }
    }]
  }'
```

**2. Launch EC2 Instance:**

```bash
# Launch t3.xlarge instance (4 vCPU, 16GB RAM)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name cbam-key \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cbam-server-01}]'
```

**3. Install Application:**

```bash
# SSH to instance
ssh -i cbam-key.pem ubuntu@ec2-xxx.amazonaws.com

# Install dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv awscli

# Clone and install app (same as standard installation)
git clone https://github.com/greenlang/cbam-importer-copilot.git
cd cbam-importer-copilot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**4. Configure S3 Integration:**

```python
# scripts/s3_integration.py
import boto3
from sdk.cbam_sdk import cbam_build_report

s3 = boto3.client('s3')

# Download input from S3
s3.download_file('cbam-input', 'shipments.csv', '/tmp/shipments.csv')

# Generate report
report = cbam_build_report(
    input_file='/tmp/shipments.csv',
    importer_name="Acme Steel BV",
    importer_country="NL",
    importer_eori="NL123456789012",
    output_path='/tmp/cbam_report.json'
)

# Upload to S3
s3.upload_file('/tmp/cbam_report.json', 'cbam-output',
               f'reports/{report.report_id}.json')

print(f"✓ Report uploaded to s3://cbam-output/reports/{report.report_id}.json")
```

---

### Azure Deployment

#### Architecture: VM + Blob Storage

```bash
# Create resource group
az group create --name cbam-rg --location westeurope

# Create storage account
az storage account create \
  --name cbamstorage \
  --resource-group cbam-rg \
  --location westeurope \
  --sku Standard_LRS

# Create VM
az vm create \
  --resource-group cbam-rg \
  --name cbam-vm \
  --image UbuntuLTS \
  --size Standard_D4s_v3 \
  --admin-username cbamadmin \
  --ssh-key-values ~/.ssh/id_rsa.pub

# Install application (same as standard installation)
```

---

## Configuration Management

### Environment Variables

**Production environment:**

```bash
# /etc/environment (system-wide)
CBAM_ENVIRONMENT=production
CBAM_LOG_LEVEL=INFO
CBAM_DATABASE_URL=postgresql://cbam:password@localhost:5432/cbam
CBAM_S3_BUCKET=cbam-output
CBAM_BACKUP_ENABLED=true
CBAM_BACKUP_RETENTION_DAYS=1460  # 4 years
```

### Secrets Management

**AWS Secrets Manager:**

```python
import boto3
import json

def get_secret(secret_name):
    client = boto3.client('secretsmanager', region_name='eu-west-1')
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response['SecretString'])

# Usage
importer_config = get_secret('cbam/importer/acme-nl')
report = cbam_build_report(
    input_file="shipments.csv",
    importer_name=importer_config['name'],
    importer_country=importer_config['country'],
    importer_eori=importer_config['eori']
)
```

---

## Security Hardening

### Application Security

**1. File Permissions:**

```bash
# Restrict access to sensitive files
chmod 600 /home/cbam/config/*.yaml
chmod 700 /home/cbam/data
chmod 700 /home/cbam/output

# Set ownership
chown -R cbam:cbam /home/cbam
```

**2. Network Security:**

```bash
# Firewall rules (ufw)
sudo ufw allow ssh
sudo ufw allow from 10.0.0.0/8 to any port 5432  # PostgreSQL (internal only)
sudo ufw enable
```

**3. HTTPS Enforcement:**

```bash
# If using web interface, enforce HTTPS
sudo apt install -y certbot
sudo certbot certonly --standalone -d cbam.example.com
```

### Data Encryption

**At Rest:**

```bash
# Encrypt output directory
sudo apt install -y ecryptfs-utils
sudo mount -t ecryptfs /home/cbam/output /home/cbam/output
```

**In Transit:**

```bash
# Use SCP with SSH keys
scp -i cbam-key.pem shipments.csv cbam@server:/home/cbam/data/

# Or SFTP
sftp -i cbam-key.pem cbam@server
```

---

## Monitoring & Logging

### Application Logging

**Configure structured logging:**

```python
# config/logging.yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: '%(asctime)s %(name)s %(levelname)s %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /home/cbam/logs/cbam.log
    maxBytes: 10485760  # 10MB
    backupCount: 10

  syslog:
    class: logging.handlers.SysLogHandler
    level: WARNING
    formatter: standard
    address: /dev/log

root:
  level: INFO
  handlers: [console, file, syslog]
```

### Health Checks

```bash
# scripts/healthcheck.sh
#!/bin/bash

# Check application version
VERSION=$(gl cbam --version)
if [ $? -ne 0 ]; then
  echo "ERROR: Application not responding"
  exit 1
fi

# Check disk space
DISK_USAGE=$(df -h /home/cbam | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
  echo "WARNING: Disk usage at ${DISK_USAGE}%"
fi

# Check database connection (if applicable)
psql -h localhost -U cbam -d cbam -c "SELECT 1" > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "ERROR: Database connection failed"
  exit 1
fi

echo "✓ All health checks passed"
exit 0
```

### Monitoring with Prometheus

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'cbam'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

---

## Backup & Recovery

### Automated Backups

```bash
# scripts/backup.sh
#!/bin/bash

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/cbam/backups/daily"

# Backup output files
tar -czf $BACKUP_DIR/output_$DATE.tar.gz /home/cbam/output

# Backup database (if applicable)
pg_dump -U cbam cbam > $BACKUP_DIR/cbam_$DATE.sql

# Upload to S3
aws s3 sync $BACKUP_DIR s3://cbam-backup/$(date +%Y/%m/%d)/

# Cleanup old backups (keep 90 days local)
find $BACKUP_DIR -name "*.tar.gz" -mtime +90 -delete
find $BACKUP_DIR -name "*.sql" -mtime +90 -delete

echo "✓ Backup completed: $DATE"
```

**Cron job:**

```bash
# /etc/cron.d/cbam-backup
0 2 * * * cbam /home/cbam/app/scripts/backup.sh >> /home/cbam/logs/backup.log 2>&1
```

### Disaster Recovery

**Recovery steps:**

```bash
# 1. Restore application
git clone https://github.com/greenlang/cbam-importer-copilot.git
cd cbam-importer-copilot
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Restore configuration
aws s3 cp s3://cbam-backup/config/ /home/cbam/config/ --recursive

# 3. Restore latest data
aws s3 cp s3://cbam-backup/$(date +%Y/%m/%d)/ /home/cbam/backups/latest/ --recursive
tar -xzf /home/cbam/backups/latest/output_*.tar.gz -C /

# 4. Restore database
psql -U cbam cbam < /home/cbam/backups/latest/cbam_*.sql

# 5. Verify
gl cbam --version
ls /home/cbam/output
```

---

## Scaling & Performance

### Horizontal Scaling

**For >1M shipments/month:**

```
┌────────────────────────────────────────────────┐
│            LOAD BALANCER                       │
└────────────┬───────────────────────────────────┘
             │
      ┌──────┴──────┬──────────┬──────────┐
      │             │          │          │
┌─────▼─────┐ ┌─────▼─────┐ ┌─▼──────┐ ┌─▼──────┐
│  Worker 1 │ │  Worker 2 │ │ Worker │ │ Worker │
│           │ │           │ │    3   │ │    N   │
└───────────┘ └───────────┘ └────────┘ └────────┘
      │             │          │          │
      └─────────────┴──────────┴──────────┘
                    │
              ┌─────▼─────┐
              │ Shared DB │
              └───────────┘
```

**Implementation:**

```python
# Celery for distributed processing
from celery import Celery

app = Celery('cbam', broker='redis://localhost:6379')

@app.task
def process_report(input_file, config):
    """Process report asynchronously."""
    return cbam_build_report(
        input_file=input_file,
        config=config
    )

# Submit tasks
result = process_report.delay('shipments.csv', config)
```

### Performance Tuning

**Configuration:**

```yaml
# config/performance.yaml
performance:
  # Parallel processing
  parallel_workers: 8          # Match CPU cores
  chunk_size: 5000             # Batch size for processing

  # Memory management
  memory_limit_mb: 16384       # 16GB limit
  use_chunked_reading: true    # For large files

  # Caching
  cache_emission_factors: true
  cache_cn_codes: true

  # Database
  connection_pool_size: 20
  max_overflow: 10
```

---

**Version:** 1.0.0
**Last Updated:** 2025-10-15
**License:** MIT

---

*For troubleshooting deployment issues, see `docs/TROUBLESHOOTING.md`*
