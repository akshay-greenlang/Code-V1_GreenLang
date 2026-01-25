# GL-CBAM-APP - Quick Start Guide

## Get Started in 5 Minutes!

This guide will get GL-CBAM-APP running on your machine in under 5 minutes.

---

## Prerequisites

Make sure you have installed:
- Docker (20.10+)
- Docker Compose (2.0+)

**Don't have Docker?** Install it:
```bash
# Linux/macOS
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Verify installation
docker --version
docker-compose --version
```

---

## Method 1: Docker Compose (Recommended for Local)

### Step 1: Copy Environment File (30 seconds)

```bash
cd GL-CBAM-APP/CBAM-Importer-Copilot
cp .env.production.example .env
```

### Step 2: Generate Secrets (30 seconds)

```bash
# Option A: Using Makefile
make secrets-generate

# Option B: Manual
echo "SECRET_KEY=$(openssl rand -base64 32)"
echo "JWT_SECRET_KEY=$(openssl rand -base64 32)"
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)"
echo "REDIS_PASSWORD=$(openssl rand -base64 24)"
```

Copy the generated values and update your `.env` file.

### Step 3: Update .env (1 minute)

Open `.env` and update these required values:
```bash
# Replace these with generated secrets
SECRET_KEY=<your-generated-secret>
JWT_SECRET_KEY=<your-generated-jwt-secret>
POSTGRES_PASSWORD=<your-generated-db-password>
REDIS_PASSWORD=<your-generated-redis-password>

# Set your frontend URL (if applicable)
CORS_ORIGINS=http://localhost:3000
```

### Step 4: Start Services (2 minutes)

```bash
# Start all services
docker-compose up -d

# Wait for services to be ready (check logs)
docker-compose logs -f api
# Press Ctrl+C when you see "Application startup complete"
```

### Step 5: Verify Deployment (1 minute)

```bash
# Check service status
docker-compose ps

# Test API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "timestamp": "2024-11-08T..."}

# Open API documentation
open http://localhost:8000/docs
# or visit in browser: http://localhost:8000/docs
```

### Success! You're Running! 🎉

Access your services:
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050 (admin@greenlang.com / check .env for password)

---

## Method 2: Using Makefile (Even Easier!)

If you have `make` installed:

```bash
# Step 1: Setup environment
make env-example
make secrets-generate  # Copy these to .env

# Step 2: Edit .env with generated secrets
nano .env

# Step 3: Start everything
make up

# Step 4: Check health
make health

# Step 5: View logs
make logs-api
```

Done! Services are running!

---

## Test the CBAM Pipeline

Once services are running, test the CBAM pipeline:

```bash
# Run the demo pipeline
docker-compose exec api python cbam_pipeline.py \
  --input examples/demo_shipments.csv \
  --output /app/output/test_report.json \
  --importer-name "Test Company BV" \
  --importer-country NL \
  --importer-eori NL123456789 \
  --declarant-name "John Doe" \
  --declarant-position "Compliance Officer"

# Check the output
docker-compose exec api cat /app/output/test_report.json
```

---

## Common Commands

### View Logs
```bash
# All services
docker-compose logs -f

# API only
docker-compose logs -f api

# Database
docker-compose logs -f postgres
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart API only
docker-compose restart api
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (CAUTION: deletes data!)
docker-compose down -v
```

### Database Access
```bash
# PostgreSQL shell
docker-compose exec postgres psql -U cbam_user -d cbam_db

# pgAdmin web UI
open http://localhost:5050
```

### Redis Access
```bash
# Redis CLI
docker-compose exec redis redis-cli

# Check Redis (if password protected)
docker-compose exec redis redis-cli -a <REDIS_PASSWORD> ping
```

---

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Option 1: Stop the other service
# Option 2: Change port in .env
API_PORT=8001
```

### Database Connection Failed
```bash
# Check if PostgreSQL is running
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Services Not Starting
```bash
# Check logs for errors
docker-compose logs

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Reset Everything
```bash
# Nuclear option: Remove everything and start fresh
docker-compose down -v
rm .env
cp .env.production.example .env
# Update .env with new secrets
docker-compose up -d
```

---

## Next Steps

### For Development
1. Read `DEPLOYMENT.md` for detailed deployment options
2. Check `Makefile` for available commands (`make help`)
3. Explore API docs at http://localhost:8000/docs

### For Production
1. Follow the Kubernetes deployment guide in `DEPLOYMENT.md`
2. Configure production secrets properly
3. Set up monitoring and backups
4. Review security hardening checklist

### For CI/CD
1. Configure GitHub Actions (`.github/workflows/ci-cd.yml`)
2. Set up GitHub secrets
3. Configure deployment environments

---

## Quick Reference

### Service URLs (Default)
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- pgAdmin: http://localhost:5050
- PostgreSQL: localhost:5432
- Redis: localhost:6379

### Default Credentials (Update in .env!)
- PostgreSQL User: `cbam_user`
- PostgreSQL DB: `cbam_db`
- pgAdmin Email: Check `.env` (PGADMIN_EMAIL)

### Useful Makefile Commands
```bash
make help          # Show all commands
make up            # Start services
make down          # Stop services
make logs          # View logs
make health        # Check API health
make test          # Run tests
make db-backup     # Backup database
```

---

## Getting Help

- **Documentation**: See `DEPLOYMENT.md`
- **Infrastructure Guide**: See `DEPLOYMENT_INFRASTRUCTURE_README.md`
- **Issues**: GitHub Issues
- **Email**: cbam@greenlang.com

---

**Deployment Time: ~5 minutes**
**Difficulty: Easy**
**Status: Production Ready!**

Built with excellence by Team A1: GL-CBAM Deployment Infrastructure Builder
