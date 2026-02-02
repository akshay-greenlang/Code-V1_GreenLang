# CSRD Reporting Platform - Production Runbook

**Version:** 1.0
**Date:** 2025-10-20
**Status:** PRODUCTION
**Owner:** DevOps + Operations Team

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Deployment Procedures](#deployment-procedures)
4. [Incident Response](#incident-response)
5. [Common Operations](#common-operations)
6. [Monitoring and Alerting](#monitoring-and-alerting)
7. [Troubleshooting](#troubleshooting)
8. [Database Operations](#database-operations)
9. [Security Operations](#security-operations)
10. [Escalation and Contacts](#escalation-and-contacts)

---

## 🎯 Overview

### Purpose

This runbook provides operational procedures for the CSRD Reporting Platform in production. It covers deployment, monitoring, incident response, and troubleshooting.

### System Overview

- **Platform:** CSRD/ESRS Digital Reporting
- **Components:** 6 AI agents + API + Database + Cache
- **Users:** Corporate sustainability teams, auditors, regulators
- **SLA:** 99.5% uptime, <200ms API latency (p95)

### Critical Services

| Service | Port | Purpose | Critical? |
|---------|------|---------|-----------|
| csrd-api | 8000 | Main API endpoint | ✅ YES |
| intake-agent | 8001 | Data ingestion | ✅ YES |
| calculator-agent | 8002 | Emissions calculations | ✅ YES |
| materiality-agent | 8004 | AI assessments | ⚠️ MEDIUM |
| audit-agent | 8005 | Compliance validation | ✅ YES |
| reporting-agent | 8006 | XBRL generation | ✅ YES |
| PostgreSQL | 5432 | Primary database | ✅ YES |
| Redis | 6379 | Cache | ⚠️ MEDIUM |

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────┐
│   Users     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│        Load Balancer (ALB)          │
└──────┬──────────────────┬───────────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  API Server  │   │  API Server  │
│  (Primary)   │   │  (Standby)   │
└──────┬───────┘   └──────┬───────┘
       │                  │
       ├──────────────────┤
       │                  │
       ▼                  ▼
┌──────────────────────────────────────┐
│         Agent Services               │
│  ┌─────┬─────┬─────┬─────┬─────┐   │
│  │Intk │Calc │Mat  │Audit│Rep  │   │
│  └─────┴─────┴─────┴─────┴─────┘   │
└──────┬──────────────────┬───────────┘
       │                  │
       ▼                  ▼
┌──────────────┐   ┌──────────────┐
│  PostgreSQL  │   │    Redis     │
│   (Primary)  │   │    (Cache)   │
└──────────────┘   └──────────────┘
```

### Network Architecture

- **VPC:** 10.0.0.0/16
- **Public Subnet:** 10.0.1.0/24 (Load Balancer)
- **Private Subnet:** 10.0.2.0/24 (API + Agents)
- **Database Subnet:** 10.0.3.0/24 (PostgreSQL + Redis)

---

## 🚀 Deployment Procedures

### Pre-Deployment Checklist

- [ ] All tests passed (≥95% pass rate)
- [ ] Security scans completed (0 critical issues)
- [ ] Performance benchmarks met (all 6 SLAs)
- [ ] Database migrations prepared
- [ ] Backup taken
- [ ] Rollback plan documented
- [ ] Stakeholders notified
- [ ] Monitoring dashboards reviewed

### Blue-Green Deployment (Recommended)

**Duration:** ~30 minutes
**Downtime:** Zero

#### Step 1: Prepare Green Environment

```bash
# Deploy to green environment
cd /opt/csrd-platform
git checkout tags/v1.0.1

# Build containers
docker-compose -f docker-compose.prod.yml build

# Run database migrations (green DB)
docker-compose exec api python manage.py migrate

# Start green environment
docker-compose -f docker-compose.green.yml up -d
```

#### Step 2: Validate Green Environment

```bash
# Health checks
curl https://green.csrd.internal/health
curl https://green.csrd.internal/health/ready

# Smoke tests
python run_tests.py --quick --env=green

# Performance validation
python benchmark.py --env=green --iterations=3
```

#### Step 3: Switch Traffic to Green

```bash
# Update load balancer target group
aws elbv2 modify-target-group \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=green-instance-id

# Verify traffic routing
curl https://csrd.prod.example.com/health

# Monitor logs and metrics for 10 minutes
watch -n 5 'curl -s https://csrd.prod.example.com/health/ready | jq'
```

#### Step 4: Decommission Blue Environment

```bash
# Wait 30 minutes for in-flight requests to complete
sleep 1800

# Stop blue environment
docker-compose -f docker-compose.blue.yml down

# Keep blue environment available for 24h for rollback
```

### Rolling Deployment (Alternative)

**Duration:** ~20 minutes
**Downtime:** Zero (but brief degraded performance)

```bash
# Deploy to instances one at a time
for instance in api-1 api-2 api-3; do
    echo "Deploying to $instance"

    # Remove from load balancer
    aws elbv2 deregister-targets --target-group-arn ... --targets Id=$instance

    # Wait for connections to drain (30s)
    sleep 30

    # Deploy new version
    ssh $instance "cd /opt/csrd && git pull && docker-compose up -d --build"

    # Wait for health checks
    ssh $instance "while ! curl -s http://localhost:8000/health/ready; do sleep 5; done"

    # Add back to load balancer
    aws elbv2 register-targets --target-group-arn ... --targets Id=$instance

    # Wait before next instance
    sleep 60
done
```

### Database Migrations

```bash
# Always test migrations in staging first!

# Backup database before migration
pg_dump -h prod-db.internal -U csrd -d csrd_prod > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migration (zero-downtime approach)
python manage.py migrate --no-input

# If migration fails, rollback
python manage.py migrate <app_name> <previous_migration>

# Restore from backup if needed
psql -h prod-db.internal -U csrd -d csrd_prod < backup_YYYYMMDD_HHMMSS.sql
```

### Rollback Procedures

#### Immediate Rollback (Blue-Green)

```bash
# Switch traffic back to blue
aws elbv2 modify-target-group \
  --target-group-arn ... \
  --targets Id=blue-instance-id

# Verify rollback successful
curl https://csrd.prod.example.com/health

# Log rollback incident
echo "Rollback executed at $(date)" >> /var/log/csrd/rollback.log
```

#### Emergency Rollback

```bash
# Stop all new deployments immediately
kubectl rollout undo deployment/csrd-api

# Or for Docker:
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod-previous.yml up -d

# Verify system health
python run_tests.py --quick --env=prod
```

---

## 🚨 Incident Response

### Severity Levels

| Level | Description | Response Time | Example |
|-------|-------------|---------------|---------|
| **P0 - Critical** | Complete outage, data loss | 15 minutes | API down, database corrupted |
| **P1 - High** | Major degradation | 30 minutes | 50% error rate, slow responses |
| **P2 - Medium** | Minor degradation | 2 hours | One agent failing, cache down |
| **P3 - Low** | Cosmetic issue | Next business day | UI formatting, minor bugs |

### Incident Response Workflow

#### 1. Detection (0-5 minutes)

- **Automated:** Prometheus alerts → PagerDuty → On-call engineer
- **Manual:** User report → Support → On-call engineer

#### 2. Initial Response (5-15 minutes)

```bash
# Acknowledge alert
# Check monitoring dashboards: Grafana

# Quick health assessment
curl https://csrd.prod.example.com/health
curl https://csrd.prod.example.com/health/ready

# Check recent deployments
git log --oneline -10
kubectl get events --sort-by='.lastTimestamp' | head -20

# Check resource usage
kubectl top pods
kubectl top nodes
```

#### 3. Investigation (15-45 minutes)

**Check logs:**
```bash
# API logs
kubectl logs -f deployment/csrd-api --tail=100

# Agent logs
kubectl logs -f deployment/intake-agent --tail=100

# Database logs
kubectl exec -it postgres-0 -- tail -f /var/log/postgresql/postgresql.log
```

**Check metrics:**
- Error rates
- Latency (p50, p95, p99)
- Resource usage (CPU, memory, disk)
- Database connections

#### 4. Mitigation (45-60 minutes)

**Common mitigation actions:**
- Restart unhealthy pods
- Scale up resources
- Rollback recent deployment
- Enable circuit breakers
- Switch to degraded mode

#### 5. Resolution (60-90 minutes)

- Apply permanent fix
- Validate resolution
- Monitor for recurrence
- Update runbook

#### 6. Post-Incident (Within 48 hours)

- Write post-mortem
- Identify root cause
- Create prevention action items
- Update monitoring/alerts

### Incident Communication Template

```
INCIDENT: [Brief description]
STATUS: [Investigating / Mitigating / Resolved]
IMPACT: [Affected users / services]
STARTED: [Timestamp]
DURATION: [Elapsed time]
UPDATES:
  - [Timestamp]: [Update message]
  - [Timestamp]: [Update message]
ROOT CAUSE: [After resolution]
RESOLUTION: [Steps taken]
```

---

## ⚙️ Common Operations

### Restarting Services

```bash
# Restart API
kubectl rollout restart deployment/csrd-api

# Restart specific agent
kubectl rollout restart deployment/calculator-agent

# Restart all services (use carefully!)
kubectl rollout restart deployment --all
```

### Scaling Services

```bash
# Scale API horizontally
kubectl scale deployment/csrd-api --replicas=5

# Scale agent vertically (update resources)
kubectl set resources deployment/calculator-agent \
  --limits=cpu=2,memory=4Gi \
  --requests=cpu=1,memory=2Gi
```

### Viewing Logs

```bash
# Real-time logs
kubectl logs -f deployment/csrd-api

# Last 100 lines
kubectl logs deployment/csrd-api --tail=100

# Search logs
kubectl logs deployment/csrd-api | grep "ERROR"

# Export logs
kubectl logs deployment/csrd-api > api-logs.txt
```

### Database Backups

```bash
# Manual backup
pg_dump -h prod-db.internal -U csrd -d csrd_prod \
  | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Verify backup
gunzip -c backup_YYYYMMDD_HHMMSS.sql.gz | head -20

# Restore from backup
gunzip -c backup_YYYYMMDD_HHMMSS.sql.gz | \
  psql -h prod-db.internal -U csrd -d csrd_prod
```

### Cache Operations

```bash
# Clear Redis cache
redis-cli -h redis.internal FLUSHALL

# Check cache stats
redis-cli -h redis.internal INFO

# Monitor cache commands
redis-cli -h redis.internal MONITOR
```

---

## 📊 Monitoring and Alerting

### Key Dashboards

1. **Platform Overview** (`https://grafana.internal/d/csrd-overview`)
   - API latency, error rates, throughput
   - Agent health status
   - Resource usage

2. **Agent Performance** (`https://grafana.internal/d/csrd-agents`)
   - Execution times per agent
   - Success/failure rates
   - Memory usage

3. **Database Health** (`https://grafana.internal/d/csrd-database`)
   - Connection pools
   - Query latency
   - Lock contention

### Critical Alerts

| Alert | Threshold | Action |
|-------|-----------|--------|
| API Down | 1 minute | P0 - Immediate restart |
| High Latency | p95 >1s for 5min | P1 - Investigate |
| Error Rate | >10% for 2min | P0 - Rollback |
| Database Down | 1 minute | P0 - Failover |
| Disk Full | <5% free | P1 - Expand storage |
| Memory High | >95% for 5min | P1 - Scale up |

### Alert Response Procedures

**API Down:**
1. Check health endpoints
2. Review recent deployments
3. Check resource usage
4. Restart service if needed
5. Rollback if restart fails

**High Latency:**
1. Check database query performance
2. Review recent code changes
3. Check external API dependencies
4. Scale horizontally if needed

**Error Rate Spike:**
1. Check error logs for patterns
2. Identify affected endpoints
3. Roll back recent deployment
4. Enable circuit breakers if needed

---

## 🔧 Troubleshooting

### Issue: API Returning 500 Errors

**Symptoms:**
- HTTP 500 responses
- Errors in API logs
- Users unable to generate reports

**Diagnosis:**
```bash
# Check API logs
kubectl logs deployment/csrd-api --tail=100 | grep ERROR

# Check agent health
curl http://reporting-agent.internal:8006/health

# Check database connectivity
kubectl exec -it api-pod -- python manage.py dbshell
```

**Resolution:**
1. If agent down: Restart agent service
2. If database issue: Check connections, run VACUUM
3. If code bug: Rollback to previous version
4. If resource exhaustion: Scale up pods

### Issue: XBRL Generation Slow

**Symptoms:**
- XBRL generation >5 minutes
- Reports timing out
- High CPU usage on reporting agent

**Diagnosis:**
```bash
# Check reporting agent performance
kubectl top pod -l app=reporting-agent

# Check recent XBRL generations
kubectl logs deployment/reporting-agent | grep "XBRL generation"

# Profile slow report
python -m cProfile -o profile.stats agents/reporting_agent.py
```

**Resolution:**
1. Check data volume (large datasets = slower)
2. Optimize XBRL templates
3. Scale reporting agent pods
4. Enable caching for repeated reports

### Issue: Database Connection Pool Exhausted

**Symptoms:**
- "Too many connections" errors
- API requests hanging
- Slow database queries

**Diagnosis:**
```bash
# Check current connections
psql -c "SELECT count(*) FROM pg_stat_activity;"

# Check connections by application
psql -c "SELECT application_name, count(*) FROM pg_stat_activity GROUP BY application_name;"

# Check idle connections
psql -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'idle';"
```

**Resolution:**
1. Kill idle connections: `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '10 minutes';`
2. Increase max_connections in PostgreSQL
3. Optimize connection pooling in application
4. Add pgBouncer connection pooler

---

## 💾 Database Operations

### Regular Maintenance

```bash
# Weekly VACUUM ANALYZE
psql -c "VACUUM ANALYZE;"

# Monthly full VACUUM
psql -c "VACUUM FULL;"

# Reindex
psql -c "REINDEX DATABASE csrd_prod;"
```

### Performance Tuning

```bash
# Find slow queries
psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Find missing indexes
psql -c "SELECT schemaname, tablename, attname FROM pg_stats WHERE correlation < 0.1 ORDER BY correlation;"

# Add index
psql -c "CREATE INDEX CONCURRENTLY idx_name ON table_name(column_name);"
```

---

## 🔐 Security Operations

### Security Incident Response

1. **Contain:** Isolate affected systems
2. **Assess:** Determine scope and impact
3. **Remediate:** Apply security patches
4. **Recover:** Restore normal operations
5. **Learn:** Update security controls

### Access Review

```bash
# List database users
psql -c "\du"

# Review API keys
kubectl get secrets -n prod

# Audit log access
kubectl logs deployment/csrd-api | grep "authentication"
```

---

## 📞 Escalation and Contacts

### On-Call Rotation

| Role | Primary | Secondary |
|------|---------|-----------|
| DevOps | devops-oncall@greenlang.com | devops-lead@greenlang.com |
| Backend | backend-oncall@greenlang.com | tech-lead@greenlang.com |
| Security | security-oncall@greenlang.com | ciso@greenlang.com |

### Escalation Path

```
L1: On-Call Engineer (15 min)
  ↓ (if not resolved in 30 min)
L2: Team Lead (30 min)
  ↓ (if not resolved in 1 hour)
L3: Engineering Manager
  ↓ (if critical and not resolved in 2 hours)
L4: CTO
```

### External Contacts

- **AWS Support:** https://console.aws.amazon.com/support
- **Anthropic API Support:** support@anthropic.com
- **OpenAI API Support:** support@openai.com

---

**Last Updated:** 2025-10-20
**Next Review:** Monthly
**Document Owner:** DevOps Team
