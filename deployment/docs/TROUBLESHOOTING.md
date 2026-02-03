# GreenLang Platform - Troubleshooting Guide

**Document ID:** INFRA-001-TS
**Version:** 1.0.0
**Last Updated:** 2026-02-03
**Status:** Production

---

## Table of Contents

1. [Quick Diagnostic Commands](#quick-diagnostic-commands)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [Log Locations and Analysis](#log-locations-and-analysis)
4. [Debugging Commands](#debugging-commands)
5. [Performance Issues](#performance-issues)
6. [Support Contacts](#support-contacts)

---

## Quick Diagnostic Commands

Run these commands first when troubleshooting any issue:

```bash
# 1. Overall platform health
./deploy.sh health

# 2. Container status
docker-compose -f docker-compose-unified.yml ps

# 3. Resource usage
docker stats --no-stream

# 4. Recent logs (all services)
docker-compose -f docker-compose-unified.yml logs --tail=100

# 5. Network connectivity
docker network inspect greenlang-platform

# 6. Disk usage
docker system df
```

---

## Common Issues and Solutions

### Issue: Services Won't Start

#### Symptom
```
ERROR: Container cbam-api failed to start
```

#### Diagnosis
```bash
# Check container status and exit code
docker-compose -f docker-compose-unified.yml ps -a

# Check detailed logs
docker-compose -f docker-compose-unified.yml logs cbam-api

# Check container inspect for errors
docker inspect cbam-api | grep -A 10 "State"
```

#### Solutions

**Solution 1: Port Already in Use**
```bash
# Check what's using the port
lsof -i :8001  # For CBAM
lsof -i :8002  # For CSRD
lsof -i :8000  # For VCCI

# Kill the process or change the port
kill -9 <PID>

# Or modify docker-compose-unified.yml to use different port
```

**Solution 2: Insufficient Memory**
```bash
# Check available memory
free -h

# Check Docker memory
docker system info | grep Memory

# Increase Docker memory allocation (Docker Desktop)
# Settings > Resources > Memory > Increase to 8GB+

# Or reduce container memory limits
```

**Solution 3: Missing Environment Variables**
```bash
# Verify .env file exists
ls -la .env

# Check required variables
cat .env | grep -E "POSTGRES_PASSWORD|REDIS_PASSWORD|SHARED_JWT_SECRET"

# If missing, copy from example
cp .env.example .env
# Then edit with your values
```

**Solution 4: Dependencies Not Ready**
```bash
# Start infrastructure first
docker-compose -f docker-compose-unified.yml up -d postgres redis rabbitmq weaviate

# Wait for them to be healthy
sleep 60

# Then start applications
docker-compose -f docker-compose-unified.yml up -d cbam-api csrd-web vcci-backend vcci-worker
```

---

### Issue: Database Connection Errors

#### Symptom
```
sqlalchemy.exc.OperationalError: could not connect to server
Connection refused
```

#### Diagnosis
```bash
# Check PostgreSQL status
docker-compose -f docker-compose-unified.yml ps postgres
docker-compose -f docker-compose-unified.yml logs postgres

# Test connection
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin

# Check connection count
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "SELECT count(*) FROM pg_stat_activity;"
```

#### Solutions

**Solution 1: PostgreSQL Container Not Running**
```bash
# Start PostgreSQL
docker-compose -f docker-compose-unified.yml up -d postgres

# Wait for it to be ready
sleep 30

# Verify
docker-compose -f docker-compose-unified.yml exec postgres pg_isready -U greenlang_admin
```

**Solution 2: Max Connections Reached**
```bash
# Check current connections
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "
SELECT count(*), state FROM pg_stat_activity GROUP BY state;
"

# Kill idle connections
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE state = 'idle'
AND query_start < now() - interval '10 minutes';
"

# Restart applications to reset connection pools
docker-compose -f docker-compose-unified.yml restart cbam-api csrd-web vcci-backend
```

**Solution 3: Wrong Credentials**
```bash
# Verify credentials in .env
cat .env | grep POSTGRES

# Test connection manually
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "\conninfo"
```

**Solution 4: Database Not Initialized**
```bash
# Check if schemas exist
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "\dn"

# If schemas missing, re-run init script
docker-compose -f docker-compose-unified.yml exec -T postgres psql -U greenlang_admin -d greenlang_platform < init/shared_db_schema.sql
```

---

### Issue: Redis Connection Errors

#### Symptom
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

#### Diagnosis
```bash
# Check Redis status
docker-compose -f docker-compose-unified.yml ps redis
docker-compose -f docker-compose-unified.yml logs redis

# Test connection
docker-compose -f docker-compose-unified.yml exec redis redis-cli ping

# Check memory usage
docker-compose -f docker-compose-unified.yml exec redis redis-cli INFO memory
```

#### Solutions

**Solution 1: Redis Container Not Running**
```bash
# Start Redis
docker-compose -f docker-compose-unified.yml up -d redis

# Verify
docker-compose -f docker-compose-unified.yml exec redis redis-cli ping
# Should return: PONG
```

**Solution 2: Authentication Failed**
```bash
# Verify password in .env
cat .env | grep REDIS_PASSWORD

# Test with password
docker-compose -f docker-compose-unified.yml exec redis redis-cli -a "$REDIS_PASSWORD" ping
```

**Solution 3: Memory Limit Reached**
```bash
# Check memory usage
docker-compose -f docker-compose-unified.yml exec redis redis-cli INFO memory | grep used_memory_human

# Clear cache if needed
docker-compose -f docker-compose-unified.yml exec redis redis-cli -a "$REDIS_PASSWORD" FLUSHALL

# Or increase memory limit in docker-compose-unified.yml
```

---

### Issue: RabbitMQ Connection Errors

#### Symptom
```
pika.exceptions.AMQPConnectionError: Connection to RabbitMQ failed
```

#### Diagnosis
```bash
# Check RabbitMQ status
docker-compose -f docker-compose-unified.yml ps rabbitmq
docker-compose -f docker-compose-unified.yml logs rabbitmq

# Check via management UI
# Access: http://localhost:15672
# Login: greenlang / greenlang_rabbit_2024

# Check queue status
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmqctl list_queues
```

#### Solutions

**Solution 1: RabbitMQ Not Ready**
```bash
# RabbitMQ takes time to start (30-60 seconds)
docker-compose -f docker-compose-unified.yml up -d rabbitmq
sleep 60

# Verify
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmq-diagnostics ping
```

**Solution 2: Virtual Host Not Created**
```bash
# List virtual hosts
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmqctl list_vhosts

# Create if missing
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmqctl add_vhost greenlang_platform
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmqctl set_permissions -p greenlang_platform greenlang ".*" ".*" ".*"
```

**Solution 3: Queue Stuck**
```bash
# Purge a specific queue
docker-compose -f docker-compose-unified.yml exec rabbitmq rabbitmqctl purge_queue <queue_name> -p greenlang_platform

# Or restart RabbitMQ (queues are persistent)
docker-compose -f docker-compose-unified.yml restart rabbitmq
```

---

### Issue: Application Returns 5xx Errors

#### Symptom
```
HTTP 500 Internal Server Error
HTTP 502 Bad Gateway
HTTP 503 Service Unavailable
```

#### Diagnosis
```bash
# Check application logs
docker-compose -f docker-compose-unified.yml logs --tail=200 cbam-api | grep -i error

# Check health endpoint
curl -v http://localhost:8001/health

# Check resource usage
docker stats cbam-api
```

#### Solutions

**Solution 1: Application Crash - Restart**
```bash
# Simple restart
docker-compose -f docker-compose-unified.yml restart cbam-api

# Or force recreate
docker-compose -f docker-compose-unified.yml up -d --force-recreate cbam-api
```

**Solution 2: Out of Memory**
```bash
# Check container memory
docker stats cbam-api --no-stream

# Increase memory limit in docker-compose-unified.yml:
# deploy:
#   resources:
#     limits:
#       memory: 4G

# Apply change
docker-compose -f docker-compose-unified.yml up -d cbam-api
```

**Solution 3: Backend Service Unavailable**
```bash
# Check all dependencies
docker-compose -f docker-compose-unified.yml ps

# Restart if any are unhealthy
docker-compose -f docker-compose-unified.yml restart postgres redis rabbitmq

# Wait and retry
sleep 30
curl http://localhost:8001/health
```

---

### Issue: Slow Response Times

#### Symptom
```
API requests taking > 5 seconds
Timeouts on certain endpoints
```

#### Diagnosis
```bash
# Check response time
time curl http://localhost:8001/health

# Check database queries
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC
LIMIT 10;
"

# Check Redis latency
docker-compose -f docker-compose-unified.yml exec redis redis-cli --latency

# Check resource usage
docker stats
```

#### Solutions

**Solution 1: Database Slow Queries**
```bash
# Find slow queries
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
"

# Run VACUUM ANALYZE
docker-compose -f docker-compose-unified.yml exec postgres vacuumdb -U greenlang_admin -d greenlang_platform --analyze
```

**Solution 2: Redis Cache Miss**
```bash
# Check hit rate
docker-compose -f docker-compose-unified.yml exec redis redis-cli INFO stats | grep keyspace

# Warm up cache by restarting application
docker-compose -f docker-compose-unified.yml restart cbam-api
```

**Solution 3: Increase Resources**
```bash
# Scale horizontally
docker-compose -f docker-compose-unified.yml up -d --scale cbam-api=3

# Or increase container resources
# Edit docker-compose-unified.yml and redeploy
```

---

### Issue: Authentication Failures

#### Symptom
```
HTTP 401 Unauthorized
Invalid or expired token
```

#### Diagnosis
```bash
# Check JWT secret is consistent
cat .env | grep JWT

# Test token generation
curl -X POST http://localhost:8001/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password&username=admin@greenlang.com&password=admin123"

# Check application logs
docker-compose -f docker-compose-unified.yml logs cbam-api | grep -i auth
```

#### Solutions

**Solution 1: JWT Secret Mismatch**
```bash
# Ensure all apps use same JWT secret
# Check .env
grep SHARED_JWT_SECRET .env

# Restart all applications to pick up changes
docker-compose -f docker-compose-unified.yml restart cbam-api csrd-web vcci-backend
```

**Solution 2: Token Expired**
```bash
# Default expiry is 30 minutes
# Request new token
curl -X POST http://localhost:8001/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=password&username=admin@greenlang.com&password=admin123"
```

**Solution 3: User Not Found**
```bash
# Check if user exists
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT email, is_active FROM public.users WHERE email = 'admin@greenlang.com';
"

# Create user if missing
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
INSERT INTO public.users (email, password_hash, is_active)
VALUES ('admin@greenlang.com', 'hashed_password', true);
"
```

---

### Issue: Disk Space Full

#### Symptom
```
No space left on device
Write errors in logs
```

#### Diagnosis
```bash
# Check disk usage
df -h

# Check Docker disk usage
docker system df

# Find large containers
docker ps --size

# Find large volumes
docker volume ls
for vol in $(docker volume ls -q); do
  echo "$vol: $(docker volume inspect $vol --format '{{.Mountpoint}}')"
done
```

#### Solutions

**Solution 1: Clean Docker Resources**
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes (CAREFUL!)
docker volume prune -f

# Full cleanup
docker system prune -a -f
```

**Solution 2: Truncate Logs**
```bash
# Find large log files
find /var/lib/docker/containers -name "*-json.log" -size +100M

# Truncate (preserves file, removes content)
truncate -s 0 /var/lib/docker/containers/<container_id>/<container_id>-json.log

# Or configure log rotation in /etc/docker/daemon.json
```

**Solution 3: Clean Application Data**
```bash
# Check volume sizes
docker volume inspect cbam-uploads --format '{{.Mountpoint}}'
du -sh /var/lib/docker/volumes/cbam-uploads/_data

# Clean old uploads (be careful!)
# Backup first, then remove old files
```

---

## Log Locations and Analysis

### Docker Container Logs

```bash
# All services
docker-compose -f docker-compose-unified.yml logs

# Specific service
docker-compose -f docker-compose-unified.yml logs cbam-api

# Follow logs in real-time
docker-compose -f docker-compose-unified.yml logs -f cbam-api

# Last N lines
docker-compose -f docker-compose-unified.yml logs --tail=100 cbam-api

# Since timestamp
docker-compose -f docker-compose-unified.yml logs --since="2026-02-03T10:00:00" cbam-api

# Filter for errors
docker-compose -f docker-compose-unified.yml logs cbam-api 2>&1 | grep -i error
```

### Application Log Files

| Service | Volume | Log Path |
|---------|--------|----------|
| CBAM API | cbam-logs | /app/logs/cbam.log |
| CSRD Web | csrd-logs | /app/logs/csrd.log |
| VCCI Backend | vcci-logs | /app/logs/vcci.log |

```bash
# Access log files directly
docker-compose -f docker-compose-unified.yml exec cbam-api cat /app/logs/cbam.log

# Or copy to host
docker cp cbam-api:/app/logs/cbam.log ./cbam.log
```

### PostgreSQL Logs

```bash
# View PostgreSQL logs
docker-compose -f docker-compose-unified.yml logs postgres

# Query log (if enabled)
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "
SELECT pg_stat_statements.query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;
"
```

### Log Analysis Commands

```bash
# Count errors by type
docker-compose -f docker-compose-unified.yml logs cbam-api 2>&1 | grep -i error | sort | uniq -c | sort -rn

# Find recent errors
docker-compose -f docker-compose-unified.yml logs --since 1h cbam-api 2>&1 | grep -i error

# Extract timestamps of errors
docker-compose -f docker-compose-unified.yml logs cbam-api 2>&1 | grep -i error | awk '{print $1, $2}'

# Find stack traces
docker-compose -f docker-compose-unified.yml logs cbam-api 2>&1 | grep -A 20 "Traceback"
```

---

## Debugging Commands

### Container Debugging

```bash
# Enter container shell
docker-compose -f docker-compose-unified.yml exec cbam-api /bin/bash

# Or for Alpine-based containers
docker-compose -f docker-compose-unified.yml exec cbam-api /bin/sh

# Check environment variables
docker-compose -f docker-compose-unified.yml exec cbam-api env | sort

# Check running processes
docker-compose -f docker-compose-unified.yml exec cbam-api ps aux

# Check network from inside container
docker-compose -f docker-compose-unified.yml exec cbam-api curl http://postgres:5432
```

### Database Debugging

```bash
# Connect to PostgreSQL
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform

# Useful queries:
# Active connections
SELECT * FROM pg_stat_activity WHERE state = 'active';

# Long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query
FROM pg_stat_activity
WHERE state != 'idle'
AND now() - pg_stat_activity.query_start > interval '1 minute';

# Table sizes
SELECT schemaname, relname, pg_size_pretty(pg_total_relation_size(relid))
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;

# Lock analysis
SELECT * FROM pg_locks WHERE NOT granted;
```

### Network Debugging

```bash
# Check network
docker network inspect greenlang-platform

# Test connectivity between containers
docker-compose -f docker-compose-unified.yml exec cbam-api ping -c 3 postgres
docker-compose -f docker-compose-unified.yml exec cbam-api ping -c 3 redis

# DNS resolution
docker-compose -f docker-compose-unified.yml exec cbam-api nslookup postgres

# Port connectivity
docker-compose -f docker-compose-unified.yml exec cbam-api nc -zv postgres 5432
docker-compose -f docker-compose-unified.yml exec cbam-api nc -zv redis 6379
```

### Resource Debugging

```bash
# Container resource usage
docker stats

# Detailed container inspection
docker inspect cbam-api | jq '.[0].State'
docker inspect cbam-api | jq '.[0].HostConfig.Memory'

# Volume inspection
docker volume inspect cbam-data

# Check for OOM kills
dmesg | grep -i "out of memory"
journalctl -k | grep -i "out of memory"
```

---

## Performance Issues

### Diagnosing Performance

```bash
# Response time test
for i in {1..10}; do
  time curl -s http://localhost:8001/health > /dev/null
done

# Load test (requires hey or ab)
hey -n 1000 -c 10 http://localhost:8001/health

# Database performance
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT
    schemaname,
    relname,
    seq_scan,
    seq_tup_read,
    idx_scan,
    idx_tup_fetch
FROM pg_stat_user_tables
ORDER BY seq_scan DESC
LIMIT 10;
"
```

### Performance Optimization

```bash
# PostgreSQL tuning
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -c "
-- Show current settings
SHOW shared_buffers;
SHOW effective_cache_size;
SHOW work_mem;
"

# Redis optimization
docker-compose -f docker-compose-unified.yml exec redis redis-cli CONFIG GET maxmemory
docker-compose -f docker-compose-unified.yml exec redis redis-cli CONFIG GET maxmemory-policy

# Check for missing indexes
docker-compose -f docker-compose-unified.yml exec postgres psql -U greenlang_admin -d greenlang_platform -c "
SELECT
    schemaname,
    relname,
    seq_scan - idx_scan AS too_much_seq,
    CASE WHEN seq_scan - idx_scan > 0 THEN 'Missing Index?' ELSE 'OK' END AS status
FROM pg_stat_user_tables
WHERE seq_scan - idx_scan > 1000
ORDER BY too_much_seq DESC;
"
```

---

## Support Contacts

### Internal Support

| Team | Contact | Scope |
|------|---------|-------|
| **Platform Operations** | platform-ops@greenlang.io | Infrastructure, deployment |
| **Database Team** | dba@greenlang.io | PostgreSQL, Redis issues |
| **Security Team** | security@greenlang.io | Security incidents |
| **On-Call Engineer** | oncall@greenlang.io | Urgent issues (24/7) |

### Escalation Path

```
Level 1: On-Call Engineer
         - Response: 15 minutes
         - Contact: PagerDuty / Slack #platform-incidents

Level 2: Team Lead
         - Response: 30 minutes
         - Contact: Slack DM or phone

Level 3: Engineering Manager
         - Response: 1 hour
         - Contact: Phone

Level 4: CTO (P1 incidents only)
         - Response: 2 hours
         - Contact: Phone
```

### External Vendor Support

| Vendor | Support Contact | Account |
|--------|-----------------|---------|
| **AWS** | AWS Support Console | Enterprise Support |
| **Anthropic** | support@anthropic.com | API Support |
| **OpenAI** | support@openai.com | API Support |
| **Docker** | support@docker.com | Business Support |

### Useful Resources

- **Platform Documentation:** `deployment/docs/`
- **API Documentation:** http://localhost:8001/docs (CBAM), http://localhost:8002/docs (CSRD), http://localhost:8000/docs (VCCI)
- **Grafana Dashboards:** http://localhost:3000
- **Prometheus Metrics:** http://localhost:9090
- **RabbitMQ Management:** http://localhost:15672

---

## Quick Reference Card

```
+==============================================================================+
|                    TROUBLESHOOTING QUICK REFERENCE                           |
+==============================================================================+

HEALTH CHECKS
  ./deploy.sh health                          - Platform health
  curl localhost:8001/health                  - CBAM health
  curl localhost:8002/health                  - CSRD health
  curl localhost:8000/health/live             - VCCI health

LOGS
  docker-compose -f docker-compose-unified.yml logs -f <service>
  docker-compose -f docker-compose-unified.yml logs --tail=100 <service>

RESTART
  docker-compose -f docker-compose-unified.yml restart <service>
  docker-compose -f docker-compose-unified.yml up -d --force-recreate <service>

DATABASE
  docker-compose exec postgres psql -U greenlang_admin -d greenlang_platform

REDIS
  docker-compose exec redis redis-cli -a $REDIS_PASSWORD

COMMON FIXES
  - Port in use:     lsof -i :<port> && kill -9 <pid>
  - OOM:             Increase memory in docker-compose
  - DB connections:  Restart applications
  - Disk full:       docker system prune -a -f

+==============================================================================+
```

---

**Document Owner:** Platform Operations Team
**Review Cycle:** Monthly
**Next Review:** 2026-03-03
