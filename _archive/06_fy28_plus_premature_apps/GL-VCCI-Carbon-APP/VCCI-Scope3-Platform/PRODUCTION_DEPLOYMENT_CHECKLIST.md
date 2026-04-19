# GL-VCCI Production Deployment Checklist
## Scope 3 Carbon Intelligence Platform v2.0

**Version:** 2.0.0
**Date:** November 8, 2025
**Environment:** Production
**Deployment Type:** Initial GA Launch

---

## Overview

This comprehensive checklist ensures all critical aspects of production deployment are validated before launching GL-VCCI to General Availability. Complete all items in order and verify each checkpoint.

**Total Items:** 67
**Estimated Time:** 8-12 hours (full production deployment)

---

## 1. PRE-DEPLOYMENT PREPARATION (10 items)

### 1.1 Infrastructure Readiness
- [ ] **1.1.1** Kubernetes cluster provisioned (EKS/GKE/AKS)
  - Cluster version: 1.28+
  - Node pools configured (3 AZs minimum)
  - Auto-scaling enabled (min: 3, max: 50 nodes)
  - Node taints and labels configured

- [ ] **1.1.2** Managed database provisioned (PostgreSQL 15+)
  - Multi-AZ deployment enabled
  - Automated backups configured (daily, 30-day retention)
  - Point-in-time recovery enabled
  - Read replicas configured (2 minimum)
  - Connection pooling configured (PgBouncer)

- [ ] **1.1.3** Managed Redis cluster provisioned (Redis 7+)
  - Cluster mode enabled (3 shards minimum)
  - Multi-AZ replication enabled
  - AOF persistence configured
  - Memory eviction policy: allkeys-lru
  - Max memory: 16GB minimum per shard

- [ ] **1.1.4** Object storage configured (S3/GCS/Azure Blob)
  - Lifecycle policies configured
  - Versioning enabled
  - Server-side encryption enabled (AES-256 or KMS)
  - Cross-region replication configured
  - Bucket policies locked down (least privilege)

- [ ] **1.1.5** Load balancer configured
  - Application Load Balancer (ALB/GLB/Azure LB)
  - SSL/TLS certificates installed (wildcard cert)
  - Health check endpoints configured
  - Connection draining enabled (300s)
  - Cross-zone load balancing enabled

### 1.2 Security & Access
- [ ] **1.2.1** SSL/TLS certificates validated
  - Wildcard certificate for *.vcci.greenlang.io
  - Certificate expiration > 60 days
  - Certificate chain complete
  - OCSP stapling enabled

- [ ] **1.2.2** Secrets management configured
  - AWS Secrets Manager / GCP Secret Manager / Azure Key Vault
  - Database credentials rotated
  - API keys generated and stored
  - Service account keys secured
  - External API credentials (Ecoinvent, DESNZ, EPA)

- [ ] **1.2.3** IAM roles and policies configured
  - Service accounts created (principle of least privilege)
  - Pod-level IAM roles (IRSA/Workload Identity)
  - Cross-service access policies defined
  - Audit logging enabled for all role assumptions

- [ ] **1.2.4** Network security groups configured
  - VPC/VNet configured with private subnets
  - Security groups locked down (port 443 only from ALB)
  - Database accessible only from cluster
  - Redis accessible only from cluster
  - No public IPs on worker nodes

- [ ] **1.2.5** Container registry access configured
  - ECR/GCR/ACR authenticated
  - Image pull secrets created
  - Vulnerability scanning enabled
  - Image signing configured (Notary/Cosign)

---

## 2. DATABASE PREPARATION (8 items)

### 2.1 Schema & Migrations
- [ ] **2.1.1** Database created and initialized
  - Database: `vcci_production`
  - Character set: UTF-8
  - Collation: en_US.UTF-8
  - Extensions: pgcrypto, uuid-ossp, pg_trgm, btree_gin

- [ ] **2.1.2** Alembic migration history verified
  - All migrations applied (check `alembic_version` table)
  - No pending migrations (run `alembic current`)
  - Migration checksum validated

- [ ] **2.1.3** Database indexes created
  - 47 indexes verified (check `database/indexes.sql`)
  - Partial indexes for active records
  - GIN indexes for JSONB columns
  - Full-text search indexes

- [ ] **2.1.4** Database partitions created (if applicable)
  - Emissions table partitioned by reporting_period (quarterly)
  - Audit logs partitioned by month
  - Partition maintenance scheduled

### 2.2 Data & Performance
- [ ] **2.2.1** Reference data loaded
  - Emission factors (Ecoinvent, DESNZ, EPA) - 150,000+ factors
  - Industry mappings (NAICS, ISIC) - 10,000+ codes
  - Country codes (ISO 3166)
  - Currency codes (ISO 4217)
  - GWP values (IPCC AR6)

- [ ] **2.2.2** Database performance tuned
  - Connection pool sized (min: 10, max: 100 per pod)
  - Shared buffers: 25% of RAM
  - Effective cache size: 75% of RAM
  - Work mem: 64MB
  - Maintenance work mem: 2GB
  - Auto-vacuum configured (scale factor: 0.05)

- [ ] **2.2.3** Database monitoring configured
  - CloudWatch/Stackdriver metrics enabled
  - Slow query log enabled (> 1 second)
  - Connection count alerts (> 80% max)
  - Replication lag alerts (> 5 seconds)
  - Disk space alerts (> 80% used)

- [ ] **2.2.4** Database backup verified
  - Automated backup schedule confirmed (daily 2 AM UTC)
  - Backup retention: 30 days
  - Point-in-time recovery tested
  - Restore procedure documented
  - Backup encryption verified

---

## 3. APPLICATION DEPLOYMENT (12 items)

### 3.1 Container Images
- [ ] **3.1.1** Docker images built and tagged
  - Backend API: `vcci-backend:2.0.0`
  - Frontend: `vcci-frontend:2.0.0`
  - Worker: `vcci-worker:2.0.0`
  - All images tagged with git SHA

- [ ] **3.1.2** Images scanned for vulnerabilities
  - Trivy scan passed (0 critical, 0 high)
  - Snyk scan passed
  - No known CVEs in base images

- [ ] **3.1.3** Images pushed to registry
  - All images in production registry
  - Image digests recorded
  - Image pull secrets validated

### 3.2 Kubernetes Manifests
- [ ] **3.2.1** Namespace created
  - Namespace: `vcci-production`
  - Resource quotas applied
  - Limit ranges configured
  - Network policies applied

- [ ] **3.2.2** ConfigMaps deployed
  - Application config
  - Nginx config
  - Logging config
  - All configs validated (no syntax errors)

- [ ] **3.2.3** Secrets deployed
  - Database credentials
  - Redis credentials
  - API keys (Ecoinvent, DESNZ, EPA)
  - JWT signing keys
  - Encryption keys (AES-256)

- [ ] **3.2.4** Deployments applied
  - Backend API deployment (replicas: 3)
  - Frontend deployment (replicas: 2)
  - Worker deployment (replicas: 3)
  - All pods in Running state
  - No CrashLoopBackOff errors

- [ ] **3.2.5** Services configured
  - Backend service (ClusterIP)
  - Frontend service (ClusterIP)
  - All services have endpoints

- [ ] **3.2.6** Ingress configured
  - Ingress controller deployed (Nginx/Traefik)
  - TLS termination configured
  - Path routing configured
  - Rate limiting configured (100 req/min per IP)

- [ ] **3.2.7** Horizontal Pod Autoscaling (HPA) configured
  - Backend HPA: CPU > 70%, min: 3, max: 20
  - Worker HPA: CPU > 80%, min: 3, max: 50
  - Memory-based scaling: > 85%

- [ ] **3.2.8** Pod Disruption Budgets (PDB) configured
  - Backend PDB: minAvailable: 2
  - Frontend PDB: minAvailable: 1
  - Worker PDB: maxUnavailable: 1

- [ ] **3.2.9** Resource requests and limits set
  - Backend: requests (500m CPU, 1Gi RAM), limits (2 CPU, 4Gi RAM)
  - Worker: requests (1 CPU, 2Gi RAM), limits (4 CPU, 8Gi RAM)
  - Frontend: requests (100m CPU, 256Mi RAM), limits (500m CPU, 1Gi RAM)

---

## 4. CONFIGURATION VALIDATION (9 items)

### 4.1 Environment Variables
- [ ] **4.1.1** Database connection strings validated
  - `DATABASE_URL` points to production database
  - Connection pooling configured
  - SSL mode: require

- [ ] **4.1.2** Redis connection strings validated
  - `REDIS_URL` points to production cluster
  - TLS enabled
  - Password authentication configured

- [ ] **4.1.3** External API credentials configured
  - Ecoinvent API key valid (expiration > 90 days)
  - DESNZ API key valid
  - EPA API key valid
  - Rate limits documented

- [ ] **4.1.4** Feature flags configured
  - Production flags set (debug: false, logging: info)
  - ML inference enabled
  - Email notifications enabled
  - Background jobs enabled

- [ ] **4.1.5** CORS settings configured
  - Allowed origins: https://vcci.greenlang.io
  - Allowed methods: GET, POST, PUT, DELETE, PATCH
  - Credentials allowed: true

### 4.2 Application Settings
- [ ] **4.2.1** Logging configured
  - Log level: INFO (ERROR for production)
  - Structured logging (JSON format)
  - Log aggregation (CloudWatch/Stackdriver/ELK)
  - Log retention: 90 days

- [ ] **4.2.2** Metrics collection configured
  - Prometheus metrics endpoint: `/metrics`
  - Custom business metrics enabled
  - Metric retention: 15 days

- [ ] **4.2.3** Rate limiting configured
  - API rate limits: 1000 req/hour per tenant
  - Burst limits: 100 req/minute
  - Upload rate limits: 100 MB/hour

- [ ] **4.2.4** Session management configured
  - Session timeout: 8 hours
  - JWT expiration: 1 hour
  - Refresh token expiration: 7 days
  - Session storage: Redis

---

## 5. MONITORING & OBSERVABILITY (10 items)

### 5.1 Metrics & Dashboards
- [ ] **5.1.1** Prometheus deployed
  - Scraping interval: 30s
  - Retention: 15 days
  - Persistent volume: 100 GB
  - High availability: 2 replicas

- [ ] **5.1.2** Grafana deployed
  - Admin credentials secured
  - Prometheus data source configured
  - Production dashboard imported (grafana-vcci-dashboard.json)
  - Alerting configured

- [ ] **5.1.3** Key metrics validated
  - API request rate (req/s)
  - API latency (p50, p95, p99)
  - Error rate (5xx errors)
  - Database connection pool usage
  - Redis memory usage
  - Worker queue depth

- [ ] **5.1.4** Business metrics configured
  - Active tenants
  - Total suppliers processed
  - Emissions calculations per day
  - Data ingestion volume (MB/day)
  - Report generation count

### 5.2 Logging & Tracing
- [ ] **5.2.1** Centralized logging configured
  - ELK Stack / CloudWatch Logs / Stackdriver
  - Log aggregation from all pods
  - Log search and filtering enabled
  - Log retention: 90 days

- [ ] **5.2.2** Application logs validated
  - Structured JSON logs
  - Request IDs in all logs
  - Error stack traces captured
  - Sensitive data masked (PII, credentials)

- [ ] **5.2.3** Distributed tracing configured (optional)
  - Jaeger / OpenTelemetry
  - Trace sampling: 10%
  - Trace retention: 7 days

### 5.3 Alerting
- [ ] **5.3.1** Alert rules configured
  - API error rate > 1% (critical)
  - API latency p99 > 2s (warning)
  - Database CPU > 80% (warning)
  - Database disk > 85% (critical)
  - Pod restart rate > 5/hour (warning)
  - Worker queue depth > 10,000 (warning)

- [ ] **5.3.2** Alert channels configured
  - PagerDuty / Opsgenie / Slack
  - On-call rotation defined
  - Escalation policy configured

- [ ] **5.3.3** Synthetic monitoring configured
  - Uptime checks (1-minute interval)
  - API health checks
  - End-to-end workflow checks (every 15 minutes)
  - Alert on 2 consecutive failures

---

## 6. SECURITY VALIDATION (8 items)

### 6.1 Authentication & Authorization
- [ ] **6.1.1** OAuth 2.0 / OIDC configured
  - Identity provider integrated (Okta/Auth0/Azure AD)
  - Tenant isolation verified
  - MFA enabled for admin accounts

- [ ] **6.1.2** RBAC policies configured
  - 6 roles defined (Admin, Manager, Analyst, Supplier, Auditor, Guest)
  - Role assignments tested
  - Permission boundaries enforced

- [ ] **6.1.3** API authentication validated
  - JWT token validation working
  - API key authentication working
  - Token refresh mechanism working
  - Expired token handling verified

### 6.2 Data Protection
- [ ] **6.2.1** Encryption at rest verified
  - Database encryption enabled
  - Object storage encryption enabled (S3-SSE/KMS)
  - Redis encryption enabled

- [ ] **6.2.2** Encryption in transit verified
  - TLS 1.3 for all external connections
  - TLS 1.2 minimum for internal connections
  - Certificate pinning configured

- [ ] **6.2.3** Data masking configured
  - PII fields masked in logs
  - Sensitive data redacted in error messages
  - Data export anonymization working

### 6.3 Compliance & Auditing
- [ ] **6.3.1** Audit logging enabled
  - All authentication events logged
  - All data modifications logged
  - Admin actions logged
  - Audit log retention: 7 years (SOC 2 requirement)

- [ ] **6.3.2** SOC 2 controls validated
  - Access controls enforced
  - Change management process followed
  - Incident response plan documented
  - Business continuity plan documented

---

## 7. PERFORMANCE VALIDATION (6 items)

### 7.1 Load Testing
- [ ] **7.1.1** API load testing completed
  - 10,000 concurrent users supported
  - API latency p95 < 500ms (target met)
  - API latency p99 < 1000ms (target met)
  - 0% error rate under normal load

- [ ] **7.1.2** Database load testing completed
  - 1,000 queries/second sustained
  - No connection pool exhaustion
  - No lock contention issues
  - Query performance within SLA (95% < 100ms)

- [ ] **7.1.3** Worker throughput validated
  - 10,000 suppliers processed per hour
  - 1M+ emission calculations per day
  - Queue processing latency < 5 seconds
  - No task failures under load

### 7.2 Scalability
- [ ] **7.2.1** Horizontal scaling validated
  - Auto-scaling triggered at 70% CPU
  - Scale-up time < 2 minutes
  - Scale-down graceful (5-minute cooldown)
  - No connection drops during scaling

- [ ] **7.2.2** Database scaling validated
  - Read replicas distributing load
  - Replication lag < 1 second
  - Failover time < 30 seconds

- [ ] **7.2.3** Cache hit rate optimized
  - Redis hit rate > 95%
  - Cache eviction rate < 1%
  - Cache memory usage < 80%

---

## 8. DISASTER RECOVERY (6 items)

### 8.1 Backup Procedures
- [ ] **8.1.1** Database backup procedure tested
  - Full backup completed successfully
  - Incremental backups working
  - Backup integrity verified (checksum)
  - Restore time < 1 hour (RPO: 15 minutes, RTO: 1 hour)

- [ ] **8.1.2** Object storage backup configured
  - Cross-region replication enabled
  - Versioning enabled (30-day retention)
  - Lifecycle policies configured

- [ ] **8.1.3** Configuration backup created
  - Kubernetes manifests versioned (Git)
  - ConfigMaps exported
  - Secrets documented (not exported)
  - Infrastructure as Code (Terraform) versioned

### 8.2 Recovery Procedures
- [ ] **8.2.1** Database restore procedure tested
  - Point-in-time recovery tested
  - Restore completed in < 1 hour
  - Data integrity verified post-restore

- [ ] **8.2.2** Application rollback procedure tested
  - Previous version deployable
  - Database migration rollback tested
  - Zero-downtime rollback verified

- [ ] **8.2.3** Disaster recovery plan documented
  - DR runbook created (docs/runbooks/DATA_RECOVERY.md)
  - RTO defined: 1 hour
  - RPO defined: 15 minutes
  - DR drill scheduled (quarterly)

---

## 9. INTEGRATION TESTING (5 items)

### 9.1 External Integrations
- [ ] **9.1.1** ERP connectors validated
  - SAP S/4HANA connector working (test tenant)
  - Oracle Fusion connector working (test tenant)
  - Workday connector working (test tenant)
  - Data synchronization verified

- [ ] **9.1.2** Emission factor APIs validated
  - Ecoinvent API responding (< 100ms)
  - DESNZ API responding (< 200ms)
  - EPA API responding (< 200ms)
  - Fallback to proxy factors working

- [ ] **9.1.3** Email service configured
  - SMTP/SendGrid/SES configured
  - Email templates validated
  - Test emails sent successfully
  - SPF/DKIM/DMARC records configured

### 9.2 End-to-End Workflows
- [ ] **9.2.1** Supplier onboarding workflow tested
  - Supplier registration complete
  - Email invitation sent
  - Data upload successful
  - Dashboard accessible

- [ ] **9.2.2** Emissions calculation workflow tested
  - Data ingestion successful (CSV/Excel/JSON)
  - Category 1 calculation complete (< 5 seconds)
  - Category 4 calculation complete (< 3 seconds)
  - Category 6 calculation complete (< 3 seconds)
  - Report generation successful (< 10 seconds)

---

## 10. DOCUMENTATION & TRAINING (8 items)

### 10.1 Documentation Completeness
- [ ] **10.1.1** API documentation published
  - OpenAPI spec available (/docs endpoint)
  - Postman collection available
  - Code examples (Python, JavaScript) available
  - Rate limits documented

- [ ] **10.1.2** User guides published
  - Getting Started Guide
  - Dashboard Usage Guide
  - Data Upload Guide
  - Reporting Guide
  - Supplier Portal Guide

- [ ] **10.1.3** Admin documentation published
  - Deployment Guide
  - Operations Guide
  - User Management Guide
  - Tenant Management Guide
  - Security Guide

- [ ] **10.1.4** Runbooks created
  - Incident Response
  - Database Failover
  - Scaling Operations
  - Certificate Renewal
  - Data Recovery
  - Performance Tuning
  - Security Incident
  - Deployment Rollback
  - Capacity Planning

### 10.2 Knowledge Transfer
- [ ] **10.2.1** Operations team trained
  - 2-day training completed
  - Runbooks reviewed
  - Access to production granted
  - On-call rotation established

- [ ] **10.2.2** Support team trained
  - User guides reviewed
  - Common issues documented
  - Support ticket system configured (Zendesk/Jira)

- [ ] **10.2.3** Customer success team trained
  - Sales playbook reviewed
  - Demo environment accessible
  - Onboarding checklist created

- [ ] **10.2.4** Executive dashboard configured
  - Business metrics visible
  - Monthly reporting automated
  - SLA tracking configured

---

## 11. FINAL VALIDATION (5 items)

### 11.1 Pre-Launch Checklist
- [ ] **11.1.1** Smoke tests passed
  - All health check endpoints responding
  - Sample API requests successful
  - Frontend loads without errors
  - Background jobs running

- [ ] **11.1.2** Security scan passed
  - OWASP ZAP scan passed (0 high vulnerabilities)
  - Penetration test completed (3rd party)
  - Security findings remediated

- [ ] **11.1.3** Performance benchmarks met
  - API latency p95 < 500ms ✅
  - Database query p95 < 100ms ✅
  - Worker throughput > 10,000 suppliers/hour ✅
  - Frontend load time < 2 seconds ✅

- [ ] **11.1.4** Compliance validated
  - SOC 2 Type II certification active
  - GDPR compliance verified
  - ISO 27001 controls documented

- [ ] **11.1.5** Change management approved
  - Change request submitted (CAB)
  - Deployment window scheduled
  - Rollback plan approved
  - Stakeholders notified

---

## 12. POST-DEPLOYMENT (5 items)

### 12.1 Launch Activities
- [ ] **12.1.1** DNS cutover completed
  - Production domain pointed to load balancer
  - DNS propagation verified (nslookup)
  - SSL certificate validated (browser)

- [ ] **12.1.2** Monitoring alerts validated
  - All alerts firing correctly
  - Alert routing working (PagerDuty/Slack)
  - No false positives

- [ ] **12.1.3** Initial traffic validated
  - First production requests successful
  - No errors in logs
  - Metrics flowing to Grafana
  - All pods healthy

### 12.2 Post-Launch Monitoring
- [ ] **12.2.1** 24-hour soak test
  - System stable for 24 hours
  - No memory leaks detected
  - No performance degradation
  - No unexpected errors

- [ ] **12.2.2** Production readiness report published
  - Final metrics documented
  - Known issues documented
  - Success criteria met
  - Sign-off obtained (CTO, VP Engineering)

---

## SIGN-OFF

### Deployment Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **CTO** | | | |
| **VP Engineering** | | | |
| **Security Lead** | | | |
| **DevOps Lead** | | | |
| **QA Lead** | | | |

### Deployment Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| **Deployment Time** | < 8 hours | _____ | ⬜ |
| **Downtime** | 0 minutes | _____ | ⬜ |
| **Failed Health Checks** | 0 | _____ | ⬜ |
| **Post-Deploy Errors** | 0 | _____ | ⬜ |
| **Performance Degradation** | 0% | _____ | ⬜ |

---

## ROLLBACK CRITERIA

If ANY of the following conditions are met, initiate rollback immediately:

1. **Error Rate > 1%** for more than 5 minutes
2. **API Latency p95 > 2 seconds** for more than 10 minutes
3. **Database CPU > 90%** for more than 5 minutes
4. **More than 20% of pods in CrashLoopBackOff**
5. **Critical security vulnerability discovered**
6. **Data corruption detected**
7. **Unable to meet SLA commitments**

**Rollback Procedure:** See `docs/runbooks/DEPLOYMENT_ROLLBACK.md`

---

## SUPPORT CONTACTS

| Team | Contact | Escalation |
|------|---------|------------|
| **DevOps** | devops@greenlang.io | oncall-devops@greenlang.io |
| **Security** | security@greenlang.io | CISO@greenlang.io |
| **Engineering** | engineering@greenlang.io | VP-Engineering@greenlang.io |
| **Product** | product@greenlang.io | CPO@greenlang.io |

**Emergency Hotline:** +1-XXX-XXX-XXXX (24/7)

---

**Checklist Version:** 2.0.0
**Last Updated:** November 8, 2025
**Next Review:** Post-deployment (within 7 days)
