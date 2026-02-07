# POL-012: Backup and Recovery Policy

**Document Control**

| Attribute | Value |
|-----------|-------|
| Document ID | POL-012 |
| Version | 1.0 |
| Classification | Confidential |
| Policy Tier | Tier 3 - Compliance |
| Owner | Director of Infrastructure |
| Approved By | Chief Technology Officer (CTO) |
| Effective Date | 2026-02-06 |
| Last Review | 2026-02-06 |
| Next Review | 2027-02-06 |

---

## 1. Purpose

This policy establishes requirements for the backup and recovery of GreenLang Climate OS data and systems. The purpose is to ensure business continuity, protect against data loss, and enable timely recovery from system failures, security incidents, or disasters.

Effective backup and recovery capabilities are essential for maintaining customer trust, meeting regulatory obligations, and ensuring the availability of critical climate reporting services. This policy defines what must be backed up, how often, where backups are stored, and how recovery operations are conducted.

This policy supports compliance with SOC 2 Type II (Availability criteria), ISO 27001:2022 (A.8.13 Information backup), and regulatory requirements for data retention and disaster recovery.

---

## 2. Scope

This policy applies to:

- **Systems**: All production systems, databases, file storage, and configuration management
- **Data**: Customer data, application data, system configurations, audit logs, and compliance records
- **Environments**: Production, staging, and disaster recovery environments
- **Personnel**: Infrastructure teams, database administrators, security operations, and on-call engineers
- **Locations**: AWS cloud infrastructure across all regions and availability zones

### 2.1 Out of Scope

- Development and testing environments (covered by separate development backup guidelines)
- Personal workstations (covered by endpoint management policy)
- Third-party SaaS applications (governed by vendor SLAs)

---

## 3. Policy Statement

GreenLang maintains comprehensive backup and recovery capabilities to protect against data loss and ensure business continuity. All critical systems and data must be backed up according to defined schedules, stored securely, and tested regularly to verify recoverability.

### 3.1 Backup Scope and Schedule

#### 3.1.1 Database Backups

| Database Type | Backup Type | Frequency | Retention | Technology |
|--------------|-------------|-----------|-----------|------------|
| **PostgreSQL (Aurora)** | Automated snapshot | Daily (full) | 35 days | AWS RDS automated backups |
| **PostgreSQL (Aurora)** | Continuous | Point-in-time (5-min granularity) | 35 days | AWS PITR |
| **PostgreSQL (Aurora)** | Manual snapshot | Weekly | 90 days | AWS manual snapshots |
| **TimescaleDB** | Full backup | Daily | 35 days | pgBackRest |
| **TimescaleDB** | Incremental | Hourly | 7 days | pgBackRest |
| **Redis (ElastiCache)** | RDB snapshot | Daily | 35 days | AWS ElastiCache |
| **Redis (ElastiCache)** | AOF persistence | Continuous | N/A (in-memory) | Redis AOF |

#### 3.1.2 File System and Object Storage Backups

| Storage Type | Backup Type | Frequency | Retention | Technology |
|--------------|-------------|-----------|-----------|------------|
| **S3 Buckets** | Versioning | On change | Per lifecycle policy | S3 Versioning |
| **S3 Buckets** | Cross-region replication | Continuous | Same as source | S3 CRR |
| **EFS Volumes** | AWS Backup | Daily | 35 days | AWS Backup |
| **EBS Volumes** | Snapshot | Daily | 35 days | AWS Data Lifecycle Manager |
| **Container Images** | Replication | On push | 90 days | ECR replication |

#### 3.1.3 Configuration Backups

| Configuration Type | Backup Method | Frequency | Retention | Storage |
|--------------------|---------------|-----------|-----------|---------|
| **Infrastructure as Code** | Git versioning | On change | Indefinite | GitHub |
| **Kubernetes manifests** | Git versioning | On change | Indefinite | GitHub |
| **Secrets (Vault)** | Vault backup | Daily + on change | 90 days | Encrypted S3 |
| **DNS records** | Export | Daily | 90 days | Encrypted S3 |
| **IAM policies** | AWS Config | On change | 90 days | AWS Config |
| **Database schemas** | Schema dump | Daily | 90 days | Encrypted S3 |

#### 3.1.4 Log Backups

| Log Type | Backup Method | Frequency | Retention | Storage |
|----------|---------------|-----------|-----------|---------|
| **Application logs** | Loki ingestion | Continuous | 30 days (hot), 365 days (cold) | Loki + S3 |
| **Audit logs** | Loki ingestion | Continuous | 365 days | Loki + S3 |
| **Security logs** | SIEM ingestion | Continuous | 365 days | SIEM + S3 |
| **CloudTrail** | AWS native | Continuous | 365 days | S3 with Object Lock |

### 3.2 Retention Periods

#### 3.2.1 Standard Retention

| Data Category | Production Retention | Archive Retention | Total |
|--------------|---------------------|-------------------|-------|
| **Operational backups** | 35 days | N/A | 35 days |
| **Weekly snapshots** | 90 days | N/A | 90 days |
| **Monthly snapshots** | 12 months | N/A | 12 months |
| **Annual snapshots** | N/A | 7 years | 7 years |

#### 3.2.2 Compliance-Driven Retention

| Requirement | Data Type | Retention Period | Rationale |
|-------------|-----------|------------------|-----------|
| **CBAM Compliance** | Emissions data, reports | 7 years | EU CBAM regulation |
| **Financial Records** | Invoices, transactions | 7 years | Tax and accounting |
| **Audit Trails** | Security and access logs | 7 years | SOC 2 / ISO 27001 |
| **Customer Contracts** | Agreements, amendments | 10 years post-termination | Legal requirements |
| **Legal Hold** | Any data under litigation | Until released | Legal counsel directive |

### 3.3 Backup Storage

#### 3.3.1 Primary Backup Storage

| Storage Location | Purpose | Encryption | Access Control |
|-----------------|---------|------------|----------------|
| **Same region, different AZ** | Primary backup | AES-256 (AWS KMS) | IAM roles, least privilege |
| **AWS S3 Standard** | Hot storage (0-30 days) | AES-256-GCM | Bucket policies, VPC endpoints |
| **AWS S3 Standard-IA** | Warm storage (30-90 days) | AES-256-GCM | Same as above |

#### 3.3.2 Secondary Backup Storage (Disaster Recovery)

| Storage Location | Purpose | Encryption | Access Control |
|-----------------|---------|------------|----------------|
| **Different AWS region** | DR backup | AES-256 (regional KMS) | Cross-account access |
| **AWS S3 Glacier** | Cold storage (90+ days) | AES-256-GCM | Vault policies |
| **AWS S3 Glacier Deep Archive** | Archive (1+ years) | AES-256-GCM | Vault policies |

#### 3.3.3 Backup Encryption Requirements

- All backups must be encrypted at rest using AES-256
- Encryption keys managed via AWS KMS with automatic rotation
- Backup encryption keys stored separately from data encryption keys
- Cross-region backups use region-specific KMS keys
- Decryption requires explicit IAM permissions and MFA for sensitive data

### 3.4 Recovery Objectives

#### 3.4.1 Critical Systems

Critical systems directly support customer-facing functionality and regulatory compliance.

| System | RTO | RPO | Priority |
|--------|-----|-----|----------|
| **PostgreSQL (Customer Data)** | 4 hours | 1 hour | P1 |
| **CBAM Reporting Service** | 4 hours | 1 hour | P1 |
| **Authentication Service** | 2 hours | 15 minutes | P1 |
| **API Gateway** | 2 hours | N/A (stateless) | P1 |
| **Redis Cache** | 4 hours | 4 hours (rebuildable) | P1 |
| **S3 Document Storage** | 4 hours | 1 hour | P1 |

#### 3.4.2 Standard Systems

Standard systems support internal operations and non-time-sensitive functions.

| System | RTO | RPO | Priority |
|--------|-----|-----|----------|
| **Internal Tools** | 24 hours | 24 hours | P2 |
| **Development Databases** | 48 hours | 24 hours | P3 |
| **Analytics Platform** | 24 hours | 24 hours | P2 |
| **Monitoring Systems** | 8 hours | 4 hours | P2 |
| **Logging Infrastructure** | 8 hours | 1 hour | P2 |

### 3.5 Restoration Testing

#### 3.5.1 Testing Schedule

| Test Type | Frequency | Scope | Documentation |
|-----------|-----------|-------|---------------|
| **Automated restore verification** | Daily | Sample tables, files | Automated report |
| **Database point-in-time recovery** | Monthly | Full database to test environment | Runbook execution log |
| **Full system recovery** | Quarterly | Complete application stack | DR test report |
| **DR site failover** | Annually | Production failover to DR region | Executive summary |

#### 3.5.2 Testing Procedures

1. **Pre-test Preparation**
   - Notify stakeholders of test window
   - Prepare test environment (isolated from production)
   - Document current backup state and target recovery point

2. **Test Execution**
   - Initiate restore using documented runbook
   - Record actual RTO (time from initiation to service availability)
   - Verify data integrity (record counts, checksums, application tests)
   - Document any issues encountered

3. **Post-test Activities**
   - Clean up test environment
   - Update runbooks based on findings
   - Report results to management
   - Track remediation items

### 3.6 Backup Verification Procedures

#### 3.6.1 Automated Verification

| Check | Frequency | Method | Alert Threshold |
|-------|-----------|--------|-----------------|
| Backup completion | Daily | CloudWatch metrics | Failure |
| Backup size variance | Daily | Compare to baseline | +/- 20% |
| Encryption verification | Weekly | KMS key validation | Any failure |
| Integrity checksum | Weekly | SHA-256 verification | Mismatch |
| Cross-region replication | Daily | S3 replication metrics | Lag > 1 hour |

#### 3.6.2 Manual Verification

- Monthly: Spot-check restore of random backup set
- Quarterly: Verify backup encryption keys are recoverable
- Annually: Full inventory reconciliation against backup catalog

### 3.7 Backup Failure Handling

#### 3.7.1 Alerting Requirements

| Failure Type | Alert Time | Notification Channel | Escalation |
|--------------|------------|---------------------|------------|
| **Backup job failure** | Within 15 minutes | PagerDuty, Slack | On-call engineer |
| **Replication lag > 1 hour** | Within 15 minutes | PagerDuty, Slack | On-call engineer |
| **Storage capacity < 20%** | Within 1 hour | Email, Slack | Infrastructure lead |
| **Encryption key unavailable** | Immediate | PagerDuty | Security + Infrastructure |
| **Multiple consecutive failures** | Within 15 minutes | PagerDuty | Engineering manager |

#### 3.7.2 Response Procedures

1. **Immediate Response** (within 15 minutes)
   - Acknowledge alert
   - Assess scope of failure
   - Initiate manual backup if automated backup failed

2. **Investigation** (within 1 hour)
   - Identify root cause
   - Determine data at risk
   - Implement temporary mitigation

3. **Resolution** (within 4 hours)
   - Fix underlying issue
   - Verify backup success
   - Document incident

4. **Follow-up** (within 24 hours)
   - Post-incident review
   - Update runbooks if needed
   - Implement preventive measures

---

## 4. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **CTO** | Policy approval, DR budget allocation, executive escalation point |
| **Director of Infrastructure** | Policy ownership, backup strategy, vendor management |
| **Database Administrators** | Database backup configuration, restore testing, capacity planning |
| **Platform Engineering** | Infrastructure backup automation, monitoring, tooling |
| **Security Operations** | Backup access control, encryption key management, audit support |
| **On-Call Engineers** | Backup failure response, emergency restores, incident management |
| **Compliance** | Retention requirements, audit evidence, regulatory alignment |

---

## 5. Procedures

### 5.1 Initiating a Data Restore

1. Create ticket in IT service management system specifying:
   - System and data to restore
   - Target recovery point (date/time)
   - Reason for restore
   - Target environment (production/test)
2. Obtain approval (manager for test, director for production)
3. DBA or infrastructure engineer executes restore per runbook
4. Requestor validates restored data
5. Document completion and close ticket

### 5.2 Disaster Recovery Activation

1. Incident commander declares DR event
2. Notify executive leadership and stakeholders
3. Activate DR runbook for affected systems
4. Execute failover to DR region
5. Verify service restoration
6. Communicate status to customers
7. Plan and execute failback when primary recovers

### 5.3 Legal Hold Implementation

1. Legal counsel issues hold notice specifying data scope
2. Compliance team identifies affected backup sets
3. Infrastructure team suspends automatic deletion for affected data
4. Document hold in legal hold register
5. Maintain hold until written release from legal counsel

---

## 6. Exceptions

Exceptions to this policy require:

1. Written business justification with risk assessment
2. Approval from Director of Infrastructure and CISO
3. Compensating controls documented
4. Time-limited exception (maximum 6 months)
5. Quarterly review of exception status

Exception requests must be submitted via the IT governance process.

---

## 7. Enforcement

Violations of this policy may result in:

- System access restrictions
- Performance management actions
- Disciplinary action for willful violations
- Incident reporting per security incident policy

Systems found non-compliant with backup requirements will be flagged for immediate remediation.

---

## 8. Related Documents

| Document ID | Document Name |
|-------------|---------------|
| POL-005 | Data Retention Policy |
| POL-007 | Business Continuity Policy |
| POL-011 | Encryption and Key Management Policy |
| PRD-INFRA-002 | PostgreSQL Database Infrastructure PRD |
| PRD-INFRA-004 | S3 Object Storage PRD |
| RUN-BACKUP-001 | Database Backup Runbook |
| RUN-DR-001 | Disaster Recovery Runbook |
| RUN-RESTORE-001 | Data Restoration Runbook |

---

## 9. Definitions

| Term | Definition |
|------|------------|
| **RTO** | Recovery Time Objective - maximum acceptable time to restore service after disruption |
| **RPO** | Recovery Point Objective - maximum acceptable data loss measured in time |
| **PITR** | Point-in-Time Recovery - ability to restore to any point within retention window |
| **DR** | Disaster Recovery - process of restoring systems after major disruption |
| **Legal Hold** | Requirement to preserve data relevant to litigation or investigation |
| **Hot Storage** | Immediately accessible backup storage |
| **Cold Storage** | Archived backup storage with retrieval delay |
| **Cross-Region Replication** | Automatic copying of data to a different geographic region |

---

## 10. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-06 | Infrastructure Team | Initial policy creation |

---

**Document Classification: Confidential**

*This policy is the property of GreenLang Climate OS. Unauthorized distribution, copying, or disclosure is prohibited.*
