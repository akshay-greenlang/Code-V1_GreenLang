# Citations Audit Compliance

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `CitationsAuditGap` | Critical | Missing version entries detected for citation changes (gap > 1 hour with active writes) |
| `CitationsProvenanceChainBroken` | Critical | SHA-256 provenance hash chain verification failure |
| `CitationsHashIntegrityFailure` | Critical | Citation content hash verification failure |
| `CitationsVerificationFailureSpike` | Warning | >50% of citation verifications failing over 15 minutes |

**Thresholds:**

```promql
# CitationsAuditGap
# Gap detection: write operations occurring but no corresponding version entries
(
  sum(rate(gl_citations_operations_total{operation=~"create|update|delete|verify"}[5m]))
  - sum(rate(gl_citations_version_writes_total[5m]))
) > 1
# sustained for 1 hour

# CitationsProvenanceChainBroken
# Provenance hash chain verification failure
increase(gl_citations_provenance_chain_broken_total[5m]) > 0

# CitationsHashIntegrityFailure
# Content hash integrity verification failure
increase(gl_citations_hash_integrity_failures_total[5m]) > 0

# CitationsVerificationFailureSpike
# Verification rejection rate above 50%
(
  sum(rate(gl_citations_verifications_total{status="rejected"}[15m]))
  / sum(rate(gl_citations_verifications_total[15m]))
) > 0.50
```

---

## Description

These alerts fire when the audit and compliance subsystem of the Citations & Evidence Agent service (AGENT-FOUND-005) detects integrity issues in the version history, provenance chain, content hashes, or verification pipeline. The audit system is a critical component of the zero-hallucination guarantee and is required for SOC 2, ISO 27001, CSRD, CBAM, and other regulatory compliance frameworks.

### How the Audit System Works

The Citations & Evidence Agent maintains a multi-layered audit trail:

1. **Citation Version History** -- Every create, update, delete, verify, and supersede operation on citations is recorded in the `citation_versions` hypertable with:
   - Timestamp (UTC, microsecond precision)
   - Changed-by user identity
   - Change type (create, update, delete, verify, supersede, expire, restore, merge)
   - Changed fields (JSONB diff)
   - Previous and current content hashes (SHA-256 provenance chain)
   - Change reason

2. **Content Hash Integrity** -- Each citation stores a SHA-256 hash of its content (title, authors, source, publication date, key values). This hash is verified on every read to detect data corruption or unauthorized modification.

3. **Provenance Hash Chain** -- Each citation version includes a SHA-256 hash linking to the previous version's hash, creating a tamper-evident chain analogous to a blockchain.

4. **Verification Audit Trail** -- Every verification event is recorded in the `citation_verifications` hypertable with method, status, verifier, and details.

5. **Evidence Package Finalization** -- Finalized evidence packages are immutable and include a SHA-256 package hash computed over all items and their citations.

### Regulatory Requirements

| Framework | Requirement | How Citations Service Satisfies |
|-----------|-------------|-------------------------------|
| **SOC 2** | Change management controls; audit trail for all data source changes | Complete version history with user ID, reason, and timestamp for every citation change |
| **ISO 27001** | A.12.4 Logging and monitoring; A.14.2 Security in development | Provenance hash chain provides tamper-evident logging; all changes are versioned |
| **CSRD** | Assurance over sustainability reporting data and sources | Evidence packages with hash-verified finalization; verification audit trail |
| **CBAM** | Verifiable emission factors and calculation parameters | Every data value traceable to a verified citation with provenance hash |
| **GHG Protocol** | Documentation of all data sources and methodologies | Methodology references linked to verified citations; data source attributions |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Users may not notice audit gaps immediately; hash failures are detected asynchronously |
| **Data Impact** | Critical | Audit trail integrity is compromised; citation changes may not be traceable |
| **SLA Impact** | Medium | Service remains available for citation operations, but audit operations may fail |
| **Revenue Impact** | High | Customers depend on auditable evidence packages for regulatory submissions |
| **Compliance Impact** | Critical | SOC 2, ISO 27001, CSRD, CBAM compliance directly impacted; evidence packages may be rejected |
| **Downstream Impact** | High | Compliance reporting agents cannot generate auditor-ready reports; regulatory submission workflows blocked |

---

## Symptoms

### Audit Gap

- `CitationsAuditGap` alert firing
- `gl_citations_version_writes_total` not incrementing despite active write operations
- Evidence packages showing gaps in citation version sequences
- SOC 2 auditor flagging missing change records

### Provenance Chain Break

- `CitationsProvenanceChainBroken` alert firing
- `gl_citations_provenance_chain_broken_total` incrementing
- Provenance verification returning "hash mismatch" for specific citations
- Logs showing "provenance chain broken at version N" or "hash mismatch"

### Content Hash Failure

- `CitationsHashIntegrityFailure` alert firing
- `gl_citations_hash_integrity_failures_total` incrementing
- Citation content not matching its stored hash
- Possible data corruption or unauthorized modification

### Verification Failure Spike

- `CitationsVerificationFailureSpike` alert firing
- High percentage of citations failing verification (DOI lookup, URL validation)
- External source authority services may be unavailable

---

## Diagnostic Steps

### Step 1: Check Audit Integrity Metrics

```promql
# Version write rate vs operation rate (should be 1:1)
rate(gl_citations_version_writes_total[5m])
rate(gl_citations_operations_total{operation=~"create|update|delete|verify"}[5m])

# Provenance chain break count
rate(gl_citations_provenance_chain_broken_total[5m])

# Content hash integrity failure count
rate(gl_citations_hash_integrity_failures_total[5m])

# Verification failure rate
rate(gl_citations_verifications_total{status="rejected"}[5m])
```

### Step 2: Verify Version History Completeness

```bash
# Check for citations without version entries
kubectl run pg-gaps --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT c.citation_id, c.title, c.updated_at, cv.version_number
   FROM citations_service.citations c
   LEFT JOIN (
     SELECT citation_id, MAX(version_number) as version_number
     FROM citations_service.citation_versions
     GROUP BY citation_id
   ) cv ON c.citation_id = cv.citation_id
   WHERE cv.citation_id IS NULL
   ORDER BY c.updated_at DESC
   LIMIT 20;"
```

### Step 3: Verify Provenance Hash Chain

```bash
# Check for hash chain breaks in version history
kubectl run pg-hash-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "WITH versioned AS (
    SELECT citation_id, version_number, current_hash, previous_hash,
           LAG(current_hash) OVER (PARTITION BY citation_id ORDER BY version_number) as expected_previous
    FROM citations_service.citation_versions
  )
  SELECT citation_id, version_number, current_hash, previous_hash, expected_previous
  FROM versioned
  WHERE version_number > 1
    AND previous_hash != expected_previous
  ORDER BY citation_id, version_number
  LIMIT 20;"
```

### Step 4: Check Content Hash Integrity

```bash
# Check for citations where stored content hash might be stale
kubectl run pg-content-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT citation_id, title, content_hash, updated_at
   FROM citations_service.citations
   WHERE content_hash IS NOT NULL
     AND updated_at > NOW() - INTERVAL '24 hours'
   ORDER BY updated_at DESC
   LIMIT 20;"
```

### Step 5: Check Verification Pipeline

```bash
# Check recent verification results
kubectl run pg-verify --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT status, verification_method, COUNT(*), MAX(timestamp) as latest
   FROM citations_service.citation_verifications
   WHERE timestamp > NOW() - INTERVAL '24 hours'
   GROUP BY status, verification_method
   ORDER BY COUNT(*) DESC;"
```

---

## Resolution Steps

### Option 1: Repair Audit Gaps

If version entries are missing but citation data is intact, reconstruct the missing entries from the citation state.

```bash
# Step 1: Identify the gap period
kubectl run pg-gap-period --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT
    COUNT(*) as citations_without_versions,
    MIN(c.updated_at) as earliest_gap,
    MAX(c.updated_at) as latest_gap
   FROM citations_service.citations c
   LEFT JOIN (
     SELECT DISTINCT citation_id FROM citations_service.citation_versions
   ) cv ON c.citation_id = cv.citation_id
   WHERE cv.citation_id IS NULL;"

# Step 2: Use the admin API to repair
curl -X POST http://localhost:8080/v1/citations/admin/repair-audit-gaps \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "dry_run": true
  }' | python3 -m json.tool
```

### Option 2: Rebuild Provenance Hash Chain

If the provenance chain is broken, rebuild from the version history.

```bash
curl -X POST http://localhost:8080/v1/citations/admin/rebuild-provenance \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "scope": "broken_only",
    "dry_run": true
  }' | python3 -m json.tool
```

### Option 3: Recompute Content Hashes

If content hashes are stale, recompute them from current citation data.

```bash
curl -X POST http://localhost:8080/v1/citations/admin/recompute-hashes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "scope": "mismatched_only",
    "dry_run": true
  }' | python3 -m json.tool
```

### Option 4: Retry Failed Verifications

If verification failures are due to transient external service issues, retry.

```bash
curl -X POST http://localhost:8080/v1/citations/admin/retry-verifications \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "status": "rejected",
    "since": "2026-02-07T00:00:00Z",
    "dry_run": true
  }' | python3 -m json.tool
```

### Option 5: Generate Compliance Export

When an auditor requests citation audit evidence:

```bash
curl -s -X POST http://localhost:8080/v1/citations/export \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "period_start": "2025-01-01",
    "period_end": "2026-02-08",
    "include_version_history": true,
    "include_verifications": true,
    "include_evidence_packages": true,
    "include_provenance": true,
    "format": "JSON",
    "sign": true
  }' | python3 -m json.tool
```

---

## Post-Resolution Verification

```promql
# 1. Audit gap should be resolved
rate(gl_citations_version_writes_total[5m]) > 0

# 2. Provenance chain breaks should be 0
rate(gl_citations_provenance_chain_broken_total[5m]) == 0

# 3. Content hash failures should be 0
rate(gl_citations_hash_integrity_failures_total[5m]) == 0

# 4. Version write rate should match operation rate
abs(
  rate(gl_citations_operations_total{operation=~"create|update|delete|verify"}[5m]) -
  rate(gl_citations_version_writes_total[5m])
) < 0.01
```

---

## Audit Compliance Procedures

### SOC 2 Audit Request Procedure

When a SOC 2 auditor requests citation audit evidence:

1. **Generate the compliance export** for the requested period (use Option 5 above)
2. **Run provenance chain verification** and include the results
3. **Provide the export package** with the SHA-256 integrity hash and digital signature
4. **Document any audit repairs** that were performed during the audit period

### Regulatory Submission Procedure

When preparing citation audit trails for CSRD, CBAM, or other regulatory submissions:

1. **Generate a period-specific export** covering the reporting period
2. **Verify all provenance chains** are intact for the period
3. **Include evidence packages** with finalization hashes
4. **Include methodology references** and regulatory mappings
5. **Include verification audit trail** showing citation verification history

### Incident Documentation for Auditors

If an audit integrity incident occurred during the reporting period:

1. **Document the incident timeline** -- when detected, diagnosed, and resolved
2. **Document the root cause** -- what caused the audit gap or provenance break
3. **Document the remediation** -- what repair actions were taken (with timestamps)
4. **Include before/after integrity check results** -- proving the issue was fully resolved
5. **Note any data quality impact** -- whether any calculations used unaudited citations during the gap

---

## Prevention

### Audit Trail Protection

1. **Immutable version history** -- The citation_versions table uses TimescaleDB hypertable with append-only semantics
2. **Content hash verification** -- Every citation read verifies the stored content hash
3. **Provenance chain validation** -- Automated daily verification of the entire provenance chain
4. **Evidence package immutability** -- Finalized packages cannot be modified
5. **90-day retention on verifications** -- Verification records retained for audit compliance

### Access Control

1. **Application-only write access** -- Only the citations service has write access to citations tables
2. **Read-only roles for analysts** -- Analyst users have read-only access via `greenlang_readonly`
3. **RLS policies** -- Row-level security enforces tenant isolation
4. **Admin operations are audited** -- Admin API calls are logged with elevated audit detail

### Monitoring

- **Dashboard:** Citations & Evidence Agent Health (`/d/citations-service`)
- **Key metrics to watch:**
  - `gl_citations_version_writes_total` rate (should match operation rate)
  - `gl_citations_provenance_chain_broken_total` (should be 0)
  - `gl_citations_hash_integrity_failures_total` (should be 0)
  - `gl_citations_verifications_total` by status (should show verification activity)

### Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Verification failure spike, no active audit requests | On-call engineer | Within 30 minutes |
| L2 | Audit gap detected, no active regulatory submission | Platform team lead + compliance team | 15 minutes |
| L3 | Provenance chain broken or hash integrity failure during active audit | Platform team + compliance team + CTO notification | Immediate |
| L4 | Evidence of data tampering or unauthorized modifications | Security team + compliance team + legal + incident commander | Immediate |

### Runbook Maintenance

- **Last reviewed:** 2026-02-08
- **Owner:** Platform Team + Compliance Team + Security Team
- **Review cadence:** Quarterly or after any audit-related incident
- **Related alerts:** `CitationsServiceDown`, `CitationsHighErrorRate`, `CitationsExpiredCitations`
- **Related dashboards:** Citations & Evidence Agent Health
- **Related runbooks:** [Citations Service Down](./citations-service-down.md)
