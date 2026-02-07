# Assumptions Audit Compliance

## Alert

**Alert Names:**

| Alert | Severity | Condition |
|-------|----------|-----------|
| `AssumptionsAuditGap` | Critical | Missing audit entries detected in change log (gap > 5 minutes with active writes) |
| `AssumptionsProvenanceChainBroken` | Critical | SHA-256 provenance hash chain verification failure |
| `AssumptionsExportIntegrityFailure` | Critical | Audit export package hash verification failure |
| `AssumptionsAuditRetentionRisk` | Warning | Audit records approaching retention limit without archive |

**Thresholds:**

```promql
# AssumptionsAuditGap
# Gap detection: write operations occurring but no corresponding change log entries
increase(gl_assumptions_operations_total{operation="write", status="success"}[5m]) > 0
  and increase(gl_assumptions_audit_entries_total[5m]) == 0
# sustained for 5 minutes

# AssumptionsProvenanceChainBroken
# Provenance hash chain verification failure
increase(gl_assumptions_provenance_verification_failures_total[5m]) > 0

# AssumptionsExportIntegrityFailure
# Export package integrity verification failure
increase(gl_assumptions_export_integrity_failures_total[5m]) > 0

# AssumptionsAuditRetentionRisk
# Audit records within 90 days of retention limit
gl_assumptions_audit_oldest_record_days > (gl_assumptions_audit_retention_days - 90)
```

---

## Description

These alerts fire when the audit and compliance subsystem of the Assumptions Registry service (AGENT-FOUND-004) detects integrity issues in the audit trail, provenance chain, or export packages. The audit system is a critical component of the zero-hallucination guarantee and is required for SOC 2, ISO 27001, CSRD, CBAM, and other regulatory compliance frameworks.

### How the Audit System Works

The Assumptions Registry maintains a multi-layered audit trail:

1. **Change Log** -- Every create, update, delete, and read operation on assumptions is recorded in the `assumptions_change_log` table with:
   - Timestamp (UTC, microsecond precision)
   - User ID (authenticated identity)
   - Operation type (create, update, delete, read, import, export)
   - Assumption key and version
   - Old value and new value (for updates)
   - Change reason (required, non-empty)
   - Source citation (required for creates and updates)
   - Scenario ID (if the operation was on a scenario override)
   - Validation status (passed, failed, bypassed)
   - Request metadata (IP address, user agent, correlation ID)

2. **Provenance Hash Chain** -- Each assumption version includes a SHA-256 provenance hash computed over:
   ```
   hash = SHA-256(
     assumption_key +
     version_number +
     value +
     unit +
     source +
     effective_date +
     user_id +
     change_reason +
     previous_hash
   )
   ```
   This creates a tamper-evident chain: if any historical value is modified, all subsequent hashes become invalid. The chain is analogous to a blockchain -- each version's hash depends on the previous version's hash.

3. **Version History** -- Complete version history for every assumption, stored in the `assumptions_version_history` table. Every version is immutable once created; updates create new versions rather than modifying existing ones.

4. **Audit Export Packages** -- On demand, the service generates signed export packages containing:
   - Complete assumption snapshot (all active assumptions with their current values)
   - Version history for the requested period
   - Change log entries for the requested period
   - Provenance hash chain for verification
   - Package-level SHA-256 integrity hash
   - Digital signature using the export signing key

### What Audit Compliance Issues Mean

1. **Audit Gap (Missing Change Log Entries)**: Write operations occurred but no corresponding change log entries were created. This is a critical integrity violation -- it means some assumption changes are not tracked in the audit trail. Possible causes include database transaction failures, async write pipeline errors, or a bug that bypasses the audit middleware.

2. **Provenance Chain Break**: A provenance hash verification failure means either:
   - A historical assumption value was modified directly in the database (data tampering or accidental direct SQL update)
   - The hash computation algorithm or input format changed between versions (serialization change)
   - A database migration modified the provenance data
   - A restore from backup introduced inconsistent versions

3. **Export Integrity Failure**: An audit export package's integrity hash does not match the computed hash of its contents. This means the export was corrupted during generation or transmission.

4. **Retention Risk**: Audit records are approaching the configured retention limit. If records are deleted before they are archived or exported, the audit trail will have permanent gaps that cannot be recovered.

### Regulatory Requirements

| Framework | Requirement | How Assumptions Registry Satisfies |
|-----------|-------------|-----------------------------------|
| **SOC 2** | Change management controls; audit trail for all configuration changes | Complete change log with user ID, reason, and timestamp for every assumption change |
| **ISO 27001** | A.12.4 Logging and monitoring; A.14.2 Security in development | Provenance hash chain provides tamper-evident logging; all changes are versioned |
| **CSRD** | Assurance over sustainability reporting data and assumptions | Audit export packages with integrity verification for auditor review |
| **CBAM** | Verifiable emission factors and calculation parameters | Every assumption is traceable to a source citation with provenance hash |
| **GHG Protocol** | Documentation of all assumptions and uncertainty | Scenario management with sensitivity analysis; complete version history |

---

## Impact Assessment

| Category | Impact Level | Details |
|----------|--------------|---------|
| **User Impact** | Medium | Users may not notice audit gaps immediately; export failures are visible |
| **Data Impact** | Critical | Audit trail integrity is compromised; assumption changes may not be traceable |
| **SLA Impact** | Medium | Service remains available for assumption operations, but audit operations may fail |
| **Revenue Impact** | High | Customers depend on audit-ready exports for regulatory submissions; failed audits may result in penalties |
| **Compliance Impact** | Critical | SOC 2, ISO 27001, CSRD, CBAM compliance directly impacted; regulatory submissions may be rejected without verifiable audit trails; third-party assurance packages compromised |
| **Downstream Impact** | High | Compliance reporting agents cannot generate auditor-ready reports; regulatory submission workflows blocked |

---

## Symptoms

### Audit Gap

- `AssumptionsAuditGap` alert firing
- `gl_assumptions_audit_entries_total` not incrementing despite active write operations
- Audit export packages showing gaps in timestamp sequences
- SOC 2 auditor flagging missing change records

### Provenance Chain Break

- `AssumptionsProvenanceChainBroken` alert firing
- `gl_assumptions_provenance_verification_failures_total` incrementing
- Provenance verification endpoint returning "hash mismatch" for specific assumptions
- Audit export packages failing integrity checks on provenance section
- Logs showing "provenance chain broken at version N" or "hash mismatch"

### Export Integrity Failure

- `AssumptionsExportIntegrityFailure` alert firing
- `gl_assumptions_export_integrity_failures_total` incrementing
- Users reporting "export verification failed" when downloading audit packages
- Exported files failing hash verification on the receiving end

### Retention Risk

- `AssumptionsAuditRetentionRisk` alert firing
- `gl_assumptions_audit_oldest_record_days` approaching the retention limit
- Audit records from early periods not yet archived

---

## Diagnostic Steps

### Step 1: Check Audit Integrity Metrics

```bash
# Port-forward to the assumptions service
kubectl port-forward -n greenlang svc/assumptions-service 8080:8080

# Get current audit metrics
curl -s http://localhost:8080/metrics | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Audit entries total: {data.get(\"audit_entries_total\", 0)}')
print(f'Provenance verifications: {data.get(\"provenance_verifications_total\", 0)}')
print(f'Provenance failures: {data.get(\"provenance_verification_failures_total\", 0)}')
print(f'Export operations: {data.get(\"export_total\", 0)}')
print(f'Export integrity failures: {data.get(\"export_integrity_failures_total\", 0)}')
print(f'Oldest record (days): {data.get(\"audit_oldest_record_days\", 0)}')
print(f'Retention limit (days): {data.get(\"audit_retention_days\", 0)}')
"
```

```promql
# Audit entry rate vs write operation rate (should be 1:1)
rate(gl_assumptions_audit_entries_total[5m])
rate(gl_assumptions_operations_total{operation="write", status="success"}[5m])

# Provenance verification failure rate
rate(gl_assumptions_provenance_verification_failures_total[5m])

# Export integrity failure rate
rate(gl_assumptions_export_integrity_failures_total[5m])

# Audit record age
gl_assumptions_audit_oldest_record_days

# Audit gap detection (audit entries should track write operations)
rate(gl_assumptions_operations_total{operation="write", status="success"}[5m]) -
  rate(gl_assumptions_audit_entries_total[5m])
```

### Step 2: Verify Audit Trail Completeness

```bash
# Check for gaps in change log timestamps
kubectl run pg-gaps --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "WITH time_series AS (
    SELECT
      date_trunc('minute', created_at) as minute,
      count(*) as entries
    FROM assumptions_change_log
    WHERE created_at > NOW() - INTERVAL '24 hours'
    GROUP BY date_trunc('minute', created_at)
    ORDER BY minute
  ),
  gaps AS (
    SELECT
      minute,
      entries,
      lead(minute) OVER (ORDER BY minute) as next_minute,
      EXTRACT(EPOCH FROM lead(minute) OVER (ORDER BY minute) - minute) / 60 as gap_minutes
    FROM time_series
  )
  SELECT minute, entries, next_minute, gap_minutes
  FROM gaps
  WHERE gap_minutes > 5
  ORDER BY gap_minutes DESC
  LIMIT 20;"

# Check for write operations without corresponding audit entries
kubectl run pg-missing --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vh.assumption_key, vh.version, vh.created_at as version_created,
          cl.created_at as audit_created
   FROM assumptions_version_history vh
   LEFT JOIN assumptions_change_log cl
     ON vh.assumption_key = cl.assumption_key
     AND vh.version = cl.version
     AND cl.operation IN ('create', 'update')
   WHERE vh.created_at > NOW() - INTERVAL '24 hours'
     AND cl.id IS NULL
   ORDER BY vh.created_at DESC
   LIMIT 20;"

# Check for change log entries with missing required fields
kubectl run pg-incomplete --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT id, assumption_key, operation, user_id, change_reason, created_at
   FROM assumptions_change_log
   WHERE created_at > NOW() - INTERVAL '24 hours'
     AND (user_id IS NULL OR user_id = ''
          OR change_reason IS NULL OR change_reason = ''
          OR operation IS NULL)
   ORDER BY created_at DESC
   LIMIT 20;"
```

### Step 3: Verify Provenance Hash Chain Integrity

```bash
# Run the full provenance chain verification
curl -s -X POST http://localhost:8080/v1/assumptions/admin/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Total assumptions verified: {data.get(\"total_verified\", 0)}')
print(f'Total versions checked: {data.get(\"total_versions_checked\", 0)}')
print(f'Chain intact: {data.get(\"chain_intact\", False)}')
print(f'Broken links: {data.get(\"broken_links\", 0)}')
print(f'Verification time: {data.get(\"duration_ms\", 0)}ms')
print()
if data.get('broken_links_details'):
    print('Broken provenance links:')
    for link in data['broken_links_details']:
        print(f\"  Assumption: {link['key']}, Version: {link['version']}\")
        print(f\"    Expected hash: {link['expected_hash'][:16]}...\")
        print(f\"    Computed hash: {link['computed_hash'][:16]}...\")
        print(f\"    Previous hash: {link['previous_hash'][:16]}...\")
        print(f\"    Created at: {link['created_at']}\")
        print()
"

# Verify a specific assumption's provenance chain
curl -s -X POST http://localhost:8080/v1/assumptions/<assumption_key>/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Assumption: {data.get(\"key\")}')
print(f'Versions: {data.get(\"total_versions\", 0)}')
print(f'Chain intact: {data.get(\"chain_intact\", False)}')
if not data.get('chain_intact'):
    print(f'Break at version: {data.get(\"break_at_version\")}')
    print(f'Break reason: {data.get(\"break_reason\")}')
"

# Check the database directly for hash inconsistencies
kubectl run pg-hash-check --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT vh1.assumption_key, vh1.version, vh1.provenance_hash,
          vh1.previous_hash, vh2.provenance_hash as expected_previous
   FROM assumptions_version_history vh1
   LEFT JOIN assumptions_version_history vh2
     ON vh1.assumption_key = vh2.assumption_key
     AND vh1.version = vh2.version + 1
   WHERE vh1.version > 1
     AND vh1.previous_hash != vh2.provenance_hash
   ORDER BY vh1.created_at DESC
   LIMIT 20;"
```

### Step 4: Check for Direct Database Modifications

```bash
# Check PostgreSQL audit log for direct SQL modifications
kubectl run pg-audit --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT query, usename, client_addr, query_start
   FROM pg_stat_activity
   WHERE query LIKE '%assumptions_%'
     AND query NOT LIKE '%SELECT%'
     AND usename != 'greenlang_app'
   ORDER BY query_start DESC;"

# Check for recent DDL or DML operations by non-application users
kubectl run pg-ddl --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT * FROM pg_stat_statements
   WHERE query LIKE '%assumptions_%'
     AND query ~* '(UPDATE|DELETE|INSERT|ALTER|DROP)'
   ORDER BY calls DESC
   LIMIT 20;"
```

### Step 5: Check Export Integrity

```bash
# Check logs for export errors
kubectl logs -n greenlang -l app=assumptions-service --tail=500 \
  | grep -i "export\|integrity\|sign\|hash.*mismatch\|package"

# Test export generation and verification
curl -s -X POST http://localhost:8080/v1/assumptions/export \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "period_start": "2026-01-01",
    "period_end": "2026-02-07",
    "include_version_history": true,
    "include_change_log": true,
    "include_provenance": true,
    "format": "JSON"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Export ID: {data.get(\"export_id\")}')
print(f'Status: {data.get(\"status\")}')
print(f'Assumptions included: {data.get(\"assumption_count\", 0)}')
print(f'Versions included: {data.get(\"version_count\", 0)}')
print(f'Change log entries: {data.get(\"change_log_count\", 0)}')
print(f'Package hash: {data.get(\"package_hash\", \"N/A\")[:16]}...')
print(f'Signature valid: {data.get(\"signature_valid\", False)}')
"

# Verify an existing export package
curl -s -X POST http://localhost:8080/v1/assumptions/export/verify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "export_id": "<export_id>",
    "expected_hash": "<package_hash>"
  }' | python3 -m json.tool
```

### Step 6: Check Audit Retention Status

```bash
# Check the oldest and newest audit records
kubectl run pg-retention --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT
    min(created_at) as oldest_record,
    max(created_at) as newest_record,
    EXTRACT(DAY FROM NOW() - min(created_at)) as oldest_days,
    count(*) as total_records,
    pg_size_pretty(pg_relation_size('assumptions_change_log')) as table_size
   FROM assumptions_change_log;"

# Check retention configuration
kubectl get configmap assumptions-service-config -n greenlang -o yaml \
  | grep -i "retention"

# Check if archival jobs are running
kubectl get cronjobs -n greenlang | grep assumptions
kubectl logs -n greenlang -l app=assumptions-archival-job --tail=50
```

### Step 7: Cross-Reference with Downstream Provenance

```bash
# Check if downstream calculations reference provenance hashes that exist
kubectl run pg-downstream --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT DISTINCT cl.correlation_id, cl.assumption_key, cl.provenance_hash,
          cl.created_at
   FROM assumptions_change_log cl
   WHERE cl.created_at > NOW() - INTERVAL '7 days'
     AND cl.correlation_id IS NOT NULL
   ORDER BY cl.created_at DESC
   LIMIT 20;"
```

---

## Resolution Steps

### Option 1: Repair Audit Gaps by Cross-Referencing Version History

If audit entries are missing but the version history is intact, reconstruct the missing audit entries.

```bash
# Step 1: Identify the gap period
kubectl run pg-gap-period --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT
    min(vh.created_at) as gap_start,
    max(vh.created_at) as gap_end,
    count(*) as missing_entries
   FROM assumptions_version_history vh
   LEFT JOIN assumptions_change_log cl
     ON vh.assumption_key = cl.assumption_key
     AND vh.version = cl.version
   WHERE vh.created_at > NOW() - INTERVAL '7 days'
     AND cl.id IS NULL;"

# Step 2: Reconstruct audit entries from version history
curl -X POST http://localhost:8080/v1/assumptions/admin/repair-audit-gaps \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "gap_start": "2026-02-06T00:00:00Z",
    "gap_end": "2026-02-07T00:00:00Z",
    "dry_run": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Gap period: {data.get(\"gap_start\")} to {data.get(\"gap_end\")}')
print(f'Missing entries found: {data.get(\"missing_count\", 0)}')
print(f'Entries reconstructed: {data.get(\"reconstructed_count\", 0)}')
print(f'Dry run: {data.get(\"dry_run\", True)}')
if data.get('reconstructed_entries'):
    for e in data['reconstructed_entries'][:5]:
        print(f\"  {e['assumption_key']} v{e['version']}: {e['operation']} at {e['timestamp']}\")
"

# Step 3: If dry run looks correct, run the actual repair
curl -X POST http://localhost:8080/v1/assumptions/admin/repair-audit-gaps \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "gap_start": "2026-02-06T00:00:00Z",
    "gap_end": "2026-02-07T00:00:00Z",
    "dry_run": false,
    "repair_reason": "Reconstructed missing audit entries from version history. Incident: INC-XXXX"
  }' | python3 -m json.tool
```

**Important:** Reconstructed entries are marked with `source = 'audit_repair'` in the change log to distinguish them from entries created during normal operations. This ensures transparency for auditors.

### Option 2: Rebuild Provenance Hash Chain

If the provenance hash chain is broken due to a serialization change or data migration, rebuild the chain from the version history.

```bash
# Step 1: Identify the break point
curl -s -X POST http://localhost:8080/v1/assumptions/admin/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('broken_links_details'):
    first_break = data['broken_links_details'][0]
    print(f'First break at: {first_break[\"key\"]} version {first_break[\"version\"]}')
    print(f'Created at: {first_break[\"created_at\"]}')
"

# Step 2: Rebuild the provenance chain (requires maintenance window)
curl -X POST http://localhost:8080/v1/assumptions/admin/rebuild-provenance \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "scope": "broken_only",
    "dry_run": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Assumptions affected: {data.get(\"affected_count\", 0)}')
print(f'Versions to rehash: {data.get(\"versions_to_rehash\", 0)}')
print(f'Estimated time: {data.get(\"estimated_duration_seconds\", 0)}s')
print(f'Dry run: {data.get(\"dry_run\", True)}')
"

# Step 3: If dry run looks correct, execute the rebuild
curl -X POST http://localhost:8080/v1/assumptions/admin/rebuild-provenance \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "scope": "broken_only",
    "dry_run": false,
    "reason": "Provenance chain rebuild after hash mismatch. Incident: INC-XXXX"
  }' | python3 -m json.tool
```

**Important:** Provenance chain rebuilds create a new audit entry documenting the rebuild. The old (broken) hashes are preserved in a separate `provenance_rebuild_log` table for forensic analysis.

### Option 3: Investigate and Remediate Direct Database Modifications

If the provenance chain break was caused by direct database modifications:

```bash
# Step 1: Check PostgreSQL logs for unauthorized modifications
kubectl logs -n greenlang -l app=postgresql --tail=2000 \
  | grep -i "assumptions_\|UPDATE\|DELETE\|INSERT" \
  | grep -v "SELECT"

# Step 2: Check who has direct database access
kubectl run pg-access --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT rolname, rolsuper, rolcreaterole, rolcreatedb
   FROM pg_roles
   WHERE rolcanlogin = true
   ORDER BY rolname;"

# Step 3: Revoke direct write access to assumptions tables for non-application roles
kubectl run pg-revoke --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "REVOKE INSERT, UPDATE, DELETE ON assumptions_registry,
    assumptions_version_history, assumptions_change_log,
    assumptions_scenario_overrides
   FROM greenlang_readonly;
   -- Add audit trigger to catch future direct modifications
   CREATE OR REPLACE FUNCTION audit_direct_modification()
   RETURNS TRIGGER AS \$\$
   BEGIN
     INSERT INTO assumptions_direct_modification_log
       (table_name, operation, user_name, old_data, new_data, modified_at)
     VALUES (TG_TABLE_NAME, TG_OP, current_user, row_to_json(OLD), row_to_json(NEW), NOW());
     RETURN NEW;
   END;
   \$\$ LANGUAGE plpgsql;"

# Step 4: Rebuild provenance for affected records (use Option 2 above)
```

### Option 4: Generate Compliance Export for Auditors

When an auditor requests an assumption audit package:

```bash
# Generate a full compliance export
curl -s -X POST http://localhost:8080/v1/assumptions/export \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "period_start": "2025-01-01",
    "period_end": "2026-02-07",
    "include_version_history": true,
    "include_change_log": true,
    "include_provenance": true,
    "include_scenarios": true,
    "include_validation_rules": true,
    "include_dependency_graph": true,
    "format": "JSON",
    "sign": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Export ID: {data.get(\"export_id\")}')
print(f'Status: {data.get(\"status\")}')
print(f'Period: {data.get(\"period_start\")} to {data.get(\"period_end\")}')
print(f'Assumptions: {data.get(\"assumption_count\", 0)}')
print(f'Versions: {data.get(\"version_count\", 0)}')
print(f'Change log entries: {data.get(\"change_log_count\", 0)}')
print(f'Package hash (SHA-256): {data.get(\"package_hash\", \"N/A\")}')
print(f'Signature: {data.get(\"signature\", \"N/A\")[:32]}...')
print(f'Download URL: {data.get(\"download_url\", \"N/A\")}')
"

# Verify the export package integrity
curl -s -X POST http://localhost:8080/v1/assumptions/export/verify \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "export_id": "<export_id>"
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Export ID: {data.get(\"export_id\")}')
print(f'Hash valid: {data.get(\"hash_valid\", False)}')
print(f'Signature valid: {data.get(\"signature_valid\", False)}')
print(f'Provenance chain intact: {data.get(\"provenance_intact\", False)}')
print(f'Change log complete: {data.get(\"change_log_complete\", False)}')
if not data.get('hash_valid') or not data.get('provenance_intact'):
    print('WARNING: Export integrity check FAILED. See details above.')
"
```

### Option 5: Archive Audit Records Before Retention Expiry

If audit records are approaching the retention limit:

```bash
# Check current retention status
kubectl run pg-retention-status --rm -it --image=postgres:15 -n greenlang --restart=Never -- \
  psql -h greenlang-db.postgres.svc.cluster.local -U greenlang -d greenlang -c \
  "SELECT
    date_trunc('month', created_at) as month,
    count(*) as entries,
    pg_size_pretty(sum(pg_column_size(t.*))::bigint) as estimated_size
   FROM assumptions_change_log t
   GROUP BY date_trunc('month', created_at)
   ORDER BY month ASC;"

# Trigger an archival job
curl -X POST http://localhost:8080/v1/assumptions/admin/archive \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "archive_before": "2025-01-01",
    "destination": "s3://greenlang-audit-archive/assumptions/",
    "compress": true,
    "verify_after_archive": true,
    "dry_run": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f'Records to archive: {data.get(\"record_count\", 0)}')
print(f'Estimated size: {data.get(\"estimated_size\", \"N/A\")}')
print(f'Destination: {data.get(\"destination\", \"N/A\")}')
print(f'Dry run: {data.get(\"dry_run\", True)}')
"

# If dry run looks correct, execute the archival
curl -X POST http://localhost:8080/v1/assumptions/admin/archive \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "archive_before": "2025-01-01",
    "destination": "s3://greenlang-audit-archive/assumptions/",
    "compress": true,
    "verify_after_archive": true,
    "dry_run": false
  }' | python3 -m json.tool
```

### Option 6: Run Full Integrity Verification

For a comprehensive check of the entire audit system:

```bash
# Run the full integrity verification suite
curl -s -X POST http://localhost:8080/v1/assumptions/admin/full-integrity-check \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
print('=== Assumptions Registry Full Integrity Check ===')
print()
checks = data.get('checks', {})
for check_name, result in checks.items():
    status = 'PASS' if result.get('passed') else 'FAIL'
    print(f'[{status}] {check_name}')
    if not result.get('passed'):
        print(f'       Details: {result.get(\"details\", \"N/A\")}')
        print(f'       Affected: {result.get(\"affected_count\", 0)} records')
print()
print(f'Overall: {data.get(\"overall_status\", \"UNKNOWN\")}')
print(f'Duration: {data.get(\"duration_ms\", 0)}ms')
print(f'Timestamp: {data.get(\"timestamp\", \"N/A\")}')
"
```

The full integrity check verifies:
1. **Audit completeness** -- Every version has a corresponding change log entry
2. **Provenance chain** -- All SHA-256 hash chains are intact
3. **Required fields** -- All change log entries have user_id, change_reason, and source
4. **Version consistency** -- Version numbers are sequential with no gaps
5. **Scenario consistency** -- All scenario overrides reference valid baseline assumptions
6. **Validation rule integrity** -- All active validation rules reference valid assumptions
7. **Export signing key validity** -- The export signing key is valid and not expired

---

## Post-Resolution Verification

```promql
# 1. Audit gap should be resolved (entries tracking writes again)
rate(gl_assumptions_audit_entries_total[5m]) > 0

# 2. Provenance verification failures should be 0
rate(gl_assumptions_provenance_verification_failures_total[5m]) == 0

# 3. Export integrity failures should be 0
rate(gl_assumptions_export_integrity_failures_total[5m]) == 0

# 4. Audit retention should be within safe range
gl_assumptions_audit_oldest_record_days < (gl_assumptions_audit_retention_days - 90)

# 5. Audit entry rate should match write operation rate
abs(
  rate(gl_assumptions_operations_total{operation="write", status="success"}[5m]) -
  rate(gl_assumptions_audit_entries_total[5m])
) < 0.01
```

```bash
# 6. Run a targeted provenance verification for recently repaired assumptions
curl -s -X POST http://localhost:8080/v1/assumptions/admin/verify-provenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('chain_intact'):
    print('PASS: Provenance hash chain is fully intact')
else:
    print(f'FAIL: {data.get(\"broken_links\", 0)} broken links remaining')
"

# 7. Generate and verify a test export
curl -s -X POST http://localhost:8080/v1/assumptions/export \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "period_start": "2026-02-01",
    "period_end": "2026-02-07",
    "include_version_history": true,
    "include_change_log": true,
    "include_provenance": true,
    "format": "JSON",
    "sign": true
  }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
if data.get('status') == 'completed':
    print(f'PASS: Export generated successfully (ID: {data.get(\"export_id\")})')
    print(f'  Package hash: {data.get(\"package_hash\", \"N/A\")[:16]}...')
    print(f'  Signature valid: {data.get(\"signature_valid\", False)}')
else:
    print(f'FAIL: Export status is {data.get(\"status\")}')
"
```

---

## Audit Compliance Procedures

### SOC 2 Audit Request Procedure

When a SOC 2 auditor requests assumption audit evidence:

1. **Generate the compliance export** for the requested period (use Option 4 above)
2. **Run the full integrity check** (use Option 6 above) and include the results
3. **Provide the export package** with the SHA-256 integrity hash and digital signature
4. **Provide the verification endpoint** so the auditor can independently verify the export
5. **Document any audit repairs** that were performed during the audit period

### Regulatory Submission Procedure

When preparing assumption audit trails for CSRD, CBAM, or other regulatory submissions:

1. **Generate a period-specific export** covering the reporting period
2. **Verify all provenance chains** are intact for the period
3. **Cross-reference with downstream calculation provenance** to ensure end-to-end traceability
4. **Include scenario documentation** if regulatory submission includes scenario analysis
5. **Include validation rule documentation** to demonstrate data quality controls

### Incident Documentation for Auditors

If an audit integrity incident occurred during the reporting period:

1. **Document the incident timeline** -- when it was detected, diagnosed, and resolved
2. **Document the root cause** -- what caused the audit gap or provenance break
3. **Document the remediation** -- what repair actions were taken (with timestamps)
4. **Include before/after integrity check results** -- proving the issue was fully resolved
5. **Note any data quality impact** -- whether any calculations used unaudited assumptions during the gap

---

## Prevention

### Audit Trail Protection

1. **Database trigger protection** -- Database triggers that prevent direct modifications to audit tables without going through the application layer
2. **Immutable change log** -- The change log table has no UPDATE or DELETE permissions for the application user; entries are append-only
3. **Provenance verification scheduled job** -- Daily automated verification of the entire provenance chain
4. **Export integrity monitoring** -- Every export triggers an automatic integrity verification
5. **Retention alert threshold** -- Alert when records are within 90 days of the retention limit

### Access Control

1. **Application-only write access** -- Only the assumptions service application user has write access to assumptions tables
2. **Read-only roles for analysts** -- Analyst users have read-only access via a separate database role
3. **No direct SQL access in production** -- All assumption modifications must go through the API
4. **Admin operations are audited** -- Admin API calls (repair, rebuild, archive) are logged in the change log with elevated audit detail

### Monitoring

- **Dashboard:** Assumptions Registry Health (`/d/assumptions-service-health`) -- audit panels
- **Dashboard:** Assumptions Operations Overview (`/d/assumptions-operations-overview`)
- **Alert:** `AssumptionsAuditGap` (this alert)
- **Alert:** `AssumptionsProvenanceChainBroken` (this alert)
- **Alert:** `AssumptionsExportIntegrityFailure` (this alert)
- **Alert:** `AssumptionsAuditRetentionRisk` (this alert)
- **Key metrics to watch:**
  - `gl_assumptions_audit_entries_total` rate (should match write operation rate)
  - `gl_assumptions_provenance_verification_failures_total` (should be 0)
  - `gl_assumptions_export_integrity_failures_total` (should be 0)
  - `gl_assumptions_audit_oldest_record_days` (should be well within retention limit)
  - `gl_assumptions_provenance_verifications_total` (should show regular verification activity)

### Escalation Path

| Level | Condition | Contact | Response Time |
|-------|-----------|---------|---------------|
| L1 | Audit retention risk, no active audit requests | On-call engineer | Within 30 minutes |
| L2 | Audit gap detected, no active regulatory submission | Platform team lead + compliance team | 15 minutes |
| L3 | Provenance chain broken, or export integrity failure during active audit | Platform team + compliance team + CTO notification | Immediate |
| L4 | Evidence of data tampering or unauthorized modifications | Security team + compliance team + legal + incident commander | Immediate |

### Runbook Maintenance

- **Last reviewed:** 2026-02-07
- **Owner:** Platform Team + Compliance Team + Security Team
- **Review cadence:** Quarterly or after any audit-related incident
- **Related alerts:** `AssumptionsServiceDown`, `AssumptionsValidationFailureSpike`, `AssumptionsScenarioDrift`
- **Related dashboards:** Assumptions Registry Health, Assumptions Operations Overview
- **Related runbooks:** [Assumptions Service Down](./assumptions-service-down.md), [Assumption Validation Failures](./assumption-validation-failures.md), [Scenario Drift Detection](./scenario-drift-detection.md)
