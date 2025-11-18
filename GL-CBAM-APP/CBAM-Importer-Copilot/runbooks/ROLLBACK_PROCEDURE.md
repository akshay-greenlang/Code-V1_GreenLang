# CBAM IMPORTER COPILOT - ROLLBACK PROCEDURE

**Version:** 1.0.0
**Last Updated:** 2025-11-18
**Owner:** GreenLang CBAM Operations Team
**Classification:** OPERATIONAL CRITICAL

---

## PURPOSE

This runbook defines rollback procedures for the CBAM Importer Copilot, ensuring rapid recovery from failed deployments, configuration errors, or system issues while preserving CBAM reporting data integrity and compliance audit trails.

---

## WHEN TO EXECUTE ROLLBACK

Initiate rollback if:
- ✗ Emissions calculations producing incorrect results (compliance risk)
- ✗ CBAM validation rules failing for previously valid shipments
- ✗ Performance degradation >50% after deployment
- ✗ Critical bugs discovered in production
- ✗ Database migration failures
- ✗ Pod crash loops (>5 restarts in 10 minutes)
- ✗ Data corruption or loss detected
- ✗ EU reporting deadline at risk due to system issues

---

## PRE-ROLLBACK CHECKLIST

- [ ] Document current system state and error symptoms
- [ ] Capture logs from failing pods
- [ ] Export recent calculation audit trails for compliance
- [ ] Record current ConfigMap and Secret versions
- [ ] Identify last known good deployment version
- [ ] Verify rollback target availability in container registry
- [ ] Notify stakeholders (Engineering Manager, EU Importers)
- [ ] Get approval for P0/P1 rollbacks (VP Engineering)
- [ ] Prepare rollback communication plan
- [ ] Ensure database backup is recent (<4 hours old)

---

## ROLLBACK TYPES

### 1. Configuration Rollback (Fastest - 5 minutes)

**When to Use:** ConfigMap/Secret change caused issues, code is stable

**Impact:** Minimal - Configuration only, no code changes

#### Procedure

```bash
# 1. Identify current ConfigMap version
kubectl get configmap cbam-config -n greenlang -o yaml > /tmp/current-config.yaml
cat /tmp/current-config.yaml | grep "resourceVersion:"

# 2. List ConfigMap history (if using GitOps)
git log --oneline k8s/configmap.yaml | head -10

# 3. Restore from backup
kubectl apply -f backup/configmap-$(date -d "yesterday" +%Y%m%d).yaml

# OR restore from Git
git checkout HEAD~1 k8s/configmap.yaml
kubectl apply -f k8s/configmap.yaml

# 4. Restart pods to apply new config
kubectl rollout restart deployment/cbam-importer -n greenlang

# 5. Monitor rollout
kubectl rollout status deployment/cbam-importer -n greenlang -w

# 6. Verify configuration applied
kubectl exec -n greenlang deployment/cbam-importer -- \
  env | grep -E "CN_CODES_PATH|EMISSION_FACTORS_PATH|CBAM_RULES_PATH"

# 7. Test with sample shipment
curl -X POST http://cbam-importer:8000/validate \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/sample_shipment.json
```

#### Verification

```bash
# Check health
curl http://cbam-importer:8000/health | jq '.status'

# Verify emissions calculations match baseline
python tests/verify_calculations.py \
  --sample-shipments tests/fixtures/baseline_shipments.csv \
  --expected-results tests/baselines/expected_emissions.json

# Monitor error rates for 10 minutes
watch -n 30 'curl -s http://cbam-importer:8001/metrics | grep cbam_errors_total'
```

---

### 2. Application Rollback (Medium - 10 minutes)

**When to Use:** Code deployment caused issues

**Impact:** Moderate - Application code changes reverted

#### Procedure

```bash
# 1. Check rollout history
kubectl rollout history deployment/cbam-importer -n greenlang

# Output example:
# REVISION  CHANGE-CAUSE
# 1         Initial deployment
# 2         Update emission factors
# 3         Add CN code validation
# 4         Current (BROKEN)

# 2. Review specific revision
kubectl rollout history deployment/cbam-importer -n greenlang --revision=3

# 3. Rollback to previous version (revision N-1)
kubectl rollout undo deployment/cbam-importer -n greenlang

# OR rollback to specific revision
kubectl rollout undo deployment/cbam-importer -n greenlang --to-revision=3

# 4. Monitor rollback progress
kubectl rollout status deployment/cbam-importer -n greenlang -w

# 5. Verify all pods running
kubectl get pods -n greenlang -l app=cbam-importer -o wide

# Expected: All pods in Running state, READY 1/1

# 6. Check deployed image version
kubectl get deployment cbam-importer -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# Should show previous image tag

# 7. Verify application version
curl http://cbam-importer:8000/version

# Expected: Previous version number
```

#### Post-Rollback Validation

```bash
# 1. Run end-to-end test
python cbam_pipeline.py \
  --input tests/fixtures/e2e_test_shipments.csv \
  --output /tmp/rollback_test.json \
  --importer-name "Rollback Test Co" \
  --importer-country NL \
  --importer-eori NL000000000000 \
  --declarant-name "Test User" \
  --declarant-position "Tester"

# 2. Verify calculations deterministic
python tests/verify_determinism.py \
  --report1 /tmp/rollback_test.json \
  --report2 /tmp/rollback_test_run2.json

# 3. Check all agents healthy
curl http://cbam-importer:8000/health/agents | jq '.agents[] | select(.status != "healthy")'

# Expected: No output (all agents healthy)

# 4. Monitor metrics
curl http://cbam-importer:8001/metrics | grep -E "cbam_pipeline_runs_total|cbam_errors_total|cbam_validation_failures"
```

---

### 3. Full System Rollback (Comprehensive - 30 minutes)

**When to Use:** Major issues affecting multiple components, database schema changes, complete system failure

**Impact:** High - Full application, configuration, and potentially database rollback

#### Procedure

```bash
# ============================================================================
# PHASE 1: STOP CURRENT SYSTEM
# ============================================================================

# 1. Scale deployment to 0 (stop all processing)
kubectl scale deployment/cbam-importer --replicas=0 -n greenlang

# 2. Verify all pods terminated
kubectl get pods -n greenlang -l app=cbam-importer

# Expected: No pods running

# 3. Document current state
kubectl get all -n greenlang -l app=cbam-importer -o yaml > /tmp/current_state_$(date +%Y%m%d_%H%M%S).yaml

# ============================================================================
# PHASE 2: DATABASE ROLLBACK (IF NEEDED)
# ============================================================================

# Only perform if database schema changed or data corrupted

# 1. Check recent database backups
kubectl exec -n greenlang deployment/postgres -- \
  ls -lh /backups/ | tail -10

# 2. Identify backup to restore (within last 24 hours)
BACKUP_FILE="cbam_db_20251117_200000.sql"

# 3. Stop database connections
kubectl scale deployment/cbam-importer --replicas=0 -n greenlang

# 4. Restore database
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang < /backups/$BACKUP_FILE

# 5. Verify database restore
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "SELECT COUNT(*) FROM shipments;"

# 6. Check database version/schema
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang -c "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1;"

# ============================================================================
# PHASE 3: RESTORE KUBERNETES RESOURCES
# ============================================================================

# 1. Restore ConfigMap
kubectl apply -f backup/configmap-$(date -d "yesterday" +%Y%m%d).yaml

# 2. Restore Secrets (if changed)
kubectl apply -f backup/secrets-$(date -d "yesterday" +%Y%m%d).yaml

# 3. Restore Deployment (specific image version)
kubectl apply -f backup/deployment-$(date -d "yesterday" +%Y%m%d).yaml

# OR manually set image version
kubectl set image deployment/cbam-importer \
  cbam-importer=greenlang/cbam-importer:v1.2.3 \
  -n greenlang

# 4. Restore Service
kubectl apply -f backup/service-$(date -d "yesterday" +%Y%m%d).yaml

# 5. Restore Ingress
kubectl apply -f backup/ingress-$(date -d "yesterday" +%Y%m%d).yaml

# ============================================================================
# PHASE 4: SCALE UP AND VERIFY
# ============================================================================

# 1. Scale deployment to normal replica count
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang

# 2. Monitor pod startup
kubectl get pods -n greenlang -l app=cbam-importer -w

# Wait for all pods to be Running and READY 1/1

# 3. Check pod logs for errors
kubectl logs -n greenlang deployment/cbam-importer --tail=50

# 4. Verify all endpoints
curl http://cbam-importer:8000/health
curl http://cbam-importer:8000/health/agents
curl http://cbam-importer:8000/health/database

# ============================================================================
# PHASE 5: FUNCTIONAL VALIDATION
# ============================================================================

# 1. Test shipment intake
curl -X POST http://cbam-importer:8000/intake \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/test_shipments.json

# 2. Test emissions calculation
curl -X POST http://cbam-importer:8000/calculate \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/validated_shipments.json

# 3. Test report generation
python cbam_pipeline.py \
  --input tests/fixtures/baseline_shipments.csv \
  --output /tmp/full_rollback_test.json \
  --importer-name "Test Co" \
  --importer-country NL \
  --importer-eori NL000000000000 \
  --declarant-name "Test" \
  --declarant-position "Tester"

# 4. Verify calculations match baseline
python tests/verify_calculations.py \
  --actual /tmp/full_rollback_test.json \
  --expected tests/baselines/expected_report.json

# 5. Check compliance validation
jq '.validation_results.is_valid' /tmp/full_rollback_test.json

# Expected: true
```

---

## DATA PRESERVATION DURING ROLLBACK

### Critical Data to Preserve

```bash
# 1. Export calculation audit trails (COMPLIANCE CRITICAL)
kubectl exec -n greenlang deployment/cbam-importer -- \
  python -c "
from backend.app import export_all_audit_trails
export_all_audit_trails('/exports/audit_trails_$(date +%Y%m%d_%H%M%S).json')
"

# 2. Backup current metrics
curl http://cbam-importer:8001/metrics > /backup/metrics_$(date +%Y%m%d_%H%M%S).txt

# 3. Export recent reports
kubectl exec -n greenlang deployment/cbam-importer -- \
  psql $DATABASE_URL -c \
  "COPY (SELECT * FROM reports WHERE created_at > NOW() - INTERVAL '7 days')
   TO STDOUT CSV HEADER" > /backup/recent_reports_$(date +%Y%m%d).csv

# 4. Backup shipment data
kubectl exec -n greenlang deployment/cbam-importer -- \
  psql $DATABASE_URL -c \
  "COPY (SELECT * FROM shipments WHERE import_date > NOW() - INTERVAL '30 days')
   TO STDOUT CSV HEADER" > /backup/recent_shipments_$(date +%Y%m%d).csv

# 5. Export supplier profiles
cp examples/demo_suppliers.yaml /backup/suppliers_$(date +%Y%m%d).yaml

# 6. Export emission factors database
cp data/emission_factors.py /backup/emission_factors_$(date +%Y%m%d).py
```

---

## ROLLBACK COMMUNICATION

### Internal Notification Template

```
ROLLBACK EXECUTED: CBAM Importer Copilot

Time: {{timestamp}}
Severity: P{{0-2}}
Reason: {{brief description of issue}}

Rollback Details:
- Type: Configuration / Application / Full System
- From Version: {{current_version/revision}}
- To Version: {{rollback_version/revision}}
- Duration: {{minutes}}

Impact:
- Processing Interrupted: {{yes/no}}
- Reports Affected: {{count}}
- Data Integrity: {{preserved/requires validation}}
- EU Compliance Risk: {{none/low/medium/high}}

Validation Status:
- Health Checks: {{Pass/Fail}}
- Functional Tests: {{Pass/Fail}}
- Calculation Verification: {{Pass/Fail}}
- CBAM Compliance: {{Pass/Fail}}

Current Status: {{Stable/Monitoring/Issues}}

Next Steps:
1. {{action item}}
2. {{action item}}

Incident Response Team:
- Lead: {{name}}
- Contact: {{email/phone}}
```

### EU Importer Notification Template

```
CBAM Importer Copilot System Update

Dear {{importer_name}},

We have rolled back the CBAM Importer Copilot to a previous stable version due to {{brief reason}}.

Impact Assessment:
- Your Data: All shipment data and calculations preserved
- Recent Reports: {{affected/not affected}}
- Next Submission: {{on track/delayed by X hours}}
- Data Quality: No degradation

Actions Required:
{{none / please re-upload X shipments / please contact support}}

Current Status: System is stable and operating normally

EU CBAM Compliance: All compliance requirements maintained throughout rollback

Timeline:
- Issue Detected: {{timestamp}}
- Rollback Completed: {{timestamp}}
- System Verified: {{timestamp}}

We apologize for any inconvenience. Please contact our support team if you have questions.

Contact: cbam-support@greenlang.io | +31-20-xxx-xxxx
```

---

## ROLLBACK FAILURE SCENARIOS

### If Rollback Fails

1. **Pods Still Failing After Rollback:**
```bash
# Collect diagnostics
kubectl describe pod -n greenlang <failing-pod>
kubectl logs -n greenlang <failing-pod> --previous

# Check node resources
kubectl describe node <node-name>

# Try manual pod deletion
kubectl delete pod -n greenlang <failing-pod>
```

2. **Database Restore Fails:**
```bash
# Use older backup
OLDER_BACKUP="cbam_db_20251116_200000.sql"
kubectl exec -n greenlang deployment/postgres -- \
  psql greenlang < /backups/$OLDER_BACKUP

# If all backups fail, rebuild from source
# Use emergency shipment export from EU importers
```

3. **Complete System Failure:**
```bash
# Nuclear option: Redeploy from scratch
kubectl delete namespace greenlang
kubectl create namespace greenlang

# Restore from GitOps repository
kubectl apply -k k8s/overlays/production/

# Restore database from backup
# Load from backup/cbam_db_YYYYMMDD.sql

# Verify with end-to-end tests
```

### Emergency Manual Mode

If automated system cannot be restored:

```bash
# 1. Export all shipments to CSV
python scripts/emergency_export_shipments.py \
  --database-backup /backups/cbam_db_latest.sql \
  --output /emergency/shipments_export.csv

# 2. Process using offline scripts
python scripts/offline_cbam_calculator.py \
  --shipments /emergency/shipments_export.csv \
  --suppliers examples/demo_suppliers.yaml \
  --cn-codes data/cn_codes.json \
  --emission-factors data/emission_factors.py \
  --output /emergency/manual_cbam_report.json

# 3. Validate report
python scripts/validate_cbam_report.py \
  --report /emergency/manual_cbam_report.json \
  --rules rules/cbam_rules.yaml

# 4. Submit to EU Registry manually
# Use EU CBAM Transitional Registry web portal
```

---

## POST-ROLLBACK ACTIONS

### Immediate (0-4 hours)

- [ ] Monitor system stability (all health checks green)
- [ ] Review all error logs for recurring issues
- [ ] Verify emissions calculations match baselines
- [ ] Check compliance validation passing
- [ ] Confirm recent reports are valid
- [ ] Update incident tracker
- [ ] Notify all stakeholders of resolution

### Short-term (24 hours)

- [ ] Root cause analysis completed
- [ ] Document lessons learned
- [ ] Update runbooks with new findings
- [ ] Plan permanent fix for original issue
- [ ] Schedule post-mortem meeting (P0/P1 only)
- [ ] Review and update backup procedures
- [ ] Verify all EU importer data intact

### Long-term (1 week)

- [ ] Implement preventive measures
- [ ] Enhance testing procedures (prevent recurrence)
- [ ] Update deployment checklist
- [ ] Review change management process
- [ ] Train team on lessons learned
- [ ] Update monitoring and alerts
- [ ] Conduct rollback drill (practice scenario)

---

## ROLLBACK TESTING

### Quarterly Rollback Drill

**Purpose:** Ensure team proficiency and procedure accuracy

**Procedure:**
1. Schedule drill during maintenance window
2. Deploy intentionally broken version to staging
3. Execute full rollback procedure
4. Measure time to recovery
5. Document issues encountered
6. Update runbook based on findings

**Success Criteria:**
- Configuration rollback: <5 minutes
- Application rollback: <10 minutes
- Full system rollback: <30 minutes
- All tests passing post-rollback
- Zero data loss

---

## RELATED RUNBOOKS

- INCIDENT_RESPONSE.md - For incident triage and response
- TROUBLESHOOTING.md - For diagnosing issues
- MAINTENANCE.md - For routine operations
- SCALING_GUIDE.md - For performance scaling

---

## APPENDIX: Quick Reference Commands

```bash
# Check current deployment version
kubectl get deployment cbam-importer -n greenlang -o jsonpath='{.spec.template.spec.containers[0].image}'

# List rollout history
kubectl rollout history deployment/cbam-importer -n greenlang

# Rollback to previous
kubectl rollout undo deployment/cbam-importer -n greenlang

# Rollback to specific revision
kubectl rollout undo deployment/cbam-importer -n greenlang --to-revision=3

# Monitor rollback
kubectl rollout status deployment/cbam-importer -n greenlang -w

# Scale replicas
kubectl scale deployment/cbam-importer --replicas=3 -n greenlang

# Restart deployment
kubectl rollout restart deployment/cbam-importer -n greenlang

# Check pod health
kubectl get pods -n greenlang -l app=cbam-importer

# View logs
kubectl logs -n greenlang deployment/cbam-importer --tail=100

# Test health endpoint
curl http://cbam-importer:8000/health
```

---

**Document Control:**
- **Version:** 1.0.0
- **Last Updated:** 2025-11-18
- **Next Review:** 2025-12-18
- **Owner:** CBAM Operations Team
- **Approvers:** VP Engineering, SRE Lead

---

*This rollback procedure should be practiced quarterly and updated based on lessons learned from actual rollbacks and drills.*
