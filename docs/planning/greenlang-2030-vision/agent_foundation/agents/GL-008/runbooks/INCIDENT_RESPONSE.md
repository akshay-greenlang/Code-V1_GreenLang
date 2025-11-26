# GL-008 SteamTrapInspector - Incident Response Runbook

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team
**On-Call Rotation:** https://pagerduty.com/greenlang/gl-008

---

## Table of Contents

1. [Incident Classification](#incident-classification)
2. [General Response Procedures](#general-response-procedures)
3. [Incident Scenarios](#incident-scenarios)
4. [Escalation Matrix](#escalation-matrix)
5. [Post-Incident Review](#post-incident-review)
6. [Communication Templates](#communication-templates)

---

## Incident Classification

### Priority Levels

| Priority | Description | Response Time | Example |
|----------|-------------|---------------|---------|
| **P0 - Critical** | Complete service outage affecting production facilities | 15 minutes | All trap inspections failing, security breach |
| **P1 - High** | Major functionality degraded, multiple customers impacted | 30 minutes | Mass sensor failures, ML model down |
| **P2 - Medium** | Partial functionality impacted, limited customer impact | 2 hours | High false positive rate, slow inspections |
| **P3 - Low** | Minor issues, workaround available | 1 business day | Individual sensor issues, minor UI bugs |
| **P4 - Planning** | Enhancement requests, non-urgent improvements | As scheduled | Feature requests, optimization tasks |

### Impact Assessment Matrix

| Impact | 1-10 Sites | 11-50 Sites | 51-100 Sites | 100+ Sites |
|--------|------------|-------------|--------------|------------|
| **Complete Outage** | P1 | P0 | P0 | P0 |
| **Major Degradation** | P2 | P1 | P1 | P0 |
| **Minor Issues** | P3 | P2 | P2 | P1 |
| **Cosmetic/UX** | P4 | P3 | P3 | P2 |

### Severity Indicators

**P0/P1 Indicators:**
- No trap inspections completing
- Energy calculations returning incorrect values (>10% error)
- Security vulnerability actively exploited
- Data loss or corruption
- SLA breach imminent (<15 minutes to breach)

**P2 Indicators:**
- False positive rate >30%
- Inspection latency >30 seconds
- Partial sensor array failure (>20% sensors offline)
- ML model degradation (accuracy <80%)

**P3/P4 Indicators:**
- Individual sensor malfunctions
- Performance degradation <20%
- Non-critical feature unavailable
- Documentation issues

---

## General Response Procedures

### Step 1: Alert Receipt (0-5 minutes)

1. **Acknowledge the alert** in PagerDuty/incident system
2. **Join the incident channel:** `#incident-gl008-{incident_id}`
3. **Assess initial severity:**
   ```bash
   # Check service health
   kubectl get pods -n greenlang-gl008

   # Check recent errors
   kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=100 --since=10m | grep ERROR

   # Check metrics dashboard
   open https://grafana.greenlang.io/d/gl008-overview
   ```

4. **Declare incident level** using the classification matrix above

### Step 2: Initial Assessment (5-10 minutes)

1. **Gather context:**
   ```bash
   # Check system status
   curl -H "Authorization: Bearer $API_TOKEN" \
     https://api.greenlang.io/v1/steam-trap/health

   # Check recent deployments
   kubectl rollout history deployment/steam-trap-inspector -n greenlang-gl008

   # Check resource utilization
   kubectl top pods -n greenlang-gl008
   kubectl top nodes -l environment=production
   ```

2. **Document findings** in incident tracker:
   - Time incident started
   - Affected components
   - Customer impact (number of sites)
   - Initial symptoms

3. **Notify stakeholders** (for P0/P1 incidents)

### Step 3: Communication (10-15 minutes)

**P0/P1 Communication Template:**

```
INCIDENT: GL-008 Steam Trap Inspector - [ISSUE SUMMARY]
Priority: P0/P1
Status: Investigating
Impact: [X] sites unable to conduct trap inspections
Start Time: [YYYY-MM-DD HH:MM UTC]
Incident Channel: #incident-gl008-{id}
Incident Commander: @[name]

Current Status:
- [What we know]
- [What we're investigating]
- [Next update in 15 minutes]

Affected Services:
- [ ] Acoustic inspection
- [ ] Thermal imaging
- [ ] Energy calculations
- [ ] API endpoints
```

### Step 4: Mitigation (15+ minutes)

1. **Apply immediate fixes** (see scenario-specific procedures below)
2. **Monitor impact reduction:**
   ```bash
   # Watch error rates
   watch -n 5 'kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=50 | grep ERROR | wc -l'

   # Monitor inspection success rate
   curl https://api.greenlang.io/v1/steam-trap/metrics/success-rate
   ```

3. **Update incident status** every 15-30 minutes (P0/P1) or hourly (P2)

### Step 5: Resolution & Verification

1. **Verify service restoration:**
   ```bash
   # Run health checks
   ./scripts/health-check.sh --comprehensive

   # Test full inspection workflow
   curl -X POST https://api.greenlang.io/v1/steam-trap/inspection \
     -H "Authorization: Bearer $API_TOKEN" \
     -d @test-inspection.json

   # Verify metrics returned to baseline
   ```

2. **Monitor for 30 minutes** before declaring resolved

3. **Update status page** and notify customers

### Step 6: Post-Incident Activities

1. **Schedule post-incident review** (within 48 hours for P0/P1)
2. **Document timeline** in incident tracker
3. **Create action items** for follow-up work
4. **Update runbooks** if new learnings discovered

---

## Incident Scenarios

### Scenario 1: Mass Trap Failure Detection

**Symptoms:**
- Multiple steam traps flagged as failed simultaneously
- No pattern in trap locations or ages
- Alert volume spike (>50 traps in <10 minutes)

**Likely Causes:**
- ML model false positive surge
- Sensor calibration drift
- Environmental interference (electrical noise)
- Software bug in detection algorithm

**Response Procedure:**

#### Phase 1: Immediate Assessment (0-10 minutes)

```bash
# Check false positive rate
curl https://api.greenlang.io/v1/steam-trap/metrics/false-positive-rate

# Review recent detections
psql $DB_URL -c "
  SELECT
    detected_at,
    COUNT(*) as failure_count,
    AVG(confidence_score) as avg_confidence
  FROM trap_inspections
  WHERE status = 'FAILED'
    AND detected_at > NOW() - INTERVAL '1 hour'
  GROUP BY detected_at
  ORDER BY detected_at DESC
  LIMIT 20;
"

# Check ML model version
kubectl get configmap -n greenlang-gl008 ml-model-config -o yaml | grep version

# Review sensor calibration status
curl https://api.greenlang.io/v1/steam-trap/sensors/calibration-status
```

#### Phase 2: Mitigation (10-30 minutes)

**If ML model issue suspected:**

```bash
# Temporarily increase confidence threshold
kubectl patch configmap ml-model-config -n greenlang-gl008 \
  --type merge \
  -p '{"data":{"CONFIDENCE_THRESHOLD":"0.85"}}'

# Restart pods to apply config
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008

# Monitor impact
watch -n 10 'curl -s https://api.greenlang.io/v1/steam-trap/metrics/detection-rate'
```

**If sensor calibration drift:**

```bash
# Pause automated inspections
curl -X POST https://api.greenlang.io/v1/steam-trap/admin/pause-inspections \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Run calibration routine for all sensors
./scripts/calibrate-sensors.sh --all-sites --emergency-mode

# Resume inspections after calibration
curl -X POST https://api.greenlang.io/v1/steam-trap/admin/resume-inspections \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**If environmental interference:**

```bash
# Enable noise filtering
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ACOUSTIC_NOISE_FILTER=aggressive

# Reduce thermal sensitivity temporarily
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  THERMAL_SENSITIVITY=medium
```

#### Phase 3: Validation (30-60 minutes)

```bash
# Test inspections on known-good traps
./scripts/validate-inspections.sh --test-set=baseline-traps

# Compare current vs. historical detection rates
psql $DB_URL -c "
  WITH hourly_stats AS (
    SELECT
      date_trunc('hour', detected_at) as hour,
      COUNT(*) as failures,
      AVG(confidence_score) as avg_conf
    FROM trap_inspections
    WHERE status = 'FAILED'
      AND detected_at > NOW() - INTERVAL '7 days'
    GROUP BY 1
  )
  SELECT
    hour,
    failures,
    avg_conf,
    AVG(failures) OVER (ORDER BY hour ROWS BETWEEN 23 PRECEDING AND CURRENT ROW) as rolling_avg
  FROM hourly_stats
  ORDER BY hour DESC
  LIMIT 48;
"
```

#### Escalation Criteria:

- False positive rate remains >30% after 1 hour
- Unable to identify root cause within 2 hours
- Customer complaints about excessive alerts

**Escalate to:** ML Engineering Team + Product Owner

---

### Scenario 2: Sensor Communication Loss

**Symptoms:**
- Multiple sensors showing "offline" status
- No data received from sensor arrays
- Inspection jobs timing out

**Likely Causes:**
- Network connectivity issues
- Sensor gateway failures
- Power outages at facility
- Firmware bugs in sensor nodes

**Response Procedure:**

#### Phase 1: Assessment (0-10 minutes)

```bash
# Check sensor connectivity status
curl https://api.greenlang.io/v1/steam-trap/sensors/status | jq '.offline_sensors'

# Group offline sensors by site
psql $DB_URL -c "
  SELECT
    site_id,
    site_name,
    COUNT(*) as offline_count,
    array_agg(sensor_id) as sensor_ids
  FROM sensors
  WHERE status = 'OFFLINE'
    AND last_seen < NOW() - INTERVAL '10 minutes'
  GROUP BY site_id, site_name
  ORDER BY offline_count DESC;
"

# Check sensor gateway logs
kubectl logs -n greenlang-gl008 -l component=sensor-gateway --tail=200 | grep -E "(ERROR|WARN|connection)"

# Test network connectivity to sensor gateways
./scripts/test-sensor-connectivity.sh --all-sites
```

#### Phase 2: Diagnosis (10-20 minutes)

**Pattern 1: Single site affected**

```bash
# Contact site operations
./scripts/notify-site-ops.sh --site-id=$SITE_ID \
  --message="Sensor communication loss detected. Please check power and network connectivity."

# Check if site network is responsive
ping -c 5 $SITE_GATEWAY_IP
traceroute $SITE_GATEWAY_IP

# Review site-specific logs
./scripts/get-site-logs.sh --site-id=$SITE_ID --hours=2
```

**Pattern 2: Multiple sites affected**

```bash
# Check central gateway status
kubectl get pods -n greenlang-gl008 -l component=sensor-gateway

# Review cloud connectivity
curl https://api.greenlang.io/v1/steam-trap/network/gateway-status

# Check for DDoS or network attacks
kubectl logs -n greenlang-gl008 -l component=api-gateway | grep -E "(429|503)" | wc -l
```

**Pattern 3: Specific sensor types offline**

```bash
# Check by sensor type
psql $DB_URL -c "
  SELECT
    sensor_type,
    COUNT(*) as offline_count,
    MIN(last_seen) as oldest_offline
  FROM sensors
  WHERE status = 'OFFLINE'
  GROUP BY sensor_type;
"

# Check for firmware version correlation
psql $DB_URL -c "
  SELECT
    firmware_version,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'OFFLINE' THEN 1 ELSE 0 END) as offline
  FROM sensors
  GROUP BY firmware_version
  ORDER BY offline DESC;
"
```

#### Phase 3: Mitigation (20-45 minutes)

**For network issues:**

```bash
# Restart sensor gateway pods
kubectl rollout restart deployment/sensor-gateway -n greenlang-gl008

# Increase timeout values temporarily
kubectl set env deployment/sensor-gateway -n greenlang-gl008 \
  SENSOR_TIMEOUT=60s \
  RETRY_ATTEMPTS=5

# Enable fallback connectivity mode (cellular if available)
curl -X POST https://api.greenlang.io/v1/steam-trap/admin/enable-fallback-connectivity \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

**For firmware issues:**

```bash
# Rollback problematic firmware version
./scripts/rollback-sensor-firmware.sh \
  --firmware-version=$PROBLEMATIC_VERSION \
  --target-version=$STABLE_VERSION

# Monitor rollback progress
watch -n 30 './scripts/sensor-firmware-status.sh'
```

**For power issues:**

```bash
# Switch affected sensors to low-power mode
./scripts/enable-low-power-mode.sh --site-id=$SITE_ID

# Schedule technician dispatch
./scripts/create-maintenance-ticket.sh \
  --site-id=$SITE_ID \
  --priority=high \
  --issue="Power supply issue - sensors offline"
```

#### Phase 4: Recovery Verification (45-60 minutes)

```bash
# Monitor sensor recovery
watch -n 15 'curl -s https://api.greenlang.io/v1/steam-trap/sensors/status | jq ".online_sensors_count"'

# Test data flow from recovered sensors
./scripts/test-sensor-data-flow.sh --recently-recovered

# Verify inspection jobs completing
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=50 | grep "Inspection completed"
```

**Escalation Criteria:**
- >50% of sensors offline for >1 hour
- Site network unresponsive for >30 minutes
- No recovery progress after mitigation attempts

**Escalate to:** Network Engineering + Site Operations Manager

---

### Scenario 3: False Positive Surge

**Symptoms:**
- False positive rate >30% (baseline: 5-10%)
- Customer complaints about incorrect failure alerts
- Many traps marked as "failed" but operating normally

**Likely Causes:**
- ML model drift
- Changes in operating conditions (pressure, temperature)
- Acoustic interference (new equipment nearby)
- Calibration issues

**Response Procedure:**

#### Phase 1: Quantify Impact (0-15 minutes)

```bash
# Calculate current false positive rate
psql $DB_URL -c "
  SELECT
    DATE(detected_at) as date,
    COUNT(*) as total_failures,
    SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
  FROM trap_inspections
  WHERE detected_at > NOW() - INTERVAL '7 days'
    AND status = 'FAILED'
  GROUP BY DATE(detected_at)
  ORDER BY date DESC;
"

# Compare to baseline
curl https://api.greenlang.io/v1/steam-trap/metrics/false-positive-trend?days=30

# Identify most affected sites
psql $DB_URL -c "
  SELECT
    s.site_name,
    COUNT(*) as false_positives,
    COUNT(DISTINCT t.trap_id) as affected_traps
  FROM trap_inspections ti
  JOIN traps t ON ti.trap_id = t.id
  JOIN sites s ON t.site_id = s.id
  WHERE ti.detected_at > NOW() - INTERVAL '24 hours'
    AND ti.status = 'FAILED'
    AND ti.verified_status = 'FALSE_POSITIVE'
  GROUP BY s.site_name
  ORDER BY false_positives DESC
  LIMIT 10;
"
```

#### Phase 2: Root Cause Analysis (15-30 minutes)

```bash
# Check ML model metrics
curl https://api.greenlang.io/v1/steam-trap/ml/model-metrics | jq '.accuracy, .precision, .recall'

# Review feature distributions
python scripts/analyze_feature_drift.py --lookback-days=7

# Check for environmental changes
psql $DB_URL -c "
  SELECT
    trap_id,
    AVG(ambient_temp) as avg_temp,
    AVG(ambient_pressure) as avg_pressure,
    STDDEV(acoustic_baseline) as noise_variation
  FROM inspection_telemetry
  WHERE captured_at > NOW() - INTERVAL '7 days'
  GROUP BY trap_id
  HAVING STDDEV(acoustic_baseline) > 10.0
  ORDER BY noise_variation DESC
  LIMIT 20;
"

# Review recent model deployments
kubectl describe deployment steam-trap-inspector -n greenlang-gl008 | grep -A 5 "ML_MODEL_VERSION"
```

#### Phase 3: Immediate Mitigation (30-60 minutes)

**Option A: Increase Confidence Threshold**

```bash
# Gradually increase threshold
for threshold in 0.80 0.85 0.90; do
  echo "Testing threshold: $threshold"

  kubectl patch configmap ml-model-config -n greenlang-gl008 \
    --type merge \
    -p "{\"data\":{\"CONFIDENCE_THRESHOLD\":\"$threshold\"}}"

  kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
  kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

  sleep 300  # Wait 5 minutes

  # Check new FP rate
  new_fp_rate=$(curl -s https://api.greenlang.io/v1/steam-trap/metrics/false-positive-rate | jq -r '.current_rate')
  echo "New FP rate: $new_fp_rate%"

  if (( $(echo "$new_fp_rate < 15" | bc -l) )); then
    echo "Acceptable FP rate achieved at threshold $threshold"
    break
  fi
done
```

**Option B: Enable Multi-Factor Verification**

```bash
# Require both acoustic AND thermal confirmation
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  VERIFICATION_MODE=multi_sensor \
  REQUIRED_CONFIRMATIONS=2

# Restart to apply
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

**Option C: Rollback ML Model**

```bash
# Rollback to previous stable model version
./scripts/rollback-ml-model.sh --target-version=v2.4.1

# Verify rollback
kubectl get configmap ml-model-config -n greenlang-gl008 -o yaml | grep MODEL_VERSION
```

#### Phase 4: Long-Term Fix (1-4 hours)

```bash
# Retrain model with recent data
python scripts/retrain_model.py \
  --training-data=last_30_days \
  --include-false-positives \
  --validation-split=0.2

# Evaluate new model
python scripts/evaluate_model.py \
  --model-path=models/steam_trap_v2.5.0.pkl \
  --test-data=validation_set.csv

# Deploy new model to staging
kubectl set image deployment/steam-trap-inspector-staging \
  steam-trap-inspector=greenlang/steam-trap-inspector:v2.5.0 \
  -n greenlang-staging

# Test in staging for 24 hours before production deployment
```

**Escalation Criteria:**
- False positive rate >50%
- Unable to reduce FP rate below 20% within 4 hours
- Customer escalations to executive level

**Escalate to:** ML Engineering Lead + Product VP

---

### Scenario 4: Energy Calculation Anomaly

**Symptoms:**
- Energy loss calculations incorrect (>10% deviation from expected)
- Negative energy values reported
- Inconsistent energy units (kWh vs. BTU)

**Likely Causes:**
- Bug in energy calculation algorithm
- Incorrect steam property data (pressure/temperature)
- Unit conversion errors
- Database data corruption

**Response Procedure:**

#### Phase 1: Validate Issue (0-10 minutes)

```bash
# Check recent energy calculations
psql $DB_URL -c "
  SELECT
    trap_id,
    inspection_id,
    energy_loss_kwh,
    steam_pressure_psi,
    steam_temp_f,
    calculated_at
  FROM energy_calculations
  WHERE calculated_at > NOW() - INTERVAL '1 hour'
  ORDER BY ABS(energy_loss_kwh) DESC
  LIMIT 20;
"

# Look for negative or extreme values
psql $DB_URL -c "
  SELECT
    COUNT(*) as anomaly_count,
    MIN(energy_loss_kwh) as min_energy,
    MAX(energy_loss_kwh) as max_energy,
    AVG(energy_loss_kwh) as avg_energy
  FROM energy_calculations
  WHERE calculated_at > NOW() - INTERVAL '24 hours'
    AND (energy_loss_kwh < 0 OR energy_loss_kwh > 10000);
"

# Check calculation service logs
kubectl logs -n greenlang-gl008 -l component=energy-calculator --tail=200 | grep -E "(ERROR|NaN|Infinity)"
```

#### Phase 2: Isolate Root Cause (10-30 minutes)

**Test calculation algorithm:**

```bash
# Run test calculations with known inputs
python scripts/test_energy_calculations.py --test-suite=validation

# Expected output:
# ✓ Test 1: 150 psi, 366°F, 10 lb/hr leak → 143.7 kWh/year
# ✓ Test 2: 100 psi, 338°F, 5 lb/hr leak → 71.8 kWh/year
# ✗ Test 3: 200 psi, 388°F, 20 lb/hr leak → -287.4 kWh/year (EXPECTED: 287.4)
```

**Check steam property tables:**

```bash
# Verify steam table data integrity
psql $DB_URL -c "
  SELECT
    pressure_psi,
    temp_fahrenheit,
    enthalpy_btu_lb,
    entropy_btu_lb_f
  FROM steam_properties
  WHERE pressure_psi IN (100, 150, 200)
  ORDER BY pressure_psi;
"

# Compare against NIST reference data
./scripts/validate-steam-tables.sh --reference=NIST_IAPWS
```

**Review recent code changes:**

```bash
# Check recent commits to energy calculation module
git log --since="7 days ago" --oneline -- src/energy_calculator/

# Review specific commit if suspicious
git show <commit_hash>
```

#### Phase 3: Apply Fix (30-60 minutes)

**For algorithm bug:**

```bash
# Apply hotfix patch
git cherry-pick <fix_commit_hash>

# Build and deploy
docker build -t greenlang/steam-trap-inspector:hotfix-energy-calc .
kubectl set image deployment/steam-trap-inspector \
  steam-trap-inspector=greenlang/steam-trap-inspector:hotfix-energy-calc \
  -n greenlang-gl008

# Monitor deployment
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008
```

**For data corruption:**

```bash
# Backup current steam property table
pg_dump $DB_URL -t steam_properties > steam_properties_backup.sql

# Restore from known-good backup
psql $DB_URL < steam_properties_reference_2025-11-01.sql

# Verify restoration
./scripts/validate-steam-tables.sh --reference=NIST_IAPWS
```

**For unit conversion error:**

```bash
# Update unit conversion constants
kubectl patch configmap energy-calc-config -n greenlang-gl008 \
  --type merge \
  -p '{"data":{
    "BTU_TO_KWH": "0.000293071",
    "LB_TO_KG": "0.453592",
    "PSI_TO_KPA": "6.89476"
  }}'

# Restart pods
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

#### Phase 4: Recalculate Affected Records (1-2 hours)

```bash
# Identify affected calculations
psql $DB_URL -c "
  CREATE TEMP TABLE affected_calculations AS
  SELECT id, trap_id, inspection_id, calculated_at
  FROM energy_calculations
  WHERE calculated_at > '2025-11-20'  -- Date when issue started
    AND (energy_loss_kwh < 0 OR energy_loss_kwh > 10000);
"

# Trigger recalculation job
kubectl create job recalculate-energy-$(date +%s) \
  --from=cronjob/energy-recalculation \
  -n greenlang-gl008

# Monitor recalculation progress
kubectl logs -n greenlang-gl008 -l job-name=recalculate-energy-* --follow
```

#### Phase 5: Validate Fix (2-3 hours)

```bash
# Run comprehensive validation
./scripts/validate-energy-calculations.sh --date-range="last_7_days"

# Compare before/after
psql $DB_URL -c "
  WITH before_fix AS (
    SELECT AVG(energy_loss_kwh) as avg_before
    FROM energy_calculations
    WHERE calculated_at BETWEEN '2025-11-20' AND '2025-11-25'
  ),
  after_fix AS (
    SELECT AVG(energy_loss_kwh) as avg_after
    FROM energy_calculations
    WHERE calculated_at > '2025-11-25'
  )
  SELECT
    avg_before,
    avg_after,
    ROUND(100.0 * (avg_after - avg_before) / avg_before, 2) as pct_change
  FROM before_fix, after_fix;
"

# Customer notification of corrected data
./scripts/notify-customers.sh \
  --template=energy_calculation_correction \
  --affected-date-range="2025-11-20_to_2025-11-25"
```

**Escalation Criteria:**
- Unable to identify root cause within 2 hours
- Issue affects >1000 calculations
- Financial impact >$50K in incorrect energy loss reports

**Escalate to:** Engineering Manager + Product Owner + Finance

---

### Scenario 5: ML Model Degradation

**Symptoms:**
- Model accuracy drops below 80% (baseline: 92-95%)
- Increased prediction latency
- Model inference errors in logs

**Likely Causes:**
- Concept drift (trap behavior changes)
- Training data staleness
- Resource constraints (CPU/memory)
- Model corruption

**Response Procedure:**

#### Phase 1: Assess Degradation (0-15 minutes)

```bash
# Check current model metrics
curl https://api.greenlang.io/v1/steam-trap/ml/metrics | jq '.'

# Expected output:
# {
#   "accuracy": 0.78,  # DOWN from 0.93
#   "precision": 0.75,
#   "recall": 0.82,
#   "f1_score": 0.78,
#   "inference_time_ms": 450,  # UP from 150ms
#   "model_version": "v2.4.2"
# }

# Check model performance trend
psql $DB_URL -c "
  SELECT
    DATE(prediction_time) as date,
    AVG(CASE WHEN prediction = actual_status THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(inference_time_ms) as avg_latency,
    COUNT(*) as prediction_count
  FROM ml_predictions
  WHERE prediction_time > NOW() - INTERVAL '30 days'
    AND actual_status IS NOT NULL
  GROUP BY DATE(prediction_time)
  ORDER BY date DESC
  LIMIT 30;
"

# Check for resource constraints
kubectl top pods -n greenlang-gl008 -l component=ml-service
```

#### Phase 2: Diagnose Root Cause (15-30 minutes)

**Check for concept drift:**

```bash
# Analyze feature distributions
python scripts/detect_concept_drift.py \
  --baseline-period="2025-09-01:2025-10-01" \
  --current-period="2025-11-01:2025-11-26"

# Expected output:
# Feature drift detected:
#   - acoustic_frequency_mean: KS statistic = 0.18 (SIGNIFICANT)
#   - thermal_delta: KS statistic = 0.22 (SIGNIFICANT)
#   - pressure_variance: KS statistic = 0.08 (OK)
```

**Check model health:**

```bash
# Validate model file integrity
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  python -c "import pickle; pickle.load(open('/models/steam_trap_v2.4.2.pkl', 'rb'))"

# Check model size (corruption indicator)
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  ls -lh /models/steam_trap_v2.4.2.pkl

# Expected: ~45MB, if significantly different → corruption
```

**Check resource availability:**

```bash
# Review resource requests vs. limits
kubectl describe pod -n greenlang-gl008 -l component=ml-service | grep -A 5 "Limits:"

# Check for OOMKilled events
kubectl get events -n greenlang-gl008 --field-selector reason=OOMKilled --sort-by='.lastTimestamp'
```

#### Phase 3: Immediate Mitigation (30-60 minutes)

**Option A: Rollback to Previous Model**

```bash
# Rollback to last known-good model version
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ML_MODEL_VERSION=v2.4.1

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
kubectl rollout status deployment/steam-trap-inspector -n greenlang-gl008

# Verify accuracy improvement
sleep 300
curl https://api.greenlang.io/v1/steam-trap/ml/metrics | jq '.accuracy'
```

**Option B: Scale Resources**

```bash
# Increase CPU/memory allocation
kubectl patch deployment steam-trap-inspector -n greenlang-gl008 --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/cpu",
    "value": "2000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/memory",
    "value": "4Gi"
  }
]'
```

**Option C: Enable Ensemble Fallback**

```bash
# Use ensemble of previous models
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  PREDICTION_MODE=ensemble \
  ENSEMBLE_MODELS=v2.4.1,v2.3.5,v2.4.0

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

#### Phase 4: Long-Term Remediation (2-8 hours)

```bash
# Retrain model with recent data
python scripts/retrain_model.py \
  --training-data=last_60_days \
  --validation-split=0.2 \
  --feature-engineering=auto \
  --hyperparameter-tuning=bayesian

# Evaluate new model
python scripts/evaluate_model.py \
  --model-path=models/steam_trap_v2.5.0.pkl \
  --test-data=validation_set.csv \
  --min-accuracy=0.90

# Expected output:
# Model Evaluation Results:
#   Accuracy: 0.94
#   Precision: 0.93
#   Recall: 0.95
#   F1 Score: 0.94
#   Inference Time: 120ms avg
#   ✓ All metrics pass minimum thresholds

# Deploy to staging
kubectl set image deployment/steam-trap-inspector-staging \
  steam-trap-inspector=greenlang/steam-trap-inspector:v2.5.0 \
  -n greenlang-staging

# Monitor staging for 24-48 hours
./scripts/monitor-model-performance.sh \
  --environment=staging \
  --duration=48h \
  --alert-threshold=0.90
```

**Escalation Criteria:**
- Accuracy below 75% after rollback
- Unable to retrain model within 8 hours
- Ongoing concept drift requiring architecture changes

**Escalate to:** ML Engineering Lead + Data Science Team

---

### Scenario 6: Database Connection Failure

**Symptoms:**
- Database connection errors in application logs
- Inspection jobs failing with DB errors
- API endpoints returning 500 errors

**Likely Causes:**
- Database server down
- Connection pool exhausted
- Network connectivity issues
- Database credential rotation failure

**Response Procedure:**

#### Phase 1: Confirm DB Status (0-5 minutes)

```bash
# Test database connectivity
psql $DB_URL -c "SELECT 1;"

# Check connection pool status
curl https://api.greenlang.io/v1/steam-trap/admin/db-pool-status | jq '.'

# Expected output:
# {
#   "pool_size": 20,
#   "active_connections": 18,
#   "idle_connections": 2,
#   "waiting_clients": 45  # PROBLEM: waiting > 0
# }

# Check PostgreSQL server status
kubectl get pods -n database -l app=postgresql
```

#### Phase 2: Quick Remediation (5-15 minutes)

**If connection pool exhausted:**

```bash
# Increase pool size temporarily
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  DB_POOL_SIZE=50 \
  DB_POOL_TIMEOUT=30

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

**If database pod unhealthy:**

```bash
# Restart PostgreSQL pod
kubectl rollout restart statefulset/postgresql -n database

# Monitor recovery
kubectl rollout status statefulset/postgresql -n database
```

**If credential rotation failed:**

```bash
# Update database credentials from secret manager
./scripts/rotate-db-credentials.sh --force-update

# Restart application pods
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

#### Phase 3: Verify Recovery (15-30 minutes)

```bash
# Test end-to-end workflow
curl -X POST https://api.greenlang.io/v1/steam-trap/inspection \
  -H "Authorization: Bearer $API_TOKEN" \
  -d @test-inspection.json

# Monitor error rate
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=100 | grep "database" | grep ERROR
```

**Escalation Criteria:**
- Database unreachable for >15 minutes
- Data loss suspected
- Replica lag >10 minutes

**Escalate to:** Database Administrator + Infrastructure Team

---

### Scenario 7: High Latency

**Symptoms:**
- API response times >5 seconds (baseline: <1s)
- Inspection jobs taking >60 seconds (baseline: 10-15s)
- Customer complaints about slow performance

**Likely Causes:**
- Resource contention (CPU/memory)
- Database slow queries
- External API dependency slowdown
- Network bottlenecks

**Response Procedure:**

#### Phase 1: Identify Bottleneck (0-15 minutes)

```bash
# Check API latency by endpoint
curl https://api.greenlang.io/v1/steam-trap/metrics/latency | jq '.'

# Check database query performance
psql $DB_URL -c "
  SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time
  FROM pg_stat_statements
  WHERE mean_exec_time > 1000
  ORDER BY mean_exec_time DESC
  LIMIT 10;
"

# Check pod resource utilization
kubectl top pods -n greenlang-gl008

# Check for CPU throttling
kubectl describe pods -n greenlang-gl008 -l app=steam-trap-inspector | grep -A 5 "cpu"
```

#### Phase 2: Quick Wins (15-30 minutes)

```bash
# Scale up replicas
kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=6

# Add read replicas for DB queries
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  DB_READ_REPLICA_URL=postgresql://read-replica.db.svc.cluster.local:5432/greenlang
```

#### Phase 3: Optimize (30-90 minutes)

```bash
# Add missing database indexes
psql $DB_URL -c "
  CREATE INDEX CONCURRENTLY idx_inspections_trap_id_date
  ON trap_inspections(trap_id, detected_at DESC);
"

# Enable query caching
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ENABLE_QUERY_CACHE=true \
  CACHE_TTL=300
```

**Escalation Criteria:**
- Latency >10 seconds after scaling
- No identifiable bottleneck within 2 hours

**Escalate to:** Performance Engineering Team

---

### Scenario 8: Security Breach

**Symptoms:**
- Unauthorized access attempts logged
- Unusual API traffic patterns
- Data exfiltration alerts
- Compromised credentials detected

**Response Procedure:**

#### Phase 1: IMMEDIATE CONTAINMENT (0-5 minutes)

```bash
# STOP: Do not delay - execute immediately

# 1. Revoke all API tokens
./scripts/emergency-revoke-tokens.sh --all

# 2. Enable IP allowlist (emergency mode)
kubectl set env deployment/api-gateway -n greenlang-gl008 \
  SECURITY_MODE=lockdown \
  ALLOWED_IPS=10.0.0.0/8,172.16.0.0/12

# 3. Disable public endpoints
kubectl scale deployment/api-gateway -n greenlang-gl008 --replicas=0

# 4. Alert security team
./scripts/security-alert.sh --priority=P0 --incident="GL-008 Security Breach Suspected"
```

#### Phase 2: Investigate (5-30 minutes)

```bash
# Review access logs
kubectl logs -n greenlang-gl008 -l component=api-gateway --since=24h | grep -E "(401|403|suspicious)"

# Check for data exfiltration
psql $DB_URL -c "
  SELECT
    user_id,
    COUNT(*) as request_count,
    SUM(response_size_bytes) as total_data_transferred
  FROM api_access_logs
  WHERE timestamp > NOW() - INTERVAL '24 hours'
  GROUP BY user_id
  ORDER BY total_data_transferred DESC
  LIMIT 20;
"

# Preserve evidence
./scripts/capture-forensics.sh --incident-id=$INCIDENT_ID
```

#### Phase 3: Remediation (30+ minutes)

**Follow company security incident response policy. Do not proceed without Security Team approval.**

**Escalation:** IMMEDIATE escalation to CISO + Legal + Incident Response Team

---

## Escalation Matrix

### On-Call Rotation

| Role | Primary | Secondary | Manager |
|------|---------|-----------|---------|
| **Platform Engineer** | @engineer-1 | @engineer-2 | @eng-manager |
| **Database Admin** | @dba-1 | @dba-2 | @dba-manager |
| **ML Engineer** | @ml-eng-1 | @ml-eng-2 | @ml-manager |
| **Security** | @security-1 | @security-2 | @ciso |
| **Product** | @product-owner | @product-manager | @vp-product |

### Escalation Paths

**P0 Incidents:**
- **0-15 min:** On-call Platform Engineer
- **15-30 min:** Engineering Manager + Product Owner
- **30-60 min:** VP Engineering + VP Product
- **60+ min:** CTO + Customer Success VP

**P1 Incidents:**
- **0-30 min:** On-call Platform Engineer
- **30-90 min:** Engineering Manager
- **90+ min:** VP Engineering

**P2 Incidents:**
- **0-2 hours:** On-call Platform Engineer
- **2-4 hours:** Engineering Manager
- **4+ hours:** VP Engineering (if customer-impacting)

**Security Incidents (Any Priority):**
- **Immediate:** Security Team + CISO + Legal

### Contact Methods

**Emergency Contact:**
- PagerDuty: https://greenlang.pagerduty.com
- Phone: +1-800-GREENLANG
- Slack: #incident-response

**During Business Hours:**
- Slack: #gl008-support
- Email: gl008-support@greenlang.io
- Jira: https://greenlang.atlassian.net

---

## Post-Incident Review

### Timeline

- **P0/P1:** Within 48 hours
- **P2:** Within 1 week
- **P3/P4:** As needed

### PIR Template

```markdown
# Post-Incident Review: [INCIDENT TITLE]

**Incident ID:** INC-XXXXX
**Date:** YYYY-MM-DD
**Duration:** X hours Y minutes
**Severity:** PX
**Incident Commander:** @name

## Summary

[2-3 sentence summary of what happened]

## Impact

- **Customer Impact:** X sites affected, Y inspections failed
- **Financial Impact:** $Z estimated
- **Reputation Impact:** [High/Medium/Low]

## Timeline

| Time (UTC) | Event |
|------------|-------|
| HH:MM | Alert triggered |
| HH:MM | Engineer acknowledged |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident closed |

## Root Cause

[Detailed explanation of what caused the incident]

## What Went Well

- [Thing 1]
- [Thing 2]

## What Went Poorly

- [Thing 1]
- [Thing 2]

## Action Items

| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| [Action 1] | @owner | YYYY-MM-DD | High |
| [Action 2] | @owner | YYYY-MM-DD | Medium |

## Lessons Learned

[Key takeaways for future incidents]
```

---

## Communication Templates

### Initial Customer Notification (P0/P1)

```
Subject: [Action Required] GL-008 Steam Trap Inspector Service Disruption

Dear [Customer Name],

We are currently experiencing a service disruption with the GL-008 Steam Trap Inspector application.

**Impact:** [Describe specific impact to customer]
**Status:** Our engineering team is actively working to resolve the issue
**Estimated Resolution:** [Time estimate or "Under investigation"]

**What You Can Do:**
- [Workaround if available, or "No action required at this time"]

We will provide updates every [frequency]. You can check real-time status at:
https://status.greenlang.io/incidents/[incident-id]

For urgent assistance, contact: support@greenlang.io or +1-800-GREENLANG

We apologize for the inconvenience.

GreenLang Operations Team
```

### Resolution Notification

```
Subject: [Resolved] GL-008 Steam Trap Inspector Service Restored

Dear [Customer Name],

The service disruption with GL-008 Steam Trap Inspector has been resolved.

**Issue:** [Brief description]
**Resolution Time:** [Duration]
**Next Steps:** [Any actions customers should take]

**Post-Incident Report:** A detailed analysis will be shared within 48 hours.

If you continue to experience issues, please contact support@greenlang.io.

Thank you for your patience.

GreenLang Operations Team
```

---

## Appendix: Quick Reference

### Health Check Commands

```bash
# Overall system health
curl https://api.greenlang.io/v1/steam-trap/health

# Component health
kubectl get pods -n greenlang-gl008
kubectl get svc -n greenlang-gl008
kubectl top pods -n greenlang-gl008

# Database health
psql $DB_URL -c "SELECT 1;"

# Sensor connectivity
curl https://api.greenlang.io/v1/steam-trap/sensors/status
```

### Log Aggregation

```bash
# All errors in last hour
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=1h | grep ERROR

# Specific component
kubectl logs -n greenlang-gl008 -l component=energy-calculator --tail=200

# Follow logs
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --follow
```

### Metrics Dashboards

- **Overview:** https://grafana.greenlang.io/d/gl008-overview
- **Performance:** https://grafana.greenlang.io/d/gl008-performance
- **ML Metrics:** https://grafana.greenlang.io/d/gl008-ml-metrics
- **Infrastructure:** https://grafana.greenlang.io/d/gl008-infrastructure

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
**Maintained By:** Platform Operations Team
