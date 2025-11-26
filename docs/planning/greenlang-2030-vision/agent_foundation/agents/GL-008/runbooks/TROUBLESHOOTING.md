# GL-008 SteamTrapInspector - Troubleshooting Guide

**Version:** 1.0.0
**Last Updated:** 2025-11-26
**Owner:** Platform Operations Team

---

## Table of Contents

1. [Common Issues](#common-issues)
2. [Sensor Issues](#sensor-issues)
3. [Performance Issues](#performance-issues)
4. [Data Quality Issues](#data-quality-issues)
5. [Integration Issues](#integration-issues)
6. [ML Model Issues](#ml-model-issues)
7. [Database Issues](#database-issues)
8. [Network Issues](#network-issues)
9. [Diagnostic Tools](#diagnostic-tools)
10. [Advanced Troubleshooting](#advanced-troubleshooting)

---

## Common Issues

### Issue 1: Inspection Jobs Failing

**Symptoms:**
- Inspection jobs stuck in "pending" or "processing" status
- Jobs timeout after 60 seconds
- Error: "Inspection failed: timeout exceeded"

**Diagnostic Steps:**

```bash
# Check job queue status
curl https://api.greenlang.io/v1/steam-trap/jobs/queue-status | jq '.'

# Expected healthy output:
# {
#   "pending": 5,
#   "processing": 12,
#   "completed": 1543,
#   "failed": 3,
#   "avg_processing_time_sec": 15
# }

# Check worker pod status
kubectl get pods -n greenlang-gl008 -l component=worker

# Check for resource constraints
kubectl top pods -n greenlang-gl008 -l component=worker

# Review failed job logs
psql $DB_URL -c "
  SELECT
    id,
    trap_id,
    status,
    error_message,
    created_at,
    updated_at
  FROM inspection_jobs
  WHERE status = 'FAILED'
    AND created_at > NOW() - INTERVAL '1 hour'
  ORDER BY created_at DESC
  LIMIT 10;
"
```

**Common Causes & Resolutions:**

**Cause 1: Insufficient Worker Capacity**

```bash
# Check current worker count
kubectl get deployment worker -n greenlang-gl008 -o jsonpath='{.spec.replicas}'

# Scale up workers
kubectl scale deployment/worker -n greenlang-gl008 --replicas=10

# Enable autoscaling
kubectl autoscale deployment worker -n greenlang-gl008 \
  --min=5 --max=20 --cpu-percent=70
```

**Cause 2: Sensor Timeout**

```bash
# Increase sensor timeout
kubectl set env deployment/worker -n greenlang-gl008 \
  SENSOR_TIMEOUT=90s \
  SENSOR_RETRY_ATTEMPTS=5

# Restart workers
kubectl rollout restart deployment/worker -n greenlang-gl008
```

**Cause 3: Database Connection Pool Exhausted**

```bash
# Check connection pool
curl https://api.greenlang.io/v1/steam-trap/admin/db-pool-status | jq '.'

# Increase pool size
kubectl set env deployment/worker -n greenlang-gl008 \
  DB_POOL_SIZE=30 \
  DB_POOL_TIMEOUT=60

kubectl rollout restart deployment/worker -n greenlang-gl008
```

**Verification:**

```bash
# Monitor job completion rate
watch -n 10 'curl -s https://api.greenlang.io/v1/steam-trap/jobs/queue-status | jq ".completed"'

# Check error logs
kubectl logs -n greenlang-gl008 -l component=worker --tail=50 | grep -v "INFO"
```

---

### Issue 2: High False Positive Rate

**Symptoms:**
- More than 20% of failure alerts are false positives
- Customer complaints about incorrect alerts
- Manual verification contradicts automated detection

**Diagnostic Steps:**

```bash
# Calculate current false positive rate
psql $DB_URL -c "
  WITH fp_stats AS (
    SELECT
      DATE(detected_at) as date,
      COUNT(*) as total_failures,
      SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
      SUM(CASE WHEN verified_status = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) as true_positives,
      SUM(CASE WHEN verified_status IS NULL THEN 1 ELSE 0 END) as unverified
    FROM trap_inspections
    WHERE detected_at > NOW() - INTERVAL '30 days'
      AND status = 'FAILED'
    GROUP BY DATE(detected_at)
  )
  SELECT
    date,
    total_failures,
    false_positives,
    true_positives,
    unverified,
    ROUND(100.0 * false_positives / NULLIF(false_positives + true_positives, 0), 2) as fp_rate
  FROM fp_stats
  ORDER BY date DESC
  LIMIT 14;
"

# Check ML model confidence scores
psql $DB_URL -c "
  SELECT
    CASE
      WHEN confidence_score < 0.7 THEN '< 0.7'
      WHEN confidence_score < 0.8 THEN '0.7-0.8'
      WHEN confidence_score < 0.9 THEN '0.8-0.9'
      ELSE '0.9+'
    END as confidence_range,
    COUNT(*) as count,
    SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
  FROM trap_inspections
  WHERE detected_at > NOW() - INTERVAL '7 days'
    AND status = 'FAILED'
    AND verified_status IS NOT NULL
  GROUP BY 1
  ORDER BY confidence_range;
"

# Check for sensor-specific issues
psql $DB_URL -c "
  SELECT
    s.sensor_id,
    s.sensor_type,
    COUNT(*) as detections,
    SUM(CASE WHEN ti.verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    ROUND(100.0 * SUM(CASE WHEN ti.verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
  FROM trap_inspections ti
  JOIN sensors s ON ti.sensor_id = s.id
  WHERE ti.detected_at > NOW() - INTERVAL '7 days'
    AND ti.status = 'FAILED'
    AND ti.verified_status IS NOT NULL
  GROUP BY s.sensor_id, s.sensor_type
  HAVING COUNT(*) > 10
  ORDER BY fp_rate DESC
  LIMIT 20;
"
```

**Resolution Steps:**

**Step 1: Adjust Confidence Threshold**

```bash
# Current threshold
kubectl get configmap ml-model-config -n greenlang-gl008 -o yaml | grep CONFIDENCE_THRESHOLD

# Increase threshold gradually
kubectl patch configmap ml-model-config -n greenlang-gl008 \
  --type merge \
  -p '{"data":{"CONFIDENCE_THRESHOLD":"0.85"}}'

# Restart to apply
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008

# Monitor impact for 1 hour
sleep 3600

# Check new FP rate
psql $DB_URL -c "
  SELECT
    COUNT(*) as total,
    SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as fp,
    ROUND(100.0 * SUM(CASE WHEN verified_status = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) / COUNT(*), 2) as fp_rate
  FROM trap_inspections
  WHERE detected_at > NOW() - INTERVAL '1 hour'
    AND status = 'FAILED'
    AND verified_status IS NOT NULL;
"
```

**Step 2: Enable Multi-Sensor Verification**

```bash
# Require both acoustic AND thermal confirmation for failures
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  VERIFICATION_MODE=multi_sensor \
  REQUIRED_SENSOR_AGREEMENT=2

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

**Step 3: Recalibrate Problematic Sensors**

```bash
# Identify sensors with high FP rate
sensors_to_calibrate=$(psql $DB_URL -t -c "
  SELECT array_to_json(array_agg(sensor_id))
  FROM (
    SELECT s.sensor_id
    FROM trap_inspections ti
    JOIN sensors s ON ti.sensor_id = s.id
    WHERE ti.detected_at > NOW() - INTERVAL '7 days'
      AND ti.status = 'FAILED'
      AND ti.verified_status = 'FALSE_POSITIVE'
    GROUP BY s.sensor_id
    HAVING COUNT(*) > 10
  ) sub;
")

# Run calibration for these sensors
./scripts/calibrate-sensors.sh --sensor-ids="$sensors_to_calibrate"
```

**Step 4: Retrain ML Model (if threshold adjustment insufficient)**

```bash
# Collect recent false positives for training
psql $DB_URL -c "
  COPY (
    SELECT
      ti.acoustic_features,
      ti.thermal_features,
      ti.pressure_features,
      CASE WHEN ti.verified_status = 'FALSE_POSITIVE' THEN 0 ELSE 1 END as label
    FROM trap_inspections ti
    WHERE ti.detected_at > NOW() - INTERVAL '60 days'
      AND ti.verified_status IS NOT NULL
  ) TO '/tmp/retraining_data.csv' WITH CSV HEADER;
"

# Retrain model
python scripts/retrain_model.py \
  --training-data=/tmp/retraining_data.csv \
  --class-weights=balanced \
  --validation-split=0.2 \
  --min-accuracy=0.92

# Deploy new model to staging first
kubectl set image deployment/steam-trap-inspector-staging \
  steam-trap-inspector=greenlang/steam-trap-inspector:v2.5.0-fp-fix \
  -n greenlang-staging

# Monitor staging for 48 hours before production
```

---

### Issue 3: Slow Inspection Times

**Symptoms:**
- Inspections taking >30 seconds (baseline: 10-15s)
- Queue backlog building up
- Customer complaints about delays

**Diagnostic Steps:**

```bash
# Check average inspection time
psql $DB_URL -c "
  SELECT
    DATE(created_at) as date,
    COUNT(*) as inspections,
    AVG(EXTRACT(EPOCH FROM (updated_at - created_at))) as avg_duration_sec,
    MAX(EXTRACT(EPOCH FROM (updated_at - created_at))) as max_duration_sec
  FROM inspection_jobs
  WHERE created_at > NOW() - INTERVAL '7 days'
    AND status = 'COMPLETED'
  GROUP BY DATE(created_at)
  ORDER BY date DESC;
"

# Profile inspection time by component
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --tail=1000 | \
  grep "duration_ms" | \
  jq -r '[.component, .duration_ms] | @csv'

# Check resource utilization
kubectl top pods -n greenlang-gl008 -l app=steam-trap-inspector

# Check for slow database queries
psql $DB_URL -c "
  SELECT
    query,
    calls,
    mean_exec_time,
    max_exec_time
  FROM pg_stat_statements
  WHERE query LIKE '%trap_inspections%'
  ORDER BY mean_exec_time DESC
  LIMIT 10;
"
```

**Resolution Steps:**

**Step 1: Optimize Database Queries**

```bash
# Add missing indexes
psql $DB_URL -c "
  -- Index for trap lookup
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traps_site_id_status
  ON traps(site_id, status) WHERE status = 'ACTIVE';

  -- Index for recent inspections
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inspections_trap_id_date
  ON trap_inspections(trap_id, detected_at DESC);

  -- Index for sensor data queries
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_readings_sensor_id_timestamp
  ON sensor_readings(sensor_id, captured_at DESC);
"

# Analyze tables for query planner
psql $DB_URL -c "
  ANALYZE trap_inspections;
  ANALYZE sensor_readings;
  ANALYZE traps;
"
```

**Step 2: Enable Caching**

```bash
# Enable Redis caching for sensor data
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ENABLE_CACHE=true \
  CACHE_BACKEND=redis \
  CACHE_TTL=300 \
  REDIS_URL=redis://redis.greenlang-gl008.svc.cluster.local:6379

# Restart to apply
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

**Step 3: Optimize ML Inference**

```bash
# Enable batch inference (process multiple traps together)
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  BATCH_INFERENCE=true \
  BATCH_SIZE=10 \
  BATCH_TIMEOUT=5s

# Use lighter model for screening, heavy model for confirmation
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  INFERENCE_MODE=hierarchical \
  SCREENING_MODEL=lightweight \
  CONFIRMATION_MODEL=full
```

**Step 4: Scale Resources**

```bash
# Increase CPU allocation
kubectl patch deployment steam-trap-inspector -n greenlang-gl008 --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/cpu",
    "value": "2000m"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/cpu",
    "value": "4000m"
  }
]'

# Scale horizontally
kubectl scale deployment/steam-trap-inspector -n greenlang-gl008 --replicas=8
```

**Verification:**

```bash
# Monitor inspection times
watch -n 30 'psql $DB_URL -t -c "
  SELECT
    COUNT(*) as recent_inspections,
    ROUND(AVG(EXTRACT(EPOCH FROM (updated_at - created_at)))) as avg_duration_sec
  FROM inspection_jobs
  WHERE created_at > NOW() - INTERVAL '\''10 minutes'\''
    AND status = '\''COMPLETED'\'';
"'
```

---

## Sensor Issues

### Issue 4: Acoustic Sensor Failures

**Symptoms:**
- Acoustic sensor status: "OFFLINE" or "DEGRADED"
- No acoustic data in inspection results
- Error: "Acoustic sensor communication timeout"

**Diagnostic Steps:**

```bash
# List offline acoustic sensors
curl https://api.greenlang.io/v1/steam-trap/sensors/status?type=acoustic | jq '.offline_sensors'

# Check sensor last communication
psql $DB_URL -c "
  SELECT
    sensor_id,
    site_id,
    status,
    last_reading_at,
    NOW() - last_reading_at as time_since_last_reading,
    battery_level,
    signal_strength
  FROM sensors
  WHERE sensor_type = 'ACOUSTIC'
    AND status != 'ONLINE'
  ORDER BY last_reading_at
  LIMIT 20;
"

# Check for error patterns
kubectl logs -n greenlang-gl008 -l component=sensor-gateway --tail=500 | \
  grep "acoustic" | \
  grep -E "(ERROR|TIMEOUT)"

# Test connectivity to specific sensor
./scripts/test-sensor-connection.sh --sensor-id=$SENSOR_ID --type=acoustic
```

**Resolution Steps:**

**Step 1: Power Cycle Sensor**

```bash
# Send power cycle command via sensor gateway
curl -X POST https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/power-cycle \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Wait 60 seconds for sensor to reboot
sleep 60

# Check status
curl https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/status | jq '.'
```

**Step 2: Update Sensor Firmware**

```bash
# Check current firmware version
curl https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/info | jq '.firmware_version'

# Check for available updates
curl https://api.greenlang.io/v1/steam-trap/sensors/firmware/available | jq '.'

# Update to latest firmware
curl -X POST https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/update-firmware \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"target_version": "v3.2.1"}'

# Monitor update progress
watch -n 10 'curl -s https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/status | jq ".firmware_update_status"'
```

**Step 3: Recalibrate Sensor**

```bash
# Run acoustic calibration
./scripts/calibrate-acoustic-sensor.sh --sensor-id=$SENSOR_ID

# Verify calibration
curl https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/calibration-status | jq '.'
```

**Step 4: Replace Sensor (if hardware failure)**

```bash
# Mark sensor for replacement
psql $DB_URL -c "
  UPDATE sensors
  SET status = 'NEEDS_REPLACEMENT',
      notes = 'Hardware failure detected $(date +%Y-%m-%d)'
  WHERE sensor_id = '$SENSOR_ID';
"

# Create maintenance ticket
./scripts/create-maintenance-ticket.sh \
  --sensor-id=$SENSOR_ID \
  --issue="Acoustic sensor hardware failure" \
  --priority=high
```

---

### Issue 5: Thermal Camera Issues

**Symptoms:**
- Thermal images corrupted or missing
- Temperature readings outside expected range
- Error: "Thermal camera initialization failed"

**Diagnostic Steps:**

```bash
# Check thermal camera status
psql $DB_URL -c "
  SELECT
    sensor_id,
    site_id,
    status,
    last_image_captured_at,
    avg_image_quality_score,
    error_count_24h
  FROM thermal_cameras
  WHERE status != 'ONLINE'
    OR avg_image_quality_score < 0.7
  ORDER BY error_count_24h DESC;
"

# Review thermal image quality
curl https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/thermal/recent-images | jq '.images[] | {timestamp, quality_score, resolution}'

# Check for lens obstruction or calibration drift
python scripts/analyze_thermal_images.py \
  --sensor-id=$SENSOR_ID \
  --lookback-hours=24 \
  --check-obstruction \
  --check-calibration
```

**Resolution Steps:**

**Step 1: Clean Thermal Camera Lens**

```bash
# Create maintenance alert for on-site cleaning
./scripts/notify-site-maintenance.sh \
  --sensor-id=$SENSOR_ID \
  --task="Clean thermal camera lens" \
  --priority=medium

# Temporarily increase image contrast/brightness
curl -X PATCH https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/thermal/settings \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "contrast": 1.3,
    "brightness": 1.1,
    "auto_exposure": true
  }'
```

**Step 2: Recalibrate Thermal Camera**

```bash
# Run thermal calibration routine
./scripts/calibrate-thermal-camera.sh \
  --sensor-id=$SENSOR_ID \
  --reference-temp=ambient

# Verify calibration
curl https://api.greenlang.io/v1/steam-trap/sensors/$SENSOR_ID/thermal/calibration-status | jq '.'
```

**Step 3: Adjust Thermal Thresholds**

```bash
# Update temperature thresholds based on site conditions
psql $DB_URL -c "
  UPDATE thermal_cameras
  SET
    min_expected_temp = 50,
    max_expected_temp = 450,
    anomaly_threshold = 30
  WHERE sensor_id = '$SENSOR_ID';
"
```

---

### Issue 6: Sensor Calibration Drift

**Symptoms:**
- Detection accuracy declining over time
- Sensor readings don't match manual measurements
- Increasing number of false positives/negatives

**Diagnostic Steps:**

```bash
# Check last calibration dates
psql $DB_URL -c "
  SELECT
    sensor_id,
    sensor_type,
    last_calibrated_at,
    NOW() - last_calibrated_at as time_since_calibration,
    calibration_drift_score
  FROM sensors
  WHERE last_calibrated_at < NOW() - INTERVAL '90 days'
    OR calibration_drift_score > 0.15
  ORDER BY calibration_drift_score DESC;
"

# Analyze calibration drift trends
python scripts/analyze_calibration_drift.py \
  --sensors=all \
  --lookback-days=180

# Compare sensor readings to reference values
./scripts/validate-sensor-accuracy.sh --reference-traps=baseline_set
```

**Resolution Steps:**

**Step 1: Automated Recalibration**

```bash
# Run calibration for all drifted sensors
./scripts/calibrate-sensors.sh \
  --drift-threshold=0.15 \
  --all-types \
  --schedule-off-peak

# Monitor calibration progress
watch -n 60 './scripts/calibration-status.sh'
```

**Step 2: Update Calibration Schedule**

```bash
# Set more frequent calibration for high-use sensors
psql $DB_URL -c "
  UPDATE sensors
  SET calibration_frequency_days = 60
  WHERE sensor_id IN (
    SELECT sensor_id
    FROM sensor_usage_stats
    WHERE daily_reading_count > 100
  );
"

# Enable automated calibration reminders
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  AUTO_CALIBRATION_ENABLED=true \
  CALIBRATION_CHECK_FREQUENCY=daily
```

---

## Performance Issues

### Issue 7: API Response Latency

**Symptoms:**
- API responses taking >2 seconds (baseline: <500ms)
- Timeout errors on client applications
- User complaints about slow dashboard loading

**Diagnostic Steps:**

```bash
# Check API latency metrics
curl https://api.greenlang.io/v1/steam-trap/metrics/latency | jq '.'

# Analyze slow endpoints
kubectl logs -n greenlang-gl008 -l component=api-gateway --tail=1000 | \
  jq -r 'select(.duration_ms > 2000) | [.endpoint, .duration_ms, .timestamp] | @csv' | \
  sort -t, -k2 -rn | head -20

# Check API gateway resource usage
kubectl top pods -n greenlang-gl008 -l component=api-gateway

# Review database query performance
psql $DB_URL -c "
  SELECT
    substring(query, 1, 100) as query_snippet,
    calls,
    mean_exec_time,
    max_exec_time,
    stddev_exec_time
  FROM pg_stat_statements
  WHERE mean_exec_time > 500
  ORDER BY mean_exec_time DESC
  LIMIT 15;
"
```

**Resolution Steps:**

**Step 1: Enable Response Caching**

```bash
# Configure Redis for API response caching
kubectl set env deployment/api-gateway -n greenlang-gl008 \
  CACHE_ENABLED=true \
  CACHE_TTL_SECONDS=300 \
  CACHE_BACKEND=redis \
  REDIS_URL=redis://redis.greenlang-gl008.svc.cluster.local:6379

# Restart API gateway
kubectl rollout restart deployment/api-gateway -n greenlang-gl008
```

**Step 2: Optimize Database Queries**

```bash
# Add query optimization indexes
psql $DB_URL -c "
  -- Optimize trap lookups
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_traps_site_status
  ON traps(site_id, status) INCLUDE (trap_id, location, type);

  -- Optimize recent inspections query
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inspections_date_status
  ON trap_inspections(detected_at DESC, status) INCLUDE (trap_id, confidence_score);

  -- Optimize site analytics
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inspections_site_date
  ON trap_inspections(site_id, DATE(detected_at));
"

# Enable query result caching at DB level
psql $DB_URL -c "
  ALTER DATABASE greenlang SET track_io_timing = on;
  ALTER DATABASE greenlang SET shared_preload_libraries = 'pg_stat_statements';
"
```

**Step 3: Implement Pagination**

```bash
# Enforce pagination limits
kubectl set env deployment/api-gateway -n greenlang-gl008 \
  MAX_PAGE_SIZE=100 \
  DEFAULT_PAGE_SIZE=20

# Update API clients to use pagination
echo "Notify API consumers to update their implementations"
```

**Step 4: Scale API Gateway**

```bash
# Scale API gateway horizontally
kubectl scale deployment/api-gateway -n greenlang-gl008 --replicas=6

# Enable autoscaling
kubectl autoscale deployment api-gateway -n greenlang-gl008 \
  --min=4 --max=12 --cpu-percent=70
```

**Verification:**

```bash
# Monitor API latency improvements
watch -n 10 'curl -s https://api.greenlang.io/v1/steam-trap/metrics/latency | jq ".p95_latency_ms"'

# Load test to verify
./scripts/load-test-api.sh --duration=300 --rps=100
```

---

### Issue 8: Memory Leaks

**Symptoms:**
- Pod memory usage continuously increasing
- Pods being OOMKilled and restarted
- Degrading performance over time

**Diagnostic Steps:**

```bash
# Check memory usage trends
kubectl top pods -n greenlang-gl008 --sort-by=memory

# Review OOMKilled events
kubectl get events -n greenlang-gl008 --field-selector reason=OOMKilled --sort-by='.lastTimestamp'

# Check pod memory limits
kubectl describe pods -n greenlang-gl008 -l app=steam-trap-inspector | grep -A 5 "Limits:"

# Generate heap dump (Python)
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  python -c "import objgraph; objgraph.show_most_common_types(limit=20)"
```

**Resolution Steps:**

**Step 1: Increase Memory Limits (Temporary)**

```bash
# Increase memory limits
kubectl patch deployment steam-trap-inspector -n greenlang-gl008 --type='json' -p='[
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/limits/memory",
    "value": "4Gi"
  },
  {
    "op": "replace",
    "path": "/spec/template/spec/containers/0/resources/requests/memory",
    "value": "2Gi"
  }
]'
```

**Step 2: Enable Memory Profiling**

```bash
# Enable memory profiling
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ENABLE_MEMORY_PROFILING=true \
  PROFILE_OUTPUT_PATH=/tmp/memory-profile.prof

# Collect profile after 1 hour
sleep 3600
kubectl cp greenlang-gl008/steam-trap-inspector-xxxx:/tmp/memory-profile.prof ./memory-profile.prof

# Analyze profile
python -m memory_profiler memory-profile.prof
```

**Step 3: Fix Common Leak Sources**

```bash
# Clear ML model cache periodically
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  CLEAR_MODEL_CACHE_INTERVAL=3600

# Limit in-memory cache size
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  MAX_CACHE_SIZE_MB=512

# Enable garbage collection tuning
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  PYTHONHASHSEED=random \
  MALLOC_TRIM_THRESHOLD_=100000
```

**Step 4: Implement Pod Recycling**

```bash
# Restart pods proactively every 12 hours
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  MAX_POD_LIFETIME_HOURS=12

# Or use a CronJob for scheduled restarts
kubectl create -f - <<EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: restart-steam-trap-inspector
  namespace: greenlang-gl008
spec:
  schedule: "0 */12 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: pod-restarter
          containers:
          - name: kubectl
            image: bitnami/kubectl:latest
            command:
            - /bin/sh
            - -c
            - kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
          restartPolicy: OnFailure
EOF
```

---

## Data Quality Issues

### Issue 9: Inconsistent Energy Calculations

**Symptoms:**
- Energy loss values fluctuating wildly
- Negative energy values
- Energy calculations don't match manual calculations

**Diagnostic Steps:**

```bash
# Check for anomalous energy calculations
psql $DB_URL -c "
  SELECT
    inspection_id,
    trap_id,
    energy_loss_kwh,
    steam_pressure_psi,
    steam_temp_f,
    leak_rate_lb_hr,
    calculated_at
  FROM energy_calculations
  WHERE calculated_at > NOW() - INTERVAL '24 hours'
    AND (
      energy_loss_kwh < 0
      OR energy_loss_kwh > 10000
      OR energy_loss_kwh IS NULL
    )
  ORDER BY ABS(energy_loss_kwh) DESC
  LIMIT 20;
"

# Validate steam property data
psql $DB_URL -c "
  SELECT
    pressure_psi,
    temp_fahrenheit,
    enthalpy_btu_lb,
    entropy_btu_lb_f
  FROM steam_properties
  WHERE enthalpy_btu_lb IS NULL
    OR enthalpy_btu_lb < 0
    OR temp_fahrenheit < 32;
"

# Test calculation algorithm
python scripts/test_energy_calculations.py --verbose
```

**Resolution Steps:**

**Step 1: Validate Input Data**

```bash
# Check for missing or invalid steam parameters
psql $DB_URL -c "
  UPDATE trap_inspections
  SET needs_recalculation = true
  WHERE (
    steam_pressure_psi IS NULL
    OR steam_pressure_psi <= 0
    OR steam_temp_f IS NULL
    OR steam_temp_f < 212  -- Below boiling point of water
  )
  AND detected_at > NOW() - INTERVAL '7 days';
"

# Fill missing steam parameters from site defaults
psql $DB_URL -c "
  UPDATE trap_inspections ti
  SET
    steam_pressure_psi = COALESCE(ti.steam_pressure_psi, s.default_steam_pressure_psi),
    steam_temp_f = COALESCE(ti.steam_temp_f, s.default_steam_temp_f)
  FROM traps t
  JOIN sites s ON t.site_id = s.id
  WHERE ti.trap_id = t.id
    AND ti.needs_recalculation = true;
"
```

**Step 2: Recalculate Energy Values**

```bash
# Trigger recalculation job
kubectl create job recalculate-energy-$(date +%s) \
  --from=cronjob/energy-recalculation \
  -n greenlang-gl008 \
  -- --date-range="last_7_days" --force

# Monitor progress
kubectl logs -n greenlang-gl008 -l job-name=recalculate-energy-* --follow
```

**Step 3: Update Steam Property Tables**

```bash
# Restore steam property tables from NIST reference
psql $DB_URL < reference_data/steam_properties_nist_iapws.sql

# Verify restoration
./scripts/validate-steam-tables.sh --reference=NIST_IAPWS
```

---

### Issue 10: Missing Inspection Data

**Symptoms:**
- Inspection records missing required fields
- NULL values in critical columns
- Data integrity constraint violations

**Diagnostic Steps:**

```bash
# Check for incomplete inspection records
psql $DB_URL -c "
  SELECT
    'acoustic_features' as missing_field,
    COUNT(*) as count
  FROM trap_inspections
  WHERE acoustic_features IS NULL
    AND detected_at > NOW() - INTERVAL '7 days'
  UNION ALL
  SELECT
    'thermal_features',
    COUNT(*)
  FROM trap_inspections
  WHERE thermal_features IS NULL
    AND detected_at > NOW() - INTERVAL '7 days'
  UNION ALL
  SELECT
    'confidence_score',
    COUNT(*)
  FROM trap_inspections
  WHERE confidence_score IS NULL
    AND detected_at > NOW() - INTERVAL '7 days';
"

# Check for orphaned records
psql $DB_URL -c "
  SELECT
    COUNT(*) as orphaned_inspections
  FROM trap_inspections ti
  LEFT JOIN traps t ON ti.trap_id = t.id
  WHERE t.id IS NULL;
"
```

**Resolution Steps:**

```bash
# Add data validation constraints
psql $DB_URL -c "
  -- Ensure critical fields are populated
  ALTER TABLE trap_inspections
  ADD CONSTRAINT chk_acoustic_features
  CHECK (acoustic_features IS NOT NULL OR status = 'SENSOR_OFFLINE');

  ALTER TABLE trap_inspections
  ADD CONSTRAINT chk_confidence_score
  CHECK (confidence_score BETWEEN 0 AND 1 OR status != 'COMPLETED');
"

# Backfill missing data where possible
./scripts/backfill-inspection-data.sh --date-range="last_7_days"

# Archive incomplete records
psql $DB_URL -c "
  INSERT INTO trap_inspections_archive
  SELECT * FROM trap_inspections
  WHERE (acoustic_features IS NULL OR thermal_features IS NULL)
    AND detected_at < NOW() - INTERVAL '30 days';

  DELETE FROM trap_inspections
  WHERE id IN (SELECT id FROM trap_inspections_archive);
"
```

---

## Integration Issues

### Issue 11: ERP Connector Failures

**Symptoms:**
- Failed to sync trap inventory from ERP
- Error: "ERP connection timeout"
- Trap data out of sync with ERP system

**Diagnostic Steps:**

```bash
# Check ERP connector status
curl https://api.greenlang.io/v1/steam-trap/integrations/erp/status | jq '.'

# Review connector logs
kubectl logs -n greenlang-gl008 -l component=erp-connector --tail=200

# Check last successful sync
psql $DB_URL -c "
  SELECT
    erp_system,
    last_sync_at,
    NOW() - last_sync_at as time_since_sync,
    last_sync_status,
    records_synced,
    error_message
  FROM erp_sync_status
  ORDER BY last_sync_at DESC;
"

# Test ERP connectivity
./scripts/test-erp-connection.sh --system=SAP --site-id=$SITE_ID
```

**Resolution Steps:**

**Step 1: Verify Credentials**

```bash
# Test ERP credentials
curl -X POST https://api.greenlang.io/v1/steam-trap/integrations/erp/test-connection \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "erp_system": "SAP",
    "site_id": "'$SITE_ID'"
  }'

# Rotate credentials if expired
./scripts/rotate-erp-credentials.sh --system=SAP --site-id=$SITE_ID
```

**Step 2: Increase Timeout Values**

```bash
# Increase ERP connection timeout
kubectl set env deployment/erp-connector -n greenlang-gl008 \
  ERP_CONNECTION_TIMEOUT=120s \
  ERP_READ_TIMEOUT=180s \
  RETRY_ATTEMPTS=5 \
  RETRY_BACKOFF=exponential

kubectl rollout restart deployment/erp-connector -n greenlang-gl008
```

**Step 3: Manual Sync**

```bash
# Trigger manual sync
curl -X POST https://api.greenlang.io/v1/steam-trap/integrations/erp/sync \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "erp_system": "SAP",
    "site_id": "'$SITE_ID'",
    "sync_type": "full",
    "force": true
  }'

# Monitor sync progress
watch -n 10 'curl -s https://api.greenlang.io/v1/steam-trap/integrations/erp/sync-status | jq ".status"'
```

---

### Issue 12: MQTT Message Delivery Failures

**Symptoms:**
- Sensor data not reaching backend
- MQTT broker connection errors
- Message queue backlog

**Diagnostic Steps:**

```bash
# Check MQTT broker status
kubectl get pods -n greenlang-gl008 -l app=mqtt-broker

# Check message queue depth
curl https://api.greenlang.io/v1/steam-trap/mqtt/stats | jq '.queue_depth'

# Review MQTT logs
kubectl logs -n greenlang-gl008 -l app=mqtt-broker --tail=200 | grep -E "(ERROR|WARN|disconnected)"

# Test MQTT connectivity
mosquitto_sub -h mqtt.greenlang-gl008.svc.cluster.local \
  -t "sensors/+/data" \
  -u $MQTT_USER \
  -P $MQTT_PASSWORD \
  -C 10
```

**Resolution Steps:**

```bash
# Restart MQTT broker
kubectl rollout restart deployment/mqtt-broker -n greenlang-gl008

# Increase MQTT message retention
kubectl set env deployment/mqtt-broker -n greenlang-gl008 \
  MAX_QUEUED_MESSAGES=10000 \
  MESSAGE_RETENTION_SECONDS=3600

# Scale MQTT broker
kubectl scale deployment/mqtt-broker -n greenlang-gl008 --replicas=3
```

---

## ML Model Issues

### Issue 13: Model Prediction Errors

**Symptoms:**
- ML model returning errors during inference
- Error: "Feature shape mismatch"
- Predictions returning NaN or Infinity

**Diagnostic Steps:**

```bash
# Check model version and health
curl https://api.greenlang.io/v1/steam-trap/ml/model-info | jq '.'

# Review prediction errors
kubectl logs -n greenlang-gl008 -l component=ml-service --tail=500 | \
  grep -E "(prediction|inference)" | \
  grep ERROR

# Test model with sample data
python scripts/test_ml_model.py \
  --model-path=/models/steam_trap_v2.4.2.pkl \
  --test-data=test_samples/sample_1.json
```

**Resolution Steps:**

```bash
# Rollback to previous model version
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ML_MODEL_VERSION=v2.4.1

kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008

# Verify model file integrity
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  md5sum /models/steam_trap_v2.4.2.pkl

# Re-download model from artifact store
./scripts/download-ml-model.sh --version=v2.4.2 --verify-checksum
```

---

### Issue 14: RUL Prediction Drift

**Symptoms:**
- Remaining Useful Life predictions inaccurate
- RUL values don't correlate with actual failures
- Drift from expected failure patterns

**Diagnostic Steps:**

```bash
# Compare predicted vs actual RUL
psql $DB_URL -c "
  SELECT
    DATE(ti.detected_at) as date,
    AVG(ti.predicted_rul_days) as avg_predicted_rul,
    AVG(EXTRACT(EPOCH FROM (t.actual_failure_date - ti.detected_at))/86400) as avg_actual_rul,
    AVG(ti.predicted_rul_days) - AVG(EXTRACT(EPOCH FROM (t.actual_failure_date - ti.detected_at))/86400) as rul_error_days
  FROM trap_inspections ti
  JOIN traps t ON ti.trap_id = t.id
  WHERE t.actual_failure_date IS NOT NULL
    AND ti.detected_at > NOW() - INTERVAL '90 days'
  GROUP BY DATE(ti.detected_at)
  ORDER BY date DESC
  LIMIT 30;
"

# Analyze RUL prediction accuracy by trap type
python scripts/analyze_rul_accuracy.py --lookback-days=90
```

**Resolution Steps:**

```bash
# Retrain RUL model with recent failure data
python scripts/retrain_rul_model.py \
  --training-data=last_180_days \
  --include-recent-failures \
  --validation-split=0.2

# Update RUL model
kubectl cp models/rul_model_v2.1.0.pkl \
  greenlang-gl008/steam-trap-inspector-xxxx:/models/rul_model.pkl

# Restart to load new model
kubectl rollout restart deployment/steam-trap-inspector -n greenlang-gl008
```

---

## Database Issues

### Issue 15: Slow Queries

**Symptoms:**
- Database queries taking >5 seconds
- Application timeouts
- High database CPU usage

**Diagnostic Steps:**

```bash
# Identify slow queries
psql $DB_URL -c "
  SELECT
    substring(query, 1, 100) as query_snippet,
    calls,
    mean_exec_time,
    max_exec_time,
    total_exec_time,
    ROUND(100.0 * total_exec_time / SUM(total_exec_time) OVER (), 2) as pct_total_time
  FROM pg_stat_statements
  ORDER BY mean_exec_time DESC
  LIMIT 20;
"

# Check for missing indexes
psql $DB_URL -c "
  SELECT
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
  FROM pg_stats
  WHERE schemaname = 'public'
    AND tablename IN ('trap_inspections', 'sensor_readings', 'traps')
    AND n_distinct > 100
    AND correlation < 0.1;
"

# Check table bloat
psql $DB_URL -c "
  SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_live_tup,
    n_dead_tup,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 2) as dead_tuple_pct
  FROM pg_stat_user_tables
  WHERE n_dead_tup > 10000
  ORDER BY n_dead_tup DESC;
"
```

**Resolution Steps:**

```bash
# Create missing indexes
psql $DB_URL -c "
  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_inspections_site_date
  ON trap_inspections(site_id, DATE(detected_at));

  CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sensor_readings_timestamp
  ON sensor_readings(captured_at DESC) WHERE status = 'VALID';
"

# Vacuum bloated tables
psql $DB_URL -c "
  VACUUM ANALYZE trap_inspections;
  VACUUM ANALYZE sensor_readings;
"

# Update table statistics
psql $DB_URL -c "
  ANALYZE trap_inspections;
  ANALYZE sensor_readings;
  ANALYZE traps;
"
```

---

## Diagnostic Tools

### Comprehensive Health Check

```bash
#!/bin/bash
# File: scripts/health-check.sh

echo "=== GL-008 Steam Trap Inspector Health Check ==="
echo ""

echo "1. Kubernetes Pods Status:"
kubectl get pods -n greenlang-gl008
echo ""

echo "2. API Health:"
curl -s https://api.greenlang.io/v1/steam-trap/health | jq '.'
echo ""

echo "3. Database Connection:"
psql $DB_URL -c "SELECT COUNT(*) as trap_count FROM traps;" -t
echo ""

echo "4. Sensor Status:"
curl -s https://api.greenlang.io/v1/steam-trap/sensors/status | jq '{online: .online_sensors_count, offline: .offline_sensors_count}'
echo ""

echo "5. Recent Inspections:"
psql $DB_URL -c "SELECT COUNT(*) FROM trap_inspections WHERE detected_at > NOW() - INTERVAL '1 hour';" -t
echo ""

echo "6. Error Rate (last hour):"
kubectl logs -n greenlang-gl008 -l app=steam-trap-inspector --since=1h | grep ERROR | wc -l
echo ""

echo "7. Resource Usage:"
kubectl top pods -n greenlang-gl008 -l app=steam-trap-inspector
echo ""

echo "Health check complete."
```

### Log Analyzer

```bash
#!/bin/bash
# File: scripts/analyze-logs.sh

NAMESPACE="greenlang-gl008"
HOURS=${1:-1}

echo "Analyzing logs from last $HOURS hours..."

echo ""
echo "=== Error Summary ==="
kubectl logs -n $NAMESPACE -l app=steam-trap-inspector --since=${HOURS}h | \
  grep ERROR | \
  awk '{print $NF}' | \
  sort | \
  uniq -c | \
  sort -rn

echo ""
echo "=== Warning Summary ==="
kubectl logs -n $NAMESPACE -l app=steam-trap-inspector --since=${HOURS}h | \
  grep WARN | \
  awk '{print $NF}' | \
  sort | \
  uniq -c | \
  sort -rn

echo ""
echo "=== Top Slow Operations ==="
kubectl logs -n $NAMESPACE -l app=steam-trap-inspector --since=${HOURS}h | \
  grep "duration_ms" | \
  jq -r '[.operation, .duration_ms] | @csv' | \
  sort -t, -k2 -rn | \
  head -10
```

---

## Advanced Troubleshooting

### Memory Profiling

```bash
# Enable memory profiling
kubectl set env deployment/steam-trap-inspector -n greenlang-gl008 \
  ENABLE_PROFILING=true \
  PROFILE_TYPE=memory

# Collect heap dump
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  python -c "
import gc
import objgraph
gc.collect()
objgraph.show_most_common_types(limit=50)
"

# Analyze memory growth
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  python -m memory_profiler app.py
```

### Network Diagnostics

```bash
# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  nslookup api.greenlang.io

# Test connectivity to database
kubectl run -it --rm debug --image=postgres:15 --restart=Never -- \
  psql $DB_URL -c "SELECT 1;"

# Trace network path
kubectl run -it --rm debug --image=busybox --restart=Never -- \
  traceroute api.greenlang.io
```

### Performance Profiling

```bash
# CPU profiling
kubectl exec -n greenlang-gl008 deployment/steam-trap-inspector -- \
  python -m cProfile -o profile.stats app.py

# Analyze profile
kubectl cp greenlang-gl008/steam-trap-inspector-xxxx:/profile.stats ./profile.stats
python -m pstats profile.stats
```

---

**Document Version:** 1.0.0
**Last Reviewed:** 2025-11-26
**Next Review:** 2026-02-26
**Maintained By:** Platform Operations Team
