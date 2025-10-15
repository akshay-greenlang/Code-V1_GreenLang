# Rollback Plan - IndustrialProcessHeatAgent_AI

## Pre-Rollback Checklist
- [ ] Backup current configuration
- [ ] Notify stakeholders
- [ ] Prepare rollback artifact (previous version)
- [ ] Verify previous version availability
- [ ] Check for in-flight requests
- [ ] Document incident details

## Rollback Steps

### Step 1: Stop Current Version
```bash
# Stop the current agent instance
kubectl scale deployment industrial-process-heat-agent --replicas=0
# Or for standalone deployment
systemctl stop greenlang-industrial-process-heat
```

### Step 2: Restore Previous Pack Version
```bash
# Restore previous pack version
cd /opt/greenlang/packs
cp -r industrial_process_heat.backup industrial_process_heat
# Or use pack manager
greenlang-pack rollback industrial/process_heat_agent --to-version=0.9.0
```

### Step 3: Restart with Previous Configuration
```bash
# Restart with previous version
kubectl apply -f deployments/industrial-process-heat-agent-v0.9.0.yaml
# Or for standalone
systemctl start greenlang-industrial-process-heat
```

### Step 4: Verify Health Check Passes
```bash
# Verify health endpoint
curl https://api.greenlang.com/api/v1/agents/industrial/process_heat/health
# Expected: {"status": "healthy", "version": "0.9.0"}
```

### Step 5: Monitor for 15 Minutes
- Check error rates in monitoring dashboard
- Verify latency metrics return to baseline
- Confirm no new errors in logs
- Test with sample queries

## Post-Rollback

### Immediate Actions
1. Document root cause of failure
2. Preserve logs and traces from failed version
3. Create incident report
4. Update status page

### Follow-up Actions
1. Schedule post-mortem meeting within 24 hours
2. Identify missing test coverage
3. Update deployment procedures if needed
4. Plan fix and re-deployment strategy

## Rollback Decision Criteria

Trigger immediate rollback if:
- Error rate > 5% for 5 consecutive minutes
- P95 latency > 5000ms for 10 minutes
- Agent crashes or becomes unresponsive
- Data corruption detected
- Security vulnerability discovered

Consider gradual rollback if:
- Error rate between 1-5%
- Latency degradation but still functional
- Non-critical functionality issues
- User complaints increasing

## Communication Plan

### Internal Notification
```
Subject: ROLLBACK - IndustrialProcessHeatAgent v1.0.0

Issue: [Brief description]
Impact: [User impact]
Action: Rolling back to v0.9.0
Timeline: [Estimated completion]
Status: [In Progress/Complete]
```

### External Notification
```
Status Page Update:
"We are experiencing issues with our Industrial Process Heat Agent.
We are rolling back to the previous version to restore service.
Expected resolution: [time]"
```

## Contact Information
- On-Call Engineer: oncall-engineering-team@greenlang.com
- Platform Team: oncall-platform-team@greenlang.com
- Incident Manager: incident-manager@greenlang.com

## Testing After Rollback
```python
# Test script to verify rollback success
import requests

def test_rollback():
    # Health check
    health = requests.get("https://api.greenlang.com/api/v1/agents/industrial/process_heat/health")
    assert health.json()["version"] == "0.9.0"

    # Sample query
    query = {
        "industry_type": "Food & Beverage",
        "process_type": "pasteurization",
        "production_rate": 1000,
        "temperature_requirement": 72,
        "current_fuel_type": "natural_gas",
        "latitude": 35.0
    }
    result = requests.post("https://api.greenlang.com/api/v1/agents/industrial/process_heat/execute", json=query)
    assert result.status_code == 200
    assert "solar_fraction" in result.json()["data"]

    print("Rollback verification: PASS")

if __name__ == "__main__":
    test_rollback()
```

## Lessons Learned Template
1. What went wrong?
2. What was the impact?
3. How did we detect it?
4. How did we respond?
5. What could we have done better?
6. What are the action items?
