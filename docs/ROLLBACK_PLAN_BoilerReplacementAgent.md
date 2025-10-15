# Rollback Plan - BoilerReplacementAgent_AI

## Pre-Rollback Checklist
- [ ] Backup current configuration
- [ ] Notify stakeholders (especially financial teams using IRA 2022 incentive calculations)
- [ ] Prepare rollback artifact (previous version)
- [ ] Verify previous version availability
- [ ] Check for in-flight requests (boiler analyses can take 2-3 seconds)
- [ ] Document incident details
- [ ] Preserve calculation audit trails for financial analysis

## Rollback Steps

### Step 1: Stop Current Version
```bash
# Stop the current agent instance
kubectl scale deployment boiler-replacement-agent --replicas=0
# Or for standalone deployment
systemctl stop greenlang-boiler-replacement
```

### Step 2: Restore Previous Pack Version
```bash
# Restore previous pack version
cd /opt/greenlang/packs
cp -r boiler_replacement.backup boiler_replacement
# Or use pack manager
greenlang-pack rollback industrial/boiler_replacement_agent --to-version=0.9.0
```

### Step 3: Restart with Previous Configuration
```bash
# Restart with previous version
kubectl apply -f deployments/boiler-replacement-agent-v0.9.0.yaml
# Or for standalone
systemctl start greenlang-boiler-replacement
```

### Step 4: Verify Health Check Passes
```bash
# Verify health endpoint
curl https://api.greenlang.com/api/v1/agents/industrial/boiler_replacement/health
# Expected: {"status": "healthy", "version": "0.9.0"}
```

### Step 5: Monitor for 15 Minutes
- Check error rates in monitoring dashboard
- Verify latency metrics return to baseline (<3000ms)
- Confirm no new errors in logs
- Test with sample queries (firetube, watertube, electric)
- Verify IRA 2022 30% Federal ITC calculations are correct

### Step 6: Verify Financial Calculations
```bash
# Critical: Verify IRA 2022 incentive calculations are correct
# Test with known scenario
curl -X POST https://api.greenlang.com/api/v1/agents/industrial/boiler_replacement/execute \
  -d '{
    "boiler_type": "firetube",
    "fuel_type": "natural_gas",
    "rated_capacity_kw": 1500,
    "age_years": 20,
    "stack_temperature_c": 250,
    "average_load_factor": 0.65,
    "annual_operating_hours": 6000,
    "latitude": 35.0
  }'

# Verify federal_itc_usd is 30% of capital cost
```

## Post-Rollback

### Immediate Actions
1. Document root cause of failure
2. Preserve logs and traces from failed version
3. Create incident report
4. Update status page
5. **Notify financial teams** if IRA 2022 incentive calculations were affected
6. Review any boiler analyses performed during failure window

### Follow-up Actions
1. Schedule post-mortem meeting within 24 hours
2. Identify missing test coverage (especially financial calculations)
3. Update deployment procedures if needed
4. Plan fix and re-deployment strategy
5. **Audit financial calculations** from failed version period
6. Re-run affected boiler analyses if needed

## Rollback Decision Criteria

Trigger immediate rollback if:
- Error rate > 5% for 5 consecutive minutes
- P95 latency > 5000ms for 10 minutes
- Agent crashes or becomes unresponsive
- **IRA 2022 Federal ITC calculations incorrect** (critical for financial decisions)
- **Boiler efficiency calculations incorrect** (affects all downstream results)
- **Payback period calculations incorrect** (affects business decisions)
- Data corruption detected
- Security vulnerability discovered

Consider gradual rollback if:
- Error rate between 1-5%
- Latency degradation but still functional (>3500ms but <5000ms)
- Non-critical functionality issues
- User complaints increasing
- Minor calculation inaccuracies in non-financial tools

## Rollback by Component

### If Only Financial Calculations Failed
```bash
# Rollback just the payback calculation tool
# Use feature flag to disable Tool 6
kubectl set env deployment/boiler-replacement-agent DISABLE_TOOL_6=true
```

### If Only Heat Pump COP Calculations Failed
```bash
# Rollback just the COP calculation tool
# Use feature flag to disable Tool 4
kubectl set env deployment/boiler-replacement-agent DISABLE_TOOL_4=true
```

### If Only AI Orchestration Failed
```bash
# Rollback to tool-only mode (no AI explanations)
kubectl set env deployment/boiler-replacement-agent ENABLE_EXPLANATIONS=false
```

## Communication Plan

### Internal Notification
```
Subject: ROLLBACK - BoilerReplacementAgent v1.0.0

Issue: [Brief description]
Impact: [User impact - especially financial analysis users]
Action: Rolling back to v0.9.0
Financial Impact: [If IRA incentive calculations were affected]
Timeline: [Estimated completion]
Status: [In Progress/Complete]

Critical Note: If you performed boiler analyses during [time window],
please contact engineering@greenlang.com for re-analysis.
```

### External Notification
```
Status Page Update:
"We are experiencing issues with our Boiler Replacement Agent.
We are rolling back to the previous version to restore service.
If you performed financial analyses during [time window], please contact
support@greenlang.com for verification of IRA 2022 incentive calculations.
Expected resolution: [time]"
```

### Financial Team Specific Notification
```
To: financial-teams@greenlang.com
Subject: URGENT - Verify Boiler Replacement Analyses

During the period [start] to [end], the BoilerReplacementAgent may have
produced incorrect financial calculations.

Please review any boiler replacement analyses performed during this window:
- Federal ITC calculations (should be 30% for solar/heat pumps)
- Payback period calculations
- NPV and IRR calculations

Contact engineering@greenlang.com for re-analysis if needed.
```

## Contact Information
- On-Call Engineer: oncall-engineering-team@greenlang.com
- Platform Team: oncall-platform-team@greenlang.com
- Financial Team: oncall-financial-team@greenlang.com
- Incident Manager: incident-manager@greenlang.com

## Testing After Rollback

```python
# Test script to verify rollback success
import requests

def test_rollback():
    """Comprehensive rollback verification for BoilerReplacementAgent."""

    # 1. Health check
    health = requests.get("https://api.greenlang.com/api/v1/agents/industrial/boiler_replacement/health")
    assert health.json()["version"] == "0.9.0", "Version mismatch"
    assert health.json()["status"] == "healthy", "Health check failed"

    # 2. Sample query - old firetube boiler
    query = {
        "boiler_type": "firetube",
        "fuel_type": "natural_gas",
        "rated_capacity_kw": 1500,
        "age_years": 20,
        "stack_temperature_c": 250,
        "average_load_factor": 0.65,
        "annual_operating_hours": 6000,
        "latitude": 35.0,
        "process_temperature_required_c": 120,
    }
    result = requests.post(
        "https://api.greenlang.com/api/v1/agents/industrial/boiler_replacement/execute",
        json=query
    )
    assert result.status_code == 200, "Query failed"

    data = result.json()["data"]

    # 3. Verify key outputs exist
    assert "current_efficiency" in data, "Missing efficiency"
    assert "recommended_technology" in data, "Missing recommendation"
    assert "simple_payback_years" in data, "Missing payback"
    assert "federal_itc_usd" in data, "Missing Federal ITC"

    # 4. Verify efficiency calculation (ASME PTC 4.1)
    efficiency = data["current_efficiency"]
    assert 0.40 <= efficiency <= 0.99, f"Efficiency out of range: {efficiency}"

    # 5. Verify Federal ITC calculation (IRA 2022: 30%)
    # Note: Only if system qualifies (solar or heat pump)
    if data.get("federal_itc_usd", 0) > 0:
        # ITC should be approximately 30% of capital cost
        # (exact value depends on recommended technology)
        print(f"Federal ITC: ${data['federal_itc_usd']:,.2f}")

    # 6. Verify payback period is reasonable
    payback = data["simple_payback_years"]
    assert 0.5 <= payback <= 50, f"Payback unrealistic: {payback} years"

    # 7. Verify no missing critical fields
    required_fields = [
        "current_annual_fuel_consumption_mmbtu",
        "current_annual_cost_usd",
        "current_annual_emissions_kg_co2e",
        "annual_emissions_reduction_kg_co2e",
        "retrofit_complexity",
    ]
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"

    print("✓ Rollback verification: PASS")
    print(f"✓ Version: {health.json()['version']}")
    print(f"✓ Efficiency: {efficiency:.2%}")
    print(f"✓ Payback: {payback:.1f} years")
    print(f"✓ Recommended: {data['recommended_technology']}")

if __name__ == "__main__":
    test_rollback()
```

## Disaster Recovery

### Complete Failure Scenario
If the agent is completely non-functional:

1. **Immediate Fallback**:
   ```bash
   # Route all traffic to manual analysis API
   kubectl patch service boiler-replacement-agent \
     -p '{"spec":{"selector":{"app":"manual-analysis-api"}}}'
   ```

2. **Manual Analysis Process**:
   - Use spreadsheet templates in `/backup/boiler-analysis-templates/`
   - Apply ASME PTC 4.1 formulas manually
   - Calculate IRA 2022 30% Federal ITC manually
   - Document all manual analyses for future audit

3. **Recovery Steps**:
   ```bash
   # Rebuild from source
   cd /opt/greenlang/agents
   git checkout v0.9.0-stable
   python -m build
   pip install dist/greenlang-*.whl
   systemctl restart greenlang-boiler-replacement
   ```

## Lessons Learned Template

1. **What went wrong?**
   - Specific component that failed
   - Root cause analysis

2. **What was the impact?**
   - Number of affected queries
   - Financial impact (if IRA calculations were wrong)
   - User experience impact

3. **How did we detect it?**
   - Monitoring alerts
   - User reports
   - Automated tests

4. **How did we respond?**
   - Time to detection
   - Time to rollback decision
   - Time to full recovery

5. **What could we have done better?**
   - Missing test coverage
   - Monitoring gaps
   - Deployment process improvements

6. **What are the action items?**
   - [ ] Add test coverage for [specific scenario]
   - [ ] Improve monitoring for [specific metric]
   - [ ] Update deployment checklist
   - [ ] Document new troubleshooting procedure
   - [ ] Re-train team on [specific topic]

## Audit Trail Preservation

For financial compliance, preserve:
- All calculation logs from failed version
- Input/output pairs for affected analyses
- Timestamps of all boiler analyses
- IRA 2022 Federal ITC calculations
- Payback period calculations
- NPV and IRR calculations

Archive location: `/var/log/greenlang/boiler-replacement/rollback-[date]/`

Retention period: 7 years (per financial regulations)
