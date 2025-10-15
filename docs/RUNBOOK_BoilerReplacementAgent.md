# BoilerReplacementAgent_AI - Operations Runbook

## Health Check
`GET /api/v1/agents/industrial/boiler_replacement/health`

Expected: `{"status": "healthy", "version": "1.0.0"}`

## Common Issues

### Issue: High Latency (>3500ms)
**Symptoms**: Slow response times
**Diagnosis**: Check tool call count in logs
**Resolution**: Optimize tool calls, increase timeout
**Escalation**: oncall-platform-team@greenlang.com

### Issue: Budget Exceeded
**Symptoms**: BudgetExceeded errors
**Diagnosis**: Check cost tracking in logs
**Resolution**: Increase budget_usd parameter (default: $0.15)
**Escalation**: oncall-ai-team@greenlang.com

### Issue: Invalid Boiler Type
**Symptoms**: ValueError with unknown boiler type
**Diagnosis**: Check input validation in logs
**Resolution**: Verify boiler_type is in allowed list (firetube, watertube, condensing, electric_resistance, electrode)
**Escalation**: oncall-support-team@greenlang.com

### Issue: Efficiency Calculation Errors
**Symptoms**: Incorrect efficiency values (outside 40-99% range)
**Diagnosis**: Check age_years and stack_temperature_c values
**Resolution**: Verify boiler age is realistic (<60 years), stack temp is reasonable (30-400°C)
**Escalation**: oncall-engineering-team@greenlang.com

### Issue: IRA 2022 Incentive Calculation Errors
**Symptoms**: Federal ITC not calculated correctly
**Diagnosis**: Check federal_itc_eligible flag
**Resolution**: Verify system qualifies for 30% Federal ITC (solar thermal or heat pumps only)
**Escalation**: oncall-financial-team@greenlang.com

### Issue: Heat Pump COP Unrealistic Values
**Symptoms**: COP outside 2-6 range
**Diagnosis**: Check temperature lift (sink - source)
**Resolution**: Verify sink and source temperatures are correct, ensure temperature lift is not too extreme
**Escalation**: oncall-engineering-team@greenlang.com

## Monitoring
- Latency: alert if p95 > 3000ms
- Error rate: alert if > 1%
- Cost: alert if avg > $0.12/query
- Tool call count: alert if avg > 12 per query
- Success rate: alert if < 98%

## Performance Baselines
- Average latency: 1800-2500ms
- Average cost: $0.08-0.12 per query
- Tool calls per query: 8-10
- Success rate: 98%+

## Deployment Information
- Package: boiler_replacement
- Version: 1.0.0
- Dependencies: pydantic>=2.0, numpy>=1.24, scipy>=1.11
- Resource Requirements: 768MB RAM, 2 CPU cores

## Tool-Specific Troubleshooting

### Tool 1: calculate_boiler_efficiency
**Common Issues**:
- Stack temperature too high (>400°C) → Check sensor calibration
- Age degradation excessive → Verify boiler age, check for maintenance records
**Fix**: Validate input ranges, ensure ASME PTC 4.1 formulas are applied correctly

### Tool 2: calculate_annual_fuel_consumption
**Common Issues**:
- Load factor outside 0-1 range → Validate input
- Fuel consumption too high → Check efficiency calculation first
**Fix**: Ensure load factor and operating hours are realistic

### Tool 3: calculate_solar_thermal_sizing
**Common Issues**:
- Solar fraction unrealistic → Check process temperature and latitude
- Collector area too large → Verify annual heat demand is correct
**Fix**: Use modified f-Chart method, ensure temperature constraints

### Tool 4: calculate_heat_pump_cop
**Common Issues**:
- COP too low → Check temperature lift, verify source temp
- COP too high (>6.0) → Temperature lift may be too small
**Fix**: Apply Carnot efficiency corrections, cap at 6.0

### Tool 5: calculate_hybrid_system_performance
**Common Issues**:
- Energy balance doesn't add up → Check solar fraction input
- System efficiency unrealistic → Verify COP and backup efficiency
**Fix**: Ensure energy balance: solar + heat pump + backup = total demand

### Tool 6: estimate_payback_period
**Common Issues**:
- Payback negative or >50 years → Check capital cost and savings
- IRR calculation fails → Verify discount rate and analysis period
**Fix**: Ensure annual savings > 0, validate financial inputs

### Tool 7: calculate_retrofit_integration_requirements
**Common Issues**:
- Retrofit cost unrealistic → Check capacity and building age
- Space requirements too large → Verify technology type
**Fix**: Use rule-based cost models, adjust for building age

### Tool 8: compare_replacement_technologies
**Common Issues**:
- Recommended technology doesn't match criteria → Check weights
- Technology scores identical → Verify scoring matrix
**Fix**: Apply weighted multi-criteria analysis correctly

## API Endpoints

### Execute Analysis
```
POST /api/v1/agents/industrial/boiler_replacement/execute
```

Request body:
```json
{
  "boiler_type": "firetube",
  "fuel_type": "natural_gas",
  "rated_capacity_kw": 1500,
  "age_years": 20,
  "stack_temperature_c": 250,
  "average_load_factor": 0.65,
  "annual_operating_hours": 6000,
  "latitude": 35.0
}
```

Expected response time: <3500ms
Expected cost: <$0.15

### Health Check
```
GET /api/v1/agents/industrial/boiler_replacement/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "agent_id": "industrial/boiler_replacement_agent",
  "metrics": {
    "tool_call_count": 42,
    "ai_call_count": 5,
    "total_cost_usd": 0.45
  }
}
```

## Escalation Procedures

### Severity 1 (Critical)
- Agent completely unavailable
- Error rate >10%
**Response Time**: 15 minutes
**Contacts**: oncall-platform-team@greenlang.com

### Severity 2 (High)
- Performance degradation (latency >5000ms)
- Budget exceeded frequently
**Response Time**: 1 hour
**Contacts**: oncall-ai-team@greenlang.com

### Severity 3 (Medium)
- Calculation errors affecting <5% of queries
- Non-critical tool failures
**Response Time**: 4 hours
**Contacts**: oncall-engineering-team@greenlang.com

### Severity 4 (Low)
- Documentation issues
- Feature requests
**Response Time**: Next business day
**Contacts**: oncall-support-team@greenlang.com

## Support Contacts
- Platform Team: oncall-platform-team@greenlang.com
- AI Team: oncall-ai-team@greenlang.com
- Engineering Team: oncall-engineering-team@greenlang.com
- Financial Team: oncall-financial-team@greenlang.com
- Support Team: oncall-support-team@greenlang.com

## Maintenance Windows
- Regular maintenance: Every Sunday 2-4 AM UTC
- Emergency patches: Communicated via Slack #greenlang-ops
- Agent version updates: Deployed during regular maintenance

## Performance Tuning

### If Latency is High
1. Check tool call count - should be 8-10 per query
2. Reduce analysis_period_years for financial calculations
3. Consider caching frequently accessed boiler data
4. Increase timeout to 4000ms if needed

### If Cost is High
1. Reduce budget_usd to $0.12 (from default $0.15)
2. Check if AI is making redundant tool calls
3. Monitor token usage in ChatSession
4. Consider rate limiting expensive queries

### If Accuracy is Low
1. Verify all physics formulas (ASME PTC 4.1, Carnot efficiency)
2. Check emission factors and fuel costs are up to date
3. Validate IRA 2022 Federal ITC percentage (30%)
4. Review determinism tests (temperature=0, seed=42)

## Logging and Debugging

### Key Log Messages
```
INFO: Tool call: calculate_boiler_efficiency (args: boiler_type=firetube, age_years=20)
INFO: Efficiency calculated: 68.5%
INFO: Tool call: estimate_payback_period (args: capital_cost_usd=500000)
INFO: Payback period: 2.3 years with 30% Federal ITC
```

### Debug Mode
Set environment variable: `GREENLANG_DEBUG=1`
Enables verbose logging for all tool calls and AI interactions

### Performance Profiling
```python
agent = BoilerReplacementAgent_AI(budget_usd=1.0)
summary = agent.get_performance_summary()
print(summary)  # Shows AI calls, tool calls, cost breakdown
```
