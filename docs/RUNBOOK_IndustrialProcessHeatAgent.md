# IndustrialProcessHeatAgent_AI - Operations Runbook

## Health Check
`GET /api/v1/agents/industrial/process_heat/health`

Expected: `{"status": "healthy", "version": "1.0.0"}`

## Common Issues

### Issue: High Latency (>3000ms)
**Symptoms**: Slow response times
**Diagnosis**: Check tool call count in logs
**Resolution**: Optimize tool calls, increase timeout
**Escalation**: oncall-platform-team@greenlang.com

### Issue: Budget Exceeded
**Symptoms**: BudgetExceeded errors
**Diagnosis**: Check cost tracking in logs
**Resolution**: Increase budget_usd parameter
**Escalation**: oncall-ai-team@greenlang.com

### Issue: Invalid Process Type
**Symptoms**: ValueError with unknown process type
**Diagnosis**: Check input validation in logs
**Resolution**: Verify process_type is in allowed list (drying, pasteurization, sterilization, evaporation, distillation, washing, preheating, curing, metal_treating)
**Escalation**: oncall-support-team@greenlang.com

### Issue: Solar Fraction Calculation Errors
**Symptoms**: Incorrect solar fraction values
**Diagnosis**: Check latitude and irradiance values
**Resolution**: Verify location data is correct, check for missing annual_irradiance parameter
**Escalation**: oncall-engineering-team@greenlang.com

## Monitoring
- Latency: alert if p95 > 2500ms
- Error rate: alert if > 1%
- Cost: alert if avg > $0.08/query
- Tool call count: alert if avg > 10 per query
- Success rate: alert if < 98%

## Performance Baselines
- Average latency: 1200-1800ms
- Average cost: $0.03-0.05 per query
- Tool calls per query: 6-8
- Success rate: 99%+

## Deployment Information
- Package: industrial_process_heat
- Version: 1.0.0
- Dependencies: pydantic>=2.0, numpy>=1.24
- Resource Requirements: 512MB RAM, 1 CPU core

## Support Contacts
- Platform Team: oncall-platform-team@greenlang.com
- AI Team: oncall-ai-team@greenlang.com
- Engineering Team: oncall-engineering-team@greenlang.com
- Support Team: oncall-support-team@greenlang.com
