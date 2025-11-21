# GL-001 ProcessHeatOrchestrator - Operations Runbooks

Comprehensive operational runbooks for GL-001 ProcessHeatOrchestrator production operations across multi-plant industrial facilities.

## Overview

This directory contains production operations runbooks covering troubleshooting, incident response, rollback procedures, and scaling operations for the GL-001 ProcessHeatOrchestrator - the master coordinator managing process heat operations across entire industrial enterprises.

**Critical Context**: GL-001 is a **master orchestrator** coordinating 99 specialized sub-agents (GL-002 through GL-100) managing industrial process heat operations. It integrates with SCADA, ERP, and CMMS systems while ensuring safety, compliance, and optimal efficiency across multiple manufacturing plants.

## Runbooks Index

### 1. TROUBLESHOOTING.md

**Purpose**: Diagnose and resolve common production issues across multi-plant operations

**When to Use**:
- Master orchestrator not starting or pods crashing
- Sub-agent coordination failures (GL-002 through GL-100)
- Multi-plant SCADA/ERP integration failures
- Master database or message bus connectivity issues
- Heat distribution optimization failures
- Multi-agent task delegation errors

**Common Scenarios**:
- Configuration errors preventing master startup
- Missing environment variables for multi-plant setup
- Database connection pool exhaustion from agent load
- Message bus failures affecting agent coordination
- OOMKilled or CPU throttling under heavy orchestration
- Sub-agent timeouts and cascade failures
- Multi-plant heat balancing errors
- LP solver failures for heat distribution optimization

**Quick Links**:
- [Master Orchestrator Not Starting](TROUBLESHOOTING.md#master-orchestrator-not-starting)
- [Sub-Agent Coordination Failures](TROUBLESHOOTING.md#sub-agent-coordination-failures)
- [Multi-Plant Integration Issues](TROUBLESHOOTING.md#multi-plant-integration-issues)
- [Heat Distribution Optimization Failures](TROUBLESHOOTING.md#heat-distribution-optimization-failures)
- [Performance Degradation](TROUBLESHOOTING.md#performance-degradation)

---

### 2. INCIDENT_RESPONSE.md

**Purpose**: Emergency procedures for production incidents affecting process heat operations

**When to Use**:
- P0 critical incidents (total heat loss, multi-plant failure, safety hazard)
- P1 high severity incidents (single plant heat loss, sub-agent cascade failure)
- P2 medium severity incidents (performance degradation, optimization failures)
- Any production issue requiring escalation to Plant Safety Officer

**Severity Levels**:
- **P0 (Critical)**: Total facility heat loss, multi-plant cascade failure, safety hazard, immediate response
- **P1 (High)**: Single plant heat loss, SCADA outage, major sub-agent failure, 15-minute response
- **P2 (Medium)**: Heat distribution inefficiency, minor sub-agent degradation, 1-hour response
- **P3 (Low)**: Monitoring alerts, non-critical optimization issues, 4-hour response
- **P4 (Informational)**: Performance tuning opportunities, 1-day response

**Includes**:
- Severity definitions specific to process heat operations
- Response procedures by severity with safety protocols
- Escalation paths: On-Call → Plant Operations Manager → Plant Safety Officer → VP Operations
- Communication templates for multi-plant coordination
- Emergency heat loss procedures (restore critical heat within 30 minutes)
- Sub-agent cascade failure recovery
- Post-incident reviews with safety analysis

**Quick Links**:
- [P0 Response - Total Heat Loss](INCIDENT_RESPONSE.md#p0---critical-total-heat-loss)
- [P1 Response - Plant Heat Loss](INCIDENT_RESPONSE.md#p1---high-single-plant-heat-loss)
- [Sub-Agent Cascade Failure](INCIDENT_RESPONSE.md#sub-agent-cascade-failure-recovery)
- [Emergency Heat Restoration](INCIDENT_RESPONSE.md#emergency-heat-restoration-procedures)
- [Escalation Paths](INCIDENT_RESPONSE.md#escalation-paths)
- [Safety Protocols](INCIDENT_RESPONSE.md#safety-protocols-for-heat-operations)

---

### 3. ROLLBACK_PROCEDURE.md

**Purpose**: Safe version rollback procedures for master orchestrator and coordinated sub-agents

**When to Use**:
- Recent deployment causing multi-plant issues (<2 hours)
- Need to revert master orchestrator to previous stable version
- Sub-agent rollback synchronization required
- Database migration failures affecting multi-plant data
- Critical bugs discovered in heat optimization algorithms
- Heat distribution calculation errors

**Rollback Methods**:
- **Emergency Rollback** (<5 minutes): Immediate undo for critical heat loss scenarios
- **Coordinated Multi-Agent Rollback** (10 minutes): Synchronized rollback of master + sub-agents
- **Specific Revision Rollback** (15 minutes): Rollback to specific stable version
- **Blue-Green Rollback** (20 minutes): Zero-downtime rollback for master orchestrator
- **Partial Rollback** (8 minutes): Single plant or single sub-agent rollback
- **ConfigMap/Secret Rollback** (2 minutes): Configuration-only rollback across plants

**Includes**:
- When to rollback decision matrix for master orchestrator
- Pre-rollback checklist for multi-plant safety
- Step-by-step rollback procedures with sub-agent coordination
- Verification and validation steps for all plants
- Database migration rollback for multi-plant data
- Heat operation safety during rollback (maintain critical heat supply)
- Communication templates for multi-plant rollback coordination

**Quick Links**:
- [When to Rollback Decision Matrix](ROLLBACK_PROCEDURE.md#when-to-rollback-decision-matrix)
- [Emergency Rollback (<5 min)](ROLLBACK_PROCEDURE.md#emergency-rollback-5-minutes)
- [Multi-Agent Coordinated Rollback](ROLLBACK_PROCEDURE.md#coordinated-multi-agent-rollback)
- [Partial Rollback (Single Plant)](ROLLBACK_PROCEDURE.md#partial-rollback-single-plant)
- [Verification for All Plants](ROLLBACK_PROCEDURE.md#multi-plant-verification-procedures)

---

### 4. SCALING_GUIDE.md

**Purpose**: Scale master orchestrator and sub-agents to handle multi-plant load

**When to Use**:
- CPU/Memory usage >80% on master orchestrator
- Adding new plants (5+ plants)
- Sub-agent count increasing (50+ active agents)
- Response times increasing for heat optimization
- Error rates due to orchestration capacity
- Planned facility expansions or seasonal load increases
- Multi-region deployment for global operations

**Scaling Types**:
- **Horizontal Scaling**: Add/remove master pods (3-20 replicas based on plant count)
- **Vertical Scaling**: Increase pod resources for complex optimization
- **Sub-Agent Scaling**: Scale individual agent groups (boiler, furnace, heat recovery)
- **Database Scaling**: PostgreSQL + TimescaleDB for multi-plant time-series data
- **Message Bus Scaling**: Kafka/RabbitMQ for agent coordination
- **Multi-Region Scaling**: Geographic distribution for global facilities

**Capacity Planning Formulas**:
```
Master Replicas = ceil(plant_count / 5) + ceil(active_subagent_count / 10)
Master CPU = 1000m + (plant_count * 100m) + (subagent_count * 50m)
Master Memory = 2Gi + (plant_count * 200Mi) + (subagent_count * 100Mi)
Database Connections = plant_count * 5 + subagent_count * 2
Message Queue Workers = ceil(subagent_count / 20)
```

**Includes**:
- Scaling triggers and thresholds for master orchestrator
- Manual and automatic scaling procedures (HPA)
- Resource optimization for multi-plant operations
- Performance testing procedures with load scenarios
- Capacity planning for large facilities (10+ plants)
- Cost optimization strategies for enterprise deployment
- Sub-agent scaling coordination

**Quick Links**:
- [When to Scale Master Orchestrator](SCALING_GUIDE.md#when-to-scale-master)
- [Horizontal Scaling (Multi-Plant)](SCALING_GUIDE.md#horizontal-scaling-hpa)
- [Sub-Agent Group Scaling](SCALING_GUIDE.md#sub-agent-group-scaling)
- [Database Scaling (TimescaleDB)](SCALING_GUIDE.md#database-scaling-timescaledb)
- [Multi-Region Deployment](SCALING_GUIDE.md#multi-region-deployment)
- [Capacity Planning Calculator](SCALING_GUIDE.md#capacity-planning-calculator)

---

## Quick Reference Guide

### Common Commands

#### Check Master Orchestrator Health
```bash
# Master pod status
kubectl get pods -n greenlang | grep gl-001

# Health endpoint
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -f http://localhost:8000/api/v1/health

# Master orchestrator metrics
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | grep -E '(orchestration_latency|agent_coordination|heat_optimization)'

# Resource usage
kubectl top pods -n greenlang | grep gl-001

# Sub-agent coordination status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | jq '.coordinated_agents'
```

#### Check Multi-Plant Status
```bash
# All plants status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/status | jq '.plants[] | {plant_id, heat_status, efficiency}'

# Plant heat supply status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status

# Sub-agent health by plant
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/health | jq 'group_by(.plant_id)'
```

#### Emergency Rollback
```bash
# Emergency rollback for master orchestrator (5 minutes)
kubectl rollout undo deployment/gl-001-process-heat-orchestrator -n greenlang
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang --timeout=5m

# Verify critical heat operations restored
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/critical-heat-status
```

#### Scale Master Orchestrator
```bash
# Manual scale for large facility (10 plants, 50 sub-agents)
kubectl scale deployment gl-001-process-heat-orchestrator --replicas=8 -n greenlang

# Check HPA status
kubectl get hpa gl-001-process-heat-orchestrator-hpa -n greenlang

# Check sub-agent scaling
kubectl get hpa -n greenlang | grep -E '(gl-002|gl-003|gl-004|gl-005)'
```

#### View Master Orchestrator Logs
```bash
# Recent logs
kubectl logs -n greenlang deployment/gl-001-process-heat-orchestrator --tail=100

# Follow logs
kubectl logs -f -n greenlang deployment/gl-001-process-heat-orchestrator

# Filter for orchestration errors
kubectl logs -n greenlang deployment/gl-001-process-heat-orchestrator --tail=500 | \
  grep -E '(ERROR|coordination_failed|agent_timeout|heat_optimization_failed)'

# Previous pod logs (if crashed)
kubectl logs -n greenlang <pod-name> --previous
```

#### Restart Master Orchestrator
```bash
# Graceful restart with sub-agent coordination
kubectl rollout restart deployment/gl-001-process-heat-orchestrator -n greenlang

# Monitor restart and verify agents reconnect
kubectl rollout status deployment/gl-001-process-heat-orchestrator -n greenlang

# Verify sub-agents reconnected
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | jq '.agents[] | select(.status != "connected")'
```

#### Multi-Plant Coordination Commands
```bash
# Check message bus health (Kafka/RabbitMQ)
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/message-bus/health

# View agent coordination queue
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/coordination/queue | jq '.pending_tasks'

# Check heat distribution optimization status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/optimization/heat-distribution/status
```

---

## Decision Trees

### Issue Resolution Decision Tree

```
Is critical heat loss occurring (any plant)?
├─ YES → Use INCIDENT_RESPONSE.md (P0 - Total Heat Loss)
│         └─ Follow Emergency Heat Restoration Procedures
│         └─ Escalate to Plant Safety Officer immediately
│         └─ Recent deployment? → Consider Emergency Rollback (<5 min)
└─ NO
   └─ Is master orchestrator down (all pods failing)?
      ├─ YES → Use INCIDENT_RESPONSE.md (P0 - Master Orchestrator Failure)
      │         └─ Recent deployment? → Use ROLLBACK_PROCEDURE.md
      │         └─ Check sub-agent status (may need coordinated restart)
      └─ NO
         └─ Are multiple sub-agents failing (cascade)?
            ├─ YES → Use INCIDENT_RESPONSE.md (P1 - Cascade Failure)
            │         └─ Check message bus health
            │         └─ Recent deployment? → Use ROLLBACK_PROCEDURE.md
            └─ NO
               └─ Is heat distribution optimization failing?
                  ├─ YES → Use TROUBLESHOOTING.md (Heat Optimization Issues)
                  │         └─ Check LP solver status
                  │         └─ Review plant constraint violations
                  └─ NO
                     └─ Is performance degraded (high latency, low throughput)?
                        ├─ YES → Use TROUBLESHOOTING.md (Performance Issues)
                        │         └─ Resource constraints? → Use SCALING_GUIDE.md
                        │         └─ Database slow? → Scale TimescaleDB
                        └─ NO → Use TROUBLESHOOTING.md (specific issue)
```

### Scaling Decision Tree

```
What type of issue?
├─ High CPU/Memory on master → SCALING_GUIDE.md (Vertical or Horizontal Scaling)
│   └─ Calculate: Replicas = ceil(plants/5) + ceil(agents/10)
│
├─ Adding new plants (5+) → SCALING_GUIDE.md (Multi-Plant Scaling)
│   └─ Scale master, database, message bus proportionally
│
├─ Sub-agent count growing (50+) → SCALING_GUIDE.md (Sub-Agent Scaling)
│   └─ Scale by agent group (boiler, furnace, heat recovery)
│
├─ Slow heat optimization (<2s target) → TROUBLESHOOTING.md first, then SCALING_GUIDE.md
│   └─ May need vertical scaling for complex LP problems
│
├─ Database slow (>100ms query) → SCALING_GUIDE.md (Database Scaling)
│   └─ Add read replicas, scale TimescaleDB
│
├─ Message bus overload → SCALING_GUIDE.md (Message Bus Scaling)
│   └─ Scale Kafka partitions or RabbitMQ nodes
│
└─ Planned facility expansion → SCALING_GUIDE.md (Capacity Planning)
    └─ Use capacity formulas to estimate required resources
```

### Rollback Decision Tree

```
Should we rollback?
├─ Critical heat loss after deployment? → YES - Emergency Rollback (<5 min)
│
├─ Master orchestrator failing after deployment? → YES - Coordinated Rollback (10 min)
│   └─ Coordinate sub-agent rollback if using new master features
│
├─ Heat optimization errors after deployment? → YES - Specific Revision Rollback (15 min)
│   └─ Verify calculation correctness before rollback
│
├─ Single plant issues after deployment? → MAYBE - Partial Rollback (8 min)
│   └─ If isolated to one plant, rollback only that plant's config
│
├─ Performance degradation after deployment? → MAYBE - Monitor 15 minutes
│   └─ If CPU/latency doesn't improve, rollback
│   └─ Else, scale resources (may be load-related)
│
└─ Minor issues after deployment? → NO - Fix forward
    └─ Use hotfix deployment instead of rollback
```

---

## On-Call Quick Start

If you're on-call for GL-001 ProcessHeatOrchestrator and got paged:

### Step 1: Acknowledge Alert (Within 5 Minutes)
- Acknowledge in PagerDuty
- Post acknowledgement in #gl-001-incidents Slack channel
- Note: P0 alerts for heat loss require **immediate action** (safety critical)

### Step 2: Assess Severity

**P0 (Critical) - Immediate Action Required**:
- Total heat loss across facility (any plant)
- Master orchestrator completely down (all pods failing)
- Multi-plant cascade failure (3+ plants affected)
- Safety hazard (pressure alarm, fire risk, chemical exposure)
- **Action**: Use INCIDENT_RESPONSE.md P0 procedures immediately

**P1 (High) - 15 Minute Response**:
- Single plant heat loss
- Major sub-agent cascade failure (10+ agents)
- SCADA connection loss (multi-plant)
- Critical heat distribution optimization failure
- **Action**: Use INCIDENT_RESPONSE.md P1 procedures

**P2 (Medium) - 1 Hour Response**:
- Heat distribution inefficiency (>20% from optimal)
- Minor sub-agent degradation (3-5 agents)
- Performance degradation (latency >5s)
- **Action**: Use TROUBLESHOOTING.md

### Step 3: Check Recent Changes
```bash
# Check deployment history
kubectl rollout history deployment/gl-001-process-heat-orchestrator -n greenlang

# Check recent config changes
kubectl get configmap gl-001-config -n greenlang -o yaml | grep -A 5 "last-applied"

# Check sub-agent deployment history (if cascade failure)
kubectl rollout history deployment/gl-002-boiler-efficiency -n greenlang
kubectl rollout history deployment/gl-003-combustion-optimizer -n greenlang
```

**Rollback Criteria**:
- Deployment <2 hours ago AND issue matches deployment timing → Consider rollback
- Multiple plants affected AND recent master deployment → Emergency rollback
- Sub-agent cascade AND new master version → Coordinated rollback

### Step 4: Assess Impact

```bash
# Master orchestrator status
kubectl get pods -n greenlang | grep gl-001

# Multi-plant heat status
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/plants/heat-status | jq '.plants[] | select(.heat_loss == true)'

# Sub-agent coordination health
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/agents/status | jq '.agents[] | select(.status != "healthy")'

# Error rate and latency
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | \
  grep -E '(error_rate|orchestration_latency|heat_optimization_failures)'
```

### Step 5: Follow Appropriate Runbook

**Decision Matrix**:
| Condition | Runbook | Priority |
|-----------|---------|----------|
| Critical heat loss | INCIDENT_RESPONSE.md (P0) | IMMEDIATE |
| Master down + recent deploy | ROLLBACK_PROCEDURE.md (Emergency) | IMMEDIATE |
| Multi-plant cascade | INCIDENT_RESPONSE.md (P0/P1) | IMMEDIATE |
| Single plant heat loss | INCIDENT_RESPONSE.md (P1) | 15 min |
| Heat optimization failing | TROUBLESHOOTING.md → Heat Distribution | 1 hour |
| Sub-agent coordination issues | TROUBLESHOOTING.md → Agent Coordination | 1 hour |
| High CPU/Memory | SCALING_GUIDE.md | 1 hour |
| Performance degradation | TROUBLESHOOTING.md → Performance | 1 hour |

### Step 6: Communicate

**P0/P1 Communication Template**:
```
#gl-001-incidents

🚨 P0/P1 INCIDENT: [Brief Description]

Severity: P0/P1
Affected Plants: [Plant IDs]
Impact: [Heat loss? Sub-agents down? Count]
Started: [Timestamp]
On-Call: @your-name

Status: Investigating / Mitigating / Resolved
ETA: [Time estimate]

Current Actions:
- [Action 1]
- [Action 2]

Next Update: [Time]
```

**Escalation**:
- **P0**: Immediately escalate to Plant Operations Manager AND Plant Safety Officer
- **P1**: Escalate to Plant Operations Manager if not resolved in 30 minutes
- **P2**: Escalate if not resolved in 2 hours

### Step 7: Update Status Page (P0/P1 Only)

For P0/P1 incidents affecting production:
```bash
# Update status page
curl -X POST https://status.greenlang.io/api/v1/incidents \
  -H "Authorization: Bearer $STATUS_API_KEY" \
  -d '{
    "name": "GL-001 Process Heat Operations Issue",
    "status": "investigating",
    "impact": "critical",
    "affected_components": ["process-heat-orchestrator"],
    "message": "Investigating heat loss at Plant-001"
  }'
```

---

## Monitoring and Alerting

### Dashboards

- **Grafana Main Dashboard**: https://grafana.greenlang.io/d/gl-001/process-heat-orchestrator
- **Multi-Plant Heat Status**: https://grafana.greenlang.io/d/gl-001-plants/multi-plant-heat-status
- **Sub-Agent Coordination**: https://grafana.greenlang.io/d/gl-001-agents/agent-coordination
- **Heat Optimization Metrics**: https://grafana.greenlang.io/d/gl-001-optimization/heat-optimization
- **Error Analysis**: https://grafana.greenlang.io/d/gl-001-errors/error-analysis
- **Scaling Metrics**: https://grafana.greenlang.io/d/gl-001-scaling/scaling-operations

### Key Metrics

```bash
# Check all key metrics at once
kubectl exec -n greenlang deployment/gl-001-process-heat-orchestrator -- \
  curl -s http://localhost:8000/api/v1/metrics | \
  grep -E '(orchestration_latency|agent_coordination_success_rate|heat_optimization_time|error_rate|cpu|memory|db_pool|message_bus_lag)'
```

**Target Metrics**:
- Orchestration latency (p95): <2 seconds
- Agent coordination success rate: >99%
- Heat optimization time (p95): <5 seconds
- Error rate: <0.5%
- CPU usage: <70%
- Memory usage: <75%
- Database pool utilization: <80%
- Message bus lag: <100ms
- Sub-agent availability: >99%
- Heat distribution efficiency: >85%

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| **Critical Heat Loss** | Any plant | Any plant | INCIDENT_RESPONSE.md (P0) |
| Master pod count | 1/3 available | 0/3 available | INCIDENT_RESPONSE.md (P0) |
| Sub-agent cascade | 5 agents down | 10+ agents down | INCIDENT_RESPONSE.md (P1) |
| Orchestration latency (p95) | >3s | >5s | TROUBLESHOOTING.md → SCALING_GUIDE.md |
| Heat optimization failures | >5% | >20% | TROUBLESHOOTING.md |
| Agent coordination failures | >2% | >10% | TROUBLESHOOTING.md |
| Error rate | >2% | >10% | TROUBLESHOOTING.md |
| CPU usage | >70% | >85% | SCALING_GUIDE.md |
| Memory usage | >75% | >90% | SCALING_GUIDE.md |
| Database connections | >80% | >95% | SCALING_GUIDE.md |
| Message bus lag | >500ms | >2s | SCALING_GUIDE.md |
| SCADA connection loss | 1 plant | Multi-plant | INCIDENT_RESPONSE.md (P1) |

### PagerDuty Alert Routing

| Alert | Severity | Response Time | Escalation |
|-------|----------|---------------|------------|
| `gl-001.critical_heat_loss` | P0 | Immediate | Primary → Secondary → Plant Safety Officer |
| `gl-001.master_down` | P0 | Immediate | Primary → Secondary → Operations Manager |
| `gl-001.multi_plant_cascade` | P0 | Immediate | Primary → Secondary → Operations Manager |
| `gl-001.plant_heat_loss` | P1 | 15 min | Primary → Secondary → Operations Manager (30 min) |
| `gl-001.agent_cascade` | P1 | 15 min | Primary → Secondary |
| `gl-001.scada_outage` | P1 | 15 min | Primary → Secondary |
| `gl-001.optimization_failure` | P2 | 1 hour | Primary → Secondary |
| `gl-001.performance_degradation` | P2 | 1 hour | Primary only |
| `gl-001.high_resource_usage` | P2 | 1 hour | Primary only |

---

## Escalation Contacts

### Primary On-Call Engineer
- **PagerDuty**: Service ID "GL-001 Primary"
- **Slack**: @gl-001-oncall-primary
- **Response Time**: 5 minutes for all alerts
- **Responsibility**: Initial investigation and mitigation

### Secondary On-Call Engineer
- **PagerDuty**: Service ID "GL-001 Secondary"
- **Slack**: @gl-001-oncall-secondary
- **Response Time**: 10 minutes (if primary doesn't respond)
- **Responsibility**: Support primary or take over if unavailable

### Plant Operations Manager
- **Escalation Trigger**: P0 incidents OR P1 incidents >30 minutes unresolved
- **Contact**: Via PagerDuty or plant-operations@greenlang.io
- **Slack**: @plant-operations-manager
- **Authority**: Can authorize emergency maintenance, process shutdown

### Plant Safety Officer
- **Escalation Trigger**: Any P0 heat loss incident OR safety hazard
- **Contact**: 24/7 hotline (see PagerDuty) or safety@greenlang.io
- **Slack**: @plant-safety-officer
- **Authority**: Can order immediate shutdown, override all automation

### Engineering Manager
- **Escalation Trigger**: P0 incidents >1 hour OR P1 incidents >4 hours
- **Contact**: engineering-manager@greenlang.io
- **Slack**: @gl-001-engineering-manager
- **Responsibility**: Resource allocation, external vendor engagement

### VP of Operations
- **Escalation Trigger**: Unresolved P0 incidents >2 hours OR multi-facility impact
- **Contact**: vp-operations@greenlang.io
- **Responsibility**: Executive decision making, customer communication

### External Vendors (24/7 Support)

**SCADA Vendor (Siemens/Honeywell)**:
- **Contact**: Via PagerDuty "SCADA Vendor" service
- **When to Call**: Multi-plant SCADA outage, integration failures
- **SLA**: 2-hour response for critical issues

**Database Vendor (TimescaleDB Enterprise Support)**:
- **Contact**: support@timescale.com or enterprise hotline
- **When to Call**: Database performance issues, corruption, replication failures
- **SLA**: 1-hour response for production-down scenarios

**Cloud Provider (AWS Enterprise Support)**:
- **Contact**: Via AWS Console or dedicated TAM
- **When to Call**: Infrastructure issues, networking, EKS problems
- **SLA**: 15-minute response for business-critical issues

---

## Runbook Maintenance

### Update Frequency
- **Quarterly Review**: Review all runbooks for accuracy and completeness
- **Post-Incident Update**: Update within 1 week after major incident (P0/P1)
- **Architecture Change Update**: Update within 1 sprint after significant changes
- **New Scenario Addition**: Add new troubleshooting scenarios as discovered

### Review Process
1. Engineering team reviews runbooks quarterly
2. Incorporate lessons learned from incidents
3. Test procedures in staging environment
4. Update based on feedback from on-call engineers
5. Version control all changes in Git

### Contributors
- **Platform Engineering**: Infrastructure procedures, Kubernetes operations
- **DevOps Team**: CI/CD, deployment procedures, scaling operations
- **Process Heat Engineering**: Domain-specific troubleshooting, safety protocols
- **SRE Team**: Monitoring, alerting, performance optimization
- **Operations Team**: Real-world incident feedback, practical improvements

### Feedback Process
- **Found an issue?** Create JIRA ticket: Project "GL-001", Type "Runbook Issue", Label "runbook-improvement"
- **Have a suggestion?** Post in #gl-001-team Slack channel
- **Need clarification?** Ask in #gl-001-ops or #gl-001-oncall
- **Urgent correction?** Contact @gl-001-oncall-lead immediately

---

## Related Documentation

### Internal Documentation
- **Architecture**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001\ARCHITECTURE.md`
- **Agent Specification**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001\agent_spec.yaml`
- **Deployment Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001\deployment\README.md`
- **API Documentation**: https://docs.greenlang.io/agents/gl-001/api
- **Integration Guide**: `C:\Users\aksha\Code-V1_GreenLang\GreenLang_2030\agent_foundation\agents\GL-001\INTEGRATION_GUIDE.md`
- **Sub-Agent Runbooks**:
  - GL-002 (Boiler Efficiency): `../GL-002/runbooks/`
  - GL-003 (Combustion Optimizer): `../GL-003/runbooks/`
  - GL-004 (Emissions Control): `../GL-004/runbooks/`
  - GL-005 (Heat Recovery): `../GL-005/runbooks/`

### External Resources
- **Kubernetes Documentation**: https://kubernetes.io/docs/
- **PostgreSQL Performance**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **TimescaleDB Operations**: https://docs.timescale.com/
- **Prometheus Queries**: https://prometheus.io/docs/prometheus/latest/querying/basics/
- **Kafka Operations**: https://kafka.apache.org/documentation/#operations
- **SCADA Integration (OPC UA)**: https://opcfoundation.org/developer-tools/

### Industry Standards
- **ISO 50001**: Energy Management Systems
- **ASME PTC 4**: Performance Test Codes for Boilers
- **EPA CEMS**: Continuous Emissions Monitoring Systems
- **ISA-95**: Enterprise-Control System Integration
- **IEC 62264**: Enterprise-Control Integration

---

## Training and Certification

### Required Training for On-Call

**Prerequisites** (before on-call rotation):
1. **Kubernetes Operations Basics** (2 hours)
   - Pod management, deployments, services
   - kubectl commands, log viewing
   - Resource management

2. **GL-001 Architecture Overview** (2 hours)
   - Master orchestrator design
   - Sub-agent coordination patterns
   - Multi-plant data flows
   - SCADA/ERP integration

3. **Process Heat Domain Training** (3 hours)
   - Industrial process heat basics
   - Safety protocols and critical heat operations
   - Heat distribution optimization concepts
   - Emissions compliance requirements

4. **Incident Response Protocol** (2 hours)
   - Severity classification for process heat
   - Emergency heat restoration procedures
   - Escalation paths and communication
   - Post-incident review process

5. **On-Call Shadowing** (1 week)
   - Shadow experienced on-call engineer
   - Observe real incident handling
   - Practice troubleshooting scenarios
   - Learn communication protocols

### Runbook Certification Process

To be certified for GL-001 on-call rotation:

1. **Complete Required Training**: All 5 prerequisite courses
2. **Read All Runbooks**: README, TROUBLESHOOTING, INCIDENT_RESPONSE, ROLLBACK_PROCEDURE, SCALING_GUIDE
3. **Pass Runbook Quiz**: 80% pass rate (30 questions covering all scenarios)
4. **Complete Simulated Incidents**: Successfully handle 5 scenarios:
   - P0: Total heat loss scenario
   - P1: Sub-agent cascade failure
   - P2: Heat optimization failure
   - Emergency rollback procedure
   - Multi-plant scaling scenario
5. **Shadow On-Call**: 1 week minimum with experienced engineer
6. **Review by On-Call Lead**: Final approval for on-call rotation

### Simulated Incident Drills

**Monthly Gameday Exercises** (2 hours):
- Practice P0/P1 incident response
- Test emergency rollback procedures
- Validate escalation paths
- Practice multi-plant coordination
- Review recent real incidents

**Quarterly Full-Scale Drills** (4 hours):
- Multi-plant failure scenarios
- Sub-agent cascade recovery
- Database failover procedures
- Multi-region deployment testing
- Vendor coordination exercises

**Drill Scenarios**:
1. Total heat loss at Plant-001 (P0)
2. Master orchestrator pod failure cascade (P0)
3. SCADA connection loss affecting 3 plants (P1)
4. Sub-agent cascade: 15 agents down (P1)
5. Heat optimization LP solver failure (P2)
6. Database connection pool exhaustion (P2)
7. Emergency rollback with sub-agent coordination
8. Scaling for facility expansion (10 → 15 plants)

---

## Safety Protocols

### Critical Heat Operations Safety

**CRITICAL**: GL-001 manages life-safety critical heat operations. Always prioritize safety over optimization or cost savings.

**Safety-Critical Scenarios**:
- **Boiler pressure alarms**: Immediate escalation to Plant Safety Officer
- **Combustion instability**: May indicate explosion risk
- **Heat loss affecting safety systems**: Emergency procedures required
- **Chemical process temperature drops**: Product safety and environmental risk

**Emergency Heat Restoration Priority**:
1. **Life safety systems**: Hospital HVAC, chemical reactor heating (restore <10 minutes)
2. **Product safety**: Food processing, pharmaceutical manufacturing (restore <30 minutes)
3. **Equipment protection**: Freeze protection, condensate systems (restore <1 hour)
4. **Production optimization**: Normal manufacturing (restore <4 hours)

**Never Compromise On**:
- Pressure safety limits (auto-shutdown if exceeded)
- Emissions compliance (may incur regulatory penalties)
- Equipment protective limits (prevent catastrophic failure)
- Personnel safety (lock-out tag-out during maintenance)

**Safety Checklist Before Any Change**:
- [ ] Reviewed impact on critical heat supply
- [ ] Confirmed backup heat sources available
- [ ] Notified Plant Safety Officer (for P0/P1 changes)
- [ ] Verified protective interlocks remain active
- [ ] Confirmed emergency shutdown systems functional
- [ ] Documented rollback procedure

---

## Changelog

### Version 1.0 (2025-11-17)
- Initial creation of all 5 runbooks for GL-001
- README.md: Comprehensive overview and quick reference
- TROUBLESHOOTING.md: Multi-plant and sub-agent coordination issues
- INCIDENT_RESPONSE.md: Emergency procedures with safety protocols
- ROLLBACK_PROCEDURE.md: Multi-agent coordinated rollback procedures
- SCALING_GUIDE.md: Master orchestrator and sub-agent scaling

### Future Improvements
- [ ] Add screenshots and architecture diagrams
- [ ] Create video walkthroughs for complex procedures
- [ ] Expand SCADA vendor-specific troubleshooting
- [ ] Add chaos engineering scenarios
- [ ] Create automated runbook testing framework
- [ ] Add multi-region disaster recovery procedures
- [ ] Expand sub-agent specific coordination troubleshooting

---

## Support

### Questions?
Ask in Slack:
- **#gl-001-team**: General questions about GL-001
- **#gl-001-ops**: Operations and deployment questions
- **#gl-001-incidents**: Active incidents only (monitored 24/7)
- **#gl-001-oncall**: On-call engineer questions and handoffs
- **#process-heat**: Process heat domain questions

### Urgent?
Page on-call:
- **PagerDuty**: Service "GL-001 Production - Process Heat Orchestrator"
- **Phone**: See PagerDuty escalation policy for emergency hotline
- **Slack**: @gl-001-oncall-primary (mention for immediate notification)

### Documentation Issues?
- **JIRA Ticket**: Project "GL-001", Type "Documentation", Label "runbook-improvement"
- **Assignee**: TechWriter team
- **Priority**: Set based on urgency (P0 for safety issues, P3 for minor clarifications)
- **Include**: Runbook name, section, issue description, suggested fix

---

## Appendix: Multi-Plant Reference Data

### Typical Plant Configurations

| Plant Type | Heat Demand (MW) | Sub-Agents | Master Replicas | DB Connections |
|------------|------------------|------------|-----------------|----------------|
| Small (1-2 plants) | 10-50 MW | 10-20 | 3 | 30-50 |
| Medium (3-5 plants) | 50-200 MW | 20-40 | 5 | 50-100 |
| Large (6-10 plants) | 200-500 MW | 40-70 | 8 | 100-200 |
| Enterprise (10+ plants) | 500+ MW | 70-99 | 10-20 | 200-400 |

### Sub-Agent Groups by Function

**Boiler and Steam** (GL-002, GL-012, GL-016, GL-017, GL-022, GL-042, GL-043, GL-044):
- Critical for heat generation
- High coordination with master
- Typical: 2-4 agents per plant

**Combustion and Emissions** (GL-004, GL-005, GL-010, GL-018, GL-021, GL-026, GL-029, GL-053):
- Safety and compliance critical
- Real-time coordination required
- Typical: 1-2 agents per plant

**Heat Recovery** (GL-006, GL-014, GL-020, GL-024, GL-030, GL-033, GL-038, GL-039):
- Efficiency optimization
- Lower priority than critical heat
- Typical: 1-2 agents per plant

### Contact Information Quick Reference

| Role | Slack | Email | PagerDuty | Phone |
|------|-------|-------|-----------|-------|
| Primary On-Call | @gl-001-oncall-primary | See PagerDuty | GL-001 Primary | See PD |
| Secondary On-Call | @gl-001-oncall-secondary | See PagerDuty | GL-001 Secondary | See PD |
| Plant Safety Officer | @plant-safety-officer | safety@greenlang.io | Safety Escalation | 24/7 Hotline |
| Operations Manager | @plant-operations-manager | plant-ops@greenlang.io | Operations | See PD |
| Engineering Manager | @gl-001-engineering-manager | eng-mgr@greenlang.io | Engineering | Business hours |

---

**Last Updated**: 2025-11-17
**Version**: 1.0
**Maintained By**: GreenLang Platform Engineering & Process Heat Team
**Review Cycle**: Quarterly or post-major-incident
