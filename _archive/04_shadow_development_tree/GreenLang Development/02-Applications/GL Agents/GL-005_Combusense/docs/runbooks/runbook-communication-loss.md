# Runbook: Communication Loss

## Document Control
| Field | Value |
|-------|-------|
| Document ID | GL005-RB-003 |
| Version | 1.0.0 |
| Last Updated | 2025-12-22 |
| Owner | GreenLang Operations Team |
| Classification | HIGH - Safety Related |
| Review Cycle | Monthly |

---

## Overview

This runbook provides procedures for handling communication loss between the GL-005 CombustionControlAgent and industrial control systems (DCS, PLC, SCADA). Communication failures can prevent control actions, compromise safety systems, and require manual intervention.

**Systems Covered:**
- DCS (Distributed Control System) - OPC UA / Modbus TCP
- PLC (Programmable Logic Controller) - Modbus TCP
- SCADA (Supervisory Control and Data Acquisition)
- Network infrastructure (switches, firewalls, routers)

---

## DCS/PLC Connectivity Loss

### Detection Criteria

| Connection Type | Detection Method | Timeout | Alert |
|----------------|------------------|---------|-------|
| OPC UA Primary | Connection state | 10s | CRITICAL |
| Modbus TCP Backup | Connection state | 10s | HIGH |
| Data subscription | Heartbeat timeout | 5s | WARNING |
| Write confirmation | Ack timeout | 2s | HIGH |

### Symptoms

**Immediate Indicators:**
- Alert: `gl005_dcs_connection_status{status="disconnected"}`
- DCS/HMI shows "COMMUNICATION FAILURE"
- Control outputs frozen at last values
- Alarms not propagating to operator console

**System Logs:**
```
CRITICAL - DCSConnector - OPC UA connection lost to opc.tcp://dcs.plant.com:4840
WARNING - DCSConnector - Circuit breaker OPEN for DCS connection
INFO - DCSConnector - Attempting failover to Modbus TCP backup
ERROR - PLCConnector - Modbus TCP connection timeout after 5 attempts
```

### Immediate Actions

**Step 1: Verify Connection Status**
```bash
# Check DCS connector status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/dcs/status | jq '.'

# Check PLC connector status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/plc/status | jq '.'

# Check circuit breaker states
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "circuit_breaker_state"
```

**Step 2: Verify Automatic Failover**
```bash
# Check if backup connection is active
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/active-protocol | jq '.'

# Expected response (failover successful):
# {
#   "primary_protocol": "opc_ua",
#   "primary_status": "DISCONNECTED",
#   "active_protocol": "modbus_tcp",
#   "active_status": "CONNECTED",
#   "failover_time": "2025-12-22T10:30:00Z"
# }
```

**Step 3: Notify Operations**
```bash
# If both primary and backup failed
# IMMEDIATELY notify control room and escalate

# Check if control is still possible
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/control/capability | jq '.'
```

**Step 4: Enable Manual Mode (if no control)**
```bash
# Switch to manual mode at DCS (field operation)
# This is done at the DCS/HMI console, not via GL-005

# Notify GL-005 of manual mode
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/acknowledge-manual-mode \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"reason": "Communication failure - DCS manual takeover"}'
```

### Root Cause Investigation

**Network Layer Diagnostics:**
```bash
# Ping DCS host
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 5 <DCS_HOST>

# Check port connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  nc -zv <DCS_HOST> 4840

# Check network latency
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  traceroute <DCS_HOST>

# Check for packet loss
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  ping -c 100 <DCS_HOST> | tail -5
```

**Application Layer Diagnostics:**
```bash
# Check OPC UA server health
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from asyncua import Client
import asyncio

async def check():
    client = Client(url='opc.tcp://<DCS_HOST>:4840')
    try:
        await asyncio.wait_for(client.connect(), timeout=5)
        print('OPC UA server reachable')
        await client.disconnect()
    except Exception as e:
        print(f'OPC UA server unreachable: {e}')

asyncio.run(check())
"

# Check Modbus TCP connectivity
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  python -c "
from pymodbus.client import ModbusTcpClient

client = ModbusTcpClient('<PLC_HOST>', port=502)
if client.connect():
    print('Modbus server reachable')
    result = client.read_holding_registers(0, 1)
    print(f'Read test: {result}')
    client.close()
else:
    print('Modbus server unreachable')
"
```

**Common Root Causes:**

| Cause | Probability | Symptoms |
|-------|-------------|----------|
| Network switch failure | HIGH | Multiple systems affected |
| Firewall rule change | MEDIUM | Sudden connectivity loss |
| DCS server crash | MEDIUM | OPC UA specific failure |
| IP address conflict | LOW | Intermittent connectivity |
| Cable damage | LOW | Affects specific path |
| TLS certificate expiry | MEDIUM | OPC UA authentication failure |

### Recovery Procedures

**Scenario A: Network Issue**
```bash
# If network switch/firewall issue identified:
# 1. Contact network operations team
# 2. Check for recent network changes
# 3. Verify VLAN configuration
# 4. Check firewall rules for OPC UA (port 4840) and Modbus (port 502)
```

**Scenario B: DCS Server Issue**
```bash
# Work with DCS vendor/administrator:
# 1. Check DCS server logs
# 2. Restart OPC UA server if needed
# 3. Verify OPC UA security certificates
# 4. Check DCS server resource utilization
```

**Scenario C: GL-005 Configuration Issue**
```bash
# Update connection configuration
kubectl set env deployment/gl-005-combustion-control -n greenlang \
  DCS_OPCUA_ENDPOINT="opc.tcp://<NEW_HOST>:4840" \
  DCS_MODBUS_HOST="<NEW_PLC_HOST>"

# Force circuit breaker reset
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/integrations/circuit-breaker/reset \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"protocol": "opc_ua"}'
```

**Scenario D: TLS/Certificate Issue**
```bash
# Check certificate expiry
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  openssl s_client -connect <DCS_HOST>:4840 2>/dev/null | \
  openssl x509 -noout -dates

# Update certificates if expired
kubectl create secret tls dcs-certificates \
  --cert=new-cert.pem --key=new-key.pem \
  -n greenlang --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment to pick up new certificates
kubectl rollout restart deployment/gl-005-combustion-control -n greenlang
```

---

## SCADA Timeout Handling

### Detection Criteria

| Issue | Detection | Timeout | Impact |
|-------|-----------|---------|--------|
| MQTT connection lost | Heartbeat missing | 30s | Data not published |
| OPC UA server unresponsive | Client timeout | 10s | HMI not updated |
| Data queue backup | Queue depth >1000 | N/A | Delayed updates |
| Alarm publication failure | ACK timeout | 5s | Missed alarms |

### Symptoms

**Operator Console:**
- "STALE DATA" indicators on HMI
- Trend charts frozen
- Alarms not appearing in alarm list
- Heartbeat indicator red

**System Logs:**
```
WARNING - SCADAIntegration - MQTT connection lost to mqtt.plant.com
ERROR - SCADAIntegration - OPC UA publish failed: timeout
INFO - SCADAIntegration - Data queue depth: 1500 (exceeds threshold)
CRITICAL - SCADAIntegration - Alarm publication failed for TEMP_HIGH_001
```

### Immediate Actions

**Step 1: Check SCADA Connection Status**
```bash
# Check SCADA integration status
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/scada/status | jq '.'

# Check MQTT connection
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8001/metrics | grep "scada_connection_status"

# Check data queue depth
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/scada/queue-depth | jq '.'
```

**Step 2: Verify Local Data Logging**
```bash
# Ensure local historian is capturing data
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/historian/status | jq '.'

# Check local log buffer
kubectl logs -n greenlang deployment/gl-005-combustion-control --tail=50 | \
  grep "SCADA\|historian"
```

**Step 3: Manual Alarm Notification**
```bash
# If SCADA alarms not propagating, use alternative notification
# Notify control room directly

# Check for pending critical alarms
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/alarms/active | jq '.[] | select(.severity == "CRITICAL")'
```

### Recovery Procedures

**MQTT Reconnection:**
```bash
# Force MQTT reconnection
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/integrations/scada/mqtt/reconnect \
  -H "Authorization: Bearer $OPERATOR_TOKEN"

# Monitor reconnection
kubectl logs -n greenlang deployment/gl-005-combustion-control -f | \
  grep "MQTT"
```

**OPC UA Server Restart:**
```bash
# Restart SCADA OPC UA server (on GL-005 side)
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/integrations/scada/opcua/restart \
  -H "Authorization: Bearer $OPERATOR_TOKEN"
```

**Queue Flush:**
```bash
# If queue is backing up, flush non-critical data
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/integrations/scada/queue/flush \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"priority_threshold": "LOW", "retain_alarms": true}'
```

---

## Network Partition Recovery

### Detection

Network partitions occur when GL-005 can reach some systems but not others.

**Symptoms:**
- DCS connection healthy, SCADA connection lost
- Partial sensor data available
- Inconsistent readings from redundant systems

### Diagnosis

```bash
# Check all integration endpoints
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/integrations/health | jq '.'

# Expected output:
# {
#   "dcs": {"status": "CONNECTED", "latency_ms": 15},
#   "plc": {"status": "CONNECTED", "latency_ms": 8},
#   "scada": {"status": "DISCONNECTED", "latency_ms": null},
#   "historian": {"status": "CONNECTED", "latency_ms": 25}
# }

# Check network segmentation
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  bash -c "
    echo 'DCS Network:'; ping -c 1 <DCS_HOST> | grep 'bytes from'
    echo 'PLC Network:'; ping -c 1 <PLC_HOST> | grep 'bytes from'
    echo 'SCADA Network:'; ping -c 1 <SCADA_HOST> | grep 'bytes from'
    echo 'Corporate Network:'; ping -c 1 <CORP_HOST> | grep 'bytes from'
  "
```

### Recovery

**Step 1: Identify Partition Boundary**
```bash
# Determine which network segment is unreachable
# Contact network operations with findings
```

**Step 2: Prioritize Critical Connections**
```bash
# If DCS/PLC connected, control is possible
# SCADA loss is secondary (monitoring only)

# Increase local logging to compensate for SCADA loss
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/logging/increase-retention \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{"retention_hours": 72}'
```

**Step 3: Coordinate with Network Team**
```bash
# Document affected systems and impact
# Escalate to network operations
# Track resolution in incident ticket
```

---

## Manual Mode Operation

### When to Use Manual Mode

| Condition | Action |
|-----------|--------|
| All automated control connections lost | REQUIRED |
| Safety system connection lost | REQUIRED |
| Control instability from communication delays | OPTIONAL |
| Planned maintenance | PLANNED |

### Manual Takeover Procedure

**Step 1: Notify GL-005 of Manual Takeover**
```bash
# Acknowledge manual mode in GL-005
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/manual-takeover \
  -H "Authorization: Bearer $OPERATOR_TOKEN" \
  -d '{
    "reason": "Communication failure",
    "operator_id": "OPERATOR_001",
    "expected_duration_hours": 2
  }'
```

**Step 2: At DCS/HMI Console**
1. Place all control loops in MANUAL mode
2. Note current setpoints and output values
3. Verify manual control is responsive
4. Monitor process closely (increase operator attention)

**Step 3: At Field Level (if DCS also unavailable)**
1. Locate manual bypass switches
2. Use portable instruments for readings
3. Adjust fuel valve manually based on visual flame
4. Maintain safe excess air (increase for margin)
5. Stay in communication with control room via radio

### Operator Guidance During Manual Mode

**Target Parameters (Conservative):**
| Parameter | Normal | Manual Mode Target |
|-----------|--------|-------------------|
| Excess Air | 10-15% | 18-20% (safety margin) |
| Fuel Flow | Optimized | Reduced 10% |
| Combustion Temp | 1000-1200C | 900-1100C |
| O2 in Flue Gas | 2-4% | 4-6% (safety margin) |

**Do:**
- Make small, incremental adjustments
- Wait for process to stabilize before next change
- Document all manual actions
- Monitor flame visually when possible
- Keep communication channels open

**Do Not:**
- Make rapid changes to fuel or air
- Ignore unusual sounds or vibrations
- Operate above normal capacity
- Leave station unattended

### Return to Automatic Control

**Prerequisites:**
- [ ] Communication restored and verified stable (>10 min)
- [ ] All sensor readings valid and matching redundant sensors
- [ ] Control system shows ready status
- [ ] Operator verifies process is stable
- [ ] Supervisor authorizes return to AUTO

**Procedure:**
```bash
# Verify GL-005 readiness
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -s http://localhost:8000/control/readiness | jq '.'

# Initiate return to automatic
kubectl exec -n greenlang deployment/gl-005-combustion-control -- \
  curl -X POST http://localhost:8000/control/resume-auto \
  -H "Authorization: Bearer $SUPERVISOR_TOKEN" \
  -d '{
    "supervisor_id": "SUPERVISOR_001",
    "bumpless_transfer": true
  }'
```

At DCS console:
1. Verify GL-005 setpoints match current process
2. Place loops in CASCADE one at a time
3. Monitor for smooth transition
4. Verify control is tracking setpoints
5. Document return to AUTO

---

## Prevention Measures

### Redundancy Configuration

```yaml
# Recommended communication redundancy
primary_protocol: opc_ua
secondary_protocol: modbus_tcp
tertiary_protocol: local_historian

# Network redundancy
primary_network: industrial_vlan_100
secondary_network: industrial_vlan_200
network_failover_time: 2s

# Watchdog configuration
dcs_watchdog_timeout: 10s
plc_watchdog_timeout: 5s
scada_watchdog_timeout: 30s
```

### Monitoring

```yaml
# Prometheus alerts for communication health
- alert: DCSConnectionUnstable
  expr: rate(gl005_dcs_connection_errors_total[5m]) > 0.1
  for: 2m
  labels:
    severity: warning

- alert: CommunicationLatencyHigh
  expr: gl005_dcs_read_latency_seconds > 0.1
  for: 5m
  labels:
    severity: warning

- alert: CircuitBreakerHalfOpen
  expr: gl005_circuit_breaker_state == 2
  for: 1m
  labels:
    severity: warning
```

### Maintenance

| Task | Frequency | Owner |
|------|-----------|-------|
| Network path test | Weekly | Network Operations |
| Certificate expiry check | Monthly | Security |
| Failover drill | Quarterly | Operations |
| Manual mode drill | Quarterly | Operations |
| Full communication audit | Annually | Engineering |

---

## Appendix

### A. Connection State Diagram

```
CONNECTED (Normal)
    |
    v [Communication Error]
CONNECTING (Retry)
    |
    +---> [Success] --> CONNECTED
    |
    v [Failure Threshold Exceeded]
CIRCUIT_OPEN (Blocked)
    |
    v [Recovery Timeout Elapsed]
HALF_OPEN (Testing)
    |
    +---> [Test Success] --> CONNECTED
    |
    +---> [Test Failure] --> CIRCUIT_OPEN
```

### B. Related Documentation

- [runbook-emergency-shutdown.md](./runbook-emergency-shutdown.md) - Shutdown due to communication loss
- [DCS Connector Documentation](../../integrations/dcs_connector.py) - Connection implementation
- [SCADA Integration Documentation](../../integrations/scada_integration.py) - SCADA protocols

### C. Emergency Contacts

| Role | Contact |
|------|---------|
| Network Operations | @network-oncall |
| DCS Support (Vendor) | (XXX) XXX-XXXX |
| PLC Support (Vendor) | (XXX) XXX-XXXX |
| Control Room | (XXX) XXX-XXXX |

### D. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-22 | GL-TechWriter | Initial version |
