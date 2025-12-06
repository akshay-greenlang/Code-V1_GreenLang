# GreenLang Process Heat Platform - Operator Manual

**Document Version:** 1.0.0
**Last Updated:** 2025-12-06
**Classification:** Operational Documentation
**Intended Audience:** Plant Operators, Control Room Personnel, Shift Supervisors

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Login and Authentication](#2-login-and-authentication)
3. [Dashboard Navigation](#3-dashboard-navigation)
4. [Agent Monitoring and Status](#4-agent-monitoring-and-status)
5. [Alarm Acknowledgment and Response](#5-alarm-acknowledgment-and-response)
6. [Manual Overrides and Bypasses](#6-manual-overrides-and-bypasses)
7. [Shift Handover Procedures](#7-shift-handover-procedures)
8. [Emergency Procedures](#8-emergency-procedures)
9. [Appendix A: Agent Reference Card](#appendix-a-agent-reference-card)
10. [Appendix B: Alarm Priority Matrix](#appendix-b-alarm-priority-matrix)

---

## 1. System Overview

### 1.1 Purpose

The GreenLang Process Heat Platform is an intelligent multi-agent system designed to optimize thermal operations, reduce emissions, and ensure regulatory compliance. The platform continuously monitors boilers, steam systems, heat exchangers, and emissions equipment to provide real-time optimization recommendations and safety oversight.

### 1.2 System Architecture

```
+------------------------------------------------------------------+
|                    GreenLang Process Heat Platform                 |
+------------------------------------------------------------------+
|                                                                    |
|  +--------------------+     +--------------------+                 |
|  | GL-001 Thermal     |<--->| GL-002 Boiler      |                 |
|  | Command Center     |     | Optimizer          |                 |
|  | (Orchestrator)     |     | (ASME PTC 4.1)     |                 |
|  +--------------------+     +--------------------+                 |
|           |                          |                             |
|           v                          v                             |
|  +--------------------+     +--------------------+                 |
|  | GL-003 Steam       |     | GL-005 Combustion  |                 |
|  | Distribution       |     | Diagnostics        |                 |
|  +--------------------+     +--------------------+                 |
|           |                          |                             |
|           v                          v                             |
|  +--------------------+     +--------------------+                 |
|  | GL-010 Emissions   |     | GL-013 Predictive  |                 |
|  | Guardian           |     | Maintenance        |                 |
|  +--------------------+     +--------------------+                 |
|                                                                    |
+------------------------------------------------------------------+
|                     Safety Integrity Layer (SIL-2/SIL-3)          |
+------------------------------------------------------------------+
```

### 1.3 Key Components

| Component | Description | Responsibility |
|-----------|-------------|----------------|
| **GL-001 Thermal Command** | Central orchestrator | Coordinates all agents, manages workflows, API gateway |
| **GL-002 Boiler Optimizer** | Boiler efficiency agent | ASME PTC 4.1 calculations, combustion tuning |
| **GL-003 Steam Distribution** | Steam system agent | Steam header optimization, condensate return |
| **GL-005 Combustion Diagnostics** | Combustion analysis | Air-fuel ratio, flame stability |
| **GL-010 Emissions Guardian** | Emissions monitoring | EPA Method 19 calculations, permit compliance |
| **GL-011 Fuel Optimization** | Fuel management | Fuel blending, cost optimization |
| **GL-013 Predictive Maintenance** | Maintenance prediction | Vibration analysis, failure prediction |

### 1.4 Safety Design Principles

The platform is designed with the following safety principles:

1. **Fail-Safe Design**: All agents fail to a safe state on error
2. **Safety Integrity Levels**: SIL-2 minimum for all agents, SIL-3 for safety-critical functions
3. **Emergency Shutdown Integration**: Direct interface with plant ESD system
4. **Watchdog Timers**: All agents monitored with 5-second timeout
5. **Zero Hallucination**: All calculations use deterministic algorithms, no AI/ML in critical path

---

## 2. Login and Authentication

### 2.1 Accessing the Operator Console

**Step 1:** Navigate to the GreenLang Process Heat console:
- URL: `https://processheat.greenlang.io` (cloud) or `https://[local-server]/processheat` (on-premises)

**Step 2:** Enter your credentials:

```
+-------------------------------------------+
|         GreenLang Process Heat            |
|                                           |
|  Username: [operator@plant.com         ]  |
|  Password: [************************  ]   |
|                                           |
|  [ ] Remember this device (7 days)        |
|                                           |
|         [    Sign In    ]                 |
|                                           |
|  Forgot password? | Request access        |
+-------------------------------------------+
```

[Screenshot placeholder: Login screen]

**Step 3:** Complete multi-factor authentication (MFA):
- Enter the 6-digit code from your authenticator app
- Or approve the push notification on your registered device

### 2.2 Role-Based Access

| Role | Dashboard Access | Agent Control | Overrides | Configuration |
|------|-----------------|---------------|-----------|---------------|
| Viewer | Read-only | None | None | None |
| Operator | Full | Start/Stop | Temporary | None |
| Shift Supervisor | Full | Full | All | Limited |
| Engineer | Full | Full | All | Full |
| Administrator | Full | Full | All | Full |

### 2.3 Session Management

- Session timeout: 8 hours (configurable)
- Maximum concurrent sessions: 3 per user
- Automatic logout after 30 minutes of inactivity
- Session can be extended from the user menu

### 2.4 Password Requirements

- Minimum 12 characters
- Must include uppercase, lowercase, number, and special character
- Cannot reuse last 12 passwords
- Expires every 90 days
- Account locks after 5 failed attempts (15-minute lockout)

---

## 3. Dashboard Navigation

### 3.1 Main Dashboard Layout

```
+------------------------------------------------------------------+
|  [Logo] GreenLang Process Heat    [Alarms: 2] [User: J.Smith v]  |
+------------------------------------------------------------------+
|  Home | Agents | Alarms | Trends | Reports | Settings            |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  | SYSTEM STATUS    |  | EFFICIENCY       |  | EMISSIONS        |  |
|  | [RUNNING]        |  | 87.3%            |  | COMPLIANT        |  |
|  | Agents: 18/18    |  | Target: 88.0%    |  | CO2: 1,245 lb/hr |  |
|  | Uptime: 45d 12h  |  | Trend: +0.2%     |  | NOx: 0.12 lb/hr  |  |
|  +------------------+  +------------------+  +------------------+  |
|                                                                    |
|  +---------------------------------------------------------------+ |
|  |                    BOILER OVERVIEW                             | |
|  | +---------------+ +---------------+ +---------------+          | |
|  | | Boiler B-001  | | Boiler B-002  | | Boiler B-003  |          | |
|  | | [ONLINE]      | | [ONLINE]      | | [STANDBY]     |          | |
|  | | Load: 85%     | | Load: 72%     | | Load: 0%      |          | |
|  | | Eff: 87.1%    | | Eff: 86.9%    | | Eff: --       |          | |
|  | | O2: 2.8%      | | O2: 3.1%      | | O2: --        |          | |
|  | +---------------+ +---------------+ +---------------+          | |
|  +---------------------------------------------------------------+ |
|                                                                    |
|  +---------------------------+  +--------------------------------+ |
|  | ACTIVE RECOMMENDATIONS    |  | RECENT ALARMS                  | |
|  | 1. Reduce B-001 O2 to     |  | [!] High O2 B-002 - 14:32     | |
|  |    2.5% (+0.3% eff)       |  | [!] Low econ eff - 13:45      | |
|  | 2. Clean economizer       |  | [i] Shift start - 12:00       | |
|  |    B-002 (schedule)       |  |                                | |
|  +---------------------------+  +--------------------------------+ |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Main dashboard]

### 3.2 Navigation Menu

| Menu Item | Description |
|-----------|-------------|
| **Home** | Main dashboard with system overview |
| **Agents** | Individual agent status and control |
| **Alarms** | Active and historical alarms |
| **Trends** | Real-time and historical data trends |
| **Reports** | Generate and view reports |
| **Settings** | User preferences and system configuration |

### 3.3 Widget Types

**Status Widget**: Shows current state (Running/Stopped/Error)

**KPI Widget**: Displays key performance indicators with targets

**Trend Widget**: Mini chart showing recent values (1 hour default)

**Alarm Widget**: List of recent/active alarms with severity

**Recommendation Widget**: AI-generated optimization suggestions

### 3.4 Customizing the Dashboard

1. Click the gear icon in the top-right corner of the dashboard
2. Select "Edit Layout"
3. Drag and drop widgets to reposition
4. Click "+" to add new widgets
5. Right-click a widget to configure or remove
6. Click "Save Layout" when finished

---

## 4. Agent Monitoring and Status

### 4.1 Agent Status Screen

Access: Navigate to **Agents** from the main menu

```
+------------------------------------------------------------------+
|  AGENT STATUS                                    [Refresh] [Filter]|
+------------------------------------------------------------------+
|  Agent ID   | Name              | Status  | Health | Last Update  |
+------------------------------------------------------------------+
|  GL-001     | Thermal Command   | RUNNING | 100%   | 2s ago       |
|  GL-002-B001| Boiler Opt B-001  | RUNNING | 98%    | 5s ago       |
|  GL-002-B002| Boiler Opt B-002  | RUNNING | 95%    | 3s ago       |
|  GL-002-B003| Boiler Opt B-003  | STANDBY | 100%   | 1m ago       |
|  GL-003     | Steam Distribution| RUNNING | 97%    | 4s ago       |
|  GL-005     | Combustion Diag   | RUNNING | 100%   | 2s ago       |
|  GL-010     | Emissions Guardian| RUNNING | 99%    | 1s ago       |
|  GL-011     | Fuel Optimization | RUNNING | 96%    | 6s ago       |
|  GL-013     | Pred. Maintenance | RUNNING | 100%   | 10s ago      |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Agent status list]

### 4.2 Agent States

| State | Icon | Description |
|-------|------|-------------|
| **INITIALIZING** | Yellow spinner | Agent starting up |
| **READY** | Blue circle | Agent ready but not processing |
| **RUNNING** | Green circle | Agent actively processing |
| **PROCESSING** | Green spinner | Agent handling a task |
| **WAITING** | Blue pause | Agent waiting for input |
| **ERROR** | Red X | Agent in error state |
| **SHUTDOWN** | Gray circle | Agent stopped gracefully |
| **EMERGENCY_STOP** | Red octagon | Agent in ESD state |
| **STANDBY** | Yellow circle | Agent in standby mode |

### 4.3 Viewing Agent Details

Click on any agent row to view detailed information:

```
+------------------------------------------------------------------+
|  GL-002-B001 - Boiler Optimizer B-001                    [X Close]|
+------------------------------------------------------------------+
|  Status: RUNNING               Version: 1.0.0                     |
|  Safety Level: SIL-2           Uptime: 45d 12h 34m                |
+------------------------------------------------------------------+
|  CURRENT METRICS                                                  |
|  +------------------------+  +------------------------+           |
|  | Net Efficiency         |  | Combustion Efficiency  |           |
|  | 87.1%                  |  | 89.2%                  |           |
|  | Target: 88.0%          |  | Target: 90.0%          |           |
|  +------------------------+  +------------------------+           |
|                                                                    |
|  RECENT CALCULATIONS                                              |
|  +--------------------------------------------------------------+ |
|  | Time     | Type              | Result    | Provenance Hash   | |
|  +--------------------------------------------------------------+ |
|  | 14:32:05 | Efficiency Calc   | 87.1%     | a3f2c9...         | |
|  | 14:32:00 | Combustion Anal   | 89.2%     | b7e1d4...         | |
|  | 14:31:55 | Economizer Anal   | 78% eff   | c4a8f2...         | |
|  +--------------------------------------------------------------+ |
|                                                                    |
|  ACTIVE RECOMMENDATIONS                                           |
|  [!] Reduce O2 setpoint from 2.8% to 2.5% for +0.3% efficiency   |
|                                                                    |
|  [View History] [View Trends] [Restart Agent] [View Audit Log]    |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Agent detail view]

### 4.4 Agent Health Indicators

| Health % | Status | Description |
|----------|--------|-------------|
| 95-100% | Healthy | All subsystems operating normally |
| 80-94% | Degraded | Minor issues, monitoring recommended |
| 50-79% | Warning | Performance issues, investigation needed |
| 0-49% | Critical | Significant problems, immediate action required |

Health is calculated based on:
- Communication latency
- Processing time
- Error rate
- Memory/CPU usage
- Heartbeat consistency

### 4.5 Starting and Stopping Agents

**To Stop an Agent:**
1. Navigate to the agent detail view
2. Click "Stop Agent" button (or right-click agent in list)
3. Confirm the action in the dialog
4. Agent will gracefully shut down after completing current task

**To Start an Agent:**
1. Navigate to the agent detail view
2. Click "Start Agent" button
3. Agent will run safety checks and initialize
4. Status changes to RUNNING when ready

**Note:** Stopping safety-critical agents requires Shift Supervisor authorization and will generate an audit record.

---

## 5. Alarm Acknowledgment and Response

### 5.1 Alarm Priority Levels

| Priority | Color | Response Time | Description |
|----------|-------|---------------|-------------|
| **Emergency** | Red, flashing | Immediate | Safety-critical, ESD may be required |
| **High** | Red | < 5 minutes | Significant issue, immediate operator action |
| **Medium** | Orange | < 15 minutes | Issue requiring attention |
| **Low** | Yellow | < 60 minutes | Informational, schedule action |
| **Info** | Blue | N/A | Status change, no action required |

### 5.2 Alarm List Screen

Access: Navigate to **Alarms** from the main menu

```
+------------------------------------------------------------------+
|  ACTIVE ALARMS (3)                      [Ack All] [Filter] [Sound]|
+------------------------------------------------------------------+
|  Pri | Time     | Source    | Description              | Actions |
+------------------------------------------------------------------+
|  [!] | 14:32:15 | GL-002    | High O2: 4.2% (limit 4%) | [Ack]   |
|  [!] | 14:28:03 | GL-010    | NOx trending high        | [Ack]   |
|  [i] | 14:25:00 | GL-013    | Vibration elevated B-002 | [Ack]   |
+------------------------------------------------------------------+
|                                                                    |
|  ALARM HISTORY (Last 24 hours)          [Export] [Search]         |
+------------------------------------------------------------------+
|  Pri | Time     | Source    | Description         | Status        |
+------------------------------------------------------------------+
|  [!] | 13:45:22 | GL-002    | Low econ efficiency | Ack by JSmith |
|  [i] | 12:00:00 | SYSTEM    | Shift change        | Auto-cleared  |
|  [!] | 11:32:15 | GL-010    | High CO: 180 ppm    | Cleared       |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Alarm list]

### 5.3 Acknowledging Alarms

**Single Alarm:**
1. Click the [Ack] button next to the alarm
2. Enter optional comment (required for Emergency alarms)
3. Click "Confirm"
4. Alarm status changes to "Acknowledged"

**Multiple Alarms:**
1. Check the boxes next to alarms to acknowledge
2. Click "Ack Selected" button
3. Enter comment
4. Confirm action

**Note:** Acknowledging an alarm does not clear it. The alarm remains active until the underlying condition is resolved.

### 5.4 Alarm Response Procedures

**Emergency Alarms:**
1. Assess the situation immediately
2. Notify Shift Supervisor
3. Implement emergency response if required
4. Document actions taken
5. Acknowledge alarm with detailed comment

**High Priority Alarms:**
1. Review alarm details and history
2. Check related trends and parameters
3. Take corrective action
4. Monitor for improvement
5. Acknowledge and document

**Standard Response Checklist:**
- [ ] Identify the alarm source and type
- [ ] Review alarm history (is this recurring?)
- [ ] Check related equipment status
- [ ] Identify root cause if possible
- [ ] Take appropriate action
- [ ] Document response in alarm comments
- [ ] Acknowledge alarm

### 5.5 Alarm Shelving

For known issues requiring extended time to resolve:

1. Right-click on the alarm
2. Select "Shelve Alarm"
3. Enter shelve duration (max 8 hours for operators)
4. Enter reason for shelving
5. Confirm action

**Shelving Rules:**
- Emergency alarms cannot be shelved
- Maximum shelve time: 8 hours (operators), 24 hours (supervisors)
- Shelved alarms appear in separate list
- Alarm automatically un-shelves when duration expires
- All shelving actions are logged for audit

---

## 6. Manual Overrides and Bypasses

### 6.1 Override Types

| Type | Duration | Authorization | Use Case |
|------|----------|---------------|----------|
| **Temporary Override** | 1-8 hours | Operator | Tuning, testing |
| **Extended Override** | 8-24 hours | Supervisor | Maintenance |
| **Permanent Override** | Until removed | Engineer | Design change |
| **Safety Bypass** | 4 hours max | Supervisor + Engineer | Emergency only |

### 6.2 Applying a Temporary Override

**Step 1:** Navigate to the agent or parameter to override

**Step 2:** Click "Override" button or right-click and select "Override Value"

**Step 3:** Complete the override form:

```
+------------------------------------------+
|  TEMPORARY OVERRIDE                       |
+------------------------------------------+
|  Parameter: Boiler B-001 O2 Setpoint     |
|  Current Value: 2.8%                     |
|  Agent Recommended: 2.5%                 |
|                                          |
|  Override Value: [3.0        ] %         |
|  Duration: [4 hours    v]                |
|  Reason: [Combustion tuning - test    ]  |
|                                          |
|  [ ] Notify Shift Supervisor             |
|                                          |
|         [Cancel]  [Apply Override]       |
+------------------------------------------+
```

[Screenshot placeholder: Override dialog]

**Step 4:** Review and confirm the override

**Step 5:** Monitor the effect of the override

### 6.3 Viewing Active Overrides

Access: **Settings > Active Overrides**

```
+------------------------------------------------------------------+
|  ACTIVE OVERRIDES                                    [Export]     |
+------------------------------------------------------------------+
|  Parameter          | Override | Expires   | Applied By | Reason |
+------------------------------------------------------------------+
|  B-001 O2 Setpoint  | 3.0%     | 18:32:00 | JSmith     | Tuning |
|  B-002 Load Limit   | 90%      | 20:00:00 | MJones     | Maint  |
+------------------------------------------------------------------+
```

### 6.4 Removing an Override

1. Navigate to **Settings > Active Overrides**
2. Click "Remove" next to the override
3. Confirm removal
4. System returns to automatic control

### 6.5 Safety Bypasses

**WARNING: Safety bypasses should only be used in emergency situations and require dual authorization.**

**Procedure:**
1. Obtain verbal authorization from Shift Supervisor
2. Navigate to **Settings > Safety Bypasses**
3. Select the interlock or safety function to bypass
4. Enter Supervisor authorization code
5. Enter reason and expected duration
6. Both Operator and Supervisor must enter their credentials
7. Bypass is logged to permanent audit record

**Safety Bypass Rules:**
- Maximum duration: 4 hours
- Must be renewed if additional time needed
- Automatic notification to plant management
- Requires documented permit-to-work
- Cannot bypass ESD functions

---

## 7. Shift Handover Procedures

### 7.1 Pre-Handover Checklist (Outgoing Shift)

Complete the following before shift change:

- [ ] Review and acknowledge all active alarms
- [ ] Document any abnormal conditions
- [ ] Record all active overrides and bypasses
- [ ] Complete shift log entries
- [ ] Summarize key events and actions taken
- [ ] Note any pending tasks or follow-up items
- [ ] Review agent status and health

### 7.2 Generating Shift Report

**Step 1:** Navigate to **Reports > Shift Report**

**Step 2:** The system automatically generates a report including:
- Active alarms summary
- Alarm response actions taken
- Override and bypass activity
- Efficiency summary (average, min, max)
- Emissions summary
- Agent status changes
- Key events timeline

**Step 3:** Review and add comments

**Step 4:** Click "Finalize Report"

```
+------------------------------------------------------------------+
|  SHIFT REPORT - Day Shift 2025-12-06                             |
+------------------------------------------------------------------+
|  Shift: 06:00 - 18:00        Operator: J. Smith                  |
|  Supervisor: M. Jones        Report ID: SR-2025-12-06-001        |
+------------------------------------------------------------------+
|                                                                   |
|  SUMMARY                                                          |
|  - All boilers operated normally                                  |
|  - B-002 economizer cleaning scheduled for next outage           |
|  - High O2 alarm on B-001 at 14:32, corrected                    |
|                                                                   |
|  ACTIVE ALARMS AT SHIFT END: 1                                   |
|  - GL-013: Elevated vibration B-002 pump (monitoring)            |
|                                                                   |
|  OVERRIDES IN EFFECT: 1                                          |
|  - B-001 O2 setpoint: 3.0% (expires 18:32)                       |
|                                                                   |
|  EFFICIENCY SUMMARY                                               |
|  - Average: 87.2%                                                 |
|  - Minimum: 85.8% (10:15 during load swing)                      |
|  - Maximum: 88.1% (14:45)                                        |
|                                                                   |
|  EMISSIONS                                                        |
|  - Status: COMPLIANT                                              |
|  - CO2: 12,450 lb (shift total)                                  |
|  - NOx: 1.2 lb (shift total)                                     |
|                                                                   |
|  NOTES FOR INCOMING SHIFT:                                        |
|  [Watch B-002 pump vibration, may need bearing check          ]   |
|                                                                   |
|  [Print Report] [Email Report] [Save and Close]                   |
+------------------------------------------------------------------+
```

[Screenshot placeholder: Shift report]

### 7.3 Handover Meeting Agenda

1. **System Status** (2 min)
   - Overall system health
   - Active agents status

2. **Active Alarms** (3 min)
   - Review each active alarm
   - Explain any shelved alarms

3. **Active Overrides/Bypasses** (2 min)
   - Status and expiration times
   - Reason for each override

4. **Key Events** (5 min)
   - Significant events during shift
   - Actions taken and outcomes

5. **Pending Tasks** (3 min)
   - Work in progress
   - Scheduled activities

6. **Concerns/Watch Items** (5 min)
   - Equipment to monitor closely
   - Anticipated issues

### 7.4 Accepting Shift Handover

**Incoming Shift Operator:**

1. Log in to the system
2. Navigate to **Reports > Accept Handover**
3. Review the shift report
4. Physically walk through the control room with outgoing operator
5. Ask questions about any unclear items
6. Sign the electronic handover log
7. Outgoing operator logs out

```
+------------------------------------------+
|  HANDOVER ACCEPTANCE                      |
+------------------------------------------+
|  Shift Report: SR-2025-12-06-001         |
|  Outgoing: J. Smith                      |
|  Incoming: K. Davis                      |
|                                          |
|  [ ] I have reviewed the shift report    |
|  [ ] I have reviewed all active alarms   |
|  [ ] I have reviewed active overrides    |
|  [ ] I have performed control room walk  |
|  [ ] I accept responsibility for shift   |
|                                          |
|  Comments: [                          ]   |
|                                          |
|         [Cancel]  [Accept Handover]      |
+------------------------------------------+
```

---

## 8. Emergency Procedures

### 8.1 Emergency Shutdown (ESD) Procedure

**WHEN TO INITIATE ESD:**
- Imminent safety hazard
- Major equipment failure
- Fire or explosion risk
- Multiple high-priority alarms indicating system failure
- Loss of critical instrumentation

**ESD PROCEDURE:**

**Step 1:** Press the physical ESD button (if available) OR

**Step 2:** In the GreenLang console:
1. Navigate to **Agents > Emergency Actions**
2. Click "EMERGENCY SHUTDOWN"
3. Confirm by entering your credentials
4. Enter reason for ESD

```
+------------------------------------------+
|  EMERGENCY SHUTDOWN CONFIRMATION          |
+------------------------------------------+
|                                          |
|     !!!  WARNING  !!!                    |
|                                          |
|  You are about to trigger an EMERGENCY   |
|  SHUTDOWN of all process heat systems.   |
|                                          |
|  This will:                              |
|  - Stop all boilers                      |
|  - Close all fuel valves                 |
|  - Activate ESD relays                   |
|  - Generate emergency notification       |
|                                          |
|  Reason: [                           ]    |
|                                          |
|  Username: [                         ]    |
|  Password: [                         ]    |
|                                          |
|  [Cancel]  [CONFIRM EMERGENCY SHUTDOWN]  |
+------------------------------------------+
```

**Step 3:** Notify Shift Supervisor immediately

**Step 4:** Follow plant emergency response procedures

**Step 5:** Document all actions in emergency log

### 8.2 ESD Recovery Procedure

**Prerequisites for Recovery:**
- Root cause identified and corrected
- All equipment inspected and safe
- Management authorization obtained
- Documented permit to restart

**Recovery Steps:**

1. Obtain authorization from Operations Manager
2. Complete pre-startup safety review
3. Navigate to **Agents > Emergency Actions**
4. Click "Reset Emergency Shutdown"
5. Enter authorization codes (requires Supervisor + Engineer)
6. Follow startup sequence:
   - Reset ESD relays
   - Start Thermal Command orchestrator
   - Start individual agents in sequence
   - Verify all agents healthy
   - Begin gradual equipment restart

### 8.3 Agent Failure Response

**If a single agent fails:**

1. Note the agent ID and error message
2. Check if critical operations are affected
3. Review agent logs for error details
4. Attempt agent restart:
   - Navigate to agent detail view
   - Click "Restart Agent"
5. If restart fails, escalate to Engineering

**If orchestrator (GL-001) fails:**

1. Immediately notify Shift Supervisor
2. Check if agents are still operating (they should be in standalone mode)
3. Verify safety systems are functional
4. Attempt orchestrator restart
5. If unsuccessful, prepare for manual operation

### 8.4 Communication Failure Response

**If network connection lost:**

1. Verify local displays are functioning
2. Agents will continue operating with last known setpoints
3. Contact IT support
4. Prepare for manual intervention if needed
5. Do not restart agents until connectivity restored

**If DCS communication lost:**

1. Agents will enter SAFE_STATE
2. Manual control may be required
3. Notify Control Systems Engineering
4. Follow manual operating procedures

### 8.5 Emergency Contact List

| Role | Name | Phone | Backup |
|------|------|-------|--------|
| Shift Supervisor | [Name] | [Phone] | [Backup Phone] |
| Operations Manager | [Name] | [Phone] | [Backup Phone] |
| Control Systems Engineer | [Name] | [Phone] | [Backup Phone] |
| Safety Manager | [Name] | [Phone] | [Backup Phone] |
| GreenLang Support | 24/7 Hotline | 1-800-XXX-XXXX | support@greenlang.io |

---

## Appendix A: Agent Reference Card

### Quick Reference: Process Heat Agents

| Agent ID | Name | Primary Function | Key Metrics |
|----------|------|------------------|-------------|
| GL-001 | Thermal Command | Orchestration | System status, agent health |
| GL-002 | Boiler Optimizer | Efficiency | Net efficiency %, losses |
| GL-003 | Steam Distribution | Steam headers | Header pressure, flow balance |
| GL-005 | Combustion Diagnostics | Combustion | O2, CO, excess air |
| GL-006 | Waste Heat Recovery | Heat recovery | Recovery rate, economizer duty |
| GL-007 | Furnace Monitor | Furnace operations | Temperature profile, draft |
| GL-008 | Steam Trap Monitor | Steam traps | Trap status, losses |
| GL-009 | Thermal Fluid | Thermal oil systems | Fluid condition, temperature |
| GL-010 | Emissions Guardian | Emissions | CO2, NOx, permit status |
| GL-011 | Fuel Optimization | Fuel management | Fuel cost, blend ratio |
| GL-013 | Predictive Maintenance | Maintenance | Failure probability, vibration |
| GL-014 | Heat Exchanger | Heat exchangers | Effectiveness, fouling |
| GL-015 | Insulation Analysis | Insulation | Heat loss, surface temp |
| GL-016 | Water Treatment | Boiler water | TDS, pH, conductivity |
| GL-017 | Condenser Optimization | Condensers | Vacuum, cleanliness |
| GL-018 | Combustion Control | Burner control | Firing rate, O2 trim |

---

## Appendix B: Alarm Priority Matrix

### Alarm Categorization

| Category | Emergency | High | Medium | Low |
|----------|-----------|------|--------|-----|
| **Safety** | ESD triggered, Fire detection | SIL function deviation | Interlock test due | Safety system healthy |
| **Efficiency** | - | Below guarantee | 2% below target | 1% below target |
| **Emissions** | Permit exceedance | Near limit | Trending up | Minor elevation |
| **Equipment** | Major failure | Failure warning | Performance degraded | Minor issue |
| **Communication** | Safety comm lost | Multiple agents lost | Single agent lost | Intermittent |

### Response Time Requirements

| Priority | Response | Acknowledge | Resolve |
|----------|----------|-------------|---------|
| Emergency | Immediate | Within 1 min | As required |
| High | < 2 min | < 5 min | < 1 hour |
| Medium | < 5 min | < 15 min | < 4 hours |
| Low | < 15 min | < 60 min | Next shift |
| Info | N/A | When convenient | N/A |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-06 | GL-TechWriter | Initial release |

**Approval:**

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Operations Manager | _____________ | _____________ | _____________ |
| Safety Manager | _____________ | _____________ | _____________ |
| Quality Assurance | _____________ | _____________ | _____________ |

---

*For additional support, contact GreenLang Support at support@greenlang.io or 1-800-XXX-XXXX*
