# GL-010 EMISSIONWATCH Incident Response Runbook

## Document Control

| Property | Value |
|----------|-------|
| Document ID | GL-010-RUNBOOK-IR-001 |
| Version | 1.0.0 |
| Last Updated | 2025-11-26 |
| Owner | GL-010 Operations Team |
| Classification | Internal |
| Review Cycle | Quarterly |

---

## Table of Contents

1. [Overview](#1-overview)
2. [Severity Definitions](#2-severity-definitions)
3. [Incident Response Framework](#3-incident-response-framework)
4. [Emissions Violation Incidents](#4-emissions-violation-incidents)
5. [CEMS Failure Incidents](#5-cems-failure-incidents)
6. [Reporting Failure Incidents](#6-reporting-failure-incidents)
7. [System Failure Incidents](#7-system-failure-incidents)
8. [Escalation Matrix](#8-escalation-matrix)
9. [Communication Templates](#9-communication-templates)
10. [Post-Incident Review](#10-post-incident-review)
11. [Regulatory Notification Requirements](#11-regulatory-notification-requirements)
12. [Appendices](#12-appendices)

---

## 1. Overview

### 1.1 Purpose

This Incident Response Runbook provides comprehensive procedures for detecting, responding to, mitigating, and recovering from incidents affecting the GL-010 EMISSIONWATCH (EmissionsComplianceAgent) system. The runbook ensures rapid, consistent, and compliant responses to emissions monitoring and compliance incidents.

### 1.2 Scope

This runbook covers:
- Emissions limit violations and exceedances
- Continuous Emissions Monitoring System (CEMS) failures
- Regulatory reporting failures and delays
- System and infrastructure failures
- Integration and connectivity issues

### 1.3 Audience

- On-Call Engineers
- Site Operations Teams
- Environmental Compliance Officers
- DevOps/SRE Teams
- Regulatory Affairs Personnel
- Legal Teams (for violation incidents)

### 1.4 Critical Contacts

| Role | Name | Phone | Email | Escalation Time |
|------|------|-------|-------|-----------------|
| Primary On-Call | Rotation Schedule | +1-555-0100 | oncall-gl010@greenlang.io | Immediate |
| Secondary On-Call | Rotation Schedule | +1-555-0101 | oncall-gl010-backup@greenlang.io | 15 minutes |
| Operations Manager | [Name] | +1-555-0102 | ops-manager@greenlang.io | 30 minutes |
| Environmental Compliance Lead | [Name] | +1-555-0103 | compliance-lead@greenlang.io | 30 minutes |
| VP Engineering | [Name] | +1-555-0104 | vp-eng@greenlang.io | 1 hour |
| Legal Counsel | [Name] | +1-555-0105 | legal@greenlang.io | As needed |
| Regulatory Affairs | [Name] | +1-555-0106 | regulatory@greenlang.io | 30 minutes |

### 1.5 System Overview

GL-010 EMISSIONWATCH monitors and ensures compliance with environmental regulations including:
- Clean Air Act (CAA) requirements
- National Emission Standards for Hazardous Air Pollutants (NESHAP)
- New Source Performance Standards (NSPS)
- EU Emissions Trading System (EU ETS)
- State Implementation Plans (SIPs)
- Local air quality permits

**Key Capabilities:**
- Real-time CEMS data collection and validation
- Emissions calculations (mass, concentration, rates)
- Regulatory limit comparison and exceedance detection
- Automated regulatory report generation
- Alert and notification management
- Historical data archival and retrieval

---

## 2. Severity Definitions

### 2.1 Severity Level Matrix

| Severity | Definition | Response Time | Resolution Target | Examples |
|----------|------------|---------------|-------------------|----------|
| SEV1 | Critical - Active regulatory violation or imminent compliance failure | 5 minutes | 1 hour | CEMS down >4 hours, reporting deadline missed, active emissions violation |
| SEV2 | High - Significant risk to compliance or data quality | 15 minutes | 4 hours | Emissions >95% of limit, CEMS data quality degraded, report submission failure |
| SEV3 | Medium - Elevated risk requiring attention | 1 hour | 8 hours | Emissions 80-95% of limit, minor CEMS issues, validation warnings |
| SEV4 | Low - Minor issues with minimal compliance impact | 4 hours | 24 hours | Performance degradation, non-critical alerts |

### 2.2 SEV1 (Critical) Criteria

An incident is classified as SEV1 when ANY of the following conditions exist:

**Emissions Violations:**
- Any pollutant concentration exceeds permit limit
- Opacity readings exceed permit limit for >6 consecutive minutes
- Mass emissions rate exceeds hourly/daily limit
- Excess emissions event lasting >60 minutes
- Multiple pollutant limits exceeded simultaneously

**CEMS System Failures:**
- Primary and backup analyzers both offline
- CEMS data acquisition failure >4 hours
- Data quality score drops below 75%
- Missing data exceeds regulatory threshold (25% of operating hour)
- Calibration drift exceeds quality assurance limits

**Reporting Failures:**
- Quarterly emissions report submission deadline missed
- Annual emissions inventory deadline missed
- EPA CEDRI submission failure on deadline day
- EU ETS verification deadline missed
- State agency submission failure on deadline day

**System Failures:**
- Complete GL-010 agent unavailability
- Database failure with potential data loss
- All notification systems failed during active violation

### 2.3 SEV2 (High) Criteria

An incident is classified as SEV2 when ANY of the following conditions exist:

**Emissions Concerns:**
- Any pollutant at >95% of permit limit (trending toward violation)
- Opacity at >90% of permit limit
- Emissions trending toward 3-hour/12-hour rolling average exceedance
- Startup or shutdown emissions approaching limits

**CEMS Issues:**
- Single analyzer failure (backup operational)
- Data quality score 75-90%
- Calibration drift warning threshold reached
- Flow monitor accuracy degraded
- Temperature/pressure compensation errors

**Reporting Issues:**
- Report submission failure >24 hours before deadline
- Report validation errors requiring correction
- Data export failures for external systems
- Electronic signature system unavailable

**System Issues:**
- Agent performance severely degraded
- Database query timeouts affecting operations
- Integration connector failures to critical systems
- Alert delivery delays >15 minutes

### 2.4 SEV3 (Medium) Criteria

**Emissions Concerns:**
- Any pollutant at 80-95% of permit limit
- Unusual emissions patterns detected
- Process upset conditions detected
- Emissions calculator discrepancy >5%

**CEMS Issues:**
- Minor analyzer drift within acceptable range
- Single data point validation failures
- Non-critical sensor warnings
- Scheduled maintenance approaching

**Reporting Issues:**
- Report generation taking longer than normal
- Non-critical validation warnings
- Format conversion issues
- Archive retrieval delays

**System Issues:**
- Elevated memory or CPU utilization
- Non-critical service degradation
- Log aggregation delays
- Dashboard refresh issues

### 2.5 SEV4 (Low) Criteria

**Emissions Concerns:**
- Emissions elevated but well within limits (<80%)
- Minor process variations detected
- Informational compliance alerts

**CEMS Issues:**
- Informational CEMS alerts
- Scheduled calibration reminders
- Non-critical diagnostic messages

**Reporting Issues:**
- Report template updates needed
- Non-critical formatting improvements
- Documentation updates required

**System Issues:**
- Minor performance variations
- Informational system alerts
- Non-critical update notifications
- Disk space warnings (>30% free)

---

## 3. Incident Response Framework

### 3.1 Incident Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           INCIDENT RESPONSE LIFECYCLE                                │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │DETECTION │───►│ TRIAGE   │───►│ RESPONSE │───►│ RECOVERY │───►│ REVIEW   │      │
│  │          │    │          │    │          │    │          │    │          │      │
│  │- Alerts  │    │- Severity│    │- Contain │    │- Restore │    │- RCA     │      │
│  │- Monitor │    │- Assign  │    │- Mitigate│    │- Validate│    │- Action  │      │
│  │- Report  │    │- Notify  │    │- Document│    │- Monitor │    │- Improve │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                                      │
│  Timeline:                                                                           │
│  SEV1: 5min      15min          30min-1hr       1-4hr          24-48hr             │
│  SEV2: 15min     30min          1-4hr           4-8hr          1 week              │
│  SEV3: 1hr       2hr            4-8hr           8-24hr         2 weeks             │
│  SEV4: 4hr       8hr            24hr            24-48hr        30 days             │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Incident Response Roles

| Role | Responsibility | Assigned To |
|------|----------------|-------------|
| Incident Commander (IC) | Overall incident coordination, decisions, communications | On-Call Lead |
| Technical Lead | Technical investigation and resolution | Senior Engineer |
| Communications Lead | Stakeholder updates, regulatory notifications | Compliance Officer |
| Scribe | Documentation, timeline tracking | Available Engineer |
| Subject Matter Expert | Domain expertise (CEMS, calculations, regulations) | As needed |

### 3.3 Incident Declaration

**When to Declare an Incident:**

1. Automated alert triggers incident criteria
2. Manual observation of anomaly meeting severity criteria
3. External report of compliance issue
4. Regulatory agency notification
5. Customer/facility report of problem

**Declaration Procedure:**

```bash
# Create incident ticket via CLI
greenlang incident create \
  --agent GL-010 \
  --severity SEV1 \
  --title "NOx Limit Exceedance at Facility XYZ" \
  --description "NOx concentration exceeded permit limit of 50 ppm" \
  --affected-facility "facility-xyz-001" \
  --affected-pollutant "NOx" \
  --current-value "62 ppm" \
  --limit-value "50 ppm"

# Or via Slack
/incident create GL-010 SEV1 "NOx Limit Exceedance at Facility XYZ"
```

### 3.4 Initial Response Checklist

**First 5 Minutes (SEV1):**

- [ ] Acknowledge the alert
- [ ] Verify the alert is not a false positive
- [ ] Assess actual vs. reported severity
- [ ] Start incident tracking document
- [ ] Page secondary on-call if needed
- [ ] Begin initial triage

**First 15 Minutes (SEV1):**

- [ ] Identify incident type and category
- [ ] Assign incident roles
- [ ] Create war room (Slack channel, video call)
- [ ] Notify key stakeholders per escalation matrix
- [ ] Begin investigation
- [ ] Document initial findings

**First 30 Minutes (SEV1):**

- [ ] Identify root cause or probable cause
- [ ] Implement containment measures
- [ ] Assess regulatory notification requirements
- [ ] Prepare regulatory notification if required
- [ ] Update incident status
- [ ] Communicate to stakeholders

### 3.5 War Room Procedures

**War Room Setup:**

```
Slack Channel: #incident-gl010-{date}-{id}
Video Call: https://meet.greenlang.io/incident-gl010-{id}
Document: https://docs.greenlang.io/incidents/{id}
Dashboard: https://monitor.greenlang.io/incidents/{id}
```

**War Room Etiquette:**
- Keep audio on mute when not speaking
- Use Slack for non-urgent communications
- Incident Commander controls discussion flow
- Scribe maintains timeline in shared document
- All actions logged with timestamps

---

## 4. Emissions Violation Incidents

### 4.1 INC-001: NOx Limit Exceedance

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Emissions Violation |
| Typical Severity | SEV1 |
| Regulatory Impact | Immediate notification required |
| Common Causes | Combustion anomaly, SCR malfunction, excess air issues |

#### Detection

**Automated Detection:**
- CEMS NOx analyzer reading > permit limit
- 1-hour rolling average calculation > limit
- Rate of change indicator shows rapid increase
- Trend analysis predicts limit exceedance

**Alert Example:**
```json
{
  "alert_id": "GL010-NOX-001-20251126T143022Z",
  "severity": "critical",
  "type": "emissions_violation",
  "pollutant": "NOx",
  "facility_id": "facility-xyz-001",
  "unit_id": "boiler-001",
  "current_value": 62.3,
  "limit_value": 50.0,
  "unit": "ppm",
  "averaging_period": "1-hour",
  "duration_exceeded": "PT15M",
  "timestamp": "2025-11-26T14:30:22Z"
}
```

**Manual Detection Indicators:**
- Visible stack emissions changes
- Process operator reports
- Control room alarm activation
- Regulatory inspection finding

#### Initial Response (First 5 Minutes)

**Step 1: Acknowledge and Verify**

```bash
# Acknowledge the alert
greenlang alert ack GL010-NOX-001-20251126T143022Z \
  --responder "John Smith" \
  --notes "Investigating NOx exceedance"

# Verify CEMS reading is valid
greenlang cems status --facility facility-xyz-001 --unit boiler-001 --pollutant NOx

# Check for recent calibration or maintenance
greenlang cems calibration-log --facility facility-xyz-001 --last 24h
```

**Step 2: Quick Validation Checks**

```bash
# Compare with backup analyzer if available
greenlang cems compare-analyzers \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --pollutant NOx \
  --time-range "2025-11-26T14:00:00Z/2025-11-26T14:30:00Z"

# Check data quality indicators
greenlang cems data-quality \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --time-range "last-1h"

# Review process conditions
greenlang process-data \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --parameters "load,fuel_flow,air_flow,o2" \
  --time-range "last-1h"
```

**Step 3: Assess Severity and Duration**

```bash
# Calculate exceedance duration
greenlang emissions exceedance-duration \
  --facility facility-xyz-001 \
  --pollutant NOx \
  --limit 50

# Check regulatory averaging period status
greenlang compliance averaging-period-status \
  --facility facility-xyz-001 \
  --pollutant NOx \
  --periods "1h,3h,24h"
```

#### Investigation Steps

**Step 4: Root Cause Analysis**

Potential Root Causes for NOx Exceedance:

| Cause Category | Specific Causes | Investigation Method |
|----------------|-----------------|----------------------|
| Combustion Issues | High flame temperature | Check furnace temperature trends |
| | Excess air ratio incorrect | Review O2 readings |
| | Fuel quality changes | Check fuel analysis data |
| | Burner malfunction | Review burner diagnostics |
| NOx Control Issues | SCR catalyst degradation | Check ammonia slip, temp differential |
| | SNCR injection failure | Review reagent flow rates |
| | Low NOx burner issues | Check burner configuration |
| Process Issues | Load ramp too fast | Review load change rate |
| | Startup/shutdown | Check operational mode |
| | Unit trip/upset | Review alarm history |
| Measurement Issues | Analyzer drift | Check calibration data |
| | Interference | Review spectral data |
| | Sample system issue | Check sample flow, temperature |

```bash
# Check SCR system status
greenlang scr status --facility facility-xyz-001 --unit boiler-001

# Review ammonia injection
greenlang scr ammonia-flow \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --time-range "last-4h"

# Check catalyst temperature
greenlang scr catalyst-temp \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --time-range "last-4h"

# Review combustion parameters
greenlang combustion analysis \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --time-range "last-4h"
```

**Step 5: Data Collection**

```bash
# Export all relevant data for analysis
greenlang data export \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --parameters "all-emissions,all-process,all-cems" \
  --time-range "2025-11-26T12:00:00Z/2025-11-26T15:00:00Z" \
  --format csv \
  --output /data/incidents/GL010-NOX-001/

# Generate incident data package
greenlang incident data-package \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --include-cems \
  --include-process \
  --include-logs \
  --include-calibration
```

#### Mitigation Actions

**Immediate Mitigation Options:**

| Action | Description | Timeline | Responsible |
|--------|-------------|----------|-------------|
| Increase ammonia injection | Boost SCR/SNCR reagent flow | Immediate | Control Room |
| Reduce load | Lower combustion intensity | 5-15 min | Operations |
| Adjust air/fuel ratio | Optimize combustion | 5-15 min | Control Room |
| Switch fuel | Use lower-NOx fuel if available | 15-30 min | Operations |
| Manual control override | Bypass automatic controls | Immediate | Control Room |

**Mitigation Procedure:**

```bash
# Log mitigation action start
greenlang incident action-start \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --action "increase-ammonia-injection" \
  --target-value "120% of normal" \
  --operator "John Smith"

# Monitor response
greenlang watch emissions \
  --facility facility-xyz-001 \
  --pollutant NOx \
  --interval 1m \
  --duration 30m

# Log mitigation results
greenlang incident action-complete \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --action "increase-ammonia-injection" \
  --result "NOx reduced from 62 ppm to 45 ppm" \
  --effective true
```

#### Regulatory Notification Requirements

**EPA Requirements (40 CFR Part 60/63):**
- Notify within 2 business days of event discovery
- Submit written report within 30 days
- Include in quarterly excess emissions report

**State Requirements (varies by jurisdiction):**
- Check state SIP for notification timeline
- Typically 24-48 hours for telephone notification
- Written report within 10-30 days

```bash
# Check specific regulatory requirements
greenlang compliance notification-requirements \
  --facility facility-xyz-001 \
  --violation-type "nox-exceedance" \
  --jurisdiction "federal,state-ca"

# Prepare regulatory notification
greenlang regulatory notify \
  --facility facility-xyz-001 \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --agency "EPA-Region9,CARB" \
  --notification-type "excess-emissions" \
  --format "electronic"

# Generate excess emissions report
greenlang report generate excess-emissions \
  --facility facility-xyz-001 \
  --event-start "2025-11-26T14:15:00Z" \
  --event-end "2025-11-26T15:30:00Z" \
  --pollutant "NOx" \
  --cause "SCR ammonia injection pump failure" \
  --corrective-action "Repaired pump, increased maintenance frequency"
```

#### Recovery Procedures

**Step 6: Verify Resolution**

```bash
# Confirm emissions back in compliance
greenlang compliance status \
  --facility facility-xyz-001 \
  --pollutant NOx \
  --time-range "last-1h"

# Verify CEMS data quality
greenlang cems data-quality \
  --facility facility-xyz-001 \
  --unit boiler-001 \
  --time-range "last-1h"

# Check all averaging periods
greenlang emissions averaging-periods \
  --facility facility-xyz-001 \
  --pollutant NOx \
  --show-status
```

**Step 7: Document Resolution**

```bash
# Update incident record
greenlang incident update \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --status "resolved" \
  --resolution "SCR ammonia pump repaired, NOx levels returned to compliance" \
  --resolution-time "2025-11-26T15:45:00Z"

# Create resolution report
greenlang incident report \
  --incident-id GL010-NOX-001-20251126T143022Z \
  --format pdf \
  --output /reports/incidents/GL010-NOX-001-resolution.pdf
```

#### Post-Incident Review

**Required Actions:**
- [ ] Schedule post-incident review within 5 business days
- [ ] Prepare incident timeline and data
- [ ] Identify root cause(s)
- [ ] Document lessons learned
- [ ] Create action items for prevention
- [ ] Update runbooks if needed
- [ ] File required regulatory reports

---

### 4.2 INC-002: SOx Limit Exceedance

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Emissions Violation |
| Typical Severity | SEV1 |
| Regulatory Impact | Immediate notification required |
| Common Causes | High sulfur fuel, scrubber malfunction, bypass stack operation |

#### Detection

**Automated Detection:**
```json
{
  "alert_id": "GL010-SOX-002-20251126T091500Z",
  "severity": "critical",
  "type": "emissions_violation",
  "pollutant": "SO2",
  "facility_id": "facility-abc-002",
  "unit_id": "power-unit-002",
  "current_value": 185.7,
  "limit_value": 150.0,
  "unit": "ppm",
  "averaging_period": "30-day rolling",
  "timestamp": "2025-11-26T09:15:00Z"
}
```

**Manual Detection Indicators:**
- Scrubber differential pressure changes
- pH levels in scrubber liquor abnormal
- Limestone/reagent consumption changes
- Stack opacity changes

#### Initial Response (First 5 Minutes)

```bash
# Acknowledge alert
greenlang alert ack GL010-SOX-002-20251126T091500Z

# Verify CEMS SO2 reading
greenlang cems status --facility facility-abc-002 --pollutant SO2

# Check scrubber status
greenlang fgd status --facility facility-abc-002 --unit scrubber-001

# Review fuel sulfur content
greenlang fuel sulfur-content \
  --facility facility-abc-002 \
  --time-range "last-7d"
```

#### Investigation Steps

**Potential Root Causes:**

| Cause Category | Specific Causes | Investigation |
|----------------|-----------------|---------------|
| Fuel Issues | High sulfur coal/oil delivered | Check fuel receiving records |
| | Fuel blend changed | Review fuel management system |
| | Fuel specification violation | Check fuel certificates |
| Scrubber Issues | Low reagent inventory | Check silo/tank levels |
| | Reagent feed system failure | Check feed rate, pump status |
| | Absorber efficiency degraded | Check L/G ratio, pH |
| | Mist eliminator plugged | Check pressure differential |
| | Bypass damper open | Check damper position |
| Process Issues | High load operation | Review load profile |
| | Scrubber bypassed | Check bypass records |
| | Multiple units on single scrubber | Check unit configuration |
| Measurement Issues | Analyzer interference | Check for moisture, particulate |
| | Sample line issue | Check sample flow, conditioning |
| | Calibration drift | Review calibration records |

```bash
# Comprehensive FGD diagnostic
greenlang fgd diagnostic \
  --facility facility-abc-002 \
  --parameters "all" \
  --time-range "last-24h"

# Check reagent system
greenlang fgd reagent-status \
  --facility facility-abc-002

# Review SO2 removal efficiency
greenlang fgd efficiency \
  --facility facility-abc-002 \
  --time-range "last-48h"

# Compare inlet vs outlet SO2
greenlang fgd so2-removal \
  --facility facility-abc-002 \
  --show-trend
```

#### Mitigation Actions

| Action | Description | Timeline |
|--------|-------------|----------|
| Increase reagent feed | Boost limestone/lime slurry rate | Immediate |
| Adjust L/G ratio | Increase liquid-to-gas ratio | 5-10 min |
| Reduce load | Lower unit output | 5-15 min |
| Switch fuel | Use lower sulfur fuel | 1-4 hours |
| Activate spare scrubber | Bring backup online | 1-2 hours |

```bash
# Log mitigation start
greenlang incident action-start \
  --incident-id GL010-SOX-002-20251126T091500Z \
  --action "increase-reagent-feed" \
  --target-value "125% of normal"

# Monitor scrubber response
greenlang watch fgd \
  --facility facility-abc-002 \
  --parameters "so2-outlet,efficiency,reagent-rate" \
  --interval 5m \
  --duration 2h
```

#### Regulatory Notification

```bash
# Generate SO2 exceedance notification
greenlang regulatory notify \
  --facility facility-abc-002 \
  --incident-id GL010-SOX-002-20251126T091500Z \
  --violation-type "so2-exceedance" \
  --agency "EPA,State-DEQ"

# Check acid rain program requirements
greenlang compliance acid-rain-status \
  --facility facility-abc-002 \
  --year 2025
```

---

### 4.3 INC-003: CO2 Reporting Threshold Exceeded

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Reporting Threshold Alert |
| Typical Severity | SEV2 (no immediate violation, but reporting triggered) |
| Regulatory Impact | Triggers additional reporting requirements |
| Common Causes | Production increase, new emission sources, calculation update |

#### Detection

```json
{
  "alert_id": "GL010-CO2-003-20251126T000000Z",
  "severity": "high",
  "type": "reporting_threshold",
  "parameter": "CO2_annual",
  "facility_id": "facility-def-003",
  "current_value": 26150.5,
  "threshold_value": 25000.0,
  "unit": "metric_tons_CO2e",
  "period": "annual_ytd",
  "timestamp": "2025-11-26T00:00:00Z",
  "message": "Annual CO2e emissions have exceeded 25,000 MT threshold - EPA GHG reporting triggered"
}
```

#### Initial Response

```bash
# Verify CO2 calculation
greenlang emissions verify-co2 \
  --facility facility-def-003 \
  --year 2025 \
  --show-calculation

# Check reporting requirements triggered
greenlang compliance reporting-requirements \
  --facility facility-def-003 \
  --threshold-exceeded "25000-mt-co2e"

# Review emission sources
greenlang emissions sources \
  --facility facility-def-003 \
  --year 2025 \
  --by-source
```

#### Investigation Steps

**Determine Threshold Exceedance Cause:**

```bash
# Compare to previous year
greenlang emissions compare \
  --facility facility-def-003 \
  --parameter CO2e \
  --year1 2024 \
  --year2 2025 \
  --by-month

# Identify contributing sources
greenlang emissions top-sources \
  --facility facility-def-003 \
  --parameter CO2e \
  --year 2025 \
  --limit 10

# Check for new sources
greenlang sources changes \
  --facility facility-def-003 \
  --year 2025
```

#### Actions Required

**If Threshold Legitimately Exceeded:**

```bash
# Update facility reporting status
greenlang facility update \
  --facility-id facility-def-003 \
  --ghg-reporting-required true \
  --threshold-exceeded-date "2025-11-26"

# Register for EPA GHG reporting
greenlang regulatory ghg-program \
  --facility facility-def-003 \
  --action "register" \
  --program "EPA-GHGRP"

# Set up required monitoring
greenlang monitoring configure \
  --facility facility-def-003 \
  --program "GHGRP" \
  --methodology "40-CFR-98-Subpart-C"
```

**If Calculation Error Found:**

```bash
# Correct emissions calculation
greenlang emissions recalculate \
  --facility facility-def-003 \
  --year 2025 \
  --correction-reason "fuel-carbon-content-update"

# Document correction
greenlang incident update \
  --incident-id GL010-CO2-003-20251126T000000Z \
  --status "resolved" \
  --resolution "Calculation error identified and corrected, actual CO2e = 23,500 MT"
```

---

### 4.4 INC-004: PM/Opacity Violation

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Emissions Violation |
| Typical Severity | SEV1 |
| Regulatory Impact | Visible emission standards, community complaints |
| Common Causes | ESP failure, baghouse leak, combustion upset, soot blowing |

#### Detection

```json
{
  "alert_id": "GL010-PM-004-20251126T103045Z",
  "severity": "critical",
  "type": "opacity_violation",
  "pollutant": "PM/Opacity",
  "facility_id": "facility-ghi-004",
  "unit_id": "kiln-001",
  "current_value": 25.5,
  "limit_value": 20.0,
  "unit": "percent_opacity",
  "averaging_period": "6-minute",
  "consecutive_violations": 3,
  "timestamp": "2025-11-26T10:30:45Z"
}
```

#### Initial Response

```bash
# Acknowledge alert
greenlang alert ack GL010-PM-004-20251126T103045Z

# Check COM (Continuous Opacity Monitor) status
greenlang cems com-status \
  --facility facility-ghi-004 \
  --unit kiln-001

# Check particulate control device status
greenlang pcd status \
  --facility facility-ghi-004 \
  --device-type "ESP,baghouse"

# Review process conditions
greenlang process-data \
  --facility facility-ghi-004 \
  --unit kiln-001 \
  --parameters "temp,feed_rate,draft" \
  --time-range "last-2h"
```

#### Investigation Steps

**Potential Root Causes:**

| Device Type | Potential Causes |
|-------------|------------------|
| ESP (Electrostatic Precipitator) | Power supply failure, rapper malfunction, high resistivity ash, electrode damage |
| Baghouse | Bag leak/failure, pulse cleaning issues, high temperature damage, bypassed compartment |
| Wet Scrubber | Low liquor flow, mist eliminator plugged, nozzle blockage |
| Combustion | Incomplete combustion, soot blowing, startup/shutdown |

```bash
# ESP diagnostic
greenlang esp diagnostic \
  --facility facility-ghi-004 \
  --parameters "power,current,voltage,rapping" \
  --fields "all"

# Baghouse diagnostic
greenlang baghouse diagnostic \
  --facility facility-ghi-004 \
  --parameters "differential_pressure,pulse,leak_detection"

# Check for soot blowing
greenlang soot-blower status \
  --facility facility-ghi-004 \
  --time-range "last-2h"
```

#### Mitigation Actions

**ESP Mitigation:**
```bash
# Increase ESP power
greenlang esp action \
  --facility facility-ghi-004 \
  --action "increase-power" \
  --target-value "max"

# Activate backup field
greenlang esp action \
  --facility facility-ghi-004 \
  --action "activate-backup-field" \
  --field-id "field-3"
```

**Baghouse Mitigation:**
```bash
# Increase cleaning frequency
greenlang baghouse action \
  --facility facility-ghi-004 \
  --action "increase-pulse-frequency" \
  --target-value "2x-normal"

# Isolate leaking compartment
greenlang baghouse action \
  --facility facility-ghi-004 \
  --action "isolate-compartment" \
  --compartment-id "comp-4"
```

#### Regulatory Notification

```bash
# Check visible emission notification requirements
greenlang compliance visible-emissions \
  --facility facility-ghi-004 \
  --check-notification-required

# Generate opacity exceedance report
greenlang report generate opacity-exceedance \
  --facility facility-ghi-004 \
  --event-id GL010-PM-004-20251126T103045Z \
  --include "6-minute-averages,com-data,process-data"
```

---

### 4.5 INC-005: Multi-Pollutant Exceedance

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Multiple Emissions Violations |
| Typical Severity | SEV1 (Critical) |
| Regulatory Impact | Multiple regulatory programs affected |
| Common Causes | Major process upset, control device failure, emergency bypass |

#### Detection

```json
{
  "alert_id": "GL010-MULTI-005-20251126T141500Z",
  "severity": "critical",
  "type": "multi_pollutant_exceedance",
  "facility_id": "facility-jkl-005",
  "unit_id": "incinerator-001",
  "violations": [
    {
      "pollutant": "NOx",
      "current_value": 185.3,
      "limit_value": 150.0,
      "unit": "ppm"
    },
    {
      "pollutant": "CO",
      "current_value": 125.6,
      "limit_value": 100.0,
      "unit": "ppm"
    },
    {
      "pollutant": "SO2",
      "current_value": 32.1,
      "limit_value": 25.0,
      "unit": "ppm"
    }
  ],
  "timestamp": "2025-11-26T14:15:00Z",
  "message": "Multiple pollutant limits exceeded simultaneously - potential major process upset"
}
```

#### Initial Response (First 5 Minutes)

**This is a Critical Incident - Escalate Immediately**

```bash
# Escalate to all stakeholders
greenlang escalate \
  --incident-id GL010-MULTI-005-20251126T141500Z \
  --severity SEV1 \
  --notify "oncall,operations-manager,compliance-lead,vp-engineering"

# Emergency status check
greenlang emergency-status \
  --facility facility-jkl-005 \
  --unit incinerator-001

# All CEMS status
greenlang cems status-all \
  --facility facility-jkl-005 \
  --unit incinerator-001

# Process safety check
greenlang safety-check \
  --facility facility-jkl-005 \
  --unit incinerator-001
```

#### Investigation Steps

**Potential Root Causes for Multi-Pollutant Event:**

| Category | Causes |
|----------|--------|
| Major Process Upset | Temperature excursion, feed composition change, equipment malfunction |
| Control Device Failure | Multiple control devices failed, common utility loss |
| Emergency Situation | Fire, explosion, emergency shutdown, bypass stack activation |
| Measurement Issue | Common analyzer issue (dilution system), sample system failure |
| Operational Error | Incorrect setpoints, manual override, procedural violation |

```bash
# Comprehensive system diagnostic
greenlang diagnostic full \
  --facility facility-jkl-005 \
  --unit incinerator-001 \
  --include "process,cems,controls,safety"

# Check for common cause
greenlang correlation analysis \
  --facility facility-jkl-005 \
  --unit incinerator-001 \
  --parameters "all-emissions,all-process" \
  --time-range "last-4h"

# Review alarm history
greenlang alarms history \
  --facility facility-jkl-005 \
  --time-range "last-4h" \
  --severity "warning,critical"
```

#### Mitigation Actions

**Coordinated Response Required:**

```bash
# Create coordinated action plan
greenlang incident action-plan \
  --incident-id GL010-MULTI-005-20251126T141500Z \
  --actions "[
    {\"action\": \"reduce-load\", \"target\": \"50%\", \"owner\": \"operations\"},
    {\"action\": \"boost-all-controls\", \"target\": \"max\", \"owner\": \"control-room\"},
    {\"action\": \"verify-controls-operational\", \"owner\": \"maintenance\"},
    {\"action\": \"prepare-shutdown-if-needed\", \"owner\": \"operations\"}
  ]"

# Monitor all pollutants
greenlang watch emissions \
  --facility facility-jkl-005 \
  --pollutants "NOx,CO,SO2,PM" \
  --interval 1m \
  --alert-threshold-pct 90
```

#### Regulatory Notification

**Multiple Agency Notification May Be Required:**

```bash
# Check all notification requirements
greenlang compliance multi-pollutant-notification \
  --facility facility-jkl-005 \
  --pollutants "NOx,CO,SO2" \
  --jurisdiction "federal,state,local"

# Prepare coordinated notification
greenlang regulatory notify-batch \
  --facility facility-jkl-005 \
  --incident-id GL010-MULTI-005-20251126T141500Z \
  --agencies "[
    {\"agency\": \"EPA-Region\", \"type\": \"telephone\"},
    {\"agency\": \"State-DEQ\", \"type\": \"telephone\"},
    {\"agency\": \"Local-AQMD\", \"type\": \"email\"}
  ]"
```

---

## 5. CEMS Failure Incidents

### 5.1 INC-006: Primary Analyzer Failure

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | CEMS Equipment Failure |
| Typical Severity | SEV2 (if backup available), SEV1 (if no backup) |
| Regulatory Impact | Data substitution may be required |
| Common Causes | Component failure, power issue, calibration drift, environmental conditions |

#### Detection

```json
{
  "alert_id": "GL010-CEMS-006-20251126T082315Z",
  "severity": "high",
  "type": "cems_analyzer_failure",
  "analyzer_type": "primary",
  "pollutant": "NOx",
  "facility_id": "facility-mno-006",
  "unit_id": "turbine-001",
  "analyzer_id": "NOx-analyzer-001",
  "failure_mode": "no_response",
  "backup_status": "operational",
  "last_valid_reading": {
    "value": 42.3,
    "timestamp": "2025-11-26T08:22:45Z"
  },
  "timestamp": "2025-11-26T08:23:15Z"
}
```

#### Initial Response

```bash
# Acknowledge alert
greenlang alert ack GL010-CEMS-006-20251126T082315Z

# Check analyzer detailed status
greenlang cems analyzer-status \
  --facility facility-mno-006 \
  --analyzer-id NOx-analyzer-001 \
  --detail-level full

# Verify backup analyzer operational
greenlang cems analyzer-status \
  --facility facility-mno-006 \
  --analyzer-id NOx-analyzer-002-backup \
  --detail-level full

# Switch to backup if not already
greenlang cems switch-analyzer \
  --facility facility-mno-006 \
  --pollutant NOx \
  --from NOx-analyzer-001 \
  --to NOx-analyzer-002-backup \
  --reason "primary-failure"
```

#### Investigation Steps

**Potential Failure Modes:**

| Failure Mode | Symptoms | Investigation |
|--------------|----------|---------------|
| Power Failure | No response, display off | Check power supply, fuses, circuit |
| Communication Loss | No data, status timeout | Check cables, network, protocol |
| Optical Degradation | Drift, noise, span error | Check cell, windows, light source |
| Sample System | No flow, conditioning fail | Check pump, filters, dryers |
| Environmental | High temp, humidity | Check shelter HVAC, purge air |
| Component Failure | Error codes, fault status | Check diagnostics, error log |

```bash
# Full analyzer diagnostic
greenlang cems analyzer-diagnostic \
  --facility facility-mno-006 \
  --analyzer-id NOx-analyzer-001 \
  --tests "power,communication,optical,sample,environmental"

# Check error codes
greenlang cems error-codes \
  --analyzer-id NOx-analyzer-001 \
  --time-range "last-24h"

# Review maintenance history
greenlang cems maintenance-history \
  --analyzer-id NOx-analyzer-001 \
  --time-range "last-90d"
```

#### Mitigation Actions

```bash
# If backup operational, ensure data continuity
greenlang cems verify-data-flow \
  --facility facility-mno-006 \
  --pollutant NOx \
  --source backup

# Dispatch maintenance
greenlang maintenance create-ticket \
  --equipment NOx-analyzer-001 \
  --priority high \
  --issue "Analyzer not responding" \
  --reference-incident GL010-CEMS-006-20251126T082315Z

# Set up substitute data if needed
greenlang cems configure-substitute-data \
  --facility facility-mno-006 \
  --pollutant NOx \
  --method "backup-analyzer" \
  --start-time "2025-11-26T08:23:15Z"
```

#### Data Handling Requirements

**EPA 40 CFR Part 75 Requirements:**

```bash
# Check data substitution requirements
greenlang compliance substitute-data-requirements \
  --facility facility-mno-006 \
  --program "Part75" \
  --pollutant NOx \
  --missing-data-start "2025-11-26T08:23:15Z"

# Generate DAHS report
greenlang cems dahs-report \
  --facility facility-mno-006 \
  --date "2025-11-26" \
  --include-substitute-data-flags
```

---

### 5.2 INC-007: Backup Analyzer Failure

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | CEMS Equipment Failure |
| Typical Severity | SEV2 (primary operational), SEV1 (if primary also has issues) |
| Regulatory Impact | Redundancy compromised |
| Common Causes | Extended non-use, calibration drift, maintenance overdue |

#### Detection

```json
{
  "alert_id": "GL010-CEMS-007-20251126T143000Z",
  "severity": "high",
  "type": "cems_analyzer_failure",
  "analyzer_type": "backup",
  "pollutant": "SO2",
  "facility_id": "facility-pqr-007",
  "unit_id": "boiler-003",
  "analyzer_id": "SO2-analyzer-002-backup",
  "failure_mode": "calibration_failure",
  "primary_status": "operational",
  "timestamp": "2025-11-26T14:30:00Z"
}
```

#### Initial Response

```bash
# Acknowledge and assess
greenlang alert ack GL010-CEMS-007-20251126T143000Z

# Verify primary is stable
greenlang cems analyzer-status \
  --facility facility-pqr-007 \
  --analyzer-id SO2-analyzer-001-primary \
  --detail-level full

# Assess backup failure details
greenlang cems calibration-status \
  --analyzer-id SO2-analyzer-002-backup \
  --detail-level full
```

#### Mitigation Actions

```bash
# Schedule immediate backup repair
greenlang maintenance create-ticket \
  --equipment SO2-analyzer-002-backup \
  --priority high \
  --issue "Calibration failure during backup verification" \
  --sla "24h"

# Increase primary monitoring
greenlang monitoring adjust \
  --analyzer-id SO2-analyzer-001-primary \
  --health-check-interval "5m" \
  --alert-threshold "warning"

# Prepare contingency plan
greenlang cems contingency-plan \
  --facility facility-pqr-007 \
  --pollutant SO2 \
  --scenario "primary-failure-no-backup"
```

---

### 5.3 INC-008: Data Quality Below 90%

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | CEMS Data Quality Degradation |
| Typical Severity | SEV2 |
| Regulatory Impact | May affect data certification, quarterly reports |
| Common Causes | Missing data, calibration issues, interference, validation failures |

#### Detection

```json
{
  "alert_id": "GL010-DQ-008-20251126T060000Z",
  "severity": "high",
  "type": "data_quality_degradation",
  "facility_id": "facility-stu-008",
  "unit_id": "unit-001",
  "metric": "data_availability",
  "current_value": 87.3,
  "threshold_value": 90.0,
  "period": "calendar_quarter",
  "affected_pollutants": ["NOx", "SO2", "CO2"],
  "timestamp": "2025-11-26T06:00:00Z"
}
```

#### Initial Response

```bash
# Get detailed data quality breakdown
greenlang cems data-quality-report \
  --facility facility-stu-008 \
  --period "Q4-2025" \
  --detail-level full

# Identify data quality issues
greenlang cems data-quality-issues \
  --facility facility-stu-008 \
  --period "Q4-2025" \
  --group-by "cause"
```

#### Investigation Steps

**Data Quality Components:**

| Component | Target | Common Issues |
|-----------|--------|---------------|
| Data Availability | >90% | Missing data, communication gaps |
| Calibration Success | >95% | Failed calibrations, drift |
| QA Test Success | >100% of required | Failed RATAs, CGAs |
| Valid Data Hours | >90% | Validation flags, out-of-range |

```bash
# Analyze missing data periods
greenlang cems missing-data-analysis \
  --facility facility-stu-008 \
  --period "Q4-2025"

# Check calibration history
greenlang cems calibration-history \
  --facility facility-stu-008 \
  --period "Q4-2025" \
  --show-failures

# Review validation flags
greenlang cems validation-flags \
  --facility facility-stu-008 \
  --period "Q4-2025" \
  --group-by "flag-type"
```

#### Mitigation Actions

```bash
# Create data quality improvement plan
greenlang cems data-quality-improvement-plan \
  --facility facility-stu-008 \
  --target-availability 95.0 \
  --deadline "2025-12-31"

# Schedule additional maintenance
greenlang maintenance schedule \
  --facility facility-stu-008 \
  --type "preventive" \
  --equipment "all-cems" \
  --frequency "weekly-until-improved"

# Implement enhanced monitoring
greenlang monitoring enhance \
  --facility facility-stu-008 \
  --add-checks "communication,calibration,validation" \
  --alert-on-any-issue
```

---

### 5.4 INC-009: Missing Data Exceeds 25% of Hour

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | CEMS Data Gap |
| Typical Severity | SEV1 (regulatory threshold) |
| Regulatory Impact | Substitute data required, may affect emissions calculations |
| Common Causes | Analyzer downtime, communication failure, data system issue |

#### Detection

```json
{
  "alert_id": "GL010-MD-009-20251126T150000Z",
  "severity": "critical",
  "type": "missing_data_threshold",
  "facility_id": "facility-vwx-009",
  "unit_id": "boiler-001",
  "pollutant": "NOx",
  "missing_minutes": 18,
  "hour": "2025-11-26T14:00:00Z",
  "threshold_minutes": 15,
  "cause": "communication_failure",
  "timestamp": "2025-11-26T15:00:00Z"
}
```

#### Initial Response

```bash
# Acknowledge and investigate
greenlang alert ack GL010-MD-009-20251126T150000Z

# Check current CEMS status
greenlang cems status \
  --facility facility-vwx-009 \
  --unit boiler-001 \
  --pollutant NOx

# Identify missing data period
greenlang cems missing-data \
  --facility facility-vwx-009 \
  --hour "2025-11-26T14:00:00Z" \
  --detail
```

#### Mitigation Actions

```bash
# Apply substitute data procedure
greenlang cems substitute-data \
  --facility facility-vwx-009 \
  --pollutant NOx \
  --hour "2025-11-26T14:00:00Z" \
  --method "40-CFR-75-appendix-D"

# Document substitution
greenlang cems document-substitution \
  --facility facility-vwx-009 \
  --hour "2025-11-26T14:00:00Z" \
  --reason "Communication failure - 18 minutes missing data" \
  --substitute-value 65.3 \
  --method "90th-percentile-lookback"

# Fix communication issue
greenlang cems troubleshoot-communication \
  --facility facility-vwx-009 \
  --unit boiler-001
```

---

### 5.5 INC-010: Calibration Drift Detected

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | CEMS Calibration Issue |
| Typical Severity | SEV2 (within limits), SEV1 (exceeds limits) |
| Regulatory Impact | May invalidate data, require recalibration |
| Common Causes | Optical degradation, cell contamination, temperature effects, aging components |

#### Detection

```json
{
  "alert_id": "GL010-CAL-010-20251126T070000Z",
  "severity": "high",
  "type": "calibration_drift",
  "facility_id": "facility-yza-010",
  "unit_id": "turbine-002",
  "analyzer_id": "NOx-analyzer-003",
  "drift_type": "span",
  "measured_value": 95.2,
  "reference_value": 100.0,
  "drift_percentage": -4.8,
  "limit_percentage": 5.0,
  "status": "warning",
  "timestamp": "2025-11-26T07:00:00Z"
}
```

#### Initial Response

```bash
# Acknowledge alert
greenlang alert ack GL010-CAL-010-20251126T070000Z

# Get full calibration details
greenlang cems calibration-detail \
  --analyzer-id NOx-analyzer-003 \
  --date "2025-11-26"

# Review drift trend
greenlang cems drift-trend \
  --analyzer-id NOx-analyzer-003 \
  --period "last-30d"
```

#### Investigation Steps

```bash
# Analyze drift pattern
greenlang cems drift-analysis \
  --analyzer-id NOx-analyzer-003 \
  --include "environmental-correlation,usage-correlation"

# Check reference gas cylinder
greenlang cems reference-gas-status \
  --facility facility-yza-010 \
  --analyzer-id NOx-analyzer-003

# Review maintenance schedule
greenlang cems maintenance-due \
  --analyzer-id NOx-analyzer-003
```

#### Mitigation Actions

```bash
# Schedule calibration adjustment
greenlang cems calibration-adjust \
  --analyzer-id NOx-analyzer-003 \
  --type "span" \
  --schedule "immediate"

# If adjustment fails, schedule full calibration
greenlang cems schedule-calibration \
  --analyzer-id NOx-analyzer-003 \
  --type "full" \
  --include "zero,upscale,linearity"

# Data correction if needed
greenlang cems apply-drift-correction \
  --analyzer-id NOx-analyzer-003 \
  --period "since-last-valid-calibration" \
  --method "linear-interpolation"
```

---

## 6. Reporting Failure Incidents

### 6.1 INC-011: Quarterly Report Submission Failed

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Regulatory Report Failure |
| Typical Severity | SEV1 (if deadline imminent), SEV2 (if time remaining) |
| Regulatory Impact | Potential violation, penalties |
| Common Causes | System error, validation failure, connectivity issue, data gap |

#### Detection

```json
{
  "alert_id": "GL010-RPT-011-20251126T093000Z",
  "severity": "critical",
  "type": "report_submission_failure",
  "report_type": "quarterly_emissions",
  "facility_id": "facility-bcd-011",
  "reporting_period": "Q3-2025",
  "deadline": "2025-11-30T23:59:59Z",
  "days_remaining": 4,
  "failure_reason": "xml_validation_error",
  "error_details": "Schema validation failed at line 4523",
  "timestamp": "2025-11-26T09:30:00Z"
}
```

#### Initial Response (First 5 Minutes)

```bash
# Acknowledge and assess urgency
greenlang alert ack GL010-RPT-011-20251126T093000Z

# Check report status
greenlang report status \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --detail-level full

# View validation errors
greenlang report validation-errors \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --detail-level full
```

#### Investigation Steps

**Common Report Validation Errors:**

| Error Type | Description | Resolution |
|------------|-------------|------------|
| Schema Validation | XML doesn't match XSD | Fix XML structure |
| Data Validation | Values out of range | Correct data |
| Cross-Reference | Inconsistent facility/unit IDs | Update references |
| Calculation | Totals don't match detail | Recalculate |
| Missing Required | Required fields empty | Populate data |
| Format | Date/number format wrong | Fix format |

```bash
# Detailed validation report
greenlang report validate \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --schema "EPA-ERT-v3.0" \
  --output-errors detailed

# Check data completeness
greenlang report data-completeness \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011

# Identify specific issues
greenlang report issues \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --group-by "severity"
```

#### Mitigation Actions

```bash
# Auto-fix known issues
greenlang report auto-fix \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --fix-types "format,calculation,schema"

# Manual corrections for remaining issues
greenlang report edit \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --corrections-file /fixes/quarterly-Q3-corrections.json

# Regenerate report
greenlang report regenerate \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011

# Re-validate
greenlang report validate \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --schema "EPA-ERT-v3.0"

# Submit
greenlang report submit \
  --report-id quarterly-emissions-Q3-2025-facility-bcd-011 \
  --portal "EPA-CEDRI" \
  --confirm
```

---

### 6.2 INC-012: Excess Emissions Report Missed

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Regulatory Report Violation |
| Typical Severity | SEV1 |
| Regulatory Impact | Violation of notification requirements |
| Common Causes | Alert failure, process gap, personnel unavailable |

#### Detection

```json
{
  "alert_id": "GL010-EER-012-20251126T080000Z",
  "severity": "critical",
  "type": "missed_report_deadline",
  "report_type": "excess_emissions",
  "facility_id": "facility-efg-012",
  "event_date": "2025-11-24",
  "deadline": "2025-11-26T00:00:00Z",
  "deadline_type": "2-business-day",
  "status": "overdue",
  "excess_event": {
    "pollutant": "NOx",
    "duration_hours": 2.5,
    "max_exceedance_pct": 15
  },
  "timestamp": "2025-11-26T08:00:00Z"
}
```

#### Initial Response

**This is a Regulatory Violation - Escalate Immediately**

```bash
# Escalate
greenlang escalate \
  --incident-id GL010-EER-012-20251126T080000Z \
  --severity SEV1 \
  --notify "compliance-lead,regulatory-affairs,legal"

# Get event details
greenlang emissions excess-event \
  --facility facility-efg-012 \
  --date "2025-11-24" \
  --pollutant NOx \
  --detail-level full
```

#### Mitigation Actions

```bash
# Prepare late notification
greenlang regulatory prepare-late-notification \
  --facility facility-efg-012 \
  --event-date "2025-11-24" \
  --event-type "excess-emissions" \
  --include-explanation

# Submit with late designation
greenlang regulatory submit-late \
  --facility facility-efg-012 \
  --report-type "excess-emissions" \
  --original-deadline "2025-11-26T00:00:00Z" \
  --late-reason "System notification failure" \
  --corrective-action "Enhanced alert verification implemented"

# Document for compliance record
greenlang compliance document-violation \
  --facility facility-efg-012 \
  --violation-type "late-excess-emissions-report" \
  --violation-date "2025-11-26" \
  --corrective-action "Process improvements implemented"
```

---

### 6.3 INC-013: Annual Emissions Inventory Delayed

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Annual Report Delay |
| Typical Severity | SEV2 (if time remaining), SEV1 (deadline imminent) |
| Regulatory Impact | Potential penalties, audit flag |
| Common Causes | Data aggregation issues, verification delays, system problems |

#### Detection

```json
{
  "alert_id": "GL010-AEI-013-20251126T100000Z",
  "severity": "high",
  "type": "annual_inventory_delay",
  "facility_id": "facility-hij-013",
  "reporting_year": 2024,
  "deadline": "2025-03-31T23:59:59Z",
  "current_status": "data_validation",
  "completion_percentage": 75,
  "blocking_issues": [
    "Missing HAP data for Q2",
    "Unresolved calculation discrepancy",
    "Pending third-party verification"
  ],
  "timestamp": "2025-11-26T10:00:00Z"
}
```

#### Investigation Steps

```bash
# Get inventory status
greenlang inventory status \
  --facility facility-hij-013 \
  --year 2024 \
  --detail-level full

# Identify blockers
greenlang inventory blockers \
  --facility facility-hij-013 \
  --year 2024

# Check data completeness by category
greenlang inventory completeness \
  --facility facility-hij-013 \
  --year 2024 \
  --by-category
```

#### Mitigation Actions

```bash
# Address missing HAP data
greenlang inventory resolve-data-gap \
  --facility facility-hij-013 \
  --year 2024 \
  --category "HAP" \
  --period "Q2"

# Resolve calculation discrepancy
greenlang emissions reconcile \
  --facility facility-hij-013 \
  --year 2024 \
  --auto-resolve-minor

# Expedite verification
greenlang verification expedite \
  --facility facility-hij-013 \
  --year 2024 \
  --priority "critical"
```

---

### 6.4 INC-014: EPA CEDRI Connection Failure

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Integration Failure |
| Typical Severity | SEV2 (reports not due), SEV1 (reports due) |
| Regulatory Impact | Cannot submit required reports |
| Common Causes | EPA system maintenance, network issue, certificate expired, API change |

#### Detection

```json
{
  "alert_id": "GL010-CEDRI-014-20251126T113000Z",
  "severity": "high",
  "type": "integration_failure",
  "integration": "EPA_CEDRI",
  "failure_type": "connection_refused",
  "last_successful": "2025-11-25T18:45:00Z",
  "consecutive_failures": 15,
  "pending_submissions": 3,
  "timestamp": "2025-11-26T11:30:00Z"
}
```

#### Initial Response

```bash
# Check EPA CEDRI status
greenlang integration status EPA_CEDRI

# Check for EPA maintenance announcements
greenlang integration announcements EPA_CEDRI

# Test connection
greenlang integration test EPA_CEDRI --verbose
```

#### Investigation Steps

```bash
# Detailed connectivity test
greenlang integration diagnose EPA_CEDRI \
  --tests "dns,ssl,auth,api"

# Check certificate status
greenlang integration certificates EPA_CEDRI

# Review recent changes
greenlang integration changes EPA_CEDRI --last 7d

# Check EPA status page
greenlang web-check https://cdx.epa.gov/status
```

#### Mitigation Actions

```bash
# If certificate issue
greenlang integration renew-certificate EPA_CEDRI

# If configuration issue
greenlang integration reconfigure EPA_CEDRI \
  --from-template current

# Enable manual submission mode
greenlang integration enable-manual-mode EPA_CEDRI

# Queue reports for retry
greenlang report queue-retry \
  --integration EPA_CEDRI \
  --retry-interval "30m"

# Monitor for recovery
greenlang watch integration EPA_CEDRI \
  --until-recovered \
  --notify-on-recovery
```

---

### 6.5 INC-015: EU ETS Submission Error

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | International Reporting Failure |
| Typical Severity | SEV1 (compliance year end), SEV2 (mid-year) |
| Regulatory Impact | EU ETS compliance, allowance surrender deadline |
| Common Causes | Data format mismatch, verification issue, registry problem |

#### Detection

```json
{
  "alert_id": "GL010-ETS-015-20251126T140000Z",
  "severity": "critical",
  "type": "ets_submission_error",
  "facility_id": "facility-eu-015",
  "installation_id": "EU-DE-123456",
  "submission_type": "annual_emissions",
  "reporting_year": 2024,
  "error_code": "VER-003",
  "error_message": "Verification statement signature invalid",
  "deadline": "2025-03-31T23:59:59Z",
  "timestamp": "2025-11-26T14:00:00Z"
}
```

#### Initial Response

```bash
# Check EU ETS registry status
greenlang integration status EU_ETS

# Get submission details
greenlang ets submission-status \
  --installation-id EU-DE-123456 \
  --year 2024

# Review error details
greenlang ets error-detail \
  --error-code "VER-003"
```

#### Investigation Steps

```bash
# Verify verification statement
greenlang ets verify-statement \
  --installation-id EU-DE-123456 \
  --year 2024

# Check verifier credentials
greenlang ets verifier-status \
  --installation-id EU-DE-123456

# Validate submission package
greenlang ets validate-package \
  --installation-id EU-DE-123456 \
  --year 2024 \
  --schema "EU-ETS-MRV-2024"
```

#### Mitigation Actions

```bash
# Contact verifier for corrected statement
greenlang ets request-correction \
  --installation-id EU-DE-123456 \
  --issue "signature-invalid" \
  --verifier-contact "verifier@example.com"

# Prepare resubmission
greenlang ets prepare-resubmission \
  --installation-id EU-DE-123456 \
  --year 2024

# Submit corrected package
greenlang ets submit \
  --installation-id EU-DE-123456 \
  --year 2024 \
  --package /submissions/EU-DE-123456-2024-corrected.xml \
  --verification-statement /submissions/verification-corrected.pdf
```

---

## 7. System Failure Incidents

### 7.1 INC-016: Agent Pod Crash/Restart

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | System Availability |
| Typical Severity | SEV2 (single pod), SEV1 (multiple pods/service impact) |
| Service Impact | Temporary loss of processing capability |
| Common Causes | OOM, application error, resource limits, node issues |

#### Detection

```json
{
  "alert_id": "GL010-POD-016-20251126T094523Z",
  "severity": "high",
  "type": "pod_restart",
  "pod_name": "gl-010-emissionwatch-7f8d9c4b5-abc12",
  "namespace": "gl-agents",
  "restart_count": 5,
  "time_window": "PT30M",
  "last_exit_code": 137,
  "termination_reason": "OOMKilled",
  "timestamp": "2025-11-26T09:45:23Z"
}
```

#### Initial Response

```bash
# Check pod status
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch

# Check pod events
kubectl describe pod gl-010-emissionwatch-7f8d9c4b5-abc12 -n gl-agents

# Check recent logs
kubectl logs gl-010-emissionwatch-7f8d9c4b5-abc12 -n gl-agents --previous --tail=500

# Check resource usage
kubectl top pod gl-010-emissionwatch-7f8d9c4b5-abc12 -n gl-agents
```

#### Investigation Steps

```bash
# Check memory usage trend
greenlang metrics query \
  --metric "container_memory_working_set_bytes" \
  --pod "gl-010-emissionwatch-*" \
  --time-range "last-4h"

# Check for memory leaks
greenlang diagnostics memory-profile \
  --agent GL-010 \
  --duration 5m

# Review application logs for errors
kubectl logs -n gl-agents -l app=gl-010-emissionwatch \
  --since=2h | grep -i "error\|exception\|oom"

# Check node health
kubectl describe node $(kubectl get pod gl-010-emissionwatch-7f8d9c4b5-abc12 -n gl-agents -o jsonpath='{.spec.nodeName}')
```

#### Mitigation Actions

```bash
# Increase memory limits if OOM
kubectl patch deployment gl-010-emissionwatch -n gl-agents --type=merge -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "emissionwatch",
          "resources": {
            "limits": {
              "memory": "8Gi"
            },
            "requests": {
              "memory": "4Gi"
            }
          }
        }]
      }
    }
  }
}'

# Force pod restart on different node
kubectl delete pod gl-010-emissionwatch-7f8d9c4b5-abc12 -n gl-agents

# Scale up replicas if needed
kubectl scale deployment gl-010-emissionwatch -n gl-agents --replicas=3

# Monitor recovery
kubectl get pods -n gl-agents -l app=gl-010-emissionwatch -w
```

---

### 7.2 INC-017: Database Connectivity Loss

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Infrastructure Failure |
| Typical Severity | SEV1 |
| Service Impact | All data operations halted |
| Common Causes | Network issue, database server down, connection pool exhausted |

#### Detection

```json
{
  "alert_id": "GL010-DB-017-20251126T120000Z",
  "severity": "critical",
  "type": "database_connectivity_failure",
  "database": "gl010-emissions-db",
  "connection_string": "postgresql://emissions-db.greenlang.io:5432/gl010",
  "last_successful_query": "2025-11-26T11:58:45Z",
  "error": "connection refused",
  "affected_operations": ["data_collection", "calculation", "reporting"],
  "timestamp": "2025-11-26T12:00:00Z"
}
```

#### Initial Response

```bash
# Check database connectivity
greenlang db test-connection --database gl010-emissions-db

# Check database server status
greenlang db server-status --database gl010-emissions-db

# Check connection pool
greenlang db pool-status --database gl010-emissions-db
```

#### Investigation Steps

```bash
# Ping database server
ping emissions-db.greenlang.io

# Check DNS resolution
nslookup emissions-db.greenlang.io

# Check port connectivity
nc -zv emissions-db.greenlang.io 5432

# Check database logs
greenlang db logs --database gl010-emissions-db --last 30m

# Check for blocking queries
greenlang db blocking-queries --database gl010-emissions-db
```

#### Mitigation Actions

```bash
# Reset connection pool
greenlang db reset-connections --database gl010-emissions-db

# Failover to replica if available
greenlang db failover --database gl010-emissions-db --to replica-1

# Enable data buffering
greenlang agent configure --agent GL-010 --enable-data-buffer

# Monitor recovery
greenlang db watch --database gl010-emissions-db --until-healthy
```

---

### 7.3 INC-018: Integration Connector Failure

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Integration Failure |
| Typical Severity | SEV2 (single source), SEV1 (critical source) |
| Service Impact | Data from specific source unavailable |
| Common Causes | API change, authentication failure, rate limiting, network issue |

#### Detection

```json
{
  "alert_id": "GL010-INT-018-20251126T083000Z",
  "severity": "high",
  "type": "integration_connector_failure",
  "connector": "cems-modbus-connector",
  "facility_id": "facility-xyz-018",
  "protocol": "Modbus/TCP",
  "target_address": "192.168.1.100:502",
  "failure_type": "connection_timeout",
  "consecutive_failures": 10,
  "last_successful": "2025-11-26T08:15:00Z",
  "timestamp": "2025-11-26T08:30:00Z"
}
```

#### Initial Response

```bash
# Check connector status
greenlang connector status cems-modbus-connector

# Test connectivity
greenlang connector test cems-modbus-connector --verbose

# Check for recent changes
greenlang connector changes cems-modbus-connector --last 24h
```

#### Investigation Steps

```bash
# Detailed connectivity test
greenlang connector diagnose cems-modbus-connector \
  --tests "network,protocol,authentication,data"

# Check firewall rules
greenlang network check \
  --source gl-010-pod \
  --destination 192.168.1.100:502

# Review connector logs
greenlang connector logs cems-modbus-connector --last 1h

# Check CEMS device status (if accessible)
greenlang cems device-status --address 192.168.1.100
```

#### Mitigation Actions

```bash
# Restart connector
greenlang connector restart cems-modbus-connector

# Reconfigure if needed
greenlang connector reconfigure cems-modbus-connector \
  --timeout 60 \
  --retry-count 5

# Enable fallback data source
greenlang connector enable-fallback \
  --primary cems-modbus-connector \
  --fallback cems-backup-connector

# Monitor recovery
greenlang watch connector cems-modbus-connector --until-recovered
```

---

### 7.4 INC-019: Alert Notification Failure

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Notification System Failure |
| Typical Severity | SEV1 (during active incident), SEV2 (normal operations) |
| Service Impact | Staff not receiving critical alerts |
| Common Causes | Email service down, SMS gateway issue, Slack API failure |

#### Detection

```json
{
  "alert_id": "GL010-NOTIF-019-20251126T110000Z",
  "severity": "critical",
  "type": "notification_failure",
  "notification_channels": ["email", "sms", "slack"],
  "failed_channels": ["email", "sms"],
  "operational_channels": ["slack"],
  "failed_notifications": 45,
  "time_window": "PT1H",
  "affected_alerts": ["emissions", "cems", "system"],
  "timestamp": "2025-11-26T11:00:00Z"
}
```

#### Initial Response

```bash
# Check notification system status
greenlang notifications status

# Test each channel
greenlang notifications test --channel email --recipient oncall@greenlang.io
greenlang notifications test --channel sms --recipient +15550100
greenlang notifications test --channel slack --channel "#gl-010-alerts"

# Get failed notifications
greenlang notifications failed --last 1h
```

#### Investigation Steps

```bash
# Check email service
greenlang notifications diagnose email

# Check SMS gateway
greenlang notifications diagnose sms

# Check Slack integration
greenlang notifications diagnose slack

# Review notification logs
greenlang notifications logs --last 2h --level error
```

#### Mitigation Actions

```bash
# Enable backup notification channel
greenlang notifications enable-backup

# Route critical alerts to working channel
greenlang notifications reroute \
  --severity "critical,high" \
  --from "email,sms" \
  --to "slack"

# Resend failed notifications
greenlang notifications resend-failed \
  --time-range "last-1h" \
  --severity "critical,high"

# Monitor notification health
greenlang watch notifications --until-all-healthy
```

---

### 7.5 INC-020: Cache Corruption

#### Overview

| Property | Value |
|----------|-------|
| Incident Type | Data Integrity Issue |
| Typical Severity | SEV2 (performance impact), SEV1 (calculation errors) |
| Service Impact | Incorrect calculations, slow performance |
| Common Causes | Memory issue, incomplete write, race condition |

#### Detection

```json
{
  "alert_id": "GL010-CACHE-020-20251126T141500Z",
  "severity": "high",
  "type": "cache_corruption",
  "cache_type": "emission_factors",
  "corruption_detected": "checksum_mismatch",
  "affected_keys": 127,
  "total_keys": 5000,
  "impact": "emission_calculations_may_be_incorrect",
  "timestamp": "2025-11-26T14:15:00Z"
}
```

#### Initial Response

```bash
# Check cache health
greenlang cache health --type emission_factors

# Identify corrupted entries
greenlang cache corrupted --type emission_factors --detail

# Check for recent calculation errors
greenlang emissions calculation-errors --last 4h
```

#### Investigation Steps

```bash
# Analyze corruption pattern
greenlang cache analyze-corruption \
  --type emission_factors \
  --output /analysis/cache-corruption-analysis.json

# Check memory health
greenlang diagnostics memory --component cache

# Review cache operations log
greenlang cache operations-log --last 4h
```

#### Mitigation Actions

```bash
# Clear corrupted cache entries
greenlang cache clear-corrupted --type emission_factors

# Rebuild cache from authoritative source
greenlang cache rebuild \
  --type emission_factors \
  --source database

# Verify cache integrity
greenlang cache verify --type emission_factors

# Recalculate affected emissions
greenlang emissions recalculate \
  --time-range "2025-11-26T10:00:00Z/2025-11-26T14:15:00Z" \
  --verify-against-source
```

---

## 8. Escalation Matrix

### 8.1 Escalation Levels

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ESCALATION MATRIX                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Level 1: On-Call Engineer                                                          │
│  └── Response: 5 min (SEV1), 15 min (SEV2), 1 hr (SEV3), 4 hr (SEV4)              │
│                                                                                      │
│  Level 2: Technical Lead + Operations Manager                                        │
│  └── Escalate if: No progress in 15 min (SEV1), 1 hr (SEV2)                        │
│  └── Contact: +1-555-0102, ops-manager@greenlang.io                                │
│                                                                                      │
│  Level 3: VP Engineering + Compliance Lead                                          │
│  └── Escalate if: No progress in 30 min (SEV1), 2 hr (SEV2)                        │
│  └── Contact: +1-555-0104, +1-555-0103                                             │
│                                                                                      │
│  Level 4: Executive Team + Legal                                                    │
│  └── Escalate if: Regulatory violation confirmed, public impact                     │
│  └── Contact: +1-555-0105 (Legal), +1-555-0107 (CTO)                              │
│                                                                                      │
│  Regulatory Affairs: Always notify for emissions violations                          │
│  └── Contact: +1-555-0106, regulatory@greenlang.io                                 │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Escalation Decision Tree

```
                           INCIDENT DETECTED
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    Assess Severity Level     │
                    └─────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
      SEV1/SEV2             SEV3/SEV4            Not an Incident
          │                       │                       │
          ▼                       ▼                       ▼
    Page On-Call           Email On-Call            Log & Close
          │                       │
          ▼                       │
   ┌──────────────┐               │
   │ Emissions    │               │
   │ Violation?   │               │
   └──────────────┘               │
      │Yes    │No                 │
      ▼       │                   │
  Notify      │                   │
  Regulatory  │                   │
  Affairs     │                   │
      │       │                   │
      └───────┼───────────────────┘
              ▼
    ┌─────────────────────────────┐
    │   Progress within SLA?       │
    └─────────────────────────────┘
              │
         Yes  │  No
              │   │
              ▼   ▼
          Continue  Escalate to
          Response  Next Level
```

### 8.3 On-Call Rotation Schedule

| Week | Primary On-Call | Secondary On-Call | Compliance Contact |
|------|----------------|-------------------|-------------------|
| Week 1 | Team A | Team B | Compliance Officer 1 |
| Week 2 | Team B | Team C | Compliance Officer 2 |
| Week 3 | Team C | Team D | Compliance Officer 1 |
| Week 4 | Team D | Team A | Compliance Officer 2 |

**Rotation Rules:**
- Handoff at 09:00 Monday local time
- 15-minute overlap for briefing
- On-call engineer must respond within SLA
- Secondary takes over if primary unresponsive

### 8.4 Escalation Procedures

**Escalating to Level 2:**

```bash
# Escalate incident to Level 2
greenlang escalate \
  --incident-id {incident-id} \
  --to-level 2 \
  --reason "No resolution progress after 15 minutes" \
  --current-status "Investigating root cause"
```

**Escalating to Regulatory Affairs:**

```bash
# Escalate to regulatory team
greenlang escalate \
  --incident-id {incident-id} \
  --to-team regulatory-affairs \
  --reason "Emissions violation confirmed" \
  --violation-type "NOx exceedance" \
  --duration "45 minutes" \
  --max-exceedance "124% of limit"
```

**Escalating to Legal:**

```bash
# Escalate to legal team
greenlang escalate \
  --incident-id {incident-id} \
  --to-team legal \
  --reason "Potential enforcement action" \
  --regulatory-notification-sent true \
  --expected-penalty-range "$10,000-$50,000"
```

---

## 9. Communication Templates

### 9.1 Initial Incident Notification

**Subject:** [SEV{X}] GL-010 EMISSIONWATCH Incident - {Brief Description}

```markdown
## Incident Notification

**Incident ID:** {incident-id}
**Severity:** SEV{X}
**Status:** Investigating
**Time Detected:** {timestamp}

### Summary
{One or two sentence description of the incident}

### Impact
- **Facilities Affected:** {list}
- **Pollutants Affected:** {list}
- **Compliance Status:** {compliant/non-compliant/at-risk}
- **Regulatory Notification Required:** {yes/no}

### Current Actions
- {Action 1}
- {Action 2}

### Next Update
{time} or when significant change occurs

### Contacts
- **Incident Commander:** {name} ({contact})
- **Technical Lead:** {name} ({contact})

---
This is an automated notification from GL-010 EMISSIONWATCH.
```

### 9.2 Status Update

**Subject:** [UPDATE {N}] [SEV{X}] {Incident ID} - {Brief Status}

```markdown
## Incident Status Update #{N}

**Incident ID:** {incident-id}
**Severity:** SEV{X} (unchanged/escalated from SEV{Y})
**Status:** {Investigating/Mitigating/Recovering/Resolved}
**Duration:** {time since detection}

### Progress Since Last Update
- {Progress item 1}
- {Progress item 2}

### Current Status
{Current state description}

### Root Cause
{Known/Suspected/Under Investigation}
{If known: brief description}

### Actions in Progress
- {Action 1} - {Owner} - ETA {time}
- {Action 2} - {Owner} - ETA {time}

### Outstanding Issues
- {Issue 1}
- {Issue 2}

### Next Update
{time} or upon resolution

---
Incident Commander: {name}
```

### 9.3 Resolution Notification

**Subject:** [RESOLVED] [SEV{X}] {Incident ID} - {Brief Description}

```markdown
## Incident Resolution Notification

**Incident ID:** {incident-id}
**Severity:** SEV{X}
**Status:** RESOLVED
**Duration:** {total duration}

### Resolution Summary
{Brief description of how the incident was resolved}

### Root Cause
{Root cause description}

### Impact Summary
- **Duration of Impact:** {time}
- **Facilities Affected:** {count}
- **Data Impact:** {description}
- **Regulatory Impact:** {description}

### Corrective Actions Taken
- {Action 1}
- {Action 2}

### Preventive Measures
- {Measure 1}
- {Measure 2}

### Post-Incident Review
Scheduled for: {date/time}
Attendees: {list}

### Regulatory Follow-up Required
{Yes - describe / No}

---
Thank you to everyone who assisted with this incident.
Incident Commander: {name}
```

### 9.4 Regulatory Agency Notification

**Subject:** Excess Emissions Notification - {Facility Name} - {Date}

```markdown
## Excess Emissions Notification

**TO:** {Regulatory Agency}
**FROM:** {Company Name}
**DATE:** {notification date}
**RE:** Excess Emissions Event Notification

### Facility Information
- **Facility Name:** {name}
- **Facility ID:** {EPA/state ID}
- **Permit Number:** {permit number}
- **Location:** {address}

### Event Information
- **Date of Event:** {date}
- **Start Time:** {start time}
- **End Time:** {end time}
- **Duration:** {duration}
- **Affected Unit(s):** {unit IDs}

### Emissions Information
- **Pollutant(s):** {pollutant list}
- **Applicable Limit:** {limit value and units}
- **Maximum Emission During Event:** {max value and units}
- **Estimated Excess Emissions:** {quantity}

### Cause of Event
{Detailed description of what caused the excess emissions}

### Corrective Actions
{Description of actions taken to correct the situation}

### Preventive Measures
{Description of measures to prevent recurrence}

### Contact Information
- **Environmental Manager:** {name}
- **Phone:** {phone}
- **Email:** {email}

---
This notification is being provided pursuant to {regulatory citation}.
```

### 9.5 Internal Stakeholder Update

**Subject:** GL-010 Emissions Incident Update - {Facility} - Action Required

```markdown
## Internal Stakeholder Update

**Priority:** {High/Medium/Low}
**Action Required:** {Yes/No}
**Response Deadline:** {if applicable}

### Situation Summary
{Executive-level summary of the incident}

### Business Impact
- **Operational Impact:** {description}
- **Financial Impact:** {estimated cost range}
- **Regulatory Impact:** {description}
- **Reputation Impact:** {description}

### Actions Required from You
{Specific actions needed from the recipient}

### Timeline
- **Event Occurred:** {time}
- **Expected Resolution:** {time}
- **Regulatory Deadline:** {if applicable}

### Support Needed
{Any support needed from the stakeholder}

### Questions?
Contact: {name} at {contact}

---
This is an internal communication. Please do not forward externally.
```

---

## 10. Post-Incident Review

### 10.1 Post-Incident Review Process

**Timeline:**
- SEV1: Review within 3 business days
- SEV2: Review within 5 business days
- SEV3: Review within 10 business days
- SEV4: Review within 30 days (may be batched)

**Required Attendees:**
- Incident Commander
- Technical Lead
- Operations Manager
- Compliance Lead (for emissions incidents)
- Affected facility representative

### 10.2 Post-Incident Review Template

```markdown
# Post-Incident Review

## Incident Information
- **Incident ID:** {id}
- **Severity:** {severity}
- **Date/Time:** {start} to {end}
- **Duration:** {duration}
- **Incident Commander:** {name}

## Executive Summary
{2-3 paragraph summary of the incident}

## Timeline
| Time | Event | Actor |
|------|-------|-------|
| {time} | {event} | {person/system} |
| ... | ... | ... |

## Impact Assessment

### Emissions Impact
- **Pollutants Affected:** {list}
- **Duration of Exceedance:** {time}
- **Maximum Exceedance:** {value}
- **Estimated Excess Emissions:** {quantity}

### Data Impact
- **Data Availability Impact:** {description}
- **Data Quality Impact:** {description}
- **Records Affected:** {count}

### Business Impact
- **Operational Impact:** {description}
- **Financial Impact:** {estimated cost}

### Regulatory Impact
- **Violations Incurred:** {list}
- **Notifications Made:** {list}
- **Expected Penalties:** {range}
- **Audit Risk:** {assessment}

## Root Cause Analysis

### What Happened
{Detailed technical description}

### Why It Happened (5 Whys)
1. Why? {answer 1}
2. Why? {answer 2}
3. Why? {answer 3}
4. Why? {answer 4}
5. Why? {answer 5 - root cause}

### Contributing Factors
- {Factor 1}
- {Factor 2}
- {Factor 3}

## What Went Well
- {Positive 1}
- {Positive 2}

## What Could Be Improved
- {Improvement 1}
- {Improvement 2}

## Action Items

| ID | Action | Owner | Due Date | Status |
|----|--------|-------|----------|--------|
| 1 | {action} | {owner} | {date} | Open |
| 2 | {action} | {owner} | {date} | Open |

## Lessons Learned
1. {Lesson 1}
2. {Lesson 2}

## Runbook Updates Required
- [ ] {Runbook update 1}
- [ ] {Runbook update 2}

## Follow-up Review
Date: {date} to verify action items completed

---
Review Completed: {date}
Reviewed By: {attendees}
```

### 10.3 Action Item Tracking

```bash
# Create action items from PIR
greenlang pir create-actions \
  --incident-id {incident-id} \
  --actions "[
    {
      \"action\": \"Implement automated SCR monitoring alerts\",
      \"owner\": \"John Smith\",
      \"due_date\": \"2025-12-15\",
      \"priority\": \"high\"
    },
    {
      \"action\": \"Update NOx exceedance runbook\",
      \"owner\": \"Jane Doe\",
      \"due_date\": \"2025-12-01\",
      \"priority\": \"medium\"
    }
  ]"

# Track action item progress
greenlang pir action-status --incident-id {incident-id}

# Close action item
greenlang pir close-action \
  --incident-id {incident-id} \
  --action-id 1 \
  --resolution "Alerts implemented and tested"
```

---

## 11. Regulatory Notification Requirements

### 11.1 Federal Requirements (EPA)

| Event Type | Notification Timeline | Method | Reference |
|------------|----------------------|--------|-----------|
| Excess Emissions | 2 business days | Phone + Written | 40 CFR 60.7(a)(3) |
| Emergency | Immediate | Phone | 40 CFR 70.6(g) |
| Deviation from Permit | Semiannual report | Written | 40 CFR 70.6(a)(3)(iii) |
| CEMS Data Loss >25% | Quarterly report | Electronic | 40 CFR 75.57 |
| Annual Emissions Report | March 31 | Electronic (CEDRI) | 40 CFR 98 |

### 11.2 State Requirements (Example - California)

| Event Type | Notification Timeline | Method | Reference |
|------------|----------------------|--------|-----------|
| Excess Emissions | Same day | Phone | CARB Rule 430 |
| Equipment Breakdown | 1 hour | Phone | Local AQMD Rule |
| Visible Emissions | Immediate | Phone | Rule 401 |
| Odor Complaint | Immediate | Phone | Rule 402 |

### 11.3 EU ETS Requirements

| Event Type | Notification Timeline | Method | Reference |
|------------|----------------------|--------|-----------|
| Annual Emissions | March 31 | Registry submission | EU ETS Directive |
| Verification Statement | March 31 | PDF upload | MRV Regulation |
| Changes to MP | 30 days | Competent Authority | Article 15 |
| Force Majeure | Immediate | Written | Article 47 |

### 11.4 Regulatory Contact Directory

```yaml
# US Federal
EPA_Region_1:
  phone: "+1-617-918-1111"
  email: "r1_air@epa.gov"
  hours: "8:30 AM - 5:00 PM ET"

EPA_Region_9:
  phone: "+1-415-947-8000"
  email: "r9.air@epa.gov"
  hours: "8:00 AM - 4:30 PM PT"

# California
CARB:
  phone: "+1-800-242-4450"
  email: "helpline@arb.ca.gov"
  hours: "8:00 AM - 5:00 PM PT"

SCAQMD:
  phone: "+1-800-288-7664"
  emergency: "+1-800-CUT-SMOG"
  hours: "24/7 hotline"

# EU
EU_ETS_Registry:
  url: "https://ec.europa.eu/clima/ets/registry"
  helpdesk: "ets-registry@ec.europa.eu"
```

### 11.5 Notification Decision Tree

```
                    EMISSIONS EVENT DETECTED
                              │
                              ▼
                   ┌─────────────────────┐
                   │  Event Type?         │
                   └─────────────────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
     ▼                        ▼                        ▼
 Excess              Emergency/            CEMS
 Emissions           Breakdown             Outage
     │                        │                        │
     ▼                        ▼                        ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Duration?   │      │ Immediate   │      │ Duration?   │
│ >60 min?    │      │ Notification│      │ >25% hour?  │
└─────────────┘      │ Required    │      └─────────────┘
     │                └─────────────┘           │
  Yes│No                                    Yes│No
     │                                         │
     ▼                                         ▼
Notification                          Document in
Required per                          Quarterly
Permit                                Report
```

---

## 12. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|------------|
| CEMS | Continuous Emissions Monitoring System |
| CEDRI | Compliance and Emissions Data Reporting Interface (EPA) |
| COM | Continuous Opacity Monitor |
| DAHS | Data Acquisition and Handling System |
| ETS | Emissions Trading System |
| RATA | Relative Accuracy Test Audit |
| SCR | Selective Catalytic Reduction |
| SNCR | Selective Non-Catalytic Reduction |
| ESP | Electrostatic Precipitator |
| FGD | Flue Gas Desulfurization |

### Appendix B: Quick Reference Commands

```bash
# Emissions Status
greenlang emissions status --facility {facility-id}
greenlang emissions current --facility {facility-id} --pollutant {pollutant}
greenlang compliance status --facility {facility-id}

# CEMS Operations
greenlang cems status --facility {facility-id}
greenlang cems data-quality --facility {facility-id}
greenlang cems calibration-status --facility {facility-id}

# Incident Management
greenlang incident create --agent GL-010 --severity SEV{X}
greenlang incident update --incident-id {id} --status {status}
greenlang escalate --incident-id {id} --to-level {level}

# Reporting
greenlang report status --facility {facility-id}
greenlang report generate --type {type} --facility {facility-id}
greenlang regulatory notify --facility {facility-id} --type {type}
```

### Appendix C: Related Documentation

| Document | Location | Purpose |
|----------|----------|---------|
| Troubleshooting Guide | ./TROUBLESHOOTING.md | Detailed diagnostic procedures |
| Rollback Procedure | ./ROLLBACK_PROCEDURE.md | System rollback steps |
| Scaling Guide | ./SCALING_GUIDE.md | Capacity management |
| Maintenance Guide | ./MAINTENANCE.md | Routine maintenance |
| API Documentation | /docs/api/gl-010/ | API reference |
| Architecture Guide | /docs/architecture/gl-010/ | System architecture |

### Appendix D: Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-26 | GL-TechWriter | Initial release |

### Appendix E: Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Operations Manager | | | |
| Compliance Lead | | | |
| VP Engineering | | | |

---

**Document Classification:** Internal Use Only

**Next Review Date:** 2026-02-26

**Feedback:** Submit feedback to docs@greenlang.io with subject "GL-010 Incident Response Feedback"
