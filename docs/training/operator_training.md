# GreenLang Operator Training

## Module Overview

**Duration:** 8 hours
**Prerequisites:** [Fundamentals](./fundamentals.md)
**Level:** Intermediate

This module prepares operators to effectively use GreenLang for day-to-day process heat management, respond to alarms, troubleshoot issues, and optimize performance.

---

## Part 1: Day-to-Day Operations

### 1.1 Operator Dashboard Overview

The GreenLang Operator Dashboard is your primary interface for monitoring and controlling process heat operations.

#### Dashboard Layout

```
+------------------------------------------------------------------+
|  GreenLang Operator Dashboard                    [User] [Logout] |
+------------------------------------------------------------------+
|                                                                   |
|  +-------------------+  +------------------------------------+   |
|  | Navigation        |  | Main Display Area                  |   |
|  |                   |  |                                    |   |
|  | > Overview        |  |  +------------+  +------------+    |   |
|  |   Furnace Zone 1  |  |  | Zone 1     |  | Zone 2     |    |   |
|  |   Furnace Zone 2  |  |  | 847C       |  | 912C       |    |   |
|  |   Furnace Zone 3  |  |  | [NORMAL]   |  | [WARNING]  |    |   |
|  |   Cooling System  |  |  +------------+  +------------+    |   |
|  |                   |  |                                    |   |
|  | > Alarms          |  |  +------------+  +------------+    |   |
|  | > Reports         |  |  | Zone 3     |  | Cooling    |    |   |
|  | > Settings        |  |  | 1105C      |  | 45C        |    |   |
|  |                   |  |  | [NORMAL]   |  | [NORMAL]   |    |   |
|  +-------------------+  |  +------------+  +------------+    |   |
|                         |                                    |   |
|                         +------------------------------------+   |
|                                                                   |
|  +------------------------------------------------------------+  |
|  | Active Alarms: 2      | System Status: RUNNING             |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

#### Key Dashboard Elements

| Element | Description | Location |
|---------|-------------|----------|
| Navigation Panel | Access different views and zones | Left sidebar |
| Main Display | Real-time process visualization | Center |
| Status Bar | Active alarms and system status | Bottom |
| User Menu | Profile, settings, logout | Top right |

### 1.2 Monitoring Process Heat

#### Real-Time Monitoring

**Temperature Displays:**

```
Zone 1 Temperature Monitor
+-----------------------------------------+
| Current: 847C        Setpoint: 850C     |
| Deviation: -3C       Status: NORMAL     |
+-----------------------------------------+
| Trend (Last 4 Hours):                   |
|                                         |
|  870 |                    *             |
|  860 |              *  *     *          |
|  850 |----*--*--*-----------*--*--*---- |
|  840 |  *                            *  |
|  830 |                                  |
|      +----------------------------------+
|       12:00  13:00  14:00  15:00  16:00 |
+-----------------------------------------+
| Rate of Change: +0.5C/min               |
| Prediction (1hr): 852C [849-855C]       |
+-----------------------------------------+
```

**What to Monitor:**

| Parameter | Normal Range | Action Trigger |
|-----------|--------------|----------------|
| Temperature Deviation | +/- 5C | > 10C |
| Rate of Change | +/- 2C/min | > 5C/min |
| ML Anomaly Score | 0.0 - 0.3 | > 0.5 |
| Equipment Health | > 80% | < 70% |

#### Monitoring Best Practices

1. **Regular Scans:** Review all zones every 15-30 minutes
2. **Trend Analysis:** Look for gradual changes, not just current values
3. **Cross-Reference:** Compare related parameters (temperature vs. fuel flow)
4. **Shift Handover:** Document any abnormalities for incoming shift

### 1.3 Standard Operating Procedures

#### Shift Start Procedure

```
SHIFT START CHECKLIST

[ ] 1. Log into GreenLang dashboard
[ ] 2. Review active alarms (acknowledge any handed over)
[ ] 3. Check all zone temperatures and trends
[ ] 4. Review ML health indicators
[ ] 5. Read shift handover notes
[ ] 6. Verify communication systems operational
[ ] 7. Confirm emergency contacts available
[ ] 8. Document shift start in operator log
```

#### Routine Monitoring Procedure

```
HOURLY MONITORING PROCEDURE

1. Zone Temperature Check
   - Open Overview dashboard
   - Verify all zones within +/- 5C of setpoint
   - Note any deviations in operator log

2. Equipment Health Check
   - Review ML health predictions
   - Check for maintenance recommendations
   - Report any concerns to maintenance

3. Energy Efficiency Check
   - Review specific energy consumption
   - Compare to baseline
   - Note any significant deviations

4. Safety System Check
   - Verify all safety interlocks active
   - Check emergency shutdown availability
   - Confirm alarm system operational
```

#### Shift End Procedure

```
SHIFT END CHECKLIST

[ ] 1. Complete final monitoring round
[ ] 2. Document any open issues or concerns
[ ] 3. Update shift log with summary
[ ] 4. Brief incoming operator on status
[ ] 5. Hand over any active alarms
[ ] 6. Log out of GreenLang dashboard
```

### 1.4 Using ML Predictions

#### Viewing Predictions

```
ML Predictions Panel
+-----------------------------------------+
| Temperature Forecast - Zone 1           |
+-----------------------------------------+
| Time       | Prediction | Confidence    |
|------------|------------|---------------|
| +15 min    | 849C       | 97%           |
| +30 min    | 852C       | 94%           |
| +1 hour    | 858C       | 89%           |
| +2 hours   | 865C       | 82%           |
| +4 hours   | 871C       | 73%           |
+-----------------------------------------+
| [!] Prediction: Temperature will exceed |
|     warning threshold in ~3 hours       |
+-----------------------------------------+
| Recommended Action:                     |
| - Reduce fuel input by 3-5%             |
| - Increase cooling water flow           |
+-----------------------------------------+
```

#### Acting on Predictions

| Confidence | Timeframe | Action |
|------------|-----------|--------|
| > 90% | < 30 min | Take immediate action |
| 80-90% | 30-60 min | Prepare for action, monitor closely |
| 70-80% | 1-2 hours | Increased monitoring, plan intervention |
| < 70% | > 2 hours | Note for awareness, standard monitoring |

---

## Part 2: Alarm Response Procedures

### 2.1 Alarm System Overview

GreenLang implements ISA 18.2 compliant alarm management:

#### Alarm Priorities

| Priority | Color | Response Time | Description |
|----------|-------|---------------|-------------|
| Critical | Red | Immediate | Safety-critical, requires immediate action |
| High | Orange | 5 minutes | Significant process impact |
| Medium | Yellow | 15 minutes | Moderate impact, timely response needed |
| Low | Blue | 30 minutes | Minor issue, schedule response |
| Advisory | Gray | Informational | No action required |

#### Alarm States

```
Alarm Lifecycle:

  +----------+    +------------+    +----------+
  |  CLEAR   |--->|   ACTIVE   |--->|  RETURN  |
  +----------+    +------+-----+    +-----+----+
                         |               |
                         v               |
                  +------+-----+         |
                  | ACKNOWLEDGED|<-------+
                  +------+-----+
                         |
                         v
                  +------+-----+
                  |   CLEAR    |
                  +------------+
```

### 2.2 Alarm Response Workflow

#### Step 1: Acknowledge

```
ALARM ACKNOWLEDGEMENT

When an alarm activates:

1. Click on the alarm notification
2. Review alarm details:
   - Parameter: Zone 1 Temperature
   - Value: 875C
   - Limit: 870C (High)
   - Time: 14:32:15

3. Click [Acknowledge] button
4. Alarm status changes from ACTIVE to ACKNOWLEDGED
```

#### Step 2: Diagnose

```
ALARM DIAGNOSIS

1. Review related parameters:
   - Fuel flow rate: 105% of normal
   - Air flow rate: 98% of normal
   - Product throughput: Normal

2. Check ML explanation:
   "Temperature increase caused primarily by
    elevated fuel flow rate. Fuel valve may
    be stuck or controller setpoint incorrect."

3. Review recent events:
   - 14:25 - Operator adjusted fuel setpoint
   - 14:28 - Fuel flow increased
   - 14:32 - Temperature alarm activated
```

#### Step 3: Take Action

```
CORRECTIVE ACTIONS

Based on diagnosis, take appropriate action:

Option A: Adjust Process Parameter
  1. Navigate to fuel flow control
  2. Reduce setpoint by 5%
  3. Monitor temperature response
  4. Document action in operator log

Option B: Notify Maintenance
  1. Create maintenance ticket
  2. Priority: HIGH
  3. Description: "Fuel valve potentially stuck"
  4. Notify shift supervisor

Option C: Escalate
  1. If unable to diagnose
  2. Contact shift supervisor
  3. Document escalation
```

#### Step 4: Verify and Document

```
VERIFICATION AND DOCUMENTATION

1. Monitor parameter return to normal
   - Temperature: 875C -> 865C -> 855C -> 850C
   - Verify stable for 15+ minutes

2. Clear alarm when condition resolved
   - Click [Clear] when parameter normal

3. Document in operator log:
   "14:32 - Zone 1 high temperature alarm (875C)
    14:35 - Acknowledged, diagnosed as fuel flow high
    14:38 - Reduced fuel setpoint by 5%
    14:55 - Temperature returned to normal (850C)
    14:58 - Alarm cleared"
```

### 2.3 Common Alarm Scenarios

#### Scenario 1: High Temperature Alarm

**Alarm:** Zone temperature exceeds high limit

**Possible Causes:**
- Fuel flow too high
- Cooling system malfunction
- Product load reduction
- Control system fault

**Response Procedure:**

```
HIGH TEMPERATURE RESPONSE

1. IMMEDIATE (< 1 minute):
   - Acknowledge alarm
   - Check if safety interlock activated
   - Assess severity and trend

2. DIAGNOSE (1-3 minutes):
   - Check fuel flow rate
   - Check cooling water flow
   - Check product throughput
   - Review ML explanation

3. ACT (based on cause):
   a. High fuel flow:
      - Reduce fuel setpoint
      - Check fuel valve operation
   b. Low cooling:
      - Increase cooling water
      - Check pump operation
   c. Low throughput:
      - Adjust production rate
      - Or reduce heat input

4. MONITOR:
   - Watch temperature trend
   - Verify rate of decrease
   - Expect normal in 15-30 minutes

5. ESCALATE IF:
   - Temperature continues rising
   - Unable to identify cause
   - Safety system activates
```

#### Scenario 2: Equipment Degradation Alert

**Alarm:** ML predicts equipment degradation

**Response Procedure:**

```
EQUIPMENT DEGRADATION RESPONSE

1. REVIEW PREDICTION:
   - Equipment: Furnace burner #3
   - Health Score: 65% (was 85% last week)
   - Predicted failure: 7-14 days
   - Confidence: 78%

2. GATHER INFORMATION:
   - Check operating parameters
   - Review maintenance history
   - Note any unusual observations

3. CREATE MAINTENANCE REQUEST:
   - Priority: HIGH
   - Type: Predictive maintenance
   - Description: Include ML prediction
   - Attach relevant data

4. IMPLEMENT INTERIM MEASURES:
   - Increase monitoring frequency
   - Set up additional alerts
   - Consider reduced load if appropriate

5. DOCUMENT:
   - Log prediction and actions
   - Track equipment performance
   - Update shift handover
```

#### Scenario 3: Anomaly Detection Alert

**Alarm:** ML detects anomalous behavior

**Response Procedure:**

```
ANOMALY RESPONSE

1. REVIEW ANOMALY DETAILS:
   - Type: Multivariate anomaly
   - Score: 0.73 (threshold: 0.5)
   - Contributing factors:
     * Temperature variance: High
     * Fuel/air ratio: Unusual
     * Cycle time: Irregular

2. INVESTIGATE:
   - Compare to historical patterns
   - Check for external factors
   - Review recent changes

3. CLASSIFY ANOMALY:
   a. Known cause (acceptable):
      - Document reason
      - Suppress future alerts if appropriate
   b. Unknown cause (investigate):
      - Increase monitoring
      - Notify engineering
   c. Process issue (act):
      - Take corrective action
      - Document resolution

4. FEEDBACK TO ML:
   - Confirm or reject anomaly
   - Improves future detection
```

### 2.4 Alarm Management Best Practices

#### Do's

- **DO** acknowledge alarms promptly
- **DO** investigate all alarms, even if they clear quickly
- **DO** document your actions and findings
- **DO** ask for help when uncertain
- **DO** provide feedback on nuisance alarms

#### Don'ts

- **DON'T** ignore or suppress alarms without investigation
- **DON'T** leave alarms acknowledged indefinitely
- **DON'T** disable safety interlocks
- **DON'T** assume alarms are false without verification
- **DON'T** make multiple changes simultaneously

---

## Part 3: Troubleshooting Common Issues

### 3.1 Troubleshooting Methodology

Follow the systematic PACE methodology:

```
P - Problem Definition
    What exactly is wrong?
    When did it start?
    What changed recently?

A - Analysis
    Gather data and evidence
    Review logs and trends
    Check ML explanations

C - Correction
    Implement fix
    Verify resolution
    Document solution

E - Evaluation
    Prevent recurrence
    Update procedures
    Share lessons learned
```

### 3.2 Common Issues and Solutions

#### Issue 1: Temperature Oscillation

**Symptoms:**
- Temperature swings above and below setpoint
- Regular, repetitive pattern
- Cannot maintain stable temperature

**Diagnostic Steps:**

```
TEMPERATURE OSCILLATION DIAGNOSIS

1. Review oscillation pattern:
   - Period: 5 minutes (typical PID issue)
   - Amplitude: +/- 15C
   - Shape: Sinusoidal

2. Check control parameters:
   - Current PID tuning
   - Compare to baseline
   - Check for recent changes

3. Check mechanical systems:
   - Valve operation
   - Sensor response time
   - Process delays
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| PID gains too high | Reduce proportional and/or integral gain |
| Valve sticking | Request maintenance, check air supply |
| Sensor lag | Check sensor condition, replace if needed |
| Process change | Re-tune PID for new conditions |

#### Issue 2: Consistently Off-Setpoint

**Symptoms:**
- Temperature stable but offset from setpoint
- Offset persists despite control action
- No oscillation

**Diagnostic Steps:**

```
OFFSET DIAGNOSIS

1. Verify measurements:
   - Compare multiple sensors
   - Check calibration dates
   - Perform spot check with portable device

2. Check control system:
   - Verify setpoint is correct
   - Check for manual mode
   - Review control output

3. Check process conditions:
   - Has load changed?
   - Are all heat sources active?
   - Is cooling operating normally?
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Sensor drift | Recalibrate or replace sensor |
| Insufficient heat input | Check burner operation, fuel supply |
| Excessive heat loss | Check insulation, close doors/hatches |
| Changed load | Adjust process parameters |

#### Issue 3: ML Predictions Inaccurate

**Symptoms:**
- Predictions consistently wrong
- Confidence scores declining
- Alert fatigue from false alarms

**Diagnostic Steps:**

```
ML ACCURACY DIAGNOSIS

1. Check data quality:
   - Are sensors reporting correctly?
   - Any gaps in data collection?
   - Has data format changed?

2. Review recent changes:
   - Process modifications?
   - New materials?
   - Operating condition changes?

3. Assess model status:
   - Last training date
   - Data drift indicators
   - Model health metrics
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Data quality issues | Fix sensor/communication issues |
| Process changes | Request model retraining |
| Concept drift | Enable auto-retraining |
| Configuration error | Verify model parameters |

#### Issue 4: Dashboard Not Updating

**Symptoms:**
- Values appear frozen
- Timestamps not advancing
- Alarms not appearing

**Diagnostic Steps:**

```
DASHBOARD UPDATE DIAGNOSIS

1. Check connectivity:
   - Network status indicator
   - Try refreshing page
   - Check other systems

2. Verify data source:
   - Is data being collected?
   - Check OPC/Modbus status
   - Verify edge gateway

3. Check browser:
   - Clear cache
   - Try different browser
   - Check for errors (F12)
```

**Common Causes and Solutions:**

| Cause | Solution |
|-------|----------|
| Network disconnection | Contact IT, check cables |
| Data source down | Restart edge gateway |
| Browser issue | Refresh, clear cache, restart browser |
| Server issue | Contact system administrator |

### 3.3 Escalation Procedures

#### When to Escalate

Escalate immediately if:
- Safety system has activated
- Unable to diagnose within 15 minutes
- Multiple related alarms occurring
- Equipment damage is possible
- Regulatory compliance is at risk

#### Escalation Contacts

| Level | Contact | Response Time |
|-------|---------|---------------|
| Level 1 | Shift Supervisor | 5 minutes |
| Level 2 | Process Engineer | 15 minutes |
| Level 3 | Plant Manager | 30 minutes |
| Level 4 | Emergency Response | Immediate |

#### Escalation Information

When escalating, provide:

```
ESCALATION REPORT TEMPLATE

1. Summary:
   "High temperature alarm in Zone 1, unable to control"

2. Timeline:
   - 14:30: Alarm activated
   - 14:35: Acknowledged, began diagnosis
   - 14:45: Reduced fuel flow, no improvement
   - 14:50: Escalating

3. Current Status:
   - Temperature: 890C (setpoint 850C)
   - Trend: Still rising slowly
   - Fuel flow: Reduced to 90%

4. Actions Taken:
   - Acknowledged alarm
   - Checked fuel system
   - Reduced fuel setpoint
   - Verified cooling system operating

5. Suspected Cause:
   - Possibly stuck fuel valve

6. Assistance Needed:
   - Maintenance to check valve
   - Engineering guidance on interim measures
```

---

## Part 4: Performance Optimization Tips

### 4.1 Energy Efficiency Optimization

#### Understanding Energy Metrics

```
ENERGY EFFICIENCY DASHBOARD

Current Shift Performance:

  Specific Energy: 485 kWh/ton
  Target:          450 kWh/ton
  Variance:        +7.8%

  +----------------------------------+
  |         Efficiency Trend         |
  |   500|    *                      |
  |   480|  *   *  *                 |
  |   460|*          *  *            |
  |   440|--------------*--*--*----- | Target
  |   420|                           |
  |      +---------------------------+
  |       06:00  09:00  12:00  15:00 |
  +----------------------------------+

  Top Energy Losses:
  1. Flue gas (23%)
  2. Wall losses (12%)
  3. Opening losses (8%)
  4. Cooling water (5%)
```

#### Energy Optimization Strategies

**Strategy 1: Minimize Opening Losses**

```
OPENING LOSS REDUCTION

Impact: 5-10% energy savings

Actions:
- Close doors/hatches promptly
- Install air curtains at openings
- Schedule operations to minimize openings
- Use viewing ports instead of doors

Monitoring:
- Track door open events
- Compare energy during open vs. closed
- Set alerts for extended open times
```

**Strategy 2: Optimize Combustion**

```
COMBUSTION OPTIMIZATION

Impact: 3-8% energy savings

Key Parameters:
- O2 in flue gas: Target 2-3%
- CO in flue gas: Target < 100 ppm
- Fuel/air ratio: Maintain stoichiometric + 5-10%

Actions:
- Monitor flue gas composition
- Adjust air dampers regularly
- Clean burners as scheduled
- Report unusual flame characteristics

GreenLang Support:
- ML monitors combustion efficiency
- Alerts on suboptimal conditions
- Provides adjustment recommendations
```

**Strategy 3: Reduce Heat Losses**

```
HEAT LOSS REDUCTION

Impact: 2-5% energy savings

Actions:
- Report damaged insulation immediately
- Keep seals and gaskets in good condition
- Ensure cooling water flow is appropriate
- Don't overcool - maintain design temps

Signs of Excessive Loss:
- Hot spots on equipment exterior
- Cooling water outlet too hot
- Higher than normal fuel consumption
```

### 4.2 Process Quality Optimization

#### Using ML for Quality

```
QUALITY PREDICTION PANEL

Current Batch Prediction:
  Quality Score: 94%

  Key Factors:
  +-------------------------+--------+
  | Factor                  | Impact |
  +-------------------------+--------+
  | Temperature uniformity  |  +5%   |
  | Heating rate           |  +3%   |
  | Holding time           |  +2%   |
  | Cooling rate           |  -1%   |
  +-------------------------+--------+

  Recommendation:
  "Slightly reduce cooling rate in Phase 3
   for optimal quality. Current: 15C/min
   Recommended: 12C/min"
```

#### Quality Optimization Tips

| Area | Best Practice | Benefit |
|------|---------------|---------|
| Temperature Uniformity | Monitor multiple zones, balance heat | Consistent quality |
| Heating Rate | Follow recipe, avoid rushing | Proper metallurgy |
| Holding Time | Maintain prescribed soak time | Complete transformation |
| Atmosphere | Monitor and control gas composition | Surface quality |

### 4.3 Using GreenLang Optimization Features

#### Setpoint Recommendations

```
SETPOINT OPTIMIZATION

GreenLang ML has identified potential improvements:

Current Operating Point:
  Zone 1: 850C    Zone 2: 920C    Zone 3: 880C
  Energy: 485 kWh/ton

Recommended Operating Point:
  Zone 1: 845C    Zone 2: 915C    Zone 3: 885C
  Predicted Energy: 462 kWh/ton
  Predicted Quality: No change

  [Apply Recommendations]  [Dismiss]  [Details]
```

#### Automated vs. Manual Optimization

| Mode | Description | When to Use |
|------|-------------|-------------|
| Manual | Operator reviews and applies recommendations | Initial adoption, unusual conditions |
| Semi-Auto | ML suggests, operator approves | Standard operations |
| Full Auto | ML adjusts within limits automatically | Proven, stable processes |

### 4.4 Continuous Improvement

#### Providing Feedback

Your feedback improves GreenLang ML:

```
FEEDBACK MECHANISMS

1. Prediction Feedback:
   After ML prediction:
   - Was it accurate? [Yes] [No]
   - Actual outcome: [_____]

2. Recommendation Feedback:
   After implementing suggestion:
   - Did it improve performance? [Yes] [No] [Partially]
   - Comments: [_____]

3. Anomaly Feedback:
   When anomaly detected:
   - Was this a real anomaly? [Yes] [No]
   - What was the cause? [_____]
```

#### Shift Performance Reviews

Weekly performance review checklist:

```
WEEKLY PERFORMANCE REVIEW

[ ] Review energy efficiency trends
[ ] Compare to targets and benchmarks
[ ] Identify best and worst performing shifts
[ ] Analyze alarm frequency and response times
[ ] Review ML prediction accuracy
[ ] Document improvement opportunities
[ ] Share findings with team
```

---

## Summary

In this module, you learned:

1. **Day-to-Day Operations:** Dashboard navigation, monitoring procedures, shift routines
2. **Alarm Response:** ISA 18.2 alarm management, response procedures, common scenarios
3. **Troubleshooting:** PACE methodology, common issues, escalation procedures
4. **Optimization:** Energy efficiency, quality optimization, continuous improvement

---

## Knowledge Check

### Questions

1. What are the five alarm priority levels in GreenLang?
2. What is the PACE troubleshooting methodology?
3. What should you do if temperature is oscillating around setpoint?
4. When should you escalate an issue?
5. How can you provide feedback to improve ML predictions?

### Practical Exercises

1. **Alarm Response:** Practice responding to a simulated high temperature alarm
2. **Troubleshooting:** Diagnose a simulated offset issue
3. **Optimization:** Analyze energy data and identify improvement opportunities
4. **Documentation:** Complete a shift handover log for a scenario

---

## Next Steps

After completing this module:

- **Practice** in the sandbox environment
- **Review** site-specific procedures
- **Shadow** experienced operators
- **Prepare** for operator certification

---

*Module Version: 1.0.0*
*Last Updated: December 2025*
