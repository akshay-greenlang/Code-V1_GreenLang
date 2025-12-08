# GL-008 Steam Trap Module Specification

**Module Name:** TrapCatcher
**Module ID:** GL-008
**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

The GL-008 Steam Trap Module provides automated detection and diagnosis of steam trap failures across industrial facilities. It analyzes acoustic signatures, temperature differentials, and IR imaging to identify failed traps, prioritize maintenance, and quantify cost savings.

## Key Capabilities

| Capability | Description |
|------------|-------------|
| Acoustic Analysis | Ultrasonic leak detection (20-100 kHz) |
| Temperature Analysis | Inlet/outlet differential |
| IR Imaging | Thermal pattern analysis |
| Failure Detection | Pattern matching for failure modes |
| Priority Scoring | 1-5 maintenance priority scale |
| Energy Loss | Steam and cost quantification |
| Work Order Generation | CMMS integration |

## Detection Methods

### 1. Acoustic Analysis
```
Ultrasonic Signature Analysis:
- Frequency range: 20-100 kHz
- Sampling rate: 250 kHz
- FFT analysis for pattern matching
- dB threshold comparison
```

### 2. Temperature Differential
```
Normal Operation:
- Inlet: Steam temperature
- Outlet: Near steam temp (slight drop)

Failed Open (Blow-Through):
- Inlet: Steam temperature
- Outlet: Steam temperature (no drop)

Failed Closed (Blocked):
- Inlet: Steam temperature
- Outlet: Significantly cooler
```

### 3. IR Thermal Imaging
```
Pattern Analysis:
- Hot plume detection (blow-through)
- Cold spots (blocked)
- Normal heat signature reference
```

## Failure Modes Detected

| Failure Mode | Detection Method | Priority |
|--------------|------------------|----------|
| Blow-Through | Acoustic + Temp | Critical |
| Blocked/Stuck Closed | Temp differential | High |
| Leaking | Acoustic + Temp | Medium |
| Cold/Not Operating | Temp | Low |
| Intermittent | Pattern analysis | Medium |

## Steam Loss Calculation

### Energy Loss (Failed Open Trap)
```
Steam_Loss (lb/hr) = C x sqrt(dP) x Orifice_Area

Energy_Loss (Btu/hr) = Steam_Loss x h_fg

Cost_Loss ($/hr) = Steam_Loss x Steam_Cost ($/1000 lb)
```

### Typical Loss by Orifice Size

| Orifice (in) | dP (psig) | Steam Loss (lb/hr) | Annual Cost ($) |
|--------------|-----------|--------------------|-----------------
| 1/8 | 100 | 26 | $2,280 |
| 3/16 | 100 | 59 | $5,170 |
| 1/4 | 100 | 105 | $9,200 |
| 3/8 | 100 | 237 | $20,800 |
| 1/2 | 100 | 421 | $36,900 |

*Based on $10/1000 lb steam cost, 8760 hr/yr*

## Priority Scoring (1-5)

| Priority | Criteria | Action Timeline |
|----------|----------|-----------------|
| 5 - Critical | Blow-through, large orifice | Immediate |
| 4 - High | Blow-through, medium orifice | Within 1 week |
| 3 - Medium | Leaking or intermittent | Within 1 month |
| 2 - Low | Minor issues, small traps | Next scheduled PM |
| 1 - Monitor | Aging but functional | Continue monitoring |

## Survey Report Output

```
+------------------------------------------------------------------+
|                    STEAM TRAP SURVEY REPORT                       |
+------------------------------------------------------------------+
| Facility: Chemical Plant Alpha        | Date: December 1, 2025   |
| Traps Surveyed: 450                   | Surveyor: Auto-GL-008    |
+------------------------------------------------------------------+

SUMMARY
-------
| Status          | Count | Percentage | Annual Loss  |
|-----------------|-------|------------|--------------|
| Operating       | 382   | 85%        | -            |
| Failed Open     | 41    | 9%         | $285,000     |
| Failed Closed   | 18    | 4%         | $45,000*     |
| Leaking         | 9     | 2%         | $32,000      |
|-----------------|-------|------------|--------------|
| TOTAL           | 450   | 100%       | $362,000     |

*Failed closed loss = backup heat/rework costs

TOP 10 PRIORITY REPAIRS
-----------------------
| Trap ID    | Location       | Failure    | Size | Annual Loss | Priority |
|------------|----------------|------------|------|-------------|----------|
| ST-101     | Reactor 1      | Blow-thru  | 1/2" | $36,900     | 5        |
| ST-047     | Distillation   | Blow-thru  | 3/8" | $20,800     | 5        |
| ST-203     | Dryer 2        | Blow-thru  | 3/8" | $20,800     | 5        |
| ...        | ...            | ...        | ...  | ...         | ...      |

RECOMMENDED ACTIONS
-------------------
1. Replace 10 critical failed traps immediately ($180K annual savings)
2. Schedule repair of 31 remaining failed traps within 30 days
3. Implement continuous monitoring for high-priority locations
```

## CMMS Integration

| System | Integration Method |
|--------|-------------------|
| SAP PM | RFC/BAPI |
| Maximo | REST API |
| eMaint | REST API |
| Fiix | REST API |
| Maintenance Connection | API |

### Work Order Fields
```
{
  "work_order_type": "Corrective Maintenance",
  "asset_id": "ST-101",
  "location": "Building A, Line 3",
  "priority": "Critical",
  "failure_mode": "Blow-Through",
  "estimated_steam_loss": "421 lb/hr",
  "estimated_annual_cost": "$36,900",
  "recommended_replacement": "Spirax Sarco TD52",
  "parts_required": ["TD52 1/2\" NPT", "Gasket kit"],
  "estimated_labor_hours": 2.0,
  "detected_by": "GL-008 TrapCatcher",
  "detection_timestamp": "2025-12-01T10:45:32Z",
  "provenance_hash": "sha256:abc123..."
}
```

## Standards Compliance

| Standard | Description |
|----------|-------------|
| ISO 6552 | Automatic steam traps - Definitions |
| ISO 7841 | Steam traps - Steam loss determination |
| ISO 7842 | Steam traps - Selection |
| ASME B31.1 | Power Piping |
| TES Methodology | Trap Energy Savings |

## Integrations

- Primary: ThermalCommand (GL-001)
- Supporting: GL-003 (Steam System)
- Equipment: Ultrasonic detectors, thermal cameras
- SCADA: Real-time monitoring

## Pricing

| Tier | Traps Surveyed | Add-on Price |
|------|----------------|--------------|
| Basic | Up to 200 | $1,500/month |
| Standard | 200-500 | $3,000/month |
| Enterprise | 500+ | Custom |

### Survey Services

| Service | Price |
|---------|-------|
| Initial Survey (per trap) | $15 |
| Annual Re-survey (per trap) | $8 |
| Continuous Monitoring Setup | $5,000 |

---

*GL-008 TrapCatcher - Find the Leaks, Save the Steam*
