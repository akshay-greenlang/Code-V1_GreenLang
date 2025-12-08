# GL-020 Economizer Module Specification

**Module Name:** EconoPulse
**Module ID:** GL-020
**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

The GL-020 Economizer Module provides real-time economizer performance monitoring, fouling analysis, cleaning alert generation, and efficiency loss tracking. It maximizes feedwater preheat recovery while preventing tube damage.

## Key Capabilities

| Capability | Description |
|------------|-------------|
| Performance Monitoring | Real-time U-value tracking |
| Fouling Analysis | Rf calculation and trending |
| Cleaning Alerts | Threshold and predictive alerts |
| Efficiency Tracking | Heat recovery efficiency |
| Soot Blower Integration | Cleaning feedback loop |
| Energy Loss Quantification | MMBtu and $ impact |

## Monitored Parameters

### Temperature Points
| Parameter | Unit | Source |
|-----------|------|--------|
| Feedwater Inlet | F/C | RTD |
| Feedwater Outlet | F/C | RTD |
| Flue Gas Inlet | F/C | Thermocouple |
| Flue Gas Outlet | F/C | Thermocouple |
| Tube Wall (optional) | F/C | Thermocouple |

### Flow & Pressure
| Parameter | Unit | Source |
|-----------|------|--------|
| Feedwater Flow | GPM, lb/hr | Orifice, Vortex |
| Flue Gas Flow | ACFM | Stack flow |
| Pressure Drop | in H2O | DP transmitter |

## Key Calculations

### Heat Transfer Coefficient
```
U = Q / (A x LMTD)

Q = m_dot x Cp x dT (water side)

LMTD = (dT1 - dT2) / ln(dT1/dT2)
```

### Fouling Resistance
```
Rf = (1/U_actual) - (1/U_clean)

Rf_threshold = 0.0003 m2K/W (typical)
```

### Effectiveness
```
Effectiveness = (T_water_out - T_water_in) / (T_gas_in - T_water_in)

Design: 70-85%
Fouled: <60% triggers alert
```

### Energy Loss
```
Loss_MMBtu = (Q_design - Q_actual) x Hours / 1E6

Loss_$ = Loss_MMBtu x Fuel_Cost
```

## Cleaning Alert Levels

| Level | Condition | Action |
|-------|-----------|--------|
| Normal | Rf < 0.0001 | Normal operation |
| Monitor | 0.0001 < Rf < 0.0002 | Increase soot blowing |
| Warning | 0.0002 < Rf < 0.0003 | Schedule cleaning |
| Critical | Rf > 0.0003 | Clean immediately |

## Soot Blower Integration

### Optimization Algorithm
```
Soot Blower Frequency = f(Rf_rate, Fuel_cost, Cleaning_cost)

Optimize for minimum total cost:
Total_Cost = Energy_Loss + Cleaning_Cost + Tube_Wear
```

### Feedback Metrics
| Metric | Use |
|--------|-----|
| Pre-blow Rf | Baseline |
| Post-blow Rf | Effectiveness |
| Recovery % | Cleaning quality |
| Frequency | Optimization input |

## Performance Dashboard

```
+------------------------------------------------------------------+
|                    ECONOMIZER PERFORMANCE                         |
+------------------------------------------------------------------+
| Equipment: Economizer-01    | Status: MONITOR | Last: 10:45:32   |
+------------------------------------------------------------------+
| Metric              | Design  | Actual  | % Design | Trend       |
|---------------------|---------|---------|----------|-------------|
| Duty (MMBtu/hr)     | 25.0    | 22.5    | 90%      | Declining   |
| U-value (Btu/ft2hF) | 15.0    | 12.8    | 85%      | Declining   |
| Effectiveness       | 82%     | 75%     | 91%      | Stable      |
| dP (in H2O)         | 2.0     | 2.8     | 140%     | Rising      |
| Rf (m2K/W)          | 0       | 0.00018 | -        | Rising      |
+------------------------------------------------------------------+
| Energy Loss: 2.5 MMBtu/hr = $7.50/hr = $66,000/year              |
| Recommendation: Schedule cleaning within 2 weeks                  |
+------------------------------------------------------------------+
```

## Integrations

- Primary: BoilerOptimizer (GL-002)
- Supporting: GL-001 (Orchestrator), GL-018 (Flue Gas)
- SCADA: Real-time data acquisition

## Economizer Types

| Type | Application |
|------|-------------|
| Cast Iron | Low pressure, small boilers |
| Steel Finned | Package boilers |
| Carbon Steel | Industrial boilers |
| Stainless Steel | Condensing economizers |
| Extended Surface | High efficiency |

## Pricing

| Tier | Economizers | Add-on Price |
|------|-------------|--------------|
| Basic | 1-3 | $1,200/month |
| Standard | 4-10 | $2,500/month |
| Enterprise | 10+ | Custom |

---

*GL-020 EconoPulse - Maximize Feedwater Heat Recovery*
