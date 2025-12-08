# GL-015 Insulation Module Specification

**Module Name:** InsulScan
**Module ID:** GL-015
**Version:** 1.0.0
**Status:** Production Ready

---

## Overview

The GL-015 Insulation Module provides production-grade pipeline orchestration for thermal insulation inspection and analysis. It uses thermal imaging to detect heat loss, assess insulation condition, and prioritize repairs.

## Pipeline Stages

| Stage | Function | Output |
|-------|----------|--------|
| Input Validation | Validate thermal images | Validated inputs |
| Image Preprocessing | Prepare images for analysis | Enhanced images |
| Thermal Analysis | Detect hotspots | Anomaly locations |
| Heat Loss Calculation | Calculate losses per location | kW and $/year |
| Degradation Assessment | Assess insulation condition | Condition score |
| Energy Quantification | Facility-wide loss | Total energy loss |
| Repair Prioritization | ROI-based ranking | Priority list |
| Economic Analysis | Cost-benefit analysis | ROI metrics |
| Report Generation | Inspection report | PDF report |

## Key Calculations

### Heat Loss (Bare Pipe)
```
Q = h x A x (T_surface - T_ambient)

Where:
- h = Heat transfer coefficient (W/m2K)
- A = Surface area (m2)
- T = Temperatures (C)
```

### Insulation Effectiveness
```
Effectiveness = (Q_bare - Q_insulated) / Q_bare x 100%
```

### Thermal Conductivity
```
Q = k x A x dT / thickness

Where:
- k = Thermal conductivity (W/mK)
- dT = Temperature difference
```

### Economic Loss
```
Annual_Loss = Q x Hours x Energy_Cost

Example:
100 kW x 8760 hr x $0.03/kWh = $26,280/year
```

## Inspection Types

| Type | Scope | Duration |
|------|-------|----------|
| Full Facility | All insulated surfaces | 2-5 days |
| Unit Survey | Specific process unit | 0.5-1 day |
| Problem Area | Known issues | 2-4 hours |
| Compliance Audit | DOE/ASHRAE standards | 1-2 days |
| Energy Assessment | Energy-focused | 1-2 days |

## Thermal Camera Integration

| Manufacturer | Models Supported |
|--------------|-----------------|
| FLIR | T-series, E-series, A-series |
| FLUKE | Ti400, Ti450, TiX series |
| Testo | 890, 885, 883 |
| Hikvision | DS-2TD series |

## Degradation Categories

| Category | Description | Priority |
|----------|-------------|----------|
| Missing | No insulation | Critical |
| Damaged | Physical damage | High |
| Wet | Moisture intrusion | High |
| Compressed | Thickness reduced | Medium |
| Aged | Normal degradation | Low |

## Integrations

- Primary: WasteHeatRecovery (GL-006)
- Supporting: GL-001 (Orchestrator)
- Thermal Cameras: HTTP/MQTT protocols

## Standards Compliance

- ASTM C680 - Heat loss calculations
- ASHRAE 90.1 - Insulation requirements
- DOE Industrial Insulation - Best practices
- 3E Plus - Economic thickness

## Pricing

| Tier | Inspections/year | Add-on Price |
|------|------------------|--------------|
| Basic | 2 | $1,500/month |
| Standard | 6 | $3,000/month |
| Unlimited | Unlimited | $5,000/month |

---

*GL-015 InsulScan - See the Heat, Save the Energy*
