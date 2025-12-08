# BoilerOptimizer (GL-002/GL-018) Product Specification

**Product Name:** BoilerOptimizer
**Module IDs:** GL-002 (Core), GL-018 (Flue Gas Analysis)
**Codenames:** BoilerEfficiencyOptimizer, FLUEFLOW
**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** December 2025

---

## Executive Summary

### Product Vision

BoilerOptimizer is a comprehensive boiler efficiency optimization platform that combines advanced combustion control with real-time flue gas analysis. By integrating GL-002 (BoilerEfficiencyOptimizer) with GL-018 (FLUEFLOW Flue Gas Analyzer), the platform delivers 10-20% fuel savings through continuous optimization of combustion parameters, air-fuel ratios, and emissions control.

### Value Proposition

| Challenge | BoilerOptimizer Solution | Business Impact |
|-----------|--------------------------|-----------------|
| Suboptimal combustion | Continuous air-fuel ratio optimization | 8-15% fuel savings |
| High excess air | Real-time O2 trim control | 3-5% efficiency gain |
| Manual tuning | Automated combustion tuning | 90% reduction in tuning time |
| Emissions compliance risk | Proactive compliance monitoring | Zero permit violations |
| Unplanned shutdowns | Predictive fault detection | 50% fewer outages |

### Target Market

- **Industries:** Power generation, process heating, district heating, food processing, chemicals
- **Boiler Types:** Fire-tube, water-tube, steam generators, hot water boilers
- **Capacity Range:** 10 MMBtu/hr to 500+ MMBtu/hr
- **Fuel Types:** Natural gas, fuel oil, coal, biomass, hydrogen blends, multi-fuel

---

## Combustion Optimization Capabilities

### 1. Air-Fuel Ratio Optimization

Real-time optimization of combustion air to fuel ratio for maximum efficiency.

**Optimization Algorithm:**
```
Optimal Excess Air = f(Fuel Type, Load, Ambient Conditions, Equipment Age)

For Natural Gas:
- Full load (>80%): 10-15% excess air
- Partial load (50-80%): 15-20% excess air
- Low load (<50%): 20-30% excess air

O2 Setpoint = (Excess Air x 20.9) / (100 + Excess Air)
```

**Control Actions:**
| Parameter | Range | Resolution | Response Time |
|-----------|-------|------------|---------------|
| O2 Setpoint | 1.0-8.0% | 0.1% | <30 seconds |
| Air Damper Position | 0-100% | 0.5% | <15 seconds |
| Fuel Valve Position | 0-100% | 0.1% | <10 seconds |
| Burner Firing Rate | 20-100% | 1% | <60 seconds |

### 2. Flue Gas Analysis (GL-018 FLUEFLOW)

Continuous monitoring and analysis of flue gas composition.

**Monitored Parameters:**
| Parameter | Unit | Range | Accuracy |
|-----------|------|-------|----------|
| Oxygen (O2) | % | 0-25% | +/- 0.1% |
| Carbon Dioxide (CO2) | % | 0-20% | +/- 0.2% |
| Carbon Monoxide (CO) | ppm | 0-5000 | +/- 5 ppm |
| Nitrogen Oxides (NOx) | ppm | 0-2000 | +/- 2 ppm |
| Sulfur Dioxide (SO2) | ppm | 0-3000 | +/- 5 ppm |
| Stack Temperature | F | 200-1000 | +/- 2 F |
| Opacity | % | 0-100% | +/- 2% |

**Calculated Parameters:**
- Combustion efficiency (direct and indirect methods)
- Excess air percentage
- Heat loss breakdown (stack, radiation, blowdown)
- Fuel-specific emission factors

### 3. Combustion Efficiency Calculation

Industry-standard efficiency calculations per ASME PTC 4.1.

**Direct Method:**
```
Efficiency = (Steam Output Energy / Fuel Input Energy) x 100%

Where:
Steam Output Energy = Steam Flow x (Steam Enthalpy - Feedwater Enthalpy)
Fuel Input Energy = Fuel Flow x Higher Heating Value (HHV)
```

**Indirect (Heat Loss) Method:**
```
Efficiency = 100% - Sum of All Losses

Losses Include:
- Dry Flue Gas Loss (typically 3-8%)
- Moisture in Fuel Loss (0-3%)
- Moisture from Combustion Loss (4-6%)
- Radiation and Convection Loss (0.5-2%)
- Blowdown Loss (0.5-3%)
- Unburned Carbon Loss (0-2%)
```

### 4. Burner Management

Advanced burner control and optimization features.

**Features:**
- Multi-burner coordination and balancing
- Automatic light-off and shutdown sequences
- Flame monitoring and stability assessment
- Turndown ratio optimization
- Low-NOx burner control strategies

**Burner Health Indicators:**
| Indicator | Healthy | Warning | Critical |
|-----------|---------|---------|----------|
| Flame Stability Index | >0.9 | 0.7-0.9 | <0.7 |
| Ignition Success Rate | >99% | 95-99% | <95% |
| CO at Setpoint | <50 ppm | 50-200 ppm | >200 ppm |
| Flame Scanner Signal | Strong | Marginal | Weak |

---

## Efficiency Improvement Metrics

### Typical Efficiency Gains by Source

| Optimization Area | Efficiency Gain | Annual Savings (100 MMBtu/hr boiler) |
|-------------------|-----------------|--------------------------------------|
| Excess Air Reduction | 1-3% | $50,000 - $150,000 |
| O2 Trim Control | 0.5-1.5% | $25,000 - $75,000 |
| Stack Temperature Optimization | 1-2% | $50,000 - $100,000 |
| Radiation Loss Reduction | 0.2-0.5% | $10,000 - $25,000 |
| Blowdown Heat Recovery | 0.5-1% | $25,000 - $50,000 |
| Combustion Tuning | 1-2% | $50,000 - $100,000 |
| **Total Potential** | **4-10%** | **$210,000 - $500,000** |

### Performance Benchmarks

| Metric | Baseline | With BoilerOptimizer | Improvement |
|--------|----------|---------------------|-------------|
| Thermal Efficiency | 78-82% | 86-92% | +6-12% |
| Excess O2 | 4-8% | 2-4% | -50% |
| CO Emissions | 100-500 ppm | <50 ppm | -80% |
| NOx Emissions | 100-200 ppm | 30-80 ppm | -60% |
| Stack Temperature | 400-500F | 300-350F | -100F |
| Fuel Cost per MMBtu | $4.50 | $3.80 | -15% |

---

## Emissions Reduction Features

### Continuous Emissions Monitoring

Real-time tracking of all regulated pollutants.

**Compliance Frameworks:**
| Framework | Coverage | Reporting |
|-----------|----------|-----------|
| EPA 40 CFR Part 60 | NSPS for boilers | Automated |
| EPA 40 CFR Part 63 | NESHAP/MACT | Automated |
| EU IED (2010/75/EU) | BAT-AEL limits | Automated |
| State/Local Permits | Site-specific | Configurable |

### Low-NOx Combustion Strategies

| Strategy | NOx Reduction | Efficiency Impact |
|----------|---------------|-------------------|
| Low-Excess Air (LEA) | 10-20% | +1-2% efficiency |
| Staged Combustion | 30-50% | Neutral |
| Flue Gas Recirculation (FGR) | 40-60% | -0.5% efficiency |
| Water/Steam Injection | 50-70% | -2-3% efficiency |
| Low-NOx Burners | 30-70% | +1% efficiency |
| SCR Integration | 80-95% | -0.5% efficiency |

### Emissions Reporting

Automated generation of regulatory reports:
- Daily emissions summaries
- Monthly compliance reports
- Quarterly excess emissions reports
- Annual emissions inventories
- Real-time permit limit tracking

---

## Integration with DCS/SCADA

### Supported Control Systems

| Vendor | System | Protocol | Integration Level |
|--------|--------|----------|-------------------|
| Honeywell | Experion PKS | OPC UA, Modbus | Full |
| ABB | 800xA, Symphony | OPC UA, IEC 61850 | Full |
| Emerson | DeltaV, Ovation | OPC UA, Modbus | Full |
| Siemens | PCS 7, SPPA-T3000 | OPC UA, S7 | Full |
| Yokogawa | CENTUM VP | OPC UA, Modbus | Full |
| Rockwell | PlantPAx | OPC UA, EtherNet/IP | Full |
| GE | Mark VIe | OPC UA, Modbus | Full |
| Schneider | Foxboro, Triconex | OPC UA, Modbus | Full |

### Integration Architecture

```
+------------------+      +-------------------+      +-----------------+
|   BoilerOptimizer|      |    Integration    |      |      DCS        |
|   Application    | <--> |      Gateway      | <--> |   Controller    |
+------------------+      +-------------------+      +-----------------+
        |                          |                         |
   Analytics &              Data Mapping &              Real-time
   Optimization            Protocol Translation          Control
        |                          |                         |
+------------------+      +-------------------+      +-----------------+
| Recommendations  |      |   Tag Database    |      |  Field Devices  |
| & Setpoints      |      |   (1000+ points)  |      |  (Sensors, etc) |
+------------------+      +-------------------+      +-----------------+
```

### Data Points (Typical Installation)

| Category | Points | Frequency |
|----------|--------|-----------|
| Process Variables | 200+ | 1 second |
| Control Outputs | 50+ | 1 second |
| Alarms & Events | 100+ | On change |
| Performance Calculations | 50+ | 10 seconds |
| Trend Data | 300+ | 1 minute |

---

## ROI Calculator Inputs

### Required Data for ROI Analysis

**Operational Data:**
| Parameter | Unit | Source |
|-----------|------|--------|
| Annual fuel consumption | MMBtu, therms, gallons | Fuel bills |
| Average fuel cost | $/MMBtu | Invoices |
| Annual operating hours | hours | Operations |
| Average boiler load | % | SCADA |
| Current thermal efficiency | % | Test reports |
| Steam production rate | lb/hr | Meters |

**Financial Data:**
| Parameter | Default | Range |
|-----------|---------|-------|
| Discount rate | 8% | 5-15% |
| Analysis period | 10 years | 5-20 years |
| Electricity cost | $0.08/kWh | $0.05-$0.20 |
| Carbon price | $50/ton CO2 | $25-$150 |
| Maintenance escalation | 3%/year | 2-5% |

### ROI Calculation Model

```
Annual Savings = Fuel Savings + Emissions Savings + Maintenance Savings

Where:
Fuel Savings = (Efficiency Improvement / Current Efficiency) x Annual Fuel Cost

Emissions Savings = CO2 Reduction (tons) x Carbon Price ($/ton)

Maintenance Savings = Reduced Downtime x Production Value
```

**Sample ROI Analysis (100 MMBtu/hr boiler):**

| Metric | Year 1 | Year 3 | Year 5 |
|--------|--------|--------|--------|
| Fuel Savings | $285,000 | $925,000 | $1,625,000 |
| Emissions Savings | $45,000 | $145,000 | $255,000 |
| Maintenance Savings | $35,000 | $115,000 | $200,000 |
| **Total Savings** | **$365,000** | **$1,185,000** | **$2,080,000** |
| Implementation Cost | $150,000 | - | - |
| **Net Savings** | **$215,000** | **$1,035,000** | **$1,930,000** |
| Simple Payback | 5 months | - | - |
| 5-Year ROI | - | - | 1,287% |

---

## Technical Specifications

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Processing | 4 cores, 2.5 GHz | 8 cores, 3.0 GHz |
| Memory | 8 GB RAM | 16 GB RAM |
| Storage | 100 GB SSD | 250 GB SSD |
| Network | 100 Mbps | 1 Gbps |
| OS | Windows Server 2019 / Ubuntu 20.04 | Latest LTS |

### Performance Specifications

| Metric | Specification |
|--------|---------------|
| Control loop rate | 1-10 seconds |
| Calculation cycle | <500 ms |
| Data acquisition | 1000+ points/second |
| Optimization cycle | 30-60 seconds |
| Alarm response | <1 second |
| Report generation | <30 seconds |

### Safety & Redundancy

| Feature | Description |
|---------|-------------|
| Watchdog monitoring | Automatic failsafe on communication loss |
| Limits enforcement | Hard and soft limits on all outputs |
| Operator override | Manual override capability |
| Audit logging | All changes logged with timestamp |
| Redundancy | Hot standby option available |
| Failsafe outputs | Configurable fail-safe positions |

---

## Pricing & Licensing

### Subscription Pricing

| Tier | Capacity | Monthly | Annual (15% discount) |
|------|----------|---------|----------------------|
| Small | <50 MMBtu/hr | $3,500 | $35,700 |
| Medium | 50-150 MMBtu/hr | $6,500 | $66,300 |
| Large | 150-300 MMBtu/hr | $10,000 | $102,000 |
| Enterprise | >300 MMBtu/hr | Custom | Custom |

### What's Included

**All Tiers:**
- GL-002 BoilerEfficiencyOptimizer
- GL-018 FLUEFLOW Flue Gas Analyzer
- Real-time efficiency calculations
- Combustion optimization
- Emissions monitoring
- Performance dashboards
- Standard reporting
- Email/ticket support

**Medium & Above:**
- Multi-boiler coordination
- Advanced analytics
- Custom report builder
- API access
- Phone support (12x5)

**Large & Above:**
- Unlimited boilers
- Predictive maintenance integration
- Dedicated success manager
- 24x7 support
- Quarterly business reviews

### Implementation Services

| Service | Price |
|---------|-------|
| Standard Implementation | $25,000 |
| Complex Integration | $50,000+ |
| On-site Training (2 days) | $8,000 |
| Annual Tune-up Service | $15,000/year |
| Custom Development | $200/hour |

---

## Appendix A: ASME PTC 4.1 Compliance

BoilerOptimizer fully implements the ASME PTC 4.1 standard for steam generator performance testing:

**Covered Calculations:**
- Input-output (direct) efficiency method
- Heat loss (indirect) efficiency method
- Energy balance calculations
- Uncertainty analysis
- Fuel analysis integration
- Correction factors (ambient, fuel, load)

---

## Appendix B: Emission Factor Database

Pre-loaded emission factors for common fuels (source: EPA AP-42, IPCC 2006):

| Fuel | CO2 (kg/MMBtu) | NOx (lb/MMBtu) | SO2 (lb/MMBtu) |
|------|----------------|----------------|----------------|
| Natural Gas | 53.06 | 0.10 | 0.0006 |
| Fuel Oil #2 | 73.16 | 0.14 | 0.17 |
| Fuel Oil #6 | 75.10 | 0.28 | 0.95 |
| Coal (Bituminous) | 93.30 | 0.55 | 1.20 |
| Biomass (Wood) | 93.80 | 0.22 | 0.02 |
| Propane | 62.87 | 0.09 | 0.0006 |

---

**Document Control:**
- **Author:** GreenLang Product Management
- **Approved By:** VP Product
- **Next Review:** Q2 2026

---

*BoilerOptimizer - Maximizing Combustion Efficiency, Minimizing Emissions*
