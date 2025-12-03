# GL-018 FLUEFLOW

**Intelligent Flue Gas Analysis and Combustion Optimization Agent**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/greenlang/gl-018-flueflow)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Combustion Fundamentals](#combustion-fundamentals)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Integration Guides](#integration-guides)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)
- [Compliance Standards](#compliance-standards)
- [Contributing](#contributing)
- [License and Support](#license-and-support)

---

## Overview

GL-018 FLUEFLOW is an advanced AI-powered agent that analyzes flue gas composition to optimize combustion efficiency, reduce fuel consumption, minimize emissions, and ensure regulatory compliance. It provides real-time monitoring, predictive analytics, and automated control for industrial combustion systems including boilers, furnaces, heaters, and gas turbines.

### Purpose

Industrial combustion systems require precise air-fuel ratio control to:
- **Maximize Combustion Efficiency**: Optimize fuel consumption and heat transfer to minimize operating costs
- **Minimize Emissions**: Control NOx, CO, SO2, and particulate matter to meet regulatory limits
- **Prevent Equipment Damage**: Avoid high-temperature corrosion, thermal stress, and incomplete combustion
- **Ensure Safety**: Prevent dangerous conditions like flame instability, flashback, and explosive mixtures
- **Optimize Excess Air**: Balance efficiency (lower excess air) with safety and emissions (adequate excess air)

FLUEFLOW automates these complex tasks using deterministic stoichiometric calculations, industry-standard combustion formulas, and AI-powered insights.

### Target Market

- **Total Addressable Market (TAM)**: $4 Billion
- **Industries**: Power Generation, Oil & Gas, Chemical, Refining, Steel, Cement, Glass, Pulp & Paper
- **Equipment Types**: Boilers, furnaces, process heaters, gas turbines, kilns, incinerators
- **Fuel Types**: Natural gas, fuel oil, coal, biomass, hydrogen, waste fuels, mixed fuels
- **Priority**: P1 (High Priority)
- **Target Release**: Q1 2026

### Business Value

- **Fuel Savings**: 3-8% reduction in fuel consumption through optimal air-fuel ratio control
- **Emissions Reduction**: 20-40% reduction in NOx emissions through combustion tuning
- **Efficiency Improvement**: 2-5% increase in combustion efficiency
- **Carbon Savings**: 5-10% reduction in CO2 emissions (fuel savings)
- **Maintenance Cost Reduction**: 30% reduction in high-temperature corrosion and slagging
- **Compliance**: Automated monitoring against EPA, ASME, ISO 9001, and regional air quality standards

---

## Key Features

### Flue Gas Composition Analysis

- **Real-time Monitoring**: Continuous tracking of combustion gas composition
  - Oxygen (O2): 0-21% (accuracy: ±0.1%)
  - Carbon Dioxide (CO2): 0-20% (accuracy: ±0.1%)
  - Carbon Monoxide (CO): 0-500 ppm (accuracy: ±5 ppm)
  - Nitrogen Oxides (NOx): 0-500 ppm (accuracy: ±2 ppm)
  - Sulfur Dioxide (SO2): 0-1000 ppm
  - Stack Temperature: 100-1500°C
  - Flue Gas Flow Rate: Mass/volumetric flow measurement

- **Multi-point Sampling**: Monitors flue gas at multiple locations
  - Economizer outlet
  - Air preheater outlet
  - Stack (post-cleanup)
  - Furnace exit (optional)

- **Compliance Monitoring**: Automatic verification against EPA, EU IED, and local regulations
- **Trend Analysis**: Historical data analysis with anomaly detection
- **Data Quality Validation**: Real-time QA/QC of analyzer measurements

### Combustion Efficiency Calculation

- **Thermal Efficiency**: Real-time calculation using ASME PTC 4.1 methodology
  - Direct method (heat input vs. output)
  - Indirect method (flue gas loss analysis)
  - Radiation and convection losses
  - Unburned fuel losses

- **Stoichiometric Calculations**: Chemical balance for combustion reactions
  ```
  Fuel + Air (O2 + N2) → CO2 + H2O + N2 + Excess O2
  ```
  - Theoretical air requirement calculation
  - Actual air-fuel ratio determination
  - Excess air percentage calculation
  - Combustion product composition prediction

- **Efficiency Loss Analysis**: Breakdown of energy losses
  - Dry flue gas loss (sensible heat)
  - Moisture loss (latent + sensible heat)
  - Incomplete combustion loss (CO, unburned hydrocarbons)
  - Radiation and convection loss
  - Ash pit loss (solid fuels)
  - Unaccounted losses

### Air-Fuel Ratio Optimization

- **Optimal Excess Air Determination**
  - Target: 10-20% excess air for gas fuels (3-4% O2 dry)
  - Target: 15-30% excess air for liquid fuels (2.5-3.5% O2 dry)
  - Target: 20-50% excess air for solid fuels (3-6% O2 dry)
  - Dynamic adjustment based on load, fuel quality, burner condition

- **Trim Control Recommendations**
  - Real-time setpoint adjustments for O2 or CO2 control
  - Adaptive control based on fuel variability
  - Load-following optimization

- **Safety Constraints**
  - Minimum excess air to prevent CO formation
  - Maximum excess air to prevent flue gas recirculation
  - Flame stability monitoring

### Emissions Control

- **NOx Optimization**
  - Thermal NOx reduction through temperature control
  - Fuel NOx minimization through combustion staging
  - Low-NOx burner performance monitoring
  - Flue gas recirculation (FGR) optimization
  - Selective Catalytic Reduction (SCR) integration

- **CO Control**
  - CO breakthrough detection (incomplete combustion)
  - Burner tuning recommendations
  - Air distribution optimization

- **SO2 and Particulate Monitoring**
  - Fuel sulfur tracking
  - Scrubber performance monitoring
  - Opacity and particulate matter correlation

### Predictive Maintenance

- **Burner Performance Degradation Detection**
  - CO increase trends indicating burner fouling
  - O2 variability indicating air register issues
  - Flame stability metrics

- **Heat Transfer Surface Fouling**
  - Stack temperature rise indicating soot buildup
  - Efficiency degradation trending

- **Analyzer Calibration Tracking**
  - Drift detection and auto-calibration scheduling
  - Cross-validation with redundant sensors

### SCADA/DCS Integration

- **OPC-UA Support**: Secure, standards-based communication
- **Modbus TCP/RTU**: Legacy equipment integration
- **Profibus/Foundation Fieldbus**: Field device communication
- **Real-time Data Acquisition**: Sub-second sampling rates
- **Setpoint Control**: Automated O2/fuel flow trim control
- **Alarm Management**: Configurable alerts with severity levels

---

## Combustion Fundamentals

### Combustion Chemistry and Stoichiometry

Combustion is a rapid exothermic oxidation reaction between fuel and oxygen that produces heat, carbon dioxide, and water.

#### Complete Combustion Reactions

**Natural Gas (Methane)**:
```
CH4 + 2 O2 → CO2 + 2 H2O + 890 kJ/mol
```

**Fuel Oil (Approximated as C12H26)**:
```
C12H26 + 18.5 O2 → 12 CO2 + 13 H2O + 7513 kJ/mol
```

**Coal (Approximated as C)**:
```
C + O2 → CO2 + 393.5 kJ/mol
```

**Hydrogen** (emerging fuel):
```
2 H2 + O2 → 2 H2O + 484 kJ/mol
```

#### Theoretical Air Requirement

The **stoichiometric** (theoretical) air requirement is the minimum amount of air needed for complete combustion.

**For Natural Gas (CH4)**:
```
CH4 + 2 O2 + 7.52 N2 → CO2 + 2 H2O + 7.52 N2
```

Theoretical air = 2 mol O2 × (1 mol air / 0.21 mol O2) = 9.52 mol air per mol CH4

**For Fuel Oil (C12H26)**:
```
Theoretical O2 = [C + 0.25(H - Cl/35.5) + S - 0.5×O] / 12 × 32 kg O2/kg fuel
```

Where: C, H, O, S, Cl are mass fractions from ultimate analysis

**Theoretical Air (kg/kg fuel)**:
```
Air_theoretical = O2_theoretical / 0.232
```
(Air is 23.2% oxygen by mass)

#### Excess Air

**Excess Air** is the amount of air supplied beyond the theoretical requirement, expressed as a percentage:

```
Excess Air (%) = [(Actual Air - Theoretical Air) / Theoretical Air] × 100
```

**Why Excess Air is Needed**:
1. **Ensure Complete Combustion**: Real burners don't achieve perfect fuel-air mixing
2. **Prevent CO Formation**: Insufficient air leads to incomplete combustion and toxic CO
3. **Safety Margin**: Account for fuel variability and burner wear

**Why Too Much Excess Air is Bad**:
1. **Efficiency Loss**: Extra air must be heated, wasting fuel
2. **Increased NOx**: Higher flame temperatures produce more thermal NOx
3. **Flue Gas Volume**: Larger fans, ducts, and stack required

**Optimal Excess Air Ranges**:
- Natural Gas: 10-20% (3-4% O2 dry)
- No. 2 Fuel Oil: 15-25% (2.5-3.5% O2 dry)
- Heavy Fuel Oil: 20-30% (3-4.5% O2 dry)
- Pulverized Coal: 20-40% (3-6% O2 dry)
- Biomass: 30-50% (4-7% O2 dry)

#### Oxygen in Flue Gas

O2 concentration in dry flue gas is used to infer excess air:

```
Excess Air (%) ≈ O2 (%) / (21 - O2 (%)) × 100
```

**Example**: If O2 = 3.0% in dry flue gas:
```
Excess Air = 3.0 / (21 - 3.0) × 100 = 16.7%
```

**Oxygen Correction**: Emissions are typically corrected to a reference O2 level:
- Natural gas: 3% O2 reference
- Oil: 3% O2 reference
- Coal: 6% O2 reference

```
Emission_corrected = Emission_measured × (21 - O2_reference) / (21 - O2_measured)
```

### Combustion Efficiency Calculation

#### ASME PTC 4.1 Indirect Method (Heat Loss Method)

Combustion efficiency is calculated by subtracting heat losses from 100%:

```
η_combustion = 100% - L_dry_gas - L_moisture - L_incomplete - L_radiation - L_other
```

**1. Dry Flue Gas Loss** (largest loss, typically 5-15%):

```
L_dry_gas = [(m_flue × Cp_flue × (T_flue - T_ambient)] / Q_fuel × 100%
```

Simplified approximation:
```
L_dry_gas (%) ≈ K × (T_flue - T_ambient) × (CO2_max / CO2_actual)
```

Where:
- K = empirical constant (~0.01 for natural gas, ~0.011 for oil)
- T_flue = stack temperature (°C)
- T_ambient = combustion air temperature (°C)
- CO2_max = maximum theoretical CO2 (no excess air)
- CO2_actual = measured CO2 in dry flue gas

**2. Moisture Loss** (latent + sensible heat in water vapor):

```
L_moisture (%) = [(9H + W) × (597 + 0.46(T_flue - T_ambient))] / HHV × 100%
```

Where:
- H = hydrogen content in fuel (mass fraction)
- W = moisture content in fuel (mass fraction)
- 9H = water formed from H2 combustion (9 kg H2O per kg H2)
- 597 = latent heat of vaporization (kcal/kg)
- HHV = higher heating value of fuel (kcal/kg)

**3. Incomplete Combustion Loss**:

```
L_incomplete (%) = (CO / (CO + CO2)) × C_fuel × (C_CO / C_fuel) × 100%
```

Simplified:
```
L_incomplete (%) ≈ 12,600 × CO (%) / HHV (kJ/kg)
```

**4. Radiation and Convection Loss** (2-3% for large boilers, 5-10% for small):

```
L_radiation (%) = f(boiler_size, load, insulation)
```

Typically taken from ASME charts based on capacity and load.

**5. Ash Pit Loss** (solid fuels only, typically 0.5-2%):

```
L_ash = (Unburned_carbon × C_combustion_value) / HHV × 100%
```

#### Typical Efficiency by Fuel Type

| Fuel Type | Typical Efficiency | O2 (dry) | Stack Temp (°C) |
|-----------|-------------------|----------|-----------------|
| Natural Gas | 80-85% | 3-4% | 150-200 |
| No. 2 Oil | 78-84% | 2.5-3.5% | 180-220 |
| Heavy Oil | 75-82% | 3-4.5% | 200-250 |
| Bituminous Coal | 72-80% | 4-6% | 150-180 |
| Biomass | 65-75% | 5-8% | 180-220 |

### Factors Affecting Combustion Efficiency

1. **Excess Air**
   - Too low: Incomplete combustion, CO formation, smoke
   - Too high: Sensible heat loss, reduced efficiency

2. **Stack Temperature**
   - Every 40°F (22°C) increase = ~1% efficiency loss
   - Caused by: Fouled heat transfer surfaces, high excess air

3. **Fuel Quality**
   - Heating value variability
   - Moisture content (especially biomass)
   - Ash and sulfur content

4. **Load**
   - Part-load operation typically less efficient
   - Boiler sized for peak load operates at reduced efficiency at lower loads

5. **Ambient Conditions**
   - Cold combustion air increases efficiency (denser air)
   - Hot ambient reduces efficiency

### Emissions Formation Mechanisms

#### NOx Formation

**1. Thermal NOx** (dominant at high temperatures):
```
N2 + O2 → 2 NO (at T > 1300°C)
```
Exponentially increases with temperature (doubles every 90°C above 1400°C)

**2. Fuel NOx** (from nitrogen in fuel):
```
Fuel-N + O2 → NO + other products
```
Dominant for coal and heavy oils with high nitrogen content

**3. Prompt NOx** (hydrocarbon radicals):
```
CH + N2 → HCN + N → NO
```
Minor contributor (5-10% of total NOx)

**NOx Control Strategies**:
- Reduce peak flame temperature (staging, FGR, water/steam injection)
- Limit excess air (less O2 available)
- Low-NOx burners (delayed mixing, staging)
- Post-combustion: SCR, SNCR

#### CO Formation

CO forms when combustion is incomplete due to:
1. **Insufficient oxygen** (too low excess air)
2. **Poor fuel-air mixing** (burner issues, air distribution)
3. **Low temperature** (flame quenching, cold furnace walls)
4. **Residence time** (insufficient time for CO → CO2)

**CO → CO2 Oxidation**:
```
2 CO + O2 → 2 CO2 + heat
```

**CO Control**:
- Maintain minimum excess air (typically >10%)
- Optimize burner operation and air distribution
- Ensure adequate furnace temperature
- Maintain proper flame stability

#### SO2 Formation

Sulfur in fuel oxidizes to SO2:
```
S + O2 → SO2
```

SO2 emissions proportional to fuel sulfur content. Control through:
- Low-sulfur fuel
- Flue gas desulfurization (wet/dry scrubbers)
- Sorbent injection

---

## Architecture Overview

FLUEFLOW employs a modular, event-driven architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GL-018 FLUEFLOW                             │
│             Flue Gas Analysis & Combustion Agent                 │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Data Layer  │      │ Logic Layer  │      │ Integration  │
│              │      │              │      │   Layer      │
└──────────────┘      └──────────────┘      └──────────────┘
        │                      │                      │
        │                      │                      │
   ┌────┴────┐          ┌─────┴─────┐          ┌────┴────┐
   │         │          │           │          │         │
   ▼         ▼          ▼           ▼          ▼         ▼
Config   Models   Calculators   Tools     SCADA/DCS   Analyzers
```

### Core Components

1. **Configuration Layer** (`config.py`)
   - Pydantic models for type-safe configuration
   - Combustor specifications and operating parameters
   - Fuel properties (composition, heating value)
   - Emissions limits (regulatory compliance)
   - Integration settings

2. **Combustion Calculators** (`calculators/`)
   - Stoichiometric air requirement calculation
   - Combustion product composition (CO2, H2O, N2, O2)
   - Excess air calculation from O2/CO2 measurements
   - Thermal efficiency calculation (ASME PTC 4.1)
   - Heat loss breakdown (dry gas, moisture, incomplete)
   - Air-fuel ratio optimization
   - Emissions calculations (NOx, CO, SO2)

3. **Flue Gas Analyzer Integration** (`integrations/analyzer_integration.py`)
   - O2, CO2, CO, NOx, SO2 analyzer communication
   - Multi-protocol support (4-20mA, Modbus, HART, Profibus)
   - Automatic calibration tracking
   - Data quality validation (range checks, drift detection)

4. **SCADA/DCS Integration** (`integrations/scada_integration.py`)
   - OPC-UA, Modbus, Profibus clients
   - Real-time tag monitoring (O2, CO, NOx, stack temp, fuel flow, air flow)
   - Alarm management
   - Historical data retrieval
   - Setpoint control (O2 trim, fuel flow)

5. **Optimization Engine**
   - Multi-objective optimization (efficiency, emissions, safety)
   - Constraint-based optimization (minimum O2, maximum CO, NOx limits)
   - Adaptive control for varying fuel quality
   - Load-following optimization

6. **Reporting & Analytics**
   - Performance dashboards (Grafana)
   - Compliance reports (EPA, EU IED)
   - Trend analysis
   - Fuel consumption tracking
   - Emissions tracking

### Data Flow

```
Flue Gas Analyzers → Data Acquisition → Validation →
  → Stoichiometric Calculation → Efficiency Calculation →
    → Optimization → Control Actions → DCS/SCADA → Burners
```

**Key Principles**:
- **Deterministic Calculations**: All combustion and efficiency calculations use deterministic algorithms (no LLM hallucinations)
- **Zero Hallucination**: LLMs only used for narrative generation, not calculations
- **Provenance Tracking**: SHA-256 hash of all calculations for audit trail
- **Real-time Processing**: Sub-second response times
- **Fault Tolerance**: Graceful degradation if sensors fail

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

---

## Installation

### Prerequisites

- **Python**: 3.10 or higher
- **Operating System**: Linux, Windows, macOS
- **Memory**: 4 GB RAM minimum
- **CPU**: 4 cores minimum
- **Network**: Access to SCADA/DCS and analyzer networks

### Option 1: Docker Installation (Recommended)

```bash
# Pull the Docker image
docker pull greenlang/gl-018-flueflow:latest

# Run the container
docker run -d \
  --name flueflow \
  -p 8000:8000 \
  -p 9090:9090 \
  -e SCADA_ENDPOINT=opc.tcp://dcs-server:4840 \
  -e LOG_LEVEL=INFO \
  -v /path/to/config:/app/config \
  greenlang/gl-018-flueflow:latest
```

### Option 2: Python Package Installation

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository
git clone https://github.com/greenlang/gl-018-flueflow.git
cd gl-018-flueflow

# Install dependencies
pip install -r requirements.txt

# Install agent
pip install -e .
```

### Option 3: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/gl-018-flueflow.git
cd gl-018-flueflow

# Install dependencies
pip install -r requirements.txt

# Run agent
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check version
python -c "from greenlang.GL_018 import __version__; print(__version__)"

# Health check
curl http://localhost:8000/health
```

---

## Configuration

### Configuration File Structure

FLUEFLOW uses a YAML-based configuration system. Create a `config/flueflow.yaml` file:

```yaml
agent:
  agent_id: GL-018
  agent_name: FLUEFLOW
  version: 1.0.0
  monitoring_interval_seconds: 5  # High-frequency monitoring

combustors:
  - combustor_id: BOILER-001
    combustor_type: package_boiler
    fuel_type: natural_gas
    design_capacity_mmbtu_hr: 100
    design_efficiency_pct: 82.0
    operating_pressure_psig: 150
    stack_height_ft: 120
    burner_type: low_nox
    has_economizer: true
    has_air_preheater: true
    control_mode: O2_TRIM  # O2_TRIM, CO2_CONTROL, or MANUAL

fuel_properties:
  natural_gas:
    fuel_name: Pipeline Natural Gas
    fuel_type: natural_gas
    higher_heating_value_btu_scf: 1020
    lower_heating_value_btu_scf: 920
    composition:  # Mole fractions
      CH4: 0.945
      C2H6: 0.032
      C3H8: 0.008
      N2: 0.012
      CO2: 0.003
    theoretical_air_scf_per_scf_fuel: 9.52
    stoichiometric_co2_pct: 11.7

flue_gas_analyzers:
  - analyzer_id: O2-001
    parameter: oxygen
    location: economizer_outlet
    measurement_range_min_pct: 0.0
    measurement_range_max_pct: 21.0
    accuracy_pct: 0.1
    response_time_seconds: 3
    protocol: modbus_tcp
    modbus_address: 192.168.1.50
    modbus_register: 40001
    scada_tag: BOILER.FLUEGAS.O2

  - analyzer_id: CO-001
    parameter: carbon_monoxide
    location: economizer_outlet
    measurement_range_min_ppm: 0
    measurement_range_max_ppm: 500
    accuracy_ppm: 5
    response_time_seconds: 10
    protocol: modbus_tcp
    scada_tag: BOILER.FLUEGAS.CO

  - analyzer_id: NOX-001
    parameter: nitrogen_oxides
    location: stack
    measurement_range_min_ppm: 0
    measurement_range_max_ppm: 500
    accuracy_ppm: 2
    response_time_seconds: 60
    protocol: analog_4_20ma
    scada_tag: BOILER.FLUEGAS.NOX

  - analyzer_id: TEMP-001
    parameter: stack_temperature
    location: stack
    measurement_range_min_f: 100
    measurement_range_max_f: 600
    accuracy_f: 5
    response_time_seconds: 5
    scada_tag: BOILER.FLUEGAS.TEMP

scada_integration:
  scada_system_name: Plant DCS
  protocol: OPC-UA
  server_address: 192.168.1.100
  server_port: 4840
  authentication_required: true
  username: flueflow_user
  enable_ssl: true
  polling_interval_seconds: 1  # Fast polling for combustion control

emissions_limits:
  BOILER-001:
    nox_limit_ppm_at_3pct_o2: 30  # EPA limit for natural gas
    co_limit_ppm: 50
    so2_limit_ppm: 100
    opacity_limit_pct: 20
    regulatory_authority: EPA
    permit_number: ABC-12345

optimization_settings:
  target_excess_air_pct: 15.0  # Target for natural gas
  min_excess_air_pct: 10.0  # Safety minimum
  max_excess_air_pct: 25.0  # Maximum allowed
  target_o2_dry_pct: 3.0  # Target O2 for natural gas
  max_co_ppm: 50  # CO limit (incomplete combustion)
  efficiency_target_pct: 82.0
  optimization_frequency_seconds: 60  # Recalculate every minute
```

### Environment Variables

```bash
# SCADA Connection
export SCADA_ENDPOINT="opc.tcp://192.168.1.100:4840"
export SCADA_USERNAME="flueflow_user"
export SCADA_PASSWORD="secure_password"

# Analyzer Connection
export ANALYZER_MODBUS_HOST="192.168.1.50"
export ANALYZER_MODBUS_PORT="502"

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"

# Performance
export CACHE_TTL_SECONDS="60"
export MAX_PARALLEL_CALCULATIONS="20"

# Security
export ENABLE_AUTH="true"
export JWT_SECRET_KEY="your-secret-key"

# Determinism
export DETERMINISTIC_MODE="true"
export ZERO_HALLUCINATION="true"
export RANDOM_SEED="42"
```

### Fuel Properties Configuration

For custom fuels, define complete ultimate analysis and heating values:

```yaml
fuel_properties:
  custom_fuel_oil:
    fuel_name: No. 2 Fuel Oil
    fuel_type: liquid
    higher_heating_value_btu_lb: 19500
    lower_heating_value_btu_lb: 18300
    ultimate_analysis:  # Mass fractions (dry basis)
      carbon: 0.870
      hydrogen: 0.125
      sulfur: 0.002
      nitrogen: 0.001
      oxygen: 0.002
    moisture_pct: 0.5
    ash_pct: 0.01
    theoretical_air_lb_per_lb_fuel: 14.7
```

---

## Usage

### Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a 10-minute getting started guide.

### Basic Operations

#### Start the Agent

```bash
# Using Docker
docker start flueflow

# Using Python
python main.py --config config/flueflow.yaml
```

#### Analyze Flue Gas Composition

```python
from greenlang.GL_018 import FlueGasAnalyzer, FlueGasData

# Initialize agent
agent = FlueGasAnalyzer(config_path="config/flueflow.yaml")

# Submit flue gas measurements
flue_gas_data = FlueGasData(
    combustor_id="BOILER-001",
    timestamp="2025-12-02T10:30:00Z",
    oxygen_pct_dry=3.2,
    carbon_dioxide_pct_dry=10.8,
    carbon_monoxide_ppm=45,
    nitrogen_oxides_ppm=28,
    stack_temperature_f=350,
    fuel_flow_rate_mmbtu_hr=85.0,
    steam_flow_rate_lb_hr=85000,
    feedwater_temperature_f=240,
    ambient_temperature_f=75
)

# Analyze combustion performance
result = await agent.analyze_combustion(flue_gas_data)

print(f"Combustion Efficiency: {result.combustion_efficiency_pct:.2f}%")
print(f"Excess Air: {result.excess_air_pct:.1f}%")
print(f"Fuel Savings Potential: ${result.annual_fuel_savings_usd:,.0f}/year")
print(f"Emissions Compliance: {result.emissions_compliant}")
```

**Output Example**:
```
Combustion Efficiency: 81.2%
Excess Air: 18.5%
Fuel Savings Potential: $45,230/year
Emissions Compliance: True
```

#### Calculate Combustion Efficiency

```python
from greenlang.GL_018 import EfficiencyCalculator, FuelProperties

# Calculate efficiency using indirect method (ASME PTC 4.1)
efficiency_calc = EfficiencyCalculator(fuel_properties=fuel_props)

efficiency_result = await efficiency_calc.calculate_efficiency(
    oxygen_pct_dry=3.2,
    stack_temperature_f=350,
    ambient_temperature_f=75,
    carbon_monoxide_ppm=45,
    fuel_type="natural_gas",
    fuel_moisture_pct=0.0
)

print(f"Overall Efficiency: {efficiency_result.overall_efficiency_pct:.2f}%")
print(f"Dry Gas Loss: {efficiency_result.dry_gas_loss_pct:.2f}%")
print(f"Moisture Loss: {efficiency_result.moisture_loss_pct:.2f}%")
print(f"Incomplete Combustion Loss: {efficiency_result.incomplete_combustion_loss_pct:.2f}%")
print(f"Radiation Loss: {efficiency_result.radiation_loss_pct:.2f}%")
```

**Output Example**:
```
Overall Efficiency: 81.2%
Dry Gas Loss: 13.5%
Moisture Loss: 3.8%
Incomplete Combustion Loss: 0.3%
Radiation Loss: 1.2%
```

#### Optimize Air-Fuel Ratio

```python
# Get air-fuel ratio recommendations
optimization = await agent.optimize_air_fuel_ratio(
    flue_gas_data=flue_gas_data,
    combustor_specs=combustor_specs,
    fuel_properties=fuel_properties,
    constraints={
        "min_excess_air_pct": 10.0,
        "max_co_ppm": 50,
        "nox_limit_ppm": 30
    }
)

print(f"Current Excess Air: {optimization.current_excess_air_pct:.1f}%")
print(f"Optimal Excess Air: {optimization.optimal_excess_air_pct:.1f}%")
print(f"Current Efficiency: {optimization.current_efficiency_pct:.2f}%")
print(f"Optimized Efficiency: {optimization.optimized_efficiency_pct:.2f}%")
print(f"Efficiency Gain: {optimization.efficiency_gain_pct:.2f} percentage points")
print(f"Annual Fuel Savings: {optimization.annual_fuel_savings_mmbtu:,.0f} MMBtu")
print(f"Annual Cost Savings: ${optimization.annual_cost_savings_usd:,.0f}")
print(f"CO2 Reduction: {optimization.annual_co2_reduction_tons:,.0f} tons/year")

print(f"\nRecommended Actions:")
for action in optimization.recommended_actions:
    print(f"  - {action}")
```

**Output Example**:
```
Current Excess Air: 18.5%
Optimal Excess Air: 15.0%
Current Efficiency: 81.2%
Optimized Efficiency: 82.1%
Efficiency Gain: 0.9 percentage points
Annual Fuel Savings: 7,850 MMBtu
Annual Cost Savings: $45,230
CO2 Reduction: 419 tons/year

Recommended Actions:
  - Reduce combustion air damper position by 5%
  - Adjust O2 trim setpoint from 3.2% to 3.0%
  - Monitor CO levels closely during adjustment
  - Verify burner flame stability after changes
```

#### Emissions Analysis

```python
# Assess emissions compliance
emissions_analysis = await agent.analyze_emissions(
    flue_gas_data=flue_gas_data,
    emissions_limits=emissions_limits,
    fuel_properties=fuel_properties
)

print(f"NOx Emissions: {emissions_analysis.nox_ppm_at_3pct_o2:.1f} ppm @ 3% O2")
print(f"NOx Limit: {emissions_analysis.nox_limit_ppm:.1f} ppm @ 3% O2")
print(f"NOx Compliance: {emissions_analysis.nox_compliant}")
print(f"NOx Margin: {emissions_analysis.nox_margin_pct:.1f}%")

print(f"\nCO Emissions: {emissions_analysis.co_ppm:.0f} ppm")
print(f"CO Limit: {emissions_analysis.co_limit_ppm:.0f} ppm")
print(f"CO Compliance: {emissions_analysis.co_compliant}")

print(f"\nAnnual NOx Emissions: {emissions_analysis.annual_nox_emissions_tons:.1f} tons/year")
print(f"Annual CO2 Emissions: {emissions_analysis.annual_co2_emissions_tons:,.0f} tons/year")
```

### REST API Usage

FLUEFLOW exposes a REST API on port 8000:

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "agent_id": "GL-018",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "scada_connected": true,
  "analyzers_connected": 4,
  "last_reading_timestamp": "2025-12-02T10:30:00Z"
}
```

#### Submit Flue Gas Data

```bash
curl -X POST http://localhost:8000/api/v1/flue-gas \
  -H "Content-Type: application/json" \
  -d '{
    "combustor_id": "BOILER-001",
    "timestamp": "2025-12-02T10:30:00Z",
    "oxygen_pct_dry": 3.2,
    "carbon_dioxide_pct_dry": 10.8,
    "carbon_monoxide_ppm": 45,
    "nitrogen_oxides_ppm": 28,
    "stack_temperature_f": 350,
    "fuel_flow_rate_mmbtu_hr": 85.0
  }'
```

Response:
```json
{
  "combustor_id": "BOILER-001",
  "timestamp": "2025-12-02T10:30:00Z",
  "combustion_efficiency_pct": 81.2,
  "excess_air_pct": 18.5,
  "emissions_compliant": true,
  "nox_ppm_at_3pct_o2": 28.0,
  "co_ppm": 45,
  "annual_fuel_cost_usd": 1825000,
  "potential_annual_savings_usd": 45230,
  "recommended_actions": [
    "Reduce excess air to 15% to improve efficiency",
    "Monitor CO levels during adjustment"
  ]
}
```

#### Get Efficiency Calculation

```bash
curl -X POST http://localhost:8000/api/v1/calculate-efficiency \
  -H "Content-Type: application/json" \
  -d @efficiency_request.json
```

#### Get Air-Fuel Ratio Optimization

```bash
curl -X POST http://localhost:8000/api/v1/optimize-air-fuel \
  -H "Content-Type: application/json" \
  -d @optimization_request.json
```

#### Get Performance Report

```bash
curl http://localhost:8000/api/v1/performance-report/BOILER-001?period=24h
```

Response includes:
- Average efficiency
- Fuel consumption trends
- Emissions summary
- Cost analysis
- Optimization opportunities

---

## API Reference

### Core Classes

#### FlueGasAnalyzer

Main orchestrator for flue gas analysis and combustion optimization.

```python
class FlueGasAnalyzer:
    def __init__(self, config_path: str):
        """Initialize agent with configuration."""

    async def analyze_combustion(
        self,
        flue_gas_data: FlueGasData,
        combustor_specs: CombustorSpecifications,
        fuel_properties: FuelProperties
    ) -> CombustionAnalysisResult:
        """Analyze combustion performance from flue gas measurements."""

    async def calculate_efficiency(
        self,
        flue_gas_data: FlueGasData,
        fuel_properties: FuelProperties,
        method: str = "indirect"  # "indirect" or "direct"
    ) -> EfficiencyResult:
        """Calculate combustion efficiency using ASME PTC 4.1."""

    async def optimize_air_fuel_ratio(
        self,
        flue_gas_data: FlueGasData,
        combustor_specs: CombustorSpecifications,
        fuel_properties: FuelProperties,
        constraints: OptimizationConstraints
    ) -> AirFuelOptimizationResult:
        """Optimize air-fuel ratio for maximum efficiency and compliance."""

    async def analyze_emissions(
        self,
        flue_gas_data: FlueGasData,
        emissions_limits: EmissionsLimits,
        fuel_properties: FuelProperties
    ) -> EmissionsAnalysisResult:
        """Analyze emissions compliance and calculate annual emissions."""

    async def detect_combustion_issues(
        self,
        flue_gas_data: FlueGasData,
        historical_baseline: FlueGasBaseline
    ) -> CombustionDiagnosticResult:
        """Detect combustion issues (burner fouling, air leaks, etc.)."""
```

### Data Models

#### FlueGasData

```python
class FlueGasData(BaseModel):
    combustor_id: str
    timestamp: datetime
    oxygen_pct_dry: float  # 0-21% (dry basis)
    carbon_dioxide_pct_dry: Optional[float]  # 0-20% (dry basis)
    carbon_monoxide_ppm: float  # 0-500 ppm
    nitrogen_oxides_ppm: Optional[float]  # 0-500 ppm (as NO2)
    sulfur_dioxide_ppm: Optional[float]  # 0-1000 ppm
    stack_temperature_f: float  # 100-1500°F
    flue_gas_flow_rate_scfm: Optional[float]
    fuel_flow_rate_mmbtu_hr: float
    steam_flow_rate_lb_hr: Optional[float]
    feedwater_temperature_f: Optional[float]
    ambient_temperature_f: float
    barometric_pressure_inhg: Optional[float] = 29.92
```

#### FuelProperties

```python
class FuelProperties(BaseModel):
    fuel_name: str
    fuel_type: FuelType  # natural_gas, fuel_oil, coal, biomass, hydrogen
    higher_heating_value_btu_lb: float  # HHV
    lower_heating_value_btu_lb: float  # LHV
    ultimate_analysis: UltimateAnalysis  # C, H, O, N, S fractions
    moisture_pct: float = 0.0
    ash_pct: float = 0.0
    theoretical_air_lb_per_lb_fuel: float
    stoichiometric_co2_pct: float  # Max CO2 at zero excess air
```

#### CombustionAnalysisResult

```python
class CombustionAnalysisResult(BaseModel):
    combustor_id: str
    timestamp: datetime
    combustion_efficiency_pct: float  # 60-90%
    thermal_efficiency_pct: Optional[float]  # If heat output known
    excess_air_pct: float  # 5-100%
    air_fuel_ratio: float  # Actual AFR
    stoichiometric_air_fuel_ratio: float  # Theoretical AFR
    dry_gas_loss_pct: float
    moisture_loss_pct: float
    incomplete_combustion_loss_pct: float
    radiation_loss_pct: float
    unaccounted_loss_pct: float
    emissions_compliant: bool
    nox_ppm_at_reference_o2: float
    co_ppm: float
    potential_efficiency_gain_pct: float
    annual_fuel_cost_usd: float
    potential_annual_savings_usd: float
    annual_co2_emissions_tons: float
    recommended_actions: List[str]
```

See [TOOL_SPECIFICATIONS.md](TOOL_SPECIFICATIONS.md) for complete API documentation.

---

## Integration Guides

### SCADA/DCS Integration

FLUEFLOW supports multiple industrial communication protocols:

#### OPC-UA Integration

```python
from greenlang.GL_018.integrations import create_scada_client, ConnectionProtocol

# Create OPC-UA client
scada_client = create_scada_client(
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.100",
    port=4840,
    username="flueflow",
    password="password",
    enable_ssl=True
)

# Connect
await scada_client.connect()

# Register flue gas analyzer tags
from greenlang.GL_018.integrations import create_standard_fluegas_tags
tags = create_standard_fluegas_tags()
scada_client.register_tags(tags)

# Subscribe to real-time updates
async def on_o2_change(data_point):
    print(f"O2 changed: {data_point.value}%")
    # Trigger efficiency recalculation

await scada_client.subscribe_tag("FLUEGAS_O2_01", on_o2_change)

# Read current values
o2_value = await scada_client.read_tag("FLUEGAS_O2_01")
co_value = await scada_client.read_tag("FLUEGAS_CO_01")
stack_temp = await scada_client.read_tag("STACK_TEMP_01")

# Write O2 trim setpoint
await scada_client.write_tag("O2_TRIM_SETPOINT", 3.0)
```

#### Modbus TCP Integration

```python
# Create Modbus client for flue gas analyzer
analyzer_client = create_scada_client(
    protocol=ConnectionProtocol.MODBUS_TCP,
    host="192.168.1.50",
    port=502,
    modbus_unit_id=1
)

await analyzer_client.connect()

# Read O2 measurement (holding register 40001)
o2_value = await analyzer_client.read_tag("40001")

# Read CO measurement (holding register 40002)
co_value = await analyzer_client.read_tag("40002")
```

### Flue Gas Analyzer Integration

Supported analyzer manufacturers:
- **Servomex**: O2, CO, CO2, NOx analyzers
- **Siemens Ultramat**: Continuous gas analyzers
- **ABB**: AO2020, EL3020, ACX analyzers
- **Yokogawa**: ZR series O2 analyzers
- **Horiba**: PG-350 series
- **Teledyne**: TOCGC, T400 series
- **Ametek**: Thermox O2 analyzers

**Integration Methods**:
- Analog 4-20 mA (via SCADA/DCS AI modules)
- Modbus TCP/RTU
- Profibus DP
- HART protocol
- OPC-UA

### Burner Management System (BMS) Integration

```python
from greenlang.GL_018.config import BMSIntegration

bms_config = BMSIntegration(
    bms_system_name="Honeywell BMS",
    protocol="OPC-UA",
    endpoint="opc.tcp://bms-server:4840",
    enable_interlock_monitoring=True,
    enable_flame_monitoring=True,
    enable_fuel_valve_control=False  # Safety: manual only
)

# Monitor burner status
burner_status = await flueflow_agent.get_burner_status()
print(f"Burners Running: {burner_status.burners_in_service}")
print(f"Flame Stability: {burner_status.flame_stability_score}")
```

### Continuous Emissions Monitoring System (CEMS) Integration

```python
from greenlang.GL_018.integrations import CEMSIntegration

cems_config = CEMSIntegration(
    cems_vendor="Teledyne API",
    protocol="Modbus_TCP",
    host="192.168.1.60",
    enable_data_acquisition=True,
    reporting_frequency_minutes=15,
    regulatory_authority="EPA",
    submit_to_regulatory_portal=True  # Auto-submit CEMS data
)

# FLUEFLOW will:
# 1. Read CEMS data in real-time
# 2. Validate against permit limits
# 3. Generate quarterly CEMS reports
# 4. Submit data to EPA CAMD or state portals
```

### Agent Coordination

FLUEFLOW can coordinate with other GreenLang agents:

- **GL-001 THERMOSYNC**: Steam system optimization
- **GL-016 WATERGUARD**: Boiler water treatment coordination
- **GL-013 PREDICTMAINT**: Burner maintenance scheduling

```yaml
integrations:
  - name: greenlang_agents
    type: agent_coordination
    protocol: message_bus
    agents:
      - GL-001  # THERMOSYNC
      - GL-016  # WATERGUARD
      - GL-013  # PREDICTMAINT
```

**Use Case**: When FLUEFLOW detects high CO indicating incomplete combustion, it notifies GL-013 PREDICTMAINT to schedule burner inspection and tuning.

---

## Monitoring and Alerting

### Prometheus Metrics

FLUEFLOW exports metrics on port 9090:

```bash
# Scrape metrics
curl http://localhost:9090/metrics
```

**Key Metrics**:
```
# Combustion efficiency (%)
flueflow_combustion_efficiency_pct{combustor_id="BOILER-001"} 81.2

# Excess air (%)
flueflow_excess_air_pct{combustor_id="BOILER-001"} 18.5

# Oxygen in flue gas (% dry)
flueflow_oxygen_pct_dry{combustor_id="BOILER-001"} 3.2

# Carbon monoxide (ppm)
flueflow_carbon_monoxide_ppm{combustor_id="BOILER-001"} 45

# Nitrogen oxides (ppm @ 3% O2)
flueflow_nox_ppm_at_3pct_o2{combustor_id="BOILER-001"} 28

# Stack temperature (°F)
flueflow_stack_temperature_f{combustor_id="BOILER-001"} 350

# Fuel consumption (MMBtu/hr)
flueflow_fuel_consumption_mmbtu_hr{combustor_id="BOILER-001"} 85.0

# Annual fuel cost (USD)
flueflow_annual_fuel_cost_usd{combustor_id="BOILER-001"} 1825000

# Potential annual savings (USD)
flueflow_potential_savings_usd{combustor_id="BOILER-001"} 45230

# CO2 emissions (tons/year)
flueflow_co2_emissions_tons_year{combustor_id="BOILER-001"} 8425

# Emissions compliance (1=compliant, 0=violation)
flueflow_emissions_compliant{combustor_id="BOILER-001"} 1

# Total optimizations
flueflow_optimizations_total 2847

# Alerts by severity and type
flueflow_alerts_total{severity="critical",type="high_co"} 0
flueflow_alerts_total{severity="warning",type="high_excess_air"} 3
```

### Grafana Dashboards

Pre-built dashboards available in `monitoring/grafana/`:

1. **Combustion Overview**: Efficiency, excess air, emissions at-a-glance
2. **Flue Gas Analysis**: Real-time O2, CO2, CO, NOx trends
3. **Efficiency Dashboard**: Heat losses breakdown, optimization opportunities
4. **Emissions Compliance**: NOx, CO, SO2 vs. limits with compliance status
5. **Fuel Consumption**: Fuel usage, cost, savings potential
6. **Diagnostic Dashboard**: Burner health, analyzer status, system diagnostics

Import dashboards:
```bash
# Copy dashboard JSON to Grafana
cp monitoring/grafana/*.json /var/lib/grafana/dashboards/
```

### Alert Configuration

Configure alerts in `config/alerts.yaml`:

```yaml
alerts:
  - name: high_carbon_monoxide
    condition: "carbon_monoxide_ppm > 100"
    severity: critical
    description: "High CO indicates incomplete combustion - safety hazard"
    notification_channels:
      - email: operations@company.com
      - pagerduty: combustion-oncall
    recommended_actions:
      - "Increase excess air immediately"
      - "Check burner condition"
      - "Verify fuel quality"

  - name: low_combustion_efficiency
    condition: "combustion_efficiency_pct < 75.0"
    severity: high
    notification_channels:
      - email: energy@company.com
      - slack: #energy-alerts

  - name: nox_limit_exceeded
    condition: "nox_ppm_at_3pct_o2 > nox_limit_ppm"
    severity: critical
    description: "NOx emissions exceeding permit limit"
    notification_channels:
      - email: environmental@company.com
      - pagerduty: compliance-oncall

  - name: high_excess_air
    condition: "excess_air_pct > 30.0"
    severity: warning
    description: "Excess air too high - efficiency opportunity"
    notification_channels:
      - email: energy@company.com

  - name: analyzer_fault
    condition: "analyzer_status == 'fault'"
    severity: high
    notification_channels:
      - email: instrumentation@company.com
```

### Notification Channels

Supported channels:
- Email (SMTP)
- Slack
- PagerDuty
- Microsoft Teams
- SMS (Twilio)
- Webhook

---

## Troubleshooting

### Common Issues

#### High Carbon Monoxide (CO)

**Symptoms**: CO > 100 ppm, incomplete combustion alarm

**Root Causes**:
1. Insufficient excess air (O2 too low)
2. Poor fuel-air mixing (burner issue)
3. Flame impingement on cold surfaces
4. Burner fouling or wear
5. Fuel quality issues

**Solutions**:
1. Increase excess air (increase combustion air damper opening)
2. Verify minimum O2 setpoint is adequate (typically >2% for gas)
3. Inspect and clean burners
4. Check fuel pressure and quality
5. Verify proper burner flame pattern
6. Check for air leaks in furnace (false O2 reading)

**Diagnostic Commands**:
```bash
# Check current operating parameters
curl http://localhost:8000/api/v1/combustion-status/BOILER-001

# Get diagnostic report
curl http://localhost:8000/api/v1/diagnostics/BOILER-001

# Review historical CO trends
curl http://localhost:8000/api/v1/trends/BOILER-001?parameter=CO&hours=24
```

#### Low Combustion Efficiency

**Symptoms**: Efficiency < 75%, high fuel consumption

**Root Causes**:
1. Excessive excess air (O2 too high)
2. High stack temperature (fouled heat transfer surfaces)
3. Air leaks in furnace or ductwork
4. Incomplete combustion (high CO)
5. Poor insulation (high radiation loss)

**Solutions**:
1. Reduce excess air to optimal level (15-20% for gas)
2. Clean heat transfer surfaces (economizer, air preheater)
3. Repair air leaks
4. Tune burners to minimize CO
5. Improve insulation on boiler casing

**Efficiency Troubleshooting Tool**:
```python
# Get efficiency breakdown and recommendations
efficiency_diag = await agent.diagnose_efficiency_loss(
    combustor_id="BOILER-001",
    target_efficiency_pct=82.0
)

print(f"Current Efficiency: {efficiency_diag.current_efficiency_pct}%")
print(f"Target Efficiency: {efficiency_diag.target_efficiency_pct}%")
print(f"Efficiency Gap: {efficiency_diag.efficiency_gap_pct}%")

print("\nLoss Breakdown:")
for loss in efficiency_diag.heat_losses:
    print(f"  {loss.loss_type}: {loss.loss_pct:.2f}% (${loss.annual_cost_usd:,.0f}/year)")

print("\nTop Recommendations:")
for rec in efficiency_diag.top_recommendations:
    print(f"  - {rec.action}")
    print(f"    Savings: {rec.annual_savings_usd:,.0f} USD/year")
```

#### High Stack Temperature

**Symptoms**: Stack temp > 400°F for gas boilers, efficiency loss

**Root Causes**:
1. Fouled heat transfer surfaces (soot, scale)
2. Excessive excess air
3. High feedwater temperature (reduced economizer duty)
4. Economizer bypass valve open
5. Tube leaks (reduced water flow)

**Solutions**:
1. Clean fire-side surfaces (soot blowing, chemical cleaning)
2. Clean water-side surfaces (if scale buildup)
3. Reduce excess air
4. Close economizer bypass valve
5. Inspect for tube leaks

**Impact**: Every 40°F increase in stack temp = ~1% efficiency loss

#### NOx Emissions Exceeding Limit

**Symptoms**: NOx > permit limit, compliance violation

**Root Causes**:
1. High flame temperature (excessive excess air or high preheat)
2. Low-NOx burners not functioning properly
3. Flue gas recirculation (FGR) system not operating
4. High fuel nitrogen content
5. SCR catalyst deactivated or spent

**Solutions**:
1. Reduce excess air (lowers peak flame temperature)
2. Optimize burner operation (staging, delayed mixing)
3. Enable/optimize FGR system
4. Reduce combustion air preheat temperature (if possible)
5. Replace or regenerate SCR catalyst
6. Consider water/steam injection (emergency measure)

#### Analyzer Calibration Drift

**Symptoms**: Erratic readings, mass balance errors

**Root Causes**:
1. Analyzer drift (span/zero)
2. Sample line plugging or leaks
3. Sample conditioning system failure
4. Ambient temperature effects

**Solutions**:
1. Perform analyzer calibration (zero and span gas)
2. Clean sample lines and filters
3. Check sample pump operation
4. Verify heated sample line temperature
5. Replace analyzer cell (O2 sensor, electrochemical cells)

**Calibration Schedule**:
- O2 analyzer: Weekly calibration check, monthly full calibration
- CO/NOx analyzers: Daily zero check, weekly calibration
- CEMS: Daily calibration, quarterly relative accuracy test audit (RATA)

### Diagnostic Commands

```bash
# Check agent status
curl http://localhost:8000/health

# Get SCADA connection status
curl http://localhost:8000/api/v1/scada/status

# Get analyzer status
curl http://localhost:8000/api/v1/analyzers/status

# Get active alarms
curl http://localhost:8000/api/v1/alarms/active

# Get recent flue gas readings
curl http://localhost:8000/api/v1/flue-gas/BOILER-001?hours=24

# Get combustion diagnostic report
curl http://localhost:8000/api/v1/diagnostics/BOILER-001

# Force analyzer recalibration
curl -X POST http://localhost:8000/api/v1/calibrate

# Export logs
docker logs flueflow > flueflow.log
```

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: DEBUG
  format: json
  enable_trace: true
  log_calculations: true  # Log all stoichiometric calculations
```

Or via environment:
```bash
export LOG_LEVEL=DEBUG
export ENABLE_TRACE=true
export LOG_CALCULATIONS=true
```

### Performance Issues

If experiencing slow response times:

1. Check CPU/memory usage: `docker stats flueflow`
2. Review cache settings: increase `CACHE_TTL_SECONDS`
3. Reduce polling frequency if SCADA saturated
4. Enable batch processing for multiple combustors
5. Check database performance (if using persistent storage)
6. Verify network latency to SCADA/analyzers

---

## Performance Optimization

### Best Practices for Combustion Efficiency

#### 1. Optimize Excess Air

**Target Ranges** (O2 dry basis):
- Natural gas: 3.0-4.0% O2 (10-20% excess air)
- No. 2 oil: 2.5-3.5% O2 (15-25% excess air)
- Heavy oil: 3.0-4.5% O2 (20-30% excess air)
- Coal: 3.5-6.0% O2 (20-40% excess air)

**Procedure**:
1. Start at current O2 setpoint
2. Reduce O2 setpoint by 0.25% increments
3. Monitor CO (must stay < 50 ppm)
4. Monitor flame stability
5. Wait 10 minutes for system to stabilize
6. Measure efficiency at each setpoint
7. Select optimal setpoint with maximum efficiency and safe CO levels

**ROI**: 1% reduction in excess air = ~0.5% efficiency improvement

#### 2. Minimize Stack Temperature

**Target Ranges**:
- Natural gas: 250-350°F (with economizer)
- Fuel oil: 300-400°F
- Coal: 300-350°F (with air preheater)

**Actions**:
- Clean heat transfer surfaces quarterly (or more frequently)
- Optimize soot blowing frequency
- Repair air leaks
- Ensure economizer is not bypassed
- Inspect for tube leaks

**ROI**: Reducing stack temp from 400°F to 350°F = ~1.2% efficiency improvement

#### 3. Load-Following Optimization

Efficiency varies with load. Optimize for each load point:

```python
# Configure load-based optimization
load_optimization_config = {
    "25_pct_load": {"target_o2_pct": 4.0, "target_excess_air_pct": 20},
    "50_pct_load": {"target_o2_pct": 3.5, "target_excess_air_pct": 17},
    "75_pct_load": {"target_o2_pct": 3.0, "target_excess_air_pct": 15},
    "100_pct_load": {"target_o2_pct": 3.0, "target_excess_air_pct": 15}
}
```

#### 4. Fuel Quality Monitoring

Track fuel heating value and adjust controls:

```python
# FLUEFLOW automatically adjusts for fuel variability
fuel_quality_monitor = await agent.monitor_fuel_quality(
    combustor_id="BOILER-001"
)

if fuel_quality_monitor.fuel_hhv_variability_pct > 5:
    print("High fuel quality variability detected")
    print("Recommend: Enable adaptive fuel-air ratio control")
```

#### 5. Combustion Air Preheating

Preheat combustion air using flue gas waste heat:

**Benefits**:
- 1% efficiency improvement per 40°F air preheat
- Improved fuel-air mixing
- Lower excess air requirement

**Considerations**:
- Increased NOx formation (higher flame temperature)
- May require low-NOx burners or FGR

#### 6. Economizer and Air Preheater Maintenance

**Economizer**:
- Reduces stack temperature by recovering heat to preheat feedwater
- 10-15% efficiency improvement over non-economizer boilers
- Clean annually or when stack temp rises > 50°F

**Air Preheater**:
- Preheats combustion air using flue gas
- 3-5% efficiency improvement
- Clean annually or when differential pressure rises

### Advanced Optimization Techniques

#### Model Predictive Control (MPC)

FLUEFLOW includes optional MPC module:

```yaml
advanced_optimization:
  enable_model_predictive_control: true
  prediction_horizon_minutes: 15
  control_horizon_minutes: 5
  update_frequency_seconds: 30
  optimization_objectives:
    - minimize_fuel_consumption
    - minimize_nox_emissions
    - maintain_steam_pressure
  constraints:
    - min_excess_air_pct: 10.0
    - max_co_ppm: 50
    - max_nox_ppm: 30
```

**Benefits**:
- 1-3% additional efficiency improvement
- Faster response to load changes
- Simultaneous efficiency and emissions optimization

#### Machine Learning for Fuel Quality Adaptation

Train ML models to predict optimal setpoints based on fuel quality:

```python
# Enable ML-based fuel adaptation
ml_config = {
    "enable_ml_fuel_adaptation": true,
    "training_data_days": 90,
    "retrain_frequency_days": 30,
    "features": [
        "fuel_flow_rate",
        "flue_gas_oxygen",
        "stack_temperature",
        "ambient_temperature",
        "steam_flow_rate"
    ],
    "target": "combustion_efficiency_pct"
}
```

---

## Compliance Standards

### EPA Regulations (United States)

#### 40 CFR Part 60 - New Source Performance Standards (NSPS)

**Subpart Dc - Small Industrial Boilers**:
- Applicability: Boilers > 10 MMBtu/hr, constructed after 1987
- NOx Limits:
  - Natural gas: 0.10 lb/MMBtu (30 ppm @ 3% O2)
  - No. 2 oil: 0.20 lb/MMBtu
  - Coal: 0.50-0.70 lb/MMBtu

**Subpart Db - Large Industrial Boilers**:
- Applicability: Boilers > 100 MMBtu/hr
- Continuous Emissions Monitoring System (CEMS) required
- Quarterly reporting to EPA CAMD

#### 40 CFR Part 63 - National Emissions Standards for Hazardous Air Pollutants (NESHAP)

**Subpart JJJJJJ - Area Source Boilers**:
- Tune-up requirements every 2 years (gas), annually (oil/coal)
- Energy assessment required
- O2 or CO2 monitoring required

#### 40 CFR Part 64 - Compliance Assurance Monitoring (CAM)

- Applies to major sources with control devices
- Requires monitoring of combustion parameters (O2, CO)
- FLUEFLOW provides automated CAM compliance

**FLUEFLOW Compliance Features**:
```python
# Automatic EPA compliance reporting
epa_config = {
    "enable_epa_reporting": true,
    "facility_id": "123456",
    "unit_id": "BOILER-001",
    "subpart": "Dc",
    "cems_enabled": true,
    "auto_submit_quarterly_reports": true,
    "submit_to_camd": true,
    "camd_username": "facility_user",
    "camd_password": "encrypted_password"
}
```

### ASME Standards

#### ASME PTC 4.1 - Fired Steam Generators

Industry-standard test code for measuring boiler efficiency:
- Direct method (heat output / heat input)
- Indirect method (100% - heat losses)
- FLUEFLOW implements both methods

#### ASME PTC 4 - Fired Steam Generators Performance Test Codes

Complete test procedures for acceptance testing.

### ISO Standards

#### ISO 9001 - Quality Management Systems

FLUEFLOW supports ISO 9001 compliance through:
- Documented procedures
- Calibration tracking
- Data provenance and audit trails
- Continuous improvement metrics

#### ISO 50001 - Energy Management Systems

- Energy performance indicators (EnPIs)
- Energy baseline establishment
- Continuous monitoring and targeting

#### ISO 14001 - Environmental Management Systems

- Emissions tracking and reporting
- Environmental performance monitoring

### EU Industrial Emissions Directive (IED)

**Best Available Techniques (BAT) for Large Combustion Plants**:
- NOx: 50-100 mg/Nm³ for natural gas (3% O2)
- Energy efficiency: > 80% for natural gas boilers
- Continuous emissions monitoring required for > 50 MW

### Regional Air Quality Standards

FLUEFLOW supports regional regulations:
- **SCAQMD** (South Coast Air Quality Management District, California)
- **BAAQMD** (Bay Area Air Quality Management District)
- **TCEQ** (Texas Commission on Environmental Quality)
- **NYSDEC** (New York State Department of Environmental Conservation)

Configure regional limits:
```yaml
emissions_limits:
  BOILER-001:
    regulatory_authority: SCAQMD
    rule_number: Rule 1146
    nox_limit_ppm_at_3pct_o2: 9  # Very stringent limit
    co_limit_ppm: 50
    permit_number: "S-12345"
    compliance_demonstration_method: CEMS
```

---

## Contributing

We welcome contributions from the community!

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/gl-018-flueflow.git
cd gl-018-flueflow

# Create development environment
python3.10 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v --cov=. --cov-report=html
```

### Code Standards

- **Python Version**: 3.10+
- **Style Guide**: PEP 8, enforced by Black and Flake8
- **Type Hints**: Required for all functions
- **Docstrings**: Google style with equation citations
- **Test Coverage**: Minimum 85%

### Testing Combustion Calculations

All stoichiometric and efficiency calculations must have unit tests with verified hand calculations:

```python
def test_excess_air_calculation():
    """Test excess air calculation from O2 measurement.

    Reference: ASME PTC 4.1, Section 5.2.3
    """
    o2_pct_dry = 3.0
    expected_excess_air_pct = 16.67

    calculator = CombustionCalculator(fuel_type="natural_gas")
    result = calculator.calculate_excess_air(o2_pct_dry=o2_pct_dry)

    assert abs(result.excess_air_pct - expected_excess_air_pct) < 0.1
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run linting: `black . && flake8 . && isort .`
5. Run tests: `pytest tests/ -v`
6. Verify calculations against reference standards
7. Commit: `git commit -m "feat: add my feature"`
8. Push: `git push origin feature/my-feature`
9. Open a Pull Request

### Commit Message Format

Follow Conventional Commits:

```
feat: add hydrogen fuel support with stoichiometric calculations
fix: correct LHV calculation for fuel oils per ASTM D240
docs: update combustion efficiency calculation methodology
test: add tests for excess air calculation edge cases
refactor: simplify flue gas composition calculation
```

---

## License and Support

### License

Copyright 2025 GreenLang Technologies
Proprietary Software - All Rights Reserved

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

For licensing inquiries: licensing@greenlang.io

### Support

#### Documentation

- **Main Documentation**: [https://greenlang.io/agents/GL-018](https://greenlang.io/agents/GL-018)
- **API Reference**: [https://api.greenlang.io/GL-018](https://api.greenlang.io/GL-018)
- **Knowledge Base**: [https://support.greenlang.io](https://support.greenlang.io)

#### Contact

- **Technical Support**: support@greenlang.io
- **Sales**: sales@greenlang.io
- **Security Issues**: security@greenlang.io

#### Community

- **Slack**: [greenlang-community.slack.com](https://greenlang-community.slack.com)
- **GitHub Issues**: [github.com/greenlang/gl-018-flueflow/issues](https://github.com/greenlang/gl-018-flueflow/issues)

#### Support Tiers

**Standard Support** (included with license):
- Email support (24-hour response time)
- Documentation access
- Security updates

**Premium Support** (additional cost):
- 4-hour response time
- Phone support
- Dedicated support engineer
- Combustion optimization consulting
- On-site analyzer integration assistance

**Enterprise Support** (additional cost):
- 1-hour response time
- 24/7 phone support
- Dedicated technical account manager
- Custom fuel properties development
- Annual combustion tuning service
- Training and onboarding

---

## Acknowledgments

FLUEFLOW development was guided by industry standards and best practices from:

- **ASME** (American Society of Mechanical Engineers) - PTC 4.1 Standard
- **EPA** (Environmental Protection Agency) - NSPS and NESHAP regulations
- **CIBO** (Council of Industrial Boiler Owners)
- **ABMA** (American Boiler Manufacturers Association)
- **IFRF** (International Flame Research Foundation)
- **Combustion Institute** - Combustion chemistry research

Special thanks to combustion engineers, boiler operators, and environmental compliance professionals who provided domain knowledge and feedback.

### Key References

1. ASME PTC 4.1-2013: "Fired Steam Generators Performance Test Codes"
2. ASME PTC 4-2013: "Fired Steam Generators"
3. EPA 40 CFR Part 60: "Standards of Performance for New Stationary Sources"
4. "Combustion Engineering and Fuel Technology" by H.C. Barnett and R.R. Hibbard
5. "The Coen & Hamworthy Combustion Handbook" by Stephen B. Londerville
6. "Efficient Boiler Operations Sourcebook" by Harry Taplin

---

## Roadmap

### Q1 2026 (v1.0)
- ✅ Core flue gas analysis (O2, CO2, CO, NOx)
- ✅ Combustion efficiency calculation (ASME PTC 4.1)
- ✅ Air-fuel ratio optimization
- ✅ Emissions compliance monitoring
- ✅ SCADA/DCS integration (OPC-UA, Modbus)
- ✅ Prometheus metrics and Grafana dashboards
- ✅ EPA compliance reporting

### Q2 2026 (v1.1)
- Hydrogen fuel blend support (H2 + natural gas)
- Advanced diagnostics (burner fouling detection, air leak detection)
- Machine learning for fuel quality adaptation
- Mobile app for remote monitoring
- Multi-site management dashboard

### Q3 2026 (v1.2)
- Model Predictive Control (MPC) for advanced optimization
- Digital twin integration
- SCR/SNCR catalyst performance monitoring
- Carbon capture integration (post-combustion capture)
- Advanced emissions modeling

### Q4 2026 (v2.0)
- AI-powered root cause analysis
- Automated burner tuning recommendations
- Integration with carbon trading platforms
- Advanced predictive maintenance (burner life prediction)
- Hydrogen-only combustion support (100% H2)

---

## Version History

### v1.0.0 (January 2026)
- Initial production release
- Core combustion analysis functionality
- SCADA/DCS and analyzer integration
- EPA compliance reporting
- Monitoring and alerting
- Comprehensive documentation

---

**GL-018 FLUEFLOW** - Intelligent Flue Gas Analysis for Combustion Optimization

*Reduce fuel consumption, minimize emissions, and ensure compliance while maximizing combustion efficiency.*

For questions or support, contact: support@greenlang.io
