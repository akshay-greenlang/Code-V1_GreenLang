# GL-016 WATERGUARD

**Intelligent Boiler Water Treatment Quality Control Agent**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/greenlang/gl-016-waterguard)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-green.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-Production%20Ready-success.svg)]()

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Water Treatment Fundamentals](#water-treatment-fundamentals)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Integration Guides](#integration-guides)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License and Support](#license-and-support)

---

## Overview

GL-016 WATERGUARD is an advanced AI-powered agent that manages boiler water chemistry to prevent scale formation, corrosion, and steam carryover while optimizing water consumption, chemical usage, and energy efficiency. It provides real-time monitoring, predictive analytics, and automated control for industrial boiler water treatment systems.

### Purpose

Industrial boilers require precise water chemistry control to:
- **Prevent Scale Formation**: Calcium carbonate, calcium sulfate, magnesium silicate, and silica deposits reduce heat transfer efficiency and can cause tube failures
- **Prevent Corrosion**: Dissolved oxygen, low pH, and caustic conditions cause metal loss, pitting, and stress cracking
- **Optimize Blowdown**: Excessive blowdown wastes water and energy; insufficient blowdown leads to concentration buildup
- **Control Chemical Dosing**: Proper chemical treatment protects equipment while minimizing costs and environmental impact

WATERGUARD automates these complex tasks using deterministic calculations, industry-standard formulas, and AI-powered insights.

### Target Market

- **Total Addressable Market (TAM)**: $5 Billion
- **Industries**: Food & Beverage, Chemical, Pharmaceutical, Pulp & Paper, Textiles, Healthcare, Institutional
- **Boiler Types**: Fire-tube, water-tube, package boilers, HRSGs (0-3000 psig)
- **Priority**: P1 (High Priority)
- **Target Release**: Q2 2026

### Business Value

- **Water Savings**: 30% reduction in makeup water consumption
- **Chemical Cost Reduction**: 25% savings on treatment chemicals
- **Energy Efficiency**: 15% reduction in blowdown heat loss
- **Scale Prevention**: 80% reduction in scale-related incidents
- **Corrosion Control**: 70% reduction in corrosion failures
- **Compliance**: Automated monitoring against ASME, ABMA, and EPA standards

---

## Key Features

### Water Chemistry Management

- **Real-time Monitoring**: Continuous tracking of 30+ water quality parameters
  - pH (boiler water, feedwater, makeup water)
  - Conductivity and TDS
  - Dissolved oxygen (ppb-level precision)
  - Silica, phosphate, alkalinity
  - Hardness (calcium, magnesium)
  - Iron, copper, chlorides, sulfates
  - Chemical residuals (sulfite, hydrazine, amines)

- **Compliance Monitoring**: Automatic verification against ASME, ABMA, and ISO standards
- **Trend Analysis**: Historical data analysis with anomaly detection
- **Multi-point Sampling**: Monitors raw water, feedwater, boiler water, condensate return, and blowdown

### Blowdown Optimization

- **Cycles of Concentration (COC)**: Real-time calculation and optimization
  - Formula: `COC = Boiler_Conductivity / Makeup_Conductivity`
  - Optimal COC determination based on water quality limits
  - Blowdown rate calculation: `Blowdown% = 100 / (COC - 1)`

- **Energy Recovery Analysis**: Heat recovery potential from blowdown streams
- **Water Balance Tracking**: Mass balance verification for leak detection
- **Cost-Benefit Analysis**: ROI calculations for blowdown optimization

### Chemical Dosing Control

- **Oxygen Scavengers**: Sulfite, hydrazine, DEHA, carbohydrazide, organic blends
  - Stoichiometric calculations for oxygen removal
  - Residual monitoring and control
  - Safety factor adjustments

- **Scale Inhibitors**: Phosphate, polymer, chelant, and combination programs
  - Hardness-based dosing calculations
  - Silica scale prevention
  - Coordinated phosphate control for high-pressure boilers

- **pH Control**: Alkalinity builders (caustic soda, soda ash, TSP)
  - Target pH maintenance
  - Caustic stress prevention
  - Coordinated control with other treatments

- **Condensate Treatment**: Filming and neutralizing amines
- **Antifoam Agents**: Carryover prevention

### Risk Assessment

- **Scale Formation Risk**
  - Langelier Saturation Index (LSI) calculation
  - Calcium sulfate saturation analysis
  - Silica scaling potential
  - High-risk location identification

- **Corrosion Risk Assessment**
  - Oxygen pitting risk (dissolved oxygen monitoring)
  - Caustic gouging risk (high alkalinity/caustic)
  - Acidic corrosion (low pH, carbonic acid)
  - Stress corrosion cracking
  - Erosion-corrosion in high-velocity areas

### Predictive Maintenance

- **Inspection Scheduling**: Risk-based inspection frequency
- **Tube Life Prediction**: Corrosion and scale impact on tube longevity
- **Chemical Inventory Management**: Automatic reorder point calculations
- **Equipment Health Monitoring**: Analyzer calibration tracking, pump status

### SCADA Integration

- **OPC-UA Support**: Secure, standards-based communication
- **Modbus TCP/RTU**: Legacy equipment integration
- **Real-time Data Acquisition**: Sub-second sampling rates
- **Setpoint Control**: Automated chemical dosing pump control
- **Alarm Management**: Configurable alerts with severity levels

---

## Water Treatment Fundamentals

### Boiler Water Chemistry Overview

Proper boiler water chemistry prevents three primary problems:

#### 1. Scale Formation

**Mechanism**: Dissolved minerals (calcium, magnesium, silica) precipitate out of solution when:
- Water is heated (inverse solubility)
- Water is concentrated through evaporation
- Solubility limits are exceeded

**Common Scale Types**:
- **Calcium Carbonate (CaCO₃)**: Most common, forms at alkaline pH
- **Calcium Sulfate (CaSO₄)**: Forms when sulfate levels are high
- **Magnesium Silicate**: Forms in high-pressure boilers
- **Silica (SiO₂)**: Particularly problematic in turbines

**Consequences**:
- Reduced heat transfer (scale is an insulator)
- Tube overheating and failure
- Increased fuel consumption
- Reduced boiler capacity

**Prevention**:
- Water softening or demineralization
- Chemical scale inhibitors (phosphate, polymers, chelants)
- Blowdown to control concentration
- pH control

#### 2. Corrosion

**Mechanism**: Electrochemical attack of metal surfaces by:
- **Dissolved Oxygen**: Primary corrosion agent in boilers
- **Low pH**: Acidic attack of metal
- **High pH**: Caustic attack (caustic embrittlement)

**Common Corrosion Types**:
- **Oxygen Pitting**: Localized attack creating deep pits
- **General Corrosion**: Uniform metal loss across surfaces
- **Caustic Gouging**: High-alkalinity attack in high-heat-flux areas
- **Stress Corrosion Cracking**: Combined stress and chemical attack

**Consequences**:
- Tube failures and leaks
- Contamination of steam and condensate
- Unplanned outages
- Safety hazards

**Prevention**:
- Oxygen removal (mechanical deaeration + chemical scavenging)
- pH control (alkaline conditions, typically pH 10.5-11.5)
- Protective film formation (phosphate, amine)
- Materials selection

#### 3. Carryover

**Mechanism**: Liquid water or dissolved solids carried over with steam due to:
- Excessive TDS (foaming)
- Alkalinity too high (foaming)
- Organic contamination
- Mechanical issues (high water level, inadequate steam separation)

**Consequences**:
- Turbine deposits (if steam turbine)
- Process contamination
- Reduced steam quality
- Equipment damage

**Prevention**:
- TDS control through blowdown
- Antifoam chemicals
- Mechanical separators
- Proper water level control

### Treatment Programs

#### Phosphate Treatment (Low-Medium Pressure)

Best for: Boilers < 1000 psig with hard makeup water

**Mechanism**: Phosphate reacts with hardness to form insoluble hydroxyapatite sludge that can be removed by blowdown
```
10Ca²⁺ + 6PO₄³⁻ + 2OH⁻ → Ca₁₀(PO₄)₆(OH)₂ (hydroxyapatite)
```

**Advantages**:
- Effective hardness precipitation
- Provides alkalinity control
- Relatively simple to manage

**Limitations**:
- Not suitable for high-pressure boilers (phosphate hideout)
- Requires regular sludge removal
- Can form damaging deposits if pH not controlled

**Target Ranges** (ASME standards):
- Boiler Water pH: 10.5-11.5
- Phosphate (as PO₄): 15-50 ppm (pressure-dependent)
- Alkalinity: Coordinated with phosphate

#### All-Volatile Treatment (AVT) (High Pressure)

Best for: Boilers > 1000 psig, especially with steam turbines

**Mechanism**: Uses only volatile chemicals (ammonia, amines, hydrazine) that don't leave residues

**Advantages**:
- No solid residues
- Ideal for superheated steam
- Turbine-safe

**Limitations**:
- Requires very high-quality feedwater (< 0.1 ppm hardness)
- More expensive pretreatment needed
- pH control can be challenging

**Target Ranges**:
- Boiler Water pH: 9.5-10.0
- Cation Conductivity: < 0.2 µS/cm
- Silica: < 20 ppb

#### Coordinated Phosphate Treatment (Medium-High Pressure)

Best for: Boilers 600-1500 psig

**Mechanism**: Maintains specific phosphate-to-pH ratio to prevent caustic attack

**Congruent Control**: pH and phosphate maintained in a ratio that prevents free caustic
```
Congruent pH = 11.0 + 0.05 × [PO₄ concentration]
```

**Advantages**:
- Prevents caustic attack
- Provides hardness precipitation
- Suitable for moderate to high pressures

---

## Architecture Overview

WATERGUARD employs a modular, event-driven architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                     GL-016 WATERGUARD                           │
│                 Boiler Water Treatment Agent                     │
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
Config   Models   Calculators   Tools     SCADA      ERP
```

### Core Components

1. **Configuration Layer** (`config.py`)
   - Pydantic models for type-safe configuration
   - Boiler specifications and operating parameters
   - Water quality limits (ASME/ABMA standards)
   - Chemical inventory management
   - Integration settings

2. **Water Chemistry Calculators** (`calculators/`)
   - Cycles of concentration (COC)
   - Langelier Saturation Index (LSI)
   - Blowdown optimization
   - Chemical dosing calculations
   - Water balance
   - Energy efficiency impact

3. **SCADA Integration** (`integrations/scada_integration.py`)
   - OPC-UA and Modbus client
   - Real-time tag monitoring
   - Alarm management
   - Historical data retrieval
   - Setpoint control

4. **Risk Assessment Engine**
   - Scale formation risk scoring
   - Corrosion mechanism identification
   - Predictive analytics

5. **Optimization Engine**
   - Multi-objective optimization (water, energy, cost)
   - Constraint-based optimization (ASME limits)
   - ROI calculation

6. **Reporting & Analytics**
   - Performance dashboards (Grafana)
   - Compliance reports
   - Trend analysis
   - Incident tracking

### Data Flow

```
SCADA System → Data Acquisition → Validation →
  → Calculation Engine → Risk Assessment →
    → Optimization → Control Actions → SCADA System
```

**Key Principles**:
- **Deterministic Calculations**: All water chemistry and blowdown calculations use deterministic algorithms (no LLM hallucinations)
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
- **Memory**: 2 GB RAM minimum
- **CPU**: 2 cores minimum
- **Network**: Access to SCADA/DCS system

### Option 1: Docker Installation (Recommended)

```bash
# Pull the Docker image
docker pull greenlang/gl-016-waterguard:latest

# Run the container
docker run -d \
  --name waterguard \
  -p 8000:8000 \
  -p 9090:9090 \
  -e SCADA_ENDPOINT=opc.tcp://scada-server:4840 \
  -e LOG_LEVEL=INFO \
  -v /path/to/config:/app/config \
  greenlang/gl-016-waterguard:latest
```

### Option 2: Python Package Installation

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Clone repository
git clone https://github.com/greenlang/gl-016-waterguard.git
cd gl-016-waterguard

# Install dependencies
pip install -r requirements.txt

# Install agent
pip install -e .
```

### Option 3: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/gl-016-waterguard.git
cd gl-016-waterguard

# Install dependencies
pip install -r requirements.txt

# Run agent
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check version
python -c "from greenlang.GL_016 import __version__; print(__version__)"

# Health check
curl http://localhost:8000/health
```

---

## Configuration

### Configuration File Structure

WATERGUARD uses a YAML-based configuration system. Create a `config/waterguard.yaml` file:

```yaml
agent:
  agent_id: GL-016
  agent_name: WATERGUARD
  version: 1.0.0
  monitoring_interval_seconds: 60

boilers:
  - boiler_id: BOILER-001
    boiler_type: watertube
    operating_pressure_psig: 150
    operating_temperature_f: 366
    steam_capacity_lb_hr: 50000
    makeup_water_rate_gpm: 10
    condensate_return_pct: 85
    water_source: demineralized
    treatment_program: phosphate
    design_cycles_of_concentration: 15

scada_integration:
  scada_system_name: Plant SCADA
  protocol: OPC-UA
  server_address: 192.168.1.100
  server_port: 4840
  authentication_required: true
  username: waterguard_user
  enable_ssl: true
  polling_interval_seconds: 5

  water_analyzers:
    - analyzer_id: ANALYZER-PH-001
      analyzer_type: ph
      measurement_parameter: pH
      scada_tag: BOILER.FEEDWATER.PH
      measurement_range_min: 0.0
      measurement_range_max: 14.0
      location: Feedwater line

chemical_inventory:
  chemicals:
    - chemical_id: CHEM-TSP-001
      chemical_name: Trisodium Phosphate
      chemical_type: phosphate
      concentration_pct: 30
      density_lb_gal: 10.2
      current_inventory_gallons: 150
      min_inventory_gallons: 50
```

### Environment Variables

```bash
# SCADA Connection
export SCADA_ENDPOINT="opc.tcp://192.168.1.100:4840"
export SCADA_USERNAME="waterguard_user"
export SCADA_PASSWORD="secure_password"

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"

# Performance
export CACHE_TTL_SECONDS="300"
export MAX_PARALLEL_CALCULATIONS="10"

# Security
export ENABLE_AUTH="true"
export JWT_SECRET_KEY="your-secret-key"

# Determinism
export DETERMINISTIC_MODE="true"
export ZERO_HALLUCINATION="true"
export RANDOM_SEED="42"
```

### Water Quality Limits Configuration

Limits are automatically configured based on boiler pressure and treatment program following ASME standards. Override if needed:

```yaml
water_quality_limits:
  BOILER-001:
    ph_min: 10.5
    ph_max: 11.5
    total_dissolved_solids_max_ppm: 3500
    silica_max_ppm: 150
    total_hardness_max_ppm: 0.3
    dissolved_oxygen_max_ppb: 7
    phosphate_min_ppm: 20
    phosphate_max_ppm: 60
```

---

## Usage

### Quick Start

See [QUICKSTART.md](QUICKSTART.md) for a 10-minute getting started guide.

### Basic Operations

#### Start the Agent

```bash
# Using Docker
docker start waterguard

# Using Python
python main.py --config config/waterguard.yaml
```

#### Monitor Water Quality

```python
from greenlang.GL_016 import BoilerWaterTreatmentAgent, WaterChemistryData

# Initialize agent
agent = BoilerWaterTreatmentAgent(config_path="config/waterguard.yaml")

# Submit water chemistry data
water_data = WaterChemistryData(
    boiler_id="BOILER-001",
    timestamp="2025-12-02T10:30:00Z",
    boiler_water_ph=11.2,
    boiler_water_conductivity=3200,
    total_dissolved_solids_ppm=3000,
    dissolved_oxygen_ppb=5.2,
    phosphate_ppm=35,
    silica_ppm=120
)

# Analyze water chemistry
result = await agent.analyze_water_chemistry(water_data)

print(f"Compliance Status: {result.water_quality_compliant}")
print(f"Deviations: {result.deviations}")
print(f"Recommended Actions: {result.control_actions}")
```

#### Optimize Blowdown

```python
from greenlang.GL_016 import BlowdownData

blowdown_data = BlowdownData(
    boiler_id="BOILER-001",
    timestamp="2025-12-02T10:30:00Z",
    total_blowdown_rate_kg_hr=500,
    cycles_of_concentration=12.5,
    control_mode="TDS_CONTROL"
)

# Get blowdown optimization recommendations
optimization = await agent.optimize_blowdown_rate(
    water_chemistry=water_data,
    blowdown_parameters=blowdown_data,
    boiler_operating_parameters=operating_data
)

print(f"Optimal COC: {optimization.optimal_cycles_of_concentration}")
print(f"Optimal Blowdown Rate: {optimization.optimal_blowdown_rate_kg_hr} kg/hr")
print(f"Annual Water Savings: {optimization.annual_water_savings_m3} m³")
print(f"Annual Cost Savings: ${optimization.annual_cost_savings_usd}")
```

#### Assess Scale Risk

```python
# Assess scale formation risk
scale_risk = await agent.assess_scale_risk(
    water_chemistry=water_data,
    boiler_operating_parameters=operating_data
)

print(f"Overall Scale Risk: {scale_risk.overall_scale_risk_level}")
print(f"Scale Risk Score: {scale_risk.scale_risk_score}/100")
print(f"LSI: {scale_risk.scale_forming_potential.calcium_carbonate_scaling_index}")
print(f"Mitigation Actions: {scale_risk.mitigation_actions}")
```

### REST API Usage

WATERGUARD exposes a REST API on port 8000:

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "agent_id": "GL-016",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "scada_connected": true
}
```

#### Submit Water Chemistry Data

```bash
curl -X POST http://localhost:8000/api/v1/water-chemistry \
  -H "Content-Type: application/json" \
  -d '{
    "boiler_id": "BOILER-001",
    "timestamp": "2025-12-02T10:30:00Z",
    "boiler_water_ph": 11.2,
    "boiler_water_conductivity": 3200,
    "total_dissolved_solids_ppm": 3000,
    "dissolved_oxygen_ppb": 5.2
  }'
```

#### Get Blowdown Optimization

```bash
curl -X POST http://localhost:8000/api/v1/optimize-blowdown \
  -H "Content-Type: application/json" \
  -d @blowdown_request.json
```

#### Get Performance Report

```bash
curl http://localhost:8000/api/v1/performance-report/BOILER-001?period=24h
```

---

## API Reference

### Core Classes

#### BoilerWaterTreatmentAgent

Main orchestrator for water treatment operations.

```python
class BoilerWaterTreatmentAgent:
    def __init__(self, config_path: str):
        """Initialize agent with configuration."""

    async def analyze_water_chemistry(
        self,
        water_chemistry: WaterChemistryData,
        boiler_design: BoilerDesignSpecifications
    ) -> WaterTreatmentResult:
        """Analyze water chemistry against standards."""

    async def optimize_blowdown_rate(
        self,
        water_chemistry: WaterChemistryData,
        blowdown_parameters: BlowdownData,
        boiler_operating_parameters: BoilerOperatingParameters,
        cost_parameters: Optional[CostParameters] = None
    ) -> BlowdownOptimizationResult:
        """Calculate optimal blowdown rate."""

    async def optimize_chemical_dosing(
        self,
        water_chemistry: WaterChemistryData,
        chemical_dosing: ChemicalDosingData,
        boiler_operating_parameters: BoilerOperatingParameters
    ) -> ChemicalOptimizationResult:
        """Optimize chemical dosing rates."""

    async def assess_scale_risk(
        self,
        water_chemistry: WaterChemistryData,
        boiler_operating_parameters: BoilerOperatingParameters
    ) -> ScaleCorrosionRiskAssessment:
        """Assess scale formation risk."""
```

### Data Models

#### WaterChemistryData

```python
class WaterChemistryData(BaseModel):
    boiler_id: str
    timestamp: datetime
    boiler_water_ph: float  # 6.0-14.0
    feedwater_ph: float
    boiler_water_conductivity: float  # µS/cm
    total_dissolved_solids_ppm: float
    dissolved_oxygen_ppb: float
    silica_ppm: float
    phosphate_ppm: Optional[float]
    # ... 20+ more parameters
```

See [TOOL_SPECIFICATIONS.md](TOOL_SPECIFICATIONS.md) for complete API documentation.

---

## Integration Guides

### SCADA Integration

WATERGUARD supports multiple industrial communication protocols:

#### OPC-UA Integration

```python
from greenlang.GL_016.integrations import create_scada_client, ConnectionProtocol

# Create OPC-UA client
scada_client = create_scada_client(
    protocol=ConnectionProtocol.OPC_UA,
    host="192.168.1.100",
    port=4840,
    username="waterguard",
    password="password",
    enable_ssl=True
)

# Connect
await scada_client.connect()

# Register water quality tags
from greenlang.GL_016.integrations import create_standard_water_tags
tags = create_standard_water_tags()
scada_client.register_tags(tags)

# Subscribe to real-time updates
async def on_ph_change(data_point):
    print(f"pH changed: {data_point.value}")

await scada_client.subscribe_tag("FW_PH_01", on_ph_change)

# Read current values
ph_value = await scada_client.read_tag("FW_PH_01")
print(f"Current pH: {ph_value.value}")

# Write setpoint
await scada_client.write_tag("CHEM_DOSE_SETPOINT", 25.5)
```

#### Modbus TCP Integration

```python
# Create Modbus client
scada_client = create_scada_client(
    protocol=ConnectionProtocol.MODBUS_TCP,
    host="192.168.1.101",
    port=502,
    modbus_unit_id=1
)

await scada_client.connect()

# Read holding register
value = await scada_client.read_tag("40001")
```

### Water Analyzer Integration

Supported analyzer types:
- **pH Meters**: Rosemount, Endress+Hauser, Yokogawa
- **Conductivity Meters**: Thornton, ABB, Mettler Toledo
- **Dissolved Oxygen**: Orbisphere, Hach
- **Multi-parameter**: Hach, Swan, Endress+Hauser

### Chemical Dosing System Integration

- **Metering Pumps**: ProMinent, Grundfos, Stenner
- **Proportional Feeders**: Feed-rate proportional to steam flow
- **Control Modes**: Manual, automatic, cascade

### ERP Integration

```python
from greenlang.GL_016.config import ERPIntegration

erp_config = ERPIntegration(
    erp_system_name="SAP",
    api_endpoint="https://erp.company.com/api",
    authentication_type="oauth2",
    enable_chemical_ordering=True,
    enable_cost_tracking=True,
    auto_reorder_threshold_days=14
)

# Agent will automatically create purchase orders when
# chemical inventory falls below threshold
```

### Agent Coordination

WATERGUARD can coordinate with other GreenLang agents:

- **GL-001 THERMOSYNC**: Process heat orchestration
- **GL-002**: Boiler efficiency optimization
- **GL-013 PREDICTMAINT**: Predictive maintenance scheduling

```yaml
integrations:
  - name: greenlang_agents
    type: agent_coordination
    protocol: message_bus
    agents:
      - GL-001  # THERMOSYNC
      - GL-002  # Boiler efficiency
      - GL-013  # PREDICTMAINT
```

---

## Monitoring and Alerting

### Prometheus Metrics

WATERGUARD exports metrics on port 9090:

```bash
# Scrape metrics
curl http://localhost:9090/metrics
```

**Key Metrics**:
```
# Water quality score (0-100)
waterguard_water_quality_score{boiler_id="BOILER-001"} 95.5

# Cycles of concentration
waterguard_cycles_of_concentration{boiler_id="BOILER-001"} 15.2

# Blowdown rate (kg/hr)
waterguard_blowdown_rate_kg_hr{boiler_id="BOILER-001"} 450

# Scale risk score (0-100)
waterguard_scale_risk_score{boiler_id="BOILER-001"} 25

# Corrosion risk score (0-100)
waterguard_corrosion_risk_score{boiler_id="BOILER-001"} 15

# Chemical cost (USD/hr)
waterguard_chemical_cost_usd_hr{boiler_id="BOILER-001"} 12.50

# Total optimizations
waterguard_optimizations_total 1523

# Alerts by severity and type
waterguard_alerts_total{severity="critical",type="scale_risk"} 2
```

### Grafana Dashboards

Pre-built dashboards available in `monitoring/grafana/`:

1. **Water Treatment Overview**: High-level KPIs
2. **Water Chemistry Monitoring**: Real-time parameter trends
3. **Blowdown Optimization**: COC, blowdown rates, savings
4. **Chemical Dosing Management**: Dosing rates, inventory, costs
5. **Scale and Corrosion Risk**: Risk scores, incident tracking
6. **Performance Metrics**: Efficiency, compliance, savings

Import dashboards:
```bash
# Copy dashboard JSON to Grafana
cp monitoring/grafana/*.json /var/lib/grafana/dashboards/
```

### Alert Configuration

Configure alerts in `config/alerts.yaml`:

```yaml
alerts:
  - name: high_scale_risk
    condition: "scale_risk_score > 75"
    severity: critical
    notification_channels:
      - email: operations@company.com
      - slack: #boiler-alerts

  - name: water_quality_violation
    condition: "water_quality_compliant == false"
    severity: high
    notification_channels:
      - email: water-treatment@company.com
      - pagerduty: waterguard-oncall
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

#### SCADA Connection Failed

**Symptoms**: `ConnectionError: Failed to connect to SCADA`

**Solutions**:
1. Verify SCADA server address and port
2. Check network connectivity: `ping <scada-host>`
3. Verify firewall rules allow OPC-UA (4840) or Modbus (502)
4. Check credentials if authentication enabled
5. Verify OPC-UA security policy matches
6. Check SCADA server logs

```bash
# Test OPC-UA connection
python -c "from asyncua import Client; import asyncio; asyncio.run(Client('opc.tcp://192.168.1.100:4840').connect())"
```

#### Water Quality Violations

**Symptoms**: Persistent alarms for pH, TDS, or other parameters

**Solutions**:
1. Verify analyzer calibration
2. Check if limits are correctly configured for boiler pressure
3. Review blowdown control mode
4. Check chemical dosing pump operation
5. Verify feedwater pretreatment is working

#### Blowdown Rate Too High

**Symptoms**: Excessive blowdown, high water/energy costs

**Solutions**:
1. Review COC calculation accuracy
2. Check conductivity analyzer calibration
3. Verify makeup water quality hasn't changed
4. Review chemical treatment program effectiveness
5. Check for steam leaks (false COC reading)

#### Chemical Dosing Issues

**Symptoms**: Low residuals, out-of-spec water chemistry

**Solutions**:
1. Verify chemical pumps are running
2. Check pump calibration
3. Review dosing rate calculations
4. Check chemical tank levels
5. Verify injection points aren't plugged

### Diagnostic Commands

```bash
# Check agent status
curl http://localhost:8000/health

# Get SCADA connection status
curl http://localhost:8000/api/v1/scada/status

# Get active alarms
curl http://localhost:8000/api/v1/alarms/active

# Get recent water chemistry readings
curl http://localhost:8000/api/v1/water-chemistry/BOILER-001?hours=24

# Force recalibration
curl -X POST http://localhost:8000/api/v1/calibrate

# Export logs
docker logs waterguard > waterguard.log
```

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: DEBUG
  format: json
  enable_trace: true
```

Or via environment:
```bash
export LOG_LEVEL=DEBUG
export ENABLE_TRACE=true
```

### Performance Issues

If experiencing slow response times:

1. Check CPU/memory usage: `docker stats waterguard`
2. Review cache settings: increase `CACHE_TTL_SECONDS`
3. Reduce polling frequency if SCADA saturated
4. Enable batch processing for multiple boilers
5. Check database performance (if using persistent storage)

---

## Contributing

We welcome contributions from the community!

### Development Setup

```bash
# Clone repository
git clone https://github.com/greenlang/gl-016-waterguard.git
cd gl-016-waterguard

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
- **Docstrings**: Google style
- **Test Coverage**: Minimum 80%

### Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run linting: `black . && flake8 . && isort .`
5. Run tests: `pytest tests/`
6. Commit: `git commit -m "feat: add my feature"`
7. Push: `git push origin feature/my-feature`
8. Open a Pull Request

### Commit Message Format

Follow Conventional Commits:

```
feat: add cycles of concentration calculation
fix: correct LSI formula for high-pressure boilers
docs: update SCADA integration guide
test: add tests for chemical dosing optimizer
refactor: simplify water balance calculator
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

- **Main Documentation**: [https://greenlang.io/agents/GL-016](https://greenlang.io/agents/GL-016)
- **API Reference**: [https://api.greenlang.io/GL-016](https://api.greenlang.io/GL-016)
- **Knowledge Base**: [https://support.greenlang.io](https://support.greenlang.io)

#### Contact

- **Technical Support**: support@greenlang.io
- **Sales**: sales@greenlang.io
- **Security Issues**: security@greenlang.io

#### Community

- **Slack**: [greenlang-community.slack.com](https://greenlang-community.slack.com)
- **GitHub Issues**: [github.com/greenlang/gl-016-waterguard/issues](https://github.com/greenlang/gl-016-waterguard/issues)

#### Support Tiers

**Standard Support** (included with license):
- Email support (24-hour response time)
- Documentation access
- Security updates

**Premium Support** (additional cost):
- 4-hour response time
- Phone support
- Dedicated support engineer
- On-site assistance available

**Enterprise Support** (additional cost):
- 1-hour response time
- 24/7 phone support
- Dedicated technical account manager
- Custom integration assistance
- Training and onboarding

---

## Acknowledgments

WATERGUARD development was guided by industry standards and best practices from:

- **ASME** (American Society of Mechanical Engineers)
- **ABMA** (American Boiler Manufacturers Association)
- **NACE** (National Association of Corrosion Engineers)
- **ASTM** (American Society for Testing and Materials)
- **TAPPI** (Technical Association of the Pulp and Paper Industry)

Special thanks to water treatment experts, boiler operators, and industrial engineers who provided domain knowledge and feedback.

---

## Roadmap

### Q2 2026 (v1.0)
- ✅ Core water chemistry monitoring
- ✅ Blowdown optimization
- ✅ Chemical dosing control
- ✅ Scale and corrosion risk assessment
- ✅ SCADA integration (OPC-UA, Modbus)
- ✅ Prometheus metrics and Grafana dashboards

### Q3 2026 (v1.1)
- Machine learning for predictive optimization
- Advanced trend analysis and forecasting
- Mobile app for remote monitoring
- Integration with more ERP systems
- Multi-site management dashboard

### Q4 2026 (v2.0)
- AI-powered root cause analysis
- Automated chemical ordering and inventory
- Advanced predictive maintenance
- Digital twin integration
- Carbon footprint tracking

---

## Version History

### v1.0.0 (December 2025)
- Initial production release
- Core water treatment functionality
- SCADA integration
- Monitoring and alerting
- Comprehensive documentation

---

**GL-016 WATERGUARD** - Intelligent Water Treatment for Industrial Boilers
*Reduce water consumption, chemical costs, and energy waste while ensuring equipment reliability and regulatory compliance.*

For questions or support, contact: support@greenlang.io
