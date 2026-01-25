# GL-015 INSULSCAN Quick Start Guide

Get started with INSULSCAN in 5 minutes. This guide walks you through installation, configuration, and your first thermal inspection analysis.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Your First Inspection](#your-first-inspection)
5. [Understanding Results](#understanding-results)
6. [Common Use Cases](#common-use-cases)
7. [Next Steps](#next-steps)

---

## Prerequisites

Before you begin, ensure you have the following:

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.11 | 3.11+ |
| **Memory** | 4 GB | 8 GB |
| **Disk** | 10 GB | 50 GB |
| **OS** | Linux, Windows, macOS | Ubuntu 22.04 LTS |

### Required Accounts and Access

- GreenLang API credentials (client_id and client_secret)
- Database connection (PostgreSQL)
- Cache connection (Redis)
- Blob storage (S3 or compatible)

### Thermal Camera (Optional for Testing)

If testing with real thermal data:
- FLIR, Fluke, Testo, or compatible IR camera
- Camera SDK installed (for direct integration)

---

## Installation

### Option 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install INSULSCAN
pip install greenlang-gl015

# Verify installation
python -c "from gl_015 import InsulationInspectionAgent; print('OK')"
```

### Option 2: From Source

```bash
# Clone repository
git clone https://github.com/greenlang/agents.git
cd agents/gl-015

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -m gl_015 --version
```

### Option 3: Docker

```bash
# Pull the image
docker pull greenlang/gl-015:latest

# Run container
docker run -d \
  --name gl-015 \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -e REDIS_URL="redis://host:6379/0" \
  greenlang/gl-015:latest

# Verify
curl http://localhost:8000/health
```

---

## Configuration

### Step 1: Create Configuration File

Create `config.yaml` in your working directory:

```yaml
# config.yaml - GL-015 INSULSCAN Configuration

# Agent Settings
agent:
  id: GL-015
  codename: INSULSCAN
  version: "1.0.0"
  environment: development

# Runtime Settings
runtime:
  deterministic: true
  zero_hallucination: true
  temperature: 0.0
  seed: 42

# Heat Loss Thresholds (W/m2)
thresholds:
  heat_flux_minor: 50.0
  heat_flux_moderate: 150.0
  heat_flux_severe: 300.0
  heat_flux_critical: 500.0

# Economic Parameters
economics:
  electricity_cost_per_kwh: 0.10
  natural_gas_cost_per_mmbtu: 4.50
  carbon_cost_per_ton: 50.0
  labor_cost_per_hour: 85.0

# Safety Limits (ASTM C1055)
safety:
  max_safe_surface_temp_c: 60.0
  max_process_temp_c: 800.0

# Logging
logging:
  level: INFO
  format: json
```

### Step 2: Set Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://localhost:5432/insulscan

# Cache
REDIS_URL=redis://localhost:6379/0

# Storage
S3_BUCKET=insulscan-images
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# API Authentication
API_KEY=your_api_key
JWT_SECRET=your_jwt_secret

# Mode
DETERMINISTIC_MODE=true
ZERO_HALLUCINATION=true
```

### Step 3: Validate Configuration

```bash
# Validate configuration
python -m gl_015.validate_config --config config.yaml

# Expected output:
# Configuration valid!
# - Agent: GL-015 INSULSCAN v1.0.0
# - Environment: development
# - Deterministic: enabled
# - Zero-hallucination: enabled
```

---

## Your First Inspection

### API Method: Submit Thermal Image for Analysis

```python
#!/usr/bin/env python3
"""
GL-015 INSULSCAN - Quick Start Example
First thermal inspection analysis
"""

from gl_015 import InsulationInspectionAgent
from gl_015.config import InsulationConfig
from decimal import Decimal

# Step 1: Initialize the agent
config = InsulationConfig(
    deterministic=True,
    zero_hallucination=True
)
agent = InsulationInspectionAgent(config)

# Step 2: Prepare thermal image data
# (In production, this comes from your thermal camera)
thermal_image_data = {
    "image_id": "IMG-2025-001",
    "camera_type": "FLIR",
    "camera_model": "T540",
    # Simulated 10x10 temperature matrix (degrees Celsius)
    "temperature_matrix": [
        [45.0, 46.2, 47.1, 48.5, 52.3, 55.8, 48.2, 46.1, 45.3, 45.0],
        [45.2, 46.5, 47.8, 49.2, 53.1, 56.4, 49.1, 46.8, 45.6, 45.1],
        [45.1, 46.3, 48.2, 50.1, 54.2, 57.1, 50.3, 47.2, 45.8, 45.2],
        [45.3, 46.8, 48.5, 51.2, 55.3, 58.2, 51.5, 47.8, 46.1, 45.3],
        [45.2, 46.5, 48.1, 50.8, 54.8, 57.8, 51.0, 47.5, 45.9, 45.1],
        [45.0, 46.2, 47.5, 49.8, 53.5, 56.5, 49.8, 46.9, 45.5, 45.0],
        [44.9, 45.8, 46.9, 48.5, 51.2, 54.1, 48.2, 46.2, 45.2, 44.9],
        [44.8, 45.5, 46.2, 47.2, 49.1, 51.2, 47.0, 45.8, 45.0, 44.8],
        [44.7, 45.2, 45.8, 46.5, 47.8, 49.2, 46.3, 45.5, 44.9, 44.7],
        [44.6, 45.0, 45.5, 46.0, 47.0, 48.1, 45.9, 45.2, 44.8, 44.6],
    ],
    "min_temp_c": 44.6,
    "max_temp_c": 58.2,
    "avg_temp_c": 48.5,
    "resolution_width": 10,
    "resolution_height": 10,
    "emissivity_setting": 0.90,  # Aluminum jacketing
    "capture_timestamp": "2025-12-01T10:00:00Z"
}

# Step 3: Prepare ambient conditions
ambient_conditions = {
    "ambient_temp_c": 25.0,
    "wind_speed_m_s": 1.5,
    "relative_humidity_percent": 55.0,
    "sky_condition": "overcast"
}

# Step 4: Prepare equipment parameters
equipment_parameters = {
    "equipment_id": "PIPE-001",
    "equipment_type": "pipe",
    "equipment_tag": "10-P-101-A",
    "process_temp_c": 180.0,
    "pipe_diameter_mm": 150.0,
    "pipe_length_m": 10.0,
    "operating_hours_per_year": 8000
}

# Step 5: Prepare insulation specifications
insulation_specifications = {
    "insulation_type": "mineral_wool",
    "insulation_thickness_mm": 75.0,
    "jacket_type": "aluminum",
    "age_years": 12.0
}

# Step 6: Run the analysis
print("Running thermal inspection analysis...")
result = agent.process(
    thermal_image_data=thermal_image_data,
    ambient_conditions=ambient_conditions,
    equipment_parameters=equipment_parameters,
    insulation_specifications=insulation_specifications
)

# Step 7: Display results
print("\n" + "="*60)
print("INSPECTION RESULTS")
print("="*60)

print(f"\nInspection ID: {result.inspection_id}")
print(f"Provenance Hash: {result.provenance_hash}")

print("\n--- Heat Loss Analysis ---")
print(f"Total Heat Loss: {result.heat_loss_analysis.total_heat_loss_w} W")
print(f"Heat Loss per meter: {result.heat_loss_analysis.heat_loss_per_m_w} W/m")
print(f"Annual Energy Loss: {result.heat_loss_analysis.annual_energy_loss_mwh} MWh")
print(f"Annual Energy Cost: ${result.heat_loss_analysis.annual_energy_cost_usd}")
print(f"Thermal Efficiency: {result.heat_loss_analysis.thermal_efficiency_percent}%")

print("\n--- Degradation Assessment ---")
print(f"Overall Condition: {result.degradation_assessment.overall_condition}")
print(f"Condition Score: {result.degradation_assessment.condition_score}/100")
print(f"Degradation Severity: {result.degradation_assessment.degradation_severity}")
print(f"CUI Risk Level: {result.degradation_assessment.cui_risk_level}")
print(f"Remaining Life: {result.degradation_assessment.remaining_life_years} years")

print("\n--- Repair Recommendations ---")
print(f"Priority Ranking: {result.repair_priorities.priority_ranking}")
print(f"Recommended Action: {result.repair_priorities.recommended_action}")
print(f"Estimated Repair Cost: ${result.repair_priorities.estimated_repair_cost_usd}")
print(f"Payback Period: {result.repair_priorities.payback_period_months} months")
print(f"ROI: {result.repair_priorities.roi_percent}%")

print("\n" + "="*60)
print("Analysis complete!")
```

### Running the Example

```bash
# Save the above code as first_inspection.py
python first_inspection.py
```

### Expected Output

```
Running thermal inspection analysis...

============================================================
INSPECTION RESULTS
============================================================

Inspection ID: INS-2025-001-ABC123
Provenance Hash: sha256:abc123def456...

--- Heat Loss Analysis ---
Total Heat Loss: 2450.50 W
Heat Loss per meter: 245.05 W/m
Annual Energy Loss: 19.60 MWh
Annual Energy Cost: $1960.00
Thermal Efficiency: 72.50%

--- Degradation Assessment ---
Overall Condition: fair
Condition Score: 68.5/100
Degradation Severity: moderate
CUI Risk Level: medium
Remaining Life: 5.2 years

--- Repair Recommendations ---
Priority Ranking: 3
Recommended Action: partial_replacement
Estimated Repair Cost: $3500.00
Payback Period: 21 months
ROI: 56.00%

============================================================
Analysis complete!
```

---

## Understanding Results

### Heat Loss Analysis

| Field | Description | Units |
|-------|-------------|-------|
| `total_heat_loss_w` | Total heat lost from the inspected surface | Watts |
| `heat_loss_per_m_w` | Heat loss rate per meter of pipe | W/m |
| `annual_energy_loss_mwh` | Projected annual energy waste | MWh |
| `annual_energy_cost_usd` | Annual cost of wasted energy | USD |
| `thermal_efficiency_percent` | Current vs design efficiency | % |

### Degradation Assessment

| Condition | Score Range | Description |
|-----------|-------------|-------------|
| **Excellent** | 90-100 | New or like-new |
| **Good** | 75-89 | Minor wear |
| **Fair** | 60-74 | Moderate degradation |
| **Poor** | 40-59 | Significant damage |
| **Critical** | 0-39 | Immediate action required |

### CUI Risk Levels

| Level | Risk Score | Action |
|-------|------------|--------|
| **Low** | 0-25 | Routine monitoring |
| **Medium** | 26-50 | Scheduled inspection |
| **High** | 51-75 | Priority inspection |
| **Critical** | 76-100 | Immediate investigation |

### Recommended Actions

| Action | Description |
|--------|-------------|
| `monitor` | Continue routine inspection schedule |
| `spot_repair` | Repair small damaged areas (<1m) |
| `partial_replacement` | Replace damaged sections (1-10m) |
| `full_replacement` | Complete insulation replacement |
| `cui_investigation` | Investigate for corrosion under insulation |

---

## Common Use Cases

### Use Case 1: Direct Heat Loss Calculation

Calculate heat loss without thermal image:

```python
from gl_015.calculators import HeatLossCalculator
from gl_015.calculators.heat_loss_calculator import (
    SurfaceGeometry, SurfaceMaterial
)
from decimal import Decimal

# Initialize calculator
calculator = HeatLossCalculator(precision=6)

# Calculate heat loss
result = calculator.calculate_total_heat_loss(
    process_temperature_c=Decimal("180.0"),
    ambient_temperature_c=Decimal("25.0"),
    surface_temperature_c=Decimal("45.0"),
    outer_diameter_m=Decimal("0.225"),  # 150mm pipe + insulation
    length_m=Decimal("10.0"),
    geometry=SurfaceGeometry.CYLINDER_HORIZONTAL,
    surface_material=SurfaceMaterial.ALUMINUM_JACKETING,
    wind_speed_m_s=Decimal("2.0")
)

print(f"Total Heat Loss: {result.total_heat_loss_w} W")
print(f"Convection: {result.convection_fraction * 100:.1f}%")
print(f"Radiation: {result.radiation_fraction * 100:.1f}%")
```

### Use Case 2: Hotspot Detection

Detect thermal anomalies in an image:

```python
from gl_015.calculators import ThermalImageAnalyzer
from decimal import Decimal

# Initialize analyzer
analyzer = ThermalImageAnalyzer(
    pixel_size_m=Decimal("0.005"),  # 5mm per pixel
    default_emissivity=Decimal("0.90")
)

# Detect hotspots
hotspots = analyzer.detect_hotspots(
    temperature_matrix=thermal_image_data["temperature_matrix"],
    ambient_temperature_c=Decimal("25.0"),
    delta_t_threshold=Decimal("10.0"),  # 10C above ambient
    min_hotspot_pixels=3
)

for hotspot in hotspots:
    print(f"Hotspot {hotspot.hotspot_id}:")
    print(f"  Peak Temperature: {hotspot.peak_temperature_c}C")
    print(f"  Delta-T: {hotspot.delta_t_from_ambient}C")
    print(f"  Severity: {hotspot.severity_score}")
    print(f"  Location: ({hotspot.center_x}, {hotspot.center_y})")
```

### Use Case 3: Repair Cost Estimation

Estimate repair costs and ROI:

```python
from gl_015.calculators import RepairPrioritizationEngine
from gl_015.calculators.repair_prioritization import (
    ThermalDefect, DefectLocation, EconomicParameters,
    EquipmentType, DamageType, InsulationMaterial
)
from decimal import Decimal
from datetime import date

# Initialize engine
engine = RepairPrioritizationEngine(
    economic_params=EconomicParameters(
        energy_cost_per_kwh=Decimal("0.12"),
        operating_hours_per_year=Decimal("8000"),
        discount_rate_percent=Decimal("8.0"),
        carbon_price_per_tonne=Decimal("50.0")
    )
)

# Define defect
defect = ThermalDefect(
    defect_id="DEF-001",
    location=DefectLocation(
        equipment_tag="PIPE-001",
        equipment_type=EquipmentType.PIPE,
        area_code="AREA-A",
        unit_code="UNIT-1"
    ),
    damage_type=DamageType.MISSING,
    length_m=Decimal("5.0"),
    process_temperature_c=Decimal("180.0"),
    surface_temperature_c=Decimal("120.0"),
    ambient_temperature_c=Decimal("25.0"),
    heat_loss_w_per_m=Decimal("450.0"),
    insulation_material=InsulationMaterial.MINERAL_WOOL,
    insulation_thickness_mm=Decimal("75.0")
)

# Calculate ROI
roi_result = engine.calculate_repair_roi(defect)

print(f"Repair Cost: ${roi_result.estimated_repair_cost}")
print(f"Annual Savings: ${roi_result.annual_energy_savings}")
print(f"Simple Payback: {roi_result.simple_payback_years} years")
print(f"NPV (15 years): ${roi_result.npv_over_life}")
print(f"ROI: {roi_result.roi_percent}%")
```

### Use Case 4: Batch Processing

Process multiple inspections:

```python
from gl_015 import InsulationInspectionAgent
from gl_015.config import InsulationConfig

# Initialize agent with batch settings
config = InsulationConfig(
    batch_enabled=True,
    max_batch_size=50,
    parallel_workers=4
)
agent = InsulationInspectionAgent(config)

# Prepare batch data
images = [...]  # List of thermal image data
equipment_list = [...]  # List of equipment parameters
insulation_list = [...]  # List of insulation specifications

# Process batch
batch_result = await agent.process_batch(
    images=images,
    ambient_conditions=ambient_conditions,
    equipment_list=equipment_list,
    insulation_specs=insulation_list
)

print(f"Processed: {batch_result.total_processed}")
print(f"Successful: {batch_result.successful}")
print(f"Failed: {batch_result.failed}")

# Export results
batch_result.export_to_csv("inspection_results.csv")
batch_result.export_to_excel("inspection_results.xlsx")
```

---

## Next Steps

### 1. Explore Full Documentation

- [README.md](README.md) - Complete feature documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design details
- [API Reference](README.md#api-reference) - Full API documentation

### 2. Configure Integrations

Set up connections to your systems:

```python
# Thermal Camera Integration
from gl_015.integrations.thermal_cameras import FLIRConnector

flir = FLIRConnector(
    sdk_path="/opt/flir/atlas",
    camera_ip="192.168.1.100"
)

# CMMS Integration
from gl_015.integrations.cmms import SAPPMConnector

sap = SAPPMConnector(
    host="sap.example.com",
    client="100",
    username="your_user",
    password="your_pass"
)
```

### 3. Set Up Monitoring

Enable observability for production:

```yaml
# Add to config.yaml
monitoring:
  metrics_enabled: true
  metrics_port: 8001
  tracing_enabled: true
  sampling_rate: 0.1
```

### 4. Deploy to Production

See deployment guides:

- [Docker Deployment](README.md#docker-deployment)
- [Kubernetes Deployment](README.md#kubernetes-deployment)

---

## Getting Help

### Resources

- **Documentation:** https://docs.greenlang.io/agents/gl-015
- **API Reference:** https://api.greenlang.io/docs
- **Community Forum:** https://community.greenlang.io

### Support

- **Email:** support@greenlang.io
- **GitHub Issues:** https://github.com/greenlang/agents/issues

### Troubleshooting

Common issues and solutions:

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install greenlang-gl015` |
| `ConnectionError` | Check DATABASE_URL and REDIS_URL |
| `ValidationError` | Verify input data types (use Decimal) |
| `TimeoutError` | Increase timeout or reduce batch size |

---

**You're now ready to start analyzing thermal data with INSULSCAN!**

*For production deployments, review the full [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md) documentation.*
