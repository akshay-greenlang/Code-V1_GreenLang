# GreenLang Quickstart Guide

Welcome to GreenLang - the Climate Intelligence Platform! This comprehensive guide will help you get up and running with GreenLang in under 10 minutes, whether you're a developer looking to integrate climate calculations or a business user wanting to analyze your carbon footprint.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
- Install and configure GreenLang using PyPI or Docker
- Calculate carbon emissions for buildings and processes
- Use the CLI for quick calculations
- Run your first YAML pipeline
- Explore real-world examples

## 📋 Prerequisites & System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher
- **Memory**: 2GB RAM (4GB recommended for large datasets)
- **Storage**: 500MB free space
- **Internet**: Active connection for emission factor updates

### Supported Platforms
- **Operating Systems**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **Container Platforms**: Docker 20.10+, Kubernetes 1.24+
- **Cloud Providers**: AWS, Azure, GCP (with container support)

### Check Your Environment

```bash
# Verify Python version
python --version
# Should output: Python 3.10.x or higher

# Check pip is available
pip --version

# Verify internet connectivity
curl -I https://pypi.org/simple/ || ping pypi.org
```

**Troubleshooting Prerequisites:**
- **Python too old**: Install from [python.org](https://python.org/downloads)
- **pip missing**: Run `python -m ensurepip --upgrade`
- **Windows PATH issues**: Add Python to your system PATH
- **macOS SSL errors**: Run `/Applications/Python\ 3.x/Install\ Certificates.command`

## 🚀 Installation Guide

### Option 1: PyPI Installation (Recommended)

#### Basic Installation
```bash
# Install the latest stable version
pip install greenlang-cli==0.3.0

# Verify installation
gl version
```

**Expected Output:**
```
GreenLang v0.3.0
Infrastructure for Climate Intelligence
https://greenlang.io
```

#### Installation with Extras
```bash
# For data analysis capabilities
pip install greenlang-cli[analytics]==0.3.0

# For full feature set (recommended for production)
pip install greenlang-cli[full]==0.3.0

# For development work
pip install greenlang-cli[dev]==0.3.0
```

#### Virtual Environment (Recommended)
```bash
# Create isolated environment
python -m venv greenlang-env

# Activate environment
# On Windows:
greenlang-env\Scripts\activate
# On macOS/Linux:
source greenlang-env/bin/activate

# Install GreenLang
pip install greenlang-cli[full]==0.3.0
```

### Option 2: Docker Installation

#### Pull and Verify
```bash
# Pull the official image
docker pull ghcr.io/greenlang/greenlang:0.3.0

# Verify installation
docker run --rm ghcr.io/greenlang/greenlang:0.3.0 version
```

#### Docker Compose Setup
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  greenlang:
    image: ghcr.io/greenlang/greenlang:0.3.0
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - GL_LOG_LEVEL=INFO
      - GL_CACHE_DIR=/app/cache
    ports:
      - "8080:8080"
```

```bash
# Start with Docker Compose
docker-compose up -d
```

### Installation Verification

Run the comprehensive health check:
```bash
# Using PyPI installation
gl doctor --verbose

# Using Docker
docker run --rm ghcr.io/greenlang/greenlang:0.3.0 doctor --verbose
```

**Successful Installation Indicators:**
- ✅ No error messages during installation
- ✅ `gl version` shows v0.3.0
- ✅ `gl doctor` passes all checks
- ✅ Python imports work: `python -c "from greenlang.sdk import GreenLangClient"`

## 🧮 Your First Calculation

Let's calculate carbon emissions for a typical office building:

### Method 1: Python SDK (Interactive)

Create `first_calculation.py`:
```python
from greenlang.sdk import GreenLangClient
from greenlang.models import BuildingData, FuelConsumption

# Initialize the client
client = GreenLangClient()

# Define building characteristics
building = BuildingData(
    name="Demo Office Building",
    building_type="commercial_office",
    area_m2=2500,
    location="San Francisco, CA",
    occupancy=150,
    year_built=2015
)

# Define energy consumption
energy_data = [
    FuelConsumption(
        fuel_type="electricity",
        consumption=50000,
        unit="kWh",
        period="annual"
    ),
    FuelConsumption(
        fuel_type="natural_gas",
        consumption=1000,
        unit="therms",
        period="annual"
    )
]

# Calculate carbon footprint
try:
    result = client.calculate_building_emissions(
        building=building,
        energy_consumption=energy_data,
        include_scope3=False  # Start with Scope 1 & 2
    )

    if result.success:
        print(f"🏢 Building: {building.name}")
        print(f"📊 Total Annual Emissions: {result.total_emissions_tons:.2f} metric tons CO2e")
        print(f"📏 Emission Intensity: {result.intensity_per_sqft:.3f} kgCO2e/sqft")
        print(f"⚡ Electricity: {result.breakdown.electricity_emissions:.1f} kg CO2e")
        print(f"🔥 Natural Gas: {result.breakdown.gas_emissions:.1f} kg CO2e")

        # Performance benchmarking
        if result.benchmark:
            print(f"🎯 Performance Rating: {result.benchmark.rating}")
            print(f"📈 Compared to Similar Buildings: {result.benchmark.percentile}th percentile")
    else:
        print(f"❌ Calculation failed: {result.errors}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Try: pip install greenlang-cli[analytics] for enhanced calculations")
```

Run your calculation:
```bash
python first_calculation.py
```

**Expected Output:**
```
🏢 Building: Demo Office Building
📊 Total Annual Emissions: 28.45 metric tons CO2e
📏 Emission Intensity: 0.259 kgCO2e/sqft
⚡ Electricity: 24,500.0 kg CO2e
🔥 Natural Gas: 3,950.0 kg CO2e
🎯 Performance Rating: B
📈 Compared to Similar Buildings: 65th percentile
```

### Method 2: CLI Quick Calculation

```bash
# Simple calculation using CLI
gl calc \
  --fuel-type electricity \
  --consumption 50000 \
  --unit kWh \
  --location "San Francisco" \
  --format json

# With multiple fuel types
gl calc \
  --building-type office \
  --area 2500 \
  --fuels "electricity:50000:kWh,natural_gas:1000:therms" \
  --location "San Francisco" \
  --output results.json
```

### Method 3: Docker Calculation

```bash
# Create input data file
cat > building_data.json << EOF
{
  "building": {
    "name": "Demo Office",
    "type": "commercial_office",
    "area_m2": 2500,
    "location": "San Francisco",
    "occupancy": 150
  },
  "energy_consumption": [
    {
      "fuel_type": "electricity",
      "consumption": 50000,
      "unit": "kWh",
      "period": "annual"
    },
    {
      "fuel_type": "natural_gas",
      "consumption": 1000,
      "unit": "therms",
      "period": "annual"
    }
  ]
}
EOF

# Calculate using Docker
docker run --rm \
  -v $(pwd):/data \
  ghcr.io/greenlang/greenlang:0.3.0 \
  calc --input /data/building_data.json --output /data/results.json

# View results
cat results.json | jq '.'
```

## 🎯 Exploring Core Features

### 1. Building Performance Benchmarking

```python
from greenlang.sdk import GreenLangClient
from greenlang.agents import BenchmarkAgent

client = GreenLangClient()
benchmark_agent = BenchmarkAgent()

# Analyze building performance
result = benchmark_agent.analyze_performance({
    "total_emissions_kg": 28450,  # From previous calculation
    "building_area_m2": 2500,
    "building_type": "commercial_office",
    "location": "San Francisco",
    "year_built": 2015
})

print(f"🎯 Energy Star Score Equivalent: {result.energy_star_score}/100")
print(f"📊 Performance Level: {result.performance_level}")
print(f"🏆 Ranking: {result.percentile}th percentile")
print(f"💡 Improvement Potential: {result.improvement_potential:.1f}% reduction possible")

# Get specific recommendations
for rec in result.recommendations[:3]:  # Top 3 recommendations
    print(f"💡 {rec.title}: {rec.estimated_savings:.1f} tCO2e/year savings")
```

### 2. Real-time Monitoring Setup

```python
from greenlang.sdk import GreenLangClient
from greenlang.monitoring import RealTimeMonitor
from datetime import datetime
import time

client = GreenLangClient()
monitor = RealTimeMonitor(client)

# Set up monitoring for key metrics
building_config = {
    "building_id": "demo-office-001",
    "baseline_emissions": 28.45,  # tCO2e/year
    "alert_threshold": 1.1,  # 10% above baseline
    "monitoring_interval": 300  # 5 minutes
}

# Simulate real-time data collection
def simulate_hourly_monitoring():
    """Simulate 24-hour monitoring cycle"""
    base_consumption = {"electricity": 200, "gas": 5}

    for hour in range(24):
        # Simulate variable load throughout the day
        time_factor = 0.7 + 0.3 * abs(sin(hour * 3.14159 / 12))  # Peak during day

        current_data = {
            "timestamp": datetime.now().isoformat(),
            "electricity_kwh": base_consumption["electricity"] * time_factor,
            "gas_therms": base_consumption["gas"],
            "occupancy": max(0, 150 * time_factor) if 7 <= hour <= 19 else 10
        }

        # Calculate real-time emissions
        result = client.calculate_emissions_realtime(current_data)

        if result.success:
            emissions_rate = result.hourly_emissions_kg
            daily_projection = emissions_rate * 24

            print(f"Hour {hour:02d}: {emissions_rate:.1f} kg CO2e/hr "
                  f"(Daily projection: {daily_projection:.1f} kg)")

            # Check for alerts
            if daily_projection > building_config["baseline_emissions"] * 1000 / 365 * 1.1:
                print(f"⚠️  Alert: Emissions {(daily_projection / (building_config['baseline_emissions'] * 1000 / 365) - 1) * 100:.1f}% above baseline")

        time.sleep(1)  # Simulate time passing

# Run monitoring simulation
print("🔄 Starting 24-hour emissions monitoring simulation...")
simulate_hourly_monitoring()
```

### 3. YAML Pipeline Workflow

Create `carbon_analysis_pipeline.yaml`:
```yaml
version: "1.0"
name: "Comprehensive Carbon Analysis"
description: "End-to-end carbon footprint analysis with optimization recommendations"

inputs:
  building_data:
    type: object
    required: true
    description: "Building characteristics and energy consumption data"

stages:
  - name: data_validation
    type: validation
    agent: data_validator
    parameters:
      strict_mode: true
      validate_emission_factors: true
    outputs:
      - validated_data

  - name: emissions_calculation
    type: calculation
    agent: building_emissions_agent
    depends_on: [data_validation]
    parameters:
      include_scope3: false
      use_regional_factors: true
      calculation_method: "ghg_protocol"
    outputs:
      - total_emissions
      - emissions_breakdown

  - name: benchmarking
    type: analysis
    agent: benchmark_agent
    depends_on: [emissions_calculation]
    parameters:
      comparison_dataset: "energy_star"
      include_peer_analysis: true
    outputs:
      - performance_rating
      - peer_comparison

  - name: optimization_analysis
    type: optimization
    agent: building_optimizer
    depends_on: [emissions_calculation, benchmarking]
    parameters:
      target_reduction: 0.30  # 30% reduction goal
      max_payback_years: 7
      include_renewable_options: true
    outputs:
      - optimization_recommendations
      - cost_benefit_analysis

  - name: report_generation
    type: reporting
    agent: report_generator
    depends_on: [optimization_analysis]
    parameters:
      format: ["pdf", "json", "excel"]
      template: "executive_summary"
      include_charts: true
    outputs:
      - executive_report
      - detailed_analysis
      - action_plan

outputs:
  summary:
    total_emissions: "{{ stages.emissions_calculation.outputs.total_emissions }}"
    performance_rating: "{{ stages.benchmarking.outputs.performance_rating }}"
    top_recommendations: "{{ stages.optimization_analysis.outputs.optimization_recommendations[:3] }}"

  reports:
    executive_summary: "{{ stages.report_generation.outputs.executive_report }}"
    detailed_analysis: "{{ stages.report_generation.outputs.detailed_analysis }}"

error_handling:
  retry_failed_stages: true
  max_retries: 3
  fallback_calculations: true
```

Run the pipeline:
```bash
# Using CLI (when available in production)
gl pipeline run carbon_analysis_pipeline.yaml \
  --input building_data.json \
  --output analysis_results/

# Using Python SDK
python -c "
from greenlang.pipelines import PipelineRunner
import json

# Load pipeline and data
runner = PipelineRunner('carbon_analysis_pipeline.yaml')
with open('building_data.json') as f:
    input_data = json.load(f)

# Execute pipeline
result = runner.run(input_data)
print(f'Pipeline completed: {result.success}')
print(f'Total emissions: {result.outputs.summary.total_emissions} tCO2e')
"
```

## 🌍 Use Case Examples

### Smart Building Optimization

```python
from greenlang.sdk import GreenLangClient
from greenlang.agents import HVACOptimizer, LightingOptimizer

client = GreenLangClient()

# Analyze HVAC system efficiency
hvac_optimizer = HVACOptimizer()
hvac_analysis = hvac_optimizer.analyze_system({
    "building_area": 2500,
    "hvac_type": "central_air",
    "age_years": 8,
    "efficiency_rating": 13,  # SEER
    "annual_runtime_hours": 2800,
    "current_consumption_kwh": 35000
})

print(f"🌡️ HVAC Optimization Potential:")
print(f"   Current Efficiency: {hvac_analysis.current_efficiency:.1f} SEER")
print(f"   Recommended Upgrade: {hvac_analysis.recommended_system}")
print(f"   Potential Savings: {hvac_analysis.annual_savings_kwh:,.0f} kWh/year")
print(f"   CO2 Reduction: {hvac_analysis.co2_savings_tons:.1f} tCO2e/year")
print(f"   ROI Payback: {hvac_analysis.payback_years:.1f} years")
```

### Industrial Process Analysis

```python
from greenlang.agents import IndustrialProcessAgent

# Analyze manufacturing process emissions
process_agent = IndustrialProcessAgent()
result = process_agent.analyze_process({
    "process_type": "steel_production",
    "annual_output_tons": 50000,
    "energy_inputs": [
        {"type": "coal", "consumption": 75000, "unit": "tons"},
        {"type": "electricity", "consumption": 120000, "unit": "MWh"},
        {"type": "natural_gas", "consumption": 8000, "unit": "MCF"}
    ],
    "location": "Pittsburgh, PA"
})

print(f"🏭 Industrial Process Analysis:")
print(f"   Total Process Emissions: {result.total_emissions_tons:,.0f} tCO2e/year")
print(f"   Emission Intensity: {result.intensity_per_output:.2f} tCO2e/ton product")
print(f"   Industry Benchmark: {result.benchmark_percentile}th percentile")
```

### Renewable Energy Planning

```python
from greenlang.agents import SolarOptimizer, RenewableEnergyAgent

# Assess solar installation potential
solar_optimizer = SolarOptimizer()
solar_analysis = solar_optimizer.assess_potential({
    "building_location": "San Francisco, CA",
    "roof_area_m2": 800,
    "roof_orientation": "south",
    "roof_tilt": 30,
    "annual_electricity_consumption": 50000,  # kWh
    "current_electricity_rate": 0.28  # $/kWh
})

print(f"☀️ Solar Installation Analysis:")
print(f"   Recommended System Size: {solar_analysis.recommended_capacity_kw:.1f} kW")
print(f"   Annual Generation: {solar_analysis.annual_generation_kwh:,.0f} kWh")
print(f"   Grid Independence: {solar_analysis.grid_independence_percent:.1f}%")
print(f"   Annual CO2 Savings: {solar_analysis.co2_savings_tons:.1f} tCO2e")
print(f"   Financial Payback: {solar_analysis.financial_payback_years:.1f} years")
```

## 🔧 Common Use Cases & Troubleshooting

### Common Issues and Solutions

#### Installation Issues
```bash
# Issue: Permission denied during pip install
# Solution: Use user installation
pip install --user greenlang-cli==0.3.0

# Issue: Package conflicts
# Solution: Use virtual environment
python -m venv fresh-env
source fresh-env/bin/activate  # Linux/macOS
# or fresh-env\Scripts\activate  # Windows
pip install greenlang-cli==0.3.0

# Issue: Docker permission denied
# Solution: Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Then logout and login again
```

#### Runtime Issues
```bash
# Issue: Import errors
# Solution: Verify installation and dependencies
python -c "import greenlang; print('✅ GreenLang imported successfully')"

# Issue: Calculation errors
# Solution: Enable debug mode
export GL_DEBUG=1
python your_script.py

# Issue: Network timeouts
# Solution: Configure timeout and retry settings
export GL_NETWORK_TIMEOUT=30
export GL_MAX_RETRIES=3
```

#### Data Issues
```bash
# Issue: Invalid emission factors
# Solution: Update emission factor database
gl update-emission-factors --region US

# Issue: Unrealistic calculation results
# Solution: Validate input data
gl validate-data --input your_data.json --strict

# Issue: Missing regional data
# Solution: Use fallback region or global averages
gl calc --fallback-region global --input your_data.json
```

### Performance Optimization

```python
# For large datasets, use batch processing
from greenlang.sdk import GreenLangClient
from greenlang.utils import BatchProcessor

client = GreenLangClient()
batch_processor = BatchProcessor(client, batch_size=100)

# Process multiple buildings efficiently
buildings = [...]  # List of building data
results = batch_processor.calculate_emissions_batch(buildings)

# Use caching for repeated calculations
client.enable_cache(cache_duration=3600)  # 1 hour cache
```

## 🎉 Next Steps

Congratulations! You've successfully set up GreenLang and run your first calculations. Here's what to explore next:

### Immediate Next Steps (Next 30 minutes)
1. **Try the Examples**: Run the complete examples in `examples/quickstart/`
2. **Explore CLI Commands**: Run `gl --help` to see all available commands
3. **Test Different Buildings**: Try different building types and locations
4. **Setup Monitoring**: Configure real-time emissions monitoring

### Short-term Goals (Next Week)
1. **Custom Calculations**: Modify the examples for your specific use case
2. **Pipeline Development**: Create your first custom YAML pipeline
3. **Integration Testing**: Connect GreenLang to your existing systems
4. **Advanced Features**: Explore machine learning optimization features

### Long-term Development (Next Month)
1. **Custom Agents**: Develop specialized agents for your industry
2. **Production Deployment**: Set up GreenLang in your production environment
3. **Team Integration**: Train your team and establish best practices
4. **Contribute Back**: Share your improvements with the community

### Key Resources for Next Steps
- **[Complete Examples Library](../examples/quickstart/)** - Ready-to-run code
- **[API Documentation](https://greenlang.io/api)** - Complete reference
- **[Community Discord](https://discord.gg/greenlang)** - Get help and share ideas
- **[GitHub Discussions](https://github.com/greenlang/greenlang/discussions)** - Feature requests and Q&A

## 📞 Getting Help

### Community Support
- **Discord Community**: [discord.gg/greenlang](https://discord.gg/greenlang) - Real-time help
- **GitHub Issues**: [github.com/greenlang/greenlang/issues](https://github.com/greenlang/greenlang/issues) - Bug reports
- **Stack Overflow**: Tag questions with `greenlang` - Community Q&A

### Professional Support
- **Enterprise Support**: Contact support@greenlang.io for SLA-backed assistance
- **Consulting Services**: Professional implementation and optimization services
- **Training Programs**: Team training and certification programs

### Documentation and Resources
- **API Reference**: Complete technical documentation
- **Video Tutorials**: Step-by-step guided tutorials
- **Best Practices Guide**: Production deployment recommendations
- **Industry Examples**: Sector-specific implementation guides

---

**Ready to make an impact? Start calculating, optimizing, and reducing emissions today! 🌍**

*Built with GreenLang v0.3.0 - The Climate Intelligence Platform*