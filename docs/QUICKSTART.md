# GreenLang 10-Minute Quickstart Guide

Get up and running with GreenLang - the Climate Intelligence Framework - in just 10 minutes. This guide targets the current version (v0.3.0) and will help you perform your first carbon calculations and explore core features.

## 🚀 Prerequisites Check (1 minute)

Before we start, let's make sure you have everything ready:

### System Requirements
```bash
# Check Python version (3.10+ required)
python --version
# Should show Python 3.10.x or higher

# Check pip is working
pip --version

# Verify internet connection for package downloads
curl -I https://pypi.org/simple/
```

**✅ Success Indicators:**
- Python 3.10 or higher installed
- pip is available and working
- Internet connection is active

**❌ Troubleshooting:**
- **Python too old**: Install Python 3.10+ from [python.org](https://python.org/downloads)
- **pip missing**: Run `python -m ensurepip --upgrade`
- **Windows users**: Make sure Python is in your PATH

## 📦 Installation (2 minutes)

### Option 1: Quick Install with pip (Recommended)
```bash
# Basic installation
pip install greenlang-cli

# Verify installation
gl version
```

**Expected output:**
```
GreenLang v0.3.0
Infrastructure for Climate Intelligence
https://greenlang.io
```

### Option 2: Full Installation with Analytics
```bash
# Install with analytics capabilities
pip install greenlang-cli[analytics]

# Or install everything
pip install greenlang-cli[full]
```

### Option 3: Docker Installation
```bash
# Pull the latest Docker image
docker pull greenlang/core:v0.3.0

# Test the installation
docker run --rm greenlang/core:v0.3.0 version
```

**✅ Success Indicators:**
- `gl version` command works
- No error messages during installation
- Version shows v0.3.0

**❌ Troubleshooting:**
- **Permission denied**: Use `pip install --user greenlang-cli`
- **Package not found**: Update pip with `pip install --upgrade pip`
- **Docker issues**: Ensure Docker is running

## 🧮 First Calculation Example (3 minutes)

Let's calculate carbon emissions for a simple office building:

### Step 1: Environment Check
```bash
# Run the built-in health check
gl doctor
```

**Expected output:**
```
GreenLang Environment Check

[OK] GreenLang Version: v0.3.0
[OK] Python Version: 3.11.x
[OK] Config Directory: /home/user/.greenlang

All checks passed!
```

### Step 2: Create Sample Data
Create a file called `office_building.json`:

```json
{
  "building": {
    "name": "Tech Office A",
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
  ],
  "period": {
    "year": 2024,
    "duration_months": 12
  }
}
```

### Step 3: Run Your First Calculation
```bash
# Initialize a new project (optional but recommended)
gl init quickstart-demo --path .

# Calculate emissions using Python SDK
python3 -c "
from greenlang.sdk import GreenLangClient

# Create client
client = GreenLangClient()

# Sample data
fuels = [
    {'fuel_type': 'electricity', 'consumption': 50000, 'unit': 'kWh'},
    {'fuel_type': 'natural_gas', 'consumption': 1000, 'unit': 'therms'}
]

building_info = {
    'type': 'commercial_office',
    'area': 2500,
    'occupancy': 150
}

# Calculate carbon footprint
result = client.calculate_carbon_footprint(fuels, building_info)

if result['success']:
    print(f'✅ Total Emissions: {result[\"data\"][\"total_emissions_tons\"]:.2f} metric tons CO2e')
    print(f'📊 Emission Breakdown:')
    for item in result['data'].get('breakdown', []):
        print(f'   - {item[\"fuel_type\"]}: {item[\"emissions_kg\"]:.0f} kg CO2e')
else:
    print(f'❌ Error: {result[\"errors\"]}')
"
```

**✅ Success Indicators:**
- No import errors
- Calculation completes successfully
- Shows realistic emission values (10-50 tons for the example)

**❌ Troubleshooting:**
- **Import error**: Try `pip install greenlang-cli --upgrade`
- **Calculation fails**: Check your JSON syntax and data values
- **Unrealistic values**: Verify fuel consumption units

## 🔍 Exploring Core Features (4 minutes)

### Feature 1: Building Performance Benchmarking
```python
# Create advanced_example.py
from greenlang.sdk import GreenLangClient
from greenlang.agents.benchmark import BenchmarkAgent

client = GreenLangClient()

# Calculate with benchmarking
benchmark_agent = BenchmarkAgent()
result = benchmark_agent.analyze_building({
    "total_emissions_kg": 45000,  # From previous calculation
    "building_area": 2500,
    "building_type": "commercial_office",
    "location": "San Francisco"
})

print(f"🎯 Benchmark Rating: {result['data']['rating']}")
print(f"📈 Performance: {result['data']['performance_level']}")
print(f"💡 Recommendations: {len(result['data']['recommendations'])} available")
```

### Feature 2: YAML Workflows
Create `simple_workflow.yaml`:

```yaml
version: "1.0"
name: "Quick Carbon Analysis"

stages:
  - name: emissions_calculation
    type: calculation
    agent: fuel_agent
    parameters:
      include_scope3: false
      region: "US"

  - name: intensity_calculation
    type: analysis
    agent: carbon_agent
    parameters:
      normalize_by_area: true

  - name: benchmarking
    type: benchmark
    agent: benchmark_agent
    parameters:
      building_type: "office"

outputs:
  - total_emissions
  - intensity_per_sqft
  - benchmark_rating
```

Run the workflow:
```bash
# This will be available in future releases
# gl pipeline run simple_workflow.yaml --input office_building.json
```

### Feature 3: Real-time Monitoring (Preview)
```python
# Preview of monitoring capabilities
from greenlang.sdk import GreenLangClient
from datetime import datetime

client = GreenLangClient()

# Simulate hourly monitoring
for hour in range(24):
    reading = {
        "timestamp": datetime.now().isoformat(),
        "electricity_kwh": 200 + hour * 10,  # Variable load
        "gas_therms": 5
    }

    # Quick emission check
    result = client.calculate_emissions("electricity", reading["electricity_kwh"], "kWh")

    if result["success"] and hour % 6 == 0:  # Report every 6 hours
        print(f"Hour {hour}: {result['data']['emissions_kg']:.1f} kg CO2e")
```

**✅ Success Indicators:**
- Benchmark analysis provides meaningful ratings
- YAML workflows validate (even if not fully executable yet)
- Monitoring code runs without errors

## 🎯 Success! What's Next?

Congratulations! You've successfully:
- ✅ Installed GreenLang CLI v0.3.0
- ✅ Calculated your first carbon footprint
- ✅ Explored benchmarking capabilities
- ✅ Previewed advanced features

### Next Steps:

**Immediate (5 minutes):**
```bash
# Explore available agents
python3 -c "
from greenlang.agents import get_available_agents
agents = get_available_agents()
print('Available agents:')
for agent in agents:
    print(f'  - {agent}')
"

# Test different building types
gl doctor --verbose  # More detailed system check
```

**Short Term (30 minutes):**
- Try the [Example Gallery](https://greenlang.io/examples)
- Read [Building Agent Deep Dive](https://greenlang.io/docs/agents/building)
- Experiment with different fuel types and building configurations

**Long Term (1+ hours):**
- Build a custom agent for your specific use case
- Set up automated monitoring workflows
- Integrate with your existing building management systems
- Explore the [Pack ecosystem](https://greenlang.io/packs)

## 🔗 Important Links

### Documentation
- **Full Documentation**: [https://greenlang.io/docs](https://greenlang.io/docs)
- **API Reference**: [https://greenlang.io/api](https://greenlang.io/api)
- **Best Practices**: [https://greenlang.io/best-practices](https://greenlang.io/best-practices)

### Examples & Tutorials
- **Example Gallery**: [https://greenlang.io/examples](https://greenlang.io/examples)
- **Building Analysis Tutorial**: [./examples/tutorials/](./examples/tutorials/)
- **Custom Agent Tutorial**: [./examples/tutorials/custom_agent_30_lines.py](./examples/tutorials/custom_agent_30_lines.py)

### Community & Support
- **Discord Community**: [https://discord.gg/greenlang](https://discord.gg/greenlang)
- **GitHub Issues**: [https://github.com/greenlang/greenlang/issues](https://github.com/greenlang/greenlang/issues)
- **Stack Overflow**: Tag questions with `greenlang`

### Data & Standards
- **Global Emission Factors**: Included for 12+ major economies
- **Building Benchmarks**: Based on ENERGY STAR and local standards
- **Industry Standards**: Compliant with GHG Protocol and ISO 14064

## 🆘 Common Issues & Solutions

### Installation Issues
```bash
# Update pip first
pip install --upgrade pip setuptools wheel

# Clear pip cache if needed
pip cache purge

# Install from source if PyPI fails
pip install git+https://github.com/greenlang/greenlang.git
```

### Runtime Issues
```bash
# Clear GreenLang cache
rm -rf ~/.greenlang/cache

# Reset configuration
gl doctor --reset

# Enable debug mode
export GL_DEBUG=1
gl version
```

### Performance Issues
- **Slow calculations**: Use `greenlang-cli[analytics]` for optimized computations
- **Memory issues**: Process buildings in batches for large portfolios
- **Network timeouts**: Check internet connection for emission factor updates

## 🌱 Join the Movement

GreenLang is more than a framework - it's a community working toward a sustainable future. Every calculation, optimization, and innovation brings us closer to our net-zero goals.

**Start making an impact:**
- Calculate your building's footprint today
- Identify optimization opportunities
- Share your results and learnings
- Contribute to the open-source ecosystem

---

**Ready to build climate-intelligent applications? Let's code green! 🌍**

*Created with GreenLang v0.3.0 - The Climate Intelligence Framework*