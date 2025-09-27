# GreenLang Quickstart Examples

This directory contains ready-to-run examples that demonstrate core GreenLang capabilities. All examples are designed to work with both PyPI and Docker installations.

## 📁 What's Included

### 1. Hello World Example (`hello.yaml`)
- **Purpose**: Simple first calculation to verify your installation
- **Demonstrates**: Basic CLI usage and carbon footprint calculation
- **Runtime**: < 30 seconds
- **Use Case**: Office building energy consumption analysis

### 2. Data Processing Example (`process-data.yaml`)
- **Purpose**: Real-world data processing pipeline
- **Demonstrates**: YAML workflows, data validation, and reporting
- **Runtime**: 1-2 minutes
- **Use Case**: Multi-building portfolio analysis

### 3. Sample Data Files
- **`sample-building.json`**: Single building energy data
- **`sample-portfolio.json`**: Multiple buildings for batch processing
- **`sample-industrial.json`**: Industrial process emissions data

### 4. Execution Scripts
- **`run.sh`**: Linux/macOS execution script
- **`run.bat`**: Windows execution script
- **`docker-run.sh`**: Docker-based execution

## 🚀 Quick Start

### Prerequisites
```bash
# Verify GreenLang is installed
gl version

# Or using Docker
docker run --rm ghcr.io/greenlang/greenlang:0.3.0 version
```

### Running Examples

#### Option 1: Using PyPI Installation
```bash
# Navigate to quickstart directory
cd examples/quickstart

# Run hello world example
python hello-world.py

# Run data processing pipeline
gl pipeline run process-data.yaml --input sample-portfolio.json

# Execute all examples
./run.sh  # Linux/macOS
# or
run.bat   # Windows
```

#### Option 2: Using Docker
```bash
# Navigate to quickstart directory
cd examples/quickstart

# Run with Docker
./docker-run.sh

# Or manually
docker run --rm -v $(pwd):/data ghcr.io/greenlang/greenlang:0.3.0 \
  calc --input /data/sample-building.json --output /data/results.json
```

## 📊 Expected Outputs

### Hello World Example
```
🏢 Building: Demo Office Building
📊 Total Annual Emissions: 28.45 metric tons CO2e
📏 Emission Intensity: 0.259 kgCO2e/sqft
⚡ Electricity: 24,500.0 kg CO2e
🔥 Natural Gas: 3,950.0 kg CO2e
🎯 Performance Rating: B
```

### Data Processing Pipeline
- **Portfolio Summary**: JSON report with aggregated emissions
- **Individual Reports**: Per-building analysis results
- **Optimization Recommendations**: Energy efficiency suggestions
- **Benchmark Comparisons**: Performance vs. industry standards

## 🔧 Customization Guide

### Modifying Building Data
Edit `sample-building.json` to match your building:
```json
{
  "building": {
    "name": "Your Building Name",
    "type": "commercial_office",  // or retail, warehouse, hospital, etc.
    "area_m2": 5000,             // Your building area
    "location": "Your City, State",
    "occupancy": 200             // Number of occupants
  },
  "energy_consumption": [
    {
      "fuel_type": "electricity",
      "consumption": 75000,       // Your annual kWh
      "unit": "kWh",
      "period": "annual"
    }
  ]
}
```

### Adding New Fuel Types
Supported fuel types:
- `electricity`
- `natural_gas`
- `fuel_oil`
- `propane`
- `diesel`
- `gasoline`
- `coal`
- `biomass`

### Customizing Calculations
Modify the Python examples to:
- Change emission factor regions
- Include Scope 3 emissions
- Add custom benchmarking
- Implement real-time monitoring

## 🌟 Building Your First Custom Example

### 1. Copy an Existing Example
```bash
cp hello-world.py my-custom-calculation.py
```

### 2. Modify for Your Use Case
```python
from greenlang.sdk import GreenLangClient

client = GreenLangClient()

# Your custom building data
my_building = {
    "name": "My Custom Building",
    "building_type": "retail",  # Changed from office
    "area_m2": 1200,           # Smaller building
    "location": "Austin, TX",   # Different location
    "occupancy": 50
}

# Your actual consumption data
my_energy_data = [
    {
        "fuel_type": "electricity",
        "consumption": 25000,   # Your actual kWh
        "unit": "kWh",
        "period": "annual"
    }
]

# Run calculation
result = client.calculate_building_emissions(
    building=my_building,
    energy_consumption=my_energy_data
)
```

### 3. Test Your Example
```bash
python my-custom-calculation.py
```

## 📈 Advanced Examples

### Real-time Monitoring
```python
# See advanced-monitoring.py for complete example
from greenlang.monitoring import RealTimeMonitor

monitor = RealTimeMonitor()
monitor.start_monitoring("building-001", interval_seconds=300)
```

### Batch Processing
```python
# See batch-processing.py for complete example
from greenlang.utils import BatchProcessor

processor = BatchProcessor(batch_size=50)
results = processor.process_buildings(building_list)
```

### Custom Agents
```python
# See custom-agent.py for complete example
from greenlang.agents import BaseAgent

class MyCustomAgent(BaseAgent):
    def calculate_emissions(self, data):
        # Your custom logic here
        return emissions_result
```

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Verify installation
pip show greenlang-cli

# Reinstall if needed
pip install --upgrade greenlang-cli==0.3.0
```

#### Calculation Errors
```bash
# Enable debug mode
export GL_DEBUG=1
python hello-world.py

# Validate your data
gl validate-data --input sample-building.json
```

#### File Not Found
```bash
# Ensure you're in the correct directory
pwd
ls -la *.json *.py
```

#### Docker Issues
```bash
# Check Docker is running
docker version

# Pull latest image
docker pull ghcr.io/greenlang/greenlang:0.3.0
```

### Getting Help

If you encounter issues:
1. Check the [troubleshooting guide](../../docs/quickstart.md#troubleshooting)
2. Join our [Discord community](https://discord.gg/greenlang)
3. Open an [issue on GitHub](https://github.com/greenlang/greenlang/issues)

## 🎯 Next Steps

After running these examples:

1. **Explore More Examples**: Check out `../tutorials/` for advanced scenarios
2. **Read the Documentation**: Visit [greenlang.io/docs](https://greenlang.io/docs)
3. **Join the Community**: Connect with other users on Discord
4. **Contribute**: Share your own examples with the community

## 📄 License

These examples are part of the GreenLang project and are released under the MIT License.

---

**Ready to start building? Pick an example and run it now! 🚀**

*Created with GreenLang v0.3.0 - The Climate Intelligence Platform*