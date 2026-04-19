# HVAC Measures Pack Card

## Overview
- **Name**: hvac-measures
- **Version**: 1.0.0
- **Domain**: HVAC System Optimization
- **License**: Apache-2.0
- **Author**: GreenLang Team
- **Created**: 2024-01-15
- **Updated**: 2024-01-15

## Description
The HVAC Measures pack provides comprehensive tools for optimizing heating, ventilation, and air conditioning systems in commercial and residential buildings. It focuses on balancing energy efficiency with occupant comfort while meeting indoor air quality standards.

## Key Features
- **Thermal Comfort Analysis**: PMV/PPD calculations per ASHRAE 55
- **Energy Usage Modeling**: Predictive models for consumption patterns
- **Ventilation Optimization**: Balance IAQ with energy efficiency
- **Multi-zone Support**: Handle complex building configurations
- **Weather Integration**: Climate-responsive optimization
- **COVID-19 Measures**: Enhanced ventilation strategies

## Agents

### thermal_comfort
Analyzes and optimizes thermal comfort conditions.
- **Methods**: PMV, PPD, adaptive comfort model
- **Standards**: ASHRAE 55, ISO 7730
- **Outputs**: Setpoint recommendations, comfort zones

### energy_calculator
Calculates energy consumption and costs.
- **Inputs**: Building model, weather data, schedules
- **Calculations**: Heating/cooling loads, peak demand
- **Outputs**: kWh usage, cost estimates, savings potential

### ventilation_optimizer
Optimizes ventilation rates for IAQ and efficiency.
- **Standards**: ASHRAE 62.1, EN 15251
- **Features**: Demand-controlled ventilation, economizer cycles
- **COVID-19**: Enhanced filtration and air changes

## Datasets

### hvac_performance
Historical performance data from various HVAC systems.
- **Size**: 50,000 records
- **Features**: Temperature, humidity, energy use, occupancy
- **Update Frequency**: Monthly

### climate_zones
Climate zone definitions and typical weather patterns.
- **Coverage**: Global climate zones
- **Format**: JSON with zone characteristics
- **Use Case**: Default weather profiles

## Models

### efficiency_predictor
Machine learning model for predicting HVAC efficiency.
- **Algorithm**: Random Forest Regressor
- **Accuracy**: R² = 0.92
- **Features**: 15 input variables
- **Training Data**: 100,000 building-hours

## Usage

### Basic Usage
```bash
# Run with defaults
gl run hvac-measures

# Specify building parameters
gl run hvac-measures \
  --env BUILDING_TYPE=hospital \
  --env FLOOR_AREA=50000 \
  --env ZONE_COUNT=20
```

### Advanced Configuration
```yaml
# custom-hvac.yaml
extends: packs/hvac-measures/gl.yaml
environment:
  BUILDING_TYPE: school
  TARGET_PMV: -0.5  # Slightly cool
  COVID_MEASURES: maximum
  WEATHER_FILE: datasets/local_weather.csv
```

### Integration Example
```python
from packs.hvac_measures import thermal_comfort

# Calculate thermal comfort
result = thermal_comfort.calculate_pmv(
    air_temp=22.0,
    mean_radiant_temp=21.5,
    air_velocity=0.1,
    relative_humidity=50,
    metabolic_rate=1.2,
    clothing_insulation=1.0
)
print(f"PMV: {result['pmv']:.2f}")
print(f"PPD: {result['ppd']:.1f}%")
```

## Performance Metrics
- **Execution Time**: p50: 15s, p95: 45s, p99: 58s
- **Memory Usage**: 500MB typical, 1.8GB peak
- **Accuracy**: ±5% energy prediction accuracy
- **Optimization Potential**: 15-30% energy savings typical

## Validation & Testing

### Test Coverage
- Unit tests: 95% coverage
- Integration tests: 12 scenarios
- Golden tests: Deterministic validation
- Performance tests: Load and stress testing

### Run Tests
```bash
# Validate pack
gl pack validate hvac-measures

# Run unit tests
cd packs/hvac-measures && pytest tests/

# Run golden tests
gl run hvac-measures --test-mode --deterministic
```

## Standards Compliance
- **ASHRAE 55-2020**: Thermal Comfort
- **ASHRAE 62.1-2019**: Ventilation for IAQ
- **ASHRAE 90.1-2019**: Energy Standard
- **ISO 7730**: Thermal Comfort
- **EN 15251**: Indoor Environmental Criteria
- **LEED v4.1**: Green Building Certification

## Limitations
- Requires detailed building geometry for accuracy
- Weather data quality affects predictions
- Limited to mechanically conditioned spaces
- Not suitable for industrial HVAC systems

## Citation
```bibtex
@software{hvac_measures,
  title = {HVAC Measures Pack for GreenLang},
  author = {GreenLang Team},
  year = {2024},
  version = {1.0.0},
  url = {https://github.com/greenlang/packs/hvac-measures}
}
```

## Support
- **Documentation**: https://docs.greenlang.io/packs/hvac-measures
- **Issues**: https://github.com/greenlang/packs/issues
- **Community**: Discord #hvac-optimization

## Changelog

### v1.0.0 (2024-01-15)
- Initial release with three core agents
- Support for commercial buildings
- COVID-19 ventilation enhancements
- Integration with weather data
- PDF report generation