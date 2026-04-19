# Boiler → Solar Preheat Pack (v0.2.0)

## Purpose
Optimize industrial boiler operations by preheating feedwater using solar thermal collectors. This pack analyzes the technical and economic feasibility of integrating solar thermal systems with existing boilers to reduce natural gas consumption and CO₂e emissions.

## Key Features
- **Baseline Analysis**: Comprehensive boiler performance modeling using IAPWS-IF97 steam properties
- **Solar Resource Assessment**: Location-specific DNI data from ERA5 or NREL sources
- **Thermal Integration**: Hour-by-hour simulation of solar preheat contribution
- **Carbon Impact**: Emission reduction calculations with regional factors
- **Economic Analysis**: Full financial modeling with NPV, IRR, and payback

## Inputs

### Required Parameters
- **Site Information**
  - Location (latitude, longitude)
  - Altitude (meters)
  - Timezone
  
- **Boiler Configuration**
  - Type (fire-tube, water-tube)
  - Fuel type (natural gas, diesel, coal)
  - Operating pressure (bar)
  - Temperature (°C)
  - Efficiency (%)
  - Capacity (tons/hour steam)
  - Annual operating hours
  - Load factor

- **Solar System Design**
  - Collector type (flat plate, evacuated tube, parabolic trough)
  - Aperture area (m²)
  - Optical efficiency
  - Thermal efficiency
  - Tilt angle and azimuth
  - Storage capacity (hours)

### Datasets
- **Emission Factors**: Regional grid and fuel emission factors (vintage ≥ 2024)
- **Weather Data**: Hourly DNI, ambient temperature, wind speed

## Outputs

### Artifacts
- **report.pdf**: Executive summary for CFO/CTO
- **report.html**: Interactive web dashboard
- **hourly_profile.csv**: 8760-hour simulation results

### Key Metrics
- **Solar Fraction**: Percentage of thermal demand met by solar (%)
- **CO₂e Avoided**: Annual emission reductions (tons)
- **Fuel Saved**: Natural gas or other fuel savings (MMBtu/year)
- **Payback Period**: Simple payback (years)
- **NPV**: Net present value over project lifetime (USD)
- **IRR**: Internal rate of return (%)

## Assumptions

### Technical
- Steady-state hourly energy balance (no transient effects)
- IAPWS-IF97 steam properties for enthalpy calculations
- Fixed collector efficiency across temperature range
- No degradation over time (can be adjusted)
- Perfect thermal integration (no mismatch losses)

### Economic
- Constant fuel prices (escalation can be added)
- Fixed O&M costs at 2% of CAPEX annually
- Carbon price applicable in jurisdiction
- 20-year project lifetime
- No subsidies included (can be added)

### Data Quality
- ERA5 reanalysis data at 0.25° resolution
- Emission factors from government sources
- Annual average boiler efficiency

## Validation

### Test Coverage
- Unit tests for each agent
- Integration tests for full pipeline
- Golden run comparison for determinism
- Performance benchmarks

### Quality Assurance
- Input validation with range checks
- Output sanity checks
- Mass and energy balance verification
- Economic model cross-validation

## Limitations

### Technical
- No transient startup/shutdown losses
- Simplified storage model (well-mixed tank)
- DNI resolution limited to data source grid
- No soiling or shading analysis
- Fixed heat exchanger effectiveness

### Economic
- Does not include financing costs
- Incentives must be added manually
- No detailed cash flow modeling
- Currency conversion not included

### Geographic
- Best suited for locations with DNI > 1800 kWh/m²/year
- Requires reliable grid emission factors
- Limited to regions with weather data coverage

## License
Commercial - Contact team@greenlang.io for licensing

## Support
- Documentation: https://greenlang.io/docs/boiler-solar
- Issues: https://github.com/greenlang/boiler-solar/issues
- Email: support@greenlang.io

## Compliance
- ISO 50001 Energy Management compatible
- GHG Protocol Scope 1 emissions methodology
- ASHRAE 90.1 thermal efficiency standards
- IEA Solar Heating and Cooling Programme guidelines

## Version History
- v0.2.0 (2025-01): Added economics agent, improved solar resource data
- v0.1.0 (2024-12): Initial release with basic thermal modeling

## Citation
If using this pack for research or reports, please cite:
```
GreenLang Boiler-Solar Pack v0.2.0. GreenLang Team, 2025.
https://greenlang.io/packs/boiler-solar
```