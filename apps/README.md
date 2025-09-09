# GreenLang Applications

This directory contains ready-to-use applications built on the GreenLang framework.

## Structure

```
apps/
├── README.md
├── climatenza_app/      # Solar thermal feasibility application
├── Building_101/        # Building emissions calculator
├── lca_toolkit/         # Life Cycle Assessment tools (planned)
├── carbon_dashboard/    # Web-based carbon dashboard (planned)
└── industrial_sim/      # Industrial simulation apps (planned)
```

## Available Applications

### 1. Climatenza App
Solar thermal feasibility analysis for industrial applications.
- TMY solar data integration
- Hourly load matching
- Economic analysis
- [Documentation](climatenza_app/README.md)

### 2. Building 101
Commercial building emissions calculator.
- Multi-fuel tracking
- HVAC system modeling
- Benchmark comparisons
- [Documentation](Building_101/README.md)

## Running Applications

Each app can be run independently:

```bash
# Climatenza solar analysis
cd apps/climatenza_app
python -m climatenza_app.main --site config.yaml

# Building emissions
cd apps/Building_101
gl run building_emissions.yaml
```

## Development

To create a new application:

1. Create a directory under `apps/`
2. Include:
   - `README.md` with documentation
   - `requirements.txt` for dependencies
   - Entry point script
   - Configuration examples
3. Register in the main app registry

## App Requirements

Applications should:
- Use GreenLang SDK/CLI for calculations
- Provide clear documentation
- Include example configurations
- Have comprehensive tests
- Support YAML/JSON configuration

## Roadmap

Planned applications:
- **LCA Toolkit**: Full lifecycle assessment
- **Carbon Dashboard**: Real-time monitoring
- **Industrial Sim**: Process optimization
- **Fleet Manager**: Transportation emissions
- **Supply Chain**: Scope 3 tracking

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing new applications.