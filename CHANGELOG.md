# Changelog

All notable changes to the GreenLang project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.1] - 2025-01-23

### Added

#### FuelAgent
- Performance caching with @lru_cache decorators for emission factor lookups
- Batch processing capability for multiple fuel sources
- Comprehensive FUEL_CONSTANTS dictionary with standard/renewable fuel lists
- Fuel switching recommendations with cost/emission comparisons
- 7 integration examples demonstrating real-world usage patterns
- 6 dedicated test fixtures for various fuel scenarios
- Enhanced docstrings with Args/Returns format

#### BoilerAgent
- Async support with async_batch_process() for concurrent operations
- Performance tracking using psutil for memory/CPU monitoring
- Unit conversion library (UnitConverter) for centralized conversions
- External configuration file (boiler_efficiencies.json)
- JSON Schema validation (boiler_input.schema.json)
- Export capabilities for JSON, CSV, and Excel formats
- Historical data tracking for performance analysis
- 10 comprehensive integration examples
- 6 specialized test fixtures for boiler scenarios
- Support for 13+ boiler types including heat pumps and fuel cells
- Regional efficiency standards (US, EU, UK, CN)
- Altitude and maintenance adjustments
- Oversizing factor calculations

#### Utility Libraries
- `unit_converter.py`: Centralized energy, mass, volume conversions
- `performance_tracker.py`: Performance monitoring and benchmarking with context managers

#### Configuration Management
- External JSON configuration for boiler efficiencies
- JSON Schema validation for input data
- Support for fuel type aliasing (oil → fuel_oil)

### Changed
- Maintained version at 0.0.1
- Enhanced FuelAgent with 17 essential components
- Enhanced BoilerAgent with 17 essential components plus 8 additional features
- Updated GREENLANG_DOCUMENTATION.md with detailed agent features
- Updated README.md with new features section

### Security
- Removed exposed API keys from repository
- Updated .env.example with clear documentation about optional API features
- Added security warnings in README.md about not committing secrets
- Ensured .gitignore properly excludes sensitive files (.env, api_keys.json, secrets.yaml)
- Documented that all core functionality works without API keys
- Added new dependencies: psutil, jsonschema, aiofiles

### Fixed
- FuelAgent missing components: fixtures, performance helpers, constants, recommendations
- BoilerAgent missing components: performance optimizations, integration examples
- Emission factor lookups now use correct method signatures
- Fuel type mapping for compatibility (oil → fuel_oil, lpg → propane)

### Removed
- 27 duplicate test files and redundant documentation
- Consolidated overlapping fixture files
- Removed redundant workflow examples

## [Pre-release] - 2024-12-15

### Initial Development
- Initial development of GreenLang
- Core agent framework with BaseAgent
- 5 core agents: InputValidator, Fuel, Carbon, Report, Benchmark
- 4 specialized agents: GridFactor, BuildingProfile, Intensity, Recommendation
- CLI interface with calc, analyze, and benchmark commands
- YAML workflow engine
- Global emission factors for 12 regions
- Support for 15+ building types
- 200+ unit tests with 85% coverage
- Type-safe APIs with mypy enforcement
- Rich terminal output with progress tracking
- Comprehensive documentation

### Infrastructure
- Repository structure with /agents, /cli, /examples, /tests
- pyproject.toml configuration
- GitHub Actions CI/CD pipeline
- Docker support
- Windows batch file support

## [Unreleased]

### Planned
- GraphQL API interface
- Real-time monitoring dashboard
- Integration with IoT sensors
- Machine learning predictions
- Carbon offset marketplace integration
- Blockchain-based carbon credits
- Mobile application
- REST API v2
- Kubernetes deployment manifests
- Terraform infrastructure as code

---

## Version Guidelines

- **Major (X.0.0)**: Breaking API changes, major architectural shifts
- **Minor (0.X.0)**: New features, enhancements, non-breaking changes
- **Patch (0.0.X)**: Bug fixes, documentation updates, minor improvements

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.