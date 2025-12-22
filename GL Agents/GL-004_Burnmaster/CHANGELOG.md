# Changelog

All notable changes to GL-004 BURNMASTER will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Integration with GL-018 UNIFIEDCOMBUSTION
- Multi-burner coordination optimization
- Predictive maintenance for burner components
- Advanced flame image analysis using CNN

## [1.0.0] - 2025-01-15

### Added
- Initial release of GL-004 BURNMASTER
- Core combustion calculation engine
  - Stoichiometric ratio calculation for multiple fuel types
  - Excess air and lambda value computation
  - Combustion efficiency calculations
  - Adiabatic flame temperature estimation
  - Flue gas loss calculations
- Air-fuel ratio optimization
  - Real-time O2 trim recommendations
  - NOx-efficiency trade-off optimization
  - Load-dependent setpoint curves
- Emissions prediction and control
  - NOx prediction using thermal/prompt/fuel NOx models
  - CO prediction for incomplete combustion detection
  - Emissions reduction strategy recommendations
- Flame stability monitoring
  - Flame scanner signal analysis
  - Stability index calculation
  - Flameout prediction and alerting
- Turndown optimization
  - Efficient operation across burner load range
  - Minimum stable load recommendations
  - Multi-burner sequencing support
- Explainability features
  - SHAP feature attribution for ML predictions
  - LIME local explanations
  - Physics-based explanation generation
- Safety envelope enforcement
  - Combustion safety limits
  - Flameout protection logic
  - Emissions limit monitoring
  - DCS interlock integration
- Control interfaces
  - Air-fuel ratio controller
  - O2 trim controller
  - Flame stability controller
  - Damper position controller
- API services
  - REST API with FastAPI
  - GraphQL API with Strawberry
  - gRPC services for high-performance use cases
- Integration connectors
  - OPC-UA for OT data acquisition
  - Kafka for streaming data
  - DCS/PLC connectivity
  - CEMS integration
- Monitoring and observability
  - Prometheus metrics
  - OpenTelemetry tracing
  - Health endpoints
  - Alert management
- Audit and compliance
  - Full calculation provenance tracking
  - SHA-256 hashing for audit trails
  - Evidence package generation
- Uncertainty quantification
  - Sensor uncertainty propagation
  - Confidence bounds on predictions
  - Quality gate validation
- Comprehensive test suite
  - Unit tests for all calculators
  - Integration tests for API services
  - Validation tests against reference equations
  - Performance benchmarks
- Documentation
  - API reference documentation
  - Architecture documentation
  - User guide
  - Developer guide

### Technical Details
- Python 3.10+ support
- Pydantic v2 for data validation
- Type hints throughout codebase
- 85%+ test coverage target
- Sub-millisecond calculation performance

## [0.9.0] - 2024-12-01

### Added
- Beta release for internal testing
- Core calculation engine implementation
- Basic API endpoints
- Initial test coverage

### Known Issues
- Performance optimization needed for batch calculations
- CEMS integration incomplete

## [0.8.0] - 2024-11-01

### Added
- Alpha release
- Proof of concept implementation
- Basic stoichiometric calculations

---

## Version History Summary

| Version | Date | Status |
|---------|------|--------|
| 1.0.0 | 2025-01-15 | Current Release |
| 0.9.0 | 2024-12-01 | Beta |
| 0.8.0 | 2024-11-01 | Alpha |

## Migration Notes

### Upgrading to 1.0.0

No breaking changes from 0.9.0. Recommended upgrades:

1. Update all dependencies to latest versions
2. Run database migrations (if applicable)
3. Update configuration files for new features
4. Review new safety envelope settings

## Deprecation Notices

- GL-004 BURNMASTER will be consolidated into GL-018 UNIFIEDCOMBUSTION in Q2 2026
- All existing APIs will continue to work via compatibility layer
- Migration guide will be provided in GL-018 documentation
