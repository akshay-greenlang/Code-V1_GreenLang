# Changelog

All notable changes to GL-Agent-Factory will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CONTRIBUTING.md with full developer onboarding guide
- CHANGELOG.md for project-wide change tracking
- CODE_OF_CONDUCT.md based on Contributor Covenant
- SECURITY.md with vulnerability disclosure policy
- GOVERNANCE.md defining project governance structure
- Registry audit system for detecting duplicate agent IDs
- OpenAPI specification with Swagger UI integration
- Centralized EmissionFactorRepository service
- IoC dependency injection container
- Comprehensive integration test suite
- Feature flag service for controlled rollouts
- QUDT ontology integration for unit conversions
- Scope 3 GHG Protocol categories 8-15 implementation
- ISO 50001 Energy Management System module
- MQTT connector for Industrial IoT integration
- Monte Carlo uncertainty engine for calculations
- Documentation templates and guides

### Changed
- Updated agent registry with unique ID enforcement
- Enhanced emission factor service with repository pattern

### Fixed
- Resolved duplicate agent ID conflicts in registry
- Fixed registry health check for new agent modules

## [0.9.0] - 2024-12-12

### Added
- **Core Platform**
  - FastAPI-based backend with async support
  - Multi-tenant architecture with tenant isolation
  - Redis-backed rate limiting and caching
  - PostgreSQL database with Alembic migrations
  - OpenTelemetry distributed tracing

- **Agent Framework**
  - 143 specialized calculation agents (GL-001 to GL-100+)
  - AgentRegistry with lazy instantiation
  - BaseCalculator abstract class for deterministic calculations
  - Agent lifecycle management (create, execute, validate)
  - Provenance tracking for audit trails

- **Climate & Compliance Agents (GL-001 to GL-013)**
  - GL-001: Carbon Emissions Calculator
  - GL-002: CBAM (Carbon Border Adjustment Mechanism)
  - GL-003: CSRD (Corporate Sustainability Reporting Directive)
  - GL-004: EUDR (EU Deforestation Regulation)
  - GL-005: Building Energy Performance
  - GL-006: Scope 3 Emissions
  - GL-007: EU Taxonomy Alignment
  - GL-008: Green Claims Verification
  - GL-009: Product Carbon Footprint
  - GL-010: Science-Based Targets (SBTi)
  - GL-011: Climate Risk Assessment
  - GL-012: Carbon Offset Verification
  - GL-013: California SB 253 Compliance

- **Process Heat Baseline Agents (GL-020 to GL-035)**
  - Economizer performance analysis
  - Burner efficiency optimization
  - Heat recovery system analysis
  - Combustion optimization
  - Furnace heat balance
  - Steam trap management
  - Insulation assessment
  - Boiler efficiency analysis
  - Air-fuel ratio optimization
  - Heat exchanger performance

- **Process Heat Optimization Agents (GL-036 to GL-065)**
  - Electrification feasibility analysis
  - Biomass conversion assessment
  - Solar thermal integration
  - Draft control optimization
  - Refractory performance
  - Heat loss reduction
  - Process control optimization
  - VFD (Variable Frequency Drive) analysis
  - Thermal oxidizer efficiency
  - Heat treatment optimization
  - Drying system optimization
  - Microwave heating assessment
  - Resistance heating analysis

- **Advanced Analytics Agents (GL-066 to GL-080)**
  - Energy audit automation
  - Commissioning verification
  - Demand forecasting
  - Emergency response planning
  - Contractor management
  - Vendor selection optimization
  - Life Cycle Assessment (LCA)
  - Circular economy analysis
  - Water-energy nexus
  - Grid services optimization

- **Business & Financial Agents (GL-081 to GL-100)**
  - Procurement advisory
  - Net zero planning
  - Sustainability reporting
  - Benchmark comparison
  - Innovation scouting
  - Risk assessment
  - Business case builder
  - Incentive optimization
  - Financing optimization
  - Asset valuation
  - M&A analysis
  - Insurance advisory
  - Tax optimization
  - Stakeholder reporting
  - Strategic planning
  - Cybersecurity assessment
  - Data quality management
  - Regulatory tracking
  - Knowledge management
  - Kaizen driver

- **Emission Factor Database**
  - EPA emission factors (US)
  - DEFRA emission factors (UK)
  - IEA grid factors (International)
  - IPCC GWP values (AR5, AR6)
  - Regional fallback hierarchy
  - Version pinning for reproducibility

- **API Infrastructure**
  - RESTful API with OpenAPI 3.1 documentation
  - JWT and API Key authentication
  - Rate limiting with tier-based quotas
  - WebSocket support for real-time updates
  - GraphQL endpoint (beta)
  - Batch processing API

- **Testing Framework**
  - Unit test suite with pytest
  - Integration test framework
  - End-to-end test suite
  - Golden test framework for determinism
  - Benchmark testing
  - 85% minimum coverage requirement

- **CLI Tools**
  - Agent generator scaffold
  - Test runner integration
  - Registry management commands
  - Database migration helpers

- **DevOps Infrastructure**
  - Docker containerization
  - Kubernetes manifests
  - GitHub Actions CI/CD
  - Prometheus metrics export
  - Grafana dashboard templates

### Changed
- Migrated from Flask to FastAPI for better async support
- Updated Pydantic to v2 for improved validation
- Enhanced logging with structlog for structured output

### Fixed
- Memory leak in long-running agent executions
- Race condition in concurrent registry access
- Timezone handling in emission factor lookups

### Security
- Implemented input validation on all API endpoints
- Added SQL injection protection
- Enabled CORS with strict origin checking
- Added rate limiting to prevent abuse

## [0.8.0] - 2024-11-15

### Added
- Initial agent framework architecture
- Base calculator engine design
- Emission factor data loading
- Registry prototype

### Changed
- Refined agent interface contracts

## [0.7.0] - 2024-10-20

### Added
- Project scaffolding
- Documentation structure
- Development environment setup

---

## Release Notes Guidelines

### Version Numbering
- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Categories
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Features to be removed in future
- **Removed**: Features removed in this release
- **Fixed**: Bug fixes
- **Security**: Security-related changes

### Contributors
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

[Unreleased]: https://github.com/greenlang/GL-Agent-Factory/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/greenlang/GL-Agent-Factory/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/greenlang/GL-Agent-Factory/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/greenlang/GL-Agent-Factory/releases/tag/v0.7.0
