# Changelog

All notable changes to GreenLang will be documented in this file.

## [0.3.0] - 2025-01-24

### Added
- **Pack System**: Complete pack management system with installation, validation, and publishing
- **Orchestrator**: Advanced pipeline orchestration with parallel execution support
- **Connectors**: Extensible connector framework for external integrations
- **Signing & Verification**: Comprehensive artifact signing with OIDC/keyless support
- **Policy Engine**: OPA-based policy enforcement for egress control
- **SBOM Generation**: Automated Software Bill of Materials generation
- **Security Framework**: Enhanced security with wrapped HTTP calls and audit logging
- **Docker Support**: Multi-arch Docker images with cosign signatures
- **Hub Integration**: GreenLang Hub for pack discovery and distribution

### Changed
- Migrated to dynamic versioning from VERSION file
- Enhanced CLI with rich output and progress indicators
- Improved error handling and recovery mechanisms
- Upgraded dependencies for security and performance

### Fixed
- Version consistency across all components
- Docker build reproducibility issues
- Pack installation permission issues
- Policy evaluation performance

### Security
- Implemented secure HTTP wrapper with policy enforcement
- Added signature verification for all artifacts
- Enhanced audit logging for compliance
- Removed hardcoded secrets and credentials

### Deprecated
- Legacy pack format (v1) - will be removed in v0.4.0
- Direct HTTP calls without security wrapper

### Removed
- Obsolete test files and fixtures
- Redundant configuration options

## [0.2.3] - 2025-09-23

### Added Features

- Modify RpmDBEntry to include modularityLabel for cyclonedx [[#4212](https://github.com/anchore/syft/pull/4212) @sfc-gh-rmaj]
- Add locations onto packages read from Java native image SBOMs [[#4186](https://github.com/anchore/syft/pull/4186) @rudsberg]

**[(Full Changelog)](https://github.com/anchore/syft/compare/v1.32.0...v1.33.0)**
