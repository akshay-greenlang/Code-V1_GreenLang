# Changelog

All notable changes to the GreenLang Agent Factory CLI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Integration with GreenLang Agent SDK
- Real agent generation engine
- Registry API implementation
- Advanced template system
- Plugin architecture
- Web UI for agent management

## [0.1.0] - 2024-12-09

### Added

#### Core CLI Framework
- Typer-based CLI with Rich formatting
- Command structure: `gl agent`, `gl template`, `gl registry`
- Global options: `--version`, `--quiet`, `--help`
- Entry point: `gl` command

#### Agent Commands
- `gl agent create` - Generate agent from specification
- `gl agent validate` - Validate agent specification
- `gl agent test` - Run agent tests
- `gl agent publish` - Publish agent to registry
- `gl agent list` - List all local agents
- `gl agent info` - Show agent details

#### Template Commands
- `gl template list` - List available templates
- `gl template init` - Initialize agent from template
- `gl template show` - Show template details

#### Registry Commands
- `gl registry search` - Search for agents
- `gl registry pull` - Pull agent from registry
- `gl registry push` - Push agent to registry
- `gl registry login` - Authenticate with registry
- `gl registry logout` - Logout from registry

#### Rich Console Features
- Color-coded output (green=success, red=error, yellow=warning)
- Progress bars for long operations
- Styled tables for agent listings
- Tree views for directory structure
- Syntax highlighting for YAML/code
- Markdown rendering
- Info panels for structured data

#### Configuration Management
- YAML-based configuration (`config/factory.yaml`)
- Default settings for agent generation
- Registry configuration
- Generator settings
- Validation and testing options
- Environment-specific configs

#### Templates
- Basic agent template
- Regulatory compliance agent template
- Template specification format
- Template initialization system

#### Project Structure
- Auto-generated directory structure
- Agent organization (agents/, specs/, tests/)
- Configuration management
- Template storage

#### Documentation
- Comprehensive README with examples
- Installation guide (INSTALL.md)
- Quick start guide (QUICKSTART.md)
- Inline help for all commands
- Template documentation

#### Development Tools
- pytest test suite
- Black code formatting
- Ruff linting
- mypy type checking
- Test fixtures and utilities
- Makefile for common tasks

#### Packaging
- PyPI-ready package structure
- pyproject.toml configuration
- setup.py for compatibility
- requirements.txt
- MANIFEST.in for package data
- .gitignore
- MIT License

#### Options and Flags
- `--verbose` for detailed output
- `--quiet` for minimal output
- `--dry-run` for simulation mode
- `--output` for custom output paths
- `--force` for overwriting
- `--strict` for strict validation
- `--coverage` for test coverage
- `--parallel` for parallel testing

### Features

- **Validation Engine**: Multi-level spec validation with warnings/errors
- **Progress Tracking**: Rich progress bars for generation/testing/publishing
- **Error Handling**: Comprehensive error messages with suggestions
- **Test Framework**: Built-in test runner with coverage reporting
- **Documentation Generation**: Auto-generate README and docs
- **Deployment Configs**: Generate Dockerfile and K8s manifests
- **Version Management**: Semantic versioning support
- **Audit Trail**: Track agent creation and modifications
- **Search and Discovery**: Search local agents by type/status
- **Format Options**: Output as table/JSON/YAML

### Technical Details

- Python 3.11+ required
- Typer 0.12.0+ for CLI framework
- Rich 13.7.0+ for terminal formatting
- Pydantic 2.5.0+ for validation
- PyYAML 6.0.1+ for config parsing
- Cross-platform support (Windows/Linux/macOS)

### Notes

This is the initial release with core functionality. The agent generation,
registry integration, and template system use placeholder implementations
that will be replaced with full functionality in future releases.

## [0.0.1] - 2024-12-08

### Added
- Project initialization
- Basic structure planning
- Requirements gathering

---

## Version Categories

### Major Version (X.0.0)
- Breaking changes
- Major architecture changes
- Incompatible API changes

### Minor Version (0.X.0)
- New features
- Backwards-compatible additions
- Significant enhancements

### Patch Version (0.0.X)
- Bug fixes
- Documentation updates
- Minor improvements

---

## Future Roadmap

### v0.2.0 (Next Release)
- [ ] Integrate with GreenLang Agent SDK
- [ ] Implement real agent generation
- [ ] Connect to actual registry
- [ ] Add more templates
- [ ] Enhanced validation rules

### v0.3.0
- [ ] Plugin system for custom generators
- [ ] Advanced template engine (Jinja2)
- [ ] Multi-agent orchestration
- [ ] Team collaboration features
- [ ] CI/CD integration

### v1.0.0 (Stable Release)
- [ ] Production-ready agent generation
- [ ] Full registry implementation
- [ ] Comprehensive template library
- [ ] Enterprise features
- [ ] Web UI
- [ ] API server mode

---

[Unreleased]: https://github.com/greenlang/agent-factory/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/greenlang/agent-factory/releases/tag/v0.1.0
[0.0.1]: https://github.com/greenlang/agent-factory/releases/tag/v0.0.1
