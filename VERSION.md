# GreenLang Version Information

## Current Version: 0.3.0

### Version Summary
- **Release Date**: September 23, 2025
- **Status**: Production Release
- **Python Compatibility**: 3.10+
- **License**: MIT

### Version Verification
To verify your installation:
```bash
gl --version
python -c "import greenlang; print(greenlang.__version__)"
```

## Version 0.3.0 - Q4'25 Climate Intelligence Baseline (2025-09-24)

### Major Changes
- 🏗️ **Single Source of Truth for Versioning**: Implemented VERSION file as SSOT
- 📦 **PyPI & Docker Ready**: First public release with signed artifacts
- 🔒 **Default-Deny Policies**: Security-first approach with policy enforcement
- 🎯 **Pack Architecture**: Transition from agents to packs for domain logic
- 🚀 **Production Ready**: Exit bar criteria met for v0.3.0 release

### Infrastructure Improvements
- Dynamic version loading from VERSION file
- Consistent version across CLI, packages, and Docker images
- Version parity enforcement in CI/CD
- Automated version consistency checks

### Security Enhancements
- Default-deny policy enforcement
- Signed artifact support
- SBOM generation capability
- Sandbox capability gating

### Recent Additions
- Enhanced FuelAgent with caching and recommendations
- New BoilerAgent for thermal systems
- Improved fixture organization
- Performance optimizations
- Security improvements (removed API keys, updated documentation)
- AI Assistant feature documentation (optional OpenAI integration)

### Version History
- **v0.3.0** (2025-09-24) - Q4'25 Climate Intelligence Baseline
- **v0.2.3** (2025-09) - Security hardening release
- **v0.2.0** (2025-09) - Infrastructure Seed Release
- **v0.1.0** (2025-09) - Internal infrastructure refactor
- **v0.0.1** (2025-01) - Initial release with enhanced agents

### Upgrade Instructions
```bash
pip install --upgrade greenlang
```

### Breaking Changes
None in v0.0.1 (initial release)

### Future Roadmap
- [ ] HVAC Agent
- [ ] Transportation Agent
- [ ] Water/Waste Agents
- [ ] API Server
- [ ] Cloud Integration