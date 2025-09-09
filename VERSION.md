# GreenLang Version Information

## Current Version: 0.0.1

### Version Summary
- **Release Date**: January 2025
- **Status**: Production Ready (Safe for Public Release)
- **Python Compatibility**: 3.8+
- **License**: MIT

### Version Verification
To verify your installation:
```bash
gl --version
python -c "import greenlang; print(greenlang.__version__)"
```

### Major Features in v0.0.1
- ✅ Complete agent framework (11 agents)
- ✅ CLI with multiple commands
- ✅ Workflow orchestration
- ✅ Global emission factors database
- ✅ Comprehensive testing suite
- ✅ Type hints throughout
- ✅ Documentation and examples

### Recent Additions
- Enhanced FuelAgent with caching and recommendations
- New BoilerAgent for thermal systems
- Improved fixture organization
- Performance optimizations
- Security improvements (removed API keys, updated documentation)
- AI Assistant feature documentation (optional OpenAI integration)

### Version History
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