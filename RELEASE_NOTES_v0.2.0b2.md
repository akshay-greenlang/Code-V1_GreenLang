# v0.2.0b2 – Infra Seed (Beta 2)

🌍 **GreenLang v0.2.0b2** brings the foundational infrastructure for climate-intelligent development with enhanced stability and core framework improvements.

## 📦 Installation

### From TestPyPI (Beta Channel)

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==0.2.0b2

# Verify installation
gl --version
```

### From GitHub Releases

```bash
# Download wheel directly
curl -L https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-0.2.0b2-py3-none-any.whl -o greenlang-0.2.0b2-py3-none-any.whl
pip install greenlang-0.2.0b2-py3-none-any.whl
```

## 🚀 10-Minute Quickstart

### 1. Initialize Your First Climate-Aware Project

```bash
# Create a new project
mkdir my-green-project
cd my-green-project

# Initialize GreenLang workspace
gl init --project-type=basic
```

### 2. Create Your First Policy

```bash
# Generate a sample energy-efficiency policy
gl policy create --template=energy-efficiency --name=my-first-policy

# View the generated policy
gl policy list
```

### 3. Run Basic Climate Intelligence

```bash
# Execute a simple carbon assessment
gl run --mode=assessment --target=./

# View carbon impact metrics
gl metrics show --type=carbon
```

### 4. Explore Core Components

```bash
# List available agents
gl agents list

# Check system health
gl health check

# View framework capabilities
gl info --verbose
```

### 5. Run Example Workflows

```bash
# Execute a sample green computing workflow
gl workflow run --template=renewable-scheduling

# Monitor real-time carbon impact
gl monitor start --metrics=carbon,energy
```

## ⚠️ Known Issues

### Dependencies
- **pandas/numpy currently required**: These will become optional dependencies in v0.2.1. Currently needed for the solar resource agent's data processing capabilities.
- **Large dependency footprint**: Working on modularizing dependencies for lighter installs.

### Warnings
- **Pydantic deprecation warnings**: Some third-party dependencies trigger Pydantic v1 compatibility warnings. These are harmless and will be resolved as dependencies update.
- **asyncio event loop warnings**: May appear on Windows systems when using certain LLM integrations.

### Platform Compatibility
- **Windows path handling**: Some file path operations may require absolute paths on Windows systems.
- **macOS M1/M2**: All features supported, some NumPy operations may show performance warnings.

### Features in Development
- **Hub connectivity**: Pack discovery and sharing features are experimental.
- **Advanced LLM integrations**: OpenAI/Anthropic integrations are functional but may have rate limiting issues.
- **Real-time carbon monitoring**: Requires external carbon intensity APIs that may have availability limitations.

## 🔄 Call for Feedback

We're actively developing GreenLang and need your input to make it better! Please help us by:

### 🐛 Reporting Issues
- **Bug reports**: [GitHub Issues](https://github.com/YOUR_ORG/greenlang/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/YOUR_ORG/greenlang/discussions)

### 💡 What We're Looking For
1. **Installation experience**: Did the setup process work smoothly on your system?
2. **CLI usability**: Are the commands intuitive? What's confusing?
3. **Performance feedback**: How does it perform with your typical workloads?
4. **Use case validation**: Does it solve real problems in your climate-aware development?
5. **Documentation gaps**: What's missing or unclear in our docs?

### 🗣️ Community Channels
- **Discord**: [Join our community](https://discord.gg/greenlang)
- **Twitter**: [@GreenLangDev](https://twitter.com/greenlangdev)
- **Email**: feedback@greenlang.org

## 📄 Release Artifacts

### Core Distribution
- **Wheel**: [`greenlang-0.2.0b2-py3-none-any.whl`](https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-0.2.0b2-py3-none-any.whl)
- **Source**: [`greenlang-0.2.0b2.tar.gz`](https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-0.2.0b2.tar.gz)

### Software Bill of Materials (SBOM)
- **Full SBOM**: [`greenlang-full-0.2.0.spdx.json`](https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-full-0.2.0.spdx.json)
- **Distribution SBOM**: [`greenlang-dist-0.2.0.spdx.json`](https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-dist-0.2.0.spdx.json)
- **Runner SBOM**: [`greenlang-runner-0.2.0.spdx.json`](https://github.com/YOUR_ORG/greenlang/releases/download/v0.2.0b2/greenlang-runner-0.2.0.spdx.json)

### Verification
All artifacts are signed and include integrity checksums. Verify using:

```bash
# Verify wheel integrity (if checksums provided)
sha256sum greenlang-0.2.0b2-py3-none-any.whl

# Verify SBOM signatures (if signing implemented)
cosign verify --key cosign.pub greenlang-full-0.2.0.spdx.json
```

---

## 🌱 What's Next?

### v0.2.1 (Target: Q4 2025)
- Modular dependencies (pandas/numpy become optional)
- Enhanced Windows compatibility
- Improved LLM integration stability
- Extended carbon monitoring capabilities

### v0.3.0 (Target: Q1 2026)
- Full hub connectivity and pack ecosystem
- Advanced policy orchestration
- Real-time carbon optimization
- Enterprise security features

---

**Ready to build the future of sustainable software?**

Install GreenLang v0.2.0b2 today and join the climate intelligence revolution!

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ greenlang==0.2.0b2
gl init --help
```