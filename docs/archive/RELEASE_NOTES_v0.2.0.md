# 🚀 GreenLang v0.2.0 - Infra Seed (Production Release)

## Overview
We're thrilled to announce the production release of **GreenLang v0.2.0**, the world's first Climate Intelligence orchestration framework. After successful beta testing with strong positive feedback, GreenLang is now available on PyPI for general use.

## 🎯 What is GreenLang?
GreenLang is the **LangChain of Climate Intelligence** - a revolutionary platform that brings LangChain-style modularity and composability to sustainable computing and climate-aware software development.

## 📦 Installation

### From PyPI (Recommended)
```bash
# Core installation
pip install greenlang

# With analytics features (pandas, numpy)
pip install greenlang[analytics]

# Full feature set
pip install greenlang[full]
```

### From Source
```bash
git clone https://github.com/greenlang/greenlang.git
cd greenlang
pip install -e .
```

## ✨ Key Features

### 🔧 Core Capabilities
- **CLI Tool**: Full `gl` command-line interface for climate-intelligent operations
- **Pack Management**: Create, validate, and manage reusable climate intelligence packs
- **Policy Engine**: Default-deny security policies with OPA integration
- **Cross-Platform**: Python 3.10+ support on Windows, macOS, and Linux
- **SDK**: Comprehensive Python SDK for building climate-aware applications

### 🌍 Climate Intelligence
- **Carbon Tracking**: Real-time emissions monitoring across your software lifecycle
- **Energy Optimization**: Intelligent routing to renewable energy regions
- **Green Scheduling**: Execute workloads during low-carbon windows
- **Impact Reporting**: Detailed sustainability metrics and visualizations

### 🔒 Enterprise Security
- **Default-Deny Policies**: Security-first approach to resource access
- **SBOM Generation**: Automated Software Bill of Materials with Syft integration
- **Signed Attestations**: Cryptographic proofs of sustainable practices
- **Supply Chain Security**: Comprehensive dependency verification

## 🚀 Quick Start

### 1. Verify Installation
```bash
gl --version
# Output: GreenLang v0.2.0

gl --help
# Shows available commands
```

### 2. Create Your First Pack
```bash
gl init pack-basic my-climate-app
cd my-climate-app
gl pack validate .
```

### 3. Run Climate Analysis
```bash
# Create a simple pipeline
cat > pipeline.yaml << EOF
name: climate-analysis
version: 1.0.0
steps:
  - name: analyze
    pack: demo/carbon-calculator
    inputs:
      energy_kwh: 100
      region: us-west-2
EOF

# Execute the pipeline
gl run pipeline.yaml
```

## 📊 What's New in v0.2.0

### Major Improvements
- **Modular Dependencies**: Core package is now lighter with optional extras
- **Enhanced CLI**: Improved command structure and help documentation
- **Better Testing**: Comprehensive test suite with 187+ tests
- **Code Quality**: Entire codebase formatted with Black and cleaned with autoflake
- **Cross-Platform**: Fixed path handling for Windows compatibility
- **CI/CD**: GitHub Actions workflow for automated testing

### Breaking Changes
- None - this is the first production release

### Known Issues
- Some CLI commands may have PowerShell-specific argument parsing issues
- Pydantic may show deprecation warnings (will be fixed in v0.2.1)

## 📚 Documentation

- **GitHub Repository**: https://github.com/greenlang/greenlang
- **PyPI Package**: https://pypi.org/project/greenlang/
- **Documentation**: https://docs.greenlang.io (coming soon)
- **Examples**: See `/examples` directory in the repository

## 🤝 Contributing

We welcome contributions! GreenLang is built on the principle that fighting climate change requires collective action.

- **Report Issues**: https://github.com/greenlang/greenlang/issues
- **Submit PRs**: See CONTRIBUTING.md for guidelines
- **Join Community**: Discord/Slack links in README

## 🙏 Acknowledgments

Thank you to all our beta testers who provided invaluable feedback, and to the open-source community for supporting sustainable software development.

## 📜 License

GreenLang is licensed under the Apache License 2.0. See LICENSE for details.

## 🎯 What's Next

### Upcoming in v0.2.1
- Fix Pydantic deprecation warnings
- Add more pack templates
- Improve documentation
- Enhanced cloud provider integrations

### Roadmap for v0.3.0
- Kubernetes operator for green scheduling
- Real-time grid carbon intensity integration
- ML-powered carbon optimization
- Advanced reporting dashboards

## 💚 Our Mission

**"Making every line of code count in the fight against climate change."**

GreenLang isn't just a technology platform - it's a movement. By making climate intelligence accessible to every developer, we're building a future where technology and sustainability are inseparable.

---

**The LangChain of Climate Intelligence is here. Join us in building a sustainable future, one line of code at a time.**

🌍 Code Green. Deploy Clean. Save Tomorrow. 🌱