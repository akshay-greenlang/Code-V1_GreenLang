# GreenLang Comprehensive Demo
**Week of 2025-09-26**

This directory contains a comprehensive demonstration of GreenLang's core features, showcasing the platform's security-first architecture and complete workflow capabilities.

## 🎯 Demo Overview

This demo validates GreenLang's enterprise-grade features through a realistic climate analysis scenario, demonstrating:

- **Installation & Setup** - Complete environment setup from scratch
- **Security-First Architecture** - Default-deny capability model with fine-grained controls
- **Pack Management** - Creation, validation, and execution of secure analysis packs
- **Supply Chain Security** - SBOM generation and signature verification workflows
- **Policy Enforcement** - Runtime security controls and capability restrictions
- **Observability** - Metrics collection and performance monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Git
- Unix-like environment (Linux, macOS, WSL on Windows)

### Running the Demo

```bash
# Make the script executable
chmod +x run_demo.sh

# Run the comprehensive demo
./run_demo.sh
```

The script is fully interactive and will guide you through each demonstration step.

## 📋 Demo Flow

### 1. Installation Demo
- Creates isolated virtual environment
- Installs GreenLang from source or wheel
- Verifies installation and shows version information

### 2. Environment Diagnostics
- Runs `gl doctor` to validate the environment
- Checks for required dependencies and configurations

### 3. Pack Creation & Management
- Creates a sample climate analysis pack with security configurations
- Demonstrates pack structure and metadata
- Shows capability-based security declarations

### 4. SBOM Generation
- Generates Software Bill of Materials for the demo pack
- Shows provenance tracking and dependency analysis
- Creates CycloneDX-compliant SBOM documents

### 5. Security Policy Demonstration
- Tests network access restrictions (default deny)
- Validates filesystem access controls
- Shows subprocess execution blocking
- Demonstrates capability-based security model

### 6. Secure Pipeline Execution
- Executes climate analysis with security restrictions
- Shows data processing within controlled environment
- Generates analysis report in secure output location

### 7. Metrics Collection
- Collects performance and execution metrics
- Shows observability data and system telemetry
- Generates comprehensive execution report

## 📁 Generated Artifacts

After running the demo, you'll find:

```
examples/weekly/2025-09-26/
├── run_demo.sh                    # Main demo script
├── README.md                      # This documentation
├── RESULTS.md                     # Execution results report
├── metrics_report.json           # Performance metrics
├── demo-packs/                   # Created demo packs
│   └── climate-demo/
│       ├── pack.yaml             # Pack manifest with security
│       ├── gl.yaml               # Pipeline definition
│       ├── climate_analyzer.py   # Secure analysis agent
│       └── sbom.json             # Software Bill of Materials
└── data/
    └── demo_climate_data.json    # Sample input data
```

## 🔒 Security Features Demonstrated

### Default-Deny Capability Model
- **Network Access**: Denied by default, explicit allow-lists required
- **Filesystem Access**: Restricted to declared read/write paths only
- **Subprocess Execution**: Completely disabled unless explicitly allowed
- **Time Operations**: Fine-grained control over clock access

### Supply Chain Security
- **Pack Signatures**: Cryptographic verification of pack integrity
- **SBOM Generation**: Complete dependency tracking and provenance
- **Vulnerability Scanning**: Integration points for security analysis

### Runtime Protection
- **Sandbox Execution**: Isolated execution environments
- **Resource Limits**: Memory, CPU, and I/O restrictions
- **Policy Enforcement**: OPA-based policy evaluation at runtime

## 🏗️ Architecture Highlights

### Pack-Based Design
- **Modular Architecture**: Self-contained analysis units
- **Dependency Management**: Explicit dependency declarations
- **Version Compatibility**: Semantic versioning with compatibility checks

### Pipeline Orchestration
- **DAG Execution**: Dependency-aware step execution
- **Data Flow**: Secure inter-step data passing
- **Error Handling**: Comprehensive error recovery and reporting

### Security Integration
- **Zero Trust**: Every operation requires explicit permission
- **Least Privilege**: Minimal required capabilities only
- **Auditability**: Complete execution audit trails

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|--------|
| Setup Time | ~30s | Including virtual environment creation |
| Execution Time | ~15s | Core pipeline execution |
| Memory Overhead | <50MB | Security monitoring overhead |
| Security Validation | <5s | Policy evaluation and checks |

## 🔧 Customization Options

### Environment Variables
```bash
# Hub configuration
export GL_HUB="https://your-hub.example.com"
export GL_REGION="us-west-2"

# Security settings
export GL_POLICY_BUNDLE="/path/to/policies"
export GL_TELEMETRY="on"

# Performance tuning
export GL_CACHE_DIR="/tmp/gl-cache"
export GL_PARALLEL_JOBS="4"
```

### Demo Modifications
The demo script accepts several environment variables for customization:

```bash
# Skip interactive prompts
export DEMO_NON_INTERACTIVE=true

# Use different pack template
export DEMO_PACK_TEMPLATE="advanced"

# Custom output directory
export DEMO_OUTPUT_DIR="/custom/path"
```

## 🐛 Troubleshooting

### Common Issues

**Virtual Environment Creation Fails**
```bash
# Install python3-venv if missing
sudo apt-get install python3-venv  # Ubuntu/Debian
brew install python@3.10           # macOS
```

**Permission Denied on /tmp/outputs**
```bash
# Create directory manually with correct permissions
sudo mkdir -p /tmp/outputs
sudo chmod 755 /tmp/outputs
```

**SBOM Generation Unavailable**
- The demo includes fallback mock SBOM generation
- Real SBOM generation requires additional tools and configuration

### Debug Mode
```bash
# Run with verbose output
bash -x run_demo.sh

# Enable GreenLang debug logging
export GL_DEBUG=true
./run_demo.sh
```

## 🧪 Testing Integration

The demo can be integrated into CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Run GreenLang Demo
  run: |
    cd examples/weekly/2025-09-26
    export DEMO_NON_INTERACTIVE=true
    ./run_demo.sh

- name: Validate Results
  run: |
    test -f examples/weekly/2025-09-26/RESULTS.md
    test -f /tmp/outputs/climate_report.json
```

## 📈 Scaling Considerations

### Production Deployment
- **Container Integration**: Demo pack can be containerized
- **Kubernetes Support**: Native K8s execution backend available
- **Multi-tenancy**: Isolated execution per tenant/project

### Enterprise Features
- **RBAC Integration**: Fine-grained access controls
- **Audit Logging**: Complete audit trail in structured format
- **Compliance Reporting**: Automated compliance validation

## 🤝 Contributing

To extend this demo:

1. **Add New Features**: Extend the demo script with additional GreenLang capabilities
2. **Improve Security**: Add more comprehensive security validations
3. **Performance Testing**: Include load testing and benchmarking
4. **Documentation**: Enhance inline documentation and examples

## 📚 Additional Resources

- **GreenLang Documentation**: `/docs/getting-started.md`
- **Security Guide**: `/docs/security/`
- **Pack Development**: `/examples/packs/`
- **API Reference**: Generated documentation in `/docs/api/`

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-09-26 | Initial comprehensive demo |
| - | - | Added SBOM generation |
| - | - | Enhanced security validations |
| - | - | Improved error handling |

---

**Generated by**: GreenLang Demo Framework v1.0
**Last Updated**: 2025-09-26
**Compatibility**: GreenLang v0.3.0+