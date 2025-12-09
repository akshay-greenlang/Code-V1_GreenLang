# Quick Start Guide

Get started with the GreenLang Agent Factory CLI in 5 minutes.

## Installation

```bash
# Navigate to CLI directory
cd C:\Users\aksha\Code-V1_GreenLang\GL-Agent-Factory\cli

# Install CLI
pip install -e .

# Verify installation
gl --version
```

## Your First Agent in 5 Steps

### Step 1: Initialize Project

```bash
# Create project directory
mkdir my-agent-project
cd my-agent-project

# Initialize Agent Factory project
gl init
```

This creates:
```
my-agent-project/
├── agents/          # Generated agents
├── specs/           # Agent specifications
├── tests/           # Test files
├── templates/       # Custom templates
└── config/          # Configuration files
    └── factory.yaml
```

### Step 2: List Available Templates

```bash
gl template list
```

Output:
```
Available Templates:

┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ ID         ┃ Name                         ┃ Version ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ basic      │ Basic Agent                  │ 1.0.0   │
│ regulatory │ Regulatory Compliance Agent  │ 1.0.0   │
│ api        │ API Integration Agent        │ 1.0.0   │
└────────────┴──────────────────────────────┴─────────┘
```

### Step 3: Create Agent Specification

```bash
# Initialize from template
gl template init basic --id my-first-agent

# Or create manually
cat > specs/my-first-agent.yaml << EOF
metadata:
  id: "my-first-agent"
  name: "My First Agent"
  version: "0.1.0"
  type: "basic"
  description: "My first GreenLang agent"

capabilities:
  - "task-execution"

architecture:
  framework: "greenlang"

dependencies:
  - "greenlang-sdk>=1.0.0"
EOF
```

### Step 4: Generate Agent

```bash
# Validate specification
gl agent validate specs/my-first-agent.yaml

# Generate agent
gl agent create specs/my-first-agent.yaml
```

Output:
```
Creating agent from specification: specs/my-first-agent.yaml

┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Agent Information                               ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ID          │ my-first-agent                   │
│ Name        │ My First Agent                   │
│ Version     │ 0.1.0                            │
│ Type        │ basic                            │
│ Output      │ agents/my-first-agent            │
└─────────────┴──────────────────────────────────┘

✓ Validation passed!

Agent generated successfully!

Output directory: agents/my-first-agent
Files created: 5

📦 my-first-agent
├── 📄 agent.py
├── 📄 agent.yaml
├── 📄 README.md
├── 📄 Dockerfile
└── 📁 tests
    └── 📄 test_agent.py
```

### Step 5: Test Agent

```bash
# Run tests
gl agent test agents/my-first-agent

# View agent details
gl agent info my-first-agent
```

## Next Steps

### Create a Regulatory Agent

```bash
# Initialize from regulatory template
gl template init regulatory --id nfpa86-agent

# Edit specification
# Add standards, capabilities, etc.
vim specs/nfpa86-agent.yaml

# Generate agent
gl agent create specs/nfpa86-agent.yaml

# Test
gl agent test agents/nfpa86-agent --coverage
```

### Publish to Registry

```bash
# Login to registry
gl registry login

# Publish agent
gl agent publish agents/my-first-agent --tag v0.1.0

# Search registry
gl registry search "my first"

# Pull published agent
gl registry pull my-first-agent:v0.1.0
```

### Customize Configuration

Edit `config/factory.yaml`:

```yaml
version: "1.0"

defaults:
  output_dir: "my-agents"  # Change output directory
  test_dir: "my-tests"

generator:
  enable_validation: true
  enable_tests: true
  enable_documentation: true

registry:
  url: "https://my-registry.example.com"
```

## Common Workflows

### Development Workflow

```bash
# 1. Create spec
gl template init basic --id dev-agent

# 2. Edit spec
vim specs/dev-agent.yaml

# 3. Validate
gl agent validate specs/dev-agent.yaml --strict

# 4. Generate
gl agent create specs/dev-agent.yaml --verbose

# 5. Test
gl agent test agents/dev-agent --coverage

# 6. Iterate (edit code, rerun tests)

# 7. Publish
gl agent publish agents/dev-agent --tag v1.0.0
```

### Regulatory Compliance Workflow

```bash
# 1. Initialize regulatory agent
gl template init regulatory --id compliance-agent

# 2. Configure standards
vim specs/compliance-agent.yaml
# Add NFPA 86, OSHA, etc.

# 3. Generate with all features
gl agent create specs/compliance-agent.yaml

# 4. Run comprehensive tests
gl agent test agents/compliance-agent --verbose --coverage

# 5. Generate documentation
gl agent info compliance-agent --verbose

# 6. Publish with documentation
gl agent publish agents/compliance-agent --tag v1.0.0
```

### Team Collaboration Workflow

```bash
# Developer A: Create agent
gl agent create specs/shared-agent.yaml
git add agents/shared-agent specs/shared-agent.yaml
git commit -m "Add shared agent"
git push

# Developer B: Pull and test
git pull
gl agent test agents/shared-agent

# Developer B: Make improvements
vim agents/shared-agent/agent.py
gl agent test agents/shared-agent

# Developer B: Publish new version
gl agent publish agents/shared-agent --tag v1.1.0
```

## CLI Cheat Sheet

### Agent Commands

```bash
gl agent create <spec>              # Generate agent
gl agent validate <spec>            # Validate spec
gl agent test <agent>               # Run tests
gl agent publish <agent>            # Publish to registry
gl agent list                       # List all agents
gl agent info <id>                  # Show agent details
```

### Template Commands

```bash
gl template list                    # List templates
gl template init <name>             # Initialize from template
gl template show <name>             # Show template details
```

### Registry Commands

```bash
gl registry search <query>          # Search registry
gl registry pull <id:version>       # Pull agent
gl registry push <agent>            # Push agent
gl registry login                   # Login to registry
gl registry logout                  # Logout
```

### Options

```bash
--verbose, -v                       # Verbose output
--quiet, -q                         # Quiet mode
--output, -o <dir>                  # Output directory
--dry-run                           # Simulate without changes
--help                              # Show help
```

## Examples

### Example 1: NFPA 86 Compliance Agent

```bash
# Create specification
cat > specs/nfpa86.yaml << EOF
metadata:
  id: "nfpa86-furnace-agent"
  name: "NFPA 86 Furnace Compliance Agent"
  version: "1.0.0"
  type: "regulatory"

capabilities:
  - "code-compliance-validation"
  - "safety-analysis"

configuration:
  standards:
    - name: "NFPA 86"
      version: "2023"
EOF

# Generate
gl agent create specs/nfpa86.yaml

# Test
gl agent test agents/nfpa86-furnace-agent --coverage

# Publish
gl agent publish agents/nfpa86-furnace-agent --tag v1.0.0
```

### Example 2: API Integration Agent

```bash
# Initialize
gl template init api --id weather-api-agent

# Generate
gl agent create specs/weather-api-agent.yaml

# Test
gl agent test agents/weather-api-agent
```

### Example 3: Multi-Agent System

```bash
# Create multiple agents
gl agent create specs/data-collector.yaml
gl agent create specs/data-processor.yaml
gl agent create specs/report-generator.yaml

# Test all
gl agent test agents/data-collector
gl agent test agents/data-processor
gl agent test agents/report-generator

# List all
gl agent list
```

## Troubleshooting

### Command not found: gl

```bash
# Check installation
pip show greenlang-agent-factory-cli

# Reinstall
pip install -e .
```

### Validation errors

```bash
# Use verbose mode to see details
gl agent validate specs/my-agent.yaml --verbose

# Check required fields
# - metadata.id
# - metadata.name
# - capabilities
```

### Test failures

```bash
# Run tests verbosely
gl agent test agents/my-agent --verbose

# Check test files exist
ls agents/my-agent/tests/
```

## Getting Help

```bash
# General help
gl --help

# Command-specific help
gl agent --help
gl agent create --help
gl template --help
gl registry --help
```

## Resources

- Full Documentation: [README.md](README.md)
- Installation Guide: [INSTALL.md](INSTALL.md)
- Templates: `cli/templates/`
- Examples: Search registry with `gl registry search`

## What's Next?

1. Explore available templates: `gl template list`
2. Read the full README: [README.md](README.md)
3. Create your first regulatory agent
4. Join the community and share your agents
5. Contribute new templates

Happy agent building!
