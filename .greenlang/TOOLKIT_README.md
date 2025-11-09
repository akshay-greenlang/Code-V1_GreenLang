# GreenLang Migration Toolkit

**Automated tools for migrating from custom code to GreenLang infrastructure**

---

## Overview

The GreenLang Migration Toolkit is a comprehensive suite of automated tools designed to help development teams migrate from custom implementations to standardized GreenLang infrastructure components.

### Key Features

- **Automated Detection**: Scans code for migration opportunities
- **Safe Migration**: Dry-run mode for all operations
- **Code Generation**: Boilerplate generation for new infrastructure
- **Real-time Dashboard**: Live migration progress tracking
- **Compliance**: ADR generation for custom implementations
- **Metrics**: Detailed usage and adoption analytics

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r .greenlang/toolkit-requirements.txt
```

### 2. Run Initial Assessment

```bash
python .greenlang/cli/greenlang.py report --directory GL-CBAM-APP --format html --output assessment.html
```

### 3. Scan for Opportunities

```bash
python .greenlang/cli/greenlang.py migrate --app GL-CBAM-APP --dry-run --format html --output opportunities.html
```

### 4. Launch Dashboard

```bash
python .greenlang/cli/greenlang.py dashboard --directory GL-CBAM-APP
# Open http://localhost:8080
```

---

## Tools Included

| Tool | Purpose | Command |
|------|---------|---------|
| **Migration Scanner** | Detect migration opportunities | `greenlang migrate` |
| **Import Rewriter** | Rewrite imports to GreenLang | `greenlang imports` |
| **Agent Converter** | Convert to Agent base class | `greenlang agents` |
| **Dependency Updater** | Update requirements.txt | `greenlang deps` |
| **Code Generator** | Generate infrastructure code | `greenlang generate` |
| **Usage Reporter** | Generate usage metrics | `greenlang report` |
| **ADR Generator** | Create ADRs | `greenlang adr` |
| **Dashboard** | Real-time progress tracking | `greenlang dashboard` |

---

## Common Use Cases

### Migrate LLM Code

```bash
# Scan for LLM code to migrate
python .greenlang/cli/greenlang.py migrate --app myapp --category llm --dry-run

# Apply high-confidence migrations
python .greenlang/cli/greenlang.py migrate --app myapp --category llm --min-confidence 0.9 --auto-fix
```

### Convert Custom Agents

```bash
# Preview agent conversions
python .greenlang/cli/greenlang.py agents --path myapp/agents --dry-run --show-diff

# Apply conversions
python .greenlang/cli/greenlang.py agents --path myapp/agents
```

### Generate New Infrastructure

```bash
# Create new agent
python .greenlang/cli/greenlang.py generate --type agent --name DataProcessor --batch

# Create pipeline
python .greenlang/cli/greenlang.py generate --type pipeline --name ProcessingPipeline --agents "Agent1,Agent2"

# Create LLM session
python .greenlang/cli/greenlang.py generate --type llm-session --name ChatManager --provider openai
```

### Track Progress

```bash
# Generate usage report
python .greenlang/cli/greenlang.py report --format html --output weekly-report.html

# Start dashboard
python .greenlang/cli/greenlang.py dashboard
```

---

## Documentation

- **[Installation Guide](INSTALLATION.md)**: Setup and installation
- **[Migration Guide](MIGRATION_TOOLKIT_GUIDE.md)**: Complete reference
- **Sample Code**: `test_sample.py` for testing

---

## Metrics & Goals

### Target Metrics

- **IUM (Infrastructure Usage Metric)**: > 80%
- **Adoption Rate**: > 90% of files
- **ADR Coverage**: 100% of custom code

### Measured by Reports

```bash
python .greenlang/cli/greenlang.py report --format html --output metrics.html
```

---

## Architecture

```
.greenlang/
├── scripts/              # Individual migration tools
│   ├── migrate_to_infrastructure.py
│   ├── rewrite_imports.py
│   ├── convert_to_base_agent.py
│   ├── update_dependencies.py
│   ├── generate_infrastructure_code.py
│   ├── generate_usage_report.py
│   ├── create_adr.py
│   └── serve_dashboard.py
├── cli/                  # Unified CLI
│   └── greenlang.py
├── adr/                  # Architecture Decision Records
├── TOOLKIT_README.md     # This file
├── INSTALLATION.md
├── MIGRATION_TOOLKIT_GUIDE.md
├── toolkit-requirements.txt
└── test_sample.py
```

---

## Support

- **Documentation**: See `MIGRATION_TOOLKIT_GUIDE.md`
- **Installation Help**: See `INSTALLATION.md`
- **Examples**: Check `test_sample.py`

---

**Version**: 1.0.0
**Date**: 2025-11-09
**Author**: GreenLang Migration Team
