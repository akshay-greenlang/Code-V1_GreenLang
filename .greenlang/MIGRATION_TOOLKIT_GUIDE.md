# GreenLang Migration Toolkit Guide

Complete guide to migrating from custom code to GreenLang infrastructure using automated tools.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Tools Reference](#tools-reference)
5. [Migration Workflow](#migration-workflow)
6. [Examples](#examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

---

## Overview

The GreenLang Migration Toolkit provides automated tools to help developers migrate from custom code to GreenLang infrastructure, ensuring:

- **Consistency**: Standardized infrastructure usage across all applications
- **Quality**: Automated detection of migration opportunities
- **Efficiency**: Batch processing and automated refactoring
- **Compliance**: ADR generation and infrastructure usage tracking

### Toolkit Components

| Tool | Purpose | File |
|------|---------|------|
| **Migration Scanner** | Detect migration opportunities | `migrate_to_infrastructure.py` |
| **Import Rewriter** | Rewrite imports to GreenLang | `rewrite_imports.py` |
| **Agent Converter** | Convert to Agent base class | `convert_to_base_agent.py` |
| **Dependency Updater** | Update requirements.txt | `update_dependencies.py` |
| **Code Generator** | Generate infrastructure boilerplate | `generate_infrastructure_code.py` |
| **Usage Reporter** | Generate usage metrics | `generate_usage_report.py` |
| **ADR Generator** | Create Architecture Decision Records | `create_adr.py` |
| **Dashboard** | Real-time migration dashboard | `serve_dashboard.py` |
| **CLI** | Unified command-line interface | `greenlang.py` |

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for diff generation)
- pip (Python package manager)

### Install Dependencies

```bash
# Install required Python packages
pip install astor flask

# Optional: Install for enhanced features
pip install matplotlib plotly  # For advanced visualizations
```

### Setup

```bash
# Navigate to GreenLang root
cd C:\Users\aksha\Code-V1_GreenLang

# Make CLI executable (optional)
chmod +x .greenlang/cli/greenlang.py

# Add to PATH (optional, for 'greenlang' command)
# Windows:
set PATH=%PATH%;C:\Users\aksha\Code-V1_GreenLang\.greenlang\cli

# Linux/Mac:
export PATH=$PATH:/path/to/Code-V1_GreenLang/.greenlang/cli
```

---

## Quick Start

### 1. Scan Your Application

```bash
# Scan for migration opportunities
python .greenlang/cli/greenlang.py migrate --app GL-CBAM-APP --dry-run --format html --output report.html
```

### 2. Generate Usage Report

```bash
# Generate infrastructure usage report
python .greenlang/cli/greenlang.py report --directory GL-CBAM-APP --format html --output usage.html
```

### 3. View Dashboard

```bash
# Start real-time dashboard
python .greenlang/cli/greenlang.py dashboard --directory GL-CBAM-APP --port 8080

# Open browser: http://localhost:8080
```

### 4. Generate Infrastructure Code

```bash
# Create a new agent
python .greenlang/cli/greenlang.py generate --type agent --name DataProcessorAgent --batch --output my_agent.py
```

---

## Tools Reference

### 1. Migration Scanner

**Purpose**: Detects custom code patterns and suggests GreenLang replacements.

**Usage**:
```bash
python .greenlang/scripts/migrate_to_infrastructure.py <path> [options]
```

**Options**:
- `--dry-run`: Preview changes without applying
- `--auto-fix`: Automatically apply suggestions
- `--format {text,json,html}`: Report format
- `--output <file>`: Output file
- `--category <category>`: Filter by category (llm, agent, cache, etc.)
- `--min-confidence <0-1>`: Minimum confidence threshold

**Detection Patterns**:

| Pattern | Detects | Suggests |
|---------|---------|----------|
| `openai_client` | OpenAI API usage | `greenlang.intelligence.ChatSession` |
| `custom_agent` | Custom agent classes | `greenlang.sdk.base.Agent` |
| `jsonschema_validation` | JSONSchema usage | `greenlang.validation.ValidationFramework` |
| `redis_client` | Redis client | `greenlang.cache.CacheManager` |
| `batch_processing` | Custom batch processing | `Agent.batch_process()` |

**Example**:
```bash
# Scan single file
python migrate_to_infrastructure.py mycode.py --dry-run

# Scan directory with HTML report
python migrate_to_infrastructure.py GL-CBAM-APP --format html --output report.html

# Auto-fix with high confidence
python migrate_to_infrastructure.py mycode.py --auto-fix --min-confidence 0.8
```

---

### 2. Import Rewriter

**Purpose**: Rewrites imports from external libraries to GreenLang equivalents.

**Usage**:
```bash
python .greenlang/scripts/rewrite_imports.py <path> [options]
```

**Options**:
- `--dry-run`: Preview changes
- `--show-diff`: Show unified diff
- `--git-diff`: Generate git diff

**Import Mappings**:

| Original | GreenLang |
|----------|-----------|
| `from openai import OpenAI` | `from greenlang.intelligence import ChatSession` |
| `from redis import Redis` | `from greenlang.cache import CacheManager` |
| `from jsonschema import validate` | `from greenlang.validation import ValidationFramework` |
| `from langchain import Chain` | `from greenlang.sdk.base import Pipeline` |

**Example**:
```bash
# Rewrite imports in file
python rewrite_imports.py mycode.py --show-diff

# Rewrite entire directory
python rewrite_imports.py GL-CBAM-APP --dry-run

# Apply changes and show git diff
python rewrite_imports.py mycode.py --git-diff
```

---

### 3. Agent Converter

**Purpose**: Converts custom agent classes to inherit from `greenlang.sdk.base.Agent`.

**Usage**:
```bash
python .greenlang/scripts/convert_to_base_agent.py <path> [options]
```

**Options**:
- `--dry-run`: Preview changes
- `--show-diff`: Show code differences

**Transformations**:

**Before**:
```python
class MyAgent:
    def process(self, data):
        return self.analyze(data)
```

**After**:
```python
from greenlang.sdk.base import Agent

class MyAgent(Agent):
    def execute(self, input_data):
        return self.analyze(input_data)
```

**Example**:
```bash
# Convert agent class
python convert_to_base_agent.py my_agent.py --show-diff

# Convert all agents in directory
python convert_to_base_agent.py GL-CBAM-APP/agents --dry-run
```

---

### 4. Dependency Updater

**Purpose**: Updates `requirements.txt` with GreenLang packages.

**Usage**:
```bash
python .greenlang/scripts/update_dependencies.py [options]
```

**Options**:
- `--file <path>`: Requirements file (default: requirements.txt)
- `--dry-run`: Preview changes
- `--remove-redundant`: Remove packages replaced by GreenLang
- `--scan <directory>`: Scan for used packages
- `--install`: Install packages after update
- `--show-diff`: Show diff

**Example**:
```bash
# Update requirements.txt
python update_dependencies.py --dry-run --show-diff

# Scan for used packages and update
python update_dependencies.py --scan GL-CBAM-APP --remove-redundant

# Update and install
python update_dependencies.py --install
```

---

### 5. Code Generator

**Purpose**: Generates infrastructure boilerplate code.

**Usage**:
```bash
python .greenlang/scripts/generate_infrastructure_code.py --type <type> --name <name> [options]
```

**Types**:
- `agent`: Agent class
- `pipeline`: Pipeline with multiple agents
- `llm-session`: LLM session manager
- `cache`: Cache manager
- `validation`: Validation schemas
- `config`: Configuration manager

**Options**:
- `--output <file>`: Output file
- `--agents <list>`: Agent names for pipeline (comma-separated)
- `--provider <name>`: LLM provider (openai, anthropic)
- `--model <name>`: LLM model name
- `--batch`: Include batch processing
- `--description <text>`: Description

**Examples**:
```bash
# Generate agent with batch processing
python generate_infrastructure_code.py --type agent --name DataProcessor --batch --output agent.py

# Generate pipeline
python generate_infrastructure_code.py --type pipeline --name ProcessingPipeline --agents "Agent1,Agent2,Agent3"

# Generate LLM session
python generate_infrastructure_code.py --type llm-session --name ChatManager --provider openai --model gpt-4

# Generate cache manager
python generate_infrastructure_code.py --type cache --name AppCache

# Generate validation schemas
python generate_infrastructure_code.py --type validation --name DataValidator
```

---

### 6. Usage Report Generator

**Purpose**: Generates comprehensive infrastructure usage reports.

**Usage**:
```bash
python .greenlang/scripts/generate_usage_report.py <directory> [options]
```

**Options**:
- `--format {text,json,html}`: Report format (default: html)
- `--output <file>`: Output file

**Metrics Calculated**:
- **IUM (Infrastructure Usage Metric)**: Percentage of GreenLang imports vs total imports
- **Adoption Rate**: Percentage of files using GreenLang
- **Lines of Code**: Total LOC analyzed
- **Component Usage**: Usage breakdown by component
- **Top Files**: Files with highest IUM

**Example**:
```bash
# Generate HTML report
python generate_usage_report.py GL-CBAM-APP --format html --output usage.html

# Generate JSON data
python generate_usage_report.py GL-CBAM-APP --format json --output data.json

# Text report to console
python generate_usage_report.py GL-CBAM-APP --format text
```

---

### 7. ADR Generator

**Purpose**: Creates Architecture Decision Records for custom implementations.

**Usage**:
```bash
python .greenlang/scripts/create_adr.py [options]
```

**Options**:
- `--list`: List all existing ADRs
- `--validate <file>`: Validate an ADR file
- `--adr-dir <path>`: ADR directory (default: .greenlang/adr)

**Interactive Mode**:
```bash
# Create new ADR (interactive)
python create_adr.py

# Follow prompts to create ADR
```

**Validation**:
```bash
# Validate ADR
python create_adr.py --validate .greenlang/adr/ADR-001-custom-implementation.md

# List all ADRs
python create_adr.py --list
```

**ADR Template**:
```markdown
# ADR-001: Custom Implementation Title

**Status**: proposed | accepted | rejected | deprecated | superseded

**Date**: 2025-11-09

## Context
[Describe the problem]

## GreenLang Infrastructure Alternatives Considered
- Alternative 1
- Alternative 2

## Justification for Custom Implementation
[Why infrastructure can't be used]

## Decision
[What was decided]

## Consequences
[Positive and negative impacts]

## Future Migration Plan
[How/when to migrate to infrastructure]
```

---

### 8. Migration Dashboard

**Purpose**: Real-time dashboard showing migration progress.

**Usage**:
```bash
python .greenlang/scripts/serve_dashboard.py [options]
```

**Options**:
- `--directory <path>`: Directory to monitor (default: current)
- `--port <number>`: Server port (default: 8080)

**Features**:
- **Real-time metrics**: Overall IUM, adoption rate, files migrated
- **Component usage**: Visual breakdown by component
- **Team leaderboard**: Top contributors
- **ADR tracking**: All Architecture Decision Records
- **Auto-refresh**: Updates every 5 seconds

**Example**:
```bash
# Start dashboard
python serve_dashboard.py --directory GL-CBAM-APP --port 8080

# Open browser: http://localhost:8080
```

---

### 9. CLI Tool

**Purpose**: Unified command-line interface for all tools.

**Usage**:
```bash
python .greenlang/cli/greenlang.py <command> [options]
```

**Commands**:

| Command | Description |
|---------|-------------|
| `migrate` | Scan and migrate code |
| `imports` | Rewrite imports |
| `agents` | Convert agent classes |
| `deps` | Update dependencies |
| `generate` | Generate infrastructure code |
| `report` | Generate usage report |
| `adr` | Manage ADRs |
| `dashboard` | Start migration dashboard |
| `check` | Check specific file |
| `status` | Show migration status |

**Examples**:
```bash
# Migrate application
python greenlang.py migrate --app GL-CBAM-APP --dry-run

# Generate agent
python greenlang.py generate --type agent --name MyAgent

# Create report
python greenlang.py report --format html --output report.html

# Start dashboard
python greenlang.py dashboard

# Create ADR
python greenlang.py adr

# Check file
python greenlang.py check --file mycode.py

# Show status
python greenlang.py status --directory GL-CBAM-APP
```

---

## Migration Workflow

### Step-by-Step Migration Process

#### Phase 1: Assessment (Week 1)

1. **Initial Scan**
   ```bash
   python greenlang.py report --directory GL-CBAM-APP --format html --output baseline.html
   ```

2. **Identify Opportunities**
   ```bash
   python greenlang.py migrate --app GL-CBAM-APP --dry-run --format html --output opportunities.html
   ```

3. **Review Reports**
   - Open `baseline.html` to see current state
   - Open `opportunities.html` to see migration suggestions
   - Prioritize high-confidence, low-risk migrations

#### Phase 2: Dependencies (Week 2)

1. **Update Dependencies**
   ```bash
   python greenlang.py deps --scan GL-CBAM-APP --remove-redundant --show-diff
   ```

2. **Review Changes**
   - Check `requirements.txt` diff
   - Ensure all GreenLang packages are added

3. **Install & Test**
   ```bash
   python greenlang.py deps --install
   pytest  # Run tests to ensure nothing broke
   ```

#### Phase 3: Imports (Week 3)

1. **Rewrite Imports**
   ```bash
   python greenlang.py imports --path GL-CBAM-APP --dry-run --show-diff
   ```

2. **Review Diffs**
   - Check all import changes
   - Verify no breaking changes

3. **Apply Changes**
   ```bash
   python greenlang.py imports --path GL-CBAM-APP
   pytest  # Run tests
   ```

#### Phase 4: Agents (Week 4)

1. **Convert Agents**
   ```bash
   python greenlang.py agents --path GL-CBAM-APP/agents --dry-run --show-diff
   ```

2. **Review Conversions**
   - Check each agent class conversion
   - Verify method mappings

3. **Apply & Test**
   ```bash
   python greenlang.py agents --path GL-CBAM-APP/agents
   pytest tests/test_agents.py
   ```

#### Phase 5: Code Migration (Week 5-6)

1. **Migrate Custom Code**
   ```bash
   python greenlang.py migrate --app GL-CBAM-APP --category llm --auto-fix --min-confidence 0.8
   ```

2. **Create ADRs for Exceptions**
   ```bash
   python greenlang.py adr
   # Document why custom code is needed
   ```

3. **Test Thoroughly**
   ```bash
   pytest  # All tests
   python run_integration_tests.py
   ```

#### Phase 6: Validation (Week 7)

1. **Generate Final Report**
   ```bash
   python greenlang.py report --directory GL-CBAM-APP --format html --output final-report.html
   ```

2. **Review Metrics**
   - Target: IUM > 80%
   - Target: Adoption Rate > 90%
   - All ADRs documented

3. **Launch Dashboard**
   ```bash
   python greenlang.py dashboard --directory GL-CBAM-APP
   ```

---

## Examples

### Example 1: Migrate LLM Code

**Before**:
```python
import openai

client = openai.OpenAI(api_key="sk-...")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After**:
```python
from greenlang.intelligence import ChatSession

session = ChatSession(provider="openai", model="gpt-4")
response = session.send_message("Hello")
```

**Migration Command**:
```bash
python greenlang.py migrate --app myapp --category llm --auto-fix
```

---

### Example 2: Convert Custom Agent

**Before**:
```python
class DataAnalyzer:
    def __init__(self, config):
        self.config = config

    def process(self, data):
        # Analyze data
        return analysis
```

**After**:
```python
from greenlang.sdk.base import Agent

class DataAnalyzer(Agent):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}

    def execute(self, input_data):
        # Analyze data
        return analysis
```

**Migration Command**:
```bash
python greenlang.py agents --path data_analyzer.py
```

---

### Example 3: Generate New Infrastructure

**Generate Complete Agent**:
```bash
python greenlang.py generate --type agent --name SentimentAnalyzer --batch --description "Analyzes sentiment of text data" --output sentiment_agent.py
```

**Generate Pipeline**:
```bash
python greenlang.py generate --type pipeline --name AnalysisPipeline --agents "DataLoader,SentimentAnalyzer,ResultsAggregator" --output pipeline.py
```

**Generate LLM Session**:
```bash
python greenlang.py generate --type llm-session --name GPTSession --provider openai --model gpt-4 --output llm_session.py
```

---

## Best Practices

### 1. Always Start with Dry-Run

```bash
# Good
python greenlang.py migrate --app myapp --dry-run

# Bad (no dry-run)
python greenlang.py migrate --app myapp --auto-fix
```

### 2. Review Diffs Before Applying

```bash
python greenlang.py imports --path myapp --show-diff
# Review output carefully
python greenlang.py imports --path myapp  # Apply only after review
```

### 3. Test After Each Phase

```bash
python greenlang.py deps --install
pytest  # Always test!

python greenlang.py agents --path myapp
pytest tests/test_agents.py  # Test agents
```

### 4. Document Exceptions with ADRs

```bash
# If custom code is needed:
python greenlang.py adr
# Fill out ADR explaining why
```

### 5. Monitor Progress

```bash
# Keep dashboard running
python greenlang.py dashboard --directory myapp

# Generate reports regularly
python greenlang.py report --format html --output weekly-report.html
```

### 6. Use High Confidence First

```bash
# Migrate high-confidence patterns first
python greenlang.py migrate --app myapp --min-confidence 0.9 --auto-fix

# Then lower confidence
python greenlang.py migrate --app myapp --min-confidence 0.7 --dry-run
```

---

## Troubleshooting

### Issue: Migration tool not finding opportunities

**Solution**:
```bash
# Check if imports are present
python greenlang.py check --file mycode.py

# Lower confidence threshold
python greenlang.py migrate --app myapp --min-confidence 0.5
```

### Issue: Import rewriter breaking code

**Solution**:
```bash
# Always use dry-run first
python greenlang.py imports --path myapp --dry-run --show-diff

# Review diffs carefully
# Apply to single file first
python greenlang.py imports --path myfile.py
pytest tests/test_myfile.py
```

### Issue: Agent converter changing behavior

**Solution**:
```bash
# Check diff before applying
python greenlang.py agents --path myagent.py --show-diff

# Test thoroughly after conversion
pytest tests/test_agents.py -v
```

### Issue: Dashboard not updating

**Solution**:
```bash
# Ensure Flask is installed
pip install flask

# Check directory path
python greenlang.py dashboard --directory $(pwd)/GL-CBAM-APP

# Check port availability
python greenlang.py dashboard --port 8081
```

### Issue: Dependencies conflict

**Solution**:
```bash
# Scan for actual usage first
python greenlang.py deps --scan GL-CBAM-APP

# Review suggested packages
python greenlang.py deps --dry-run --show-diff

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
python greenlang.py deps --install
```

---

## FAQ

### Q: Can I undo migrations?

**A**: Yes! All tools support `--dry-run` and generate diffs. Use git:
```bash
git diff  # Review changes
git checkout -- myfile.py  # Undo specific file
git reset --hard HEAD  # Undo all (use carefully!)
```

### Q: How do I migrate gradually?

**A**: Migrate by category:
```bash
# Week 1: LLM code
python greenlang.py migrate --category llm --auto-fix

# Week 2: Caching
python greenlang.py migrate --category cache --auto-fix

# Week 3: Agents
python greenlang.py agents --path myapp
```

### Q: What if GreenLang doesn't support my use case?

**A**: Create an ADR:
```bash
python greenlang.py adr
# Document your requirements
# Justify custom implementation
# Plan future migration
```

### Q: How do I track team progress?

**A**: Use the dashboard:
```bash
python greenlang.py dashboard --directory myapp
# View at http://localhost:8080
# Shows team leaderboard and metrics
```

### Q: Can I customize migration patterns?

**A**: Yes! Edit `migrate_to_infrastructure.py`:
```python
# Add custom pattern in _initialize_patterns()
MigrationPattern(
    name="my_custom_pattern",
    description="My custom migration",
    detect_pattern=r"my_pattern",
    detect_imports=["mylib"],
    suggest_code="from greenlang import MyFeature",
    suggest_imports=["from greenlang import MyFeature"],
    category="custom",
    confidence=0.9
)
```

### Q: How do I validate migration quality?

**A**:
```bash
# Generate usage report
python greenlang.py report --format html --output report.html

# Check metrics:
# - IUM should be > 80%
# - Adoption rate should be > 90%
# - All custom code should have ADRs
```

---

## Support

For issues, questions, or contributions:

- **Documentation**: `.greenlang/MIGRATION_TOOLKIT_GUIDE.md`
- **Issues**: Create GitHub issue
- **Email**: greenlang-support@example.com

---

**Last Updated**: 2025-11-09
**Version**: 1.0.0
**Author**: GreenLang Migration Team
