# GL-VCCI CLI - Command Line Interface

**Version:** 1.0.0
**Date:** 2025-11-08
**Status:** Production Ready ✓

## Overview

The GL-VCCI CLI provides a powerful, beautiful command-line interface for the Scope 3 Carbon Intelligence Platform. Built with Typer and Rich, it offers an intuitive way to manage carbon accounting workflows from the terminal.

## Features

- **Multi-format Data Ingestion** (CSV, JSON, Excel, XML, PDF)
- **Supplier Engagement Campaigns** (Email, Analytics, Gamification)
- **End-to-End Pipeline Orchestration** (Intake → Calculate → Analyze → Report)
- **Beautiful Terminal UI** (Progress bars, tables, panels, trees)
- **Agent Integration** (ValueChainIntakeAgent, SupplierEngagementAgent, etc.)
- **Comprehensive Error Handling**
- **Real-time Progress Tracking**

## Installation

```bash
# Install dependencies
pip install typer rich

# Optional dependencies for full functionality
pip install pandas openpyxl pdfplumber sendgrid
```

## Quick Start

```bash
# Check platform status
python -m cli.main status

# Show platform info
python -m cli.main info

# Run complete pipeline
python -m cli.main pipeline run --input data/ --output results/ --categories all

# Get help
python -m cli.main --help
```

## Command Structure

```
cli/
├── main.py                 # Main CLI entry point
├── commands/              # Command modules
│   ├── __init__.py
│   ├── intake.py          # Data ingestion commands
│   ├── engage.py          # Supplier engagement commands
│   └── pipeline.py        # Pipeline orchestration commands
└── README.md              # This file
```

## Available Commands

### Core Commands (built into main.py)
- `status` - Check platform status and health
- `calculate` - Calculate Scope 3 emissions for specific category
- `analyze` - Analyze emissions data for insights
- `report` - Generate compliance reports
- `config` - Manage platform configuration
- `categories` - List all 15 Scope 3 categories
- `info` - Display platform information

### Intake Commands (intake.py)
- `intake file` - Ingest single file
- `intake batch` - Batch process directory
- `intake status` - Show intake status

### Engagement Commands (engage.py)
- `engage list` - List all campaigns
- `engage create` - Create new campaign
- `engage send` - Send campaign emails
- `engage status` - Show campaign analytics
- `engage leaderboard` - Display supplier rankings

### Pipeline Commands (pipeline.py)
- `pipeline run` - Execute complete workflow
- `pipeline status` - Show pipeline run status

## Examples

### Example 1: Data Ingestion
```bash
# Ingest CSV file
python -m cli.main intake file --file suppliers.csv

# Batch process directory
python -m cli.main intake batch --directory data/ --pattern "*.csv"
```

### Example 2: Supplier Engagement
```bash
# Create campaign
python -m cli.main engage create --name "Q1 2026" --template standard

# Send emails
python -m cli.main engage send --campaign-id CAMP-ABC123

# Check status
python -m cli.main engage status --campaign-id CAMP-ABC123 --detailed
```

### Example 3: Complete Pipeline
```bash
# Run full pipeline
python -m cli.main pipeline run \
  --input data/suppliers.csv \
  --output results/q4_2025/ \
  --categories all \
  --format ghg-protocol
```

### Example 4: Step-by-Step Processing
```bash
# Step 1: Ingest
python -m cli.main intake file --file data.csv

# Step 2: Calculate
python -m cli.main calculate --category 1 --input data.csv --output results.json

# Step 3: Analyze
python -m cli.main analyze --input results.json --type hotspot

# Step 4: Report
python -m cli.main report --input results.json --format ghg-protocol
```

## Usage Patterns

### Global Options
```bash
# Verbose output
python -m cli.main --verbose <command>

# JSON output
python -m cli.main --json <command>

# Custom config file
python -m cli.main --config myconfig.yaml <command>

# Show version
python -m cli.main --version
```

### Getting Help
```bash
# Main help
python -m cli.main --help

# Command-specific help
python -m cli.main intake --help
python -m cli.main engage --help
python -m cli.main pipeline --help

# Subcommand help
python -m cli.main intake file --help
python -m cli.main engage create --help
```

## Rich UI Components

The CLI uses Rich library for beautiful terminal output:

### Progress Bars
- Spinners for indeterminate operations
- Progress bars for tracked operations
- Multi-stage progress tracking

### Tables
- Color-coded status indicators
- Sortable columns
- Pagination support

### Panels
- Success/error messages
- Configuration displays
- Summary information

### Trees
- Hierarchical data display
- Pipeline stage results
- Configuration structure

## Architecture

```
CLI Layer (Typer + Rich)
    ↓
Command Modules
    ↓
Agent Layer
    ↓
Core Services
    ↓
Data Layer
```

## Agent Integration

### ValueChainIntakeAgent
```python
from services.agents.intake.agent import ValueChainIntakeAgent

agent = ValueChainIntakeAgent(tenant_id="cli-user")
result = agent.ingest_file(file_path, format, entity_type)
```

### SupplierEngagementAgent
```python
from services.agents.engagement.agent import SupplierEngagementAgent

agent = SupplierEngagementAgent()
campaign = agent.create_campaign(name, target_suppliers, duration_days)
```

### Pipeline Integration
- Intake → ValueChainIntakeAgent
- Calculate → Scope3CalculatorAgent
- Analyze → HotspotAnalysisAgent
- Report → ReportingAgent

## Error Handling

All commands include comprehensive error handling:

```python
try:
    result = agent.process()
except AgentError as e:
    console.print(f"[red]Error:[/red] {str(e)}")
    sys.exit(1)
```

Exit codes:
- `0` - Success
- `1` - Error (agent unavailable, file not found, etc.)
- `2` - Invalid arguments

## Development

### Adding New Commands

1. Create command file in `cli/commands/`:
```python
# cli/commands/mycommand.py
import typer
from rich.console import Console

mycommand_app = typer.Typer(name="mycommand")
console = Console()

@mycommand_app.command("action")
def my_action():
    """Action description."""
    console.print("Hello!")

__all__ = ["mycommand_app"]
```

2. Update `cli/commands/__init__.py`:
```python
from .mycommand import mycommand_app

__all__ = ["mycommand_app"]
```

3. Register in `cli/main.py`:
```python
from cli.commands.mycommand import mycommand_app

app.add_typer(mycommand_app, name="mycommand")
```

### Testing

```bash
# Test import
python -c "from cli.main import app; print('Success!')"

# Test command
python -m cli.main --help
python -m cli.main mycommand --help
```

## Documentation

- **CLI_COMMANDS_SUMMARY.md** - Comprehensive implementation summary
- **CLI_QUICK_REFERENCE.md** - Quick reference guide
- **cli/README.md** - This file

## Dependencies

### Required
```
typer>=0.9.0
rich>=13.0.0
```

### Optional
```
pandas>=2.0.0          # CSV/Excel parsing
openpyxl>=3.1.0        # Excel support
pdfplumber>=0.10.0     # PDF parsing
sendgrid>=6.11.0       # Email sending
```

## Troubleshooting

### ImportError: No module named 'services'
Add platform root to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/VCCI-Scope3-Platform"
```

### Agent not available
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Format not detected
Specify format explicitly:
```bash
python -m cli.main intake file --file data.txt --format csv
```

## Performance

### Optimize for Speed
```bash
# Disable LLM
python -m cli.main calculate --category 1 --input data.csv --no-llm

# Disable Monte Carlo
python -m cli.main calculate --category 1 --input data.csv --no-mc

# Both
python -m cli.main pipeline run --input data/ --output results/ --no-llm --no-mc
```

### Batch Processing
```bash
# More efficient than individual files
python -m cli.main intake batch --directory data/
```

## Future Enhancements

- [ ] Shell completion (bash/zsh)
- [ ] Configuration file support (.vcci.yaml)
- [ ] Interactive TUI mode (textual)
- [ ] Watch mode for auto-reload
- [ ] Docker support
- [ ] Web dashboard integration
- [ ] Export to Excel/CSV
- [ ] Email/Slack notifications
- [ ] Plugin system
- [ ] Scheduled tasks

## Contributing

When adding new commands:
1. Follow existing patterns in `intake.py`, `engage.py`, `pipeline.py`
2. Use Rich for all UI components
3. Include comprehensive error handling
4. Add detailed docstrings with examples
5. Update documentation

## Support

- Check `--help` for any command
- See CLI_QUICK_REFERENCE.md for common patterns
- Review CLI_COMMANDS_SUMMARY.md for detailed information

## License

Part of GL-VCCI Scope 3 Carbon Intelligence Platform

---

**Built with:** Typer + Rich
**Python:** 3.9+
**Status:** Production Ready ✓
