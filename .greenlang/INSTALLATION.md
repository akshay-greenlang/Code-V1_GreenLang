# GreenLang Migration Toolkit - Installation Guide

## Prerequisites

Before using the Migration Toolkit, ensure you have:

### Required
- **Python 3.8+**: Download from [python.org](https://www.python.org/downloads/)
- **pip**: Usually comes with Python
- **Git**: For diff generation and version control

### Verify Installation

```bash
# Check Python version
python --version  # or python3 --version

# Check pip
pip --version  # or pip3 --version

# Check Git
git --version
```

## Installing Python Dependencies

### Core Dependencies

```bash
# Install required packages
pip install astor

# Optional: For enhanced features
pip install flask  # For dashboard server
```

### Dependency List

Create a `toolkit-requirements.txt`:

```text
# Core dependencies
astor>=0.8.1

# Optional but recommended
flask>=2.3.0

# For advanced features (optional)
matplotlib>=3.7.0
plotly>=5.14.0
```

Install all:
```bash
pip install -r .greenlang/toolkit-requirements.txt
```

## Verifying Installation

### 1. Check File Structure

Ensure all files are present:

```bash
.greenlang/
├── scripts/
│   ├── migrate_to_infrastructure.py
│   ├── rewrite_imports.py
│   ├── convert_to_base_agent.py
│   ├── update_dependencies.py
│   ├── generate_infrastructure_code.py
│   ├── generate_usage_report.py
│   ├── create_adr.py
│   └── serve_dashboard.py
├── cli/
│   └── greenlang.py
├── adr/
│   └── (ADRs will be created here)
├── MIGRATION_TOOLKIT_GUIDE.md
└── INSTALLATION.md
```

### 2. Test Individual Tools

```bash
# Test migration scanner
python .greenlang/scripts/migrate_to_infrastructure.py --help

# Test import rewriter
python .greenlang/scripts/rewrite_imports.py --help

# Test code generator
python .greenlang/scripts/generate_infrastructure_code.py --help

# Test CLI
python .greenlang/cli/greenlang.py --help
```

### 3. Run Sample Test

```bash
# Test on sample file
python .greenlang/scripts/migrate_to_infrastructure.py .greenlang/test_sample.py --dry-run
```

## Platform-Specific Setup

### Windows

```batch
REM Add to PATH (optional)
set PATH=%PATH%;C:\Users\aksha\Code-V1_GreenLang\.greenlang\cli

REM Create alias (PowerShell)
Set-Alias greenlang "C:\Users\aksha\Code-V1_GreenLang\.greenlang\cli\greenlang.py"

REM Run with python
python .greenlang\cli\greenlang.py --help
```

### Linux/Mac

```bash
# Make scripts executable
chmod +x .greenlang/cli/greenlang.py
chmod +x .greenlang/scripts/*.py

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=$PATH:/path/to/Code-V1_GreenLang/.greenlang/cli

# Create alias
alias greenlang='python /path/to/Code-V1_GreenLang/.greenlang/cli/greenlang.py'

# Run directly
.greenlang/cli/greenlang.py --help
```

## Troubleshooting

### Issue: "Python not found"

**Solution**:
- Windows: Install Python from python.org or Microsoft Store
- Add Python to PATH during installation
- Verify with `python --version`

### Issue: "ModuleNotFoundError: No module named 'astor'"

**Solution**:
```bash
pip install astor
```

### Issue: "Permission denied"

**Solution** (Linux/Mac):
```bash
chmod +x .greenlang/cli/greenlang.py
```

**Solution** (Windows):
```bash
# Run with python explicitly
python .greenlang\cli\greenlang.py --help
```

### Issue: Flask not available for dashboard

**Solution**:
```bash
pip install flask

# Dashboard will work without Flask but with limited features
```

## Next Steps

1. Read the [Migration Toolkit Guide](MIGRATION_TOOLKIT_GUIDE.md)
2. Run initial assessment:
   ```bash
   python .greenlang/cli/greenlang.py report --directory . --format html --output assessment.html
   ```
3. Start migration workflow (see guide)

## Quick Start Commands

```bash
# Scan for opportunities
python .greenlang/cli/greenlang.py migrate --app GL-CBAM-APP --dry-run

# Generate report
python .greenlang/cli/greenlang.py report --format html --output report.html

# Start dashboard
python .greenlang/cli/greenlang.py dashboard

# Generate new agent
python .greenlang/cli/greenlang.py generate --type agent --name MyAgent
```

## Support

- **Documentation**: See MIGRATION_TOOLKIT_GUIDE.md
- **Issues**: Check troubleshooting section
- **Testing**: Run tools with `--help` to see options
