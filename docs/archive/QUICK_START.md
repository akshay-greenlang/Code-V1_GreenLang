# GreenLang Quick Start Guide

## Installation (Windows)

1. **Open Command Prompt or Terminal**
2. **Navigate to GreenLang folder:**
```bash
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
```

3. **Install GreenLang:**
```bash
pip install -e .
```

## Running GreenLang

### Method 1: Using the Batch File (Recommended for Windows)
```bash
# From the GreenLang directory:
greenlang [command]
# Or: ./greenlang.bat [command]
```

### Method 2: Using Python Module
```bash
python -m greenlang.cli.main [command]
```

### Method 3: If Python Scripts is in PATH
```bash
# Add to PATH first (one-time setup):
# Add C:\Users\rshar\AppData\Roaming\Python\Python313\Scripts to your PATH

greenlang [command]
```

## Essential Commands

| Command | Description | Example |
|---------|-------------|---------|
| `--version` | Show version | `./greenlang.bat --version` |
| `--help` | Show help | `./greenlang.bat --help` |
| `calc` | Interactive calculator | `./greenlang.bat calc` |
| `dev` | Developer interface | `./greenlang.bat dev` |
| `ask` | AI assistant | `./greenlang.bat ask "Calculate 1000 kWh"` |
| `init` | Create sample files | `./greenlang.bat init` |
| `agents` | List agents | `./greenlang.bat agents` |

## Quick Examples

### 1. Check Version
```bash
./greenlang.bat --version
```

### 2. Start Interactive Calculator
```bash
./greenlang.bat calc
# Enter values when prompted:
# Electricity: 1000 (kWh)
# Natural gas: 500 (therms)
# Diesel: 0 (gallons)
```

### 3. Use AI Assistant
```bash
./greenlang.bat ask "What is the carbon footprint of 5000 kWh electricity?"
```

### 4. Launch Developer Interface
```bash
./greenlang.bat dev
# Then type 'help' for available commands
```

### 5. Initialize a Project
```bash
./greenlang.bat init
# Creates workflow.yaml and workflow_input.json
```

## Troubleshooting

### If "greenlang" command not found:
Use the full command:
```bash
python -m greenlang.cli.main [command]
```

### If batch file doesn't work:
Run directly with Python:
```bash
python -m greenlang.cli.main calc
python -m greenlang.cli.main dev
python -m greenlang.cli.main --version
```

## Full Documentation
See `GREENLANG_DOCUMENTATION.md` for complete documentation.