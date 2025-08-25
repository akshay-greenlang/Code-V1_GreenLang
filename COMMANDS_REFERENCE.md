# GreenLang Commands Reference Guide

## Important: Command Usage

All commands must be prefixed with `greenlang`. You cannot run subcommands directly.

❌ **WRONG**: `agents`, `calc`, `benchmark`  
✅ **CORRECT**: `greenlang agents`, `greenlang calc`, `greenlang benchmark`

## Available Commands

### 1. Version & Help
```bash
# Show version
greenlang --version

# Show help
greenlang --help

# Show help for specific command
greenlang calc --help
greenlang analyze --help
```

### 2. Emissions Calculator
```bash
# Simple calculator (interactive)
greenlang calc

# Commercial building calculator
greenlang calc --building

# Building calculator with country
greenlang calc --building --country IN

# Load from file
greenlang calc --building --input building.json --output results.json
```

### 3. Building Analysis
```bash
# Analyze building from JSON file
greenlang analyze building_data.json

# Analyze with specific country
greenlang analyze building_data.json --country US
```

### 4. Benchmarking
```bash
# View benchmark for building type and country
greenlang benchmark --type hospital --country IN

# List all available benchmarks
greenlang benchmark --list

# Examples for different countries
greenlang benchmark --type commercial_office --country US
greenlang benchmark --type data_center --country EU
greenlang benchmark --type retail --country CN
```

### 5. Recommendations
```bash
# Interactive recommendation generator
greenlang recommend
```

### 6. Agent Management
```bash
# List all available agents (10 total)
greenlang agents

# Show details about specific agent
greenlang agent validator      # Input validation agent
greenlang agent fuel          # Fuel emissions calculator
greenlang agent carbon        # Carbon aggregation agent
greenlang agent report        # Report generation agent
greenlang agent benchmark     # Benchmarking agent
greenlang agent grid_factor   # Grid emission factors
greenlang agent building_profile  # Building profiling
greenlang agent intensity     # Intensity metrics calculator
greenlang agent recommendation  # Recommendations engine
greenlang agent boiler        # Boiler emissions calculator
```

### 7. AI Assistant
```bash
# Ask a question interactively
greenlang ask

# Ask a direct question
greenlang ask "What is the carbon footprint of a 100,000 sqft hospital in Mumbai?"

# Ask with verbose output
greenlang ask -v "Calculate emissions for 50000 sqft office with 1.5M kWh"
```

### 8. Workflow Execution
```bash
# Run a workflow
greenlang run workflow.yaml

# Run with input data
greenlang run workflow.yaml --input data.json

# Run with output file
greenlang run workflow.yaml --input data.json --output results.json

# Run with specific format
greenlang run workflow.yaml -i data.json -o results.json --format json
```

### 9. Project Initialization
```bash
# Create sample workflow
greenlang init

# Create with custom filename
greenlang init --output my_workflow.yaml
```

### 10. Developer Interface
```bash
# Launch developer UI
greenlang dev
```

## Windows Batch File Usage

If you're using the `greenlang.bat` file:

```batch
REM All commands work the same way
greenlang.bat --version
greenlang.bat calc --building
greenlang.bat agents
```

## Python Module Direct Execution

You can also run directly as a Python module:

```bash
python -m greenlang.cli.main --version
python -m greenlang.cli.main calc --building
python -m greenlang.cli.main agents
```

## Common Issues & Solutions

### Issue 1: Command not recognized
**Error**: `'agents' is not recognized as an internal or external command`  
**Solution**: Use `greenlang agents` instead of just `agents`

### Issue 2: Module not found
**Error**: `ModuleNotFoundError: No module named 'greenlang'`  
**Solution**: 
```bash
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
pip install -e .
```

### Issue 3: Pydantic warning
**Warning**: `UserWarning: Valid config keys have changed in V2`  
**Note**: This is just a warning and doesn't affect functionality. The code works fine.

## Quick Examples

### Example 1: Simple Emissions Calculation
```bash
greenlang calc
# Enter electricity: 50000 kWh
# Enter natural gas: 1000 therms
# Enter diesel: 100 gallons
```

### Example 2: Building Analysis for India
```bash
greenlang calc --building --country IN
# Select building type: hospital
# Enter area: 100000 sqft
# Enter occupancy: 500
# Enter electricity: 3500000 kWh
# Enter diesel: 50000 liters
```

### Example 3: Check Benchmarks
```bash
greenlang benchmark --type hospital --country IN
greenlang benchmark --type data_center --country US
```

### Example 4: Get Recommendations
```bash
greenlang recommend
# Building type: commercial_office
# Country: US
# Building age: 20
# Performance: Average
```

## Environment Setup

Before using commands, ensure you're in the right directory:

```bash
cd "C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang"
```

Or add to PATH for global access:
```bash
set PATH=%PATH%;C:\Users\rshar\Desktop\Akshay Makar\Tools\GreenLang\Code V1_GreenLang
```

## Testing All Commands

Run the test script to verify all commands work:
```bash
test_all_commands.bat
```

## Running Example Tests

GreenLang includes 30 comprehensive example tests demonstrating various features:

### Run all example tests
```bash
pytest -m example
```

### Run specific example categories
```bash
# Core agent examples (1-6)
pytest examples/tests/ex_0[1-6]*.py

# Advanced features (7-18)
pytest examples/tests/ex_0[7-9]*.py examples/tests/ex_1[0-8]*.py

# Property tests and patterns (19-27)
pytest examples/tests/ex_1[9]*.py examples/tests/ex_2[0-7]*.py

# Tutorials (28-30)
pytest examples/tests/ex_2[8-9]*.py examples/tests/ex_30*.py
```

### Run with minimal output
```bash
pytest -m example -q
```

---

**Remember**: Always prefix commands with `greenlang`!