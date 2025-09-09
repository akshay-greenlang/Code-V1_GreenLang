# GreenLang Commands Reference Guide

## Important: Command Usage

All commands must be prefixed with `greenlang`. You cannot run subcommands directly.

❌ **WRONG**: `agents`, `calc`, `benchmark`  
✅ **CORRECT**: `gl agents`, `gl calc`, `gl benchmark`

## Available Commands

### 1. Version & Help
```bash
# Show version
gl --version

# Show help
gl --help

# Show help for specific command
gl calc --help
gl analyze --help
```

### 2. Emissions Calculator
```bash
# Simple calculator (interactive)
gl calc

# Commercial building calculator
gl calc --building

# Building calculator with country
gl calc --building --country IN

# Load from file
gl calc --building --input building.json --output results.json
```

### 3. Building Analysis
```bash
# Analyze building from JSON file
gl analyze building_data.json

# Analyze with specific country
gl analyze building_data.json --country US
```

### 4. Benchmarking
```bash
# View benchmark for building type and country
gl benchmark --type hospital --country IN

# List all available benchmarks
gl benchmark --list

# Examples for different countries
gl benchmark --type commercial_office --country US
gl benchmark --type data_center --country EU
gl benchmark --type retail --country CN
```

### 5. Recommendations
```bash
# Interactive recommendation generator
gl recommend
```

### 6. Agent Management
```bash
# List all available agents (10 total)
gl agents

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
gl ask

# Ask a direct question
gl ask "What is the carbon footprint of a 100,000 sqft hospital in Mumbai?"

# Ask with verbose output
gl ask -v "Calculate emissions for 50000 sqft office with 1.5M kWh"
```

### 8. Workflow Execution
```bash
# Run a workflow
gl run workflow.yaml

# Run with input data
gl run workflow.yaml --input data.json

# Run with output file
gl run workflow.yaml --input data.json --output results.json

# Run with specific format
gl run workflow.yaml -i data.json -o results.json --format json
```

### 9. Project Initialization
```bash
# Create sample workflow
gl init

# Create with custom filename
gl init --output my_workflow.yaml
```

### 10. Developer Interface
```bash
# Launch developer UI
gl dev
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
**Solution**: Use `gl agents` instead of just `agents`

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
gl calc
# Enter electricity: 50000 kWh
# Enter natural gas: 1000 therms
# Enter diesel: 100 gallons
```

### Example 2: Building Analysis for India
```bash
gl calc --building --country IN
# Select building type: hospital
# Enter area: 100000 sqft
# Enter occupancy: 500
# Enter electricity: 3500000 kWh
# Enter diesel: 50000 liters
```

### Example 3: Check Benchmarks
```bash
gl benchmark --type hospital --country IN
gl benchmark --type data_center --country US
```

### Example 4: Get Recommendations
```bash
gl recommend
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