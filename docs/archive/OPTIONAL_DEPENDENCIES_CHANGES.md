# Optional Dependencies Implementation Summary

This document summarizes the changes made to make pandas and numpy optional dependencies in GreenLang.

## Overview

Pandas and numpy have been moved from required dependencies to optional dependencies, allowing users to install GreenLang with only the features they need. This reduces the installation footprint and allows for more modular usage.

## Changes Made

### 1. `pyproject.toml` Updates

- **Removed** pandas and numpy from main `dependencies` section
- **Added** new `analytics` optional dependency group containing:
  - `pandas>=2.0.0`
  - `numpy>=1.24.0`
- **Updated** existing dependency groups:
  - `test` now includes `greenlang[analytics]`
  - `full` now includes `greenlang[analytics]`

### 2. Import Guards Added

The following files were updated with proper import guards:

#### Agent Files
- `greenlang/agents/energy_balance_agent.py`
- `greenlang/agents/load_profile_agent.py`
- `greenlang/agents/solar_resource_agent.py`

Added try/except blocks with helpful error messages directing users to install `greenlang[analytics]`.

#### SDK Files
- `greenlang/sdk/enhanced_client.py`

Added import guards for CSV and Excel export functionality.

#### Example and Application Files
- `examples/climatenza_demo.py`
- `apps/climatenza_app/examples/generate_dairy_load_data.py`

Added import guards and documentation about required dependencies.

#### Files That Already Had Guards
These files already had proper import protection:
- `greenlang/runtime/executor.py` (HAS_NUMPY flag)
- `greenlang/runtime/golden.py` (HAS_NUMPY flag)
- `core/greenlang/runtime/executor.py` (HAS_NUMPY flag)
- `core/greenlang/runtime/golden.py` (HAS_NUMPY flag)
- `greenlang/agents/boiler_agent.py` (improved error message)
- `greenlang/agents/fuel_agent.py` (improved error message)
- `examples/tutorials/custom_workflow_xlsx.py` (HAS_PANDAS flag)
- `examples/tests/ex_30_workflow_xlsx_tutorial.py` (HAS_PANDAS flag)

### 3. Documentation Updates

#### README.md
Added a comprehensive "Optional Dependencies" section explaining:
- How to install different feature sets
- Available extras and their purposes
- When each dependency group is needed

### 4. Test Coverage

Created `tests/unit/test_optional_dependencies.py` with comprehensive tests:
- Verification that import guards exist in all relevant modules
- Testing that error messages are properly formatted
- Validation that core functionality works without analytics dependencies
- Confirmation that agents work when analytics dependencies are installed

## Installation Options

### Core Installation (Minimal)
```bash
pip install greenlang
```
Installs only core dependencies without pandas/numpy.

### Analytics Features
```bash
pip install greenlang[analytics]
```
Adds pandas and numpy for data processing features.

### Full Installation
```bash
pip install greenlang[full]
```
Includes all production features including analytics.

### Development
```bash
pip install greenlang[dev]
```
Includes development tools and testing utilities.

## Error Handling

When analytics dependencies are missing, users see helpful error messages like:

```
ImportError: pandas is required for the EnergyBalanceAgent.
Install it with: pip install greenlang[analytics]
```

## Impact

### Benefits
1. **Reduced installation size** for users who don't need analytics features
2. **Faster installation** without heavy dependencies
3. **Modular architecture** allowing selective feature installation
4. **Clear error messages** guiding users to install needed dependencies

### Compatibility
- **Backward compatible**: Existing installations with `greenlang[full]` or `greenlang[all]` continue to work
- **Test compatibility**: Test suite includes analytics dependencies, so CI/CD continues to work
- **Development compatibility**: Development environments can still install all features

## Files Modified

### Configuration
- `pyproject.toml`

### Source Code
- `greenlang/agents/energy_balance_agent.py`
- `greenlang/agents/load_profile_agent.py`
- `greenlang/agents/solar_resource_agent.py`
- `greenlang/agents/boiler_agent.py` (error message improvement)
- `greenlang/agents/fuel_agent.py` (error message improvement)
- `greenlang/sdk/enhanced_client.py`
- `examples/climatenza_demo.py`
- `apps/climatenza_app/examples/generate_dairy_load_data.py`

### Documentation
- `README.md`

### Tests
- `tests/unit/test_optional_dependencies.py` (new file)

## Verification

All changes have been tested to ensure:
1. Core functionality works without analytics dependencies
2. Analytics features work when dependencies are installed
3. Error messages are helpful and actionable
4. Existing functionality is preserved
5. Test suite passes completely

This implementation provides a clean, modular approach to optional dependencies while maintaining full backward compatibility and providing clear guidance to users.