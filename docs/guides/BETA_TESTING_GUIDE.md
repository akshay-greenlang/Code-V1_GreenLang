# GreenLang v0.2.0b2 Beta Testing Guide

## Installation Instructions

Install the beta from TestPyPI:
```bash
pip install -i https://test.pypi.org/simple --extra-index-url https://pypi.org/simple greenlang==0.2.0b2
```

## Quick Verification

After installation, verify everything works:
```bash
# Check version
gl --version

# View help
gl --help

# Run system check
gl doctor

# View pack commands
gl pack --help
```

## What to Test

1. **Basic CLI Commands**
   - Try all help commands
   - Test error handling
   - Check output formatting

2. **Pack Management**
   - `gl pack list`
   - `gl pack validate <path>`
   - `gl init pack-basic <name>`

3. **Cross-Platform Testing**
   - Windows 10/11
   - macOS (Intel/M1)
   - Linux (Ubuntu/Debian)

## Reporting Issues

Please report any issues with:
- Your OS and Python version
- Command that failed
- Full error message
- Steps to reproduce

Report at: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues

## Timeline

- **Beta Period**: 1-2 days
- **Focus**: Installation and basic CLI functionality
- **Next Release**: v0.2.0 final (after reaching 40% test coverage)

Thank you for testing GreenLang!