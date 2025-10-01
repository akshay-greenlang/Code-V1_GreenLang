# GreenLang Windows PATH Solution

## Overview

This document describes the comprehensive solution implemented to resolve the critical Windows PATH issue where the `gl` command is not available after `pip install` because Python's Scripts directory is not in PATH by default on Windows.

## Problem Statement

On Windows, when users install GreenLang CLI via `pip install greenlang-cli`, the `gl` command is not accessible because:

1. Python's Scripts directory (where `gl.exe` is installed) is not in the system PATH by default
2. Different Python installations (system, user, Anaconda, etc.) have Scripts directories in different locations
3. Users don't know how to fix PATH issues and assume the software is broken

## Comprehensive Solution

### 1. **Smart Windows Batch Wrapper** (`scripts/gl-wrapper.bat`)

A sophisticated batch file that:
- Auto-detects where `gl.exe` is installed across different Python environments
- Falls back to `python -m greenlang.cli` if direct execution fails
- Searches common Python installation paths:
  - User Python installations (`%USERPROFILE%\AppData\Roaming\Python\Python3*\Scripts`)
  - System Python installations (`C:\Python3*\Scripts`)
  - Anaconda/Miniconda installations
  - Local Python installations (`%LOCALAPPDATA%\Programs\Python`)
- Provides helpful error messages with troubleshooting steps

### 2. **Automatic PATH Configuration** (`greenlang/utils/windows_path.py`)

A Python module that:
- Automatically detects Python Scripts directories
- Adds the correct Scripts directory to the user PATH via Windows registry
- Works with all Python installation types (system, user, conda, virtual environments)
- Provides diagnostic functions to troubleshoot PATH issues
- Safely handles Windows registry operations with proper error handling

Key functions:
- `setup_windows_path()`: Automatically configures PATH
- `diagnose_path_issues()`: Comprehensive PATH diagnostics
- `find_gl_executable()`: Locates gl.exe across installations
- `add_to_user_path()`: Safely adds directories to user PATH

### 3. **Post-Install Script** (`greenlang/utils/post_install.py`)

Automatically runs after pip installation to:
- Set up Windows PATH configuration
- Create batch wrappers in appropriate locations
- Provide user feedback about setup status
- Handle edge cases gracefully

### 4. **Enhanced Setup Configuration**

Updated `setup.py` to include:
- Platform detection for Windows-specific setup
- Custom post-install command that runs PATH configuration
- Windows-specific scripts and data files
- Conditional installation of batch wrappers

Updated `pyproject.toml` to include:
- Windows utilities in package data
- Proper packaging of batch files and PowerShell scripts

### 5. **Enhanced CLI Self-Diagnosis** (`greenlang/cli/main.py`)

Enhanced the `gl doctor` command with:
- `--setup-path`: Automatically fix PATH issues
- `--verbose`: Detailed PATH diagnostics
- Windows-specific PATH checking
- Clear instructions for manual fixes
- Integration with Windows PATH utilities

### 6. **Universal PowerShell Installer** (`scripts/install-greenlang.ps1`)

A comprehensive PowerShell script that:
- Detects all Python installations on the system
- Handles system vs user installations
- Automatically configures PATH after installation
- Provides detailed feedback and error handling
- Works with one-line installation: `irm https://greenlang.io/install.ps1 | iex`

### 7. **Enhanced Quick Installer** (`scripts/quick-install.bat`)

Improved the existing quick installer to:
- Test the `gl` command after installation
- Automatically attempt PATH fixes
- Provide clear fallback instructions
- Handle installation failures gracefully

### 8. **Comprehensive Testing** (`scripts/test-windows-install.ps1`)

A thorough test suite that:
- Scans for all Python installations
- Tests PATH configuration
- Validates Windows utilities
- Tests different execution scenarios
- Provides detailed diagnostic reports

## Installation Scenarios Covered

The solution handles ALL Windows Python installation scenarios:

1. **System Python** (`C:\Python3*\`)
2. **User Python** (`%USERPROFILE%\AppData\Roaming\Python\`)
3. **Local Python** (`%LOCALAPPDATA%\Programs\Python\`)
4. **Anaconda/Miniconda** (various locations)
5. **Windows Store Python** (`%LOCALAPPDATA%\Microsoft\WindowsApps\`)
6. **Virtual Environments** (inherits from base Python)

## User Experience Improvements

### For New Users
1. **Automatic Setup**: PATH is configured during installation
2. **Smart Fallbacks**: Multiple ways to run the CLI if PATH fails
3. **Clear Instructions**: Helpful error messages with next steps

### For Existing Users
1. **Self-Repair**: `gl doctor --setup-path` fixes PATH issues
2. **Diagnosis**: `gl doctor --verbose` shows detailed PATH information
3. **Multiple Options**: Can use `python -m greenlang.cli` as fallback

### For Advanced Users
1. **PowerShell Installer**: Full control over installation process
2. **Batch Wrappers**: Custom wrappers for specific environments
3. **Manual Configuration**: Direct access to Windows utilities

## Key Files Added/Modified

### New Files
- `greenlang/utils/windows_path.py` - Windows PATH utilities
- `greenlang/utils/post_install.py` - Post-installation script
- `greenlang/utils/__init__.py` - Package initialization
- `scripts/gl-wrapper.bat` - Smart batch wrapper
- `scripts/install-greenlang.ps1` - Universal PowerShell installer
- `scripts/test-windows-install.ps1` - Comprehensive test suite

### Modified Files
- `setup.py` - Added Windows-specific setup and post-install command
- `pyproject.toml` - Added Windows utilities to package data
- `greenlang/cli/main.py` - Enhanced doctor command with PATH diagnosis
- `scripts/gl.bat` - Improved with smart fallbacks
- `scripts/quick-install.bat` - Enhanced with automatic PATH fixing

## Usage Examples

### For End Users
```bash
# Install via pip (automatic PATH setup)
pip install greenlang-cli

# If gl command doesn't work, fix PATH
gl doctor --setup-path

# Alternative execution methods
python -m greenlang.cli
scripts\gl-wrapper.bat
```

### For System Administrators
```powershell
# One-line installation with automatic PATH setup
irm https://greenlang.io/install.ps1 | iex

# Advanced installation with options
.\scripts\install-greenlang.ps1 -User -Force

# Test installation
.\scripts\test-windows-install.ps1 -Verbose
```

## Technical Implementation Details

### PATH Configuration Strategy
1. **User PATH**: Modifies user-level PATH to avoid requiring admin privileges
2. **Registry-based**: Uses Windows registry for persistent PATH changes
3. **Session Update**: Updates current session PATH for immediate availability
4. **Conflict Avoidance**: Checks existing PATH to avoid duplicates

### Error Handling
1. **Graceful Degradation**: Falls back to alternative execution methods
2. **Clear Messaging**: Provides actionable error messages
3. **Non-Breaking**: Installation succeeds even if PATH setup fails
4. **Recovery Options**: Multiple ways to fix issues post-installation

### Security Considerations
1. **User-Level Changes**: No administrator privileges required
2. **Registry Safety**: Proper error handling for registry operations
3. **Input Validation**: Safe handling of PATH components
4. **Minimal Permissions**: Uses least-privilege approach

## Testing and Validation

The solution includes comprehensive testing for:
- Multiple Python installation types
- Different user privilege levels
- Various Windows versions
- Edge cases and error conditions
- PATH configuration scenarios

## Future Considerations

1. **Automatic Updates**: Future versions can update PATH configuration
2. **Enterprise Deployment**: Group policy integration for large deployments
3. **Chocolatey/Scoop**: Integration with Windows package managers
4. **Windows Store**: Potential Windows Store distribution

## Impact

This solution ensures that **ALL Windows users** can use the `gl` command immediately after installation, regardless of their Python setup or technical expertise. It eliminates the most common barrier to adoption on Windows and provides a professional, user-friendly experience.

The solution is:
- **Comprehensive**: Covers all installation scenarios
- **Automatic**: Requires no user intervention
- **Safe**: Uses best practices for Windows configuration
- **Recoverable**: Provides multiple fallback options
- **Testable**: Includes thorough validation tools

This represents a complete solution to the Windows PATH problem that affects the entire Windows user base.