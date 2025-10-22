# Windows Installation Guide for GreenLang CLI

Complete guide for installing and configuring GreenLang CLI on Windows 10/11.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
- [PATH Configuration](#path-configuration)
- [Troubleshooting](#troubleshooting)
- [Verification](#verification)

---

## Prerequisites

### Required

- **Windows 10** or **Windows 11**
- **Python 3.10+** (3.10, 3.11, or 3.12 recommended)
  - Download from [python.org](https://www.python.org/downloads/)
  - Or install via Microsoft Store
  - Or use Anaconda/Miniconda

### Optional

- **PowerShell 5.1+** or **PowerShell 7+**
- **Windows Terminal** (for better CLI experience)

---

## Installation Methods

### Method 1: pip install (Recommended)

```powershell
# Install from PyPI
pip install greenlang-cli

# Verify installation
python -m greenlang.cli --version
```

### Method 2: pip install with extras

```powershell
# Install with SBOM generation support
pip install "greenlang-cli[sbom]"

# Install with full supply chain tools
pip install "greenlang-cli[supply-chain]"

# Install all optional dependencies
pip install "greenlang-cli[full]"
```

### Method 3: Development installation

```powershell
# Clone repository
git clone https://github.com/greenlang/greenlang.git
cd greenlang

# Install in editable mode
pip install -e ".[dev,test]"
```

---

## PATH Configuration

After installation, you need to ensure the `gl` command is accessible from any directory.

### Automatic PATH Setup (Recommended)

Run the doctor command with --setup-path:

```powershell
gl doctor --setup-path
```

Or if `gl` is not recognized:

```powershell
python -m greenlang.cli doctor --setup-path
```

This will:
1. Locate your Python Scripts directory
2. Find the gl.exe executable
3. Add the Scripts directory to your User PATH
4. Create a backup of your current PATH
5. Notify Windows of the environment change

**Important**: After running this command, you must:
1. Close and reopen your PowerShell/Command Prompt
2. Or restart Windows Terminal
3. Then run `gl --version` to verify

### Manual PATH Setup

If automatic setup fails, add your Python Scripts directory manually:

#### Step 1: Find your Scripts directory

```powershell
python -c "import sys; from pathlib import Path; print(Path(sys.executable).parent / 'Scripts')"
```

This will output something like:
```
C:\Users\YourName\AppData\Local\Programs\Python\Python311\Scripts
```

#### Step 2: Add to PATH

**Option A: Using Windows Settings (GUI)**

1. Press `Win + X` and select **System**
2. Click **Advanced system settings**
3. Click **Environment Variables**
4. Under **User variables**, select **Path**
5. Click **Edit**
6. Click **New**
7. Paste your Scripts directory path
8. Click **OK** on all dialogs
9. Restart your terminal

**Option B: Using PowerShell**

```powershell
# Get your Scripts directory
$scriptsDir = python -c "import sys; from pathlib import Path; print(Path(sys.executable).parent / 'Scripts')"

# Add to User PATH
$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
[Environment]::SetEnvironmentVariable('Path', "$userPath;$scriptsDir", 'User')

# Notify system
$signature = '[DllImport("user32.dll")] public static extern IntPtr SendMessageTimeout(IntPtr hWnd, int Msg, IntPtr wParam, string lParam, int fuFlags, int uTimeout, out IntPtr lpdwResult);'
Add-Type -MemberDefinition $signature -Name 'Native' -Namespace 'Win32'
[Win32.Native]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [IntPtr]0, 'Environment', 0x0002, 5000, [ref]([IntPtr]::Zero))
```

#### Step 3: Verify

Close and reopen your terminal, then:

```powershell
gl --version
```

---

## PATH Management Commands

### View Current Status

```powershell
# Check installation and PATH status
gl doctor

# Verbose diagnostics
gl doctor --verbose
```

### Backup Management

```powershell
# List available PATH backups
gl doctor --list-backups

# Revert to previous PATH (requires confirmation)
gl doctor --revert-path
```

### Backup Location

PATH backups are stored in:
```
%USERPROFILE%\.greenlang\backup\
```

Each backup file is named with a timestamp:
```
path_20250122_143055.json
```

Backups are automatically created when:
- Running `gl doctor --setup-path`
- Modifying PATH programmatically

The system keeps the last 10 backups automatically.

---

## Troubleshooting

### Issue: "gl is not recognized as an internal or external command"

**Symptoms**: Running `gl` shows an error.

**Solutions**:

1. **Check if gl.exe exists:**
   ```powershell
   python -c "import sys; from pathlib import Path; print((Path(sys.executable).parent / 'Scripts' / 'gl.exe').exists())"
   ```

2. **Use Python module directly:**
   ```powershell
   python -m greenlang.cli --version
   ```

3. **Run automatic PATH setup:**
   ```powershell
   python -m greenlang.cli doctor --setup-path
   ```

4. **Check PATH manually:**
   ```powershell
   $env:PATH -split ';' | Select-String "Python"
   ```

### Issue: Changes don't take effect

**Symptoms**: PATH updated but `gl` still not found.

**Solutions**:

1. **Restart your terminal completely**
   - Close all PowerShell/Command Prompt windows
   - Open a new terminal
   - Test again

2. **Check if terminal inherited old environment**
   ```powershell
   # Check current PATH
   echo $env:PATH

   # Force reload environment (PowerShell)
   $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
   ```

3. **Restart Windows** (if all else fails)

### Issue: Multiple Python installations

**Symptoms**: `gl` command uses wrong Python version.

**Solutions**:

1. **Use Python launchers:**
   ```powershell
   # Use specific Python version
   py -3.11 -m greenlang.cli --version
   ```

2. **Use virtual environments:**
   ```powershell
   # Create venv
   python -m venv gl-env

   # Activate
   .\gl-env\Scripts\Activate.ps1

   # Install
   pip install greenlang-cli

   # Use
   gl --version
   ```

3. **Check which gl.exe is being used:**
   ```powershell
   where.exe gl
   ```

### Issue: Permission denied errors

**Symptoms**: Cannot modify PATH or create backups.

**Solutions**:

1. **Run PowerShell as Administrator:**
   - Right-click PowerShell
   - Select "Run as Administrator"
   - Retry installation

2. **Use User PATH instead of System PATH:**
   - `gl doctor --setup-path` only modifies User PATH
   - Does not require administrator privileges

3. **Check antivirus/security software:**
   - Some security tools block PATH modifications
   - Temporarily disable or whitelist Python

### Issue: Import errors after installation

**Symptoms**: `ModuleNotFoundError` or `ImportError`.

**Solutions**:

1. **Verify installation:**
   ```powershell
   pip show greenlang-cli
   pip list | Select-String greenlang
   ```

2. **Reinstall:**
   ```powershell
   pip uninstall greenlang-cli -y
   pip install greenlang-cli --no-cache-dir
   ```

3. **Check for conflicting packages:**
   ```powershell
   pip check
   ```

---

## Verification

### Step 1: Basic Check

```powershell
# Check version
gl --version

# Expected output:
# GreenLang v0.3.0
```

### Step 2: Full Diagnostics

```powershell
# Run comprehensive check
gl doctor --verbose

# Expected output should show:
# [OK] GreenLang Version
# [OK] Python Version
# [OK] gl.exe found
# [OK] gl.exe is in PATH
```

### Step 3: Test Commands

```powershell
# List available commands
gl --help

# Test pack management
gl pack list

# Test SBOM generation
gl sbom --help

# Test doctor with PATH operations
gl doctor --list-backups
```

---

## Python Installation Scenarios

### Scenario A: python.org Installer

**Scripts Location**: `C:\Users\<You>\AppData\Local\Programs\Python\Python3XX\Scripts\`

**Installation**:
```powershell
pip install greenlang-cli
gl doctor --setup-path
```

### Scenario B: Microsoft Store Python

**Scripts Location**: `C:\Users\<You>\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.XX_***\LocalCache\local-packages\Python3XX\Scripts\`

**Installation**:
```powershell
pip install greenlang-cli
gl doctor --setup-path  # Automatic detection
```

### Scenario C: Anaconda/Miniconda

**Scripts Location**: `C:\Users\<You>\anaconda3\Scripts\` or `C:\Users\<You>\miniconda3\Scripts\`

**Installation**:
```powershell
# Activate base or specific environment
conda activate base

# Install
pip install greenlang-cli

# Setup PATH for current environment
gl doctor --setup-path
```

### Scenario D: Virtual Environment (venv)

**Scripts Location**: `<venv_path>\Scripts\`

**Installation**:
```powershell
# Create venv
python -m venv myenv

# Activate
.\myenv\Scripts\Activate.ps1

# Install
pip install greenlang-cli

# gl command available within venv automatically
gl --version
```

---

## Advanced Configuration

### Execution Policies

If you encounter "script execution disabled" errors:

```powershell
# Check current policy
Get-ExecutionPolicy

# Allow scripts for current user
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Environment Variables

Set optional GreenLang environment variables:

```powershell
# Set permanently
[Environment]::SetEnvironmentVariable('GL_PROFILE', 'production', 'User')
[Environment]::SetEnvironmentVariable('GL_REGION', 'us-west-2', 'User')

# Set for current session only
$env:GL_PROFILE = "production"
$env:GL_REGION = "us-west-2"
```

### Windows Terminal Configuration

Add GreenLang to Windows Terminal:

```json
{
  "name": "GreenLang",
  "commandline": "powershell.exe -NoExit -Command \"& {gl doctor --verbose}\"",
  "icon": "path/to/icon.png"
}
```

---

## Uninstallation

### Step 1: Revert PATH Changes

```powershell
# Restore previous PATH from backup
gl doctor --revert-path
```

### Step 2: Uninstall Package

```powershell
# Uninstall greenlang-cli
pip uninstall greenlang-cli -y

# Remove configuration (optional)
Remove-Item -Recurse -Force "$env:USERPROFILE\.greenlang"
```

---

## Support

### Get Help

- **Documentation**: https://greenlang.io/docs
- **Issues**: https://github.com/greenlang/greenlang/issues
- **Discord**: https://discord.gg/greenlang

### Diagnostic Information

When reporting issues, include:

```powershell
# Collect diagnostic info
gl doctor --verbose > diagnostics.txt

# Include system info
python --version
pip --version
$PSVersionTable
```

---

## FAQ

**Q: Do I need administrator privileges?**

A: No. `gl doctor --setup-path` modifies only the User PATH, which doesn't require admin rights.

**Q: Will this affect other Python installations?**

A: No. PATH changes are specific to the Python installation you used for `pip install`.

**Q: Can I use gl in virtual environments?**

A: Yes. When you activate a venv, the `gl` command automatically uses that environment's installation.

**Q: What if I have multiple Python versions?**

A: Use Python launchers (`py -3.11 -m greenlang.cli`) or create separate virtual environments for each version.

**Q: How do I update GreenLang?**

A:
```powershell
pip install --upgrade greenlang-cli
```

**Q: Can I automate installation in scripts?**

A: Yes. Use:
```powershell
pip install greenlang-cli && python -m greenlang.cli doctor --setup-path
```

---

## Version History

- **v0.3.0** (2025-01-24): Added automatic PATH setup, backup/restore, Windows diagnostics
- **v0.2.3**: Improved Windows compatibility
- **v0.2.0**: First Windows release

---

*Last updated: January 22, 2025*
