# -*- coding: utf-8 -*-
"""
Windows PATH Configuration Utilities for GreenLang CLI

This module provides utilities to automatically configure Windows PATH
to ensure the 'gl' command is available after pip installation.
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from greenlang.utilities.determinism import DeterministicClock

# Import winreg only on Windows
if sys.platform == "win32":
    import winreg
else:
    winreg = None

# Backup directory
BACKUP_DIR = Path.home() / ".greenlang" / "backup"


def get_python_scripts_directories() -> List[Path]:
    """
    Get all possible Python Scripts directories for the current installation.

    Returns:
        List of Path objects pointing to Scripts directories
    """
    scripts_dirs = []

    # Current Python executable location
    python_exe = Path(sys.executable)
    python_dir = python_exe.parent

    # Add Scripts directory for current Python
    scripts_dir = python_dir / "Scripts"
    if scripts_dir.exists():
        scripts_dirs.append(scripts_dir)

    # For user installations, also check the user Scripts directory
    if "--user" in sys.argv or is_user_install():
        import site
        user_base = Path(site.getusersitepackages()).parent
        user_scripts = user_base / "Scripts"
        if user_scripts.exists() and user_scripts not in scripts_dirs:
            scripts_dirs.append(user_scripts)

    return scripts_dirs


def is_user_install() -> bool:
    """Check if this is a user installation (--user flag or user site)."""
    import site
    try:
        # Check if we're in a user site-packages
        user_site = Path(site.getusersitepackages())
        for path in sys.path:
            if user_site.as_posix() in Path(path).as_posix():
                return True
    except (AttributeError, TypeError):
        pass
    return False


def is_in_path(directory: Path) -> bool:
    """Check if a directory is in the system PATH."""
    path_env = os.environ.get("PATH", "")
    path_dirs = [Path(p) for p in path_env.split(os.pathsep) if p]
    return directory in path_dirs


def get_user_path() -> str:
    """Get the current user PATH from Windows registry."""
    if sys.platform != "win32" or winreg is None:
        return ""

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            path, _ = winreg.QueryValueEx(key, "PATH")
            return path
    except (FileNotFoundError, OSError):
        return ""


def set_user_path(new_path: str) -> bool:
    """
    Set the user PATH in Windows registry.

    Args:
        new_path: The new PATH value

    Returns:
        True if successful, False otherwise
    """
    if sys.platform != "win32" or winreg is None:
        return False

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                           winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "PATH", 0, winreg.REG_EXPAND_SZ, new_path)

        # Notify system of environment change
        try:
            import win32gui
            import win32con
            win32gui.SendMessageTimeout(
                win32con.HWND_BROADCAST,
                win32con.WM_SETTINGCHANGE,
                0,
                "Environment",
                win32con.SMTO_ABORTIFHUNG,
                5000
            )
        except ImportError:
            # win32gui not available, use subprocess
            subprocess.run([
                "powershell", "-Command",
                "[System.Environment]::SetEnvironmentVariable('PATH', $env:PATH, 'User')"
            ], check=False, capture_output=True)

        return True
    except (OSError, PermissionError):
        return False


def add_to_user_path(directory: Path) -> bool:
    """
    Add a directory to the user PATH if it's not already there.

    Args:
        directory: Directory to add to PATH

    Returns:
        True if successful or already in PATH, False otherwise
    """
    if is_in_path(directory):
        return True

    # Backup current PATH before modifying
    backup_user_path()

    current_path = get_user_path()

    # Handle empty PATH
    if not current_path:
        new_path = str(directory)
    else:
        # Add to the beginning for priority
        new_path = f"{directory}{os.pathsep}{current_path}"

    return set_user_path(new_path)


def remove_from_user_path(directory: Path) -> bool:
    """
    Remove a directory from the user PATH.

    Args:
        directory: Directory to remove from PATH

    Returns:
        True if successful or not in PATH, False otherwise
    """
    if not is_in_path(directory):
        return True

    # Backup current PATH before modifying
    backup_user_path()

    current_path = get_user_path()
    if not current_path:
        return True

    # Split PATH and remove the specified directory
    path_entries = current_path.split(os.pathsep)
    directory_str = str(directory)

    # Remove all instances (case-insensitive on Windows)
    new_entries = [
        entry for entry in path_entries
        if Path(entry).resolve() != directory.resolve()
    ]

    # Reconstruct PATH
    new_path = os.pathsep.join(new_entries)

    return set_user_path(new_path)


def backup_user_path() -> Optional[Path]:
    """
    Backup the current user PATH to a JSON file.

    Returns:
        Path to the backup file if successful, None otherwise
    """
    if sys.platform != "win32":
        return None

    try:
        # Ensure backup directory exists
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)

        current_path = get_user_path()
        timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
        backup_file = BACKUP_DIR / f"path_{timestamp}.json"

        backup_data = {
            "timestamp": DeterministicClock.now().isoformat(),
            "path": current_path,
            "entries": current_path.split(os.pathsep) if current_path else []
        }

        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

        # Keep only the last 10 backups
        cleanup_old_backups(max_backups=10)

        return backup_file

    except (OSError, PermissionError, IOError):
        return None


def cleanup_old_backups(max_backups: int = 10) -> None:
    """
    Keep only the most recent PATH backups.

    Args:
        max_backups: Maximum number of backups to keep
    """
    try:
        if not BACKUP_DIR.exists():
            return

        backup_files = sorted(
            BACKUP_DIR.glob("path_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old backups beyond max_backups
        for backup_file in backup_files[max_backups:]:
            backup_file.unlink()

    except (OSError, PermissionError):
        pass


def list_path_backups() -> List[dict]:
    """
    List all available PATH backups.

    Returns:
        List of backup metadata dictionaries
    """
    backups = []

    try:
        if not BACKUP_DIR.exists():
            return backups

        for backup_file in sorted(BACKUP_DIR.glob("path_*.json"), reverse=True):
            try:
                with open(backup_file, 'r') as f:
                    data = json.load(f)
                    backups.append({
                        "file": str(backup_file),
                        "timestamp": data.get("timestamp"),
                        "entries_count": len(data.get("entries", []))
                    })
            except (json.JSONDecodeError, IOError):
                continue

    except OSError:
        pass

    return backups


def restore_path_from_backup(backup_file: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Restore user PATH from a backup file.

    Args:
        backup_file: Path to backup file. If None, uses most recent backup.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if sys.platform != "win32":
        return False, "This function is only for Windows systems"

    try:
        # If no backup file specified, use the most recent one
        if backup_file is None:
            backups = list_path_backups()
            if not backups:
                return False, "No PATH backups found"
            backup_file = Path(backups[0]["file"])

        if not backup_file.exists():
            return False, f"Backup file not found: {backup_file}"

        # Load backup
        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        old_path = backup_data.get("path", "")

        # Restore PATH
        if set_user_path(old_path):
            return True, f"Successfully restored PATH from backup: {backup_file.name}"
        else:
            return False, "Failed to restore PATH from backup"

    except (json.JSONDecodeError, IOError, OSError) as e:
        return False, f"Error restoring backup: {str(e)}"


def revert_windows_path() -> Tuple[bool, str]:
    """
    Revert Windows PATH to the most recent backup.

    Returns:
        Tuple of (success: bool, message: str)
    """
    return restore_path_from_backup()


def find_gl_executable() -> Optional[Path]:
    """Find the gl.exe executable in Python Scripts directories."""
    for scripts_dir in get_python_scripts_directories():
        gl_exe = scripts_dir / "gl.exe"
        if gl_exe.exists():
            return gl_exe
    return None


def setup_windows_path() -> Tuple[bool, str]:
    """
    Automatically configure Windows PATH for GreenLang CLI.

    Returns:
        Tuple of (success: bool, message: str)
    """
    if sys.platform != "win32":
        return False, "This function is only for Windows systems"

    # Find Python Scripts directories
    scripts_dirs = get_python_scripts_directories()

    if not scripts_dirs:
        return False, "No Python Scripts directories found"

    # Check if gl.exe exists in any Scripts directory
    gl_path = find_gl_executable()
    if not gl_path:
        return False, "gl.exe not found in any Python Scripts directory. Please reinstall GreenLang CLI."

    scripts_dir = gl_path.parent

    # Check if already in PATH
    if is_in_path(scripts_dir):
        return True, f"GreenLang CLI is already accessible via PATH: {scripts_dir}"

    # Try to add to user PATH
    if add_to_user_path(scripts_dir):
        return True, f"Successfully added {scripts_dir} to user PATH. Please restart your command prompt."
    else:
        return False, f"Failed to add {scripts_dir} to PATH. Please add it manually."


def diagnose_path_issues() -> dict:
    """
    Diagnose PATH-related issues for GreenLang CLI.

    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        "platform": sys.platform,
        "python_executable": sys.executable,
        "scripts_directories": [],
        "gl_executable_found": False,
        "gl_executable_path": None,
        "in_path": False,
        "path_entries": [],
        "recommendations": []
    }

    # Get Scripts directories
    scripts_dirs = get_python_scripts_directories()
    diagnosis["scripts_directories"] = [str(d) for d in scripts_dirs]

    # Check for gl.exe
    gl_path = find_gl_executable()
    if gl_path:
        diagnosis["gl_executable_found"] = True
        diagnosis["gl_executable_path"] = str(gl_path)
        diagnosis["in_path"] = is_in_path(gl_path.parent)

    # Get current PATH
    path_env = os.environ.get("PATH", "")
    diagnosis["path_entries"] = [p for p in path_env.split(os.pathsep) if p]

    # Generate recommendations
    if not diagnosis["gl_executable_found"]:
        diagnosis["recommendations"].append("Install GreenLang CLI: pip install greenlang-cli")
    elif not diagnosis["in_path"]:
        diagnosis["recommendations"].append(f"Add {gl_path.parent} to your PATH")
        diagnosis["recommendations"].append("Or run: gl doctor --setup-path")
    else:
        diagnosis["recommendations"].append("GreenLang CLI should be working correctly")

    return diagnosis


if __name__ == "__main__":
    # Quick test when run directly
    success, message = setup_windows_path()
    print(f"Setup result: {success}")
    print(f"Message: {message}")