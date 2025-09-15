#!/usr/bin/env python3
"""Fetch OPA (Open Policy Agent) binary for the current platform."""

import os
import platform
import sys
import urllib.request
import zipfile
import tarfile
import stat
from pathlib import Path


def get_opa_download_url(version="0.64.0"):
    """Get the OPA download URL for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform to OPA naming convention
    if system == "windows":
        return f"https://github.com/open-policy-agent/opa/releases/download/v{version}/opa_windows_amd64.exe"
    elif system == "darwin":  # macOS
        if "arm" in machine or "aarch64" in machine:
            return f"https://github.com/open-policy-agent/opa/releases/download/v{version}/opa_darwin_arm64"
        else:
            return f"https://github.com/open-policy-agent/opa/releases/download/v{version}/opa_darwin_amd64"
    else:  # Linux and others
        if "arm" in machine or "aarch64" in machine:
            return f"https://github.com/open-policy-agent/opa/releases/download/v{version}/opa_linux_arm64"
        else:
            return f"https://github.com/open-policy-agent/opa/releases/download/v{version}/opa_linux_amd64"


def download_opa(version="0.64.0", target_dir=".tools"):
    """Download OPA binary for the current platform."""
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Determine binary name
    system = platform.system().lower()
    binary_name = "opa.exe" if system == "windows" else "opa"
    binary_path = target_path / binary_name

    # Check if already exists
    if binary_path.exists():
        print(f"OPA already exists at {binary_path}")
        return str(binary_path)

    # Get download URL
    url = get_opa_download_url(version)
    print(f"Downloading OPA {version} from {url}")

    # Download the file
    temp_file = target_path / "opa_download"
    try:
        urllib.request.urlretrieve(url, temp_file)
        print(f"Downloaded to {temp_file}")

        # Move to final location
        temp_file.rename(binary_path)

        # Make executable on Unix-like systems
        if system != "windows":
            st = binary_path.stat()
            binary_path.chmod(st.st_mode | stat.S_IEXEC)

        print(f"OPA installed successfully at {binary_path}")
        return str(binary_path)

    except Exception as e:
        print(f"Error downloading OPA: {e}")
        if temp_file.exists():
            temp_file.unlink()
        sys.exit(1)


def main():
    """Main entry point."""
    version = sys.argv[1] if len(sys.argv) > 1 else "0.64.0"

    # Download OPA
    opa_path = download_opa(version)

    # Verify it works
    import subprocess
    try:
        result = subprocess.run([opa_path, "version"], capture_output=True, text=True)
        print(f"OPA version output: {result.stdout}")
    except Exception as e:
        print(f"Warning: Could not verify OPA: {e}")

    # Add to PATH suggestion
    print("\nTo use OPA, either:")
    print(f"1. Add {Path(opa_path).parent.absolute()} to your PATH")
    print(f"2. Use the full path: {Path(opa_path).absolute()}")


if __name__ == "__main__":
    main()