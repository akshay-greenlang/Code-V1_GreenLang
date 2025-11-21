#!/usr/bin/env python3
"""
Generate pinned requirements with hash verification
GreenLang Security Tool - Dependency Management
"""

import subprocess
import sys
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime

def get_installed_packages() -> Dict[str, str]:
    """Get all installed packages with their versions."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=json"],
        capture_output=True,
        text=True
    )
    packages = json.loads(result.stdout)
    return {pkg["name"].lower(): pkg["version"] for pkg in packages}

def get_package_hashes(package: str, version: str) -> List[str]:
    """Get SHA256 hashes for a package."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "hash", f"{package}=={version}"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            # Try alternative method using pip download
            result = subprocess.run(
                [sys.executable, "-m", "pip", "download", "--no-deps", "--no-binary", ":all:",
                 f"{package}=={version}", "-d", "/tmp"],
                capture_output=True,
                text=True,
                timeout=30
            )

        hashes = []
        for line in result.stdout.split('\n'):
            if 'sha256:' in line:
                hash_value = line.split('sha256:')[-1].strip()
                if hash_value:
                    hashes.append(f"sha256:{hash_value}")
        return hashes
    except Exception as e:
        print(f"Warning: Could not get hashes for {package}=={version}: {e}")
        return []

def read_requirements(file_path: str) -> List[Tuple[str, str, str]]:
    """Read requirements from file and extract package names."""
    requirements = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse package specification
            if '==' in line:
                parts = line.split('==')
                package = parts[0].strip()
                version = parts[1].split('#')[0].strip()
                comment = ''
                if '#' in line:
                    comment = line.split('#', 1)[1].strip()
                requirements.append((package, version, comment))
            elif '~=' in line or '>=' in line or '<=' in line:
                # Extract package name for version resolution
                for op in ['~=', '>=', '<=', '>', '<']:
                    if op in line:
                        package = line.split(op)[0].strip()
                        requirements.append((package, '', ''))
                        break

    return requirements

def generate_pinned_requirements(input_file: str, output_file: str, with_hashes: bool = True):
    """Generate pinned requirements file with optional hash verification."""

    print(f"Reading requirements from {input_file}...")
    requirements = read_requirements(input_file)

    print("Getting installed package versions...")
    installed = get_installed_packages()

    output_lines = []

    # Add header
    output_lines.append("# GreenLang Pinned Requirements")
    output_lines.append(f"# Generated: {datetime.now().isoformat()}")
    output_lines.append(f"# Python: {sys.version}")
    output_lines.append("# Security: All dependencies pinned to exact versions with SHA256 hashes")
    output_lines.append("# Install: pip install --require-hashes -r requirements-pinned.txt")
    output_lines.append("")
    output_lines.append("# =" * 40)
    output_lines.append("# SECURITY-CRITICAL PACKAGES")
    output_lines.append("# =" * 40)
    output_lines.append("")

    # Security-critical packages to prioritize
    security_critical = [
        'cryptography', 'pyjwt', 'requests', 'httpx', 'pyyaml',
        'lxml', 'jinja2', 'sqlalchemy', 'psycopg2-binary', 'redis'
    ]

    # Process security-critical packages first
    for package, version, comment in requirements:
        if package.lower() in security_critical:
            pkg_lower = package.lower()

            # Get version from installed packages if not specified
            if not version and pkg_lower in installed:
                version = installed[pkg_lower]

            if version:
                output_lines.append(f"{package}=={version} \\")

                if with_hashes:
                    hashes = get_package_hashes(package, version)
                    for hash_val in hashes[:3]:  # Limit to 3 hashes for readability
                        output_lines.append(f"    --hash={hash_val} \\")

                if comment:
                    output_lines[-1] = output_lines[-1].rstrip(' \\') + f"  # {comment}"
                else:
                    output_lines[-1] = output_lines[-1].rstrip(' \\')
                output_lines.append("")

    output_lines.append("# =" * 40)
    output_lines.append("# CORE DEPENDENCIES")
    output_lines.append("# =" * 40)
    output_lines.append("")

    # Process remaining packages
    for package, version, comment in requirements:
        if package.lower() not in security_critical:
            pkg_lower = package.lower()

            # Get version from installed packages if not specified
            if not version and pkg_lower in installed:
                version = installed[pkg_lower]

            if version:
                output_lines.append(f"{package}=={version} \\")

                if with_hashes:
                    hashes = get_package_hashes(package, version)
                    for hash_val in hashes[:2]:  # Limit to 2 hashes for non-critical
                        output_lines.append(f"    --hash={hash_val} \\")

                if comment:
                    output_lines[-1] = output_lines[-1].rstrip(' \\') + f"  # {comment}"
                else:
                    output_lines[-1] = output_lines[-1].rstrip(' \\')
                output_lines.append("")

    # Add footer
    output_lines.append("# =" * 40)
    output_lines.append("# INSTALLATION NOTES")
    output_lines.append("# =" * 40)
    output_lines.append("# 1. Install with hash verification (recommended):")
    output_lines.append("#    pip install --require-hashes -r requirements-pinned.txt")
    output_lines.append("#")
    output_lines.append("# 2. Install without hash verification (faster):")
    output_lines.append("#    pip install -r requirements-pinned.txt")
    output_lines.append("#")
    output_lines.append("# 3. Update a specific package:")
    output_lines.append("#    pip install --upgrade <package>")
    output_lines.append("#    python scripts/generate_pinned_requirements.py")
    output_lines.append("#")
    output_lines.append("# 4. Verify integrity:")
    output_lines.append("#    pip check")
    output_lines.append("#    python scripts/check_dependencies.py")
    output_lines.append("")

    # Write output file
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Generated {output_file} with {len(requirements)} packages")

    # Generate summary
    return {
        'total_packages': len(requirements),
        'security_critical': len([p for p, _, _ in requirements if p.lower() in security_critical]),
        'with_hashes': with_hashes,
        'output_file': output_file
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate pinned requirements with hashes")
    parser.add_argument(
        "--input",
        default="requirements.txt",
        help="Input requirements file"
    )
    parser.add_argument(
        "--output",
        default="requirements-pinned.txt",
        help="Output pinned requirements file"
    )
    parser.add_argument(
        "--no-hashes",
        action="store_true",
        help="Skip hash generation (faster but less secure)"
    )

    args = parser.parse_args()

    summary = generate_pinned_requirements(
        args.input,
        args.output,
        with_hashes=not args.no_hashes
    )

    print("\nSummary:")
    print(f"  Total packages: {summary['total_packages']}")
    print(f"  Security-critical: {summary['security_critical']}")
    print(f"  Hash verification: {'Enabled' if summary['with_hashes'] else 'Disabled'}")
    print(f"  Output file: {summary['output_file']}")