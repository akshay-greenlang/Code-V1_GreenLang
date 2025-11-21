#!/usr/bin/env python3
"""
Update dependencies to latest secure versions
GreenLang Security Tool - Dependency Updater
"""

import subprocess
import sys
import json
import re
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path

# Security-critical packages that require extra attention
SECURITY_CRITICAL = [
    'cryptography', 'pyjwt', 'requests', 'httpx', 'pyyaml',
    'lxml', 'jinja2', 'sqlalchemy', 'psycopg2-binary', 'redis',
    'werkzeug', 'django', 'flask', 'paramiko', 'pillow'
]

# Known security fixes (minimum versions)
SECURITY_MINIMUMS = {
    'cryptography': '46.0.3',
    'pyjwt': '2.8.0',
    'requests': '2.31.0',
    'httpx': '0.26.0',
    'pyyaml': '6.0.1',
    'lxml': '5.1.0',
    'jinja2': '3.1.3',
    'sqlalchemy': '2.0.25',
    'werkzeug': '3.0.1',
    'pillow': '10.2.0'
}

def run_command(cmd: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, stderr."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def get_outdated_packages() -> List[Dict[str, str]]:
    """Get list of outdated packages."""
    code, stdout, _ = run_command([sys.executable, "-m", "pip", "list", "--outdated", "--format=json"])
    if code == 0:
        return json.loads(stdout)
    return []

def check_security_advisory(package: str) -> List[str]:
    """Check for security advisories for a package."""
    advisories = []

    # Try pip-audit for security check
    try:
        code, stdout, _ = run_command([
            sys.executable, "-m", "pip_audit", "--format", "json",
            "--requirement", "/dev/stdin"
        ])
        if code == 0:
            data = json.loads(stdout)
            for vuln in data.get('vulnerabilities', []):
                if vuln.get('package') == package:
                    advisories.append(vuln.get('description', 'Security vulnerability found'))
    except:
        pass

    return advisories

def update_package(package: str, version: str = None, dry_run: bool = False) -> bool:
    """Update a single package."""
    if version:
        package_spec = f"{package}=={version}"
    else:
        package_spec = f"{package}"

    if dry_run:
        print(f"  [DRY RUN] Would update: {package_spec}")
        return True

    print(f"  Updating {package_spec}...")
    code, stdout, stderr = run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", package_spec
    ])

    if code == 0:
        print(f"  ✅ Successfully updated {package}")
        return True
    else:
        print(f"  ❌ Failed to update {package}: {stderr}")
        return False

def update_requirements_file(filename: str, updates: Dict[str, str]):
    """Update a requirements file with new versions."""
    file_path = Path(filename)
    if not file_path.exists():
        return

    content = file_path.read_text()
    original_content = content

    for package, new_version in updates.items():
        # Match various patterns like package==version, package~=version, etc.
        patterns = [
            (f"{package}==[\d.]+", f"{package}=={new_version}"),
            (f"{package}~=[\d.]+", f"{package}=={new_version}"),
            (f"{package}>=[\d.]+", f"{package}=={new_version}"),
            (f"{package}<=[\d.]+", f"{package}=={new_version}"),
            (f"{package}>[\d.]+", f"{package}=={new_version}"),
            (f"{package}<[\d.]+", f"{package}=={new_version}"),
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    if content != original_content:
        # Backup original file
        backup_path = file_path.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak")
        file_path.rename(backup_path)
        print(f"  Backed up {filename} to {backup_path}")

        # Write updated content
        file_path.write_text(content)
        print(f"  Updated {filename}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Update GreenLang dependencies")
    parser.add_argument(
        "--security-only",
        action="store_true",
        help="Only update security-critical packages"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    parser.add_argument(
        "--update-files",
        action="store_true",
        help="Update requirements files with new versions"
    )
    parser.add_argument(
        "--check-advisories",
        action="store_true",
        help="Check for security advisories"
    )

    args = parser.parse_args()

    print("GreenLang Dependency Update Tool")
    print("=" * 50)
    print(f"Date: {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    print()

    # Get outdated packages
    print("Checking for outdated packages...")
    outdated = get_outdated_packages()

    if not outdated:
        print("✅ All packages are up to date!")
        return

    print(f"Found {len(outdated)} outdated packages")
    print()

    # Categorize packages
    security_updates = []
    other_updates = []

    for pkg in outdated:
        package_name = pkg['name'].lower()
        if package_name in SECURITY_CRITICAL:
            security_updates.append(pkg)
        else:
            other_updates.append(pkg)

    # Display security updates
    if security_updates:
        print("🔒 SECURITY-CRITICAL UPDATES:")
        for pkg in security_updates:
            name = pkg['name']
            current = pkg['version']
            latest = pkg['latest_version']

            # Check against minimum security versions
            min_version = SECURITY_MINIMUMS.get(name.lower())
            if min_version:
                print(f"  {name}: {current} → {latest} (minimum: {min_version})")
            else:
                print(f"  {name}: {current} → {latest}")

            # Check for advisories if requested
            if args.check_advisories:
                advisories = check_security_advisory(name)
                for advisory in advisories:
                    print(f"    ⚠️  {advisory}")
        print()

    # Display other updates
    if other_updates and not args.security_only:
        print("📦 OTHER UPDATES:")
        for pkg in other_updates[:10]:  # Show first 10
            print(f"  {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
        if len(other_updates) > 10:
            print(f"  ... and {len(other_updates) - 10} more")
        print()

    # Perform updates
    if not args.dry_run:
        response = input("Proceed with updates? (y/n/security-only): ").strip().lower()

        if response == 'n':
            print("Update cancelled.")
            return
        elif response == 'security-only' or response == 's':
            args.security_only = True

    # Update packages
    updates_to_apply = security_updates if args.security_only else outdated
    successful_updates = {}

    print("\nUpdating packages...")
    for pkg in updates_to_apply:
        name = pkg['name']
        version = pkg['latest_version']

        if update_package(name, version, args.dry_run):
            successful_updates[name] = version

    # Update requirements files
    if args.update_files and successful_updates and not args.dry_run:
        print("\nUpdating requirements files...")
        files_to_update = [
            "requirements.txt",
            "requirements-pinned.txt",
            "pyproject.toml"
        ]

        for filename in files_to_update:
            if Path(filename).exists():
                update_requirements_file(filename, successful_updates)

    # Regenerate pinned requirements
    if successful_updates and not args.dry_run:
        print("\nRegenerating pinned requirements...")
        script_path = Path(__file__).parent / "generate_pinned_requirements.py"
        if script_path.exists():
            run_command([sys.executable, str(script_path)])

    # Final report
    print("\n" + "=" * 50)
    print("UPDATE SUMMARY")
    print("=" * 50)

    if args.dry_run:
        print("DRY RUN - No actual changes made")
    else:
        print(f"Successfully updated: {len(successful_updates)} packages")

        if successful_updates:
            print("\nUpdated packages:")
            for name, version in successful_updates.items():
                print(f"  ✅ {name} → {version}")

    print("\nNext steps:")
    print("1. Run tests: pytest")
    print("2. Check for breaking changes in updated packages")
    print("3. Commit the updated requirements files")
    print("4. Deploy to staging environment first")

if __name__ == "__main__":
    main()