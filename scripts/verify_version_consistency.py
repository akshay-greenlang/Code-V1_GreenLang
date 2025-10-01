#!/usr/bin/env python3
"""
Version Consistency Verification Script for GreenLang
=====================================================

This script verifies that all version-related files in the GreenLang project
maintain consistency and follow semantic versioning standards.

Checks performed:
1. VERSION file contains valid semantic version
2. pyproject.toml has matching version
3. setup.py correctly reads from VERSION file
4. greenlang/_version.py correctly reads the version
5. greenlang package imports show correct __version__
6. All versions match across files

Exit codes:
- 0: All checks passed
- 1: Version inconsistency found
- 2: Invalid semantic version format
- 3: Missing required files
- 4: Import/runtime errors

Usage examples:
  # Run from project root (auto-detect)
  python scripts/verify_version_consistency.py

  # Specify project root explicitly
  python scripts/verify_version_consistency.py /path/to/project

  # Use in CI/CD pipeline
  python scripts/verify_version_consistency.py || exit 1

  # Get help
  python scripts/verify_version_consistency.py --help
"""

import sys
import re
import subprocess
import tempfile
import textwrap
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        # Try to set console to UTF-8 mode for better emoji support
        os.system("chcp 65001 > nul 2>&1")
    except:
        pass

# ANSI color codes for output formatting
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Unicode-safe symbols that work across platforms
class Symbols:
    CHECK = "✓" if sys.platform != "win32" else "[OK]"
    CROSS = "✗" if sys.platform != "win32" else "[ERROR]"
    WARNING = "⚠" if sys.platform != "win32" else "[WARN]"
    INFO = "ℹ" if sys.platform != "win32" else "[INFO]"
    SEARCH = "🔍" if sys.platform != "win32" else "[CHECKING]"

class VersionChecker:
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the version checker with project root directory."""
        if project_root is None:
            # Auto-detect project root by looking for VERSION file
            current_path = Path(__file__).resolve()
            for parent in [current_path.parent] + list(current_path.parents):
                if (parent / "VERSION").exists():
                    project_root = parent
                    break

            if project_root is None:
                raise RuntimeError("Could not find project root (no VERSION file found)")

        self.project_root = Path(project_root).resolve()
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # Define file paths
        self.version_file = self.project_root / "VERSION"
        self.pyproject_file = self.project_root / "pyproject.toml"
        self.setup_file = self.project_root / "setup.py"
        self.version_module = self.project_root / "greenlang" / "_version.py"
        self.init_module = self.project_root / "greenlang" / "__init__.py"

    def log_info(self, message: str) -> None:
        """Log an informational message."""
        print(f"{Colors.BLUE}{Symbols.INFO}  {message}{Colors.END}")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        print(f"{Colors.GREEN}{Symbols.CHECK} {message}{Colors.END}")

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        print(f"{Colors.YELLOW}{Symbols.WARNING}  {message}{Colors.END}")
        self.warnings.append(message)

    def log_error(self, message: str) -> None:
        """Log an error message."""
        print(f"{Colors.RED}{Symbols.CROSS} {message}{Colors.END}")
        self.errors.append(message)

    def validate_semantic_version(self, version: str) -> bool:
        """
        Validate that version follows semantic versioning (semver) format.

        Accepts formats like:
        - 1.0.0
        - 1.0.0-alpha
        - 1.0.0-beta.1
        - 1.0.0+build.1
        """
        # Basic semver regex pattern
        semver_pattern = r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$'

        if not re.match(semver_pattern, version):
            return False

        # Additional validation for reasonable version numbers
        try:
            major, minor, patch = version.split('-')[0].split('+')[0].split('.')
            major_int = int(major)
            minor_int = int(minor)
            patch_int = int(patch)

            # Check for reasonable bounds
            if major_int > 999 or minor_int > 999 or patch_int > 999:
                return False

            return True
        except (ValueError, IndexError):
            return False

    def read_version_file(self) -> Optional[str]:
        """Read and validate the VERSION file."""
        self.log_info("Checking VERSION file...")

        if not self.version_file.exists():
            self.log_error(f"VERSION file not found at {self.version_file}")
            return None

        try:
            version_content = self.version_file.read_text(encoding='utf-8').strip()

            if not version_content:
                self.log_error("VERSION file is empty")
                return None

            # Handle case where VERSION file might have extra whitespace or newlines
            lines = [line.strip() for line in version_content.splitlines() if line.strip()]
            if len(lines) != 1:
                self.log_warning(f"VERSION file contains {len(lines)} non-empty lines, expected 1")
                version_content = lines[0] if lines else ""

            if not self.validate_semantic_version(version_content):
                self.log_error(f"VERSION file contains invalid semantic version: '{version_content}'")
                return None

            self.log_success(f"VERSION file contains valid version: {version_content}")
            return version_content

        except Exception as e:
            self.log_error(f"Error reading VERSION file: {e}")
            return None

    def check_pyproject_version(self, expected_version: str) -> bool:
        """Check pyproject.toml version matches expected version."""
        self.log_info("Checking pyproject.toml version...")

        if not self.pyproject_file.exists():
            self.log_error(f"pyproject.toml not found at {self.pyproject_file}")
            return False

        try:
            content = self.pyproject_file.read_text(encoding='utf-8')

            # Look for version = "x.y.z" in [project] section
            version_pattern = r'version\s*=\s*["\']([^"\']+)["\']'
            matches = re.findall(version_pattern, content)

            if not matches:
                self.log_error("No version found in pyproject.toml")
                return False

            if len(matches) > 1:
                self.log_warning(f"Multiple version declarations found in pyproject.toml: {matches}")

            pyproject_version = matches[0]

            if pyproject_version != expected_version:
                self.log_error(f"pyproject.toml version '{pyproject_version}' != VERSION file '{expected_version}'")
                return False

            self.log_success(f"pyproject.toml version matches: {pyproject_version}")
            return True

        except Exception as e:
            self.log_error(f"Error reading pyproject.toml: {e}")
            return False

    def check_setup_py_version(self, expected_version: str) -> bool:
        """Check that setup.py correctly reads from VERSION file."""
        self.log_info("Checking setup.py version reading logic...")

        if not self.setup_file.exists():
            self.log_warning("setup.py not found (this is OK for modern Python packages)")
            return True

        try:
            content = self.setup_file.read_text(encoding='utf-8')

            # Check if setup.py reads from VERSION file
            if 'VERSION' not in content:
                self.log_warning("setup.py does not reference VERSION file")
                return True

            # Look for version assignment pattern
            if 'version_file.read_text' in content or 'read_text()' in content:
                self.log_success("setup.py correctly reads from VERSION file")

                # Try to extract the fallback version if any
                fallback_pattern = r'version\s*=\s*["\']([^"\']+)["\']'
                fallback_matches = re.findall(fallback_pattern, content)

                for fallback in fallback_matches:
                    if fallback != expected_version:
                        self.log_warning(f"setup.py has fallback version '{fallback}' != VERSION file '{expected_version}'")

                return True
            else:
                self.log_warning("setup.py VERSION file reading pattern not recognized")
                return True

        except Exception as e:
            self.log_error(f"Error reading setup.py: {e}")
            return False

    def check_version_module(self, expected_version: str) -> bool:
        """Check greenlang/_version.py implementation."""
        self.log_info("Checking greenlang/_version.py...")

        if not self.version_module.exists():
            self.log_error(f"Version module not found at {self.version_module}")
            return False

        try:
            content = self.version_module.read_text(encoding='utf-8')

            # Check for proper fallback to VERSION file
            if 'VERSION' not in content:
                self.log_warning("_version.py does not reference VERSION file for fallback")

            # Check for reasonable fallback version
            fallback_pattern = r'__version__\s*=\s*["\']([^"\']+)["\']'
            fallback_matches = re.findall(fallback_pattern, content)

            for fallback in fallback_matches:
                if fallback != expected_version:
                    self.log_warning(f"_version.py has fallback version '{fallback}' != VERSION file '{expected_version}'")

            self.log_success("_version.py structure looks correct")
            return True

        except Exception as e:
            self.log_error(f"Error reading _version.py: {e}")
            return False

    def test_version_import(self, expected_version: str) -> bool:
        """Test importing greenlang and checking __version__."""
        self.log_info("Testing greenlang.__version__ import...")

        # Create a temporary test script to avoid import issues
        test_script = textwrap.dedent(f'''
        import sys
        import os

        # Add project root to path
        project_root = r"{self.project_root}"
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            import greenlang
            version = getattr(greenlang, '__version__', None)

            if version is None:
                print("ERROR: greenlang.__version__ is None")
                sys.exit(1)

            print(f"FOUND_VERSION:{{version}}")

            expected = "{expected_version}"
            if version != expected:
                print(f"ERROR: Version mismatch - found '{{version}}', expected '{{expected}}'")
                sys.exit(1)

            print("SUCCESS: Version import test passed")
            sys.exit(0)

        except ImportError as e:
            print(f"ERROR: Failed to import greenlang: {{e}}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Unexpected error: {{e}}")
            sys.exit(1)
        ''')

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name

            # Run the test script
            result = subprocess.run(
                [sys.executable, temp_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            # Clean up
            Path(temp_script).unlink(missing_ok=True)

            if result.returncode == 0:
                # Extract the found version for confirmation
                for line in result.stdout.splitlines():
                    if line.startswith("FOUND_VERSION:"):
                        found_version = line.split(":", 1)[1]
                        self.log_success(f"greenlang.__version__ == '{found_version}'")
                        return True

                self.log_success("greenlang version import test passed")
                return True
            else:
                self.log_error(f"Version import test failed:")
                if result.stdout:
                    for line in result.stdout.strip().splitlines():
                        if line.startswith("ERROR:"):
                            self.log_error(f"  {line}")
                if result.stderr:
                    for line in result.stderr.strip().splitlines():
                        self.log_error(f"  stderr: {line}")
                return False

        except subprocess.TimeoutExpired:
            self.log_error("Version import test timed out")
            return False
        except Exception as e:
            self.log_error(f"Error running version import test: {e}")
            return False

    def check_init_module(self) -> bool:
        """Check that greenlang/__init__.py imports __version__ correctly."""
        self.log_info("Checking greenlang/__init__.py version import...")

        if not self.init_module.exists():
            self.log_error(f"Init module not found at {self.init_module}")
            return False

        try:
            content = self.init_module.read_text(encoding='utf-8')

            # Check for version import
            if 'from ._version import __version__' in content:
                self.log_success("__init__.py correctly imports __version__ from _version")
                return True
            elif '__version__' in content:
                self.log_warning("__init__.py references __version__ but import pattern not recognized")
                return True
            else:
                self.log_warning("__init__.py does not import __version__")
                return True

        except Exception as e:
            self.log_error(f"Error reading __init__.py: {e}")
            return False

    def run_all_checks(self) -> Tuple[bool, Dict[str, any]]:
        """Run all version consistency checks."""
        print(f"\n{Colors.BOLD}{Symbols.SEARCH} GreenLang Version Consistency Verification{Colors.END}")
        print(f"Project root: {self.project_root}")
        print("=" * 60)

        results = {
            'version_file': False,
            'pyproject_toml': False,
            'setup_py': False,
            'version_module': False,
            'init_module': False,
            'version_import': False,
            'version': None
        }

        # Step 1: Read and validate VERSION file
        version = self.read_version_file()
        if version is None:
            return False, results

        results['version'] = version
        results['version_file'] = True

        # Step 2: Check pyproject.toml
        results['pyproject_toml'] = self.check_pyproject_version(version)

        # Step 3: Check setup.py (optional)
        results['setup_py'] = self.check_setup_py_version(version)

        # Step 4: Check _version.py module
        results['version_module'] = self.check_version_module(version)

        # Step 5: Check __init__.py
        results['init_module'] = self.check_init_module()

        # Step 6: Test actual import
        results['version_import'] = self.test_version_import(version)

        # Summary
        print("\n" + "=" * 60)

        all_passed = all([
            results['version_file'],
            results['pyproject_toml'],
            results['setup_py'],
            results['version_module'],
            results['init_module'],
            results['version_import']
        ])

        if all_passed:
            self.log_success(f"All version consistency checks passed! Version: {version}")
        else:
            self.log_error("Some version consistency checks failed!")

        if self.warnings:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.END}")
            for warning in self.warnings:
                print(f"  {Symbols.WARNING}  {warning}")

        if self.errors:
            print(f"\n{Colors.RED}Errors:{Colors.END}")
            for error in self.errors:
                print(f"  {Symbols.CROSS} {error}")

        print()
        return all_passed, results

def main():
    """Main entry point for the version consistency checker."""
    try:
        # Handle help argument
        if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
            print(__doc__)
            print(f"\nUsage: {sys.argv[0]} [PROJECT_ROOT]")
            print("\nArguments:")
            print("  PROJECT_ROOT    Optional path to the project root directory")
            print("                  (auto-detected if not provided)")
            print("\nExit codes:")
            print("  0  All checks passed")
            print("  1  Version inconsistency found")
            print("  2  Invalid semantic version format")
            print("  3  Missing required files")
            print("  4  Import/runtime errors")
            sys.exit(0)

        # Allow specifying project root as command line argument
        project_root = None
        if len(sys.argv) > 1:
            project_root = Path(sys.argv[1])

        checker = VersionChecker(project_root)
        success, results = checker.run_all_checks()

        if success:
            print(f"{Colors.GREEN}{Colors.BOLD}{Symbols.CHECK} All version consistency checks passed!{Colors.END}")
            sys.exit(0)
        else:
            print(f"{Colors.RED}{Colors.BOLD}{Symbols.CROSS} Version consistency checks failed!{Colors.END}")
            if not results['version_file']:
                sys.exit(3)  # Missing/invalid files
            elif results['version'] and not checker.validate_semantic_version(results['version']):
                sys.exit(2)  # Invalid semantic version
            elif not results['version_import']:
                sys.exit(4)  # Import/runtime errors
            else:
                sys.exit(1)  # Version inconsistency

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}{Symbols.CROSS} Interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"{Colors.RED}{Symbols.CROSS} Unexpected error: {e}{Colors.END}")
        sys.exit(1)

if __name__ == '__main__':
    main()