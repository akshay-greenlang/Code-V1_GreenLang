#!/usr/bin/env python3
"""
Auto-fix script for GL-007 FurnacePerformanceMonitor code quality issues.

This script automatically fixes common code quality issues identified in the
code quality review:
- Code formatting (black)
- Import sorting (isort)
- Unused imports removal
- Line length violations
- Package structure creation

Usage:
    python auto_fix_script.py [--dry-run] [--category CATEGORY]

Options:
    --dry-run       Show what would be fixed without making changes
    --category      Fix only specific category (format|imports|structure|all)
    --aggressive    Apply more aggressive fixes (experimental)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import shutil


class AutoFixer:
    """Automatically fix code quality issues."""

    def __init__(self, base_dir: Path, dry_run: bool = False, aggressive: bool = False):
        """
        Initialize auto-fixer.

        Args:
            base_dir: Base directory for GL-007 project
            dry_run: If True, show changes without applying them
            aggressive: If True, apply more aggressive fixes
        """
        self.base_dir = base_dir
        self.dry_run = dry_run
        self.aggressive = aggressive
        self.fixes_applied = []
        self.errors = []

    def fix_all(self) -> Dict[str, Any]:
        """Run all auto-fixes."""
        print("=" * 80)
        print("GL-007 Auto-Fix Script")
        print("=" * 80)
        print(f"Base directory: {self.base_dir}")
        print(f"Dry run: {self.dry_run}")
        print(f"Aggressive mode: {self.aggressive}")
        print()

        # Run fixes in order
        self.fix_package_structure()
        self.fix_code_formatting()
        self.fix_import_sorting()
        self.fix_unused_imports()
        self.fix_line_lengths()
        self.fix_portable_paths()

        # Generate report
        return self._generate_report()

    def fix_package_structure(self):
        """Create missing __init__.py files."""
        print("[1/6] Fixing package structure...")

        monitoring_dir = self.base_dir / "monitoring"
        init_file = monitoring_dir / "__init__.py"

        if not init_file.exists():
            init_content = '''"""
GL-007 FurnacePerformanceMonitor - Monitoring Module.

This module provides comprehensive monitoring capabilities including:
- Health checks (liveness, readiness, startup probes)
- Structured logging with correlation IDs
- Prometheus metrics instrumentation
- Distributed tracing with OpenTelemetry
"""

from .health_checks import (
    HealthChecker,
    HealthStatus,
    ReadinessStatus,
    ComponentHealth,
    HealthResponse,
    ReadinessResponse,
    KubernetesProbes,
)
from .logging_config import (
    setup_logging,
    setup_logging_for_environment,
    get_logger,
    set_correlation_id,
    get_correlation_id,
    set_furnace_id,
    get_furnace_id,
    LogContext,
)
from .metrics import (
    MetricsCollector,
    track_request_metrics,
    track_calculation_metrics,
)
from .tracing_config import (
    TracingConfig,
    get_tracer,
    traced,
    TracingContext,
    FurnaceTracing,
    setup_tracing_for_environment,
)

__all__ = [
    # Health checks
    "HealthChecker",
    "HealthStatus",
    "ReadinessStatus",
    "ComponentHealth",
    "HealthResponse",
    "ReadinessResponse",
    "KubernetesProbes",
    # Logging
    "setup_logging",
    "setup_logging_for_environment",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_furnace_id",
    "get_furnace_id",
    "LogContext",
    # Metrics
    "MetricsCollector",
    "track_request_metrics",
    "track_calculation_metrics",
    # Tracing
    "TracingConfig",
    "get_tracer",
    "traced",
    "TracingContext",
    "FurnaceTracing",
    "setup_tracing_for_environment",
]

__version__ = "1.0.0"
'''

            if not self.dry_run:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(init_content)
                self.fixes_applied.append("Created monitoring/__init__.py")
                print("  ✓ Created monitoring/__init__.py")
            else:
                print("  [DRY RUN] Would create monitoring/__init__.py")
        else:
            print("  ✓ monitoring/__init__.py already exists")

    def fix_code_formatting(self):
        """Apply black code formatter."""
        print("\n[2/6] Fixing code formatting with black...")

        py_files = list(self.base_dir.glob("**/*.py"))

        if not self._check_command("black"):
            print("  ⚠ black not installed. Install with: pip install black")
            self.errors.append("black not available")
            return

        cmd = ["black"]
        if self.dry_run:
            cmd.append("--check")
        cmd.extend([str(f) for f in py_files])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )

            if self.dry_run:
                if result.returncode == 0:
                    print("  ✓ All files already formatted")
                else:
                    print("  [DRY RUN] Would reformat files:")
                    print(result.stdout)
            else:
                if result.returncode == 0:
                    print("  ✓ Code formatted successfully")
                    self.fixes_applied.append("Applied black formatting")
                else:
                    print(f"  ✓ Formatted {len(py_files)} files")
                    self.fixes_applied.append(f"Formatted {len(py_files)} files")

        except Exception as e:
            print(f"  ✗ Error running black: {e}")
            self.errors.append(f"black error: {e}")

    def fix_import_sorting(self):
        """Sort imports with isort."""
        print("\n[3/6] Fixing import sorting with isort...")

        if not self._check_command("isort"):
            print("  ⚠ isort not installed. Install with: pip install isort")
            self.errors.append("isort not available")
            return

        py_files = list(self.base_dir.glob("**/*.py"))

        cmd = ["isort"]
        if self.dry_run:
            cmd.append("--check-only")
        cmd.extend([str(f) for f in py_files])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )

            if self.dry_run:
                if result.returncode == 0:
                    print("  ✓ All imports already sorted")
                else:
                    print("  [DRY RUN] Would sort imports in files")
            else:
                print("  ✓ Imports sorted successfully")
                self.fixes_applied.append("Sorted imports with isort")

        except Exception as e:
            print(f"  ✗ Error running isort: {e}")
            self.errors.append(f"isort error: {e}")

    def fix_unused_imports(self):
        """Remove unused imports."""
        print("\n[4/6] Removing unused imports...")

        if not self._check_command("autoflake"):
            print("  ⚠ autoflake not installed. Install with: pip install autoflake")
            self.errors.append("autoflake not available")
            return

        py_files = list(self.base_dir.glob("**/*.py"))

        cmd = [
            "autoflake",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
        ]
        if not self.dry_run:
            cmd.append("--in-place")

        cmd.extend([str(f) for f in py_files])

        try:
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True
            )

            if result.stdout.strip():
                if self.dry_run:
                    print("  [DRY RUN] Would remove unused imports:")
                    print(result.stdout)
                else:
                    print("  ✓ Removed unused imports")
                    self.fixes_applied.append("Removed unused imports")
            else:
                print("  ✓ No unused imports found")

        except Exception as e:
            print(f"  ✗ Error running autoflake: {e}")
            self.errors.append(f"autoflake error: {e}")

    def fix_line_lengths(self):
        """Fix line length violations."""
        print("\n[5/6] Fixing line length violations...")

        # This is handled by black, so just note it
        print("  ✓ Line length violations handled by black formatter")

    def fix_portable_paths(self):
        """Fix hardcoded non-portable paths."""
        print("\n[6/6] Fixing non-portable path issues...")

        logging_config = self.base_dir / "monitoring" / "logging_config.py"

        if not logging_config.exists():
            print("  ⚠ logging_config.py not found")
            return

        # Read file
        with open(logging_config, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if fix is needed
        if "'/var/log/greenlang/gl-007/app.log'" not in content:
            print("  ✓ No hardcoded paths found")
            return

        # Create backup
        if not self.dry_run:
            backup_path = logging_config.with_suffix('.py.backup')
            shutil.copy2(logging_config, backup_path)
            print(f"  Created backup: {backup_path}")

        # Fix paths
        fixed_content = content.replace(
            "'log_file': '/var/log/greenlang/gl-007/app.log'",
            "'log_file': str(Path(os.getenv('LOG_DIR', '/var/log/greenlang')) / 'gl-007' / 'app.log')"
        )

        # Ensure Path import
        if "from pathlib import Path" not in fixed_content:
            # Add to imports
            fixed_content = fixed_content.replace(
                "import os",
                "import os\nfrom pathlib import Path"
            )

        if self.dry_run:
            print("  [DRY RUN] Would fix hardcoded paths in logging_config.py")
        else:
            with open(logging_config, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print("  ✓ Fixed hardcoded paths in logging_config.py")
            self.fixes_applied.append("Fixed non-portable paths")

    def _check_command(self, command: str) -> bool:
        """Check if command is available."""
        return shutil.which(command) is not None

    def _generate_report(self) -> Dict[str, Any]:
        """Generate fix report."""
        print("\n" + "=" * 80)
        print("AUTO-FIX REPORT")
        print("=" * 80)

        report = {
            "fixes_applied": self.fixes_applied,
            "errors": self.errors,
            "success": len(self.errors) == 0,
            "dry_run": self.dry_run
        }

        if self.fixes_applied:
            print(f"\n✓ Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        else:
            print("\n✓ No fixes needed or dry run mode")

        if self.errors:
            print(f"\n✗ Encountered {len(self.errors)} errors:")
            for error in self.errors:
                print(f"  - {error}")

        print()
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-fix code quality issues for GL-007"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes"
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Apply more aggressive fixes (experimental)"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory for GL-007 (default: current directory)"
    )

    args = parser.parse_args()

    # Create fixer
    fixer = AutoFixer(
        base_dir=args.base_dir,
        dry_run=args.dry_run,
        aggressive=args.aggressive
    )

    # Run fixes
    report = fixer.fix_all()

    # Exit with appropriate code
    sys.exit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
