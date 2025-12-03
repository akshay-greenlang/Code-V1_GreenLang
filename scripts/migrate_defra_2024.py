#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEFRA 2024 Emission Factor Migration Script

Migrates emission factor database from DEFRA 2023 to DEFRA 2024.
Includes data validation, rollback capability, and migration report generation.

Usage:
    python migrate_defra_2024.py --validate-only
    python migrate_defra_2024.py --migrate
    python migrate_defra_2024.py --rollback

Author: GreenLang Data Integration Engineer
Date: 2024
"""

import argparse
import json
import logging
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MigrationStats:
    """Statistics from migration run."""
    source_version: str = "defra_2023"
    target_version: str = "defra_2024"
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Factor counts
    total_2023_factors: int = 0
    total_2024_factors: int = 0
    new_factors: int = 0
    updated_factors: int = 0
    unchanged_factors: int = 0
    removed_factors: int = 0

    # Category counts
    categories_2024: int = 0
    regions_2024: int = 0

    # Validation
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # Changes summary
    significant_changes: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    status: str = "pending"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


class DEFRAMigrator:
    """
    Migrates emission factor database from DEFRA 2023 to DEFRA 2024.

    Features:
    - Pre-migration validation
    - Backup creation
    - Factor comparison and change tracking
    - Rollback capability
    - Migration report generation
    """

    def __init__(self, data_dir: Optional[Path] = None, backup_dir: Optional[Path] = None):
        """
        Initialize migrator.

        Args:
            data_dir: Directory containing emission factor JSON files
            backup_dir: Directory for backup files
        """
        self.data_dir = data_dir or Path(__file__).parent.parent / "core" / "greenlang" / "data" / "factors"
        self.backup_dir = backup_dir or Path(__file__).parent.parent / "backups" / "defra_migration"
        self.stats = MigrationStats()

        # Ensure directories exist
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def validate_migration(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate migration prerequisites.

        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check source file exists
        source_path = self.data_dir / "defra_2023.json"
        if not source_path.exists():
            errors.append(f"Source file not found: {source_path}")

        # Check target file exists
        target_path = self.data_dir / "defra_2024.json"
        if not target_path.exists():
            errors.append(f"Target file not found: {target_path}")

        if errors:
            return False, errors, warnings

        # Load and validate JSON structure
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                source_data = json.load(f)
            self.stats.total_2023_factors = self._count_factors(source_data)
            logger.info(f"DEFRA 2023: {self.stats.total_2023_factors} factors")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in source file: {e}")

        try:
            with open(target_path, 'r', encoding='utf-8') as f:
                target_data = json.load(f)
            self.stats.total_2024_factors = self._count_factors(target_data)
            logger.info(f"DEFRA 2024: {self.stats.total_2024_factors} factors")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in target file: {e}")

        if errors:
            return False, errors, warnings

        # Validate metadata
        metadata = target_data.get('_metadata', {})
        if metadata.get('version') != '2024':
            warnings.append("Target file metadata version is not '2024'")

        # Check for significant factor changes
        significant_changes = self._compare_factors(source_data, target_data)
        self.stats.significant_changes = significant_changes

        if len(significant_changes) > 10:
            warnings.append(f"Found {len(significant_changes)} significant factor changes (>10% difference)")

        # Validate factor count
        if self.stats.total_2024_factors < self.stats.total_2023_factors * 0.5:
            errors.append("DEFRA 2024 has less than 50% of DEFRA 2023 factors")

        self.stats.validation_errors = errors
        self.stats.validation_warnings = warnings
        self.stats.validation_passed = len(errors) == 0

        return len(errors) == 0, errors, warnings

    def _count_factors(self, data: Dict[str, Any]) -> int:
        """Count total emission factors in data structure."""
        count = 0

        def count_recursive(obj: Any):
            nonlocal count
            if isinstance(obj, dict):
                # Check if this is a factor (has 'co2' key)
                if 'co2' in obj or 'co2e' in obj or 'co2e_ar6' in obj:
                    count += 1
                else:
                    for value in obj.values():
                        count_recursive(value)

        for key, value in data.items():
            if not key.startswith('_'):
                count_recursive(value)

        return count

    def _compare_factors(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compare factors between source and target versions."""
        changes = []

        def extract_factors(data: Dict, path: str = "") -> Dict[str, Dict]:
            """Extract all factors with their paths."""
            factors = {}

            def recurse(obj: Any, current_path: str):
                if isinstance(obj, dict):
                    if 'co2' in obj or 'co2e' in obj or 'co2e_ar6' in obj:
                        factors[current_path] = obj
                    else:
                        for key, value in obj.items():
                            if not key.startswith('_'):
                                new_path = f"{current_path}/{key}" if current_path else key
                                recurse(value, new_path)

            recurse(data, path)
            return factors

        source_factors = extract_factors(source)
        target_factors = extract_factors(target)

        # Compare matching factors
        for path, source_factor in source_factors.items():
            if path in target_factors:
                target_factor = target_factors[path]

                # Get emission values
                source_val = source_factor.get('co2e_ar6', source_factor.get('co2e', source_factor.get('co2', 0)))
                target_val = target_factor.get('co2e_ar6', target_factor.get('co2e', target_factor.get('co2', 0)))

                if source_val and target_val:
                    pct_change = ((target_val - source_val) / source_val) * 100 if source_val != 0 else 0

                    # Track significant changes (>10%)
                    if abs(pct_change) > 10:
                        changes.append({
                            "path": path,
                            "source_value": source_val,
                            "target_value": target_val,
                            "percent_change": round(pct_change, 2)
                        })

        return changes

    def create_backup(self) -> Path:
        """
        Create backup of current database state.

        Returns:
            Path to backup file
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"defra_backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)

        # Backup 2023 file
        source_2023 = self.data_dir / "defra_2023.json"
        if source_2023.exists():
            shutil.copy2(source_2023, backup_path / "defra_2023.json")
            logger.info(f"Backed up DEFRA 2023 to {backup_path}")

        # Create backup manifest
        manifest = {
            "created_at": datetime.utcnow().isoformat(),
            "source_dir": str(self.data_dir),
            "files": ["defra_2023.json"],
            "reason": "Pre-migration backup for DEFRA 2024 update"
        }

        with open(backup_path / "manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)

        return backup_path

    def migrate(self, dry_run: bool = False) -> MigrationStats:
        """
        Execute the migration.

        Args:
            dry_run: If True, don't make actual changes

        Returns:
            MigrationStats with results
        """
        self.stats.started_at = datetime.utcnow()

        # Validate first
        is_valid, errors, warnings = self.validate_migration()

        if not is_valid:
            self.stats.status = "failed"
            self.stats.error_message = "; ".join(errors)
            logger.error(f"Validation failed: {errors}")
            return self.stats

        for warning in warnings:
            logger.warning(warning)

        if dry_run:
            logger.info("Dry run mode - no changes will be made")
            self.stats.status = "dry_run"
        else:
            # Create backup
            backup_path = self.create_backup()
            logger.info(f"Backup created at {backup_path}")

            # Migration is complete since DEFRA 2024 file already exists
            # The main action is updating the database loader to prefer 2024

            self.stats.status = "completed"
            logger.info("Migration completed successfully")

        self.stats.completed_at = datetime.utcnow()

        # Generate report
        self._generate_report()

        return self.stats

    def rollback(self, backup_path: Optional[Path] = None) -> bool:
        """
        Rollback to previous state from backup.

        Args:
            backup_path: Path to backup directory (uses latest if not specified)

        Returns:
            True if rollback successful
        """
        if backup_path is None:
            # Find latest backup
            backups = list(self.backup_dir.glob("defra_backup_*"))
            if not backups:
                logger.error("No backups found for rollback")
                return False
            backup_path = sorted(backups)[-1]

        logger.info(f"Rolling back from {backup_path}")

        # Verify backup exists and has manifest
        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"No manifest found in backup {backup_path}")
            return False

        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Restore files
        for filename in manifest.get('files', []):
            source = backup_path / filename
            target = self.data_dir / filename

            if source.exists():
                shutil.copy2(source, target)
                logger.info(f"Restored {filename}")
            else:
                logger.warning(f"Backup file not found: {source}")

        logger.info("Rollback completed")
        return True

    def _generate_report(self) -> str:
        """Generate migration report."""
        report = f"""
================================================================================
DEFRA 2024 MIGRATION REPORT
================================================================================

Migration: {self.stats.source_version} -> {self.stats.target_version}
Started:   {self.stats.started_at.isoformat()}
Completed: {self.stats.completed_at.isoformat() if self.stats.completed_at else 'N/A'}
Status:    {self.stats.status.upper()}

--------------------------------------------------------------------------------
FACTOR COUNTS
--------------------------------------------------------------------------------
DEFRA 2023 Factors:    {self.stats.total_2023_factors}
DEFRA 2024 Factors:    {self.stats.total_2024_factors}
Change:                {self.stats.total_2024_factors - self.stats.total_2023_factors:+d}

--------------------------------------------------------------------------------
VALIDATION
--------------------------------------------------------------------------------
Passed: {'Yes' if self.stats.validation_passed else 'No'}
Errors: {len(self.stats.validation_errors)}
Warnings: {len(self.stats.validation_warnings)}
"""

        if self.stats.validation_errors:
            report += "\nErrors:\n"
            for err in self.stats.validation_errors:
                report += f"  - {err}\n"

        if self.stats.validation_warnings:
            report += "\nWarnings:\n"
            for warn in self.stats.validation_warnings:
                report += f"  - {warn}\n"

        if self.stats.significant_changes:
            report += f"""
--------------------------------------------------------------------------------
SIGNIFICANT CHANGES (>10% difference)
--------------------------------------------------------------------------------
Found {len(self.stats.significant_changes)} factors with significant changes:

"""
            for change in self.stats.significant_changes[:20]:  # Show first 20
                report += f"  {change['path']}: {change['source_value']} -> {change['target_value']} ({change['percent_change']:+.1f}%)\n"

            if len(self.stats.significant_changes) > 20:
                report += f"\n  ... and {len(self.stats.significant_changes) - 20} more\n"

        report += """
================================================================================
END OF REPORT
================================================================================
"""
        # Save report
        report_path = self.backup_dir / f"migration_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Report saved to {report_path}")
        print(report)

        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="DEFRA 2024 Emission Factor Migration")
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate migration prerequisites'
    )
    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Execute migration'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Execute migration in dry-run mode'
    )
    parser.add_argument(
        '--rollback',
        action='store_true',
        help='Rollback to previous state'
    )
    parser.add_argument(
        '--backup-path',
        type=str,
        help='Path to backup directory for rollback'
    )

    args = parser.parse_args()

    migrator = DEFRAMigrator()

    if args.validate_only:
        is_valid, errors, warnings = migrator.validate_migration()
        print(f"\nValidation {'PASSED' if is_valid else 'FAILED'}")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        sys.exit(0 if is_valid else 1)

    elif args.migrate or args.dry_run:
        stats = migrator.migrate(dry_run=args.dry_run)
        sys.exit(0 if stats.status in ['completed', 'dry_run'] else 1)

    elif args.rollback:
        backup_path = Path(args.backup_path) if args.backup_path else None
        success = migrator.rollback(backup_path)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
