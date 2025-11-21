#!/usr/bin/env python3
"""
Database Migration Validator

Validates database migration files for safety and best practices.
"""

import sys
import re
import ast
from pathlib import Path
from typing import List, Tuple


class MigrationValidator:
    """Validator for database migration files."""

    # Dangerous operations that should be reviewed
    DANGEROUS_OPERATIONS = [
        "DROP TABLE",
        "DROP DATABASE",
        "TRUNCATE",
        "DROP COLUMN",
        "ALTER COLUMN.*DROP",
    ]

    # Operations that require special attention
    REVIEW_OPERATIONS = [
        "ADD COLUMN.*NOT NULL",  # May fail if table has data
        "ALTER COLUMN.*SET NOT NULL",
        "CREATE UNIQUE INDEX",  # May fail with duplicates
        "ALTER TABLE.*ADD CONSTRAINT.*UNIQUE",
    ]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.dangerous_ops: List[Tuple[int, str]] = []

    def validate(self) -> bool:
        """Validate the migration file."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse as Python AST
            try:
                tree = ast.parse(content)
                self._validate_structure(tree)
            except SyntaxError as e:
                self.errors.append(f"Python syntax error: {e}")
                return False

            # Check for dangerous operations
            self._check_dangerous_operations(content)

            # Check for migration dependencies
            self._check_dependencies(content)

            # Check for rollback capability
            self._check_rollback(content)

            # Check for data migration best practices
            self._check_data_migration(content)

            return len(self.errors) == 0

        except FileNotFoundError:
            self.errors.append(f"File not found: {self.file_path}")
            return False
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            return False

    def _validate_structure(self, tree: ast.Module) -> None:
        """Validate migration file structure."""
        # Check for required functions
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if "upgrade" not in functions:
            self.errors.append("Migration missing 'upgrade' function")

        if "downgrade" not in functions:
            self.warnings.append(
                "Migration missing 'downgrade' function - rollback not possible"
            )

    def _check_dangerous_operations(self, content: str) -> None:
        """Check for dangerous database operations."""
        lines = content.split("\n")

        for line_num, line in enumerate(lines, start=1):
            # Check dangerous operations
            for operation in self.DANGEROUS_OPERATIONS:
                if re.search(operation, line, re.IGNORECASE):
                    self.dangerous_ops.append((line_num, line.strip()))
                    self.warnings.append(
                        f"Line {line_num}: Dangerous operation detected: {operation}"
                    )

            # Check operations requiring review
            for operation in self.REVIEW_OPERATIONS:
                if re.search(operation, line, re.IGNORECASE):
                    self.warnings.append(
                        f"Line {line_num}: Operation requires review: {operation}"
                    )

    def _check_dependencies(self, content: str) -> None:
        """Check for proper migration dependencies."""
        # Look for revision and down_revision
        if "revision = " not in content:
            self.errors.append("Migration missing 'revision' identifier")

        if "down_revision = " not in content:
            self.warnings.append(
                "Migration missing 'down_revision' - may not be properly chained"
            )

        # Check for branch labels if this is a branching migration
        if "branch_labels = " in content:
            if "depends_on = " not in content:
                self.warnings.append(
                    "Branch migration may need 'depends_on' specification"
                )

    def _check_rollback(self, content: str) -> None:
        """Check for proper rollback capability."""
        if "def downgrade():" not in content:
            self.warnings.append(
                "No downgrade function - rollback not possible"
            )
        else:
            # Check if downgrade is empty
            if "pass" in content.split("def downgrade():")[1].split("def")[0]:
                self.warnings.append(
                    "Downgrade function is empty - rollback may not work"
                )

    def _check_data_migration(self, content: str) -> None:
        """Check for data migration best practices."""
        # Check for bulk operations
        if re.search(r"\.execute\(.*SELECT.*FROM.*\)", content, re.IGNORECASE | re.DOTALL):
            if "batch" not in content.lower():
                self.warnings.append(
                    "Data migration detected - consider using batch operations for large datasets"
                )

        # Check for foreign key handling
        if re.search(r"ALTER TABLE.*ADD CONSTRAINT.*FOREIGN KEY", content, re.IGNORECASE):
            self.warnings.append(
                "Foreign key constraint added - ensure data integrity before applying"
            )

        # Check for index creation on large tables
        if re.search(r"CREATE.*INDEX", content, re.IGNORECASE):
            self.warnings.append(
                "Index creation detected - may take time on large tables, consider CONCURRENTLY option"
            )

        # Check for transaction handling
        if "execute" in content and "transaction" not in content.lower():
            self.warnings.append(
                "Consider explicit transaction handling for data modifications"
            )

        # Check for NULL handling when adding NOT NULL columns
        if re.search(r"ADD COLUMN.*NOT NULL", content, re.IGNORECASE):
            if "DEFAULT" not in content:
                self.errors.append(
                    "Adding NOT NULL column without DEFAULT - will fail if table has data"
                )

    def print_results(self) -> None:
        """Print validation results."""
        if self.errors:
            print(f"\n{'='*80}")
            print(f"VALIDATION FAILED: {self.file_path}")
            print(f"{'='*80}")
            for error in self.errors:
                print(f"ERROR: {error}")

        if self.dangerous_ops:
            print(f"\n{'='*80}")
            print(f"DANGEROUS OPERATIONS: {self.file_path}")
            print(f"{'='*80}")
            print("The following dangerous operations were detected:")
            for line_num, line in self.dangerous_ops:
                print(f"  Line {line_num}: {line}")
            print("\nPlease review these operations carefully!")
            print("They can cause data loss or service disruption.")

        if self.warnings:
            print(f"\n{'='*80}")
            print(f"WARNINGS: {self.file_path}")
            print(f"{'='*80}")
            for warning in self.warnings:
                print(f"WARNING: {warning}")

        if not self.errors and not self.dangerous_ops and not self.warnings:
            print(f"✓ {self.file_path} - Valid")


def main() -> int:
    """Main entry point for the hook."""
    if len(sys.argv) < 2:
        print("Usage: validate-migrations.py <migration.py> [migration.py ...]")
        return 1

    all_valid = True
    has_dangerous_ops = False

    for file_path in sys.argv[1:]:
        validator = MigrationValidator(file_path)
        is_valid = validator.validate()
        validator.print_results()

        if not is_valid:
            all_valid = False

        if validator.dangerous_ops:
            has_dangerous_ops = True

    # Fail if there are errors OR dangerous operations
    if not all_valid:
        return 1

    if has_dangerous_ops:
        print("\n" + "="*80)
        print("DANGEROUS OPERATIONS DETECTED")
        print("="*80)
        print("Your migration contains potentially dangerous operations.")
        print("Please review carefully before proceeding.")
        print("\nTo bypass this check (NOT RECOMMENDED):")
        print("  git commit --no-verify")
        print("="*80)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
