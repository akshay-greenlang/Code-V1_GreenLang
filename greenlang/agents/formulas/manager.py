"""
Formula Manager - Main Orchestration Layer

This module provides the high-level API for formula management,
including version control, rollback, A/B testing, and migration.

This is the primary interface for interacting with the formula system.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from pathlib import Path
import logging

from greenlang.formulas.repository import FormulaRepository
from greenlang.formulas.engine import FormulaExecutionEngine
from greenlang.formulas.models import (
    FormulaMetadata,
    FormulaVersion,
    FormulaDependency,
    FormulaExecutionResult,
    FormulaComparisonResult,
    ABTest,
    VersionStatus,
    FormulaCategory,
)
from greenlang.exceptions import (
    ValidationError,
    ProcessingError,
)

logger = logging.getLogger(__name__)


class FormulaManager:
    """
    Central formula management system.

    This class provides the main API for:
    - Creating and versioning formulas
    - Executing formulas with provenance tracking
    - Rolling back to previous versions
    - A/B testing between formula variants
    - Comparing formula versions
    - Migrating formulas from external sources

    Example:
        >>> manager = FormulaManager("formulas.db")
        >>> result = manager.execute_formula("E1-1", {"scope1": 100, "scope2": 50})
        >>> manager.create_new_version("E1-1", {...}, "Fixed calculation bug")
        >>> manager.rollback_to_version("E1-1", 2)
    """

    def __init__(self, db_path: str, schema_path: Optional[str] = None):
        """
        Initialize formula manager.

        Args:
            db_path: Path to SQLite database
            schema_path: Path to schema.sql (optional)
        """
        self.repository = FormulaRepository(db_path, schema_path)
        self.engine = FormulaExecutionEngine(self.repository)

        logger.info(f"FormulaManager initialized with database: {db_path}")

    # ========================================================================
    # FORMULA EXECUTION
    # ========================================================================

    def execute_formula(
        self,
        formula_code: str,
        input_data: Dict[str, Any],
        version_number: Optional[int] = None,
        agent_name: Optional[str] = None,
        calculation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Any:
        """
        Execute formula and return output value.

        Args:
            formula_code: Formula code to execute
            input_data: Input values
            version_number: Specific version (None = latest active)
            agent_name: Name of agent executing formula
            calculation_id: ID linking to broader calculation
            user_id: User ID for audit trail

        Returns:
            Calculated output value

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If execution fails
        """
        result = self.engine.execute(
            formula_code=formula_code,
            input_data=input_data,
            version_number=version_number,
            agent_name=agent_name,
            calculation_id=calculation_id,
            user_id=user_id,
        )

        return result.output_value

    def execute_formula_full(
        self,
        formula_code: str,
        input_data: Dict[str, Any],
        version_number: Optional[int] = None,
        agent_name: Optional[str] = None,
        calculation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> FormulaExecutionResult:
        """
        Execute formula and return full execution result with provenance.

        Use this when you need the complete audit trail.

        Returns:
            FormulaExecutionResult with output, hashes, and timing
        """
        return self.engine.execute(
            formula_code=formula_code,
            input_data=input_data,
            version_number=version_number,
            agent_name=agent_name,
            calculation_id=calculation_id,
            user_id=user_id,
        )

    # ========================================================================
    # FORMULA METADATA MANAGEMENT
    # ========================================================================

    def create_formula(
        self,
        formula_code: str,
        formula_name: str,
        category: FormulaCategory,
        description: Optional[str] = None,
        standard_reference: Optional[str] = None,
        created_by: str = "system",
    ) -> int:
        """
        Create new formula metadata.

        Args:
            formula_code: Unique formula code (e.g., "E1-1")
            formula_name: Human-readable name
            category: Formula category
            description: Detailed description
            standard_reference: Reference to regulatory standard
            created_by: User creating the formula

        Returns:
            Formula ID

        Raises:
            ValidationError: If formula already exists
        """
        formula = FormulaMetadata(
            formula_code=formula_code,
            formula_name=formula_name,
            category=category,
            description=description,
            standard_reference=standard_reference,
            created_by=created_by,
        )

        formula_id = self.repository.create_formula(formula)
        logger.info(f"Created formula {formula_code} (id={formula_id})")

        return formula_id

    def get_formula(self, formula_code: str) -> Optional[FormulaMetadata]:
        """Get formula metadata by code."""
        return self.repository.get_formula_by_code(formula_code)

    def list_formulas(
        self, category: Optional[FormulaCategory] = None
    ) -> List[FormulaMetadata]:
        """List all formulas, optionally filtered by category."""
        return self.repository.list_formulas(
            category=category.value if category else None
        )

    # ========================================================================
    # VERSION MANAGEMENT
    # ========================================================================

    def create_new_version(
        self,
        formula_code: str,
        formula_data: Dict[str, Any],
        change_notes: str,
        created_by: str = "system",
        auto_activate: bool = False,
    ) -> int:
        """
        Create new formula version.

        Args:
            formula_code: Formula code
            formula_data: Version data (expression, inputs, etc.)
            change_notes: Description of changes
            created_by: User creating version
            auto_activate: Automatically activate this version

        Returns:
            Version ID

        Raises:
            ValidationError: If formula doesn't exist or data is invalid
        """
        # Get formula metadata
        formula = self.repository.get_formula_by_code(formula_code)
        if not formula:
            raise ValidationError(f"Formula {formula_code} not found")

        # Get next version number
        latest_version = self.repository.get_latest_version_number(formula.id)
        next_version = latest_version + 1

        # Create version
        version = FormulaVersion(
            formula_id=formula.id,
            version_number=next_version,
            formula_expression=formula_data['formula_expression'],
            calculation_type=formula_data['calculation_type'],
            required_inputs=formula_data.get('required_inputs', []),
            optional_inputs=formula_data.get('optional_inputs', []),
            output_unit=formula_data.get('output_unit'),
            output_type=formula_data.get('output_type', 'numeric'),
            validation_rules=formula_data.get('validation_rules'),
            deterministic=formula_data.get('deterministic', True),
            zero_hallucination=formula_data.get('zero_hallucination', True),
            version_status=VersionStatus.DRAFT,
            change_notes=change_notes,
            example_calculation=formula_data.get('example_calculation'),
            created_by=created_by,
        )

        version_id = self.repository.create_version(version)

        logger.info(
            f"Created version {next_version} for {formula_code} (id={version_id})"
        )

        # Auto-activate if requested
        if auto_activate:
            self.activate_version(formula_code, next_version)

        return version_id

    def get_active_formula(
        self, formula_code: str, as_of_date: Optional[date] = None
    ) -> Optional[FormulaVersion]:
        """
        Get active formula version as of a specific date.

        Args:
            formula_code: Formula code
            as_of_date: Date to check (defaults to today)

        Returns:
            Active FormulaVersion or None
        """
        return self.repository.get_active_version(formula_code, as_of_date)

    def get_version(
        self, formula_code: str, version_number: int
    ) -> Optional[FormulaVersion]:
        """Get specific formula version."""
        return self.repository.get_version(formula_code, version_number)

    def list_versions(self, formula_code: str) -> List[FormulaVersion]:
        """List all versions of a formula."""
        return self.repository.list_versions(formula_code)

    def activate_version(
        self,
        formula_code: str,
        version_number: int,
        effective_from: Optional[date] = None,
    ):
        """
        Activate a formula version.

        This will:
        1. Deactivate current active version (set effective_to)
        2. Activate new version (set status=active, effective_from)

        Args:
            formula_code: Formula code
            version_number: Version to activate
            effective_from: Date version becomes active (default: today)

        Raises:
            ValidationError: If version doesn't exist
        """
        # Get version to activate
        new_version = self.repository.get_version(formula_code, version_number)
        if not new_version:
            raise ValidationError(
                f"Version {version_number} not found for {formula_code}"
            )

        # Get currently active version
        current_active = self.repository.get_active_version(formula_code)

        activation_date = effective_from or date.today()

        # Deactivate current version
        if current_active:
            # Set effective_to to day before new activation
            self.repository.set_effective_dates(
                current_active.id,
                current_active.effective_from or date.today(),
                activation_date,
            )
            self.repository.update_version_status(
                current_active.id, VersionStatus.DEPRECATED
            )

            logger.info(
                f"Deactivated version {current_active.version_number} of {formula_code}"
            )

        # Activate new version
        self.repository.update_version_status(new_version.id, VersionStatus.ACTIVE)
        self.repository.set_effective_dates(
            new_version.id, activation_date, None
        )

        logger.info(
            f"Activated version {version_number} of {formula_code} "
            f"effective from {activation_date}"
        )

    def rollback_to_version(
        self, formula_code: str, version_number: int
    ) -> int:
        """
        Rollback to a previous formula version.

        This creates a NEW version that is a copy of the old version,
        preserving the complete audit trail.

        Args:
            formula_code: Formula code
            version_number: Version to rollback to

        Returns:
            New version ID (copy of old version)

        Raises:
            ValidationError: If version doesn't exist
        """
        # Get version to rollback to
        old_version = self.repository.get_version(formula_code, version_number)
        if not old_version:
            raise ValidationError(
                f"Version {version_number} not found for {formula_code}"
            )

        # Create new version as copy
        formula_data = {
            'formula_expression': old_version.formula_expression,
            'calculation_type': old_version.calculation_type,
            'required_inputs': old_version.required_inputs,
            'optional_inputs': old_version.optional_inputs,
            'output_unit': old_version.output_unit,
            'output_type': old_version.output_type,
            'validation_rules': old_version.validation_rules,
            'deterministic': old_version.deterministic,
            'zero_hallucination': old_version.zero_hallucination,
        }

        new_version_id = self.create_new_version(
            formula_code=formula_code,
            formula_data=formula_data,
            change_notes=f"Rollback to version {version_number}",
            created_by="system",
            auto_activate=True,
        )

        logger.info(
            f"Rolled back {formula_code} to version {version_number} "
            f"(new version id={new_version_id})"
        )

        return new_version_id

    # ========================================================================
    # DEPENDENCY MANAGEMENT
    # ========================================================================

    def add_dependency(
        self,
        formula_code: str,
        version_number: int,
        depends_on_formula_code: str,
        depends_on_version: Optional[int] = None,
        dependency_type: str = "required",
    ) -> int:
        """
        Add formula dependency.

        Args:
            formula_code: Formula that has dependency
            version_number: Version of formula
            depends_on_formula_code: Formula this depends on
            depends_on_version: Specific version (None = latest)
            dependency_type: "required" or "optional"

        Returns:
            Dependency ID
        """
        # Get version
        version = self.repository.get_version(formula_code, version_number)
        if not version:
            raise ValidationError(
                f"Version {version_number} not found for {formula_code}"
            )

        dependency = FormulaDependency(
            formula_version_id=version.id,
            depends_on_formula_code=depends_on_formula_code,
            depends_on_version_number=depends_on_version,
            dependency_type=dependency_type,
        )

        dep_id = self.repository.add_dependency(dependency)

        logger.info(
            f"Added dependency: {formula_code} v{version_number} "
            f"depends on {depends_on_formula_code}"
        )

        return dep_id

    def resolve_dependencies(
        self, formula_code: str, version_number: Optional[int] = None
    ) -> List[str]:
        """
        Get topologically sorted list of formula dependencies.

        Args:
            formula_code: Formula code
            version_number: Specific version (None = active)

        Returns:
            List of formula codes in execution order
        """
        if version_number:
            version = self.repository.get_version(formula_code, version_number)
        else:
            version = self.repository.get_active_version(formula_code)

        if not version:
            raise ValidationError(f"Version not found for {formula_code}")

        dependencies = self.repository.get_dependencies(version.id)

        # Simple topological sort (for now, just return dependency list)
        dep_codes = [dep.depends_on_formula_code for dep in dependencies]

        return dep_codes

    # ========================================================================
    # VERSION COMPARISON
    # ========================================================================

    def compare_versions(
        self, formula_code: str, version_a: int, version_b: int
    ) -> FormulaComparisonResult:
        """
        Compare two formula versions.

        Args:
            formula_code: Formula code
            version_a: First version number
            version_b: Second version number

        Returns:
            FormulaComparisonResult with differences

        Raises:
            ValidationError: If versions don't exist
        """
        v_a = self.repository.get_version(formula_code, version_a)
        v_b = self.repository.get_version(formula_code, version_b)

        if not v_a or not v_b:
            raise ValidationError("One or both versions not found")

        # Compare expressions
        expression_changed = v_a.formula_expression != v_b.formula_expression

        # Compare inputs
        inputs_a = set(v_a.required_inputs)
        inputs_b = set(v_b.required_inputs)
        inputs_changed = inputs_a != inputs_b
        added_inputs = list(inputs_b - inputs_a)
        removed_inputs = list(inputs_a - inputs_b)

        # Compare output units
        output_unit_changed = v_a.output_unit != v_b.output_unit

        # Compare validation rules
        validation_rules_changed = v_a.validation_rules != v_b.validation_rules

        # Performance comparison
        avg_time_diff_ms = None
        avg_time_diff_pct = None

        if v_a.avg_execution_time_ms and v_b.avg_execution_time_ms:
            avg_time_diff_ms = v_b.avg_execution_time_ms - v_a.avg_execution_time_ms
            avg_time_diff_pct = (avg_time_diff_ms / v_a.avg_execution_time_ms) * 100

        return FormulaComparisonResult(
            formula_code=formula_code,
            version_a=version_a,
            version_b=version_b,
            expression_changed=expression_changed,
            inputs_changed=inputs_changed,
            output_unit_changed=output_unit_changed,
            validation_rules_changed=validation_rules_changed,
            expression_diff=f"A: {v_a.formula_expression}\nB: {v_b.formula_expression}"
            if expression_changed
            else None,
            added_inputs=added_inputs,
            removed_inputs=removed_inputs,
            avg_time_diff_ms=avg_time_diff_ms,
            avg_time_diff_pct=avg_time_diff_pct,
        )

    # ========================================================================
    # A/B TESTING (Future Enhancement)
    # ========================================================================

    def create_ab_test(
        self,
        formula_code: str,
        control_version: int,
        variant_version: int,
        traffic_split: float = 0.5,
        test_name: Optional[str] = None,
    ) -> int:
        """
        Create A/B test between two formula versions.

        Args:
            formula_code: Formula to test
            control_version: Control version number
            variant_version: Variant version number
            traffic_split: % traffic to variant (0.0-1.0)
            test_name: Name for test (optional)

        Returns:
            Test ID

        Note: This is a placeholder for future A/B testing implementation
        """
        logger.warning("A/B testing not yet implemented")
        raise NotImplementedError("A/B testing coming soon")

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def close(self):
        """Close database connection."""
        self.repository.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
