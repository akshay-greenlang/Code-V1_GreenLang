"""
Formula Repository - Database Access Layer

This module provides the data access layer for formula versioning,
handling all database operations with proper error handling and
connection management.

Zero-hallucination approach: All operations are deterministic database
transactions with complete audit trails.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
import sqlite3
import json
import logging
from pathlib import Path

from greenlang.formulas.models import (
    FormulaMetadata,
    FormulaVersion,
    FormulaDependency,
    FormulaExecutionResult,
    ABTest,
    FormulaMigration,
    VersionStatus,
)
from greenlang.exceptions import (
    ValidationError,
    ProcessingError,
    IntegrationError,
)

logger = logging.getLogger(__name__)


class FormulaRepository:
    """
    Repository pattern for formula data access.

    This class encapsulates all database operations for formulas,
    providing a clean interface for CRUD operations with proper
    error handling and connection management.

    Example:
        >>> repo = FormulaRepository("formulas.db")
        >>> formula = repo.get_formula_by_code("E1-1")
        >>> version = repo.get_active_version("E1-1")
    """

    def __init__(self, db_path: str, schema_path: Optional[str] = None):
        """
        Initialize repository with database connection.

        Args:
            db_path: Path to SQLite database file
            schema_path: Path to schema.sql file (optional, for initialization)

        Raises:
            IntegrationError: If database connection fails
        """
        self.db_path = db_path
        self.schema_path = schema_path or str(
            Path(__file__).parent / "schema.sql"
        )
        self._connection: Optional[sqlite3.Connection] = None

        # Initialize database if it doesn't exist
        if not Path(db_path).exists():
            logger.info(f"Creating new database at {db_path}")
            self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._connection is None:
            try:
                self._connection = sqlite3.connect(
                    self.db_path,
                    detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
                )
                self._connection.row_factory = sqlite3.Row
                logger.debug(f"Connected to database: {self.db_path}")
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise IntegrationError(f"Database connection failed: {e}") from e

        return self._connection

    def _initialize_database(self):
        """Initialize database schema from schema.sql."""
        try:
            schema_sql = Path(self.schema_path).read_text()
            conn = self._get_connection()

            # Execute schema in a transaction
            conn.executescript(schema_sql)
            conn.commit()

            logger.info("Database schema initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise IntegrationError(f"Database initialization failed: {e}") from e

    def close(self):
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.debug("Database connection closed")

    # ========================================================================
    # FORMULA METADATA OPERATIONS
    # ========================================================================

    def create_formula(self, formula: FormulaMetadata) -> int:
        """
        Create new formula metadata.

        Args:
            formula: Formula metadata to create

        Returns:
            Formula ID

        Raises:
            ValidationError: If formula already exists
            IntegrationError: If database operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO formulas (
                    formula_code, formula_name, category, description,
                    standard_reference, created_by
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    formula.formula_code,
                    formula.formula_name,
                    formula.category,
                    formula.description,
                    formula.standard_reference,
                    formula.created_by,
                ),
            )

            conn.commit()
            formula_id = cursor.lastrowid

            logger.info(f"Created formula {formula.formula_code} (id={formula_id})")
            return formula_id

        except sqlite3.IntegrityError as e:
            logger.error(f"Formula {formula.formula_code} already exists")
            raise ValidationError(
                f"Formula {formula.formula_code} already exists"
            ) from e
        except Exception as e:
            logger.error(f"Failed to create formula: {e}")
            raise IntegrationError(f"Failed to create formula: {e}") from e

    def get_formula_by_code(self, formula_code: str) -> Optional[FormulaMetadata]:
        """
        Get formula metadata by code.

        Args:
            formula_code: Formula code (e.g., "E1-1")

        Returns:
            FormulaMetadata or None if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM formulas WHERE formula_code = ?", (formula_code,)
            )

            row = cursor.fetchone()
            if not row:
                return None

            return FormulaMetadata(
                id=row["id"],
                formula_code=row["formula_code"],
                formula_name=row["formula_name"],
                category=row["category"],
                description=row["description"],
                standard_reference=row["standard_reference"],
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get formula {formula_code}: {e}")
            raise IntegrationError(f"Failed to get formula: {e}") from e

    def get_formula_by_id(self, formula_id: int) -> Optional[FormulaMetadata]:
        """Get formula metadata by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM formulas WHERE id = ?", (formula_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return FormulaMetadata(
                id=row["id"],
                formula_code=row["formula_code"],
                formula_name=row["formula_name"],
                category=row["category"],
                description=row["description"],
                standard_reference=row["standard_reference"],
                created_at=datetime.fromisoformat(row["created_at"]),
                created_by=row["created_by"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )

        except Exception as e:
            logger.error(f"Failed to get formula by id {formula_id}: {e}")
            raise IntegrationError(f"Failed to get formula: {e}") from e

    def list_formulas(
        self, category: Optional[str] = None, limit: int = 100, offset: int = 0
    ) -> List[FormulaMetadata]:
        """
        List all formulas with optional filtering.

        Args:
            category: Filter by category (optional)
            limit: Maximum results to return
            offset: Number of results to skip

        Returns:
            List of FormulaMetadata
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if category:
                cursor.execute(
                    """
                    SELECT * FROM formulas
                    WHERE category = ?
                    ORDER BY formula_code
                    LIMIT ? OFFSET ?
                    """,
                    (category, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM formulas
                    ORDER BY formula_code
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )

            formulas = []
            for row in cursor.fetchall():
                formulas.append(
                    FormulaMetadata(
                        id=row["id"],
                        formula_code=row["formula_code"],
                        formula_name=row["formula_name"],
                        category=row["category"],
                        description=row["description"],
                        standard_reference=row["standard_reference"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                        created_by=row["created_by"],
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )

            return formulas

        except Exception as e:
            logger.error(f"Failed to list formulas: {e}")
            raise IntegrationError(f"Failed to list formulas: {e}") from e

    # ========================================================================
    # FORMULA VERSION OPERATIONS
    # ========================================================================

    def create_version(self, version: FormulaVersion) -> int:
        """
        Create new formula version.

        Args:
            version: Formula version to create

        Returns:
            Version ID

        Raises:
            ValidationError: If version already exists
            IntegrationError: If database operation fails
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Convert to JSON dict for storage
            data = version.to_json_dict()

            cursor.execute(
                """
                INSERT INTO formula_versions (
                    formula_id, version_number, formula_expression,
                    calculation_type, required_inputs, optional_inputs,
                    output_unit, output_type, validation_rules,
                    deterministic, zero_hallucination, version_status,
                    effective_from, effective_to, change_notes,
                    example_calculation, ab_test_group, ab_traffic_weight,
                    created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data['formula_id'],
                    data['version_number'],
                    data['formula_expression'],
                    data['calculation_type'],
                    data['required_inputs'],
                    data['optional_inputs'],
                    data['output_unit'],
                    data['output_type'],
                    data['validation_rules'],
                    data['deterministic'],
                    data['zero_hallucination'],
                    data['version_status'],
                    data['effective_from'],
                    data['effective_to'],
                    data['change_notes'],
                    data['example_calculation'],
                    data['ab_test_group'],
                    data['ab_traffic_weight'],
                    data['created_by'],
                ),
            )

            conn.commit()
            version_id = cursor.lastrowid

            logger.info(
                f"Created version {version.version_number} for formula_id={version.formula_id}"
            )
            return version_id

        except sqlite3.IntegrityError as e:
            logger.error(f"Version {version.version_number} already exists")
            raise ValidationError(
                f"Version {version.version_number} already exists"
            ) from e
        except Exception as e:
            logger.error(f"Failed to create version: {e}")
            raise IntegrationError(f"Failed to create version: {e}") from e

    def get_version(
        self, formula_code: str, version_number: int
    ) -> Optional[FormulaVersion]:
        """Get specific formula version."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT fv.* FROM formula_versions fv
                JOIN formulas f ON fv.formula_id = f.id
                WHERE f.formula_code = ? AND fv.version_number = ?
                """,
                (formula_code, version_number),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_formula_version(row)

        except Exception as e:
            logger.error(f"Failed to get version: {e}")
            raise IntegrationError(f"Failed to get version: {e}") from e

    def get_active_version(
        self, formula_code: str, as_of_date: Optional[date] = None
    ) -> Optional[FormulaVersion]:
        """
        Get active version of formula as of a specific date.

        Args:
            formula_code: Formula code
            as_of_date: Date to check (defaults to today)

        Returns:
            Active FormulaVersion or None
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            check_date = as_of_date or date.today()

            cursor.execute(
                """
                SELECT fv.* FROM formula_versions fv
                JOIN formulas f ON fv.formula_id = f.id
                WHERE f.formula_code = ?
                    AND fv.version_status = 'active'
                    AND (fv.effective_from IS NULL OR fv.effective_from <= ?)
                    AND (fv.effective_to IS NULL OR fv.effective_to >= ?)
                ORDER BY fv.version_number DESC
                LIMIT 1
                """,
                (formula_code, check_date.isoformat(), check_date.isoformat()),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_formula_version(row)

        except Exception as e:
            logger.error(f"Failed to get active version: {e}")
            raise IntegrationError(f"Failed to get active version: {e}") from e

    def get_latest_version_number(self, formula_id: int) -> int:
        """Get the latest version number for a formula."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT MAX(version_number) as max_version
                FROM formula_versions
                WHERE formula_id = ?
                """,
                (formula_id,),
            )

            row = cursor.fetchone()
            return row["max_version"] or 0

        except Exception as e:
            logger.error(f"Failed to get latest version number: {e}")
            raise IntegrationError(f"Failed to get latest version number: {e}") from e

    def list_versions(
        self, formula_code: str
    ) -> List[FormulaVersion]:
        """List all versions of a formula."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT fv.* FROM formula_versions fv
                JOIN formulas f ON fv.formula_id = f.id
                WHERE f.formula_code = ?
                ORDER BY fv.version_number DESC
                """,
                (formula_code,),
            )

            versions = []
            for row in cursor.fetchall():
                versions.append(self._row_to_formula_version(row))

            return versions

        except Exception as e:
            logger.error(f"Failed to list versions: {e}")
            raise IntegrationError(f"Failed to list versions: {e}") from e

    def update_version_status(
        self, version_id: int, new_status: VersionStatus
    ):
        """Update version status."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE formula_versions
                SET version_status = ?
                WHERE id = ?
                """,
                (new_status.value, version_id),
            )

            conn.commit()
            logger.info(f"Updated version {version_id} status to {new_status}")

        except Exception as e:
            logger.error(f"Failed to update version status: {e}")
            raise IntegrationError(f"Failed to update version status: {e}") from e

    def set_effective_dates(
        self, version_id: int, effective_from: date, effective_to: Optional[date] = None
    ):
        """Set effective date range for a version."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE formula_versions
                SET effective_from = ?, effective_to = ?
                WHERE id = ?
                """,
                (
                    effective_from.isoformat(),
                    effective_to.isoformat() if effective_to else None,
                    version_id,
                ),
            )

            conn.commit()
            logger.info(f"Set effective dates for version {version_id}")

        except Exception as e:
            logger.error(f"Failed to set effective dates: {e}")
            raise IntegrationError(f"Failed to set effective dates: {e}") from e

    # ========================================================================
    # DEPENDENCY OPERATIONS
    # ========================================================================

    def add_dependency(self, dependency: FormulaDependency) -> int:
        """Add formula dependency."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO formula_dependencies (
                    formula_version_id, depends_on_formula_code,
                    depends_on_version_number, dependency_type
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    dependency.formula_version_id,
                    dependency.depends_on_formula_code,
                    dependency.depends_on_version_number,
                    dependency.dependency_type,
                ),
            )

            conn.commit()
            return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to add dependency: {e}")
            raise IntegrationError(f"Failed to add dependency: {e}") from e

    def get_dependencies(self, version_id: int) -> List[FormulaDependency]:
        """Get all dependencies for a formula version."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM formula_dependencies
                WHERE formula_version_id = ?
                """,
                (version_id,),
            )

            dependencies = []
            for row in cursor.fetchall():
                dependencies.append(
                    FormulaDependency(
                        id=row["id"],
                        formula_version_id=row["formula_version_id"],
                        depends_on_formula_code=row["depends_on_formula_code"],
                        depends_on_version_number=row["depends_on_version_number"],
                        dependency_type=row["dependency_type"],
                    )
                )

            return dependencies

        except Exception as e:
            logger.error(f"Failed to get dependencies: {e}")
            raise IntegrationError(f"Failed to get dependencies: {e}") from e

    # ========================================================================
    # EXECUTION LOG OPERATIONS
    # ========================================================================

    def log_execution(self, execution: FormulaExecutionResult) -> int:
        """Log formula execution."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            data = execution.to_json_dict()

            cursor.execute(
                """
                INSERT INTO formula_execution_log (
                    formula_version_id, agent_name, calculation_id, user_id,
                    input_hash, output_hash, input_data, output_value,
                    execution_time_ms, execution_status, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data['formula_version_id'],
                    data['agent_name'],
                    data['calculation_id'],
                    data['user_id'],
                    data['input_hash'],
                    data['output_hash'],
                    data['input_data'],
                    data['output_value'],
                    data['execution_time_ms'],
                    data['execution_status'],
                    data['error_message'],
                ),
            )

            conn.commit()
            return cursor.lastrowid

        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
            raise IntegrationError(f"Failed to log execution: {e}") from e

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _row_to_formula_version(self, row: sqlite3.Row) -> FormulaVersion:
        """Convert database row to FormulaVersion model."""
        return FormulaVersion(
            id=row["id"],
            formula_id=row["formula_id"],
            version_number=row["version_number"],
            formula_expression=row["formula_expression"],
            calculation_type=row["calculation_type"],
            required_inputs=json.loads(row["required_inputs"]),
            optional_inputs=json.loads(row["optional_inputs"]) if row["optional_inputs"] else [],
            output_unit=row["output_unit"],
            output_type=row["output_type"],
            validation_rules=json.loads(row["validation_rules"]) if row["validation_rules"] else None,
            deterministic=bool(row["deterministic"]),
            zero_hallucination=bool(row["zero_hallucination"]),
            version_status=row["version_status"],
            effective_from=date.fromisoformat(row["effective_from"]) if row["effective_from"] else None,
            effective_to=date.fromisoformat(row["effective_to"]) if row["effective_to"] else None,
            change_notes=row["change_notes"],
            example_calculation=row["example_calculation"],
            ab_test_group=row["ab_test_group"],
            ab_traffic_weight=row["ab_traffic_weight"],
            avg_execution_time_ms=row["avg_execution_time_ms"],
            execution_count=row["execution_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        self.close()
