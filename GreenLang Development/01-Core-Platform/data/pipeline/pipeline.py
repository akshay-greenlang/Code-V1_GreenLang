# -*- coding: utf-8 -*-
"""
Automated Import Pipeline

Production-grade import pipeline with:
- Scheduled imports (daily/weekly)
- Pre-import validation
- Duplicate detection and resolution
- Success/failure logging
- Automatic rollback on failure
- Transaction management
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import shutil
import logging
import asyncio
import json
import hashlib
from dataclasses import asdict

import schedule
import yaml

from .models import ImportJob, ImportStatus, FactorVersion, ChangeLog, ChangeType
from .validator import EmissionFactorValidator
from ..models import EmissionResult
from greenlang.utilities.determinism import FinancialDecimal, DeterministicClock

logger = logging.getLogger(__name__)


class RollbackManager:
    """
    Manage database rollback capabilities.

    Creates backups before imports and enables rollback on failure.
    """

    def __init__(self, db_path: str, backup_dir: Optional[str] = None):
        """
        Initialize rollback manager.

        Args:
            db_path: Path to database
            backup_dir: Directory for backups (default: db_path/backups)
        """
        self.db_path = Path(db_path)

        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = self.db_path.parent / 'backups'

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_backup(self, job_id: str) -> str:
        """
        Create database backup.

        Args:
            job_id: Import job ID

        Returns:
            Path to backup file
        """
        timestamp = DeterministicClock.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{job_id}_{timestamp}.db"
        backup_path = self.backup_dir / backup_name

        logger.info(f"Creating backup: {backup_path}")

        # Copy database file
        if self.db_path.exists():
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Backup created: {backup_path} ({backup_path.stat().st_size / 1024:.1f} KB)")
        else:
            logger.warning(f"Database {self.db_path} does not exist, creating empty backup")
            backup_path.touch()

        return str(backup_path)

    def rollback(self, backup_path: str) -> bool:
        """
        Rollback database to backup.

        Args:
            backup_path: Path to backup file

        Returns:
            True if successful
        """
        backup_file = Path(backup_path)

        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False

        try:
            logger.warning(f"Rolling back database to: {backup_path}")

            # Close any open connections first
            # Copy backup over current database
            shutil.copy2(backup_file, self.db_path)

            logger.info("Rollback successful")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """
        Clean up old backup files.

        Args:
            keep_days: Number of days to keep backups

        Returns:
            Number of backups deleted
        """
        cutoff_time = DeterministicClock.now() - timedelta(days=keep_days)
        deleted_count = 0

        for backup_file in self.backup_dir.glob("backup_*.db"):
            file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)

            if file_time < cutoff_time:
                logger.info(f"Deleting old backup: {backup_file}")
                backup_file.unlink()
                deleted_count += 1

        logger.info(f"Cleaned up {deleted_count} old backups")
        return deleted_count


class AutomatedImportPipeline:
    """
    Automated import pipeline with validation and rollback.

    Handles the complete import workflow:
    1. Pre-import validation
    2. Database backup
    3. Import execution
    4. Post-import validation
    5. Rollback on failure
    6. Change tracking
    """

    def __init__(
        self,
        db_path: str,
        backup_dir: Optional[str] = None,
        enable_versioning: bool = True
    ):
        """
        Initialize import pipeline.

        Args:
            db_path: Path to SQLite database
            backup_dir: Directory for backups
            enable_versioning: Enable factor versioning
        """
        self.db_path = db_path
        self.rollback_manager = RollbackManager(db_path, backup_dir)
        self.validator = EmissionFactorValidator()
        self.enable_versioning = enable_versioning
        self.conn: Optional[sqlite3.Connection] = None

    async def execute_import(
        self,
        job: ImportJob,
        validate_before: bool = True,
        rollback_on_failure: bool = True
    ) -> ImportJob:
        """
        Execute import job with full pipeline.

        Args:
            job: Import job configuration
            validate_before: Run pre-import validation
            rollback_on_failure: Rollback on failure

        Returns:
            Updated ImportJob with results
        """
        logger.info(f"Starting import job: {job.job_id} - {job.job_name}")
        job.status = ImportStatus.RUNNING
        job.started_at = DeterministicClock.now()

        backup_path = None

        try:
            # Step 1: Pre-import validation
            if validate_before:
                logger.info("Running pre-import validation...")
                job.status = ImportStatus.VALIDATING

                validation_passed = await self._validate_source_files(job)
                job.pre_import_validation_passed = validation_passed

                if not validation_passed and job.validation_result:
                    if job.validation_result.quality_score < 50:
                        raise ValueError(
                            f"Validation failed: Quality score {job.validation_result.quality_score} < 50"
                        )
                    else:
                        logger.warning(
                            f"Validation warnings present, quality score: {job.validation_result.quality_score}"
                        )

            # Step 2: Create backup
            if rollback_on_failure:
                backup_path = self.rollback_manager.create_backup(job.job_id)
                job.backup_path = backup_path
                job.can_rollback = True

            # Step 3: Execute import
            job.status = ImportStatus.RUNNING
            logger.info("Executing import...")

            await self._execute_import_logic(job)

            # Step 4: Verify import
            logger.info("Verifying import...")
            verification_passed = await self._verify_import(job)

            if not verification_passed:
                raise ValueError("Post-import verification failed")

            # Step 5: Mark as complete
            job.status = ImportStatus.COMPLETED
            job.completed_at = DeterministicClock.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            logger.info(f"Import job completed successfully: {job.job_id}")
            logger.info(f"  Processed: {job.total_factors_processed}")
            logger.info(f"  Successful: {job.successful_imports}")
            logger.info(f"  Failed: {job.failed_imports}")
            logger.info(f"  Duration: {job.duration_seconds:.2f}s")

            return job

        except Exception as e:
            logger.error(f"Import job failed: {e}", exc_info=True)

            job.status = ImportStatus.FAILED
            job.completed_at = DeterministicClock.now()
            job.duration_seconds = (job.completed_at - job.started_at).total_seconds()

            job.errors.append({
                'error': str(e),
                'timestamp': DeterministicClock.now().isoformat()
            })

            # Rollback if enabled
            if rollback_on_failure and backup_path:
                logger.warning("Attempting rollback...")

                if self.rollback_manager.rollback(backup_path):
                    job.status = ImportStatus.ROLLED_BACK
                    job.rolled_back = True
                    job.rollback_timestamp = DeterministicClock.now()
                    logger.info("Rollback successful")
                else:
                    logger.error("Rollback failed!")

            return job

        finally:
            await self.validator.close()

    async def _validate_source_files(self, job: ImportJob) -> bool:
        """Validate all source YAML files."""
        all_results = []

        for yaml_path in job.source_files:
            logger.info(f"Validating file: {yaml_path}")

            result = await self.validator.validate_file(yaml_path)
            all_results.append(result)

            logger.info(f"  Valid: {result.is_valid}")
            logger.info(f"  Quality Score: {result.quality_score}")
            logger.info(f"  Errors: {len(result.errors)}")
            logger.info(f"  Warnings: {len(result.warnings)}")

        # Aggregate results
        total_records = sum(r.total_records for r in all_results)
        valid_records = sum(r.valid_records for r in all_results)
        invalid_records = sum(r.invalid_records for r in all_results)

        all_errors = []
        all_warnings = []
        for r in all_results:
            all_errors.extend(r.errors)
            all_warnings.extend(r.warnings)

        avg_quality = sum(r.quality_score for r in all_results) / len(all_results) if all_results else 0

        from .models import ValidationResult

        job.validation_result = ValidationResult(
            validation_id=f"pre_import_{job.job_id}",
            is_valid=invalid_records == 0,
            quality_score=avg_quality,
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            warning_records=len(all_warnings),
            rules_passed=[],
            rules_failed=[],
            errors=all_errors,
            warnings=all_warnings,
            validation_duration_ms=0.0
        )

        return job.validation_result.is_valid or job.validation_result.quality_score >= 50

    async def _execute_import_logic(self, job: ImportJob):
        """Execute the actual import logic."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")

        try:
            for yaml_path in job.source_files:
                await self._import_yaml_file(yaml_path, job)

            self.conn.commit()

        except Exception as e:
            self.conn.rollback()
            raise e

        finally:
            if self.conn:
                self.conn.close()
                self.conn = None

    async def _import_yaml_file(self, yaml_path: str, job: ImportJob):
        """Import single YAML file."""
        logger.info(f"Importing file: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Remove metadata section
        if 'metadata' in data:
            del data['metadata']

        # Process each category
        for category, category_data in data.items():
            if not isinstance(category_data, dict):
                continue

            # Process each factor
            for factor_key, factor_data in category_data.items():
                if not isinstance(factor_data, dict):
                    continue

                factor_id = f"{category}_{factor_key}".lower()
                job.total_factors_processed += 1

                try:
                    # Check for duplicates
                    cursor = self.conn.cursor()
                    cursor.execute(
                        "SELECT factor_id, emission_factor_value FROM emission_factors WHERE factor_id = ?",
                        (factor_id,)
                    )
                    existing = cursor.fetchone()

                    if existing:
                        # Handle duplicate
                        await self._handle_duplicate(
                            factor_id,
                            factor_data,
                            existing,
                            job
                        )
                    else:
                        # Insert new factor
                        await self._insert_factor(
                            factor_id,
                            factor_data,
                            category,
                            factor_key,
                            job
                        )

                    job.successful_imports += 1
                    job.progress_percent = (job.successful_imports / job.total_factors_processed) * 100

                except Exception as e:
                    logger.error(f"Failed to import {factor_id}: {e}")
                    job.failed_imports += 1
                    job.errors.append({
                        'factor_id': factor_id,
                        'error': str(e),
                        'file': yaml_path
                    })

    async def _handle_duplicate(
        self,
        factor_id: str,
        new_data: Dict[str, Any],
        existing_data: tuple,
        job: ImportJob
    ):
        """Handle duplicate factor (update or skip)."""
        logger.debug(f"Duplicate factor detected: {factor_id}")

        # For now, skip duplicates (could implement update logic)
        job.duplicate_factors += 1
        job.skipped_imports += 1
        job.warnings.append({
            'factor_id': factor_id,
            'message': 'Duplicate factor skipped (already exists in database)'
        })

    async def _insert_factor(
        self,
        factor_id: str,
        factor_data: Dict[str, Any],
        category: str,
        subcategory: str,
        job: ImportJob
    ):
        """Insert new emission factor."""
        # Extract basic fields (reuse logic from import_emission_factors.py)
        primary_value = self._extract_primary_value(factor_data)
        primary_unit = self._extract_primary_unit(factor_data)

        name = factor_data.get('name', factor_id.replace('_', ' ').title())
        scope = factor_data.get('scope', 'Unknown')
        source_org = factor_data.get('source', 'Unknown')
        source_uri = factor_data.get('uri', factor_data.get('source_uri', 'https://example.com'))
        last_updated = factor_data.get('last_updated', DeterministicClock.now().date().isoformat())

        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO emission_factors (
                factor_id, name, category, subcategory,
                emission_factor_value, unit,
                scope, source_org, source_uri,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            factor_id, name, category, subcategory,
            primary_value, primary_unit,
            scope, source_org, source_uri,
            last_updated
        ))

        logger.debug(f"Inserted factor: {factor_id}")

    def _extract_primary_value(self, factor_data: Dict[str, Any]) -> float:
        """Extract primary emission factor value."""
        for key, value in factor_data.items():
            if key.startswith('emission_factor_kg_co2e_per_'):
                if isinstance(value, (int, float)) and value > 0:
                    return float(value)

        # Fallback
        for key in ['emission_factor', 'emission_factor_value']:
            if key in factor_data:
                return FinancialDecimal.from_string(factor_data[key])

        raise ValueError("No emission factor value found")

    def _extract_primary_unit(self, factor_data: Dict[str, Any]) -> str:
        """Extract primary unit."""
        for key in factor_data.keys():
            if key.startswith('emission_factor_kg_co2e_per_'):
                return key.replace('emission_factor_kg_co2e_per_', '')

        return factor_data.get('unit', 'unit')

    async def _verify_import(self, job: ImportJob) -> bool:
        """Verify import was successful."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check that records were added
            cursor.execute("SELECT COUNT(*) FROM emission_factors")
            count = cursor.fetchone()[0]

            conn.close()

            if count == 0:
                logger.error("Verification failed: No records in database")
                return False

            logger.info(f"Verification passed: {count} factors in database")
            return True

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False


class ScheduledImporter:
    """
    Scheduled import job manager.

    Runs imports on a schedule (daily, weekly, etc).
    """

    def __init__(
        self,
        pipeline: AutomatedImportPipeline,
        source_files: List[str],
        db_path: str
    ):
        """
        Initialize scheduled importer.

        Args:
            pipeline: Import pipeline
            source_files: List of YAML files to import
            db_path: Database path
        """
        self.pipeline = pipeline
        self.source_files = source_files
        self.db_path = db_path
        self.jobs_history: List[ImportJob] = []

    def schedule_daily(self, time_str: str = "02:00"):
        """
        Schedule daily import.

        Args:
            time_str: Time to run (HH:MM format)
        """
        logger.info(f"Scheduling daily import at {time_str}")

        schedule.every().day.at(time_str).do(self._run_scheduled_import)

    def schedule_weekly(self, day: str = "sunday", time_str: str = "02:00"):
        """
        Schedule weekly import.

        Args:
            day: Day of week (monday, tuesday, etc)
            time_str: Time to run (HH:MM format)
        """
        logger.info(f"Scheduling weekly import on {day} at {time_str}")

        getattr(schedule.every(), day.lower()).at(time_str).do(self._run_scheduled_import)

    def _run_scheduled_import(self):
        """Run scheduled import job."""
        job_id = f"scheduled_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}"

        job = ImportJob(
            job_id=job_id,
            job_name=f"Scheduled Import {DeterministicClock.now().strftime('%Y-%m-%d %H:%M')}",
            source_files=self.source_files,
            target_database=self.db_path,
            validate_before_import=True,
            triggered_by="scheduler",
            trigger_type="scheduled"
        )

        logger.info(f"Running scheduled import: {job_id}")

        # Run import asynchronously
        result = asyncio.run(self.pipeline.execute_import(job))

        self.jobs_history.append(result)

        # Log results
        if result.status == ImportStatus.COMPLETED:
            logger.info(f"Scheduled import completed: {result.success_rate:.1f}% success rate")
        else:
            logger.error(f"Scheduled import failed: {result.status}")

        return result

    def run_scheduler(self):
        """Run scheduler loop (blocking)."""
        logger.info("Starting scheduler...")

        while True:
            schedule.run_pending()
            asyncio.sleep(60)  # Check every minute

    def get_recent_jobs(self, limit: int = 10) -> List[ImportJob]:
        """Get recent job history."""
        return self.jobs_history[-limit:]
