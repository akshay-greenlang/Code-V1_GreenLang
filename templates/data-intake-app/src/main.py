# -*- coding: utf-8 -*-
"""
Data Intake Application
=======================

Production-ready data intake application built entirely with GreenLang infrastructure.
Demonstrates ingestion from multiple formats with validation, caching, and monitoring.

Features:
- Multi-format data ingestion (CSV, Excel, JSON, XML)
- Schema validation with ValidationFramework
- Performance optimization with CacheManager
- Complete observability with Telemetry
- Provenance tracking for audit trails
- Database persistence with DatabaseManager
- Zero custom code - 100% infrastructure

Author: GreenLang Platform Team
Version: 1.0.0
"""

import asyncio
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from greenlang.agents.templates import IntakeAgent, DataFormat
from greenlang.validation import (
from greenlang.determinism import DeterministicClock
    ValidationFramework,
    SchemaValidator,
    RulesEngine,
    Rule,
    RuleOperator,
    DataQualityValidator
)
from greenlang.cache import CacheManager, initialize_cache_manager, get_cache_manager
from greenlang.provenance import ProvenanceTracker
from greenlang.telemetry import get_logger, get_metrics_collector, TelemetryManager
from greenlang.config import ConfigManager, get_config_manager
from greenlang.db import DatabaseManager


class DataIntakeApplication:
    """
    Production-ready data intake application.

    This application demonstrates how to build a complete data intake system
    using ONLY GreenLang infrastructure components.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data intake application.

        Args:
            config_path: Path to configuration file (optional)
        """
        # Initialize configuration
        self.config = get_config_manager()
        if config_path:
            self.config.load_from_file(config_path)

        # Initialize telemetry
        self.telemetry = TelemetryManager()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()

        # Initialize cache
        initialize_cache_manager(
            enable_l1=self.config.get("cache.enable_l1", True),
            enable_l2=self.config.get("cache.enable_l2", False),
            enable_l3=self.config.get("cache.enable_l3", False)
        )
        self.cache = get_cache_manager()

        # Initialize database
        self.db = DatabaseManager(
            connection_string=self.config.get("database.url", "sqlite:///data_intake.db")
        )

        # Initialize provenance tracker
        self.provenance = ProvenanceTracker(name="data_intake_app")

        # Initialize validation framework
        self.validation = self._setup_validation()

        # Initialize intake agent
        self.intake_agent = IntakeAgent(
            schema=self._get_validation_schema(),
            validation_framework=self.validation
        )

        self.logger.info("Data Intake Application initialized successfully")

    def _get_validation_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for data validation.

        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "facility_id": {
                    "type": "string",
                    "pattern": "^[A-Z0-9-]+$",
                    "description": "Unique facility identifier"
                },
                "facility_name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 255
                },
                "emissions": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Total emissions in kg CO2e"
                },
                "energy_consumption": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Energy consumption in kWh"
                },
                "reporting_period": {
                    "type": "string",
                    "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                    "description": "Reporting period (YYYY-MM-DD)"
                },
                "data_quality_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Data quality score (0-100)"
                }
            },
            "required": [
                "facility_id",
                "facility_name",
                "emissions",
                "reporting_period"
            ]
        }

    def _setup_validation(self) -> ValidationFramework:
        """
        Setup multi-layer validation framework.

        Returns:
            Configured validation framework
        """
        framework = ValidationFramework()

        # Layer 1: Schema validation
        schema_validator = SchemaValidator(self._get_validation_schema())
        framework.add_validator("schema", schema_validator.validate)

        # Layer 2: Business rules
        rules_engine = RulesEngine()

        # Rule: Emissions must be non-negative
        rules_engine.add_rule(Rule(
            name="emissions_non_negative",
            field="emissions",
            operator=RuleOperator.GREATER_EQUAL,
            value=0,
            message="Emissions cannot be negative"
        ))

        # Rule: Energy consumption must be non-negative
        rules_engine.add_rule(Rule(
            name="energy_non_negative",
            field="energy_consumption",
            operator=RuleOperator.GREATER_EQUAL,
            value=0,
            message="Energy consumption cannot be negative"
        ))

        # Rule: Data quality score must be within valid range
        rules_engine.add_rule(Rule(
            name="data_quality_range",
            field="data_quality_score",
            operator=RuleOperator.BETWEEN,
            value=(0, 100),
            message="Data quality score must be between 0 and 100"
        ))

        framework.add_validator("business_rules", rules_engine.validate)

        # Layer 3: Data quality checks
        quality_validator = DataQualityValidator(
            completeness_threshold=0.95,
            consistency_checks=True,
            outlier_detection=True
        )
        framework.add_validator("quality", quality_validator.validate)

        return framework

    async def ingest_file(
        self,
        file_path: str,
        format: DataFormat,
        store_in_db: bool = True,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest data from a file.

        Args:
            file_path: Path to the file to ingest
            format: Data format (CSV, Excel, JSON, XML)
            store_in_db: Whether to store ingested data in database
            validate: Whether to validate data

        Returns:
            Ingestion result dictionary with statistics
        """
        operation_id = f"ingest_{Path(file_path).stem}_{DeterministicClock.now().isoformat()}"

        with self.provenance.track_operation(operation_id):
            start_time = DeterministicClock.now()

            try:
                self.logger.info(f"Starting ingestion: {file_path} ({format.value})")
                self.metrics.increment("ingestion.started")

                # Check cache first
                cache_key = f"ingestion:{file_path}:{format.value}"
                cached_result = await self.cache.get(cache_key)

                if cached_result:
                    self.logger.info("Returning cached ingestion result")
                    self.metrics.increment("ingestion.cache_hit")
                    return cached_result

                # Ingest data
                result = await self.intake_agent.ingest(
                    file_path=file_path,
                    format=format,
                    validate=validate
                )

                if not result.success:
                    self.logger.error(f"Ingestion failed: {result.validation_issues}")
                    self.metrics.increment("ingestion.failed")

                    return {
                        "success": False,
                        "errors": [str(issue) for issue in result.validation_issues],
                        "duration_seconds": (DeterministicClock.now() - start_time).total_seconds()
                    }

                # Track provenance
                self.provenance.add_metadata("file_path", file_path)
                self.provenance.add_metadata("format", format.value)
                self.provenance.add_metadata("rows_ingested", result.rows_read)
                self.provenance.add_metadata("validation_passed", True)

                self.provenance.track_data_transformation(
                    source=file_path,
                    destination="validated_data",
                    transformation="intake_validation",
                    input_records=result.rows_read,
                    output_records=result.rows_valid
                )

                # Store in database if requested
                if store_in_db and result.data is not None:
                    await self._store_in_database(result.data)
                    self.provenance.add_metadata("stored_in_db", True)

                # Build result
                ingestion_result = {
                    "success": True,
                    "rows_read": result.rows_read,
                    "rows_valid": result.rows_valid,
                    "validation_issues": len(result.validation_issues),
                    "duration_seconds": (DeterministicClock.now() - start_time).total_seconds(),
                    "provenance_id": self.provenance.get_record().record_id,
                    "data_sample": result.data.head(5).to_dict() if result.data is not None else None
                }

                # Cache the result
                await self.cache.set(cache_key, ingestion_result, ttl=3600)

                # Update metrics
                self.metrics.increment("ingestion.completed")
                self.metrics.record("ingestion.rows", result.rows_read)
                self.metrics.record("ingestion.duration", ingestion_result["duration_seconds"])

                self.logger.info(
                    f"Ingestion completed: {result.rows_read} rows in "
                    f"{ingestion_result['duration_seconds']:.2f}s"
                )

                return ingestion_result

            except Exception as e:
                self.logger.error(f"Ingestion error: {str(e)}", exc_info=True)
                self.metrics.increment("ingestion.error")

                return {
                    "success": False,
                    "error": str(e),
                    "duration_seconds": (DeterministicClock.now() - start_time).total_seconds()
                }

    async def _store_in_database(self, data: pd.DataFrame) -> None:
        """
        Store ingested data in database.

        Args:
            data: DataFrame to store
        """
        try:
            await self.db.store_dataframe(
                data,
                table_name="ingested_data",
                if_exists="append"
            )
            self.logger.info(f"Stored {len(data)} rows in database")

        except Exception as e:
            self.logger.error(f"Database storage error: {str(e)}")
            raise

    async def batch_ingest(
        self,
        file_configs: list[Dict[str, Any]],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest multiple files in batch.

        Args:
            file_configs: List of file configurations, each containing:
                - file_path: Path to file
                - format: Data format
                - validate: Whether to validate (optional)
            parallel: Whether to process files in parallel

        Returns:
            Batch ingestion results
        """
        self.logger.info(f"Starting batch ingestion of {len(file_configs)} files")

        with self.provenance.track_operation("batch_ingestion"):
            start_time = DeterministicClock.now()

            if parallel:
                # Parallel ingestion using asyncio
                tasks = [
                    self.ingest_file(
                        file_path=config["file_path"],
                        format=config["format"],
                        validate=config.get("validate", True)
                    )
                    for config in file_configs
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Sequential ingestion
                results = []
                for config in file_configs:
                    result = await self.ingest_file(
                        file_path=config["file_path"],
                        format=config["format"],
                        validate=config.get("validate", True)
                    )
                    results.append(result)

            # Aggregate results
            total_rows = sum(r.get("rows_read", 0) for r in results if isinstance(r, dict) and r.get("success"))
            successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))

            batch_result = {
                "total_files": len(file_configs),
                "successful": successful,
                "failed": len(file_configs) - successful,
                "total_rows_ingested": total_rows,
                "duration_seconds": (DeterministicClock.now() - start_time).total_seconds(),
                "individual_results": results
            }

            self.logger.info(
                f"Batch ingestion completed: {successful}/{len(file_configs)} files, "
                f"{total_rows} total rows"
            )

            return batch_result

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get application statistics.

        Returns:
            Statistics dictionary
        """
        cache_analytics = self.cache.get_analytics()

        return {
            "cache": {
                "total_requests": cache_analytics.total_requests,
                "hit_rate": cache_analytics.hit_rate,
                "evictions": cache_analytics.evictions
            },
            "provenance": {
                "total_operations": len(self.provenance.chain_of_custody),
                "data_transformations": len(self.provenance.context.data_lineage)
            },
            "agent": self.intake_agent.get_stats()
        }

    async def shutdown(self) -> None:
        """Gracefully shutdown the application."""
        self.logger.info("Shutting down Data Intake Application")

        # Save provenance record
        provenance_record = self.provenance.get_record()
        self.logger.info(f"Provenance record: {provenance_record.record_id}")

        # Close database connections
        await self.db.close()

        # Shutdown telemetry
        self.telemetry.shutdown()

        self.logger.info("Shutdown complete")


async def main():
    """Main entry point for the application."""
    # Initialize application
    app = DataIntakeApplication(config_path="config/config.yaml")

    try:
        # Example: Ingest single file
        result = await app.ingest_file(
            file_path="data/sample_emissions.csv",
            format=DataFormat.CSV,
            validate=True
        )

        print(f"\nIngestion Result:")
        print(f"  Success: {result['success']}")
        print(f"  Rows: {result.get('rows_read', 0)}")
        print(f"  Duration: {result['duration_seconds']:.2f}s")

        # Example: Batch ingestion
        batch_configs = [
            {"file_path": "data/facility_1.csv", "format": DataFormat.CSV},
            {"file_path": "data/facility_2.csv", "format": DataFormat.CSV},
            {"file_path": "data/facility_3.csv", "format": DataFormat.CSV}
        ]

        batch_result = await app.batch_ingest(batch_configs, parallel=True)

        print(f"\nBatch Ingestion Result:")
        print(f"  Files processed: {batch_result['total_files']}")
        print(f"  Successful: {batch_result['successful']}")
        print(f"  Total rows: {batch_result['total_rows_ingested']}")

        # Get statistics
        stats = app.get_statistics()
        print(f"\nApplication Statistics:")
        print(f"  Cache hit rate: {stats['cache']['hit_rate']:.1f}%")
        print(f"  Provenance operations: {stats['provenance']['total_operations']}")

    finally:
        # Shutdown
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
