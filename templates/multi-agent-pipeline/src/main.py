# -*- coding: utf-8 -*-
"""
Multi-Agent Pipeline Application
=================================

Production-ready multi-agent pipeline demonstrating orchestration of multiple agents.
Built entirely with GreenLang infrastructure.

Pipeline: Intake → Calculate → Report

Features:
- Agent-to-agent communication
- Pipeline orchestration with error handling
- Distributed processing capability
- Complete monitoring and observability
- Provenance tracking across agents
- 100% infrastructure

Author: GreenLang Platform Team
Version: 1.0.0
"""

import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from greenlang.agents.templates import IntakeAgent, CalculatorAgent, ReportingAgent, DataFormat, ReportFormat
from greenlang.core import Pipeline, PipelineStage, Orchestrator
from greenlang.provenance import ProvenanceTracker
from greenlang.telemetry import get_logger, get_metrics_collector, TelemetryManager
from greenlang.config import get_config_manager
from greenlang.determinism import DeterministicClock


class MultiAgentPipelineApplication:
    """
    Multi-agent pipeline for end-to-end emissions processing.

    Pipeline stages:
    1. Intake: Load and validate emissions data
    2. Calculate: Perform emissions calculations
    3. Report: Generate compliance reports
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the multi-agent pipeline."""
        # Initialize infrastructure
        self.config = get_config_manager()
        if config_path:
            self.config.load_from_file(config_path)

        self.telemetry = TelemetryManager()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        self.provenance = ProvenanceTracker(name="multi_agent_pipeline")

        # Initialize agents
        self.intake_agent = IntakeAgent()
        self.calculator_agent = CalculatorAgent()
        self.reporting_agent = ReportingAgent()

        # Register formulas
        self._register_formulas()

        # Initialize pipeline
        self.pipeline = self._create_pipeline()

        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            pipeline=self.pipeline,
            max_retries=3,
            timeout_seconds=300
        )

        self.logger.info("Multi-Agent Pipeline initialized")

    def _register_formulas(self):
        """Register calculation formulas."""
        def calculate_scope1(activity: float, factor: float) -> float:
            return activity * factor

        def calculate_scope2(electricity: float, grid_factor: float) -> float:
            return electricity * grid_factor

        def calculate_total(scope1: float, scope2: float, scope3: float = 0) -> float:
            return scope1 + scope2 + scope3

        self.calculator_agent.register_formula("scope1", calculate_scope1)
        self.calculator_agent.register_formula("scope2", calculate_scope2)
        self.calculator_agent.register_formula("total", calculate_total)

    def _create_pipeline(self) -> Pipeline:
        """Create the agent pipeline."""
        pipeline = Pipeline(name="emissions_processing")

        # Stage 1: Data Intake
        async def intake_stage(data: Dict[str, Any]) -> Dict[str, Any]:
            """Intake and validate data."""
            self.logger.info("Stage 1: Data Intake")

            result = await self.intake_agent.ingest(
                data=data.get("input_file"),
                format=DataFormat(data.get("format", "csv")),
                validate=True
            )

            if not result.success:
                raise ValueError(f"Intake failed: {result.validation_issues}")

            return {"data": result.data, "rows": result.rows_read}

        pipeline.add_stage(PipelineStage(
            name="intake",
            handler=intake_stage,
            timeout=60
        ))

        # Stage 2: Calculate Emissions
        async def calculate_stage(data: Dict[str, Any]) -> Dict[str, Any]:
            """Calculate emissions."""
            self.logger.info("Stage 2: Calculate Emissions")

            df = data["data"]
            results = []

            for _, row in df.iterrows():
                # Calculate scope1
                scope1_result = await self.calculator_agent.calculate(
                    "scope1",
                    {"activity": row["activity"], "factor": row["factor"]}
                )

                # Calculate scope2
                scope2_result = await self.calculator_agent.calculate(
                    "scope2",
                    {"electricity": row["electricity"], "grid_factor": row["grid_factor"]}
                )

                # Calculate total
                total_result = await self.calculator_agent.calculate(
                    "total",
                    {
                        "scope1": scope1_result.value,
                        "scope2": scope2_result.value,
                        "scope3": 0
                    }
                )

                results.append({
                    "facility": row["facility"],
                    "scope1": scope1_result.value,
                    "scope2": scope2_result.value,
                    "total": total_result.value
                })

            return {"results": results}

        pipeline.add_stage(PipelineStage(
            name="calculate",
            handler=calculate_stage,
            timeout=120
        ))

        # Stage 3: Generate Report
        async def report_stage(data: Dict[str, Any]) -> Dict[str, Any]:
            """Generate compliance report."""
            self.logger.info("Stage 3: Generate Report")

            import pandas as pd
            results_df = pd.DataFrame(data["results"])

            report_result = await self.reporting_agent.generate_report(
                data=results_df,
                format=ReportFormat.JSON
            )

            return {
                "report": report_result.data,
                "format": "JSON",
                "records": len(results_df)
            }

        pipeline.add_stage(PipelineStage(
            name="report",
            handler=report_stage,
            timeout=60
        ))

        return pipeline

    async def run_pipeline(
        self,
        input_file: str,
        format: str = "csv"
    ) -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Args:
            input_file: Path to input data file
            format: Data format (csv, excel, json)

        Returns:
            Pipeline execution result
        """
        with self.provenance.track_operation("pipeline_execution"):
            start_time = DeterministicClock.now()

            try:
                self.logger.info(f"Starting pipeline execution: {input_file}")
                self.metrics.increment("pipeline.started")

                # Execute pipeline
                result = await self.orchestrator.execute({
                    "input_file": input_file,
                    "format": format
                })

                duration = (DeterministicClock.now() - start_time).total_seconds()

                # Track provenance
                self.provenance.add_metadata("input_file", input_file)
                self.provenance.add_metadata("stages_completed", len(self.pipeline.stages))
                self.provenance.add_metadata("duration", duration)

                # Update metrics
                self.metrics.increment("pipeline.completed")
                self.metrics.record("pipeline.duration", duration)

                self.logger.info(f"Pipeline completed in {duration:.2f}s")

                return {
                    "success": True,
                    "result": result,
                    "duration_seconds": duration,
                    "provenance_id": self.provenance.get_record().record_id
                }

            except Exception as e:
                self.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
                self.metrics.increment("pipeline.failed")
                return {
                    "success": False,
                    "error": str(e),
                    "duration_seconds": (DeterministicClock.now() - start_time).total_seconds()
                }

    async def shutdown(self):
        """Shutdown the application."""
        self.logger.info("Shutting down Multi-Agent Pipeline")
        self.telemetry.shutdown()


async def main():
    """Main entry point."""
    app = MultiAgentPipelineApplication()

    try:
        result = await app.run_pipeline(
            input_file="data/emissions.csv",
            format="csv"
        )

        print(f"\nPipeline Result:")
        print(f"  Success: {result['success']}")
        print(f"  Duration: {result['duration_seconds']:.2f}s")

        if result['success']:
            print(f"  Records processed: {result['result']['records']}")

    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
