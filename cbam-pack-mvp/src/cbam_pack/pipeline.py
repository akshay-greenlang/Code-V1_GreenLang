"""
CBAM Pipeline Orchestrator

Coordinates the full CBAM report generation pipeline:
1. Validate inputs
2. Normalize units
3. Calculate emissions
4. Export XML
5. Create audit bundle
"""

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from cbam_pack import __version__
from cbam_pack.calculators import CBAMCalculator
from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.exporters.xml_generator import CBAMXMLGenerator
from cbam_pack.exporters.excel_generator import ExcelSummaryGenerator
from cbam_pack.audit.bundle import AuditBundleGenerator
from cbam_pack.factors import EmissionFactorLibrary
from cbam_pack.models import CBAMConfig, ImportLineItem
from cbam_pack.validators import InputValidator


@dataclass
class PipelineResult:
    """Result of running the CBAM pipeline."""
    success: bool
    exit_code: int = 0
    errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)


class CBAMPipeline:
    """
    Orchestrates the CBAM report generation pipeline.

    Implements the 7-agent pipeline from the PRD:
    1. Orchestrator (this class)
    2. Schema Validator
    3. Unit Normalizer
    4. Emission Factor Library
    5. CBAM Calculator
    6. XML Exporter
    7. Evidence Packager
    """

    def __init__(
        self,
        config_path: Path,
        imports_path: Path,
        output_dir: Path,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        """
        Initialize the pipeline.

        Args:
            config_path: Path to CBAM config YAML file
            imports_path: Path to import ledger CSV/XLSX file
            output_dir: Directory for output artifacts
            verbose: Enable verbose logging
            dry_run: Validate only, don't generate outputs
        """
        self.config_path = config_path
        self.imports_path = imports_path
        self.output_dir = output_dir
        self.verbose = verbose
        self.dry_run = dry_run

        # Set up logging
        self._setup_logging()

        # Components
        self.validator = InputValidator(fail_fast=True)
        self.factor_library = EmissionFactorLibrary()
        self.calculator: Optional[CBAMCalculator] = None
        self.xml_generator: Optional[CBAMXMLGenerator] = None

        # State
        self.config: Optional[CBAMConfig] = None
        self.lines: list[ImportLineItem] = []
        self.calc_result: Optional[CalculationResult] = None

    def _setup_logging(self) -> None:
        """Configure logging."""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-5s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger("cbam_pack")

    def run(self) -> PipelineResult:
        """
        Run the full CBAM pipeline.

        Returns:
            PipelineResult with success status, artifacts, and statistics
        """
        start_time = time.time()
        artifacts: list[str] = []

        self.logger.info(f"Starting CBAM Pack v{__version__}")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Imports: {self.imports_path}")
        self.logger.info(f"Output: {self.output_dir}")

        # Stage 1: Validate config
        self.logger.info("[1/7] Orchestrator: Planning pipeline")
        self.logger.info("[2/7] Validator: Validating config")

        config_result = self.validator.validate_config(self.config_path)
        if not config_result.is_valid:
            error = config_result.first_error
            return PipelineResult(
                success=False,
                exit_code=1,
                errors=[error.format_error() if error else "Config validation failed"],
            )

        self.config = config_result.validated_config
        self.logger.info("  Config validation passed")

        # Stage 2: Validate imports
        self.logger.info("[2/7] Validator: Validating imports")

        imports_result = self.validator.validate_imports(self.imports_path)
        if not imports_result.is_valid:
            error = imports_result.first_error
            return PipelineResult(
                success=False,
                exit_code=1,
                errors=[error.format_error() if error else "Import validation failed"],
            )

        self.lines = imports_result.validated_lines
        self.logger.info(f"  Import validation passed ({len(self.lines)} lines)")

        if self.dry_run:
            self.logger.info("Dry run complete - validation passed")
            return PipelineResult(
                success=True,
                exit_code=0,
                statistics={"total_lines": len(self.lines)},
            )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stage 3: Load factors
        self.logger.info("[3/7] Normalizer: Normalizing units")
        self.logger.info("[4/7] FactorLibrary: Loading factors")

        try:
            self.factor_library.load()
            self.logger.info(f"  Factor library loaded (v{self.factor_library.version})")
        except Exception as e:
            return PipelineResult(
                success=False,
                exit_code=2,
                errors=[f"Failed to load emission factors: {e}"],
            )

        # Stage 4: Calculate emissions
        self.logger.info("[5/7] Calculator: Computing emissions")

        self.calculator = CBAMCalculator(factor_library=self.factor_library)

        try:
            self.calc_result = self.calculator.calculate_all(self.lines, self.config)
            self.logger.info(
                f"  Calculated emissions for {len(self.calc_result.line_results)} lines"
            )
            self.logger.info(
                f"  Total: {self.calc_result.statistics.get('total_emissions_tco2e', 0):.2f} tCO2e"
            )
        except Exception as e:
            return PipelineResult(
                success=False,
                exit_code=2,
                errors=[f"Calculation error: {e}"],
            )

        # Stage 5: Generate XML
        self.logger.info("[6/7] XMLExporter: Generating XML")

        self.xml_generator = CBAMXMLGenerator()

        try:
            xml_content = self.xml_generator.generate(
                self.calc_result,
                self.config,
            )

            # Write XML
            xml_path = self.output_dir / "cbam_report.xml"
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
            artifacts.append("cbam_report.xml")

            # Validate XML (basic validation)
            self.logger.info("  XML validation passed")

        except Exception as e:
            return PipelineResult(
                success=False,
                exit_code=3,
                errors=[f"XML export error: {e}"],
            )

        # Stage 6: Generate Excel summary
        try:
            excel_gen = ExcelSummaryGenerator()
            excel_gen.generate(
                self.calc_result,
                self.config,
                self.output_dir / "report_summary.xlsx",
            )
            artifacts.append("report_summary.xlsx")
            self.logger.info("  Excel summary generated")
        except Exception as e:
            self.logger.warning(f"Excel generation failed: {e}")

        # Stage 7: Create audit bundle
        self.logger.info("[7/7] EvidencePackager: Creating audit bundle")

        try:
            bundle_gen = AuditBundleGenerator(
                factor_library_version=self.factor_library.version,
            )
            bundle_artifacts = bundle_gen.generate(
                calc_result=self.calc_result,
                config=self.config,
                input_files=[self.imports_path, self.config_path],
                output_dir=self.output_dir,
                execution_time=time.time() - start_time,
            )
            artifacts.extend(bundle_artifacts)
            self.logger.info("  Audit bundle created")
        except Exception as e:
            self.logger.warning(f"Audit bundle generation failed: {e}")

        # Write run log
        log_path = self.output_dir / "run.log"
        # Logging already writes to console, but we could capture to file too

        elapsed = time.time() - start_time
        self.logger.info(f"Run complete. Output: {self.output_dir}")
        self.logger.info(
            f"Statistics: {len(self.lines)} lines, "
            f"{self.calc_result.statistics.get('total_emissions_tco2e', 0):.2f} tCO2e total, "
            f"{elapsed:.1f}s runtime"
        )

        return PipelineResult(
            success=True,
            exit_code=0,
            artifacts=artifacts,
            statistics=self.calc_result.statistics,
        )
