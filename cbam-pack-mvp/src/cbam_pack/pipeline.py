"""
CBAM Pipeline Orchestrator

Coordinates the full CBAM report generation pipeline:
1. Validate inputs
2. Normalize units
3. Calculate emissions
4. Run policy validation
5. Export XML with XSD validation
6. Create audit bundle with evidence
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
from cbam_pack.policy import PolicyEngine, PolicyResult, PolicyStatus


@dataclass
class PipelineResult:
    """Result of running the CBAM pipeline."""
    success: bool
    exit_code: int = 0
    errors: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    statistics: dict = field(default_factory=dict)
    policy_result: Optional[dict] = None
    xml_validation: Optional[dict] = None
    gap_summary: Optional[dict] = None
    lines_using_defaults: list[dict] = field(default_factory=list)


class CBAMPipeline:
    """
    Orchestrates the CBAM report generation pipeline.

    Implements the 8-agent pipeline from the PRD:
    1. Orchestrator (this class)
    2. Schema Validator
    3. Policy Engine
    4. Unit Normalizer
    5. Emission Factor Library
    6. CBAM Calculator
    7. XML Exporter
    8. Evidence Packager
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
        self.policy_engine: Optional[PolicyEngine] = None

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
        policy_result_dict: Optional[dict] = None
        xml_validation_dict: Optional[dict] = None
        gap_summary: Optional[dict] = None
        lines_using_defaults: list[dict] = []

        self.logger.info(f"Starting CBAM Pack v{__version__}")
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Imports: {self.imports_path}")
        self.logger.info(f"Output: {self.output_dir}")

        # Stage 1: Validate config
        self.logger.info("[1/8] Orchestrator: Planning pipeline")
        self.logger.info("[2/8] Validator: Validating config")

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
        self.logger.info("[2/8] Validator: Validating imports")

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
        self.logger.info("[3/8] Normalizer: Normalizing units")
        self.logger.info("[4/8] FactorLibrary: Loading factors")

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
        self.logger.info("[5/8] Calculator: Computing emissions")

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

        # Stage 5: Policy validation
        self.logger.info("[3/8] PolicyEngine: Evaluating compliance")

        self.policy_engine = PolicyEngine.from_yaml_config(self.config)
        policy_result = self.policy_engine.evaluate(self.calc_result, self.config)
        policy_result_dict = policy_result.to_dict()

        self.logger.info(f"  Policy status: {policy_result.status.value}")
        if policy_result.violations:
            for v in policy_result.violations:
                self.logger.warning(f"  VIOLATION: {v.message}")
        if policy_result.warnings:
            for w in policy_result.warnings:
                self.logger.warning(f"  WARNING: {w.message}")

        # Build lines_using_defaults for UI drilldown
        lines_using_defaults = self._build_default_lines_detail()

        # Stage 6: Generate XML with validation
        self.logger.info("[6/8] XMLExporter: Generating XML")

        self.xml_generator = CBAMXMLGenerator()

        try:
            xml_content = self.xml_generator.generate(
                self.calc_result,
                self.config,
                validate=True,  # Enable XSD validation
            )

            # Get validation result
            xml_validation_dict = self.xml_generator.get_validation_result()
            if xml_validation_dict:
                self.logger.info(
                    f"  XML Schema Validation: {xml_validation_dict.get('status', 'UNKNOWN')}"
                )

            # Write XML
            xml_path = self.output_dir / "cbam_report.xml"
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_content)
            artifacts.append("cbam_report.xml")

        except Exception as e:
            return PipelineResult(
                success=False,
                exit_code=3,
                errors=[f"XML export error: {e}"],
            )

        # Stage 7: Generate Excel summary
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

        # Stage 8: Create audit bundle with evidence
        self.logger.info("[7/8] EvidencePackager: Creating audit bundle")

        try:
            bundle_gen = AuditBundleGenerator(
                factor_library_version=self.factor_library.version,
            )

            # Pass validation results to bundle generator
            bundle_gen.set_xml_validation_result(xml_validation_dict or {})
            bundle_gen.set_policy_result(policy_result_dict or {})

            bundle_artifacts = bundle_gen.generate(
                calc_result=self.calc_result,
                config=self.config,
                input_files=[self.imports_path, self.config_path],
                output_dir=self.output_dir,
                execution_time=time.time() - start_time,
                lines=self.lines,  # Pass lines for gap report
            )
            artifacts.extend(bundle_artifacts)
            self.logger.info("  Audit bundle created")

            # Read gap summary for response
            gap_report_path = self.output_dir / "audit" / "gap_report.json"
            if gap_report_path.exists():
                with open(gap_report_path, "r") as f:
                    gap_data = json.load(f)
                    gap_summary = gap_data.get("summary", {})

        except Exception as e:
            self.logger.warning(f"Audit bundle generation failed: {e}")

        # Write run log
        log_path = self.output_dir / "run.log"
        # Logging already writes to console, but we could capture to file too

        elapsed = time.time() - start_time
        self.logger.info(f"[8/8] Complete. Output: {self.output_dir}")
        self.logger.info(
            f"Statistics: {len(self.lines)} lines, "
            f"{self.calc_result.statistics.get('total_emissions_tco2e', 0):.2f} tCO2e total, "
            f"{elapsed:.1f}s runtime"
        )

        # Determine success based on policy
        success = True
        if policy_result.status == PolicyStatus.FAIL and not policy_result.can_export:
            success = False

        return PipelineResult(
            success=success,
            exit_code=0 if success else 4,
            artifacts=artifacts,
            statistics=self.calc_result.statistics,
            policy_result=policy_result_dict,
            xml_validation=xml_validation_dict,
            gap_summary=gap_summary,
            lines_using_defaults=lines_using_defaults,
        )

    def _build_default_lines_detail(self) -> list[dict]:
        """Build detailed information about lines using default factors."""
        from cbam_pack.models import MethodType

        details = []
        line_lookup = {line.line_id: line for line in self.lines}

        for result in self.calc_result.line_results:
            if result.method_direct == MethodType.DEFAULT:
                line = line_lookup.get(result.line_id)
                if line:
                    missing_fields = []
                    if line.supplier_direct_emissions is None:
                        missing_fields.append("supplier_direct_emissions")
                    if line.supplier_indirect_emissions is None:
                        missing_fields.append("supplier_indirect_emissions")
                    if not line.supplier_id:
                        missing_fields.append("supplier_id")
                    if not line.installation_id:
                        missing_fields.append("installation_id")

                    details.append({
                        "line_id": line.line_id,
                        "cn_code": line.cn_code,
                        "product_description": line.product_description,
                        "country_of_origin": line.country_of_origin,
                        "supplier_id": line.supplier_id,
                        "reason": "No supplier-specific emission data provided",
                        "missing_fields": missing_fields,
                        "recommended_action": (
                            f"Request emission data from supplier"
                            if line.supplier_id
                            else "Identify supplier and request CBAM data package"
                        ),
                        "total_emissions_tco2e": float(result.total_emissions_tco2e),
                    })

        return details
