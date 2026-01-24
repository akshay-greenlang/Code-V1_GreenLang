"""
Audit Bundle Generator

Creates comprehensive audit bundles for CBAM reports.
Implements the Evidence Packager agent from the PRD.
"""

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from cbam_pack import __version__
from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import CBAMConfig, Claim, ClaimType


class AuditBundleGenerator:
    """
    Generates audit bundles for CBAM compliance.

    Creates:
    - claims.json: Formal claims about emissions
    - lineage.json: Data lineage and provenance
    - assumptions.json: All assumptions made
    - run_manifest.json: Execution metadata
    - checksums.json: SHA-256 hashes of all artifacts
    """

    def __init__(
        self,
        factor_library_version: str = "unknown",
    ):
        """
        Initialize the audit bundle generator.

        Args:
            factor_library_version: Version of the emission factor library used
        """
        self.factor_library_version = factor_library_version

    def generate(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        input_files: list[Path],
        output_dir: Path,
        execution_time: float,
    ) -> list[str]:
        """
        Generate the complete audit bundle.

        Args:
            calc_result: Calculation results
            config: CBAM configuration
            input_files: List of input file paths
            output_dir: Output directory
            execution_time: Total execution time in seconds

        Returns:
            List of generated artifact filenames
        """
        artifacts = []

        # Create audit subdirectory
        audit_dir = output_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        # Generate claims.json
        claims_path = audit_dir / "claims.json"
        self._generate_claims(calc_result, config, claims_path)
        artifacts.append("audit/claims.json")

        # Generate lineage.json
        lineage_path = audit_dir / "lineage.json"
        self._generate_lineage(calc_result, config, input_files, lineage_path)
        artifacts.append("audit/lineage.json")

        # Generate assumptions.json
        assumptions_path = audit_dir / "assumptions.json"
        self._generate_assumptions(calc_result, assumptions_path)
        artifacts.append("audit/assumptions.json")

        # Generate run_manifest.json
        manifest_path = audit_dir / "run_manifest.json"
        self._generate_manifest(
            calc_result, config, input_files, execution_time, manifest_path
        )
        artifacts.append("audit/run_manifest.json")

        # Generate checksums.json (must be last)
        checksums_path = audit_dir / "checksums.json"
        self._generate_checksums(output_dir, checksums_path)
        artifacts.append("audit/checksums.json")

        return artifacts

    def _generate_claims(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        output_path: Path,
    ) -> None:
        """Generate claims.json with formal emissions claims."""
        stats = calc_result.statistics
        claims = []

        # Total emissions claim
        claims.append(
            Claim(
                claim_type=ClaimType.TOTAL_EMISSIONS,
                value=stats.get("total_emissions_tco2e", 0),
                unit="tCO2e",
                confidence="HIGH" if stats.get("default_usage_percent", 100) < 50 else "MEDIUM",
                methodology="CBAM Transitional Methodology",
                period=f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
                scope="Direct + Indirect Embedded Emissions",
            ).model_dump(mode='json')
        )

        # Direct emissions claim
        claims.append(
            Claim(
                claim_type=ClaimType.DIRECT_EMISSIONS,
                value=stats.get("total_direct_emissions_tco2e", 0),
                unit="tCO2e",
                confidence="HIGH" if stats.get("lines_with_supplier_direct_data", 0) > 0 else "MEDIUM",
                methodology="CBAM Transitional Methodology - Direct Emissions",
                period=f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
                scope="Direct Embedded Emissions (Scope 1 equivalent)",
            ).model_dump(mode='json')
        )

        # Indirect emissions claim
        claims.append(
            Claim(
                claim_type=ClaimType.INDIRECT_EMISSIONS,
                value=stats.get("total_indirect_emissions_tco2e", 0),
                unit="tCO2e",
                confidence="HIGH" if stats.get("lines_with_supplier_indirect_data", 0) > 0 else "MEDIUM",
                methodology="CBAM Transitional Methodology - Indirect Emissions",
                period=f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
                scope="Indirect Embedded Emissions (Scope 2 equivalent)",
            ).model_dump(mode='json')
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"claims": claims, "generated_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)

    def _generate_lineage(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        input_files: list[Path],
        output_path: Path,
    ) -> None:
        """Generate lineage.json with data provenance."""
        lineage = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "inputs": [],
            "transformations": [],
            "outputs": [],
        }

        # Input files
        for file_path in input_files:
            if file_path.exists():
                file_hash = self._compute_file_hash(file_path)
                lineage["inputs"].append({
                    "type": "file",
                    "path": str(file_path),
                    "sha256": file_hash,
                    "size_bytes": file_path.stat().st_size,
                })

        # Emission factor library
        lineage["inputs"].append({
            "type": "emission_factor_library",
            "version": self.factor_library_version,
            "source": "GreenLang CBAM Defaults 2024",
        })

        # Transformations (pipeline stages)
        lineage["transformations"] = [
            {
                "stage": 1,
                "name": "schema_validation",
                "description": "Validated input files against CBAM schema",
                "result": "passed",
            },
            {
                "stage": 2,
                "name": "unit_normalization",
                "description": "Normalized all quantities to tonnes",
                "result": "passed",
            },
            {
                "stage": 3,
                "name": "factor_lookup",
                "description": "Retrieved emission factors from library",
                "result": "passed",
            },
            {
                "stage": 4,
                "name": "emissions_calculation",
                "description": "Calculated direct and indirect emissions",
                "lines_processed": calc_result.statistics.get("total_lines", 0),
                "result": "passed",
            },
            {
                "stage": 5,
                "name": "aggregation",
                "description": "Aggregated results by CN code and country",
                "result": "passed",
            },
        ]

        # Outputs
        lineage["outputs"] = [
            {
                "type": "xml_report",
                "filename": "cbam_report.xml",
                "description": "CBAM transitional quarterly report",
            },
            {
                "type": "excel_summary",
                "filename": "report_summary.xlsx",
                "description": "Human-readable summary report",
            },
            {
                "type": "audit_bundle",
                "directory": "audit/",
                "description": "Complete audit evidence package",
            },
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(lineage, f, indent=2)

    def _generate_assumptions(
        self,
        calc_result: CalculationResult,
        output_path: Path,
    ) -> None:
        """Generate assumptions.json with all assumptions made."""
        assumptions_data = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "total_assumptions": len(calc_result.assumptions),
            "assumptions": [],
        }

        for assumption in calc_result.assumptions:
            assumptions_data["assumptions"].append({
                "type": assumption.type.value,
                "description": assumption.description,
                "rationale": assumption.rationale,
                "applies_to_lines": assumption.applies_to,
                "line_count": len(assumption.applies_to),
                "factor_ref": assumption.factor_ref,
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(assumptions_data, f, indent=2)

    def _generate_manifest(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        input_files: list[Path],
        execution_time: float,
        output_path: Path,
    ) -> None:
        """Generate run_manifest.json with execution metadata."""
        manifest = {
            "version": "1.0",
            "pack": {
                "name": "GreenLang CBAM Pack",
                "version": __version__,
                "type": "cbam-transitional",
            },
            "execution": {
                "started_at": (datetime.utcnow()).isoformat() + "Z",
                "duration_seconds": round(execution_time, 2),
                "status": "success",
            },
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "inputs": {
                "config_file": str(input_files[1]) if len(input_files) > 1 else None,
                "imports_file": str(input_files[0]) if input_files else None,
            },
            "configuration": {
                "declarant_eori": config.declarant.eori_number,
                "reporting_period": f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
                "aggregation_policy": config.settings.aggregation.value,
            },
            "results": {
                "total_lines": calc_result.statistics.get("total_lines", 0),
                "total_emissions_tco2e": calc_result.statistics.get("total_emissions_tco2e", 0),
                "default_usage_percent": calc_result.statistics.get("default_usage_percent", 0),
            },
            "factors": {
                "library_version": self.factor_library_version,
                "source": "GreenLang CBAM Defaults",
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _generate_checksums(
        self,
        output_dir: Path,
        output_path: Path,
    ) -> None:
        """Generate checksums.json with SHA-256 hashes of all artifacts."""
        checksums = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "algorithm": "SHA-256",
            "files": [],
        }

        # Hash all files in output directory (excluding checksums.json itself)
        for file_path in output_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "checksums.json":
                rel_path = file_path.relative_to(output_dir)
                file_hash = self._compute_file_hash(file_path)
                checksums["files"].append({
                    "path": str(rel_path),
                    "sha256": file_hash,
                    "size_bytes": file_path.stat().st_size,
                })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
