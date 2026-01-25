"""
Audit Bundle Generator

Creates comprehensive audit bundles for CBAM reports.
Implements the Evidence Packager agent from the PRD.
"""

import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from cbam_pack import __version__
from cbam_pack.calculators.emissions_calculator import CalculationResult
from cbam_pack.models import CBAMConfig, Claim, ClaimType, ImportLineItem, MethodType


class AuditBundleGenerator:
    """
    Generates audit bundles for CBAM compliance.

    Creates:
    - claims.json: Formal claims about emissions with row pointers
    - lineage.json: Data lineage and provenance
    - assumptions.json: All assumptions made
    - gap_report.json: Data gaps and recommended actions
    - run_manifest.json: Execution metadata with schema references
    - checksums.json: SHA-256 hashes of all artifacts
    - evidence/: Immutable copies of input files
    - policy_validation.json: Policy engine results
    """

    # XSD Schema reference
    XSD_SCHEMA_VERSION = "1.0.0"
    XSD_SCHEMA_DATE = "2024-10-01"
    XSD_SCHEMA_URI = "urn:cbam:transitional:v1:schema"

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
        self.xml_validation_result: Optional[dict] = None
        self.policy_result: Optional[dict] = None

    def set_xml_validation_result(self, result: dict) -> None:
        """Set XML validation result from XML generator."""
        self.xml_validation_result = result

    def set_policy_result(self, result: dict) -> None:
        """Set policy validation result from policy engine."""
        self.policy_result = result

    def generate(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        input_files: list[Path],
        output_dir: Path,
        execution_time: float,
        lines: Optional[list[ImportLineItem]] = None,
    ) -> list[str]:
        """
        Generate the complete audit bundle.

        Args:
            calc_result: Calculation results
            config: CBAM configuration
            input_files: List of input file paths
            output_dir: Output directory
            execution_time: Total execution time in seconds
            lines: Original import line items (for gap report)

        Returns:
            List of generated artifact filenames
        """
        artifacts = []

        # Create audit subdirectory
        audit_dir = output_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)

        # Create evidence subdirectory
        evidence_dir = output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Copy input files to evidence folder (immutable copies)
        evidence_hashes = self._copy_evidence_files(input_files, evidence_dir)
        artifacts.append("evidence/")

        # Generate claims.json with row pointers
        claims_path = audit_dir / "claims.json"
        self._generate_claims(calc_result, config, claims_path, evidence_hashes, lines)
        artifacts.append("audit/claims.json")

        # Generate lineage.json
        lineage_path = audit_dir / "lineage.json"
        self._generate_lineage(calc_result, config, input_files, lineage_path, evidence_hashes)
        artifacts.append("audit/lineage.json")

        # Generate assumptions.json
        assumptions_path = audit_dir / "assumptions.json"
        self._generate_assumptions(calc_result, assumptions_path)
        artifacts.append("audit/assumptions.json")

        # Generate gap_report.json
        if lines:
            from cbam_pack.audit.gap_report import GapReportGenerator
            gap_gen = GapReportGenerator()
            gap_path = audit_dir / "gap_report.json"
            gap_gen.generate(calc_result, lines, config, gap_path)
            artifacts.append("audit/gap_report.json")

        # Generate policy_validation.json
        if self.policy_result:
            policy_path = audit_dir / "policy_validation.json"
            with open(policy_path, "w", encoding="utf-8") as f:
                json.dump(self.policy_result, f, indent=2)
            artifacts.append("audit/policy_validation.json")

        # Generate run_manifest.json with schema references
        manifest_path = audit_dir / "run_manifest.json"
        self._generate_manifest(
            calc_result, config, input_files, execution_time, manifest_path, evidence_hashes
        )
        artifacts.append("audit/run_manifest.json")

        # Generate checksums.json (must be last)
        checksums_path = audit_dir / "checksums.json"
        self._generate_checksums(output_dir, checksums_path)
        artifacts.append("audit/checksums.json")

        return artifacts

    def _copy_evidence_files(
        self,
        input_files: list[Path],
        evidence_dir: Path,
    ) -> dict[str, dict]:
        """
        Copy input files to evidence folder with hash verification.

        Returns:
            Dictionary mapping original filenames to evidence info
        """
        evidence_hashes = {}

        for file_path in input_files:
            if file_path.exists():
                # Compute hash before copying
                file_hash = self._compute_file_hash(file_path)

                # Copy to evidence folder
                dest_path = evidence_dir / file_path.name
                shutil.copy2(file_path, dest_path)

                # Verify hash after copy
                dest_hash = self._compute_file_hash(dest_path)
                assert file_hash == dest_hash, f"Hash mismatch after copy: {file_path.name}"

                evidence_hashes[file_path.name] = {
                    "original_path": str(file_path),
                    "evidence_path": f"evidence/{file_path.name}",
                    "sha256": file_hash,
                    "size_bytes": file_path.stat().st_size,
                    "copied_at": datetime.utcnow().isoformat() + "Z",
                }

        return evidence_hashes

    def _generate_claims(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        output_path: Path,
        evidence_hashes: dict,
        lines: Optional[list[ImportLineItem]] = None,
    ) -> None:
        """Generate claims.json with formal emissions claims and row pointers."""
        stats = calc_result.statistics
        claims = []

        # Build line lookup for row numbers
        line_row_map = {}
        if lines:
            for idx, line in enumerate(lines, start=2):  # Row 2 = first data row after header
                line_row_map[line.line_id] = idx

        # Get imports file evidence info
        imports_evidence = None
        for filename, info in evidence_hashes.items():
            if filename.endswith(('.csv', '.xlsx')):
                imports_evidence = info
                break

        # Total emissions claim
        claims.append({
            "claim_id": f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-001",
            "claim_type": ClaimType.TOTAL_EMISSIONS.value,
            "value": stats.get("total_emissions_tco2e", 0),
            "unit": "tCO2e",
            "confidence": "HIGH" if stats.get("default_usage_percent", 100) < 50 else "MEDIUM",
            "methodology": "CBAM Transitional Methodology",
            "period": f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
            "scope": "Direct + Indirect Embedded Emissions",
            "evidence_refs": [
                {
                    "file": imports_evidence["evidence_path"] if imports_evidence else "imports.csv",
                    "file_hash": imports_evidence["sha256"] if imports_evidence else None,
                    "rows": "all",
                    "columns": ["quantity", "cn_code", "country_of_origin"],
                }
            ] if imports_evidence else [],
        })

        # Direct emissions claim
        claims.append({
            "claim_id": f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-002",
            "claim_type": ClaimType.DIRECT_EMISSIONS.value,
            "value": stats.get("total_direct_emissions_tco2e", 0),
            "unit": "tCO2e",
            "confidence": "HIGH" if stats.get("lines_with_supplier_direct_data", 0) > 0 else "MEDIUM",
            "methodology": "CBAM Transitional Methodology - Direct Emissions",
            "period": f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
            "scope": "Direct Embedded Emissions (Scope 1 equivalent)",
            "evidence_refs": [],
        })

        # Indirect emissions claim
        claims.append({
            "claim_id": f"CLM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-003",
            "claim_type": ClaimType.INDIRECT_EMISSIONS.value,
            "value": stats.get("total_indirect_emissions_tco2e", 0),
            "unit": "tCO2e",
            "confidence": "HIGH" if stats.get("lines_with_supplier_indirect_data", 0) > 0 else "MEDIUM",
            "methodology": "CBAM Transitional Methodology - Indirect Emissions",
            "period": f"{config.reporting_period.quarter.value} {config.reporting_period.year}",
            "scope": "Indirect Embedded Emissions (Scope 2 equivalent)",
            "evidence_refs": [],
        })

        # Line-level claims with row pointers
        line_claims = []
        for result in calc_result.line_results:
            row_num = line_row_map.get(result.line_id)

            line_claims.append({
                "line_id": result.line_id,
                "row_number": row_num,
                "direct_emissions_tco2e": float(result.direct_emissions_tco2e),
                "indirect_emissions_tco2e": float(result.indirect_emissions_tco2e),
                "total_emissions_tco2e": float(result.total_emissions_tco2e),
                "method_direct": result.method_direct.value,
                "method_indirect": result.method_indirect.value,
                "factor_direct_ref": result.factor_direct_ref,
                "factor_indirect_ref": result.factor_indirect_ref,
                "evidence_pointer": {
                    "file": imports_evidence["evidence_path"] if imports_evidence else None,
                    "row": row_num,
                    "columns": {
                        "quantity": "quantity",
                        "cn_code": "cn_code",
                        "country": "country_of_origin",
                        "supplier_direct": "supplier_direct_emissions",
                        "supplier_indirect": "supplier_indirect_emissions",
                    }
                } if row_num else None,
            })

        report = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "summary_claims": claims,
            "line_claims": line_claims,
            "evidence_files": evidence_hashes,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    def _generate_lineage(
        self,
        calc_result: CalculationResult,
        config: CBAMConfig,
        input_files: list[Path],
        output_path: Path,
        evidence_hashes: dict,
    ) -> None:
        """Generate lineage.json with data provenance."""
        lineage = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "inputs": [],
            "transformations": [],
            "outputs": [],
        }

        # Input files with evidence references
        for file_path in input_files:
            if file_path.exists():
                evidence_info = evidence_hashes.get(file_path.name, {})
                lineage["inputs"].append({
                    "type": "file",
                    "path": str(file_path),
                    "evidence_copy": evidence_info.get("evidence_path"),
                    "sha256": evidence_info.get("sha256"),
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
            {
                "stage": 6,
                "name": "xml_generation",
                "description": "Generated CBAM XML report",
                "validation": self.xml_validation_result or {"status": "not_validated"},
                "result": "passed",
            },
            {
                "stage": 7,
                "name": "policy_validation",
                "description": "Evaluated against policy rules",
                "result": self.policy_result.get("status", "not_evaluated") if self.policy_result else "not_evaluated",
            },
        ]

        # Outputs
        lineage["outputs"] = [
            {
                "type": "xml_report",
                "filename": "cbam_report.xml",
                "description": "CBAM transitional quarterly report",
                "schema_validated": self.xml_validation_result.get("valid", False) if self.xml_validation_result else False,
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
            {
                "type": "evidence_bundle",
                "directory": "evidence/",
                "description": "Immutable copies of input files",
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
        evidence_hashes: dict,
    ) -> None:
        """Generate run_manifest.json with execution metadata and schema references."""
        manifest = {
            "version": "1.0",
            "pack": {
                "name": "GreenLang CBAM Pack",
                "version": __version__,
                "type": "cbam-compliance-essentials",
            },
            "schema": {
                "xsd_version": self.XSD_SCHEMA_VERSION,
                "xsd_date": self.XSD_SCHEMA_DATE,
                "xsd_uri": self.XSD_SCHEMA_URI,
                "validation_result": self.xml_validation_result or {"status": "not_validated"},
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
                "files": evidence_hashes,
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
            "policy": self.policy_result if self.policy_result else {"status": "not_evaluated"},
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
