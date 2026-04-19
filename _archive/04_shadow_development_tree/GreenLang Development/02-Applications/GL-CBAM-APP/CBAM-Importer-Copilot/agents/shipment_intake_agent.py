# -*- coding: utf-8 -*-
"""
ShipmentIntakeAgent_AI - Data Ingestion, Validation, and Enrichment for CBAM Shipments

This agent is responsible for:
1. Reading shipment data from various sources (CSV, JSON, Excel)
2. Validating each shipment against CBAM requirements
3. Enriching shipments with CN code metadata
4. Linking shipments to supplier records when actual emissions data available
5. Flagging data quality issues
6. Outputting validated shipment records in standard format

Version: 1.0.0
Author: GreenLang CBAM Team
License: Proprietary
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from jsonschema import Draft7Validator, ValidationError
from pydantic import BaseModel, Field, validator
from greenlang.determinism import DeterministicClock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ERROR CODES (from CBAM rules)
# ============================================================================

ERROR_CODES = {
    "E001": "Missing required field",
    "E002": "Invalid CN code",
    "E003": "Invalid date format",
    "E004": "Negative or zero mass",
    "E005": "Emissions calculation error",
    "E006": "Complex goods threshold exceeded",
    "E007": "Supplier not found",
    "E008": "Invalid EORI number",
    "E009": "Country not in EU",
    "E010": "Schema validation failed",
    "W001": "Import date outside quarter",
    "W002": "Emission factor outside typical range",
    "W003": "Supplier data older than 2 years",
    "W004": "Using default values (actual data preferred)",
    "W005": "Incomplete supplier data",
    "I001": "Report generated successfully",
    "I002": "Validation passed with warnings",
}

# EU Member States (EU27)
EU_MEMBER_STATES = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR",
    "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL",
    "PL", "PT", "RO", "SK", "SI", "ES", "SE"
}


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ValidationIssue(BaseModel):
    """Represents a validation error or warning."""
    shipment_id: Optional[str] = None
    error_code: str
    severity: str  # "error", "warning", "info"
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    suggestion: Optional[str] = None


class EnrichmentData(BaseModel):
    """Additional metadata added during enrichment."""
    product_group: Optional[str] = None
    product_description: Optional[str] = None
    supplier_found: bool = False
    supplier_name: Optional[str] = None
    validation_status: str = "valid"  # "valid", "invalid", "warning"


class ShipmentRecord(BaseModel):
    """Represents a validated and enriched shipment record."""
    # Required fields
    shipment_id: str
    import_date: str
    quarter: str
    cn_code: str
    origin_iso: str
    net_mass_kg: float

    # Optional fields
    product_group: Optional[str] = None
    product_description: Optional[str] = None
    origin_country: Optional[str] = None
    importer_name: Optional[str] = None
    importer_country: Optional[str] = None
    port_of_entry: Optional[str] = None
    importer_reference: Optional[str] = None
    has_actual_emissions: Optional[str] = "NO"
    supplier_id: Optional[str] = None
    notes: Optional[str] = None

    # Enrichment (added by agent)
    _enrichment: Optional[EnrichmentData] = None


# ============================================================================
# SHIPMENT INTAKE AGENT
# ============================================================================

class ShipmentIntakeAgent:
    """
    Ingests, validates, and enriches shipment data for CBAM reporting.

    This agent follows a tool-first architecture with ZERO LLM usage
    for validation and enrichment (LLM only for error messages).

    Performance target: 1000 shipments/second
    Test coverage target: 90%
    """

    def __init__(
        self,
        cn_codes_path: Union[str, Path],
        cbam_rules_path: Union[str, Path],
        suppliers_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the ShipmentIntakeAgent.

        Args:
            cn_codes_path: Path to CN codes JSON file
            cbam_rules_path: Path to CBAM rules YAML file
            suppliers_path: Path to suppliers YAML file (optional)
            schema_path: Path to shipment JSON schema (optional)
        """
        self.cn_codes_path = Path(cn_codes_path)
        self.cbam_rules_path = Path(cbam_rules_path)
        self.suppliers_path = Path(suppliers_path) if suppliers_path else None
        self.schema_path = Path(schema_path) if schema_path else None

        # Load reference data
        self.cn_codes = self._load_cn_codes()
        self.cbam_rules = self._load_cbam_rules()
        self.suppliers = self._load_suppliers() if self.suppliers_path else {}
        self.schema = self._load_schema() if self.schema_path else None

        # Statistics
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"ShipmentIntakeAgent initialized with {len(self.cn_codes)} CN codes")

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    def _load_cn_codes(self) -> Dict[str, Any]:
        """Load CN codes mapping from JSON file."""
        try:
            with open(self.cn_codes_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Remove metadata
            if "_metadata" in data:
                del data["_metadata"]
            logger.info(f"Loaded {len(data)} CN codes")
            return data
        except Exception as e:
            logger.error(f"Failed to load CN codes: {e}")
            raise

    def _load_cbam_rules(self) -> Dict[str, Any]:
        """Load CBAM rules from YAML file."""
        try:
            with open(self.cbam_rules_path, 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            logger.info(f"Loaded CBAM rules")
            return rules
        except Exception as e:
            logger.error(f"Failed to load CBAM rules: {e}")
            raise

    def _load_suppliers(self) -> Dict[str, Any]:
        """Load suppliers from YAML file."""
        if not self.suppliers_path or not self.suppliers_path.exists():
            return {}

        try:
            with open(self.suppliers_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Convert to dict keyed by supplier_id
            suppliers_dict = {}
            if "suppliers" in data:
                for supplier in data["suppliers"]:
                    suppliers_dict[supplier["supplier_id"]] = supplier

            logger.info(f"Loaded {len(suppliers_dict)} suppliers")
            return suppliers_dict
        except Exception as e:
            logger.warning(f"Failed to load suppliers: {e}")
            return {}

    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load JSON schema for validation."""
        if not self.schema_path or not self.schema_path.exists():
            return None

        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            logger.info("Loaded shipment JSON schema")
            return schema
        except Exception as e:
            logger.warning(f"Failed to load schema: {e}")
            return None

    # ========================================================================
    # FILE INGESTION
    # ========================================================================

    def read_shipments(self, input_path: Union[str, Path]) -> pd.DataFrame:
        """
        Read shipments from file (CSV, JSON, Excel).

        Args:
            input_path: Path to input file

        Returns:
            DataFrame with shipment records

        Raises:
            ValueError: If file format not supported or file cannot be read
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        # Detect format by extension
        suffix = input_path.suffix.lower()

        try:
            if suffix == '.csv':
                df = pd.read_csv(input_path, encoding='utf-8')
            elif suffix == '.json':
                df = pd.read_json(input_path)
            elif suffix in ['.xlsx', '.xls']:
                df = pd.read_excel(input_path)
            elif suffix == '.tsv':
                df = pd.read_csv(input_path, sep='\t', encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"Read {len(df)} shipments from {input_path}")
            return df

        except UnicodeDecodeError:
            # Try alternative encoding
            logger.warning("UTF-8 encoding failed, trying Latin-1")
            if suffix == '.csv':
                df = pd.read_csv(input_path, encoding='latin-1')
                return df
            else:
                raise

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def validate_shipment(self, shipment: Dict[str, Any]) -> Tuple[bool, List[ValidationIssue]]:
        """
        Validate a single shipment record.

        Args:
            shipment: Shipment dictionary

        Returns:
            Tuple of (is_valid, list of validation issues)
        """
        issues = []

        # Required fields check
        required_fields = ["shipment_id", "import_date", "quarter", "cn_code", "origin_iso", "net_mass_kg"]
        for field in required_fields:
            if field not in shipment or shipment[field] is None or shipment[field] == "":
                issues.append(ValidationIssue(
                    shipment_id=shipment.get("shipment_id"),
                    error_code="E001",
                    severity="error",
                    message=f"Missing required field: {field}",
                    field=field,
                    suggestion="Ensure all required fields are populated"
                ))

        # If critical fields missing, return early
        if any(issue.field in ["shipment_id", "cn_code", "net_mass_kg"] for issue in issues):
            return False, issues

        # CN code validation
        cn_code = str(shipment.get("cn_code", ""))
        if not re.match(r'^\d{8}$', cn_code):
            issues.append(ValidationIssue(
                shipment_id=shipment.get("shipment_id"),
                error_code="E002",
                severity="error",
                message=f"Invalid CN code format: {cn_code} (must be 8 digits)",
                field="cn_code",
                value=cn_code
            ))
        elif cn_code not in self.cn_codes:
            issues.append(ValidationIssue(
                shipment_id=shipment.get("shipment_id"),
                error_code="E002",
                severity="error",
                message=f"CN code not recognized as CBAM-covered good: {cn_code}",
                field="cn_code",
                value=cn_code,
                suggestion="Check if this is a valid CBAM product code from Annex I"
            ))

        # Mass validation
        try:
            mass = float(shipment.get("net_mass_kg", 0))
            if mass <= 0:
                issues.append(ValidationIssue(
                    shipment_id=shipment.get("shipment_id"),
                    error_code="E004",
                    severity="error",
                    message=f"Mass must be positive: {mass}",
                    field="net_mass_kg",
                    value=mass
                ))
        except (ValueError, TypeError):
            issues.append(ValidationIssue(
                shipment_id=shipment.get("shipment_id"),
                error_code="E004",
                severity="error",
                message="Invalid mass value (must be a number)",
                field="net_mass_kg",
                value=shipment.get("net_mass_kg")
            ))

        # Date validation
        import_date = shipment.get("import_date")
        if import_date:
            try:
                parsed_date = pd.to_datetime(import_date)
                # Check if date is in quarter
                quarter = shipment.get("quarter", "")
                if quarter and not self._is_date_in_quarter(parsed_date, quarter):
                    issues.append(ValidationIssue(
                        shipment_id=shipment.get("shipment_id"),
                        error_code="W001",
                        severity="warning",
                        message=f"Import date {import_date} is outside quarter {quarter}",
                        field="import_date"
                    ))
            except:
                issues.append(ValidationIssue(
                    shipment_id=shipment.get("shipment_id"),
                    error_code="E003",
                    severity="error",
                    message=f"Invalid date format: {import_date}",
                    field="import_date",
                    value=import_date
                ))

        # Country code validation
        origin_iso = shipment.get("origin_iso", "")
        if origin_iso and not re.match(r'^[A-Z]{2}$', origin_iso):
            issues.append(ValidationIssue(
                shipment_id=shipment.get("shipment_id"),
                error_code="E009",
                severity="error",
                message=f"Invalid origin country code: {origin_iso}",
                field="origin_iso",
                value=origin_iso
            ))

        # EU importer check
        importer_country = shipment.get("importer_country", "")
        if importer_country and importer_country not in EU_MEMBER_STATES:
            issues.append(ValidationIssue(
                shipment_id=shipment.get("shipment_id"),
                error_code="E009",
                severity="error",
                message=f"Importer country {importer_country} is not an EU member state",
                field="importer_country",
                value=importer_country
            ))

        # Determine if valid (no errors, warnings OK)
        has_errors = any(issue.severity == "error" for issue in issues)
        is_valid = not has_errors

        return is_valid, issues

    def _is_date_in_quarter(self, date: pd.Timestamp, quarter: str) -> bool:
        """Check if date falls within the specified quarter."""
        try:
            # Parse quarter (e.g., "2025Q4")
            match = re.match(r'^(\d{4})Q([1-4])$', quarter)
            if not match:
                return True  # Don't fail on invalid quarter format

            year = int(match.group(1))
            q = int(match.group(2))

            # Determine quarter date range
            quarter_starts = {
                1: (1, 1),
                2: (4, 1),
                3: (7, 1),
                4: (10, 1)
            }
            quarter_ends = {
                1: (3, 31),
                2: (6, 30),
                3: (9, 30),
                4: (12, 31)
            }

            start_month, start_day = quarter_starts[q]
            end_month, end_day = quarter_ends[q]

            start_date = pd.Timestamp(year, start_month, start_day)
            end_date = pd.Timestamp(year, end_month, end_day)

            return start_date <= date <= end_date
        except:
            return True  # Don't fail validation on date check errors

    # ========================================================================
    # ENRICHMENT
    # ========================================================================

    def enrich_shipment(self, shipment: Dict[str, Any]) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Enrich shipment with CN code metadata and supplier information.

        Args:
            shipment: Shipment dictionary

        Returns:
            Tuple of (enriched shipment, list of warnings)
        """
        warnings = []
        enrichment = EnrichmentData()

        # CN code enrichment
        cn_code = str(shipment.get("cn_code", ""))
        if cn_code in self.cn_codes:
            cn_info = self.cn_codes[cn_code]
            enrichment.product_group = cn_info.get("product_group")
            enrichment.product_description = cn_info.get("description")

            # Also add to main shipment record
            if "product_group" not in shipment or not shipment["product_group"]:
                shipment["product_group"] = enrichment.product_group
            if "product_description" not in shipment or not shipment["product_description"]:
                shipment["product_description"] = enrichment.product_description

        # Supplier lookup
        supplier_id = shipment.get("supplier_id")
        if supplier_id and self.suppliers:
            if supplier_id in self.suppliers:
                supplier = self.suppliers[supplier_id]
                enrichment.supplier_found = True
                enrichment.supplier_name = supplier.get("company_name")

                # Check product group match
                supplier_groups = supplier.get("product_groups", [])
                shipment_group = enrichment.product_group
                if shipment_group and shipment_group not in supplier_groups:
                    warnings.append(ValidationIssue(
                        shipment_id=shipment.get("shipment_id"),
                        error_code="W005",
                        severity="warning",
                        message=f"Product group mismatch: shipment has {shipment_group}, supplier produces {supplier_groups}",
                        field="supplier_id"
                    ))
            else:
                warnings.append(ValidationIssue(
                    shipment_id=shipment.get("shipment_id"),
                    error_code="W005",
                    severity="warning",
                    message=f"Supplier {supplier_id} not found in supplier database",
                    field="supplier_id",
                    value=supplier_id
                ))

        # Add enrichment to shipment
        shipment["_enrichment"] = enrichment.dict()

        return shipment, warnings

    # ========================================================================
    # MAIN PROCESSING
    # ========================================================================

    def process(
        self,
        input_file: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Process shipments: read, validate, enrich, and output.

        Args:
            input_file: Path to input shipments file
            output_file: Path for output file (optional)

        Returns:
            Result dictionary with metadata, validated shipments, and errors
        """
        self.stats["start_time"] = DeterministicClock.now()

        # Read input
        df = self.read_shipments(input_file)
        self.stats["total_records"] = len(df)

        # Process each shipment
        validated_shipments = []
        all_errors = []

        for idx, row in df.iterrows():
            shipment = row.to_dict()

            # Validate
            is_valid, issues = self.validate_shipment(shipment)

            # Enrich
            if is_valid:
                shipment, warnings = self.enrich_shipment(shipment)
                issues.extend(warnings)

            # Track statistics
            if is_valid:
                self.stats["valid_records"] += 1
                if any(issue.severity == "warning" for issue in issues):
                    self.stats["warnings"] += 1
            else:
                self.stats["invalid_records"] += 1

            # Add validation status to enrichment
            if "_enrichment" in shipment:
                shipment["_enrichment"]["validation_status"] = "valid" if is_valid else "invalid"

            validated_shipments.append(shipment)
            all_errors.extend([issue.dict() for issue in issues])

        self.stats["end_time"] = DeterministicClock.now()
        processing_time = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()

        # Build result
        result = {
            "metadata": {
                "processed_at": self.stats["end_time"].isoformat(),
                "input_file": str(input_file),
                "total_records": self.stats["total_records"],
                "valid_records": self.stats["valid_records"],
                "invalid_records": self.stats["invalid_records"],
                "warnings": self.stats["warnings"],
                "processing_time_seconds": processing_time,
                "records_per_second": self.stats["total_records"] / processing_time if processing_time > 0 else 0
            },
            "shipments": validated_shipments,
            "validation_errors": all_errors
        }

        # Write output if path provided
        if output_file:
            self.write_output(result, output_file)

        logger.info(f"Processed {self.stats['total_records']} shipments in {processing_time:.2f}s "
                   f"({result['metadata']['records_per_second']:.0f} records/sec)")
        logger.info(f"Valid: {self.stats['valid_records']}, Invalid: {self.stats['invalid_records']}, "
                   f"Warnings: {self.stats['warnings']}")

        return result

    def write_output(self, result: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """Write result to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Wrote output to {output_path}")

    def get_validation_report(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a validation summary report."""
        errors_by_code = {}
        for error in result["validation_errors"]:
            code = error["error_code"]
            if code not in errors_by_code:
                errors_by_code[code] = {
                    "code": code,
                    "description": ERROR_CODES.get(code, "Unknown error"),
                    "count": 0,
                    "severity": error["severity"]
                }
            errors_by_code[code]["count"] += 1

        return {
            "summary": result["metadata"],
            "errors_by_code": list(errors_by_code.values()),
            "is_ready_for_next_stage": result["metadata"]["invalid_records"] == 0
        }


# ============================================================================
# CLI INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CBAM Shipment Intake Agent")
    parser.add_argument("--input", required=True, help="Input shipments file (CSV/JSON/Excel)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--cn-codes", required=True, help="Path to CN codes JSON")
    parser.add_argument("--rules", required=True, help="Path to CBAM rules YAML")
    parser.add_argument("--suppliers", help="Path to suppliers YAML (optional)")
    parser.add_argument("--schema", help="Path to shipment JSON schema (optional)")

    args = parser.parse_args()

    # Create agent
    agent = ShipmentIntakeAgent(
        cn_codes_path=args.cn_codes,
        cbam_rules_path=args.rules,
        suppliers_path=args.suppliers,
        schema_path=args.schema
    )

    # Process
    result = agent.process(
        input_file=args.input,
        output_file=args.output
    )

    # Print report
    report = agent.get_validation_report(result)
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    print(f"Total Records: {report['summary']['total_records']}")
    print(f"Valid: {report['summary']['valid_records']}")
    print(f"Invalid: {report['summary']['invalid_records']}")
    print(f"Warnings: {report['summary']['warnings']}")
    print(f"Ready for next stage: {report['is_ready_for_next_stage']}")

    if report['errors_by_code']:
        print("\nErrors/Warnings by Code:")
        for error_summary in report['errors_by_code']:
            print(f"  {error_summary['code']} ({error_summary['severity']}): "
                  f"{error_summary['description']} - Count: {error_summary['count']}")
