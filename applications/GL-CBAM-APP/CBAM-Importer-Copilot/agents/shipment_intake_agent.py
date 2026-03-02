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

Version: 1.1.0
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

# v1.1: Supplier Portal Integration (optional dependency)
try:
    from ..supplier_portal.supplier_registry import SupplierRegistryEngine
    from ..supplier_portal.emissions_submission import EmissionsSubmissionEngine
    from ..supplier_portal.data_exchange import DataExchangeService
    SUPPLIER_PORTAL_AVAILABLE = True
except ImportError:
    SUPPLIER_PORTAL_AVAILABLE = False

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
    supplier_installation_id: Optional[str] = None
    data_source: str = "default_value"  # "supplier_actual", "regional_factor", "default_value"
    supplier_verified: bool = False
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
    supplier_installation_id: Optional[str] = None
    data_source: Optional[str] = "default_value"
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
        schema_path: Optional[Union[str, Path]] = None,
        supplier_portal_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the ShipmentIntakeAgent.

        Args:
            cn_codes_path: Path to CN codes JSON file
            cbam_rules_path: Path to CBAM rules YAML file
            suppliers_path: Path to suppliers YAML file (optional)
            schema_path: Path to shipment JSON schema (optional)
            supplier_portal_config: v1.1 supplier portal configuration (optional)
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

        # v1.1: Initialize supplier portal engines (if available)
        self.supplier_portal_config = supplier_portal_config or {}
        self.supplier_portal_enabled = (
            SUPPLIER_PORTAL_AVAILABLE
            and self.supplier_portal_config.get("enabled", False)
        )
        self._init_supplier_portal()

        # Statistics
        self.stats = {
            "total_records": 0,
            "valid_records": 0,
            "invalid_records": 0,
            "warnings": 0,
            "supplier_linked": 0,
            "supplier_enriched": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"ShipmentIntakeAgent initialized with {len(self.cn_codes)} CN codes")
        if self.supplier_portal_enabled:
            logger.info("Supplier portal integration enabled (v1.1)")

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
    # v1.1: SUPPLIER PORTAL INTEGRATION
    # ========================================================================

    def _init_supplier_portal(self) -> None:
        """Initialize supplier portal engines if available and enabled."""
        self.supplier_registry = None
        self.emissions_submission = None
        self.data_exchange = None

        if not self.supplier_portal_enabled:
            return

        try:
            self.supplier_registry = SupplierRegistryEngine(
                max_installations=self.supplier_portal_config.get(
                    "max_installations_per_supplier", 50
                )
            )
            self.emissions_submission = EmissionsSubmissionEngine(
                retention_years=self.supplier_portal_config.get(
                    "emissions_data_retention_years", 7
                )
            )
            self.data_exchange = DataExchangeService()
            logger.info("Supplier portal engines initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize supplier portal engines: {e}")
            self.supplier_portal_enabled = False

    def enrich_with_supplier_data(
        self,
        shipment: Dict[str, Any],
        supplier_id: str
    ) -> Tuple[Dict[str, Any], List[ValidationIssue]]:
        """
        Enrich shipment with verified supplier emissions data from the portal.

        Prefers supplier actual data over default emission factors.
        Tracks the data source for audit trail (supplier_actual vs
        regional_factor vs default_value).

        Args:
            shipment: Shipment dictionary to enrich
            supplier_id: Supplier identifier to look up

        Returns:
            Tuple of (enriched shipment, list of validation issues)
        """
        issues = []
        shipment_id = shipment.get("shipment_id", "UNKNOWN")

        if not self.supplier_portal_enabled or not self.supplier_registry:
            return shipment, issues

        try:
            # Step 1: Look up supplier in registry
            supplier_record = self.supplier_registry.get_supplier(supplier_id)
            if not supplier_record:
                issues.append(ValidationIssue(
                    shipment_id=shipment_id,
                    error_code="W005",
                    severity="warning",
                    message=f"Supplier {supplier_id} not found in portal registry",
                    field="supplier_id",
                    value=supplier_id
                ))
                return shipment, issues

            # Step 2: Validate supplier EORI if strict mode
            eori_mode = self.supplier_portal_config.get("eori_validation", "strict")
            supplier_eori = supplier_record.get("eori_number", "")
            if eori_mode == "strict" and not re.match(r'^[A-Z]{2}[A-Z0-9]{1,15}$', supplier_eori):
                issues.append(ValidationIssue(
                    shipment_id=shipment_id,
                    error_code="E008",
                    severity="error",
                    message=f"Supplier {supplier_id} has invalid EORI: {supplier_eori}",
                    field="supplier_eori",
                    value=supplier_eori
                ))

            # Step 3: Find the installation matching this shipment
            installation_id = shipment.get("supplier_installation_id")
            if not installation_id:
                installation_id = self._match_installation(
                    supplier_record, shipment
                )

            if not installation_id:
                issues.append(ValidationIssue(
                    shipment_id=shipment_id,
                    error_code="W005",
                    severity="warning",
                    message=f"No matching installation found for supplier {supplier_id}",
                    field="supplier_installation_id"
                ))
                return shipment, issues

            # Step 4: Retrieve verified emissions data for installation
            emissions_data = self.emissions_submission.get_verified_emissions(
                supplier_id=supplier_id,
                installation_id=installation_id
            )

            if emissions_data and emissions_data.get("verified", False):
                # Supplier actual verified data is highest quality
                shipment["has_actual_emissions"] = "YES"
                shipment["supplier_installation_id"] = installation_id
                shipment["data_source"] = "supplier_actual"

                # Check verification expiry
                expiry_alert_days = self.supplier_portal_config.get(
                    "verification_expiry_alert_days", 90
                )
                verification_date = emissions_data.get("verification_date")
                if verification_date:
                    try:
                        v_date = pd.to_datetime(verification_date)
                        days_since = (pd.Timestamp.now() - v_date).days
                        if days_since > 365:
                            issues.append(ValidationIssue(
                                shipment_id=shipment_id,
                                error_code="W003",
                                severity="warning",
                                message=f"Supplier verification is {days_since} days old",
                                field="verification_date",
                                value=verification_date,
                                suggestion="Request updated verification from supplier"
                            ))
                        elif days_since > (365 - expiry_alert_days):
                            issues.append(ValidationIssue(
                                shipment_id=shipment_id,
                                error_code="W003",
                                severity="warning",
                                message=f"Supplier verification expires in {365 - days_since} days",
                                field="verification_date",
                                value=verification_date
                            ))
                    except Exception:
                        pass

                self.stats["supplier_enriched"] += 1
                logger.debug(
                    f"Shipment {shipment_id} enriched with verified supplier data "
                    f"from installation {installation_id}"
                )

            elif emissions_data and not emissions_data.get("verified", False):
                # Supplier actual but unverified
                shipment["has_actual_emissions"] = "YES"
                shipment["supplier_installation_id"] = installation_id
                shipment["data_source"] = "supplier_actual"

                issues.append(ValidationIssue(
                    shipment_id=shipment_id,
                    error_code="W004",
                    severity="warning",
                    message="Supplier emissions data is unverified",
                    field="data_source",
                    suggestion="Request third-party verification from supplier"
                ))
                self.stats["supplier_enriched"] += 1
            else:
                # No emissions data from portal, fall back to default
                shipment["data_source"] = "default_value"
                issues.append(ValidationIssue(
                    shipment_id=shipment_id,
                    error_code="W004",
                    severity="warning",
                    message=f"No emissions data from supplier portal for {supplier_id}",
                    field="data_source",
                    suggestion="Using default emission factors"
                ))

        except Exception as e:
            logger.error(f"Supplier portal enrichment failed for {shipment_id}: {e}")
            issues.append(ValidationIssue(
                shipment_id=shipment_id,
                error_code="W005",
                severity="warning",
                message=f"Supplier portal lookup failed: {str(e)}",
                field="supplier_id"
            ))

        return shipment, issues

    def _match_installation(
        self,
        supplier_record: Dict[str, Any],
        shipment: Dict[str, Any]
    ) -> Optional[str]:
        """
        Match a supplier installation to a shipment based on country and CN code.

        Args:
            supplier_record: Supplier record from the registry
            shipment: Shipment dictionary

        Returns:
            Installation ID if matched, None otherwise
        """
        installations = supplier_record.get("installations", [])
        origin_iso = shipment.get("origin_iso", "")
        cn_code = str(shipment.get("cn_code", ""))

        for installation in installations:
            inst_country = installation.get("country_iso", "")
            inst_cn_codes = installation.get("cn_codes_produced", [])

            if inst_country == origin_iso and cn_code in inst_cn_codes:
                return installation.get("installation_id")

        # Fallback: match by country only if single installation in that country
        country_matches = [
            inst for inst in installations
            if inst.get("country_iso") == origin_iso
        ]
        if len(country_matches) == 1:
            return country_matches[0].get("installation_id")

        return None

    def link_shipment_to_supplier(
        self,
        shipment: Dict[str, Any]
    ) -> Optional[str]:
        """
        Auto-link a shipment to a supplier installation.

        Matches based on origin country + CN code + installation name.
        Uses the supplier portal registry when available, falls back to
        local suppliers YAML otherwise.

        Args:
            shipment: Shipment dictionary

        Returns:
            Supplier ID if matched, None otherwise
        """
        origin_iso = shipment.get("origin_iso", "")
        cn_code = str(shipment.get("cn_code", ""))
        installation_name = shipment.get("installation_name", "")

        # v1.1: Try supplier portal registry first
        if self.supplier_portal_enabled and self.supplier_registry:
            try:
                match = self.supplier_registry.find_supplier_by_installation(
                    country_iso=origin_iso,
                    cn_code=cn_code,
                    installation_name=installation_name
                )
                if match:
                    self.stats["supplier_linked"] += 1
                    logger.debug(
                        f"Auto-linked shipment to supplier {match} "
                        f"via portal (country={origin_iso}, cn={cn_code})"
                    )
                    return match
            except Exception as e:
                logger.warning(f"Portal supplier linking failed: {e}")

        # Fallback: try local suppliers YAML
        for supplier_id, supplier in self.suppliers.items():
            supplier_country = supplier.get("country", "")
            supplier_cn_codes = supplier.get("cn_codes_produced", [])
            supplier_installation = supplier.get("installation_name", "")

            # Match by country + CN code
            if supplier_country == origin_iso and cn_code in supplier_cn_codes:
                # If installation name provided, also match on that
                if installation_name and supplier_installation:
                    if installation_name.lower() == supplier_installation.lower():
                        self.stats["supplier_linked"] += 1
                        return supplier_id
                else:
                    self.stats["supplier_linked"] += 1
                    return supplier_id

        return None

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

                # v1.1: Supplier portal integration
                if self.supplier_portal_enabled:
                    # Auto-link to supplier if not already linked
                    auto_link = self.supplier_portal_config.get(
                        "auto_link_suppliers", True
                    )
                    if auto_link and not shipment.get("supplier_id"):
                        linked_supplier = self.link_shipment_to_supplier(shipment)
                        if linked_supplier:
                            shipment["supplier_id"] = linked_supplier

                    # Enrich with supplier portal data
                    supplier_id = shipment.get("supplier_id")
                    if supplier_id:
                        shipment, portal_issues = self.enrich_with_supplier_data(
                            shipment, supplier_id
                        )
                        issues.extend(portal_issues)

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
                "supplier_linked": self.stats["supplier_linked"],
                "supplier_enriched": self.stats["supplier_enriched"],
                "supplier_portal_enabled": self.supplier_portal_enabled,
                "processing_time_seconds": processing_time,
                "records_per_second": self.stats["total_records"] / processing_time if processing_time > 0 else 0,
                "agent_version": "1.1.0"
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
    parser.add_argument("--portal-config", help="Path to supplier portal config YAML (optional)")

    args = parser.parse_args()

    # Load supplier portal config if provided
    portal_config = None
    if args.portal_config:
        with open(args.portal_config, 'r', encoding='utf-8') as f:
            portal_config = yaml.safe_load(f).get("supplier_portal", {})

    # Create agent
    agent = ShipmentIntakeAgent(
        cn_codes_path=args.cn_codes,
        cbam_rules_path=args.rules,
        suppliers_path=args.suppliers,
        schema_path=args.schema,
        supplier_portal_config=portal_config
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
