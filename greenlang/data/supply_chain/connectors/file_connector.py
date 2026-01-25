"""
File Connector for CSV/Excel Import and Export.

This module provides file-based data import and export functionality
for supply chain data, supporting:
- CSV import/export
- Excel import/export (xlsx)
- Template generation
- Data validation
- Batch processing

Example:
    >>> from greenlang.supply_chain.connectors import FileConnector
    >>> connector = FileConnector()
    >>>
    >>> # Import suppliers from CSV
    >>> result = connector.import_suppliers_csv("suppliers.csv")
    >>> print(f"Imported {result.success_count} suppliers")
    >>>
    >>> # Export to Excel
    >>> connector.export_suppliers_excel(suppliers, "export.xlsx")
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, BinaryIO, TextIO, Union
import io

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    Material,
    SupplierRelationship,
    Address,
    ExternalIdentifiers,
    GeoLocation,
    SupplierTier,
    SupplierStatus,
    RelationshipType,
    CommodityType,
)

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    EXCEL = "xlsx"
    JSON = "json"


class ImportStatus(Enum):
    """Import record status."""
    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ImportError:
    """
    Import error details.

    Attributes:
        row: Row number (1-indexed)
        column: Column name
        value: Original value
        error: Error message
        severity: Error severity
    """
    row: int
    column: str
    value: Any
    error: str
    severity: str = "error"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row": self.row,
            "column": self.column,
            "value": self.value,
            "error": self.error,
            "severity": self.severity,
        }


@dataclass
class ImportResult:
    """
    Result of an import operation.

    Attributes:
        total_rows: Total rows processed
        success_count: Successfully imported
        error_count: Failed imports
        warning_count: Imports with warnings
        skipped_count: Skipped rows
        errors: List of errors
        warnings: List of warnings
        imported_records: Successfully imported records
    """
    total_rows: int = 0
    success_count: int = 0
    error_count: int = 0
    warning_count: int = 0
    skipped_count: int = 0
    errors: List[ImportError] = field(default_factory=list)
    warnings: List[ImportError] = field(default_factory=list)
    imported_records: List[Any] = field(default_factory=list)
    import_time: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_rows == 0:
            return 0.0
        return self.success_count / self.total_rows

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_rows": self.total_rows,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "skipped_count": self.skipped_count,
            "success_rate": self.success_rate,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "import_time": self.import_time.isoformat(),
        }


class FileConnector:
    """
    File-based data import and export connector.

    Supports CSV and Excel file formats for importing and exporting
    supply chain data with validation and error handling.

    Example:
        >>> connector = FileConnector()
        >>>
        >>> # Import suppliers
        >>> result = connector.import_suppliers_csv("suppliers.csv")
        >>> for supplier in result.imported_records:
        ...     graph.add_supplier(supplier)
        >>>
        >>> # Export with custom template
        >>> connector.export_suppliers_csv(
        ...     suppliers,
        ...     "export.csv",
        ...     columns=["id", "name", "country_code", "tier"]
        ... )
    """

    # Default column mappings for CSV import
    SUPPLIER_COLUMN_MAPPING = {
        "id": ["id", "supplier_id", "vendor_id", "code"],
        "name": ["name", "supplier_name", "company_name", "vendor_name"],
        "tier": ["tier", "supplier_tier", "tier_level"],
        "country_code": ["country_code", "country", "iso_country"],
        "street": ["street", "address", "street_address", "address_1"],
        "city": ["city", "town"],
        "state": ["state", "province", "region", "state_province"],
        "postal_code": ["postal_code", "zip", "zip_code", "postcode"],
        "lei": ["lei", "legal_entity_identifier"],
        "duns": ["duns", "duns_number", "d_u_n_s"],
        "vat_number": ["vat", "vat_number", "tax_id", "tax_number"],
        "contact_name": ["contact_name", "contact", "primary_contact"],
        "contact_email": ["contact_email", "email"],
        "contact_phone": ["contact_phone", "phone", "telephone"],
        "annual_spend": ["annual_spend", "spend", "annual_value"],
        "currency": ["currency", "currency_code"],
        "industry_code": ["industry_code", "naics", "sic_code", "nace"],
    }

    FACILITY_COLUMN_MAPPING = {
        "id": ["id", "facility_id", "site_id"],
        "name": ["name", "facility_name", "site_name"],
        "supplier_id": ["supplier_id", "parent_supplier", "vendor_id"],
        "facility_type": ["facility_type", "type", "site_type"],
        "latitude": ["latitude", "lat"],
        "longitude": ["longitude", "lon", "lng"],
        "street": ["street", "address"],
        "city": ["city"],
        "country_code": ["country_code", "country"],
    }

    RELATIONSHIP_COLUMN_MAPPING = {
        "id": ["id", "relationship_id"],
        "source_supplier_id": ["source_id", "supplier_id", "from_supplier"],
        "target_supplier_id": ["target_id", "customer_id", "to_supplier"],
        "relationship_type": ["type", "relationship_type", "relation"],
        "annual_spend": ["annual_spend", "spend", "value"],
        "currency": ["currency"],
    }

    def __init__(
        self,
        encoding: str = "utf-8",
        delimiter: str = ",",
        date_format: str = "%Y-%m-%d",
    ):
        """
        Initialize the file connector.

        Args:
            encoding: File encoding
            delimiter: CSV delimiter
            date_format: Date parsing format
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.date_format = date_format

        logger.info("FileConnector initialized")

    def _find_column(
        self,
        headers: List[str],
        mapping: List[str]
    ) -> Optional[int]:
        """
        Find column index from mapping.

        Args:
            headers: List of header names
            mapping: List of possible column names

        Returns:
            Column index or None if not found
        """
        headers_lower = [h.lower().strip() for h in headers]
        for name in mapping:
            if name.lower() in headers_lower:
                return headers_lower.index(name.lower())
        return None

    def _parse_tier(self, value: str) -> SupplierTier:
        """Parse tier value to SupplierTier enum."""
        if not value:
            return SupplierTier.UNKNOWN

        value_lower = value.lower().strip()
        if value_lower in ["1", "tier_1", "tier1", "t1"]:
            return SupplierTier.TIER_1
        elif value_lower in ["2", "tier_2", "tier2", "t2"]:
            return SupplierTier.TIER_2
        elif value_lower in ["3", "tier_3", "tier3", "t3"]:
            return SupplierTier.TIER_3
        elif value_lower in ["n", "tier_n", "tiern", "tn"]:
            return SupplierTier.TIER_N
        return SupplierTier.UNKNOWN

    def _parse_decimal(self, value: str) -> Optional[Decimal]:
        """Parse string to Decimal."""
        if not value or value.strip() == "":
            return None
        try:
            # Remove currency symbols and commas
            cleaned = value.replace("$", "").replace("EUR", "").replace(",", "").strip()
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    def _parse_date(self, value: str) -> Optional[date]:
        """Parse string to date."""
        if not value or value.strip() == "":
            return None
        try:
            return datetime.strptime(value.strip(), self.date_format).date()
        except ValueError:
            return None

    # =========================================================================
    # Supplier Import/Export
    # =========================================================================

    def import_suppliers_csv(
        self,
        file_path: Union[str, Path, TextIO],
        column_mapping: Optional[Dict[str, List[str]]] = None,
        skip_header: bool = True,
        validate: bool = True,
    ) -> ImportResult:
        """
        Import suppliers from CSV file.

        Args:
            file_path: Path to CSV file or file-like object
            column_mapping: Custom column mapping
            skip_header: Whether first row is header
            validate: Whether to validate data

        Returns:
            ImportResult with imported suppliers
        """
        result = ImportResult()
        mapping = column_mapping or self.SUPPLIER_COLUMN_MAPPING

        # Handle file-like object or path
        if isinstance(file_path, (str, Path)):
            file_handle = open(file_path, "r", encoding=self.encoding, newline="")
            should_close = True
        else:
            file_handle = file_path
            should_close = False

        try:
            reader = csv.reader(file_handle, delimiter=self.delimiter)

            # Read headers
            headers = next(reader, [])
            if not headers:
                result.errors.append(ImportError(
                    row=1, column="", value="", error="Empty file or no headers"
                ))
                return result

            # Build column index map
            col_map = {}
            for field, names in mapping.items():
                idx = self._find_column(headers, names)
                if idx is not None:
                    col_map[field] = idx

            # Process rows
            for row_num, row in enumerate(reader, start=2):
                result.total_rows += 1

                try:
                    supplier = self._parse_supplier_row(row, col_map, row_num, result)
                    if supplier:
                        if validate:
                            valid, validation_errors = self._validate_supplier(supplier)
                            if not valid:
                                for error in validation_errors:
                                    result.warnings.append(ImportError(
                                        row=row_num,
                                        column=error[0],
                                        value=error[1],
                                        error=error[2],
                                        severity="warning"
                                    ))
                                result.warning_count += 1

                        result.imported_records.append(supplier)
                        result.success_count += 1

                except Exception as e:
                    result.errors.append(ImportError(
                        row=row_num,
                        column="",
                        value=str(row),
                        error=str(e)
                    ))
                    result.error_count += 1

        finally:
            if should_close:
                file_handle.close()

        logger.info(
            f"Imported {result.success_count}/{result.total_rows} suppliers"
        )
        return result

    def _parse_supplier_row(
        self,
        row: List[str],
        col_map: Dict[str, int],
        row_num: int,
        result: ImportResult,
    ) -> Optional[Supplier]:
        """Parse a CSV row into a Supplier entity."""

        def get_value(field: str) -> str:
            idx = col_map.get(field)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return ""

        # Required fields
        supplier_id = get_value("id")
        name = get_value("name")

        if not name:
            result.errors.append(ImportError(
                row=row_num,
                column="name",
                value="",
                error="Supplier name is required"
            ))
            result.error_count += 1
            return None

        # Generate ID if not provided
        if not supplier_id:
            supplier_id = f"IMP-{row_num:06d}"

        # Build address
        address = None
        street = get_value("street")
        city = get_value("city")
        if street or city:
            address = Address(
                street_line_1=street or "",
                city=city or "",
                state_province=get_value("state"),
                postal_code=get_value("postal_code"),
                country_code=get_value("country_code") or "",
            )

        # Build external identifiers
        external_ids = ExternalIdentifiers(
            lei=get_value("lei") or None,
            duns=get_value("duns") or None,
            vat_number=get_value("vat_number") or None,
        )

        # Industry codes
        industry_codes = {}
        industry_code = get_value("industry_code")
        if industry_code:
            industry_codes["NAICS"] = industry_code

        return Supplier(
            id=supplier_id,
            name=name,
            tier=self._parse_tier(get_value("tier")),
            country_code=get_value("country_code") or None,
            address=address,
            external_ids=external_ids,
            annual_spend=self._parse_decimal(get_value("annual_spend")),
            currency=get_value("currency") or "USD",
            industry_codes=industry_codes,
        )

    def _validate_supplier(
        self,
        supplier: Supplier
    ) -> Tuple[bool, List[Tuple[str, Any, str]]]:
        """Validate supplier data."""
        errors = []

        # Validate country code
        if supplier.country_code and len(supplier.country_code) != 2:
            errors.append((
                "country_code",
                supplier.country_code,
                "Country code should be 2 characters (ISO 3166-1 alpha-2)"
            ))

        # Validate LEI format
        if supplier.external_ids.lei and len(supplier.external_ids.lei) != 20:
            errors.append((
                "lei",
                supplier.external_ids.lei,
                "LEI should be 20 characters"
            ))

        # Validate DUNS format
        if supplier.external_ids.duns:
            duns = supplier.external_ids.duns.replace("-", "")
            if len(duns) != 9 or not duns.isdigit():
                errors.append((
                    "duns",
                    supplier.external_ids.duns,
                    "DUNS should be 9 digits"
                ))

        return len(errors) == 0, errors

    def export_suppliers_csv(
        self,
        suppliers: List[Supplier],
        file_path: Union[str, Path, TextIO],
        columns: Optional[List[str]] = None,
    ) -> int:
        """
        Export suppliers to CSV file.

        Args:
            suppliers: List of suppliers to export
            file_path: Output file path or file-like object
            columns: Specific columns to export (None for all)

        Returns:
            Number of suppliers exported
        """
        default_columns = [
            "id", "name", "tier", "country_code", "status",
            "street", "city", "state", "postal_code",
            "lei", "duns", "vat_number",
            "annual_spend", "currency",
            "contact_name", "contact_email",
        ]
        columns = columns or default_columns

        # Handle file-like object or path
        if isinstance(file_path, (str, Path)):
            file_handle = open(file_path, "w", encoding=self.encoding, newline="")
            should_close = True
        else:
            file_handle = file_path
            should_close = False

        try:
            writer = csv.writer(file_handle, delimiter=self.delimiter)

            # Write header
            writer.writerow(columns)

            # Write data
            for supplier in suppliers:
                row = self._supplier_to_row(supplier, columns)
                writer.writerow(row)

        finally:
            if should_close:
                file_handle.close()

        logger.info(f"Exported {len(suppliers)} suppliers to CSV")
        return len(suppliers)

    def _supplier_to_row(
        self,
        supplier: Supplier,
        columns: List[str]
    ) -> List[str]:
        """Convert supplier to CSV row."""
        row = []
        for col in columns:
            if col == "id":
                row.append(supplier.id)
            elif col == "name":
                row.append(supplier.name)
            elif col == "tier":
                row.append(str(supplier.tier.value))
            elif col == "country_code":
                row.append(supplier.country_code or "")
            elif col == "status":
                row.append(supplier.status.value)
            elif col == "street":
                row.append(supplier.address.street_line_1 if supplier.address else "")
            elif col == "city":
                row.append(supplier.address.city if supplier.address else "")
            elif col == "state":
                row.append(supplier.address.state_province if supplier.address else "")
            elif col == "postal_code":
                row.append(supplier.address.postal_code if supplier.address else "")
            elif col == "lei":
                row.append(supplier.external_ids.lei or "")
            elif col == "duns":
                row.append(supplier.external_ids.duns or "")
            elif col == "vat_number":
                row.append(supplier.external_ids.vat_number or "")
            elif col == "annual_spend":
                row.append(str(supplier.annual_spend) if supplier.annual_spend else "")
            elif col == "currency":
                row.append(supplier.currency)
            elif col == "contact_name":
                row.append(supplier.contact.primary_contact_name if supplier.contact else "")
            elif col == "contact_email":
                row.append(supplier.contact.primary_email if supplier.contact else "")
            else:
                row.append("")
        return row

    # =========================================================================
    # Facility Import/Export
    # =========================================================================

    def import_facilities_csv(
        self,
        file_path: Union[str, Path, TextIO],
        column_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> ImportResult:
        """
        Import facilities from CSV file.

        Args:
            file_path: Path to CSV file
            column_mapping: Custom column mapping

        Returns:
            ImportResult with imported facilities
        """
        result = ImportResult()
        mapping = column_mapping or self.FACILITY_COLUMN_MAPPING

        if isinstance(file_path, (str, Path)):
            file_handle = open(file_path, "r", encoding=self.encoding, newline="")
            should_close = True
        else:
            file_handle = file_path
            should_close = False

        try:
            reader = csv.reader(file_handle, delimiter=self.delimiter)
            headers = next(reader, [])

            col_map = {}
            for field, names in mapping.items():
                idx = self._find_column(headers, names)
                if idx is not None:
                    col_map[field] = idx

            for row_num, row in enumerate(reader, start=2):
                result.total_rows += 1

                try:
                    facility = self._parse_facility_row(row, col_map, row_num, result)
                    if facility:
                        result.imported_records.append(facility)
                        result.success_count += 1
                except Exception as e:
                    result.errors.append(ImportError(
                        row=row_num, column="", value=str(row), error=str(e)
                    ))
                    result.error_count += 1

        finally:
            if should_close:
                file_handle.close()

        return result

    def _parse_facility_row(
        self,
        row: List[str],
        col_map: Dict[str, int],
        row_num: int,
        result: ImportResult,
    ) -> Optional[Facility]:
        """Parse a CSV row into a Facility entity."""

        def get_value(field: str) -> str:
            idx = col_map.get(field)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return ""

        facility_id = get_value("id") or f"FAC-{row_num:06d}"
        name = get_value("name")
        supplier_id = get_value("supplier_id")

        if not name or not supplier_id:
            result.errors.append(ImportError(
                row=row_num,
                column="name/supplier_id",
                value="",
                error="Facility name and supplier_id are required"
            ))
            result.error_count += 1
            return None

        # Parse location
        location = None
        lat_str = get_value("latitude")
        lon_str = get_value("longitude")
        if lat_str and lon_str:
            try:
                location = GeoLocation(
                    latitude=float(lat_str),
                    longitude=float(lon_str)
                )
            except ValueError:
                result.warnings.append(ImportError(
                    row=row_num,
                    column="latitude/longitude",
                    value=f"{lat_str},{lon_str}",
                    error="Invalid coordinates",
                    severity="warning"
                ))

        # Parse address
        address = None
        street = get_value("street")
        city = get_value("city")
        if street or city:
            address = Address(
                street_line_1=street or "",
                city=city or "",
                country_code=get_value("country_code") or "",
            )

        return Facility(
            id=facility_id,
            name=name,
            supplier_id=supplier_id,
            facility_type=get_value("facility_type") or "production",
            location=location,
            address=address,
        )

    # =========================================================================
    # Relationship Import/Export
    # =========================================================================

    def import_relationships_csv(
        self,
        file_path: Union[str, Path, TextIO],
        column_mapping: Optional[Dict[str, List[str]]] = None,
    ) -> ImportResult:
        """
        Import supplier relationships from CSV file.

        Args:
            file_path: Path to CSV file
            column_mapping: Custom column mapping

        Returns:
            ImportResult with imported relationships
        """
        result = ImportResult()
        mapping = column_mapping or self.RELATIONSHIP_COLUMN_MAPPING

        if isinstance(file_path, (str, Path)):
            file_handle = open(file_path, "r", encoding=self.encoding, newline="")
            should_close = True
        else:
            file_handle = file_path
            should_close = False

        try:
            reader = csv.reader(file_handle, delimiter=self.delimiter)
            headers = next(reader, [])

            col_map = {}
            for field, names in mapping.items():
                idx = self._find_column(headers, names)
                if idx is not None:
                    col_map[field] = idx

            for row_num, row in enumerate(reader, start=2):
                result.total_rows += 1

                try:
                    relationship = self._parse_relationship_row(
                        row, col_map, row_num, result
                    )
                    if relationship:
                        result.imported_records.append(relationship)
                        result.success_count += 1
                except Exception as e:
                    result.errors.append(ImportError(
                        row=row_num, column="", value=str(row), error=str(e)
                    ))
                    result.error_count += 1

        finally:
            if should_close:
                file_handle.close()

        return result

    def _parse_relationship_row(
        self,
        row: List[str],
        col_map: Dict[str, int],
        row_num: int,
        result: ImportResult,
    ) -> Optional[SupplierRelationship]:
        """Parse a CSV row into a SupplierRelationship entity."""

        def get_value(field: str) -> str:
            idx = col_map.get(field)
            if idx is not None and idx < len(row):
                return row[idx].strip()
            return ""

        source_id = get_value("source_supplier_id")
        target_id = get_value("target_supplier_id")

        if not source_id or not target_id:
            result.errors.append(ImportError(
                row=row_num,
                column="source/target",
                value="",
                error="Source and target supplier IDs are required"
            ))
            result.error_count += 1
            return None

        # Parse relationship type
        rel_type_str = get_value("relationship_type").lower()
        try:
            rel_type = RelationshipType(rel_type_str) if rel_type_str else RelationshipType.SUPPLIER
        except ValueError:
            rel_type = RelationshipType.SUPPLIER

        return SupplierRelationship(
            id=get_value("id") or f"REL-{row_num:06d}",
            source_supplier_id=source_id,
            target_supplier_id=target_id,
            relationship_type=rel_type,
            annual_spend=self._parse_decimal(get_value("annual_spend")),
            currency=get_value("currency") or "USD",
        )

    # =========================================================================
    # Template Generation
    # =========================================================================

    def generate_supplier_template(
        self,
        file_path: Union[str, Path],
        format: ExportFormat = ExportFormat.CSV,
        include_sample_data: bool = True,
    ) -> None:
        """
        Generate a supplier import template.

        Args:
            file_path: Output file path
            format: Output format
            include_sample_data: Include sample rows
        """
        headers = [
            "id", "name", "tier", "country_code",
            "street", "city", "state", "postal_code",
            "lei", "duns", "vat_number",
            "annual_spend", "currency",
            "contact_name", "contact_email", "contact_phone",
            "industry_code"
        ]

        sample_data = []
        if include_sample_data:
            sample_data = [
                [
                    "SUP001", "Acme Manufacturing GmbH", "1", "DE",
                    "Industriestrasse 1", "Munich", "Bavaria", "80331",
                    "5493001KJTIIGC8Y1R12", "123456789", "DE123456789",
                    "1000000", "EUR",
                    "Hans Mueller", "hans.mueller@acme.de", "+49 89 12345",
                    "332710"
                ],
                [
                    "SUP002", "Global Supplies Inc", "1", "US",
                    "100 Main Street", "New York", "NY", "10001",
                    "", "987654321", "",
                    "500000", "USD",
                    "John Smith", "jsmith@globalsupplies.com", "+1 212 555 1234",
                    "423450"
                ],
            ]

        with open(file_path, "w", encoding=self.encoding, newline="") as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerow(headers)
            for row in sample_data:
                writer.writerow(row)

        logger.info(f"Generated supplier template: {file_path}")

    def generate_facility_template(
        self,
        file_path: Union[str, Path],
        include_sample_data: bool = True,
    ) -> None:
        """
        Generate a facility import template.

        Args:
            file_path: Output file path
            include_sample_data: Include sample rows
        """
        headers = [
            "id", "name", "supplier_id", "facility_type",
            "latitude", "longitude",
            "street", "city", "state", "postal_code", "country_code"
        ]

        sample_data = []
        if include_sample_data:
            sample_data = [
                [
                    "FAC001", "Munich Production Plant", "SUP001", "production",
                    "48.1351", "11.5820",
                    "Industriestrasse 1", "Munich", "Bavaria", "80331", "DE"
                ],
            ]

        with open(file_path, "w", encoding=self.encoding, newline="") as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            writer.writerow(headers)
            for row in sample_data:
                writer.writerow(row)

        logger.info(f"Generated facility template: {file_path}")

    # =========================================================================
    # JSON Export
    # =========================================================================

    def export_to_json(
        self,
        data: Dict[str, Any],
        file_path: Union[str, Path],
        indent: int = 2,
    ) -> None:
        """
        Export data to JSON file.

        Args:
            data: Data to export
            file_path: Output file path
            indent: JSON indentation
        """
        with open(file_path, "w", encoding=self.encoding) as f:
            json.dump(data, f, indent=indent, default=str)

        logger.info(f"Exported data to JSON: {file_path}")

    def import_from_json(
        self,
        file_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Import data from JSON file.

        Args:
            file_path: Input file path

        Returns:
            Parsed JSON data
        """
        with open(file_path, "r", encoding=self.encoding) as f:
            return json.load(f)
