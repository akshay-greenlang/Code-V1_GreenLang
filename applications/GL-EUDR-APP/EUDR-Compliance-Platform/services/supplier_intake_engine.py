# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Supplier Intake Engine - Supplier Data Ingestion and Normalization

Manages supplier data lifecycle: creation, update, search, bulk import,
compliance status, and risk summary. Handles ERP data normalization from
SAP, Oracle, and CSV formats into the canonical Supplier model.

Zero-Hallucination Guarantees:
    - All validations are rule-based (country codes, commodity enums)
    - No LLM used for data normalization or classification
    - ERP mappings are deterministic field-to-field transforms
    - SHA-256 provenance on bulk imports

Example:
    >>> from services.supplier_intake_engine import SupplierIntakeEngine
    >>> from services.config import EUDRAppConfig
    >>> engine = SupplierIntakeEngine(EUDRAppConfig())
    >>> supplier = engine.create_supplier(request)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from services.config import (
    ComplianceStatus,
    EUDRAppConfig,
    EUDRCommodity,
    RiskLevel,
)
from services.models import (
    BulkImportResult,
    Plot,
    PlotCreateRequest,
    Procurement,
    ProcurementCreateRequest,
    Supplier,
    SupplierComplianceStatus,
    SupplierCreateRequest,
    SupplierFilterRequest,
    SupplierRiskSummary,
    SupplierUpdateRequest,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a UUID v4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Country Code Validation (ISO 3166-1 alpha-3 subset)
# ---------------------------------------------------------------------------

VALID_COUNTRY_CODES: Set[str] = {
    "ABW", "AFG", "AGO", "ALB", "AND", "ARE", "ARG", "ARM", "AUS", "AUT",
    "AZE", "BDI", "BEL", "BEN", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH",
    "BLR", "BLZ", "BOL", "BRA", "BRN", "BTN", "BWA", "CAF", "CAN", "CHE",
    "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COL", "COM", "CPV", "CRI",
    "CUB", "CYP", "CZE", "DEU", "DJI", "DNK", "DOM", "DZA", "ECU", "EGY",
    "ERI", "ESP", "EST", "ETH", "FIN", "FJI", "FRA", "GAB", "GBR", "GEO",
    "GHA", "GIN", "GMB", "GNB", "GNQ", "GRC", "GTM", "GUY", "HND", "HRV",
    "HTI", "HUN", "IDN", "IND", "IRL", "IRN", "IRQ", "ISL", "ISR", "ITA",
    "JAM", "JOR", "JPN", "KAZ", "KEN", "KGZ", "KHM", "KOR", "KWT", "LAO",
    "LBN", "LBR", "LBY", "LKA", "LSO", "LTU", "LUX", "LVA", "MAR", "MDA",
    "MDG", "MEX", "MKD", "MLI", "MLT", "MMR", "MNE", "MNG", "MOZ", "MRT",
    "MUS", "MWI", "MYS", "NAM", "NER", "NGA", "NIC", "NLD", "NOR", "NPL",
    "NZL", "OMN", "PAK", "PAN", "PER", "PHL", "PNG", "POL", "PRT", "PRY",
    "QAT", "ROU", "RUS", "RWA", "SAU", "SDN", "SEN", "SGP", "SLE", "SLV",
    "SOM", "SRB", "SSD", "SUR", "SVK", "SVN", "SWE", "SWZ", "SYR", "TCD",
    "TGO", "THA", "TJK", "TKM", "TLS", "TTO", "TUN", "TUR", "TWN", "TZA",
    "UGA", "UKR", "URY", "USA", "UZB", "VEN", "VNM", "YEM", "ZAF", "ZMB",
    "ZWE",
}


# ---------------------------------------------------------------------------
# ERP Field Mappings
# ---------------------------------------------------------------------------

SAP_FIELD_MAP: Dict[str, str] = {
    "LIFNR": "tax_id",
    "NAME1": "name",
    "LAND1": "country",
    "STRAS": "address",
    "SMTP_ADDR": "contact_email",
    "TELF1": "contact_phone",
    "WAERS": "currency",
    "BRSCH": "industry",
}

ORACLE_FIELD_MAP: Dict[str, str] = {
    "VENDOR_ID": "tax_id",
    "VENDOR_NAME": "name",
    "COUNTRY_CODE": "country",
    "ADDRESS_LINE_1": "address",
    "EMAIL_ADDRESS": "contact_email",
    "PHONE_NUMBER": "contact_phone",
    "VENDOR_TYPE": "vendor_type",
}

CSV_FIELD_MAP: Dict[str, str] = {
    "supplier_name": "name",
    "supplier_id": "tax_id",
    "country_code": "country",
    "country": "country",
    "email": "contact_email",
    "phone": "contact_phone",
    "address": "address",
    "commodity": "commodity",
    "commodities": "commodity",
}


# ===========================================================================
# Supplier Intake Engine
# ===========================================================================


class SupplierIntakeEngine:
    """Manages supplier data ingestion, normalization, and CRUD operations.

    Provides thread-safe in-memory storage for suppliers, plots, and
    procurements. Handles ERP data normalization from SAP, Oracle, and
    CSV formats. Computes compliance status and risk summaries.

    Attributes:
        _config: Application configuration.
        _lock: Reentrant lock for thread safety.
        _suppliers: In-memory supplier storage keyed by ID.
        _plots: In-memory plot storage keyed by ID.
        _procurements: In-memory procurement storage keyed by ID.

    Example:
        >>> engine = SupplierIntakeEngine(config)
        >>> supplier = engine.create_supplier(SupplierCreateRequest(
        ...     name="AgroTrade", country="BRA",
        ...     commodities=["coffee", "cocoa"]
        ... ))
    """

    def __init__(self, config: EUDRAppConfig) -> None:
        """Initialize SupplierIntakeEngine.

        Args:
            config: Application configuration.
        """
        self._config = config
        self._lock = threading.RLock()
        self._suppliers: Dict[str, Supplier] = {}
        self._plots: Dict[str, Plot] = {}
        self._procurements: Dict[str, Procurement] = {}
        logger.info("SupplierIntakeEngine initialized")

    # -----------------------------------------------------------------------
    # Supplier CRUD
    # -----------------------------------------------------------------------

    def create_supplier(self, data: SupplierCreateRequest) -> Supplier:
        """Create a new supplier record.

        Validates country code and commodity types, then creates the
        supplier with PENDING compliance status.

        Args:
            data: Supplier creation request.

        Returns:
            Created Supplier record.

        Raises:
            ValueError: If country code is invalid.
        """
        country = data.country.upper()
        self._validate_country_code(country)

        # Determine initial risk level based on country
        risk_level = self._assess_initial_risk(country)

        supplier = Supplier(
            name=data.name.strip(),
            country=country,
            tax_id=data.tax_id,
            address=data.address,
            contact_email=data.contact_email,
            contact_phone=data.contact_phone,
            commodities=data.commodities,
            risk_level=risk_level,
            compliance_status=ComplianceStatus.PENDING,
            certifications=data.certifications,
            notes=data.notes,
        )

        with self._lock:
            self._suppliers[supplier.id] = supplier

        logger.info(
            "Created supplier %s: name=%s, country=%s, risk=%s",
            supplier.id,
            supplier.name,
            country,
            risk_level.value,
        )
        return supplier

    def update_supplier(
        self, supplier_id: str, data: SupplierUpdateRequest
    ) -> Supplier:
        """Update an existing supplier record.

        Only non-None fields in the update request are applied.

        Args:
            supplier_id: Supplier ID to update.
            data: Fields to update.

        Returns:
            Updated Supplier record.

        Raises:
            ValueError: If supplier not found.
            ValueError: If country code is invalid.
        """
        with self._lock:
            supplier = self._suppliers.get(supplier_id)
            if supplier is None:
                raise ValueError(f"Supplier not found: {supplier_id}")

            if data.name is not None:
                supplier.name = data.name.strip()
            if data.country is not None:
                country = data.country.upper()
                self._validate_country_code(country)
                supplier.country = country
            if data.tax_id is not None:
                supplier.tax_id = data.tax_id
            if data.address is not None:
                supplier.address = data.address
            if data.contact_email is not None:
                supplier.contact_email = data.contact_email
            if data.contact_phone is not None:
                supplier.contact_phone = data.contact_phone
            if data.commodities is not None:
                supplier.commodities = data.commodities
            if data.compliance_status is not None:
                supplier.compliance_status = data.compliance_status
            if data.certifications is not None:
                supplier.certifications = data.certifications
            if data.notes is not None:
                supplier.notes = data.notes

            supplier.updated_at = _utcnow()

        logger.info("Updated supplier %s", supplier_id)
        return supplier

    def get_supplier(self, supplier_id: str) -> Optional[Supplier]:
        """Retrieve a supplier by ID.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            Supplier if found, None otherwise.
        """
        with self._lock:
            return self._suppliers.get(supplier_id)

    def list_suppliers(
        self, filters: Optional[SupplierFilterRequest] = None
    ) -> List[Supplier]:
        """List suppliers with optional filtering.

        Args:
            filters: Filtering criteria.

        Returns:
            List of matching Supplier records.
        """
        with self._lock:
            suppliers = list(self._suppliers.values())

        if filters is None:
            return suppliers

        # Apply filters
        if filters.country:
            suppliers = [
                s for s in suppliers
                if s.country == filters.country.upper()
            ]
        if filters.commodity:
            suppliers = [
                s for s in suppliers
                if filters.commodity in s.commodities
            ]
        if filters.risk_level:
            suppliers = [
                s for s in suppliers
                if s.risk_level == filters.risk_level
            ]
        if filters.compliance_status:
            suppliers = [
                s for s in suppliers
                if s.compliance_status == filters.compliance_status
            ]
        if filters.search_query:
            query = filters.search_query.lower()
            suppliers = [
                s for s in suppliers
                if query in s.name.lower()
                or (s.tax_id and query in s.tax_id.lower())
                or query in s.country.lower()
            ]

        # Apply pagination
        offset = filters.offset
        limit = filters.limit
        return suppliers[offset: offset + limit]

    def search_suppliers(self, query: str) -> List[Supplier]:
        """Search suppliers by name, tax_id, or country.

        Args:
            query: Search query string.

        Returns:
            List of matching Supplier records.
        """
        if not query or not query.strip():
            return []

        query_lower = query.strip().lower()

        with self._lock:
            results = [
                s
                for s in self._suppliers.values()
                if query_lower in s.name.lower()
                or (s.tax_id and query_lower in s.tax_id.lower())
                or query_lower in s.country.lower()
                or any(query_lower in c.value for c in s.commodities)
            ]

        return results

    def delete_supplier(self, supplier_id: str) -> bool:
        """Delete a supplier by ID.

        Also removes associated plots and procurements.

        Args:
            supplier_id: Supplier to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._lock:
            if supplier_id not in self._suppliers:
                return False

            del self._suppliers[supplier_id]

            # Remove associated plots
            plot_ids_to_remove = [
                pid
                for pid, p in self._plots.items()
                if p.supplier_id == supplier_id
            ]
            for pid in plot_ids_to_remove:
                del self._plots[pid]

            # Remove associated procurements
            proc_ids_to_remove = [
                pid
                for pid, p in self._procurements.items()
                if p.supplier_id == supplier_id
            ]
            for pid in proc_ids_to_remove:
                del self._procurements[pid]

        logger.info(
            "Deleted supplier %s with %d plots and %d procurements",
            supplier_id,
            len(plot_ids_to_remove),
            len(proc_ids_to_remove),
        )
        return True

    # -----------------------------------------------------------------------
    # Plot Management
    # -----------------------------------------------------------------------

    def add_plot(
        self, supplier_id: str, data: PlotCreateRequest
    ) -> Plot:
        """Add a production plot for a supplier.

        Args:
            supplier_id: Supplier to add the plot to.
            data: Plot creation data.

        Returns:
            Created Plot record.

        Raises:
            ValueError: If supplier not found.
        """
        with self._lock:
            supplier = self._suppliers.get(supplier_id)
            if supplier is None:
                raise ValueError(f"Supplier not found: {supplier_id}")

        plot = Plot(
            supplier_id=supplier_id,
            name=data.name.strip(),
            coordinates=data.coordinates,
            centroid_lat=data.centroid_lat,
            centroid_lon=data.centroid_lon,
            area_hectares=data.area_hectares,
            commodity=data.commodity,
            country_iso3=data.country_iso3.upper(),
            region=data.region,
        )

        with self._lock:
            self._plots[plot.id] = plot
            if plot.id not in supplier.plots:
                supplier.plots.append(plot.id)

        logger.info(
            "Added plot %s for supplier %s: commodity=%s, country=%s",
            plot.id,
            supplier_id,
            plot.commodity.value,
            plot.country_iso3,
        )
        return plot

    def get_plot(self, plot_id: str) -> Optional[Plot]:
        """Get a plot by ID.

        Args:
            plot_id: Plot identifier.

        Returns:
            Plot if found, None otherwise.
        """
        with self._lock:
            return self._plots.get(plot_id)

    def list_plots(
        self, supplier_id: Optional[str] = None
    ) -> List[Plot]:
        """List plots, optionally filtered by supplier.

        Args:
            supplier_id: Filter by supplier ID.

        Returns:
            List of Plot records.
        """
        with self._lock:
            plots = list(self._plots.values())

        if supplier_id:
            plots = [p for p in plots if p.supplier_id == supplier_id]

        return plots

    # -----------------------------------------------------------------------
    # Procurement Management
    # -----------------------------------------------------------------------

    def add_procurement(
        self, supplier_id: str, data: ProcurementCreateRequest
    ) -> Procurement:
        """Create a procurement record for a supplier.

        Args:
            supplier_id: Supplier this procurement is from.
            data: Procurement creation data.

        Returns:
            Created Procurement record.

        Raises:
            ValueError: If supplier not found.
        """
        with self._lock:
            supplier = self._suppliers.get(supplier_id)
            if supplier is None:
                raise ValueError(f"Supplier not found: {supplier_id}")

        procurement = Procurement(
            supplier_id=supplier_id,
            po_number=data.po_number.strip(),
            commodity=data.commodity,
            quantity=data.quantity,
            unit=data.unit,
            harvest_date=data.harvest_date,
            shipment_date=data.shipment_date,
            arrival_date=data.arrival_date,
            origin_plot_ids=data.origin_plot_ids,
            notes=data.notes,
        )

        with self._lock:
            self._procurements[procurement.id] = procurement

        logger.info(
            "Added procurement %s for supplier %s: PO=%s, commodity=%s",
            procurement.id,
            supplier_id,
            procurement.po_number,
            procurement.commodity.value,
        )
        return procurement

    def get_procurement(self, procurement_id: str) -> Optional[Procurement]:
        """Get a procurement record by ID.

        Args:
            procurement_id: Procurement identifier.

        Returns:
            Procurement if found, None otherwise.
        """
        with self._lock:
            return self._procurements.get(procurement_id)

    def list_procurements(
        self, supplier_id: Optional[str] = None
    ) -> List[Procurement]:
        """List procurements, optionally filtered by supplier.

        Args:
            supplier_id: Filter by supplier ID.

        Returns:
            List of Procurement records.
        """
        with self._lock:
            procs = list(self._procurements.values())

        if supplier_id:
            procs = [p for p in procs if p.supplier_id == supplier_id]

        return procs

    # -----------------------------------------------------------------------
    # Bulk Import
    # -----------------------------------------------------------------------

    def bulk_import_suppliers(
        self, records: List[Dict[str, Any]]
    ) -> BulkImportResult:
        """Bulk import supplier records from raw dictionaries.

        Each record should contain at minimum 'name' and 'country' fields.
        Records are validated individually; failures do not block others.

        Args:
            records: List of raw supplier data dictionaries.

        Returns:
            BulkImportResult with import statistics.
        """
        result = BulkImportResult(total_records=len(records))
        start_time = _utcnow()

        for idx, record in enumerate(records):
            try:
                # Normalize field names
                normalized = self._normalize_raw_record(record)

                # Validate required fields
                name = normalized.get("name", "").strip()
                country = normalized.get("country", "").strip().upper()

                if not name:
                    raise ValueError(f"Record {idx}: 'name' is required")
                if not country:
                    raise ValueError(f"Record {idx}: 'country' is required")

                self._validate_country_code(country)

                # Parse commodities
                commodities = self._parse_commodities(
                    normalized.get("commodity", "")
                )

                # Create supplier
                request = SupplierCreateRequest(
                    name=name,
                    country=country,
                    tax_id=normalized.get("tax_id"),
                    address=normalized.get("address"),
                    contact_email=normalized.get("contact_email"),
                    contact_phone=normalized.get("contact_phone"),
                    commodities=commodities,
                )
                supplier = self.create_supplier(request)
                result.imported += 1
                result.suppliers.append(supplier.id)

            except Exception as exc:
                result.failed += 1
                result.errors.append({
                    "index": idx,
                    "record": record,
                    "error": str(exc),
                })
                logger.warning(
                    "Bulk import record %d failed: %s", idx, exc
                )

        elapsed_ms = (_utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Bulk import completed: %d/%d imported, %d failed (%.1fms)",
            result.imported,
            result.total_records,
            result.failed,
            elapsed_ms,
        )
        return result

    # -----------------------------------------------------------------------
    # Compliance & Risk
    # -----------------------------------------------------------------------

    def get_compliance_status(
        self, supplier_id: str
    ) -> SupplierComplianceStatus:
        """Compute comprehensive compliance status for a supplier.

        Evaluates plot risk, document completeness, and DDS status.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            SupplierComplianceStatus with all metrics.

        Raises:
            ValueError: If supplier not found.
        """
        with self._lock:
            supplier = self._suppliers.get(supplier_id)
            if supplier is None:
                raise ValueError(f"Supplier not found: {supplier_id}")

            plots = [
                p for p in self._plots.values()
                if p.supplier_id == supplier_id
            ]

        plots_total = len(plots)
        plots_compliant = sum(
            1 for p in plots if p.is_deforestation_free is True
        )
        plots_at_risk = sum(
            1
            for p in plots
            if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )

        # Determine issues
        issues: List[str] = []
        if plots_total == 0:
            issues.append("No plots registered")
        if plots_at_risk > 0:
            issues.append(f"{plots_at_risk} plots at high/critical risk")

        return SupplierComplianceStatus(
            supplier_id=supplier_id,
            compliance_status=supplier.compliance_status,
            risk_level=supplier.risk_level,
            plots_total=plots_total,
            plots_compliant=plots_compliant,
            plots_at_risk=plots_at_risk,
            issues=issues,
        )

    def get_risk_summary(self, supplier_id: str) -> SupplierRiskSummary:
        """Compute risk summary for a supplier.

        Args:
            supplier_id: Supplier identifier.

        Returns:
            SupplierRiskSummary with risk breakdown.

        Raises:
            ValueError: If supplier not found.
        """
        with self._lock:
            supplier = self._suppliers.get(supplier_id)
            if supplier is None:
                raise ValueError(f"Supplier not found: {supplier_id}")

            plots = [
                p for p in self._plots.values()
                if p.supplier_id == supplier_id
            ]

        # Country risk from config
        country_risk = 0.5
        if supplier.country in self._config.high_risk_countries:
            country_risk = 0.8

        # Satellite risk from plot assessments
        satellite_risks = [
            p.risk_level for p in plots
            if p.satellite_status != SatelliteAssessmentStatus.NOT_ASSESSED
        ]
        satellite_risk = 0.5
        if satellite_risks:
            risk_values = {
                RiskLevel.LOW: 0.1,
                RiskLevel.STANDARD: 0.3,
                RiskLevel.HIGH: 0.7,
                RiskLevel.CRITICAL: 0.95,
            }
            vals = [risk_values.get(r, 0.3) for r in satellite_risks]
            satellite_risk = sum(vals) / len(vals)

        high_risk_plots = sum(
            1
            for p in plots
            if p.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
        )

        # Weighted overall risk
        overall = (
            self._config.risk_weight_satellite * satellite_risk
            + self._config.risk_weight_country * country_risk
            + self._config.risk_weight_supplier * 0.3  # Baseline supplier risk
            + self._config.risk_weight_document * 0.4  # Baseline doc risk
        )

        risk_level = self._classify_risk(overall)

        return SupplierRiskSummary(
            supplier_id=supplier_id,
            overall_risk=round(overall, 4),
            risk_level=risk_level,
            country_risk=round(country_risk, 4),
            satellite_risk=round(satellite_risk, 4),
            document_risk=0.4,
            supplier_history_risk=0.3,
            high_risk_plots=high_risk_plots,
            last_assessed=_utcnow(),
        )

    # -----------------------------------------------------------------------
    # ERP Normalization
    # -----------------------------------------------------------------------

    def normalize_erp_data(
        self, erp_type: str, raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize ERP-specific field names to canonical format.

        Supports SAP, Oracle, and CSV field mappings. Unknown fields
        are passed through unchanged.

        Args:
            erp_type: ERP system type ("sap", "oracle", "csv").
            raw_data: Raw data with ERP-specific field names.

        Returns:
            Normalized dictionary with canonical field names.

        Raises:
            ValueError: If erp_type is not supported.
        """
        erp_type_lower = erp_type.lower().strip()

        field_maps: Dict[str, Dict[str, str]] = {
            "sap": SAP_FIELD_MAP,
            "oracle": ORACLE_FIELD_MAP,
            "csv": CSV_FIELD_MAP,
        }

        field_map = field_maps.get(erp_type_lower)
        if field_map is None:
            raise ValueError(
                f"Unsupported ERP type: {erp_type}. "
                f"Supported: {list(field_maps.keys())}"
            )

        normalized: Dict[str, Any] = {}
        for key, value in raw_data.items():
            canonical_key = field_map.get(key, key)
            normalized[canonical_key] = value

        # Normalize country code
        if "country" in normalized:
            normalized["country"] = self._normalize_country(
                str(normalized["country"])
            )

        logger.debug(
            "Normalized %s data: %d fields mapped",
            erp_type_lower,
            len(normalized),
        )
        return normalized

    def normalize_erp_batch(
        self, erp_type: str, raw_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize a batch of ERP records.

        Args:
            erp_type: ERP system type.
            raw_records: List of raw records.

        Returns:
            List of normalized records.
        """
        return [
            self.normalize_erp_data(erp_type, record)
            for record in raw_records
        ]

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dictionary with supplier, plot, and procurement counts.
        """
        with self._lock:
            suppliers = list(self._suppliers.values())
            plots = list(self._plots.values())
            procs = list(self._procurements.values())

        # Compliance breakdown
        compliance_breakdown: Dict[str, int] = {}
        for s in suppliers:
            key = s.compliance_status.value
            compliance_breakdown[key] = compliance_breakdown.get(key, 0) + 1

        # Commodity breakdown
        commodity_breakdown: Dict[str, int] = {}
        for s in suppliers:
            for c in s.commodities:
                commodity_breakdown[c.value] = (
                    commodity_breakdown.get(c.value, 0) + 1
                )

        # Country breakdown
        country_breakdown: Dict[str, int] = {}
        for s in suppliers:
            country_breakdown[s.country] = (
                country_breakdown.get(s.country, 0) + 1
            )

        return {
            "total_suppliers": len(suppliers),
            "total_plots": len(plots),
            "total_procurements": len(procs),
            "compliance_breakdown": compliance_breakdown,
            "commodity_breakdown": commodity_breakdown,
            "country_breakdown": country_breakdown,
        }

    # -----------------------------------------------------------------------
    # Private Helpers
    # -----------------------------------------------------------------------

    def _validate_country_code(self, code: str) -> None:
        """Validate an ISO 3166-1 alpha-3 country code.

        Args:
            code: Country code to validate.

        Raises:
            ValueError: If code is not a valid ISO-3 code.
        """
        if code not in VALID_COUNTRY_CODES:
            raise ValueError(
                f"Invalid ISO 3166-1 alpha-3 country code: {code}"
            )

    def _assess_initial_risk(self, country: str) -> RiskLevel:
        """Determine initial risk level based on country.

        Args:
            country: ISO-3 country code.

        Returns:
            Initial RiskLevel classification.
        """
        if country in self._config.high_risk_countries:
            return RiskLevel.HIGH
        return RiskLevel.STANDARD

    def _classify_risk(self, score: float) -> RiskLevel:
        """Classify a risk score into a RiskLevel.

        Args:
            score: Risk score between 0 and 1.

        Returns:
            Classified RiskLevel.
        """
        if score >= self._config.risk_threshold_critical:
            return RiskLevel.CRITICAL
        if score >= self._config.risk_threshold_high:
            return RiskLevel.HIGH
        if score >= 0.3:
            return RiskLevel.STANDARD
        return RiskLevel.LOW

    def _normalize_raw_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize field names in a raw import record.

        Applies CSV field mapping and lowercases field names.

        Args:
            record: Raw record dictionary.

        Returns:
            Normalized dictionary.
        """
        normalized: Dict[str, Any] = {}
        for key, value in record.items():
            canonical = CSV_FIELD_MAP.get(key, key.lower())
            normalized[canonical] = value
        return normalized

    def _parse_commodities(
        self, commodity_str: Any
    ) -> List[EUDRCommodity]:
        """Parse commodity string or list into EUDRCommodity enums.

        Handles comma-separated strings, lists, and single values.

        Args:
            commodity_str: Raw commodity value(s).

        Returns:
            List of valid EUDRCommodity values.
        """
        if not commodity_str:
            return []

        # Handle list input
        if isinstance(commodity_str, list):
            raw_values = commodity_str
        else:
            raw_values = [
                s.strip()
                for s in str(commodity_str).split(",")
                if s.strip()
            ]

        commodities: List[EUDRCommodity] = []
        valid_values = {c.value for c in EUDRCommodity}

        for val in raw_values:
            val_lower = val.lower().strip().replace(" ", "_")
            if val_lower in valid_values:
                commodities.append(EUDRCommodity(val_lower))
            else:
                logger.warning(
                    "Unknown commodity '%s', skipping. Valid: %s",
                    val,
                    list(valid_values),
                )

        return commodities

    def _normalize_country(self, raw_country: str) -> str:
        """Normalize a country value to ISO 3166-1 alpha-3.

        Handles 2-letter codes and common country names.

        Args:
            raw_country: Raw country string.

        Returns:
            Normalized 3-letter country code, or original if unrecognized.
        """
        value = raw_country.strip().upper()

        # Already 3-letter code
        if len(value) == 3 and value in VALID_COUNTRY_CODES:
            return value

        # Common 2-letter to 3-letter mappings
        iso2_to_iso3: Dict[str, str] = {
            "BR": "BRA", "ID": "IDN", "CD": "COD", "CG": "COG",
            "CM": "CMR", "MY": "MYS", "PG": "PNG", "BO": "BOL",
            "PE": "PER", "CO": "COL", "DE": "DEU", "FR": "FRA",
            "NL": "NLD", "SE": "SWE", "FI": "FIN", "US": "USA",
            "CA": "CAN", "AU": "AUS", "NZ": "NZL", "JP": "JPN",
            "GB": "GBR", "IT": "ITA", "ES": "ESP", "PT": "PRT",
            "AT": "AUT", "BE": "BEL", "CH": "CHE", "NO": "NOR",
            "DK": "DNK", "IE": "IRL", "PL": "POL", "CZ": "CZE",
            "GH": "GHA", "CI": "CIV", "NG": "NGA", "KE": "KEN",
            "TZ": "TZA", "VN": "VNM", "TH": "THA", "EC": "ECU",
            "GT": "GTM", "HN": "HND", "IN": "IND", "CN": "CHN",
            "MX": "MEX",
        }

        if value in iso2_to_iso3:
            return iso2_to_iso3[value]

        # Return as-is if we cannot normalize
        return value
