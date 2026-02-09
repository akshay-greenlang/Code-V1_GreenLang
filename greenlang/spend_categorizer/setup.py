# -*- coding: utf-8 -*-
"""
Spend Data Categorizer Service Setup - AGENT-DATA-009

Provides ``configure_spend_categorizer(app)`` which wires up the
Spend Data Categorizer SDK (record ingestion, taxonomy classification,
Scope 3 mapping, emission calculation, rule engine, analytics engine,
report generator, provenance tracker) and mounts the REST API.

Also exposes ``get_spend_categorizer(app)`` for programmatic access
and the ``SpendCategorizerService`` facade class.

Usage:
    >>> from fastapi import FastAPI
    >>> from greenlang.spend_categorizer.setup import configure_spend_categorizer
    >>> app = FastAPI()
    >>> import asyncio
    >>> service = asyncio.run(configure_spend_categorizer(app))

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-009 Spend Data Categorizer (GL-DATA-SUP-002)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.spend_categorizer.config import (
    SpendCategorizerConfig,
    get_config,
)
from greenlang.spend_categorizer.metrics import (
    PROMETHEUS_AVAILABLE,
    record_ingestion,
    record_classification,
    record_scope3_mapping,
    record_emission_calculation,
    record_rule_evaluation,
    record_report_generation,
    record_classification_confidence,
    record_processing_duration,
    update_active_batches,
    update_total_spend,
    record_processing_error,
    record_factor_lookup,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment, misc]
    FASTAPI_AVAILABLE = False


# ===================================================================
# Lightweight Pydantic models used by the facade
# ===================================================================


class SpendRecordResponse(BaseModel):
    """Spend record response model.

    Attributes:
        record_id: Unique record identifier.
        vendor_name: Vendor/supplier name.
        vendor_id: Vendor identifier.
        description: Spend description or line item text.
        amount: Spend amount in original currency.
        currency: ISO 4217 currency code.
        amount_usd: Normalized amount in USD.
        date: Transaction date (ISO 8601).
        cost_center: Cost center or department.
        gl_account: General ledger account code.
        po_number: Purchase order number.
        source: Data source identifier.
        taxonomy_code: Assigned taxonomy code (UNSPSC, NAICS, etc.).
        taxonomy_description: Human-readable taxonomy label.
        taxonomy_system: Taxonomy system used (unspsc, naics, nace).
        classification_confidence: Classification confidence score (0.0-1.0).
        scope3_category: GHG Protocol Scope 3 category (1-15).
        scope3_category_name: Scope 3 category display name.
        emissions_kg_co2e: Calculated emissions in kg CO2e.
        emission_factor: Emission factor used (kg CO2e per USD).
        emission_factor_source: Source of emission factor (eeio, exiobase, defra).
        status: Record processing status (ingested, classified, mapped, calculated).
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    vendor_name: str = Field(default="")
    vendor_id: str = Field(default="")
    description: str = Field(default="")
    amount: float = Field(default=0.0)
    currency: str = Field(default="USD")
    amount_usd: float = Field(default=0.0)
    date: str = Field(default="")
    cost_center: str = Field(default="")
    gl_account: str = Field(default="")
    po_number: str = Field(default="")
    source: str = Field(default="manual")
    taxonomy_code: str = Field(default="")
    taxonomy_description: str = Field(default="")
    taxonomy_system: str = Field(default="")
    classification_confidence: float = Field(default=0.0)
    scope3_category: int = Field(default=0)
    scope3_category_name: str = Field(default="")
    emissions_kg_co2e: float = Field(default=0.0)
    emission_factor: float = Field(default=0.0)
    emission_factor_source: str = Field(default="")
    status: str = Field(default="ingested")
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class ClassificationResponse(BaseModel):
    """Classification result for a spend record.

    Attributes:
        classification_id: Unique classification identifier.
        record_id: Source spend record identifier.
        taxonomy_code: Assigned taxonomy code.
        taxonomy_description: Human-readable taxonomy label.
        taxonomy_system: Taxonomy system (unspsc, naics, nace, custom).
        confidence: Classification confidence score (0.0-1.0).
        confidence_label: Confidence label (HIGH, MEDIUM, LOW).
        alternative_codes: Alternative taxonomy codes considered.
        rule_id: Rule that matched (if rule-based classification).
        method: Classification method (rule, keyword, ml, manual).
        provenance_hash: SHA-256 provenance hash.
        classified_at: Timestamp of classification.
    """
    classification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str = Field(default="")
    taxonomy_code: str = Field(default="")
    taxonomy_description: str = Field(default="")
    taxonomy_system: str = Field(default="unspsc")
    confidence: float = Field(default=0.0)
    confidence_label: str = Field(default="LOW")
    alternative_codes: List[Dict[str, Any]] = Field(default_factory=list)
    rule_id: Optional[str] = Field(default=None)
    method: str = Field(default="keyword")
    provenance_hash: str = Field(default="")
    classified_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class Scope3AssignmentResponse(BaseModel):
    """Scope 3 category assignment result.

    Attributes:
        assignment_id: Unique assignment identifier.
        record_id: Source spend record identifier.
        scope3_category: GHG Protocol Scope 3 category number (1-15).
        scope3_category_name: Scope 3 category display name.
        taxonomy_code: Taxonomy code used for mapping.
        mapping_confidence: Mapping confidence score (0.0-1.0).
        mapping_method: Mapping method (taxonomy_lookup, rule, manual).
        provenance_hash: SHA-256 provenance hash.
        assigned_at: Timestamp of assignment.
    """
    assignment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str = Field(default="")
    scope3_category: int = Field(default=0)
    scope3_category_name: str = Field(default="")
    taxonomy_code: str = Field(default="")
    mapping_confidence: float = Field(default=0.0)
    mapping_method: str = Field(default="taxonomy_lookup")
    provenance_hash: str = Field(default="")
    assigned_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class EmissionCalculationResponse(BaseModel):
    """Emission calculation result for a spend record.

    Attributes:
        calculation_id: Unique calculation identifier.
        record_id: Source spend record identifier.
        emissions_kg_co2e: Calculated emissions in kg CO2e.
        emission_factor: Emission factor applied (kg CO2e per USD).
        emission_factor_source: Source database (eeio, exiobase, defra, ecoinvent).
        emission_factor_version: Version of the source database.
        taxonomy_code: Taxonomy code used for factor lookup.
        spend_usd: Spend amount in USD used for calculation.
        methodology: Calculation methodology (spend_based, hybrid).
        provenance_hash: SHA-256 provenance hash.
        calculated_at: Timestamp of calculation.
    """
    calculation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    record_id: str = Field(default="")
    emissions_kg_co2e: float = Field(default=0.0)
    emission_factor: float = Field(default=0.0)
    emission_factor_source: str = Field(default="eeio")
    emission_factor_version: str = Field(default="")
    taxonomy_code: str = Field(default="")
    spend_usd: float = Field(default=0.0)
    methodology: str = Field(default="spend_based")
    provenance_hash: str = Field(default="")
    calculated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class CategoryRuleResponse(BaseModel):
    """Category classification rule definition.

    Attributes:
        rule_id: Unique rule identifier.
        name: Rule display name.
        description: Rule description.
        taxonomy_code: Target taxonomy code to assign.
        taxonomy_system: Taxonomy system (unspsc, naics, nace).
        scope3_category: Optional Scope 3 category override.
        conditions: Rule conditions (vendor patterns, keywords, GL codes).
        priority: Rule priority (lower = higher priority).
        is_active: Whether the rule is currently active.
        match_count: Number of records matched by this rule.
        provenance_hash: SHA-256 provenance hash.
        created_at: Timestamp of creation.
        updated_at: Timestamp of last update.
    """
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: str = Field(default="")
    taxonomy_code: str = Field(default="")
    taxonomy_system: str = Field(default="unspsc")
    scope3_category: Optional[int] = Field(default=None)
    conditions: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=100)
    is_active: bool = Field(default=True)
    match_count: int = Field(default=0)
    provenance_hash: str = Field(default="")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class AnalyticsResponse(BaseModel):
    """Analytics summary for spend categorization.

    Attributes:
        total_records: Total spend records processed.
        total_classified: Total records classified.
        total_mapped: Total records mapped to Scope 3.
        total_calculated: Total records with emission calculations.
        total_spend_usd: Total spend in USD.
        total_emissions_kg_co2e: Total calculated emissions in kg CO2e.
        avg_confidence: Average classification confidence.
        classification_rate_pct: Percentage of records classified.
        top_categories: Top spend categories by amount.
        top_vendors: Top vendors by spend amount.
        scope3_breakdown: Spend and emissions by Scope 3 category.
        provenance_hash: SHA-256 provenance hash.
        generated_at: Timestamp of analytics generation.
    """
    total_records: int = Field(default=0)
    total_classified: int = Field(default=0)
    total_mapped: int = Field(default=0)
    total_calculated: int = Field(default=0)
    total_spend_usd: float = Field(default=0.0)
    total_emissions_kg_co2e: float = Field(default=0.0)
    avg_confidence: float = Field(default=0.0)
    classification_rate_pct: float = Field(default=0.0)
    top_categories: List[Dict[str, Any]] = Field(default_factory=list)
    top_vendors: List[Dict[str, Any]] = Field(default_factory=list)
    scope3_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class ReportResponse(BaseModel):
    """Generated report metadata.

    Attributes:
        report_id: Unique report identifier.
        report_type: Report type (summary, detailed, emissions, scope3).
        format: Report format (json, csv, excel, pdf).
        record_count: Number of records included.
        total_spend_usd: Total spend covered.
        total_emissions_kg_co2e: Total emissions covered.
        content: Report content (for JSON format) or download URL.
        provenance_hash: SHA-256 provenance hash.
        generated_at: Timestamp of report generation.
    """
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    report_type: str = Field(default="summary")
    format: str = Field(default="json")
    record_count: int = Field(default=0)
    total_spend_usd: float = Field(default=0.0)
    total_emissions_kg_co2e: float = Field(default=0.0)
    content: Any = Field(default=None)
    provenance_hash: str = Field(default="")
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class SpendCategorizerStatisticsResponse(BaseModel):
    """Aggregate statistics for the spend categorizer service.

    Attributes:
        total_records: Total spend records ingested.
        total_classified: Total records classified.
        total_scope3_mapped: Total records mapped to Scope 3.
        total_emissions_calculated: Total emission calculations.
        total_rules: Total classification rules defined.
        active_rules: Currently active classification rules.
        total_reports: Total reports generated.
        total_spend_usd: Total spend in USD.
        total_emissions_kg_co2e: Total calculated emissions.
        avg_confidence: Average classification confidence.
        active_batches: Number of currently active batches.
    """
    total_records: int = Field(default=0)
    total_classified: int = Field(default=0)
    total_scope3_mapped: int = Field(default=0)
    total_emissions_calculated: int = Field(default=0)
    total_rules: int = Field(default=0)
    active_rules: int = Field(default=0)
    total_reports: int = Field(default=0)
    total_spend_usd: float = Field(default=0.0)
    total_emissions_kg_co2e: float = Field(default=0.0)
    avg_confidence: float = Field(default=0.0)
    active_batches: int = Field(default=0)


# ===================================================================
# Provenance helper
# ===================================================================


class _ProvenanceTracker:
    """Minimal provenance tracker recording SHA-256 audit entries.

    Attributes:
        entries: List of provenance entries.
        entry_count: Number of entries recorded.
    """

    def __init__(self) -> None:
        self._entries: List[Dict[str, Any]] = []
        self.entry_count: int = 0

    def record(
        self,
        entity_type: str,
        entity_id: str,
        action: str,
        data_hash: str,
        user_id: str = "system",
    ) -> str:
        """Record a provenance entry and return its hash.

        Args:
            entity_type: Type of entity (spend_record, classification, scope3, emission, rule, etc.).
            entity_id: Entity identifier.
            action: Action performed (ingest, classify, map, calculate, create, etc.).
            data_hash: SHA-256 hash of associated data.
            user_id: User or system that performed the action.

        Returns:
            SHA-256 hash of the provenance entry itself.
        """
        entry = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "action": action,
            "data_hash": data_hash,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        entry_hash = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()
        ).hexdigest()
        entry["entry_hash"] = entry_hash
        self._entries.append(entry)
        self.entry_count += 1
        return entry_hash


# ===================================================================
# Scope 3 category reference data
# ===================================================================

_SCOPE3_CATEGORIES: Dict[int, str] = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel- and Energy-Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}


# ===================================================================
# SpendCategorizerService facade
# ===================================================================

# Thread-safe singleton lock
_singleton_lock = threading.Lock()
_singleton_instance: Optional["SpendCategorizerService"] = None


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class SpendCategorizerService:
    """Unified facade over the Spend Data Categorizer SDK.

    Aggregates all categorizer engines (record ingestion, taxonomy classifier,
    Scope 3 mapper, emission calculator, rule engine, analytics engine,
    report generator, provenance tracker) through a single entry point with
    convenience methods for common operations.

    Each method records provenance and updates self-monitoring metrics.

    Attributes:
        config: SpendCategorizerConfig instance.
        provenance: _ProvenanceTracker instance for SHA-256 audit trails.

    Example:
        >>> service = SpendCategorizerService()
        >>> records = service.ingest_records([
        ...     {"vendor_name": "Acme Corp", "amount": 50000, "description": "Office supplies"},
        ... ])
        >>> print(records[0].record_id, records[0].status)
    """

    def __init__(
        self,
        config: Optional[SpendCategorizerConfig] = None,
    ) -> None:
        """Initialize the Spend Categorizer Service facade.

        Instantiates all 7 internal engines plus the provenance tracker:
        - RecordIngestionEngine
        - TaxonomyClassifier
        - Scope3Mapper
        - EmissionCalculator
        - RuleEngine
        - AnalyticsEngine
        - ReportGenerator

        Args:
            config: Optional configuration. Uses global config if None.
        """
        self.config = config or get_config()

        # Provenance tracker
        self.provenance = _ProvenanceTracker()

        # Engine placeholders -- real implementations are injected by the
        # respective SDK modules at import time. We use a lazy-init approach
        # so that setup.py can be imported without the full SDK installed.
        self._record_ingestion_engine: Any = None
        self._taxonomy_classifier: Any = None
        self._scope3_mapper: Any = None
        self._emission_calculator: Any = None
        self._rule_engine: Any = None
        self._analytics_engine: Any = None
        self._report_generator: Any = None

        self._init_engines()

        # In-memory stores (production uses DB; these are SDK-level caches)
        self._records: Dict[str, SpendRecordResponse] = {}
        self._classifications: Dict[str, ClassificationResponse] = {}
        self._scope3_assignments: Dict[str, Scope3AssignmentResponse] = {}
        self._emission_calculations: Dict[str, EmissionCalculationResponse] = {}
        self._rules: Dict[str, CategoryRuleResponse] = {}
        self._reports: Dict[str, ReportResponse] = {}

        # Emission factor cache (taxonomy_code -> factor dict)
        self._emission_factors: Dict[str, Dict[str, Any]] = {}
        self._init_default_emission_factors()

        # Statistics
        self._stats = SpendCategorizerStatisticsResponse()
        self._started = False

        logger.info("SpendCategorizerService facade created")

    # ------------------------------------------------------------------
    # Engine properties
    # ------------------------------------------------------------------

    @property
    def record_ingestion_engine(self) -> Any:
        """Get the RecordIngestionEngine instance."""
        return self._record_ingestion_engine

    @property
    def taxonomy_classifier(self) -> Any:
        """Get the TaxonomyClassifier engine instance."""
        return self._taxonomy_classifier

    @property
    def scope3_mapper(self) -> Any:
        """Get the Scope3Mapper engine instance."""
        return self._scope3_mapper

    @property
    def emission_calculator(self) -> Any:
        """Get the EmissionCalculator engine instance."""
        return self._emission_calculator

    @property
    def rule_engine(self) -> Any:
        """Get the RuleEngine engine instance."""
        return self._rule_engine

    @property
    def analytics_engine(self) -> Any:
        """Get the AnalyticsEngine engine instance."""
        return self._analytics_engine

    @property
    def report_generator(self) -> Any:
        """Get the ReportGenerator engine instance."""
        return self._report_generator

    # ------------------------------------------------------------------
    # Engine initialization
    # ------------------------------------------------------------------

    def _init_engines(self) -> None:
        """Attempt to import and initialise SDK engines.

        Engines are optional; missing imports are logged as warnings and
        the service continues in degraded mode.
        """
        try:
            from greenlang.spend_categorizer.record_ingestion import RecordIngestionEngine
            self._record_ingestion_engine = RecordIngestionEngine(self.config)
        except ImportError:
            logger.warning("RecordIngestionEngine not available; using stub")

        try:
            from greenlang.spend_categorizer.taxonomy_classifier import TaxonomyClassifier
            self._taxonomy_classifier = TaxonomyClassifier(self.config)
        except ImportError:
            logger.warning("TaxonomyClassifier not available; using stub")

        try:
            from greenlang.spend_categorizer.scope3_mapper import Scope3Mapper
            self._scope3_mapper = Scope3Mapper(self.config)
        except ImportError:
            logger.warning("Scope3Mapper not available; using stub")

        try:
            from greenlang.spend_categorizer.emission_calculator import EmissionCalculator
            self._emission_calculator = EmissionCalculator(self.config)
        except ImportError:
            logger.warning("EmissionCalculator not available; using stub")

        try:
            from greenlang.spend_categorizer.rule_engine import RuleEngine
            self._rule_engine = RuleEngine(self.config)
        except ImportError:
            logger.warning("RuleEngine not available; using stub")

        try:
            from greenlang.spend_categorizer.spend_analytics import SpendAnalyticsEngine
            self._analytics_engine = SpendAnalyticsEngine(self.config)
        except ImportError:
            logger.warning("SpendAnalyticsEngine not available; using stub")

        try:
            from greenlang.spend_categorizer.report_generator import ReportGeneratorEngine
            self._report_generator = ReportGeneratorEngine(self.config)
        except ImportError:
            logger.warning("ReportGeneratorEngine not available; using stub")

    def _init_default_emission_factors(self) -> None:
        """Initialize a minimal set of default emission factors for the facade.

        Production uses the EmissionCalculator engine with full EPA EEIO/EXIOBASE
        databases. This provides a small default set for SDK-level usage.
        """
        defaults = {
            "43000000": {"factor": 0.42, "source": "eeio", "desc": "IT Equipment"},
            "44000000": {"factor": 0.35, "source": "eeio", "desc": "Office Supplies"},
            "78000000": {"factor": 0.28, "source": "eeio", "desc": "Transportation"},
            "80000000": {"factor": 0.18, "source": "eeio", "desc": "Management Services"},
            "72000000": {"factor": 0.22, "source": "eeio", "desc": "Building Materials"},
            "47000000": {"factor": 0.31, "source": "eeio", "desc": "Cleaning Products"},
            "50000000": {"factor": 0.55, "source": "eeio", "desc": "Food and Beverage"},
            "15000000": {"factor": 0.85, "source": "eeio", "desc": "Fuels and Lubricants"},
            "26000000": {"factor": 0.65, "source": "eeio", "desc": "Electrical Equipment"},
            "76000000": {"factor": 0.15, "source": "eeio", "desc": "Cleaning Services"},
        }
        self._emission_factors = defaults

    # ------------------------------------------------------------------
    # Record ingestion
    # ------------------------------------------------------------------

    def ingest_records(
        self,
        records: List[Dict[str, Any]],
        source: str = "manual",
    ) -> List[SpendRecordResponse]:
        """Ingest a batch of spend records.

        Args:
            records: List of record dicts with spend data fields.
            source: Data source identifier (csv, excel, api, erp, manual).

        Returns:
            List of SpendRecordResponse with ingested record details.

        Raises:
            ValueError: If records list is empty or exceeds max_records.
        """
        start_time = time.time()

        if not records:
            raise ValueError("Records list must not be empty")
        if len(records) > self.config.max_records:
            raise ValueError(
                f"Record count {len(records)} exceeds maximum {self.config.max_records}"
            )

        update_active_batches(1)
        ingested: List[SpendRecordResponse] = []

        for raw in records:
            record = SpendRecordResponse(
                vendor_name=raw.get("vendor_name", ""),
                vendor_id=raw.get("vendor_id", ""),
                description=raw.get("description", ""),
                amount=float(raw.get("amount", 0.0)),
                currency=raw.get("currency", self.config.default_currency),
                amount_usd=float(raw.get("amount_usd", raw.get("amount", 0.0))),
                date=raw.get("date", ""),
                cost_center=raw.get("cost_center", ""),
                gl_account=raw.get("gl_account", ""),
                po_number=raw.get("po_number", ""),
                source=source,
                status="ingested",
            )
            record.provenance_hash = _compute_hash(record)

            self._records[record.record_id] = record

            # Record provenance
            self.provenance.record(
                entity_type="spend_record",
                entity_id=record.record_id,
                action="ingest",
                data_hash=record.provenance_hash,
            )

            # Update metrics
            record_ingestion(source)
            update_total_spend(record.amount_usd)

            ingested.append(record)

        # Update statistics
        self._stats.total_records += len(ingested)
        self._stats.total_spend_usd += sum(r.amount_usd for r in ingested)
        update_active_batches(-1)

        record_processing_duration("ingest", time.time() - start_time)

        logger.info(
            "Ingested %d spend records from source=%s (total USD: %.2f)",
            len(ingested), source,
            sum(r.amount_usd for r in ingested),
        )
        return ingested

    def ingest_csv(
        self,
        file_path: str,
        source: str = "csv",
    ) -> List[SpendRecordResponse]:
        """Ingest spend records from a CSV file.

        Parses the CSV file and delegates to ingest_records().

        Args:
            file_path: Path to CSV file.
            source: Data source label.

        Returns:
            List of ingested SpendRecordResponse records.

        Raises:
            ValueError: If file cannot be parsed.
        """
        start_time = time.time()
        try:
            import csv as csv_mod
            records: List[Dict[str, Any]] = []
            with open(file_path, "r", encoding="utf-8-sig") as fh:
                reader = csv_mod.DictReader(fh)
                for row in reader:
                    records.append(dict(row))

            result = self.ingest_records(records, source=source)
            record_processing_duration("ingest_csv", time.time() - start_time)
            return result
        except Exception as exc:
            record_processing_error("data")
            raise ValueError(f"Failed to parse CSV: {exc}") from exc

    def ingest_excel(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        source: str = "excel",
    ) -> List[SpendRecordResponse]:
        """Ingest spend records from an Excel file.

        Args:
            file_path: Path to Excel file (.xlsx).
            sheet_name: Optional sheet name (uses first sheet if None).
            source: Data source label.

        Returns:
            List of ingested SpendRecordResponse records.

        Raises:
            ValueError: If file cannot be parsed or openpyxl is not installed.
        """
        start_time = time.time()
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            ws = wb[sheet_name] if sheet_name else wb.active
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                raise ValueError("Excel file is empty")

            headers = [str(h).strip().lower() for h in rows[0]]
            records: List[Dict[str, Any]] = []
            for row in rows[1:]:
                record = dict(zip(headers, row))
                records.append(record)

            wb.close()
            result = self.ingest_records(records, source=source)
            record_processing_duration("ingest_excel", time.time() - start_time)
            return result
        except ImportError:
            raise ValueError("openpyxl is required for Excel ingestion")
        except Exception as exc:
            record_processing_error("data")
            raise ValueError(f"Failed to parse Excel: {exc}") from exc

    def get_record(self, record_id: str) -> Optional[SpendRecordResponse]:
        """Get a spend record by ID.

        Args:
            record_id: Record identifier.

        Returns:
            SpendRecordResponse or None if not found.
        """
        return self._records.get(record_id)

    def list_records(
        self,
        source: Optional[str] = None,
        status: Optional[str] = None,
        vendor_name: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[SpendRecordResponse]:
        """List spend records with optional filters.

        Args:
            source: Optional source filter.
            status: Optional status filter.
            vendor_name: Optional vendor name filter (substring match).
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of SpendRecordResponse instances.
        """
        records = list(self._records.values())

        if source is not None:
            records = [r for r in records if r.source == source]
        if status is not None:
            records = [r for r in records if r.status == status]
        if vendor_name is not None:
            vendor_lower = vendor_name.lower()
            records = [
                r for r in records
                if vendor_lower in r.vendor_name.lower()
            ]

        return records[offset:offset + limit]

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_record(
        self,
        record_id: str,
        taxonomy_system: Optional[str] = None,
    ) -> ClassificationResponse:
        """Classify a spend record into a taxonomy category.

        Uses deterministic rule-based and keyword matching for classification.
        No LLM is used for taxonomy assignment (zero-hallucination).

        Args:
            record_id: Spend record identifier.
            taxonomy_system: Taxonomy system override (uses config default if None).

        Returns:
            ClassificationResponse with classification details.

        Raises:
            ValueError: If record not found.
        """
        start_time = time.time()

        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Record {record_id} not found")

        taxonomy = taxonomy_system or self.config.default_taxonomy

        # Try rule-based classification first
        classification = self._classify_by_rules(record, taxonomy)

        # Fall back to keyword-based if no rule matched
        if classification is None:
            classification = self._classify_by_keyword(record, taxonomy)

        classification.provenance_hash = _compute_hash(classification)
        self._classifications[classification.classification_id] = classification

        # Update the record
        record.taxonomy_code = classification.taxonomy_code
        record.taxonomy_description = classification.taxonomy_description
        record.taxonomy_system = classification.taxonomy_system
        record.classification_confidence = classification.confidence
        record.status = "classified"
        record.updated_at = datetime.now(timezone.utc).isoformat()
        record.provenance_hash = _compute_hash(record)

        # Record metrics
        record_classification(taxonomy)
        record_classification_confidence(classification.confidence)
        record_processing_duration("classify", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="classification",
            entity_id=classification.classification_id,
            action="classify",
            data_hash=classification.provenance_hash,
        )

        # Update statistics
        self._stats.total_classified += 1
        self._update_avg_confidence(classification.confidence)

        logger.info(
            "Classified record %s -> %s (%s, confidence=%.2f)",
            record_id, classification.taxonomy_code,
            taxonomy, classification.confidence,
        )
        return classification

    def classify_batch(
        self,
        record_ids: List[str],
        taxonomy_system: Optional[str] = None,
    ) -> List[ClassificationResponse]:
        """Classify multiple spend records in batch.

        Args:
            record_ids: List of record identifiers.
            taxonomy_system: Taxonomy system override.

        Returns:
            List of ClassificationResponse instances.
        """
        update_active_batches(1)
        results = []
        for rid in record_ids:
            try:
                result = self.classify_record(rid, taxonomy_system=taxonomy_system)
                results.append(result)
            except ValueError as exc:
                logger.warning("Batch classify skipped %s: %s", rid, exc)
        update_active_batches(-1)
        return results

    def get_classification(
        self,
        classification_id: str,
    ) -> Optional[ClassificationResponse]:
        """Get a classification result by ID.

        Args:
            classification_id: Classification identifier.

        Returns:
            ClassificationResponse or None if not found.
        """
        return self._classifications.get(classification_id)

    # ------------------------------------------------------------------
    # Scope 3 mapping
    # ------------------------------------------------------------------

    def map_scope3(
        self,
        record_id: str,
    ) -> Scope3AssignmentResponse:
        """Map a spend record to a GHG Protocol Scope 3 category.

        Uses deterministic taxonomy-to-Scope-3 mapping tables.
        No LLM is used for category assignment (zero-hallucination).

        Args:
            record_id: Spend record identifier.

        Returns:
            Scope3AssignmentResponse with mapping details.

        Raises:
            ValueError: If record not found or not yet classified.
        """
        start_time = time.time()

        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Record {record_id} not found")
        if not record.taxonomy_code:
            raise ValueError(
                f"Record {record_id} must be classified before Scope 3 mapping"
            )

        # Deterministic mapping from taxonomy code to Scope 3
        scope3_cat = self._map_taxonomy_to_scope3(record.taxonomy_code)
        scope3_name = _SCOPE3_CATEGORIES.get(scope3_cat, "Unknown")

        assignment = Scope3AssignmentResponse(
            record_id=record_id,
            scope3_category=scope3_cat,
            scope3_category_name=scope3_name,
            taxonomy_code=record.taxonomy_code,
            mapping_confidence=0.85 if scope3_cat > 0 else 0.0,
            mapping_method="taxonomy_lookup",
        )
        assignment.provenance_hash = _compute_hash(assignment)
        self._scope3_assignments[assignment.assignment_id] = assignment

        # Update record
        record.scope3_category = scope3_cat
        record.scope3_category_name = scope3_name
        record.status = "mapped"
        record.updated_at = datetime.now(timezone.utc).isoformat()
        record.provenance_hash = _compute_hash(record)

        # Record metrics
        record_scope3_mapping(f"cat{scope3_cat}")
        record_processing_duration("map_scope3", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="scope3",
            entity_id=assignment.assignment_id,
            action="map",
            data_hash=assignment.provenance_hash,
        )

        # Update statistics
        self._stats.total_scope3_mapped += 1

        logger.info(
            "Mapped record %s -> Scope 3 Category %d (%s)",
            record_id, scope3_cat, scope3_name,
        )
        return assignment

    def map_scope3_batch(
        self,
        record_ids: List[str],
    ) -> List[Scope3AssignmentResponse]:
        """Map multiple records to Scope 3 categories in batch.

        Args:
            record_ids: List of record identifiers.

        Returns:
            List of Scope3AssignmentResponse instances.
        """
        update_active_batches(1)
        results = []
        for rid in record_ids:
            try:
                result = self.map_scope3(rid)
                results.append(result)
            except ValueError as exc:
                logger.warning("Batch scope3 map skipped %s: %s", rid, exc)
        update_active_batches(-1)
        return results

    def get_scope3_assignment(
        self,
        assignment_id: str,
    ) -> Optional[Scope3AssignmentResponse]:
        """Get a Scope 3 assignment by ID.

        Args:
            assignment_id: Assignment identifier.

        Returns:
            Scope3AssignmentResponse or None if not found.
        """
        return self._scope3_assignments.get(assignment_id)

    # ------------------------------------------------------------------
    # Emission calculation
    # ------------------------------------------------------------------

    def calculate_emissions(
        self,
        record_id: str,
        factor_source: Optional[str] = None,
    ) -> EmissionCalculationResponse:
        """Calculate emissions for a spend record.

        Uses deterministic spend-based emission calculation:
        emissions = spend_usd * emission_factor (kg CO2e/USD).

        No LLM is used for emission calculation (zero-hallucination).

        Args:
            record_id: Spend record identifier.
            factor_source: Emission factor source override (eeio, exiobase, defra).

        Returns:
            EmissionCalculationResponse with calculation details.

        Raises:
            ValueError: If record not found or not yet classified.
        """
        start_time = time.time()

        record = self._records.get(record_id)
        if record is None:
            raise ValueError(f"Record {record_id} not found")
        if not record.taxonomy_code:
            raise ValueError(
                f"Record {record_id} must be classified before emission calculation"
            )

        source = factor_source or "eeio"

        # Look up emission factor
        factor_info = self._lookup_emission_factor(record.taxonomy_code, source)
        record_factor_lookup(source)

        emission_factor = factor_info.get("factor", 0.0)
        emissions = record.amount_usd * emission_factor

        calculation = EmissionCalculationResponse(
            record_id=record_id,
            emissions_kg_co2e=round(emissions, 4),
            emission_factor=emission_factor,
            emission_factor_source=source,
            emission_factor_version=self._get_factor_version(source),
            taxonomy_code=record.taxonomy_code,
            spend_usd=record.amount_usd,
            methodology="spend_based",
        )
        calculation.provenance_hash = _compute_hash(calculation)
        self._emission_calculations[calculation.calculation_id] = calculation

        # Update record
        record.emissions_kg_co2e = round(emissions, 4)
        record.emission_factor = emission_factor
        record.emission_factor_source = source
        record.status = "calculated"
        record.updated_at = datetime.now(timezone.utc).isoformat()
        record.provenance_hash = _compute_hash(record)

        # Record metrics
        record_emission_calculation(source)
        record_processing_duration("calculate_emissions", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="emission",
            entity_id=calculation.calculation_id,
            action="calculate",
            data_hash=calculation.provenance_hash,
        )

        # Update statistics
        self._stats.total_emissions_calculated += 1
        self._stats.total_emissions_kg_co2e += round(emissions, 4)

        logger.info(
            "Calculated emissions for record %s: %.4f kg CO2e "
            "(USD %.2f x factor %.4f from %s)",
            record_id, emissions, record.amount_usd, emission_factor, source,
        )
        return calculation

    def calculate_emissions_batch(
        self,
        record_ids: List[str],
        factor_source: Optional[str] = None,
    ) -> List[EmissionCalculationResponse]:
        """Calculate emissions for multiple records in batch.

        Args:
            record_ids: List of record identifiers.
            factor_source: Emission factor source override.

        Returns:
            List of EmissionCalculationResponse instances.
        """
        update_active_batches(1)
        results = []
        for rid in record_ids:
            try:
                result = self.calculate_emissions(rid, factor_source=factor_source)
                results.append(result)
            except ValueError as exc:
                logger.warning("Batch emission calc skipped %s: %s", rid, exc)
        update_active_batches(-1)
        return results

    def get_emission_factor(
        self,
        taxonomy_code: str,
        source: str = "eeio",
    ) -> Dict[str, Any]:
        """Get the emission factor for a taxonomy code.

        Args:
            taxonomy_code: Taxonomy code to look up.
            source: Emission factor source (eeio, exiobase, defra).

        Returns:
            Dict with factor, source, description.
        """
        return self._lookup_emission_factor(taxonomy_code, source)

    def list_emission_factors(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List available emission factors.

        Args:
            limit: Maximum number of factors to return.
            offset: Number of factors to skip.

        Returns:
            List of emission factor dicts.
        """
        factors = []
        for code, info in self._emission_factors.items():
            factors.append({
                "taxonomy_code": code,
                "factor": info.get("factor", 0.0),
                "source": info.get("source", "eeio"),
                "description": info.get("desc", ""),
            })
        return factors[offset:offset + limit]

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def create_rule(
        self,
        name: str,
        taxonomy_code: str,
        conditions: Dict[str, Any],
        taxonomy_system: str = "unspsc",
        scope3_category: Optional[int] = None,
        description: str = "",
        priority: int = 100,
    ) -> CategoryRuleResponse:
        """Create a new classification rule.

        Args:
            name: Rule display name.
            taxonomy_code: Target taxonomy code to assign.
            conditions: Rule conditions (vendor_pattern, keywords, gl_codes, etc.).
            taxonomy_system: Taxonomy system (unspsc, naics, nace).
            scope3_category: Optional Scope 3 category override.
            description: Rule description.
            priority: Rule priority (lower = higher priority).

        Returns:
            CategoryRuleResponse with rule details.

        Raises:
            ValueError: If name or taxonomy_code is empty.
        """
        start_time = time.time()

        if not name.strip():
            raise ValueError("Rule name must not be empty")
        if not taxonomy_code.strip():
            raise ValueError("Taxonomy code must not be empty")

        rule = CategoryRuleResponse(
            name=name,
            description=description,
            taxonomy_code=taxonomy_code,
            taxonomy_system=taxonomy_system,
            scope3_category=scope3_category,
            conditions=conditions,
            priority=priority,
            is_active=True,
        )
        rule.provenance_hash = _compute_hash(rule)
        self._rules[rule.rule_id] = rule

        # Record provenance
        self.provenance.record(
            entity_type="rule",
            entity_id=rule.rule_id,
            action="create",
            data_hash=rule.provenance_hash,
        )

        # Update statistics
        self._stats.total_rules += 1
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        record_processing_duration("create_rule", time.time() - start_time)

        logger.info(
            "Created rule %s (%s -> %s, priority=%d)",
            rule.rule_id, name, taxonomy_code, priority,
        )
        return rule

    def get_rule(self, rule_id: str) -> Optional[CategoryRuleResponse]:
        """Get a classification rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            CategoryRuleResponse or None if not found.
        """
        return self._rules.get(rule_id)

    def list_rules(
        self,
        is_active: Optional[bool] = None,
        taxonomy_system: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CategoryRuleResponse]:
        """List classification rules with optional filters.

        Args:
            is_active: Optional active status filter.
            taxonomy_system: Optional taxonomy system filter.
            limit: Maximum number of rules to return.
            offset: Number of rules to skip.

        Returns:
            List of CategoryRuleResponse instances.
        """
        rules = list(self._rules.values())

        if is_active is not None:
            rules = [r for r in rules if r.is_active == is_active]
        if taxonomy_system is not None:
            rules = [r for r in rules if r.taxonomy_system == taxonomy_system]

        # Sort by priority (lower = higher priority)
        rules.sort(key=lambda r: r.priority)
        return rules[offset:offset + limit]

    def update_rule(
        self,
        rule_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        taxonomy_code: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        priority: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> CategoryRuleResponse:
        """Update an existing classification rule.

        Args:
            rule_id: Rule identifier.
            name: New rule name (optional).
            description: New description (optional).
            taxonomy_code: New taxonomy code (optional).
            conditions: New conditions (optional).
            priority: New priority (optional).
            is_active: New active status (optional).

        Returns:
            Updated CategoryRuleResponse.

        Raises:
            ValueError: If rule not found.
        """
        start_time = time.time()

        rule = self._rules.get(rule_id)
        if rule is None:
            raise ValueError(f"Rule {rule_id} not found")

        if name is not None:
            rule.name = name
        if description is not None:
            rule.description = description
        if taxonomy_code is not None:
            rule.taxonomy_code = taxonomy_code
        if conditions is not None:
            rule.conditions = conditions
        if priority is not None:
            rule.priority = priority
        if is_active is not None:
            rule.is_active = is_active

        rule.updated_at = datetime.now(timezone.utc).isoformat()
        rule.provenance_hash = _compute_hash(rule)

        # Record provenance
        self.provenance.record(
            entity_type="rule",
            entity_id=rule_id,
            action="update",
            data_hash=rule.provenance_hash,
        )

        # Update statistics
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        record_processing_duration("update_rule", time.time() - start_time)

        logger.info("Updated rule %s", rule_id)
        return rule

    def delete_rule(self, rule_id: str) -> bool:
        """Delete a classification rule.

        Args:
            rule_id: Rule identifier.

        Returns:
            True if deleted, False if not found.
        """
        rule = self._rules.pop(rule_id, None)
        if rule is None:
            return False

        self.provenance.record(
            entity_type="rule",
            entity_id=rule_id,
            action="delete",
            data_hash=_compute_hash({"rule_id": rule_id, "deleted": True}),
        )

        self._stats.total_rules = max(0, self._stats.total_rules - 1)
        self._stats.active_rules = sum(
            1 for r in self._rules.values() if r.is_active
        )

        logger.info("Deleted rule %s", rule_id)
        return True

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_analytics(self) -> AnalyticsResponse:
        """Get analytics summary for all spend data.

        All analytics are deterministic aggregations of recorded data.
        No LLM is used for analytics computation (zero-hallucination).

        Returns:
            AnalyticsResponse with summary metrics.
        """
        start_time = time.time()

        records = list(self._records.values())
        classified = [r for r in records if r.taxonomy_code]
        mapped = [r for r in records if r.scope3_category > 0]
        calculated = [r for r in records if r.emissions_kg_co2e > 0]

        total_spend = sum(r.amount_usd for r in records)
        total_emissions = sum(r.emissions_kg_co2e for r in calculated)
        avg_conf = (
            sum(r.classification_confidence for r in classified) / max(len(classified), 1)
        )
        class_rate = (len(classified) / max(len(records), 1)) * 100.0

        # Top categories by spend
        cat_spend: Dict[str, float] = {}
        for r in classified:
            key = r.taxonomy_code
            cat_spend[key] = cat_spend.get(key, 0.0) + r.amount_usd
        top_categories = sorted(
            [{"taxonomy_code": k, "spend_usd": v} for k, v in cat_spend.items()],
            key=lambda x: x["spend_usd"],
            reverse=True,
        )[:10]

        # Top vendors by spend
        vendor_spend: Dict[str, float] = {}
        for r in records:
            key = r.vendor_name or "Unknown"
            vendor_spend[key] = vendor_spend.get(key, 0.0) + r.amount_usd
        top_vendors = sorted(
            [{"vendor_name": k, "spend_usd": v} for k, v in vendor_spend.items()],
            key=lambda x: x["spend_usd"],
            reverse=True,
        )[:10]

        # Scope 3 breakdown
        scope3_data: Dict[int, Dict[str, float]] = {}
        for r in mapped:
            cat = r.scope3_category
            if cat not in scope3_data:
                scope3_data[cat] = {"spend_usd": 0.0, "emissions_kg_co2e": 0.0}
            scope3_data[cat]["spend_usd"] += r.amount_usd
            scope3_data[cat]["emissions_kg_co2e"] += r.emissions_kg_co2e
        scope3_breakdown = [
            {
                "category": k,
                "category_name": _SCOPE3_CATEGORIES.get(k, "Unknown"),
                "spend_usd": v["spend_usd"],
                "emissions_kg_co2e": v["emissions_kg_co2e"],
            }
            for k, v in sorted(scope3_data.items())
        ]

        analytics = AnalyticsResponse(
            total_records=len(records),
            total_classified=len(classified),
            total_mapped=len(mapped),
            total_calculated=len(calculated),
            total_spend_usd=round(total_spend, 2),
            total_emissions_kg_co2e=round(total_emissions, 4),
            avg_confidence=round(avg_conf, 4),
            classification_rate_pct=round(class_rate, 2),
            top_categories=top_categories,
            top_vendors=top_vendors,
            scope3_breakdown=scope3_breakdown,
        )
        analytics.provenance_hash = _compute_hash(analytics)

        record_processing_duration("analytics", time.time() - start_time)

        self.provenance.record(
            entity_type="analytics",
            entity_id="global",
            action="generate",
            data_hash=analytics.provenance_hash,
        )

        logger.info(
            "Generated analytics: %d records, %d classified (%.1f%%), "
            "USD %.2f total, %.4f kg CO2e",
            len(records), len(classified), class_rate,
            total_spend, total_emissions,
        )
        return analytics

    def get_hotspots(
        self,
        top_n: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get emission hotspots (highest emitting categories/vendors).

        Args:
            top_n: Number of top hotspots to return.

        Returns:
            List of hotspot dicts sorted by emissions descending.
        """
        start_time = time.time()

        records = [
            r for r in self._records.values()
            if r.emissions_kg_co2e > 0
        ]

        # Hotspots by taxonomy code
        hotspot_data: Dict[str, Dict[str, Any]] = {}
        for r in records:
            key = r.taxonomy_code or "unclassified"
            if key not in hotspot_data:
                hotspot_data[key] = {
                    "taxonomy_code": key,
                    "taxonomy_description": r.taxonomy_description,
                    "spend_usd": 0.0,
                    "emissions_kg_co2e": 0.0,
                    "record_count": 0,
                }
            hotspot_data[key]["spend_usd"] += r.amount_usd
            hotspot_data[key]["emissions_kg_co2e"] += r.emissions_kg_co2e
            hotspot_data[key]["record_count"] += 1

        hotspots = sorted(
            hotspot_data.values(),
            key=lambda x: x["emissions_kg_co2e"],
            reverse=True,
        )[:top_n]

        record_processing_duration("hotspots", time.time() - start_time)

        logger.info("Generated %d hotspots", len(hotspots))
        return hotspots

    def get_trends(
        self,
        group_by: str = "month",
    ) -> List[Dict[str, Any]]:
        """Get spend and emissions trends over time.

        Args:
            group_by: Time grouping (day, week, month, quarter, year).

        Returns:
            List of trend data points.
        """
        start_time = time.time()

        records = list(self._records.values())
        period_data: Dict[str, Dict[str, float]] = {}

        for r in records:
            if not r.date:
                continue
            period_key = self._extract_period(r.date, group_by)
            if period_key not in period_data:
                period_data[period_key] = {
                    "spend_usd": 0.0,
                    "emissions_kg_co2e": 0.0,
                    "record_count": 0,
                }
            period_data[period_key]["spend_usd"] += r.amount_usd
            period_data[period_key]["emissions_kg_co2e"] += r.emissions_kg_co2e
            period_data[period_key]["record_count"] += 1

        trends = [
            {"period": k, **v}
            for k, v in sorted(period_data.items())
        ]

        record_processing_duration("trends", time.time() - start_time)

        logger.info("Generated %d trend data points (group_by=%s)", len(trends), group_by)
        return trends

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(
        self,
        report_type: str = "summary",
        report_format: str = "json",
    ) -> ReportResponse:
        """Generate a spend categorization report.

        Args:
            report_type: Report type (summary, detailed, emissions, scope3).
            report_format: Output format (json, csv, excel, pdf).

        Returns:
            ReportResponse with report content or download reference.
        """
        start_time = time.time()

        analytics = self.get_analytics()

        records = list(self._records.values())
        total_spend = sum(r.amount_usd for r in records)
        total_emissions = sum(r.emissions_kg_co2e for r in records)

        content: Any = None
        if report_format == "json":
            content = {
                "report_type": report_type,
                "analytics": analytics.model_dump(mode="json"),
                "records_summary": {
                    "total": len(records),
                    "classified": analytics.total_classified,
                    "mapped": analytics.total_mapped,
                    "calculated": analytics.total_calculated,
                },
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

            if report_type == "detailed":
                content["records"] = [
                    r.model_dump(mode="json") for r in records
                ]
            elif report_type == "emissions":
                content["emission_calculations"] = [
                    c.model_dump(mode="json")
                    for c in self._emission_calculations.values()
                ]
            elif report_type == "scope3":
                content["scope3_assignments"] = [
                    a.model_dump(mode="json")
                    for a in self._scope3_assignments.values()
                ]

        report = ReportResponse(
            report_type=report_type,
            format=report_format,
            record_count=len(records),
            total_spend_usd=round(total_spend, 2),
            total_emissions_kg_co2e=round(total_emissions, 4),
            content=content,
        )
        report.provenance_hash = _compute_hash(report)
        self._reports[report.report_id] = report

        # Record metrics
        record_report_generation(report_format)
        record_processing_duration("generate_report", time.time() - start_time)

        # Record provenance
        self.provenance.record(
            entity_type="report",
            entity_id=report.report_id,
            action="generate",
            data_hash=report.provenance_hash,
        )

        # Update statistics
        self._stats.total_reports += 1

        logger.info(
            "Generated %s report (%s): %d records, USD %.2f, %.4f kg CO2e",
            report_type, report_format, len(records), total_spend, total_emissions,
        )
        return report

    # ------------------------------------------------------------------
    # Statistics and health
    # ------------------------------------------------------------------

    def get_statistics(self) -> SpendCategorizerStatisticsResponse:
        """Get aggregated spend categorizer statistics.

        Returns:
            SpendCategorizerStatisticsResponse summary.
        """
        return self._stats

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the service.

        Returns:
            Health status dict.
        """
        return {
            "status": "healthy" if self._started else "not_started",
            "service": "spend-categorizer",
            "started": self._started,
            "records": len(self._records),
            "classifications": len(self._classifications),
            "scope3_assignments": len(self._scope3_assignments),
            "emission_calculations": len(self._emission_calculations),
            "rules": len(self._rules),
            "reports": len(self._reports),
            "provenance_entries": self.provenance.entry_count,
            "prometheus_available": PROMETHEUS_AVAILABLE,
        }

    # ------------------------------------------------------------------
    # Convenience getters
    # ------------------------------------------------------------------

    def get_provenance(self) -> _ProvenanceTracker:
        """Get the ProvenanceTracker instance.

        Returns:
            _ProvenanceTracker used by this service.
        """
        return self.provenance

    def get_metrics(self) -> Dict[str, Any]:
        """Get spend categorizer service metrics summary.

        Returns:
            Dictionary with service metric summaries.
        """
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "started": self._started,
            "total_records": self._stats.total_records,
            "total_classified": self._stats.total_classified,
            "total_scope3_mapped": self._stats.total_scope3_mapped,
            "total_emissions_calculated": self._stats.total_emissions_calculated,
            "total_rules": self._stats.total_rules,
            "active_rules": self._stats.active_rules,
            "total_reports": self._stats.total_reports,
            "total_spend_usd": self._stats.total_spend_usd,
            "total_emissions_kg_co2e": self._stats.total_emissions_kg_co2e,
            "avg_confidence": self._stats.avg_confidence,
            "provenance_entries": self.provenance.entry_count,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_by_rules(
        self,
        record: SpendRecordResponse,
        taxonomy_system: str,
    ) -> Optional[ClassificationResponse]:
        """Attempt to classify a record using active rules.

        Rules are evaluated in priority order (lowest number = highest priority).

        Args:
            record: Spend record to classify.
            taxonomy_system: Target taxonomy system.

        Returns:
            ClassificationResponse if a rule matches, None otherwise.
        """
        active_rules = sorted(
            [r for r in self._rules.values() if r.is_active and r.taxonomy_system == taxonomy_system],
            key=lambda r: r.priority,
        )

        for rule in active_rules:
            if self._rule_matches(rule, record):
                record_rule_evaluation("match")
                rule.match_count += 1

                return ClassificationResponse(
                    record_id=record.record_id,
                    taxonomy_code=rule.taxonomy_code,
                    taxonomy_description=rule.description or rule.name,
                    taxonomy_system=taxonomy_system,
                    confidence=0.95,
                    confidence_label="HIGH",
                    rule_id=rule.rule_id,
                    method="rule",
                )

            record_rule_evaluation("no_match")

        return None

    def _rule_matches(
        self,
        rule: CategoryRuleResponse,
        record: SpendRecordResponse,
    ) -> bool:
        """Check if a rule matches a spend record.

        Evaluates vendor_pattern, keywords, and gl_codes conditions.

        Args:
            rule: Rule to evaluate.
            record: Record to match against.

        Returns:
            True if all conditions match.
        """
        conditions = rule.conditions

        # Vendor pattern match
        vendor_pattern = conditions.get("vendor_pattern", "")
        if vendor_pattern:
            if vendor_pattern.lower() not in record.vendor_name.lower():
                return False

        # Keyword match (any keyword in description)
        keywords = conditions.get("keywords", [])
        if keywords:
            desc_lower = record.description.lower()
            if not any(kw.lower() in desc_lower for kw in keywords):
                return False

        # GL account match
        gl_codes = conditions.get("gl_codes", [])
        if gl_codes:
            if record.gl_account not in gl_codes:
                return False

        # Cost center match
        cost_centers = conditions.get("cost_centers", [])
        if cost_centers:
            if record.cost_center not in cost_centers:
                return False

        return True

    def _classify_by_keyword(
        self,
        record: SpendRecordResponse,
        taxonomy_system: str,
    ) -> ClassificationResponse:
        """Classify a record using keyword-based matching.

        Deterministic keyword matching against known categories.

        Args:
            record: Spend record to classify.
            taxonomy_system: Target taxonomy system.

        Returns:
            ClassificationResponse with best match.
        """
        desc = record.description.lower()
        vendor = record.vendor_name.lower()
        combined = f"{desc} {vendor}"

        # Keyword-to-taxonomy mapping (deterministic)
        keyword_map = {
            "43000000": (["computer", "laptop", "server", "software", "it ", "hardware", "tech"], "IT Equipment"),
            "44000000": (["office", "paper", "pen", "stationery", "supplies", "printer"], "Office Supplies"),
            "78000000": (["transport", "shipping", "freight", "logistics", "delivery", "courier"], "Transportation"),
            "80000000": (["consulting", "advisory", "management", "professional", "legal"], "Management Services"),
            "72000000": (["building", "construction", "maintenance", "facilities", "hvac"], "Building Materials"),
            "47000000": (["cleaning", "janitorial", "sanitiz", "detergent", "chemical"], "Cleaning Products"),
            "50000000": (["food", "catering", "beverage", "meal", "restaurant", "canteen"], "Food and Beverage"),
            "15000000": (["fuel", "gas", "diesel", "petrol", "energy", "electricity"], "Fuels and Lubricants"),
            "26000000": (["electrical", "wiring", "cable", "power supply", "transformer"], "Electrical Equipment"),
            "76000000": (["cleaning service", "waste management", "recycling", "disposal"], "Cleaning Services"),
        }

        best_code = ""
        best_desc = "Uncategorized"
        best_confidence = 0.0

        for code, (keywords, description) in keyword_map.items():
            match_count = sum(1 for kw in keywords if kw in combined)
            if match_count > 0:
                confidence = min(0.3 + (match_count * 0.15), 0.85)
                if confidence > best_confidence:
                    best_code = code
                    best_desc = description
                    best_confidence = confidence

        # Determine confidence label
        if best_confidence >= self.config.high_confidence_threshold:
            confidence_label = "HIGH"
        elif best_confidence >= self.config.medium_confidence_threshold:
            confidence_label = "MEDIUM"
        else:
            confidence_label = "LOW"

        return ClassificationResponse(
            record_id=record.record_id,
            taxonomy_code=best_code,
            taxonomy_description=best_desc,
            taxonomy_system=taxonomy_system,
            confidence=round(best_confidence, 4),
            confidence_label=confidence_label,
            method="keyword",
        )

    def _map_taxonomy_to_scope3(self, taxonomy_code: str) -> int:
        """Map a taxonomy code to a GHG Protocol Scope 3 category.

        Deterministic mapping using taxonomy code prefix ranges.

        Args:
            taxonomy_code: UNSPSC or other taxonomy code.

        Returns:
            Scope 3 category number (1-15), or 1 as default.
        """
        if not taxonomy_code:
            return 1

        # UNSPSC-based mapping to Scope 3 categories
        prefix = taxonomy_code[:2] if len(taxonomy_code) >= 2 else taxonomy_code
        mapping = {
            "10": 1, "11": 1, "12": 1, "13": 1, "14": 1,
            "15": 3,  # Fuels -> Fuel-related
            "20": 2, "21": 2, "22": 2, "23": 2, "24": 2,
            "25": 2, "26": 2, "27": 2,
            "30": 1, "31": 1, "32": 1, "39": 1,
            "40": 1, "41": 1, "42": 1, "43": 1, "44": 1,
            "45": 1, "46": 1, "47": 1,
            "50": 1, "51": 1, "52": 1, "53": 1,
            "55": 1, "56": 1,
            "60": 1,
            "70": 1, "71": 1, "72": 2, "73": 1,
            "76": 1,
            "78": 4,  # Transportation
            "80": 1, "81": 1, "82": 1, "83": 1, "84": 1,
            "85": 1, "86": 6,  # Travel
            "90": 5, "91": 5, "92": 5, "93": 5,  # Waste
            "95": 1,
        }
        return mapping.get(prefix, 1)

    def _lookup_emission_factor(
        self,
        taxonomy_code: str,
        source: str,
    ) -> Dict[str, Any]:
        """Look up an emission factor for a taxonomy code.

        Args:
            taxonomy_code: Taxonomy code.
            source: Factor source database.

        Returns:
            Dict with factor value and metadata.
        """
        # Check exact match first
        factor_info = self._emission_factors.get(taxonomy_code)
        if factor_info is not None:
            return factor_info

        # Try prefix match (first 2 digits of UNSPSC)
        prefix = taxonomy_code[:2] + "000000" if len(taxonomy_code) >= 2 else ""
        factor_info = self._emission_factors.get(prefix)
        if factor_info is not None:
            return factor_info

        # Default factor
        return {
            "factor": 0.25,
            "source": source,
            "desc": "Default emission factor",
        }

    def _get_factor_version(self, source: str) -> str:
        """Get the version string for an emission factor source.

        Args:
            source: Factor source (eeio, exiobase, defra, ecoinvent).

        Returns:
            Version string.
        """
        version_map = {
            "eeio": self.config.eeio_version,
            "exiobase": self.config.exiobase_version,
            "defra": self.config.defra_version,
            "ecoinvent": self.config.ecoinvent_version,
        }
        return version_map.get(source, "unknown")

    def _extract_period(self, date_str: str, group_by: str) -> str:
        """Extract a period key from a date string.

        Args:
            date_str: ISO 8601 date string.
            group_by: Grouping type (day, week, month, quarter, year).

        Returns:
            Period key string.
        """
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return "unknown"

        if group_by == "day":
            return dt.strftime("%Y-%m-%d")
        elif group_by == "week":
            return f"{dt.year}-W{dt.isocalendar()[1]:02d}"
        elif group_by == "month":
            return dt.strftime("%Y-%m")
        elif group_by == "quarter":
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{quarter}"
        elif group_by == "year":
            return str(dt.year)
        else:
            return dt.strftime("%Y-%m")

    def _update_avg_confidence(self, confidence: float) -> None:
        """Update running average classification confidence.

        Args:
            confidence: Latest confidence value.
        """
        total = self._stats.total_classified
        if total <= 0:
            self._stats.avg_confidence = confidence
            return
        prev_avg = self._stats.avg_confidence
        self._stats.avg_confidence = (
            (prev_avg * (total - 1) + confidence) / total
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Start the spend categorizer service.

        Safe to call multiple times.
        """
        if self._started:
            logger.debug("SpendCategorizerService already started; skipping")
            return

        logger.info("SpendCategorizerService starting up...")
        self._started = True
        logger.info("SpendCategorizerService startup complete")

    def shutdown(self) -> None:
        """Shutdown the spend categorizer service and release resources."""
        if not self._started:
            return

        self._started = False
        logger.info("SpendCategorizerService shut down")


# ===================================================================
# Thread-safe singleton access
# ===================================================================


def _get_singleton() -> SpendCategorizerService:
    """Get or create the singleton SpendCategorizerService instance.

    Returns:
        The singleton SpendCategorizerService.
    """
    global _singleton_instance
    if _singleton_instance is None:
        with _singleton_lock:
            if _singleton_instance is None:
                _singleton_instance = SpendCategorizerService()
    return _singleton_instance


# ===================================================================
# FastAPI integration
# ===================================================================


async def configure_spend_categorizer(
    app: Any,
    config: Optional[SpendCategorizerConfig] = None,
) -> SpendCategorizerService:
    """Configure the Spend Categorizer Service on a FastAPI application.

    Creates the SpendCategorizerService, stores it in app.state, mounts
    the spend categorizer API router, and starts the service.

    Args:
        app: FastAPI application instance.
        config: Optional spend categorizer config.

    Returns:
        SpendCategorizerService instance.
    """
    global _singleton_instance

    service = SpendCategorizerService(config=config)

    # Store as singleton
    with _singleton_lock:
        _singleton_instance = service

    # Attach to app state
    app.state.spend_categorizer_service = service

    # Mount spend categorizer API router
    try:
        from greenlang.spend_categorizer.api.router import router as cat_router
        if cat_router is not None:
            app.include_router(cat_router)
            logger.info("Spend categorizer service API router mounted")
    except ImportError:
        logger.warning(
            "Spend categorizer router not available; API not mounted"
        )

    # Start service
    service.startup()

    logger.info("Spend categorizer service configured on app")
    return service


def get_spend_categorizer(app: Any) -> SpendCategorizerService:
    """Get the SpendCategorizerService instance from app state.

    Args:
        app: FastAPI application instance.

    Returns:
        SpendCategorizerService instance.

    Raises:
        RuntimeError: If spend categorizer service not configured.
    """
    service = getattr(app.state, "spend_categorizer_service", None)
    if service is None:
        raise RuntimeError(
            "Spend categorizer service not configured. "
            "Call configure_spend_categorizer(app) first."
        )
    return service


def get_router(service: Optional[SpendCategorizerService] = None) -> Any:
    """Get the spend categorizer API router.

    Args:
        service: Optional service instance (unused, kept for API compat).

    Returns:
        FastAPI APIRouter or None if FastAPI not available.
    """
    try:
        from greenlang.spend_categorizer.api.router import router
        return router
    except ImportError:
        return None


__all__ = [
    "SpendCategorizerService",
    "configure_spend_categorizer",
    "get_spend_categorizer",
    "get_router",
    # Models
    "SpendRecordResponse",
    "ClassificationResponse",
    "Scope3AssignmentResponse",
    "EmissionCalculationResponse",
    "CategoryRuleResponse",
    "AnalyticsResponse",
    "ReportResponse",
    "SpendCategorizerStatisticsResponse",
]
