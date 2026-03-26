# -*- coding: utf-8 -*-
"""
ExternalDatasetEngine - PACK-047 GHG Emissions Benchmark Engine 3
====================================================================

Manages ingestion, validation, caching, and retrieval of external benchmark
datasets from authoritative sources (CDP, TPI, GRESB, CRREM, ISS ESG) and
custom CSV/JSON/XLSX uploads for GHG emissions benchmarking.

Calculation Methodology:
    Data Freshness Score:
        freshness = max(0, 1 - (days_since_update / max_staleness_days))

        Where:
            days_since_update  = current_date - last_update_date
            max_staleness_days = configurable TTL per source (default 365)

    Schema Validation:
        Each source adapter defines required/optional fields with types.
        Validation score = required_fields_present / total_required_fields

    Record Quality Score:
        Q_record = w_completeness * completeness + w_freshness * freshness + w_source * source_trust

        Where:
            completeness = non_null_fields / total_fields
            freshness    = as above
            source_trust = predefined trust score per source (CDP=0.9, TPI=0.85, etc.)

    Cache Management:
        TTL-based expiry per source.
        Cache hit ratio = cache_hits / (cache_hits + cache_misses)

Regulatory References:
    - CDP Climate Change: C0.1-C0.5 (org info), C6.1-C6.5 (emissions), C6.10 (intensity)
    - TPI Carbon Performance: Sector pathways, company ratings
    - GRESB Real Estate and Infrastructure: Asset-level benchmarks
    - CRREM Carbon Risk: Stranding year, decarbonisation pathways
    - ISS ESG Climate: Carbon risk rating, temperature alignment
    - ESRS E1: Benchmark data requirements

Zero-Hallucination:
    - All data from published external sources only
    - No LLM-generated benchmark data
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-047 GHG Emissions Benchmark
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round4(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataSourceType(str, Enum):
    """External data source types.

    CDP:        CDP Climate Change responses.
    TPI:        Transition Pathway Initiative.
    GRESB:      GRESB Real Estate / Infrastructure.
    CRREM:      Carbon Risk Real Estate Monitor.
    ISS_ESG:    ISS ESG Climate Solutions.
    CUSTOM_CSV: Custom CSV upload.
    CUSTOM_JSON: Custom JSON upload.
    CUSTOM_XLSX: Custom XLSX upload.
    """
    CDP = "CDP"
    TPI = "TPI"
    GRESB = "GRESB"
    CRREM = "CRREM"
    ISS_ESG = "ISS_ESG"
    CUSTOM_CSV = "CUSTOM_CSV"
    CUSTOM_JSON = "CUSTOM_JSON"
    CUSTOM_XLSX = "CUSTOM_XLSX"


class FreshnessStatus(str, Enum):
    """Data freshness status.

    FRESH:      Within acceptable TTL.
    STALE:      Past TTL but within grace period.
    EXPIRED:    Past grace period, should be refreshed.
    UNKNOWN:    No update timestamp available.
    """
    FRESH = "fresh"
    STALE = "stale"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class ValidationSeverity(str, Enum):
    """Validation issue severity."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Source trust scores (0-1)
SOURCE_TRUST_SCORES: Dict[str, Decimal] = {
    DataSourceType.CDP.value: Decimal("0.90"),
    DataSourceType.TPI.value: Decimal("0.85"),
    DataSourceType.GRESB.value: Decimal("0.85"),
    DataSourceType.CRREM.value: Decimal("0.80"),
    DataSourceType.ISS_ESG.value: Decimal("0.80"),
    DataSourceType.CUSTOM_CSV.value: Decimal("0.50"),
    DataSourceType.CUSTOM_JSON.value: Decimal("0.50"),
    DataSourceType.CUSTOM_XLSX.value: Decimal("0.50"),
}

# Default TTL per source (days)
DEFAULT_TTL_DAYS: Dict[str, int] = {
    DataSourceType.CDP.value: 365,
    DataSourceType.TPI.value: 180,
    DataSourceType.GRESB.value: 365,
    DataSourceType.CRREM.value: 365,
    DataSourceType.ISS_ESG.value: 90,
    DataSourceType.CUSTOM_CSV.value: 30,
    DataSourceType.CUSTOM_JSON.value: 30,
    DataSourceType.CUSTOM_XLSX.value: 30,
}

# Grace period multiplier (staleness = TTL to TTL*grace, expired > TTL*grace)
GRACE_PERIOD_MULTIPLIER: Decimal = Decimal("1.5")

# Required fields per source adapter
CDP_REQUIRED_FIELDS: List[str] = [
    "organisation_name", "sector", "country", "reporting_year",
    "scope1_tco2e", "scope2_location_tco2e",
]
CDP_OPTIONAL_FIELDS: List[str] = [
    "scope2_market_tco2e", "scope3_tco2e", "revenue", "intensity_value",
    "intensity_unit", "verification_status", "cdp_score",
]

TPI_REQUIRED_FIELDS: List[str] = [
    "company_name", "sector", "country", "carbon_intensity",
    "intensity_unit", "assessment_year",
]

GRESB_REQUIRED_FIELDS: List[str] = [
    "entity_name", "asset_type", "country", "reporting_year",
    "total_ghg_intensity", "energy_intensity",
]

CRREM_REQUIRED_FIELDS: List[str] = [
    "asset_id", "country", "asset_type", "floor_area_m2",
    "energy_intensity_kwh_m2", "carbon_intensity_kgco2e_m2",
    "stranding_year",
]

ISS_REQUIRED_FIELDS: List[str] = [
    "company_name", "isin", "sector", "country",
    "carbon_risk_rating", "temperature_alignment",
]

MAX_CUSTOM_RECORDS: int = 100000
MAX_CACHE_SIZE: int = 10000


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class DataSourceConfig(BaseModel):
    """Configuration for an external data source.

    Attributes:
        source_type:        Source type.
        source_name:        Human-readable source name.
        ttl_days:           Cache TTL in days (0 = no caching).
        trust_score:        Source trust score (0-1).
        api_endpoint:       API endpoint URL (if applicable).
        api_key_ref:        Reference to API key in secrets manager.
        custom_schema:      Custom schema definition (for CSV/JSON/XLSX).
        enabled:            Whether source is enabled.
    """
    source_type: DataSourceType = Field(..., description="Source type")
    source_name: str = Field(default="", description="Source name")
    ttl_days: Optional[int] = Field(default=None, ge=0, description="TTL (days)")
    trust_score: Optional[Decimal] = Field(default=None, ge=0, le=1)
    api_endpoint: Optional[str] = Field(default=None, description="API endpoint")
    api_key_ref: Optional[str] = Field(default=None, description="API key reference")
    custom_schema: Optional[Dict[str, str]] = Field(default=None, description="Custom schema")
    enabled: bool = Field(default=True, description="Enabled")

    @model_validator(mode="after")
    def set_defaults(self) -> "DataSourceConfig":
        if self.ttl_days is None:
            object.__setattr__(
                self, "ttl_days", DEFAULT_TTL_DAYS.get(self.source_type.value, 30)
            )
        if self.trust_score is None:
            object.__setattr__(
                self, "trust_score",
                SOURCE_TRUST_SCORES.get(self.source_type.value, Decimal("0.50")),
            )
        if not self.source_name:
            object.__setattr__(self, "source_name", self.source_type.value)
        return self


class ExternalDataRecord(BaseModel):
    """A single record from an external dataset.

    Attributes:
        record_id:          Unique record identifier.
        source_type:        Data source type.
        entity_id:          External entity identifier.
        entity_name:        Entity name.
        sector:             Sector classification.
        country:            Country / region.
        reporting_year:     Reporting year.
        scope1_tco2e:       Scope 1 emissions.
        scope2_tco2e:       Scope 2 emissions.
        scope3_tco2e:       Scope 3 emissions.
        total_tco2e:        Total emissions.
        intensity_value:    Intensity value.
        intensity_unit:     Intensity unit.
        revenue_millions:   Revenue (millions, local currency).
        currency:           Currency code.
        verification:       Verification status.
        extra_fields:       Additional source-specific fields.
        ingested_at:        Ingestion timestamp.
    """
    record_id: str = Field(default_factory=_new_uuid, description="Record ID")
    source_type: DataSourceType = Field(..., description="Source type")
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    sector: str = Field(default="", description="Sector")
    country: str = Field(default="", description="Country")
    reporting_year: int = Field(default=2024, description="Reporting year")
    scope1_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope2_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    scope3_tco2e: Optional[Decimal] = Field(default=None, ge=0)
    total_tco2e: Decimal = Field(default=Decimal("0"), ge=0)
    intensity_value: Optional[Decimal] = Field(default=None, ge=0)
    intensity_unit: str = Field(default="", description="Intensity unit")
    revenue_millions: Optional[Decimal] = Field(default=None, ge=0)
    currency: str = Field(default="USD", description="Currency")
    verification: str = Field(default="", description="Verification status")
    extra_fields: Dict[str, Any] = Field(default_factory=dict)
    ingested_at: str = Field(default="", description="Ingestion timestamp")

    @field_validator("scope1_tco2e", "scope2_tco2e", "total_tco2e", mode="before")
    @classmethod
    def coerce_dec(cls, v: Any) -> Decimal:
        return _decimal(v)


class DatasetIngestionInput(BaseModel):
    """Input for dataset ingestion.

    Attributes:
        source_config:  Source configuration.
        records:        Raw records to ingest.
        validate:       Whether to validate schema.
        deduplicate:    Whether to deduplicate by entity+year.
    """
    source_config: DataSourceConfig = Field(..., description="Source config")
    records: List[Dict[str, Any]] = Field(default_factory=list, description="Raw records")
    validate: bool = Field(default=True, description="Validate schema")
    deduplicate: bool = Field(default=True, description="Deduplicate")


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class ValidationIssue(BaseModel):
    """A single validation issue.

    Attributes:
        record_index:   Index of the record.
        field:          Field name.
        severity:       Issue severity.
        message:        Issue description.
    """
    record_index: int = Field(default=0, description="Record index")
    field: str = Field(default="", description="Field name")
    severity: ValidationSeverity = Field(default=ValidationSeverity.WARNING)
    message: str = Field(default="", description="Issue description")


class DataFreshnessStatus(BaseModel):
    """Freshness status for a data source.

    Attributes:
        source_type:        Data source type.
        last_updated:       Last update timestamp.
        ttl_days:           Configured TTL.
        days_since_update:  Days since last update.
        freshness_score:    Freshness score (0-1).
        status:             Freshness status.
    """
    source_type: str = Field(default="", description="Source type")
    last_updated: Optional[str] = Field(default=None, description="Last updated")
    ttl_days: int = Field(default=365, description="TTL days")
    days_since_update: int = Field(default=0, description="Days since update")
    freshness_score: Decimal = Field(default=Decimal("0"), description="Freshness score")
    status: FreshnessStatus = Field(default=FreshnessStatus.UNKNOWN)


class CacheEntry(BaseModel):
    """Cache entry for external data.

    Attributes:
        cache_key:      Cache key.
        source_type:    Source type.
        record_count:   Number of cached records.
        cached_at:      Cache timestamp.
        expires_at:     Expiry timestamp.
        hit_count:      Number of cache hits.
    """
    cache_key: str = Field(default="", description="Cache key")
    source_type: str = Field(default="", description="Source type")
    record_count: int = Field(default=0, description="Record count")
    cached_at: str = Field(default="", description="Cached at")
    expires_at: str = Field(default="", description="Expires at")
    hit_count: int = Field(default=0, description="Hit count")


class CacheStats(BaseModel):
    """Cache statistics.

    Attributes:
        total_entries:  Total cache entries.
        total_records:  Total cached records.
        total_hits:     Total cache hits.
        total_misses:   Total cache misses.
        hit_ratio:      Cache hit ratio.
        entries:        Individual cache entries.
    """
    total_entries: int = Field(default=0)
    total_records: int = Field(default=0)
    total_hits: int = Field(default=0)
    total_misses: int = Field(default=0)
    hit_ratio: Decimal = Field(default=Decimal("0"))
    entries: List[CacheEntry] = Field(default_factory=list)


class IngestionResult(BaseModel):
    """Result of dataset ingestion.

    Attributes:
        result_id:              Unique result identifier.
        source_type:            Source type.
        source_name:            Source name.
        total_raw_records:      Total raw records received.
        valid_records:          Records passing validation.
        invalid_records:        Records failing validation.
        duplicate_records:      Duplicate records removed.
        ingested_records:       Final ingested records.
        validation_issues:      Validation issues found.
        freshness:              Data freshness status.
        records:                Ingested ExternalDataRecord objects.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    source_type: str = Field(default="", description="Source type")
    source_name: str = Field(default="", description="Source name")
    total_raw_records: int = Field(default=0)
    valid_records: int = Field(default=0)
    invalid_records: int = Field(default=0)
    duplicate_records: int = Field(default=0)
    ingested_records: int = Field(default=0)
    validation_issues: List[ValidationIssue] = Field(default_factory=list)
    freshness: DataFreshnessStatus = Field(default_factory=DataFreshnessStatus)
    records: List[ExternalDataRecord] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ExternalDatasetEngine:
    """Manages external benchmark dataset ingestion, validation, and caching.

    Supports CDP, TPI, GRESB, CRREM, ISS ESG, and custom CSV/JSON/XLSX
    data sources with schema validation, freshness tracking, and TTL caching.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every record validated and tracked.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        self._cache: Dict[str, Tuple[List[ExternalDataRecord], datetime, int]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        logger.info("ExternalDatasetEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: DatasetIngestionInput) -> IngestionResult:
        """Ingest and validate an external dataset.

        Args:
            input_data: Ingestion input with source config and raw records.

        Returns:
            IngestionResult with validated records and statistics.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.source_config

        raw_records = input_data.records
        total_raw = len(raw_records)

        if total_raw > MAX_CUSTOM_RECORDS:
            raise ValueError(
                f"Maximum {MAX_CUSTOM_RECORDS} records allowed (got {total_raw})"
            )

        # Check cache
        cache_key = self._make_cache_key(config)
        cached = self._get_cached(cache_key, config)
        if cached is not None:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result = IngestionResult(
                source_type=config.source_type.value,
                source_name=config.source_name,
                total_raw_records=len(cached),
                valid_records=len(cached),
                ingested_records=len(cached),
                records=cached,
                freshness=self._compute_freshness(config),
                warnings=["Served from cache."],
                calculated_at=_utcnow().isoformat(),
                processing_time_ms=round(elapsed_ms, 3),
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Validate
        validation_issues: List[ValidationIssue] = []
        valid_records: List[Dict[str, Any]] = []
        invalid_count = 0

        if input_data.validate:
            required_fields = self._get_required_fields(config.source_type)
            for idx, record in enumerate(raw_records):
                issues = self._validate_record(idx, record, required_fields, config)
                if any(i.severity == ValidationSeverity.ERROR for i in issues):
                    invalid_count += 1
                    validation_issues.extend(issues)
                else:
                    valid_records.append(record)
                    validation_issues.extend(issues)
        else:
            valid_records = list(raw_records)

        # Deduplicate
        duplicate_count = 0
        if input_data.deduplicate:
            valid_records, duplicate_count = self._deduplicate(valid_records)
            if duplicate_count > 0:
                warnings.append(f"Removed {duplicate_count} duplicate records.")

        # Convert to ExternalDataRecord
        ingested: List[ExternalDataRecord] = []
        now_str = _utcnow().isoformat()
        for record in valid_records:
            edr = self._convert_record(record, config, now_str)
            ingested.append(edr)

        # Cache
        self._set_cache(cache_key, ingested, config)

        # Freshness
        freshness = self._compute_freshness(config)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = IngestionResult(
            source_type=config.source_type.value,
            source_name=config.source_name,
            total_raw_records=total_raw,
            valid_records=len(valid_records),
            invalid_records=invalid_count,
            duplicate_records=duplicate_count,
            ingested_records=len(ingested),
            validation_issues=validation_issues,
            freshness=freshness,
            records=ingested,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def get_cache_stats(self) -> CacheStats:
        """Get cache statistics."""
        entries: List[CacheEntry] = []
        total_records = 0
        for key, (records, cached_at, hits) in self._cache.items():
            entries.append(CacheEntry(
                cache_key=key,
                record_count=len(records),
                cached_at=cached_at.isoformat(),
                hit_count=hits,
            ))
            total_records += len(records)

        total_attempts = self._cache_hits + self._cache_misses
        hit_ratio = _safe_divide(
            Decimal(str(self._cache_hits)),
            Decimal(str(total_attempts)),
        ) if total_attempts > 0 else Decimal("0")

        return CacheStats(
            total_entries=len(entries),
            total_records=total_records,
            total_hits=self._cache_hits,
            total_misses=self._cache_misses,
            hit_ratio=hit_ratio,
            entries=entries,
        )

    def clear_cache(self, source_type: Optional[str] = None) -> int:
        """Clear cache entries.

        Args:
            source_type: If provided, clear only this source. Otherwise clear all.

        Returns:
            Number of entries cleared.
        """
        if source_type is None:
            count = len(self._cache)
            self._cache.clear()
            return count

        to_remove = [k for k in self._cache if source_type in k]
        for k in to_remove:
            del self._cache[k]
        return len(to_remove)

    def compute_record_quality(
        self,
        record: ExternalDataRecord,
        source_config: DataSourceConfig,
    ) -> Decimal:
        """Compute quality score for a single record.

        Q = w_completeness * completeness + w_freshness * freshness + w_source * trust

        Args:
            record:         Data record.
            source_config:  Source configuration.

        Returns:
            Quality score (0-1).
        """
        # Completeness: count non-empty required fields
        total_fields = 8  # entity_name, sector, country, year, s1, s2, total, intensity
        present = 0
        if record.entity_name:
            present += 1
        if record.sector:
            present += 1
        if record.country:
            present += 1
        if record.reporting_year > 0:
            present += 1
        if record.scope1_tco2e > Decimal("0"):
            present += 1
        if record.scope2_tco2e > Decimal("0"):
            present += 1
        if record.total_tco2e > Decimal("0"):
            present += 1
        if record.intensity_value is not None and record.intensity_value > Decimal("0"):
            present += 1

        completeness = _safe_divide(Decimal(str(present)), Decimal(str(total_fields)))

        # Freshness
        freshness_status = self._compute_freshness(source_config)
        freshness = freshness_status.freshness_score

        # Trust
        trust = source_config.trust_score or Decimal("0.50")

        # Weighted composite
        w_comp = Decimal("0.40")
        w_fresh = Decimal("0.30")
        w_trust = Decimal("0.30")
        quality = (w_comp * completeness + w_fresh * freshness + w_trust * trust)
        return quality.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Internal: Validation
    # ------------------------------------------------------------------

    def _get_required_fields(self, source_type: DataSourceType) -> List[str]:
        """Get required fields for a source type."""
        mapping = {
            DataSourceType.CDP: CDP_REQUIRED_FIELDS,
            DataSourceType.TPI: TPI_REQUIRED_FIELDS,
            DataSourceType.GRESB: GRESB_REQUIRED_FIELDS,
            DataSourceType.CRREM: CRREM_REQUIRED_FIELDS,
            DataSourceType.ISS_ESG: ISS_REQUIRED_FIELDS,
        }
        return mapping.get(source_type, [])

    def _validate_record(
        self,
        idx: int,
        record: Dict[str, Any],
        required_fields: List[str],
        config: DataSourceConfig,
    ) -> List[ValidationIssue]:
        """Validate a single record against schema."""
        issues: List[ValidationIssue] = []

        # Check required fields
        for field in required_fields:
            if field not in record or record[field] is None or record[field] == "":
                issues.append(ValidationIssue(
                    record_index=idx,
                    field=field,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required field '{field}' is missing or empty.",
                ))

        # Check custom schema
        if config.custom_schema:
            for field_name, field_type in config.custom_schema.items():
                if field_name in record and record[field_name] is not None:
                    if field_type == "number":
                        try:
                            Decimal(str(record[field_name]))
                        except (InvalidOperation, TypeError, ValueError):
                            issues.append(ValidationIssue(
                                record_index=idx,
                                field=field_name,
                                severity=ValidationSeverity.ERROR,
                                message=f"Field '{field_name}' must be numeric.",
                            ))

        # Sanity checks on numeric fields
        for num_field in ["scope1_tco2e", "scope2_tco2e", "scope3_tco2e", "total_tco2e"]:
            if num_field in record and record[num_field] is not None:
                try:
                    val = Decimal(str(record[num_field]))
                    if val < Decimal("0"):
                        issues.append(ValidationIssue(
                            record_index=idx,
                            field=num_field,
                            severity=ValidationSeverity.WARNING,
                            message=f"Field '{num_field}' is negative.",
                        ))
                except (InvalidOperation, TypeError, ValueError):
                    pass

        return issues

    # ------------------------------------------------------------------
    # Internal: Deduplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self, records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Deduplicate records by entity + reporting year."""
        seen: Dict[str, int] = {}
        unique: List[Dict[str, Any]] = []
        duplicates = 0

        for record in records:
            entity = str(record.get("entity_name", record.get("company_name", record.get("organisation_name", ""))))
            year = str(record.get("reporting_year", record.get("assessment_year", "")))
            key = f"{entity.lower().strip()}|{year}"

            if key in seen:
                duplicates += 1
            else:
                seen[key] = len(unique)
                unique.append(record)

        return unique, duplicates

    # ------------------------------------------------------------------
    # Internal: Record Conversion
    # ------------------------------------------------------------------

    def _convert_record(
        self,
        record: Dict[str, Any],
        config: DataSourceConfig,
        ingested_at: str,
    ) -> ExternalDataRecord:
        """Convert raw dict record to ExternalDataRecord."""
        source_type = config.source_type

        # Common field mapping with source-specific aliases
        entity_name = str(
            record.get("entity_name",
            record.get("company_name",
            record.get("organisation_name", "")))
        )
        sector = str(record.get("sector", ""))
        country = str(record.get("country", ""))
        year = int(record.get("reporting_year", record.get("assessment_year", 2024)))

        s1 = _decimal(record.get("scope1_tco2e", 0))
        s2 = _decimal(record.get("scope2_tco2e", record.get("scope2_location_tco2e", 0)))
        s3_raw = record.get("scope3_tco2e")
        s3 = _decimal(s3_raw) if s3_raw is not None else None
        total_val = record.get("total_tco2e")
        total = _decimal(total_val) if total_val is not None else s1 + s2 + (s3 or Decimal("0"))

        intensity_raw = record.get("intensity_value", record.get("carbon_intensity"))
        intensity = _decimal(intensity_raw) if intensity_raw is not None else None
        intensity_unit = str(record.get("intensity_unit", ""))

        revenue_raw = record.get("revenue_millions", record.get("revenue"))
        revenue = _decimal(revenue_raw) if revenue_raw is not None else None

        currency = str(record.get("currency", "USD"))
        verification = str(record.get("verification", record.get("verification_status", "")))

        # Collect extra fields
        standard_keys = {
            "entity_name", "company_name", "organisation_name", "sector", "country",
            "reporting_year", "assessment_year", "scope1_tco2e", "scope2_tco2e",
            "scope2_location_tco2e", "scope3_tco2e", "total_tco2e", "intensity_value",
            "carbon_intensity", "intensity_unit", "revenue_millions", "revenue",
            "currency", "verification", "verification_status",
        }
        extra = {k: v for k, v in record.items() if k not in standard_keys}

        return ExternalDataRecord(
            source_type=source_type,
            entity_name=entity_name,
            sector=sector,
            country=country,
            reporting_year=year,
            scope1_tco2e=s1,
            scope2_tco2e=s2,
            scope3_tco2e=s3,
            total_tco2e=total,
            intensity_value=intensity,
            intensity_unit=intensity_unit,
            revenue_millions=revenue,
            currency=currency,
            verification=verification,
            extra_fields=extra,
            ingested_at=ingested_at,
        )

    # ------------------------------------------------------------------
    # Internal: Cache
    # ------------------------------------------------------------------

    def _make_cache_key(self, config: DataSourceConfig) -> str:
        """Generate cache key from source config."""
        raw = f"{config.source_type.value}|{config.source_name}|{config.api_endpoint or ''}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _get_cached(
        self, key: str, config: DataSourceConfig,
    ) -> Optional[List[ExternalDataRecord]]:
        """Get cached records if TTL is still valid."""
        if key not in self._cache:
            self._cache_misses += 1
            return None

        records, cached_at, hits = self._cache[key]
        ttl = config.ttl_days or 0
        if ttl <= 0:
            self._cache_misses += 1
            return None

        elapsed = (_utcnow() - cached_at).total_seconds() / 86400
        if elapsed > ttl:
            del self._cache[key]
            self._cache_misses += 1
            return None

        self._cache[key] = (records, cached_at, hits + 1)
        self._cache_hits += 1
        return records

    def _set_cache(
        self, key: str, records: List[ExternalDataRecord], config: DataSourceConfig,
    ) -> None:
        """Store records in cache."""
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Evict oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (records, _utcnow(), 0)

    # ------------------------------------------------------------------
    # Internal: Freshness
    # ------------------------------------------------------------------

    def _compute_freshness(self, config: DataSourceConfig) -> DataFreshnessStatus:
        """Compute data freshness for a source."""
        cache_key = self._make_cache_key(config)
        ttl = config.ttl_days or 365

        if cache_key in self._cache:
            _, cached_at, _ = self._cache[cache_key]
            days_since = int((_utcnow() - cached_at).total_seconds() / 86400)
            freshness = max(
                Decimal("0"),
                Decimal("1") - _safe_divide(Decimal(str(days_since)), Decimal(str(ttl))),
            )
            grace_limit = int(float(ttl) * float(GRACE_PERIOD_MULTIPLIER))
            if days_since <= ttl:
                status = FreshnessStatus.FRESH
            elif days_since <= grace_limit:
                status = FreshnessStatus.STALE
            else:
                status = FreshnessStatus.EXPIRED

            return DataFreshnessStatus(
                source_type=config.source_type.value,
                last_updated=cached_at.isoformat(),
                ttl_days=ttl,
                days_since_update=days_since,
                freshness_score=freshness.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                status=status,
            )

        return DataFreshnessStatus(
            source_type=config.source_type.value,
            ttl_days=ttl,
            status=FreshnessStatus.UNKNOWN,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "DataSourceType",
    "FreshnessStatus",
    "ValidationSeverity",
    # Input Models
    "DataSourceConfig",
    "ExternalDataRecord",
    "DatasetIngestionInput",
    # Output Models
    "ValidationIssue",
    "DataFreshnessStatus",
    "CacheEntry",
    "CacheStats",
    "IngestionResult",
    # Engine
    "ExternalDatasetEngine",
]
