# -*- coding: utf-8 -*-
"""
PortfolioDataBridge - Portfolio Holdings, NAV, and Sector Data Ingestion
=========================================================================

This module ingests portfolio holdings, NAV data, sector/geographic
classification, and benchmark composition for SFDR Article 8 disclosures.
It validates portfolio completeness and maps sector codes to NACE/GICS
classification systems.

Architecture:
    Portfolio System --> PortfolioDataBridge --> Validated Holdings
                              |
                              v
    NAV History + Sector Allocation + Geographic Allocation

Example:
    >>> config = PortfolioDataBridgeConfig()
    >>> bridge = PortfolioDataBridge(config)
    >>> holdings = bridge.import_holdings(raw_data)
    >>> sector = bridge.get_sector_allocation(holdings)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning("AgentStub: failed to load %s: %s", self.agent_id, exc)
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class SectorClassification(str, Enum):
    """Sector classification system."""
    NACE = "nace"
    GICS = "gics"
    ICB = "icb"


class DataFormat(str, Enum):
    """Portfolio data format."""
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"
    API = "api"


class DataCategory(str, Enum):
    """Portfolio data category."""
    HOLDINGS = "holdings"
    NAV_HISTORY = "nav_history"
    SECTOR_CLASSIFICATION = "sector_classification"
    GEOGRAPHIC_ALLOCATION = "geographic_allocation"
    BENCHMARK_COMPOSITION = "benchmark_composition"


# =============================================================================
# Data Models
# =============================================================================


class PortfolioDataBridgeConfig(BaseModel):
    """Configuration for the Portfolio Data Bridge."""
    data_format: DataFormat = Field(
        default=DataFormat.JSON, description="Input data format"
    )
    nav_source: str = Field(
        default="manual", description="NAV data source"
    )
    sector_classification: SectorClassification = Field(
        default=SectorClassification.NACE,
        description="Sector classification system",
    )
    reporting_currency: str = Field(
        default="EUR", description="Reporting currency"
    )
    validate_isins: bool = Field(
        default=True, description="Validate ISIN codes"
    )
    validate_weights: bool = Field(
        default=True, description="Validate weight sum = 100%"
    )
    weight_tolerance_pct: float = Field(
        default=5.0, ge=0.0, le=20.0,
        description="Acceptable deviation from 100% weight",
    )
    max_holdings: int = Field(
        default=5000, ge=1, description="Maximum number of holdings"
    )


HOLDING_FIELDS: List[str] = [
    "isin", "name", "ticker", "sedol", "cusip",
    "weight", "market_value", "notional_value", "quantity", "price",
    "sector", "sector_code", "industry", "sub_industry",
    "country", "country_code", "region", "currency",
    "asset_class", "instrument_type", "maturity_date",
    "esg_rating", "esg_score", "carbon_intensity",
    "taxonomy_eligible", "taxonomy_aligned",
]

DATA_CATEGORIES: Dict[str, str] = {
    "holdings": "Portfolio holdings with position-level detail",
    "nav_history": "Historical NAV data for the fund",
    "sector_classification": "Sector/industry classification mapping",
    "geographic_allocation": "Geographic allocation by country/region",
    "benchmark_composition": "Reference benchmark holdings and weights",
}


class ValidatedHolding(BaseModel):
    """A validated portfolio holding."""
    isin: str = Field(default="", description="ISIN code")
    name: str = Field(default="", description="Security name")
    ticker: str = Field(default="", description="Ticker symbol")
    weight: float = Field(default=0.0, description="Portfolio weight %")
    market_value: float = Field(default=0.0, description="Market value")
    currency: str = Field(default="EUR", description="Currency")
    sector: str = Field(default="", description="Sector name")
    sector_code: str = Field(default="", description="Sector code (NACE/GICS)")
    country: str = Field(default="", description="Country name")
    country_code: str = Field(default="", description="ISO country code")
    asset_class: str = Field(default="equity", description="Asset class")
    esg_rating: str = Field(default="", description="ESG rating")
    taxonomy_eligible: bool = Field(default=False, description="Taxonomy eligible")
    taxonomy_aligned: bool = Field(default=False, description="Taxonomy aligned")
    validation_status: str = Field(default="valid", description="Validation status")
    validation_notes: List[str] = Field(
        default_factory=list, description="Validation notes"
    )


class ImportResult(BaseModel):
    """Result of a holdings import."""
    total_records: int = Field(default=0, description="Total records received")
    valid_records: int = Field(default=0, description="Valid records accepted")
    invalid_records: int = Field(default=0, description="Invalid records rejected")
    total_weight_pct: float = Field(default=0.0, description="Sum of weights")
    total_market_value: float = Field(default=0.0, description="Total market value")
    currency: str = Field(default="EUR", description="Currency")
    holdings: List[ValidatedHolding] = Field(
        default_factory=list, description="Validated holdings"
    )
    sector_count: int = Field(default=0, description="Unique sectors")
    country_count: int = Field(default=0, description="Unique countries")
    errors: List[str] = Field(default_factory=list, description="Import errors")
    warnings: List[str] = Field(default_factory=list, description="Import warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    imported_at: str = Field(default="", description="Import timestamp")


class NAVEntry(BaseModel):
    """A single NAV observation."""
    date: str = Field(default="", description="Observation date")
    nav: float = Field(default=0.0, description="Net Asset Value")
    nav_per_share: float = Field(default=0.0, description="NAV per share")
    shares_outstanding: float = Field(default=0.0, description="Shares outstanding")
    currency: str = Field(default="EUR", description="Currency")


# =============================================================================
# Portfolio Data Bridge
# =============================================================================


class PortfolioDataBridge:
    """Bridge for ingesting portfolio data into SFDR Article 8 pipeline.

    Imports portfolio holdings, NAV history, sector/geographic classification,
    and benchmark composition. Validates ISIN codes, weight sums, and required
    fields for downstream SFDR processing.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs for data connectors.

    Example:
        >>> bridge = PortfolioDataBridge(PortfolioDataBridgeConfig())
        >>> result = bridge.import_holdings(raw_data)
        >>> print(f"Imported {result.valid_records} holdings")
    """

    def __init__(self, config: Optional[PortfolioDataBridgeConfig] = None) -> None:
        """Initialize the Portfolio Data Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or PortfolioDataBridgeConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "excel_normalizer": _AgentStub(
                "GL-DATA-X-002",
                "greenlang.agents.data.excel_csv_normalizer",
                "ExcelCSVNormalizer",
            ),
            "erp_connector": _AgentStub(
                "GL-DATA-X-003",
                "greenlang.agents.data.erp_finance_connector",
                "ERPFinanceConnector",
            ),
        }

        self.logger.info(
            "PortfolioDataBridge initialized: format=%s, sector=%s, currency=%s",
            self.config.data_format.value,
            self.config.sector_classification.value,
            self.config.reporting_currency,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def import_holdings(
        self,
        raw_data: List[Dict[str, Any]],
    ) -> ImportResult:
        """Import and validate portfolio holdings.

        Processes raw holding records, validates ISINs and weights,
        normalizes sector codes, and produces validated holdings.

        Args:
            raw_data: List of raw holding dictionaries.

        Returns:
            ImportResult with validated holdings and statistics.
        """
        errors: List[str] = []
        warnings: List[str] = []
        validated: List[ValidatedHolding] = []
        total_weight = 0.0
        total_mv = 0.0

        if len(raw_data) > self.config.max_holdings:
            errors.append(
                f"Exceeds max holdings: {len(raw_data)} > {self.config.max_holdings}"
            )
            raw_data = raw_data[:self.config.max_holdings]

        for idx, record in enumerate(raw_data):
            holding, hold_errors, hold_warnings = self._validate_holding(
                record, idx
            )
            errors.extend(hold_errors)
            warnings.extend(hold_warnings)

            if holding is not None:
                validated.append(holding)
                total_weight += holding.weight
                total_mv += holding.market_value

        # Weight validation
        if self.config.validate_weights and validated:
            if abs(total_weight - 100.0) > self.config.weight_tolerance_pct:
                warnings.append(
                    f"Portfolio weights sum to {total_weight:.2f}%, "
                    f"expected 100% +/- {self.config.weight_tolerance_pct}%"
                )

        sectors = set(h.sector for h in validated if h.sector)
        countries = set(h.country_code for h in validated if h.country_code)

        result = ImportResult(
            total_records=len(raw_data),
            valid_records=len(validated),
            invalid_records=len(raw_data) - len(validated),
            total_weight_pct=round(total_weight, 4),
            total_market_value=round(total_mv, 2),
            currency=self.config.reporting_currency,
            holdings=validated,
            sector_count=len(sectors),
            country_count=len(countries),
            errors=errors,
            warnings=warnings,
            imported_at=_utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data({
            "total": result.total_records,
            "valid": result.valid_records,
            "weight": result.total_weight_pct,
            "mv": result.total_market_value,
        })

        self.logger.info(
            "Holdings imported: %d/%d valid, weight=%.2f%%, mv=%.2f %s",
            result.valid_records, result.total_records,
            total_weight, total_mv, self.config.reporting_currency,
        )
        return result

    def get_sector_allocation(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate sector allocation from holdings.

        Args:
            holdings: List of validated holdings.

        Returns:
            Dictionary keyed by sector with weight and count.
        """
        allocation: Dict[str, Dict[str, Any]] = {}

        for h in holdings:
            sector = h.get("sector", "Other")
            if not sector:
                sector = "Other"

            if sector not in allocation:
                allocation[sector] = {"weight_pct": 0.0, "count": 0, "holdings": []}

            allocation[sector]["weight_pct"] += float(h.get("weight", 0.0))
            allocation[sector]["count"] += 1
            allocation[sector]["holdings"].append(h.get("isin", ""))

        # Round weights
        for sector in allocation:
            allocation[sector]["weight_pct"] = round(
                allocation[sector]["weight_pct"], 2
            )

        return dict(sorted(
            allocation.items(),
            key=lambda x: x[1]["weight_pct"],
            reverse=True,
        ))

    def get_geographic_allocation(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate geographic allocation from holdings.

        Args:
            holdings: List of validated holdings.

        Returns:
            Dictionary keyed by country with weight and count.
        """
        allocation: Dict[str, Dict[str, Any]] = {}

        for h in holdings:
            country = h.get("country_code", h.get("country", "Other"))
            if not country:
                country = "Other"

            if country not in allocation:
                allocation[country] = {"weight_pct": 0.0, "count": 0}

            allocation[country]["weight_pct"] += float(h.get("weight", 0.0))
            allocation[country]["count"] += 1

        for country in allocation:
            allocation[country]["weight_pct"] = round(
                allocation[country]["weight_pct"], 2
            )

        return dict(sorted(
            allocation.items(),
            key=lambda x: x[1]["weight_pct"],
            reverse=True,
        ))

    def get_nav_history(
        self,
        nav_data: List[Dict[str, Any]],
    ) -> List[NAVEntry]:
        """Parse and validate NAV history data.

        Args:
            nav_data: Raw NAV data records.

        Returns:
            List of validated NAVEntry objects.
        """
        entries: List[NAVEntry] = []

        for record in nav_data:
            try:
                entry = NAVEntry(
                    date=str(record.get("date", "")),
                    nav=float(record.get("nav", 0.0)),
                    nav_per_share=float(record.get("nav_per_share", 0.0)),
                    shares_outstanding=float(
                        record.get("shares_outstanding", 0.0)
                    ),
                    currency=str(
                        record.get("currency", self.config.reporting_currency)
                    ),
                )
                entries.append(entry)
            except (ValueError, TypeError) as exc:
                self.logger.warning("Invalid NAV record: %s", exc)

        self.logger.info("NAV history parsed: %d entries", len(entries))
        return entries

    def validate_portfolio(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate portfolio completeness for SFDR disclosures.

        Checks that all required data fields are populated and that
        coverage thresholds are met for PAI and taxonomy calculations.

        Args:
            holdings: List of validated holdings.

        Returns:
            Validation report with pass/fail per check.
        """
        checks: List[Dict[str, Any]] = []
        total = len(holdings)

        # Check 1: ISIN coverage
        isin_count = sum(1 for h in holdings if h.get("isin"))
        checks.append({
            "check": "isin_coverage",
            "status": "pass" if isin_count == total else "warning",
            "detail": f"{isin_count}/{total} holdings have ISIN",
        })

        # Check 2: Sector coverage
        sector_count = sum(1 for h in holdings if h.get("sector"))
        checks.append({
            "check": "sector_coverage",
            "status": "pass" if sector_count >= total * 0.8 else "warning",
            "detail": f"{sector_count}/{total} holdings have sector",
        })

        # Check 3: ESG rating coverage
        esg_count = sum(1 for h in holdings if h.get("esg_rating"))
        checks.append({
            "check": "esg_rating_coverage",
            "status": "pass" if esg_count >= total * 0.5 else "warning",
            "detail": f"{esg_count}/{total} holdings have ESG rating",
        })

        # Check 4: Weight sum
        total_weight = sum(float(h.get("weight", 0.0)) for h in holdings)
        weight_ok = abs(total_weight - 100.0) <= self.config.weight_tolerance_pct
        checks.append({
            "check": "weight_sum",
            "status": "pass" if weight_ok else "fail",
            "detail": f"Weight sum: {total_weight:.2f}%",
        })

        # Check 5: Country coverage
        country_count = sum(1 for h in holdings if h.get("country_code"))
        checks.append({
            "check": "country_coverage",
            "status": "pass" if country_count >= total * 0.7 else "warning",
            "detail": f"{country_count}/{total} holdings have country",
        })

        passed = sum(1 for c in checks if c["status"] == "pass")
        score = round((passed / max(len(checks), 1)) * 100, 1)

        return {
            "checks": checks,
            "passed": passed,
            "total": len(checks),
            "score": score,
            "overall": "pass" if score >= 60 else "fail",
            "provenance_hash": _hash_data({
                "holdings": total, "checks": len(checks), "score": score,
            }),
        }

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _validate_holding(
        self,
        record: Dict[str, Any],
        index: int,
    ) -> Tuple[Optional[ValidatedHolding], List[str], List[str]]:
        """Validate a single holding record.

        Args:
            record: Raw holding data.
            index: Record index for error messages.

        Returns:
            Tuple of (validated holding or None, errors, warnings).
        """
        errors: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []

        isin = str(record.get("isin", "")).strip()
        if not isin:
            errors.append(f"Holding {index}: missing ISIN")
            return None, errors, warnings

        if self.config.validate_isins and not self._validate_isin_format(isin):
            warnings.append(f"Holding {index}: ISIN '{isin}' has invalid format")
            notes.append("ISIN format warning")

        name = str(record.get("name", "")).strip()
        weight = float(record.get("weight", 0.0))
        market_value = float(record.get("market_value", 0.0))

        if weight < 0:
            errors.append(f"Holding {index}: negative weight {weight}")
            return None, errors, warnings

        if weight == 0 and market_value == 0:
            warnings.append(f"Holding {index} ({isin}): zero weight and market value")

        # Normalize sector code
        sector = str(record.get("sector", "")).strip()
        sector_code = str(record.get("sector_code", "")).strip()

        # Normalize country
        country = str(record.get("country", "")).strip()
        country_code = str(record.get("country_code", "")).strip()
        if not country_code and country:
            country_code = country[:2].upper()

        holding = ValidatedHolding(
            isin=isin,
            name=name,
            ticker=str(record.get("ticker", "")),
            weight=weight,
            market_value=market_value,
            currency=str(
                record.get("currency", self.config.reporting_currency)
            ),
            sector=sector,
            sector_code=sector_code,
            country=country,
            country_code=country_code,
            asset_class=str(record.get("asset_class", "equity")),
            esg_rating=str(record.get("esg_rating", "")),
            taxonomy_eligible=bool(record.get("taxonomy_eligible", False)),
            taxonomy_aligned=bool(record.get("taxonomy_aligned", False)),
            validation_status="valid" if not notes else "warning",
            validation_notes=notes,
        )

        return holding, errors, warnings

    @staticmethod
    def _validate_isin_format(isin: str) -> bool:
        """Validate ISIN format (2 letters + 9 alphanumeric + 1 check digit).

        Args:
            isin: ISIN code to validate.

        Returns:
            True if format is valid.
        """
        if len(isin) != 12:
            return False
        if not re.match(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$", isin):
            return False
        return True
