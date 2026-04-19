# -*- coding: utf-8 -*-
"""
EUDRBridge - EUDR Deforestation Due Diligence Bridge for PACK-020
====================================================================

Connects EUDR (EU Deforestation Regulation) due diligence data to Battery
Regulation supply chain requirements. Specifically targets natural rubber
sourcing for battery seals, gaskets, and EV tyre components. Maps EUDR
compliance statuses, commodity risk assessments, and geolocation data to
Battery Regulation Art 39 supply chain due diligence.

Natural rubber is an EUDR-regulated commodity (Annex I) commonly used in:
    - Battery pack seals and gaskets
    - Thermal interface materials
    - Vibration dampening components
    - EV tyres (for complete vehicle battery passport scope)

Legal References:
    - Regulation (EU) 2023/1115 (EUDR) - Deforestation-free supply chains
    - Regulation (EU) 2023/1542, Art 39 (Supply chain due diligence)
    - Art 39(2)(d): Identifying environmental risks in the supply chain
    - EUDR Art 3-9: Due diligence system requirements

EUDR-Battery Regulation Overlap:
    - EUDR rubber DD satisfies Battery Reg Art 39 for rubber components
    - EUDR geolocation data supplements passport supply chain traceability
    - EUDR risk assessment feeds into Art 40 risk management

Relevant Commodities (EUDR Annex I applicable to batteries):
    - Rubber (primary: seals, gaskets, tyre components)
    - Wood (secondary: packaging, pallets)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-020 Battery Passport Prep Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EUDRCommodity(str, Enum):
    """EUDR Annex I regulated commodities relevant to batteries."""

    RUBBER = "rubber"
    WOOD = "wood"

class DeforestationStatus(str, Enum):
    """EUDR deforestation compliance status."""

    DEFORESTATION_FREE = "deforestation_free"
    RISK_IDENTIFIED = "risk_identified"
    NON_COMPLIANT = "non_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"

class CountryBenchmark(str, Enum):
    """EUDR country benchmarking risk levels (Art 29)."""

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    NOT_CLASSIFIED = "not_classified"

class DDSystemStatus(str, Enum):
    """Due diligence system maturity status."""

    OPERATIONAL = "operational"
    DEVELOPING = "developing"
    PLANNED = "planned"
    NOT_STARTED = "not_started"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class EUDRBridgeConfig(BaseModel):
    """Configuration for the EUDR Bridge."""

    pack_id: str = Field(default="PACK-020")
    reporting_year: int = Field(default=2025, ge=2020, le=2030)
    enable_provenance: bool = Field(default=True)
    eudr_cutoff_date: str = Field(
        default="2020-12-31",
        description="EUDR deforestation cutoff date (31 Dec 2020)",
    )
    commodities_in_scope: List[EUDRCommodity] = Field(
        default_factory=lambda: [EUDRCommodity.RUBBER, EUDRCommodity.WOOD]
    )

class DeforestationAssessment(BaseModel):
    """Deforestation risk assessment for a supply chain node."""

    assessment_id: str = Field(default_factory=_new_uuid)
    commodity: EUDRCommodity = Field(default=EUDRCommodity.RUBBER)
    supplier_name: str = Field(default="")
    country_of_origin: str = Field(default="")
    country_benchmark: CountryBenchmark = Field(default=CountryBenchmark.NOT_CLASSIFIED)
    geolocation_available: bool = Field(default=False)
    geolocation_coordinates: Optional[str] = Field(None)
    deforestation_status: DeforestationStatus = Field(
        default=DeforestationStatus.NOT_ASSESSED
    )
    satellite_verification: bool = Field(default=False)
    legality_verified: bool = Field(default=False)
    assessment_date: Optional[str] = Field(None)

class DeforestationStatusResult(BaseModel):
    """Result of deforestation status check."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    assessments: List[DeforestationAssessment] = Field(default_factory=list)
    total_assessed: int = Field(default=0)
    deforestation_free_count: int = Field(default=0)
    risk_identified_count: int = Field(default=0)
    non_compliant_count: int = Field(default=0)
    overall_status: DeforestationStatus = Field(
        default=DeforestationStatus.NOT_ASSESSED
    )
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class CommodityRiskResult(BaseModel):
    """Result of commodity risk mapping."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    commodity_risks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    high_risk_countries: List[str] = Field(default_factory=list)
    standard_risk_countries: List[str] = Field(default_factory=list)
    low_risk_countries: List[str] = Field(default_factory=list)
    battery_reg_articles_satisfied: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class RubberSourcingResult(BaseModel):
    """Result of rubber sourcing validation."""

    operation_id: str = Field(default_factory=_new_uuid)
    status: str = Field(default="pending")
    rubber_suppliers_total: int = Field(default=0)
    rubber_suppliers_compliant: int = Field(default=0)
    rubber_suppliers_non_compliant: int = Field(default=0)
    compliance_rate_pct: float = Field(default=0.0)
    countries_of_origin: List[str] = Field(default_factory=list)
    certifications_present: List[str] = Field(default_factory=list)
    battery_components_using_rubber: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Country Benchmark Data (EUDR Art 29)
# ---------------------------------------------------------------------------

COUNTRY_BENCHMARKS: Dict[str, CountryBenchmark] = {
    # Southeast Asia (rubber producing)
    "THA": CountryBenchmark.STANDARD,
    "IDN": CountryBenchmark.HIGH,
    "MYS": CountryBenchmark.HIGH,
    "VNM": CountryBenchmark.STANDARD,
    "KHM": CountryBenchmark.HIGH,
    "LAO": CountryBenchmark.HIGH,
    "MMR": CountryBenchmark.HIGH,
    # West Africa
    "CIV": CountryBenchmark.HIGH,
    "GHA": CountryBenchmark.HIGH,
    "CMR": CountryBenchmark.HIGH,
    "NGA": CountryBenchmark.HIGH,
    "LBR": CountryBenchmark.HIGH,
    # South America
    "BRA": CountryBenchmark.HIGH,
    "GTM": CountryBenchmark.STANDARD,
    # Low risk
    "FRA": CountryBenchmark.LOW,
    "DEU": CountryBenchmark.LOW,
    "JPN": CountryBenchmark.LOW,
    "USA": CountryBenchmark.LOW,
    "KOR": CountryBenchmark.LOW,
    "IND": CountryBenchmark.STANDARD,
    "LKA": CountryBenchmark.STANDARD,
    "CHN": CountryBenchmark.STANDARD,
}

BATTERY_RUBBER_COMPONENTS: List[str] = [
    "battery_pack_seals",
    "cell_gaskets",
    "thermal_interface_pads",
    "vibration_dampeners",
    "cable_insulation",
    "coolant_hoses",
    "mounting_grommets",
]

# ---------------------------------------------------------------------------
# EUDRBridge
# ---------------------------------------------------------------------------

class EUDRBridge:
    """EUDR deforestation due diligence bridge for PACK-020.

    Connects EUDR compliance data for rubber and wood commodities to
    Battery Regulation Art 39 supply chain due diligence. Maps
    deforestation risk, commodity benchmarks, and geolocation data
    to battery passport requirements.

    Attributes:
        config: Bridge configuration.
        _assessments: Cached deforestation assessments.

    Example:
        >>> bridge = EUDRBridge(EUDRBridgeConfig())
        >>> result = bridge.get_deforestation_status(context)
        >>> assert result.status == "completed"
    """

    def __init__(self, config: Optional[EUDRBridgeConfig] = None) -> None:
        """Initialize EUDRBridge."""
        self.config = config or EUDRBridgeConfig()
        self._assessments: List[DeforestationAssessment] = []
        logger.info(
            "EUDRBridge initialized (commodities=%s, cutoff=%s)",
            [c.value for c in self.config.commodities_in_scope],
            self.config.eudr_cutoff_date,
        )

    def get_deforestation_status(
        self, context: Dict[str, Any]
    ) -> DeforestationStatusResult:
        """Get deforestation compliance status for battery supply chain.

        Args:
            context: Pipeline context with EUDR assessment data.

        Returns:
            DeforestationStatusResult with assessment outcomes.
        """
        result = DeforestationStatusResult(started_at=utcnow())

        try:
            raw_assessments = context.get("eudr_assessments", [])
            parsed: List[DeforestationAssessment] = []

            for a in raw_assessments:
                country = a.get("country_of_origin", "")
                benchmark = COUNTRY_BENCHMARKS.get(
                    country, CountryBenchmark.NOT_CLASSIFIED
                )
                assessment = DeforestationAssessment(
                    commodity=EUDRCommodity(a.get("commodity", "rubber")),
                    supplier_name=a.get("supplier_name", ""),
                    country_of_origin=country,
                    country_benchmark=benchmark,
                    geolocation_available=a.get("geolocation_available", False),
                    geolocation_coordinates=a.get("geolocation_coordinates"),
                    deforestation_status=DeforestationStatus(
                        a.get("deforestation_status", "not_assessed")
                    ),
                    satellite_verification=a.get("satellite_verification", False),
                    legality_verified=a.get("legality_verified", False),
                    assessment_date=a.get("assessment_date"),
                )
                parsed.append(assessment)

            self._assessments = parsed
            result.assessments = parsed
            result.total_assessed = len(parsed)
            result.deforestation_free_count = sum(
                1 for a in parsed
                if a.deforestation_status == DeforestationStatus.DEFORESTATION_FREE
            )
            result.risk_identified_count = sum(
                1 for a in parsed
                if a.deforestation_status == DeforestationStatus.RISK_IDENTIFIED
            )
            result.non_compliant_count = sum(
                1 for a in parsed
                if a.deforestation_status == DeforestationStatus.NON_COMPLIANT
            )

            result.overall_status = self._determine_overall_status(result)
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "total": result.total_assessed,
                    "free": result.deforestation_free_count,
                    "non_compliant": result.non_compliant_count,
                })

            logger.info(
                "Deforestation status: %d assessed, %d free, %d risk, %d non-compliant",
                result.total_assessed,
                result.deforestation_free_count,
                result.risk_identified_count,
                result.non_compliant_count,
            )

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.error("Deforestation status check failed: %s", str(exc))

        result.completed_at = utcnow()
        if result.started_at:
            result.duration_ms = (
                result.completed_at - result.started_at
            ).total_seconds() * 1000
        return result

    def map_commodity_risks(
        self, context: Dict[str, Any]
    ) -> CommodityRiskResult:
        """Map commodity-specific deforestation risks for battery supply chain.

        Args:
            context: Pipeline context with commodity sourcing data.

        Returns:
            CommodityRiskResult with risk mapping by commodity and country.
        """
        result = CommodityRiskResult()

        try:
            if not self._assessments:
                self.get_deforestation_status(context)

            for commodity in self.config.commodities_in_scope:
                commodity_assessments = [
                    a for a in self._assessments if a.commodity == commodity
                ]
                countries = list({
                    a.country_of_origin for a in commodity_assessments
                    if a.country_of_origin
                })

                result.commodity_risks[commodity.value] = {
                    "assessments": len(commodity_assessments),
                    "deforestation_free": sum(
                        1 for a in commodity_assessments
                        if a.deforestation_status == DeforestationStatus.DEFORESTATION_FREE
                    ),
                    "risk_identified": sum(
                        1 for a in commodity_assessments
                        if a.deforestation_status == DeforestationStatus.RISK_IDENTIFIED
                    ),
                    "countries": countries,
                    "high_risk_countries": [
                        c for c in countries
                        if COUNTRY_BENCHMARKS.get(c) == CountryBenchmark.HIGH
                    ],
                }

            all_countries = list({
                a.country_of_origin for a in self._assessments
                if a.country_of_origin
            })
            result.high_risk_countries = [
                c for c in all_countries
                if COUNTRY_BENCHMARKS.get(c) == CountryBenchmark.HIGH
            ]
            result.standard_risk_countries = [
                c for c in all_countries
                if COUNTRY_BENCHMARKS.get(c) == CountryBenchmark.STANDARD
            ]
            result.low_risk_countries = [
                c for c in all_countries
                if COUNTRY_BENCHMARKS.get(c) == CountryBenchmark.LOW
            ]

            result.battery_reg_articles_satisfied = ["Art 39"]
            if all(
                a.deforestation_status == DeforestationStatus.DEFORESTATION_FREE
                for a in self._assessments
            ) and self._assessments:
                result.battery_reg_articles_satisfied.append("Art 39(2)(d)")

            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(result.commodity_risks)

            logger.info(
                "Commodity risks mapped: %d commodities, %d high-risk countries",
                len(result.commodity_risks),
                len(result.high_risk_countries),
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Commodity risk mapping failed: %s", str(exc))

        return result

    def validate_rubber_sourcing(
        self, context: Dict[str, Any]
    ) -> RubberSourcingResult:
        """Validate rubber sourcing for battery components against EUDR.

        Args:
            context: Pipeline context with rubber sourcing data.

        Returns:
            RubberSourcingResult with compliance rates and component mapping.
        """
        result = RubberSourcingResult()

        try:
            if not self._assessments:
                self.get_deforestation_status(context)

            rubber_assessments = [
                a for a in self._assessments
                if a.commodity == EUDRCommodity.RUBBER
            ]

            result.rubber_suppliers_total = len(rubber_assessments)
            result.rubber_suppliers_compliant = sum(
                1 for a in rubber_assessments
                if a.deforestation_status == DeforestationStatus.DEFORESTATION_FREE
            )
            result.rubber_suppliers_non_compliant = sum(
                1 for a in rubber_assessments
                if a.deforestation_status == DeforestationStatus.NON_COMPLIANT
            )

            if result.rubber_suppliers_total > 0:
                result.compliance_rate_pct = round(
                    result.rubber_suppliers_compliant
                    / result.rubber_suppliers_total
                    * 100,
                    1,
                )
            else:
                result.compliance_rate_pct = 0.0

            result.countries_of_origin = list({
                a.country_of_origin for a in rubber_assessments
                if a.country_of_origin
            })
            result.certifications_present = list({
                "FSC" for a in rubber_assessments
                if a.legality_verified
            })
            result.battery_components_using_rubber = list(
                BATTERY_RUBBER_COMPONENTS
            )
            result.status = "completed"

            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash({
                    "total": result.rubber_suppliers_total,
                    "compliant": result.rubber_suppliers_compliant,
                    "rate": result.compliance_rate_pct,
                })

            logger.info(
                "Rubber sourcing: %d suppliers, %.1f%% compliant",
                result.rubber_suppliers_total,
                result.compliance_rate_pct,
            )

        except Exception as exc:
            result.status = "failed"
            logger.error("Rubber sourcing validation failed: %s", str(exc))

        return result

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status.

        Returns:
            Dict with bridge status information.
        """
        return {
            "pack_id": self.config.pack_id,
            "reporting_year": self.config.reporting_year,
            "commodities_in_scope": [c.value for c in self.config.commodities_in_scope],
            "eudr_cutoff_date": self.config.eudr_cutoff_date,
            "assessments_loaded": len(self._assessments),
            "countries_benchmarked": len(COUNTRY_BENCHMARKS),
            "battery_rubber_components": len(BATTERY_RUBBER_COMPONENTS),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _determine_overall_status(
        result: DeforestationStatusResult,
    ) -> DeforestationStatus:
        """Determine overall deforestation status from individual assessments."""
        if result.total_assessed == 0:
            return DeforestationStatus.NOT_ASSESSED
        if result.non_compliant_count > 0:
            return DeforestationStatus.NON_COMPLIANT
        if result.risk_identified_count > 0:
            return DeforestationStatus.RISK_IDENTIFIED
        if result.deforestation_free_count == result.total_assessed:
            return DeforestationStatus.DEFORESTATION_FREE
        return DeforestationStatus.UNDER_REVIEW
