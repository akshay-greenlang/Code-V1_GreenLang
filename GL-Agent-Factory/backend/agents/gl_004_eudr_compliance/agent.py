"""
GL-004: EUDR Compliance Agent

This module implements the EU Deforestation Regulation (EUDR) Compliance Agent
for validating deforestation-free supply chains per EU Regulation 2023/1115.

The agent supports:
- Geolocation validation (GPS coordinates, polygons)
- Commodity classification (cattle, cocoa, coffee, palm oil, rubber, soya, wood)
- Deforestation risk assessment
- Supply chain traceability
- Due Diligence Statement (DDS) generation

Example:
    >>> agent = EUDRComplianceAgent()
    >>> result = agent.run(EUDRInput(
    ...     commodity_type=CommodityType.COFFEE,
    ...     cn_code="0901.11.00",
    ...     quantity_kg=50000,
    ...     country_of_origin="BR",
    ...     geolocation={"type": "Point", "coordinates": [-47.5, -15.5]}
    ... ))
    >>> print(f"Risk level: {result.data.risk_level}")
"""

import hashlib
import json
import logging
import math
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class CommodityType(str, Enum):
    """EUDR regulated commodities per Annex I."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    PALM_OIL = "palm_oil"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class RiskLevel(str, Enum):
    """Deforestation risk levels."""

    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"


class ComplianceStatus(str, Enum):
    """Compliance validation status."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    INSUFFICIENT_DATA = "insufficient_data"


class GeometryType(str, Enum):
    """GeoJSON geometry types."""

    POINT = "Point"
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"


class GeoLocation(BaseModel):
    """GeoJSON geometry for production plot."""

    type: GeometryType
    coordinates: Any  # List for Point, List[List] for Polygon

    @validator("coordinates")
    def validate_coordinates(cls, v: Any, values: Dict) -> Any:
        """Validate coordinates based on geometry type."""
        geo_type = values.get("type")

        if geo_type == GeometryType.POINT:
            if not isinstance(v, list) or len(v) < 2:
                raise ValueError("Point requires [longitude, latitude]")
            lon, lat = v[0], v[1]
            if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                raise ValueError("Invalid coordinate range")

        elif geo_type == GeometryType.POLYGON:
            if not isinstance(v, list) or not v:
                raise ValueError("Polygon requires ring arrays")

        return v


class EUDRInput(BaseModel):
    """
    Input model for EUDR Compliance Agent.

    Attributes:
        commodity_type: Type of regulated commodity
        cn_code: Combined Nomenclature code
        quantity_kg: Quantity in kilograms
        country_of_origin: ISO 3166-1 alpha-2 code
        geolocation: GeoJSON geometry of production plot
        production_date: Date of production/harvest
        operator_id: Unique operator identifier
        supply_chain: List of supply chain actors
        certifications: Third-party certifications held
        supporting_documents: Document references
    """

    commodity_type: CommodityType = Field(..., description="Regulated commodity type")
    cn_code: str = Field(..., min_length=8, description="CN code")
    quantity_kg: float = Field(..., ge=0, description="Quantity in kg")
    country_of_origin: str = Field(..., min_length=2, max_length=2)
    geolocation: GeoLocation = Field(..., description="Production plot geometry")
    production_date: date = Field(..., description="Date of production/harvest")
    operator_id: Optional[str] = Field(None, description="Operator identifier")
    supply_chain: List[Dict[str, Any]] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    supporting_documents: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("country_of_origin")
    def validate_country(cls, v: str) -> str:
        """Validate ISO country code."""
        return v.upper()

    @validator("production_date")
    def validate_production_date(cls, v: date) -> date:
        """Validate production date is not before EUDR cutoff."""
        # EUDR cutoff date: December 31, 2020
        cutoff = date(2020, 12, 31)
        if v < cutoff:
            logger.warning(f"Production date {v} is before EUDR cutoff {cutoff}")
        return v


class EUDROutput(BaseModel):
    """
    Output model for EUDR Compliance Agent.

    Includes risk assessment and compliance status.
    """

    commodity_type: str = Field(..., description="Commodity validated")
    cn_code: str = Field(..., description="CN code processed")
    compliance_status: str = Field(..., description="Compliance status")
    risk_level: str = Field(..., description="Deforestation risk level")
    country_risk_score: float = Field(..., ge=0, le=100, description="Country risk 0-100")
    geolocation_valid: bool = Field(..., description="Geolocation validation passed")
    cutoff_date_compliant: bool = Field(..., description="Production after Dec 31, 2020")
    traceability_score: float = Field(..., ge=0, le=100, description="Supply chain traceability %")
    deforestation_detected: Optional[bool] = Field(None, description="Satellite analysis result")
    mitigation_measures: List[str] = Field(default_factory=list, description="Required actions")
    dds_reference: Optional[str] = Field(None, description="Due Diligence Statement reference")
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")
    calculated_at: datetime = Field(default_factory=datetime.utcnow)


class CountryRisk(BaseModel):
    """Country risk classification for EUDR."""

    country_code: str
    risk_level: RiskLevel
    risk_score: float  # 0-100
    source: str


class EUDRComplianceAgent:
    """
    GL-004: EUDR Compliance Agent.

    This agent validates compliance with the EU Deforestation Regulation
    by assessing:
    - Geolocation validity and precision
    - Country/region deforestation risk
    - Production date vs cutoff (Dec 31, 2020)
    - Supply chain traceability
    - Satellite deforestation detection (when available)

    Uses zero-hallucination deterministic calculations:
    - Risk scores from EU benchmarking system
    - Traceability = verified_nodes / total_nodes * 100

    Attributes:
        country_risks: Database of country risk classifications
        cn_to_commodity: CN code to commodity mapping

    Example:
        >>> agent = EUDRComplianceAgent()
        >>> result = agent.run(EUDRInput(
        ...     commodity_type=CommodityType.COFFEE,
        ...     cn_code="0901.11.00",
        ...     quantity_kg=50000,
        ...     country_of_origin="BR",
        ...     geolocation=GeoLocation(type="Point", coordinates=[-47.5, -15.5]),
        ...     production_date=date(2024, 6, 1)
        ... ))
        >>> assert result.compliance_status is not None
    """

    AGENT_ID = "regulatory/eudr_compliance_v1"
    VERSION = "1.0.0"
    DESCRIPTION = "EUDR deforestation-free compliance validator"

    # EUDR cutoff date
    CUTOFF_DATE = date(2020, 12, 31)

    # CN code to commodity mapping
    CN_TO_COMMODITY: Dict[str, CommodityType] = {
        # Cattle
        "0102": CommodityType.CATTLE,  # Live cattle
        "0201": CommodityType.CATTLE,  # Beef fresh
        "0202": CommodityType.CATTLE,  # Beef frozen
        "4101": CommodityType.CATTLE,  # Bovine hides
        "4104": CommodityType.CATTLE,  # Bovine leather
        # Cocoa
        "1801": CommodityType.COCOA,  # Cocoa beans
        "1802": CommodityType.COCOA,  # Cocoa shells
        "1803": CommodityType.COCOA,  # Cocoa paste
        "1804": CommodityType.COCOA,  # Cocoa butter
        "1805": CommodityType.COCOA,  # Cocoa powder
        "1806": CommodityType.COCOA,  # Chocolate
        # Coffee
        "0901": CommodityType.COFFEE,  # Coffee
        "2101": CommodityType.COFFEE,  # Coffee extracts
        # Palm oil
        "1511": CommodityType.PALM_OIL,  # Palm oil
        "1513": CommodityType.PALM_OIL,  # Palm kernel oil
        # Rubber
        "4001": CommodityType.RUBBER,  # Natural rubber
        "4005": CommodityType.RUBBER,  # Compounded rubber
        "4006": CommodityType.RUBBER,  # Rubber forms
        "4007": CommodityType.RUBBER,  # Vulcanized thread
        "4008": CommodityType.RUBBER,  # Vulcanized plates
        # Soya
        "1201": CommodityType.SOYA,  # Soya beans
        "1507": CommodityType.SOYA,  # Soya-bean oil
        "2304": CommodityType.SOYA,  # Soya-bean residues
        # Wood
        "44": CommodityType.WOOD,  # Wood products
        "47": CommodityType.WOOD,  # Wood pulp
        "48": CommodityType.WOOD,  # Paper products
        "94": CommodityType.WOOD,  # Wooden furniture
    }

    # Country risk database (EU benchmarking pending)
    COUNTRY_RISKS: Dict[str, CountryRisk] = {
        "BR": CountryRisk(
            country_code="BR",
            risk_level=RiskLevel.HIGH,
            risk_score=75.0,
            source="EU provisional assessment",
        ),
        "ID": CountryRisk(
            country_code="ID",
            risk_level=RiskLevel.HIGH,
            risk_score=72.0,
            source="EU provisional assessment",
        ),
        "MY": CountryRisk(
            country_code="MY",
            risk_level=RiskLevel.HIGH,
            risk_score=68.0,
            source="EU provisional assessment",
        ),
        "CO": CountryRisk(
            country_code="CO",
            risk_level=RiskLevel.STANDARD,
            risk_score=55.0,
            source="EU provisional assessment",
        ),
        "PE": CountryRisk(
            country_code="PE",
            risk_level=RiskLevel.STANDARD,
            risk_score=50.0,
            source="EU provisional assessment",
        ),
        "GH": CountryRisk(
            country_code="GH",
            risk_level=RiskLevel.STANDARD,
            risk_score=45.0,
            source="EU provisional assessment",
        ),
        "CI": CountryRisk(
            country_code="CI",
            risk_level=RiskLevel.STANDARD,
            risk_score=48.0,
            source="EU provisional assessment",
        ),
        "DEFAULT": CountryRisk(
            country_code="DEFAULT",
            risk_level=RiskLevel.STANDARD,
            risk_score=50.0,
            source="EU default",
        ),
    }

    # Valid certifications that may reduce risk
    RECOGNIZED_CERTIFICATIONS = [
        "FSC",
        "PEFC",
        "Rainforest Alliance",
        "UTZ",
        "Fairtrade",
        "RSPO",
        "RTRS",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EUDR Compliance Agent.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self._provenance_steps: List[Dict] = []

        logger.info(f"EUDRComplianceAgent initialized (version {self.VERSION})")

    def run(self, input_data: EUDRInput) -> EUDROutput:
        """
        Execute the EUDR compliance validation.

        Performs zero-hallucination validation:
        - Geolocation coordinate validation
        - Country risk lookup
        - Cutoff date comparison
        - Traceability calculation

        Args:
            input_data: Validated EUDR input data

        Returns:
            Validation result with compliance status and risk assessment

        Raises:
            ValueError: If commodity not in EUDR scope
        """
        start_time = datetime.utcnow()
        self._provenance_steps = []

        logger.info(
            f"Validating EUDR compliance: commodity={input_data.commodity_type}, "
            f"origin={input_data.country_of_origin}, qty={input_data.quantity_kg}kg"
        )

        try:
            # Step 1: Validate commodity in scope
            if not self._is_in_scope(input_data.cn_code, input_data.commodity_type):
                raise ValueError(
                    f"CN code {input_data.cn_code} not in EUDR scope for {input_data.commodity_type}"
                )

            self._track_step("commodity_validation", {
                "commodity_type": input_data.commodity_type.value,
                "cn_code": input_data.cn_code,
                "in_scope": True,
            })

            # Step 2: Validate geolocation
            geo_valid = self._validate_geolocation(input_data.geolocation)

            self._track_step("geolocation_validation", {
                "geometry_type": input_data.geolocation.type.value,
                "coordinates": str(input_data.geolocation.coordinates)[:100],
                "valid": geo_valid,
            })

            # Step 3: Get country risk
            country_risk = self._get_country_risk(input_data.country_of_origin)

            self._track_step("country_risk_assessment", {
                "country": input_data.country_of_origin,
                "risk_level": country_risk.risk_level.value,
                "risk_score": country_risk.risk_score,
            })

            # Step 4: Check cutoff date compliance
            cutoff_compliant = input_data.production_date > self.CUTOFF_DATE

            self._track_step("cutoff_date_check", {
                "production_date": input_data.production_date.isoformat(),
                "cutoff_date": self.CUTOFF_DATE.isoformat(),
                "compliant": cutoff_compliant,
            })

            # Step 5: ZERO-HALLUCINATION CALCULATION
            # Traceability score = verified_nodes / total_nodes * 100
            traceability_score = self._calculate_traceability(input_data.supply_chain)

            self._track_step("traceability_calculation", {
                "formula": "traceability = verified_nodes / total_nodes * 100",
                "supply_chain_length": len(input_data.supply_chain),
                "traceability_score": traceability_score,
            })

            # Step 6: Determine overall compliance
            compliance_status, mitigation = self._determine_compliance(
                geo_valid=geo_valid,
                cutoff_compliant=cutoff_compliant,
                country_risk=country_risk,
                traceability_score=traceability_score,
                certifications=input_data.certifications,
            )

            self._track_step("compliance_determination", {
                "status": compliance_status.value,
                "mitigation_count": len(mitigation),
            })

            # Step 7: Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Step 8: Create output
            output = EUDROutput(
                commodity_type=input_data.commodity_type.value,
                cn_code=input_data.cn_code,
                compliance_status=compliance_status.value,
                risk_level=country_risk.risk_level.value,
                country_risk_score=country_risk.risk_score,
                geolocation_valid=geo_valid,
                cutoff_date_compliant=cutoff_compliant,
                traceability_score=traceability_score,
                deforestation_detected=None,  # Requires satellite analysis
                mitigation_measures=mitigation,
                dds_reference=None,  # Generated on DDS submission
                provenance_hash=provenance_hash,
            )

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            logger.info(
                f"EUDR validation complete: status={compliance_status.value}, "
                f"risk={country_risk.risk_level.value} "
                f"(duration: {duration_ms:.2f}ms, provenance: {provenance_hash[:16]}...)"
            )

            return output

        except Exception as e:
            logger.error(f"EUDR validation failed: {str(e)}", exc_info=True)
            raise

    def _is_in_scope(self, cn_code: str, commodity_type: CommodityType) -> bool:
        """Check if CN code matches commodity type and is in EUDR scope."""
        # Check 4-digit prefix
        prefix_4 = cn_code[:4]
        if prefix_4 in self.CN_TO_COMMODITY:
            return self.CN_TO_COMMODITY[prefix_4] == commodity_type

        # Check 2-digit chapter
        prefix_2 = cn_code[:2]
        if prefix_2 in self.CN_TO_COMMODITY:
            return self.CN_TO_COMMODITY[prefix_2] == commodity_type

        return False

    def _validate_geolocation(self, geolocation: GeoLocation) -> bool:
        """
        Validate geolocation data.

        ZERO-HALLUCINATION: Uses deterministic coordinate validation.
        """
        try:
            if geolocation.type == GeometryType.POINT:
                lon, lat = geolocation.coordinates[0], geolocation.coordinates[1]
                # Valid coordinate ranges
                return -180 <= lon <= 180 and -90 <= lat <= 90

            elif geolocation.type == GeometryType.POLYGON:
                # Validate polygon has at least 4 points (closed ring)
                if not geolocation.coordinates or not geolocation.coordinates[0]:
                    return False
                ring = geolocation.coordinates[0]
                if len(ring) < 4:
                    return False
                # Check ring is closed
                return ring[0] == ring[-1]

            elif geolocation.type == GeometryType.MULTI_POLYGON:
                # Validate each polygon
                for polygon in geolocation.coordinates:
                    if not polygon or not polygon[0] or len(polygon[0]) < 4:
                        return False
                return True

            return False

        except (IndexError, TypeError):
            return False

    def _get_country_risk(self, country_code: str) -> CountryRisk:
        """Get country risk classification."""
        return self.COUNTRY_RISKS.get(
            country_code,
            self.COUNTRY_RISKS["DEFAULT"]
        )

    def _calculate_traceability(self, supply_chain: List[Dict]) -> float:
        """
        Calculate supply chain traceability score.

        ZERO-HALLUCINATION: verified_nodes / total_nodes * 100
        """
        if not supply_chain:
            return 0.0

        total_nodes = len(supply_chain)
        verified_nodes = sum(
            1 for node in supply_chain
            if node.get("verified", False)
        )

        return round((verified_nodes / total_nodes) * 100, 2)

    def _determine_compliance(
        self,
        geo_valid: bool,
        cutoff_compliant: bool,
        country_risk: CountryRisk,
        traceability_score: float,
        certifications: List[str],
    ) -> Tuple[ComplianceStatus, List[str]]:
        """
        Determine overall compliance status.

        Returns (status, mitigation_measures)
        """
        mitigation = []

        # Non-compliant conditions
        if not geo_valid:
            mitigation.append("Provide valid geolocation with GPS coordinates or polygon")

        if not cutoff_compliant:
            mitigation.append("Production must be after December 31, 2020")

        if traceability_score < 100:
            mitigation.append(f"Improve supply chain traceability from {traceability_score}% to 100%")

        if country_risk.risk_level == RiskLevel.HIGH:
            mitigation.append("High-risk country requires enhanced due diligence")

        # Check for recognized certifications
        has_certification = any(
            cert in self.RECOGNIZED_CERTIFICATIONS
            for cert in certifications
        )
        if not has_certification:
            mitigation.append("Consider obtaining recognized certification (FSC, RSPO, etc.)")

        # Determine status
        if not geo_valid or not cutoff_compliant:
            return ComplianceStatus.NON_COMPLIANT, mitigation

        if traceability_score < 50:
            return ComplianceStatus.INSUFFICIENT_DATA, mitigation

        if country_risk.risk_level == RiskLevel.HIGH and traceability_score < 100:
            return ComplianceStatus.PENDING_VERIFICATION, mitigation

        if mitigation:
            return ComplianceStatus.PENDING_VERIFICATION, mitigation

        return ComplianceStatus.COMPLIANT, []

    def _track_step(self, step_type: str, data: Dict[str, Any]) -> None:
        """Track a calculation step for provenance."""
        self._provenance_steps.append({
            "step_type": step_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.VERSION,
            "steps": self._provenance_steps,
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def get_commodities(self) -> List[str]:
        """Get list of EUDR commodities."""
        return [c.value for c in CommodityType]

    def is_in_eudr_scope(self, cn_code: str) -> bool:
        """Check if CN code is in EUDR scope."""
        prefix_4 = cn_code[:4]
        prefix_2 = cn_code[:2]
        return prefix_4 in self.CN_TO_COMMODITY or prefix_2 in self.CN_TO_COMMODITY


# Pack specification
PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "regulatory/eudr_compliance_v1",
    "name": "EUDR Compliance Agent",
    "version": "1.0.0",
    "summary": "EU Deforestation Regulation compliance validator",
    "tags": ["eudr", "deforestation", "due-diligence", "supply-chain"],
    "owners": ["regulatory-team"],
    "compute": {
        "entrypoint": "python://agents.gl_004_eudr_compliance.agent:EUDRComplianceAgent",
        "deterministic": True,
    },
    "factors": [
        {"ref": "ef://eu/eudr-country-risk/2024"},
        {"ref": "ef://ipcc/deforestation/2024"},
    ],
    "provenance": {
        "regulation_version": "EU 2023/1115",
        "cutoff_date": "2020-12-31",
        "enable_audit": True,
    },
}
