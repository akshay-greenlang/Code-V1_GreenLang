"""
EUDR (EU Deforestation Regulation) Traceability Module.

This module provides comprehensive traceability infrastructure for compliance
with the EU Deforestation Regulation (EU 2023/1115). It supports:

- Plot-level geolocation for commodity origins
- Chain of custody tracking through supply chain
- Due diligence statement generation
- Risk assessment at plot, supplier, and aggregate levels
- Compliance status tracking

Key Requirements Addressed:
- Article 9: Geolocation requirements (plot coordinates)
- Article 10: Due diligence statements
- Article 11: Risk assessment and mitigation
- Article 12: Information system integration

EUDR Timeline:
- December 30, 2024: Entry into force for large operators
- June 30, 2025: Entry into force for SMEs

Example:
    >>> from greenlang.supply_chain.eudr import EUDRTraceabilityManager
    >>> manager = EUDRTraceabilityManager(operator_id="OP123")
    >>>
    >>> # Register a production plot
    >>> plot = manager.register_plot(
    ...     plot_id="PLOT001",
    ...     commodity=EUDRCommodity.COCOA,
    ...     latitude=-4.0383,
    ...     longitude=-79.2036,
    ...     country_code="EC",
    ...     producer_name="Cacao Farm Ecuador"
    ... )
    >>>
    >>> # Generate due diligence statement
    >>> statement = manager.generate_due_diligence_statement(
    ...     product_id="PROD001",
    ...     plot_ids=["PLOT001"]
    ... )
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
from typing import Optional, List, Dict, Any, Set, Tuple
import logging
import json

from greenlang.supply_chain.models.entity import (
    Supplier,
    Facility,
    GeoLocation,
    CommodityType,
)

logger = logging.getLogger(__name__)


class EUDRCommodity(Enum):
    """
    EUDR regulated commodities and derived products.

    The seven commodities covered by EUDR and their derived products.
    """
    # Primary commodities
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

    # Derived products (selected examples)
    BEEF = "beef"
    LEATHER = "leather"
    CHOCOLATE = "chocolate"
    PALM_OIL = "palm_oil"
    NATURAL_RUBBER = "natural_rubber"
    TYRES = "tyres"
    SOYBEAN_OIL = "soybean_oil"
    SOYBEAN_MEAL = "soybean_meal"
    TIMBER = "timber"
    FURNITURE = "furniture"
    PAPER = "paper"
    CHARCOAL = "charcoal"

    @property
    def is_primary(self) -> bool:
        """Check if this is a primary commodity."""
        return self in {
            EUDRCommodity.CATTLE,
            EUDRCommodity.COCOA,
            EUDRCommodity.COFFEE,
            EUDRCommodity.OIL_PALM,
            EUDRCommodity.RUBBER,
            EUDRCommodity.SOYA,
            EUDRCommodity.WOOD,
        }

    @property
    def primary_commodity(self) -> "EUDRCommodity":
        """Get the primary commodity for a derived product."""
        derived_mapping = {
            EUDRCommodity.BEEF: EUDRCommodity.CATTLE,
            EUDRCommodity.LEATHER: EUDRCommodity.CATTLE,
            EUDRCommodity.CHOCOLATE: EUDRCommodity.COCOA,
            EUDRCommodity.PALM_OIL: EUDRCommodity.OIL_PALM,
            EUDRCommodity.NATURAL_RUBBER: EUDRCommodity.RUBBER,
            EUDRCommodity.TYRES: EUDRCommodity.RUBBER,
            EUDRCommodity.SOYBEAN_OIL: EUDRCommodity.SOYA,
            EUDRCommodity.SOYBEAN_MEAL: EUDRCommodity.SOYA,
            EUDRCommodity.TIMBER: EUDRCommodity.WOOD,
            EUDRCommodity.FURNITURE: EUDRCommodity.WOOD,
            EUDRCommodity.PAPER: EUDRCommodity.WOOD,
            EUDRCommodity.CHARCOAL: EUDRCommodity.WOOD,
        }
        return derived_mapping.get(self, self)


class RiskLevel(Enum):
    """EUDR risk classification levels."""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    UNKNOWN = "unknown"


class ComplianceStatus(Enum):
    """EUDR compliance status for products/operators."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    UNDER_REVIEW = "under_review"
    EXEMPTED = "exempted"


class LandUseType(Enum):
    """Land use classification for deforestation assessment."""
    FOREST = "forest"
    PLANTATION = "plantation"
    AGRICULTURAL = "agricultural"
    PASTURE = "pasture"
    DEGRADED = "degraded"
    OTHER = "other"


@dataclass
class PlotRecord:
    """
    Plot-level record for EUDR geolocation requirements.

    Represents a production plot (farm, forest, plantation) where
    EUDR commodities are produced. Required for Article 9 compliance.

    Attributes:
        id: Unique plot identifier
        commodity: EUDR commodity produced
        latitude: Plot center latitude
        longitude: Plot center longitude
        polygon_wkt: Well-Known Text polygon boundary (optional)
        area_hectares: Plot area in hectares
        country_code: ISO country code
        region: Administrative region
        producer_id: Producer/farmer ID
        producer_name: Producer name
        production_date: Date of harvest/production
        quantity: Production quantity
        unit: Unit of measure
        certification: Certification standard (FSC, RSPO, etc.)
        land_use_type: Land use classification
        deforestation_free: Deforestation-free status
        deforestation_cutoff_date: Date for deforestation assessment
        legal_compliance: Legal compliance status
        supporting_documents: List of document references
        risk_level: Assessed risk level
        created_at: Record creation timestamp
        updated_at: Last update timestamp
        metadata: Additional attributes
    """
    id: str
    commodity: EUDRCommodity
    latitude: float
    longitude: float
    polygon_wkt: Optional[str] = None
    area_hectares: Optional[Decimal] = None
    country_code: str = ""
    region: Optional[str] = None
    producer_id: Optional[str] = None
    producer_name: Optional[str] = None
    production_date: Optional[date] = None
    quantity: Optional[Decimal] = None
    unit: str = "kg"
    certification: Optional[str] = None
    land_use_type: LandUseType = LandUseType.OTHER
    deforestation_free: bool = False
    deforestation_cutoff_date: date = field(default_factory=lambda: date(2020, 12, 31))
    legal_compliance: bool = False
    supporting_documents: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate plot record data."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Invalid longitude: {self.longitude}")

    @property
    def location(self) -> GeoLocation:
        """Get plot location as GeoLocation object."""
        return GeoLocation(
            latitude=self.latitude,
            longitude=self.longitude,
            polygon_wkt=self.polygon_wkt,
        )

    @property
    def is_compliant(self) -> bool:
        """Check if plot meets basic EUDR requirements."""
        return self.deforestation_free and self.legal_compliance

    def calculate_hash(self) -> str:
        """
        Calculate a hash for integrity verification.

        Returns:
            SHA-256 hash of key plot attributes
        """
        data = f"{self.id}:{self.commodity.value}:{self.latitude}:{self.longitude}:{self.production_date}"
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "commodity": self.commodity.value,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "polygon_wkt": self.polygon_wkt,
            "area_hectares": str(self.area_hectares) if self.area_hectares else None,
            "country_code": self.country_code,
            "region": self.region,
            "producer_id": self.producer_id,
            "producer_name": self.producer_name,
            "production_date": self.production_date.isoformat() if self.production_date else None,
            "quantity": str(self.quantity) if self.quantity else None,
            "unit": self.unit,
            "certification": self.certification,
            "land_use_type": self.land_use_type.value,
            "deforestation_free": self.deforestation_free,
            "deforestation_cutoff_date": self.deforestation_cutoff_date.isoformat(),
            "legal_compliance": self.legal_compliance,
            "supporting_documents": self.supporting_documents,
            "risk_level": self.risk_level.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotRecord":
        """Create from dictionary representation."""
        return cls(
            id=data["id"],
            commodity=EUDRCommodity(data["commodity"]),
            latitude=data["latitude"],
            longitude=data["longitude"],
            polygon_wkt=data.get("polygon_wkt"),
            area_hectares=Decimal(data["area_hectares"]) if data.get("area_hectares") else None,
            country_code=data.get("country_code", ""),
            region=data.get("region"),
            producer_id=data.get("producer_id"),
            producer_name=data.get("producer_name"),
            production_date=date.fromisoformat(data["production_date"]) if data.get("production_date") else None,
            quantity=Decimal(data["quantity"]) if data.get("quantity") else None,
            unit=data.get("unit", "kg"),
            certification=data.get("certification"),
            land_use_type=LandUseType(data.get("land_use_type", "other")),
            deforestation_free=data.get("deforestation_free", False),
            deforestation_cutoff_date=date.fromisoformat(data.get("deforestation_cutoff_date", "2020-12-31")),
            legal_compliance=data.get("legal_compliance", False),
            supporting_documents=data.get("supporting_documents", []),
            risk_level=RiskLevel(data.get("risk_level", "unknown")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ChainOfCustodyRecord:
    """
    Chain of custody record for tracking product movement.

    Documents the transfer of EUDR commodities between supply chain
    actors, maintaining traceability from origin to final product.

    Attributes:
        id: Unique record identifier
        transaction_id: Transaction/shipment reference
        source_operator_id: Sending operator ID
        source_operator_name: Sending operator name
        target_operator_id: Receiving operator ID
        target_operator_name: Receiving operator name
        commodity: EUDR commodity
        product_description: Product description
        quantity: Transaction quantity
        unit: Unit of measure
        batch_number: Batch/lot identifier
        origin_plot_ids: Source plot IDs
        transaction_date: Date of transaction
        transport_mode: Mode of transport
        transport_documents: Transport document references
        customs_declaration: Customs declaration reference
        cn_code: Combined Nomenclature code
        hs_code: Harmonized System code
        verification_status: Verification status
        verified_by: Verifier identity
        verified_at: Verification timestamp
        blockchain_hash: Blockchain transaction hash (optional)
        metadata: Additional attributes
    """
    id: str
    transaction_id: str
    source_operator_id: str
    source_operator_name: str
    target_operator_id: str
    target_operator_name: str
    commodity: EUDRCommodity
    product_description: str
    quantity: Decimal
    unit: str = "kg"
    batch_number: Optional[str] = None
    origin_plot_ids: List[str] = field(default_factory=list)
    transaction_date: datetime = field(default_factory=datetime.utcnow)
    transport_mode: Optional[str] = None
    transport_documents: List[str] = field(default_factory=list)
    customs_declaration: Optional[str] = None
    cn_code: Optional[str] = None
    hs_code: Optional[str] = None
    verification_status: ComplianceStatus = ComplianceStatus.PENDING_VERIFICATION
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    blockchain_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_operator_id: str,
        source_operator_name: str,
        target_operator_id: str,
        target_operator_name: str,
        commodity: EUDRCommodity,
        product_description: str,
        quantity: Decimal,
        **kwargs
    ) -> "ChainOfCustodyRecord":
        """Factory method to create a new chain of custody record."""
        record_id = f"COC-{uuid.uuid4().hex[:12].upper()}"
        transaction_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
        return cls(
            id=record_id,
            transaction_id=transaction_id,
            source_operator_id=source_operator_id,
            source_operator_name=source_operator_name,
            target_operator_id=target_operator_id,
            target_operator_name=target_operator_name,
            commodity=commodity,
            product_description=product_description,
            quantity=quantity,
            **kwargs
        )

    def verify(self, verifier: str) -> None:
        """Mark record as verified."""
        self.verification_status = ComplianceStatus.COMPLIANT
        self.verified_by = verifier
        self.verified_at = datetime.utcnow()

    def calculate_hash(self) -> str:
        """Calculate integrity hash for the record."""
        data = json.dumps({
            "id": self.id,
            "transaction_id": self.transaction_id,
            "source": self.source_operator_id,
            "target": self.target_operator_id,
            "commodity": self.commodity.value,
            "quantity": str(self.quantity),
            "date": self.transaction_date.isoformat(),
            "plots": sorted(self.origin_plot_ids),
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "transaction_id": self.transaction_id,
            "source_operator_id": self.source_operator_id,
            "source_operator_name": self.source_operator_name,
            "target_operator_id": self.target_operator_id,
            "target_operator_name": self.target_operator_name,
            "commodity": self.commodity.value,
            "product_description": self.product_description,
            "quantity": str(self.quantity),
            "unit": self.unit,
            "batch_number": self.batch_number,
            "origin_plot_ids": self.origin_plot_ids,
            "transaction_date": self.transaction_date.isoformat(),
            "transport_mode": self.transport_mode,
            "transport_documents": self.transport_documents,
            "customs_declaration": self.customs_declaration,
            "cn_code": self.cn_code,
            "hs_code": self.hs_code,
            "verification_status": self.verification_status.value,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "blockchain_hash": self.blockchain_hash,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class DueDiligenceStatement:
    """
    Due diligence statement for EUDR compliance (Article 10).

    This is the formal declaration submitted to the EU Information System
    confirming that products are deforestation-free and legally produced.

    Attributes:
        id: Statement reference number
        operator_id: Operator identification
        operator_name: Operator legal name
        operator_country: Operator country
        statement_type: Type (import, export, domestic)
        commodity: EUDR commodity
        product_description: Product description
        cn_codes: Combined Nomenclature codes
        quantity: Total quantity
        unit: Unit of measure
        origin_countries: Countries of origin
        origin_plots: Plot references
        chain_of_custody: Chain of custody records
        risk_assessment_result: Risk assessment outcome
        risk_level: Overall risk level
        risk_mitigation_measures: Mitigation measures applied
        deforestation_free_declaration: Declaration of deforestation-free
        legal_compliance_declaration: Declaration of legal compliance
        supporting_evidence: Evidence references
        submission_date: Statement submission date
        validity_start: Validity period start
        validity_end: Validity period end
        status: Compliance status
        eu_reference_number: EU system reference (after submission)
        digital_signature: Digital signature
        metadata: Additional attributes
    """
    id: str
    operator_id: str
    operator_name: str
    operator_country: str
    statement_type: str = "import"  # import, export, domestic
    commodity: EUDRCommodity = EUDRCommodity.COCOA
    product_description: str = ""
    cn_codes: List[str] = field(default_factory=list)
    quantity: Decimal = Decimal("0")
    unit: str = "kg"
    origin_countries: List[str] = field(default_factory=list)
    origin_plots: List[str] = field(default_factory=list)
    chain_of_custody: List[str] = field(default_factory=list)
    risk_assessment_result: str = ""
    risk_level: RiskLevel = RiskLevel.UNKNOWN
    risk_mitigation_measures: List[str] = field(default_factory=list)
    deforestation_free_declaration: bool = False
    legal_compliance_declaration: bool = False
    supporting_evidence: List[str] = field(default_factory=list)
    submission_date: Optional[datetime] = None
    validity_start: Optional[date] = None
    validity_end: Optional[date] = None
    status: ComplianceStatus = ComplianceStatus.PENDING_VERIFICATION
    eu_reference_number: Optional[str] = None
    digital_signature: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        operator_id: str,
        operator_name: str,
        operator_country: str,
        commodity: EUDRCommodity,
        **kwargs
    ) -> "DueDiligenceStatement":
        """Factory method to create a new due diligence statement."""
        statement_id = f"DDS-{uuid.uuid4().hex[:12].upper()}"
        return cls(
            id=statement_id,
            operator_id=operator_id,
            operator_name=operator_name,
            operator_country=operator_country,
            commodity=commodity,
            **kwargs
        )

    @property
    def is_complete(self) -> bool:
        """Check if statement has all required fields."""
        return all([
            self.operator_id,
            self.operator_name,
            self.cn_codes,
            self.quantity > 0,
            self.origin_countries,
            self.origin_plots,
            self.deforestation_free_declaration,
            self.legal_compliance_declaration,
        ])

    @property
    def is_valid(self) -> bool:
        """Check if statement is currently valid."""
        if not self.validity_start or not self.validity_end:
            return False
        today = date.today()
        return self.validity_start <= today <= self.validity_end

    def sign(self, signature: str) -> None:
        """Apply digital signature to the statement."""
        self.digital_signature = signature
        self.updated_at = datetime.utcnow()

    def submit(self) -> None:
        """Mark statement as submitted."""
        if not self.is_complete:
            raise ValueError("Statement is incomplete")
        self.submission_date = datetime.utcnow()
        self.status = ComplianceStatus.PENDING_VERIFICATION
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "operator_country": self.operator_country,
            "statement_type": self.statement_type,
            "commodity": self.commodity.value,
            "product_description": self.product_description,
            "cn_codes": self.cn_codes,
            "quantity": str(self.quantity),
            "unit": self.unit,
            "origin_countries": self.origin_countries,
            "origin_plots": self.origin_plots,
            "chain_of_custody": self.chain_of_custody,
            "risk_assessment_result": self.risk_assessment_result,
            "risk_level": self.risk_level.value,
            "risk_mitigation_measures": self.risk_mitigation_measures,
            "deforestation_free_declaration": self.deforestation_free_declaration,
            "legal_compliance_declaration": self.legal_compliance_declaration,
            "supporting_evidence": self.supporting_evidence,
            "submission_date": self.submission_date.isoformat() if self.submission_date else None,
            "validity_start": self.validity_start.isoformat() if self.validity_start else None,
            "validity_end": self.validity_end.isoformat() if self.validity_end else None,
            "status": self.status.value,
            "eu_reference_number": self.eu_reference_number,
            "digital_signature": self.digital_signature,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


class EUDRTraceabilityManager:
    """
    Central manager for EUDR traceability and due diligence.

    Provides comprehensive functionality for:
    - Plot registration and management
    - Chain of custody tracking
    - Due diligence statement generation
    - Risk assessment aggregation
    - Compliance reporting

    Example:
        >>> manager = EUDRTraceabilityManager(
        ...     operator_id="OP123",
        ...     operator_name="My Company Ltd"
        ... )
        >>>
        >>> # Register production plots
        >>> plot = manager.register_plot(
        ...     plot_id="PLOT001",
        ...     commodity=EUDRCommodity.COCOA,
        ...     latitude=-4.0383,
        ...     longitude=-79.2036,
        ...     country_code="EC"
        ... )
        >>>
        >>> # Record chain of custody
        >>> coc = manager.record_transfer(
        ...     source_operator_id="FARM001",
        ...     target_operator_id="TRADER001",
        ...     commodity=EUDRCommodity.COCOA,
        ...     quantity=Decimal("5000"),
        ...     origin_plot_ids=["PLOT001"]
        ... )
        >>>
        >>> # Generate due diligence statement
        >>> statement = manager.generate_due_diligence_statement(
        ...     commodity=EUDRCommodity.COCOA,
        ...     product_description="Cocoa beans",
        ...     cn_codes=["18010000"],
        ...     quantity=Decimal("5000"),
        ...     origin_plot_ids=["PLOT001"]
        ... )
    """

    # EUDR deforestation cutoff date
    DEFORESTATION_CUTOFF = date(2020, 12, 31)

    # High-risk countries (EUDR benchmarking)
    HIGH_RISK_COUNTRIES: Set[str] = {
        "BR", "ID", "MY", "AR", "PY", "BO", "CO", "PE", "EC",
        "CG", "CD", "CM", "CI", "GH", "NG", "LA", "MM", "PG",
    }

    def __init__(
        self,
        operator_id: str,
        operator_name: str,
        operator_country: str = "DE",
    ):
        """
        Initialize the EUDR traceability manager.

        Args:
            operator_id: Operator identification number
            operator_name: Legal name of the operator
            operator_country: Country of the operator
        """
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.operator_country = operator_country

        # Data stores
        self._plots: Dict[str, PlotRecord] = {}
        self._chain_of_custody: Dict[str, ChainOfCustodyRecord] = {}
        self._statements: Dict[str, DueDiligenceStatement] = {}

        # Indexes
        self._plots_by_commodity: Dict[EUDRCommodity, Set[str]] = {}
        self._plots_by_country: Dict[str, Set[str]] = {}

        logger.info(
            f"EUDRTraceabilityManager initialized for operator: {operator_id}"
        )

    # =========================================================================
    # Plot Management
    # =========================================================================

    def register_plot(
        self,
        plot_id: str,
        commodity: EUDRCommodity,
        latitude: float,
        longitude: float,
        country_code: str,
        **kwargs
    ) -> PlotRecord:
        """
        Register a production plot for EUDR traceability.

        Args:
            plot_id: Unique plot identifier
            commodity: EUDR commodity produced
            latitude: Plot latitude
            longitude: Plot longitude
            country_code: ISO country code
            **kwargs: Additional PlotRecord attributes

        Returns:
            Created PlotRecord
        """
        # Assess initial risk level
        risk_level = self._assess_plot_risk(country_code, commodity)

        plot = PlotRecord(
            id=plot_id,
            commodity=commodity,
            latitude=latitude,
            longitude=longitude,
            country_code=country_code,
            risk_level=risk_level,
            **kwargs
        )

        # Store and index
        self._plots[plot_id] = plot

        if commodity not in self._plots_by_commodity:
            self._plots_by_commodity[commodity] = set()
        self._plots_by_commodity[commodity].add(plot_id)

        if country_code not in self._plots_by_country:
            self._plots_by_country[country_code] = set()
        self._plots_by_country[country_code].add(plot_id)

        logger.info(f"Registered plot: {plot_id} in {country_code}")
        return plot

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        """Get plot by ID."""
        return self._plots.get(plot_id)

    def get_plots_by_commodity(self, commodity: EUDRCommodity) -> List[PlotRecord]:
        """Get all plots for a commodity."""
        plot_ids = self._plots_by_commodity.get(commodity, set())
        return [self._plots[pid] for pid in plot_ids if pid in self._plots]

    def get_plots_by_country(self, country_code: str) -> List[PlotRecord]:
        """Get all plots in a country."""
        plot_ids = self._plots_by_country.get(country_code, set())
        return [self._plots[pid] for pid in plot_ids if pid in self._plots]

    def update_plot_compliance(
        self,
        plot_id: str,
        deforestation_free: bool,
        legal_compliance: bool,
        supporting_documents: Optional[List[str]] = None,
    ) -> Optional[PlotRecord]:
        """
        Update compliance status for a plot.

        Args:
            plot_id: Plot identifier
            deforestation_free: Deforestation-free status
            legal_compliance: Legal compliance status
            supporting_documents: Supporting document references

        Returns:
            Updated PlotRecord or None if not found
        """
        plot = self._plots.get(plot_id)
        if not plot:
            return None

        plot.deforestation_free = deforestation_free
        plot.legal_compliance = legal_compliance
        if supporting_documents:
            plot.supporting_documents.extend(supporting_documents)
        plot.updated_at = datetime.utcnow()

        # Reassess risk based on compliance
        if deforestation_free and legal_compliance:
            if plot.risk_level == RiskLevel.HIGH:
                plot.risk_level = RiskLevel.STANDARD
            elif plot.risk_level == RiskLevel.STANDARD:
                plot.risk_level = RiskLevel.LOW

        logger.info(f"Updated compliance for plot: {plot_id}")
        return plot

    def _assess_plot_risk(
        self,
        country_code: str,
        commodity: EUDRCommodity,
    ) -> RiskLevel:
        """
        Assess initial risk level for a plot.

        Based on:
        - Country risk classification
        - Commodity risk profile

        Args:
            country_code: ISO country code
            commodity: EUDR commodity

        Returns:
            Assessed risk level
        """
        # High-risk country
        if country_code in self.HIGH_RISK_COUNTRIES:
            return RiskLevel.HIGH

        # Commodity-specific risk (simplified)
        high_risk_commodities = {
            EUDRCommodity.CATTLE,
            EUDRCommodity.SOYA,
            EUDRCommodity.OIL_PALM,
        }
        if commodity.primary_commodity in high_risk_commodities:
            return RiskLevel.STANDARD

        return RiskLevel.STANDARD

    # =========================================================================
    # Chain of Custody
    # =========================================================================

    def record_transfer(
        self,
        source_operator_id: str,
        source_operator_name: str,
        target_operator_id: str,
        target_operator_name: str,
        commodity: EUDRCommodity,
        product_description: str,
        quantity: Decimal,
        origin_plot_ids: List[str],
        **kwargs
    ) -> ChainOfCustodyRecord:
        """
        Record a chain of custody transfer.

        Args:
            source_operator_id: Sending operator ID
            source_operator_name: Sending operator name
            target_operator_id: Receiving operator ID
            target_operator_name: Receiving operator name
            commodity: EUDR commodity
            product_description: Product description
            quantity: Transfer quantity
            origin_plot_ids: Source plot IDs
            **kwargs: Additional record attributes

        Returns:
            Created ChainOfCustodyRecord
        """
        record = ChainOfCustodyRecord.create(
            source_operator_id=source_operator_id,
            source_operator_name=source_operator_name,
            target_operator_id=target_operator_id,
            target_operator_name=target_operator_name,
            commodity=commodity,
            product_description=product_description,
            quantity=quantity,
            origin_plot_ids=origin_plot_ids,
            **kwargs
        )

        self._chain_of_custody[record.id] = record
        logger.info(f"Recorded transfer: {record.id}")
        return record

    def get_chain_of_custody(self, record_id: str) -> Optional[ChainOfCustodyRecord]:
        """Get chain of custody record by ID."""
        return self._chain_of_custody.get(record_id)

    def trace_product_origin(
        self,
        product_batch: str,
    ) -> List[PlotRecord]:
        """
        Trace a product batch back to its origin plots.

        Args:
            product_batch: Batch/lot number

        Returns:
            List of origin PlotRecords
        """
        # Find all CoC records for this batch
        plot_ids: Set[str] = set()
        for record in self._chain_of_custody.values():
            if record.batch_number == product_batch:
                plot_ids.update(record.origin_plot_ids)

        return [self._plots[pid] for pid in plot_ids if pid in self._plots]

    def get_full_chain(
        self,
        operator_id: str,
        commodity: Optional[EUDRCommodity] = None,
    ) -> List[ChainOfCustodyRecord]:
        """
        Get full chain of custody for an operator.

        Args:
            operator_id: Operator to trace
            commodity: Filter by commodity (optional)

        Returns:
            List of related CoC records
        """
        records = []
        for record in self._chain_of_custody.values():
            if record.target_operator_id == operator_id or record.source_operator_id == operator_id:
                if commodity is None or record.commodity == commodity:
                    records.append(record)

        # Sort by transaction date
        records.sort(key=lambda r: r.transaction_date)
        return records

    # =========================================================================
    # Due Diligence Statements
    # =========================================================================

    def generate_due_diligence_statement(
        self,
        commodity: EUDRCommodity,
        product_description: str,
        cn_codes: List[str],
        quantity: Decimal,
        origin_plot_ids: List[str],
        statement_type: str = "import",
        **kwargs
    ) -> DueDiligenceStatement:
        """
        Generate a due diligence statement.

        Args:
            commodity: EUDR commodity
            product_description: Product description
            cn_codes: Combined Nomenclature codes
            quantity: Total quantity
            origin_plot_ids: Origin plot IDs
            statement_type: Statement type (import, export, domestic)
            **kwargs: Additional statement attributes

        Returns:
            Generated DueDiligenceStatement
        """
        # Gather origin information
        origin_countries: Set[str] = set()
        origin_plots_verified: List[str] = []
        total_risk_score = 0.0
        plots_compliant = True

        for plot_id in origin_plot_ids:
            plot = self._plots.get(plot_id)
            if plot:
                origin_countries.add(plot.country_code)
                origin_plots_verified.append(plot_id)
                if plot.risk_level == RiskLevel.HIGH:
                    total_risk_score += 3
                elif plot.risk_level == RiskLevel.STANDARD:
                    total_risk_score += 2
                else:
                    total_risk_score += 1

                if not plot.is_compliant:
                    plots_compliant = False

        # Determine overall risk level
        avg_risk = total_risk_score / len(origin_plots_verified) if origin_plots_verified else 3
        if avg_risk >= 2.5:
            overall_risk = RiskLevel.HIGH
        elif avg_risk >= 1.5:
            overall_risk = RiskLevel.STANDARD
        else:
            overall_risk = RiskLevel.LOW

        # Get related chain of custody records
        coc_ids = []
        for record in self._chain_of_custody.values():
            if any(pid in record.origin_plot_ids for pid in origin_plot_ids):
                coc_ids.append(record.id)

        # Risk assessment result
        risk_assessment = self._perform_risk_assessment(
            commodity, origin_countries, origin_plots_verified
        )

        # Create statement
        statement = DueDiligenceStatement.create(
            operator_id=self.operator_id,
            operator_name=self.operator_name,
            operator_country=self.operator_country,
            commodity=commodity,
            statement_type=statement_type,
            product_description=product_description,
            cn_codes=cn_codes,
            quantity=quantity,
            origin_countries=list(origin_countries),
            origin_plots=origin_plots_verified,
            chain_of_custody=coc_ids,
            risk_assessment_result=risk_assessment,
            risk_level=overall_risk,
            deforestation_free_declaration=plots_compliant,
            legal_compliance_declaration=plots_compliant,
            **kwargs
        )

        self._statements[statement.id] = statement
        logger.info(f"Generated due diligence statement: {statement.id}")
        return statement

    def get_statement(self, statement_id: str) -> Optional[DueDiligenceStatement]:
        """Get due diligence statement by ID."""
        return self._statements.get(statement_id)

    def _perform_risk_assessment(
        self,
        commodity: EUDRCommodity,
        origin_countries: Set[str],
        plot_ids: List[str],
    ) -> str:
        """
        Perform risk assessment for due diligence statement.

        Args:
            commodity: EUDR commodity
            origin_countries: Countries of origin
            plot_ids: Origin plot IDs

        Returns:
            Risk assessment summary
        """
        risks = []

        # Country risk
        high_risk_origins = origin_countries.intersection(self.HIGH_RISK_COUNTRIES)
        if high_risk_origins:
            risks.append(
                f"High-risk countries in supply chain: {', '.join(high_risk_origins)}"
            )

        # Plot compliance
        non_compliant_plots = []
        for plot_id in plot_ids:
            plot = self._plots.get(plot_id)
            if plot and not plot.is_compliant:
                non_compliant_plots.append(plot_id)

        if non_compliant_plots:
            risks.append(
                f"Non-compliant plots identified: {len(non_compliant_plots)}"
            )

        # Missing traceability
        missing_plots = set(plot_ids) - set(self._plots.keys())
        if missing_plots:
            risks.append(
                f"Plots with incomplete traceability: {len(missing_plots)}"
            )

        if not risks:
            return "No significant risks identified. All traceability requirements met."

        return "Risk factors identified: " + "; ".join(risks)

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Generate compliance summary report.

        Returns:
            Dictionary with compliance statistics
        """
        total_plots = len(self._plots)
        compliant_plots = sum(1 for p in self._plots.values() if p.is_compliant)
        high_risk_plots = sum(
            1 for p in self._plots.values()
            if p.risk_level == RiskLevel.HIGH
        )

        total_statements = len(self._statements)
        compliant_statements = sum(
            1 for s in self._statements.values()
            if s.status == ComplianceStatus.COMPLIANT
        )

        return {
            "operator_id": self.operator_id,
            "operator_name": self.operator_name,
            "report_date": datetime.utcnow().isoformat(),
            "plots": {
                "total": total_plots,
                "compliant": compliant_plots,
                "compliance_rate": compliant_plots / total_plots if total_plots else 0,
                "high_risk": high_risk_plots,
            },
            "chain_of_custody": {
                "total_records": len(self._chain_of_custody),
            },
            "due_diligence_statements": {
                "total": total_statements,
                "compliant": compliant_statements,
                "pending": total_statements - compliant_statements,
            },
            "commodities": {
                commodity.value: len(plots)
                for commodity, plots in self._plots_by_commodity.items()
            },
            "countries": {
                country: len(plots)
                for country, plots in self._plots_by_country.items()
            },
        }

    def export_for_eu_system(
        self,
        statement_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Export due diligence statement for EU Information System.

        Args:
            statement_id: Statement ID to export

        Returns:
            Formatted data for EU system submission
        """
        statement = self._statements.get(statement_id)
        if not statement:
            return None

        # Get full plot details
        plot_details = []
        for plot_id in statement.origin_plots:
            plot = self._plots.get(plot_id)
            if plot:
                plot_details.append({
                    "plot_id": plot.id,
                    "latitude": plot.latitude,
                    "longitude": plot.longitude,
                    "polygon": plot.polygon_wkt,
                    "country": plot.country_code,
                    "producer": plot.producer_name,
                    "production_date": plot.production_date.isoformat() if plot.production_date else None,
                })

        return {
            "statement_reference": statement.id,
            "operator": {
                "id": statement.operator_id,
                "name": statement.operator_name,
                "country": statement.operator_country,
            },
            "product": {
                "commodity": statement.commodity.value,
                "description": statement.product_description,
                "cn_codes": statement.cn_codes,
                "quantity": str(statement.quantity),
                "unit": statement.unit,
            },
            "traceability": {
                "origin_countries": statement.origin_countries,
                "plots": plot_details,
            },
            "declarations": {
                "deforestation_free": statement.deforestation_free_declaration,
                "legal_compliance": statement.legal_compliance_declaration,
            },
            "risk_assessment": {
                "level": statement.risk_level.value,
                "result": statement.risk_assessment_result,
                "mitigation": statement.risk_mitigation_measures,
            },
            "submission": {
                "date": statement.submission_date.isoformat() if statement.submission_date else None,
                "signature": statement.digital_signature,
            },
        }
