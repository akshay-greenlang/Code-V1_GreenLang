# -*- coding: utf-8 -*-
"""
EUDR Traceability Connector Service Data Models - AGENT-DATA-004: EUDR Connector

Pydantic v2 data models for the EUDR Traceability Connector SDK. Defines
all enumerations, core data models, and request wrappers required for
EU Deforestation Regulation (EUDR) compliance traceability.

EU Deforestation Regulation (EUDR - Regulation (EU) 2023/1115) requires
operators and traders placing specific commodities on the EU market to
demonstrate that products are deforestation-free, legally produced, and
covered by a due diligence statement.

Models:
    - Enumerations: EUDRCommodity, RiskLevel, ComplianceStatus, LandUseType,
        CustodyModel, DDSStatus, DDSType, SubmissionStatus
    - Core models: GeolocationData, PlotRecord, CustodyTransfer, BatchRecord,
        RiskScore, CommodityClassification, SupplierDeclaration,
        DueDiligenceStatement, ComplianceCheckResult, EUSubmissionRecord,
        EUDRStatistics
    - Request models: RegisterPlotRequest, RecordTransferRequest,
        GenerateDDSRequest, AssessRiskRequest, ClassifyCommodityRequest,
        RegisterDeclarationRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-004 EUDR Traceability Connector
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _today() -> date:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date()


# =============================================================================
# Enumerations
# =============================================================================


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities and their derived products.

    The EU Deforestation Regulation covers seven primary commodities
    and their key derived products as listed in Annex I of Regulation
    (EU) 2023/1115.
    """

    # Primary commodities
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"

    # Derived products - cattle
    BEEF = "beef"
    LEATHER = "leather"

    # Derived products - cocoa
    CHOCOLATE = "chocolate"

    # Derived products - oil palm
    PALM_OIL = "palm_oil"

    # Derived products - rubber
    NATURAL_RUBBER = "natural_rubber"
    TYRES = "tyres"

    # Derived products - soya
    SOYBEAN_OIL = "soybean_oil"
    SOYBEAN_MEAL = "soybean_meal"

    # Derived products - wood
    TIMBER = "timber"
    FURNITURE = "furniture"
    PAPER = "paper"
    CHARCOAL = "charcoal"


class RiskLevel(str, Enum):
    """Risk classification levels for EUDR country benchmarking.

    Based on Article 29 of Regulation (EU) 2023/1115, the Commission
    classifies countries or parts thereof into risk categories.
    """

    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class ComplianceStatus(str, Enum):
    """EUDR compliance status for plots, products, or operators.

    Reflects the outcome of compliance verification against EUDR
    requirements including deforestation-free status, legality,
    and traceability obligations.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    UNDER_REVIEW = "under_review"
    INSUFFICIENT_DATA = "insufficient_data"
    EXEMPTED = "exempted"


class LandUseType(str, Enum):
    """Land use classification for production plots.

    Used to determine the type of land on which the relevant commodity
    was produced, relevant for deforestation and forest degradation
    assessment under EUDR Article 2.
    """

    FOREST = "forest"
    PLANTATION = "plantation"
    AGRICULTURAL = "agricultural"
    PASTURE = "pasture"
    DEGRADED = "degraded"
    OTHER = "other"


class CustodyModel(str, Enum):
    """Chain of custody models for commodity traceability.

    Defines how EUDR-relevant commodities are tracked through the
    supply chain from plot of origin to final product placement.
    """

    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"


class DDSStatus(str, Enum):
    """Lifecycle status of a Due Diligence Statement (DDS).

    Tracks the DDS through its lifecycle from creation to verification
    or rejection as required by EUDR Article 4.
    """

    DRAFT = "draft"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class DDSType(str, Enum):
    """Type of Due Diligence Statement based on market activity.

    Operators must submit different DDS types depending on whether
    they are importing, exporting, or operating domestically within
    the EU, as specified in EUDR Articles 4 and 5.
    """

    IMPORT_PLACEMENT = "import_placement"
    EXPORT = "export"
    DOMESTIC = "domestic"


class SubmissionStatus(str, Enum):
    """Status of a submission to the EU Information System.

    Tracks the lifecycle of electronic submissions to the EU
    Information System as mandated by EUDR Article 33.
    """

    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ERROR = "error"


# =============================================================================
# Primary Commodity Mapping
# =============================================================================

#: Maps derived products back to their primary EUDR commodity.
DERIVED_TO_PRIMARY: Dict[EUDRCommodity, EUDRCommodity] = {
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

#: Set of the seven primary EUDR commodities.
PRIMARY_COMMODITIES = frozenset({
    EUDRCommodity.CATTLE,
    EUDRCommodity.COCOA,
    EUDRCommodity.COFFEE,
    EUDRCommodity.OIL_PALM,
    EUDRCommodity.RUBBER,
    EUDRCommodity.SOYA,
    EUDRCommodity.WOOD,
})


# =============================================================================
# Core Data Models
# =============================================================================


class GeolocationData(BaseModel):
    """Geolocation data for a production plot as required by EUDR Article 9.

    EUDR mandates geolocation of all plots of land where the relevant
    commodity was produced. For plots larger than four hectares, polygon
    coordinates are required. For smaller plots, a single GPS point
    (latitude/longitude) is sufficient.

    Attributes:
        latitude: GPS latitude in decimal degrees (WGS84).
        longitude: GPS longitude in decimal degrees (WGS84).
        polygon_coordinates: Ordered list of [longitude, latitude] pairs
            forming a closed polygon boundary. Required for plots above
            the hectare threshold.
        plot_area_hectares: Total area of the plot in hectares.
        country_code: ISO 3166-1 alpha-2 country code.
        region: Administrative region or state within the country.
        sub_region: Sub-regional administrative division.
        coordinate_precision: Number of decimal places for coordinates.
        coordinate_system: Coordinate reference system identifier.
    """

    model_config = ConfigDict(from_attributes=True)

    latitude: float = Field(
        ...,
        ge=-90.0,
        le=90.0,
        description="GPS latitude in decimal degrees (WGS84)",
    )
    longitude: float = Field(
        ...,
        ge=-180.0,
        le=180.0,
        description="GPS longitude in decimal degrees (WGS84)",
    )
    polygon_coordinates: Optional[List[List[float]]] = Field(
        None,
        description=(
            "Ordered list of [longitude, latitude] pairs forming a "
            "closed polygon boundary for the plot"
        ),
    )
    plot_area_hectares: float = Field(
        ...,
        gt=0.0,
        description="Total area of the plot in hectares",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    region: str = Field(
        default="",
        description="Administrative region or state within the country",
    )
    sub_region: str = Field(
        default="",
        description="Sub-regional administrative division",
    )
    coordinate_precision: int = Field(
        default=6,
        ge=1,
        le=10,
        description="Number of decimal places for coordinates",
    )
    coordinate_system: str = Field(
        default="WGS84",
        description="Coordinate reference system identifier",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("polygon_coordinates")
    @classmethod
    def validate_polygon_coordinates(
        cls, v: Optional[List[List[float]]],
    ) -> Optional[List[List[float]]]:
        """Validate polygon coordinate pairs are well-formed.

        Each coordinate pair must contain exactly two values:
        [longitude, latitude] with valid ranges.
        """
        if v is None:
            return v
        if len(v) < 3:
            raise ValueError(
                "polygon_coordinates must have at least 3 coordinate pairs "
                "to form a valid polygon"
            )
        for i, pair in enumerate(v):
            if len(pair) != 2:
                raise ValueError(
                    f"Coordinate pair at index {i} must have exactly "
                    f"2 values [longitude, latitude], got {len(pair)}"
                )
            lon, lat = pair
            if not (-180.0 <= lon <= 180.0):
                raise ValueError(
                    f"Longitude at index {i} must be between -180 and 180, "
                    f"got {lon}"
                )
            if not (-90.0 <= lat <= 90.0):
                raise ValueError(
                    f"Latitude at index {i} must be between -90 and 90, "
                    f"got {lat}"
                )
        return v

    @model_validator(mode="after")
    def validate_polygon_required_for_large_plots(self) -> GeolocationData:
        """Enforce polygon requirement for plots above 4 hectares.

        Per EUDR Article 9(1)(d), plots of land larger than four hectares
        must include polygon geolocation data.
        """
        if self.plot_area_hectares > 4.0 and self.polygon_coordinates is None:
            raise ValueError(
                f"polygon_coordinates are required for plots larger than "
                f"4.0 hectares (plot area: {self.plot_area_hectares} ha)"
            )
        return self


class PlotRecord(BaseModel):
    """Record of a registered production plot for EUDR traceability.

    Represents a single plot of land where an EUDR-relevant commodity
    is produced, including geolocation, compliance status, and
    deforestation-free declaration data.

    Attributes:
        plot_id: Unique identifier for this plot record.
        geolocation: Geolocation data for the production plot.
        commodity: EUDR commodity produced on this plot.
        producer_id: Identifier for the producer or farm operator.
        producer_name: Human-readable name of the producer.
        production_date: Date of most recent production activity.
        harvest_date: Date of most recent harvest.
        quantity_kg: Quantity produced in kilograms.
        unit: Unit of measurement for quantity.
        country_code: ISO 3166-1 alpha-2 country code.
        certification: Optional certification scheme identifier.
        land_use_type: Type of land use on this plot.
        deforestation_free: Whether the plot is declared deforestation-free.
        deforestation_cutoff_date: Reference date for deforestation-free status.
        legal_compliance: Whether production complies with local laws.
        supporting_documents: List of supporting document references.
        risk_level: Risk classification for this plot.
        created_at: Timestamp when the plot record was created.
        updated_at: Timestamp of the last update to this record.
        metadata: Additional metadata key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    plot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this plot record",
    )
    geolocation: GeolocationData = Field(
        ...,
        description="Geolocation data for the production plot",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity produced on this plot",
    )
    producer_id: str = Field(
        ...,
        description="Identifier for the producer or farm operator",
    )
    producer_name: str = Field(
        ...,
        description="Human-readable name of the producer",
    )
    production_date: Optional[date] = Field(
        None,
        description="Date of most recent production activity",
    )
    harvest_date: Optional[date] = Field(
        None,
        description="Date of most recent harvest",
    )
    quantity_kg: Optional[float] = Field(
        None,
        ge=0.0,
        description="Quantity produced in kilograms",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    certification: Optional[str] = Field(
        None,
        description="Optional certification scheme identifier (e.g. FSC, RSPO)",
    )
    land_use_type: LandUseType = Field(
        ...,
        description="Type of land use on this plot",
    )
    deforestation_free: bool = Field(
        ...,
        description="Whether the plot is declared deforestation-free since cutoff",
    )
    deforestation_cutoff_date: date = Field(
        default=date(2020, 12, 31),
        description="Reference date for deforestation-free status (EUDR: 2020-12-31)",
    )
    legal_compliance: bool = Field(
        ...,
        description="Whether production complies with local laws of the country",
    )
    supporting_documents: List[str] = Field(
        default_factory=list,
        description="List of supporting document references or file paths",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Risk classification for this plot",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the plot record was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the last update to this record",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata key-value pairs",
    )

    @field_validator("producer_id")
    @classmethod
    def validate_producer_id(cls, v: str) -> str:
        """Validate producer_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("producer_id must be non-empty")
        return v

    @field_validator("producer_name")
    @classmethod
    def validate_producer_name(cls, v: str) -> str:
        """Validate producer_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("producer_name must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class CustodyTransfer(BaseModel):
    """Record of a chain of custody transfer between operators.

    Represents a single transfer event in the supply chain where
    custody of an EUDR-relevant commodity or product moves from
    one operator to another, maintaining full traceability.

    Attributes:
        transfer_id: Unique identifier for this transfer record.
        transaction_id: External transaction or invoice identifier.
        source_operator_id: Identifier of the operator transferring custody.
        source_operator_name: Name of the source operator.
        target_operator_id: Identifier of the operator receiving custody.
        target_operator_name: Name of the target operator.
        commodity: EUDR commodity being transferred.
        product_description: Description of the product being transferred.
        quantity: Quantity being transferred.
        unit: Unit of measurement for quantity.
        batch_number: Optional batch or lot identifier.
        origin_plot_ids: List of origin plot IDs for traceability.
        custody_model: Chain of custody model applied.
        transaction_date: Date and time of the transfer.
        transport_mode: Mode of transport (e.g. road, sea, rail, air).
        transport_documents: List of transport document references.
        customs_declaration: Customs declaration reference number.
        cn_code: EU Combined Nomenclature code.
        hs_code: Harmonized System code.
        verification_status: Current compliance verification status.
        verified_by: Identifier of the verifier.
        verified_at: Timestamp of verification.
        created_at: Timestamp when the transfer was recorded.
        metadata: Additional metadata key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    transfer_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this transfer record",
    )
    transaction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="External transaction or invoice identifier",
    )
    source_operator_id: str = Field(
        ...,
        description="Identifier of the operator transferring custody",
    )
    source_operator_name: str = Field(
        ...,
        description="Name of the source operator",
    )
    target_operator_id: str = Field(
        ...,
        description="Identifier of the operator receiving custody",
    )
    target_operator_name: str = Field(
        ...,
        description="Name of the target operator",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity being transferred",
    )
    product_description: str = Field(
        ...,
        description="Description of the product being transferred",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity being transferred",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    batch_number: Optional[str] = Field(
        None,
        description="Optional batch or lot identifier",
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list,
        description="List of origin plot IDs for traceability",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model applied to this transfer",
    )
    transaction_date: datetime = Field(
        default_factory=_utcnow,
        description="Date and time of the transfer",
    )
    transport_mode: Optional[str] = Field(
        None,
        description="Mode of transport (e.g. road, sea, rail, air)",
    )
    transport_documents: List[str] = Field(
        default_factory=list,
        description="List of transport document references",
    )
    customs_declaration: Optional[str] = Field(
        None,
        description="Customs declaration reference number",
    )
    cn_code: Optional[str] = Field(
        None,
        description="EU Combined Nomenclature code",
    )
    hs_code: Optional[str] = Field(
        None,
        description="Harmonized System code",
    )
    verification_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING_VERIFICATION,
        description="Current compliance verification status",
    )
    verified_by: Optional[str] = Field(
        None,
        description="Identifier of the verifier",
    )
    verified_at: Optional[datetime] = Field(
        None,
        description="Timestamp of verification",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the transfer was recorded",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata key-value pairs",
    )

    @field_validator("source_operator_id")
    @classmethod
    def validate_source_operator_id(cls, v: str) -> str:
        """Validate source_operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_operator_id must be non-empty")
        return v

    @field_validator("source_operator_name")
    @classmethod
    def validate_source_operator_name(cls, v: str) -> str:
        """Validate source_operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_operator_name must be non-empty")
        return v

    @field_validator("target_operator_id")
    @classmethod
    def validate_target_operator_id(cls, v: str) -> str:
        """Validate target_operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_operator_id must be non-empty")
        return v

    @field_validator("target_operator_name")
    @classmethod
    def validate_target_operator_name(cls, v: str) -> str:
        """Validate target_operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_operator_name must be non-empty")
        return v

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v


class BatchRecord(BaseModel):
    """Record of a batch or lot of EUDR-relevant commodities.

    Represents a grouped collection of commodities from one or more
    origin plots, used for mass balance and segregated custody models.

    Attributes:
        batch_id: Unique identifier for this batch.
        parent_batch_ids: IDs of parent batches (for split/merge tracking).
        commodity: EUDR commodity in this batch.
        product_description: Description of the product in this batch.
        quantity: Total quantity in this batch.
        unit: Unit of measurement for quantity.
        origin_plot_ids: List of origin plot IDs contributing to this batch.
        custody_model: Chain of custody model applied to this batch.
        created_at: Timestamp when the batch was created.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch",
    )
    parent_batch_ids: List[str] = Field(
        default_factory=list,
        description="IDs of parent batches (for split/merge tracking)",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity in this batch",
    )
    product_description: str = Field(
        ...,
        description="Description of the product in this batch",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total quantity in this batch",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list,
        description="List of origin plot IDs contributing to this batch",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model applied to this batch",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the batch was created",
    )

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v


class RiskScore(BaseModel):
    """Risk assessment score for a plot, product, or operator.

    Implements composite risk scoring based on country, commodity,
    supplier, and traceability dimensions as required by EUDR
    Article 10 for the due diligence risk assessment step.

    Attributes:
        assessment_id: Unique identifier for this risk assessment.
        target_type: Type of entity assessed (plot, product, operator).
        target_id: Identifier of the assessed entity.
        country_risk_score: Country-level risk score (0-100).
        commodity_risk_score: Commodity-level risk score (0-100).
        supplier_risk_score: Supplier-level risk score (0-100).
        traceability_risk_score: Traceability completeness score (0-100).
        overall_risk_score: Weighted composite risk score (0-100).
        risk_level: Resulting risk classification.
        risk_factors: List of identified risk factors.
        mitigation_measures: List of recommended mitigation measures.
        methodology: Risk assessment methodology identifier.
        assessed_at: Timestamp of the assessment.
    """

    model_config = ConfigDict(from_attributes=True)

    assessment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this risk assessment",
    )
    target_type: str = Field(
        ...,
        description="Type of entity assessed: 'plot', 'product', or 'operator'",
    )
    target_id: str = Field(
        ...,
        description="Identifier of the assessed entity",
    )
    country_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Country-level risk score (0-100)",
    )
    commodity_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Commodity-level risk score (0-100)",
    )
    supplier_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Supplier-level risk score (0-100)",
    )
    traceability_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Traceability completeness risk score (0-100)",
    )
    overall_risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Weighted composite risk score (0-100)",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Resulting risk classification based on overall score",
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="List of identified risk factors",
    )
    mitigation_measures: List[str] = Field(
        default_factory=list,
        description="List of recommended risk mitigation measures",
    )
    methodology: str = Field(
        default="EUDR_Standard_Risk_Assessment_v1",
        description="Risk assessment methodology identifier",
    )
    assessed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the assessment was performed",
    )

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        """Validate target_type is one of the allowed values."""
        allowed = {"plot", "product", "operator"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"target_type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        """Validate target_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_id must be non-empty")
        return v


class CommodityClassification(BaseModel):
    """Classification of a product against the EUDR commodity list.

    Maps a product to its EUDR commodity category and identifies
    its CN (Combined Nomenclature) and HS (Harmonized System) codes
    for customs and regulatory reporting.

    Attributes:
        classification_id: Unique identifier for this classification.
        product_name: Name of the product being classified.
        commodity: Matched EUDR commodity category.
        cn_code: EU Combined Nomenclature code.
        hs_code: Harmonized System code.
        is_derived_product: Whether the product is derived from a primary commodity.
        primary_commodity: Primary commodity if this is a derived product.
        product_composition: Composition percentages by commodity (for blends).
        classified_at: Timestamp when the classification was performed.
    """

    model_config = ConfigDict(from_attributes=True)

    classification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this classification",
    )
    product_name: str = Field(
        ...,
        description="Name of the product being classified",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="Matched EUDR commodity category",
    )
    cn_code: str = Field(
        ...,
        description="EU Combined Nomenclature code",
    )
    hs_code: str = Field(
        ...,
        description="Harmonized System code",
    )
    is_derived_product: bool = Field(
        default=False,
        description="Whether the product is derived from a primary commodity",
    )
    primary_commodity: Optional[EUDRCommodity] = Field(
        None,
        description="Primary commodity if this is a derived product",
    )
    product_composition: Optional[Dict[str, float]] = Field(
        None,
        description="Composition percentages by commodity (for blends)",
    )
    classified_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the classification was performed",
    )

    @field_validator("product_name")
    @classmethod
    def validate_product_name(cls, v: str) -> str:
        """Validate product_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_name must be non-empty")
        return v

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code(cls, v: str) -> str:
        """Validate cn_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("cn_code must be non-empty")
        return v

    @field_validator("hs_code")
    @classmethod
    def validate_hs_code(cls, v: str) -> str:
        """Validate hs_code is non-empty."""
        if not v or not v.strip():
            raise ValueError("hs_code must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_derived_product_has_primary(self) -> CommodityClassification:
        """Ensure derived products reference their primary commodity."""
        if self.is_derived_product and self.primary_commodity is None:
            raise ValueError(
                "primary_commodity is required when is_derived_product is True"
            )
        return self


class SupplierDeclaration(BaseModel):
    """Declaration from a supplier regarding EUDR compliance.

    Captures supplier-level attestations about deforestation-free
    sourcing, legal production, and traceability as supporting
    evidence for the operator's due diligence process.

    Attributes:
        declaration_id: Unique identifier for this declaration.
        supplier_id: Identifier for the declaring supplier.
        supplier_name: Human-readable name of the supplier.
        supplier_country: Country where the supplier is based (ISO alpha-2).
        declaration_date: Date the declaration was made.
        commodities_covered: List of EUDR commodities covered by this declaration.
        confirms_deforestation_free: Whether supplier confirms deforestation-free.
        confirms_legal_production: Whether supplier confirms legal production.
        confirms_traceability: Whether supplier confirms traceability to plot.
        valid_from: Start date of the declaration validity.
        valid_until: Optional end date of the declaration validity.
        documentation_provided: List of supporting document references.
        signatory_name: Name of the person who signed the declaration.
    """

    model_config = ConfigDict(from_attributes=True)

    declaration_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this declaration",
    )
    supplier_id: str = Field(
        ...,
        description="Identifier for the declaring supplier",
    )
    supplier_name: str = Field(
        ...,
        description="Human-readable name of the supplier",
    )
    supplier_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country where the supplier is based (ISO 3166-1 alpha-2)",
    )
    declaration_date: date = Field(
        default_factory=_today,
        description="Date the declaration was made",
    )
    commodities_covered: List[EUDRCommodity] = Field(
        ...,
        min_length=1,
        description="List of EUDR commodities covered by this declaration",
    )
    confirms_deforestation_free: bool = Field(
        ...,
        description="Whether supplier confirms deforestation-free sourcing",
    )
    confirms_legal_production: bool = Field(
        ...,
        description="Whether supplier confirms compliance with local laws",
    )
    confirms_traceability: bool = Field(
        ...,
        description="Whether supplier confirms traceability to plot of production",
    )
    valid_from: date = Field(
        default_factory=_today,
        description="Start date of the declaration validity period",
    )
    valid_until: Optional[date] = Field(
        None,
        description="Optional end date of the declaration validity period",
    )
    documentation_provided: List[str] = Field(
        default_factory=list,
        description="List of supporting document references",
    )
    signatory_name: Optional[str] = Field(
        None,
        description="Name of the person who signed the declaration",
    )

    @field_validator("supplier_id")
    @classmethod
    def validate_supplier_id(cls, v: str) -> str:
        """Validate supplier_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_id must be non-empty")
        return v

    @field_validator("supplier_name")
    @classmethod
    def validate_supplier_name(cls, v: str) -> str:
        """Validate supplier_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_name must be non-empty")
        return v

    @field_validator("supplier_country")
    @classmethod
    def validate_supplier_country(cls, v: str) -> str:
        """Validate supplier country is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "supplier_country must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @model_validator(mode="after")
    def validate_validity_dates(self) -> SupplierDeclaration:
        """Ensure valid_until is after valid_from if both are set."""
        if self.valid_until is not None and self.valid_until < self.valid_from:
            raise ValueError(
                f"valid_until ({self.valid_until}) must be on or after "
                f"valid_from ({self.valid_from})"
            )
        return self


class DueDiligenceStatement(BaseModel):
    """Due Diligence Statement (DDS) as required by EUDR Article 4.

    Represents the formal due diligence statement that operators and
    traders must submit to the EU Information System before placing
    relevant commodities on the EU market or exporting them.

    Attributes:
        dds_id: Unique identifier for this due diligence statement.
        operator_id: Identifier of the operator filing the DDS.
        operator_name: Legal name of the operator.
        operator_country: Country where the operator is established.
        operator_eori: EORI number of the operator (for customs).
        dds_type: Type of DDS based on market activity.
        commodity: EUDR commodity covered by this DDS.
        product_description: Description of the products covered.
        cn_codes: List of CN codes for the products.
        quantity: Total quantity covered by this DDS.
        unit: Unit of measurement for quantity.
        origin_countries: List of countries of origin.
        origin_plot_ids: List of origin plot IDs for traceability.
        custody_transfer_ids: List of custody transfer IDs in the chain.
        risk_assessment_id: Identifier of the associated risk assessment.
        risk_level: Risk level determined by the risk assessment.
        deforestation_free_declaration: Operator declares deforestation-free.
        legal_compliance_declaration: Operator declares legal compliance.
        risk_mitigation_measures: Measures taken to mitigate identified risks.
        supporting_evidence: List of supporting evidence references.
        status: Current lifecycle status of the DDS.
        eu_reference_number: Reference number assigned by the EU System.
        digital_signature: Digital signature for authenticity verification.
        submission_date: Date when the DDS was submitted to the EU System.
        validity_start: Start date of the DDS validity period.
        validity_end: End date of the DDS validity period.
        created_at: Timestamp when the DDS was created.
        updated_at: Timestamp of the last update to this DDS.
    """

    model_config = ConfigDict(from_attributes=True)

    dds_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this due diligence statement",
    )
    operator_id: str = Field(
        ...,
        description="Identifier of the operator filing the DDS",
    )
    operator_name: str = Field(
        ...,
        description="Legal name of the operator",
    )
    operator_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country where the operator is established (ISO alpha-2)",
    )
    operator_eori: Optional[str] = Field(
        None,
        description="EORI number of the operator for customs identification",
    )
    dds_type: DDSType = Field(
        default=DDSType.IMPORT_PLACEMENT,
        description="Type of DDS based on market activity",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity covered by this DDS",
    )
    product_description: str = Field(
        ...,
        description="Description of the products covered by this DDS",
    )
    cn_codes: List[str] = Field(
        default_factory=list,
        description="List of EU Combined Nomenclature codes for the products",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total quantity covered by this DDS",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement for quantity",
    )
    origin_countries: List[str] = Field(
        default_factory=list,
        description="List of ISO alpha-2 country codes of origin",
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list,
        description="List of origin plot IDs for traceability",
    )
    custody_transfer_ids: List[str] = Field(
        default_factory=list,
        description="List of custody transfer IDs in the supply chain",
    )
    risk_assessment_id: Optional[str] = Field(
        None,
        description="Identifier of the associated risk assessment",
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD,
        description="Risk level determined by the risk assessment",
    )
    deforestation_free_declaration: bool = Field(
        ...,
        description="Operator declares products are deforestation-free",
    )
    legal_compliance_declaration: bool = Field(
        ...,
        description="Operator declares products comply with local laws",
    )
    risk_mitigation_measures: List[str] = Field(
        default_factory=list,
        description="Measures taken to mitigate identified risks",
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="List of supporting evidence document references",
    )
    status: DDSStatus = Field(
        default=DDSStatus.DRAFT,
        description="Current lifecycle status of the DDS",
    )
    eu_reference_number: Optional[str] = Field(
        None,
        description="Reference number assigned by the EU Information System",
    )
    digital_signature: Optional[str] = Field(
        None,
        description="Digital signature for authenticity verification",
    )
    submission_date: Optional[datetime] = Field(
        None,
        description="Date and time when the DDS was submitted to the EU System",
    )
    validity_start: Optional[date] = Field(
        None,
        description="Start date of the DDS validity period",
    )
    validity_end: Optional[date] = Field(
        None,
        description="End date of the DDS validity period",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the DDS was created",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp of the last update to this DDS",
    )

    @field_validator("operator_id")
    @classmethod
    def validate_operator_id(cls, v: str) -> str:
        """Validate operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_id must be non-empty")
        return v

    @field_validator("operator_name")
    @classmethod
    def validate_operator_name(cls, v: str) -> str:
        """Validate operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("operator_name must be non-empty")
        return v

    @field_validator("operator_country")
    @classmethod
    def validate_operator_country(cls, v: str) -> str:
        """Validate operator country is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "operator_country must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_validity_period(self) -> DueDiligenceStatement:
        """Ensure validity_end is after validity_start if both are set."""
        if (
            self.validity_start is not None
            and self.validity_end is not None
            and self.validity_end < self.validity_start
        ):
            raise ValueError(
                f"validity_end ({self.validity_end}) must be on or after "
                f"validity_start ({self.validity_start})"
            )
        return self


class ComplianceCheckResult(BaseModel):
    """Result of a compliance check against a specific EUDR article.

    Represents the outcome of verifying a plot, product, or operator
    against a specific requirement of the EU Deforestation Regulation.

    Attributes:
        check_id: Unique identifier for this compliance check.
        target_type: Type of entity checked (plot, product, operator, dds).
        target_id: Identifier of the checked entity.
        article_checked: EUDR article number being checked.
        requirement: Description of the requirement being verified.
        is_compliant: Whether the target meets the requirement.
        details: Detailed explanation of the check outcome.
        remediation: Suggested remediation action if non-compliant.
        checked_at: Timestamp when the check was performed.
    """

    model_config = ConfigDict(from_attributes=True)

    check_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this compliance check",
    )
    target_type: str = Field(
        ...,
        description="Type of entity checked: 'plot', 'product', 'operator', or 'dds'",
    )
    target_id: str = Field(
        ...,
        description="Identifier of the checked entity",
    )
    article_checked: str = Field(
        ...,
        description="EUDR article number being checked (e.g. 'Article 3')",
    )
    requirement: str = Field(
        ...,
        description="Description of the EUDR requirement being verified",
    )
    is_compliant: bool = Field(
        ...,
        description="Whether the target meets the requirement",
    )
    details: str = Field(
        ...,
        description="Detailed explanation of the check outcome",
    )
    remediation: Optional[str] = Field(
        None,
        description="Suggested remediation action if non-compliant",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the compliance check was performed",
    )

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        """Validate target_type is one of the allowed values."""
        allowed = {"plot", "product", "operator", "dds"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"target_type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        """Validate target_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_id must be non-empty")
        return v

    @field_validator("article_checked")
    @classmethod
    def validate_article_checked(cls, v: str) -> str:
        """Validate article_checked is non-empty."""
        if not v or not v.strip():
            raise ValueError("article_checked must be non-empty")
        return v

    @field_validator("requirement")
    @classmethod
    def validate_requirement(cls, v: str) -> str:
        """Validate requirement is non-empty."""
        if not v or not v.strip():
            raise ValueError("requirement must be non-empty")
        return v

    @field_validator("details")
    @classmethod
    def validate_details(cls, v: str) -> str:
        """Validate details is non-empty."""
        if not v or not v.strip():
            raise ValueError("details must be non-empty")
        return v


class EUSubmissionRecord(BaseModel):
    """Record of a submission to the EU Information System.

    Tracks the lifecycle and status of electronic submissions to the
    EU Information System as mandated by EUDR Article 33.

    Attributes:
        submission_id: Unique identifier for this submission.
        dds_id: Associated due diligence statement identifier.
        submission_status: Current status of the submission.
        eu_reference: Reference number assigned by the EU System.
        submitted_at: Timestamp when the submission was sent.
        response_at: Timestamp when the EU System responded.
        error_message: Error message if the submission failed.
        retry_count: Number of retry attempts made.
    """

    model_config = ConfigDict(from_attributes=True)

    submission_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this submission record",
    )
    dds_id: str = Field(
        ...,
        description="Associated due diligence statement identifier",
    )
    submission_status: SubmissionStatus = Field(
        default=SubmissionStatus.PENDING,
        description="Current status of the EU Information System submission",
    )
    eu_reference: Optional[str] = Field(
        None,
        description="Reference number assigned by the EU Information System",
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the submission was sent to the EU System",
    )
    response_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the EU System responded",
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if the submission failed",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        description="Number of retry attempts made for this submission",
    )

    @field_validator("dds_id")
    @classmethod
    def validate_dds_id(cls, v: str) -> str:
        """Validate dds_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dds_id must be non-empty")
        return v


class EUDRStatistics(BaseModel):
    """Aggregated statistics for the EUDR traceability service.

    Provides high-level operational metrics for monitoring the overall
    health and activity of the EUDR traceability connector.

    Attributes:
        total_plots: Total number of registered production plots.
        compliant_plots: Number of plots with compliant status.
        total_transfers: Total number of recorded custody transfers.
        total_dds: Total number of due diligence statements created.
        submitted_dds: Number of DDS submitted to the EU System.
        pending_verifications: Number of items pending verification.
        high_risk_plots: Number of plots classified as high risk.
        commodities_breakdown: Count of plots by EUDR commodity.
        countries_breakdown: Count of plots by country of origin.
    """

    model_config = ConfigDict(from_attributes=True)

    total_plots: int = Field(
        default=0,
        ge=0,
        description="Total number of registered production plots",
    )
    compliant_plots: int = Field(
        default=0,
        ge=0,
        description="Number of plots with compliant status",
    )
    total_transfers: int = Field(
        default=0,
        ge=0,
        description="Total number of recorded custody transfers",
    )
    total_dds: int = Field(
        default=0,
        ge=0,
        description="Total number of due diligence statements created",
    )
    submitted_dds: int = Field(
        default=0,
        ge=0,
        description="Number of DDS submitted to the EU Information System",
    )
    pending_verifications: int = Field(
        default=0,
        ge=0,
        description="Number of items pending compliance verification",
    )
    high_risk_plots: int = Field(
        default=0,
        ge=0,
        description="Number of plots classified as high risk",
    )
    commodities_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of plots by EUDR commodity",
    )
    countries_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of plots by country of origin",
    )


# =============================================================================
# Request Models
# =============================================================================


class RegisterPlotRequest(BaseModel):
    """Request body for registering a new production plot.

    Attributes:
        geolocation: Geolocation data for the production plot.
        commodity: EUDR commodity produced on this plot.
        producer_id: Identifier for the producer or farm operator.
        producer_name: Human-readable name of the producer.
        country_code: ISO 3166-1 alpha-2 country code.
        certification: Optional certification scheme identifier.
        land_use_type: Type of land use on this plot.
        supporting_documents: List of supporting document references.
    """

    model_config = ConfigDict(extra="forbid")

    geolocation: GeolocationData = Field(
        ...,
        description="Geolocation data for the production plot",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity produced on this plot",
    )
    producer_id: str = Field(
        ...,
        description="Identifier for the producer or farm operator",
    )
    producer_name: str = Field(
        ...,
        description="Human-readable name of the producer",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    certification: Optional[str] = Field(
        None,
        description="Optional certification scheme identifier (e.g. FSC, RSPO)",
    )
    land_use_type: LandUseType = Field(
        ...,
        description="Type of land use on this plot",
    )
    supporting_documents: List[str] = Field(
        default_factory=list,
        description="List of supporting document references or file paths",
    )

    @field_validator("producer_id")
    @classmethod
    def validate_producer_id(cls, v: str) -> str:
        """Validate producer_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("producer_id must be non-empty")
        return v

    @field_validator("producer_name")
    @classmethod
    def validate_producer_name(cls, v: str) -> str:
        """Validate producer_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("producer_name must be non-empty")
        return v

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_code must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class RecordTransferRequest(BaseModel):
    """Request body for recording a chain of custody transfer.

    Attributes:
        source_operator_id: Identifier of the operator transferring custody.
        source_operator_name: Name of the source operator.
        target_operator_id: Identifier of the operator receiving custody.
        target_operator_name: Name of the target operator.
        commodity: EUDR commodity being transferred.
        product_description: Description of the product being transferred.
        quantity: Quantity being transferred.
        origin_plot_ids: List of origin plot IDs for traceability.
        custody_model: Chain of custody model to apply.
        batch_number: Optional batch or lot identifier.
        transport_mode: Mode of transport.
        cn_code: Optional EU Combined Nomenclature code.
        hs_code: Optional Harmonized System code.
    """

    model_config = ConfigDict(extra="forbid")

    source_operator_id: str = Field(
        ...,
        description="Identifier of the operator transferring custody",
    )
    source_operator_name: str = Field(
        ...,
        description="Name of the source operator",
    )
    target_operator_id: str = Field(
        ...,
        description="Identifier of the operator receiving custody",
    )
    target_operator_name: str = Field(
        ...,
        description="Name of the target operator",
    )
    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity being transferred",
    )
    product_description: str = Field(
        ...,
        description="Description of the product being transferred",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Quantity being transferred in the specified unit",
    )
    origin_plot_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of origin plot IDs for traceability (at least one)",
    )
    custody_model: CustodyModel = Field(
        default=CustodyModel.SEGREGATED,
        description="Chain of custody model to apply",
    )
    batch_number: Optional[str] = Field(
        None,
        description="Optional batch or lot identifier",
    )
    transport_mode: Optional[str] = Field(
        None,
        description="Mode of transport (e.g. road, sea, rail, air)",
    )
    cn_code: Optional[str] = Field(
        None,
        description="Optional EU Combined Nomenclature code",
    )
    hs_code: Optional[str] = Field(
        None,
        description="Optional Harmonized System code",
    )

    @field_validator("source_operator_id")
    @classmethod
    def validate_source_operator_id(cls, v: str) -> str:
        """Validate source_operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_operator_id must be non-empty")
        return v

    @field_validator("source_operator_name")
    @classmethod
    def validate_source_operator_name(cls, v: str) -> str:
        """Validate source_operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_operator_name must be non-empty")
        return v

    @field_validator("target_operator_id")
    @classmethod
    def validate_target_operator_id(cls, v: str) -> str:
        """Validate target_operator_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_operator_id must be non-empty")
        return v

    @field_validator("target_operator_name")
    @classmethod
    def validate_target_operator_name(cls, v: str) -> str:
        """Validate target_operator_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_operator_name must be non-empty")
        return v

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v


class GenerateDDSRequest(BaseModel):
    """Request body for generating a due diligence statement.

    Attributes:
        commodity: EUDR commodity covered by this DDS.
        product_description: Description of the products covered.
        cn_codes: List of CN codes for the products.
        quantity: Total quantity covered by this DDS.
        origin_plot_ids: List of origin plot IDs for traceability.
        dds_type: Type of DDS based on market activity.
    """

    model_config = ConfigDict(extra="forbid")

    commodity: EUDRCommodity = Field(
        ...,
        description="EUDR commodity covered by this DDS",
    )
    product_description: str = Field(
        ...,
        description="Description of the products covered by this DDS",
    )
    cn_codes: List[str] = Field(
        ...,
        min_length=1,
        description="List of EU Combined Nomenclature codes (at least one)",
    )
    quantity: Decimal = Field(
        ...,
        gt=Decimal("0"),
        description="Total quantity covered by this DDS",
    )
    origin_plot_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of origin plot IDs for traceability (at least one)",
    )
    dds_type: DDSType = Field(
        default=DDSType.IMPORT_PLACEMENT,
        description="Type of DDS based on market activity",
    )

    @field_validator("product_description")
    @classmethod
    def validate_product_description(cls, v: str) -> str:
        """Validate product_description is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_description must be non-empty")
        return v


class AssessRiskRequest(BaseModel):
    """Request body for performing a risk assessment.

    Attributes:
        target_type: Type of entity to assess (plot, product, operator).
        target_id: Identifier of the entity to assess.
        commodity: Optional commodity to consider in the assessment.
        country_codes: Optional list of countries to consider.
    """

    model_config = ConfigDict(extra="forbid")

    target_type: str = Field(
        ...,
        description="Type of entity to assess: 'plot', 'product', or 'operator'",
    )
    target_id: str = Field(
        ...,
        description="Identifier of the entity to assess",
    )
    commodity: Optional[EUDRCommodity] = Field(
        None,
        description="Optional EUDR commodity to consider in the assessment",
    )
    country_codes: Optional[List[str]] = Field(
        None,
        description="Optional list of ISO alpha-2 country codes to consider",
    )

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str) -> str:
        """Validate target_type is one of the allowed values."""
        allowed = {"plot", "product", "operator"}
        v_lower = v.strip().lower()
        if v_lower not in allowed:
            raise ValueError(
                f"target_type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v_lower

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, v: str) -> str:
        """Validate target_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("target_id must be non-empty")
        return v

    @field_validator("country_codes")
    @classmethod
    def validate_country_codes(
        cls, v: Optional[List[str]],
    ) -> Optional[List[str]]:
        """Validate country codes are uppercase two-letter ISO codes."""
        if v is None:
            return v
        validated = []
        for code in v:
            code = code.upper().strip()
            if len(code) != 2 or not code.isalpha():
                raise ValueError(
                    f"Each country code must be a two-letter ISO 3166-1 "
                    f"alpha-2 code, got '{code}'"
                )
            validated.append(code)
        return validated


class ClassifyCommodityRequest(BaseModel):
    """Request body for classifying a product against EUDR commodity list.

    Attributes:
        product_name: Name of the product to classify.
        hs_code: Optional Harmonized System code.
        cn_code: Optional EU Combined Nomenclature code.
    """

    model_config = ConfigDict(extra="forbid")

    product_name: str = Field(
        ...,
        description="Name of the product to classify",
    )
    hs_code: Optional[str] = Field(
        None,
        description="Optional Harmonized System code for code-based classification",
    )
    cn_code: Optional[str] = Field(
        None,
        description="Optional EU Combined Nomenclature code for classification",
    )

    @field_validator("product_name")
    @classmethod
    def validate_product_name(cls, v: str) -> str:
        """Validate product_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("product_name must be non-empty")
        return v


class RegisterDeclarationRequest(BaseModel):
    """Request body for registering a supplier declaration.

    Attributes:
        supplier_id: Identifier for the declaring supplier.
        supplier_name: Human-readable name of the supplier.
        supplier_country: Country where the supplier is based (ISO alpha-2).
        commodities_covered: List of EUDR commodities covered.
        confirms_deforestation_free: Whether supplier confirms deforestation-free.
        confirms_legal_production: Whether supplier confirms legal production.
        confirms_traceability: Whether supplier confirms traceability.
        valid_until: Optional end date of declaration validity.
    """

    model_config = ConfigDict(extra="forbid")

    supplier_id: str = Field(
        ...,
        description="Identifier for the declaring supplier",
    )
    supplier_name: str = Field(
        ...,
        description="Human-readable name of the supplier",
    )
    supplier_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country where the supplier is based (ISO 3166-1 alpha-2)",
    )
    commodities_covered: List[EUDRCommodity] = Field(
        ...,
        min_length=1,
        description="List of EUDR commodities covered (at least one)",
    )
    confirms_deforestation_free: bool = Field(
        ...,
        description="Whether supplier confirms deforestation-free sourcing",
    )
    confirms_legal_production: bool = Field(
        ...,
        description="Whether supplier confirms compliance with local laws",
    )
    confirms_traceability: bool = Field(
        ...,
        description="Whether supplier confirms traceability to plot of production",
    )
    valid_until: Optional[date] = Field(
        None,
        description="Optional end date of the declaration validity period",
    )

    @field_validator("supplier_id")
    @classmethod
    def validate_supplier_id(cls, v: str) -> str:
        """Validate supplier_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_id must be non-empty")
        return v

    @field_validator("supplier_name")
    @classmethod
    def validate_supplier_name(cls, v: str) -> str:
        """Validate supplier_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("supplier_name must be non-empty")
        return v

    @field_validator("supplier_country")
    @classmethod
    def validate_supplier_country(cls, v: str) -> str:
        """Validate supplier country is uppercase two-letter ISO code."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "supplier_country must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


__all__ = [
    # Enumerations
    "EUDRCommodity",
    "RiskLevel",
    "ComplianceStatus",
    "LandUseType",
    "CustodyModel",
    "DDSStatus",
    "DDSType",
    "SubmissionStatus",
    # Constants
    "DERIVED_TO_PRIMARY",
    "PRIMARY_COMMODITIES",
    # Core data models
    "GeolocationData",
    "PlotRecord",
    "CustodyTransfer",
    "BatchRecord",
    "RiskScore",
    "CommodityClassification",
    "SupplierDeclaration",
    "DueDiligenceStatement",
    "ComplianceCheckResult",
    "EUSubmissionRecord",
    "EUDRStatistics",
    # Request models
    "RegisterPlotRequest",
    "RecordTransferRequest",
    "GenerateDDSRequest",
    "AssessRiskRequest",
    "ClassifyCommodityRequest",
    "RegisterDeclarationRequest",
]
