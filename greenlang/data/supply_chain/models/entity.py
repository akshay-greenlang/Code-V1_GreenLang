"""
Supply Chain Entity Models.

This module defines comprehensive data models for multi-tier supply chain mapping,
supporting Scope 3 emissions tracking and EUDR compliance requirements.

Models:
- Supplier: Company entity with tier classification and external identifiers
- Facility: Physical site with geolocation for traceability
- Product: Finished goods with CN codes
- Material: Raw materials and components with commodity classification
- SupplierRelationship: Typed connections between suppliers

Example:
    >>> from greenlang.supply_chain.models import Supplier, Facility, SupplierTier
    >>> supplier = Supplier(
    ...     id="SUP001",
    ...     name="Acme Manufacturing Ltd",
    ...     tier=SupplierTier.TIER_1,
    ...     country_code="DE"
    ... )
    >>> facility = Facility(
    ...     id="FAC001",
    ...     name="Acme Berlin Plant",
    ...     supplier_id="SUP001",
    ...     location=GeoLocation(latitude=52.5200, longitude=13.4050)
    ... )
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Set


class SupplierTier(Enum):
    """
    Supply chain tier classification.

    Tier 1: Direct suppliers with contractual relationships
    Tier 2: Suppliers to Tier 1 suppliers
    Tier 3: Suppliers to Tier 2 suppliers
    Tier N: Extended supply chain beyond Tier 3
    """
    TIER_1 = 1
    TIER_2 = 2
    TIER_3 = 3
    TIER_N = 99  # Extended tiers beyond 3
    UNKNOWN = 0

    @classmethod
    def from_int(cls, value: int) -> "SupplierTier":
        """Create tier from integer value."""
        if value == 1:
            return cls.TIER_1
        elif value == 2:
            return cls.TIER_2
        elif value == 3:
            return cls.TIER_3
        elif value > 3:
            return cls.TIER_N
        return cls.UNKNOWN


class RelationshipType(Enum):
    """
    Types of supplier relationships in the supply chain.

    These relationship types support different supply chain mapping scenarios
    and are critical for EUDR chain of custody tracking.
    """
    SUPPLIER = "supplier"  # Direct material/component supplier
    MANUFACTURER = "manufacturer"  # Contract manufacturer
    DISTRIBUTOR = "distributor"  # Distribution/logistics partner
    PROCESSOR = "processor"  # Processing/transformation
    TRADER = "trader"  # Trading intermediary
    TRANSPORTER = "transporter"  # Logistics/transport
    STORAGE = "storage"  # Warehousing/storage
    AGENT = "agent"  # Buying agent/broker
    SUBCONTRACTOR = "subcontractor"  # Outsourced production
    RAW_MATERIAL = "raw_material"  # Raw material origin

    @classmethod
    def from_string(cls, value: str) -> "RelationshipType":
        """Create relationship type from string."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.SUPPLIER  # Default fallback


class SupplierStatus(Enum):
    """Supplier qualification and activity status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING_QUALIFICATION = "pending_qualification"
    QUALIFIED = "qualified"
    SUSPENDED = "suspended"
    BLOCKED = "blocked"
    ARCHIVED = "archived"


class CommodityType(Enum):
    """
    EUDR regulated commodity types.

    These are the seven commodities covered by the EU Deforestation Regulation.
    """
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"
    # Derived products
    LEATHER = "leather"
    CHOCOLATE = "chocolate"
    PALM_OIL = "palm_oil"
    FURNITURE = "furniture"
    PAPER = "paper"
    OTHER = "other"


@dataclass
class GeoLocation:
    """
    Geographic coordinates for facility and plot-level traceability.

    Supports both point locations and polygon boundaries for EUDR compliance.

    Attributes:
        latitude: Latitude in decimal degrees (-90 to 90)
        longitude: Longitude in decimal degrees (-180 to 180)
        altitude_m: Altitude in meters above sea level (optional)
        accuracy_m: GPS accuracy in meters (optional)
        polygon_wkt: WKT representation for polygon boundaries (optional)
    """
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None
    accuracy_m: Optional[float] = None
    polygon_wkt: Optional[str] = None  # Well-Known Text for polygon boundaries

    def __post_init__(self):
        """Validate coordinate ranges."""
        if not -90 <= self.latitude <= 90:
            raise ValueError(f"Latitude must be between -90 and 90: {self.latitude}")
        if not -180 <= self.longitude <= 180:
            raise ValueError(f"Longitude must be between -180 and 180: {self.longitude}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "latitude": self.latitude,
            "longitude": self.longitude,
        }
        if self.altitude_m is not None:
            result["altitude_m"] = self.altitude_m
        if self.accuracy_m is not None:
            result["accuracy_m"] = self.accuracy_m
        if self.polygon_wkt is not None:
            result["polygon_wkt"] = self.polygon_wkt
        return result

    def distance_km(self, other: "GeoLocation") -> float:
        """
        Calculate great-circle distance to another location using Haversine formula.

        Args:
            other: Another GeoLocation instance

        Returns:
            Distance in kilometers
        """
        import math

        R = 6371  # Earth's radius in km

        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c


@dataclass
class Address:
    """
    Standardized address model for supplier and facility locations.

    Supports international address formats with normalization for matching.

    Attributes:
        street_line_1: Primary street address
        street_line_2: Secondary address line (suite, building, etc.)
        city: City or municipality
        state_province: State, province, or region
        postal_code: ZIP or postal code
        country_code: ISO 3166-1 alpha-2 country code
        country_name: Full country name
    """
    street_line_1: str
    city: str
    country_code: str
    street_line_2: Optional[str] = None
    state_province: Optional[str] = None
    postal_code: Optional[str] = None
    country_name: Optional[str] = None

    def normalized(self) -> str:
        """
        Get normalized address string for matching.

        Returns:
            Lowercase, normalized address string
        """
        parts = [
            self.street_line_1,
            self.street_line_2,
            self.city,
            self.state_province,
            self.postal_code,
            self.country_code,
        ]
        return " ".join(p.lower().strip() for p in parts if p)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "street_line_1": self.street_line_1,
            "street_line_2": self.street_line_2,
            "city": self.city,
            "state_province": self.state_province,
            "postal_code": self.postal_code,
            "country_code": self.country_code,
            "country_name": self.country_name,
        }


@dataclass
class ExternalIdentifiers:
    """
    External identifier codes for entity resolution.

    Supports multiple identifier schemes for accurate supplier matching.

    Attributes:
        lei: Legal Entity Identifier (20 characters, ISO 17442)
        duns: D-U-N-S Number (9 digits)
        vat_number: VAT/Tax registration number
        company_registry_id: National company registry ID
        sap_vendor_id: SAP vendor master ID
        oracle_supplier_id: Oracle supplier ID
        ariba_network_id: SAP Ariba Network ID (AN)
        coupa_supplier_id: Coupa supplier ID
        custom_ids: Dictionary for additional custom identifiers
    """
    lei: Optional[str] = None  # Legal Entity Identifier
    duns: Optional[str] = None  # D-U-N-S Number
    vat_number: Optional[str] = None
    company_registry_id: Optional[str] = None
    sap_vendor_id: Optional[str] = None
    oracle_supplier_id: Optional[str] = None
    ariba_network_id: Optional[str] = None
    coupa_supplier_id: Optional[str] = None
    custom_ids: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate identifier formats."""
        if self.lei and len(self.lei) != 20:
            raise ValueError(f"LEI must be 20 characters: {self.lei}")
        if self.duns and (len(self.duns) != 9 or not self.duns.isdigit()):
            raise ValueError(f"DUNS must be 9 digits: {self.duns}")

    def has_any_identifier(self) -> bool:
        """Check if any external identifier is present."""
        return any([
            self.lei,
            self.duns,
            self.vat_number,
            self.company_registry_id,
            self.sap_vendor_id,
            self.oracle_supplier_id,
            self.ariba_network_id,
            self.coupa_supplier_id,
            bool(self.custom_ids),
        ])

    def get_primary_identifier(self) -> Optional[tuple[str, str]]:
        """
        Get the most authoritative identifier.

        Returns:
            Tuple of (identifier_type, identifier_value) or None
        """
        # Priority order: LEI > DUNS > VAT > Company Registry
        if self.lei:
            return ("lei", self.lei)
        if self.duns:
            return ("duns", self.duns)
        if self.vat_number:
            return ("vat_number", self.vat_number)
        if self.company_registry_id:
            return ("company_registry_id", self.company_registry_id)
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        if self.lei:
            result["lei"] = self.lei
        if self.duns:
            result["duns"] = self.duns
        if self.vat_number:
            result["vat_number"] = self.vat_number
        if self.company_registry_id:
            result["company_registry_id"] = self.company_registry_id
        if self.sap_vendor_id:
            result["sap_vendor_id"] = self.sap_vendor_id
        if self.oracle_supplier_id:
            result["oracle_supplier_id"] = self.oracle_supplier_id
        if self.ariba_network_id:
            result["ariba_network_id"] = self.ariba_network_id
        if self.coupa_supplier_id:
            result["coupa_supplier_id"] = self.coupa_supplier_id
        if self.custom_ids:
            result["custom_ids"] = self.custom_ids
        return result


@dataclass
class ContactInfo:
    """Contact information for supplier communications."""
    primary_contact_name: Optional[str] = None
    primary_email: Optional[str] = None
    primary_phone: Optional[str] = None
    sustainability_contact_name: Optional[str] = None
    sustainability_email: Optional[str] = None
    procurement_contact_name: Optional[str] = None
    procurement_email: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_contact_name": self.primary_contact_name,
            "primary_email": self.primary_email,
            "primary_phone": self.primary_phone,
            "sustainability_contact_name": self.sustainability_contact_name,
            "sustainability_email": self.sustainability_email,
            "procurement_contact_name": self.procurement_contact_name,
            "procurement_email": self.procurement_email,
        }


@dataclass
class Supplier:
    """
    Supplier entity representing a company in the supply chain.

    This is the core entity for supply chain mapping, supporting:
    - Multi-tier classification (Tier 1, 2, 3, N)
    - External identifier integration (LEI, DUNS, VAT)
    - Parent-child corporate hierarchies
    - Qualification and status tracking

    Attributes:
        id: Unique internal identifier
        name: Legal company name
        tier: Supply chain tier classification
        country_code: ISO 3166-1 alpha-2 country code
        status: Supplier qualification status
        external_ids: External identifier codes
        address: Registered address
        parent_supplier_id: Parent company ID for corporate hierarchies
        industry_codes: NAICS/SIC/NACE industry classification codes
        commodities: EUDR commodities supplied
        annual_spend: Annual procurement spend (for Scope 3 allocation)
        currency: Currency code for spend amounts
        created_at: Record creation timestamp
        updated_at: Last update timestamp
        metadata: Additional custom attributes

    Example:
        >>> supplier = Supplier(
        ...     id="SUP001",
        ...     name="Acme Manufacturing GmbH",
        ...     tier=SupplierTier.TIER_1,
        ...     country_code="DE",
        ...     external_ids=ExternalIdentifiers(
        ...         lei="5493001KJTIIGC8Y1R12",
        ...         duns="123456789"
        ...     ),
        ...     annual_spend=Decimal("5000000.00"),
        ...     currency="EUR"
        ... )
    """
    id: str
    name: str
    tier: SupplierTier = SupplierTier.UNKNOWN
    country_code: Optional[str] = None
    status: SupplierStatus = SupplierStatus.ACTIVE
    external_ids: ExternalIdentifiers = field(default_factory=ExternalIdentifiers)
    address: Optional[Address] = None
    contact: Optional[ContactInfo] = None
    parent_supplier_id: Optional[str] = None
    industry_codes: Dict[str, str] = field(default_factory=dict)  # NAICS, SIC, NACE
    commodities: List[CommodityType] = field(default_factory=list)
    annual_spend: Optional[Decimal] = None
    currency: str = "USD"
    emission_factor_kg_co2e_per_usd: Optional[Decimal] = None
    certifications: List[str] = field(default_factory=list)
    sustainability_rating: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate supplier data."""
        if not self.id:
            raise ValueError("Supplier ID is required")
        if not self.name:
            raise ValueError("Supplier name is required")
        if self.country_code and len(self.country_code) != 2:
            raise ValueError(f"Country code must be 2 characters: {self.country_code}")

    @classmethod
    def create(
        cls,
        name: str,
        tier: SupplierTier = SupplierTier.UNKNOWN,
        country_code: Optional[str] = None,
        **kwargs
    ) -> "Supplier":
        """
        Factory method to create a new supplier with auto-generated ID.

        Args:
            name: Legal company name
            tier: Supply chain tier
            country_code: ISO country code
            **kwargs: Additional attributes

        Returns:
            New Supplier instance
        """
        supplier_id = f"SUP-{uuid.uuid4().hex[:12].upper()}"
        return cls(
            id=supplier_id,
            name=name,
            tier=tier,
            country_code=country_code,
            **kwargs
        )

    def update_tier(self, new_tier: SupplierTier) -> None:
        """Update supplier tier and timestamp."""
        self.tier = new_tier
        self.updated_at = datetime.utcnow()

    def is_high_risk_country(self) -> bool:
        """
        Check if supplier is in a high-risk country for EUDR/CSDDD.

        Returns:
            True if country is classified as high deforestation risk
        """
        # Countries with high deforestation risk (EUDR benchmarking)
        high_risk_countries = {
            "BR",  # Brazil
            "ID",  # Indonesia
            "MY",  # Malaysia
            "AR",  # Argentina
            "PY",  # Paraguay
            "BO",  # Bolivia
            "CO",  # Colombia
            "PE",  # Peru
            "EC",  # Ecuador
            "CG",  # Congo
            "CD",  # DR Congo
            "CM",  # Cameroon
            "CI",  # Cote d'Ivoire
            "GH",  # Ghana
            "NG",  # Nigeria
        }
        return self.country_code in high_risk_countries if self.country_code else False

    def has_eudr_commodity(self) -> bool:
        """Check if supplier provides EUDR-regulated commodities."""
        eudr_commodities = {
            CommodityType.CATTLE,
            CommodityType.COCOA,
            CommodityType.COFFEE,
            CommodityType.OIL_PALM,
            CommodityType.RUBBER,
            CommodityType.SOYA,
            CommodityType.WOOD,
            CommodityType.LEATHER,
            CommodityType.CHOCOLATE,
            CommodityType.PALM_OIL,
            CommodityType.FURNITURE,
            CommodityType.PAPER,
        }
        return any(c in eudr_commodities for c in self.commodities)

    def to_dict(self) -> Dict[str, Any]:
        """Convert supplier to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "tier": self.tier.value,
            "country_code": self.country_code,
            "status": self.status.value,
            "external_ids": self.external_ids.to_dict() if self.external_ids else {},
            "address": self.address.to_dict() if self.address else None,
            "contact": self.contact.to_dict() if self.contact else None,
            "parent_supplier_id": self.parent_supplier_id,
            "industry_codes": self.industry_codes,
            "commodities": [c.value for c in self.commodities],
            "annual_spend": str(self.annual_spend) if self.annual_spend else None,
            "currency": self.currency,
            "emission_factor_kg_co2e_per_usd": (
                str(self.emission_factor_kg_co2e_per_usd)
                if self.emission_factor_kg_co2e_per_usd else None
            ),
            "certifications": self.certifications,
            "sustainability_rating": self.sustainability_rating,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Supplier":
        """Create supplier from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            tier=SupplierTier(data.get("tier", 0)),
            country_code=data.get("country_code"),
            status=SupplierStatus(data.get("status", "active")),
            external_ids=ExternalIdentifiers(**data.get("external_ids", {})),
            address=Address(**data["address"]) if data.get("address") else None,
            contact=ContactInfo(**data["contact"]) if data.get("contact") else None,
            parent_supplier_id=data.get("parent_supplier_id"),
            industry_codes=data.get("industry_codes", {}),
            commodities=[CommodityType(c) for c in data.get("commodities", [])],
            annual_spend=Decimal(data["annual_spend"]) if data.get("annual_spend") else None,
            currency=data.get("currency", "USD"),
            emission_factor_kg_co2e_per_usd=(
                Decimal(data["emission_factor_kg_co2e_per_usd"])
                if data.get("emission_factor_kg_co2e_per_usd") else None
            ),
            certifications=data.get("certifications", []),
            sustainability_rating=data.get("sustainability_rating"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Facility:
    """
    Physical facility/site for a supplier.

    Facilities represent specific locations where production, processing,
    or storage occurs. Critical for EUDR traceability and Scope 3 allocation.

    Attributes:
        id: Unique facility identifier
        name: Facility name
        supplier_id: Parent supplier ID
        facility_type: Type of facility (factory, warehouse, farm, etc.)
        location: Geographic coordinates
        address: Physical address
        production_capacity: Annual production capacity
        capacity_unit: Unit for production capacity
        commodities_processed: EUDR commodities processed at this facility
        certifications: Facility-level certifications (FSC, RSPO, etc.)
        active: Whether facility is currently active
        created_at: Record creation timestamp
        updated_at: Last update timestamp
        metadata: Additional custom attributes
    """
    id: str
    name: str
    supplier_id: str
    facility_type: str = "production"  # production, warehouse, farm, processing, port
    location: Optional[GeoLocation] = None
    address: Optional[Address] = None
    production_capacity: Optional[Decimal] = None
    capacity_unit: Optional[str] = None
    commodities_processed: List[CommodityType] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        supplier_id: str,
        facility_type: str = "production",
        **kwargs
    ) -> "Facility":
        """Factory method to create a new facility with auto-generated ID."""
        facility_id = f"FAC-{uuid.uuid4().hex[:12].upper()}"
        return cls(
            id=facility_id,
            name=name,
            supplier_id=supplier_id,
            facility_type=facility_type,
            **kwargs
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert facility to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "supplier_id": self.supplier_id,
            "facility_type": self.facility_type,
            "location": self.location.to_dict() if self.location else None,
            "address": self.address.to_dict() if self.address else None,
            "production_capacity": str(self.production_capacity) if self.production_capacity else None,
            "capacity_unit": self.capacity_unit,
            "commodities_processed": [c.value for c in self.commodities_processed],
            "certifications": self.certifications,
            "active": self.active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Facility":
        """Create facility from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            supplier_id=data["supplier_id"],
            facility_type=data.get("facility_type", "production"),
            location=GeoLocation(**data["location"]) if data.get("location") else None,
            address=Address(**data["address"]) if data.get("address") else None,
            production_capacity=Decimal(data["production_capacity"]) if data.get("production_capacity") else None,
            capacity_unit=data.get("capacity_unit"),
            commodities_processed=[CommodityType(c) for c in data.get("commodities_processed", [])],
            certifications=data.get("certifications", []),
            active=data.get("active", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Material:
    """
    Raw material or component in the supply chain.

    Materials are the inputs to production processes and form the basis
    for material flow tracking and EUDR commodity traceability.

    Attributes:
        id: Unique material identifier
        name: Material name
        cn_code: Combined Nomenclature code (EU customs classification)
        hs_code: Harmonized System code (6-digit)
        commodity_type: EUDR commodity classification
        unit_of_measure: Standard unit (kg, tonnes, m3, etc.)
        emission_factor_kg_co2e: Emission factor per unit
        description: Material description
        cas_number: CAS registry number for chemicals
        metadata: Additional custom attributes
    """
    id: str
    name: str
    cn_code: Optional[str] = None  # EU Combined Nomenclature (8 digits)
    hs_code: Optional[str] = None  # Harmonized System (6 digits)
    commodity_type: Optional[CommodityType] = None
    unit_of_measure: str = "kg"
    emission_factor_kg_co2e: Optional[Decimal] = None
    description: Optional[str] = None
    cas_number: Optional[str] = None  # For chemicals
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, **kwargs) -> "Material":
        """Factory method to create a new material with auto-generated ID."""
        material_id = f"MAT-{uuid.uuid4().hex[:12].upper()}"
        return cls(id=material_id, name=name, **kwargs)

    def is_eudr_regulated(self) -> bool:
        """Check if material is regulated under EUDR."""
        if self.commodity_type:
            return self.commodity_type != CommodityType.OTHER

        # Check CN codes for EUDR commodities
        eudr_cn_prefixes = {
            "0102",  # Live cattle
            "0201", "0202",  # Beef
            "4101", "4104", "4107",  # Leather/hides
            "1801",  # Cocoa beans
            "1803", "1804", "1805", "1806",  # Cocoa products
            "0901",  # Coffee
            "1511",  # Palm oil
            "1513",  # Coconut/palm kernel oil
            "4001",  # Natural rubber
            "4005", "4006", "4007",  # Rubber products
            "1201",  # Soybeans
            "1507", "2304",  # Soybean oil/cake
            "44",  # Wood
            "47", "48", "49",  # Paper and pulp
            "94",  # Furniture (wood)
        }

        if self.cn_code:
            return any(self.cn_code.startswith(prefix) for prefix in eudr_cn_prefixes)

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert material to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "cn_code": self.cn_code,
            "hs_code": self.hs_code,
            "commodity_type": self.commodity_type.value if self.commodity_type else None,
            "unit_of_measure": self.unit_of_measure,
            "emission_factor_kg_co2e": str(self.emission_factor_kg_co2e) if self.emission_factor_kg_co2e else None,
            "description": self.description,
            "cas_number": self.cas_number,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Create material from dictionary representation."""
        return cls(
            id=data["id"],
            name=data["name"],
            cn_code=data.get("cn_code"),
            hs_code=data.get("hs_code"),
            commodity_type=CommodityType(data["commodity_type"]) if data.get("commodity_type") else None,
            unit_of_measure=data.get("unit_of_measure", "kg"),
            emission_factor_kg_co2e=Decimal(data["emission_factor_kg_co2e"]) if data.get("emission_factor_kg_co2e") else None,
            description=data.get("description"),
            cas_number=data.get("cas_number"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Product:
    """
    Finished product or intermediate good.

    Products are the outputs of production processes and can be
    composed of multiple materials. Supports bill of materials (BOM)
    tracking for Scope 3 and EUDR.

    Attributes:
        id: Unique product identifier
        name: Product name
        sku: Stock keeping unit
        cn_code: Combined Nomenclature code
        hs_code: Harmonized System code
        supplier_id: Supplier that produces this product
        bill_of_materials: List of (material_id, quantity) tuples
        emission_factor_kg_co2e: Product-level emission factor
        weight_kg: Product weight
        description: Product description
        metadata: Additional custom attributes
    """
    id: str
    name: str
    sku: Optional[str] = None
    cn_code: Optional[str] = None
    hs_code: Optional[str] = None
    supplier_id: Optional[str] = None
    bill_of_materials: List[Dict[str, Any]] = field(default_factory=list)  # [{material_id, quantity, unit}]
    emission_factor_kg_co2e: Optional[Decimal] = None
    weight_kg: Optional[Decimal] = None
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, **kwargs) -> "Product":
        """Factory method to create a new product with auto-generated ID."""
        product_id = f"PRD-{uuid.uuid4().hex[:12].upper()}"
        return cls(id=product_id, name=name, **kwargs)

    def add_material(self, material_id: str, quantity: Decimal, unit: str = "kg") -> None:
        """Add a material to the bill of materials."""
        self.bill_of_materials.append({
            "material_id": material_id,
            "quantity": str(quantity),
            "unit": unit,
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert product to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "sku": self.sku,
            "cn_code": self.cn_code,
            "hs_code": self.hs_code,
            "supplier_id": self.supplier_id,
            "bill_of_materials": self.bill_of_materials,
            "emission_factor_kg_co2e": str(self.emission_factor_kg_co2e) if self.emission_factor_kg_co2e else None,
            "weight_kg": str(self.weight_kg) if self.weight_kg else None,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class SupplierRelationship:
    """
    Relationship between two suppliers in the supply chain.

    Represents the edges in the supply chain graph, capturing:
    - Buyer-supplier relationships
    - Material flows between entities
    - Relationship metadata for traceability

    Attributes:
        id: Unique relationship identifier
        source_supplier_id: Upstream supplier (seller)
        target_supplier_id: Downstream supplier (buyer)
        relationship_type: Type of relationship
        materials: Materials exchanged in this relationship
        products: Products exchanged in this relationship
        annual_volume: Annual transaction volume
        volume_unit: Unit for volume measurement
        annual_spend: Annual procurement spend
        currency: Currency code
        contract_start_date: Contract start date
        contract_end_date: Contract end date
        active: Whether relationship is currently active
        verified: Whether relationship has been verified
        verification_date: Date of last verification
        verification_source: Source of verification
        metadata: Additional custom attributes
    """
    id: str
    source_supplier_id: str  # Upstream supplier (seller)
    target_supplier_id: str  # Downstream supplier (buyer)
    relationship_type: RelationshipType = RelationshipType.SUPPLIER
    materials: List[str] = field(default_factory=list)  # Material IDs
    products: List[str] = field(default_factory=list)  # Product IDs
    annual_volume: Optional[Decimal] = None
    volume_unit: Optional[str] = None
    annual_spend: Optional[Decimal] = None
    currency: str = "USD"
    contract_start_date: Optional[date] = None
    contract_end_date: Optional[date] = None
    active: bool = True
    verified: bool = False
    verification_date: Optional[datetime] = None
    verification_source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_supplier_id: str,
        target_supplier_id: str,
        relationship_type: RelationshipType = RelationshipType.SUPPLIER,
        **kwargs
    ) -> "SupplierRelationship":
        """Factory method to create a new relationship with auto-generated ID."""
        relationship_id = f"REL-{uuid.uuid4().hex[:12].upper()}"
        return cls(
            id=relationship_id,
            source_supplier_id=source_supplier_id,
            target_supplier_id=target_supplier_id,
            relationship_type=relationship_type,
            **kwargs
        )

    def verify(self, source: str) -> None:
        """Mark relationship as verified."""
        self.verified = True
        self.verification_date = datetime.utcnow()
        self.verification_source = source
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary representation."""
        return {
            "id": self.id,
            "source_supplier_id": self.source_supplier_id,
            "target_supplier_id": self.target_supplier_id,
            "relationship_type": self.relationship_type.value,
            "materials": self.materials,
            "products": self.products,
            "annual_volume": str(self.annual_volume) if self.annual_volume else None,
            "volume_unit": self.volume_unit,
            "annual_spend": str(self.annual_spend) if self.annual_spend else None,
            "currency": self.currency,
            "contract_start_date": self.contract_start_date.isoformat() if self.contract_start_date else None,
            "contract_end_date": self.contract_end_date.isoformat() if self.contract_end_date else None,
            "active": self.active,
            "verified": self.verified,
            "verification_date": self.verification_date.isoformat() if self.verification_date else None,
            "verification_source": self.verification_source,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SupplierRelationship":
        """Create relationship from dictionary representation."""
        return cls(
            id=data["id"],
            source_supplier_id=data["source_supplier_id"],
            target_supplier_id=data["target_supplier_id"],
            relationship_type=RelationshipType(data.get("relationship_type", "supplier")),
            materials=data.get("materials", []),
            products=data.get("products", []),
            annual_volume=Decimal(data["annual_volume"]) if data.get("annual_volume") else None,
            volume_unit=data.get("volume_unit"),
            annual_spend=Decimal(data["annual_spend"]) if data.get("annual_spend") else None,
            currency=data.get("currency", "USD"),
            contract_start_date=date.fromisoformat(data["contract_start_date"]) if data.get("contract_start_date") else None,
            contract_end_date=date.fromisoformat(data["contract_end_date"]) if data.get("contract_end_date") else None,
            active=data.get("active", True),
            verified=data.get("verified", False),
            verification_date=datetime.fromisoformat(data["verification_date"]) if data.get("verification_date") else None,
            verification_source=data.get("verification_source"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.utcnow(),
            metadata=data.get("metadata", {}),
        )


# Type aliases for convenience
SupplierList = List[Supplier]
FacilityList = List[Facility]
MaterialList = List[Material]
ProductList = List[Product]
RelationshipList = List[SupplierRelationship]
