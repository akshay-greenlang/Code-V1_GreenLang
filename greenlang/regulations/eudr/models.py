# -*- coding: utf-8 -*-
"""
greenlang/regulations/eudr/models.py

EUDR Data Models

This module defines the data models for EU Deforestation Regulation compliance.
All models are designed for:
- Zero-hallucination guarantee (deterministic validation)
- Complete audit trail (provenance tracking)
- Regulatory compliance (EUDR Article requirements)

Reference: Regulation (EU) 2023/1115

Author: GreenLang Framework Team
Date: November 2025
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional, List, Dict, Literal
from enum import Enum
import hashlib
import json


class EUDRCommodity(str, Enum):
    """
    EUDR Covered Commodities (Article 1)

    The seven commodities covered by the regulation, plus derived products.
    """
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class EUDRComplianceStatus(str, Enum):
    """Compliance status for EUDR due diligence"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    INSUFFICIENT_DATA = "insufficient_data"
    EXEMPTED = "exempted"


class RiskLevel(str, Enum):
    """Risk levels per EUDR Article 29"""
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


@dataclass
class GeolocationData:
    """
    Geolocation data for production plot (Article 9)

    EUDR requires geolocation of all plots where commodities were produced.
    For plots > 4 hectares, polygon coordinates are required.
    For plots <= 4 hectares, a single point (centroid) is sufficient.
    """
    # Coordinates (WGS84)
    latitude: float
    longitude: float

    # Polygon for plots > 4 hectares (list of [lat, lon] pairs)
    polygon_coordinates: Optional[List[List[float]]] = None

    # Plot metadata
    plot_area_hectares: Optional[float] = None
    country_code: str = ""  # ISO 3166-1 alpha-2
    region: Optional[str] = None
    sub_region: Optional[str] = None

    # Validation
    coordinate_precision: int = 6  # Decimal places
    coordinate_system: str = "WGS84"

    def __post_init__(self):
        """Validate coordinates"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90, got {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Longitude must be between -180 and 180, got {self.longitude}")

        # Polygon required for large plots
        if self.plot_area_hectares and self.plot_area_hectares > 4.0:
            if not self.polygon_coordinates:
                raise ValueError(
                    "Polygon coordinates required for plots > 4 hectares per EUDR Article 9"
                )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "latitude": round(self.latitude, self.coordinate_precision),
            "longitude": round(self.longitude, self.coordinate_precision),
            "polygon_coordinates": self.polygon_coordinates,
            "plot_area_hectares": self.plot_area_hectares,
            "country_code": self.country_code,
            "region": self.region,
            "sub_region": self.sub_region,
            "coordinate_system": self.coordinate_system,
        }


@dataclass
class ProductionPlot:
    """
    Production plot information for EUDR traceability.

    Represents a single plot of land where a covered commodity was produced.
    """
    plot_id: str
    geolocation: GeolocationData
    commodity: EUDRCommodity

    # Production details
    production_date: date
    harvest_date: Optional[date] = None
    quantity_kg: Optional[float] = None

    # Legal compliance (Article 3)
    production_country: str = ""  # ISO country code
    is_legally_produced: bool = False
    legal_compliance_evidence: Optional[str] = None

    # Deforestation status (Article 3)
    # Cut-off date: December 31, 2020
    DEFORESTATION_CUTOFF_DATE: date = field(default=date(2020, 12, 31), repr=False)

    # Land use status
    was_forested_after_cutoff: Optional[bool] = None
    deforestation_verified: bool = False
    forest_degradation_verified: bool = False

    # Supplier information
    supplier_id: Optional[str] = None
    supplier_name: Optional[str] = None

    # Certification (if applicable)
    certification_scheme: Optional[str] = None
    certification_id: Optional[str] = None
    certification_valid_until: Optional[date] = None

    def is_deforestation_free(self) -> bool:
        """
        Check if plot is deforestation-free per EUDR Article 3.

        Returns True if the plot was not forested after Dec 31, 2020,
        or if deforestation has been verified as not occurring.
        """
        if self.was_forested_after_cutoff is False:
            return True
        if self.deforestation_verified and not self.was_forested_after_cutoff:
            return True
        return False

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "plot_id": self.plot_id,
            "geolocation": self.geolocation.to_dict(),
            "commodity": self.commodity.value,
            "production_date": self.production_date.isoformat(),
            "harvest_date": self.harvest_date.isoformat() if self.harvest_date else None,
            "quantity_kg": self.quantity_kg,
            "production_country": self.production_country,
            "is_legally_produced": self.is_legally_produced,
            "is_deforestation_free": self.is_deforestation_free(),
            "supplier_id": self.supplier_id,
            "certification_scheme": self.certification_scheme,
        }


@dataclass
class EUDRProduct:
    """
    EUDR-covered product with full traceability.

    Represents a product placed on the EU market that must comply with EUDR.
    """
    product_id: str
    product_name: str
    commodity: EUDRCommodity

    # Product classification
    hs_code: str  # Harmonized System code (6-8 digits)
    cn_code: Optional[str] = None  # Combined Nomenclature code (EU-specific)
    product_description: Optional[str] = None

    # Derived product (e.g., chocolate from cocoa, leather from cattle)
    is_derived_product: bool = False
    derived_from_commodity: Optional[EUDRCommodity] = None

    # Quantity
    quantity_kg: float = 0.0
    quantity_units: Optional[float] = None
    unit_type: Optional[str] = None

    # Traceability - linked production plots
    production_plots: List[ProductionPlot] = field(default_factory=list)

    # Supplier chain
    supplier_chain: List[str] = field(default_factory=list)

    # Compliance status
    compliance_status: EUDRComplianceStatus = EUDRComplianceStatus.PENDING_VERIFICATION

    # Due diligence reference
    due_diligence_id: Optional[str] = None

    def get_countries_of_production(self) -> List[str]:
        """Get list of all production countries for this product"""
        countries = set()
        for plot in self.production_plots:
            if plot.production_country:
                countries.add(plot.production_country)
        return sorted(list(countries))

    def is_fully_traceable(self) -> bool:
        """Check if product has full geolocation traceability"""
        if not self.production_plots:
            return False
        return all(
            plot.geolocation.latitude != 0 and plot.geolocation.longitude != 0
            for plot in self.production_plots
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "product_id": self.product_id,
            "product_name": self.product_name,
            "commodity": self.commodity.value,
            "hs_code": self.hs_code,
            "cn_code": self.cn_code,
            "quantity_kg": self.quantity_kg,
            "is_derived_product": self.is_derived_product,
            "countries_of_production": self.get_countries_of_production(),
            "is_fully_traceable": self.is_fully_traceable(),
            "compliance_status": self.compliance_status.value,
            "production_plots_count": len(self.production_plots),
        }


@dataclass
class RiskAssessment:
    """
    EUDR Risk Assessment (Article 10)

    Assesses the risk that products are not compliant with EUDR requirements.
    """
    assessment_id: str
    assessment_date: datetime
    product: EUDRProduct

    # Risk scores (0-100, higher = more risk)
    country_risk_score: float = 0.0
    commodity_risk_score: float = 0.0
    supplier_risk_score: float = 0.0
    traceability_risk_score: float = 0.0

    # Overall risk
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.STANDARD

    # Risk factors identified
    risk_factors: List[str] = field(default_factory=list)

    # Mitigation measures applied
    mitigation_measures: List[str] = field(default_factory=list)

    # Assessment methodology
    methodology: str = "EUDR_Standard_Risk_Assessment_v1"

    def __post_init__(self):
        """Calculate overall risk and assign level"""
        self._calculate_overall_risk()
        self._assign_risk_level()

    def _calculate_overall_risk(self):
        """
        Calculate overall risk score.

        DETERMINISTIC calculation - same inputs always produce same output.
        """
        # Weighted average of risk factors
        weights = {
            "country": 0.30,
            "commodity": 0.20,
            "supplier": 0.25,
            "traceability": 0.25,
        }

        self.overall_risk_score = (
            self.country_risk_score * weights["country"] +
            self.commodity_risk_score * weights["commodity"] +
            self.supplier_risk_score * weights["supplier"] +
            self.traceability_risk_score * weights["traceability"]
        )

    def _assign_risk_level(self):
        """Assign risk level based on overall score"""
        if self.overall_risk_score < 30:
            self.risk_level = RiskLevel.LOW
        elif self.overall_risk_score < 70:
            self.risk_level = RiskLevel.STANDARD
        else:
            self.risk_level = RiskLevel.HIGH

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "assessment_id": self.assessment_id,
            "assessment_date": self.assessment_date.isoformat(),
            "product_id": self.product.product_id,
            "country_risk_score": self.country_risk_score,
            "commodity_risk_score": self.commodity_risk_score,
            "supplier_risk_score": self.supplier_risk_score,
            "traceability_risk_score": self.traceability_risk_score,
            "overall_risk_score": self.overall_risk_score,
            "risk_level": self.risk_level.value,
            "risk_factors": self.risk_factors,
            "mitigation_measures": self.mitigation_measures,
            "methodology": self.methodology,
        }


@dataclass
class SupplierDeclaration:
    """
    Supplier declaration for EUDR compliance.

    Suppliers must provide declarations confirming compliance with EUDR.
    """
    declaration_id: str
    supplier_id: str
    supplier_name: str
    declaration_date: date

    # Supplier details
    supplier_country: str
    supplier_address: Optional[str] = None
    supplier_registration_number: Optional[str] = None

    # Declaration scope
    commodities_covered: List[EUDRCommodity] = field(default_factory=list)
    products_covered: List[str] = field(default_factory=list)

    # Declaration statements
    confirms_deforestation_free: bool = False
    confirms_legal_production: bool = False
    confirms_traceability: bool = False

    # Supporting documentation
    documentation_provided: List[str] = field(default_factory=list)

    # Validity
    valid_from: date = field(default_factory=date.today)
    valid_until: Optional[date] = None

    # Signature
    signatory_name: Optional[str] = None
    signatory_position: Optional[str] = None

    def is_valid(self, check_date: Optional[date] = None) -> bool:
        """Check if declaration is valid on given date"""
        check = check_date or date.today()
        if check < self.valid_from:
            return False
        if self.valid_until and check > self.valid_until:
            return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "declaration_id": self.declaration_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "declaration_date": self.declaration_date.isoformat(),
            "commodities_covered": [c.value for c in self.commodities_covered],
            "confirms_deforestation_free": self.confirms_deforestation_free,
            "confirms_legal_production": self.confirms_legal_production,
            "confirms_traceability": self.confirms_traceability,
            "is_valid": self.is_valid(),
        }


@dataclass
class DueDiligenceStatement:
    """
    EUDR Due Diligence Statement (Article 4)

    The formal statement that must be submitted to EU authorities before
    placing covered products on the EU market.
    """
    statement_id: str
    submission_date: datetime

    # Operator information
    operator_name: str
    operator_country: str
    operator_eori_number: Optional[str] = None  # EU EORI number

    # Product information
    products: List[EUDRProduct] = field(default_factory=list)

    # Risk assessment
    risk_assessment: Optional[RiskAssessment] = None

    # Compliance declarations
    declares_deforestation_free: bool = False
    declares_legal_production: bool = False
    declares_due_diligence_performed: bool = False

    # Supporting information
    production_plots: List[ProductionPlot] = field(default_factory=list)
    supplier_declarations: List[SupplierDeclaration] = field(default_factory=list)

    # Statement status
    status: EUDRComplianceStatus = EUDRComplianceStatus.PENDING_VERIFICATION

    # Reference numbers
    eu_information_system_ref: Optional[str] = None
    customs_declaration_ref: Optional[str] = None

    # Audit trail
    content_hash: str = field(init=False, default="")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Generate content hash for audit trail"""
        self.content_hash = self._generate_hash()

    def _generate_hash(self) -> str:
        """Generate SHA-256 hash of statement content"""
        hash_data = {
            "statement_id": self.statement_id,
            "operator_name": self.operator_name,
            "products": [p.product_id for p in self.products],
            "declares_deforestation_free": self.declares_deforestation_free,
            "declares_legal_production": self.declares_legal_production,
            "submission_date": self.submission_date.isoformat(),
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def is_complete(self) -> bool:
        """Check if statement has all required information"""
        required_conditions = [
            len(self.products) > 0,
            all(p.is_fully_traceable() for p in self.products),
            self.declares_deforestation_free,
            self.declares_legal_production,
            self.declares_due_diligence_performed,
            self.risk_assessment is not None,
        ]
        return all(required_conditions)

    def get_compliance_status(self) -> EUDRComplianceStatus:
        """Determine compliance status based on statement content"""
        if not self.is_complete():
            return EUDRComplianceStatus.INSUFFICIENT_DATA

        # Check all products are deforestation-free
        all_deforestation_free = all(
            all(plot.is_deforestation_free() for plot in product.production_plots)
            for product in self.products
            if product.production_plots
        )

        # Check all products are legally produced
        all_legal = all(
            all(plot.is_legally_produced for plot in product.production_plots)
            for product in self.products
            if product.production_plots
        )

        if all_deforestation_free and all_legal:
            return EUDRComplianceStatus.COMPLIANT
        else:
            return EUDRComplianceStatus.NON_COMPLIANT

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "statement_id": self.statement_id,
            "submission_date": self.submission_date.isoformat(),
            "operator_name": self.operator_name,
            "operator_country": self.operator_country,
            "products_count": len(self.products),
            "production_plots_count": len(self.production_plots),
            "declares_deforestation_free": self.declares_deforestation_free,
            "declares_legal_production": self.declares_legal_production,
            "declares_due_diligence_performed": self.declares_due_diligence_performed,
            "is_complete": self.is_complete(),
            "compliance_status": self.get_compliance_status().value,
            "content_hash": self.content_hash,
            "eu_information_system_ref": self.eu_information_system_ref,
        }

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
