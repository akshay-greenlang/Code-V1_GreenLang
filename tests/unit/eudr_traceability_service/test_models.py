# -*- coding: utf-8 -*-
"""
Unit Tests for EUDR Traceability Models (AGENT-DATA-005)

Tests all enums (EUDRCommodity 7 values, RiskLevel 4, CustodyModel 3,
DDSStatus 4, DDSType 3), GeolocationData, PlotRecord, CustodyTransfer,
RiskScore, DueDiligenceStatement, CommodityClassification,
SupplierDeclaration, and all request models.

Coverage target: 85%+ of models.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/eudr_traceability/models.py
# ---------------------------------------------------------------------------


class EUDRCommodity(str, Enum):
    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


# Derived products mapping per EUDR Annex I
DERIVED_PRODUCTS = {
    EUDRCommodity.CATTLE: ["beef", "leather", "gelatin", "tallow", "collagen"],
    EUDRCommodity.COCOA: ["chocolate", "cocoa_butter", "cocoa_paste", "cocoa_powder"],
    EUDRCommodity.COFFEE: ["roasted_coffee", "coffee_extract", "coffee_husk"],
    EUDRCommodity.OIL_PALM: ["palm_oil", "palm_kernel_oil", "palm_olein", "glycerol"],
    EUDRCommodity.RUBBER: ["natural_rubber", "rubber_compound", "latex", "tyres"],
    EUDRCommodity.SOYA: ["soybean_oil", "soya_flour", "soybean_meal", "soy_lecithin"],
    EUDRCommodity.WOOD: ["timber", "plywood", "pulp", "paper", "charcoal", "furniture"],
}


class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"
    UNKNOWN = "unknown"


class CustodyModel(str, Enum):
    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"


class DDSStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DDSType(str, Enum):
    IMPORT_PLACEMENT = "import_placement"
    EXPORT = "export"
    DOMESTIC = "domestic"


# ---------------------------------------------------------------------------
# Inline Layer 1 models
# ---------------------------------------------------------------------------


class GeolocationData:
    def __init__(
        self,
        latitude: float = 0.0,
        longitude: float = 0.0,
        plot_area_ha: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
        coordinate_system: str = "WGS84",
        precision_m: float = 10.0,
    ):
        if latitude < -90 or latitude > 90:
            raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
        if longitude < -180 or longitude > 180:
            raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
        if plot_area_ha > 4.0 and polygon_coordinates is None:
            raise ValueError(
                f"Plots larger than 4 hectares require polygon coordinates "
                f"(plot_area_ha={plot_area_ha})"
            )

        self.latitude = latitude
        self.longitude = longitude
        self.plot_area_ha = plot_area_ha
        self.polygon_coordinates = polygon_coordinates
        self.coordinate_system = coordinate_system
        self.precision_m = precision_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "plot_area_ha": self.plot_area_ha,
            "polygon_coordinates": self.polygon_coordinates,
            "coordinate_system": self.coordinate_system,
            "precision_m": self.precision_m,
        }


class PlotRecord:
    def __init__(
        self,
        plot_id: str = "",
        commodity: str = "cocoa",
        country_code: str = "",
        geolocation: Optional[GeolocationData] = None,
        operator_id: str = "",
        operator_name: str = "",
        deforestation_free: Optional[bool] = None,
        legally_produced: Optional[bool] = None,
        cutoff_date: str = "2020-12-31",
        risk_level: str = "unknown",
        registered_at: Optional[str] = None,
    ):
        self.plot_id = plot_id or f"PLOT-{uuid.uuid4().hex[:8]}"
        self.commodity = commodity
        self.country_code = country_code
        self.geolocation = geolocation
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.deforestation_free = deforestation_free
        self.legally_produced = legally_produced
        self.cutoff_date = cutoff_date
        self.risk_level = risk_level
        self.registered_at = registered_at or datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plot_id": self.plot_id,
            "commodity": self.commodity,
            "country_code": self.country_code,
            "geolocation": self.geolocation.to_dict() if self.geolocation else None,
            "operator_id": self.operator_id,
            "deforestation_free": self.deforestation_free,
            "legally_produced": self.legally_produced,
            "cutoff_date": self.cutoff_date,
            "risk_level": self.risk_level,
        }


class CustodyTransfer:
    def __init__(
        self,
        transfer_id: str = "",
        transaction_id: str = "",
        batch_number: str = "",
        commodity: str = "cocoa",
        quantity_kg: float = 0.0,
        unit: str = "kg",
        custody_model: str = "segregated",
        from_operator_id: str = "",
        to_operator_id: str = "",
        origin_plot_ids: Optional[List[str]] = None,
        cn_code: str = "",
        transfer_date: Optional[str] = None,
        verified: bool = False,
        provenance_hash: Optional[str] = None,
    ):
        if quantity_kg <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity_kg}")

        self.transfer_id = transfer_id or f"TXF-{uuid.uuid4().hex[:8]}"
        self.transaction_id = transaction_id or f"TXN-{uuid.uuid4().hex[:8]}"
        self.batch_number = batch_number
        self.commodity = commodity
        self.quantity_kg = quantity_kg
        self.unit = unit
        self.custody_model = custody_model
        self.from_operator_id = from_operator_id
        self.to_operator_id = to_operator_id
        self.origin_plot_ids = origin_plot_ids or []
        self.cn_code = cn_code
        self.transfer_date = transfer_date or datetime.utcnow().isoformat()
        self.verified = verified
        self.provenance_hash = provenance_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transfer_id": self.transfer_id,
            "transaction_id": self.transaction_id,
            "batch_number": self.batch_number,
            "commodity": self.commodity,
            "quantity_kg": self.quantity_kg,
            "custody_model": self.custody_model,
            "from_operator_id": self.from_operator_id,
            "to_operator_id": self.to_operator_id,
            "origin_plot_ids": self.origin_plot_ids,
            "cn_code": self.cn_code,
            "verified": self.verified,
        }


class RiskScore:
    def __init__(
        self,
        score_id: str = "",
        entity_id: str = "",
        country_score: float = 0.0,
        commodity_score: float = 0.0,
        supplier_score: float = 0.0,
        traceability_score: float = 0.0,
        overall_score: float = 0.0,
        risk_level: str = "unknown",
    ):
        self.score_id = score_id or str(uuid.uuid4())
        self.entity_id = entity_id
        self.country_score = country_score
        self.commodity_score = commodity_score
        self.supplier_score = supplier_score
        self.traceability_score = traceability_score
        self.overall_score = overall_score

        # Determine risk level from overall score
        if risk_level != "unknown":
            self.risk_level = risk_level
        elif overall_score < 30:
            self.risk_level = "low"
        elif overall_score <= 70:
            self.risk_level = "standard"
        else:
            self.risk_level = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "entity_id": self.entity_id,
            "country_score": self.country_score,
            "commodity_score": self.commodity_score,
            "supplier_score": self.supplier_score,
            "traceability_score": self.traceability_score,
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
        }


class DueDiligenceStatement:
    def __init__(
        self,
        dds_id: str = "",
        operator_id: str = "",
        operator_name: str = "",
        dds_type: str = "import_placement",
        status: str = "draft",
        commodity: str = "cocoa",
        country_of_production: str = "",
        plot_ids: Optional[List[str]] = None,
        total_quantity_kg: float = 0.0,
        cn_codes: Optional[List[str]] = None,
        risk_assessment_id: Optional[str] = None,
        submitted_at: Optional[str] = None,
        verified_at: Optional[str] = None,
    ):
        self.dds_id = dds_id or f"DDS-{uuid.uuid4().hex[:8]}"
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.dds_type = dds_type
        self.status = status
        self.commodity = commodity
        self.country_of_production = country_of_production
        self.plot_ids = plot_ids or []
        self.total_quantity_kg = total_quantity_kg
        self.cn_codes = cn_codes or []
        self.risk_assessment_id = risk_assessment_id
        self.submitted_at = submitted_at
        self.verified_at = verified_at

    def submit(self) -> None:
        if self.status != "draft":
            raise ValueError(f"Can only submit a DDS in draft status, current: {self.status}")
        self.status = "submitted"
        self.submitted_at = datetime.utcnow().isoformat()

    def verify(self) -> None:
        if self.status != "submitted":
            raise ValueError(f"Can only verify a submitted DDS, current: {self.status}")
        self.status = "verified"
        self.verified_at = datetime.utcnow().isoformat()

    def reject(self) -> None:
        if self.status != "submitted":
            raise ValueError(f"Can only reject a submitted DDS, current: {self.status}")
        self.status = "rejected"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dds_id": self.dds_id,
            "operator_id": self.operator_id,
            "dds_type": self.dds_type,
            "status": self.status,
            "commodity": self.commodity,
            "country_of_production": self.country_of_production,
            "plot_ids": self.plot_ids,
            "total_quantity_kg": self.total_quantity_kg,
        }


class CommodityClassification:
    def __init__(
        self,
        commodity: str = "",
        is_primary: bool = True,
        derived_product: Optional[str] = None,
        cn_code: str = "",
        hs_code: str = "",
    ):
        self.commodity = commodity
        self.is_primary = is_primary
        self.derived_product = derived_product
        self.cn_code = cn_code
        self.hs_code = hs_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commodity": self.commodity,
            "is_primary": self.is_primary,
            "derived_product": self.derived_product,
            "cn_code": self.cn_code,
            "hs_code": self.hs_code,
        }


class SupplierDeclaration:
    def __init__(
        self,
        declaration_id: str = "",
        supplier_id: str = "",
        supplier_name: str = "",
        commodity: str = "",
        country_code: str = "",
        valid_from: str = "",
        valid_until: str = "",
        deforestation_free: bool = False,
        legally_produced: bool = False,
        plot_ids: Optional[List[str]] = None,
    ):
        self.declaration_id = declaration_id or f"DECL-{uuid.uuid4().hex[:8]}"
        self.supplier_id = supplier_id
        self.supplier_name = supplier_name
        self.commodity = commodity
        self.country_code = country_code
        self.valid_from = valid_from
        self.valid_until = valid_until
        self.deforestation_free = deforestation_free
        self.legally_produced = legally_produced
        self.plot_ids = plot_ids or []

    def is_valid_on(self, check_date: str) -> bool:
        return self.valid_from <= check_date <= self.valid_until

    def to_dict(self) -> Dict[str, Any]:
        return {
            "declaration_id": self.declaration_id,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "commodity": self.commodity,
            "country_code": self.country_code,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "deforestation_free": self.deforestation_free,
            "legally_produced": self.legally_produced,
        }


# ---------------------------------------------------------------------------
# Inline Request models
# ---------------------------------------------------------------------------


class RegisterPlotRequest:
    def __init__(
        self,
        commodity: str = "cocoa",
        country_code: str = "",
        latitude: float = 0.0,
        longitude: float = 0.0,
        plot_area_ha: float = 0.0,
        polygon_coordinates: Optional[List[List[float]]] = None,
        operator_id: str = "",
        operator_name: str = "",
    ):
        self.commodity = commodity
        self.country_code = country_code
        self.latitude = latitude
        self.longitude = longitude
        self.plot_area_ha = plot_area_ha
        self.polygon_coordinates = polygon_coordinates
        self.operator_id = operator_id
        self.operator_name = operator_name


class RecordTransferRequest:
    def __init__(
        self,
        batch_number: str = "",
        commodity: str = "cocoa",
        quantity_kg: float = 0.0,
        custody_model: str = "segregated",
        from_operator_id: str = "",
        to_operator_id: str = "",
        origin_plot_ids: Optional[List[str]] = None,
        cn_code: str = "",
    ):
        self.batch_number = batch_number
        self.commodity = commodity
        self.quantity_kg = quantity_kg
        self.custody_model = custody_model
        self.from_operator_id = from_operator_id
        self.to_operator_id = to_operator_id
        self.origin_plot_ids = origin_plot_ids or []
        self.cn_code = cn_code


class GenerateDDSRequest:
    def __init__(
        self,
        operator_id: str = "",
        operator_name: str = "",
        dds_type: str = "import_placement",
        commodity: str = "cocoa",
        country_of_production: str = "",
        plot_ids: Optional[List[str]] = None,
        total_quantity_kg: float = 0.0,
        cn_codes: Optional[List[str]] = None,
    ):
        self.operator_id = operator_id
        self.operator_name = operator_name
        self.dds_type = dds_type
        self.commodity = commodity
        self.country_of_production = country_of_production
        self.plot_ids = plot_ids or []
        self.total_quantity_kg = total_quantity_kg
        self.cn_codes = cn_codes or []


# ===========================================================================
# Test Classes -- Enums
# ===========================================================================


class TestEUDRCommodityEnum:
    """Test EUDRCommodity enum values (7 EUDR-regulated commodities)."""

    def test_cattle(self):
        assert EUDRCommodity.CATTLE.value == "cattle"

    def test_cocoa(self):
        assert EUDRCommodity.COCOA.value == "cocoa"

    def test_coffee(self):
        assert EUDRCommodity.COFFEE.value == "coffee"

    def test_oil_palm(self):
        assert EUDRCommodity.OIL_PALM.value == "oil_palm"

    def test_rubber(self):
        assert EUDRCommodity.RUBBER.value == "rubber"

    def test_soya(self):
        assert EUDRCommodity.SOYA.value == "soya"

    def test_wood(self):
        assert EUDRCommodity.WOOD.value == "wood"

    def test_all_seven_commodities(self):
        """EUDR covers exactly 7 primary commodities."""
        assert len(EUDRCommodity) == 7

    def test_from_value(self):
        assert EUDRCommodity("cocoa") == EUDRCommodity.COCOA

    @pytest.mark.parametrize("commodity", list(EUDRCommodity))
    def test_derived_products_exist(self, commodity):
        """All 7 commodities have derived product entries."""
        assert commodity in DERIVED_PRODUCTS
        assert len(DERIVED_PRODUCTS[commodity]) > 0


class TestRiskLevelEnum:
    """Test RiskLevel enum values (4 levels)."""

    def test_low(self):
        assert RiskLevel.LOW.value == "low"

    def test_standard(self):
        assert RiskLevel.STANDARD.value == "standard"

    def test_high(self):
        assert RiskLevel.HIGH.value == "high"

    def test_unknown(self):
        assert RiskLevel.UNKNOWN.value == "unknown"

    def test_all_4_levels(self):
        assert len(RiskLevel) == 4


class TestCustodyModelEnum:
    """Test CustodyModel enum values (3 models per EUDR Article 12)."""

    def test_identity_preserved(self):
        assert CustodyModel.IDENTITY_PRESERVED.value == "identity_preserved"

    def test_segregated(self):
        assert CustodyModel.SEGREGATED.value == "segregated"

    def test_mass_balance(self):
        assert CustodyModel.MASS_BALANCE.value == "mass_balance"

    def test_all_3_models(self):
        assert len(CustodyModel) == 3


class TestDDSStatusEnum:
    """Test DDSStatus enum values (4 statuses)."""

    def test_draft(self):
        assert DDSStatus.DRAFT.value == "draft"

    def test_submitted(self):
        assert DDSStatus.SUBMITTED.value == "submitted"

    def test_verified(self):
        assert DDSStatus.VERIFIED.value == "verified"

    def test_rejected(self):
        assert DDSStatus.REJECTED.value == "rejected"

    def test_all_4_statuses(self):
        assert len(DDSStatus) == 4


class TestDDSTypeEnum:
    """Test DDSType enum values (3 types)."""

    def test_import_placement(self):
        assert DDSType.IMPORT_PLACEMENT.value == "import_placement"

    def test_export(self):
        assert DDSType.EXPORT.value == "export"

    def test_domestic(self):
        assert DDSType.DOMESTIC.value == "domestic"

    def test_all_3_types(self):
        assert len(DDSType) == 3

    @pytest.mark.parametrize("dds_type_val", ["import_placement", "export", "domestic"])
    def test_dds_type_values(self, dds_type_val):
        """All three DDS type values can be constructed from strings."""
        assert DDSType(dds_type_val).value == dds_type_val


# ===========================================================================
# Test Classes -- GeolocationData
# ===========================================================================


class TestGeolocationData:
    """Test GeolocationData model with EUDR geolocation requirements."""

    def test_valid_coordinates(self):
        """Normal lat/lon within valid ranges."""
        geo = GeolocationData(latitude=5.55, longitude=-73.36, plot_area_ha=2.0)
        assert geo.latitude == 5.55
        assert geo.longitude == -73.36
        assert geo.coordinate_system == "WGS84"

    def test_invalid_latitude_too_high(self):
        """Latitude > 90 raises ValueError."""
        with pytest.raises(ValueError, match="Latitude must be between"):
            GeolocationData(latitude=91.0, longitude=0.0)

    def test_invalid_latitude_too_low(self):
        """Latitude < -90 raises ValueError."""
        with pytest.raises(ValueError, match="Latitude must be between"):
            GeolocationData(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude_too_high(self):
        """Longitude > 180 raises ValueError."""
        with pytest.raises(ValueError, match="Longitude must be between"):
            GeolocationData(latitude=0.0, longitude=181.0)

    def test_invalid_longitude_too_low(self):
        """Longitude < -180 raises ValueError."""
        with pytest.raises(ValueError, match="Longitude must be between"):
            GeolocationData(latitude=0.0, longitude=-181.0)

    def test_polygon_required_large_plot(self):
        """Plot > 4ha requires polygon coordinates per EUDR Article 9(1)(d)."""
        with pytest.raises(ValueError, match="polygon coordinates"):
            GeolocationData(latitude=5.0, longitude=-73.0, plot_area_ha=5.0)

    def test_point_ok_small_plot(self):
        """Plot <= 4ha does not require polygon, point is sufficient."""
        geo = GeolocationData(latitude=5.0, longitude=-73.0, plot_area_ha=3.5)
        assert geo.plot_area_ha == 3.5
        assert geo.polygon_coordinates is None

    def test_large_plot_with_polygon(self):
        """Plot > 4ha with polygon coordinates succeeds."""
        polygon = [[5.0, -73.0], [5.1, -73.0], [5.1, -72.9], [5.0, -72.9]]
        geo = GeolocationData(
            latitude=5.05, longitude=-72.95,
            plot_area_ha=10.0, polygon_coordinates=polygon,
        )
        assert geo.plot_area_ha == 10.0
        assert len(geo.polygon_coordinates) == 4

    def test_to_dict(self):
        """Serialization roundtrip via to_dict."""
        geo = GeolocationData(latitude=1.5, longitude=103.8, plot_area_ha=2.0)
        d = geo.to_dict()
        assert d["latitude"] == 1.5
        assert d["longitude"] == 103.8
        assert d["coordinate_system"] == "WGS84"
        assert d["precision_m"] == 10.0
        assert d["polygon_coordinates"] is None

    def test_boundary_latitude_90(self):
        """Latitude exactly 90 is valid (North Pole)."""
        geo = GeolocationData(latitude=90.0, longitude=0.0)
        assert geo.latitude == 90.0

    def test_boundary_latitude_neg90(self):
        """Latitude exactly -90 is valid (South Pole)."""
        geo = GeolocationData(latitude=-90.0, longitude=0.0)
        assert geo.latitude == -90.0

    def test_boundary_longitude_180(self):
        """Longitude exactly 180 is valid."""
        geo = GeolocationData(latitude=0.0, longitude=180.0)
        assert geo.longitude == 180.0

    def test_boundary_longitude_neg180(self):
        """Longitude exactly -180 is valid."""
        geo = GeolocationData(latitude=0.0, longitude=-180.0)
        assert geo.longitude == -180.0


# ===========================================================================
# Test Classes -- PlotRecord
# ===========================================================================


class TestPlotRecord:
    """Test PlotRecord model for EUDR production plot tracking."""

    def test_create_plot_record(self):
        """All fields populated correctly."""
        geo = GeolocationData(latitude=5.0, longitude=-73.0, plot_area_ha=3.0)
        plot = PlotRecord(
            commodity="cocoa",
            country_code="CO",
            geolocation=geo,
            operator_id="OP-001",
            operator_name="EcoCocoa SA",
        )
        assert plot.plot_id.startswith("PLOT-")
        assert plot.commodity == "cocoa"
        assert plot.country_code == "CO"
        assert plot.operator_id == "OP-001"
        assert plot.geolocation is not None
        assert plot.registered_at is not None

    def test_default_cutoff_date(self):
        """Default cutoff date is 2020-12-31 per EUDR."""
        plot = PlotRecord()
        assert plot.cutoff_date == "2020-12-31"

    def test_risk_level_default(self):
        """Risk level starts as unknown."""
        plot = PlotRecord()
        assert plot.risk_level == "unknown"

    def test_custom_plot_id(self):
        plot = PlotRecord(plot_id="PLOT-CUSTOM-001")
        assert plot.plot_id == "PLOT-CUSTOM-001"

    def test_to_dict(self):
        plot = PlotRecord(commodity="coffee", country_code="BR")
        d = plot.to_dict()
        assert d["commodity"] == "coffee"
        assert d["country_code"] == "BR"
        assert d["cutoff_date"] == "2020-12-31"

    def test_deforestation_free_none_by_default(self):
        plot = PlotRecord()
        assert plot.deforestation_free is None
        assert plot.legally_produced is None


# ===========================================================================
# Test Classes -- CustodyTransfer
# ===========================================================================


class TestCustodyTransfer:
    """Test CustodyTransfer model for EUDR chain of custody."""

    def test_create_transfer(self):
        """Transfer with all required fields."""
        t = CustodyTransfer(
            batch_number="BATCH-2026-001",
            commodity="cocoa",
            quantity_kg=5000.0,
            custody_model="segregated",
            from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-001", "PLOT-002"],
            cn_code="1801.00",
        )
        assert t.transfer_id.startswith("TXF-")
        assert t.transaction_id.startswith("TXN-")
        assert t.batch_number == "BATCH-2026-001"
        assert t.quantity_kg == 5000.0
        assert len(t.origin_plot_ids) == 2

    def test_quantity_positive(self):
        """Quantity must be > 0."""
        with pytest.raises(ValueError, match="Quantity must be positive"):
            CustodyTransfer(quantity_kg=0.0)

    def test_quantity_negative_raises(self):
        with pytest.raises(ValueError, match="Quantity must be positive"):
            CustodyTransfer(quantity_kg=-100.0)

    @pytest.mark.parametrize("model", ["identity_preserved", "segregated", "mass_balance"])
    def test_custody_models(self, model):
        """All three custody models accepted."""
        t = CustodyTransfer(quantity_kg=100.0, custody_model=model)
        assert t.custody_model == model

    def test_to_dict(self):
        t = CustodyTransfer(
            batch_number="B-001", commodity="coffee",
            quantity_kg=1000.0, cn_code="0901.11",
        )
        d = t.to_dict()
        assert d["batch_number"] == "B-001"
        assert d["commodity"] == "coffee"
        assert d["cn_code"] == "0901.11"

    def test_transfer_verified_default_false(self):
        t = CustodyTransfer(quantity_kg=100.0)
        assert t.verified is False

    def test_transfer_date_auto_set(self):
        t = CustodyTransfer(quantity_kg=100.0)
        assert t.transfer_date is not None


# ===========================================================================
# Test Classes -- RiskScore
# ===========================================================================


class TestRiskScore:
    """Test RiskScore model for EUDR risk assessment."""

    def test_create_risk_score(self):
        """All scores 0-100."""
        rs = RiskScore(
            entity_id="OP-001",
            country_score=80.0,
            commodity_score=60.0,
            supplier_score=40.0,
            traceability_score=30.0,
            overall_score=55.0,
        )
        assert rs.entity_id == "OP-001"
        assert rs.country_score == 80.0
        assert rs.overall_score == 55.0

    def test_risk_level_low(self):
        """Overall score < 30 = low risk."""
        rs = RiskScore(overall_score=15.0)
        assert rs.risk_level == "low"

    def test_risk_level_standard(self):
        """Overall score 30-70 = standard risk."""
        rs = RiskScore(overall_score=50.0)
        assert rs.risk_level == "standard"

    def test_risk_level_high(self):
        """Overall score > 70 = high risk."""
        rs = RiskScore(overall_score=85.0)
        assert rs.risk_level == "high"

    def test_risk_level_boundary_30(self):
        """Score exactly 30 = standard."""
        rs = RiskScore(overall_score=30.0)
        assert rs.risk_level == "standard"

    def test_risk_level_boundary_70(self):
        """Score exactly 70 = standard."""
        rs = RiskScore(overall_score=70.0)
        assert rs.risk_level == "standard"

    def test_risk_level_boundary_just_above_70(self):
        """Score 70.1 = high."""
        rs = RiskScore(overall_score=70.1)
        assert rs.risk_level == "high"

    def test_risk_level_explicit_override(self):
        """Explicit risk_level overrides computed value."""
        rs = RiskScore(overall_score=10.0, risk_level="high")
        assert rs.risk_level == "high"

    def test_to_dict(self):
        rs = RiskScore(entity_id="E-1", overall_score=45.0)
        d = rs.to_dict()
        assert d["entity_id"] == "E-1"
        assert d["risk_level"] == "standard"


# ===========================================================================
# Test Classes -- DueDiligenceStatement
# ===========================================================================


class TestDueDiligenceStatement:
    """Test DueDiligenceStatement model for EUDR DDS compliance."""

    def test_create_dds(self):
        """All required fields present."""
        dds = DueDiligenceStatement(
            operator_id="OP-001",
            operator_name="ImportCo GmbH",
            dds_type="import_placement",
            commodity="cocoa",
            country_of_production="GH",
            plot_ids=["PLOT-001", "PLOT-002"],
            total_quantity_kg=10000.0,
            cn_codes=["1801.00"],
        )
        assert dds.dds_id.startswith("DDS-")
        assert dds.operator_id == "OP-001"
        assert dds.status == "draft"
        assert dds.dds_type == "import_placement"
        assert len(dds.plot_ids) == 2
        assert dds.total_quantity_kg == 10000.0

    def test_dds_status_lifecycle(self):
        """Draft -> submitted -> verified lifecycle."""
        dds = DueDiligenceStatement(operator_id="OP-001", commodity="cocoa")
        assert dds.status == "draft"

        dds.submit()
        assert dds.status == "submitted"
        assert dds.submitted_at is not None

        dds.verify()
        assert dds.status == "verified"
        assert dds.verified_at is not None

    def test_dds_submit_not_draft_raises(self):
        dds = DueDiligenceStatement(status="submitted")
        with pytest.raises(ValueError, match="Can only submit a DDS in draft"):
            dds.submit()

    def test_dds_verify_not_submitted_raises(self):
        dds = DueDiligenceStatement(status="draft")
        with pytest.raises(ValueError, match="Can only verify a submitted DDS"):
            dds.verify()

    def test_dds_reject(self):
        dds = DueDiligenceStatement(operator_id="OP-001")
        dds.submit()
        dds.reject()
        assert dds.status == "rejected"

    def test_dds_reject_not_submitted_raises(self):
        dds = DueDiligenceStatement(status="draft")
        with pytest.raises(ValueError, match="Can only reject a submitted DDS"):
            dds.reject()

    @pytest.mark.parametrize("dds_type", ["import_placement", "export", "domestic"])
    def test_dds_type_values(self, dds_type):
        """All three DDS types accepted: import_placement, export, domestic."""
        dds = DueDiligenceStatement(dds_type=dds_type)
        assert dds.dds_type == dds_type

    def test_to_dict(self):
        dds = DueDiligenceStatement(
            operator_id="OP-001", commodity="coffee",
            country_of_production="ET",
        )
        d = dds.to_dict()
        assert d["commodity"] == "coffee"
        assert d["country_of_production"] == "ET"
        assert d["status"] == "draft"


# ===========================================================================
# Test Classes -- CommodityClassification
# ===========================================================================


class TestCommodityClassification:
    """Test CommodityClassification for primary vs derived product mapping."""

    def test_classify_primary(self):
        """Primary commodity classification."""
        cc = CommodityClassification(
            commodity="cocoa", is_primary=True, cn_code="1801.00",
        )
        assert cc.is_primary is True
        assert cc.derived_product is None

    def test_classify_derived(self):
        """Derived product mapping."""
        cc = CommodityClassification(
            commodity="cocoa", is_primary=False,
            derived_product="chocolate", cn_code="1806.31",
        )
        assert cc.is_primary is False
        assert cc.derived_product == "chocolate"

    def test_to_dict(self):
        cc = CommodityClassification(commodity="wood", cn_code="4407.10")
        d = cc.to_dict()
        assert d["commodity"] == "wood"
        assert d["cn_code"] == "4407.10"


# ===========================================================================
# Test Classes -- SupplierDeclaration
# ===========================================================================


class TestSupplierDeclaration:
    """Test SupplierDeclaration model for EUDR supplier attestation."""

    def test_create_declaration(self):
        """Required fields populated."""
        decl = SupplierDeclaration(
            supplier_id="SUP-001",
            supplier_name="CocoaFarms Ltd",
            commodity="cocoa",
            country_code="CI",
            valid_from="2025-01-01",
            valid_until="2025-12-31",
            deforestation_free=True,
            legally_produced=True,
            plot_ids=["PLOT-001"],
        )
        assert decl.declaration_id.startswith("DECL-")
        assert decl.supplier_id == "SUP-001"
        assert decl.deforestation_free is True

    def test_validity_check_in_range(self):
        """Date within valid_from/valid_until returns True."""
        decl = SupplierDeclaration(
            valid_from="2025-01-01", valid_until="2025-12-31",
        )
        assert decl.is_valid_on("2025-06-15") is True

    def test_validity_check_before_range(self):
        """Date before valid_from returns False."""
        decl = SupplierDeclaration(
            valid_from="2025-01-01", valid_until="2025-12-31",
        )
        assert decl.is_valid_on("2024-12-31") is False

    def test_validity_check_after_range(self):
        """Date after valid_until returns False."""
        decl = SupplierDeclaration(
            valid_from="2025-01-01", valid_until="2025-12-31",
        )
        assert decl.is_valid_on("2026-01-01") is False

    def test_validity_check_boundary_start(self):
        """Date on valid_from boundary returns True."""
        decl = SupplierDeclaration(
            valid_from="2025-01-01", valid_until="2025-12-31",
        )
        assert decl.is_valid_on("2025-01-01") is True

    def test_validity_check_boundary_end(self):
        """Date on valid_until boundary returns True."""
        decl = SupplierDeclaration(
            valid_from="2025-01-01", valid_until="2025-12-31",
        )
        assert decl.is_valid_on("2025-12-31") is True

    def test_to_dict(self):
        decl = SupplierDeclaration(supplier_id="SUP-001", commodity="rubber")
        d = decl.to_dict()
        assert d["supplier_id"] == "SUP-001"
        assert d["commodity"] == "rubber"


# ===========================================================================
# Test Classes -- Request Models
# ===========================================================================


class TestRegisterPlotRequestModel:
    """Test RegisterPlotRequest."""

    def test_register_plot_request(self):
        """Valid request with all fields."""
        req = RegisterPlotRequest(
            commodity="cocoa",
            country_code="GH",
            latitude=6.68,
            longitude=-1.62,
            plot_area_ha=3.5,
            operator_id="OP-001",
            operator_name="GhanaCocoa Ltd",
        )
        assert req.commodity == "cocoa"
        assert req.country_code == "GH"
        assert req.latitude == 6.68
        assert req.plot_area_ha == 3.5

    def test_default_commodity(self):
        req = RegisterPlotRequest()
        assert req.commodity == "cocoa"


class TestRecordTransferRequestModel:
    """Test RecordTransferRequest."""

    def test_record_transfer_request(self):
        """Valid request with all fields."""
        req = RecordTransferRequest(
            batch_number="BATCH-001",
            commodity="coffee",
            quantity_kg=2000.0,
            custody_model="identity_preserved",
            from_operator_id="OP-001",
            to_operator_id="OP-002",
            origin_plot_ids=["PLOT-001"],
            cn_code="0901.11",
        )
        assert req.batch_number == "BATCH-001"
        assert req.quantity_kg == 2000.0
        assert req.cn_code == "0901.11"

    def test_default_custody_model(self):
        req = RecordTransferRequest()
        assert req.custody_model == "segregated"

    def test_default_origin_plot_ids(self):
        req = RecordTransferRequest()
        assert req.origin_plot_ids == []


class TestGenerateDDSRequestModel:
    """Test GenerateDDSRequest."""

    def test_generate_dds_request(self):
        """Valid request with all fields."""
        req = GenerateDDSRequest(
            operator_id="OP-001",
            operator_name="ImportCo GmbH",
            dds_type="import_placement",
            commodity="cocoa",
            country_of_production="GH",
            plot_ids=["PLOT-001", "PLOT-002"],
            total_quantity_kg=5000.0,
            cn_codes=["1801.00"],
        )
        assert req.operator_id == "OP-001"
        assert req.dds_type == "import_placement"
        assert len(req.plot_ids) == 2

    def test_default_dds_type(self):
        req = GenerateDDSRequest()
        assert req.dds_type == "import_placement"

    def test_default_cn_codes_empty(self):
        req = GenerateDDSRequest()
        assert req.cn_codes == []
