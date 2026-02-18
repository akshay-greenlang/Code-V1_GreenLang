# -*- coding: utf-8 -*-
"""
Unit tests for Refrigerants & F-Gas Agent Data Models - AGENT-MRV-002

Tests all 15 enumerations and 14 Pydantic data models for the
Refrigerants & F-Gas Agent SDK. Validates enum member counts,
string values, model construction, validation constraints, default
values, optional fields, serialization, and error paths.

Target: 150+ tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from greenlang.refrigerants_fgas.models import (
    # Constants
    VERSION,
    MAX_CALCULATIONS_PER_BATCH,
    MAX_GASES_PER_RESULT,
    MAX_TRACE_STEPS,
    MAX_BLEND_COMPONENTS,
    MAX_EQUIPMENT_PROFILES,
    MAX_SERVICE_EVENTS,
    GWP_REFERENCE,
    # Enums
    RefrigerantCategory,
    RefrigerantType,
    GWPSource,
    GWPTimeframe,
    CalculationMethod,
    EquipmentType,
    EquipmentStatus,
    ServiceEventType,
    LifecycleStage,
    CalculationStatus,
    ReportingPeriod,
    RegulatoryFramework,
    ComplianceStatus,
    PhaseDownSchedule,
    UnitType,
    # Models
    GWPValue,
    BlendComponent,
    RefrigerantProperties,
    EquipmentProfile,
    ServiceEvent,
    LeakRateProfile,
    CalculationInput,
    MassBalanceData,
    GasEmission,
    CalculationResult,
    BatchCalculationRequest,
    BatchCalculationResponse,
    UncertaintyResult,
    ComplianceRecord,
)


# ===================================================================
# Constants tests
# ===================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_version(self):
        assert VERSION == "1.0.0"

    def test_max_calculations_per_batch(self):
        assert MAX_CALCULATIONS_PER_BATCH == 10_000

    def test_max_gases_per_result(self):
        assert MAX_GASES_PER_RESULT == 20

    def test_max_trace_steps(self):
        assert MAX_TRACE_STEPS == 200

    def test_max_blend_components(self):
        assert MAX_BLEND_COMPONENTS == 10

    def test_max_equipment_profiles(self):
        assert MAX_EQUIPMENT_PROFILES == 1_000

    def test_max_service_events(self):
        assert MAX_SERVICE_EVENTS == 10_000

    def test_gwp_reference_has_ar4(self):
        assert "AR4" in GWP_REFERENCE

    def test_gwp_reference_has_ar5(self):
        assert "AR5" in GWP_REFERENCE

    def test_gwp_reference_has_ar6(self):
        assert "AR6" in GWP_REFERENCE

    def test_gwp_reference_co2_always_1(self):
        for ar in GWP_REFERENCE.values():
            assert ar["CO2"] == 1.0

    def test_gwp_reference_sf6_ar6(self):
        assert GWP_REFERENCE["AR6"]["SF6"] == 25200.0


# ===================================================================
# Enum: RefrigerantCategory (10 values)
# ===================================================================


class TestRefrigerantCategory:
    """Test RefrigerantCategory enum completeness and values."""

    def test_member_count(self):
        assert len(RefrigerantCategory) == 10

    @pytest.mark.parametrize("member,value", [
        (RefrigerantCategory.HFC, "hfc"),
        (RefrigerantCategory.HFC_BLEND, "hfc_blend"),
        (RefrigerantCategory.HFO, "hfo"),
        (RefrigerantCategory.PFC, "pfc"),
        (RefrigerantCategory.SF6, "sf6"),
        (RefrigerantCategory.NF3, "nf3"),
        (RefrigerantCategory.HCFC, "hcfc"),
        (RefrigerantCategory.CFC, "cfc"),
        (RefrigerantCategory.NATURAL, "natural"),
        (RefrigerantCategory.OTHER, "other"),
    ])
    def test_category_value(self, member, value):
        assert member.value == value

    def test_is_str_enum(self):
        assert isinstance(RefrigerantCategory.HFC, str)

    def test_lookup_by_value(self):
        assert RefrigerantCategory("hfc") == RefrigerantCategory.HFC


# ===================================================================
# Enum: RefrigerantType (52 values)
# ===================================================================


class TestRefrigerantType:
    """Test RefrigerantType enum completeness and category groupings."""

    def test_total_member_count(self):
        assert len(RefrigerantType) >= 52

    # HFC pure substances
    @pytest.mark.parametrize("member", [
        "R_32", "R_125", "R_134A", "R_143A", "R_152A",
        "R_227EA", "R_236FA", "R_245FA", "R_365MFC", "R_23", "R_41",
    ])
    def test_hfc_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # HFC Blends
    @pytest.mark.parametrize("member", [
        "R_404A", "R_407A", "R_407C", "R_407F", "R_410A",
        "R_413A", "R_417A", "R_422D", "R_427A", "R_438A",
        "R_448A", "R_449A", "R_452A", "R_454B", "R_507A", "R_508B",
    ])
    def test_hfc_blend_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # HFOs
    @pytest.mark.parametrize("member", [
        "R_1234YF", "R_1234ZE", "R_1233ZD", "R_1336MZZ",
    ])
    def test_hfo_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # PFCs
    @pytest.mark.parametrize("member", [
        "CF4", "C2F6", "C3F8", "C_C4F8", "C4F10", "C5F12", "C6F14",
    ])
    def test_pfc_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # SF6 and NF3
    @pytest.mark.parametrize("member", ["SF6_GAS", "NF3_GAS", "SO2F2"])
    def test_sf6_nf3_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # HCFCs
    @pytest.mark.parametrize("member", ["R_22", "R_123", "R_141B", "R_142B"])
    def test_hcfc_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # CFCs
    @pytest.mark.parametrize("member", [
        "R_11", "R_12", "R_113", "R_114", "R_115", "R_502",
    ])
    def test_cfc_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    # Natural
    @pytest.mark.parametrize("member", ["R_717", "R_744", "R_290", "R_600A"])
    def test_natural_members_exist(self, member: str):
        assert hasattr(RefrigerantType, member)

    def test_custom_member_exists(self):
        assert hasattr(RefrigerantType, "CUSTOM")

    def test_is_str_enum(self):
        assert isinstance(RefrigerantType.R_410A, str)

    def test_r410a_value(self):
        assert RefrigerantType.R_410A.value == "R_410A"


# ===================================================================
# Enum: GWPSource (5 values)
# ===================================================================


class TestGWPSource:
    """Test GWPSource enum."""

    def test_member_count(self):
        assert len(GWPSource) == 5

    @pytest.mark.parametrize("member,value", [
        (GWPSource.AR4, "AR4"),
        (GWPSource.AR5, "AR5"),
        (GWPSource.AR6, "AR6"),
        (GWPSource.AR6_20YR, "AR6_20YR"),
        (GWPSource.CUSTOM, "CUSTOM"),
    ])
    def test_gwp_source_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: GWPTimeframe (2 values)
# ===================================================================


class TestGWPTimeframe:
    """Test GWPTimeframe enum."""

    def test_member_count(self):
        assert len(GWPTimeframe) == 2

    def test_20yr_value(self):
        assert GWPTimeframe.GWP_20YR.value == "GWP_20YR"

    def test_100yr_value(self):
        assert GWPTimeframe.GWP_100YR.value == "GWP_100YR"


# ===================================================================
# Enum: CalculationMethod (5 values)
# ===================================================================


class TestCalculationMethod:
    """Test CalculationMethod enum."""

    def test_member_count(self):
        assert len(CalculationMethod) == 5

    @pytest.mark.parametrize("member,value", [
        (CalculationMethod.EQUIPMENT_BASED, "EQUIPMENT_BASED"),
        (CalculationMethod.MASS_BALANCE, "MASS_BALANCE"),
        (CalculationMethod.SCREENING, "SCREENING"),
        (CalculationMethod.DIRECT_MEASUREMENT, "DIRECT_MEASUREMENT"),
        (CalculationMethod.TOP_DOWN, "TOP_DOWN"),
    ])
    def test_method_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: EquipmentType (15 values)
# ===================================================================


class TestEquipmentType:
    """Test EquipmentType enum."""

    def test_member_count(self):
        assert len(EquipmentType) == 15

    @pytest.mark.parametrize("member,value", [
        (EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED, "commercial_refrigeration_centralized"),
        (EquipmentType.COMMERCIAL_REFRIGERATION_STANDALONE, "commercial_refrigeration_standalone"),
        (EquipmentType.INDUSTRIAL_REFRIGERATION, "industrial_refrigeration"),
        (EquipmentType.RESIDENTIAL_AC, "residential_ac"),
        (EquipmentType.COMMERCIAL_AC, "commercial_ac"),
        (EquipmentType.CHILLERS_CENTRIFUGAL, "chillers_centrifugal"),
        (EquipmentType.CHILLERS_SCREW, "chillers_screw"),
        (EquipmentType.HEAT_PUMPS, "heat_pumps"),
        (EquipmentType.TRANSPORT_REFRIGERATION, "transport_refrigeration"),
        (EquipmentType.SWITCHGEAR, "switchgear"),
        (EquipmentType.SEMICONDUCTOR, "semiconductor"),
        (EquipmentType.FIRE_SUPPRESSION, "fire_suppression"),
        (EquipmentType.FOAM_BLOWING, "foam_blowing"),
        (EquipmentType.AEROSOLS, "aerosols"),
        (EquipmentType.SOLVENTS, "solvents"),
    ])
    def test_equipment_type_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: EquipmentStatus (4 values)
# ===================================================================


class TestEquipmentStatus:
    """Test EquipmentStatus enum."""

    def test_member_count(self):
        assert len(EquipmentStatus) == 4

    @pytest.mark.parametrize("member,value", [
        (EquipmentStatus.ACTIVE, "active"),
        (EquipmentStatus.INACTIVE, "inactive"),
        (EquipmentStatus.DECOMMISSIONED, "decommissioned"),
        (EquipmentStatus.MAINTENANCE, "maintenance"),
    ])
    def test_status_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: ServiceEventType (7 values)
# ===================================================================


class TestServiceEventType:
    """Test ServiceEventType enum."""

    def test_member_count(self):
        assert len(ServiceEventType) == 7

    @pytest.mark.parametrize("member,value", [
        (ServiceEventType.INSTALLATION, "installation"),
        (ServiceEventType.RECHARGE, "recharge"),
        (ServiceEventType.REPAIR, "repair"),
        (ServiceEventType.RECOVERY, "recovery"),
        (ServiceEventType.LEAK_CHECK, "leak_check"),
        (ServiceEventType.DECOMMISSIONING, "decommissioning"),
        (ServiceEventType.CONVERSION, "conversion"),
    ])
    def test_event_type_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: LifecycleStage (3 values)
# ===================================================================


class TestLifecycleStage:
    """Test LifecycleStage enum."""

    def test_member_count(self):
        assert len(LifecycleStage) == 3

    @pytest.mark.parametrize("member,value", [
        (LifecycleStage.INSTALLATION, "installation"),
        (LifecycleStage.OPERATING, "operating"),
        (LifecycleStage.END_OF_LIFE, "end_of_life"),
    ])
    def test_lifecycle_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: CalculationStatus (5 values)
# ===================================================================


class TestCalculationStatus:
    """Test CalculationStatus enum."""

    def test_member_count(self):
        assert len(CalculationStatus) == 5

    @pytest.mark.parametrize("member,value", [
        (CalculationStatus.PENDING, "pending"),
        (CalculationStatus.RUNNING, "running"),
        (CalculationStatus.COMPLETED, "completed"),
        (CalculationStatus.FAILED, "failed"),
        (CalculationStatus.CANCELLED, "cancelled"),
    ])
    def test_status_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: ReportingPeriod (3 values)
# ===================================================================


class TestReportingPeriod:
    """Test ReportingPeriod enum."""

    def test_member_count(self):
        assert len(ReportingPeriod) == 3

    @pytest.mark.parametrize("member,value", [
        (ReportingPeriod.MONTHLY, "monthly"),
        (ReportingPeriod.QUARTERLY, "quarterly"),
        (ReportingPeriod.ANNUAL, "annual"),
    ])
    def test_period_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: RegulatoryFramework (9 values)
# ===================================================================


class TestRegulatoryFramework:
    """Test RegulatoryFramework enum."""

    def test_member_count(self):
        assert len(RegulatoryFramework) == 9

    @pytest.mark.parametrize("member,value", [
        (RegulatoryFramework.GHG_PROTOCOL, "ghg_protocol"),
        (RegulatoryFramework.ISO_14064, "iso_14064"),
        (RegulatoryFramework.CSRD_ESRS_E1, "csrd_esrs_e1"),
        (RegulatoryFramework.EPA_40CFR98_DD, "epa_40cfr98_dd"),
        (RegulatoryFramework.EPA_40CFR98_OO, "epa_40cfr98_oo"),
        (RegulatoryFramework.EPA_40CFR98_L, "epa_40cfr98_l"),
        (RegulatoryFramework.EU_FGAS_2024_573, "eu_fgas_2024_573"),
        (RegulatoryFramework.KIGALI_AMENDMENT, "kigali_amendment"),
        (RegulatoryFramework.UK_FGAS, "uk_fgas"),
    ])
    def test_framework_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: ComplianceStatus (5 values)
# ===================================================================


class TestComplianceStatus:
    """Test ComplianceStatus enum."""

    def test_member_count(self):
        assert len(ComplianceStatus) == 5

    @pytest.mark.parametrize("member,value", [
        (ComplianceStatus.COMPLIANT, "compliant"),
        (ComplianceStatus.WARNING, "warning"),
        (ComplianceStatus.NON_COMPLIANT, "non_compliant"),
        (ComplianceStatus.EXEMPTED, "exempted"),
        (ComplianceStatus.NOT_APPLICABLE, "not_applicable"),
    ])
    def test_compliance_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: PhaseDownSchedule (3 values)
# ===================================================================


class TestPhaseDownSchedule:
    """Test PhaseDownSchedule enum."""

    def test_member_count(self):
        assert len(PhaseDownSchedule) == 3

    @pytest.mark.parametrize("member,value", [
        (PhaseDownSchedule.EU_FGAS, "eu_fgas"),
        (PhaseDownSchedule.KIGALI_A5, "kigali_a5"),
        (PhaseDownSchedule.KIGALI_NON_A5, "kigali_non_a5"),
    ])
    def test_schedule_value(self, member, value):
        assert member.value == value


# ===================================================================
# Enum: UnitType (6 values)
# ===================================================================


class TestUnitType:
    """Test UnitType enum."""

    def test_member_count(self):
        assert len(UnitType) == 6

    @pytest.mark.parametrize("member,value", [
        (UnitType.KG, "kg"),
        (UnitType.LB, "lb"),
        (UnitType.OZ, "oz"),
        (UnitType.GRAM, "gram"),
        (UnitType.TONNE, "tonne"),
        (UnitType.METRIC_TON, "metric_ton"),
    ])
    def test_unit_value(self, member, value):
        assert member.value == value


# ===================================================================
# Model: GWPValue
# ===================================================================


class TestGWPValue:
    """Test GWPValue Pydantic model."""

    def test_valid_construction(self):
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=2088.0)
        assert gwp.gwp_source == GWPSource.AR6
        assert gwp.value == 2088.0

    def test_default_timeframe(self):
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=100.0)
        assert gwp.timeframe == GWPTimeframe.GWP_100YR

    def test_custom_timeframe(self):
        gwp = GWPValue(
            gwp_source=GWPSource.AR6_20YR,
            timeframe=GWPTimeframe.GWP_20YR,
            value=4340.0,
        )
        assert gwp.timeframe == GWPTimeframe.GWP_20YR

    def test_effective_date_none_by_default(self):
        gwp = GWPValue(gwp_source=GWPSource.AR5, value=1.0)
        assert gwp.effective_date is None

    def test_effective_date_set(self):
        dt = datetime(2021, 8, 9, tzinfo=timezone.utc)
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=1.0, effective_date=dt)
        assert gwp.effective_date == dt

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError):
            GWPValue(gwp_source=GWPSource.AR6, value=-1.0)

    def test_zero_value_allowed(self):
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=0.0)
        assert gwp.value == 0.0

    def test_serialization_roundtrip(self):
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=2088.0)
        d = gwp.model_dump()
        gwp2 = GWPValue(**d)
        assert gwp2.value == gwp.value


# ===================================================================
# Model: BlendComponent
# ===================================================================


class TestBlendComponent:
    """Test BlendComponent model and weight_fraction validation."""

    def test_valid_construction(self):
        bc = BlendComponent(
            refrigerant_type=RefrigerantType.R_32,
            weight_fraction=0.5,
            gwp=675.0,
        )
        assert bc.weight_fraction == 0.5
        assert bc.gwp == 675.0

    def test_weight_fraction_at_1(self):
        bc = BlendComponent(
            refrigerant_type=RefrigerantType.R_134A,
            weight_fraction=1.0,
        )
        assert bc.weight_fraction == 1.0

    def test_weight_fraction_small(self):
        bc = BlendComponent(
            refrigerant_type=RefrigerantType.R_125,
            weight_fraction=0.001,
        )
        assert bc.weight_fraction == 0.001

    def test_weight_fraction_zero_raises(self):
        with pytest.raises(ValidationError):
            BlendComponent(
                refrigerant_type=RefrigerantType.R_32,
                weight_fraction=0.0,
            )

    def test_weight_fraction_negative_raises(self):
        with pytest.raises(ValidationError):
            BlendComponent(
                refrigerant_type=RefrigerantType.R_32,
                weight_fraction=-0.1,
            )

    def test_weight_fraction_above_1_raises(self):
        with pytest.raises(ValidationError):
            BlendComponent(
                refrigerant_type=RefrigerantType.R_32,
                weight_fraction=1.01,
            )

    def test_gwp_none_by_default(self):
        bc = BlendComponent(
            refrigerant_type=RefrigerantType.R_32,
            weight_fraction=0.5,
        )
        assert bc.gwp is None

    def test_negative_gwp_raises(self):
        with pytest.raises(ValidationError):
            BlendComponent(
                refrigerant_type=RefrigerantType.R_32,
                weight_fraction=0.5,
                gwp=-10.0,
            )


# ===================================================================
# Model: RefrigerantProperties
# ===================================================================


class TestRefrigerantProperties:
    """Test RefrigerantProperties model."""

    def _make_r134a(self, **overrides) -> RefrigerantProperties:
        defaults = {
            "refrigerant_type": RefrigerantType.R_134A,
            "category": RefrigerantCategory.HFC,
            "name": "R-134a (1,1,1,2-Tetrafluoroethane)",
            "formula": "CH2FCF3",
            "molecular_weight": 102.03,
            "boiling_point_c": -26.3,
            "odp": 0.0,
            "atmospheric_lifetime_years": 14.0,
        }
        defaults.update(overrides)
        return RefrigerantProperties(**defaults)

    def test_valid_construction(self):
        props = self._make_r134a()
        assert props.refrigerant_type == RefrigerantType.R_134A

    def test_default_values(self):
        props = self._make_r134a()
        assert props.is_blend is False
        assert props.is_regulated is True
        assert props.phase_out_date is None
        assert props.gwp_values == {}
        assert props.blend_components is None

    def test_with_gwp_values(self):
        gwp = GWPValue(gwp_source=GWPSource.AR6, value=1430.0)
        props = self._make_r134a(gwp_values={"AR6_100yr": gwp})
        assert "AR6_100yr" in props.gwp_values
        assert props.gwp_values["AR6_100yr"].value == 1430.0

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            RefrigerantProperties(
                refrigerant_type=RefrigerantType.R_134A,
                category=RefrigerantCategory.HFC,
                name="",
            )

    def test_blend_with_components(self):
        components = [
            BlendComponent(refrigerant_type=RefrigerantType.R_32, weight_fraction=0.5, gwp=675.0),
            BlendComponent(refrigerant_type=RefrigerantType.R_125, weight_fraction=0.5, gwp=3500.0),
        ]
        props = RefrigerantProperties(
            refrigerant_type=RefrigerantType.R_410A,
            category=RefrigerantCategory.HFC_BLEND,
            name="R-410A",
            blend_components=components,
            is_blend=True,
        )
        assert props.is_blend is True
        assert len(props.blend_components) == 2

    def test_blend_weights_must_sum_to_1(self):
        components = [
            BlendComponent(refrigerant_type=RefrigerantType.R_32, weight_fraction=0.3),
            BlendComponent(refrigerant_type=RefrigerantType.R_125, weight_fraction=0.3),
        ]
        with pytest.raises(ValidationError, match="sum"):
            RefrigerantProperties(
                refrigerant_type=RefrigerantType.R_410A,
                category=RefrigerantCategory.HFC_BLEND,
                name="R-410A bad blend",
                blend_components=components,
            )

    def test_blend_weights_within_tolerance(self):
        """Weights summing to 0.995 are within 0.01 tolerance."""
        components = [
            BlendComponent(refrigerant_type=RefrigerantType.R_32, weight_fraction=0.498),
            BlendComponent(refrigerant_type=RefrigerantType.R_125, weight_fraction=0.497),
        ]
        props = RefrigerantProperties(
            refrigerant_type=RefrigerantType.R_410A,
            category=RefrigerantCategory.HFC_BLEND,
            name="R-410A approx",
            blend_components=components,
        )
        assert len(props.blend_components) == 2

    def test_negative_odp_raises(self):
        with pytest.raises(ValidationError):
            self._make_r134a(odp=-0.1)

    def test_negative_molecular_weight_raises(self):
        with pytest.raises(ValidationError):
            self._make_r134a(molecular_weight=-50.0)

    def test_serialization_roundtrip(self):
        props = self._make_r134a()
        d = props.model_dump()
        props2 = RefrigerantProperties(**d)
        assert props2.name == props.name


# ===================================================================
# Model: EquipmentProfile
# ===================================================================


class TestEquipmentProfile:
    """Test EquipmentProfile model."""

    def test_valid_construction(self, sample_equipment_profile: Dict[str, Any]):
        ep = EquipmentProfile(**sample_equipment_profile)
        assert ep.equipment_id == "eq_test_001"
        assert ep.charge_kg == 25.0

    def test_default_equipment_id_generated(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
        )
        assert ep.equipment_id.startswith("eq_")
        assert len(ep.equipment_id) > 3

    def test_default_equipment_count(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.SWITCHGEAR,
            refrigerant_type=RefrigerantType.SF6_GAS,
            charge_kg=5.0,
        )
        assert ep.equipment_count == 1

    def test_default_status_active(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.HEAT_PUMPS,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=15.0,
        )
        assert ep.status == EquipmentStatus.ACTIVE

    def test_zero_charge_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                refrigerant_type=RefrigerantType.R_410A,
                charge_kg=0.0,
            )

    def test_negative_charge_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                refrigerant_type=RefrigerantType.R_410A,
                charge_kg=-5.0,
            )

    def test_equipment_count_zero_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                refrigerant_type=RefrigerantType.R_410A,
                charge_kg=10.0,
                equipment_count=0,
            )

    def test_custom_leak_rate_boundary_0(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
            custom_leak_rate=0.0,
        )
        assert ep.custom_leak_rate == 0.0

    def test_custom_leak_rate_boundary_1(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.COMMERCIAL_AC,
            refrigerant_type=RefrigerantType.R_410A,
            charge_kg=10.0,
            custom_leak_rate=1.0,
        )
        assert ep.custom_leak_rate == 1.0

    def test_custom_leak_rate_above_1_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                refrigerant_type=RefrigerantType.R_410A,
                charge_kg=10.0,
                custom_leak_rate=1.1,
            )

    def test_custom_leak_rate_negative_raises(self):
        with pytest.raises(ValidationError):
            EquipmentProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                refrigerant_type=RefrigerantType.R_410A,
                charge_kg=10.0,
                custom_leak_rate=-0.01,
            )

    def test_optional_fields_default_none(self):
        ep = EquipmentProfile(
            equipment_type=EquipmentType.CHILLERS_CENTRIFUGAL,
            refrigerant_type=RefrigerantType.R_134A,
            charge_kg=100.0,
        )
        assert ep.installation_date is None
        assert ep.location is None
        assert ep.custom_leak_rate is None


# ===================================================================
# Model: ServiceEvent
# ===================================================================


class TestServiceEvent:
    """Test ServiceEvent model."""

    def test_valid_construction(self):
        se = ServiceEvent(
            equipment_id="eq_001",
            event_type=ServiceEventType.RECHARGE,
            date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            refrigerant_added_kg=5.0,
        )
        assert se.event_type == ServiceEventType.RECHARGE
        assert se.refrigerant_added_kg == 5.0

    def test_default_event_id_generated(self):
        se = ServiceEvent(
            equipment_id="eq_001",
            event_type=ServiceEventType.LEAK_CHECK,
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert se.event_id.startswith("svc_")

    def test_default_added_and_recovered_zero(self):
        se = ServiceEvent(
            equipment_id="eq_001",
            event_type=ServiceEventType.LEAK_CHECK,
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert se.refrigerant_added_kg == 0.0
        assert se.refrigerant_recovered_kg == 0.0

    def test_notes_default_none(self):
        se = ServiceEvent(
            equipment_id="eq_001",
            event_type=ServiceEventType.REPAIR,
            date=datetime(2025, 3, 10, tzinfo=timezone.utc),
        )
        assert se.notes is None

    def test_negative_added_kg_raises(self):
        with pytest.raises(ValidationError):
            ServiceEvent(
                equipment_id="eq_001",
                event_type=ServiceEventType.RECHARGE,
                date=datetime(2025, 1, 1, tzinfo=timezone.utc),
                refrigerant_added_kg=-1.0,
            )

    def test_negative_recovered_kg_raises(self):
        with pytest.raises(ValidationError):
            ServiceEvent(
                equipment_id="eq_001",
                event_type=ServiceEventType.RECOVERY,
                date=datetime(2025, 1, 1, tzinfo=timezone.utc),
                refrigerant_recovered_kg=-10.0,
            )

    def test_empty_equipment_id_raises(self):
        with pytest.raises(ValidationError):
            ServiceEvent(
                equipment_id="",
                event_type=ServiceEventType.RECHARGE,
                date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )

    @pytest.mark.parametrize("event_type", list(ServiceEventType))
    def test_all_event_types_accepted(self, event_type: ServiceEventType):
        se = ServiceEvent(
            equipment_id="eq_test",
            event_type=event_type,
            date=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
        assert se.event_type == event_type


# ===================================================================
# Model: LeakRateProfile
# ===================================================================


class TestLeakRateProfile:
    """Test LeakRateProfile model."""

    def test_valid_construction(self):
        lr = LeakRateProfile(
            equipment_type=EquipmentType.COMMERCIAL_REFRIGERATION_CENTRALIZED,
            base_rate=0.15,
            effective_rate=0.15,
        )
        assert lr.base_rate == 0.15

    def test_default_lifecycle_stage(self):
        lr = LeakRateProfile(
            equipment_type=EquipmentType.COMMERCIAL_AC,
            base_rate=0.10,
        )
        assert lr.lifecycle_stage == LifecycleStage.OPERATING

    def test_default_factors(self):
        lr = LeakRateProfile(
            equipment_type=EquipmentType.HEAT_PUMPS,
            base_rate=0.05,
        )
        assert lr.age_factor == 1.0
        assert lr.climate_factor == 1.0
        assert lr.ldar_factor == 1.0

    def test_base_rate_zero_allowed(self):
        lr = LeakRateProfile(
            equipment_type=EquipmentType.SWITCHGEAR,
            base_rate=0.0,
        )
        assert lr.base_rate == 0.0

    def test_base_rate_negative_raises(self):
        with pytest.raises(ValidationError):
            LeakRateProfile(
                equipment_type=EquipmentType.SWITCHGEAR,
                base_rate=-0.01,
            )

    def test_base_rate_above_1_raises(self):
        with pytest.raises(ValidationError):
            LeakRateProfile(
                equipment_type=EquipmentType.SWITCHGEAR,
                base_rate=1.01,
            )

    def test_ldar_factor_above_1_raises(self):
        with pytest.raises(ValidationError):
            LeakRateProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                base_rate=0.10,
                ldar_factor=1.1,
            )

    def test_age_factor_zero_raises(self):
        with pytest.raises(ValidationError):
            LeakRateProfile(
                equipment_type=EquipmentType.COMMERCIAL_AC,
                base_rate=0.10,
                age_factor=0.0,
            )


# ===================================================================
# Model: MassBalanceData
# ===================================================================


class TestMassBalanceData:
    """Test MassBalanceData model."""

    def test_valid_construction(self, sample_mass_balance_data: Dict[str, Any]):
        mbd = MassBalanceData(**sample_mass_balance_data)
        assert mbd.beginning_inventory_kg == 500.0
        assert mbd.ending_inventory_kg == 450.0

    def test_defaults_zero(self):
        mbd = MassBalanceData(
            refrigerant_type=RefrigerantType.R_22,
            beginning_inventory_kg=100.0,
            ending_inventory_kg=80.0,
        )
        assert mbd.purchases_kg == 0.0
        assert mbd.sales_kg == 0.0
        assert mbd.acquisitions_kg == 0.0
        assert mbd.divestitures_kg == 0.0
        assert mbd.capacity_change_kg == 0.0

    def test_negative_beginning_inventory_raises(self):
        with pytest.raises(ValidationError):
            MassBalanceData(
                refrigerant_type=RefrigerantType.R_22,
                beginning_inventory_kg=-10.0,
                ending_inventory_kg=80.0,
            )

    def test_negative_ending_inventory_raises(self):
        with pytest.raises(ValidationError):
            MassBalanceData(
                refrigerant_type=RefrigerantType.R_22,
                beginning_inventory_kg=100.0,
                ending_inventory_kg=-5.0,
            )

    def test_negative_purchases_raises(self):
        with pytest.raises(ValidationError):
            MassBalanceData(
                refrigerant_type=RefrigerantType.R_22,
                beginning_inventory_kg=100.0,
                ending_inventory_kg=80.0,
                purchases_kg=-20.0,
            )

    def test_capacity_change_can_be_negative(self):
        """capacity_change_kg has no ge=0 constraint."""
        mbd = MassBalanceData(
            refrigerant_type=RefrigerantType.R_134A,
            beginning_inventory_kg=100.0,
            ending_inventory_kg=80.0,
            capacity_change_kg=-50.0,
        )
        assert mbd.capacity_change_kg == -50.0

    def test_serialization_roundtrip(self, sample_mass_balance_data: Dict[str, Any]):
        mbd = MassBalanceData(**sample_mass_balance_data)
        d = mbd.model_dump()
        mbd2 = MassBalanceData(**d)
        assert mbd2.beginning_inventory_kg == mbd.beginning_inventory_kg


# ===================================================================
# Model: GasEmission
# ===================================================================


class TestGasEmission:
    """Test GasEmission model."""

    def _make_emission(self, **overrides) -> GasEmission:
        defaults = {
            "refrigerant_type": RefrigerantType.R_410A,
            "gas_name": "R-410A",
            "loss_kg": 5.0,
            "gwp_applied": 2088.0,
            "gwp_source": "AR6",
            "emissions_kg_co2e": 10440.0,
            "emissions_tco2e": 10.44,
        }
        defaults.update(overrides)
        return GasEmission(**defaults)

    def test_valid_construction(self):
        ge = self._make_emission()
        assert ge.emissions_tco2e == 10.44

    def test_default_not_blend_component(self):
        ge = self._make_emission()
        assert ge.is_blend_component is False

    def test_blend_component_flag(self):
        ge = self._make_emission(is_blend_component=True)
        assert ge.is_blend_component is True

    def test_negative_loss_kg_raises(self):
        with pytest.raises(ValidationError):
            self._make_emission(loss_kg=-1.0)

    def test_negative_gwp_raises(self):
        with pytest.raises(ValidationError):
            self._make_emission(gwp_applied=-100.0)

    def test_negative_emissions_tco2e_raises(self):
        with pytest.raises(ValidationError):
            self._make_emission(emissions_tco2e=-0.01)

    def test_empty_gas_name_raises(self):
        with pytest.raises(ValidationError):
            self._make_emission(gas_name="")

    def test_empty_gwp_source_raises(self):
        with pytest.raises(ValidationError):
            self._make_emission(gwp_source="")


# ===================================================================
# Model: CalculationResult
# ===================================================================


class TestCalculationResult:
    """Test CalculationResult model."""

    def test_valid_construction(self):
        cr = CalculationResult(method=CalculationMethod.EQUIPMENT_BASED)
        assert cr.method == CalculationMethod.EQUIPMENT_BASED

    def test_default_calculation_id_generated(self):
        cr = CalculationResult(method=CalculationMethod.MASS_BALANCE)
        assert cr.calculation_id.startswith("calc_")

    def test_default_values(self):
        cr = CalculationResult(method=CalculationMethod.SCREENING)
        assert cr.results == []
        assert cr.total_loss_kg == 0.0
        assert cr.total_emissions_tco2e == 0.0
        assert cr.blend_decomposition is False
        assert cr.uncertainty is None
        assert cr.provenance_hash == ""
        assert cr.calculation_trace == []

    def test_timestamp_is_utc(self):
        cr = CalculationResult(method=CalculationMethod.EQUIPMENT_BASED)
        assert cr.timestamp.tzinfo is not None

    def test_with_results(self):
        emission = GasEmission(
            refrigerant_type=RefrigerantType.R_134A,
            gas_name="R-134a",
            loss_kg=10.0,
            gwp_applied=1430.0,
            gwp_source="AR6",
            emissions_kg_co2e=14300.0,
            emissions_tco2e=14.3,
        )
        cr = CalculationResult(
            method=CalculationMethod.EQUIPMENT_BASED,
            results=[emission],
            total_loss_kg=10.0,
            total_emissions_tco2e=14.3,
        )
        assert len(cr.results) == 1
        assert cr.total_emissions_tco2e == 14.3


# ===================================================================
# Model: CalculationInput
# ===================================================================


class TestCalculationInput:
    """Test CalculationInput model."""

    def test_default_method(self):
        ci = CalculationInput()
        assert ci.calculation_method == CalculationMethod.EQUIPMENT_BASED

    def test_default_gwp_source(self):
        ci = CalculationInput()
        assert ci.gwp_source == GWPSource.AR6

    def test_default_gwp_timeframe(self):
        ci = CalculationInput()
        assert ci.gwp_timeframe == GWPTimeframe.GWP_100YR

    def test_equipment_based_with_profiles(self, sample_equipment_profile: Dict[str, Any]):
        ep = EquipmentProfile(**sample_equipment_profile)
        ci = CalculationInput(equipment_profiles=[ep])
        assert len(ci.equipment_profiles) == 1

    def test_mass_balance_with_data(self, sample_mass_balance_data: Dict[str, Any]):
        mbd = MassBalanceData(**sample_mass_balance_data)
        ci = CalculationInput(
            calculation_method=CalculationMethod.MASS_BALANCE,
            mass_balance_data=[mbd],
        )
        assert len(ci.mass_balance_data) == 1

    def test_screening_with_charge(self):
        ci = CalculationInput(
            calculation_method=CalculationMethod.SCREENING,
            screening_charge_kg=100.0,
            screening_equipment_type=EquipmentType.COMMERCIAL_AC,
            screening_leak_rate=0.10,
        )
        assert ci.screening_charge_kg == 100.0

    def test_optional_fields_default_none(self):
        ci = CalculationInput()
        assert ci.equipment_profiles is None
        assert ci.mass_balance_data is None
        assert ci.screening_charge_kg is None
        assert ci.reporting_period is None
        assert ci.organization_id is None
        assert ci.tenant_id is None
        assert ci.service_events is None


# ===================================================================
# Model: BatchCalculationRequest / Response
# ===================================================================


class TestBatchCalculationRequest:
    """Test BatchCalculationRequest model."""

    def test_valid_construction(self):
        req = BatchCalculationRequest(calculations=[CalculationInput()])
        assert len(req.calculations) == 1

    def test_parallel_default_false(self):
        req = BatchCalculationRequest(calculations=[CalculationInput()])
        assert req.parallel is False

    def test_empty_calculations_raises(self):
        with pytest.raises(ValidationError):
            BatchCalculationRequest(calculations=[])

    def test_parallel_true(self):
        req = BatchCalculationRequest(
            calculations=[CalculationInput()],
            parallel=True,
        )
        assert req.parallel is True


class TestBatchCalculationResponse:
    """Test BatchCalculationResponse model."""

    def test_default_values(self):
        resp = BatchCalculationResponse()
        assert resp.results == []
        assert resp.total_emissions_tco2e == 0.0
        assert resp.success_count == 0
        assert resp.failure_count == 0
        assert resp.processing_time_ms == 0.0
        assert resp.provenance_hash == ""

    def test_with_results(self):
        cr = CalculationResult(
            method=CalculationMethod.EQUIPMENT_BASED,
            total_emissions_tco2e=10.0,
        )
        resp = BatchCalculationResponse(
            results=[cr],
            total_emissions_tco2e=10.0,
            success_count=1,
        )
        assert len(resp.results) == 1
        assert resp.success_count == 1


# ===================================================================
# Model: UncertaintyResult
# ===================================================================


class TestUncertaintyResult:
    """Test UncertaintyResult model."""

    def _make_uncertainty(self, **overrides) -> UncertaintyResult:
        defaults = {
            "method": "monte_carlo",
            "mean": 10.0,
            "std_dev": 1.5,
            "ci_90_lower": 7.5,
            "ci_90_upper": 12.5,
            "ci_95_lower": 7.0,
            "ci_95_upper": 13.0,
            "ci_99_lower": 6.0,
            "ci_99_upper": 14.0,
            "iterations": 5000,
        }
        defaults.update(overrides)
        return UncertaintyResult(**defaults)

    def test_valid_construction(self):
        ur = self._make_uncertainty()
        assert ur.method == "monte_carlo"
        assert ur.mean == 10.0

    def test_data_quality_score_none_by_default(self):
        ur = self._make_uncertainty()
        assert ur.data_quality_score is None

    def test_data_quality_score_valid(self):
        ur = self._make_uncertainty(data_quality_score=3.5)
        assert ur.data_quality_score == 3.5

    def test_data_quality_score_below_1_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(data_quality_score=0.5)

    def test_data_quality_score_above_5_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(data_quality_score=5.1)

    def test_negative_mean_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(mean=-1.0)

    def test_zero_iterations_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(iterations=0)

    def test_negative_std_dev_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(std_dev=-0.1)

    def test_empty_method_raises(self):
        with pytest.raises(ValidationError):
            self._make_uncertainty(method="")

    def test_serialization_roundtrip(self):
        ur = self._make_uncertainty(data_quality_score=4.0)
        d = ur.model_dump()
        ur2 = UncertaintyResult(**d)
        assert ur2.mean == ur.mean
        assert ur2.data_quality_score == ur.data_quality_score


# ===================================================================
# Model: ComplianceRecord
# ===================================================================


class TestComplianceRecord:
    """Test ComplianceRecord model."""

    def test_valid_construction(self):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.EU_FGAS_2024_573,
            status=ComplianceStatus.COMPLIANT,
        )
        assert cr.framework == RegulatoryFramework.EU_FGAS_2024_573

    def test_optional_fields_default_none(self):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            status=ComplianceStatus.COMPLIANT,
        )
        assert cr.quota_co2e is None
        assert cr.usage_co2e is None
        assert cr.remaining_co2e is None
        assert cr.phase_down_target_pct is None
        assert cr.notes is None

    def test_with_quota_tracking(self):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.EU_FGAS_2024_573,
            status=ComplianceStatus.WARNING,
            quota_co2e=100000.0,
            usage_co2e=85000.0,
            remaining_co2e=15000.0,
            phase_down_target_pct=45.0,
        )
        assert cr.quota_co2e == 100000.0
        assert cr.usage_co2e == 85000.0

    def test_negative_quota_raises(self):
        with pytest.raises(ValidationError):
            ComplianceRecord(
                framework=RegulatoryFramework.EU_FGAS_2024_573,
                status=ComplianceStatus.COMPLIANT,
                quota_co2e=-100.0,
            )

    def test_phase_down_target_above_100_raises(self):
        with pytest.raises(ValidationError):
            ComplianceRecord(
                framework=RegulatoryFramework.KIGALI_AMENDMENT,
                status=ComplianceStatus.COMPLIANT,
                phase_down_target_pct=101.0,
            )

    def test_phase_down_target_negative_raises(self):
        with pytest.raises(ValidationError):
            ComplianceRecord(
                framework=RegulatoryFramework.KIGALI_AMENDMENT,
                status=ComplianceStatus.COMPLIANT,
                phase_down_target_pct=-1.0,
            )

    @pytest.mark.parametrize("framework", list(RegulatoryFramework))
    def test_all_frameworks_accepted(self, framework: RegulatoryFramework):
        cr = ComplianceRecord(
            framework=framework,
            status=ComplianceStatus.NOT_APPLICABLE,
        )
        assert cr.framework == framework

    @pytest.mark.parametrize("status", list(ComplianceStatus))
    def test_all_statuses_accepted(self, status: ComplianceStatus):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            status=status,
        )
        assert cr.status == status

    def test_notes_with_text(self):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.UK_FGAS,
            status=ComplianceStatus.COMPLIANT,
            notes="Annual audit passed with no findings.",
        )
        assert "Annual audit" in cr.notes

    def test_serialization_roundtrip(self):
        cr = ComplianceRecord(
            framework=RegulatoryFramework.ISO_14064,
            status=ComplianceStatus.COMPLIANT,
            quota_co2e=50000.0,
        )
        d = cr.model_dump()
        cr2 = ComplianceRecord(**d)
        assert cr2.quota_co2e == cr.quota_co2e
