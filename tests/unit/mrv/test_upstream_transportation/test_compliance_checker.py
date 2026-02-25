"""
Unit tests for ComplianceCheckerEngine.

Tests regulatory framework compliance checking for upstream transportation
including GHG Protocol, ISO 14083, GLEC, CSRD, CDP, SBTi, GRI.

Tests:
- Framework-specific compliance checks (GHG Protocol, ISO 14083, GLEC, etc.)
- Boundary definitions (DPP, EXW, FCA, etc.)
- Incoterms classification
- Mode coverage requirements
- WTW/TTW scope requirements
- Transport chain completeness
- Allocation method consistency
- Double counting prevention
- Reefer/warehousing inclusion
- Data quality minimums
- Overall compliance scoring
- Recommendations generation
"""

import pytest
from decimal import Decimal
from typing import Dict, List, Any

from greenlang.mrv.upstream_transportation.engines.compliance_checker import (
    ComplianceCheckerEngine,
    ComplianceInput,
    ComplianceResult,
    FrameworkRequirement,
    ComplianceStatus,
    RegulatoryFramework,
)
from greenlang.mrv.upstream_transportation.models import (
    TransportMode,
    EmissionScope,
    DataQualityTier,
    AllocationMethod,
)


@pytest.fixture
def engine():
    """Create ComplianceCheckerEngine instance."""
    return ComplianceCheckerEngine()


@pytest.fixture
def ghg_protocol_input():
    """GHG Protocol compliant input."""
    return ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",  # Delivered Place Paid
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        includes_reefer=False,
        includes_warehousing=True,
        total_co2e_kg=Decimal("15000"),
        transport_chain_complete=True,
    )


@pytest.fixture
def iso_14083_input():
    """ISO 14083 compliant input (WTW required)."""
    return ComplianceInput(
        framework=RegulatoryFramework.ISO_14083,
        scope=EmissionScope.WTW,  # ISO 14083 requires WTW
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.AIR],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        includes_reefer=True,
        includes_warehousing=True,
        total_co2e_kg=Decimal("25000"),
        transport_chain_complete=True,
    )


# ============================================================================
# GHG Protocol Compliance
# ============================================================================


def test_check_ghg_protocol_compliant(engine, ghg_protocol_input):
    """Test GHG Protocol compliant input passes."""
    result = engine.check_compliance(ghg_protocol_input)

    assert isinstance(result, ComplianceResult)
    assert result.framework == RegulatoryFramework.GHG_PROTOCOL
    assert result.status == ComplianceStatus.COMPLIANT
    assert result.score >= 0.9  # High compliance score
    assert len(result.issues) == 0


def test_check_ghg_protocol_missing_boundary(engine):
    """Test GHG Protocol fails without boundary definition."""
    invalid_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary=None,  # Missing boundary
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
    )

    result = engine.check_compliance(invalid_input)

    assert result.status in [ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIAL]
    assert any("boundary" in issue.lower() for issue in result.issues)


# ============================================================================
# ISO 14083 Compliance
# ============================================================================


def test_check_iso_14083_wtw_required(engine, iso_14083_input):
    """Test ISO 14083 requires WTW scope."""
    result = engine.check_compliance(iso_14083_input)

    assert result.status == ComplianceStatus.COMPLIANT
    assert iso_14083_input.scope == EmissionScope.WTW


def test_check_iso_14083_ttw_fails(engine):
    """Test ISO 14083 fails with TTW scope."""
    ttw_input = ComplianceInput(
        framework=RegulatoryFramework.ISO_14083,
        scope=EmissionScope.TTW,  # TTW not allowed for ISO 14083
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
    )

    result = engine.check_compliance(ttw_input)

    assert result.status == ComplianceStatus.NON_COMPLIANT
    assert any("wtw" in issue.lower() for issue in result.issues)


# ============================================================================
# GLEC Framework Compliance
# ============================================================================


def test_check_glec_framework(engine):
    """Test GLEC Framework compliance check."""
    glec_input = ComplianceInput(
        framework=RegulatoryFramework.GLEC,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME, TransportMode.AIR],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_1,  # GLEC prefers Tier 1
        includes_warehousing=True,
        total_co2e_kg=Decimal("20000"),
        transport_chain_complete=True,
        glec_methodology_version="3.0",
    )

    result = engine.check_compliance(glec_input)

    assert result.status == ComplianceStatus.COMPLIANT
    assert result.framework == RegulatoryFramework.GLEC


# ============================================================================
# CSRD ESRS E1 Compliance
# ============================================================================


def test_check_csrd_esrs_e1(engine):
    """Test CSRD ESRS E1 compliance check."""
    csrd_input = ComplianceInput(
        framework=RegulatoryFramework.CSRD,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        includes_warehousing=True,
        total_co2e_kg=Decimal("18000"),
        transport_chain_complete=True,
        double_counting_checked=True,
    )

    result = engine.check_compliance(csrd_input)

    assert result.status == ComplianceStatus.COMPLIANT


# ============================================================================
# CDP Compliance
# ============================================================================


def test_check_cdp(engine):
    """Test CDP compliance check."""
    cdp_input = ComplianceInput(
        framework=RegulatoryFramework.CDP,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.AIR, TransportMode.MARITIME],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        includes_warehousing=True,
        total_co2e_kg=Decimal("30000"),
        disclosure_quality="high",
    )

    result = engine.check_compliance(cdp_input)

    assert result.status == ComplianceStatus.COMPLIANT


# ============================================================================
# SBTi Compliance
# ============================================================================


def test_check_sbti(engine):
    """Test SBTi compliance check."""
    sbti_input = ComplianceInput(
        framework=RegulatoryFramework.SBTI,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        includes_warehousing=True,
        total_co2e_kg=Decimal("22000"),
        sbti_target_coverage=True,
    )

    result = engine.check_compliance(sbti_input)

    assert result.status == ComplianceStatus.COMPLIANT


# ============================================================================
# GRI 305 Compliance
# ============================================================================


def test_check_gri_305(engine):
    """Test GRI 305 compliance check."""
    gri_input = ComplianceInput(
        framework=RegulatoryFramework.GRI_305,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_2,
        total_co2e_kg=Decimal("8000"),
        gri_disclosure_complete=True,
    )

    result = engine.check_compliance(gri_input)

    assert result.status == ComplianceStatus.COMPLIANT


# ============================================================================
# Multi-Framework Compliance
# ============================================================================


def test_check_all_frameworks(engine):
    """Test checking compliance against all frameworks."""
    comprehensive_input = ComplianceInput(
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME, TransportMode.AIR],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_1,
        includes_reefer=True,
        includes_warehousing=True,
        total_co2e_kg=Decimal("35000"),
        transport_chain_complete=True,
        double_counting_checked=True,
    )

    frameworks = [
        RegulatoryFramework.GHG_PROTOCOL,
        RegulatoryFramework.ISO_14083,
        RegulatoryFramework.GLEC,
        RegulatoryFramework.CSRD,
    ]

    results = {}
    for framework in frameworks:
        comprehensive_input.framework = framework
        results[framework] = engine.check_compliance(comprehensive_input)

    # All should be compliant
    assert all(r.status == ComplianceStatus.COMPLIANT for r in results.values())


# ============================================================================
# Payment Boundary (Incoterms)
# ============================================================================


def test_payment_boundary_dpp_cat4(engine):
    """Test DPP boundary → Category 4 (Upstream Transportation)."""
    classification = engine.classify_payment_boundary(
        incoterm="DPP",  # Delivered Place Paid
        seller_perspective=True,
    )

    assert classification["ghg_protocol_category"] == "Category 4"
    assert classification["boundary_type"] == "upstream_transportation"


def test_payment_boundary_exw_cat9(engine):
    """Test EXW boundary → Category 9 (Downstream Transportation)."""
    classification = engine.classify_payment_boundary(
        incoterm="EXW",  # Ex Works
        seller_perspective=True,
    )

    assert classification["ghg_protocol_category"] == "Category 9"
    assert classification["boundary_type"] == "downstream_transportation"


# ============================================================================
# Incoterms Classification
# ============================================================================


def test_incoterms_classification(engine):
    """Test Incoterms classification for emissions responsibility."""
    incoterms = ["EXW", "FCA", "FOB", "CFR", "CIF", "DAP", "DDP"]

    classifications = {
        term: engine.classify_incoterm(term) for term in incoterms
    }

    # EXW: buyer pays all transport
    assert classifications["EXW"]["buyer_transport_responsibility"] > 0.9

    # DDP: seller pays all transport
    assert classifications["DDP"]["seller_transport_responsibility"] > 0.9

    # FOB: split responsibility
    assert 0.3 < classifications["FOB"]["seller_transport_responsibility"] < 0.7


# ============================================================================
# Mode Coverage
# ============================================================================


def test_mode_coverage_all_modes(engine):
    """Test mode coverage check with all modes."""
    input_all_modes = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[
            TransportMode.ROAD,
            TransportMode.RAIL,
            TransportMode.MARITIME,
            TransportMode.AIR,
        ],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("50000"),
    )

    result = engine.check_compliance(input_all_modes)

    # All major modes covered
    assert result.mode_coverage_complete is True


# ============================================================================
# WTW/TTW Requirements
# ============================================================================


def test_wtw_mandatory_iso_14083(engine):
    """Test ISO 14083 mandates WTW scope."""
    requirements = engine.get_framework_requirements(RegulatoryFramework.ISO_14083)

    assert requirements.scope_required == EmissionScope.WTW
    assert requirements.wtw_mandatory is True


# ============================================================================
# Transport Chain Completeness
# ============================================================================


def test_transport_chain_completeness(engine):
    """Test transport chain completeness check."""
    complete_chain = {
        "origin": "Factory",
        "destination": "Customer",
        "legs": [
            {"mode": TransportMode.ROAD, "distance_km": 100},
            {"mode": TransportMode.MARITIME, "distance_km": 8000},
            {"mode": TransportMode.ROAD, "distance_km": 150},
        ],
        "hubs": [
            {"type": "port", "location": "Port A"},
            {"type": "port", "location": "Port B"},
        ],
    }

    is_complete, gaps = engine.check_chain_completeness(complete_chain)

    assert is_complete is True
    assert len(gaps) == 0


# ============================================================================
# Allocation Method Consistency
# ============================================================================


def test_allocation_method_consistency(engine):
    """Test allocation method consistency across legs."""
    chain = {
        "legs": [
            {"allocation_method": AllocationMethod.MASS},
            {"allocation_method": AllocationMethod.MASS},
            {"allocation_method": AllocationMethod.MASS},
        ]
    }

    is_consistent, issues = engine.check_allocation_consistency(chain)

    assert is_consistent is True
    assert len(issues) == 0


# ============================================================================
# Double Counting Prevention
# ============================================================================


def test_double_counting_cat1(engine):
    """Test double counting check for Category 1 (Purchased Goods)."""
    inventory = {
        "cat1_purchased_goods": {
            "includes_upstream_transport": True,
            "co2e_kg": Decimal("100000"),
        },
        "cat4_upstream_transport": {
            "co2e_kg": Decimal("15000"),
        },
    }

    has_double_counting, warnings = engine.check_double_counting(inventory)

    # Warning: Cat 1 may already include transport
    assert has_double_counting is True
    assert len(warnings) > 0


def test_double_counting_cat3_wtw(engine):
    """Test double counting check for Category 3 WTW."""
    inventory = {
        "cat3_fuel_energy": {
            "scope": EmissionScope.WTW,
            "co2e_kg": Decimal("50000"),
        },
        "cat4_upstream_transport": {
            "fuel_upstream_emissions": Decimal("5000"),  # Already in Cat 3 WTW
        },
    }

    has_double_counting, warnings = engine.check_double_counting(inventory)

    # Warning: WTW in Cat 3 includes fuel upstream
    assert has_double_counting is True


def test_double_counting_cat3_ttw(engine):
    """Test no double counting with Category 3 TTW."""
    inventory = {
        "cat3_fuel_energy": {
            "scope": EmissionScope.TTW,  # TTW only
            "co2e_kg": Decimal("50000"),
        },
        "cat4_upstream_transport": {
            "fuel_upstream_emissions": Decimal("5000"),  # OK to include separately
        },
    }

    has_double_counting, warnings = engine.check_double_counting(inventory)

    # No double counting: Cat 3 TTW doesn't include upstream
    assert has_double_counting is False


def test_double_counting_cat9(engine):
    """Test double counting check for Category 9 (Downstream)."""
    inventory = {
        "cat4_upstream_transport": {
            "incoterm": "DPP",
            "co2e_kg": Decimal("15000"),
        },
        "cat9_downstream_transport": {
            "incoterm": "DPP",  # Same incoterm - potential overlap
            "co2e_kg": Decimal("8000"),
        },
    }

    has_double_counting, warnings = engine.check_double_counting(inventory)

    # Warning: Same incoterm in both categories
    assert has_double_counting is True


# ============================================================================
# Reefer Inclusion
# ============================================================================


def test_reefer_inclusion_food_company(engine):
    """Test reefer emissions required for food companies."""
    food_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.MARITIME],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("20000"),
        industry_sector="food_beverage",
        includes_reefer=False,  # Missing reefer
    )

    result = engine.check_compliance(food_input)

    # Should have warning about missing reefer
    assert any("refrigerat" in issue.lower() for issue in result.warnings)


# ============================================================================
# Warehousing Inclusion
# ============================================================================


def test_warehousing_inclusion_3pl(engine):
    """Test warehousing emissions required for 3PL."""
    logistics_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("10000"),
        industry_sector="logistics_3pl",
        includes_warehousing=False,  # Missing warehousing
    )

    result = engine.check_compliance(logistics_input)

    # Should have warning about missing warehousing
    assert any("warehous" in issue.lower() for issue in result.warnings)


# ============================================================================
# Data Quality Minimum
# ============================================================================


def test_data_quality_minimum(engine):
    """Test data quality minimum requirements."""
    low_quality_input = ComplianceInput(
        framework=RegulatoryFramework.ISO_14083,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        data_quality_tier=DataQualityTier.TIER_3,  # Spend-based (low)
        total_co2e_kg=Decimal("5000"),
    )

    result = engine.check_compliance(low_quality_input)

    # Warning: ISO 14083 prefers Tier 1/2
    assert any("data quality" in w.lower() for w in result.warnings)


# ============================================================================
# Overall Compliance Score
# ============================================================================


def test_overall_compliance_score(engine, ghg_protocol_input):
    """Test overall compliance score calculation."""
    result = engine.check_compliance(ghg_protocol_input)

    # Score 0-1
    assert 0.0 <= result.score <= 1.0

    # Compliant should have high score
    if result.status == ComplianceStatus.COMPLIANT:
        assert result.score >= 0.9


# ============================================================================
# Compliance Summary
# ============================================================================


def test_compliance_summary(engine, ghg_protocol_input):
    """Test compliance summary generation."""
    result = engine.check_compliance(ghg_protocol_input)
    summary = engine.generate_summary(result)

    assert "framework" in summary
    assert "status" in summary
    assert "score" in summary
    assert "issues" in summary
    assert "recommendations" in summary


# ============================================================================
# Recommendations
# ============================================================================


def test_get_recommendations(engine):
    """Test recommendation generation for non-compliant input."""
    non_compliant_input = ComplianceInput(
        framework=RegulatoryFramework.ISO_14083,
        scope=EmissionScope.TTW,  # Should be WTW
        boundary=None,  # Missing
        modes_included=[TransportMode.ROAD],
        allocation_method=None,  # Missing
        data_quality_tier=DataQualityTier.TIER_3,  # Low
        total_co2e_kg=Decimal("5000"),
        transport_chain_complete=False,
    )

    result = engine.check_compliance(non_compliant_input)
    recommendations = result.recommendations

    assert len(recommendations) > 0
    # Should recommend WTW, boundary, allocation method
    assert any("wtw" in r.lower() for r in recommendations)
    assert any("boundary" in r.lower() for r in recommendations)
    assert any("allocation" in r.lower() for r in recommendations)


# ============================================================================
# Framework Score Range
# ============================================================================


def test_framework_score_range_0_to_1(engine):
    """Test framework compliance score is 0-1."""
    inputs = [
        ComplianceInput(
            framework=RegulatoryFramework.GHG_PROTOCOL,
            scope=EmissionScope.WTW,
            boundary="DPP",
            modes_included=[TransportMode.ROAD],
            allocation_method=AllocationMethod.MASS,
            total_co2e_kg=Decimal("5000"),
        ),
        ComplianceInput(
            framework=RegulatoryFramework.ISO_14083,
            scope=EmissionScope.TTW,  # Non-compliant
            boundary=None,
            modes_included=[TransportMode.ROAD],
            total_co2e_kg=Decimal("5000"),
        ),
    ]

    for inp in inputs:
        result = engine.check_compliance(inp)
        assert 0.0 <= result.score <= 1.0


# ============================================================================
# Critical Issues
# ============================================================================


def test_critical_issue_fails_framework(engine):
    """Test critical issue causes non-compliance."""
    critical_input = ComplianceInput(
        framework=RegulatoryFramework.ISO_14083,
        scope=EmissionScope.TTW,  # Critical: ISO 14083 requires WTW
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
    )

    result = engine.check_compliance(critical_input)

    assert result.status == ComplianceStatus.NON_COMPLIANT
    assert len(result.critical_issues) > 0


# ============================================================================
# Framework Independence
# ============================================================================


def test_all_frameworks_independent(engine):
    """Test framework checks are independent."""
    base_input = ComplianceInput(
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
    )

    frameworks = [
        RegulatoryFramework.GHG_PROTOCOL,
        RegulatoryFramework.ISO_14083,
        RegulatoryFramework.GLEC,
    ]

    results = []
    for framework in frameworks:
        base_input.framework = framework
        results.append(engine.check_compliance(base_input))

    # Different frameworks may have different results
    # At minimum, all should return valid results
    assert all(isinstance(r, ComplianceResult) for r in results)


# ============================================================================
# Edge Cases
# ============================================================================


def test_unknown_framework_raises(engine):
    """Test unknown framework raises error."""
    invalid_input = ComplianceInput(
        framework="UNKNOWN_FRAMEWORK",
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
    )

    with pytest.raises(ValueError, match="unknown framework"):
        engine.check_compliance(invalid_input)


def test_missing_required_fields_partial(engine):
    """Test missing required fields gives partial compliance."""
    partial_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary=None,  # Missing
        modes_included=[TransportMode.ROAD],
        allocation_method=None,  # Missing
        total_co2e_kg=Decimal("5000"),
    )

    result = engine.check_compliance(partial_input)

    assert result.status == ComplianceStatus.PARTIAL


def test_zero_emissions_compliant(engine):
    """Test zero emissions can still be compliant (e.g., data issue)."""
    zero_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("0"),  # Zero
        transport_chain_complete=True,
    )

    result = engine.check_compliance(zero_input)

    # Should have warning but can be compliant from methodology perspective
    assert len(result.warnings) > 0


def test_very_high_emissions_no_limit(engine):
    """Test very high emissions don't fail compliance (no upper limit)."""
    high_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD, TransportMode.AIR],
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("10000000"),  # 10,000 tonnes
        transport_chain_complete=True,
    )

    result = engine.check_compliance(high_input)

    # High emissions don't affect compliance
    assert result.status == ComplianceStatus.COMPLIANT


def test_partial_mode_coverage_warning(engine):
    """Test partial mode coverage issues warning."""
    partial_coverage_input = ComplianceInput(
        framework=RegulatoryFramework.GHG_PROTOCOL,
        scope=EmissionScope.WTW,
        boundary="DPP",
        modes_included=[TransportMode.ROAD],  # Only road, missing maritime/air
        allocation_method=AllocationMethod.MASS,
        total_co2e_kg=Decimal("5000"),
        expected_modes=[TransportMode.ROAD, TransportMode.MARITIME],  # Expected both
    )

    result = engine.check_compliance(partial_coverage_input)

    # Should warn about missing maritime
    assert len(result.warnings) > 0


def test_allocation_method_varies_by_leg_acceptable(engine):
    """Test varying allocation methods by leg is acceptable."""
    chain = {
        "legs": [
            {"allocation_method": AllocationMethod.MASS},
            {"allocation_method": AllocationMethod.VOLUME},  # Different but justified
            {"allocation_method": AllocationMethod.TEU},  # Container leg
        ]
    }

    is_consistent, issues = engine.check_allocation_consistency(
        chain, allow_variation=True
    )

    # Variation allowed if justified
    assert is_consistent is True or len(issues) == 0
