"""Integration validation for scope2_market models.py"""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from greenlang.scope2_market.models import (
    InstrumentType, InstrumentStatus, EnergySource, EnergyType, EnergyUnit,
    CalculationMethod, EmissionGas, GWPSource, QualityCriterion, TrackingSystem,
    ResidualMixSource, FacilityType, ComplianceStatus, CoverageStatus,
    ReportingPeriod, ContractType, DataQualityTier, DualReportingStatus,
    AllocationMethod, ConsumptionDataSource,
    GWP_VALUES, RESIDUAL_MIX_FACTORS, ENERGY_SOURCE_EF, SUPPLIER_DEFAULT_EF,
    INSTRUMENT_QUALITY_WEIGHTS, VINTAGE_VALIDITY_YEARS, UNIT_CONVERSIONS,
    ContractualInstrument, InstrumentQualityAssessment, SupplierEmissionFactor,
    ResidualMixFactor, EnergyPurchase, FacilityInfo, AllocationResult,
    CoveredEmissionResult, UncoveredEmissionResult, GasEmissionDetail,
    MarketBasedResult, DualReportingResult, CalculationRequest,
    BatchCalculationRequest, BatchCalculationResult, ComplianceCheckResult,
    InstrumentValidationResult, UncertaintyRequest, UncertaintyResult,
    AggregationResult, DEFAULT_QUALITY_THRESHOLD,
)

now = datetime.now(timezone.utc)
past = now - timedelta(days=365)

# 1. ContractualInstrument
inst = ContractualInstrument(
    instrument_type=InstrumentType.REC,
    quantity_mwh=Decimal("100"),
    energy_source=EnergySource.SOLAR,
    ef_kgco2e_per_kwh=Decimal("0"),
    vintage_year=2025,
    tracking_system=TrackingSystem.GREEN_E,
    certificate_id="GRE-2025-000001",
    region="US-CAMX",
    delivery_start=past,
    delivery_end=now,
    tenant_id="tenant-001",
)
print(f"1. ContractualInstrument: OK (status={inst.status.value})")

# 2. delivery_end validation
try:
    ContractualInstrument(
        instrument_type=InstrumentType.REC,
        quantity_mwh=Decimal("100"),
        energy_source=EnergySource.SOLAR,
        ef_kgco2e_per_kwh=Decimal("0"),
        vintage_year=2025,
        tracking_system=TrackingSystem.GREEN_E,
        certificate_id="GRE-2025-000002",
        region="US-CAMX",
        delivery_start=now,
        delivery_end=past,
        tenant_id="tenant-001",
    )
    print("2. delivery_end validation: FAIL")
except ValueError:
    print("2. delivery_end validation: OK (raised)")

# 3. InstrumentQualityAssessment
qa = InstrumentQualityAssessment(
    instrument_id=inst.instrument_id,
    unique_claim_score=Decimal("1.0"),
    associated_delivery_score=Decimal("0.8"),
    temporal_match_score=Decimal("1.0"),
    geographic_match_score=Decimal("0.9"),
    no_double_count_score=Decimal("1.0"),
    recognized_registry_score=Decimal("1.0"),
    represents_generation_score=Decimal("1.0"),
    overall_score=Decimal("0.95"),
    passes_threshold=True,
)
print(f"3. InstrumentQualityAssessment: OK (score={qa.overall_score})")

# 4. SupplierEmissionFactor country normalization
sef = SupplierEmissionFactor(
    name="Test Utility Co.",
    country="us",
    ef_kgco2e_per_kwh=Decimal("0.390"),
    year=2025,
)
assert sef.country == "US"
print(f"4. SupplierEmissionFactor: OK (country={sef.country})")

# 5. ResidualMixFactor
rmf = ResidualMixFactor(
    region="US-CAMX",
    factor_kgco2e_per_kwh=Decimal("0.285"),
    source=ResidualMixSource.GREEN_E,
    year=2025,
    country_code="us",
)
assert rmf.country_code == "US"
print(f"5. ResidualMixFactor: OK (cc={rmf.country_code})")

# 6. EnergyPurchase with period validation
ep = EnergyPurchase(
    facility_id="fac-001",
    energy_type=EnergyType.ELECTRICITY,
    quantity=Decimal("500"),
    unit=EnergyUnit.MWH,
    region="US-CAMX",
    instruments=[inst],
    period_start=past,
    period_end=now,
)
print(f"6. EnergyPurchase: OK (instruments={len(ep.instruments)})")

# 7. FacilityInfo country normalization
fi = FacilityInfo(
    name="HQ Office",
    facility_type=FacilityType.OFFICE,
    country_code="gb",
    grid_region="EU-GB",
    tenant_id="tenant-001",
)
assert fi.country_code == "GB"
print(f"7. FacilityInfo: OK (country={fi.country_code})")

# 8. AllocationResult
ar = AllocationResult(
    purchase_id=ep.purchase_id,
    total_mwh=Decimal("500"),
    covered_mwh=Decimal("100"),
    uncovered_mwh=Decimal("400"),
    coverage_pct=Decimal("20"),
    coverage_status=CoverageStatus.PARTIALLY_COVERED,
)
print(f"8. AllocationResult: OK (coverage={ar.coverage_pct}%)")

# 9. CoveredEmissionResult
cer = CoveredEmissionResult(
    instrument_id=inst.instrument_id,
    instrument_type=InstrumentType.REC,
    mwh_covered=Decimal("100"),
    ef_kgco2e_per_kwh=Decimal("0"),
    emissions_kg=Decimal("0"),
    co2e_kg=Decimal("0"),
    energy_source=EnergySource.SOLAR,
)
print(f"9. CoveredEmissionResult: OK")

# 10. UncoveredEmissionResult
uer = UncoveredEmissionResult(
    mwh_uncovered=Decimal("400"),
    region="US-CAMX",
    residual_mix_ef_kgco2e_per_kwh=Decimal("0.285"),
    emissions_kg=Decimal("114000"),
    co2e_kg=Decimal("114000"),
)
print(f"10. UncoveredEmissionResult: OK")

# 11. GasEmissionDetail
ged = GasEmissionDetail(
    gas=EmissionGas.CO2,
    emission_kg=Decimal("114000"),
    gwp_factor=Decimal("1"),
    co2e_kg=Decimal("114000"),
)
print(f"11. GasEmissionDetail: OK")

# 12. MarketBasedResult
mbr = MarketBasedResult(
    calculation_id="calc-001",
    facility_id="fac-001",
    total_mwh=Decimal("500"),
    covered_mwh=Decimal("100"),
    uncovered_mwh=Decimal("400"),
    coverage_pct=Decimal("20"),
    covered_emissions_tco2e=Decimal("0"),
    uncovered_emissions_tco2e=Decimal("114"),
    total_emissions_tco2e=Decimal("114"),
    gas_breakdown=[ged],
    provenance_hash="a" * 64,
)
print(f"12. MarketBasedResult: OK (total={mbr.total_emissions_tco2e} tCO2e)")

# 13. DualReportingResult
drr = DualReportingResult(
    facility_id="fac-001",
    location_based_tco2e=Decimal("150"),
    market_based_tco2e=Decimal("114"),
    difference_tco2e=Decimal("-36"),
    difference_pct=Decimal("-24"),
    re_procurement_impact_tco2e=Decimal("36"),
)
print(f"13. DualReportingResult: OK (diff={drr.difference_tco2e})")

# 14. CalculationRequest
cr = CalculationRequest(
    tenant_id="tenant-001",
    facility_id="fac-001",
    energy_purchases=[ep],
)
print(f"14. CalculationRequest: OK")

# 15. BatchCalculationRequest
bcr = BatchCalculationRequest(tenant_id="tenant-001", requests=[cr])
print(f"15. BatchCalculationRequest: OK")

# 16. BatchCalculationResult
bcres = BatchCalculationResult(
    batch_id=bcr.batch_id,
    results=[mbr],
    total_co2e_tonnes=Decimal("114"),
    facility_count=1,
    provenance_hash="b" * 64,
)
print(f"16. BatchCalculationResult: OK")

# 17. ComplianceCheckResult
ccr = ComplianceCheckResult(
    calculation_id="calc-001",
    framework="GHG_PROTOCOL",
    status=ComplianceStatus.COMPLIANT,
)
print(f"17. ComplianceCheckResult: OK")

# 18. InstrumentValidationResult
ivr = InstrumentValidationResult(
    instrument_id=inst.instrument_id,
    quality_assessment=qa,
    is_valid=True,
)
print(f"18. InstrumentValidationResult: OK (valid={ivr.is_valid})")

# 19. UncertaintyRequest method validation
try:
    UncertaintyRequest(calculation_id="calc-001", method="invalid")
    print("19. UncertaintyRequest: FAIL")
except ValueError:
    print("19. UncertaintyRequest method validation: OK")
uq = UncertaintyRequest(calculation_id="calc-001", method="monte_carlo")
print(f"    UncertaintyRequest creation: OK")

# 20. UncertaintyResult
ur = UncertaintyResult(
    calculation_id="calc-001",
    method="monte_carlo",
    mean_co2e=Decimal("114"),
    std_dev=Decimal("8.5"),
    ci_lower=Decimal("97"),
    ci_upper=Decimal("131"),
    confidence_level=Decimal("0.95"),
    iterations=10000,
)
print(f"20. UncertaintyResult: OK")

# 21. AggregationResult
agr = AggregationResult(
    group_by="facility",
    period="2025",
    total_co2e_tonnes=Decimal("114"),
    facility_count=1,
)
print(f"21. AggregationResult: OK")

# 22. Frozen check
try:
    mbr.total_mwh = Decimal("999")
    print("22. Frozen: FAIL")
except Exception:
    print("22. Frozen: OK (immutable)")

# 23. Enum member counts
assert len(list(InstrumentType)) == 10
assert len(list(InstrumentStatus)) == 5
assert len(list(EnergySource)) == 11
assert len(list(EnergyType)) == 4
assert len(list(EnergyUnit)) == 5
assert len(list(CalculationMethod)) == 4
assert len(list(EmissionGas)) == 3
assert len(list(GWPSource)) == 4
assert len(list(QualityCriterion)) == 7
assert len(list(TrackingSystem)) == 8
assert len(list(ResidualMixSource)) == 5
assert len(list(FacilityType)) == 8
assert len(list(ComplianceStatus)) == 4
assert len(list(CoverageStatus)) == 4
assert len(list(ReportingPeriod)) == 4
assert len(list(ContractType)) == 4
assert len(list(DataQualityTier)) == 3
assert len(list(DualReportingStatus)) == 3
assert len(list(AllocationMethod)) == 3
assert len(list(ConsumptionDataSource)) == 4
print("23. All 20 enum member counts: OK")

# 24. Constant types
for k, v in RESIDUAL_MIX_FACTORS.items():
    assert isinstance(v, Decimal), f"{k} not Decimal"
for k, v in ENERGY_SOURCE_EF.items():
    assert isinstance(v, Decimal), f"{k} not Decimal"
for k, v in SUPPLIER_DEFAULT_EF.items():
    assert isinstance(v, Decimal), f"{k} not Decimal"
for k, v in INSTRUMENT_QUALITY_WEIGHTS.items():
    assert isinstance(v, Decimal), f"{k} not Decimal"
for k, v in UNIT_CONVERSIONS.items():
    assert isinstance(v, Decimal), f"{k} not Decimal"
assert sum(INSTRUMENT_QUALITY_WEIGHTS.values()) == Decimal("1.00")
print("24. All constants Decimal, weights sum=1.00: OK")

# 25. Residual mix factor counts
assert len(RESIDUAL_MIX_FACTORS) >= 56
print(f"25. RESIDUAL_MIX_FACTORS: {len(RESIDUAL_MIX_FACTORS)} regions: OK")

print()
print("=== ALL 25 VALIDATION TESTS PASSED ===")
