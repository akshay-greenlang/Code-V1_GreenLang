"""Quick functional test for SupplierSpecificCalculatorEngine."""
import sys
sys.path.insert(0, ".")

from decimal import Decimal
from datetime import date

from greenlang.agents.mrv.fuel_energy_activities.supplier_specific_calculator import (
    SupplierSpecificCalculatorEngine,
)
from greenlang.agents.mrv.fuel_energy_activities.models import (
    FuelConsumptionRecord, ElectricityConsumptionRecord,
    SupplierFuelData, FuelType, FuelCategory,
    EnergyType, AccountingMethod, GridRegionType,
    SupplierDataSource, AllocationMethod,
    Activity3aResult,
)

engine = SupplierSpecificCalculatorEngine()

# --- Test 1: assess_miq_grade ---
print("=== Test 1: assess_miq_grade ===")
assert engine.assess_miq_grade(Decimal("0.03")) == "A"
assert engine.assess_miq_grade(Decimal("0.10")) == "B"
assert engine.assess_miq_grade(Decimal("0.40")) == "C"
assert engine.assess_miq_grade(Decimal("1.0")) == "D"
assert engine.assess_miq_grade(Decimal("2.5")) == "E"
assert engine.assess_miq_grade(Decimal("5.0")) == "F"
print("  PASS")

# --- Test 2: get_miq_upstream_adjustment ---
print("=== Test 2: get_miq_upstream_adjustment ===")
assert engine.get_miq_upstream_adjustment("A") == Decimal("0.15")
assert engine.get_miq_upstream_adjustment("F") == Decimal("1.30")
print("  PASS")

# --- Test 3: assess_ogmp_level ---
print("=== Test 3: assess_ogmp_level ===")
assert engine.assess_ogmp_level({}) == 1
assert engine.assess_ogmp_level({"engineering_estimates": True}) == 2
assert engine.assess_ogmp_level({"source_measurement": True}) == 3
assert engine.assess_ogmp_level({"site_measurement": True}) == 4
assert engine.assess_ogmp_level({
    "external_verification": True,
    "reconciliation": True,
    "site_measurement": True,
}) == 5
print("  PASS")

# --- Test 4: calculate_ppa_upstream ---
print("=== Test 4: calculate_ppa_upstream ===")
ppa_supplier = SupplierFuelData(
    supplier_id="PPA-001", supplier_name="SolarCo",
    fuel_type=FuelType.NATURAL_GAS, upstream_ef=Decimal("0"),
    data_source=SupplierDataSource.PPA,
)
ppa_ef = engine.calculate_ppa_upstream(ppa_supplier, "onshore_wind")
assert ppa_ef == Decimal("0.00100000"), f"Got {ppa_ef}"
print(f"  PPA onshore_wind EF: {ppa_ef} -- PASS")

# --- Test 5: validate_supplier_data ---
print("=== Test 5: validate_supplier_data ===")
valid_supplier = SupplierFuelData(
    supplier_id="SUP-001", supplier_name="Acme Gas",
    fuel_type=FuelType.NATURAL_GAS, upstream_ef=Decimal("0.01850"),
    verification_level="third_party_verified", miq_grade="A",
    reporting_year=2025,
)
is_valid, issues = engine.validate_supplier_data(valid_supplier)
assert is_valid, f"Expected valid, got issues: {issues}"
print(f"  Valid supplier: is_valid={is_valid}, issues={len(issues)} -- PASS")

# --- Test 6: validate_epd ---
print("=== Test 6: validate_epd ===")
epd = {
    "epd_number": "EPD-12345",
    "programme_operator": "EPD International",
    "product_category": "Natural Gas",
    "declared_unit": "kWh",
    "upstream_ef": Decimal("0.020"),
    "verification_body": "Bureau Veritas",
}
is_valid_epd, epd_issues = engine.validate_epd(epd)
assert is_valid_epd, f"EPD validation failed: {epd_issues}"
print(f"  EPD valid: {is_valid_epd}, issues={len(epd_issues)} -- PASS")

# --- Test 7: calculate_fuel_upstream ---
print("=== Test 7: calculate_fuel_upstream ===")
fuel_rec = FuelConsumptionRecord(
    fuel_type=FuelType.NATURAL_GAS,
    quantity=Decimal("10000"),
    unit="kWh",
    quantity_kwh=Decimal("10000"),
    period_start=date(2025, 1, 1),
    period_end=date(2025, 12, 31),
    reporting_year=2025,
    supplier_id="SUP-001",
)
supplier = SupplierFuelData(
    supplier_id="SUP-001", supplier_name="Acme Gas",
    fuel_type=FuelType.NATURAL_GAS, upstream_ef=Decimal("0.02000"),
    verification_level="third_party_verified",
)
result_3a = engine.calculate_fuel_upstream(fuel_rec, supplier, "AR6")
assert result_3a.emissions_total > Decimal("0")
expected = Decimal("200.00000000")
assert result_3a.emissions_total == expected, f"Got {result_3a.emissions_total}"
print(f"  Emissions: {result_3a.emissions_total} kgCO2e -- PASS")

# --- Test 8: calculate_fuel_upstream with MiQ grade ---
print("=== Test 8: calculate_fuel_upstream with MiQ grade ===")
supplier_miq = SupplierFuelData(
    supplier_id="SUP-002", supplier_name="CleanGas",
    fuel_type=FuelType.NATURAL_GAS, upstream_ef=Decimal("0.02000"),
    verification_level="certified", miq_grade="A",
)
result_miq = engine.calculate_fuel_upstream(fuel_rec, supplier_miq, "AR6")
expected_miq = Decimal("30.00000000")
assert result_miq.emissions_total == expected_miq, f"Got {result_miq.emissions_total}"
print(f"  MiQ Grade A emissions: {result_miq.emissions_total} kgCO2e -- PASS")

# --- Test 9: calculate_electricity_upstream ---
print("=== Test 9: calculate_electricity_upstream ===")
elec_rec = ElectricityConsumptionRecord(
    energy_type=EnergyType.ELECTRICITY,
    quantity_kwh=Decimal("50000"),
    grid_region="US",
    period_start=date(2025, 1, 1),
    period_end=date(2025, 12, 31),
    reporting_year=2025,
    supplier_id="ELEC-001",
)
elec_supplier = SupplierFuelData(
    supplier_id="ELEC-001", supplier_name="GridPower",
    fuel_type=FuelType.NATURAL_GAS, upstream_ef=Decimal("0.03500"),
    verification_level="self_declared",
    data_source=SupplierDataSource.EPD,
)
result_3b = engine.calculate_electricity_upstream(elec_rec, elec_supplier, "AR6")
assert result_3b.emissions_total == Decimal("1750.00000000"), f"Got {result_3b.emissions_total}"
print(f"  Electricity upstream: {result_3b.emissions_total} kgCO2e -- PASS")

# --- Test 10: allocate_supplier_emissions ---
print("=== Test 10: allocate_supplier_emissions ===")
allocated = engine.allocate_supplier_emissions(
    total_emissions=Decimal("100000"),
    allocation_method=AllocationMethod.REVENUE,
    allocation_data={
        "entity_share": Decimal("5000000"),
        "total": Decimal("20000000"),
    }
)
assert allocated == Decimal("25000.00000000"), f"Got {allocated}"
print(f"  Allocated: {allocated} kgCO2e -- PASS")

# --- Test 11: assess_coverage ---
print("=== Test 11: assess_coverage ===")
all_records = [
    FuelConsumptionRecord(
        fuel_type=FuelType.NATURAL_GAS, quantity=Decimal("10000"),
        unit="kWh", quantity_kwh=Decimal("10000"),
        period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
        reporting_year=2025, supplier_id="SUP-001",
    ),
    FuelConsumptionRecord(
        fuel_type=FuelType.DIESEL, quantity=Decimal("10000"),
        unit="kWh", quantity_kwh=Decimal("10000"),
        period_start=date(2025, 1, 1), period_end=date(2025, 12, 31),
        reporting_year=2025,
    ),
]
supplier_records = [all_records[0]]
coverage = engine.assess_coverage(supplier_records, all_records)
assert coverage == Decimal("50.00000000"), f"Got {coverage}"
print(f"  Coverage: {coverage}% -- PASS")

# --- Test 12: assess_verification_level ---
print("=== Test 12: assess_verification_level ===")
v_score = engine.assess_verification_level(valid_supplier)
assert v_score == Decimal("1.0"), f"Got {v_score}"
print(f"  Verification score: {v_score} -- PASS")

# --- Test 13: compare_with_average ---
print("=== Test 13: compare_with_average ===")
avg_result = Activity3aResult(
    fuel_record_id="REC-001", fuel_type=FuelType.NATURAL_GAS,
    fuel_consumed_kwh=Decimal("10000"), wtt_ef_total=Decimal("0.0246"),
    emissions_total=Decimal("246.0"), dqi_score=Decimal("3.0"),
    uncertainty_pct=Decimal("25.0"),
)
comparison = engine.compare_with_average(result_3a, avg_result)
assert "absolute_difference_kgco2e" in comparison
assert "recommendation" in comparison
print(f"  Recommendation: {comparison['recommendation']} -- PASS")

# --- Test 14: assess_dqi ---
print("=== Test 14: assess_dqi ===")
dqi = engine.assess_dqi(valid_supplier)
assert Decimal("1.0") <= dqi.composite <= Decimal("5.0")
assert dqi.tier in ("Very High", "High", "Medium", "Low", "Very Low")
print(f"  DQI composite: {dqi.composite}, tier: {dqi.tier} -- PASS")

# --- Test 15: quantify_uncertainty ---
print("=== Test 15: quantify_uncertainty ===")
unc = engine.quantify_uncertainty(valid_supplier, "analytical")
assert unc.ci_lower <= unc.mean <= unc.ci_upper
assert unc.confidence_level == Decimal("95.0")
print(f"  Uncertainty: mean={unc.mean} CI=[{unc.ci_lower}, {unc.ci_upper}] -- PASS")

# --- Test 16: aggregate_by_supplier ---
print("=== Test 16: aggregate_by_supplier ===")
agg = engine.aggregate_by_supplier([result_3a, result_miq])
assert len(agg) > 0
total_agg = sum(agg.values())
assert total_agg == result_3a.emissions_total + result_miq.emissions_total
print(f"  Aggregation: {len(agg)} suppliers, total={total_agg} -- PASS")

# --- Test 17: configure and reset ---
print("=== Test 17: configure and reset ===")
engine.configure({"decimal_places": 6})
engine.reset()
stats = engine.get_statistics()
assert all(v == 0 for v in stats.values())
print("  Reset stats: all zero -- PASS")

# --- Test 18: blend_with_average ---
print("=== Test 18: blend_with_average ===")
engine2 = SupplierSpecificCalculatorEngine()
blended = engine2.blend_with_average(
    supplier_results=[result_3a],
    average_results=[avg_result],
    coverage_pct=Decimal("60"),
)
assert "blended_total_kgco2e" in blended
assert blended["blended_total_kgco2e"] > Decimal("0")
print(f"  Blended total: {blended['blended_total_kgco2e']} -- PASS")

# --- Test 19: batch calculation ---
print("=== Test 19: calculate_batch ===")
batch_results = engine2.calculate_batch(
    records=[fuel_rec, elec_rec],
    supplier_data_map={
        "SUP-001": supplier,
        "ELEC-001": elec_supplier,
    },
)
assert len(batch_results) == 2
print(f"  Batch results: {len(batch_results)} results -- PASS")

# --- Final stats ---
print()
print("=== Final Statistics ===")
final_stats = engine2.get_statistics()
for k, v in sorted(final_stats.items()):
    print(f"  {k}: {v}")

print()
print("ALL 19 TESTS PASSED")
