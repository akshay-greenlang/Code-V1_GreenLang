"""Verification script for climate_hazard/models.py"""
import sys
import types

# Block gis_connector imports so stubs are used
sys.modules["greenlang.gis_connector"] = types.ModuleType("x")
sys.modules["greenlang.gis_connector.spatial_analyzer"] = types.ModuleType("x")
sys.modules["greenlang.gis_connector.boundary_resolver"] = types.ModuleType("x")
sys.modules["greenlang.gis_connector.crs_transformer"] = types.ModuleType("x")

from greenlang.climate_hazard import models as mod

# Verify models (14)
models = [
    mod.Location, mod.HazardSource, mod.HazardDataRecord, mod.HazardEvent,
    mod.RiskIndex, mod.ScenarioProjection, mod.Asset, mod.ExposureResult,
    mod.SensitivityProfile, mod.AdaptiveCapacityProfile, mod.VulnerabilityScore,
    mod.CompoundHazard, mod.ComplianceReport, mod.PipelineRun,
]
print(f"SDK Models: {len(models)}")
for m in models:
    print(f"  {m.__name__}: {len(m.model_fields)} fields")

# Verify request models (8)
requests = [
    mod.RegisterSourceRequest, mod.IngestDataRequest, mod.CalculateRiskRequest,
    mod.ProjectScenarioRequest, mod.RegisterAssetRequest, mod.AssessExposureRequest,
    mod.ScoreVulnerabilityRequest, mod.GenerateReportRequest,
]
print(f"Request Models: {len(requests)}")
for r in requests:
    print(f"  {r.__name__}: {len(r.model_fields)} fields")

# Layer 1 stubs
print(f"SpatialAnalyzerEngine available: {mod._SAE_AVAILABLE}")
print(f"BoundaryResolverEngine available: {mod._BRE_AVAILABLE}")
print(f"CRSTransformerEngine available: {mod._CTE_AVAILABLE}")
print(f"__all__ length: {len(mod.__all__)}")

# Instantiation tests
loc = mod.Location(latitude=51.5074, longitude=-0.1278)
print(f"Location OK: lat={loc.latitude}, lon={loc.longitude}")

asset = mod.Asset(name="London HQ", asset_type=mod.AssetType.FACILITY, location=loc)
print(f"Asset OK: {asset.name}, type={asset.asset_type.value}")

risk = mod.RiskIndex(
    hazard_type=mod.HazardType.RIVERINE_FLOOD, location=loc,
    risk_score=45.0, risk_level=mod.RiskLevel.MEDIUM,
)
print(f"RiskIndex OK: score={risk.risk_score}, level={risk.risk_level.value}")

proj = mod.ScenarioProjection(
    hazard_type=mod.HazardType.EXTREME_HEAT, location=loc,
    scenario=mod.Scenario.SSP2_4_5, time_horizon=mod.TimeHorizon.MID_TERM,
)
print(f"ScenarioProjection OK: scenario={proj.scenario.value}")

vuln = mod.VulnerabilityScore(
    entity_id="asset-1", hazard_type=mod.HazardType.DROUGHT,
    exposure_score=0.7, sensitivity_score=0.6,
    adaptive_capacity_score=0.3, vulnerability_score=68.0,
    vulnerability_level=mod.VulnerabilityLevel.HIGH,
)
print(f"VulnerabilityScore OK: score={vuln.vulnerability_score}")

compound = mod.CompoundHazard(
    primary_hazard=mod.HazardType.EXTREME_PRECIPITATION,
    secondary_hazards=[mod.HazardType.LANDSLIDE],
    correlation_factor=0.75, amplification_factor=1.5,
)
print(f"CompoundHazard OK: primary={compound.primary_hazard.value}")

event = mod.HazardEvent(
    hazard_type=mod.HazardType.TROPICAL_CYCLONE, location=loc,
    start_date=mod._utcnow(), intensity=185.0, deaths=12,
    economic_loss_usd=5e9,
)
print(f"HazardEvent OK: type={event.hazard_type.value}, deaths={event.deaths}")

pipeline = mod.PipelineRun(status="running", stages_completed=["ingest", "risk"])
print(f"PipelineRun OK: status={pipeline.status}")

source = mod.HazardSource(
    name="Aqueduct Floods", source_type=mod.DataSourceType.GLOBAL_DATABASE,
    hazard_types=[mod.HazardType.RIVERINE_FLOOD, mod.HazardType.COASTAL_FLOOD],
)
print(f"HazardSource OK: {source.name}")

record = mod.HazardDataRecord(
    source_id="src-1", hazard_type=mod.HazardType.DROUGHT,
    location=loc, intensity=3.5, probability=0.1,
)
print(f"HazardDataRecord OK: intensity={record.intensity}")

exposure = mod.ExposureResult(
    asset_id="a-1", hazard_type=mod.HazardType.WILDFIRE,
    exposure_level=mod.ExposureLevel.HIGH, composite_score=72.5,
)
print(f"ExposureResult OK: level={exposure.exposure_level.value}")

sensitivity = mod.SensitivityProfile(
    entity_id="a-1", factors={"water_dependency": 0.8},
    overall_sensitivity=mod.SensitivityLevel.HIGH,
)
print(f"SensitivityProfile OK: {sensitivity.overall_sensitivity.value}")

capacity = mod.AdaptiveCapacityProfile(
    entity_id="a-1", indicators={"insurance": 0.9},
    overall_capacity=mod.AdaptiveCapacity.HIGH,
)
print(f"AdaptiveCapacityProfile OK: {capacity.overall_capacity.value}")

report = mod.ComplianceReport(framework="tcfd", title="Q4 Physical Risk")
print(f"ComplianceReport OK: framework={report.framework}")

# Request models
reg_src = mod.RegisterSourceRequest(name="ERA5", source_type=mod.DataSourceType.REANALYSIS)
print(f"RegisterSourceRequest OK")

ingest = mod.IngestDataRequest(source_id="src-1", batch_size=500)
print(f"IngestDataRequest OK")

calc_risk = mod.CalculateRiskRequest(
    hazard_type=mod.HazardType.EXTREME_HEAT, location=loc,
)
print(f"CalculateRiskRequest OK")

proj_req = mod.ProjectScenarioRequest(
    hazard_type=mod.HazardType.SEA_LEVEL_RISE, location=loc,
    scenario=mod.Scenario.SSP5_8_5, time_horizon=mod.TimeHorizon.END_CENTURY,
)
print(f"ProjectScenarioRequest OK")

reg_asset = mod.RegisterAssetRequest(
    name="Plant A", asset_type=mod.AssetType.FACILITY, location=loc,
)
print(f"RegisterAssetRequest OK")

assess_exp = mod.AssessExposureRequest(
    asset_id="a-1", hazard_type=mod.HazardType.COASTAL_FLOOD,
)
print(f"AssessExposureRequest OK")

score_vuln = mod.ScoreVulnerabilityRequest(
    entity_id="a-1", hazard_type=mod.HazardType.WATER_STRESS,
)
print(f"ScoreVulnerabilityRequest OK")

gen_report = mod.GenerateReportRequest(
    report_type=mod.ReportType.SCENARIO_ANALYSIS, framework="csrd_esrs",
)
print(f"GenerateReportRequest OK")

# Validation tests
print()
print("--- Validation Tests ---")

errors = 0

try:
    mod.Location(latitude=91.0, longitude=0.0)
    print("FAIL: lat=91 accepted"); errors += 1
except Exception:
    print("Validation OK: lat=91 rejected")

try:
    mod.Location(latitude=0.0, longitude=181.0)
    print("FAIL: lon=181 accepted"); errors += 1
except Exception:
    print("Validation OK: lon=181 rejected")

try:
    mod.Asset(name="", asset_type=mod.AssetType.FACILITY, location=loc)
    print("FAIL: empty name accepted"); errors += 1
except Exception:
    print("Validation OK: empty name rejected")

try:
    mod.PipelineRun(status="invalid_status")
    print("FAIL: invalid status accepted"); errors += 1
except Exception:
    print("Validation OK: invalid status rejected")

try:
    mod.Location(latitude=0, longitude=0, extra_field="bad")
    print("FAIL: extra field accepted"); errors += 1
except Exception:
    print("Extra forbid OK: extra field rejected")

try:
    mod.RiskIndex(
        hazard_type=mod.HazardType.DROUGHT, location=loc, risk_score=101.0,
    )
    print("FAIL: risk_score=101 accepted"); errors += 1
except Exception:
    print("Validation OK: risk_score=101 rejected")

try:
    mod.HazardDataRecord(
        source_id="", hazard_type=mod.HazardType.DROUGHT, location=loc,
    )
    print("FAIL: empty source_id accepted"); errors += 1
except Exception:
    print("Validation OK: empty source_id rejected")

try:
    mod.Location(latitude=40.0, longitude=-74.0, country_code="us")
    print("FAIL: lowercase country_code accepted"); errors += 1
except Exception:
    print("Validation OK: lowercase country_code rejected")

try:
    mod.CompoundHazard(
        primary_hazard=mod.HazardType.DROUGHT, correlation_factor=1.5,
    )
    print("FAIL: correlation_factor=1.5 accepted"); errors += 1
except Exception:
    print("Validation OK: correlation_factor=1.5 rejected")

try:
    mod.VulnerabilityScore(
        entity_id="", hazard_type=mod.HazardType.DROUGHT,
    )
    print("FAIL: empty entity_id accepted"); errors += 1
except Exception:
    print("Validation OK: empty entity_id rejected")

print()
if errors == 0:
    print("ALL CHECKS PASSED")
else:
    print(f"FAILED: {errors} checks")
    sys.exit(1)
