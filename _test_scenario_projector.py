#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Smoke tests for ScenarioProjectorEngine."""
import sys
import importlib.util

sys.path.insert(0, ".")

# Patch sys.modules to prevent chain imports
sys.modules["greenlang.climate_hazard.provenance"] = type(sys)("fake_prov")
sys.modules["greenlang.climate_hazard.metrics"] = type(sys)("fake_metrics")
sys.modules["greenlang.climate_hazard.config"] = type(sys)("fake_config")

spec = importlib.util.spec_from_file_location(
    "scenario_projector",
    r"greenlang\climate_hazard\scenario_projector.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ScenarioProjectorEngine = mod.ScenarioProjectorEngine

engine = ScenarioProjectorEngine(provenance=False)

# Test 1: list_scenarios
scenarios = engine.list_scenarios()
print(f"Scenarios: {len(scenarios)}")
assert len(scenarios) == 8

# Test 2: list_time_horizons
horizons = engine.list_time_horizons()
print(f"Time horizons: {len(horizons)}")
assert len(horizons) == 5

# Test 3: list_hazard_types
hazards = engine.list_hazard_types()
print(f"Hazard types: {len(hazards)}")
assert len(hazards) == 12

# Test 4: calculate_warming_delta
delta = engine.calculate_warming_delta("ssp2_4.5", "MID_TERM")
print(f"Warming delta SSP2-4.5 MID_TERM: {delta}C")
assert abs(delta - 1.485) < 0.001

delta_end = engine.calculate_warming_delta("ssp5_8.5", "END_CENTURY")
print(f"Warming delta SSP5-8.5 END_CENTURY: {delta_end}C")
assert abs(delta_end - 4.4) < 0.001

delta_base = engine.calculate_warming_delta("ssp2_4.5", "BASELINE")
print(f"Warming delta SSP2-4.5 BASELINE: {delta_base}C")
assert delta_base == 0.0

# Test 5: get_scaling_factors
factors = engine.get_scaling_factors("EXTREME_HEAT")
print(f"EXTREME_HEAT intensity_factor: {factors['intensity_factor']}")
assert factors["intensity_factor"] == 2.0
assert factors["frequency_factor"] == 1.8
assert factors["intensity_direction"] == "increasing"

# Test 6: get_scenario_info
info = engine.get_scenario_info("ssp2_4.5")
print(f"SSP2-4.5 name: {info['name']}, warming: {info['warming_by_2100']}C")
assert info["name"] == "SSP2-4.5"
assert info["warming_by_2100"] == 2.7

# Test 7: project_hazard
baseline = {
    "probability": 0.15,
    "intensity": 0.6,
    "frequency": 2.0,
    "duration_days": 5.0,
}
result = engine.project_hazard(
    hazard_type="EXTREME_HEAT",
    location={"lat": 40.7128, "lon": -74.0060, "name": "New York"},
    baseline_risk=baseline,
    scenario="ssp2_4.5",
    time_horizon="MID_TERM",
)
print(f"Projection ID: {result['projection_id']}")
print(f"Warming delta: {result['warming_delta_c']}C")
print(
    f"Baseline prob: {baseline['probability']} -> "
    f"Projected: {result['projected_risk']['probability']}"
)
print(
    f"Baseline int:  {baseline['intensity']} -> "
    f"Projected: {result['projected_risk']['intensity']}"
)
print(
    f"Baseline freq: {baseline['frequency']} -> "
    f"Projected: {result['projected_risk']['frequency']}"
)
print(
    f"Baseline dur:  {baseline['duration_days']} -> "
    f"Projected: {result['projected_risk']['duration_days']}"
)
assert result["projected_risk"]["probability"] > baseline["probability"]
assert result["projected_risk"]["intensity"] > baseline["intensity"]
assert result["projected_risk"]["frequency"] > baseline["frequency"]
assert result["projected_risk"]["duration_days"] > baseline["duration_days"]

# Test 8: get_projection
stored = engine.get_projection(result["projection_id"])
assert stored is not None
assert stored["projection_id"] == result["projection_id"]
print("Stored projection retrieved: OK")

# Test 9: project_multi_scenario
multi = engine.project_multi_scenario(
    hazard_type="DROUGHT",
    location="California",
    baseline_risk={
        "probability": 0.3,
        "intensity": 0.7,
        "frequency": 1.0,
        "duration_days": 30.0,
    },
    scenarios=["ssp1_2.6", "ssp2_4.5", "ssp5_8.5"],
    time_horizon="END_CENTURY",
)
print(f"Multi-scenario: {multi['scenarios_count']} scenarios compared")
assert multi["scenarios_count"] == 3
assert len(multi["scenario_comparison"]) == 3
assert multi["scenario_comparison"][0]["scenario"] == "ssp5_8.5"
print(f"Highest risk scenario: {multi['scenario_comparison'][0]['scenario']}")

# Test 10: project_time_series
ts = engine.project_time_series(
    hazard_type="COASTAL_FLOOD",
    location={"lat": 25.76, "lon": -80.19, "name": "Miami"},
    baseline_risk={
        "probability": 0.2,
        "intensity": 0.5,
        "frequency": 3.0,
        "duration_days": 2.0,
    },
    scenario="ssp3_7.0",
)
print(f"Time series: {ts['horizons_count']} horizons, trend={ts['trend_direction']}")
assert ts["horizons_count"] == 5
assert ts["trend_direction"] == "increasing"

# Test 11: EXTREME_COLD should decrease with warming
cold_proj = engine.project_hazard(
    hazard_type="EXTREME_COLD",
    location="Arctic",
    baseline_risk={
        "probability": 0.4,
        "intensity": 0.8,
        "frequency": 5.0,
        "duration_days": 10.0,
    },
    scenario="ssp5_8.5",
    time_horizon="END_CENTURY",
)
print(
    f"EXTREME_COLD prob: {cold_proj['projected_risk']['probability']} (should decrease)"
)
assert cold_proj["projected_risk"]["probability"] < 0.4
assert cold_proj["projected_risk"]["frequency"] < 5.0

# Test 12: TROPICAL_CYCLONE frequency should decrease
tc_proj = engine.project_hazard(
    hazard_type="TROPICAL_CYCLONE",
    location="Caribbean",
    baseline_risk={
        "probability": 0.25,
        "intensity": 0.7,
        "frequency": 4.0,
        "duration_days": 3.0,
    },
    scenario="ssp2_4.5",
    time_horizon="END_CENTURY",
)
print(
    f"TROPICAL_CYCLONE freq: {tc_proj['projected_risk']['frequency']} "
    "(should decrease from 4.0)"
)
assert tc_proj["projected_risk"]["frequency"] < 4.0
assert tc_proj["projected_risk"]["intensity"] > 0.7

# Test 13: list_projections with filters
all_projs = engine.list_projections()
print(f"Total stored projections: {len(all_projs)}")

drought_projs = engine.list_projections(hazard_type="DROUGHT")
print(f"DROUGHT projections: {len(drought_projs)}")

# Test 14: get_statistics
stats = engine.get_statistics()
print(f"Stats - total_projections: {stats['total_projections']}")
print(f"Stats - total_multi_scenario: {stats['total_multi_scenario']}")
print(f"Stats - total_time_series: {stats['total_time_series']}")

# Test 15: warming matrix
matrix = engine.get_warming_matrix()
assert matrix["ssp2_4.5"]["END_CENTURY"] == 2.7
assert matrix["ssp2_4.5"]["BASELINE"] == 0.0
print(f"Warming matrix OK, ssp2_4.5/END_CENTURY = {matrix['ssp2_4.5']['END_CENTURY']}C")

# Test 16: clear
summary = engine.clear()
print(f"Cleared: {summary['projections_cleared']} projections")
assert len(engine) == 0

# Test 17: __repr__
print(f"repr: {repr(engine)}")

# Test 18: SEA_LEVEL_RISE additive intensity
slr = engine.project_hazard(
    hazard_type="SEA_LEVEL_RISE",
    location="Maldives",
    baseline_risk={
        "probability": 0.5,
        "intensity": 0.1,
        "frequency": 1.0,
        "duration_days": 365.0,
    },
    scenario="ssp5_8.5",
    time_horizon="END_CENTURY",
)
print(f"SEA_LEVEL_RISE intensity: {slr['projected_risk']['intensity']} (additive)")
assert slr["projected_risk"]["intensity"] > 0.1

# Test 19: Scenario name variations
delta2 = engine.calculate_warming_delta("SSP2-4.5", "mid_term")
print(f"Warming delta (name variation): {delta2}C")
assert abs(delta2 - 1.485) < 0.001

# Test 20: BASELINE returns unchanged risk
base_proj = engine.project_hazard(
    hazard_type="DROUGHT",
    location="Test",
    baseline_risk={
        "probability": 0.5,
        "intensity": 0.5,
        "frequency": 2.0,
        "duration_days": 10.0,
    },
    scenario="ssp2_4.5",
    time_horizon="BASELINE",
)
assert base_proj["projected_risk"]["probability"] == 0.5
assert base_proj["projected_risk"]["intensity"] == 0.5
assert base_proj["projected_risk"]["frequency"] == 2.0
assert base_proj["projected_risk"]["duration_days"] == 10.0
print("BASELINE returns unchanged: OK")

# Test 21: export_projections
all_projs_export = engine.export_projections(format="dict")
print(f"Exported {len(all_projs_export)} projections as dict")
json_export = engine.export_projections(format="json")
print(f"JSON export length: {len(json_export)} chars")

# Test 22: search_projections
search_results = engine.search_projections(query="Maldives")
print(f"Search for Maldives: {len(search_results)} results")

# Test 23: project_all_hazards
batch = engine.project_all_hazards(
    location="Tokyo",
    baseline_risks={
        "EXTREME_HEAT": {
            "probability": 0.3,
            "intensity": 0.6,
            "frequency": 3.0,
            "duration_days": 7.0,
        },
        "RIVERINE_FLOOD": {
            "probability": 0.15,
            "intensity": 0.4,
            "frequency": 2.0,
            "duration_days": 3.0,
        },
    },
    scenario="ssp2_4.5",
    time_horizon="MID_TERM",
)
print(f"Batch projection: {batch['hazard_count']} hazards projected")
assert batch["hazard_count"] == 2

# Test 24: compare_scenarios_over_time
comparison = engine.compare_scenarios_over_time(
    hazard_type="WILDFIRE",
    location="Australia",
    baseline_risk={
        "probability": 0.25,
        "intensity": 0.65,
        "frequency": 2.0,
        "duration_days": 14.0,
    },
    scenarios=["ssp1_2.6", "ssp2_4.5", "ssp5_8.5"],
)
print(
    f"Scenario-time matrix: {comparison['summary']['total_projections']} projections"
)
assert comparison["summary"]["total_projections"] == 15  # 3 scenarios x 5 horizons

# Test 25: Error handling - invalid scenario
try:
    engine.calculate_warming_delta("invalid_scenario", "MID_TERM")
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"Invalid scenario error: OK ({str(e)[:50]}...)")

# Test 26: Error handling - invalid hazard type
try:
    engine.get_scaling_factors("INVALID_HAZARD")
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"Invalid hazard error: OK ({str(e)[:50]}...)")

# Test 27: import_projections
import_result = engine.import_projections([
    {"projection_id": "test-import-1", "hazard_type": "DROUGHT"},
    {"projection_id": "test-import-2", "hazard_type": "WILDFIRE"},
])
print(f"Imported: {import_result['imported']}, skipped: {import_result['skipped']}")
assert import_result["imported"] == 2

print()
print("ALL 27 TESTS PASSED")
