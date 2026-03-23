"""Smoke test for RuleRegistryEngine."""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "rule_registry",
    r"C:\Users\aksha\Code-V1_GreenLang\greenlang\validation_rule_engine\rule_registry.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
RuleRegistryEngine = mod.RuleRegistryEngine

# Test 1: Initialize
engine = RuleRegistryEngine()
print("1. Init OK")

# Test 2: Register a rule
rule = engine.register_rule(
    name="completeness_co2e",
    rule_type="COMPLETENESS",
    column="co2e_tonnes",
    operator="IS_NULL",
    threshold=0.0,
    severity="HIGH",
    description="CO2e column must not be null",
    tags=["emissions", "scope1"],
)
assert rule["rule_id"]
assert rule["status"] == "active"
assert rule["version"] == "1.0.0"
assert rule["rule_type"] == "COMPLETENESS"
assert rule["provenance_hash"]
print("2. Register OK: id=" + rule["rule_id"][:8])

# Test 3: Get rule
retrieved = engine.get_rule(rule["rule_id"])
assert retrieved is not None
assert retrieved["name"] == "completeness_co2e"
print("3. Get rule OK")

# Test 4: Get rule by name
by_name = engine.get_rule_by_name("completeness_co2e")
assert by_name is not None
assert by_name["rule_id"] == rule["rule_id"]
print("4. Get by name OK")

# Test 5: Update rule (patch bump - cosmetic)
updated = engine.update_rule(rule["rule_id"], description="Updated description")
assert updated["version"] == "1.0.1"
assert updated["description"] == "Updated description"
print("5. Update (patch) OK: v=" + updated["version"])

# Test 6: Update rule (minor bump - additive)
updated2 = engine.update_rule(rule["rule_id"], threshold=5.0)
assert updated2["version"] == "1.1.0"
print("6. Update (minor) OK: v=" + updated2["version"])

# Test 7: Update rule (major bump - breaking)
updated3 = engine.update_rule(rule["rule_id"], column="new_column")
assert updated3["version"] == "2.0.0"
print("7. Update (major) OK: v=" + updated3["version"])

# Test 8: Version history
versions = engine.get_rule_versions(rule["rule_id"])
assert len(versions) >= 4
print("8. Version history OK: count=" + str(len(versions)))

# Test 9: Rollback
rolled = engine.rollback_rule(rule["rule_id"], "1.0.0")
assert rolled["description"] == "CO2e column must not be null"
assert rolled["column"] == "co2e_tonnes"
print("9. Rollback OK: v=" + rolled["version"])

# Test 10: Clone
clone = engine.clone_rule(rule["rule_id"], "clone_of_co2e")
assert clone["name"] == "clone_of_co2e"
assert clone["rule_id"] != rule["rule_id"]
print("10. Clone OK")

# Test 11: Search
results = engine.search_rules(rule_type="COMPLETENESS")
assert len(results) == 2
print("11. Search OK: count=" + str(len(results)))

# Test 12: Search with tags
results_tagged = engine.search_rules(tags=["emissions"])
assert len(results_tagged) == 2
print("12. Search tags OK: count=" + str(len(results_tagged)))

# Test 13: Search with name pattern
results_pat = engine.search_rules(name_pattern="clone")
assert len(results_pat) == 1
print("13. Search pattern OK")

# Test 14: Register more rules
rule2 = engine.register_rule(
    name="range_temp", rule_type="RANGE",
    column="temp", operator="BETWEEN",
    parameters={"min": -50, "max": 60},
    severity="MEDIUM",
)
print("14. Second rule OK")

# Test 15: Bulk register
bulk = engine.bulk_register([
    {"name": "b1", "rule_type": "FORMAT", "column": "email", "operator": "MATCHES", "severity": "LOW"},
    {"name": "b2", "rule_type": "UNIQUENESS", "column": "id", "operator": "EQUALS", "severity": "CRITICAL"},
    {"name": "", "rule_type": "RANGE", "column": "x", "operator": "GREATER_THAN"},
])
assert bulk["registered"] == 2
assert bulk["failed"] == 1
print("15. Bulk register OK: registered=" + str(bulk["registered"]) + " failed=" + str(bulk["failed"]))

# Test 16: Statistics
stats = engine.get_statistics()
assert stats["total_rules"] == 5
assert "COMPLETENESS" in stats["by_type"]
assert "RANGE" in stats["by_type"]
print("16. Statistics OK: total=" + str(stats["total_rules"]))

# Test 17: List rule types
types = engine.list_rule_types()
assert "RANGE" in types
print("17. List types OK: " + str(types))

# Test 18: List severities
sevs = engine.list_severities()
assert "HIGH" in sevs
print("18. List severities OK: " + str(sevs))

# Test 19: Export
exported = engine.export_rules()
assert len(exported) == 5
print("19. Export OK: count=" + str(len(exported)))

# Test 20: Export filtered
exported_range = engine.export_rules(rule_type="RANGE")
assert len(exported_range) == 1
print("20. Export filtered OK: count=" + str(len(exported_range)))

# Test 21: Import
engine2 = RuleRegistryEngine()
imported = engine2.import_rules(exported)
assert imported["imported"] == 5
print("21. Import OK: imported=" + str(imported["imported"]))

# Test 22: Delete (soft)
assert engine.delete_rule(rule2["rule_id"]) is True
archived = engine.get_rule(rule2["rule_id"])
assert archived["status"] == "archived"
print("22. Soft delete OK")

# Test 23: Delete (hard)
assert engine.delete_rule(rule2["rule_id"], hard=True) is True
assert engine.get_rule(rule2["rule_id"]) is None
print("23. Hard delete OK")

# Test 24: Delete non-existent
assert engine.delete_rule("nonexistent-id") is False
print("24. Delete non-existent OK")

# Test 25: Duplicate name
try:
    engine.register_rule(
        name="completeness_co2e", rule_type="RANGE",
        column="x", operator="EQUALS",
    )
    assert False, "Should have raised ValueError"
except ValueError:
    print("25. Duplicate name rejected OK")

# Test 26: Invalid rule_type
try:
    engine.register_rule(
        name="bad", rule_type="INVALID",
        column="x", operator="EQUALS",
    )
    assert False, "Should have raised"
except ValueError:
    print("26. Invalid rule_type rejected OK")

# Test 27: Invalid operator
try:
    engine.register_rule(
        name="bad2", rule_type="RANGE",
        column="x", operator="INVALID",
    )
    assert False, "Should have raised"
except ValueError:
    print("27. Invalid operator rejected OK")

# Test 28: Invalid severity
try:
    engine.register_rule(
        name="bad3", rule_type="RANGE",
        column="x", operator="EQUALS", severity="INVALID",
    )
    assert False, "Should have raised"
except ValueError:
    print("28. Invalid severity rejected OK")

# Test 29: Provenance chain
chain = engine.get_provenance_chain()
assert len(chain) > 0
print("29. Provenance chain OK: entries=" + str(len(chain)))

# Test 30: Clear
engine.clear()
assert engine.get_statistics()["total_rules"] == 0
print("30. Clear OK")

# Test 31: Introspection properties
engine.register_rule(
    name="prop_test", rule_type="COMPLETENESS",
    column="col", operator="IS_NULL",
)
assert engine.rule_count == 1
assert engine.provenance_chain_length > 0
print("31. Properties OK")

# Test 32: get_all_rule_ids
ids = engine.get_all_rule_ids()
assert len(ids) == 1
print("32. get_all_rule_ids OK")

# Test 33: list_rules
listed = engine.list_rules()
assert len(listed) == 1
print("33. list_rules OK")

print()
print("ALL 33 TESTS PASSED")
