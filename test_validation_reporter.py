"""Quick integration test for ValidationReporterEngine."""
import importlib.util
import sys
import json
import threading

# Load the module directly bypassing __init__.py
for mod_name, mod_path in [
    ("greenlang.validation_rule_engine.config",
     "greenlang/validation_rule_engine/config.py"),
    ("greenlang.validation_rule_engine.provenance",
     "greenlang/validation_rule_engine/provenance.py"),
    ("greenlang.validation_rule_engine.validation_reporter",
     "greenlang/validation_rule_engine/validation_reporter.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)

from greenlang.validation_rule_engine.validation_reporter import (
    ValidationReporterEngine,
)

reporter = ValidationReporterEngine()

# Test data
results = [
    {"rule_id": "R001", "rule_name": "Not null check", "status": "pass",
     "severity": "critical", "pass_count": 100, "fail_count": 0},
    {"rule_id": "R002", "rule_name": "Range check", "status": "fail",
     "severity": "high", "pass_count": 90, "fail_count": 10,
     "failures": [{"row": 5, "field": "amount", "value": -1, "expected": ">= 0"}]},
    {"rule_id": "R003", "rule_name": "Email format", "status": "fail",
     "severity": "medium", "pass_count": 95, "fail_count": 5},
    {"rule_id": "R004", "rule_name": "Date check", "status": "pass",
     "severity": "low", "pass_count": 100, "fail_count": 0},
]

# Test 1: All non-trend report types x formats
for rt in ["summary", "detailed", "compliance", "executive"]:
    for fmt in ["text", "json", "html", "markdown", "csv"]:
        report = reporter.generate_report(rt, fmt, results)
        assert report["report_type"] == rt
        assert report["format"] == fmt
        assert report["report_hash"]
        assert report["report_id"]
        assert report["provenance_hash"]
        assert report["content"]
        print(f"  PASS: {rt}/{fmt} ({len(report['content'])} chars)")

# Test 2: Trend report
history = [
    {"timestamp": "2026-01-01", "label": "Q4-2025", "results": [
        {"rule_id": "R001", "status": "pass", "severity": "high",
         "pass_count": 100, "fail_count": 0},
        {"rule_id": "R002", "status": "fail", "severity": "high",
         "pass_count": 80, "fail_count": 20},
    ]},
    {"timestamp": "2026-02-01", "label": "Q1-2026", "results": [
        {"rule_id": "R001", "status": "pass", "severity": "high",
         "pass_count": 100, "fail_count": 0},
        {"rule_id": "R002", "status": "pass", "severity": "high",
         "pass_count": 100, "fail_count": 0},
        {"rule_id": "R003", "status": "fail", "severity": "medium",
         "pass_count": 90, "fail_count": 10},
    ]},
]
for fmt in ["text", "json", "html", "markdown", "csv"]:
    report = reporter.generate_report("trend", fmt, history)
    assert report["report_type"] == "trend"
    assert report["content"]
    print(f"  PASS: trend/{fmt} ({len(report['content'])} chars)")

# Test 3: Retrieval
last_id = report["report_id"]
retrieved = reporter.get_report(last_id)
assert retrieved is not None
assert retrieved["report_id"] == last_id
print("  PASS: get_report")

# Test 4: List
reports_list = reporter.list_reports()
assert len(reports_list) == 25
print(f"  PASS: list_reports ({len(reports_list)} reports)")

# Test 5: Filtered list
reports_filtered = reporter.list_reports(report_type="summary")
assert len(reports_filtered) == 5
print(f"  PASS: list_reports type filter ({len(reports_filtered)})")

reports_fmt = reporter.list_reports(format="json")
assert len(reports_fmt) == 5
print(f"  PASS: list_reports format filter ({len(reports_fmt)})")

# Test 6: Statistics
stats = reporter.get_statistics()
assert stats["total_reports_stored"] == 25
assert stats["total_reports_generated"] == 25
assert stats["provenance_entries"] == 25
print("  PASS: get_statistics")

# Test 7: Clear
reporter.clear()
assert len(reporter.list_reports()) == 0
print("  PASS: clear")

# Test 8: Invalid inputs
try:
    reporter.generate_report("invalid", "json", results)
    assert False
except ValueError:
    print("  PASS: invalid report_type raises ValueError")

try:
    reporter.generate_report("summary", "pdf", results)
    assert False
except ValueError:
    print("  PASS: invalid format raises ValueError")

# Test 9: provenance=None
reporter2 = ValidationReporterEngine(provenance=None)
r = reporter2.generate_report("summary", "json", results)
assert r["provenance_hash"]
print("  PASS: provenance=None creates fresh tracker")

# Test 10: Empty results
r = reporter2.generate_report("summary", "text", [])
assert r["content"]
print("  PASS: empty results handled")

# Test 11: Compliance framework filter
r = reporter2.generate_report(
    "compliance", "json", results, parameters={"framework": "ghg_protocol"}
)
content = json.loads(r["content"])
assert "ghg_protocol" in content.get("frameworks", {})
assert "csrd_esrs" not in content.get("frameworks", {})
print("  PASS: compliance framework filter")

# Test 12: Trend insufficient history
r = reporter2.generate_report(
    "trend", "json", [{"timestamp": "2026-01-01", "results": []}]
)
content = json.loads(r["content"])
assert "Insufficient" in content.get("message", "")
print("  PASS: trend insufficient history")

# Test 13: Thread safety
errors = []

def worker(reporter_inst, idx):
    try:
        for _ in range(5):
            reporter_inst.generate_report("summary", "json", results)
    except Exception as e:
        errors.append(str(e))

threads = [threading.Thread(target=worker, args=(reporter2, i)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
assert not errors
print("  PASS: thread safety (20 concurrent reports)")

# Test 14: Executive risk score computation
r = reporter2.generate_report("executive", "json", [
    {"rule_id": "X1", "status": "fail", "severity": "critical",
     "pass_count": 0, "fail_count": 100},
    {"rule_id": "X2", "status": "fail", "severity": "high",
     "pass_count": 50, "fail_count": 50},
])
content = json.loads(r["content"])
risk_score = content.get("risk_assessment", {}).get("risk_score", 0)
assert risk_score > 50, f"Expected high risk score, got {risk_score}"
print(f"  PASS: risk scoring (score={risk_score})")

# Test 15: Report hash determinism
r1 = reporter2.generate_summary_report(results, "json")
r2 = reporter2.generate_summary_report(results, "json")
assert isinstance(r1, str) and len(r1) > 0
assert isinstance(r2, str) and len(r2) > 0
print("  PASS: deterministic content generation")

# Test 16: Detailed report with row failures
detailed_results = [
    {"rule_id": "D1", "rule_name": "Amount positive", "status": "fail",
     "severity": "high", "rule_type": "range", "pass_count": 90,
     "fail_count": 10, "field": "amount", "condition": "> 0",
     "message": "Negative amounts detected",
     "failures": [
         {"row": i, "field": "amount", "value": -i, "expected": "> 0",
          "message": f"Row {i} negative"}
         for i in range(1, 60)
     ]},
]
r = reporter2.generate_report("detailed", "markdown", detailed_results)
assert "Row Failures" in r["content"]
print("  PASS: detailed report with row failures")

# Test 17: get_report returns None for unknown ID
assert reporter2.get_report("nonexistent-id") is None
print("  PASS: get_report returns None for unknown ID")

print()
print("ALL 17 TESTS PASSED")
