#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Priority 3C: Run Ledger Implementation - Validation Test
=========================================================

This test validates that the Run Ledger properly:
1. Records pipeline executions with unique IDs
2. Calculates and stores input/output hashes
3. Provides audit trail functionality
4. Supports querying and analysis
5. Verifies reproducibility
"""

import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from greenlang.provenance.ledger import RunLedger, write_run_ledger, verify_run_ledger
from greenlang.determinism import DeterministicClock


def test_run_ledger():
    """Test RunLedger class functionality"""
    
    print("Priority 3C: Run Ledger Implementation Test")
    print("=" * 60)
    
    # Create temporary ledger for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        ledger_path = Path(tmpdir) / "test_ledger.jsonl"
        ledger = RunLedger(ledger_path)
        
        # Test 1: Record Run
        print("\n1. Testing run recording...")
        
        test_inputs = {
            "param1": "value1",
            "param2": 123,
            "config": {"nested": "data"}
        }
        
        test_outputs = {
            "result": "success",
            "metrics": {
                "accuracy": 0.95,
                "performance": 100
            }
        }
        
        test_metadata = {
            "backend": "local",
            "duration": 1.234,
            "user": "test-user"
        }
        
        # Record a run
        run_id = ledger.record_run(
            pipeline="test-pipeline",
            inputs=test_inputs,
            outputs=test_outputs,
            metadata=test_metadata
        )
        
        # Validate run ID format
        try:
            uuid.UUID(run_id)
            print(f"   [OK] Valid UUID generated: {run_id}")
        except ValueError:
            print(f"   [FAIL] Invalid UUID: {run_id}")
            return False
        
        # Check ledger file exists
        assert ledger_path.exists(), "Ledger file not created"
        print(f"   [OK] Ledger file created: {ledger_path}")
        
        # Test 2: Retrieve Run
        print("\n2. Testing run retrieval...")
        
        retrieved = ledger.get_run(run_id)
        
        assert retrieved is not None, "Run not retrieved"
        assert retrieved["id"] == run_id, "Run ID mismatch"
        assert retrieved["pipeline"] == "test-pipeline", "Pipeline name mismatch"
        
        print(f"   [OK] Run retrieved successfully")
        print(f"   [OK] Pipeline: {retrieved['pipeline']}")
        print(f"   [OK] Timestamp: {retrieved['timestamp']}")
        
        # Test 3: Hash Verification
        print("\n3. Testing hash calculation...")
        
        # Verify input hash
        import hashlib
        expected_input_hash = hashlib.sha256(
            json.dumps(test_inputs, sort_keys=True).encode()
        ).hexdigest()
        
        assert retrieved["input_hash"] == expected_input_hash, "Input hash mismatch"
        print(f"   [OK] Input hash correct: {retrieved['input_hash'][:16]}...")
        
        # Verify output hash
        expected_output_hash = hashlib.sha256(
            json.dumps(test_outputs, sort_keys=True).encode()
        ).hexdigest()
        
        assert retrieved["output_hash"] == expected_output_hash, "Output hash mismatch"
        print(f"   [OK] Output hash correct: {retrieved['output_hash'][:16]}...")
        
        # Test 4: Multiple Runs
        print("\n4. Testing multiple run recording...")
        
        # Record multiple runs
        run_ids = []
        for i in range(5):
            run_id = ledger.record_run(
                pipeline=f"pipeline-{i % 2}",  # Alternate between 2 pipelines
                inputs={"index": i},
                outputs={"result": i * 2},
                metadata={"iteration": i}
            )
            run_ids.append(run_id)
        
        print(f"   [OK] Recorded 5 additional runs")
        
        # Test listing
        all_runs = ledger.list_runs(limit=10)
        assert len(all_runs) >= 6, f"Expected at least 6 runs, got {len(all_runs)}"
        print(f"   [OK] Total runs in ledger: {len(all_runs)}")
        
        # Test filtering by pipeline
        pipeline_0_runs = ledger.list_runs(pipeline="pipeline-0", limit=10)
        assert len(pipeline_0_runs) >= 2, "Pipeline filtering failed"
        print(f"   [OK] Filtered runs for pipeline-0: {len(pipeline_0_runs)}")
        
        # Test 5: Duplicate Detection
        print("\n5. Testing duplicate run detection...")
        
        # Record duplicate input
        dup_input = {"index": 0}
        dup_id1 = ledger.record_run(
            pipeline="dup-test",
            inputs=dup_input,
            outputs={"result": "A"},
            metadata={}
        )
        
        dup_id2 = ledger.record_run(
            pipeline="dup-test",
            inputs=dup_input,  # Same input
            outputs={"result": "A"},  # Same output (reproducible)
            metadata={}
        )
        
        # Find duplicates
        input_hash = hashlib.sha256(
            json.dumps(dup_input, sort_keys=True).encode()
        ).hexdigest()
        
        duplicates = ledger.find_duplicate_runs(input_hash, "dup-test")
        assert len(duplicates) >= 2, "Duplicate detection failed"
        print(f"   [OK] Found {len(duplicates)} runs with same input")
        
        # Test 6: Reproducibility Verification
        print("\n6. Testing reproducibility verification...")
        
        output_hash = hashlib.sha256(
            json.dumps({"result": "A"}, sort_keys=True).encode()
        ).hexdigest()
        
        is_reproducible = ledger.verify_reproducibility(
            input_hash, output_hash, "dup-test"
        )
        
        assert is_reproducible, "Reproducibility check failed"
        print("   [OK] Reproducibility verified for identical inputs/outputs")
        
        # Test non-reproducible case
        ledger.record_run(
            pipeline="dup-test",
            inputs=dup_input,
            outputs={"result": "B"},  # Different output
            metadata={}
        )
        
        is_reproducible = ledger.verify_reproducibility(
            input_hash, output_hash, "dup-test"
        )
        
        assert not is_reproducible, "Should detect non-reproducibility"
        print("   [OK] Non-reproducibility detected correctly")
        
        # Test 7: Statistics
        print("\n7. Testing statistics generation...")
        
        stats = ledger.get_statistics(days=30)
        
        assert stats["total_runs"] >= 8, "Statistics count wrong"
        assert stats["unique_inputs"] >= 6, "Unique inputs count wrong"  # Adjusted for duplicate inputs
        assert len(stats["pipelines"]) >= 3, "Pipeline count wrong"
        
        print(f"   [OK] Total runs: {stats['total_runs']}")
        print(f"   [OK] Unique inputs: {stats['unique_inputs']}")
        print(f"   [OK] Unique outputs: {stats['unique_outputs']}")
        print(f"   [OK] Pipelines: {stats['pipelines']}")
        
        # Test 8: Export
        print("\n8. Testing ledger export...")
        
        export_path = Path(tmpdir) / "export.json"
        ledger.export_to_json(export_path, days=30)
        
        assert export_path.exists(), "Export file not created"
        
        with open(export_path) as f:
            export_data = json.load(f)
        
        assert "runs" in export_data, "Missing runs in export"
        assert "statistics" in export_data, "Missing statistics in export"
        assert len(export_data["runs"]) >= 8, "Export missing runs"
        
        print(f"   [OK] Exported {len(export_data['runs'])} runs to JSON")
        
        return True


def test_single_run_ledger():
    """Test single run ledger functions"""
    
    print("\n9. Testing single run ledger functions...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "run.json"
        
        # Create mock result and context
        class MockResult:
            success = True
            outputs = {"result": "test"}
            metrics = {"score": 0.99}
        
        class MockContext:
            started_at = DeterministicClock.utcnow()
            start_time = 0
            pipeline_spec = {"name": "test"}
            inputs = {"param": "value"}
            config = {"setting": "value"}
            backend = "local"
            profile = "dev"
            artifacts = []
        
        result = MockResult()
        ctx = MockContext()
        
        # Write ledger
        ledger_path = write_run_ledger(result, ctx, output_path)
        
        assert ledger_path.exists(), "Run ledger not created"
        print(f"   [OK] Single run ledger created: {ledger_path}")
        
        # Verify ledger
        is_valid = verify_run_ledger(ledger_path)
        assert is_valid, "Run ledger verification failed"
        print("   [OK] Run ledger integrity verified")
        
        # Read ledger
        from greenlang.provenance.ledger import read_run_ledger
        ledger_data = read_run_ledger(ledger_path)
        
        assert ledger_data["metadata"]["status"] == "success", "Status wrong"
        assert "pipeline_hash" in ledger_data["spec"], "Missing pipeline hash"
        assert "inputs_hash" in ledger_data["spec"], "Missing inputs hash"
        
        print("   [OK] Run ledger read successfully")
        
        return True


def main():
    """Run Priority 3C validation tests"""
    
    # Test RunLedger class
    ledger_success = test_run_ledger()
    
    # Test single run ledger
    single_success = test_single_run_ledger()
    
    # Summary
    print("\n" + "=" * 60)
    
    if ledger_success and single_success:
        print("PRIORITY 3C VALIDATION: ALL TESTS PASSED")
        print("=" * 60)
        print("\nRun Ledger Features Verified:")
        print("- Pipeline execution recording with UUIDs")
        print("- Input/output hash calculation")
        print("- Run retrieval and querying")
        print("- Duplicate detection")
        print("- Reproducibility verification")
        print("- Statistics generation")
        print("- Ledger export to JSON")
        print("- Single run ledger with integrity checks")
        print("\nThe platform now has complete audit trail and")
        print("provenance tracking for compliance and governance.")
    else:
        print("PRIORITY 3C VALIDATION: SOME TESTS FAILED")
        print("\nPlease review the failures above.")
    
    return 0 if (ledger_success and single_success) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())