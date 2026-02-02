#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual test for deterministic pipeline execution
"""

import json
import os
from pathlib import Path
from greenlang.cli.cmd_run import run

class MockContext:
    def __init__(self):
        self.invoked_subcommand = None

def test_deterministic_execution():
    """Test pipeline multiple times and compare outputs"""
    print("=" * 60)
    print("Testing Deterministic Pipeline Execution")
    print("=" * 60)
    
    ctx = MockContext()
    
    # Run pipeline 3 times with identical inputs
    for i in range(1, 4):
        print(f"\nRun #{i}:")
        artifacts_dir = f"pipeline_test_run_{i}"
        
        try:
            result = run(
                ctx=ctx,
                pipeline="test_python_pipeline.yaml",
                inputs="inputs.json", 
                artifacts=artifacts_dir,
                backend="local",
                profile="dev",
                audit=False
            )
            print(f"[OK] Run #{i} completed")
            
            # Check artifacts generated
            artifacts_path = Path(artifacts_dir)
            if artifacts_path.exists():
                artifact_files = list(artifacts_path.glob("*"))
                print(f"  Generated {len(artifact_files)} artifacts")
                for f in artifact_files[:3]:  # Show first 3
                    print(f"    - {f.name}")
            
        except SystemExit as e:
            if e.code == 0:
                print(f"[OK] Run #{i} completed successfully")
            else:
                print(f"[FAIL] Run #{i} failed with exit code: {e.code}")
        except Exception as e:
            print(f"[FAIL] Run #{i} failed with error: {e}")
    
    print("\n" + "=" * 60)
    print("Comparing outputs for determinism...")
    print("=" * 60)
    
    # Compare artifacts between runs
    artifacts_1 = Path("pipeline_test_run_1")
    artifacts_2 = Path("pipeline_test_run_2") 
    artifacts_3 = Path("pipeline_test_run_3")
    
    if all(p.exists() for p in [artifacts_1, artifacts_2, artifacts_3]):
        files_1 = list(artifacts_1.glob("*"))
        files_2 = list(artifacts_2.glob("*"))
        files_3 = list(artifacts_3.glob("*"))
        
        print(f"Run 1: {len(files_1)} files")
        print(f"Run 2: {len(files_2)} files") 
        print(f"Run 3: {len(files_3)} files")
        
        if len(files_1) == len(files_2) == len(files_3):
            print("[OK] All runs generated same number of artifacts")
            
            # Compare file contents if JSON
            for f1 in files_1:
                f2 = artifacts_2 / f1.name
                f3 = artifacts_3 / f1.name
                
                if f2.exists() and f3.exists():
                    if f1.suffix == '.json':
                        try:
                            with open(f1) as file1, open(f2) as file2, open(f3) as file3:
                                data1 = json.load(file1)
                                data2 = json.load(file2)
                                data3 = json.load(file3)
                                
                                if data1 == data2 == data3:
                                    print(f"[OK] {f1.name}: Identical across all runs")
                                else:
                                    print(f"[FAIL] {f1.name}: Different outputs detected")
                        except Exception as e:
                            print(f"[WARN] Could not compare {f1.name}: {e}")
        else:
            print("[FAIL] Different number of artifacts generated")
    else:
        print("[WARN] Not all artifact directories exist")

if __name__ == "__main__":
    test_deterministic_execution()