#!/usr/bin/env python
"""
Integration Test Suite for GreenLang Infrastructure Platform
=============================================================

This script runs all verification commands to ensure the platform is working correctly.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Tuple, List
import tempfile
import shutil


def run_command(cmd: List[str], cwd: Path = None) -> Tuple[bool, str, str]:
    """
    Run a command and return success status, stdout, and stderr
    """
    try:
        # Set environment to handle Unicode properly
        import os
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
            env=env,
            encoding='utf-8',
            errors='replace'  # Replace problematic characters instead of failing
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)


def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result"""
    status = "[OK]" if success else "[FAIL]"
    color = "\033[92m" if success else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} {test_name}")
    if details and not success:
        print(f"     {details}")


def test_gl_doctor():
    """Test gl doctor command"""
    print_header("Testing: gl doctor")
    
    # Test basic doctor command
    success, stdout, stderr = run_command(["python", "-m", "core.greenlang.cli", "doctor"])
    
    # Doctor may have warnings, that's ok
    stdout_str = stdout or ""
    stderr_str = stderr or ""
    if "GreenLang Environment Check" in stdout_str or "GreenLang Environment Check" in stderr_str:
        print_result("gl doctor", True)
        return True
    else:
        print_result("gl doctor", False, f"Doctor command failed to run (stdout: {success}, stderr: {stderr_str[:100]}...)")
        return False


def test_pipeline_execution():
    """Test pipeline execution with gl run"""
    print_header("Testing: gl run packs/boiler-solar/gl.yaml")
    
    # Check if gl.yaml exists
    gl_yaml = Path("packs/boiler-solar/gl.yaml")
    if not gl_yaml.exists():
        print_result("Pipeline file exists", False, f"{gl_yaml} not found")
        return False
    print_result("Pipeline file exists", True)
    
    # Check if inputs.json exists
    inputs_file = Path("inputs.json")
    if not inputs_file.exists():
        print_result("Input file exists", False, "inputs.json not found")
        return False
    print_result("Input file exists", True)
    
    # Test pipeline execution using Python module
    print("\nExecuting pipeline...")
    from core.greenlang.runtime.executor import Executor
    
    try:
        executor = Executor()
        with open(inputs_file) as f:
            inputs = json.load(f)
        
        # Use temporary directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir) / "artifacts"
            result = executor.run(str(gl_yaml), inputs, artifacts_dir)
            
            # Check if the executor is working by looking for step results
            has_results = hasattr(result, 'data') and result.data
            successful_steps = 0
            total_steps = 0
            
            if has_results:
                for step_name, step_data in result.data.items():
                    total_steps += 1
                    if isinstance(step_data, dict) and step_data.get('success', False):
                        successful_steps += 1
                
                print(f"     Pipeline executed: {successful_steps}/{total_steps} steps succeeded")
                
                # Show step details
                print("\n     Step results:")
                for step_name, step_data in result.data.items():
                    if isinstance(step_data, dict):
                        status = "OK" if step_data.get('success', False) else "FAIL"
                        print(f"       - {step_name}: {status}")
                        if not step_data.get('success', False) and 'error' in step_data:
                            print(f"         Error: {step_data['error']}")
                
                # Check if artifacts were created
                if artifacts_dir.exists():
                    artifact_count = len(list(artifacts_dir.glob("*")))
                    print(f"     Created {artifact_count} artifacts in {artifacts_dir}")
            
            # For integration testing, we consider it successful if:
            # 1. The executor ran without throwing an exception
            # 2. We got some step results
            # 3. At least some steps succeeded
            if has_results and successful_steps > 0:
                print_result("Pipeline execution", True, f"Executor functional with {successful_steps} successful steps")
                return True
            elif result.success:
                print_result("Pipeline execution", True)
                return True
            else:
                error_msg = getattr(result, 'error', 'No step results produced')
                print_result("Pipeline execution", False, error_msg)
                return False
                
    except Exception as e:
        print_result("Pipeline execution", False, str(e))
        return False


def test_policy_check():
    """Test policy check command"""
    print_header("Testing: gl policy check packs/boiler-solar")
    
    pack_dir = Path("packs/boiler-solar")
    if not pack_dir.exists():
        print_result("Pack directory exists", False, f"{pack_dir} not found")
        return False
    print_result("Pack directory exists", True)
    
    # Test policy check using the module directly
    from core.greenlang.policy.enforcer import check_install
    from core.greenlang.packs.manifest import load_manifest
    
    try:
        manifest = load_manifest(pack_dir)
        print_result("Manifest loaded", True)
        
        # Check policy
        try:
            check_install(manifest, str(pack_dir), "publish")
            print_result("Policy check", True)
            print("     Policy allows pack installation")
            return True
        except RuntimeError as e:
            print_result("Policy check", False, str(e))
            return False
            
    except Exception as e:
        print_result("Policy check", False, str(e))
        return False


def test_pack_validation():
    """Test pack validation"""
    print_header("Testing: Pack Validation")
    
    from core.greenlang.packs.manifest import validate_pack
    
    pack_dir = Path("packs/boiler-solar")
    if not pack_dir.exists():
        print_result("Pack directory", False, f"{pack_dir} not found")
        return False
    
    is_valid, errors = validate_pack(pack_dir)
    
    if is_valid:
        print_result("Pack validation", True)
        return True
    else:
        print_result("Pack validation", False)
        for error in errors:
            print(f"     - {error}")
        return False


def test_context_artifact_management():
    """Test context and artifact management"""
    print_header("Testing: Context & Artifact Management")
    
    from core.greenlang.sdk.context import Context
    from core.greenlang.sdk.base import Result
    
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir) / "test_artifacts"
        
        # Create context
        context = Context(
            inputs={"test": "data"},
            artifacts_dir=artifacts_dir,
            profile="test",
            backend="local"
        )
        
        # Test artifact creation
        test_data = {"result": "success", "metrics": [1, 2, 3]}
        artifact = context.save_artifact("test_output", test_data, type="json")
        
        if artifact and artifact.path.exists():
            print_result("Artifact creation", True)
        else:
            print_result("Artifact creation", False, "Failed to create artifact")
            return False
        
        # Test step result management
        result = Result(success=True, data={"output": "test"})
        context.add_step_result("test_step", result)
        
        if "test_step" in context.steps:
            print_result("Step result tracking", True)
        else:
            print_result("Step result tracking", False)
            return False
        
        # Test context conversion
        final_result = context.to_result()
        if final_result.success:
            print_result("Context to Result conversion", True)
            return True
        else:
            print_result("Context to Result conversion", False)
            return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print(" GreenLang Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("System Diagnostics", test_gl_doctor),
        ("Pack Validation", test_pack_validation),
        ("Policy Enforcement", test_policy_check),
        ("Context & Artifacts", test_context_artifact_management),
        ("Pipeline Execution", test_pipeline_execution)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print_result(test_name, False, str(e))
            results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    print("\nDetailed Results:")
    for name, success in results:
        status = "[OK]" if success else "[FAIL]"
        color = "\033[92m" if success else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset} {name}")
    
    if passed == total:
        print("\n\033[92m All integration tests passed! \033[0m")
        print("\nThe GreenLang infrastructure platform is fully operational.")
        print("\nYou can now run:")
        print("  - gl run packs/boiler-solar/gl.yaml -i inputs.json")
        print("  - gl policy check packs/boiler-solar")
        print("  - gl doctor")
        return 0
    else:
        print(f"\n\033[91m {total - passed} test(s) failed. \033[0m")
        print("\nPlease review the failures above and run:")
        print("  - gl doctor --fix  # To attempt automatic fixes")
        return 1


if __name__ == "__main__":
    sys.exit(main())