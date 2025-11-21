# -*- coding: utf-8 -*-
"""Acceptance test script for FRMW-202 - gl init agent command."""

import sys
import subprocess
from pathlib import Path
import shutil

def test_compute_template():
    """Test compute template generation."""
    print("=" * 70)
    print("TEST 1: Compute Template")
    print("=" * 70)

    # Clean previous test
    test_dir = Path("./test_output/test-boiler-compute")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Import and run
    sys.path.insert(0, str(Path.cwd()))
    from greenlang.cli.cmd_init_agent import (
        generate_compute_agent,
        generate_common_files,
        generate_test_suite,
        generate_examples,
        generate_documentation,
        generate_gitignore,
        generate_ci_workflow,
        validate_generated_agent
    )

    # Create agent
    agent_dir = Path("./test_output/test-boiler-compute")
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "src" / "test_boiler_compute").mkdir(parents=True, exist_ok=True)
    (agent_dir / "tests").mkdir(exist_ok=True)
    (agent_dir / "docs").mkdir(exist_ok=True)
    (agent_dir / "examples").mkdir(exist_ok=True)

    print(f"\n[1/7] Creating compute agent in: {agent_dir}")
    generate_compute_agent(
        agent_dir=agent_dir,
        pack_id='test-boiler-compute',
        python_pkg='test_boiler_compute',
        class_name='TestBoilerCompute',
        license='apache-2.0',
        author='Test Author',
        realtime=False,
        spec_data={}
    )
    print("✅ Agent files generated")

    print("\n[2/7] Generating common files...")
    generate_common_files(agent_dir, 'test-boiler-compute', 'test_boiler_compute', 'apache-2.0', 'Test Author')
    print("✅ Common files generated")

    print("\n[3/7] Generating test suite...")
    generate_test_suite(agent_dir, 'test-boiler-compute', 'test_boiler_compute', 'TestBoilerCompute', 'compute')
    print("✅ Test suite generated")

    print("\n[4/7] Generating examples...")
    generate_examples(agent_dir, 'test-boiler-compute', 'test_boiler_compute', 'compute', False)
    print("✅ Examples generated")

    print("\n[5/7] Generating documentation...")
    generate_documentation(agent_dir, 'test-boiler-compute', 'compute', False)
    print("✅ Documentation generated")

    print("\n[6/7] Generating gitignore and CI...")
    generate_gitignore(agent_dir)
    generate_ci_workflow(agent_dir, 'test-boiler-compute', 'local')
    print("✅ Gitignore and CI generated")

    print("\n[7/7] Validating generated agent...")
    result = validate_generated_agent(agent_dir)

    if not result['valid']:
        print("❌ Validation failed:")
        for error in result.get('errors', []):
            print(f"  ERROR: {error}")
        return False
    else:
        print("✅ Validation passed")

    # Check key files exist
    required_files = [
        'pack.yaml',
        'pyproject.toml',
        'README.md',
        'CHANGELOG.md',
        'LICENSE',
        '.gitignore',
        'src/test_boiler_compute/__init__.py',
        'src/test_boiler_compute/agent.py',
        'src/test_boiler_compute/schemas.py',
        'src/test_boiler_compute/provenance.py',
        'tests/__init__.py',
        'tests/conftest.py',
        'tests/test_agent.py',
        'examples/pipeline.gl.yaml',
        'examples/input.sample.json',
        '.github/workflows/ci.yml'
    ]

    print("\n[CHECK] Verifying file structure...")
    missing_files = []
    for file in required_files:
        file_path = agent_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"  ✅ {file}")

    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False

    print("\n" + "=" * 70)
    print("✅ COMPUTE TEMPLATE TEST PASSED")
    print("=" * 70)
    return True

def test_ai_template():
    """Test AI template generation."""
    print("\n" + "=" * 70)
    print("TEST 2: AI Template")
    print("=" * 70)

    # Clean previous test
    test_dir = Path("./test_output/test-ai-advisor")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    from greenlang.cli.cmd_init_agent import (
        generate_ai_agent,
        generate_common_files,
        generate_test_suite,
        generate_examples,
        generate_documentation,
        validate_generated_agent
    )

    # Create agent
    agent_dir = Path("./test_output/test-ai-advisor")
    agent_dir.mkdir(parents=True, exist_ok=True)
    (agent_dir / "src" / "test_ai_advisor").mkdir(parents=True, exist_ok=True)
    (agent_dir / "tests").mkdir(exist_ok=True)
    (agent_dir / "docs").mkdir(exist_ok=True)
    (agent_dir / "examples").mkdir(exist_ok=True)

    print(f"\n[1/5] Creating AI agent in: {agent_dir}")
    generate_ai_agent(
        agent_dir=agent_dir,
        pack_id='test-ai-advisor',
        python_pkg='test_ai_advisor',
        class_name='TestAiAdvisor',
        license='apache-2.0',
        author='Test Author',
        realtime=True,
        spec_data={}
    )
    print("✅ AI agent files generated")

    print("\n[2/5] Generating common files...")
    generate_common_files(agent_dir, 'test-ai-advisor', 'test_ai_advisor', 'apache-2.0', 'Test Author')
    print("✅ Common files generated")

    print("\n[3/5] Generating test suite...")
    generate_test_suite(agent_dir, 'test-ai-advisor', 'test_ai_advisor', 'TestAiAdvisor', 'ai')
    print("✅ Test suite generated")

    print("\n[4/5] Generating examples...")
    generate_examples(agent_dir, 'test-ai-advisor', 'test_ai_advisor', 'ai', True)
    print("✅ Examples generated")

    print("\n[5/5] Validating...")
    result = validate_generated_agent(agent_dir)

    if not result['valid']:
        print("❌ Validation failed:")
        for error in result.get('errors', []):
            print(f"  ERROR: {error}")
        return False
    else:
        print("✅ Validation passed")

    # Check AI-specific files
    ai_files = [
        'src/test_ai_advisor/ai_tools.py',
        'src/test_ai_advisor/realtime.py'
    ]

    print("\n[CHECK] Verifying AI-specific files...")
    for file in ai_files:
        file_path = agent_dir / file
        if not file_path.exists():
            print(f"  ❌ Missing: {file}")
            return False
        else:
            print(f"  ✅ {file}")

    # Check for "no naked numbers" test
    test_file = agent_dir / "tests" / "test_agent.py"
    with open(test_file, 'r') as f:
        test_content = f.read()
        if 'test_no_naked_numbers_enforcement' in test_content:
            print("  ✅ 'no naked numbers' test present")
        else:
            print("  ❌ 'no naked numbers' test missing")
            return False

    print("\n" + "=" * 70)
    print("✅ AI TEMPLATE TEST PASSED")
    print("=" * 70)
    return True

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FRMW-202 ACCEPTANCE TESTS")
    print("=" * 70)

    results = []

    # Test 1: Compute template
    try:
        results.append(("Compute Template", test_compute_template()))
    except Exception as e:
        print(f"\n[FAIL] Compute template test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Compute Template", False))

    # Test 2: AI template
    try:
        results.append(("AI Template", test_ai_template()))
    except Exception as e:
        print(f"\n[FAIL] AI template test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("AI Template", False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("\n[SUCCESS] ALL ACCEPTANCE TESTS PASSED")
        sys.exit(0)
    else:
        print("\n[ERROR] SOME TESTS FAILED")
        sys.exit(1)
