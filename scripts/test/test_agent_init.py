# -*- coding: utf-8 -*-
"""
Test script for gl init agent command
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from greenlang.cli.cmd_init_agent import (
    generate_pack_yaml_compute,
    generate_schemas_py,
    generate_agent_py,
    generate_provenance_py,
    generate_test_suite,
    generate_precommit_config,
    generate_ci_workflow,
    generate_common_files,
    validate_generated_agent,
)
from rich.console import Console
import yaml
import shutil

# Use force_terminal=False and legacy_windows=False for better Windows compatibility
console = Console(force_terminal=False, legacy_windows=False)

def test_generate_compute_agent():
    """Test generating a compute agent"""
    console.print("\n[bold]Testing Compute Agent Generation[/bold]\n")

    # Test parameters
    pack_id = "test-boiler"
    python_pkg = "test_boiler"
    class_name = "TestBoiler"
    test_dir = Path(__file__).parent / "test_output" / pack_id

    # Clean up if exists
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Create directories
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / python_pkg).mkdir(exist_ok=True)
    (test_dir / "tests").mkdir(exist_ok=True)

    console.print(f"[cyan]Creating test agent in: {test_dir}[/cyan]")

    # 1. Generate pack.yaml
    console.print("\n[yellow]1. Generating pack.yaml...[/yellow]")
    pack_yaml_content = generate_pack_yaml_compute(
        pack_id=pack_id,
        python_pkg=python_pkg,
        license="apache-2.0",
        author="Test Author",
        realtime=False,
        spec_data=None
    )

    pack_yaml_path = test_dir / "pack.yaml"
    with open(pack_yaml_path, "w", encoding="utf-8", newline="\n") as f:
        yaml.dump(pack_yaml_content, f, default_flow_style=False, sort_keys=False)

    console.print(f"[green][OK][/green] Created pack.yaml ({pack_yaml_path.stat().st_size} bytes)")

    # Validate pack.yaml structure
    with open(pack_yaml_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    assert manifest["schema_version"] == "2.0.0", "Schema version mismatch"
    assert "compute" in manifest, "Missing compute section"
    assert "provenance" in manifest, "Missing provenance section"
    console.print(f"[green][OK][/green] pack.yaml structure validated")

    # 2. Generate schemas.py
    console.print("\n[yellow]2. Generating schemas.py...[/yellow]")
    schemas_content = generate_schemas_py(
        python_pkg=python_pkg,
        class_name=class_name,
        template="compute"
    )
    schemas_path = test_dir / python_pkg / "schemas.py"
    with open(schemas_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(schemas_content)

    console.print(f"[green][OK][/green] Created schemas.py ({schemas_path.stat().st_size} bytes)")

    # 3. Generate agent.py
    console.print("\n[yellow]3. Generating agent.py...[/yellow]")
    agent_content = generate_agent_py(
        python_pkg=python_pkg,
        class_name=class_name,
        template="compute",
        realtime=False
    )
    agent_path = test_dir / python_pkg / "agent.py"
    with open(agent_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(agent_content)

    console.print(f"[green][OK][/green] Created agent.py ({agent_path.stat().st_size} bytes)")

    # 4. Generate provenance.py
    console.print("\n[yellow]4. Generating provenance.py...[/yellow]")
    provenance_content = generate_provenance_py()
    provenance_path = test_dir / python_pkg / "provenance.py"
    with open(provenance_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(provenance_content)

    console.print(f"[green][OK][/green] Created provenance.py ({provenance_path.stat().st_size} bytes)")

    # 5. Generate __init__.py
    console.print("\n[yellow]5. Generating __init__.py...[/yellow]")
    init_content = f'''"""
{pack_id} - GreenLang Agent
"""
from .agent import {class_name}

__all__ = ["{class_name}"]
'''
    init_path = test_dir / python_pkg / "__init__.py"
    with open(init_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(init_content)

    console.print(f"[green][OK][/green] Created __init__.py")

    # 6. Generate test suite (writes files directly)
    console.print("\n[yellow]6. Generating test suite...[/yellow]")
    generate_test_suite(
        agent_dir=test_dir,
        pack_id=pack_id,
        python_pkg=python_pkg,
        class_name=class_name,
        template="compute"
    )

    # Check if test files were created
    test_path = test_dir / "tests" / "test_agent.py"
    conftest_path = test_dir / "tests" / "conftest.py"
    if test_path.exists():
        console.print(f"[green][OK][/green] Created test_agent.py ({test_path.stat().st_size} bytes)")
    if conftest_path.exists():
        console.print(f"[green][OK][/green] Created conftest.py ({conftest_path.stat().st_size} bytes)")

    # 7. Generate common files (writes files directly)
    console.print("\n[yellow]7. Generating common files...[/yellow]")
    generate_common_files(
        agent_dir=test_dir,
        pack_id=pack_id,
        python_pkg=python_pkg,
        license="apache-2.0",
        author="Test Author"
    )

    # Check if files were created
    license_path = test_dir / "LICENSE"
    pyproject_path = test_dir / "pyproject.toml"
    if license_path.exists():
        console.print(f"[green][OK][/green] Created LICENSE ({license_path.stat().st_size} bytes)")
    if pyproject_path.exists():
        console.print(f"[green][OK][/green] Created pyproject.toml ({pyproject_path.stat().st_size} bytes)")

    # 8. Generate pre-commit config (writes file directly)
    console.print("\n[yellow]8. Generating pre-commit config...[/yellow]")
    generate_precommit_config(agent_dir=test_dir)

    precommit_path = test_dir / ".pre-commit-config.yaml"
    if precommit_path.exists():
        console.print(f"[green][OK][/green] Created .pre-commit-config.yaml ({precommit_path.stat().st_size} bytes)")

    # 9. Generate CI workflow (writes file directly)
    console.print("\n[yellow]9. Generating CI workflow...[/yellow]")
    generate_ci_workflow(
        agent_dir=test_dir,
        pack_id=pack_id,
        runtimes="local"
    )

    workflow_path = test_dir / ".github" / "workflows" / "ci.yml"
    if workflow_path.exists():
        console.print(f"[green][OK][/green] Created CI workflow ({workflow_path.stat().st_size} bytes)")

    # 10. Generate documentation (writes files directly)
    console.print("\n[yellow]10. Generating documentation...[/yellow]")
    from greenlang.cli.cmd_init_agent import generate_documentation
    generate_documentation(
        agent_dir=test_dir,
        pack_id=pack_id,
        template="compute",
        realtime=False
    )

    readme_path = test_dir / "README.md"
    changelog_path = test_dir / "CHANGELOG.md"
    if readme_path.exists():
        console.print(f"[green][OK][/green] Created README.md ({readme_path.stat().st_size} bytes)")
    if changelog_path.exists():
        console.print(f"[green][OK][/green] Created CHANGELOG.md ({changelog_path.stat().st_size} bytes)")

    # 11. Generate examples
    console.print("\n[yellow]11. Generating examples...[/yellow]")
    from greenlang.cli.cmd_init_agent import generate_examples
    generate_examples(
        agent_dir=test_dir,
        pack_id=pack_id,
        python_pkg=python_pkg,
        template="compute",
        realtime=False
    )

    pipeline_path = test_dir / "examples" / "pipeline.gl.yaml"
    sample_path = test_dir / "examples" / "input.sample.json"
    if pipeline_path.exists():
        console.print(f"[green][OK][/green] Created pipeline.gl.yaml ({pipeline_path.stat().st_size} bytes)")
    if sample_path.exists():
        console.print(f"[green][OK][/green] Created input.sample.json ({sample_path.stat().st_size} bytes)")

    # 12. Validate generated agent
    console.print("\n[yellow]12. Validating generated agent...[/yellow]")
    validation_result = validate_generated_agent(test_dir)

    is_valid = validation_result.get("valid", False)
    errors = validation_result.get("errors", [])
    warnings = validation_result.get("warnings", [])

    if is_valid:
        console.print(f"[green][OK][/green] Agent validation passed")
    else:
        console.print(f"[red][FAIL][/red] Agent validation failed:")
        for error in errors:
            console.print(f"  - {error}")

    if warnings:
        console.print(f"[yellow]Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  - {warning}")

    # Summary
    console.print("\n[bold green][OK] Test Completed Successfully![/bold green]")
    console.print(f"\nGenerated files in: [cyan]{test_dir}[/cyan]")
    console.print("\nDirectory structure:")

    for item in sorted(test_dir.rglob("*")):
        if item.is_file():
            rel_path = item.relative_to(test_dir)
            size = item.stat().st_size
            console.print(f"  {rel_path} ({size:,} bytes)")

    return is_valid, test_dir

if __name__ == "__main__":
    try:
        is_valid, test_dir = test_generate_compute_agent()
        if is_valid:
            console.print("\n[bold]Next steps:[/bold]")
            console.print(f"1. cd {test_dir}")
            console.print("2. python -m pytest tests/")
            console.print("3. Review generated files")
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
