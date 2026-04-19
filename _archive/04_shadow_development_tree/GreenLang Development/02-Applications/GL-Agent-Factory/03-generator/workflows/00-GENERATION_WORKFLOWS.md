# Agent Generation Workflows

**Version**: 1.0.0
**Status**: Design
**Last Updated**: 2025-12-03
**Owner**: GL Backend Developer

---

## Executive Summary

This document defines the step-by-step workflows for generating agent packs from AgentSpec v2 YAML. It covers the complete generation pipeline from spec validation to production-ready pack assembly, including error handling, validation checkpoints, and quality gates.

**Key Workflows**:
- **Create**: Generate new agent pack from AgentSpec YAML
- **Update**: Update existing agent pack from modified spec
- **Validate**: Validate AgentSpec and generated code
- **Test**: Run generated test suite
- **Publish**: Publish agent pack to registry

---

## 1. Agent Creation Workflow

### 1.1 Command

```bash
gl agent create --spec specs/fuel_agent.yaml --output agents/fuel_agent
```

### 1.2 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                  Agent Creation Workflow                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│ 1. Load Spec     │  Load AgentSpec YAML
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Validate Spec │  Schema + Semantic validation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Parse Spec    │  Convert to Pydantic models
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Build Context │  Create template context
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Select        │  Choose templates based on
│    Templates     │  agent type (calculator/LLM)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Generate Code │  Render all templates
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 7. Validate Code │  Syntax, types, quality
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 8. Create Pack   │  Assemble directory structure
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 9. Write Files   │  Write to disk
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 10. Run Tests    │  Execute generated tests
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 11. Generate     │  Create README, docs
│     Docs         │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ✓ Complete       │  Agent pack ready
└──────────────────┘
```

### 1.3 Step-by-Step Implementation

#### Step 1: Load Spec

```python
from pathlib import Path
import yaml
from typing import Dict, Any


def load_spec(spec_path: Path) -> Dict[str, Any]:
    """
    Load AgentSpec YAML file.

    Args:
        spec_path: Path to AgentSpec YAML

    Returns:
        Parsed YAML as dictionary

    Raises:
        FileNotFoundError: If spec file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with open(spec_path, 'r') as f:
        try:
            spec_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")

    logger.info(f"Loaded spec from {spec_path}")
    return spec_data
```

#### Step 2: Validate Spec

```python
from greenlang.specs.agentspec_v2 import AgentSpecV2
from greenlang.specs.errors import GLValidationError


def validate_spec(spec_data: Dict[str, Any]) -> ValidationResult:
    """
    Validate AgentSpec against v2 schema.

    Validation stages:
    1. Schema validation (Pydantic)
    2. Semantic validation (business rules)
    3. Dependency validation (tools/calculators exist)
    4. Compliance validation (GreenLang standards)

    Args:
        spec_data: Parsed spec dictionary

    Returns:
        ValidationResult with errors if invalid

    Raises:
        GLValidationError: If validation fails
    """
    validator = AgentSpecValidator()

    # Stage 1: Schema validation
    try:
        spec = AgentSpecV2(**spec_data)
    except ValidationError as e:
        return ValidationResult(
            is_valid=False,
            errors=[f"Schema validation failed: {e}"]
        )

    # Stage 2: Semantic validation
    semantic_errors = validator.validate_semantics(spec)
    if semantic_errors:
        return ValidationResult(
            is_valid=False,
            errors=semantic_errors
        )

    # Stage 3: Dependency validation
    dependency_errors = validator.validate_dependencies(spec)
    if dependency_errors:
        return ValidationResult(
            is_valid=False,
            errors=dependency_errors
        )

    # Stage 4: Compliance validation
    compliance_errors = validator.validate_compliance(spec)
    if compliance_errors:
        return ValidationResult(
            is_valid=False,
            errors=compliance_errors
        )

    logger.info(f"Spec validation passed: {spec.id}")
    return ValidationResult(is_valid=True, errors=[])
```

#### Step 3: Parse Spec

```python
def parse_spec(spec_data: Dict[str, Any]) -> AgentSpecV2:
    """
    Parse validated spec into Pydantic model.

    Args:
        spec_data: Validated spec dictionary

    Returns:
        AgentSpecV2 model
    """
    spec = AgentSpecV2(**spec_data)
    logger.info(f"Parsed spec: {spec.id} v{spec.version}")
    return spec
```

#### Step 4: Build Context

```python
def build_context(spec: AgentSpecV2) -> Dict[str, Any]:
    """
    Build template context from spec.

    Args:
        spec: AgentSpec v2 model

    Returns:
        Template context dictionary
    """
    context_builder = TemplateContextBuilder()
    context = context_builder.build_context(spec)

    logger.info(
        f"Built context for {context['agent_name']}",
        extra={
            "agent_id": spec.id,
            "tools": len(spec.ai.tools) if spec.ai else 0
        }
    )
    return context
```

#### Step 5: Select Templates

```python
def select_templates(spec: AgentSpecV2) -> Dict[str, str]:
    """
    Select appropriate templates based on agent type.

    Args:
        spec: AgentSpec v2 model

    Returns:
        Dictionary mapping artifact type to template name
    """
    selector = TemplateSelector()

    templates = {
        "agent_class": selector.select_agent_template(spec),
        "tools": selector.select_tool_templates(spec),
        "tests": selector.select_test_templates(spec),
        "graph": selector.select_graph_template(spec),
        "deployment": selector.select_deployment_templates(spec),
        "docs": selector.select_doc_templates(spec),
    }

    logger.info(f"Selected templates: {templates}")
    return templates
```

#### Step 6: Generate Code

```python
def generate_code(
    spec: AgentSpecV2,
    context: Dict[str, Any],
    templates: Dict[str, str]
) -> Dict[str, str]:
    """
    Generate all code artifacts from templates.

    Args:
        spec: AgentSpec v2 model
        context: Template context
        templates: Selected templates

    Returns:
        Dictionary mapping file path to generated code
    """
    template_engine = GreenLangTemplateEngine(TEMPLATE_DIR)
    generated = {}

    # Generate agent class
    logger.debug("Generating agent class")
    generated["agent.py"] = template_engine.render(
        templates["agent_class"],
        context
    )

    # Generate tools
    if spec.ai and spec.ai.tools:
        logger.debug(f"Generating {len(spec.ai.tools)} tools")
        for tool in spec.ai.tools:
            tool_context = {**context, "tool": tool}
            generated["tools.py"] = template_engine.render(
                templates["tools"],
                tool_context
            )

    # Generate tests
    logger.debug("Generating test suite")
    generated["tests/test_agent.py"] = template_engine.render(
        templates["tests"],
        context
    )

    # Generate graph config
    logger.debug("Generating graph configuration")
    generated["graph/agent_graph.yaml"] = template_engine.render(
        templates["graph"],
        context
    )

    # Generate deployment
    logger.debug("Generating deployment configs")
    generated["deployment/Dockerfile"] = template_engine.render(
        templates["deployment"],
        context
    )

    # Generate docs
    logger.debug("Generating documentation")
    generated["README.md"] = template_engine.render(
        templates["docs"],
        context
    )

    logger.info(f"Generated {len(generated)} files")
    return generated
```

#### Step 7: Validate Code

```python
def validate_code(generated: Dict[str, str]) -> ValidationResult:
    """
    Validate generated code quality.

    Checks:
    - Python syntax validity
    - Type hints present
    - Docstrings present
    - No common security issues
    - Code formatting (Ruff)

    Args:
        generated: Dictionary of generated code

    Returns:
        ValidationResult
    """
    validator = GeneratedCodeValidator()
    errors = []

    for file_path, code in generated.items():
        if file_path.endswith('.py'):
            # Syntax check
            try:
                compile(code, file_path, 'exec')
            except SyntaxError as e:
                errors.append(f"{file_path}: Syntax error: {e}")

            # Type hint check
            if not validator.has_type_hints(code):
                errors.append(f"{file_path}: Missing type hints")

            # Docstring check
            if not validator.has_docstrings(code):
                errors.append(f"{file_path}: Missing docstrings")

            # Security check (basic)
            security_issues = validator.check_security(code)
            if security_issues:
                errors.extend([
                    f"{file_path}: {issue}"
                    for issue in security_issues
                ])

    if errors:
        return ValidationResult(is_valid=False, errors=errors)

    logger.info("Code validation passed")
    return ValidationResult(is_valid=True, errors=[])
```

#### Step 8: Create Pack

```python
def create_pack_structure(
    spec: AgentSpecV2,
    output_dir: Path
) -> Dict[str, Path]:
    """
    Create pack directory structure.

    Args:
        spec: AgentSpec v2 model
        output_dir: Output directory

    Returns:
        Dictionary mapping component to directory path
    """
    pack_dir = output_dir / spec.id.replace('/', '_')

    directories = {
        "root": pack_dir,
        "agent": pack_dir / "agent",
        "tests": pack_dir / "tests",
        "graph": pack_dir / "graph",
        "deployment": pack_dir / "deployment",
        "docs": pack_dir / "docs",
        "monitoring": pack_dir / "monitoring",
    }

    for name, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

    logger.info(f"Created pack structure at {pack_dir}")
    return directories
```

#### Step 9: Write Files

```python
def write_files(
    generated: Dict[str, str],
    directories: Dict[str, Path],
    spec: AgentSpecV2
) -> List[Path]:
    """
    Write generated files to disk.

    Args:
        generated: Generated code dictionary
        directories: Directory structure
        spec: AgentSpec v2 model

    Returns:
        List of written file paths
    """
    written_files = []
    root_dir = directories["root"]

    for relative_path, content in generated.items():
        file_path = root_dir / relative_path

        # Create parent directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(file_path, 'w') as f:
            f.write(content)

        written_files.append(file_path)
        logger.debug(f"Wrote file: {file_path}")

    # Copy pack.yaml
    pack_yaml_path = root_dir / "pack.yaml"
    with open(pack_yaml_path, 'w') as f:
        yaml.dump(spec.dict(), f, sort_keys=False)
    written_files.append(pack_yaml_path)

    logger.info(f"Wrote {len(written_files)} files")
    return written_files
```

#### Step 10: Run Tests

```python
def run_generated_tests(pack_dir: Path) -> TestResult:
    """
    Run generated test suite.

    Args:
        pack_dir: Pack directory path

    Returns:
        TestResult with pass/fail status
    """
    import subprocess

    logger.info("Running generated tests")

    result = subprocess.run(
        ["pytest", str(pack_dir / "tests"), "-v"],
        capture_output=True,
        text=True,
        cwd=pack_dir
    )

    if result.returncode == 0:
        logger.info("All tests passed")
        return TestResult(success=True, output=result.stdout)
    else:
        logger.error(f"Tests failed: {result.stderr}")
        return TestResult(success=False, output=result.stderr)
```

#### Step 11: Generate Docs

```python
def generate_documentation(
    spec: AgentSpecV2,
    pack_dir: Path
) -> List[Path]:
    """
    Generate comprehensive documentation.

    Args:
        spec: AgentSpec v2 model
        pack_dir: Pack directory

    Returns:
        List of generated doc files
    """
    doc_generator = DocumentationGenerator()
    docs = []

    # README.md
    readme_path = pack_dir / "README.md"
    readme_content = doc_generator.generate_readme(spec)
    readme_path.write_text(readme_content)
    docs.append(readme_path)

    # ARCHITECTURE.md
    arch_path = pack_dir / "ARCHITECTURE.md"
    arch_content = doc_generator.generate_architecture(spec)
    arch_path.write_text(arch_content)
    docs.append(arch_path)

    # API.md
    api_path = pack_dir / "docs" / "API.md"
    api_content = doc_generator.generate_api_docs(spec)
    api_path.write_text(api_content)
    docs.append(api_path)

    logger.info(f"Generated {len(docs)} documentation files")
    return docs
```

### 1.4 Complete Workflow Function

```python
def create_agent_workflow(
    spec_path: Path,
    output_dir: Path,
    overwrite: bool = False,
    skip_tests: bool = False
) -> AgentPack:
    """
    Complete agent creation workflow.

    Args:
        spec_path: Path to AgentSpec YAML
        output_dir: Output directory
        overwrite: Whether to overwrite existing pack
        skip_tests: Whether to skip running tests

    Returns:
        AgentPack with metadata

    Raises:
        ValidationError: If validation fails
        GenerationError: If generation fails
    """
    logger.info(f"Starting agent creation from {spec_path}")

    try:
        # Step 1: Load Spec
        spec_data = load_spec(spec_path)

        # Step 2: Validate Spec
        validation_result = validate_spec(spec_data)
        if not validation_result.is_valid:
            raise ValidationError(validation_result.errors)

        # Step 3: Parse Spec
        spec = parse_spec(spec_data)

        # Step 4: Build Context
        context = build_context(spec)

        # Step 5: Select Templates
        templates = select_templates(spec)

        # Step 6: Generate Code
        generated = generate_code(spec, context, templates)

        # Step 7: Validate Code
        code_validation = validate_code(generated)
        if not code_validation.is_valid:
            raise GenerationError(code_validation.errors)

        # Step 8: Create Pack Structure
        directories = create_pack_structure(spec, output_dir)

        # Step 9: Write Files
        written_files = write_files(generated, directories, spec)

        # Step 10: Run Tests
        if not skip_tests:
            test_result = run_generated_tests(directories["root"])
            if not test_result.success:
                logger.warning("Tests failed, but pack was generated")

        # Step 11: Generate Docs
        docs = generate_documentation(spec, directories["root"])

        # Create AgentPack metadata
        pack = AgentPack(
            spec_id=spec.id,
            version=spec.version,
            pack_dir=directories["root"],
            files=written_files,
            docs=docs,
            timestamp=datetime.now(),
            generator_version="1.0.0"
        )

        logger.info(f"Agent pack created successfully: {pack.pack_dir}")
        return pack

    except Exception as e:
        logger.error(f"Agent creation failed: {e}", exc_info=True)
        raise
```

---

## 2. Agent Update Workflow

### 2.1 Command

```bash
gl agent update --spec specs/fuel_agent.yaml --pack agents/fuel_agent
```

### 2.2 Workflow Diagram

```
┌──────────────────┐
│ 1. Load New Spec │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. Load Existing │
│    Pack          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. Detect        │
│    Changes       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. Generate Diff │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 5. Backup        │
│    Existing Pack │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 6. Regenerate    │
│    Changed Files │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 7. Merge Custom  │
│    Code          │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 8. Run Tests     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ✓ Updated        │
└──────────────────┘
```

### 2.3 Implementation

```python
def update_agent_workflow(
    spec_path: Path,
    pack_dir: Path,
    backup: bool = True
) -> UpdateResult:
    """
    Update existing agent pack from modified spec.

    Args:
        spec_path: Path to updated AgentSpec
        pack_dir: Existing pack directory
        backup: Whether to backup existing pack

    Returns:
        UpdateResult with changes

    Raises:
        ValidationError: If validation fails
        UpdateError: If update fails
    """
    logger.info(f"Updating agent pack at {pack_dir}")

    # Load new spec
    new_spec_data = load_spec(spec_path)
    new_spec = parse_spec(new_spec_data)

    # Load existing pack.yaml
    existing_pack_path = pack_dir / "pack.yaml"
    if not existing_pack_path.exists():
        raise UpdateError("pack.yaml not found in pack directory")

    existing_spec_data = load_spec(existing_pack_path)
    existing_spec = parse_spec(existing_spec_data)

    # Detect changes
    changes = detect_changes(existing_spec, new_spec)

    if not changes:
        logger.info("No changes detected")
        return UpdateResult(success=True, changes=[])

    logger.info(f"Detected {len(changes)} changes")

    # Backup existing pack
    if backup:
        backup_dir = pack_dir.parent / f"{pack_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copytree(pack_dir, backup_dir)
        logger.info(f"Backed up to {backup_dir}")

    # Regenerate changed files
    context = build_context(new_spec)
    templates = select_templates(new_spec)

    for change in changes:
        if change.type == "input_added":
            # Regenerate agent.py with new input field
            regenerate_agent_class(new_spec, context, templates, pack_dir)
        elif change.type == "output_added":
            # Regenerate agent.py with new output field
            regenerate_agent_class(new_spec, context, templates, pack_dir)
        elif change.type == "tool_added":
            # Generate new tool wrapper
            regenerate_tools(new_spec, context, templates, pack_dir)

    # Run tests
    test_result = run_generated_tests(pack_dir)
    if not test_result.success:
        logger.error("Tests failed after update")
        return UpdateResult(success=False, changes=changes, errors=[test_result.output])

    logger.info("Agent pack updated successfully")
    return UpdateResult(success=True, changes=changes)
```

---

## 3. Validation Workflow

### 3.1 Command

```bash
gl agent validate --spec specs/fuel_agent.yaml
```

### 3.2 Validation Stages

```python
def validate_workflow(spec_path: Path) -> ValidationReport:
    """
    Comprehensive validation of AgentSpec.

    Validation stages:
    1. YAML syntax
    2. AgentSpec v2 schema
    3. Semantic rules
    4. Dependencies
    5. Compliance
    6. Best practices

    Args:
        spec_path: Path to AgentSpec

    Returns:
        ValidationReport with detailed results
    """
    report = ValidationReport(spec_path=spec_path)

    # Stage 1: YAML syntax
    logger.info("Stage 1: Validating YAML syntax")
    try:
        spec_data = load_spec(spec_path)
        report.add_success("yaml_syntax", "Valid YAML")
    except yaml.YAMLError as e:
        report.add_error("yaml_syntax", f"Invalid YAML: {e}")
        return report  # Can't continue if YAML is invalid

    # Stage 2: Schema validation
    logger.info("Stage 2: Validating AgentSpec v2 schema")
    try:
        spec = AgentSpecV2(**spec_data)
        report.add_success("schema", "Valid AgentSpec v2 schema")
    except ValidationError as e:
        report.add_error("schema", f"Schema validation failed: {e}")
        return report

    # Stage 3: Semantic validation
    logger.info("Stage 3: Validating semantic rules")
    semantic_validator = SemanticValidator()
    semantic_errors = semantic_validator.validate(spec)
    if semantic_errors:
        for error in semantic_errors:
            report.add_error("semantics", error)
    else:
        report.add_success("semantics", "All semantic rules passed")

    # Stage 4: Dependency validation
    logger.info("Stage 4: Validating dependencies")
    dependency_validator = DependencyValidator()
    dependency_errors = dependency_validator.validate(spec)
    if dependency_errors:
        for error in dependency_errors:
            report.add_error("dependencies", error)
    else:
        report.add_success("dependencies", "All dependencies available")

    # Stage 5: Compliance validation
    logger.info("Stage 5: Validating compliance")
    compliance_validator = ComplianceValidator()
    compliance_errors = compliance_validator.validate(spec)
    if compliance_errors:
        for error in compliance_errors:
            report.add_error("compliance", error)
    else:
        report.add_success("compliance", "Meets GreenLang standards")

    # Stage 6: Best practices
    logger.info("Stage 6: Checking best practices")
    best_practice_checker = BestPracticeChecker()
    warnings = best_practice_checker.check(spec)
    for warning in warnings:
        report.add_warning("best_practices", warning)

    return report
```

---

## 4. Test Workflow

### 4.1 Command

```bash
gl agent test --pack agents/fuel_agent
```

### 4.2 Test Suite Execution

```python
def test_workflow(pack_dir: Path, coverage: bool = True) -> TestReport:
    """
    Run complete test suite for agent pack.

    Test stages:
    1. Unit tests
    2. Integration tests
    3. Determinism tests
    4. Performance benchmarks
    5. Coverage report

    Args:
        pack_dir: Agent pack directory
        coverage: Whether to generate coverage report

    Returns:
        TestReport with results
    """
    report = TestReport(pack_dir=pack_dir)

    # Stage 1: Unit tests
    logger.info("Running unit tests")
    unit_result = run_pytest(
        pack_dir / "tests" / "test_agent.py",
        markers="unit"
    )
    report.add_result("unit", unit_result)

    # Stage 2: Integration tests
    logger.info("Running integration tests")
    integration_result = run_pytest(
        pack_dir / "tests" / "test_integration.py",
        markers="integration"
    )
    report.add_result("integration", integration_result)

    # Stage 3: Determinism tests
    logger.info("Running determinism tests")
    determinism_result = run_pytest(
        pack_dir / "tests" / "test_determinism.py",
        markers="determinism"
    )
    report.add_result("determinism", determinism_result)

    # Stage 4: Performance benchmarks
    logger.info("Running performance benchmarks")
    performance_result = run_pytest(
        pack_dir / "tests" / "test_performance.py",
        markers="benchmark"
    )
    report.add_result("performance", performance_result)

    # Stage 5: Coverage report
    if coverage:
        logger.info("Generating coverage report")
        coverage_result = run_coverage(pack_dir)
        report.coverage = coverage_result

    return report
```

---

## 5. Publish Workflow

### 5.1 Command

```bash
gl agent publish --pack agents/fuel_agent --registry https://registry.greenlang.io
```

### 5.2 Publishing Pipeline

```python
def publish_workflow(
    pack_dir: Path,
    registry_url: str,
    dry_run: bool = False
) -> PublishResult:
    """
    Publish agent pack to registry.

    Publishing stages:
    1. Validate pack completeness
    2. Run all tests
    3. Generate SBOM
    4. Sign package
    5. Upload to registry
    6. Verify upload

    Args:
        pack_dir: Agent pack directory
        registry_url: Registry URL
        dry_run: If True, don't actually publish

    Returns:
        PublishResult
    """
    logger.info(f"Publishing pack from {pack_dir} to {registry_url}")

    # Stage 1: Validate pack
    logger.info("Validating pack completeness")
    if not validate_pack_complete(pack_dir):
        raise PublishError("Pack validation failed")

    # Stage 2: Run tests
    logger.info("Running test suite")
    test_report = test_workflow(pack_dir)
    if not test_report.all_passed():
        raise PublishError("Tests failed, cannot publish")

    # Stage 3: Generate SBOM
    logger.info("Generating SBOM")
    sbom = generate_sbom(pack_dir)
    sbom_path = pack_dir / "sbom" / "sbom.json"
    sbom_path.write_text(json.dumps(sbom, indent=2))

    # Stage 4: Sign package
    logger.info("Signing package")
    signature = sign_package(pack_dir)

    # Stage 5: Upload to registry
    if not dry_run:
        logger.info(f"Uploading to {registry_url}")
        upload_result = upload_to_registry(
            pack_dir=pack_dir,
            registry_url=registry_url,
            signature=signature
        )

        # Stage 6: Verify upload
        logger.info("Verifying upload")
        if not verify_upload(upload_result.pack_id, registry_url):
            raise PublishError("Upload verification failed")

        logger.info(f"Published successfully: {upload_result.pack_url}")
        return PublishResult(
            success=True,
            pack_id=upload_result.pack_id,
            pack_url=upload_result.pack_url
        )
    else:
        logger.info("Dry run complete, no upload performed")
        return PublishResult(success=True, dry_run=True)
```

---

## 6. Error Handling

### 6.1 Error Recovery Strategy

```python
class WorkflowError(Exception):
    """Base class for workflow errors."""
    pass


class ValidationError(WorkflowError):
    """Validation failed."""
    pass


class GenerationError(WorkflowError):
    """Code generation failed."""
    pass


class UpdateError(WorkflowError):
    """Update failed."""
    pass


class PublishError(WorkflowError):
    """Publishing failed."""
    pass


def handle_workflow_error(
    error: WorkflowError,
    stage: str,
    context: Dict[str, Any]
) -> None:
    """
    Handle workflow error with recovery.

    Args:
        error: Workflow error
        stage: Stage where error occurred
        context: Workflow context
    """
    logger.error(
        f"Workflow failed at stage '{stage}': {error}",
        extra={"stage": stage, "context": context}
    )

    # Attempt recovery
    if isinstance(error, ValidationError):
        # Show validation errors and suggest fixes
        show_validation_help(error)
    elif isinstance(error, GenerationError):
        # Clean up partial generation
        cleanup_partial_generation(context.get("pack_dir"))
    elif isinstance(error, UpdateError):
        # Restore from backup
        restore_from_backup(context.get("backup_dir"))
    elif isinstance(error, PublishError):
        # Rollback upload
        rollback_upload(context.get("upload_id"))
```

---

## 7. Progress Reporting

### 7.1 Progress Tracking

```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def create_agent_with_progress(spec_path: Path, output_dir: Path) -> AgentPack:
    """Create agent with rich progress reporting."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:

        # Add tasks
        load_task = progress.add_task("Loading spec...", total=None)
        validate_task = progress.add_task("Validating spec...", total=None)
        generate_task = progress.add_task("Generating code...", total=None)
        test_task = progress.add_task("Running tests...", total=None)
        doc_task = progress.add_task("Generating docs...", total=None)

        try:
            # Load spec
            spec_data = load_spec(spec_path)
            progress.update(load_task, completed=True)

            # Validate
            validate_spec(spec_data)
            progress.update(validate_task, completed=True)

            # Generate
            spec = parse_spec(spec_data)
            context = build_context(spec)
            templates = select_templates(spec)
            generated = generate_code(spec, context, templates)
            progress.update(generate_task, completed=True)

            # Write files
            directories = create_pack_structure(spec, output_dir)
            write_files(generated, directories, spec)

            # Run tests
            run_generated_tests(directories["root"])
            progress.update(test_task, completed=True)

            # Generate docs
            generate_documentation(spec, directories["root"])
            progress.update(doc_task, completed=True)

            return AgentPack(
                spec_id=spec.id,
                pack_dir=directories["root"]
            )

        except Exception as e:
            logger.error(f"Failed: {e}")
            raise
```

---

## 8. Dry Run Mode

### 8.1 Dry Run Implementation

```python
def create_agent_dry_run(spec_path: Path) -> DryRunResult:
    """
    Dry run of agent creation (no files written).

    Args:
        spec_path: Path to AgentSpec

    Returns:
        DryRunResult with preview of what would be generated
    """
    logger.info(f"Dry run for {spec_path}")

    # Load and validate
    spec_data = load_spec(spec_path)
    validation_result = validate_spec(spec_data)

    if not validation_result.is_valid:
        return DryRunResult(
            success=False,
            errors=validation_result.errors
        )

    # Parse and generate (in-memory only)
    spec = parse_spec(spec_data)
    context = build_context(spec)
    templates = select_templates(spec)
    generated = generate_code(spec, context, templates)

    # Validate generated code
    code_validation = validate_code(generated)

    return DryRunResult(
        success=code_validation.is_valid,
        files=list(generated.keys()),
        lines_of_code=sum(len(code.split('\n')) for code in generated.values()),
        templates_used=templates,
        errors=code_validation.errors if not code_validation.is_valid else []
    )
```

---

## Summary

The Generation Workflows provide:

1. **Create Workflow**: Generate new agent packs from specs
2. **Update Workflow**: Update existing packs from modified specs
3. **Validate Workflow**: Comprehensive spec validation
4. **Test Workflow**: Complete test suite execution
5. **Publish Workflow**: Package signing and registry publishing
6. **Error Handling**: Recovery strategies for common failures
7. **Progress Reporting**: Rich CLI progress indicators
8. **Dry Run Mode**: Preview generation without file writes

**Next Step**: Implement CLI commands that execute these workflows.

---

**Document Status**: Design Complete
**Implementation Status**: Pending
