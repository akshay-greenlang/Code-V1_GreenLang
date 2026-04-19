# GreenLang CBAM MVP - Ralphy Task List

Based on PRD: `../2026_PRD_MVP/GreenLang_PRD_MVP_2026.md`

## Overview

Building the CBAM Compliance Pack MVP - a deterministic CLI tool that generates EU CBAM Transitional Registry XML reports with full audit bundles.

---

## Milestone 1: Pack Spec + Input Template

### M1.1 - Create project scaffolding and pyproject.toml
- [ ] Create `pyproject.toml` with dependencies: click, pydantic, openpyxl, lxml, pyyaml, python-dateutil
- [ ] Create `src/cbam_pack/__init__.py` with version "1.0.0"
- [ ] Create `src/cbam_pack/cli.py` with basic click CLI structure for `gl run cbam` command
- [ ] Create `requirements.txt` and `requirements-dev.txt`
- [ ] Add pytest, ruff, mypy to dev dependencies

### M1.2 - Create JSON Schema for import ledger validation
- [ ] Create `schemas/imports.schema.json` with fields: line_id (required string), quarter (enum Q1-Q4), year (integer 2023-2030), cn_code (string 8-digit), product_description (string), country_of_origin (ISO 3166-1 alpha-2), quantity (number positive), unit (enum kg|tonnes)
- [ ] Add optional fields: supplier_id, installation_id, supplier_direct_emissions (number >=0), supplier_indirect_emissions (number >=0), supplier_certificate_ref
- [ ] Document validation rules in schema description

### M1.3 - Create JSON Schema for config file validation
- [ ] Create `schemas/config.schema.json` with declarant object (name, eori_number, address, contact), reporting_period (quarter, year), optional representative, settings (factor_policy, aggregation, xml_schema_version, validation_strictness)
- [ ] Add required field markers and validation patterns (EORI format, email format)

### M1.4 - Create import ledger template files
- [ ] Create `templates/imports_template.csv` with headers: line_id,quarter,year,cn_code,product_description,country_of_origin,quantity,unit,supplier_id,installation_id,supplier_direct_emissions,supplier_indirect_emissions,supplier_certificate_ref
- [ ] Add 3 example rows with realistic steel import data from China
- [ ] Create `templates/imports_template.xlsx` with same structure using openpyxl, add column validation where possible

### M1.5 - Create CBAM config template
- [ ] Create `templates/cbam_config_template.yaml` with full example configuration including declarant info, reporting_period Q1 2025, settings with defaults_first policy
- [ ] Add comments explaining each field

### M1.6 - Create INPUT_SPECIFICATION.md documentation
- [ ] Create `docs/INPUT_SPECIFICATION.md` documenting all fields, validation rules, supported CN codes (72xx, 73xx, 76xx), country codes, units
- [ ] Include examples of valid and invalid inputs

---

## Milestone 2: Validator + Error Taxonomy

### M2.1 - Create error taxonomy module
- [ ] Create `src/cbam_pack/errors.py` with CBAMError base exception class
- [ ] Define ValidationError (VAL-001 to VAL-010), CalculationError (CALC-001 to CALC-004), ExportError (XML-001 to XML-003) classes
- [ ] Each error has: code, category, message template, location (file:row:column when applicable)
- [ ] Create `format_error()` function that produces actionable error messages with fix guidance

### M2.2 - Create data models with Pydantic
- [ ] Create `src/cbam_pack/models.py` with Pydantic models: ImportLineItem, EmissionsResult, Claim, EvidenceRef, Assumption, RunManifest
- [ ] Use Decimal type for all numeric emissions/financial values
- [ ] Add validators for cn_code (8 digits, starts with 72/73/76), country_of_origin (ISO 3166-1 alpha-2), quantity (positive)
- [ ] Add model_config for JSON serialization with proper decimal handling

### M2.3 - Create schema validator module
- [ ] Create `src/cbam_pack/validators/schema_validator.py`
- [ ] Implement `validate_imports_schema(data: dict) -> ValidationResult` using JSON Schema
- [ ] Implement `validate_config_schema(data: dict) -> ValidationResult`
- [ ] Return first error encountered (fail-fast) with row/column location

### M2.4 - Create business rule validator
- [ ] Create `src/cbam_pack/validators/business_validator.py`
- [ ] Implement CN code validation: must be 8 digits, must start with 72, 73, or 76
- [ ] Implement country code validation against ISO 3166-1 alpha-2 list
- [ ] Implement quantity validation (positive, reasonable range)
- [ ] Implement unit validation (kg or tonnes only)
- [ ] Implement duplicate line_id detection
- [ ] Implement quarter/year consistency check

### M2.5 - Create file reader module
- [ ] Create `src/cbam_pack/readers/file_reader.py`
- [ ] Implement `read_csv(path: Path) -> list[dict]` with proper encoding handling
- [ ] Implement `read_xlsx(path: Path) -> list[dict]` using openpyxl
- [ ] Implement `read_yaml(path: Path) -> dict` for config files
- [ ] Auto-detect file type from extension

### M2.6 - Create validation orchestrator
- [ ] Create `src/cbam_pack/validators/__init__.py`
- [ ] Implement `validate_inputs(imports_path: Path, config_path: Path) -> ValidationResult`
- [ ] Chain schema validation then business rule validation
- [ ] Return comprehensive ValidationResult with is_valid, errors list, validated_data

### M2.7 - Write unit tests for validators
- [ ] Create `tests/test_validators.py`
- [ ] Test all VAL-001 to VAL-010 error cases
- [ ] Test valid inputs pass validation
- [ ] Test edge cases: empty file, missing columns, extra columns
- [ ] Achieve 80%+ coverage on validator modules

---

## Milestone 3: Calculation + Method Selection

### M3.1 - Create CBAM emission factors database
- [ ] Create `data/emission_factors/cbam_defaults_2024.json` with default emission factors for Steel (CN 72xx, 73xx) and Aluminum (CN 76xx) by country of origin (CN, IN, RU, TR, UA, other)
- [ ] Include direct_emissions_factor (tCO2e/tonne) and indirect_emissions_factor (tCO2e/tonne) for each product/country combination
- [ ] Add metadata: source (EU Commission Implementing Regulation), version, effective_date, expiry_date

### M3.2 - Create electricity emission factors by country
- [ ] Create `data/emission_factors/electricity_factors_2024.json` with grid emission factors (tCO2e/MWh) for major countries
- [ ] Include CN, IN, RU, TR, UA, DE, FR, IT, ES, PL, and "OTHER" default
- [ ] Add source attribution

### M3.3 - Create emission factor library module
- [ ] Create `src/cbam_pack/factors/factor_library.py`
- [ ] Implement `EmissionFactorLibrary` class that loads factors from JSON files
- [ ] Implement `get_factor(cn_code: str, country: str, factor_type: str) -> EmissionFactor`
- [ ] Return factor with source, version, effective_date for audit trail
- [ ] Raise CALC-001 if factor not found

### M3.4 - Create unit normalizer module
- [ ] Create `src/cbam_pack/calculators/unit_normalizer.py`
- [ ] Implement `normalize_quantity(value: Decimal, from_unit: str, to_unit: str) -> Decimal`
- [ ] Support conversions: kg to tonnes (divide by 1000), tonnes to kg (multiply by 1000)
- [ ] Raise CALC-003 for unsupported unit conversions
- [ ] Log all conversions for audit trail

### M3.5 - Create CBAM emissions calculator
- [ ] Create `src/cbam_pack/calculators/emissions_calculator.py`
- [ ] Implement `CBAMCalculator` class
- [ ] Implement `calculate_line_emissions(line: ImportLineItem, factors: EmissionFactorLibrary) -> EmissionsResult`
- [ ] Logic: if supplier_direct_emissions provided, use it (method=supplier_specific); else use default factor (method=default)
- [ ] Same logic for indirect emissions
- [ ] Calculate: direct = factor * quantity, indirect = factor * quantity, total = direct + indirect
- [ ] Use Decimal for all calculations, round to 4 decimal places intermediate, 2 for output

### M3.6 - Create assumptions recorder
- [ ] Create `src/cbam_pack/audit/assumptions.py`
- [ ] Implement `AssumptionsRecorder` class
- [ ] Track each assumption: type (default_factor, method_selection), description, rationale, applies_to lines
- [ ] Generate assumptions.json output

### M3.7 - Create calculation orchestrator
- [ ] Create `src/cbam_pack/calculators/__init__.py`
- [ ] Implement `calculate_all_emissions(lines: list[ImportLineItem], config: CBAMConfig) -> CalculationResult`
- [ ] Process each line, collect results and assumptions
- [ ] Aggregate by CN code + country of origin as per config
- [ ] Return CalculationResult with results, assumptions, statistics

### M3.8 - Write unit tests for calculator
- [ ] Create `tests/test_calculator.py`
- [ ] Test direct emissions calculation with default factors
- [ ] Test supplier-specific override
- [ ] Test indirect emissions calculation
- [ ] Test aggregation logic
- [ ] Test unit conversion
- [ ] Verify Decimal precision

---

## Milestone 4: XML Export + XSD Validation

### M4.1 - Obtain and bundle EU CBAM XSD schema
- [ ] Create `schemas/cbam_transitional_registry_v2.xsd` - create a simplified but valid XSD based on EU CBAM Transitional Registry requirements
- [ ] Include elements: CBAMReport, Header, ReportingPeriod, Declarant, ImportedGoods, GoodsItem, EmbeddedEmissions, Summary
- [ ] Document schema version in README

### M4.2 - Create XML generator module
- [ ] Create `src/cbam_pack/exporters/xml_generator.py`
- [ ] Implement `CBAMXMLGenerator` class using lxml
- [ ] Implement `generate_xml(results: CalculationResult, config: CBAMConfig) -> str`
- [ ] Build XML structure: Header (ReportingPeriod, Declarant), ImportedGoods (list of GoodsItem with emissions), Summary (totals)
- [ ] Use proper XML namespaces
- [ ] Format output with pretty printing

### M4.3 - Create XSD validator module
- [ ] Create `src/cbam_pack/validators/xsd_validator.py`
- [ ] Implement `validate_xml_against_xsd(xml_content: str, xsd_path: Path) -> ValidationResult`
- [ ] Use lxml for validation
- [ ] Parse XSD validation errors into XML-001 format with line numbers

### M4.4 - Create XML export orchestrator
- [ ] Create `src/cbam_pack/exporters/__init__.py`
- [ ] Implement `export_cbam_report(results: CalculationResult, config: CBAMConfig, output_path: Path) -> ExportResult`
- [ ] Generate XML, validate against XSD, write to file
- [ ] Return ExportResult with success status, xml_path, validation_result

### M4.5 - Write unit tests for XML export
- [ ] Create `tests/test_xml_export.py`
- [ ] Test XML generation produces valid structure
- [ ] Test XSD validation passes for valid XML
- [ ] Test XSD validation fails with proper errors for invalid XML
- [ ] Test encoding handling for special characters

---

## Milestone 5: Audit Bundle + Run Manifest

### M5.1 - Create claims generator
- [ ] Create `src/cbam_pack/audit/claims.py`
- [ ] Implement `generate_claims(results: CalculationResult, input_refs: list[EvidenceRef]) -> list[Claim]`
- [ ] Each emission value becomes a Claim with: claim_id (UUID), value, unit, claim_type, method, evidence_refs, factor_ref

### M5.2 - Create lineage graph generator
- [ ] Create `src/cbam_pack/audit/lineage.py`
- [ ] Implement `generate_lineage(inputs: list[Path], transformations: list[str], outputs: list[Path]) -> LineageGraph`
- [ ] Create nodes for inputs, transformations (agents), outputs
- [ ] Create edges showing data flow
- [ ] Output as JSON with nodes[] and edges[]

### M5.3 - Create evidence packager
- [ ] Create `src/cbam_pack/audit/evidence.py`
- [ ] Implement `package_evidence(input_files: list[Path], output_dir: Path) -> list[EvidenceRef]`
- [ ] Copy input files to evidence/ subdirectory
- [ ] Calculate SHA-256 hash for each file
- [ ] Return EvidenceRef for each file

### M5.4 - Create run manifest generator
- [ ] Create `src/cbam_pack/audit/manifest.py`
- [ ] Implement `generate_manifest(config: CBAMConfig, inputs: list[Path], outputs: list[Path]) -> RunManifest`
- [ ] Include: run_id (UUID), timestamp (ISO 8601), pack_version, factor_library_version, config_hash, input_files with hashes, output_files with hashes, runtime info (python version, platform)

### M5.5 - Create gap report generator
- [ ] Create `src/cbam_pack/audit/gap_report.py`
- [ ] Implement `generate_gap_report(results: CalculationResult, assumptions: list[Assumption]) -> GapReport`
- [ ] Identify lines using default factors (improvement opportunity)
- [ ] Identify missing supplier data by supplier_id
- [ ] Calculate improvement_potential estimate
- [ ] Generate actionable recommendations

### M5.6 - Create Excel summary generator
- [ ] Create `src/cbam_pack/exporters/excel_generator.py`
- [ ] Implement `generate_excel_summary(results: CalculationResult, config: CBAMConfig, output_path: Path)`
- [ ] Create sheets: Summary (totals, declarant info), Line Details (per-line breakdown), Assumptions (list), Factors Used (factor table)
- [ ] Use openpyxl with formatting

### M5.7 - Create audit bundle orchestrator
- [ ] Create `src/cbam_pack/audit/__init__.py`
- [ ] Implement `create_audit_bundle(results: CalculationResult, config: CBAMConfig, inputs: list[Path], outputs: dict, output_dir: Path)`
- [ ] Coordinate all audit generators
- [ ] Write: claims.json, lineage.json, assumptions.json, gap_report.json, run_manifest.json
- [ ] Copy evidence files

### M5.8 - Create determinism tests
- [ ] Create `tests/test_determinism.py`
- [ ] Run same inputs twice, verify identical output hashes
- [ ] Test with different timestamps (should not affect output content except manifest timestamp)
- [ ] Verify no random ordering in JSON outputs

---

## Milestone 6: Demo + Pilot Packaging

### M6.1 - Implement main CLI command
- [ ] Update `src/cbam_pack/cli.py` with full `gl run cbam` implementation
- [ ] Arguments: --config/-c (required), --imports/-i (required), --out/-o (required)
- [ ] Options: --verbose/-v, --dry-run, --version, --help
- [ ] Exit codes: 0=success, 1=validation error, 2=calculation error, 3=export error

### M6.2 - Implement pipeline orchestrator
- [ ] Create `src/cbam_pack/pipeline.py`
- [ ] Implement `run_cbam_pipeline(config_path: Path, imports_path: Path, output_dir: Path, dry_run: bool = False) -> PipelineResult`
- [ ] Execute stages in order: validate -> normalize -> calculate -> export_xml -> bundle
- [ ] Log progress for each stage
- [ ] Handle errors gracefully with proper exit codes

### M6.3 - Implement logging system
- [ ] Create `src/cbam_pack/logging.py`
- [ ] Implement structured logging with levels: ERROR, WARNING, INFO, DEBUG
- [ ] Write to run.log in output directory
- [ ] Support --verbose flag for DEBUG level
- [ ] Format: timestamp level message

### M6.4 - Create golden dataset 1: basic steel import
- [ ] Create `data/golden_datasets/golden_basic/imports.csv` with 10 steel import lines from China, defaults only
- [ ] Create `data/golden_datasets/golden_basic/config.yaml`
- [ ] Create `data/golden_datasets/golden_basic/expected/` with expected outputs
- [ ] Document expected totals

### M6.5 - Create golden dataset 2: mixed steel and aluminum
- [ ] Create `data/golden_datasets/golden_mixed/imports.csv` with 25 lines: steel + aluminum, multiple countries, some with supplier data
- [ ] Create config and expected outputs
- [ ] Include supplier-specific overrides

### M6.6 - Create golden dataset 3: edge cases
- [ ] Create `data/golden_datasets/golden_edge/imports.csv` with 15 lines testing: unit conversion (kg input), aggregation, boundary values
- [ ] Create config and expected outputs

### M6.7 - Create integration tests with golden datasets
- [ ] Create `tests/test_integration.py`
- [ ] Test full pipeline with each golden dataset
- [ ] Verify outputs match expected
- [ ] Verify XSD validation passes
- [ ] Verify determinism

### M6.8 - Create Dockerfile
- [ ] Create `Dockerfile` with Python 3.11 base
- [ ] Install dependencies from requirements.txt
- [ ] Copy source code
- [ ] Set entrypoint to CLI
- [ ] Multi-stage build for smaller image

### M6.9 - Create docker-compose.yml
- [ ] Create `docker-compose.yml` for easy pilot deployment
- [ ] Volume mounts for data/ and output/
- [ ] Environment variables for configuration

### M6.10 - Create README.md with quick start
- [ ] Create `README.md` with: Overview, Installation (pip and Docker), Quick Start (5 minutes), CLI Reference, Examples
- [ ] Include screenshot/output examples

### M6.11 - Create QUICK_START.md
- [ ] Create `docs/QUICK_START.md` with step-by-step first report guide
- [ ] Include: Install, Download template, Fill data, Run command, Review outputs, Submit to registry

### M6.12 - Create demo script
- [ ] Create `scripts/demo.sh` that runs a complete demonstration
- [ ] Use golden_basic dataset
- [ ] Show progress and outputs
- [ ] Explain each artifact

### M6.13 - Final test coverage check
- [ ] Run `pytest --cov=src --cov-report=html`
- [ ] Verify ≥80% coverage on calculation logic and validation
- [ ] Document any coverage gaps

---

## Completion Criteria

All tasks must pass:
- [ ] XSD-valid XML generated for all golden datasets
- [ ] Complete audit bundle (claims.json, lineage.json, assumptions.json, run_manifest.json, evidence/) produced
- [ ] Deterministic reruns verified (identical hashes)
- [ ] Steel (72xx, 73xx) and Aluminum (76xx) CN codes supported
- [ ] Direct + indirect emissions calculated
- [ ] Gap report generated
- [ ] Error handling with actionable messages (fail-fast)
- [ ] Unit test coverage ≥80%
- [ ] All golden dataset tests pass
- [ ] Docker deployment works
